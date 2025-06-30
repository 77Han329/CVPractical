import torch
import torch.nn as nn
from typing import Union
from copy import deepcopy
from collections import OrderedDict
from pytorch_lightning import LightningModule

from validation_loss.image_ldm_main.jutils import instantiate_from_config
from validation_loss.image_ldm_main.jutils import load_partial_from_config
from validation_loss.image_ldm_main.jutils import exists, freeze, default

from validation_loss.image_ldm_main.ldm.logging import log_images
from validation_loss.image_ldm_main.ldm.metrics import ImageMetricTracker
from torchmetrics.aggregation import CatMetric


def un_normalize_ims(ims):
    """ Convert from [-1, 1] to [0, 255] """
    ims = ((ims * 127.5) + 127.5).clip(0, 255).to(torch.uint8)
    return ims


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    print("Updating EMA model with decay:", decay)
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        if not param.requires_grad:
            continue
        # unwrap DDP
        if name.startswith('module.'):
            name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


class TrainerModuleLatentFlow(LightningModule):
    def __init__(
        self,
        # flow model
        model: Union[dict, nn.Module],
        flow_cfg: dict,
        first_stage_cfg: dict = None,
        # learning
        lr: float = 1e-4,
        weight_decay: float = 0.,
        ema_rate: float = 0.9999,
        lr_scheduler_cfg: dict = None,
        # logging
        log_grad_norm: bool = False,
        n_images_to_vis: int = 16,
        sample_kwargs: dict = None
    ):
        super().__init__()

        # rectified flow
        self.flow = instantiate_from_config(flow_cfg)

        # unet/transformer model
        self.model = model if isinstance(model, nn.Module) else instantiate_from_config(model)

        # EMA model
        self.ema_model = None
        self.ema_rate = ema_rate
        if ema_rate > 0:
            self.ema_model = deepcopy(self.model)
            freeze(self.ema_model)
            self.ema_model.eval()
            update_ema(self.ema_model, self.model, decay=0)     # ensure EMA is in sync
        
        # first stage settings
        self.first_stage = None
        if exists(first_stage_cfg):
            first_stage = instantiate_from_config(first_stage_cfg)
            self.first_stage = torch.compile(first_stage, fullgraph=True)
            freeze(self.first_stage)
            self.first_stage.eval().to(self.device)

        # training parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler_cfg = lr_scheduler_cfg
        self.log_grad_norm = log_grad_norm

        # visualization
        self.sample_kwargs = sample_kwargs or {}
        self.n_images_to_vis = n_images_to_vis
        self.image_shape = None
        self.latent_shape = None
        self.generator = torch.Generator()

        # evaluation
        self.metric_tracker = ImageMetricTracker().to(self.device)

        # SD3 & Meta Movie Gen show that val loss correlates with human quality
        # and compute the loss in equidistant segments in (0, 1) to reduce variance
        self.val_losses = CatMetric().to(self.device)        # sync across GPUs
        self.val_images = None
        self.val_epochs = 0

        self.save_hyperparameters()

        # signal handler for slurm, flag to make sure the signal
        # is not handled at an incorrect state, e.g. during weights update
        self.stop_training = False

    # dummy function to be compatible
    def stop_training_method(self):
        pass

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.lr, weight_decay=self.weight_decay
        )
        out = dict(optimizer=opt)
        if exists(self.lr_scheduler_cfg):
            sch = load_partial_from_config(self.lr_scheduler_cfg)
            sch = sch(optimizer=opt)
            out["lr_scheduler"] = sch
        return out
    
    @torch.no_grad()
    def encode(self, x):
        return self.first_stage.encode(x) if exists(self.first_stage) else x
    
    @torch.no_grad()
    def decode(self, z):
        return self.first_stage.decode(z) if exists(self.first_stage) else z
    
    def forward(self, batch):
        ims = batch["image"]
        latent = batch.get("latent", self.encode(ims))
        label = batch.get("label", None)

        # compute loss
        loss = self.flow.training_losses(model=self.model, x1=latent, y=label)

        return loss
    
    def training_step(self, batch, batch_idx):
        # compute loss
        loss = self.forward(batch)

        if isinstance(loss, tuple):
            assert len(loss) == 2, "Loss tuple should be of length 2, shall be (loss, dict)."
            loss, loss_dict = loss
        else:
            loss_dict = {}

        # logging
        bs = batch["image"].shape[0]
        self.log("train/loss", loss, on_step=True, on_epoch=False, batch_size=bs, sync_dist=False)
        for k, v in loss_dict.items():
            self.log(f"train/{k}", v, on_step=True, on_epoch=False, batch_size=bs, sync_dist=False)
        if self.log_grad_norm:
            grad_norm = get_grad_norm(self.model)
            self.log("train/grad_norm", grad_norm, on_step=True, on_epoch=False, sync_dist=False)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # first checking for trainer ensures that the module can be also used with accelerate
        if exists(self._trainer) and exists(self.lr_scheduler_cfg): self.lr_schedulers().step()
        if exists(self.ema_model): update_ema(self.ema_model, self.model, decay=self.ema_rate)
        if self.stop_training: self.stop_training_method()

    def validation_step(self, batch, batch_idx):
        ims = batch["image"]
        label = batch.get("label", None)
        latent = default(batch.get("latent"), None)
        bs = ims.shape[0]
        
        g = self.generator.manual_seed(batch_idx + self.global_rank * 16102024)
        sample_model = self.ema_model if exists(self.ema_model) else self.model

        # flow models val loss shows correlation with human quality
        if hasattr(self.flow, 'validation_losses'):
            latent = default(latent, self.encode(ims))
            noise = torch.randn(latent.shape, generator=g, dtype=ims.dtype).to(ims.device)
            _, val_loss_per_segment = self.flow.validation_losses(model=sample_model, x1=latent, x0=noise, y=label)
            self.val_losses.update(val_loss_per_segment.unsqueeze(0))

        # Skip generation if model doesn't support it
        if not hasattr(self.flow, 'generate'):
            samples = None
        else:
            # generation
            if self.latent_shape is None:
                _latent = self.encode(ims)
                self.latent_shape = _latent.shape[1:]
                self.batch_size = bs
            
            # sample images
            z = torch.randn((bs, *self.latent_shape), generator=g, dtype=ims.dtype).to(ims.device)
            samples = self.flow.generate(model=sample_model, x=z, y=label, **self.sample_kwargs)
            samples = self.decode(samples)

        # metrics
        if samples is not None:  # Only compute metrics if we have samples
            self.metric_tracker(ims, samples)

        # save the images for visualization
        if self.val_images is None:
            real_ims = un_normalize_ims(ims)
            fake_ims = un_normalize_ims(samples) if samples is not None else None
            self.val_images = {
                "real": real_ims[:self.n_images_to_vis],
                "fake": fake_ims[:self.n_images_to_vis] if fake_ims is not None else None,
            }

    def on_validation_epoch_end(self, split=8):
        # visualization
        for key, ims in self.val_images.items():
            #log_images(self.logger, ims, f"val/{key}/samples", stack="row", split=split, step=self.global_step)
            pass
        
        # reset val images
        self.val_images = None

        # compute metrics
        metrics = self.metric_tracker.aggregate()
        for k, v in metrics.items():
            self.log(f"val/{k}", v, sync_dist=True)
        self.metric_tracker.reset()
        print(f"val_losses: {self.val_losses.value}")

        # compute val loss if available (Flow models)
        if len(self.val_losses.value) > 0:
            val_losses = self.val_losses.compute()          # (N batches, segments)
            val_losses = val_losses.mean(0)                 # mean per segment
            for i, loss in enumerate(val_losses):
                self.log(f"val/loss_segment_{i}", loss, sync_dist=True)
            self.log("val/loss", val_losses.mean(), sync_dist=True)
            self.val_losses.reset()

        # log some information
        self.val_epochs += 1
        print(f"Val epoch {self.val_epochs:,} | Optimizer step {self.global_step:,}")
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.print(metric_str)


def get_grad_norm(model):
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

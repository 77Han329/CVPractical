import wandb
import torch
import einops
from PIL import Image
from jutils import ims_to_grid    
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers import TensorBoardLogger


def log_images(logger, ims, tag, stack="row", split=4, step=0):
    """
    Args:
        logger: Logger class
        ims: torch.Tensor or np.ndarray of shape (b, c, h, w) in range [0, 255]
        tag: str, key to log the images
    """
    assert len(ims.shape) == 4, f"ims shape should be (b, c, h, w), got {ims.shape}"
    ims = ims_to_grid(ims, stack=stack, split=split)
    if isinstance(ims, torch.Tensor): ims = ims.cpu().numpy()
    
    if isinstance(logger, WandbLogger):
        ims = Image.fromarray(ims)
        ims = wandb.Image(ims)
        logger.experiment.log({tag: ims})
    
    elif isinstance(logger, (TensorBoardLogger, torch.utils.tensorboard.SummaryWriter)):
        if isinstance(ims, torch.Tensor): ims = ims.cpu().numpy()
        ims = einops.rearrange(ims, "h w c -> c h w")
        if hasattr(logger, "experiment"):
            logger = logger.experiment
        logger.add_image(tag, ims, global_step=step)
    
    else:
        raise ValueError(f"Unknown logger type: {type(logger)}")

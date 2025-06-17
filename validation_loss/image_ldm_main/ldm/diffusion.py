import torch
import einops
import torch.nn as nn
from functools import partial

from ldm.ddim import DDIMSampler
from ldm.ddpm import GaussianDiffusion


class DiffusionFlow(nn.Module):
    def __init__(
        self,
        timesteps: int = 1000,
        beta_schedule: str = 'linear',
        loss_type: str = 'l2',
        parameterization: str = 'v',
        linear_start: float = 1e-4,
        linear_end: float = 2e-2,
        cosine_s: float = 8e-3,
    ):
        super().__init__()
        self.diffusion = GaussianDiffusion(
            timesteps=timesteps,
            beta_schedule=beta_schedule,
            loss_type=loss_type,
            parameterization=parameterization,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )

        self.ddim_sampler = DDIMSampler(self.diffusion)

    def training_losses(self, model: nn.Module, x1: torch.Tensor, x0: torch.Tensor = None, **cond_kwargs):
        loss, _ = self.diffusion.training_losses(
            model=model,
            x_start=x1,
            noise=x0,
            **cond_kwargs
        )
        return loss
    
    # def generate(self, x: torch.Tensor, sample_kwargs=None, reverse=False, return_intermediates=False, **kwargs):
    def generate(
        self,
        model: nn.Module,
        x: torch.Tensor,
        num_steps: int = 50,
        reverse=False,
        return_intermediates=False,
        progress=True,
        sample_kwargs=None,
        **kwargs
    ):
        """
        Args:
            model: model to use for sampling
            x: source minibatch (bs, *dim)
            num_steps: int, number of sampling steps to take
            sample_kwargs: dict, additional sampling arguments for the solver
                progress: bool, whether to show a progress bar
                clip_denoised: bool, whether to clip the denoised images to [-1, 1]
                use_ddpm: bool, whether to use DDPM sampling instead of DDIM
                intermediate_key: str, key to use for intermediate outputs
                    (DDIM: 'x_inter', 'pred_x0' | DDPM: 'sample' or 'pred_xstart')
                intermediate_freq: int, frequency of intermediate outputs
                __ DDIM only __:
                    eta: float, noise level for DDIM
                    temperature: float, temperature for DDIM
                    noise_dropout: float, dropout rate for DDIM
                    cfg_scale: float, scale factor for Classifier-free guidance
                    uc_cond: torch.Tensor, unconditional conditioning information
                    cond_key: str, key to use for conditioning information
            reverse: bool, whether to reverse the direction of the flow. If True,
                we map from x1 -> x0, otherwise we map from x0 -> x1.
            return_intermediates: if true, return the intermediate samples
            kwargs: additional arguments for the network (e.g. conditioning information).
        """
        if reverse:
            raise NotImplementedError("[DiffusionFlow] Reverse sampling not yet supported")
        
        sample_kwargs = sample_kwargs or {}
        if 'num_steps' in sample_kwargs:
            raise ValueError("num_steps in sample_kwargs is deprecated, directly pass num_steps to generate()")
        
        # DDPM sampling
        if sample_kwargs.get("use_ddpm", False):
            # include CFG
            forward_fn = partial(
                forward_with_cfg,
                model       = model,
                cfg_scale   = sample_kwargs.get("cfg_scale", 1.),
                uc_cond     = sample_kwargs.get("uc_cond", None),
                cond_key    = sample_kwargs.get("cond_key", "y"),
            )
            out, intermediates = self.diffusion.p_sample_loop(
                # model                   = model,           # without CFG
                model                   = forward_fn,
                noise                   = x,
                model_kwargs            = kwargs,
                progress                = progress,
                clip_denoised           = sample_kwargs.get("clip_denoised", False),
                return_intermediates    = True,
                intermediate_freq       = sample_kwargs.get("intermediate_freq", (100 if return_intermediates else 1000)),
                pbar_desc               = sample_kwargs.get("pbar_desc", "DDPM Sampling"),
                intermediate_key        = sample_kwargs.get("intermediate_key", "sample"),
            )

        # DDIM sampling
        else:
            out, intermediates = self.ddim_sampler.sample(
                model                   = model,
                noise                   = x,
                model_kwargs            = kwargs,
                ddim_steps              = num_steps,
                eta                     = sample_kwargs.get("eta", 0.),
                progress                = progress,
                temperature             = sample_kwargs.get("temperature", 1.),
                noise_dropout           = sample_kwargs.get("noise_dropout", 0.),
                log_every_t             = sample_kwargs.get("intermediate_freq", (10 if return_intermediates else 1000)),
                clip_denoised           = sample_kwargs.get("clip_denoised", False),
                cfg_scale               = sample_kwargs.get("cfg_scale", 1.),
                uc_cond                 = sample_kwargs.get("uc_cond", None),
                cond_key                = sample_kwargs.get("cond_key", "y"),
            )
            key = sample_kwargs.get("intermediate_key", "x_inter")
            intermediates = intermediates[key]

        if return_intermediates:
            return torch.stack(intermediates, 0)
        return out


def forward_with_cfg(x, t, model, cfg_scale=1.0, uc_cond=None, cond_key="y", **model_kwargs):
    """ Function to include sampling with Classifier-Free Guidance (CFG) """
    if cfg_scale == 1.0:                                # without CFG
        model_output = model(x, t, **model_kwargs)

    else:                                               # with CFG
        assert cond_key in model_kwargs, f"Condition key '{cond_key}' for CFG not found in model_kwargs"
        assert uc_cond is not None, "Unconditional condition not provided for CFG"
        kwargs = model_kwargs.copy()
        c = kwargs[cond_key]
        x_in = torch.cat([x] * 2)
        t_in = torch.cat([t] * 2)
        if uc_cond.shape[0] == 1:
            uc_cond = einops.repeat(uc_cond, '1 ... -> bs ...', bs=x.shape[0])
        c_in = torch.cat([uc_cond, c])
        kwargs[cond_key] = c_in
        model_uc, model_c = model(x_in, t_in, **kwargs).chunk(2)
        model_output = model_uc + cfg_scale * (model_c - model_uc)

    return model_output

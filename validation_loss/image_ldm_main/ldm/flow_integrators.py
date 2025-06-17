import torch
import numpy as np
from tqdm import tqdm


def pad_v_like_x(v_, x_):
    """
    Function to reshape the vector by the number of dimensions
    of x. E.g. x (bs, c, h, w), v (bs) -> v (bs, 1, 1, 1).
    """
    if isinstance(v_, float):
        return v_
    return v_.reshape(-1, *([1] * (x_.ndim - 1)))


""" ODE """


def euler(model, x, timesteps: list[float], progress=True, **kwargs):
    bs, dev = x.shape[0], x.device

    xt = x
    for t_curr, t_next in tqdm(zip(timesteps[:-1], timesteps[1:]), disable=not progress, total=len(timesteps)-1):
        t = torch.ones((bs,), dtype=x.dtype, device=dev) * t_curr
        pred = model(xt, t, **kwargs)

        dt = t_next - t_curr
        xt = xt + dt * pred

    return xt


def rk4(model, x, timesteps: list[float], progress=True, **kwargs):
    bs, dev = x.shape[0], x.device

    xt = x
    for t_curr, t_next in tqdm(zip(timesteps[:-1], timesteps[1:]), disable=not progress, total=len(timesteps)-1):
        t = torch.ones((bs,), dtype=x.dtype, device=dev) * t_curr
        
        dt = t_next - t_curr
        
        k1 = model(xt, t, **kwargs)
        k2 = model(xt + dt * k1 * 1/3, t + 1/3 * dt, **kwargs)
        k3 = model(xt + dt * (k2 - k1 * 1/3), t + 2/3 * dt, **kwargs)
        k4 = model(xt + dt * (k1 - k2 + k3), t + dt, **kwargs)

        xt = xt + (k1 + 3 * (k2 + k3) + k4) * dt * 0.125

    return xt


def ode_int(model, x, timesteps: list[float], method="euler", atol=1e-6, rtol=1e-3, progress=True, **kwargs):
    # lazy import, s.t. we don't depend on it
    import torchdiffeq

    bs, dev = x.shape[0], x.device

    def fn(t, x):
        t = torch.ones((bs,), dtype=x.dtype, device=dev) * t
        return model(x, t, **kwargs)

    if not isinstance(timesteps, torch.Tensor):
        timesteps = torch.tensor(timesteps)
    timesteps = timesteps.to(dev)
    
    samples = torchdiffeq.odeint(
        fn,
        x,
        timesteps,
        method=method,
        atol=atol,
        rtol=rtol,
    )
    
    return samples[-1]


""" SDE """


def compute_diffusion(t, form: str, norm: float, schedule = None):
    if form == "constant":
        return torch.tensor(norm).to(t.device)

    elif form == "SBDM":
        assert schedule is not None, "Schedule must be provided for 'SBDM'"
        alpha_ratio = schedule.compute_d_alpha_alpha_ratio_t(t)
        sigma_t, d_sigma_t = schedule.compute_sigma_t(t)
        diffusion = alpha_ratio * (sigma_t ** 2) - sigma_t * d_sigma_t
        return norm * diffusion

    elif form == "sigma":
        assert schedule is not None, "Schedule must be provided for 'sigma'"
        sigma_t, _ = schedule.compute_sigma_t(t)
        return norm * sigma_t

    elif form == "linear":
        return norm * (1 - t)

    elif form == "decreasing":
        return 0.25 * (norm * torch.cos(np.pi * t) + 1) ** 2

    elif form == "increasing-decreasing":
        return norm * torch.sin(np.pi * t) ** 2
        
    else:
        raise ValueError(f"Unknown diffusion form: {form}")


def euler_maruyama(
    model,
    x,
    timesteps,
    schedule,
    diffusion_form: str = "sigma",  # ["constant", "SBDM", "sigma", "linear", "decreasing", "increasing-decreasing"]
    diffusion_norm: float = 1.0,
    last_step: str = "mean",        # ["mean", "euler", "tweedie", None]
    progress: bool = True,
    **kwargs
):
    bs, dev = x.shape[0], x.device

    xt = x
    for t_curr, t_next in tqdm(zip(timesteps[:-1], timesteps[1:]), disable=not progress, total=len(timesteps)-1):
        dt = t_next - t_curr
        is_last_step = t_next == timesteps[-1]
        t = torch.ones((bs,), dtype=x.dtype, device=dev) * t_curr

        w_cur = torch.randn(xt.size()).to(dev)
        dw = w_cur * torch.sqrt(dt)

        diffusion = compute_diffusion(t, diffusion_form, diffusion_norm, schedule)
        diffusion = pad_v_like_x(diffusion, xt)

        velocity = model(xt, t, **kwargs)
        score = schedule.get_score_from_velocity(velocity, xt, t)
        drift = velocity + diffusion * score
        
        mean_x = xt + drift * dt

        if not is_last_step or last_step is None:
            xt = mean_x + torch.sqrt(2 * diffusion) * dw
        
        else:
            if last_step == "mean":
                xt = mean_x
            elif last_step == "euler":
                xt = xt + velocity * dt
            elif last_step == "tweedie":
                alpha_t = pad_v_like_x(schedule.alpha_t(t), xt).to(dev)
                sigma_t = pad_v_like_x(schedule.sigma_t(t), xt).to(dev)
                xt = xt / alpha_t + (sigma_t ** 2) / alpha_t * score
            else: raise ValueError(f"Unknown last step method: {last_step}")

    return xt


def heun(
    model,
    x,
    timesteps,
    schedule,
    diffusion_form: str = "sigma",  # ["constant", "SBDM", "sigma", "linear", "decreasing", "increasing-decreasing"]
    diffusion_norm: float = 1.0,
    last_step: str = "mean",        # ["mean", "euler", "tweedie", None]
    progress: bool = True,
    **kwargs
):
    bs, dev = x.shape[0], x.device

    xt = x
    for t_curr, t_next in tqdm(zip(timesteps[:-1], timesteps[1:]), disable=not progress, total=len(timesteps)-1):
        dt = t_next - t_curr
        is_last_step = t_next == timesteps[-1]
        t = torch.ones((bs,), dtype=x.dtype, device=dev) * t_curr

        w_cur = torch.randn(xt.size()).to(dev)
        dw = w_cur * torch.sqrt(dt)

        diffusion = compute_diffusion(t, diffusion_form, diffusion_norm, schedule)
        diffusion = pad_v_like_x(diffusion, xt)

        if not is_last_step or last_step is None:    
            xhat = xt + torch.sqrt(2 * diffusion) * dw

            # compute K1
            velocity = model(xhat, t, **kwargs)
            score = schedule.get_score_from_velocity(velocity, xhat, t)
            k1 = velocity + diffusion * score

            xp = xhat + dt * k1

            # compute K2
            diffusion = compute_diffusion(t + dt, diffusion_form, diffusion_norm, schedule)
            diffusion = pad_v_like_x(diffusion, xp)

            velocity = model(xp, t + dt, **kwargs)
            score = schedule.get_score_from_velocity(velocity, xp, t + dt)
            k2 = velocity + diffusion * score

            xt = xhat + 0.5 * dt * (k1 + k2)
        
        else:
            velocity = model(xt, t, **kwargs)
            score = schedule.get_score_from_velocity(velocity, xt, t)

            if last_step == "mean":
                drift = velocity + diffusion * score
                xt = xt + drift * dt
                
            elif last_step == "euler":
                xt = xt + velocity * dt
                
            elif last_step == "tweedie":
                alpha_t = pad_v_like_x(schedule.alpha_t(t), xt).to(dev)
                sigma_t = pad_v_like_x(schedule.sigma_t(t), xt).to(dev)
                xt = xt / alpha_t + (sigma_t ** 2) / alpha_t * score

            else:
                raise ValueError(f"Unknown last step method: {last_step}")

    return xt

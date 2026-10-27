"""Samplers for RL: the Euler integrator of sample.generate taken one step at a time.

t runs 0 (noise) -> 1 (data); tau = 1 - t is the noise level; the model predicts v = dx/dt.
  ODE   x_{i+1} = x_i + v dt                                        DGPO rollouts, evaluation
  SDE   Flow-GRPO's noise-preserving SDE, rewritten for this orientation:
            dx = [ v - sigma^2 / (2 tau) * (x - t v) ] dt + sigma dw,     sigma = a * sqrt(tau / t)
        Euler-Maruyama makes every step a Gaussian
            x_{i+1} ~ N( x_i + drift_i dt_i,  (sigma_i sqrt(dt_i))^2 I )
        whose mean depends on the model: a stochastic policy with a tractable per-step density.
        a = 0.7 in Flow-GRPO; a = 0 is the ODE. At the first step (t = 0) the noise scale uses t_1
        instead of t (Flow-GRPO's sigma_max rule), otherwise sigma would be infinite.
Log-densities are averaged over the latent dimensions (Flow-GRPO's convention), which is why the
PPO clip range is 1e-4 and not 0.2.
"""
from __future__ import annotations
import math
import torch
from .. import schedule
from ..sample import pad_ctx

LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)


def timesteps(steps, shift, device="cuda"):
    return schedule.shift(steps, shift, device=device) if shift and shift != 1 else schedule.uniform(steps, device=device)


def cfg_velocity(model, x, t, ctx, msk, null_ctx, null_msk, cfg):
    """Classifier-free-guided velocity in fp32. Conditional and unconditional branches run as one
    batch and both carry gradient: under CFG the policy *is* this combination."""
    n = x.shape[0]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        v = model(torch.cat([x, x]), torch.cat([t, t]),
                  torch.cat([ctx, null_ctx.expand(n, -1, -1)]), torch.cat([msk, null_msk.expand(n, -1)]))
    vc, vu = v.float().chunk(2)
    return vu + cfg * (vc - vu)


def mse(v, target):
    """per-sample mean squared error over the latent -> (N,)"""
    return ((v - target) ** 2).flatten(1).mean(1)


def sde_coeffs(ts, i, a):
    """(sigma_i, dt_i) for step i of the schedule ts (both 0-d tensors)."""
    t, dt = ts[i], ts[i + 1] - ts[i]
    t_eff = t if float(t) > 0 else ts[1]
    return a * torch.sqrt((1 - t) / t_eff), dt


def step_mean(x, v, ts, i, a):
    """Mean of the next state; a = 0 is plain Euler."""
    t, dt = ts[i], ts[i + 1] - ts[i]
    if a == 0:
        return x + v * dt
    sigma, _ = sde_coeffs(ts, i, a)
    tau = 1 - t
    return x * (1 - sigma ** 2 * dt / (2 * tau)) + v * (1 + sigma ** 2 * t / (2 * tau)) * dt


def gauss_logp(x_next, mean, sigma, dt):
    """log N(x_next; mean, (sigma sqrt(dt))^2 I), averaged over dimensions -> (N,)"""
    s = sigma * dt.sqrt()
    lp = -((x_next - mean) ** 2) / (2 * s ** 2) - torch.log(s) - LOG_SQRT_2PI
    return lp.flatten(1).mean(1)


@torch.no_grad()
def rollout(model, ctx, msk, null_ctx, null_msk, lat_ch, H, W, steps, cfg, shift, a=0.0, seed=0,
            chunk=128, device="cuda"):
    """ctx: (N,L,768) text -> xs (N, steps+1, C, H, W) fp32 whitened latents (xs[:, -1] is the image),
    logp (N, steps) per-step densities under the sampling policy (zeros when a = 0), ts (steps+1,)."""
    n = ctx.shape[0]
    if n > chunk:
        parts = [rollout(model, ctx[i:i+chunk], msk[i:i+chunk], null_ctx, null_msk, lat_ch, H, W, steps, cfg,
                         shift, a, seed + i, chunk, device) for i in range(0, n, chunk)]
        return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts]), parts[0][2]
    null_ctx, null_msk = pad_ctx(null_ctx, null_msk, ctx.shape[1])
    ts = timesteps(steps, shift, device)
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(n, lat_ch, H, W, device=device, generator=g)
    xs, lps = [x], []
    for i in range(steps):
        v = cfg_velocity(model, x, ts[i].expand(n), ctx, msk, null_ctx, null_msk, cfg)
        mean = step_mean(x, v, ts, i, a)
        if a > 0:
            sigma, dt = sde_coeffs(ts, i, a)
            x = mean + sigma * dt.sqrt() * torch.randn(x.shape, device=device, generator=g)
            lps.append(gauss_logp(x, mean, sigma, dt))
        else:
            x = mean
            lps.append(torch.zeros(n, device=device))
        xs.append(x)
    return torch.stack(xs, 1), torch.stack(lps, 1), ts

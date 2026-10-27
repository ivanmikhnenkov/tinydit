"""Flow-GRPO (Liu et al. 2025): GRPO on the per-step Gaussian policy of the SDE sampler.

For each prompt a group of G trajectories is sampled with the SDE (rollout.rollout, a > 0). The
reward of the final image becomes a group-relative advantage A = (r - mean_group) / std that every
step of the trajectory shares. Each step (x_j -> x_{j+1}) is then a PPO update on the ratio of the
current per-step density to the one recorded at sampling time:
    L = max( -A * ratio, -A * clip(ratio, 1-eps, 1+eps) )  +  beta * KL
with KL the Gaussian KL between the current and the frozen reference step means at the same
variance, ||mean - mean_ref||^2 / (2 sigma^2) (Flow-GRPO's code, averaged over dimensions).
Every step of every trajectory is trained on exactly once per round.
"""
from __future__ import annotations
import torch
from .rollout import cfg_velocity, step_mean, sde_coeffs, gauss_logp


def step_loss(policy, ref, xs, logp_old, adv, j, ts, a, ctx, msk, null_ctx, null_msk, cfg,
              clip=1e-4, adv_clip=5.0, beta=0.004):
    """xs: (n, steps+1, C, H, W) trajectories | logp_old: (n, steps) | adv: (n,) | j: step index"""
    n = xs.shape[0]
    x, x_next = xs[:, j], xs[:, j + 1]
    t = ts[j].expand(n)
    v = cfg_velocity(policy, x, t, ctx, msk, null_ctx, null_msk, cfg)
    mean = step_mean(x, v, ts, j, a)
    sigma, dt = sde_coeffs(ts, j, a)
    logp = gauss_logp(x_next, mean, sigma, dt)
    ratio = torch.exp(logp - logp_old[:, j])
    A = adv.clamp(-adv_clip, adv_clip)
    pol = torch.maximum(-A * ratio, -A * ratio.clamp(1 - clip, 1 + clip)).mean()
    loss = pol
    st = {"loss_pol": pol.detach(), "ratio": ratio.detach().mean(),
          "clipfrac": ((ratio.detach() - 1).abs() > clip).float().mean()}
    if beta > 0:
        with torch.no_grad():
            mean_ref = step_mean(x, cfg_velocity(ref, x, t, ctx, msk, null_ctx, null_msk, cfg), ts, j, a)
        kl = ((mean - mean_ref) ** 2).flatten(1).mean(1) / (2 * sigma ** 2)
        loss = loss + beta * kl.mean()
        st["kl"] = kl.detach().mean()
    return loss, st

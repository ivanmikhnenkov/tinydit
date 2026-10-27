"""DGPO, Direct Group Preference Optimization (Luo et al. 2025, ICLR 2026).

No policy gradient. Samples come from the deterministic ODE sampler, so nothing has a density;
instead the group is split by the sign of the advantage and trained with a Diffusion-DPO objective
at group level. The implicit reward of a sample is
    -beta * [ L_dsm^theta(x) - L_dsm^ref(x) ],    L_dsm = || v(x_t, t) - (x0 - eps) ||^2
i.e. re-noise the clean sample to a sampler timestep and measure how well the model predicts the
velocity back to it, relative to the frozen reference. The loss is
    L = -log sigmoid( -beta * sum_i A_i [ L_dsm^theta(x_i) - L_dsm^ref(x_i) ] / G )
whose gradient is  sigmoid(margin) * sum_i A_i grad L_dsm(x_i):  pull the model toward the positive
samples and away from the negative ones, gated by how far the group preference is already
satisfied (the gate is 0.5 at the start and goes to 0 as the margin opens). Weights |A| give the two
sides of the split equal mass, which is what removes the partition function of the group preference
model. One timestep per group per round, chosen among the noisiest `trunc+1` sampler steps; the
noise eps is shared inside a group so samples differ only by content. A PPO-style trust region
stops the gradient of samples whose implicit ratio exp(L_old - L_theta) has already moved past
`clip` in the advantage's direction, and beta_kl * || v - v_ref ||^2 keeps the model near the
reference (evaluated with CFG, like the rollouts). All of this follows the authors' code.
"""
from __future__ import annotations
import torch
from .rollout import cfg_velocity, mse


@torch.no_grad()
def prepare(policy, ref, x0, G, ts, trunc, ctx, msk, null_ctx, null_msk, cfg, seed, chunk=128):
    """Pick (t, eps) per group, re-noise, and score the sampling-time policy and the reference.
    x0: (N, C, H, W) clean ODE samples, N = groups * G, grouped contiguously."""
    n, dev = x0.shape[0], x0.device
    g = torch.Generator(device=dev).manual_seed(seed)
    ng = n // G
    j = torch.randint(0, trunc + 1, (ng,), generator=g, device=dev).repeat_interleave(G)
    eps = torch.randn((ng,) + x0.shape[1:], device=dev, generator=g).repeat_interleave(G, 0)
    t = ts[j]
    tt = t.view(-1, 1, 1, 1)
    x_t, target = tt * x0 + (1 - tt) * eps, x0 - eps
    old, refl, refv = [], [], []
    for i in range(0, n, chunk):
        s = slice(i, i + chunk)
        v_old = cfg_velocity(policy, x_t[s], t[s], ctx[s], msk[s], null_ctx, null_msk, cfg)
        v_ref = cfg_velocity(ref, x_t[s], t[s], ctx[s], msk[s], null_ctx, null_msk, cfg)
        old.append(mse(v_old, target[s])); refl.append(mse(v_ref, target[s])); refv.append(v_ref)
    return {"x_t": x_t, "t": t, "target": target, "j": j,
            "old_dsm": torch.cat(old), "ref_dsm": torch.cat(refl), "ref_v": torch.cat(refv)}


def loss(policy, b, s, adv, G, ctx, msk, null_ctx, null_msk, cfg, beta_dpo=100.0, clip=1e-2, beta_kl=0.02):
    """b: prepare() output | s: slice covering whole groups | adv: (N,)"""
    x_t, t, target = b["x_t"][s], b["t"][s], b["target"][s]
    v = cfg_velocity(policy, x_t, t, ctx[s], msk[s], null_ctx, null_msk, cfg)
    dsm = mse(v, target)                                                   # (n,)
    A = adv[s]
    delta = dsm - b["ref_dsm"][s]
    margin = (A * beta_dpo * delta).detach().view(-1, G).sum(1) / G        # one number per group
    w = torch.sigmoid(margin).repeat_interleave(G)
    ratio = torch.exp(b["old_dsm"][s] - dsm.detach())
    clipped = torch.where(A > 0, ratio > 1 + clip, ratio < 1 - clip)
    dsm_c = torch.where(clipped, dsm.detach(), dsm)
    pref = (w * A * dsm_c).mean()
    kl = mse(v, b["ref_v"][s])
    total = pref + beta_kl * kl.mean()
    wg = torch.sigmoid(margin)                                              # per-group gate: its spread says whether the
    return total, {"loss_pref": pref.detach(), "w": w.mean(), "margin": margin.mean(), "delta": delta.detach().mean(),   # margin is signal or noise
                   "dsm": dsm.detach().mean(), "clipfrac": clipped.float().mean(), "kl": kl.detach().mean(),
                   "w_std": wg.std() if wg.numel() > 1 else wg.sum() * 0, "w_lo": (wg < 0.1).float().mean(), "w_hi": (wg > 0.9).float().mean()}

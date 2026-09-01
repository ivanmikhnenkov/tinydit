"""Rectified flow: the training objective.

Sampling lives in sample.generate() — one Euler implementation, not three.

    e ~ N(0,I) at t=0,  z = image latent at t=1
    x_t    = (1-t)*e + t*z
    target = z - e            (the velocity; constant in t along a path)
    loss   = mse(v_theta(x_t, t, text), target)

Sampling integrates dx/dt = v_theta from t=0 to t=1.
"""
from __future__ import annotations
import torch


def sample_t(n: int, device, mode: str = "logit_normal", m: float = 0.0, s: float = 1.0):
    """SD3 found uniform t measurably worse than logit-normal, which concentrates
    training on the mid-noise region where the task is hardest."""
    if mode == "uniform":
        return torch.rand(n, device=device)
    return torch.sigmoid(torch.randn(n, device=device) * s + m)


def loss(model, z, ctx, ctx_mask=None, t_mode="logit_normal"):
    """z: (B,C,H,W) whitened latent | ctx: (B,L,D) text (already CFG-dropped)"""
    b = z.shape[0]
    t = sample_t(b, z.device, t_mode)
    e = torch.randn_like(z)
    tt = t.view(b, 1, 1, 1)
    x_t = (1 - tt) * e + tt * z
    target = z - e
    v = model(x_t, t, ctx, ctx_mask)
    per = ((v - target) ** 2).flatten(1).mean(1)      # (B,) per-sample, for t-bucketing
    return per.mean(), t, per

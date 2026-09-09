"""Rectified flow: the training objective.

    e ~ N(0,I) at t=0,  z = image latent at t=1
    x_t    = (1-t)*e + t*z
    target = z - e            (the velocity; constant in t along a path)

Loss = MSE(v, target)                           the flow-matching objective
     + w_cos  * (1 - cos(v, target))            direction term (LightningDiT): MSE is dominated by
                                                 magnitude at high noise; this keeps the direction honest
     + w_disp * dispersive(features)            Diffuse-and-Disperse (Wang & He 2025): push different
                                                 samples' intermediate features apart; no external encoder

Timestep sampling is logit-normal with a mean shift. In this repo t=1 is data, so the SD3 shift
that spends more time at high noise is a NEGATIVE mean:  m = -ln(alpha).  For a 32x32x32 latent
(m=32768 values) the SD3/RAE rule alpha = sqrt(m/4096) gives alpha ~ 2.8, m ~ -1.03.
Sampling lives in sample.generate() and uses schedule.shift(alpha) so both sides agree.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F


def shift_mean(alpha: float) -> float:
    return -math.log(alpha)


def sample_t(n: int, device, mode: str = "logit_normal", m: float = 0.0, s: float = 1.0):
    if mode == "uniform":
        return torch.rand(n, device=device)
    return torch.sigmoid(torch.randn(n, device=device) * s + m)


def dispersive(feat: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    """InfoNCE-style repulsion between samples in a batch. feat: (B, N, D) -> scalar.
    Each sample is flattened and L2-normalised, so distances live in [0, 4] regardless of the
    residual-stream scale; loss = log mean_{i!=j} exp(-||h_i - h_j||^2 / tau)."""
    h = F.normalize(feat.float().flatten(1), dim=1)
    B = h.shape[0]
    d = (2 - 2 * h @ h.t()).clamp_min(0)                       # squared L2 between unit vectors
    off = ~torch.eye(B, dtype=torch.bool, device=h.device)
    return torch.logsumexp(-d[off] / tau, dim=0) - math.log(B * (B - 1))


def loss(model, z, ctx, ctx_mask=None, t_mode="logit_normal", t_mean=0.0,
         w_cos=1.0, w_disp=0.5, disp_tau=0.5):
    """z: (B,C,H,W) whitened latent | ctx: (B,L,D) text (already CFG-dropped)
    -> total, t, per-sample mse (B,), dict of component means"""
    b = z.shape[0]
    t = sample_t(b, z.device, t_mode, m=t_mean)
    e = torch.randn_like(z)
    tt = t.view(b, 1, 1, 1)
    x_t = (1 - tt) * e + tt * z
    target = z - e
    if w_disp > 0:
        v, feat = model(x_t, t, ctx, ctx_mask, return_feat=True)
    else:
        v, feat = model(x_t, t, ctx, ctx_mask), None
    per = ((v.float() - target) ** 2).flatten(1).mean(1)          # (B,) per-sample, for t-bucketing
    mse = per.mean()
    total = mse
    parts = {"mse": mse.detach()}
    if w_cos > 0:
        cosl = (1 - F.cosine_similarity(v.float().flatten(1), target.flatten(1), dim=1)).mean()
        total = total + w_cos * cosl; parts["cos"] = cosl.detach()
    if w_disp > 0:
        dl = dispersive(feat, disp_tau)
        total = total + w_disp * dl; parts["disp"] = dl.detach()
    return total, t, per, parts

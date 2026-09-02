"""Step schedules for sampling. t runs 0 (noise) -> 1 (data), the convention used throughout."""
from __future__ import annotations
import torch


def uniform(n: int, device="cuda"):
    return torch.linspace(0, 1, n + 1, device=device)


def shift(n: int, alpha: float = 2.8, device="cuda"):
    """The SD3/FLUX timestep shift, written for this t-orientation: tau' = a*tau / (1 + (a-1)*tau)
    on the noise-side variable tau = 1-t. Concentrates steps near t=0, where global structure is
    decided; the same alpha shifts the training distribution (flow.shift_mean)."""
    tau = 1 - torch.linspace(0, 1, n + 1, device=device)
    tau = alpha * tau / (1 + (alpha - 1) * tau)
    return 1 - tau

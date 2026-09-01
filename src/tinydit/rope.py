"""2D axial rotary position embeddings.

RoPE derives position from (row, col) arithmetically, so a trained model runs at
any H×W grid with no new parameters. That is what keeps aspect-ratio bucketing and
resolution changes cheap later; learned absolute embeddings would lock the model
to one token count forever.

head_dim is split in half (the standard rotate-half convention). That half is then
split again between the row and column axes, so head_dim must be divisible by 4.
"""
from __future__ import annotations
import torch


def freqs_2d(h: int, w: int, head_dim: int, theta: float = 10000.0,
             device=None, dtype=torch.float32):
    """-> cos, sin each (h*w, head_dim//2)"""
    assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE"
    n_ax = head_dim // 4                                   # freq pairs per axis
    inv = 1.0 / (theta ** (torch.arange(n_ax, device=device, dtype=torch.float32) / n_ax))
    y = torch.arange(h, device=device, dtype=torch.float32)
    x = torch.arange(w, device=device, dtype=torch.float32)
    fy = torch.outer(y, inv)[:, None, :].expand(h, w, n_ax)     # (h,w,n_ax)
    fx = torch.outer(x, inv)[None, :, :].expand(h, w, n_ax)
    f = torch.cat([fy, fx], dim=-1).reshape(h * w, 2 * n_ax)    # (N, head_dim//2)
    return f.cos().to(dtype), f.sin().to(dtype)


def apply(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """t: (B, heads, N, head_dim); cos/sin: (N, head_dim//2)"""
    a, b = t.chunk(2, dim=-1)
    cos = cos.to(t.dtype)[None, None]
    sin = sin.to(t.dtype)[None, None]
    return torch.cat([a * cos - b * sin, a * sin + b * cos], dim=-1)

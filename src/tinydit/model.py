"""The DiT. Everything here is trained; the AE and text encoder are frozen.

Per block:
    x = x + gate1 * SelfAttn(  RMSNorm(x)*(1+scale1) + shift1 )   RoPE + QK-norm
    x = x +         CrossAttn( RMSNorm(x), K/V = text )           zero-init out proj
    x = x + gate2 * SwiGLU(    RMSNorm(x)*(1+scale2) + shift2 )

The (shift, scale, gate) triples come from an MLP on the timestep embedding and are
broadcast across every image token: a timestep applies to the whole image equally.
Text is the opposite — it needs per-token attention, hence cross-attention.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import rope


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbed(nn.Module):
    def __init__(self, dim: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:      # t: (B,) in [0,1]
        half = self.freq_dim // 2
        f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t.float().reshape(-1, 1) * 1000.0 * f[None]
        return self.mlp(torch.cat([a.cos(), a.sin()], dim=-1).to(self.mlp[0].weight.dtype))


class SelfAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.h, self.dh = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.qn = nn.RMSNorm(self.dh)       # QK-norm: the stability win from SD3/FLUX
        self.kn = nn.RMSNorm(self.dh)

    def forward(self, x, cos, sin):
        B, N, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(B, N, self.h, self.dh).transpose(1, 2) for t in (q, k, v))
        q, k = self.qn(q), self.kn(k)
        q, k = rope.apply(q, cos, sin), rope.apply(k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v)
        return self.proj(o.transpose(1, 2).reshape(B, N, -1))


class CrossAttention(nn.Module):
    def __init__(self, dim, heads, ctx_dim):
        super().__init__()
        self.h, self.dh = heads, dim // heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(ctx_dim, 2 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.qn = nn.RMSNorm(self.dh)
        self.kn = nn.RMSNorm(self.dh)
        nn.init.zeros_(self.proj.weight)     # block starts as a no-op on the residual

    def forward(self, x, ctx, ctx_mask):
        B, N, _ = x.shape
        L = ctx.shape[1]
        q = self.q(x).view(B, N, self.h, self.dh).transpose(1, 2)
        k, v = self.kv(ctx).chunk(2, dim=-1)
        k, v = (t.view(B, L, self.h, self.dh).transpose(1, 2) for t in (k, v))
        q, k = self.qn(q), self.kn(k)
        m = ctx_mask[:, None, None, :] if ctx_mask is not None else None
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        return self.proj(o.transpose(1, 2).reshape(B, N, -1))


class SwiGLU(nn.Module):
    """FLUX.2 ships a Flux2SwiGLU for exactly this slot - not an LLM-only idea."""
    def __init__(self, dim, mult=4.0):
        super().__init__()
        hidden = int(dim * mult * 2 / 3 / 64) * 64      # keep params ~= 4x MLP
        self.w12 = nn.Linear(dim, 2 * hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a) * b)


class Block(nn.Module):
    def __init__(self, dim, heads, ctx_dim, mlp_mult=4.0):
        super().__init__()
        self.n1, self.n2, self.n3 = (nn.RMSNorm(dim) for _ in range(3))
        self.attn = SelfAttention(dim, heads)
        self.cross = CrossAttention(dim, heads, ctx_dim)
        self.ff = SwiGLU(dim, mlp_mult)
        self.ada = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.ada.weight); nn.init.zeros_(self.ada.bias)   # adaLN-Zero

    def forward(self, x, t, ctx, ctx_mask, cos, sin):
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(F.silu(t)).chunk(6, dim=-1)
        x = x + g1.unsqueeze(1) * self.attn(modulate(self.n1(x), sh1, sc1), cos, sin)
        x = x + self.cross(self.n2(x), ctx, ctx_mask)
        x = x + g2.unsqueeze(1) * self.ff(modulate(self.n3(x), sh2, sc2))
        return x


class TinyDiT(nn.Module):
    def __init__(self, latent_ch=32, dim=768, depth=12, heads=12, ctx_dim=768,
                 patch=2, mlp_mult=4.0, rope_theta=10000.0):
        super().__init__()
        self.latent_ch, self.patch, self.dim, self.heads = latent_ch, patch, dim, heads
        self.head_dim = dim // heads
        self.rope_theta = rope_theta
        self.t_embed = TimestepEmbed(dim)
        self.patchify = nn.Conv2d(latent_ch, dim, patch, patch)
        self.blocks = nn.ModuleList([Block(dim, heads, ctx_dim, mlp_mult) for _ in range(depth)])
        self.norm_out = nn.RMSNorm(dim)
        self.ada_out = nn.Linear(dim, 2 * dim)
        self.out = nn.Linear(dim, patch * patch * latent_ch)
        nn.init.zeros_(self.ada_out.weight); nn.init.zeros_(self.ada_out.bias)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)   # v_theta == 0 at init
        self._cache: dict = {}

    def _rope(self, h, w, device, dtype):
        key = (h, w, device, dtype)
        if key not in self._cache:
            self._cache[key] = rope.freqs_2d(h, w, self.head_dim, self.rope_theta, device, dtype)
        return self._cache[key]

    def forward(self, z, t, ctx, ctx_mask=None):
        """z: (B,C,H,W) noisy latent | t: (B,) in [0,1] | ctx: (B,L,ctx_dim)"""
        B, _, H, W = z.shape
        x = self.patchify(z)                                  # (B, dim, H/p, W/p)
        h, w = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)                      # (B, N, dim)
        cos, sin = self._rope(h, w, z.device, torch.float32)
        temb = self.t_embed(t)
        for blk in self.blocks:
            x = blk(x, temb, ctx, ctx_mask, cos, sin)
        sh, sc = self.ada_out(F.silu(temb)).chunk(2, dim=-1)
        x = self.out(modulate(self.norm_out(x), sh, sc))      # (B, N, p*p*C)
        x = x.transpose(1, 2).reshape(B, self.latent_ch, self.patch, self.patch, h, w)
        return x.permute(0, 1, 4, 2, 5, 3).reshape(B, self.latent_ch, h * self.patch, w * self.patch)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


CONFIGS = {
    "tiny":  dict(dim=384,  depth=8,  heads=6),
    "small": dict(dim=640,  depth=12, heads=10),
    "base":  dict(dim=768,  depth=12, heads=12),
    # largest that still fits batch 256 at 256^2 with eval headroom (78 GB peak of 95)
    "mid":   dict(dim=896,  depth=12, heads=14),
    "large": dict(dim=1024, depth=16, heads=16),
}

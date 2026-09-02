"""The DiT. Everything here is trained; the AE and text encoder are frozen.

Per block (adaLN-single: one shared timestep MLP, plus a learned per-block offset table):
    x = x + gate1 * SelfAttn(  RMSNorm(x)*(1+scale1) + shift1 )   2D RoPE + QK-norm
    x = x +         CrossAttn( RMSNorm(x), K/V = text ++ learned null slots )
    x = x + gate2 * SwiGLU(    RMSNorm(x)*(1+scale2) + shift2 )

Two kinds of "register" tokens, both cheap:
  * image-stream registers: R learnable tokens appended to the image sequence from block
    `register_block` on and dropped before unpatchify. They carry no position (identity RoPE)
    and act as scratch space for global state that would otherwise be dumped on a random
    image token or on the T5 EOS state.
  * null K/V slots: a few learned key/value vectors appended to every cross-attention block,
    always unmasked, so a query that wants "no particular word" has somewhere to look.

The token grid is whatever the latent is: 2D RoPE derives position from (row, col), so the
five aspect-ratio buckets (and later resolutions) need no architectural change.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import rope


class RMSNorm(nn.Module):
    """nn.RMSNorm keeps fp32 weights; under bf16 autocast that mismatch falls back to an
    unfused kernel (PyTorch warns). Casting the weight to the activation dtype fixes it."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.weight.to(x.dtype), self.eps)


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
        self.qn = RMSNorm(self.dh)          # QK-norm: the stability win from SD3/FLUX
        self.kn = RMSNorm(self.dh)

    def forward(self, x, cos, sin):
        B, N, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(B, N, self.h, self.dh).transpose(1, 2) for t in (q, k, v))
        q, k = self.qn(q), self.kn(k)
        q, k = rope.apply(q, cos, sin), rope.apply(k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v)
        return self.proj(o.transpose(1, 2).reshape(B, N, -1))


class CrossAttention(nn.Module):
    def __init__(self, dim, heads, ctx_dim, n_null=2):
        super().__init__()
        self.h, self.dh, self.n_null = heads, dim // heads, n_null
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(ctx_dim, 2 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.qn = RMSNorm(self.dh)
        self.kn = RMSNorm(self.dh)
        # learned null key/value slots, in projected space, always attendable
        self.null_kv = nn.Parameter(torch.randn(2, n_null, dim) * 0.02) if n_null else None
        nn.init.zeros_(self.proj.weight)     # block starts as a no-op on the residual

    def forward(self, x, ctx, ctx_mask):
        B, N, _ = x.shape
        k, v = self.kv(ctx).chunk(2, dim=-1)
        if self.null_kv is not None:
            nk, nv = self.null_kv.to(k.dtype).unsqueeze(1).expand(-1, B, -1, -1)
            k, v = torch.cat([k, nk], 1), torch.cat([v, nv], 1)
            if ctx_mask is not None:
                ctx_mask = torch.cat([ctx_mask, ctx_mask.new_ones(B, self.n_null)], 1)
        L = k.shape[1]
        q = self.q(x).view(B, N, self.h, self.dh).transpose(1, 2)
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
    def __init__(self, dim, heads, ctx_dim, mlp_mult=4.0, n_null=2):
        super().__init__()
        self.n1, self.n2, self.n3 = (RMSNorm(dim) for _ in range(3))
        self.attn = SelfAttention(dim, heads)
        self.cross = CrossAttention(dim, heads, ctx_dim, n_null)
        self.ff = SwiGLU(dim, mlp_mult)
        # adaLN-single (PixArt-α): the 6*dim modulation comes from one shared MLP on the
        # timestep plus this small per-block table, instead of a 6*dim x dim linear per block.
        self.mod = nn.Parameter(torch.randn(6 * dim) / math.sqrt(dim))

    def forward(self, x, mod, ctx, ctx_mask, cos, sin):
        sh1, sc1, g1, sh2, sc2, g2 = (mod + self.mod).chunk(6, dim=-1)
        x = x + g1.unsqueeze(1) * self.attn(modulate(self.n1(x), sh1, sc1), cos, sin)
        x = x + self.cross(self.n2(x), ctx, ctx_mask)
        x = x + g2.unsqueeze(1) * self.ff(modulate(self.n3(x), sh2, sc2))
        return x


class TinyDiT(nn.Module):
    def __init__(self, latent_ch=32, dim=768, depth=12, heads=12, ctx_dim=768,
                 patch=2, mlp_mult=4.0, rope_theta=10000.0,
                 n_registers=16, register_block=3, n_null=2, feat_block=5):
        super().__init__()
        self.latent_ch, self.patch, self.dim, self.heads = latent_ch, patch, dim, heads
        self.head_dim = dim // heads
        self.rope_theta = rope_theta
        self.n_registers, self.register_block, self.feat_block = n_registers, register_block, feat_block
        self.t_embed = TimestepEmbed(dim)
        self.ada_shared = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada_shared[1].weight); nn.init.zeros_(self.ada_shared[1].bias)
        self.patchify = nn.Conv2d(latent_ch, dim, patch, patch)
        self.registers = nn.Parameter(torch.randn(n_registers, dim) * 0.02) if n_registers else None
        self.blocks = nn.ModuleList([Block(dim, heads, ctx_dim, mlp_mult, n_null) for _ in range(depth)])
        self.norm_out = RMSNorm(dim)
        self.ada_out = nn.Linear(dim, 2 * dim)
        self.out = nn.Linear(dim, patch * patch * latent_ch)
        nn.init.zeros_(self.ada_out.weight); nn.init.zeros_(self.ada_out.bias)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)   # v_theta == 0 at init

    def _rope(self, h, w, device):
        """cos/sin for the h*w image grid, plus identity rows (cos=1, sin=0) for registers.
        Computed functionally every call (a handful of tiny ops): a Python dict cache here made
        torch.compile re-guard on the dict and recompile every shape again and again."""
        cos, sin = rope.freqs_2d(h, w, self.head_dim, self.rope_theta, device, torch.float32)
        if self.n_registers:
            cos_r = torch.cat([cos, torch.ones(self.n_registers, cos.shape[1], device=device)])
            sin_r = torch.cat([sin, torch.zeros(self.n_registers, sin.shape[1], device=device)])
        else:
            cos_r, sin_r = cos, sin
        return cos, sin, cos_r, sin_r

    def forward(self, z, t, ctx, ctx_mask=None, return_feat=False):
        """z: (B,C,H,W) noisy latent | t: (B,) in [0,1] | ctx: (B,L,ctx_dim) | ctx_mask: (B,L) bool
        -> velocity (B,C,H,W)  [, features of image tokens after block `feat_block`: (B,N,dim)]"""
        B, _, H, W = z.shape
        x = self.patchify(z)                                  # (B, dim, H/p, W/p)
        h, w = x.shape[-2:]
        N = h * w
        x = x.flatten(2).transpose(1, 2)                      # (B, N, dim)
        cos, sin, cos_r, sin_r = self._rope(h, w, z.device)
        mod = self.ada_shared(self.t_embed(t))                # (B, 6*dim), shared by all blocks
        feat = None
        for i, blk in enumerate(self.blocks):
            if self.registers is not None and i == self.register_block:
                x = torch.cat([x, self.registers.to(x.dtype).unsqueeze(0).expand(B, -1, -1)], 1)
            with_reg = self.registers is not None and i >= self.register_block
            x = blk(x, mod, ctx, ctx_mask, cos_r if with_reg else cos, sin_r if with_reg else sin)
            if return_feat and i == self.feat_block:
                feat = x[:, :N]
        x = x[:, :N]                                          # drop registers
        sh, sc = self.ada_out(F.silu(self.t_embed(t))).chunk(2, dim=-1)
        x = self.out(modulate(self.norm_out(x), sh, sc))      # (B, N, p*p*C)
        x = x.transpose(1, 2).reshape(B, self.latent_ch, self.patch, self.patch, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).reshape(B, self.latent_ch, h * self.patch, w * self.patch)
        return (x, feat) if return_feat else x

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


CONFIGS = {
    "tiny":   dict(dim=384,  depth=8,  heads=6),
    "small":  dict(dim=640,  depth=12, heads=10),
    "base":   dict(dim=768,  depth=12, heads=12),
    "base16": dict(dim=768,  depth=16, heads=12),
    "run1":   dict(dim=896,  depth=16, heads=14),     # the pretraining model, ~210M params
    "large":  dict(dim=1024, depth=16, heads=16),
}

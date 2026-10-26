"""tinydit as a diffusers model: the same network as src/tinydit/model.py, same parameter names (so the
released safetensors load unchanged), self-contained so it can ship as remote code in the Hub repo.

The one real difference is the time convention at the boundary. tinydit was trained with t=1 as data and
predicts the velocity data - noise; diffusers flow-matching schedulers count sigma from the noise side
(timestep = sigma * 1000, 1000 = pure noise) and expect noise - data, as SD3 and FLUX do. forward() takes
the diffusers convention and converts, so the standard scheduler, pipelines and training scripts apply as is.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin


def rope_2d(h: int, w: int, head_dim: int, theta: float, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Axial 2D RoPE for an h*w grid -> cos, sin each (h*w, head_dim//2). Half the frequency pairs
    encode the row, half the column, so any grid shape works with no new parameters."""
    n_ax = head_dim // 4
    inv = 1.0 / (theta ** (torch.arange(n_ax, device=device, dtype=torch.float32) / n_ax))
    fy = torch.outer(torch.arange(h, device=device, dtype=torch.float32), inv)[:, None, :].expand(h, w, n_ax)
    fx = torch.outer(torch.arange(w, device=device, dtype=torch.float32), inv)[None, :, :].expand(h, w, n_ax)
    f = torch.cat([fy, fx], dim=-1).reshape(h * w, 2 * n_ax)
    return f.cos(), f.sin()


def apply_rope(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """t: (B, heads, N, head_dim); cos/sin: (N, head_dim//2), rotate-half convention."""
    a, b = t.chunk(2, dim=-1)
    cos, sin = cos.to(t.dtype)[None, None], sin.to(t.dtype)[None, None]
    return torch.cat([a * cos - b * sin, a * sin + b * cos], dim=-1)


class RMSNorm(nn.Module):
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

    def forward(self, t: torch.Tensor) -> torch.Tensor:      # t: (B,) in [0,1], 1 = data
        half = self.freq_dim // 2
        f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t.float().reshape(-1, 1) * 1000.0 * f[None]
        return self.mlp(torch.cat([a.cos(), a.sin()], dim=-1).to(self.mlp[0].weight.dtype))


class SelfAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h, self.dh = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.qn = RMSNorm(self.dh)
        self.kn = RMSNorm(self.dh)

    def forward(self, x, cos, sin):
        B, N, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(B, N, self.h, self.dh).transpose(1, 2) for t in (q, k, v))
        q, k = self.qn(q), self.kn(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v)
        return self.proj(o.transpose(1, 2).reshape(B, N, -1))


class CrossAttention(nn.Module):
    """K/V = text tokens ++ learned null slots that are never masked, so a query that wants no
    particular word has somewhere to look (and an empty prompt never leaves a row fully masked)."""
    def __init__(self, dim, heads, ctx_dim, n_null=2):
        super().__init__()
        self.h, self.dh, self.n_null = heads, dim // heads, n_null
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(ctx_dim, 2 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.qn = RMSNorm(self.dh)
        self.kn = RMSNorm(self.dh)
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
    def __init__(self, dim, mult=4.0):
        super().__init__()
        hidden = int(dim * mult * 2 / 3 / 64) * 64
        self.w12 = nn.Linear(dim, 2 * hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a) * b)


class TinyDiTBlock(nn.Module):
    def __init__(self, dim, heads, ctx_dim, mlp_mult=4.0, n_null=2):
        super().__init__()
        self.n1, self.n2, self.n3 = (RMSNorm(dim) for _ in range(3))
        self.attn = SelfAttention(dim, heads)
        self.cross = CrossAttention(dim, heads, ctx_dim, n_null)
        self.ff = SwiGLU(dim, mlp_mult)
        self.mod = nn.Parameter(torch.randn(6 * dim) / math.sqrt(dim))   # adaLN-single: per-block offset to the shared modulation

    def forward(self, x, mod, ctx, ctx_mask, cos, sin):
        sh1, sc1, g1, sh2, sc2, g2 = (mod + self.mod).chunk(6, dim=-1)
        x = x + g1.unsqueeze(1) * self.attn(modulate(self.n1(x), sh1, sc1), cos, sin)
        x = x + self.cross(self.n2(x), ctx, ctx_mask)
        x = x + g2.unsqueeze(1) * self.ff(modulate(self.n3(x), sh2, sc2))
        return x


class TinyDiTTransformer2DModel(ModelMixin, ConfigMixin):
    """adaLN-single DiT with 2D RoPE, QK-norm, SwiGLU, register tokens and null cross-attention slots.

    latents_mean / latents_std are the per-channel statistics the model's latent space was whitened with:
    the pipeline multiplies by std and adds mean before handing latents to the FLUX.2 VAE decoder."""

    _supports_gradient_checkpointing = True
    _no_split_modules = ["TinyDiTBlock"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 32,
        patch_size: int = 2,
        hidden_size: int = 896,
        num_layers: int = 16,
        num_attention_heads: int = 14,
        text_dim: int = 768,
        mlp_ratio: float = 4.0,
        rope_theta: float = 10000.0,
        num_registers: int = 16,
        register_block: int = 3,
        num_null_tokens: int = 2,
        latents_mean: list[float] | None = None,
        latents_std: list[float] | None = None,
    ):
        super().__init__()
        dim = hidden_size
        self.t_embed = TimestepEmbed(dim)
        self.ada_shared = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada_shared[1].weight); nn.init.zeros_(self.ada_shared[1].bias)
        self.patchify = nn.Conv2d(in_channels, dim, patch_size, patch_size)
        self.registers = nn.Parameter(torch.randn(num_registers, dim) * 0.02) if num_registers else None
        self.blocks = nn.ModuleList([TinyDiTBlock(dim, num_attention_heads, text_dim, mlp_ratio, num_null_tokens)
                                     for _ in range(num_layers)])
        self.norm_out = RMSNorm(dim)
        self.ada_out = nn.Linear(dim, 2 * dim)
        self.out = nn.Linear(dim, patch_size * patch_size * in_channels)
        nn.init.zeros_(self.ada_out.weight); nn.init.zeros_(self.ada_out.bias)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)   # v == 0 at init, as in training
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
        return_dict: bool = True,
    ):
        """hidden_states: (B,C,H,W) noisy latent | timestep: (B,) or scalar, sigma * 1000 |
        encoder_hidden_states: (B,L,text_dim) T5 states | encoder_attention_mask: (B,L), 1 = keep
        -> velocity noise - data, (B,C,H,W)"""
        cfg = self.config
        B, _, H, W = hidden_states.shape
        p, R = cfg.patch_size, cfg.num_registers
        t = 1 - torch.as_tensor(timestep, device=hidden_states.device).float().expand(B) / 1000
        mask = encoder_attention_mask.bool() if encoder_attention_mask is not None else None

        x = self.patchify(hidden_states)
        h, w = x.shape[-2:]
        N = h * w
        x = x.flatten(2).transpose(1, 2)
        cos, sin = rope_2d(h, w, cfg.hidden_size // cfg.num_attention_heads, cfg.rope_theta, x.device)
        if R:   # registers carry no position: identity rotation rows
            cos_r = torch.cat([cos, cos.new_ones(R, cos.shape[1])])
            sin_r = torch.cat([sin, sin.new_zeros(R, sin.shape[1])])
        mod = self.ada_shared(self.t_embed(t))
        for i, blk in enumerate(self.blocks):
            if R and i == cfg.register_block:
                x = torch.cat([x, self.registers.to(x.dtype).unsqueeze(0).expand(B, -1, -1)], 1)
            c, s = (cos_r, sin_r) if R and i >= cfg.register_block else (cos, sin)
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = self._gradient_checkpointing_func(blk, x, mod, encoder_hidden_states, mask, c, s)
            else:
                x = blk(x, mod, encoder_hidden_states, mask, c, s)
        x = x[:, :N]
        sh, sc = self.ada_out(F.silu(self.t_embed(t))).chunk(2, dim=-1)
        x = self.out(modulate(self.norm_out(x), sh, sc))
        x = x.transpose(1, 2).reshape(B, cfg.in_channels, p, p, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).reshape(B, cfg.in_channels, h * p, w * p)
        out = -x    # trained as data - noise; diffusers expects noise - data

        if not return_dict:
            return (out,)
        return Transformer2DModelOutput(sample=out)

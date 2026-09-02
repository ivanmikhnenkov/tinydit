"""Measure real training-step throughput on this GPU for candidate model sizes and text lengths.

    PYTHONPATH=src .venv/bin/python scripts/bench_step.py [--bs 256] [--compile]

One step = forward + backward + fused AdamW on random latents (B,32,32,32) and random T5
context (B,L,768), bf16 autocast, exactly like train.py. Reports s/step, img/s, peak memory
and the implied wall-clock for a 110M-sample run (4.4M images x 25 epochs).
"""
from __future__ import annotations
import argparse, time, torch, torch.nn.functional as F
from tinydit.model import TinyDiT
from tinydit import flow

SIZES = {  # name: (dim, depth, heads)
    "768x12 (current base)": (768, 12, 12),
    "768x16": (768, 16, 12),
    "896x16": (896, 16, 14),
    "1024x16": (1024, 16, 16),
}


def bench(dim, depth, heads, bs, L, compile_, steps=20, warm=5):
    torch.manual_seed(0)
    m = TinyDiT(latent_ch=32, dim=dim, depth=depth, heads=heads, ctx_dim=768).cuda()
    n = m.n_params()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.0, fused=True)
    net = torch.compile(m) if compile_ else m
    z = torch.randn(bs, 32, 32, 32, device="cuda")
    ctx = torch.randn(bs, L, 768, device="cuda")
    lens = torch.randint(8, L + 1, (bs,), device="cuda")
    msk = torch.arange(L, device="cuda")[None] < lens[:, None]
    torch.cuda.reset_peak_memory_stats()
    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l, _, _ = flow.loss(net, z, ctx, msk)
        opt.zero_grad(set_to_none=True); l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    t0 = time.time()
    for _ in range(warm): step()
    torch.cuda.synchronize(); tw = time.time() - t0
    t0 = time.time()
    for _ in range(steps): step()
    torch.cuda.synchronize(); dt = (time.time() - t0) / steps
    mem = torch.cuda.max_memory_allocated() / 2**30
    del m, net, opt; torch.cuda.empty_cache()
    return n, dt, mem, tw


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--compile", action="store_true"); ap.add_argument("--only", default=None)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    # peak bf16 matmul with warmup, for an MFU reference
    x = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    for _ in range(5): x @ x
    torch.cuda.synchronize(); t = time.time()
    for _ in range(20): x @ x
    torch.cuda.synchronize(); peak = 20 * 2 * 8192**3 / (time.time() - t) / 1e12
    print(f"bf16 matmul peak ~{peak:.0f} TFLOP/s  | bs {a.bs} | compile {a.compile}", flush=True)
    print(f"{'model':24s} {'params':>7s} {'L':>4s} {'s/step':>7s} {'img/s':>6s} {'GB':>5s} {'TFLOP/s':>8s} {'MFU':>5s} {'110M samples':>13s}")
    N_IMG = 256  # image tokens at 256^2, patch 2
    for name, (dim, depth, heads) in SIZES.items():
        if a.only and a.only not in name: continue
        for L in ([32, 64, 128] if name.startswith("768x16") else [128]):
            n, dt, mem, tw = bench(dim, depth, heads, a.bs, L, a.compile)
            # 6 * params_used * tokens, counting text K/V and adaLN roughly; good enough for MFU
            flops = 6 * n * N_IMG * a.bs
            tf = flops / dt / 1e12
            hours = 110e6 / (a.bs / dt) / 3600
            print(f"{name:24s} {n/1e6:6.1f}M {L:4d} {dt:7.3f} {a.bs/dt:6.0f} {mem:5.1f} {tf:8.0f} {100*tf/peak:4.0f}% {hours/24:9.1f} days"
                  + (f"   (compile+warmup {tw:.0f}s)" if a.compile else ""), flush=True)

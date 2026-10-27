"""Prompt curriculum pre-pass: measure the base model's pass rate per training prompt and keep the band where group-relative
RL has a signal (the pass@k study: prompts solved 0 of 64 times never teach, prompts solved 61+ have nothing to teach).
    PYTHONPATH=src .venv/bin/python scripts/rl/prompt_band.py --n 3000 --seeds 8 --lo 1 --hi 6
-> out/diag/prompt_band.json (per-prompt counts), prompts/rl_stage2/train_band.json (the kept prompts, same format as train.json)
"""
import argparse, json, os, random, sys, time
sys.path.insert(0, "src")
import numpy as np, torch
ap = argparse.ArgumentParser(); ap.add_argument("--prompts", default="prompts/rl_stage2/train.json"); ap.add_argument("--n", type=int, default=3000); ap.add_argument("--seeds", type=int, default=8)
ap.add_argument("--lo", type=int, default=1); ap.add_argument("--hi", type=int, default=6); ap.add_argument("--steps", type=int, default=10); ap.add_argument("--cfg", type=float, default=4.0); ap.add_argument("--shift", type=float, default=2.8)
ap.add_argument("--base", default="out/runs/run1/ema_0400000.safetensors"); ap.add_argument("--chunk", type=int, default=128); ap.add_argument("--batch", type=int, default=16); ap.add_argument("--thr", type=float, default=0.5)
ap.add_argument("--out", default="prompts/rl_stage2/train_band.json"); ap.add_argument("--scores", default="out/diag/prompt_band.json")
a = ap.parse_args(); torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
from tinydit import ae as AE
from tinydit.sample import load_model, decode_chunked
from tinydit.rl import rollout as R, judges as JD
from tinydit.rl.train import Text
allp = json.load(open(a.prompts)); rng = random.Random(0)
by_src = {}
for q in allp: by_src.setdefault(q["source"], []).append(q)
per = a.n // len(by_src); items = sum([rng.sample(v, min(per, len(v))) for v in by_src.values()], []); rng.shuffle(items)
print(f"{len(items)} prompts from {{{', '.join(f'{k}: {len(v)}' for k, v in by_src.items())}}}, {a.seeds} seeds, {a.steps}-step ODE", flush=True)
lat_ch = AE.SPECS["flux2"]["ch"]; HW = 32
model, _ = load_model(a.base, lat_ch, "run1"); model.eval(); enc = Text("out/models"); aem = AE.load("flux2", "out/models/flux2")
st = json.load(open(next(q for q in ("out/cache/run1/stats.json", "out/hf_model/stats.json") if os.path.exists(q))))
mean = torch.tensor(st["mean"], device="cuda").view(1, -1, 1, 1).float(); std = torch.tensor(st["std"], device="cuda").view(1, -1, 1, 1).float()
vlm = JD.VLM("qwen3vl8b"); counts = np.zeros(len(items), dtype=int); meanp = np.zeros(len(items)); t0 = time.time()
for i in range(0, len(items), a.chunk):
    sl = items[i:i + a.chunk]; texts = [q["prompt"] for q in sl]; ctx, msk = enc(texts); imgs = []
    for s in range(a.seeds):
        xs, _, _ = R.rollout(model, ctx, msk, enc.null_ctx, enc.null_msk, lat_ch, HW, HW, a.steps, a.cfg, a.shift, 0.0, seed=30_000 + s * 1000 + i, chunk=a.chunk)
        imgs.append(decode_chunked(aem, xs[:, -1], mean, std))
    imgs = np.concatenate(imgs); sc = vlm.yesno(imgs, "align", texts * a.seeds, a.batch).reshape(a.seeds, len(sl))
    counts[i:i + len(sl)] = (sc >= a.thr).sum(0); meanp[i:i + len(sl)] = sc.mean(0)
    print(f"  {i + len(sl)}/{len(items)}  {time.time() - t0:.0f}s  band so far {((counts[:i+len(sl)] >= a.lo) & (counts[:i+len(sl)] <= a.hi)).mean():.2f}", flush=True)
keep = [dict(q, base_pass=int(c), base_p=float(m)) for q, c, m in zip(items, counts, meanp) if a.lo <= c <= a.hi]
json.dump([dict(q, base_pass=int(c), base_p=float(m)) for q, c, m in zip(items, counts, meanp)], open(a.scores, "w"))
json.dump(keep, open(a.out, "w"), indent=0)
hist = np.bincount(counts, minlength=a.seeds + 1)
print(f"pass counts of {a.seeds}: " + " ".join(f"{k}:{v}" for k, v in enumerate(hist)) + f"\nkept {len(keep)} / {len(items)} prompts with {a.lo} <= passes <= {a.hi} -> {a.out}", flush=True)
for src in by_src: 
    sel = [q for q in keep if q["source"] == src]; print(f"  {src}: {len(sel)} kept", flush=True)

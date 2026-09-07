"""Metrics of the final EMA weights on the frozen held-out FID set at different sampler settings.

    PYTHONPATH=src .venv/bin/python scripts/eval_steps.py --run run1 --settings 8:2.8 20:2.8 50:2.8 20:1.0
"""
import argparse, json, os, sys, time
sys.path.insert(0, "src")
import numpy as np, torch
from tinydit import ae as AE, metrics
from tinydit.data import BucketCache
from tinydit.sample import load_model, generate, decode_chunked

ap = argparse.ArgumentParser(); ap.add_argument("--run", default="run1"); ap.add_argument("--cache", default="out/cache/run1")
ap.add_argument("--config", default="run1"); ap.add_argument("--cfg", type=float, default=4.0)
ap.add_argument("--settings", nargs="+", default=["20:2.8", "50:2.8"]); a = ap.parse_args()
out = os.path.join("out", "runs", a.run)
EP = json.load(open(os.path.join(out, "eval_prompts.json")))
C = BucketCache(a.cache); lat_ch = AE.SPECS["flux2"]["ch"]
snap = sorted(f for f in os.listdir(out) if f.startswith("ema_") and f.endswith(".safetensors"))[-1]
model, _ = load_model(os.path.join(out, snap), lat_ch, a.config); aem = AE.load("flux2", "out/models/flux2")
null_ctx, null_msk = C.null()
ref, per_b = [], {}
for b in C.names:
    vi = C.val_images(b); vp = [(c, j) for bb, c, j in EP["fid"] if bb == b]
    per_b[b] = vp; ref += [np.asarray(vi[j]) for _, j in vp if j < len(vi)]
print(f"{snap}: {sum(len(v) for v in per_b.values())} held-out prompts, {len(ref)} references", flush=True)
for s in a.settings:
    steps, shift = int(s.split(":")[0]), float(s.split(":")[1]); t0 = time.time(); gen, txt = [], []
    for b in C.names:
        ps = [c for c, _ in per_b[b]]
        if not ps: continue
        W, H = C.buckets[b]; ctx, msk = C.text_for(ps)
        z = generate(model, ctx, msk, null_ctx, null_msk, lat_ch, H // 8, W // 8, steps, a.cfg, 777, shift)
        gen += list(decode_chunked(aem, z, C.mean, C.std)); txt += ps
    fid, fdd = metrics.frechet(gen, ref)
    by = {}
    for i, im in enumerate(gen): by.setdefault(im.shape, []).append(i)
    def per_shape(fn): return float(np.mean(sum((fn(np.stack([gen[i] for i in ids]), [txt[i] for i in ids]) for ids in by.values()), [])))
    rec = dict(steps=steps, shift=shift, fid=round(fid, 2), fd_dino=round(fdd, 1), clip=round(per_shape(metrics.clip_score), 4),
               pickscore=round(per_shape(metrics.pickscore), 3), hpsv2=round(per_shape(metrics.hpsv2), 4), secs=int(time.time() - t0))
    print("RESULT", json.dumps(rec), flush=True)
metrics.unload()

"""Regenerate one evaluation (grid images + CLIP/PickScore/HPSv2 rows) from a saved EMA snapshot with
the run's frozen prompt set, so every eval of a run is on identical prompts. FID/FD-DINOv2/object
accuracy values already in metrics.jsonl are kept (they were computed on the same frozen FID set).

    python scripts/regen_eval.py --run run1 --step 10000
"""
import argparse, json, os, sys
sys.path.insert(0, "src")
import numpy as np, torch
from tinydit import ae as AE, metrics
from tinydit.data import BucketCache
from tinydit.sample import load_model, generate, decode_chunked, render_sections, write_eval_assets

p = argparse.ArgumentParser()
p.add_argument("--run", default="run1"); p.add_argument("--step", type=int, required=True)
p.add_argument("--cache", default="out/cache/run1"); p.add_argument("--config", default="run1")
p.add_argument("--steps", type=int, default=20); p.add_argument("--cfg", type=float, default=4.0)
p.add_argument("--shift", type=float, default=2.8); p.add_argument("--novel", default="prompts/novel.txt")
a = p.parse_args()
out = os.path.join("out", "runs", a.run)
EP = json.load(open(os.path.join(out, "eval_prompts.json")))
novel = [l.strip() for l in open(a.novel) if l.strip()]
C = BucketCache(a.cache)
lat_ch = AE.SPECS["flux2"]["ch"]
model, _ = load_model(os.path.join(out, f"ema_{a.step:07d}.safetensors"), lat_ch, a.config)
aem = AE.load("flux2", "out/models/flux2")
null_ctx, null_msk = C.null()

def gen(prompts, wh, seed):
    ctx, msk = C.text_for(prompts); W, H = wh
    z = generate(model, ctx, msk, null_ctx, null_msk, lat_ch, H // 8, W // 8, a.steps, a.cfg, seed, a.shift)
    return decode_chunked(aem, z, C.mean, C.std)

secs, unseen_imgs, unseen_txt = [], [], []
for b in C.names:
    ps = [c for bb, c, _ in EP["grid"] if bb == b]
    if not ps: continue
    img = gen(ps, tuple(C.buckets[b]), 1234); sc = metrics.clip_score(img, ps)
    secs.append((f"UNSEEN {b.replace('_', ':')} · held-out captions ({C.buckets[b][0]}x{C.buckets[b][1]})", ps, img, sc, sc))
    unseen_imgs += list(img); unseen_txt += ps
img_n = gen(novel, (256, 256), 4321); sc_n = metrics.clip_score(img_n, novel)
secs.append(("NOVEL · hand-written prompts", novel[:12], img_n[:12], sc_n[:12], sc_n))
render_sections(secs, os.path.join(out, "evals", f"{a.step}.png"), a.step, a.cfg, a.steps)
write_eval_assets(secs, os.path.join(out, "evals"), a.step, a.cfg, a.steps)

def per_shape(fn, imgs, txt):
    by = {}
    for i, im in enumerate(imgs): by.setdefault(im.shape, []).append(i)
    vals = []
    for ids in by.values(): vals += fn(np.stack([imgs[i] for i in ids]), [txt[i] for i in ids])
    return float(np.mean(vals))
rec = {"clip_unseen": per_shape(metrics.clip_score, unseen_imgs, unseen_txt), "clip_novel": float(np.mean(sc_n))}
for name, fn in (("pickscore", metrics.pickscore), ("hpsv2", metrics.hpsv2)):
    rec[f"{name}_unseen"] = per_shape(fn, unseen_imgs, unseen_txt); rec[f"{name}_novel"] = float(np.mean(fn(img_n, novel)))
metrics.unload()

# update the eval row in place (same inode: the trainer appends with O_APPEND, so no rename)
mp = os.path.join(out, "metrics.jsonl")
with open(mp, "r+") as f:
    lines = f.read().split("\n")
    for i, l in enumerate(lines):
        if not l: continue
        r = json.loads(l)
        if r.get("step") == a.step and "eval" in r:
            r.update(rec); lines[i] = json.dumps(r); print("updated row:", lines[i][:200])
    f.seek(0); f.write("\n".join(lines)); f.truncate()
print("regenerated eval", a.step, {k: round(v, 3) for k, v in rec.items()})

"""Cheap scorers over the reward-eval set. Each image gets: clip, pickscore, hpsv2 (own prompt), the same three with a
mismatched caption (alignment test), pickscore/hpsv2 after a JPEG-90 re-encode (noise floor), DINOv2 features -> a
real-vs-render probe trained on half of the reference images, and saturation / sharpness statistics.
Writes out/reward_eval/scores/cheap.json  {group: {file: {metric: value}}}.
    .venv/bin/python scripts/reward_eval/score_cheap.py
"""
import io, json, os, random, sys, time
sys.path.insert(0, "src")
import numpy as np, torch
from PIL import Image
from tinydit import metrics
E = "out/reward_eval"; M = json.load(open(f"{E}/manifest.json")); random.seed(0)
def load(files): return np.stack([np.asarray(Image.open(f"{E}/{f}").convert("RGB")).transpose(2, 0, 1) for f in files])
def jpeg(imgs, q=90):
    out = []
    for im in imgs:
        b = io.BytesIO(); Image.fromarray(im.transpose(1, 2, 0)).save(b, "JPEG", quality=q); out.append(np.asarray(Image.open(b).convert("RGB")).transpose(2, 0, 1))
    return np.stack(out)
def stats(imgs):
    out = []
    for im in imgs:
        hsv = np.asarray(Image.fromarray(im.transpose(1, 2, 0)).convert("HSV")).astype(np.float32)
        g = np.asarray(Image.fromarray(im.transpose(1, 2, 0)).convert("L")).astype(np.float32)
        lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
        out.append(dict(sat=float(hsv[..., 1].mean() / 255), sharp=float(lap.var()), contrast=float(g.std() / 255)))
    return out
groups = {"samples": [(s["file"], s["prompt"]) for s in M["samples"]],
          "ref_real": [(f"ref/real/{r['idx']:04d}.png", r["caption"]) for r in M["ref"]["real"]],
          "ref_flux": [(f"ref/flux/{r['idx']:04d}.png", r["caption"]) for r in M["ref"]["flux"]]}
R = {g: {} for g in groups}; feats = {}
t0 = time.time()
for g, items in groups.items():
    files = [f for f, _ in items]; prompts = [p for _, p in items]
    mis = prompts[:]; random.shuffle(mis); mis = [m if m != p else prompts[(i + 1) % len(prompts)] for i, (m, p) in enumerate(zip(mis, prompts))]
    imgs = load(files); jp = jpeg(imgs); st = stats(imgs)
    sc = {"clip": metrics.clip_score(imgs, prompts), "clip_mis": metrics.clip_score(imgs, mis),
          "pickscore": metrics.pickscore(imgs, prompts), "pickscore_mis": metrics.pickscore(imgs, mis), "pickscore_jpeg": metrics.pickscore(jp, prompts),
          "hpsv2": metrics.hpsv2(imgs, prompts), "hpsv2_mis": metrics.hpsv2(imgs, mis), "hpsv2_jpeg": metrics.hpsv2(jp, prompts)}
    feats[g] = np.asarray(metrics._dino_feats(imgs, "cuda"))
    for i, f in enumerate(files):
        R[g][f] = {k: float(v[i]) for k, v in sc.items()}; R[g][f].update(st[i]); R[g][f]["mismatch_prompt"] = mis[i]
    print(f"  {g}: {len(files)} images scored, {time.time() - t0:.0f}s", flush=True)
metrics.unload()
# real-vs-render probe on DINOv2 CLS features: train on the second half of each reference set, test on the first half
def probe(Xtr, ytr, Xte):
    Xtr = torch.tensor(Xtr).float(); ytr = torch.tensor(ytr).float(); mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    w = torch.zeros(Xtr.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=200)
    def closure():
        opt.zero_grad(); z = ((Xtr - mu) / sd) @ w + b
        l = torch.nn.functional.binary_cross_entropy_with_logits(z, ytr) + 1e-2 * (w ** 2).sum(); l.backward(); return l
    opt.step(closure)
    with torch.no_grad(): return {k: torch.sigmoid(((torch.tensor(X).float() - mu) / sd) @ w + b).numpy() for k, X in Xte.items()}
fr, ff = feats["ref_real"], feats["ref_flux"]; h = len(fr) // 2
pr = probe(np.concatenate([fr[h:], ff[h:]]), np.r_[np.ones(len(fr) - h), np.zeros(len(ff) - h)],
           {"ref_real": fr, "ref_flux": ff, "samples": feats["samples"]})
for g in groups:
    for i, (f, _) in enumerate(groups[g]): R[g][f]["dino_real"] = float(pr[g][i]); R[g][f]["dino_probe_train"] = bool(g != "samples" and i >= h)
os.makedirs(f"{E}/scores", exist_ok=True); json.dump(R, open(f"{E}/scores/cheap.json", "w"))
np.savez(f"{E}/scores/dino_feats.npz", **feats)
print(f"done in {time.time() - t0:.0f}s -> {E}/scores/cheap.json", flush=True)

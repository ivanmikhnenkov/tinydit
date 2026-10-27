"""pass@k: does RL make the model solve new prompts, or solve the same prompts more often?
For each checkpoint: the 125 fixed held-out prompts x K seeds with the deployment sampler (20-step ODE, CFG 4, shift 2.8); every
image judged with the alignment question by Qwen3-VL-8B (the reward model) and Qwen3.5-9B (independent).
pass@k per prompt = 1 - C(n-c, k) / C(n, k), c = samples with P(yes) >= threshold (Chen et al. 2021 unbiased estimator);
soft best-of-k = expected maximum P(yes) over k draws (order statistics). Both averaged over prompts, overall and per family.
    PYTHONPATH=src .venv/bin/python scripts/rl/passk.py --ckpt base=out/runs/run1/ema_0400000.safetensors --ckpt v5_221=out/runs/rl2_dgpo_v5/policy_000221.safetensors
-> out/passk/images_<name>.npy (K,P,3,256,256 uint8), scores_<name>_<judge>.npy (K,P), summary.json, grids/<prompt>_<name>.jpg
"""
import argparse, json, os, sys, time
sys.path.insert(0, "src")
import numpy as np, torch
from math import comb
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", action="append", required=True, help="name=path (.safetensors), repeatable")
ap.add_argument("--k", type=int, default=64); ap.add_argument("--judges", default="qwen3vl8b,qwen35_9b"); ap.add_argument("--thr", type=float, default=0.5)
ap.add_argument("--eval-set", default="prompts/rl_stage2/eval.json"); ap.add_argument("--steps", type=int, default=20); ap.add_argument("--cfg", type=float, default=4.0)
ap.add_argument("--shift", type=float, default=2.8); ap.add_argument("--batch", type=int, default=16); ap.add_argument("--out", default="out/passk"); ap.add_argument("--config", default="run1")
a = ap.parse_args(); os.makedirs(os.path.join(a.out, "grids"), exist_ok=True)
EV = json.load(open(a.eval_set)); items = EV["prompts"]; prompts = [q["prompt"] for q in items]; P = len(prompts); K = a.k
names = [c.split("=", 1)[0] for c in a.ckpt]; paths = dict(c.split("=", 1) for c in a.ckpt)
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True

# ---- generation
todo = [n for n in names if not os.path.exists(os.path.join(a.out, f"images_{n}.npy"))]
if todo:
    from tinydit import ae as AE
    from tinydit.sample import load_model, decode_chunked
    from tinydit.rl import rollout as R
    from tinydit.rl.train import Text
    lat_ch = AE.SPECS["flux2"]["ch"]; HW = 32
    enc = Text("out/models"); aem = AE.load("flux2", "out/models/flux2")
    st = json.load(open(next(q for q in ("out/cache/run1/stats.json", "out/hf_model/stats.json") if os.path.exists(q))))
    mean = torch.tensor(st["mean"], device="cuda").view(1, -1, 1, 1).float(); std = torch.tensor(st["std"], device="cuda").view(1, -1, 1, 1).float()
    ctx, msk = enc(prompts)
    for n in todo:
        t0 = time.time(); model, _ = load_model(paths[n], lat_ch, a.config); model.eval()
        imgs = np.zeros((K, P, 3, 256, 256), dtype=np.uint8)
        for s in range(K):
            xs, _, _ = R.rollout(model, ctx, msk, enc.null_ctx, enc.null_msk, lat_ch, HW, HW, a.steps, a.cfg, a.shift, 0.0, seed=20_000 + s, chunk=128)
            imgs[s] = decode_chunked(aem, xs[:, -1], mean, std)
            if s % 16 == 15: print(f"  {n}: {s+1}/{K} seeds, {time.time()-t0:.0f}s", flush=True)
        np.save(os.path.join(a.out, f"images_{n}.npy"), imgs); del model; torch.cuda.empty_cache()
        print(f"{n}: {K*P} images in {time.time()-t0:.0f}s", flush=True)
    del aem, enc; torch.cuda.empty_cache()

# ---- judging
for jn in a.judges.split(","):
    todo = [n for n in names if not os.path.exists(os.path.join(a.out, f"scores_{n}_{jn}.npy"))]
    if not todo: continue
    from tinydit.rl import judges as JD
    vlm = JD.VLM(jn); rep = [p for _ in range(K) for p in prompts]
    for n in todo:
        t0 = time.time(); imgs = np.load(os.path.join(a.out, f"images_{n}.npy"), mmap_mode="r")
        sc = np.zeros((K, P), dtype=np.float32)
        for s0 in range(0, K, 8):                                   # 8 seeds = 1000 images per pass keeps the PIL conversion small
            sl = np.ascontiguousarray(imgs[s0:s0 + 8]).reshape(-1, 3, 256, 256)
            sc[s0:s0 + 8] = vlm.yesno(sl, "align", rep[:len(sl)], a.batch).reshape(-1, P)
        np.save(os.path.join(a.out, f"scores_{n}_{jn}.npy"), sc)
        print(f"{jn} x {n}: mean P(yes) {sc.mean():.3f}, {K*P/(time.time()-t0):.1f} img/s", flush=True)
    del vlm; torch.cuda.empty_cache()

# ---- summary
def pass_at_k(c, n, k): return 1.0 - comb(n - c, k) / comb(n, k) if n - c >= k else 1.0
def soft_best(x, k):                                                # E[max of k draws without replacement], x sorted ascending
    n = len(x); return sum(x[i] * comb(i, k - 1) for i in range(k - 1, n)) / comb(n, k)
ks = [k for k in (1, 2, 4, 8, 16, 32, 64, 128) if k <= K]; fams = sorted({q["source"] for q in items}); S = {}
for jn in a.judges.split(","):
    S[jn] = {}
    for n in names:
        sc = np.load(os.path.join(a.out, f"scores_{n}_{jn}.npy")); c = (sc >= a.thr).sum(0); srt = np.sort(sc, 0)
        rec = {"mean_p": float(sc.mean()), "c_per_prompt": c.tolist(), "n_zero": int((c == 0).sum()), "n_all": int((c == K).sum()),
               "pass": {str(k): float(np.mean([pass_at_k(int(ci), K, k) for ci in c])) for k in ks},
               "soft": {str(k): float(np.mean([soft_best(srt[:, p], k) for p in range(P)])) for k in ks}}
        for f in fams:
            sel = np.array([q["source"] == f for q in items])
            rec[f"pass_{f}"] = {str(k): float(np.mean([pass_at_k(int(ci), K, k) for ci in c[sel]])) for k in ks}
        S[jn][n] = rec
    base = np.array(S[jn][names[0]]["c_per_prompt"])
    for n in names[1:]:
        c = np.array(S[jn][n]["c_per_prompt"]); S[jn][n]["newly_reachable"] = int(((base == 0) & (c > 0)).sum()); S[jn][n]["lost"] = int(((base > 0) & (c == 0)).sum())
        S[jn][n]["newly_reachable_ids"] = [items[i]["id"] for i in np.where((base == 0) & (c > 0))[0]]
json.dump({"k": K, "thr": a.thr, "names": names, "prompts": items, "summary": S}, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
for jn in S:
    print(f"\n{jn}  pass@k (thr {a.thr})          " + " ".join(f"{k:>6d}" for k in ks) + "   zero  newly lost")
    for n in names:
        r = S[jn][n]; print(f"  {n:10s} mean {r['mean_p']:.3f}  " + " ".join(f"{r['pass'][str(k)]:6.3f}" for k in ks) + f"   {r['n_zero']:4d}  {r.get('newly_reachable', 0):5d} {r.get('lost', 0):4d}")
    for n in names:
        r = S[jn][n]; print(f"  {n:10s} soft best-of-k " + " ".join(f"{r['soft'][str(k)]:6.3f}" for k in ks))

# ---- grids: hardest prompts under the base model (8B), 4x4 of the first 16 seeds per checkpoint
j0 = a.judges.split(",")[0]; sc0 = np.load(os.path.join(a.out, f"scores_{names[0]}_{j0}.npy")); order = np.argsort(sc0.mean(0))[:8]
try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
except OSError: font = ImageFont.load_default()
for p in order:
    for n in names:
        imgs = np.load(os.path.join(a.out, f"images_{n}.npy"), mmap_mode="r"); sc = np.load(os.path.join(a.out, f"scores_{n}_{j0}.npy"))
        canvas = Image.new("RGB", (4 * 128, 4 * 140), (247, 248, 248)); d = ImageDraw.Draw(canvas)
        for s in range(16):
            r, c = divmod(s, 4); canvas.paste(Image.fromarray(np.ascontiguousarray(imgs[s, p]).transpose(1, 2, 0)).resize((128, 128), Image.LANCZOS), (c * 128, r * 140))
            d.text((c * 128 + 2, r * 140 + 128), f"{sc[s, p]:.2f}", font=font, fill=(60, 70, 76))
        canvas.save(os.path.join(a.out, "grids", f"{items[p]['id']}_{n}.jpg"), quality=88)
print(f"grids for {[items[p]['id'] for p in order]} -> {a.out}/grids")

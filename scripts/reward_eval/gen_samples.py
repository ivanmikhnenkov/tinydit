"""Base-model samples for the reward/judge study: every prompt in the set x seeds, 20-step ODE, CFG 4, 256x256,
plus the reference real / FLUX held-out images exported as PNGs. Writes out/reward_eval/{samples,ref}/ and manifest.json.
    .venv/bin/python scripts/reward_eval/gen_samples.py --ckpt out/runs/run1/ema_0400000.safetensors
"""
import argparse, json, os, sys, time
sys.path.insert(0, "src")
import numpy as np, torch
from PIL import Image
from tinydit import ae as AE, text as T
from tinydit.sample import load_model, generate, decode_chunked
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", default="out/runs/run1/ema_0400000.safetensors")
ap.add_argument("--set", default="prompts/rl_eval/set_v1.json"); ap.add_argument("--out", default="out/reward_eval")
ap.add_argument("--chunk", type=int, default=16); a = ap.parse_args()
S = json.load(open(a.set)); os.makedirs(f"{a.out}/samples", exist_ok=True)
for k in ("real", "flux"): os.makedirs(f"{a.out}/ref/{k}", exist_ok=True)
vi = np.load("out/cache/run1/1_1/val_images.npy", mmap_mode="r")
for k in ("real", "flux"):
    for r in S["ref"][k]:
        p = f"{a.out}/ref/{k}/{r['idx']:04d}.png"
        if not os.path.exists(p): Image.fromarray(np.asarray(vi[r["idx"]]).transpose(1, 2, 0)).save(p)
st = json.load(open("out/cache/run1/stats.json"))
mean = torch.tensor(st["mean"]).view(1, -1, 1, 1).cuda(); std = torch.tensor(st["std"]).view(1, -1, 1, 1).cuda()
tok, enc = T.load("out/models/t5"); nctx, nmsk = T.embed(tok, enc, [""], T.MAX_SHORT)
model, _ = load_model(a.ckpt, 32, "run1"); aem = AE.load("flux2", "out/models/flux2")
man, t0 = [], time.time()
sp = S["sampler"]
for i in range(0, len(S["prompts"]), a.chunk):
    ps = S["prompts"][i:i + a.chunk]; texts = [p["prompt"] for p in ps]
    e, m = T.embed_mixed(tok, enc, texts, [len(t.split()) > 30 for t in texts])
    for seed in range(S["seeds"]):
        todo = [k for k, p in enumerate(ps) if not os.path.exists(f"{a.out}/samples/{p['id']}_s{seed}.png")]
        if todo:
            z = generate(model, e.float(), m, nctx.float(), nmsk, 32, 32, 32, sp["steps"], sp["cfg"], 1000 * seed + i, sp["shift"])
            img = decode_chunked(aem, z, mean, std)
            for k in todo: Image.fromarray(img[k].transpose(1, 2, 0)).save(f"{a.out}/samples/{ps[k]['id']}_s{seed}.png")
        for k, p in enumerate(ps): man.append(dict(p, seed=seed, file=f"samples/{p['id']}_s{seed}.png"))
    print(f"  {i + len(ps)}/{len(S['prompts'])} prompts, {time.time() - t0:.0f}s", flush=True)
json.dump({"samples": man, "ref": S["ref"], "sampler": sp, "ckpt": a.ckpt}, open(f"{a.out}/manifest.json", "w"))
print(f"done: {len(man)} samples, {len(S['ref']['real'])} real refs, {len(S['ref']['flux'])} flux refs", flush=True)

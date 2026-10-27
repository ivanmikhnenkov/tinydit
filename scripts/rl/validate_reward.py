"""Reward validation on hand-made hacks: score blank, noise, over-saturated, text-stamped and tiled versions of base samples
with every stage-2 term and report where they land inside their prompt group (they must sink to the bottom).
    PYTHONPATH=src .venv/bin/python scripts/rl/validate_reward.py"""
import json, sys, numpy as np
sys.path.insert(0, "src")
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from tinydit.rl.reward_stage2 import Stage2Reward
E = "out/reward_eval"; M = json.load(open(f"{E}/manifest.json"))
ids = ["usual00", "usual03", "odd02", "odd03", "odd16", "real00", "flux03", "odd19"]
R2 = Stage2Reward(batch=8)
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
def hacks(im, prompt):
    pil = Image.fromarray(im.transpose(1, 2, 0)); out = {}
    out["blank"] = np.full_like(im, 128); out["noise"] = np.random.default_rng(0).integers(0, 256, im.shape, dtype=np.uint8)
    out["saturated"] = np.asarray(ImageEnhance.Color(ImageEnhance.Contrast(pil).enhance(1.6)).enhance(2.2)).transpose(2, 0, 1)
    t = pil.copy(); ImageDraw.Draw(t).text((8, 8), prompt[:28], font=font, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0)); out["text"] = np.asarray(t).transpose(2, 0, 1)
    sm = pil.resize((128, 128)); tl = Image.new("RGB", (256, 256)); [tl.paste(sm, (x, y)) for x in (0, 128) for y in (0, 128)]; out["tiled"] = np.asarray(tl).transpose(2, 0, 1)
    return out
rows = []
for pid in ids:
    items = [s for s in M["samples"] if s["id"] == pid]; prompt = items[0]["prompt"]
    imgs = np.stack([np.asarray(Image.open(f"{E}/{s['file']}").convert("RGB")).transpose(2, 0, 1) for s in items])
    hk = hacks(imgs[0], prompt); names = ["seed%d" % i for i in range(len(imgs))] + list(hk)
    allimgs = np.concatenate([imgs, np.stack(list(hk.values()))])
    T = R2.terms(allimgs, [prompt] * len(allimgs)); adv, A = R2.advantage(T, np.zeros(len(allimgs), dtype=int))
    order = np.argsort(-adv); rank = {names[i]: int(np.where(order == i)[0][0]) + 1 for i in range(len(names))}
    rows.append({"id": pid, "n": len(names), "names": names, "all_terms": {t: [float(x) for x in T[t]] for t in T}, "rank": {k: rank[k] for k in hk}, "adv": {k: float(adv[len(imgs) + j]) for j, k in enumerate(hk)},
                 "terms": {k: {t: float(T[t][len(imgs) + j]) for t in ("align", "real", "quality", "anatomy", "style")} for j, k in enumerate(hk)}, "veto": {k: float(T["veto"][len(imgs) + j]) for j, k in enumerate(hk)}})
    print(f"  {pid:8s} ranks of hacks among {len(names)}: " + " ".join(f"{k}={rank[k]}" for k in hk) + "  vetoed: " + ",".join(k for k in hk if T["veto"][len(imgs) + list(hk).index(k)] > 0), flush=True)
json.dump(rows, open(f"{E}/validate_hacks.json", "w"), indent=1); print("-> out/reward_eval/validate_hacks.json")

"""Progress clip v2: a 3x2 grid of prompts, seeds chosen at the final checkpoint by PickScore so the
endings are strong, held longer at the start and accelerating toward the end.

    PYTHONPATH=src .venv/bin/python scripts/make_progress_video.py --run run1 --out out/videos
"""
import argparse, base64, glob, io, os, sys
sys.path.insert(0, "src")
import numpy as np, imageio.v3 as iio
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file
from tinydit.playground import Engine
from tinydit import metrics

ap = argparse.ArgumentParser(); ap.add_argument("--run", default="run1"); ap.add_argument("--cache", default="out/cache/run1")
ap.add_argument("--out", default="out/videos"); ap.add_argument("--seeds", type=int, default=8); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"; FONTB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
BG, INK, INK2, ACC = (252, 252, 251), (11, 11, 11), (82, 81, 78), (42, 120, 214)
dec = lambda u: Image.open(io.BytesIO(base64.b64decode(u.split(",")[1]))).convert("RGB")
PROMPTS = ["a red tractor parked next to a blue rowing boat on a sandy beach", "a bowl of fresh strawberries on a wooden table",
           "a lighthouse on a rocky coast at sunset", "a golden retriever wearing sunglasses sitting on a yellow armchair",
           "a bowl of ramen with a soft-boiled egg and green onions on a dark table", "a white horse standing in a field of sunflowers"]
E = Engine(a.run, a.cache); own = E.model.state_dict()
gen = lambda p, seed: dec(E.generate(p, 256, 256, steps=20, cfg=4.0, shift=2.8, seed=seed, n_images=1, trajectory=0)["images"][0])

# 1) at the final checkpoint, pick the best of N seeds per prompt by PickScore (a learned human-preference model)
best = []
for p in PROMPTS:
    cands = [gen(p, s) for s in range(a.seeds)]
    sc = metrics.pickscore(np.stack([np.asarray(c).transpose(2, 0, 1) for c in cands]), [p] * len(cands))
    best.append(int(np.argmax(sc))); print(f"  {p[:50]:50s} best seed {best[-1]} (PickScore {max(sc):.2f})", flush=True)
metrics.unload()

# 2) render every snapshot with those seeds
snaps = sorted(glob.glob(f"out/runs/{a.run}/ema_*.safetensors")); S = 2; gap = 22; cols = 3
fT, fS, fC = ImageFont.truetype(FONTB, 30), ImageFont.truetype(FONT, 20), ImageFont.truetype(FONT, 16)
def compose(imgs, step):
    ps = [im.resize((256 * S, 256 * S), Image.LANCZOS) for im in imgs]
    W = cols * 256 * S + (cols + 1) * gap; rows = (len(ps) + cols - 1) // cols; H = 100 + rows * (256 * S + 30 + gap) + 20
    im = Image.new("RGB", (W + W % 2, H + H % 2), BG); d = ImageDraw.Draw(im)
    d.text((gap, 18), "tinydit: a 210M text-to-image transformer trained from scratch on one GPU", font=fT, fill=INK)
    d.text((gap, 60), "same prompts and seeds at every checkpoint of the 3.5-day run", font=fS, fill=INK2)
    lab = f"step {step//1000}k / 400k"; d.text((W - gap - d.textlength(lab, font=fS), 60), lab, font=fS, fill=INK)
    for k, (p, cap) in enumerate(zip(ps, PROMPTS)):
        r, c = divmod(k, cols); x = gap + c * (256 * S + gap); y = 100 + r * (256 * S + 30 + gap)
        im.paste(p, (x, y)); mx = int(256 * S / 8.6); d.text((x, y + 256 * S + 7), cap if len(cap) <= mx else cap[:mx - 1] + "…", font=fC, fill=INK2)
    bw = W - 2 * gap; by = H - 8; d.rectangle([gap, by - 5, gap + bw, by], fill=(230, 229, 225)); d.rectangle([gap, by - 5, gap + int(bw * step / 400000), by], fill=ACC)
    return np.asarray(im)

FPS = 24; frames = []; n = len(snaps)
for k, sp in enumerate(snaps):
    st = int(os.path.basename(sp)[4:11]); sd = load_file(sp); E.model.load_state_dict({kk: v.to(own[kk].dtype) for kk, v in sd.items()})
    f = compose([gen(p, best[j]) for j, p in enumerate(PROMPTS)], st)
    hold = int(round(FPS * (0.7 - 0.55 * (k / (n - 1)) ** 0.8)))     # 0.7 s at the start, easing to 0.15 s near the end
    if k == 0: hold = int(FPS * 1.2)
    if k == n - 1: hold = int(FPS * 3.5)
    frames += [f] * max(hold, 3)
E.reload()
mp4 = os.path.join(a.out, "tinydit_progress_v2.mp4"); iio.imwrite(mp4, frames, fps=FPS, codec="libx264", pixelformat="yuv420p", macro_block_size=1)
small = [np.asarray(Image.fromarray(f).resize((960, int(f.shape[0] * 960 / f.shape[1])), Image.LANCZOS)) for f in frames[::3]]
gif = os.path.join(a.out, "tinydit_progress_v2.gif"); iio.imwrite(gif, small, duration=int(3000 / FPS), loop=0)
Image.fromarray(frames[-1]).save(os.path.join(a.out, "tinydit_progress_v2_last.jpg"), quality=88)
print(f"wrote {mp4} ({os.path.getsize(mp4)/1e6:.1f} MB, {len(frames)/FPS:.1f} s), {gif} ({os.path.getsize(gif)/1e6:.1f} MB), frame size {frames[0].shape}", flush=True)

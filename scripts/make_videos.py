"""Two short clips for posts: the same prompts improving across every EMA snapshot, and one sampling
trajectory. MP4 (h264) plus a GIF of each.

    PYTHONPATH=src .venv/bin/python scripts/make_videos.py --run run1 --out out/videos
"""
import argparse, base64, glob, io, os, sys
sys.path.insert(0, "src")
import numpy as np, imageio.v3 as iio
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file
from tinydit.playground import Engine

ap = argparse.ArgumentParser(); ap.add_argument("--run", default="run1"); ap.add_argument("--cache", default="out/cache/run1")
ap.add_argument("--out", default="out/videos"); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"; FONTB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
BG, INK, INK2, ACC = (252, 252, 251), (11, 11, 11), (82, 81, 78), (42, 120, 214)
dec = lambda u: Image.open(io.BytesIO(base64.b64decode(u.split(",")[1]))).convert("RGB")
E = Engine(a.run, a.cache); own = E.model.state_dict()


def frame(panels, title, subtitle, captions, step=None, total=None, scale=2, gap=24, band=150):
    """panels: list of PIL images (same height). Returns an RGB frame with title, captions and a progress bar."""
    ps = [p.resize((p.width * scale, p.height * scale), Image.LANCZOS) for p in panels]
    W = sum(p.width for p in ps) + gap * (len(ps) + 1); H = ps[0].height + band
    im = Image.new("RGB", (W + W % 2, H + H % 2), BG); d = ImageDraw.Draw(im)
    fT, fS, fC = ImageFont.truetype(FONTB, 30), ImageFont.truetype(FONT, 20), ImageFont.truetype(FONT, 17)
    d.text((gap, 18), title, font=fT, fill=INK); d.text((gap, 60), subtitle, font=fS, fill=INK2)
    x = gap; y = 100
    for p, c in zip(ps, captions):
        im.paste(p, (x, y)); d.text((x, y + p.height + 8), c[:int(p.width / 9.5)] + ("…" if len(c) > int(p.width / 9.5) else ""), font=fC, fill=INK2); x += p.width + gap
    if step is not None:
        bw = W - 2 * gap; by = H - 10
        d.rectangle([gap, by - 4, gap + bw, by], fill=(230, 229, 225)); d.rectangle([gap, by - 4, gap + int(bw * step / total), by], fill=ACC)
        lab = f"step {step//1000}k / {total//1000}k"; tw = d.textlength(lab, font=fS); d.text((W - gap - tw, 60), lab, font=fS, fill=INK)
    return np.asarray(im)


def write(frames, name, fps, gif_width=900):
    mp4 = os.path.join(a.out, name + ".mp4")
    iio.imwrite(mp4, frames, fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=None, plugin="pyav") if False else \
        iio.imwrite(mp4, frames, fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=1)
    small = [np.asarray(Image.fromarray(f).resize((gif_width, int(f.shape[0] * gif_width / f.shape[1])), Image.LANCZOS)) for f in frames[::2]]
    gif = os.path.join(a.out, name + ".gif"); iio.imwrite(gif, small, duration=int(2000 / fps), loop=0)
    print(f"wrote {mp4} ({os.path.getsize(mp4)/1e6:.1f} MB, {len(frames)/fps:.1f} s) and {gif} ({os.path.getsize(gif)/1e6:.1f} MB)", flush=True)


# ---- clip 1: same prompts and seeds across all EMA snapshots -------------------------------------------
prompts = ["a red tractor parked next to a blue rowing boat on a sandy beach", "three green apples on a white plate next to a black coffee cup",
           "a lighthouse on a rocky cliff during a storm"]
snaps = sorted(glob.glob(f"out/runs/{a.run}/ema_*.safetensors")); FPS = 20; frames = []
for k, sp in enumerate(snaps):
    st = int(os.path.basename(sp)[4:11]); sd = load_file(sp); E.model.load_state_dict({kk: v.to(own[kk].dtype) for kk, v in sd.items()})
    panels = [dec(E.generate(p, 256, 256, steps=20, cfg=4.0, shift=2.8, seed=100 + j, trajectory=0)["images"][0]) for j, p in enumerate(prompts)]
    f = frame(panels, "tinydit: a 210M text-to-image transformer, trained from scratch on one GPU",
              "same prompt, same seed, every 10k steps of training", prompts, st, 400000)
    hold = 7 if k < len(snaps) - 1 else 50                       # 0.35 s per snapshot, 2.5 s on the final one
    frames += [f] * (14 if k == 0 else hold)
E.reload(); write(frames, "tinydit_progress", FPS)

# ---- clip 2: one sampling trajectory -------------------------------------------------------------------
r = E.generate("a wooden rowing boat pulled up on a pebble shore", 320, 208, steps=20, cfg=4.0, shift=2.8, seed=7, trajectory=20)
frames = []
for i, fr in enumerate(r["trajectory"]):
    f = frame([dec(fr["x_t"]), dec(fr["x1_hat"])], "how one image is sampled: 20 steps of rectified flow",
              f"step {i+1} of 20 · t = {fr['t']:.2f} · left: what the sampler holds · right: the model's prediction of the final image",
              ["noisy state x_t", "predicted final image"], scale=2)
    frames += [f] * (6 if i < 19 else 40)
write(frames, "tinydit_trajectory", FPS)

"""Render the README figures from the final run: prompt-type grids, aspect ratios, training progress
from EMA snapshots, the sampling trajectory, a schedule comparison, and metric curves.

    PYTHONPATH=src .venv/bin/python scripts/make_readme_figures.py --run run1 --out docs [--charts-only]
"""
import argparse, base64, io, json, os, sys, textwrap
sys.path.insert(0, "src")
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"; FONTB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
BG, INK, INK2 = (252, 252, 251), (11, 11, 11), (82, 81, 78)
S1, S2, GRID, SURF = "#2a78d6", "#eb6834", "#e6e5e1", "#fcfcfb"          # dataviz reference palette

SETS = {
 "objects": ["a photo of a golden retriever running on a beach", "a red double-decker bus on a city street",
             "a bowl of fresh strawberries on a wooden table", "a grey tabby cat sitting on a windowsill",
             "a yellow vintage car parked in front of a garage", "a brown horse standing in a green field",
             "a stack of pancakes with blueberries and syrup", "a wooden sailboat on a calm lake"],
 "scenes": ["a mountain lake at sunrise with mist over the water", "a narrow cobblestone street in an old European town",
            "a snowy forest path with tall pine trees", "a desert road under a clear blue sky",
            "a tropical beach with palm trees and turquoise water", "a city skyline at night reflected in a river",
            "a lavender field with a farmhouse in the distance", "a waterfall in a lush green canyon"],
 "composition": ["two red apples and one green pear on a white plate", "a blue mug to the left of a yellow book on a desk",
                 "a black dog and a white dog sitting side by side on grass", "three candles of different heights on a dark table",
                 "a small orange kitten inside a large cardboard box", "a bicycle leaning against a red brick wall under a green awning",
                 "a wooden chair next to a window with rain outside", "a stone bridge over a river with a red boat underneath"],
 "style": ["a portrait of an old fisherman in dramatic side lighting, black and white photograph",
           "macro photograph of a dew drop on a green leaf", "a foggy harbor at dawn, soft pastel light",
           "neon signs reflecting on a wet street at night", "a bowl of ramen shot from above, warm restaurant lighting",
           "silhouette of a cyclist against an orange sunset", "long exposure of a waterfall, silky smooth water",
           "golden hour photo of a wheat field"],
 "long": ["A weathered wooden fishing boat painted in faded blue and white rests on a pebble beach at low tide. Coils of rope and two orange buoys lie beside it, and a low stone breakwater runs along the right edge of the frame under an overcast sky.",
          "A cozy reading corner with a deep green velvet armchair, a small round side table holding a steaming cup of tea and an open hardcover book, and a tall brass floor lamp casting warm light over a shelf of well-worn paperbacks.",
          "An elderly farmer in a straw hat and a checked shirt carries a wicker basket of freshly picked tomatoes through a vegetable garden, with rows of staked plants behind him and a red barn in the distance under bright midday sun.",
          "A minimalist kitchen counter in pale oak with a matte black kettle, a ceramic bowl of lemons, and a single sprig of rosemary in a glass jar, lit by soft window light from the left."],
 "failures": ["a street sign that says OPEN", "a close-up portrait of a smiling woman with freckles",
              "a crowd of people at a concert", "a hand holding a coffee cup",
              "a red cube on top of a blue cube on top of a green cube", "a clock showing 3 o'clock",
              "seven birds sitting on a wire", "a cat wearing a hat riding a skateboard"],
}


def grid(items, cols, cell_w, out, cap_lines=2, pad=8, title=None):
    """items: [(PIL image, caption)] -> JPEG. Images scaled to cell_w wide, captions below."""
    f11, f12b = ImageFont.truetype(FONT, 11), ImageFont.truetype(FONTB, 12)
    lh = 14; cap_h = cap_lines * lh + 6
    rows = (len(items) + cols - 1) // cols
    heights = [max(int(im.height * cell_w / im.width) for im, _ in items[r*cols:(r+1)*cols]) for r in range(rows)]
    top = 26 if title else 0
    W = cols * (cell_w + pad) + pad; H = top + sum(h + cap_h + pad for h in heights) + pad
    canvas = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(canvas)
    if title: d.text((pad, 7), title, font=f12b, fill=INK)
    y = top + pad
    for r in range(rows):
        for c in range(cols):
            k = r * cols + c
            if k >= len(items): break
            im, cap = items[k]
            x = pad + c * (cell_w + pad); canvas.paste(im.resize((cell_w, int(im.height * cell_w / im.width)), Image.LANCZOS), (x, y))
            width = max(10, int(cell_w / 5.6)); all_lines = textwrap.wrap(cap, width=width); lines = all_lines[:cap_lines]
            if len(all_lines) > cap_lines: lines[-1] = lines[-1][:-1] + "…"
            for i, ln in enumerate(lines): d.text((x, y + heights[r] + 3 + i * lh), ln, font=f11, fill=INK2)
        y += heights[r] + cap_h + pad
    canvas.save(out, "JPEG", quality=84, optimize=True); print("wrote", out, canvas.size, f"{os.path.getsize(out)/1e3:.0f} KB", flush=True)


def make_images(run, cache, out, steps=20, grids_only=False):
    from tinydit.playground import Engine
    from safetensors.torch import load_file
    E = Engine(run, cache)
    dec = lambda u: Image.open(io.BytesIO(base64.b64decode(u.split(",")[1]))).convert("RGB")
    def gen(prompt, w=256, h=256, steps=steps, cfg=4.0, shift=2.8, seed=0):
        return dec(E.generate(prompt, width=w, height=h, steps=steps, cfg=cfg, shift=shift, seed=seed, n_images=1, trajectory=0)["images"][0])
    for name, prompts in SETS.items():
        grid([(gen(p, seed=11 + i), p) for i, p in enumerate(prompts)], 4, 224, os.path.join(out, f"grid_{name}.jpg"), cap_lines=4 if name == "long" else 2)
    P = "a lighthouse on a rocky coast at sunset"
    shapes = [(256, 256), (288, 224), (224, 288), (320, 208), (208, 320), (448, 256)]
    grid([(gen(P, w, h, seed=5), f"{w}x{h}" + ("  (not a training shape)" if (w, h) == (448, 256) else "")) for w, h in shapes],
         6, 150, os.path.join(out, "grid_aspect.jpg"), cap_lines=2, title=P)
    if grids_only: return
    snaps = [10000, 50000, 100000, 200000, 300000, 400000]
    prog = ["a red tractor parked next to a blue rowing boat on a sandy beach", "three green apples on a white plate next to a black coffee cup",
            "a lighthouse on a rocky cliff during a storm"]
    own = E.model.state_dict(); rows = []
    for st in snaps:
        sd = load_file(f"out/runs/{run}/ema_{st:07d}.safetensors"); E.model.load_state_dict({k: v.to(own[k].dtype) for k, v in sd.items()})
        for j, p in enumerate(prog): rows.append((j, st, gen(p, seed=100 + j)))
    E.reload()
    grid([(im, f"step {st//1000}k") for j in range(len(prog)) for (jj, st, im) in rows if jj == j], len(snaps), 140,
         os.path.join(out, "progress.jpg"), cap_lines=1, title="same prompt and seed, EMA snapshots over training")
    r = E.generate("a wooden rowing boat pulled up on a pebble shore", 320, 208, steps=20, cfg=4.0, shift=2.8, seed=7, trajectory=8)
    items = [(dec(f["x_t"]), f"x_t  step {f['step']}  t={f['t']:.2f}") for f in r["trajectory"]] + [(dec(f["x1_hat"]), f"prediction  t={f['t']:.2f}") for f in r["trajectory"]]
    grid(items, 8, 140, os.path.join(out, "trajectory.jpg"), cap_lines=1,
         title="top: the noisy state the sampler is in · bottom: what the model thinks the final image is, at each of 8 of the 20 steps")
    P2 = "a blue ceramic teapot and two cups on a wooden tray"
    cfgs = [(20, 2.8, 4.0, "20 steps, shift 2.8 (training setting)"), (20, 1.0, 4.0, "20 steps, no shift"), (8, 2.8, 4.0, "8 steps, shift 2.8"),
            (50, 2.8, 4.0, "50 steps, shift 2.8"), (20, 2.8, 1.0, "cfg 1 (no guidance)"), (20, 2.8, 7.0, "cfg 7")]
    grid([(gen(P2, 256, 256, steps=st, cfg=cfg, shift=sh, seed=3), lab) for st, sh, cfg, lab in cfgs], 6, 150,
         os.path.join(out, "grid_sampling.jpg"), cap_lines=2, title=P2)


def make_charts(run, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.ticker as mt
    rows = [json.loads(l) for l in open(f"out/runs/{run}/metrics.jsonl") if l.strip()]
    ev = [x for x in rows if "fid" in x]; va = [x for x in rows if "val_loss" in x]; tr = [x for x in rows if "loss_mse" in x]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.titlecolor": "#0b0b0b", "axes.titleweight": "bold", "axes.titlesize": 10})
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.4), facecolor=SURF)
    def panel(ax, series, title, pct=False, label_last=None):
        ax.set_facecolor(SURF)
        for si, (xs, ys, col, lab) in enumerate(series):
            ax.plot(xs, ys, color=col, lw=2, solid_capstyle="round", solid_joinstyle="round", label=lab)
            ax.plot(xs[-1], ys[-1], "o", ms=6, color=col, markeredgecolor=SURF, markeredgewidth=2)
            if label_last is None or si == label_last:
                txt = f"{ys[-1]*100:.0f}%" if pct else (f"{ys[-1]:.3f}" if ys[-1] < 10 else f"{ys[-1]:.1f}")
                ax.annotate(txt, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5, color="#0b0b0b")
        ax.axvline(300, color=GRID, lw=1)
        # the vertical hairline marks the start of the learning-rate decay (explained in the README caption)
        ax.set_title(title, loc="left"); ax.grid(axis="y", color=GRID, lw=1); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.set_xlim(0, 440); ax.set_xlabel("step (k)")
        if pct: ax.yaxis.set_major_formatter(mt.PercentFormatter(1.0, decimals=0))
        if len(series) > 1: ax.legend(frameon=False, fontsize=8, loc="lower left" if "loss" in title.lower() else "lower right")
    ks = [x["step"] / 1000 for x in ev]
    panel(axes[0, 0], [(ks, [x["fid"] for x in ev], S1, None)], "FID  (lower is better)")
    panel(axes[0, 1], [(ks, [x["fd_dino"] for x in ev], S1, None)], "FD-DINOv2  (lower is better)")
    panel(axes[0, 2], [(ks, [x["obj_acc"] for x in ev], S1, None)], "Object accuracy  (80 COCO classes × 4)", pct=True)
    panel(axes[1, 0], [(ks, [x["pickscore_unseen"] for x in ev], S1, "held-out captions"), (ks, [x["pickscore_novel"] for x in ev], S2, "novel prompts")], "PickScore  (higher is better)")
    panel(axes[1, 1], [(ks, [x["hpsv2_unseen"] for x in ev], S1, "held-out captions"), (ks, [x["hpsv2_novel"] for x in ev], S2, "novel prompts")], "HPSv2.1  (higher is better)")
    w = {}
    for x in tr: w.setdefault(x["step"] // 5000, []).append(x["loss_mse"])
    tk = sorted(w)
    panel(axes[1, 2], [([k * 5 + 2.5 for k in tk], [float(np.mean(w[k])) for k in tk], S2, "training (mse, 5k avg)"),
                       ([x["step"] / 1000 for x in va], [x["val_loss"] for x in va], S1, "held-out")], "Flow-matching loss", label_last=1)
    axes[1, 2].set_ylim(0.72, 0.9)
    fig.subplots_adjust(left=0.05, right=0.985, top=0.94, bottom=0.09, wspace=0.28, hspace=0.5)
    p = os.path.join(out, "metrics.png"); fig.savefig(p, dpi=110, facecolor=SURF); print("wrote", p, f"{os.path.getsize(p)/1e3:.0f} KB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--run", default="run1"); ap.add_argument("--cache", default="out/cache/run1")
    ap.add_argument("--out", default="docs"); ap.add_argument("--charts-only", action="store_true")
    ap.add_argument("--steps", type=int, default=20); ap.add_argument("--grids-only", action="store_true"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if not a.charts_only: make_images(a.run, a.cache, a.out, a.steps, a.grids_only)
    if not a.grids_only: make_charts(a.run, a.out)
    print("TOTAL docs size:", round(sum(os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out)) / 1e6, 2), "MB")

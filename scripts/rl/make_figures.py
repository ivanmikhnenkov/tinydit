"""Render the four static figures for docs/rl/README.md from the files under out/ (CPU only, no GPU).

  .venv/bin/python scripts/rl/make_figures.py          # writes docs/rl/{passk.png,raccoon.jpg,controls.png,hack.png}

Sources: out/passk/summary.json, out/passk/grids/odd01_{base,v10_192}.jpg,
out/runs/{diag_beta100,diag_flip,diag_shuffle}/metrics.jsonl, out/runs/rl2_dgpo_v5/{metrics,judges}.jsonl.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

OUT = "docs/rl"; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
                     "axes.grid": True, "grid.alpha": 0.3, "legend.frameon": False})
C = {"base": "#444444", "v5_221": "#1f77b4", "v10_192": "#d62728", "v5_308": "#2ca02c",
     "normal": "#1f77b4", "flip": "#d62728", "shuffle": "#ff7f0e"}

def evals(run):
    return [json.loads(l) for l in open(f"out/runs/{run}/metrics.jsonl") if '"eval": true' in l]

# 1. pass@k curves, both judges, base + three checkpoints (125 prompts x 64 seeds)
s = json.load(open("out/passk/summary.json"))
label = {"base": "base", "v5_221": "v5 r221 (full reward)", "v10_192": "v10 r192 (alignment only)", "v5_308": "v5 r308"}
fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
for ax, (judge, title) in zip(axes, [("qwen35_9b", "independent judge (Qwen3.5-9B)"), ("qwen3vl8b", "reward model (Qwen3-VL-8B)")]):
    for name in ["base", "v5_221", "v10_192", "v5_308"]:
        p = s["summary"][judge][name]["pass"]; ks = sorted(int(k) for k in p)
        ax.plot(ks, [p[str(k)] for k in ks], marker="o", ms=3.5, lw=1.6, color=C[name], label=label[name])
    ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8, 16, 32, 64]); ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64])
    ax.set_xlabel("k (samples per prompt)"); ax.set_title(title, fontsize=11); ax.set_ylim(0.75, 1.0)
axes[0].set_ylabel("pass@k (P(yes) >= 0.5)"); axes[0].legend(fontsize=9, loc="lower right")
fig.tight_layout(); fig.savefig(f"{OUT}/passk.png"); plt.close(fig)

# 2. raccoon grid: 16 of 64 seeds, base vs v10 r192, 8B P(yes) under each cell (from out/passk/grids)
cnt = {j: s["summary"][j] for j in s["summary"]}; pi = [p["id"] for p in s["prompts"]].index("odd01")
panels = [("base", "base"), ("v10_192", "after RL (v10 r192)")]
imgs = [Image.open(f"out/passk/grids/odd01_{n}.jpg").convert("RGB") for n, _ in panels]
W, H, gap, head = imgs[0].width, imgs[0].height, 24, 44
canvas = Image.new("RGB", (2 * W + gap, H + head), "white"); d = ImageDraw.Draw(canvas)
try: font = ImageFont.truetype("DejaVuSans.ttf", 15)
except OSError: font = ImageFont.load_default()
for i, ((n, t), im) in enumerate(zip(panels, imgs)):
    x = i * (W + gap); canvas.paste(im, (x, head))
    c8, c9 = cnt["qwen3vl8b"][n]["c_per_prompt"][pi], cnt["qwen35_9b"][n]["c_per_prompt"][pi]
    d.text((x + 2, 4), f"{t}: pass {c8}/64 (8B), {c9}/64 (9B)", font=font, fill=(30, 30, 30))
    d.text((x + 2, 24), "\"a raccoon driving a car\", seeds 0-15 of 64, 8B P(yes) under each", font=font, fill=(110, 110, 110))
canvas.save(f"{OUT}/raccoon.jpg", quality=90)

# 3. advantage controls: held-out sharpness and 8B alignment vs round
runs = [("diag_beta100", "normal advantages", "normal"), ("diag_flip", "advantages x -1 (flipped)", "flip"), ("diag_shuffle", "advantages shuffled in group", "shuffle")]
fig, axes = plt.subplots(1, 2, figsize=(8, 3.3))
for run, lab, key in runs:
    e = evals(run); r = [x["round"] for x in e]
    axes[0].plot(r, [x["eval_px_sharp"] for x in e], marker="o", ms=3.5, lw=1.6, color=C[key], label=lab)
    axes[1].plot(r, [x["eval_align"] for x in e], marker="o", ms=3.5, lw=1.6, color=C[key], label=lab)
axes[0].set_ylabel("sharpness (Laplacian variance)"); axes[1].set_ylabel("alignment, 8B P(yes)")
for ax in axes: ax.set_xlabel("round (beta 100, 0.34 GPU-h total)")
axes[0].legend(fontsize=9); fig.tight_layout(); fig.savefig(f"{OUT}/controls.png"); plt.close(fig)

# 4. the realism-probe hack in v5: probe reward up, FD-DINOv2 up (worse), 9B "real photograph?" flat
e = evals("rl2_dgpo_v5"); j = [json.loads(l) for l in open("out/runs/rl2_dgpo_v5/judges.jsonl")]
h = [x["gpu_hours"] for x in e]; hj = [x["gpu_hours"] for x in j]; pick = next(x["gpu_hours"] for x in e if x["round"] == 221)
fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
axes[0].plot(h, [x["eval_real"] for x in e], color="#1f77b4", lw=1.6, label="DINOv2 realism probe (reward term)")
axes[0].plot(hj, [x["j_real"] for x in j], color="#444444", lw=1.6, label="9B judge: \"real photograph?\"")
axes[0].set_ylim(0.55, 0.75); axes[0].set_ylabel("P(real)"); axes[0].legend(fontsize=8, loc="upper left")
axes[1].plot(h, [x["eval_fd_dino"] for x in e], color="#d62728", lw=1.6); axes[1].set_ylabel("FD-DINOv2 (lower is better)")
axes[2].plot(h, [x["eval_px_sat"] for x in e], color="#ff7f0e", lw=1.6); axes[2].set_ylabel("mean saturation")
for ax in axes:
    ax.set_xlabel("GPU-hours in v5"); ax.axvline(pick, color="#888888", ls="--", lw=1)
axes[1].set_title("dashed line: r221, the checkpoint picked", fontsize=9, color="#555555")
fig.tight_layout(); fig.savefig(f"{OUT}/hack.png"); plt.close(fig)
print("wrote", sorted(os.listdir(OUT)))

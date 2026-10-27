"""Train the real-vs-render probe on DINOv2 features: real = out/data/refs/real_256.npz (COCO val 2017 + Pexels/COCO refs),
render = out/data/refs/flux_256.npz (FLUX-Reason Aesthetics). Holds out 20 % for the AUROC. Saves out/models/judges/dino_probe.pt.
    .venv/bin/python scripts/rl/train_dino_probe.py"""
import json, os, sys
sys.path.insert(0, "src")
import numpy as np, torch
from tinydit import metrics
real = np.load("out/data/refs/real_256.npz")["images"]; flux = np.load("out/data/refs/flux_256.npz")["images"]
# hack-augmented real photos join the "not a clean photograph" class, so the probe also rejects the cheap exits of a preference
# reward: boosted saturation / contrast, text stamped on the image, 2x2 tiling, and pure noise (validate_reward.py showed the
# plain probe rating a saturated copy 0.86 and noise 0.91).
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
rng = np.random.default_rng(0); font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
words = ["a photo of", "the cat", "sunset beach", "red car", "golden retriever", "mountain lake", "fresh bread", "city street"]
def aug(im, kind):
    pil = Image.fromarray(im.transpose(1, 2, 0))
    if kind == "sat": return np.asarray(ImageEnhance.Color(ImageEnhance.Contrast(pil).enhance(rng.uniform(1.3, 1.8))).enhance(rng.uniform(1.6, 2.4))).transpose(2, 0, 1)
    if kind == "text":
        t = pil.copy(); ImageDraw.Draw(t).text((int(rng.integers(4, 60)), int(rng.integers(4, 200))), " ".join(rng.choice(words, 2)), font=font, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0)); return np.asarray(t).transpose(2, 0, 1)
    if kind == "desat": return np.asarray(ImageEnhance.Color(pil).enhance(rng.uniform(0.2, 0.6))).transpose(2, 0, 1)
    if kind == "lowcon": return np.asarray(ImageEnhance.Contrast(pil).enhance(rng.uniform(0.45, 0.7))).transpose(2, 0, 1)
    if kind == "blur":
        from PIL import ImageFilter; return np.asarray(pil.filter(ImageFilter.GaussianBlur(rng.uniform(1.2, 2.5)))).transpose(2, 0, 1)
    if kind == "tile":
        sm = pil.resize((128, 128)); tl = Image.new("RGB", (256, 256)); [tl.paste(sm, (x, y)) for x in (0, 128) for y in (0, 128)]; return np.asarray(tl).transpose(2, 0, 1)
sel = rng.permutation(len(real))
hacks = np.stack([aug(real[i], "sat") for i in sel[:1000]] + [aug(real[i], "desat") for i in sel[1000:2000]] + [aug(real[i], "lowcon") for i in sel[2000:2500]]
                 + [aug(real[i], "blur") for i in sel[2500:3000]] + [aug(real[i], "text") for i in sel[3000:3600]] + [aug(real[i], "tile") for i in sel[3600:3900]]
                 + [rng.integers(0, 256, real[0].shape, dtype=np.uint8) for _ in range(200)])
feats = [metrics._dino_feats(real, "cuda"), metrics._dino_feats(flux, "cuda"), metrics._dino_feats(hacks, "cuda")]
y = [1] * len(real) + [0] * len(flux) + [0] * len(hacks)
print(f"  {len(real)} real, {len(flux)} FLUX renders, {len(hacks)} hack-augmented reals (1000 saturated, 1000 desaturated, 500 low-contrast, 500 blurred, 600 text, 300 tiled, 200 noise)", flush=True)
X = np.concatenate(feats); y = np.array(y, dtype=np.float32)
rng = np.random.default_rng(0); idx = rng.permutation(len(y)); te, tr = idx[:len(y) // 5], idx[len(y) // 5:]
Xtr = torch.tensor(X[tr]).float(); ytr = torch.tensor(y[tr]); mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
w = torch.zeros(X.shape[1], requires_grad=True); bias = torch.zeros(1, requires_grad=True); opt = torch.optim.LBFGS([w, bias], max_iter=300)
def closure():
    opt.zero_grad(); z = ((Xtr - mu) / sd) @ w + bias
    l = torch.nn.functional.binary_cross_entropy_with_logits(z, ytr) + 1e-2 * (w ** 2).sum(); l.backward(); return l
opt.step(closure)
with torch.no_grad():
    p = torch.sigmoid(((torch.tensor(X[te]).float() - mu) / sd) @ w + bias).numpy()
pos, neg = p[y[te] == 1], p[y[te] == 0]; auroc = (pos[:, None] > neg[None, :]).mean()
kind = np.array(["real"] * len(real) + ["flux"] * len(flux) + ["hack"] * len(hacks))[te]
for k in ("flux", "hack"):
    negk = p[(y[te] == 0) & (kind == k)]; print(f"  held-out AUROC real vs {k}: {(pos[:, None] > negk[None, :]).mean():.4f}", flush=True)
torch.save({"mu": mu.detach(), "sd": sd.detach(), "w": w.detach(), "b": bias.detach(), "n_train": int(len(tr)), "auroc_heldout": float(auroc)}, "out/models/judges/dino_probe.pt")
print(f"trained on {len(tr)} ({int(y[tr].sum())} real / {int((1-y[tr]).sum())} render), held-out AUROC {auroc:.4f} on {len(te)} -> out/models/judges/dino_probe.pt", flush=True)
metrics.unload()

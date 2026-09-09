"""Sampling: Euler integration of the learned velocity with classifier-free guidance, plus
labelled comparison grids. One sampler, used by training snapshots, evaluation and the CLI.

The step schedule is the same SD3/FLUX shift used at training time (schedule.shift): more
steps near t=0, where the global layout is decided. Any latent grid shape works.
"""
from __future__ import annotations
import argparse, json, os
import numpy as np, torch
from . import ae as AE, text as T, schedule
from .model import TinyDiT, CONFIGS


def load_model(ckpt_path, latent_ch, config, use_ema=True, device="cuda"):
    """Loads either a full training checkpoint (.pt with model/ema) or a bf16 EMA snapshot (.safetensors)."""
    m = TinyDiT(latent_ch=latent_ch, ctx_dim=T.D_MODEL, **CONFIGS[config]).to(device)
    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = {k: v.to(device) for k, v in load_file(ckpt_path).items()}
        m.load_state_dict({k: v.to(m.state_dict()[k].dtype) for k, v in sd.items()}); m.eval()
        return m, 0
    ck = torch.load(ckpt_path, map_location=device)
    sd = ck["model"]
    if use_ema and "ema" in ck:
        ema = {k: v.to(device) for k, v in ck["ema"].items()}
        sd = {k: (ema[k].to(v.dtype) if k in ema else v) for k, v in sd.items()}
    m.load_state_dict(sd); m.eval()
    return m, ck.get("step", 0)


def pad_ctx(e, m, L):
    """Right-pad a (B,l,D) context and its (B,l) mask to length L with masked zeros."""
    if e.shape[1] == L: return e, m
    B, l, D = e.shape
    e2 = torch.zeros(B, L, D, device=e.device, dtype=e.dtype); e2[:, :l] = e
    m2 = torch.zeros(B, L, device=m.device, dtype=torch.bool); m2[:, :l] = m
    return e2, m2


@torch.no_grad()
def generate(model, ctx, msk, null_ctx, null_msk, latent_ch, H, W, steps=20, cfg=4.0,
             seed=0, shift=2.8, device="cuda", chunk=64):
    """ctx: (N,L,768) text | H,W: latent grid in pixels/8 (e.g. 32,32) -> whitened latents (N,C,H,W).
    Chunked because CFG doubles the batch."""
    if ctx.shape[0] > chunk:
        outs = []
        for i in range(0, ctx.shape[0], chunk):
            outs.append(generate(model, ctx[i:i+chunk], msk[i:i+chunk], null_ctx, null_msk,
                                 latent_ch, H, W, steps, cfg, seed + i, shift, device, chunk))
        return torch.cat(outs, 0)
    n, L = ctx.shape[0], ctx.shape[1]
    null_ctx, null_msk = pad_ctx(null_ctx, null_msk, L)
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(n, latent_ch, H, W, device=device, generator=g)
    ts = schedule.shift(steps, shift, device=device) if shift and shift != 1 else schedule.uniform(steps, device=device)
    for i in range(steps):
        t = ts[i].expand(n); dt = ts[i + 1] - ts[i]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = model(torch.cat([x, x]), torch.cat([t, t]),
                      torch.cat([ctx, null_ctx.expand(n, -1, -1)]),
                      torch.cat([msk, null_msk.expand(n, -1)])).float()
        vc, vu = v.chunk(2)
        x = x + (vu + cfg * (vc - vu)) * dt
    return x


def decode_chunked(aem, z, mean, std, chunk=16):
    """whitened latents -> uint8 images (N,3,H,W). Chunked: decoding 128 latents at once allocates ~8 GB."""
    outs = []
    for i in range(0, z.shape[0], chunk):
        y = AE.decode(aem, (z[i:i+chunk] * std + mean).float())
        outs.append(((y + 1) * 127.5).clamp(0, 255).byte().cpu().numpy())
        del y
    torch.cuda.empty_cache()
    return np.concatenate(outs, 0)


FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _font(path, size):
    from PIL import ImageFont
    try: return ImageFont.truetype(path, size)
    except OSError: return ImageFont.load_default()


def _wrap(draw, text, font, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=font) <= width: cur = t
        else:
            if cur: lines.append(cur)
            cur = w
    if cur: lines.append(cur)
    return lines


def render_sections(secs, out_path, step, cfg, nsteps, cols=4, pad=10, cap_h=68):
    """secs: list of (label, prompts, imgs_u8 (n,3,H,W) [, per_image_scores [, all_scores]]).
    Sections may have different image shapes; each section lays out its own grid."""
    from PIL import Image, ImageDraw
    fs, fb, ft = _font(FONT, 13), _font(FONT_B, 20), _font(FONT_B, 16)
    hdr = 42
    dims = []
    for sec in secs:
        H, W = sec[2].shape[-2], sec[2].shape[-1]
        rows = (len(sec[1]) + cols - 1) // cols
        dims.append((H, W, rows))
    total_w = max(cols * (W + pad) + pad for H, W, r in dims)
    total_h = sum(hdr + r * (H + cap_h + pad) for H, W, r in dims) + 56
    canvas = Image.new("RGB", (total_w, total_h), (247, 248, 248))
    d = ImageDraw.Draw(canvas)
    d.text((pad, 14), f"tinydit · step {step:,} · cfg {cfg} · {nsteps} steps", font=ft, fill=(20, 23, 26))
    y = 50
    for sec, (H, W, rows) in zip(secs, dims):
        label, prompts, imgs = sec[0], sec[1], sec[2]
        scores = sec[3] if len(sec) > 3 else None
        allsc = sec[4] if len(sec) > 4 else scores
        d.rectangle([pad, y, total_w - pad, y + 28], fill=(214, 0, 110))
        right = f"CLIP {sum(allsc)/len(allsc):.3f}  ·  n={len(allsc)}" if allsc else ""
        rw = d.textlength(right, font=fb) if right else 0
        if right: d.text((total_w - pad - 12 - rw, y + 5), right, font=fb, fill=(255, 255, 255))
        lab = label
        while d.textlength(lab, font=fb) > total_w - 2 * pad - 24 - rw and len(lab) > 4: lab = lab[:-2]
        if lab != label: lab = lab.rstrip() + "…"
        d.text((pad + 10, y + 5), lab, font=fb, fill=(255, 255, 255))
        y += hdr
        cellw, cellh = W + pad, H + cap_h + pad
        for k, pr in enumerate(prompts):
            r, c = divmod(k, cols)
            x0, y0 = pad + c * cellw, y + r * cellh
            canvas.paste(Image.fromarray(imgs[k].transpose(1, 2, 0)), (x0, y0))
            ty = y0 + H + 4
            if scores:
                sc = scores[k]
                col = (15, 123, 108) if sc >= 0.28 else ((183, 121, 31) if sc >= 0.22 else (194, 24, 91))
                d.text((x0 + 2, ty), f"CLIP {sc:.3f}", font=fs, fill=col); ty += 16
            for ln in _wrap(d, pr, fs, W - 4)[:3]:
                d.text((x0 + 2, ty), ln, font=fs, fill=(60, 70, 76)); ty += 16
        y += rows * cellh + 8
    canvas.save(out_path)


def write_eval_assets(secs, outdir, step, cfg, nsteps):
    """Each generated image as its own PNG plus a manifest, for the dashboard filmstrip."""
    from PIL import Image
    d = os.path.join(outdir, f"{step}"); os.makedirs(d, exist_ok=True)
    man = {"step": step, "cfg": cfg, "steps": nsteps, "sections": []}
    for si, sec in enumerate(secs):
        label, prompts, imgs = sec[0], sec[1], sec[2]
        scores = sec[3] if len(sec) > 3 else None
        items = []
        for k in range(len(prompts)):
            fn = f"{si}_{k}.png"
            Image.fromarray(imgs[k].transpose(1, 2, 0)).save(os.path.join(d, fn))
            items.append({"file": fn, "prompt": prompts[k], "clip": scores[k] if scores else None,
                          "shape": f"{imgs.shape[-1]}x{imgs.shape[-2]}"})
        man["sections"].append({"label": label, "n": len(prompts),
                                "clip": (sum(sec[4]) / len(sec[4])) if len(sec) > 4 and sec[4] else None, "items": items})
    json.dump(man, open(os.path.join(d, "manifest.json"), "w"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--sets", nargs="+", required=True, help="LABEL=path.txt, one per section")
    p.add_argument("--out", required=True)
    p.add_argument("--cache", default="out/cache/run1"); p.add_argument("--models", default="out/models")
    p.add_argument("--ae", default="flux2"); p.add_argument("--config", default="run1")
    p.add_argument("--shape", default="256x256", help="WxH pixels, multiple of 16")
    p.add_argument("--steps", type=int, default=20); p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--shift", type=float, default=2.8); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cols", type=int, default=4); p.add_argument("--raw", action="store_true")
    p.add_argument("--no-clip", action="store_true")
    a = p.parse_args()
    sets = []
    for spec in a.sets:
        label, _, path = spec.partition("=")
        sets.append((label, [l.strip() for l in open(path) if l.strip()]))
    lat_ch = AE.SPECS[a.ae]["ch"]
    st = json.load(open(os.path.join(a.cache, "stats.json")))
    mean = torch.tensor(st["mean"]).view(1, -1, 1, 1).cuda(); std = torch.tensor(st["std"]).view(1, -1, 1, 1).cuda()
    W, H = (int(v) for v in a.shape.lower().split("x"))
    tok, enc = T.load(os.path.join(a.models, "t5"))
    embs = [T.embed_mixed(tok, enc, ps, [len(x.split()) > 30 for x in ps]) for _, ps in sets]
    nctx, nmsk = T.embed(tok, enc, [""], T.MAX_SHORT)
    del enc; torch.cuda.empty_cache()
    model, step = load_model(a.ckpt, lat_ch, a.config, use_ema=not a.raw)
    zs = [generate(model, c.float(), m, nctx.float(), nmsk, lat_ch, H // 8, W // 8, a.steps, a.cfg, a.seed, a.shift)
          for c, m in embs]
    del model; torch.cuda.empty_cache()
    aem = AE.load(a.ae, os.path.join(a.models, a.ae))
    secs = []
    for (label, ps), z in zip(sets, zs):
        img = decode_chunked(aem, z, mean, std)
        sc = None
        if not a.no_clip:
            from . import metrics
            sc = metrics.clip_score(img, ps)
        secs.append((label, ps, img) if sc is None else (label, ps, img, sc, sc))
        print(f"  {label}: {len(ps)} prompts" + (f" | CLIP {sum(sc)/len(sc):.4f}" if sc else ""))
    render_sections(secs, a.out, step, a.cfg, a.steps, cols=a.cols)
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()

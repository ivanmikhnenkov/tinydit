"""Generate from arbitrary prompts using a saved checkpoint.

The training snapshot grid uses *training* captions, which tracks optimisation but
cannot show generalisation. This tool takes any prompt list — held-out val captions
or hand-written ones — so seen and unseen can be compared at the same checkpoint.
"""
from __future__ import annotations
import argparse, json, os
import numpy as np, torch
from . import ae as AE, text as T
from .model import TinyDiT, CONFIGS


def load_model(ckpt_path, latent_ch, config, use_ema=True, device="cuda"):
    ck = torch.load(ckpt_path, map_location=device)
    m = TinyDiT(latent_ch=latent_ch, ctx_dim=T.D_MODEL, **CONFIGS[config]).to(device)
    sd = ck["model"]
    if use_ema and "ema" in ck:
        ema = {k: v.to(device) for k, v in ck["ema"].items()}
        sd = {k: (ema[k].to(v.dtype) if k in ema else v) for k, v in sd.items()}
    m.load_state_dict(sd); m.eval()
    return m, ck.get("step", 0)


@torch.no_grad()
def generate(model, ctx, msk, null_ctx, null_msk, latent_ch, res, steps=50, cfg=4.0,
             seed=0, device="cuda", chunk=32):
    """Chunked because CFG doubles the batch: 128 prompts would be a 256-image forward."""
    if ctx.shape[0] > chunk:
        outs = []
        for i in range(0, ctx.shape[0], chunk):
            outs.append(generate(model, ctx[i:i+chunk],
                                 msk[i:i+chunk] if msk is not None else None,
                                 null_ctx, null_msk, latent_ch, res, steps, cfg,
                                 seed + i, device, chunk))
        return torch.cat(outs, 0)
    g = torch.Generator(device=device).manual_seed(seed)
    n = ctx.shape[0]
    x = torch.randn(n, latent_ch, res // 8, res // 8, device=device, generator=g)
    ts = torch.linspace(0, 1, steps + 1, device=device)
    for i in range(steps):
        t = ts[i].expand(n); dt = ts[i + 1] - ts[i]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = model(torch.cat([x, x]), torch.cat([t, t]),
                      torch.cat([ctx, null_ctx.expand_as(ctx)]),
                      torch.cat([msk, null_msk.expand_as(msk)])).float()
        vc, vu = v.chunk(2)
        x = x + (vu + cfg * (vc - vu)) * dt
    return x


def decode_chunked(aem, z, mean, std, chunk=16):
    """Decoding 128 latents at 256^2 in one call allocates ~8 GB; go in chunks."""
    import numpy as np
    outs = []
    for i in range(0, z.shape[0], chunk):
        y = AE.decode(aem, (z[i:i+chunk] * std + mean).float())
        outs.append(((y + 1) * 127.5).clamp(0, 255).byte().cpu().numpy())
        del y; torch.cuda.empty_cache()
    return np.concatenate(outs, 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--sets", nargs="+", required=True,
                   help="LABEL=path.txt, one per section (e.g. Seen=out/p_seen.txt)")
    p.add_argument("--out", required=True)
    p.add_argument("--dataset", default="coco"); p.add_argument("--ae", default="flux2")
    p.add_argument("--cache", default="out/cache"); p.add_argument("--models", default="out/models")
    p.add_argument("--config", default="base")
    p.add_argument("--res", type=int, default=256)
    p.add_argument("--steps", type=int, default=50); p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--max-len", type=int, default=32)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--raw", action="store_true", help="use raw weights instead of EMA")
    p.add_argument("--no-clip", action="store_true", help="skip CLIP prompt-adherence scoring")
    p.add_argument("--render-n", type=int, default=12,
                   help="images drawn per set; CLIP still scores the whole set")
    a = p.parse_args()

    sets = []
    for spec in a.sets:
        label, _, path = spec.partition("=")
        sets.append((label, [l.strip() for l in open(path) if l.strip()]))
    lat_ch = AE.SPECS[a.ae]["ch"]
    stats = json.load(open(os.path.join(a.cache, a.dataset, f"lat_{a.ae}_{a.res}.json")))
    mean = torch.tensor(stats["mean"]).view(1, -1, 1, 1).cuda()
    std = torch.tensor(stats["std"]).view(1, -1, 1, 1).cuda()

    tok, enc = T.load(os.path.join(a.models, "t5"))
    embs = [T.embed(tok, enc, ps, a.max_len) for _, ps in sets]
    nctx, nmsk = T.embed(tok, enc, [""], a.max_len)
    del enc; torch.cuda.empty_cache()

    model, step = load_model(a.ckpt, lat_ch, a.config, use_ema=not a.raw)
    zs = [generate(model, c.float(), m, nctx.float(), nmsk, lat_ch, a.res, a.steps, a.cfg, a.seed)
          for c, m in embs]
    del model; torch.cuda.empty_cache()

    aem = AE.load(a.ae, os.path.join(a.models, a.ae))
    secs, summary = [], {}
    for (label, ps), z in zip(sets, zs):
        img = decode_chunked(aem, z, mean, std)
        sc = None
        if not a.no_clip:
            try:
                from . import clipscore
                per, mean_s = clipscore.score(img, ps, os.path.join(a.models, "clip"))
                sc = per; summary[label.split(" ")[0]] = round(mean_s, 4)
            except Exception as ex:
                print(f"  (clip skipped: {str(ex)[:70]})")
        k = min(a.render_n, len(ps))
        secs.append((label, ps[:k], img[:k])
                    if sc is None else
                    (f"{label}  [{k} of {len(ps)} shown]", ps[:k], img[:k], sc[:k], sc))
        print(f"  {label}: {len(ps)} prompts" + (f" | CLIP {summary.get(label.split(' ')[0]):.4f}" if sc else ""))
    if summary:
        json.dump(dict(step=step, clip=summary),
                  open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)
    render_sections(secs, a.out, step, a.cfg, a.steps, cols=a.cols)


FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _wrap(draw, text, font, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=font) <= width:
            cur = t
        else:
            if cur: lines.append(cur)
            cur = w
    if cur: lines.append(cur)
    return lines


def render_sections(secs, out_path, step, cfg, nsteps, cols=4, pad=10, cap_h=68):
    """secs: (label, prompts, imgs) or (label, prompts, imgs, clip_scores).
    Caption sits under its own image; CLIP score is shown beside it."""
    from PIL import Image, ImageDraw, ImageFont
    f = ImageFont.truetype(FONT, 13)
    fb = ImageFont.truetype(FONT_B, 20)
    fs = ImageFont.truetype(FONT, 13)
    H = W = secs[0][2].shape[-1]
    cellw, cellh = W + pad, H + cap_h + pad
    hdr = 42
    rows_per = [(len(sec[1]) + cols - 1) // cols for sec in secs]
    total_h = sum(hdr + r * cellh for r in rows_per) + 56
    total_w = cols * cellw + pad
    canvas = Image.new("RGB", (total_w, total_h), (247, 248, 248))
    d = ImageDraw.Draw(canvas)
    d.text((pad, 14), f"tinydit · step {step:,} · cfg {cfg} · {nsteps} ODE steps",
           font=ImageFont.truetype(FONT_B, 16), fill=(20, 23, 26))
    y = 50
    for si, sec in enumerate(secs):
        label, prompts, imgs = sec[0], sec[1], sec[2]
        scores = sec[3] if len(sec) > 3 else None
        d.rectangle([pad, y, total_w - pad, y + 28], fill=(214, 0, 110))
        allsc = sec[4] if len(sec) > 4 else scores
        right = f"CLIP {sum(allsc)/len(allsc):.3f}  ·  n={len(allsc)}" if allsc else ""
        if right:
            rw = d.textlength(right, font=fb)
            d.text((total_w - pad - 12 - rw, y + 5), right, font=fb, fill=(255, 255, 255))
        else:
            rw = 0
        # truncate the label if it would run into the right-hand text
        avail = total_w - 2*pad - 24 - rw
        lab = label
        while d.textlength(lab, font=fb) > avail and len(lab) > 4:
            lab = lab[:-2]
        if lab != label: lab = lab.rstrip() + "…"
        d.text((pad + 10, y + 5), lab, font=fb, fill=(255, 255, 255))
        y += hdr
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
        y += rows_per[si] * cellh + 8
    canvas.save(out_path)
    print(f"  wrote {out_path}  ({total_w}x{total_h})")


if __name__ == "__main__":
    main()


def write_eval_assets(secs, outdir, step, cfg, nsteps):
    """Write each generated image as its own file plus a manifest.

    The composite PNG is fine for a glance but you cannot open or save one panel from it.
    Individual webp + JSON lets the dashboard lay out the same grid while every image
    stays a real, downloadable file."""
    import os, json
    from PIL import Image
    d = os.path.join(outdir, str(step))
    os.makedirs(d, exist_ok=True)
    man = {"step": step, "cfg": cfg, "ode_steps": nsteps, "sets": []}
    for sec in secs:
        label, prompts, imgs = sec[0], sec[1], sec[2]
        shown = sec[3] if len(sec) > 3 else None
        allsc = sec[4] if len(sec) > 4 else shown
        slug = "".join(c if c.isalnum() else "_" for c in label.split(" ")[0]).lower()
        items = []
        for i, pr in enumerate(prompts):
            fn_ = f"{slug}_{i:02d}.webp"
            Image.fromarray(imgs[i].transpose(1, 2, 0)).save(os.path.join(d, fn_), "WEBP", quality=92)
            items.append({"file": fn_, "prompt": pr,
                          "clip": round(float(shown[i]), 4) if shown else None})
        man["sets"].append({"label": label, "slug": slug, "items": items,
                            "clip": round(float(sum(allsc)/len(allsc)), 4) if allsc else None,
                            "n": len(allsc) if allsc else len(prompts)})
    json.dump(man, open(os.path.join(d, "manifest.json"), "w"), indent=1)
    return d

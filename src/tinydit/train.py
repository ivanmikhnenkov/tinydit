"""Training loop.

Data comes from data.BucketCache (one aspect bucket per batch, captions encoded live by T5, latents
prefetched from memmaps). torch.compile is on by default: one static graph per bucket shape.

Checkpoints: `ckpt_last.pt` (fp32 weights + optimizer + EMA + step, for exact resume; rotated) and
`ema_<step>.safetensors` (bf16 EMA weights only, ~420 MB, for eval / inference / post-hoc EMA).
Metrics stream to metrics.jsonl (keys in CONTRACTS.md) for dashboard.html.
"""
from __future__ import annotations
import argparse, json, os, random, signal, time
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.abspath("out/inductor_cache"))
import numpy as np, torch, torch._dynamo
from . import ae as AE, flow, text as T
from .data import BucketCache
from .model import TinyDiT, CONFIGS
STOP = False
def _sig(*a):
    global STOP; STOP = True
    print("\n[signal] finishing step then checkpointing...", flush=True)


class EMA:
    """fp32 shadow weights, foreach update, decay warmed up as min(decay, (1+t)/(10+t))."""
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.keys = [k for k, v in model.state_dict().items() if v.dtype.is_floating_point]
        self.shadow = [model.state_dict()[k].detach().clone().float() for k in self.keys]
        self.t = 0
    @torch.no_grad()
    def update(self, model):
        self.t += 1
        d = min(self.decay, (1 + self.t) / (10 + self.t))
        sd = model.state_dict()
        src = [sd[k].detach() for k in self.keys]
        if src[0].dtype != torch.float32: src = [s.float() for s in src]
        torch._foreach_lerp_(self.shadow, src, 1 - d)
    def state_dict(self):
        return {"t": self.t, "shadow": {k: v.cpu() for k, v in zip(self.keys, self.shadow)}}
    def load_state_dict(self, st, device="cuda"):
        self.t = st.get("t", 0); sh = st["shadow"]
        self.shadow = [sh[k].to(device).float() for k in self.keys]
    def copy_to(self, model):
        sd = model.state_dict(); ema = dict(zip(self.keys, self.shadow))
        return {k: (ema[k].to(v.dtype) if k in ema else v) for k, v in sd.items()}
    def bf16_state(self):
        return {k: v.to(torch.bfloat16).cpu().contiguous() for k, v in zip(self.keys, self.shadow)}


def lr_at(step, base, warmup):
    return base * min(1.0, (step + 1) / max(warmup, 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="run1"); p.add_argument("--cache", default="out/cache/run1")
    p.add_argument("--models", default="out/models"); p.add_argument("--ae", default="flux2")
    p.add_argument("--config", default="run1")
    p.add_argument("--steps", type=int, default=400000); p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4); p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--cfg-drop", type=float, default=0.10); p.add_argument("--p-long", type=float, default=0.50)
    p.add_argument("--ema", type=float, default=0.9999)
    p.add_argument("--shift", type=float, default=2.8, help="timestep shift alpha for training and sampling")
    p.add_argument("--w-cos", type=float, default=1.0); p.add_argument("--w-disp", type=float, default=0.5)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--log-every", type=int, default=20); p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--sample-every", type=int, default=500); p.add_argument("--ckpt-every", type=int, default=2500)
    p.add_argument("--ema-snapshot-every", type=int, default=10000)
    p.add_argument("--eval-every", type=int, default=5000, help="grids + CLIP/PickScore/HPS (0 disables)")
    p.add_argument("--fid-every", type=int, default=10000, help="FID/FD-DINOv2/object accuracy (0 disables)")
    p.add_argument("--fid-n", type=int, default=3000); p.add_argument("--eval-steps", type=int, default=20)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--novel", default="prompts/novel.txt")
    p.add_argument("--resume", default=None); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    signal.signal(signal.SIGTERM, _sig); signal.signal(signal.SIGINT, _sig)
    torch.manual_seed(a.seed); random.seed(a.seed)
    out = os.path.join("out", "runs", a.run)
    for d in ("samples", "evals"): os.makedirs(os.path.join(out, d), exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True

    C = BucketCache(a.cache, models=a.models)
    lat_ch = AE.SPECS[a.ae]["ch"]
    model = TinyDiT(latent_ch=lat_ch, ctx_dim=T.D_MODEL, **CONFIGS[a.config]).cuda()
    print(f"  model {a.config}: {model.n_params()/1e6:.1f}M params", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0, fused=True)
    ema = EMA(model, a.ema)
    step0 = 0
    if a.resume and os.path.exists(a.resume):
        ck = torch.load(a.resume, map_location="cuda", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); ema.load_state_dict(ck["ema"]); step0 = ck["step"]
        print(f"  resumed from {a.resume} at step {step0}", flush=True)
    if not a.no_compile:   # one graph per bucket shape (5) plus val/eval variants; default limit is 8
        for attr in ("recompile_limit", "cache_size_limit"):
            if hasattr(torch._dynamo.config, attr): setattr(torch._dynamo.config, attr, 64)
    net = model if a.no_compile else torch.compile(model, dynamic=False)
    t_mean = flow.shift_mean(a.shift)

    # fixed prompts/noise for the filmstrip: 8 held-out captions, square
    fixed = [c for _, c, _ in C.val_prompts(2)][:8] or ["a photo"]
    fixed_ctx, fixed_msk = C.text_for(fixed)
    null_ctx, null_msk = C.null()
    novel = [l.strip() for l in open(a.novel) if l.strip()] if os.path.exists(a.novel) else []
    json.dump({"prompts": fixed, "config": vars(a), "params_m": round(model.n_params() / 1e6, 1),
               "buckets": {b: C.buckets[b] for b in C.names}}, open(os.path.join(out, "meta.json"), "w"), indent=1)
    aem = AE.load(a.ae, os.path.join(a.models, a.ae))
    mf = open(os.path.join(out, "metrics.jsonl"), "a")
    def log(**kw): mf.write(json.dumps(kw) + "\n"); mf.flush()

    from .sample import generate, decode_chunked, render_sections, write_eval_assets

    def with_ema(fn):
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema.copy_to(model)); model.eval()
        try: return fn()
        finally: model.load_state_dict(backup); model.train()

    def gen_images(prompts, shape_wh, seed):
        ctx, msk = C.text_for(prompts)
        W, H = shape_wh
        z = generate(model, ctx, msk, null_ctx, null_msk, lat_ch, H // 8, W // 8, a.eval_steps, a.cfg, seed, a.shift)
        return decode_chunked(aem, z, C.mean, C.std)

    @torch.no_grad()
    def snapshot(step):
        from PIL import Image
        def go():
            img = gen_images(fixed, (256, 256), 1234)
            n, _, H, W = img.shape; grid = np.zeros((H, n * W, 3), np.uint8)
            for k in range(n): grid[:, k * W:(k + 1) * W] = img[k].transpose(1, 2, 0)
            Image.fromarray(grid).save(os.path.join(out, "samples", f"{step:07d}.png"))
        with_ema(go)

    # Evaluation prompts are frozen in the run directory so every eval uses identical prompts even if
    # the cache changes (rows appended later). Created on first use, never rewritten.
    ep_path = os.path.join(out, "eval_prompts.json")
    EP = json.load(open(ep_path)) if os.path.exists(ep_path) else {}
    if "grid" not in EP: EP["grid"] = C.val_prompts(4, rng_seed=7)
    if "fid" not in EP: EP["fid"] = C.val_prompts(max(1, a.fid_n // len(C.names)), rng_seed=11)
    EP = {k: [tuple(x) for x in v] for k, v in EP.items()}
    json.dump(EP, open(ep_path, "w"))
    print(f"  eval prompts: {len(EP['grid'])} grid + {len(EP['fid'])} fid from {ep_path}", flush=True)

    @torch.no_grad()
    def evaluate(step, do_fid):
        from . import metrics
        rec = {}
        def go():
            secs = []
            # UNSEEN: held-out captions, 4 per bucket, in their own bucket shape
            vp = EP["grid"]                                     # frozen for the whole run (eval_prompts.json)
            unseen_imgs, unseen_txt = [], []
            for b in C.names:
                ps = [c for bb, c, _ in vp if bb == b]
                if not ps: continue
                img = gen_images(ps, tuple(C.buckets[b]), 1234)
                sc = metrics.clip_score(img, ps)
                secs.append((f"UNSEEN {b.replace('_', ':')} · held-out captions ({C.buckets[b][0]}x{C.buckets[b][1]})", ps, img, sc, sc))
                unseen_imgs += list(img); unseen_txt += ps
            # NOVEL: hand-written prompts, square
            if novel:
                img = gen_images(novel, (256, 256), 4321); sc = metrics.clip_score(img, novel)
                secs.append((f"NOVEL · hand-written prompts", novel[:12], img[:12], sc[:12], sc))
            render_sections(secs, os.path.join(out, "evals", f"{step}.png"), step, a.cfg, a.eval_steps)
            write_eval_assets(secs, os.path.join(out, "evals"), step, a.cfg, a.eval_steps)
            idx = sorted(int(f[:-4]) for f in os.listdir(os.path.join(out, "evals")) if f.endswith(".png") and f[:-4].isdigit())
            json.dump(idx, open(os.path.join(out, "evals", "index.json"), "w"))
            def per_shape(fn, imgs, txt):
                out_ = []
                by = {}
                for i, im in enumerate(imgs): by.setdefault(im.shape, []).append(i)
                for ids in by.values():
                    sc = fn(np.stack([imgs[i] for i in ids]), [txt[i] for i in ids]); out_ += sc
                return float(np.mean(out_))
            rec["clip_unseen"] = per_shape(metrics.clip_score, unseen_imgs, unseen_txt)
            if novel: rec["clip_novel"] = float(np.mean(sc))
            for name, fn in (("pickscore", metrics.pickscore), ("hpsv2", metrics.hpsv2)):
                try:
                    rec[f"{name}_unseen"] = per_shape(fn, unseen_imgs, unseen_txt)
                    if novel: rec[f"{name}_novel"] = float(np.mean(fn(img, novel)))
                except Exception as ex: print(f"  ({name} skipped: {str(ex)[:80]})", flush=True)
            if do_fid:
                try:
                    gen, ref = [], []
                    per_b = max(1, a.fid_n // len(C.names))
                    for b in C.names:
                        vi = C.val_images(b)
                        if vi is None or len(vi) == 0: continue
                        vp_b = [(c, j) for bb, c, j in EP["fid"] if bb == b]
                        if not vp_b: continue
                        img = gen_images([c for c, _ in vp_b], tuple(C.buckets[b]), 777)
                        gen += list(img); ref += [np.asarray(vi[j]) for _, j in vp_b if j < len(vi)]
                    rec["fid"], rec["fd_dino"] = metrics.frechet(gen, ref)
                    op = metrics.object_prompts(4)
                    img = gen_images([q for q, _ in op], (256, 256), 99)
                    rec["obj_acc"], _ = metrics.object_accuracy(img, [c for _, c in op])
                except Exception as ex: print(f"  (fid/obj skipped: {str(ex)[:100]})", flush=True)
            metrics.unload()
        with_ema(go)
        return rec

    def save_full(step):
        tmp = os.path.join(out, "ckpt_last.pt.tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "ema": ema.state_dict(),
                    "step": step, "config": vars(a)}, tmp)
        os.replace(tmp, os.path.join(out, "ckpt_last.pt"))

    def save_ema(step):
        from safetensors.torch import save_file
        save_file(ema.bf16_state(), os.path.join(out, f"ema_{step:07d}.safetensors"), metadata={"step": str(step), "config": a.config})

    @torch.no_grad()
    def val_loss():
        tot, n, per_src = 0.0, 0, {}
        for b, s, z, ctx, msk in C.val_batches():
            with torch.autocast("cuda", dtype=torch.bfloat16):   # eager: val batches have their own shapes
                l, _, per, _ = flow.loss(model, z, ctx, msk, t_mean=t_mean, w_cos=0, w_disp=0)
            v = per.sum().item(); k = per.numel()
            tot += v; n += k; ps = per_src.setdefault(s, [0.0, 0]); ps[0] += v; ps[1] += k
        rec = {"val_loss": tot / max(n, 1)}
        for s, (v, k) in per_src.items(): rec[f"val_loss_{s}"] = v / k
        return rec

    C.start(a.bs, a.cfg_drop, a.p_long, seed=a.seed + step0)
    print(f"  train {C.n_train:,} rows, held out {C.n_val:,} | {a.steps} steps, bs {a.bs}, "
          f"lr {a.lr}, shift {a.shift} -> {out}", flush=True)
    model.train(); t0 = time.time(); seen = 0; acc = {}; nacc = 0; t_lo = t_hi = 0.0; n_lo = n_hi = 0
    for step in range(step0, a.steps):
        for g in opt.param_groups: g["lr"] = lr_at(step, a.lr, a.warmup)
        b, z, ctx, msk = C.batch()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l, tt, per, parts = flow.loss(net, z, ctx, msk, t_mean=t_mean, w_cos=a.w_cos, w_disp=a.w_disp)
        opt.zero_grad(set_to_none=True)
        l.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); ema.update(model)
        with torch.no_grad():
            lo = tt < 0.5
            if lo.any(): t_lo += per[lo].mean().item(); n_lo += 1
            if (~lo).any(): t_hi += per[~lo].mean().item(); n_hi += 1
        acc["loss"] = acc.get("loss", 0.0) + l.item()
        for k, v in parts.items(): acc["loss_" + k] = acc.get("loss_" + k, 0.0) + v.item()
        nacc += 1; seen += a.bs
        if (step + 1) % a.log_every == 0:
            dt = time.time() - t0
            rec = {k: v / nacc for k, v in acc.items()}
            rec.update(step=step + 1, lr=opt.param_groups[0]["lr"], gnorm=float(gn), ips=seen / dt, secs=dt,
                       bucket=b, tokens=int(ctx.shape[0] * (z.shape[-1] // 2) * (z.shape[-2] // 2)))
            if n_lo: rec["loss_t_lo"] = t_lo / n_lo
            if n_hi: rec["loss_t_hi"] = t_hi / n_hi
            log(**rec)
            print(f"    step {step+1:>7} loss {rec['loss']:.4f} (mse {rec.get('loss_mse', 0):.4f}) gn {float(gn):.2f} "
                  f"{seen/dt:.0f} img/s [{b}]", flush=True)
            acc = {}; nacc = 0; t_lo = t_hi = 0.0; n_lo = n_hi = 0
        if a.val_every and (step + 1) % a.val_every == 0:
            model.eval(); rec = val_loss(); model.train(); log(step=step + 1, **rec)
        if a.sample_every and ((step + 1) % a.sample_every == 0 or step + 1 == a.steps):
            snapshot(step + 1); log(step=step + 1, sample=f"samples/{step+1:07d}.png")
        if (step + 1) % a.ckpt_every == 0: save_full(step + 1)
        if a.ema_snapshot_every and (step + 1) % a.ema_snapshot_every == 0: save_ema(step + 1)
        if a.eval_every and (step + 1) % a.eval_every == 0:
            rec = evaluate(step + 1, do_fid=bool(a.fid_every) and (step + 1) % a.fid_every == 0)
            log(step=step + 1, eval=f"evals/{step+1}.png", **rec)
            print("    eval " + " ".join(f"{k}={v:.3f}" for k, v in rec.items()), flush=True)
        if STOP:
            save_full(step + 1); print(f"[stopped] checkpoint at step {step+1}", flush=True); break
    if not STOP: save_full(a.steps); save_ema(a.steps)
    print("done", flush=True)


if __name__ == "__main__":
    main()

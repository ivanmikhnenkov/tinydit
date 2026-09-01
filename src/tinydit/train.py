"""Training loop.

Everything is precached, so there is no dataloader: latents live on the GPU and a
batch is a gather. Metrics stream to metrics.jsonl and sample grids to samples/,
which dashboard.html polls live.

Checkpoints carry model + EMA + optimizer + step, and one is written on SIGTERM,
so the run can be killed to free the GPU and resumed exactly.
"""
from __future__ import annotations
import argparse, json, math, os, random, signal, time
import numpy as np, torch
from . import ae as AE, flow
from .model import TinyDiT, CONFIGS

STOP = False
def _sig(*a):
    global STOP; STOP = True
    print("\n[signal] finishing step then checkpointing...", flush=True)


def _latent_stats(path, sample, device):
    """Per-channel whitening stats. Prefer the file the ingest writes; if it is absent
    (an interrupted or still-running ingest) derive them from the cached latents, which
    is what the file would have contained anyway."""
    if os.path.exists(path):
        st = json.load(open(path))
    else:
        x = sample.float()
        st = {"mean": x.mean(dim=(0, 2, 3)).tolist(),
              "std": x.std(dim=(0, 2, 3)).clamp_min(1e-6).tolist()}
        json.dump(st, open(path, "w"), indent=1)
        print(f"  computed latent stats from cache -> {os.path.basename(path)}", flush=True)
    m = torch.tensor(st["mean"], device=device).view(1, -1, 1, 1).float()
    sd = torch.tensor(st["std"], device=device).view(1, -1, 1, 1).float()
    return m, sd


class Cache:
    """Latents in CPU RAM, captions encoded live.

    A million 256^2 latents is 65 GB — far too large to sit on a 95 GB card alongside a
    model and optimizer. They live in system RAM and each batch is transferred: 16.8 MB,
    about 2 ms over PCIe against an 800 ms step. Captions are encoded on the fly for the
    same reason caching them was dropped: 49 GB saved for 2% of a step.
    """
    def __init__(self, root, datasets, ae_name, res, max_len=64, device="cuda",
                 overfit=0, val_n=0, models="out/models", weights=None):
        from . import text as T
        if isinstance(datasets, str): datasets = [datasets]
        zs, caps = [], []
        for ds in datasets:
            d = os.path.join(root, ds)
            lat = np.load(os.path.join(d, f"lat_{ae_name}_{res}.npy"), mmap_mode="r")
            cj, cl = os.path.join(d, "captions.json"), os.path.join(d, "captions.jsonl")
            if os.path.exists(cj):
                cp = json.load(open(cj))["caps"]
            else:   # ingest still running: the jsonl is appended per shard
                cp = [json.loads(l) for l in open(cl)]
            n = min(lat.shape[0], len(cp))
            meta_p = os.path.join(d, "meta.json")
            if os.path.exists(meta_p):
                mt = json.load(open(meta_p))
                n = min(n, mt.get("n_128", n) if res <= 128 else mt.get("n", n))
            if overfit: n = min(n, overfit)
            print(f"    {ds}: {n:,} images", flush=True)
            zs.append(torch.from_numpy(np.ascontiguousarray(lat[:n])))   # stays on CPU
            caps.append(cp[:n])
        self.z = torch.cat(zs) if len(zs) > 1 else zs[0]
        self.caps = [c for cl in caps for c in cl]
        self.n = self.z.shape[0]
        self.val_n = 0 if overfit else min(val_n, max(self.n // 10, 0))
        self.n_train = self.n - self.val_n
        self.mean, self.std = _latent_stats(
            os.path.join(root, datasets[0], f"lat_{ae_name}_{res}.json"),
            self.z[:4096].to(device), device)
        self.tok, self.enc = T.load(os.path.join(models, "t5"))
        self.max_len, self.dev = max_len, device
        self._null = T.embed(self.tok, self.enc, [""], max_len)
        print(f"  cache: {self.n:,} images ({self.n_train:,} train / {self.val_n:,} held out), "
              f"latents {tuple(self.z.shape[1:])} in RAM "
              f"({self.z.numel()*2/1e9:.1f} GB); captions live", flush=True)

    def _encode(self, texts):
        from . import text as T
        e, m = T.embed(self.tok, self.enc, texts, self.max_len)
        return e.float(), m

    def _texts(self, idx, cfg_drop):
        out = []
        for j in idx:
            if random.random() < cfg_drop:
                out.append(""); continue
            c = random.choice(self.caps[j])
            # length augmentation: sometimes clip to a clause so short prompts are seen too
            if random.random() < 0.25:
                w = c.split()
                if len(w) > 6: c = " ".join(w[:random.randint(4, max(5, len(w)//2))])
            out.append(c)
        return out

    def batch(self, bs, cfg_drop=0.1, gen=None, val=False):
        lo, hi = (self.n_train, self.n) if val else (0, self.n_train)
        idx = torch.randint(lo, hi, (bs,), generator=gen)          # CPU indices
        z = self.z[idx].to(self.dev, non_blocking=True).float()
        z = (z - self.mean) / self.std
        ctx, msk = self._encode(self._texts(idx.tolist(), cfg_drop))
        return z, ctx, msk

    def null(self, bs):
        e, m = self._null
        return e.float().expand(bs, -1, -1), m.expand(bs, -1)

    def text_for(self, texts):
        return self._encode(list(texts))


class BucketCache:
    """Aspect-ratio bucketed latents. Each batch is drawn from a single bucket so the
    shapes match; RoPE handles the differing grids with no architectural change."""
    def __init__(self, root, dataset, ae_name, max_len=64, device="cuda",
                 val_n=0, models="out/models"):
        from . import text as T
        d = os.path.join(root, dataset)
        meta = json.load(open(os.path.join(d, "meta.json")))
        self.dims = meta["bucket_dims"]
        self.keys, self.z, self.caps = [], {}, {}
        for k, n in meta["buckets"].items():
            if n < 256: continue
            arr = np.load(os.path.join(d, f"bucket_{ae_name}_{k}.npy"), mmap_mode="r")[:n]
            cp = [json.loads(l) for l in open(os.path.join(d, f"captions_{k}.jsonl"))][:n]
            n = min(n, len(cp))
            self.z[k] = torch.from_numpy(np.ascontiguousarray(arr[:n]))   # CPU
            self.caps[k] = cp[:n]; self.keys.append(k)
        k0 = self.keys[0]
        self.mean, self.std = _latent_stats(os.path.join(d, f"bucket_{ae_name}_{k0}.json"),
                                            self.z[k0][:4096].to(device), device)
        sizes = np.array([len(self.caps[k]) for k in self.keys], dtype=np.float64)
        self.p = sizes / sizes.sum()
        self.n = int(sizes.sum()); self.val_n = 0; self.n_train = self.n
        self.tok, self.enc = T.load(os.path.join(models, "t5"))
        self.max_len, self.dev = max_len, device
        self._null = T.embed(self.tok, self.enc, [""], max_len)
        print("  buckets: " + ", ".join(f"{k} {len(self.caps[k]):,} "
              f"({self.dims[k][0]}x{self.dims[k][1]})" for k in self.keys), flush=True)

    def _encode(self, texts):
        from . import text as T
        e, m = T.embed(self.tok, self.enc, texts, self.max_len)
        return e.float(), m

    def batch(self, bs, cfg_drop=0.1, gen=None, val=False):
        k = self.keys[int(np.random.choice(len(self.keys), p=self.p))]
        zk, ck = self.z[k], self.caps[k]
        i = torch.randint(0, len(ck), (bs,), generator=gen)
        z = zk[i].to(self.dev, non_blocking=True).float()
        z = (z - self.mean) / self.std
        drop = (torch.rand(bs, generator=gen) < cfg_drop).tolist()
        texts = ["" if drop[j] else random.choice(ck[t]) for j, t in enumerate(i.tolist())]
        ctx, msk = self._encode(texts)
        return z, ctx, msk

    def null(self, bs):
        e, m = self._null
        return e.float().expand(bs, -1, -1), m.expand(bs, -1)

    def text_for(self, texts):
        return self._encode(list(texts))


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1-self.decay)
    def copy_to(self, model):
        sd = model.state_dict()
        return {k: (self.shadow[k].to(sd[k].dtype) if k in self.shadow else v)
                for k, v in sd.items()}


def lr_at(step, base, warmup):
    return base * min(1.0, (step + 1) / max(warmup, 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="cub_flux2")
    p.add_argument("--dataset", default="cub", nargs="+"); p.add_argument("--ae", default="flux2")
    p.add_argument("--cache", default="out/cache"); p.add_argument("--models", default="out/models")
    p.add_argument("--config", default="base")
    p.add_argument("--res", type=int, default=256)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--cfg-drop", type=float, default=0.1)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--overfit", type=int, default=0)
    p.add_argument("--ckpt-every", type=int, default=2000)
    p.add_argument("--sample-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--sample-steps", type=int, default=40)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--resume", default=None)
    p.add_argument("--max-len", type=int, default=64)
    p.add_argument("--eval-every", type=int, default=5000,
                   help="steps between seen/unseen/novel evaluations (0 disables)")
    p.add_argument("--eval-sets", nargs="*", default=[
        "SEEN — training captions=out/prompts_seen.txt",
        "UNSEEN — COCO val2017, never trained on=out/prompts_heldout.txt",
        "NOVEL — hand-written compositions=out/prompts_novel.txt"])
    p.add_argument("--val-n", type=int, default=2048,
                   help="images held out of training for a validation loss (0 disables)")
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--eval-cfg", type=float, default=4.0)
    p.add_argument("--no-clip", action="store_true")
    p.add_argument("--buckets", action="store_true",
                   help="train on aspect-ratio buckets instead of square crops")
    p.add_argument("--eval-render", type=int, default=12,
                   help="images drawn per set; CLIP scores the full set")
    p.add_argument("--eval-steps", type=int, default=50)
    a = p.parse_args()

    signal.signal(signal.SIGTERM, _sig); signal.signal(signal.SIGINT, _sig)
    a.dataset = a.dataset if isinstance(a.dataset, list) else [a.dataset]
    out = os.path.join("out", "runs", a.run); os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out, "samples"), exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True

    if a.buckets:
        ds0 = a.dataset[0] if isinstance(a.dataset, list) else a.dataset
        C = BucketCache(a.cache, ds0, a.ae, a.max_len, val_n=a.val_n, models=a.models)
    else:
        C = Cache(a.cache, a.dataset, a.ae, a.res, a.max_len, overfit=a.overfit,
                  val_n=a.val_n, models=a.models)
    lat_ch = AE.SPECS[a.ae]["ch"]
    model = TinyDiT(latent_ch=lat_ch, ctx_dim=768, **CONFIGS[a.config]).cuda()
    print(f"  model {a.config}: {model.n_params()/1e6:.1f}M params", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    ema = EMA(model, a.ema)
    step0 = 0
    if a.resume and os.path.exists(a.resume):
        ck = torch.load(a.resume, map_location="cuda")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        ema.shadow = {k: v.cuda() for k, v in ck["ema"].items()}; step0 = ck["step"]
        print(f"  resumed from {a.resume} at step {step0}", flush=True)
    net = torch.compile(model) if a.compile else model

    # fixed prompts + fixed noise so the sample filmstrip is comparable across steps
    if a.buckets:
        k0 = C.keys[0]
        fixed_prompts = [C.caps[k0][i][0]
                         for i in range(0, len(C.caps[k0]), max(1, len(C.caps[k0]) // 8))][:8]
    else:
        fixed_img = list(range(0, C.n_train, max(1, C.n_train // 8)))[:8]
        fixed_prompts = [C.caps[i][0] for i in fixed_img]
    fixed_ctx, fixed_msk = C.text_for(fixed_prompts)
    null_ctx, null_msk = C.null(1)
    json.dump({"prompts": fixed_prompts,
               "config": vars(a), "params_m": round(model.n_params()/1e6, 1)},
              open(os.path.join(out, "meta.json"), "w"), indent=1)
    aem = AE.load(a.ae, os.path.join(a.models, a.ae))

    # Encode the evaluation prompts once, then drop T5: keeping 36 embeddings costs
    # a few MB, whereas holding the encoder would cost 110M params of VRAM all run.
    EVAL = []
    if a.eval_every > 0:
        from . import text as _T
        want = [(lb, [l.strip() for l in open(pt) if l.strip()])
                for lb, _, pt in (sp.partition("=") for sp in a.eval_sets)
                if os.path.exists(pt)]
        if want:
            for lb, ps in want:
                e, m = C.text_for(ps)
                EVAL.append((lb, ps, e, m))
            EVAL_NULL = C.null(1)
            print(f"  eval: {sum(len(p) for _, p, _, _ in EVAL)} prompts across "
                  f"{len(EVAL)} sets, every {a.eval_every} steps", flush=True)
        else:
            a.eval_every = 0
    os.makedirs(os.path.join(out, "evals"), exist_ok=True)

    @torch.no_grad()
    def evaluate(step):
        from .sample import generate, render_sections, decode_chunked, write_eval_assets
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema.copy_to(model)); model.eval()
        secs = []; clip_means = {}
        for lb, ps, e, m in EVAL:
            z = generate(model, e, m, EVAL_NULL[0], EVAL_NULL[1], lat_ch, a.res,
                         a.eval_steps, a.eval_cfg, seed=1234)
            img = decode_chunked(aem, z, C.mean, C.std)
            sc = None
            if not a.no_clip:
                try:
                    from . import clipscore
                    per, mn = clipscore.score(img, ps, os.path.join(a.models, "clip"))
                    sc = per; clip_means[lb.split(" ")[0]] = round(mn, 4)
                except Exception as ex:
                    print(f"  (clip skipped: {str(ex)[:60]})", flush=True)
            k = min(a.eval_render, len(ps))
            secs.append((lb, ps[:k], img[:k]) if sc is None
                        else (lb, ps[:k], img[:k], sc[:k], sc))
        render_sections(secs, os.path.join(out, "evals", f"{step}.png"),
                        step, a.eval_cfg, a.eval_steps)
        write_eval_assets(secs, os.path.join(out, "evals"), step, a.eval_cfg, a.eval_steps)
        idx = sorted(int(f[:-4]) for f in os.listdir(os.path.join(out, "evals"))
                     if f.endswith(".png") and f[:-4].isdigit())
        json.dump(idx, open(os.path.join(out, "evals", "index.json"), "w"))
        model.load_state_dict(backup); model.train()
        return clip_means

    mf = open(os.path.join(out, "metrics.jsonl"), "a")
    def log(**kw):
        mf.write(json.dumps(kw) + "\n"); mf.flush()

    @torch.no_grad()
    def snapshot(step):
        from PIL import Image
        sd_backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema.copy_to(model))
        model.eval()
        from .sample import generate, decode_chunked
        x = generate(model, fixed_ctx, fixed_msk, null_ctx, null_msk, lat_ch, a.res,
                     a.sample_steps, a.cfg, seed=1234)
        img = decode_chunked(aem, x, C.mean, C.std)
        n, _, H, W = img.shape
        grid = np.zeros((H, n*W, 3), np.uint8)
        for k in range(n): grid[:, k*W:(k+1)*W] = img[k].transpose(1, 2, 0)
        Image.fromarray(grid).save(os.path.join(out, "samples", f"{step:07d}.png"))
        model.load_state_dict(sd_backup); model.train()

    def save(step, tag="last"):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "ema": {k: v.cpu() for k, v in ema.shadow.items()},
                    "step": step, "config": vars(a)},
                   os.path.join(out, f"ckpt_{tag}.pt"))

    @torch.no_grad()
    def val_loss():
        if C.val_n == 0: return None
        g = torch.Generator().manual_seed(99)   # CPU: indices are drawn on CPU now
        tot = 0.0
        for _ in range(4):
            z, ctx, msk = C.batch(min(a.bs, 128), cfg_drop=0.0, gen=g, val=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                l, _, _ = flow.loss(net, z, ctx, msk)
            tot += l.item()
        return tot / 4

    print(f"  train {C.n_train} imgs, held out {C.val_n} | {a.steps} steps @ res {a.res}, "
          f"bs {a.bs} -> {out}", flush=True)
    model.train(); t0 = time.time(); seen = 0; acc = 0.0; nacc = 0
    t_lo = t_hi = 0.0; n_lo = n_hi = 0; cur_val = None
    for step in range(step0, a.steps):
        for g in opt.param_groups: g["lr"] = lr_at(step, a.lr, a.warmup)
        z, ctx, msk = C.batch(a.bs, 0.0 if a.overfit else a.cfg_drop)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l, tt, per = flow.loss(net, z, ctx, msk)
        opt.zero_grad(set_to_none=True)
        l.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); ema.update(model)
        with torch.no_grad():
            lo_m = tt < 0.5
            if lo_m.any(): t_lo += per[lo_m].mean().item(); n_lo += 1
            if (~lo_m).any(): t_hi += per[~lo_m].mean().item(); n_hi += 1
        acc += l.item(); nacc += 1; seen += a.bs
        if (step+1) % a.log_every == 0:
            dt = time.time()-t0
            rec = dict(step=step+1, loss=acc/nacc, lr=opt.param_groups[0]["lr"],
                       gnorm=float(gn), ips=seen/dt, res=a.res, secs=dt)
            if n_lo: rec["loss_t_lo"] = t_lo/n_lo
            if n_hi: rec["loss_t_hi"] = t_hi/n_hi
            if cur_val is not None: rec["val_loss"] = cur_val
            log(**rec)
            t_lo = t_hi = 0.0; n_lo = n_hi = 0
            print(f"    step {step+1:>7} loss {acc/nacc:.4f} gn {float(gn):.2f} "
                  f"{seen/dt:.0f} img/s", flush=True)
            acc = 0.0; nacc = 0
        if a.val_n and (step+1) % a.val_every == 0:
            model.eval(); cur_val = val_loss(); model.train()
        if (step+1) % a.sample_every == 0 or step+1 == a.steps:
            snapshot(step+1); log(step=step+1, sample=f"samples/{step+1:07d}.png")
        if (step+1) % a.ckpt_every == 0:
            save(step+1)
        if a.eval_every > 0 and (step+1) % a.eval_every == 0:
            cm = evaluate(step+1)
            rec = dict(step=step+1, eval=f"evals/{step+1}.png")
            for k, v in (cm or {}).items(): rec[f"clip_{k.lower()}"] = v
            log(**rec)
        if STOP:
            save(step+1, "interrupt"); print(f"[stopped] checkpoint at step {step+1}"); break
    save(a.steps if not STOP else step+1, "final" if not STOP else "interrupt")
    print("done", flush=True)


if __name__ == "__main__":
    main()

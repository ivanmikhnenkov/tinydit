"""RL post-training loop, shared by both algorithms and both rewards (stage-1 JPEG, stage-2 judges).

    python -m tinydit.rl.train --alg grpo --objective compress --hours 1.5
    python -m tinydit.rl.train --alg dgpo --objective incompress --hours 1.5

One round = P prompts x G images: rollout (SDE for GRPO, ODE for DGPO) -> decode -> reward ->
group-relative advantages -> update -> log. Everything but the rollout mode and the update is
identical, so runs compare on GPU-hours: `gpu_hours` is the CUDA-synchronised train time
(rollout + reward + update); evaluation and checkpoints are excluded from that clock.
Evaluation, every --eval-every-min train minutes, uses the deployment sampler (20-step ODE, CFG 4)
on all 45 prompts x --eval-seeds seeds: JPEG kB, CLIP, PickScore, and a 16-image grid for the
dashboard (rl_dashboard.html). Output: out/runs/<run>/{meta.json, metrics.jsonl, evals/, ckpt_last.pt,
policy_final.safetensors}; metric keys are listed in notes/CONTRACTS.md.
"""
from __future__ import annotations
import argparse, json, os, random, signal, time
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.abspath("out/inductor_cache"))
import numpy as np, torch, torch._dynamo
from .. import ae as AE, text as T
from ..sample import load_model, decode_chunked, pad_ctx
from . import rollout as R, grpo, dgpo, rewards as RW
from ..train import EMA

STOP = False
def _sig(*a):
    global STOP; STOP = True
    print("\n[signal] finishing the round, then checkpointing...", flush=True)


class Text:
    """flan-t5-base, prompts padded to a fixed width so torch.compile sees one shape."""
    def __init__(self, models):
        self.tok, self.enc = T.load(os.path.join(models, "t5"))
        e, m = T.embed(self.tok, self.enc, [""], T.MAX_SHORT)
        self.null_ctx, self.null_msk = pad_ctx(e.float(), m, T.MAX_LONG)
    def __call__(self, texts):
        e, m = T.embed_mixed(self.tok, self.enc, texts, [False] * len(texts))
        return pad_ctx(e.float(), m, T.MAX_LONG)


def grid_png(imgs_u8, labels, path, cols=8):
    """(n,3,H,W) -> one image, `cols` per row, a caption strip under each cell."""
    from PIL import Image, ImageDraw, ImageFont
    n, _, H, W = imgs_u8.shape; rows = (n + cols - 1) // cols; cap = 16
    canvas = Image.new("RGB", (cols * W, rows * (H + cap)), (247, 248, 248))
    d = ImageDraw.Draw(canvas)
    try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except OSError: font = ImageFont.load_default()
    for k in range(n):
        r, c = divmod(k, cols)
        canvas.paste(Image.fromarray(imgs_u8[k].transpose(1, 2, 0)), (c * W, r * (H + cap)))
        d.text((c * W + 3, r * (H + cap) + H + 2), labels[k], font=font, fill=(60, 70, 76))
    canvas.save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alg", choices=["grpo", "dgpo"], required=True)
    p.add_argument("--objective", choices=list(RW.OBJECTIVES), default="compress")
    p.add_argument("--run", default=None, help="default rl_<alg>_<objective>")
    p.add_argument("--base", default="out/runs/run1/ema_0400000.safetensors"); p.add_argument("--config", default="run1")
    p.add_argument("--ref", default=None, help="reference weights for the KL / DGPO margin (default: --base)")
    p.add_argument("--reward", choices=["jpeg", "stage2"], default="jpeg", help="jpeg: DDPO's kB objective; stage2: align + real + quality + anatomy")
    p.add_argument("--prompts", default="prompts/rl_stage2/train.json", help="stage2 training prompts (json list of {prompt, source})")
    p.add_argument("--eval-set", default="prompts/rl_stage2/eval.json", help="stage2 fixed evaluation prompts + grid")
    p.add_argument("--refs", default="out/data/refs/real_256.npz", help="real 256x256 images for FID / FD-DINOv2 at eval")
    p.add_argument("--stats", default=None, help="latent whitening stats.json (default: <cache>/stats.json, then out/hf_model/stats.json)")
    p.add_argument("--quality", choices=["hpsv3", "hpsv2"], default="hpsv3")
    p.add_argument("--w-align", type=float, default=0.40); p.add_argument("--w-real", type=float, default=0.30)
    p.add_argument("--w-qual", type=float, default=0.15); p.add_argument("--w-anat", type=float, default=0.15); p.add_argument("--w-style", type=float, default=0.15)
    p.add_argument("--judge-batch", type=int, default=8); p.add_argument("--stop-fd", type=float, default=0.0, help="stop when eval FD-DINOv2 exceeds its running best by this factor (e.g. 1.015) for --stop-patience evals; 0 = off"); p.add_argument("--stop-patience", type=int, default=2); p.add_argument("--real-source", choices=["probe", "vlm"], default="probe"); p.add_argument("--ema", type=float, default=0.0, help="EMA decay of the policy for evaluation and snapshots (0 = off; 0.995 ~ 50 rounds)"); p.add_argument("--veto-align", type=float, default=0.0, help="veto samples with alignment P(yes) below this (0 = off)")
    p.add_argument("--fid-refs", type=int, default=2000, help="how many real references FID / FD use")
    p.add_argument("--models", default="out/models"); p.add_argument("--ae", default="flux2")
    p.add_argument("--cache", default="out/cache/run1", help="only stats.json (latent whitening) is read")
    p.add_argument("--hours", type=float, default=1.5, help="train-time budget in GPU-hours")
    p.add_argument("--rounds", type=int, default=10**6)
    p.add_argument("--prompts-per-round", type=int, default=16); p.add_argument("--group", type=int, default=16)
    p.add_argument("--steps", type=int, default=10); p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--shift", type=float, default=2.8)
    p.add_argument("--noise-level", type=float, default=0.7, help="SDE noise a (GRPO rollouts)")
    p.add_argument("--lr", type=float, default=1e-5); p.add_argument("--minibatch", type=int, default=64)
    p.add_argument("--beta", type=float, default=None, help="KL weight; default 0.004 (grpo) / 0.02 (dgpo)")
    p.add_argument("--clip", type=float, default=1e-4, help="GRPO PPO clip on the per-step density ratio")
    p.add_argument("--adv-clip", type=float, default=5.0); p.add_argument("--adv-sign", type=float, default=1.0, help="multiplies every advantage; -1 = diagnostic run whose reward must fall"); p.add_argument("--adv-shuffle", action="store_true", help="diagnostic: permute the advantages inside each group (same update shape, no link to the reward)")
    p.add_argument("--beta-dpo", type=float, default=100.0); p.add_argument("--dgpo-clip", type=float, default=1e-2)
    p.add_argument("--trunc", type=int, default=4, help="DGPO trains on sampler steps 0..trunc (the noisiest)")
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--eval-every-min", type=float, default=5.0); p.add_argument("--eval-steps", type=int, default=20)
    p.add_argument("--eval-seeds", type=int, default=2); p.add_argument("--eval-n", type=int, default=45)
    p.add_argument("--ckpt-every-min", type=float, default=15.0)
    p.add_argument("--rollout-chunk", type=int, default=128)
    p.add_argument("--warmup-rounds", type=int, default=1, help="rounds run without optimizer step or log: compilation stays off the clock")
    p.add_argument("--no-compile", action="store_true"); p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if a.beta is None: a.beta = 0.004 if a.alg == "grpo" else 0.02
    a.run = a.run or (f"rl_{a.alg}_{a.objective}" if a.reward == "jpeg" else f"rl2_{a.alg}")
    sign = RW.OBJECTIVES[a.objective]
    G, P = a.group, a.prompts_per_round
    N = G * P
    assert a.minibatch % G == 0 or a.alg == "grpo", "DGPO minibatches must hold whole groups"

    signal.signal(signal.SIGTERM, _sig); signal.signal(signal.SIGINT, _sig)
    torch.manual_seed(a.seed); random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    out = os.path.join("out", "runs", a.run); os.makedirs(os.path.join(out, "evals"), exist_ok=True)

    lat_ch = AE.SPECS[a.ae]["ch"]
    policy, _ = load_model(a.base, lat_ch, a.config); policy.train()
    ref, _ = load_model(a.ref or a.base, lat_ch, a.config); ref.requires_grad_(False)
    print(f"  policy {a.config}: {policy.n_params()/1e6:.1f}M params from {a.base}", flush=True)
    opt = torch.optim.AdamW(policy.parameters(), lr=a.lr, betas=(0.9, 0.999), weight_decay=0.0, fused=True)
    ema = EMA(policy, a.ema) if a.ema > 0 else None                  # rollouts use the live policy; evals and snapshots the EMA
    if not a.no_compile:
        for attr in ("recompile_limit", "cache_size_limit"):
            if hasattr(torch._dynamo.config, attr): setattr(torch._dynamo.config, attr, 64)
    net = policy if a.no_compile else torch.compile(policy, dynamic=False)
    ref_net = ref if a.no_compile else torch.compile(ref, dynamic=False)
    enc = Text(a.models)
    stats_path = next(q for q in ([a.stats] if a.stats else []) + [os.path.join(a.cache, "stats.json"), "out/hf_model/stats.json", "out/models/stats.json"] if q and os.path.exists(q))
    st = json.load(open(stats_path))
    mean = torch.tensor(st["mean"], device="cuda").view(1, -1, 1, 1).float()
    std = torch.tensor(st["std"], device="cuda").view(1, -1, 1, 1).float()
    aem = AE.load(a.ae, os.path.join(a.models, a.ae))
    from .. import metrics
    HW = 32                                              # 256x256 only: the objective is per image size
    eval_prompts = [RW.prompt(x) for x in RW.ANIMALS[:a.eval_n]]
    R2 = EV2 = TRAIN_PROMPTS = REFS = None
    if a.reward == "stage2":
        from .reward_stage2 import Stage2Reward
        R2 = Stage2Reward({"align": a.w_align, "real": a.w_real, "quality": a.w_qual, "anatomy": a.w_anat, "style": a.w_style}, a.quality, batch=a.judge_batch, veto_align=a.veto_align, real_source=a.real_source)
        TRAIN_PROMPTS = [q["prompt"] for q in json.load(open(a.prompts))]
        EV2 = json.load(open(a.eval_set)); eval_prompts = [q["prompt"] for q in EV2["prompts"]]
        if os.path.exists(a.refs):
            z = np.load(a.refs); REFS = np.asarray(z["images"][:a.fid_refs])
        print(f"  stage2 reward {R2.w} quality={a.quality}; {len(TRAIN_PROMPTS):,} train prompts; eval {len(eval_prompts)} prompts x {EV2['seeds']} seeds; refs {0 if REFS is None else len(REFS)}", flush=True)
        torch.cuda.empty_cache()

    state = {"round": 0, "samples": 0, "train_secs": 0.0, "wall_secs": 0.0, "last_eval": -1e9, "last_ckpt": 0.0}
    rng = random.Random(a.seed)
    ck_path = os.path.join(out, "ckpt_last.pt")
    if a.resume and os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location="cuda", weights_only=False)
        policy.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); state = ck["state"]; rng.setstate(ck["rng"])
        if ema is not None and ck.get("ema"): ema.load_state_dict(ck["ema"])
        elif ema is not None: print("  (no EMA in the checkpoint: EMA starts from the current policy)", flush=True)
        print(f"  resumed at round {state['round']} ({state['train_secs']/3600:.2f} GPU-h)", flush=True)
    json.dump({"rl": True, "alg": a.alg, "objective": a.objective if a.reward == "jpeg" else "stage2", "reward": a.reward, "config": vars(a),
               "prompts": ([q["prompt"] for q in EV2["prompts"] if q["id"] in EV2["grid"]] if EV2 else eval_prompts[:16]),
               "params_m": round(policy.n_params() / 1e6, 1)}, open(os.path.join(out, "meta.json"), "w"), indent=1)
    mf = open(os.path.join(out, "metrics.jsonl"), "a")
    def log(**kw): mf.write(json.dumps(kw) + "\n"); mf.flush()
    def sync(): torch.cuda.synchronize(); return time.time()
    def base_row(): return dict(round=state["round"], samples=state["samples"], gpu_hours=state["train_secs"] / 3600,
                                wall_hours=state["wall_secs"] / 3600)

    @torch.no_grad()
    def evaluate():
        t0 = time.time(); policy.eval()
        imgs, txt = [], []
        for s in range(a.eval_seeds):
            ctx, msk = enc(eval_prompts)
            xs, _, _ = R.rollout(policy, ctx, msk, enc.null_ctx, enc.null_msk, lat_ch, HW, HW, a.eval_steps, a.cfg,
                                 a.shift, 0.0, seed=10_000 + s, chunk=a.rollout_chunk)
            imgs.append(decode_chunked(aem, xs[:, -1], mean, std)); txt += eval_prompts
        policy.train()
        imgs = np.concatenate(imgs); kb = RW.jpeg_kb(imgs, a.jpeg_quality)
        rec = {"eval_kb": float(kb.mean()), "eval_kb_std": float(kb.std()), "eval_reward": float(sign * kb.mean())}
        for name, fn in (("eval_clip", metrics.clip_score), ("eval_pickscore", metrics.pickscore)):
            try: rec[name] = float(np.mean(fn(imgs, txt)))
            except Exception as ex: print(f"  ({name} skipped: {str(ex)[:80]})", flush=True)
        k = min(16, len(eval_prompts)); fn = f"{state['round']:06d}.png"
        grid_png(imgs[:k], [f"{RW.ANIMALS[i]}  {kb[i]:.1f} kB" for i in range(k)], os.path.join(out, "evals", fn))
        idx_p = os.path.join(out, "evals", "index.json")
        idx = json.load(open(idx_p)) if os.path.exists(idx_p) else []
        idx = [e for e in idx if e["round"] != state["round"]] + [dict(base_row(), file=f"evals/{fn}", **rec)]
        json.dump(sorted(idx, key=lambda e: e["round"]), open(idx_p, "w"))
        rec["eval_secs"] = time.time() - t0
        return rec

    def with_eval_weights(fn):
        if ema is None: return fn()
        backup = {k: v.detach().clone() for k, v in policy.state_dict().items()}
        policy.load_state_dict(ema.copy_to(policy))
        try: return fn()
        finally: policy.load_state_dict(backup)

    @torch.no_grad()
    def evaluate2():
        return with_eval_weights(_evaluate2)

    @torch.no_grad()
    def _evaluate2():
        """Fixed held-out prompts x seeds with the deployment sampler (EMA weights when --ema); every reward term, pixel
        statistics, CLIP, FID and FD-DINOv2 against real references; per-image PNGs + manifest; the 16-prompt grid."""
        t0 = time.time(); policy.eval()
        items = EV2["prompts"]; imgs, meta = [], []
        for s in range(EV2["seeds"]):
            ctx, msk = enc([q["prompt"] for q in items])
            xs, _, _ = R.rollout(policy, ctx, msk, enc.null_ctx, enc.null_msk, lat_ch, HW, HW, a.eval_steps, a.cfg, a.shift, 0.0,
                                 seed=10_000 + s, chunk=a.rollout_chunk)
            imgs.append(decode_chunked(aem, xs[:, -1], mean, std)); meta += [dict(q, seed=s) for q in items]
        policy.train(); imgs = np.concatenate(imgs)
        T = R2.terms(imgs, [m["prompt"] for m in meta])
        rec = {f"eval_{k}": float(np.mean(v)) for k, v in T.items()}
        for src in sorted({m["source"] for m in meta}):
            sel = np.array([m["source"] == src for m in meta]); rec[f"eval_align_{src}"] = float(T["align"][sel].mean()); rec[f"eval_real_{src}"] = float(T["real"][sel].mean())
        try: rec["eval_clip"] = float(np.mean(metrics.clip_score(imgs, [m["prompt"] for m in meta])))
        except Exception as ex: print(f"  (clip skipped: {str(ex)[:80]})", flush=True)
        if REFS is not None:
            try: rec["eval_fid"], rec["eval_fd_dino"] = metrics.frechet(list(imgs), list(REFS))
            except Exception as ex: print(f"  (fid skipped: {str(ex)[:80]})", flush=True)
        metrics.unload()
        rnd = state["round"]; d = os.path.join(out, "evals", f"{rnd}"); os.makedirs(d, exist_ok=True)
        from PIL import Image
        man = {"round": rnd, "gpu_hours": state["train_secs"] / 3600, "samples": state["samples"], "items": []}
        for i, m in enumerate(meta):
            fn = f"{m['id']}_s{m['seed']}.png"; Image.fromarray(imgs[i].transpose(1, 2, 0)).save(os.path.join(d, fn))
            man["items"].append(dict(file=fn, id=m["id"], prompt=m["prompt"], source=m["source"], seed=m["seed"],
                                     **{k: float(T[k][i]) for k in ("align", "real", "probe", "quality", "anatomy", "style", "px_sat", "px_sharp", "px_contrast", "veto")}))
        json.dump(man, open(os.path.join(d, "manifest.json"), "w"))
        gi = [i for i, m in enumerate(meta) if m["id"] in EV2["grid"] and m["seed"] == EV2.get("grid_seed", 0)]
        grid_png(imgs[gi], [f"{meta[i]['id']}  a{T['align'][i]:.2f} r{T['real'][i]:.2f} q{T['quality'][i]:.1f} n{T['anatomy'][i]:.2f}" for i in gi],
                 os.path.join(out, "evals", f"{rnd:06d}.png"))
        idx_p = os.path.join(out, "evals", "index.json")
        idx = json.load(open(idx_p)) if os.path.exists(idx_p) else []
        idx = [e for e in idx if e["round"] != rnd] + [dict(base_row(), file=f"evals/{rnd:06d}.png", **rec)]
        json.dump(sorted(idx, key=lambda e: e["round"]), open(idx_p, "w"))
        from safetensors.torch import save_file                   # policy weights == EMA weights here when --ema (swapped in)
        save_file({k: v.detach().to(torch.bfloat16).cpu().contiguous() for k, v in policy.state_dict().items()},
                  os.path.join(out, f"policy_{rnd:06d}.safetensors"), metadata={"round": str(rnd), "ema": str(a.ema)})
        rec["eval_secs"] = time.time() - t0; torch.cuda.empty_cache()
        return rec

    def save():
        tmp = ck_path + ".tmp"
        torch.save({"model": policy.state_dict(), "opt": opt.state_dict(), "state": state, "rng": rng.getstate(),
                    "config": vars(a), "ema": ema.state_dict() if ema is not None else None}, tmp)
        os.replace(tmp, ck_path)

    EVAL = evaluate2 if a.reward == "stage2" else evaluate
    if state["round"] == 0:
        rec = EVAL(); log(**base_row(), eval=True, **rec); state["last_eval"] = 0.0
        print("  base eval " + " ".join(f"{k}={v:.3f}" for k, v in rec.items()), flush=True)
    print(f"  {a.alg} / {a.objective}: {P} prompts x {G} = {N} samples per round, {a.steps} steps, cfg {a.cfg}, "
          f"lr {a.lr}, budget {a.hours} GPU-h -> {out}", flush=True)
    wall0 = time.time() - state["wall_secs"]
    warm = a.warmup_rounds
    while state["round"] < a.rounds and state["train_secs"] / 3600 < a.hours and not STOP:
        rnd = state["round"]; dry = warm > 0                       # warm-up: compile every graph, no step, no log
        # ---- rollout: P prompts, G samples each (contiguous groups)
        t0 = sync()
        if a.reward == "stage2": texts = [q for q in rng.sample(TRAIN_PROMPTS, P) for _ in range(G)]
        else: texts = [RW.prompt(x) for x in rng.sample(RW.ANIMALS, P) for _ in range(G)]
        ctx, msk = enc(texts)
        xs, logp, ts = R.rollout(net, ctx, msk, enc.null_ctx, enc.null_msk, lat_ch, HW, HW, a.steps, a.cfg, a.shift,
                                 a.noise_level if a.alg == "grpo" else 0.0, seed=a.seed * 10**6 + (999_999 if dry else rnd), chunk=a.rollout_chunk)
        t1 = sync()
        # ---- reward and group-relative advantage (group mean, global std: both papers' default)
        imgs = decode_chunked(aem, xs[:, -1], mean, std)
        if a.reward == "stage2":
            T = R2.terms(imgs, texts); adv_np, A2 = R2.advantage(T, np.repeat(np.arange(P), G))
            adv = torch.tensor(adv_np * a.adv_sign, device="cuda", dtype=torch.float32)
            kb = np.zeros(N, dtype=np.float32); r = adv; rg = adv.view(P, G)
        else:
            kb = RW.jpeg_kb(imgs, a.jpeg_quality)
            r = torch.tensor(sign * kb, device="cuda")
            rg = r.view(P, G)
            adv = ((rg - rg.mean(1, keepdim=True)) / (r.std() + 1e-4)).flatten() * a.adv_sign
        if a.adv_shuffle:                                           # control: same advantage distribution, random assignment
            perm = torch.cat([torch.randperm(G, device="cuda") + g * G for g in range(P)]); adv = adv[perm]; r = adv; rg = adv.view(P, G)
        t2 = sync()
        # ---- update: every sample used once; one optimizer step per minibatch
        acc, nacc, gns = {}, 0, []
        def bump(st):
            for k, v in st.items(): acc[k] = acc.get(k, 0.0) + float(v)
        if a.alg == "grpo":
            for i in range(0, N, a.minibatch):
                s = slice(i, i + a.minibatch); opt.zero_grad(set_to_none=True); tot = 0.0
                for j in range(a.steps):
                    l, stj = grpo.step_loss(net, ref_net, xs[s], logp[s], adv[s], j, ts, a.noise_level, ctx[s], msk[s],
                                            enc.null_ctx, enc.null_msk, a.cfg, a.clip, a.adv_clip, a.beta)
                    (l / a.steps).backward(); tot += l.item() / a.steps
                    for k, v in stj.items(): acc[k] = acc.get(k, 0.0) + float(v) / a.steps
                acc["loss"] = acc.get("loss", 0.0) + tot; nacc += 1
                gns.append(float(torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)))
                if not dry:
                    opt.step()
                    if ema is not None: ema.update(policy)
        else:
            b = dgpo.prepare(net, ref_net, xs[:, -1], G, ts, a.trunc, ctx, msk, enc.null_ctx, enc.null_msk, a.cfg,
                             seed=a.seed * 10**6 + (999_999 if dry else rnd) + 1, chunk=a.rollout_chunk)
            for i in range(0, N, a.minibatch):
                s = slice(i, i + a.minibatch); opt.zero_grad(set_to_none=True)
                l, stj = dgpo.loss(net, b, s, adv, G, ctx, msk, enc.null_ctx, enc.null_msk, a.cfg,
                                   a.beta_dpo, a.dgpo_clip, a.beta)
                l.backward(); bump(stj); acc["loss"] = acc.get("loss", 0.0) + l.item(); nacc += 1
                gns.append(float(torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)))
                if not dry:
                    opt.step()
                    if ema is not None: ema.update(policy)
        t3 = sync()
        if dry:
            warm -= 1; wall0 = time.time() - state["wall_secs"]
            print(f"    warm-up round done in {t3 - t0:.0f}s (graphs compiled; not counted)", flush=True); continue
        state["round"] += 1; state["samples"] += N; state["train_secs"] += t3 - t0; state["wall_secs"] = time.time() - wall0
        rec = base_row()
        rec.update({k: v / nacc for k, v in acc.items()})
        rec.update(gnorm=float(np.mean(gns)), lr=a.lr, adv_abs=float(adv.abs().mean()), t_rollout=t1 - t0, t_reward=t2 - t1, t_update=t3 - t2, sps=N / (t3 - t0))
        if a.reward == "stage2":
            for k in ("align", "real", "probe", "quality", "anatomy", "style", "px_sat", "px_sharp", "px_contrast", "veto"): rec[k] = float(np.mean(T[k]))
            for k in ("align", "real", "quality", "anatomy"): rec[f"gstd_{k}"] = float(np.mean([np.std(T[k][np.repeat(np.arange(P), G) == g]) for g in range(P)]))
            rec["group_std"] = rec["gstd_align"]; rec["reward"] = float(sum(R2.w[k] * np.mean(T[k]) for k in R2.w))
        else:
            rec.update(kb=float(kb.mean()), kb_std=float(kb.std()), reward=float(r.mean()), group_std=float(rg.std(1).mean()))
        log(**rec)
        head = (f"kB {rec['kb']:.1f}" if a.reward == "jpeg" else f"align {rec['align']:.3f} real {rec['real']:.3f} qual {rec['quality']:.2f} anat {rec['anatomy']:.3f}")
        print(f"    round {rec['round']:>5} {rec['gpu_hours']:.3f} GPU-h  {head} (group std {rec['group_std']:.3f})  "
              f"loss {rec['loss']:.4f}  gn {rec['gnorm']:.2f}  {rec['t_rollout']:.1f}+{rec['t_reward']:.1f}+{rec['t_update']:.1f}s", flush=True)
        if a.reward == "stage2": torch.cuda.empty_cache()
        if (state["train_secs"] - state["last_eval"]) / 60 >= a.eval_every_min:
            ev = EVAL(); state["last_eval"] = state["train_secs"]; log(**base_row(), eval=True, **ev)
            print("    eval " + " ".join(f"{k}={v:.3f}" for k, v in ev.items() if not k.startswith("eval_align_") and not k.startswith("eval_real_")), flush=True)
            if a.stop_fd and ev.get("eval_fd_dino"):
                fd = ev["eval_fd_dino"]; state["fd_best"] = min(state.get("fd_best", fd), fd)
                state["fd_bad"] = state.get("fd_bad", 0) + 1 if fd > state["fd_best"] * a.stop_fd else 0
                if state["fd_bad"] >= a.stop_patience:
                    print(f"[stop] FD-DINOv2 {fd:.0f} above its best {state['fd_best']:.0f} x {a.stop_fd} for {a.stop_patience} evals", flush=True); break
        if (state["train_secs"] - state["last_ckpt"]) / 60 >= a.ckpt_every_min:
            save(); state["last_ckpt"] = state["train_secs"]
    if state["last_eval"] != state["train_secs"]:
        ev = EVAL(); state["last_eval"] = state["train_secs"]; log(**base_row(), eval=True, **ev)
    save()
    if not STOP:
        from safetensors.torch import save_file
        save_file({k: v.detach().to(torch.bfloat16).cpu().contiguous() for k, v in policy.state_dict().items()},
                  os.path.join(out, "policy_final.safetensors"), metadata={"round": str(state["round"]), "config": a.config})
    print(f"done: {state['round']} rounds, {state['samples']:,} samples, {state['train_secs']/3600:.2f} GPU-h", flush=True)


if __name__ == "__main__":
    main()

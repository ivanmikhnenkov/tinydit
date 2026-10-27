"""Sanity checks of the RL machinery on the real model, no training.
  1  gauss_logp == torch.distributions.Normal log-density (averaged over dims)
  2  GRPO: per-step density ratio recomputed under the sampling policy is 1, and how much numerical noise the 1e-4 clip sees
  3  DGPO: policy == reference -> margin 0, gate 0.5, implicit ratio 1, KL 0
  4  DGPO: three optimizer steps with synthetic +-1 advantages lower the denoising loss on positive samples and raise it on negatives
  5  stage-2 advantage: higher term inside a group -> positive advantage; style only centred; vetoed -> group minimum
  6  SDE (a = 0.7) and ODE samples from the same seeds have the same pixel statistics (the marginal-preserving claim)
    PYTHONPATH=src .venv/bin/python scripts/rl/sanity_checks.py [--base ...]   -> out/diag/sanity.json
"""
import argparse, json, os, sys, types
sys.path.insert(0, "src")
import numpy as np, torch
from tinydit import ae as AE
from tinydit.sample import load_model, decode_chunked
from tinydit.rl import rollout as R, dgpo, rewards as RW
from tinydit.rl.rollout import cfg_velocity, step_mean, sde_coeffs, gauss_logp, mse
from tinydit.rl.train import Text
from tinydit.rl.reward_stage2 import Stage2Reward
from tinydit.rl.judges import pixel_stats

ap = argparse.ArgumentParser(); ap.add_argument("--base", default="out/runs/run1/ema_0400000.safetensors"); ap.add_argument("--config", default="run1")
a = ap.parse_args(); torch.manual_seed(0); out = {}
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
def say(k, **kw): out[k] = kw; print(f"[{k}] " + " ".join(f"{n}={v:.3g}" if isinstance(v, float) else f"{n}={v}" for n, v in kw.items()), flush=True)

# 1 ---------------------------------------------------------------------------------------------------------------------------
x = torch.randn(4, 32, 32, 32, device="cuda"); m = torch.randn_like(x); sigma = torch.tensor(0.7, device="cuda"); dt = torch.tensor(0.05, device="cuda")
ref = torch.distributions.Normal(m, sigma * dt.sqrt()).log_prob(x).flatten(1).mean(1)
say("1_gauss_logp", max_abs_err=float((gauss_logp(x, m, sigma, dt) - ref).abs().max()), ok=bool(torch.allclose(gauss_logp(x, m, sigma, dt), ref, atol=1e-5)))

# models ----------------------------------------------------------------------------------------------------------------------
lat_ch = AE.SPECS["flux2"]["ch"]; HW = 32
policy, _ = load_model(a.base, lat_ch, a.config); policy.train()
refm, _ = load_model(a.base, lat_ch, a.config); refm.requires_grad_(False)
enc = Text("out/models"); aem = AE.load("flux2", "out/models/flux2")
st = json.load(open(next(q for q in ("out/cache/run1/stats.json", "out/hf_model/stats.json") if os.path.exists(q))))
mean = torch.tensor(st["mean"], device="cuda").view(1, -1, 1, 1).float(); std = torch.tensor(st["std"], device="cuda").view(1, -1, 1, 1).float()
G = 8; prompts = [RW.prompt(x) for x in RW.ANIMALS[:4] for _ in range(G)]; N = len(prompts)
ctx, msk = enc(prompts); cfg, shift, steps = 4.0, 2.8, 10

# 2 ---------------------------------------------------------------------------------------------------------------------------
xs, logp, ts = R.rollout(policy, ctx, msk, enc.null_ctx, enc.null_msk, lat_ch, HW, HW, steps, cfg, shift, 0.7, seed=1)
rows = []
with torch.no_grad():
    for j in range(steps):
        xj, xn = xs[:, j], xs[:, j + 1]; t = ts[j].expand(N)
        v = cfg_velocity(policy, xj, t, ctx, msk, enc.null_ctx, enc.null_msk, cfg)
        sg, dt = sde_coeffs(ts, j, 0.7); lp = gauss_logp(xn, step_mean(xj, v, ts, j, 0.7), sg, dt)
        ratio = torch.exp(lp - logp[:, j]); rows.append(dict(step=j, ratio_mean=float(ratio.mean()), max_abs_dev=float((ratio - 1).abs().max()), clipfrac_1e4=float(((ratio - 1).abs() > 1e-4).float().mean())))
say("2_grpo_ratio_at_step0", ratio_mean=float(np.mean([r["ratio_mean"] for r in rows])), max_abs_dev=float(max(r["max_abs_dev"] for r in rows)),
    clipfrac_1e4=float(np.mean([r["clipfrac_1e4"] for r in rows])), per_step=rows)

# 3 ---------------------------------------------------------------------------------------------------------------------------
xs0, _, ts = R.rollout(policy, ctx, msk, enc.null_ctx, enc.null_msk, lat_ch, HW, HW, steps, cfg, shift, 0.0, seed=2)
x0 = xs0[:, -1]
b = dgpo.prepare(policy, refm, x0, G, ts, 4, ctx, msk, enc.null_ctx, enc.null_msk, cfg, seed=3)
adv = torch.tensor([1.0] * (G // 2) + [-1.0] * (G // 2), device="cuda").repeat(N // G)
l, s3 = dgpo.loss(policy, b, slice(0, N), adv, G, ctx, msk, enc.null_ctx, enc.null_msk, cfg, 100.0, 1e-2, 0.001)
with torch.no_grad():
    v = cfg_velocity(policy, b["x_t"], b["t"], ctx, msk, enc.null_ctx, enc.null_msk, cfg); dsm = mse(v, b["target"])
    ratio = torch.exp(b["old_dsm"] - dsm)
say("3_dgpo_identity", margin=float(s3["margin"]), gate=float(s3["w"]), ratio_max_abs_dev=float((ratio - 1).abs().max()), clipfrac=float(s3["clipfrac"]), kl=float(s3["kl"]),
    old_minus_ref_dsm_max=float((b["old_dsm"] - b["ref_dsm"]).abs().max()))

# 4 ---------------------------------------------------------------------------------------------------------------------------
opt = torch.optim.AdamW(policy.parameters(), lr=2e-5, betas=(0.9, 0.999), weight_decay=0.0)
hist = []
for it in range(3):
    opt.zero_grad(set_to_none=True)
    l, s = dgpo.loss(policy, b, slice(0, N), adv, G, ctx, msk, enc.null_ctx, enc.null_msk, cfg, 100.0, 1e-2, 0.001)
    l.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
    hist.append({k: float(v) for k, v in s.items()})
with torch.no_grad():
    v = cfg_velocity(policy, b["x_t"], b["t"], ctx, msk, enc.null_ctx, enc.null_msk, cfg); dsm = mse(v, b["target"])
    d = (dsm - b["old_dsm"]); pos, neg = d[adv > 0], d[adv < 0]
    AD = (adv * (dsm - b["ref_dsm"])).view(-1, G).sum(1) / G
say("4_dgpo_direction", dsm_change_positive=float(pos.mean()), dsm_change_negative=float(neg.mean()), frac_pos_down=float((pos < 0).float().mean()), frac_neg_up=float((neg > 0).float().mean()),
    mean_A_delta_per_group=float(AD.mean()), margin_beta100=float(100 * AD.mean()), gate_beta100=float(torch.sigmoid(100 * AD).mean()), gate_beta1000=float(torch.sigmoid(1000 * AD).mean()),
    gate_beta5000=float(torch.sigmoid(5000 * AD).mean()), steps=hist)

# 5 ---------------------------------------------------------------------------------------------------------------------------
self = types.SimpleNamespace(w={"align": 1.0, "style": 0.1})
t5 = {"align": np.array([0.2, 0.9, 0.5, 0.5, 0.1, 0.1, 0.8, 0.8]), "style": np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), "veto": np.array([0, 0, 0, 0, 0, 1, 0, 0], dtype=np.float32)}
advv, A5 = Stage2Reward.advantage(self, t5, np.repeat(np.arange(2), 4))
say("5_advantage", align_high_positive=bool(A5["align"][1] > 0 and A5["align"][0] < 0), group_centred=bool(abs(A5["align"][:4].sum()) < 1e-9 and abs(A5["align"][4:].sum()) < 1e-9),
    style_centred_unscaled=bool(abs(A5["style"][2] - (-0.75)) < 1e-9), veto_is_group_min=bool(advv[5] == min(advv[4:].min(), -2.0)), adv=[round(float(x), 3) for x in advv])

# 6 ---------------------------------------------------------------------------------------------------------------------------
policy.eval(); policy.load_state_dict(refm.state_dict())
P6 = [RW.prompt(x) for x in RW.ANIMALS[:16] for _ in range(4)]; c6, m6 = enc(P6)
res = {}
for name, aa in (("ode", 0.0), ("sde", 0.7)):
    xs6, _, _ = R.rollout(policy, c6, m6, enc.null_ctx, enc.null_msk, lat_ch, HW, HW, steps, cfg, shift, aa, seed=4)
    im = decode_chunked(aem, xs6[:, -1], mean, std); ps = pixel_stats(im)
    res[name] = dict(sat=float(ps["sat"].mean()), sharp=float(ps["sharp"].mean()), contrast=float(ps["contrast"].mean()), jpeg_kb=float(RW.jpeg_kb(im, 95).mean()))
say("6_sde_vs_ode", **{f"{k}_{n}": v for n, r in res.items() for k, v in r.items()})
json.dump(out, open("out/diag/sanity.json", "w"), indent=1); print("-> out/diag/sanity.json")

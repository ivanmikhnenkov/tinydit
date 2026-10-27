"""Collect everything the post-training diagnostics produced into one JSON for the report:
sanity checks, gate / margin / KL histories of the stage-1, stage-2 and diag runs, their evals with the 9B judge merged, pass@k.
    .venv/bin/python scripts/rl/collect_diag.py -> out/diag/report_data.json"""
import base64, glob, json, os
import numpy as np
D = {"sanity": json.load(open("out/diag/sanity.json")) if os.path.exists("out/diag/sanity.json") else None, "runs": {}, "passk": None, "pick": None}
TR = ["w", "margin", "kl", "delta", "clipfrac", "loss_pref", "align", "px_sharp", "px_sat", "gstd_align", "gnorm", "kb", "w_std", "w_lo", "w_hi"]
EVK = ["eval_align", "eval_align_hand_unusual", "eval_align_heldout_real", "eval_align_animal", "eval_align_hand_usual", "eval_real", "eval_probe", "eval_anatomy", "eval_px_sharp", "eval_px_sat",
       "eval_px_contrast", "eval_fd_dino", "eval_fid", "eval_clip", "eval_kb", "j_align", "j_align_hand_unusual", "j_align_heldout_real", "j_real", "j_aislop"]
def sub(xs, n=80):
    if len(xs) <= n: return xs
    idx = np.linspace(0, len(xs) - 1, n).round().astype(int); return [xs[i] for i in idx]
for run in sorted(glob.glob("out/runs/rl_dgpo_*") + glob.glob("out/runs/rl2_dgpo*") + glob.glob("out/runs/diag_*")):
    mp = os.path.join(run, "metrics.jsonl")
    if not os.path.exists(mp): continue
    rows = [json.loads(l) for l in open(mp)]; tr = [r for r in rows if not r.get("eval")]; ev = [r for r in rows if r.get("eval")]
    tr = list({r["round"]: r for r in tr}.values()); ev = list({r["round"]: r for r in ev}.values())       # a resumed run replays a few rounds: keep the last row per round
    if not tr: continue
    jp = os.path.join(run, "judges.jsonl"); J = {json.loads(l)["round"]: json.loads(l) for l in open(jp)} if os.path.exists(jp) else {}
    for e in ev: e.update({k: v for k, v in J.get(e["round"], {}).items() if k.startswith("j_")})
    meta = json.load(open(os.path.join(run, "meta.json"))) if os.path.exists(os.path.join(run, "meta.json")) else {}
    cfg = meta.get("config", {})
    D["runs"][os.path.basename(run)] = {
        "config": {k: cfg.get(k) for k in ("beta_dpo", "beta", "lr", "ema", "adv_sign", "w_align", "w_real", "w_qual", "w_anat", "w_style", "base", "ref", "hours", "rounds", "objective", "reward")},
        "rounds": len(tr), "gpu_hours": tr[-1]["gpu_hours"],
        "train": {k: sub([round(r[k], 5) for r in tr if k in r]) for k in TR if k in tr[0]}, "train_round": sub([r["round"] for r in tr]),
        "train_round_full_last10": {k: float(np.mean([r[k] for r in tr[-10:]])) for k in ("w", "margin", "kl", "delta", "w_std", "w_lo", "w_hi") if k in tr[-1]},
        "margin_mean": float(np.mean([r["margin"] for r in tr])) if "margin" in tr[0] else None, "margin_std": float(np.std([r["margin"] for r in tr])) if "margin" in tr[0] else None,
        "gate_mean": float(np.mean([r["w"] for r in tr])) if "w" in tr[0] else None,
        "train_round_full_first10": {k: float(np.mean([r[k] for r in tr[:10]])) for k in ("w", "margin", "kl", "delta") if k in tr[0]},
        "evals": [{"round": e["round"], "gpu_hours": round(e["gpu_hours"], 3), **{k: round(e[k], 4) for k in EVK if k in e}} for e in ev]}
if os.path.exists("out/passk/summary.json"):
    D["passk"] = json.load(open("out/passk/summary.json")); D["passk"]["grids"] = {}
    for f in sorted(glob.glob("out/passk/grids/*.jpg")):
        D["passk"]["grids"][os.path.basename(f)[:-4]] = base64.b64encode(open(f, "rb").read()).decode()
if os.path.exists("out/passk_long/summary.json"):
    D["passk_long"] = json.load(open("out/passk_long/summary.json"))
if os.path.exists("out/diag/prompt_band.json"):
    B = json.load(open("out/diag/prompt_band.json")); c = np.array([q["base_pass"] for q in B])
    D["band"] = {"n": len(B), "hist": np.bincount(c, minlength=9).tolist(), "kept": int(((c >= 1) & (c <= 6)).sum()),
                 "per_source": {s: {"n": int(sum(q["source"] == s for q in B)), "kept": int(sum(q["source"] == s and 1 <= q["base_pass"] <= 6 for q in B))} for s in sorted({q["source"] for q in B})}}
if os.path.exists("out/passk/bias_base.json"): D["bias"] = json.load(open("out/passk/bias_base.json"))
if os.path.exists("out/diag/pick_beta.json"):
    try: D["pick"] = json.load(open("out/diag/pick_beta.json"))
    except Exception: pass
json.dump(D, open("out/diag/report_data.json", "w"))
print(f"runs: {list(D['runs'])}; passk: {bool(D['passk'])}; grids: {len(D['passk']['grids']) if D['passk'] else 0}; {os.path.getsize('out/diag/report_data.json')/1e6:.1f} MB")

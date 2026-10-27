"""Per-eval table of a stage-2 run with the post-hoc judges merged, and a checkpoint pick:
best 9B alignment among rounds whose 9B "AI look?" is not worse than the baseline's and whose FD-DINOv2 is within 1 % of the run's best.
    .venv/bin/python scripts/rl/summarize_run.py --run rl2_dgpo_v5 [--chain rl2_dgpo,rl2_dgpo_v2,...]"""
import argparse, json, os
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--chain", default="")
a = ap.parse_args()
def load(run):
    rows = [json.loads(l) for l in open(f"out/runs/{run}/metrics.jsonl")]; ev = [r for r in rows if r.get("eval")]
    jp = f"out/runs/{run}/judges.jsonl"; J = {json.loads(l)["round"]: json.loads(l) for l in open(jp)} if os.path.exists(jp) else {}
    for e in ev: e.update(J.get(e["round"], {})); e["run"] = run
    return ev
evs = sum([load(r) for r in a.chain.split(",") if r], []) + load(a.run)
cols = [("round", 0, 5), ("gpu_hours", 2, 6), ("eval_align", 3, 7), ("j_align", 3, 7), ("eval_align_hand_unusual", 3, 7), ("j_align_hand_unusual", 3, 7), ("eval_real", 3, 7), ("j_real", 3, 7), ("j_aislop", 3, 7),
        ("eval_anatomy", 3, 7), ("eval_quality", 2, 6), ("eval_px_sat", 3, 6), ("eval_fid", 1, 6), ("eval_fd_dino", 0, 6), ("eval_clip", 3, 6)]
print("run        " + " ".join(f"{k.replace('eval_','').replace('hand_','')[:w]:>{w}s}" for k, _, w in cols))
for e in evs:
    print(f"{e['run'][-10:]:10s} " + " ".join((f"{e[k]:{w}.{d}f}" if isinstance(e.get(k), (int, float)) else f"{'—':>{w}s}") for k, d, w in cols))
base = evs[0]; run_ev = [e for e in evs if e["run"] == a.run and e.get("j_align") is not None]
if run_ev:
    fd_best = min(e["eval_fd_dino"] for e in run_ev if e.get("eval_fd_dino"))
    ok = [e for e in run_ev if e.get("j_aislop", 1) <= base.get("j_aislop", 1) + 0.01 and e.get("eval_fd_dino", 1e9) <= fd_best * 1.01]
    pick = max(ok or run_ev, key=lambda e: e["j_align"])
    print(f"\npick: {a.run} round {pick['round']} ({pick['gpu_hours']:.2f} h): 9B align {pick['j_align']:.3f} (base {base.get('j_align', float('nan')):.3f}), "
          f"9B AI-look {pick.get('j_aislop', float('nan')):.3f} (base {base.get('j_aislop', float('nan')):.3f}), FD {pick['eval_fd_dino']:.0f} (best {fd_best:.0f})  -> out/runs/{a.run}/policy_{pick['round']:06d}.safetensors")

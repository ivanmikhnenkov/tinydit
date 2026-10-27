"""Post-hoc heavy judges over a run's evaluation images (they do not fit beside training): Qwen3.5-9B yes/no questions
("shows the prompt?", "real photograph?", "AI look?") on every eval round not yet judged, and optionally Qwen3.8-27B
error descriptions on the 16-prompt grid. Appends rows to out/runs/<run>/judges.jsonl, which the dashboard merges by round.
    .venv/bin/python scripts/rl/judge_evals.py --run rl2_dgpo [--describe]
"""
import argparse, json, os, sys, time
sys.path.insert(0, "src")
import numpy as np
from PIL import Image
from tinydit.rl import judges as JD
ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--judge", default="qwen35_9b")
ap.add_argument("--describe", action="store_true", help="also run Qwen3.8-27B descriptions on the grid prompts"); ap.add_argument("--batch", type=int, default=8)
a = ap.parse_args(); out = os.path.join("out", "runs", a.run); jp = os.path.join(out, "judges.jsonl")
done = {json.loads(l)["round"] for l in open(jp)} if os.path.exists(jp) else set()
rounds = sorted(int(d) for d in os.listdir(os.path.join(out, "evals")) if d.isdigit() and os.path.exists(os.path.join(out, "evals", d, "manifest.json")))
todo = [r for r in rounds if r not in done]
print(f"{a.run}: {len(rounds)} evals, {len(todo)} to judge", flush=True)
if not todo: sys.exit(0)
vlm = JD.VLM(a.judge); big = JD.VLM("qwen38_27b") if a.describe else None
EV = json.load(open(json.load(open(os.path.join(out, "meta.json")))["config"]["eval_set"]))
for r in todo:
    t0 = time.time(); d = os.path.join(out, "evals", str(r)); man = json.load(open(os.path.join(d, "manifest.json")))
    imgs = np.stack([np.asarray(Image.open(os.path.join(d, it["file"])).convert("RGB")).transpose(2, 0, 1) for it in man["items"]])
    prompts = [it["prompt"] for it in man["items"]]
    q = {k: vlm.yesno(imgs, k, prompts, a.batch) for k in ("align", "real", "aislop")}
    rec = dict(round=r, gpu_hours=man["gpu_hours"], samples=man["samples"], judge=a.judge,
               **{f"j_{k}": float(v.mean()) for k, v in q.items()})
    for src in sorted({it["source"] for it in man["items"]}):
        sel = np.array([it["source"] == src for it in man["items"]]); rec[f"j_align_{src}"] = float(q["align"][sel].mean())
    for i, it in enumerate(man["items"]): it.update({f"j_{k}": float(v[i]) for k, v in q.items()})
    if big is not None:
        gi = [i for i, it in enumerate(man["items"]) if it["id"] in EV["grid"] and it["seed"] == EV.get("grid_seed", 0)]
        texts = big.generate(imgs[gi], "describe", [prompts[i] for i in gi], max_new=100, batch=4)
        for i, t in zip(gi, texts): man["items"][i]["describe_27b"] = t
        rec["j_describe_flagged"] = float(np.mean([not (t.strip().lower().startswith("none") or " no " in t.lower()[:120]) for t in texts]))
    json.dump(man, open(os.path.join(d, "manifest.json"), "w"))
    open(jp, "a").write(json.dumps(rec) + "\n")
    print(f"  round {r}: " + " ".join(f"{k}={v:.3f}" for k, v in rec.items() if isinstance(v, float) and not k.startswith('j_align_')) + f"  ({time.time()-t0:.0f}s)", flush=True)

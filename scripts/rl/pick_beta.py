"""Pick beta_dpo for the long A/B from the 40-round sweep: the arm whose end-of-run gate is closest to 0.25 on a log scale
(open enough to keep learning, closed enough to show the preference mechanism works). If beta 100 is the closest, extrapolate
beta = 100 * ln 3 / |margin_100| (sigmoid(-ln 3) = 0.25), clipped to [300, 5000], rounded to 100."""
import json, math, sys
import numpy as np
arms = {}
for b in (100, 1000, 5000):
    rows = [json.loads(l) for l in open(f"out/runs/diag_beta{b}/metrics.jsonl")]; tr = [r for r in rows if not r.get("eval")]
    arms[b] = (float(np.mean([r["w"] for r in tr[-10:]])), float(np.mean([r["margin"] for r in tr[-10:]])))
best = min(arms, key=lambda b: abs(math.log(max(arms[b][0], 1e-4)) - math.log(0.25)))
if best == 100:
    best = int(round(min(5000, max(300, 100 * math.log(3) / max(abs(arms[100][1]), 1e-6))) / 100) * 100)
print(json.dumps({"arms": {str(b): {"gate": w, "margin": m} for b, (w, m) in arms.items()}, "pick": best}), file=sys.stderr); print(best)

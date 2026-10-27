"""Hard-negative alignment test with the cheap scorers: each hand-prompt sample scored with its own prompt and with two
minimally edited prompts (subject or action swapped). Writes out/reward_eval/scores/cheap_hardneg.json {file: {own:{...}, neg:[{...},{...}]}}."""
import json, sys
sys.path.insert(0, "src")
import numpy as np
from PIL import Image
from tinydit import metrics
E = "out/reward_eval"; M = json.load(open(f"{E}/manifest.json")); H = json.load(open("prompts/rl_eval/set_v1.json"))["hard_negatives"]
items = [s for s in M["samples"] if s["id"] in H]
imgs = np.stack([np.asarray(Image.open(f"{E}/{s['file']}").convert("RGB")).transpose(2, 0, 1) for s in items])
own = [s["prompt"] for s in items]; negs = [[H[s["id"]][0] for s in items], [H[s["id"]][1] for s in items]]
R = {}
sc = {"own": {k: fn(imgs, own) for k, fn in (("clip", metrics.clip_score), ("pickscore", metrics.pickscore), ("hpsv2", metrics.hpsv2))}}
for j in range(2): sc[f"neg{j}"] = {k: fn(imgs, negs[j]) for k, fn in (("clip", metrics.clip_score), ("pickscore", metrics.pickscore), ("hpsv2", metrics.hpsv2))}
for i, s in enumerate(items):
    R[s["file"]] = {"own": {k: float(v[i]) for k, v in sc["own"].items()}, "neg": [{k: float(v[i]) for k, v in sc[f"neg{j}"].items()} for j in range(2)], "neg_prompts": H[s["id"]]}
json.dump(R, open(f"{E}/scores/cheap_hardneg.json", "w")); print("hardneg scored:", len(R))

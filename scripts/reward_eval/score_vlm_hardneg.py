"""Hard-negative alignment test for a VLM judge: P(yes) on the own prompt vs on two minimally edited prompts.
Writes out/reward_eval/scores/<model>_hardneg.json in the same shape as cheap_hardneg.json.
    .venv/bin/python scripts/reward_eval/score_vlm_hardneg.py --model qwen35_4b"""
import argparse, json, os, sys
sys.path.insert(0, "scripts/reward_eval"); sys.path.insert(0, "src")
from score_vlm import Judge, Q, img448, E
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--batch", type=int, default=16); a = ap.parse_args()
M = json.load(open(f"{E}/manifest.json")); H = json.load(open("prompts/rl_eval/set_v1.json"))["hard_negatives"]
items = [s for s in M["samples"] if s["id"] in H]
jd = Judge(a.model); R = {}
for k in range(0, len(items), a.batch):
    ch = items[k:k + a.batch]; ims = [img448(s["file"]) for s in ch]
    if jd.kind == "hpsv3":
        key = "hps"; own = jd.hps(ims, [s["prompt"] for s in ch]); negs = [jd.hps(ims, [H[s["id"]][j] for s in ch]) for j in range(2)]
    else:
        key = "align"; own = jd.yesno(ims, [Q["align"].format(prompt=s["prompt"]) for s in ch])
        negs = [jd.yesno(ims, [Q["align"].format(prompt=H[s["id"]][j]) for s in ch]) for j in range(2)]
    for i, s in enumerate(ch):
        R[s["file"]] = {"own": {key: float(own[i])}, "neg": [{key: float(negs[j][i])} for j in range(2)], "neg_prompts": H[s["id"]]}
json.dump(R, open(f"{E}/scores/{a.model}_hardneg.json", "w")); print(f"{a.model} hardneg scored: {len(R)}")

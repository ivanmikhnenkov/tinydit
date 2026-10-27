"""Re-score one evaluation round's images with the current realism probe and the 9B judge (align / real / AI look), so rounds
from different runs are comparable. Writes <evals>/<round>/rescored.json {file: {real_cur, j_align, j_real, j_aislop}}.
    PYTHONPATH=src .venv/bin/python scripts/rl/rescore_eval.py out/runs/rl2_dgpo/evals/0 [more dirs...]"""
import json, os, sys, numpy as np
sys.path.insert(0, "src")
from PIL import Image
from tinydit.rl import judges as JD
dirs = sys.argv[1:]; vlm = JD.VLM("qwen35_9b"); probe = JD.DinoProbe()
for d in dirs:
    man = json.load(open(os.path.join(d, "manifest.json"))); items = man["items"]
    imgs = np.stack([np.asarray(Image.open(os.path.join(d, it["file"])).convert("RGB")).transpose(2, 0, 1) for it in items]); prompts = [it["prompt"] for it in items]
    out = {"real_cur": probe(imgs)}
    for k in ("align", "real", "aislop"): out["j_" + k] = vlm.yesno(imgs, k, prompts, 8)
    json.dump({it["file"]: {k: float(v[i]) for k, v in out.items()} for i, it in enumerate(items)}, open(os.path.join(d, "rescored.json"), "w"))
    print(f"{d}: {len(items)} images  " + "  ".join(f"{k} {float(np.mean(v)):.3f}" for k, v in out.items()), flush=True)

"""Download judge/reward models from the Hub with curl (Python HTTP hangs on this host: IPv6).
Resumable, sequential, priority order. Skips files that already have the right size.
    .venv/bin/python scripts/reward_eval/download_judges.py [names...]
"""
import json, os, subprocess, sys, time
sys.path.insert(0, "src")
from tinydit.ae import hf_token
REPOS = {"hpsv3": "MizzenAI/HPSv3", "qwen3vl8b": "Qwen/Qwen3-VL-8B-Instruct", "qwen35_4b": "Qwen/Qwen3.5-4B",
         "qwen35_9b": "Qwen/Qwen3.5-9B", "qwen36_35b": "Qwen/Qwen3.6-35B-A3B", "qwen38_27b": "Qwen/Qwen3.8-27B"}
SKIP_EXT = (".md", ".png", ".jpg", ".gif", ".gitattributes")
tok = hf_token()
def api(repo):
    out = subprocess.run(["curl", "-sL", "--fail", "-m", "60", f"https://huggingface.co/api/models/{repo}?blobs=true"], capture_output=True, text=True, check=True).stdout
    return json.loads(out)
for name in (sys.argv[1:] or ["hpsv3", "qwen3vl8b", "qwen35_4b", "qwen35_9b", "qwen36_35b"]):
    repo = REPOS[name]; d = os.path.join("out/models/judges", name); os.makedirs(d, exist_ok=True)
    files = [(s["rfilename"], s.get("size") or 0) for s in api(repo)["siblings"] if not s["rfilename"].endswith(SKIP_EXT)]
    tot = sum(sz for _, sz in files); t0 = time.time()
    print(f"== {name} ({repo}): {len(files)} files, {tot/1e9:.1f} GB", flush=True)
    for fn, sz in files:
        out = os.path.join(d, fn); os.makedirs(os.path.dirname(out), exist_ok=True)
        if os.path.exists(out) and os.path.getsize(out) == sz: continue
        url = f"https://huggingface.co/{repo}/resolve/main/{fn}"
        cmd = ["curl", "-sL", "--fail", "-C", "-", "-o", out] + (["-H", f"Authorization: Bearer {tok}"] if tok else []) + [url]
        for attempt in range(5):
            r = subprocess.run(cmd)
            if r.returncode == 0 and os.path.getsize(out) == sz: break
            time.sleep(10)
        else:
            print(f"  FAILED {fn}", flush=True); continue
        print(f"  {fn} {sz/1e9:.2f} GB  ({(time.time()-t0)/60:.1f} min)", flush=True)
    print(f"== {name} done in {(time.time()-t0)/60:.1f} min", flush=True)

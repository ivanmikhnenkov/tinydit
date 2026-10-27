"""~1,500 FLUX-Reason Aesthetics renders at 256x256 (centre crop) for the real-vs-render probe, from one parquet shard,
same clarity/structure filter as the pretraining ingest. -> out/data/refs/flux_256.npz"""
import io, os, sys
sys.path.insert(0, "src")
import numpy as np
from PIL import Image
from tinydit.ingest import _curl, _token, HF
import pyarrow.parquet as pq, subprocess, json as _json
def _flux_shards():   # via curl: Python HTTP hangs on this host (IPv6)
    out = []
    for part in ("Aesthetics-Part01", "Aesthetics-Part02"):
        hdr = ["-H", f"Authorization: Bearer {_token()}"] if _token() else []
        r = subprocess.run(["curl", "-sL", "--fail", "-m", "120", *hdr, f"{HF}/api/datasets/LucasFang/FLUX-Reason-6M/tree/main/{part}?recursive=true"], capture_output=True, text=True, check=True)
        out += sorted(x["path"] for x in _json.loads(r.stdout) if x["path"].endswith(".parquet"))
    return out
want = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
shards = _flux_shards(); tmp = "out/data/refs/_flux_pq"; os.makedirs(tmp, exist_ok=True)
imgs, caps = [], []
for rel in shards[::max(1, len(shards) // 3)][:3]:
    local = os.path.join(tmp, os.path.basename(rel)); _curl(f"{HF}/datasets/LucasFang/FLUX-Reason-6M/resolve/main/{rel}", local, _token())
    pf = pq.ParquetFile(local)
    for rg in range(pf.num_row_groups):
        tb = pf.read_row_group(rg, columns=["image", "caption_entity", "score_image_clarity", "score_image_structure"]).to_pydict()
        for j in range(len(tb["image"])):
            if (tb["score_image_clarity"][j] or 0) < 9 or (tb["score_image_structure"][j] or 0) < 9: continue
            im = tb["image"][j]; b = im["bytes"] if isinstance(im, dict) else im
            pil = Image.open(io.BytesIO(b)).convert("RGB"); w, h = pil.size; s = min(w, h)
            pil = pil.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((256, 256), Image.LANCZOS)
            imgs.append(np.asarray(pil).transpose(2, 0, 1)); caps.append(tb["caption_entity"][j] or "")
            if len(imgs) >= want: break
        if len(imgs) >= want: break
    os.remove(local); print(f"  {rel}: total {len(imgs)}", flush=True)
    if len(imgs) >= want: break
np.savez("out/data/refs/flux_256.npz", images=np.stack(imgs), caption=np.array(caps)); print("flux refs", len(imgs), flush=True)

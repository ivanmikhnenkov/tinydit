"""Dataset download and caching.

Everything is precomputed once: images are decoded, centre-cropped, resized and
stored as uint8; latents and T5 embeddings are then derived from that. Training
reads only the cached arrays, so there is no dataloader in the training loop at
all — the whole of CUB fits in VRAM.

Layout under out/cache/<dataset>/:
  images_<res>.npy   uint8   (N, 3, res, res)
  captions.json      {"caps": [[str, ...], ...]}   per image
  lat_<ae>_<res>.npy float16 (N, C, res/8, res/8)
  lat_<ae>_<res>.json  {"mean": [...], "std": [...]}  per-channel whitening stats
  t5_<len>.npy       float16 (M, L, 768)   M = total captions across all images
  t5_<len>_mask.npy  bool    (M, L)
  t5_<len>_owner.npy int32   (M,)          caption index -> image index
"""
from __future__ import annotations
import io, json, os, subprocess, sys

HF = "https://huggingface.co/datasets"
DATASETS = {
    "cub": dict(
        parts=[f"{HF}/cassiekang/cub200_dataset/resolve/refs%2Fconvert%2Fparquet/default/{s}/0000.parquet"
               for s in ("train", "test")],
        image_col="image", caption_cols=["text"]),
}


def _curl(url: str, out: str, token: str | None = None):
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cmd = ["curl", "-sL", "--fail", "-o", out]
    if token:
        cmd += ["-H", f"Authorization: Bearer {token}"]
    subprocess.run(cmd + [url], check=True)
    print(f"  fetched {os.path.basename(out)} ({os.path.getsize(out)/1e6:.1f} MB)", flush=True)
    return out


def prepare(name: str, root: str, res_list=(256, 128), token: str | None = None):
    """parquet -> images_<res>.npy + captions.json"""
    import numpy as np, pyarrow.parquet as pq
    from PIL import Image
    spec = DATASETS[name]
    d = os.path.join(root, name); os.makedirs(d, exist_ok=True)
    raw = os.path.join(d, "raw")
    files = [_curl(u, os.path.join(raw, f"part{i}.parquet"), token)
             for i, u in enumerate(spec["parts"])]

    caps_all, n = [], 0
    tables = [pq.read_table(f) for f in files]
    n = sum(t.num_rows for t in tables)
    print(f"  {n} rows across {len(tables)} parquet files", flush=True)
    mm = {r: np.lib.format.open_memmap(os.path.join(d, f"images_{r}.npy"), mode="w+",
                                       dtype=np.uint8, shape=(n, 3, r, r)) for r in res_list}
    i = 0
    for t in tables:
        cols = {c: t.column(c).to_pylist() for c in [spec["image_col"], *spec["caption_cols"]]}
        for k in range(t.num_rows):
            rec = cols[spec["image_col"]][k]
            b = rec["bytes"] if isinstance(rec, dict) else rec
            im = Image.open(io.BytesIO(b)).convert("RGB")
            w, h = im.size; s = min(w, h)
            im = im.crop(((w-s)//2, (h-s)//2, (w-s)//2+s, (h-s)//2+s))
            for r in res_list:
                mm[r][i] = np.asarray(im.resize((r, r), Image.LANCZOS)).transpose(2, 0, 1)
            cs = []
            for c in spec["caption_cols"]:
                v = cols[c][k]
                cs += [x for x in (v if isinstance(v, list) else [v]) if isinstance(x, str) and x.strip()]
            caps_all.append(cs)
            i += 1
            if i % 2000 == 0:
                print(f"    {i}/{n}", flush=True)
    for r in res_list:
        mm[r].flush()
    json.dump({"caps": caps_all}, open(os.path.join(d, "captions.json"), "w"))
    print(f"  wrote images_{list(res_list)} and {len(caps_all)} caption lists", flush=True)


def latents(name: str, root: str, ae_name: str, res: int, batch: int = 32, models="out/models"):
    """images_<res>.npy -> lat_<ae>_<res>.npy (+ per-channel whitening stats)"""
    import numpy as np, torch
    from . import ae as AE
    d = os.path.join(root, name)
    imgs = np.load(os.path.join(d, f"images_{res}.npy"), mmap_mode="r")
    n = imgs.shape[0]
    m = AE.load(ae_name, os.path.join(models, ae_name))
    C, L = AE.SPECS[ae_name]["ch"], res // AE.SPECS[ae_name]["f"]
    out = np.lib.format.open_memmap(os.path.join(d, f"lat_{ae_name}_{res}.npy"), mode="w+",
                                    dtype=np.float16, shape=(n, C, L, L))
    tot = torch.zeros(C, dtype=torch.float64); sq = torch.zeros(C, dtype=torch.float64); cnt = 0
    for i in range(0, n, batch):
        x = torch.from_numpy(np.ascontiguousarray(imgs[i:i+batch])).float().cuda() / 127.5 - 1.0
        z = AE.encode(m, x).float()
        out[i:i+batch] = z.cpu().numpy().astype(np.float16)
        tot += z.sum(dim=(0, 2, 3)).double().cpu(); sq += (z**2).sum(dim=(0, 2, 3)).double().cpu()
        cnt += z.shape[0] * z.shape[2] * z.shape[3]
        if i % (batch*50) == 0: print(f"    {i}/{n}", flush=True)
    out.flush()
    mean = (tot/cnt); std = (sq/cnt - mean**2).clamp_min(1e-8).sqrt()
    json.dump({"mean": mean.tolist(), "std": std.tolist()},
              open(os.path.join(d, f"lat_{ae_name}_{res}.json"), "w"), indent=1)
    print(f"  latents {out.shape} | per-channel std range "
          f"{std.min():.3f}-{std.max():.3f}", flush=True)


def text(name: str, root: str, max_len: int = 32, batch: int = 256, models="out/models"):
    """captions.json -> flat T5 embedding cache + owner index"""
    import numpy as np, torch
    from . import text as T
    d = os.path.join(root, name)
    caps = json.load(open(os.path.join(d, "captions.json")))["caps"]
    flat, owner = [], []
    for i, cs in enumerate(caps):
        for c in cs:
            flat.append(c); owner.append(i)
    flat.append("")                                  # index -1: the null caption for CFG
    owner.append(-1)
    tok, enc = T.load(os.path.join(models, "t5"))
    m = len(flat)
    emb = np.lib.format.open_memmap(os.path.join(d, f"t5_{max_len}.npy"), mode="w+",
                                    dtype=np.float16, shape=(m, max_len, T.D_MODEL))
    msk = np.lib.format.open_memmap(os.path.join(d, f"t5_{max_len}_mask.npy"), mode="w+",
                                    dtype=bool, shape=(m, max_len))
    trunc = 0
    for i in range(0, m, batch):
        chunk = flat[i:i+batch]
        e, k = T.embed(tok, enc, chunk, max_len)
        emb[i:i+batch] = e.float().cpu().numpy().astype(np.float16)
        msk[i:i+batch] = k.cpu().numpy()
        trunc += int((k.sum(1) == max_len).sum())
        if i % (batch*20) == 0: print(f"    {i}/{m}", flush=True)
    emb.flush(); msk.flush()
    np.save(os.path.join(d, f"t5_{max_len}_owner.npy"), np.asarray(owner, dtype=np.int32))
    print(f"  {m} captions -> {emb.shape} | {trunc} hit the {max_len}-token cap "
          f"({100*trunc/m:.1f}%)", flush=True)


def check(name: str, root: str, ae_name: str, res: int, k: int = 8, models="out/models"):
    """Decode cached latents back to PNG so the cache can be eyeballed."""
    import numpy as np, torch
    from PIL import Image
    from . import ae as AE
    d = os.path.join(root, name)
    lat = np.load(os.path.join(d, f"lat_{ae_name}_{res}.npy"), mmap_mode="r")
    imgs = np.load(os.path.join(d, f"images_{res}.npy"), mmap_mode="r")
    caps = json.load(open(os.path.join(d, "captions.json")))["caps"]
    idx = np.linspace(0, lat.shape[0]-1, k).astype(int)
    m = AE.load(ae_name, os.path.join(models, ae_name))
    z = torch.from_numpy(np.ascontiguousarray(lat[idx])).float().cuda()
    y = ((AE.decode(m, z)+1)*127.5).clamp(0, 255).byte().cpu().numpy()
    strip = np.zeros((2*res, k*res, 3), np.uint8)
    for j in range(k):
        strip[:res, j*res:(j+1)*res] = imgs[idx[j]].transpose(1, 2, 0)
        strip[res:, j*res:(j+1)*res] = y[j].transpose(1, 2, 0)
    p = os.path.join(d, f"check_{ae_name}_{res}.png")
    Image.fromarray(strip).save(p)
    err = np.abs(imgs[idx].astype(np.float32) - y.astype(np.float32)).mean()
    print(f"  wrote {p} (top: original, bottom: decoded) | mean abs err {err:.2f}/255")
    for j in idx[:4]: print(f"    [{j}] {caps[j][0][:70]}")


def prepare_coco(root: str, res_list=(256, 128), workers: int = 24):
    """COCO ships as a zip of JPEGs plus a JSON of annotations, not parquet.

    Decoding 123k JPEGs is the slow part, so it runs on a thread pool writing
    straight into the memmaps (PIL releases the GIL during decode)."""
    import json as _json, zipfile
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np
    from PIL import Image
    d = os.path.join(root, "coco"); raw = os.path.join(d, "raw")
    with zipfile.ZipFile(os.path.join(raw, "annotations.zip")) as z:
        ann = _json.loads(z.read("annotations/captions_train2017.json"))
    caps_by_id = {}
    for a in ann["annotations"]:
        caps_by_id.setdefault(a["image_id"], []).append(a["caption"].strip())
    imgs = [im for im in ann["images"] if im["id"] in caps_by_id]
    imgs.sort(key=lambda x: x["id"])
    n = len(imgs)
    print(f"  {n} images with captions; "
          f"{sum(len(v) for v in caps_by_id.values())} captions total", flush=True)

    mm = {r: np.lib.format.open_memmap(os.path.join(d, f"images_{r}.npy"), mode="w+",
                                       dtype=np.uint8, shape=(n, 3, r, r)) for r in res_list}
    zf = zipfile.ZipFile(os.path.join(raw, "train2017.zip"))
    lock = __import__("threading").Lock()
    done = [0]

    def work(args):
        i, rec = args
        try:
            with lock:
                b = zf.read("train2017/" + rec["file_name"])
            im = Image.open(io.BytesIO(b)).convert("RGB")
            w, h = im.size; s = min(w, h)
            im = im.crop(((w-s)//2, (h-s)//2, (w-s)//2+s, (h-s)//2+s))
            for r in res_list:
                mm[r][i] = np.asarray(im.resize((r, r), Image.LANCZOS)).transpose(2, 0, 1)
        except Exception as ex:
            print(f"    !! {rec['file_name']}: {ex}", flush=True)
        done[0] += 1
        if done[0] % 10000 == 0:
            print(f"    {done[0]}/{n}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, enumerate(imgs)))
    for r in res_list:
        mm[r].flush()
    caps = [caps_by_id[im["id"]][:5] for im in imgs]
    _json.dump({"caps": caps}, open(os.path.join(d, "captions.json"), "w"))
    print(f"  wrote images_{list(res_list)} and {len(caps)} caption lists "
          f"(mean {sum(len(c) for c in caps)/len(caps):.2f} per image)", flush=True)


HF_PQ = ("https://huggingface.co/datasets/gmongaras/CC12M_and_Imagenet21K_Recap"
         "/resolve/main/{}")

# 384^2-area buckets. Every dim divisible by 8 (the AE factor) and each latent side even
# (the patchify stride), so RoPE sees a clean grid in all five shapes.
BUCKETS = [("1_1", 384, 384), ("4_3", 448, 336), ("3_4", 336, 448),
           ("3_2", 480, 320), ("2_3", 320, 480)]
BUCKET_SHARE = {"1_1": .24, "4_3": .18, "3_4": .11, "3_2": .37, "2_3": .10}


def _bucket_for(w, h):
    import math
    ar = w / max(h, 1)
    return min(BUCKETS, key=lambda b: abs(math.log(ar / (b[1] / b[2]))))


def ingest_parquet(root: str, name: str = "cc12m", ae_name: str = "flux2",
                   res_list=(256, 128), target: int = 1_000_000, target_384: int = 160_000,
                   target_128: int = 320_000,
                   first_file: int = 5100, last_file: int = 8699,
                   models: str = "out/models", batch: int = 128, workers: int = 32,
                   files_list: str = "out/pq/files.txt"):
    """Stream CC12M parquet shards through the autoencoder, keeping only latents.

    Shards are taken with a stride across the CC12M half (rows 13.16M+) so the subset is
    not a narrow slice of the corpus. Each shard is downloaded, encoded and deleted, so
    peak extra disk is ~2 files. Resumable by shard position.
    """
    import io, queue, subprocess, threading
    import numpy as np, torch, pyarrow.parquet as pqt
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    from . import ae as AE

    d = os.path.join(root, name); os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, "_pq"); os.makedirs(tmp, exist_ok=True)
    prog_p = os.path.join(d, "progress.json")

    allf = [l.strip() for l in open(files_list) if l.strip()]
    pool = allf[first_file:last_file + 1]
    need = int(target / 2400) + 80
    stride = max(1, len(pool) // need)
    shards = pool[::stride][:need]

    prog = json.load(open(prog_p)) if os.path.exists(prog_p) else {
        "done": [], "rows": 0, "b": {k: 0 for k, _, _ in BUCKETS}}
    done_set = set(prog.get("done", range(prog.get("i", 0))))
    C = AE.SPECS[ae_name]["ch"]
    mode = lambda p: "r+" if os.path.exists(p) else "w+"
    mm = {}
    for r in res_list:
        p_ = os.path.join(d, f"lat_{ae_name}_{r}.npy")
        mm[r] = np.lib.format.open_memmap(p_, mode=mode(p_), dtype=np.float16,
                                          shape=(target, C, r // 8, r // 8))
    bmm, bcap = {}, {}
    for k, W, H in BUCKETS:
        n = int(target_384 * BUCKET_SHARE[k] * 1.3)
        bcap[k] = n
        p_ = os.path.join(d, f"bucket_{ae_name}_{k}.npy")
        bmm[k] = np.lib.format.open_memmap(p_, mode=mode(p_), dtype=np.float16,
                                           shape=(n, C, H // 8, W // 8))
    cap_f = open(os.path.join(d, "captions.jsonl"), "a")
    bcap_f = {k: open(os.path.join(d, f"captions_{k}.jsonl"), "a") for k, _, _ in BUCKETS}
    m = AE.load(ae_name, os.path.join(models, ae_name))
    tot = torch.zeros(C, dtype=torch.float64); sq = torch.zeros(C, dtype=torch.float64); cnt = 0

    # Downloads dominate once encoding is trimmed, so fetch several shards at once.
    # Queue depth bounds disk: ~0.5 GB a shard, 5 in flight is ~2.5 GB.
    q: queue.Queue = queue.Queue(maxsize=5)
    stop = threading.Event()          # lets producers exit once the consumer has enough
    def producer(items, n_workers=5):
        from concurrent.futures import ThreadPoolExecutor
        def one(it):
            if stop.is_set(): return
            idx, rel = it
            p_ = os.path.join(tmp, os.path.basename(rel))
            for attempt in range(3):
                try:
                    if not (os.path.exists(p_) and os.path.getsize(p_) > 1_000_000):
                        subprocess.run(["curl", "-sL", "--fail", "--retry", "2",
                                        "-o", p_, HF_PQ.format(rel)], check=True)
                    while not stop.is_set():
                        try: q.put((idx, p_), timeout=5); break
                        except queue.Full: continue
                    return
                except Exception as ex:
                    if attempt == 2:
                        print("    !! fetch " + rel + ": " + str(ex)[:60], flush=True)
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            list(ex.map(one, items))
        q.put((None, None))

    todo = [(i, r) for i, r in enumerate(shards) if i not in done_set]
    threading.Thread(target=producer, args=(todo,), daemon=True).start()
    rows = prog["rows"]; bn = dict(prog["b"])
    print(f"  {len(shards)} shards selected (stride {stride}); done {len(done_set)}, "
          f"{rows:,}/{target:,} rows", flush=True)

    while rows < target:
        idx, path = q.get()
        if idx is None: break
        try:
            t = pqt.read_table(path, columns=["image", "recaption"])
            imgs = t.column("image").to_pylist(); caps = t.column("recaption").to_pylist()
        except Exception as ex:
            print(f"    !! read {path}: {ex}", flush=True); os.remove(path); continue

        # the 384 pass is ~64% of encode cost; stage 1 and 3 need far fewer images
        # than stage 2, so cap each extra resolution at its own quota
        want384 = sum(bn.values()) < target_384
        want128 = rows < target_128
        active = [r for r in res_list if r == max(res_list) or want128]
        def prep(a):
            b, c = a
            try:
                raw = b["bytes"] if isinstance(b, dict) else b
                im = Image.open(io.BytesIO(raw)).convert("RGB")
                W, H = im.size; s = min(W, H)
                sq_ = im.crop(((W-s)//2, (H-s)//2, (W-s)//2+s, (H-s)//2+s))
                o = {r: np.asarray(sq_.resize((r, r), Image.LANCZOS)).transpose(2, 0, 1) for r in active}
                if want384:
                    k, bw, bh = _bucket_for(W, H)
                    sc = max(bw / W, bh / H)
                    rz = im.resize((max(bw, int(W*sc+.5)), max(bh, int(H*sc+.5))), Image.LANCZOS)
                    w2, h2 = rz.size
                    o["b"] = (k, np.asarray(rz.crop(((w2-bw)//2, (h2-bh)//2, (w2-bw)//2+bw, (h2-bh)//2+bh))
                                            ).transpose(2, 0, 1))
                return o, (c or "").strip()
            except Exception:
                return None
        with ThreadPoolExecutor(max_workers=workers) as ex:
            dec = [x for x in ex.map(prep, zip(imgs, caps)) if x and x[1]]

        for s0 in range(0, len(dec), batch):
            sel = dec[s0:s0+batch]
            if rows + len(sel) > target: sel = sel[:target - rows]
            if not sel: break
            for r in active:
                x = torch.from_numpy(np.stack([v[0][r] for v in sel])).float().cuda()/127.5 - 1.0
                z = AE.encode(m, x)
                mm[r][rows:rows+len(sel)] = z.cpu().numpy().astype(np.float16)
                if r == max(res_list):
                    tot += z.double().sum(dim=(0,2,3)).cpu(); sq += (z.double()**2).sum(dim=(0,2,3)).cpu()
                    cnt += z.shape[0]*z.shape[2]*z.shape[3]
            for v in sel: cap_f.write(json.dumps([v[1]]) + "\n")
            rows += len(sel)
            if want384:
                by = {}
                for v in sel:
                    if "b" in v[0]:
                        by.setdefault(v[0]["b"][0], []).append((v[0]["b"][1], v[1]))
                for k, items in by.items():
                    room = bcap[k] - bn[k]
                    items = items[:room]
                    if not items: continue
                    x = torch.from_numpy(np.stack([a for a, _ in items])).float().cuda()/127.5 - 1.0
                    z = AE.encode(m, x)
                    bmm[k][bn[k]:bn[k]+len(items)] = z.cpu().numpy().astype(np.float16)
                    for _, cc in items: bcap_f[k].write(json.dumps([cc]) + "\n")
                    bn[k] += len(items)
        cap_f.flush(); [f.flush() for f in bcap_f.values()]
        try: os.remove(path)
        except FileNotFoundError: pass
        done_set.add(idx)
        json.dump({"done": sorted(done_set), "rows": rows, "b": bn}, open(prog_p, "w"))
        json.dump({"n": rows, "n_128": min(rows, target_128), "buckets": bn,
                   "bucket_dims": {k: [W, H] for k, W, H in BUCKETS}},
                  open(os.path.join(d, "meta.json"), "w"), indent=1)
        if idx % 5 == 0 or rows >= target:
            print(f"    shard {idx:>4}/{len(shards)}  rows {rows:,}/{target:,} "
                  f"({100*rows/target:.1f}%)  buckets {sum(bn.values()):,}/{target_384:,}", flush=True)

    stop.set()                        # unblock any producer waiting on a full queue
    for r in res_list: mm[r].flush()
    for k in bmm: bmm[k].flush()
    if cnt:
        mean = tot/cnt; std = (sq/cnt - mean**2).clamp_min(1e-8).sqrt()
        st = {"mean": mean.tolist(), "std": std.tolist()}
        for r in res_list: json.dump(st, open(os.path.join(d, f"lat_{ae_name}_{r}.json"), "w"), indent=1)
        for k, _, _ in BUCKETS: json.dump(st, open(os.path.join(d, f"bucket_{ae_name}_{k}.json"), "w"), indent=1)
    json.dump({"caps": [json.loads(l) for l in open(os.path.join(d, "captions.jsonl"))]},
              open(os.path.join(d, "captions.json"), "w"))
    json.dump({"n": rows, "n_128": min(rows, target_128), "buckets": bn,
               "bucket_dims": {k: [W, H] for k, W, H in BUCKETS}},
              open(os.path.join(d, "meta.json"), "w"), indent=1)
    print(f"  done: {rows:,} rows, buckets {bn}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "latents", "text", "check", "ingest"])
    ap.add_argument("--dataset", default="cub")
    ap.add_argument("--root", default="out/cache")
    ap.add_argument("--ae", default="flux2")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=32)
    ap.add_argument("--token", default=None)
    ap.add_argument("--shards", type=int, default=204)
    ap.add_argument("--target", type=int, default=1030000)
    a = ap.parse_args()
    if a.cmd == "prepare":
        if a.dataset == "coco": prepare_coco(a.root)
        else: prepare(a.dataset, a.root, token=a.token)
    elif a.cmd == "latents": latents(a.dataset, a.root, a.ae, a.res)
    elif a.cmd == "text": text(a.dataset, a.root, a.max_len)
    elif a.cmd == "check": check(a.dataset, a.root, a.ae, a.res)
    elif a.cmd == "ingest": ingest_parquet(a.root, a.dataset, a.ae, target=a.target)

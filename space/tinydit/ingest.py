"""Bucketed, streaming ingest for run 1 (see notes/CONTRACTS.md for the binding layout).

Three sources, one process each, each writing its OWN sub-cache so they can run concurrently
without locks; `merge` then concatenates the sub-caches into the final layout:

    out/cache/<name>/src_<source>/<bucket>/{lat.npy, rows.jsonl, val_images.npy}
    out/cache/<name>/src_<source>/progress.json
        --merge-->
    out/cache/<name>/<bucket>/{lat.npy, rows.jsonl, val_images.npy}
    out/cache/<name>/{meta.json, stats.json}

Every source streams: download -> decode -> bucket -> FLUX.2 AE encode -> append fp16 latent
row + one JSON line -> delete the raw shard. Images are never kept. A run is resumable: the
number of valid rows per bucket is the line count of rows.jsonl, and progress.json remembers
where each source stopped. Bucket assignment is seeded by the sample id, so a resumed run
makes the same choices.

    python -m tinydit.ingest coco   --cache out/cache/run1
    python -m tinydit.ingest pexels --cache out/cache/run1
    python -m tinydit.ingest pexels2 --cache out/cache/run1
    python -m tinydit.ingest flux   --cache out/cache/run1 --target 1200000
    python -m tinydit.ingest merge  --cache out/cache/run1 [--remove-src]
    python -m tinydit.ingest merge  --cache out/cache/run1 --append src_pexels2   # add a source later; stats.json untouched
    python -m tinydit.ingest check  --cache out/cache/run1

coco   train2017 (zip if complete, else image URLs), human captions = short, GPT-4V = long
pexels bghira/photo-concept-bucket via the Pexels CDN at 640 px wide, CogVLM = short,
       zlab-princeton/i1-captions (Qwen3-VL) joined by Pexels id = long
pexels2 animetimm/pexels-tagger-v0-w640-ws-full (gated, 127 WebDataset tars, 640 px wide), the
       ids NOT in photo-concept-bucket; long = i1 Qwen3-VL captions, short = their first sentence;
       rows without an i1 caption are skipped; src is "pexels" so both sets share one weight
flux   LucasFang/FLUX-Reason-6M Aesthetics parts, clarity>=9 & structure>=9,
       caption_detail = long (caption_entity if cropped >20%), caption_entity = short
"""
from __future__ import annotations
import hashlib, io, json, os, queue, random, shutil, subprocess, sys, threading, time, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np

BUCKETS = {"1_1": (256, 256), "4_3": (288, 224), "3_4": (224, 288), "3_2": (320, 208), "2_3": (208, 320)}
SOURCES = ("coco", "pexels", "pexels2", "flux")
AE_NAME, LATENT_CH, F = "flux2", 32, 8
VAL_PER_SOURCE = 1000
HF = "https://huggingface.co"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) tinydit-ingest/0.1"}


def _log(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def _token():
    for p in ("/root/volume/.tinydit_token", "/home/ivan/volume/.tinydit_token"):
        if os.path.exists(p):
            for line in open(p):
                if line.startswith("HF_TOKEN="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("HF_TOKEN")


def _curl(url, out, token=None, quiet=True):
    """Resumable download with curl; returns path. Skips if the file is already complete
    (curl -C - exits 0 immediately when the server says the range is satisfied)."""
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cmd = ["curl", "-sL", "--fail", "--retry", "5", "-C", "-", "-o", out]
    if token: cmd += ["-H", f"Authorization: Bearer {token}"]
    r = subprocess.run(cmd + [url])
    if r.returncode not in (0, 33):        # 33 = range not satisfiable = already complete
        raise RuntimeError(f"curl failed ({r.returncode}) for {url}")
    return out


# ------------------------------------------------------------------ bucketing -----------
def _crop_frac(ar, bucket):
    bw, bh = BUCKETS[bucket]; bar = bw / bh
    return 1 - min(ar, bar) / max(ar, bar)


def assign_bucket(w, h, real: bool, seed: str):
    """-> (bucket or None, crop fraction). Rules from CONTRACTS.md."""
    ar = w / h
    if ar > 2.0 or ar < 0.5:
        return None, 1.0
    rng = random.Random(hashlib.md5(seed.encode()).hexdigest())
    if real:
        b = min(BUCKETS, key=lambda k: _crop_frac(ar, k))
        if _crop_frac(ar, "1_1") <= 0.30 and rng.random() < 0.5:
            b = "1_1"
    else:
        b = "1_1" if rng.random() < 0.45 else rng.choice(["4_3", "3_4", "3_2", "2_3"])
    return b, _crop_frac(ar, b)


def fit_to_bucket(im, bucket):
    """PIL image -> uint8 (3, H, W) array covering the bucket, centre-cropped, LANCZOS."""
    from PIL import Image
    bw, bh = BUCKETS[bucket]
    W, H = im.size
    if getattr(im, "format", None) == "JPEG":
        im.draft("RGB", (bw * 2, bh * 2))          # cheap JPEG DCT downscale first
        W2, H2 = im.size
    im = im.convert("RGB")
    W, H = im.size
    s = max(bw / W, bh / H)
    nw, nh = max(bw, round(W * s)), max(bh, round(H * s))
    im = im.resize((nw, nh), Image.LANCZOS, reducing_gap=3.0)
    l, t = (nw - bw) // 2, (nh - bh) // 2
    im = im.crop((l, t, l + bw, t + bh))
    return np.asarray(im).transpose(2, 0, 1)


def first_sentence(s: str) -> str:
    s = (s or "").strip()
    for sep in (". ", "! ", "? ", ".\n"):
        i = s.find(sep)
        if 0 < i < len(s) - 1:
            return s[:i + 1].strip()
    return s


# ------------------------------------------------------------------ writer --------------
class SubCache:
    """Per-source sub-cache: one memmap + jsonl + val_images per bucket, append-only."""
    def __init__(self, root, source, capacity):
        self.root, self.source = root, source
        self.dir = os.path.join(root, f"src_{source}"); os.makedirs(self.dir, exist_ok=True)
        self.lat, self.val_img, self.rows_f, self.n, self.n_val_b = {}, {}, {}, {}, {}
        for b, (bw, bh) in BUCKETS.items():
            d = os.path.join(self.dir, b); os.makedirs(d, exist_ok=True)
            lp, vp, rp = os.path.join(d, "lat.npy"), os.path.join(d, "val_images.npy"), os.path.join(d, "rows.jsonl")
            shape = (capacity, LATENT_CH, bh // F, bw // F)
            if os.path.exists(lp):
                self.lat[b] = np.load(lp, mmap_mode="r+")
                if self.lat[b].shape[0] < capacity:
                    raise RuntimeError(f"{lp} capacity {self.lat[b].shape[0]} < requested {capacity}")
            else:
                self.lat[b] = np.lib.format.open_memmap(lp, mode="w+", dtype=np.float16, shape=shape)
            if os.path.exists(vp):
                self.val_img[b] = np.load(vp, mmap_mode="r+")
            else:
                self.val_img[b] = np.lib.format.open_memmap(vp, mode="w+", dtype=np.uint8,
                                                            shape=(VAL_PER_SOURCE, 3, bh, bw))
            n = 0; nv = 0
            if os.path.exists(rp):
                for line in open(rp):
                    if line.strip():
                        n += 1
                        if '"val": true' in line: nv += 1
            self.n[b], self.n_val_b[b] = n, nv
            self.rows_f[b] = open(rp, "a")
        self.total = sum(self.n.values()); self.n_val = sum(self.n_val_b.values())
        self.prog_p = os.path.join(self.dir, "progress.json")
        self.prog = json.load(open(self.prog_p)) if os.path.exists(self.prog_p) else {}

    def append(self, bucket, z_fp16, rows, imgs):
        """z_fp16: (k, C, h, w) np.float16; rows: list of dicts (with 'val'); imgs: list of uint8 (3,H,W)"""
        k = len(rows); i0 = self.n[bucket]
        if i0 + k > self.lat[bucket].shape[0]:
            raise RuntimeError(f"bucket {bucket} full ({self.lat[bucket].shape[0]} rows); raise --capacity")
        self.lat[bucket][i0:i0 + k] = z_fp16
        for r, im in zip(rows, imgs):
            if r["val"]:
                j = self.n_val_b[bucket]
                if j < self.val_img[bucket].shape[0]:
                    self.val_img[bucket][j] = im
                self.n_val_b[bucket] += 1
            self.rows_f[bucket].write(json.dumps(r, ensure_ascii=False) + "\n")
        self.rows_f[bucket].flush()
        self.n[bucket] += k; self.total += k
        self.n_val = sum(self.n_val_b.values())

    def save_progress(self, **kw):
        self.prog.update(kw); self.prog["rows"] = self.n; self.prog["total"] = self.total
        json.dump(self.prog, open(self.prog_p, "w"), indent=1)

    def flush(self):
        for b in BUCKETS:
            self.lat[b].flush(); self.val_img[b].flush(); self.rows_f[b].flush()


class Encoder:
    """Batches decoded images per bucket and pushes them through the frozen FLUX.2 AE."""
    def __init__(self, cache: SubCache, batch=64, models="out/models"):
        import torch
        from . import ae as AE
        self.torch, self.AE, self.cache, self.batch = torch, AE, cache, batch
        AE.fetch(AE_NAME, models)
        self.ae = AE.load(AE_NAME, os.path.join(models, AE_NAME))
        self.pending = {b: [] for b in BUCKETS}
        self.written = 0

    def add(self, bucket, img, row):
        self.pending[bucket].append((img, row))
        if len(self.pending[bucket]) >= self.batch:
            self._encode(bucket)

    def _encode(self, bucket):
        items = self.pending[bucket]; self.pending[bucket] = []
        if not items: return
        imgs = [it[0] for it in items]; rows = [it[1] for it in items]
        x = self.torch.from_numpy(np.stack(imgs)).cuda(non_blocking=True).float() / 127.5 - 1.0
        with self.torch.autocast("cuda", dtype=self.torch.bfloat16):
            z = self.AE.encode(self.ae, x)
        z = z.float().cpu().numpy().astype(np.float16)
        self.cache.append(bucket, z, rows, imgs)
        self.written += len(items)

    def flush(self):
        for b in BUCKETS: self._encode(b)
        self.cache.flush()


class Progress:
    def __init__(self, name, total=None, every=500):
        self.name, self.total, self.every = name, total, every
        self.t0 = time.time(); self.ok = 0; self.fail = 0; self.drop = 0; self.last = 0

    def tick(self, ok=0, fail=0, drop=0, force=False):
        self.ok += ok; self.fail += fail; self.drop += drop
        if force or self.ok - self.last >= self.every:
            self.last = self.ok; dt = time.time() - self.t0
            tot = f"/{self.total:,}" if self.total else ""
            _log(f"[{self.name}] rows {self.ok:,}{tot}  {self.ok/max(dt,1e-6):.1f} rows/s  "
                 f"fails {self.fail}  dropped(aspect) {self.drop}  elapsed {dt/60:.1f} min")


def _row(src, id_, long_caps, short_caps, crop, w, h, val):
    long_caps = [c.strip() for c in long_caps if c and c.strip()]
    short_caps = [c.strip() for c in short_caps if c and c.strip()]
    if not short_caps: short_caps = [first_sentence(long_caps[0])] if long_caps else []
    if not long_caps: long_caps = list(short_caps)
    if not long_caps: return None
    return {"src": src, "id": str(id_), "long": long_caps, "short": short_caps,
            "crop": round(float(crop), 4), "w": int(w), "h": int(h), "val": bool(val)}


def _fetch(url, timeout=20, retries=3):
    import requests
    for a in range(retries):
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            if r.status_code == 200 and r.content: return r.content
            if r.status_code in (403, 404, 410): return None
        except Exception:
            pass
        time.sleep(1.5 * (a + 1))
    return None


# ------------------------------------------------------------------ coco ----------------
def ingest_coco(cache_root, limit=0, capacity=130_000, workers=24, batch=64):
    from PIL import Image
    raw = os.path.join("out", "cache", "coco", "raw"); os.makedirs(raw, exist_ok=True)
    ann_zip = _curl("http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
                    os.path.join(raw, "annotations.zip"))
    ann = json.loads(zipfile.ZipFile(ann_zip).read("annotations/captions_train2017.json"))
    caps = {}
    for a in ann["annotations"]: caps.setdefault(a["image_id"], []).append(a["caption"].strip())
    images = sorted([im for im in ann["images"] if im["id"] in caps], key=lambda x: x["id"])
    # GPT-4V long captions (LVIS split of COCO), joined by image id parsed from the url
    gpt_p = _curl(f"{HF}/datasets/laion/220k-GPT4Vision-captions-from-LIVIS/resolve/main/lvis_caption_url.parquet",
                  os.path.join(raw, "lvis_caption_url.parquet"), _token())
    import pyarrow.parquet as pq
    t = pq.read_table(gpt_p, columns=["url", "caption"]).to_pydict()
    gpt = {}
    for u, c in zip(t["url"], t["caption"]):
        if "/train2017/" in u:
            try: gpt.setdefault(int(u.rsplit("/", 1)[1].split(".")[0]), []).append(c.strip())
            except ValueError: pass
    _log(f"[coco] {len(images):,} images, {sum(len(v) for v in caps.values()):,} human captions, "
         f"GPT-4V captions for {sum(1 for im in images if im['id'] in gpt):,} of them")

    zpath = os.path.join(raw, "train2017.zip"); zf = None
    try:
        zf = zipfile.ZipFile(zpath); zf.getinfo("train2017/000000000009.jpg")
        _log("[coco] reading images from train2017.zip")
    except Exception:
        zf = None; _log("[coco] train2017.zip not complete; fetching images by URL (slower)")
    zlock = threading.Lock()

    cache = SubCache(cache_root, "coco", capacity); enc = Encoder(cache, batch)
    start = cache.prog.get("next", 0)
    if limit: images = images[:limit]
    prog = Progress("coco", len(images)); prog.ok = start; prog.last = start
    _log(f"[coco] resuming at image index {start} (rows so far {cache.total:,})")

    def work(im):
        try:
            if zf is not None:
                with zlock: b = zf.read("train2017/" + im["file_name"])
            else:
                b = _fetch("http://images.cocodataset.org/train2017/" + im["file_name"])
                if b is None: return im, None, "fetch"
            bucket, crop = assign_bucket(im["width"], im["height"], True, f"coco{im['id']}")
            if bucket is None: return im, None, "aspect"
            pil = Image.open(io.BytesIO(b))
            return im, (bucket, crop, fit_to_bucket(pil, bucket)), None
        except Exception as ex:
            return im, None, f"{type(ex).__name__}"

    CH = 512
    with ThreadPoolExecutor(workers) as ex:
        for c0 in range(start, len(images), CH):
            chunk = images[c0:c0 + CH]
            for im, res, err in ex.map(work, chunk):
                if res is None:
                    prog.tick(fail=err != "aspect", drop=err == "aspect"); continue
                bucket, crop, arr = res
                val = cache.total + sum(len(v) for v in enc.pending.values()) < VAL_PER_SOURCE
                row = _row("coco", im["id"], gpt.get(im["id"], []), caps[im["id"]][:5], crop,
                           im["width"], im["height"], val)
                if row is None: prog.tick(fail=1); continue
                enc.add(bucket, arr, row); prog.tick(ok=1)
            enc.flush(); cache.save_progress(next=c0 + CH)
    enc.flush(); cache.save_progress(next=len(images), done=True); prog.tick(force=True)
    _log(f"[coco] done: {cache.total:,} rows " + str(cache.n))


# ------------------------------------------------------------------ pexels --------------
def _load_i1_pexels(raw, need_ids: set, keep=3):
    """Qwen3-VL captions for the Pexels ids we will use. Streams the 3 parquet files
    (4.3 GB) row-group by row-group, keeping only matching keys."""
    import pyarrow.parquet as pq
    tok = _token(); out = {}
    for i in (1, 2, 3):
        p = _curl(f"{HF}/datasets/zlab-princeton/i1-captions/resolve/main/pexels/train-0000{i}.parquet",
                  os.path.join(raw, f"i1_pexels_train-0000{i}.parquet"), tok)
        pf = pq.ParquetFile(p)
        cols = [c for c in pf.schema_arrow.names if c.startswith("caption")][:keep]
        for rg in range(pf.num_row_groups):
            tb = pf.read_row_group(rg, columns=["key", *cols]).to_pydict()
            for j, k in enumerate(tb["key"]):
                if k in need_ids:
                    out[k] = [tb[c][j] for c in cols if tb[c][j]]
        _log(f"[pexels] i1 captions file {i}: matched {len(out):,} of {len(need_ids):,} ids so far")
    return out


def ingest_pexels(cache_root, limit=0, capacity=600_000, workers=32, batch=64):
    from PIL import Image
    import pyarrow.parquet as pq
    raw = os.path.join("out", "cache", "pexels", "raw"); os.makedirs(raw, exist_ok=True)
    p = _curl(f"{HF}/datasets/bghira/photo-concept-bucket/resolve/main/photo-concept-bucket.parquet",
              os.path.join(raw, "photo-concept-bucket.parquet"), _token())
    t = pq.read_table(p, columns=["id", "url", "width", "height", "cogvlm_caption"]).to_pydict()
    rows = list(zip(t["id"], t["url"], t["width"], t["height"], t["cogvlm_caption"]))
    if limit: rows = rows[:limit]
    long_caps = _load_i1_pexels(raw, {str(r[0]) for r in rows})
    _log(f"[pexels] {len(rows):,} rows; Qwen3-VL long captions for {len(long_caps):,} ({100*len(long_caps)/max(1,len(rows)):.1f}%)")

    cache = SubCache(cache_root, "pexels", capacity); enc = Encoder(cache, batch)
    start = cache.prog.get("next", 0)
    prog = Progress("pexels", len(rows)); prog.ok = start; prog.last = start
    _log(f"[pexels] resuming at row {start} (rows so far {cache.total:,})")

    def work(r):
        pid, url, w, h, cap = r
        try:
            bucket, crop = assign_bucket(w, h, True, f"pexels{pid}")
            if bucket is None: return r, None, "aspect"
            b = _fetch(url.split("?")[0] + "?auto=compress&cs=tinysrgb&w=640")
            if b is None: return r, None, "fetch"
            pil = Image.open(io.BytesIO(b))
            return r, (bucket, crop, fit_to_bucket(pil, bucket)), None
        except Exception as ex:
            return r, None, type(ex).__name__

    CH = 1024
    with ThreadPoolExecutor(workers) as ex:
        for c0 in range(start, len(rows), CH):
            for r, res, err in ex.map(work, rows[c0:c0 + CH]):
                if res is None:
                    prog.tick(fail=err != "aspect", drop=err == "aspect"); continue
                bucket, crop, arr = res
                pid, url, w, h, cap = r
                val = cache.total + sum(len(v) for v in enc.pending.values()) < VAL_PER_SOURCE
                row = _row("pexels", pid, long_caps.get(str(pid), []), [cap], crop, w, h, val)
                if row is None: prog.tick(fail=1); continue
                enc.add(bucket, arr, row); prog.tick(ok=1)
            enc.flush(); cache.save_progress(next=c0 + CH)
    enc.flush(); cache.save_progress(next=len(rows), done=True); prog.tick(force=True)
    _log(f"[pexels] done: {cache.total:,} rows " + str(cache.n))


# ------------------------------------------------------------------ pexels2 -------------
PEXELS2_REPO = "animetimm/pexels-tagger-v0-w640-ws-full"


def _pexels2_shards():
    import requests
    h = dict(UA); tok = _token()
    if tok: h["Authorization"] = f"Bearer {tok}"
    r = requests.get(f"{HF}/api/datasets/{PEXELS2_REPO}/tree/main", headers=h, timeout=60,
                     params={"recursive": "true"}); r.raise_for_status()
    tars = sorted(x["path"] for x in r.json() if x["path"].endswith(".tar"))
    order = {"train": 0, "val": 1, "test": 2}
    return sorted(tars, key=lambda p: (order.get(p.split("/")[0], 9), p))


def _iter_tar_pairs(path):
    """Stream a WebDataset tar: yield (key, webp_bytes, json_dict) for each complete pair."""
    import tarfile
    buf = {}
    with tarfile.open(path, mode="r|") as tf:
        for m in tf:
            if not m.isfile(): continue
            base, _, ext = os.path.basename(m.name).rpartition(".")
            data = tf.extractfile(m).read()
            d = buf.setdefault(base, {}); d[ext.lower()] = data
            if "webp" in d and "json" in d:
                buf.pop(base)
                try: meta = json.loads(d["json"])
                except Exception: continue
                yield base, d["webp"], meta


def ingest_pexels2(cache_root, limit=0, capacity=2_300_000, workers=24, batch=64, prefetch=2):
    """The gated 2.8M-image Pexels pool (640 px wide, aspect kept), minus the ids already covered
    by the CDN `pexels` source. Captions come only from the i1 Qwen3-VL join; rows without one are
    skipped. Rows carry src="pexels" so training weights treat both Pexels sets as one source."""
    from PIL import Image
    import pyarrow.parquet as pq
    raw = os.path.join("out", "cache", "pexels2", "raw"); os.makedirs(raw, exist_ok=True)
    tmp = os.path.join("out", "cache", "pexels2", "_tar"); os.makedirs(tmp, exist_ok=True)
    tok = _token()
    # ids already ingested through the CDN list -> skip here
    pcb = _curl(f"{HF}/datasets/bghira/photo-concept-bucket/resolve/main/photo-concept-bucket.parquet",
                os.path.join("out", "cache", "pexels", "raw", "photo-concept-bucket.parquet"), tok)
    skip = {str(i) for i in pq.read_table(pcb, columns=["id"]).column("id").to_pylist()}
    meta = json.load(open(_curl(f"{HF}/datasets/{PEXELS2_REPO}/resolve/main/meta.json",
                                os.path.join(raw, "meta.json"), tok)))
    all_ids = {str(i) for i in meta["exist_ids"]}
    need = all_ids - skip
    _log(f"[pexels2] {len(all_ids):,} ids in the repo, {len(all_ids & skip):,} already covered by the CDN set, "
         f"{len(need):,} to ingest")
    shards = _pexels2_shards()
    _log(f"[pexels2] {len(shards)} tar shards")

    cache = SubCache(cache_root, "pexels2", capacity); enc = Encoder(cache, batch)
    done = set(cache.prog.get("done_shards", []))
    want_rows = limit or len(need)
    prog = Progress("pexels2", want_rows if limit else None); prog.ok = cache.total; prog.last = cache.total
    stats = cache.prog.get("stats", {"skip_cdn": 0, "no_caption": 0})

    q: queue.Queue = queue.Queue(maxsize=prefetch)
    stop = threading.Event()
    def producer():
        for i, rel in enumerate(shards):
            if stop.is_set(): break
            if i in done: continue
            local = os.path.join(tmp, rel.replace("/", "_"))
            try:
                _curl(f"{HF}/datasets/{PEXELS2_REPO}/resolve/main/{rel}", local, tok)
            except Exception as ex:
                _log(f"[pexels2] !! download {rel}: {ex}"); continue
            while not stop.is_set():
                try: q.put((i, local), timeout=5); break
                except queue.Full: continue
        q.put((None, None))
    threading.Thread(target=producer, daemon=True).start()

    long_caps = None                      # loaded after the first shard when --limit (smoke) is set
    if not limit:
        long_caps = _load_i1_pexels(os.path.join("out", "cache", "pexels", "raw"), need)
        _log(f"[pexels2] Qwen3-VL captions for {len(long_caps):,} of {len(need):,} ids ({100*len(long_caps)/max(1,len(need)):.1f}%)")

    def work(item):
        pid, b, w, h = item
        try:
            bucket, crop = assign_bucket(w, h, True, f"pexels{pid}")
            if bucket is None: return item, None, "aspect"
            pil = Image.open(io.BytesIO(b))
            return (pid, w, h), (bucket, crop, fit_to_bucket(pil, bucket)), None
        except Exception as ex:
            return item, None, type(ex).__name__

    with ThreadPoolExecutor(workers) as ex:
        while cache.total < want_rows:
            i, local = q.get()
            if i is None: break
            t0 = time.time(); n_shard = 0
            try:
                pairs = []
                for key, wb, meta_j in _iter_tar_pairs(local):
                    pid = str(meta_j.get("id", key))
                    if pid in skip: stats["skip_cdn"] += 1; continue
                    pairs.append((pid, wb, int(meta_j.get("width", 0)) or 640, int(meta_j.get("height", 0)) or 640))
                if long_caps is None:     # smoke mode: captions only for this shard's ids
                    long_caps = _load_i1_pexels(os.path.join("out", "cache", "pexels", "raw"), {p[0] for p in pairs})
                    _log(f"[pexels2] (smoke) captions for {len(long_caps):,} of {len(pairs):,} ids in the first shard")
                items = []
                for p in pairs:
                    if p[0] in long_caps: items.append(p)
                    else: stats["no_caption"] += 1
                for c0 in range(0, len(items), 1024):
                    for meta_t, res, err in ex.map(work, items[c0:c0 + 1024]):
                        if res is None:
                            prog.tick(fail=err != "aspect", drop=err == "aspect"); continue
                        pid, w, h = meta_t; bucket, crop, arr = res
                        caps = long_caps[pid]
                        val = cache.total + sum(len(v) for v in enc.pending.values()) < VAL_PER_SOURCE
                        row = _row("pexels", pid, caps, [first_sentence(caps[0])], crop, w, h, val)
                        if row is None: prog.tick(fail=1); continue
                        enc.add(bucket, arr, row); prog.tick(ok=1); n_shard += 1
                        if cache.total + sum(len(v) for v in enc.pending.values()) >= want_rows: break
                    if cache.total + sum(len(v) for v in enc.pending.values()) >= want_rows: break
                enc.flush()
            except Exception as ex:
                _log(f"[pexels2] !! shard {local}: {type(ex).__name__}: {str(ex)[:120]}")
            try: os.remove(local)
            except FileNotFoundError: pass
            done.add(i)
            cache.save_progress(done_shards=sorted(done), n_shards=len(shards), stats=stats)
            _log(f"[pexels2] shard {i+1}/{len(shards)} -> {n_shard:,} rows in {time.time()-t0:.0f}s "
                 f"(skipped cdn {stats['skip_cdn']:,}, no caption {stats['no_caption']:,})")
    stop.set(); enc.flush()
    cache.save_progress(done_shards=sorted(done), stats=stats, done=len(done) >= len(shards) or bool(limit))
    prog.tick(force=True)
    _log(f"[pexels2] done: {cache.total:,} rows " + str(cache.n))


# ------------------------------------------------------------------ flux ----------------
def _flux_shards():
    import requests
    out = []
    for part in ("Aesthetics-Part01", "Aesthetics-Part02"):
        url = f"{HF}/api/datasets/LucasFang/FLUX-Reason-6M/tree/main/{part}"
        h = dict(UA); tok = _token()
        if tok: h["Authorization"] = f"Bearer {tok}"
        r = requests.get(url, headers=h, timeout=60, params={"recursive": "true"}); r.raise_for_status()
        out += sorted(x["path"] for x in r.json() if x["path"].endswith(".parquet"))
    return out


def ingest_flux(cache_root, limit=0, target=1_200_000, capacity=1_300_000, workers=24, batch=64,
                min_clarity=9, min_structure=9, prefetch=2):
    from PIL import Image
    import pyarrow.parquet as pq
    tmp = os.path.join("out", "cache", "flux", "_pq"); os.makedirs(tmp, exist_ok=True)
    shards = _flux_shards()
    want_rows = limit or target
    need = int(want_rows / (5000 * 0.80)) + 2                       # ~80% survive the score filter
    stride = max(1, len(shards) // need)
    sel = shards[::stride][:need]
    _log(f"[flux] {len(shards)} Aesthetics shards; using {len(sel)} (stride {stride}) for {want_rows:,} rows")

    cache = SubCache(cache_root, "flux", capacity); enc = Encoder(cache, batch)
    done = set(cache.prog.get("done_shards", []))
    prog = Progress("flux", want_rows); prog.ok = cache.total; prog.last = cache.total
    tok = _token()

    q: queue.Queue = queue.Queue(maxsize=prefetch)
    stop = threading.Event()
    def producer():
        for i, rel in enumerate(sel):
            if stop.is_set(): break
            if i in done: continue
            local = os.path.join(tmp, os.path.basename(rel))
            try:
                _curl(f"{HF}/datasets/LucasFang/FLUX-Reason-6M/resolve/main/{rel}", local, tok)
            except Exception as ex:
                _log(f"[flux] !! download {rel}: {ex}"); continue
            while not stop.is_set():
                try: q.put((i, local)); break
                except queue.Full: time.sleep(1)
        q.put((None, None))
    threading.Thread(target=producer, daemon=True).start()

    def work(item):
        rid, b, det, ent, comp = item
        try:
            pil = Image.open(io.BytesIO(b)); w, h = pil.size
            bucket, crop = assign_bucket(w, h, False, f"flux{rid}")
            if bucket is None: return item, None, "aspect"
            return (rid, det, ent, comp, w, h), (bucket, crop, fit_to_bucket(pil, bucket)), None
        except Exception as ex:
            return item, None, type(ex).__name__

    cols = ["id", "image", "caption_detail", "caption_entity", "caption_composition",
            "score_image_clarity", "score_image_structure"]
    with ThreadPoolExecutor(workers) as ex:
        while cache.total < want_rows:
            i, local = q.get()
            if i is None: break
            try:
                pf = pq.ParquetFile(local)
                for rg in range(pf.num_row_groups):
                    if cache.total >= want_rows: break
                    tb = pf.read_row_group(rg, columns=cols).to_pydict()
                    items = []
                    for j in range(len(tb["id"])):
                        if (tb["score_image_clarity"][j] or 0) < min_clarity or (tb["score_image_structure"][j] or 0) < min_structure:
                            continue
                        im = tb["image"][j]; b = im["bytes"] if isinstance(im, dict) else im
                        items.append((tb["id"][j], b, tb["caption_detail"][j], tb["caption_entity"][j], tb["caption_composition"][j]))
                    for meta, res, err in ex.map(work, items):
                        if res is None:
                            prog.tick(fail=err != "aspect", drop=err == "aspect"); continue
                        rid, det, ent, comp, w, h = meta
                        bucket, crop, arr = res
                        short = [c for c in (ent, comp, first_sentence(det)) if c and c.strip()][:1]
                        long = [det] if crop <= 0.20 or not short else [short[0]]
                        val = cache.total + sum(len(v) for v in enc.pending.values()) < VAL_PER_SOURCE
                        row = _row("flux", rid, long, short, crop, w, h, val)
                        if row is None: prog.tick(fail=1); continue
                        enc.add(bucket, arr, row); prog.tick(ok=1)
                        if cache.total + sum(len(v) for v in enc.pending.values()) >= want_rows: break
                enc.flush()
            except Exception as ex:
                _log(f"[flux] !! shard {local}: {type(ex).__name__}: {str(ex)[:100]}")
            try: os.remove(local)
            except FileNotFoundError: pass
            done.add(i); cache.save_progress(done_shards=sorted(done), n_shards=len(sel))
    stop.set(); enc.flush(); cache.save_progress(done_shards=sorted(done), done=cache.total >= want_rows)
    prog.tick(force=True)
    _log(f"[flux] done: {cache.total:,} rows " + str(cache.n))


# ------------------------------------------------------------------ merge / check -------
def merge(cache_root, remove_src=False, stats_rows=40_000, exclude=()):
    """exclude: sub-cache names still being written (e.g. src_pexels2) that must be left alone and
    added later with --append."""
    srcs = [d for d in sorted(os.listdir(cache_root)) if d.startswith("src_") and os.path.isdir(os.path.join(cache_root, d))
            and d not in set(exclude)]
    if exclude: print(f"  merge: excluding {sorted(set(exclude))}", flush=True)
    if not srcs: raise SystemExit("no src_* sub-caches found")
    _log(f"[merge] sources: {srcs}")
    tot_sum = np.zeros(LATENT_CH, np.float64); tot_sq = np.zeros(LATENT_CH, np.float64); cnt = 0
    rng = np.random.default_rng(0)
    for b, (bw, bh) in BUCKETS.items():
        parts = []
        for s in srcs:
            d = os.path.join(cache_root, s, b); rp = os.path.join(d, "rows.jsonl")
            n = sum(1 for l in open(rp) if l.strip()) if os.path.exists(rp) else 0
            if n: parts.append((s, d, n))
        n_tot = sum(p[2] for p in parts)
        od = os.path.join(cache_root, b); os.makedirs(od, exist_ok=True)
        lat_out = np.lib.format.open_memmap(os.path.join(od, "lat.npy"), mode="w+", dtype=np.float16,
                                            shape=(max(n_tot, 1), LATENT_CH, bh // F, bw // F))
        rows_out = open(os.path.join(od, "rows.jsonl"), "w"); vals = []; i = 0
        for s, d, n in parts:
            lat = np.load(os.path.join(d, "lat.npy"), mmap_mode="r")
            for c0 in range(0, n, 4096):
                c1 = min(n, c0 + 4096); lat_out[i + c0:i + c1] = lat[c0:c1]
            nv = 0
            for line in open(os.path.join(d, "rows.jsonl")):
                if line.strip():
                    rows_out.write(line if line.endswith("\n") else line + "\n")
                    if '"val": true' in line: nv += 1
            vi = np.load(os.path.join(d, "val_images.npy"), mmap_mode="r")
            vals.append(np.asarray(vi[:min(nv, vi.shape[0])]))
            k = min(n, max(1, stats_rows * n // max(1, sum(p[2] for p in parts) * len(BUCKETS))))
            idx = np.sort(rng.choice(n, size=min(n, k), replace=False))
            zs = lat[idx].astype(np.float64)
            tot_sum += zs.sum(axis=(0, 2, 3)); tot_sq += (zs ** 2).sum(axis=(0, 2, 3)); cnt += zs.shape[0] * zs.shape[2] * zs.shape[3]
            i += n
        rows_out.close(); lat_out.flush()
        v = np.concatenate(vals) if vals else np.zeros((0, 3, bh, bw), np.uint8)
        np.save(os.path.join(od, "val_images.npy"), v)
        _log(f"[merge] {b}: {n_tot:,} rows, {v.shape[0]} val images " + ", ".join(f"{s[4:]}={n}" for s, _, n in parts))
    mean = tot_sum / max(cnt, 1); std = np.sqrt(np.maximum(tot_sq / max(cnt, 1) - mean ** 2, 1e-8))
    json.dump({"mean": mean.tolist(), "std": std.tolist()}, open(os.path.join(cache_root, "stats.json"), "w"), indent=1)
    json.dump({"ae": AE_NAME, "latent_ch": LATENT_CH, "buckets": {k: list(v) for k, v in BUCKETS.items()},
               "sources": sorted(_row_sources(cache_root)), "merged": [s[4:] for s in srcs]},
              open(os.path.join(cache_root, "meta.json"), "w"), indent=1)
    _log(f"[merge] stats: std range {std.min():.3f}-{std.max():.3f}; wrote meta.json, stats.json")
    if remove_src:
        for s in srcs: shutil.rmtree(os.path.join(cache_root, s))
        _log("[merge] removed sub-caches")


def _row_sources(cache_root):
    """Distinct `src` values present in the final rows.jsonl files (cheap string scan)."""
    out = set()
    for b in BUCKETS:
        rp = os.path.join(cache_root, b, "rows.jsonl")
        if not os.path.exists(rp): continue
        for line in open(rp):
            i = line.find('"src": "')
            if i >= 0:
                j = line.find('"', i + 8); out.add(line[i + 8:j])
    return out


def _count_lines(p):
    return sum(1 for l in open(p) if l.strip()) if os.path.exists(p) else 0


def append_subcache(cache_root, src_dir, headroom=0, force=False):
    """Append one sub-cache (e.g. `src_pexels2`) to an already merged cache without touching
    stats.json or the order of existing rows. Per bucket: latents go into free capacity of
    lat.npy (rows beyond the current rows.jsonl line count) or, if there is not enough, the
    memmap is rewritten into a larger file (old rows first, unchanged); rows.jsonl gets the new
    lines; val_images.npy gets the new val images after the existing ones, so it stays aligned
    with the val rows in rows.jsonl order. Run only while no trainer has the cache open."""
    sd = os.path.join(cache_root, src_dir)
    if not os.path.isdir(sd): raise SystemExit(f"{sd} not found")
    meta_p = os.path.join(cache_root, "meta.json")
    if not os.path.exists(meta_p): raise SystemExit("cache has no meta.json: run a plain merge first")
    meta = json.load(open(meta_p))
    name = src_dir[4:] if src_dir.startswith("src_") else src_dir
    if name in meta.get("merged", []) and not force:
        raise SystemExit(f"{name} is already merged into {cache_root} (use --force to append again)")
    _log(f"[append] {src_dir} -> {cache_root} (stats.json left untouched)")
    total_new = 0
    for b, (bw, bh) in BUCKETS.items():
        d = os.path.join(sd, b); rp_new = os.path.join(d, "rows.jsonl")
        n_new = _count_lines(rp_new)
        od = os.path.join(cache_root, b); os.makedirs(od, exist_ok=True)
        rp_old = os.path.join(od, "rows.jsonl"); lp_old = os.path.join(od, "lat.npy"); vp_old = os.path.join(od, "val_images.npy")
        n_old = _count_lines(rp_old)
        if n_new == 0:
            _log(f"[append] {b}: nothing to add ({n_old:,} rows stay)"); continue
        lat_new = np.load(os.path.join(d, "lat.npy"), mmap_mode="r")
        shape_tail = (LATENT_CH, bh // F, bw // F)
        if os.path.exists(lp_old):
            lat_old = np.load(lp_old, mmap_mode="r+")
            assert lat_old.shape[1:] == shape_tail and lat_old.shape[0] >= n_old, f"{lp_old} inconsistent with rows.jsonl"
        else:
            lat_old = None
        cap_old = lat_old.shape[0] if lat_old is not None else 0
        if cap_old >= n_old + n_new:                          # free capacity: write in place
            for c0 in range(0, n_new, 4096):
                c1 = min(n_new, c0 + 4096); lat_old[n_old + c0:n_old + c1] = lat_new[c0:c1]
            lat_old.flush(); grown = False
        else:                                                  # grow: rewrite into a larger file
            tmp = lp_old + ".growing"
            big = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(n_old + n_new + headroom, *shape_tail))
            for c0 in range(0, n_old, 4096):
                c1 = min(n_old, c0 + 4096); big[c0:c1] = lat_old[c0:c1]
            for c0 in range(0, n_new, 4096):
                c1 = min(n_new, c0 + 4096); big[n_old + c0:n_old + c1] = lat_new[c0:c1]
            big.flush(); del big, lat_old
            os.replace(tmp, lp_old); grown = True
        # val images: existing ones first (rows order), then the new sub-cache's val rows
        nv_new = 0
        for line in open(rp_new):
            if '"val": true' in line: nv_new += 1
        vi_new = np.load(os.path.join(d, "val_images.npy"), mmap_mode="r")
        nv_new = min(nv_new, vi_new.shape[0])
        vi_old = np.load(vp_old, mmap_mode="r") if os.path.exists(vp_old) else np.zeros((0, 3, bh, bw), np.uint8)
        v = np.concatenate([np.asarray(vi_old), np.asarray(vi_new[:nv_new])]) if (vi_old.shape[0] or nv_new) else np.zeros((0, 3, bh, bw), np.uint8)
        np.save(vp_old + ".tmp.npy", v); os.replace(vp_old + ".tmp.npy", vp_old)
        # rows last, so a crash before this point leaves the line count (= valid rows) unchanged
        with open(rp_old, "a") as fo:
            for line in open(rp_new):
                if line.strip(): fo.write(line if line.endswith("\n") else line + "\n")
        total_new += n_new
        _log(f"[append] {b}: {n_old:,} + {n_new:,} rows ({'grew memmap' if grown else 'in place'}), "
             f"val images {vi_old.shape[0]} + {nv_new}")
    meta["merged"] = meta.get("merged", []) + [name]
    meta["sources"] = sorted(_row_sources(cache_root))
    json.dump(meta, open(meta_p, "w"), indent=1)
    _log(f"[append] done: +{total_new:,} rows; meta.json updated (sources {meta['sources']}, merged {meta['merged']})")


def check(cache_root, k=4, models="out/models"):
    import torch
    from PIL import Image
    from . import ae as AE
    meta = json.load(open(os.path.join(cache_root, "meta.json")))
    m = AE.load(AE_NAME, os.path.join(models, AE_NAME))
    for b, (bw, bh) in meta["buckets"].items():
        d = os.path.join(cache_root, b)
        rows = [json.loads(l) for l in open(os.path.join(d, "rows.jsonl")) if l.strip()]
        lat = np.load(os.path.join(d, "lat.npy"), mmap_mode="r")
        n = len(rows)
        assert lat.shape[0] >= n, f"{b}: lat rows {lat.shape[0]} < jsonl rows {n}"
        if n == 0: _log(f"[check] {b}: empty"); continue
        idx = np.linspace(0, n - 1, min(k, n)).astype(int)
        z = torch.from_numpy(np.ascontiguousarray(lat[idx])).float().cuda()
        y = ((AE.decode(m, z) + 1) * 127.5).clamp(0, 255).byte().cpu().numpy()
        strip = np.concatenate([y[j].transpose(1, 2, 0) for j in range(len(idx))], axis=1)
        Image.fromarray(strip).save(os.path.join(d, "check.png"))
        by_src = {}
        for r in rows: by_src.setdefault(r["src"], []).append(r["crop"])
        _log(f"[check] {b}: {n:,} rows | " + ", ".join(f"{s}: {len(c)} (mean crop {100*np.mean(c):.1f}%)" for s, c in by_src.items())
             + f" | val {sum(r['val'] for r in rows)} | check.png written")
        for j in idx[:2]:
            r = rows[j]; _log(f"    [{j}] {r['src']} {r['id']} crop {r['crop']} val {r['val']} | short: {r['short'][0][:90]} | long: {r['long'][0][:90]}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=[*SOURCES, "merge", "check"])
    ap.add_argument("--cache", default="out/cache/run1")
    ap.add_argument("--limit", type=int, default=0, help="rows per source (smoke tests)")
    ap.add_argument("--target", type=int, default=1_200_000, help="flux rows to keep")
    ap.add_argument("--capacity", type=int, default=0, help="memmap capacity per bucket for this source")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--remove-src", action="store_true")
    ap.add_argument("--append", default=None, help="merge: append this sub-cache (e.g. src_pexels2) to an already merged cache")
    ap.add_argument("--headroom", type=int, default=0, help="merge --append: extra free rows when a memmap must be grown")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--exclude", default="", help="merge: comma-separated sub-caches to leave alone (still being written)")
    a = ap.parse_args()
    os.makedirs(a.cache, exist_ok=True)
    if a.cmd == "coco":
        ingest_coco(a.cache, a.limit, a.capacity or (a.limit + 64 if a.limit else 130_000), a.workers or 24, a.batch)
    elif a.cmd == "pexels":
        ingest_pexels(a.cache, a.limit, a.capacity or (a.limit + 64 if a.limit else 600_000), a.workers or 32, a.batch)
    elif a.cmd == "pexels2":
        ingest_pexels2(a.cache, a.limit, a.capacity or (a.limit + 64 if a.limit else 2_300_000), a.workers or 24, a.batch)
    elif a.cmd == "flux":
        ingest_flux(a.cache, a.limit, a.target, a.capacity or ((a.limit or a.target) + 5064), a.workers or 24, a.batch)
    elif a.cmd == "merge":
        if a.append: append_subcache(a.cache, a.append, a.headroom, a.force)
        else: merge(a.cache, a.remove_src, exclude=[x for x in a.exclude.split(",") if x])
    elif a.cmd == "check":
        check(a.cache)

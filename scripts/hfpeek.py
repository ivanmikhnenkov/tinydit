"""Peek into Hugging Face datasets without downloading whole shards.

  hfpeek.py tree   <repo_id> [--max 40]            list files (REST API, no auth needed)
  hfpeek.py schema <repo_id> <path.parquet>        parquet schema + row-group layout + 1 row
  hfpeek.py sample <spec.json> <out.json>          pull N rows per dataset into a preview JSON

Parquet files are read through HTTP Range requests (footer + one row group), so a
500 MB shard costs a few MB. URL-only datasets have their images fetched directly.
"""
from __future__ import annotations
import base64, io, json, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor
import socket as _socket
# IPv6 is configured but unrouted on this host: Python tries every v6 address first and
# hangs for minutes, while curl falls back to v4 instantly. Force v4 for everything here.
_gai = _socket.getaddrinfo
def _gai_v4(*a, **k): return [r for r in _gai(*a, **k) if r[0] == _socket.AF_INET]
_socket.getaddrinfo = _gai_v4
import requests
import pyarrow.parquet as pq
from PIL import Image

HF = "https://huggingface.co"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) tinydit-dataset-preview/0.1"}
TOKEN = None
for p in ("/home/ivan/volume/.tinydit_token",):
    if os.path.exists(p):
        for line in open(p):
            if line.startswith("HF_TOKEN="):
                TOKEN = line.split("=", 1)[1].strip().strip('"').strip("'")


def _hdr(auth=True):
    h = dict(UA)
    if auth and TOKEN: h["Authorization"] = f"Bearer {TOKEN}"
    return h


def tree(repo, path="", rev="main", recursive=True):
    out, url = [], f"{HF}/api/datasets/{repo}/tree/{rev}/{path}".rstrip("/")
    params = {"recursive": "true"} if recursive else {}
    while url:
        r = requests.get(url, params=params, headers=_hdr(), timeout=60)
        r.raise_for_status()
        out += r.json()
        url = r.links.get("next", {}).get("url"); params = {}
    return out


class HttpFile(io.RawIOBase):
    """Seekable read-only file over HTTP Range requests (what pyarrow needs)."""
    def __init__(self, url):
        self.url, self.pos = url, 0
        r = requests.head(url, headers=_hdr(), allow_redirects=True, timeout=60)
        r.raise_for_status()
        self.size = int(r.headers["Content-Length"]); self.n_req = 0; self.n_bytes = 0
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]; return self.pos
    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0: return 0
        h = _hdr(); h["Range"] = f"bytes={self.pos}-{self.pos+n-1}"
        r = requests.get(self.url, headers=h, allow_redirects=True, timeout=120)
        r.raise_for_status()
        data = r.content[:n]; b[:len(data)] = data
        self.pos += len(data); self.n_req += 1; self.n_bytes += len(data)
        return len(data)


def open_parquet(repo, path, rev="main"):
    f = HttpFile(f"{HF}/datasets/{repo}/resolve/{rev}/{path}")
    return f, pq.ParquetFile(io.BufferedReader(f, buffer_size=4 << 20))


def read_rows(pf, cols, n, rg=None, seed=0):
    rg = random.Random(seed).randrange(pf.num_row_groups) if rg is None else rg
    t = pf.read_row_group(rg, columns=cols)
    rows = t.to_pylist()
    random.Random(seed).shuffle(rows)
    return rows[:n], rg


def fetch_image(url, timeout=15):
    try:
        r = requests.get(url, headers=UA, timeout=timeout, allow_redirects=True)
        if r.status_code != 200 or not r.content: return None, f"http {r.status_code}"
        Image.open(io.BytesIO(r.content)).verify()
        return r.content, None
    except Exception as ex:
        return None, f"{type(ex).__name__}"


def thumb(b, box=320, q=82):
    im = Image.open(io.BytesIO(b)); im.load()
    W, H = im.size
    im = im.convert("RGB"); im.thumbnail((box, box), Image.LANCZOS)
    o = io.BytesIO(); im.save(o, "JPEG", quality=q)
    return base64.b64encode(o.getvalue()).decode(), W, H


def get_nested(d, key):
    for k in key.split("."):
        if d is None: return None
        if isinstance(d, list): d = d[int(k)] if k.isdigit() and int(k) < len(d) else None
        else: d = d.get(k) if isinstance(d, dict) else None
    return d


def sample_dataset(spec, n=20, seed=0):
    """spec: {name, repo, file, image_col | url_col, caption_cols[], extra_cols[], note}"""
    t0 = time.time(); f, pf = open_parquet(spec["repo"], spec["file"], spec.get("rev", "main"))
    want = [c for c in [spec.get("image_col"), spec.get("url_col"), *spec.get("caption_cols", []),
                        *spec.get("extra_cols", [])] if c]
    schema_cols = set(pf.schema_arrow.names)
    want_top = [c.split(".")[0] for c in want]
    missing = [c for c in want_top if c not in schema_cols]
    if missing: raise KeyError(f"columns {missing} not in {sorted(schema_cols)}")
    rows, rg = read_rows(pf, sorted(set(want_top)), n * 2 if spec.get("url_col") else n, seed=seed)
    out, fails = [], []
    def build(row):
        if spec.get("image_col"):
            v = get_nested(row, spec["image_col"])
            b = v.get("bytes") if isinstance(v, dict) else v
            if not b and isinstance(v, dict) and v.get("path"): return None, "path-only"
            err = None
        else:
            url = get_nested(row, spec["url_col"])
            b, err = fetch_image(url)
        if not b: return None, err or "no-bytes"
        try: th, W, H = thumb(b)
        except Exception as ex: return None, f"decode {type(ex).__name__}"
        caps = {}
        for c in spec.get("caption_cols", []):
            v = get_nested(row, c)
            if isinstance(v, list): v = " | ".join(str(x) for x in v[:5])
            if v is not None: caps[c] = str(v)
        extra = {c: get_nested(row, c) for c in spec.get("extra_cols", [])}
        return dict(thumb=th, w=W, h=H, caps=caps, extra={k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v)[:200]) for k, v in extra.items()}), None
    with ThreadPoolExecutor(16) as ex:
        for r, err in ex.map(build, rows):
            if r: out.append(r)
            else: fails.append(err)
            if len(out) >= n: break
    return dict(name=spec["name"], repo=spec["repo"], file=spec["file"], note=spec.get("note", ""),
                row_group=rg, n_row_groups=pf.num_row_groups, rows_in_file=pf.metadata.num_rows,
                file_bytes=f.size, bytes_read=f.n_bytes, http_requests=f.n_req,
                n_tried=len(out) + len(fails), n_ok=len(out), fails=fails, secs=round(time.time() - t0, 1),
                samples=out[:n], columns=pf.schema_arrow.names)


# ----------------------------------------------------------------------------------------
# Non-parquet sources. Each returns a list of row dicts shaped like a parquet row so the
# same spec fields (image_col / url_col / caption_cols) apply.
# ----------------------------------------------------------------------------------------
import gzip, tarfile, sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _range_get(url, nbytes, auth=None):
    auth = ("huggingface.co" in url) if auth is None else auth
    h = _hdr(auth); h["Range"] = f"bytes=0-{nbytes-1}"
    r = requests.get(url, headers=h, allow_redirects=True, timeout=300)
    r.raise_for_status()
    return r.content


def rows_tar_head(url, nbytes, gz=False, json_key=None):
    """Read the first `nbytes` of a (possibly gzipped) tar and pair <key>.jpg/.png with
    <key>.json or .txt. Truncated final member is dropped. -> [{key, image{bytes}, meta}]"""
    raw = _range_get(url, nbytes)
    if gz:
        try: raw = gzip.GzipFile(fileobj=io.BytesIO(raw)).read()
        except (EOFError, OSError, gzip.BadGzipFile): 
            d = gzip._GzipReader(io.BytesIO(raw)); out = bytearray()
            try:
                while True:
                    chunk = d.read(1 << 20)
                    if not chunk: break
                    out += chunk
            except Exception: pass
            raw = bytes(out)
    by = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r|") as tf:
            for m in tf:
                if not m.isfile(): continue
                base, _, ext = m.name.rpartition(".")
                base = os.path.basename(base)
                try: data = tf.extractfile(m).read()
                except Exception: break
                if len(data) != m.size: break
                by.setdefault(base, {})[ext.lower()] = data
    except Exception:
        pass
    rows = []
    for k, d in by.items():
        img = next((d[e] for e in ("jpg", "jpeg", "png", "webp") if e in d), None)
        if img is None: continue
        meta = {}
        if "json" in d:
            try: meta = json.loads(d["json"])
            except Exception: meta = {}
        elif "txt" in d: meta = {"txt": d["txt"].decode("utf-8", "ignore")}
        rows.append(dict(key=k, image={"bytes": img}, **meta))
    return rows


def rows_tar_json(url, nbytes):
    """Tar of small .json records (megalith-10m-sharecap style)."""
    raw = _range_get(url, nbytes); rows = []
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r|") as tf:
            for m in tf:
                if not m.isfile() or not m.name.endswith(".json"): continue
                try: data = tf.extractfile(m).read()
                except Exception: break
                if len(data) != m.size: break
                try: rows.append(json.loads(data))
                except Exception: pass
    except Exception: pass
    return rows


def rows_jsonl_head(url, nbytes):
    raw = _range_get(url, nbytes).decode("utf-8", "ignore").split("\n")[:-1]
    rows = []
    for l in raw:
        try: rows.append(json.loads(l))
        except Exception: pass
    return rows


def sample_generic(spec, n=20, seed=0):
    t0 = time.time(); kind = spec["kind"]; url = spec["url"]; nbytes = int(spec.get("head_mb", 8) * 1e6)
    if kind == "tar_head": rows = rows_tar_head(url, nbytes, gz=False)
    elif kind == "targz_head": rows = rows_tar_head(url, nbytes, gz=True)
    elif kind == "tar_json": rows = rows_tar_json(url, nbytes)
    elif kind == "jsonl_head": rows = rows_jsonl_head(url, nbytes)
    elif kind == "tfrecord_head":
        from tfrecord_peek import iter_records
        rows = []
        for ex in iter_records(_range_get(url, nbytes)):
            caps = [c.decode() for c in ex.get("caption", [])]
            rows.append({"image": {"bytes": ex["image"][0]}, "caption": caps[0] if caps else "",
                         "caption_alt": caps[1] if len(caps) > 1 else "", "n_captions": len(caps),
                         "image_name": ex.get("image_name", [b""])[0].decode()})
    else: raise ValueError(kind)
    if spec.get("join"):        # optional side table: {url, key_col, caption_cols} matched on key
        j = spec["join"]; side = rows_jsonl_head(j["url"], int(j.get("head_mb", 8) * 1e6))
        idx = {str(r.get(j["key_col"])): r for r in side}
        for r in rows:
            s = idx.get(str(r.get(spec.get("key_col", "key"))))
            if s: r.update({c: s.get(c) for c in j["caption_cols"]})
    random.Random(seed).shuffle(rows)
    out, fails = [], []
    def build(row):
        if spec.get("image_col"):
            v = get_nested(row, spec["image_col"]); b = v.get("bytes") if isinstance(v, dict) else v; err = None
        else:
            b, err = fetch_image(get_nested(row, spec["url_col"]))
        if not b: return None, err or "no-bytes"
        try: th, W, H = thumb(b)
        except Exception as ex: return None, f"decode {type(ex).__name__}"
        caps = {}
        for c in spec.get("caption_cols", []):
            v = get_nested(row, c)
            if isinstance(v, list): v = " | ".join(str(x) for x in v[:5])
            if v is not None: caps[c] = str(v)
        extra = {c: get_nested(row, c) for c in spec.get("extra_cols", [])}
        return dict(thumb=th, w=W, h=H, caps=caps, extra={k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v)[:200]) for k, v in extra.items()}), None
    with ThreadPoolExecutor(16) as ex:
        for r, err in ex.map(build, rows[: (n * 2 if spec.get("url_col") else n)]):
            if r: out.append(r)
            else: fails.append(err)
            if len(out) >= n: break
    return dict(name=spec["name"], repo=spec.get("repo", ""), file=url, note=spec.get("note", ""),
                rows_in_head=len(rows), bytes_read=nbytes, file_bytes=nbytes, http_requests=1, n_tried=len(out) + len(fails), n_ok=len(out),
                fails=fails, secs=round(time.time() - t0, 1), samples=out[:n], columns=sorted(set(k for r in rows[:5] for k in r)))


# patch the dispatcher: specs with "kind" go through sample_generic
_sample_parquet = sample_dataset
def sample_dataset(spec, n=20, seed=0):
    return sample_generic(spec, n, seed) if spec.get("kind") else _sample_parquet(spec, n, seed)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "tree":
        repo = sys.argv[2]; mx = int(sys.argv[sys.argv.index("--max") + 1]) if "--max" in sys.argv else 40
        fs = tree(repo)
        tot = sum(x.get("size", 0) for x in fs)
        print(f"{repo}: {len(fs)} entries, {tot/1e9:.1f} GB")
        exts = {}
        for x in fs:
            e = x["path"].rsplit(".", 1)[-1] if "." in x["path"] else "?"
            exts[e] = exts.get(e, 0) + 1
        print("  by ext:", dict(sorted(exts.items(), key=lambda z: -z[1])[:6]))
        for x in fs[:mx]: print(f"  {x.get('size',0)/1e6:9.1f} MB  {x['path']}")
    elif cmd == "schema":
        repo, path = sys.argv[2], sys.argv[3]
        f, pf = open_parquet(repo, path)
        print(f"{path}: {f.size/1e6:.1f} MB, {pf.metadata.num_rows} rows, {pf.num_row_groups} row groups")
        print(pf.schema_arrow)
        t = pf.read_row_group(0)
        row = t.slice(0, 1).to_pylist()[0]
        for k, v in row.items():
            if isinstance(v, (bytes, bytearray)): v = f"<{len(v)} bytes>"
            elif isinstance(v, dict) and "bytes" in v: v = f"<image {len(v['bytes'] or b'')} bytes, path={v.get('path')}>"
            print(f"  {k}: {str(v)[:300]}")
        print(f"  [{f.n_req} requests, {f.n_bytes/1e6:.1f} MB read]")
    elif cmd == "sample":
        specs = json.load(open(sys.argv[2])); outp = sys.argv[3]
        n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 20
        res = json.load(open(outp)) if os.path.exists(outp) else {}
        only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
        for s in specs:
            if only and s["name"] not in only: continue
            try:
                r = sample_dataset(s, n=n)
                res[s["name"]] = r
                print(f"  {s['name']:28s} ok {r['n_ok']}/{r['n_tried']}  read {r['bytes_read']/1e6:.1f} MB of {r['file_bytes']/1e6:.0f} MB in {r['secs']}s  fails={dict((x, r['fails'].count(x)) for x in set(r['fails']))}", flush=True)
            except Exception as ex:
                print(f"  {s['name']:28s} ERROR {type(ex).__name__}: {str(ex)[:200]}", flush=True)
                res[s["name"]] = dict(name=s["name"], repo=s["repo"], error=f"{type(ex).__name__}: {str(ex)[:300]}")
            json.dump(res, open(outp, "w"))

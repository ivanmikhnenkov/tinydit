"""Dataset access for training: bucketed latent memmaps + captions, T5 encoded live.

Layout written by ingest.py (notes/CONTRACTS.md): out/cache/<name>/<bucket>/{lat.npy, rows.jsonl,
val_images.npy} plus meta.json and stats.json. Every batch is one aspect-ratio bucket; buckets are
drawn with P(b) = sum_s w_s P(b|s) and rows inside a batch follow P(s|b), so the global source mix
is exactly the configured weights whatever the per-source bucket shapes are.
"""
from __future__ import annotations
import json, os, queue, threading
import numpy as np, torch
from . import text as T

SOURCE_W = {"pexels": .60, "flux": .25, "coco": .15}


class BucketCache:
    """Latents on disk (memmap), captions in RAM, T5 live."""
    def __init__(self, root, device="cuda", weights=SOURCE_W, models="out/models"):
        meta = json.load(open(os.path.join(root, "meta.json")))
        self.root, self.dev, self.buckets = root, device, meta["buckets"]         # name -> [W, H]
        self.lat, self.rows, self.train_idx, self.val_idx = {}, {}, {}, {}
        for b in list(self.buckets):
            p = os.path.join(root, b, "rows.jsonl")
            if not os.path.exists(p): continue
            rows = [json.loads(l) for l in open(p)]
            if not rows: continue
            lat = np.load(os.path.join(root, b, "lat.npy"), mmap_mode="r")
            n = min(len(rows), lat.shape[0]); rows = rows[:n]
            self.lat[b], self.rows[b] = lat, rows
            self.train_idx[b] = {s: np.array([i for i, r in enumerate(rows) if r["src"] == s and not r["val"]], dtype=np.int64)
                                 for s in weights}
            self.val_idx[b] = np.array([i for i, r in enumerate(rows) if r["val"]], dtype=np.int64)
        self.names = [b for b in self.buckets if b in self.rows]
        assert self.names, f"no buckets with rows under {root}"
        self.sources = [s for s in weights if any(len(self.train_idx[b][s]) for b in self.names)]
        assert self.sources, f"no training rows under {root} (all rows flagged val?)"
        w = np.array([weights[s] for s in self.sources]); w = w / w.sum()
        cnt = np.array([[len(self.train_idx[b][s]) for b in self.names] for s in self.sources], dtype=np.float64)
        p_b_given_s = cnt / np.maximum(cnt.sum(1, keepdims=True), 1)                  # (S, B)
        self.p_b = (w[:, None] * p_b_given_s).sum(0); self.p_b /= self.p_b.sum()       # (B,)
        self.p_s_given_b = (w[:, None] * p_b_given_s) / np.maximum((w[:, None] * p_b_given_s).sum(0, keepdims=True), 1e-12)
        self.n_train = int(cnt.sum()); self.n_val = int(sum(len(v) for v in self.val_idx.values()))
        st = json.load(open(os.path.join(root, "stats.json")))
        self.mean = torch.tensor(st["mean"], device=device).view(1, -1, 1, 1).float()
        self.std = torch.tensor(st["std"], device=device).view(1, -1, 1, 1).float()
        self.tok, self.enc = T.load(os.path.join(models, "t5"))
        self._null = T.embed(self.tok, self.enc, [""], T.MAX_SHORT)
        print("  cache: " + ", ".join(f"{b} {len(self.rows[b]):,} ({self.buckets[b][0]}x{self.buckets[b][1]}, p={self.p_b[i]:.2f})"
                                     for i, b in enumerate(self.names)), flush=True)
        print(f"  sources {dict(zip(self.sources, [round(x, 3) for x in w]))} | train {self.n_train:,} / val {self.n_val:,}", flush=True)
        self._q: queue.Queue = queue.Queue(maxsize=3); self._bs = None; self._thread = None

    # ---- CPU side: pick rows, gather latents ------------------------------------------------
    def _pick(self, bs, cfg_drop, p_long, rng):
        bi = rng.choice(len(self.names), p=self.p_b); b = self.names[bi]
        src = rng.choice(len(self.sources), size=bs, p=self.p_s_given_b[:, bi])
        idx = np.array([rng.choice(self.train_idx[b][self.sources[s]]) for s in src])
        order = np.argsort(idx); idx = idx[order]                                          # sorted reads are friendlier
        z = torch.from_numpy(np.ascontiguousarray(self.lat[b][idx])).pin_memory()
        texts, is_long = [], []
        for i in idx:
            r = self.rows[b][int(i)]; u = rng.random()
            if u < cfg_drop: texts.append(""); is_long.append(False)
            elif u < cfg_drop + p_long: texts.append(rng.choice(r["long"])); is_long.append(True)
            else: texts.append(rng.choice(r["short"])); is_long.append(False)
        return b, z, texts, is_long

    def _worker(self, bs, cfg_drop, p_long, seed):
        rng = np.random.default_rng(seed)
        while True:
            self._q.put(self._pick(bs, cfg_drop, p_long, rng))

    def start(self, bs, cfg_drop, p_long, seed=0):
        self._thread = threading.Thread(target=self._worker, args=(bs, cfg_drop, p_long, seed), daemon=True)
        self._thread.start()

    # ---- GPU side ------------------------------------------------------------------------------
    def _encode(self, texts, is_long):
        e, m = T.embed_mixed(self.tok, self.enc, texts, is_long, self.dev)
        e2 = torch.zeros(e.shape[0], T.MAX_LONG, T.D_MODEL, device=self.dev, dtype=torch.float32); e2[:, :e.shape[1]] = e.float()
        m2 = torch.zeros(e.shape[0], T.MAX_LONG, device=self.dev, dtype=torch.bool); m2[:, :m.shape[1]] = m
        return e2, m2                                       # fixed width: one compiled graph per bucket

    def batch(self):
        b, z, texts, is_long = self._q.get()
        z = ((z.to(self.dev, non_blocking=True).float() - self.mean) / self.std)
        ctx, msk = self._encode(texts, is_long)
        return b, z, ctx, msk

    def val_batches(self, bs=128, rng_seed=99):
        """Yield (bucket, source, z, ctx, msk) over all val rows, grouped by bucket and source."""
        rng = np.random.default_rng(rng_seed)
        for b in self.names:
            vi = self.val_idx[b]
            if len(vi) == 0: continue
            by_src = {}
            for i in vi: by_src.setdefault(self.rows[b][int(i)]["src"], []).append(int(i))
            for s, ids in by_src.items():
                for k in range(0, len(ids), bs):
                    idx = np.array(sorted(ids[k:k+bs]))
                    z = torch.from_numpy(np.ascontiguousarray(self.lat[b][idx])).to(self.dev).float()
                    z = (z - self.mean) / self.std
                    texts, is_long = [], []
                    for i in idx:
                        r = self.rows[b][int(i)]; lg = rng.random() < 0.5
                        texts.append(rng.choice(r["long"] if lg else r["short"])); is_long.append(lg)
                    ctx, msk = self._encode(texts, is_long)
                    yield b, s, z, ctx, msk

    def val_prompts(self, n_per_bucket, rng_seed=7):
        """Held-out captions for eval grids/metrics: [(bucket, caption, ref_image_index)]."""
        rng = np.random.default_rng(rng_seed); out = []
        for b in self.names:
            vi = self.val_idx[b]
            if len(vi) == 0: continue
            take = rng.choice(len(vi), size=min(n_per_bucket, len(vi)), replace=False)
            for j in sorted(take):
                r = self.rows[b][int(vi[j])]
                out.append((b, rng.choice(r["long"] if rng.random() < 0.5 else r["short"]), int(j)))
        return out

    def val_images(self, b):
        p = os.path.join(self.root, b, "val_images.npy")
        return np.load(p, mmap_mode="r") if os.path.exists(p) else None

    def null(self):
        return self._null[0].float(), self._null[1]

    def text_for(self, texts):
        return self._encode(texts, [len(t.split()) > 30 for t in texts])

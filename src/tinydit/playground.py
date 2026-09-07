"""Local playground server: generate from a prompt with the newest EMA snapshot, and expose what the
model does while it samples (intermediate states and predictions, cross-attention per word, the
register tokens). Runs on the host next to the trainer; GPU work is serialised.

    PYTHONPATH=src .venv/bin/python -m tinydit.playground --run run1 --port 7181
    → http://localhost:7181/
"""
from __future__ import annotations
import argparse, base64, glob, io, json, math, os, threading, time
import numpy as np, torch, torch.nn.functional as F
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from . import ae as AE, text as T, schedule, rope
from .model import TinyDiT, CONFIGS
from .sample import decode_chunked, pad_ctx

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root


def b64_u8(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a.astype(np.uint8)).tobytes()).decode()


def png_data_url(img_u8: np.ndarray) -> str:
    from PIL import Image
    b = io.BytesIO(); Image.fromarray(img_u8.transpose(1, 2, 0)).save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def row_norm_u8(a: torch.Tensor) -> np.ndarray:
    """(R, C) -> uint8 with every row scaled so its max is 255."""
    a = a.float(); a = a / a.amax(dim=-1, keepdim=True).clamp_min(1e-12)
    return (a * 255).round().clamp(0, 255).byte().cpu().numpy()


class Engine:
    def __init__(self, run: str, cache: str, models: str = "out/models", device="cuda"):
        self.run, self.cache, self.models, self.dev = run, cache, models, device
        self.lock = threading.Lock()
        meta = json.load(open(os.path.join("out", "runs", run, "meta.json")))
        self.config = meta["config"]["config"]; self.train_args = meta["config"]
        self.buckets = meta.get("buckets", {})
        st = json.load(open(os.path.join(cache, "stats.json")))
        self.mean = torch.tensor(st["mean"], device=device).view(1, -1, 1, 1).float()
        self.std = torch.tensor(st["std"], device=device).view(1, -1, 1, 1).float()
        self.tok, self.enc = T.load(os.path.join(models, "t5"))
        self.aem = AE.load("flux2", os.path.join(models, "flux2"))
        self.lat_ch = AE.SPECS["flux2"]["ch"]
        self.model = TinyDiT(latent_ch=self.lat_ch, ctx_dim=T.D_MODEL, **CONFIGS[self.config]).to(device).eval()
        self.snapshot = None; self.step = 0
        self.reload()
        self.null_ctx, self.null_msk = T.embed(self.tok, self.enc, [""], T.MAX_SHORT)
        self._hooks = []

    def reload(self):
        snaps = sorted(glob.glob(os.path.join("out", "runs", self.run, "ema_*.safetensors")))
        if not snaps: raise FileNotFoundError("no EMA snapshot yet")
        from safetensors.torch import load_file
        sd = load_file(snaps[-1]); own = self.model.state_dict()
        self.model.load_state_dict({k: v.to(own[k].dtype) for k, v in sd.items()}, strict=True)
        self.snapshot = os.path.basename(snaps[-1]); self.step = int(self.snapshot[4:11])
        return self.info()

    def info(self):
        m = self.model
        return {"run": self.run, "step": self.step, "snapshot": self.snapshot,
                "defaults": {"steps": int(self.train_args.get("eval_steps", 20)), "cfg": float(self.train_args.get("cfg", 4.0)),
                             "shift": float(self.train_args.get("shift", 2.8)), "width": 256, "height": 256, "seed": 0},
                "sizes": [list(v) for v in self.buckets.values()] or [[256, 256]],
                "max_tokens": T.MAX_LONG, "n_registers": m.n_registers, "register_block": m.register_block,
                "n_null": m.blocks[0].cross.n_null, "depth": len(m.blocks)}

    # ------------------------------------------------------------------ attention capture (forward hooks)
    def _install_hooks(self, store: dict, N: int):
        m = self.model
        def self_hook(bi):
            def hook(mod, args, out):
                x, cos, sin = args[0][:1], args[1], args[2]               # conditional sample only
                B, Nt, _ = x.shape
                q, k, v = mod.qkv(x.float()).chunk(3, dim=-1)
                q = mod.qn(q.view(B, Nt, mod.h, mod.dh).transpose(1, 2)); k = mod.kn(k.view(B, Nt, mod.h, mod.dh).transpose(1, 2))
                q, k = rope.apply(q, cos, sin), rope.apply(k, cos, sin)
                att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(mod.dh), dim=-1)[0].mean(0)   # (Nt, Nt)
                if Nt > N:
                    store.setdefault("img2reg", {})[bi] = att[:N, N:].sum(-1)                        # (N,)
                    store.setdefault("reg2img", {})[bi] = att[N:, :N]                                # (R, N)
            return hook
        def cross_hook(bi):
            def hook(mod, args, out):
                x, ctx, msk = args[0][:1], args[1][:1], args[2][:1]
                B, Nt, _ = x.shape
                k, v = mod.kv(ctx.float()).chunk(2, dim=-1)
                if mod.null_kv is not None:
                    nk = mod.null_kv[0].float().unsqueeze(0).expand(B, -1, -1); k = torch.cat([k, nk], 1)
                    msk = torch.cat([msk, msk.new_ones(B, mod.n_null)], 1)
                L = k.shape[1]
                q = mod.qn(mod.q(x.float()).view(B, Nt, mod.h, mod.dh).transpose(1, 2))
                k = mod.kn(k.view(B, L, mod.h, mod.dh).transpose(1, 2))
                logits = q @ k.transpose(-1, -2) / math.sqrt(mod.dh)
                logits = logits.masked_fill(~msk[:, None, None, :], float("-inf"))
                att = torch.softmax(logits, dim=-1)[0].mean(0)[:N]                                  # (N, L)
                keep = msk[0]
                store.setdefault("cross", {})[bi] = att[:, keep]                                     # (N, L_eff+null)
            return hook
        def block_hook(bi):
            def hook(mod, args, out):
                x = out[:1].float()
                if x.shape[1] > N:
                    store.setdefault("regnorm", {})[bi] = (x[0, N:].norm(dim=-1), x[0, :N].norm(dim=-1).mean())
            return hook
        for bi, blk in enumerate(m.blocks):
            self._hooks += [blk.attn.register_forward_hook(self_hook(bi)), blk.cross.register_forward_hook(cross_hook(bi)),
                            blk.register_forward_hook(block_hook(bi))]

    def _remove_hooks(self):
        for h in self._hooks: h.remove()
        self._hooks = []

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def generate(self, prompt: str, width=256, height=256, steps=20, cfg=4.0, shift=2.8, seed=0,
                 n_images=1, trajectory=8, attention=False, attention_steps=None):
        t0 = time.time()
        width, height = max(128, min(512, width // 16 * 16)), max(128, min(512, height // 16 * 16))
        steps = max(1, min(100, int(steps))); n_images = max(1, min(4, int(n_images)))
        H, W = height // 8, width // 8; gh, gw = H // 2, W // 2; N = gh * gw
        is_long = len(prompt.split()) > 30
        ctx, msk = T.embed_mixed(self.tok, self.enc, [prompt] * n_images, [is_long] * n_images, self.dev)
        ctx = ctx.float(); ids = self.tok([prompt], truncation=True, max_length=T.MAX_LONG if is_long else T.MAX_SHORT).input_ids[0]
        tokens = self.tok.convert_ids_to_tokens(ids)[:int(msk[0].sum())] + [f"<null {i+1}>" for i in range(self.model.blocks[0].cross.n_null)]
        nctx, nmsk = pad_ctx(self.null_ctx.float(), self.null_msk, ctx.shape[1])
        ts = schedule.shift(steps, shift, device=self.dev) if shift and shift != 1 else schedule.uniform(steps, device=self.dev)
        g = torch.Generator(device=self.dev).manual_seed(int(seed))
        x = torch.randn(n_images, self.lat_ch, H, W, device=self.dev, generator=g)
        traj_steps = sorted(set(int(round(i)) for i in np.linspace(0, steps - 1, trajectory))) if trajectory else []
        att_steps = sorted(set(int(s) for s in (attention_steps or [1, steps // 3, 2 * steps // 3]) if 0 <= int(s) < steps)) if attention else []
        traj, att_out, reg_stats = [], [], []
        for i in range(steps):
            t = ts[i].expand(n_images); dt = ts[i + 1] - ts[i]
            store = {}
            if i in att_steps: self._install_hooks(store, N)
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    v = self.model(torch.cat([x, x]), torch.cat([t, t]),
                                   torch.cat([ctx, nctx.expand(n_images, -1, -1)]), torch.cat([msk, nmsk.expand(n_images, -1)])).float()
            finally:
                if i in att_steps: self._remove_hooks()
            vc, vu = v.chunk(2); vg = vu + cfg * (vc - vu)
            if i in traj_steps:
                x1 = x[:1] + (1 - ts[i]) * vg[:1]
                traj.append({"step": i, "t": float(ts[i]),
                             "x_t": png_data_url(decode_chunked(self.aem, x[:1], self.mean, self.std)[0]),
                             "x1_hat": png_data_url(decode_chunked(self.aem, x1, self.mean, self.std)[0])})
            if i in att_steps:
                blocks = []
                for bi in range(len(self.model.blocks)):
                    cr = store["cross"][bi]
                    blocks.append({"block": bi, "cross": b64_u8(row_norm_u8(cr)),
                                   "cross_mass": (cr.sum(0) / N).tolist(),
                                   "img2reg": b64_u8(row_norm_u8(store["img2reg"][bi][None])[0]) if bi in store.get("img2reg", {}) else None,
                                   "reg2img": b64_u8(row_norm_u8(store["reg2img"][bi])) if bi in store.get("reg2img", {}) else None})
                    if bi in store.get("regnorm", {}):
                        rn, im = store["regnorm"][bi]
                        reg_stats.append({"step": i, "block": bi, "norms": rn.tolist(), "img_norm": float(im),
                                          "img2reg_mean": float(store["img2reg"][bi].mean())})
                att_out.append({"step": i, "t": float(ts[i]), "blocks": blocks})
            x = x + vg * dt
        imgs = decode_chunked(self.aem, x, self.mean, self.std)
        torch.cuda.empty_cache()
        return {"images": [png_data_url(im) for im in imgs], "ms": int((time.time() - t0) * 1000),
                "schedule": [float(v) for v in ts], "grid": [gh, gw], "tokens": tokens,
                "trajectory": traj, "attention": att_out, "register_stats": reg_stats,
                "params": {"prompt": prompt, "width": width, "height": height, "steps": steps, "cfg": cfg, "shift": shift, "seed": seed}}


def make_handler(engine: Engine, page: str):
    class H(BaseHTTPRequestHandler):
        def _json(self, code, obj):
            b = json.dumps(obj).encode(); self.send_response(code)
            self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(b)
        def do_GET(self):
            p = self.path.split("?")[0]
            if p in ("/", "/playground.html"):
                b = open(page, "rb").read(); self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(b)))
                self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(b)
            elif p == "/api/info": self._json(200, engine.info())
            else: self._json(404, {"error": "not found"})
        def do_POST(self):
            p = self.path.split("?")[0]
            n = int(self.headers.get("Content-Length", 0)); body = json.loads(self.rfile.read(n) or b"{}")
            try:
                if p == "/api/reload":
                    with engine.lock: self._json(200, engine.reload())
                elif p == "/api/generate":
                    if not str(body.get("prompt", "")).strip(): return self._json(400, {"error": "empty prompt"})
                    kw = {k: body[k] for k in ("width", "height", "steps", "cfg", "shift", "seed", "n_images", "trajectory", "attention", "attention_steps") if k in body}
                    with engine.lock: self._json(200, engine.generate(str(body["prompt"]), **kw))
                else: self._json(404, {"error": "not found"})
            except Exception as ex:
                import traceback; traceback.print_exc(); self._json(500, {"error": f"{type(ex).__name__}: {ex}"})
        def log_message(self, fmt, *args): print(f"  {self.address_string()} {fmt % args}", flush=True)
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="run1"); ap.add_argument("--cache", default="out/cache/run1")
    ap.add_argument("--port", type=int, default=7181); ap.add_argument("--page", default=os.path.join(HERE, "playground.html"))
    a = ap.parse_args()
    eng = Engine(a.run, a.cache)
    print(f"  playground: {eng.snapshot} (step {eng.step}) -> http://127.0.0.1:{a.port}/", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(eng, a.page)).serve_forever()


if __name__ == "__main__":
    main()

"""Frozen autoencoder wrappers. Both candidates are KL autoencoders, so the
interface is identical; only the spatial factor and channel count differ."""
from __future__ import annotations
import os, subprocess

SPECS = {
    "flux2": dict(repo="black-forest-labs/FLUX.2-dev", subfolder="vae",
                  cls="AutoencoderKLFlux2", f=8, ch=32, gated=True),
    "sd15":  dict(repo="stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="vae",
                  cls="AutoencoderKL", f=8, ch=4, gated=False),
    # Krea 2 reuses Qwen-Image's autoencoder: a Wan-lineage 3D VAE. Same f8c16 geometry
    # as FLUX.1, different architecture — it expects (B,C,T,H,W), so stills get T=1.
    "krea2": dict(repo="krea/Krea-2-Turbo", subfolder="vae",
                  cls="AutoencoderKLQwenImage", f=8, ch=16, gated=True, temporal=True),
}
TOKEN_PATHS = (
    os.path.expanduser("~/volume/.tinydit_token"),   # HF_TOKEN=... in a 0600 file outside the repo
    "/home/ivan/volume/.tinydit_token",              # same file as seen from the container
)


def hf_token() -> str | None:
    """Kept out of argv and out of the HTTP-served tree; read from a 0600 file."""
    for p in TOKEN_PATHS:
        if os.path.exists(p):
            for line in open(p):
                if line.startswith("HF_TOKEN="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("HF_TOKEN")


def fetch(name: str, dest: str) -> str:
    """Download an AE with curl. `huggingface_hub.snapshot_download` hangs on this
    machine (xet backend), curl does not."""
    spec = SPECS[name]
    d = os.path.join(dest, name)
    os.makedirs(d, exist_ok=True)
    tok = hf_token() if spec["gated"] else None
    for fn in ("config.json", "diffusion_pytorch_model.safetensors"):
        out = os.path.join(d, fn)
        if os.path.exists(out) and os.path.getsize(out) > 0:
            continue
        url = f"https://huggingface.co/{spec['repo']}/resolve/main/{spec['subfolder']}/{fn}"
        cmd = ["curl", "-sL", "--fail", "-o", out]
        if tok:
            cmd += ["-H", f"Authorization: Bearer {tok}"]
        subprocess.run(cmd + [url], check=True)
        print(f"  fetched {name}/{fn} ({os.path.getsize(out)/1e6:.1f} MB)", flush=True)
    return d


def load(name: str, path: str, device="cuda", dtype=None):
    import torch, diffusers
    dtype = dtype or torch.float32
    cls = getattr(diffusers, SPECS[name]["cls"])
    ae = cls.from_pretrained(path, torch_dtype=dtype).eval().to(device)
    ae.requires_grad_(False)
    ae._tinydit_temporal = SPECS[name].get("temporal", False)
    return ae


def encode(ae, x: torch.Tensor) -> torch.Tensor:
    """x: (B,3,H,W) in [-1,1] -> latent (B,C,H/8,W/8). Posterior mean (mode),
    deterministic: we want a fixed target per image, not a resampled one."""
    import torch
    with torch.no_grad():
        if getattr(ae, "_tinydit_temporal", False):
            return ae.encode(x.unsqueeze(2)).latent_dist.mean.squeeze(2)
        return ae.encode(x).latent_dist.mean


def decode(ae, z: torch.Tensor) -> torch.Tensor:
    """latent -> (B,3,H,W) in [-1,1]."""
    import torch
    with torch.no_grad():
        if getattr(ae, "_tinydit_temporal", False):
            return ae.decode(z.unsqueeze(2)).sample.squeeze(2).clamp(-1, 1)
        return ae.decode(z).sample.clamp(-1, 1)


if __name__ == "__main__":
    import sys
    dest = sys.argv[1] if len(sys.argv) > 1 else "out/models"
    for n in SPECS:
        print(f"== {n} ==")
        fetch(n, dest)

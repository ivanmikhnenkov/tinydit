"""Check the diffusers export against the original code, in fp32 on whatever device is given:
  1. the transformer alone: original v(z, t) == -new(z, 1000 * (1 - t)) on random inputs
  2. end to end: TinyDiTPipeline vs tinydit.sample.generate, same prompt, seed, steps, CFG and shift
     (the original encodes prompts of <=30 words with a 48-token limit; the test prompts are well under it)

    PYTHONPATH=src python hub/parity.py --hub out/hub/tinydit-256 --device cpu
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np, torch
from PIL import Image
from diffusers import DiffusionPipeline
from tinydit import sample as S, text as T

PROMPTS = [("a lighthouse on a rocky coast at sunset", 320, 208),
           ("three green apples on a white plate next to a black coffee cup", 256, 256)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hub", default="out/hub/tinydit-256")
    p.add_argument("--release", default="out/models/tinydit-256/model.safetensors")
    p.add_argument("--t5", default="out/models/t5")
    p.add_argument("--device", default="cpu")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--out", default="out/hub/parity")
    a = p.parse_args()
    dev = a.device
    torch.manual_seed(0)
    pipe = DiffusionPipeline.from_pretrained(a.hub, trust_remote_code=True, torch_dtype=torch.float32).to(dev)
    pipe.set_progress_bar_config(disable=True)
    cfg = pipe.transformer.config
    orig, _ = S.load_model(a.release, cfg.in_channels, "run1", device=dev)

    # 1. transformer
    z = torch.randn(2, cfg.in_channels, 26, 40, device=dev)
    t = torch.tensor([0.13, 0.82], device=dev)
    ctx = torch.randn(2, 11, cfg.text_dim, device=dev)
    msk = torch.ones(2, 11, dtype=torch.bool, device=dev); msk[1, 6:] = False
    with torch.no_grad():
        v0 = orig(z, t, ctx, msk)
        v1 = -pipe.transformer(z, (1 - t) * 1000, ctx, msk).sample
    print(f"transformer: max |diff| {(v0 - v1).abs().max():.2e}, max |v| {v0.abs().max():.2f}")

    # 2. end to end
    tok, enc = T.load(a.t5, device=dev, dtype=torch.float32)
    nctx, nmsk = T.embed(tok, enc, [""], T.MAX_SHORT, device=dev)
    mean = torch.tensor(cfg.latents_mean, device=dev).view(1, -1, 1, 1)
    std = torch.tensor(cfg.latents_std, device=dev).view(1, -1, 1, 1)
    os.makedirs(a.out, exist_ok=True)
    report = []
    for i, (prompt, W, H) in enumerate(PROMPTS):
        t0 = time.time()
        ctx, msk = T.embed_mixed(tok, enc, [prompt], [len(prompt.split()) > 30], device=dev)
        z0 = S.generate(orig, ctx.float(), msk, nctx.float(), nmsk, cfg.in_channels, H // 8, W // 8,
                        a.steps, 4.0, seed=0, shift=2.8, device=dev)
        t1 = time.time()
        z1 = pipe(prompt, width=W, height=H, num_inference_steps=a.steps, guidance_scale=4.0,
                  generator=torch.Generator(dev).manual_seed(0), output_type="latent").images
        t2 = time.time()
        with torch.no_grad():
            imgs = [pipe.vae.decode(z * std + mean).sample for z in (z0, z1)]
        px = [((x[0].clamp(-1, 1).permute(1, 2, 0) + 1) * 127.5).round().byte().cpu().numpy() for x in imgs]
        dz = (z0 - z1).abs().max().item()
        dpx = np.abs(px[0].astype(int) - px[1].astype(int))
        Image.fromarray(np.concatenate(px, 1)).save(os.path.join(a.out, f"parity_{i}.png"))
        r = dict(prompt=prompt, shape=f"{W}x{H}", latent_max_diff=dz, pixel_max_diff=int(dpx.max()),
                 pixel_mean_diff=float(dpx.mean()), sec_original=round(t1 - t0, 1), sec_pipeline=round(t2 - t1, 1))
        report.append(r)
        print(json.dumps(r))
    json.dump(report, open(os.path.join(a.out, "parity.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

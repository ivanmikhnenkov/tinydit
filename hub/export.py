"""Assemble the diffusers layout of the Hub repo ivanmikhnenkov/tinydit-256:

    model_index.json  pipeline_tinydit.py  README.md
    transformer/  config.json (incl. latent whitening stats)  transformer_tinydit.py  diffusion_pytorch_model.safetensors
    text_encoder/ tokenizer/   flan-t5-base encoder (fp32) and its tokenizer files as published by Google
    vae/                       FLUX.2 autoencoder as published in black-forest-labs/FLUX.2-dev (Apache 2.0)
    scheduler/                 FlowMatchEulerDiscreteScheduler, shift 2.8

The transformer weights are the released bf16 EMA file copied byte for byte (same parameter names), after a
strict load into TinyDiTTransformer2DModel. The root model.safetensors / config.json / stats.json of the
original release stay in the repo for tinydit's own code and the Space; `_ignore_files` keeps diffusers from
downloading the root weights a second time. FLUX.2-dev is gated, so fetching its vae/ needs HF_TOKEN.

    python hub/export.py --out out/hub/tinydit-256
    hf upload ivanmikhnenkov/tinydit-256 out/hub/tinydit-256 .
"""
from __future__ import annotations
import argparse, json, os, shutil, sys
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

HERE = os.path.dirname(os.path.abspath(__file__))
T5_FILES = ["config.json", "model.safetensors", "spiece.model", "tokenizer.json",
            "tokenizer_config.json", "special_tokens_map.json"]


def fetch(models: str) -> dict[str, str]:
    os.makedirs(models, exist_ok=True)
    return {
        "release": snapshot_download("ivanmikhnenkov/tinydit-256", local_dir=os.path.join(models, "tinydit-256"),
                                     allow_patterns=["model.safetensors", "config.json", "stats.json"]),
        "t5": snapshot_download("google/flan-t5-base", local_dir=os.path.join(models, "t5"), allow_patterns=T5_FILES),
        "vae": os.path.join(snapshot_download("black-forest-labs/FLUX.2-dev", local_dir=os.path.join(models, "flux2-dev"),
                                              allow_patterns=["vae/*"]), "vae"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="out/hub/tinydit-256")
    p.add_argument("--models", default="out/models", help="where the release, T5 and VAE are downloaded")
    a = p.parse_args()
    src = fetch(a.models)
    os.makedirs(a.out, exist_ok=True)

    sys.path.insert(0, HERE)
    from transformer_tinydit import TinyDiTTransformer2DModel
    rel = json.load(open(os.path.join(src["release"], "config.json")))
    stats = json.load(open(os.path.join(src["release"], "stats.json")))
    model = TinyDiTTransformer2DModel(
        in_channels=rel["latent_ch"], patch_size=rel["patch"], hidden_size=rel["dim"], num_layers=rel["depth"],
        num_attention_heads=rel["heads"], text_dim=rel["ctx_dim"], num_registers=rel["n_registers"],
        register_block=rel["register_block"], num_null_tokens=rel["n_null"],
        latents_mean=stats["mean"], latents_std=stats["std"])
    weights = os.path.join(src["release"], "model.safetensors")
    model.load_state_dict(load_file(weights), strict=True)
    d = os.path.join(a.out, "transformer")
    model.save_config(d)
    shutil.copyfile(weights, os.path.join(d, "diffusion_pytorch_model.safetensors"))
    shutil.copyfile(os.path.join(HERE, "transformer_tinydit.py"), os.path.join(d, "transformer_tinydit.py"))
    print(f"  transformer: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, strict load ok")

    from transformers import T5EncoderModel
    T5EncoderModel.from_pretrained(src["t5"]).save_pretrained(os.path.join(a.out, "text_encoder"))
    os.makedirs(os.path.join(a.out, "tokenizer"), exist_ok=True)
    for fn in T5_FILES[2:]:
        shutil.copyfile(os.path.join(src["t5"], fn), os.path.join(a.out, "tokenizer", fn))
    os.makedirs(os.path.join(a.out, "vae"), exist_ok=True)
    for fn in ("config.json", "diffusion_pytorch_model.safetensors"):
        shutil.copyfile(os.path.join(src["vae"], fn), os.path.join(a.out, "vae", fn))

    from diffusers import FlowMatchEulerDiscreteScheduler
    import diffusers
    FlowMatchEulerDiscreteScheduler(shift=rel["sampler"]["shift"]).save_pretrained(os.path.join(a.out, "scheduler"))
    index = {
        "_class_name": ["pipeline_tinydit", "TinyDiTPipeline"],
        "_diffusers_version": diffusers.__version__,
        "_ignore_files": ["model.safetensors"],   # the original-layout copy at the repo root, for tinydit's own code
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "text_encoder": ["transformers", "T5EncoderModel"],
        "tokenizer": ["transformers", "T5TokenizerFast"],
        "transformer": ["transformer_tinydit", "TinyDiTTransformer2DModel"],
        "vae": ["diffusers", "AutoencoderKLFlux2"],
    }
    json.dump(index, open(os.path.join(a.out, "model_index.json"), "w"), indent=2)
    shutil.copyfile(os.path.join(HERE, "pipeline_tinydit.py"), os.path.join(a.out, "pipeline_tinydit.py"))
    shutil.copyfile(os.path.join(HERE, "model_card.md"), os.path.join(a.out, "README.md"))
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()

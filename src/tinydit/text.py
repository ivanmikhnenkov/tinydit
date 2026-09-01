"""Frozen T5 text encoder.

T5 rather than CLIP: CLIP's text tower is trained contrastively against whole
images, which makes its token states weak at compositional detail. PixArt and
SANA both use T5 encoders for cross-attention for this reason. flan-t5-base has
d_model=768, which matches the DiT width so K/V projections stay square."""
from __future__ import annotations
import os, subprocess

REPO = "google/flan-t5-base"
FILES = ("config.json", "spiece.model", "tokenizer.json",
         "special_tokens_map.json", "tokenizer_config.json", "model.safetensors")
D_MODEL = 768


def fetch(dest: str) -> str:
    os.makedirs(dest, exist_ok=True)
    for fn in FILES:
        out = os.path.join(dest, fn)
        if os.path.exists(out) and os.path.getsize(out) > 0:
            continue
        url = f"https://huggingface.co/{REPO}/resolve/main/{fn}"
        r = subprocess.run(["curl", "-sL", "--fail", "-o", out, url])
        if r.returncode != 0:
            os.path.exists(out) and os.remove(out)
            print(f"  (skip {fn})", flush=True)
        else:
            print(f"  fetched {fn} ({os.path.getsize(out)/1e6:.1f} MB)", flush=True)
    return dest


def load(path: str, device="cuda", dtype=None):
    import torch
    from transformers import T5EncoderModel, AutoTokenizer
    dtype = dtype or torch.bfloat16
    tok = AutoTokenizer.from_pretrained(path)
    enc = T5EncoderModel.from_pretrained(path, torch_dtype=dtype).eval().to(device)
    enc.requires_grad_(False)
    return tok, enc


def embed(tok, enc, captions: list[str], max_len: int, device="cuda"):
    """-> (emb (B,L,768), mask (B,L) bool). Padding is masked out of cross-attention
    rather than zeroed, so the model never attends to pad positions."""
    import torch
    b = tok(captions, padding="max_length", truncation=True,
            max_length=max_len, return_tensors="pt").to(device)
    with torch.no_grad():
        out = enc(input_ids=b.input_ids, attention_mask=b.attention_mask).last_hidden_state
    return out, b.attention_mask.bool()

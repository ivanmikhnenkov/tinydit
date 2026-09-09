"""Frozen T5 text encoder.

T5 rather than CLIP: CLIP's text tower is trained contrastively against whole images, which
makes its token states weak at compositional detail. PixArt and SANA use T5 encoders for
cross-attention for this reason. flan-t5-base has d_model=768. It is small enough to run live
every batch (no embedding cache on disk), which is what makes long captions affordable:
captions are encoded as two sub-batches (long ones up to 128 tokens, short ones up to 48),
each padded only to its own longest member, so short prompts never pay for 128 slots."""
from __future__ import annotations
import os, subprocess

REPO = "google/flan-t5-base"
FILES = ("config.json", "spiece.model", "tokenizer.json",
         "special_tokens_map.json", "tokenizer_config.json", "model.safetensors")
D_MODEL = 768
MAX_LONG, MAX_SHORT = 128, 48


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
    enc = T5EncoderModel.from_pretrained(path, torch_dtype=dtype,
                                         attn_implementation="sdpa").eval().to(device)
    enc.requires_grad_(False)
    return tok, enc


def embed(tok, enc, captions: list[str], max_len: int, device="cuda", pad_to_max=False):
    """-> (emb (B,L,768), mask (B,L) bool). Padding is masked out of cross-attention rather
    than zeroed. L = longest caption in the batch unless pad_to_max."""
    import torch
    b = tok(captions, padding="max_length" if pad_to_max else "longest", truncation=True,
            max_length=max_len, return_tensors="pt").to(device)
    with torch.no_grad():
        out = enc(input_ids=b.input_ids, attention_mask=b.attention_mask).last_hidden_state
    return out, b.attention_mask.bool()


def embed_mixed(tok, enc, captions: list[str], is_long, device="cuda",
                max_long=MAX_LONG, max_short=MAX_SHORT):
    """Encode long and short captions as separate sub-batches, then scatter into one
    (B, L, 768) tensor with L = the longer of the two sub-batch widths.
    is_long: list[bool]; empty strings should be marked short."""
    import torch
    idx_l = [i for i, f in enumerate(is_long) if f]
    idx_s = [i for i, f in enumerate(is_long) if not f]
    parts = []
    if idx_l: parts.append((idx_l, *embed(tok, enc, [captions[i] for i in idx_l], max_long, device)))
    if idx_s: parts.append((idx_s, *embed(tok, enc, [captions[i] for i in idx_s], max_short, device)))
    L = max(e.shape[1] for _, e, _ in parts)
    B = len(captions)
    emb = torch.zeros(B, L, D_MODEL, device=device, dtype=parts[0][1].dtype)
    msk = torch.zeros(B, L, device=device, dtype=torch.bool)
    for idx, e, m in parts:
        ii = torch.tensor(idx, device=device)
        emb[ii, :e.shape[1]] = e; msk[ii, :m.shape[1]] = m
    return emb, msk

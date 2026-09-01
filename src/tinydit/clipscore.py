"""CLIP score: cosine similarity between a generated image and its prompt.

The only metric here that measures *prompt adherence* rather than reconstruction or
optimisation. Loss says how well the flow is fit; CLIP score says whether the picture
matches the words. Values are cosine similarities (roughly 0.15-0.35 in practice),
not probabilities, so only relative movement matters.
"""
from __future__ import annotations
import numpy as np, torch

_M = {}

def load(path="out/models/clip", device="cuda"):
    if "m" not in _M:
        from transformers import CLIPModel, CLIPProcessor
        _M["m"] = CLIPModel.from_pretrained(path).eval().to(device)
        _M["p"] = CLIPProcessor.from_pretrained(path)
        for q in _M["m"].parameters(): q.requires_grad_(False)
    return _M["m"], _M["p"]


@torch.no_grad()
def score(images_u8, prompts, path="out/models/clip", device="cuda", batch=32):
    """images_u8: (N,3,H,W) uint8 array | prompts: list[str] -> (per_image, mean)"""
    m, p = load(path, device)
    from PIL import Image
    out = []
    for i in range(0, len(prompts), batch):
        ims = [Image.fromarray(x.transpose(1, 2, 0)) for x in images_u8[i:i+batch]]
        inp = p(text=prompts[i:i+batch], images=ims, return_tensors="pt",
                padding=True, truncation=True, max_length=77).to(device)
        ie = m.get_image_features(pixel_values=inp["pixel_values"])
        te = m.get_text_features(input_ids=inp["input_ids"],
                                 attention_mask=inp.get("attention_mask"))
        ie = ie / ie.norm(dim=-1, keepdim=True)
        te = te / te.norm(dim=-1, keepdim=True)
        out += (ie * te).sum(-1).float().cpu().tolist()
    return out, float(np.mean(out))

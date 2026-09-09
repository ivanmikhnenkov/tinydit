"""Automatic evaluation metrics. All models are frozen, loaded lazily on first use and released
with unload() so they never share the GPU with the training model for longer than an eval.

  clip_score       CLIP ViT-B/32 image-text cosine: prompt adherence only, ~0.15-0.35
  pickscore        PickScore v1 (CLIP-H fine-tuned on 500k Pick-a-Pic human preferences); higher better, ~18-23
  hpsv2            HPSv2.1 (CLIP-H fine-tuned on Human Preference Dataset v2); higher better, ~0.20-0.30
  frechet          FID (Inception-v3 pool3) and FD-DINOv2 (DINOv2-B CLS) between generated and reference images
  object_accuracy  "a photo of a {COCO class}": fraction where a COCO Faster R-CNN detects that class (conf >= 0.5)

PickScore and HPSv2 are the same family of model that Flow-GRPO will later use as a reward, so their
curves during pretraining are the baselines the RL stage is measured against.
"""
from __future__ import annotations
import os
import numpy as np, torch

MODELS = "out/models"
_M: dict = {}


def unload():
    for k in list(_M): del _M[k]
    torch.cuda.empty_cache()


def _vec(o):
    """transformers >= 4.5x returns output objects from get_*_features; older versions a tensor."""
    if torch.is_tensor(o): return o
    for k in ("image_embeds", "text_embeds", "pooler_output"):
        if getattr(o, k, None) is not None: return getattr(o, k)
    return o[0]


def _pil(images_u8):
    from PIL import Image
    return [Image.fromarray(x.transpose(1, 2, 0)) for x in images_u8]


# ------------------------------------------------------------------ CLIP-family scores
def _clip_like(name, path, device):
    if name not in _M:
        from transformers import CLIPModel, CLIPProcessor
        m = CLIPModel.from_pretrained(path, torch_dtype=torch.float16).eval().to(device)
        m.requires_grad_(False)
        _M[name] = (m, CLIPProcessor.from_pretrained(path))
    return _M[name]


@torch.no_grad()
def _clip_pairs(name, path, images_u8, prompts, device, batch, scale=1.0):
    m, p = _clip_like(name, path, device)
    out = []
    for i in range(0, len(prompts), batch):
        inp = p(text=prompts[i:i+batch], images=_pil(images_u8[i:i+batch]), return_tensors="pt",
                padding=True, truncation=True, max_length=77).to(device)
        ie = _vec(m.get_image_features(pixel_values=inp["pixel_values"].half()))
        te = _vec(m.get_text_features(input_ids=inp["input_ids"], attention_mask=inp.get("attention_mask")))
        ie = ie / ie.norm(dim=-1, keepdim=True); te = te / te.norm(dim=-1, keepdim=True)
        out += (scale * (ie * te).sum(-1)).float().cpu().tolist()
    return out


def clip_score(images_u8, prompts, device="cuda", batch=64):
    return _clip_pairs("clip", os.path.join(MODELS, "clip"), images_u8, prompts, device, batch)


def pickscore(images_u8, prompts, device="cuda", batch=32):
    """PickScore is reported as logit_scale * cosine (the model's logit_scale.exp() ~ 100)."""
    m, _ = _clip_like("pickscore", os.path.join(MODELS, "pickscore"), device)
    return _clip_pairs("pickscore", os.path.join(MODELS, "pickscore"), images_u8, prompts, device, batch,
                       scale=float(m.logit_scale.exp()))


@torch.no_grad()
def hpsv2(images_u8, prompts, device="cuda", batch=32):
    if "hps" not in _M:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained=None, precision="fp16", device=device)
        sd = torch.load(os.path.join(MODELS, "hpsv2", "HPS_v2.1_compressed.pt"), map_location="cpu")
        model.load_state_dict(sd.get("state_dict", sd)); model.eval()
        _M["hps"] = (model, preprocess, open_clip.get_tokenizer("ViT-H-14"))
    model, pre, tok = _M["hps"]
    out = []
    for i in range(0, len(prompts), batch):
        ims = torch.stack([pre(x) for x in _pil(images_u8[i:i+batch])]).to(device).half()
        tx = tok(prompts[i:i+batch]).to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            o = model(ims, tx)
            ie, te = o[0] if isinstance(o, (tuple, list)) else o["image_features"], o[1] if isinstance(o, (tuple, list)) else o["text_features"]
        out += (ie * te).sum(-1).float().cpu().tolist()
    return out


# ------------------------------------------------------------------ Fréchet distances
@torch.no_grad()
def _inception_feats(images_u8, device, batch=64):
    if "inc" not in _M:
        from pytorch_fid.inception import InceptionV3
        import pytorch_fid.inception as pfi
        w = os.path.join(MODELS, "fid", "pt_inception-2015-12-05-6726825d.pth")
        if os.path.exists(w):   # use the local copy instead of downloading
            orig = pfi.fid_inception_v3
            def local():
                m = pfi._inception_v3(num_classes=1008, aux_logits=False, weights=None)
                m.Mixed_5b = pfi.FIDInceptionA(192, pool_features=32); m.Mixed_5c = pfi.FIDInceptionA(256, pool_features=64)
                m.Mixed_5d = pfi.FIDInceptionA(288, pool_features=64); m.Mixed_6b = pfi.FIDInceptionC(768, channels_7x7=128)
                m.Mixed_6c = pfi.FIDInceptionC(768, channels_7x7=160); m.Mixed_6d = pfi.FIDInceptionC(768, channels_7x7=160)
                m.Mixed_6e = pfi.FIDInceptionC(768, channels_7x7=192); m.Mixed_7b = pfi.FIDInceptionE_1(1280); m.Mixed_7c = pfi.FIDInceptionE_2(2048)
                m.load_state_dict(torch.load(w, map_location="cpu")); return m
            pfi.fid_inception_v3 = local
        _M["inc"] = InceptionV3([3], resize_input=True, normalize_input=True).eval().to(device)
    m = _M["inc"]; fs = []
    for i in range(0, len(images_u8), batch):
        x = torch.from_numpy(np.ascontiguousarray(images_u8[i:i+batch])).to(device).float() / 255.0
        fs.append(m(x)[0].squeeze(-1).squeeze(-1).double().cpu())
    return torch.cat(fs).numpy()


@torch.no_grad()
def _dino_feats(images_u8, device, batch=64):
    if "dino" not in _M:
        from transformers import AutoModel
        _M["dino"] = AutoModel.from_pretrained(os.path.join(MODELS, "dinov2"), torch_dtype=torch.float16).eval().to(device)
    m = _M["dino"]; fs = []
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    for i in range(0, len(images_u8), batch):
        x = torch.from_numpy(np.ascontiguousarray(images_u8[i:i+batch])).to(device).float() / 255.0
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False, antialias=True)
        x = ((x - mean) / std).half()
        fs.append(m(pixel_values=x).pooler_output.double().cpu())
    return torch.cat(fs).numpy()


def _frechet(a, b):
    from scipy import linalg
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean = linalg.sqrtm(s1 @ s2)
    if not np.isfinite(covmean).all():
        off = np.eye(s1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((s1 + off) @ (s2 + off))
    covmean = np.asarray(covmean).real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def frechet(gen_u8, ref_u8, device="cuda"):
    """gen_u8, ref_u8: lists/arrays of (3,H,W) uint8 images (shapes may differ between items;
    each item is resized by the feature extractor). -> (fid, fd_dino)"""
    def groups(imgs):   # batch items of identical shape together
        by = {}
        for x in imgs: by.setdefault(x.shape, []).append(x)
        return [np.stack(v) for v in by.values()]
    inc_g = np.concatenate([_inception_feats(g, device) for g in groups(gen_u8)])
    inc_r = np.concatenate([_inception_feats(g, device) for g in groups(ref_u8)])
    din_g = np.concatenate([_dino_feats(g, device) for g in groups(gen_u8)])
    din_r = np.concatenate([_dino_feats(g, device) for g in groups(ref_u8)])
    return _frechet(inc_g, inc_r), _frechet(din_g, din_r)


# ------------------------------------------------------------------ object accuracy
COCO_CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]


def object_prompts(seeds=4):
    return [(f"a photo of a {c}", c) for c in COCO_CLASSES for _ in range(seeds)]


@torch.no_grad()
def object_accuracy(images_u8, classes, device="cuda", conf=0.5, batch=16):
    """images_u8: (N,3,H,W); classes: list of COCO class names, one per image -> (fraction, per-image bools)"""
    if "det" not in _M:
        import torchvision
        os.environ.setdefault("TORCH_HOME", os.path.join(MODELS, "torchhub"))
        w = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        _M["det"] = (torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=w).eval().to(device), w.meta["categories"])
    m, cats = _M["det"]; hits = []
    for i in range(0, len(images_u8), batch):
        x = [torch.from_numpy(np.ascontiguousarray(im)).to(device).float() / 255.0 for im in images_u8[i:i+batch]]
        for out, cls in zip(m(x), classes[i:i+batch]):
            keep = out["scores"] >= conf
            names = {cats[int(l)] for l in out["labels"][keep]}
            hits.append(cls in names)
    return float(np.mean(hits)) if hits else 0.0, hits

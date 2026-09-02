"""Download the frozen evaluation models into out/models/ (idempotent). Token is read from the
0600 token file and sent as a header, never placed on a command line."""
import os, sys, requests
sys.path.insert(0, "src")
from tinydit.ae import hf_token
TOK = hf_token()
H = {"Authorization": f"Bearer {TOK}"} if TOK else {}

def get(url, out):
    if os.path.exists(out) and os.path.getsize(out) > 0: return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with requests.get(url, headers=H, stream=True, timeout=600, allow_redirects=True) as r:
        r.raise_for_status()
        with open(out + ".part", "wb") as f:
            for chunk in r.iter_content(1 << 22): f.write(chunk)
    os.rename(out + ".part", out); print(f"fetched {out} ({os.path.getsize(out)/1e6:.0f} MB)", flush=True)

def hf(repo, files, dest):
    for fn in files: get(f"https://huggingface.co/{repo}/resolve/main/{fn}", os.path.join(dest, fn))

CLIP_AUX = ["config.json", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"]
hf("openai/clip-vit-base-patch32", CLIP_AUX + ["pytorch_model.bin"], "out/models/clip")
hf("yuvalkirstain/PickScore_v1", ["config.json", "model.safetensors"], "out/models/pickscore")
hf("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", [f for f in CLIP_AUX if f != "config.json"], "out/models/pickscore")
hf("xswu/HPSv2", ["HPS_v2.1_compressed.pt"], "out/models/hpsv2")
hf("facebook/dinov2-base", ["config.json", "preprocessor_config.json", "model.safetensors"], "out/models/dinov2")
get("https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth",
    "out/models/fid/pt_inception-2015-12-05-6726825d.pth")
os.environ["TORCH_HOME"] = "out/models/torchhub"
import torchvision
torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="COCO_V1"); print("fetched fasterrcnn", flush=True)
print("ALL METRIC WEIGHTS DONE")

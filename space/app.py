"""Hugging Face Space for tinydit-256 (ZeroGPU): prompt -> image, with an optional view of the
sampling trajectory. Models load once at startup; the GPU is attached per request by ZeroGPU."""
import os, json, io, base64, torch, gradio as gr
try:
    import spaces; GPU = spaces.GPU
except Exception:                                   # local / CPU fallback
    def GPU(*a, **k): return (lambda f: f) if not callable(a[0] if a else None) else a[0]
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file
import diffusers
from tinydit.model import TinyDiT
from tinydit import text as T, schedule

REPO = "ivanmikhnenkov/tinydit-256"; TOK = os.environ.get("HF_TOKEN")
cfg = json.load(open(hf_hub_download(REPO, "config.json"))); stats = json.load(open(hf_hub_download(REPO, "stats.json")))
model = TinyDiT(latent_ch=cfg["latent_ch"], ctx_dim=cfg["ctx_dim"], dim=cfg["dim"], depth=cfg["depth"], heads=cfg["heads"],
                n_registers=cfg["n_registers"], register_block=cfg["register_block"], n_null=cfg["n_null"]).eval()
sd = load_file(hf_hub_download(REPO, "model.safetensors")); own = model.state_dict()
model.load_state_dict({k: v.to(own[k].dtype) for k, v in sd.items()})
tok, enc = T.load(snapshot_download("google/flan-t5-base", allow_patterns=["*.json", "*.model", "model.safetensors"]), device="cpu", dtype=torch.float32)
vae = diffusers.AutoencoderKLFlux2.from_pretrained(os.path.join(snapshot_download("black-forest-labs/FLUX.2-dev", allow_patterns=["vae/*"], token=TOK), "vae"),
                                                   torch_dtype=torch.float32).eval()
mean = torch.tensor(stats["mean"]).view(1, -1, 1, 1); std = torch.tensor(stats["std"]).view(1, -1, 1, 1)
model.to("cuda"); enc.to("cuda"); vae.to("cuda"); mean, std = mean.cuda(), std.cuda()
SHAPES = {"256 × 256": (256, 256), "288 × 224 (4:3)": (288, 224), "224 × 288 (3:4)": (224, 288), "320 × 208 (3:2)": (320, 208), "208 × 320 (2:3)": (208, 320)}


def to_img(z):
    x = vae.decode((z * std + mean).float()).sample.clamp(-1, 1)
    return ((x[0].permute(1, 2, 0) + 1) * 127.5).round().byte().cpu().numpy()


@GPU(duration=60)
@torch.no_grad()
def generate(prompt, shape, steps, cfgs, seed, show_traj):
    if not prompt.strip(): raise gr.Error("Write a prompt first.")
    W, H = SHAPES[shape]; steps = int(steps); dev = "cuda"
    ctx, msk = T.embed_mixed(tok, enc, [prompt], [len(prompt.split()) > 30], device=dev); ctx = ctx.float()
    nctx0, nmsk0 = T.embed(tok, enc, [""], T.MAX_SHORT, device=dev)
    L = ctx.shape[1]; nctx = torch.zeros(1, L, ctx.shape[2], device=dev); nctx[:, :nctx0.shape[1]] = nctx0.float()
    nmsk = torch.zeros(1, L, dtype=torch.bool, device=dev); nmsk[:, :nmsk0.shape[1]] = nmsk0
    g = torch.Generator(device=dev).manual_seed(int(seed)); x = torch.randn(1, cfg["latent_ch"], H // 8, W // 8, device=dev, generator=g)
    ts = schedule.shift(steps, cfg["sampler"]["shift"], device=dev); traj = []
    keep = set(int(round(i)) for i in torch.linspace(0, steps - 1, 6).tolist()) if show_traj else set()
    for i in range(steps):
        t = ts[i].expand(1); dt = ts[i + 1] - ts[i]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([ctx, nctx]), torch.cat([msk, nmsk])).float()
        vc, vu = v.chunk(2); vg = vu + float(cfgs) * (vc - vu)
        if i in keep: traj.append((to_img(x + (1 - ts[i]) * vg), f"prediction at step {i+1}/{steps}, t={float(ts[i]):.2f}"))
        x = x + vg * dt
    return to_img(x), traj


with gr.Blocks(title="tinydit-256") as demo:
    gr.Markdown("# tinydit-256\nA 210M text-to-image diffusion transformer trained from scratch on one GPU in 3.5 days. "
                "Defaults are the training-time sampler (20 steps, CFG 4). "
                "[Code](https://github.com/ivanmikhnenkov/tinydit) · [Weights](https://huggingface.co/ivanmikhnenkov/tinydit-256) · [ivanmikhnenkov.com](https://ivanmikhnenkov.com)")
    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(label="Prompt", value="a red tractor parked next to a blue rowing boat on a sandy beach", lines=3)
            shape = gr.Dropdown(list(SHAPES), value="256 × 256", label="Size (the five training shapes)")
            steps = gr.Slider(4, 50, value=20, step=1, label="Steps"); cfgs = gr.Slider(1.0, 8.0, value=4.0, step=0.5, label="Guidance (CFG)")
            seed = gr.Number(value=0, precision=0, label="Seed"); show_traj = gr.Checkbox(value=True, label="Show how the image forms (6 intermediate predictions)")
            btn = gr.Button("Generate", variant="primary")
        with gr.Column(scale=1):
            out = gr.Image(label="Result", type="numpy")
            traj = gr.Gallery(label="The model's prediction of the final image along the sampling steps", columns=3, height=260)
    gr.Examples([["three green apples on a white plate next to a black coffee cup", "256 × 256", 20, 4.0, 5, True],
                 ["a lighthouse on a rocky coast at sunset", "320 × 208 (3:2)", 20, 4.0, 1, True],
                 ["a golden retriever wearing sunglasses sitting on a yellow armchair", "224 × 288 (3:4)", 20, 4.0, 2, True],
                 ["a bowl of ramen with a soft-boiled egg and green onions on a dark table", "256 × 256", 20, 4.0, 2, True]],
                inputs=[prompt, shape, steps, cfgs, seed, show_traj])
    btn.click(generate, [prompt, shape, steps, cfgs, seed, show_traj], [out, traj])
    prompt.submit(generate, [prompt, shape, steps, cfgs, seed, show_traj], [out, traj])
demo.queue(max_size=16).launch()

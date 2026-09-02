# tinydit

A small text-to-image diffusion transformer trained from scratch on one GPU, built to understand how
modern image generation works. Frozen pretrained autoencoder + frozen pretrained text encoder +
a **~210M-parameter DiT trained from scratch** on rectified flow, at 256²-area with aspect-ratio
buckets. After pretraining it becomes the base model for RL practice (Flow-GRPO and friends).

Everything heavy (weights, latent caches, runs, shards) lives under `out/` and is git-ignored; the
repo holds code, prompts, notes and the Docker recipe, so the whole thing is reproducible from a clone.

---

## Decisions (September 2026 re-plan)

Every choice below was checked against current papers and shipping models; the evidence with URLs is
in `notes/2026-09-02_architecture_research.md`, the dataset survey in `notes/2026-09-02_dataset_research.md`,
throughput measurements in `notes/2026-09-02_benchmarks.md`. Interfaces between ingest, training and the
dashboard are pinned in `notes/CONTRACTS.md`.

### The three pieces

| Piece | What | Trained? | Params |
|---|---|---|---|
| Autoencoder | FLUX.2 AE (`AutoencoderKLFlux2`), RGB ↔ 32-channel latent, ÷8 per side | frozen | 84M |
| Text encoder | `flan-t5-base` encoder → per-token states (768-d), run live every batch | frozen | 110M |
| **DiT** | `run1` config: dim 896, 16 blocks, 14 heads of 64 | **trained** | **209M** |

**Why FLUX.2's AE.** Its 32-channel latent was trained with semantic regularisation and, at DiT-XL on
ImageNet-256, beats FLUX.1's 16 channels (gFID 3.70 vs 10.13, BFL tech blog). Latents are whitened per
channel with statistics from our own cache: the official FLUX.2 code applies BatchNorm statistics after
patchify and diffusers' encoder does not, so this is what the reference pipeline does, not a deviation.

**Why flan-t5-base.** Text-encoder size buys little at this scale (SANA: T5-Large ≈ T5-XXL on FID;
FLUX.1 with T5-Base is within 2 FID of T5-XXL, losses concentrated in text rendering). Caption quality
matters far more, which is why the data work below is about captions. Encoding live costs 4–9% of a step
after the long/short sub-batching in `text.py`; caching embeddings would cost 400 GB of disk.

### DiT block

```
t ──► sinusoidal ──► MLP ──► shared Linear(dim, 6·dim)  + per-block table  ──► (shift, scale, gate) × 2   [adaLN-single]

x = x + gate₁ · SelfAttn( RMSNorm(x)·(1+scale₁) + shift₁ )     2D RoPE + QK-norm, 16 register tokens from block 3
x = x +         CrossAttn( RMSNorm(x), K/V = T5 tokens ++ 2 learned null slots )   zero-init out-proj
x = x + gate₂ · SwiGLU(   RMSNorm(x)·(1+scale₂) + shift₂ )
```

- **adaLN-single** (PixArt-α): one shared modulation MLP plus a 6·dim table per block, instead of a
  6·dim×dim linear per block. Per-block adaLN was 27% of the old model's parameters for zero compute;
  the saving is spent on depth (12 → 16 blocks at the same parameter count).
- **Registers.** 16 learnable tokens appended to the image sequence from block 3 and dropped before
  unpatchify, with identity RoPE. DiTs have no high-norm outlier tokens yet still gain from registers
  (VAE-latent DiT-B FID 10.40 → 9.40; a T2I model +4 GenEval). Plus 2 learned null key/value slots per
  cross-attention block, so a query that wants "nothing in particular" no longer has to lean on T5's EOS.
- **No pooled text into adaLN.** With cross-attention present it slightly hurts alignment (Deep Fusion).
- **Patchify** `Conv2d(32, dim, k=2, s=2)`; **depatchify** adaLN-modulated linear + pixel shuffle, zero-init.
- **RMSNorm with weights cast to the activation dtype**: the stock module falls back to an unfused
  kernel under bf16 autocast (PyTorch warns about it).

### Objective

```
x_t    = (1-t)·e + t·z          e ~ N(0,I) at t=0, z the whitened latent at t=1
loss   = mse(v, z-e) + 1.0·(1 - cos(v, z-e)) + 0.5·dispersive(block-5 features)
t      ~ sigmoid(N(0,1) - ln 2.8)        logit-normal with the SD3 shift toward high noise
```

- **Timestep shift** α = 2.8 (the SD3/RAE rule √(32·32·32/4096)) for training and sampling: a 32-channel
  latent needs more of the budget at high noise than SD's 4 channels.
- **Cosine velocity term** (LightningDiT): MSE is dominated by magnitude at high noise; this keeps the
  direction honest. ~10% FID in their ablation chain.
- **Dispersive loss** (Wang & He 2025): repel different samples' block-5 features; no external encoder,
  no extra data. REPA would add another ~20% but needs the clean image or DINOv2 features for every
  training sample (~150 KB each, 700 GB here), so it is deferred.
- **10% caption dropout** to the empty string for classifier-free guidance; sampling uses 20 Euler steps
  with the same shift schedule and CFG 4.

### Training recipe

| | value | why |
|---|---|---|
| optimizer | fused AdamW, lr 2e-4, betas (0.9, 0.95), wd 0, warmup 2k, clip 1.0 | LightningDiT / Lumina-2 |
| precision | bf16 autocast, **fp32 master weights and optimizer state** | bf16 weights alone lose the tiny updates (PRX: FID 18.2 → 21.9) |
| EMA | 0.9999 with (1+t)/(10+t) warm-up, multi-tensor update | 0.999 has a 1k-step horizon, far too short for a 100k+ run |
| batch | 256, one aspect bucket per batch | ~256 tokens per sample in every bucket |
| compile | `torch.compile`, one static graph per bucket shape | 2.4× over eager, half the memory (measured) |
| checkpoints | `ckpt_last.pt` fp32 (weights+opt+EMA) rotated every 2.5k steps; `ema_<step>.safetensors` bf16 every 10k | exact resume vs. small shareable weights |

Measured on the RTX PRO 6000 Blackwell (300 W): `run1` at batch 192 compiled runs 461 img/s at 70% MFU
(`notes/2026-09-02_benchmarks.md`), so 110M samples (~27 epochs of 4.1M images) take about 2.8 days.

### Data

Real photos with accurate captions, aspect ratios preserved, one synthetic mix-in for prompt adherence.
Twenty real rows from every candidate were inspected before choosing (`scripts/hfpeek.py`,
`scripts/preview_gallery.py`; the survey notes list what was rejected and why).

| source | images | share of batches | captions long / short |
|---|---|---|---|
| Pexels: `bghira/photo-concept-bucket` (CDN at 640 px) + the gated `animetimm/pexels-tagger-v0-w640-ws-full` tars, deduplicated by Pexels id | 2.8M | 60% | Qwen3-VL-30B (i1-captions, by Pexels id) / CogVLM or first sentence |
| FLUX-Reason-6M, Aesthetics parts, filtered by clarity+structure score | 1.2M | 25% | caption_detail / caption_entity |
| COCO 2017 train + GPT-4V captions (`laion/220k-GPT4Vision-captions-from-LIVIS`) | 118k | 15% | GPT-4V / 5 human captions |

- **Buckets.** Five shapes of ~256 tokens: 256×256, 288×224, 224×288, 320×208, 208×320. Nearest bucket
  for real photos (mean crop 3–5%); 4:3 and 3:4 frames go square half the time, and FLUX squares are spread
  over all five shapes (45% stay square), so no bucket is dominated by one source. Anything wider than 2:1
  is dropped (≈1%). Position comes from RoPE, so shapes cost nothing architecturally.
- **Captions.** Per sample: 50% a long caption cut at 128 T5 tokens, 40% a short one, 10% empty.
  Long/short are encoded as separate sub-batches padded to their own longest (`text.embed_mixed`).
- **Storage.** Images are never kept: each shard is streamed, bucketed, encoded to a 64 KB latent and
  deleted (`ingest.py`). 4.1M images ≈ 265 GB of fp16 latents. `scripts/launch_run1.sh` waits for the
  four ingest processes, merges, checks and starts the run.
- **Held out.** The first 1,000 ingested rows of each source are `val` and never trained on; they provide
  the validation loss, the unseen-caption grids and the FID references. `prompts/novel.txt` holds
  hand-written compositions absent from the data.

### Evaluation (automatic, on the dashboard)

| metric | what it measures | cadence |
|---|---|---|
| loss, mse / cos / disp, grad norm, lr, img/s | optimisation | every 20 steps |
| held-out loss, per source | fit vs. memorisation, per data source | every 500 steps |
| sample filmstrip (fixed prompts, fixed noise) | qualitative progress | every 500 steps |
| CLIP score, **PickScore**, **HPSv2.1** on held-out and novel prompts | adherence and learned human preference; PickScore/HPS are the reward-model family the RL stage will use | every 5k steps |
| **FID** and **FD-DINOv2** vs 3,000 held-out images, **object accuracy** (80 COCO classes × 4 seeds, Faster R-CNN) | realism, and "does it draw the object" | every 10k steps |

Each dashboard chart has an ⓘ in its top-right corner explaining the metric, its direction and range.
SSIM is deliberately absent: it needs a pixel-aligned target, which a text-to-image sample does not have.

---

## Running it

Everything runs in the Docker image (CUDA 12.8 for Blackwell, torch 2.11, a C compiler for
`torch.compile`). The volume is bind-mounted at the same path as on the host and commands run as the
host user, so paths, logs and file ownership are identical inside and outside the container.

```bash
docker/run.sh build                     # once; docker/Dockerfile + requirements.txt
docker/run.sh up                        # long-lived container `tinydit`
docker/run.sh exec 'python -m tinydit.ae'                     # fetch the frozen AEs (needs the HF token file)
docker/run.sh exec 'python -c "from tinydit import text; text.fetch(\"out/models/t5\")"'
docker/run.sh exec 'python scripts/fetch_metric_models.py'    # CLIP, PickScore, HPSv2.1, DINOv2, Inception, Faster R-CNN

# data (see `python -m tinydit.ingest --help`; each source is resumable, run them in parallel)
docker/run.sh exec 'python -m tinydit.ingest coco   --cache out/cache/run1'
docker/run.sh exec 'python -m tinydit.ingest pexels --cache out/cache/run1 --workers 64'
docker/run.sh exec 'python -m tinydit.ingest pexels2 --cache out/cache/run1'   # gated 2.8M tars, accept terms on HF first
docker/run.sh exec 'python -m tinydit.ingest flux   --cache out/cache/run1 --target 1200000'
docker/run.sh exec 'python -m tinydit.ingest merge  --cache out/cache/run1 --remove-src'
docker/run.sh exec 'python -m tinydit.ingest check  --cache out/cache/run1'   # counts, crops, decoded check strip

# train + monitor
docker/run.sh exec 'python -m tinydit.train --run run1 --cache out/cache/run1 --config run1'
cd /home/ivan/volume/learning && python3 -m http.server 7180 --bind 127.0.0.1   # then http://localhost:7180/tinydit/dashboard.html?run=run1

# sample from a checkpoint
docker/run.sh exec 'python -m tinydit.sample --ckpt out/runs/run1/ema_0100000.safetensors --sets NOVEL=prompts/novel.txt --out out/novel.png --shape 320x208'
```

The HF token is read from `/home/ivan/volume/.tinydit_token` (`HF_TOKEN=...`, mode 0600, outside the
repo and outside the HTTP-served tree); it is needed for the gated FLUX.2 AE and the gated Pexels set.
Long-running jobs are started with `nohup docker/run.sh exec '...' > out/logs/<name>.log &` and are
resumable (`ingest` per source via `progress.json`, `train` via `--resume out/runs/<run>/ckpt_last.pt`).

## Layout

```
src/tinydit/
  ae.py         frozen autoencoders: fetch, load, encode, decode
  text.py       frozen flan-t5-base; long/short sub-batch encoding
  model.py      the DiT (adaLN-single, registers, null K/V, 2D RoPE, QK-norm, SwiGLU)
  flow.py       rectified flow + cosine + dispersive losses, shifted timestep sampling
  schedule.py   step schedules (shift, karras, ...) and ODE solvers
  sample.py     Euler+CFG sampler for any grid shape, labelled grids, EMA/bf16 checkpoint loading
  metrics.py    CLIP, PickScore, HPSv2.1, FID, FD-DINOv2, object accuracy
  ingest.py     streaming download → bucket → encode → memmap cache
  train.py      bucketed loop, live T5, compile, EMA, checkpoints, periodic evaluation
  rope.py       2D rotary embeddings
  data.py       legacy single-resolution caching (CUB / COCO), kept for reference
scripts/
  hfpeek.py             sample any HF dataset shard through HTTP range requests
  preview_gallery.py    dataset contact sheets
  bench_step.py         training-step throughput benchmark
  bench_text.py         T5 live-encoding cost
  fetch_metric_models.py
  tfrecord_peek.py      minimal TFRecord/Example reader (no TensorFlow)
docker/                 Dockerfile, run.sh
prompts/novel.txt       hand-written evaluation prompts
notes/                  research notes with sources, benchmarks, dataset preview spec, contracts
dashboard.html          live monitor (polls out/runs/<run>/metrics.jsonl)
out/                    models, caches, runs — git-ignored
```

## Machine notes

RTX PRO 6000 Blackwell Max-Q (96 GB, 300 W), 48 cores, 251 GB RAM, 1.7 TB NVMe RAID-1. IPv6 is
configured but unrouted on this host: Python networking hangs unless forced to IPv4 (see
`scripts/hfpeek.py`); inside Docker (IPv4 bridge) this is not an issue. The Pexels 2.8M set requires
accepting its terms once on Hugging Face (auto-approved).

## Status

| stage | state |
|---|---|
| Dataset survey (18 candidates sampled) | done — `notes/2026-09-02_dataset_research.md` |
| Architecture and recipe decisions | done — this README |
| Throughput benchmarks | done — `notes/2026-09-02_benchmarks.md` |
| Ingest run 1 | in progress |
| Pretraining run 1 | pending ingest |
| Flow-GRPO practice | after pretraining |

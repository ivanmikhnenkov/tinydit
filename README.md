# tinydit

A 210M-parameter text-to-image diffusion transformer trained from scratch on one GPU at 256², with a
frozen FLUX.2 autoencoder and a frozen flan-t5-base text encoder. Built to learn how modern image
generation works, and as the base model for RL practice next. Everything heavy lives under `out/` and is
git-ignored; a clone plus the commands below reproduces the run. Sources for each decision are in `notes/`.

## Results

400k steps, 24 epochs over 4.2M images, 3.5 days on one GPU. Final EMA weights, fixed seeds, no
cherry-picking within a set. Grids sampled with 50 steps, CFG 4, shift 2.8; metrics measured with the
20-step training sampler.

**Objects and animals.** Short COCO-style prompts.

![objects](docs/grid_objects.jpg)

**Scenes.**

![scenes](docs/grid_scenes.jpg)

**Composition: colours, counts, relations.** Counting slips: "two apples and one pear" became two and two.

![composition](docs/grid_composition.jpg)

**Lighting and style words.**

![style](docs/grid_style.jpg)

**Long captions** in the style of the training data (60–90 words, up to 128 T5 tokens).

![long captions](docs/grid_long.jpg)

**Aspect ratios.** One prompt and seed in the five training shapes, plus an untrained 448×256 frame.

![aspect ratios](docs/grid_aspect.jpg)

**Where it fails:** readable text, close faces, crowds, counts above three, clock hands, stacked geometry.

![failure cases](docs/grid_failures.jpg)

### Dynamics

Held-out metrics every 10k steps; the vertical line marks the start of the learning-rate decay at 300k.

![metrics](docs/metrics.png)

| step | FID | FD-DINOv2 | object acc. | CLIP held-out / novel | PickScore held-out / novel | HPSv2.1 held-out / novel |
|---|---|---|---|---|---|---|
| 10k | 33.7 | 570 | 65% | 0.300 / 0.333 | 19.5 / 20.2 | 0.199 / 0.213 |
| 50k | 29.1 | 324 | 83% | 0.322 / 0.349 | 20.3 / 21.6 | 0.231 / 0.251 |
| 100k | 28.1 | 274 | 88% | 0.323 / 0.359 | 20.6 / 22.0 | 0.238 / 0.264 |
| 200k | 27.5 | 244 | 90% | 0.319 / 0.359 | 20.8 / 22.3 | 0.248 / 0.272 |
| 300k | 27.1 | 229 | 92% | 0.322 / 0.361 | 20.8 / 22.5 | 0.251 / 0.276 |
| 400k | 27.0 | 218 | 90% | 0.322 / 0.361 | 20.9 / 22.6 | 0.254 / 0.277 |

- Layout and objects are learned in the first 10k steps; realism and preference keep improving to the end,
  FID and object accuracy saturate around 300k. The learning-rate decay over the last 100k steps added 5%
  on FD-DINOv2 and the run's best preference scores.
- The flow-matching loss barely moves (0.805 → 0.754) because most of it is irreducible target variance;
  training and held-out loss stayed equal, so nothing was memorised.
- Short hand-written prompts score higher than long held-out captions on every preference metric.

**The same prompt over training**, EMA snapshots, same seed:

![progress](docs/progress.jpg)

**How one sample happens.** The shifted schedule spends 15 of 20 steps below t = 0.5; the prediction
(bottom row) has the layout after 3 steps and only sharpens afterwards.

![trajectory](docs/trajectory.jpg)

**Inference settings**, same prompt and seed; measured on the 2,456 held-out prompts:

| sampler | FID | FD-DINOv2 | PickScore | HPSv2.1 | time per image |
|---|---|---|---|---|---|
| 8 steps, shift 2.8 | 28.4 | 234 | 20.81 | 0.243 | 0.4× |
| 20 steps, shift 2.8 (training setting, used for all metrics) | 27.0 | 218 | 20.84 | 0.247 | 1× |
| 50 steps, shift 2.8 (used for the figures) | 26.6 | 216 | 20.82 | 0.247 | 2.5× |
| 20 steps, no shift | 27.3 | 228 | 20.76 | 0.243 | 1× |

The training-time shift is worth more than doubling the step count.

![sampling settings](docs/grid_sampling.jpg)

Final validation loss 0.754 (train/held-out gap zero throughout). Weights: `out/runs/run1/ema_0400000.safetensors`
(bf16 EMA, 418 MB) plus 39 earlier EMA snapshots every 10k steps and the fp32 resume checkpoint. The
playground (`tinydit.playground`) serves these weights for interactive prompts, trajectories and attention maps.

## Model

| piece | what | trained |
|---|---|---|
| autoencoder | FLUX.2 AE, RGB ↔ 32-channel latent at ÷8; latents whitened per channel with cache statistics | no |
| text encoder | flan-t5-base encoder, run live per batch; long captions ≤128 tokens, short ≤48, encoded as separate sub-batches | no |
| DiT (`run1`) | dim 896, 16 blocks, 14 heads, 209M params | yes |

```
t ─► sinusoidal ─► MLP ─► shared Linear(dim, 6·dim) + per-block table ─► (shift, scale, gate) × 2      adaLN-single
x = x + gate₁ · SelfAttn( RMSNorm(x)·(1+scale₁) + shift₁ )     2D RoPE, QK-norm, 16 register tokens from block 3
x = x +         CrossAttn( RMSNorm(x), K/V = T5 tokens ++ 2 learned null slots )       zero-init out-proj
x = x + gate₂ · SwiGLU(   RMSNorm(x)·(1+scale₂) + shift₂ )
```

- **adaLN-single** (PixArt-α): per-block modulation matrices were 27% of the parameters for no compute;
  the saving went into depth.
- **Registers**: 16 learnable image-stream tokens (dropped before unpatchify) plus 2 learned null
  key/value slots per cross-attention, so global state and "attend to nothing" have a home other than
  a random patch or the T5 EOS state.
- **No pooled text into adaLN**: with cross-attention present it slightly hurts alignment.
- Position comes from 2D RoPE, so any latent grid works: the five training buckets, and later other
  shapes or resolutions, need no architectural change.

## Objective and recipe

```
x_t  = (1-t)·e + t·z,   t ~ sigmoid(N(0,1) − ln 2.8)            rectified flow, SD3 shift toward high noise
loss = mse(v, z−e) + 1.0·(1 − cos(v, z−e)) + 0.5·dispersive(block-5 features)
```

| | |
|---|---|
| optimizer | fused AdamW, lr 2e-4, betas (0.9, 0.95), wd 0, warmup 2k, grad clip 1.0; constant, then linear decay to zero over the last 25% of steps (`--decay-start`) |
| precision | bf16 autocast with fp32 master weights and optimizer state |
| EMA | 0.9999, warmed up; bf16 snapshots every 10k steps (`ema_<step>.safetensors`, 420 MB) |
| batch | 256, one aspect bucket per batch |
| compile | `torch.compile`, one static graph per bucket shape: 2.4× over eager, half the memory |
| sampling | 20 Euler steps on the same shifted schedule, CFG 4, 10% caption dropout in training |

Measured on an RTX PRO 6000 Blackwell (300 W): the real run does 0.76 s/step at batch 256 (338 img/s,
including live T5 encoding and the extra losses), so 400k steps take ~3.5 days; a concurrent ingest
sharing the GPU halves that rate (`notes/2026-09-02_benchmarks.md`).

## Data

Real photos with accurate captions, aspect ratios preserved, one synthetic mix-in for prompt adherence.
Twenty real rows of every candidate were inspected before choosing (`scripts/hfpeek.py`,
`scripts/preview_gallery.py`; rejected sets and why in `notes/2026-09-02_dataset_research.md`).

| source | images | share of batches | long / short captions |
|---|---|---|---|
| Pexels: `bghira/photo-concept-bucket` (CDN, 640 px) + gated `animetimm/pexels-tagger-v0-w640-ws-full`, deduplicated by id | 2.8M | 60% | Qwen3-VL-30B (`zlab-princeton/i1-captions`) / CogVLM or first sentence |
| FLUX-Reason-6M, Aesthetics parts, clarity ≥ 9 and structure ≥ 9 | 1.2M | 25% | caption_detail / caption_entity |
| COCO 2017 train + GPT-4V captions (`laion/220k-GPT4Vision-captions-from-LIVIS`) | 118k | 15% | GPT-4V / the 5 human captions |

- **Buckets**: 256×256, 288×224, 224×288, 320×208, 208×320 (~256 tokens each). Real photos take the
  nearest shape (mean crop 3–5%); 4:3 and 3:4 frames go square half the time and FLUX squares are spread
  over all shapes, so no bucket is dominated by one source. Wider than 2:1 is dropped (~1%).
- **Captions per sample**: 50% long, 40% short, 10% empty.
- **Storage**: images are streamed, bucketed, encoded to a 64 KB latent and deleted; 4.1M images ≈ 265 GB.
- **Held out**: the first 1,000 rows of each source are never trained on; they give the validation loss,
  the unseen-caption grids and the FID references. `prompts/novel.txt` holds hand-written compositions.

## Evaluation (dashboard)

| what | cadence |
|---|---|
| loss (mse / cos / disp), grad norm, lr, img/s | 20 steps |
| held-out loss, per source; fixed-prompt filmstrip | 500 steps |
| grids of held-out captions in all five shapes + novel prompts; CLIP, PickScore, HPSv2.1 | 5k steps |
| FID and FD-DINOv2 vs 3,000 held-out images; object accuracy (80 COCO classes × 4 seeds, Faster R-CNN) | 10k steps |

PickScore and HPSv2.1 are the reward-model family the RL stage will use, so their pretraining curves
are the baselines for it. Every chart has an ⓘ explaining the metric, its direction and range.

## Run

Everything runs in the Docker image (CUDA 12.8 for Blackwell, torch 2.11, a C compiler for compile).
The volume is mounted at the same path as on the host and commands run as the host user.

```bash
docker/run.sh build && docker/run.sh up
docker/run.sh exec 'python -m tinydit.ae'                                              # frozen AEs
docker/run.sh exec 'python -c "from tinydit import text; text.fetch(\"out/models/t5\")"'
docker/run.sh exec 'python scripts/fetch_metric_models.py'                             # CLIP, PickScore, HPSv2.1, DINOv2, Inception, Faster R-CNN

docker/run.sh exec 'python -m tinydit.ingest coco    --cache out/cache/run1'           # each source resumable, run in parallel
docker/run.sh exec 'python -m tinydit.ingest pexels  --cache out/cache/run1 --workers 64'
docker/run.sh exec 'python -m tinydit.ingest pexels2 --cache out/cache/run1'           # gated set: accept its terms on HF first
docker/run.sh exec 'python -m tinydit.ingest flux    --cache out/cache/run1 --target 1200000'
docker/run.sh exec 'python -m tinydit.ingest merge   --cache out/cache/run1 --remove-src'
docker/run.sh exec 'python -m tinydit.ingest check   --cache out/cache/run1'

docker/run.sh exec 'python -m tinydit.train --run run1 --cache out/cache/run1 --config run1'
cd .. && python3 -m http.server 7180 --bind 127.0.0.1        # http://localhost:7180/tinydit/dashboard.html?run=run1

docker/run.sh exec 'python -m tinydit.sample --ckpt out/runs/run1/ema_0100000.safetensors --sets NOVEL=prompts/novel.txt --out out/novel.png --shape 320x208'
```

`python -m tinydit.playground --run run1 --port 7181` (host venv, next to the trainer) serves
`playground.html` at http://localhost:7181/: generate from any prompt with the newest EMA snapshot, watch
the intermediate states and predictions along the sampling schedule, and inspect cross-attention per word
(including the learned null slots) and the register tokens, per block and per step.

`scripts/launch_run1.sh` waits for the ingests, merges, checks and starts the run; `scripts/append_pexels2.sh`
adds a source that finished later (`merge --append`, whitening statistics untouched) and resumes from
`ckpt_last.pt`. Long jobs: `nohup docker/run.sh exec '...' > out/logs/<name>.log &`.
The HF token is read from `/home/ivan/volume/.tinydit_token` (`HF_TOKEN=...`, mode 0600), outside the repo.

## Layout

```
src/tinydit/   ae.py text.py model.py flow.py schedule.py data.py train.py sample.py metrics.py ingest.py rope.py playground.py
scripts/       hfpeek.py preview_gallery.py tfrecord_peek.py  (dataset survey)   bench_step.py   fetch_metric_models.py
               launch_run1.sh append_pexels2.sh
docker/        Dockerfile run.sh          prompts/novel.txt          dashboard.html          notes/          out/ (ignored)
```

Host quirk: IPv6 is configured but unrouted, so Python networking on the host must force IPv4
(`scripts/hfpeek.py` does); inside the container it is not an issue.

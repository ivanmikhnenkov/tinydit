# tinydit

A small text-to-image diffusion transformer, trained from scratch on one GPU, built to
understand how modern image generation actually works rather than to compete with anything.

Frozen pretrained autoencoder + frozen pretrained text encoder + a **custom DiT trained
from scratch** on rectified flow.

---

## Architecture

### The three pieces

| Piece | What | Trained? | Params |
|---|---|---|---|
| Autoencoder | FLUX.2 AE (`AutoencoderKLFlux2`) — RGB ↔ 32-channel latent, ÷8 per side | frozen | 84.0M |
| Text encoder | `flan-t5-base` encoder → `(B, L, 768)` token states | frozen | 109.6M |
| **DiT** | our model: predicts the flow velocity in latent space | **trained** | **~158M** |

**Why T5 and not CLIP.** CLIP's text tower is trained contrastively against whole images,
which makes its per-token states weak at compositional detail ("a *black* dog chasing a
*white* dog"). PixArt and SANA both use T5 encoders for cross-attention for exactly this
reason. `d_model=768` also matches our DiT width, so K/V projections stay square.

**Why FLUX.2's AE.** Measured at 256²: `3×256×256 → 32×32×32`, i.e. 32,768 latent values,
6× compression. It reconstructs at ~36 dB PSNR (see `../flux2_ae_casestudy/`). The open
question is whether 32 channels is *harder to generate in* than SD 1.5's 4 — which is why
the SD 1.5 AE is cached alongside for a controlled comparison.

### DiT block

`base` config: **dim 768, depth 12, heads 12** (head_dim 64), ~158M params.

```
t ──► sinusoidal ──► MLP ──► per-block (shift, scale, gate) × 2

x = x + gate₁ · SelfAttn( RMSNorm(x)·(1+scale₁) + shift₁ )    2D RoPE + QK-norm
x = x +         CrossAttn( RMSNorm(x), K/V = T5 tokens )      zero-init out-proj
x = x + gate₂ · SwiGLU(   RMSNorm(x)·(1+scale₂) + shift₂ )
```

- **Patchify** `Conv2d(latent_ch, dim, k=2, s=2)` → 256 tokens at 256², 64 at 128².
- **Depatchify** adaLN-modulated `Linear` + `pixel_shuffle`, zero-initialised.
- **Cross-attention in every block** (PixArt-Σ / SANA style), text padded to 32 tokens with
  padding masked out of attention.

Every choice above was verified against shipping SOTA rather than recalled. From
`diffusers/models/transformers/transformer_flux2.py`:

| Choice | Evidence it is current |
|---|---|
| adaLN-Zero modulation | FLUX.2: `(shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp)`; `x = x + gate_mlp * ff_output` |
| SwiGLU FFN | FLUX.2 ships a `Flux2SwiGLU` class — *not* an LLM-only idea |
| QK-norm | FLUX.2 `norm_q`/`norm_k` are `torch.nn.RMSNorm`; Z-Image config sets `qk_norm: true` |
| RMSNorm | used throughout FLUX.2 |
| 2D RoPE | FLUX.2 `axes_dims_rope: [32,32,32,32]`; Z-Image `axes_dims: [32,48,48]` |

Two deliberate divergences: SOTA uses `head_dim=128` (we use 64, better at our smaller
width), and FLUX.2/Z-Image use **MMDiT** joint attention rather than cross-attention. MMDiT
lets the text representation evolve through the network; with a *frozen* encoder and
32-token captions that buys much less, and cross-attention is what the efficiency-focused
models (SANA, PixArt-Σ) use.

### Objective — rectified flow

With `e ~ N(0,I)` at t=0 and `z` the image latent at t=1:

```
x_t    = (1-t)·e + t·z
target = z - e                     # the velocity; constant in t
loss   = mse( v_θ(x_t, t, text), z - e )
```

Sampling integrates `dx/dt = v_θ` from t=0 → t=1 with Euler steps.

- **t is sampled logit-normal**, not uniform — SD3 showed uniform is measurably worse.
- **10% caption dropout** (swap in the embedding of `""`) to enable classifier-free guidance.
- Latents are **whitened per channel** using statistics computed from the cache itself.
  We deliberately do *not* reuse FLUX.2's `vae.bn`, whose statistics were fitted for its own
  packed 128-channel latents, not for a DiT trained from scratch.

---

## Training plan

### Phase A — CUB-200 sanity check (~1–2h)
11,788 bird photos. Captions are the template `"A photo of a {species}"` — only **183
unique captions per 500 sampled**, vocabulary 223. This is a 200-way class problem wearing
a sentence costume, so it does *not* exercise cross-attention. It is here to prove the
pipeline end-to-end and put images on screen quickly.

Gate before moving on: overfit-on-8-images must reach near-zero loss and reproduce those
8 images. If it does not, the objective is wired wrong.

### Phase B — COCO 2017, the real run (~8–20h)
123,353 images, 5 genuine captions each, vocabulary an order of magnitude larger. The only
one of the candidates big enough that a 158M model will not simply memorise it inside the
budget. **Random 1-of-5 caption per sample per epoch** — text-side augmentation that teaches
the model which content is invariant and which phrasing is incidental.

### Resolution curriculum
Both phases run 128² first, then 256². Tokens = `(res/16)²`:

| res | latent | tokens | cost vs 256² | s/step (base, bs 256) |
|---|---|---|---|---|
| 128² | 16×16 | 64 | 0.25× | ~0.045 |
| 256² | 32×32 | 256 | 1× | ~0.178 |
| 384² | 48×48 | 576 | ~2.4× | ~0.42 |
| 512² | 64×64 | 1024 | ~4.8× | ~0.85 |

Scaling is *mostly linear* in token count — at dim 768 attention is only ~6% of cost at 256
tokens, rising to ~24% at 1024. The cheap 128² stage learns layout, colour and composition;
the 256² stage learns texture.

**256² is the ceiling because the data says so**, not for want of GPU. Measured short sides:
CUB 325/357/433, Flickr30k 332/**360**/375, COCO 360/**427**/481 (p10/p50/p90). Over 90% of
Flickr30k cannot even supply a 384² crop. Training above 256 would mean learning from
upscaled blur. Higher output resolution is a job for a separate upscaler afterwards.

### Phase C — the autoencoder comparison
Same DiT config, same steps, same seeds — once on FLUX.2 latents (32ch), once on SD 1.5
latents (4ch). FLUX.2 reconstructs far better, but the DiT must model 8× more values, so
SD 1.5 may well produce *cleaner samples* despite the worse reconstruction ceiling. This is
the direct sequel to the compression study next door.

### Phase D — aspect-ratio bucketing
Square-only until the pipeline is proven, then buckets. This costs a dataloader change and
no retraining, because **2D RoPE derives position from `(row, col)`** and works on any grid;
learned absolute embeddings would have locked us to one shape permanently.

Buckets sized to hold ~256 tokens each, chosen to match the measured aspect distribution
(p50 ≈ 1.4; common native sizes 500×375, 500×333, 640×480):

| bucket | pixels | latent | tokens |
|---|---|---|---|
| 1:1 | 256×256 | 32×32 | 256 |
| 4:3 | 288×224 | 36×28 | 252 |
| 3:4 | 224×288 | 28×36 | 252 |
| 3:2 | 320×208 | 40×26 | 260 |
| 2:3 | 208×320 | 26×40 | 260 |

**16:9 and 9:16 are deliberately absent** — at aspect 1.78 they would be near-empty for
these corpora. RoPE still lets you *sample* at aspect ratios never trained on, with some
quality loss, so an empty bucket is not a hard limit on output shape.

Bucketing also recovers real data: centre-cropping a 640×480 to square **discards ~30% of
every frame**, frequently including objects the caption names.

---

## Monitoring

`dashboard.html` polls the run directory every 5s and redraws — no build step, no
external libraries. Serve the parent directory and open it:

```bash
cd /home/ivan/volume/learning && python3 -m http.server 7180 --bind 127.0.0.1
# then http://localhost:7180/tinydit/dashboard.html
```

Six charts, each with an ⓘ explaining what it measures:

| chart | source | cadence |
|---|---|---|
| Loss | `loss` | every `--log-every` (100) steps |
| Held-out loss | `val_loss` on `--val-n` (2048) images never trained on | every `--val-every` (500) |
| Loss by timestep | `loss_t_lo` / `loss_t_hi`, split at t=0.5 | every log interval |
| Gradient norm | `gnorm`, global L2, clipped at 1.0 | every log interval |
| CLIP score | `clip_seen` / `clip_unseen` / `clip_novel` | every `--eval-every` (5000) |
| Throughput | `ips`, cumulative since process start | every log interval |

Lines are coloured by resolution (blue 128², magenta 256²) with a dashed marker at the
switch. **Loss is not comparable across colours** — each resolution has its own latent
whitening, so the target has a different scale.

Below the charts: an **evaluation** filmstrip (seen / unseen / novel grids) and a
**samples** filmstrip, both with a scrub slider and ← → keys. "Follow latest" auto-advances
to new snapshots only if you were already on the newest; dragging a slider switches it off
so polling never interrupts you.

### Evaluating generalisation

The samples grid uses *training* captions, so it tracks optimisation only. The evaluation
grid is the one that shows generalisation, over three sets of 128 prompts:

- **SEEN** — training captions
- **UNSEEN** — COCO val2017 captions (val images were never cached or trained on)
- **NOVEL** — systematic compositions absent from COCO, built from an axis grid
  (`{animal} in {place}`, `{colour} {object} on {surface}`)

CLIP scores all 128; only `--eval-render` (12) are drawn. At n=128 the 95% CI on the mean
is ±0.007, so differences below that are noise — the SEEN↔UNSEEN gap has stayed inside it,
which is the evidence for "generalising, not memorising".

Checkpoints (model + EMA + optimizer + step) every `--ckpt-every` steps, plus one on
SIGTERM — so the GPU can be freed and the run resumed exactly with `--resume`.

## Layout

```
src/tinydit/
  ae.py         frozen autoencoders (FLUX.2, SD 1.5): fetch, load, encode, decode
  text.py       frozen flan-t5-base encoder
  clipscore.py  CLIP prompt-adherence scoring
  data.py       download + cache images / latents / text embeddings
  rope.py       2D rotary embeddings
  model.py      the DiT
  flow.py       rectified-flow training objective
  sample.py     Euler sampler with CFG, labelled comparison grids  (the only sampler)
  train.py      loop, EMA, checkpoints, live metrics, periodic evaluation
scripts/
  preview_datasets.py   dataset survey -> preview_data.json
  test_model.py         shape / zero-init / non-square checks
  lr_range_test.py      LR range test (Smith 2015)
  eval_watch.py         out-of-process evaluation for runs predating --eval-every
  curriculum_chain.sh   128² -> 256² stage chaining
out/            models, caches, runs (gitignored)
```

Compute runs in the `ivan_dev` container (torch 2.8 + cu128); files live on the host bind
mount, so both sides see the same paths. The HF token is read from
`/root/volume/.tinydit_token` (mode 0600, deliberately outside the HTTP-served tree) and is
never passed on a command line.

```bash
D="docker exec -w /root/volume/learning/tinydit ivan_dev bash -lc"
$D 'PYTHONPATH=src /opt/venv/bin/python -m tinydit.data prepare --dataset coco'
$D 'PYTHONPATH=src /opt/venv/bin/python -m tinydit.data latents --dataset coco --ae flux2 --res 256'
$D 'PYTHONPATH=src /opt/venv/bin/python -m tinydit.data text    --dataset coco'
$D 'PYTHONPATH=src /opt/venv/bin/python -m tinydit.train --run coco_flux2 --dataset coco --ae flux2 --res 256'
$D 'PYTHONPATH=src /opt/venv/bin/python -m tinydit.sample --ckpt out/runs/coco_flux2/ckpt_last.pt \
     --sets SEEN=out/prompts_seen.txt UNSEEN=out/prompts_heldout.txt --out out/eval.png'
```

## Status

| stage | state | result |
|---|---|---|
| Dataset survey | done | `dataset_preview.html` — CUB has 183 unique captions per 500; COCO/Flickr ~all unique |
| Frozen components | done | FLUX.2 AE 84.0M · SD 1.5 AE 83.7M · T5 enc 109.6M · CLIP ViT-B/32 |
| Model + flow | done | 158.0M params, zero-init exact, non-square grids OK |
| **Overfit-8 gate** | **passed** | loss 2.0 → 0.029; recon 2.65/255 vs the AE's own 2.37 floor |
| Phase A — CUB | done, 6,010 steps | recognisable birds, correct per-species habitats |
| Phase B — COCO | 45k @128² + 30k @256² | prompts followed on held-out captions |
| Phase C — SD 1.5 AE | **next** | latents already cached at both resolutions |
| Phase D — bucketing | pending | 5 buckets specced; RoPE means no retrain |

### Measured

| res | tokens | img/s | s/step (bs 256) |
|---|---|---|---|
| 128² | 64 | ~1390 | 0.184 |
| 256² | 256 | ~320 | 0.800 |

The 4.3× gap matches the 4× token ratio — cost is near-linear in sequence length at this width.

**Curriculum transfer**: resuming a 128²-trained checkpoint at 256² moved loss 0.990 → 1.037
and back under 1.002 within 100 steps. RoPE plus a convolutional patchify means no weight is
resolution-bound.

**Learning rate**: an LR range test from the step-70k checkpoint put the loss minimum at
1.39e-4 and divergence onset at 3.5e-3, against the 1e-4 in use — near-optimal, no change
warranted. Gradient clipping fired 3 times in the whole run, all inside warmup.

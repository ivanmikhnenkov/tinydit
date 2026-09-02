# Dataset survey for tinydit (2026-09-02) — verified against HF API / datasets-server

Samples: `scripts/hfpeek.py sample notes/dataset_preview_spec.json out.json` → `scripts/preview_gallery.py` (contact-sheet artifact).

## Real photos, bytes hosted on HF (no scraping)
| HF id | images | format / size | res | captions | license | verdict |
|---|---|---|---|---|---|---|
| drawthingsai/megalith-10m | ~9.5M | WDS tars, 959 × ~1.9 GB = 1.8 TB | 1024 max side | none (json.caption = Flickr page URL → join) | images PD/CC0/PDM | **Megalith image mirror** |
| moondream/megalith-mdqa | 5.46M | parquet, 3,920 files, 1.06 TB | 1024 | Moondream 1-sentence + QA | openrail | easiest Megalith-with-captions |
| BLIP3o/BLIP3o-Pretrain-Long-Caption `sa_*.tar` | ~11M SA-1B | WDS, 1,000 × 565 MB = 565 GB | 512² centre crop | Qwen2.5-VL-7B ~120 tok | SA-1B research-only | licensed stock photos at training res |
| hanlincs/InternVL-SA1B-Caption-WebDataset | ~11M | WDS 1.61 TB | 512 short side, aspect kept | InternVL paragraph | SA-1B terms | SA-1B with aspect ratio |
| Fhrozen/openimages-narratives-v2 | 1.76M | parquet `train` 580 GB (train_0..f splits likely dup) | ≤1024 | Qwen3-VL-30B object-focused ≤256 tok | OI CC-BY-2.0 | strong |
| common-canvas/commoncatalog-cc-by | 14.6M | parquet partitioned by min side; 256–512 ≈102 GB, 512–768 ≈226 GB | native | BLIP-2 short (weak) + Qwen3.5 recap join | CC-BY | permissive; needs recaps |
| Spawning/pd12m-full | ~12M | WDS 2,509 × 11.7 GB ≈ 29 TB | original | Florence-2 | CDLA-P-2.0 | museum/archive heavy; too big |
| anthracite-org/pixmo-cap-images | 708k | parquet 372 GB | full | human-speech→LLM ~200 words | ODC-BY | best caption fidelity, mixed web content |
| visual-layer/imagenet-1k-vl-enriched | 1.33M | parquet 107 GB (gated) | ~500 | BLIP-2 (often wrong) | ImageNet terms | use i1 imagenet22k captions instead |
| zlab-princeton/i1-pexels-tfrecord | 2.8M | TFRecord 128 × 2.1 GB ≈ 269 GB | **256²** centre crop | i1-captions `pexels` (Qwen3-VL-30B ×5) | MIT tag | Pexels stock at our res |
| i1-datasets/i1-pexels-512-resolution-1m-tfrecord | 1M | 306 GB | 512² | same | | |

## Caption join tables (better than shipped captions)
- zlab-princeton/i1-captions (MIT): Qwen3-VL-30B-A3B, up to 5 dense paragraphs/image; configs megalith10m (9.39M, key = 9-digit row idx), yfcc (97.9M, key = photoid), pexels (2.8M), imagenet22k (13.7M, has `short`). Key joins to drawthingsai/wusize tars UNVERIFIED — spot check.
- BootsofLagrangian/{pd12m,megalith-cc0,commoncatalog-cc-by}-recap-qwen3p5-35b-a3b (CC-BY-4.0): Qwen3.5-35B, 150–450 words; join by `image_shard`/`image_member` or `source_url`.
- CaptionEmporium/flickr-megalith-10m-internvl2-multi-caption: InternVL2-8B + Florence-2 + ShareCap, long AND short per image, join `url_source`.
- aipicasso/megalith-10m-florence2 (8.94M, `url_source`, 60–100 words). drawthingsai/megalith-10m-sharecap (json per key, flowery 170 words).
- laion/220k-GPT4Vision-captions-from-LIVIS: GPT-4V for COCO/LVIS (~66 words), URLs on images.cocodataset.org. PursuitOfDataScience/llama4-maverick-coco-captions: 1-sentence Llama-4 COCO captions. UCSC-VLAA/Recap-COCO-30K.
- undefined443/cc12m-wds-coco-recaptioned (2.99M, 344 GB): Nemotron-12B-VL COCO-style short captions on CC12M.

## URL-only (need img2dataset)
madebyollin/megalith-10m (9.58M Flickr, 96.5% success in 2024); Spawning/PD12M (12.4M, own S3, stable) / PD3M / megalith-cc0 (2.39M CC0, AWS Open Data S3, has aesthetic_score); ptx0/photo-concept-bucket (568k Pexels, CogVLM); allenai/pixmo-cap; tomg-group-umd/pixelprose (16.9M web).

## Synthetic
LucasFang/FLUX-Reason-6M (5.89M FLUX.1-dev 1024², parquet bytes 882 GB, 8 caption fields + quality scores, Apache; 256²/512² TFRecord repacks: zlab-princeton/i1-fluxreason-tfrecord, i1-datasets/i1-fluxreason-512-resolution-1m-tfrecord 302 GB). jackyhate/text-to-image-2M (grab-bag: DALL-E 3 anime-heavy + ~700k real; filter by filename prefix). Photoroom/midjourney-v6-recap (1.24M, 193 GB, MIT). JourneyDB gated + non-commercial. DiffusionDB dated/NSFW.

## Preference sets for RL later
pickapic-anonymous/pickapic_v1 (584k pairs, 202 GB; original yuvalkirstain repos gone), sayakpaul/pickapic_v2_webdataset (563 GB), ymhao/HPDv2 (798k, 31.7 GB), MizzenAI/HPDv3 (1.14M pairs, 141 GB, MIT), zai-org/ImageRewardDB (137k), data-is-better-together/open-image-preferences-v1-binarized, Rapidata/text-2-image-Rich-Human-Feedback, Exploration/richhf_18k_with_images.

## What others trained on / recaption literature
PixArt-α: 10M SAM (LLaVA captions) + 4M JourneyDB + 10M internal; PixArt-Σ 60% long / 40% short raw. Sana: 4 VLM captions per image, CLIP-weighted sampling. Lumina-Image 2.0: long/medium/short/tags. Playground v3: six lengths. Kolors: 50/50 original/synthetic. MicroDiT: CC12M + SA1B + JourneyDB + DiffusionDB (~40% synthetic → CLIP 26.67→28.14, GPT-4o pref 63% vs 21%). BLIP3-o: 25M, ~10% 20-token captions. OpenUni: t2i-2M + LAION-Aes-6M + Megalith-10M + RedCaps-5M. oboro (2025): Megalith-only, Florence-2, 256²→512². i1 (2026): 168M, Qwen3-VL-30B, 5 captions/img sampled per step; long captions win but hurt short prompts unless mixed; equal per-dataset weighting robust; dropping narrow-domain iNaturalist helped. DALL-E 3: 95% synthetic / 5% GT captions. "How to Train your T2I Model" (2025): random-length captions remove the long/short trade-off. Hallucination: LLaVA-1.5 ~17%, Share-Captioner ~20%.

## Unverified
Megalith 9-digit key equivalence across drawthingsai/wusize/i1; i1 yfcc key = CommonCatalog photoid; Fhrozen train vs train_0..f duplication; CommonCatalog per-bucket counts (extrapolated).

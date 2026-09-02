# Architecture research report (agent output, 2026-09-02) — key facts with sources

## 1 Registers
- Darcet et al. ViTs Need Registers https://arxiv.org/abs/2309.16588 — 4 learnable tokens; 1 already removes artifacts.
- Registers Matter for Pixel-Space DiTs https://arxiv.org/abs/2605.16147 — DiTs have no norm outliers yet benefit; insert from layer 4; 32>16>4 for B; VAE-latent B FID 10.40→9.40, L 2.53→2.38; RAE-latent hurt; gains at high noise.
- Taming Outlier Tokens in DiTs https://arxiv.org/abs/2605.05206 — 36 registers from block 8 of RAE-DiT-XL: FID 5.89→4.58; T2I GenEval 42.6→46.6; 100 degrades.
- Attention Sinks in DiTs https://arxiv.org/html/2605.09313v2 — in SD3 99.9% of sinks are text tokens; ablating them leaves CLIP-T unchanged.
- Text Template Tokens Are Implicit Semantic Registers https://arxiv.org/html/2607.19139v1 — template tokens absorb 76% (Qwen-Image) / 92% (FLUX.2) of image→text attention.
- Padding Tone https://arxiv.org/abs/2501.06751 — cross-attn models with frozen encoders ignore pads; T5 EOS is TinyDiT's only global slot.
- Precedents: Stable Cascade c_clip_seq=4 context tokens; imagen-pytorch learnable null_kv per cross-attn; StreamingLLM sink token; gpt-oss per-head sink bias.
- REG https://arxiv.org/abs/2507.01467 — one DINOv2 CLS token denoised jointly: SiT-B 33.0→15.2 (class-cond ImageNet).
- Rec: 16 image-stream registers from block ~3, dropped before unpatchify (+6% attn FLOPs); 1–4 learnable null K/V slots per cross-attn (~75k params). Skip test-time registers.

## 2 Pooled text → adaLN / adaLN-single
- GenTron https://arxiv.org/html/2312.04557v1 T2 — pooled-only adaLN 34.32 vs cross-attn 47.84 (T2I-CompBench).
- Deep Fusion (Meta) https://arxiv.org/html/2505.10046 — adding pooled CLIP to adaLN: GenEval 0.51→0.50, DPG 76.6→76.2, FID 27.33→24.00; "weakens alignment".
- SD3/FLUX add pooled with no ablation; SANA/PixArt-Σ/Lumina-2 timestep-only; HunyuanDiT rejected adaLN for text.
- adaLN-single PixArt-α https://arxiv.org/html/2310.00426v3 Fig 6: 833M→611M, FID slightly higher, visuals on par. DiT-Air https://arxiv.org/html/2503.10618 shared adaLN 902M→631M same val loss. AiM https://arxiv.org/html/2408.12245 adaLN-single 93M 4.21 vs adaLN 134M 3.52 (not param-matched).
- Decision: no pooled text. adaLN-single, and the freed parameters go into depth (16 blocks).

## 3 Text encoder
- SANA T9 https://arxiv.org/html/2410.10629 — T5-XXL 6.1/27.1; T5-Large 6.1/26.2; Gemma-2-2B 6.0/26.9 (+CHI +2.2 GenEval), last layer + RMSNorm + 0.01 scale.
- Scaling Down Text Encoders https://arxiv.org/html/2503.19897 — FLUX.1 COCO FID XXL 22.36 / XL 23.17 / L 24.19 / Base 24.32 / Small 25.10; losses mostly text rendering.
- Comprehensive Study CVPR'25 https://arxiv.org/html/2506.08210 — T5-XXL 0.741 VQAScore; Mistral-7B layer 15 0.725 vs last 0.675; normalized layer-average 0.769.
- LI-DiT https://arxiv.org/html/2406.11831 raw LLaMA3-8B fails to beat T5-XL; ELLA/LLM4GEN T5-XL > TinyLlama/Llama-2.
- DiT-Air T4: bidir CLIP 70.4, 2.8B LLM 72.6, T5-XXL 65.3 GenEval. i1 (2026) https://arxiv.org/html/2606.11289 — T5Gemma beats Qwen3/FG-CLIP2 at 256².
- Layer choice: Lumina-2/Z-Image/Kolors hidden_states[-2]; FLUX.2 layers (10,20,30) of Mistral-24B; klein (9,18,27) of Qwen3.
- PRX https://huggingface.co/blog/Photoroom/prx-part2 — short captions FID 36.8 vs long 18.2 at 256².
- Rec: keep flan-t5-base now; flan-t5-large or T5Gemma-2 when captions get long; Qwen3 only with mid-layer + norm + prefix.

## 4 VAE
- SD3 T3: rFID 2.41/1.56/1.06 for d=4/8/16; higher d harder.
- VA-VAE https://arxiv.org/html/2501.01423 T2: DiT-B 160ep gFID f16d16 16.24 → d32 22.62 → d64 36.83; DINOv2 alignment recovers d32 to 15.82.
- DC-AE 1.5 https://arxiv.org/html/2508.00413v1; Diffusability https://arxiv.org/html/2502.14831.
- BFL representation comparison https://bfl.ai/techblog/representation-comparison/ — DiT-XL IN-256 gFID FLUX.2 3.70, RAE 3.10, SD 7.73, FLUX.1 10.13; FLUX.2 VAE trained with semantic regularization.
- RAE https://arxiv.org/html/2510.11690 — width ≥ token dim; shift α=√(m/4096) FID 23.08→4.81.
- FLUX.2 official applies BatchNorm2d(affine=False) after 2×2 patchify (https://raw.githubusercontent.com/black-forest-labs/flux2/main/src/flux2/autoencoder.py); diffusers encode() never applies bn → dataset-stat whitening is correct.
- Decision: FLUX.2 c32 + per-channel whitening; train+sample shift α=2.8 at 256²-area buckets (m=32·32·32 → √8=2.83).

## 5 Recipe
- REPA https://arxiv.org/abs/2410.06940 — DINOv2-B block 8/28 λ=0.5; T2I MMDiT dim768 MS-COCO 150K bs256 FID 6.05→4.73. HASTE https://arxiv.org/html/2505.16792 stop at ~60%. PRX DINOv3 14.64 vs DINOv2 16.6 vs none 18.2.
- TREAD https://arxiv.org/abs/2501.04765 — MS-COCO DiT-B 35.68→20.55, 1.86→3.03 it/s; PRX worse at 256² (12.08→21.61).
- LightningDiT chain XL/80ep: RF 17.20 → lognorm 13.99 → +cosine 12.52 → SwiGLU 10.10 → RMSNorm 9.25 → RoPE 7.13.
- Dispersive loss https://arxiv.org/abs/2506.09027 — SiT-B/2 36.49→32.35, block 3, λ=τ=0.5.
- Contrastive FM https://arxiv.org/abs/2506.05350 — CC3M MMDiT+REPA 24→19; needs own CFG rule.
- Skip: Immiscible, REPA-E (frozen VAE), JiT x-pred, MeanFlow, DDT, MicroDiT masking (DiT-Tiny 3.79→~7.0 FID).
- SD3 T1 t-sampling: lognorm(0,1) rank 1.54 vs mode 2.75, cosmap 4.13, uniform 5.67. Min-SNR redundant with lognorm (Kingma & Gao).
- Optim: LightningDiT lr 2e-4 bs1024 β2 .95 wd 0; Lumina-2 2e-4 (.9,.95) eps 1e-15; MicroDiT 2.4e-4 wd .1; Muon PRX 15.55 vs AdamW 18.20; muP 2.9× only if sweeping widths.
- EMA: CIFAR RF .999→5.78, .9999→4.77, .99999→3.91 (https://arxiv.org/html/2402.06461); EDM2 post-hoc EMA; PRX: bf16 weights cost FID 18.2→21.9 → fp32 master.
- CFG: p_uncond .1≈.2; autoguidance with undertrained ckpt EDM2-S 2.23→1.51; guidance interval 1.81→1.40; GFT/MG bake CFG (DiT-B 43.5→7.24) but complicate RL.
- head_dim 64 vs 128: no controlled ablation. Depth:width 64 matches DiT scaling law https://arxiv.org/abs/2410.08184. Cross-attn every block vs sparse: no DiT ablation; Intermediate-Fusion ViT better FID/CLIP fusing mid layers −20% FLOPs https://arxiv.org/abs/2403.16530.

## 6 RL readiness
- Flow-GRPO https://arxiv.org/html/2505.05470 — SDE σ_t=a√(t/(1−t)) a=.7; rollouts 10 steps, inference 40; G=24, KL .04; Fast variant no CFG.
- DanceGRPO https://arxiv.org/html/2505.07818 — ε=.3; CFG doubles VRAM; 25–50 steps; optimize 60% timesteps.
- MixGRPO https://arxiv.org/html/2507.21802v3 — SDE only in a 4-step sliding window.
- Implications: keep plain velocity RF + Euler ODE; non-distilled small base; good 10-step quality; EMA as reference policy; CFG doubles RL cost.

## Top-10 (agent ranking)
Adopted for run 1: timestep shift; lr 2e-4 / EMA .9999 / fp32 master weights / bf16 EMA snapshots; cosine velocity loss; 16 registers + null K/V; dispersive loss; adaLN-single with depth 16; FLUX.2 VAE; long+short captions. Deferred: REPA (needs ~150 KB of pixels or features per image on disk), TREAD.
Not worth: pooled text; MMDiT; Immiscible; JiT; MeanFlow; REPA-E; DC-AE/RAE; Min-SNR; muP; head_dim 128; MicroDiT masking; Gemma-3-270M/ModernBERT/CLIP-only; test-time registers; GFT before pretraining.

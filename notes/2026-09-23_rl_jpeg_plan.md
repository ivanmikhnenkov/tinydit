# RL stage 1: JPEG sanity objectives, Flow-GRPO vs DGPO (2026-09-23)

Before a reward worth optimising, learn the two families of diffusion-RL algorithms on objectives
that cannot be argued with, and measure how fast each converges per GPU-hour on this model.
DDPO (Black et al. 2023) used the same pair: JPEG **compressibility** (reward = −file size in kB at
quality 95) and **incompressibility** (+kB). They need no reward model and pull in opposite directions.

## Shared setup (identical for every run)
| | |
|---|---|
| base weights | `out/runs/run1/ema_0400000.safetensors` (the released 400k-step EMA), full fine-tune, fp32 master, bf16 autocast, torch.compile |
| prompts | DDPO's 45 animals as "a photo of a {animal}"; 16 prompts per round, 16 images each = 256 samples/round |
| rollout | 10 steps, shift 2.8, CFG 4, 256×256 |
| optimiser | AdamW lr 1e-5, betas (0.9, 0.999), no weight decay, grad clip 1.0; one optimizer step per 64 samples, every sample used once per round |
| advantage | (r − mean of its group) / std over the whole round (both codebases' `global_std`), clipped at ±5 for GRPO |
| budget | 1.5 GPU-hours of train time per run (rollout + decode/reward + update, CUDA-synchronised); evaluation is off the clock |
| evaluation | every 5 train-minutes: 20-step ODE, CFG 4, all 45 prompts × 2 seeds → kB, CLIP, PickScore; a 16-image grid of the first 16 animals, seed 0 |

## The two algorithms
| | Flow-GRPO (Liu et al. 2025) | DGPO (Luo et al. 2025, ICLR 2026) |
|---|---|---|
| rollout | SDE (noise level 0.7): every step is a Gaussian whose mean depends on the model → a policy with a density | ODE: deterministic, no density |
| what is trained | every one of the 10 steps of every trajectory | one re-noised timestep per group, drawn from the 5 noisiest sampler steps; noise shared inside the group |
| loss | PPO clip (1e-4 on the per-dimension-averaged density ratio) + 0.004 × Gaussian KL to the frozen base | group-level Diffusion-DPO: σ-gated Σ Aᵢ·L_dsm(xᵢ), β = 100, trust-region clip 1e-2 + 0.02 × ‖v − v_ref‖² |
| update cost per round | 10 fwd+bwd per sample (×2 for CFG) + 10 reference fwd | 1 fwd+bwd per sample (×2) + 1 old-policy fwd + 1 reference fwd |
| paper's claim | — | ~20× less wall-clock than Flow-GRPO to the same GenEval score |

Hyper-parameters are the papers' own (Flow-GRPO `config/base.py`; DGPO `config/dgpo.py`), except the
optimiser, which is shared and full-parameter: both papers fine-tune 2B+ models with LoRA, which is a
memory device, not part of the algorithm, and a 209M model does not need it.

## Where to read the code, in order
1. `src/tinydit/rl/rewards.py` — JPEG kB, the 45 prompts, the two signs.
2. `src/tinydit/rl/rollout.py` — the Euler step written once; the SDE, its per-step Gaussian, the density. The docstring carries the SDE in this repo's t-orientation.
3. `src/tinydit/rl/grpo.py` — 30 lines: ratio, clip, KL.
4. `src/tinydit/rl/dgpo.py` — `prepare` (re-noise, score old policy and reference) and `loss` (margin, gate, trust region).
5. `src/tinydit/rl/train.py` — the loop both share; GPU-hour accounting; evaluation; `metrics.jsonl` keys in `notes/CONTRACTS.md`.
6. `rl_dashboard.html` — every `rl_*` run overlaid, x-axis switchable between GPU-hours, samples and rounds; the "same GPU-hours, side by side" strip.

## What to look at
- **eval kB vs GPU-hours** is the result. Rollout kB is inflated for GRPO by the SDE noise (the last step still adds noise, as in the reference code), so compare algorithms on eval kB.
- **PickScore and CLIP at equal kB**: the price of the reward. A run that reaches 3 kB by drawing grey squares loses CLIP; one that draws a flat cartoon animal keeps it. DDPO's compressibility images were smooth and detail-free; incompressibility ones were busy high-contrast textures.
- **group std**: the signal both feed on. When it collapses the policy has gone deterministic for the prompt.
- **DGPO's gate σ(margin)**: 0.5 at the start, falls as the preference is satisfied. **GRPO's ratio and clip fraction**: ratio at 1.0000 ± few 1e-4; a rising clip fraction means the learning rate is too aggressive for the per-round drift.
- The two KL curves measure different quantities (Gaussian KL of step means vs velocity MSE); compare each with itself.

## Run order and launch
`scripts/launch_rl_jpeg.sh` waits for a free GPU, then runs `rl_dgpo_compress`, `rl_grpo_compress`, `rl_dgpo_incompress`, `rl_grpo_incompress`
(≈ 6 GPU-hours plus evaluation). Dashboard: http://localhost:7180/tinydit/rl_dashboard.html

## Next
Stage 2 is a reward worth having (PickScore / HPSv2 / object accuracy on the held-out prompts), with the
algorithm chosen by this experiment. Same repo, tag `v2-rl`; weights to `tinydit-256-grpo` with a `base_model` link.

## Results (all four runs finished 2026-09-23 22:48; 1.5 GPU-h each)
| run | rounds | samples | s/round (rollout + decode/reward + update) | eval kB start → best → end | CLIP end | PickScore end |
|---|---|---|---|---|---|---|
| DGPO compress | 573 | 146,688 | 4.2 + 3.1 + 2.1 = 9.4 | 39.2 → 22.7 (0.09 h) → 30.9 | 0.302 | 20.77 |
| Flow-GRPO compress | 223 | 57,088 | 4.2 + 3.1 + 17.0 = 24.3 | 39.2 → 1.7 (1.05 h, black frames) → 1.7 | 0.244 | 18.10 |
| DGPO incompress | 567 | 145,152 | 4.2 + 3.2 + 2.1 = 9.5 | 39.2 → 55.8 (0.09 h) → 47.1 | 0.302 | 20.91 |
| Flow-GRPO incompress | 220 | 56,320 | 4.2 + 3.4 + 17.0 = 24.6 | 39.2 → 86.0 (1.1 h, noise fields) → 86.0 | 0.215 | 16.33 |

Base: 39.2 kB, CLIP 0.302, PickScore 20.99. Report with curves and grids: https://claude.ai/code/artifact/28eeb94e-ed19-48e2-8a15-1cfd6a30959d

What happened, and why (same on both objectives):
- **DGPO** moves in minutes (23 kB / 56 kB at 0.09 GPU-h with CLIP and PickScore unchanged), then relaxes back toward the base. Its gate
  σ(margin) falls from 0.5 to ~0.08 within 20 min (0.02 at the end of incompress) as the group preference is satisfied, which almost switches the
  preference gradient off; the KL term (0.02 on ‖v − v_ref‖², the paper's OCR/GenEval setting) then wins and pulls the model back.
- **Flow-GRPO** never stops: its KL (0.004 on a step-mean distance of ~1e-3) restrains nothing, so it walks to the reward's global optimum,
  a black frame (1.7 kB) or a blue-yellow noise field (86 kB). Group std → 0 there: deterministic policy, no signal left. The useful models are
  mid-run (round 65 compress: 9 kB, CLIP 0.307; round 52 incompress: 72 kB, CLIP 0.293).
- **Speed**: DGPO 2.6× rounds per GPU-hour (update 2.1 s vs 17 s); ahead at equal GPU-hours for the first ~15 min, then Flow-GRPO keeps descending.
- Flow-GRPO's clip fraction sat at ~0.24 from the first round: with densities averaged over 32k dims, the bf16 re-evaluation noise alone crosses the 1e-4 clip.

Next: DGPO with KL 0.001 (its PickScore setting) or EMA reference; Flow-GRPO with KL ×100 or an early stop on CLIP; then stage 2 on a reward with an interior optimum.

## Follow-ups (2026-09-24)
- Report now carries the verdict, the lessons, DGPO's loss derived step by step, a map of post-training families
  (policy gradient / likelihood-free / offline DPO / reward gradient / reward-weighted SFT / test-time), and the stage-2 plan:
  DGPO (KL 0.001 or EMA reference) as the main method, Flow-GRPO (KL x100) as the baseline, a DRaFT-style reward-gradient run as the
  cheap third point, DiffusionNFT as the second likelihood-free method if wanted.
- `exercises/dgpo/`: 2-D toy of DGPO (flow-matching MLP over two modes, reward = x[0]) with four functions to implement, 12 property
  tests (`.venv/bin/python -m pytest exercises/dgpo -q`; `DGPO_IMPL=solution` runs the reference) and eight questions.
  The toy uses beta_dpo = 3: 2-D deltas are ~30x the DiT's per-dimension ones, and beta = 100 closes the gate in the first round.

## Stage 2 preparation (2026-09-26)
- Reward-design study notes (families, VLM-as-judge modes, hack watchlist, validation checklist, tinydit candidates) in the
  "Reward Design for TinyDiT" artifact. Shortlist: counting + two-object binding with the COCO detector (task) with PickScore as a guard
  (advantage-level sum) = stage 2a; PickScore with HPSv2.1 held out = stage 2b; VQAScore (Qwen2.5-VL-7B, P(yes)) as the held-out judge
  first, as a reward later. Run the validation checklist on base samples before training.
- Toy finding (exercises/dgpo): with a fixed reference the DGPO gate closes for good (ranking margin, not level) and progress continues at
  ~5 % speed; KL 0.02 alone does not reverse it in 2-D; an EMA reference (0.3 old + 0.7 policy per round) keeps the gate at 0.5 and goes
  all the way. The real-model pull-back is still unexplained: queued test = continue rl_dgpo_compress from policy_final with (a) same ref,
  KL 0 (train.py now has --ref) and (b) ref reset to the policy; 15 min each once the GPU is free (a root vLLM engine held 91 GB on 09-26).

## Reward / judge study (2026-09-29), report: "Judges for TinyDiT RL" artifact
Objectives for stage 2 (Ivan): fewer artifacts (animals/objects), better prompt alignment, keep the realistic style; a general reward, not a narrow one.
Set: 120 prompts (40 held-out real captions, 40 held-out FLUX captions, 20 usual, 20 unusual hand-written) x 8 seeds; 150 real + 150 FLUX
references; 176 labelled (21 artifacts, 18 uncertain, 137 clean; 14 alignment failures). Cheap results: PickScore/HPSv2 rate FLUX renders above
real photos (AUROC 0.29 / 0.23) -> anti-realism as raw rewards; DINOv2 real-vs-render probe 0.994, tinydit in between; CLIP misses action
failures (0.62) while HPSv2 flags them (0.90); only HPSv2 has artifact signal (0.73); group SNR of preference models < 2; tinydit renders in
the caption's dialect (FLUX-style captions -> FLUX look, probe 0.11 vs 0.78). Judges: HPSv3 (re-implemented, Qwen2.5-VL-7B + ranknet head on
<|Reward|>), Qwen3-VL-8B, Qwen3.5-4B/9B, Qwen3.6-35B-A3B (72 GB, judge-only), Qwen3.8-27B (not downloaded, disk). scripts/reward_eval/run_judges.sh
waits for the GPU (root vLLM engine occupied it on 09-29) and runs everything; analyze.py produces analysis.json.

### Judge results (2026-09-30 02:30, four of five; 27B running)
HPSv3 (re-implemented: Qwen2-VL-7B class + Qwen2-VL config, HPSv3.safetensors symlinked as model.safetensors, <|Reward|> token, ranknet head):
real-vs-FLUX 0.21 (prefers renders like HPSv2/PickScore), artifacts 0.75, hard-neg 0.95, group SNR 2.9, 10 img/s. Qwen3.5-4B: align failures 0.81,
SNR 3.2, anatomy/real questions SNR ~1, 30 img/s. Qwen3-VL-8B: failures 0.94, hard-neg 0.98, SNR 4.1, anatomy 0.74 (SNR 2.3), rubric integrity
0.79, 32 img/s -> best per-sample reward. Qwen3.5-9B: failures 0.94, hard-neg 0.99, AI-look real-vs-FLUX 0.92 -> best judge, 16 img/s.
Artifact ceiling ~0.75-0.79 for everything; descriptions say "none" for gross anatomy. Rubric 1-10 worse than logits everywhere but 8B integrity.
Stage-2 reward proposal: advantages of [Qwen3-VL-8B align P(yes)] + [DINOv2 real-vs-render probe] + [HPSv3 mu, small weight]; judges Qwen3.5-9B,
labelled sets, FD-DINOv2. To raise the artifact ceiling: judge at 896^2, label ~500 samples for a DINO artifact probe, pairwise judging.
Qwen3.8-27B (2026-09-30 03:00): best "real photograph?" judge (real vs FLUX 0.89), AI-look 0.90, alignment stricter than the labels
(failures 0.77, hard-neg 0.97, says no to 26 % of unusual-prompt samples), anatomy 0.75, rubric integrity 0.77, descriptions flag 71 % of
artifacts but also 53 % of clean images; 10 img/s yes/no, 2.6 rubric, 1.1 describe, 73 GB peak at batch 16, loads in 13 s -> judge only.
Report final at the "Judges for TinyDiT RL" artifact. The chatbot vLLM container is stopped (docker start chatbot-nebius_vllm_gemma4_vision_dev-1 to restore).
Explorer artifact (12 images x every judge, per-image / per-scorer / matrix views): https://claude.ai/code/artifact/ff8a63ef-58e6-465e-8bad-b31c9379b3c5 (data from scripts/reward_eval, built in the session scratchpad).

## Stage 2 build (2026-09-30 evening)
The latent cache (out/cache/run1) was deleted to free disk, so: whitening stats now read from out/hf_model/stats.json (train.py --stats
fallback); real references rebuilt from COCO val2017 (5,000) + 150 Pexels/COCO refs -> out/data/refs/real_256.npz; FLUX renders re-fetched
from FLUX-Reason shards (scripts/rl/fetch_flux_refs.py, curl for the Hub API) -> flux_256.npz; realism probe retrained by
scripts/rl/train_dino_probe.py (provisional 150+150 probe in place until then). Prompts: prompts/rl_stage2/train.json = 20k COCO captions +
8,352 synthetic "subject doing X somewhere" + object x place; eval.json = 45 animals + 40 hand + 40 held-out captions, 16-prompt grid.
Code: src/tinydit/rl/judges.py (VLM yes/no + generate, HPSv3 re-impl, DinoProbe, pixel_stats), reward_stage2.py (weights 0.40 align /
0.30 real / 0.15 HPSv3 / 0.15 anatomy, veto), train.py --reward stage2 (terms, advantages, eval with per-image manifests, pixel stats,
FID/FD-DINO vs real refs, policy snapshot per eval), scripts/rl/judge_evals.py (post-hoc 9B + 27B descriptions -> judges.jsonl),
scripts/rl/validate_reward.py (hacks), scripts/rl/launch_stage2.sh (waits for GPU: probe -> validation -> smoke -> run -> judges).
rl_dashboard.html: stage-2 charts (terms, eval by source, realism with 9B, quality/anatomy, saturation/contrast/sharpness, FD/FID, CLIP,
9B judge) + "fixed held-out prompts over training" section with baseline-vs-current cards and per-image scores.
Combined reward on base samples: SNR 3.4; inside groups it follows align (+0.49) and HPSv3 (+0.54) more than real (+0.19).
GPU was re-occupied by the chatbot vLLM container in the afternoon (someone restarted the stack); launcher waits.

## Stage-2 run (2026-09-30 evening -> 2026-10-01 01:14)
Five launches of the same 4-GPU-h DGPO recipe; each of v1-v4 exposed a reward-shaping bug within 20 min (HPSv3 dominating advantages;
alignment veto at -2 on 18 % of samples + inherited closed gate; one-sided probe rewarding desaturation; the style guard shifting the
distribution), fixed and restarted from the last snapshot. v5: 308 rounds, 25 evals, judged by the 9B/27B. Held-out 9B alignment
0.805->0.837 (unusual prompts 0.717->0.83), 9B AI-look 0.031->0.020, 9B real 0.633->0.655; probe 0.583->0.718 but FD-DINOv2 worsened after
~2.5 h (probe partly gamed late); anatomy flat 0.92-0.94; 27B flags unchanged (8/16 -> 7/16). From the original weights: 8B align
0.784->0.837, sat 0.35->0.26, sharpness 2590->2220, FID 149.4->149.0, FD 1929->1969, CLIP flat.
Pick: out/runs/rl2_dgpo_v5/policy_000221.safetensors (rule in scripts/rl/summarize_run.py). Next: lr 2e-5 from the pick with the judge
pass in the loop and FD/9B as the stop rule; an artifact probe from ~500 labels; the remaining alignment failures are unseen compositions.
Dashboards: rl2_dashboard.html (stage 2), report artifact section 6.

## Paused 2026-10-01 ~14:10 (Ivan: "stop train for now, resume when I tell you")
State: v6 (lr 2e-5, no HPSv3) softened images within 45 min; v7 (realism = 8B "real photograph?") still softened; v8 = alignment-only
diagnostic (w_align 1.0, style 0.10, others 0) from out/runs/rl2_dgpo_v6/policy_000062.safetensors, stopped after its base eval with a
checkpoint in out/runs/rl2_dgpo_v8/ckpt_last.pt. Resume command (same place, same settings):
  cd ~/volume/learning/tinydit && RUN=rl2_dgpo_v8 HOURS=1 SKIP_CHECKS=1 EXTRA="--base out/runs/rl2_dgpo_v6/policy_000062.safetensors --ref out/runs/rl2_dgpo_v6/policy_000062.safetensors --real-source vlm --w-align 1.0 --w-real 0 --w-qual 0 --w-anat 0 --w-style 0.10 --lr 2e-5 --eval-every-min 15 --stop-fd 1.015 --stop-patience 2" nohup scripts/rl/launch_stage2.sh > out/logs/launch_stage2_v8.log 2>&1 &
  (the launcher adds --resume automatically when ckpt_last.pt exists). Purpose of v8: does sharpness hold with alignment alone? Then re-add
  anatomy / realism one at a time at small weight for the 6-h run.

## lr 2e-5 round, 2026-10-01 12:29 -> 2026-10-02 01:40 (v6-v10)
v6 (no HPSv3) and v7 (8B real question) softened images within 30-45 min; v8 alignment-only kept sharpness; v9 (anatomy 0.10) softened
again -> anatomy + realism terms are the softeners. v10 = alignment 0.90 + style 0.10, EMA 0.995, lr 2e-5, 3.3 GPU-h (stopped by hand:
realism monitors fell for 2 h; FD stayed flat so the FD rule never fired). Judged: 9B align 0.827 -> 0.835-0.840 (plateau), 9B real
0.644 -> 0.553, AI-look 0.022 -> 0.036, CLIP 0.324 -> 0.318. Pick by rule = v10 r192 (9B align 0.835, AI-look 0.028, FD 1952) but it does
not beat v5 r221 (0.837 / 0.020 / real 0.655). STAGE-2 RESULT = out/runs/rl2_dgpo_v5/policy_000221.safetensors. RL plateau with this
base ~0.84 judged alignment; next gains are pretraining (REPA continuation, 512^2, text encoder). Policy EMA (--ema) now in train.py.

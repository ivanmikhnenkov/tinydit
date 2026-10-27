# RL results page: outline for review

Target: `docs/rl/README.md`, at most 1,200 words, one table, at most 4 static figures.
Every number below names the file it comes from. All `out/` files are git-ignored; the page cites them by path.

## Headline bullets

1. **Alignment pass@1 rose, pass@64 did not.** 125 held-out prompts x 64 seeds, 20-step ODE, CFG 4, pass = judge P(yes) >= 0.5.
   Qwen3-VL-8B (the reward model): base pass@1 0.784 -> 0.831 (v10 r192), 0.820 (v5 r221); pass@64 0.968 -> 0.968 / 0.960.
   Qwen3.5-9B (independent): pass@1 0.805 -> 0.854 / 0.847; pass@64 0.960 for every checkpoint.
   Source: `out/passk/summary.json`.
   Note for the reviewer: the task text says "0.79 -> 0.84". That pair comes from the two-seed training evals
   (`out/runs/rl2_dgpo_v5/metrics.jsonl`, eval_align 0.784 -> 0.837 at r308). The 64-seed figures are 0.78 -> 0.83 (8B)
   or 0.81 -> 0.85 (9B). The page will use the 64-seed numbers unless told otherwise.

2. **DGPO's gate never left 0.5 on this reward, at any beta.** Gate = sigmoid(group margin), mean over a run:
   v5 0.485 (308 rounds), v10 0.492 (314); beta sweep at 100 / 1000 / 5000: 0.491 / 0.473 / 0.485 (40 rounds each).
   At beta 5000 the gate is a coin flip per group: std across the 16 groups 0.36, 31 % of groups below 0.1, 21 % above 0.9
   (last 10 rounds). Stage 1 (JPEG objective, same code) closed it: 0.43 -> 0.07, margin -3.9.
   Sources: `out/runs/{rl2_dgpo_v5,rl2_dgpo_v10,diag_beta100,diag_beta1000,diag_beta5000,rl_dgpo_compress}/metrics.jsonl`
   (fields w, margin, w_std, w_lo, w_hi); aggregates in `out/diag/report_data.json`.

3. **The update itself softens images; the judges do not ask for it.** Alignment-only recipe, beta 100, 40 rounds (0.34 GPU-h),
   held-out sharpness = Laplacian variance over 125 prompts x 2 seeds:
   normal advantages 2591 -> 1778 (-31 %), 8B alignment 0.784 -> 0.811;
   advantages x -1: 2591 -> 3658 (+41 %), alignment 0.784 -> 0.759 (low 0.738 at r36);
   advantages shuffled inside each group: 2591 -> 2395 (-8 %), alignment 0.784 -> 0.767, with the same weight movement
   (KL to reference 0.0155 vs 0.0167 for the normal run).
   Judge sharpness preference on the base's own 64 samples per prompt (within-prompt Spearman, P(yes) vs sharpness):
   -0.017 (8B), +0.026 (9B).
   Sources: `out/runs/{diag_beta100,diag_flip,diag_shuffle}/metrics.jsonl` (eval rows and last-10 kl), `out/passk/bias_base.json`.

4. **Training only on prompts the base solves 1-6 times out of 8 gives the same gain in half the GPU-hours, with sharpness kept.**
   Band: 3,000 training prompts x 8 seeds under the 8B judge; 299 never pass, 1,014 always pass, 1,291 kept (454 captions,
   837 synthetic). Source: `out/diag/prompt_band.json` -> `prompts/rl_stage2/train_band.json`.
   Two 1.2 GPU-h arms (140 rounds, alignment-only, beta 100, EMA 0.995), final checkpoints at 125 prompts x 32 seeds:
   8B pass@1 base 0.780, all prompts 0.813, band 0.811; pass@32 0.960 / 0.968 / 0.960.
   9B pass@1 0.805 / 0.840 / 0.840; pass@32 0.960 / 0.952 / 0.968. Source: `out/passk_long/summary.json`.
   Speed: 9B alignment at 0.5 GPU-h is 0.827 (band, r60) vs 0.797 (all prompts, r59); all prompts reaches 0.828 at 1.2 GPU-h.
   Two-seed evals. Source: `out/runs/{diag_long_curr,diag_long_beta100}/judges.jsonl`.
   Sharpness at the final eval: all prompts 1682 (-35 % vs base 2591), band 2475 (-4 %). 9B "real photograph?": 0.597 vs 0.611
   (base 0.607); 9B AI-look 0.035 vs 0.031 (base 0.031). FD-DINOv2 is worse for the band arm: 1973 vs 1953 (base 1929); the page says so.
   Source: `out/runs/{diag_long_beta100,diag_long_curr}/metrics.jsonl` and `judges.jsonl`.

## Setup lines (sources)
Model 208.8M (`out/runs/*/meta.json` params_m), 256 px, FLUX.2 VAE, flan-t5-base. Rewards: Qwen3-VL-8B P(yes) alignment,
DINOv2 realism probe, HPSv3, 8B anatomy question (`src/tinydit/rl/reward_stage2.py`); judge Qwen3.5-9B. Sampler: 10-step ODE
for rollouts, 20-step ODE / CFG 4 / shift 2.8 for evals (`meta.json` config). Compute: one RTX PRO 6000; GPU-hours per row below.

## Table rows and the GPU-hours column
Rows: base; v5 r221 (stage-2 pick); v10 r192 (alignment-only pick); v5 r308; all-prompts 1.2 h arm; band 1.2 h arm.
Columns: pass@1, pass@64, sharpness, FD-DINOv2, GPU-hours; each row states n and judge.
- pass@1 / pass@64: `out/passk/summary.json` (125 x 64). The two 1.2 h arms were judged at 32 seeds (`out/passk_long/summary.json`),
  so their row shows pass@32 with n stated. Re-judging them at 64 seeds would cost about 25 GPU-min via the queue; not planned unless asked.
- sharpness, FD-DINOv2: the run's own eval at that round (125 x 2): base 2591 / 1929; v5 r221 2216 / 1969; v10 r192 1940 / 1952;
  v5 r308 2352 / 1966; all-prompts r140 1682 / 1953; band r140 2475 / 1973. Source: `out/runs/<run>/metrics.jsonl` eval rows.
- GPU-hours counted along the resume chain (`meta.json` base fields): v1 0.34 (to r26) + v3 0.56 + v4 0.20 + v5 = v5 r221 3.98,
  v5 r308 5.11; v10 r192 = 3.98 + v6 0.51 + 2.01 = 6.50; the long arms 1.20 each from the base. The discarded v2 (0.36) is not counted.

## Figures (4)
1. `passk.png`: pass@k vs k (1..64, log x), two panels (8B, 9B), base + three checkpoints. Data: `out/passk/summary.json`.
2. `raccoon.jpg`: "a raccoon driving a car", 16 of 64 seeds, base vs v10 r192, 8B P(yes) under each cell; 8B passes 4/64 -> 26/64,
   9B 12/64 -> 40/64. Files: `out/passk/grids/odd01_base.jpg`, `odd01_v10_192.jpg`; counts from `summary.json` c_per_prompt.
3. `controls.png`: held-out sharpness and 8B alignment vs round for normal / flipped / shuffled advantages.
   Data: `out/runs/{diag_beta100,diag_flip,diag_shuffle}/metrics.jsonl`.
4. `hack.png`: the realism-probe hack in v5: probe term 0.583 -> 0.718 over 4 GPU-h while saturation fell 0.292 -> 0.257 and
   FD-DINOv2 went from 1956 (r39) to 1989 (r247). Caught by FD-DINOv2 moving the wrong way and the 9B "real photograph?" judge
   (0.633 -> 0.655 only); the pick rule in `scripts/rl/summarize_run.py` (FD within 1 % of the run's best) chose r221, not r308.
   Data: `out/runs/rl2_dgpo_v5/metrics.jsonl`, `judges.jsonl`. Alternative if preferred: the v6/v7/v9 softening from the
   realism and anatomy terms (sharpness 2217 -> 1776 in 45 min, `out/runs/rl2_dgpo_v6/metrics.jsonl`).

Rendering: matplotlib is not in `.venv`. Either add it (pinned in `requirements.txt`, user space) or draw the three charts with
PIL, which is installed. Figure 2 is a PIL paste either way.

## "What didn't work" and limits (bullets, to be kept short)
- Four stage-2 launches (v1-v4) each exposed a reward-shaping bug within 20 min: HPSv3 dominating advantages; alignment veto on 18 %
  of samples; one-sided probe rewarding desaturation; the style guard shifting the distribution (`notes/2026-09-23_rl_jpeg_plan.md`).
- Realism and anatomy terms soften images within 30-45 min (v6, v7, v9); the artifact judge ceiling is AUROC 0.79 at 256 px
  (`out/reward_eval/analysis.json`).
- Beta 1000 / 5000 learn slower than 100 and soften the same (bullet 2 sources).
- SDE rollouts (Flow-GRPO, a = 0.7) are 31 % sharper than ODE samples from the same seeds (`out/diag/sanity.json`, check 6).
- Limits: 256 px, 209M, one GPU, judge ceilings, two-seed training evals (+-0.01), 5-9 prompts no checkpoint ever solves.

## Reproduce (commands to list)
`scripts/rl/passk.py` (seeds 20000+), `scripts/rl/prompt_band.py` (seeds 30000+), `scripts/rl/sanity_checks.py`,
`python -m tinydit.rl.train --reward stage2 --alg dgpo ...` with the exact flags from `scripts/rl/diagnostics3.sh` and
`launch_stage2.sh`; `scripts/rl/judge_evals.py`; config paths `prompts/rl_stage2/{train,train_band,eval}.json`; `--seed 0`.
Code to ship: `src/tinydit/rl/`, `scripts/rl/`, `exercises/dgpo/`; link from the main README "Results" section.

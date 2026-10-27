#!/usr/bin/env bash
# Post-training diagnostics after stage 2, sequential on one GPU:
#   pass@k on four checkpoints under two judges -> beta_dpo sweep (does the DGPO gate move?) -> advantage-flip control
#   -> 1.2 GPU-h A/B of beta 100 vs the picked beta (alignment-only recipe, EMA 0.995) -> Qwen3.5-9B judge pass on every run.
#   nohup scripts/rl/diagnostics.sh > out/logs/diagnostics.log 2>&1 &
set -u; cd "$(dirname "$0")/../.."; export PYTHONPATH=src; mkdir -p out/logs out/diag out/passk
log(){ echo "$(date +%H:%M:%S) $*"; }
PY=.venv/bin/python; BASE=out/runs/run1/ema_0400000.safetensors
ALIGN="--reward stage2 --alg dgpo --base $BASE --w-align 1.0 --w-real 0 --w-qual 0 --w-anat 0 --w-style 0.10 --lr 2e-5 --beta 0.001 --no-compile --ckpt-every-min 30"
log "pass@k"
$PY scripts/rl/passk.py --ckpt base=$BASE --ckpt v5_221=out/runs/rl2_dgpo_v5/policy_000221.safetensors \
    --ckpt v10_192=out/runs/rl2_dgpo_v10/policy_000192.safetensors --ckpt v5_308=out/runs/rl2_dgpo_v5/policy_000308.safetensors > out/logs/diag_passk.log 2>&1; log "  exit $?"
for b in 100 1000 5000; do
  log "beta sweep $b"; $PY -m tinydit.rl.train $ALIGN --run diag_beta$b --beta-dpo $b --rounds 40 --hours 1 --eval-every-min 6 > out/logs/diag_beta$b.log 2>&1; log "  exit $?"
done
log "advantage flip"; $PY -m tinydit.rl.train $ALIGN --run diag_flip --beta-dpo 100 --adv-sign -1 --rounds 40 --hours 1 --eval-every-min 6 > out/logs/diag_flip.log 2>&1; log "  exit $?"
PICK=$($PY scripts/rl/pick_beta.py 2> out/diag/pick_beta.json); log "picked beta $PICK ($(cat out/diag/pick_beta.json))"
for b in 100 $PICK; do
  log "long beta $b"; $PY -m tinydit.rl.train $ALIGN --run diag_long_beta$b --beta-dpo $b --ema 0.995 --hours 1.2 --eval-every-min 15 > out/logs/diag_long_beta$b.log 2>&1; log "  exit $?"
done
for r in diag_beta100 diag_beta1000 diag_beta5000 diag_flip diag_long_beta100 diag_long_beta$PICK; do
  log "judge $r"; $PY scripts/rl/judge_evals.py --run $r >> out/logs/diag_judges.log 2>&1; log "  exit $?"
done
log "done"

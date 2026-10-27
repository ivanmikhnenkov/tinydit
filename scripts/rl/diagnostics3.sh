#!/usr/bin/env bash
# Third driver, replacing the tail of diagnostics.sh (its picker chose beta 5000, whose per-group gate the sweep showed to be
# a coin flip): the 1.2 GPU-h beta-100 control from the base, the prompt-curriculum arm on the same recipe, then the 9B judge
# pass over every diag run.   nohup scripts/rl/diagnostics3.sh > out/logs/diagnostics3.log 2>&1 &
set -u; cd "$(dirname "$0")/../.."; export PYTHONPATH=src
log(){ echo "$(date +%H:%M:%S) $*"; }
PY=.venv/bin/python; BASE=out/runs/run1/ema_0400000.safetensors
ALIGN="--reward stage2 --alg dgpo --base $BASE --w-align 1.0 --w-real 0 --w-qual 0 --w-anat 0 --w-style 0.10 --lr 2e-5 --beta 0.001 --no-compile --ckpt-every-min 30"
log "long beta 100 (control)"; $PY -m tinydit.rl.train $ALIGN --run diag_long_beta100 --beta-dpo 100 --ema 0.995 --hours 1.2 --eval-every-min 15 > out/logs/diag_long_beta100.log 2>&1; log "  exit $?"
log "curriculum arm"; $PY -m tinydit.rl.train $ALIGN --run diag_long_curr --prompts prompts/rl_stage2/train_band.json --beta-dpo 100 --ema 0.995 --hours 1.2 --eval-every-min 15 > out/logs/diag_long_curr.log 2>&1; log "  exit $?"
for r in diag_beta100 diag_beta1000 diag_beta5000 diag_flip diag_long_beta100 diag_long_curr; do
  log "judge $r"; $PY scripts/rl/judge_evals.py --run $r >> out/logs/diag_judges.log 2>&1; log "  exit $?"
done
log "done"

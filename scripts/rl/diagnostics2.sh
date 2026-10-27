#!/usr/bin/env bash
# Second driver: waits for diagnostics.sh to finish, then runs the prompt-curriculum arm (train only on prompts the base
# solves 1-6 times out of 8; prompts/rl_stage2/train_band.json from scripts/rl/prompt_band.py) and its 9B judge pass.
#   nohup scripts/rl/diagnostics2.sh > out/logs/diagnostics2.log 2>&1 &
set -u; cd "$(dirname "$0")/../.."; export PYTHONPATH=src
log(){ echo "$(date +%H:%M:%S) $*"; }
PY=.venv/bin/python; BASE=out/runs/run1/ema_0400000.safetensors
ALIGN="--reward stage2 --alg dgpo --base $BASE --w-align 1.0 --w-real 0 --w-qual 0 --w-anat 0 --w-style 0.10 --lr 2e-5 --beta 0.001 --no-compile --ckpt-every-min 30"
until grep -qE ':[0-9]{2} done$' out/logs/diagnostics.log; do sleep 60; done; log "driver 1 finished"
until [ -f prompts/rl_stage2/train_band.json ]; do sleep 60; done
log "curriculum arm"; $PY -m tinydit.rl.train $ALIGN --run diag_long_curr --prompts prompts/rl_stage2/train_band.json --beta-dpo 100 --ema 0.995 --hours 1.2 --eval-every-min 15 > out/logs/diag_long_curr.log 2>&1; log "  exit $?"
log "judge"; $PY scripts/rl/judge_evals.py --run diag_long_curr >> out/logs/diag_judges.log 2>&1; log "  exit $?"
log "done"

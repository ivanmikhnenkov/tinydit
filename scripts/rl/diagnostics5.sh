#!/usr/bin/env bash
# Fifth driver (replaces diagnostics4.sh after the disk filled during the curriculum arm): wait for diagnostics3.sh, resume the
# curriculum arm from its last checkpoint, judge it, then the shuffled-advantage control and its judge pass.
#   nohup scripts/rl/diagnostics5.sh > out/logs/diagnostics5.log 2>&1 &
set -u; cd "$(dirname "$0")/../.."; export PYTHONPATH=src
log(){ echo "$(date +%H:%M:%S) $*"; }
PY=.venv/bin/python; BASE=out/runs/run1/ema_0400000.safetensors
ALIGN="--reward stage2 --alg dgpo --base $BASE --w-align 1.0 --w-real 0 --w-qual 0 --w-anat 0 --w-style 0.10 --lr 2e-5 --beta 0.001 --no-compile --ckpt-every-min 30"
until grep -qE ':[0-9]{2} done$' out/logs/diagnostics3.log; do sleep 60; done; log "driver 3 finished"
log "curriculum arm, resumed"; $PY -m tinydit.rl.train $ALIGN --run diag_long_curr --prompts prompts/rl_stage2/train_band.json --beta-dpo 100 --ema 0.995 --hours 1.2 --eval-every-min 15 --resume > out/logs/diag_long_curr_resume.log 2>&1; log "  exit $?"
log "judge diag_long_curr"; $PY scripts/rl/judge_evals.py --run diag_long_curr >> out/logs/diag_judges.log 2>&1; log "  exit $?"
log "shuffled advantages"; $PY -m tinydit.rl.train $ALIGN --run diag_shuffle --beta-dpo 100 --adv-shuffle --rounds 40 --hours 1 --eval-every-min 6 > out/logs/diag_shuffle.log 2>&1; log "  exit $?"
log "judge diag_shuffle"; $PY scripts/rl/judge_evals.py --run diag_shuffle >> out/logs/diag_judges.log 2>&1; log "  exit $?"
log "done"

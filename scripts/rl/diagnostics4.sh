#!/usr/bin/env bash
# Fourth driver: after diagnostics3.sh, the shuffled-advantage control (40 rounds, beta 100, alignment-only recipe): if the
# images still soften with advantages that carry no information, the softening is a property of the update, not the reward.
#   nohup scripts/rl/diagnostics4.sh > out/logs/diagnostics4.log 2>&1 &
set -u; cd "$(dirname "$0")/../.."; export PYTHONPATH=src
log(){ echo "$(date +%H:%M:%S) $*"; }
PY=.venv/bin/python; BASE=out/runs/run1/ema_0400000.safetensors
ALIGN="--reward stage2 --alg dgpo --base $BASE --w-align 1.0 --w-real 0 --w-qual 0 --w-anat 0 --w-style 0.10 --lr 2e-5 --beta 0.001 --no-compile --ckpt-every-min 30"
until grep -qE ':[0-9]{2} done$' out/logs/diagnostics3.log; do sleep 60; done; log "driver 3 finished"
log "shuffled advantages"; $PY -m tinydit.rl.train $ALIGN --run diag_shuffle --beta-dpo 100 --adv-shuffle --rounds 40 --hours 1 --eval-every-min 6 > out/logs/diag_shuffle.log 2>&1; log "  exit $?"
log "judge"; $PY scripts/rl/judge_evals.py --run diag_shuffle >> out/logs/diag_judges.log 2>&1; log "  exit $?"
log "done"

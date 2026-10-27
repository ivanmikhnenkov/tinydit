#!/usr/bin/env bash
# Sixth driver: after diagnostics5.sh, a precise comparison of the two 1.2 GPU-h arms against the base: 125 held-out prompts x
# 32 seeds under both judges (pass@k, per family), then drop the optimizer checkpoints of the finished diag runs.
#   nohup scripts/rl/diagnostics6.sh > out/logs/diagnostics6.log 2>&1 &
set -u; cd "$(dirname "$0")/../.."; export PYTHONPATH=src
log(){ echo "$(date +%H:%M:%S) $*"; }
PY=.venv/bin/python; BASE=out/runs/run1/ema_0400000.safetensors
until grep -qE ':[0-9]{2} done$' out/logs/diagnostics5.log; do sleep 60; done; log "driver 5 finished"
rm -f out/runs/diag_long_curr/ckpt_last.pt out/runs/diag_shuffle/ckpt_last.pt out/runs/diag_shuffle/policy_*.safetensors
C=$(ls out/runs/diag_long_beta100/policy_0*.safetensors | sort | tail -1); K=$(ls out/runs/diag_long_curr/policy_0*.safetensors | sort | tail -1); log "control $C  curriculum $K"
log "pass@32"; $PY scripts/rl/passk.py --k 32 --out out/passk_long --ckpt base=$BASE --ckpt control=$C --ckpt curriculum=$K > out/logs/diag_passk_long.log 2>&1; log "  exit $?"
log "done"

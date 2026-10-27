#!/usr/bin/env bash
# RL stage 1: Flow-GRPO and DGPO, each on JPEG compressibility and incompressibility, run one after
# another on the single GPU with equal train-time budgets (HOURS each, default 1.5). Waits until the
# GPU is free (< 20 GB used) before starting. A run whose policy_final.safetensors exists is skipped
# and a run with a checkpoint is resumed, so the script can simply be re-run after an interruption.
#   nohup scripts/launch_rl_jpeg.sh > out/logs/launch_rl_jpeg.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
HOURS=${HOURS:-1.5}
log() { echo "$(date +%H:%M:%S) $*"; }
used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
waited=""
while [ "$(used)" -gt 20000 ]; do
  [ -z "$waited" ] && log "GPU busy ($(used) MiB used), waiting for it to be freed" && waited=1
  sleep 60
done
log "GPU free ($(used) MiB used)"
for obj in compress incompress; do for alg in dgpo grpo; do
  run=rl_${alg}_${obj}
  if [ -f "out/runs/$run/policy_final.safetensors" ]; then log "skip $run (finished)"; continue; fi
  resume=""; [ -f "out/runs/$run/ckpt_last.pt" ] && resume="--resume"
  log "start $run $resume"
  docker/run.sh exec "python -m tinydit.rl.train --alg $alg --objective $obj --run $run --hours $HOURS $resume" > "out/logs/$run.log" 2>&1
  log "exit $run ($?)"
done; done
log "all done"

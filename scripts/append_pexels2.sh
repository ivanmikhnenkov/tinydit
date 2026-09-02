#!/usr/bin/env bash
# When the pexels2 ingest has finished: stop the trainer (it checkpoints on SIGTERM), append the new
# rows to the live cache without touching the whitening stats, and resume training from ckpt_last.pt.
#   nohup scripts/append_pexels2.sh > out/logs/append_pexels2.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
log() { echo "$(date +%H:%M:%S) $*"; }
while pgrep -f "tinydit.ingest pexels2 " > /dev/null; do sleep 120; done
log "pexels2 ingest finished: $(tail -1 out/logs/ingest_pexels2.log 2>/dev/null | cut -c1-120)"
# wait until training is actually running (launch_run1.sh may still be merging)
until pgrep -f "tinydit.train --run run1" > /dev/null; do sleep 60; done
sleep 600   # let it get past compile and write at least one checkpoint interval
PID=$(pgrep -f "python -m tinydit.train --run run1" | head -1)
log "stopping trainer pid $PID (SIGTERM -> checkpoint)"; kill -TERM "$PID"
while pgrep -f "tinydit.train --run run1" > /dev/null; do sleep 10; done
log "trainer stopped; appending src_pexels2"
docker/run.sh exec 'python -m tinydit.ingest merge --cache out/cache/run1 --append src_pexels2' || { log "append failed; restarting trainer on the old cache"; }
docker/run.sh exec 'python -m tinydit.ingest check --cache out/cache/run1'
log "resuming training"
docker/run.sh exec 'python -m tinydit.train --run run1 --cache out/cache/run1 --config run1 --steps 400000 --resume out/runs/run1/ckpt_last.pt' >> out/logs/train_run1.log 2>&1
log "training process exited"

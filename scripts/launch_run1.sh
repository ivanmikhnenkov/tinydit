#!/usr/bin/env bash
# Wait until the three ingest processes have exited, merge the per-source caches, sanity-check,
# then start the pretraining run. Idempotent: skips merge if meta.json already exists.
#   nohup scripts/launch_run1.sh > out/logs/launch_run1.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
log() { echo "$(date +%H:%M:%S) $*"; }
for src in coco pexels flux; do
  while pgrep -f "tinydit.ingest $src " > /dev/null; do sleep 60; done
  log "ingest $src finished (last log line: $(tail -1 out/logs/ingest_$src.log 2>/dev/null | cut -c1-120))"
done
if [ ! -f out/cache/run1/meta.json ]; then
  log "merging"; docker/run.sh exec 'python -m tinydit.ingest merge --cache out/cache/run1 --remove-src --exclude src_pexels2' || { log "merge failed"; exit 1; }
fi
log "check"; docker/run.sh exec 'python -m tinydit.ingest check --cache out/cache/run1'
log "starting training"
docker/run.sh exec 'python -m tinydit.train --run run1 --cache out/cache/run1 --config run1 --steps 400000' > out/logs/train_run1.log 2>&1
log "training process exited"

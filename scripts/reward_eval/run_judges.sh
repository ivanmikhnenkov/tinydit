#!/usr/bin/env bash
# Run the VLM / HPSv3 scorings one after another; Qwen3.8-27B last, after its download finishes.
# Re-runnable: score_vlm.py skips (group, mode) pairs already in the score file.
#   nohup scripts/reward_eval/run_judges.sh > out/logs/run_judges.log 2>&1 &
set -uo pipefail; cd "$(dirname "$0")/../.."
log() { echo "$(date +%H:%M:%S) $*"; }
run() { m=$1; log "start $m"
  .venv/bin/python scripts/reward_eval/score_vlm.py --model $m --modes align,artifact,real,aislop --batch 16 >> out/logs/judge_$m.log 2>&1; log "  yes/no exit $?"
  .venv/bin/python scripts/reward_eval/score_vlm.py --model $m --modes rubric --groups samples --batch 16 >> out/logs/judge_$m.log 2>&1; log "  rubric exit $?"
  .venv/bin/python scripts/reward_eval/score_vlm.py --model $m --modes describe --groups samples --subset out/reward_eval/labels.json --batch 8 >> out/logs/judge_$m.log 2>&1; log "  describe exit $?"
  .venv/bin/python scripts/reward_eval/analyze.py > out/reward_eval/analysis.txt 2>&1; }
for m in qwen35_4b qwen3vl8b qwen35_9b hpsv3; do run $m; done
while ! grep -q "== qwen38_27b done" out/logs/download_27b.log 2>/dev/null; do sleep 60; done
log "27B download done"; run qwen38_27b
log "all done"

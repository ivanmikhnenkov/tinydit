#!/usr/bin/env bash
# Stage 2: wait for a free GPU, finish the realism probe if the FLUX references are in, validate the reward on hand-made hacks,
# then run DGPO with the stage-2 reward, and judge every eval round with the 9B afterwards.
#   nohup scripts/rl/launch_stage2.sh > out/logs/launch_stage2.log 2>&1 &
set -uo pipefail; cd "$(dirname "$0")/../.."
export PYTHONPATH=src
HOURS=${HOURS:-4}; RUN=${RUN:-rl2_dgpo}; EXTRA=${EXTRA:-}
log() { echo "$(date +%H:%M:%S) $*"; }
used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
w=""; while [ "$(used)" -gt 20000 ]; do [ -z "$w" ] && log "GPU busy ($(used) MiB), waiting" && w=1; sleep 60; done; log "GPU free"
if [ -f out/data/refs/flux_256.npz ] && ! .venv/bin/python -c "import torch;d=torch.load('out/models/judges/dino_probe.pt');assert not d.get('provisional')" 2>/dev/null; then
  log "training the realism probe on the full references"; .venv/bin/python scripts/rl/train_dino_probe.py > out/logs/dino_probe.log 2>&1; tail -1 out/logs/dino_probe.log; fi
if [ -z "${SKIP_CHECKS:-}" ]; then
log "reward validation on hacks"; .venv/bin/python scripts/rl/validate_reward.py > out/logs/validate_reward.log 2>&1; grep -E "ranks|->" out/logs/validate_reward.log
log "smoke test of the stage-2 loop"; rm -rf out/runs/smoke2
.venv/bin/python -m tinydit.rl.train --reward stage2 --alg dgpo --run smoke2 --prompts-per-round 2 --group 4 --minibatch 8 --rollout-chunk 8 --steps 4 --trunc 2 --eval-steps 4 --rounds 2 --no-compile --eval-every-min 0 --ckpt-every-min 0 --fid-refs 300 > out/logs/smoke2.log 2>&1 \
  || { log "SMOKE FAILED, not starting the run"; grep -E "Error|error|Traceback" out/logs/smoke2.log | tail -3; exit 1; }
log "smoke ok"; rm -rf out/runs/smoke2
fi
log "start $RUN"
resume=""; [ -f "out/runs/$RUN/ckpt_last.pt" ] && resume="--resume"
.venv/bin/python -m tinydit.rl.train --reward stage2 --alg dgpo --run "$RUN" --beta 0.001 --hours "$HOURS" --eval-every-min 10 --ckpt-every-min 20 --no-compile $EXTRA $resume > "out/logs/$RUN.log" 2>&1
log "exit $RUN ($?)"
log "judging eval rounds with the 9B"; .venv/bin/python scripts/rl/judge_evals.py --run "$RUN" --describe > "out/logs/judge_$RUN.log" 2>&1; log "judge exit $?"
log "all done"

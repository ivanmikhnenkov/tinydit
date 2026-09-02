#!/usr/bin/env bash
# Build the image once, keep one long-lived container, exec commands into it.
#   docker/run.sh build                      build image tinydit:latest
#   docker/run.sh up                         start (or restart) the container `tinydit`
#   docker/run.sh exec <cmd...>              run a command inside, in the repo dir
#   docker/run.sh shell                      interactive shell
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
VOL=${TINYDIT_VOLUME:-/home/ivan/volume}
case "${1:-}" in
  build) docker build -t tinydit:latest -f "$HERE/docker/Dockerfile" "$HERE" ;;
  up)    docker rm -f tinydit >/dev/null 2>&1 || true
         docker run -d --name tinydit --gpus all --shm-size 32g --ipc=host \
           -v "$VOL":/root/volume -w /root/volume/learning/tinydit \
           -e HF_HUB_DISABLE_XET=1 tinydit:latest sleep infinity ;;
  # commands run as the HOST user so files under out/ stay editable from the host; caches are
  # redirected into out/ because that uid has no home inside the container
  exec)  shift; docker exec --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HOME=/root/volume/learning/tinydit/out/hf_home \
           -e TORCH_HOME=/root/volume/learning/tinydit/out/models/torchhub -e TORCHINDUCTOR_CACHE_DIR=/root/volume/learning/tinydit/out/inductor_cache \
           -e PYTHONPATH=/root/volume/learning/tinydit/src -w /root/volume/learning/tinydit tinydit bash -c "cd /root/volume/learning/tinydit && $*" ;;
  shell) docker exec -it --user "$(id -u):$(id -g)" -e HOME=/tmp -e PYTHONPATH=/root/volume/learning/tinydit/src -w /root/volume/learning/tinydit tinydit bash ;;
  *) sed -n '2,7p' "$0"; exit 1 ;;
esac

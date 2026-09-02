#!/usr/bin/env bash
# Build the image once, keep one long-lived container, exec commands into it.
#   docker/run.sh build                      build image tinydit:latest
#   docker/run.sh up                         start (or restart) the container `tinydit`
#   docker/run.sh exec <cmd...>              run a command inside, in the repo dir, as the host user
#   docker/run.sh shell                      interactive shell
# The volume is mounted at the SAME path as on the host, so absolute paths, logs and the token file
# location are identical inside and outside. Commands run as the host uid so files stay editable.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
VOL=${TINYDIT_VOLUME:-/home/ivan/volume}
ENV=(-e HOME=/tmp -e "USER=$(id -un)" -e "LOGNAME=$(id -un)" -e "HF_HOME=$HERE/out/hf_home" -e "TORCH_HOME=$HERE/out/models/torchhub"
     -e "TORCHINDUCTOR_CACHE_DIR=$HERE/out/inductor_cache" -e "PYTHONPATH=$HERE/src" -e HF_HUB_DISABLE_XET=1)
case "${1:-}" in
  build) docker build -t tinydit:latest -f "$HERE/docker/Dockerfile" "$HERE" ;;
  up)    docker rm -f tinydit >/dev/null 2>&1 || true
         docker run -d --name tinydit --gpus all --shm-size 32g --ipc=host \
           -v "$VOL":"$VOL" -w "$HERE" "${ENV[@]}" tinydit:latest sleep infinity ;;
  exec)  shift; docker exec --user "$(id -u):$(id -g)" "${ENV[@]}" -w "$HERE" tinydit bash -c "$*" ;;
  shell) docker exec -it --user "$(id -u):$(id -g)" "${ENV[@]}" -w "$HERE" tinydit bash ;;
  *) sed -n '2,8p' "$0"; exit 1 ;;
esac

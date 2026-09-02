# tinydit training image: CUDA 12.8 (Blackwell sm_120 needs >= 12.8), a C compiler for torch.compile/Triton,
# Python 3.12 and the pinned requirements. The repo and all data are bind-mounted, never copied in.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:/root/.local/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev build-essential git curl ca-certificates libjpeg-turbo8 zlib1g \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && uv venv /opt/venv --python 3.12
COPY requirements.txt /tmp/requirements.txt
RUN uv pip install --python /opt/venv/bin/python -r /tmp/requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
WORKDIR /root/volume/learning/tinydit
CMD ["bash"]

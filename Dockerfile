# CUDA 12.2 / PyTorch 2.4 baseline
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Adjust as needed for your GPUs keeping in mind that e.g. CUDA 11.8 only
# supports up to capability 8.6. This is necessary because GPUs are not
# available during Docker build.
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6+PTX"
# Parallelism when building Flash Attention. If you have a ton of cores and
# DRAM, setting this higher can help with the otherwise outrageous build times.
ENV MAX_JOBS=16

# System deps for CUDA extensions & rendering utils
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      git git-lfs curl ca-certificates build-essential cmake ninja-build pkg-config \
      ffmpeg \
      libgl1 libglib2.0-0 libxext6 libxrender1 libxi6 libxxf86vm1 libxfixes3 \
      python3-dev python3-pip nano \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install uv

# Pip defaults
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
# HF cache
ENV HUGGINGFACE_HUB_CACHE=/opt/.cache/huggingface

# Create venv
ENV VENV_DIR=/opt/venv
RUN uv venv --python 3.10 "$VENV_DIR"
ENV PATH="$VENV_DIR/bin:$PATH"

# Copy repo
WORKDIR /opt/TRELLIS
COPY . /opt/TRELLIS

# Set PYTHONPATH to include the project directory
ENV PYTHONPATH=/opt/TRELLIS

# Build: install dependencies via the project's script & install optional backends
SHELL ["/bin/bash", "-lc"]
RUN bash -Eeuo pipefail setup.sh --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast --demo --platform="cuda"

# Runtime defaults
ENV SPCONV_ALGO=native
# ENV ATTN_BACKEND=xformers  # Uncomment if you drop --flash-attn

CMD ["/bin/bash"]

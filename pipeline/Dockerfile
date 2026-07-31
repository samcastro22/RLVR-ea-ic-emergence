# For running the pipeline outside Colab, on a machine with its own
# NVIDIA GPU and the NVIDIA Container Toolkit installed. This is for a local
# workstation or a cloud GPU instance instead.
#
# Build:
#   docker build -t rlvr-pipeline .
#
# Run (mount your data folder and pass your API keys as environment
# variables rather than baking them into the image):
#   docker run --gpus all \
#     -e OPENAI_API_KEY=... \
#     -e HF_TOKEN=... \
#     -e RLVR_BASE_PATH=/workspace \
#     -v /path/to/RLVR_Project:/workspace \
#     rlvr-pipeline

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/pipeline

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# vLLM is installed as a separate step, in this exact order, because
# vLLM 0.20.0 and later default to a CUDA 13.0 build, which conflicts
# with the CUDA 12.8 base image and torch build used here. Installing
# it after the base requirements, first with --no-deps to avoid its own
# dependency resolution overriding the pinned torch build, then normally
# to resolve its remaining sub-dependencies, is the sequence confirmed
# to produce a working torch 2.10.0+cu128 environment.
RUN pip3 install --no-cache-dir --force-reinstall --no-deps vllm==0.19.0 \
    && pip3 install --no-cache-dir vllm==0.19.0

COPY . .

CMD ["python3", "smoke_test.py"]

# MAGI-2 inference image — fully self-contained, no pre-clone steps needed.
FROM nvcr.io/nvidia/pytorch:25.10-py3

ARG http_proxy
ARG https_proxy
ARG no_proxy
ARG PIP_INDEX_URL=https://pypi.org/simple

ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    MAGI_COMPILE_CACHE_ROOT_DIR=/tmp/magi_compiler_cache \
    MAGI_ATTENTION_WORKSPACE_BASE=/tmp/magi_attention \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    NCCL_ALGO=^NVLS


WORKDIR /workspace

RUN apt-get -qq update && \
    DEBIAN_FRONTEND=noninteractive apt-get -qq install -y --no-install-recommends \
    ca-certificates git build-essential ninja-build ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel ninja

ARG FLASH_ATTENTION_REF=b613d9e2c8475945baff3fd68f2030af1b890acf
# flash-attention (hopper backend)
RUN git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention && \
    cd /tmp/flash-attention && git fetch --depth 1 origin ${FLASH_ATTENTION_REF} && git checkout ${FLASH_ATTENTION_REF} && \
    git submodule update --init --recursive --depth 1 --jobs 8 && \
    cd hopper && python setup.py install && \
    rm -rf /tmp/flash-attention

ARG MAGI_ATTENTION_REF=2c6413571c2cac6a80d1f85a434c6713fe0f5286
# MagiAttention
RUN git clone --depth 1 https://github.com/SandAI-org/MagiAttention.git /opt/MagiAttention && \
    cd /opt/MagiAttention && git fetch --depth 1 origin ${MAGI_ATTENTION_REF} && git checkout ${MAGI_ATTENTION_REF} && \
    git submodule update --init --recursive --jobs 8 && \
    pip install -r requirements.txt && \
    MAGI_ATTENTION_BUILD_COMPUTE_CAPABILITY=90 pip install --no-build-isolation . && \
    cd extensions && pip install --no-deps --no-build-isolation .

ARG MAGI_COMPILER_REF=5950612ddf1205f9ba9c3238a8f02a078023e15c
# MagiCompiler
RUN git clone --depth 1 https://github.com/SandAI-org/MagiCompiler.git /opt/MagiCompiler && \
    cd /opt/MagiCompiler && git fetch --depth 1 origin ${MAGI_COMPILER_REF} && git checkout ${MAGI_COMPILER_REF} && \
    pip install -r requirements.txt && \
    pip install --no-build-isolation .

# MAGI-2 Python deps (excluding torch which is in the base image)
COPY requirements.txt /tmp/magi2-requirements.txt
RUN grep -vE '^(torch|torchvision|torchaudio)([=<>]|$)' /tmp/magi2-requirements.txt \
    > /tmp/magi2-requirements-no-torch.txt && \
    pip install -r /tmp/magi2-requirements-no-torch.txt && \
    rm /tmp/magi2-requirements*.txt


# Fix Conv3D bf16 OOM/perf regression in PyTorch 2.9 + cuDNN < 9.15
# See: https://github.com/pytorch/pytorch/issues/166790
#      https://github.com/pytorch/pytorch/issues/166643
RUN pip install --no-deps nvidia-cudnn-cu13>=9.15 && \
    rm -f /usr/lib/x86_64-linux-gnu/libcudnn* && \
    cp /usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib/libcudnn* /usr/lib/x86_64-linux-gnu/ && \
    ldconfig

# Application code
COPY inference /workspace/inference
COPY configs /workspace/configs
COPY assets /workspace/assets
COPY scripts /workspace/scripts
COPY README.md LICENSE requirements.txt /workspace/

ARG BUILD_INFO=unknown
RUN printf '%s\n' "${BUILD_INFO}" > /etc/magi2-build-info

CMD ["bash"]

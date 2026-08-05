#!/usr/bin/env bash
# Generate the demo batch in assets/demo_samples.json with a single model load.
#
#   bash scripts/run_demo.sh                       # 1080p
#   RESOLUTION=540p bash scripts/run_demo.sh       # 540p preview only
#   SAMPLES=my_samples.json bash scripts/run_demo.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RESOLUTION=${RESOLUTION:-1080p}
OUTPUT_DIR=${OUTPUT_DIR:-output/demo}
# 10s is the only duration the model supports.
SECONDS_PER_VIDEO=10
SEED=${SEED:-42}

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_ALGO="^NVLS"
export OMP_NUM_THREADS=64

GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "$GPUS" -eq 0 ] && GPUS=8

SAMPLES=${SAMPLES:-assets/demo_samples.json}

echo "[demo] $(date +%T) resolution=$RESOLUTION, ${SECONDS_PER_VIDEO}s each, samples=$SAMPLES"

exec torchrun --nnodes=1 --nproc_per_node="$GPUS" --rdzv-backend=c10d \
    --rdzv-endpoint="localhost:${MASTER_PORT:-29500}" \
    inference/pipeline/entry.py \
    --resolution "$RESOLUTION" --seconds "$SECONDS_PER_VIDEO" --seed "$SEED" \
    --samples "$SAMPLES" --output "$OUTPUT_DIR"

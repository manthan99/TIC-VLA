#!/usr/bin/env bash
# Launch TIC-VLA stage-1 (VLM CoT fine-tuning) on all GPUs.
# Usage: scripts/launch_stage1_vlm.sh [extra train.py args]
set -euo pipefail
cd "$(dirname "$0")/.."

source /home/nvidia/miniconda3/etc/profile.d/conda.sh
conda activate tic-vla
source .env.training

RUN_NAME="vlm_baseline_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${TICVLA_OUTPUT_DIR}/logs/${RUN_NAME}.log"
mkdir -p "${TICVLA_OUTPUT_DIR}/logs"

echo "Logging to ${LOG_FILE}"
python -m ticvla.training.train --stage vlm --config configs/train_vlm.yaml "$@" 2>&1 | tee "${LOG_FILE}"

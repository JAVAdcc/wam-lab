#!/usr/bin/env bash
set -euo pipefail

WAM_LAB_ROOT="${WAM_LAB_ROOT:-${HOME}/workspace/code/wam-lab}"
FASTWAM_ROOT="${FASTWAM_ROOT:-${HOME}/workspace/code/FastWAM}"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/workspace/envs/fastwam/bin/fastwam-python}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-$FASTWAM_ROOT/checkpoints}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${HOME}/workspace/.libero}"
export DIFFSYNTH_DOWNLOAD_SOURCE="${DIFFSYNTH_DOWNLOAD_SOURCE:-huggingface}"

cd "$WAM_LAB_ROOT"
exec "$PYTHON_BIN" src/vla_eval/model_servers/fastwam.py \
  --config configs/model_servers/fastwam/libero_uncond_2cam224.yaml "$@"

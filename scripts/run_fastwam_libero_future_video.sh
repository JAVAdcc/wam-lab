#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-${HOME}/workspace}"
WAM_LAB_ROOT="${WAM_LAB_ROOT:-${WORKSPACE_ROOT}/code/wam-lab}"
FASTWAM_ROOT="${FASTWAM_ROOT:-${WORKSPACE_ROOT}/code/FastWAM}"
PYTHON_BIN="${PYTHON_BIN:-${WORKSPACE_ROOT}/envs/fastwam/bin/fastwam-python}"

TASK_CONFIG="${TASK_CONFIG:-libero_uncond_2cam224_1e-4}"
CHECKPOINT="${CHECKPOINT:-${FASTWAM_ROOT}/checkpoints/fastwam_release/libero_uncond_2cam224.pt}"
DATASET_STATS_PATH="${DATASET_STATS_PATH:-${FASTWAM_ROOT}/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"
TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_spatial}"
TASK_ID="${TASK_ID:-0}"
NUM_TRIALS="${NUM_TRIALS:-1}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
REPLAN_STEPS="${REPLAN_STEPS:-10}"
ACTION_HORIZON="${ACTION_HORIZON:-}"

if [[ -z "${OUTPUT_DIR:-}" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  OUTPUT_DIR="${FASTWAM_ROOT}/evaluate_results/wam_lab_future_video_${TS}"
fi

export OUTPUT_DIR
export TASK_CONFIG CHECKPOINT DATASET_STATS_PATH TASK_SUITE_NAME TASK_ID NUM_TRIALS
export GPU_ID DEVICE MIXED_PRECISION NUM_INFERENCE_STEPS REPLAN_STEPS ACTION_HORIZON
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${FASTWAM_ROOT}/checkpoints}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${WORKSPACE_ROOT}/.libero}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

mkdir -p "$OUTPUT_DIR"

cd "$FASTWAM_ROOT"
"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import torch

fastwam_root = Path.cwd()
sys.path.insert(0, str(fastwam_root))
sys.path.insert(0, str(fastwam_root / "src"))
sys.path.insert(0, str(fastwam_root / "experiments" / "libero"))

original_torch_load = torch.load


def patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return original_torch_load(*args, **kwargs)


torch.load = patched_load

overrides = [
    "task=" + os.environ["TASK_CONFIG"],
    "ckpt=" + os.environ["CHECKPOINT"],
    "model.redirect_common_files=false",
    "mixed_precision=" + os.environ["MIXED_PRECISION"],
    "EVALUATION.dataset_stats_path=" + os.environ["DATASET_STATS_PATH"],
    "EVALUATION.output_dir=" + os.environ["OUTPUT_DIR"],
    "EVALUATION.task_suite_name=" + os.environ["TASK_SUITE_NAME"],
    "EVALUATION.task_id=" + os.environ["TASK_ID"],
    "EVALUATION.num_trials=" + os.environ["NUM_TRIALS"],
    "EVALUATION.device=" + os.environ["DEVICE"],
    "EVALUATION.num_inference_steps=" + os.environ["NUM_INFERENCE_STEPS"],
    "EVALUATION.replan_steps=" + os.environ["REPLAN_STEPS"],
    "EVALUATION.visualize_future_video=true",
    "gpu_id=" + os.environ["GPU_ID"],
]
if os.environ.get("ACTION_HORIZON"):
    overrides.append("EVALUATION.action_horizon=" + os.environ["ACTION_HORIZON"])

sys.argv = ["experiments/libero/eval_libero_single.py", *overrides]
runpy.run_path("experiments/libero/eval_libero_single.py", run_name="__main__")
PY

REPORT_PATH="$("$PYTHON_BIN" "$WAM_LAB_ROOT/scripts/make_fastwam_future_video_report.py" --output-dir "$OUTPUT_DIR")"

echo "Future-video output: $OUTPUT_DIR"
echo "Future-video report: $REPORT_PATH"
echo "Artifacts:"
find "$OUTPUT_DIR" -maxdepth 4 -type f \( -name '*.mp4' -o -name '*.json' -o -name '*.html' \) | sort

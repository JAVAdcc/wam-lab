#!/usr/bin/env bash
set -euo pipefail

WAM_LAB_ROOT="${WAM_LAB_ROOT:-${HOME}/workspace/code/wam-lab}"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/workspace/envs/fastwam/bin/fastwam-python}"
CONFIG="${CONFIG:-configs/benchmarks/libero/fastwam_smoke.yaml}"

cd "$WAM_LAB_ROOT"
exec "$PYTHON_BIN" -m vla_eval.cli.main run --config "$CONFIG" --no-docker "$@"

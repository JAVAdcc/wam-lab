#!/usr/bin/env bash
set -euo pipefail

WAM_LAB_ROOT="${WAM_LAB_ROOT:-${HOME}/workspace/code/wam-lab}"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/workspace/envs/fastwam/bin/fastwam-python}"
CONFIG="${CONFIG:-configs/benchmarks/libero/fastwam_live_sim.yaml}"
SERVER_PORT="${SERVER_PORT:-8000}"
LIVE_PORT="${LIVE_PORT:-8765}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
WAIT_SERVER_SEC="${WAIT_SERVER_SEC:-180}"
PREVIEW_DELAY_SEC="${PREVIEW_DELAY_SEC:-8}"

if [[ -z "${OUTPUT_DIR:-}" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  OUTPUT_DIR="${WAM_LAB_ROOT}/results/fastwam_libero_live_sim_${TS}"
fi

RUN_USER="${USER:-$(id -un)}"
LOG_DIR="${LOG_DIR:-/tmp/${RUN_USER}/wam_lab_fastwam_live_sim_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

cd "$WAM_LAB_ROOT"

server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if ss -ltn 2>/dev/null | grep -q ":${SERVER_PORT} "; then
  echo "Port ${SERVER_PORT} is already in use; stop the existing model server or set SERVER_PORT."
  exit 1
fi
if ss -ltn 2>/dev/null | grep -q ":${LIVE_PORT} "; then
  echo "Port ${LIVE_PORT} is already in use; stop the existing live viewer or set LIVE_PORT and update the config."
  exit 1
fi

echo "Starting FastWAM server on 127.0.0.1:${SERVER_PORT}; log=${LOG_DIR}/server.log"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" scripts/run_fastwam_server.sh \
  --host 127.0.0.1 \
  --port "$SERVER_PORT" \
  >"${LOG_DIR}/server.log" 2>&1 &
server_pid="$!"

deadline=$((SECONDS + WAIT_SERVER_SEC))
while true; do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "FastWAM server exited early. Last log lines:"
    tail -80 "${LOG_DIR}/server.log" || true
    exit 1
  fi
  if grep -q "FastWAM ready" "${LOG_DIR}/server.log"; then
    break
  fi
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for FastWAM server. Last log lines:"
    tail -80 "${LOG_DIR}/server.log" || true
    exit 1
  fi
  sleep 2
done

echo "FastWAM server ready."
echo "Forward this from your local machine if it is not already forwarded:"
echo "  ssh -N -L 127.0.0.1:7860:127.0.0.1:${LIVE_PORT} -p 26 yiming@47.116.73.163"
echo "Then open:"
echo "  http://127.0.0.1:7860/"
echo "Starting rollout in ${PREVIEW_DELAY_SEC}s; benchmark log=${LOG_DIR}/run.log"
sleep "$PREVIEW_DELAY_SEC"

"$PYTHON_BIN" -m vla_eval.cli.main run \
  --config "$CONFIG" \
  --no-docker \
  --server-url "ws://127.0.0.1:${SERVER_PORT}" \
  --output-dir "$OUTPUT_DIR" \
  "$@" 2>&1 | tee "${LOG_DIR}/run.log"

echo "Live sim output: ${OUTPUT_DIR}"
echo "Logs: ${LOG_DIR}"

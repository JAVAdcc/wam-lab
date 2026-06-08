#!/usr/bin/env bash
set -euo pipefail

WAM_LAB_ROOT="${WAM_LAB_ROOT:-${HOME}/workspace/code/wam-lab}"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/workspace/envs/fastwam/bin/fastwam-python}"
CONFIG="${CONFIG:-configs/benchmarks/libero/fastwam_native_viewer.yaml}"
SERVER_PORT="${SERVER_PORT:-8000}"
DISPLAY_NUM="${DISPLAY_NUM:-99}"
NOVNC_PORT="${NOVNC_PORT:-6080}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
WAIT_SERVER_SEC="${WAIT_SERVER_SEC:-180}"
PREVIEW_DELAY_SEC="${PREVIEW_DELAY_SEC:-20}"

if [[ -z "${OUTPUT_DIR:-}" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  OUTPUT_DIR="${WAM_LAB_ROOT}/results/fastwam_libero_native_viewer_${TS}"
fi

RUN_USER="${USER:-$(id -un)}"
LOG_DIR="${LOG_DIR:-/tmp/${RUN_USER}/wam_lab_fastwam_native_viewer_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

print_novnc_forward_hint() {
  local ssh_port="${SSH_PORT_HINT:-}"
  local ssh_host="${SSH_HOST_HINT:-47.116.73.163}"
  local ssh_user="${SSH_USER_HINT:-${USER:-yiming}}"
  local local_port="${LOCAL_NOVNC_PORT:-7861}"

  if [[ -z "$ssh_port" && -n "${SSH_CONNECTION:-}" ]]; then
    local client_ip client_port server_ip server_port
    read -r client_ip client_port server_ip server_port <<< "${SSH_CONNECTION}"
    ssh_port="$server_port"
  fi
  ssh_port="${ssh_port:-22}"

  echo "Forward noVNC from local:"
  echo "  ssh -N -L 127.0.0.1:${local_port}:127.0.0.1:${NOVNC_PORT} -p ${ssh_port} ${ssh_user}@${ssh_host}"
  echo "Open:"
  echo "  http://127.0.0.1:${local_port}/vnc.html?host=127.0.0.1&port=${local_port}&autoconnect=1&resize=scale"
}

tcp_port_open() {
  local port="$1"
  "$PYTHON_BIN" - "$port" <<'PY' >/dev/null 2>&1
import socket
import sys

sock = socket.socket()
sock.settimeout(0.5)
try:
    sock.connect(("127.0.0.1", int(sys.argv[1])))
except OSError:
    sys.exit(1)
finally:
    sock.close()
PY
}

cd "$WAM_LAB_ROOT"

if ! tcp_port_open "$NOVNC_PORT"; then
  scripts/start_mujoco_viewer_display.sh start
fi

server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if tcp_port_open "$SERVER_PORT"; then
  echo "Port ${SERVER_PORT} is already in use; stop the existing model server or set SERVER_PORT."
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
print_novnc_forward_hint
echo "Starting native-viewer rollout in ${PREVIEW_DELAY_SEC}s; benchmark log=${LOG_DIR}/run.log"
echo "The benchmark will also hold after reset so the MuJoCo viewer is visible before actions start."
sleep "$PREVIEW_DELAY_SEC"

unset PYOPENGL_PLATFORM
unset EGL_PLATFORM
DISPLAY=":${DISPLAY_NUM}" MUJOCO_GL=glfw "$PYTHON_BIN" -m vla_eval.cli.main run \
  --config "$CONFIG" \
  --no-docker \
  --server-url "ws://127.0.0.1:${SERVER_PORT}" \
  --output-dir "$OUTPUT_DIR" \
  "$@" 2>&1 | tee "${LOG_DIR}/run.log"

echo "Native-viewer output: ${OUTPUT_DIR}"
echo "Logs: ${LOG_DIR}"

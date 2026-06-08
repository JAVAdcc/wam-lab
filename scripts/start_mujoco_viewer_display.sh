#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-start}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${HOME}/workspace}"
STACK_ROOT="${STACK_ROOT:-${WORKSPACE_ROOT}/tools/mujoco-viewer-stack}"
PYTHON_BIN="${PYTHON_BIN:-${WORKSPACE_ROOT}/envs/fastwam/bin/fastwam-python}"
DISPLAY_NUM="${DISPLAY_NUM:-99}"
VNC_PORT="${VNC_PORT:-5901}"
NOVNC_PORT="${NOVNC_PORT:-6080}"
SCREEN_GEOMETRY="${SCREEN_GEOMETRY:-1280x900x24}"
LOG_DIR="${LOG_DIR:-${WORKSPACE_ROOT}/logs/mujoco_viewer_display_${DISPLAY_NUM}_${NOVNC_PORT}}"
PID_FILE="${PID_FILE:-${LOG_DIR}/pids}"
RUN_USER="${USER:-$(id -un)}"

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

  echo "Forward from local:"
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

LIB_PATH="${STACK_ROOT}/lib/x86_64-linux-gnu:${STACK_ROOT}/usr/lib/x86_64-linux-gnu:${STACK_ROOT}/usr/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$LIB_PATH"
export PATH="${STACK_ROOT}/usr/bin:${PATH}"

stop_display() {
  if [[ -f "$PID_FILE" ]]; then
    while read -r pid; do
      [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
    done < "$PID_FILE"
    rm -f "$PID_FILE"
  fi
  # pidfiles can go stale if the display was first brought up manually while
  # probing a host. Keep stop idempotent by also matching the display/ports.
  pkill -u "$RUN_USER" -f "${STACK_ROOT}/usr/bin/proot .*${STACK_ROOT}/usr/bin/Xvfb :${DISPLAY_NUM}" 2>/dev/null || true
  pkill -u "$RUN_USER" -f "${STACK_ROOT}/usr/bin/Xvfb :${DISPLAY_NUM}" 2>/dev/null || true
  pkill -u "$RUN_USER" -f "x11vnc .* -display :${DISPLAY_NUM} .* -rfbport ${VNC_PORT}" 2>/dev/null || true
  pkill -u "$RUN_USER" -f "websockify .*127[.]0[.]0[.]1:${NOVNC_PORT} .*127[.]0[.]0[.]1:${VNC_PORT}" 2>/dev/null || true
  rm -f "/tmp/.X${DISPLAY_NUM}-lock"
}

status_display() {
  echo "DISPLAY=:${DISPLAY_NUM}"
  echo "noVNC=http://127.0.0.1:${NOVNC_PORT}/vnc.html"
  if tcp_port_open "$VNC_PORT"; then
    echo "vnc_port_open=yes"
  else
    echo "vnc_port_open=no"
  fi
  if tcp_port_open "$NOVNC_PORT"; then
    echo "novnc_port_open=yes"
  else
    echo "novnc_port_open=no"
  fi
  if [[ -f "$PID_FILE" ]]; then
    echo "pids=$(tr '\n' ' ' < "$PID_FILE")"
  else
    echo "pids="
  fi
  if command -v ss >/dev/null 2>&1; then
    ss -ltnp 2>/dev/null | grep -E ":(${VNC_PORT}|${NOVNC_PORT})" || true
  fi
}

case "$ACTION" in
  stop)
    stop_display
    exit 0
    ;;
  status)
    status_display
    exit 0
    ;;
  start)
    ;;
  *)
    echo "Usage: $0 [start|stop|status]"
    exit 2
    ;;
esac

for bin in "$STACK_ROOT/usr/bin/Xvfb" "$STACK_ROOT/usr/bin/x11vnc" "$STACK_ROOT/usr/bin/fluxbox" "$STACK_ROOT/usr/bin/proot"; do
  [[ -x "$bin" ]] || {
    echo "Missing $bin. Run scripts/setup_mujoco_viewer_stack.sh first."
    exit 1
  }
done

if tcp_port_open "$VNC_PORT"; then
  echo "VNC port ${VNC_PORT} is already in use."
  status_display
  exit 1
fi
if tcp_port_open "$NOVNC_PORT"; then
  echo "noVNC port ${NOVNC_PORT} is already in use."
  status_display
  exit 1
fi

mkdir -p "$LOG_DIR"
rm -f "$PID_FILE" "/tmp/.X${DISPLAY_NUM}-lock"

"$STACK_ROOT/usr/bin/proot" -0 \
  -b "$STACK_ROOT/usr/bin/xkbcomp:/usr/bin/xkbcomp" \
  -b "$STACK_ROOT/usr/share/X11/xkb:/usr/share/X11/xkb" \
  "$STACK_ROOT/usr/bin/Xvfb" ":${DISPLAY_NUM}" \
    -screen 0 "$SCREEN_GEOMETRY" -ac -noreset \
    >"${LOG_DIR}/xvfb.log" 2>&1 &
echo "$!" >> "$PID_FILE"

deadline=$((SECONDS + 20))
until DISPLAY=":${DISPLAY_NUM}" xdpyinfo >/dev/null 2>&1; do
  if (( SECONDS >= deadline )); then
    echo "Xvfb did not become ready. Log:"
    sed -n '1,120p' "${LOG_DIR}/xvfb.log" || true
    stop_display
    exit 1
  fi
  sleep 1
done

DISPLAY=":${DISPLAY_NUM}" fluxbox >"${LOG_DIR}/fluxbox.log" 2>&1 &
echo "$!" >> "$PID_FILE"

DISPLAY=":${DISPLAY_NUM}" x11vnc \
  -display ":${DISPLAY_NUM}" -localhost -forever -shared -nopw -rfbport "$VNC_PORT" \
  >"${LOG_DIR}/x11vnc.log" 2>&1 &
echo "$!" >> "$PID_FILE"

sleep 1
"$PYTHON_BIN" -m websockify --web "$STACK_ROOT/usr/share/novnc" \
  "127.0.0.1:${NOVNC_PORT}" "127.0.0.1:${VNC_PORT}" \
  >"${LOG_DIR}/websockify.log" 2>&1 &
echo "$!" >> "$PID_FILE"

sleep 1
curl -fsS "http://127.0.0.1:${NOVNC_PORT}/vnc.html" >/dev/null

echo "MuJoCo viewer display ready."
echo "Remote DISPLAY=:${DISPLAY_NUM}"
print_novnc_forward_hint
echo "Logs: ${LOG_DIR}"

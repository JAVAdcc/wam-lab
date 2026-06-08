#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-${HOME}/workspace}"
STACK_ROOT="${STACK_ROOT:-${WORKSPACE_ROOT}/tools/mujoco-viewer-stack}"
APT_DIR="${APT_DIR:-${WORKSPACE_ROOT}/scratch/mujoco-viewer-apt}"
PYTHON_BIN="${PYTHON_BIN:-${WORKSPACE_ROOT}/envs/fastwam/bin/fastwam-python}"

PACKAGES=(
  xvfb
  x11vnc
  novnc
  fluxbox
  websockify
  x11-utils
  x11-xkb-utils
  xkb-data
  proot
  menu
  libfribidi0
  libfontenc1
  libxkbfile1
  libtalloc2
  liblzo2-2
  libxfont2
  libvncserver1
  libvncclient1
  libxdamage1
  libavahi-common3
  libavahi-client3
  libxpm4
  libxrandr2
  libxinerama1
  libxrender1
  libxft2
  libxext6
  libx11-6
  libxxf86dga1
  libfreetype6
  libfontconfig1
)

add_first_available_package() {
  local candidate
  local pkg
  for pkg in "$@"; do
    candidate="$(apt-cache policy "$pkg" 2>/dev/null | awk '/Candidate:/ { print $2; exit }')"
    if [[ -n "$candidate" && "$candidate" != "(none)" ]]; then
      PACKAGES+=("$pkg")
      return 0
    fi
  done
  echo "No apt candidate found for any of: $*"
  return 1
}

add_first_available_package libimlib2 libimlib2t64

mkdir -p "$STACK_ROOT" "$APT_DIR"

cd "$APT_DIR"
apt-get download "${PACKAGES[@]}"
for deb in ./*.deb; do
  dpkg -x "$deb" "$STACK_ROOT"
done

for package in websockify vncdotool; do
  "$PYTHON_BIN" -m pip show "$package" >/dev/null 2>&1 || "$PYTHON_BIN" -m pip install "$package"
done

LIB_PATH="${STACK_ROOT}/lib/x86_64-linux-gnu:${STACK_ROOT}/usr/lib/x86_64-linux-gnu:${STACK_ROOT}/usr/lib:${LD_LIBRARY_PATH:-}"
missing=0
for bin in "$STACK_ROOT/usr/bin/Xvfb" "$STACK_ROOT/usr/bin/x11vnc" "$STACK_ROOT/usr/bin/fluxbox" "$STACK_ROOT/usr/bin/proot" "$STACK_ROOT/usr/bin/xdpyinfo"; do
  if ! LD_LIBRARY_PATH="$LIB_PATH" ldd "$bin" | grep -q "not found"; then
    echo "deps-ok $bin"
  else
    echo "missing runtime dependency for $bin"
    LD_LIBRARY_PATH="$LIB_PATH" ldd "$bin" | grep "not found" || true
    missing=1
  fi
done

if [[ "$missing" -ne 0 ]]; then
  exit 1
fi

echo "MuJoCo viewer stack ready: $STACK_ROOT"

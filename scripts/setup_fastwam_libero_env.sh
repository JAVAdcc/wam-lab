#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-$HOME/workspace}"
WAM_LAB_ROOT="${WAM_LAB_ROOT:-$WORKSPACE_ROOT/code/wam-lab}"
FASTWAM_ROOT="${FASTWAM_ROOT:-$WORKSPACE_ROOT/code/FastWAM}"
LIBERO_ROOT="${LIBERO_ROOT:-$WORKSPACE_ROOT/code/LIBERO}"
LIBERO_REPO="${LIBERO_REPO:-https://github.com/Lifelong-Robot-Learning/LIBERO.git}"
LIBERO_COMMIT="${LIBERO_COMMIT:-8f1084e3132a39270c3a13ebe37270a43ece2a01}"
PYTHON_BIN="${PYTHON_BIN:-$WORKSPACE_ROOT/envs/fastwam/bin/fastwam-python}"
CONSTRAINTS_FILE="${CONSTRAINTS_FILE:-$WORKSPACE_ROOT/scratch/fastwam-libero-constraints.txt}"

mkdir -p "$WORKSPACE_ROOT/code" "$WORKSPACE_ROOT/data/libero/datasets" "$WORKSPACE_ROOT/scratch"

if [[ ! -d "$LIBERO_ROOT/.git" ]]; then
  git clone "$LIBERO_REPO" "$LIBERO_ROOT"
fi
if ! git -C "$LIBERO_ROOT" cat-file -e "${LIBERO_COMMIT}^{commit}" 2>/dev/null; then
  git -C "$LIBERO_ROOT" fetch --depth 1 origin "$LIBERO_COMMIT"
fi
git -C "$LIBERO_ROOT" checkout --detach "$LIBERO_COMMIT"
printf "LIBERO_COMMIT %s\n" "$(git -C "$LIBERO_ROOT" rev-parse HEAD)"

printf "%s\n" \
  "numpy==1.26.4" \
  "transformers==4.49.0" \
  "hydra-core==1.3.2" \
  "omegaconf==2.3.0" \
  > "$CONSTRAINTS_FILE"

"$PYTHON_BIN" -m pip install -e "$WAM_LAB_ROOT"
"$PYTHON_BIN" -m pip install --constraint "$CONSTRAINTS_FILE" \
  "numpy==1.26.4" \
  "mujoco==3.3.2" \
  "opencv-python==4.6.0.66" \
  "robosuite==1.4.0" \
  "bddl==1.0.1" \
  "gym==0.25.2" \
  "easydict==1.9" \
  "cloudpickle==2.1.0" \
  "future==0.18.2" \
  "matplotlib==3.5.3" \
  h5py
"$PYTHON_BIN" -m pip install --constraint "$CONSTRAINTS_FILE" -e "$LIBERO_ROOT"

"$PYTHON_BIN" - <<PY_INNER
from pathlib import Path
import site
import yaml

workspace = Path("$WORKSPACE_ROOT")
libero_root = Path("$LIBERO_ROOT")
site_packages = next(Path(p) for p in site.getsitepackages() if p.endswith("site-packages"))
(site_packages / "libero_source.pth").write_text(str(libero_root) + "\n", encoding="utf-8")

config_root = workspace / ".libero"
benchmark_root = libero_root / "libero" / "libero"
dataset_root = workspace / "data" / "libero" / "datasets"
config_root.mkdir(parents=True, exist_ok=True)
dataset_root.mkdir(parents=True, exist_ok=True)
(config_root / "config.yaml").write_text(
    yaml.safe_dump({
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(dataset_root),
        "assets": str(benchmark_root / "assets"),
    }),
    encoding="utf-8",
)

egl_file = workspace / ".egl" / "nvidia_icd.json"
egl_file.parent.mkdir(parents=True, exist_ok=True)
egl_file.write_text(
    '{"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}\n',
    encoding="utf-8",
)
print("LIBERO_CONFIG_PATH", config_root)
print("EGL_VENDOR_FILE", egl_file)
PY_INNER

"$PYTHON_BIN" -m pip check

#!/usr/bin/env bash
set -euo pipefail

FASTWAM_ROOT="${FASTWAM_ROOT:-${HOME}/workspace/code/FastWAM}"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/workspace/envs/fastwam/bin/fastwam-python}"
FASTWAM_HF_REPO="${FASTWAM_HF_REPO:-yuanty/fastwam}"
FASTWAM_HF_REVISION="${FASTWAM_HF_REVISION:-139eebb6d90cdd9bdbbe465f72c6edc9ad5a518a}"
FASTWAM_RELEASE_DIR="${FASTWAM_RELEASE_DIR:-./checkpoints/fastwam_release}"

cd "$FASTWAM_ROOT"
"$PYTHON_BIN" -m huggingface_hub.commands.huggingface_cli download \
  --revision "$FASTWAM_HF_REVISION" \
  --local-dir "$FASTWAM_RELEASE_DIR" \
  "$FASTWAM_HF_REPO" \
  libero_uncond_2cam224.pt \
  libero_uncond_2cam224_dataset_stats.json

sha256sum \
  "$FASTWAM_RELEASE_DIR/libero_uncond_2cam224.pt" \
  "$FASTWAM_RELEASE_DIR/libero_uncond_2cam224_dataset_stats.json"

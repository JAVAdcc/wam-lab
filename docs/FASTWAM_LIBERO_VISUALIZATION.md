# FastWAM + LIBERO Visualization

This document covers the first-stage visualization path for the released
FastWAM LIBERO checkpoint in WAM-Lab.

## Live Simulation View

Use this when you want to watch the same LIBERO simulation that FastWAM is
controlling during inference.

On your local machine:

```bash
ssh -N -L 127.0.0.1:7860:127.0.0.1:8765 -p 26 yiming@47.116.73.163
```

On the port-26 host:

```bash
cd ~/workspace/code/wam-lab
CUDA_VISIBLE_DEVICES=0 scripts/run_fastwam_libero_live_sim.sh
```

Then open:

```text
http://127.0.0.1:7860/
```

Notes:

- The browser stream is the actual rendered LIBERO `agentview` frame from the
  running robosuite/MuJoCo environment, published through
  `EpisodeRecorder.record_video`.
- This is not the sparse point-cloud/3D-trace path.
- The live stream exists only while the benchmark process is running. The
  latest episode mp4 remains under the configured result directory.
- The live config uses paced realtime mode at 5 Hz for human inspection. The
  normal action-only sweep remains sync mode for fast evaluation.
- A fully interactive native MuJoCo viewer with draggable camera is a different
  path. It requires a remote graphical session such as X11/noVNC/VirtualGL and
  a LIBERO environment created with `has_renderer=True`. The current port-26
  host did not have Xvfb/noVNC/x11vnc/vglrun available during this validation.

## Future Video Comparison

Use this when you want FastWAM's predicted future video compared against the
actual rollout observations.

```bash
cd ~/workspace/code/wam-lab
CUDA_VISIBLE_DEVICES=0 scripts/run_fastwam_libero_future_video.sh
```

The script wraps FastWAM's official `experiments/libero/eval_libero_single.py`
with:

- `EVALUATION.visualize_future_video=true`
- `model.redirect_common_files=false`
- a local `torch.load(weights_only=False)` compatibility patch for LIBERO init
  states under PyTorch 2.6+

Outputs:

- rollout mp4 under `.../videos/`
- per-replan and `replan=all` gt/pred comparison mp4s under
  `.../predicted_videos/`
- task result JSON
- `future_video_report.html`

In FastWAM's gt/pred clips, the top row is prediction and the bottom row is
ground truth.

## Broader LIBERO-Spatial Sweep

Run every LIBERO-Spatial task once through the WAM-Lab FastWAM bridge:

```bash
cd ~/workspace/code/wam-lab
CUDA_VISIBLE_DEVICES=0 scripts/run_fastwam_server.sh --host 127.0.0.1 --port 8000
CONFIG=configs/benchmarks/libero/fastwam_spatial_1trial.yaml \
  scripts/run_fastwam_libero_smoke.sh --server-url ws://127.0.0.1:8000
```

This is broader than the smoke test, but it is not the full official LIBERO
benchmark. A full official run would need many initial states per task and
additional LIBERO suites.

# FastWAM + LIBERO Visualization

This document covers the first-stage visualization path for the released
FastWAM LIBERO checkpoint in WAM-Lab.

## Live Simulation View

Use this when you want to watch the same LIBERO simulation that FastWAM is
controlling during inference.

On your local machine:

```bash
ssh -N -L 127.0.0.1:7860:127.0.0.1:8765 -p 22 yiming@47.116.73.163
```

On the target host:

```bash
cd ~/workspace/code/wam-lab
CUDA_VISIBLE_DEVICES=7 SSH_PORT_HINT=22 scripts/run_fastwam_libero_live_sim.sh
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
- The helper script uses a Python socket probe for localhost port checks, so it
  works on minimal containers that do not provide `ss`.
- A fully interactive native MuJoCo viewer with draggable camera is a different
  path; see the next section.

## Native MuJoCo Viewer

Use this when you need MuJoCo's interactive viewer, including mouse camera
interaction through VNC/noVNC. The default config uses
`mujoco.viewer.launch_passive(model, data)` and calls `viewer.sync()` during the
rollout. This is different from robosuite's OpenCV `offscreen render` window,
which only displays a fixed camera image and does not handle mouse camera
dragging.

Prepare the noVNC/X11 stack once on the target host:

```bash
cd ~/workspace/code/wam-lab
scripts/setup_mujoco_viewer_stack.sh
```

Start the remote display:

```bash
cd ~/workspace/code/wam-lab
scripts/start_mujoco_viewer_display.sh start
```

Forward noVNC from your local machine:

```bash
ssh -N -L 127.0.0.1:7861:127.0.0.1:6080 -p 22 yiming@47.116.73.163
```

Open:

```text
http://127.0.0.1:7861/vnc.html?host=127.0.0.1&port=7861&autoconnect=1&resize=scale
```

Run FastWAM + LIBERO with the native viewer enabled:

```bash
cd ~/workspace/code/wam-lab
CUDA_VISIBLE_DEVICES=7 SSH_PORT_HINT=22 scripts/run_fastwam_libero_native_viewer.sh
```

Notes:

- The benchmark config is `configs/benchmarks/libero/fastwam_native_viewer.yaml`.
- `LIBEROBenchmark(native_renderer=true, native_viewer_backend="mujoco_passive")`
  creates LIBERO with offscreen camera observations for the policy, then opens a
  separate MuJoCo passive viewer on the same `MjModel` / `MjData`.
- `native_viewer_backend="opencv"` is still available as a fixed-camera fallback,
  but it should not be used when you need draggable camera interaction.
- Native viewer rendering is strict by default: if the onscreen render call
  fails, the benchmark fails instead of reporting policy success with a broken
  viewer.
- Native GLFW and headless EGL LIBERO modes are intentionally not mixed inside
  one Python process. Run native-viewer and headless/offscreen sweeps as
  separate commands.
- The benchmark process runs with `DISPLAY=:99` and `MUJOCO_GL=glfw`.
- The helper scripts infer the SSH port from `SSH_CONNECTION` when possible.
  Set `SSH_PORT_HINT`, `SSH_HOST_HINT`, `SSH_USER_HINT`, or
  `LOCAL_NOVNC_PORT` when the printed noVNC forwarding command needs to match a
  different tunnel or local port.
- The default native config pauses for 30 seconds after the first post-reset
  onscreen render and 10 seconds at the terminal scene. This is deliberate: it
  gives you time to focus the noVNC window and drag/zoom the MuJoCo camera
  before and after the policy rollout. Adjust
  `native_viewer_reset_hold_sec` / `native_viewer_end_hold_sec` in the config
  for longer manual inspection.
- The display stack is installed under `~/workspace/tools/mujoco-viewer-stack`;
  no sudo is required. The Ubuntu Xvfb package expects `/usr/bin/xkbcomp`, so
  `start_mujoco_viewer_display.sh` uses `proot` to expose the workspace copy of
  `xkbcomp` and XKB data at the paths Xvfb expects.
- Stop the display with:

```bash
cd ~/workspace/code/wam-lab
scripts/start_mujoco_viewer_display.sh stop
```

## Future Video Comparison

Use this when you want FastWAM's predicted future video compared against the
actual rollout observations.

```bash
cd ~/workspace/code/wam-lab
GPU_ID=7 scripts/run_fastwam_libero_future_video.sh
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

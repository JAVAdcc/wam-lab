# 2026-06-07 FastWAM LIBERO Live/Future Validation

## Objective

Validate FastWAM + LIBERO beyond the first smoke test, add a direct live
simulation viewing path, make future-video comparison reproducible, and expose
LIBERO through a native MuJoCo/robosuite viewer for interactive inspection.

## Assumptions

- Active host: port 26, `~/workspace/code/wam-lab`.
- Active GPU: GPU0 only.
- FastWAM checkpoint:
  `~/workspace/code/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224.pt`.
- LIBERO suite focus for the first bridge sweep: `libero_spatial`.

## Implemented

- Added `vla_eval.live_video`, a process-local HTTP/MJPEG server for live
  rendered frames.
- Added recorder-level `recording.live_video` and `recording.live_video_config`
  support.
- Added `configs/benchmarks/libero/fastwam_live_sim.yaml`.
- Added `scripts/run_fastwam_libero_live_sim.sh`.
- Added `scripts/run_fastwam_libero_future_video.sh`.
- Added `scripts/make_fastwam_future_video_report.py`.
- Added `configs/benchmarks/libero/fastwam_spatial_1trial.yaml`.
- Added recorder unit coverage for live-video sink without mp4 recording.
- Added `LIBEROBenchmark(native_renderer=true)` support using MuJoCo passive
  viewer sync. The default `native_viewer_backend="mujoco_passive"` opens
  `mujoco.viewer.launch_passive(model, data)` on the same LIBERO sim.
- Kept robosuite/OpenCV onscreen rendering as
  `native_viewer_backend="opencv"` for fixed-camera debugging only.
- Added a strict native render check so viewer render failures fail the
  diagnostic benchmark instead of being hidden behind policy success.
- Added an in-process GL backend guard: LIBERO native GLFW and headless EGL
  modes must run in separate Python processes.
- Added a workspace-local native viewer display stack:
  `scripts/setup_mujoco_viewer_stack.sh`,
  `scripts/start_mujoco_viewer_display.sh`, and
  `scripts/run_fastwam_libero_native_viewer.sh`.
- Added `configs/benchmarks/libero/fastwam_native_viewer.yaml` with post-reset
  and terminal holds for manual camera inspection.

## Validations

- `python -m compileall -q src tests scripts/make_fastwam_future_video_report.py`
  passed on port 26.
- Targeted unit test passed:
  `tests/test_recording_sqlite.py::test_episode_recorder_live_video_can_run_without_mp4`.
- Config validation passed after additions: `160/160 configs valid`.
- Live sim realtime run:
  - Output:
    `~/workspace/code/wam-lab/results/fastwam_libero_live_sim_20260607_152923`
  - Result: `1/1 success`, 71 steps, 14.63s, realtime mode at 5 Hz.
  - Live HTTP status was observed through the local tunnel with frame ids
    increasing from 43 to 66 and a final success frame 72.
  - `latest.jpg` was a valid 256x256 JPEG.
- Future-video run:
  - Output:
    `~/workspace/code/FastWAM/evaluate_results/wam_lab_future_video_20260607_153117`
  - Result: `1/1 success`, PSNR mean `26.681755913479872`.
  - Generated rollout mp4, per-replan gt/pred clips, `replan=all` gt/pred clip,
    result JSON, and `future_video_report.html`.
  - `replan=all` gt/pred clip verified as 448x448, 21 frames, nonblank.
- LIBERO-Spatial 1-trial sweep:
  - Output:
    `~/workspace/code/wam-lab/results/fastwam_libero_spatial_1trial_20260607_153500`
  - Result: 10/10 tasks success, 10 mp4s, 10 jsonl files, aggregate written.
- Native MuJoCo viewer run:
  - Output:
    `~/workspace/code/wam-lab/results/fastwam_libero_native_viewer_20260607_162649`
  - Result: `1/1 success`, 86 steps. The 10s terminal hold intentionally
    appears as a long `env.step` in timing logs.
  - VNC screenshot captured during the post-reset hold:
    `/tmp/yiming/wam_lab_fastwam_native_viewer_20260607_162649/native_viewer_post_reset_strict.png`.
  - Local copy:
    `/Users/javadcc/code/ssh/H200/artifacts/port26_fastwam_native_viewer/native_viewer_post_reset_strict.png`.
  - This run used robosuite's OpenCV `offscreen render` window. It showed the
    full LIBERO tabletop scene, but did not provide draggable camera
    interaction. The interactive path was subsequently changed to
    `mujoco.viewer.launch_passive`.
- MuJoCo passive viewer run:
  - Output:
    `~/workspace/code/wam-lab/results/fastwam_libero_native_viewer_20260607_164853`
  - Result: `1/1 success`, 71 steps.
  - VNC screenshot captured during the post-reset hold:
    `/tmp/yiming/wam_lab_fastwam_native_viewer_20260607_164853/passive_before_drag.png`.
  - Local copy:
    `/Users/javadcc/code/ssh/H200/artifacts/port26_fastwam_native_viewer/passive/passive_before_drag.png`.
  - The screenshot shows the MuJoCo passive viewer window titled
    `MuJoCo : base` with MuJoCo's left/right UI panels, confirming this is no
    longer the fixed-camera OpenCV `offscreen render` window.
- Native viewer display stack:
  - `scripts/setup_mujoco_viewer_stack.sh` validated Xvfb, x11vnc, fluxbox, and
    proot runtime dependencies under `~/workspace/tools/mujoco-viewer-stack`.
  - `scripts/start_mujoco_viewer_display.sh stop/start/status` verified clean
    service state: no stale Xvfb/proot processes after stop and expected
    `5901`/`6080` listeners after start.
- Config validation after native viewer additions: `161/161 configs valid`.

## Known Debt

- Browser live view streams the fixed `agentview` RGB camera. Use the native
  viewer path when draggable MuJoCo camera interaction is needed.
- The native viewer path uses Xvfb+x11vnc+noVNC through workspace-local
  packages and `proot`. This is suitable for debugging and visual inspection,
  but not optimized for benchmark throughput or GPU-accelerated desktop
  rendering.
- Automated screenshot capture was verified for both the old OpenCV window and
  the MuJoCo passive viewer. The old window was not interactive. The active
  native viewer implementation is now the MuJoCo passive viewer path; validate
  camera dragging through noVNC before relying on it for manual inspection.
- Future-video integration is currently a WAM-Lab wrapper around FastWAM's
  official eval script, not yet surfaced through the WAM-Lab policy server API.
- The spatial sweep is not the full official LIBERO evaluation protocol.

## Resource Notes

- FastWAM server/model load peaked around 25 GB on GPU0 during these runs.
- GPU0 returned to 0 MiB after each managed run.

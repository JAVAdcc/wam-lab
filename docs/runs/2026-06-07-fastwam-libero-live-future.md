# 2026-06-07 FastWAM LIBERO Live/Future Validation

## Objective

Validate FastWAM + LIBERO beyond the first smoke test, add a direct live
simulation viewing path, and make future-video comparison reproducible.

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

## Known Debt

- Browser live view streams the fixed `agentview` RGB camera. It is direct
  simulation rendering, but not an interactive native MuJoCo viewer with
  draggable camera.
- Native interactive MuJoCo viewer requires a remote graphical stack. On port 26,
  `Xvfb`, `x11vnc`, `websockify`, and `vglrun` were not available, and `DISPLAY`
  was unset.
- Future-video integration is currently a WAM-Lab wrapper around FastWAM's
  official eval script, not yet surfaced through the WAM-Lab policy server API.
- The spatial sweep is not the full official LIBERO evaluation protocol.

## Resource Notes

- FastWAM server/model load peaked around 25 GB on GPU0 during these runs.
- GPU0 returned to 0 MiB after each managed run.

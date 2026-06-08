# 2026-06-08 Port22 FastWAM LIBERO Native Viewer

## Objective

Run the FastWAM LIBERO policy path on the port22 machine with the native
MuJoCo passive viewer available through noVNC, using a GPU with enough free
memory.

## Assumptions

- Development and execution stay under `~/workspace`.
- Port22 has the same WAM-Lab/FastWAM/LIBERO layout as port26.
- GPU7 is acceptable for the first run because it has the most free memory in
  the current snapshot.

## Current State

- `~/workspace` resolves to `/home/yiming/workspace` on port22.
- The WAM-Lab checkout is on `codex/wam-lab-infra-3d`.
- FastWAM weights and the single LIBERO spatial HDF5 are already present.
- The display stack starts on `DISPLAY=:99` with noVNC on remote port `6080`.

## Open Risks

- A human should still do a manual noVNC drag/zoom pass for subjective
  usability, even though automated noVNC mouse input changed the view.
- Future-video comparison is a separate FastWAM official-eval path and was not
  covered by this native-viewer run.

## Completed Validation

- `bash -n` passed for the viewer stack scripts.
- `pytest tests/test_wam_lab_contracts.py -q` passed: 8 tests.
- `vla_eval.cli.main test --validate` passed: 161/161 configs.
- The viewer stack setup completed with all checked binaries reporting
  `deps-ok`.
- noVNC on remote port `6080` returned HTTP 200.
- Port22 `SSH_CONNECTION` reports remote SSH port `22`, so the generated noVNC
  forwarding hint is correct for this host.
- FastWAM server loaded on `CUDA_VISIBLE_DEVICES=7`; peak observed GPU7 memory
  was about 71.7 GB used with 71.5 GB still free.
- The second real native-viewer run succeeded: 1/1 task, 71 steps, success
  video with 72 frames.
- The native viewer screenshot title is `MuJoCo : base`, with native MuJoCo UI
  panels visible.
- Automated noVNC drag produced a second screenshot with a changed view
  (`diff_bbox=(270, 30, 660, 466)`), showing that mouse interaction reaches the
  passive viewer.

## Artifacts

- Remote log dir:
  `/tmp/yiming/wam_lab_fastwam_native_viewer_port22_20260608_112112`
- Remote result dir:
  `/home/yiming/workspace/code/wam-lab/results/fastwam_libero_native_viewer_port22_20260608_112112`
- Local artifact dir:
  `/Users/javadcc/code/ssh/H200/artifacts/port22_fastwam_native_viewer`
- Local files:
  `native_viewer_port22_reset_hold.png`,
  `native_viewer_port22_after_drag.png`, `episode_success.mp4`,
  `episode_success.jsonl`, `aggregate.json`, `recording.sqlite`.

## Failed Attempts

- The first directory sync placed files in the remote repo root because the
  rsync command did not preserve `scripts/` and `docs/`. Those duplicate root
  files were removed immediately, and the files were re-synced to the correct
  directories.
- Port22 does not provide `ss`, so scripts that used `ss` for localhost port
  checks would mis-detect active noVNC/model-server ports. Port checks were
  replaced with a Python socket probe.
- The first real FastWAM run loaded the policy successfully but failed during
  LIBERO reset with `MUJOCO_EGL_DEVICE_ID ... got 7` because native-viewer mode
  skipped the EGL vendor-file bootstrap required by robosuite offscreen camera
  observations. Native-viewer backend selection now writes the NVIDIA EGL ICD
  file while keeping the interactive viewer on GLFW.

## Next Actions

- Run reviewer pass over the script portability and EGL bootstrap changes.
- Commit and push the port22 native-viewer fixes.
- Run or refresh the future-video comparison path separately.

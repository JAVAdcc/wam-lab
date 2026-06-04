# WAM-Lab Phase 1: FastWAM + LIBERO

## Goal

Run the released FastWAM LIBERO checkpoint through the vla-evaluation-harness episode loop, while keeping the benchmark/policy bridge explicit and leaving a 3D privileged observation path for sparse-representation experiments.

## Architecture

- Benchmark process: `LIBEROBenchmark` emits standard vla-eval observations plus optional `raw_libero_obs` and `privileged_3d`.
- Policy process: `FastWAMLiberoServer` loads `~/workspace/code/FastWAM` by default and reuses FastWAM's official LIBERO preprocessing and `_predict_action_chunk`.
- Wire protocol: existing vla-eval WebSocket/msgpack protocol; FastWAM returns action chunks under `{"actions": np.ndarray[T, 7]}`.

## Environment

All paths below stay under `/home/yiming/workspace`.

Use the prepared FastWAM venv on the H200/L20X host:

```bash
source /home/yiming/workspace/envs/fastwam/bin/activate
cd /home/yiming/workspace/code/wam-lab
python -m pip install -e .
```

Install LIBERO/MuJoCo into the same environment before running local no-Docker evaluation. The setup script keeps NumPy/Hydra/Transformers pinned to the FastWAM-compatible versions, installs LIBERO as an editable checkout under `/home/yiming/workspace/code/LIBERO`, writes `LIBERO_CONFIG_PATH` under `/home/yiming/workspace/.libero`, and creates a user-space EGL vendor JSON under `/home/yiming/workspace/.egl`:

```bash
scripts/setup_fastwam_libero_env.sh
```

Validate the wam-lab configuration without launching a policy server:

```bash
wam-lab test --validate
```

Download the released checkpoint and stats. The script pins the Hugging Face model revision and prints SHA256 checksums for the two downloaded files:

```bash
scripts/download_fastwam_libero_release.sh
```

FastWAM also needs Wan common files for inference. `scripts/run_fastwam_server.sh` sets `DIFFSYNTH_DOWNLOAD_SOURCE=huggingface` by default and the server config passes `model.redirect_common_files=false`, so the public files are downloaded from `Wan-AI/Wan2.2-TI2V-5B` instead of the unavailable DiffSynth redirect repo.

## Run

Terminal 1:

`run_fastwam_server.sh` sets `DIFFSYNTH_MODEL_BASE_PATH` to the FastWAM `checkpoints/` directory unless it is already set.

```bash
scripts/run_fastwam_server.sh
```

Terminal 2:

```bash
scripts/run_fastwam_libero_smoke.sh
```

For 3D privileged data:

```bash
CONFIG=configs/benchmarks/libero/fastwam_smoke_3d.yaml scripts/run_fastwam_libero_smoke.sh
```

## 3D Privileged Fields

When `send_privileged_3d=true`, the LIBERO observation includes:

- `object_poses`: object/fixture body positions and quaternions from MuJoCo state.
- `contacts`: active geom contact pairs and contact distance.
- `cameras`: robosuite camera intrinsics and extrinsics.
- `depth` and `segmentation`: raw robosuite depth/segmentation observation keys, if enabled.
- `pointcloud`: optional stride-sampled world-frame point cloud from depth when `pointcloud_stride` is set.
- `mesh_summary`: MuJoCo mesh count and mesh names. Full per-object mesh extraction is intentionally not part of phase 1.

## Known Debt

- `vla-eval test -c configs/model_servers/fastwam/libero_uncond_2cam224.yaml` is an expensive server smoke test. It loads a 6B-parameter FastWAM policy and should only be run when the release checkpoint plus Wan VAE/text encoder/tokenizer are present.
- The project smoke harness benchmark test still expects Docker. Phase 1 no-Docker LIBERO validation is done with `wam-lab run --no-docker` through `scripts/run_fastwam_libero_smoke.sh`.
- FastWAM loading still depends on the upstream FastWAM checkout and its environment; this phase does not vendor FastWAM.
- Full mesh extraction and stable per-object mesh-to-body association are deferred.
- Current smoke config runs one task/episode. Full LIBERO scoring should be added after the bridge is verified and GPUs are available.

## Working Log

- 2026-06-04: Forked `vla-evaluation-harness` into `/home/yiming/workspace/code/wam-lab` and added a FastWAM LIBERO model server plus no-Docker LIBERO smoke configs.
- 2026-06-04: Installed LIBERO/robosuite/MuJoCo into the prepared FastWAM venv with FastWAM-compatible NumPy/Hydra/Transformers pins; `pip check` passed.
- 2026-06-04: Pinned LIBERO to commit `8f1084e3132a39270c3a13ebe37270a43ece2a01` and the FastWAM HF release to revision `139eebb6d90cdd9bdbbe465f72c6edc9ad5a518a`.
- 2026-06-04: Downloaded the FastWAM LIBERO release checkpoint/stats and the Wan common files required by FastWAM inference.
- 2026-06-04: Full FastWAM+LIBERO no-Docker smoke and 3D privileged-observation smoke both completed successfully on `CUDA_VISIBLE_DEVICES=4`.
- 2026-06-04: Reviewer pass flagged import-time EGL file writes, global `torch.load` monkey-patching, and unpinned external revisions; these were fixed before publishing.

## Validation Evidence

- `wam-lab test --validate`: 157/157 benchmark configs valid.
- LIBERO 3D observation smoke: produced `raw_libero_obs`, depth/segmentation, object poses, camera intrinsics/extrinsics, and stride-sampled pointcloud for `libero_spatial` task 0.
- FastWAM release SHA256:
  - `libero_uncond_2cam224.pt`: `1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579`
  - `libero_uncond_2cam224_dataset_stats.json`: `30f81ad7d5076e97323e3328bce003e01a04cb21327b5bacd21bb72846768638`
- FastWAM server load: release checkpoint loaded on `CUDA_VISIBLE_DEVICES=4`; Wan common files downloaded from `Wan-AI/Wan2.2-TI2V-5B`; server started on `ws://0.0.0.0:8000`.
- `configs/benchmarks/libero/fastwam_smoke.yaml`: 1/1 success, 70 steps, aggregate at `results/fastwam_libero_smoke/libero_fastwam_smoke_aggregate.json`.
- `configs/benchmarks/libero/fastwam_smoke_3d.yaml`: 1/1 success, 70 steps, aggregate at `results/fastwam_libero_smoke_3d/libero_fastwam_smoke_3d_aggregate.json`.

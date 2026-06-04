# WAM-Lab Infra Roadmap

## Current Objective

Extend WAM-Lab from a FastWAM+LIBERO smoke bridge into a more auditable evaluation
infra for WAM policies and 3D/sparse-representation experiments.

## Track A: Policy/Benchmark Adapter Reliability

The vla-evaluation-harness base is useful, but many compatibility checks are
runtime warnings. WAM-Lab should make interface assumptions explicit before
large policy servers are loaded.

Phase 2 additions:

- `vla_eval.contracts.InterfaceContract`: a lightweight policy/benchmark
  contract containing action spec, observation spec, observation params, and
  optional real payload-key requirements.
- `FastWAMLiberoServer.interface_contract()`: exposes the FastWAM-LIBERO wire
  contract without loading FastWAM weights.
- `merge_observation_params()`: mirrors orchestrator behavior so contract tests
  can verify which server-requested benchmark params are applied, ignored, or
  already set.
- Compatibility reports expose `payload_check_skipped` so future matrix tools do
  not present convention-only checks as strict payload validation.
- Contract tests check FastWAM-LIBERO compatibility and required payload keys
  without triggering LIBERO EGL setup or GPU allocation.

Open reliability work:

- Add a CLI command or script that prints a matrix of policy config vs benchmark
  config compatibility.
- Promote selected spec mismatches from warnings to fail-fast checks for WAM-Lab
  smoke configs.
- Add mock-server contract regression tests for action chunking, observation
  params, and recording behavior across LIBERO/CALVIN/Simpler-style stubs.

## Track B: 3D and Sparse Features

LIBERO can expose depth, segmentation, camera matrices, object poses, contacts,
and sampled point clouds. WAM-Lab should keep these as privileged features that
can be recorded and replayed without changing policy inference.

Phase 2 additions:

- `vla_eval.features.sparse3d.SparseSceneConfig`
- `extract_sparse_scene_features(privileged_3d, config)`
- deterministic voxel downsampling plus max-point limiting
- JSON-friendly sparse scene output for SQLite/JSONL recording
- non-finite point/contact values are filtered or converted to `null`; sparse
  config fails fast on non-positive limits
- LIBERO params:
  - `record_sparse_3d`
  - `sparse_pointcloud_voxel_size`
  - `sparse_pointcloud_max_points`
- `configs/benchmarks/libero/fastwam_visual_trace.yaml` records both normal
  rollout video and compact `sparse_3d` step fields.
- `record_sparse_3d` is recorder-only by default: it enables internal depth,
  segmentation, and pointcloud extraction but does not add `privileged_3d` to
  the observation payload sent to the policy server unless `send_privileged_3d`
  is explicitly set.

Feature schema:

```text
sparse_scene_v1
  points_xyz: [N, 3]
  segmentation_instance: [N] optional
  camera_names: [C]
  camera_indices: [N]
  objects:
    names: [M]
    positions: [M, 3]
    quaternions_wxyz: [M, 4]
  contacts: [{geom1, geom2, dist}]
  counts: {points, cameras, objects, contacts}
```

Open 3D work:

- Generate an HTML/Plotly 3D viewer from recorded `sparse_3d` rows.
- Add end-effector trajectory and action vectors to the sparse trace.
- Add object-level sparse descriptors beyond raw pose: relative object graph,
  contact graph, and camera-frame occupancy.
- Run one full FastWAM visual trace episode when GPU free memory is at least
  30-35 GiB.
- Run FastWAM future-video prediction separately with
  `EVALUATION.visualize_future_video=true`; this uses `infer_joint` and VAE
  decoding, so it is expected to require more memory than action-only inference.

## Working Log

- 2026-06-04: Phase 1 FastWAM+LIBERO smoke and 3D smoke passed.
- 2026-06-04: GPU4 action-only FastWAM inference observed roughly 24 GiB
  incremental VRAM use; future-video prediction was deferred because current
  free memory was only about 25 GiB.
- 2026-06-04: Added contract and sparse-3D scaffolds for CPU-level reliability
  tests without loading FastWAM weights.
- 2026-06-04: Reviewer pass tightened the first implementation: sparse 3D
  recording no longer pollutes policy payloads, contracts distinguish
  convention specs from required payload keys, and sparse features avoid
  non-standard JSON values.

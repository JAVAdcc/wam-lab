from __future__ import annotations

from pathlib import Path

import numpy as np

from vla_eval.benchmarks.libero.benchmark import LIBEROBenchmark
from vla_eval.contracts import InterfaceContract, check_contracts, contract_from_endpoint, merge_observation_params
from vla_eval.model_servers.fastwam import FastWAMLiberoServer


def _minimal_libero_raw_obs() -> dict:
    return {
        "agentview_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
    }


def test_fastwam_libero_contract_matches_libero_benchmark_without_loading_weights(tmp_path, monkeypatch):
    egl_file = tmp_path / "egl" / "nvidia_icd.json"
    monkeypatch.setenv("WAM_LAB_EGL_VENDOR_FILE", str(egl_file))

    policy = FastWAMLiberoServer.interface_contract()
    merge = merge_observation_params(
        LIBEROBenchmark,
        {"suite": "libero_spatial", "seed": 7},
        policy.observation_params,
    )
    benchmark = LIBEROBenchmark(**merge.merged_params)
    raw_obs = _minimal_libero_raw_obs()
    obs = benchmark.make_obs(raw_obs, {"name": "pick up the bowl"})
    base_contract = contract_from_endpoint("libero_spatial", benchmark)
    bench_contract = InterfaceContract(
        name=base_contract.name,
        action_spec=base_contract.action_spec,
        observation_spec=base_contract.observation_spec,
        produced_payload_keys=tuple(obs.keys()),
    )

    report = check_contracts(policy, bench_contract)

    assert report.compatible, report.warnings
    assert report.payload_check_skipped is False
    assert merge.applied == {
        "send_raw_libero_obs": True,
    }
    assert merge.ignored == {}
    assert set(policy.required_payload_keys).issubset(obs.keys())
    assert not egl_file.exists(), "Contract audit must not trigger LIBERO/EGL bootstrap side effects"


def test_observation_param_merge_ignores_unknown_params_and_preserves_explicit_values():
    merge = merge_observation_params(
        LIBEROBenchmark,
        {"send_state": False},
        {"send_state": True, "send_wrist_image": True, "not_a_libero_param": True},
    )

    assert merge.merged_params["send_state"] is False
    assert merge.merged_params["send_wrist_image"] is True
    assert "not_a_libero_param" not in merge.merged_params
    assert merge.already_set == {"send_state": False}
    assert merge.ignored == {"not_a_libero_param": True}


def test_record_sparse_3d_does_not_send_privileged_payload_by_default():
    benchmark = LIBEROBenchmark(
        suite="libero_spatial",
        send_raw_libero_obs=True,
        send_privileged_3d=False,
        record_sparse_3d=True,
    )

    obs = benchmark.make_obs(_minimal_libero_raw_obs(), {"name": "task"})

    assert benchmark.camera_depths is True
    assert benchmark.pointcloud_stride == 16
    assert "raw_libero_obs" in obs
    assert "privileged_3d" not in obs


def test_contract_report_marks_payload_check_skipped_when_no_payload_keys_are_available():
    policy = FastWAMLiberoServer.interface_contract()
    benchmark = InterfaceContract(name="empty", action_spec=policy.action_spec, observation_spec=policy.observation_spec)

    report = check_contracts(policy, benchmark)

    assert report.compatible
    assert report.payload_check_skipped is True


def test_fastwam_visual_trace_config_exists():
    config = Path("configs/benchmarks/libero/fastwam_visual_trace.yaml")
    assert config.exists()

from __future__ import annotations

import json

import numpy as np
import pytest

from vla_eval.features.sparse3d import SparseSceneConfig, extract_sparse_scene_features, voxel_downsample


def test_voxel_downsample_keeps_aligned_metadata():
    points = np.array(
        [
            [0.001, 0.001, 0.001],
            [0.002, 0.002, 0.002],
            [0.030, 0.000, 0.000],
        ],
        dtype=np.float32,
    )
    labels = np.array([10, 11, 12], dtype=np.int32)
    camera_indices = np.array([0, 0, 1], dtype=np.int32)

    out_points, out_labels, out_cameras = voxel_downsample(
        points,
        voxel_size=0.02,
        labels=labels,
        camera_indices=camera_indices,
    )

    assert out_points.shape == (2, 3)
    assert out_labels.tolist() == [10, 12]
    assert out_cameras.tolist() == [0, 1]


def test_sparse_scene_features_are_compact_and_json_friendly():
    privileged = {
        "pointcloud": {
            "agentview": {
                "xyz": np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.05, 0.0, 0.0]], dtype=np.float32),
                "segmentation_instance": np.array([1, 1, 2], dtype=np.int32),
            },
            "wrist": {
                "xyz": np.array([[0.0, 0.05, 0.0], [0.0, 0.06, 0.0]], dtype=np.float32),
                "segmentation_instance": np.array([3, 3], dtype=np.int32),
            },
        },
        "object_poses": {
            "obj_b": {"pos": np.array([1.0, 2.0, 3.0]), "quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0])},
            "obj_a": {"pos": np.array([0.0, 0.0, 0.0]), "quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0])},
        },
        "contacts": [{"geom1": "a", "geom2": "b", "dist": -0.01}],
    }

    feature = extract_sparse_scene_features(
        privileged,
        SparseSceneConfig(voxel_size=0.02, max_points=3),
    )

    assert feature["schema"] == "sparse_scene_v1"
    assert feature["counts"]["points"] <= 3
    assert feature["counts"]["objects"] == 2
    assert feature["counts"]["contacts"] == 1
    assert feature["objects"]["names"] == ["obj_a", "obj_b"]
    assert isinstance(feature["points_xyz"], list)
    assert isinstance(feature["segmentation_instance"], list)
    json.dumps(feature, allow_nan=False)


def test_sparse_scene_features_use_none_for_missing_object_pose_values():
    feature = extract_sparse_scene_features(
        {"object_poses": {"bad_object": {"pos": [1.0], "quat_wxyz": None}}},
        SparseSceneConfig(max_points=8),
    )

    assert feature["objects"]["positions"] == [[1.0, None, None]]
    assert feature["objects"]["quaternions_wxyz"] == [[None, None, None, None]]
    assert feature["objects"]["valid"] == [False]
    json.dumps(feature, allow_nan=False)


def test_sparse_scene_features_filter_nonfinite_points_and_contacts():
    feature = extract_sparse_scene_features(
        {
            "pointcloud": {
                "agentview": {
                    "xyz": np.array([[0.0, 0.0, 0.0], [np.nan, 1.0, 1.0], [0.1, 0.0, 0.0]], dtype=np.float32),
                    "segmentation_instance": np.array([1, 2, 3], dtype=np.int32),
                }
            },
            "contacts": [{"geom1": "a", "geom2": "b", "dist": np.nan}],
        },
        SparseSceneConfig(max_points=8),
    )

    assert feature["counts"]["points"] == 2
    assert feature["segmentation_instance"] == [1, 3]
    assert feature["contacts"] == [{"geom1": "a", "geom2": "b", "dist": None}]
    json.dumps(feature, allow_nan=False)


def test_sparse_scene_config_rejects_zero_or_negative_limits():
    with pytest.raises(ValueError):
        SparseSceneConfig(max_points=0)
    with pytest.raises(ValueError):
        SparseSceneConfig(voxel_size=0.0)

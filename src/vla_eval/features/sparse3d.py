"""Sparse 3D feature extraction from privileged simulator observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SparseSceneConfig:
    """Controls compact 3D feature extraction.

    The output is JSON-friendly by default so it can be stored in the harness
    step recorder without a custom binary side channel.
    """

    voxel_size: float | None = 0.02
    max_points: int | None = 2048
    include_segmentation: bool = True
    include_contacts: bool = True
    include_object_poses: bool = True
    camera_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.max_points is not None and self.max_points <= 0:
            raise ValueError("SparseSceneConfig.max_points must be positive, or None for no limit.")
        if self.voxel_size is not None and self.voxel_size <= 0:
            raise ValueError("SparseSceneConfig.voxel_size must be positive, or None to disable voxel downsampling.")


def _as_2d_points(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    return arr.reshape(-1, 3)


def _as_1d_int(value: Any, length: int, fill: int = -1) -> np.ndarray:
    if value is None:
        return np.full(length, fill, dtype=np.int32)
    arr = np.asarray(value).reshape(-1)
    if len(arr) != length:
        return np.full(length, fill, dtype=np.int32)
    return arr.astype(np.int32)


def _fixed_float_list(value: Any, length: int) -> tuple[list[float | None], bool]:
    if value is None:
        return [None] * length, False
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    out: list[float | None] = [None] * length
    valid = len(arr) >= length
    for idx in range(min(length, len(arr))):
        item = float(arr[idx])
        if np.isfinite(item):
            out[idx] = item
        else:
            valid = False
    return out, valid and all(item is not None for item in out)



def voxel_downsample(
    points: np.ndarray,
    *,
    voxel_size: float | None,
    labels: np.ndarray | None = None,
    camera_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Deterministically keep one point per voxel.

    Returns points and any aligned metadata arrays. The first point observed in
    each voxel is retained, preserving deterministic behavior across runs.
    """

    points = _as_2d_points(points)
    if points.shape[0] == 0 or voxel_size is None or voxel_size <= 0:
        return points, labels, camera_indices

    keys = np.floor(points / float(voxel_size)).astype(np.int64)
    _, first_indices = np.unique(keys, axis=0, return_index=True)
    keep = np.sort(first_indices)
    points_out = points[keep]
    labels_out = labels[keep] if labels is not None else None
    camera_out = camera_indices[keep] if camera_indices is not None else None
    return points_out, labels_out, camera_out


def _limit_points(
    points: np.ndarray,
    labels: np.ndarray | None,
    camera_indices: np.ndarray | None,
    max_points: int | None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    if max_points is None or points.shape[0] <= max_points:
        return points, labels, camera_indices
    keep = np.linspace(0, points.shape[0] - 1, int(max_points), dtype=np.int64)
    points = points[keep]
    labels = labels[keep] if labels is not None else None
    camera_indices = camera_indices[keep] if camera_indices is not None else None
    return points, labels, camera_indices


def _collect_points(
    privileged_3d: dict[str, Any],
    config: SparseSceneConfig,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, list[str]]:
    pointcloud = privileged_3d.get("pointcloud", {})
    if not isinstance(pointcloud, dict):
        return np.empty((0, 3), dtype=np.float32), None, None, []

    points_by_camera = []
    labels_by_camera = []
    camera_indices_by_camera = []
    camera_names: list[str] = []
    allowed = set(config.camera_names) if config.camera_names is not None else None
    for camera_name in sorted(pointcloud):
        if allowed is not None and camera_name not in allowed:
            continue
        cloud = pointcloud[camera_name]
        if not isinstance(cloud, dict) or "xyz" not in cloud:
            continue
        xyz = _as_2d_points(cloud["xyz"])
        finite = np.isfinite(xyz).all(axis=1)
        xyz = xyz[finite]
        if xyz.shape[0] == 0:
            continue
        camera_idx = len(camera_names)
        camera_names.append(str(camera_name))
        points_by_camera.append(xyz)
        labels = _as_1d_int(cloud.get("segmentation_instance"), len(finite))
        labels_by_camera.append(labels[finite])
        camera_indices_by_camera.append(np.full(xyz.shape[0], camera_idx, dtype=np.int32))

    if not points_by_camera:
        return np.empty((0, 3), dtype=np.float32), None, None, camera_names

    points = np.concatenate(points_by_camera, axis=0).astype(np.float32)
    labels = np.concatenate(labels_by_camera, axis=0).astype(np.int32)
    camera_indices = np.concatenate(camera_indices_by_camera, axis=0).astype(np.int32)
    points, labels, camera_indices = voxel_downsample(
        points,
        voxel_size=config.voxel_size,
        labels=labels,
        camera_indices=camera_indices,
    )
    points, labels, camera_indices = _limit_points(points, labels, camera_indices, config.max_points)
    if not config.include_segmentation:
        labels = None
    return points, labels, camera_indices, camera_names


def _object_pose_features(privileged_3d: dict[str, Any]) -> dict[str, Any]:
    object_poses = privileged_3d.get("object_poses", {})
    if not isinstance(object_poses, dict):
        return {"names": [], "positions": [], "quaternions_wxyz": []}
    names = sorted(str(name) for name in object_poses)
    positions = []
    quats = []
    valid = []
    for name in names:
        pose = object_poses[name]
        pos, pos_valid = _fixed_float_list(pose.get("pos"), 3)
        quat, quat_valid = _fixed_float_list(pose.get("quat_wxyz"), 4)
        positions.append(pos)
        quats.append(quat)
        valid.append(bool(pos_valid and quat_valid))
    return {
        "names": names,
        "positions": positions,
        "quaternions_wxyz": quats,
        "valid": valid,
    }


def _contact_features(privileged_3d: dict[str, Any]) -> list[dict[str, Any]]:
    contacts = privileged_3d.get("contacts", [])
    if not isinstance(contacts, list):
        return []
    out = []
    for contact in contacts:
        if not isinstance(contact, dict):
            continue
        out.append(
            {
                "geom1": str(contact.get("geom1", "")),
                "geom2": str(contact.get("geom2", "")),
                "dist": _finite_float_or_none(contact.get("dist", 0.0)),
            }
        )
    return out


def _finite_float_or_none(value: Any) -> float | None:
    try:
        item = float(value)
    except (TypeError, ValueError):
        return None
    return item if np.isfinite(item) else None


def extract_sparse_scene_features(
    privileged_3d: dict[str, Any],
    config: SparseSceneConfig | None = None,
) -> dict[str, Any]:
    """Convert privileged 3D observation data into a compact sparse feature dict."""

    cfg = config or SparseSceneConfig()
    points, labels, camera_indices, camera_names = _collect_points(privileged_3d, cfg)
    feature: dict[str, Any] = {
        "schema": "sparse_scene_v1",
        "points_xyz": points.astype(np.float32).tolist(),
        "camera_names": camera_names,
        "camera_indices": [] if camera_indices is None else camera_indices.astype(np.int32).tolist(),
        "counts": {
            "points": int(points.shape[0]),
            "cameras": int(len(camera_names)),
        },
    }
    if labels is not None:
        feature["segmentation_instance"] = labels.astype(np.int32).tolist()
    if cfg.include_object_poses:
        objects = _object_pose_features(privileged_3d)
        feature["objects"] = objects
        feature["counts"]["objects"] = int(len(objects["names"]))
    if cfg.include_contacts:
        contacts = _contact_features(privileged_3d)
        feature["contacts"] = contacts
        feature["counts"]["contacts"] = int(len(contacts))
    return feature

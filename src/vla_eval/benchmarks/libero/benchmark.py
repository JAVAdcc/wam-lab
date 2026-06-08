"""LIBERO benchmark implementation."""

from __future__ import annotations

import os
import functools
import logging
import time
from pathlib import Path
from contextlib import contextmanager
from typing import Any

import math

import numpy as np

from vla_eval.benchmarks.base import StepBenchmark, StepResult
from vla_eval.benchmarks.libero.utils import preprocess_libero_image
from vla_eval.features.sparse3d import SparseSceneConfig, extract_sparse_scene_features
from vla_eval.rotation import matrix_to_quat, quat_to_axisangle
from vla_eval.specs import (
    GRIPPER_CLOSE_POS,
    IMAGE_RGB,
    LANGUAGE,
    POSITION_DELTA,
    ROTATION_AA,
    RAW,
    STATE_EEF_POS_AA_GRIP,
    DimSpec,
)
from vla_eval.types import Action, EpisodeResult, Observation, Task

# EGL for headless rendering
_DEFAULT_WORKSPACE_ROOT = Path(os.environ.get("WAM_LAB_WORKSPACE_ROOT", Path.home() / "workspace")).expanduser()
os.environ.setdefault("MUJOCO_GL", "egl")

logger = logging.getLogger(__name__)
_LIBERO_GL_BACKEND: str | None = None


def _ensure_egl_vendor_file() -> None:
    """Point PyOpenGL/MuJoCo EGL at the user-space NVIDIA ICD file."""
    egl_vendor_file = Path(
        os.environ.get("WAM_LAB_EGL_VENDOR_FILE", str(_DEFAULT_WORKSPACE_ROOT / ".egl" / "nvidia_icd.json"))
    ).expanduser()
    if "__EGL_VENDOR_LIBRARY_FILENAMES" not in os.environ:
        egl_vendor_file.parent.mkdir(parents=True, exist_ok=True)
        if not egl_vendor_file.exists():
            egl_vendor_file.write_text(
                '{"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}\n',
                encoding="utf-8",
            )
        os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(egl_vendor_file)


def _bootstrap_headless_rendering() -> None:
    """Prepare user-space EGL config lazily before robosuite imports."""
    if os.environ.get("MUJOCO_GL") != "egl":
        return
    _ensure_egl_vendor_file()
    os.environ.setdefault("EGL_PLATFORM", "device")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def _select_libero_gl_backend(native_renderer: bool) -> None:
    """Select the MuJoCo GL backend before LIBERO/robosuite imports.

    MuJoCo / PyOpenGL backend selection is process-global after import. Running
    native GLFW rendering and headless EGL rendering in one Python process is
    therefore intentionally rejected instead of silently contaminating the next
    benchmark.
    """
    global _LIBERO_GL_BACKEND
    target = "glfw" if native_renderer else "egl"
    if _LIBERO_GL_BACKEND is not None and _LIBERO_GL_BACKEND != target:
        raise RuntimeError(
            "Cannot mix LIBERO MuJoCo GL backends in one process: "
            f"already initialized {_LIBERO_GL_BACKEND!r}, requested {target!r}. "
            "Run native-viewer and headless/offscreen LIBERO benchmarks in separate processes."
        )
    if native_renderer:
        os.environ["MUJOCO_GL"] = "glfw"
        # LIBERO still creates an offscreen render context for policy camera
        # observations, even when the interactive MuJoCo viewer uses GLFW.
        _ensure_egl_vendor_file()
        os.environ.pop("PYOPENGL_PLATFORM", None)
        os.environ.pop("EGL_PLATFORM", None)
    else:
        os.environ["MUJOCO_GL"] = "egl"
        _bootstrap_headless_rendering()
    _LIBERO_GL_BACKEND = target


def _ensure_libero_config() -> None:
    """Create a non-interactive LIBERO path config under the user workspace."""
    config_root = Path(os.environ.get("LIBERO_CONFIG_PATH", str(_DEFAULT_WORKSPACE_ROOT / ".libero"))).expanduser()
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(config_root))
    config_file = config_root / "config.yaml"
    if config_file.exists():
        return

    benchmark_root = Path(
        os.environ.get("LIBERO_BENCHMARK_ROOT", str(_DEFAULT_WORKSPACE_ROOT / "code" / "LIBERO" / "libero" / "libero"))
    ).expanduser()
    dataset_root = Path(
        os.environ.get("LIBERO_DATASET_ROOT", str(_DEFAULT_WORKSPACE_ROOT / "data" / "libero" / "datasets"))
    ).expanduser()
    config_root.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)
    config = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(dataset_root),
        "assets": str(benchmark_root / "assets"),
    }

    import yaml

    config_file.write_text(yaml.safe_dump(config), encoding="utf-8")


@contextmanager
def _libero_torch_load_compat():
    """Temporarily use PyTorch pre-2.6 torch.load semantics for LIBERO init states."""
    import torch

    original_torch_load = torch.load

    @functools.wraps(original_torch_load)
    def patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = patched_load
    try:
        yield
    finally:
        torch.load = original_torch_load


def _quat_to_axisangle_robosuite(quat: np.ndarray) -> np.ndarray:
    """Robosuite-style quat [x,y,z,w] → axis-angle. No antipodal normalization."""
    q = quat.copy()
    if q[3] > 1.0:
        q[3] = 1.0
    elif q[3] < -1.0:
        q[3] = -1.0
    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (q[:3] * 2.0 * math.acos(q[3]) / den).astype(np.float32)


LIBERO_ENV_RESOLUTION = 256
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]

MAX_STEP_MAPPING = {
    "libero_spatial": 220,
    "libero_goal": 300,
    "libero_object": 280,
    "libero_10": 520,
    "libero_90": 400,
}


class LIBEROBenchmark(StepBenchmark):
    """LIBERO tabletop manipulation benchmark (MuJoCo/robosuite).

    Non-obvious behaviors:
        - **PyTorch compat**: Patches ``torch.load`` to use
          ``weights_only=False`` only while loading LIBERO initial-state files
          (numpy arrays stored via ``torch.save``).
        - **Headless rendering**: Lazily sets ``EGL_PLATFORM=device`` and
          ``PYOPENGL_PLATFORM=egl`` before LIBERO/robosuite imports for
          GPU-accelerated headless rendering.
        - **Dummy wait steps**: At episode start, ``num_steps_wait`` steps
          (default 10) are executed with a fixed open-gripper action to let
          objects settle in the physics simulation.
        - **Suite-specific max_steps**: libero_spatial=220, libero_object=280,
          libero_goal=300, libero_10=520, libero_90=400.
        - **Image preprocessing**: robosuite renders images with inverted axes.
          Both agentview and wrist images are flipped ``[::-1, ::-1]`` to
          correct orientation, then resized to 256×256 with padding.

    Args:
        suite: LIBERO suite name (e.g. "libero_spatial", "libero_10").
        seed: Random seed for environment initialization.
        num_steps_wait: Dummy action steps at episode start (default 10).
        send_wrist_image: Include wrist camera image in observations.
        send_state: Include proprioceptive 8-D state
            ``[pos3, axisangle3, gripper2]`` in observations.
        absolute_action: Use absolute (world-frame) actions instead of delta.
            When True, sets ``robot.controller.use_delta = False`` after the
            initial dummy-wait steps.
        max_steps: Override the default suite-specific max step count.
            When None, uses ``MAX_STEP_MAPPING[suite]``.
        env_seed: Seed for ``env.seed()``.  When None, defaults to ``seed``.
            OpenVLA reference uses ``env_seed=0`` separately from ``seed=7``.
        native_renderer: Create a native MuJoCo diagnostic viewer in addition
            to camera observations.
            Requires a valid X11/GLFW display and is intended for interactive
            diagnostics, not headless benchmark throughput.
        native_viewer_backend: Viewer implementation used when
            ``native_renderer`` is enabled. ``mujoco_passive`` launches
            ``mujoco.viewer.launch_passive`` with a free interactive camera;
            ``opencv`` preserves robosuite's fixed-camera OpenCV renderer.
        render_camera: Robosuite viewer camera used when ``native_renderer`` is
            enabled with ``native_viewer_backend="opencv"``.
        native_render_strict: Raise if the onscreen viewer cannot be rendered.
            Keep this enabled for diagnostic native-viewer runs so policy
            success cannot hide a broken viewer.
        native_viewer_reset_hold_sec: Pause after the first post-reset native
            viewer render. This gives a human time to focus the viewer and
            adjust the camera before policy actions start.
        native_viewer_end_hold_sec: Pause after a successful/terminal step so
            the final scene can be inspected before the episode closes.
    """

    _ALL_RECORD_FIELDS = frozenset({"reward", "done", "success", "privileged_3d_summary", "sparse_3d"})

    def __init__(
        self,
        suite: str = "libero_spatial",
        seed: int = 7,
        num_steps_wait: int = 10,
        send_wrist_image: bool = False,
        send_state: bool = False,
        absolute_action: bool = False,
        max_steps: int | None = None,
        env_seed: int | None = None,
        quat_no_antipodal: bool = False,
        send_raw_libero_obs: bool = False,
        send_privileged_3d: bool = False,
        camera_depths: bool = False,
        camera_segmentations: Any = None,
        pointcloud_stride: int | None = None,
        record_privileged_summary: bool = False,
        record_sparse_3d: bool = False,
        sparse_pointcloud_voxel_size: float | None = 0.02,
        sparse_pointcloud_max_points: int | None = 2048,
        native_renderer: bool = False,
        native_viewer_backend: str = "mujoco_passive",
        render_camera: str = "agentview",
        native_render_strict: bool = True,
        native_viewer_reset_hold_sec: float = 0.0,
        native_viewer_end_hold_sec: float = 0.0,
    ) -> None:
        super().__init__()
        self.suite = suite
        self.seed = seed
        self._quat_to_aa = _quat_to_axisangle_robosuite if quat_no_antipodal else quat_to_axisangle
        self.env_seed = env_seed if env_seed is not None else seed
        self.num_steps_wait = num_steps_wait
        self.send_wrist_image = send_wrist_image
        self.send_state = send_state
        self.absolute_action = absolute_action
        self._max_steps = max_steps
        self.send_raw_libero_obs = send_raw_libero_obs
        self.record_sparse_3d = record_sparse_3d
        self.send_privileged_3d = bool(send_privileged_3d)
        self._enable_privileged_3d = bool(send_privileged_3d or record_sparse_3d)
        self.camera_depths = bool(camera_depths or self._enable_privileged_3d)
        if camera_segmentations is None and self._enable_privileged_3d:
            camera_segmentations = ["instance", "element"]
        self.camera_segmentations = camera_segmentations
        self.pointcloud_stride = 16 if record_sparse_3d and pointcloud_stride is None else pointcloud_stride
        self.record_privileged_summary = record_privileged_summary
        self.native_renderer = bool(native_renderer)
        if native_viewer_backend not in {"mujoco_passive", "opencv"}:
            raise ValueError("native_viewer_backend must be one of: mujoco_passive, opencv")
        self.native_viewer_backend = native_viewer_backend
        self.render_camera = str(render_camera)
        self.native_render_strict = bool(native_render_strict)
        self.native_viewer_reset_hold_sec = max(0.0, float(native_viewer_reset_hold_sec))
        self.native_viewer_end_hold_sec = max(0.0, float(native_viewer_end_hold_sec))
        self._sparse_scene_config = SparseSceneConfig(
            voxel_size=sparse_pointcloud_voxel_size,
            max_points=sparse_pointcloud_max_points,
        )
        self._env = None
        self._native_viewer = None
        self._task_suite = None
        self._current_task_id: int | None = None
        self._native_render_warned = False

    def cleanup(self) -> None:
        self._close_native_viewer()
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None

    def _init_libero(self) -> None:
        """Lazily initialize LIBERO (heavy imports)."""
        if self._task_suite is not None:
            return
        _select_libero_gl_backend(self.native_renderer)
        _ensure_libero_config()

        from libero.libero import benchmark

        benchmark_dict = benchmark.get_benchmark_dict()
        self._task_suite = benchmark_dict[self.suite]()

    def get_tasks(self) -> list[Task]:
        self._init_libero()
        assert self._task_suite is not None
        tasks = []
        for task_id in range(self._task_suite.n_tasks):
            task = self._task_suite.get_task(task_id)
            tasks.append(
                {
                    "name": task.language,
                    "suite": self.suite,
                    "task_id": task_id,
                    "task_obj": task,
                }
            )
        return tasks

    def reset(self, task: Task) -> Any:
        from pathlib import Path

        from libero.libero import get_libero_path
        from libero.libero.envs.env_wrapper import ControlEnv, OffScreenRenderEnv

        task_obj = task["task_obj"]
        task_id = task["task_id"]
        episode_idx = task.get("episode_idx", 0)

        # Only create a new env when the task changes (reuse across episodes)
        if self._env is None or self._current_task_id != task_id:
            if self._env is not None:
                self._close_native_viewer()
                self._env.close()

            bddl_file = Path(get_libero_path("bddl_files")) / task_obj.problem_folder / task_obj.bddl_file
            env_args = {
                "bddl_file_name": str(bddl_file),
                "camera_heights": LIBERO_ENV_RESOLUTION,
                "camera_widths": LIBERO_ENV_RESOLUTION,
                "camera_depths": self.camera_depths,
                "camera_segmentations": self.camera_segmentations,
            }
            if self.native_renderer:
                if self.native_viewer_backend == "opencv":
                    env_args.update(
                        {
                            "has_renderer": True,
                            "has_offscreen_renderer": True,
                            "render_camera": self.render_camera,
                            "use_camera_obs": True,
                        }
                    )
                else:
                    env_args.update(
                        {
                            "has_renderer": False,
                            "has_offscreen_renderer": True,
                            "use_camera_obs": True,
                        }
                    )
                env = ControlEnv(**env_args)
            else:
                env = OffScreenRenderEnv(**env_args)
            env.seed(self.env_seed)
            self._env = env
            self._current_task_id = task_id

        # Reset env before setting init state (matches reference)
        self._env.reset()

        # Set initial state
        assert self._task_suite is not None
        with _libero_torch_load_compat():
            initial_states = self._task_suite.get_task_init_states(task_id)
        obs = self._env.set_init_state(initial_states[episode_idx])

        # Run dummy action wait steps (always in delta mode to avoid slamming to origin)
        for _ in range(self.num_steps_wait):
            obs, _, _, _ = self._env.step(LIBERO_DUMMY_ACTION)
            self._render_native_viewer()

        # Switch to absolute action mode after settling (e.g. for X-VLA)
        if self.absolute_action:
            for robot in self._env.robots:
                robot.controller.use_delta = False

        self._recorder.record_video(self._extract_frame(obs))
        self._render_native_viewer()
        self._hold_native_viewer(self.native_viewer_reset_hold_sec, "post-reset")
        return obs

    def step(self, action: Action) -> StepResult:
        raw_action = action.get("actions", action.get("action"))
        if isinstance(raw_action, np.ndarray):
            raw_action = raw_action.tolist()
        assert len(raw_action) == 7, f"Action dimension mismatch: got {len(raw_action)}, expected 7"

        # Discretize gripper
        if raw_action[-1] < 0:
            gripper = -1.0
        else:
            gripper = 1.0
        processed_action = raw_action[:-1] + [gripper]

        assert self._env is not None
        obs, reward, done, info = self._env.step(processed_action)
        self._render_native_viewer()
        self._recorder.record_video(self._extract_frame(obs))
        record_fields: dict[str, Any] = {"reward": float(reward), "done": bool(done), "success": bool(done)}
        if self.record_privileged_summary and self._enable_privileged_3d:
            record_fields["privileged_3d_summary"] = self._privileged_summary(obs)
        if self.record_sparse_3d and self._enable_privileged_3d:
            record_fields["sparse_3d"] = extract_sparse_scene_features(
                self._extract_privileged_3d(obs),
                self._sparse_scene_config,
            )
        self._recorder.record_step(**record_fields)
        if done:
            self._hold_native_viewer(self.native_viewer_end_hold_sec, "terminal")
        return StepResult(obs=obs, reward=reward, done=done, info=info)

    def _render_native_viewer(self) -> None:
        if not self.native_renderer or self._env is None:
            return
        if self.native_viewer_backend == "mujoco_passive":
            self._sync_mujoco_passive_viewer()
            return
        try:
            render_fn = getattr(self._env, "render", None)
            if render_fn is not None:
                render_fn()
                return
            inner = getattr(self._env, "env", None)
            inner_render_fn = getattr(inner, "render", None)
            if inner_render_fn is not None:
                inner_render_fn()
                return
            raise RuntimeError("Native renderer is enabled, but LIBERO env exposes no render() method")
        except Exception as exc:
            if self.native_render_strict:
                raise RuntimeError("Native MuJoCo viewer render failed") from exc
            if not self._native_render_warned:
                logger.exception("Native MuJoCo viewer render failed; continuing benchmark without viewer updates")
                self._native_render_warned = True

    def _sync_mujoco_passive_viewer(self) -> None:
        assert self._env is not None
        try:
            handle = self._native_viewer
            if handle is None:
                import mujoco
                import mujoco.viewer

                sim = self._env.sim
                handle = mujoco.viewer.launch_passive(
                    sim.model._model,
                    sim.data._data,
                    show_left_ui=True,
                    show_right_ui=True,
                )
                handle.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                handle.cam.fixedcamid = -1
                self._configure_passive_viewer_camera(handle.cam)
                self._native_viewer = handle
            if not handle.is_running():
                raise RuntimeError("MuJoCo passive viewer is not running")
            handle.sync()
        except Exception as exc:
            if self.native_render_strict:
                raise RuntimeError("Native MuJoCo passive viewer sync failed") from exc
            if not self._native_render_warned:
                logger.exception("Native MuJoCo passive viewer sync failed; continuing benchmark without viewer updates")
                self._native_render_warned = True

    @staticmethod
    def _configure_passive_viewer_camera(cam: Any) -> None:
        cam.lookat[:] = np.array([0.0, 0.0, 0.75], dtype=np.float64)
        cam.distance = 2.0
        cam.azimuth = 90.0
        cam.elevation = -35.0

    def _close_native_viewer(self) -> None:
        handle = self._native_viewer
        self._native_viewer = None
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass

    def _hold_native_viewer(self, seconds: float, phase: str) -> None:
        if not self.native_renderer or seconds <= 0:
            return
        deadline = time.monotonic() + seconds
        logger.info("Holding native MuJoCo viewer for %.1fs at %s", seconds, phase)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            self._render_native_viewer()
            time.sleep(min(0.25, remaining))

    @staticmethod
    def _extract_frame(raw_obs: Any) -> np.ndarray | None:
        if not isinstance(raw_obs, dict):
            return None
        frame = raw_obs.get("agentview_image")
        if frame is None:
            return None
        # Robosuite renders agentview/wrist inverted; flip to upright.
        return np.ascontiguousarray(frame[::-1, ::-1])

    def make_obs(self, raw_obs: Any, task: Task) -> Observation:
        img = preprocess_libero_image(raw_obs["agentview_image"], LIBERO_ENV_RESOLUTION)

        obs_dict: dict[str, Any] = {
            "images": {"agentview": img},
            "task_description": task["name"],
            "language": task["name"],
        }

        if self.send_wrist_image:
            wrist = preprocess_libero_image(raw_obs["robot0_eye_in_hand_image"], LIBERO_ENV_RESOLUTION)
            obs_dict["images"]["wrist"] = wrist

        if self.send_state:
            # Both sources: observation (default) and controller.
            # Most models (Pi0, OFT, GR00T) use obs; X-VLA uses controller.
            obs_dict["states"] = np.concatenate(
                [
                    raw_obs["robot0_eef_pos"],
                    self._quat_to_aa(raw_obs["robot0_eef_quat"]),
                    raw_obs["robot0_gripper_qpos"],
                ]
            )
            assert self._env is not None
            robot = self._env.robots[0]
            ee_pos = np.asarray(robot.controller.ee_pos, dtype=np.float32)
            ee_ori_mat = np.asarray(robot.controller.ee_ori_mat, dtype=np.float32)
            ee_aa = quat_to_axisangle(matrix_to_quat(ee_ori_mat))
            obs_dict["controller_states"] = np.concatenate(
                [ee_pos, ee_aa, np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float32)]
            )

        if self.send_raw_libero_obs:
            obs_dict["raw_libero_obs"] = self._extract_raw_libero_obs(raw_obs)

        if self.send_privileged_3d:
            obs_dict["privileged_3d"] = self._extract_privileged_3d(raw_obs)

        return obs_dict

    @staticmethod
    def _extract_raw_libero_obs(raw_obs: Any) -> dict[str, Any]:
        keys = [
            "agentview_image",
            "robot0_eye_in_hand_image",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ]
        out = {key: raw_obs[key] for key in keys if key in raw_obs}
        for key, value in raw_obs.items():
            if key.endswith("_depth") or "_segmentation_" in key:
                out[key] = value
        missing = [key for key in keys if key not in out]
        if missing:
            raise KeyError(f"LIBERO raw observation missing required keys: {missing}")
        return out

    def _extract_privileged_3d(self, raw_obs: Any) -> dict[str, Any]:
        assert self._env is not None
        sim = self._env.sim
        inner_env = self._env.env
        privileged: dict[str, Any] = {
            "object_poses": self._object_poses(inner_env),
            "contacts": self._contacts(sim),
            "cameras": self._camera_calibration(sim),
            "mesh_summary": self._mesh_summary(sim),
        }
        depth = {k: v for k, v in raw_obs.items() if k.endswith("_depth")}
        segmentation = {k: v for k, v in raw_obs.items() if "_segmentation_" in k}
        if depth:
            privileged["depth"] = depth
        if segmentation:
            privileged["segmentation"] = segmentation
        if self.pointcloud_stride is not None and depth:
            privileged["pointcloud"] = self._sample_pointcloud(sim, depth, segmentation, int(self.pointcloud_stride))
        return privileged

    @staticmethod
    def _object_poses(inner_env: Any) -> dict[str, dict[str, Any]]:
        poses: dict[str, dict[str, Any]] = {}
        for name, body_id in getattr(inner_env, "obj_body_id", {}).items():
            poses[str(name)] = {
                "pos": np.asarray(inner_env.sim.data.body_xpos[body_id], dtype=np.float32),
                "quat_wxyz": np.asarray(inner_env.sim.data.body_xquat[body_id], dtype=np.float32),
            }
        return poses

    @staticmethod
    def _contacts(sim: Any) -> list[dict[str, Any]]:
        contacts = []
        for contact in sim.data.contact[: sim.data.ncon]:
            contacts.append(
                {
                    "geom1": sim.model.geom_id2name(contact.geom1),
                    "geom2": sim.model.geom_id2name(contact.geom2),
                    "dist": float(contact.dist),
                }
            )
        return contacts

    def _camera_calibration(self, sim: Any) -> dict[str, dict[str, np.ndarray]]:
        from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix

        cameras: dict[str, dict[str, np.ndarray]] = {}
        camera_names = getattr(self._env.env, "camera_names", ["agentview", "robot0_eye_in_hand"])
        camera_heights = getattr(self._env.env, "camera_heights", [LIBERO_ENV_RESOLUTION] * len(camera_names))
        camera_widths = getattr(self._env.env, "camera_widths", [LIBERO_ENV_RESOLUTION] * len(camera_names))
        for name, height, width in zip(camera_names, camera_heights, camera_widths):
            cameras[str(name)] = {
                "intrinsic": get_camera_intrinsic_matrix(sim, str(name), int(height), int(width)).astype(np.float32),
                "extrinsic": get_camera_extrinsic_matrix(sim, str(name)).astype(np.float32),
            }
        return cameras

    @staticmethod
    def _mesh_summary(sim: Any) -> dict[str, Any]:
        mesh_names = []
        for mesh_id in range(int(getattr(sim.model, "nmesh", 0))):
            try:
                mesh_names.append(sim.model.mesh_id2name(mesh_id))
            except Exception:
                mesh_names.append(str(mesh_id))
        return {"nmesh": int(getattr(sim.model, "nmesh", 0)), "mesh_names": mesh_names}

    def _sample_pointcloud(
        self,
        sim: Any,
        depth_by_key: dict[str, np.ndarray],
        segmentation_by_key: dict[str, np.ndarray],
        stride: int,
    ) -> dict[str, dict[str, np.ndarray]]:
        from robosuite.utils.camera_utils import (
            get_camera_extrinsic_matrix,
            get_camera_intrinsic_matrix,
            get_real_depth_map,
        )

        stride = max(1, int(stride))
        clouds: dict[str, dict[str, np.ndarray]] = {}
        for depth_key, depth in depth_by_key.items():
            camera_name = depth_key[: -len("_depth")]
            depth_arr = np.asarray(depth).squeeze()
            if depth_arr.size == 0:
                continue
            if float(np.nanmax(depth_arr)) <= 1.0 and float(np.nanmin(depth_arr)) >= 0.0:
                depth_arr = get_real_depth_map(sim, depth_arr)
            height, width = depth_arr.shape
            rows, cols = np.mgrid[0:height:stride, 0:width:stride]
            z = depth_arr[rows, cols].reshape(-1).astype(np.float32)
            valid = np.isfinite(z) & (z > 0)
            rows_f = rows.reshape(-1).astype(np.float32)[valid]
            cols_f = cols.reshape(-1).astype(np.float32)[valid]
            z = z[valid]
            k = get_camera_intrinsic_matrix(sim, camera_name, height, width).astype(np.float32)
            ext = get_camera_extrinsic_matrix(sim, camera_name).astype(np.float32)
            x = (cols_f - k[0, 2]) * z / k[0, 0]
            y = (rows_f - k[1, 2]) * z / k[1, 1]
            cam_points = np.stack([x, y, z, np.ones_like(z)], axis=1)
            world_points = (ext @ cam_points.T).T[:, :3].astype(np.float32)
            cloud: dict[str, np.ndarray] = {"xyz": world_points}
            seg_key = f"{camera_name}_segmentation_instance"
            if seg_key in segmentation_by_key:
                seg = np.asarray(segmentation_by_key[seg_key]).squeeze()
                labels = seg[rows.reshape(-1)[valid], cols.reshape(-1)[valid]]
                cloud["segmentation_instance"] = labels.astype(np.int32)
            clouds[camera_name] = cloud
        return clouds

    def _privileged_summary(self, raw_obs: Any) -> dict[str, Any]:
        assert self._env is not None
        sim = self._env.sim
        inner_env = self._env.env
        return {
            "num_objects": len(getattr(inner_env, "obj_body_id", {})),
            "num_contacts": int(sim.data.ncon),
            "depth_keys": sorted([k for k in raw_obs if k.endswith("_depth")]),
            "segmentation_keys": sorted([k for k in raw_obs if "_segmentation_" in k]),
        }

    def check_done(self, step_result: StepResult) -> bool:
        return step_result.done

    def get_step_result(self, step_result: StepResult) -> EpisodeResult:
        return {"success": step_result.done}

    def get_metadata(self) -> dict[str, Any]:
        return {
            "max_steps": self._max_steps or MAX_STEP_MAPPING.get(self.suite, 300),
            "max_episodes_per_task": 50,  # bounded by initial_states per task
            "suite": self.suite,
        }

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {
            "position": POSITION_DELTA,
            "rotation": ROTATION_AA,
            "gripper": GRIPPER_CLOSE_POS,
        }

    def get_observation_spec(self) -> dict[str, DimSpec]:
        spec: dict[str, DimSpec] = {
            "agentview": IMAGE_RGB,
            "language": LANGUAGE,
        }
        if self.send_wrist_image:
            spec["wrist"] = IMAGE_RGB
        if self.send_state:
            spec["state"] = STATE_EEF_POS_AA_GRIP
        if self.send_raw_libero_obs:
            spec["raw_libero_obs"] = RAW
        if self.send_privileged_3d:
            spec["privileged_3d"] = RAW
        return spec

    def render(self) -> np.ndarray | None:
        try:
            assert self._env is not None
            render_fn = getattr(self._env, "render", None)
            if render_fn is not None:
                return render_fn()
            inner = getattr(self._env, "env", None)
            inner_render_fn = getattr(inner, "render", None)
            if inner_render_fn is not None:
                return inner_render_fn()
            return None
        except Exception:
            return None

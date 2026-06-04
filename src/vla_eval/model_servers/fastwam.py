"""FastWAM model server for LIBERO.

This adapter intentionally reuses FastWAM's official LIBERO preprocessing and
inference utilities. The benchmark should send ``raw_libero_obs`` so action
conventions stay identical to ``FastWAM/experiments/libero/eval_libero_single.py``.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from vla_eval.contracts import InterfaceContract
from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.specs import (
    GRIPPER_CLOSE_POS,
    IMAGE_RGB,
    LANGUAGE,
    POSITION_DELTA,
    RAW,
    ROTATION_AA,
    STATE_EEF_POS_AA_GRIP,
    DimSpec,
)
from vla_eval.types import Action, Observation

logger = logging.getLogger(__name__)


class FastWAMLiberoServer(PredictModelServer):
    """FastWAM policy server for LIBERO release checkpoints.

    Args:
        fastwam_root: Path to a local FastWAM checkout.
        checkpoint: FastWAM checkpoint file, e.g.
            ``~/workspace/code/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224.pt``.
        dataset_stats_path: Dataset stats JSON matching the checkpoint.
        task_config: FastWAM Hydra task config. The released LIBERO checkpoint
            uses ``libero_uncond_2cam224_1e-4``.
        config_name: FastWAM Hydra config name, normally ``sim_libero``.
        device: Torch device for model inference.
        mixed_precision: ``no``, ``fp16``, or ``bf16``. Matches FastWAM eval.
        action_horizon: Number of actions to sample per inference. ``None``
            falls back to FastWAM's config-derived horizon.
        num_inference_steps: Action diffusion inference steps.
        chunk_size: Number of actions served from each sampled chunk before
            the server replans.
    """

    @classmethod
    def interface_contract(cls) -> InterfaceContract:
        """Return the FastWAM-LIBERO wire contract without loading weights."""
        return InterfaceContract(
            name="fastwam_libero",
            action_spec={"position": POSITION_DELTA, "rotation": ROTATION_AA, "gripper": GRIPPER_CLOSE_POS},
            observation_spec={
                "language": LANGUAGE,
                "raw_libero_obs": RAW,
            },
            observation_params={
                "send_raw_libero_obs": True,
            },
            required_payload_keys=("raw_libero_obs", "task_description"),
            source="vla_eval.model_servers.fastwam:FastWAMLiberoServer",
            notes=(
                "Uses FastWAM's LIBERO eval preprocessing; requires LIBERO raw obs.",
                "Not benchmark-agnostic until additional FastWAM benchmark adapters exist.",
            ),
        )

    def __init__(
        self,
        fastwam_root: str = "~/workspace/code/FastWAM",
        checkpoint: str = "~/workspace/code/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224.pt",
        dataset_stats_path: str = (
            "~/workspace/code/FastWAM/checkpoints/fastwam_release/"
            "libero_uncond_2cam224_dataset_stats.json"
        ),
        task_config: str = "libero_uncond_2cam224_1e-4",
        config_name: str = "sim_libero",
        device: str = "cuda",
        mixed_precision: str = "bf16",
        action_horizon: int | None = None,
        num_inference_steps: int = 10,
        observation_params: str | dict[str, Any] | None = None,
        extra_overrides: str | list[str] | None = None,
        *,
        chunk_size: int = 10,
        action_ensemble: str = "newest",
        **kwargs: Any,
    ) -> None:
        super().__init__(chunk_size=chunk_size, action_ensemble=action_ensemble, **kwargs)
        self.fastwam_root = Path(os.path.expanduser(os.path.expandvars(fastwam_root))).resolve()
        self.checkpoint = Path(os.path.expanduser(os.path.expandvars(checkpoint))).resolve()
        self.dataset_stats_path = Path(os.path.expanduser(os.path.expandvars(dataset_stats_path))).resolve()
        self.task_config = task_config
        self.config_name = config_name
        self.device = device
        self.mixed_precision = mixed_precision
        self.action_horizon_override = action_horizon
        self.num_inference_steps = int(num_inference_steps)
        self._extra_obs_params: dict[str, Any] = {}
        self.extra_overrides: list[str] = []
        if observation_params or extra_overrides:
            import json

            if observation_params:
                self._extra_obs_params = (
                    json.loads(observation_params) if isinstance(observation_params, str) else dict(observation_params)
                )
            if extra_overrides:
                if isinstance(extra_overrides, str):
                    parsed = json.loads(extra_overrides) if extra_overrides.strip().startswith("[") else [extra_overrides]
                    self.extra_overrides = [str(item) for item in parsed]
                else:
                    self.extra_overrides = [str(item) for item in extra_overrides]

        self._init_fastwam()

    def get_observation_params(self) -> dict[str, Any]:
        params: dict[str, Any] = dict(self.interface_contract().observation_params)
        params.update(self._extra_obs_params)
        return params

    def get_action_spec(self) -> dict[str, DimSpec]:
        return dict(self.interface_contract().action_spec)

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return dict(self.interface_contract().observation_spec)

    def _init_fastwam(self) -> None:
        if not self.fastwam_root.is_dir():
            raise FileNotFoundError(f"FastWAM checkout not found: {self.fastwam_root}")
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"FastWAM checkpoint not found: {self.checkpoint}")
        if not self.dataset_stats_path.is_file():
            raise FileNotFoundError(f"FastWAM dataset stats not found: {self.dataset_stats_path}")

        # FastWAM's eval module uses both project-root and experiments/libero
        # imports. Preserve those exact imports instead of copying internals.
        sys.path.insert(0, str(self.fastwam_root / "src"))
        sys.path.insert(0, str(self.fastwam_root))
        sys.path.insert(0, str(self.fastwam_root / "experiments" / "libero"))
        os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(self.fastwam_root / "checkpoints"))
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.chdir(self.fastwam_root)

        from vla_eval.benchmarks.libero.benchmark import _ensure_libero_config

        _ensure_libero_config()

        from experiments.libero.eval_libero_single import (
            _load_model_checkpoint,
            _mixed_precision_to_model_dtype,
            _predict_action_chunk,
        )

        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        for name, fn in {
            "eval": eval,
            "max": lambda x: max(x),
            "split": lambda s, idx: s.split("/")[int(idx)],
        }.items():
            if not OmegaConf.has_resolver(name):
                OmegaConf.register_new_resolver(name, fn)

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        overrides = [
            f"task={self.task_config}",
            f"ckpt={self.checkpoint}",
            f"EVALUATION.dataset_stats_path={self.dataset_stats_path}",
            f"EVALUATION.device={self.device}",
            f"EVALUATION.num_inference_steps={self.num_inference_steps}",
            f"mixed_precision={self.mixed_precision}",
        ]
        if self.action_horizon_override is not None:
            overrides.append(f"EVALUATION.action_horizon={int(self.action_horizon_override)}")
        overrides.extend(self.extra_overrides)

        with initialize_config_dir(version_base="1.3", config_dir=str(self.fastwam_root / "configs")):
            cfg = compose(config_name=self.config_name, overrides=overrides)
        self.cfg = cfg

        from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
        from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

        self._predict_action_chunk = _predict_action_chunk
        model_dtype = _mixed_precision_to_model_dtype(self.mixed_precision)
        logger.info("Loading FastWAM model checkpoint=%s stats=%s", self.checkpoint, self.dataset_stats_path)
        self.model = instantiate(cfg.model, model_dtype=model_dtype, device=self.device)
        _load_model_checkpoint(self.model, str(self.checkpoint))
        self.model = self.model.to(self.device).eval()

        dataset_stats = load_dataset_stats_from_json(str(self.dataset_stats_path))
        self.processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
        self.processor.set_normalizer_from_stats(dataset_stats)

        action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
        self.action_horizon = (
            int(cfg.data.train.num_frames) - 1 if action_horizon_cfg is None else int(action_horizon_cfg)
        )
        if self.action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {self.action_horizon}")

        video_size = cfg.data.train.get("video_size", [224, 448])
        self.input_h = int(video_size[0])
        self.input_w = int(video_size[1])
        logger.info(
            "FastWAM ready: task=%s horizon=%d chunk_size=%s input_hw=(%d,%d)",
            self.task_config,
            self.action_horizon,
            self.chunk_size,
            self.input_h,
            self.input_w,
        )

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        raw_obs = obs.get("raw_libero_obs")
        if raw_obs is None:
            raise ValueError(
                "FastWAMLiberoServer requires obs['raw_libero_obs']. "
                "Set LIBEROBenchmark(send_raw_libero_obs=True), or use the provided wam-lab configs."
            )
        task_description = str(obs.get("task_description", ""))
        actions, _, _ = self._predict_action_chunk(
            obs=raw_obs,
            task_description=task_description,
            model=self.model,
            processor=self.processor,
            cfg=self.cfg,
            action_horizon=self.action_horizon,
            input_w=self.input_w,
            input_h=self.input_h,
            model_device=self.device,
        )
        return {"actions": np.asarray(actions, dtype=np.float32)}


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(FastWAMLiberoServer)

"""Lightweight policy/benchmark interface contracts.

Contracts are intentionally independent from model construction: a policy can
declare interface conventions and required payload keys without loading weights.
This lets WAM-Lab audit compatibility before spending GPU memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import Any

from vla_eval.specs import DimSpec, check_specs


@dataclass(frozen=True)
class InterfaceContract:
    """Action/observation convention plus optional real payload-key contract."""

    name: str
    action_spec: dict[str, DimSpec] = field(default_factory=dict)
    observation_spec: dict[str, DimSpec] = field(default_factory=dict)
    observation_params: dict[str, Any] = field(default_factory=dict)
    required_payload_keys: tuple[str, ...] = ()
    produced_payload_keys: tuple[str, ...] = ()
    source: str = ""
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "action_spec": {key: value.to_dict() for key, value in self.action_spec.items()},
            "observation_spec": {key: value.to_dict() for key, value in self.observation_spec.items()},
            "observation_params": dict(self.observation_params),
            "required_payload_keys": list(self.required_payload_keys),
            "produced_payload_keys": list(self.produced_payload_keys),
            "source": self.source,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class ObservationParamMerge:
    """Result of applying server-requested observation params to a benchmark."""

    merged_params: dict[str, Any]
    applied: dict[str, Any]
    ignored: dict[str, Any]
    already_set: dict[str, Any]


@dataclass(frozen=True)
class CompatibilityReport:
    """Compatibility result between a policy contract and a benchmark contract."""

    policy: str
    benchmark: str
    warnings: tuple[str, ...]
    payload_check_skipped: bool = False

    @property
    def compatible(self) -> bool:
        return len(self.warnings) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "benchmark": self.benchmark,
            "compatible": self.compatible,
            "warnings": list(self.warnings),
            "payload_check_skipped": self.payload_check_skipped,
        }


def merge_observation_params(
    benchmark_cls: type,
    explicit_params: dict[str, Any],
    observation_params: dict[str, Any],
) -> ObservationParamMerge:
    """Apply server observation params exactly like the orchestrator does.

    Unknown params are ignored instead of silently added. Explicit benchmark
    config values win over server defaults.
    """

    sig = inspect.signature(benchmark_cls.__init__)
    merged = dict(explicit_params)
    applied: dict[str, Any] = {}
    ignored: dict[str, Any] = {}
    already_set: dict[str, Any] = {}
    for key, value in observation_params.items():
        if key in merged:
            already_set[key] = merged[key]
        elif key in sig.parameters:
            merged[key] = value
            applied[key] = value
        else:
            ignored[key] = value
    return ObservationParamMerge(merged_params=merged, applied=applied, ignored=ignored, already_set=already_set)


def contract_from_endpoint(name: str, endpoint: Any, *, source: str = "") -> InterfaceContract:
    """Build a contract from an instantiated benchmark or model server."""

    obs_params = endpoint.get_observation_params() if hasattr(endpoint, "get_observation_params") else {}
    return InterfaceContract(
        name=name,
        action_spec=endpoint.get_action_spec(),
        observation_spec=endpoint.get_observation_spec(),
        observation_params=obs_params,
        source=source or type(endpoint).__name__,
    )


def check_contracts(policy: InterfaceContract, benchmark: InterfaceContract) -> CompatibilityReport:
    """Compare a policy server contract against a benchmark contract."""

    warnings = check_specs(
        server_action=policy.action_spec,
        bench_action=benchmark.action_spec,
        server_obs=policy.observation_spec,
        bench_obs=benchmark.observation_spec,
    )
    payload_check_skipped = bool(policy.required_payload_keys and not benchmark.produced_payload_keys)
    if policy.required_payload_keys and benchmark.produced_payload_keys:
        produced = set(benchmark.produced_payload_keys)
        for key in policy.required_payload_keys:
            if key not in produced:
                warnings.append(f"payload [{key}]: policy requires it but benchmark observation does not produce it")
    return CompatibilityReport(
        policy=policy.name,
        benchmark=benchmark.name,
        warnings=tuple(warnings),
        payload_check_skipped=payload_check_skipped,
    )

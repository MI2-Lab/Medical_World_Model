"""Lazy public contracts for the preregistered Local--Global pilot.

Keeping package import side-effect free is a security property: command-line
entrypoints can import the preregistration verifier without importing or
executing any pilot/upstream model module before the lock has passed.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "ARMS": ("contracts", "ARMS"),
    "ARM_SPECS": ("contracts", "ARM_SPECS"),
    "ArmSpec": ("contracts", "ArmSpec"),
    "arm_spec": ("contracts", "arm_spec"),
    "LocalGlobalResponseWorldModel": ("model", "LocalGlobalResponseWorldModel"),
    "build_model": ("model", "build_model"),
    "build_objective": ("model", "build_objective"),
    "common_initialization_sha256": ("model", "common_initialization_sha256"),
    "load_checkpoint": ("model", "load_checkpoint"),
    "load_checkpoint_for_evaluation": ("model", "load_checkpoint_for_evaluation"),
    "load_selected_checkpoint": ("model", "load_selected_checkpoint"),
    "paired_initialization_report": ("model", "paired_initialization_report"),
    "shared_initialization_sha256": ("model", "shared_initialization_sha256"),
    "transition_sha256": ("model", "transition_sha256"),
    "validate_model_contract": ("model", "validate_model_contract"),
    "build_fixed_c1b_local_weights": ("pooling", "build_fixed_c1b_local_weights"),
    "derived_final_feature_shape": ("pooling", "derived_final_feature_shape"),
    "expected_feature_shape": ("pooling", "expected_feature_shape"),
    "fixed_physical_local_weights": ("pooling", "fixed_physical_local_weights"),
    "pooling_contract": ("pooling", "pooling_contract"),
    "validate_actual_final_feature": ("pooling", "validate_actual_final_feature"),
    "weighted_average_pool": ("pooling", "weighted_average_pool"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(f".{module_name}", __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


__all__ = [
    "ARMS",
    "ARM_SPECS",
    "ArmSpec",
    "LocalGlobalResponseWorldModel",
    "arm_spec",
    "build_fixed_c1b_local_weights",
    "build_model",
    "build_objective",
    "common_initialization_sha256",
    "derived_final_feature_shape",
    "expected_feature_shape",
    "fixed_physical_local_weights",
    "load_checkpoint",
    "load_checkpoint_for_evaluation",
    "load_selected_checkpoint",
    "paired_initialization_report",
    "pooling_contract",
    "shared_initialization_sha256",
    "transition_sha256",
    "validate_actual_final_feature",
    "validate_model_contract",
    "weighted_average_pool",
]

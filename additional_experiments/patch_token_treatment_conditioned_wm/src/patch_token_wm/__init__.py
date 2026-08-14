"""Core API for A1, with lazy torch-dependent imports.

Evaluation/data utilities must remain importable in CPU analysis environments
that do not install PyTorch.  PEP 562 lookup below preserves the convenient
top-level API while importing contracts/model/objective only when requested.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MODULE = {
    **dict.fromkeys(
        (
            "C1B_INPUT_SHAPE_ZYX",
            "C1B_SPACING_XYZ_MM",
            "CLINICAL_FEATURES",
            "FIXED_ARM_VOCAB",
            "FORMAL_LOCAL_TOKEN_COUNT",
            "FORMAL_MASKED_TOKEN_COUNT",
            "G3_MODEL_SHA256",
            "NOMINAL_TEMPORAL_BITS",
            "SpatialVisitEncoder3D",
            "TransitionCondition",
            "nominal_temporal_bits",
            "source_contract",
        ),
        "contracts",
    ),
    **dict.fromkeys(
        (
            "LocalTokenGeometry",
            "build_local_token_geometry",
            "derived_feature_shape",
            "feature_cell_coordinates_xyz_mm",
            "gather_local_tokens",
            "sinusoidal_physical_position_encoding",
            "weighted_local_mean",
        ),
        "geometry",
    ),
    **dict.fromkeys(
        (
            "A1PatchTokenWorldModel",
            "ConditionTokenEncoder",
            "MaskedFutureTokenPredictor",
            "PatchTokenJEPA",
            "PatchTokenOutput",
            "PatchTokenWorldModel",
            "deterministic_mask_indices",
            "source_to_query_block_mask",
        ),
        "model",
    ),
    **dict.fromkeys(
        (
            "A1Objective",
            "PatchTokenObjective",
            "SIGReg",
            "normalized_masked_token_mse",
            "patient_mean_ftv_smooth_l1",
        ),
        "objective",
    ),
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORT_MODULE))


__all__ = [
    "A1Objective",
    "A1PatchTokenWorldModel",
    "C1B_INPUT_SHAPE_ZYX",
    "C1B_SPACING_XYZ_MM",
    "CLINICAL_FEATURES",
    "ConditionTokenEncoder",
    "FIXED_ARM_VOCAB",
    "FORMAL_LOCAL_TOKEN_COUNT",
    "FORMAL_MASKED_TOKEN_COUNT",
    "G3_MODEL_SHA256",
    "LocalTokenGeometry",
    "MaskedFutureTokenPredictor",
    "NOMINAL_TEMPORAL_BITS",
    "PatchTokenJEPA",
    "PatchTokenObjective",
    "PatchTokenOutput",
    "PatchTokenWorldModel",
    "SIGReg",
    "SpatialVisitEncoder3D",
    "TransitionCondition",
    "build_local_token_geometry",
    "derived_feature_shape",
    "deterministic_mask_indices",
    "feature_cell_coordinates_xyz_mm",
    "gather_local_tokens",
    "nominal_temporal_bits",
    "normalized_masked_token_mse",
    "patient_mean_ftv_smooth_l1",
    "sinusoidal_physical_position_encoding",
    "source_contract",
    "source_to_query_block_mask",
    "weighted_local_mean",
]

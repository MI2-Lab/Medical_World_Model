"""Frozen arm and geometry contracts for the Local--Global response pilot."""

from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import Any, Mapping, Sequence


ARMS = ("GAP0", "GAP3", "LOCAL0", "LOCAL3", "LG0", "LG3")
ARCHITECTURES = ("GAP", "LOCAL", "LOCAL_GLOBAL")

C1B_INPUT_SHAPE_ZYX = (112, 176, 160)
C1B_SPACING_XYZ_MM = (0.9, 0.9, 2.0)
LOCAL_WINDOW_MM_XYZ = (64.0, 64.0, 64.0)
IMAGE_CHANNELS = 7
FINAL_FEATURE_CHANNELS = 128
RESPONSE_DIM = 192

MODEL_KWARGS: Mapping[str, Any] = {
    "image_channels": IMAGE_CHANNELS,
    "base_channels": 16,
    "latent_dim": RESPONSE_DIM,
    "predictor_depth": 3,
    "predictor_heads": 4,
    "predictor_mlp_dim": 512,
    "dropout": 0.1,
}


@dataclass(frozen=True)
class ArmSpec:
    """One preregistered response-state arm."""

    name: str
    architecture: str
    grounded: bool
    upstream_model_name: str

    def __post_init__(self) -> None:
        if self.name not in ARMS:
            raise ValueError(f"arm name must be one of {ARMS}")
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {ARCHITECTURES}")
        expected_model = "G3" if self.grounded else "G1"
        if self.upstream_model_name != expected_model:
            raise ValueError("upstream model name and grounding status disagree")


ARM_SPECS: Mapping[str, ArmSpec] = {
    "GAP0": ArmSpec("GAP0", "GAP", False, "G1"),
    "GAP3": ArmSpec("GAP3", "GAP", True, "G3"),
    "LOCAL0": ArmSpec("LOCAL0", "LOCAL", False, "G1"),
    "LOCAL3": ArmSpec("LOCAL3", "LOCAL", True, "G3"),
    "LG0": ArmSpec("LG0", "LOCAL_GLOBAL", False, "G1"),
    "LG3": ArmSpec("LG3", "LOCAL_GLOBAL", True, "G3"),
}


def arm_spec(arm: str) -> ArmSpec:
    name = str(arm).strip().upper()
    if name not in ARM_SPECS:
        raise ValueError(f"pilot arm must be one of {ARMS}; got {arm!r}")
    return ARM_SPECS[name]


def _exact_int_triplet(values: Sequence[int], expected: tuple[int, int, int], name: str) -> tuple[int, int, int]:
    if isinstance(values, (str, bytes)) or len(values) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    try:
        parsed = tuple(operator.index(value) for value in values)
    except TypeError as exc:
        raise ValueError(f"{name} must contain exact integers") from exc
    if any(isinstance(value, bool) for value in values):
        raise ValueError(f"{name} must contain exact integers")
    if parsed != expected:
        raise ValueError(f"{name} is frozen to {expected}; got {parsed}")
    return parsed


def _exact_float_triplet(
    values: Sequence[float], expected: tuple[float, float, float], name: str
) -> tuple[float, float, float]:
    if isinstance(values, (str, bytes)) or len(values) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    parsed = tuple(float(value) for value in values)
    if parsed != expected:
        raise ValueError(f"{name} is frozen to {expected}; got {parsed}")
    return parsed


def validate_c1b_geometry(
    input_shape_zyx: Sequence[int],
    spacing_xyz_mm: Sequence[float],
    local_window_mm_xyz: Sequence[float],
) -> tuple[tuple[int, int, int], tuple[float, float, float], tuple[float, float, float]]:
    """Reject any geometry/window drift from the preregistered C1B-H contract."""

    return (
        _exact_int_triplet(input_shape_zyx, C1B_INPUT_SHAPE_ZYX, "input_shape_zyx"),
        _exact_float_triplet(spacing_xyz_mm, C1B_SPACING_XYZ_MM, "spacing_xyz_mm"),
        _exact_float_triplet(
            local_window_mm_xyz, LOCAL_WINDOW_MM_XYZ, "local_window_mm_xyz"
        ),
    )


__all__ = [
    "ARCHITECTURES",
    "ARMS",
    "ARM_SPECS",
    "ArmSpec",
    "C1B_INPUT_SHAPE_ZYX",
    "C1B_SPACING_XYZ_MM",
    "FINAL_FEATURE_CHANNELS",
    "IMAGE_CHANNELS",
    "LOCAL_WINDOW_MM_XYZ",
    "MODEL_KWARGS",
    "RESPONSE_DIM",
    "arm_spec",
    "validate_c1b_geometry",
]

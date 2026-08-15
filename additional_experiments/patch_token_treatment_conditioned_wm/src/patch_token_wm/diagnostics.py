"""Outcome-free token-dynamics diagnostics for the patch-token pilot."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np


SHUFFLED_TRANSITION_ORDER = (1, 2, 0)  # [T1,T2,T3] -> [T2,T3,T1]


def _finite_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name} must be nonempty and finite")
    return array


def token_cosine(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = _finite_array(prediction, "prediction")
    target = _finite_array(target, "target")
    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError("prediction and target must share [...,token_dim] shape")
    numerator = np.sum(prediction * target, axis=-1)
    denominator = np.linalg.norm(prediction, axis=-1) * np.linalg.norm(target, axis=-1)
    cosine = numerator / np.maximum(denominator, np.finfo(np.float64).eps)
    return float(np.mean(cosine))


def normalized_token_mse(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = _finite_array(prediction, "prediction")
    target = _finite_array(target, "target")
    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError("prediction and target must share [...,token_dim] shape")

    def normalize(value: np.ndarray) -> np.ndarray:
        centered = value - value.mean(axis=-1, keepdims=True)
        scale = np.sqrt(np.mean(centered**2, axis=-1, keepdims=True) + 1e-6)
        return centered / scale

    return float(np.mean((normalize(prediction) - normalize(target)) ** 2))


def token_standard_deviation(value: np.ndarray) -> float:
    array = _finite_array(value, "tokens")
    if array.ndim < 3:
        raise ValueError("tokens must include sample, token, and channel axes")
    flattened = array.reshape(-1, array.shape[-2], array.shape[-1])
    return float(np.std(flattened, axis=(0, 1), ddof=0).mean())


def cyclic_shuffled_targets(target_by_transition: np.ndarray) -> np.ndarray:
    target = _finite_array(target_by_transition, "target_by_transition")
    if target.ndim < 3 or target.shape[1] != 3:
        raise ValueError("target array must have transition axis 1 of length three")
    return target[:, SHUFFLED_TRANSITION_ORDER, ...]


def spatial_band_labels(coords_xyz_mm: np.ndarray) -> np.ndarray:
    coords = _finite_array(coords_xyz_mm, "coords_xyz_mm")
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("physical coordinates must be [tokens,3] XYZ millimetres")
    radius = np.max(np.abs(coords), axis=1)
    labels = np.full(len(coords), "outer_local", dtype="U16")
    labels[radius <= 24.0] = "inner_local"
    labels[radius <= 16.0] = "central"
    if set(labels) != {"central", "inner_local", "outer_local"}:
        raise ValueError("formal coordinates must populate all three diagnostic bands")
    return labels


def spatial_error_summary(
    prediction: np.ndarray,
    target: np.ndarray,
    mask_indices: np.ndarray,
    coords_xyz_mm: np.ndarray,
) -> dict[str, float]:
    """Return mean normalized-token error by fixed coordinate band.

    Arrays are expected as `[patients,3,masked,dim]` and mask indices as
    `[patients,3,masked]`.  The result is aggregate and carries no identifiers.
    """

    prediction = _finite_array(prediction, "prediction")
    target = _finite_array(target, "target")
    indices = np.asarray(mask_indices)
    coords = _finite_array(coords_xyz_mm, "coords_xyz_mm")
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction/target must share [patients,3,masked,dim]")
    if indices.shape != prediction.shape[:-1]:
        raise ValueError("mask indices must match prediction without channel axis")
    if (
        indices.dtype.kind not in "iu"
        or np.any(indices < 0)
        or np.any(indices >= len(coords))
    ):
        raise ValueError("mask indices are outside the coordinate table")
    if prediction.shape[1] != 3:
        raise ValueError("diagnostic requires exactly three transitions")

    def normalize(value: np.ndarray) -> np.ndarray:
        centered = value - value.mean(axis=-1, keepdims=True)
        return centered / np.sqrt(np.mean(centered**2, axis=-1, keepdims=True) + 1e-6)

    error = np.mean((normalize(prediction) - normalize(target)) ** 2, axis=-1)
    labels = spatial_band_labels(coords)[indices]
    output: dict[str, float] = {}
    for transition, visit in enumerate(("T1", "T2", "T3")):
        for band in ("central", "inner_local", "outer_local"):
            selected = error[:, transition][labels[:, transition] == band]
            output[f"{visit}_{band}_normalized_mse"] = (
                float(np.mean(selected)) if selected.size else math.nan
            )
            output[f"{visit}_{band}_token_predictions"] = int(selected.size)
    return output


@dataclass(frozen=True)
class DynamicsSummary:
    actual_cosine: float
    shuffled_cosine: float
    actual_normalized_mse: float
    shuffled_normalized_mse: float
    target_std: float
    prediction_std: float

    @property
    def cosine_gain(self) -> float:
        return self.actual_cosine - self.shuffled_cosine

    @property
    def normalized_mse_relative_improvement(self) -> float:
        if self.shuffled_normalized_mse <= 0:
            return math.nan
        return (
            self.shuffled_normalized_mse - self.actual_normalized_mse
        ) / self.shuffled_normalized_mse

    def to_dict(self) -> dict[str, float]:
        return {
            "actual_cosine": self.actual_cosine,
            "shuffled_cosine": self.shuffled_cosine,
            "cosine_gain": self.cosine_gain,
            "actual_normalized_mse": self.actual_normalized_mse,
            "shuffled_normalized_mse": self.shuffled_normalized_mse,
            "normalized_mse_relative_improvement": self.normalized_mse_relative_improvement,
            "target_std": self.target_std,
            "prediction_std": self.prediction_std,
        }


def summarize_dynamics(
    prediction: np.ndarray,
    target: np.ndarray,
    shuffled_target: np.ndarray,
) -> DynamicsSummary:
    prediction = _finite_array(prediction, "prediction")
    target = _finite_array(target, "target")
    shuffled = _finite_array(shuffled_target, "shuffled_target")
    if prediction.shape != target.shape or target.shape != shuffled.shape:
        raise ValueError(
            "actual, prediction, and shuffled tensors must have identical shape"
        )
    return DynamicsSummary(
        actual_cosine=token_cosine(prediction, target),
        shuffled_cosine=token_cosine(prediction, shuffled),
        actual_normalized_mse=normalized_token_mse(prediction, target),
        shuffled_normalized_mse=normalized_token_mse(prediction, shuffled),
        target_std=token_standard_deviation(target),
        prediction_std=token_standard_deviation(prediction),
    )


def aggregate_seed_dynamics(cell_rows: list[Mapping[str, float]]) -> dict[str, float]:
    if not cell_rows:
        raise ValueError("seed dynamics aggregation requires at least one fold")
    required = tuple(DynamicsSummary.__dataclass_fields__)
    output = {
        name: float(np.mean([float(row[name]) for row in cell_rows]))
        for name in required
    }
    summary = DynamicsSummary(**output)
    return {**summary.to_dict(), "folds": len(cell_rows)}


__all__ = [
    "DynamicsSummary",
    "SHUFFLED_TRANSITION_ORDER",
    "aggregate_seed_dynamics",
    "cyclic_shuffled_targets",
    "normalized_token_mse",
    "spatial_band_labels",
    "spatial_error_summary",
    "summarize_dynamics",
    "token_cosine",
    "token_standard_deviation",
]

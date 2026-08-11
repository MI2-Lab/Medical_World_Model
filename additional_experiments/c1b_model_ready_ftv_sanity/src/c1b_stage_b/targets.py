"""FTV-only grounding targets and literal natural-scale change targets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .data import FTVRecord


def grounding_raw_map(
    records: Mapping[str, FTVRecord],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Expose only value and measurement-valid/observable loss eligibility."""

    return {
        patient_id: (
            np.asarray(record.values, dtype=np.float64),
            np.asarray(record.grounding_eligible, dtype=bool),
        )
        for patient_id, record in records.items()
    }


def fit_grounding_transform(
    records: Mapping[str, FTVRecord],
    outer_train_ids: Iterable[str],
    fold: int,
    output_path: str | Path | None = None,
    apply_ids: Iterable[str] | None = None,
) -> tuple[Any, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Fit upstream transform on outer-train observable measurements only."""

    from .upstream import PooledFTVTransform

    train_ids = tuple(str(value) for value in outer_train_ids)
    raw = grounding_raw_map(records)
    fit_raw = {patient_id: raw[patient_id] for patient_id in train_ids if patient_id in raw}
    transform = PooledFTVTransform.fit(fit_raw, train_ids, int(fold))
    selected_ids = tuple(records) if apply_ids is None else tuple(str(value) for value in apply_ids)
    transformed = {
        patient_id: transform.transform_values(*raw[patient_id])
        for patient_id in selected_ids
        if patient_id in raw
    }
    if output_path is not None:
        transform.save(output_path)
    return transform, transformed


def literal_delta_targets(
    values: np.ndarray,
    measurement_valid: np.ndarray,
    observable: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return literal ``FTV[t+1] - FTV[t]`` without log transformation."""

    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(measurement_valid, dtype=bool) & np.isfinite(values)
    if values.shape != (4,) or valid.shape != (4,):
        raise ValueError("literal delta requires four visit values and validity flags")
    if observable is not None:
        observable = np.asarray(observable, dtype=bool)
        if observable.shape != (4,):
            raise ValueError("observable mask must have shape [4]")
        valid &= observable
    delta = values[1:] - values[:-1]
    delta_valid = valid[1:] & valid[:-1] & np.isfinite(delta)
    return delta.astype(np.float64, copy=False), delta_valid


def static_targets(
    record: FTVRecord, *, observable_only: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.asarray(record.measurement_valid, dtype=bool) & np.isfinite(record.values)
    if observable_only:
        valid &= np.asarray(record.observable, dtype=bool)
    return np.asarray(record.values, dtype=np.float64), valid


__all__ = [
    "fit_grounding_transform",
    "grounding_raw_map",
    "literal_delta_targets",
    "static_targets",
]

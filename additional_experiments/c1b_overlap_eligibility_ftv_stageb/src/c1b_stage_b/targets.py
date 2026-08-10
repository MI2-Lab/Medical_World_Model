"""FTV-only grounding targets and literal natural-scale change targets."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
import sys
import types
from typing import Any, Iterable, Mapping

import numpy as np

from .data import FTVRecord
from .contracts import G3_SRC, LOCKED_G3_TARGETS_SHA256, file_sha256


def _pooled_ftv_transform_class() -> Any:
    """Load only frozen G3 target code, without importing its Torch model package."""

    package_name = "_c1b_stage_b_frozen_dgrs_targets"
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = [str((G3_SRC / "dgrs").resolve())]
        package.__package__ = package_name
        sys.modules[package_name] = package
    module = importlib.import_module(f"{package_name}.targets")
    expected = (G3_SRC / "dgrs" / "targets.py").resolve()
    if Path(inspect.getfile(module.PooledFTVTransform)).resolve() != expected:
        raise ImportError("PooledFTVTransform did not resolve to frozen G3 target code")
    if file_sha256(expected) != LOCKED_G3_TARGETS_SHA256:
        raise ImportError("frozen G3 FTV target implementation hash drifted")
    return module.PooledFTVTransform


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


def static_raw_map(
    records: Mapping[str, FTVRecord], *, observable_only: bool
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    output: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for patient_id, record in records.items():
        valid = np.asarray(record.measurement_valid, dtype=bool)
        if observable_only:
            valid &= np.asarray(record.observable, dtype=bool)
        output[patient_id] = (
            np.asarray(record.values, dtype=np.float64),
            valid,
        )
    return output


def _validate_frozen_transform(transform: Any, fold: int) -> None:
    if int(transform.fold) != int(fold):
        raise ValueError("FTV transform fold identity drifted")
    if tuple(float(value) for value in transform.winsor_quantiles) != (0.01, 0.99):
        raise ValueError("FTV transform must use frozen 1/99 winsorization")
    if not np.isfinite(float(transform.epsilon)) or float(transform.epsilon) <= 0:
        raise ValueError("FTV transform epsilon must be positive and outer-train fitted")
    if not np.isfinite(float(transform.scale_iqr)) or float(transform.scale_iqr) <= 0:
        raise ValueError("FTV transform must use a positive outer-train IQR scale")


def fit_static_probe_transform(
    records: Mapping[str, FTVRecord],
    outer_train_ids: Iterable[str],
    fold: int,
) -> Any:
    """Fit the exact grounding transform on outer-train observable visits only."""

    train_ids = tuple(str(value) for value in outer_train_ids)
    raw: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for patient_id in train_ids:
        record = records.get(patient_id)
        if record is None:
            continue
        raw[patient_id] = (
            np.asarray(record.values, dtype=np.float64),
            np.asarray(record.grounding_eligible, dtype=bool),
        )
    transform = _pooled_ftv_transform_class().fit(
        raw, train_ids, int(fold), quantiles=(0.01, 0.99)
    )
    _validate_frozen_transform(transform, fold)
    return transform


def fit_grounding_transform(
    records: Mapping[str, FTVRecord],
    outer_train_ids: Iterable[str],
    fold: int,
    output_path: str | Path | None = None,
    apply_ids: Iterable[str] | None = None,
) -> tuple[Any, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Fit upstream transform on outer-train observable measurements only."""

    train_ids = tuple(str(value) for value in outer_train_ids)
    raw = grounding_raw_map(records)
    fit_raw = {patient_id: raw[patient_id] for patient_id in train_ids if patient_id in raw}
    transform = _pooled_ftv_transform_class().fit(
        fit_raw, train_ids, int(fold), quantiles=(0.01, 0.99)
    )
    _validate_frozen_transform(transform, fold)
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
    "fit_static_probe_transform",
    "grounding_raw_map",
    "literal_delta_targets",
    "static_targets",
    "static_raw_map",
]

"""Fold-safe static SPH targets for the residual-grounding pilot.

This module is intentionally outcome-blind.  It reads only the patient/fold
columns needed to recover the frozen outer splits and only the FTV/SPH columns
from the authenticated radiomics transition table.  In particular, it has no
clinical-table or pCR dependency.

The residual convention is the one frozen by the preceding non-FTV audit.  For
each outer fold and visit independently:

1. fit 1st/99th percentile winsor limits on outer-train patients;
2. use identity-transformed SPH and log1p-transformed FTV, then population-z
   standardize both using outer-train statistics;
3. fit ``Ridge(alpha=1, fit_intercept=True)`` from FTV-z to SPH-z;
4. define epsilon as observed SPH-z minus the fitted conditional value; and
5. population-z standardize epsilon using outer-train patients.

Thus S1 is SPH-z and S2 is the second, residual z-coordinate.  Validation and
test values are transformed only after every fitted quantity is frozen.
Identifier-bearing arrays remain in memory.  Public serialization contains
only aggregate counts, transforms, and fitted coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


VISITS: tuple[str, ...] = ("T0", "T1", "T2", "T3")
SPH_FIELDS: tuple[str, ...] = tuple(f"SPHERICITY_{visit}" for visit in VISITS)
FTV_FIELDS: tuple[str, ...] = tuple(f"FTV_{visit}" for visit in VISITS)
ADJACENT_VISITS: tuple[tuple[str, str], ...] = (
    ("T0", "T1"),
    ("T1", "T2"),
    ("T2", "T3"),
)
ALLOWED_SPLITS = frozenset(("train", "val", "test"))
DEFAULT_WINSOR_QUANTILES: tuple[float, float] = (0.01, 0.99)
DEFAULT_RESIDUALIZER_ALPHA = 1.0

_TARGET_COLUMNS = frozenset(
    {
        "patient_id",
        "trial_id",
        "transition",
        "start_visit",
        "end_visit",
        "ftv_start",
        "ftv_end",
        "ftv_valid",
        "sphericity_start",
        "sphericity_end",
        "sphericity_valid",
    }
)
_FOLD_COLUMNS = frozenset({"patient_id", "fold", "split"})
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {"patient_id", "patient_ids", "trial_id", "trial_ids", "label_pcr", "pcr"}
)


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 of a file without interpreting its contents."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def authenticate_file(
    path: str | Path,
    expected_sha256: str,
    *,
    label: str,
) -> tuple[Path, str]:
    """Resolve and authenticate an immutable input, failing closed on drift."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing {label}: {resolved}")
    expected = str(expected_sha256).strip().lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError(f"invalid expected SHA-256 for {label}")
    observed = file_sha256(resolved)
    if observed != expected:
        raise ValueError(
            f"{label} SHA-256 drift: expected {expected}, observed {observed}"
        )
    return resolved, observed


def _readonly_float_array(values: Any, *, ndim: int, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != ndim:
        raise ValueError(f"{label} must be {ndim}-dimensional")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values")
    array = np.array(array, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


def _coerce_strict_bool(series: pd.Series, *, label: str) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.to_numpy(dtype=bool)
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    unexpected = sorted(set(normalized) - set(mapping))
    if unexpected:
        raise ValueError(f"{label} contains non-boolean values: {unexpected[:3]}")
    return normalized.map(mapping).to_numpy(dtype=bool)


def _assign_endpoint(
    destination: np.ndarray,
    row_index: int,
    visit_index: int,
    value: float,
    *,
    label: str,
) -> None:
    if not math.isfinite(value):
        raise ValueError(f"non-finite endpoint for {label}")
    previous = destination[row_index, visit_index]
    if np.isfinite(previous) and not np.isclose(previous, value, rtol=0.0, atol=1e-12):
        raise ValueError(f"inconsistent repeated endpoint for {label}")
    destination[row_index, visit_index] = float(value)


@dataclass(frozen=True)
class StaticTargetTable:
    """Identifier-aligned natural-unit FTV and SPH matrices held in memory."""

    patient_ids: tuple[str, ...]
    sphericity: np.ndarray
    ftv: np.ndarray
    source_sha256: str

    def __post_init__(self) -> None:
        if not self.patient_ids or len(set(self.patient_ids)) != len(self.patient_ids):
            raise ValueError("patient IDs must be non-empty and unique")
        sphericity = _readonly_float_array(self.sphericity, ndim=2, label="sphericity")
        ftv = _readonly_float_array(self.ftv, ndim=2, label="ftv")
        expected_shape = (len(self.patient_ids), len(VISITS))
        if sphericity.shape != expected_shape or ftv.shape != expected_shape:
            raise ValueError(f"target matrices must have shape {expected_shape}")
        if np.any(ftv < 0.0):
            raise ValueError("FTV must be non-negative for the frozen log1p transform")
        object.__setattr__(self, "sphericity", sphericity)
        object.__setattr__(self, "ftv", ftv)

    @property
    def n_patients(self) -> int:
        return len(self.patient_ids)

    @property
    def patient_to_index(self) -> Mapping[str, int]:
        return {patient_id: index for index, patient_id in enumerate(self.patient_ids)}


def load_static_sph_ftv_table(
    path: str | Path,
    expected_sha256: str,
    *,
    expected_patient_count: int | None = 375,
    expected_patient_ids: Sequence[str] | None = None,
) -> StaticTargetTable:
    """Authenticate and reconstruct SPHERICITY_T0..T3 and FTV_T0..T3.

    Only the allowlisted transition-table columns are materialized.  Repeated
    interval endpoints must agree exactly up to 1e-12.
    """

    resolved, observed_sha256 = authenticate_file(
        path, expected_sha256, label="radiomics transition target table"
    )
    frame = pd.read_csv(resolved, usecols=lambda name: name in _TARGET_COLUMNS)
    missing = _TARGET_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"radiomics target schema is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("radiomics target table is empty")

    frame = frame.copy()
    frame["patient_id"] = frame["patient_id"].astype(str)
    if frame["patient_id"].str.len().eq(0).any():
        raise ValueError("radiomics target table contains an empty patient ID")
    if not _coerce_strict_bool(frame["ftv_valid"], label="ftv_valid").all():
        raise ValueError("radiomics target table contains an invalid FTV endpoint")
    if not _coerce_strict_bool(
        frame["sphericity_valid"], label="sphericity_valid"
    ).all():
        raise ValueError("radiomics target table contains an invalid SPH endpoint")

    patient_ids = tuple(sorted(frame["patient_id"].unique().tolist()))
    if expected_patient_count is not None and len(patient_ids) != int(expected_patient_count):
        raise ValueError(
            "radiomics target patient-count drift: "
            f"expected {int(expected_patient_count)}, observed {len(patient_ids)}"
        )
    if expected_patient_ids is not None:
        expected_set = {str(value) for value in expected_patient_ids}
        if set(patient_ids) != expected_set:
            raise ValueError("radiomics target patient set differs from the frozen population")

    expected_rows = len(patient_ids) * len(ADJACENT_VISITS)
    if len(frame) != expected_rows:
        raise ValueError(
            f"radiomics target table must contain exactly {expected_rows} adjacent rows"
        )
    trial_counts = frame.groupby("patient_id", sort=False)["trial_id"].nunique(dropna=False)
    if not (trial_counts == 1).all():
        raise ValueError("patient-to-trial mapping is not one-to-one")

    patient_to_index = {patient_id: index for index, patient_id in enumerate(patient_ids)}
    sphericity = np.full((len(patient_ids), len(VISITS)), np.nan, dtype=np.float64)
    ftv = np.full_like(sphericity, np.nan)
    observed_pairs: dict[str, set[tuple[str, str]]] = {
        patient_id: set() for patient_id in patient_ids
    }

    for row in frame.itertuples(index=False):
        patient_id = str(row.patient_id)
        start_visit = str(row.start_visit)
        end_visit = str(row.end_visit)
        pair = (start_visit, end_visit)
        if pair not in ADJACENT_VISITS:
            raise ValueError(f"non-adjacent or unknown visit pair: {pair}")
        if pair in observed_pairs[patient_id]:
            raise ValueError(f"duplicate transition row for {patient_id}/{pair}")
        observed_pairs[patient_id].add(pair)
        start_index = VISITS.index(start_visit)
        end_index = VISITS.index(end_visit)
        row_index = patient_to_index[patient_id]
        _assign_endpoint(
            sphericity,
            row_index,
            start_index,
            float(row.sphericity_start),
            label=f"{patient_id}/SPH/{start_visit}",
        )
        _assign_endpoint(
            sphericity,
            row_index,
            end_index,
            float(row.sphericity_end),
            label=f"{patient_id}/SPH/{end_visit}",
        )
        _assign_endpoint(
            ftv,
            row_index,
            start_index,
            float(row.ftv_start),
            label=f"{patient_id}/FTV/{start_visit}",
        )
        _assign_endpoint(
            ftv,
            row_index,
            end_index,
            float(row.ftv_end),
            label=f"{patient_id}/FTV/{end_visit}",
        )

    required_pairs = set(ADJACENT_VISITS)
    if any(pairs != required_pairs for pairs in observed_pairs.values()):
        raise ValueError("one or more patients do not have exactly the three adjacent transitions")
    if not np.isfinite(sphericity).all() or not np.isfinite(ftv).all():
        raise ValueError("reconstructed SPH/FTV matrices are incomplete")

    return StaticTargetTable(
        patient_ids=patient_ids,
        sphericity=sphericity,
        ftv=ftv,
        source_sha256=observed_sha256,
    )


def load_fold_split_map(
    path: str | Path,
    expected_sha256: str,
    *,
    fold: int,
    expected_patient_ids: Sequence[str] | None = None,
) -> dict[str, str]:
    """Read only patient/fold/split from the authenticated frozen manifest.

    The source manifest may physically contain an outcome column, but the
    allowlist passed to pandas prevents that column from being materialized.
    """

    resolved, _ = authenticate_file(path, expected_sha256, label="outer-fold manifest")
    frame = pd.read_csv(resolved, usecols=lambda name: name in _FOLD_COLUMNS)
    missing = _FOLD_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"outer-fold manifest is missing columns: {sorted(missing)}")
    frame = frame.loc[pd.to_numeric(frame["fold"], errors="raise") == int(fold)].copy()
    if frame.empty:
        raise ValueError(f"outer-fold manifest has no rows for fold {fold}")
    frame["patient_id"] = frame["patient_id"].astype(str)
    if frame["patient_id"].duplicated().any():
        raise ValueError(f"outer-fold manifest has duplicate patients in fold {fold}")
    frame["split"] = frame["split"].astype(str).str.strip().str.lower()
    unexpected = sorted(set(frame["split"]) - ALLOWED_SPLITS)
    if unexpected:
        raise ValueError(f"outer-fold manifest contains unknown splits: {unexpected}")
    split_map = dict(zip(frame["patient_id"], frame["split"], strict=True))
    if expected_patient_ids is not None:
        expected = {str(value) for value in expected_patient_ids}
        missing_patients = expected - set(split_map)
        if missing_patients:
            raise ValueError(
                f"outer-fold manifest is missing {len(missing_patients)} target patients"
            )
        split_map = {patient_id: split_map[patient_id] for patient_id in expected}
    return split_map


@dataclass(frozen=True)
class FittedTargetTransform:
    lower: float
    upper: float
    mean: float
    scale: float
    log1p: bool

    def transform_unscaled(self, values: Any) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if not np.isfinite(array).all():
            raise ValueError("target transform received non-finite values")
        clipped = np.clip(array, self.lower, self.upper)
        if self.log1p:
            if np.any(clipped <= -1.0):
                raise ValueError("log1p target contains a value <= -1 after clipping")
            clipped = np.log1p(clipped)
        return clipped

    def transform(self, values: Any) -> np.ndarray:
        return (self.transform_unscaled(values) - self.mean) / self.scale

    def inverse_standardized(self, values: Any) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        transformed = array * self.scale + self.mean
        return np.expm1(transformed) if self.log1p else transformed

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "winsor_lower": float(self.lower),
            "winsor_upper": float(self.upper),
            "family_transform": "log1p" if self.log1p else "identity",
            "train_mean_after_family_transform": float(self.mean),
            "train_population_scale": float(self.scale),
        }


def fit_target_transform(
    values: Any,
    fit_mask: Any,
    *,
    log1p: bool,
    quantiles: Sequence[float] = DEFAULT_WINSOR_QUANTILES,
) -> FittedTargetTransform:
    """Fit the exact train-winsor/family-transform/population-z convention."""

    array = np.asarray(values, dtype=np.float64)
    mask = np.asarray(fit_mask, dtype=bool)
    if array.ndim != 1 or mask.shape != array.shape:
        raise ValueError("target values and fit mask must be aligned one-dimensional arrays")
    mask = mask & np.isfinite(array)
    if int(mask.sum()) < 3:
        raise ValueError("too few outer-train rows for a target transform")
    quantile_array = np.asarray(tuple(quantiles), dtype=np.float64)
    if quantile_array.shape != (2,) or not np.allclose(
        quantile_array, np.asarray(DEFAULT_WINSOR_QUANTILES), rtol=0.0, atol=0.0
    ):
        raise ValueError("the frozen target convention requires 1st/99th quantiles")
    lower, upper = np.quantile(array[mask], quantile_array)
    clipped = np.clip(array[mask], lower, upper)
    if log1p:
        if np.any(clipped <= -1.0):
            raise ValueError("outer-train FTV is invalid for log1p")
        clipped = np.log1p(clipped)
    mean = float(np.mean(clipped))
    scale = float(np.std(clipped, ddof=0))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("outer-train target is constant or has invalid scale")
    return FittedTargetTransform(
        lower=float(lower),
        upper=float(upper),
        mean=mean,
        scale=scale,
        log1p=bool(log1p),
    )


@dataclass(frozen=True)
class VisitResidualizer:
    fold: int
    visit: str
    sph_transform: FittedTargetTransform
    ftv_transform: FittedTargetTransform
    alpha: float
    coefficient: float
    intercept: float
    residual_center: float
    residual_scale: float
    n_train: int

    def conditional_sph_z(self, ftv: Any) -> np.ndarray:
        return self.intercept + self.coefficient * self.ftv_transform.transform(ftv)

    def s1_target(self, sphericity: Any) -> np.ndarray:
        return self.sph_transform.transform(sphericity)

    def s2_target(self, sphericity: Any, ftv: Any) -> np.ndarray:
        epsilon = self.s1_target(sphericity) - self.conditional_sph_z(ftv)
        return (epsilon - self.residual_center) / self.residual_scale

    def inverse_s1_target(self, prediction: Any) -> np.ndarray:
        return self.sph_transform.inverse_standardized(prediction)

    def reconstruct_sphericity(self, residual_prediction: Any, ftv: Any) -> np.ndarray:
        residual_z = np.asarray(residual_prediction, dtype=np.float64)
        epsilon = residual_z * self.residual_scale + self.residual_center
        sph_z = self.conditional_sph_z(ftv) + epsilon
        return self.sph_transform.inverse_standardized(sph_z)

    def _payload_without_id(self) -> dict[str, Any]:
        return {
            "fold": int(self.fold),
            "visit": self.visit,
            "fit_scope": "outer_train_only",
            "n_train": int(self.n_train),
            "winsor_quantiles": [0.01, 0.99],
            "sph_transform": self.sph_transform.to_public_dict(),
            "ftv_transform": self.ftv_transform.to_public_dict(),
            "model": "sklearn.linear_model.Ridge",
            "alpha": float(self.alpha),
            "fit_intercept": True,
            "predictor": "FTV_train_winsorized_log1p_population_z",
            "response": "SPH_train_winsorized_identity_population_z",
            "coefficient": [float(self.coefficient)],
            "intercept": float(self.intercept),
            "residual_definition": "epsilon=SPH_z-Ridge(FTV_z)",
            "residual_train_mean": float(self.residual_center),
            "residual_train_population_scale": float(self.residual_scale),
            "s1_target": "SPH_z",
            "s2_target": "population_z(epsilon)_using_outer_train_only",
        }

    @property
    def residualizer_id(self) -> str:
        return canonical_sha256(self._payload_without_id())

    def to_public_dict(self) -> dict[str, Any]:
        payload = self._payload_without_id()
        return {"residualizer_id": self.residualizer_id, **payload}


@dataclass(frozen=True)
class FoldTargetBundle:
    """In-memory targets plus identifier-free public fit metadata for one fold."""

    fold: int
    patient_ids: tuple[str, ...]
    splits: np.ndarray
    natural_sphericity: np.ndarray
    natural_ftv: np.ndarray
    s1_targets: np.ndarray
    s2_targets: np.ndarray
    epsilon: np.ndarray
    conditional_sph_z: np.ndarray
    residualizers: tuple[VisitResidualizer, ...]
    source_sha256: str

    def __post_init__(self) -> None:
        expected_shape = (len(self.patient_ids), len(VISITS))
        splits = np.asarray(self.splits, dtype=str)
        if splits.shape != (len(self.patient_ids),):
            raise ValueError("split vector does not align with patients")
        if set(splits) - ALLOWED_SPLITS:
            raise ValueError("split vector contains an unknown split")
        splits = np.array(splits, dtype=str, copy=True)
        splits.setflags(write=False)
        object.__setattr__(self, "splits", splits)
        for field_name in (
            "natural_sphericity",
            "natural_ftv",
            "s1_targets",
            "s2_targets",
            "epsilon",
            "conditional_sph_z",
        ):
            array = _readonly_float_array(getattr(self, field_name), ndim=2, label=field_name)
            if array.shape != expected_shape:
                raise ValueError(f"{field_name} must have shape {expected_shape}")
            object.__setattr__(self, field_name, array)
        if tuple(fit.visit for fit in self.residualizers) != VISITS:
            raise ValueError("residualizers must be ordered T0 through T3")

    @property
    def train_mask(self) -> np.ndarray:
        return self.splits == "train"

    @property
    def patient_to_index(self) -> Mapping[str, int]:
        return {patient_id: index for index, patient_id in enumerate(self.patient_ids)}

    def target_matrix(self, arm: str) -> np.ndarray:
        normalized = str(arm).strip().upper()
        if normalized == "S1":
            return self.s1_targets
        if normalized in {"S2", "S2L10", "S2_L10", "S2_LAMBDA_0.10"}:
            return self.s2_targets
        raise ValueError(f"arm {arm!r} has no SPH supervision target")

    def target_for_patient(self, patient_id: str, arm: str) -> np.ndarray:
        try:
            index = self.patient_to_index[str(patient_id)]
        except KeyError as error:
            raise KeyError(f"unknown target patient: {patient_id}") from error
        return np.asarray(self.target_matrix(arm)[index], dtype=np.float64).copy()

    def reconstruct_sphericity(
        self,
        residual_predictions: Any,
        *,
        ftv: Any | None = None,
    ) -> np.ndarray:
        predictions = np.asarray(residual_predictions, dtype=np.float64)
        if predictions.shape != self.s2_targets.shape:
            raise ValueError(
                f"residual predictions must have shape {self.s2_targets.shape}"
            )
        ftv_values = self.natural_ftv if ftv is None else np.asarray(ftv, dtype=np.float64)
        if ftv_values.shape != predictions.shape:
            raise ValueError("FTV reconstruction values do not align with predictions")
        reconstructed = np.empty_like(predictions)
        for visit_index, fitted in enumerate(self.residualizers):
            reconstructed[:, visit_index] = fitted.reconstruct_sphericity(
                predictions[:, visit_index], ftv_values[:, visit_index]
            )
        return reconstructed

    def to_public_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": 1,
            "artifact": "fold_static_sph_residualizer",
            "fold": int(self.fold),
            "source_target_table_sha256": self.source_sha256,
            "fit_scope": "outer_train_only_per_fold_and_visit",
            "n_patients": len(self.patient_ids),
            "split_counts": {
                split: int(np.sum(self.splits == split)) for split in sorted(ALLOWED_SPLITS)
            },
            "visits": list(VISITS),
            "dynamic_sph_supervision": False,
            "residualizers": [fitted.to_public_dict() for fitted in self.residualizers],
        }
        _assert_identifier_free_public_payload(payload)
        return payload


def _normalize_splits(
    patient_ids: Sequence[str], split_by_patient: Mapping[str, str]
) -> np.ndarray:
    normalized_mapping = {
        str(patient_id): str(split).strip().lower()
        for patient_id, split in split_by_patient.items()
    }
    missing = [patient_id for patient_id in patient_ids if patient_id not in normalized_mapping]
    if missing:
        raise ValueError(f"split mapping is missing {len(missing)} target patients")
    splits = np.asarray([normalized_mapping[patient_id] for patient_id in patient_ids], dtype=str)
    unexpected = sorted(set(splits) - ALLOWED_SPLITS)
    if unexpected:
        raise ValueError(f"split mapping contains unknown splits: {unexpected}")
    counts = {split: int(np.sum(splits == split)) for split in ALLOWED_SPLITS}
    if any(count == 0 for count in counts.values()):
        raise ValueError(f"each outer split must be non-empty: {counts}")
    return splits


def fit_fold_target_bundle(
    table: StaticTargetTable,
    split_by_patient: Mapping[str, str],
    *,
    fold: int,
    quantiles: Sequence[float] = DEFAULT_WINSOR_QUANTILES,
    alpha: float = DEFAULT_RESIDUALIZER_ALPHA,
) -> FoldTargetBundle:
    """Fit all four static visit targets from outer-train rows only."""

    if not np.isclose(float(alpha), DEFAULT_RESIDUALIZER_ALPHA, rtol=0.0, atol=0.0):
        raise ValueError("the frozen residualizer requires Ridge alpha=1")
    splits = _normalize_splits(table.patient_ids, split_by_patient)
    train_mask = splits == "train"
    s1_targets = np.empty_like(table.sphericity)
    s2_targets = np.empty_like(table.sphericity)
    epsilon = np.empty_like(table.sphericity)
    conditional = np.empty_like(table.sphericity)
    fitted_visits: list[VisitResidualizer] = []

    # All fitted state below depends exclusively on train_mask.  The complete
    # arrays are transformed only after each visit fit is frozen.
    for visit_index, visit in enumerate(VISITS):
        sph_values = table.sphericity[:, visit_index]
        ftv_values = table.ftv[:, visit_index]
        sph_transform = fit_target_transform(
            sph_values, train_mask, log1p=False, quantiles=quantiles
        )
        ftv_transform = fit_target_transform(
            ftv_values, train_mask, log1p=True, quantiles=quantiles
        )
        sph_z = sph_transform.transform(sph_values)
        ftv_z = ftv_transform.transform(ftv_values)
        ridge = Ridge(alpha=float(alpha), fit_intercept=True)
        ridge.fit(ftv_z[train_mask, None], sph_z[train_mask])
        coefficient = float(np.asarray(ridge.coef_, dtype=np.float64).reshape(-1)[0])
        intercept = float(np.asarray(ridge.intercept_, dtype=np.float64).reshape(-1)[0])
        conditional_visit = intercept + coefficient * ftv_z
        epsilon_visit = sph_z - conditional_visit
        residual_center = float(np.mean(epsilon_visit[train_mask]))
        residual_scale = float(np.std(epsilon_visit[train_mask], ddof=0))
        # This matches the predecessor audit's defensive convention.  A
        # constant SPH target already fails above; this branch only guards an
        # exceptionally degenerate residual coordinate.
        if not math.isfinite(residual_scale) or residual_scale <= 0.0:
            residual_scale = 1.0

        fitted = VisitResidualizer(
            fold=int(fold),
            visit=visit,
            sph_transform=sph_transform,
            ftv_transform=ftv_transform,
            alpha=float(alpha),
            coefficient=coefficient,
            intercept=intercept,
            residual_center=residual_center,
            residual_scale=residual_scale,
            n_train=int(train_mask.sum()),
        )
        s1_targets[:, visit_index] = sph_z
        epsilon[:, visit_index] = epsilon_visit
        conditional[:, visit_index] = conditional_visit
        s2_targets[:, visit_index] = (
            epsilon_visit - residual_center
        ) / residual_scale
        fitted_visits.append(fitted)

    return FoldTargetBundle(
        fold=int(fold),
        patient_ids=table.patient_ids,
        splits=splits,
        natural_sphericity=table.sphericity,
        natural_ftv=table.ftv,
        s1_targets=s1_targets,
        s2_targets=s2_targets,
        epsilon=epsilon,
        conditional_sph_z=conditional,
        residualizers=tuple(fitted_visits),
        source_sha256=table.source_sha256,
    )


def _assert_identifier_free_public_payload(value: Any, *, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized in _FORBIDDEN_PUBLIC_KEYS:
                raise ValueError(f"identifier/outcome field is forbidden in public payload: {path}.{key}")
            _assert_identifier_free_public_payload(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_identifier_free_public_payload(child, path=f"{path}[{index}]")


def save_public_residualizer_json(
    path: str | Path,
    bundle: FoldTargetBundle,
) -> None:
    """Atomically write aggregate fold coefficients/transforms without IDs."""

    payload = bundle.to_public_dict()
    _assert_identifier_free_public_payload(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
        Path(temporary).replace(output)
    finally:
        Path(temporary).unlink(missing_ok=True)


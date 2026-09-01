"""Outer-train-only stable, residualized radiomics PCA target construction."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from .contracts import (
    EXPERIMENT_ROOT,
    VISITS,
    atomic_json,
    canonical_sha256,
    file_sha256,
    load_protocol,
    patient_order_sha256,
    private_patient_token,
)


@dataclass(frozen=True)
class RawRadiomics:
    patient_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    values: np.ndarray
    ftv: np.ndarray
    ftv_mask: np.ndarray
    local_volume_mm3: np.ndarray
    roi_mask: np.ndarray
    source_hashes: tuple[str, ...]


def load_raw_radiomics(
    patient_ids: Iterable[str],
    radiomics_dir: str | Path,
    roi_dir: str | Path,
) -> RawRadiomics:
    ids = tuple(map(str, patient_ids))
    values: list[np.ndarray] = []
    ftv: list[np.ndarray] = []
    ftv_mask: list[np.ndarray] = []
    volume: list[np.ndarray] = []
    roi_mask: list[np.ndarray] = []
    hashes: list[str] = []
    names: tuple[str, ...] | None = None
    for patient_id in ids:
        token = private_patient_token(patient_id)
        rad_path = Path(radiomics_dir) / f"{token}.private.npz"
        roi_path = Path(roi_dir) / f"{token}.private.npz"
        with np.load(rad_path, allow_pickle=False) as payload:
            if str(payload["patient_id"].item()) != patient_id:
                raise ValueError("radiomics archive identity mismatch")
            current_names = tuple(payload["feature_name"].astype(str).tolist())
            current_values = np.asarray(payload["value"], dtype=np.float32)
        if names is None:
            names = current_names
        if current_names != names or current_values.shape != (4, 3, len(names)):
            raise ValueError("raw radiomics feature contract differs across patients")
        with np.load(roi_path, allow_pickle=False) as payload:
            if str(payload["patient_id"].item()) != patient_id:
                raise ValueError("ROI archive identity mismatch")
            ftv.append(np.asarray(payload["ftv"], dtype=np.float32))
            ftv_mask.append(np.asarray(payload["ftv_mask"], dtype=bool))
            volume.append(np.asarray(payload["local_mask_volume_mm3"], dtype=np.float32))
            roi_mask.append(np.asarray(payload["radiomics_mask"], dtype=bool))
        values.append(current_values)
        hashes.extend((file_sha256(rad_path), file_sha256(roi_path)))
    if names is None:
        raise ValueError("radiomics population is empty")
    return RawRadiomics(
        ids,
        names,
        np.stack(values),
        np.stack(ftv),
        np.stack(ftv_mask),
        np.stack(volume),
        np.stack(roi_mask),
        tuple(hashes),
    )


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3 or np.ptp(left) <= 0 or np.ptp(right) <= 0:
        return float("nan")
    return float(spearmanr(left, right).statistic)


def select_stable_features(
    raw: RawRadiomics,
    train_patient_ids: Iterable[str],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    protocol = load_protocol()["radiomics"]
    train_set = set(map(str, train_patient_ids))
    patient_mask = np.asarray([patient_id in train_set for patient_id in raw.patient_ids])
    if not patient_mask.any():
        raise ValueError("outer train has no radiomics patients")
    original = raw.values[patient_mask, :, 0].reshape(-1, len(raw.feature_names))
    eroded = raw.values[patient_mask, :, 1].reshape(-1, len(raw.feature_names))
    dilated = raw.values[patient_mask, :, 2].reshape(-1, len(raw.feature_names))
    eligible_rows = raw.roi_mask[patient_mask].reshape(-1)
    denominator = int(eligible_rows.sum())
    if denominator <= 0:
        raise ValueError("outer train has no valid radiomics visits")
    audit: list[dict[str, Any]] = []
    selected: list[tuple[int, float, str]] = []
    for feature_index, name in enumerate(raw.feature_names):
        finite_original = eligible_rows & np.isfinite(original[:, feature_index])
        coverage = float(finite_original.sum() / denominator)
        values = original[finite_original, feature_index]
        iqr = float(np.quantile(values, 0.75) - np.quantile(values, 0.25)) if len(values) else 0.0
        comparable = (
            eligible_rows
            & np.isfinite(original[:, feature_index])
            & np.isfinite(eroded[:, feature_index])
            & np.isfinite(dilated[:, feature_index])
        )
        comparable_fraction = float(comparable.sum() / denominator)
        pairwise = (
            _safe_spearman(original[comparable, feature_index], eroded[comparable, feature_index]),
            _safe_spearman(original[comparable, feature_index], dilated[comparable, feature_index]),
            _safe_spearman(eroded[comparable, feature_index], dilated[comparable, feature_index]),
        )
        finite_pairwise = [value for value in pairwise if np.isfinite(value)]
        stability = float(np.median(finite_pairwise)) if finite_pairwise else float("nan")
        passed = (
            coverage >= float(protocol["feature_coverage_train"])
            and iqr > 0
            and comparable_fraction >= float(protocol["stability_comparable"])
            and np.isfinite(stability)
            and stability >= float(protocol["stability_spearman"])
        )
        audit.append(
            {
                "feature_name": name,
                "coverage": coverage,
                "iqr": iqr,
                "comparable_fraction": comparable_fraction,
                "pairwise_spearman": list(pairwise),
                "median_pairwise_spearman": stability,
                "passed": bool(passed),
            }
        )
        if passed:
            selected.append((feature_index, stability, name))
    selected.sort(key=lambda value: (-value[1], value[2]))
    selected = selected[: int(protocol["maximum_features"])]
    if len(selected) < int(protocol["minimum_features"]):
        raise RuntimeError(f"fold is NO-GO: only {len(selected)} stable features")
    return np.asarray([value[0] for value in selected], dtype=np.int64), audit


def design_matrix(
    ftv: np.ndarray,
    volume: np.ndarray,
    continuous_mean: np.ndarray | None = None,
    continuous_sd: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ftv = np.asarray(ftv, dtype=np.float64).reshape(-1)
    volume = np.asarray(volume, dtype=np.float64).reshape(-1)
    if len(ftv) != len(volume) or len(ftv) % 4:
        raise ValueError("residualization rows must be complete T0-T3 blocks")
    continuous = np.column_stack((np.log1p(ftv), np.log1p(volume)))
    if continuous_mean is None:
        continuous_mean = continuous.mean(axis=0)
    if continuous_sd is None:
        continuous_sd = continuous.std(axis=0, ddof=0)
    continuous_sd = np.where(np.asarray(continuous_sd) > 0, continuous_sd, 1.0)
    standardized = (continuous - continuous_mean) / continuous_sd
    visits = np.tile(np.eye(4, dtype=np.float64), (len(ftv) // 4, 1))
    matrix = np.column_stack(
        (
            np.ones(len(ftv)),
            standardized,
            visits,
            standardized[:, [0]] * visits,
        )
    )
    return matrix, np.asarray(continuous_mean), np.asarray(continuous_sd)


def _fit_ridge(design: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ target)


def build_fold_targets(
    raw: RawRadiomics,
    fold: int,
    train_patient_ids: Iterable[str],
    output_dir: str | Path,
) -> dict[str, Any]:
    train_set = set(map(str, train_patient_ids))
    selected, stability_audit = select_stable_features(raw, train_set)
    n = len(raw.patient_ids)
    features = raw.values[:, :, 0, selected]
    row_patient_train = np.repeat(
        np.asarray([patient_id in train_set for patient_id in raw.patient_ids]), 4
    )
    flat_features = features.reshape(n * 4, len(selected)).astype(np.float64)
    flat_ftv = raw.ftv.reshape(-1).astype(np.float64)
    flat_ftv_mask = raw.ftv_mask.reshape(-1)
    flat_volume = raw.local_volume_mm3.reshape(-1).astype(np.float64)
    flat_roi = raw.roi_mask.reshape(-1)
    prerequisite = (
        np.isfinite(flat_features).all(axis=1)
        & flat_ftv_mask
        & np.isfinite(flat_ftv)
        & (flat_ftv >= 0)
        & np.isfinite(flat_volume)
        & (flat_volume > 0)
        & flat_roi
    )
    fit_rows = prerequisite & row_patient_train
    if int(fit_rows.sum()) < max(32, len(selected)):
        raise RuntimeError("fold is NO-GO: insufficient complete outer-train rows")
    all_design, mean, sd = design_matrix(
        raw.ftv,
        raw.local_volume_mm3,
        continuous_mean=np.asarray(
            [np.log1p(flat_ftv[fit_rows]).mean(), np.log1p(flat_volume[fit_rows]).mean()]
        ),
        continuous_sd=np.asarray(
            [np.log1p(flat_ftv[fit_rows]).std(ddof=0), np.log1p(flat_volume[fit_rows]).std(ddof=0)]
        ),
    )
    # `design_matrix` returns the supplied statistics, preserving one transform.
    _, mean, sd = design_matrix(raw.ftv, raw.local_volume_mm3, mean, sd)
    alpha = float(load_protocol()["radiomics"]["ridge_alpha"])
    coefficients = _fit_ridge(all_design[fit_rows], flat_features[fit_rows], alpha)
    residual = flat_features - all_design @ coefficients
    residual_mean = residual[fit_rows].mean(axis=0)
    residual_sd = residual[fit_rows].std(axis=0, ddof=0)
    if np.any(residual_sd <= 0) or not np.isfinite(residual_sd).all():
        raise RuntimeError("selected residual feature has zero/non-finite train SD")
    standardized = (residual - residual_mean) / residual_sd
    pca = PCA(n_components=16, whiten=False, svd_solver="full")
    pca.fit(standardized[fit_rows])
    pc = np.full((n * 4, 16), np.nan, dtype=np.float64)
    transform_rows = prerequisite
    pc[transform_rows] = pca.transform(standardized[transform_rows])
    pc_mean = pc[fit_rows].mean(axis=0)
    pc_sd = pc[fit_rows].std(axis=0, ddof=0)
    if np.any(pc_sd <= 0) or not np.isfinite(pc_sd).all():
        raise RuntimeError("PCA component has zero/non-finite outer-train SD")
    pc[transform_rows] = (pc[transform_rows] - pc_mean) / pc_sd

    # FTV target transformation is independently outer-train-only.
    ftv_fit = row_patient_train & flat_ftv_mask & np.isfinite(flat_ftv) & (flat_ftv >= 0)
    log_ftv = np.log1p(np.where(np.isfinite(flat_ftv), flat_ftv, 0.0))
    ftv_mean = float(log_ftv[ftv_fit].mean())
    ftv_sd = float(log_ftv[ftv_fit].std(ddof=0))
    if not np.isfinite(ftv_sd) or ftv_sd <= 0:
        raise RuntimeError("outer-train FTV SD is invalid")
    transformed_ftv = ((log_ftv - ftv_mean) / ftv_sd).reshape(n, 4).astype(np.float32)
    transformed_ftv[~raw.ftv_mask] = 0.0
    target = pc.reshape(n, 4, 16).astype(np.float32)
    target_mask = transform_rows.reshape(n, 4)
    target[~target_mask] = 0.0

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    target_path = output / f"fold_{int(fold)}_targets.private.npz"
    np.savez_compressed(
        target_path,
        patient_id=np.asarray(raw.patient_ids, dtype="U32"),
        ftv=transformed_ftv,
        ftv_mask=raw.ftv_mask.astype(np.uint8),
        radiomics=target,
        radiomics_mask=target_mask.astype(np.uint8),
    )
    selected_names = [raw.feature_names[index] for index in selected]
    transform = {
        "schema_version": 1,
        "fold": int(fold),
        "train_patient_order_sha256": patient_order_sha256(sorted(train_set)),
        "selected_feature_names": selected_names,
        "selected_feature_indices": selected.tolist(),
        "stability_audit": stability_audit,
        "continuous_mean": mean.tolist(),
        "continuous_sd": sd.tolist(),
        "ridge_alpha": alpha,
        "ridge_coefficients": coefficients.tolist(),
        "residual_mean": residual_mean.tolist(),
        "residual_sd": residual_sd.tolist(),
        "pca_components": pca.components_.tolist(),
        "pca_mean": pca.mean_.tolist(),
        "pca_explained_variance": pca.explained_variance_.tolist(),
        "pc_mean": pc_mean.tolist(),
        "pc_sd": pc_sd.tolist(),
        "ftv_mean": ftv_mean,
        "ftv_sd": ftv_sd,
        "source_private_hashes_sha256": canonical_sha256(raw.source_hashes),
        "outcome_fields_read": [],
        "clinical_fields_read": [],
    }
    transform_path = output / f"fold_{int(fold)}_transform.private.json"
    atomic_json(transform_path, transform)
    summary = {
        "schema_version": 1,
        "status": "PASS",
        "fold": int(fold),
        "selected_features": len(selected_names),
        "outer_train_complete_visits": int(fit_rows.sum()),
        "all_target_valid_visits": int(target_mask.sum()),
        "pca_components": 16,
        "target_sha256": file_sha256(target_path),
        "transform_sha256": file_sha256(transform_path),
        "feature_list_sha256": canonical_sha256(selected_names),
        "fit_scope": "outer_train_only",
        "validation_test_transform_only": True,
        "outcome_fields_read": [],
    }
    atomic_json(EXPERIMENT_ROOT / f"metrics/fold_{int(fold)}_target_gate.json", summary)
    return summary


__all__ = [
    "RawRadiomics", "build_fold_targets", "design_matrix", "load_raw_radiomics",
    "select_stable_features"
]

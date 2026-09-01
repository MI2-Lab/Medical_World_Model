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
    variant_mask: np.ndarray
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
    variant_mask: list[np.ndarray] = []
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
            current_variant_mask = np.asarray(payload["variant_valid"], dtype=bool)
        if names is None:
            names = current_names
        if current_names != names or current_values.shape != (4, 3, len(names)):
            raise ValueError("raw radiomics feature contract differs across patients")
        if current_variant_mask.shape != (4, 3):
            raise ValueError("raw radiomics variant-valid contract differs across patients")
        with np.load(roi_path, allow_pickle=False) as payload:
            if str(payload["patient_id"].item()) != patient_id:
                raise ValueError("ROI archive identity mismatch")
            ftv.append(np.asarray(payload["ftv"], dtype=np.float32))
            ftv_mask.append(np.asarray(payload["ftv_mask"], dtype=bool))
            volume.append(np.asarray(payload["local_mask_volume_mm3"], dtype=np.float32))
            roi_mask.append(np.asarray(payload["radiomics_mask"], dtype=bool))
        variant_mask.append(current_variant_mask)
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
        np.stack(variant_mask),
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
    values = raw.values[patient_mask]
    eligible = raw.roi_mask[patient_mask]
    variant_eligible = raw.variant_mask[patient_mask]
    audit: list[dict[str, Any]] = []
    selected: list[tuple[int, float, str]] = []
    for feature_index, name in enumerate(raw.feature_names):
        per_visit: dict[str, dict[str, Any]] = {}
        stability_values: list[float] = []
        coverage_pass = True
        stability_pass = True
        pooled_original: list[np.ndarray] = []
        for visit_index, visit in enumerate(("T0", "T1", "T2")):
            visit_eligible = eligible[:, visit_index]
            denominator = int(visit_eligible.sum())
            if denominator <= 0:
                raise ValueError(f"outer train has no valid radiomics visits at {visit}")
            original = values[:, visit_index, 0, feature_index]
            eroded = values[:, visit_index, 1, feature_index]
            dilated = values[:, visit_index, 2, feature_index]
            original_finite = visit_eligible & np.isfinite(original)
            dilation_finite = visit_eligible & np.isfinite(dilated)
            original_coverage = float(original_finite.sum() / denominator)
            dilation_coverage = float(dilation_finite.sum() / denominator)
            outward_rows = original_finite & dilation_finite
            outward = _safe_spearman(original[outward_rows], dilated[outward_rows])

            symmetric_eligible = visit_eligible & variant_eligible[:, visit_index, 1]
            symmetric_denominator = int(symmetric_eligible.sum())
            if symmetric_denominator <= 0:
                raise ValueError(f"outer train has no erosion-eligible visits at {visit}")
            symmetric_rows = (
                symmetric_eligible
                & np.isfinite(original)
                & np.isfinite(eroded)
                & np.isfinite(dilated)
            )
            symmetric_comparable = float(symmetric_rows.sum() / symmetric_denominator)
            pairwise = (
                _safe_spearman(original[symmetric_rows], eroded[symmetric_rows]),
                _safe_spearman(original[symmetric_rows], dilated[symmetric_rows]),
                _safe_spearman(eroded[symmetric_rows], dilated[symmetric_rows]),
            )
            finite_pairwise = [value for value in pairwise if np.isfinite(value)]
            symmetric = float(np.median(finite_pairwise)) if len(finite_pairwise) == 3 else float("nan")
            visit_coverage_pass = (
                original_coverage >= float(protocol["feature_coverage_train"])
                and dilation_coverage >= float(protocol["feature_coverage_train"])
                and symmetric_comparable >= float(protocol["symmetric_comparable_train"])
            )
            visit_stability_pass = (
                np.isfinite(outward)
                and np.isfinite(symmetric)
                and outward >= float(protocol["stability_spearman"])
                and symmetric >= float(protocol["stability_spearman"])
            )
            coverage_pass &= visit_coverage_pass
            stability_pass &= visit_stability_pass
            stability_values.extend((outward, symmetric))
            pooled_original.append(original[original_finite])
            per_visit[visit] = {
                "valid_original_visits": denominator,
                "erosion_eligible_visits": symmetric_denominator,
                "original_coverage": original_coverage,
                "dilation_coverage": dilation_coverage,
                "symmetric_comparable_fraction": symmetric_comparable,
                "outward_spearman": outward,
                "symmetric_pairwise_spearman": list(pairwise),
                "symmetric_median_spearman": symmetric,
                "coverage_pass": bool(visit_coverage_pass),
                "stability_pass": bool(visit_stability_pass),
            }
        pooled = np.concatenate(pooled_original) if pooled_original else np.asarray([])
        iqr = float(np.quantile(pooled, 0.75) - np.quantile(pooled, 0.25)) if len(pooled) else 0.0
        stability = float(np.min(stability_values)) if all(np.isfinite(stability_values)) else float("nan")
        passed = (
            coverage_pass
            and iqr > 0
            and stability_pass
        )
        audit.append(
            {
                "feature_name": name,
                "iqr": iqr,
                "per_visit": per_visit,
                "minimum_early_stability": stability,
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
    ftv_array = np.asarray(ftv, dtype=np.float64)
    volume_array = np.asarray(volume, dtype=np.float64)
    if ftv_array.shape != volume_array.shape:
        raise ValueError("residualization FTV/volume shapes differ")
    if ftv_array.ndim == 1:
        if len(ftv_array) % 4:
            raise ValueError("one-dimensional residualization rows must be T0-T3 blocks")
        visits_per_patient = 4
    elif ftv_array.ndim == 2 and ftv_array.shape[1] in (3, 4):
        visits_per_patient = int(ftv_array.shape[1])
    else:
        raise ValueError("residualization rows must be [N,3], [N,4], or flat T0-T3")
    ftv = ftv_array.reshape(-1)
    volume = volume_array.reshape(-1)
    continuous = np.column_stack((np.log1p(ftv), np.log1p(volume)))
    if continuous_mean is None:
        continuous_mean = continuous.mean(axis=0)
    if continuous_sd is None:
        continuous_sd = continuous.std(axis=0, ddof=0)
    continuous_sd = np.where(np.asarray(continuous_sd) > 0, continuous_sd, 1.0)
    standardized = (continuous - continuous_mean) / continuous_sd
    visits = np.tile(
        np.eye(visits_per_patient, dtype=np.float64),
        (len(ftv) // visits_per_patient, 1),
    )
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
    features = raw.values[:, :3, 0, selected]
    row_patient_train = np.repeat(
        np.asarray([patient_id in train_set for patient_id in raw.patient_ids]), 3
    )
    flat_features = features.reshape(n * 3, len(selected)).astype(np.float64)
    flat_ftv = raw.ftv[:, :3].reshape(-1).astype(np.float64)
    flat_ftv_mask = raw.ftv_mask[:, :3].reshape(-1)
    flat_volume = raw.local_volume_mm3[:, :3].reshape(-1).astype(np.float64)
    flat_roi = raw.roi_mask[:, :3].reshape(-1)
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
        raw.ftv[:, :3],
        raw.local_volume_mm3[:, :3],
        continuous_mean=np.asarray(
            [np.log1p(flat_ftv[fit_rows]).mean(), np.log1p(flat_volume[fit_rows]).mean()]
        ),
        continuous_sd=np.asarray(
            [np.log1p(flat_ftv[fit_rows]).std(ddof=0), np.log1p(flat_volume[fit_rows]).std(ddof=0)]
        ),
    )
    # `design_matrix` returns the supplied statistics, preserving one transform.
    _, mean, sd = design_matrix(raw.ftv[:, :3], raw.local_volume_mm3[:, :3], mean, sd)
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
    pc = np.full((n * 3, 16), np.nan, dtype=np.float64)
    transform_rows = prerequisite
    pc[transform_rows] = pca.transform(standardized[transform_rows])
    pc_mean = pc[fit_rows].mean(axis=0)
    pc_sd = pc[fit_rows].std(axis=0, ddof=0)
    if np.any(pc_sd <= 0) or not np.isfinite(pc_sd).all():
        raise RuntimeError("PCA component has zero/non-finite outer-train SD")
    pc[transform_rows] = (pc[transform_rows] - pc_mean) / pc_sd

    # FTV target transformation is independently outer-train-only.
    full_flat_ftv = raw.ftv.reshape(-1).astype(np.float64)
    full_ftv_mask = raw.ftv_mask.reshape(-1)
    full_row_patient_train = np.repeat(
        np.asarray([patient_id in train_set for patient_id in raw.patient_ids]), 4
    )
    ftv_fit = (
        full_row_patient_train
        & full_ftv_mask
        & np.isfinite(full_flat_ftv)
        & (full_flat_ftv >= 0)
    )
    log_ftv = np.log1p(np.where(np.isfinite(full_flat_ftv), full_flat_ftv, 0.0))
    ftv_mean = float(log_ftv[ftv_fit].mean())
    ftv_sd = float(log_ftv[ftv_fit].std(ddof=0))
    if not np.isfinite(ftv_sd) or ftv_sd <= 0:
        raise RuntimeError("outer-train FTV SD is invalid")
    transformed_ftv = ((log_ftv - ftv_mean) / ftv_sd).reshape(n, 4).astype(np.float32)
    transformed_ftv[~raw.ftv_mask] = 0.0
    target = np.zeros((n, 4, 16), dtype=np.float32)
    target_mask = np.zeros((n, 4), dtype=bool)
    target[:, :3] = pc.reshape(n, 3, 16).astype(np.float32)
    target_mask[:, :3] = transform_rows.reshape(n, 3)
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
        "grounding_visits": ["T0", "T1", "T2"],
        "t3_radiomics_mask_false": bool(not target_mask[:, 3].any()),
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

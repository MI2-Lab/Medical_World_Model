"""End-to-end post-freeze evaluation for the clinical-residual state pilot.

Only this module may combine frozen representations with outcome labels.  It
writes patient-level OOF rows solely below gitignored ``predictions/`` and
publishes aggregate tables below ``metrics/``.
"""

from __future__ import annotations

from collections import defaultdict
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .diagnostics import (
    augmentation_cosine_consistency,
    canonical_correlation_summary,
    covariance_spectrum_summary,
    nearest_neighbor_jaccard_stability,
    per_dimension_variance_summary,
    standardized_cross_covariance_summary,
)
from .evaluation_contracts import (
    ARMS,
    EXPERIMENT_ROOT,
    F0Asset,
    FOLDS,
    FactorizedAsset,
    SEEDS,
    load_clinical,
    load_evaluation_config,
    load_f0_asset,
    load_factorized_asset,
    load_fold_assignments,
    load_stage_b_ftv_records,
    load_ftv_wide,
)
from .evaluation_lock import (
    LOCK_PATH as EVALUATION_LOCK_PATH,
    require_before_outcome_access,
)
from .evaluation_modeling import (
    ClinicalEncoder,
    binary_metrics,
    fit_binary_logistic,
    fit_linear_residualizer,
    fit_multiclass_logistic,
    multiclass_metrics,
    paired_stratified_bootstrap,
)
from .response_probes import run_matched_response_probes


TIMINGS: Mapping[str, tuple[int, ...]] = {
    "T0": (0,),
    "T0-T1": (0, 1),
    "T0-T2": (0, 1, 2),
}
PROFILE_ENDPOINTS = ("T0", "T1", "T2", "T3")
SUBTYPE_CLASSES = tuple(
    sorted(("HR+/HER2-", "HR-/HER2-", "HR+/HER2+", "HR-/HER2+"))
)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            frame.to_csv(stream, index=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _split_indices(split: np.ndarray, eligible: np.ndarray | None = None) -> dict[str, np.ndarray]:
    mask = np.ones(len(split), dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
    output = {name: np.flatnonzero(mask & (split == name)) for name in ("train", "val", "test")}
    if any(not len(values) for values in output.values()):
        raise ValueError("every eligible train/val/test split must be nonempty")
    return output


def _aligned(frame: pd.DataFrame, patient_id: np.ndarray) -> pd.DataFrame:
    indexed = frame.set_index("patient_id", verify_integrity=True)
    missing = set(patient_id.astype(str)) - set(indexed.index.astype(str))
    if missing:
        raise ValueError(f"aligned table misses patients: {sorted(missing)[:3]}")
    return indexed.loc[patient_id.astype(str)].reset_index()


def _prefix(state: np.ndarray, timing: str) -> np.ndarray:
    visits = TIMINGS[timing]
    return state[:, visits, :].reshape(len(state), -1).astype(np.float64, copy=False)


def _state_map(asset: FactorizedAsset | F0Asset) -> Mapping[str, np.ndarray]:
    if isinstance(asset, FactorizedAsset):
        return {"z_R": asset.z_R, "z_P": asset.z_P, "z_R+z_P": asset.full}
    return {"F0": asset.state}


def load_all_assets(
    config: Mapping[str, Any], folds: pd.DataFrame
) -> tuple[list[FactorizedAsset], list[F0Asset]]:
    factorized = [
        load_factorized_asset(config, folds, arm, seed, fold)
        for seed in SEEDS
        for arm in ARMS
        for fold in FOLDS
    ]
    f0 = [load_f0_asset(config, folds, seed, fold) for seed in SEEDS for fold in FOLDS]
    if len(factorized) != 20 or len(f0) != 10:
        raise AssertionError("feature cell matrix is incomplete")
    return factorized, f0


def _future_prediction_diagnostic(
    predicted: np.ndarray,
    target: np.ndarray,
) -> tuple[float, float]:
    """Return predictor/persistence MSE in one EMA-target coordinate space."""

    prediction = np.asarray(predicted, dtype=np.float64)
    ema_target = np.asarray(target, dtype=np.float64)
    if (
        prediction.shape != ema_target.shape
        or prediction.ndim != 3
        or prediction.shape[1:] != (3, 96)
        or not np.isfinite(prediction).all()
        or not np.isfinite(ema_target).all()
    ):
        raise ValueError("future diagnostic tensors must be finite aligned [N,3,96]")
    prediction_mse = float(np.mean(np.square(prediction[:, 1:] - ema_target[:, 1:])))
    persistence_mse = float(
        np.mean(np.square(ema_target[:, :-1] - ema_target[:, 1:]))
    )
    return prediction_mse, persistence_mse


def compute_state_diagnostics(
    factorized: Sequence[FactorizedAsset], f0: Sequence[F0Asset], config: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute outcome-free fold diagnostics and cross-seed NN stability."""

    rows: list[dict[str, Any]] = []
    for asset in factorized:
        test = asset.split == "test"
        response = asset.z_R[test].reshape(-1, 96)
        phenotype = asset.z_P[test].reshape(-1, 96)
        variance_r = per_dimension_variance_summary(response)
        variance_p = per_dimension_variance_summary(phenotype)
        spectrum_r = covariance_spectrum_summary(response)
        spectrum_p = covariance_spectrum_summary(phenotype)
        cross = standardized_cross_covariance_summary(response, phenotype)
        cca = canonical_correlation_summary(
            response,
            phenotype,
            top_k=int(config["diagnostics"]["cca_components"]),
        )
        augmentation = (
            augmentation_cosine_consistency(
                phenotype, asset.z_P_aug[test].reshape(-1, 96)
            )
            if asset.z_P_aug is not None
            else None
        )
        # Prediction targets live in the EMA target-projector coordinate
        # system.  A valid persistence baseline must live there too.  The
        # export contains EMA targets for T1/T2/T3, so the two later
        # transitions can be compared exactly: predicted T2/T3 versus target
        # T2/T3, and persisted target T1/T2 versus target T2/T3.  The exported
        # online context is intentionally not mixed into this MSE comparison.
        if asset.z_P_future_pred is not None and asset.z_P_future_target is not None:
            future_mse, persistence_mse = _future_prediction_diagnostic(
                asset.z_P_future_pred[test], asset.z_P_future_target[test]
            )
        else:
            future_mse = float(asset.metadata.get("future_phenotype_loss", math.nan))
            persistence_mse = math.nan
        rows.append(
            {
                "arm": asset.arm,
                "seed_base": asset.seed_base,
                "fold": asset.fold,
                "scope": "outer_test_all_visits_outcome_free",
                "n_patient_visits": int(len(response)),
                "z_R_mean_std": float(variance_r["mean_std"]),
                "z_P_mean_std": float(variance_p["mean_std"]),
                "z_R_effective_rank": float(spectrum_r["effective_rank"]),
                "z_P_effective_rank": float(spectrum_p["effective_rank"]),
                "z_R_cov_trace": float(spectrum_r["trace"]),
                "z_P_cov_trace": float(spectrum_p["trace"]),
                "z_R_eigen_top1_fraction": float(spectrum_r["explained_variance_ratio"][0]),
                "z_P_eigen_top1_fraction": float(spectrum_p["explained_variance_ratio"][0]),
                "z_R_eigenspectrum_json": json.dumps(spectrum_r["eigenvalues"].tolist()),
                "z_P_eigenspectrum_json": json.dumps(spectrum_p["eigenvalues"].tolist()),
                "z_R_collapsed_dimensions": int(variance_r["collapsed_dimensions"]),
                "z_P_collapsed_dimensions": int(variance_p["collapsed_dimensions"]),
                "z_R_variance_per_dimension_json": json.dumps(
                    variance_r["variance"].tolist()
                ),
                "z_P_variance_per_dimension_json": json.dumps(
                    variance_p["variance"].tolist()
                ),
                "z_R_std_per_dimension_json": json.dumps(variance_r["std"].tolist()),
                "z_P_std_per_dimension_json": json.dumps(variance_p["std"].tolist()),
                "standardized_crosscov_frobenius": float(cross["frobenius_norm"]),
                "standardized_crosscov_rms": float(cross["root_mean_squared_norm"]),
                "cca_mean_top10": float(cca["mean_canonical_correlation"]),
                "cca_max": float(cca["max_canonical_correlation"]),
                "augmentation_mean_cosine": float(augmentation["mean_cosine"])
                if augmentation is not None
                else math.nan,
                "future_phenotype_mse": future_mse,
                "future_persistence_mse": persistence_mse,
                "future_mse_improvement_over_persistence": persistence_mse - future_mse,
                "future_comparison_transitions": "T1_to_T2|T2_to_T3",
                "future_comparison_coordinate_space": "ema_target_projector",
                "optional_diagnostic_tensors_present": bool(
                    asset.z_P_aug is not None and asset.z_P_future_pred is not None
                ),
                "pcr_labels_used": False,
            }
        )

    # An unseparated 192-D control is summarized by its preregistered first/last
    # 96 coordinates; it is descriptive only and supplies Gate-B's control norm.
    for asset in f0:
        test = asset.split == "test"
        state = asset.state[test].reshape(-1, 192).astype(np.float64)
        cross = standardized_cross_covariance_summary(state[:, :96], state[:, 96:])
        spectrum = covariance_spectrum_summary(state)
        variance = per_dimension_variance_summary(state)
        rows.append(
            {
                "arm": "F0",
                "seed_base": asset.seed_base,
                "fold": asset.fold,
                "scope": "outer_test_all_visits_unseparated_control",
                "n_patient_visits": int(len(state)),
                "z_R_mean_std": float(variance["mean_std"]),
                "z_P_mean_std": math.nan,
                "z_R_effective_rank": float(spectrum["effective_rank"]),
                "z_P_effective_rank": math.nan,
                "z_R_cov_trace": float(spectrum["trace"]),
                "z_P_cov_trace": math.nan,
                "z_R_eigen_top1_fraction": float(spectrum["explained_variance_ratio"][0]),
                "z_P_eigen_top1_fraction": math.nan,
                "z_R_eigenspectrum_json": json.dumps(spectrum["eigenvalues"].tolist()),
                "z_P_eigenspectrum_json": "",
                "z_R_collapsed_dimensions": int(variance["collapsed_dimensions"]),
                "z_P_collapsed_dimensions": math.nan,
                "z_R_variance_per_dimension_json": json.dumps(
                    variance["variance"].tolist()
                ),
                "z_P_variance_per_dimension_json": "",
                "z_R_std_per_dimension_json": json.dumps(variance["std"].tolist()),
                "z_P_std_per_dimension_json": "",
                "standardized_crosscov_frobenius": float(cross["frobenius_norm"]),
                "standardized_crosscov_rms": float(cross["root_mean_squared_norm"]),
                "cca_mean_top10": math.nan,
                "cca_max": math.nan,
                "augmentation_mean_cosine": math.nan,
                "future_phenotype_mse": math.nan,
                "future_persistence_mse": math.nan,
                "future_mse_improvement_over_persistence": math.nan,
                "optional_diagnostic_tensors_present": False,
                "pcr_labels_used": False,
            }
        )

    diagnostic = pd.DataFrame(rows)
    # Neighbors are defined only within a shared checkpoint coordinate system.
    # Therefore compare seeds within each fold, then aggregate the five fold
    # summaries; never concatenate independently trained fold spaces.
    nn_rows: list[dict[str, Any]] = []
    top_k = int(config["diagnostics"]["nearest_neighbors_k"])
    by_cell = {(asset.arm, asset.seed_base, asset.fold): asset for asset in factorized}
    for arm in ARMS:
        fold_results: list[dict[str, Any]] = []
        for fold in FOLDS:
            representations: dict[int, np.ndarray] = {}
            identifiers: dict[int, np.ndarray] = {}
            for seed in SEEDS:
                asset = by_cell[(arm, seed, fold)]
                test = asset.split == "test"
                order = np.argsort(asset.patient_id[test], kind="stable")
                representations[seed] = _prefix(asset.z_P[test], "T0-T2")[order]
                identifiers[seed] = asset.patient_id[test][order]
            result = nearest_neighbor_jaccard_stability(
                representations, identifiers, top_k=top_k, metric="cosine"
            )
            fold_results.append(result)
            nn_rows.append(
                {
                    "arm": arm,
                    "state": "z_P",
                    "timing": "T0-T2",
                    "fold": fold,
                    "aggregation": "within_fold_across_seed_pair",
                    "top_k": top_k,
                    "metric": "cosine",
                    "n_patients": int(result["n_patients"]),
                    "mean_jaccard": float(result["mean_jaccard"]),
                    "median_jaccard": float(result["median_jaccard"]),
                    "valid_patient_comparisons": int(result["valid_patient_comparisons"]),
                    "ambiguous_patient_comparisons": int(result["ambiguous_patient_comparisons"]),
                    "seed_pairs": int(len(result["seed_pair_summaries"])),
                    "pcr_labels_used": False,
                }
            )
        valid_means = np.asarray(
            [result["mean_jaccard"] for result in fold_results], dtype=float
        )
        valid_medians = np.asarray(
            [result["median_jaccard"] for result in fold_results], dtype=float
        )
        nn_rows.append(
            {
                "arm": arm,
                "state": "z_P",
                "timing": "T0-T2",
                "fold": -1,
                "aggregation": "unweighted_mean_of_5_within_fold_seed_comparisons",
                "top_k": top_k,
                "metric": "cosine",
                "n_patients": int(sum(result["n_patients"] for result in fold_results)),
                "mean_jaccard": float(np.mean(valid_means))
                if np.isfinite(valid_means).all()
                else math.nan,
                "median_jaccard": float(np.mean(valid_medians))
                if np.isfinite(valid_medians).all()
                else math.nan,
                "valid_patient_comparisons": int(
                    sum(result["valid_patient_comparisons"] for result in fold_results)
                ),
                "ambiguous_patient_comparisons": int(
                    sum(result["ambiguous_patient_comparisons"] for result in fold_results)
                ),
                "seed_pairs": len(FOLDS),
                "pcr_labels_used": False,
            }
        )
    return diagnostic, pd.DataFrame(nn_rows)


def expand_state_diagnostic_tables(
    diagnostics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize public per-dimension variance/std and full eigenspectra."""

    dimension_rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []
    for row in diagnostics.itertuples(index=False):
        state_names = ("F0",) if row.arm == "F0" else ("z_R", "z_P")
        prefixes = ("z_R",) if row.arm == "F0" else ("z_R", "z_P")
        for state, prefix in zip(state_names, prefixes, strict=True):
            variance_text = getattr(row, f"{prefix}_variance_per_dimension_json")
            std_text = getattr(row, f"{prefix}_std_per_dimension_json")
            eigen_text = getattr(row, f"{prefix}_eigenspectrum_json")
            if not variance_text or not std_text or not eigen_text:
                continue
            variance = np.asarray(json.loads(variance_text), dtype=float)
            std = np.asarray(json.loads(std_text), dtype=float)
            eigenvalues = np.asarray(json.loads(eigen_text), dtype=float)
            if variance.shape != std.shape or variance.ndim != 1:
                raise ValueError("variance/std diagnostic arrays differ")
            for dimension, (var, deviation) in enumerate(
                zip(variance, std, strict=True)
            ):
                dimension_rows.append(
                    {
                        "arm": row.arm,
                        "seed_base": int(row.seed_base),
                        "fold": int(row.fold),
                        "state": state,
                        "dimension": dimension,
                        "variance": float(var),
                        "std": float(deviation),
                        "pcr_labels_used": False,
                    }
                )
            total = float(eigenvalues.sum())
            for index, eigenvalue in enumerate(eigenvalues):
                spectrum_rows.append(
                    {
                        "arm": row.arm,
                        "seed_base": int(row.seed_base),
                        "fold": int(row.fold),
                        "state": state,
                        "eigen_index": index,
                        "eigenvalue": float(eigenvalue),
                        "explained_fraction": float(eigenvalue / total)
                        if total > 0
                        else 0.0,
                        "pcr_labels_used": False,
                    }
                )
    return pd.DataFrame(dimension_rows), pd.DataFrame(spectrum_rows)


def run_profile_probes(
    assets: Sequence[FactorizedAsset | F0Asset],
    clinical: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions: list[dict[str, Any]] = []
    c_grid = config["logistic_c_grid"]
    for asset in assets:
        aligned = _aligned(clinical, asset.patient_id)
        indices = _split_indices(asset.split)
        arm = asset.arm if isinstance(asset, FactorizedAsset) else "F0"
        for state_name, state in _state_map(asset).items():
            for visit, endpoint in enumerate(PROFILE_ENDPOINTS):
                matrix = state[:, visit].astype(np.float64)
                for target in ("label_hr", "label_her2"):
                    labels = aligned[target].to_numpy(dtype=int)
                    fit = fit_binary_logistic(
                        matrix[indices["train"]], labels[indices["train"]],
                        matrix[indices["val"]], labels[indices["val"]], c_grid,
                        class_weight="balanced", random_state=asset.seed_base + asset.fold,
                        solver=str(config["model_selection"]["binary_logistic_solver"]),
                    )
                    test = indices["test"]
                    probability = fit.predict(matrix[test])
                    for row, value in zip(test, probability, strict=True):
                        predictions.append(
                            {
                                "patient_id": asset.patient_id[row], "fold": asset.fold,
                                "seed_base": asset.seed_base, "arm": arm, "state": state_name,
                                "endpoint": endpoint, "target": target,
                                "y_true": str(labels[row]), "probability": float(value),
                                "probability_json": "", "selected_c": fit.selected_c,
                            }
                        )
                labels_multi = aligned["hr_her2_subtype"].astype(str).to_numpy()
                fit_multi = fit_multiclass_logistic(
                    matrix[indices["train"]], labels_multi[indices["train"]],
                    matrix[indices["val"]], labels_multi[indices["val"]], c_grid,
                    random_state=asset.seed_base + asset.fold,
                    solver=str(config["model_selection"]["multiclass_logistic_solver"]),
                    expected_classes=SUBTYPE_CLASSES,
                )
                test = indices["test"]
                probability_multi = fit_multi.predict(matrix[test])
                for row, values in zip(test, probability_multi, strict=True):
                    predictions.append(
                        {
                            "patient_id": asset.patient_id[row], "fold": asset.fold,
                            "seed_base": asset.seed_base, "arm": arm, "state": state_name,
                            "endpoint": endpoint, "target": "subtype",
                            "y_true": str(labels_multi[row]), "probability": math.nan,
                            "probability_json": json.dumps(values.tolist()),
                            "selected_c": fit_multi.selected_c,
                        }
                    )
    private = pd.DataFrame(predictions)
    keys = ["seed_base", "arm", "state", "endpoint", "target"]
    rows: list[dict[str, Any]] = []
    for key, group in private.groupby(keys, sort=True):
        common = dict(zip(keys, key, strict=True))
        if key[-1] == "subtype":
            probability = np.asarray([json.loads(value) for value in group.probability_json])
            values = multiclass_metrics(group.y_true, probability, SUBTYPE_CLASSES)
            rows.append(
                common
                | values
                | {
                    "auroc": float(values["auroc_macro_ovr"]),
                    "flip_invariant_decodability": math.nan,
                }
            )
        else:
            values = binary_metrics(group.y_true.astype(int), group.probability)
            rows.append(
                common
                | values
                | {
                    "auroc_macro_ovr": math.nan,
                    "accuracy": math.nan,
                    "flip_invariant_decodability": 0.5
                    + abs(float(values["auroc"]) - 0.5),
                }
            )
    metrics = pd.DataFrame(rows)
    macro_rows: list[dict[str, Any]] = []
    for key, group in metrics.groupby(["seed_base", "arm", "state", "target"], sort=True):
        macro_rows.append(
            dict(zip(["seed_base", "arm", "state", "target"], key, strict=True))
            | {
                "endpoint": "T0_T1_T2_macro",
                "n": int(group.loc[group.endpoint.isin(("T0", "T1", "T2")), "n"].sum()),
                "auroc": float(group.loc[group.endpoint.isin(("T0", "T1", "T2")), "auroc"].mean()),
                "flip_invariant_decodability": float(
                    group.loc[
                        group.endpoint.isin(("T0", "T1", "T2")),
                        "flip_invariant_decodability",
                    ].mean()
                )
                if group.loc[
                    group.endpoint.isin(("T0", "T1", "T2")),
                    "flip_invariant_decodability",
                ].notna().any()
                else math.nan,
                "auprc": float(group.loc[group.endpoint.isin(("T0", "T1", "T2")), "auprc"].mean())
                if "auprc" in group and group.loc[group.endpoint.isin(("T0", "T1", "T2")), "auprc"].notna().any()
                else math.nan,
                "brier": float(group.loc[group.endpoint.isin(("T0", "T1", "T2")), "brier"].mean())
                if "brier" in group and group.loc[group.endpoint.isin(("T0", "T1", "T2")), "brier"].notna().any()
                else math.nan,
                "n_positive": int(group.loc[group.endpoint.isin(("T0", "T1", "T2")), "n_positive"].sum())
                if "n_positive" in group and group.loc[group.endpoint.isin(("T0", "T1", "T2")), "n_positive"].notna().any()
                else math.nan,
                "auroc_macro_ovr": float(group.loc[group.endpoint.isin(("T0", "T1", "T2")), "auroc_macro_ovr"].mean())
                if group.loc[group.endpoint.isin(("T0", "T1", "T2")), "auroc_macro_ovr"].notna().any()
                else math.nan,
                "accuracy": float(group.loc[group.endpoint.isin(("T0", "T1", "T2")), "accuracy"].mean())
                if group.loc[group.endpoint.isin(("T0", "T1", "T2")), "accuracy"].notna().any()
                else math.nan,
            }
        )
    return pd.concat((metrics, pd.DataFrame(macro_rows)), ignore_index=True), private


ClinicalCacheEntry = tuple[dict[str, float], dict[str, int], float, float]


def _make_clinical_cache_entry(
    patient_id: np.ndarray,
    labels: np.ndarray,
    probability: np.ndarray,
    selected_c: float,
    validation_auroc: float,
) -> ClinicalCacheEntry:
    identifiers = np.asarray(patient_id).astype(str)
    truth = np.asarray(labels, dtype=int)
    prediction = np.asarray(probability, dtype=np.float64)
    if not (identifiers.shape == truth.shape == prediction.shape):
        raise AssertionError("clinical baseline cache vectors differ")
    if len(set(identifiers)) != len(identifiers):
        raise AssertionError("clinical baseline test patient IDs repeat")
    return (
        dict(zip(identifiers, prediction, strict=True)),
        dict(zip(identifiers, truth, strict=True)),
        float(selected_c),
        float(validation_auroc),
    )


def _read_clinical_cache_entry(
    entry: ClinicalCacheEntry,
    patient_id: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    probability_by_patient, label_by_patient, selected_c, validation_auroc = entry
    identifiers = np.asarray(patient_id).astype(str)
    truth = np.asarray(labels, dtype=int)
    if identifiers.shape != truth.shape or set(identifiers) != set(probability_by_patient):
        raise AssertionError("clinical baseline test patient set differs across arms")
    probability = np.asarray(
        [probability_by_patient[value] for value in identifiers], dtype=np.float64
    )
    expected_labels = np.asarray(
        [label_by_patient[value] for value in identifiers], dtype=int
    )
    if not np.array_equal(truth, expected_labels):
        raise AssertionError("clinical baseline test labels differ across arms")
    return probability, selected_c, validation_auroc


def run_pcr_models(
    assets: Sequence[FactorizedAsset | F0Asset],
    clinical: pd.DataFrame,
    ftv: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run fold-safe MRI/clinical/FTV pCR families and return OOF rows."""

    predictions: list[dict[str, Any]] = []
    c_grid = config["logistic_c_grid"]
    ftv_ids = set(ftv.patient_id.astype(str))
    clinical_probability_cache: dict[tuple[int, int, str, str], ClinicalCacheEntry] = {}
    for asset in assets:
        aligned_clinical = _aligned(clinical, asset.patient_id)
        aligned_ftv = ftv.set_index("patient_id", verify_integrity=True).reindex(asset.patient_id)
        labels = aligned_clinical["label_pcr"].to_numpy(dtype=int)
        arm = asset.arm if isinstance(asset, FactorizedAsset) else "F0"
        for population in ("full_808", "ftv_complete_375"):
            eligible = (
                np.ones(len(asset.patient_id), dtype=bool)
                if population == "full_808"
                else np.asarray([value in ftv_ids for value in asset.patient_id], dtype=bool)
            )
            indices = _split_indices(asset.split, eligible)
            encoder = ClinicalEncoder().fit(aligned_clinical.iloc[indices["train"]])
            clinical_matrix = encoder.transform(aligned_clinical)
            for timing in TIMINGS:
                state_features = {
                    name: _prefix(state, timing) for name, state in _state_map(asset).items()
                }
                feature_sets: dict[str, np.ndarray] = {"C": clinical_matrix}
                if isinstance(asset, FactorizedAsset):
                    z_r, z_p, full = (
                        state_features["z_R"], state_features["z_P"], state_features["z_R+z_P"]
                    )
                    feature_sets.update(
                        {
                            "z_R": z_r,
                            "z_P": z_p,
                            "z_R+z_P": full,
                            "C+z_R": np.concatenate((clinical_matrix, z_r), axis=1),
                            "C+z_P": np.concatenate((clinical_matrix, z_p), axis=1),
                            "C+z_R+z_P": np.concatenate((clinical_matrix, full), axis=1),
                        }
                    )
                    if bool(config.get("optional_f3_linear_residualization", False)):
                        residualizer = fit_linear_residualizer(
                            clinical_matrix[indices["train"]], z_p[indices["train"]]
                        )
                        z_p_res = residualizer.transform(clinical_matrix, z_p)
                        feature_sets.update(
                            {
                                "z_P_res": z_p_res,
                                "C+z_P_res": np.concatenate((clinical_matrix, z_p_res), axis=1),
                                "C+z_R+z_P_res": np.concatenate(
                                    (clinical_matrix, z_r, z_p_res), axis=1
                                ),
                            }
                        )
                else:
                    f0_state = state_features["F0"]
                    feature_sets.update(
                        {
                            "F0": f0_state,
                            "C+F0": np.concatenate((clinical_matrix, f0_state), axis=1),
                        }
                    )
                if population == "ftv_complete_375":
                    visit_columns = [f"FTV_T{visit}" for visit in TIMINGS[timing]]
                    ftv_matrix = np.log1p(aligned_ftv[visit_columns].to_numpy(dtype=float))
                    if not np.isfinite(ftv_matrix[eligible]).all():
                        raise ValueError("FTV prefix contains missing/non-finite values")
                    # Ineligible NaNs are never fitted or predicted; fill to retain row alignment.
                    ftv_matrix[~eligible] = 0.0
                    feature_sets = {"C": clinical_matrix, "C+F": np.concatenate((clinical_matrix, ftv_matrix), axis=1)}
                    if isinstance(asset, FactorizedAsset):
                        z_r, z_p, full = (
                            state_features["z_R"], state_features["z_P"], state_features["z_R+z_P"]
                        )
                        feature_sets.update(
                            {
                                "C+F+z_R": np.concatenate((clinical_matrix, ftv_matrix, z_r), axis=1),
                                "C+F+z_P": np.concatenate((clinical_matrix, ftv_matrix, z_p), axis=1),
                                "C+F+z_R+z_P": np.concatenate((clinical_matrix, ftv_matrix, full), axis=1),
                            }
                        )
                        if bool(config.get("optional_f3_linear_residualization", False)):
                            residualizer = fit_linear_residualizer(
                                clinical_matrix[indices["train"]], z_p[indices["train"]]
                            )
                            z_p_res = residualizer.transform(clinical_matrix, z_p)
                            feature_sets.update(
                                {
                                    "C+F+z_P_res": np.concatenate(
                                        (clinical_matrix, ftv_matrix, z_p_res), axis=1
                                    ),
                                    "C+F+z_R+z_P_res": np.concatenate(
                                        (clinical_matrix, ftv_matrix, z_r, z_p_res), axis=1
                                    ),
                                }
                            )
                    else:
                        f0_state = state_features["F0"]
                        feature_sets["C+F+F0"] = np.concatenate(
                            (clinical_matrix, ftv_matrix, f0_state), axis=1
                        )
                for model_name, matrix in feature_sets.items():
                    cache_key = (asset.seed_base, asset.fold, population, timing)
                    if model_name == "C" and cache_key in clinical_probability_cache:
                        test = indices["test"]
                        test_ids = asset.patient_id[test].astype(str)
                        (
                            probability,
                            fit_selected_c,
                            fit_validation_auroc,
                        ) = _read_clinical_cache_entry(
                            clinical_probability_cache[cache_key],
                            test_ids,
                            labels[test],
                        )
                    else:
                        fit = fit_binary_logistic(
                            matrix[indices["train"]], labels[indices["train"]],
                            matrix[indices["val"]], labels[indices["val"]], c_grid,
                            random_state=asset.seed_base + asset.fold,
                            solver=str(config["model_selection"]["binary_logistic_solver"]),
                        )
                        test = indices["test"]
                        probability = fit.predict(matrix[test])
                        fit_selected_c = fit.selected_c
                        fit_validation_auroc = fit.validation_auroc
                        if model_name == "C":
                            test_ids = asset.patient_id[test].astype(str)
                            clinical_probability_cache[cache_key] = _make_clinical_cache_entry(
                                test_ids,
                                labels[test],
                                probability,
                                fit_selected_c,
                                fit_validation_auroc,
                            )
                    for row, value in zip(test, probability, strict=True):
                        predictions.append(
                            {
                                "patient_id": asset.patient_id[row],
                                "fold": asset.fold,
                                "seed_base": asset.seed_base,
                                "arm": arm,
                                "population": population,
                                "timing": timing,
                                "model": model_name,
                                "label_pcr": int(labels[row]),
                                "probability": float(value),
                                "selected_c": fit_selected_c,
                                "validation_auroc": fit_validation_auroc,
                            }
                        )
    private = pd.DataFrame(predictions)
    rows: list[dict[str, Any]] = []
    keys = ["seed_base", "arm", "population", "timing", "model"]
    for key, group in private.groupby(keys, sort=True):
        if group.patient_id.duplicated().any():
            raise AssertionError("OOF pCR group repeats a patient")
        expected_n = 808 if key[2] == "full_808" else 375
        if len(group) != expected_n:
            raise AssertionError(f"OOF pCR group expected {expected_n}, got {len(group)}")
        rows.append(dict(zip(keys, key, strict=True)) | binary_metrics(group.label_pcr, group.probability))
    return pd.DataFrame(rows), private


def _paired_group(frame: pd.DataFrame, arm: str, population: str, timing: str, model: str, seed: int) -> pd.DataFrame:
    selected = frame.loc[
        frame.arm.eq(arm)
        & frame.population.eq(population)
        & frame.timing.eq(timing)
        & frame.model.eq(model)
        & frame.seed_base.eq(seed)
    ].copy()
    if selected.empty or selected.patient_id.duplicated().any():
        raise ValueError(f"missing/duplicate OOF group {arm}/{population}/{timing}/{model}/{seed}")
    return selected.sort_values("patient_id", kind="stable").reset_index(drop=True)


def run_bootstrap_effects(
    pcr_oof: pd.DataFrame, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparisons = (
        ("full_808", "MRI_full_vs_zR", "z_R", "z_R+z_P"),
        ("full_808", "beyond_C_full_vs_zR", "C+z_R", "C+z_R+z_P"),
        ("full_808", "beyond_C_zP_vs_C", "C", "C+z_P"),
        ("ftv_complete_375", "beyond_C_F_full_vs_zR", "C+F+z_R", "C+F+z_R+z_P"),
        ("ftv_complete_375", "beyond_C_F_zP_vs_C_F", "C+F", "C+F+z_P"),
    )
    bootstrap = config["bootstrap"]
    summaries: list[dict[str, Any]] = []
    draw_frames: list[pd.DataFrame] = []
    comparison_index = 0
    for arm in ARMS:
        for seed in SEEDS:
            for timing in TIMINGS:
                for population, comparison, baseline_model, augmented_model in comparisons:
                    baseline = _paired_group(pcr_oof, arm, population, timing, baseline_model, seed)
                    augmented = _paired_group(pcr_oof, arm, population, timing, augmented_model, seed)
                    if not np.array_equal(baseline.patient_id, augmented.patient_id) or not np.array_equal(
                        baseline.label_pcr, augmented.label_pcr
                    ):
                        raise AssertionError("paired OOF comparison is not patient/label aligned")
                    random_seed = int(bootstrap["random_seed"]) + 1009 * comparison_index
                    summary, draws = paired_stratified_bootstrap(
                        baseline.patient_id,
                        baseline.fold,
                        baseline.label_pcr,
                        baseline.probability,
                        augmented.probability,
                        n_bootstrap=int(bootstrap["replicates"]),
                        confidence_level=float(bootstrap["confidence_level"]),
                        random_state=random_seed,
                    )
                    common = {
                        "arm": arm,
                        "seed_base": seed,
                        "population": population,
                        "timing": timing,
                        "comparison": comparison,
                        "baseline_model": baseline_model,
                        "augmented_model": augmented_model,
                    }
                    summaries.append(common | summary)
                    draws = draws.copy()
                    draws.insert(0, "draw", np.arange(len(draws), dtype=int))
                    for column, value in reversed(tuple(common.items())):
                        draws.insert(0, column, value)
                    draw_frames.append(draws)
                    comparison_index += 1

    # Directly paired representation-arm contrasts and F0 context.
    cross_comparisons = (
        ("full_808", "factorized_F1_vs_F0_MRI", "F0", "F0", "F1", "z_R+z_P"),
        ("full_808", "factorized_F2_vs_F0_MRI", "F0", "F0", "F2", "z_R+z_P"),
        ("ftv_complete_375", "factorized_F1_vs_F0_beyond_CF", "F0", "C+F+F0", "F1", "C+F+z_R+z_P"),
        ("ftv_complete_375", "factorized_F2_vs_F0_beyond_CF", "F0", "C+F+F0", "F2", "C+F+z_R+z_P"),
        ("ftv_complete_375", "adversarial_F2_vs_F1_full", "F1", "C+F+z_R+z_P", "F2", "C+F+z_R+z_P"),
    )
    for seed in SEEDS:
        for timing in TIMINGS:
            for population, comparison, base_arm, baseline_model, aug_arm, augmented_model in cross_comparisons:
                baseline = _paired_group(pcr_oof, base_arm, population, timing, baseline_model, seed)
                augmented = _paired_group(pcr_oof, aug_arm, population, timing, augmented_model, seed)
                if not np.array_equal(baseline.patient_id, augmented.patient_id) or not np.array_equal(
                    baseline.label_pcr, augmented.label_pcr
                ):
                    raise AssertionError("cross-arm comparison is not patient aligned")
                random_seed = int(bootstrap["random_seed"]) + 1009 * comparison_index
                summary, draws = paired_stratified_bootstrap(
                    baseline.patient_id,
                    baseline.fold,
                    baseline.label_pcr,
                    baseline.probability,
                    augmented.probability,
                    n_bootstrap=int(bootstrap["replicates"]),
                    confidence_level=float(bootstrap["confidence_level"]),
                    random_state=random_seed,
                )
                common = {
                    "arm": aug_arm,
                    "seed_base": seed,
                    "population": population,
                    "timing": timing,
                    "comparison": comparison,
                    "baseline_model": f"{base_arm}:{baseline_model}",
                    "augmented_model": f"{aug_arm}:{augmented_model}",
                }
                summaries.append(common | summary)
                draws = draws.copy()
                draws.insert(0, "draw", np.arange(len(draws), dtype=int))
                for column, value in reversed(tuple(common.items())):
                    draws.insert(0, column, value)
                draw_frames.append(draws)
                comparison_index += 1
    return pd.DataFrame(summaries), pd.concat(draw_frames, ignore_index=True)


def _records(frame: pd.DataFrame, **conditions: Any) -> pd.DataFrame:
    selected = frame
    for column, value in conditions.items():
        selected = selected.loc[selected[column].eq(value)]
    return selected.copy()


def build_decision_summary(
    diagnostics: pd.DataFrame,
    nearest_neighbors: pd.DataFrame,
    response: pd.DataFrame,
    profile: pd.DataFrame,
    pcr: pd.DataFrame,
    bootstrap: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gates_config = config["gates"]
    static_floor = float(gates_config["response_static_ftv_spearman_degradation_floor"])
    response_effects: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    delta_effects: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for arm in ARMS:
        for seed in SEEDS:
            f0_static = _records(response, arm="F0", state="F0", seed_base=seed, task="static", endpoint="macro")
            f0_delta = _records(response, arm="F0", state="F0", seed_base=seed, task="delta", endpoint="macro")
            current_static = _records(response, arm=arm, state="z_R", seed_base=seed, task="static", endpoint="macro")
            current_delta = _records(response, arm=arm, state="z_R", seed_base=seed, task="delta", endpoint="macro")
            if any(len(value) != 1 for value in (f0_static, f0_delta, current_static, current_delta)):
                raise AssertionError("response gate metric cell is missing")
            response_effects[arm][str(seed)]["static_spearman_delta"] = float(
                current_static.iloc[0].spearman - f0_static.iloc[0].spearman
            )
            delta_effects[arm][str(seed)]["delta_spearman_delta"] = float(
                current_delta.iloc[0].spearman - f0_delta.iloc[0].spearman
            )
    response_effects_finite = all(
        math.isfinite(value["static_spearman_delta"])
        for arm in response_effects.values()
        for value in arm.values()
    ) and all(
        math.isfinite(value["delta_spearman_delta"])
        for arm in delta_effects.values()
        for value in arm.values()
    )
    static_pass = response_effects_finite and all(
        value["static_spearman_delta"] >= static_floor
        for arm in response_effects.values()
        for value in arm.values()
    )
    # "Systematic" means both independent seeds degrade for the same arm.
    no_systematic_delta = response_effects_finite and all(
        not all(values[str(seed)]["delta_spearman_delta"] < 0.0 for seed in SEEDS)
        for values in delta_effects.values()
    )
    gate_a = static_pass and no_systematic_delta

    threshold_rank = float(config["diagnostics"]["effective_rank_floor"])
    threshold_std = float(config["diagnostics"]["phenotype_mean_std_floor"])
    threshold_aug = float(config["diagnostics"]["augmentation_cosine_floor"])
    f1_diag = diagnostics.loc[diagnostics.arm.eq("F1")]
    f2_diag = diagnostics.loc[diagnostics.arm.eq("F2")]
    f0_diag = diagnostics.loc[diagnostics.arm.eq("F0")]
    f1_noncollapse = bool(
        f1_diag.z_P_effective_rank.ge(threshold_rank).all()
        and f1_diag.z_P_mean_std.ge(threshold_std).all()
    )
    f2_noncollapse = bool(
        f2_diag.z_P_effective_rank.ge(threshold_rank).all()
        and f2_diag.z_P_mean_std.ge(threshold_std).all()
    )
    crosscov_by_seed: dict[str, dict[str, float]] = {}
    crosscov_lower = True
    for seed in SEEDS:
        f1_value = float(f1_diag.loc[f1_diag.seed_base.eq(seed), "standardized_crosscov_rms"].mean())
        f0_value = float(f0_diag.loc[f0_diag.seed_base.eq(seed), "standardized_crosscov_rms"].mean())
        crosscov_by_seed[str(seed)] = {"F1": f1_value, "F0_unseparated_halves": f0_value, "delta": f1_value - f0_value}
        crosscov_lower &= f1_value < f0_value
    def image_safeguard(frame: pd.DataFrame) -> bool:
        return bool(
            frame.optional_diagnostic_tensors_present.all()
            and frame.augmentation_mean_cosine.ge(threshold_aug).all()
            and np.isfinite(frame.future_phenotype_mse).all()
            and frame.future_mse_improvement_over_persistence.gt(0.0).all()
        )

    f1_image_information = image_safeguard(f1_diag)
    f2_image_information = image_safeguard(f2_diag)
    gate_b = bool(f1_noncollapse and crosscov_lower and f1_image_information)

    redundancy: dict[str, dict[str, dict[str, float | bool]]] = {}
    consistent_targets: list[str] = []
    for target in ("label_hr", "label_her2"):
        redundancy[target] = {}
        consistent = True
        for seed in SEEDS:
            f1 = _records(
                profile, arm="F1", state="z_P", target=target,
                endpoint="T0_T1_T2_macro", seed_base=seed,
            )
            f2 = _records(
                profile, arm="F2", state="z_P", target=target,
                endpoint="T0_T1_T2_macro", seed_base=seed,
            )
            if len(f1) != 1 or len(f2) != 1:
                raise AssertionError("profile redundancy metric cell is missing")
            f1_auroc = float(f1.iloc[0].auroc)
            f2_auroc = float(f2.iloc[0].auroc)
            # Each static endpoint is first converted to a symmetric linear
            # association score, then T0/T1/T2 are averaged.  Transforming
            # only the mean raw AUROC could cancel equally decodable probes on
            # opposite sides of chance.
            f1_decodability = float(f1.iloc[0].flip_invariant_decodability)
            f2_decodability = float(f2.iloc[0].flip_invariant_decodability)
            delta = f2_decodability - f1_decodability
            redundancy[target][str(seed)] = {
                "F1_raw_auroc": f1_auroc,
                "F2_raw_auroc": f2_auroc,
                "F1_flip_invariant_decodability": f1_decodability,
                "F2_flip_invariant_decodability": f2_decodability,
                "F2_minus_F1_decodability": delta,
                "decreased": delta < 0.0,
            }
            consistent &= delta < 0.0
        if consistent:
            consistent_targets.append(target)
    gate_c = bool(consistent_targets and f2_noncollapse and f2_image_information)

    complementarity: dict[str, dict[str, float]] = {}
    timing_passes: list[str] = []
    strong_timings: list[str] = []
    for timing in gates_config["phenotype_complementarity_timings"]:
        cells = _records(
            bootstrap,
            arm="F2",
            comparison="beyond_C_F_full_vs_zR",
            timing=timing,
        )
        values = {
            str(int(row.seed_base)): float(row.delta_auroc)
            for row in cells.itertuples(index=False)
        }
        if set(values) != {str(seed) for seed in SEEDS}:
            raise AssertionError("critical complementarity bootstrap cells are missing")
        complementarity[timing] = values
        both_positive = all(value > 0.0 for value in values.values())
        if both_positive:
            timing_passes.append(timing)
        if both_positive and float(np.mean(list(values.values()))) >= float(
            gates_config["phenotype_complementarity_strong_mean"]
        ):
            strong_timings.append(timing)
    gate_d = bool(timing_passes)

    gate_payload = {
        "A_RESPONSE_PRESERVED": {
            "pass": gate_a,
            "label": "RESPONSE_STATE_PRESERVED" if gate_a else "RESPONSE_STATE_NOT_PRESERVED",
            "static_floor": static_floor,
            "static_effects": response_effects,
            "delta_effects": delta_effects,
            "all_response_effects_finite": response_effects_finite,
            "delta_systematic_degradation": not no_systematic_delta,
        },
        "B_FACTORIZATION_WORKS": {
            "pass": gate_b,
            "label": "FACTORISED_STATE_VALID" if gate_b else "FACTORISED_STATE_NOT_VALID",
            "F1_noncollapsed": f1_noncollapse,
            "crosscov_lower_than_F0_unseparated_control": crosscov_lower,
            "crosscov_aggregation": (
                "within_seed_unweighted_mean_of_5_fold_outer_test_"
                "standardized_crosscov_rms; both_seeds_strictly_lower"
            ),
            "crosscov_by_seed": crosscov_by_seed,
            "image_information_safeguard": f1_image_information,
            "image_information_retained": bool(
                f1_noncollapse and f1_image_information
            ),
            "F1_future_prediction_beats_persistence_every_fold": f1_image_information,
            "F1_nearest_neighbor_mean_jaccard": (
                float(
                    nearest_neighbors.loc[
                        nearest_neighbors.arm.eq("F1")
                        & nearest_neighbors.fold.eq(-1),
                        "mean_jaccard",
                    ].iloc[0]
                )
                if math.isfinite(
                    float(
                        nearest_neighbors.loc[
                            nearest_neighbors.arm.eq("F1")
                            & nearest_neighbors.fold.eq(-1),
                            "mean_jaccard",
                        ].iloc[0]
                    )
                )
                else None
            ),
        },
        "C_CLINICAL_REDUNDANCY_REDUCED": {
            "pass": gate_c,
            "label": "CLINICAL_REDUNDANCY_REDUCED" if gate_c else "CLINICAL_REDUNDANCY_NOT_REDUCED",
            "targets_decreased_in_both_seeds": consistent_targets,
            "F2_noncollapsed": f2_noncollapse,
            "F2_image_information_safeguard": f2_image_information,
            "F2_image_information_retained": bool(
                f2_noncollapse and f2_image_information
            ),
            "F2_future_prediction_beats_persistence_every_fold": f2_image_information,
            "profile_auroc": redundancy,
        },
        "D_PHENOTYPE_COMPLEMENTARITY": {
            "pass": gate_d,
            "label": "CLINICAL_RESIDUAL_PHENOTYPE_VALUE_SUPPORTED"
            if gate_d
            else "CLINICAL_RESIDUAL_PHENOTYPE_VALUE_NOT_SUPPORTED",
            "critical_comparison": "C+F+z_R+z_P minus C+F+z_R",
            "effects_by_timing_seed": complementarity,
            "positive_both_seed_timings": timing_passes,
            "strong_mean_ge_0_03_timings": strong_timings,
        },
    }

    factorization_ranking: dict[str, Any] = {
        "F1_mean_auroc": None,
        "F2_mean_auroc": None,
        "F0_mean_auroc": None,
        "F1_strictly_best": False,
        "F1_phenotype_positive_both_seed_timings": [],
    }
    f1_positive_timings: list[str] = []
    for timing in gates_config["phenotype_complementarity_timings"]:
        cells = _records(
            bootstrap,
            arm="F1",
            comparison="beyond_C_F_full_vs_zR",
            timing=timing,
        )
        values = {
            int(row.seed_base): float(row.delta_auroc)
            for row in cells.itertuples(index=False)
        }
        if set(values) == set(SEEDS) and all(value > 0.0 for value in values.values()):
            f1_positive_timings.append(timing)
    if not pcr.empty:
        ranking_cells = pcr.loc[
            pcr.population.eq("ftv_complete_375")
            & pcr.timing.isin(("T0-T1", "T0-T2"))
            & (
                (pcr.arm.eq("F1") & pcr.model.eq("C+F+z_R+z_P"))
                | (pcr.arm.eq("F2") & pcr.model.eq("C+F+z_R+z_P"))
                | (pcr.arm.eq("F0") & pcr.model.eq("C+F+F0"))
            )
        ]
        means = ranking_cells.groupby("arm").auroc.mean().to_dict()
        if set(means) == {"F0", "F1", "F2"}:
            factorization_ranking = {
                "F1_mean_auroc": float(means["F1"]),
                "F2_mean_auroc": float(means["F2"]),
                "F0_mean_auroc": float(means["F0"]),
                "F1_strictly_best": bool(
                    means["F1"] > means["F0"] and means["F1"] > means["F2"]
                ),
                "F1_phenotype_positive_both_seed_timings": f1_positive_timings,
            }

    if all(gate_payload[key]["pass"] for key in gate_payload):
        classification = "CLINICAL-RESIDUAL PHENOTYPE STATE VALIDATED"
        class_code = "A"
    elif (
        gate_a
        and gate_b
        and bool(factorization_ranking["F1_strictly_best"])
        and bool(f1_positive_timings)
    ):
        classification = "FACTORIZATION HELPS, ADVERSARIAL RESIDUALIZATION DOES NOT"
        class_code = "B"
    elif gate_a and gate_b and gate_c and not gate_d:
        classification = "CLINICAL RESIDUALIZATION WORKS BUT NO pCR VALUE"
        class_code = "C"
    else:
        classification = "FACTORIZATION NOT SUPPORTED"
        class_code = "D"

    # Compact aggregate evidence used verbatim by the report generator.
    mri_increment: dict[str, dict[str, float]] = {}
    clinical_increment: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        mri_increment[arm], clinical_increment[arm] = {}, {}
        for seed in SEEDS:
            rows = _records(
                bootstrap, arm=arm, seed_base=seed,
                comparison="MRI_full_vs_zR", timing="T0-T2",
            )
            clinical_rows = _records(
                bootstrap, arm=arm, seed_base=seed,
                comparison="beyond_C_full_vs_zR", timing="T0-T2",
            )
            mri_increment[arm][str(seed)] = float(rows.iloc[0].delta_auroc)
            clinical_increment[arm][str(seed)] = float(clinical_rows.iloc[0].delta_auroc)
    return {
        "schema_version": 1,
        "experiment": "clinical_residual_phenotype_state",
        "phase": "post_export_evaluation",
        "representation_frozen_before_pcr_access": True,
        "feature_cells": {"factorized": 20, "F0": 10},
        "patient_level_outputs_committed": False,
        "gates": gate_payload,
        "classification": {"code": class_code, "label": classification},
        "supporting_effects": {
            "MRI_full_minus_zR_T0_T2": mri_increment,
            "clinical_full_minus_zR_T0_T2": clinical_increment,
            "factorization_arm_ranking_T0_T1_T0_T2": factorization_ranking,
        },
        "bootstrap": {
            "replicates": int(config["bootstrap"]["replicates"]),
            "unit": "patient",
            "stratification": "outer_fold_x_outcome",
            "confidence_level": float(config["bootstrap"]["confidence_level"]),
        },
        "interpretation_scope": "two-seed internal OOF pilot; no external validation",
    }


def run_evaluation(
    *,
    config_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Execute evaluation after all 20 factorized exports are immutable."""

    config_source = Path(
        config_path or EXPERIMENT_ROOT / "configs" / "evaluation.json"
    ).resolve()
    config = load_evaluation_config(config_source)
    # Construct the complete feature/response view without opening any outcome
    # column.  The immutable evaluation lock binds these exact assets and every
    # evaluator byte; only a successful verification permits load_clinical().
    folds = load_fold_assignments(config)
    ftv_records, ftv_provenance = load_stage_b_ftv_records(config, folds)
    factorized, f0 = load_all_assets(config, folds)
    evaluation_boundary = require_before_outcome_access(config_path=config_source)
    clinical = load_clinical(config, folds)
    ftv = load_ftv_wide(config, set(clinical.patient_id.astype(str)))
    metrics_dir = EXPERIMENT_ROOT / "metrics"
    predictions_dir = EXPERIMENT_ROOT / "predictions"
    public_paths = {
        "diagnostics": metrics_dir / "state_diagnostics.csv",
        "dimension_diagnostics": metrics_dir / "state_dimension_diagnostics.csv",
        "eigenspectra": metrics_dir / "state_covariance_eigenspectra.csv",
        "nearest": metrics_dir / "nearest_neighbor_stability.csv",
        "response": metrics_dir / "response_metrics.csv",
        "profile": metrics_dir / "phenotype_probes.csv",
        "pcr": metrics_dir / "pcr_metrics.csv",
        "bootstrap": metrics_dir / "paired_bootstrap_effects.csv",
        "decision": metrics_dir / "decision_summary.json",
    }
    private_paths = {
        "response": predictions_dir / "response_oof.private.csv",
        "profile": predictions_dir / "profile_oof.private.csv",
        "pcr": predictions_dir / "pcr_oof.private.csv",
        "draws": predictions_dir / "bootstrap_draws.private.csv",
    }
    existing = [path for path in (*public_paths.values(), *private_paths.values()) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("evaluation outputs exist; pass --overwrite for known artifacts")

    diagnostics, nearest = compute_state_diagnostics(factorized, f0, config)
    dimension_diagnostics, eigenspectra = expand_state_diagnostic_tables(diagnostics)
    response_metrics, response_oof = run_matched_response_probes(
        [*factorized, *f0],
        ftv_records,
        alphas=tuple(float(value) for value in config["ridge_alphas"]),
        expected_measurement_valid_patient_count=int(
            config["frozen_inputs"]["expected_ftv_patient_count"]
        ),
    )
    profile_metrics, profile_oof = run_profile_probes([*factorized, *f0], clinical, config)
    pcr_metrics, pcr_oof = run_pcr_models([*factorized, *f0], clinical, ftv, config)
    bootstrap_effects, bootstrap_draws = run_bootstrap_effects(pcr_oof, config)
    decision = build_decision_summary(
        diagnostics,
        nearest,
        response_metrics,
        profile_metrics,
        pcr_metrics,
        bootstrap_effects,
        config,
    )
    decision["evaluation_lock"] = {
        "path": str(EVALUATION_LOCK_PATH.relative_to(EXPERIMENT_ROOT)),
        "payload_sha256": evaluation_boundary["lock_sha256"],
        "outcome_access_permitted_only_after_verification": True,
        "stage_b_response_adapter": ftv_provenance["adapter"],
    }
    for key, frame in {
        "diagnostics": diagnostics,
        "dimension_diagnostics": dimension_diagnostics,
        "eigenspectra": eigenspectra,
        "nearest": nearest,
        "response": response_metrics,
        "profile": profile_metrics,
        "pcr": pcr_metrics,
        "bootstrap": bootstrap_effects,
    }.items():
        _atomic_csv(frame, public_paths[key])
    for key, frame in {
        "response": response_oof,
        "profile": profile_oof,
        "pcr": pcr_oof,
        "draws": bootstrap_draws,
    }.items():
        _atomic_csv(frame, private_paths[key])
    _atomic_json(decision, public_paths["decision"])
    return decision


__all__ = [
    "PROFILE_ENDPOINTS",
    "SUBTYPE_CLASSES",
    "TIMINGS",
    "build_decision_summary",
    "compute_state_diagnostics",
    "expand_state_diagnostic_tables",
    "load_all_assets",
    "run_bootstrap_effects",
    "run_evaluation",
    "run_pcr_models",
    "run_profile_probes",
]

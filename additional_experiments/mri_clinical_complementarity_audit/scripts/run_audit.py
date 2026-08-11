#!/usr/bin/env python3
"""Run the frozen MRI--clinical complementarity audit end to end.

The runner consumes the 20 already-exported LOCAL cells.  It never loads MRI
volumes or changes an upstream checkpoint.  All patient-level outputs are
written below gitignored ``predictions/``; only aggregate results are public.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
SCRIPTS_ROOT = EXPERIMENT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from data_contracts import (  # noqa: E402
    LocalFeatureAsset,
    TrainOnlyClinicalEncoder,
    file_sha256,
    ftv_timing_prefix,
    load_all_local_feature_assets,
    load_clinical_table,
    load_config,
    load_fold_manifest,
    load_ftv_wide,
    mri_timing_prefix,
)
from modeling import (  # noqa: E402
    binary_metrics,
    clinical_probability_error,
    fit_binary_logistic,
    fit_clinical_error_ridge,
    fit_ftv_mri_residualizer,
    fit_multiclass_logistic,
    multiclass_metrics,
    paired_fold_stratified_bootstrap,
)


TIMINGS = ("T0", "T1", "T2", "T3")
PROFILE_VIEWS: Mapping[str, tuple[str, int]] = {
    "T0": ("static", 0),
    "T1": ("static", 1),
    "T2": ("static", 2),
    "T3": ("static", 3),
    "long_T0_T1": ("prefix", 1),
    "long_T0_T2": ("prefix", 2),
    "long_T0_T3": ("prefix", 3),
}
SUBTYPE_CLASSES = tuple(sorted(("HR+/HER2-", "HR-/HER2-", "HR+/HER2+", "HR-/HER2+")))
SUBTYPE_PROBABILITY_COLUMNS: Mapping[str, str] = {
    "HR+/HER2-": "prob_hr_pos_her2_neg",
    "HR-/HER2-": "prob_hr_neg_her2_neg",
    "HR+/HER2+": "prob_hr_pos_her2_pos",
    "HR-/HER2+": "prob_hr_neg_her2_pos",
}
PRIMARY_MODELS = (
    "C",
    "M",
    "F",
    "C+F",
    "C+M",
    "C+F+M",
    "M_residual",
    "C+F+M_residual",
    "C+M_error_correction",
)
FULL_MODELS = ("C", "M", "C+M", "C+M_error_correction")
CLINICAL_BASELINE_CONTRACTS = (
    "C1_hr_her2",
    "C_condition_without_treatment",
    "C_condition_with_treatment",
    "C2_full_without_treatment",
    "C2_full_with_treatment",
)
SUBGROUPS = ("HR+/HER2-", "HR-/HER2-", "HER2+")


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
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
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _require_output_policy(paths: Sequence[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        preview = "\n".join(str(path) for path in existing[:5])
        raise FileExistsError(
            "formal outputs already exist; pass --overwrite to replace only the "
            f"known audit artifacts:\n{preview}"
        )


def _stable_group_key(values: Any) -> tuple[Any, ...]:
    return values if isinstance(values, tuple) else (values,)


def _aligned_clinical(
    clinical: pd.DataFrame, patient_ids: Sequence[str]
) -> pd.DataFrame:
    indexed = clinical.set_index("patient_id", verify_integrity=True)
    missing = sorted(set(str(value) for value in patient_ids) - set(indexed.index))
    if missing:
        raise ValueError(f"clinical table misses LOCAL patients: {missing[:5]}")
    return indexed.loc[[str(value) for value in patient_ids]].reset_index()


def _aligned_ftv(ftv: pd.DataFrame, patient_ids: Sequence[str]) -> pd.DataFrame:
    indexed = ftv.set_index("patient_id", verify_integrity=True)
    missing = sorted(set(str(value) for value in patient_ids) - set(indexed.index))
    if missing:
        raise ValueError(f"FTV table misses selected patients: {missing[:5]}")
    return indexed.loc[[str(value) for value in patient_ids]].reset_index()


def _population_mask(
    asset: LocalFeatureAsset, ftv_ids: set[str], population: str
) -> np.ndarray:
    if population == "full_808":
        return np.ones(len(asset.patient_id), dtype=bool)
    if population == "ftv_complete_375":
        return np.asarray([value in ftv_ids for value in asset.patient_id], dtype=bool)
    raise ValueError(f"unknown population: {population}")


def _split_indices(split: np.ndarray) -> dict[str, np.ndarray]:
    output = {name: np.flatnonzero(split == name) for name in ("train", "val", "test")}
    if any(len(values) == 0 for values in output.values()):
        raise ValueError("train/val/test split must be non-empty")
    if sum(len(values) for values in output.values()) != len(split):
        raise ValueError("split contains an unknown label")
    return output


def _clinical_matrix(
    config: Mapping[str, Any],
    contract: str,
    clinical: pd.DataFrame,
    indices: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, TrainOnlyClinicalEncoder]:
    encoder = TrainOnlyClinicalEncoder.from_config(config, contract)
    encoder.fit(clinical.iloc[indices["train"]])
    matrix = encoder.transform(clinical)
    return matrix, encoder


def _fit_binary(
    matrix: np.ndarray,
    labels: np.ndarray,
    indices: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    *,
    class_weight: str | Mapping[int, float] | None = None,
) -> Any:
    logistic = config["logistic"]
    return fit_binary_logistic(
        matrix[indices["train"]],
        labels[indices["train"]],
        matrix[indices["val"]],
        labels[indices["val"]],
        logistic["c_grid"],
        class_weight=class_weight,
        solver=str(logistic["solver"]),
        max_iter=int(logistic["max_iter"]),
        random_state=0,
    )


def _profile_matrix(state: np.ndarray, view: str) -> np.ndarray:
    kind, index = PROFILE_VIEWS[view]
    if kind == "static":
        return np.asarray(state[:, index, :], dtype=np.float64)
    return np.asarray(mri_timing_prefix(state, index), dtype=np.float64)


def _binary_metric_row(frame: pd.DataFrame, *, fold_thresholds: bool) -> dict[str, Any]:
    labels = frame["y_true"].to_numpy(dtype=np.int64)
    probability = frame["predicted_probability"].to_numpy(dtype=np.float64)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("metric group must contain both classes")
    if fold_thresholds:
        prediction = frame["predicted_label"].to_numpy(dtype=np.int64)
    else:
        prediction = (probability >= 0.5).astype(np.int64)
    return {
        "n": int(len(frame)),
        "n_positive": int(labels.sum()),
        "n_negative": int((labels == 0).sum()),
        "auroc": float(roc_auc_score(labels, probability)),
        "auprc": float(average_precision_score(labels, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "brier": float(np.mean(np.square(probability - labels))),
    }


def _aggregate_binary_predictions(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
    *,
    fold_thresholds: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(list(group_columns), sort=True, dropna=False):
        values = _stable_group_key(key)
        if group["patient_id"].duplicated().any():
            raise ValueError(
                f"OOF group repeats patients: {dict(zip(group_columns, values))}"
            )
        rows.append(
            {
                **dict(zip(group_columns, values, strict=True)),
                **_binary_metric_row(group, fold_thresholds=fold_thresholds),
            }
        )
    return pd.DataFrame(rows)


def _append_binary_predictions(
    rows: list[dict[str, Any]],
    *,
    patient_ids: np.ndarray,
    fold: int,
    labels: np.ndarray,
    test_indices: np.ndarray,
    fit: Any,
    matrix: np.ndarray,
    metadata: Mapping[str, Any],
) -> None:
    probability = fit.predict_proba(matrix[test_indices])
    prediction = (probability >= fit.threshold_selection.threshold).astype(np.int64)
    for offset, row_index in enumerate(test_indices):
        rows.append(
            {
                "patient_id": str(patient_ids[row_index]),
                "fold": int(fold),
                **dict(metadata),
                "y_true": int(labels[row_index]),
                "predicted_probability": float(probability[offset]),
                "predicted_label": int(prediction[offset]),
                "threshold": float(fit.threshold_selection.threshold),
            }
        )


def _hyperparameter_row(
    fit: Any,
    *,
    task: str,
    fold: int,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "task": task,
        "fold": int(fold),
        **dict(metadata),
        "selected_C": float(fit.selected_c),
        "validation_auroc": float(fit.validation_auroc),
        "validation_threshold": float(fit.threshold_selection.threshold),
        "train_rows": int(fit.train_rows),
        "validation_rows": int(fit.validation_rows),
        "feature_dim": int(fit.feature_dim),
    }


def run_profile_probes(
    config: Mapping[str, Any],
    clinical: pd.DataFrame,
    assets: Mapping[tuple[int, str, int], LocalFeatureAsset],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    hyperparameters: list[dict[str, Any]] = []
    profile_weight = config["profile_logistic"]["class_weight"]
    for (seed, arm, fold), asset in sorted(assets.items()):
        aligned = _aligned_clinical(clinical, asset.patient_id)
        split_indices = _split_indices(asset.split)
        for view in PROFILE_VIEWS:
            matrix = _profile_matrix(asset.response_state, view)
            for target, column in (("HR", "label_hr"), ("HER2", "label_her2")):
                labels = aligned[column].to_numpy(dtype=np.int64)
                fit = _fit_binary(
                    matrix,
                    labels,
                    split_indices,
                    config,
                    class_weight=profile_weight,
                )
                metadata = {"seed": seed, "arm": arm, "view": view, "target": target}
                _append_binary_predictions(
                    rows,
                    patient_ids=asset.patient_id,
                    fold=fold,
                    labels=labels,
                    test_indices=split_indices["test"],
                    fit=fit,
                    matrix=matrix,
                    metadata=metadata,
                )
                hyperparameters.append(
                    _hyperparameter_row(
                        fit, task="profile_binary", fold=fold, metadata=metadata
                    )
                )

            subtype = aligned["hr_her2_subtype"].astype(str).to_numpy()
            train = split_indices["train"]
            validation = split_indices["val"]
            test = split_indices["test"]
            fit_multi = fit_multiclass_logistic(
                matrix[train],
                subtype[train],
                matrix[validation],
                subtype[validation],
                config["logistic"]["c_grid"],
                solver=str(config["logistic"]["solver"]),
                max_iter=int(config["logistic"]["max_iter"]),
                random_state=0,
            )
            probability = fit_multi.predict_proba(matrix[test])
            predicted = fit_multi.predict(matrix[test])
            class_index = {
                str(value): index for index, value in enumerate(fit_multi.classes)
            }
            if set(class_index) != set(SUBTYPE_CLASSES):
                raise ValueError("multiclass profile model class contract drifted")
            for offset, row_index in enumerate(test):
                row: dict[str, Any] = {
                    "patient_id": str(asset.patient_id[row_index]),
                    "fold": int(fold),
                    "seed": int(seed),
                    "arm": str(arm),
                    "view": view,
                    "target": "subtype_4class",
                    "y_true": str(subtype[row_index]),
                    "predicted_probability": math.nan,
                    "predicted_label": str(predicted[offset]),
                    "threshold": math.nan,
                }
                for subtype_name, column in SUBTYPE_PROBABILITY_COLUMNS.items():
                    row[column] = float(probability[offset, class_index[subtype_name]])
                rows.append(row)
            hyperparameters.append(
                {
                    "task": "profile_multiclass",
                    "fold": int(fold),
                    "seed": int(seed),
                    "arm": str(arm),
                    "view": view,
                    "target": "subtype_4class",
                    "selected_C": float(fit_multi.selected_c),
                    "validation_auroc": float(fit_multi.validation_macro_ovr_auroc),
                    "validation_threshold": math.nan,
                    "train_rows": int(fit_multi.train_rows),
                    "validation_rows": int(fit_multi.validation_rows),
                    "feature_dim": int(fit_multi.feature_dim),
                }
            )

    predictions = pd.DataFrame(rows)
    metric_rows: list[dict[str, Any]] = []
    group_columns = ["seed", "arm", "view", "target"]
    for key, group in predictions.groupby(group_columns, sort=True):
        seed, arm, view, target = key
        if group["patient_id"].duplicated().any() or len(group) != 808:
            raise ValueError(f"profile OOF coverage drifted for {key}")
        if target != "subtype_4class":
            values = _binary_metric_row(group, fold_thresholds=True)
        else:
            probabilities = group[
                [SUBTYPE_PROBABILITY_COLUMNS[value] for value in SUBTYPE_CLASSES]
            ].to_numpy(dtype=np.float64)
            multi = multiclass_metrics(
                group["y_true"].astype(str).to_numpy(),
                probabilities,
                classes=SUBTYPE_CLASSES,
            )
            values = {
                "n": int(multi["n"]),
                "n_positive": math.nan,
                "n_negative": math.nan,
                "auroc": float(multi["macro_ovr_auroc"]),
                "auprc": float(multi["macro_ovr_auprc"]),
                "balanced_accuracy": float(multi["balanced_accuracy"]),
                "brier": math.nan,
            }
        metric_rows.append(
            {"seed": seed, "arm": arm, "view": view, "target": target, **values}
        )
    return predictions, pd.DataFrame(metric_rows), hyperparameters


def run_clinical_baselines(
    config: Mapping[str, Any],
    clinical: pd.DataFrame,
    ftv_ids: set[str],
    assets: Mapping[tuple[int, str, int], LocalFeatureAsset],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    hyperparameters: list[dict[str, Any]] = []
    for population in ("full_808", "ftv_complete_375"):
        for fold in range(5):
            asset = assets[(2026, "LOCAL0", fold)]
            population_mask = _population_mask(asset, ftv_ids, population)
            patient_ids = asset.patient_id[population_mask]
            split = asset.split[population_mask]
            aligned = _aligned_clinical(clinical, patient_ids)
            labels = aligned["label_pcr"].to_numpy(dtype=np.int64)
            indices = _split_indices(split)
            for contract in CLINICAL_BASELINE_CONTRACTS:
                matrix, _ = _clinical_matrix(config, contract, aligned, indices)
                fit = _fit_binary(matrix, labels, indices, config)
                metadata = {"population": population, "clinical_contract": contract}
                _append_binary_predictions(
                    rows,
                    patient_ids=patient_ids,
                    fold=fold,
                    labels=labels,
                    test_indices=indices["test"],
                    fit=fit,
                    matrix=matrix,
                    metadata=metadata,
                )
                hyperparameters.append(
                    _hyperparameter_row(
                        fit, task="clinical_baseline", fold=fold, metadata=metadata
                    )
                )
    predictions = pd.DataFrame(rows)
    metrics = _aggregate_binary_predictions(
        predictions, ["population", "clinical_contract"]
    )
    expected = {"full_808": 808, "ftv_complete_375": 375}
    for row in metrics.itertuples(index=False):
        if int(row.n) != expected[str(row.population)]:
            raise ValueError("clinical baseline OOF coverage drifted")
    return predictions, metrics, hyperparameters


def _pcr_feature_sets(
    clinical_matrix: np.ndarray,
    mri: np.ndarray,
    ftv: np.ndarray | None,
    mri_residual: np.ndarray | None,
    population: str,
) -> Mapping[str, np.ndarray]:
    if population == "full_808":
        return {
            "C": clinical_matrix,
            "M": mri,
            "C+M": np.concatenate((clinical_matrix, mri), axis=1),
        }
    if ftv is None or mri_residual is None:
        raise ValueError("FTV population requires FTV and residual MRI matrices")
    return {
        "C": clinical_matrix,
        "M": mri,
        "F": ftv,
        "C+F": np.concatenate((clinical_matrix, ftv), axis=1),
        "C+M": np.concatenate((clinical_matrix, mri), axis=1),
        "C+F+M": np.concatenate((clinical_matrix, ftv, mri), axis=1),
        "M_residual": mri_residual,
        "C+F+M_residual": np.concatenate((clinical_matrix, ftv, mri_residual), axis=1),
    }


def run_pcr_models(
    config: Mapping[str, Any],
    clinical: pd.DataFrame,
    ftv_wide: pd.DataFrame,
    assets: Mapping[tuple[int, str, int], LocalFeatureAsset],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[dict[str, Any]],
]:
    prediction_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    hyperparameters: list[dict[str, Any]] = []
    ftv_ids = set(ftv_wide["patient_id"].astype(str))
    primary_contract = str(config["primary_clinical_contract"])
    ridge_alphas = config["residual_ridge_alphas"]

    for (seed, arm, fold), asset in sorted(assets.items()):
        for population in ("full_808", "ftv_complete_375"):
            population_mask = _population_mask(asset, ftv_ids, population)
            patient_ids = asset.patient_id[population_mask]
            split = asset.split[population_mask]
            state = asset.response_state[population_mask]
            aligned = _aligned_clinical(clinical, patient_ids)
            labels = aligned["label_pcr"].to_numpy(dtype=np.int64)
            indices = _split_indices(split)
            clinical_matrix, _ = _clinical_matrix(
                config, primary_contract, aligned, indices
            )
            aligned_ftv = (
                _aligned_ftv(ftv_wide, patient_ids)
                if population == "ftv_complete_375"
                else None
            )

            for timing in TIMINGS:
                mri = np.asarray(mri_timing_prefix(state, timing), dtype=np.float64)
                ftv = (
                    np.asarray(ftv_timing_prefix(aligned_ftv, timing), dtype=np.float64)
                    if aligned_ftv is not None
                    else None
                )
                mri_residual: np.ndarray | None = None
                if ftv is not None:
                    residualizer = fit_ftv_mri_residualizer(
                        ftv[indices["train"]], mri[indices["train"]]
                    )
                    mri_residual = residualizer.transform(ftv, mri)

                feature_sets = _pcr_feature_sets(
                    clinical_matrix, mri, ftv, mri_residual, population
                )
                fits: dict[str, Any] = {}
                for model_name, matrix in feature_sets.items():
                    fit = _fit_binary(matrix, labels, indices, config)
                    fits[model_name] = fit
                    metadata = {
                        "population": population,
                        "seed": int(seed),
                        "arm": str(arm),
                        "timing": timing,
                        "model": model_name,
                        "clinical_contract": primary_contract,
                    }
                    _append_binary_predictions(
                        prediction_rows,
                        patient_ids=patient_ids,
                        fold=fold,
                        labels=labels,
                        test_indices=indices["test"],
                        fit=fit,
                        matrix=matrix,
                        metadata=metadata,
                    )
                    hyperparameters.append(
                        _hyperparameter_row(
                            fit, task="pcr", fold=fold, metadata=metadata
                        )
                    )

                # Secondary clinical-error test.  Clinical probabilities are
                # generated by the already locked train-fitted C model.
                clinical_fit = fits["C"]
                probability = {
                    name: clinical_fit.predict_proba(clinical_matrix[index])
                    for name, index in indices.items()
                }
                error = {
                    name: clinical_probability_error(labels[index], probability[name])
                    for name, index in indices.items()
                }
                error_fit = fit_clinical_error_ridge(
                    mri[indices["train"]],
                    error["train"],
                    mri[indices["val"]],
                    error["val"],
                    ridge_alphas,
                )
                test_error_prediction = error_fit.predict(mri[indices["test"]])
                corrected = np.clip(
                    probability["test"] + test_error_prediction, 0.0, 1.0
                )
                for offset, row_index in enumerate(indices["test"]):
                    shared = {
                        "patient_id": str(patient_ids[row_index]),
                        "fold": int(fold),
                        "population": population,
                        "seed": int(seed),
                        "arm": str(arm),
                        "timing": timing,
                    }
                    prediction_rows.append(
                        {
                            **shared,
                            "model": "C+M_error_correction",
                            "clinical_contract": primary_contract,
                            "y_true": int(labels[row_index]),
                            "predicted_probability": float(corrected[offset]),
                            "predicted_label": int(corrected[offset] >= 0.5),
                            "threshold": 0.5,
                        }
                    )
                    residual_rows.append(
                        {
                            **shared,
                            "y_true": int(labels[row_index]),
                            "clinical_probability": float(probability["test"][offset]),
                            "clinical_error_true": float(error["test"][offset]),
                            "clinical_error_predicted": float(
                                test_error_prediction[offset]
                            ),
                            "corrected_probability": float(corrected[offset]),
                            "selected_alpha": float(error_fit.selected_alpha),
                            "validation_mse": float(error_fit.validation_mse),
                        }
                    )
                hyperparameters.append(
                    {
                        "task": "clinical_error_ridge",
                        "fold": int(fold),
                        "population": population,
                        "seed": int(seed),
                        "arm": str(arm),
                        "timing": timing,
                        "model": "C+M_error_correction",
                        "clinical_contract": primary_contract,
                        "selected_C": math.nan,
                        "selected_alpha": float(error_fit.selected_alpha),
                        "validation_auroc": math.nan,
                        "validation_mse": float(error_fit.validation_mse),
                        "validation_threshold": math.nan,
                        "train_rows": int(error_fit.train_rows),
                        "validation_rows": int(error_fit.validation_rows),
                        "feature_dim": int(error_fit.feature_dim),
                    }
                )

    predictions = pd.DataFrame(prediction_rows)
    group_columns = [
        "population",
        "seed",
        "arm",
        "timing",
        "model",
        "clinical_contract",
    ]
    metrics = _aggregate_binary_predictions(predictions, group_columns)
    fold_metrics = _aggregate_binary_predictions(predictions, [*group_columns, "fold"])
    expected = {"full_808": 808, "ftv_complete_375": 375}
    for row in metrics.itertuples(index=False):
        if int(row.n) != expected[str(row.population)]:
            raise ValueError(f"pCR OOF coverage drifted for {row.population}")

    residual_predictions = pd.DataFrame(residual_rows)
    residual_metric_rows: list[dict[str, Any]] = []
    residual_groups = ["population", "seed", "arm", "timing"]
    for key, group in residual_predictions.groupby(residual_groups, sort=True):
        if group["patient_id"].duplicated().any():
            raise ValueError(f"clinical residual OOF repeats patients for {key}")
        truth = group["clinical_error_true"].to_numpy(dtype=np.float64)
        predicted = group["clinical_error_predicted"].to_numpy(dtype=np.float64)
        labels = group["y_true"].to_numpy(dtype=np.int64)
        clinical_probability = group["clinical_probability"].to_numpy(dtype=np.float64)
        corrected_probability = group["corrected_probability"].to_numpy(
            dtype=np.float64
        )
        variance = float(np.sum(np.square(truth - np.mean(truth))))
        clinical_metric = binary_metrics(labels, clinical_probability)
        corrected_metric = binary_metrics(labels, corrected_probability)
        residual_metric_rows.append(
            {
                **dict(zip(residual_groups, key, strict=True)),
                "n": int(len(group)),
                "mse": float(np.mean(np.square(truth - predicted))),
                "r2": (
                    float(1.0 - np.sum(np.square(truth - predicted)) / variance)
                    if variance > 0
                    else math.nan
                ),
                "pearson": float(pearsonr(truth, predicted).statistic),
                "spearman": float(spearmanr(truth, predicted).statistic),
                "clinical_auroc": float(clinical_metric["auroc"]),
                "corrected_auroc": float(corrected_metric["auroc"]),
                "delta_auroc": float(
                    corrected_metric["auroc"] - clinical_metric["auroc"]
                ),
                "clinical_auprc": float(clinical_metric["auprc"]),
                "corrected_auprc": float(corrected_metric["auprc"]),
                "delta_auprc": float(
                    corrected_metric["auprc"] - clinical_metric["auprc"]
                ),
                "clinical_brier": float(clinical_metric["brier"]),
                "corrected_brier": float(corrected_metric["brier"]),
                "brier_improvement": float(
                    clinical_metric["brier"] - corrected_metric["brier"]
                ),
            }
        )
    return (
        predictions,
        metrics,
        fold_metrics,
        residual_predictions,
        pd.DataFrame(residual_metric_rows),
        hyperparameters,
    )


def _subgroup_label(frame: pd.DataFrame) -> np.ndarray:
    return np.where(
        frame["label_her2"].to_numpy(dtype=np.int64) == 1,
        "HER2+",
        np.where(
            frame["label_hr"].to_numpy(dtype=np.int64) == 1,
            "HR+/HER2-",
            "HR-/HER2-",
        ),
    )


def run_subgroup_models(
    config: Mapping[str, Any],
    clinical: pd.DataFrame,
    assets: Mapping[tuple[int, str, int], LocalFeatureAsset],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    hyperparameters: list[dict[str, Any]] = []
    contract = "C_subtype_remaining_with_treatment"
    for (seed, arm, fold), asset in sorted(assets.items()):
        aligned_all = _aligned_clinical(clinical, asset.patient_id)
        group_values = _subgroup_label(aligned_all)
        for subgroup in SUBGROUPS:
            mask = group_values == subgroup
            patient_ids = asset.patient_id[mask]
            split = asset.split[mask]
            state = asset.response_state[mask]
            aligned = aligned_all.loc[mask].reset_index(drop=True)
            labels = aligned["label_pcr"].to_numpy(dtype=np.int64)
            indices = _split_indices(split)
            clinical_matrix, _ = _clinical_matrix(config, contract, aligned, indices)
            for timing in TIMINGS:
                mri = np.asarray(mri_timing_prefix(state, timing), dtype=np.float64)
                feature_sets = {
                    "remaining_clinical": clinical_matrix,
                    "M": mri,
                    "remaining_clinical+M": np.concatenate(
                        (clinical_matrix, mri), axis=1
                    ),
                }
                for model_name, matrix in feature_sets.items():
                    fit = _fit_binary(matrix, labels, indices, config)
                    metadata = {
                        "seed": int(seed),
                        "arm": str(arm),
                        "timing": timing,
                        "subgroup": subgroup,
                        "model": model_name,
                    }
                    _append_binary_predictions(
                        rows,
                        patient_ids=patient_ids,
                        fold=fold,
                        labels=labels,
                        test_indices=indices["test"],
                        fit=fit,
                        matrix=matrix,
                        metadata=metadata,
                    )
                    hyperparameters.append(
                        _hyperparameter_row(
                            fit, task="subgroup", fold=fold, metadata=metadata
                        )
                    )
    predictions = pd.DataFrame(rows)
    metrics = _aggregate_binary_predictions(
        predictions, ["seed", "arm", "timing", "subgroup", "model"]
    )
    expected = {"HR+/HER2-": 320, "HR-/HER2-": 287, "HER2+": 201}
    for row in metrics.itertuples(index=False):
        if int(row.n) != expected[str(row.subgroup)]:
            raise ValueError(f"subgroup OOF coverage drifted for {row.subgroup}")
    return predictions, metrics, hyperparameters


def incremental_effects(pcr_metrics: pd.DataFrame) -> pd.DataFrame:
    specifications = {
        "C+M_vs_C": ("C", "C+M"),
        "C+F+M_vs_C+F": ("C+F", "C+F+M"),
    }
    rows: list[dict[str, Any]] = []
    run_columns = ["population", "seed", "arm", "timing", "clinical_contract"]
    for key, group in pcr_metrics.groupby(run_columns, sort=True):
        by_model = group.set_index("model", verify_integrity=True)
        for comparison_name, (
            reference_name,
            comparison_name_model,
        ) in specifications.items():
            if (
                reference_name not in by_model.index
                or comparison_name_model not in by_model.index
            ):
                continue
            reference = by_model.loc[reference_name]
            comparison = by_model.loc[comparison_name_model]
            if int(reference["n"]) != int(comparison["n"]):
                raise ValueError("incremental metric model sizes differ")
            rows.append(
                {
                    **dict(zip(run_columns, key, strict=True)),
                    "comparison": comparison_name,
                    "reference_model": reference_name,
                    "comparison_model": comparison_name_model,
                    "n": int(reference["n"]),
                    "delta_auroc": float(comparison["auroc"] - reference["auroc"]),
                    "delta_auprc": float(comparison["auprc"] - reference["auprc"]),
                    "brier_improvement": float(
                        reference["brier"] - comparison["brier"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def run_bootstrap(
    config: Mapping[str, Any], pcr_predictions: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bootstrap = config["bootstrap"]
    specifications = (
        ("full_808", "C+M_vs_C", "C", "C+M"),
        ("ftv_complete_375", "C+M_vs_C", "C", "C+M"),
        ("ftv_complete_375", "C+F+M_vs_C+F", "C+F", "C+F+M"),
    )
    summary_frames: list[pd.DataFrame] = []
    draw_frames: list[pd.DataFrame] = []
    base_seed = int(bootstrap["random_seed"])
    counter = 0
    for (
        population,
        comparison_name,
        reference_name,
        comparison_name_model,
    ) in specifications:
        selected = pcr_predictions.loc[pcr_predictions["population"].eq(population)]
        for (seed, arm, timing), run in selected.groupby(
            ["seed", "arm", "timing"], sort=True
        ):
            reference = run.loc[run["model"].eq(reference_name)]
            comparison = run.loc[run["model"].eq(comparison_name_model)]
            if reference.empty or comparison.empty:
                raise ValueError(
                    f"bootstrap pair missing for {population}/{comparison_name}"
                )
            result = paired_fold_stratified_bootstrap(
                reference,
                comparison,
                n_bootstrap=int(bootstrap["replicates"]),
                confidence_level=float(bootstrap["confidence_level"]),
                seed=base_seed + counter,
            )
            counter += 1
            metadata = {
                "population": population,
                "seed": int(seed),
                "arm": str(arm),
                "timing": str(timing),
                "comparison": comparison_name,
                "reference_model": reference_name,
                "comparison_model": comparison_name_model,
            }
            summary_frames.append(
                result.summary.rename(
                    columns={
                        "reference": "reference_value",
                        "comparison": "comparison_value",
                    }
                ).assign(**metadata)
            )
            draw_frames.append(result.draws.assign(**metadata))
    summary = pd.concat(summary_frames, ignore_index=True)
    draws = pd.concat(draw_frames, ignore_index=True)
    leading = [
        "population",
        "seed",
        "arm",
        "timing",
        "comparison",
        "reference_model",
        "comparison_model",
    ]
    return summary.loc[:, [*leading, *[c for c in summary if c not in leading]]], draws


def cohort_summary(clinical: pd.DataFrame, ftv_ids: set[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for population, subset in (
        ("full_808", clinical),
        ("ftv_complete_375", clinical.loc[clinical["patient_id"].isin(ftv_ids)]),
        ("ftv_unavailable_433", clinical.loc[~clinical["patient_id"].isin(ftv_ids)]),
    ):
        rows.append(
            {
                "population": population,
                "n": int(len(subset)),
                "pcr_positive": int(subset["label_pcr"].sum()),
                "pcr_prevalence": float(subset["label_pcr"].mean()),
                "hr_positive": int(subset["label_hr"].sum()),
                "her2_positive": int(subset["label_her2"].sum()),
                "mp1": int(subset["label_mp"].sum()),
                "age_missing": int(subset["age_at_screening"].isna().sum()),
                "race_missing": int(subset["race_simple"].isna().sum()),
                "menopause_missing": int(
                    subset["menopausal_status_simple"].isna().sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def input_manifest(
    config: Mapping[str, Any],
    assets: Mapping[tuple[int, str, int], LocalFeatureAsset],
) -> pd.DataFrame:
    paths = config["paths"]
    rows: list[dict[str, Any]] = []
    for kind, path_key, hash_key in (
        ("clinical_labels", "clinical_labels", "clinical_labels_sha256"),
        ("fold_manifest", "fold_manifest", "fold_manifest_sha256"),
        ("ftv_transition_table", "ftv_table", "ftv_table_sha256"),
        (
            "local_preregistration_lock",
            "local_preregistration_lock",
            "local_preregistration_lock_sha256",
        ),
    ):
        path = Path(paths[path_key])
        rows.append(
            {
                "kind": kind,
                "seed": math.nan,
                "arm": "",
                "fold": math.nan,
                "path": str(path),
                "sha256": str(paths[hash_key]),
                "size_bytes": int(path.stat().st_size),
                "checkpoint_sha256": "",
                "selection_sha256": "",
            }
        )
    for (seed, arm, fold), asset in sorted(assets.items()):
        rows.append(
            {
                "kind": "local_response_state",
                "seed": int(seed),
                "arm": str(arm),
                "fold": int(fold),
                "path": str(asset.path),
                "sha256": str(asset.metadata["feature_sha256"]),
                "size_bytes": int(asset.path.stat().st_size),
                "checkpoint_sha256": str(asset.metadata["checkpoint_sha256"]),
                "selection_sha256": str(asset.metadata["selection_sha256"]),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=EXPERIMENT_ROOT / "configs" / "audit.json"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=None,
        help="Testing override; formal run uses the locked config value 2000.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.umask(0o077)
    started = time.time()
    config = load_config(args.config, repo_root=REPO_ROOT, verify_paths=True)
    if args.bootstrap_replicates is not None:
        if args.bootstrap_replicates <= 0:
            raise ValueError("--bootstrap-replicates must be positive")
        config = dict(config)
        config["bootstrap"] = {
            **dict(config["bootstrap"]),
            "replicates": int(args.bootstrap_replicates),
        }

    output_paths = {
        "profile_predictions": EXPERIMENT_ROOT
        / "predictions"
        / "profile_oof.private.csv",
        "clinical_predictions": EXPERIMENT_ROOT
        / "predictions"
        / "clinical_baselines_oof.private.csv",
        "pcr_predictions": EXPERIMENT_ROOT / "predictions" / "pcr_oof.private.csv",
        "subgroup_predictions": EXPERIMENT_ROOT
        / "predictions"
        / "subgroup_oof.private.csv",
        "residual_predictions": EXPERIMENT_ROOT
        / "predictions"
        / "clinical_residual_oof.private.csv",
        "bootstrap_draws": EXPERIMENT_ROOT
        / "predictions"
        / "bootstrap_draws.private.csv",
        "profile_metrics": EXPERIMENT_ROOT / "metrics" / "profile_oof_metrics.csv",
        "clinical_metrics": EXPERIMENT_ROOT
        / "metrics"
        / "clinical_baseline_metrics.csv",
        "pcr_metrics": EXPERIMENT_ROOT / "metrics" / "pcr_oof_metrics.csv",
        "pcr_fold_metrics": EXPERIMENT_ROOT / "metrics" / "pcr_fold_metrics.csv",
        "incremental": EXPERIMENT_ROOT / "metrics" / "incremental_effects.csv",
        "bootstrap": EXPERIMENT_ROOT / "metrics" / "bootstrap_ci.csv",
        "subgroup_metrics": EXPERIMENT_ROOT / "metrics" / "subgroup_metrics.csv",
        "clinical_residual_metrics": EXPERIMENT_ROOT
        / "metrics"
        / "clinical_residual_metrics.csv",
        "cohort_summary": EXPERIMENT_ROOT / "metrics" / "cohort_summary.csv",
        "hyperparameters": EXPERIMENT_ROOT
        / "metrics"
        / "hyperparameter_selections.csv",
        "input_manifest": EXPERIMENT_ROOT / "metrics" / "input_manifest.csv",
        "run_summary": EXPERIMENT_ROOT / "metrics" / "run_summary.json",
    }
    _require_output_policy(tuple(output_paths.values()), args.overwrite)

    paths = config["paths"]
    folds = load_fold_manifest(paths["fold_manifest"], paths["fold_manifest_sha256"])
    clinical = load_clinical_table(
        paths["clinical_labels"], paths["clinical_labels_sha256"], folds
    )
    ftv = load_ftv_wide(
        paths["ftv_table"],
        paths["ftv_table_sha256"],
        folds,
    )
    assets = load_all_local_feature_assets(config, folds)
    ftv_ids = set(ftv["patient_id"].astype(str))
    print(
        "validated inputs: 808 clinical, 375 FTV-complete, 20 LOCAL cells", flush=True
    )

    profile_predictions, profile_metrics, profile_hyperparameters = run_profile_probes(
        config, clinical, assets
    )
    print("completed HR/HER2/subtype representation probes", flush=True)
    clinical_predictions, clinical_metrics, clinical_hyperparameters = (
        run_clinical_baselines(config, clinical, ftv_ids, assets)
    )
    print("completed nested clinical baselines with/without treatment", flush=True)
    (
        pcr_predictions,
        pcr_metrics,
        pcr_fold_metrics,
        residual_predictions,
        residual_metrics,
        pcr_hyperparameters,
    ) = run_pcr_models(config, clinical, ftv, assets)
    print("completed matched pCR, FTV, and residual model families", flush=True)
    subgroup_predictions, subgroup_metrics, subgroup_hyperparameters = (
        run_subgroup_models(config, clinical, assets)
    )
    print("completed three subtype-conditioned analyses", flush=True)
    incremental = incremental_effects(pcr_metrics)
    bootstrap_summary, bootstrap_draws = run_bootstrap(config, pcr_predictions)
    print(
        f"completed paired patient bootstrap ({config['bootstrap']['replicates']} replicates)",
        flush=True,
    )

    hyperparameters = pd.DataFrame(
        [
            *profile_hyperparameters,
            *clinical_hyperparameters,
            *pcr_hyperparameters,
            *subgroup_hyperparameters,
        ]
    )
    outputs: Mapping[str, pd.DataFrame] = {
        "profile_predictions": profile_predictions,
        "clinical_predictions": clinical_predictions,
        "pcr_predictions": pcr_predictions,
        "subgroup_predictions": subgroup_predictions,
        "residual_predictions": residual_predictions,
        "bootstrap_draws": bootstrap_draws,
        "profile_metrics": profile_metrics,
        "clinical_metrics": clinical_metrics,
        "pcr_metrics": pcr_metrics,
        "pcr_fold_metrics": pcr_fold_metrics,
        "incremental": incremental,
        "bootstrap": bootstrap_summary,
        "subgroup_metrics": subgroup_metrics,
        "clinical_residual_metrics": residual_metrics,
        "cohort_summary": cohort_summary(clinical, ftv_ids),
        "hyperparameters": hyperparameters,
        "input_manifest": input_manifest(config, assets),
    }
    for name, frame in outputs.items():
        _atomic_csv(frame, output_paths[name])

    summary = {
        "schema_version": 1,
        "experiment": "mri_clinical_complementarity_audit",
        "evidence_status": config["evidence_status"],
        "parent_commit": config["parent_commit"],
        "branch": "feature/mri-clinical-complementarity-audit",
        "formal_bootstrap_replicates": int(config["bootstrap"]["replicates"]),
        "n_local_cells": len(assets),
        "n_profile_patients": 808,
        "n_primary_ftv_patients": 375,
        "elapsed_seconds": float(time.time() - started),
        "artifacts": {
            name: {
                "path": str(path.relative_to(EXPERIMENT_ROOT)),
                "sha256": file_sha256(path),
                "size_bytes": int(path.stat().st_size),
            }
            for name, path in output_paths.items()
            if name != "run_summary" and path.exists()
        },
    }
    _atomic_json(summary, output_paths["run_summary"])
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

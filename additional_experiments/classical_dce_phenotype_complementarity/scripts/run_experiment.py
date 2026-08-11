#!/usr/bin/env python3
"""Run the preregistered classical DCE phenotype complementarity baseline.

All data-dependent transforms and model choices are fitted on outer-train and
selected on outer-validation.  Patient-level splits and predictions are
written only to gitignored ``*.private.csv`` files; committed outputs are
aggregate metrics and manifests.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    r2_score,
    roc_auc_score,
)

from data_contracts import (
    EXPERIMENT_ROOT,
    FAMILIES,
    NONFTV_FAMILIES,
    RAW_COLUMNS,
    SUBTYPE_ORDER,
    VISITS,
    FeatureFrame,
    SplitSpec,
    aggregate_split_manifest,
    build_feature_frame,
    clinical_frame,
    load_config,
    load_primary_cohort,
    make_clinical_preprocessor,
    make_mri_matched_splits,
    make_primary_splits,
    private_split_frame,
    sha256_file,
)
from modeling import (
    FTVResidualizer,
    NumericRadiomicsTransformer,
    fit_ridge_redundancy,
    paired_fold_stratified_bootstrap,
    predict_binary,
    tune_binary_classifier,
    tune_multiclass_classifier,
)


PRIMARY_MODELS: dict[str, tuple[bool, tuple[str, ...]]] = {
    "C": (True, ()),
    "F": (False, ("FTV",)),
    "N": (False, NONFTV_FAMILIES),
    "FULL": (False, FAMILIES),
    "C+F": (True, ("FTV",)),
    "C+N": (True, NONFTV_FAMILIES),
    "C+FULL": (True, FAMILIES),
}

ABLATION_MODELS: dict[str, tuple[bool, tuple[str, ...]]] = {
    "C+F+D": (True, ("FTV", "LD")),
    "C+F+S": (True, ("FTV", "SPH")),
    "C+F+B": (True, ("FTV", "BPE")),
}

TIMING_LABELS = {
    "T0": "T0",
    "T1": "T1",
    "T2": "T2",
    "T3": "T3 (late/pre-surgery)",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True, default=_json_default)
        handle.write("\n")
    temporary.replace(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _scenario_strategy(scenario: str) -> str:
    if scenario == "complete_case":
        return "strict"
    if scenario == "train_median_indicator":
        return "median_indicator"
    raise ValueError(f"unknown missingness scenario {scenario}")


def _log_columns(feature_frame: FeatureFrame) -> list[str]:
    return feature_frame.metadata.loc[
        feature_frame.metadata["transform"] == "log1p", "feature"
    ].tolist()


def _radiomics_transformer(
    config: Mapping[str, Any], feature_frame: FeatureFrame, scenario: str
) -> NumericRadiomicsTransformer:
    quantiles = tuple(float(value) for value in config["preprocessing"]["winsor_quantiles"])
    return NumericRadiomicsTransformer(
        winsor_quantiles=quantiles,
        log_columns=_log_columns(feature_frame),
        log_transform="log1p",
        missing_strategy=_scenario_strategy(scenario),
        with_scaling=True,
    )


def _eligible_mask(cohort: pd.DataFrame, timing: str, view: str, scenario: str) -> np.ndarray:
    if scenario == "train_median_indicator":
        return np.ones(len(cohort), dtype=bool)
    full = build_feature_frame(cohort, timing, view, FAMILIES).values
    return np.isfinite(full.to_numpy(dtype=float)).all(axis=1)


def _filtered(indices: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    return np.asarray(indices, dtype=int)[eligible[np.asarray(indices, dtype=int)]]


def _fit_components(
    *,
    cohort: pd.DataFrame,
    config: Mapping[str, Any],
    timing: str,
    view: str,
    clinical_needed: bool,
    families: Sequence[str],
    scenario: str,
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, NumericRadiomicsTransformer | None, FeatureFrame | None]:
    train_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    if clinical_needed:
        raw_clinical = clinical_frame(cohort, config)
        processor = make_clinical_preprocessor(config)
        train_parts.append(np.asarray(processor.fit_transform(raw_clinical.iloc[train]), dtype=float))
        validation_parts.append(np.asarray(processor.transform(raw_clinical.iloc[validation]), dtype=float))
        test_parts.append(np.asarray(processor.transform(raw_clinical.iloc[test]), dtype=float))

    fitted_radiomics: NumericRadiomicsTransformer | None = None
    feature_frame: FeatureFrame | None = None
    if families:
        feature_frame = build_feature_frame(cohort, timing, view, families)
        fitted_radiomics = _radiomics_transformer(config, feature_frame, scenario)
        train_parts.append(fitted_radiomics.fit_transform(feature_frame.values.iloc[train]))
        validation_parts.append(fitted_radiomics.transform(feature_frame.values.iloc[validation]))
        test_parts.append(fitted_radiomics.transform(feature_frame.values.iloc[test]))

    if not train_parts:
        raise ValueError("model has neither clinical nor radiomics inputs")
    return (
        np.concatenate(train_parts, axis=1),
        np.concatenate(validation_parts, axis=1),
        np.concatenate(test_parts, axis=1),
        fitted_radiomics,
        feature_frame,
    )


def _transform_audit_rows(
    transformer: NumericRadiomicsTransformer,
    feature_frame: FeatureFrame,
    *,
    protocol: str,
    scenario: str,
    view: str,
    timing: str,
    feature_set: str,
    fold: int,
    n_train: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    names = transformer.feature_names_in_.tolist()
    metadata = feature_frame.metadata.set_index("feature")
    for index, name in enumerate(names):
        row = metadata.loc[name]
        rows.append(
            {
                "fit_scope": "outer_train_only",
                "protocol": protocol,
                "scenario": scenario,
                "view": view,
                "timing": timing,
                "feature_set": feature_set,
                "fold": fold,
                "n_train": n_train,
                "feature": name,
                "family": row["family"],
                "role": row["role"],
                "transform": row["transform"],
                "winsor_lower": transformer.clip_lower_[index],
                "winsor_upper": transformer.clip_upper_[index],
                "train_median_after_transform": transformer.medians_[index],
                "train_mean_after_imputation": transformer.mean_[index],
                "train_scale": transformer.scale_[index],
                "all_missing_in_train": bool(transformer.all_missing_mask_[index]),
            }
        )
    return rows


def _tune_binary(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    model_type: str,
    config: Mapping[str, Any],
    random_state: int,
):
    grids = config["models"]
    return tune_binary_classifier(
        train_x,
        train_y,
        validation_x,
        validation_y,
        model_type=model_type,
        logistic_c_grid=grids["logistic_C"],
        svm_c_grid=grids["svm_C"],
        svm_gamma_grid=grids["svm_gamma"],
        class_weight=None,
        random_state=random_state,
    )


def run_pcr_models(
    *,
    cohort: pd.DataFrame,
    splits: Sequence[SplitSpec],
    config: Mapping[str, Any],
    population: str,
    models: Mapping[str, tuple[bool, tuple[str, ...]]],
    scenarios: Sequence[str],
    model_types: Sequence[str],
    views: Sequence[str],
    timings: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    transform_rows: list[dict[str, Any]] = []
    audited: set[tuple[Any, ...]] = set()

    for scenario in scenarios:
        for view in views:
            for timing in timings:
                eligible = _eligible_mask(cohort, timing, view, scenario)
                for model_name, (clinical_needed, families) in models.items():
                    for model_type in model_types:
                        for split in splits:
                            train = _filtered(split.train, eligible)
                            validation = _filtered(split.validation, eligible)
                            test = _filtered(split.test, eligible)
                            if min(len(train), len(validation), len(test)) == 0:
                                raise ValueError(f"empty split after missingness filtering: {model_name} {timing}")
                            train_x, validation_x, test_x, transformer, feature_frame = _fit_components(
                                cohort=cohort,
                                config=config,
                                timing=timing,
                                view=view,
                                clinical_needed=clinical_needed,
                                families=families,
                                scenario=scenario,
                                train=train,
                                validation=validation,
                                test=test,
                            )
                            random_state = int(config["seed"]) + 1000 * split.fold + (1 if model_type == "rbf_svm" else 0)
                            fit = _tune_binary(
                                train_x,
                                cohort.iloc[train]["pCR"].to_numpy(dtype=int),
                                validation_x,
                                cohort.iloc[validation]["pCR"].to_numpy(dtype=int),
                                model_type=model_type,
                                config=config,
                                random_state=random_state,
                            )
                            probability, predicted = predict_binary(fit, test_x)
                            for local, index in enumerate(test):
                                prediction_rows.append(
                                    {
                                        "trial_id": cohort.iloc[index]["trial_id"],
                                        "protocol": split.protocol,
                                        "population": population,
                                        "scenario": scenario,
                                        "view": view,
                                        "timing": timing,
                                        "timing_label": TIMING_LABELS[timing],
                                        "model_type": fit.model_type,
                                        "model": model_name,
                                        "fold": split.fold,
                                        "y_true": int(cohort.iloc[index]["pCR"]),
                                        "predicted_probability": float(probability[local]),
                                        "predicted_label": int(predicted[local]),
                                        "threshold": float(fit.threshold),
                                        "n_features": int(train_x.shape[1]),
                                    }
                                )
                            selection_rows.append(
                                {
                                    "task": "pCR",
                                    "protocol": split.protocol,
                                    "population": population,
                                    "scenario": scenario,
                                    "view": view,
                                    "timing": timing,
                                    "model_type": fit.model_type,
                                    "model": model_name,
                                    "fold": split.fold,
                                    "n_train": len(train),
                                    "n_validation": len(validation),
                                    "n_test": len(test),
                                    "n_features": train_x.shape[1],
                                    "validation_auroc": fit.validation_auroc,
                                    "validation_balanced_accuracy": fit.validation_balanced_accuracy,
                                    "threshold": fit.threshold,
                                    "best_params": json.dumps(fit.best_params, sort_keys=True),
                                }
                            )
                            audit_key = (split.protocol, scenario, view, timing, tuple(families), split.fold)
                            if transformer is not None and feature_frame is not None and audit_key not in audited:
                                transform_rows.extend(
                                    _transform_audit_rows(
                                        transformer,
                                        feature_frame,
                                        protocol=split.protocol,
                                        scenario=scenario,
                                        view=view,
                                        timing=timing,
                                        feature_set="+".join(families),
                                        fold=split.fold,
                                        n_train=len(train),
                                    )
                                )
                                audited.add(audit_key)
    return pd.DataFrame(prediction_rows), pd.DataFrame(selection_rows), pd.DataFrame(transform_rows)


def aggregate_binary_predictions(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "protocol", "population", "scenario", "view", "timing", "timing_label",
        "model_type", "model", "fold", "y_true", "predicted_probability", "predicted_label",
    }
    if not required.issubset(predictions.columns):
        raise ValueError(f"prediction table lacks columns: {sorted(required - set(predictions.columns))}")
    group_columns = [
        "protocol", "population", "scenario", "view", "timing", "timing_label", "model_type", "model"
    ]

    def summarize(group: pd.DataFrame) -> dict[str, Any]:
        y = group["y_true"].to_numpy(dtype=int)
        probability = group["predicted_probability"].to_numpy(dtype=float)
        predicted = group["predicted_label"].to_numpy(dtype=int)
        if len(np.unique(y)) != 2:
            raise ValueError("aggregate binary metric group does not contain both classes")
        return {
            "n": len(group),
            "n_positive": int(y.sum()),
            "n_negative": int((y == 0).sum()),
            "auroc": roc_auc_score(y, probability),
            "auprc": average_precision_score(y, probability),
            "balanced_accuracy": balanced_accuracy_score(y, predicted),
            "brier": brier_score_loss(y, probability),
        }

    oof_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(group_columns, sort=True, dropna=False):
        oof_rows.append({**dict(zip(group_columns, keys)), **summarize(group)})
        for fold, fold_group in group.groupby("fold", sort=True):
            fold_rows.append({**dict(zip(group_columns, keys)), "fold": int(fold), **summarize(fold_group)})
    return pd.DataFrame(oof_rows), pd.DataFrame(fold_rows)


def run_profile_probes(
    *,
    cohort: pd.DataFrame,
    splits: Sequence[SplitSpec],
    config: Mapping[str, Any],
    population: str,
    views: Sequence[str],
    timings: Sequence[str],
    model_types: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    feature_sets = {"N": NONFTV_FAMILIES, "FULL": FAMILIES}
    for view in views:
        for timing in timings:
            eligible = _eligible_mask(cohort, timing, view, "complete_case")
            for feature_set, families in feature_sets.items():
                frame = build_feature_frame(cohort, timing, view, families)
                for model_type in model_types:
                    for target in ("HR", "HER2", "subtype"):
                        for split in splits:
                            train = _filtered(split.train, eligible)
                            validation = _filtered(split.validation, eligible)
                            test = _filtered(split.test, eligible)
                            transformer = _radiomics_transformer(config, frame, "complete_case")
                            train_x = transformer.fit_transform(frame.values.iloc[train])
                            validation_x = transformer.transform(frame.values.iloc[validation])
                            test_x = transformer.transform(frame.values.iloc[test])
                            random_state = int(config["seed"]) + 1000 * split.fold + (1 if model_type == "rbf_svm" else 0)
                            if target in {"HR", "HER2"}:
                                fit = _tune_binary(
                                    train_x,
                                    cohort.iloc[train][target].to_numpy(dtype=int),
                                    validation_x,
                                    cohort.iloc[validation][target].to_numpy(dtype=int),
                                    model_type=model_type,
                                    config=config,
                                    random_state=random_state,
                                )
                                probability, predicted = predict_binary(fit, test_x)
                                best_params = fit.best_params
                                validation_auroc = fit.validation_auroc
                                threshold = fit.threshold
                                for local, index in enumerate(test):
                                    rows.append(
                                        {
                                            "trial_id": cohort.iloc[index]["trial_id"],
                                            "protocol": split.protocol,
                                            "population": population,
                                            "view": view,
                                            "timing": timing,
                                            "timing_label": TIMING_LABELS[timing],
                                            "feature_set": feature_set,
                                            "model_type": fit.model_type,
                                            "target": target,
                                            "fold": split.fold,
                                            "y_true": int(cohort.iloc[index][target]),
                                            "predicted_probability": float(probability[local]),
                                            "predicted_label": int(predicted[local]),
                                            "threshold": float(threshold),
                                            **{f"probability__{label}": np.nan for label in SUBTYPE_ORDER},
                                        }
                                    )
                            else:
                                grids = config["models"]
                                fit = tune_multiclass_classifier(
                                    train_x,
                                    cohort.iloc[train]["subtype"].to_numpy(),
                                    validation_x,
                                    cohort.iloc[validation]["subtype"].to_numpy(),
                                    model_type=model_type,
                                    logistic_c_grid=grids["logistic_C"],
                                    svm_c_grid=grids["svm_C"],
                                    svm_gamma_grid=grids["svm_gamma"],
                                    class_weight=None,
                                    random_state=random_state,
                                    classes=SUBTYPE_ORDER,
                                )
                                raw_probability = fit.predict_proba(test_x)
                                probability = np.column_stack(
                                    [raw_probability[:, np.flatnonzero(fit.classes_ == label)[0]] for label in SUBTYPE_ORDER]
                                )
                                predicted = np.asarray(SUBTYPE_ORDER, dtype=object)[np.argmax(probability, axis=1)]
                                best_params = fit.best_params
                                validation_auroc = fit.validation_auroc
                                threshold = np.nan
                                for local, index in enumerate(test):
                                    probability_columns = {
                                        f"probability__{label}": float(probability[local, column])
                                        for column, label in enumerate(SUBTYPE_ORDER)
                                    }
                                    rows.append(
                                        {
                                            "trial_id": cohort.iloc[index]["trial_id"],
                                            "protocol": split.protocol,
                                            "population": population,
                                            "view": view,
                                            "timing": timing,
                                            "timing_label": TIMING_LABELS[timing],
                                            "feature_set": feature_set,
                                            "model_type": fit.model_type,
                                            "target": target,
                                            "fold": split.fold,
                                            "y_true": cohort.iloc[index]["subtype"],
                                            "predicted_probability": np.nan,
                                            "predicted_label": predicted[local],
                                            "threshold": np.nan,
                                            **probability_columns,
                                        }
                                    )
                            selections.append(
                                {
                                    "task": target,
                                    "protocol": split.protocol,
                                    "population": population,
                                    "scenario": "complete_case",
                                    "view": view,
                                    "timing": timing,
                                    "model_type": fit.model_type,
                                    "model": feature_set,
                                    "fold": split.fold,
                                    "n_train": len(train),
                                    "n_validation": len(validation),
                                    "n_test": len(test),
                                    "n_features": train_x.shape[1],
                                    "validation_auroc": validation_auroc,
                                    "validation_balanced_accuracy": np.nan,
                                    "threshold": threshold,
                                    "best_params": json.dumps(best_params, sort_keys=True),
                                }
                            )
    return pd.DataFrame(rows), pd.DataFrame(selections)


def aggregate_profile_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "protocol", "population", "view", "timing", "timing_label", "feature_set", "model_type", "target"
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(group_columns, sort=True):
        target = group["target"].iloc[0]
        y = group["y_true"].to_numpy()
        if target in {"HR", "HER2"}:
            probability = group["predicted_probability"].to_numpy(dtype=float)
            predicted = group["predicted_label"].to_numpy(dtype=int)
            metrics = {
                "n": len(group),
                "n_positive": int(y.astype(int).sum()),
                "auroc": roc_auc_score(y.astype(int), probability),
                "auprc": average_precision_score(y.astype(int), probability),
                "balanced_accuracy": balanced_accuracy_score(y.astype(int), predicted),
                "brier": brier_score_loss(y.astype(int), probability),
            }
        else:
            probability = group[[f"probability__{label}" for label in SUBTYPE_ORDER]].to_numpy(dtype=float)
            one_hot = np.column_stack([(y == label).astype(int) for label in SUBTYPE_ORDER])
            predicted = group["predicted_label"].to_numpy()
            metrics = {
                "n": len(group),
                "n_positive": np.nan,
                "auroc": float(np.mean([roc_auc_score(one_hot[:, i], probability[:, i]) for i in range(4)])),
                "auprc": float(np.mean([average_precision_score(one_hot[:, i], probability[:, i]) for i in range(4)])),
                "balanced_accuracy": balanced_accuracy_score(y, predicted),
                "brier": float(np.mean(np.sum((one_hot - probability) ** 2, axis=1))),
            }
        rows.append({**dict(zip(group_columns, keys)), **metrics})
    return pd.DataFrame(rows)


def run_redundancy_and_residualization(
    *,
    cohort: pd.DataFrame,
    splits: Sequence[SplitSpec],
    config: Mapping[str, Any],
    views: Sequence[str],
    timings: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    redundancy_predictions: list[dict[str, Any]] = []
    redundancy_folds: list[dict[str, Any]] = []
    residual_predictions: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    raw_clinical = clinical_frame(cohort, config)

    for view in views:
        for timing in timings:
            eligible = _eligible_mask(cohort, timing, view, "complete_case")
            f_frame = build_feature_frame(cohort, timing, view, ("FTV",))
            n_frame = build_feature_frame(cohort, timing, view, NONFTV_FAMILIES)
            current_f_frame = build_feature_frame(cohort, timing, "static", ("FTV",))
            for split in splits:
                train = _filtered(split.train, eligible)
                validation = _filtered(split.validation, eligible)
                test = _filtered(split.test, eligible)

                f_transform = _radiomics_transformer(config, f_frame, "complete_case")
                n_transform = _radiomics_transformer(config, n_frame, "complete_case")
                target_transform = _radiomics_transformer(config, current_f_frame, "complete_case")
                f_train = f_transform.fit_transform(f_frame.values.iloc[train])
                f_validation = f_transform.transform(f_frame.values.iloc[validation])
                f_test = f_transform.transform(f_frame.values.iloc[test])
                n_train = n_transform.fit_transform(n_frame.values.iloc[train])
                n_validation = n_transform.transform(n_frame.values.iloc[validation])
                n_test = n_transform.transform(n_frame.values.iloc[test])
                target_train = target_transform.fit_transform(current_f_frame.values.iloc[train])[:, 0]
                target_validation = target_transform.transform(current_f_frame.values.iloc[validation])[:, 0]
                target_test = target_transform.transform(current_f_frame.values.iloc[test])[:, 0]

                ridge = fit_ridge_redundancy(
                    n_train,
                    target_train,
                    n_validation,
                    target_validation,
                    alphas=config["models"]["ridge_alpha"],
                )
                predicted_ftv = np.asarray(ridge.predict(n_test), dtype=float)
                fold_r2 = r2_score(target_test, predicted_ftv)
                fold_spearman = spearmanr(target_test, predicted_ftv).statistic
                redundancy_folds.append(
                    {
                        "protocol": split.protocol,
                        "population": "clinical_radiomics_complete_384",
                        "view": view,
                        "timing": timing,
                        "timing_label": TIMING_LABELS[timing],
                        "fold": split.fold,
                        "n": len(test),
                        "alpha": ridge.alpha,
                        "r2": fold_r2,
                        "spearman": fold_spearman,
                    }
                )
                for local, index in enumerate(test):
                    redundancy_predictions.append(
                        {
                            "trial_id": cohort.iloc[index]["trial_id"],
                            "protocol": split.protocol,
                            "population": "clinical_radiomics_complete_384",
                            "view": view,
                            "timing": timing,
                            "timing_label": TIMING_LABELS[timing],
                            "fold": split.fold,
                            "ftv_true": target_test[local],
                            "ftv_predicted": predicted_ftv[local],
                        }
                    )

                # Fixed alpha=1 on standardized features is chosen before outcome
                # modeling; the map itself is fitted only on outer train.
                residualizer = FTVResidualizer(
                    alpha=float(config["models"]["residualizer_alpha"])
                ).fit(f_train, n_train)
                residual_train = residualizer.transform(f_train, n_train)
                residual_validation = residualizer.transform(f_validation, n_validation)
                residual_test = residualizer.transform(f_test, n_test)

                clinical_processor = make_clinical_preprocessor(config)
                clinical_train = np.asarray(clinical_processor.fit_transform(raw_clinical.iloc[train]), dtype=float)
                clinical_validation = np.asarray(clinical_processor.transform(raw_clinical.iloc[validation]), dtype=float)
                clinical_test = np.asarray(clinical_processor.transform(raw_clinical.iloc[test]), dtype=float)

                residual_specs = {
                    "N_res": (residual_train, residual_validation, residual_test),
                    "C+F+N_res": (
                        np.concatenate((clinical_train, f_train, residual_train), axis=1),
                        np.concatenate((clinical_validation, f_validation, residual_validation), axis=1),
                        np.concatenate((clinical_test, f_test, residual_test), axis=1),
                    ),
                }
                for model_name, (train_x, validation_x, test_x) in residual_specs.items():
                    fit = _tune_binary(
                        train_x,
                        cohort.iloc[train]["pCR"].to_numpy(dtype=int),
                        validation_x,
                        cohort.iloc[validation]["pCR"].to_numpy(dtype=int),
                        model_type="logistic",
                        config=config,
                        random_state=int(config["seed"]) + 1000 * split.fold,
                    )
                    probability, predicted = predict_binary(fit, test_x)
                    for local, index in enumerate(test):
                        residual_predictions.append(
                            {
                                "trial_id": cohort.iloc[index]["trial_id"],
                                "protocol": split.protocol,
                                "population": "clinical_radiomics_complete_384",
                                "scenario": "complete_case",
                                "view": view,
                                "timing": timing,
                                "timing_label": TIMING_LABELS[timing],
                                "model_type": "logistic",
                                "model": model_name,
                                "fold": split.fold,
                                "y_true": int(cohort.iloc[index]["pCR"]),
                                "predicted_probability": probability[local],
                                "predicted_label": predicted[local],
                                "threshold": fit.threshold,
                                "n_features": train_x.shape[1],
                            }
                        )
                    selections.append(
                        {
                            "task": "pCR_residualized",
                            "protocol": split.protocol,
                            "population": "clinical_radiomics_complete_384",
                            "scenario": "complete_case",
                            "view": view,
                            "timing": timing,
                            "model_type": "logistic",
                            "model": model_name,
                            "fold": split.fold,
                            "n_train": len(train),
                            "n_validation": len(validation),
                            "n_test": len(test),
                            "n_features": train_x.shape[1],
                            "validation_auroc": fit.validation_auroc,
                            "validation_balanced_accuracy": fit.validation_balanced_accuracy,
                            "threshold": fit.threshold,
                            "best_params": json.dumps(fit.best_params, sort_keys=True),
                        }
                    )

    redundancy_private = pd.DataFrame(redundancy_predictions)
    aggregate_rows: list[dict[str, Any]] = []
    group_columns = ["protocol", "population", "view", "timing", "timing_label"]
    for keys, group in redundancy_private.groupby(group_columns, sort=True):
        true = group["ftv_true"].to_numpy(dtype=float)
        predicted = group["ftv_predicted"].to_numpy(dtype=float)
        fold_group = pd.DataFrame(redundancy_folds)
        mask = np.ones(len(fold_group), dtype=bool)
        for column, value in zip(group_columns, keys):
            mask &= fold_group[column].eq(value).to_numpy()
        relevant_folds = fold_group.loc[mask]
        aggregate_rows.append(
            {
                **dict(zip(group_columns, keys)),
                "n": len(group),
                "r2": r2_score(true, predicted),
                "spearman": spearmanr(true, predicted).statistic,
                "fold_mean_r2": relevant_folds["r2"].mean(),
                "fold_min_r2": relevant_folds["r2"].min(),
                "fold_max_r2": relevant_folds["r2"].max(),
                "fold_mean_spearman": relevant_folds["spearman"].mean(),
                "fold_min_spearman": relevant_folds["spearman"].min(),
                "fold_max_spearman": relevant_folds["spearman"].max(),
            }
        )
    return (
        redundancy_private,
        pd.DataFrame(redundancy_folds),
        pd.DataFrame(aggregate_rows),
        pd.DataFrame(residual_predictions),
        pd.DataFrame(selections),
    )


def _patient_set_sha256(values: Iterable[str]) -> str:
    import hashlib

    payload = "\n".join(sorted(str(value) for value in values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compute_bootstrap_effects(
    pcr_predictions: pd.DataFrame,
    residual_predictions: pd.DataFrame,
    *,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary = pcr_predictions[
        (pcr_predictions["protocol"] == "primary_stratified_384")
        & (pcr_predictions["population"] == "clinical_radiomics_complete_384")
        & (pcr_predictions["scenario"] == "complete_case")
        & (pcr_predictions["model_type"] == "logistic")
    ].copy()
    comparisons = [
        ("C+FULL_vs_C+F", "C+F", "C+FULL", pcr_predictions),
        ("C+N_vs_C", "C", "C+N", pcr_predictions),
        ("C+F+N_res_vs_C+F", "C+F", "C+F+N_res", residual_predictions),
    ]
    summaries: list[dict[str, Any]] = []
    draws: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    seed = int(config["seed"])
    n_bootstrap = int(config["bootstrap_draws"])

    for comparison_index, (comparison, baseline_model, augmented_model, augmented_source) in enumerate(comparisons):
        for view in ("static", "longitudinal"):
            for timing_index, timing in enumerate(VISITS):
                baseline = primary[
                    (primary["view"] == view)
                    & (primary["timing"] == timing)
                    & (primary["model"] == baseline_model)
                ][["trial_id", "fold", "y_true", "predicted_probability"]].rename(
                    columns={"predicted_probability": "baseline_probability"}
                )
                if comparison != "C+F+N_res_vs_C+F":
                    augmented = primary[
                        (primary["view"] == view)
                        & (primary["timing"] == timing)
                        & (primary["model"] == augmented_model)
                    ][["trial_id", "fold", "y_true", "predicted_probability"]].rename(
                        columns={"predicted_probability": "augmented_probability"}
                    )
                else:
                    augmented = augmented_source[
                        (augmented_source["protocol"] == "primary_stratified_384")
                        & (augmented_source["population"] == "clinical_radiomics_complete_384")
                        & (augmented_source["scenario"] == "complete_case")
                        & (augmented_source["model_type"] == "logistic")
                        & (augmented_source["view"] == view)
                        & (augmented_source["timing"] == timing)
                        & (augmented_source["model"] == augmented_model)
                    ][["trial_id", "fold", "y_true", "predicted_probability"]].rename(
                        columns={"predicted_probability": "augmented_probability"}
                    )
                paired = baseline.merge(
                    augmented,
                    on="trial_id",
                    how="inner",
                    validate="one_to_one",
                    suffixes=("_base", "_aug"),
                )
                if len(paired) != 384:
                    raise ValueError(f"key comparison {comparison}/{view}/{timing} is not n=384")
                if not np.array_equal(paired["fold_base"], paired["fold_aug"]):
                    raise ValueError("paired comparison fold mismatch")
                if not np.array_equal(paired["y_true_base"], paired["y_true_aug"]):
                    raise ValueError("paired comparison label mismatch")
                bootstrap_seed = seed + 100000 * comparison_index + 1000 * (0 if view == "static" else 1) + timing_index
                result = paired_fold_stratified_bootstrap(
                    paired["y_true_base"].to_numpy(dtype=int),
                    paired["baseline_probability"].to_numpy(dtype=float),
                    paired["augmented_probability"].to_numpy(dtype=float),
                    paired["fold_base"].to_numpy(dtype=int),
                    patient_ids=paired["trial_id"].to_numpy(),
                    n_bootstrap=n_bootstrap,
                    confidence_level=0.95,
                    random_state=bootstrap_seed,
                    stratify_outcome=True,
                    return_distributions=True,
                )
                summary = {
                    "protocol": "primary_stratified_384",
                    "population": "clinical_radiomics_complete_384",
                    "scenario": "complete_case",
                    "view": view,
                    "timing": timing,
                    "timing_label": TIMING_LABELS[timing],
                    "model_type": "logistic",
                    "comparison": comparison,
                    "baseline_model": baseline_model,
                    "augmented_model": augmented_model,
                    "n": result["n_patients"],
                    "n_positive": int(paired["y_true_base"].sum()),
                    "n_bootstrap": result["n_bootstrap"],
                    "bootstrap_seed": bootstrap_seed,
                    "stratification": result["stratification"],
                    "delta_auroc": result["delta_auroc"]["estimate"],
                    "delta_auroc_ci_low": result["delta_auroc"]["ci_low"],
                    "delta_auroc_ci_high": result["delta_auroc"]["ci_high"],
                    "delta_auprc": result["delta_auprc"]["estimate"],
                    "delta_auprc_ci_low": result["delta_auprc"]["ci_low"],
                    "delta_auprc_ci_high": result["delta_auprc"]["ci_high"],
                    "brier_improvement": result["brier_improvement"]["estimate"],
                    "brier_improvement_ci_low": result["brier_improvement"]["ci_low"],
                    "brier_improvement_ci_high": result["brier_improvement"]["ci_high"],
                }
                summaries.append(summary)
                distributions = result["distributions"]
                for draw_index in range(n_bootstrap):
                    draws.append(
                        {
                            "comparison": comparison,
                            "view": view,
                            "timing": timing,
                            "draw": draw_index,
                            "delta_auroc": distributions["delta_auroc"][draw_index],
                            "delta_auprc": distributions["delta_auprc"][draw_index],
                            "brier_improvement": distributions["brier_improvement"][draw_index],
                        }
                    )
                manifests.append(
                    {
                        "protocol": "primary_stratified_384",
                        "population": "clinical_radiomics_complete_384",
                        "scenario": "complete_case",
                        "view": view,
                        "timing": timing,
                        "comparison": comparison,
                        "baseline_model": baseline_model,
                        "augmented_model": augmented_model,
                        "n": len(paired),
                        "pCR_positive": int(paired["y_true_base"].sum()),
                        "missingness_exclusions": 0,
                        "exclusion_reason": "none_within_selected_384_row_workbook",
                        "patient_set_sha256": _patient_set_sha256(paired["trial_id"]),
                    }
                )
    return pd.DataFrame(summaries), pd.DataFrame(draws), pd.DataFrame(manifests)


def build_population_manifest(
    pcr_predictions: pd.DataFrame,
    key_manifest: pd.DataFrame,
) -> pd.DataFrame:
    rows = key_manifest.to_dict("records")
    for protocol, population, excluded, reason in (
        (
            "primary_stratified_384",
            "clinical_radiomics_complete_384",
            0,
            "none_within_selected_384_row_workbook",
        ),
        (
            "locked_mri_manifest_375",
            "mri_matched_375",
            9,
            "nine_workbook_patients_lack_complete_four_visit_MRI_membership",
        ),
    ):
        source = pcr_predictions[
            (pcr_predictions["protocol"] == protocol)
            & (pcr_predictions["population"] == population)
            & (pcr_predictions["model_type"] == "logistic")
            & (pcr_predictions["model"] == "C")
        ]
        for (scenario, view, timing), group in source.groupby(["scenario", "view", "timing"], sort=True):
            rows.append(
                {
                    "protocol": protocol,
                    "population": population,
                    "scenario": scenario,
                    "view": view,
                    "timing": timing,
                    "comparison": "all_primary_model_families",
                    "baseline_model": "C/F/N/FULL/C+F/C+N/C+FULL",
                    "augmented_model": "same_exact_patient_set",
                    "n": len(group),
                    "pCR_positive": int(group["y_true"].sum()),
                    "missingness_exclusions": excluded,
                    "exclusion_reason": reason,
                    "patient_set_sha256": _patient_set_sha256(group["trial_id"]),
                }
            )
    manifest = pd.DataFrame(rows).drop_duplicates().sort_values(
        ["protocol", "scenario", "view", "timing", "comparison"]
    )
    return manifest.reset_index(drop=True)


def lr_vs_svm_table(metrics: pd.DataFrame) -> pd.DataFrame:
    primary = metrics[
        (metrics["protocol"] == "primary_stratified_384")
        & (metrics["population"] == "clinical_radiomics_complete_384")
        & (metrics["scenario"] == "complete_case")
        & metrics["model"].isin(PRIMARY_MODELS)
    ]
    index = ["protocol", "population", "scenario", "view", "timing", "timing_label", "model"]
    measures = ["auroc", "auprc", "balanced_accuracy", "brier"]
    pivot = primary.pivot(index=index, columns="model_type", values=measures)
    model_label = {"logistic": "logistic", "rbf_svm": "svm"}
    pivot.columns = [f"{model_label.get(model_type, model_type)}_{metric}" for metric, model_type in pivot.columns]
    pivot = pivot.reset_index()
    for metric in measures:
        logistic = f"logistic_{metric}"
        svm = f"svm_{metric}"
        if logistic in pivot and svm in pivot:
            delta_name = "delta_svm_minus_lr" if metric == "auroc" else f"delta_svm_minus_lr_{metric}"
            pivot[delta_name] = pivot[svm] - pivot[logistic]
    return pivot


def feature_correlation_matrix(cohort: pd.DataFrame) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    for family in FAMILIES:
        for visit in VISITS:
            columns[f"{family}_{visit}"] = pd.to_numeric(cohort[RAW_COLUMNS[family][visit]], errors="raise")
    frame = pd.DataFrame(columns)
    correlation = frame.corr(method="spearman")
    correlation.index.name = "feature"
    return correlation.reset_index()


def build_mri_traditional_comparisons(
    pcr_metrics: pd.DataFrame,
    profile_metrics: pd.DataFrame,
    mri_pcr: pd.DataFrame,
    mri_profile: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create descriptive same-patient traditional-vs-frozen-MRI summaries.

    The MRI AUROC is the unweighted mean of four seed×arm sensitivity cells;
    the traditional AUROC is one locked-fold LR OOF estimate.  They share the
    exact 375 patients and outer test folds but are not independent replicates.
    """

    traditional_pcr = pcr_metrics[
        (pcr_metrics["protocol"] == "locked_mri_manifest_375")
        & (pcr_metrics["population"] == "mri_matched_375")
        & (pcr_metrics["scenario"] == "complete_case")
        & (pcr_metrics["view"] == "longitudinal")
        & (pcr_metrics["model_type"] == "logistic")
    ]
    model_pairs = (("N", "M"), ("C+N", "C+M"), ("C+FULL", "C+F+M"))
    pcr_rows: list[dict[str, Any]] = []
    for timing in VISITS:
        for traditional_model, mri_model in model_pairs:
            traditional = traditional_pcr[
                (traditional_pcr["timing"] == timing) & (traditional_pcr["model"] == traditional_model)
            ]
            mri = mri_pcr[(mri_pcr["timing"] == timing) & (mri_pcr["model"] == mri_model)]
            if len(traditional) != 1 or len(mri) != 1:
                raise ValueError(f"missing MRI/traditional pCR comparison {timing}/{traditional_model}/{mri_model}")
            traditional_row = traditional.iloc[0]
            mri_row = mri.iloc[0]
            if int(traditional_row["n"]) != int(mri_row["n_patients_per_cell"]):
                raise ValueError("MRI/traditional pCR comparison population mismatch")
            pcr_rows.append(
                {
                    "population": "mri_matched_375",
                    "task": "pCR",
                    "view": "longitudinal",
                    "timing": timing,
                    "timing_label": TIMING_LABELS[timing],
                    "target": "pCR",
                    "traditional_model": traditional_model,
                    "mri_model": mri_model,
                    "n": int(traditional_row["n"]),
                    "traditional_auroc": float(traditional_row["auroc"]),
                    "mri_auroc": float(mri_row["auroc_mean"]),
                    "difference_mri_minus_traditional": float(mri_row["auroc_mean"] - traditional_row["auroc"]),
                    "mri_aggregation": "mean_of_four_seed_x_arm_cells_without_patient_pooling",
                }
            )

    traditional_profile = profile_metrics[
        (profile_metrics["protocol"] == "locked_mri_manifest_375")
        & (profile_metrics["population"] == "mri_matched_375")
        & (profile_metrics["model_type"] == "logistic")
    ]
    profile_rows: list[dict[str, Any]] = []
    for _, traditional_row in traditional_profile.iterrows():
        view = str(traditional_row["view"])
        timing = str(traditional_row["timing"])
        if view == "longitudinal" and timing == "T0":
            continue  # identical to static T0; avoid double counting.
        mri_view = timing if view == "static" else f"long_T0_{timing}"
        mri_target = "subtype_4class" if traditional_row["target"] == "subtype" else traditional_row["target"]
        mri = mri_profile[
            (mri_profile["view"] == mri_view) & (mri_profile["target"] == mri_target)
        ]
        if len(mri) != 1:
            raise ValueError(f"missing MRI profile comparison {mri_view}/{mri_target}")
        mri_row = mri.iloc[0]
        if int(traditional_row["n"]) != int(mri_row["n_patients_per_cell"]):
            raise ValueError("MRI/traditional profile comparison population mismatch")
        profile_rows.append(
            {
                "population": "mri_matched_375",
                "task": "profile_probe",
                "view": view,
                "timing": timing,
                "timing_label": TIMING_LABELS[timing],
                "target": traditional_row["target"],
                "traditional_model": traditional_row["feature_set"],
                "mri_model": "M",
                "n": int(traditional_row["n"]),
                "traditional_auroc": float(traditional_row["auroc"]),
                "mri_auroc": float(mri_row["auroc_mean"]),
                "difference_mri_minus_traditional": float(mri_row["auroc_mean"] - traditional_row["auroc"]),
                "mri_aggregation": "mean_of_four_seed_x_arm_cells_without_patient_pooling",
            }
        )
    return pd.DataFrame(pcr_rows), pd.DataFrame(profile_rows)


def _run_summary(
    *,
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
    pcr_metrics: pd.DataFrame,
    profile_metrics: pd.DataFrame,
    incremental: pd.DataFrame,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "experiment": config["experiment_name"],
        "status": "complete",
        "seed": int(config["seed"]),
        "bootstrap_draws": int(config["bootstrap_draws"]),
        "source": dict(provenance),
        "primary_population": {
            "name": "clinical_radiomics_complete_384",
            "n": 384,
            "pCR_positive": 113,
            "selection_boundary": "selected workbook datawith4visits; not population-wide completeness",
        },
        "mri_matched_population": {"name": "mri_matched_375", "n": 375, "pCR_positive": 110},
        "counts": {
            "pcr_metric_rows": len(pcr_metrics),
            "profile_metric_rows": len(profile_metrics),
            "incremental_effect_rows": len(incremental),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "elapsed_seconds": elapsed_seconds,
        "timing_contract": "information_timing_contract.csv",
        "t3_label": "late/pre-surgery",
    }


def run_experiment(config_path: Path, *, quick: bool = False) -> dict[str, Any]:
    started = time.time()
    config = load_config(config_path)
    if quick:
        config = json.loads(json.dumps(config))
        config["models"]["logistic_C"] = [0.1, 1.0]
        config["models"]["svm_C"] = [1.0]
        config["models"]["svm_gamma"] = [0.1]

    root = EXPERIMENT_ROOT
    metrics_dir = root / "metrics"
    predictions_dir = root / "predictions"
    features_dir = root / "features"
    configs_dir = root / "configs"
    for directory in (metrics_dir, predictions_dir, features_dir, configs_dir, root / "logs", root / "reports"):
        directory.mkdir(parents=True, exist_ok=True)

    cohort, source_provenance = load_primary_cohort(config)
    primary_splits = make_primary_splits(cohort, config)
    mri_matched, mri_splits = make_mri_matched_splits(cohort, config)
    write_csv(configs_dir / "primary_splits.private.csv", private_split_frame(cohort, primary_splits))
    write_csv(configs_dir / "mri_matched_splits.private.csv", private_split_frame(mri_matched, mri_splits))
    split_summary = pd.concat(
        [aggregate_split_manifest(cohort, primary_splits), aggregate_split_manifest(mri_matched, mri_splits)],
        ignore_index=True,
    )
    write_csv(metrics_dir / "split_summary.csv", split_summary)

    print("[1/7] primary C/F/N/FULL pCR grid", flush=True)
    model_types = ("logistic",) if quick else ("logistic", "rbf_svm")
    primary_predictions, primary_selections, primary_transforms = run_pcr_models(
        cohort=cohort,
        splits=primary_splits,
        config=config,
        population="clinical_radiomics_complete_384",
        models=PRIMARY_MODELS,
        scenarios=("complete_case", "train_median_indicator"),
        model_types=model_types,
        views=("static", "longitudinal"),
        timings=VISITS,
    )

    print("[2/7] prespecified family ablation", flush=True)
    ablation_predictions, ablation_selections, ablation_transforms = run_pcr_models(
        cohort=cohort,
        splits=primary_splits,
        config=config,
        population="clinical_radiomics_complete_384",
        models=ABLATION_MODELS,
        scenarios=("complete_case",),
        model_types=("logistic",),
        views=("static", "longitudinal"),
        timings=VISITS,
    )

    print("[3/7] locked MRI-matched 375 sensitivity", flush=True)
    matched_predictions, matched_selections, matched_transforms = run_pcr_models(
        cohort=mri_matched,
        splits=mri_splits,
        config=config,
        population="mri_matched_375",
        models=PRIMARY_MODELS,
        scenarios=("complete_case",),
        model_types=("logistic",),
        views=("longitudinal",),
        timings=VISITS,
    )

    all_predictions = pd.concat(
        [primary_predictions, ablation_predictions, matched_predictions], ignore_index=True
    )
    write_csv(predictions_dir / "pcr_oof.private.csv", all_predictions)
    pcr_metrics, pcr_fold_metrics = aggregate_binary_predictions(all_predictions)
    write_csv(metrics_dir / "pcr_oof_metrics.csv", pcr_metrics)
    write_csv(metrics_dir / "pcr_fold_metrics.csv", pcr_fold_metrics)
    main_names = set(PRIMARY_MODELS)
    write_csv(
        metrics_dir / "static_radiomics.csv",
        pcr_metrics[(pcr_metrics["view"] == "static") & pcr_metrics["model"].isin(main_names)].reset_index(drop=True),
    )
    write_csv(
        metrics_dir / "longitudinal_radiomics.csv",
        pcr_metrics[(pcr_metrics["view"] == "longitudinal") & pcr_metrics["model"].isin(main_names)].reset_index(drop=True),
    )
    ablation_reference = pcr_metrics[
        (pcr_metrics["protocol"] == "primary_stratified_384")
        & (pcr_metrics["scenario"] == "complete_case")
        & (pcr_metrics["model_type"] == "logistic")
        & pcr_metrics["model"].isin(["C+F", *ABLATION_MODELS.keys()])
    ].reset_index(drop=True)
    write_csv(metrics_dir / "family_ablation_metrics.csv", ablation_reference)

    print("[4/7] HR/HER2/subtype probes", flush=True)
    profile_predictions_primary, profile_selections_primary = run_profile_probes(
        cohort=cohort,
        splits=primary_splits,
        config=config,
        population="clinical_radiomics_complete_384",
        views=("static", "longitudinal"),
        timings=VISITS,
        model_types=model_types,
    )
    profile_predictions_matched, profile_selections_matched = run_profile_probes(
        cohort=mri_matched,
        splits=mri_splits,
        config=config,
        population="mri_matched_375",
        views=("static", "longitudinal"),
        timings=VISITS,
        model_types=("logistic",),
    )
    profile_predictions = pd.concat(
        [profile_predictions_primary, profile_predictions_matched], ignore_index=True
    )
    profile_selections = pd.concat(
        [profile_selections_primary, profile_selections_matched], ignore_index=True
    )
    write_csv(predictions_dir / "profile_oof.private.csv", profile_predictions)
    profile_metrics = aggregate_profile_predictions(profile_predictions)
    write_csv(metrics_dir / "profile_oof_metrics.csv", profile_metrics)

    print("[5/7] FTV redundancy and residualization", flush=True)
    (
        redundancy_private,
        redundancy_fold_metrics,
        redundancy_metrics,
        residual_predictions,
        residual_selections,
    ) = run_redundancy_and_residualization(
        cohort=cohort,
        splits=primary_splits,
        config=config,
        views=("static", "longitudinal"),
        timings=VISITS,
    )
    write_csv(predictions_dir / "ftv_redundancy_oof.private.csv", redundancy_private)
    write_csv(predictions_dir / "residualized_pcr_oof.private.csv", residual_predictions)
    write_csv(metrics_dir / "redundancy_fold_metrics.csv", redundancy_fold_metrics)
    write_csv(metrics_dir / "redundancy_metrics.csv", redundancy_metrics)
    residual_metrics, residual_fold_metrics = aggregate_binary_predictions(residual_predictions)
    write_csv(metrics_dir / "residualization_metrics.csv", residual_metrics)
    write_csv(metrics_dir / "residualization_fold_metrics.csv", residual_fold_metrics)

    print("[6/7] paired 2,000-draw patient bootstrap", flush=True)
    incremental, bootstrap_draws, key_manifest = compute_bootstrap_effects(
        all_predictions, residual_predictions, config=config
    )
    write_csv(metrics_dir / "incremental_effects.csv", incremental)
    write_csv(predictions_dir / "bootstrap_draws.private.csv", bootstrap_draws)
    population_manifest = build_population_manifest(all_predictions, key_manifest)
    write_csv(root / "matched_population_manifest.csv", population_manifest)
    write_csv(metrics_dir / "matched_population_manifest.csv", population_manifest)

    hyperparameters = pd.concat(
        [primary_selections, ablation_selections, matched_selections, profile_selections, residual_selections],
        ignore_index=True,
    )
    transforms = pd.concat(
        [primary_transforms, ablation_transforms, matched_transforms], ignore_index=True
    ).drop_duplicates()
    write_csv(metrics_dir / "hyperparameter_selections.csv", hyperparameters)
    write_csv(metrics_dir / "preprocessing_audit.csv", transforms)
    write_csv(metrics_dir / "lr_vs_svm.csv", lr_vs_svm_table(pcr_metrics))
    write_csv(metrics_dir / "feature_correlation_matrix.csv", feature_correlation_matrix(cohort))

    print("[7/7] frozen MRI aggregate reference and run summary", flush=True)
    from mri_reference import build_reference, write_reference_outputs

    reference = build_reference(config_path)
    write_reference_outputs(reference, metrics_dir, overwrite=True)
    mri_pcr_comparison, mri_profile_comparison = build_mri_traditional_comparisons(
        pcr_metrics,
        profile_metrics,
        reference.pcr_metrics,
        reference.profile_metrics,
    )
    write_csv(metrics_dir / "mri_reference_traditional_pcr_comparison.csv", mri_pcr_comparison)
    write_csv(metrics_dir / "mri_reference_traditional_profile_comparison.csv", mri_profile_comparison)
    summary = _run_summary(
        config=config,
        provenance=source_provenance,
        pcr_metrics=pcr_metrics,
        profile_metrics=profile_metrics,
        incremental=incremental,
        elapsed_seconds=time.time() - started,
    )
    summary["artifacts"] = {
        "pcr_oof_metrics_sha256": sha256_file(metrics_dir / "pcr_oof_metrics.csv"),
        "profile_oof_metrics_sha256": sha256_file(metrics_dir / "profile_oof_metrics.csv"),
        "incremental_effects_sha256": sha256_file(metrics_dir / "incremental_effects.csv"),
        "matched_population_manifest_sha256": sha256_file(root / "matched_population_manifest.csv"),
    }
    summary["quick_mode"] = bool(quick)
    write_json(metrics_dir / "run_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=EXPERIMENT_ROOT / "configs" / "experiment.json",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Developer smoke mode with reduced hyperparameter grids; never use for final conclusions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_experiment(args.config.resolve(), quick=args.quick)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), flush=True)


if __name__ == "__main__":
    main()

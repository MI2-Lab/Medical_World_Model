#!/usr/bin/env python3
"""Run the fold-safe compact MRI--clinical fusion audit end to end.

The runner consumes only Goal 2's frozen LOCAL response states and contracts.
It never loads MRI volumes, changes a checkpoint, or trains an encoder.  PCA,
FTV residualization, clinical preprocessing, base models, and stacking models
are fitted inside the applicable training boundary.  Patient-level artifacts
remain below gitignored ``features/`` and ``predictions/``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
SCRIPTS_ROOT = EXPERIMENT_ROOT / "scripts"
GOAL2_ROOT = REPO_ROOT / "additional_experiments" / "mri_clinical_complementarity_audit"
GOAL2_SCRIPTS = GOAL2_ROOT / "scripts"
for directory in (SCRIPTS_ROOT, GOAL2_SCRIPTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from compact_modeling import (  # noqa: E402
    fit_fixed_c_logistic,
    fit_train_pca,
    make_gaussian_random_projection,
    probability_to_logit,
    stratified_inner_assignments,
    strict_inner_oof_probabilities,
)
from contracts import (  # noqa: E402
    POPULATIONS,
    TIMINGS,
    atomic_write_csv,
    atomic_write_json,
    build_population_view,
    file_sha256,
    load_frozen_goal2_inputs,
    load_goal2_contract_module,
    require_known_output_policy,
)
from modeling import (  # noqa: E402
    binary_metrics,
    fit_binary_logistic,
    fit_ftv_mri_residualizer,
    fit_multiclass_logistic,
    multiclass_metrics,
)
from summaries import (  # noqa: E402
    ComparisonSpec,
    aggregate_pcr_predictions,
    aggregate_profile_predictions,
    summarize_paired_comparisons,
)


PCA_DIMENSIONS = (8, 16, 32, 64)
RP_DIMENSIONS = (16, 32)
SUBTYPE_CLASSES = tuple(
    sorted(("HR+/HER2-", "HR-/HER2-", "HR+/HER2+", "HR-/HER2+"))
)
SUBTYPE_PROBABILITY_COLUMNS: Mapping[str, str] = {
    "HR+/HER2-": "prob_hr_pos_her2_neg",
    "HR-/HER2-": "prob_hr_neg_her2_neg",
    "HR+/HER2+": "prob_hr_pos_her2_pos",
    "HR-/HER2+": "prob_hr_neg_her2_pos",
}
EXPECTED_COUNTS = {"full_808": 808, "ftv_complete_375": 375}
PRIMARY_CLINICAL_CONTRACT = "C2_full_with_treatment"


def _matrix_hash(values: np.ndarray, *, label: str) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(label.encode("utf-8"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _stable_seed(*values: Any, base: int = 0) -> int:
    payload = "|".join(str(value) for value in values).encode("utf-8")
    return int((int(base) + int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")) % (2**32 - 1))


def _model_key(model_family: str, representation: str, dimension: int) -> str:
    if representation == "none":
        return model_family
    if representation == "raw":
        return f"{model_family}|RAW"
    if representation == "pca":
        return f"{model_family}|PCA{dimension}"
    if representation == "pca_selected":
        return f"{model_family}|PCA_SELECTED"
    if representation == "random_projection":
        return f"{model_family}|R{dimension}"
    if representation == "late_fusion_candidate":
        return f"{model_family}|PCA{dimension}"
    if representation == "late_fusion":
        return f"{model_family}|PCA_SELECTED"
    raise ValueError(f"unknown representation {representation!r}")


def _clinical_matrix(
    goal2: Any,
    goal2_config: Mapping[str, Any],
    clinical: pd.DataFrame,
    indices: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, Any]:
    encoder = goal2.TrainOnlyClinicalEncoder.from_config(
        goal2_config, PRIMARY_CLINICAL_CONTRACT
    )
    encoder.fit(clinical.iloc[indices["train"]])
    return np.asarray(encoder.transform(clinical), dtype=np.float64), encoder


def _fit_binary(
    matrix: np.ndarray,
    labels: np.ndarray,
    indices: Mapping[str, np.ndarray],
    goal2_config: Mapping[str, Any],
    *,
    class_weight: str | Mapping[int, float] | None = None,
) -> Any:
    logistic = goal2_config["logistic"]
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


def _prediction_rows(
    fit: Any,
    matrix: np.ndarray,
    *,
    patient_ids: np.ndarray,
    labels: np.ndarray,
    indices: Mapping[str, np.ndarray],
    fold: int,
    population: str,
    seed: int,
    arm: str,
    timing: str,
    model_family: str,
    representation: str,
    dimension: int,
    raw_input_dim: int,
    selected_fold_dimension: int = -1,
    selected_by_validation: bool = False,
) -> list[dict[str, Any]]:
    test = indices["test"]
    probability = fit.predict_proba(matrix[test])
    threshold = float(fit.threshold_selection.threshold)
    rows: list[dict[str, Any]] = []
    key = _model_key(model_family, representation, dimension)
    for offset, row_index in enumerate(test):
        rows.append(
            {
                "patient_id": str(patient_ids[row_index]),
                "fold": int(fold),
                "population": population,
                "seed": int(seed),
                "arm": str(arm),
                "timing": timing,
                "model_family": model_family,
                "representation": representation,
                "dimension": int(dimension),
                "raw_input_dim": int(raw_input_dim),
                "selected_fold_dimension": int(selected_fold_dimension),
                "selected_by_validation": bool(selected_by_validation),
                "model_key": key,
                "clinical_contract": PRIMARY_CLINICAL_CONTRACT,
                "y_true": int(labels[row_index]),
                "predicted_probability": float(probability[offset]),
                "predicted_label": int(probability[offset] >= threshold),
                "threshold": threshold,
                "selected_C": float(fit.selected_c),
            }
        )
    return rows


def _diagnostic_row(
    fit: Any,
    matrix: np.ndarray,
    labels: np.ndarray,
    indices: Mapping[str, np.ndarray],
    *,
    fold: int,
    population: str,
    seed: int,
    arm: str,
    timing: str,
    model_family: str,
    representation: str,
    dimension: int,
    raw_input_dim: int,
) -> dict[str, Any]:
    metrics: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        positions = indices[split]
        metrics[split] = binary_metrics(
            labels[positions], fit.predict_proba(matrix[positions])
        )
    return {
        "population": population,
        "seed": int(seed),
        "arm": arm,
        "fold": int(fold),
        "timing": timing,
        "model_family": model_family,
        "representation": representation,
        "dimension": int(dimension),
        "raw_input_dim": int(raw_input_dim),
        "feature_dim": int(fit.feature_dim),
        "train_rows": int(fit.train_rows),
        "validation_rows": int(fit.validation_rows),
        "selected_C": float(fit.selected_c),
        "validation_selection_auroc": float(fit.validation_auroc),
        "train_auroc": float(metrics["train"]["auroc"]),
        "validation_auroc": float(metrics["val"]["auroc"]),
        "test_auroc": float(metrics["test"]["auroc"]),
        "train_auprc": float(metrics["train"]["auprc"]),
        "validation_auprc": float(metrics["val"]["auprc"]),
        "test_auprc": float(metrics["test"]["auprc"]),
        "train_brier": float(metrics["train"]["brier"]),
        "validation_brier": float(metrics["val"]["brier"]),
        "test_brier": float(metrics["test"]["brier"]),
        "train_test_auroc_gap": float(
            metrics["train"]["auroc"] - metrics["test"]["auroc"]
        ),
    }


def _select_dimension(
    candidates: Mapping[int, tuple[Any, np.ndarray]],
) -> tuple[int, Any, np.ndarray]:
    if set(candidates) != set(PCA_DIMENSIONS):
        raise ValueError("PCA candidate registry must contain exactly 8/16/32/64")
    best_score = max(float(value[0].validation_auroc) for value in candidates.values())
    eligible = [
        dimension
        for dimension, (fit, _) in candidates.items()
        if float(fit.validation_auroc) >= best_score - 1e-12
    ]
    selected = min(eligible)
    fit, matrix = candidates[selected]
    return selected, fit, matrix


def _save_pca_artifact(
    projector: Any,
    *,
    population: str,
    seed: int,
    arm: str,
    fold: int,
    timing: str,
) -> Path:
    path = (
        EXPERIMENT_ROOT
        / "features"
        / "pca"
        / f"seed_{seed}"
        / arm
        / f"fold_{fold}"
        / population
        / f"{timing}.private.npz"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary,
        mean=np.asarray(projector.model.mean_, dtype=np.float64),
        components=np.asarray(projector.model.components_, dtype=np.float64),
        explained_variance=np.asarray(
            projector.model.explained_variance_, dtype=np.float64
        ),
        explained_variance_ratio=np.asarray(
            projector.model.explained_variance_ratio_, dtype=np.float64
        ),
        singular_values=np.asarray(projector.model.singular_values_, dtype=np.float64),
        train_rows=np.asarray(projector.train_rows, dtype=np.int64),
        input_dim=np.asarray(projector.input_dim, dtype=np.int64),
        parameter_sha256=np.asarray(projector.parameter_sha256),
    )
    temporary.replace(path)
    return path


def _append_selected_family(
    *,
    candidates: Mapping[int, tuple[Any, np.ndarray]],
    prediction_rows: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    patient_ids: np.ndarray,
    labels: np.ndarray,
    indices: Mapping[str, np.ndarray],
    fold: int,
    population: str,
    seed: int,
    arm: str,
    timing: str,
    model_family: str,
    raw_input_dim: int,
) -> tuple[int, Any, np.ndarray]:
    selected, selected_fit, selected_matrix = _select_dimension(candidates)
    for dimension, (fit, matrix) in sorted(candidates.items()):
        prediction_rows.extend(
            _prediction_rows(
                fit,
                matrix,
                patient_ids=patient_ids,
                labels=labels,
                indices=indices,
                fold=fold,
                population=population,
                seed=seed,
                arm=arm,
                timing=timing,
                model_family=model_family,
                representation="pca",
                dimension=dimension,
                raw_input_dim=raw_input_dim,
                selected_fold_dimension=selected,
                selected_by_validation=dimension == selected,
            )
        )
    prediction_rows.extend(
        _prediction_rows(
            selected_fit,
            selected_matrix,
            patient_ids=patient_ids,
            labels=labels,
            indices=indices,
            fold=fold,
            population=population,
            seed=seed,
            arm=arm,
            timing=timing,
            model_family=model_family,
            representation="pca_selected",
            dimension=-1,
            raw_input_dim=raw_input_dim,
            selected_fold_dimension=selected,
            selected_by_validation=True,
        )
    )
    selection_rows.append(
        {
            "population": population,
            "seed": int(seed),
            "arm": arm,
            "fold": int(fold),
            "timing": timing,
            "model_family": model_family,
            "raw_input_dim": int(raw_input_dim),
            "selected_dimension": int(selected),
            "selected_C": float(selected_fit.selected_c),
            "validation_auroc": float(selected_fit.validation_auroc),
            "selection_metric": "validation_auroc",
            "tie_break": "smaller_dimension_then_smaller_C",
            "test_used_for_selection": False,
        }
    )
    return selected, selected_fit, selected_matrix


@dataclass(frozen=True)
class LateCandidate:
    dimension: int
    meta_fit: Any
    test_matrix: np.ndarray
    train_matrix: np.ndarray
    validation_matrix: np.ndarray
    reference_oof: Any
    mri_oof: Any
    reference_fit: Any
    mri_fit: Any


def _strict_reference_oof(
    *,
    goal2: Any,
    goal2_config: Mapping[str, Any],
    clinical: pd.DataFrame,
    labels: np.ndarray,
    outer_train: np.ndarray,
    patient_ids: np.ndarray,
    assignments: Any,
    selected_c: float,
    ftv_matrix: np.ndarray | None,
) -> tuple[Any, tuple[str, ...]]:
    local_labels = labels[outer_train]
    fit_hashes: list[str] = []

    def fit_predict(inner_train: np.ndarray, inner_holdout: np.ndarray) -> np.ndarray:
        global_train = outer_train[inner_train]
        global_holdout = outer_train[inner_holdout]
        encoder = goal2.TrainOnlyClinicalEncoder.from_config(
            goal2_config, PRIMARY_CLINICAL_CONTRACT
        )
        encoder.fit(clinical.iloc[global_train])
        train_matrix = np.asarray(
            encoder.transform(clinical.iloc[global_train]), dtype=np.float64
        )
        holdout_matrix = np.asarray(
            encoder.transform(clinical.iloc[global_holdout]), dtype=np.float64
        )
        if ftv_matrix is not None:
            train_matrix = np.concatenate(
                (train_matrix, ftv_matrix[global_train]), axis=1
            )
            holdout_matrix = np.concatenate(
                (holdout_matrix, ftv_matrix[global_holdout]), axis=1
            )
        fit = fit_fixed_c_logistic(
            train_matrix,
            labels[global_train],
            selected_c,
            solver=str(goal2_config["logistic"]["solver"]),
            max_iter=int(goal2_config["logistic"]["max_iter"]),
            random_state=0,
        )
        fit_hashes.append(fit.parameter_sha256)
        return fit.predict_proba(holdout_matrix)

    result = strict_inner_oof_probabilities(assignments, fit_predict)
    if tuple(result.patient_ids) != tuple(patient_ids[outer_train].astype(str)):
        raise AssertionError("reference OOF patient order drifted")
    if len(fit_hashes) != int(assignments.n_splits):
        raise AssertionError("reference OOF did not fit one model per inner fold")
    if set(np.unique(local_labels)) != {0, 1}:
        raise AssertionError("outer train labels lost a class")
    return result, tuple(fit_hashes)


def _inner_pca_cache(
    raw_outer_train: np.ndarray,
    assignments: Any,
    *,
    max_components: int,
) -> dict[int, tuple[np.ndarray, str]]:
    cache: dict[int, tuple[np.ndarray, str]] = {}
    for inner_fold in range(assignments.n_splits):
        inner_train, _ = assignments.indices(inner_fold)
        projector = fit_train_pca(
            raw_outer_train[inner_train], max_components=max_components
        )
        cache[inner_fold] = (
            projector.transform(raw_outer_train),
            projector.parameter_sha256,
        )
    return cache


def _strict_mri_oof(
    *,
    raw_outer_train: np.ndarray,
    labels_outer_train: np.ndarray,
    assignments: Any,
    inner_pca: Mapping[int, tuple[np.ndarray, str]],
    dimension: int,
    selected_c: float,
    goal2_config: Mapping[str, Any],
) -> tuple[Any, tuple[str, ...]]:
    model_hashes: list[str] = []
    callback_index = 0

    def fit_predict(inner_train: np.ndarray, inner_holdout: np.ndarray) -> np.ndarray:
        nonlocal callback_index
        inner_fold = callback_index
        callback_index += 1
        projected, _ = inner_pca[inner_fold]
        matrix = projected[:, :dimension]
        fit = fit_fixed_c_logistic(
            matrix[inner_train],
            labels_outer_train[inner_train],
            selected_c,
            solver=str(goal2_config["logistic"]["solver"]),
            max_iter=int(goal2_config["logistic"]["max_iter"]),
            random_state=0,
        )
        model_hashes.append(fit.parameter_sha256)
        return fit.predict_proba(matrix[inner_holdout])

    result = strict_inner_oof_probabilities(assignments, fit_predict)
    if callback_index != assignments.n_splits:
        raise AssertionError("MRI OOF callback coverage drifted")
    return result, tuple(model_hashes)


def _late_prediction_rows(
    candidate: LateCandidate,
    *,
    patient_ids: np.ndarray,
    labels: np.ndarray,
    indices: Mapping[str, np.ndarray],
    fold: int,
    population: str,
    seed: int,
    arm: str,
    timing: str,
    model_family: str,
    representation: str,
    dimension: int,
    raw_input_dim: int,
    selected_fold_dimension: int,
    selected_by_validation: bool,
) -> list[dict[str, Any]]:
    test = indices["test"]
    probability = candidate.meta_fit.predict_proba(candidate.test_matrix)
    threshold = float(candidate.meta_fit.threshold_selection.threshold)
    key = _model_key(model_family, representation, dimension)
    return [
        {
            "patient_id": str(patient_ids[row_index]),
            "fold": int(fold),
            "population": population,
            "seed": int(seed),
            "arm": arm,
            "timing": timing,
            "model_family": model_family,
            "representation": representation,
            "dimension": int(dimension),
            "raw_input_dim": int(raw_input_dim),
            "selected_fold_dimension": int(selected_fold_dimension),
            "selected_by_validation": bool(selected_by_validation),
            "model_key": key,
            "clinical_contract": PRIMARY_CLINICAL_CONTRACT,
            "y_true": int(labels[row_index]),
            "predicted_probability": float(probability[offset]),
            "predicted_label": int(probability[offset] >= threshold),
            "threshold": threshold,
            "selected_C": float(candidate.meta_fit.selected_c),
        }
        for offset, row_index in enumerate(test)
    ]


def _run_late_fusion(
    *,
    goal2: Any,
    goal2_config: Mapping[str, Any],
    compact_config: Mapping[str, Any],
    clinical: pd.DataFrame,
    patient_ids: np.ndarray,
    labels: np.ndarray,
    indices: Mapping[str, np.ndarray],
    raw: np.ndarray,
    reference_family: str,
    reference_fit: Any,
    reference_matrix: np.ndarray,
    mri_candidates: Mapping[int, tuple[Any, np.ndarray]],
    ftv_matrix: np.ndarray | None,
    fold: int,
    population: str,
    seed: int,
    arm: str,
    timing: str,
    prediction_rows: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    oof_rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
) -> None:
    outer_train = indices["train"]
    outer_validation = indices["val"]
    outer_test = indices["test"]
    late = compact_config["late_fusion"]
    assignments = stratified_inner_assignments(
        patient_ids[outer_train],
        labels[outer_train],
        n_splits=int(late["inner_folds"]),
        seed=int(late["inner_seed"]),
    )
    reference_oof, reference_fit_hashes = _strict_reference_oof(
        goal2=goal2,
        goal2_config=goal2_config,
        clinical=clinical,
        labels=labels,
        outer_train=outer_train,
        patient_ids=patient_ids,
        assignments=assignments,
        selected_c=float(reference_fit.selected_c),
        ftv_matrix=ftv_matrix,
    )
    raw_outer_train = raw[outer_train]
    inner_pca = _inner_pca_cache(
        raw_outer_train,
        assignments,
        max_components=int(compact_config["pca"]["max_components"]),
    )
    clip = float(late["probability_clip"])
    candidate_map: dict[int, LateCandidate] = {}
    candidate_diagnostic_metrics: dict[
        int, tuple[dict[str, Any], dict[str, Any]]
    ] = {}
    model_family = f"LateFusion({reference_family},M)"

    reference_validation_probability = reference_fit.predict_proba(
        reference_matrix[outer_validation]
    )
    reference_test_probability = reference_fit.predict_proba(
        reference_matrix[outer_test]
    )
    for dimension in PCA_DIMENSIONS:
        mri_fit, mri_matrix = mri_candidates[dimension]
        mri_oof, mri_fit_hashes = _strict_mri_oof(
            raw_outer_train=raw_outer_train,
            labels_outer_train=labels[outer_train],
            assignments=assignments,
            inner_pca=inner_pca,
            dimension=dimension,
            selected_c=float(mri_fit.selected_c),
            goal2_config=goal2_config,
        )
        meta_train = np.column_stack(
            (
                probability_to_logit(reference_oof.probabilities, clip=clip),
                probability_to_logit(mri_oof.probabilities, clip=clip),
            )
        )
        meta_validation = np.column_stack(
            (
                probability_to_logit(reference_validation_probability, clip=clip),
                probability_to_logit(
                    mri_fit.predict_proba(mri_matrix[outer_validation]), clip=clip
                ),
            )
        )
        meta_test = np.column_stack(
            (
                probability_to_logit(reference_test_probability, clip=clip),
                probability_to_logit(
                    mri_fit.predict_proba(mri_matrix[outer_test]), clip=clip
                ),
            )
        )
        logistic = goal2_config["logistic"]
        meta_fit = fit_binary_logistic(
            meta_train,
            labels[outer_train],
            meta_validation,
            labels[outer_validation],
            logistic["c_grid"],
            solver=str(logistic["solver"]),
            max_iter=int(logistic["max_iter"]),
            random_state=0,
        )
        candidate = LateCandidate(
            dimension=dimension,
            meta_fit=meta_fit,
            test_matrix=meta_test,
            train_matrix=meta_train,
            validation_matrix=meta_validation,
            reference_oof=reference_oof,
            mri_oof=mri_oof,
            reference_fit=reference_fit,
            mri_fit=mri_fit,
        )
        candidate_map[dimension] = candidate
        train_metric = binary_metrics(
            labels[outer_train], meta_fit.predict_proba(meta_train)
        )
        validation_metric = binary_metrics(
            labels[outer_validation], meta_fit.predict_proba(meta_validation)
        )
        candidate_diagnostic_metrics[dimension] = (train_metric, validation_metric)
        for local_index, patient_id in enumerate(reference_oof.patient_ids):
            inner_fold = int(reference_oof.inner_fold[local_index])
            if inner_fold != int(mri_oof.inner_fold[local_index]):
                raise AssertionError("reference/MRI inner-fold assignments differ")
            oof_rows.append(
                {
                    "patient_id": patient_id,
                    "outer_fold": int(fold),
                    "inner_fold": inner_fold,
                    "population": population,
                    "seed": int(seed),
                    "arm": arm,
                    "timing": timing,
                    "late_model_family": model_family,
                    "dimension": int(dimension),
                    "y_true": int(labels[outer_train][local_index]),
                    "reference_probability": float(
                        reference_oof.probabilities[local_index]
                    ),
                    "mri_probability": float(mri_oof.probabilities[local_index]),
                    "reference_logit": float(meta_train[local_index, 0]),
                    "mri_logit": float(meta_train[local_index, 1]),
                    "assignment_sha256": assignments.assignment_sha256,
                    "reference_fit_sha256": reference_fit_hashes[inner_fold],
                    "mri_fit_sha256": mri_fit_hashes[inner_fold],
                    "inner_pca_sha256": inner_pca[inner_fold][1],
                    "outer_validation_row": False,
                    "outer_test_row": False,
                }
            )

    best_validation = max(
        float(candidate.meta_fit.validation_auroc)
        for candidate in candidate_map.values()
    )
    selected_dimension = min(
        dimension
        for dimension, candidate in candidate_map.items()
        if float(candidate.meta_fit.validation_auroc) >= best_validation - 1e-12
    )
    # Test labels are first read only after the validation-selected dimension
    # has been frozen. Fixed-dimension test diagnostics remain prespecified
    # sensitivities and never feed back into this selection.
    for dimension, candidate in sorted(candidate_map.items()):
        train_metric, validation_metric = candidate_diagnostic_metrics[dimension]
        test_metric = binary_metrics(
            labels[outer_test], candidate.meta_fit.predict_proba(candidate.test_matrix)
        )
        diagnostic_rows.append(
            {
                "population": population,
                "seed": int(seed),
                "arm": arm,
                "fold": int(fold),
                "timing": timing,
                "model_family": model_family,
                "dimension": int(dimension),
                "raw_input_dim": int(raw.shape[1]),
                "reference_selected_C": float(reference_fit.selected_c),
                "mri_selected_C": float(candidate.mri_fit.selected_c),
                "meta_selected_C": float(candidate.meta_fit.selected_c),
                "validation_selection_auroc": float(
                    candidate.meta_fit.validation_auroc
                ),
                "train_auroc": float(train_metric["auroc"]),
                "validation_auroc": float(validation_metric["auroc"]),
                "test_auroc": float(test_metric["auroc"]),
                "train_test_auroc_gap": float(
                    train_metric["auroc"] - test_metric["auroc"]
                ),
                "inner_folds": int(assignments.n_splits),
                "inner_assignment_sha256": assignments.assignment_sha256,
                "reference_oof_sha256": candidate.reference_oof.prediction_sha256,
                "mri_oof_sha256": candidate.mri_oof.prediction_sha256,
                "strict_oof": True,
            }
        )
    for dimension, candidate in sorted(candidate_map.items()):
        prediction_rows.extend(
            _late_prediction_rows(
                candidate,
                patient_ids=patient_ids,
                labels=labels,
                indices=indices,
                fold=fold,
                population=population,
                seed=seed,
                arm=arm,
                timing=timing,
                model_family=model_family,
                representation="late_fusion_candidate",
                dimension=dimension,
                raw_input_dim=raw.shape[1],
                selected_fold_dimension=selected_dimension,
                selected_by_validation=dimension == selected_dimension,
            )
        )
    selected = candidate_map[selected_dimension]
    prediction_rows.extend(
        _late_prediction_rows(
            selected,
            patient_ids=patient_ids,
            labels=labels,
            indices=indices,
            fold=fold,
            population=population,
            seed=seed,
            arm=arm,
            timing=timing,
            model_family=model_family,
            representation="late_fusion",
            dimension=-1,
            raw_input_dim=raw.shape[1],
            selected_fold_dimension=selected_dimension,
            selected_by_validation=True,
        )
    )
    selection_rows.append(
        {
            "population": population,
            "seed": int(seed),
            "arm": arm,
            "fold": int(fold),
            "timing": timing,
            "model_family": model_family,
            "raw_input_dim": int(raw.shape[1]),
            "selected_dimension": int(selected_dimension),
            "selected_C": float(selected.meta_fit.selected_c),
            "validation_auroc": float(selected.meta_fit.validation_auroc),
            "selection_metric": "validation_auroc",
            "tie_break": "smaller_dimension_then_smaller_C",
            "test_used_for_selection": False,
        }
    )


def _append_profile_predictions(
    *,
    fit: Any,
    matrix: np.ndarray,
    patient_ids: np.ndarray,
    fold: int,
    labels: np.ndarray,
    test: np.ndarray,
    seed: int,
    arm: str,
    timing: str,
    target: str,
    representation: str,
    dimension: int,
    raw_input_dim: int,
) -> list[dict[str, Any]]:
    probability = fit.predict_proba(matrix[test])
    threshold = float(fit.threshold_selection.threshold)
    return [
        {
            "patient_id": str(patient_ids[row_index]),
            "fold": int(fold),
            "seed": int(seed),
            "arm": arm,
            "timing": timing,
            "target": target,
            "representation": representation,
            "dimension": int(dimension),
            "raw_input_dim": int(raw_input_dim),
            "y_true": int(labels[row_index]),
            "predicted_probability": float(probability[offset]),
            "predicted_label": int(probability[offset] >= threshold),
            "threshold": threshold,
            **{column: math.nan for column in SUBTYPE_PROBABILITY_COLUMNS.values()},
        }
        for offset, row_index in enumerate(test)
    ]


def _run_profile_probes(
    *,
    goal2_config: Mapping[str, Any],
    patient_ids: np.ndarray,
    clinical: pd.DataFrame,
    labels_by_target: Mapping[str, np.ndarray],
    indices: Mapping[str, np.ndarray],
    representations: Mapping[tuple[str, int], np.ndarray],
    raw_input_dim: int,
    seed: int,
    arm: str,
    fold: int,
    timing: str,
    prediction_rows: list[dict[str, Any]],
    hyperparameter_rows: list[dict[str, Any]],
) -> None:
    train, validation, test = (
        indices["train"],
        indices["val"],
        indices["test"],
    )
    logistic = goal2_config["logistic"]
    class_weight = goal2_config["profile_logistic"]["class_weight"]
    for (representation, dimension), matrix in representations.items():
        for target in ("HR", "HER2"):
            labels = labels_by_target[target]
            fit = fit_binary_logistic(
                matrix[train],
                labels[train],
                matrix[validation],
                labels[validation],
                logistic["c_grid"],
                class_weight=class_weight,
                solver=str(logistic["solver"]),
                max_iter=int(logistic["max_iter"]),
                random_state=0,
            )
            prediction_rows.extend(
                _append_profile_predictions(
                    fit=fit,
                    matrix=matrix,
                    patient_ids=patient_ids,
                    fold=fold,
                    labels=labels,
                    test=test,
                    seed=seed,
                    arm=arm,
                    timing=timing,
                    target=target,
                    representation=representation,
                    dimension=dimension,
                    raw_input_dim=raw_input_dim,
                )
            )
            hyperparameter_rows.append(
                {
                    "seed": int(seed),
                    "arm": arm,
                    "fold": int(fold),
                    "timing": timing,
                    "target": target,
                    "representation": representation,
                    "dimension": int(dimension),
                    "raw_input_dim": int(raw_input_dim),
                    "selected_C": float(fit.selected_c),
                    "validation_auroc": float(fit.validation_auroc),
                    "feature_dim": int(fit.feature_dim),
                }
            )

        subtype = labels_by_target["subtype_4class"]
        fit_multi = fit_multiclass_logistic(
            matrix[train],
            subtype[train],
            matrix[validation],
            subtype[validation],
            logistic["c_grid"],
            solver=str(logistic["solver"]),
            max_iter=int(logistic["max_iter"]),
            random_state=0,
        )
        probability = fit_multi.predict_proba(matrix[test])
        predicted = fit_multi.predict(matrix[test])
        class_index = {
            str(value): index for index, value in enumerate(fit_multi.classes)
        }
        if set(class_index) != set(SUBTYPE_CLASSES):
            raise ValueError("profile subtype class contract drifted")
        for offset, row_index in enumerate(test):
            row: dict[str, Any] = {
                "patient_id": str(patient_ids[row_index]),
                "fold": int(fold),
                "seed": int(seed),
                "arm": arm,
                "timing": timing,
                "target": "subtype_4class",
                "representation": representation,
                "dimension": int(dimension),
                "raw_input_dim": int(raw_input_dim),
                "y_true": str(subtype[row_index]),
                "predicted_probability": math.nan,
                "predicted_label": str(predicted[offset]),
                "threshold": math.nan,
            }
            for subtype_name, column in SUBTYPE_PROBABILITY_COLUMNS.items():
                row[column] = float(probability[offset, class_index[subtype_name]])
            prediction_rows.append(row)
        hyperparameter_rows.append(
            {
                "seed": int(seed),
                "arm": arm,
                "fold": int(fold),
                "timing": timing,
                "target": "subtype_4class",
                "representation": representation,
                "dimension": int(dimension),
                "raw_input_dim": int(raw_input_dim),
                "selected_C": float(fit_multi.selected_c),
                "validation_auroc": float(fit_multi.validation_macro_ovr_auroc),
                "feature_dim": int(fit_multi.feature_dim),
            }
        )


def _save_random_projection_artifact(
    projection: Any, *, timing: str, dimension: int
) -> Path:
    path = (
        EXPERIMENT_ROOT
        / "features"
        / "random_projection"
        / timing
        / f"R{dimension}.private.npz"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary,
        matrix=np.asarray(projection.matrix, dtype=np.float64),
        input_dim=np.asarray(projection.input_dim, dtype=np.int64),
        output_dim=np.asarray(projection.output_dim, dtype=np.int64),
        seed=np.asarray(projection.seed, dtype=np.int64),
        matrix_sha256=np.asarray(projection.matrix_sha256),
    )
    temporary.replace(path)
    return path


def run_experiment(inputs: Any) -> Mapping[str, pd.DataFrame]:
    goal2 = load_goal2_contract_module(repo_root=REPO_ROOT)
    goal2_config = inputs.goal2_config
    compact_config = inputs.compact_config
    pcr_prediction_rows: list[dict[str, Any]] = []
    profile_prediction_rows: list[dict[str, Any]] = []
    pca_variance_rows: list[dict[str, Any]] = []
    pca_component_rows: list[dict[str, Any]] = []
    pca_artifact_rows: list[dict[str, Any]] = []
    rp_ledger_rows: list[dict[str, Any]] = []
    dimension_selection_rows: list[dict[str, Any]] = []
    fit_diagnostic_rows: list[dict[str, Any]] = []
    late_oof_rows: list[dict[str, Any]] = []
    late_diagnostic_rows: list[dict[str, Any]] = []
    profile_hyperparameter_rows: list[dict[str, Any]] = []
    rp_cache: dict[tuple[str, int], Any] = {}

    for cell_index, ((seed, arm, fold), asset) in enumerate(
        sorted(inputs.assets.items()), start=1
    ):
        for population in POPULATIONS:
            view = build_population_view(
                asset, inputs.clinical, inputs.ftv_wide, population
            )
            patient_ids = view.patient_id
            labels = view.clinical["label_pcr"].to_numpy(dtype=np.int64)
            indices = view.indices
            clinical_matrix, _ = _clinical_matrix(
                goal2, goal2_config, view.clinical, indices
            )
            clinical_fit = _fit_binary(
                clinical_matrix, labels, indices, goal2_config
            )
            labels_by_target: Mapping[str, np.ndarray] = {
                "HR": view.clinical["label_hr"].to_numpy(dtype=np.int64),
                "HER2": view.clinical["label_her2"].to_numpy(dtype=np.int64),
                "subtype_4class": view.clinical["hr_her2_subtype"]
                .astype(str)
                .to_numpy(),
            }

            for timing in TIMINGS:
                raw = np.asarray(
                    goal2.mri_timing_prefix(view.response_state, timing),
                    dtype=np.float64,
                )
                raw_input_dim = int(raw.shape[1])
                projector = fit_train_pca(
                    raw[indices["train"]],
                    max_components=int(compact_config["pca"]["max_components"]),
                    svd_solver=str(compact_config["pca"]["svd_solver"]),
                    whiten=bool(compact_config["pca"]["whiten"]),
                )
                projected = projector.transform(raw)
                artifact = _save_pca_artifact(
                    projector,
                    population=population,
                    seed=seed,
                    arm=arm,
                    fold=fold,
                    timing=timing,
                )
                common_pca = {
                    "population": population,
                    "seed": int(seed),
                    "arm": arm,
                    "fold": int(fold),
                    "timing": timing,
                    "raw_input_dim": raw_input_dim,
                    "train_rows": int(len(indices["train"])),
                    "fit_scope": "outer_train_timing_prefix",
                    "validation_rows_in_fit": 0,
                    "test_rows_in_fit": 0,
                    "fitted_transform_sha256": projector.parameter_sha256,
                }
                for row in projector.variance_ledger(PCA_DIMENSIONS):
                    pca_variance_rows.append({**common_pca, **row})
                for row in projector.component_variance_ledger():
                    pca_component_rows.append({**common_pca, **row})
                pca_artifact_rows.append(
                    {
                        **common_pca,
                        "artifact_path": str(artifact.relative_to(EXPERIMENT_ROOT)),
                        "artifact_sha256": file_sha256(artifact),
                    }
                )

                # Repeated per timing so every paired model cell has an exact
                # clinical reference row, while the underlying fit is unchanged.
                pcr_prediction_rows.extend(
                    _prediction_rows(
                        clinical_fit,
                        clinical_matrix,
                        patient_ids=patient_ids,
                        labels=labels,
                        indices=indices,
                        fold=fold,
                        population=population,
                        seed=seed,
                        arm=arm,
                        timing=timing,
                        model_family="C",
                        representation="none",
                        dimension=-1,
                        raw_input_dim=raw_input_dim,
                    )
                )
                fit_diagnostic_rows.append(
                    _diagnostic_row(
                        clinical_fit,
                        clinical_matrix,
                        labels,
                        indices,
                        fold=fold,
                        population=population,
                        seed=seed,
                        arm=arm,
                        timing=timing,
                        model_family="C",
                        representation="none",
                        dimension=-1,
                        raw_input_dim=raw_input_dim,
                    )
                )

                ftv_matrix: np.ndarray | None = None
                ftv_fit: Any | None = None
                clinical_ftv_fit: Any | None = None
                clinical_ftv_matrix: np.ndarray | None = None
                if population == "ftv_complete_375":
                    if view.ftv_wide is None:
                        raise AssertionError("FTV population has no aligned FTV table")
                    ftv_matrix = np.asarray(
                        goal2.ftv_timing_prefix(view.ftv_wide, timing),
                        dtype=np.float64,
                    )
                    clinical_ftv_matrix = np.concatenate(
                        (clinical_matrix, ftv_matrix), axis=1
                    )
                    for family, matrix in (
                        ("F", ftv_matrix),
                        ("C+F", clinical_ftv_matrix),
                    ):
                        fit = _fit_binary(matrix, labels, indices, goal2_config)
                        if family == "F":
                            ftv_fit = fit
                        else:
                            clinical_ftv_fit = fit
                        pcr_prediction_rows.extend(
                            _prediction_rows(
                                fit,
                                matrix,
                                patient_ids=patient_ids,
                                labels=labels,
                                indices=indices,
                                fold=fold,
                                population=population,
                                seed=seed,
                                arm=arm,
                                timing=timing,
                                model_family=family,
                                representation="none",
                                dimension=-1,
                                raw_input_dim=raw_input_dim,
                            )
                        )
                        fit_diagnostic_rows.append(
                            _diagnostic_row(
                                fit,
                                matrix,
                                labels,
                                indices,
                                fold=fold,
                                population=population,
                                seed=seed,
                                arm=arm,
                                timing=timing,
                                model_family=family,
                                representation="none",
                                dimension=-1,
                                raw_input_dim=raw_input_dim,
                            )
                        )

                raw_family_matrices: dict[str, np.ndarray] = {
                    "M": raw,
                    "C+M": np.concatenate((clinical_matrix, raw), axis=1),
                }
                if clinical_ftv_matrix is not None:
                    raw_family_matrices["C+F+M"] = np.concatenate(
                        (clinical_ftv_matrix, raw), axis=1
                    )
                for family, matrix in raw_family_matrices.items():
                    fit = _fit_binary(matrix, labels, indices, goal2_config)
                    pcr_prediction_rows.extend(
                        _prediction_rows(
                            fit,
                            matrix,
                            patient_ids=patient_ids,
                            labels=labels,
                            indices=indices,
                            fold=fold,
                            population=population,
                            seed=seed,
                            arm=arm,
                            timing=timing,
                            model_family=family,
                            representation="raw",
                            dimension=raw_input_dim,
                            raw_input_dim=raw_input_dim,
                        )
                    )
                    fit_diagnostic_rows.append(
                        _diagnostic_row(
                            fit,
                            matrix,
                            labels,
                            indices,
                            fold=fold,
                            population=population,
                            seed=seed,
                            arm=arm,
                            timing=timing,
                            model_family=family,
                            representation="raw",
                            dimension=raw_input_dim,
                            raw_input_dim=raw_input_dim,
                        )
                    )

                pca_candidates: dict[str, dict[int, tuple[Any, np.ndarray]]] = {
                    "M": {},
                    "C+M": {},
                }
                if ftv_matrix is not None and clinical_ftv_matrix is not None:
                    pca_candidates.update(
                        {
                            "C+F+M": {},
                            "M_residual": {},
                            "C+F+M_residual": {},
                        }
                    )
                pca_diagnostic_tasks: list[
                    tuple[int, str, Any, np.ndarray]
                ] = []
                for dimension in PCA_DIMENSIONS:
                    compact = projected[:, :dimension]
                    family_matrices: dict[str, np.ndarray] = {
                        "M": compact,
                        "C+M": np.concatenate((clinical_matrix, compact), axis=1),
                    }
                    if ftv_matrix is not None and clinical_ftv_matrix is not None:
                        residualizer = fit_ftv_mri_residualizer(
                            ftv_matrix[indices["train"]],
                            compact[indices["train"]],
                        )
                        compact_residual = residualizer.transform(
                            ftv_matrix, compact
                        )
                        family_matrices.update(
                            {
                                "C+F+M": np.concatenate(
                                    (clinical_ftv_matrix, compact), axis=1
                                ),
                                "M_residual": compact_residual,
                                "C+F+M_residual": np.concatenate(
                                    (clinical_ftv_matrix, compact_residual), axis=1
                                ),
                            }
                        )
                    for family, matrix in family_matrices.items():
                        fit = _fit_binary(matrix, labels, indices, goal2_config)
                        pca_candidates[family][dimension] = (fit, matrix)
                        pca_diagnostic_tasks.append(
                            (dimension, family, fit, matrix)
                        )
                # Each family locks k from validation AUROC before any PCA
                # candidate reads held-out test labels or emits test scores.
                for family, candidates in pca_candidates.items():
                    _append_selected_family(
                        candidates=candidates,
                        prediction_rows=pcr_prediction_rows,
                        selection_rows=dimension_selection_rows,
                        patient_ids=patient_ids,
                        labels=labels,
                        indices=indices,
                        fold=fold,
                        population=population,
                        seed=seed,
                        arm=arm,
                        timing=timing,
                        model_family=family,
                        raw_input_dim=raw_input_dim,
                    )
                for dimension, family, fit, matrix in pca_diagnostic_tasks:
                    fit_diagnostic_rows.append(
                        _diagnostic_row(
                            fit,
                            matrix,
                            labels,
                            indices,
                            fold=fold,
                            population=population,
                            seed=seed,
                            arm=arm,
                            timing=timing,
                            model_family=family,
                            representation="pca",
                            dimension=dimension,
                            raw_input_dim=raw_input_dim,
                        )
                    )

                for dimension in RP_DIMENSIONS:
                    cache_key = (timing, dimension)
                    projection = rp_cache.get(cache_key)
                    if projection is None:
                        projection = make_gaussian_random_projection(
                            raw_input_dim,
                            dimension,
                            seed=int(compact_config["random_projection"]["seed"]),
                        )
                        rp_cache[cache_key] = projection
                        artifact_path = _save_random_projection_artifact(
                            projection, timing=timing, dimension=dimension
                        )
                        rp_ledger_rows.append(
                            {
                                "timing": timing,
                                "raw_input_dim": raw_input_dim,
                                "dimension": dimension,
                                "seed": int(projection.seed),
                                "distribution": projection.distribution,
                                "matrix_sha256": projection.matrix_sha256,
                                "artifact_path": str(
                                    artifact_path.relative_to(EXPERIMENT_ROOT)
                                ),
                                "artifact_sha256": file_sha256(artifact_path),
                                "reads_labels": False,
                                "reads_patient_data": False,
                            }
                        )
                    random_compact = projection.transform(raw)
                    rp_family_matrices: dict[str, np.ndarray] = {
                        "M": random_compact,
                        "C+M": np.concatenate(
                            (clinical_matrix, random_compact), axis=1
                        ),
                    }
                    if clinical_ftv_matrix is not None:
                        rp_family_matrices["C+F+M"] = np.concatenate(
                            (clinical_ftv_matrix, random_compact), axis=1
                        )
                    for family, matrix in rp_family_matrices.items():
                        fit = _fit_binary(matrix, labels, indices, goal2_config)
                        pcr_prediction_rows.extend(
                            _prediction_rows(
                                fit,
                                matrix,
                                patient_ids=patient_ids,
                                labels=labels,
                                indices=indices,
                                fold=fold,
                                population=population,
                                seed=seed,
                                arm=arm,
                                timing=timing,
                                model_family=family,
                                representation="random_projection",
                                dimension=dimension,
                                raw_input_dim=raw_input_dim,
                            )
                        )
                        fit_diagnostic_rows.append(
                            _diagnostic_row(
                                fit,
                                matrix,
                                labels,
                                indices,
                                fold=fold,
                                population=population,
                                seed=seed,
                                arm=arm,
                                timing=timing,
                                model_family=family,
                                representation="random_projection",
                                dimension=dimension,
                                raw_input_dim=raw_input_dim,
                            )
                        )

                if population == "full_808":
                    _run_profile_probes(
                        goal2_config=goal2_config,
                        patient_ids=patient_ids,
                        clinical=view.clinical,
                        labels_by_target=labels_by_target,
                        indices=indices,
                        representations={
                            ("raw", raw_input_dim): raw,
                            ("pca16", 16): projected[:, :16],
                            ("pca32", 32): projected[:, :32],
                        },
                        raw_input_dim=raw_input_dim,
                        seed=seed,
                        arm=arm,
                        fold=fold,
                        timing=timing,
                        prediction_rows=profile_prediction_rows,
                        hyperparameter_rows=profile_hyperparameter_rows,
                    )

                _run_late_fusion(
                    goal2=goal2,
                    goal2_config=goal2_config,
                    compact_config=compact_config,
                    clinical=view.clinical,
                    patient_ids=patient_ids,
                    labels=labels,
                    indices=indices,
                    raw=raw,
                    reference_family="C",
                    reference_fit=clinical_fit,
                    reference_matrix=clinical_matrix,
                    mri_candidates=pca_candidates["M"],
                    ftv_matrix=None,
                    fold=fold,
                    population=population,
                    seed=seed,
                    arm=arm,
                    timing=timing,
                    prediction_rows=pcr_prediction_rows,
                    selection_rows=dimension_selection_rows,
                    oof_rows=late_oof_rows,
                    diagnostic_rows=late_diagnostic_rows,
                )
                if population == "ftv_complete_375":
                    if clinical_ftv_fit is None or clinical_ftv_matrix is None:
                        raise AssertionError("missing C+F base for late fusion")
                    _run_late_fusion(
                        goal2=goal2,
                        goal2_config=goal2_config,
                        compact_config=compact_config,
                        clinical=view.clinical,
                        patient_ids=patient_ids,
                        labels=labels,
                        indices=indices,
                        raw=raw,
                        reference_family="C+F",
                        reference_fit=clinical_ftv_fit,
                        reference_matrix=clinical_ftv_matrix,
                        mri_candidates=pca_candidates["M"],
                        ftv_matrix=ftv_matrix,
                        fold=fold,
                        population=population,
                        seed=seed,
                        arm=arm,
                        timing=timing,
                        prediction_rows=pcr_prediction_rows,
                        selection_rows=dimension_selection_rows,
                        oof_rows=late_oof_rows,
                        diagnostic_rows=late_diagnostic_rows,
                    )

            print(
                f"cell {cell_index:02d}/20 {seed}/{arm}/fold{fold} "
                f"{population} complete",
                flush=True,
            )

    return {
        "pcr_predictions": pd.DataFrame(pcr_prediction_rows),
        "profile_predictions": pd.DataFrame(profile_prediction_rows),
        "pca_variance": pd.DataFrame(pca_variance_rows),
        "pca_components": pd.DataFrame(pca_component_rows),
        "pca_artifacts": pd.DataFrame(pca_artifact_rows),
        "random_projection_ledger": pd.DataFrame(rp_ledger_rows),
        "dimension_selection": pd.DataFrame(dimension_selection_rows),
        "fit_diagnostics": pd.DataFrame(fit_diagnostic_rows),
        "late_oof": pd.DataFrame(late_oof_rows),
        "late_diagnostics": pd.DataFrame(late_diagnostic_rows),
        "profile_hyperparameters": pd.DataFrame(profile_hyperparameter_rows),
    }


def _comparison_specs() -> tuple[ComparisonSpec, ...]:
    both = tuple(POPULATIONS)
    ftv = ("ftv_complete_375",)
    return (
        ComparisonSpec(
            "delta1_C_plus_Mk_vs_C",
            {"model_key": "C"},
            {"model_key": "C+M|PCA_SELECTED"},
            both,
        ),
        ComparisonSpec(
            "delta2_CF_plus_Mk_vs_CF",
            {"model_key": "C+F"},
            {"model_key": "C+F+M|PCA_SELECTED"},
            ftv,
        ),
        ComparisonSpec(
            "delta3_late_C_Mk_vs_C",
            {"model_key": "C"},
            {"model_key": "LateFusion(C,M)|PCA_SELECTED"},
            both,
        ),
        ComparisonSpec(
            "delta4_late_CF_Mk_vs_CF",
            {"model_key": "C+F"},
            {"model_key": "LateFusion(C+F,M)|PCA_SELECTED"},
            ftv,
        ),
        ComparisonSpec(
            "raw_vs_compact_M",
            {"model_key": "M|RAW"},
            {"model_key": "M|PCA_SELECTED"},
            both,
        ),
        ComparisonSpec(
            "raw_vs_compact_C_plus_M",
            {"model_key": "C+M|RAW"},
            {"model_key": "C+M|PCA_SELECTED"},
            both,
        ),
        ComparisonSpec(
            "raw_vs_compact_CF_plus_M",
            {"model_key": "C+F+M|RAW"},
            {"model_key": "C+F+M|PCA_SELECTED"},
            ftv,
        ),
        ComparisonSpec(
            "residual_beyond_ftv",
            {"model_key": "C+F"},
            {"model_key": "C+F+M_residual|PCA_SELECTED"},
            ftv,
        ),
        ComparisonSpec(
            "late_vs_concat_C_M",
            {"model_key": "C+M|PCA_SELECTED"},
            {"model_key": "LateFusion(C,M)|PCA_SELECTED"},
            both,
        ),
        ComparisonSpec(
            "late_vs_concat_CF_M",
            {"model_key": "C+F+M|PCA_SELECTED"},
            {"model_key": "LateFusion(C+F,M)|PCA_SELECTED"},
            ftv,
        ),
        ComparisonSpec(
            "RP16_C_plus_M_vs_C",
            {"model_key": "C"},
            {"model_key": "C+M|R16"},
            both,
        ),
        ComparisonSpec(
            "RP32_C_plus_M_vs_C",
            {"model_key": "C"},
            {"model_key": "C+M|R32"},
            both,
        ),
        ComparisonSpec(
            "RP16_CF_plus_M_vs_CF",
            {"model_key": "C+F"},
            {"model_key": "C+F+M|R16"},
            ftv,
        ),
        ComparisonSpec(
            "RP32_CF_plus_M_vs_CF",
            {"model_key": "C+F"},
            {"model_key": "C+F+M|R32"},
            ftv,
        ),
    )


def _pca_variance_table(frame: pd.DataFrame) -> pd.DataFrame:
    groups = ["population", "timing", "dimension", "raw_input_dim"]
    return (
        frame.groupby(groups, as_index=False, sort=True)
        .agg(
            n_outer_pca_fits=("fitted_transform_sha256", "nunique"),
            cumulative_explained_variance_mean=(
                "cumulative_explained_variance_ratio",
                "mean",
            ),
            cumulative_explained_variance_min=(
                "cumulative_explained_variance_ratio",
                "min",
            ),
            cumulative_explained_variance_max=(
                "cumulative_explained_variance_ratio",
                "max",
            ),
        )
        .sort_values(groups, kind="stable")
        .reset_index(drop=True)
    )


def _dimension_frequency_table(frame: pd.DataFrame) -> pd.DataFrame:
    groups = ["population", "timing", "model_family", "selected_dimension"]
    counts = (
        frame.groupby(groups, as_index=False, sort=True)
        .size()
        .rename(columns={"size": "n_folds_selected"})
    )
    totals = counts.groupby(groups[:-1])["n_folds_selected"].transform("sum")
    counts["selection_fraction"] = counts["n_folds_selected"] / totals
    return counts


def _overfitting_table(
    diagnostics: pd.DataFrame, selections: pd.DataFrame
) -> pd.DataFrame:
    keys = ["population", "seed", "arm", "fold", "timing", "model_family"]
    raw = diagnostics.loc[
        diagnostics["representation"].eq("raw")
        & diagnostics["model_family"].isin(("M", "C+M", "C+F+M"))
    ].copy()
    raw["audit_representation"] = "raw"
    selected = diagnostics.loc[diagnostics["representation"].eq("pca")].merge(
        selections.loc[:, [*keys, "selected_dimension"]],
        on=keys,
        how="inner",
        validate="many_to_one",
    )
    selected = selected.loc[
        selected["dimension"].eq(selected["selected_dimension"])
        & selected["model_family"].isin(("M", "C+M", "C+F+M"))
    ].copy()
    selected["audit_representation"] = "compact_selected"
    combined = pd.concat((raw, selected), ignore_index=True)
    groups = ["population", "timing", "model_family", "audit_representation"]
    return (
        combined.groupby(groups, as_index=False, sort=True)
        .agg(
            n_fold_cells=("fold", "size"),
            train_auroc_mean=("train_auroc", "mean"),
            validation_auroc_mean=("validation_auroc", "mean"),
            test_auroc_mean=("test_auroc", "mean"),
            train_test_auroc_gap_mean=("train_test_auroc_gap", "mean"),
            train_test_auroc_gap_min=("train_test_auroc_gap", "min"),
            train_test_auroc_gap_max=("train_test_auroc_gap", "max"),
        )
        .reset_index(drop=True)
    )


def _raw_regression_check(pcr_metrics: pd.DataFrame) -> pd.DataFrame:
    oracle = pd.read_csv(GOAL2_ROOT / "metrics" / "pcr_oof_metrics.csv")
    mapping = {
        "C": "C",
        "F": "F",
        "C+F": "C+F",
        "M": "M|RAW",
        "C+M": "C+M|RAW",
        "C+F+M": "C+F+M|RAW",
    }
    oracle = oracle.loc[oracle["model"].isin(mapping)].copy()
    oracle["model_key"] = oracle["model"].map(mapping)
    keys = ["population", "seed", "arm", "timing", "model_key"]
    observed = pcr_metrics.loc[
        pcr_metrics["model_key"].isin(set(mapping.values())),
        [*keys, "n", "auroc", "auprc", "brier"],
    ]
    merged = oracle.loc[:, [*keys, "n", "auroc", "auprc", "brier"]].merge(
        observed,
        on=keys,
        how="outer",
        validate="one_to_one",
        suffixes=("_goal2", "_compact_audit"),
        indicator=True,
    )
    for metric in ("auroc", "auprc", "brier"):
        merged[f"abs_diff_{metric}"] = (
            merged[f"{metric}_compact_audit"] - merged[f"{metric}_goal2"]
        ).abs()
    merged["pass"] = (
        merged["_merge"].eq("both")
        & merged["n_goal2"].eq(merged["n_compact_audit"])
        & merged[
            ["abs_diff_auroc", "abs_diff_auprc", "abs_diff_brier"]
        ].max(axis=1).le(1e-12)
    )
    return merged


def _selected_metrics_table(
    metrics: pd.DataFrame, model_keys: Sequence[str]
) -> pd.DataFrame:
    return metrics.loc[metrics["model_key"].isin(model_keys)].sort_values(
        ["population", "seed", "arm", "timing", "model_key"], kind="stable"
    )


def make_tables(
    *,
    frames: Mapping[str, pd.DataFrame],
    pcr_metrics: pd.DataFrame,
    profile_metrics: pd.DataFrame,
    point_effects: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
) -> Mapping[str, pd.DataFrame]:
    return {
        "table1": _pca_variance_table(frames["pca_variance"]),
        "table2": _selected_metrics_table(
            pcr_metrics, ("M|RAW", "M|PCA_SELECTED")
        ),
        "table3": point_effects.loc[
            point_effects["comparison_name"].eq("delta1_C_plus_Mk_vs_C")
        ].reset_index(drop=True),
        "table4": point_effects.loc[
            point_effects["comparison_name"].eq("delta2_CF_plus_Mk_vs_CF")
        ].reset_index(drop=True),
        "table5": point_effects.loc[
            point_effects["comparison_name"].str.startswith("raw_vs_compact")
        ].reset_index(drop=True),
        "table6": point_effects.loc[
            point_effects["comparison_name"].isin(
                (
                    "delta3_late_C_Mk_vs_C",
                    "delta4_late_CF_Mk_vs_CF",
                    "late_vs_concat_C_M",
                    "late_vs_concat_CF_M",
                )
            )
        ].reset_index(drop=True),
        "table7": profile_metrics.copy(),
        "table8": bootstrap_summary.copy(),
        "dimension_frequency": _dimension_frequency_table(
            frames["dimension_selection"]
        ),
        "overfitting": _overfitting_table(
            frames["fit_diagnostics"], frames["dimension_selection"]
        ),
        "residual": _selected_metrics_table(
            pcr_metrics,
            ("M_residual|PCA_SELECTED", "C+F+M_residual|PCA_SELECTED"),
        ),
        "random_projection": _selected_metrics_table(
            pcr_metrics,
            (
                "M|R16",
                "M|R32",
                "C+M|R16",
                "C+M|R32",
                "C+F+M|R16",
                "C+F+M|R32",
            ),
        ),
    }


BASE_OUTPUT_PATHS: Mapping[str, Path] = {
    "pcr_predictions": EXPERIMENT_ROOT / "predictions" / "pcr_oof.private.csv",
    "profile_predictions": EXPERIMENT_ROOT
    / "predictions"
    / "profile_oof.private.csv",
    "late_oof": EXPERIMENT_ROOT
    / "predictions"
    / "late_fusion_inner_oof.private.csv",
    "pca_variance": EXPERIMENT_ROOT / "metrics" / "pca_explained_variance.csv",
    "pca_components": EXPERIMENT_ROOT
    / "metrics"
    / "pca_component_explained_variance.csv",
    "pca_artifacts": EXPERIMENT_ROOT / "metrics" / "pca_artifact_manifest.csv",
    "random_projection_ledger": EXPERIMENT_ROOT
    / "metrics"
    / "random_projection_ledger.csv",
    "dimension_selection": EXPERIMENT_ROOT
    / "metrics"
    / "selected_dimensions_by_fold.csv",
    "fit_diagnostics": EXPERIMENT_ROOT / "metrics" / "fit_diagnostics.csv",
    "late_diagnostics": EXPERIMENT_ROOT
    / "metrics"
    / "late_fusion_diagnostics.csv",
    "profile_hyperparameters": EXPERIMENT_ROOT
    / "metrics"
    / "profile_hyperparameters.csv",
}

DERIVED_OUTPUT_PATHS: Mapping[str, Path] = {
    "pcr_metrics": EXPERIMENT_ROOT / "metrics" / "pcr_oof_metrics.csv",
    "profile_metrics": EXPERIMENT_ROOT / "metrics" / "profile_oof_metrics.csv",
    "paired_effects": EXPERIMENT_ROOT / "metrics" / "paired_effects.csv",
    "bootstrap_summary": EXPERIMENT_ROOT / "metrics" / "bootstrap_ci.csv",
    "bootstrap_draws": EXPERIMENT_ROOT
    / "predictions"
    / "bootstrap_draws.private.csv",
    "raw_regression": EXPERIMENT_ROOT / "metrics" / "goal2_raw_regression_check.csv",
    "table1": EXPERIMENT_ROOT
    / "metrics"
    / "table1_pca_dimension_explained_variance.csv",
    "table2": EXPERIMENT_ROOT
    / "metrics"
    / "table2_mri_only_raw_vs_compact.csv",
    "table3": EXPERIMENT_ROOT / "metrics" / "table3_c_vs_c_plus_m.csv",
    "table4": EXPERIMENT_ROOT / "metrics" / "table4_cf_vs_cf_plus_m.csv",
    "table5": EXPERIMENT_ROOT
    / "metrics"
    / "table5_raw_vs_compact_paired_effects.csv",
    "table6": EXPERIMENT_ROOT / "metrics" / "table6_late_fusion.csv",
    "table7": EXPERIMENT_ROOT / "metrics" / "table7_profile_decodability.csv",
    "table8": EXPERIMENT_ROOT / "metrics" / "table8_bootstrap_ci.csv",
    "dimension_frequency": EXPERIMENT_ROOT
    / "metrics"
    / "dimension_selection_frequency.csv",
    "overfitting": EXPERIMENT_ROOT / "metrics" / "overfitting_diagnostics.csv",
    "residual": EXPERIMENT_ROOT / "metrics" / "residualized_compact_metrics.csv",
    "random_projection": EXPERIMENT_ROOT
    / "metrics"
    / "random_projection_sensitivity.csv",
    "run_summary": EXPERIMENT_ROOT / "metrics" / "run_summary.json",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=EXPERIMENT_ROOT / "configs" / "audit.json"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Reuse already written private/base ledgers and rerun summaries/bootstrap.",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=None,
        help="Testing override; the formal contract requires 2000.",
    )
    return parser.parse_args(argv)


def _read_base_frames() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for name, path in BASE_OUTPUT_PATHS.items():
        if not path.is_file():
            raise FileNotFoundError(f"summarize-only input is missing: {path}")
        frames[name] = pd.read_csv(path)
    return frames


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    os.umask(0o077)
    started = time.time()
    if args.bootstrap_replicates is not None and args.bootstrap_replicates <= 0:
        raise ValueError("--bootstrap-replicates must be positive")

    if args.summarize_only:
        require_known_output_policy(
            DERIVED_OUTPUT_PATHS.values(), args.overwrite, output_root=EXPERIMENT_ROOT
        )
        frames = _read_base_frames()
        inputs = load_frozen_goal2_inputs(
            args.config, repo_root=REPO_ROOT, load_assets=False
        )
        print("loaded existing base ledgers for summarize-only mode", flush=True)
    else:
        require_known_output_policy(
            [*BASE_OUTPUT_PATHS.values(), *DERIVED_OUTPUT_PATHS.values()],
            args.overwrite,
            output_root=EXPERIMENT_ROOT,
        )
        inputs = load_frozen_goal2_inputs(
            args.config, repo_root=REPO_ROOT, load_assets=True
        )
        print(
            "validated frozen inputs: 808 clinical, 375 FTV-complete, "
            "20 LOCAL cells",
            flush=True,
        )
        frames = dict(run_experiment(inputs))
        for name, path in BASE_OUTPUT_PATHS.items():
            atomic_write_csv(frames[name], path)
        print("wrote private OOF ledgers and public transform diagnostics", flush=True)

    pcr_metrics = aggregate_pcr_predictions(frames["pcr_predictions"])
    profile_metrics = aggregate_profile_predictions(
        frames["profile_predictions"],
        expected_patient_count=808,
        required_representations=("raw", "pca16", "pca32"),
    )
    bootstrap_config = inputs.compact_config["bootstrap"]
    n_bootstrap = (
        int(args.bootstrap_replicates)
        if args.bootstrap_replicates is not None
        else int(bootstrap_config["replicates"])
    )
    comparison_summary = summarize_paired_comparisons(
        frames["pcr_predictions"],
        _comparison_specs(),
        n_bootstrap=n_bootstrap,
        confidence_level=float(bootstrap_config["confidence_level"]),
        random_seed=int(bootstrap_config["random_seed"]),
    )
    raw_regression = _raw_regression_check(pcr_metrics)
    if not raw_regression["pass"].all():
        failed = raw_regression.loc[~raw_regression["pass"]]
        raise RuntimeError(
            f"raw Goal 2 regression check failed for {len(failed)} metric cells"
        )
    tables = make_tables(
        frames=frames,
        pcr_metrics=pcr_metrics,
        profile_metrics=profile_metrics,
        point_effects=comparison_summary.point_effects,
        bootstrap_summary=comparison_summary.bootstrap_summary,
    )
    derived_frames: Mapping[str, pd.DataFrame] = {
        "pcr_metrics": pcr_metrics,
        "profile_metrics": profile_metrics,
        "paired_effects": comparison_summary.point_effects,
        "bootstrap_summary": comparison_summary.bootstrap_summary,
        "bootstrap_draws": comparison_summary.bootstrap_draws,
        "raw_regression": raw_regression,
        **tables,
    }
    for name, frame in derived_frames.items():
        atomic_write_csv(frame, DERIVED_OUTPUT_PATHS[name])

    artifact_paths = {
        **BASE_OUTPUT_PATHS,
        **{name: path for name, path in DERIVED_OUTPUT_PATHS.items() if name != "run_summary"},
    }
    run_summary = {
        "schema_version": 1,
        "experiment": "compact_mri_clinical_fusion_audit",
        "branch": str(inputs.compact_config["branch"]),
        "parent_commit": str(inputs.compact_config["parent_commit"]),
        "evidence_status": str(inputs.compact_config["evidence_status"]),
        "pca_semantics": str(
            inputs.compact_config["pca"]["output_dimension_semantics"]
        ),
        "raw_prefix_dimensions": {"T0": 192, "T1": 384, "T2": 576, "T3": 768},
        "bootstrap_replicates": n_bootstrap,
        "formal_bootstrap": n_bootstrap == int(bootstrap_config["replicates"]),
        "n_local_cells": 20,
        "n_pcr_prediction_rows": int(len(frames["pcr_predictions"])),
        "n_profile_prediction_rows": int(len(frames["profile_predictions"])),
        "n_late_inner_oof_rows": int(len(frames["late_oof"])),
        "n_bootstrap_comparison_cells": int(
            comparison_summary.bootstrap_summary[
                ["comparison_name", "population", "seed", "arm", "timing"]
            ].drop_duplicates().shape[0]
        ),
        "raw_goal2_regression_pass": bool(raw_regression["pass"].all()),
        "elapsed_seconds": float(time.time() - started),
        "summarize_only": bool(args.summarize_only),
        "artifacts": {
            name: {
                "path": str(path.relative_to(EXPERIMENT_ROOT)),
                "sha256": file_sha256(path),
                "size_bytes": int(path.stat().st_size),
            }
            for name, path in artifact_paths.items()
            if path.exists()
        },
    }
    atomic_write_json(run_summary, DERIVED_OUTPUT_PATHS["run_summary"])
    print(json.dumps(run_summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

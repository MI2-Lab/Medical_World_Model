from __future__ import annotations

from dataclasses import replace
import inspect
import json
import os
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foundation_mri.data import (  # noqa: E402
    ClinicalTable,
    FOLDS,
    HR_HER2_SUBTYPES,
    load_fold_manifest,
)
import foundation_mri.evaluation as evaluation_module  # noqa: E402
from foundation_mri.evaluation import (  # noqa: E402
    BINARY_LOGISTIC_TOL,
    ClinicalEncoder,
    LOGISTIC_MAX_ITER,
    MULTICLASS_LOGISTIC_TOL,
    aggregate_binary_predictions,
    aggregate_continuous_predictions,
    aggregate_multiclass_predictions,
    binary_metrics,
    configure_metric_free_progress,
    ensure_public_safe,
    evaluate_binary_cv,
    evaluate_multiclass_cv,
    evaluate_ridge_cv,
    metric_free_progress,
    select_logistic,
    select_multiclass_logistic,
    select_ridge,
    timing_matrix,
    write_private_csv,
    write_public_csv,
)


def _fixture(tmp_path: Path, n: int = 30) -> tuple[np.ndarray, np.ndarray, object, ClinicalTable]:
    patient_ids = np.asarray([f"P{index:03d}" for index in range(n)], dtype=str)
    labels = (np.arange(n) % 2).astype(np.int64)
    rows = []
    for fold in FOLDS:
        for index, patient_id in enumerate(patient_ids):
            split = (
                "test"
                if index % 5 == fold
                else "val"
                if (index + 1) % 5 == fold
                else "train"
            )
            rows.append(
                {
                    "patient_id": patient_id,
                    "fold": fold,
                    "split": split,
                    "label_pcr": labels[index],
                }
            )
    path = tmp_path / "folds.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    folds = load_fold_manifest(path, expected_n=n, expected_sha256=None)
    age = np.linspace(30.0, 70.0, n)
    age[4] = np.nan
    hr = labels.copy()
    her2 = ((np.arange(n) // 2) % 2).astype(np.int64)
    subtype = np.asarray(
        [
            f"HR{'+' if hr_value else '-'}/HER2{'+' if her2_value else '-'}"
            for hr_value, her2_value in zip(hr, her2, strict=True)
        ],
        dtype=str,
    )
    clinical = ClinicalTable(
        patient_ids,
        labels,
        hr,
        her2,
        ((np.arange(n) // 3) % 2).astype(np.int64),
        age,
        np.asarray([f"arm_{index % 3}" for index in range(n)], dtype=str),
        subtype,
        "0" * 64,
    )
    return patient_ids, labels, folds, clinical


def _assert_fold_progress(path: Path, task_family: str) -> None:
    records = [json.loads(line) for line in path.read_text().splitlines()]
    fold_records = [record for record in records if record["event"].startswith("fold_")]
    assert [record["event"] for record in fold_records] == [
        event
        for _ in FOLDS
        for event in (
            "fold_started",
            "fold_selection_completed",
            "fold_completed",
        )
    ]
    assert [record["fold"] for record in fold_records] == [
        fold for fold in FOLDS for _ in range(3)
    ]
    assert all(record["task_family"] == task_family for record in fold_records)
    assert all(
        set(record)
        == {
            "event",
            "task_family",
            "model",
            "spatial",
            "timing_or_endpoint",
            "fold",
            "feature_dim",
        }
        for record in fold_records
    )


def test_timing_matrix_is_exact_preregistered_1_3_6_contract() -> None:
    state = np.asarray(
        [
            [[1.0, 2.0], [4.0, 8.0], [10.0, 20.0], [99.0, 99.0]],
            [[2.0, 3.0], [5.0, 9.0], [11.0, 21.0], [98.0, 98.0]],
        ],
        dtype=np.float32,
    )
    assert np.array_equal(timing_matrix(state, "T0"), state[:, 0])
    expected_t1 = np.concatenate((state[:, 0], state[:, 1], state[:, 1] - state[:, 0]), axis=1)
    assert np.array_equal(timing_matrix(state, "T0-T1"), expected_t1)
    expected_t2 = np.concatenate(
        (
            state[:, 0],
            state[:, 1],
            state[:, 2],
            state[:, 1] - state[:, 0],
            state[:, 2] - state[:, 1],
            state[:, 2] - state[:, 0],
        ),
        axis=1,
    )
    assert np.array_equal(timing_matrix(state, "T0-T2"), expected_t2)
    assert timing_matrix(state, "T0-T2").shape[1] == 6 * state.shape[-1]
    with pytest.raises(ValueError):
        timing_matrix(state, "T0-T3")


def test_clinical_encoder_uses_train_age_and_exact_train_arm_vocabulary(tmp_path: Path) -> None:
    _, _, folds, clinical = _fixture(tmp_path)
    train = np.flatnonzero(folds.roles(0) == "train")
    encoder = ClinicalEncoder.fit(clinical, train)
    expected_age_mean = float(np.nanmean(clinical.age[train]))
    assert encoder.age_mean == pytest.approx(expected_age_mean)
    matrix = encoder.transform(clinical)
    assert matrix.shape == (30, 4 + len(encoder.arms))
    assert matrix[4, 3] == pytest.approx(expected_age_mean)
    assert np.all(matrix[:, 4:].sum(axis=1) == 1.0)

    changed = clinical.arm.copy()
    changed[0] = "unseen_arm"
    with pytest.raises(ValueError, match="absent from outer-train"):
        encoder.transform(replace(clinical, arm=changed))


def test_logistic_selector_has_no_test_interface_and_scaler_is_train_only() -> None:
    rng = np.random.default_rng(2026)
    train_x = rng.normal(loc=2.0, size=(40, 4))
    train_y = np.tile([0, 1], 20)
    validation_x = rng.normal(loc=200.0, size=(20, 4))
    validation_y = np.tile([0, 1], 10)
    signature = inspect.signature(select_logistic)
    assert not any("test" in name.lower() for name in signature.parameters)
    selected = select_logistic(
        train_x,
        train_y,
        validation_x,
        validation_y,
        penalties=("l2",),
        c_grid=(0.1, 1.0),
    )
    assert np.allclose(selected.scaler.mean_, train_x.mean(axis=0))
    assert not np.allclose(selected.scaler.mean_, validation_x.mean(axis=0))
    assert {row["C"] for row in selected.grid} == {0.1, 1.0}
    assert selected.model.solver == "liblinear"
    assert selected.model.class_weight == "balanced"
    assert selected.model.tol == BINARY_LOGISTIC_TOL == 1e-7
    assert all(row["solver"] == "liblinear" for row in selected.grid)
    assert all(row["tol"] == BINARY_LOGISTIC_TOL for row in selected.grid)
    assert all(row["max_iter"] == LOGISTIC_MAX_ITER for row in selected.grid)
    assert all(
        row["n_iter_contract"] == "integer_0_le_n_iter_lt_max_iter"
        for row in selected.grid
    )
    assert all(row["convergence_warning_observed"] is False for row in selected.grid)
    assert all(row["converged_before_max_iter"] is True for row in selected.grid)
    assert all(0 <= row["n_iter"] < row["max_iter"] for row in selected.grid)


@pytest.mark.parametrize(
    "invalid_iterations",
    (
        None,
        np.asarray([], dtype=np.int64),
        np.asarray([-1], dtype=np.int64),
        np.asarray([1.0], dtype=np.float64),
        np.asarray([np.nan], dtype=np.float64),
        np.asarray(["1"], dtype=str),
        np.asarray([LOGISTIC_MAX_ITER], dtype=np.int64),
        np.asarray([3, LOGISTIC_MAX_ITER], dtype=np.int64),
    ),
    ids=(
        "missing",
        "empty",
        "negative",
        "noninteger-dtype",
        "nan",
        "string",
        "equals-max-iter",
        "one-class-equals-max-iter",
    ),
)
def test_logistic_iteration_gate_rejects_empty_invalid_negative_or_maxed_counts(
    invalid_iterations: object,
) -> None:
    class Candidate:
        max_iter = LOGISTIC_MAX_ITER
        n_iter_ = invalid_iterations

    with pytest.raises(RuntimeError, match="n_iter_"):
        evaluation_module._validated_logistic_iterations(
            Candidate(), solver="synthetic", penalty="l2", c_value=0.1
        )

    passing = Candidate()
    passing.n_iter_ = np.asarray([0, LOGISTIC_MAX_ITER - 1], dtype=np.int64)
    assert evaluation_module._validated_logistic_iterations(
        passing, solver="synthetic", penalty="l2", c_value=0.1
    ) == (0, LOGISTIC_MAX_ITER - 1)


def test_binary_candidate_iteration_gate_precedes_validation_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(2026)
    train_x = rng.normal(size=(40, 4))
    train_y = np.tile([0, 1], 20)
    validation_x = rng.normal(size=(20, 4))
    validation_y = np.tile([0, 1], 10)
    original_fit = evaluation_module.LogisticRegression.fit
    original_predict_proba = evaluation_module.LogisticRegression.predict_proba
    validation_prediction_calls = 0

    def maxed_fit(model: object, matrix: np.ndarray, labels: np.ndarray) -> object:
        fitted = original_fit(model, matrix, labels)
        fitted.n_iter_ = np.asarray([fitted.max_iter], dtype=np.int64)
        return fitted

    def tracked_predict_proba(model: object, matrix: np.ndarray) -> np.ndarray:
        nonlocal validation_prediction_calls
        validation_prediction_calls += 1
        return original_predict_proba(model, matrix)

    monkeypatch.setattr(evaluation_module.LogisticRegression, "fit", maxed_fit)
    monkeypatch.setattr(
        evaluation_module.LogisticRegression, "predict_proba", tracked_predict_proba
    )
    with pytest.raises(RuntimeError, match="strictly before max_iter"):
        select_logistic(
            train_x,
            train_y,
            validation_x,
            validation_y,
            penalties=("l2",),
            c_grid=(0.1,),
        )
    assert validation_prediction_calls == 0


def test_multiclass_gate_rejects_a_would_be_nonselected_grid_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(2026)
    classes = ("class0", "class1", "class2", "class3")
    train_y = np.tile(classes, 12)
    validation_y = np.tile(classes, 4)
    train_index = np.tile(np.arange(len(classes)), 12)
    validation_index = np.tile(np.arange(len(classes)), 4)
    train_x = np.eye(len(classes))[train_index] + rng.normal(scale=0.05, size=(48, 4))
    validation_x = np.eye(len(classes))[validation_index] + rng.normal(
        scale=0.05, size=(16, 4)
    )
    original_fit = evaluation_module.LogisticRegression.fit
    original_predict_proba = evaluation_module.LogisticRegression.predict_proba
    smaller_c_probabilities: list[np.ndarray] = []
    maxed_candidate_fit_count = 0
    maxed_candidate_prediction_calls = 0

    def selectively_maxed_fit(
        model: object, matrix: np.ndarray, labels: np.ndarray
    ) -> object:
        nonlocal maxed_candidate_fit_count
        fitted = original_fit(model, matrix, labels)
        if float(fitted.C) == pytest.approx(1.0):
            maxed_candidate_fit_count += 1
            if maxed_candidate_fit_count == 2:
                fitted.n_iter_ = np.asarray([fitted.max_iter], dtype=np.int64)
        return fitted

    def tied_predict_proba(model: object, matrix: np.ndarray) -> np.ndarray:
        nonlocal maxed_candidate_prediction_calls
        if float(model.C) == pytest.approx(0.1):
            probability = original_predict_proba(model, matrix)
            smaller_c_probabilities.append(probability.copy())
            return probability
        maxed_candidate_prediction_calls += 1
        assert len(smaller_c_probabilities) == len(classes)
        # Identical validation probabilities make C=1.0 lose the smaller-C tie-break.
        return smaller_c_probabilities[maxed_candidate_prediction_calls - 1].copy()

    monkeypatch.setattr(
        evaluation_module.LogisticRegression, "fit", selectively_maxed_fit
    )
    monkeypatch.setattr(
        evaluation_module.LogisticRegression, "predict_proba", tied_predict_proba
    )
    with pytest.raises(
        RuntimeError,
        match=r"C=1 estimator_class=class1.*strictly before max_iter",
    ):
        select_multiclass_logistic(
            train_x,
            train_y,
            validation_x,
            validation_y,
            classes=classes,
            penalties=("l2",),
            c_grid=(0.1, 1.0),
        )
    assert len(smaller_c_probabilities) == len(classes)
    assert maxed_candidate_fit_count == 2
    assert maxed_candidate_prediction_calls == 0


def test_explicit_ovr_selector_has_balanced_liblinear_estimators_and_simplex_probabilities(
) -> None:
    rng = np.random.default_rng(2026)
    classes = ("class0", "class1", "class2", "class3")
    train_y = np.tile(classes, 12)
    validation_y = np.tile(classes, 4)
    train_index = np.tile(np.arange(len(classes)), 12)
    validation_index = np.tile(np.arange(len(classes)), 4)
    train_x = np.eye(len(classes))[train_index] + rng.normal(scale=0.05, size=(48, 4))
    validation_x = np.eye(len(classes))[validation_index] + rng.normal(
        scale=0.05, size=(16, 4)
    )

    selected = select_multiclass_logistic(
        train_x,
        train_y,
        validation_x,
        validation_y,
        classes=classes,
        penalties=("l2",),
        c_grid=(0.1,),
    )
    assert np.allclose(selected.scaler.mean_, train_x.mean(axis=0))
    assert tuple(selected.model.classes_) == tuple(sorted(classes))
    assert len(selected.model.estimators_) == len(classes)
    assert selected.model.coef_.shape == (len(classes), train_x.shape[1])
    assert selected.model.intercept_.shape == (len(classes),)
    assert all(estimator.solver == "liblinear" for estimator in selected.model.estimators_)
    assert all(
        estimator.class_weight == "balanced" for estimator in selected.model.estimators_
    )
    assert all(
        estimator.tol == MULTICLASS_LOGISTIC_TOL
        for estimator in selected.model.estimators_
    )
    assert all(
        estimator.max_iter == LOGISTIC_MAX_ITER
        for estimator in selected.model.estimators_
    )
    probability = selected.model.predict_proba(
        selected.scaler.transform(validation_x)
    )
    assert probability.shape == (len(validation_x), len(classes))
    assert np.isfinite(probability).all()
    assert np.all(probability >= 0.0)
    assert np.all(probability <= 1.0)
    assert np.allclose(probability.sum(axis=1), 1.0, atol=1e-12, rtol=1e-12)
    grid_row = selected.grid[0]
    assert grid_row["solver"] == "explicit_one_vs_rest_liblinear"
    assert grid_row["underlying_solver"] == "liblinear"
    assert grid_row["estimator_count"] == len(classes)
    iterations = json.loads(grid_row["n_iter_by_class_json"])
    assert set(iterations) == set(classes)
    assert all(
        0 <= value < LOGISTIC_MAX_ITER
        for values in iterations.values()
        for value in values
    )

    repeated = select_multiclass_logistic(
        train_x,
        train_y,
        validation_x,
        validation_y,
        classes=classes,
        penalties=("l2",),
        c_grid=(0.1,),
    )
    repeated_probability = repeated.model.predict_proba(
        repeated.scaler.transform(validation_x)
    )
    assert np.array_equal(selected.model.coef_, repeated.model.coef_)
    assert np.array_equal(selected.model.intercept_, repeated.model.intercept_)
    assert np.array_equal(probability, repeated_probability)


def test_explicit_ovr_accepts_zero_iteration_all_zero_l1_candidate() -> None:
    rng = np.random.default_rng(2026)
    classes = ("class0", "class1", "class2", "class3")
    train_y = np.tile(classes, 12)
    validation_y = np.tile(classes, 4)
    train_index = np.tile(np.arange(len(classes)), 12)
    validation_index = np.tile(np.arange(len(classes)), 4)
    train_x = np.eye(len(classes))[train_index] + rng.normal(scale=0.05, size=(48, 4))
    validation_x = np.eye(len(classes))[validation_index] + rng.normal(
        scale=0.05, size=(16, 4)
    )
    selected = select_multiclass_logistic(
        train_x,
        train_y,
        validation_x,
        validation_y,
        classes=classes,
        penalties=("l1",),
        c_grid=(0.001,),
    )
    assert selected.grid[0]["n_iter"] == 0
    assert all(
        value == 0
        for values in json.loads(selected.grid[0]["n_iter_by_class_json"]).values()
        for value in values
    )
    assert np.count_nonzero(selected.model.coef_) == 0
    assert np.count_nonzero(selected.model.intercept_) == 0
    probability = selected.model.predict_proba(
        selected.scaler.transform(validation_x)
    )
    assert np.array_equal(
        probability,
        np.full((len(validation_x), len(classes)), 0.25, dtype=np.float64),
    )


def test_binary_convergence_warning_fails_before_validation_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(2026)
    train_x = rng.normal(size=(40, 4))
    train_y = np.tile([0, 1], 20)
    validation_x = rng.normal(size=(20, 4))
    validation_y = np.tile([0, 1], 10)
    original_fit = evaluation_module.LogisticRegression.fit
    original_predict_proba = evaluation_module.LogisticRegression.predict_proba
    validation_prediction_calls = 0

    def warning_fit(model: object, matrix: np.ndarray, labels: np.ndarray) -> object:
        fitted = original_fit(model, matrix, labels)
        warnings.warn("forced convergence warning", evaluation_module.ConvergenceWarning)
        return fitted

    def tracked_predict_proba(model: object, matrix: np.ndarray) -> np.ndarray:
        nonlocal validation_prediction_calls
        validation_prediction_calls += 1
        return original_predict_proba(model, matrix)

    monkeypatch.setattr(evaluation_module.LogisticRegression, "fit", warning_fit)
    monkeypatch.setattr(
        evaluation_module.LogisticRegression, "predict_proba", tracked_predict_proba
    )
    with pytest.raises(RuntimeError, match="emitted ConvergenceWarning"):
        select_logistic(
            train_x,
            train_y,
            validation_x,
            validation_y,
            penalties=("l2",),
            c_grid=(0.1,),
        )
    assert validation_prediction_calls == 0


def test_explicit_ovr_underlying_warning_fails_before_candidate_probability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(2026)
    classes = ("class0", "class1", "class2", "class3")
    train_y = np.tile(classes, 12)
    validation_y = np.tile(classes, 4)
    train_index = np.tile(np.arange(len(classes)), 12)
    validation_index = np.tile(np.arange(len(classes)), 4)
    train_x = np.eye(len(classes))[train_index] + rng.normal(scale=0.05, size=(48, 4))
    validation_x = np.eye(len(classes))[validation_index] + rng.normal(
        scale=0.05, size=(16, 4)
    )
    original_fit = evaluation_module.LogisticRegression.fit
    original_predict_proba = evaluation_module.LogisticRegression.predict_proba
    fit_calls = 0
    validation_prediction_calls = 0

    def second_estimator_warns(
        model: object, matrix: np.ndarray, labels: np.ndarray
    ) -> object:
        nonlocal fit_calls
        fit_calls += 1
        fitted = original_fit(model, matrix, labels)
        if fit_calls == 2:
            warnings.warn("forced convergence warning", evaluation_module.ConvergenceWarning)
        return fitted

    def tracked_predict_proba(model: object, matrix: np.ndarray) -> np.ndarray:
        nonlocal validation_prediction_calls
        validation_prediction_calls += 1
        return original_predict_proba(model, matrix)

    monkeypatch.setattr(
        evaluation_module.LogisticRegression, "fit", second_estimator_warns
    )
    monkeypatch.setattr(
        evaluation_module.LogisticRegression, "predict_proba", tracked_predict_proba
    )
    with pytest.raises(
        RuntimeError,
        match=r"estimator_class=class1 emitted ConvergenceWarning",
    ):
        select_multiclass_logistic(
            train_x,
            train_y,
            validation_x,
            validation_y,
            classes=classes,
            penalties=("l2",),
            c_grid=(0.1,),
        )
    assert fit_calls == 2
    assert validation_prediction_calls == 0


def test_metric_free_progress_is_private_exclusive_and_rejects_unsafe_fields(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    progress_path = tmp_path / "selection-progress.jsonl"
    assert configure_metric_free_progress(progress_path) == progress_path
    try:
        metric_free_progress(
            "candidate_started",
            family="binary",
            solver="liblinear",
            penalty="l2",
            C=0.1,
            max_iter=LOGISTIC_MAX_ITER,
            tol=BINARY_LOGISTIC_TOL,
        )
        for forbidden in (
            "metric_value",
            "auc",
            "score",
            "probability",
            "patient_count",
            "target",
            "y_true",
            "y_pred",
        ):
            with pytest.raises(ValueError, match="forbidden"):
                metric_free_progress("blocked", **{forbidden: 1})
        with pytest.raises(TypeError, match="JSON scalar"):
            metric_free_progress("blocked", nested={"safe": 1})
    finally:
        configure_metric_free_progress(None)

    assert os.stat(progress_path).st_mode & 0o777 == 0o600
    records = [json.loads(line) for line in progress_path.read_text().splitlines()]
    assert records == [
        {
            "C": 0.1,
            "event": "candidate_started",
            "family": "binary",
            "max_iter": LOGISTIC_MAX_ITER,
            "penalty": "l2",
            "solver": "liblinear",
            "tol": BINARY_LOGISTIC_TOL,
        }
    ]
    assert json.loads(capsys.readouterr().err.strip()) == records[0]
    with pytest.raises(FileExistsError):
        configure_metric_free_progress(progress_path)


def test_selector_emits_metric_free_candidate_start_and_completion(tmp_path: Path) -> None:
    rng = np.random.default_rng(2026)
    train_x = rng.normal(size=(40, 4))
    train_y = np.tile([0, 1], 20)
    validation_x = rng.normal(size=(20, 4))
    validation_y = np.tile([0, 1], 10)
    progress_path = tmp_path / "candidate-progress.jsonl"
    configure_metric_free_progress(progress_path)
    try:
        select_logistic(
            train_x,
            train_y,
            validation_x,
            validation_y,
            penalties=("l2",),
            c_grid=(0.1,),
        )
    finally:
        configure_metric_free_progress(None)
    records = [json.loads(line) for line in progress_path.read_text().splitlines()]
    assert [record["event"] for record in records] == [
        "candidate_started",
        "candidate_completed",
    ]
    assert records[0]["tol"] == BINARY_LOGISTIC_TOL
    assert records[0]["max_iter"] == LOGISTIC_MAX_ITER
    assert 0 <= records[1]["n_iter"] < LOGISTIC_MAX_ITER
    assert records[1]["elapsed_seconds"] >= 0.0
    forbidden = evaluation_module._FORBIDDEN_PROGRESS_KEY_FRAGMENTS
    assert all(
        not any(fragment in key.lower() for fragment in forbidden)
        for record in records
        for key in record
    )


def test_binary_cv_is_one_oof_score_per_patient_and_public_metrics_have_no_ids(
    tmp_path: Path,
) -> None:
    patient_ids, labels, folds, _ = _fixture(tmp_path)
    rng = np.random.default_rng(2026)
    matrix = np.column_stack(
        (
            labels + rng.normal(scale=0.2, size=len(labels)),
            rng.normal(size=len(labels)),
            rng.normal(size=len(labels)),
        )
    )
    progress_path = tmp_path / "binary-fold-progress.jsonl"
    configure_metric_free_progress(progress_path)
    try:
        result = evaluate_binary_cv(
            patient_ids=patient_ids,
            targets=labels,
            fold_manifest=folds,
            matrices=matrix,
            target_name="pCR",
            model_name="synthetic_mri",
            spatial="LOCAL",
            timing="T0",
            analysis_population="full_30",
            penalties=("l2",),
            c_grid=(0.1,),
            require_manifest_pcr_match=True,
        )
    finally:
        configure_metric_free_progress(None)
    _assert_fold_progress(progress_path, "binary")
    assert len(result.predictions) == len(patient_ids)
    assert result.predictions["patient_id"].nunique() == len(patient_ids)
    assert set(result.predictions["split"]) == {"test"}
    assert result.predictions["test_predict_proba_call_count"].eq(1).all()
    assert result.selections["test_used_for_scaler"].eq(False).all()
    assert result.selections["test_used_for_hyperparameter_selection"].eq(False).all()
    assert not {
        "scaler_mean_json",
        "scaler_scale_json",
        "coef_json",
        "intercept_json",
    }.intersection(result.selections.columns)
    assert result.selections["scaler_mean_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert result.selections["coef_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert result.selections["coef_shape_json"].eq("[1,3]").all()
    assert result.selections["tol"].eq(BINARY_LOGISTIC_TOL).all()
    assert result.selections["max_iter"].eq(LOGISTIC_MAX_ITER).all()
    assert result.selections["n_iter_contract"].eq(
        "integer_0_le_n_iter_lt_max_iter"
    ).all()
    assert result.selections["convergence_warning_observed"].eq(False).all()
    assert result.selections[
        "all_grid_candidates_converged_before_max_iter"
    ].eq(True).all()
    assert (result.selections["grid_max_n_iter"] < LOGISTIC_MAX_ITER).all()
    for fold, row in result.selections.set_index("fold").iterrows():
        assert row["scaler_train_rows"] == int(np.sum(folds.roles(int(fold)) == "train"))

    metrics = aggregate_binary_predictions(result.predictions)
    assert set(metrics["aggregation"]) == {"pooled_oof", "outer_fold_macro"}
    assert {"auroc", "auprc", "brier", "calibration_slope", "calibration_intercept", "ece_10bin"}.issubset(
        metrics.columns
    )
    assert "patient_id" not in metrics.columns
    ensure_public_safe(metrics)
    direct = binary_metrics(labels, np.clip(0.15 + 0.7 * labels, 0, 1))
    assert direct["auroc"] == pytest.approx(1.0)
    assert direct["auprc"] == pytest.approx(1.0)
    assert direct["ece_10bin"] == pytest.approx(0.15)


def test_reliable_subtype_probe_is_explicit_ovr_validation_only_and_test_single_use(
    tmp_path: Path,
) -> None:
    patient_ids, _, folds, clinical = _fixture(tmp_path, n=40)
    rng = np.random.default_rng(2026)
    class_index = np.asarray(
        [HR_HER2_SUBTYPES.index(value) for value in clinical.subtype], dtype=np.float64
    )
    matrix = np.column_stack(
        (class_index + rng.normal(scale=0.1, size=len(class_index)), rng.normal(size=(len(class_index), 3)))
    )
    assert not any(
        "test" in name.lower()
        for name in inspect.signature(select_multiclass_logistic).parameters
    )
    progress_path = tmp_path / "multiclass-fold-progress.jsonl"
    configure_metric_free_progress(progress_path)
    try:
        result = evaluate_multiclass_cv(
            patient_ids=patient_ids,
            targets=clinical.subtype,
            classes=HR_HER2_SUBTYPES,
            fold_manifest=folds,
            matrices=matrix,
            target_name="HR_HER2_subtype",
            model_name="synthetic_encoder",
            spatial="LOCAL",
            timing="T0",
            analysis_population="full_40",
            penalties=("l2",),
            c_grid=(0.1,),
        )
    finally:
        configure_metric_free_progress(None)
    _assert_fold_progress(progress_path, "multiclass")
    progress_records = [
        json.loads(line) for line in progress_path.read_text().splitlines()
    ]
    estimator_started = [
        row for row in progress_records if row["event"] == "estimator_started"
    ]
    estimator_completed = [
        row for row in progress_records if row["event"] == "estimator_completed"
    ]
    assert len(estimator_started) == len(FOLDS) * len(HR_HER2_SUBTYPES)
    assert len(estimator_completed) == len(estimator_started)
    assert all(row["solver"] == "liblinear" for row in estimator_started)
    assert all(0 <= row["n_iter"] < LOGISTIC_MAX_ITER for row in estimator_completed)
    assert len(result.predictions) == len(patient_ids)
    assert result.predictions["test_predict_proba_call_count"].eq(1).all()
    assert result.selections["solver"].eq("explicit_one_vs_rest_liblinear").all()
    assert result.selections["underlying_solver"].eq("liblinear").all()
    assert result.selections["estimator_count_per_candidate"].eq(4).all()
    assert result.selections["tol"].eq(MULTICLASS_LOGISTIC_TOL).all()
    assert result.selections["max_iter"].eq(LOGISTIC_MAX_ITER).all()
    assert result.selections["n_iter_contract"].eq(
        "integer_0_le_n_iter_lt_max_iter"
    ).all()
    assert result.selections["convergence_warning_observed"].eq(False).all()
    assert result.selections[
        "all_grid_candidates_converged_before_max_iter"
    ].eq(True).all()
    assert (result.selections["grid_max_n_iter"] < LOGISTIC_MAX_ITER).all()
    for payload in result.selections["grid_validation_metrics_json"]:
        grid = json.loads(payload)
        assert all(row["solver"] == "explicit_one_vs_rest_liblinear" for row in grid)
        assert all(row["underlying_solver"] == "liblinear" for row in grid)
        assert all(row["estimator_count"] == 4 for row in grid)
        assert all(row["tol"] == MULTICLASS_LOGISTIC_TOL for row in grid)
        assert all(row["max_iter"] == LOGISTIC_MAX_ITER for row in grid)
        assert all(
            row["n_iter_contract"] == "integer_0_le_n_iter_lt_max_iter"
            for row in grid
        )
        assert all(row["convergence_warning_observed"] is False for row in grid)
        assert all(row["converged_before_max_iter"] is True for row in grid)
        assert all(0 <= row["n_iter"] < row["max_iter"] for row in grid)
    assert result.selections["test_used_for_hyperparameter_selection"].eq(False).all()
    assert not {"scaler_mean_json", "scaler_scale_json", "coef_json"}.intersection(
        result.selections.columns
    )
    assert result.selections["coef_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    metrics = aggregate_multiclass_predictions(result.predictions)
    assert {
        "macro_ovr_auroc",
        "macro_ovr_auprc",
        "multiclass_brier",
        "toplabel_ece_10bin",
    }.issubset(metrics.columns)
    ensure_public_safe(metrics)


def test_ridge_selection_and_cv_keep_test_out_of_alpha_and_scalers(tmp_path: Path) -> None:
    patient_ids, _, folds, _ = _fixture(tmp_path)
    target = np.linspace(1.0, 20.0, len(patient_ids))
    matrix = np.column_stack((target, target**0.5, np.sin(target)))
    assert not any("test" in name.lower() for name in inspect.signature(select_ridge).parameters)
    progress_path = tmp_path / "ridge-fold-progress.jsonl"
    configure_metric_free_progress(progress_path)
    try:
        result = evaluate_ridge_cv(
            patient_ids=patient_ids,
            targets=target,
            fold_manifest=folds,
            matrices=matrix,
            target_name="FTV",
            model_name="synthetic_encoder",
            spatial="GLOBAL",
            task="static",
            endpoint="T0",
            analysis_population="radiomics_complete_case_30",
            alphas=(0.01, 1.0),
            target_transform="log1p",
        )
    finally:
        configure_metric_free_progress(None)
    _assert_fold_progress(progress_path, "ridge")
    assert len(result.predictions) == len(patient_ids)
    assert result.predictions["test_predict_call_count"].eq(1).all()
    assert result.selections["test_used_for_scaler"].eq(False).all()
    assert result.selections["test_used_for_alpha_selection"].eq(False).all()
    assert result.selections["test_used_for_prediction_bounds"].eq(False).all()
    assert result.selections["prediction_bound_policy"].eq(
        "outer_train_transformed_min_max"
    ).all()
    assert result.selections["solver"].eq("lsqr").all()
    assert result.selections["n_iter_contract"].eq(
        "integer_0_le_n_iter_lt_max_iter"
    ).all()
    assert result.selections["all_grid_candidates_converged_before_max_iter"].eq(
        True
    ).all()
    assert not {
        "x_scaler_mean_json",
        "x_scaler_scale_json",
        "coef_json",
    }.intersection(result.selections.columns)
    assert result.selections["x_scaler_mean_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert result.selections["coef_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    metrics = aggregate_continuous_predictions(result.predictions)
    assert set(metrics["aggregation"]) == {"pooled_oof", "outer_fold_macro"}
    assert {"r2", "spearman", "rmse", "mae"}.issubset(metrics.columns)
    ensure_public_safe(metrics)


def test_metric_free_stderr_mirror_is_best_effort_but_jsonl_write_is_hard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BrokenMirror:
        def write(self, _value: str) -> None:
            raise BrokenPipeError("detached synthetic stderr")

        def flush(self) -> None:
            raise AssertionError("flush must not follow a failed write")

    mirror_path = tmp_path / "mirror.private.jsonl"
    configure_metric_free_progress(mirror_path)
    try:
        monkeypatch.setattr(evaluation_module.sys, "stderr", BrokenMirror())
        metric_free_progress("synthetic_event", fold=0)
    finally:
        configure_metric_free_progress(None)
    assert json.loads(mirror_path.read_text()) == {
        "event": "synthetic_event",
        "fold": 0,
    }

    hard_path = tmp_path / "hard.private.jsonl"
    configure_metric_free_progress(hard_path)
    try:
        monkeypatch.setattr(
            evaluation_module.os,
            "write",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("synthetic fd failure")),
        )
        with pytest.raises(OSError, match="synthetic fd failure"):
            metric_free_progress("synthetic_event", fold=1)
    finally:
        configure_metric_free_progress(None)


def test_ridge_iteration_parameter_and_warning_gates_precede_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FittedState:
        max_iter = 10
        n_iter_ = np.asarray([0], dtype=np.int64)
        coef_ = np.asarray([0.0])
        intercept_ = np.asarray(0.0)

    assert evaluation_module._validated_ridge_iterations(
        FittedState(), alpha=0.1
    ) == (0,)
    for invalid, pattern in (
        (np.asarray([], dtype=np.int64), "empty or invalid"),
        (np.asarray([1.5]), "empty or invalid"),
        (np.asarray([-1]), "negative"),
        (np.asarray([10]), "did not converge"),
    ):
        state = FittedState()
        state.n_iter_ = invalid
        with pytest.raises(RuntimeError, match=pattern):
            evaluation_module._validated_ridge_iterations(state, alpha=0.1)
    state = FittedState()
    state.coef_ = np.asarray([np.inf])
    state.fit = lambda *_args, **_kwargs: state
    with pytest.raises(RuntimeError, match="non-finite fitted coef_"):
        evaluation_module._fit_validated_ridge(
            state, np.ones((2, 1)), np.zeros(2), alpha=0.1
        )

    rng = np.random.default_rng(2026)
    train_x = rng.normal(size=(30, 3))
    train_y = rng.uniform(0.0, 5.0, size=30)
    validation_x = rng.normal(size=(10, 3))
    validation_y = rng.uniform(0.0, 5.0, size=10)
    original_fit = evaluation_module.Ridge.fit
    original_predict = evaluation_module.Ridge.predict
    fit_calls = 0
    validation_prediction_calls = 0

    def second_candidate_warns(model: object, matrix: np.ndarray, targets: np.ndarray):
        nonlocal fit_calls
        fit_calls += 1
        fitted = original_fit(model, matrix, targets)
        if fit_calls == 2:
            warnings.warn("forced Ridge warning", evaluation_module.ConvergenceWarning)
        return fitted

    def tracked_predict(model: object, matrix: np.ndarray):
        nonlocal validation_prediction_calls
        validation_prediction_calls += 1
        return original_predict(model, matrix)

    monkeypatch.setattr(evaluation_module.Ridge, "fit", second_candidate_warns)
    monkeypatch.setattr(evaluation_module.Ridge, "predict", tracked_predict)
    with pytest.raises(RuntimeError, match="emitted ConvergenceWarning"):
        select_ridge(
            train_x,
            train_y,
            validation_x,
            validation_y,
            alphas=(0.01, 1.0),
            target_transform="log1p",
        )
    assert fit_calls == 2
    assert validation_prediction_calls == 1


def test_ridge_log1p_bounds_prevent_test_only_overflow_and_are_train_only(
    tmp_path: Path,
) -> None:
    patient_ids, _, folds, _ = _fixture(tmp_path)
    target = np.exp(np.linspace(0.0, 3.0, len(patient_ids))) - 1.0
    base = np.column_stack((np.log1p(target), np.sin(np.arange(len(target)))))
    matrices = {fold: base.copy() for fold in FOLDS}
    fold = 3
    roles = folds.roles(fold, patient_ids)
    matrices[fold][roles == "test", 0] = 1e6
    result = evaluate_ridge_cv(
        patient_ids=patient_ids,
        targets=target,
        fold_manifest=folds,
        matrices=matrices,
        target_name="FTV",
        model_name="synthetic_encoder",
        spatial="LOCAL",
        task="static",
        endpoint="T0",
        analysis_population="radiomics_complete_case_30",
        alphas=(0.01,),
        target_transform="log1p",
    )
    fold_predictions = result.predictions.loc[result.predictions["fold"].eq(fold)]
    train_max = float(np.max(target[roles == "train"]))
    assert np.isfinite(fold_predictions["y_pred"]).all()
    assert fold_predictions["prediction_clipped_to_outer_train_bounds"].all()
    assert np.allclose(fold_predictions["y_pred"], train_max)
    selection = result.selections.loc[result.selections["fold"].eq(fold)].iloc[0]
    assert selection["test_predictions_clipped"] == len(fold_predictions)
    assert selection["test_clip_rate"] == 1.0
    assert selection["test_used_for_prediction_bounds"] is False or not bool(
        selection["test_used_for_prediction_bounds"]
    )
    with pytest.raises(FloatingPointError):
        evaluation_module._target_inverse(np.asarray([1e6]), "log1p")
    with pytest.raises(FloatingPointError, match="pre-clipping"):
        evaluation_module._bound_transformed_ridge_predictions(
            np.asarray([np.inf]),
            target_transform="log1p",
            train_transformed_min=0.0,
            train_transformed_max=1.0,
        )


def test_ridge_progress_does_not_claim_fold_complete_after_selection_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patient_ids, _, folds, _ = _fixture(tmp_path)
    target = np.linspace(1.0, 10.0, len(patient_ids))
    matrix = np.column_stack((np.log1p(target), np.sin(target)))
    progress = tmp_path / "ridge-failure.private.jsonl"
    monkeypatch.setattr(
        evaluation_module,
        "_target_inverse",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            FloatingPointError("synthetic post-selection failure")
        ),
    )
    configure_metric_free_progress(progress)
    try:
        with pytest.raises(FloatingPointError, match="synthetic post-selection"):
            evaluate_ridge_cv(
                patient_ids=patient_ids,
                targets=target,
                fold_manifest=folds,
                matrices=matrix,
                target_name="FTV",
                model_name="synthetic_encoder",
                spatial="LOCAL",
                task="static",
                endpoint="T0",
                analysis_population="radiomics_complete_case_30",
                alphas=(0.1,),
                target_transform="log1p",
            )
    finally:
        configure_metric_free_progress(None)
    events = [
        row["event"]
        for row in map(json.loads, progress.read_text().splitlines())
        if row["event"].startswith("fold_")
    ]
    assert events == ["fold_started", "fold_selection_completed"]


def test_private_and_public_writers_enforce_filename_privacy_and_permissions(
    tmp_path: Path,
) -> None:
    private = pd.DataFrame({"patient_id": ["P001"], "y_score": [0.5]})
    private_path = tmp_path / "predictions.private.csv"
    write_private_csv(private, private_path)
    assert private_path.is_file()
    assert os.stat(private_path).st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match="must end"):
        write_private_csv(private, tmp_path / "predictions.csv")
    with pytest.raises(ValueError, match="identifier/path"):
        write_public_csv(private, tmp_path / "metrics.csv")
    with pytest.raises(ValueError, match="absolute path"):
        ensure_public_safe(pd.DataFrame({"model": ["/secret/model"]}))

    public = pd.DataFrame({"model": ["safe_model"], "auroc": [0.7]})
    public_path = tmp_path / "metrics.csv"
    write_public_csv(public, public_path)
    assert os.stat(public_path).st_mode & 0o777 == 0o644

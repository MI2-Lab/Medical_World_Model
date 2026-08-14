from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import sys
import warnings

import numpy as np
import pytest
from scipy.special import expit, softmax
from sklearn.exceptions import ConvergenceWarning


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_audit as audit  # noqa: E402


def _array_sha256(values: np.ndarray, dtype: str) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _legacy_golden_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(37)
    features = rng.normal(size=(73, 9))
    labels = np.asarray(["A", "B", "C", "D"] * 18 + ["A"])
    permutation = rng.permutation(len(labels))
    features = features[permutation]
    labels = labels[permutation]
    test_features = rng.normal(size=(17, 9))
    return features, labels, test_features


def _separable_four_class_data() -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    centers = np.asarray([[-4.0, -4.0], [4.0, -4.0], [-4.0, 4.0], [4.0, 4.0]])
    classes = np.asarray(audit.SUBTYPE_CLASSES)
    train = np.vstack(
        [center + offset for center in centers for offset in (-0.2, 0.0, 0.2)]
    )
    train_labels = np.repeat(classes, 3)
    validation = np.vstack(
        [center + offset for center in centers for offset in (-0.1, 0.1)]
    )
    validation_labels = np.repeat(classes, 2)
    test = centers.copy()
    test_labels = classes.copy()
    return train, train_labels, validation, validation_labels, test, test_labels


def test_exact_legacy_adapter_matches_fixed_sklearn_172_golden() -> None:
    features, labels, test_features = _legacy_golden_fixture()
    model = audit._ExactLegacyMulticlassLiblinear(
        C=0.1,
        penalty="l2",
        solver="liblinear",
        class_weight="balanced",
        max_iter=10_000,
        random_state=0,
    ).fit(features, labels)
    probability = model.predict_proba(test_features)

    assert model.classes_.tolist() == ["A", "B", "C", "D"]
    assert model.coef_.shape == (4, 9)
    assert model.intercept_.shape == (4,)
    assert model.n_iter_.shape == (4,)
    assert probability.shape == (17, 4)
    assert (
        _array_sha256(model.coef_, "<f8")
        == "35c8a12264c8ff4dde552525a3ff02aac18b6df444cb014889ffc5d740773034"
    )
    assert (
        _array_sha256(model.intercept_, "<f8")
        == "5d868e12fdaa9cd3e059240426737f2574c34ff043bdbb4d24e8e8de9a6d147c"
    )
    assert (
        _array_sha256(model.n_iter_, "<i8")
        == "38df89b71e6aa2b361da02c1960692788be8f78764a50866b24f7c7710cb0dbe"
    )
    assert (
        _array_sha256(probability, "<f8")
        == "a4a86b43a109c52873fb8c0408cd1ef51adcb8e35887e25da1c2f95ec99bcf08"
    )

    decision = model.decision_function(test_features)
    expected = expit(decision)
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_array_equal(probability, expected)
    np.testing.assert_allclose(probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-15)
    assert not np.allclose(probability, softmax(decision, axis=1), rtol=0.0, atol=1e-6)


def test_exact_legacy_fit_keeps_train_scaler_contract_and_smaller_c_tie() -> None:
    train, train_y, validation, validation_y, _test, _test_y = (
        _separable_four_class_data()
    )
    fit = audit._fit_multiclass_logistic_exact_legacy(
        train,
        train_y,
        validation,
        validation_y,
        c_grid=[100.0, 0.01, 1.0],
        solver="liblinear",
        max_iter=10_000,
        random_state=0,
    )

    np.testing.assert_allclose(fit.scaler.mean_, train.mean(axis=0))
    assert fit.selected_c == pytest.approx(0.01)
    assert fit.validation_macro_ovr_auroc == pytest.approx(1.0)
    assert fit.model.solver == "liblinear"
    assert fit.model.penalty == "l2"
    assert fit.model.class_weight == "balanced"
    assert fit.model.max_iter == 10_000
    assert fit.model.random_state == 0
    assert fit.model.coef_.shape == (4, 2)
    assert fit.model.intercept_.shape == (4,)
    assert fit.model.n_iter_.shape == (4,)
    assert fit.classes == tuple(sorted(audit.SUBTYPE_CLASSES))
    probability = fit.predict_proba(validation)
    assert probability.shape == (len(validation), 4)
    np.testing.assert_allclose(probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-15)

    with pytest.raises(ValueError, match="must remain liblinear"):
        audit._fit_multiclass_logistic_exact_legacy(
            train,
            train_y,
            validation,
            validation_y,
            c_grid=[0.1],
            solver="lbfgs",
        )


@pytest.mark.parametrize(
    "changed",
    [
        {"l1_ratio": 0.5},
        {"n_jobs": 2},
        {"max_iter": 9_999},
        {"random_state": 1},
    ],
)
def test_exact_legacy_estimator_rejects_constructor_contract_drift(
    changed: dict,
) -> None:
    features, labels, _test_features = _legacy_golden_fixture()
    parameters = {
        "C": 0.1,
        "penalty": "l2",
        "solver": "liblinear",
        "class_weight": "balanced",
        "max_iter": 10_000,
        "random_state": 0,
    }
    parameters.update(changed)
    model = audit._ExactLegacyMulticlassLiblinear(**parameters)
    with pytest.raises(ValueError, match="contract drifted"):
        model.fit(features, labels)


def test_subtype_append_uses_exact_adapter_without_mutating_config() -> None:
    train, train_y, validation, validation_y, test, test_y = (
        _separable_four_class_data()
    )
    matrix = np.vstack((train, validation, test))
    labels = np.concatenate((train_y, validation_y, test_y))
    indices = {
        "train": np.arange(len(train)),
        "val": np.arange(len(train), len(train) + len(validation)),
        "test": np.arange(len(train) + len(validation), len(matrix)),
    }
    config = {"logistic": {"solver": "liblinear", "max_iter": 10_000}}
    config_before = copy.deepcopy(config)
    predictions: list[dict] = []
    hyperparameters: list[dict] = []
    audit._append_multiclass_fit(
        predictions,
        hyperparameters,
        patient_ids=np.asarray([f"patient-{index}" for index in range(len(matrix))]),
        fold=0,
        labels=labels,
        matrix=matrix,
        indices=indices,
        config=config,
        grid=[0.1],
        metadata={
            "analysis": "phenotype",
            "population": "full_808",
            "seed": 2026,
            "arm": "LOCAL3",
            "view": "T0",
            "target": "subtype_4class",
            "variant": "P1",
            "clinical_contract": "",
        },
    )

    assert config == config_before
    assert len(predictions) == 4
    assert len(hyperparameters) == 1
    assert hyperparameters[0]["class_weight"] == "balanced"
    assert hyperparameters[0]["c_grid"] == "[0.1]"
    for row in predictions:
        probability = np.asarray(
            [row[column] for column in audit.SUBTYPE_PROBABILITY_COLUMNS]
        )
        assert np.isfinite(probability).all()
        assert probability.sum() == pytest.approx(1.0, abs=1e-15)
        assert tuple(row[column] for column in audit.SUBTYPE_LABEL_COLUMNS) == tuple(
            audit.SUBTYPE_CLASSES
        )


def test_binary_penalty_warning_filter_is_narrow_and_preserves_estimator() -> None:
    matrix = np.asarray(
        [[-3.0, -1.0], [-2.0, -0.5], [-1.0, -0.2], [1.0, 0.2], [2.0, 0.5], [3.0, 1.0]]
    )
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    indices = {
        "train": np.asarray([0, 1, 3, 4]),
        "val": np.asarray([2, 5]),
        "test": np.asarray([], dtype=np.int64),
    }
    config = {"logistic": {"solver": "liblinear", "max_iter": 10_000}}
    config_before = copy.deepcopy(config)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = audit._fit_binary(
            matrix,
            labels,
            indices,
            config,
            grid=[0.1],
            class_weight="balanced",
        )

    assert config == config_before
    assert fit.model.solver == "liblinear"
    assert fit.model.penalty == "l2"
    assert fit.model.class_weight == "balanced"
    assert not any(
        "'penalty' was deprecated in version 1.8" in str(item.message)
        for item in caught
    )


def test_binary_penalty_warning_filter_does_not_hide_other_future_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()

    def fake_fit(*_args, **_kwargs):
        warnings.warn("unrelated future warning", FutureWarning)
        return sentinel

    monkeypatch.setattr(audit, "fit_binary_logistic", fake_fit)
    with pytest.warns(FutureWarning, match="unrelated future warning"):
        observed = audit._fit_binary(
            np.ones((3, 1)),
            np.asarray([0, 1, 0]),
            {
                "train": np.asarray([0]),
                "val": np.asarray([1]),
                "test": np.asarray([2]),
            },
            {"logistic": {"solver": "liblinear", "max_iter": 10_000}},
            grid=[0.1],
            class_weight="balanced",
        )
    assert observed is sentinel


def test_binary_penalty_warning_filter_preserves_convergence_as_error() -> None:
    rng = np.random.RandomState(99)
    matrix = rng.normal(size=(100, 8))
    labels = np.asarray([0, 1] * 50)
    indices = {
        "train": np.arange(80),
        "val": np.arange(80, 100),
        "test": np.asarray([], dtype=np.int64),
    }
    with pytest.raises(ConvergenceWarning, match="failed to converge"):
        audit._fit_binary(
            matrix,
            labels,
            indices,
            {"logistic": {"solver": "liblinear", "max_iter": 1}},
            grid=[100.0],
            class_weight="balanced",
        )

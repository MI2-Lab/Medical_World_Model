from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


SRC = Path(__file__).parents[1] / "src"
sys.path.insert(0, str(SRC))

from conditional_ceiling.evaluation import (  # noqa: E402
    C_GRID,
    FTV_MODEL_FAMILIES,
    FULL_MODEL_FAMILIES,
    aggregate_oof_metrics,
    clinical_response_subgroups,
    compute_generalization_gaps,
    evaluate_feature_families,
    fit_compact_logistic,
    fit_profile_probe,
    generalization_gap_table,
    subgroup_metrics,
)


def _data() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(260812)
    n = 160
    latent = rng.normal(size=n)
    labels = (latent + rng.normal(scale=0.35, size=n) > 0).astype(int)
    mri = rng.normal(scale=0.05, size=(n, 72))
    mri[:, 0] = latent
    clinical = np.column_stack(
        (latent + rng.normal(scale=1.2, size=n), rng.normal(size=(n, 3)))
    )
    ftv = np.column_stack(
        (latent + rng.normal(scale=0.8, size=n), rng.normal(size=(n, 1)))
    )
    return {
        "labels": labels,
        "mri": mri,
        "clinical": clinical,
        "ftv": ftv,
        "train": np.arange(0, 100),
        "validation": np.arange(100, 130),
        "test": np.arange(130, 160),
        "patient_ids": np.array([f"P{index:03d}" for index in range(n)]),
    }


def test_compact_fit_is_outer_train_only_and_validation_selected() -> None:
    data = _data()
    train, validation, test = data["train"], data["validation"], data["test"]
    fit = fit_compact_logistic(
        data["mri"][train],
        data["labels"][train],
        data["mri"][validation],
        data["labels"][validation],
        data["mri"][test],
    )

    assert fit.selected_dimension in (8, 16, 32, 64)
    assert fit.selected_c in C_GRID
    assert fit.selection_metric == "validation_auroc"
    assert fit.tie_break == "smaller_dimension_then_smaller_C"
    np.testing.assert_allclose(fit.pca.mean_, data["mri"][train].mean(axis=0))
    np.testing.assert_allclose(
        fit.predict_proba(data["mri"][test]), fit.test_probabilities
    )
    assert fit.train_probabilities.shape == (100,)
    assert fit.validation_probabilities.shape == (30,)
    assert fit.test_probabilities.shape == (30,)


def test_compact_fit_ties_prefer_smaller_dimension_then_c() -> None:
    rng = np.random.default_rng(7)
    labels = np.tile([0, 1], 80)
    features = rng.normal(scale=1e-3, size=(160, 70))
    features[:, 0] = labels * 10.0 + rng.normal(scale=1e-3, size=160)
    fit = fit_compact_logistic(
        features[:100],
        labels[:100],
        features[100:130],
        labels[100:130],
        features[130:],
    )
    assert fit.validation_auroc == pytest.approx(1.0)
    assert fit.selected_dimension == 8
    assert fit.selected_c == min(C_GRID)


def test_registered_feature_families_enforce_population_separation() -> None:
    data = _data()
    common = dict(
        labels=data["labels"],
        mri_features=data["mri"],
        clinical_features=data["clinical"],
        train_indices=data["train"],
        validation_indices=data["validation"],
        test_indices=data["test"],
        patient_ids=data["patient_ids"],
        c_grid=(0.01, 1.0),
    )
    full = evaluate_feature_families(**common, population="full_808")
    assert tuple(full.fits) == FULL_MODEL_FAMILIES
    assert set(full.predictions["model_family"]) == set(FULL_MODEL_FAMILIES)
    assert set(full.predictions["split"]) == {"train", "validation", "test"}
    assert len(full.predictions) == 3 * len(data["labels"])
    aggregate = aggregate_oof_metrics(full.predictions)
    assert set(aggregate["model_family"]) == set(FULL_MODEL_FAMILIES)
    assert set(aggregate["n"]) == {30}

    ftv = evaluate_feature_families(
        **common,
        population="ftv_complete_375",
        ftv_features=data["ftv"],
    )
    assert tuple(ftv.fits) == FTV_MODEL_FAMILIES
    assert set(ftv.predictions["model_family"]) == set(FTV_MODEL_FAMILIES)

    with pytest.raises(ValueError, match="may not enter"):
        evaluate_feature_families(
            **common, population="full_808", ftv_features=data["ftv"]
        )
    with pytest.raises(ValueError, match="requires aligned FTV"):
        evaluate_feature_families(**common, population="ftv_complete_375")


def test_profile_probe_subgroups_and_generalization_primitives() -> None:
    data = _data()
    train, validation, test = data["train"], data["validation"], data["test"]
    profile = (data["mri"][:, 0] > 0).astype(int)
    probe = fit_profile_probe(
        data["mri"][train],
        profile[train],
        data["mri"][validation],
        profile[validation],
        data["mri"][test],
        c_grid=(0.01, 1.0),
    )
    assert probe.classes == (0, 1)
    assert probe.validation_score > 0.8
    assert probe.test_probabilities.shape == (30, 2)

    hr = np.tile([0, 1, 0], 20)
    her2 = np.tile([0, 0, 1], 20)
    groups = clinical_response_subgroups(hr, her2)
    assert set(groups) == {"HR-/HER2-", "HR+/HER2-", "HER2+"}
    subgroup_y = np.tile([0, 1], 30)
    subgroup_p = np.where(subgroup_y == 1, 0.8, 0.2)
    table = subgroup_metrics(
        subgroup_y, subgroup_p, groups, min_samples=20, min_per_class=5
    )
    assert table["eligible"].all()
    assert (table["auroc"] == 1.0).all()

    train_metrics = {name: value for name, value in zip(
        ("auroc", "auprc", "brier", "calibration_slope", "ece10"),
        (0.9, 0.8, 0.1, 1.1, 0.05),
    )}
    validation_metrics = {**train_metrics, "auroc": 0.8}
    test_metrics = {**train_metrics, "auroc": 0.7, "brier": 0.2}
    gaps = compute_generalization_gaps(train_metrics, validation_metrics, test_metrics)
    assert gaps["train_test_auroc_gap"] == pytest.approx(0.2)
    assert gaps["train_test_brier_gap"] == pytest.approx(-0.1)

    full = evaluate_feature_families(
        labels=data["labels"],
        mri_features=data["mri"],
        clinical_features=data["clinical"],
        train_indices=train,
        validation_indices=validation,
        test_indices=test,
        population="full_808",
        c_grid=(0.01,),
    )
    gap_table = generalization_gap_table(full.predictions)
    assert set(gap_table["model_family"]) == set(FULL_MODEL_FAMILIES)
    assert "train_test_auroc_gap" in gap_table


def test_profile_probe_supports_multiclass_subtype_with_ovr_liblinear() -> None:
    rng = np.random.default_rng(81)
    labels = np.tile(np.arange(4), 40)
    features = rng.normal(size=(160, 12))
    features[:, 0] = labels + rng.normal(scale=0.35, size=len(labels))

    probe = fit_profile_probe(
        features[:100],
        labels[:100],
        features[100:132],
        labels[100:132],
        features[132:],
        c_grid=(0.1, 1.0),
    )

    assert probe.classes == (0, 1, 2, 3)
    assert probe.score_name == "macro_ovr_auroc"
    assert probe.validation_score > 0.8
    assert probe.test_probabilities.shape == (28, 4)
    np.testing.assert_allclose(probe.test_probabilities.sum(axis=1), 1.0)

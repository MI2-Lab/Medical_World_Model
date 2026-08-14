from __future__ import annotations

import inspect
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

_MODULE_PATH = EXPERIMENT_ROOT / "src" / "patch_token_wm" / "downstream.py"
_SPEC = importlib.util.spec_from_file_location(
    "patch_token_wm_downstream", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
downstream = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = downstream
_SPEC.loader.exec_module(downstream)

from patch_token_wm_downstream import (  # type: ignore[import-not-found]  # noqa: E402
    C2_FULL_WITH_TREATMENT_FIELDS,
    PCR_DOWNSTREAM_PURPOSE,
    DownstreamContractError,
    FoldClinicalPreprocessor,
    build_literal_delta_rows,
    build_pcr_feature_sets,
    build_static_rows,
    causal_ftv_prefix,
    chronological_mri_prefix,
    load_pcr_labels_downstream_only,
)


def _clinical_train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "label_hr": [0, 1, 1, 0],
            "label_her2": [1, 0, 0, 1],
            "label_mp": [0, 1, np.nan, 0],
            "age_at_screening": [20.0, 40.0, np.nan, 60.0],
            "race_simple": ["White", "Asian", None, "White"],
            "menopausal_status_simple": [
                "Premenopausal",
                "Postmenopausal",
                "Premenopausal",
                None,
            ],
            "ethnicity": ["Hispanic", "Not Hispanic", None, "Not Hispanic"],
            "arm": [
                "Paclitaxel",
                "Paclitaxel + Trastuzumab",
                "Paclitaxel",
                "Paclitaxel + Trastuzumab",
            ],
        }
    )


def test_c2_preprocessor_is_train_only_median_sorted_missing_unseen_and_scaled() -> (
    None
):
    train = _clinical_train()
    preprocessor = FoldClinicalPreprocessor(outer_fold=2)
    train_scaled = preprocessor.fit_transform(
        train, train_patient_ids=["P0", "P1", "P2", "P3"]
    )

    assert preprocessor.fields == C2_FULL_WITH_TREATMENT_FIELDS
    assert preprocessor.numeric_medians_["age_at_screening"] == pytest.approx(40.0)
    assert preprocessor.numeric_medians_["label_mp"] == pytest.approx(0.0)
    assert preprocessor.categories_["race_simple"] == tuple(
        sorted(("White", "Asian", "__MISSING__"))
    )
    assert preprocessor.categories_["arm"] == tuple(
        sorted(("Paclitaxel", "Paclitaxel + Trastuzumab", "__MISSING__"))
    )
    assert train_scaled.shape[0] == len(train)
    np.testing.assert_allclose(train_scaled.mean(axis=0), 0.0, atol=1e-12)
    assert int(preprocessor.scaler_.n_samples_seen_) == len(train)

    heldout = train.iloc[:2].copy()
    heldout["age_at_screening"] = [np.nan, 500.0]
    heldout["race_simple"] = ["NEVER_SEEN", None]
    heldout["arm"] = ["EXACT_UNSEEN_ASSIGNED_ARM", None]
    encoded = preprocessor.encode(heldout)
    names = preprocessor.feature_names_
    assert encoded[0, names.index("age_at_screening")] == pytest.approx(40.0)
    race_columns = [
        index for index, name in enumerate(names) if name.startswith("race_simple=")
    ]
    arm_columns = [index for index, name in enumerate(names) if name.startswith("arm=")]
    assert encoded[0, race_columns].sum() == pytest.approx(0.0)
    assert encoded[0, arm_columns].sum() == pytest.approx(0.0)
    assert encoded[1, names.index("race_simple=__MISSING__")] == pytest.approx(1.0)
    assert encoded[1, names.index("arm=__MISSING__")] == pytest.approx(1.0)
    assert "race_simple=NEVER_SEEN" not in names
    assert "arm=EXACT_UNSEEN_ASSIGNED_ARM" not in names
    assert np.isfinite(preprocessor.transform(heldout)).all()

    provenance = preprocessor.provenance
    assert provenance["clinical_contract"] == "C2_full_with_treatment"
    assert provenance["fit_scope"] == "outer_train_only"
    assert provenance["train_patient_order_sha256"]
    assert provenance["heldout_unseen_nonmissing"].startswith("all_zero")


def test_clinical_preprocessor_rejects_nontrain_fit_refit_and_padded_categories() -> (
    None
):
    train = _clinical_train()
    with pytest.raises(DownstreamContractError, match="split='train'"):
        FoldClinicalPreprocessor(outer_fold=0).fit(train, split="val")
    fitted = FoldClinicalPreprocessor(outer_fold=0).fit(train)
    with pytest.raises(DownstreamContractError, match="single-fit"):
        fitted.fit(train)
    padded = train.copy()
    padded.loc[0, "race_simple"] = " White"
    with pytest.raises(DownstreamContractError, match="padded whitespace"):
        FoldClinicalPreprocessor(outer_fold=1).fit(padded)


def test_mri_and_ftv_prefixes_are_chronological_causal_and_no_future_mixing() -> None:
    states = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
    ftv = np.asarray([[0.0, 1.0, 3.0, 7.0], [1.0, 2.0, 4.0, 8.0]])

    mri = chronological_mri_prefix(states, "T1", state_dim=3)
    expected_mri = states[:, :2].reshape(2, 6)
    np.testing.assert_array_equal(mri, expected_mri)
    altered_states = states.copy()
    altered_states[:, 2:] = -999.0
    np.testing.assert_array_equal(
        mri, chronological_mri_prefix(altered_states, 1, state_dim=3)
    )

    causal = causal_ftv_prefix(ftv, "T1")
    np.testing.assert_allclose(causal, np.log1p(ftv[:, :2]))
    altered_ftv = ftv.copy()
    altered_ftv[:, 2:] = 100_000.0
    np.testing.assert_array_equal(causal, causal_ftv_prefix(altered_ftv, 1))
    with pytest.raises(DownstreamContractError, match="non-negative"):
        causal_ftv_prefix(np.full((2, 4), -1.0), "T0")


def test_pcr_feature_sets_keep_locked_order_and_have_no_outcome_argument() -> None:
    clinical = np.arange(10, dtype=float).reshape(2, 5)
    states = np.arange(2 * 4 * 3, dtype=float).reshape(2, 4, 3)
    ftv = np.asarray([[0.0, 1.0, 3.0, 7.0], [1.0, 2.0, 4.0, 8.0]])
    features = build_pcr_feature_sets(clinical, states, "T1", ftv=ftv, state_dim=3)

    assert set(features) == {"C", "M", "F", "C+M", "C+F", "C+F+M"}
    np.testing.assert_array_equal(features["C+M"][:, :5], clinical)
    np.testing.assert_array_equal(features["C+M"][:, 5:], states[:, :2].reshape(2, 6))
    np.testing.assert_array_equal(features["C+F"][:, :5], clinical)
    np.testing.assert_allclose(features["C+F"][:, 5:], np.log1p(ftv[:, :2]))
    np.testing.assert_array_equal(features["C+F+M"][:, :5], clinical)
    np.testing.assert_allclose(features["C+F+M"][:, 5:7], np.log1p(ftv[:, :2]))
    np.testing.assert_array_equal(features["C+F+M"][:, 7:], states[:, :2].reshape(2, 6))
    signature = inspect.signature(build_pcr_feature_sets)
    assert not any(
        term in parameter.casefold()
        for parameter in signature.parameters
        for term in ("pcr", "label", "outcome", "target")
    )


def test_static_and_literal_delta_rows_use_natural_targets_and_matched_validity() -> (
    None
):
    ids = ["P0", "P1", "P2"]
    states = np.asarray(
        [
            [[0.0, 1.0], [2.0, 4.0], [5.0, 9.0], [9.0, 16.0]],
            [[1.0, 1.0], [3.0, 5.0], [6.0, 10.0], [10.0, 17.0]],
            [[2.0, 1.0], [4.0, 6.0], [7.0, 11.0], [11.0, 18.0]],
        ]
    )
    ftv = np.asarray(
        [[1.0, 3.0, 8.0, 10.0], [2.0, np.nan, 5.0, 9.0], [3.0, 4.0, 6.0, 12.0]]
    )
    valid = np.isfinite(ftv)
    state_valid = np.ones((3, 4), dtype=bool)
    state_valid[1, 1] = False
    state_valid[2, 2] = False

    static = build_static_rows(
        ids,
        states,
        ftv,
        "T1",
        ftv_valid=valid,
        state_valid=state_valid,
    )
    assert static.patient_ids == ("P0", "P2")
    np.testing.assert_array_equal(static.source_row_indices, [0, 2])
    np.testing.assert_array_equal(static.features, states[[0, 2], 1])
    np.testing.assert_array_equal(static.targets, [3.0, 4.0])
    assert static.target_semantics == "natural_FTV_at_observed_visit"
    assert static.excluded_invalid_targets == 1

    delta = build_literal_delta_rows(
        ids,
        states,
        ftv,
        "T1_to_T2",
        ftv_valid=valid,
        state_valid=state_valid,
    )
    assert delta.patient_ids == ("P0",)
    np.testing.assert_array_equal(delta.features, states[[0], 2] - states[[0], 1])
    np.testing.assert_array_equal(delta.targets, [5.0])
    assert delta.target_semantics == "literal_natural_FTV_end_minus_FTV_start"
    assert delta.endpoint == "T1_to_T2"
    assert delta.excluded_invalid_targets == 2


def test_pcr_labels_require_explicit_downstream_purpose_and_preserve_order() -> None:
    table = pd.DataFrame({"patient_id": ["P0", "P1", "P2"], "label_pcr": [0, 1, 0]})
    with pytest.raises(PermissionError, match="frozen_downstream_probe"):
        load_pcr_labels_downstream_only(
            table, ["P2", "P0"], purpose="world_model_training"
        )
    labels = load_pcr_labels_downstream_only(
        table, ["P2", "P1", "P0"], purpose=PCR_DOWNSTREAM_PURPOSE
    )
    np.testing.assert_array_equal(labels, [0, 1, 0])
    broken = table.copy()
    broken.loc[0, "label_pcr"] = 2
    with pytest.raises(DownstreamContractError, match="binary 0/1"):
        load_pcr_labels_downstream_only(
            broken, ["P0", "P1"], purpose=PCR_DOWNSTREAM_PURPOSE
        )


def test_no_nonexplicit_downstream_api_accepts_pcr_or_labels() -> None:
    explicit = "load_pcr_labels_downstream_only"

    for name in downstream.__all__:
        value = getattr(downstream, name)
        if not inspect.isfunction(value) or name == explicit:
            continue
        parameters = inspect.signature(value).parameters
        assert not any("pcr" in parameter.casefold() for parameter in parameters)

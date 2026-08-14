from __future__ import annotations

import pandas as pd
import pytest
import torch

from conditional_ceiling.strata import (
    build_exact_strata,
    conditional_pair_masks,
)


def _training_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": [f"synthetic-{index}" for index in range(10)],
            "label_pcr": [0, 0, 1, 1, 0, 1, 1, 0, 0, 0],
            "label_hr": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            "label_her2": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            "arm": ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C"],
            "fold": [2] * 10,
            "split": ["train"] * 10,
        }
    )


def test_exact_strata_eligibility_and_audit_counts() -> None:
    result = build_exact_strata(_training_rows())
    assignments = result.assignments
    # First two strata contain two members of both classes: all eight anchors
    # have a partner and an opposite class. Last stratum has only class zero.
    assert assignments["eligible_anchor"].tolist() == [True] * 8 + [False, False]
    assert result.audit.total_strata == 3
    assert result.audit.usable_strata == 2
    assert result.audit.bidirectionally_usable_strata == 2
    assert result.audit.usable_patients == 8
    assert result.audit.dropped_anchors == 2
    assert result.audit.dropped_no_same_class_partner == 0
    assert result.audit.dropped_no_opposite_class == 2
    assert result.audit.pcr_class_distribution == {0: 6, 1: 4}
    assert result.audit.usable_pcr_class_distribution == {0: 4, 1: 4}
    assert result.audit.as_dict()["unmatched_fallback_used"] is False
    assert result.audit.as_dict()["test_patients_used"] is False


def test_exact_strata_reject_nontraining_rows_and_field_fallback() -> None:
    rows = _training_rows()
    rows.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="outer-train"):
        build_exact_strata(rows)
    with pytest.raises(ValueError, match="matching fields are frozen"):
        build_exact_strata(_training_rows(), matching_fields=("label_hr", "label_her2"))


def test_pair_masks_are_self_excluding_and_cross_stratum_free() -> None:
    labels = torch.tensor([0, 0, 1, 0, 0, 1])
    strata = torch.tensor([4, 4, 4, 9, 9, 9])
    masks = conditional_pair_masks(labels, strata)
    assert not masks.denominator.diagonal().any()
    assert not masks.denominator[:3, 3:].any()
    assert not masks.denominator[3:, :3].any()
    assert masks.positives[0].tolist() == [False, True, False, False, False, False]
    assert masks.negatives[0].tolist() == [False, False, True, False, False, False]
    # Only the class-zero members have a same-class partner in each stratum.
    assert masks.eligible_anchor.tolist() == [True, True, False, True, True, False]


def test_pair_masks_reject_false_eligibility_claim_instead_of_fallback() -> None:
    labels = torch.tensor([0, 1, 0])
    strata = torch.tensor([0, 0, 1])
    with pytest.raises(ValueError, match="lacks a same-class partner"):
        conditional_pair_masks(labels, strata, torch.tensor([True, False, False]))

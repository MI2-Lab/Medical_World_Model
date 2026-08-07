from __future__ import annotations

import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from shortcut_audit.auditlib.matching import (
    MatchingConfig,
    match_follow_up_donors,
)


def _patient(
    patient_id: str,
    *,
    fold: int = 0,
    hr: int = 1,
    her2: int = 0,
    treatment_family: str = "taxane",
    volume: float = 10.0,
    has_t1: bool = True,
    has_t2: bool = True,
    age: float = 50.0,
    mammaprint: int = 1,
    label_pcr: int = 0,
) -> dict[str, object]:
    return {
        "patient_id": patient_id,
        "fold": fold,
        "hr": hr,
        "her2": her2,
        "treatment_family": treatment_family,
        "baseline_lesion_volume": volume,
        "has_t1": has_t1,
        "has_t2": has_t2,
        "age": age,
        "mammaprint": mammaprint,
        # The frame may contain this label; the matcher must ignore it.
        "label_pcr": label_pcr,
    }


class MatchingTest(unittest.TestCase):
    def test_outcome_feature_is_rejected_and_existing_label_is_ignored(self) -> None:
        frame = pd.DataFrame(
            [
                _patient("A", volume=10.0, label_pcr=0),
                _patient("B", volume=11.0, label_pcr=1),
                _patient("C", volume=12.0, label_pcr=0),
            ]
        )
        config = MatchingConfig(max_donors=1, seed=7)

        with self.assertRaisesRegex(ValueError, "outcome/label"):
            match_follow_up_donors(
                frame,
                config,
                matching_features=("label_pcr",),
            )

        first = match_follow_up_donors(frame, config).mapping
        changed_labels = frame.copy()
        changed_labels["label_pcr"] = 1 - changed_labels["label_pcr"]
        second = match_follow_up_donors(changed_labels, config).mapping
        assert_frame_equal(first, second)

    def test_matching_is_deterministic_including_seeded_ties(self) -> None:
        frame = pd.DataFrame(
            [
                _patient("R", volume=10.0),
                _patient("D1", volume=12.0),
                _patient("D2", volume=12.0),
                _patient("D3", volume=12.0),
            ]
        )
        config = MatchingConfig(max_donors=2, seed=123)
        first = match_follow_up_donors(frame, config)
        second = match_follow_up_donors(frame.sample(frac=1.0, random_state=9), config)

        first_r = first.mapping.loc[
            first.mapping["recipient_patient_id"] == "R"
        ].reset_index(drop=True)
        second_r = second.mapping.loc[
            second.mapping["recipient_patient_id"] == "R"
        ].reset_index(drop=True)
        assert_frame_equal(first_r, second_r)
        self.assertEqual(first_r["audit_repetition"].tolist(), [1, 2])

    def test_hard_subtype_and_treatment_match_precedes_volume(self) -> None:
        frame = pd.DataFrame(
            [
                _patient("R", volume=10.0),
                _patient("EXACT", volume=30.0),
                _patient("WRONG_SUBTYPE", hr=0, volume=10.1),
                _patient("WRONG_TREATMENT", treatment_family="platinum", volume=10.2),
            ]
        )
        result = match_follow_up_donors(frame, MatchingConfig(max_donors=1, seed=3))
        match = result.mapping.loc[result.mapping["recipient_patient_id"] == "R"].iloc[0]

        self.assertEqual(match["donor_patient_id"], "EXACT")
        self.assertEqual(match["matching_level"], "hard_subtype_treatment_visit")
        self.assertTrue(match["subtype_match"])
        self.assertTrue(match["treatment_family_match"])

    def test_no_candidate_is_reported_without_cross_fold_borrowing(self) -> None:
        frame = pd.DataFrame(
            [
                _patient("ONLY_FOLD_0", fold=0),
                _patient("ONLY_FOLD_1", fold=1),
            ]
        )
        result = match_follow_up_donors(frame, MatchingConfig(max_donors=1))

        self.assertTrue(result.mapping.empty)
        self.assertEqual(len(result.failures), 2)
        self.assertEqual(set(result.failures["status"]), {"unmatched"})
        self.assertEqual(
            set(result.failures["failure_reason"]),
            {"no_same_fold_nonself_donor"},
        )
        self.assertEqual(result.success_stats["n_failed_recipients"], 2)
        self.assertEqual(result.success_stats["success_rate"], 0.0)

    def test_fold_isolation_no_self_donor_and_visit_compatibility(self) -> None:
        frame = pd.DataFrame(
            [
                _patient("A", fold=0, volume=10.0, has_t2=True),
                _patient("B", fold=0, volume=11.0, has_t2=True),
                _patient("BAD_VISIT", fold=0, volume=10.1, has_t2=False),
                _patient("C", fold=1, volume=10.0, has_t2=True),
                _patient("D", fold=1, volume=11.0, has_t2=True),
            ]
        )
        result = match_follow_up_donors(frame, MatchingConfig(max_donors=1, seed=19))
        lookup = frame.set_index("patient_id")["fold"].to_dict()

        self.assertFalse(result.mapping.empty)
        for row in result.mapping.itertuples(index=False):
            self.assertNotEqual(row.recipient_patient_id, row.donor_patient_id)
            self.assertEqual(lookup[row.recipient_patient_id], lookup[row.donor_patient_id])
            self.assertEqual(row.fold, lookup[row.recipient_patient_id])
            self.assertTrue(row.visit_availability_compatible)

        donor_for_a = result.mapping.loc[
            result.mapping["recipient_patient_id"] == "A", "donor_patient_id"
        ].item()
        self.assertEqual(donor_for_a, "B")

    def test_shortfall_balance_and_relaxed_level_are_explicit(self) -> None:
        frame = pd.DataFrame(
            [
                _patient("R", volume=10.0),
                _patient("RELAXED", treatment_family="platinum", volume=10.1),
            ]
        )
        config = MatchingConfig(
            max_donors=2,
            seed=11,
            allow_relaxed_matches=True,
        )
        result = match_follow_up_donors(frame, config)
        recipient_match = result.mapping.loc[
            result.mapping["recipient_patient_id"] == "R"
        ].iloc[0]
        recipient_failure = result.failures.loc[
            result.failures["recipient_patient_id"] == "R"
        ].iloc[0]

        self.assertEqual(recipient_match["matching_level"], "relaxed_subtype_visit")
        self.assertEqual(recipient_failure["status"], "partial")
        self.assertEqual(
            recipient_failure["failure_reason"], "fewer_than_requested_donors"
        )
        self.assertIn("mean_volume_distance_z", result.balance_stats.columns)
        self.assertIn(
            "relaxed_subtype_visit",
            result.success_stats["counts_by_matching_level"],
        )


if __name__ == "__main__":
    unittest.main()

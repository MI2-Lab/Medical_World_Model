from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(
    0,
    str(
        REPO_ROOT
        / "additional_experiments/c1b_model_ready_ftv_sanity/src"
    ),
)

from c1b_overlap_stageb.eligibility import (  # noqa: E402
    VISITS,
    build_patient_eligibility,
    count_valid_source_voxels,
)
from c1b_overlap_stageb.io import verify_preregistration  # noqa: E402
from c1b_sanity.dce7 import _valid_source_footprint_mask_xyz  # noqa: E402


def synthetic_visits(*, second_patient_t2_zero: bool = False) -> pd.DataFrame:
    rows = []
    for patient_id in ("synthetic-a", "synthetic-b"):
        for visit in VISITS:
            valid = (
                0
                if second_patient_t2_zero
                and patient_id == "synthetic-b"
                and visit == "T2"
                else 7
            )
            rows.append(
                {
                    "patient_id": patient_id,
                    "cohort": "synthetic",
                    "visit": visit,
                    "valid_source_voxels": valid,
                    "target_grid_voxels": 10,
                }
            )
    return pd.DataFrame(rows)


class EligibilityContractTests(unittest.TestCase):
    def test_preregistration_hash_is_frozen(self) -> None:
        lock = verify_preregistration(experiment_root=ROOT)
        self.assertIs(lock["preregistered_before_new_cohort_statistics"], True)
        self.assertIs(lock["expected_population_change_is_not_a_result"], True)

    def test_exact_valid_count_identity_and_zero_shift(self) -> None:
        identity = np.eye(4, dtype=np.float64)
        self.assertEqual(
            count_valid_source_voxels(identity, (5, 4, 3), (5, 4, 3), x_slab=2),
            60,
        )
        outside = np.eye(4, dtype=np.float64)
        outside[0, 3] = 100.0
        self.assertEqual(
            count_valid_source_voxels(outside, (5, 4, 3), (5, 4, 3)), 0
        )

    def test_exact_valid_count_uses_half_voxel_footprint_boundary(self) -> None:
        mapping = np.eye(4, dtype=np.float64)
        mapping[:3, 3] = (-0.5, -0.5, -0.5)
        self.assertEqual(
            count_valid_source_voxels(mapping, (3, 3, 3), (3, 3, 3)), 27
        )
        mapping[2, 3] = -0.5000001
        self.assertEqual(
            count_valid_source_voxels(mapping, (3, 3, 3), (3, 3, 3)), 18
        )

    def test_slabbed_count_matches_frozen_builder_helper_randomized(self) -> None:
        rng = np.random.default_rng(2026)
        for _ in range(100):
            source_shape = tuple(int(value) for value in rng.integers(3, 12, size=3))
            output_shape = tuple(int(value) for value in rng.integers(2, 10, size=3))
            linear = np.eye(3) + rng.normal(0.0, 0.2, size=(3, 3))
            if abs(np.linalg.det(linear)) < 0.1:
                linear += np.eye(3)
            mapping = np.eye(4, dtype=np.float64)
            mapping[:3, :3] = linear
            mapping[:3, 3] = rng.uniform(-4.0, 4.0, size=3)
            expected = int(
                np.count_nonzero(
                    _valid_source_footprint_mask_xyz(
                        mapping, output_shape, source_shape
                    )
                )
            )
            observed = count_valid_source_voxels(
                mapping, output_shape, source_shape, x_slab=3
            )
            self.assertEqual(observed, expected)

    def test_four_visit_and_excludes_whole_patient_for_one_zero_visit(self) -> None:
        patients = build_patient_eligibility(
            synthetic_visits(second_patient_t2_zero=True)
        )
        first = patients.set_index("patient_id").loc["synthetic-a"]
        second = patients.set_index("patient_id").loc["synthetic-b"]
        self.assertIs(bool(first["eligible"]), True)
        self.assertEqual(first["exclusion_reason"], "")
        self.assertIs(bool(second["eligible"]), False)
        self.assertEqual(second["zero_overlap_visit_count"], 1)
        self.assertEqual(
            second["exclusion_reason"],
            "ZERO_VALID_SOURCE_OVERLAP_IN_REQUIRED_VISIT",
        )

    def test_incomplete_patient_fails_closed(self) -> None:
        frame = synthetic_visits().query(
            "not (patient_id == 'synthetic-a' and visit == 'T3')"
        )
        with self.assertRaisesRegex(ValueError, "exactly T0-T3"):
            build_patient_eligibility(frame)

    def test_plan_and_lock_do_not_assert_expected_population_as_result(self) -> None:
        lock = json.loads((ROOT / "configs/preregistration_lock.json").read_text())
        self.assertIs(lock["expected_population_change_is_not_a_result"], True)
        source = (ROOT / "scripts/run_technical_eligibility.py").read_text()
        self.assertNotIn("== 947", source)
        self.assertNotIn("== 948", source)
        self.assertNotIn("CASE_ZERO_OVERLAP", source)


if __name__ == "__main__":
    unittest.main()


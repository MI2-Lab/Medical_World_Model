from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from ispy_jepa_tmi_clean.corejepa.data.records import PatientRecord
from shortcut_audit.auditlib.baseline_features import (
    ClinicalFeatureSpec,
    clinical_features,
    clinical_geometry_features,
    geometry_features,
    static_t0_features,
    timepoint_only_features,
)


def _record(patient_id: str, arm: str, age: float, pcr: int) -> PatientRecord:
    return PatientRecord(
        patient_id=patient_id,
        cohort="ispy2",
        arm=arm,
        hr=1,
        her2=0,
        mp=1,
        age=age,
        manifest_path=Path("unused.json"),
        pcr=pcr,
    )


class BaselineFeatureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [_record("A", "arm-a", 40.0, 0), _record("B", "arm-b", 60.0, 1)]
        self.geometry = np.arange(2 * 4 * 9, dtype=np.float32).reshape(2, 4, 9) / 100

    def test_clinical_spec_and_features_do_not_use_outcome(self) -> None:
        spec = ClinicalFeatureSpec.fit(self.records)
        features = clinical_features(self.records, spec)
        changed = [
            _record(record.patient_id, record.arm, record.age, 1 - int(record.pcr))
            for record in self.records
        ]
        self.assertEqual(features.shape, (2, 3, 9))
        np.testing.assert_array_equal(features, clinical_features(changed, spec))
        np.testing.assert_array_equal(features[0, :, :-3], np.repeat(features[0, :1, :-3], 3, axis=0))
        np.testing.assert_array_equal(features[:, :, -3:], np.broadcast_to(np.eye(3), (2, 3, 3)))

    def test_unseen_arm_fails_closed(self) -> None:
        spec = ClinicalFeatureSpec.fit(self.records[:1])
        with self.assertRaisesRegex(ValueError, "未见 arm"):
            clinical_features(self.records, spec)

    def test_geometry_never_reads_t3_for_audit_decisions(self) -> None:
        first = geometry_features(self.geometry)
        modified = self.geometry.copy()
        modified[:, 3] += 10_000
        second = geometry_features(modified)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (2, 3, 57))
        np.testing.assert_array_equal(first[:, 0, 18:36], 0.0)

    def test_combined_timepoint_and_static_shapes(self) -> None:
        spec = ClinicalFeatureSpec.fit(self.records)
        combined = clinical_geometry_features(self.records, self.geometry, spec)
        self.assertEqual(combined.shape, (2, 3, 63))
        np.testing.assert_array_equal(timepoint_only_features(2), combined[:, :, -3:])
        t0 = np.arange(10, dtype=np.float32).reshape(2, 5)
        static = static_t0_features(t0)
        self.assertEqual(static.shape, (2, 3, 8))
        np.testing.assert_array_equal(static[:, 0, :-3], static[:, 2, :-3])

    def test_invalid_geometry_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "\[N,4,9\]"):
            geometry_features(np.zeros((2, 3, 9), dtype=np.float32))
        bad = self.geometry.copy()
        bad[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "非有限"):
            geometry_features(bad)


if __name__ == "__main__":
    unittest.main()

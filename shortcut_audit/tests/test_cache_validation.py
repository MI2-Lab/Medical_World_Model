from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ispy_jepa_tmi_clean.corejepa.data.response_targets import response_feature_names
from shortcut_audit.auditlib.cache_validation import validate_response_feature_cache


class ResponseCacheValidationTests(unittest.TestCase):
    def _cache(self, path: Path, patient_ids: list[str]) -> None:
        names = response_feature_names()
        x_visit = np.ones((len(patient_ids), 4, len(names)), dtype=np.float32)
        np.savez_compressed(
            path,
            x_visit=x_visit,
            patient_ids=np.asarray(patient_ids),
            feature_names=np.asarray(names),
            roi_sources=np.full((len(patient_ids), 4), "ftv", dtype="U3"),
        )

    def test_accepts_exact_schema_and_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            self._cache(path, ["P0", "P1"])
            result = validate_response_feature_cache(path, ["P0", "P1"])
        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["x_visit_shape"], [2, 4, 106])
        self.assertEqual(result["raw_response_shape"], [2, 3, 18])
        self.assertEqual(result["roi_source_counts"], {"ftv": 8})

    def test_rejects_patient_reordering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            self._cache(path, ["P1", "P0"])
            with self.assertRaisesRegex(ValueError, "patient order"):
                validate_response_feature_cache(path, ["P0", "P1"])

    def test_rejects_infinite_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            names = response_feature_names()
            x_visit = np.ones((1, 4, len(names)), dtype=np.float32)
            x_visit[0, 0, 0] = np.inf
            np.savez_compressed(
                path,
                x_visit=x_visit,
                patient_ids=np.asarray(["P0"]),
                feature_names=np.asarray(names),
                roi_sources=np.full((1, 4), "ftv", dtype="U3"),
            )
            with self.assertRaisesRegex(ValueError, "无穷"):
                validate_response_feature_cache(path, ["P0"])


if __name__ == "__main__":
    unittest.main()

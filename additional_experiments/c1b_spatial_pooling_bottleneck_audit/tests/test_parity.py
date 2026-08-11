from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c1b_spatial_audit.parity import compare_p0_asset  # noqa: E402
from c1b_spatial_audit.probes import FrozenStateAsset  # noqa: E402


class P0ParityTests(unittest.TestCase):
    @staticmethod
    def _candidate(state: np.ndarray) -> FrozenStateAsset:
        patients = np.asarray([f"P{index:04d}" for index in range(808)])
        split = np.asarray(["train"] * 525 + ["val"] * 121 + ["test"] * 162)
        return FrozenStateAsset(
            patient_id=patients,
            split=split,
            state=state,
            state_valid=np.ones((808, 4), dtype=bool),
            arm="L1",
            seed_base=2026,
            fold=0,
            pooling="P0",
        )

    @staticmethod
    def _write_reference(path: Path, state: np.ndarray) -> None:
        patients = np.asarray([f"P{index:04d}" for index in range(808)])
        split = np.asarray(["train"] * 525 + ["val"] * 121 + ["test"] * 162)
        np.savez_compressed(
            path,
            patient_id=patients,
            split=split,
            response_state=state,
            arm=np.asarray("L1"),
            seed_base=np.asarray(2026, dtype=np.int64),
            fold=np.asarray(0, dtype=np.int64),
        )

    def test_exact_candidate_passes_and_records_bitwise_fraction(self) -> None:
        generator = np.random.default_rng(11)
        state = generator.normal(size=(808, 4, 192)).astype(np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            reference = Path(temporary) / "reference.private.npz"
            self._write_reference(reference, state)
            row = compare_p0_asset(self._candidate(state.copy()), reference)
        self.assertEqual(row["status"], "PASS")
        self.assertEqual(row["allclose_fraction"], 1.0)
        self.assertEqual(row["bitwise_equal_fraction"], 1.0)
        self.assertEqual(row["max_absolute_error"], 0.0)

    def test_out_of_tolerance_or_identity_drift_fails(self) -> None:
        state = np.zeros((808, 4, 192), dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            reference = Path(temporary) / "reference.private.npz"
            self._write_reference(reference, state)
            changed = state.copy()
            changed[0, 0, 0] = 1e-3
            row = compare_p0_asset(self._candidate(changed), reference)
            self.assertEqual(row["status"], "FAIL")
            bad = self._candidate(state.copy())
            bad.patient_id[0] = "WRONG"
            with self.assertRaisesRegex(ValueError, "patient order"):
                compare_p0_asset(bad, reference)

    def test_invalid_visit_or_non_p0_is_rejected(self) -> None:
        state = np.zeros((808, 4, 192), dtype=np.float32)
        candidate = self._candidate(state)
        object.__setattr__(candidate, "pooling", "PLOCAL")
        with self.assertRaisesRegex(ValueError, "only consume"):
            compare_p0_asset(candidate, "unused")


if __name__ == "__main__":
    unittest.main()

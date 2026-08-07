from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from shortcut_audit.auditlib.folds import (
    fold_split_indices,
    held_out_assignments,
    load_fold_manifest,
    validate_fold_manifest,
)


def valid_manifest() -> pd.DataFrame:
    rows = []
    patients = [f"P{index}" for index in range(5)]
    for fold in range(5):
        for index, patient_id in enumerate(patients):
            split = "test" if index == fold else "val" if index == (fold + 1) % 5 else "train"
            rows.append({"patient_id": patient_id, "fold": fold, "split": split, "label_pcr": index % 2})
    return pd.DataFrame(rows)


class FoldManifestTest(unittest.TestCase):
    def test_valid_oof_contract_and_indices(self) -> None:
        frame, summary = validate_fold_manifest(valid_manifest())
        self.assertEqual(summary["n_rows"], 25)
        self.assertEqual(summary["n_patients"], 5)
        held_out = held_out_assignments(frame)
        self.assertEqual(len(held_out), 5)
        indices = fold_split_indices(frame, ["P4", "P3", "P2", "P1", "P0"], 0)
        self.assertEqual(len(indices["test"]), 1)
        self.assertEqual(len(indices["val"]), 1)
        self.assertEqual(len(indices["train"]), 3)

    def test_duplicate_patient_fold_is_rejected(self) -> None:
        frame = pd.concat([valid_manifest(), valid_manifest().iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "出现多次"):
            validate_fold_manifest(frame)

    def test_label_instability_is_rejected(self) -> None:
        frame = valid_manifest()
        frame.loc[(frame.patient_id == "P0") & (frame.fold == 1), "label_pcr"] = 1
        with self.assertRaisesRegex(ValueError, "label"):
            validate_fold_manifest(frame)

    def test_shared_candidate_when_present(self) -> None:
        path = Path(
            "/data/data/Preprocessed/I-SPY2/"
            "_matched_breastdcedl_t0_dicomrepair_rgb224_seed2026/"
            "matched_patient_cv_splits_seed2026.csv"
        )
        if not path.exists():
            self.skipTest("共享五折候选在当前机器不存在")
        _, summary = load_fold_manifest(path)
        self.assertEqual(summary["n_rows"], 4040)
        self.assertEqual(summary["n_patients"], 808)
        self.assertEqual(
            summary["sha256"],
            "143e482d711225c0611006d99bd7345d2fa1a5c16c65fbaf8399341a0d26aa38",
        )


if __name__ == "__main__":
    unittest.main()

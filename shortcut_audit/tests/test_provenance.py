from __future__ import annotations

import unittest

import torch

from shortcut_audit.auditlib.provenance import validate_checkpoint_payload


def payload() -> dict[str, object]:
    return {
        "model": {"weight": torch.ones(1)},
        "config": {"model": {}},
        "condition": {
            "feature_names": ["time", "HR"],
            "arm_vocab": {"A": 0},
            "age_mean": 50.0,
            "age_std": 10.0,
        },
        "response_transform": {"median": [0.0]},
        "patient_ids": ["P0", "P1", "P2", "E0"],
        "n_primary": 3,
        "splits": {
            "primary_train": [0],
            "pretrain_train": [0, 3],
            "validation": [1],
            "test": [2],
        },
        "epoch": 4,
        "validation": {"prediction": 0.5},
    }


class CheckpointProvenanceTest(unittest.TestCase):
    def test_valid_clean_payload_and_fold_alignment(self) -> None:
        summary = validate_checkpoint_payload(
            payload(),
            expected_primary_ids=["P2", "P0", "P1"],
            expected_fold_ids={"train": ["P0"], "val": ["P1"], "test": ["P2"]},
        )
        self.assertEqual(summary["n_primary"], 3)
        self.assertEqual(summary["n_extra"], 1)

    def test_legacy_schema_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "clean schema"):
            validate_checkpoint_payload({"model": {"weight": torch.ones(1)}, "epoch": 1})

    def test_split_overlap_is_rejected(self) -> None:
        value = payload()
        value["splits"]["validation"] = [0]  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "重叠"):
            validate_checkpoint_payload(value)

    def test_extra_record_in_primary_split_is_rejected(self) -> None:
        value = payload()
        value["splits"]["primary_train"] = [0, 3]  # type: ignore[index]
        value["splits"]["pretrain_train"] = [0, 3]  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "extra"):
            validate_checkpoint_payload(value)


if __name__ == "__main__":
    unittest.main()

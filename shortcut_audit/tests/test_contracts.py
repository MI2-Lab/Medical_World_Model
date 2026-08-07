from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from shortcut_audit.auditlib.contracts import (
    PREDICTION_COLUMNS,
    validate_label_alignment,
    validate_prediction_frame,
    write_prediction_csv,
)


def prediction_frame(condition: str = "native") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "fold": 0,
                "decision_point": "T0-T1",
                "audit_condition": condition,
                "y_true": 1,
                "predicted_probability": 0.8,
                "predicted_label": 1,
                "threshold": 0.5,
                "checkpoint": "/checkpoint/fold0.pt",
                "donor_patient_id": np.nan,
                "repetition_id": np.nan,
                "matching_distance": np.nan,
            }
        ],
        columns=PREDICTION_COLUMNS,
    )


class PredictionContractTest(unittest.TestCase):
    def test_valid_frame_and_atomic_write(self) -> None:
        frame = validate_prediction_frame(prediction_frame())
        with tempfile.TemporaryDirectory() as directory:
            path = write_prediction_csv(frame, Path(directory) / "predictions.csv")
            loaded = pd.read_csv(path)
        self.assertEqual(list(loaded.columns), list(PREDICTION_COLUMNS))

    def test_label_threshold_mismatch_is_rejected(self) -> None:
        frame = prediction_frame()
        frame.loc[0, "predicted_label"] = 0
        with self.assertRaisesRegex(ValueError, "不一致"):
            validate_prediction_frame(frame)

    def test_donor_self_match_is_rejected(self) -> None:
        frame = prediction_frame("followup_swap")
        frame["donor_patient_id"] = frame["donor_patient_id"].astype("object")
        frame.loc[0, "donor_patient_id"] = "P1"
        frame.loc[0, "repetition_id"] = 0
        frame.loc[0, "matching_distance"] = 0.0
        with self.assertRaisesRegex(ValueError, "不得等于"):
            validate_prediction_frame(frame, require_donor=True)

    def test_cross_condition_label_mismatch_is_rejected(self) -> None:
        native = prediction_frame("native")
        audit = prediction_frame("repeated_t0")
        audit.loc[0, "y_true"] = 0
        with self.assertRaisesRegex(ValueError, "label 不一致"):
            validate_label_alignment([native, audit])


if __name__ == "__main__":
    unittest.main()

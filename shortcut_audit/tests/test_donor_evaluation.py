from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import torch
from torch.utils.data import Dataset

from shortcut_audit.auditlib.contracts import DECISION_POINTS
from shortcut_audit.auditlib.donor_evaluation import (
    ARTIFACT_FILENAMES,
    AUDIT_CONDITION,
    _checkpoint_identity,
    run_matched_donor_fold_audit,
)
from shortcut_audit.auditlib.matching import MatchingConfig
from shortcut_audit.auditlib.readouts import AuditReadoutConfig, fit_fold_readout


HELDOUT_IDS = ("A", "B", "C", "D", "E", "F")


class _HeldoutDataset(Dataset):
    def __init__(self) -> None:
        self.ids = list(HELDOUT_IDS)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        offset = float(index * 10)
        image = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8, 1, 1, 1) + offset
        geometry = (
            torch.arange(4 * 9, dtype=torch.float32).reshape(4, 9) / 10.0 + offset
        )
        condition = torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4) / 100.0
        return {
            "patient_id": self.ids[index],
            "image": image,
            "geometry": geometry,
            "condition": condition,
        }


class _Response:
    def __init__(self, state: torch.Tensor) -> None:
        self.future_state = state
        self.latent_correction = state * 0.01


class _TinyModel(torch.nn.Module):
    def encode_visits(self, image, geometry):
        return torch.stack((image[:, :, 0, 0, 0, 0], geometry[:, :, 0]), dim=-1)

    def encode_targets(self, image, geometry):
        return self.encode_visits(image, geometry) + 0.5

    def image_transition(self, state, condition):
        return state + condition[..., :2]

    def response_transition(self, geometry, condition):
        return _Response(geometry[..., :2] + condition[..., :2])

    def forecast_response(self, geometry, condition):
        return self.response_transition(geometry[:, :-1], condition).future_state

    def forward(self, image, geometry, condition):
        class Output:
            pass

        output = Output()
        output.visit_state = self.encode_visits(image, geometry)
        output.target = self.encode_targets(image, geometry)[:, 1:]
        output.image_prediction = self.image_transition(
            output.visit_state[:, :-1], condition
        )
        response = self.response_transition(geometry[:, :-1], condition)
        output.prediction = output.image_prediction + response.latent_correction
        output.future_response_state = response.future_state
        return output


def _metadata(*, flip_pcr: bool = False) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": HELDOUT_IDS,
            "fold": [2] * len(HELDOUT_IDS),
            "hr": [1] * len(HELDOUT_IDS),
            "her2": [0] * len(HELDOUT_IDS),
            "treatment_family": [
                "taxane",
                "taxane",
                "taxane",
                "platinum",
                "platinum",
                "platinum",
            ],
            "baseline_lesion_volume": [10.0, 10.2, 10.4, 10.1, 10.3, 10.5],
            "has_t1": [True] * len(HELDOUT_IDS),
            "has_t2": [True] * len(HELDOUT_IDS),
            "age": [40, 42, 44, 41, 43, 45],
            "mammaprint": [0, 1, 0, 1, 0, 1],
            # This column coexists in the frame but must never enter matching.
            "pCR": ([1, 0, 1, 0, 1, 0] if flip_pcr else [0, 1, 0, 1, 0, 1]),
        }
    )


def _readout_bundle():
    rng = np.random.default_rng(31)
    all_ids = np.asarray(
        ["TR0", "TR1", "TR2", "TR3", "V0", "V1", "V2", "V3", *HELDOUT_IDS]
    )
    labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    states = rng.normal(size=(len(all_ids), 3, 2)).astype(np.float32)
    states[:, :, 0] += labels[:, None] * 0.5
    return fit_fold_readout(
        states,
        labels,
        all_ids,
        range(0, 4),
        range(4, 8),
        fold=2,
        test_indices=range(8, 14),
        config=AuditReadoutConfig(
            penalties=("l2",),
            c_grid=(0.1,),
            max_iter=200,
            random_state=7,
        ),
    )


def _run(
    output_dir: Path,
    *,
    metadata: pd.DataFrame | None = None,
    labels: dict[str, int] | None = None,
    allow_relaxed_matches: bool = True,
):
    dataset = _HeldoutDataset()
    return run_matched_donor_fold_audit(
        fold=2,
        heldout_metadata=_metadata() if metadata is None else metadata,
        base_dataset=dataset,
        patient_index={
            patient_id: index for index, patient_id in enumerate(dataset.ids)
        },
        model=_TinyModel().eval(),
        readout_bundle=_readout_bundle(),
        labels_by_patient=(
            {patient_id: index % 2 for index, patient_id in enumerate(HELDOUT_IDS)}
            if labels is None
            else labels
        ),
        checkpoint="mock-fold-2.pt",
        output_dir=output_dir,
        matching_config=MatchingConfig(
            max_donors=3,
            seed=2028,
            allow_relaxed_matches=allow_relaxed_matches,
        ),
        inference_batch_size=5,
        caller_provenance={"test_fixture": "tiny"},
    )


class MatchedDonorFoldAuditTest(unittest.TestCase):
    def test_checkpoint_identity_reads_embedded_sha256(self) -> None:
        expected = "a" * 64
        identity = _checkpoint_identity(f"missing-checkpoint.pt#sha256={expected}")
        self.assertEqual(identity["sha256"], expected)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            _checkpoint_identity("missing-checkpoint.pt#sha256=invalid")

    def test_end_to_end_exports_strict_order_diagnostics_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "fold_02" / "matched_donor"
            result = _run(output_dir)

            self.assertTrue(output_dir.is_dir())
            self.assertEqual(
                set(path.name for path in output_dir.iterdir()),
                set(ARTIFACT_FILENAMES.values()),
            )
            self.assertEqual(len(result.mapping), len(HELDOUT_IDS) * 3)
            self.assertEqual(len(result.latent_diagnostics), len(result.mapping) * 3)
            self.assertEqual(len(result.predictions), len(result.mapping) * 3)
            self.assertTrue(
                result.mapping["pair_index"].eq(range(len(result.mapping))).all()
            )
            self.assertTrue(
                result.mapping.groupby("recipient_patient_id", sort=False)
                .size()
                .eq(3)
                .all()
            )
            self.assertGreater(result.mapping["treatment_family_relaxed"].sum(), 0)
            self.assertTrue(
                result.mapping["arm_relaxed"].equals(
                    result.mapping["treatment_family_relaxed"]
                )
            )
            self.assertGreater(
                result.latent_diagnostics["response_state_l2_change"].max(), 0
            )

            expected_prediction_order = [
                (
                    row.recipient_patient_id,
                    row.donor_patient_id,
                    row.audit_repetition,
                    decision_point,
                )
                for row in result.mapping.itertuples(index=False)
                for decision_point in DECISION_POINTS
            ]
            observed_prediction_order = list(
                zip(
                    result.predictions["patient_id"],
                    result.predictions["donor_patient_id"],
                    result.predictions["repetition_id"],
                    result.predictions["decision_point"],
                    strict=True,
                )
            )
            self.assertEqual(observed_prediction_order, expected_prediction_order)
            self.assertTrue(
                result.predictions["audit_condition"].eq(AUDIT_CONDITION).all()
            )

            with (output_dir / ARTIFACT_FILENAMES["provenance"]).open(
                encoding="utf-8"
            ) as stream:
                provenance = json.load(stream)
            self.assertTrue(provenance["matching"]["outcome_blind"])
            self.assertNotIn("pCR", provenance["matching"]["baseline_columns_consumed"])
            self.assertIn("pCR", provenance["matching"]["coexisting_columns_excluded"])
            self.assertEqual(
                provenance["contracts"]["target"],
                "fixed recipient native EMA target for every donor pair",
            )
            self.assertEqual(
                provenance["contracts"]["primary_readout_dependency"],
                "geometry plus condition only",
            )
            self.assertEqual(
                set(provenance["artifacts"]),
                set(ARTIFACT_FILENAMES).difference({"provenance"}),
            )

    def test_pcr_changes_cannot_affect_matching_or_probability(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = _run(root / "first", metadata=_metadata(flip_pcr=False))
            flipped_labels = {
                patient_id: 1 - index % 2
                for index, patient_id in enumerate(HELDOUT_IDS)
            }
            second = _run(
                root / "second",
                metadata=_metadata(flip_pcr=True),
                labels=flipped_labels,
            )

            assert_frame_equal(first.mapping, second.mapping)
            probability_columns = [
                "patient_id",
                "donor_patient_id",
                "repetition_id",
                "decision_point",
                "predicted_probability",
                "predicted_label",
                "threshold",
            ]
            assert_frame_equal(
                first.predictions[probability_columns],
                second.predictions[probability_columns],
            )
            np.testing.assert_array_equal(
                1 - first.predictions["y_true"].to_numpy(),
                second.predictions["y_true"].to_numpy(),
            )
            self.assertEqual(
                first.provenance["matching"]["baseline_input_sha256"],
                second.provenance["matching"]["baseline_input_sha256"],
            )

    def test_no_overwrite_and_exact_heldout_order_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "complete"
            _run(output_dir)
            with self.assertRaisesRegex(FileExistsError, "拒绝覆盖"):
                _run(output_dir)

            reordered = _metadata().iloc[::-1].reset_index(drop=True)
            with self.assertRaisesRegex(ValueError, "patient/order"):
                _run(root / "wrong-order", metadata=reordered)
            self.assertFalse((root / "wrong-order").exists())

    def test_strict_matching_records_no_relaxation_and_shortfalls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = _run(
                Path(directory) / "strict",
                allow_relaxed_matches=False,
            )
            self.assertFalse(result.mapping["matching_relaxed"].any())
            self.assertFalse(result.mapping["treatment_family_relaxed"].any())
            self.assertFalse(result.mapping["arm_relaxed"].any())
            self.assertTrue(result.recipient_diagnostics["status"].eq("partial").all())
            self.assertEqual(result.success_stats["n_any_relaxed_pairs"], 0)
            self.assertEqual(
                result.success_stats["n_treatment_family_relaxed_pairs"], 0
            )
            self.assertEqual(result.success_stats["n_arm_relaxed_pairs"], 0)


if __name__ == "__main__":
    unittest.main()

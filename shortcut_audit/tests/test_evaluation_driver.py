from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from shortcut_audit.auditlib import evaluation_driver as driver_module
from shortcut_audit.auditlib.evaluation_driver import (
    BASELINE_METADATA_COLUMNS,
    BASELINE_VOLUME_UNIT,
    OutcomeBlindHeldoutMemoryDataset,
    evaluate_retrained_fold_b_to_f,
    prepare_heldout_donor_data,
)
from shortcut_audit.scripts import evaluate_retrained_fold as cli_module


class _TripwireRecord:
    """Raises if metadata construction ever reaches the held-out outcome."""

    def __init__(
        self,
        patient_id: str,
        *,
        arm: str,
        hr: int,
        her2: int,
        mp: int,
        age: float,
    ) -> None:
        self.patient_id = patient_id
        self.cohort = "ispy2"
        self.arm = arm
        self.hr = hr
        self.her2 = her2
        self.mp = mp
        self.age = age

    @property
    def pcr(self):
        raise AssertionError("donor metadata 不得读取 pCR")


def _item(patient_id: str, voxel_count: int) -> dict[str, object]:
    image = torch.zeros((4, 8, 2, 2, 2), dtype=torch.float32)
    image[0, 7].reshape(-1)[:voxel_count] = 1.0
    geometry = torch.zeros((4, 9), dtype=torch.float32)
    geometry[0, 0] = voxel_count / 8.0
    condition = torch.zeros((3, 5), dtype=torch.float32)
    return {
        "patient_id": patient_id,
        "image": image,
        "geometry": geometry,
        "condition": condition,
        "routing_target": torch.tensor(0),
    }


class _CountingDataset(Dataset):
    def __init__(self) -> None:
        self.items = [
            _item("B", 2),
            _item("UNUSED", 1),
            _item("A", 1),
            _item("C", 3),
        ]
        self.accessed: list[int] = []

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        self.accessed.append(int(index))
        return self.items[int(index)]


def _context():
    dataset = _CountingDataset()
    records = (
        _TripwireRecord("A", arm="Paclitaxel", hr=1, her2=0, mp=1, age=40.0),
        _TripwireRecord(
            "B", arm="Paclitaxel + Pembrolizumab", hr=0, her2=0, mp=0, age=50.0
        ),
        _TripwireRecord(
            "C", arm="Paclitaxel + Trastuzumab", hr=0, her2=1, mp=1, age=np.nan
        ),
    )
    context = SimpleNamespace(
        fold=2,
        checkpoint="fold-02.pt#sha256=abc",
        model=torch.nn.Identity().eval(),
        readout_bundle=SimpleNamespace(test_patient_ids=("A", "B", "C")),
        base_dataset=dataset,
        patient_index={"A": 2, "B": 0, "C": 3},
        heldout_records=records,
        labels_by_patient={"A": 0, "B": 1, "C": 0},
        device=torch.device("cpu"),
    )
    return context, dataset


class HeldoutMetadataTest(unittest.TestCase):
    def test_metadata_is_baseline_only_ordered_and_materialized_once(self) -> None:
        context, source = _context()
        prepared = prepare_heldout_donor_data(context)

        self.assertEqual(tuple(prepared.metadata.columns), BASELINE_METADATA_COLUMNS)
        self.assertEqual(tuple(prepared.metadata["patient_id"]), ("A", "B", "C"))
        self.assertEqual(
            prepared.metadata["baseline_lesion_volume"].tolist(), [1, 2, 3]
        )
        self.assertTrue(
            prepared.metadata["baseline_lesion_volume_unit"]
            .eq(BASELINE_VOLUME_UNIT)
            .all()
        )
        self.assertEqual(
            prepared.metadata["treatment_family"].tolist(),
            ["taxane", "io", "her2_targeted"],
        )
        self.assertEqual(source.accessed, [2, 0, 3])
        self.assertEqual(prepared.patient_index, {"A": 0, "B": 1, "C": 2})
        self.assertEqual(prepared.source_patient_index, context.patient_index)
        self.assertIsInstance(prepared.dataset, OutcomeBlindHeldoutMemoryDataset)
        self.assertEqual(
            set(prepared.dataset[0]),
            {"patient_id", "image", "geometry", "condition"},
        )
        forbidden = {"pcr", "label_pcr", "outcome", "y_true", "target"}
        self.assertFalse(forbidden.intersection(prepared.metadata.columns))

    def test_order_or_geometry_drift_fails_closed(self) -> None:
        context, _ = _context()
        context.readout_bundle.test_patient_ids = ("B", "A", "C")
        with self.assertRaisesRegex(ValueError, "patient/order"):
            prepare_heldout_donor_data(context)

        context, source = _context()
        source.items[2]["geometry"][0, 0] = 0.99
        with self.assertRaisesRegex(ValueError, "ROI voxel count"):
            prepare_heldout_donor_data(context)


class DriverOrchestrationTest(unittest.TestCase):
    def test_calls_core_then_strict_e_and_passes_outcome_free_memory_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation_output = root / "evaluation"
            donor_output = root / "donor"
            context, source = _context()
            events: list[str] = []

            def core_side_effect(*args, **kwargs):
                events.append("core")
                Path(kwargs["output_dir"]).mkdir(parents=True)
                native = Path(kwargs["output_dir"]) / "predictions" / "native.csv"
                return SimpleNamespace(
                    fold=2,
                    output_dir=Path(kwargs["output_dir"]),
                    donor_context=context,
                    artifact_paths={"predictions_native": native},
                )

            donor_result = SimpleNamespace(
                output_dir=donor_output,
                mapping=pd.DataFrame({"pair": [0]}),
                predictions=pd.DataFrame({"prediction": [0]}),
            )

            def donor_side_effect(**kwargs):
                events.append("donor")
                self.assertEqual(
                    tuple(kwargs["heldout_metadata"].columns), BASELINE_METADATA_COLUMNS
                )
                self.assertNotIn("pcr", kwargs["heldout_metadata"].columns)
                self.assertIsInstance(
                    kwargs["base_dataset"], OutcomeBlindHeldoutMemoryDataset
                )
                self.assertEqual(kwargs["patient_index"], {"A": 0, "B": 1, "C": 2})
                self.assertIs(kwargs["labels_by_patient"], context.labels_by_patient)
                self.assertEqual(kwargs["matching_config"].max_donors, 10)
                self.assertFalse(kwargs["matching_config"].allow_relaxed_matches)
                self.assertEqual(kwargs["output_dir"], donor_output.resolve())
                self.assertEqual(
                    kwargs["caller_provenance"]["outcome_columns_in_matching_metadata"],
                    [],
                )
                return donor_result

            with mock.patch.object(
                driver_module,
                "evaluate_retrained_fold",
                side_effect=core_side_effect,
            ) as core_mock, mock.patch.object(
                driver_module,
                "run_matched_donor_fold_audit",
                side_effect=donor_side_effect,
            ) as donor_mock:
                result = evaluate_retrained_fold_b_to_f(
                    root / "fold_02",
                    fold=2,
                    legacy_x_cache_dir=root / "cache",
                    evaluation_output_dir=evaluation_output,
                    donor_output_dir=donor_output,
                    device="cpu",
                    batch_size=8,
                    workers=0,
                )

            self.assertEqual(events, ["core", "donor"])
            self.assertEqual(source.accessed, [2, 0, 3])
            self.assertEqual(result.fold, 2)
            self.assertIs(result.donor, donor_result)
            core_mock.assert_called_once()
            donor_mock.assert_called_once()
            self.assertEqual(
                Path(core_mock.call_args.kwargs["output_dir"]),
                evaluation_output.resolve(),
            )

    def test_no_overwrite_or_nested_outputs_before_core_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            donor_output = root / "donor"
            donor_output.mkdir()
            with mock.patch.object(
                driver_module, "evaluate_retrained_fold"
            ) as core_mock, self.assertRaisesRegex(FileExistsError, "donor"):
                evaluate_retrained_fold_b_to_f(
                    root / "fold",
                    fold=0,
                    legacy_x_cache_dir=root / "cache",
                    evaluation_output_dir=root / "evaluation",
                    donor_output_dir=donor_output,
                )
            core_mock.assert_not_called()

            with mock.patch.object(
                driver_module, "evaluate_retrained_fold"
            ) as core_mock, self.assertRaisesRegex(ValueError, "互不嵌套"):
                evaluate_retrained_fold_b_to_f(
                    root / "fold",
                    fold=0,
                    legacy_x_cache_dir=root / "cache",
                    evaluation_output_dir=root / "new-evaluation",
                    donor_output_dir=root / "new-evaluation" / "donor",
                )
            core_mock.assert_not_called()


class EvaluationCliTest(unittest.TestCase):
    @staticmethod
    def _arguments(root: Path) -> list[str]:
        return [
            "--fold",
            "2",
            "--gpu",
            "1",
            "--fold-dir",
            str(root / "fold_02"),
            "--eval-output",
            str(root / "evaluation"),
            "--donor-output",
            str(root / "donor"),
            "--cache",
            str(root / "cache"),
        ]

    def test_cli_requires_explicit_authorization_before_git_or_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            cli_module, "_git"
        ) as git_mock, redirect_stderr(io.StringIO()), self.assertRaises(
            SystemExit
        ) as raised:
            cli_module.main(self._arguments(Path(directory)))
        self.assertEqual(raised.exception.code, 2)
        git_mock.assert_not_called()

    def test_cli_dispatches_formal_cuda_driver(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            combined = SimpleNamespace(
                fold=2,
                evaluation=SimpleNamespace(output_dir=root / "evaluation"),
                donor=SimpleNamespace(
                    output_dir=root / "donor",
                    mapping=pd.DataFrame({"pair": [0, 1]}),
                    predictions=pd.DataFrame({"row": range(6)}),
                ),
                donor_metadata=pd.DataFrame({"patient_id": ["A", "B"]}),
            )
            with mock.patch.object(
                cli_module,
                "_git",
                side_effect=[cli_module.EXPECTED_BRANCH, cli_module.EXPECTED_COMMIT],
            ), mock.patch.object(
                driver_module,
                "evaluate_retrained_fold_b_to_f",
                return_value=combined,
            ) as run_mock, redirect_stdout(
                io.StringIO()
            ) as stdout:
                exit_code = cli_module.main(
                    [*self._arguments(root), "--allow-evaluation"]
                )

            self.assertEqual(exit_code, 0)
            run_mock.assert_called_once()
            self.assertEqual(run_mock.call_args.kwargs["fold"], 2)
            self.assertEqual(run_mock.call_args.kwargs["device"], "cuda:1")
            self.assertIn('"status": "complete"', stdout.getvalue())
            self.assertIn('"donor_pairs": 2', stdout.getvalue())


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from dataclasses import asdict, replace
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import stat
import sys
import tempfile
import unittest
from unittest import mock

import torch


sys.dont_write_bytecode = True
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = EXPERIMENT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lg_response_pilot import matrix, training  # noqa: E402


EXPECTED_SEALED_SOURCE_SHA256 = {
    "__init__.py": "2976c6d040c506b4f1b1db5374718d1c3edf341805d0e6a9d176f0c02fa37a47",
    "contracts.py": "48d7738b6764780ba2e784f826be44ac718fdbb0beb526ec31c3c5525cba4bf9",
    "data.py": "948a25aa00eeaf68a11f5a0bcf7c4d0c7592786a36ebb9bce472361745eebb59",
    "gate.py": "babb748a71eba0c36d802a8e15c861387d506de67cf754ec58ae96f1d3341555",
    "inputs.py": "40965e509afa059ce2674c7a7fde18cd9097e1eb00d05021738d8cf9f6346177",
    "targets.py": "06434db46cf76e6f39ff6eb1c476933885e90ed0a4c952dcc0a3477a25996c7b",
    "training.py": "2edf546628e447bdd1b9715f60f105d1a5952763bd782aabddbae298fae62f52",
    "upstream.py": "dfc03ab80590d1b57240a8ce210c75245bce4dd3bad9a4d655d8d63a1f96d54f",
}


def epoch_row(
    epoch: int,
    *,
    state: float,
    ftv: float,
    representation_std: float = 0.10,
    grounded_patients: int = 8,
) -> dict[str, object]:
    return {
        "epoch": epoch,
        "val_loss": state + ftv,
        "val_base_objective": state,
        "val_state_loss": state,
        "val_ftv_loss": ftv,
        "val_grounded_patients": grounded_patients,
        "val_representation_std": representation_std,
        "finite": True,
        # These deliberately disagree across rows.  The frozen selector must
        # remain blind to every downstream/test-only field.
        "test_ftv": 1000.0 - epoch,
        "delta_ftv": -1000.0 + epoch,
        "pcr": float(epoch),
    }


class SealedHelperTest(unittest.TestCase):
    def test_exact_source_hash_inventory_and_import_identity(self) -> None:
        self.assertEqual(dict(training.SEALED_SOURCE_SHA256), EXPECTED_SEALED_SOURCE_SHA256)
        self.assertEqual(training.verify_sealed_stage_b_sources(), EXPECTED_SEALED_SOURCE_SHA256)

        import c1b_stage_b.training as sealed

        for name in (
            "logical_patient_batches",
            "logical_sigreg_surrogate",
            "physical_patient_batches",
            "run_logical_train_epoch",
            "run_validation_epoch",
            "scale_microbatch_components",
            "select_checkpoint",
        ):
            self.assertIs(getattr(training, name), getattr(sealed, name), name)

    def test_formal_batch_and_optimizer_contract_is_closed(self) -> None:
        self.assertEqual(
            asdict(training.formal_hyperparameters()),
            dict(training.FORMAL_HYPERPARAMETERS),
        )
        self.assertEqual(training.EFFECTIVE_BATCH_SIZE, 32)
        self.assertEqual(training.FORMAL_HYPERPARAMETERS["physical_batch_size"], 4)
        self.assertEqual(training.FORMAL_HYPERPARAMETERS["accumulation_steps"], 8)
        self.assertEqual(training.SEED_BASES, (2026, 3026, 4026, 5026, 6026))
        self.assertEqual(training.FOLDS, tuple(range(5)))
        self.assertEqual(training.validate_seed_fold(6026, 4), 6030)
        for invalid in (2026.0, True, 7026):
            with self.subTest(invalid_seed=invalid):
                with self.assertRaises(ValueError):
                    training.validate_seed_fold(invalid, 0)  # type: ignore[arg-type]


class CheckpointSelectionTest(unittest.TestCase):
    def test_no_grounding_uses_validation_state_only(self) -> None:
        rows = [
            epoch_row(1, state=0.90, ftv=0.90),
            epoch_row(2, state=0.80, ftv=9.00),
            epoch_row(3, state=0.85, ftv=0.01),
        ]
        selected = training.select_checkpoint(
            rows, grounded=False, min_representation_std=0.05
        )
        self.assertEqual(selected["selected_epoch"], 2)
        self.assertEqual(selected["selection_mode"], "primary")
        self.assertTrue(selected["experiment_pass"])
        self.assertFalse(selected["test_data_used"])

    def test_grounded_uses_ftv_only_inside_paired_state_safety_gate(self) -> None:
        rows = [
            epoch_row(1, state=0.95, ftv=0.40),
            epoch_row(2, state=1.04, ftv=0.20),
            epoch_row(3, state=1.06, ftv=0.01),
        ]
        selected = training.select_checkpoint(
            rows,
            grounded=True,
            min_representation_std=0.05,
            paired_baseline_state_loss=1.0,
        )
        self.assertEqual(selected["selected_epoch"], 2)
        self.assertEqual(selected["allowed_state_loss"], 1.05)
        self.assertEqual(selected["selection_mode"], "primary")
        self.assertTrue(selected["optimization_safety_pass"])

    def test_grounded_fallback_is_marked_failed(self) -> None:
        rows = [
            epoch_row(1, state=1.08, ftv=0.10),
            epoch_row(2, state=1.06, ftv=0.50),
        ]
        selected = training.select_checkpoint(
            rows,
            grounded=True,
            min_representation_std=0.05,
            paired_baseline_state_loss=1.0,
        )
        self.assertEqual(selected["selected_epoch"], 2)
        self.assertEqual(selected["selection_mode"], "fallback_base_gate_failed")
        self.assertFalse(selected["experiment_pass"])
        self.assertFalse(selected["optimization_safety_pass"])


class MatrixTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "formal"
        self.groups = matrix.build_matrix_groups(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_exact_inventory_order_pairing_and_three_gpu_assignment(self) -> None:
        self.assertEqual(
            matrix.FORMAL_ARM_ORDER, ("GAP0", "GAP3", "LOCAL0", "LOCAL3")
        )
        self.assertEqual(len(self.groups), 25)
        cells = [cell for group in self.groups for cell in group.cells]
        self.assertEqual(len(cells), 100)
        self.assertEqual(
            len({(cell.seed_base, cell.fold, cell.arm) for cell in cells}), 100
        )
        self.assertEqual(
            [(group.seed_base, group.fold) for group in self.groups],
            [
                (seed, fold)
                for seed in (2026, 3026, 4026, 5026, 6026)
                for fold in range(5)
            ],
        )
        self.assertEqual(
            [
                sum(group.device == device for group in self.groups)
                for device in matrix.FORMAL_DEVICES
            ],
            [9, 8, 8],
        )
        for group in self.groups:
            self.assertEqual(tuple(cell.arm for cell in group.cells), matrix.FORMAL_ARM_ORDER)
            by_arm = {cell.arm: cell for cell in group.cells}
            for grounded, baseline in training.BASELINE_BY_GROUNDED.items():
                self.assertLess(
                    matrix.FORMAL_ARM_ORDER.index(baseline),
                    matrix.FORMAL_ARM_ORDER.index(grounded),
                )
                self.assertEqual(
                    by_arm[grounded].paired_baseline_selection,
                    by_arm[baseline].output_dir / "selection.json",
                )
            for baseline in ("GAP0", "LOCAL0"):
                self.assertIsNone(by_arm[baseline].paired_baseline_selection)

    def test_matrix_validator_rejects_order_pairing_and_device_drift(self) -> None:
        first = self.groups[0]
        reordered = replace(first, cells=tuple(reversed(first.cells)))
        with self.assertRaisesRegex(ValueError, "arm order"):
            matrix.validate_matrix_groups((reordered, *self.groups[1:]), matrix.FORMAL_DEVICES)

        local3 = first.cells[3]
        wrong_pair = replace(
            local3,
            paired_baseline_selection=first.cells[0].output_dir / "selection.json",
        )
        wrong_cells = (*first.cells[:3], wrong_pair)
        with self.assertRaisesRegex(ValueError, "paired baseline"):
            matrix.validate_matrix_groups(
                (replace(first, cells=wrong_cells), *self.groups[1:]),
                matrix.FORMAL_DEVICES,
            )

        wrong_device = replace(first, device="cuda:2")
        with self.assertRaisesRegex(ValueError, "device assignment"):
            matrix.validate_matrix_groups(
                (wrong_device, *self.groups[1:]), matrix.FORMAL_DEVICES
            )

    def test_scheduler_preserves_each_gpu_sequential_stream(self) -> None:
        observed = {device: [] for device in matrix.FORMAL_DEVICES}

        def run_cell(cell: matrix.MatrixCell) -> None:
            observed[cell.device].append((cell.seed_base, cell.fold, cell.arm))

        completed = matrix.execute_matrix_groups(
            self.groups, matrix.FORMAL_DEVICES, run_cell
        )
        self.assertEqual(len(completed), matrix.FORMAL_CELL_COUNT)
        for device in matrix.FORMAL_DEVICES:
            expected = [
                (cell.seed_base, cell.fold, cell.arm)
                for group in self.groups
                if group.device == device
                for cell in group.cells
            ]
            self.assertEqual(observed[device], expected)

    def test_command_adds_baseline_only_for_grounded_cell(self) -> None:
        baseline, grounded = self.groups[0].cells[:2]
        common = {
            "python_executable": "/python",
            "train_script": "/train_cell.py",
            "stage_a_sentinel": "/stage_a.json",
            "stage_a_sentinel_sha256": "a" * 64,
            "data_contract": "/data_contract.json",
            "data_contract_sha256": "b" * 64,
            "preregistration_lock_sha256": "c" * 64,
        }
        baseline_command = matrix.build_train_command(baseline, **common)
        grounded_command = matrix.build_train_command(grounded, **common)
        self.assertNotIn("--paired-baseline-selection", baseline_command)
        flag = grounded_command.index("--paired-baseline-selection")
        self.assertEqual(
            grounded_command[flag + 1], str(grounded.paired_baseline_selection)
        )
        sentinel_hash_flag = grounded_command.index("--stage-a-sentinel-sha256")
        self.assertEqual(grounded_command[sentinel_hash_flag + 1], "a" * 64)
        lock_hash_flag = grounded_command.index("--preregistration-lock-sha256")
        self.assertEqual(grounded_command[lock_hash_flag + 1], "c" * 64)


class BaselineAndPrivacyTest(unittest.TestCase):
    def test_each_grounded_arm_accepts_only_its_matching_baseline(self) -> None:
        hyperparameters = training.formal_hyperparameters()
        common = {
            "seed_base": 2026,
            "fold": 2,
            "effective_seed": 2028,
            "paired_initialization_sha256": "i" * 64,
            "hyperparameters": hyperparameters,
            "train_patient_sha256": "t" * 64,
            "val_patient_sha256": "v" * 64,
            "data_provenance_sha256": "d" * 64,
            "preregistration_lock_sha256": "l" * 64,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "selection.json"
            for grounded, baseline in training.BASELINE_BY_GROUNDED.items():
                payload = {
                    "arm": baseline,
                    "seed_base": 2026,
                    "fold": 2,
                    "effective_seed": 2028,
                    "selection_mode": "primary",
                    "experiment_pass": True,
                    "test_data_used": False,
                    "paired_initialization_sha256": "i" * 64,
                    "hyperparameters": asdict(hyperparameters),
                    "train_patient_sha256": "t" * 64,
                    "val_patient_sha256": "v" * 64,
                    "data_provenance_sha256": "d" * 64,
                    "preregistration_status": "PASS",
                    "preregistration_lock_sha256": "l" * 64,
                    "preregistration": {
                        "status": "PASS",
                        "lock_sha256": "l" * 64,
                    },
                    "selected_validation_state_loss": 1.0,
                }
                path.write_text(json.dumps(payload), encoding="utf-8")
                metric, observed = training.validate_paired_baseline(
                    path, grounded_arm=grounded, **common
                )
                self.assertEqual(metric, 1.0)
                self.assertEqual(observed["arm"], baseline)
                wrong = next(
                    value
                    for value in ("GAP0", "LOCAL0")
                    if value != baseline
                )
                payload["arm"] = wrong
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "arm mismatch"):
                    training.validate_paired_baseline(
                        path, grounded_arm=grounded, **common
                    )
                payload["arm"] = baseline
                payload["preregistration"]["lock_sha256"] = "z" * 64
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(
                    ValueError, "preregistration lock mismatch"
                ):
                    training.validate_paired_baseline(
                        path, grounded_arm=grounded, **common
                    )

    def test_private_training_writers_use_owner_only_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            json_path = root / "selection.json"
            history_path = root / "history.csv"
            checkpoint_path = root / "selected.pt"
            training._atomic_json(json_path, {"status": "ok"})
            training._write_history(history_path, [{"epoch": 1, "loss": 1.0}])
            training._atomic_torch_save(checkpoint_path, {"epoch": 1})
            for path in (json_path, history_path, checkpoint_path):
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600, path.name)

    def test_selection_contains_postprocessing_compatible_lock_binding(self) -> None:
        class TinyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))

            def model_config(self) -> dict[str, object]:
                return {"arm": "GAP0"}

            def architecture_contract(self) -> dict[str, object]:
                return {"architecture": "GAP"}

            def parameter_counts(self) -> dict[str, int]:
                return {"trainable_total": 1}

        class TinyDataset:
            def __init__(self, patient_ids: tuple[str, ...]) -> None:
                self.patient_ids = patient_ids

            def __len__(self) -> int:
                return len(self.patient_ids)

        patients = tuple(f"patient_{index:02d}" for index in range(32))
        dataset = TinyDataset(patients)
        authorization = SimpleNamespace(path=Path("sentinel.json"), sha256="s" * 64)
        train_stats = {
            "loss": 1.0,
            "base_loss": 1.0,
            "state_loss": 0.9,
            "ftv_loss": 0.0,
            "grounded_patients": 0.0,
            "representation_std": 0.10,
            "optimizer_steps": 1.0,
        }
        val_stats = {
            "loss": 1.0,
            "base_loss": 1.0,
            "state_loss": 0.9,
            "ftv_loss": 0.0,
            "grounded_patients": 0.0,
            "representation_std": 0.10,
        }
        lock = {"status": "PASS", "lock_sha256": "l" * 64}
        spec = SimpleNamespace(name="GAP0", architecture="GAP", grounded=False)
        with tempfile.TemporaryDirectory() as temporary, mock.patch.multiple(
            training,
            arm_spec=mock.Mock(return_value=spec),
            validate_model_contract=mock.Mock(return_value=None),
            shared_initialization_sha256=mock.Mock(return_value="i" * 64),
            transition_sha256=mock.Mock(return_value="x" * 64),
            logical_patient_batches=mock.Mock(return_value=(patients,)),
            run_logical_train_epoch=mock.Mock(return_value=train_stats),
            run_validation_epoch=mock.Mock(return_value=val_stats),
        ):
            output = Path(temporary) / "cell"
            selection = training.train_epochs(
                arm="GAP0",
                seed_base=2026,
                fold=0,
                model=TinyModel(),
                objective=torch.nn.Identity(),
                train_dataset=dataset,
                val_dataset=dataset,
                device=torch.device("cpu"),
                output_dir=output,
                authorization=authorization,
                hyperparameters=training.formal_hyperparameters(),
                paired_initialization_sha256="i" * 64,
                data_provenance={"schema_version": 1},
                preregistration=lock,
            )
            self.assertEqual(selection["preregistration"], lock)
            stored = json.loads((output / "selection.json").read_text())
            self.assertEqual(stored["preregistration"], lock)
            selected = torch.load(
                output / "selected.pt", map_location="cpu", weights_only=True
            )
            self.assertEqual(selected["selection"]["preregistration"], lock)

    def test_private_input_root_changes_physical_path_but_not_lock_key(self) -> None:
        relative = Path(
            "additional_experiments/c1b_overlap_eligibility_ftv_stageb/"
            "manifests/stage_b_data_contract.private.json"
        )
        scripts = EXPERIMENT_ROOT / "scripts"
        with tempfile.TemporaryDirectory() as temporary:
            private_root = Path(temporary) / "private-input-repository"
            with mock.patch.dict(
                os.environ,
                {"MWM_PRIVATE_INPUT_REPO_ROOT": str(private_root)},
            ):
                for script_name in ("train_cell.py", "run_matrix.py"):
                    module_name = f"_confirmation_test_{script_name[:-3]}"
                    spec = importlib.util.spec_from_file_location(
                        module_name, scripts / script_name
                    )
                    assert spec is not None and spec.loader is not None
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    try:
                        spec.loader.exec_module(module)
                        self.assertEqual(
                            module.DEFAULT_DATA_CONTRACT,
                            (private_root / relative).resolve(),
                        )
                        self.assertEqual(
                            module.DATA_CONTRACT_LOCK_KEY, relative.as_posix()
                        )
                        self.assertEqual(
                            module._resolve_default_data_contract({}),
                            (module.REPO_ROOT / relative).resolve(),
                        )
                        if script_name == "run_matrix.py":
                            with mock.patch.multiple(
                                torch.cuda,
                                is_available=mock.Mock(return_value=True),
                                device_count=mock.Mock(return_value=3),
                                get_device_name=mock.Mock(
                                    side_effect=lambda index: f"GPU-{index}"
                                ),
                                get_device_capability=mock.Mock(
                                    return_value=(9, 0)
                                ),
                            ):
                                runtime = module._runtime_provenance(
                                    ("cuda:0", "cuda:1", "cuda:2")
                                )
                            self.assertEqual(runtime["torch_version"], str(torch.__version__))
                            self.assertEqual(
                                [gpu["name"] for gpu in runtime["requested_gpus"]],
                                ["GPU-0", "GPU-1", "GPU-2"],
                            )
                    finally:
                        sys.modules.pop(module_name, None)

    def test_entrypoints_disable_bytecode_before_local_imports(self) -> None:
        for name in ("train_cell.py", "run_matrix.py"):
            source = (EXPERIMENT_ROOT / "scripts" / name).read_text(encoding="utf-8")
            disable = source.index("sys.dont_write_bytecode = True")
            local_import = min(
                position
                for marker in ("from freeze_preregistration", "from lg_response_pilot")
                if (position := source.index(marker)) >= 0
            )
            self.assertLess(disable, local_import, name)
            main_start = source.index("def main() -> None:")
            lock_check = source.index("verify_preregistration()", main_start)
            stage_a_read = source.index(
                "args.stage_a_sentinel = require_canonical_file(", main_start
            )
            self.assertLess(lock_check, stage_a_read, name)
            first_pilot_import = source.index("import lg_response_pilot", main_start)
            self.assertLess(lock_check, first_pilot_import, name)


if __name__ == "__main__":
    unittest.main()

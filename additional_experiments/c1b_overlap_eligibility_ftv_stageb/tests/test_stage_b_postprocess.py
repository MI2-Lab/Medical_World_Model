from __future__ import annotations

import json
import importlib.util
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c1b_stage_b.contracts import file_sha256  # noqa: E402
from c1b_stage_b.gate import StageAAuthorization  # noqa: E402
from c1b_stage_b.postprocess import (  # noqa: E402
    FORMAL_DEVICES,
    FORMAL_HYPERPARAMETERS,
    build_feature_command,
    build_postprocess_cells,
    build_probe_command,
    validate_training_matrix,
)


def _load_driver_module():
    path = ROOT / "scripts" / "run_stage_b_postprocessing.py"
    spec = importlib.util.spec_from_file_location("stage_b_postprocess_driver_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the postprocessing driver")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FormalPostprocessTests(unittest.TestCase):
    def test_selection_history_inventory_helpers_detect_live_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cell = build_postprocess_cells(
                root / "checkpoints" / "formal_4x8_restart1",
                root / "features" / "formal_4x8_restart1",
                root / "predictions" / "formal_4x8_restart1",
            )[0]
            cell.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            cell.selection_path.write_text('{"selected_epoch": 1}\n', encoding="utf-8")
            cell.history_path.write_text("epoch,val_loss\n1,1.0\n", encoding="utf-8")
            driver = _load_driver_module()

            inventory = driver._hash_selection_history_inventory((cell,))
            key = f"seed_{cell.seed_base}/{cell.arm}/fold_{cell.fold}"
            self.assertEqual(
                inventory,
                {
                    key: {
                        "selection_sha256": file_sha256(cell.selection_path),
                        "history_sha256": file_sha256(cell.history_path),
                    }
                },
            )
            driver._require_selection_history_unchanged((cell,), inventory)

            cell.selection_path.write_text('{"selected_epoch": 2}\n', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "changed after preflight"):
                driver._require_selection_history_unchanged((cell,), inventory)

            cell.selection_path.write_text('{"selected_epoch": 1}\n', encoding="utf-8")
            cell.history_path.write_text("epoch,val_loss\n1,9.0\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "changed after preflight"):
                driver._require_selection_history_unchanged((cell,), inventory)

    def test_feature_fail_fast_attributes_the_originating_child(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cells = build_postprocess_cells(
                root / "checkpoints" / "formal_4x8_restart1",
                root / "features" / "formal_4x8_restart1",
                root / "predictions" / "formal_4x8_restart1",
            )[:3]
            driver = _load_driver_module()

            def command(cell):
                if cell.index == 0:
                    return (sys.executable, "-c", "raise SystemExit(7)")
                return (sys.executable, "-c", "import time; time.sleep(10)")

            identity = (
                f"seed={cells[0].seed_base}, fold={cells[0].fold}, "
                f"arm={cells[0].arm}, device={cells[0].device}"
            )
            with self.assertRaisesRegex(driver.PostprocessExecutionError, identity):
                driver._execute_feature_exports(cells, FORMAL_DEVICES, command)

    def test_formal_run_claim_is_exclusive_and_persistent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            claim = Path(directory) / "postprocessing_claim.json"
            driver = _load_driver_module()
            driver._claim_formal_run(claim, {"status": "CLAIMED"})
            self.assertEqual(
                json.loads(claim.read_text(encoding="utf-8")),
                {"status": "CLAIMED"},
            )
            with self.assertRaisesRegex(FileExistsError, "already claimed"):
                driver._claim_formal_run(claim, {"status": "SECOND"})
            self.assertEqual(
                json.loads(claim.read_text(encoding="utf-8")),
                {"status": "CLAIMED"},
            )

    def test_parent_interrupt_aborts_detached_feature_children_promptly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cells = build_postprocess_cells(
                root / "checkpoints" / "formal_4x8_restart1",
                root / "features" / "formal_4x8_restart1",
                root / "predictions" / "formal_4x8_restart1",
            )[:3]
            driver = _load_driver_module()
            timer = threading.Timer(
                0.1, lambda: os.kill(os.getpid(), signal.SIGINT)
            )
            started = time.monotonic()
            timer.start()
            try:
                with self.assertRaisesRegex(
                    driver.PostprocessExecutionError, "received SIGINT"
                ):
                    driver._execute_feature_exports(
                        cells,
                        FORMAL_DEVICES,
                        lambda _cell: (
                            sys.executable,
                            "-c",
                            "import time; time.sleep(10)",
                        ),
                    )
            finally:
                timer.cancel()
            self.assertLess(time.monotonic() - started, 2.0)

    def test_exact_forty_cell_paths_devices_and_child_commands(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cells = build_postprocess_cells(
                root / "checkpoints" / "formal_4x8_restart1",
                root / "features" / "formal_4x8_restart1",
                root / "predictions" / "formal_4x8_restart1",
            )
            self.assertEqual(len(cells), 40)
            self.assertEqual(
                len({(cell.seed_base, cell.fold, cell.arm) for cell in cells}), 40
            )
            self.assertEqual(
                [sum(cell.device == device for cell in cells) for device in FORMAL_DEVICES],
                [14, 13, 13],
            )
            first = cells[0]
            self.assertEqual(first.feature_path.name, "response_state.private.npz")
            self.assertEqual(
                first.feature_metadata_path.name,
                "response_state.private.metadata.json",
            )
            self.assertEqual(first.probe_metadata_path.name, "probe_metadata.json")
            feature = build_feature_command(
                first,
                python_executable="/python",
                export_script="/export.py",
                stage_a_sentinel="/go.json",
                data_contract="/data.json",
                data_contract_sha256="a" * 64,
            )
            self.assertEqual(feature[0:2], ("/python", "/export.py"))
            self.assertEqual(feature[feature.index("--device") + 1], "cuda:0")
            self.assertEqual(feature[feature.index("--batch-size") + 1], "4")
            self.assertEqual(feature[feature.index("--workers") + 1], "2")
            probe = build_probe_command(
                first,
                python_executable="/python",
                probe_script="/probe.py",
                stage_a_sentinel="/go.json",
                data_contract="/data.json",
                data_contract_sha256="a" * 64,
            )
            self.assertEqual(probe[probe.index("--features") + 1], str(first.feature_path))
            self.assertEqual(probe[probe.index("--output-dir") + 1], str(first.probe_dir))

    def test_matrix_gate_requires_exact_complete_hashed_forty_cell_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_root = root / "checkpoints" / "formal_4x8_restart1"
            cells = build_postprocess_cells(
                checkpoint_root,
                root / "features" / "formal_4x8_restart1",
                root / "predictions" / "formal_4x8_restart1",
            )
            authorization = StageAAuthorization(
                root / "STAGE_A_GO.json",
                "b" * 64,
                {},
                10,
                40,
                "c" * 64,
            )
            runs = []
            for cell in cells:
                cell.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                cell.history_path.write_text("epoch,val_loss\n1,1.0\n", encoding="utf-8")
                cell.checkpoint_path.write_bytes(b"synthetic selected checkpoint")
                selection = {
                    "schema_version": 1,
                    "selection_mode": "primary",
                    "arm": cell.arm,
                    "seed_base": cell.seed_base,
                    "fold": cell.fold,
                    "effective_seed": cell.seed_base + cell.fold,
                    "test_data_used": False,
                    "stage_a_sentinel_sha256": authorization.sha256,
                    "global_fallback_restart": False,
                    "finite_status": True,
                    "selected_representation_std": 0.1,
                    "hyperparameters": dict(FORMAL_HYPERPARAMETERS),
                    "history_sha256": file_sha256(cell.history_path),
                }
                cell.selection_path.write_text(
                    json.dumps(selection), encoding="utf-8"
                )
                runs.append(
                    {
                        "seed_base": cell.seed_base,
                        "fold": cell.fold,
                        "arm": cell.arm,
                        "selection_path": str(cell.selection_path),
                    }
                )
            completion = {
                "schema_version": 1,
                "status": "COMPLETE",
                "run_count": 40,
                "stage_a_sentinel_sha256": authorization.sha256,
                "batch_contract": {
                    "effective": 32,
                    "physical": 4,
                    "accumulation": 8,
                    "global_for_all_arms": True,
                },
                "runs": runs,
            }
            (checkpoint_root / "matrix_complete.json").write_text(
                json.dumps(completion), encoding="utf-8"
            )
            result = validate_training_matrix(
                checkpoint_root, cells, authorization
            )
            self.assertEqual(result["run_count"], 40)
            self.assertEqual(result["batch_contract"]["physical"], 4)
            cells[-1].checkpoint_path.unlink()
            with self.assertRaisesRegex(ValueError, "selected checkpoint is missing"):
                validate_training_matrix(checkpoint_root, cells, authorization)


if __name__ == "__main__":
    unittest.main()

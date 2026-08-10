from __future__ import annotations

from dataclasses import dataclass
from contextlib import redirect_stderr
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c1b_spatial_audit.contracts import (  # noqa: E402
    TIMEPOINTS,
    UPSTREAM_ROOT,
    cell_key,
    cells,
    file_sha256,
)
from c1b_spatial_audit.probe_runner import (  # noqa: E402
    EXPORTER_METADATA_FIELDS,
    FORMAL_POOLINGS,
    NUISANCE_POOLINGS,
    NUISANCE_TARGETS,
    ProbeCellKey,
    ProbeMatrixPlan,
    discover_feature_matrix,
    execute_probe_cell,
    execute_probe_plan,
    expected_feature_dimension,
    expected_feature_keys,
    expected_sidecar_keys,
    feature_path_for,
    load_nuisance_targets,
    output_path_for,
    parse_poolings,
    split_order_sha256,
    validate_alternative_gates,
    validate_exporter_completion,
    validate_stage_poolings,
)
from c1b_spatial_audit.probes import ordered_patient_sha256  # noqa: E402
from c1b_spatial_audit.sidecars import NUISANCE_COLUMNS  # noqa: E402


@dataclass(frozen=True)
class SyntheticFTVRecord:
    values: np.ndarray
    measurement_valid: np.ndarray
    observable: np.ndarray

    @property
    def grounding_eligible(self) -> np.ndarray:
        return self.measurement_valid & self.observable & np.isfinite(self.values)


class SyntheticMatrix:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.feature_root = root / "features"
        self.probe_root = root / "probes"
        self.locked_root = root / "locked"
        self.sidecar = root / "audit_sidecars.private.npz"
        self.sidecar.write_bytes(b"synthetic-sidecar")
        self.patient_ids = np.asarray([f"p{index:02d}" for index in range(10)])
        self.folds = self._folds()
        self.lock = self._lock()
        self.preregistration = root / "PREREGISTRATION_LOCK.json"
        self.preregistration.write_text(json.dumps(self.lock), encoding="utf-8")
        self.preregistration_sha256 = file_sha256(self.preregistration)
        self._write_p0_assets()

    def _folds(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for fold in range(5):
            test = {2 * fold, 2 * fold + 1}
            validation = {(2 * fold + 2) % 10, (2 * fold + 3) % 10}
            for index, patient_id in enumerate(self.patient_ids):
                split = "test" if index in test else "val" if index in validation else "train"
                rows.append(
                    {"patient_id": str(patient_id), "fold": fold, "split": split}
                )
        return pd.DataFrame(rows)

    def _lock(self) -> dict[str, object]:
        checkpoints: dict[str, dict[str, object]] = {}
        references: dict[str, dict[str, object]] = {}
        for seed, arm, fold in cells():
            name = cell_key(seed, arm, fold)
            directory = self.locked_root / f"seed_{seed}" / arm / f"fold_{fold}"
            directory.mkdir(parents=True, exist_ok=True)
            checkpoint = directory / "selected.pt"
            checkpoint.write_bytes(f"checkpoint::{name}".encode("utf-8"))
            reference = directory / "response_state.private.npz"
            reference.write_bytes(f"reference::{name}".encode("utf-8"))
            reference_metadata = reference.with_suffix(".metadata.json")
            reference_metadata.write_text(
                json.dumps({"identity": name}), encoding="utf-8"
            )
            checkpoints[name] = {
                "path": str(checkpoint),
                "sha256": file_sha256(checkpoint),
                "size_bytes": checkpoint.stat().st_size,
                "mtime_ns": checkpoint.stat().st_mtime_ns,
            }
            references[name] = {
                "feature_path": str(reference),
                "feature_sha256": file_sha256(reference),
                "feature_metadata_path": str(reference_metadata),
                "feature_metadata_sha256": file_sha256(reference_metadata),
                "patient_order_sha256": ordered_patient_sha256(self.patient_ids),
            }
        return {
            "schema_version": 1,
            "status": "FROZEN_BEFORE_NEW_FEATURE_OR_PROBE",
            "formal_cell_count": 40,
            "plan_sha256": "a" * 64,
            "config_sha256": "b" * 64,
            "selected_checkpoints": checkpoints,
            "formal_p0_references": references,
        }

    def split_for(self, fold: int) -> np.ndarray:
        return self.folds.loc[
            self.folds["fold"].eq(fold), "split"
        ].to_numpy(dtype=str)

    def _state(self, seed: int, arm: str, fold: int) -> np.ndarray:
        state = np.zeros((len(self.patient_ids), 4, 192), dtype=np.float32)
        arm_offset = {"L1": 0.0, "L3": 0.3, "N1": 0.6, "N3": 0.9}[arm]
        for patient in range(len(self.patient_ids)):
            for visit in range(4):
                state[patient, visit, :6] = (
                    patient,
                    visit,
                    patient * (visit + 1),
                    np.sin(patient + visit),
                    arm_offset,
                    seed / 1000.0 + fold,
                )
        return state

    def _write_p0_assets(self) -> None:
        sentinel_sha256 = file_sha256(UPSTREAM_ROOT / "STAGE_A_GO.json")
        for seed, arm, fold in cells():
            key = ProbeCellKey(seed, arm, fold, "P0")
            name = key.checkpoint_key
            binding = self.lock["selected_checkpoints"][name]
            reference = self.lock["formal_p0_references"][name]
            path = feature_path_for(self.feature_root, "final", key)
            path.parent.mkdir(parents=True, exist_ok=True)
            split = self.split_for(fold)
            state = self._state(seed, arm, fold)
            valid = np.ones(state.shape[:2], dtype=bool)
            np.savez_compressed(
                path,
                patient_id=self.patient_ids,
                split=split,
                state=state,
                state_valid=valid,
                arm=np.asarray(arm),
                seed_base=np.asarray(seed),
                fold=np.asarray(fold),
                pooling=np.asarray("P0"),
            )
            path.chmod(0o600)
            metadata = {
                "schema_version": 1,
                "stage": "final",
                "status": "COMPLETE",
                "arm": arm,
                "seed_base": seed,
                "fold": fold,
                "pooling": "P0",
                "pooling_slug": "p0",
                "feature_path": str(path.resolve()),
                "feature_sha256": file_sha256(path),
                "state_shape": list(state.shape),
                "state_dtype": "float32",
                "state_valid_shape": list(valid.shape),
                "state_valid_count": int(valid.sum()),
                "patient_count": len(self.patient_ids),
                "patient_order_sha256": ordered_patient_sha256(self.patient_ids),
                "split_order_sha256": split_order_sha256(split),
                "checkpoint_path": binding["path"],
                "checkpoint_sha256": binding["sha256"],
                "checkpoint_lock_key": name,
                "reference_feature_path": reference["feature_path"],
                "reference_feature_sha256": reference["feature_sha256"],
                "reference_feature_metadata_path": reference[
                    "feature_metadata_path"
                ],
                "reference_feature_metadata_sha256": reference[
                    "feature_metadata_sha256"
                ],
                "preregistration_lock_sha256": self.preregistration_sha256,
                "plan_sha256": self.lock["plan_sha256"],
                "config_sha256": self.lock["config_sha256"],
                "sidecar_path": str(self.sidecar.resolve()),
                "sidecar_sha256": file_sha256(self.sidecar),
                "sidecar_keys_used": [],
                "data_contract_provenance_sha256": "c" * 64,
                "checkpoint_data_provenance_sha256": "d" * 64,
                "stage_a_sentinel_sha256": sentinel_sha256,
                "implementation_sha256": {
                    name: file_sha256(ROOT / "src" / "c1b_spatial_audit" / name)
                    for name in ("exporter.py", "pooling.py", "runtime.py", "contracts.py")
                },
                "device": "cuda:0",
                "batch_size": 4,
                "workers": 2,
                "feature_tensor": "full_model.encoder_output_before_gap",
                "response_projection": "frozen_online_Linear128x192_plus_LayerNorm",
                "training_performed": False,
                "projector_called": False,
                "transition_called": False,
                "target_encoder_called": False,
                "ftv_head_called": False,
                "test_labels_used": False,
            }
            if set(metadata) != EXPORTER_METADATA_FIELDS:
                raise AssertionError("synthetic exporter metadata fixture drifted")
            metadata_path = path.with_suffix(".metadata.json")
            metadata_path.write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            metadata_path.chmod(0o600)

    def nuisance_csv(self) -> Path:
        rows: list[dict[str, object]] = []
        for patient_index, patient_id in enumerate(self.patient_ids):
            for visit_index, visit in enumerate(TIMEPOINTS):
                row: dict[str, object] = {
                    "patient_id": str(patient_id),
                    "visit": visit,
                }
                for target_index, target in enumerate(NUISANCE_TARGETS):
                    row[target] = (
                        0.1 * patient_index + 0.01 * visit_index + target_index
                    )
                rows.append(row)
        path = self.root / "nuisance_targets.private.csv"
        pd.DataFrame(rows, columns=NUISANCE_COLUMNS).to_csv(path, index=False)
        path.chmod(0o600)
        return path

    def records(self) -> dict[str, SyntheticFTVRecord]:
        records: dict[str, SyntheticFTVRecord] = {}
        for patient_index, patient_id in enumerate(self.patient_ids):
            visit = np.arange(4, dtype=np.float64)
            records[str(patient_id)] = SyntheticFTVRecord(
                values=(
                    10.0
                    + patient_index
                    + (1.0 + 0.1 * patient_index) * visit
                    + 0.05 * visit**2
                ),
                measurement_valid=np.ones(4, dtype=bool),
                observable=np.ones(4, dtype=bool),
            )
        return records


class MatrixShapeTests(unittest.TestCase):
    def test_pooling_parser_and_preregistered_availability(self) -> None:
        self.assertEqual(parse_poolings("plocal,p0"), ("P0", "PLOCAL"))
        self.assertEqual(
            parse_poolings("plocal_pvalid_secondary,p0"),
            ("P0", "PLOCAL+PVALID_SECONDARY"),
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_poolings("P0,p0")
        with self.assertRaisesRegex(ValueError, "unknown"):
            parse_poolings("P0,LEARNED")
        self.assertEqual(len(expected_feature_keys("final", ("P0",))), 40)
        self.assertEqual(
            len(
                expected_feature_keys(
                    "final", FORMAL_POOLINGS
                )
            ),
            180,
        )
        self.assertEqual(
            len(expected_feature_keys("s3", ("P0", "PLOCAL", "PORACLE"))), 100
        )
        self.assertEqual(expected_feature_dimension("final", "P0"), 192)
        self.assertEqual(expected_feature_dimension("final", "PLOCAL+GLOBAL"), 384)
        self.assertEqual(
            expected_feature_dimension("final", "PLOCAL+PVALID_SECONDARY"), 384
        )
        self.assertEqual(
            expected_sidecar_keys("N1", "PLOCAL+PVALID_SECONDARY"),
            ("c1b_local_weight_final", "c1b_valid_weight_final"),
        )
        secondary = expected_feature_keys(
            "final", ("PLOCAL+PVALID_SECONDARY",)
        )
        self.assertEqual(len(secondary), 20)
        self.assertTrue(all(key.arm.startswith("N") for key in secondary))
        self.assertNotIn("PLOCAL+PVALID_SECONDARY", NUISANCE_POOLINGS)
        alternatives = expected_feature_keys("final", FORMAL_POOLINGS[1:])
        self.assertEqual(len(alternatives), 140)
        self.assertEqual(
            sum(key.pooling in NUISANCE_POOLINGS for key in alternatives), 60
        )
        self.assertEqual(expected_feature_dimension("s3", "PLOCAL"), 64)
        with self.assertRaisesRegex(ValueError, "does not preregister"):
            validate_stage_poolings("s3", ("PVALID",))


class DiscoveryAndBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = SyntheticMatrix(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def discover(self):
        return discover_feature_matrix(
            feature_root=self.fixture.feature_root,
            probe_root=self.fixture.probe_root,
            stage="final",
            poolings=("P0",),
            folds=self.fixture.folds,
            lock=self.fixture.lock,
            preregistration_path=self.fixture.preregistration,
        )

    def test_exact_p0_matrix_and_output_layout_validate(self) -> None:
        specifications = self.discover()
        self.assertEqual(len(specifications), 40)
        first = specifications[0]
        self.assertEqual(first.key.pooling, "P0")
        self.assertTrue(first.include_nuisance)
        self.assertEqual(
            first.output_dir,
            output_path_for(self.fixture.probe_root, "final", first.key),
        )
        self.assertEqual(first.feature_sha256, file_sha256(first.feature_path))

    def test_missing_asset_and_checkpoint_metadata_drift_fail_closed(self) -> None:
        missing_key = ProbeCellKey(2026, "L1", 0, "P0")
        missing = feature_path_for(self.fixture.feature_root, "final", missing_key)
        missing.unlink()
        with self.assertRaisesRegex(ValueError, "matrix is incomplete"):
            self.discover()

        self.fixture._write_p0_assets()
        metadata_path = missing.with_suffix(".metadata.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["checkpoint_sha256"] = "f" * 64
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checkpoint_sha256"):
            self.discover()

    def test_split_labels_are_checked_against_frozen_fold_manifest(self) -> None:
        key = ProbeCellKey(2026, "L1", 0, "P0")
        path = feature_path_for(self.fixture.feature_root, "final", key)
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        arrays["split"] = arrays["split"].copy()
        arrays["split"][0], arrays["split"][4] = arrays["split"][4], arrays["split"][0]
        np.savez_compressed(path, **arrays)
        path.chmod(0o600)
        metadata_path = path.with_suffix(".metadata.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["feature_sha256"] = file_sha256(path)
        metadata["split_order_sha256"] = split_order_sha256(arrays["split"])
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "frozen fold manifest"):
            self.discover()


class NuisanceAndGateTests(unittest.TestCase):
    def test_exporter_completion_binds_exact_180_metadata_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            feature_root = Path(temporary) / "features"
            final_root = feature_root / "final"
            final_root.mkdir(parents=True)
            inventory: dict[str, str] = {}
            for index in range(180):
                metadata = final_root / f"asset_{index:03d}.private.metadata.json"
                metadata.write_text(json.dumps({"index": index}), encoding="utf-8")
                metadata.chmod(0o600)
                inventory[str(metadata)] = file_sha256(metadata)
            preregistration_sha256 = "a" * 64
            sidecar_sha256 = "b" * 64
            preflight = feature_root / "feature_export_preflight.private.json"
            preflight.write_text(
                json.dumps(
                    {
                        "status": "PREFLIGHT_PASS",
                        "stage": "final",
                        "cell_count": 40,
                        "expected_asset_count": 180,
                        "preregistration_lock_sha256": preregistration_sha256,
                        "sidecar_sha256": sidecar_sha256,
                    }
                ),
                encoding="utf-8",
            )
            preflight.chmod(0o600)
            completion = feature_root / "feature_export_complete.private.json"
            completion.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "COMPLETE",
                        "stage": "final",
                        "run_count": 40,
                        "expected_asset_count": 180,
                        "cell_count": 40,
                        "feature_metadata_sha256": inventory,
                        "preflight_sha256": file_sha256(preflight),
                        "sidecar_sha256": sidecar_sha256,
                        "preregistration_lock_sha256": preregistration_sha256,
                    }
                ),
                encoding="utf-8",
            )
            completion.chmod(0o600)
            self.assertEqual(
                validate_exporter_completion(
                    feature_root,
                    stage="final",
                    preregistration_sha256=preregistration_sha256,
                ),
                file_sha256(completion),
            )
            first = final_root / "asset_000.private.metadata.json"
            first.write_text("drift", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inventory drifted"):
                validate_exporter_completion(
                    feature_root,
                    stage="final",
                    preregistration_sha256=preregistration_sha256,
                )

    def test_exact_ten_target_sidecar_is_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticMatrix(Path(temporary))
            path = fixture.nuisance_csv()
            targets = load_nuisance_targets(path, fixture.patient_ids)
            self.assertEqual(tuple(targets), NUISANCE_TARGETS)
            self.assertEqual(len(targets), 10)
            np.testing.assert_allclose(
                targets[NUISANCE_TARGETS[0]]["p03"],
                np.asarray([0.30, 0.31, 0.32, 0.33]),
            )
            frame = pd.read_csv(path).drop(columns=[NUISANCE_TARGETS[-1]])
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "schema drifted"):
                load_nuisance_targets(path, fixture.patient_ids)

    def test_alternatives_require_both_authorizing_p0_gates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            equivalence = root / "p0_equivalence_gate.json"
            replication = root / "p0_probe_replication_gate.json"
            preregistration_sha256 = "a" * 64
            with self.assertRaises(FileNotFoundError):
                validate_alternative_gates(
                    ("P0",),
                    preregistration_sha256=preregistration_sha256,
                    equivalence_gate_path=equivalence,
                    probe_replication_gate_path=replication,
                )
            equivalence.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "probe_execution_authorized": True,
                        "formal_cells": 40,
                        "preregistration_lock_sha256": preregistration_sha256,
                    }
                ),
                encoding="utf-8",
            )
            replication.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "alternate_pooling_interpretation_authorized": True,
                        "formal_cells": 40,
                    }
                ),
                encoding="utf-8",
            )
            p0_hashes = validate_alternative_gates(
                ("P0",),
                preregistration_sha256=preregistration_sha256,
                equivalence_gate_path=equivalence,
                probe_replication_gate_path=root / "intentionally_absent.json",
            )
            self.assertEqual(set(p0_hashes), {"p0_equivalence_gate_sha256"})
            hashes = validate_alternative_gates(
                ("PLOCAL",),
                preregistration_sha256=preregistration_sha256,
                equivalence_gate_path=equivalence,
                probe_replication_gate_path=replication,
            )
            self.assertEqual(len(hashes), 2)
            payload = json.loads(replication.read_text(encoding="utf-8"))
            payload["alternate_pooling_interpretation_authorized"] = False
            replication.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(PermissionError, "replication"):
                validate_alternative_gates(
                    ("PLOCAL",),
                    preregistration_sha256=preregistration_sha256,
                    equivalence_gate_path=equivalence,
                    probe_replication_gate_path=replication,
                )


class AtomicExecutionTests(unittest.TestCase):
    def test_process_pool_execution_publishes_one_private_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticMatrix(Path(temporary))
            specifications = discover_feature_matrix(
                feature_root=fixture.feature_root,
                probe_root=fixture.probe_root,
                stage="final",
                poolings=("P0",),
                folds=fixture.folds,
                lock=fixture.lock,
                preregistration_path=fixture.preregistration,
            )[:2]
            nuisance_path = fixture.nuisance_csv()
            targets = load_nuisance_targets(nuisance_path, fixture.patient_ids)
            plan = ProbeMatrixPlan(
                stage="final",
                poolings=("P0",),
                cells=specifications,
                feature_root=fixture.feature_root,
                probe_root=fixture.probe_root,
                preregistration_path=fixture.preregistration,
                preregistration_sha256=fixture.preregistration_sha256,
                nuisance_path=nuisance_path,
                nuisance_sha256=file_sha256(nuisance_path),
                gate_sha256={"p0_equivalence_gate_sha256": "f" * 64},
                exporter_completion_sha256="e" * 64,
            )
            completion = execute_probe_plan(
                plan,
                records=fixture.records(),
                nuisance_targets=targets,
                workers=2,
            )
            self.assertEqual(completion["executed_cell_count"], 2)
            self.assertEqual(len(completion["cells"]), 2)
            self.assertTrue(plan.completion_path.is_file())
            self.assertEqual(stat.S_IMODE(plan.completion_path.stat().st_mode), 0o600)
            with self.assertRaisesRegex(FileExistsError, "completion inventory"):
                execute_probe_plan(
                    plan,
                    records=fixture.records(),
                    nuisance_targets=targets,
                    workers=2,
                )

    def test_one_validated_cell_writes_owner_only_combined_outputs_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticMatrix(Path(temporary))
            specifications = discover_feature_matrix(
                feature_root=fixture.feature_root,
                probe_root=fixture.probe_root,
                stage="final",
                poolings=("P0",),
                folds=fixture.folds,
                lock=fixture.lock,
                preregistration_path=fixture.preregistration,
            )
            targets = load_nuisance_targets(
                fixture.nuisance_csv(), fixture.patient_ids
            )
            inventory = execute_probe_cell(
                specifications[0], fixture.records(), targets
            )
            output = Path(inventory["output_dir"])
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o700)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "ridge_selection.csv",
                    "ridge_predictions.private.csv",
                    "probe_metrics.csv",
                    "probe_metadata.json",
                },
            )
            for path in output.iterdir():
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            selection = pd.read_csv(output / "ridge_selection.csv")
            self.assertEqual(len(selection), 54)
            self.assertEqual(set(selection["task"]), {"static", "delta", "nuisance"})
            self.assertEqual(
                set(selection.loc[selection["task"].eq("nuisance"), "target_name"]),
                set(NUISANCE_TARGETS),
            )
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                execute_probe_cell(specifications[0], fixture.records(), targets)

    def test_cli_exposes_required_modes_and_rejects_bad_workers(self) -> None:
        script = ROOT / "scripts" / "run_probe_matrix.py"
        spec = importlib.util.spec_from_file_location("run_probe_matrix_test", script)
        if spec is None or spec.loader is None:
            raise AssertionError("cannot import probe-matrix CLI")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        args = module.parse_args(
            ["--stage", "final", "--poolings", "P0,PLOCAL", "--workers", "2", "--execute"]
        )
        self.assertEqual(args.stage, "final")
        self.assertEqual(args.poolings, ("P0", "PLOCAL"))
        self.assertEqual(args.workers, 2)
        self.assertTrue(args.execute)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            module.parse_args(["--workers", "0"])


if __name__ == "__main__":
    unittest.main()

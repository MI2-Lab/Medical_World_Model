from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(ROOT / "scripts"))
common = load_module("mask_free_common", ROOT / "scripts" / "common.py")
export_features = load_module(
    "mask_free_export_features", ROOT / "scripts" / "export_features.py"
)
run_feature_matrix = load_module(
    "mask_free_run_feature_matrix", ROOT / "scripts" / "run_feature_matrix.py"
)


class ExtractionContractTests(unittest.TestCase):
    def test_archive_and_metadata_contracts_are_exact(self) -> None:
        expected = (
            "patient_id",
            "split",
            "R0",
            "R1",
            "R2",
            "R3",
            "R4",
            "R5",
            "R5_RP192",
            "S1",
            "S2",
            "S3",
            "S4",
            "S5",
            "arm",
            "seed_base",
            "fold",
        )
        self.assertEqual(common.FEATURE_KEYS, expected)
        self.assertEqual(export_features.FEATURE_KEYS, expected)
        self.assertEqual(common.FEATURE_FILENAME, "regional_features.private.npz")
        self.assertEqual(
            common.METADATA_FILENAME, "regional_features.private.metadata.json"
        )
        self.assertEqual(export_features.METADATA_KEYS, common.METADATA_KEYS)
        self.assertIn("r0_goal5_mean_parity", common.METADATA_KEYS)
        self.assertIn("projected_r0_local_state_parity", common.METADATA_KEYS)
        self.assertIn("lesion_mask_read", common.METADATA_KEYS)

    def test_exact_twenty_cell_order_and_paths(self) -> None:
        cells = common.cells()
        self.assertEqual(len(cells), 20)
        self.assertEqual(cells[0], (2026, "LOCAL0", 0))
        self.assertEqual(cells[5], (2026, "LOCAL3", 0))
        self.assertEqual(cells[10], (3026, "LOCAL0", 0))
        self.assertEqual(cells[-1], (3026, "LOCAL3", 4))
        path = common.feature_path(3026, "LOCAL3", 4)
        self.assertEqual(path.name, common.FEATURE_FILENAME)
        self.assertEqual(path.parent.name, "fold_4")
        self.assertEqual(common.metadata_path(3026, "LOCAL3", 4).name, common.METADATA_FILENAME)

    def test_ordered_hash_matches_real_goal5_reference(self) -> None:
        config = common.load_config(verify_extraction_inputs=False)
        goal5 = common.load_goal5_lock(config)
        record = goal5["selected_cells"]["seed_2026/LOCAL0/fold_0"]
        with np.load(record["reference"]["path"], allow_pickle=False) as archive:
            patient_id = np.asarray(archive["patient_id"]).astype(str)
            split = np.asarray(archive["split"]).astype(str)
        self.assertEqual(
            common.ordered_sha256(patient_id),
            record["reference"]["patient_order_sha256"],
        )
        self.assertEqual(
            common.ordered_sha256(split),
            record["reference"]["split_order_sha256"],
        )

    def test_config_safe_validation_does_not_open_label_paths(self) -> None:
        # The config contains these downstream paths, but extraction-safe input
        # verification explicitly does not stat/read them.
        config = json.loads(common.CONFIG_PATH.read_text(encoding="utf-8"))
        config["paths"]["clinical_labels"] = "/definitely/not/read/clinical.csv"
        config["paths"]["ftv_table"] = "/definitely/not/read/ftv.csv"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audit.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            observed = common.load_config(path, verify_extraction_inputs=True)
        self.assertEqual(observed["experiment"], "mask_free_region_aware_audit")

    def test_goal5_r0_loader_omits_oracle_members(self) -> None:
        source = (ROOT / "scripts" / "export_features.py").read_text(encoding="utf-8")
        self.assertIn('("patient_id", "split", "mean", "arm", "seed_base", "fold")', source)
        # No Oracle loader or oracle-sidecar argument exists in the new exporter.
        self.assertNotIn("def _load_oracle", source)
        self.assertNotIn("oracle_path", source)
        self.assertNotIn("load_stage_b_data(", source)
        self.assertNotIn("read_raw_ftv(", source)
        self.assertNotIn("load_clinical", source)

    def test_matrix_preflight_cannot_execute_without_flag(self) -> None:
        parsed = run_feature_matrix.parse_args([])
        self.assertFalse(parsed.execute)
        self.assertEqual(parsed.devices, "cuda:0,cuda:1,cuda:2")
        self.assertEqual(set(common.COMPLETION_KEYS), {
            "schema_version",
            "status",
            "experiment",
            "cell_count",
            "config_sha256",
            "preregistration_lock_sha256",
            "geometry_contract_sha256",
            "cells",
        })

    def test_atomic_npz_is_owner_only_and_has_exact_members(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "example.private.npz"
            common.atomic_npz(path, {"a": np.arange(3, dtype=np.float32)})
            self.assertEqual(path.stat().st_mode & 0o077, 0)
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(archive.files, ["a"])
                np.testing.assert_array_equal(archive["a"], np.arange(3, dtype=np.float32))
            with self.assertRaisesRegex(FileExistsError, "refusing to replace"):
                common.atomic_npz(path, {"a": np.zeros(3, dtype=np.float32)})

    def test_freezer_discovers_all_scripts_and_tests(self) -> None:
        inventory = common.implementation_inventory()
        expected = {
            str(path.relative_to(ROOT))
            for directory in ("scripts", "tests")
            for path in (ROOT / directory).glob("*.py")
        }
        self.assertEqual(set(inventory), expected)
        self.assertIn("scripts/freeze_preregistration.py", inventory)
        self.assertIn("tests/test_extraction_contract.py", inventory)

    def test_lock_auth_accepts_analysis_resolved_config_view(self) -> None:
        config = common.load_config(verify_extraction_inputs=False)
        goal5 = common.load_goal5_lock(config)
        payload = {
            "schema_version": 1,
            "status": common.LOCK_STATUS,
            "branch": config["branch"],
            "parent_commit_sha": config["start"]["parent_commit_sha"],
            "created_utc": "2026-08-12T06:00:00Z",
            "config_sha256": common.file_sha256(common.CONFIG_PATH),
            "config_canonical_sha256": common.canonical_sha256(config),
            "experiment_plan_sha256": common.file_sha256(common.PLAN_PATH),
            "gitignore_sha256": common.file_sha256(common.GITIGNORE_PATH),
            "implementation_sha256": common.implementation_inventory(),
            "goal5_preregistration_lock_sha256": config["paths"]["goal5_lock_sha256"],
            "goal5_feature_completion_sha256": config["paths"][
                "goal5_feature_completion_sha256"
            ],
            "formal_cell_count": 20,
            "selected_cells": goal5["selected_cells"],
            "pre_freeze_result_inventory": {
                "feature_files": 0,
                "prediction_files": 0,
                "metric_files": 0,
                "figure_files": 0,
                "report_files": 0,
                "log_files": 0,
                "manifest_files": 0,
            },
            "privacy_contract": {
                "private_patient_artifacts_owner_only": True,
                "raw_spatial_maps_persisted": False,
                "region_definition_reads_masks_labels_ftv_or_clinical": False,
            },
        }
        analysis_view = dict(config)
        analysis_view["paths"] = {
            name: Path(value) if not name.endswith("_sha256") and name.endswith(("root", "lock", "completion", "predictions", "metrics", "sidecar", "contract", "authorization", "manifest", "labels", "table")) else value
            for name, value in config["paths"].items()
        }
        analysis_view["config_path"] = common.CONFIG_PATH
        analysis_view["config_sha256"] = common.file_sha256(common.CONFIG_PATH)
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "PREREGISTRATION_LOCK.json"
            lock_path.write_text(json.dumps(payload), encoding="utf-8")
            observed = common.require_preregistration_lock(analysis_view, lock_path)
        self.assertEqual(observed["formal_cell_count"], 20)


if __name__ == "__main__":
    unittest.main()

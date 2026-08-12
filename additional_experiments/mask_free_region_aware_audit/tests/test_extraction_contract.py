from __future__ import annotations

import importlib.util
import hashlib
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

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
freeze_preregistration = load_module(
    "mask_free_freeze_preregistration", ROOT / "scripts" / "freeze_preregistration.py"
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
        prior = common.require_prior_preregistration()
        prefix = str(ROOT.relative_to(common.REPO_ROOT)) + "/"
        payload = {
            **prior,
            "schema_version": 2,
            "preregistration_revision": 1,
            "status": common.REFREEZE_LOCK_STATUS,
            "created_utc": "2026-08-12T06:00:00Z",
            "config_sha256": common.file_sha256(common.CONFIG_PATH),
            "config_canonical_sha256": common.canonical_sha256(config),
            "experiment_plan_sha256": common.file_sha256(common.PLAN_PATH),
            "gitignore_sha256": common.file_sha256(common.GITIGNORE_PATH),
            "implementation_sha256": common.implementation_inventory(),
            "prior_preregistration_commit": common.PRIOR_PREREGISTRATION_COMMIT,
            "prior_preregistration_lock_sha256": (
                common.PRIOR_PREREGISTRATION_LOCK_SHA256
            ),
            "implementation_erratum_sha256": common.file_sha256(
                common.IMPLEMENTATION_ERRATUM_PATH
            ),
            "superseded_artifacts_reused": False,
            "scientific_contract_unchanged": True,
            "pre_refreeze_result_inventory": dict(common.ZERO_RESULT_INVENTORY),
            "git_provenance_before_refreeze": {
                "base_head": common.PRIOR_PREREGISTRATION_COMMIT,
                "branch": config["branch"],
                "all_dirty_paths_confined_to_new_experiment": True,
                "tracked_paths_before_refreeze": [
                    prefix + "EXPERIMENT_PLAN.md",
                    prefix + "scripts/common.py",
                    prefix + "scripts/freeze_preregistration.py",
                    prefix + "scripts/run_audit.py",
                    prefix + "tests/test_extraction_contract.py",
                ],
                "untracked_paths_before_refreeze": [
                    prefix + "PREREGISTRATION_IMPLEMENTATION_ERRATUM.json"
                ],
            },
        }
        self.assertEqual(set(payload), set(common.REFREEZE_LOCK_KEYS))
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
        self.assertFalse(observed["superseded_artifacts_reused"])

    def test_schema2_auth_rejects_prior_sha_or_scientific_mutation(self) -> None:
        config = common.load_config(verify_extraction_inputs=False)
        prior = common.require_prior_preregistration()
        prefix = str(ROOT.relative_to(common.REPO_ROOT)) + "/"
        payload = {
            **prior,
            "schema_version": 2,
            "preregistration_revision": 1,
            "status": common.REFREEZE_LOCK_STATUS,
            "created_utc": "2026-08-12T08:00:00Z",
            "experiment_plan_sha256": common.file_sha256(common.PLAN_PATH),
            "implementation_sha256": common.implementation_inventory(),
            "prior_preregistration_commit": common.PRIOR_PREREGISTRATION_COMMIT,
            "prior_preregistration_lock_sha256": (
                common.PRIOR_PREREGISTRATION_LOCK_SHA256
            ),
            "implementation_erratum_sha256": common.file_sha256(
                common.IMPLEMENTATION_ERRATUM_PATH
            ),
            "superseded_artifacts_reused": False,
            "scientific_contract_unchanged": True,
            "pre_refreeze_result_inventory": dict(common.ZERO_RESULT_INVENTORY),
            "git_provenance_before_refreeze": {
                "base_head": common.PRIOR_PREREGISTRATION_COMMIT,
                "branch": config["branch"],
                "all_dirty_paths_confined_to_new_experiment": True,
                "tracked_paths_before_refreeze": sorted(
                    [
                        prefix + "EXPERIMENT_PLAN.md",
                        prefix + "scripts/common.py",
                        prefix + "scripts/freeze_preregistration.py",
                        prefix + "scripts/run_audit.py",
                        prefix + "tests/test_extraction_contract.py",
                    ]
                ),
                "untracked_paths_before_refreeze": [
                    prefix + "PREREGISTRATION_IMPLEMENTATION_ERRATUM.json"
                ],
            },
        }
        for name, bad_value in (
            ("prior_preregistration_lock_sha256", "0" * 64),
            ("superseded_artifacts_reused", True),
            ("scientific_contract_unchanged", False),
        ):
            corrupted = {**payload, name: bad_value}
            with tempfile.TemporaryDirectory() as temporary:
                lock_path = Path(temporary) / "PREREGISTRATION_LOCK.json"
                lock_path.write_text(json.dumps(corrupted), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "schema-2 compatibility"):
                    common.require_preregistration_lock(config, lock_path)

    def test_erratum_and_plan_are_exactly_bound_to_historical_lock(self) -> None:
        prior = common.require_prior_preregistration()
        erratum = common.require_implementation_erratum()
        common.require_erratum_plan_disclosure(prior)
        self.assertEqual(
            erratum["reason_code"], common.IMPLEMENTATION_ERRATUM_REASON
        )
        self.assertFalse(erratum["contract_scope"]["scientific_contract_changed"])
        historical = common.historical_file_bytes(
            common.PRIOR_PREREGISTRATION_COMMIT,
            common.LOCK_PATH.relative_to(common.REPO_ROOT),
        )
        self.assertEqual(
            hashlib.sha256(historical).hexdigest(),
            common.PRIOR_PREREGISTRATION_LOCK_SHA256,
        )

    def test_refreeze_requires_explicit_flag_and_atomic_old_lock_match(self) -> None:
        self.assertTrue(
            freeze_preregistration.parse_args(["--execute-refreeze"]).execute_refreeze
        )
        self.assertFalse(freeze_preregistration.parse_args([]).execute_refreeze)
        prior_bytes = common.historical_file_bytes(
            common.PRIOR_PREREGISTRATION_COMMIT,
            common.LOCK_PATH.relative_to(common.REPO_ROOT),
        )
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "PREREGISTRATION_LOCK.json"
            lock_path.write_bytes(prior_bytes)
            payload = {"schema_version": 2, "status": common.REFREEZE_LOCK_STATUS}
            with patch.object(freeze_preregistration, "LOCK_PATH", lock_path), patch.object(
                freeze_preregistration,
                "_prior_lock_bytes_from_git",
                return_value=prior_bytes,
            ):
                freeze_preregistration._replace_prior_lock(payload)
            self.assertEqual(json.loads(lock_path.read_text(encoding="utf-8")), payload)
            self.assertEqual(lock_path.stat().st_mode & 0o777, 0o644)
            lock_path.write_text("{}\n", encoding="utf-8")
            with patch.object(freeze_preregistration, "LOCK_PATH", lock_path):
                with self.assertRaisesRegex(ValueError, "non-historical"):
                    freeze_preregistration._replace_prior_lock(payload)

    def test_refreeze_inventory_allows_only_frozen_nonresults(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for directory in freeze_preregistration.RESULT_DIRECTORIES:
                (root / directory).mkdir(parents=True)
            for relative in freeze_preregistration.ALLOWED_PREFREEZE_NONRESULTS:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("placeholder\n", encoding="utf-8")
            with patch.object(freeze_preregistration, "ROOT", root):
                self.assertEqual(
                    freeze_preregistration._pre_freeze_result_inventory(),
                    common.ZERO_RESULT_INVENTORY,
                )
                (root / "predictions" / "leak.private.csv").write_text(
                    "patient_id\nforbidden\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, "result artifacts"):
                    freeze_preregistration._pre_freeze_result_inventory()

    def test_refreeze_builder_preserves_schema1_science_and_binds_new_code(self) -> None:
        config = common.load_config(verify_extraction_inputs=False)
        prior = common.require_prior_preregistration()
        prefix = str(ROOT.relative_to(common.REPO_ROOT)) + "/"
        provenance = {
            "base_head": common.PRIOR_PREREGISTRATION_COMMIT,
            "branch": config["branch"],
            "all_dirty_paths_confined_to_new_experiment": True,
            "tracked_paths_before_refreeze": sorted(
                [
                    prefix + "EXPERIMENT_PLAN.md",
                    prefix + "scripts/common.py",
                    prefix + "scripts/freeze_preregistration.py",
                    prefix + "scripts/run_audit.py",
                    prefix + "tests/test_extraction_contract.py",
                ]
            ),
            "untracked_paths_before_refreeze": [
                prefix + "PREREGISTRATION_IMPLEMENTATION_ERRATUM.json"
            ],
        }
        with patch.object(
            freeze_preregistration,
            "_require_refreeze_git_context",
            return_value=provenance,
        ), patch.object(
            freeze_preregistration, "_require_working_prior_lock", return_value=prior
        ), patch.object(
            freeze_preregistration,
            "_selected_cells",
            return_value=prior["selected_cells"],
        ), patch.object(
            freeze_preregistration,
            "_pre_freeze_result_inventory",
            return_value=dict(common.ZERO_RESULT_INVENTORY),
        ):
            payload = freeze_preregistration.build_refreeze_lock(config)
        self.assertEqual(set(payload), set(common.REFREEZE_LOCK_KEYS))
        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["status"], common.REFREEZE_LOCK_STATUS)
        self.assertEqual(
            payload["prior_preregistration_lock_sha256"],
            common.PRIOR_PREREGISTRATION_LOCK_SHA256,
        )
        self.assertFalse(payload["superseded_artifacts_reused"])
        for name in (
            "config_sha256",
            "config_canonical_sha256",
            "gitignore_sha256",
            "goal5_preregistration_lock_sha256",
            "goal5_feature_completion_sha256",
            "formal_cell_count",
            "selected_cells",
            "pre_freeze_result_inventory",
            "privacy_contract",
        ):
            self.assertEqual(payload[name], prior[name])
        self.assertEqual(payload["implementation_sha256"], common.implementation_inventory())
        self.assertNotEqual(
            payload["implementation_sha256"]["scripts/common.py"],
            prior["implementation_sha256"]["scripts/common.py"],
        )

    def test_old_artifact_lock_binding_has_no_schema2_reuse_bypass(self) -> None:
        source = inspect.getsource(common.validate_feature_cell)
        self.assertIn(
            'metadata["preregistration_lock_sha256"] != file_sha256(LOCK_PATH)',
            source,
        )
        matrix_source = (ROOT / "scripts" / "run_feature_matrix.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            '"preregistration_lock_sha256": file_sha256(LOCK_PATH)', matrix_source
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts/build_validate_model_inputs.py"
    spec = importlib.util.spec_from_file_location("new_stage_a_cache_adapter", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


adapter = load_script()


class CacheAdapterTests(unittest.TestCase):
    def _eligibility_files(self, directory: Path):
        patient_rows = []
        visit_rows = []
        for patient_id, eligible in (("dynamic-a", True), ("dynamic-b", False)):
            valid_count = 0 if not eligible else 9
            for visit in adapter.VISITS:
                count = valid_count if visit == "T2" or eligible else 9
                visit_rows.append(
                    {
                        "patient_id": patient_id,
                        "cohort": "I-SPY2",
                        "visit": visit,
                        "resolved_dce_nifti": f"/{patient_id}/{visit}.nii",
                        "grid_contract_sha256": "a" * 64,
                        "geometry_contract_sha256": "b" * 64,
                        "valid_source_voxels": count,
                        "target_grid_voxels": 10,
                        "has_valid_source_overlap": count > 0,
                    }
                )
            patient_rows.append(
                {
                    "patient_id": patient_id,
                    "cohort": "I-SPY2",
                    "candidate_visit_count": len(adapter.VISITS),
                    "valid_visit_count": len(adapter.VISITS) if eligible else len(adapter.VISITS) - 1,
                    "zero_overlap_visit_count": 0 if eligible else 1,
                    "minimum_valid_source_voxels": 9 if eligible else 0,
                    "eligible": eligible,
                    "exclusion_reason": "" if eligible else "ZERO_VALID_SOURCE_OVERLAP_IN_REQUIRED_VISIT",
                }
            )
        patients = pd.DataFrame(patient_rows)
        visits = pd.DataFrame(visit_rows)
        patient_path = directory / "patients.csv"
        visit_path = directory / "visits.csv"
        inventory_path = directory / "inventory.csv"
        patients.to_csv(patient_path, index=False)
        visits.to_csv(visit_path, index=False)
        visits.loc[visits["patient_id"].eq("dynamic-a")].drop(
            columns="has_valid_source_overlap"
        ).to_csv(inventory_path, index=False)
        return patient_path, visit_path, inventory_path

    def test_private_manifests_drive_dynamic_four_visit_population(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._eligibility_files(Path(directory))
            patients, eligible = adapter.load_technical_eligibility(*paths)
        self.assertEqual(len(patients), 2)
        self.assertEqual(set(eligible["patient_id"]), {"dynamic-a"})
        self.assertEqual(len(eligible), len(adapter.VISITS))
        self.assertTrue(eligible["valid_source_voxels"].gt(0).all())

    def test_prior_cache_hardlink_is_atomic_and_does_not_change_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "old.npz"
            destination = root / "new/cache/c1b_h/token.npz"
            source.write_bytes(b"immutable-cache-bytes")
            before = source.read_bytes()
            adapter._hardlink_atomic(source, destination)
            self.assertTrue(os.path.samefile(source, destination))
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(destination.read_bytes(), before)
            with self.assertRaises(FileExistsError):
                adapter._hardlink_atomic(source, destination)

    def test_public_gate_pass_requires_complete_all_patient_qc(self) -> None:
        row = {
            "patient_id": "PRIVATE-ID",
            "status": "PASS",
            "cache_origin": "validated_prior_cache_hardlink",
            "exact_valid_source_visit_matches": len(adapter.VISITS),
            "zero_overlap_visits": 0,
        }
        for column in (
            "exact_roundtrip_pass",
            "finite",
            "nonconstant",
            "shape_match",
            "orientation_match",
            "phase_contract_match",
            "grid_contract_match",
            "provenance_match",
            "model_loader_returns_only_image",
        ):
            row[column] = True
        public = adapter.build_public_gate(
            pd.DataFrame([row]),
            eligible_visits=len(adapter.VISITS),
            private_metrics_sha256="c" * 64,
            cache_inventory_sha256="d" * 64,
            eligibility_manifest_sha256={"private.csv": "e" * 64},
            elapsed_seconds=1.0,
        )
        self.assertEqual(public["status"], "PASS")
        self.assertEqual(public["completion_fraction"], 1.0)
        self.assertEqual(public["exact_valid_source_count_match_fraction"], 1.0)
        self.assertFalse(public["stage_b_authorized"])
        self.assertFalse(public["sidecars_are_model_inputs"])
        self.assertNotIn("PRIVATE-ID", json.dumps(public))

        failed = pd.DataFrame([{**row, "status": "FAIL", "failure_type": "ValueError"}])
        failed_public = adapter.build_public_gate(
            failed,
            eligible_visits=len(adapter.VISITS),
            private_metrics_sha256="c" * 64,
            cache_inventory_sha256="d" * 64,
            eligibility_manifest_sha256={},
            elapsed_seconds=1.0,
        )
        self.assertEqual(failed_public["status"], "FAIL")

    def test_runner_has_no_hardcoded_population_result(self) -> None:
        source = (ROOT / "scripts/build_validate_model_inputs.py").read_text()
        self.assertNotIn("== 947", source)
        self.assertNotIn("== 948", source)
        self.assertNotIn("range(947)", source)
        self.assertNotIn("range(948)", source)


if __name__ == "__main__":
    unittest.main()

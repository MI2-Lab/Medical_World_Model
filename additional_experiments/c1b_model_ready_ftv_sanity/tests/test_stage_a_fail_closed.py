import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


builder = load_script("build_validate_model_inputs")
finalizer = load_script("finalize_stage_a_gate")


class StageAFailClosedTests(unittest.TestCase):
    def test_zero_overlap_writes_private_evidence_and_terminal_no_go(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metrics").mkdir()
            (root / "manifests").mkdir()
            (root / "reports").mkdir()
            (root / "cache/c1b_h").mkdir(parents=True)
            public_fixtures = {
                "metrics/registration_strategy_decision.json": {
                    "decision_frozen": True,
                    "chosen_strategy": "H",
                    "formal_registration_pairs": 1125,
                    "registration_success_pairs": 858,
                    "registration_failure_pairs": 267,
                    "registration_success_rate": 858 / 1125,
                    "manual_review_pass": True,
                },
                "metrics/dicom_pixel_rebuild_gate.json": {
                    "status": "PASS",
                    "all_model_input_singular_visits": {
                        "visits": 146,
                        "passed": 146,
                        "failed": 0,
                        "verified_cells": 153112,
                        "max_cell_error": 0.0,
                    },
                },
                "metrics/orientation_validation_gate.json": {
                    "canonical_ras_fraction": 1.0,
                    "model_input_patients": 948,
                    "model_input_visits": 3792,
                    "array_reordering_implemented_and_unit_tested": True,
                    "header_only_label_change": False,
                },
                "metrics/registration_physical_support_summary.json": {
                    "c1b_h_ftv_retention_q05": 1.0
                },
                "metrics/support_containment_h_summary.json": {
                    "strategy": "C1B-H",
                    "formal_visits": 1500,
                    "exact_full_support_containment_rate": 0.978,
                    "physical_volume_retention": {"q05": 1.0},
                    "anchor_uses_t0_only": True,
                    "future_support_used_for_grid": False,
                },
                "metrics/ispy1_base_eligibility_summary.json": {
                    "outcome_fields_read": [],
                    "clinical_or_outcome_tables_read": [],
                    "public_artifact_contains_identifiers_or_paths": False,
                },
                "manifests/grounding_observability_summary.json": {
                    "is_model_input": False,
                    "does_not_filter_base_training": True,
                    "formal_visits": 1500,
                    "observable_visits": 1486,
                },
            }
            for relative, payload in public_fixtures.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload), encoding="utf-8")

            patient_ids = [f"PRIVATE_CASE_{index:04d}" for index in range(948)]
            records = []
            for patient_id in patient_ids:
                for visit in ("T0", "T1", "T2", "T3"):
                    records.append(
                        {
                            "patient_id": patient_id,
                            "visit": visit,
                            "cohort": "I-SPY2" if patient_id != patient_ids[0] else "I-SPY1",
                        }
                    )
            inventory = pd.DataFrame(records)
            audit = inventory.copy()
            audit["padding_fraction_bbox"] = 0.125
            audit.loc[
                audit["patient_id"].eq(patient_ids[0]) & audit["visit"].eq("T3"),
                "padding_fraction_bbox",
            ] = 1.0
            audit.to_csv(
                root / "metrics/orientation_resampling_patient_visit.private.csv",
                index=False,
            )
            validation_ids = set(patient_ids[:263])
            for patient_id in patient_ids[:262]:
                (root / "cache/c1b_h" / f"{builder.patient_token(patient_id)}.npz").touch()

            with self.assertRaisesRegex(RuntimeError, "ZERO_VALID_SOURCE_OVERLAP"):
                builder.catastrophic_source_overlap_preflight(
                    inventory,
                    validation_ids=validation_ids,
                    reasons={value: {"test_stratum"} for value in validation_ids},
                    scope="validation",
                    strategy="H",
                    overwrite=False,
                    experiment_root=root,
                )

            public_path = root / "metrics/model_input_pipeline_h_validation_gate.json"
            public = json.loads(public_path.read_text())
            self.assertEqual(public["status"], "FAIL")
            self.assertEqual(public["zero_valid_source_overlap_visits"], 1)
            self.assertEqual(public["atomic_caches_present_for_selected_patients"], 262)
            self.assertFalse(public["stage_b_authorized"])
            public_text = public_path.read_text() + (
                root / "reports/model_input_pipeline_validation.md"
            ).read_text()
            self.assertNotIn("PRIVATE_CASE_", public_text)
            self.assertNotIn(str(root), public_text)

            evidence = finalizer._validated_catastrophic_overlap_failure(root=root)
            self.assertIsNotNone(evidence)
            finalizer._close_catastrophic_overlap_no_go(
                evidence, overwrite=False, root=root
            )
            self.assertFalse((root / "STAGE_A_GO.json").exists())
            no_go = json.loads((root / "STAGE_A_NO_GO.json").read_text())
            self.assertEqual(no_go["status"], "NO-GO")
            self.assertFalse(no_go["stage_b_authorized"])
            table = pd.read_csv(root / "metrics/table1_model_ready_preprocessing_qc.csv")
            self.assertEqual(
                set(table["gate"]),
                {
                    "repaired_dicom_pixel_geometry",
                    "true_canonical_orientation",
                    "registration_success_and_strategy",
                    "formal_support_containment",
                    "formal_ftv_retention_q05",
                    "resampling_and_source_overlap",
                    "complete_dce7_builder_and_cache",
                    "leakage_exclusion_contract",
                    "geometry_and_mask_scope_contract",
                    "stage_a_model_ready",
                },
            )
            self.assertTrue(
                table.loc[
                    table["gate"].isin(
                        {
                            "resampling_and_source_overlap",
                            "complete_dce7_builder_and_cache",
                            "stage_a_model_ready",
                        }
                    ),
                    "status",
                ].eq("FAIL").all()
            )
            self.assertTrue((table["status"] == "PASS").sum() >= 7)
            public_closure = (
                (root / "STAGE_A_NO_GO.json").read_text()
                + (root / "metrics/stage_a_model_ready_gate.json").read_text()
                + (root / "reports/stage_a_gate_report.md").read_text()
            )
            self.assertNotIn("PRIVATE_CASE_", public_closure)
            self.assertNotIn(str(root), public_closure)

    def test_incomplete_failure_artifact_cannot_take_early_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metrics").mkdir()
            (root / "metrics/registration_strategy_decision.json").write_text(
                json.dumps({"decision_frozen": True, "chosen_strategy": "H"}),
                encoding="utf-8",
            )
            (root / "metrics/model_input_pipeline_h_validation_gate.json").write_text(
                json.dumps({"status": "FAIL"}), encoding="utf-8"
            )
            self.assertIsNone(
                finalizer._validated_catastrophic_overlap_failure(root=root)
            )


if __name__ == "__main__":
    unittest.main()

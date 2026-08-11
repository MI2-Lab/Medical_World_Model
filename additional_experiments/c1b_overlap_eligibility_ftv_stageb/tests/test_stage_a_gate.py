from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"stage_a_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


audit = load_script("audit_public_artifacts")
finalizer = load_script("finalize_stage_a_gate")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_frame(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    if "private" in path.name:
        path.chmod(0o600)


def make_fixture(base: Path) -> tuple[Path, Path, str]:
    repo = base / "repo"
    root = repo / "additional_experiments/c1b_overlap_eligibility_ftv_stageb"
    prior = repo / "additional_experiments/c1b_model_ready_ftv_sanity"
    provenance_audit = repo / "additional_experiments/zero_overlap_provenance_audit"
    for name in ("configs", "manifests", "metrics", "reports", "figures", "cache/c1b_h"):
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "cache").chmod(0o700)
    (root / "cache/c1b_h").chmod(0o700)
    (root / "EXPERIMENT_PLAN.md").write_text(
        "报告/指标只能包含聚合值：candidate/eligible/excluded。\n", encoding="utf-8"
    )
    plan_hash = "b" * 64
    write_json(
        root / "configs/preregistration_lock.json",
        {
            "schema_version": 1,
            "plan_sha256": plan_hash,
            "preregistered_before_new_cohort_statistics": True,
            "stage_b_requires_stage_a_go": True,
        },
    )
    write_json(root / "configs/upstream_contract_lock.json", {"schema_version": 1})
    write_json(
        root / "configs/stage_a.json",
        {
            "strategy": "C1B-H",
            "eligibility_required_visits": list(finalizer.VISITS),
            "formal_containment_minimum": 0.95,
            "ftv_retention_q05_minimum": 0.95,
            "require_cache_completion_fraction": 1.0,
        },
    )

    source_paths = {
        "candidate_inventory": prior / "manifests/model_input_inventory.private.csv",
        "candidate_key_manifest": prior
        / "metrics/orientation_resampling_patient_visit.private.csv",
        "ispy1_visit_source_eligibility": prior
        / "manifests/ispy1_base_eligibility_visits.private.csv",
    }
    for index, path in enumerate(source_paths.values()):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"private-source-{index}\n", encoding="utf-8")

    patient_rows = [
        {
            "patient_id": "private-case-a",
            "cohort": "synthetic",
            "candidate_visit_count": 4,
            "valid_visit_count": 4,
            "zero_overlap_visit_count": 0,
            "minimum_valid_source_voxels": 7,
            "eligible": True,
            "exclusion_reason": "",
        },
        {
            "patient_id": "private-case-b",
            "cohort": "synthetic",
            "candidate_visit_count": 4,
            "valid_visit_count": 3,
            "zero_overlap_visit_count": 1,
            "minimum_valid_source_voxels": 0,
            "eligible": False,
            "exclusion_reason": "ZERO_VALID_SOURCE_OVERLAP_IN_REQUIRED_VISIT",
        },
    ]
    visit_rows: list[dict[str, object]] = []
    grid_rows: list[dict[str, object]] = []
    shape_zyx = (112, 176, 160)
    spacing_xyz = (0.9, 0.9, 2.0)
    grid_center = (0.0, 0.0, 0.0)
    grid_affine = finalizer.make_c1b_grid(grid_center).affine_ras.tolist()
    source_shape = (16, 16, 16)
    source_affine = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    for patient_id in ("private-case-a", "private-case-b"):
        grid_digest = finalizer.frozen_grid_contract_sha256(
            patient_id=patient_id,
            cohort="synthetic",
            grid_shape_zyx=shape_zyx,
            grid_spacing_xyz_mm=spacing_xyz,
            grid_affine_ras=grid_affine,
        )
        geometry_digest = finalizer.geometry_contract_sha256(
            source_shape_xyz=source_shape,
            source_affine_ras=source_affine,
            grid_shape_xyz=tuple(reversed(shape_zyx)),
            grid_affine_ras=grid_affine,
        )
        grid_rows.append(
            {
                "patient_id": patient_id,
                "cohort": "synthetic",
                "grid_shape_zyx_json": json.dumps(shape_zyx),
                "grid_spacing_xyz_mm_json": json.dumps(spacing_xyz),
                "grid_affine_ras_json": json.dumps(grid_affine),
                "grid_contract_sha256": grid_digest,
                "grid_center_x_ras_mm": grid_center[0],
                "grid_center_y_ras_mm": grid_center[1],
                "grid_center_z_ras_mm": grid_center[2],
            }
        )
        for visit in finalizer.VISITS:
            valid = 0 if patient_id == "private-case-b" and visit == "T2" else 7
            visit_rows.append(
                {
                    "patient_id": patient_id,
                    "cohort": "synthetic",
                    "visit": visit,
                    "valid_source_voxels": valid,
                    "target_grid_voxels": 112 * 176 * 160,
                    "has_valid_source_overlap": valid > 0,
                    "eligibility_evidence_scope": "source_geometry_x_frozen_c1b_h_grid_only",
                    "source_shape_xyz_json": json.dumps(source_shape),
                    "source_affine_ras_json": json.dumps(source_affine),
                    "grid_contract_sha256": grid_digest,
                    "geometry_contract_sha256": geometry_digest,
                }
            )
    patient_path = root / "manifests/technical_eligibility_patients.private.csv"
    visit_path = root / "manifests/technical_eligibility_visits.private.csv"
    eligible_inventory_path = root / "manifests/eligible_model_input_inventory.private.csv"
    grid_path = root / "manifests/frozen_c1b_grids.private.csv"
    write_frame(patient_path, patient_rows)
    write_frame(visit_path, visit_rows)
    write_frame(
        eligible_inventory_path,
        [row for row in visit_rows if row["patient_id"] == "private-case-a"],
    )
    write_frame(grid_path, grid_rows)
    write_json(
        root / "metrics/frozen_c1b_grid_materialization.json",
        {
            "status": "PASS",
            "candidate_grid_rows": 2,
            "private_grid_manifest_sha256": sha256(grid_path),
            "preregistration_plan_sha256": plan_hash,
            "computes_eligibility": False,
            "evaluates_followup_overlap": False,
        },
    )
    write_json(
        root / "metrics/technical_eligibility_summary.json",
        {
            "schema_version": 1,
            "status": "PASS",
            "candidate_patients": 2,
            "candidate_visits": 8,
            "eligible_patients": 1,
            "excluded_patients": 1,
            "valid_visits": 7,
            "zero_overlap_visits": 1,
            "eligibility_input_allowlist": [
                "imaging_source",
                "raw_or_rebuilt_source_geometry",
                "frozen_c1b_h_physical_grid",
                "valid_source_overlap",
            ],
            "clinical_treatment_subtype_fields_read": [],
            "lesion_ftv_ld_sph_bpe_fields_read": [],
            "model_loss_representation_performance_fields_read": [],
            "outcome_pcr_fields_read": [],
            "patient_specific_rules": [],
            "hardcoded_population_result": False,
            "run_is_new_and_does_not_amend_prior_no_go": True,
            "preregistered_before_eligibility_results": True,
            "preregistration_plan_sha256": plan_hash,
            "contains_patient_identifiers": False,
            "private_manifest_sha256": {
                patient_path.name: sha256(patient_path),
                visit_path.name: sha256(visit_path),
                eligible_inventory_path.name: sha256(eligible_inventory_path),
                grid_path.name: sha256(grid_path),
            },
            "source_provenance_sha256": {
                name: sha256(path) for name, path in source_paths.items()
            },
        },
    )

    support_rows: list[dict[str, object]] = []
    grounding_rows: list[dict[str, object]] = []
    for patient_id in ("private-case-a", "private-case-b"):
        for visit in finalizer.VISITS:
            support_rows.append(
                {
                    "patient_id": patient_id,
                    "visit": visit,
                    "strategy": "C1B-H",
                    "physical_volume_retention": 1.0,
                    "exact_full_support_containment": True,
                    "source_boundary_touch": False,
                }
            )
            grounding_rows.append(
                {
                    "patient_id": patient_id,
                    "visit": visit,
                    "source_boundary_touch": False,
                    "ftv_measurement_valid": True,
                    "grounding_observable_mask": True,
                }
            )
    support_path = prior / "metrics/support_containment_patient_visit.private.csv"
    grounding_path = prior / "manifests/grounding_observability_manifest.private.csv"
    write_frame(support_path, support_rows)
    write_frame(grounding_path, grounding_rows)
    write_json(
        prior / "metrics/support_containment_h_summary.json",
        {
            "strategy": "C1B-H",
            "formal_patients": 2,
            "formal_visits": 8,
            "exact_full_support_containment_rate": 1.0,
            "physical_volume_retention": {"q05": 1.0},
        },
    )
    write_json(
        prior / "manifests/grounding_observability_summary.json",
        {
            "scope": "grounding_loss_eligibility_only",
            "formal_patients": 2,
            "formal_visits": 8,
            "is_model_input": False,
            "does_not_filter_base_training": True,
            "private_manifest_sha256": sha256(grounding_path),
        },
    )
    write_json(
        prior / "metrics/dicom_pixel_rebuild_gate.json",
        {
            "status": "PASS",
            "all_model_input_singular_visits": {
                "visits": 2,
                "passed": 2,
                "failed": 0,
                "pixel_order_verified_fraction": 1.0,
                "max_cell_error": 0.0,
                "max_footprint_corner_error_mm": 0.0,
                "all_finite_nonconstant": True,
                "all_qform_sform_valid": True,
                "decoded_cells": 5,
                "verified_cells": 5,
            },
        },
    )
    write_json(
        prior / "metrics/orientation_validation_gate.json",
        {
            "canonical_ras_fraction": 1.0,
            "model_input_patients": 2,
            "model_input_visits": 8,
            "array_reordering_implemented_and_unit_tested": True,
            "header_only_label_change": False,
            "left_right_consistent": True,
            "anterior_posterior_consistent": True,
            "superior_inferior_consistent": True,
            "canonical_roundtrip_corner_error_mm_max": 0.0,
            "dce_mask_footprint_corner_error_mm_max": 0.0,
        },
    )
    write_json(
        prior / "metrics/registration_strategy_decision.json",
        {
            "decision_frozen": True,
            "chosen_strategy": "H",
            "manual_review_complete": True,
            "manual_review_pass": True,
            "r_rejected": True,
            "safe_phase_resample": {"valid_source_mask_is_model_input": False},
        },
    )
    write_json(
        prior / "STAGE_A_NO_GO.json",
        {
            "status": "NO-GO",
            "stage_b_authorized": False,
            "chosen_input_strategy": "C1B-H",
        },
    )
    write_json(
        provenance_audit / "AUDIT_NOT_REPAIRABLE.json",
        {
            "decision": "AUDIT-NOT-REPAIRABLE",
            "repair_allowed": False,
            "manual_transform_trials": 0,
            "registration_transform_used_for_repair": False,
            "prior_stage_a_decision_modified": False,
            "c1b_crop_contract_modified": False,
        },
    )

    builder_hash = "a" * 64
    token = hashlib.sha256(b"private-case-a").hexdigest()
    cache_path = root / "cache/c1b_h" / f"{token}.npz"
    cache_path.write_bytes(b"synthetic-cache")
    cache_path.chmod(0o600)
    cache_file_hash = sha256(cache_path)
    cache_row = {
        "patient_id": "private-case-a",
        "cohort": "synthetic",
        "strategy": "H",
        "scope": "all",
        "selection_reasons": "technical_eligibility_all",
        "cache_path": str(cache_path),
        "cache_content_sha256": "c" * 64,
        "cache_file_sha256": cache_file_hash,
        "cache_schema_version": 3,
        "cache_patient_identity_match": True,
        "cache_complete_input_contract_match": True,
        "builder_contract_sha256": builder_hash,
        "input_provenance_sha256": "d" * 64,
        "base_only_later_support_loaded_count": 0,
        "shape_valid": True,
        "dtype_float32": True,
        "finite": True,
        "whole_visit_nonconstant": True,
        "phase_indices_in_range": True,
        "canonical_orientation": "RAS+",
        "grid_shape_zyx": "112x176x160",
        "grid_spacing_xyz_mm": "0.9x0.9x2.0",
        "valid_source_voxels_json": "[7, 7, 7, 7]",
        "eligibility_valid_source_voxels_json": "[7, 7, 7, 7]",
        "exact_valid_source_count_match": True,
        "frozen_grid_center_match": True,
        "cache_roundtrip_pass": True,
        "model_loader_only_dce7": True,
    }
    metrics_path = root / "metrics/model_input_pipeline_h_all.private.csv"
    inventory_path = root / "manifests/model_input_cache_inventory.private.csv"
    write_frame(metrics_path, [cache_row])
    write_frame(
        inventory_path,
        [
            {
                key: cache_row[key]
                for key in (
                    "patient_id",
                    "cohort",
                    "cache_path",
                    "cache_file_sha256",
                    "cache_content_sha256",
                    "builder_contract_sha256",
                    "input_provenance_sha256",
                )
            }
        ],
    )
    write_json(
        root / "metrics/model_input_pipeline_h_all_gate.json",
        {
            "schema_version": 1,
            "stage": "A",
            "strategy": "C1B-H",
            "status": "PASS",
            "eligible_patients": 1,
            "eligible_visits": 4,
            "completed_cache_patients": 1,
            "completed_cache_visits": 4,
            "cache_completion_fraction": 1.0,
            "cache_roundtrip_pass_fraction": 1.0,
            "exact_valid_source_count_match_fraction": 1.0,
            "frozen_grid_center_match_fraction": 1.0,
            "finite_fraction": 1.0,
            "whole_visit_nonconstant_fraction": 1.0,
            "phase_indices_in_range_fraction": 1.0,
            "model_loader_only_dce7_fraction": 1.0,
            "geometry_metadata_is_model_input": False,
            "base_only_later_visit_supports_loaded": 0,
            "private_artifact_sha256": {
                metrics_path.name: sha256(metrics_path),
                inventory_path.name: sha256(inventory_path),
                patient_path.name: sha256(patient_path),
                visit_path.name: sha256(visit_path),
            },
            "stage_b_authorized": False,
            "contains_patient_identifiers": False,
        },
    )
    write_json(
        root / "metrics/upstream_contract_verification.json",
        {
            "status": "PASS",
            "builder_semantic_contract_sha256": builder_hash,
            "file_sha256": {},
            "tracked_trees": {},
            "prior_stage_a_no_go_immutable": True,
            "prior_audit_not_repairable_immutable": True,
            "preregistration_plan_sha256": plan_hash,
        },
    )
    return repo, root, builder_hash


class PublicPrivacyTests(unittest.TestCase):
    def test_slash_prose_is_not_path_but_real_absolute_roots_are(self) -> None:
        pattern = audit.PATTERNS["absolute_workspace_path"]
        self.assertIsNone(
            pattern.search("报告/指标只能包含聚合值：candidate/eligible/excluded")
        )
        self.assertIsNone(pattern.search("文件/content/input provenance"))
        self.assertIsNotNone(pattern.search("source=/data/private/cohort.csv"))
        self.assertIsNotNone(pattern.search("source=/home/user/cohort.csv"))

    def test_private_identifier_is_discovered_and_public_leak_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_frame(
                root / "manifests/subjects.private.csv",
                [{"patient_id": "opaque-private-subject"}],
            )
            (root / "reports").mkdir()
            (root / "reports/summary.md").write_text(
                "opaque-private-subject", encoding="utf-8"
            )
            result = audit.scan_public_artifacts(root=root)
            self.assertEqual(result["status"], "FAIL")
            self.assertTrue(
                any(
                    item["finding"] == "private_manifest_identifier"
                    for item in result["identifier_or_path_findings"]
                )
            )


class StageAFinalizerTests(unittest.TestCase):
    def _verifiers(self, builder_hash: str):
        def preregistration_verifier(*, experiment_root: Path):
            del experiment_root
            return {
                "plan_sha256": "b" * 64,
                "preregistered_before_new_cohort_statistics": True,
                "stage_b_requires_stage_a_go": True,
            }

        def upstream_verifier(*, experiment_root: Path, repo_root: Path):
            del experiment_root, repo_root
            return {
                "status": "PASS",
                "builder_semantic_contract_sha256": builder_hash,
                "file_sha256": {},
                "tracked_trees": {},
            }

        return preregistration_verifier, upstream_verifier

    def test_all_15_pass_writes_go_and_mechanical_formal_intersection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo, root, builder_hash = make_fixture(Path(directory))
            prereg, upstream = self._verifiers(builder_hash)
            payload = finalizer.finalize_stage_a(
                experiment_root=root,
                repo_root=repo,
                preregistration_verifier=prereg,
                upstream_verifier=upstream,
            )
            self.assertEqual(payload["status"], "GO")
            self.assertIs(payload["stage_b_authorized"], True)
            self.assertEqual(payload["chosen_input_strategy"], "C1B-H")
            self.assertEqual(payload["eligible_population_patients"], 1)
            self.assertEqual(payload["eligible_population_visits"], 4)
            self.assertEqual(len(payload["gates"]), 15)
            self.assertTrue(all(row["status"] == "PASS" for row in payload["gates"]))
            self.assertTrue((root / "STAGE_A_GO.json").is_file())
            self.assertFalse((root / "STAGE_A_NO_GO.json").exists())
            table = pd.read_csv(
                root / "metrics/table1_technical_eligibility_stage_a_qc.csv"
            )
            self.assertEqual(len(table), 15)
            formal = json.loads(
                (root / "metrics/formal_eligibility_reaggregation.json").read_text()
            )
            self.assertEqual(formal["formal_patients_before_intersection"], 2)
            self.assertEqual(formal["formal_patients_after_intersection"], 1)
            self.assertEqual(formal["formal_patients_excluded"], 1)
            public_text = (
                (root / "STAGE_A_GO.json").read_text()
                + (root / "reports/stage_a_gate_report.md").read_text()
                + (root / "metrics/table1_technical_eligibility_stage_a_qc.csv").read_text()
            )
            self.assertNotIn("private-case-a", public_text)
            self.assertNotIn(str(root), public_text)

    def test_live_cache_hash_change_forces_no_go_and_unauthorized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo, root, builder_hash = make_fixture(Path(directory))
            cache_file = next((root / "cache/c1b_h").glob("*.npz"))
            cache_file.write_bytes(b"changed-after-cache-gate")
            prereg, upstream = self._verifiers(builder_hash)
            payload = finalizer.finalize_stage_a(
                experiment_root=root,
                repo_root=repo,
                preregistration_verifier=prereg,
                upstream_verifier=upstream,
            )
            self.assertEqual(payload["status"], "NO-GO")
            self.assertIs(payload["stage_b_authorized"], False)
            self.assertFalse((root / "STAGE_A_GO.json").exists())
            self.assertTrue((root / "STAGE_A_NO_GO.json").is_file())
            gates = {row["gate"]: row for row in payload["gates"]}
            self.assertEqual(gates["cache_roundtrip_and_hash"]["status"], "FAIL")

    def test_cache_hardlink_forces_no_go(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo, root, builder_hash = make_fixture(Path(directory))
            cache_file = next((root / "cache/c1b_h").glob("*.npz"))
            os.link(cache_file, cache_file.with_suffix(".linked-copy"))
            prereg, upstream = self._verifiers(builder_hash)
            payload = finalizer.finalize_stage_a(
                experiment_root=root,
                repo_root=repo,
                preregistration_verifier=prereg,
                upstream_verifier=upstream,
            )
            self.assertEqual(payload["status"], "NO-GO")
            gates = {row["gate"]: row for row in payload["gates"]}
            self.assertEqual(gates["cache_roundtrip_and_hash"]["status"], "FAIL")

    def test_group_readable_private_cache_forces_privacy_no_go(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo, root, builder_hash = make_fixture(Path(directory))
            cache_file = next((root / "cache/c1b_h").glob("*.npz"))
            cache_file.chmod(0o640)
            prereg, upstream = self._verifiers(builder_hash)
            payload = finalizer.finalize_stage_a(
                experiment_root=root,
                repo_root=repo,
                preregistration_verifier=prereg,
                upstream_verifier=upstream,
            )
            self.assertEqual(payload["status"], "NO-GO")
            gates = {row["gate"]: row for row in payload["gates"]}
            self.assertEqual(
                gates["eligibility_outcome_free_and_public_private"]["status"],
                "FAIL",
            )
            self.assertEqual(gates["cache_roundtrip_and_hash"]["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()

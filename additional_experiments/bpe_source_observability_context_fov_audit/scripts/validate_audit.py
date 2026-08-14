#!/usr/bin/env python3
"""Fail-closed validation and privacy gate for the BPE FOV audit."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_core import canonical_json_sha256, sha256_file  # noqa: E402


PUBLIC_REQUIRED = (
    "EXPERIMENT_PLAN.md",
    "configs/audit.json",
    "figures/fov_comparison.png",
    "manifests/audit_population.csv",
    "manifests/input_provenance.json",
    "metrics/acquisition_image_support_extent.csv",
    "metrics/boundary_touch_table.csv",
    "metrics/c1b_grid_validation.json",
    "metrics/context_candidate_cost_table.csv",
    "metrics/coverage_table.csv",
    "metrics/decision.json",
    "metrics/laterality_audit.csv",
    "metrics/longitudinal_geometry.csv",
    "metrics/orientation_validation.json",
    "metrics/physical_margin_distribution.csv",
    "metrics/source_geometry_summary.json",
    "reports/bpe_source_contract.md",
    "reports/final_report.md",
    "scripts/audit_core.py",
    "scripts/generate_artifacts.py",
    "scripts/run_audit.py",
    "scripts/validate_audit.py",
    "tests/test_audit_core.py",
)
PRIVATE_REQUIRED = (
    "manifests/acquisition_support.private.csv",
    "manifests/laterality_sample.private.csv",
    "manifests/raw_series_availability.private.csv",
)
FORBIDDEN_SUFFIXES = (
    ".dcm",
    ".nii",
    ".nii.gz",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".ckpt",
    ".xlsx",
    ".xls",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    for relative in (*PUBLIC_REQUIRED, *PRIVATE_REQUIRED):
        require((ROOT / relative).is_file(), f"required artifact missing: {relative}")

    bad = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        lower = path.name.lower()
        if any(lower.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES):
            bad.append(str(path.relative_to(ROOT)))
    require(not bad, f"forbidden source/model artifacts copied into experiment: {bad}")

    decision = load_json(ROOT / "metrics" / "decision.json")
    digest = decision.pop("decision_sha256")
    require(digest == canonical_json_sha256(decision), "decision JSON digest mismatch")
    require(decision["classification_code"] == "D", "scientific class is not D")
    require(decision["classification"] == "BPE_SOURCE_NOT_RELIABLY_AUDITABLE", "classification drifted")
    require(decision["source_roi_status"] == "SOURCE_ROI_NOT_AVAILABLE", "source status drifted")
    require(decision["source_roi_available"] is False, "source ROI unexpectedly available")
    require(decision["next_stage_input"] == "PAUSE_BPE", "next input is not PAUSE_BPE")
    require(decision["context_candidate"] is None, "context candidate was improperly defined")
    require(decision["local_context_phenotype_representation_pilot_authorized"] is False, "pilot improperly authorized")
    require(decision["outcome_fields_read"] == [], "outcome fields were read")
    require(decision["clinical_treatment_fields_read"] == [], "clinical/treatment fields were read")
    require(decision["bpe_values_used_for_selection_or_geometry"] is False, "BPE magnitude affected geometry")
    require(
        all(value["status"] == "NOT_EVALUABLE" for value in decision["fov_gates"].values()),
        "one or more source observability gates were forced",
    )
    require(decision["population"]["matched_primary_patients"] == 375, "primary patient count drifted")
    require(decision["population"]["matched_primary_visits"] == 1500, "primary visit count drifted")
    require(decision["population"]["raw_dicom_selected_series_available"] == 1500, "raw DICOM coverage drifted")
    require(
        decision["f2_geometry_evidence"]["raw_to_reconstructed_full_series_footprint_equivalence_recomputed"] is False,
        "F2 raw/reconstructed evidence boundary drifted",
    )
    require(
        decision["f2_geometry_evidence"]["extent_statistics_are_separate_marginals"] is True,
        "F2 marginal-summary status drifted",
    )

    coverage = pd.read_csv(ROOT / "metrics" / "coverage_table.csv")
    require(set(coverage["fov_contract"]) == {"F0", "F1", "F2"}, "FOV set drifted")
    require(coverage["observability_gate"].eq("NOT_EVALUABLE").all(), "coverage gate was forced")
    require(coverage["source_occupancy_evaluable_visits"].eq(0).all(), "coverage unexpectedly evaluable")
    require(coverage["source_occupancy_ge_0_99_fraction"].isna().all(), "unknown coverage was imputed")

    boundary = pd.read_csv(ROOT / "metrics" / "boundary_touch_table.csv")
    require(boundary["evaluable_visits"].eq(0).all(), "boundary touch unexpectedly evaluable")
    require(boundary["touch_any_visits"].isna().all(), "unknown boundary touch was imputed")
    margin = pd.read_csv(ROOT / "metrics" / "physical_margin_distribution.csv")
    require(margin["n"].eq(0).all(), "margin unexpectedly evaluable")
    require(margin["q05_mm"].isna().all(), "unknown margin was imputed")
    longitudinal = pd.read_csv(ROOT / "metrics" / "longitudinal_geometry.csv")
    require(longitudinal["evaluable_patients"].eq(0).all(), "longitudinal ROI unexpectedly evaluable")
    cost = pd.read_csv(ROOT / "metrics" / "context_candidate_cost_table.csv")
    f2_cost = cost.loc[cost["contract"].eq("F2_RECONSTRUCTED_SOURCE_SUPPORT_MARGINAL_Q50")]
    require(len(f2_cost) == 1, "F2 descriptive cost row missing")
    require(
        f2_cost.iloc[0]["role"] == "separate_marginal_summaries_not_one_realizable_tensor",
        "F2 marginal cost row overclaims a realizable tensor",
    )

    source_geometry = load_json(ROOT / "metrics" / "source_geometry_summary.json")
    require(source_geometry["source_affine_oblique_visits"] == 53, "source obliquity count drifted")
    require(
        source_geometry["raw_to_reconstructed_full_series_footprint_equivalence_recomputed"] is False,
        "source geometry overclaims raw/reconstructed footprint equivalence",
    )

    grid = load_json(ROOT / "metrics" / "c1b_grid_validation.json")
    require(grid["status"] == "PASS" and grid["patients"] == 375, "C1B grid audit failed")
    require(grid["shape_zyx"] == [112, 176, 160], "C1B shape drifted")
    require(grid["spacing_xyz_mm"] == [0.9, 0.9, 2.0], "C1B spacing drifted")
    require(grid["extent_xyz_mm"] == [144.0, 158.4, 224.0], "C1B extent drifted")

    laterality = pd.read_csv(ROOT / "metrics" / "laterality_audit.csv").iloc[0]
    require(int(laterality["sample_patients"]) == 20, "laterality sample is not 20 patients")
    require(int(laterality["sample_visits"]) == 80, "laterality sample is not four visits/patient")
    require(float(laterality["orientation_after_ras_fraction"]) == 1.0, "sample RAS validation failed")
    require(bool(laterality["source_roi_available"]) is False, "laterality table claims source ROI")

    with Image.open(ROOT / "figures" / "fov_comparison.png") as image:
        require(image.format == "PNG", "FOV figure is not PNG")
        require(image.width >= 1200 and image.height >= 800, "FOV figure resolution is too small")
        image.verify()

    source_report = (ROOT / "reports" / "bpe_source_contract.md").read_text(encoding="utf-8")
    require("SOURCE_ROI_NOT_AVAILABLE" in source_report, "source report omits fail-closed status")
    require("BPE_source_semantics" in source_report, "source semantics heading missing")
    require("BPE_source_geometry_requirements" in source_report, "source geometry heading missing")
    final_report = (ROOT / "reports" / "final_report.md").read_text(encoding="utf-8")
    for number in range(1, 15):
        require(f"### {number}." in final_report, f"final report question {number} missing")
    for marker in (
        "BPE_SOURCE_NOT_RELIABLY_AUDITABLE",
        "PAUSE_BPE",
        "NOT AUTHORIZED",
        "SOURCE_ROI_NOT_AVAILABLE",
        "Audit commit SHA",
        "Push status",
    ):
        require(marker in final_report, f"final report marker missing: {marker}")

    identifier_pattern = re.compile(r"(?:ISPY2-|ACRIN-6698-)\d{6}")
    public_rows = []
    for relative in PUBLIC_REQUIRED:
        path = ROOT / relative
        if path.suffix.lower() in {".md", ".json", ".csv", ".py"}:
            text = path.read_text(encoding="utf-8", errors="strict")
            require(not identifier_pattern.search(text), f"patient identifier leaked into {relative}")
        public_rows.append(
            {
                "artifact": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "contains_patient_identifiers": False,
            }
        )

    for relative in PRIVATE_REQUIRED:
        private_path = ROOT / relative
        completed = subprocess.run(
            ["git", "check-ignore", "-q", str(private_path)],
            cwd=REPO_ROOT,
            check=False,
        )
        require(completed.returncode == 0, f"private artifact is not gitignored: {relative}")
        require(
            private_path.stat().st_mode & 0o777 == 0o600,
            f"private artifact permissions are not 0600: {relative}",
        )

    changed_old = subprocess.run(
        ["git", "diff", "--name-only", "--", "additional_experiments"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    require(
        all(path.startswith("additional_experiments/bpe_source_observability_context_fov_audit/") for path in changed_old),
        f"old experiment directory modified: {changed_old}",
    )

    manifest_path = ROOT / "manifests" / "public_artifacts.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(public_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(public_rows)
    validation = {
        "schema_version": 1,
        "status": "PASS",
        "classification": "BPE_SOURCE_NOT_RELIABLY_AUDITABLE",
        "source_roi_status": "SOURCE_ROI_NOT_AVAILABLE",
        "public_artifacts": len(public_rows),
        "private_artifacts_gitignored": len(PRIVATE_REQUIRED),
        "tests_expected": 9,
        "outcome_fields_read": [],
        "contains_patient_identifiers": False,
    }
    (ROOT / "metrics" / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(validation, sort_keys=True))


if __name__ == "__main__":
    main()

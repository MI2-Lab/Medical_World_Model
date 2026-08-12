from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / "scripts"))

import generate_figures as figures  # noqa: E402
import generate_report as report  # noqa: E402
import validate_results as validator  # noqa: E402


def _write_csv(root: Path, name: str, rows: list[dict[str, object]]) -> None:
    path = root / "metrics" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    logical = next(
        (logical for logical, filename in figures.PUBLIC_TABLES.items() if filename == name),
        None,
    )
    if logical in validator.PUBLIC_TABLE_COLUMNS:
        if "timing" in frame and "view" not in frame:
            frame["view"] = frame["timing"]
        if "view" in frame and "timing" not in frame:
            frame["timing"] = frame["view"]
        if "variant" not in frame and "candidate" in frame:
            frame["variant"] = frame["candidate"]
        if logical == "clinical_ftv_incremental":
            frame["delta_auroc_vs_C+F+R0"] = frame["delta_auroc_vs_cf_r0"]
            frame["delta_auroc"] = frame["delta_auroc_vs_cf_r0"]
        defaults: dict[str, object] = {
            "seed": 2026,
            "arm": "LOCAL0",
            "analysis": str(logical),
            "context": "MRI_ONLY",
            "view": "T0",
            "timing": "T0",
            "timing_label": "",
            "target": "pCR",
            "variant": "R0",
            "model": "R0",
            "population": "full_808",
            "clinical_contract": "",
            "n": 808,
            "n_positive": 220,
            "n_negative": 588,
            "n_classes": 2,
            "auroc": 0.62,
            "auprc": 0.38,
            "balanced_accuracy": 0.58,
            "brier": 0.20,
            "visit": "T0",
            "r0_auroc": 0.61,
            "c_auroc": 0.61,
            "c_auprc": 0.36,
            "c_brier": 0.21,
            "delta_auroc_vs_r0": 0.0,
            "delta_auroc": 0.0,
            "delta_auroc_vs_C": 0.0,
            "delta_auprc_vs_C": 0.0,
            "brier_improvement_vs_C": 0.0,
            "delta_auroc_vs_C+F+R0": 0.0,
            "delta_auprc_vs_C+F+R0": 0.0,
            "brier_improvement_vs_C+F+R0": 0.0,
            "delta_auroc_vs_C+F": 0.0,
            "delta_auprc_vs_C+F": 0.0,
            "brier_improvement_vs_C+F": 0.0,
            "delta_auroc_vs_cf_r0": 0.0,
            "feature_dim": 128,
            "task": "static",
            "endpoint": "T0",
            "analysis_scope": "primary_measurement_valid",
            "target_semantics": "synthetic",
            "aggregation": "pooled_outer_test_folds",
            "n_test": 375,
            "spearman": 0.3,
            "pearson": 0.3,
            "r2": 0.2,
            "rmse": 1.0,
            "mae": 0.8,
            "b0_rmse": 1.2,
            "rmse_gain_over_b0": 0.16,
            "prediction_target_variance_ratio": 0.7,
            "calibration_slope": 0.9,
            "calibration_intercept": 0.0,
            "calibration_mean_bias": 0.0,
            "row_type": "mask_free",
            "source": "new",
            "candidate": "R1",
            "reference": "R0",
            "candidate_auroc": 0.62,
            "numerator_auroc_uplift": 0.01,
            "published_fixed_p3_auroc": 0.60,
            "published_peri20_auroc": 0.635,
            "published_oracle_uplift": 0.035,
            "recovery_ratio": 0.3,
            "recovery_defined": True,
            "representation_note": "synthetic mismatch disclosure",
            "matched_patient_sha256": "d" * 64,
            "reference_model": "R0",
            "comparison_model": "R1",
            "metric": "auroc",
            "comparison": 0.62,
            "estimate": 0.01,
            "improvement": 0.01,
            "ci_lower": -0.01,
            "ci_upper": 0.03,
            "confidence_level": 0.95,
            "n_patients": 375,
            "n_folds": 5,
            "n_bootstrap": 2000,
            "n_valid_bootstrap": 2000,
            "bootstrap_unit": "patient_within_outer_fold",
            "ci_method": "percentile",
            "orientation": "comparison_minus_reference",
            "bootstrap_seed": 260811,
            "seed_2026_delta_auroc": 0.01,
            "seed_3026_delta_auroc": 0.01,
            "mean_delta_auroc": 0.01,
            "both_seeds_strictly_positive": True,
            "reference_auroc": 0.61,
        }
        for column in validator.PUBLIC_TABLE_COLUMNS[logical]:
            if column not in frame:
                frame[column] = defaults[column]
        frame = frame.reindex(columns=validator.PUBLIC_TABLE_COLUMNS[logical])
    frame.to_csv(path, index=False)


def _init_fixture(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    root = repository / "additional_experiments" / "mask_free_region_aware_audit"
    for directory in ("configs", "metrics", "figures", "reports", "features"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    (root / "configs" / "audit.json").write_text(
        json.dumps(
            {
                "branch": "feature/mask-free-region-aware-audit",
                "feature_contract": {"primary_boundaries_mm": [32.0, 48.0, 64.0]},
            }
        ),
        encoding="utf-8",
    )
    (root / ".gitignore").write_text(
        "features/**\n!features/.gitkeep\npredictions/**\nlogs/**\n",
        encoding="utf-8",
    )
    (root / "PREREGISTRATION_LOCK.json").write_text(
        '{"status":"FROZEN"}\n', encoding="utf-8"
    )
    (root / "metrics" / "region_occupancy_contract.json").write_text(
        '{"status":"COMPLETE"}\n', encoding="utf-8"
    )

    regions = list(figures.PRIMARY_REGIONS)
    _write_csv(
        root,
        figures.PUBLIC_TABLES["occupancy"],
        [
            {
                "geometry": "primary",
                "region": region,
                "variant": region,
                "mean_effective_cells": 12 + index * 3,
            }
            for index, region in enumerate(regions)
        ],
    )

    mri_rows: list[dict[str, object]] = []
    for seed in (2026, 3026):
        for timing_index, timing in enumerate(("T0", "T0-T1", "T0-T2", "T0-T3")):
            for region_index, region in enumerate(regions):
                gain = {"R0": 0.0, "R1": 0.010, "R2": 0.018, "R3": 0.022, "R4": 0.020, "R5": 0.026}[region]
                mri_rows.append(
                    {
                        "seed": seed,
                        "arm": "LOCAL0",
                        "timing": timing,
                        "target": "pCR",
                        "population": "full_808",
                        "variant": region,
                        "n": 808,
                        "auroc": 0.61 + 0.005 * timing_index + gain,
                    }
                )
    _write_csv(root, figures.PUBLIC_TABLES["mri_only_pcr"], mri_rows)
    _write_csv(root, figures.PUBLIC_TABLES["clinical_pcr"], mri_rows[:12])

    incremental_rows = []
    timing_rows = []
    for seed in (2026, 3026):
        for timing in ("T0", "T0-T1", "T0-T2", "T0-T3"):
            for region in regions:
                delta = {"R0": 0.0, "R1": 0.004, "R2": 0.007, "R3": 0.009, "R4": 0.008, "R5": 0.011}[region]
                incremental_rows.append(
                    {
                        "seed": seed,
                        "arm": "LOCAL0",
                        "timing": timing,
                        "target": "pCR",
                        "variant": region,
                        "delta_auroc_vs_cf_r0": delta,
                    }
                )
                timing_rows.append(
                    {
                        "seed": seed,
                        "arm": "LOCAL0",
                        "timing": timing,
                        "variant": region,
                        "delta_auroc_vs_r0": delta + (0.01 if region != "R0" else 0),
                    }
                )
    _write_csv(root, figures.PUBLIC_TABLES["clinical_ftv_incremental"], incremental_rows)
    _write_csv(root, figures.PUBLIC_TABLES["timing_sensitivity"], timing_rows)

    phenotype_rows = []
    for seed in (2026, 3026):
        for target in ("HR", "HER2", "subtype_4class"):
            for region in regions:
                phenotype_rows.append(
                    {
                        "seed": seed,
                        "arm": "LOCAL0",
                        "visit": "T0",
                        "target": target,
                        "variant": region,
                        "auroc": 0.60 + 0.004 * regions.index(region),
                    }
                )
    _write_csv(root, figures.PUBLIC_TABLES["phenotype"], phenotype_rows)

    ftv_rows = []
    for endpoint in ("FTV", "delta_FTV"):
        for timing in ("T0", "T0-T1"):
            for region in regions:
                ftv_rows.append(
                    {
                        "seed": 2026,
                        "arm": "LOCAL0",
                        "timing": timing,
                        "endpoint": endpoint,
                        "variant": region,
                        "r2": 0.20 + 0.01 * regions.index(region),
                    }
                )
    _write_csv(root, figures.PUBLIC_TABLES["ftv"], ftv_rows)

    oracle_rows = []
    for seed, denominator in ((2026, 0.0357), (3026, 0.0333)):
        for region, ratio in (("R1", 0.15), ("R2", 0.25), ("R3", 0.32), ("R5", 0.40)):
            oracle_rows.append(
                {
                    "seed": seed,
                    "arm": "LOCAL0",
                    "timing": "T0-T1",
                    "variant": region,
                    "numerator": ratio * denominator,
                    "denominator": denominator,
                    "recovery_ratio": ratio,
                }
            )
    _write_csv(root, figures.PUBLIC_TABLES["oracle_recovery"], oracle_rows)

    bootstrap_rows = []
    for context in ("MRI_ONLY", "C_PLUS_F"):
        for timing in ("T0", "T0-T1", "T0-T2"):
            for region in ("R2", "R3", "R5"):
                estimate = 0.01 + 0.003 * ("R2", "R3", "R5").index(region)
                bootstrap_rows.append(
                    {
                        "context": context,
                        "timing": timing,
                        "candidate": region,
                        "estimate": estimate,
                        "ci_lower": estimate - 0.025,
                        "ci_upper": estimate + 0.025,
                    }
                )
    _write_csv(root, figures.PUBLIC_TABLES["bootstrap"], bootstrap_rows)
    _write_csv(
        root,
        figures.PUBLIC_TABLES["seed_consistency"],
        [
            {
                "timing": timing,
                "candidate": region,
                "seed_2026_delta_auroc": 0.01 + index * 0.004,
                "seed_3026_delta_auroc": 0.012 + index * 0.003,
            }
            for timing in ("T0", "T0-T1", "T0-T2")
            for index, region in enumerate(("R2", "R3", "R5"))
        ],
    )
    (root / "metrics" / "gates.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment": "mask_free_region_aware_audit",
                "status": "COMPLETE",
                "gates": {
                    letter: {
                        "name": name,
                        "passed": letter != "D",
                        "evaluated_comparisons": [],
                        "supporting_comparisons": [],
                    }
                    for letter, name in (
                        ("A", "MASK_FREE_REGIONAL_SIGNAL_SUPPORTED"),
                        ("B", "MASK_FREE_BEYOND_FTV_SUPPORTED"),
                        ("C", "ORACLE_SIGNAL_PARTIALLY_RECOVERED"),
                        ("D", "PROFILE_ASSOCIATED_REGIONAL_SIGNAL_SUPPORTED"),
                    )
                },
                "any_primary_candidate_two_seed_positive": True,
                "scientific_classification": "DEPLOYABLE_REGION_AWARE_SIGNAL_SUPPORTED",
                "classification_precedence": ["synthetic"],
                "contains_patient_level_data": False,
            }
        ),
        encoding="utf-8",
    )
    private_paths = {}
    for relative in validator.RUN_PRIVATE_OUTPUTS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic private aggregate fixture\n", encoding="utf-8")
        path.chmod(0o600)
        private_paths[relative] = figures.file_sha256(path)
    public_paths = {
        f"metrics/{filename}": figures.file_sha256(root / "metrics" / filename)
        for filename in figures.PUBLIC_TABLES.values()
    }
    (root / "metrics" / "run_summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment": "mask_free_region_aware_audit",
                "status": "COMPLETED",
                "branch": "feature/mask-free-region-aware-audit",
                "commit_sha": "not-yet-recorded",
                "push_status": "not-yet-recorded",
                "elapsed_seconds": 12.3,
                "feature_cells_validated_before_labels": 20,
                "formal_bootstrap_replicates": 2000,
                "outer_test_predicted_once_per_model": True,
                "new_encoder_or_jepa_training_performed": False,
                "architecture_intervention_started": False,
                "r0_goal5_p1_prediction_parity": {"passed": True},
                "public_outputs_contain_patient_level_data": False,
                "private_patient_outputs_mode": "0600",
                "scientific_classification": "DEPLOYABLE_REGION_AWARE_SIGNAL_SUPPORTED",
                "gate_results": {"A": True, "B": True, "C": True, "D": False},
                "public_outputs": public_paths,
                "private_outputs": private_paths,
            }
        ),
        encoding="utf-8",
    )
    completion_cells = []
    for seed in (2026, 3026):
        for arm in ("LOCAL0", "LOCAL3"):
            for fold in range(5):
                directory = root / "features" / f"seed_{seed}" / arm / f"fold_{fold}"
                directory.mkdir(parents=True, exist_ok=True)
                feature = directory / "regional_features.private.bin"
                metadata = directory / "regional_features.private.metadata.json"
                feature.write_bytes(f"synthetic-{seed}-{arm}-{fold}".encode())
                metadata.write_text('{"synthetic":true}\n', encoding="utf-8")
                feature.chmod(0o600)
                metadata.chmod(0o600)
                completion_cells.append(
                    {
                        "cell": f"seed_{seed}/{arm}/fold_{fold}",
                        "seed_base": seed,
                        "arm": arm,
                        "fold": fold,
                        "feature_path": str(feature),
                        "feature_sha256": figures.file_sha256(feature),
                        "metadata_path": str(metadata),
                        "metadata_sha256": figures.file_sha256(metadata),
                        "patient_order_sha256": "a" * 64,
                    }
                )
    completion = root / "features" / "feature_matrix_complete.private.json"
    completion.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment": "mask_free_region_aware_audit",
                "status": "COMPLETE",
                "cell_count": 20,
                "config_sha256": figures.file_sha256(root / "configs" / "audit.json"),
                "preregistration_lock_sha256": figures.file_sha256(
                    root / "PREREGISTRATION_LOCK.json"
                ),
                "geometry_contract_sha256": figures.file_sha256(
                    root / "metrics" / "region_occupancy_contract.json"
                ),
                "cells": completion_cells,
            }
        ),
        encoding="utf-8",
    )
    completion.chmod(0o600)
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    return root


def test_end_to_end_public_reporting_and_validation(tmp_path: Path) -> None:
    root = _init_fixture(tmp_path)
    manifest = figures.generate_all(root)
    assert tuple(manifest["figure"]) == figures.FIGURES
    assert all((root / "figures" / name).stat().st_size > 1000 for name in figures.FIGURES)

    destination = report.generate_report(root)
    text = destination.read_text(encoding="utf-8")
    assert all(marker in text for marker in report.REQUIRED_REPORT_MARKERS)
    assert "### Q12 —" in text
    assert "/data/" not in text

    subprocess.run(
        [
            "git",
            "add",
            "configs",
            "metrics",
            "figures",
            "reports",
        ],
        cwd=root,
        check=True,
    )
    result = validator.validate_all(root)
    assert result["status"] == "PASS"
    assert result["checks"]["public_privacy"]["patient_identifier_findings"] == 0
    assert result["checks"]["tracked_artifact_safety"]["raw_mri_or_checkpoint_files"] == 0


def test_public_reader_rejects_patient_identifier_column(tmp_path: Path) -> None:
    root = tmp_path / "audit"
    _write_csv(
        root,
        figures.PUBLIC_TABLES["occupancy"],
        [{"patient_id": "example", "region": "R0", "mean_effective_cells": 2}],
    )
    with pytest.raises(ValueError, match="identifier"):
        figures.read_public_table(root, figures.PUBLIC_TABLES["occupancy"])


def test_old_tree_hash_manifest_detects_change(tmp_path: Path) -> None:
    root = _init_fixture(tmp_path)
    old = root.parents[1] / "additional_experiments" / "existing_audit"
    old.mkdir()
    artifact = old / "evidence.txt"
    artifact.write_text("frozen\n", encoding="utf-8")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = root / "old_tree.json"
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": artifact.relative_to(root.parents[1]).as_posix(),
                        "sha256": digest,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert validator.verify_old_tree_manifest(root, manifest)["status"] == "PASS"
    artifact.write_text("changed\n", encoding="utf-8")
    with pytest.raises(validator.ValidationError, match="non-modification"):
        validator.verify_old_tree_manifest(root, manifest)


def test_tracked_checkpoint_is_rejected(tmp_path: Path) -> None:
    root = _init_fixture(tmp_path)
    checkpoint = root / "checkpoints" / "model.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"not a real checkpoint")
    subprocess.run(["git", "add", "."], cwd=root.parents[1], check=True)
    with pytest.raises(validator.ValidationError, match="raw_or_checkpoint"):
        validator.validate_tracked_artifacts(root)


def test_gate_parser_rejects_conflicting_representations() -> None:
    payload = {
        "gate_a": True,
        "gates": {"A": {"passed": False}, "B": False, "C": False, "D": False},
    }
    with pytest.raises(ValueError, match="conflicting Gate A"):
        report.extract_gate_results(payload)


def test_classification_c_detects_subthreshold_two_seed_signal() -> None:
    tables = {
        "seed_consistency": pd.DataFrame(
            [
                {
                    "candidate": "R3",
                    "timing": "T0-T1",
                    "seed_2026_delta_auroc": 0.005,
                    "seed_3026_delta_auroc": 0.006,
                }
            ]
        )
    }
    code, label = report.derive_classification(
        {"A": False, "B": False, "C": False, "D": False}, tables
    )
    assert code == "C"
    assert label == report.CLASSIFICATION_LABELS["C"]

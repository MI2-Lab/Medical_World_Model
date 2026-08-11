from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from verify_experiment import (  # noqa: E402
    EXPECTED_BRANCH,
    INCREMENTAL_COLUMNS,
    KEY_COMPARISONS,
    MATCHED_COLUMNS,
    MRI_TRADITIONAL_COMPARISON_COLUMNS,
    PCR_COLUMNS,
    PRIMARY_MODELS,
    REQUIRED_FIGURES,
    TIMINGS,
    VIEWS,
    verify_experiment,
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _timing_label(timing: str) -> str:
    return "T3 (late/pre-surgery)" if timing == "T3" else timing


def _build_valid_tree(tmp_path: Path) -> tuple[Path, Path]:
    """Create a tiny aggregate-only experiment; no real output is touched."""

    repo = tmp_path / "repo"
    root = repo / "additional_experiments" / "classical_dce_phenotype_complementarity"
    for directory in ("configs", "features", "predictions", "metrics", "figures", "logs", "reports"):
        (root / directory).mkdir(parents=True, exist_ok=True)

    _git(repo.parent, "init", "-q", str(repo))
    _git(repo, "config", "user.name", "Verification Test")
    _git(repo, "config", "user.email", "verification@example.invalid")
    _git(repo, "checkout", "-q", "-b", EXPECTED_BRANCH)

    (root / ".gitignore").write_text(
        "features/*.private.*\n"
        "predictions/*.private.*\n"
        "logs/*.private.*\n"
        "configs/*patient*.private.*\n"
        "__pycache__/\n"
        ".pytest_cache/\n",
        encoding="utf-8",
    )
    # This represents a patient-level intermediate and is deliberately ignored.
    (root / "predictions" / "pcr_oof.private.csv").write_text(
        "patient_id,prediction\nredacted,0.5\n", encoding="utf-8"
    )

    source_dir = tmp_path / "synthetic_sources"
    source_dir.mkdir()
    source_contracts: list[dict[str, Any]] = []
    for name, columns in (
        ("radiomics", ("feature_a", "feature_b")),
        ("clinical", ("outcome", "profile")),
        ("fold_manifest", ("fold", "split")),
    ):
        path = source_dir / f"{name}.csv"
        pd.DataFrame([[1, 2]], columns=columns).to_csv(path, index=False)
        source_contracts.append(
            {
                "name": name,
                "path": str(path),
                "sha256": _sha256(path),
                "format": "csv",
                "columns": list(columns),
                "rows": 1,
            }
        )
    _write_json(
        root / "configs" / "experiment.json",
        {
            "experiment_name": "Classical DCE Phenotype Complementarity Baseline",
            "bootstrap_draws": 2000,
            "primary_population": "clinical_radiomics_complete_384",
            "verification": {"source_contracts": source_contracts},
        },
    )

    pd.DataFrame(
        [
            ("T0", "T0", "T0", "T0", "none", "pretreatment"),
            ("T1", "T0|T1", "T1", "T0|T1", "T0_to_T1", "early_NAC"),
            (
                "T2",
                "T0|T1|T2",
                "T2",
                "T0|T1|T2",
                "T0_to_T1|T0_to_T2",
                "inter_regimen",
            ),
            (
                "T3",
                "T0|T1|T2|T3",
                "T3",
                "T0|T1|T2|T3",
                "T0_to_T1|T0_to_T2|T0_to_T3",
                "late/pre-surgery",
            ),
        ],
        columns=(
            "timing",
            "allowed_visits",
            "static_features",
            "longitudinal_absolute",
            "longitudinal_change",
            "label",
        ),
    ).to_csv(root / "information_timing_contract.csv", index=False)

    pd.DataFrame(
        [
            {
                "protocol": "primary_stratified_384",
                "population": "clinical_radiomics_complete_384",
                "scenario": scenario,
                "view": view,
                "timing": timing,
                "timing_label": _timing_label(timing),
                "model_type": model_type,
                "model": model,
                "n": 384,
                "n_positive": 113,
                "n_negative": 271,
                "auroc": 0.65,
                "auprc": 0.43,
                "balanced_accuracy": 0.61,
                "brier": 0.19,
            }
            for scenario in ("complete_case", "train_median_indicator")
            for view in VIEWS
            for timing in TIMINGS
            for model_type in ("LR", "SVM")
            for model in PRIMARY_MODELS
        ],
        columns=PCR_COLUMNS,
    ).to_csv(root / "metrics" / "pcr_oof_metrics.csv", index=False)
    pcr = pd.read_csv(root / "metrics" / "pcr_oof_metrics.csv")
    pcr.loc[pcr["view"] == "static"].to_csv(root / "metrics" / "static_radiomics.csv", index=False)
    pcr.loc[pcr["view"] == "longitudinal"].to_csv(
        root / "metrics" / "longitudinal_radiomics.csv", index=False
    )

    incremental_rows = []
    manifest_rows = []
    patient_set_hash = "a" * 64
    for view in VIEWS:
        for timing in TIMINGS:
            for scenario in ("complete_case", "train_median_indicator"):
                manifest_rows.append(
                    {
                        "protocol": "primary_stratified_384",
                        "population": "clinical_radiomics_complete_384",
                        "scenario": scenario,
                        "view": view,
                        "timing": timing,
                        "comparison": "all_primary_model_families",
                        "baseline_model": "C/F/N/FULL/C+F/C+N/C+FULL",
                        "augmented_model": "same_exact_patient_set",
                        "n": 384,
                        "pCR_positive": 113,
                        "missingness_exclusions": 0,
                        "exclusion_reason": "none",
                        "patient_set_sha256": patient_set_hash,
                    }
                )
            for baseline, augmented in KEY_COMPARISONS:
                comparison = f"{baseline}_vs_{augmented}"
                incremental_rows.append(
                    {
                        "protocol": "primary_stratified_384",
                        "population": "clinical_radiomics_complete_384",
                        "scenario": "complete_case",
                        "view": view,
                        "timing": timing,
                        "timing_label": _timing_label(timing),
                        "model_type": "LR",
                        "comparison": comparison,
                        "baseline_model": baseline,
                        "augmented_model": augmented,
                        "n": 384,
                        "n_positive": 113,
                        "n_bootstrap": 2000,
                        "delta_auroc": 0.01,
                        "delta_auroc_ci_low": -0.02,
                        "delta_auroc_ci_high": 0.04,
                        "delta_auprc": 0.015,
                        "delta_auprc_ci_low": -0.025,
                        "delta_auprc_ci_high": 0.05,
                        "brier_improvement": 0.002,
                        "brier_improvement_ci_low": -0.005,
                        "brier_improvement_ci_high": 0.009,
                    }
                )
                manifest_rows.append(
                    {
                        "protocol": "primary_stratified_384",
                        "population": "clinical_radiomics_complete_384",
                        "scenario": "complete_case",
                        "view": view,
                        "timing": timing,
                        "comparison": comparison,
                        "baseline_model": baseline,
                        "augmented_model": augmented,
                        "n": 384,
                        "pCR_positive": 113,
                        "missingness_exclusions": 0,
                        "exclusion_reason": "none",
                        "patient_set_sha256": patient_set_hash,
                    }
                )
    pd.DataFrame(incremental_rows, columns=INCREMENTAL_COLUMNS).to_csv(
        root / "metrics" / "incremental_effects.csv", index=False
    )
    manifest = pd.DataFrame(manifest_rows, columns=MATCHED_COLUMNS)
    manifest.to_csv(root / "metrics" / "matched_population_manifest.csv", index=False)
    manifest.to_csv(root / "matched_population_manifest.csv", index=False)

    pd.DataFrame([{"column": "FTV_T0", "family": "FTV"}]).to_csv(
        root / "features" / "radiomics_feature_inventory.csv", index=False
    )
    pd.DataFrame([{"scope": "primary", "missing": 0}]).to_csv(
        root / "metrics" / "missingness.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "fit_scope": "outer_train_only",
                "view": view,
                "timing": timing,
                "feature": f"FTV__absolute__{timing}",
            }
            for view in VIEWS
            for timing in TIMINGS
        ]
    ).to_csv(root / "metrics" / "preprocessing_audit.csv", index=False)
    pd.DataFrame(
        [
            {
                "protocol": "primary_stratified_384",
                "population": "clinical_radiomics_complete_384",
                "view": view,
                "timing": timing,
                "timing_label": _timing_label(timing),
                "feature_set": feature_set,
                "model_type": model_type,
                "target": target,
                "n": 384,
                "n_positive": 200,
                "auroc": 0.123456,
                "auprc": 0.7,
                "balanced_accuracy": 0.6,
                "brier": 0.2,
            }
            for view in VIEWS
            for timing in TIMINGS
            for feature_set in ("N", "FULL")
            for model_type in ("LR", "SVM")
            for target in ("HR", "HER2", "subtype")
        ]
    ).to_csv(root / "metrics" / "profile_oof_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "protocol": "primary_stratified_384",
                "population": "clinical_radiomics_complete_384",
                "view": view,
                "timing": timing,
                "n": 384,
                "r2": 0.2,
                "spearman": 0.45,
            }
            for view in VIEWS
            for timing in TIMINGS
        ]
    ).to_csv(root / "metrics" / "redundancy_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "protocol": "primary_stratified_384",
                "population": "clinical_radiomics_complete_384",
                "scenario": "complete_case",
                "view": view,
                "timing": timing,
                "model_type": "LR",
                "model": model,
                "n": 384,
                "n_positive": 113,
                "auroc": 0.63,
                "auprc": 0.42,
                "balanced_accuracy": 0.59,
                "brier": 0.2,
            }
            for view in VIEWS
            for timing in TIMINGS
            for model in ("N_res", "C+F+N_res")
        ]
    ).to_csv(root / "metrics" / "residualization_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "protocol": "primary_stratified_384",
                "population": "clinical_radiomics_complete_384",
                "scenario": "complete_case",
                "view": view,
                "timing": timing,
                "model_type": "LR",
                "model": model,
                "n": 384,
                "n_positive": 113,
                "auroc": 0.64,
                "auprc": 0.43,
                "balanced_accuracy": 0.6,
                "brier": 0.19,
            }
            for view in VIEWS
            for timing in TIMINGS
            for model in ("C+F", "C+F+D", "C+F+S", "C+F+B")
        ]
    ).to_csv(root / "metrics" / "family_ablation_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "protocol": "primary_stratified_384",
                "population": "clinical_radiomics_complete_384",
                "scenario": "complete_case",
                "view": view,
                "timing": timing,
                "model": model,
                "logistic_auroc": 0.65,
                "svm_auroc": 0.64,
                "delta_svm_minus_lr": -0.01,
            }
            for view in VIEWS
            for timing in TIMINGS
            for model in PRIMARY_MODELS
        ]
    ).to_csv(root / "metrics" / "lr_vs_svm.csv", index=False)
    pd.DataFrame(
        [{"feature": "FTV", "FTV": 1.0, "LD": 0.4}, {"feature": "LD", "FTV": 0.4, "LD": 1.0}]
    ).to_csv(root / "metrics" / "feature_correlation_matrix.csv", index=False)

    aggregation = "unweighted sensitivity-cell metrics; patient rows not pooled"
    pd.DataFrame(
        [
            {
                "population": "ftv_complete_375",
                "n_sensitivity_cells": 4,
                "n_patients_per_cell": 375,
                "aggregation": aggregation,
                "auroc_mean": 0.67,
                "auprc_mean": 0.44,
                "balanced_accuracy_mean": 0.62,
                "brier_mean": 0.18,
            }
        ]
    ).to_csv(root / "metrics" / "mri_reference_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "population": "ftv_complete_375",
                "n_sensitivity_cells": 4,
                "n_patients_per_cell": 375,
                "aggregation": aggregation,
                "auroc_mean": 0.66,
                "auprc_mean": 0.68,
                "balanced_accuracy_mean": 0.61,
            }
        ]
    ).to_csv(root / "metrics" / "mri_reference_profile_metrics.csv", index=False)
    pcr_comparisons = []
    for timing in TIMINGS:
        for traditional_model, mri_model in (("N", "M"), ("C+N", "C+M"), ("C+FULL", "C+F+M")):
            pcr_comparisons.append(
                {
                    "population": "mri_matched_375",
                    "task": "pCR",
                    "view": "longitudinal",
                    "timing": timing,
                    "timing_label": _timing_label(timing),
                    "target": "pCR",
                    "traditional_model": traditional_model,
                    "mri_model": mri_model,
                    "n": 375,
                    "traditional_auroc": 0.64,
                    "mri_auroc": 0.67,
                    "difference_mri_minus_traditional": 0.03,
                    "mri_aggregation": "mean_of_four_seed_x_arm_cells_without_patient_pooling",
                }
            )
    pd.DataFrame(pcr_comparisons, columns=MRI_TRADITIONAL_COMPARISON_COLUMNS).to_csv(
        root / "metrics" / "mri_reference_traditional_pcr_comparison.csv", index=False
    )
    profile_comparisons = []
    for view, timings in (("static", TIMINGS), ("longitudinal", TIMINGS[1:])):
        for timing in timings:
            for target in ("HR", "HER2", "subtype"):
                for traditional_model in ("N", "FULL"):
                    profile_comparisons.append(
                        {
                            "population": "mri_matched_375",
                            "task": "profile_probe",
                            "view": view,
                            "timing": timing,
                            "timing_label": _timing_label(timing),
                            "target": target,
                            "traditional_model": traditional_model,
                            "mri_model": "M",
                            "n": 375,
                            "traditional_auroc": 0.63,
                            "mri_auroc": 0.66,
                            "difference_mri_minus_traditional": 0.03,
                            "mri_aggregation": "mean_of_four_seed_x_arm_cells_without_patient_pooling",
                        }
                    )
    pd.DataFrame(profile_comparisons, columns=MRI_TRADITIONAL_COMPARISON_COLUMNS).to_csv(
        root / "metrics" / "mri_reference_traditional_profile_comparison.csv", index=False
    )
    _write_json(
        root / "metrics" / "mri_reference_provenance.json",
        {
            "matched_population": {"exact_mri_overlap": 375},
            "sensitivity_contract": {
                "n_cells": 4,
                "aggregation": aggregation,
                "duplicate_patients_pooled_across_cells": False,
            },
            "privacy": {"outputs_are_aggregate_only": True},
        },
    )
    _write_json(
        root / "metrics" / "run_summary.json",
        {
            "status": "complete",
            "quick_mode": False,
            "bootstrap_draws": 2000,
            "primary_population": {
                "name": "clinical_radiomics_complete_384",
                "n": 384,
                "pCR_positive": 113,
            },
            "mri_matched_population": {"name": "mri_matched_375", "n": 375},
            "artifacts": {
                "pcr_oof_metrics_sha256": _sha256(root / "metrics" / "pcr_oof_metrics.csv"),
                "profile_oof_metrics_sha256": _sha256(root / "metrics" / "profile_oof_metrics.csv"),
                "incremental_effects_sha256": _sha256(root / "metrics" / "incremental_effects.csv"),
                "matched_population_manifest_sha256": _sha256(
                    root / "matched_population_manifest.csv"
                ),
            },
        },
    )

    for relative in REQUIRED_FIGURES:
        path = root / relative
        path.write_bytes(b"\x89PNG\r\n\x1a\nsynthetic-figure")

    report = """# 最终报告

本报告用中文回答全部预注册问题。实际特征包括 FTV、LD、SPH 和 BPE；NONFTV 表型与 pCR、Clinical 增量均已评估。
我们报告 residual signal、HR、HER2、LR、SVM 和 MRI latent，并讨论 World Model 是否遗漏传统表型以及下一版模型的直接建议。
""" + "\n".join(
        f"### {number}. 预注册问题{number}\n所有结论均基于匹配患者、无未来访视泄漏、训练集拟合预处理和配对不确定性评估。"
        for number in range(1, 13)
    )
    (root / "reports" / "final_report.md").write_text(report, encoding="utf-8")
    return root, repo


def _status(result: dict[str, Any], check_name: str) -> str:
    return next(check["status"] for check in result["checks"] if check["name"] == check_name)


def test_valid_pending_tree_writes_verification_without_real_outputs(tmp_path: Path) -> None:
    root, _ = _build_valid_tree(tmp_path)

    result = verify_experiment(root, allow_pending_delivery=True)

    assert result["status"] == "passed"
    assert _status(result, "delivery") == "passed"
    written = json.loads((root / "metrics" / "verification.json").read_text(encoding="utf-8"))
    assert written["status"] == "passed"
    assert written["failed_checks"] == []


def test_bootstrap_requires_every_comparison_view_timing_and_2000_draws(tmp_path: Path) -> None:
    root, _ = _build_valid_tree(tmp_path)
    path = root / "metrics" / "incremental_effects.csv"
    frame = pd.read_csv(path)
    frame = frame.iloc[1:].copy()
    frame.loc[frame.index[0], "n_bootstrap"] = 1999
    frame.to_csv(path, index=False)

    result = verify_experiment(root, allow_pending_delivery=True)

    assert result["status"] == "failed"
    assert _status(result, "paired_bootstrap") == "failed"


def test_future_visit_and_population_identity_drift_fail_closed(tmp_path: Path) -> None:
    root, _ = _build_valid_tree(tmp_path)
    timing_path = root / "information_timing_contract.csv"
    timing = pd.read_csv(timing_path)
    timing.loc[timing["timing"] == "T0", "allowed_visits"] = "T0|T1"
    timing.to_csv(timing_path, index=False)
    manifest_path = root / "metrics" / "matched_population_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    secondary = manifest.index[manifest["scenario"] == "train_median_indicator"][0]
    manifest.loc[secondary, "patient_set_sha256"] = "b" * 64
    manifest.to_csv(manifest_path, index=False)

    result = verify_experiment(root, allow_pending_delivery=True)

    assert _status(result, "timing_contract") == "failed"
    assert _status(result, "population_identity") == "failed"


def test_mri_reference_rejects_pooled_sensitivity_cells(tmp_path: Path) -> None:
    root, _ = _build_valid_tree(tmp_path)
    for filename in ("mri_reference_metrics.csv", "mri_reference_profile_metrics.csv"):
        path = root / "metrics" / filename
        frame = pd.read_csv(path)
        frame["aggregation"] = "all sensitivity rows pooled"
        frame.to_csv(path, index=False)
    provenance_path = root / "metrics" / "mri_reference_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["sensitivity_contract"]["duplicate_patients_pooled_across_cells"] = True
    _write_json(provenance_path, provenance)

    result = verify_experiment(root, allow_pending_delivery=True)

    assert _status(result, "mri_reference") == "failed"


def test_source_hash_and_schema_are_checked_from_config(tmp_path: Path) -> None:
    root, _ = _build_valid_tree(tmp_path)
    config = json.loads((root / "configs" / "experiment.json").read_text(encoding="utf-8"))
    source = Path(config["verification"]["source_contracts"][0]["path"])
    source.write_text("wrong_schema\n1\n", encoding="utf-8")

    result = verify_experiment(root, allow_pending_delivery=True)

    assert _status(result, "source_hashes_and_schemas") == "failed"


def test_privacy_scan_redacts_direct_identifier_and_raw_data_is_forbidden(tmp_path: Path) -> None:
    root, _ = _build_valid_tree(tmp_path)
    (root / "metrics" / "leaky.csv").write_text(
        "cohort_member,score\n123456,0.8\n", encoding="utf-8"
    )
    (root / "features" / "copied_source.xlsx").write_bytes(b"not a workbook")

    result = verify_experiment(root, allow_pending_delivery=True)

    assert _status(result, "privacy_and_gitignore") == "failed"
    assert _status(result, "no_raw_data") == "failed"
    verification_text = (root / "metrics" / "verification.json").read_text(encoding="utf-8")
    assert "123456" not in verification_text
    assert "REDACTED_SIX_DIGIT_ID" not in verification_text  # path/line only, no copied value


def test_post_delivery_requires_and_validates_branch_sha_push_status(tmp_path: Path) -> None:
    root, repo = _build_valid_tree(tmp_path)
    missing = verify_experiment(root, allow_pending_delivery=False)
    assert _status(missing, "delivery") == "failed"

    _git(repo, "add", "additional_experiments/classical_dce_phenotype_complementarity")
    _git(repo, "commit", "-q", "-m", "synthetic verification fixture")
    sha = _git(repo, "rev-parse", "HEAD")
    _write_json(
        root / "reports" / "delivery_provenance.json",
        {"branch": EXPECTED_BRANCH, "commit_sha": sha, "push_status": "PUSHED", "push_error": ""},
    )

    delivered = verify_experiment(root, allow_pending_delivery=False)

    assert delivered["status"] == "passed"
    assert _status(delivered, "delivery") == "passed"


def test_failed_push_is_valid_only_when_real_error_is_recorded(tmp_path: Path) -> None:
    root, repo = _build_valid_tree(tmp_path)
    _git(repo, "add", "additional_experiments/classical_dce_phenotype_complementarity")
    _git(repo, "commit", "-q", "-m", "synthetic verification fixture")
    sha = _git(repo, "rev-parse", "HEAD")
    provenance_path = root / "reports" / "delivery_provenance.json"
    _write_json(
        provenance_path,
        {"branch": EXPECTED_BRANCH, "commit_sha": sha, "push_status": "GITHUB_PUSH_FAILED"},
    )
    missing_error = verify_experiment(root)
    assert _status(missing_error, "delivery") == "failed"

    _write_json(
        provenance_path,
        {
            "branch": EXPECTED_BRANCH,
            "commit_sha": sha,
            "push_status": "GITHUB_PUSH_FAILED",
            "push_error": "remote unavailable",
        },
    )
    recorded_error = verify_experiment(root)
    assert recorded_error["status"] == "passed"

#!/usr/bin/env python3
"""Independent, fail-closed verification for the compact fusion audit.

This script deliberately does not call the experiment runner's contract or
summary functions.  It re-reads the frozen inputs and emitted artifacts,
checks patient/fold coverage from the pinned manifest, and atomically writes
``metrics/verification.json``.  Private ledgers are used only as verification
inputs; identifiers are never written to the public verification record.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]

POPULATION_COUNTS = {"full_808": 808, "ftv_complete_375": 375}
POPULATION_POSITIVES = {"full_808": 275, "ftv_complete_375": 110}
SEEDS = (2026, 3026)
ARMS = ("LOCAL0", "LOCAL3")
FOLDS = (0, 1, 2, 3, 4)
TIMINGS = ("T0", "T1", "T2", "T3")
RAW_DIMENSIONS = {"T0": 192, "T1": 384, "T2": 576, "T3": 768}
PCA_DIMENSIONS = (8, 16, 32, 64)
PROFILE_REPRESENTATIONS = ("raw", "pca16", "pca32")
PROFILE_TARGETS = ("HR", "HER2", "subtype_4class")
_SUBTYPE_COLUMN_MAP = {
    "HR+/HER2-": "prob_hr_pos_her2_neg",
    "HR-/HER2-": "prob_hr_neg_her2_neg",
    "HR+/HER2+": "prob_hr_pos_her2_pos",
    "HR-/HER2+": "prob_hr_neg_her2_pos",
}
SUBTYPE_CLASSES = tuple(sorted(_SUBTYPE_COLUMN_MAP))
SUBTYPE_PROBABILITY_COLUMNS = tuple(
    _SUBTYPE_COLUMN_MAP[class_name] for class_name in SUBTYPE_CLASSES
)
FIGURE_SEMANTICS = frozenset(
    {
        "auroc_by_dimensionality",
        "delta_auroc_vs_dimension",
        "beyond_ftv_delta_auroc",
        "raw_vs_compact",
        "late_vs_early_fusion",
        "profile_decodability",
        "pca_explained_variance",
    }
)

EXPECTED_CONFIG: Mapping[str, Any] = {
    "schema_version": 1,
    "experiment": "compact_mri_clinical_fusion_audit",
    "branch": "feature/compact-mri-clinical-fusion-audit",
    "parent_commit": "064e0596348f0972decc39774336580f58e8da61",
    "evidence_status": "diagnostic_exploratory_two_seed_compact_fusion_audit",
    "source_goal2": {
        "config": (
            "additional_experiments/mri_clinical_complementarity_audit/"
            "configs/audit.json"
        ),
        "config_sha256": (
            "e95c4971bf6d4c4c39cda9560a6350d5a8eace59235da73941d0334bdd7f5c98"
        ),
        "timing_contract": (
            "additional_experiments/mri_clinical_complementarity_audit/"
            "information_timing_contract.csv"
        ),
        "timing_contract_sha256": (
            "e8d36252dd0d19c0542e1fb061242e886b028ff68f039e908776f72929e293c4"
        ),
        "final_report": (
            "additional_experiments/mri_clinical_complementarity_audit/"
            "reports/final_report.md"
        ),
        "final_report_sha256": (
            "0236fd20d4d1170720b4bdc9e3af913a139be38bfcd711df316b9d21de142f3f"
        ),
        "clinical_inventory": (
            "additional_experiments/mri_clinical_complementarity_audit/"
            "reports/clinical_feature_inventory.md"
        ),
        "clinical_inventory_sha256": (
            "50cbc417b3c60d67067f3d8c673ee9ac0e0ee200cfba9aff40a787e0ca36f262"
        ),
    },
    "populations": ["full_808", "ftv_complete_375"],
    "timings": ["T0", "T1", "T2", "T3"],
    "pca": {
        "dimensions": [8, 16, 32, 64],
        "max_components": 64,
        "svd_solver": "full",
        "whiten": False,
        "scope": "outer_train_timing_prefix",
        "output_dimension_semantics": "k_total_dimensions_per_timing_prefix",
        "selection_metric": "validation_auroc",
        "tie_break": "smaller_dimension_then_smaller_C",
    },
    "random_projection": {
        "dimensions": [16, 32],
        "distribution": "gaussian_N_0_1_over_sqrt_k",
        "seed": 260812,
        "selection": "none_prespecified_sensitivity",
    },
    "late_fusion": {
        "inner_folds": 5,
        "inner_seed": 260813,
        "probability_clip": 1e-6,
        "meta_features": ["clinical_logit", "mri_logit"],
        "train_predictions": "strict_inner_oof",
        "mri_dimension_source": "joint_outer_validation_late_fusion",
    },
    "bootstrap": {
        "replicates": 2000,
        "confidence_level": 0.95,
        "unit": "patient",
        "stratify_by": "outer_fold",
        "random_seed": 260814,
    },
    "profile": {
        "representations": ["raw", "pca16", "pca32"],
        "targets": ["HR", "HER2", "subtype_4class"],
    },
}


def _columns(value: str) -> tuple[str, ...]:
    return tuple(value.split(","))


# These are public/private machine-output contracts, not merely a list of files.
CSV_SPECS: Mapping[str, tuple[int, tuple[str, ...]]] = {
    "metrics/bootstrap_ci.csv": (1008, _columns("comparison_name,population,seed,arm,timing,metric,reference_value,comparison_value,improvement,ci_lower,ci_upper,confidence_level,n_patients,n_folds,n_bootstrap,n_valid_bootstrap,bootstrap_unit,ci_method,orientation,delta,delta_brier,reference_selector,comparison_selector,bootstrap_seed")),
    "metrics/dimension_selection_frequency.csv": (153, _columns("population,timing,model_family,selected_dimension,n_folds_selected,selection_fraction")),
    "metrics/fit_diagnostics.csv": (3760, _columns("population,seed,arm,fold,timing,model_family,representation,dimension,raw_input_dim,feature_dim,train_rows,validation_rows,selected_C,validation_selection_auroc,train_auroc,validation_auroc,test_auroc,train_auprc,validation_auprc,test_auprc,train_brier,validation_brier,test_brier,train_test_auroc_gap")),
    "metrics/goal2_raw_regression_check.csv": (144, _columns("population,seed,arm,timing,model_key,n_goal2,auroc_goal2,auprc_goal2,brier_goal2,n_compact_audit,auroc_compact_audit,auprc_compact_audit,brier_compact_audit,_merge,abs_diff_auroc,abs_diff_auprc,abs_diff_brier,pass")),
    "metrics/late_fusion_diagnostics.csv": (960, _columns("population,seed,arm,fold,timing,model_family,dimension,raw_input_dim,reference_selected_C,mri_selected_C,meta_selected_C,validation_selection_auroc,train_auroc,validation_auroc,test_auroc,train_test_auroc_gap,inner_folds,inner_assignment_sha256,reference_oof_sha256,mri_oof_sha256,strict_oof")),
    "metrics/overfitting_diagnostics.csv": (40, _columns("population,timing,model_family,audit_representation,n_fold_cells,train_auroc_mean,validation_auroc_mean,test_auroc_mean,train_test_auroc_gap_mean,train_test_auroc_gap_min,train_test_auroc_gap_max")),
    "metrics/paired_effects.csv": (336, _columns("comparison_name,population,seed,arm,timing,reference_selector,comparison_selector,n,reference_auroc,comparison_auroc,delta_auroc,reference_auprc,comparison_auprc,delta_auprc,reference_brier,comparison_brier,delta_brier,brier_improvement")),
    "metrics/pca_artifact_manifest.csv": (160, _columns("population,seed,arm,fold,timing,raw_input_dim,train_rows,fit_scope,validation_rows_in_fit,test_rows_in_fit,fitted_transform_sha256,artifact_path,artifact_sha256")),
    "metrics/pca_component_explained_variance.csv": (10240, _columns("population,seed,arm,fold,timing,raw_input_dim,train_rows,fit_scope,validation_rows_in_fit,test_rows_in_fit,fitted_transform_sha256,component,explained_variance,explained_variance_ratio,cumulative_explained_variance_ratio")),
    "metrics/pca_explained_variance.csv": (640, _columns("population,seed,arm,fold,timing,raw_input_dim,train_rows,fit_scope,validation_rows_in_fit,test_rows_in_fit,fitted_transform_sha256,dimension,input_dim,max_components,component_explained_variance_ratio,incremental_explained_variance_ratio,cumulative_explained_variance_ratio")),
    "metrics/pcr_oof_metrics.csv": (1104, _columns("population,seed,arm,timing,model_key,model_family,representation,dimension,dimension_values,n_folds,fold_sizes,n,n_positive,n_negative,auroc,auprc,balanced_accuracy,brier")),
    "metrics/profile_hyperparameters.csv": (720, _columns("seed,arm,fold,timing,target,representation,dimension,raw_input_dim,selected_C,validation_auroc,feature_dim")),
    "metrics/profile_oof_metrics.csv": (144, _columns("population,seed,arm,timing,representation,target,n_folds,n,n_positive,n_negative,n_classes,auroc,auprc,balanced_accuracy,brier,class_counts")),
    "metrics/random_projection_ledger.csv": (8, _columns("timing,raw_input_dim,dimension,seed,distribution,matrix_sha256,artifact_path,artifact_sha256,reads_labels,reads_patient_data")),
    "metrics/random_projection_sensitivity.csv": (160, _columns("population,seed,arm,timing,model_key,model_family,representation,dimension,dimension_values,n_folds,fold_sizes,n,n_positive,n_negative,auroc,auprc,balanced_accuracy,brier")),
    "metrics/residualized_compact_metrics.csv": (32, _columns("population,seed,arm,timing,model_key,model_family,representation,dimension,dimension_values,n_folds,fold_sizes,n,n_positive,n_negative,auroc,auprc,balanced_accuracy,brier")),
    "metrics/selected_dimensions_by_fold.csv": (800, _columns("population,seed,arm,fold,timing,model_family,raw_input_dim,selected_dimension,selected_C,validation_auroc,selection_metric,tie_break,test_used_for_selection")),
    "metrics/table1_pca_dimension_explained_variance.csv": (32, _columns("population,timing,dimension,raw_input_dim,n_outer_pca_fits,cumulative_explained_variance_mean,cumulative_explained_variance_min,cumulative_explained_variance_max")),
    "metrics/table2_mri_only_raw_vs_compact.csv": (64, _columns("population,seed,arm,timing,model_key,model_family,representation,dimension,dimension_values,n_folds,fold_sizes,n,n_positive,n_negative,auroc,auprc,balanced_accuracy,brier")),
    "metrics/table3_c_vs_c_plus_m.csv": (32, _columns("comparison_name,population,seed,arm,timing,reference_selector,comparison_selector,n,reference_auroc,comparison_auroc,delta_auroc,reference_auprc,comparison_auprc,delta_auprc,reference_brier,comparison_brier,delta_brier,brier_improvement")),
    "metrics/table4_cf_vs_cf_plus_m.csv": (16, _columns("comparison_name,population,seed,arm,timing,reference_selector,comparison_selector,n,reference_auroc,comparison_auroc,delta_auroc,reference_auprc,comparison_auprc,delta_auprc,reference_brier,comparison_brier,delta_brier,brier_improvement")),
    "metrics/table5_raw_vs_compact_paired_effects.csv": (80, _columns("comparison_name,population,seed,arm,timing,reference_selector,comparison_selector,n,reference_auroc,comparison_auroc,delta_auroc,reference_auprc,comparison_auprc,delta_auprc,reference_brier,comparison_brier,delta_brier,brier_improvement")),
    "metrics/table6_late_fusion.csv": (96, _columns("comparison_name,population,seed,arm,timing,reference_selector,comparison_selector,n,reference_auroc,comparison_auroc,delta_auroc,reference_auprc,comparison_auprc,delta_auprc,reference_brier,comparison_brier,delta_brier,brier_improvement")),
    "metrics/table7_profile_decodability.csv": (144, _columns("population,seed,arm,timing,representation,target,n_folds,n,n_positive,n_negative,n_classes,auroc,auprc,balanced_accuracy,brier,class_counts")),
    "metrics/table8_bootstrap_ci.csv": (1008, _columns("comparison_name,population,seed,arm,timing,metric,reference_value,comparison_value,improvement,ci_lower,ci_upper,confidence_level,n_patients,n_folds,n_bootstrap,n_valid_bootstrap,bootstrap_unit,ci_method,orientation,delta,delta_brier,reference_selector,comparison_selector,bootstrap_seed")),
    "predictions/bootstrap_draws.private.csv": (672000, _columns("comparison_name,population,seed,arm,timing,bootstrap_index,auroc_improvement,auprc_improvement,brier_improvement,delta_brier,reference_selector,comparison_selector,bootstrap_seed")),
    "predictions/late_fusion_inner_oof.private.csv": (320832, _columns("patient_id,outer_fold,inner_fold,population,seed,arm,timing,late_model_family,dimension,y_true,reference_probability,mri_probability,reference_logit,mri_logit,assignment_sha256,reference_fit_sha256,mri_fit_sha256,inner_pca_sha256,outer_validation_row,outer_test_row")),
    "predictions/pcr_oof.private.csv": (566416, _columns("patient_id,fold,population,seed,arm,timing,model_family,representation,dimension,raw_input_dim,selected_fold_dimension,selected_by_validation,model_key,clinical_contract,y_true,predicted_probability,predicted_label,threshold,selected_C")),
    "predictions/profile_oof.private.csv": (116352, _columns("patient_id,fold,seed,arm,timing,target,representation,dimension,raw_input_dim,y_true,predicted_probability,predicted_label,threshold,prob_hr_pos_her2_neg,prob_hr_neg_her2_neg,prob_hr_pos_her2_pos,prob_hr_neg_her2_pos")),
}

TABLE_FILES = tuple(
    f"metrics/table{index}_{suffix}.csv"
    for index, suffix in enumerate(
        (
            "pca_dimension_explained_variance",
            "mri_only_raw_vs_compact",
            "c_vs_c_plus_m",
            "cf_vs_cf_plus_m",
            "raw_vs_compact_paired_effects",
            "late_fusion",
            "profile_decodability",
            "bootstrap_ci",
        ),
        start=1,
    )
)

REQUIRED_RUN_ARTIFACTS = frozenset(
    {
        "bootstrap_draws", "bootstrap_summary", "dimension_frequency",
        "dimension_selection", "fit_diagnostics", "late_diagnostics", "late_oof",
        "overfitting", "paired_effects", "pca_artifacts", "pca_components",
        "pca_variance", "pcr_metrics", "pcr_predictions", "profile_hyperparameters",
        "profile_metrics", "profile_predictions", "random_projection",
        "random_projection_ledger", "raw_regression", "residual",
        "table1", "table2", "table3", "table4", "table5", "table6", "table7",
        "table8",
    }
)

FULL_MODEL_KEYS = frozenset(
    ["C", "M|RAW", "C+M|RAW"]
    + [f"{family}|PCA{dimension}" for family in ("M", "C+M", "LateFusion(C,M)") for dimension in PCA_DIMENSIONS]
    + [f"{family}|PCA_SELECTED" for family in ("M", "C+M", "LateFusion(C,M)")]
    + [f"{family}|R{dimension}" for family in ("M", "C+M") for dimension in (16, 32)]
)
FTV_MODEL_KEYS = frozenset(
    set(FULL_MODEL_KEYS)
    | {"F", "C+F", "C+F+M|RAW"}
    | {f"{family}|PCA{dimension}" for family in ("M_residual", "C+F+M_residual", "C+F+M", "LateFusion(C+F,M)") for dimension in PCA_DIMENSIONS}
    | {f"{family}|PCA_SELECTED" for family in ("M_residual", "C+F+M_residual", "C+F+M", "LateFusion(C+F,M)")}
    | {f"C+F+M|R{dimension}" for dimension in (16, 32)}
)

COMPARISON_POPULATIONS: Mapping[str, tuple[str, ...]] = {
    "delta1_C_plus_Mk_vs_C": ("full_808", "ftv_complete_375"),
    "delta2_CF_plus_Mk_vs_CF": ("ftv_complete_375",),
    "delta3_late_C_Mk_vs_C": ("full_808", "ftv_complete_375"),
    "delta4_late_CF_Mk_vs_CF": ("ftv_complete_375",),
    "raw_vs_compact_M": ("full_808", "ftv_complete_375"),
    "raw_vs_compact_C_plus_M": ("full_808", "ftv_complete_375"),
    "raw_vs_compact_CF_plus_M": ("ftv_complete_375",),
    "residual_beyond_ftv": ("ftv_complete_375",),
    "late_vs_concat_C_M": ("full_808", "ftv_complete_375"),
    "late_vs_concat_CF_M": ("ftv_complete_375",),
    "RP16_C_plus_M_vs_C": ("full_808", "ftv_complete_375"),
    "RP32_C_plus_M_vs_C": ("full_808", "ftv_complete_375"),
    "RP16_CF_plus_M_vs_CF": ("ftv_complete_375",),
    "RP32_CF_plus_M_vs_CF": ("ftv_complete_375",),
}


class AuditVerificationError(RuntimeError):
    """A verification invariant failed (messages must remain identifier-free)."""


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AuditVerificationError(message)


def _sha256(path: Path) -> str:
    _require(path.is_file() and not path.is_symlink(), f"missing regular file: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(values: pd.Series) -> bool:
    return bool(values.astype(str).str.fullmatch(r"[0-9a-f]{64}").all())


def _json_no_duplicate_keys(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise AuditVerificationError(f"duplicate JSON key in {path.name}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AuditVerificationError(f"invalid JSON: {path.name}") from error
    _require(isinstance(value, dict), f"JSON root must be an object: {path.name}")
    return value


def _safe_relative(root: Path, value: Any, label: str) -> Path:
    _require(isinstance(value, str) and value and value == value.strip(), f"invalid {label} path")
    relative = Path(value)
    _require(not relative.is_absolute(), f"{label} path must be relative")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise AuditVerificationError(f"{label} path escapes its root") from error
    return resolved


def _configured_path(repo_root: Path, value: Any, label: str) -> Path:
    _require(isinstance(value, str) and value and value == value.strip(), f"invalid {label} path")
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (repo_root / path).resolve()
    _require(resolved.is_file() and not resolved.is_symlink(), f"missing configured {label}")
    return resolved


def _exact_tuples(frame: pd.DataFrame, columns: Sequence[str]) -> set[tuple[Any, ...]]:
    return set(frame.loc[:, list(columns)].itertuples(index=False, name=None))


def _require_grid(
    frame: pd.DataFrame,
    columns: Sequence[str],
    expected: Iterable[tuple[Any, ...]],
    label: str,
) -> None:
    wanted = set(expected)
    observed = _exact_tuples(frame, columns)
    _require(
        observed == wanted,
        f"{label} grid mismatch (expected {len(wanted)} cells, observed {len(observed)})",
    )


def _boolean(series: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapped = series.astype(str).str.strip().str.lower().map({"true": True, "false": False})
    _require(mapped.notna().all(), f"{label} is not strictly boolean")
    return mapped.astype(bool)


def _finite_probability(series: pd.Series, label: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    _require(np.isfinite(values).all(), f"{label} contains non-finite values")
    _require(bool(((values >= 0.0) & (values <= 1.0)).all()), f"{label} leaves [0,1]")
    return values


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def classify_status(checks: Sequence[Mapping[str, Any]]) -> str:
    failed = [item for item in checks if item.get("status") != "PASS"]
    if not failed:
        return "PASS"
    if all(item.get("category") == "final_deliverable" for item in failed):
        return "PRE_FINAL_FAIL"
    return "FAIL"


def expected_comparison_cells() -> set[tuple[Any, ...]]:
    return {
        (name, population, seed, arm, timing)
        for name, populations in COMPARISON_POPULATIONS.items()
        for population in populations
        for seed, arm, timing in itertools.product(SEEDS, ARMS, TIMINGS)
    }


def validate_bootstrap_frames(
    summary: pd.DataFrame,
    draws: pd.DataFrame,
    *,
    expected_cells: set[tuple[Any, ...]],
    population_counts: Mapping[str, int],
    replicates: int = 2000,
) -> dict[str, int]:
    """Validate exact per-cell draw coverage and summary orientation.

    Kept as a standalone function so the most failure-prone bootstrap contract
    can be unit-tested without constructing the full experiment directory.
    """

    cell_columns = ["comparison_name", "population", "seed", "arm", "timing"]
    required_summary = {
        *cell_columns,
        "metric", "reference_value", "comparison_value", "improvement",
        "ci_lower", "ci_upper", "confidence_level", "n_patients", "n_folds",
        "n_bootstrap", "n_valid_bootstrap", "bootstrap_unit", "ci_method",
        "orientation", "delta", "delta_brier", "reference_selector",
        "comparison_selector", "bootstrap_seed",
    }
    required_draws = {
        *cell_columns,
        "bootstrap_index", "auroc_improvement", "auprc_improvement",
        "brier_improvement", "delta_brier", "reference_selector",
        "comparison_selector", "bootstrap_seed",
    }
    _require(required_summary.issubset(summary.columns), "bootstrap summary schema is incomplete")
    _require(required_draws.issubset(draws.columns), "bootstrap draw schema is incomplete")
    _require_grid(summary, cell_columns, expected_cells, "bootstrap summary")
    _require_grid(draws, cell_columns, expected_cells, "bootstrap draws")

    summary_sizes = summary.groupby(cell_columns, sort=False).size()
    _require(summary_sizes.eq(3).all(), "bootstrap summary must have three metric rows per cell")
    metric_sets = summary.groupby(cell_columns, sort=False)["metric"].agg(lambda x: frozenset(x))
    _require(
        metric_sets.eq(frozenset({"auroc", "auprc", "brier"})).all(),
        "bootstrap summary metric set drifted",
    )
    _require(not summary.duplicated([*cell_columns, "metric"]).any(), "duplicate bootstrap summary metric row")
    _require(summary["n_bootstrap"].eq(replicates).all(), "bootstrap summary replicate count drifted")
    _require(summary["n_valid_bootstrap"].eq(replicates).all(), "bootstrap summary has invalid draws")
    _require(summary["n_folds"].eq(5).all(), "bootstrap summary must retain five outer folds")
    _require(summary["confidence_level"].eq(0.95).all(), "bootstrap confidence level drifted")
    _require(summary["bootstrap_unit"].eq("patient_within_outer_fold").all(), "bootstrap unit/stratification drifted")
    _require(summary["ci_method"].eq("percentile").all(), "bootstrap CI method drifted")
    expected_n = summary["population"].map(population_counts)
    _require(expected_n.notna().all(), "bootstrap summary contains an unknown population")
    _require(summary["n_patients"].eq(expected_n).all(), "bootstrap summary pooled or dropped patients")

    draw_sizes = draws.groupby(cell_columns, sort=False).size()
    _require(draw_sizes.eq(replicates).all(), "bootstrap draw count is not exact per comparison cell")
    _require(not draws.duplicated([*cell_columns, "bootstrap_index"]).any(), "duplicate bootstrap draw index")
    indices = draws.groupby(cell_columns, sort=False)["bootstrap_index"].agg(["min", "max", "nunique"])
    _require(
        bool(
            indices["min"].eq(0).all()
            and indices["max"].eq(replicates - 1).all()
            and indices["nunique"].eq(replicates).all()
        ),
        "bootstrap indices must be exactly 0..replicates-1 per cell",
    )
    numeric_draws = draws[
        ["auroc_improvement", "auprc_improvement", "brier_improvement", "delta_brier"]
    ].apply(pd.to_numeric, errors="coerce")
    _require(np.isfinite(numeric_draws.to_numpy(dtype=float)).all(), "bootstrap draws contain non-finite effects")
    _require(
        np.allclose(
            numeric_draws["delta_brier"],
            -numeric_draws["brier_improvement"],
            rtol=0.0,
            atol=1e-12,
        ),
        "bootstrap delta_brier is not comparison-reference",
    )

    draw_cell_metadata = draws.groupby(cell_columns, sort=False).agg(
        n_reference_selectors=("reference_selector", "nunique"),
        n_comparison_selectors=("comparison_selector", "nunique"),
        n_bootstrap_seeds=("bootstrap_seed", "nunique"),
    )
    _require(draw_cell_metadata.eq(1).all(axis=None), "bootstrap metadata changes within a cell")
    unique_seeds = draws[[*cell_columns, "bootstrap_seed"]].drop_duplicates()
    _require(unique_seeds["bootstrap_seed"].nunique() == len(expected_cells), "bootstrap cell seeds are not unique")

    draw_columns = {
        "auroc": "auroc_improvement",
        "auprc": "auprc_improvement",
        "brier": "brier_improvement",
    }
    grouped_draws = {key: group for key, group in draws.groupby(cell_columns, sort=False)}
    for row in summary.itertuples(index=False):
        cell = tuple(getattr(row, column) for column in cell_columns)
        values = pd.to_numeric(grouped_draws[cell][draw_columns[row.metric]], errors="coerce").to_numpy(float)
        interval = np.quantile(values, [0.025, 0.975])
        _require(
            np.allclose([row.ci_lower, row.ci_upper], interval, rtol=0.0, atol=1e-12),
            "bootstrap percentile CI does not match stored draws",
        )
        reference = float(row.reference_value)
        comparison = float(row.comparison_value)
        if row.metric == "brier":
            _require(
                np.isclose(row.improvement, reference - comparison, rtol=0.0, atol=1e-12)
                and np.isclose(row.delta, comparison - reference, rtol=0.0, atol=1e-12)
                and np.isclose(row.delta_brier, comparison - reference, rtol=0.0, atol=1e-12),
                "Brier improvement/delta orientation drifted",
            )
        else:
            _require(
                np.isclose(row.improvement, comparison - reference, rtol=0.0, atol=1e-12)
                and np.isclose(row.delta, comparison - reference, rtol=0.0, atol=1e-12)
                and pd.isna(row.delta_brier),
                "AUROC/AUPRC improvement orientation drifted",
            )

    return {"comparison_cells": len(expected_cells), "draw_rows": len(draws)}


class AuditVerifier:
    def __init__(self, experiment_root: Path, repo_root: Path) -> None:
        self.experiment_root = experiment_root.resolve()
        self.repo_root = repo_root.resolve()
        self.checks: list[dict[str, Any]] = []
        self.config: dict[str, Any] | None = None
        self.goal2_config: dict[str, Any] | None = None
        self.fold_manifest: pd.DataFrame | None = None
        self.clinical: pd.DataFrame | None = None
        self.ftv_ids: set[str] | None = None
        self.test_fold: dict[str, int] = {}
        self.pcr_label: dict[str, int] = {}
        self.population_ids: dict[str, set[str]] = {}
        self.train_counts: dict[tuple[str, int], int] = {}

    def path(self, relative: str) -> Path:
        return self.experiment_root / relative

    def _record(self, name: str, function: Any, *, category: str = "core") -> None:
        try:
            details = function()
            if details is None:
                details = {}
            _require(isinstance(details, Mapping), f"internal check result for {name} is invalid")
            self.checks.append(
                {"name": name, "category": category, "status": "PASS", "details": dict(details)}
            )
        except Exception as error:  # keep evaluating independent checks
            self.checks.append(
                {
                    "name": name,
                    "category": category,
                    "status": "FAIL",
                    # Check messages are deliberately aggregate-only and identifier-free.
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    def run(self) -> dict[str, Any]:
        self._record("frozen_config_and_source_hashes", self.check_sources)
        self._record("git_branch_and_parent", self.check_git)
        self._record("locked_cohorts", self.check_cohorts)
        self._record("required_output_schemas_and_counts", self.check_output_schemas)
        self._record("run_summary_and_artifact_hashes", self.check_run_summary)
        self._record("pca_fit_scope_and_variance", self.check_pca)
        self._record("validation_only_dimension_selection", self.check_dimension_selection)
        self._record("random_projection_ledger", self.check_random_projection)
        self._record("pcr_exact_oof_coverage_and_metrics", self.check_pcr_oof)
        self._record("profile_exact_oof_coverage_and_metrics", self.check_profile_oof)
        self._record("strict_inner_oof_late_fusion", self.check_late_fusion)
        self._record("paired_bootstrap_draws", self.check_bootstrap)
        self._record("goal2_raw_regression", self.check_goal2_regression)
        self._record("public_identifier_privacy", self.check_public_privacy)
        self._record("eight_required_tables", self.check_tables)
        self._record("seven_required_figures", self.check_figures, category="final_deliverable")
        self._record("final_report_links", self.check_final_report, category="final_deliverable")
        status = classify_status(self.checks)
        passed = sum(item["status"] == "PASS" for item in self.checks)
        return {
            "schema_version": 1,
            "experiment": "compact_mri_clinical_fusion_audit",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "summary": {
                "checks_total": len(self.checks),
                "checks_passed": passed,
                "checks_failed": len(self.checks) - passed,
                "pre_final_only": status == "PRE_FINAL_FAIL",
            },
            "checks": self.checks,
        }

    def _need_sources(self) -> tuple[dict[str, Any], dict[str, Any]]:
        _require(self.config is not None and self.goal2_config is not None, "source verification prerequisite failed")
        return self.config, self.goal2_config

    def _need_cohorts(self) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
        _require(
            self.fold_manifest is not None and self.clinical is not None and self.ftv_ids is not None,
            "cohort verification prerequisite failed",
        )
        return self.fold_manifest, self.clinical, self.ftv_ids

    def check_sources(self) -> Mapping[str, Any]:
        config_path = self.path("configs/audit.json")
        _require(config_path.is_file(), "compact audit config is missing")
        config = _json_no_duplicate_keys(config_path)
        _require(config == EXPECTED_CONFIG, "compact audit config differs from the frozen contract")
        source = config["source_goal2"]
        observed_hashes: dict[str, str] = {}
        for key in ("config", "timing_contract", "final_report", "clinical_inventory"):
            path = _safe_relative(self.repo_root, source[key], f"Goal2 {key}")
            observed = _sha256(path)
            _require(observed == source[f"{key}_sha256"], f"pinned Goal2 {key} hash drifted")
            observed_hashes[key] = observed
        goal2_config_path = _safe_relative(self.repo_root, source["config"], "Goal2 config")
        goal2_config = _json_no_duplicate_keys(goal2_config_path)
        paths = goal2_config.get("paths")
        _require(isinstance(paths, Mapping), "Goal2 config paths object is missing")
        for key in ("clinical_labels", "fold_manifest", "ftv_table", "local_preregistration_lock"):
            input_path = _configured_path(self.repo_root, paths.get(key), key)
            expected_hash = paths.get(f"{key}_sha256")
            _require(isinstance(expected_hash, str) and re.fullmatch(r"[0-9a-f]{64}", expected_hash), f"Goal2 {key} hash is invalid")
            _require(_sha256(input_path) == expected_hash, f"hash-pinned Goal2 {key} input drifted")
        timing_copy = self.path("information_timing_contract.csv")
        _require(_sha256(timing_copy) == source["timing_contract_sha256"], "compact timing-contract copy drifted")
        self.config = config
        self.goal2_config = goal2_config
        return {"pinned_goal2_sources": len(observed_hashes), "hash_pinned_goal2_inputs": 4}

    def check_git(self) -> Mapping[str, Any]:
        config = self.config if self.config is not None else EXPECTED_CONFIG
        branch_result = subprocess.run(
            ["git", "branch", "--show-current"], cwd=self.repo_root,
            check=True, capture_output=True, text=True,
        )
        branch = branch_result.stdout.strip()
        _require(branch == config["branch"], "current branch differs from frozen audit branch")
        parent = str(config["parent_commit"])
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", parent, "HEAD"],
            cwd=self.repo_root, check=False, capture_output=True, text=True,
        )
        _require(ancestor.returncode == 0, "frozen parent commit is not an ancestor of HEAD")
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=self.repo_root,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        return {"branch": branch, "head": head, "parent_is_ancestor": True}

    def check_cohorts(self) -> Mapping[str, Any]:
        _, goal2 = self._need_sources()
        paths = goal2["paths"]
        manifest = pd.read_csv(_configured_path(self.repo_root, paths["fold_manifest"], "fold manifest"), dtype={"patient_id": str})
        _require(tuple(manifest.columns) == ("patient_id", "fold", "split", "label_pcr"), "fold manifest schema drifted")
        _require(len(manifest) == 808 * 5, "fold manifest row count drifted")
        _require(not manifest.duplicated(["patient_id", "fold"]).any(), "fold manifest duplicates patient/fold rows")
        _require(set(manifest["fold"]) == set(FOLDS), "fold manifest fold set drifted")
        _require(set(manifest["split"]) == {"train", "val", "test"}, "fold split names drifted")
        grouped = manifest.groupby("patient_id", sort=False)
        _require(manifest["patient_id"].nunique() == 808, "full cohort size drifted")
        _require(grouped.size().eq(5).all() and grouped["fold"].nunique().eq(5).all(), "patients do not occur exactly once per outer fold")
        _require(grouped["label_pcr"].nunique().eq(1).all(), "pCR label changes across folds")
        _require(manifest["split"].eq("test").groupby(manifest["patient_id"]).sum().eq(1).all(), "patients do not have exactly one held-out test fold")

        test = manifest.loc[manifest["split"].eq("test"), ["patient_id", "fold", "label_pcr"]].copy()
        _require(int(test["label_pcr"].sum()) == 275, "full-cohort pCR positive count drifted")
        full_ids = set(test["patient_id"].astype(str))
        self.test_fold = dict(zip(test["patient_id"].astype(str), test["fold"].astype(int), strict=True))
        self.pcr_label = dict(zip(test["patient_id"].astype(str), test["label_pcr"].astype(int), strict=True))

        clinical = pd.read_csv(_configured_path(self.repo_root, paths["clinical_labels"], "clinical labels"), dtype={"patient_id": str})
        required_clinical = {"patient_id", "label_pcr", "label_hr", "label_her2", "hr_her2_subtype"}
        _require(required_clinical.issubset(clinical.columns), "clinical label schema is incomplete")
        _require(len(clinical) == 808 and not clinical["patient_id"].duplicated().any(), "clinical cohort is not exactly 808 unique patients")
        _require(set(clinical["patient_id"]) == full_ids, "clinical/fold cohort membership differs")
        clinical_pcr = dict(zip(clinical["patient_id"], clinical["label_pcr"].astype(int), strict=True))
        _require(clinical_pcr == self.pcr_label, "clinical and fold-manifest pCR labels differ")

        ftv = pd.read_csv(_configured_path(self.repo_root, paths["ftv_table"], "FTV table"), dtype={"patient_id": str})
        required_ftv = {"patient_id", "transition", "start_visit", "end_visit", "ftv_valid"}
        _require(required_ftv.issubset(ftv.columns), "FTV schema is incomplete")
        _require(len(ftv) == 375 * 3 and ftv["patient_id"].nunique() == 375, "FTV cohort shape drifted")
        _require(_boolean(ftv["ftv_valid"], "FTV validity").all(), "FTV cohort includes invalid rows")
        _require(not ftv.duplicated(["patient_id", "transition"]).any(), "FTV patient/transition rows repeat")
        transition_set = frozenset({"T0→T1", "T1→T2", "T2→T3"})
        _require(ftv.groupby("patient_id")["transition"].agg(lambda x: frozenset(x)).eq(transition_set).all(), "FTV transition coverage drifted")
        ftv_ids = set(ftv["patient_id"].astype(str))
        _require(ftv_ids.issubset(full_ids), "FTV population leaves the locked cohort")
        _require(sum(self.pcr_label[value] for value in ftv_ids) == 110, "FTV-cohort pCR positive count drifted")

        self.fold_manifest = manifest
        self.clinical = clinical
        self.ftv_ids = ftv_ids
        self.population_ids = {"full_808": full_ids, "ftv_complete_375": ftv_ids}
        for population, identifiers in self.population_ids.items():
            subset = manifest.loc[manifest["patient_id"].isin(identifiers)]
            for fold in FOLDS:
                self.train_counts[(population, fold)] = int(
                    subset["fold"].eq(fold).mul(subset["split"].eq("train")).sum()
                )
        return {"full_patients": 808, "ftv_patients": 375, "outer_folds": 5}

    def check_output_schemas(self) -> Mapping[str, Any]:
        for relative, (expected_rows, expected_columns) in CSV_SPECS.items():
            path = self.path(relative)
            _require(path.is_file() and path.stat().st_size > 0, f"required CSV is missing or empty: {relative}")
            header = tuple(pd.read_csv(path, nrows=0).columns)
            _require(header == expected_columns, f"CSV schema drifted: {relative}")
            with path.open("rb") as stream:
                rows = sum(1 for _ in stream) - 1
            _require(rows == expected_rows, f"CSV row count drifted: {relative}")
        return {"csv_artifacts": len(CSV_SPECS), "expected_rows_total": sum(spec[0] for spec in CSV_SPECS.values())}

    def check_run_summary(self) -> Mapping[str, Any]:
        summary = _json_no_duplicate_keys(self.path("metrics/run_summary.json"))
        expected_scalars = {
            "schema_version": 1,
            "experiment": "compact_mri_clinical_fusion_audit",
            "branch": "feature/compact-mri-clinical-fusion-audit",
            "parent_commit": "064e0596348f0972decc39774336580f58e8da61",
            "evidence_status": "diagnostic_exploratory_two_seed_compact_fusion_audit",
            "formal_bootstrap": True,
            "bootstrap_replicates": 2000,
            "n_local_cells": 20,
            "n_pcr_prediction_rows": 566416,
            "n_profile_prediction_rows": 116352,
            "n_late_inner_oof_rows": 320832,
            "n_bootstrap_comparison_cells": 336,
            "raw_goal2_regression_pass": True,
            "pca_semantics": "k_total_dimensions_per_timing_prefix",
            "raw_prefix_dimensions": RAW_DIMENSIONS,
            "summarize_only": False,
        }
        for key, expected in expected_scalars.items():
            _require(summary.get(key) == expected, f"run summary field drifted: {key}")
        artifacts = summary.get("artifacts")
        _require(isinstance(artifacts, Mapping), "run summary artifact manifest is missing")
        _require(REQUIRED_RUN_ARTIFACTS.issubset(artifacts), "run summary omits required artifacts")
        observed_paths: set[str] = set()
        for name, record in artifacts.items():
            _require(isinstance(record, Mapping), f"run artifact record is invalid: {name}")
            path = _safe_relative(self.experiment_root, record.get("path"), f"run artifact {name}")
            _require(path.is_file() and not path.is_symlink(), f"run artifact is missing: {name}")
            relative = path.relative_to(self.experiment_root).as_posix()
            _require(relative not in observed_paths, "run summary aliases one path under multiple artifact names")
            observed_paths.add(relative)
            expected_hash = record.get("sha256")
            _require(isinstance(expected_hash, str) and re.fullmatch(r"[0-9a-f]{64}", expected_hash), f"run artifact hash is invalid: {name}")
            _require(_sha256(path) == expected_hash, f"run artifact hash drifted: {name}")
            _require(record.get("size_bytes") == path.stat().st_size, f"run artifact size drifted: {name}")
        return {"manifested_artifacts": len(artifacts), "all_hashes_match": True}

    def _pca_grid(self) -> set[tuple[Any, ...]]:
        return {
            (population, seed, arm, fold, timing)
            for population, seed, arm, fold, timing in itertools.product(
                POPULATION_COUNTS, SEEDS, ARMS, FOLDS, TIMINGS
            )
        }

    def check_pca(self) -> Mapping[str, Any]:
        self._need_cohorts()
        keys = ["population", "seed", "arm", "fold", "timing"]
        manifest = pd.read_csv(self.path("metrics/pca_artifact_manifest.csv"))
        _require_grid(manifest, keys, self._pca_grid(), "PCA artifact")
        _require(not manifest.duplicated(keys).any(), "PCA artifact key repeats")
        _require(manifest["fit_scope"].eq("outer_train_timing_prefix").all(), "PCA fit scope drifted")
        _require(manifest["validation_rows_in_fit"].eq(0).all(), "PCA fitted validation rows")
        _require(manifest["test_rows_in_fit"].eq(0).all(), "PCA fitted test rows")
        _require(_is_sha256(manifest["fitted_transform_sha256"]), "PCA transform hashes are malformed")
        _require(_is_sha256(manifest["artifact_sha256"]), "PCA artifact hashes are malformed")
        _require(manifest["fitted_transform_sha256"].nunique() == 160, "PCA fitted transforms are not unique per outer cell")
        _require(manifest["artifact_sha256"].nunique() == 160, "PCA artifacts are not unique per outer cell")
        expected_raw = manifest["timing"].map(RAW_DIMENSIONS)
        expected_train = pd.Series(
            [self.train_counts[(row.population, int(row.fold))] for row in manifest.itertuples()],
            index=manifest.index,
        )
        _require(manifest["raw_input_dim"].eq(expected_raw).all(), "PCA raw prefix dimension drifted")
        _require(manifest["train_rows"].eq(expected_train).all(), "PCA train-row ledger differs from outer-train split")
        for row in manifest.itertuples(index=False):
            artifact = _safe_relative(self.experiment_root, row.artifact_path, "PCA artifact")
            _require(artifact.is_file(), "PCA private artifact is missing")
            _require(_sha256(artifact) == row.artifact_sha256, "PCA private artifact hash drifted")

        variance = pd.read_csv(self.path("metrics/pca_explained_variance.csv"))
        variance_grid = {(*cell, dimension) for cell in self._pca_grid() for dimension in PCA_DIMENSIONS}
        variance_keys = [*keys, "dimension"]
        _require_grid(variance, variance_keys, variance_grid, "PCA variance")
        _require(not variance.duplicated(variance_keys).any(), "PCA variance key repeats")
        _require(set(variance["dimension"]) == set(PCA_DIMENSIONS), "PCA dimension set drifted")
        _require(variance["max_components"].eq(64).all(), "PCA max-component contract drifted")
        _require(variance["input_dim"].eq(variance["raw_input_dim"]).all(), "PCA input dimension ledger disagrees")
        _require(variance["validation_rows_in_fit"].eq(0).all() and variance["test_rows_in_fit"].eq(0).all(), "PCA variance ledger records val/test fit rows")
        for column in (
            "component_explained_variance_ratio",
            "incremental_explained_variance_ratio",
            "cumulative_explained_variance_ratio",
        ):
            values = pd.to_numeric(variance[column], errors="coerce").to_numpy(float)
            _require(np.isfinite(values).all() and ((values >= 0.0) & (values <= 1.0 + 1e-12)).all(), f"PCA {column} is invalid")
        monotonic = variance.sort_values(variance_keys).groupby(keys)["cumulative_explained_variance_ratio"].apply(lambda x: x.is_monotonic_increasing)
        _require(monotonic.all(), "PCA cumulative variance is not monotone")

        components = pd.read_csv(self.path("metrics/pca_component_explained_variance.csv"))
        _require_grid(components, keys, self._pca_grid(), "PCA component ledger")
        _require(components.groupby(keys).size().eq(64).all(), "PCA component ledger must contain 64 rows per fit")
        component_sets = components.groupby(keys)["component"].agg(lambda x: frozenset(x))
        _require(component_sets.eq(frozenset(range(1, 65))).all(), "PCA component indices drifted")
        _require(components["validation_rows_in_fit"].eq(0).all() and components["test_rows_in_fit"].eq(0).all(), "PCA component ledger records val/test fit rows")
        ratios = pd.to_numeric(components["explained_variance_ratio"], errors="coerce").to_numpy(float)
        _require(np.isfinite(ratios).all() and (ratios >= 0.0).all(), "PCA component ratios are invalid")
        ordered = components.sort_values([*keys, "component"]).copy()
        recomputed = ordered.groupby(keys)["explained_variance_ratio"].cumsum()
        _require(
            np.allclose(recomputed, ordered["cumulative_explained_variance_ratio"], rtol=0.0, atol=1e-12),
            "PCA cumulative explained variance does not equal component sum",
        )
        checkpoints = components.loc[components["component"].isin(PCA_DIMENSIONS), [*keys, "component", "cumulative_explained_variance_ratio"]].rename(columns={"component": "dimension", "cumulative_explained_variance_ratio": "component_cumulative"})
        joined = variance.merge(checkpoints, on=variance_keys, how="outer", validate="one_to_one", indicator=True)
        _require(joined["_merge"].eq("both").all(), "PCA variance checkpoints are incomplete")
        _require(np.allclose(joined["cumulative_explained_variance_ratio"], joined["component_cumulative"], rtol=0.0, atol=1e-12), "PCA variance checkpoint values disagree")
        return {"outer_train_pca_fits": 160, "allowed_dimensions": list(PCA_DIMENSIONS), "component_rows": 10240}

    def check_dimension_selection(self) -> Mapping[str, Any]:
        _, goal2 = self._need_sources()
        frame = pd.read_csv(self.path("metrics/selected_dimensions_by_fold.csv"))
        keys = ["population", "seed", "arm", "fold", "timing", "model_family"]
        families = {
            "full_808": ("M", "C+M", "LateFusion(C,M)"),
            "ftv_complete_375": (
                "M", "C+M", "M_residual", "C+F+M_residual", "C+F+M",
                "LateFusion(C,M)", "LateFusion(C+F,M)",
            ),
        }
        expected = {
            (population, seed, arm, fold, timing, family)
            for population, values in families.items()
            for seed, arm, fold, timing, family in itertools.product(SEEDS, ARMS, FOLDS, TIMINGS, values)
        }
        _require_grid(frame, keys, expected, "dimension selection")
        _require(not frame.duplicated(keys).any(), "dimension selection key repeats")
        _require(set(frame["selected_dimension"]) <= set(PCA_DIMENSIONS), "selected PCA dimension is not allowed")
        _require(frame["raw_input_dim"].eq(frame["timing"].map(RAW_DIMENSIONS)).all(), "selection raw dimension drifted")
        _require(frame["selection_metric"].eq("validation_auroc").all(), "dimension selection metric drifted")
        _require(frame["tie_break"].eq("smaller_dimension_then_smaller_C").all(), "dimension selection tie-break drifted")
        _require(not _boolean(frame["test_used_for_selection"], "test selection flag").any(), "test was used for PCA dimension selection")
        c_grid = {float(value) for value in goal2["logistic"]["c_grid"]}
        _require(set(frame["selected_C"].astype(float)) <= c_grid, "selected logistic C leaves frozen grid")
        _require(np.isfinite(frame["validation_auroc"].to_numpy(float)).all(), "selection AUROC is non-finite")
        return {"fold_selections": len(frame), "test_used_for_selection": False}

    def check_random_projection(self) -> Mapping[str, Any]:
        frame = pd.read_csv(self.path("metrics/random_projection_ledger.csv"))
        expected = {(timing, dimension) for timing, dimension in itertools.product(TIMINGS, (16, 32))}
        _require_grid(frame, ["timing", "dimension"], expected, "random projection")
        _require(not frame.duplicated(["timing", "dimension"]).any(), "random projection key repeats")
        _require(frame["seed"].eq(260812).all(), "random projection seed drifted")
        _require(frame["seed"].nunique() == 1, "random projection uses more than one seed")
        _require(frame["distribution"].eq("gaussian_N_0_1_over_sqrt_k").all(), "random projection distribution drifted")
        _require(frame["raw_input_dim"].eq(frame["timing"].map(RAW_DIMENSIONS)).all(), "random projection input dimension drifted")
        _require(not _boolean(frame["reads_labels"], "RP reads_labels").any(), "random projection read labels")
        _require(not _boolean(frame["reads_patient_data"], "RP reads_patient_data").any(), "random projection read patient data")
        _require(_is_sha256(frame["matrix_sha256"]) and frame["matrix_sha256"].nunique() == 8, "random projection matrix hashes are not eight unique SHA-256 values")
        _require(_is_sha256(frame["artifact_sha256"]) and frame["artifact_sha256"].nunique() == 8, "random projection artifact hashes are not unique")
        for row in frame.itertuples(index=False):
            artifact = _safe_relative(self.experiment_root, row.artifact_path, "random projection artifact")
            _require(artifact.is_file(), "random projection artifact is missing")
            _require(_sha256(artifact) == row.artifact_sha256, "random projection artifact hash drifted")
            with np.load(artifact, allow_pickle=False) as payload:
                _require(set(payload.files) == {"matrix", "input_dim", "output_dim", "seed", "matrix_sha256"}, "random projection NPZ schema drifted")
                matrix = np.asarray(payload["matrix"], dtype=np.float64)
                _require(matrix.shape == (row.raw_input_dim, row.dimension), "random projection matrix shape drifted")
                _require(np.isfinite(matrix).all(), "random projection matrix is non-finite")
                _require(int(payload["seed"]) == 260812 and str(payload["matrix_sha256"]) == row.matrix_sha256, "random projection NPZ metadata drifted")
        return {"projection_matrices": 8, "unique_matrix_hashes": 8, "seed": 260812}

    def _pcr_grid(self) -> set[tuple[Any, ...]]:
        models = {"full_808": FULL_MODEL_KEYS, "ftv_complete_375": FTV_MODEL_KEYS}
        return {
            (population, seed, arm, timing, model_key)
            for population, model_keys in models.items()
            for seed, arm, timing, model_key in itertools.product(SEEDS, ARMS, TIMINGS, model_keys)
        }

    def check_pcr_oof(self) -> Mapping[str, Any]:
        self._need_cohorts()
        from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

        group_columns = ["population", "seed", "arm", "timing", "model_key"]
        usecols = [
            "patient_id", "fold", "population", "seed", "arm", "timing",
            "model_key", "y_true", "predicted_probability", "predicted_label",
            "threshold", "raw_input_dim",
        ]
        predictions = pd.read_csv(
            self.path("predictions/pcr_oof.private.csv"),
            usecols=usecols,
            dtype={"patient_id": str},
        )
        _require(len(predictions) == 566416, "pCR prediction row count drifted")
        _require_grid(predictions, group_columns, self._pcr_grid(), "pCR OOF")
        _require(not predictions.duplicated([*group_columns, "patient_id"]).any(), "pCR OOF repeats a patient within a model cell")
        group_sizes = predictions.groupby(group_columns, sort=False).size()
        expected_sizes = group_sizes.index.get_level_values("population").map(POPULATION_COUNTS)
        _require(np.array_equal(group_sizes.to_numpy(), expected_sizes.to_numpy()), "pCR OOF cell size drifted or populations were pooled")

        full_ids = self.population_ids["full_808"]
        ftv_ids = self.population_ids["ftv_complete_375"]
        _require(predictions["patient_id"].isin(full_ids).all(), "pCR OOF contains a patient outside the locked cohort")
        ftv_rows = predictions["population"].eq("ftv_complete_375")
        _require(predictions.loc[ftv_rows, "patient_id"].isin(ftv_ids).all(), "FTV pCR OOF contains a full-only patient")
        full_rows = predictions["population"].eq("full_808")
        _require(predictions.loc[full_rows, "patient_id"].isin(full_ids).all(), "full pCR OOF membership drifted")
        expected_fold = predictions["patient_id"].map(self.test_fold)
        expected_label = predictions["patient_id"].map(self.pcr_label)
        _require(expected_fold.notna().all() and predictions["fold"].eq(expected_fold).all(), "pCR predictions are not from each patient's locked test fold")
        _require(expected_label.notna().all() and predictions["y_true"].eq(expected_label).all(), "pCR OOF labels differ from the locked manifest")
        _require(set(predictions["fold"]) == set(FOLDS), "pCR OOF fold set drifted")
        _require(predictions["raw_input_dim"].eq(predictions["timing"].map(RAW_DIMENSIONS)).all(), "pCR raw prefix dimension drifted")
        probability = _finite_probability(predictions["predicted_probability"], "pCR probability")
        threshold = _finite_probability(predictions["threshold"], "pCR threshold")
        labels = pd.to_numeric(predictions["y_true"], errors="coerce").to_numpy(float)
        predicted = pd.to_numeric(predictions["predicted_label"], errors="coerce").to_numpy(float)
        _require(set(np.unique(labels)) == {0.0, 1.0}, "pCR OOF labels are not binary")
        _require(set(np.unique(predicted)) <= {0.0, 1.0}, "pCR predicted labels are not binary")
        _require(np.array_equal((probability >= threshold).astype(int), predicted.astype(int)), "pCR predicted labels disagree with probability/threshold")

        computed_rows: list[dict[str, Any]] = []
        for key, group in predictions.groupby(group_columns, sort=False):
            y = group["y_true"].to_numpy(dtype=int)
            p = group["predicted_probability"].to_numpy(dtype=float)
            yp = group["predicted_label"].to_numpy(dtype=int)
            fold_sizes = {str(int(fold)): int(count) for fold, count in group["fold"].value_counts().sort_index().items()}
            computed_rows.append(
                {
                    **dict(zip(group_columns, key, strict=True)),
                    "n_check": len(group),
                    "n_positive_check": int(y.sum()),
                    "n_negative_check": int((y == 0).sum()),
                    "n_folds_check": int(group["fold"].nunique()),
                    "fold_sizes_check": json.dumps(fold_sizes, sort_keys=True, separators=(",", ":")),
                    "auroc_check": float(roc_auc_score(y, p)),
                    "auprc_check": float(average_precision_score(y, p)),
                    "balanced_accuracy_check": float(balanced_accuracy_score(y, yp)),
                    "brier_check": float(np.mean(np.square(p - y))),
                }
            )
        computed = pd.DataFrame(computed_rows)
        metrics = pd.read_csv(self.path("metrics/pcr_oof_metrics.csv"))
        _require_grid(metrics, group_columns, self._pcr_grid(), "pCR metrics")
        _require(not metrics.duplicated(group_columns).any(), "pCR metric key repeats")
        merged = metrics.merge(computed, on=group_columns, how="outer", validate="one_to_one", indicator=True)
        _require(merged["_merge"].eq("both").all(), "pCR metrics do not cover exact prediction cells")
        for column in ("n", "n_positive", "n_negative", "n_folds", "fold_sizes"):
            _require(merged[column].eq(merged[f"{column}_check"]).all(), f"pCR aggregate {column} disagrees with predictions")
        for metric in ("auroc", "auprc", "balanced_accuracy", "brier"):
            _require(
                np.allclose(merged[metric], merged[f"{metric}_check"], rtol=0.0, atol=1e-12),
                f"pCR aggregate {metric} disagrees with predictions",
            )
        expected_positive = merged["population"].map(POPULATION_POSITIVES)
        _require(merged["n_positive"].eq(expected_positive).all(), "pCR positive counts drifted by population")
        return {"model_cells": len(computed), "oof_rows": len(predictions), "fold_coverage": "exact_once"}

    def check_profile_oof(self) -> Mapping[str, Any]:
        _, clinical, _ = self._need_cohorts()
        from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
        from sklearn.preprocessing import label_binarize

        group_columns = ["seed", "arm", "timing", "representation", "target"]
        expected_grid = set(itertools.product(SEEDS, ARMS, TIMINGS, PROFILE_REPRESENTATIONS, PROFILE_TARGETS))
        predictions = pd.read_csv(
            self.path("predictions/profile_oof.private.csv"), dtype={"patient_id": str}
        )
        _require_grid(predictions, group_columns, expected_grid, "profile OOF")
        _require(not predictions.duplicated([*group_columns, "patient_id"]).any(), "profile OOF repeats a patient within a probe cell")
        _require(predictions.groupby(group_columns).size().eq(808).all(), "profile OOF cell is not the exact full cohort")
        _require(predictions["patient_id"].isin(self.population_ids["full_808"]).all(), "profile OOF leaves the full cohort")
        _require(predictions["fold"].eq(predictions["patient_id"].map(self.test_fold)).all(), "profile predictions are not from locked test folds")
        _require(predictions["raw_input_dim"].eq(predictions["timing"].map(RAW_DIMENSIONS)).all(), "profile raw prefix dimension drifted")
        expected_dimension = predictions["representation"].map({"pca16": 16, "pca32": 32}).fillna(predictions["timing"].map(RAW_DIMENSIONS))
        _require(predictions["dimension"].eq(expected_dimension).all(), "profile representation dimension drifted")

        clinical_index = clinical.set_index("patient_id", verify_integrity=True)
        target_columns = {"HR": "label_hr", "HER2": "label_her2", "subtype_4class": "hr_her2_subtype"}
        for target, column in target_columns.items():
            rows = predictions["target"].eq(target)
            expected = predictions.loc[rows, "patient_id"].map(clinical_index[column])
            observed = predictions.loc[rows, "y_true"]
            if target != "subtype_4class":
                observed = pd.to_numeric(observed, errors="coerce")
                expected = pd.to_numeric(expected, errors="coerce")
            else:
                observed = observed.astype(str)
                expected = expected.astype(str)
            _require(observed.reset_index(drop=True).equals(expected.reset_index(drop=True)), f"profile {target} labels differ from clinical contract")

        binary_rows = predictions["target"].isin({"HR", "HER2"})
        binary_probability = _finite_probability(predictions.loc[binary_rows, "predicted_probability"], "binary profile probability")
        binary_threshold = _finite_probability(predictions.loc[binary_rows, "threshold"], "binary profile threshold")
        binary_predicted = pd.to_numeric(predictions.loc[binary_rows, "predicted_label"], errors="coerce").to_numpy(float)
        _require(np.array_equal((binary_probability >= binary_threshold).astype(int), binary_predicted.astype(int)), "binary profile labels disagree with probability/threshold")
        _require(predictions.loc[binary_rows, list(SUBTYPE_PROBABILITY_COLUMNS)].isna().all(axis=None), "binary probes populate subtype probability columns")

        subtype_rows = predictions["target"].eq("subtype_4class")
        subtype_probability = predictions.loc[subtype_rows, list(SUBTYPE_PROBABILITY_COLUMNS)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        _require(np.isfinite(subtype_probability).all(), "subtype probabilities are non-finite")
        _require(((subtype_probability >= 0.0) & (subtype_probability <= 1.0)).all(), "subtype probabilities leave [0,1]")
        _require(np.allclose(subtype_probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-6), "subtype probabilities do not sum to one")
        subtype_predicted = np.asarray(SUBTYPE_CLASSES)[np.argmax(subtype_probability, axis=1)]
        _require(np.array_equal(subtype_predicted, predictions.loc[subtype_rows, "predicted_label"].astype(str)), "subtype predicted labels disagree with fixed-column argmax")

        computed_rows: list[dict[str, Any]] = []
        for key, group in predictions.groupby(group_columns, sort=False):
            target = str(key[-1])
            if target in {"HR", "HER2"}:
                y = pd.to_numeric(group["y_true"], errors="raise").to_numpy(dtype=int)
                probability = group["predicted_probability"].to_numpy(dtype=float)
                predicted = pd.to_numeric(group["predicted_label"], errors="raise").to_numpy(dtype=int)
                values = {
                    "n_positive_check": int(y.sum()),
                    "n_negative_check": int((y == 0).sum()),
                    "n_classes_check": 2,
                    "auroc_check": float(roc_auc_score(y, probability)),
                    "auprc_check": float(average_precision_score(y, probability)),
                    "balanced_accuracy_check": float(balanced_accuracy_score(y, predicted)),
                    "brier_check": float(np.mean(np.square(probability - y))),
                }
            else:
                y = group["y_true"].astype(str).to_numpy()
                probability = group.loc[:, list(SUBTYPE_PROBABILITY_COLUMNS)].to_numpy(dtype=float)
                predicted = np.asarray(SUBTYPE_CLASSES)[np.argmax(probability, axis=1)]
                indicator = label_binarize(y, classes=SUBTYPE_CLASSES)
                values = {
                    "n_positive_check": np.nan,
                    "n_negative_check": np.nan,
                    "n_classes_check": 4,
                    "auroc_check": float(roc_auc_score(y, probability, labels=SUBTYPE_CLASSES, multi_class="ovr", average="macro")),
                    "auprc_check": float(average_precision_score(indicator, probability, average="macro")),
                    "balanced_accuracy_check": float(balanced_accuracy_score(y, predicted)),
                    "brier_check": np.nan,
                }
            computed_rows.append(
                {
                    **dict(zip(group_columns, key, strict=True)),
                    "n_check": len(group),
                    "n_folds_check": int(group["fold"].nunique()),
                    **values,
                }
            )
        computed = pd.DataFrame(computed_rows)
        metrics = pd.read_csv(self.path("metrics/profile_oof_metrics.csv"))
        public_keys = ["population", *group_columns]
        expected_public = {("full_808", *cell) for cell in expected_grid}
        _require_grid(metrics, public_keys, expected_public, "profile metrics")
        merged = metrics.merge(computed, on=group_columns, how="outer", validate="one_to_one", indicator=True)
        _require(merged["_merge"].eq("both").all() and merged["population"].eq("full_808").all(), "profile metrics pool populations or omit cells")
        for column in ("n", "n_folds", "n_classes"):
            _require(merged[column].eq(merged[f"{column}_check"]).all(), f"profile aggregate {column} disagrees")
        for column in ("n_positive", "n_negative", "auroc", "auprc", "balanced_accuracy", "brier"):
            _require(np.allclose(merged[column], merged[f"{column}_check"], rtol=0.0, atol=1e-12, equal_nan=True), f"profile aggregate {column} disagrees")
        return {"probe_cells": len(computed), "oof_rows": len(predictions), "subtype_probability_columns": 4}

    def _late_grid(self) -> set[tuple[Any, ...]]:
        families = {
            "full_808": ("LateFusion(C,M)",),
            "ftv_complete_375": ("LateFusion(C,M)", "LateFusion(C+F,M)"),
        }
        return {
            (population, seed, arm, fold, timing, family, dimension)
            for population, family_values in families.items()
            for seed, arm, fold, timing, family, dimension in itertools.product(
                SEEDS, ARMS, FOLDS, TIMINGS, family_values, PCA_DIMENSIONS
            )
        }

    def check_late_fusion(self) -> Mapping[str, Any]:
        self._need_cohorts()
        group_columns = [
            "population", "seed", "arm", "outer_fold", "timing",
            "late_model_family", "dimension",
        ]
        predictions = pd.read_csv(
            self.path("predictions/late_fusion_inner_oof.private.csv"),
            dtype={"patient_id": str},
        )
        expected_grid = self._late_grid()
        _require_grid(predictions, group_columns, expected_grid, "late-fusion inner OOF")
        _require(not predictions.duplicated([*group_columns, "patient_id"]).any(), "late-fusion inner OOF repeats an outer-train patient/candidate")
        sizes = predictions.groupby(group_columns, sort=False).size()
        expected_sizes = np.asarray(
            [self.train_counts[(index[0], int(index[3]))] for index in sizes.index], dtype=int
        )
        _require(np.array_equal(sizes.to_numpy(dtype=int), expected_sizes), "late-fusion inner OOF candidate coverage differs from outer-train")

        for population in POPULATION_COUNTS:
            for outer_fold in FOLDS:
                rows = predictions["population"].eq(population) & predictions["outer_fold"].eq(outer_fold)
                observed = predictions.loc[rows, "patient_id"]
                expected_train = set(
                    self.fold_manifest.loc[
                        self.fold_manifest["patient_id"].isin(self.population_ids[population])
                        & self.fold_manifest["fold"].eq(outer_fold)
                        & self.fold_manifest["split"].eq("train"),
                        "patient_id",
                    ].astype(str)
                )
                _require(observed.isin(expected_train).all(), "late-fusion ledger includes outer validation/test patients")
        _require(predictions["y_true"].eq(predictions["patient_id"].map(self.pcr_label)).all(), "late-fusion inner labels differ from locked pCR labels")
        _require(set(predictions["inner_fold"]) == set(FOLDS), "late-fusion inner-fold set drifted")
        inner_sets = predictions.groupby(group_columns)["inner_fold"].agg(lambda x: frozenset(x))
        _require(inner_sets.eq(frozenset(FOLDS)).all(), "a late-fusion candidate does not use all five inner folds")
        _require(not _boolean(predictions["outer_validation_row"], "late outer-validation flag").any(), "late-fusion meta training includes outer validation rows")
        _require(not _boolean(predictions["outer_test_row"], "late outer-test flag").any(), "late-fusion meta training includes outer test rows")
        for column in ("assignment_sha256", "reference_fit_sha256", "mri_fit_sha256", "inner_pca_sha256"):
            _require(_is_sha256(predictions[column]), f"late-fusion {column} is malformed")
        _require(predictions["assignment_sha256"].nunique() == 10, "inner-fold assignments are not shared as ten population/outer-fold ledgers")
        assignment_counts = predictions.groupby(group_columns)["assignment_sha256"].nunique()
        _require(assignment_counts.eq(1).all(), "inner-fold assignment hash changes within a late candidate")
        for probability_column, logit_column in (
            ("reference_probability", "reference_logit"),
            ("mri_probability", "mri_logit"),
        ):
            probability = _finite_probability(predictions[probability_column], f"late {probability_column}")
            logit = pd.to_numeric(predictions[logit_column], errors="coerce").to_numpy(float)
            _require(np.isfinite(logit).all(), f"late {logit_column} is non-finite")
            clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
            expected_logit = np.log(clipped / (1.0 - clipped))
            _require(np.allclose(logit, expected_logit, rtol=0.0, atol=1e-10), f"late {logit_column} disagrees with probability")

        diagnostics = pd.read_csv(self.path("metrics/late_fusion_diagnostics.csv"))
        diagnostic_columns = [
            "population", "seed", "arm", "fold", "timing", "model_family", "dimension"
        ]
        _require_grid(diagnostics, diagnostic_columns, expected_grid, "late-fusion diagnostics")
        _require(not diagnostics.duplicated(diagnostic_columns).any(), "late-fusion diagnostic key repeats")
        _require(diagnostics["inner_folds"].eq(5).all(), "late-fusion diagnostics do not record five inner folds")
        _require(_boolean(diagnostics["strict_oof"], "strict_oof").all(), "late-fusion diagnostics are not strict OOF")
        _require(diagnostics["raw_input_dim"].eq(diagnostics["timing"].map(RAW_DIMENSIONS)).all(), "late-fusion raw prefix dimension drifted")
        for column in ("inner_assignment_sha256", "reference_oof_sha256", "mri_oof_sha256"):
            _require(_is_sha256(diagnostics[column]), f"late diagnostic {column} is malformed")
        assignment = predictions.groupby(group_columns, as_index=False)["assignment_sha256"].first()
        assignment = assignment.rename(
            columns={"outer_fold": "fold", "late_model_family": "model_family"}
        )
        joined = diagnostics.merge(
            assignment,
            on=diagnostic_columns,
            how="outer",
            validate="one_to_one",
            indicator=True,
        )
        _require(joined["_merge"].eq("both").all(), "late diagnostics do not cover exact inner-OOF candidates")
        _require(joined["inner_assignment_sha256"].eq(joined["assignment_sha256"]).all(), "late diagnostic assignment hash disagrees with private ledger")
        return {
            "candidate_cells": len(expected_grid),
            "inner_oof_rows": len(predictions),
            "inner_folds": 5,
            "outer_validation_or_test_rows": 0,
        }

    def check_bootstrap(self) -> Mapping[str, Any]:
        expected_cells = expected_comparison_cells()
        summary = pd.read_csv(self.path("metrics/bootstrap_ci.csv"))
        draws = pd.read_csv(self.path("predictions/bootstrap_draws.private.csv"))
        details = validate_bootstrap_frames(
            summary,
            draws,
            expected_cells=expected_cells,
            population_counts=POPULATION_COUNTS,
            replicates=2000,
        )
        cell_columns = ["comparison_name", "population", "seed", "arm", "timing"]
        paired = pd.read_csv(self.path("metrics/paired_effects.csv"))
        _require_grid(paired, cell_columns, expected_cells, "paired effects")
        _require(not paired.duplicated(cell_columns).any(), "paired point-effect cell repeats")
        expected_n = paired["population"].map(POPULATION_COUNTS)
        _require(paired["n"].eq(expected_n).all(), "paired effects pool or drop patients")
        _require(
            np.allclose(paired["delta_brier"], paired["comparison_brier"] - paired["reference_brier"], rtol=0.0, atol=1e-12)
            and np.allclose(paired["brier_improvement"], -paired["delta_brier"], rtol=0.0, atol=1e-12),
            "paired point-effect Brier orientation drifted",
        )
        metric_mapping = {
            "auroc": ("reference_auroc", "comparison_auroc", "delta_auroc"),
            "auprc": ("reference_auprc", "comparison_auprc", "delta_auprc"),
            "brier": ("reference_brier", "comparison_brier", "brier_improvement"),
        }
        for metric, (reference_column, comparison_column, effect_column) in metric_mapping.items():
            metric_summary = summary.loc[summary["metric"].eq(metric), [*cell_columns, "reference_value", "comparison_value", "improvement"]]
            joined = paired.merge(metric_summary, on=cell_columns, how="outer", validate="one_to_one", indicator=True)
            _require(joined["_merge"].eq("both").all(), f"bootstrap {metric} summary does not align with paired effects")
            _require(
                np.allclose(joined[reference_column], joined["reference_value"], rtol=0.0, atol=1e-12)
                and np.allclose(joined[comparison_column], joined["comparison_value"], rtol=0.0, atol=1e-12)
                and np.allclose(joined[effect_column], joined["improvement"], rtol=0.0, atol=1e-12),
                f"bootstrap {metric} point values disagree with paired effects",
            )
        details = dict(details)
        details.update({"replicates_per_cell": 2000, "bootstrap_unit": "patient_within_outer_fold"})
        return details

    def check_goal2_regression(self) -> Mapping[str, Any]:
        frame = pd.read_csv(self.path("metrics/goal2_raw_regression_check.csv"))
        _require(len(frame) == 144, "Goal2 raw regression row count drifted")
        _require(frame["_merge"].eq("both").all(), "Goal2 raw regression has unmatched cells")
        _require(_boolean(frame["pass"], "Goal2 raw regression pass").all(), "Goal2 raw regression contains a failed row")
        _require(frame["n_goal2"].eq(frame["n_compact_audit"]).all(), "Goal2 raw regression cohort sizes disagree")
        for metric in ("auroc", "auprc", "brier"):
            _require(frame[f"abs_diff_{metric}"].le(1e-12).all(), f"Goal2 raw {metric} regression exceeds tolerance")

        oracle = pd.read_csv(
            self.repo_root
            / "additional_experiments/mri_clinical_complementarity_audit/metrics/pcr_oof_metrics.csv"
        )
        mapping = {
            "C": "C", "F": "F", "C+F": "C+F", "M": "M|RAW",
            "C+M": "C+M|RAW", "C+F+M": "C+F+M|RAW",
        }
        oracle = oracle.loc[oracle["model"].isin(mapping)].copy()
        oracle["model_key"] = oracle["model"].map(mapping)
        current = pd.read_csv(self.path("metrics/pcr_oof_metrics.csv"))
        current = current.loc[current["model_key"].isin(mapping.values())]
        keys = ["population", "seed", "arm", "timing", "model_key"]
        direct = oracle[[*keys, "n", "auroc", "auprc", "brier"]].merge(
            current[[*keys, "n", "auroc", "auprc", "brier"]],
            on=keys,
            how="outer",
            validate="one_to_one",
            suffixes=("_goal2", "_compact"),
            indicator=True,
        )
        _require(len(direct) == 144 and direct["_merge"].eq("both").all(), "direct Goal2 raw metric grid differs")
        _require(direct["n_goal2"].eq(direct["n_compact"]).all(), "direct Goal2 raw patient counts differ")
        for metric in ("auroc", "auprc", "brier"):
            _require(np.allclose(direct[f"{metric}_goal2"], direct[f"{metric}_compact"], rtol=0.0, atol=1e-12), f"direct Goal2 raw {metric} regression failed")
        return {"raw_metric_cells": 144, "tolerance": 1e-12, "result": "PASS"}

    def check_public_privacy(self) -> Mapping[str, Any]:
        self._need_cohorts()
        public_files: list[Path] = []
        for directory_name in ("metrics", "reports", "figures"):
            directory = self.path(directory_name)
            if directory.is_dir():
                public_files.extend(
                    path for path in directory.rglob("*")
                    if path.is_file() and path.name not in {".gitkeep", "verification.json"}
                )
        _require(public_files, "no public artifacts were found for privacy scan")
        identifiers = sorted(self.population_ids["full_808"], key=len, reverse=True)
        identifier_pattern = re.compile(b"|".join(re.escape(value.encode("utf-8")) for value in identifiers))
        forbidden_headers = {"patient_id", "patient_ids", "trial_id"}
        for path in public_files:
            data = path.read_bytes()
            _require(identifier_pattern.search(data) is None, f"locked patient identifier found in public artifact: {path.relative_to(self.experiment_root)}")
            if path.suffix.lower() == ".csv":
                columns = {str(column).strip().lower() for column in pd.read_csv(path, nrows=0).columns}
                _require(not (columns & forbidden_headers), f"identifier-bearing column found in public CSV: {path.relative_to(self.experiment_root)}")
        return {"public_files_scanned": len(public_files), "locked_identifiers_found": 0}

    def check_tables(self) -> Mapping[str, Any]:
        observed = {
            path.relative_to(self.experiment_root).as_posix()
            for path in self.path("metrics").glob("table*.csv")
            if path.is_file()
        }
        _require(observed == set(TABLE_FILES), "required aggregate table set is not exactly eight")
        for relative in TABLE_FILES:
            path = self.path(relative)
            _require(path.stat().st_size > 0, f"required table is empty: {relative}")
        return {"required_tables": 8, "table_files": list(TABLE_FILES)}

    def check_figures(self) -> Mapping[str, Any]:
        manifest_path = self.path("metrics/figure_manifest.csv")
        _require(manifest_path.is_file(), "figure manifest is pending")
        manifest = pd.read_csv(manifest_path)
        expected_columns = _columns(
            "figure_order,figure_file,title,description,source_metrics,source_metrics_sha256,source_rows_used,point_unit,population_handling,t3_marked_late,private_predictions_read,width_px,height_px,bytes,figure_sha256,generator"
        )
        _require(tuple(manifest.columns) == expected_columns, "figure manifest schema drifted")
        _require(len(manifest) == 7, "figure manifest must contain exactly seven rows")
        _require(set(manifest["figure_order"]) == set(range(1, 8)), "figure manifest order must be exactly 1..7")
        semantics = {Path(value).stem for value in manifest["figure_file"].astype(str)}
        _require(semantics == FIGURE_SEMANTICS, "figure semantic-name set drifted")
        _require(not manifest["figure_file"].duplicated().any(), "figure manifest repeats a PNG path")
        _require(_is_sha256(manifest["figure_sha256"]), "figure hashes are malformed")
        _require(_is_sha256(manifest["source_metrics_sha256"]), "figure source-metric hashes are malformed")
        _require(_boolean(manifest["t3_marked_late"], "figure T3-late flag").all(), "a figure does not mark T3 as late")
        _require(not _boolean(manifest["private_predictions_read"], "figure private-read flag").any(), "a figure read private predictions")
        _require(manifest["population_handling"].eq("separate_panels_or_ftv_only_no_pooling").all(), "figure population handling permits pooling")

        png_paths: set[Path] = set()
        for row in manifest.itertuples(index=False):
            figure = _safe_relative(self.experiment_root, row.figure_file, "figure")
            try:
                figure.relative_to(self.path("figures").resolve())
            except ValueError as error:
                raise AuditVerificationError("manifested figure is outside figures/") from error
            _require(figure.suffix.lower() == ".png" and figure.is_file(), "manifested PNG is missing")
            _require(figure.stat().st_size > 0 and figure.stat().st_size == int(row.bytes), "figure byte ledger drifted")
            _require(_sha256(figure) == row.figure_sha256, "figure SHA-256 drifted")
            header = figure.read_bytes()[:24]
            _require(len(header) == 24 and header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR", "figure is not a valid PNG header")
            width = int.from_bytes(header[16:20], "big")
            height = int.from_bytes(header[20:24], "big")
            _require(width == int(row.width_px) and height == int(row.height_px) and width > 0 and height > 0, "figure dimensions drifted")
            png_paths.add(figure)

            source = _safe_relative(self.experiment_root, row.source_metrics, "figure source metric")
            _require(source.is_file() and _sha256(source) == row.source_metrics_sha256, "figure source metric hash drifted")
            generator = _safe_relative(self.experiment_root, row.generator, "figure generator")
            _require(generator.is_file(), "figure generator is missing")

        observed_pngs = {
            path.resolve() for path in self.path("figures").rglob("*.png")
            if path.is_file() and path.stat().st_size > 0
        }
        _require(observed_pngs == png_paths and len(png_paths) == 7, "nonempty PNG set differs from the seven manifested figures")
        return {"figures": 7, "semantic_names": sorted(semantics), "private_predictions_read": False}

    @staticmethod
    def _markdown_link_targets(report: Path) -> set[Path]:
        text = report.read_text(encoding="utf-8")
        targets: set[Path] = set()
        for raw_target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
            target = raw_target.strip()
            if target.startswith("<") and target.endswith(">"):
                target = target[1:-1]
            # Local report links in this audit contain no titles or URL escapes.
            if not target or "://" in target or target.startswith("#"):
                continue
            targets.add((report.parent / target).resolve())
        return targets

    def check_final_report(self) -> Mapping[str, Any]:
        report = self.path("reports/final_report.md")
        _require(report.is_file() and report.stat().st_size > 0, "final report is pending")
        links = self._markdown_link_targets(report)
        table_paths = {self.path(relative).resolve() for relative in TABLE_FILES}
        _require(table_paths.issubset(links), "final report does not link all eight required tables")
        manifest_path = self.path("metrics/figure_manifest.csv")
        _require(manifest_path.is_file(), "final report figure-link check awaits figure manifest")
        manifest = pd.read_csv(manifest_path)
        _require("figure_file" in manifest, "figure manifest lacks figure_file")
        figure_paths = {
            _safe_relative(self.experiment_root, value, "report figure").resolve()
            for value in manifest["figure_file"]
        }
        _require(figure_paths.issubset(links), "final report does not link all seven required figures")
        text = report.read_text(encoding="utf-8")
        _require("feature/compact-mri-clinical-fusion-audit" in text, "final report omits branch record")
        _require("064e0596348f0972decc39774336580f58e8da61" in text, "final report omits frozen parent commit")
        return {"linked_tables": 8, "linked_figures": 7, "report_bytes": report.stat().st_size}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to <experiment-root>/metrics/verification.json",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    experiment_root = args.experiment_root.resolve()
    repo_root = args.repo_root.resolve()
    output = args.output.resolve() if args.output is not None else experiment_root / "metrics/verification.json"
    verifier = AuditVerifier(experiment_root, repo_root)
    result = verifier.run()
    _atomic_json(output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "checks_passed": result["summary"]["checks_passed"],
                "checks_failed": result["summary"]["checks_failed"],
                "verification": str(output),
            },
            sort_keys=True,
        )
    )
    return {"PASS": 0, "PRE_FINAL_FAIL": 2, "FAIL": 1}[str(result["status"])]


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fail-closed validation for the non-FTV phenotype decodability audit.

The validator is deliberately read-only.  It checks the complete aggregate
matrix, validation/test leakage declarations, privacy boundaries, private-file
permissions, and the Git change scope.  It never opens a feature array and it
never prints a patient or trial identifier.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from audit_core import (
    AUDIT_ROOT,
    INTERVALS,
    MAIN_REPRESENTATIONS,
    ORACLE_TO_MATCHED,
    PREDICTION_COLUMNS,
    REPO_ROOT,
    VISITS,
    authenticate,
    load_config,
    load_fold_splits,
    load_targets,
    ordered_sha256,
    resolve_path,
)
from freeze_preregistration import require_preregistration_lock


EXPECTED_COUNTS = {
    "oof_metrics": 3276,
    "fold_metrics": 16380,
    "hyperparameter_selections": 16380,
    "coverage": 16380,
    "static_target_matrix": 448,
    "residual_target_matrix": 560,
    "dynamic_target_matrix": 1512,
    "representation_location_comparison": 1080,
    "oracle_localization_comparison": 756,
    "target_transform_fits": 560,
    "residualizer_fits": 175,
    "frozen_cell_manifest": 20,
    "target_contract": 16,
    "representation_contract": 7,
}

CORE_PUBLIC_FILES = {
    "metrics/oof_metrics.csv",
    "metrics/fold_metrics.csv",
    "metrics/hyperparameter_selections.csv",
    "metrics/coverage.csv",
    "metrics/target_transform_fits.csv",
    "metrics/residualizer_fits.csv",
    "metrics/static_target_matrix.csv",
    "metrics/residual_target_matrix.csv",
    "metrics/dynamic_target_matrix.csv",
    "metrics/representation_location_comparison.csv",
    "metrics/oracle_localization_comparison.csv",
    "manifests/frozen_cell_manifest.csv",
    "manifests/target_contract.csv",
    "manifests/representation_contract.csv",
    "manifests/input_provenance.json",
    "metrics/run_summary.json",
}

FINAL_PUBLIC_FILES = {
    "metrics/bpe_fov_observability_audit.csv",
    "metrics/bottleneck_diagnostics.csv",
    "metrics/dynamic_macro.csv",
    "metrics/gate_candidate_matrix.csv",
    "metrics/grounding_candidate_scorecard.csv",
    "metrics/oracle_validity_summary.csv",
    "metrics/primary_gates.json",
    "metrics/probe_integrity_summary.csv",
    "metrics/final_target_recommendation.csv",
    "manifests/public_analysis_artifacts.csv",
    "reports/final_report.md",
}

OOF_IDENTITY = [
    "seed",
    "arm",
    "representation",
    "matched_reference_for",
    "task_type",
    "target_definition",
    "target_kind",
    "target",
    "timing",
    "interval",
    "input_variant",
    "metric_space",
    "feature_dim",
]

METRIC_COLUMNS = {
    "n",
    "spearman",
    "pearson",
    "natural_r2",
    "transformed_r2",
    "rmse",
    "mae",
    "prediction_target_variance_ratio",
    "calibration_slope",
    "residual_spearman",
    "residual_transformed_r2",
    "reconstructed_natural_r2",
    "reconstructed_natural_rmse",
    "reconstructed_natural_mae",
    "natural_metric_interpretation",
    "rank_aggregation",
    "transformed_r2_aggregation",
}

FORBIDDEN_PUBLIC_COLUMN_PATTERNS = (
    re.compile(r"^(?:patient|subject|trial)_?ids?$", re.IGNORECASE),
    re.compile(r"^clinical[-_]?trial[-_]?subject[-_]?id$", re.IGNORECASE),
    re.compile(r"^label_?pcr$", re.IGNORECASE),
    re.compile(r"^y_(?:true|pred)(?:_|$)", re.IGNORECASE),
)

SENSITIVE_TRACKED_SUFFIXES = {
    ".gz",
    ".npy",
    ".npz",
    ".parquet",
    ".pt",
    ".pth",
    ".xls",
    ".xlsx",
}


class Checks:
    """Collect all validation failures without leaking sensitive values."""

    def __init__(self) -> None:
        self.checked = 0
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def require(self, condition: bool, message: str) -> None:
        self.checked += 1
        if not bool(condition):
            self.errors.append(message)

    def warn(self, condition: bool, message: str) -> None:
        if not bool(condition):
            self.warnings.append(message)


def _run_git(repo_root: Path, *arguments: str) -> list[str]:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return [line for line in completed.stdout.splitlines() if line]


def _normal_relative(path: str | Path) -> str:
    return Path(path).as_posix().lstrip("./")


def outside_audit_paths(paths: Iterable[str], audit_relative: str) -> list[str]:
    """Return changed paths outside the experiment tree (pure/testable helper)."""

    prefix = _normal_relative(audit_relative).rstrip("/") + "/"
    return sorted(
        _normal_relative(path)
        for path in paths
        if _normal_relative(path) != prefix[:-1]
        and not _normal_relative(path).startswith(prefix)
    )


def validate_repo_scope(
    repo_root: Path,
    audit_root: Path,
    parent_sha: str,
    checks: Checks,
    *,
    strict_untracked: bool,
) -> None:
    audit_relative = audit_root.resolve().relative_to(repo_root.resolve()).as_posix()
    tracked_changes = _run_git(repo_root, "diff", "--name-only", parent_sha, "--")
    outside_tracked = outside_audit_paths(tracked_changes, audit_relative)
    checks.require(
        not outside_tracked,
        "tracked changes relative to the frozen parent exist outside the new audit tree: "
        f"count={len(outside_tracked)}",
    )

    untracked = _run_git(repo_root, "ls-files", "--others", "--exclude-standard")
    outside_untracked = outside_audit_paths(untracked, audit_relative)
    if strict_untracked:
        checks.require(
            not outside_untracked,
            "untracked files exist outside the new audit tree under strict mode: "
            f"count={len(outside_untracked)}",
        )
    else:
        checks.warn(
            not outside_untracked,
            "pre-existing/unattributed untracked files outside the audit tree were not treated "
            f"as audit modifications: count={len(outside_untracked)}",
        )


def _read_csv(path: Path, checks: Checks) -> pd.DataFrame | None:
    if not path.is_file():
        checks.require(False, f"missing required aggregate CSV: {path.relative_to(AUDIT_ROOT)}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as error:  # pragma: no cover - defensive diagnostic
        checks.require(False, f"cannot parse aggregate CSV {path.name}: {type(error).__name__}")
        return None


def _is_false(series: pd.Series) -> pd.Series:
    return series.map(
        lambda value: value is False
        or value == 0
        or str(value).strip().lower() in {"false", "0"}
    )


def _is_true(value: Any) -> bool:
    return value is True or value == 1 or str(value).strip().lower() in {"true", "1"}


def _require_columns(
    frame: pd.DataFrame,
    required: Iterable[str],
    label: str,
    checks: Checks,
) -> bool:
    missing = sorted(set(required) - set(frame.columns))
    checks.require(not missing, f"{label} is missing required columns: {missing}")
    return not missing


def validate_oof_matrix(frame: pd.DataFrame, config: Mapping[str, Any], checks: Checks) -> None:
    required = {*OOF_IDENTITY, *METRIC_COLUMNS, "n_folds"}
    if not _require_columns(frame, required, "oof_metrics.csv", checks):
        return
    checks.require(len(frame) == EXPECTED_COUNTS["oof_metrics"], "OOF row count is not 3276")
    checks.require(not frame.duplicated(OOF_IDENTITY).any(), "OOF analysis identities are duplicated")
    checks.require((pd.to_numeric(frame["n_folds"], errors="coerce") == 5).all(), "OOF rows do not all combine five folds")
    checks.require((pd.to_numeric(frame["n"], errors="coerce") > 0).all(), "OOF rows contain empty cohorts")
    checks.require(set(frame["seed"].astype(int)) == set(config["frozen"]["seeds"]), "OOF seed set drifted")
    checks.require(set(frame["arm"].astype(str)) == set(config["frozen"]["arms"]), "OOF arm set drifted")

    main = frame.loc[frame["representation"].isin(MAIN_REPRESENTATIONS)].copy()
    matched = frame.loc[frame["representation"].isin(ORACLE_TO_MATCHED.values())].copy()
    checks.require(len(main) == 2520, "main Z1--Z7 OOF matrix does not contain 2520 rows")
    checks.require(len(matched) == 756, "matched Z4 oracle-reference matrix does not contain 756 rows")
    checks.require(
        set(main["representation"].astype(str)) == set(MAIN_REPRESENTATIONS),
        "main representation set is not exactly Z1--Z7",
    )
    checks.require(
        set(matched["representation"].astype(str)) == set(ORACLE_TO_MATCHED.values()),
        "matched-reference representation set drifted",
    )

    static_raw = main.loc[(main["task_type"] == "static") & (main["target_kind"] == "raw")]
    static_residual = main.loc[(main["task_type"] == "static") & (main["target_kind"] != "raw")]
    dynamic = main.loc[main["task_type"] == "dynamic"]
    checks.require(len(static_raw) == 448, "static raw matrix is not 448 rows")
    checks.require(len(static_residual) == 560, "static residual matrix is not 560 rows")
    checks.require(len(dynamic) == 1512, "dynamic matrix is not 1512 rows")
    checks.require(set(static_raw["target"].astype(str)) == {"FTV", "LD", "SPH", "BPE"}, "static raw target set drifted")
    checks.require(set(static_raw["timing"].astype(str)) == set(VISITS), "static timing set drifted")
    checks.require(set(dynamic["interval"].astype(str)) == set(INTERVALS), "dynamic interval set drifted")
    checks.require(set(dynamic["input_variant"].astype(str)) == {"difference", "prefix"}, "dynamic input variants are incomplete")
    checks.require(
        (frame["transformed_r2_aggregation"] == "outer_test_n_weighted_fold_r2").all(),
        "transformed R2 aggregation contract drifted",
    )
    checks.require(
        (frame.loc[frame["task_type"] == "static", "target_definition"] == "goal6_workbook_endpoint").all(),
        "static target-definition provenance drifted",
    )
    checks.require(
        (frame.loc[frame["task_type"] == "dynamic", "target_definition"] == "adjacent_percent_change_new_extension").all(),
        "dynamic target-definition provenance drifted",
    )

    primary = main.loc[main["target_kind"] == "residual_ftv"]
    secondary = main.loc[main["target_kind"] == "residual_ftv_ld"]
    checks.require(set(primary["target"].astype(str)) == {"LD", "SPH", "BPE"}, "FTV-only residual target set drifted")
    checks.require(set(secondary["target"].astype(str)) == {"SPH", "BPE"}, "FTV+LD residual target set drifted")
    residual = main.loc[main["target_kind"] != "raw"]
    raw = main.loc[main["target_kind"] == "raw"]
    checks.require((residual["natural_metric_interpretation"] == "conditional_target_reconstruction").all(), "residual natural-metric interpretation is ambiguous")
    checks.require((raw["natural_metric_interpretation"] == "raw_target").all(), "raw natural-metric interpretation drifted")
    checks.require((residual["rank_aggregation"] == "outer_test_n_weighted_fold_residual_metric").all(), "residual rank aggregation contract drifted")
    checks.require((raw["rank_aggregation"] == "pooled_oof_natural_target").all(), "raw rank aggregation contract drifted")
    checks.require(residual[["natural_r2", "rmse", "mae"]].isna().all().all(), "residual rows misleadingly expose natural-unit residual metrics")
    checks.require(raw[["residual_spearman", "residual_transformed_r2"]].isna().all().all(), "raw rows unexpectedly expose residual metrics")
    checks.require(
        np.allclose(
            pd.to_numeric(residual["spearman"], errors="coerce"),
            pd.to_numeric(residual["residual_spearman"], errors="coerce"),
            equal_nan=True,
        ),
        "primary residual Spearman alias disagrees with explicit residual metric",
    )
    checks.require(
        np.allclose(
            pd.to_numeric(residual["transformed_r2"], errors="coerce"),
            pd.to_numeric(residual["residual_transformed_r2"], errors="coerce"),
            equal_nan=True,
        ),
        "primary residual transformed R2 alias disagrees with explicit residual metric",
    )

    matched_static = matched.loc[matched["task_type"] == "static"]
    matched_dynamic = matched.loc[matched["task_type"] == "dynamic"]
    checks.require(len(matched_static) == 432, "matched static reference matrix is not 432 rows")
    checks.require(len(matched_dynamic) == 324, "matched dynamic reference matrix is not 324 rows")
    checks.require((matched_dynamic["input_variant"] == "difference").all(), "matched dynamic references are not literal differences only")


def _identity_set(frame: pd.DataFrame, columns: Sequence[str]) -> set[tuple[str, ...]]:
    return set(
        map(
            tuple,
            frame.loc[:, columns].fillna("").astype(str).to_numpy(),
        )
    )


def validate_fold_tables(
    root: Path,
    config: Mapping[str, Any],
    checks: Checks,
    *,
    oof: pd.DataFrame | None,
) -> None:
    expected_alphas = np.asarray(config["probe"]["alphas"], dtype=np.float64)
    for filename, count in (
        ("fold_metrics.csv", EXPECTED_COUNTS["fold_metrics"]),
        ("hyperparameter_selections.csv", EXPECTED_COUNTS["hyperparameter_selections"]),
        ("coverage.csv", EXPECTED_COUNTS["coverage"]),
    ):
        frame = _read_csv(root / "metrics" / filename, checks)
        if frame is None:
            continue
        checks.require(len(frame) == count, f"{filename} row count is not {count}")
        identity = [*OOF_IDENTITY, "fold"]
        if _require_columns(frame, identity, filename, checks):
            checks.require(not frame.duplicated(identity).any(), f"{filename} contains duplicate fold identities")
            checks.require(set(frame["fold"].astype(int)) == set(config["frozen"]["folds"]), f"{filename} fold set drifted")
            per_identity = frame.groupby(OOF_IDENTITY, dropna=False, sort=False).size()
            checks.require((per_identity == 5).all(), f"{filename} does not contain exactly five folds per OOF identity")
            if oof is not None and set(OOF_IDENTITY).issubset(oof.columns):
                checks.require(
                    _identity_set(frame, OOF_IDENTITY) == _identity_set(oof, OOF_IDENTITY),
                    f"{filename} identities do not exactly match oof_metrics.csv",
                )

        if filename == "hyperparameter_selections.csv":
            required = {
                "selected_alpha",
                "n_train",
                "n_validation",
                "n_test",
                "feature_scaler",
                "test_used_for_scaler",
                "test_used_for_alpha_selection",
                "test_predict_call_count",
                "alpha_validation_mse_json",
            }
            if _require_columns(frame, required, filename, checks):
                selected = pd.to_numeric(frame["selected_alpha"], errors="coerce").to_numpy()
                allowed = np.any(np.isclose(selected[:, None], expected_alphas[None, :]), axis=1)
                checks.require(allowed.all(), "a selected ridge alpha is outside the preregistered grid")
                checks.require(_is_false(frame["test_used_for_scaler"]).all(), "test data entered a feature scaler")
                checks.require(_is_false(frame["test_used_for_alpha_selection"]).all(), "test data entered alpha selection")
                checks.require((pd.to_numeric(frame["test_predict_call_count"], errors="coerce") == 1).all(), "outer test was not predicted exactly once per probe")
                checks.require((frame["feature_scaler"] == "outer_train_StandardScaler_population_variance").all(), "feature-scaler scope drifted")
                checks.require((frame[["n_train", "n_validation", "n_test"]].apply(pd.to_numeric, errors="coerce") >= 3).all().all(), "a probe split has fewer than three eligible rows")

        if filename == "coverage.csv" and _require_columns(
            frame,
            {"feature_valid_total", "target_valid_total", "joint_valid_total", "n_train", "n_validation", "n_test"},
            filename,
            checks,
        ):
            joint = pd.to_numeric(frame["joint_valid_total"], errors="coerce")
            split_total = frame[["n_train", "n_validation", "n_test"]].apply(pd.to_numeric, errors="coerce").sum(axis=1)
            checks.require((joint == split_total).all(), "coverage joint totals do not equal train+validation+test")
            checks.require((joint <= pd.to_numeric(frame["feature_valid_total"], errors="coerce")).all(), "joint validity exceeds feature validity")
            checks.require((joint <= pd.to_numeric(frame["target_valid_total"], errors="coerce")).all(), "joint validity exceeds target validity")


def validate_contract_tables(root: Path, checks: Checks) -> None:
    specifications = (
        ("target_transform_fits.csv", EXPECTED_COUNTS["target_transform_fits"], "fit_scope"),
        ("residualizer_fits.csv", EXPECTED_COUNTS["residualizer_fits"], "fit_scope"),
    )
    for filename, count, scope_column in specifications:
        frame = _read_csv(root / "metrics" / filename, checks)
        if frame is None:
            continue
        checks.require(len(frame) == count, f"{filename} row count is not {count}")
        if _require_columns(frame, {scope_column, "n_train"}, filename, checks):
            checks.require((frame[scope_column] == "outer_train_only").all(), f"{filename} contains a non-training-only fit")
            checks.require((pd.to_numeric(frame["n_train"], errors="coerce") >= 3).all(), f"{filename} contains an invalid training count")
    residualizers = _read_csv(root / "metrics" / "residualizer_fits.csv", checks)
    if residualizers is not None and _require_columns(
        residualizers,
        {"alpha", "model", "target_kind", "target", "predictors"},
        "residualizer_fits.csv",
        checks,
    ):
        checks.require(np.isclose(pd.to_numeric(residualizers["alpha"], errors="coerce"), 1.0).all(), "residualizer alpha drifted from 1.0")
        checks.require((residualizers["model"] == "Ridge").all(), "residualizer model is not uniformly Ridge")
        expected_predictors = np.where(
            residualizers["task_type"].astype(str) == "dynamic",
            np.where(
                residualizers["target_kind"].astype(str) == "residual_ftv",
                "delta_FTV",
                "delta_FTV+delta_LD",
            ),
            np.where(
                residualizers["target_kind"].astype(str) == "residual_ftv",
                "FTV",
                "FTV+LD",
            ),
        )
        checks.require(
            np.array_equal(residualizers["predictors"].astype(str).to_numpy(), expected_predictors),
            "residualizer predictor contract drifted",
        )

    transforms = _read_csv(root / "metrics" / "target_transform_fits.csv", checks)
    if transforms is not None and _require_columns(
        transforms,
        {"task_type", "family", "log1p"},
        "target_transform_fits.csv",
        checks,
    ):
        dynamic = transforms.loc[transforms["task_type"] == "dynamic"]
        checks.require(_is_false(dynamic["log1p"]).all(), "a dynamic target/predictor incorrectly uses log1p")
        static = transforms.loc[transforms["task_type"] == "static"]
        expected_log = static["family"].astype(str).isin({"FTV", "LD", "BPE"})
        observed_log = static["log1p"].map(_is_true)
        checks.require(np.array_equal(observed_log.to_numpy(), expected_log.to_numpy()), "static family log1p contract drifted")


def validate_matrix_exports(
    root: Path,
    checks: Checks,
    *,
    oof: pd.DataFrame | None,
) -> None:
    loaded: dict[str, pd.DataFrame] = {}
    for stem in (
        "static_target_matrix",
        "residual_target_matrix",
        "dynamic_target_matrix",
        "representation_location_comparison",
        "oracle_localization_comparison",
    ):
        frame = _read_csv(root / "metrics" / f"{stem}.csv", checks)
        if frame is None:
            continue
        loaded[stem] = frame
        checks.require(len(frame) == EXPECTED_COUNTS[stem], f"{stem}.csv row count drifted")
    if oof is not None and set(OOF_IDENTITY).issubset(oof.columns):
        main = oof.loc[oof["representation"].isin(MAIN_REPRESENTATIONS)]
        expected_subsets = {
            "static_target_matrix": main.loc[(main["task_type"] == "static") & (main["target_kind"] == "raw")],
            "residual_target_matrix": main.loc[(main["task_type"] == "static") & (main["target_kind"] != "raw")],
            "dynamic_target_matrix": main.loc[main["task_type"] == "dynamic"],
        }
        for name, expected in expected_subsets.items():
            observed = loaded.get(name)
            if observed is None or not set(OOF_IDENTITY).issubset(observed.columns):
                continue
            checks.require(
                _identity_set(observed, OOF_IDENTITY) == _identity_set(expected, OOF_IDENTITY),
                f"{name}.csv is not the exact intended OOF subset",
            )
    localization = _read_csv(root / "metrics" / "oracle_localization_comparison.csv", checks)
    if localization is not None and _require_columns(
        localization,
        {"n_oracle", "n_full_local_matched", "oracle_representation", "matched_reference"},
        "oracle_localization_comparison.csv",
        checks,
    ):
        checks.require(
            np.array_equal(
                pd.to_numeric(localization["n_oracle"], errors="coerce").to_numpy(),
                pd.to_numeric(localization["n_full_local_matched"], errors="coerce").to_numpy(),
            ),
            "oracle localization comparisons use unmatched populations",
        )
        mapping = localization[["oracle_representation", "matched_reference"]].drop_duplicates()
        observed = set(map(tuple, mapping.astype(str).to_numpy()))
        checks.require(observed == set(ORACLE_TO_MATCHED.items()), "oracle-to-matched-Z4 mapping drifted")


def _public_text_paths(root: Path) -> list[Path]:
    text_suffixes = {"", ".csv", ".json", ".md", ".py", ".sh", ".txt", ".tsv", ".yaml", ".yml"}
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.name != ".gitkeep"
        and "private" not in path.name.lower()
        and path.suffix.lower() in text_suffixes
        and ".pytest_cache" not in path.parts
        and "__pycache__" not in path.parts
    )


def forbidden_public_columns(columns: Iterable[str]) -> list[str]:
    return sorted(
        str(column)
        for column in columns
        if any(pattern.match(str(column).strip()) for pattern in FORBIDDEN_PUBLIC_COLUMN_PATTERNS)
    )


def identifier_token_hits(path: Path, identifiers: Sequence[str]) -> int:
    """Count exact identifier tokens without returning or logging their values."""

    text = path.read_text(encoding="utf-8", errors="ignore")
    hits = 0
    for identifier in identifiers:
        token = str(identifier)
        if not token:
            continue
        pattern = rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])"
        if re.search(pattern, text):
            hits += 1
    return hits


def load_private_identifiers(config: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    paths = config["paths"]
    target_path = resolve_path(paths["eligible_target_table"])
    authenticate(target_path, paths["eligible_target_table_sha256"], "eligible target table")
    frame = pd.read_csv(target_path, usecols=["patient_id", "trial_id"])
    patients = sorted(frame["patient_id"].astype(str).unique())
    trials = sorted(frame["trial_id"].astype(str).unique())
    return patients, trials


def validate_public_privacy(
    root: Path,
    config: Mapping[str, Any],
    checks: Checks,
    *,
    identifiers: tuple[Sequence[str], Sequence[str]] | None = None,
) -> None:
    public_paths = _public_text_paths(root)
    checks.require(bool(public_paths), "no public aggregate text artifacts were found")
    patient_ids, trial_ids = identifiers if identifiers is not None else load_private_identifiers(config)
    all_identifiers = [*map(str, patient_ids), *map(str, trial_ids)]
    for path in public_paths:
        if path.suffix.lower() == ".csv":
            try:
                columns = pd.read_csv(path, nrows=0).columns
            except Exception as error:
                checks.require(False, f"cannot inspect public CSV schema {path.name}: {type(error).__name__}")
                continue
            forbidden = forbidden_public_columns(columns)
            checks.require(not forbidden, f"public CSV {path.name} has identifier/patient-level columns: {forbidden}")
        hits = identifier_token_hits(path, all_identifiers)
        checks.require(hits == 0, f"public artifact {path.name} contains exact private identifier tokens: count={hits}")


def validate_tracked_privacy(repo_root: Path, root: Path, checks: Checks) -> None:
    audit_relative = root.resolve().relative_to(repo_root.resolve()).as_posix()
    tracked = _run_git(repo_root, "ls-files", audit_relative)
    sensitive: list[str] = []
    for relative in tracked:
        path = Path(relative)
        if "private" in path.name.lower() or path.suffix.lower() in SENSITIVE_TRACKED_SUFFIXES:
            if path.name != ".gitkeep":
                sensitive.append(relative)
    checks.require(not sensitive, f"sensitive/private artifacts are tracked: count={len(sensitive)}")

    private_artifacts = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and (
            "private" in path.name.lower()
            or path.suffix.lower() in SENSITIVE_TRACKED_SUFFIXES
        )
    )
    for prediction in private_artifacts:
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", str(prediction)],
            cwd=repo_root,
            check=False,
        ).returncode == 0
        relative = prediction.relative_to(root)
        checks.require(ignored, f"private artifact is not covered by .gitignore: {relative}")
        checks.require((prediction.stat().st_mode & 0o777) == 0o600, f"private artifact mode is not 0600: {relative}")
        checks.require((prediction.parent.stat().st_mode & 0o777) == 0o700, f"private artifact directory mode is not 0700: {relative.parent}")


def _private_identity(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(row[column]) for column in OOF_IDENTITY)


def validate_private_prediction_file(
    root: Path,
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
    oof: pd.DataFrame | None,
    checks: Checks,
) -> None:
    prediction = root / "predictions" / "oof_predictions.private.csv.gz"
    if not prediction.is_file():
        checks.require(False, "private OOF prediction file is missing")
        return
    if oof is None or not set(OOF_IDENTITY).issubset(oof.columns):
        checks.require(False, "public OOF identities are unavailable for private validation")
        return

    targets = load_targets(config)
    fold_splits = load_fold_splits(config, targets)
    patient_to_bit = {
        str(patient_id): index for index, patient_id in enumerate(targets.patient_ids)
    }
    public_records: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in oof.fillna("").to_dict("records"):
        identity = _private_identity(row)
        if identity in public_records:
            checks.require(False, "public OOF identity is duplicated during private validation")
            return
        public_records[identity] = {
            "expected_n": int(row["n"]),
            "expected_patient_sha256": str(row["eligible_patient_set_sha256"]),
            "expected_oof_identity_sha256": str(row["oof_identity_sha256"]),
            "bits": 0,
            "count": 0,
            "fold_mask": 0,
        }
    try:
        with gzip.open(prediction, "rt", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            header = reader.fieldnames or []
            row_count = 0
            for row in reader:
                row_count += 1
                identity = _private_identity(row)
                record = public_records.get(identity)
                if record is None:
                    checks.require(False, "private prediction contains an unknown OOF endpoint identity")
                    break
                patient_id = str(row["patient_id"])
                patient_index = patient_to_bit.get(patient_id)
                if patient_index is None:
                    checks.require(False, "private prediction contains an unknown patient identifier")
                    break
                try:
                    fold = int(row["fold"])
                except (TypeError, ValueError):
                    checks.require(False, "private prediction contains an invalid fold")
                    break
                if fold not in fold_splits or fold_splits[fold][patient_index] != "test":
                    checks.require(False, "private prediction row is not in its declared outer-test fold")
                    break
                bit = 1 << patient_index
                if int(record["bits"]) & bit:
                    checks.require(False, "private OOF prediction contains a duplicate endpoint/patient identity")
                    break
                record["bits"] = int(record["bits"]) | bit
                record["count"] = int(record["count"]) + 1
                record["fold_mask"] = int(record["fold_mask"]) | (1 << fold)
    except Exception as error:  # pragma: no cover - defensive diagnostic
        checks.require(False, f"private OOF prediction cannot be read: {type(error).__name__}")
        return
    checks.require(tuple(header) == PREDICTION_COLUMNS, "private OOF prediction schema drifted")
    expected = int(summary.get("private_prediction_rows", -1))
    checks.require(row_count == expected and row_count > 0, "private OOF prediction row count disagrees with run summary")
    checks.require(
        row_count
        == int(summary.get("expected_private_prediction_rows_from_oof_n_sum", -2)),
        "private OOF prediction count disagrees with public OOF n sum",
    )
    complete_endpoints = 0
    for record in public_records.values():
        bits = int(record["bits"])
        count = int(record["count"])
        if count != int(record["expected_n"]) or bits.bit_count() != count:
            continue
        identifiers = [
            str(targets.patient_ids[index])
            for index in range(len(targets.patient_ids))
            if bits & (1 << index)
        ]
        identity_rows = [
            f"{patient_id}|{fold}"
            for patient_id in identifiers
            for fold, labels in fold_splits.items()
            if labels[patient_to_bit[patient_id]] == "test"
        ]
        if (
            int(record["fold_mask"]) == 0b11111
            and ordered_sha256(sorted(identifiers))
            == record["expected_patient_sha256"]
            and ordered_sha256(sorted(identity_rows))
            == record["expected_oof_identity_sha256"]
        ):
            complete_endpoints += 1
    checks.require(
        complete_endpoints == EXPECTED_COUNTS["oof_metrics"],
        "private OOF patient/fold identity coverage differs from public aggregate contracts",
    )


def validate_manifests(
    root: Path,
    config: Mapping[str, Any],
    lock_verification: Mapping[str, Any],
    checks: Checks,
) -> Mapping[str, Any]:
    summary_path = root / "metrics" / "run_summary.json"
    provenance_path = root / "manifests" / "input_provenance.json"
    if not summary_path.is_file() or not provenance_path.is_file():
        checks.require(False, "run summary or input provenance manifest is missing")
        return {}
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except Exception as error:
        checks.require(False, f"run summary/provenance JSON is invalid: {type(error).__name__}")
        return {}
    checks.require(summary.get("status") == "COMPLETE", "formal run status is not COMPLETE")
    checks.require(summary.get("oof_metric_rows") == EXPECTED_COUNTS["oof_metrics"], "run summary OOF count drifted")
    checks.require(summary.get("fold_metric_rows") == EXPECTED_COUNTS["fold_metrics"], "run summary fold count drifted")
    checks.require(summary.get("feature_cells") == EXPECTED_COUNTS["frozen_cell_manifest"], "run summary feature-cell count drifted")
    checks.require(summary.get("encoder_retrained") is False, "run summary reports encoder retraining")
    checks.require(summary.get("pcr_read") is False, "run summary reports pCR read")
    checks.require(summary.get("pcr_used_for_selection") is False, "run summary reports pCR selection")
    checks.require(summary.get("test_used_for_alpha_selection") is False, "run summary reports test-based alpha selection")
    lock_sha256 = str(lock_verification.get("lock_sha256", ""))
    checks.require(bool(lock_sha256), "preregistration lock verification lacks SHA-256")
    checks.require(
        str(summary.get("preregistration_lock_sha256", "")) == lock_sha256,
        "run summary preregistration lock hash drifted",
    )
    checks.require(
        str(provenance.get("preregistration_lock_sha256", "")) == lock_sha256,
        "input provenance preregistration lock hash drifted",
    )
    checks.require(summary.get("test_predict_calls_per_probe") == 1, "run summary test-predict count drifted")
    checks.require(provenance.get("patient_count") == config["frozen"]["patient_count"], "provenance patient count drifted")
    privacy = provenance.get("privacy", {})
    checks.require(privacy.get("pcr_column_parsed") is False, "provenance reports parsing pCR")
    checks.require(privacy.get("patient_identifiers_in_public_outputs") is False, "provenance privacy declaration failed")

    cells = _read_csv(root / "manifests" / "frozen_cell_manifest.csv", checks)
    if cells is not None:
        checks.require(len(cells) == EXPECTED_COUNTS["frozen_cell_manifest"], "frozen cell manifest is not 20 rows")
        if _require_columns(cells, {"seed", "arm", "fold", "selected", "test_data_used", "pcr_used", "encoder_frozen", "training_performed"}, "frozen_cell_manifest.csv", checks):
            checks.require(not cells.duplicated(["seed", "arm", "fold"]).any(), "frozen feature cells are duplicated")
            checks.require(cells["selected"].map(_is_true).all(), "a frozen checkpoint is not marked selected")
            checks.require(_is_false(cells["test_data_used"]).all(), "a frozen checkpoint reports test-data use")
            checks.require(_is_false(cells["pcr_used"]).all(), "a frozen checkpoint reports pCR use")
            checks.require(cells["encoder_frozen"].map(_is_true).all(), "a spatial cell does not prove frozen encoder use")
            checks.require(_is_false(cells["training_performed"]).all(), "a spatial cell reports new training")
    for filename, expected in (("target_contract.csv", 16), ("representation_contract.csv", 7)):
        frame = _read_csv(root / "manifests" / filename, checks)
        if frame is not None:
            checks.require(len(frame) == expected, f"{filename} row count drifted")
    return summary


def validate_final_outputs(root: Path, checks: Checks, *, allow_pending_delivery: bool) -> None:
    for relative in sorted(FINAL_PUBLIC_FILES):
        checks.require((root / relative).is_file(), f"missing final public artifact: {relative}")
    gate_jsons = sorted((root / "metrics").glob("*gate*.json")) + sorted((root / "manifests").glob("*gate*.json"))
    checks.require(bool(gate_jsons), "machine-readable gate JSON is missing")
    figures = [
        path
        for path in (root / "figures").glob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".pdf", ".svg"}
    ]
    checks.require(bool(figures), "no aggregate figure was generated")

    scorecard_path = root / "metrics" / "grounding_candidate_scorecard.csv"
    if scorecard_path.is_file():
        scorecard = pd.read_csv(scorecard_path)
        normalized = {str(column).strip().lower() for column in scorecard.columns}
        checks.require("weighted_total" not in normalized and "weighted_score" not in normalized, "scorecard contains a forbidden weighted total")

    report_path = root / "reports" / "final_report.md"
    if report_path.is_file():
        report = report_path.read_text(encoding="utf-8")
        checks.require(bool(re.search(r"[\u4e00-\u9fff]", report)), "final report is not written in Chinese")
        checks.require("feature/nonftv-phenotype-decodability-audit" in report, "final report omits the branch")
        if not allow_pending_delivery:
            checks.require("PENDING" not in report.upper(), "final report still contains a pending delivery placeholder")
            checks.require(bool(re.search(r"\b[0-9a-f]{40}\b", report)), "final report omits a full commit SHA")
            checks.require(bool(re.search(r"push", report, re.IGNORECASE)), "final report omits push status")


def validate(
    config_path: Path,
    *,
    core_only: bool,
    strict_untracked: bool,
    skip_private_row_count: bool,
    allow_pending_delivery: bool,
    allow_descendant_head: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    lock_verification = require_preregistration_lock(
        require_exact_parent=not allow_descendant_head
    )
    checks = Checks()
    for relative in sorted(CORE_PUBLIC_FILES):
        checks.require((AUDIT_ROOT / relative).is_file(), f"missing core public artifact: {relative}")

    oof = _read_csv(AUDIT_ROOT / "metrics" / "oof_metrics.csv", checks)
    if oof is not None:
        validate_oof_matrix(oof, config, checks)
    validate_fold_tables(AUDIT_ROOT, config, checks, oof=oof)
    validate_contract_tables(AUDIT_ROOT, checks)
    validate_matrix_exports(AUDIT_ROOT, checks, oof=oof)
    summary = validate_manifests(AUDIT_ROOT, config, lock_verification, checks)
    validate_public_privacy(AUDIT_ROOT, config, checks)
    validate_tracked_privacy(REPO_ROOT, AUDIT_ROOT, checks)
    validate_repo_scope(
        REPO_ROOT,
        AUDIT_ROOT,
        str(config["parent_sha"]),
        checks,
        strict_untracked=strict_untracked,
    )
    if summary and not skip_private_row_count:
        validate_private_prediction_file(AUDIT_ROOT, summary, config, oof, checks)
    if not core_only:
        validate_final_outputs(
            AUDIT_ROOT,
            checks,
            allow_pending_delivery=allow_pending_delivery,
        )

    return {
        "status": "PASS" if not checks.errors else "FAIL",
        "checks": checks.checked,
        "error_count": len(checks.errors),
        "warning_count": len(checks.warnings),
        "errors": checks.errors,
        "warnings": checks.warnings,
        "core_only": core_only,
        "private_prediction_row_count_checked": not skip_private_row_count,
        "preregistration_lock_sha256": lock_verification["lock_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=AUDIT_ROOT / "configs" / "audit.json")
    parser.add_argument("--core-only", action="store_true", help="validate formal probe outputs before gates/report rendering")
    parser.add_argument("--strict-untracked", action="store_true", help="also fail on unrelated untracked paths outside this audit")
    parser.add_argument("--skip-private-row-count", action="store_true", help="skip the potentially slow independent gzip row count")
    parser.add_argument("--allow-pending-delivery", action="store_true", help="permit commit/push placeholders during pre-commit validation")
    parser.add_argument("--allow-descendant-head", action="store_true", help="verify immutable lock on a descendant delivery commit")
    arguments = parser.parse_args()
    result = validate(
        arguments.config,
        core_only=arguments.core_only,
        strict_untracked=arguments.strict_untracked,
        skip_private_row_count=arguments.skip_private_row_count,
        allow_pending_delivery=arguments.allow_pending_delivery,
        allow_descendant_head=arguments.allow_descendant_head,
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

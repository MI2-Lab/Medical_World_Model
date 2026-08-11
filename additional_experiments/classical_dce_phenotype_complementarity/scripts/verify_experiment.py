#!/usr/bin/env python3
"""Fail-closed verification for the classical DCE phenotype experiment.

The verifier reads only source contracts and aggregate experiment artifacts.  It
never writes patient rows: its sole output is ``metrics/verification.json``.
Run with ``--allow-pending-delivery`` before the delivery commit; omit that flag
for the final branch/SHA/push audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "feature/classical-dce-phenotype-complementarity"
EXPECTED_PRIMARY_POPULATION = "clinical_radiomics_complete_384"
EXPECTED_PRIMARY_PROTOCOL = "primary_stratified_384"
EXPECTED_MRI_N = 375
EXPECTED_PRIMARY_N = 384
EXPECTED_PRIMARY_POSITIVE = 113
TIMINGS = ("T0", "T1", "T2", "T3")
VIEWS = ("static", "longitudinal")
PRIMARY_MODELS = ("C", "F", "N", "FULL", "C+F", "C+N", "C+FULL")
PRIMARY_SCENARIOS = ("complete_case", "train_median_indicator")
KEY_COMPARISONS = (
    ("C+F", "C+FULL"),
    ("C", "C+N"),
    ("C+F", "C+F+N_RES"),
)

PCR_COLUMNS = (
    "protocol",
    "population",
    "scenario",
    "view",
    "timing",
    "timing_label",
    "model_type",
    "model",
    "n",
    "n_positive",
    "n_negative",
    "auroc",
    "auprc",
    "balanced_accuracy",
    "brier",
)
INCREMENTAL_COLUMNS = (
    "protocol",
    "population",
    "scenario",
    "view",
    "timing",
    "timing_label",
    "model_type",
    "comparison",
    "baseline_model",
    "augmented_model",
    "n",
    "n_positive",
    "n_bootstrap",
    "delta_auroc",
    "delta_auroc_ci_low",
    "delta_auroc_ci_high",
    "delta_auprc",
    "delta_auprc_ci_low",
    "delta_auprc_ci_high",
    "brier_improvement",
    "brier_improvement_ci_low",
    "brier_improvement_ci_high",
)
MATCHED_COLUMNS = (
    "protocol",
    "population",
    "scenario",
    "view",
    "timing",
    "comparison",
    "baseline_model",
    "augmented_model",
    "n",
    "pCR_positive",
    "missingness_exclusions",
    "exclusion_reason",
    "patient_set_sha256",
)
MRI_TRADITIONAL_COMPARISON_COLUMNS = (
    "population",
    "task",
    "view",
    "timing",
    "timing_label",
    "target",
    "traditional_model",
    "mri_model",
    "n",
    "traditional_auroc",
    "mri_auroc",
    "difference_mri_minus_traditional",
    "mri_aggregation",
)

REQUIRED_TABLES = (
    "features/radiomics_feature_inventory.csv",
    "metrics/missingness.csv",
    "matched_population_manifest.csv",
    "metrics/matched_population_manifest.csv",
    "metrics/static_radiomics.csv",
    "metrics/longitudinal_radiomics.csv",
    "metrics/pcr_oof_metrics.csv",
    "metrics/incremental_effects.csv",
    "metrics/profile_oof_metrics.csv",
    "metrics/redundancy_metrics.csv",
    "metrics/residualization_metrics.csv",
    "metrics/family_ablation_metrics.csv",
    "metrics/lr_vs_svm.csv",
    "metrics/mri_reference_metrics.csv",
    "metrics/mri_reference_profile_metrics.csv",
    "metrics/mri_reference_traditional_pcr_comparison.csv",
    "metrics/mri_reference_traditional_profile_comparison.csv",
    "metrics/feature_correlation_matrix.csv",
    "metrics/preprocessing_audit.csv",
)
REQUIRED_JSON = (
    "metrics/run_summary.json",
    "metrics/mri_reference_provenance.json",
)
REQUIRED_FIGURES = (
    "figures/timing_auroc.png",
    "figures/c_f_vs_c_full_auroc.png",
    "figures/delta_auroc_forest.png",
    "figures/phenotype_family_comparison.png",
    "figures/hr_her2_heatmap.png",
    "figures/residualized_results.png",
    "figures/feature_correlation_matrix.png",
)
FINAL_REPORT = "reports/final_report.md"

PREFIXED_SIX_DIGIT_ID = re.compile(r"(?:ISPY2-|ACRIN-6698-)\d{6}", re.IGNORECASE)
# A preceding decimal point excludes an ordinary six-decimal metric such as
# 0.123456. CSV cells receive the stronger exact-cell check below.
BARE_SIX_DIGIT_ID = re.compile(r"(?<![A-Za-z0-9.])\d{6}(?![A-Za-z0-9])")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}", re.IGNORECASE)
GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}", re.IGNORECASE)
VISIT_PATTERN = re.compile(r"T[0-3]")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class VerificationFailure(ValueError):
    """A user-facing contract violation."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationFailure(f"cannot read valid JSON at {path}: {error}") from error
    if not isinstance(value, dict):
        raise VerificationFailure(f"JSON root must be an object: {path}")
    return value


def _load_csv(path: Path, *, exact_columns: Sequence[str] | None = None) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except Exception as error:  # pandas exposes several parser/codec subclasses
        raise VerificationFailure(f"cannot read CSV at {path}: {error}") from error
    if exact_columns is not None and tuple(map(str, frame.columns)) != tuple(exact_columns):
        raise VerificationFailure(
            f"schema/order mismatch for {path.name}: expected {list(exact_columns)}, "
            f"observed {list(map(str, frame.columns))}"
        )
    return frame


def _load_csv_contract(
    path: Path,
    required_columns: Sequence[str],
    *,
    allowed_extra_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Load a CSV whose required columns retain their registered relative order."""

    frame = _load_csv(path)
    observed = tuple(map(str, frame.columns))
    missing = [column for column in required_columns if column not in observed]
    unexpected = [
        column
        for column in observed
        if column not in required_columns and column not in allowed_extra_columns
    ]
    observed_required = tuple(column for column in observed if column in required_columns)
    if missing or unexpected or observed_required != tuple(required_columns):
        raise VerificationFailure(
            f"schema/order mismatch for {path.name}: missing={missing}, unexpected={unexpected}"
        )
    return frame


def _require_nonempty(frame: pd.DataFrame, label: str) -> None:
    if frame.empty or len(frame.columns) == 0:
        raise VerificationFailure(f"{label} must be a non-empty aggregate table")


def _numeric(frame: pd.DataFrame, column: str, label: str) -> np.ndarray:
    if column not in frame:
        raise VerificationFailure(f"{label} is missing required column {column}")
    try:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise VerificationFailure(f"{label}.{column} must be numeric") from error
    if not np.isfinite(values).all():
        raise VerificationFailure(f"{label}.{column} contains missing or infinite values")
    return values


def _integers(frame: pd.DataFrame, column: str, label: str) -> np.ndarray:
    values = _numeric(frame, column, label)
    if not np.equal(values, np.floor(values)).all():
        raise VerificationFailure(f"{label}.{column} must contain integers")
    return values.astype(np.int64)


def _strict_strings(frame: pd.DataFrame, column: str, label: str) -> pd.Series:
    if column not in frame:
        raise VerificationFailure(f"{label} is missing required column {column}")
    values = frame[column].astype("string")
    if values.isna().any():
        raise VerificationFailure(f"{label}.{column} contains missing values")
    stripped = values.str.strip()
    if stripped.eq("").any() or not stripped.equals(values):
        raise VerificationFailure(f"{label}.{column} contains blank or padded values")
    return stripped.astype(str)


def _normalize_model(value: object) -> str:
    text = re.sub(r"\s+", "", str(value).upper()).replace("NONFTV", "N")
    text = text.replace("__", "_").replace("-RES", "_RES")
    return text


def _normalize_model_type(value: object) -> str:
    text = re.sub(r"[^A-Z0-9]", "", str(value).upper())
    if text in {"LR", "LOGISTIC", "LOGISTICREGRESSION"}:
        return "LR"
    if text in {"SVM", "RBFSVM", "SVC", "RBFSVC"}:
        return "SVM"
    return text


def _comparison_key(baseline: object, augmented: object) -> tuple[str, str]:
    return _normalize_model(baseline), _normalize_model(augmented)


def _resolve_config_path(root: Path, config_path: Path | None) -> Path:
    path = config_path if config_path is not None else root / "configs" / "experiment.json"
    path = path.expanduser().resolve()
    if not path.is_file():
        raise VerificationFailure(f"missing experiment config: {path}")
    return path


def _inside(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def _resolve_source_path(value: object, root: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _validate_generic_sources(root: Path, contracts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not contracts:
        raise VerificationFailure("verification.source_contracts cannot be empty")
    records: dict[str, Any] = {}
    for index, raw in enumerate(contracts):
        if not isinstance(raw, Mapping):
            raise VerificationFailure(f"source contract {index} must be an object")
        label = str(raw.get("name", f"source_{index}"))
        if "path" not in raw or "sha256" not in raw:
            raise VerificationFailure(f"{label} contract requires path and sha256")
        path = _resolve_source_path(raw["path"], root)
        if _inside(path, root):
            raise VerificationFailure(f"source data must remain outside the experiment: {label}")
        if not path.is_file():
            raise VerificationFailure(f"missing source file for {label}: {path}")
        expected_hash = str(raw["sha256"]).lower()
        if not SHA256_PATTERN.fullmatch(expected_hash):
            raise VerificationFailure(f"{label} has an invalid configured SHA-256")
        observed_hash = _sha256(path)
        if observed_hash != expected_hash:
            raise VerificationFailure(
                f"{label} SHA-256 mismatch: expected {expected_hash}, observed {observed_hash}"
            )

        kind = str(raw.get("format", path.suffix.lstrip("."))).lower()
        expected_columns = raw.get("columns")
        if kind == "csv":
            frame = _load_csv(path)
            if expected_columns is not None and tuple(map(str, frame.columns)) != tuple(expected_columns):
                raise VerificationFailure(f"{label} CSV schema/order differs from config")
            if "rows" in raw and len(frame) != int(raw["rows"]):
                raise VerificationFailure(
                    f"{label} row count differs: expected {raw['rows']}, observed {len(frame)}"
                )
        elif kind in {"xlsx", "xls", "excel"}:
            sheet = raw.get("sheet")
            if sheet is None:
                raise VerificationFailure(f"{label} Excel contract requires sheet")
            workbook = pd.ExcelFile(path)
            if str(sheet) not in workbook.sheet_names:
                raise VerificationFailure(f"{label} is missing configured sheet {sheet}")
            frame = pd.read_excel(path, sheet_name=str(sheet))
            if expected_columns is not None and tuple(map(str, frame.columns)) != tuple(expected_columns):
                raise VerificationFailure(f"{label} Excel schema/order differs from config")
            if "rows" in raw and len(frame) != int(raw["rows"]):
                raise VerificationFailure(
                    f"{label} row count differs: expected {raw['rows']}, observed {len(frame)}"
                )
        elif kind == "json":
            document = _load_json(path)
            if expected_columns is not None and tuple(document) != tuple(expected_columns):
                raise VerificationFailure(f"{label} JSON top-level schema/order differs from config")
        else:
            raise VerificationFailure(f"unsupported source format {kind!r} for {label}")
        records[label] = {"sha256": observed_hash, "size_bytes": path.stat().st_size}
    return {"sources": records, "mode": "config_source_contracts"}


def _validate_native_sources(root: Path, config_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Use the experiment's authoritative loaders for workbook/fold validation."""

    try:
        from data_contracts import load_primary_cohort, make_mri_matched_splits
        from mri_reference import PCR_COLUMNS as PRIVATE_PCR_COLUMNS
        from mri_reference import PROFILE_COLUMNS as PRIVATE_PROFILE_COLUMNS
    except ImportError as error:
        raise VerificationFailure(f"cannot import experiment source contracts: {error}") from error

    source = config.get("source")
    if not isinstance(source, Mapping):
        raise VerificationFailure("config.source must be an object")
    required = (
        "radiomics_workbook",
        "radiomics_sha256",
        "clinical_workbook",
        "clinical_sha256",
        "mri_fold_manifest",
        "mri_fold_manifest_sha256",
        "mri_audit_predictions",
        "mri_profile_predictions",
    )
    missing = [key for key in required if not source.get(key)]
    if missing:
        raise VerificationFailure(f"config.source is missing required entries: {missing}")
    for path_key in (
        "radiomics_workbook",
        "clinical_workbook",
        "mri_fold_manifest",
        "mri_audit_predictions",
        "mri_profile_predictions",
    ):
        path = _resolve_source_path(source[path_key], root)
        if _inside(path, root):
            raise VerificationFailure(f"configured source {path_key} is inside the experiment")
        if not path.is_file():
            raise VerificationFailure(f"configured source {path_key} does not exist: {path}")
    for hash_key in ("radiomics_sha256", "clinical_sha256", "mri_fold_manifest_sha256"):
        if not SHA256_PATTERN.fullmatch(str(source[hash_key])):
            raise VerificationFailure(f"config.source.{hash_key} must be a full SHA-256")

    try:
        cohort, provenance = load_primary_cohort(config)
        matched, splits = make_mri_matched_splits(cohort, config)
    except Exception as error:
        raise VerificationFailure(f"authoritative source contract failed: {error}") from error
    if len(cohort) != EXPECTED_PRIMARY_N or int(cohort["pCR"].sum()) != EXPECTED_PRIMARY_POSITIVE:
        raise VerificationFailure("authoritative cohort is not the required n=384, pCR+=113 population")
    if len(matched) != EXPECTED_MRI_N:
        raise VerificationFailure("locked MRI overlap is not n=375")

    private_contracts = (
        ("mri_audit_predictions", PRIVATE_PCR_COLUMNS, "pcr_oof_predictions"),
        ("mri_profile_predictions", PRIVATE_PROFILE_COLUMNS, "profile_oof_predictions"),
    )
    mri_provenance_path = root / "metrics" / "mri_reference_provenance.json"
    mri_provenance = _load_json(mri_provenance_path) if mri_provenance_path.is_file() else {}
    provenance_inputs = mri_provenance.get("inputs", {})
    for config_key, expected_columns, provenance_key in private_contracts:
        path = _resolve_source_path(source[config_key], root)
        header = pd.read_csv(path, nrows=0)
        if tuple(map(str, header.columns)) != tuple(expected_columns):
            raise VerificationFailure(f"{config_key} private source schema/order mismatch")
        recorded = provenance_inputs.get(provenance_key, {})
        recorded_hash = recorded.get("sha256") if isinstance(recorded, Mapping) else None
        if not recorded_hash or not SHA256_PATTERN.fullmatch(str(recorded_hash)):
            raise VerificationFailure(f"{config_key} lacks a full hash in MRI provenance")
        if _sha256(path) != str(recorded_hash):
            raise VerificationFailure(f"{config_key} hash disagrees with MRI provenance")

    return {
        "mode": "authoritative_experiment_loaders",
        "primary_n": len(cohort),
        "primary_pcr_positive": int(cohort["pCR"].sum()),
        "mri_overlap_n": len(matched),
        "mri_outer_folds": len(splits),
        "radiomics_sha256": provenance["radiomics_sha256"],
        "clinical_sha256": provenance["clinical_sha256"],
        "config_sha256": _sha256(config_path),
    }


def _check_sources(root: Path, config_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    verification = config.get("verification", {})
    contracts = verification.get("source_contracts") if isinstance(verification, Mapping) else None
    if contracts is not None:
        if not isinstance(contracts, list):
            raise VerificationFailure("verification.source_contracts must be a list")
        return _validate_generic_sources(root, contracts)
    return _validate_native_sources(root, config_path, config)


def _check_run_summary(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    summary = _load_json(root / "metrics" / "run_summary.json")
    if summary.get("status") != "complete":
        raise VerificationFailure("run summary does not mark the experiment complete")
    if summary.get("quick_mode") is not False:
        raise VerificationFailure("quick/developer mode cannot support final verification")
    configured_draws = int(config.get("bootstrap_draws", 0))
    if int(summary.get("bootstrap_draws", -1)) != configured_draws or configured_draws < 2000:
        raise VerificationFailure("run-summary bootstrap contract differs from config or is below 2000")
    primary = summary.get("primary_population", {})
    matched = summary.get("mri_matched_population", {})
    if not isinstance(primary, Mapping) or (
        primary.get("name") != EXPECTED_PRIMARY_POPULATION
        or int(primary.get("n", -1)) != EXPECTED_PRIMARY_N
        or int(primary.get("pCR_positive", -1)) != EXPECTED_PRIMARY_POSITIVE
    ):
        raise VerificationFailure("run summary does not certify primary n=384, pCR+=113")
    if not isinstance(matched, Mapping) or int(matched.get("n", -1)) != EXPECTED_MRI_N:
        raise VerificationFailure("run summary does not certify MRI-matched n=375")

    artifact_paths = {
        "pcr_oof_metrics_sha256": root / "metrics" / "pcr_oof_metrics.csv",
        "profile_oof_metrics_sha256": root / "metrics" / "profile_oof_metrics.csv",
        "incremental_effects_sha256": root / "metrics" / "incremental_effects.csv",
        "matched_population_manifest_sha256": root / "matched_population_manifest.csv",
    }
    artifacts = summary.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        raise VerificationFailure("run summary lacks artifact hashes")
    for key, path in artifact_paths.items():
        recorded = str(artifacts.get(key, ""))
        if not SHA256_PATTERN.fullmatch(recorded) or recorded.lower() != _sha256(path):
            raise VerificationFailure(f"run-summary artifact hash mismatch for {path.name}")

    source_summary = summary.get("source", {})
    configured_source = config.get("source", {})
    if isinstance(configured_source, Mapping) and configured_source:
        if not isinstance(source_summary, Mapping):
            raise VerificationFailure("run summary lacks source provenance")
        for key in ("radiomics_sha256", "clinical_sha256"):
            configured = str(configured_source.get(key, ""))
            if configured and str(source_summary.get(key, "")).lower() != configured.lower():
                raise VerificationFailure(f"run-summary source hash differs for {key}")
    return {
        "quick_mode": False,
        "bootstrap_draws": configured_draws,
        "artifact_hashes_verified": len(artifact_paths),
        "primary_n": EXPECTED_PRIMARY_N,
        "mri_n": EXPECTED_MRI_N,
    }


def _check_required_outputs(root: Path) -> dict[str, Any]:
    missing: list[str] = []
    empty: list[str] = []
    for relative in (*REQUIRED_TABLES, *REQUIRED_JSON, FINAL_REPORT):
        path = root / relative
        if not path.is_file():
            missing.append(relative)
        elif path.stat().st_size == 0:
            empty.append(relative)
    for relative in REQUIRED_FIGURES:
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        if path.stat().st_size <= len(PNG_SIGNATURE):
            empty.append(relative)
            continue
        with path.open("rb") as handle:
            if handle.read(len(PNG_SIGNATURE)) != PNG_SIGNATURE:
                raise VerificationFailure(f"required figure is not a PNG: {relative}")
    if missing or empty:
        raise VerificationFailure(f"required outputs missing={missing}, empty/invalid={empty}")
    for relative in REQUIRED_TABLES:
        _require_nonempty(_load_csv(root / relative), relative)
    for relative in REQUIRED_JSON:
        _load_json(root / relative)
    return {
        "tables": len(REQUIRED_TABLES),
        "figures": len(REQUIRED_FIGURES),
        "final_report": FINAL_REPORT,
    }


def _visit_set(value: object) -> set[str]:
    return set(VISIT_PATTERN.findall(str(value)))


def _check_timing_contract(root: Path) -> dict[str, Any]:
    path = root / "information_timing_contract.csv"
    expected_columns = (
        "timing",
        "allowed_visits",
        "static_features",
        "longitudinal_absolute",
        "longitudinal_change",
        "label",
    )
    contract = _load_csv(path, exact_columns=expected_columns)
    if tuple(contract["timing"].astype(str)) != TIMINGS:
        raise VerificationFailure("timing contract must contain exactly T0,T1,T2,T3 in order")
    for index, timing in enumerate(TIMINGS):
        row = contract.iloc[index]
        prefix = set(TIMINGS[: index + 1])
        if _visit_set(row["allowed_visits"]) != prefix:
            raise VerificationFailure(f"{timing} allowed_visits is not the observed prefix")
        if _visit_set(row["static_features"]) != {timing}:
            raise VerificationFailure(f"{timing} static features must use current visit only")
        if _visit_set(row["longitudinal_absolute"]) != prefix:
            raise VerificationFailure(f"{timing} longitudinal absolute visits are not prefix-safe")
        observed_change = _visit_set(row["longitudinal_change"])
        expected_change = set() if index == 0 else {"T0", *TIMINGS[1 : index + 1]}
        if observed_change != expected_change:
            raise VerificationFailure(f"{timing} longitudinal changes violate the T0-prefix contract")
        for column in ("allowed_visits", "static_features", "longitudinal_absolute", "longitudinal_change"):
            if any(TIMINGS.index(visit) > index for visit in _visit_set(row[column])):
                raise VerificationFailure(f"future visit found in {timing}.{column}")
    t3_label = str(contract.loc[contract["timing"] == "T3", "label"].iloc[0]).lower()
    if "late" not in t3_label or "pre-surgery" not in t3_label:
        raise VerificationFailure("T3 must be explicitly marked late/pre-surgery")

    realized = _load_csv(root / "metrics" / "preprocessing_audit.csv")
    required_realized = {"fit_scope", "view", "timing", "feature"}
    if not required_realized.issubset(realized.columns) or realized.empty:
        raise VerificationFailure("preprocessing audit lacks fit_scope/view/timing/feature rows")
    if not realized["fit_scope"].astype(str).eq("outer_train_only").all():
        raise VerificationFailure("a realized preprocessing transform was not fitted outer-train-only")
    _check_timing_values(realized, "preprocessing_audit")
    feature_rows_checked = 0
    for row in realized.itertuples(index=False):
        timing = str(getattr(row, "timing"))
        view = str(getattr(row, "view"))
        feature_visits = _visit_set(getattr(row, "feature"))
        if not feature_visits:
            continue
        timing_index = TIMINGS.index(timing)
        if any(TIMINGS.index(visit) > timing_index for visit in feature_visits):
            raise VerificationFailure(
                f"realized feature uses a future visit at view={view}, timing={timing}"
            )
        if view == "static" and feature_visits != {timing}:
            raise VerificationFailure("realized static feature is not current-visit-only")
        feature_rows_checked += 1
    if feature_rows_checked == 0:
        raise VerificationFailure("preprocessing audit contains no visit-addressable features")
    return {
        "timings": list(TIMINGS),
        "future_visits": 0,
        "t3_label": t3_label,
        "realized_feature_rows_checked": feature_rows_checked,
    }


def _select_primary_rows(frame: pd.DataFrame, scenario: str) -> pd.DataFrame:
    return frame[
        (frame["protocol"].astype(str) == EXPECTED_PRIMARY_PROTOCOL)
        & (frame["population"].astype(str) == EXPECTED_PRIMARY_POPULATION)
        & (frame["scenario"].astype(str) == scenario)
    ].copy()


def _check_timing_values(frame: pd.DataFrame, label: str) -> None:
    timings = set(_strict_strings(frame, "timing", label))
    views = set(_strict_strings(frame, "view", label))
    if not timings.issubset(TIMINGS) or not views.issubset(VIEWS):
        raise VerificationFailure(f"{label} contains unknown timing/view values")
    if "timing_label" in frame:
        t3 = frame.loc[frame["timing"].astype(str) == "T3", "timing_label"].astype(str).str.lower()
        if not t3.empty and not t3.map(lambda x: "late" in x and "pre-surgery" in x).all():
            raise VerificationFailure(f"{label} does not mark every T3 row late/pre-surgery")


def _check_primary_metrics(root: Path) -> dict[str, Any]:
    frame = _load_csv(root / "metrics" / "pcr_oof_metrics.csv", exact_columns=PCR_COLUMNS)
    _require_nonempty(frame, "pcr_oof_metrics.csv")
    _check_timing_values(frame, "pcr_oof_metrics")
    for column in ("n", "n_positive", "n_negative"):
        _integers(frame, column, "pcr_oof_metrics")
    for column in ("auroc", "auprc", "balanced_accuracy", "brier"):
        values = _numeric(frame, column, "pcr_oof_metrics")
        if np.any((values < 0.0) | (values > 1.0)):
            raise VerificationFailure(f"pcr_oof_metrics.{column} must lie in [0,1]")

    observed_cells = 0
    for scenario in PRIMARY_SCENARIOS:
        selected = _select_primary_rows(frame, scenario)
        if selected.empty:
            raise VerificationFailure(f"pCR metrics are missing scenario {scenario}")
        if not np.all(_integers(selected, "n", scenario) == EXPECTED_PRIMARY_N):
            raise VerificationFailure(f"{scenario} pCR rows are not all matched n=384")
        if not np.all(_integers(selected, "n_positive", scenario) == EXPECTED_PRIMARY_POSITIVE):
            raise VerificationFailure(f"{scenario} pCR rows are not all pCR+=113")
        if not np.all(_integers(selected, "n_negative", scenario) == 271):
            raise VerificationFailure(f"{scenario} pCR rows are not all pCR-=271")
        selected["_model"] = selected["model"].map(_normalize_model)
        selected["_model_type"] = selected["model_type"].map(_normalize_model_type)
        keys = ("view", "timing", "_model_type", "_model")
        if selected.duplicated(list(keys)).any():
            raise VerificationFailure(f"duplicate primary pCR metric cells in {scenario}")
        observed = set(map(tuple, selected[list(keys)].itertuples(index=False, name=None)))
        expected = {
            (view, timing, model_type, _normalize_model(model))
            for view in VIEWS
            for timing in TIMINGS
            for model_type in ("LR", "SVM")
            for model in PRIMARY_MODELS
        }
        missing = expected - observed
        if missing:
            raise VerificationFailure(
                f"{scenario} is missing {len(missing)} preregistered model/timing/view cells"
            )
        observed_cells += len(expected)
    return {
        "primary_n": EXPECTED_PRIMARY_N,
        "primary_pcr_positive": EXPECTED_PRIMARY_POSITIVE,
        "required_cells_checked": observed_cells,
        "scenarios": list(PRIMARY_SCENARIOS),
    }


def _check_filtered_view_tables(root: Path) -> dict[str, Any]:
    master = _load_csv(root / "metrics" / "pcr_oof_metrics.csv", exact_columns=PCR_COLUMNS)
    counts: dict[str, int] = {}
    sort_columns = list(PCR_COLUMNS[:8])
    for view, filename in (
        ("static", "static_radiomics.csv"),
        ("longitudinal", "longitudinal_radiomics.csv"),
    ):
        observed = _load_csv(root / "metrics" / filename, exact_columns=PCR_COLUMNS)
        if not observed["view"].astype(str).eq(view).all():
            raise VerificationFailure(f"{filename} contains rows outside view={view}")
        primary_model_names = {_normalize_model(model) for model in PRIMARY_MODELS}
        expected = master.loc[
            (master["view"].astype(str) == view)
            & master["model"].map(_normalize_model).isin(primary_model_names),
            list(PCR_COLUMNS),
        ].copy()
        if len(observed) != len(expected):
            raise VerificationFailure(f"{filename} is not a complete filtered copy of pcr_oof_metrics")
        observed = observed.sort_values(sort_columns).reset_index(drop=True)
        expected = expected.sort_values(sort_columns).reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(
                observed,
                expected,
                check_dtype=False,
                check_exact=False,
                rtol=1e-10,
                atol=1e-12,
            )
        except AssertionError as error:
            raise VerificationFailure(f"{filename} disagrees with pcr_oof_metrics: {error}") from error
        counts[view] = len(observed)
    return counts


def _required_comparison_cells() -> set[tuple[str, str, str, str]]:
    return {
        (view, timing, baseline, augmented)
        for view in VIEWS
        for timing in TIMINGS
        for baseline, augmented in KEY_COMPARISONS
    }


def _check_incremental_effects(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    frame = _load_csv_contract(
        root / "metrics" / "incremental_effects.csv",
        INCREMENTAL_COLUMNS,
        allowed_extra_columns=("bootstrap_seed", "stratification"),
    )
    _require_nonempty(frame, "incremental_effects.csv")
    _check_timing_values(frame, "incremental_effects")
    configured_draws = int(config.get("bootstrap_draws", 0))
    if configured_draws < 2000:
        raise VerificationFailure("config.bootstrap_draws must be at least 2000")

    selected = _select_primary_rows(frame, "complete_case")
    selected = selected[selected["model_type"].map(_normalize_model_type) == "LR"].copy()
    selected["_baseline"] = selected["baseline_model"].map(_normalize_model)
    selected["_augmented"] = selected["augmented_model"].map(_normalize_model)
    selected["_cell"] = list(
        zip(selected["view"], selected["timing"], selected["_baseline"], selected["_augmented"])
    )
    required = _required_comparison_cells()
    selected = selected[selected["_cell"].isin(required)]
    observed = set(selected["_cell"])
    if observed != required or len(selected) != len(required):
        raise VerificationFailure(
            f"paired bootstrap must contain exactly all {len(required)} LR comparison/view/timing cells"
        )
    if not np.all(_integers(selected, "n", "incremental_effects") == EXPECTED_PRIMARY_N):
        raise VerificationFailure("incremental effects are not all matched n=384")
    if not np.all(
        _integers(selected, "n_positive", "incremental_effects") == EXPECTED_PRIMARY_POSITIVE
    ):
        raise VerificationFailure("incremental effects are not all pCR+=113")
    draws = _integers(selected, "n_bootstrap", "incremental_effects")
    if np.any(draws < max(2000, configured_draws)):
        raise VerificationFailure("a key paired comparison has fewer than configured bootstrap draws")

    triplets = (
        ("delta_auroc", "delta_auroc_ci_low", "delta_auroc_ci_high"),
        ("delta_auprc", "delta_auprc_ci_low", "delta_auprc_ci_high"),
        ("brier_improvement", "brier_improvement_ci_low", "brier_improvement_ci_high"),
    )
    for point_column, low_column, high_column in triplets:
        point = _numeric(selected, point_column, "incremental_effects")
        low = _numeric(selected, low_column, "incremental_effects")
        high = _numeric(selected, high_column, "incremental_effects")
        if np.any(low > high):
            raise VerificationFailure(f"invalid CI ordering for {point_column}")
    return {
        "key_comparisons": len(KEY_COMPARISONS),
        "views": len(VIEWS),
        "timings": len(TIMINGS),
        "required_cells": len(required),
        "minimum_draws": int(draws.min()),
    }


def _check_population_identity(root: Path) -> dict[str, Any]:
    frame = _load_csv(
        root / "metrics" / "matched_population_manifest.csv", exact_columns=MATCHED_COLUMNS
    )
    _require_nonempty(frame, "matched_population_manifest.csv")
    _check_timing_values(frame, "matched_population_manifest")
    root_copy = root / "matched_population_manifest.csv"
    if not root_copy.is_file() or _sha256(root_copy) != _sha256(
        root / "metrics" / "matched_population_manifest.csv"
    ):
        raise VerificationFailure("root and metrics matched-population manifests are not identical")
    frame["_baseline"] = frame["baseline_model"].map(_normalize_model)
    frame["_augmented"] = frame["augmented_model"].map(_normalize_model)
    frame["_cell"] = list(zip(frame["view"], frame["timing"], frame["_baseline"], frame["_augmented"]))
    required = _required_comparison_cells()
    generic_hashes: dict[str, dict[tuple[str, str], str]] = {}
    for scenario in PRIMARY_SCENARIOS:
        selected = _select_primary_rows(frame, scenario)
        generic = selected[
            selected["comparison"].astype(str).str.strip().str.lower()
            == "all_primary_model_families"
        ].copy()
        generic["_view_timing"] = list(zip(generic["view"], generic["timing"]))
        expected_view_timings = {(view, timing) for view in VIEWS for timing in TIMINGS}
        if set(generic["_view_timing"]) != expected_view_timings or len(generic) != len(
            expected_view_timings
        ):
            raise VerificationFailure(
                f"matched manifest must contain every all-model view/timing cell for {scenario}"
            )
        if not np.all(_integers(generic, "n", scenario) == EXPECTED_PRIMARY_N):
            raise VerificationFailure(f"matched manifest {scenario} rows are not n=384")
        if not np.all(_integers(generic, "pCR_positive", scenario) == EXPECTED_PRIMARY_POSITIVE):
            raise VerificationFailure(f"matched manifest {scenario} rows are not pCR+=113")
        if not np.all(_integers(generic, "missingness_exclusions", scenario) == 0):
            raise VerificationFailure(f"{scenario} unexpectedly excludes patients")
        hashes = _strict_strings(generic, "patient_set_sha256", scenario)
        if not hashes.map(lambda value: bool(SHA256_PATTERN.fullmatch(value))).all():
            raise VerificationFailure(f"{scenario} contains an invalid patient-set SHA-256")
        generic_hashes[scenario] = dict(zip(generic["_view_timing"], hashes))
    for view_timing in generic_hashes[PRIMARY_SCENARIOS[0]]:
        if (
            generic_hashes[PRIMARY_SCENARIOS[0]][view_timing]
            != generic_hashes[PRIMARY_SCENARIOS[1]][view_timing]
        ):
            raise VerificationFailure(
                "complete-case and train-median-indicator patient populations differ"
            )

    complete_case = _select_primary_rows(frame, "complete_case")
    paired = complete_case[complete_case["_cell"].isin(required)].copy()
    if set(paired["_cell"]) != required or len(paired) != len(required):
        raise VerificationFailure(f"matched manifest lacks one of {len(required)} paired comparison cells")
    if not np.all(_integers(paired, "n", "complete_case paired") == EXPECTED_PRIMARY_N):
        raise VerificationFailure("paired comparison manifest rows are not n=384")
    if not np.all(
        _integers(paired, "pCR_positive", "complete_case paired")
        == EXPECTED_PRIMARY_POSITIVE
    ):
        raise VerificationFailure("paired comparison manifest rows are not pCR+=113")
    paired_hashes = _strict_strings(paired, "patient_set_sha256", "complete_case paired")
    for (_, row), patient_hash in zip(paired.iterrows(), paired_hashes):
        expected_hash = generic_hashes["complete_case"][(str(row["view"]), str(row["timing"]))]
        if patient_hash != expected_hash:
            raise VerificationFailure("a paired comparison does not use its all-model patient set")
    return {
        "paired_cells_checked": len(required),
        "imputation_identity_cells": len(generic_hashes["complete_case"]),
        "population_n": EXPECTED_PRIMARY_N,
        "identity_proof": "matching patient_set_sha256",
    }


def _metric_columns(frame: pd.DataFrame) -> list[str]:
    tokens = ("auroc", "auprc", "brier", "balanced_accuracy", "spearman", "r2")
    result: list[str] = []
    for column in frame.columns:
        normalized = str(column).lower()
        if any(token in normalized for token in tokens) and not any(
            word in normalized for word in ("definition", "label", "name")
        ):
            result.append(str(column))
    return result


def _check_finite_aggregate_metrics(root: Path) -> dict[str, Any]:
    filenames = (
        "profile_oof_metrics.csv",
        "redundancy_metrics.csv",
        "residualization_metrics.csv",
        "family_ablation_metrics.csv",
        "lr_vs_svm.csv",
        "mri_reference_metrics.csv",
        "mri_reference_profile_metrics.csv",
        "mri_reference_traditional_pcr_comparison.csv",
        "mri_reference_traditional_profile_comparison.csv",
    )
    checked: dict[str, list[str]] = {}
    for filename in filenames:
        frame = _load_csv(root / "metrics" / filename)
        _require_nonempty(frame, filename)
        columns = _metric_columns(frame)
        if columns:
            for column in columns:
                _numeric(frame, column, filename)
        elif {"metric", "value"}.issubset(frame.columns):
            metric_rows = frame["metric"].astype(str).str.lower().map(
                lambda value: any(token in value for token in (
                    "auroc", "auprc", "brier", "balanced accuracy", "spearman", "r2"
                ))
            )
            if not metric_rows.any():
                raise VerificationFailure(f"{filename} contains no recognized key metrics")
            _numeric(frame.loc[metric_rows], "value", filename)
            columns = ["value"]
        else:
            raise VerificationFailure(f"{filename} contains no recognized finite key metric columns")
        checked[filename] = columns

    correlation = _load_csv(root / "metrics" / "feature_correlation_matrix.csv")
    _require_nonempty(correlation, "feature_correlation_matrix.csv")
    numeric_columns = list(correlation.columns[1:])
    if not numeric_columns:
        raise VerificationFailure("feature correlation matrix has no numeric matrix columns")
    for column in numeric_columns:
        values = _numeric(correlation, str(column), "feature_correlation_matrix.csv")
        if np.any(np.abs(values) > 1.0 + 1e-12):
            raise VerificationFailure("feature correlation values must lie in [-1,1]")
    return {"tables": checked, "correlation_columns": len(numeric_columns)}


def _check_scientific_table_coverage(root: Path) -> dict[str, int]:
    expected_view_timings = {(view, timing) for view in VIEWS for timing in TIMINGS}

    profile = _load_csv(root / "metrics" / "profile_oof_metrics.csv")
    required_profile_columns = {
        "protocol",
        "population",
        "view",
        "timing",
        "feature_set",
        "model_type",
        "target",
        "n",
    }
    if not required_profile_columns.issubset(profile.columns):
        raise VerificationFailure("profile metrics schema lacks required probe dimensions")
    profile = profile[
        (profile["protocol"].astype(str) == EXPECTED_PRIMARY_PROTOCOL)
        & (profile["population"].astype(str) == EXPECTED_PRIMARY_POPULATION)
    ].copy()
    profile["_model_type"] = profile["model_type"].map(_normalize_model_type)
    profile["_feature_set"] = profile["feature_set"].map(_normalize_model)
    profile["_target"] = profile["target"].astype(str).replace({"subtype_4class": "subtype"})
    observed_profile = set(
        zip(
            profile["view"].astype(str),
            profile["timing"].astype(str),
            profile["_feature_set"],
            profile["_model_type"],
            profile["_target"],
        )
    )
    expected_profile = {
        (view, timing, feature_set, model_type, target)
        for view, timing in expected_view_timings
        for feature_set in ("N", "FULL")
        for model_type in ("LR", "SVM")
        for target in ("HR", "HER2", "subtype")
    }
    if observed_profile != expected_profile or len(profile) != len(expected_profile):
        raise VerificationFailure("HR/HER2/subtype probes do not cover N/FULL, LR/SVM, and all timings")
    if not np.all(_integers(profile, "n", "profile probes") == EXPECTED_PRIMARY_N):
        raise VerificationFailure("primary profile probes are not all n=384")

    redundancy = _load_csv(root / "metrics" / "redundancy_metrics.csv")
    redundancy = redundancy[
        (redundancy["protocol"].astype(str) == EXPECTED_PRIMARY_PROTOCOL)
        & (redundancy["population"].astype(str) == EXPECTED_PRIMARY_POPULATION)
    ].copy()
    observed_redundancy = set(
        redundancy[["view", "timing"]].astype(str).itertuples(index=False, name=None)
    )
    if observed_redundancy != expected_view_timings or len(redundancy) != len(expected_view_timings):
        raise VerificationFailure("FTV redundancy table does not cover every view/timing")
    if not np.all(_integers(redundancy, "n", "redundancy metrics") == EXPECTED_PRIMARY_N):
        raise VerificationFailure("FTV redundancy rows are not all n=384")

    residual = _load_csv(root / "metrics" / "residualization_metrics.csv")
    residual = _select_primary_rows(residual, "complete_case")
    residual = residual[residual["model_type"].map(_normalize_model_type) == "LR"].copy()
    observed_residual = set(
        zip(
            residual["view"].astype(str),
            residual["timing"].astype(str),
            residual["model"].map(_normalize_model),
        )
    )
    expected_residual = {
        (view, timing, model)
        for view, timing in expected_view_timings
        for model in ("N_RES", "C+F+N_RES")
    }
    if observed_residual != expected_residual or len(residual) != len(expected_residual):
        raise VerificationFailure("residualization table lacks N_res/C+F+N_res view/timing cells")
    if not np.all(_integers(residual, "n", "residualization") == EXPECTED_PRIMARY_N) or not np.all(
        _integers(residual, "n_positive", "residualization") == EXPECTED_PRIMARY_POSITIVE
    ):
        raise VerificationFailure("residualization rows do not use matched n=384, pCR+=113")

    family = _load_csv(root / "metrics" / "family_ablation_metrics.csv")
    family = _select_primary_rows(family, "complete_case")
    family = family[family["model_type"].map(_normalize_model_type) == "LR"].copy()
    observed_family = set(
        zip(
            family["view"].astype(str),
            family["timing"].astype(str),
            family["model"].map(_normalize_model),
        )
    )
    expected_family = {
        (view, timing, model)
        for view, timing in expected_view_timings
        for model in ("C+F", "C+F+D", "C+F+S", "C+F+B")
    }
    if observed_family != expected_family or len(family) != len(expected_family):
        raise VerificationFailure("family ablation table lacks D/S/B and baseline view/timing cells")
    if not np.all(_integers(family, "n", "family ablation") == EXPECTED_PRIMARY_N) or not np.all(
        _integers(family, "n_positive", "family ablation") == EXPECTED_PRIMARY_POSITIVE
    ):
        raise VerificationFailure("family ablation rows do not use matched n=384, pCR+=113")

    lr_svm = _load_csv(root / "metrics" / "lr_vs_svm.csv")
    required_lr_svm = {
        "protocol",
        "population",
        "scenario",
        "view",
        "timing",
        "model",
        "logistic_auroc",
        "svm_auroc",
        "delta_svm_minus_lr",
    }
    if not required_lr_svm.issubset(lr_svm.columns):
        raise VerificationFailure("LR-vs-SVM table lacks the formal paired AUROC columns")
    lr_svm = _select_primary_rows(lr_svm, "complete_case")
    observed_lr_svm = set(
        zip(
            lr_svm["view"].astype(str),
            lr_svm["timing"].astype(str),
            lr_svm["model"].map(_normalize_model),
        )
    )
    expected_lr_svm = {
        (view, timing, _normalize_model(model))
        for view, timing in expected_view_timings
        for model in PRIMARY_MODELS
    }
    if observed_lr_svm != expected_lr_svm or len(lr_svm) != len(expected_lr_svm):
        raise VerificationFailure("LR-vs-SVM table lacks every primary model/view/timing cell")
    logistic = _numeric(lr_svm, "logistic_auroc", "lr_vs_svm")
    svm = _numeric(lr_svm, "svm_auroc", "lr_vs_svm")
    delta = _numeric(lr_svm, "delta_svm_minus_lr", "lr_vs_svm")
    if not np.allclose(delta, svm - logistic, rtol=0.0, atol=1e-12):
        raise VerificationFailure("LR-vs-SVM AUROC differences are arithmetically inconsistent")

    return {
        "profile_probe_cells": len(profile),
        "redundancy_cells": len(redundancy),
        "residualization_cells": len(residual),
        "family_ablation_cells": len(family),
        "lr_vs_svm_cells": len(lr_svm),
    }


def _not_pooled(value: object) -> bool:
    text = str(value).lower().replace("_", " ").replace("-", " ")
    return bool(
        re.search(r"\bnot\s+pooled\b", text)
        or re.search(r"\bnever\s+pooled\b", text)
        or re.search(r"\bwithout\s+(?:patient\s+)?pooling\b", text)
    )


def _check_mri_traditional_comparisons(root: Path) -> dict[str, int]:
    pcr = _load_csv(
        root / "metrics" / "mri_reference_traditional_pcr_comparison.csv",
        exact_columns=MRI_TRADITIONAL_COMPARISON_COLUMNS,
    )
    profile = _load_csv(
        root / "metrics" / "mri_reference_traditional_profile_comparison.csv",
        exact_columns=MRI_TRADITIONAL_COMPARISON_COLUMNS,
    )
    expected_pcr = {
        (timing, traditional, mri)
        for timing in TIMINGS
        for traditional, mri in (("N", "M"), ("C+N", "C+M"), ("C+FULL", "C+F+M"))
    }
    observed_pcr = set(
        pcr[["timing", "traditional_model", "mri_model"]].astype(str).itertuples(index=False, name=None)
    )
    if observed_pcr != expected_pcr or len(pcr) != len(expected_pcr):
        raise VerificationFailure("MRI/traditional pCR comparison must contain all 12 matched cells")
    if not pcr["task"].astype(str).eq("pCR").all() or not pcr["target"].astype(str).eq("pCR").all():
        raise VerificationFailure("MRI/traditional pCR table has an invalid task or target")
    if not pcr["view"].astype(str).eq("longitudinal").all():
        raise VerificationFailure("MRI/traditional pCR comparisons must use the prefix view")

    normalized_target = profile["target"].astype(str).replace({"subtype_4class": "subtype"})
    expected_profile = {
        (view, timing, target, model, "M")
        for view, timing in (
            *(("static", timing) for timing in TIMINGS),
            *(("longitudinal", timing) for timing in TIMINGS[1:]),
        )
        for target in ("HR", "HER2", "subtype")
        for model in ("N", "FULL")
    }
    observed_profile = set(
        zip(
            profile["view"].astype(str),
            profile["timing"].astype(str),
            normalized_target,
            profile["traditional_model"].astype(str),
            profile["mri_model"].astype(str),
        )
    )
    if observed_profile != expected_profile or len(profile) != len(expected_profile):
        raise VerificationFailure(
            "MRI/traditional profile comparison lacks a matched static/prefix target/model cell"
        )
    if not profile["task"].astype(str).eq("profile_probe").all():
        raise VerificationFailure("MRI/traditional profile table has an invalid task")

    for label, frame in (("MRI/traditional pCR", pcr), ("MRI/traditional profile", profile)):
        if not frame["population"].astype(str).eq("mri_matched_375").all():
            raise VerificationFailure(f"{label} uses an unexpected population")
        if not np.all(_integers(frame, "n", label) == EXPECTED_MRI_N):
            raise VerificationFailure(f"{label} rows are not exactly n=375")
        aggregation = _strict_strings(frame, "mri_aggregation", label)
        if not aggregation.map(_not_pooled).all():
            raise VerificationFailure(f"{label} pools MRI sensitivity cells")
        traditional = _numeric(frame, "traditional_auroc", label)
        mri = _numeric(frame, "mri_auroc", label)
        difference = _numeric(frame, "difference_mri_minus_traditional", label)
        if np.any((traditional < 0.0) | (traditional > 1.0)) or np.any((mri < 0.0) | (mri > 1.0)):
            raise VerificationFailure(f"{label} AUROC lies outside [0,1]")
        if not np.allclose(difference, mri - traditional, rtol=0.0, atol=1e-12):
            raise VerificationFailure(f"{label} AUROC differences are arithmetically inconsistent")
        _check_timing_values(frame, label)
    return {"pcr_cells": len(pcr), "profile_cells": len(profile)}


def _check_mri_reference(root: Path) -> dict[str, Any]:
    pcr = _load_csv(root / "metrics" / "mri_reference_metrics.csv")
    profile = _load_csv(root / "metrics" / "mri_reference_profile_metrics.csv")
    provenance = _load_json(root / "metrics" / "mri_reference_provenance.json")
    for label, frame in (("MRI pCR reference", pcr), ("MRI profile reference", profile)):
        _require_nonempty(frame, label)
        patients = _integers(frame, "n_patients_per_cell", label)
        if not np.all(patients == EXPECTED_MRI_N):
            raise VerificationFailure(f"{label} does not use n=375 per sensitivity cell")
        cells = _integers(frame, "n_sensitivity_cells", label)
        if np.any(cells < 1):
            raise VerificationFailure(f"{label} has no sensitivity cells")
        aggregation = _strict_strings(frame, "aggregation", label)
        if not aggregation.map(_not_pooled).all():
            raise VerificationFailure(f"{label} does not explicitly forbid pooled sensitivity cells")

    matched = provenance.get("matched_population", {})
    sensitivity = provenance.get("sensitivity_contract", {})
    privacy = provenance.get("privacy", {})
    if not isinstance(matched, Mapping) or int(matched.get("exact_mri_overlap", -1)) != EXPECTED_MRI_N:
        raise VerificationFailure("MRI provenance does not certify exact matched n=375")
    if not isinstance(sensitivity, Mapping):
        raise VerificationFailure("MRI provenance lacks a sensitivity contract")
    if sensitivity.get("duplicate_patients_pooled_across_cells") is not False:
        raise VerificationFailure("MRI provenance does not certify unpooled sensitivity cells")
    if not _not_pooled(sensitivity.get("aggregation", "")):
        raise VerificationFailure("MRI provenance aggregation is not explicitly unpooled")
    if not isinstance(privacy, Mapping) or privacy.get("outputs_are_aggregate_only") is not True:
        raise VerificationFailure("MRI provenance does not certify aggregate-only outputs")
    comparison_counts = _check_mri_traditional_comparisons(root)
    return {
        "matched_n": EXPECTED_MRI_N,
        "sensitivity_cells": [
            int(value)
            for value in sorted(set(_integers(pcr, "n_sensitivity_cells", "MRI pCR")))
        ],
        "pooled": False,
        "traditional_comparisons": comparison_counts,
    }


def _check_final_report(root: Path) -> dict[str, Any]:
    path = root / FINAL_REPORT
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise VerificationFailure(f"cannot read UTF-8 final report: {error}") from error
    chinese_characters = re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", text)
    if len(chinese_characters) < 100:
        raise VerificationFailure("final report is not a substantive Chinese-language report")
    required_terms = (
        "FTV",
        "LD",
        "SPH",
        "BPE",
        "NONFTV",
        "pCR",
        "Clinical",
        "residual",
        "HR",
        "HER2",
        "LR",
        "SVM",
        "MRI",
        "World Model",
    )
    missing = [term for term in required_terms if term.lower() not in text.lower()]
    if missing:
        raise VerificationFailure(f"final report omits required scientific answers/terms: {missing}")
    missing_questions = [
        number
        for number in range(1, 13)
        if not re.search(rf"^#{{2,4}}\s*{number}\.\s+", text, flags=re.MULTILINE)
    ]
    if missing_questions:
        raise VerificationFailure(f"final report lacks numbered answers: {missing_questions}")
    return {
        "chinese_characters": len(chinese_characters),
        "required_terms": len(required_terms),
        "numbered_answers": 12,
    }


def _run_git(repo: Path, arguments: Sequence[str], *, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if check and completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise VerificationFailure(f"git {' '.join(arguments)} failed: {message}")
    return completed.stdout.strip()


def _git_context(root: Path) -> tuple[Path, str]:
    repo_text = _run_git(root, ("rev-parse", "--show-toplevel"))
    repo = Path(repo_text).resolve()
    try:
        relative = root.resolve().relative_to(repo).as_posix()
    except ValueError as error:
        raise VerificationFailure("experiment root is outside its Git repository") from error
    return repo, relative or "."


def _git_files(repo: Path, relative: str) -> list[Path]:
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            relative,
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise VerificationFailure(
            f"git ls-files failed: {completed.stderr.decode('utf-8', errors='replace').strip()}"
        )
    names = [value for value in completed.stdout.decode("utf-8").split("\0") if value]
    return [(repo / name).resolve() for name in names]


def _is_private_candidate(path: Path, root: Path) -> bool:
    relative = path.relative_to(root).as_posix().lower()
    name = path.name.lower()
    if ".private" in name or "patient_level" in name or "patient-level" in name:
        return True
    if relative.startswith("predictions/") and name != ".gitkeep":
        return True
    return False


def _json_contains_direct_id(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            PREFIXED_SIX_DIGIT_ID.search(str(key))
            or _json_contains_direct_id(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_json_contains_direct_id(child) for child in value)
    if isinstance(value, str):
        return bool(PREFIXED_SIX_DIGIT_ID.search(value) or BARE_SIX_DIGIT_ID.search(value))
    if isinstance(value, int) and not isinstance(value, bool):
        return 100000 <= value <= 999999
    return False


def _check_privacy_and_gitignore(root: Path) -> dict[str, Any]:
    repo, relative = _git_context(root)
    included = _git_files(repo, relative)
    included_set = set(included)
    private_candidates = [path.resolve() for path in root.rglob("*") if path.is_file() and _is_private_candidate(path, root)]
    for path in private_candidates:
        rel_repo = path.relative_to(repo).as_posix()
        if path in included_set:
            raise VerificationFailure(f"private/patient-level artifact is tracked or unignored: {path.relative_to(root)}")
        completed = subprocess.run(
            ["git", "-C", str(repo), "check-ignore", "--no-index", "-q", "--", rel_repo],
            check=False,
        )
        if completed.returncode != 0:
            raise VerificationFailure(f"private/patient-level artifact is not gitignored: {path.relative_to(root)}")

    sentinels = (
        root / "features" / "verification.private.csv",
        root / "predictions" / "verification.private.csv",
        root / "logs" / "verification.private.log",
        root / "configs" / "verification_patient.private.csv",
    )
    for sentinel in sentinels:
        rel_repo = sentinel.relative_to(repo).as_posix()
        completed = subprocess.run(
            ["git", "-C", str(repo), "check-ignore", "--no-index", "-q", "--", rel_repo],
            check=False,
        )
        if completed.returncode != 0:
            raise VerificationFailure(f".gitignore does not cover private artifact pattern: {sentinel.relative_to(root)}")

    scanned = 0
    for path in included:
        if not _inside(path, root) or path.suffix.lower() not in {".csv", ".md", ".json"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise VerificationFailure(f"tracked aggregate is not UTF-8: {path.relative_to(root)}") from error
        prefixed_match = PREFIXED_SIX_DIGIT_ID.search(text)
        if prefixed_match:
            line = text.count("\n", 0, prefixed_match.start()) + 1
            raise VerificationFailure(
                f"direct six-digit patient identifier found in aggregate {path.relative_to(root)} line {line}"
            )
        if path.suffix.lower() == ".csv":
            header = pd.read_csv(path, nrows=0)
            forbidden = {"patient-id", "trial-id", "subject-id", "clinical-trial-subject-id"}
            normalized = {str(column).strip().lower().replace("_", "-") for column in header.columns}
            if forbidden & normalized:
                raise VerificationFailure(f"patient identifier column found in aggregate {path.relative_to(root)}")
            cells = pd.read_csv(path, dtype=str, keep_default_na=False)
            exact_six_digit = cells.apply(
                lambda column: column.str.strip().str.fullmatch(r"\d{6}").any()
            ).any()
            if bool(exact_six_digit):
                raise VerificationFailure(
                    f"exact six-digit cell found in aggregate {path.relative_to(root)}"
                )
        elif path.suffix.lower() == ".json":
            if _json_contains_direct_id(_load_json(path)):
                raise VerificationFailure(
                    f"direct six-digit value found in aggregate {path.relative_to(root)}"
                )
        else:
            bare_match = BARE_SIX_DIGIT_ID.search(text)
            if bare_match:
                line = text.count("\n", 0, bare_match.start()) + 1
                raise VerificationFailure(
                    f"direct six-digit value found in aggregate {path.relative_to(root)} line {line}"
                )
        scanned += 1
    return {
        "aggregate_text_files_scanned": scanned,
        "private_files_checked": len(private_candidates),
        "six_digit_ids_found": 0,
    }


def _check_no_raw_data(root: Path) -> dict[str, Any]:
    forbidden_suffixes = (
        ".xlsx",
        ".xls",
        ".xlsm",
        ".dcm",
        ".dicom",
        ".nii",
        ".nii.gz",
        ".nrrd",
        ".mha",
        ".mhd",
        ".npy",
        ".npz",
        ".pt",
        ".pth",
        ".h5",
        ".hdf5",
    )
    forbidden_directories = {"raw", "raw_mri", "raw-mri", "mri", "images", "dicom", "nifti"}
    raster_suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    violations: list[str] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part.lower() in forbidden_directories for part in relative.parts[:-1]):
            violations.append(relative.as_posix())
            continue
        if path.is_file() and any(path.name.lower().endswith(suffix) for suffix in forbidden_suffixes):
            violations.append(relative.as_posix())
        elif (
            path.is_file()
            and path.suffix.lower() in raster_suffixes
            and (not relative.parts or relative.parts[0] != "figures")
        ):
            violations.append(relative.as_posix())
    if violations:
        raise VerificationFailure(f"raw workbook/MRI artifacts found under experiment: {violations[:10]}")
    return {"raw_workbooks": 0, "raw_mri_files": 0}


def _delivery_path(root: Path) -> Path | None:
    candidates = (
        root / "reports" / "delivery_provenance.json",
        root / "metrics" / "delivery_provenance.json",
        root / "delivery_provenance.json",
    )
    existing = [path for path in candidates if path.is_file()]
    if len(existing) > 1:
        raise VerificationFailure("multiple delivery_provenance.json files create ambiguity")
    return existing[0] if existing else None


def _check_delivery(root: Path, allow_pending: bool) -> dict[str, Any]:
    path = _delivery_path(root)
    if path is None:
        if allow_pending:
            repo, _ = _git_context(root)
            actual_branch = _run_git(repo, ("branch", "--show-current"))
            if actual_branch != EXPECTED_BRANCH:
                raise VerificationFailure(f"pre-commit branch must be {EXPECTED_BRANCH}")
            return {
                "status": "PENDING_ALLOWED",
                "reason": "pre-commit verification",
                "branch": actual_branch,
            }
        raise VerificationFailure("post-delivery verification requires delivery_provenance.json")
    document = _load_json(path)
    branch = str(document.get("branch", "")).strip()
    sha = str(document.get("commit_sha", document.get("sha", ""))).strip()
    status = str(document.get("push_status", document.get("status", ""))).strip().upper()
    error = str(document.get("push_error", document.get("error", ""))).strip()
    pending = status in {"", "PENDING", "NOT_ATTEMPTED"}
    if allow_pending and pending:
        repo, _ = _git_context(root)
        actual_branch = _run_git(repo, ("branch", "--show-current"))
        if actual_branch != EXPECTED_BRANCH or (branch and branch != EXPECTED_BRANCH):
            raise VerificationFailure(f"pre-commit branch must be {EXPECTED_BRANCH}")
        return {"status": "PENDING_ALLOWED", "branch": actual_branch}
    if branch != EXPECTED_BRANCH:
        raise VerificationFailure(f"delivery branch must be {EXPECTED_BRANCH}")
    if not GIT_SHA_PATTERN.fullmatch(sha):
        raise VerificationFailure("delivery provenance requires a full 40-character commit SHA")
    success_statuses = {"PUSHED", "SUCCESS", "PUSH_SUCCESS", "GITHUB_PUSH_SUCCESS"}
    failure_statuses = {"GITHUB_PUSH_FAILED", "PUSH_FAILED", "FAILED"}
    if status not in success_statuses | failure_statuses:
        raise VerificationFailure("delivery provenance requires a final push status")
    if status in failure_statuses and not error:
        raise VerificationFailure("failed push status requires the real push error")
    repo, _ = _git_context(root)
    actual_branch = _run_git(repo, ("branch", "--show-current"))
    actual_sha = _run_git(repo, ("rev-parse", "HEAD"))
    if actual_branch != branch:
        raise VerificationFailure("recorded delivery branch differs from the checked-out branch")
    if actual_sha.lower() != sha.lower():
        raise VerificationFailure("recorded delivery SHA differs from repository HEAD")
    return {
        "status": status,
        "branch": branch,
        "commit_sha": sha.lower(),
        "push_error_recorded": bool(error),
    }


def _sanitize_error(error: BaseException) -> str:
    text = str(error)
    text = PREFIXED_SIX_DIGIT_ID.sub("<REDACTED_PREFIXED_ID>", text)
    return BARE_SIX_DIGIT_ID.sub("<REDACTED_SIX_DIGIT_ID>", text)


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_experiment(
    experiment_root: str | Path = EXPERIMENT_ROOT,
    *,
    config_path: str | Path | None = None,
    allow_pending_delivery: bool = False,
) -> dict[str, Any]:
    """Run every verification check and atomically write the aggregate result."""

    root = Path(experiment_root).expanduser().resolve()
    config_candidate = Path(config_path) if config_path is not None else None
    try:
        resolved_config = _resolve_config_path(root, config_candidate)
        config = _load_json(resolved_config)
    except Exception as error:
        resolved_config = config_candidate.resolve() if config_candidate else root / "configs" / "experiment.json"
        config = {}
        initial_error: BaseException | None = error
    else:
        initial_error = None

    checks: list[dict[str, Any]] = []

    def run(name: str, function: Callable[[], Mapping[str, Any]]) -> None:
        try:
            details = dict(function())
        except Exception as error:
            checks.append({"name": name, "status": "failed", "error": _sanitize_error(error)})
        else:
            checks.append({"name": name, "status": "passed", "details": details})

    if initial_error is not None:
        checks.append({"name": "config", "status": "failed", "error": _sanitize_error(initial_error)})
    else:
        checks.append(
            {
                "name": "config",
                "status": "passed",
                "details": {"path": str(resolved_config), "sha256": _sha256(resolved_config)},
            }
        )
    run("required_outputs", lambda: _check_required_outputs(root))
    if initial_error is None:
        run("source_hashes_and_schemas", lambda: _check_sources(root, resolved_config, config))
        run("run_summary", lambda: _check_run_summary(root, config))
    run("timing_contract", lambda: _check_timing_contract(root))
    run("primary_metrics", lambda: _check_primary_metrics(root))
    run("static_longitudinal_consistency", lambda: _check_filtered_view_tables(root))
    if initial_error is None:
        run("paired_bootstrap", lambda: _check_incremental_effects(root, config))
    run("population_identity", lambda: _check_population_identity(root))
    run("finite_aggregate_metrics", lambda: _check_finite_aggregate_metrics(root))
    run("scientific_table_coverage", lambda: _check_scientific_table_coverage(root))
    run("mri_reference", lambda: _check_mri_reference(root))
    run("final_chinese_report", lambda: _check_final_report(root))
    run("privacy_and_gitignore", lambda: _check_privacy_and_gitignore(root))
    run("no_raw_data", lambda: _check_no_raw_data(root))
    run("delivery", lambda: _check_delivery(root, allow_pending_delivery))

    failed = [check["name"] for check in checks if check["status"] != "passed"]
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "classical_dce_phenotype_complementarity",
        "status": "passed" if not failed else "failed",
        "allow_pending_delivery": bool(allow_pending_delivery),
        "checks": checks,
        "failed_checks": failed,
    }
    output = root / "metrics" / "verification.json"
    _atomic_json(output, result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--allow-pending-delivery",
        action="store_true",
        help="Allow missing/PENDING delivery metadata during the pre-commit verification pass.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = verify_experiment(
        args.experiment_root,
        config_path=args.config,
        allow_pending_delivery=args.allow_pending_delivery,
    )
    output = Path(args.experiment_root).expanduser().resolve() / "metrics" / "verification.json"
    summary = {
        "status": result["status"],
        "verification": str(output),
        "failed_checks": result["failed_checks"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

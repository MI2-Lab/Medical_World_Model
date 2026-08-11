#!/usr/bin/env python3
"""Generate the aggregate-only MRI–clinical complementarity final report.

Only public aggregate metrics, the frozen audit configuration, the clinical
feature inventory, and optional delivery provenance are read.  Prediction,
feature, bootstrap-draw, and other patient-level artifacts are never opened.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "feature/mri-clinical-complementarity-audit"
EXPECTED_SEEDS = (2026, 3026)
EXPECTED_ARMS = ("LOCAL0", "LOCAL3")
EXPECTED_CELLS = {(seed, arm) for seed in EXPECTED_SEEDS for arm in EXPECTED_ARMS}
TIMINGS = ("T0", "T1", "T2", "T3")
PROFILE_TARGETS = ("HR", "HER2", "subtype_4class")
PROFILE_VIEWS = (
    "T0",
    "T1",
    "T2",
    "T3",
    "long_T0_T1",
    "long_T0_T2",
    "long_T0_T3",
)
PRIMARY_PCR_MODELS = (
    "C",
    "M",
    "C+M",
    "F",
    "C+F",
    "C+F+M",
    "M_residual",
    "C+F+M_residual",
    "C+M_error_correction",
)
FULL_PCR_MODELS = ("C", "M", "C+M", "C+M_error_correction")
SUBGROUPS = ("HR+/HER2-", "HR-/HER2-", "HER2+")
SUBGROUP_MODELS = ("M", "remaining_clinical", "remaining_clinical+M")
CLINICAL_CONTRACTS = (
    "C1_hr_her2",
    "C_condition_without_treatment",
    "C_condition_with_treatment",
    "C2_full_without_treatment",
    "C2_full_with_treatment",
)

PROFILE_COLUMNS = (
    "seed",
    "arm",
    "view",
    "target",
    "n",
    "auroc",
    "auprc",
    "balanced_accuracy",
)
PCR_COLUMNS = (
    "population",
    "seed",
    "arm",
    "timing",
    "model",
    "clinical_contract",
    "n",
    "auroc",
    "auprc",
    "brier",
)
BOOTSTRAP_COLUMNS = (
    "population",
    "seed",
    "arm",
    "timing",
    "comparison",
    "metric",
    "improvement",
    "ci_lower",
    "ci_upper",
    "confidence_level",
    "n_patients",
    "n_bootstrap",
    "n_valid_bootstrap",
    "bootstrap_unit",
    "orientation",
)
CLINICAL_COLUMNS = (
    "population",
    "clinical_contract",
    "n",
    "n_positive",
    "auroc",
    "auprc",
    "balanced_accuracy",
    "brier",
)
CLINICAL_RESIDUAL_COLUMNS = (
    "population",
    "seed",
    "arm",
    "timing",
    "n",
    "r2",
    "pearson",
    "spearman",
    "delta_auroc",
    "delta_auprc",
    "brier_improvement",
)
SUBGROUP_COLUMNS = (
    "seed",
    "arm",
    "timing",
    "subgroup",
    "model",
    "n",
    "n_positive",
    "auroc",
    "auprc",
    "brier",
)
COHORT_COLUMNS = (
    "population",
    "n",
    "pcr_positive",
    "pcr_prevalence",
    "hr_positive",
    "her2_positive",
    "mp1",
    "age_missing",
    "race_missing",
    "menopause_missing",
)

FORBIDDEN_ROW_LEVEL_COLUMNS = {
    "patient_id",
    "clinical_patient_id",
    "raw_patient_id",
    "y_true",
    "predicted_probability",
    "probability",
    "prediction",
}

SUMMARY_ARTIFACTS = {
    "profile_metrics": "profile_oof_metrics.csv",
    "pcr_metrics": "pcr_oof_metrics.csv",
    "bootstrap": "bootstrap_ci.csv",
    "clinical_metrics": "clinical_baseline_metrics.csv",
    "clinical_residual_metrics": "clinical_residual_metrics.csv",
    "subgroup_metrics": "subgroup_metrics.csv",
    "cohort_summary": "cohort_summary.csv",
}

CLASSIFICATION_LABELS = {
    "A": "MRI COMPLEMENTARY SIGNAL SUPPORTED",
    "B": "MRI SIGNAL MOSTLY FTV-REDUNDANT",
    "C": "MRI MOSTLY CLINICAL-REDUNDANT",
    "D": "CURRENT MRI STATE UNDERUTILIZES PHENOTYPE",
}


@dataclass(frozen=True)
class ReportInputs:
    root: Path
    profile: pd.DataFrame
    pcr: pd.DataFrame
    bootstrap: pd.DataFrame
    clinical: pd.DataFrame
    clinical_residual: pd.DataFrame
    subgroup: pd.DataFrame
    cohort: pd.DataFrame
    run_summary: dict[str, Any]
    config: dict[str, Any]
    inventory_sha256: str
    delivery: dict[str, str]
    delivery_present: bool


@dataclass(frozen=True)
class CellSummary:
    mean: float
    minimum: float
    maximum: float
    count: int


@dataclass(frozen=True)
class Classification:
    code: str
    label: str
    profile_material: bool
    mri_pcr_material: bool
    clinical_incremental_timings: tuple[str, ...]
    beyond_ftv_timings: tuple[str, ...]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=EXPERIMENT_ROOT,
        help="Experiment directory containing metrics/, configs/, and reports/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and build the report in memory without writing it.",
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid {label} JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _header(path: Path, *, label: str) -> tuple[str, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"missing aggregate {label}: {path}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        try:
            values = tuple(next(reader))
        except StopIteration as error:
            raise ValueError(f"empty aggregate {label}: {path}") from error
    if not values or any(not str(value).strip() for value in values):
        raise ValueError(f"{label} has an empty header field")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} has duplicate header fields")
    forbidden = sorted(
        {str(value).strip().lower() for value in values}.intersection(
            FORBIDDEN_ROW_LEVEL_COLUMNS
        )
    )
    if forbidden:
        raise ValueError(
            f"{label} must be aggregate-only; row-level fields are forbidden: {forbidden}"
        )
    return values


def _read_aggregate(path: Path, *, required: Sequence[str], label: str) -> pd.DataFrame:
    header = _header(path, label=label)
    missing = [column for column in required if column not in header]
    if missing:
        raise ValueError(
            f"{label} schema is missing {missing}; observed={list(header)}"
        )
    frame = pd.read_csv(path, usecols=list(required))
    if frame.empty:
        raise ValueError(f"{label} has no aggregate rows")
    return frame.loc[:, list(required)].copy()


def _text(frame: pd.DataFrame, columns: Iterable[str], *, label: str) -> None:
    for column in columns:
        if frame[column].isna().any():
            raise ValueError(f"{label}.{column} contains missing values")
        values = frame[column].astype(str).str.strip()
        if values.eq("").any():
            raise ValueError(f"{label}.{column} contains empty values")
        frame[column] = values


def _numeric(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> None:
    for column in columns:
        try:
            values = pd.to_numeric(frame[column], errors="raise").astype(float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label}.{column} must be numeric") from error
        array = values.to_numpy(dtype=float)
        if not np.isfinite(array).all():
            raise ValueError(f"{label}.{column} contains NaN or infinity")
        if minimum is not None and np.any(array < minimum):
            raise ValueError(f"{label}.{column} contains values below {minimum}")
        if maximum is not None and np.any(array > maximum):
            raise ValueError(f"{label}.{column} contains values above {maximum}")
        frame[column] = values


def _integers(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    label: str,
    minimum: int = 0,
) -> None:
    _numeric(frame, columns, label=label)
    for column in columns:
        array = frame[column].to_numpy(dtype=float)
        if not np.equal(array, np.floor(array)).all():
            raise ValueError(f"{label}.{column} must contain integers")
        values = array.astype(np.int64)
        if np.any(values < minimum):
            raise ValueError(f"{label}.{column} contains values below {minimum}")
        frame[column] = values


def _unique(frame: pd.DataFrame, keys: Sequence[str], *, label: str) -> None:
    duplicated = frame.duplicated(list(keys), keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, list(keys)].head(3).to_dict("records")
        raise ValueError(f"{label} has duplicate aggregate cells: {examples}")


def _require_values(
    frame: pd.DataFrame,
    column: str,
    required: Sequence[str],
    *,
    label: str,
) -> None:
    observed = set(frame[column].astype(str))
    missing = [value for value in required if value not in observed]
    if missing:
        raise ValueError(f"{label}.{column} is missing required values: {missing}")


def _require_four_cells(
    frame: pd.DataFrame, group_columns: Sequence[str], *, label: str
) -> None:
    for key, group in frame.groupby(list(group_columns), sort=True, dropna=False):
        observed = {
            (int(seed), str(arm))
            for seed, arm in zip(group["seed"], group["arm"], strict=True)
        }
        if len(group) != 4 or observed != EXPECTED_CELLS:
            raise ValueError(
                f"{label} group {key!r} does not contain the exact four seed×arm cells; "
                f"observed={sorted(observed)}"
            )


def _validate_profile(frame: pd.DataFrame) -> pd.DataFrame:
    label = "profile_oof_metrics"
    _text(frame, ("arm", "view", "target"), label=label)
    _integers(frame, ("seed", "n"), label=label)
    _numeric(
        frame,
        ("auroc", "auprc", "balanced_accuracy"),
        label=label,
        minimum=0.0,
        maximum=1.0,
    )
    _unique(frame, ("seed", "arm", "view", "target"), label=label)
    _require_values(frame, "target", PROFILE_TARGETS, label=label)
    _require_values(frame, "view", PROFILE_VIEWS, label=label)
    _require_four_cells(frame, ("view", "target"), label=label)
    if not frame["n"].eq(808).all():
        raise ValueError("profile_oof_metrics must use the full 808-patient cohort")
    return frame


def _validate_pcr(frame: pd.DataFrame) -> pd.DataFrame:
    label = "pcr_oof_metrics"
    _text(
        frame,
        ("population", "arm", "timing", "model", "clinical_contract"),
        label=label,
    )
    _integers(frame, ("seed", "n"), label=label)
    _numeric(
        frame,
        ("auroc", "auprc", "brier"),
        label=label,
        minimum=0.0,
        maximum=1.0,
    )
    _unique(
        frame,
        ("population", "seed", "arm", "timing", "model"),
        label=label,
    )
    _require_values(frame, "timing", TIMINGS, label=label)
    for population, models, expected_n in (
        ("full_808", FULL_PCR_MODELS, 808),
        ("ftv_complete_375", PRIMARY_PCR_MODELS, 375),
    ):
        selected = frame.loc[frame["population"].eq(population)]
        if selected.empty:
            raise ValueError(f"{label} has no {population} rows")
        _require_values(selected, "model", models, label=f"{label}/{population}")
        if not selected["n"].eq(expected_n).all():
            raise ValueError(f"{label}/{population} has incorrect OOF patient counts")
    _require_four_cells(frame, ("population", "timing", "model"), label=label)
    return frame


def _validate_bootstrap(frame: pd.DataFrame) -> pd.DataFrame:
    label = "bootstrap_ci"
    _text(
        frame,
        (
            "population",
            "arm",
            "timing",
            "comparison",
            "metric",
            "bootstrap_unit",
            "orientation",
        ),
        label=label,
    )
    frame["metric"] = frame["metric"].str.lower()
    _integers(
        frame,
        ("seed", "n_patients", "n_bootstrap", "n_valid_bootstrap"),
        label=label,
    )
    _numeric(
        frame,
        ("improvement", "ci_lower", "ci_upper"),
        label=label,
        minimum=-1.0,
        maximum=1.0,
    )
    _numeric(
        frame,
        ("confidence_level",),
        label=label,
        minimum=0.0,
        maximum=1.0,
    )
    if not np.allclose(frame["confidence_level"], 0.95, rtol=0.0, atol=1e-12):
        raise ValueError("bootstrap_ci must contain paired 95% intervals")
    if np.any(frame["ci_lower"] > frame["ci_upper"]):
        raise ValueError("bootstrap_ci has ci_lower greater than ci_upper")
    if not frame["bootstrap_unit"].eq("patient_within_outer_fold").all():
        raise ValueError("bootstrap_ci unit must be patient_within_outer_fold")
    _unique(
        frame,
        ("population", "seed", "arm", "timing", "comparison", "metric"),
        label=label,
    )
    _require_values(frame, "timing", TIMINGS, label=label)
    _require_values(frame, "metric", ("auroc", "auprc", "brier"), label=label)
    required_comparisons = (
        ("full_808", "C+M_vs_C"),
        ("ftv_complete_375", "C+M_vs_C"),
        ("ftv_complete_375", "C+F+M_vs_C+F"),
    )
    for population, comparison in required_comparisons:
        if frame.loc[
            frame["population"].eq(population) & frame["comparison"].eq(comparison)
        ].empty:
            raise ValueError(
                f"bootstrap_ci lacks population={population}, comparison={comparison}"
            )
    _require_four_cells(
        frame, ("population", "comparison", "metric", "timing"), label=label
    )
    return frame


def _validate_clinical(frame: pd.DataFrame) -> pd.DataFrame:
    label = "clinical_baseline_metrics"
    _text(frame, ("population", "clinical_contract"), label=label)
    _integers(frame, ("n", "n_positive"), label=label)
    _numeric(
        frame,
        ("auroc", "auprc", "balanced_accuracy", "brier"),
        label=label,
        minimum=0.0,
        maximum=1.0,
    )
    _unique(frame, ("population", "clinical_contract"), label=label)
    _require_values(frame, "clinical_contract", CLINICAL_CONTRACTS, label=label)
    for population, expected_n in (("full_808", 808), ("ftv_complete_375", 375)):
        selected = frame.loc[frame["population"].eq(population)]
        if (
            len(selected) != len(CLINICAL_CONTRACTS)
            or not selected["n"].eq(expected_n).all()
        ):
            raise ValueError(f"{label}/{population} coverage is incomplete")
    return frame


def _validate_clinical_residual(frame: pd.DataFrame) -> pd.DataFrame:
    label = "clinical_residual_metrics"
    _text(frame, ("population", "arm", "timing"), label=label)
    _integers(frame, ("seed", "n"), label=label)
    # R² is finite but intentionally not lower-bounded: a valid held-out R² may
    # be arbitrarily negative when the residual probe is worse than the mean.
    _numeric(frame, ("r2",), label=label)
    _numeric(
        frame,
        (
            "pearson",
            "spearman",
            "delta_auroc",
            "delta_auprc",
            "brier_improvement",
        ),
        label=label,
        minimum=-1.0,
        maximum=1.0,
    )
    _unique(frame, ("population", "seed", "arm", "timing"), label=label)
    _require_values(frame, "timing", TIMINGS, label=label)
    _require_four_cells(frame, ("population", "timing"), label=label)
    for population, expected_n in (("full_808", 808), ("ftv_complete_375", 375)):
        selected = frame.loc[frame["population"].eq(population)]
        if selected.empty or not selected["n"].eq(expected_n).all():
            raise ValueError(f"{label}/{population} coverage is incomplete")
    return frame


def _validate_subgroup(frame: pd.DataFrame) -> pd.DataFrame:
    label = "subgroup_metrics"
    _text(frame, ("arm", "timing", "subgroup", "model"), label=label)
    _integers(frame, ("seed", "n", "n_positive"), label=label)
    _numeric(
        frame,
        ("auroc", "auprc", "brier"),
        label=label,
        minimum=0.0,
        maximum=1.0,
    )
    if np.any(frame["n_positive"] > frame["n"]):
        raise ValueError("subgroup_metrics.n_positive exceeds n")
    _unique(
        frame,
        ("seed", "arm", "timing", "subgroup", "model"),
        label=label,
    )
    _require_values(frame, "timing", TIMINGS, label=label)
    _require_values(frame, "subgroup", SUBGROUPS, label=label)
    _require_values(frame, "model", SUBGROUP_MODELS, label=label)
    _require_four_cells(frame, ("subgroup", "timing", "model"), label=label)
    return frame


def _validate_cohort(frame: pd.DataFrame) -> pd.DataFrame:
    label = "cohort_summary"
    _text(frame, ("population",), label=label)
    _integers(
        frame,
        (
            "n",
            "pcr_positive",
            "hr_positive",
            "her2_positive",
            "mp1",
            "age_missing",
            "race_missing",
            "menopause_missing",
        ),
        label=label,
    )
    _numeric(
        frame,
        ("pcr_prevalence",),
        label=label,
        minimum=0.0,
        maximum=1.0,
    )
    _unique(frame, ("population",), label=label)
    _require_values(
        frame,
        "population",
        ("full_808", "ftv_complete_375", "ftv_unavailable_433"),
        label=label,
    )
    expected = {"full_808": 808, "ftv_complete_375": 375, "ftv_unavailable_433": 433}
    for population, count in expected.items():
        observed = int(frame.loc[frame["population"].eq(population), "n"].iloc[0])
        if observed != count:
            raise ValueError(f"{label}/{population} n={observed}, expected {count}")
    return frame


def _verify_run_summary(
    summary: Mapping[str, Any], metrics_dir: Path, config: Mapping[str, Any]
) -> None:
    required = {
        "experiment",
        "branch",
        "parent_commit",
        "evidence_status",
        "n_profile_patients",
        "n_primary_ftv_patients",
        "formal_bootstrap_replicates",
        "artifacts",
    }
    missing = sorted(required.difference(summary))
    if missing:
        raise ValueError(f"run_summary.json is missing fields: {missing}")
    if summary["experiment"] != "mri_clinical_complementarity_audit":
        raise ValueError("run_summary.json belongs to another experiment")
    if summary["branch"] != EXPECTED_BRANCH:
        raise ValueError(f"unexpected run-summary branch: {summary['branch']!r}")
    if str(summary["parent_commit"]) != str(config.get("parent_commit")):
        raise ValueError("run summary and audit config disagree on parent commit")
    if int(summary["n_profile_patients"]) != 808:
        raise ValueError("run_summary profile cohort is not 808")
    if int(summary["n_primary_ftv_patients"]) != 375:
        raise ValueError("run_summary primary FTV cohort is not 375")
    artifacts = summary["artifacts"]
    if not isinstance(artifacts, Mapping):
        raise ValueError("run_summary.artifacts must be an object")
    for key, filename in SUMMARY_ARTIFACTS.items():
        entry = artifacts.get(key)
        if not isinstance(entry, Mapping):
            raise ValueError(f"run_summary lacks aggregate artifact {key!r}")
        path = metrics_dir / filename
        if entry.get("path") != f"metrics/{filename}":
            raise ValueError(f"run_summary path drifted for {key!r}")
        if entry.get("sha256") != _sha256(path):
            raise ValueError(f"run_summary hash mismatch for aggregate {filename}")
        if int(entry.get("size_bytes", -1)) != path.stat().st_size:
            raise ValueError(f"run_summary size mismatch for aggregate {filename}")


def _git_branch(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None


def _delivery(path: Path, *, branch: str) -> tuple[dict[str, str], bool]:
    if not path.exists():
        return (
            {
                "branch": branch,
                "commit_sha": "PENDING",
                "push_status": "PENDING",
                "remote": "PENDING",
            },
            False,
        )
    payload = _read_json(path, label="delivery provenance")

    def first(*keys: str, default: str = "PENDING") -> str:
        for key in keys:
            value = payload.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return default

    output = {
        "branch": first("branch", default=branch),
        "commit_sha": first("commit_sha", "commit", "head_sha"),
        "push_status": first("push_status", "status"),
        "remote": first("remote", "remote_name", "remote_url"),
    }
    if output["branch"] != branch:
        raise ValueError("delivery provenance branch disagrees with the audit branch")
    return output, True


def load_inputs(experiment_root: Path) -> ReportInputs:
    root = experiment_root.expanduser().resolve()
    metrics = root / "metrics"
    reports = root / "reports"
    if not metrics.is_dir() or not reports.is_dir():
        raise FileNotFoundError("experiment root must contain metrics/ and reports/")

    paths = {filename: metrics / filename for filename in SUMMARY_ARTIFACTS.values()}
    profile = _validate_profile(
        _read_aggregate(
            paths["profile_oof_metrics.csv"],
            required=PROFILE_COLUMNS,
            label="profile_oof_metrics",
        )
    )
    pcr = _validate_pcr(
        _read_aggregate(
            paths["pcr_oof_metrics.csv"],
            required=PCR_COLUMNS,
            label="pcr_oof_metrics",
        )
    )
    bootstrap = _validate_bootstrap(
        _read_aggregate(
            paths["bootstrap_ci.csv"],
            required=BOOTSTRAP_COLUMNS,
            label="bootstrap_ci",
        )
    )
    clinical = _validate_clinical(
        _read_aggregate(
            paths["clinical_baseline_metrics.csv"],
            required=CLINICAL_COLUMNS,
            label="clinical_baseline_metrics",
        )
    )
    clinical_residual = _validate_clinical_residual(
        _read_aggregate(
            paths["clinical_residual_metrics.csv"],
            required=CLINICAL_RESIDUAL_COLUMNS,
            label="clinical_residual_metrics",
        )
    )
    subgroup = _validate_subgroup(
        _read_aggregate(
            paths["subgroup_metrics.csv"],
            required=SUBGROUP_COLUMNS,
            label="subgroup_metrics",
        )
    )
    cohort = _validate_cohort(
        _read_aggregate(
            paths["cohort_summary.csv"],
            required=COHORT_COLUMNS,
            label="cohort_summary",
        )
    )

    config = _read_json(root / "configs" / "audit.json", label="audit config")
    summary = _read_json(metrics / "run_summary.json", label="run summary")
    _verify_run_summary(summary, metrics, config)
    if set(config.get("local_cells", {}).get("seed_bases", ())) != set(EXPECTED_SEEDS):
        raise ValueError("audit config does not contain the expected two model seeds")
    if set(config.get("local_cells", {}).get("arms", ())) != set(EXPECTED_ARMS):
        raise ValueError("audit config does not contain LOCAL0 and LOCAL3")

    inventory_path = reports / "clinical_feature_inventory.md"
    if not inventory_path.is_file():
        raise FileNotFoundError(f"missing clinical feature inventory: {inventory_path}")
    inventory_text = inventory_path.read_text(encoding="utf-8")
    if not inventory_text.strip() or "future_response_state" not in inventory_text:
        raise ValueError(
            "clinical feature inventory is empty or lacks representation audit"
        )

    current_branch = _git_branch(root)
    branch = str(summary["branch"])
    if current_branch is not None and current_branch != branch:
        raise ValueError(
            f"current git branch {current_branch!r} differs from aggregate branch {branch!r}"
        )
    delivery, delivery_present = _delivery(
        reports / "delivery_provenance.json", branch=branch
    )
    return ReportInputs(
        root=root,
        profile=profile,
        pcr=pcr,
        bootstrap=bootstrap,
        clinical=clinical,
        clinical_residual=clinical_residual,
        subgroup=subgroup,
        cohort=cohort,
        run_summary=summary,
        config=config,
        inventory_sha256=_sha256(inventory_path),
        delivery=delivery,
        delivery_present=delivery_present,
    )


def _summary(frame: pd.DataFrame, column: str) -> CellSummary:
    values = frame[column].to_numpy(dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"cannot summarize empty/nonfinite column {column!r}")
    return CellSummary(
        mean=float(values.mean()),
        minimum=float(values.min()),
        maximum=float(values.max()),
        count=int(values.size),
    )


def _metric_cell(summary: CellSummary) -> str:
    return f"{summary.mean:.3f} ({summary.minimum:.3f}–{summary.maximum:.3f})"


def _delta_cell(summary: CellSummary) -> str:
    return f"{summary.mean:+.3f} " f"({summary.minimum:+.3f} to {summary.maximum:+.3f})"


def _timing_label(timing: str) -> str:
    return "T3 (late/pre-surgery)" if timing == "T3" else timing


def _view_label(view: str) -> str:
    labels = {
        "T0": "T0",
        "T1": "T1",
        "T2": "T2",
        "T3": "T3 (late/pre-surgery)",
        "long_T0_T1": "Long T0–T1",
        "long_T0_T2": "Long T0–T2",
        "long_T0_T3": "Long T0–T3 (includes late/pre-surgery)",
    }
    return labels.get(view, view)


def _target_label(target: str) -> str:
    return {
        "HR": "HR",
        "HER2": "HER2",
        "subtype_4class": "Four-class HR/HER2 subtype",
    }.get(target, target)


def _model_label(model: str) -> str:
    return model.replace("_residual", " residual").replace(
        "C+M_error_correction", "Clinical-error correction"
    )


def _escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    if not rows:
        raise ValueError("report table may not be empty")
    lines = [
        "| " + " | ".join(_escape(value) for value in headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        if len(row) != len(headers):
            raise ValueError("report table row width mismatch")
        lines.append("| " + " | ".join(_escape(value) for value in row) + " |")
    return "\n".join(lines)


def _profile_summary(data: ReportInputs) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target in PROFILE_TARGETS:
        for view in PROFILE_VIEWS:
            selected = data.profile.loc[
                data.profile["target"].eq(target) & data.profile["view"].eq(view)
            ]
            summary = _summary(selected, "auroc")
            if summary.count != 4:
                raise AssertionError("profile summary lost seed×arm cells")
            rows.append({"target": target, "view": view, **summary.__dict__})
    return pd.DataFrame(rows)


def _profile_peak(profile_summary: pd.DataFrame, target: str) -> pd.Series:
    selected = profile_summary.loc[profile_summary["target"].eq(target)]
    return selected.sort_values(
        ["mean", "view"], ascending=[False, True], kind="stable"
    ).iloc[0]


def _pcr_cell(
    data: ReportInputs, population: str, timing: str, model: str, metric: str = "auroc"
) -> CellSummary:
    selected = data.pcr.loc[
        data.pcr["population"].eq(population)
        & data.pcr["timing"].eq(timing)
        & data.pcr["model"].eq(model)
    ]
    summary = _summary(selected, metric)
    if summary.count != 4:
        raise AssertionError("pCR summary lost seed×arm cells")
    return summary


def _bootstrap_cell(
    data: ReportInputs,
    population: str,
    comparison: str,
    timing: str,
    metric: str,
) -> pd.DataFrame:
    selected = data.bootstrap.loc[
        data.bootstrap["population"].eq(population)
        & data.bootstrap["comparison"].eq(comparison)
        & data.bootstrap["timing"].eq(timing)
        & data.bootstrap["metric"].eq(metric)
    ].sort_values(["seed", "arm"], kind="stable")
    if len(selected) != 4:
        raise AssertionError("bootstrap summary lost seed×arm cells")
    return selected


def _ci_envelope(frame: pd.DataFrame) -> str:
    return f"{float(frame['ci_lower'].min()):+.3f} to {float(frame['ci_upper'].max()):+.3f}"


def _positive_ci_timings(
    data: ReportInputs, population: str, comparison: str
) -> tuple[str, ...]:
    output: list[str] = []
    for timing in TIMINGS:
        selected = _bootstrap_cell(data, population, comparison, timing, "auroc")
        if selected["improvement"].mean() > 0.0 and selected["ci_lower"].gt(0.0).all():
            output.append(timing)
    return tuple(output)


def _classify(data: ReportInputs, profile_summary: pd.DataFrame) -> Classification:
    profile_material = any(
        float(_profile_peak(profile_summary, target)["mean"]) >= 0.60
        and float(_profile_peak(profile_summary, target)["minimum"]) > 0.50
        for target in PROFILE_TARGETS
    )
    mri_summaries = [
        _pcr_cell(data, population, timing, "M")
        for population in ("full_808", "ftv_complete_375")
        for timing in TIMINGS
    ]
    mri_pcr_material = any(
        value.mean >= 0.60 and value.minimum > 0.50 for value in mri_summaries
    )
    clinical_timings = tuple(
        sorted(
            set(_positive_ci_timings(data, "full_808", "C+M_vs_C"))
            | set(_positive_ci_timings(data, "ftv_complete_375", "C+M_vs_C")),
            key=TIMINGS.index,
        )
    )
    beyond_timings = _positive_ci_timings(data, "ftv_complete_375", "C+F+M_vs_C+F")
    if clinical_timings and beyond_timings:
        code = "A"
    elif (mri_pcr_material or bool(clinical_timings)) and not beyond_timings:
        code = "B"
    elif profile_material and not clinical_timings:
        code = "C"
    else:
        code = "D"
    return Classification(
        code=code,
        label=CLASSIFICATION_LABELS[code],
        profile_material=profile_material,
        mri_pcr_material=mri_pcr_material,
        clinical_incremental_timings=clinical_timings,
        beyond_ftv_timings=beyond_timings,
    )


def _bootstrap_status(data: ReportInputs) -> tuple[int, int, bool]:
    observed_values = sorted(
        int(value) for value in data.bootstrap["n_bootstrap"].unique()
    )
    if len(observed_values) != 1:
        raise ValueError(f"bootstrap_ci mixes replicate counts: {observed_values}")
    observed = observed_values[0]
    configured = int(data.config.get("bootstrap", {}).get("replicates", 0))
    if configured <= 0:
        raise ValueError("audit config has no positive bootstrap replicate count")
    summary_value = int(data.run_summary["formal_bootstrap_replicates"])
    if summary_value != observed:
        raise ValueError("run summary and bootstrap CSV disagree on replicate count")
    return observed, configured, observed >= configured


def _subgroup_deltas(data: ReportInputs) -> pd.DataFrame:
    joint = data.subgroup.loc[
        data.subgroup["model"].eq("remaining_clinical+M"),
        ["seed", "arm", "timing", "subgroup", "auroc"],
    ].rename(columns={"auroc": "joint_auroc"})
    clinical = data.subgroup.loc[
        data.subgroup["model"].eq("remaining_clinical"),
        ["seed", "arm", "timing", "subgroup", "auroc"],
    ].rename(columns={"auroc": "clinical_auroc"})
    paired = joint.merge(
        clinical,
        on=["seed", "arm", "timing", "subgroup"],
        how="inner",
        validate="one_to_one",
    )
    paired["delta_auroc"] = paired["joint_auroc"] - paired["clinical_auroc"]
    if len(paired) != len(joint) or len(paired) != len(clinical):
        raise ValueError("subgroup clinical/joint cells do not align")
    return paired


def _residual_joint_delta(data: ReportInputs, timing: str) -> CellSummary:
    residual = data.pcr.loc[
        data.pcr["population"].eq("ftv_complete_375")
        & data.pcr["timing"].eq(timing)
        & data.pcr["model"].eq("C+F+M_residual"),
        ["seed", "arm", "auroc"],
    ].rename(columns={"auroc": "residual_auroc"})
    reference = data.pcr.loc[
        data.pcr["population"].eq("ftv_complete_375")
        & data.pcr["timing"].eq(timing)
        & data.pcr["model"].eq("C+F"),
        ["seed", "arm", "auroc"],
    ].rename(columns={"auroc": "reference_auroc"})
    paired = residual.merge(reference, on=["seed", "arm"], validate="one_to_one")
    paired["delta"] = paired["residual_auroc"] - paired["reference_auroc"]
    return _summary(paired, "delta")


def _classification_paragraph(
    classification: Classification,
    profile_summary: pd.DataFrame,
    data: ReportInputs,
) -> str:
    best_profile = profile_summary.sort_values(
        "mean", ascending=False, kind="stable"
    ).iloc[0]
    best_mri = max(
        (
            (population, timing, _pcr_cell(data, population, timing, "M"))
            for population in ("full_808", "ftv_complete_375")
            for timing in TIMINGS
        ),
        key=lambda item: item[2].mean,
    )
    if classification.code == "A":
        interpretation = (
            "Both clinical-incremental and beyond-FTV comparisons have at least one "
            "timing whose four cell-specific paired 95% CIs are entirely positive."
        )
    elif classification.code == "B":
        interpretation = (
            "MRI has descriptive pCR signal or clinical-incremental support, but no "
            "timing has four positive beyond-FTV cell-specific paired 95% CIs."
        )
    elif classification.code == "C":
        interpretation = (
            "At least one profile target reaches the material descriptive threshold, "
            "but clinical-incremental pCR value is not supported."
        )
    else:
        interpretation = (
            "Profile decodability and MRI-only pCR discrimination remain weak, and no "
            "clinical-incremental or beyond-FTV timing has four entirely positive paired CIs."
        )
    return (
        f"The evidence-based classification is **{classification.code}. "
        f"{classification.label}**. {interpretation} The best mean profile AUROC is "
        f"{float(best_profile['mean']):.3f} for {_target_label(str(best_profile['target']))} "
        f"at {_view_label(str(best_profile['view']))}; the best MRI-only pCR mean is "
        f"{best_mri[2].mean:.3f} in `{best_mri[0]}` at {_timing_label(best_mri[1])}. "
        "This A–D assignment is a structured descriptive synthesis, not a newly tuned "
        "significance threshold."
    )


def build_report(data: ReportInputs) -> tuple[str, Classification, dict[str, Any]]:
    profile_summary = _profile_summary(data)
    classification = _classify(data, profile_summary)
    observed_bootstrap, configured_bootstrap, bootstrap_complete = _bootstrap_status(
        data
    )
    evidence_prefix = "Diagnostic/exploratory two-seed evidence"
    if not bootstrap_complete:
        aggregate_status = (
            f"**PRELIMINARY/SMOKE:** {observed_bootstrap:,} paired bootstrap draws per cell "
            f"are present versus {configured_bootstrap:,} configured. CIs and classification "
            "are provisional until the formal aggregate replaces them."
        )
    else:
        aggregate_status = (
            f"**FORMAL AGGREGATE:** {observed_bootstrap:,} paired bootstrap draws per cell "
            "match the configured count. The scientific boundary remains diagnostic/exploratory "
            "because only two LOCAL model seeds are available."
        )

    lines: list[str] = [
        "# MRI–Clinical Complementarity Audit — Final Report",
        "",
        f"> {evidence_prefix}. {aggregate_status}",
        "",
        "## Executive conclusion",
        "",
        _classification_paragraph(classification, profile_summary, data),
        "",
        "Negative joint-model deltas are interpreted as **no supported incremental predictive "
        "value under this frozen linear protocol**, not as evidence that MRI is biologically "
        "harmful. Finite-sample estimation, high-dimensional concatenation, and regularization "
        "can make an augmented model score below its nested reference.",
        "",
        "## Scope, estimands, and timing",
        "",
    ]

    cohort_rows: list[list[object]] = []
    for population in ("full_808", "ftv_complete_375", "ftv_unavailable_433"):
        row = data.cohort.loc[data.cohort["population"].eq(population)].iloc[0]
        role = {
            "full_808": "Profile probes and secondary full-cohort C vs C+M",
            "ftv_complete_375": "Primary fully matched C/M/F comparisons",
            "ftv_unavailable_433": "Excluded from FTV estimand; not a comparison cohort",
        }[population]
        cohort_rows.append(
            [
                population,
                int(row["n"]),
                f"{int(row['pcr_positive'])} ({100.0 * float(row['pcr_prevalence']):.1f}%)",
                role,
            ]
        )
    lines.extend(
        [
            _table(("Population", "n", "pCR+", "Role"), cohort_rows),
            "",
            "The 808-patient and selected 375-patient results answer different estimands. "
            "Absolute scores across them are not paired effects. Every timing uses only its "
            "observed prefix under the [information timing contract](../information_timing_contract.csv). "
            "T3 is always labeled **late/pre-surgery** and is not silently combined with early "
            "prediction horizons. Seed 2026 and seed 3026 are model sensitivity replications on "
            "the same five patient folds, not independent patient samples.",
            "",
            "## Patient-profile information in LOCAL MRI states",
            "",
            "Each entry is AUROC mean (minimum–maximum) across the four seed×arm cells: "
            "2026/3026 × LOCAL0/LOCAL3.",
            "",
        ]
    )
    profile_rows: list[list[object]] = []
    for target in PROFILE_TARGETS:
        row: list[object] = [_target_label(target)]
        for view in PROFILE_VIEWS:
            values = profile_summary.loc[
                profile_summary["target"].eq(target) & profile_summary["view"].eq(view)
            ].iloc[0]
            row.append(
                _metric_cell(
                    CellSummary(
                        float(values["mean"]),
                        float(values["minimum"]),
                        float(values["maximum"]),
                        int(values["count"]),
                    )
                )
            )
        profile_rows.append(row)
    lines.extend(
        [
            _table(
                ("Target", *(_view_label(value) for value in PROFILE_VIEWS)),
                profile_rows,
            ),
            "",
            "These are correlation/decodability probes only. Even a successful HR or HER2 "
            "probe would not establish complementarity because those biomarkers are already "
            "available to the clinical model. See [profile aggregate](../metrics/profile_oof_metrics.csv) "
            "and [profile figure](../figures/profile_auroc_heatmap.png).",
            "",
            "## Clinical-only baselines, including treatment sensitivity",
            "",
        ]
    )
    clinical_rows: list[list[object]] = []
    for population in ("full_808", "ftv_complete_375"):
        for contract in CLINICAL_CONTRACTS:
            row = data.clinical.loc[
                data.clinical["population"].eq(population)
                & data.clinical["clinical_contract"].eq(contract)
            ].iloc[0]
            treatment = (
                "with treatment"
                if "with_treatment" in contract
                else (
                    "without treatment"
                    if "without_treatment" in contract
                    else "HR/HER2 only"
                )
            )
            clinical_rows.append(
                [
                    population,
                    contract,
                    treatment,
                    f"{float(row['auroc']):.3f}",
                    f"{float(row['auprc']):.3f}",
                    f"{float(row['brier']):.3f}",
                ]
            )
    lines.extend(
        [
            _table(
                (
                    "Population",
                    "Clinical contract",
                    "Treatment",
                    "AUROC",
                    "AUPRC",
                    "Brier",
                ),
                clinical_rows,
            ),
            "",
            "Treatment arm is the assigned regimen, not delivered exposure or a causal effect. "
            "The matched with/without-treatment results are therefore prediction sensitivity "
            "analyses. Field definitions and missingness are in the "
            "[clinical feature inventory](clinical_feature_inventory.md).",
            "",
            "## MRI-only and joint pCR discrimination",
            "",
            "AUROC entries are mean (minimum–maximum) across four seed×arm cells.",
            "",
        ]
    )
    full_rows: list[list[object]] = []
    for timing in TIMINGS:
        full_rows.append(
            [
                _timing_label(timing),
                *(
                    _metric_cell(_pcr_cell(data, "full_808", timing, model))
                    for model in ("C", "M", "C+M", "C+M_error_correction")
                ),
            ]
        )
    lines.extend(
        [
            "### Full 808-patient estimand",
            "",
            _table(
                (
                    "Timing",
                    "C",
                    "M",
                    "C+M",
                    "Clinical-error correction",
                ),
                full_rows,
            ),
            "",
            "### FTV-complete 375-patient estimand",
            "",
        ]
    )
    primary_rows: list[list[object]] = []
    c_m_vs_c_f: list[str] = []
    for timing in TIMINGS:
        c_m_mean = _pcr_cell(data, "ftv_complete_375", timing, "C+M").mean
        c_f_mean = _pcr_cell(data, "ftv_complete_375", timing, "C+F").mean
        c_m_vs_c_f.append(f"{_timing_label(timing)} {c_m_mean - c_f_mean:+.3f}")
        primary_rows.append(
            [
                _timing_label(timing),
                *(
                    _metric_cell(_pcr_cell(data, "ftv_complete_375", timing, model))
                    for model in (
                        "C",
                        "M",
                        "C+M",
                        "F",
                        "C+F",
                        "C+F+M",
                    )
                ),
            ]
        )
    lines.extend(
        [
            _table(
                ("Timing", "C", "M", "C+M", "F", "C+F", "C+F+M"),
                primary_rows,
            ),
            "",
            "The other prespecified FTV benchmark, mean AUROC(C+M) − AUROC(C+F) "
            "within the same 375 patients and folds, was "
            + "; ".join(c_m_vs_c_f)
            + ". These matched-cell point differences have no dedicated bootstrap interval and "
            "are interpreted descriptively.",
            "",
            "Full AUROC/AUPRC/Brier aggregates are in "
            "[pcr_oof_metrics.csv](../metrics/pcr_oof_metrics.csv); the primary timing plot is "
            "[here](../figures/primary_pcr_auroc_by_timing.png).",
            "",
            "## Paired incremental effects and 95% bootstrap intervals",
            "",
            "Positive improvement favors the augmented model for AUROC and AUPRC. Means and "
            "point ranges summarize four seed×arm cells. The displayed CI envelope is the "
            "minimum lower to maximum upper limit across four separate, fold-stratified paired "
            "patient-bootstrap 95% CIs; it is **not** a pooled CI.",
            "",
        ]
    )
    bootstrap_rows: list[list[object]] = []
    comparisons = (
        ("full_808", "C+M_vs_C", "C+M − C"),
        ("ftv_complete_375", "C+M_vs_C", "C+M − C"),
        ("ftv_complete_375", "C+F+M_vs_C+F", "C+F+M − C+F"),
    )
    for population, comparison, display in comparisons:
        for timing in TIMINGS:
            auroc = _bootstrap_cell(data, population, comparison, timing, "auroc")
            auprc = _bootstrap_cell(data, population, comparison, timing, "auprc")
            bootstrap_rows.append(
                [
                    population,
                    display,
                    _timing_label(timing),
                    _delta_cell(_summary(auroc, "improvement")),
                    _ci_envelope(auroc),
                    f"{int(auroc['ci_lower'].gt(0.0).sum())}/4",
                    _delta_cell(_summary(auprc, "improvement")),
                    _ci_envelope(auprc),
                ]
            )
    lines.extend(
        [
            _table(
                (
                    "Population",
                    "Comparison",
                    "Timing",
                    "ΔAUROC mean (cell range)",
                    "AUROC 95% cell-CI envelope",
                    "Positive AUROC CIs",
                    "ΔAUPRC mean (cell range)",
                    "AUPRC 95% cell-CI envelope",
                ),
                bootstrap_rows,
            ),
            "",
            f"Bootstrap unit: patient within outer fold; {observed_bootstrap:,} draws per cell "
            f"in this aggregate ({configured_bootstrap:,} configured). Exact cell intervals are "
            "in [bootstrap_ci.csv](../metrics/bootstrap_ci.csv). Forest plots: "
            "[C+M vs C, full 808](../figures/full_cohort_incremental_forest.png) and "
            "[C+F+M vs C+F, selected 375](../figures/beyond_ftv_incremental_forest.png).",
            "",
            "## MRI after removing FTV-associated components",
            "",
        ]
    )
    residual_rows: list[list[object]] = []
    for timing in TIMINGS:
        residual_rows.append(
            [
                _timing_label(timing),
                _metric_cell(_pcr_cell(data, "ftv_complete_375", timing, "M")),
                _metric_cell(_pcr_cell(data, "ftv_complete_375", timing, "M_residual")),
                _metric_cell(_pcr_cell(data, "ftv_complete_375", timing, "C+F")),
                _metric_cell(
                    _pcr_cell(data, "ftv_complete_375", timing, "C+F+M_residual")
                ),
                _delta_cell(_residual_joint_delta(data, timing)),
            ]
        )
    lines.extend(
        [
            _table(
                (
                    "Timing",
                    "M",
                    "M residual",
                    "C+F",
                    "C+F+M residual",
                    "Δ residual joint − C+F",
                ),
                residual_rows,
            ),
            "",
            "The FTV→MRI linear map is fitted on outer train only. This residual comparison is "
            "descriptive because the paired bootstrap table is defined for raw M. See the "
            "[residual figure](../figures/residual_mri_comparison.png).",
            "",
            "## Clinical-error test",
            "",
            "The secondary ridge test predicts `y − p_clinical` from M using outer train, selects "
            "on validation, and corrects untouched outer-test clinical probabilities. R² tests "
            "error predictability; ΔAUROC and ΔAUPRC test whether correction improves ranking.",
            "",
        ]
    )
    error_rows: list[list[object]] = []
    for population in ("full_808", "ftv_complete_375"):
        for timing in TIMINGS:
            selected = data.clinical_residual.loc[
                data.clinical_residual["population"].eq(population)
                & data.clinical_residual["timing"].eq(timing)
            ]
            error_rows.append(
                [
                    population,
                    _timing_label(timing),
                    _metric_cell(_summary(selected, "r2")),
                    _metric_cell(_summary(selected, "spearman")),
                    _delta_cell(_summary(selected, "delta_auroc")),
                    _delta_cell(_summary(selected, "delta_auprc")),
                    _delta_cell(_summary(selected, "brier_improvement")),
                ]
            )
    lines.extend(
        [
            _table(
                (
                    "Population",
                    "Timing",
                    "Error R²",
                    "Error Spearman",
                    "ΔAUROC",
                    "ΔAUPRC",
                    "Brier improvement",
                ),
                error_rows,
            ),
            "",
            "A lower Brier score after correction can reflect probability shrinkage/calibration "
            "even when residual R² and discrimination do not improve; it is not by itself "
            "evidence of complementary phenotype signal. Source: "
            "[clinical_residual_metrics.csv](../metrics/clinical_residual_metrics.csv).",
            "",
            "## Subtype-conditioned pCR findings",
            "",
            "AUROCs are mean (minimum–maximum) across four cells. The final column is paired "
            "within cell before summarization.",
            "",
        ]
    )
    subgroup_delta = _subgroup_deltas(data)
    subgroup_rows: list[list[object]] = []
    for subgroup in SUBGROUPS:
        for timing in TIMINGS:
            row: list[object] = [subgroup, _timing_label(timing)]
            for model in SUBGROUP_MODELS:
                selected = data.subgroup.loc[
                    data.subgroup["subgroup"].eq(subgroup)
                    & data.subgroup["timing"].eq(timing)
                    & data.subgroup["model"].eq(model)
                ]
                row.append(_metric_cell(_summary(selected, "auroc")))
            selected_delta = subgroup_delta.loc[
                subgroup_delta["subgroup"].eq(subgroup)
                & subgroup_delta["timing"].eq(timing)
            ]
            row.append(_delta_cell(_summary(selected_delta, "delta_auroc")))
            subgroup_rows.append(row)
    lines.extend(
        [
            _table(
                (
                    "Subtype",
                    "Timing",
                    "M",
                    "Remaining clinical",
                    "Remaining clinical+M",
                    "Δ joint − clinical",
                ),
                subgroup_rows,
            ),
            "",
            "These analyses are descriptive, particularly in smaller strata, and do not create "
            "new confirmatory claims. See [subgroup aggregate](../metrics/subgroup_metrics.csv) "
            "and [subgroup figure](../figures/subgroup_auroc.png).",
            "",
            "## Scientific classification",
            "",
            _classification_paragraph(classification, profile_summary, data),
            "",
            "Classification logic is intentionally conservative: a profile or MRI-only result is "
            "called material only when a four-cell mean reaches AUROC 0.60 and every cell is above "
            "0.50; an incremental timing is called supported only when all four cell-specific paired "
            "95% AUROC CIs are above zero. These are transparent synthesis rules, not post-hoc model "
            "selection or a substitute for replication.",
            "",
            "## Direct answers to the 12 requested questions",
            "",
        ]
    )

    hr_peak = _profile_peak(profile_summary, "HR")
    her2_peak = _profile_peak(profile_summary, "HER2")
    subtype_peak = _profile_peak(profile_summary, "subtype_4class")

    def profile_answer(row: pd.Series) -> str:
        mean = float(row["mean"])
        descriptor = (
            "material descriptive signal"
            if mean >= 0.60 and float(row["minimum"]) > 0.50
            else "only weak signal" if mean >= 0.55 else "no material signal"
        )
        return (
            f"{descriptor}; best mean AUROC {mean:.3f} "
            f"({float(row['minimum']):.3f}–{float(row['maximum']):.3f}) at "
            f"{_view_label(str(row['view']))}."
        )

    best_mri_rows = [
        (population, timing, _pcr_cell(data, population, timing, "M"))
        for population in ("full_808", "ftv_complete_375")
        for timing in TIMINGS
    ]
    best_mri = max(best_mri_rows, key=lambda value: value[2].mean)
    full_c = float(
        data.clinical.loc[
            data.clinical["population"].eq("full_808")
            & data.clinical["clinical_contract"].eq("C2_full_with_treatment"),
            "auroc",
        ].iloc[0]
    )
    primary_c = float(
        data.clinical.loc[
            data.clinical["population"].eq("ftv_complete_375")
            & data.clinical["clinical_contract"].eq("C2_full_with_treatment"),
            "auroc",
        ].iloc[0]
    )
    clinical_supported_text = (
        ", ".join(
            _timing_label(value)
            for value in classification.clinical_incremental_timings
        )
        if classification.clinical_incremental_timings
        else "none"
    )
    beyond_supported_text = (
        ", ".join(_timing_label(value) for value in classification.beyond_ftv_timings)
        if classification.beyond_ftv_timings
        else "none"
    )
    best_subgroup_m = max(
        (
            (
                subgroup,
                timing,
                _summary(
                    data.subgroup.loc[
                        data.subgroup["subgroup"].eq(subgroup)
                        & data.subgroup["timing"].eq(timing)
                        & data.subgroup["model"].eq("M")
                    ],
                    "auroc",
                ),
            )
            for subgroup in SUBGROUPS
            for timing in TIMINGS
        ),
        key=lambda value: value[2].mean,
    )
    subgroup_supported = []
    for subgroup in SUBGROUPS:
        for timing in TIMINGS:
            selected = subgroup_delta.loc[
                subgroup_delta["subgroup"].eq(subgroup)
                & subgroup_delta["timing"].eq(timing)
            ]
            if selected["delta_auroc"].gt(0.0).all():
                subgroup_supported.append(f"{subgroup}/{_timing_label(timing)}")

    residual_supported = []
    for timing in TIMINGS:
        delta = _residual_joint_delta(data, timing)
        if delta.minimum > 0.0:
            residual_supported.append(_timing_label(timing))
    error_supported = []
    for population in ("full_808", "ftv_complete_375"):
        for timing in TIMINGS:
            selected = data.clinical_residual.loc[
                data.clinical_residual["population"].eq(population)
                & data.clinical_residual["timing"].eq(timing)
            ]
            if selected["r2"].gt(0.0).all() and selected["delta_auroc"].gt(0.0).all():
                error_supported.append(f"{population}/{_timing_label(timing)}")

    contribution = {
        "A": "complementary phenotype signal is supported under this audit protocol",
        "B": "the detectable MRI signal is mostly FTV/burden-redundant",
        "C": "the detectable MRI signal is mostly molecular/clinical-redundant",
        "D": "the current MRI contribution is weak/unclear and the state underutilizes phenotype",
    }[classification.code]
    next_step = {
        "A": (
            "Replicate with the planned multi-seed confirmation and external calibration before "
            "changing the representation objective."
        ),
        "B": (
            "Emphasize FTV-disentangled morphology, enhancement heterogeneity, kinetic texture, "
            "and spatial context rather than additional burden supervision."
        ),
        "C": (
            "Target phenotype information conditional on HR/HER2/treatment—morphology, "
            "heterogeneity, kinetics, and microenvironmental context—without re-encoding known "
            "clinical labels as the endpoint."
        ),
        "D": (
            "Prioritize a richer/foundation image encoder and pCR-free phenotype-learning "
            "objectives for morphology, heterogeneity, enhancement kinetics, and multi-scale "
            "spatial context; then repeat this frozen audit unchanged."
        ),
    }[classification.code]

    direct_answers = [
        f"1. **Can LOCAL MRI predict HR?** {profile_answer(hr_peak)}",
        f"2. **Can LOCAL MRI predict HER2?** {profile_answer(her2_peak)}",
        f"3. **Can LOCAL MRI predict four-class subtype?** {profile_answer(subtype_peak)}",
        (
            "4. **How does MRI-only pCR perform?** The best mean M AUROC is "
            f"{best_mri[2].mean:.3f} ({best_mri[2].minimum:.3f}–{best_mri[2].maximum:.3f}) "
            f"in `{best_mri[0]}` at {_timing_label(best_mri[1])}; this is "
            f"{'material' if classification.mri_pcr_material else 'weak'} discrimination."
        ),
        (
            "5. **How does clinical-only perform?** Primary `C2_full_with_treatment` AUROC is "
            f"{full_c:.3f} in `full_808` and {primary_c:.3f} in `ftv_complete_375`; "
            "the table above gives C1, original-condition, full-profile, and treatment-excluded "
            "counterparts."
        ),
        (
            "6. **Does C+M beat C?** Supported timings with all four paired 95% AUROC CIs "
            f"above zero: **{clinical_supported_text}**. Negative cell means are read as no "
            "supported incremental value, not MRI harm."
        ),
        (
            "7. **Does C+F+M beat C+F?** Supported timings: "
            f"**{beyond_supported_text}** in the selected 375-patient estimand."
        ),
        (
            "8. **Does MRI retain pCR signal after removing FTV-associated components?** "
            f"Timings where all four residual-joint AUROC deltas are positive: "
            f"**{', '.join(residual_supported) if residual_supported else 'none'}**. "
            f"Clinical-error tests with uniformly positive R² and ΔAUROC: "
            f"**{', '.join(error_supported) if error_supported else 'none'}**."
        ),
        (
            "9. **Within subtype, does MRI still distinguish pCR?** Best M mean AUROC is "
            f"{best_subgroup_m[2].mean:.3f} ({best_subgroup_m[2].minimum:.3f}–"
            f"{best_subgroup_m[2].maximum:.3f}) in {best_subgroup_m[0]} at "
            f"{_timing_label(best_subgroup_m[1])}. Uniformly positive joint-minus-clinical "
            f"cells: **{', '.join(subgroup_supported) if subgroup_supported else 'none'}**; "
            "these subgroup results remain descriptive."
        ),
        f"10. **What is the dominant MRI contribution?** {contribution}.",
        (
            "11. **Does the current world model truly use MRI?** The original primary FLR does "
            "not establish that: its `future_response_state` is generated from lesion geometry + "
            "clinical/treatment condition, with the image transition exported separately. The LOCAL "
            "states tested here are image-only at inference, but their incremental pCR results must "
            "be supported before claiming useful MRI contribution."
        ),
        f"12. **What image information should be strengthened next?** {next_step}",
    ]
    lines.extend([*direct_answers, "", "## Original-FLR interpretation boundary", ""])
    lines.extend(
        [
            "The [clinical inventory](clinical_feature_inventory.md#critical-representation-boundary-original-state-is-not-mri-only) "
            "shows that the original frozen state used by FLR comes from geometry plus condition. "
            "The [forecast implementation](../../../ispy_jepa_tmi_clean/corejepa/models/corejepa.py#L149) "
            "and [frozen export](../../../ispy_jepa_tmi_clean/corejepa/training/runner.py#L290) "
            "do not pass the image latent into that primary readout. In contrast, the "
            "[LOCAL exporter](../../local_global_response_state_pilot/src/lg_response_pilot/features.py#L237) "
            "calls the image-only response encoder. Therefore this report distinguishes **original "
            "FLR behavior** from **LOCAL image-state behavior** and never uses the original state as M.",
            "",
            "## Reproducibility and delivery provenance",
            "",
        ]
    )
    provenance_rows = [
        ["Experiment branch", str(data.run_summary["branch"])],
        ["Parent commit", str(data.run_summary["parent_commit"])],
        ["Evidence status", str(data.run_summary["evidence_status"])],
        ["Clinical inventory SHA-256", data.inventory_sha256],
        ["Delivery branch", data.delivery["branch"]],
        ["Delivery commit SHA", data.delivery["commit_sha"]],
        ["Push status", data.delivery["push_status"]],
        ["Remote", data.delivery["remote"]],
        [
            "Delivery provenance file",
            (
                "[delivery_provenance.json](delivery_provenance.json)"
                if data.delivery_present
                else "PENDING"
            ),
        ],
    ]
    lines.extend(
        [
            _table(("Item", "Value"), provenance_rows),
            "",
            "Aggregate inputs and exact links:",
            "",
            "- [Experiment plan](../EXPERIMENT_PLAN.md)",
            "- [Audit configuration](../configs/audit.json)",
            "- [Clinical feature inventory](clinical_feature_inventory.md)",
            "- [Run summary](../metrics/run_summary.json)",
            "- [Cohort summary](../metrics/cohort_summary.csv)",
            "- [Clinical baselines](../metrics/clinical_baseline_metrics.csv)",
            "- [Profile OOF metrics](../metrics/profile_oof_metrics.csv)",
            "- [pCR OOF metrics](../metrics/pcr_oof_metrics.csv)",
            "- [Paired bootstrap intervals](../metrics/bootstrap_ci.csv)",
            "- [Clinical-error metrics](../metrics/clinical_residual_metrics.csv)",
            "- [Subtype metrics](../metrics/subgroup_metrics.csv)",
            "",
            "This generator reads no feature NPZ, OOF prediction, patient identifier, label row, "
            "or private bootstrap-draw file. All report values are computed from the linked "
            "aggregate metrics.",
            "",
        ]
    )
    markdown = "\n".join(lines)
    metadata = {
        "classification": classification.code,
        "classification_label": classification.label,
        "bootstrap_observed": observed_bootstrap,
        "bootstrap_configured": configured_bootstrap,
        "bootstrap_complete": bootstrap_complete,
        "report_bytes": len(markdown.encode("utf-8")),
    }
    return markdown, classification, metadata


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            if not text.endswith("\n"):
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    data = load_inputs(args.experiment_root)
    markdown, classification, metadata = build_report(data)
    output = data.root / "reports" / "final_report.md"
    if not args.dry_run:
        _atomic_write(output, markdown)
    print(
        json.dumps(
            {
                "status": "DRY_RUN_PASS" if args.dry_run else "COMPLETE",
                "aggregate_only": True,
                "classification": classification.code,
                "classification_label": classification.label,
                "output": str(output),
                "written": not args.dry_run,
                **metadata,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

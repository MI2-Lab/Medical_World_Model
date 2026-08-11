#!/usr/bin/env python3
"""Fail-closed verification for the MRI--clinical complementarity audit.

The verifier consumes the completed audit directory.  It does not import the
analysis runner (or any other experiment) and therefore remains an independent
check of output schemas, held-out pairing, cohort coverage, uncertainty, and
the public/private artifact boundary.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd


SCHEMA_VERSION = 1
TIMINGS = ("T0", "T1", "T2", "T3")
LOCAL_ARMS = ("LOCAL0", "LOCAL3")
SEED_BASES = (2026, 3026)
FOLDS = (0, 1, 2, 3, 4)
COHORT_SIZES: Mapping[str, int] = {
    "full_808": 808,
    "ftv_complete_375": 375,
}
EXPECTED_SUBGROUP_SIZES: Mapping[str, int] = {
    "HR+/HER2-": 320,
    "HR-/HER2-": 287,
    "HER2+": 201,
}
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_UNIT = "patient_within_outer_fold"

PRIVATE_TABLES = (
    "predictions/pcr_oof.private.csv",
    "predictions/profile_oof.private.csv",
    "predictions/subgroup_oof.private.csv",
)
METRIC_TABLES = (
    "metrics/pcr_oof_metrics.csv",
    "metrics/pcr_fold_metrics.csv",
    "metrics/profile_oof_metrics.csv",
    "metrics/clinical_baseline_metrics.csv",
    "metrics/incremental_effects.csv",
    "metrics/bootstrap_ci.csv",
    "metrics/subgroup_metrics.csv",
    "metrics/clinical_residual_metrics.csv",
    "metrics/cohort_summary.csv",
    "metrics/hyperparameter_selections.csv",
    "metrics/input_manifest.csv",
    "metrics/figure_manifest.csv",
)
REPORTS = (
    "reports/clinical_feature_inventory.md",
    "reports/final_report.md",
)
VERIFICATION_RELATIVE = "metrics/verification.json"

# The literal contains no identifier value and therefore does not trigger its
# own scanner.  These are the two canonical patient-key forms in the locked
# 808-patient cohort.  Merely writing the column name ``patient_id`` is allowed.
PATIENT_ID_BYTES_RE = re.compile(
    rb"(?i)(?<![A-Z0-9])(?:ISPY2-[0-9]{6}|ACRIN-6698-[0-9]{6})(?![A-Z0-9])"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_IDENTIFIER_ROOTS = frozenset({"features", "predictions", "logs"})

COLUMN_ALIASES: Mapping[str, tuple[str, ...]] = {
    "patient": ("patient_id",),
    "population": ("population", "cohort"),
    "timing": ("timing", "timepoint"),
    "view": ("view", "representation_view"),
    "arm": ("local_arm", "representation_arm", "arm"),
    "seed": ("seed_base", "model_seed", "seed"),
    "fold": ("fold", "outer_fold"),
    "model": ("model", "model_name"),
    "target": ("target", "target_name", "endpoint", "probe_target"),
    "label": ("y_true", "label_pcr", "target_value", "label"),
    "prediction": ("y_pred", "predicted_label", "prediction"),
    "probability": (
        "predicted_probability",
        "probability",
        "y_probability",
        "y_score",
        "pcr_probability",
    ),
    "subgroup": ("subgroup", "stratum", "clinical_stratum"),
    "n": ("n", "n_patients", "patients"),
    "clinical_contract": ("clinical_contract", "contract", "feature_contract"),
    "analysis": ("analysis", "task", "analysis_type", "endpoint"),
    "metric": ("metric", "metric_name"),
    "reference_model": ("reference_model", "baseline_model", "left_model"),
    "comparison_model": (
        "comparison_model",
        "augmented_model",
        "right_model",
    ),
    "reference_value": ("reference", "reference_value", "baseline_value"),
    "comparison_value": ("comparison_value", "augmented_value", "comparison"),
    "improvement": ("improvement", "effect", "delta"),
    "ci_lower": ("ci_lower", "ci_low", "lower"),
    "ci_upper": ("ci_upper", "ci_high", "upper"),
    "bootstrap_replicates": ("n_bootstrap", "bootstrap_replicates"),
    "valid_bootstrap": ("n_valid_bootstrap", "valid_replicates"),
    "bootstrap_unit": ("bootstrap_unit",),
    "path": ("path", "artifact_path", "file"),
    "sha256": ("sha256", "file_sha256"),
    "artifact": (
        "artifact",
        "input",
        "name",
        "kind",
        "figure",
        "figure_id",
    ),
}

METRIC_ALIASES: Mapping[str, tuple[str, ...]] = {
    "auroc": ("auroc", "macro_ovr_auroc"),
    "auprc": ("auprc", "macro_ovr_auprc"),
    "balanced_accuracy": ("balanced_accuracy", "balanced_acc"),
    "brier": ("brier", "brier_score"),
}


class VerificationError(ValueError):
    """An audit output violates a required acceptance contract."""


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    evidence: Any


class CheckCollector:
    def __init__(self) -> None:
        self.results: list[CheckResult] = []

    def run(self, name: str, function: Callable[[], Any]) -> None:
        try:
            evidence = function()
        except Exception as error:  # independent gates must all be attempted
            message = " ".join(str(error).split())
            # Never copy an offending identifier into the public JSON.
            safe = PATIENT_ID_BYTES_RE.sub(
                b"<redacted-patient-id>", message.encode()
            ).decode("utf-8", errors="replace")
            self.results.append(
                CheckResult(
                    name=name,
                    passed=False,
                    evidence={"error_type": type(error).__name__, "message": safe},
                )
            )
        else:
            self.results.append(CheckResult(name=name, passed=True, evidence=evidence))

    @property
    def passed(self) -> bool:
        return bool(self.results) and all(result.passed for result in self.results)


def _require_regular_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise VerificationError(f"{label} must not be a symbolic link")
    if not path.is_file():
        raise VerificationError(f"{label} is missing or is not a regular file")
    return path


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    _require_regular_file(path, label)
    try:
        frame = pd.read_csv(path)
    except (OSError, UnicodeError, pd.errors.ParserError) as error:
        raise VerificationError(f"{label} is not a readable CSV") from error
    if frame.columns.duplicated().any():
        raise VerificationError(f"{label} contains duplicate column names")
    if frame.empty:
        raise VerificationError(f"{label} must contain at least one data row")
    return frame


def _column(
    frame: pd.DataFrame,
    semantic: str,
    label: str,
    *,
    required: bool = True,
) -> str | None:
    aliases = COLUMN_ALIASES[semantic]
    matches = [candidate for candidate in aliases if candidate in frame.columns]
    if not matches:
        if required:
            raise VerificationError(
                f"{label} is missing {semantic!r}; expected one of {list(aliases)}"
            )
        return None
    return matches[0]


def _required_semantics(
    frame: pd.DataFrame, label: str, semantics: Sequence[str]
) -> dict[str, str]:
    return {semantic: str(_column(frame, semantic, label)) for semantic in semantics}


def _strings(series: pd.Series, label: str) -> pd.Series:
    if series.isna().any():
        raise VerificationError(f"{label} contains missing values")
    output = series.astype(str).str.strip()
    if output.eq("").any():
        raise VerificationError(f"{label} contains empty values")
    return output


def _numeric(series: pd.Series, label: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise VerificationError(
            f"{label} contains non-numeric, missing, or non-finite values"
        )
    return values


def _integers(
    series: pd.Series, label: str, *, minimum: int | None = None
) -> np.ndarray:
    values = _numeric(series, label)
    rounded = np.rint(values)
    if not np.array_equal(values, rounded):
        raise VerificationError(f"{label} contains non-integer values")
    output = rounded.astype(np.int64)
    if minimum is not None and np.any(output < minimum):
        raise VerificationError(f"{label} contains values below {minimum}")
    return output


def _bounded(
    series: pd.Series,
    label: str,
    *,
    lower: float,
    upper: float,
) -> np.ndarray:
    values = _numeric(series, label)
    if np.any(values < lower) or np.any(values > upper):
        raise VerificationError(f"{label} contains values outside [{lower},{upper}]")
    return values


def _metric_values(frame: pd.DataFrame, metric: str, label: str) -> np.ndarray:
    aliases = METRIC_ALIASES[metric]
    present = [column for column in aliases if column in frame.columns]
    if not present:
        raise VerificationError(
            f"{label} is missing {metric!r}; expected one of {list(aliases)}"
        )
    candidates = np.column_stack(
        [
            pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
            for column in present
        ]
    )
    finite = np.isfinite(candidates)
    if not finite.any(axis=1).all():
        raise VerificationError(
            f"{label}.{metric} is absent/non-finite in at least one row"
        )
    output = np.empty(len(frame), dtype=np.float64)
    for index in range(len(frame)):
        row = candidates[index, finite[index]]
        if row.size > 1 and not np.allclose(row, row[0], rtol=0.0, atol=1e-10):
            raise VerificationError(f"{label}.{metric} aliases disagree")
        output[index] = row[0]
    return output


def _require_probability_metrics(
    frame: pd.DataFrame,
    label: str,
    *,
    balanced_accuracy: bool,
    brier: bool,
) -> dict[str, tuple[float, float]]:
    names = ["auroc", "auprc"]
    if balanced_accuracy:
        names.append("balanced_accuracy")
    if brier:
        names.append("brier")
    evidence: dict[str, tuple[float, float]] = {}
    for name in names:
        values = _metric_values(frame, name, label)
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise VerificationError(f"{label}.{name} contains values outside [0,1]")
        evidence[name] = (float(values.min()), float(values.max()))
    return evidence


def _normalise_timing(series: pd.Series, label: str) -> pd.Series:
    values = _strings(series, label).str.upper()
    unknown = sorted(set(values) - set(TIMINGS))
    if unknown:
        raise VerificationError(f"{label} contains unsupported timings: {unknown}")
    return values


def _normalise_arm(series: pd.Series, label: str) -> pd.Series:
    values = _strings(series, label).str.upper()
    unknown = sorted(set(values) - set(LOCAL_ARMS))
    if unknown:
        raise VerificationError(f"{label} contains unsupported LOCAL arms: {unknown}")
    return values


def _normalise_population(series: pd.Series, label: str) -> pd.Series:
    values = _strings(series, label).str.lower()
    unknown = sorted(set(values) - set(COHORT_SIZES))
    if unknown:
        raise VerificationError(f"{label} contains unsupported populations: {unknown}")
    return values


def _normalise_models(series: pd.Series, label: str) -> pd.Series:
    return _strings(series, label).str.replace(r"\s+", "", regex=True).str.upper()


def _binary_labels(series: pd.Series, label: str) -> np.ndarray:
    values = _numeric(series, label)
    if not np.isin(values, (0.0, 1.0)).all():
        raise VerificationError(f"{label} must contain only binary 0/1 labels")
    return values.astype(np.int64)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _schema_evidence(
    frame: pd.DataFrame, label: str, required: Sequence[str]
) -> dict[str, Any]:
    resolved = _required_semantics(frame, label, required)
    return {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "resolved_semantics": resolved,
    }


def _frame_with_canonical_run_columns(
    frame: pd.DataFrame,
    label: str,
    *,
    include_population: bool,
    include_target: bool = False,
    include_subgroup: bool = False,
) -> tuple[pd.DataFrame, dict[str, str]]:
    semantics = ["patient"]
    if include_population:
        semantics.append("population")
    semantics.extend(("timing", "arm", "seed", "fold", "model", "label"))
    if include_target:
        semantics.append("target")
    if include_subgroup:
        semantics.append("subgroup")
    columns = _required_semantics(frame, label, semantics)
    output = frame.copy()
    output["__patient"] = _strings(frame[columns["patient"]], f"{label}.patient_id")
    if include_population:
        output["__population"] = _normalise_population(
            frame[columns["population"]], f"{label}.population"
        )
    output["__timing"] = _normalise_timing(frame[columns["timing"]], f"{label}.timing")
    output["__arm"] = _normalise_arm(frame[columns["arm"]], f"{label}.arm")
    output["__seed"] = _integers(frame[columns["seed"]], f"{label}.seed")
    if set(output["__seed"]) != set(SEED_BASES):
        raise VerificationError(
            f"{label} must contain exactly seeds {list(SEED_BASES)}"
        )
    output["__fold"] = _integers(frame[columns["fold"]], f"{label}.fold")
    if not set(output["__fold"]).issubset(set(FOLDS)):
        raise VerificationError(f"{label} contains a fold outside {list(FOLDS)}")
    output["__model"] = _normalise_models(frame[columns["model"]], f"{label}.model")
    if include_target:
        output["__target"] = _strings(frame[columns["target"]], f"{label}.target")
    if include_subgroup:
        output["__subgroup"] = _strings(frame[columns["subgroup"]], f"{label}.subgroup")
    return output, columns


def _signature(
    group: pd.DataFrame, *, include_label: bool
) -> tuple[tuple[Any, ...], ...]:
    columns = ["__patient", "__fold"]
    if include_label:
        columns.append("__label")
    return tuple(
        tuple(row)
        for row in group.loc[:, columns]
        .sort_values(["__patient", "__fold"], kind="mergesort")
        .itertuples(index=False, name=None)
    )


def _require_probability_column(
    frame: pd.DataFrame, label: str
) -> tuple[str, tuple[float, float]]:
    column = _column(frame, "probability", label)
    values = _bounded(frame[str(column)], f"{label}.{column}", lower=0.0, upper=1.0)
    return str(column), (float(values.min()), float(values.max()))


def _validate_pcr_oof(frame: pd.DataFrame) -> dict[str, Any]:
    label = PRIVATE_TABLES[0]
    data, columns = _frame_with_canonical_run_columns(
        frame, label, include_population=True
    )
    data["__label"] = _binary_labels(frame[columns["label"]], f"{label}.label")
    probability_column, probability_range = _require_probability_column(frame, label)
    group_columns = ["__population", "__timing", "__arm", "__seed", "__model"]
    if data.duplicated(group_columns + ["__patient"]).any():
        raise VerificationError(f"{label} repeats a patient within a run/model")

    expected_models = {
        "full_808": {"C", "M", "C+M"},
        "ftv_complete_375": {
            "C",
            "M",
            "F",
            "C+F",
            "C+M",
            "C+F+M",
            "M_RESIDUAL",
            "C+F+M_RESIDUAL",
        },
    }
    signatures_by_population: dict[str, tuple[tuple[Any, ...], ...]] = {}
    observed_keys: set[tuple[str, str, str, int, str]] = set()
    for key, group in data.groupby(group_columns, sort=False, dropna=False):
        population, timing, arm, seed, model = key
        expected_n = COHORT_SIZES[str(population)]
        if len(group) != expected_n:
            raise VerificationError(
                f"{label} group {key} has n={len(group)}, expected {expected_n}"
            )
        if set(group["__fold"]) != set(FOLDS):
            raise VerificationError(
                f"{label} group {key} does not cover all five folds"
            )
        signature = _signature(group, include_label=True)
        prior = signatures_by_population.setdefault(str(population), signature)
        if signature != prior:
            raise VerificationError(
                f"{label} patient/fold/label signature differs within {population}"
            )
        observed_keys.add(
            (str(population), str(timing), str(arm), int(seed), str(model))
        )

    expected_keys = {
        (population, timing, arm, seed, model)
        for population, models in expected_models.items()
        for timing in TIMINGS
        for arm in LOCAL_ARMS
        for seed in SEED_BASES
        for model in models
    }
    missing = sorted(expected_keys - observed_keys)
    if missing:
        raise VerificationError(
            f"{label} is missing {len(missing)} required population/timing/arm/seed/model cells"
        )
    return {
        "rows": int(len(frame)),
        "groups": int(data.groupby(group_columns).ngroups),
        "populations": sorted(set(data["__population"])),
        "timings": sorted(set(data["__timing"])),
        "arms": sorted(set(data["__arm"])),
        "seeds": sorted(int(value) for value in set(data["__seed"])),
        "probability_column": probability_column,
        "probability_range": probability_range,
        "pairing": "exact patient/fold/label signature across every model and run",
    }


def _normalise_profile_target(value: Any) -> str:
    text = str(value).strip().lower().replace("label_", "")
    if "subtype" in text:
        return "subtype"
    if "her2" in text:
        return "her2"
    if text in {"hr", "hormone_receptor", "hormone-receptor"}:
        return "hr"
    return text


def _validate_profile_scores(frame: pd.DataFrame, label: str) -> dict[str, Any]:
    target_column = str(_column(frame, "target", label))
    targets = frame[target_column].map(_normalise_profile_target)
    binary_rows = targets.isin({"hr", "her2"}).to_numpy()
    subtype_rows = targets.eq("subtype").to_numpy()
    scalar = _column(frame, "probability", label, required=False)
    probability_columns = [
        column
        for column in frame.columns
        if column.startswith("probability_") or column.startswith("prob_")
    ]
    json_columns = [
        column for column in ("probabilities", "probabilities_json") if column in frame
    ]
    if scalar is None and not probability_columns and not json_columns:
        raise VerificationError(
            f"{label} needs a scalar, classwise, or JSON probability field"
        )
    evidence: dict[str, Any] = {}
    if scalar is not None:
        values = pd.to_numeric(frame[scalar], errors="coerce").to_numpy(float)
        if not np.isfinite(values[binary_rows]).all() or np.any(
            (values[binary_rows] < 0.0) | (values[binary_rows] > 1.0)
        ):
            raise VerificationError(
                f"{label}.{scalar} must be a finite probability for binary probes"
            )
        if np.isfinite(values[subtype_rows]).any():
            raise VerificationError(
                f"{label}.{scalar} must be empty for the four-class subtype probe"
            )
        evidence["scalar_probability"] = {
            "column": scalar,
            "range": [
                float(values[binary_rows].min()),
                float(values[binary_rows].max()),
            ],
        }
    if probability_columns:
        matrix = np.column_stack(
            [
                pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
                for column in probability_columns
            ]
        )
        subtype_matrix = matrix[subtype_rows]
        if (
            subtype_matrix.shape[1] < 3
            or not np.isfinite(subtype_matrix).all()
            or np.any((subtype_matrix < 0.0) | (subtype_matrix > 1.0))
            or not np.allclose(subtype_matrix.sum(axis=1), 1.0, rtol=0.0, atol=1e-6)
        ):
            raise VerificationError(
                f"{label} subtype class probabilities do not sum to one"
            )
        if np.isfinite(matrix[binary_rows]).any():
            raise VerificationError(
                f"{label} subtype probability columns must be empty for binary probes"
            )
        evidence["class_probability_columns"] = probability_columns
    if json_columns:
        for column in json_columns:
            for value in _strings(frame[column], f"{label}.{column}"):
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError as error:
                    raise VerificationError(
                        f"{label}.{column} contains invalid JSON"
                    ) from error
                numbers = list(parsed.values()) if isinstance(parsed, dict) else parsed
                if not isinstance(numbers, list) or not numbers:
                    raise VerificationError(
                        f"{label}.{column} must encode probabilities"
                    )
                try:
                    vector = np.asarray(numbers, dtype=np.float64)
                except (TypeError, ValueError) as error:
                    raise VerificationError(
                        f"{label}.{column} is not numeric"
                    ) from error
                if (
                    vector.ndim != 1
                    or not np.isfinite(vector).all()
                    or np.any((vector < 0.0) | (vector > 1.0))
                    or not math.isclose(float(vector.sum()), 1.0, abs_tol=1e-6)
                ):
                    raise VerificationError(
                        f"{label}.{column} has invalid probability vectors"
                    )
        evidence["json_probability_columns"] = json_columns
    return evidence


def _validate_profile_oof(frame: pd.DataFrame) -> dict[str, Any]:
    label = PRIVATE_TABLES[1]
    columns = _required_semantics(
        frame,
        label,
        ("patient", "view", "arm", "seed", "fold", "target", "label", "prediction"),
    )
    data = frame.copy()
    data["__patient"] = _strings(frame[columns["patient"]], f"{label}.patient_id")
    data["__view"] = _strings(frame[columns["view"]], f"{label}.view")
    expected_views = {
        "T0",
        "T1",
        "T2",
        "T3",
        "long_T0_T1",
        "long_T0_T2",
        "long_T0_T3",
    }
    if set(data["__view"]) != expected_views:
        raise VerificationError(
            f"{label} does not contain the exact seven static/prefix views"
        )
    data["__arm"] = _normalise_arm(frame[columns["arm"]], f"{label}.arm")
    data["__seed"] = _integers(frame[columns["seed"]], f"{label}.seed")
    if set(data["__seed"]) != set(SEED_BASES):
        raise VerificationError(f"{label} must contain exactly both configured seeds")
    data["__fold"] = _integers(frame[columns["fold"]], f"{label}.fold")
    if not set(data["__fold"]).issubset(set(FOLDS)):
        raise VerificationError(f"{label} contains an unsupported fold")
    data["__target"] = _strings(frame[columns["target"]], f"{label}.target")
    prediction_column = _column(frame, "prediction", label)
    predictions = _strings(frame[str(prediction_column)], f"{label}.prediction")
    data["__target"] = data["__target"].map(_normalise_profile_target)
    if set(data["__target"]) != {"hr", "her2", "subtype"}:
        raise VerificationError(f"{label} must contain HR, HER2, and subtype targets")
    data["__label_raw"] = _strings(frame[columns["label"]], f"{label}.label")
    data["__prediction_raw"] = predictions
    group_columns = ["__target", "__view", "__arm", "__seed"]
    if data.duplicated(group_columns + ["__patient"]).any():
        raise VerificationError(f"{label} repeats a patient within a run/model")
    expected_grid = {
        (target, view, arm, seed)
        for target in ("hr", "her2", "subtype")
        for view in expected_views
        for arm in LOCAL_ARMS
        for seed in SEED_BASES
    }
    observed_grid: set[tuple[str, str, str, int]] = set()
    patient_fold_signature: tuple[tuple[Any, ...], ...] | None = None
    for key, group in data.groupby(group_columns, sort=False, dropna=False):
        target, view, arm, seed = key
        if len(group) != COHORT_SIZES["full_808"]:
            raise VerificationError(f"{label} group {key} must contain 808 patients")
        if set(group["__fold"]) != set(FOLDS):
            raise VerificationError(
                f"{label} group {key} does not cover all five folds"
            )
        signature = _signature(group, include_label=False)
        if patient_fold_signature is None:
            patient_fold_signature = signature
        elif signature != patient_fold_signature:
            raise VerificationError(
                f"{label} patient/fold signature differs across probe runs"
            )
        if target in {"hr", "her2"}:
            _binary_labels(group["__label_raw"], f"{label}.{target} labels")
            _binary_labels(group["__prediction_raw"], f"{label}.{target} predictions")
        else:
            classes = set(group["__label_raw"])
            if len(classes) != 4:
                raise VerificationError(
                    f"{label} subtype target must contain four classes"
                )
            if not set(group["__prediction_raw"]).issubset(classes):
                raise VerificationError(
                    f"{label} subtype predictions contain unknown classes"
                )
        observed_grid.add((str(target), str(view), str(arm), int(seed)))
    if observed_grid != expected_grid:
        raise VerificationError(
            f"{label} does not cover the exact target/timing/arm/seed grid"
        )
    score_evidence = _validate_profile_scores(frame, label)
    if "threshold" not in frame:
        raise VerificationError(f"{label} is missing the validation-selected threshold")
    threshold = pd.to_numeric(frame["threshold"], errors="coerce").to_numpy(float)
    binary_rows = data["__target"].isin({"hr", "her2"}).to_numpy()
    subtype_rows = data["__target"].eq("subtype").to_numpy()
    if (
        not np.isfinite(threshold[binary_rows]).all()
        or np.any((threshold[binary_rows] < 0.0) | (threshold[binary_rows] > 1.0))
        or np.isfinite(threshold[subtype_rows]).any()
    ):
        raise VerificationError(
            f"{label}.threshold violates the binary/subtype contract"
        )
    return {
        "rows": int(len(frame)),
        "groups": int(data.groupby(group_columns).ngroups),
        "targets": sorted(set(data["__target"])),
        "score_contract": score_evidence,
    }


def _normalise_subgroup(value: Any) -> str:
    text = str(value).strip().upper().replace(" ", "").replace("_", "")
    if text in {"HER2+", "HER2POS", "HER2POSITIVE"}:
        return "HER2+"
    if text in {"HR+/HER2-", "HRPOS/HER2NEG", "HRPOSHER2NEG"}:
        return "HR+/HER2-"
    if text in {
        "HR-/HER2-",
        "HRNEG/HER2NEG",
        "HRNEGHER2NEG",
        "TRIPLENEGATIVE",
        "TNBC",
    }:
        return "HR-/HER2-"
    return str(value).strip()


def _validate_subgroup_oof(frame: pd.DataFrame) -> dict[str, Any]:
    label = PRIVATE_TABLES[2]
    include_population = _column(frame, "population", label, required=False) is not None
    data, columns = _frame_with_canonical_run_columns(
        frame,
        label,
        include_population=include_population,
        include_subgroup=True,
    )
    if include_population and set(data["__population"]) != {"full_808"}:
        raise VerificationError(f"{label} must use population full_808")
    data["__label"] = _binary_labels(frame[columns["label"]], f"{label}.label")
    _require_probability_column(frame, label)
    data["__subgroup"] = data["__subgroup"].map(_normalise_subgroup)
    if set(data["__subgroup"]) != set(EXPECTED_SUBGROUP_SIZES):
        raise VerificationError(
            f"{label} does not contain the three prespecified subgroups"
        )
    group_columns = ["__subgroup", "__timing", "__arm", "__seed", "__model"]
    if data.duplicated(group_columns + ["__patient"]).any():
        raise VerificationError(f"{label} repeats a patient within a run/model")
    signatures: dict[str, tuple[tuple[Any, ...], ...]] = {}
    models_per_cell: dict[tuple[str, str, str, int], set[str]] = {}
    for key, group in data.groupby(group_columns, sort=False, dropna=False):
        subgroup, timing, arm, seed, model = key
        expected_n = EXPECTED_SUBGROUP_SIZES[str(subgroup)]
        if len(group) != expected_n:
            raise VerificationError(
                f"{label} group {key} has n={len(group)}, expected {expected_n}"
            )
        if set(group["__fold"]) != set(FOLDS):
            raise VerificationError(
                f"{label} group {key} does not cover all five folds"
            )
        signature = _signature(group, include_label=True)
        prior = signatures.setdefault(str(subgroup), signature)
        if signature != prior:
            raise VerificationError(
                f"{label} patient/fold/label signature differs within subgroup {subgroup}"
            )
        models_per_cell.setdefault(
            (str(subgroup), str(timing), str(arm), int(seed)), set()
        ).add(str(model))
    expected_cells = {
        (subgroup, timing, arm, seed)
        for subgroup in EXPECTED_SUBGROUP_SIZES
        for timing in TIMINGS
        for arm in LOCAL_ARMS
        for seed in SEED_BASES
    }
    if set(models_per_cell) != expected_cells:
        raise VerificationError(
            f"{label} does not cover every subgroup/timing/arm/seed cell"
        )
    if any(len(models) < 3 for models in models_per_cell.values()):
        raise VerificationError(
            f"{label} requires at least three paired models per subgroup cell"
        )
    return {
        "rows": int(len(frame)),
        "groups": int(data.groupby(group_columns).ngroups),
        "subgroup_sizes": dict(EXPECTED_SUBGROUP_SIZES),
        "pairing": "exact patient/fold/label signature across subgroup models",
    }


def _canonical_metric_name(value: Any) -> str:
    text = str(value).strip().lower().replace("score", "").replace("_", "")
    if "brier" in text:
        return "brier"
    if "auprc" in text or "averageprecision" in text:
        return "auprc"
    if "auroc" in text or text == "auc" or "rocauc" in text:
        return "auroc"
    return str(value).strip().lower()


def _validate_metric_identity_columns(
    frame: pd.DataFrame,
    label: str,
    *,
    population: bool = True,
    target: bool = False,
    subgroup: bool = False,
    fold: bool = False,
) -> dict[str, str]:
    semantics: list[str] = []
    if population:
        semantics.append("population")
    semantics.extend(("timing", "arm", "seed"))
    if fold:
        semantics.append("fold")
    semantics.append("model")
    if target:
        semantics.append("target")
    if subgroup:
        semantics.append("subgroup")
    semantics.append("n")
    columns = _required_semantics(frame, label, semantics)
    _normalise_timing(frame[columns["timing"]], f"{label}.timing")
    _normalise_arm(frame[columns["arm"]], f"{label}.arm")
    seeds = _integers(frame[columns["seed"]], f"{label}.seed")
    if set(seeds) != set(SEED_BASES):
        raise VerificationError(f"{label} must contain exactly both configured seeds")
    if fold:
        folds = _integers(frame[columns["fold"]], f"{label}.fold")
        if set(folds) != set(FOLDS):
            raise VerificationError(
                f"{label} must contain all five folds and no others"
            )
    _normalise_models(frame[columns["model"]], f"{label}.model")
    counts = _integers(frame[columns["n"]], f"{label}.n", minimum=1)
    if population:
        populations = _normalise_population(
            frame[columns["population"]], f"{label}.population"
        )
        if not fold:
            expected = np.asarray([COHORT_SIZES[value] for value in populations])
            if not np.array_equal(counts, expected):
                raise VerificationError(
                    f"{label}.n does not equal its declared cohort size"
                )
    return columns


def _validate_pcr_metrics(frame: pd.DataFrame, *, folds: bool) -> dict[str, Any]:
    label = METRIC_TABLES[1 if folds else 0]
    columns = _validate_metric_identity_columns(frame, label, fold=folds)
    ranges = _require_probability_metrics(
        frame, label, balanced_accuracy=True, brier=True
    )
    key_semantics = ["population", "timing", "arm", "seed", "model"]
    if folds:
        key_semantics.append("fold")
    keys = [columns[semantic] for semantic in key_semantics]
    if frame.duplicated(keys).any():
        raise VerificationError(f"{label} repeats a metric cell")
    populations = _normalise_population(
        frame[columns["population"]], f"{label}.population"
    )
    timings = _normalise_timing(frame[columns["timing"]], f"{label}.timing")
    arms = _normalise_arm(frame[columns["arm"]], f"{label}.arm")
    seeds = _integers(frame[columns["seed"]], f"{label}.seed")
    models = _normalise_models(frame[columns["model"]], f"{label}.model")
    actual_keys = pd.DataFrame(
        {
            "population": populations,
            "timing": timings,
            "arm": arms,
            "seed": seeds,
            "model": models,
        }
    )
    if folds:
        actual_keys["fold"] = _integers(frame[columns["fold"]], f"{label}.fold")
    expected_models = {
        "full_808": {"C", "M", "C+M", "C+M_ERROR_CORRECTION"},
        "ftv_complete_375": {
            "C",
            "M",
            "F",
            "C+F",
            "C+M",
            "C+F+M",
            "M_RESIDUAL",
            "C+F+M_RESIDUAL",
            "C+M_ERROR_CORRECTION",
        },
    }
    expected_keys = {
        (population, timing, arm, seed, model, fold)
        for population, model_names in expected_models.items()
        for timing in TIMINGS
        for arm in LOCAL_ARMS
        for seed in SEED_BASES
        for model in model_names
        for fold in (FOLDS if folds else (None,))
    }
    observed_keys = {
        (
            row.population,
            row.timing,
            row.arm,
            int(row.seed),
            row.model,
            int(row.fold) if folds else None,
        )
        for row in actual_keys.itertuples(index=False)
    }
    if observed_keys != expected_keys:
        raise VerificationError(f"{label} does not cover the exact pCR model grid")
    if folds:
        counts = _integers(frame[columns["n"]], f"{label}.n", minimum=1)
        data = frame.copy()
        data["__n"] = counts
        data["__population"] = _normalise_population(
            frame[columns["population"]], f"{label}.population"
        )
        grouped = data.groupby(
            [
                columns[name]
                for name in ("population", "timing", "arm", "seed", "model")
            ],
            dropna=False,
        )
        for key, group in grouped:
            expected = COHORT_SIZES[str(group["__population"].iloc[0])]
            if int(group["__n"].sum()) != expected or len(group) != len(FOLDS):
                raise VerificationError(
                    f"{label} fold counts do not sum to the cohort size for {key}"
                )
    return {"rows": int(len(frame)), "metric_ranges": ranges}


def _validate_profile_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    label = METRIC_TABLES[2]
    columns = _required_semantics(frame, label, ("view", "arm", "seed", "target", "n"))
    views = _strings(frame[columns["view"]], f"{label}.view")
    expected_views = {
        "T0",
        "T1",
        "T2",
        "T3",
        "long_T0_T1",
        "long_T0_T2",
        "long_T0_T3",
    }
    if set(views) != expected_views:
        raise VerificationError(f"{label} does not contain all static and prefix views")
    arms = _normalise_arm(frame[columns["arm"]], f"{label}.arm")
    seeds = _integers(frame[columns["seed"]], f"{label}.seed")
    if set(seeds) != set(SEED_BASES):
        raise VerificationError(f"{label} must contain exactly both configured seeds")
    targets = frame[columns["target"]].map(_normalise_profile_target)
    if set(targets) != {"hr", "her2", "subtype"}:
        raise VerificationError(f"{label} must contain HR, HER2, and subtype")
    counts = _integers(frame[columns["n"]], f"{label}.n", minimum=1)
    if not np.equal(counts, COHORT_SIZES["full_808"]).all():
        raise VerificationError(f"{label} rows must each report n=808")
    keys = pd.DataFrame({"view": views, "arm": arms, "seed": seeds, "target": targets})
    if keys.duplicated().any():
        raise VerificationError(f"{label} repeats a profile metric cell")
    expected_grid = {
        (view, arm, seed, target)
        for view in expected_views
        for arm in LOCAL_ARMS
        for seed in SEED_BASES
        for target in ("hr", "her2", "subtype")
    }
    if set(keys.itertuples(index=False, name=None)) != expected_grid:
        raise VerificationError(f"{label} does not cover the full profile metric grid")
    ranges = _require_probability_metrics(
        frame, label, balanced_accuracy=True, brier=False
    )
    if "brier" not in frame:
        raise VerificationError(f"{label} is missing the binary Brier column")
    brier_values = pd.to_numeric(frame["brier"], errors="coerce").to_numpy(float)
    binary = targets.isin({"hr", "her2"}).to_numpy()
    subtype = targets.eq("subtype").to_numpy()
    if (
        not np.isfinite(brier_values[binary]).all()
        or np.any((brier_values[binary] < 0.0) | (brier_values[binary] > 1.0))
        or np.isfinite(brier_values[subtype]).any()
    ):
        raise VerificationError(f"{label}.brier violates the binary/subtype contract")
    return {"rows": int(len(frame)), "metric_ranges": ranges}


def _validate_clinical_baselines(frame: pd.DataFrame) -> dict[str, Any]:
    label = METRIC_TABLES[3]
    columns = _required_semantics(
        frame, label, ("population", "clinical_contract", "n")
    )
    populations = _normalise_population(
        frame[columns["population"]], f"{label}.population"
    )
    contracts = _strings(
        frame[columns["clinical_contract"]], f"{label}.clinical_contract"
    )
    counts = _integers(frame[columns["n"]], f"{label}.n", minimum=1)
    expected = np.asarray([COHORT_SIZES[value] for value in populations])
    if not np.array_equal(counts, expected):
        raise VerificationError(f"{label}.n does not equal its declared cohort size")
    required_contracts = {
        "C1_hr_her2",
        "C_condition_without_treatment",
        "C_condition_with_treatment",
        "C2_full_without_treatment",
        "C2_full_with_treatment",
    }
    if not required_contracts.issubset(set(contracts)):
        raise VerificationError(f"{label} is missing one or more clinical contracts")
    keys = set(zip(populations, contracts, strict=True))
    expected_keys = {
        (population, contract)
        for population in COHORT_SIZES
        for contract in required_contracts
    }
    if len(keys) != len(frame) or keys != expected_keys:
        raise VerificationError(
            f"{label} does not contain the exact population/contract grid"
        )
    ranges = _require_probability_metrics(
        frame, label, balanced_accuracy=True, brier=True
    )
    return {
        "rows": int(len(frame)),
        "contracts": sorted(set(contracts)),
        "populations": sorted(set(populations)),
        "metric_ranges": ranges,
    }


def _validate_incremental_effects(
    frame: pd.DataFrame, pcr_metrics: pd.DataFrame
) -> dict[str, Any]:
    label = METRIC_TABLES[4]
    columns = _required_semantics(
        frame,
        label,
        (
            "population",
            "timing",
            "arm",
            "seed",
            "clinical_contract",
            "reference_model",
            "comparison_model",
            "n",
        ),
    )
    required_literal = ("comparison", "delta_auroc", "delta_auprc", "brier_improvement")
    missing = [column for column in required_literal if column not in frame]
    if missing:
        raise VerificationError(f"{label} is missing columns: {missing}")
    populations = _normalise_population(
        frame[columns["population"]], f"{label}.population"
    )
    timings = _normalise_timing(frame[columns["timing"]], f"{label}.timing")
    arms = _normalise_arm(frame[columns["arm"]], f"{label}.arm")
    seeds = _integers(frame[columns["seed"]], f"{label}.seed")
    if set(seeds) != set(SEED_BASES):
        raise VerificationError(f"{label} must contain exactly both configured seeds")
    contracts = _strings(
        frame[columns["clinical_contract"]], f"{label}.clinical_contract"
    )
    reference_models = _normalise_models(
        frame[columns["reference_model"]], f"{label}.reference_model"
    )
    comparison_models = _normalise_models(
        frame[columns["comparison_model"]], f"{label}.comparison_model"
    )
    comparison_names = _strings(frame["comparison"], f"{label}.comparison")
    counts = _integers(frame[columns["n"]], f"{label}.n", minimum=1)
    expected_counts = np.asarray([COHORT_SIZES[value] for value in populations])
    if not np.array_equal(counts, expected_counts):
        raise VerificationError(f"{label}.n does not equal the matched cohort size")
    effects = {
        "delta_auroc": _bounded(
            frame["delta_auroc"], f"{label}.delta_auroc", lower=-1.0, upper=1.0
        ),
        "delta_auprc": _bounded(
            frame["delta_auprc"], f"{label}.delta_auprc", lower=-1.0, upper=1.0
        ),
        "brier_improvement": _bounded(
            frame["brier_improvement"],
            f"{label}.brier_improvement",
            lower=-1.0,
            upper=1.0,
        ),
    }
    keys = pd.DataFrame(
        {
            "population": populations,
            "timing": timings,
            "arm": arms,
            "seed": seeds,
            "clinical_contract": contracts,
            "comparison": comparison_names,
            "reference_model": reference_models,
            "comparison_model": comparison_models,
        }
    )
    if keys.duplicated().any():
        raise VerificationError(f"{label} repeats an incremental comparison cell")

    required_pairs = {
        ("full_808", "C", "C+M"),
        ("ftv_complete_375", "C", "C+M"),
        ("ftv_complete_375", "C+F", "C+F+M"),
    }
    for population, reference_model, comparison_model in required_pairs:
        selected = keys[
            (keys["population"] == population)
            & (keys["reference_model"] == reference_model)
            & (keys["comparison_model"] == comparison_model)
        ]
        expected_grid = {
            (timing, arm, seed)
            for timing in TIMINGS
            for arm in LOCAL_ARMS
            for seed in SEED_BASES
        }
        observed_grid = set(
            selected.loc[:, ["timing", "arm", "seed"]].itertuples(
                index=False, name=None
            )
        )
        if observed_grid != expected_grid:
            raise VerificationError(
                f"{label} does not cover {population}:{reference_model}->{comparison_model}"
            )

    metric_label = METRIC_TABLES[0]
    metric_columns = _required_semantics(
        pcr_metrics,
        metric_label,
        (
            "population",
            "timing",
            "arm",
            "seed",
            "clinical_contract",
            "model",
        ),
    )
    lookup_frame = pcr_metrics.copy()
    lookup_frame["__population"] = _normalise_population(
        pcr_metrics[metric_columns["population"]], f"{metric_label}.population"
    )
    lookup_frame["__timing"] = _normalise_timing(
        pcr_metrics[metric_columns["timing"]], f"{metric_label}.timing"
    )
    lookup_frame["__arm"] = _normalise_arm(
        pcr_metrics[metric_columns["arm"]], f"{metric_label}.arm"
    )
    lookup_frame["__seed"] = _integers(
        pcr_metrics[metric_columns["seed"]], f"{metric_label}.seed"
    )
    lookup_frame["__contract"] = _strings(
        pcr_metrics[metric_columns["clinical_contract"]],
        f"{metric_label}.clinical_contract",
    )
    lookup_frame["__model"] = _normalise_models(
        pcr_metrics[metric_columns["model"]], f"{metric_label}.model"
    )
    metric_key = [
        "__population",
        "__timing",
        "__arm",
        "__seed",
        "__contract",
        "__model",
    ]
    if lookup_frame.duplicated(metric_key).any():
        raise VerificationError(f"{metric_label} repeats a model cell")
    lookup = lookup_frame.set_index(metric_key)
    for index, key_row in keys.iterrows():
        common = (
            key_row["population"],
            key_row["timing"],
            key_row["arm"],
            int(key_row["seed"]),
            key_row["clinical_contract"],
        )
        try:
            reference = lookup.loc[(*common, key_row["reference_model"])]
            comparison = lookup.loc[(*common, key_row["comparison_model"])]
        except KeyError as error:
            raise VerificationError(
                f"{label} references a missing pCR aggregate model cell"
            ) from error
        expected = {
            "delta_auroc": float(comparison["auroc"] - reference["auroc"]),
            "delta_auprc": float(comparison["auprc"] - reference["auprc"]),
            "brier_improvement": float(reference["brier"] - comparison["brier"]),
        }
        for metric_name, expected_value in expected.items():
            if not math.isclose(
                float(effects[metric_name][index]),
                expected_value,
                rel_tol=0.0,
                abs_tol=1e-10,
            ):
                raise VerificationError(
                    f"{label}.{metric_name} does not reproduce pcr_oof_metrics"
                )
    return {
        "rows": int(len(frame)),
        "paired_comparisons": sorted(
            f"{population}:{reference}->{comparison}"
            for population, reference, comparison in required_pairs
        ),
        "orientation": {
            "delta_auroc": "comparison-reference",
            "delta_auprc": "comparison-reference",
            "brier_improvement": "reference-comparison",
        },
    }


def _validate_effect_table(frame: pd.DataFrame, *, bootstrap: bool) -> dict[str, Any]:
    label = METRIC_TABLES[5 if bootstrap else 4]
    required = (
        "population",
        "timing",
        "arm",
        "seed",
        "reference_model",
        "comparison_model",
        "metric",
        "reference_value",
        "comparison_value",
        "improvement",
    )
    columns = _required_semantics(frame, label, required)
    populations = _normalise_population(
        frame[columns["population"]], f"{label}.population"
    )
    n_column = str(_column(frame, "n", label))
    patient_counts = _integers(frame[n_column], f"{label}.{n_column}", minimum=1)
    expected_patient_counts = np.asarray([COHORT_SIZES[value] for value in populations])
    if not np.array_equal(patient_counts, expected_patient_counts):
        raise VerificationError(
            f"{label} patient counts do not match the declared cohort"
        )
    timings = _normalise_timing(frame[columns["timing"]], f"{label}.timing")
    arms = _normalise_arm(frame[columns["arm"]], f"{label}.arm")
    seeds = _integers(frame[columns["seed"]], f"{label}.seed")
    if set(seeds) != set(SEED_BASES):
        raise VerificationError(f"{label} must contain exactly both configured seeds")
    reference_models = _normalise_models(
        frame[columns["reference_model"]], f"{label}.reference_model"
    )
    comparison_models = _normalise_models(
        frame[columns["comparison_model"]], f"{label}.comparison_model"
    )
    metrics = frame[columns["metric"]].map(_canonical_metric_name)
    if not set(metrics).issubset({"auroc", "auprc", "brier"}) or set(metrics) != {
        "auroc",
        "auprc",
        "brier",
    }:
        raise VerificationError(f"{label} must contain AUROC, AUPRC, and Brier effects")
    reference = _bounded(
        frame[columns["reference_value"]],
        f"{label}.reference_value",
        lower=0.0,
        upper=1.0,
    )
    comparison = _bounded(
        frame[columns["comparison_value"]],
        f"{label}.comparison_value",
        lower=0.0,
        upper=1.0,
    )
    improvement = _bounded(
        frame[columns["improvement"]],
        f"{label}.improvement",
        lower=-1.0,
        upper=1.0,
    )
    expected = np.where(
        metrics.to_numpy() == "brier", reference - comparison, comparison - reference
    )
    if not np.allclose(improvement, expected, rtol=0.0, atol=1e-10):
        raise VerificationError(
            f"{label} improvement orientation/value is inconsistent"
        )

    rows = pd.DataFrame(
        {
            "population": populations,
            "timing": timings,
            "arm": arms,
            "seed": seeds,
            "reference_model": reference_models,
            "comparison_model": comparison_models,
            "metric": metrics,
        }
    )
    if rows.duplicated().any():
        raise VerificationError(f"{label} repeats an effect cell")
    required_pairs = {
        ("full_808", "C", "C+M"),
        ("ftv_complete_375", "C", "C+M"),
        ("ftv_complete_375", "C+F", "C+F+M"),
    }
    observed = {
        (row.population, row.reference_model, row.comparison_model)
        for row in rows.itertuples(index=False)
    }
    if not required_pairs.issubset(observed):
        raise VerificationError(
            f"{label} is missing a prespecified incremental comparison"
        )
    for population, reference_model, comparison_model in required_pairs:
        subset = rows[
            (rows["population"] == population)
            & (rows["reference_model"] == reference_model)
            & (rows["comparison_model"] == comparison_model)
        ]
        expected_grid = {
            (timing, arm, seed, metric)
            for timing in TIMINGS
            for arm in LOCAL_ARMS
            for seed in SEED_BASES
            for metric in ("auroc", "auprc", "brier")
        }
        observed_grid = set(
            subset.loc[:, ["timing", "arm", "seed", "metric"]].itertuples(
                index=False, name=None
            )
        )
        if observed_grid != expected_grid:
            raise VerificationError(
                f"{label} comparison {population}:{reference_model}->{comparison_model} "
                "does not cover all timing/arm/seed/metric cells"
            )

    evidence: dict[str, Any] = {
        "rows": int(len(frame)),
        "orientation": {
            "auroc": "comparison-reference",
            "auprc": "comparison-reference",
            "brier": "reference-comparison",
        },
    }
    if bootstrap:
        extra = _required_semantics(
            frame,
            label,
            (
                "ci_lower",
                "ci_upper",
                "bootstrap_replicates",
                "valid_bootstrap",
                "bootstrap_unit",
            ),
        )
        lower = _bounded(
            frame[extra["ci_lower"]], f"{label}.ci_lower", lower=-1.0, upper=1.0
        )
        upper = _bounded(
            frame[extra["ci_upper"]], f"{label}.ci_upper", lower=-1.0, upper=1.0
        )
        if np.any(lower > upper):
            raise VerificationError(f"{label} contains reversed confidence intervals")
        replicates = _integers(
            frame[extra["bootstrap_replicates"]],
            f"{label}.bootstrap_replicates",
            minimum=1,
        )
        if not np.equal(replicates, BOOTSTRAP_REPLICATES).all():
            raise VerificationError(
                f"{label} must use exactly {BOOTSTRAP_REPLICATES} bootstrap replicates"
            )
        valid = _integers(
            frame[extra["valid_bootstrap"]],
            f"{label}.valid_bootstrap",
            minimum=1,
        )
        if np.any(valid > replicates):
            raise VerificationError(
                f"{label} valid replicate count exceeds requested count"
            )
        units = _strings(frame[extra["bootstrap_unit"]], f"{label}.bootstrap_unit")
        if set(units) != {BOOTSTRAP_UNIT}:
            raise VerificationError(
                f"{label} bootstrap_unit must be {BOOTSTRAP_UNIT!r}"
            )
        if (
            "n_folds" not in frame
            or not np.equal(
                _integers(frame["n_folds"], f"{label}.n_folds", minimum=1), len(FOLDS)
            ).all()
        ):
            raise VerificationError(f"{label} must report five bootstrap strata")
        if "confidence_level" not in frame or not np.allclose(
            _numeric(frame["confidence_level"], f"{label}.confidence_level"),
            0.95,
            rtol=0.0,
            atol=1e-12,
        ):
            raise VerificationError(f"{label} must report 95% confidence intervals")
        evidence.update(
            {
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_unit": BOOTSTRAP_UNIT,
                "valid_replicates_min": int(valid.min()),
            }
        )
    return evidence


def _validate_subgroup_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    label = METRIC_TABLES[6]
    columns = _validate_metric_identity_columns(
        frame, label, population=False, subgroup=True
    )
    subgroups = frame[columns["subgroup"]].map(_normalise_subgroup)
    if set(subgroups) != set(EXPECTED_SUBGROUP_SIZES):
        raise VerificationError(
            f"{label} does not contain the three prespecified subgroups"
        )
    counts = _integers(frame[columns["n"]], f"{label}.n", minimum=1)
    expected = np.asarray([EXPECTED_SUBGROUP_SIZES[value] for value in subgroups])
    if not np.array_equal(counts, expected):
        raise VerificationError(f"{label}.n does not equal its subgroup size")
    timings = _normalise_timing(frame[columns["timing"]], f"{label}.timing")
    arms = _normalise_arm(frame[columns["arm"]], f"{label}.arm")
    seeds = _integers(frame[columns["seed"]], f"{label}.seed")
    models = _normalise_models(frame[columns["model"]], f"{label}.model")
    keys = pd.DataFrame(
        {
            "subgroup": subgroups,
            "timing": timings,
            "arm": arms,
            "seed": seeds,
            "model": models,
        }
    )
    expected_models = {"REMAINING_CLINICAL", "M", "REMAINING_CLINICAL+M"}
    expected_grid = {
        (subgroup, timing, arm, seed, model)
        for subgroup in EXPECTED_SUBGROUP_SIZES
        for timing in TIMINGS
        for arm in LOCAL_ARMS
        for seed in SEED_BASES
        for model in expected_models
    }
    if (
        keys.duplicated().any()
        or set(keys.itertuples(index=False, name=None)) != expected_grid
    ):
        raise VerificationError(f"{label} does not cover the exact subgroup model grid")
    ranges = _require_probability_metrics(
        frame, label, balanced_accuracy=True, brier=True
    )
    return {"rows": int(len(frame)), "metric_ranges": ranges}


def _validate_clinical_residual_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    label = METRIC_TABLES[7]
    columns = _required_semantics(
        frame, label, ("population", "timing", "arm", "seed", "n")
    )
    populations = _normalise_population(
        frame[columns["population"]], f"{label}.population"
    )
    timings = _normalise_timing(frame[columns["timing"]], f"{label}.timing")
    arms = _normalise_arm(frame[columns["arm"]], f"{label}.arm")
    seeds = _integers(frame[columns["seed"]], f"{label}.seed")
    counts = _integers(frame[columns["n"]], f"{label}.n", minimum=1)
    if set(seeds) != set(SEED_BASES):
        raise VerificationError(f"{label} must contain exactly both configured seeds")
    expected_counts = np.asarray([COHORT_SIZES[value] for value in populations])
    if not np.array_equal(counts, expected_counts):
        raise VerificationError(f"{label}.n does not equal its declared cohort size")
    keys = pd.DataFrame(
        {"population": populations, "timing": timings, "arm": arms, "seed": seeds}
    )
    expected_grid = {
        (population, timing, arm, seed)
        for population in COHORT_SIZES
        for timing in TIMINGS
        for arm in LOCAL_ARMS
        for seed in SEED_BASES
    }
    if (
        keys.duplicated().any()
        or set(keys.itertuples(index=False, name=None)) != expected_grid
    ):
        raise VerificationError(
            f"{label} does not cover the exact residual-analysis grid"
        )
    required_metrics = (
        "mse",
        "r2",
        "pearson",
        "spearman",
        "clinical_auroc",
        "corrected_auroc",
        "delta_auroc",
        "clinical_auprc",
        "corrected_auprc",
        "delta_auprc",
        "clinical_brier",
        "corrected_brier",
        "brier_improvement",
    )
    missing = [column for column in required_metrics if column not in frame]
    if missing:
        raise VerificationError(f"{label} is missing columns: {missing}")
    ranges: dict[str, tuple[float, float]] = {}
    mse = _numeric(frame["mse"], f"{label}.mse")
    if np.any(mse < 0.0):
        raise VerificationError(f"{label}.mse must be non-negative")
    ranges["mse"] = (float(mse.min()), float(mse.max()))
    r2 = _numeric(frame["r2"], f"{label}.r2")
    if np.any(r2 > 1.0):
        raise VerificationError(f"{label}.r2 may not exceed one")
    ranges["r2"] = (float(r2.min()), float(r2.max()))
    for column in ("pearson", "spearman"):
        values = _bounded(frame[column], f"{label}.{column}", lower=-1.0, upper=1.0)
        ranges[column] = (float(values.min()), float(values.max()))
    for column in (
        "clinical_auroc",
        "corrected_auroc",
        "clinical_auprc",
        "corrected_auprc",
        "clinical_brier",
        "corrected_brier",
    ):
        values = _bounded(frame[column], f"{label}.{column}", lower=0.0, upper=1.0)
        ranges[column] = (float(values.min()), float(values.max()))
    for column in ("delta_auroc", "delta_auprc", "brier_improvement"):
        values = _bounded(frame[column], f"{label}.{column}", lower=-1.0, upper=1.0)
        ranges[column] = (float(values.min()), float(values.max()))
    if not np.allclose(
        frame["delta_auroc"].to_numpy(float),
        frame["corrected_auroc"].to_numpy(float)
        - frame["clinical_auroc"].to_numpy(float),
        rtol=0.0,
        atol=1e-10,
    ):
        raise VerificationError(f"{label}.delta_auroc has the wrong orientation")
    if not np.allclose(
        frame["delta_auprc"].to_numpy(float),
        frame["corrected_auprc"].to_numpy(float)
        - frame["clinical_auprc"].to_numpy(float),
        rtol=0.0,
        atol=1e-10,
    ):
        raise VerificationError(f"{label}.delta_auprc has the wrong orientation")
    if not np.allclose(
        frame["brier_improvement"].to_numpy(float),
        frame["clinical_brier"].to_numpy(float)
        - frame["corrected_brier"].to_numpy(float),
        rtol=0.0,
        atol=1e-10,
    ):
        raise VerificationError(f"{label}.brier_improvement has the wrong orientation")
    return {"rows": int(len(frame)), "metric_ranges": ranges, "identity": columns}


def _validate_cohort_summary(frame: pd.DataFrame) -> dict[str, Any]:
    label = METRIC_TABLES[8]
    columns = _required_semantics(frame, label, ("population", "n"))
    populations = _strings(
        frame[columns["population"]], f"{label}.population"
    ).str.lower()
    counts = _integers(frame[columns["n"]], f"{label}.n", minimum=1)
    observed: dict[str, int] = {}
    for population, count in zip(populations, counts, strict=True):
        if population in observed and observed[population] != int(count):
            raise VerificationError(f"{label} reports inconsistent n for {population}")
        observed[population] = int(count)
    expected_cohorts = {**dict(COHORT_SIZES), "ftv_unavailable_433": 433}
    if observed != expected_cohorts:
        raise VerificationError(
            f"{label} must report full_808=808, ftv_complete_375=375, and "
            "ftv_unavailable_433=433"
        )
    for column in (
        "pcr_positive",
        "hr_positive",
        "her2_positive",
        "mp1",
        "age_missing",
        "race_missing",
        "menopause_missing",
    ):
        if column in frame:
            values = _integers(frame[column], f"{label}.{column}", minimum=0)
            if np.any(values > counts):
                raise VerificationError(f"{label}.{column} exceeds cohort n")
    if "pcr_prevalence" not in frame:
        raise VerificationError(f"{label} is missing pcr_prevalence")
    prevalence = _bounded(
        frame["pcr_prevalence"], f"{label}.pcr_prevalence", lower=0.0, upper=1.0
    )
    if "pcr_positive" in frame and not np.allclose(
        prevalence,
        frame["pcr_positive"].to_numpy(float) / counts,
        rtol=0.0,
        atol=1e-10,
    ):
        raise VerificationError(f"{label}.pcr_prevalence disagrees with pcr_positive/n")
    return {"cohorts": observed, "rows": int(len(frame))}


def _validate_hyperparameters(frame: pd.DataFrame) -> dict[str, Any]:
    label = METRIC_TABLES[9]
    required = {
        "task",
        "fold",
        "selected_C",
        "selected_alpha",
        "validation_auroc",
        "validation_mse",
        "validation_threshold",
        "train_rows",
        "validation_rows",
        "feature_dim",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise VerificationError(f"{label} is missing columns: {missing}")
    tasks = _strings(frame["task"], f"{label}.task")
    expected_tasks = {
        "profile_binary",
        "profile_multiclass",
        "clinical_baseline",
        "pcr",
        "clinical_error_ridge",
        "subgroup",
    }
    if set(tasks) != expected_tasks:
        raise VerificationError(
            f"{label} does not contain the exact model-selection tasks"
        )
    folds = _integers(frame["fold"], f"{label}.fold")
    if set(folds) != set(FOLDS):
        raise VerificationError(f"{label} does not cover all five folds")
    for column in ("train_rows", "validation_rows", "feature_dim"):
        _integers(frame[column], f"{label}.{column}", minimum=1)

    selected_c = pd.to_numeric(frame["selected_C"], errors="coerce").to_numpy(float)
    selected_alpha = pd.to_numeric(frame["selected_alpha"], errors="coerce").to_numpy(
        float
    )
    validation_auroc = pd.to_numeric(
        frame["validation_auroc"], errors="coerce"
    ).to_numpy(float)
    validation_mse = pd.to_numeric(frame["validation_mse"], errors="coerce").to_numpy(
        float
    )
    threshold = pd.to_numeric(frame["validation_threshold"], errors="coerce").to_numpy(
        float
    )
    ridge_rows = tasks.eq("clinical_error_ridge").to_numpy()
    multiclass_rows = tasks.eq("profile_multiclass").to_numpy()
    binary_rows = ~(ridge_rows | multiclass_rows)
    logistic_rows = ~ridge_rows
    if (
        not np.isfinite(selected_c[logistic_rows]).all()
        or np.any(selected_c[logistic_rows] <= 0.0)
        or np.isfinite(selected_c[ridge_rows]).any()
    ):
        raise VerificationError(f"{label}.selected_C violates task applicability")
    if (
        not np.isfinite(selected_alpha[ridge_rows]).all()
        or np.any(selected_alpha[ridge_rows] <= 0.0)
        or np.isfinite(selected_alpha[logistic_rows]).any()
    ):
        raise VerificationError(f"{label}.selected_alpha violates task applicability")
    if (
        not np.isfinite(validation_auroc[logistic_rows]).all()
        or np.any(
            (validation_auroc[logistic_rows] < 0.0)
            | (validation_auroc[logistic_rows] > 1.0)
        )
        or np.isfinite(validation_auroc[ridge_rows]).any()
    ):
        raise VerificationError(f"{label}.validation_auroc violates task applicability")
    if (
        not np.isfinite(validation_mse[ridge_rows]).all()
        or np.any(validation_mse[ridge_rows] < 0.0)
        or np.isfinite(validation_mse[logistic_rows]).any()
    ):
        raise VerificationError(f"{label}.validation_mse violates task applicability")
    if (
        not np.isfinite(threshold[binary_rows]).all()
        or np.any((threshold[binary_rows] < 0.0) | (threshold[binary_rows] > 1.0))
        or np.isfinite(threshold[ridge_rows | multiclass_rows]).any()
    ):
        raise VerificationError(
            f"{label}.validation_threshold violates task applicability"
        )

    task_keys: Mapping[str, tuple[str, ...]] = {
        "profile_binary": ("fold", "seed", "arm", "view", "target"),
        "profile_multiclass": ("fold", "seed", "arm", "view", "target"),
        "clinical_baseline": ("fold", "population", "clinical_contract"),
        "pcr": (
            "fold",
            "population",
            "seed",
            "arm",
            "timing",
            "model",
            "clinical_contract",
        ),
        "clinical_error_ridge": (
            "fold",
            "population",
            "seed",
            "arm",
            "timing",
            "model",
            "clinical_contract",
        ),
        "subgroup": ("fold", "seed", "arm", "timing", "subgroup", "model"),
    }
    coverage: dict[str, int] = {}
    for task, keys in task_keys.items():
        subset = frame.loc[tasks.eq(task)]
        missing_keys = [column for column in keys if column not in frame]
        if missing_keys:
            raise VerificationError(
                f"{label} task {task} misses identity columns {missing_keys}"
            )
        for column in keys:
            _strings(subset[column], f"{label}.{task}.{column}")
        if subset.duplicated(list(keys)).any():
            raise VerificationError(
                f"{label} task {task} repeats a fold selection cell"
            )
        if "seed" in keys:
            seed_values = _integers(subset["seed"], f"{label}.{task}.seed")
            if set(seed_values) != set(SEED_BASES):
                raise VerificationError(f"{label} task {task} misses a configured seed")
        if "arm" in keys:
            arm_values = _normalise_arm(subset["arm"], f"{label}.{task}.arm")
            if set(arm_values) != set(LOCAL_ARMS):
                raise VerificationError(f"{label} task {task} misses a LOCAL arm")
        if "timing" in keys:
            timing_values = _normalise_timing(
                subset["timing"], f"{label}.{task}.timing"
            )
            if set(timing_values) != set(TIMINGS):
                raise VerificationError(f"{label} task {task} misses a timing")
        coverage[task] = int(len(subset))
    return {
        "rows": int(len(frame)),
        "task_rows": coverage,
        "selection_columns": [
            "selected_C",
            "selected_alpha",
            "validation_auroc",
            "validation_mse",
            "validation_threshold",
        ],
    }


def _resolve_manifest_path(
    value: Any, *, experiment_root: Path, repo_root: Path, figure: bool
) -> Path:
    text = str(value).strip()
    if not text:
        raise VerificationError("manifest contains an empty path")
    candidate = Path(text)
    if candidate.is_absolute():
        if figure:
            raise VerificationError("figure manifest paths must be repository-relative")
        return candidate.resolve()
    if ".." in candidate.parts:
        raise VerificationError("manifest path may not contain '..'")
    if candidate.parts and candidate.parts[0] == "additional_experiments":
        resolved = (repo_root / candidate).resolve()
    else:
        resolved = (experiment_root / candidate).resolve()
    if figure and experiment_root not in resolved.parents:
        raise VerificationError("figure manifest path escapes the experiment root")
    return resolved


def _validate_manifest(
    frame: pd.DataFrame,
    *,
    label: str,
    experiment_root: Path,
    repo_root: Path,
    figure: bool,
) -> dict[str, Any]:
    columns = _required_semantics(frame, label, ("artifact", "path", "sha256"))
    artifacts = _strings(frame[columns["artifact"]], f"{label}.artifact")
    paths = _strings(frame[columns["path"]], f"{label}.path")
    digests = _strings(frame[columns["sha256"]], f"{label}.sha256").str.lower()
    if not digests.map(lambda value: bool(SHA256_RE.fullmatch(value))).all():
        raise VerificationError(f"{label}.sha256 contains a malformed digest")
    if paths.duplicated().any():
        raise VerificationError(f"{label} contains duplicate artifact paths")
    if figure and artifacts.duplicated().any():
        raise VerificationError(f"{label} contains duplicate figure identifiers")
    resolved_paths: list[str] = []
    for path_text, expected_digest in zip(paths, digests, strict=True):
        resolved = _resolve_manifest_path(
            path_text,
            experiment_root=experiment_root,
            repo_root=repo_root,
            figure=figure,
        )
        _require_regular_file(resolved, f"{label} referenced artifact")
        if _sha256(resolved) != expected_digest:
            raise VerificationError(f"{label} contains an artifact SHA-256 mismatch")
        if "size_bytes" in frame:
            declared_size = _integers(
                frame.loc[[frame.index[len(resolved_paths)]], "size_bytes"],
                f"{label}.size_bytes",
                minimum=1,
            )[0]
            if int(declared_size) != resolved.stat().st_size:
                raise VerificationError(f"{label} contains an artifact size mismatch")
        if figure:
            resolved_paths.append(resolved.relative_to(experiment_root).as_posix())
        else:
            resolved_paths.append(path_text)
    if not figure:
        kinds = list(artifacts)
        expected_base = {
            "clinical_labels",
            "fold_manifest",
            "ftv_transition_table",
            "local_preregistration_lock",
        }
        if (
            not expected_base.issubset(set(kinds))
            or kinds.count("local_response_state") != 20
        ):
            raise VerificationError(
                f"{label} must contain four pinned base inputs and 20 LOCAL response states"
            )
        for column in ("seed", "arm", "fold"):
            if column not in frame:
                raise VerificationError(
                    f"{label} is missing LOCAL identity column {column!r}"
                )
        local_rows = frame.loc[artifacts.eq("local_response_state")]
        local_seeds = _integers(local_rows["seed"], f"{label}.seed")
        local_arms = _normalise_arm(local_rows["arm"], f"{label}.arm")
        local_folds = _integers(local_rows["fold"], f"{label}.fold")
        observed_grid = set(zip(local_seeds, local_arms, local_folds, strict=True))
        expected_grid = {
            (seed, arm, fold)
            for seed in SEED_BASES
            for arm in LOCAL_ARMS
            for fold in FOLDS
        }
        if observed_grid != expected_grid:
            raise VerificationError(f"{label} does not cover the exact 20 LOCAL cells")
    return {
        "rows": int(len(frame)),
        "verified_paths": resolved_paths,
    }


def _repo_root(experiment_root: Path) -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=experiment_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise VerificationError("experiment root is not inside a Git repository")
    return Path(result.stdout.strip()).resolve()


def _git_is_ignored(repo_root: Path, path: Path) -> bool:
    try:
        relative = path.resolve().relative_to(repo_root).as_posix()
    except ValueError as error:
        raise VerificationError(
            "ignore-check path is outside the Git repository"
        ) from error
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", relative],
        cwd=repo_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise VerificationError("git check-ignore failed")
    return result.returncode == 0


def _privacy_scan(experiment_root: Path, output_path: Path) -> dict[str, Any]:
    scanned = 0
    findings: list[dict[str, Any]] = []
    symlinks: list[str] = []
    for path in sorted(experiment_root.rglob("*")):
        relative = path.relative_to(experiment_root)
        if not relative.parts or relative.parts[0] in ALLOWED_IDENTIFIER_ROOTS:
            continue
        if path.resolve() == output_path.resolve():
            continue
        if path.is_symlink():
            symlinks.append(relative.as_posix())
            continue
        if not path.is_file():
            continue
        scanned += 1
        matches = PATIENT_ID_BYTES_RE.findall(path.read_bytes())
        if matches:
            findings.append(
                {
                    "path": relative.as_posix(),
                    "patient_identifier_occurrences": len(matches),
                }
            )
    if symlinks:
        raise VerificationError(
            f"public audit tree contains symbolic links: {symlinks}"
        )
    if findings:
        raise VerificationError(
            "canonical patient identifiers occur outside features/predictions/logs: "
            + json.dumps(findings, sort_keys=True)
        )
    return {
        "scanned_public_files": scanned,
        "allowed_private_roots": sorted(ALLOWED_IDENTIFIER_ROOTS),
        "identifier_pattern": "canonical ISPY2/ACRIN patient keys",
        "findings": 0,
    }


def _ignore_contract(
    *,
    experiment_root: Path,
    repo_root: Path,
    figure_paths: Sequence[str],
) -> dict[str, Any]:
    private_paths = [experiment_root / relative for relative in PRIVATE_TABLES]
    for root_name in sorted(ALLOWED_IDENTIFIER_ROOTS):
        private_root = experiment_root / root_name
        if private_root.is_dir():
            private_paths.extend(
                path
                for path in private_root.rglob("*")
                if path.is_file() and path.name != ".gitkeep"
            )
    private_paths = sorted(set(private_paths))
    incorrectly_public = [
        path.relative_to(experiment_root).as_posix()
        for path in private_paths
        if not _git_is_ignored(repo_root, path)
    ]
    if incorrectly_public:
        raise VerificationError(
            f"private prediction outputs are not ignored: {incorrectly_public}"
        )
    public_relatives = list(METRIC_TABLES) + list(REPORTS) + [VERIFICATION_RELATIVE]
    public_relatives.extend(figure_paths)
    incorrectly_ignored = [
        relative
        for relative in sorted(set(public_relatives))
        if _git_is_ignored(repo_root, experiment_root / relative)
    ]
    if incorrectly_ignored:
        raise VerificationError(
            f"aggregate/report/figure outputs are ignored: {incorrectly_ignored}"
        )
    return {
        "private_outputs_ignored": len(private_paths),
        "public_outputs_not_ignored": len(set(public_relatives)),
        "command": "git check-ignore",
    }


def _report_contract(experiment_root: Path) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for relative in REPORTS:
        path = _require_regular_file(experiment_root / relative, relative)
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise VerificationError(f"{relative} is not readable UTF-8") from error
        if len(text.strip()) < 200:
            raise VerificationError(f"{relative} is unexpectedly short")
        evidence[relative] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    final_text = (experiment_root / REPORTS[1]).read_text(encoding="utf-8").lower()
    required_concepts = {
        "diagnostic/exploratory": ("diagnostic", "exploratory"),
        "two-seed limitation": ("two-seed", "two seed", "2-seed", "2 seed"),
        "late T3 boundary": ("t3", "pre-surgery"),
        "matched comparison": ("matched", "paired"),
    }
    missing = [
        concept
        for concept, needles in required_concepts.items()
        if not any(needle in final_text for needle in needles)
    ]
    if missing:
        raise VerificationError(
            f"final report omits required interpretation concepts: {missing}"
        )
    return evidence


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verify(experiment_root: Path) -> dict[str, Any]:
    experiment_root = experiment_root.expanduser().resolve()
    if not experiment_root.is_dir():
        raise VerificationError("experiment root does not exist or is not a directory")
    repo_root = _repo_root(experiment_root)
    output_path = experiment_root / VERIFICATION_RELATIVE
    collector = CheckCollector()
    frames: dict[str, pd.DataFrame] = {}

    for relative in PRIVATE_TABLES + METRIC_TABLES:

        def load(relative: str = relative) -> dict[str, Any]:
            frame = _read_csv(experiment_root / relative, relative)
            frames[relative] = frame
            return {"rows": int(len(frame)), "columns": list(frame.columns)}

        collector.run(f"artifact:{relative}", load)

    collector.run(
        "reports:inventory_and_final", lambda: _report_contract(experiment_root)
    )

    schema_requirements: Mapping[str, tuple[str, ...]] = {
        PRIVATE_TABLES[0]: (
            "patient",
            "population",
            "timing",
            "arm",
            "seed",
            "fold",
            "model",
            "label",
            "probability",
        ),
        PRIVATE_TABLES[1]: (
            "patient",
            "view",
            "arm",
            "seed",
            "fold",
            "target",
            "label",
            "prediction",
        ),
        PRIVATE_TABLES[2]: (
            "patient",
            "timing",
            "arm",
            "seed",
            "fold",
            "model",
            "subgroup",
            "label",
            "probability",
        ),
        METRIC_TABLES[0]: ("population", "timing", "arm", "seed", "model", "n"),
        METRIC_TABLES[1]: ("population", "timing", "arm", "seed", "fold", "model", "n"),
        METRIC_TABLES[2]: ("view", "arm", "seed", "target", "n"),
        METRIC_TABLES[3]: ("population", "clinical_contract", "n"),
        METRIC_TABLES[4]: (
            "population",
            "timing",
            "arm",
            "seed",
            "clinical_contract",
            "reference_model",
            "comparison_model",
            "n",
        ),
        METRIC_TABLES[5]: (
            "population",
            "timing",
            "arm",
            "seed",
            "reference_model",
            "comparison_model",
            "metric",
            "reference_value",
            "comparison_value",
            "improvement",
            "ci_lower",
            "ci_upper",
            "bootstrap_replicates",
            "valid_bootstrap",
            "bootstrap_unit",
        ),
        METRIC_TABLES[6]: ("timing", "arm", "seed", "model", "subgroup", "n"),
        METRIC_TABLES[7]: ("population", "timing", "arm", "seed", "n"),
        METRIC_TABLES[8]: ("population", "n"),
        METRIC_TABLES[9]: ("analysis", "fold"),
        METRIC_TABLES[10]: ("artifact", "path", "sha256"),
        METRIC_TABLES[11]: ("artifact", "path", "sha256"),
    }
    for relative, required in schema_requirements.items():

        def schema(
            relative: str = relative, required: tuple[str, ...] = required
        ) -> Any:
            if relative not in frames:
                raise VerificationError(
                    f"{relative} was unavailable for schema validation"
                )
            return _schema_evidence(frames[relative], relative, required)

        collector.run(f"schema:{relative}", schema)

    validation_functions: Sequence[tuple[str, str, Callable[[pd.DataFrame], Any]]] = (
        ("oof:pcr_pairing_and_coverage", PRIVATE_TABLES[0], _validate_pcr_oof),
        ("oof:profile_coverage", PRIVATE_TABLES[1], _validate_profile_oof),
        (
            "oof:subgroup_pairing_and_coverage",
            PRIVATE_TABLES[2],
            _validate_subgroup_oof,
        ),
        (
            "metrics:pcr_oof",
            METRIC_TABLES[0],
            lambda frame: _validate_pcr_metrics(frame, folds=False),
        ),
        (
            "metrics:pcr_folds",
            METRIC_TABLES[1],
            lambda frame: _validate_pcr_metrics(frame, folds=True),
        ),
        ("metrics:profile", METRIC_TABLES[2], _validate_profile_metrics),
        ("metrics:clinical_baselines", METRIC_TABLES[3], _validate_clinical_baselines),
        (
            "metrics:bootstrap",
            METRIC_TABLES[5],
            lambda frame: _validate_effect_table(frame, bootstrap=True),
        ),
        ("metrics:subgroups", METRIC_TABLES[6], _validate_subgroup_metrics),
        (
            "metrics:clinical_residual",
            METRIC_TABLES[7],
            _validate_clinical_residual_metrics,
        ),
        ("metrics:cohort_summary", METRIC_TABLES[8], _validate_cohort_summary),
        ("metrics:hyperparameters", METRIC_TABLES[9], _validate_hyperparameters),
    )
    for check_name, relative, function in validation_functions:

        def validate(
            relative: str = relative, function: Callable[[pd.DataFrame], Any] = function
        ) -> Any:
            if relative not in frames:
                raise VerificationError(
                    f"{relative} was unavailable for content validation"
                )
            return function(frames[relative])

        collector.run(check_name, validate)

    def validate_incremental() -> Any:
        relative = METRIC_TABLES[4]
        pcr_relative = METRIC_TABLES[0]
        if relative not in frames or pcr_relative not in frames:
            raise VerificationError(
                "incremental_effects.csv and pcr_oof_metrics.csv are both required"
            )
        return _validate_incremental_effects(frames[relative], frames[pcr_relative])

    collector.run("metrics:incremental_effects", validate_incremental)

    input_manifest_evidence: dict[str, Any] = {}
    figure_manifest_evidence: dict[str, Any] = {}

    def validate_input_manifest() -> Any:
        relative = METRIC_TABLES[10]
        if relative not in frames:
            raise VerificationError(
                f"{relative} was unavailable for content validation"
            )
        evidence = _validate_manifest(
            frames[relative],
            label=relative,
            experiment_root=experiment_root,
            repo_root=repo_root,
            figure=False,
        )
        input_manifest_evidence.update(evidence)
        return evidence

    def validate_figure_manifest() -> Any:
        relative = METRIC_TABLES[11]
        if relative not in frames:
            raise VerificationError(
                f"{relative} was unavailable for content validation"
            )
        evidence = _validate_manifest(
            frames[relative],
            label=relative,
            experiment_root=experiment_root,
            repo_root=repo_root,
            figure=True,
        )
        figure_manifest_evidence.update(evidence)
        return evidence

    collector.run("manifest:inputs", validate_input_manifest)
    collector.run("manifest:figures", validate_figure_manifest)
    collector.run(
        "privacy:no_public_patient_identifiers",
        lambda: _privacy_scan(experiment_root, output_path),
    )
    collector.run(
        "git:private_ignored_public_visible",
        lambda: _ignore_contract(
            experiment_root=experiment_root,
            repo_root=repo_root,
            figure_paths=figure_manifest_evidence.get("verified_paths", []),
        ),
    )

    failed = [result.name for result in collector.results if not result.passed]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if collector.passed else "FAIL",
        "experiment": "mri_clinical_complementarity_audit",
        "experiment_root": experiment_root.name,
        "checks": [
            {
                "name": result.name,
                "passed": result.passed,
                "evidence": result.evidence,
            }
            for result in collector.results
        ],
        "summary": {
            "checks_total": len(collector.results),
            "checks_passed": sum(result.passed for result in collector.results),
            "checks_failed": len(failed),
            "failed_checks": failed,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Audit root (default: directory containing this script's parent)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    experiment_root = arguments.experiment_root.expanduser().resolve()
    output_path = experiment_root / VERIFICATION_RELATIVE
    try:
        payload = verify(experiment_root)
    except Exception as error:
        message = " ".join(str(error).split())
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "FAIL",
            "experiment": "mri_clinical_complementarity_audit",
            "experiment_root": experiment_root.name,
            "checks": [],
            "summary": {
                "checks_total": 0,
                "checks_passed": 0,
                "checks_failed": 1,
                "failed_checks": ["verifier_initialization"],
            },
            "fatal_error": {"error_type": type(error).__name__, "message": message},
        }
    try:
        _atomic_json(output_path, payload)
    except Exception as error:
        print(f"failed to write {VERIFICATION_RELATIVE}: {error}")
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Aggregate the frozen LOCAL MRI reference on the exact 375-patient overlap.

The source OOF files are private, patient-level artifacts from the completed
MRI--clinical complementarity audit.  This module reads them without modifying
or copying them, verifies their schema, labels, held-out folds, and exact trial
ID overlap, then writes only aggregate sensitivity summaries.

Metrics are first computed independently for each ``seed x arm`` cell.  The
four cell-level metric values are then summarized by an unweighted mean, min,
and max.  Patient rows are never pooled across cells because every cell covers
the same people.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import itertools
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from data_contracts import (
    EXPERIMENT_ROOT,
    canonical_mri_trial_id,
    load_config,
    load_primary_cohort,
    make_mri_matched_splits,
    sha256_file,
)


PCR_COLUMNS = (
    "patient_id",
    "fold",
    "population",
    "seed",
    "arm",
    "timing",
    "model",
    "clinical_contract",
    "y_true",
    "predicted_probability",
    "predicted_label",
    "threshold",
)

PROFILE_COLUMNS = (
    "patient_id",
    "fold",
    "seed",
    "arm",
    "view",
    "target",
    "y_true",
    "predicted_probability",
    "predicted_label",
    "threshold",
    "prob_hr_pos_her2_neg",
    "prob_hr_neg_her2_neg",
    "prob_hr_pos_her2_pos",
    "prob_hr_neg_her2_pos",
)

PCR_POPULATION = "ftv_complete_375"
PCR_MODELS = ("C", "M", "C+M", "C+F", "C+F+M")
PCR_TIMINGS = ("T0", "T1", "T2", "T3")
MRI_CLINICAL_CONTRACT = "C2_full_with_treatment"

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

SEEDS = (2026, 3026)
ARMS = ("LOCAL0", "LOCAL3")
SENSITIVITY_CELLS = tuple(itertools.product(SEEDS, ARMS))

TIMING_LABELS = {
    "T0": "T0",
    "T1": "T1",
    "T2": "T2",
    "T3": "T3 (late/pre-surgery)",
}

PROFILE_VIEW_LABELS = {
    "T0": "T0",
    "T1": "T1",
    "T2": "T2",
    "T3": "T3 (late/pre-surgery)",
    "long_T0_T1": "Long T0-T1",
    "long_T0_T2": "Long T0-T2",
    "long_T0_T3": "Long T0-T3 (includes late/pre-surgery)",
}

SUBTYPE_PROBABILITY_COLUMNS = {
    "HR+/HER2-": "prob_hr_pos_her2_neg",
    "HR-/HER2-": "prob_hr_neg_her2_neg",
    "HR+/HER2+": "prob_hr_pos_her2_pos",
    "HR-/HER2+": "prob_hr_neg_her2_pos",
}
# sklearn's multiclass ROC implementation requires ordered labels.
SUBTYPE_CLASSES = tuple(sorted(SUBTYPE_PROBABILITY_COLUMNS))

PCR_OUTPUT_COLUMNS = (
    "population",
    "target",
    "timing",
    "timing_label",
    "model",
    "model_role",
    "clinical_contract",
    "n_sensitivity_cells",
    "n_patients_per_cell",
    "n_positive_per_cell",
    "n_negative_per_cell",
    "auroc_mean",
    "auroc_min",
    "auroc_max",
    "auprc_mean",
    "auprc_min",
    "auprc_max",
    "balanced_accuracy_mean",
    "balanced_accuracy_min",
    "balanced_accuracy_max",
    "brier_mean",
    "brier_min",
    "brier_max",
    "aggregation",
)

PROFILE_OUTPUT_COLUMNS = (
    "population",
    "target",
    "metric_definition",
    "view",
    "timing_label",
    "n_sensitivity_cells",
    "n_patients_per_cell",
    "n_positive_per_cell",
    "n_classes",
    "auroc_mean",
    "auroc_min",
    "auroc_max",
    "auprc_mean",
    "auprc_min",
    "auprc_max",
    "balanced_accuracy_mean",
    "balanced_accuracy_min",
    "balanced_accuracy_max",
    "aggregation",
)

AGGREGATION_DESCRIPTION = (
    "unweighted mean/min/max of four seed-x-arm metrics; patient rows not pooled"
)


@dataclass(frozen=True)
class MatchedCohortContract:
    """Non-persisted patient-level contract for validating private OOF rows."""

    frame: pd.DataFrame
    test_fold_by_trial_id: Mapping[str, int]
    primary_provenance: Mapping[str, Any]

    @property
    def trial_ids(self) -> frozenset[str]:
        return frozenset(self.frame["trial_id"].astype(str))


@dataclass(frozen=True)
class ReferenceResult:
    pcr_metrics: pd.DataFrame
    profile_metrics: pd.DataFrame
    provenance: Mapping[str, Any]


def _require_exact_columns(
    frame: pd.DataFrame, expected: Sequence[str], label: str
) -> None:
    observed = tuple(str(column) for column in frame.columns)
    if observed != tuple(expected):
        raise ValueError(
            f"{label} schema/order mismatch: expected {list(expected)}, "
            f"observed {list(observed)}"
        )


def _strict_string(series: pd.Series, label: str) -> pd.Series:
    values = series.astype("string")
    if values.isna().any():
        raise ValueError(f"{label} contains missing strings")
    stripped = values.str.strip()
    if stripped.eq("").any() or not stripped.equals(values):
        raise ValueError(f"{label} contains blank or padded strings")
    return stripped.astype(str)


def _integer_series(
    series: pd.Series, label: str, allowed: Iterable[int] | None = None
) -> pd.Series:
    try:
        numeric = pd.to_numeric(series, errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain integers") from error
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{label} must contain finite integers")
    result = pd.Series(numeric.astype(np.int64), index=series.index, name=series.name)
    if allowed is not None and not set(result).issubset(set(int(value) for value in allowed)):
        raise ValueError(f"{label} contains values outside {sorted(set(allowed))}")
    return result


def _binary_series(series: pd.Series, label: str) -> pd.Series:
    result = _integer_series(series, label)
    if not result.isin((0, 1)).all():
        raise ValueError(f"{label} must contain only 0/1")
    return result.astype(np.int8)


def _probability_series(series: pd.Series, label: str) -> pd.Series:
    try:
        numeric = pd.to_numeric(series, errors="raise").astype(np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain probabilities") from error
    values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError(f"{label} must contain finite probabilities in [0,1]")
    return numeric


def _normalize_prediction_ids(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """Normalize only the two exact MRI prefixes to six-digit trial IDs."""

    output = frame.copy()
    output["patient_id"] = _strict_string(output["patient_id"], f"{label}.patient_id")
    output["trial_id"] = output["patient_id"].map(canonical_mri_trial_id)

    # A trial must not alternate between the ISPY2 and ACRIN spelling across
    # cells.  This is stronger than merely obtaining the same six-digit suffix.
    spellings = output.groupby("trial_id", sort=False)["patient_id"].nunique()
    if not spellings.eq(1).all():
        raise ValueError(f"{label} uses multiple MRI identifiers for one trial ID")
    return output


def load_matched_cohort_contract(config: Mapping[str, Any]) -> MatchedCohortContract:
    """Build the exact 375-person label and held-out-fold validation contract."""

    primary, primary_provenance = load_primary_cohort(config)
    matched, splits = make_mri_matched_splits(primary, config)
    matched = matched.copy()
    matched["trial_id"] = matched["trial_id"].astype(str)
    if len(matched) != 375 or matched["trial_id"].nunique() != 375:
        raise ValueError("MRI reference requires exactly 375 unique trial IDs")

    test_fold_by_trial_id: dict[str, int] = {}
    for split in splits:
        for index in split.test:
            trial_id = str(matched.iloc[int(index)]["trial_id"])
            if trial_id in test_fold_by_trial_id:
                raise ValueError("matched trial appears in outer test more than once")
            test_fold_by_trial_id[trial_id] = int(split.fold)
    if set(test_fold_by_trial_id) != set(matched["trial_id"]):
        raise ValueError("matched held-out-fold map does not cover all 375 trials")
    return MatchedCohortContract(
        frame=matched,
        test_fold_by_trial_id=test_fold_by_trial_id,
        primary_provenance=primary_provenance,
    )


def _validate_group_alignment(
    group: pd.DataFrame,
    contract: MatchedCohortContract,
    expected_labels: Mapping[str, Any],
    label: str,
) -> None:
    if len(group) != 375:
        raise ValueError(f"{label} must contain 375 rows, observed {len(group)}")
    if group["trial_id"].duplicated().any():
        raise ValueError(f"{label} repeats a normalized trial ID")
    observed_ids = set(group["trial_id"])
    if observed_ids != set(contract.trial_ids):
        missing = len(set(contract.trial_ids) - observed_ids)
        extra = len(observed_ids - set(contract.trial_ids))
        raise ValueError(
            f"{label} does not equal the exact MRI overlap; missing={missing}, extra={extra}"
        )

    expected_fold = group["trial_id"].map(contract.test_fold_by_trial_id).to_numpy(
        dtype=np.int64
    )
    if not np.array_equal(group["fold"].to_numpy(dtype=np.int64), expected_fold):
        raise ValueError(f"{label} fold assignments disagree with locked outer test")

    expected = group["trial_id"].map(expected_labels)
    if expected.isna().any():
        raise AssertionError(f"{label} label lookup unexpectedly failed")
    observed = group["y_true"]
    if not np.array_equal(observed.to_numpy(), expected.to_numpy()):
        raise ValueError(f"{label} labels disagree with the authoritative cohort")


def _binary_metrics(
    frame: pd.DataFrame, *, use_stored_threshold_decisions: bool
) -> dict[str, float | int]:
    labels = frame["y_true"].to_numpy(dtype=np.int64)
    probabilities = frame["predicted_probability"].to_numpy(dtype=np.float64)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("binary metric cell must contain both classes")
    if use_stored_threshold_decisions:
        predicted = frame["predicted_label"].to_numpy(dtype=np.int64)
    else:
        # This reproduces the source MRI audit's pCR aggregate convention.
        predicted = (probabilities >= 0.5).astype(np.int64)
    return {
        "n": int(len(frame)),
        "n_positive": int(labels.sum()),
        "n_negative": int((labels == 0).sum()),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "brier": float(np.mean(np.square(probabilities - labels))),
    }


def _multiclass_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    labels = frame["y_true"].astype(str).to_numpy()
    columns = [SUBTYPE_PROBABILITY_COLUMNS[label] for label in SUBTYPE_CLASSES]
    probabilities = frame.loc[:, columns].to_numpy(dtype=np.float64)
    if set(labels) != set(SUBTYPE_CLASSES):
        raise ValueError("subtype metric cell must contain all four classes")
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("subtype probabilities must be finite and in [0,1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("subtype probability rows must sum to one")
    classes = np.asarray(SUBTYPE_CLASSES)
    predicted = classes[np.argmax(probabilities, axis=1)]
    if not np.array_equal(predicted, frame["predicted_label"].astype(str).to_numpy()):
        raise ValueError("stored subtype labels disagree with probability argmax")
    indicator = label_binarize(labels, classes=classes)
    return {
        "n": int(len(frame)),
        "n_positive": -1,
        "n_classes": len(classes),
        "auroc": float(
            roc_auc_score(
                labels,
                probabilities,
                labels=classes,
                multi_class="ovr",
                average="macro",
            )
        ),
        "auprc": float(
            average_precision_score(indicator, probabilities, average="macro")
        ),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
    }


def _require_sensitivity_cells(frame: pd.DataFrame, group_label: str) -> None:
    cells = set(
        zip(
            frame["seed"].to_numpy(dtype=np.int64),
            frame["arm"].astype(str),
            strict=True,
        )
    )
    if cells != set(SENSITIVITY_CELLS):
        raise ValueError(
            f"{group_label} sensitivity cells differ; expected={SENSITIVITY_CELLS}, "
            f"observed={sorted(cells)}"
        )


def _summary_triplet(values: pd.Series, label: str) -> dict[str, float]:
    numeric = values.to_numpy(dtype=np.float64)
    if numeric.shape != (4,) or not np.isfinite(numeric).all():
        raise ValueError(f"{label} requires four finite cell metrics")
    return {
        f"{label}_mean": float(np.mean(numeric)),
        f"{label}_min": float(np.min(numeric)),
        f"{label}_max": float(np.max(numeric)),
    }


def _constant_int(group: pd.DataFrame, column: str, label: str) -> int:
    values = group[column].drop_duplicates()
    if len(values) != 1:
        raise ValueError(f"{label}.{column} differs across sensitivity cells")
    return int(values.iloc[0])


def load_pcr_predictions(path: str | Path) -> tuple[pd.DataFrame, str]:
    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = sha256_file(source)
    frame = pd.read_csv(source)
    _require_exact_columns(frame, PCR_COLUMNS, "MRI pCR predictions")
    frame = _normalize_prediction_ids(frame, "MRI pCR predictions")
    frame["fold"] = _integer_series(frame["fold"], "pCR.fold", range(5))
    frame["seed"] = _integer_series(frame["seed"], "pCR.seed", SEEDS)
    for column in ("population", "arm", "timing", "model", "clinical_contract"):
        frame[column] = _strict_string(frame[column], f"pCR.{column}")
    frame["y_true"] = _binary_series(frame["y_true"], "pCR.y_true")
    frame["predicted_probability"] = _probability_series(
        frame["predicted_probability"], "pCR.predicted_probability"
    )
    frame["predicted_label"] = _binary_series(
        frame["predicted_label"], "pCR.predicted_label"
    )
    frame["threshold"] = _probability_series(frame["threshold"], "pCR.threshold")
    expected_prediction = (
        frame["predicted_probability"].to_numpy(dtype=np.float64)
        >= frame["threshold"].to_numpy(dtype=np.float64)
    ).astype(np.int8)
    if not np.array_equal(
        expected_prediction, frame["predicted_label"].to_numpy(dtype=np.int8)
    ):
        raise ValueError("pCR predicted_label disagrees with stored threshold")
    return frame, digest


def load_profile_predictions(path: str | Path) -> tuple[pd.DataFrame, str]:
    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = sha256_file(source)
    frame = pd.read_csv(source)
    _require_exact_columns(frame, PROFILE_COLUMNS, "MRI profile predictions")
    frame = _normalize_prediction_ids(frame, "MRI profile predictions")
    frame["fold"] = _integer_series(frame["fold"], "profile.fold", range(5))
    frame["seed"] = _integer_series(frame["seed"], "profile.seed", SEEDS)
    for column in ("arm", "view", "target"):
        frame[column] = _strict_string(frame[column], f"profile.{column}")
    if set(frame["arm"]) != set(ARMS):
        raise ValueError("profile arms differ from LOCAL0/LOCAL3")
    if set(frame["view"]) != set(PROFILE_VIEWS):
        raise ValueError("profile views differ from the frozen contract")
    if set(frame["target"]) != set(PROFILE_TARGETS):
        raise ValueError("profile targets differ from HR/HER2/subtype_4class")
    return frame, digest


def build_pcr_reference_metrics(
    predictions: pd.DataFrame, contract: MatchedCohortContract
) -> pd.DataFrame:
    selected = predictions.loc[
        predictions["population"].eq(PCR_POPULATION)
        & predictions["model"].isin(PCR_MODELS)
    ].copy()
    if selected.empty:
        raise ValueError("pCR source has no matched MRI reference rows")
    if not selected["clinical_contract"].eq(MRI_CLINICAL_CONTRACT).all():
        raise ValueError("pCR MRI reference clinical contract drifted")
    if set(selected["timing"]) != set(PCR_TIMINGS):
        raise ValueError("pCR MRI reference timings differ from T0-T3")
    if set(selected["model"]) != set(PCR_MODELS):
        raise ValueError("pCR MRI reference model family is incomplete")

    labels = contract.frame.set_index("trial_id", verify_integrity=True)["pCR"].to_dict()
    cell_rows: list[dict[str, Any]] = []
    group_columns = ["seed", "arm", "timing", "model"]
    for key, group in selected.groupby(group_columns, sort=True, observed=True):
        seed, arm, timing, model = key
        label = f"pCR/{seed}/{arm}/{timing}/{model}"
        _validate_group_alignment(group, contract, labels, label)
        metrics = _binary_metrics(group, use_stored_threshold_decisions=False)
        cell_rows.append(
            {
                "seed": int(seed),
                "arm": str(arm),
                "timing": str(timing),
                "model": str(model),
                **metrics,
            }
        )
    cells = pd.DataFrame(cell_rows)

    expected_groups = len(SENSITIVITY_CELLS) * len(PCR_TIMINGS) * len(PCR_MODELS)
    if len(cells) != expected_groups:
        raise ValueError(f"expected {expected_groups} pCR cells, observed {len(cells)}")

    rows: list[dict[str, Any]] = []
    for (timing, model), group in cells.groupby(
        ["timing", "model"], sort=False, observed=True
    ):
        _require_sensitivity_cells(group, f"pCR/{timing}/{model}")
        rows.append(
            {
                "population": PCR_POPULATION,
                "target": "pCR",
                "timing": str(timing),
                "timing_label": TIMING_LABELS[str(timing)],
                "model": str(model),
                "model_role": (
                    "context" if model in {"C", "C+F"} else "MRI reference"
                ),
                "clinical_contract": MRI_CLINICAL_CONTRACT,
                "n_sensitivity_cells": len(group),
                "n_patients_per_cell": _constant_int(
                    group, "n", f"pCR/{timing}/{model}"
                ),
                "n_positive_per_cell": _constant_int(
                    group, "n_positive", f"pCR/{timing}/{model}"
                ),
                "n_negative_per_cell": _constant_int(
                    group, "n_negative", f"pCR/{timing}/{model}"
                ),
                **_summary_triplet(group["auroc"], "auroc"),
                **_summary_triplet(group["auprc"], "auprc"),
                **_summary_triplet(
                    group["balanced_accuracy"], "balanced_accuracy"
                ),
                **_summary_triplet(group["brier"], "brier"),
                "aggregation": AGGREGATION_DESCRIPTION,
            }
        )
    output = pd.DataFrame(rows)
    timing_order = {value: index for index, value in enumerate(PCR_TIMINGS)}
    model_order = {value: index for index, value in enumerate(PCR_MODELS)}
    output["_timing_order"] = output["timing"].map(timing_order)
    output["_model_order"] = output["model"].map(model_order)
    output = output.sort_values(["_timing_order", "_model_order"], kind="stable").drop(
        columns=["_timing_order", "_model_order"]
    )
    output = output.loc[:, PCR_OUTPUT_COLUMNS].reset_index(drop=True)
    if len(output) != len(PCR_TIMINGS) * len(PCR_MODELS):
        raise AssertionError("pCR aggregate row count drifted")
    return output


def _prepare_profile_group(
    group: pd.DataFrame,
    target: str,
    contract: MatchedCohortContract,
    label_maps: Mapping[str, Mapping[str, Any]],
    label: str,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    output = group.copy()
    if target in {"HR", "HER2"}:
        output["y_true"] = _binary_series(output["y_true"], f"{label}.y_true")
        output["predicted_probability"] = _probability_series(
            output["predicted_probability"], f"{label}.predicted_probability"
        )
        output["predicted_label"] = _binary_series(
            output["predicted_label"], f"{label}.predicted_label"
        )
        output["threshold"] = _probability_series(
            output["threshold"], f"{label}.threshold"
        )
        expected_prediction = (
            output["predicted_probability"].to_numpy(dtype=np.float64)
            >= output["threshold"].to_numpy(dtype=np.float64)
        ).astype(np.int8)
        if not np.array_equal(
            expected_prediction, output["predicted_label"].to_numpy(dtype=np.int8)
        ):
            raise ValueError(f"{label} predicted_label disagrees with threshold")
        subtype_columns = list(SUBTYPE_PROBABILITY_COLUMNS.values())
        if output[subtype_columns].notna().any(axis=None):
            raise ValueError(f"{label} binary rows unexpectedly contain subtype scores")
        _validate_group_alignment(output, contract, label_maps[target], label)
        return output, _binary_metrics(
            output, use_stored_threshold_decisions=True
        )

    output["y_true"] = _strict_string(output["y_true"], f"{label}.y_true")
    output["predicted_label"] = _strict_string(
        output["predicted_label"], f"{label}.predicted_label"
    )
    for column in SUBTYPE_PROBABILITY_COLUMNS.values():
        output[column] = _probability_series(output[column], f"{label}.{column}")
    if output["predicted_probability"].notna().any() or output["threshold"].notna().any():
        raise ValueError(f"{label} subtype scalar probability/threshold must be missing")
    _validate_group_alignment(output, contract, label_maps[target], label)
    return output, _multiclass_metrics(output)


def build_profile_reference_metrics(
    predictions: pd.DataFrame, contract: MatchedCohortContract
) -> pd.DataFrame:
    matched = contract.frame.set_index("trial_id", verify_integrity=True)
    label_maps: dict[str, Mapping[str, Any]] = {
        "HR": matched["HR"].astype(int).to_dict(),
        "HER2": matched["HER2"].astype(int).to_dict(),
        "subtype_4class": matched["subtype"].astype(str).to_dict(),
    }

    # Validate the source OOF coverage before filtering.  This prevents a
    # malformed file from appearing valid merely because its matched rows exist.
    source_group_columns = ["seed", "arm", "view", "target"]
    for key, source_group in predictions.groupby(
        source_group_columns, sort=True, observed=True
    ):
        if len(source_group) != 808 or source_group["trial_id"].duplicated().any():
            raise ValueError(f"profile source OOF coverage drifted for {key}")

    selected = predictions.loc[
        predictions["trial_id"].isin(contract.trial_ids)
    ].copy()
    cell_rows: list[dict[str, Any]] = []
    for key, group in selected.groupby(
        source_group_columns, sort=True, observed=True
    ):
        seed, arm, view, target = key
        label = f"profile/{seed}/{arm}/{view}/{target}"
        _, metrics = _prepare_profile_group(
            group, str(target), contract, label_maps, label
        )
        cell_rows.append(
            {
                "seed": int(seed),
                "arm": str(arm),
                "view": str(view),
                "target": str(target),
                **metrics,
            }
        )
    cells = pd.DataFrame(cell_rows)
    expected_groups = len(SENSITIVITY_CELLS) * len(PROFILE_VIEWS) * len(PROFILE_TARGETS)
    if len(cells) != expected_groups:
        raise ValueError(
            f"expected {expected_groups} matched profile cells, observed {len(cells)}"
        )

    rows: list[dict[str, Any]] = []
    for (view, target), group in cells.groupby(
        ["view", "target"], sort=False, observed=True
    ):
        _require_sensitivity_cells(group, f"profile/{view}/{target}")
        is_multiclass = target == "subtype_4class"
        n_positive: int | str = ""
        n_classes = 4 if is_multiclass else 2
        if not is_multiclass:
            n_positive = _constant_int(
                group, "n_positive", f"profile/{view}/{target}"
            )
        rows.append(
            {
                "population": PCR_POPULATION,
                "target": str(target),
                "metric_definition": "macro one-vs-rest" if is_multiclass else "binary",
                "view": str(view),
                "timing_label": PROFILE_VIEW_LABELS[str(view)],
                "n_sensitivity_cells": len(group),
                "n_patients_per_cell": _constant_int(
                    group, "n", f"profile/{view}/{target}"
                ),
                "n_positive_per_cell": n_positive,
                "n_classes": n_classes,
                **_summary_triplet(group["auroc"], "auroc"),
                **_summary_triplet(group["auprc"], "auprc"),
                **_summary_triplet(
                    group["balanced_accuracy"], "balanced_accuracy"
                ),
                "aggregation": AGGREGATION_DESCRIPTION,
            }
        )
    output = pd.DataFrame(rows)
    view_order = {value: index for index, value in enumerate(PROFILE_VIEWS)}
    target_order = {value: index for index, value in enumerate(PROFILE_TARGETS)}
    output["_view_order"] = output["view"].map(view_order)
    output["_target_order"] = output["target"].map(target_order)
    output = output.sort_values(["_view_order", "_target_order"], kind="stable").drop(
        columns=["_view_order", "_target_order"]
    )
    output = output.loc[:, PROFILE_OUTPUT_COLUMNS].reset_index(drop=True)
    if len(output) != len(PROFILE_VIEWS) * len(PROFILE_TARGETS):
        raise AssertionError("profile aggregate row count drifted")
    return output


def _source_file_record(path: Path, digest: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": digest,
        "size_bytes": int(path.stat().st_size),
        "read_only": True,
    }


def build_reference(
    config_path: str | Path | None = None,
) -> ReferenceResult:
    """Validate private inputs and return aggregate-only reference tables."""

    resolved_config = (
        Path(config_path).expanduser().resolve(strict=True)
        if config_path is not None
        else (EXPERIMENT_ROOT / "configs" / "experiment.json").resolve(strict=True)
    )
    config = load_config(resolved_config)
    contract = load_matched_cohort_contract(config)
    pcr_path = Path(config["source"]["mri_audit_predictions"]).expanduser().resolve(
        strict=True
    )
    profile_path = Path(
        config["source"]["mri_profile_predictions"]
    ).expanduser().resolve(strict=True)
    pcr_predictions, pcr_sha = load_pcr_predictions(pcr_path)
    profile_predictions, profile_sha = load_profile_predictions(profile_path)

    pcr_metrics = build_pcr_reference_metrics(pcr_predictions, contract)
    profile_metrics = build_profile_reference_metrics(profile_predictions, contract)

    matched = contract.frame
    subtype_counts = {
        str(key): int(value)
        for key, value in matched["subtype"].value_counts().sort_index().items()
    }
    provenance: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "classical_dce_phenotype_complementarity",
        "artifact": "frozen_LOCAL_MRI_reference",
        "status": "validated_aggregate_only",
        "inputs": {
            "config": _source_file_record(resolved_config, sha256_file(resolved_config)),
            "pcr_oof_predictions": _source_file_record(pcr_path, pcr_sha),
            "profile_oof_predictions": _source_file_record(profile_path, profile_sha),
            "radiomics_workbook_sha256": contract.primary_provenance[
                "radiomics_sha256"
            ],
            "clinical_workbook_sha256": contract.primary_provenance[
                "clinical_sha256"
            ],
            "fold_manifest_sha256": config["source"][
                "mri_fold_manifest_sha256"
            ],
        },
        "matched_population": {
            "name": PCR_POPULATION,
            "source_radiomics_patients": int(contract.primary_provenance["n"]),
            "exact_mri_overlap": int(len(matched)),
            "excluded_without_locked_mri_overlap": int(
                contract.primary_provenance["n"] - len(matched)
            ),
            "pcr_positive": int(matched["pCR"].sum()),
            "hr_positive": int(matched["HR"].sum()),
            "her2_positive": int(matched["HER2"].sum()),
            "subtype_counts": subtype_counts,
            "exact_trial_id_overlap_verified": True,
            "outer_test_fold_verified": True,
            "authoritative_labels_verified": True,
            "accepted_patient_id_patterns": [
                "ISPY2-<exact six digits>",
                "ACRIN-6698-<exact six digits>",
            ],
        },
        "sensitivity_contract": {
            "seeds": list(SEEDS),
            "arms": list(ARMS),
            "n_cells": len(SENSITIVITY_CELLS),
            "aggregation": AGGREGATION_DESCRIPTION,
            "duplicate_patients_pooled_across_cells": False,
        },
        "metric_contract": {
            "pcr": ["AUROC", "average precision", "balanced accuracy", "Brier"],
            "pcr_balanced_accuracy_threshold": 0.5,
            "binary_profile": ["AUROC", "average precision", "balanced accuracy"],
            "binary_profile_decision": "stored fold-validation threshold",
            "subtype": [
                "macro one-vs-rest AUROC",
                "macro average precision",
                "balanced accuracy",
            ],
        },
        "timing_labels": {
            **TIMING_LABELS,
            **{key: value for key, value in PROFILE_VIEW_LABELS.items() if key.startswith("long_")},
        },
        "privacy": {
            "patient_ids_written": False,
            "prediction_probabilities_written": False,
            "patient_level_metrics_written": False,
            "outputs_are_aggregate_only": True,
        },
    }
    return ReferenceResult(pcr_metrics, profile_metrics, provenance)


def _temporary_path(directory: Path, target_name: str) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target_name}.", suffix=".tmp", dir=directory
    )
    os.close(descriptor)
    return Path(name)


def _assert_aggregate_only(frame: pd.DataFrame, label: str) -> None:
    forbidden = {"patient_id", "trial_id", "predicted_probability", "y_true"}
    observed = forbidden.intersection(frame.columns)
    if observed:
        raise ValueError(f"{label} contains forbidden patient-level columns: {observed}")


def write_reference_outputs(
    result: ReferenceResult,
    output_dir: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> Mapping[str, Any]:
    """Atomically write the two aggregate CSVs and non-sensitive provenance."""

    directory = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else (EXPERIMENT_ROOT / "metrics").resolve()
    )
    directory.mkdir(parents=True, exist_ok=True)
    targets = {
        "pcr_metrics": directory / "mri_reference_metrics.csv",
        "profile_metrics": directory / "mri_reference_profile_metrics.csv",
        "provenance": directory / "mri_reference_provenance.json",
    }
    existing = [path for path in targets.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "MRI reference outputs already exist; use --overwrite for these exact files: "
            + ", ".join(str(path) for path in existing)
        )

    _assert_aggregate_only(result.pcr_metrics, "pCR MRI reference")
    _assert_aggregate_only(result.profile_metrics, "profile MRI reference")
    if tuple(result.pcr_metrics.columns) != PCR_OUTPUT_COLUMNS:
        raise ValueError("pCR aggregate output schema drifted")
    if tuple(result.profile_metrics.columns) != PROFILE_OUTPUT_COLUMNS:
        raise ValueError("profile aggregate output schema drifted")

    temporary: dict[str, Path] = {
        key: _temporary_path(directory, path.name) for key, path in targets.items()
    }
    try:
        result.pcr_metrics.to_csv(
            temporary["pcr_metrics"], index=False, lineterminator="\n"
        )
        result.profile_metrics.to_csv(
            temporary["profile_metrics"], index=False, lineterminator="\n"
        )
        output_records = {
            "pcr_metrics": {
                "path": str(targets["pcr_metrics"]),
                "sha256": sha256_file(temporary["pcr_metrics"]),
                "rows": int(len(result.pcr_metrics)),
                "patient_level": False,
            },
            "profile_metrics": {
                "path": str(targets["profile_metrics"]),
                "sha256": sha256_file(temporary["profile_metrics"]),
                "rows": int(len(result.profile_metrics)),
                "patient_level": False,
            },
        }
        provenance = {**dict(result.provenance), "outputs": output_records}
        with temporary["provenance"].open("w", encoding="utf-8") as handle:
            json.dump(
                provenance,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")

        # All three complete temporary files exist before any destination is
        # replaced.  No patient-level temporary is ever created.
        for key in ("pcr_metrics", "profile_metrics", "provenance"):
            os.replace(temporary[key], targets[key])
        return {
            **output_records,
            "provenance": {
                "path": str(targets["provenance"]),
                "sha256": sha256_file(targets["provenance"]),
                "patient_level": False,
            },
        }
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=EXPERIMENT_ROOT / "configs" / "experiment.json",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the three known aggregate MRI-reference outputs.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run all private-input checks and aggregation without writing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_reference(args.config)
    if args.validate_only:
        summary = {
            "status": "validated_no_outputs_written",
            "pcr_aggregate_rows": int(len(result.pcr_metrics)),
            "profile_aggregate_rows": int(len(result.profile_metrics)),
            "matched_population": result.provenance["matched_population"][
                "exact_mri_overlap"
            ],
            "sensitivity_cells": result.provenance["sensitivity_contract"][
                "n_cells"
            ],
        }
    else:
        outputs = write_reference_outputs(result, overwrite=args.overwrite)
        summary = {
            "status": "aggregate_outputs_written",
            "outputs": outputs,
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

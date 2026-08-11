"""Strict OOF summaries and paired uncertainty for the compact-fusion audit.

This module deliberately operates on patient-level outer-test prediction ledgers.
It never fits a prediction model.  Every aggregate retains population, MRI seed,
arm, and timing, and every paired comparison requires the exact same patient,
fold, and label rows before calling Goal 2's tested fold-stratified bootstrap.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


PCR_PREDICTION_COLUMNS = (
    "patient_id",
    "fold",
    "population",
    "seed",
    "arm",
    "timing",
    "model_family",
    "representation",
    "dimension",
    "model_key",
    "y_true",
    "predicted_probability",
    "predicted_label",
    "threshold",
)

PCR_CELL_COLUMNS = ("population", "seed", "arm", "timing")
PCR_OOF_GROUP_COLUMNS = (*PCR_CELL_COLUMNS, "model_key")
DEFAULT_POPULATION_SIZES: Mapping[str, int] = {
    "full_808": 808,
    "ftv_complete_375": 375,
}
DEFAULT_FOLDS = (0, 1, 2, 3, 4)

PROFILE_REPRESENTATIONS = ("raw", "pca16", "pca32")
PROFILE_TARGETS = ("HR", "HER2", "subtype_4class")
SUBTYPE_CLASSES = tuple(
    sorted(("HR+/HER2-", "HR-/HER2-", "HR+/HER2+", "HR-/HER2+"))
)
SUBTYPE_PROBABILITY_COLUMNS: Mapping[str, str] = {
    "HR+/HER2-": "prob_hr_pos_her2_neg",
    "HR-/HER2-": "prob_hr_neg_her2_neg",
    "HR+/HER2+": "prob_hr_pos_her2_pos",
    "HR-/HER2+": "prob_hr_neg_her2_pos",
}
PROFILE_REQUIRED_COLUMNS = (
    "patient_id",
    "fold",
    "seed",
    "arm",
    "timing",
    "representation",
    "target",
    "y_true",
    "predicted_probability",
    "predicted_label",
    "threshold",
    *SUBTYPE_PROBABILITY_COLUMNS.values(),
)


class SummaryContractError(ValueError):
    """Raised when an OOF prediction or pairing invariant is violated."""


@dataclass(frozen=True)
class ComparisonSpec:
    """A named exact-column selector pair.

    Selector values may be scalars, ``None`` (meaning missing), or a finite
    list/tuple/set of accepted values.  ``populations`` optionally limits the
    comparison to named estimands before cell matching.
    """

    name: str
    reference: Mapping[str, Any]
    comparison: Mapping[str, Any]
    populations: tuple[str, ...] | None = None


@dataclass(frozen=True)
class PairedComparisonSummary:
    point_effects: pd.DataFrame
    bootstrap_summary: pd.DataFrame
    bootstrap_draws: pd.DataFrame


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise SummaryContractError(f"{label} is missing columns: {missing}")


def _text(series: pd.Series, label: str) -> pd.Series:
    if series.isna().any():
        raise SummaryContractError(f"{label} contains missing values")
    values = series.astype(str)
    if values.str.strip().ne(values).any() or values.eq("").any():
        raise SummaryContractError(f"{label} contains blank or padded values")
    return values


def _integers(series: pd.Series, label: str, *, nonnegative: bool = False) -> pd.Series:
    try:
        numeric = pd.to_numeric(series, errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise SummaryContractError(f"{label} must contain integers") from error
    if (
        not np.isfinite(numeric).all()
        or not np.equal(numeric, np.floor(numeric)).all()
        or (nonnegative and np.any(numeric < 0))
    ):
        raise SummaryContractError(f"{label} must contain finite integers")
    return pd.Series(numeric.astype(np.int64), index=series.index, name=series.name)


def _binary(series: pd.Series, label: str) -> pd.Series:
    values = _integers(series, label)
    if not values.isin((0, 1)).all():
        raise SummaryContractError(f"{label} must contain only 0/1")
    return values


def _probabilities(series: pd.Series, label: str) -> pd.Series:
    try:
        values = pd.to_numeric(series, errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise SummaryContractError(f"{label} must be numeric") from error
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise SummaryContractError(f"{label} must contain finite values in [0,1]")
    return pd.Series(values, index=series.index, name=series.name)


def _fold_tuple(expected_folds: Sequence[int]) -> tuple[int, ...]:
    folds = tuple(int(value) for value in expected_folds)
    if not folds or len(folds) != len(set(folds)) or any(value < 0 for value in folds):
        raise SummaryContractError("expected_folds must be distinct non-negative integers")
    return folds


def _population_sizes(values: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(values, Mapping) or not values:
        raise SummaryContractError("expected_population_sizes must be a nonempty mapping")
    output: dict[str, int] = {}
    for name, count in values.items():
        text = str(name)
        if not text or isinstance(count, bool) or int(count) <= 0:
            raise SummaryContractError("population names/counts must be nonempty and positive")
        output[text] = int(count)
    return output


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def _dimension_summary(group: pd.DataFrame) -> dict[str, Any]:
    values: list[Any] = []
    for value in group["dimension"].tolist():
        scalar = _json_scalar(value)
        if scalar not in values:
            values.append(scalar)
    values.sort(key=lambda value: (value is None, str(value)))
    return {
        "dimension": values[0] if len(values) == 1 else pd.NA,
        "dimension_values": json.dumps(values, separators=(",", ":")),
    }


def _binary_metric_values(group: pd.DataFrame) -> dict[str, int | float]:
    labels = group["y_true"].to_numpy(dtype=np.int64)
    probabilities = group["predicted_probability"].to_numpy(dtype=np.float64)
    predictions = group["predicted_label"].to_numpy(dtype=np.int64)
    if set(np.unique(labels)) != {0, 1}:
        raise SummaryContractError("each pooled OOF metric group must contain both classes")
    return {
        "n": int(len(group)),
        "n_positive": int(labels.sum()),
        "n_negative": int((labels == 0).sum()),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "brier": float(np.mean(np.square(probabilities - labels))),
    }


def _validate_pcr_oof(
    predictions: pd.DataFrame,
    *,
    expected_population_sizes: Mapping[str, int],
    expected_folds: Sequence[int],
) -> pd.DataFrame:
    if not isinstance(predictions, pd.DataFrame) or predictions.empty:
        raise SummaryContractError("pCR predictions must be a nonempty DataFrame")
    _require_columns(predictions, PCR_PREDICTION_COLUMNS, "pCR predictions")
    frame = predictions.loc[:, PCR_PREDICTION_COLUMNS].copy()
    for column in (
        "patient_id",
        "population",
        "arm",
        "timing",
        "model_family",
        "representation",
        "model_key",
    ):
        frame[column] = _text(frame[column], f"pCR.{column}")
    frame["fold"] = _integers(frame["fold"], "pCR.fold", nonnegative=True)
    frame["seed"] = _integers(frame["seed"], "pCR.seed", nonnegative=True)
    frame["y_true"] = _binary(frame["y_true"], "pCR.y_true")
    frame["predicted_probability"] = _probabilities(
        frame["predicted_probability"], "pCR.predicted_probability"
    )
    frame["predicted_label"] = _binary(frame["predicted_label"], "pCR.predicted_label")
    frame["threshold"] = _probabilities(frame["threshold"], "pCR.threshold")
    expected_prediction = (
        frame["predicted_probability"].to_numpy()
        >= frame["threshold"].to_numpy()
    ).astype(np.int64)
    if not np.array_equal(expected_prediction, frame["predicted_label"].to_numpy()):
        raise SummaryContractError("pCR predicted_label disagrees with probability/threshold")

    sizes = _population_sizes(expected_population_sizes)
    folds = _fold_tuple(expected_folds)
    unknown = sorted(set(frame["population"]) - set(sizes))
    if unknown:
        raise SummaryContractError(f"pCR predictions contain unknown populations: {unknown}")

    canonical_by_population: dict[str, pd.DataFrame] = {}
    for key, group in frame.groupby(list(PCR_OOF_GROUP_COLUMNS), sort=True, dropna=False):
        population = str(key[0])
        if group["patient_id"].duplicated().any():
            raise SummaryContractError(f"pCR OOF group repeats patients: {key}")
        if len(group) != sizes[population]:
            raise SummaryContractError(
                f"pCR OOF group {key} has {len(group)} patients; expected {sizes[population]}"
            )
        if set(group["fold"]) != set(folds):
            raise SummaryContractError(f"pCR OOF group {key} does not cover exact folds {folds}")
        for descriptor in ("model_family", "representation"):
            if group[descriptor].nunique(dropna=False) != 1:
                raise SummaryContractError(
                    f"model_key {key[-1]!r} changes {descriptor} within one OOF cell"
                )
        current = group.loc[:, ["patient_id", "fold", "y_true"]].sort_values(
            "patient_id", kind="stable"
        ).reset_index(drop=True)
        canonical = canonical_by_population.setdefault(population, current)
        if not current.equals(canonical):
            raise SummaryContractError(
                f"pCR patient/fold/label coverage differs within population {population}"
            )
    return frame


def aggregate_pcr_predictions(
    predictions: pd.DataFrame,
    *,
    expected_population_sizes: Mapping[str, int] = DEFAULT_POPULATION_SIZES,
    expected_folds: Sequence[int] = DEFAULT_FOLDS,
) -> pd.DataFrame:
    """Aggregate exact five-fold patient OOF predictions without pooling estimands."""

    frame = _validate_pcr_oof(
        predictions,
        expected_population_sizes=expected_population_sizes,
        expected_folds=expected_folds,
    )
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(list(PCR_OOF_GROUP_COLUMNS), sort=True, dropna=False):
        fold_sizes = {
            str(int(fold)): int(count)
            for fold, count in group["fold"].value_counts().sort_index().items()
        }
        rows.append(
            {
                **dict(zip(PCR_OOF_GROUP_COLUMNS, key, strict=True)),
                "model_family": str(group["model_family"].iloc[0]),
                "representation": str(group["representation"].iloc[0]),
                **_dimension_summary(group),
                "n_folds": int(group["fold"].nunique()),
                "fold_sizes": json.dumps(fold_sizes, sort_keys=True, separators=(",", ":")),
                **_binary_metric_values(group),
            }
        )
    return pd.DataFrame(rows).sort_values(
        list(PCR_OOF_GROUP_COLUMNS), kind="stable"
    ).reset_index(drop=True)


def _normalize_comparisons(
    comparisons: Sequence[ComparisonSpec | Mapping[str, Any]],
) -> tuple[ComparisonSpec, ...]:
    output: list[ComparisonSpec] = []
    for value in comparisons:
        if isinstance(value, ComparisonSpec):
            spec = value
        elif isinstance(value, Mapping):
            try:
                populations = value.get("populations")
                if isinstance(populations, str):
                    populations = (populations,)
                elif populations is not None:
                    populations = tuple(str(item) for item in populations)
                spec = ComparisonSpec(
                    name=str(value["name"]),
                    reference=dict(value["reference"]),
                    comparison=dict(value["comparison"]),
                    populations=populations,
                )
            except (KeyError, TypeError) as error:
                raise SummaryContractError("invalid comparison specification") from error
        else:
            raise SummaryContractError("comparisons must contain ComparisonSpec or mappings")
        if not spec.name or not spec.reference or not spec.comparison:
            raise SummaryContractError("comparison name/selectors must be nonempty")
        output.append(spec)
    if not output or len({spec.name for spec in output}) != len(output):
        raise SummaryContractError("comparison names must be nonempty and unique")
    return tuple(output)


def _selector_mask(frame: pd.DataFrame, selector: Mapping[str, Any], label: str) -> pd.Series:
    missing = sorted(set(selector) - set(frame.columns))
    if missing:
        raise SummaryContractError(f"{label} selector references missing columns: {missing}")
    mask = pd.Series(True, index=frame.index)
    for column, value in selector.items():
        if value is None:
            current = frame[column].isna()
        elif isinstance(value, (list, tuple, set, frozenset, np.ndarray, pd.Index)):
            accepted = list(value)
            if not accepted:
                raise SummaryContractError(f"{label}.{column} selector has no values")
            current = frame[column].isin(accepted)
        else:
            current = frame[column].eq(value)
        mask &= current
    return mask


def _selector_json(selector: Mapping[str, Any]) -> str:
    normalized: dict[str, Any] = {}
    for key, value in sorted(selector.items()):
        if isinstance(value, (set, frozenset)):
            normalized[str(key)] = sorted(_json_scalar(item) for item in value)
        elif isinstance(value, (list, tuple, np.ndarray, pd.Index)):
            normalized[str(key)] = [_json_scalar(item) for item in value]
        else:
            normalized[str(key)] = _json_scalar(value)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def _selected_pair_frames(
    frame: pd.DataFrame,
    spec: ComparisonSpec,
    *,
    cell_columns: Sequence[str],
) -> list[tuple[tuple[Any, ...], pd.DataFrame, pd.DataFrame]]:
    if "population" not in cell_columns:
        raise SummaryContractError("paired cell columns must include population")
    if spec.populations is not None:
        unknown = sorted(set(spec.populations) - set(frame["population"]))
        if unknown:
            raise SummaryContractError(
                f"comparison {spec.name!r} requests absent populations: {unknown}"
            )
        scope = frame.loc[frame["population"].isin(spec.populations)]
    else:
        scope = frame
    reference = scope.loc[_selector_mask(scope, spec.reference, f"{spec.name}.reference")]
    comparison = scope.loc[
        _selector_mask(scope, spec.comparison, f"{spec.name}.comparison")
    ]
    if reference.empty or comparison.empty:
        raise SummaryContractError(f"comparison {spec.name!r} has an empty selector result")

    def keys(selected: pd.DataFrame) -> set[tuple[Any, ...]]:
        return {
            tuple(row)
            for row in selected.loc[:, list(cell_columns)].drop_duplicates().itertuples(
                index=False, name=None
            )
        }

    reference_keys = keys(reference)
    comparison_keys = keys(comparison)
    if reference_keys != comparison_keys:
        raise SummaryContractError(
            f"comparison {spec.name!r} reference/comparison cells differ; "
            f"reference_only={sorted(reference_keys - comparison_keys)[:3]}, "
            f"comparison_only={sorted(comparison_keys - reference_keys)[:3]}"
        )

    output: list[tuple[tuple[Any, ...], pd.DataFrame, pd.DataFrame]] = []
    for cell in sorted(reference_keys, key=lambda values: tuple(str(v) for v in values)):
        reference_mask = np.logical_and.reduce(
            [reference[column].eq(value).to_numpy() for column, value in zip(cell_columns, cell)]
        )
        comparison_mask = np.logical_and.reduce(
            [comparison[column].eq(value).to_numpy() for column, value in zip(cell_columns, cell)]
        )
        ref = reference.loc[reference_mask].copy()
        cmp = comparison.loc[comparison_mask].copy()
        if ref["patient_id"].duplicated().any() or cmp["patient_id"].duplicated().any():
            raise SummaryContractError(
                f"comparison {spec.name!r} selector matches multiple rows per patient in {cell}"
            )
        ref_ids, cmp_ids = set(ref["patient_id"]), set(cmp["patient_id"])
        if ref_ids != cmp_ids:
            raise SummaryContractError(
                f"comparison {spec.name!r} patient IDs differ in cell {cell}"
            )
        paired = ref[["patient_id", "fold", "y_true"]].merge(
            cmp[["patient_id", "fold", "y_true"]],
            on="patient_id",
            how="outer",
            validate="one_to_one",
            suffixes=("_reference", "_comparison"),
            indicator=True,
        )
        if not paired["_merge"].eq("both").all():
            raise SummaryContractError(f"comparison {spec.name!r} pairing is incomplete")
        if not np.array_equal(
            paired["fold_reference"].to_numpy(), paired["fold_comparison"].to_numpy()
        ):
            raise SummaryContractError(
                f"comparison {spec.name!r} fold assignments differ in cell {cell}"
            )
        if not np.array_equal(
            paired["y_true_reference"].to_numpy(),
            paired["y_true_comparison"].to_numpy(),
        ):
            raise SummaryContractError(
                f"comparison {spec.name!r} labels differ in cell {cell}"
            )
        output.append((cell, ref, cmp))
    return output


def paired_point_effects(
    predictions: pd.DataFrame,
    comparisons: Sequence[ComparisonSpec | Mapping[str, Any]],
    *,
    expected_population_sizes: Mapping[str, int] = DEFAULT_POPULATION_SIZES,
    expected_folds: Sequence[int] = DEFAULT_FOLDS,
    cell_columns: Sequence[str] = PCR_CELL_COLUMNS,
) -> pd.DataFrame:
    """Return paired held-out point effects for exact selector pairs."""

    frame = _validate_pcr_oof(
        predictions,
        expected_population_sizes=expected_population_sizes,
        expected_folds=expected_folds,
    )
    specs = _normalize_comparisons(comparisons)
    rows: list[dict[str, Any]] = []
    for spec in specs:
        for cell, reference, comparison in _selected_pair_frames(
            frame, spec, cell_columns=cell_columns
        ):
            ref_metric = _binary_metric_values(reference)
            cmp_metric = _binary_metric_values(comparison)
            delta_brier = float(cmp_metric["brier"] - ref_metric["brier"])
            rows.append(
                {
                    "comparison_name": spec.name,
                    **dict(zip(cell_columns, cell, strict=True)),
                    "reference_selector": _selector_json(spec.reference),
                    "comparison_selector": _selector_json(spec.comparison),
                    "n": int(ref_metric["n"]),
                    "reference_auroc": float(ref_metric["auroc"]),
                    "comparison_auroc": float(cmp_metric["auroc"]),
                    "delta_auroc": float(cmp_metric["auroc"] - ref_metric["auroc"]),
                    "reference_auprc": float(ref_metric["auprc"]),
                    "comparison_auprc": float(cmp_metric["auprc"]),
                    "delta_auprc": float(cmp_metric["auprc"] - ref_metric["auprc"]),
                    "reference_brier": float(ref_metric["brier"]),
                    "comparison_brier": float(cmp_metric["brier"]),
                    "delta_brier": delta_brier,
                    "brier_improvement": -delta_brier,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["comparison_name", *cell_columns], kind="stable"
    ).reset_index(drop=True)


@lru_cache(maxsize=1)
def _goal2_bootstrap_helper() -> Any:
    goal2_path = (
        Path(__file__).resolve().parents[2]
        / "mri_clinical_complementarity_audit"
        / "scripts"
        / "modeling.py"
    )
    if not goal2_path.is_file():
        raise FileNotFoundError(f"missing Goal 2 modeling helper: {goal2_path}")
    module_name = "_compact_audit_goal2_modeling"
    spec = importlib.util.spec_from_file_location(module_name, goal2_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Goal 2 modeling helper: {goal2_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.paired_fold_stratified_bootstrap


def _stable_bootstrap_seed(base_seed: int, comparison: str, cell: Sequence[Any]) -> int:
    payload = json.dumps(
        [comparison, *[_json_scalar(value) for value in cell]],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    offset = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return int((int(base_seed) + offset) % (2**63 - 1))


def paired_bootstrap_effects(
    predictions: pd.DataFrame,
    comparisons: Sequence[ComparisonSpec | Mapping[str, Any]],
    *,
    n_bootstrap: int = 2_000,
    confidence_level: float = 0.95,
    random_seed: int = 260_814,
    expected_population_sizes: Mapping[str, int] = DEFAULT_POPULATION_SIZES,
    expected_folds: Sequence[int] = DEFAULT_FOLDS,
    cell_columns: Sequence[str] = PCR_CELL_COLUMNS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Goal 2's patient-within-fold paired bootstrap independently per cell."""

    frame = _validate_pcr_oof(
        predictions,
        expected_population_sizes=expected_population_sizes,
        expected_folds=expected_folds,
    )
    specs = _normalize_comparisons(comparisons)
    helper = _goal2_bootstrap_helper()
    summaries: list[pd.DataFrame] = []
    draws: list[pd.DataFrame] = []
    for spec in specs:
        for cell, reference, comparison in _selected_pair_frames(
            frame, spec, cell_columns=cell_columns
        ):
            bootstrap_seed = _stable_bootstrap_seed(random_seed, spec.name, cell)
            result = helper(
                reference,
                comparison,
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                seed=bootstrap_seed,
            )
            metadata = {
                "comparison_name": spec.name,
                **dict(zip(cell_columns, cell, strict=True)),
                "reference_selector": _selector_json(spec.reference),
                "comparison_selector": _selector_json(spec.comparison),
                "bootstrap_seed": bootstrap_seed,
            }
            summary = result.summary.rename(
                columns={
                    "reference": "reference_value",
                    "comparison": "comparison_value",
                }
            ).drop(columns=["seed"])
            summary["delta"] = summary["comparison_value"] - summary["reference_value"]
            summary["delta_brier"] = np.where(
                summary["metric"].eq("brier"), summary["delta"], np.nan
            )
            summaries.append(summary.assign(**metadata))
            draw = result.draws.copy()
            draw["delta_brier"] = -draw["brier_improvement"]
            draws.append(draw.assign(**metadata))
    summary_frame = pd.concat(summaries, ignore_index=True)
    draw_frame = pd.concat(draws, ignore_index=True)
    leading = ["comparison_name", *cell_columns]
    summary_frame = summary_frame.loc[
        :, [*leading, *[column for column in summary_frame if column not in leading]]
    ].sort_values([*leading, "metric"], kind="stable", ignore_index=True)
    draw_frame = draw_frame.loc[
        :, [*leading, *[column for column in draw_frame if column not in leading]]
    ].sort_values([*leading, "bootstrap_index"], kind="stable", ignore_index=True)
    return summary_frame, draw_frame


def summarize_paired_comparisons(
    predictions: pd.DataFrame,
    comparisons: Sequence[ComparisonSpec | Mapping[str, Any]],
    **kwargs: Any,
) -> PairedComparisonSummary:
    """Convenience wrapper returning point effects, CI summary, and private draws."""

    point_keys = {
        "expected_population_sizes",
        "expected_folds",
        "cell_columns",
    }
    point_kwargs = {key: value for key, value in kwargs.items() if key in point_keys}
    point = paired_point_effects(predictions, comparisons, **point_kwargs)
    summary, draws = paired_bootstrap_effects(predictions, comparisons, **kwargs)
    return PairedComparisonSummary(point, summary, draws)


def _profile_patient_count(
    expected_patient_count: int | Mapping[str, int], population: str
) -> int:
    if isinstance(expected_patient_count, Mapping):
        sizes = _population_sizes(expected_patient_count)
        if population not in sizes:
            raise SummaryContractError(f"profile population {population!r} has no expected size")
        return sizes[population]
    if isinstance(expected_patient_count, bool) or int(expected_patient_count) <= 0:
        raise SummaryContractError("expected profile patient count must be positive")
    return int(expected_patient_count)


def aggregate_profile_predictions(
    predictions: pd.DataFrame,
    *,
    expected_patient_count: int | Mapping[str, int] = 808,
    expected_folds: Sequence[int] = DEFAULT_FOLDS,
    required_representations: Sequence[str] = PROFILE_REPRESENTATIONS,
) -> pd.DataFrame:
    """Aggregate binary HR/HER2 and fixed four-class subtype OOF probes."""

    if not isinstance(predictions, pd.DataFrame) or predictions.empty:
        raise SummaryContractError("profile predictions must be a nonempty DataFrame")
    _require_columns(predictions, PROFILE_REQUIRED_COLUMNS, "profile predictions")
    frame = predictions.copy()
    if "population" not in frame:
        frame["population"] = "full_808"
    for column in ("patient_id", "population", "arm", "timing", "representation", "target"):
        frame[column] = _text(frame[column], f"profile.{column}")
    frame["fold"] = _integers(frame["fold"], "profile.fold", nonnegative=True)
    frame["seed"] = _integers(frame["seed"], "profile.seed", nonnegative=True)
    allowed_representations = tuple(str(value) for value in required_representations)
    if not allowed_representations or len(set(allowed_representations)) != len(
        allowed_representations
    ):
        raise SummaryContractError("required profile representations must be unique")
    if not set(frame["representation"]).issubset(allowed_representations):
        raise SummaryContractError("profile predictions contain an unsupported representation")
    if not set(frame["target"]).issubset(PROFILE_TARGETS):
        raise SummaryContractError("profile predictions contain an unsupported target")
    folds = _fold_tuple(expected_folds)

    completeness_columns = ["population", "seed", "arm", "timing", "target"]
    for key, group in frame.groupby(completeness_columns, sort=True):
        if set(group["representation"]) != set(allowed_representations):
            raise SummaryContractError(
                f"profile cell {key} lacks exact representations {allowed_representations}"
            )

    group_columns = [
        "population",
        "seed",
        "arm",
        "timing",
        "representation",
        "target",
    ]
    canonical: dict[tuple[str, str], pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(group_columns, sort=True):
        population, seed, arm, timing, representation, target = key
        expected_n = _profile_patient_count(expected_patient_count, str(population))
        if group["patient_id"].duplicated().any() or len(group) != expected_n:
            raise SummaryContractError(
                f"profile OOF group {key} must contain exactly {expected_n} unique patients"
            )
        if set(group["fold"]) != set(folds):
            raise SummaryContractError(f"profile OOF group {key} lacks exact folds {folds}")

        if target in {"HR", "HER2"}:
            labels = _binary(group["y_true"], f"profile.{target}.y_true").to_numpy()
            probability = _probabilities(
                group["predicted_probability"], f"profile.{target}.probability"
            ).to_numpy()
            threshold = _probabilities(
                group["threshold"], f"profile.{target}.threshold"
            ).to_numpy()
            predicted = _binary(
                group["predicted_label"], f"profile.{target}.predicted_label"
            ).to_numpy()
            if not np.array_equal((probability >= threshold).astype(np.int64), predicted):
                raise SummaryContractError(
                    f"profile {target} predicted labels disagree with probability/threshold"
                )
            if set(np.unique(labels)) != {0, 1}:
                raise SummaryContractError(f"profile {target} must contain both classes")
            values = {
                "n": int(len(group)),
                "n_positive": int(labels.sum()),
                "n_negative": int((labels == 0).sum()),
                "n_classes": 2,
                "auroc": float(roc_auc_score(labels, probability)),
                "auprc": float(average_precision_score(labels, probability)),
                "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
                "brier": float(np.mean(np.square(probability - labels))),
                "class_counts": json.dumps(
                    {"0": int((labels == 0).sum()), "1": int(labels.sum())},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
            normalized_labels = labels.astype(str)
        else:
            labels = _text(group["y_true"], "profile.subtype.y_true").to_numpy()
            if set(labels) != set(SUBTYPE_CLASSES):
                raise SummaryContractError("subtype profile group must contain all four fixed classes")
            probability_columns = [
                SUBTYPE_PROBABILITY_COLUMNS[class_name] for class_name in SUBTYPE_CLASSES
            ]
            probability = np.column_stack(
                [
                    _probabilities(group[column], f"profile.subtype.{column}").to_numpy()
                    for column in probability_columns
                ]
            )
            if not np.allclose(probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
                raise SummaryContractError("subtype probability rows must sum to one")
            predicted = np.asarray(SUBTYPE_CLASSES)[np.argmax(probability, axis=1)]
            supplied_prediction = _text(
                group["predicted_label"], "profile.subtype.predicted_label"
            ).to_numpy()
            if not np.array_equal(predicted, supplied_prediction):
                raise SummaryContractError("subtype predicted_label disagrees with argmax")
            indicator = label_binarize(labels, classes=SUBTYPE_CLASSES)
            counts = {name: int(np.sum(labels == name)) for name in SUBTYPE_CLASSES}
            values = {
                "n": int(len(group)),
                "n_positive": np.nan,
                "n_negative": np.nan,
                "n_classes": 4,
                "auroc": float(
                    roc_auc_score(
                        labels,
                        probability,
                        labels=SUBTYPE_CLASSES,
                        multi_class="ovr",
                        average="macro",
                    )
                ),
                "auprc": float(average_precision_score(indicator, probability, average="macro")),
                "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
                "brier": np.nan,
                "class_counts": json.dumps(counts, sort_keys=True, separators=(",", ":")),
            }
            normalized_labels = labels.astype(str)

        coverage = pd.DataFrame(
            {
                "patient_id": group["patient_id"].to_numpy(),
                "fold": group["fold"].to_numpy(),
                "label": normalized_labels,
            }
        ).sort_values("patient_id", kind="stable").reset_index(drop=True)
        canonical_key = (str(population), str(target))
        expected_coverage = canonical.setdefault(canonical_key, coverage)
        if not coverage.equals(expected_coverage):
            raise SummaryContractError(
                f"profile patient/fold/label coverage differs for {canonical_key}"
            )
        rows.append(
            {
                **dict(zip(group_columns, key, strict=True)),
                "n_folds": int(group["fold"].nunique()),
                **values,
            }
        )
    return pd.DataFrame(rows).sort_values(group_columns, kind="stable").reset_index(drop=True)


__all__ = [
    "ComparisonSpec",
    "DEFAULT_FOLDS",
    "DEFAULT_POPULATION_SIZES",
    "PCR_CELL_COLUMNS",
    "PCR_PREDICTION_COLUMNS",
    "PROFILE_REPRESENTATIONS",
    "PROFILE_TARGETS",
    "PairedComparisonSummary",
    "SUBTYPE_CLASSES",
    "SUBTYPE_PROBABILITY_COLUMNS",
    "SummaryContractError",
    "aggregate_pcr_predictions",
    "aggregate_profile_predictions",
    "paired_bootstrap_effects",
    "paired_point_effects",
    "summarize_paired_comparisons",
]

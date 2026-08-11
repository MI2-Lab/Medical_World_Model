#!/usr/bin/env python3
"""Generate the seven preregistered figures from aggregate metrics only.

This module intentionally never reads OOF predictions or source workbooks.  It
validates every aggregate input needed by the complete figure set before any
final PNG is replaced, so an incomplete experiment cannot leave a mixture of
old and new figures behind.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]

PRIMARY_FILTERS = {
    "protocol": "primary_stratified_384",
    "population": "clinical_radiomics_complete_384",
    "scenario": "complete_case",
}
TIMINGS = ("T0", "T1", "T2", "T3")
VIEWS = ("static", "longitudinal")
TIMING_LABELS = {
    "T0": "T0",
    "T1": "T1",
    "T2": "T2",
    "T3": "T3 (late/pre-surgery)",
}
MODEL_ORDER = ("C", "F", "N", "FULL", "C+F", "C+N", "C+FULL")
FAMILY_MODELS = ("C+F", "C+F+D", "C+F+S", "C+F+B")
RESIDUAL_MODELS = ("N", "N_res", "C+F", "C+F+N_res")
COMPARISON_ORDER = (
    "C+FULL_vs_C+F",
    "C+N_vs_C",
    "C+F+N_res_vs_C+F",
)
FIGURE_FILENAMES = (
    "timing_auroc.png",
    "c_f_vs_c_full_auroc.png",
    "delta_auroc_forest.png",
    "phenotype_family_comparison.png",
    "hr_her2_heatmap.png",
    "residualized_results.png",
    "feature_correlation_matrix.png",
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
    "auroc",
    "auprc",
    "balanced_accuracy",
    "brier",
)
PROFILE_COLUMNS = (
    "view",
    "timing",
    "feature_set",
    "model_type",
    "target",
    "n",
    "auroc",
    "auprc",
    "balanced_accuracy",
)

_FORBIDDEN_PATIENT_LEVEL_COLUMNS = {
    "patient_id",
    "trial_id",
    "subject_id",
    "clinical-trial-subject-id",
    "y_true",
    "y_score",
    "predicted_probability",
    "predicted_label",
    "prediction_probability",
}


class AggregateDataError(ValueError):
    """Raised when a required aggregate artifact is absent or malformed."""


def _read_aggregate_csv(
    path: Path, required_columns: Iterable[str], label: str
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing required aggregate {label}: {path}. Run the experiment "
            "aggregation before generating figures."
        )
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - pandas supplies the detail
        raise AggregateDataError(f"Could not read {label} at {path}: {exc}") from exc
    if frame.empty:
        raise AggregateDataError(f"Required aggregate {label} is empty: {path}")
    missing = sorted(set(required_columns).difference(frame.columns))
    if missing:
        raise AggregateDataError(
            f"{label} is missing required columns {missing}; observed "
            f"{list(frame.columns)}"
        )
    forbidden = sorted(
        _FORBIDDEN_PATIENT_LEVEL_COLUMNS.intersection(
            str(column).strip().lower() for column in frame.columns
        )
    )
    if forbidden:
        raise AggregateDataError(
            f"{label} contains patient-level columns {forbidden}; figure "
            "generation accepts aggregate metrics only"
        )
    return frame


def _numeric(
    frame: pd.DataFrame,
    columns: Iterable[str],
    label: str,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> None:
    for column in columns:
        try:
            values = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
        except Exception as exc:
            raise AggregateDataError(f"{label}.{column} must be numeric") from exc
        if not np.isfinite(values).all():
            raise AggregateDataError(f"{label}.{column} contains non-finite values")
        if lower is not None and np.any(values < lower):
            raise AggregateDataError(f"{label}.{column} contains values below {lower}")
        if upper is not None and np.any(values > upper):
            raise AggregateDataError(f"{label}.{column} contains values above {upper}")


def _filter_value(frame: pd.DataFrame, column: str, value: str, label: str) -> pd.DataFrame:
    if column not in frame.columns:
        return frame
    mask = frame[column].astype(str).str.strip().str.casefold().eq(value.casefold())
    filtered = frame.loc[mask].copy()
    if filtered.empty:
        observed = sorted(frame[column].dropna().astype(str).unique().tolist())
        raise AggregateDataError(
            f"{label} has no {column}={value!r} rows; observed {observed}"
        )
    return filtered


def _primary_rows(
    frame: pd.DataFrame, label: str, *, logistic: bool = True
) -> pd.DataFrame:
    selected = frame.copy()
    for column, value in PRIMARY_FILTERS.items():
        selected = _filter_value(selected, column, value, label)
    if logistic:
        selected = _filter_value(selected, "model_type", "logistic", label)
    return selected


def _require_values(
    frame: pd.DataFrame, column: str, required: Iterable[str], label: str
) -> None:
    observed = set(frame[column].dropna().astype(str))
    missing = [value for value in required if value not in observed]
    if missing:
        raise AggregateDataError(
            f"{label} lacks required {column} values {missing}; observed {sorted(observed)}"
        )


def _require_unique(frame: pd.DataFrame, keys: Sequence[str], label: str) -> None:
    duplicated = frame.duplicated(list(keys), keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, list(keys)].head(5).to_dict("records")
        raise AggregateDataError(
            f"{label} has duplicate aggregate cells for {list(keys)}: {examples}"
        )


def _resolve_column(frame: pd.DataFrame, candidates: Sequence[str], label: str) -> str:
    matches = [column for column in candidates if column in frame.columns]
    if not matches:
        raise AggregateDataError(
            f"{label} requires one of columns {list(candidates)}; observed "
            f"{list(frame.columns)}"
        )
    return matches[0]


def _standardize_effects(frame: pd.DataFrame) -> pd.DataFrame:
    estimate = _resolve_column(
        frame,
        ("delta_auroc", "delta_auroc_estimate", "auroc_delta"),
        "incremental effects AUROC estimate",
    )
    low = _resolve_column(
        frame,
        (
            "delta_auroc_ci_low",
            "delta_auroc_ci_lower",
            "delta_auroc_ci95_low",
            "delta_auroc_low",
            "auroc_ci_low",
        ),
        "incremental effects AUROC lower CI",
    )
    high = _resolve_column(
        frame,
        (
            "delta_auroc_ci_high",
            "delta_auroc_ci_upper",
            "delta_auroc_ci95_high",
            "delta_auroc_high",
            "auroc_ci_high",
        ),
        "incremental effects AUROC upper CI",
    )
    result = frame.copy()
    result["_delta_auroc"] = pd.to_numeric(result[estimate], errors="raise")
    result["_ci_low"] = pd.to_numeric(result[low], errors="raise")
    result["_ci_high"] = pd.to_numeric(result[high], errors="raise")
    _numeric(result, ("_delta_auroc", "_ci_low", "_ci_high"), "incremental effects")
    if (result["_ci_low"] > result["_ci_high"]).any():
        raise AggregateDataError("incremental effects contain lower CI above upper CI")
    if (
        (result["_delta_auroc"] < result["_ci_low"] - 1e-12)
        | (result["_delta_auroc"] > result["_ci_high"] + 1e-12)
    ).any():
        raise AggregateDataError(
            "incremental effects contain an AUROC estimate outside its CI"
        )
    return result


def _validate_pcr(frame: pd.DataFrame) -> pd.DataFrame:
    selected = _primary_rows(frame, "pCR metrics")
    _numeric(
        selected,
        ("auroc", "auprc", "balanced_accuracy", "brier"),
        "pCR metrics",
        lower=0.0,
        upper=1.0,
    )
    _numeric(selected, ("n", "n_positive"), "pCR metrics", lower=0.0)
    _require_values(selected, "view", VIEWS, "pCR metrics")
    _require_values(selected, "timing", TIMINGS, "pCR metrics")
    _require_values(selected, "model", MODEL_ORDER, "pCR metrics")
    selected = selected.loc[
        selected["view"].isin(VIEWS)
        & selected["timing"].isin(TIMINGS)
        & selected["model"].isin(MODEL_ORDER)
    ].copy()
    _require_unique(selected, ("view", "timing", "model"), "primary pCR metrics")
    expected = {(view, timing, model) for view in VIEWS for timing in TIMINGS for model in MODEL_ORDER}
    observed = set(selected[["view", "timing", "model"]].itertuples(index=False, name=None))
    missing = sorted(expected.difference(observed))
    if missing:
        raise AggregateDataError(f"primary pCR metrics lack required cells: {missing[:12]}")
    return selected


def _validate_effects(frame: pd.DataFrame) -> pd.DataFrame:
    selected = _primary_rows(_standardize_effects(frame), "incremental effects")
    _require_values(selected, "comparison", COMPARISON_ORDER, "incremental effects")
    _require_values(selected, "view", VIEWS, "incremental effects")
    _require_values(selected, "timing", TIMINGS, "incremental effects")
    selected = selected.loc[
        selected["comparison"].isin(COMPARISON_ORDER)
        & selected["view"].isin(VIEWS)
        & selected["timing"].isin(TIMINGS)
    ].copy()
    _require_unique(
        selected, ("comparison", "view", "timing"), "primary incremental effects"
    )
    expected = {
        (comparison, view, timing)
        for comparison in COMPARISON_ORDER
        for view in VIEWS
        for timing in TIMINGS
    }
    observed = set(
        selected[["comparison", "view", "timing"]].itertuples(index=False, name=None)
    )
    missing = sorted(expected.difference(observed))
    if missing:
        raise AggregateDataError(f"incremental effects lack required cells: {missing[:12]}")
    return selected


def _validate_profile(frame: pd.DataFrame) -> pd.DataFrame:
    selected = _primary_rows(frame, "profile metrics")
    _numeric(
        selected,
        ("auroc", "auprc", "balanced_accuracy"),
        "profile metrics",
        lower=0.0,
        upper=1.0,
    )
    _numeric(selected, ("n",), "profile metrics", lower=0.0)
    _require_values(selected, "feature_set", ("N", "FULL"), "profile metrics")
    _require_values(selected, "target", ("HR", "HER2"), "profile metrics")
    _require_values(selected, "view", VIEWS, "profile metrics")
    _require_values(selected, "timing", TIMINGS, "profile metrics")
    selected = selected.loc[
        selected["feature_set"].isin(("N", "FULL"))
        & selected["target"].isin(("HR", "HER2", "subtype_4class", "subtype"))
        & selected["view"].isin(VIEWS)
        & selected["timing"].isin(TIMINGS)
    ].copy()
    _require_unique(
        selected, ("view", "timing", "feature_set", "target"), "profile metrics"
    )
    expected = {
        (view, timing, feature_set, target)
        for view in VIEWS
        for timing in TIMINGS
        for feature_set in ("N", "FULL")
        for target in ("HR", "HER2")
    }
    observed = set(
        selected[["view", "timing", "feature_set", "target"]].itertuples(
            index=False, name=None
        )
    )
    missing = sorted(expected.difference(observed))
    if missing:
        raise AggregateDataError(f"profile metrics lack required HR/HER2 cells: {missing[:12]}")
    return selected


def _validate_family(frame: pd.DataFrame) -> pd.DataFrame:
    selected = _primary_rows(frame, "family ablation metrics")
    _numeric(selected, ("auroc",), "family ablation metrics", lower=0.0, upper=1.0)
    _require_values(selected, "model", FAMILY_MODELS, "family ablation metrics")
    _require_values(selected, "view", VIEWS, "family ablation metrics")
    _require_values(selected, "timing", TIMINGS, "family ablation metrics")
    selected = selected.loc[
        selected["model"].isin(FAMILY_MODELS)
        & selected["view"].isin(VIEWS)
        & selected["timing"].isin(TIMINGS)
    ].copy()
    _require_unique(selected, ("view", "timing", "model"), "family ablation metrics")
    return selected


def _validate_residual(frame: pd.DataFrame) -> pd.DataFrame:
    selected = _primary_rows(frame, "residualization metrics")
    _numeric(selected, ("auroc",), "residualization metrics", lower=0.0, upper=1.0)
    _require_values(
        selected, "model", ("N_res", "C+F+N_res"), "residualization metrics"
    )
    _require_values(selected, "view", VIEWS, "residualization metrics")
    _require_values(selected, "timing", TIMINGS, "residualization metrics")
    selected = selected.loc[
        selected["model"].isin(("N_res", "C+F+N_res"))
        & selected["view"].isin(VIEWS)
        & selected["timing"].isin(TIMINGS)
    ].copy()
    _require_unique(selected, ("view", "timing", "model"), "residualization metrics")
    return selected


def _load_correlation(path: Path) -> tuple[list[str], np.ndarray]:
    frame = _read_aggregate_csv(path, (), "feature correlation matrix")
    index_column: str | None = None
    if len(frame.columns) == len(frame) + 1:
        index_column = str(frame.columns[0])
    elif str(frame.columns[0]).lower().startswith("unnamed"):
        index_column = str(frame.columns[0])
    if index_column is not None:
        labels = frame[index_column].astype(str).tolist()
        numeric_frame = frame.drop(columns=[index_column])
    else:
        labels = [str(column) for column in frame.columns]
        numeric_frame = frame
    if numeric_frame.shape[0] != numeric_frame.shape[1]:
        raise AggregateDataError(
            "feature correlation matrix must be square after its optional index column; "
            f"observed {numeric_frame.shape}"
        )
    try:
        matrix = numeric_frame.apply(pd.to_numeric, errors="raise").to_numpy(float)
    except Exception as exc:
        raise AggregateDataError("feature correlation matrix must be numeric") from exc
    if not np.isfinite(matrix).all() or np.any(np.abs(matrix) > 1.000001):
        raise AggregateDataError(
            "feature correlation matrix contains non-finite values or correlations outside [-1, 1]"
        )
    column_labels = [str(column) for column in numeric_frame.columns]
    if index_column is not None and set(labels) != set(column_labels):
        raise AggregateDataError(
            "feature correlation matrix row and column labels do not match"
        )
    if index_column is not None and labels != column_labels:
        order = {label: index for index, label in enumerate(column_labels)}
        matrix = matrix[:, [order[label] for label in labels]]
    return labels, matrix


def load_figure_inputs(metrics_dir: Path) -> Mapping[str, object]:
    """Load and validate all inputs before any figure is written."""

    pcr = _read_aggregate_csv(metrics_dir / "pcr_oof_metrics.csv", PCR_COLUMNS, "pCR metrics")
    effects = _read_aggregate_csv(
        metrics_dir / "incremental_effects.csv",
        ("comparison", "view", "timing"),
        "incremental effects",
    )
    profile = _read_aggregate_csv(
        metrics_dir / "profile_oof_metrics.csv", PROFILE_COLUMNS, "profile metrics"
    )
    family = _read_aggregate_csv(
        metrics_dir / "family_ablation_metrics.csv",
        ("view", "timing", "model_type", "model", "auroc"),
        "family ablation metrics",
    )
    residual = _read_aggregate_csv(
        metrics_dir / "residualization_metrics.csv",
        ("view", "timing", "model_type", "model", "auroc"),
        "residualization metrics",
    )
    correlation = _load_correlation(metrics_dir / "feature_correlation_matrix.csv")
    return {
        "pcr": _validate_pcr(pcr),
        "effects": _validate_effects(effects),
        "profile": _validate_profile(profile),
        "family": _validate_family(family),
        "residual": _validate_residual(residual),
        "correlation": correlation,
    }


def _style_axis(ax: plt.Axes, *, chance: bool = True) -> None:
    if chance:
        ax.axhline(0.5, color="#777777", linewidth=0.9, linestyle="--", zorder=0)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0.45, 1.0)


def _plot_timing_auroc(pcr: pd.DataFrame) -> plt.Figure:
    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.1), sharey=True, constrained_layout=True)
    for ax, view in zip(axes, VIEWS):
        subset = pcr.loc[pcr["view"].eq(view)]
        for index, model in enumerate(MODEL_ORDER):
            rows = subset.loc[subset["model"].eq(model)].set_index("timing").loc[list(TIMINGS)]
            ax.plot(
                range(len(TIMINGS)),
                rows["auroc"].to_numpy(float),
                marker="o",
                linewidth=1.8,
                markersize=4.5,
                color=colors[index % len(colors)],
                label=model,
            )
        ax.set_title(f"{view.capitalize()} view")
        ax.set_xticks(range(len(TIMINGS)), [TIMING_LABELS[t] for t in TIMINGS], rotation=18)
        ax.set_xlabel("Available information timing")
        _style_axis(ax)
    axes[0].set_ylabel("OOF AUROC")
    axes[1].legend(title="Feature set", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Timing-safe pCR performance (primary matched complete-case, logistic)")
    return fig


def _plot_cf_vs_full(pcr: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), sharey=True, constrained_layout=True)
    styles = {"C+F": ("#4c78a8", "o"), "C+FULL": ("#e45756", "s")}
    for ax, view in zip(axes, VIEWS):
        subset = pcr.loc[pcr["view"].eq(view)]
        for model, (color, marker) in styles.items():
            rows = subset.loc[subset["model"].eq(model)].set_index("timing").loc[list(TIMINGS)]
            ax.plot(
                range(len(TIMINGS)), rows["auroc"], color=color, marker=marker,
                linewidth=2.3, markersize=6, label=model,
            )
        ax.set_title(f"{view.capitalize()} view")
        ax.set_xticks(range(len(TIMINGS)), [TIMING_LABELS[t] for t in TIMINGS], rotation=18)
        ax.set_xlabel("Available information timing")
        _style_axis(ax)
    axes[0].set_ylabel("OOF AUROC")
    axes[1].legend(loc="lower right")
    fig.suptitle("Does NONFTV phenotype improve Clinical + FTV?")
    return fig


def _plot_effect_forest(effects: pd.DataFrame) -> plt.Figure:
    labels: list[str] = []
    rows: list[pd.Series] = []
    comparison_labels = {
        "C+FULL_vs_C+F": "C+FULL - C+F",
        "C+N_vs_C": "C+N - C",
        "C+F+N_res_vs_C+F": "C+F+Nres - C+F",
    }
    for comparison in COMPARISON_ORDER:
        for view in VIEWS:
            for timing in TIMINGS:
                row = effects.loc[
                    effects["comparison"].eq(comparison)
                    & effects["view"].eq(view)
                    & effects["timing"].eq(timing)
                ].iloc[0]
                labels.append(
                    f"{comparison_labels[comparison]} | {view} | {TIMING_LABELS[timing]}"
                )
                rows.append(row)
    estimates = np.asarray([float(row["_delta_auroc"]) for row in rows])
    lows = np.asarray([float(row["_ci_low"]) for row in rows])
    highs = np.asarray([float(row["_ci_high"]) for row in rows])
    y = np.arange(len(rows))[::-1]
    colors = np.repeat(("#e45756", "#4c78a8", "#72b7b2"), len(VIEWS) * len(TIMINGS))
    fig, ax = plt.subplots(figsize=(12.2, 10.2), constrained_layout=True)
    for estimate, low, high, y_value, color in zip(
        estimates, lows, highs, y, colors
    ):
        ax.hlines(y_value, low, high, color=color, linewidth=1.5, zorder=1)
        ax.vlines((low, high), y_value - 0.12, y_value + 0.12, color=color, linewidth=1)
    ax.scatter(estimates, y, c=colors, s=28, zorder=2)
    ax.axvline(0.0, color="#333333", linestyle="--", linewidth=1)
    ax.set_yticks(y, labels, fontsize=8.2)
    ax.set_xlabel("Paired patient-bootstrap ΔAUROC (augmented - reference), 95% CI")
    ax.grid(axis="x", color="#dddddd", linewidth=0.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title("Incremental effects on matched patients")
    return fig


def _plot_family(family: pd.DataFrame) -> plt.Figure:
    colors = ("#4c78a8", "#f58518", "#54a24b", "#b279a2")
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), sharey=True, constrained_layout=True)
    for ax, view in zip(axes, VIEWS):
        subset = family.loc[family["view"].eq(view)]
        for model, color in zip(FAMILY_MODELS, colors):
            rows = subset.loc[subset["model"].eq(model)].set_index("timing").loc[list(TIMINGS)]
            ax.plot(range(4), rows["auroc"], marker="o", linewidth=2, color=color, label=model)
        ax.set_title(f"{view.capitalize()} view")
        ax.set_xticks(range(4), [TIMING_LABELS[t] for t in TIMINGS], rotation=18)
        ax.set_xlabel("Available information timing")
        _style_axis(ax)
    axes[0].set_ylabel("OOF AUROC")
    axes[1].legend(title="Preregistered ablation", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Which classical phenotype family adds to Clinical + FTV?")
    return fig


def _plot_profile_heatmap(profile: pd.DataFrame) -> plt.Figure:
    targets = [target for target in ("HR", "HER2", "subtype_4class", "subtype") if target in set(profile["target"])]
    # Do not show two aliases for the same four-class endpoint.
    if "subtype_4class" in targets and "subtype" in targets:
        targets.remove("subtype")
    row_keys = [(feature_set, target) for feature_set in ("N", "FULL") for target in targets]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), constrained_layout=True)
    image = None
    for ax, view in zip(axes, VIEWS):
        matrix = np.empty((len(row_keys), len(TIMINGS)), dtype=float)
        for i, (feature_set, target) in enumerate(row_keys):
            for j, timing in enumerate(TIMINGS):
                rows = profile.loc[
                    profile["view"].eq(view)
                    & profile["timing"].eq(timing)
                    & profile["feature_set"].eq(feature_set)
                    & profile["target"].eq(target)
                ]
                matrix[i, j] = np.nan if rows.empty else float(rows.iloc[0]["auroc"])
        image = ax.imshow(matrix, cmap="viridis", vmin=0.45, vmax=0.85, aspect="auto")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                ax.text(j, i, "NA" if np.isnan(value) else f"{value:.3f}", ha="center", va="center", color="white" if np.isfinite(value) and value < 0.66 else "black", fontsize=8)
        ax.set_xticks(range(4), [TIMING_LABELS[t] for t in TIMINGS], rotation=18)
        ax.set_yticks(range(len(row_keys)), [f"{feature_set} → {target}" for feature_set, target in row_keys])
        ax.set_title(f"{view.capitalize()} view")
    assert image is not None
    fig.colorbar(image, ax=axes, shrink=0.82, label="OOF AUROC")
    fig.suptitle("HR/HER2-associated information in classical DCE phenotype")
    return fig


def _plot_residualized(pcr: pd.DataFrame, residual: pd.DataFrame) -> plt.Figure:
    combined = pd.concat(
        [
            pcr.loc[pcr["model"].isin(("N", "C+F")), ["view", "timing", "model", "auroc"]],
            residual.loc[:, ["view", "timing", "model", "auroc"]],
        ],
        ignore_index=True,
    )
    colors = {"N": "#4c78a8", "N_res": "#72b7b2", "C+F": "#f58518", "C+F+N_res": "#e45756"}
    fig, axes = plt.subplots(1, 2, figsize=(12.3, 4.9), sharey=True, constrained_layout=True)
    for ax, view in zip(axes, VIEWS):
        subset = combined.loc[combined["view"].eq(view)]
        for model in RESIDUAL_MODELS:
            rows = subset.loc[subset["model"].eq(model)].set_index("timing")
            missing = [timing for timing in TIMINGS if timing not in rows.index]
            if missing:
                raise AggregateDataError(
                    f"residualized figure lacks {model} cells for {view}: {missing}"
                )
            rows = rows.loc[list(TIMINGS)]
            ax.plot(range(4), rows["auroc"], marker="o", linewidth=2, color=colors[model], label=model)
        ax.set_title(f"{view.capitalize()} view")
        ax.set_xticks(range(4), [TIMING_LABELS[t] for t in TIMINGS], rotation=18)
        ax.set_xlabel("Available information timing")
        _style_axis(ax)
    axes[0].set_ylabel("OOF AUROC")
    axes[1].legend(title="Feature set", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("pCR signal after outer-train-only FTV residualization")
    return fig


def _plot_correlation(labels: Sequence[str], matrix: np.ndarray) -> plt.Figure:
    size = max(7.2, min(15.0, 0.52 * len(labels) + 4.0))
    fig, ax = plt.subplots(figsize=(size, size), constrained_layout=True)
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_title("Feature correlation matrix (aggregate, outcome-free)")
    fig.colorbar(image, ax=ax, shrink=0.78, label="Correlation")
    return fig


def generate_figures(
    metrics_dir: Path, output_dir: Path, *, dpi: int = 180
) -> tuple[Path, ...]:
    """Generate and atomically publish the complete preregistered figure set."""

    if dpi < 72:
        raise ValueError("dpi must be at least 72")
    inputs = load_figure_inputs(metrics_dir)
    correlation_labels, correlation_matrix = inputs["correlation"]  # type: ignore[misc]
    builders = (
        (FIGURE_FILENAMES[0], lambda: _plot_timing_auroc(inputs["pcr"])),
        (FIGURE_FILENAMES[1], lambda: _plot_cf_vs_full(inputs["pcr"])),
        (FIGURE_FILENAMES[2], lambda: _plot_effect_forest(inputs["effects"])),
        (FIGURE_FILENAMES[3], lambda: _plot_family(inputs["family"])),
        (FIGURE_FILENAMES[4], lambda: _plot_profile_heatmap(inputs["profile"])),
        (FIGURE_FILENAMES[5], lambda: _plot_residualized(inputs["pcr"], inputs["residual"])),
        (FIGURE_FILENAMES[6], lambda: _plot_correlation(correlation_labels, correlation_matrix)),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary: list[tuple[Path, Path]] = []
    try:
        for filename, builder in builders:
            destination = output_dir / filename
            temp = output_dir / f".{filename}.tmp"
            temporary.append((temp, destination))
            figure = builder()
            try:
                figure.savefig(temp, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
            finally:
                plt.close(figure)
        for temp, destination in temporary:
            os.replace(temp, destination)
    except Exception:
        for temp, _ in temporary:
            temp.unlink(missing_ok=True)
        raise
    return tuple(destination for _, destination in temporary)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=EXPERIMENT_ROOT / "metrics",
        help="Directory containing aggregate experiment CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXPERIMENT_ROOT / "figures",
        help="Destination directory for the seven PNG figures.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    outputs = generate_figures(args.metrics_dir.resolve(), args.output_dir.resolve(), dpi=args.dpi)
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

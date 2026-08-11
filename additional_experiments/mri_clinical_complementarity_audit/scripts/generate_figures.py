#!/usr/bin/env python3
"""Render the six aggregate-only complementarity-audit figures.

The script reads only deidentified aggregate CSVs from ``metrics/``.  It never
opens feature, prediction, or patient-level files.  Seed/arm cell estimates are
shown wherever the plot geometry permits; black/colored summaries are simple
visual arithmetic means and are not substituted for the prespecified formal
metrics or confidence intervals.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DPI = 220
PNG_METADATA = {"Software": "MRI Clinical Complementarity Audit"}

PROFILE_COLUMNS = ("seed", "arm", "view", "target", "auroc")
PCR_COLUMNS = ("population", "seed", "arm", "timing", "model", "auroc")
BOOTSTRAP_COLUMNS = (
    "population",
    "comparison",
    "metric",
    "seed",
    "arm",
    "timing",
    "improvement",
    "ci_lower",
    "ci_upper",
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

TIMING_ORDER = ("T0", "T1", "T2", "T3")
MODEL_ORDER = (
    "C",
    "M",
    "F",
    "C+F",
    "C+M",
    "C+F+M",
    "M_residual",
    "C+F+M_residual",
)
RESIDUAL_MODELS = ("M", "M_residual", "C+F+M", "C+F+M_residual")
PROFILE_TARGET_ORDER = ("HR", "HER2", "subtype")
VIEW_ORDER = ("T0", "T1", "T2", "T3", "T0-T1", "T0-T2", "T0-T3")
SUBGROUP_ORDER = ("HR+/HER2-", "HR-/HER2-", "HER2+")

FIGURE_FILENAMES = {
    "profile_auroc_heatmap": "profile_auroc_heatmap.png",
    "primary_pcr_auroc_by_timing": "primary_pcr_auroc_by_timing.png",
    "full_cohort_incremental_forest": "full_cohort_incremental_forest.png",
    "beyond_ftv_incremental_forest": "beyond_ftv_incremental_forest.png",
    "residual_mri_comparison": "residual_mri_comparison.png",
    "subgroup_auroc": "subgroup_auroc.png",
}

FORBIDDEN_ROW_LEVEL_COLUMNS = {
    "patient_id",
    "clinical_patient_id",
    "raw_patient_id",
    "y_true",
    "predicted_probability",
    "probability",
    "prediction",
}

MODEL_COLORS = {
    "C": "#4C78A8",
    "M": "#F58518",
    "F": "#54A24B",
    "C+F": "#72B7B2",
    "C+M": "#E45756",
    "C+F+M": "#B279A2",
    "M_residual": "#FF9DA6",
    "C+F+M_residual": "#9D755D",
}
ARM_COLORS = {
    "LOCAL0": "#4C78A8",
    "LOCAL3": "#F58518",
}
SEED_MARKERS = ("o", "s", "^", "D", "P", "X")


def _configure_style() -> None:
    """Set a deterministic, publication-readable matplotlib style."""

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.titleweight": "semibold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.65,
            "legend.frameon": False,
            "lines.linewidth": 1.8,
            "lines.markersize": 6,
        }
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=EXPERIMENT_ROOT,
        help="Experiment directory containing metrics/ and figures/.",
    )
    return parser.parse_args(argv)


def _header(path: Path, *, label: str) -> tuple[str, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label} aggregate CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        try:
            values = tuple(next(reader))
        except StopIteration as error:
            raise ValueError(f"{label} CSV is empty: {path}") from error
    if not values or any(not str(value).strip() for value in values):
        raise ValueError(f"{label} CSV has an empty header field: {path}")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} CSV has duplicate header fields: {path}")
    normalized = {str(value).strip().lower() for value in values}
    forbidden = sorted(normalized.intersection(FORBIDDEN_ROW_LEVEL_COLUMNS))
    if forbidden:
        raise ValueError(
            f"{label} must be aggregate-only; row-level columns are forbidden: {forbidden}"
        )
    return values


def _read_aggregate(path: Path, *, required: Sequence[str], label: str) -> pd.DataFrame:
    """Read only allowlisted aggregate fields after validating the header."""

    header = _header(path, label=label)
    missing = [column for column in required if column not in header]
    if missing:
        raise ValueError(
            f"{label} schema is missing columns {missing}; "
            f"observed={list(header)} in {path}"
        )
    frame = pd.read_csv(path, usecols=list(required))
    if frame.empty:
        raise ValueError(f"{label} contains no aggregate rows: {path}")
    return frame.loc[:, list(required)].copy()


def _text_columns(frame: pd.DataFrame, columns: Iterable[str], *, label: str) -> None:
    for column in columns:
        if frame[column].isna().any():
            raise ValueError(f"{label}.{column} contains missing values")
        values = frame[column].astype(str).str.strip()
        if values.eq("").any():
            raise ValueError(f"{label}.{column} contains empty values")
        frame[column] = values


def _numeric_column(
    frame: pd.DataFrame,
    column: str,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> None:
    try:
        numeric = pd.to_numeric(frame[column], errors="raise").astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}.{column} must be numeric") from error
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{label}.{column} contains NaN or infinity")
    if minimum is not None and np.any(values < minimum):
        raise ValueError(f"{label}.{column} contains values below {minimum}")
    if maximum is not None and np.any(values > maximum):
        raise ValueError(f"{label}.{column} contains values above {maximum}")
    frame[column] = numeric


def _integer_column(
    frame: pd.DataFrame,
    column: str,
    *,
    label: str,
    minimum: int | None = None,
) -> None:
    _numeric_column(frame, column, label=label)
    values = frame[column].to_numpy(dtype=float)
    if not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{label}.{column} must contain integers")
    integers = values.astype(np.int64)
    if minimum is not None and np.any(integers < minimum):
        raise ValueError(f"{label}.{column} contains values below {minimum}")
    frame[column] = integers


def _unique_rows(frame: pd.DataFrame, keys: Sequence[str], *, label: str) -> None:
    duplicated = frame.duplicated(list(keys), keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, list(keys)].head(3).to_dict("records")
        raise ValueError(f"{label} has duplicate aggregate cells: {examples}")


def _require_timings(
    frame: pd.DataFrame, *, label: str, column: str = "timing"
) -> None:
    observed = set(frame[column].astype(str))
    missing = [timing for timing in TIMING_ORDER if timing not in observed]
    if missing:
        raise ValueError(f"{label} is missing required timings: {missing}")


def _validate_profile(frame: pd.DataFrame) -> pd.DataFrame:
    _text_columns(frame, ("arm", "view", "target"), label="profile_oof_metrics")
    _integer_column(frame, "seed", label="profile_oof_metrics")
    _numeric_column(
        frame, "auroc", label="profile_oof_metrics", minimum=0.0, maximum=1.0
    )
    _unique_rows(
        frame,
        ("seed", "arm", "view", "target"),
        label="profile_oof_metrics",
    )
    missing_static = [
        value for value in TIMING_ORDER if value not in set(frame["view"])
    ]
    if missing_static:
        raise ValueError(
            f"profile_oof_metrics is missing static profile views: {missing_static}"
        )
    return frame


def _validate_pcr(frame: pd.DataFrame) -> pd.DataFrame:
    _text_columns(
        frame, ("population", "arm", "timing", "model"), label="pcr_oof_metrics"
    )
    _integer_column(frame, "seed", label="pcr_oof_metrics")
    _numeric_column(frame, "auroc", label="pcr_oof_metrics", minimum=0.0, maximum=1.0)
    _unique_rows(
        frame,
        ("population", "seed", "arm", "timing", "model"),
        label="pcr_oof_metrics",
    )
    return frame


def _validate_bootstrap(frame: pd.DataFrame) -> pd.DataFrame:
    _text_columns(
        frame,
        ("population", "comparison", "metric", "arm", "timing"),
        label="bootstrap_ci",
    )
    frame["metric"] = frame["metric"].str.lower()
    _integer_column(frame, "seed", label="bootstrap_ci")
    for column in ("improvement", "ci_lower", "ci_upper"):
        _numeric_column(frame, column, label="bootstrap_ci", minimum=-1.0, maximum=1.0)
    if np.any(frame["ci_lower"].to_numpy() > frame["ci_upper"].to_numpy()):
        raise ValueError("bootstrap_ci has ci_lower greater than ci_upper")
    _unique_rows(
        frame,
        ("population", "comparison", "metric", "seed", "arm", "timing"),
        label="bootstrap_ci",
    )
    return frame


def _validate_subgroup(frame: pd.DataFrame) -> pd.DataFrame:
    _text_columns(
        frame, ("arm", "timing", "subgroup", "model"), label="subgroup_metrics"
    )
    for column in ("seed", "n", "n_positive"):
        _integer_column(frame, column, label="subgroup_metrics", minimum=0)
    if np.any(frame["n"].to_numpy() <= 0):
        raise ValueError("subgroup_metrics.n must be positive")
    if np.any(frame["n_positive"].to_numpy() > frame["n"].to_numpy()):
        raise ValueError("subgroup_metrics.n_positive exceeds n")
    for column in ("auroc", "auprc", "brier"):
        _numeric_column(
            frame, column, label="subgroup_metrics", minimum=0.0, maximum=1.0
        )
    _unique_rows(
        frame,
        ("seed", "arm", "timing", "subgroup", "model"),
        label="subgroup_metrics",
    )
    _require_timings(frame, label="subgroup_metrics")
    return frame


def _ordered(values: Iterable[str], preferred: Sequence[str]) -> list[str]:
    observed = {str(value) for value in values}
    output = [value for value in preferred if value in observed]
    output.extend(
        sorted(observed.difference(output), key=lambda value: value.casefold())
    )
    return output


def _view_key(value: str) -> tuple[int, str]:
    normalized = str(value).replace("–", "-").replace("+", "-").replace("_", "-")
    if normalized in VIEW_ORDER:
        return VIEW_ORDER.index(normalized), normalized
    return len(VIEW_ORDER), normalized.casefold()


def _ordered_views(values: Iterable[str]) -> list[str]:
    return sorted({str(value) for value in values}, key=_view_key)


def _timing_label(value: str) -> str:
    text = str(value).replace("–", "-").replace("_", "-")
    if text == "T3":
        return "T3\n(late/pre-surgery)"
    if text.endswith("T3") and text != "T3":
        return f"{text.replace('-', '–')}\n(includes late/pre-surgery T3)"
    return text.replace("-", "–")


def _model_label(value: str) -> str:
    return str(value).replace("_residual", " residual")


def _target_label(value: str) -> str:
    normalized = str(value).strip()
    if "subtype" in normalized.lower():
        return "HR/HER2 subtype"
    return normalized.upper() if normalized.lower() in {"hr", "her2"} else normalized


def _palette(values: Sequence[str], preferred: dict[str, str]) -> dict[str, str]:
    colors = dict(preferred)
    fallback = list(plt.get_cmap("tab20").colors)
    missing = [value for value in values if value not in colors]
    for index, value in enumerate(missing):
        colors[value] = matplotlib.colors.to_hex(fallback[index % len(fallback)])
    return {value: colors[value] for value in values}


def _mean_and_cell_plot(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    models: Sequence[str],
    timings: Sequence[str],
    colors: dict[str, str],
) -> None:
    """Draw model means across seed/arm and retain all cell AUROCs."""

    x = np.arange(len(timings), dtype=float)
    for model in models:
        selected = frame.loc[frame["model"].eq(model)]
        missing = [
            timing for timing in timings if not selected["timing"].eq(timing).any()
        ]
        if missing:
            raise ValueError(f"model {model!r} is missing plotted timings: {missing}")
        means = [
            float(selected.loc[selected["timing"].eq(timing), "auroc"].mean())
            for timing in timings
        ]
        color = colors[model]
        ax.plot(
            x,
            means,
            marker="o",
            color=color,
            label=_model_label(model),
            zorder=4,
        )
        for position, timing in enumerate(timings):
            cells = selected.loc[selected["timing"].eq(timing)].sort_values(
                ["arm", "seed"], kind="stable"
            )
            offsets = (
                np.asarray([0.0])
                if len(cells) == 1
                else np.linspace(-0.085, 0.085, len(cells))
            )
            ax.scatter(
                position + offsets,
                cells["auroc"],
                s=24,
                color=color,
                alpha=0.35,
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )
    ax.set_xticks(x, [_timing_label(value) for value in timings])
    ax.set_ylabel("Held-out AUROC")
    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1.0, alpha=0.8)
    values = frame.loc[
        frame["model"].isin(models) & frame["timing"].isin(timings), "auroc"
    ].to_numpy(dtype=float)
    lower = max(0.0, min(0.5, float(values.min())) - 0.04)
    upper = min(1.0, max(0.55, float(values.max())) + 0.04)
    if upper - lower < 0.18:
        midpoint = 0.5 * (upper + lower)
        lower, upper = max(0.0, midpoint - 0.09), min(1.0, midpoint + 0.09)
    ax.set_ylim(lower, upper)


def render_profile_heatmap(frame: pd.DataFrame) -> plt.Figure:
    targets = _ordered(frame["target"], PROFILE_TARGET_ORDER)
    views = _ordered_views(frame["view"])
    missing = [
        (target, view)
        for target in targets
        for view in views
        if frame.loc[frame["target"].eq(target) & frame["view"].eq(view)].empty
    ]
    if missing:
        raise ValueError(
            f"profile heatmap has incomplete target/view cells: {missing[:5]}"
        )

    means = np.empty((len(targets), len(views)), dtype=float)
    minima = np.empty_like(means)
    maxima = np.empty_like(means)
    for row, target in enumerate(targets):
        for column, view in enumerate(views):
            values = frame.loc[
                frame["target"].eq(target) & frame["view"].eq(view), "auroc"
            ].to_numpy(dtype=float)
            means[row, column] = float(values.mean())
            minima[row, column] = float(values.min())
            maxima[row, column] = float(values.max())

    vmin = max(0.0, float(means.min()) - 0.03)
    vmax = min(1.0, float(means.max()) + 0.03)
    if math.isclose(vmin, vmax):
        vmin, vmax = max(0.0, vmin - 0.05), min(1.0, vmax + 0.05)
    figure, ax = plt.subplots(
        figsize=(
            max(9.0, 1.35 * len(views) + 2.5),
            max(4.2, 0.95 * len(targets) + 2.5),
        ),
        constrained_layout=True,
    )
    image = ax.imshow(means, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
    ax.grid(False)
    ax.set_xticks(
        np.arange(len(views)),
        [_timing_label(value) for value in views],
        rotation=25,
        ha="right",
    )
    ax.set_yticks(np.arange(len(targets)), [_target_label(value) for value in targets])
    ax.set_xlabel("Frozen LOCAL MRI representation view")
    ax.set_ylabel("Profile probe target")
    ax.set_title("Patient-profile decodability from frozen LOCAL MRI states")
    midpoint = 0.5 * (vmin + vmax)
    for row in range(len(targets)):
        for column in range(len(views)):
            color = "white" if means[row, column] < midpoint else "#111111"
            ax.text(
                column,
                row,
                f"{means[row, column]:.3f}\n[{minima[row, column]:.3f}, {maxima[row, column]:.3f}]",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )
    colorbar = figure.colorbar(image, ax=ax, shrink=0.85, pad=0.02)
    colorbar.set_label("AUROC")
    figure.text(
        0.5,
        0.005,
        "Cell text: arithmetic mean [minimum, maximum] across seed×arm cells. "
        "T3 is the late/pre-surgery assessment.",
        ha="center",
        fontsize=8.5,
        color="#444444",
    )
    return figure


def render_primary_pcr(frame: pd.DataFrame) -> plt.Figure:
    selected = frame.loc[frame["population"].eq("ftv_complete_375")].copy()
    if selected.empty:
        raise ValueError("pcr_oof_metrics has no population=ftv_complete_375 rows")
    _require_timings(selected, label="primary pCR rows")
    timings = [value for value in TIMING_ORDER if value in set(selected["timing"])]
    models = _ordered(selected["model"], MODEL_ORDER)
    colors = _palette(models, MODEL_COLORS)
    figure, ax = plt.subplots(figsize=(13.0, 7.2), constrained_layout=True)
    _mean_and_cell_plot(ax, selected, models=models, timings=timings, colors=colors)
    ax.set_title("Primary pCR prediction on the FTV-complete cohort (n=375)")
    ax.set_xlabel("Prediction timing")
    ax.legend(title="Model", ncol=2, bbox_to_anchor=(1.01, 1.0), loc="upper left")
    figure.text(
        0.5,
        0.005,
        "Lines show arithmetic means across seed×arm cells; translucent points retain every cell. "
        "T3 is late/pre-surgery.",
        ha="center",
        fontsize=8.5,
        color="#444444",
    )
    return figure


def _seed_marker_map(seeds: Sequence[int]) -> dict[int, str]:
    return {
        seed: SEED_MARKERS[index % len(SEED_MARKERS)]
        for index, seed in enumerate(seeds)
    }


def render_forest(
    frame: pd.DataFrame,
    *,
    population: str,
    comparison: str,
    title: str,
) -> plt.Figure:
    selected = frame.loc[
        frame["population"].eq(population)
        & frame["comparison"].eq(comparison)
        & frame["metric"].eq("auroc")
    ].copy()
    if selected.empty:
        raise ValueError(
            "bootstrap_ci has no rows for "
            f"population={population!r}, comparison={comparison!r}, metric='auroc'"
        )
    _require_timings(selected, label=f"bootstrap {comparison}")
    timings = [value for value in TIMING_ORDER if value in set(selected["timing"])]
    arms = sorted(selected["arm"].unique(), key=str.casefold)
    seeds = sorted(int(value) for value in selected["seed"].unique())
    arm_colors = _palette(arms, ARM_COLORS)
    seed_markers = _seed_marker_map(seeds)

    rows: list[tuple[str, pd.Series | None, float]] = []
    y = 0.0
    for timing in timings:
        cells = selected.loc[selected["timing"].eq(timing)].sort_values(
            ["arm", "seed"], kind="stable"
        )
        for _, cell in cells.iterrows():
            label = f"{_timing_label(timing).replace(chr(10), ' ')} | {cell['arm']} | seed {int(cell['seed'])}"
            rows.append((label, cell, y))
            y += 1.0
        rows.append(
            (
                f"{_timing_label(timing).replace(chr(10), ' ')} | visual mean",
                None,
                y,
            )
        )
        y += 1.7

    figure_height = max(7.0, 0.42 * len(rows) + 2.7)
    figure, ax = plt.subplots(figsize=(11.5, figure_height))
    figure.subplots_adjust(left=0.26, right=0.98, top=0.95, bottom=0.09)
    labels: list[str] = []
    positions: list[float] = []
    for label, cell, position in rows:
        labels.append(label)
        positions.append(position)
        timing = label.split(" | ", 1)[0].split(" ", 1)[0]
        if cell is None:
            mean = float(
                selected.loc[selected["timing"].eq(timing), "improvement"].mean()
            )
            ax.scatter(
                mean,
                position,
                marker="D",
                s=58,
                color="#111111",
                edgecolor="white",
                linewidth=0.6,
                zorder=5,
            )
            continue
        lower = float(cell["ci_lower"])
        upper = float(cell["ci_upper"])
        point = float(cell["improvement"])
        arm = str(cell["arm"])
        seed = int(cell["seed"])
        ax.hlines(
            position, lower, upper, color=arm_colors[arm], linewidth=1.8, alpha=0.9
        )
        ax.scatter(
            point,
            position,
            marker=seed_markers[seed],
            s=44,
            color=arm_colors[arm],
            edgecolor="white",
            linewidth=0.55,
            zorder=4,
        )

    ax.axvline(0.0, color="#333333", linestyle="--", linewidth=1.1)
    ax.set_yticks(positions, labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Held-out AUROC improvement")
    ax.set_title(title)
    all_limits = np.concatenate(
        (
            selected["ci_lower"].to_numpy(dtype=float),
            selected["ci_upper"].to_numpy(dtype=float),
            np.asarray([0.0]),
        )
    )
    margin = max(0.015, 0.08 * float(all_limits.max() - all_limits.min()))
    ax.set_xlim(float(all_limits.min()) - margin, float(all_limits.max()) + margin)
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=arm_colors[arm],
            marker=seed_markers[seed],
            linestyle="-",
            label=f"{arm}, seed {seed}",
        )
        for arm in arms
        for seed in seeds
        if not selected.loc[selected["arm"].eq(arm) & selected["seed"].eq(seed)].empty
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="#111111",
            marker="D",
            linestyle="None",
            label="Arithmetic visual mean",
        )
    )
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8.5)
    figure.text(
        0.5,
        0.018,
        "Colored intervals are cell-specific paired bootstrap CIs. Black diamonds are visual means only, "
        "not pooled confidence intervals. T3 is late/pre-surgery.",
        ha="center",
        fontsize=8.5,
        color="#444444",
    )
    return figure


def render_residual_comparison(frame: pd.DataFrame) -> plt.Figure:
    selected = frame.loc[
        frame["population"].eq("ftv_complete_375")
        & frame["model"].isin(RESIDUAL_MODELS)
    ].copy()
    if selected.empty:
        raise ValueError("pcr_oof_metrics has no residual-comparison rows")
    missing_models = [
        model for model in RESIDUAL_MODELS if model not in set(selected["model"])
    ]
    if missing_models:
        raise ValueError(
            f"pcr_oof_metrics is missing residual models: {missing_models}"
        )
    _require_timings(selected, label="residual pCR rows")
    timings = [value for value in TIMING_ORDER if value in set(selected["timing"])]
    colors = _palette(list(RESIDUAL_MODELS), MODEL_COLORS)
    figure, ax = plt.subplots(figsize=(10.5, 6.7))
    figure.subplots_adjust(left=0.10, right=0.78, top=0.90, bottom=0.18)
    _mean_and_cell_plot(
        ax,
        selected,
        models=list(RESIDUAL_MODELS),
        timings=timings,
        colors=colors,
    )
    ax.set_title("pCR signal before and after fold-train FTV residualization")
    ax.set_xlabel("Prediction timing")
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1.0), loc="upper left")
    figure.text(
        0.5,
        0.025,
        "Lines show arithmetic means across seed×arm cells; translucent points retain every cell. "
        "Residualization is fold-train only; T3 is late/pre-surgery.",
        ha="center",
        fontsize=8.5,
        color="#444444",
    )
    return figure


def _subgroup_n_label(frame: pd.DataFrame) -> str:
    values = sorted(int(value) for value in frame["n"].unique())
    if len(values) == 1:
        return f"n={values[0]}"
    return f"n={values[0]}–{values[-1]} across cells"


def render_subgroup(frame: pd.DataFrame) -> plt.Figure:
    subgroups = _ordered(frame["subgroup"], SUBGROUP_ORDER)
    models = _ordered(frame["model"], MODEL_ORDER)
    timings = [value for value in TIMING_ORDER if value in set(frame["timing"])]
    colors = _palette(models, MODEL_COLORS)
    figure, axes = plt.subplots(
        1,
        len(subgroups),
        figsize=(max(7.0, 5.2 * len(subgroups)), 6.2),
        sharey=True,
        squeeze=False,
    )
    figure.subplots_adjust(left=0.06, right=0.99, top=0.76, bottom=0.18, wspace=0.10)
    for index, subgroup in enumerate(subgroups):
        ax = axes[0, index]
        selected = frame.loc[frame["subgroup"].eq(subgroup)].copy()
        subgroup_models = [model for model in models if model in set(selected["model"])]
        _mean_and_cell_plot(
            ax,
            selected,
            models=subgroup_models,
            timings=timings,
            colors=colors,
        )
        ax.set_title(f"{subgroup}\n{_subgroup_n_label(selected)}")
        ax.set_xlabel("Prediction timing")
        if index > 0:
            ax.set_ylabel("")
    values = frame["auroc"].to_numpy(dtype=float)
    lower = max(0.0, min(0.5, float(values.min())) - 0.04)
    upper = min(1.0, max(0.55, float(values.max())) + 0.04)
    if upper - lower < 0.18:
        midpoint = 0.5 * (upper + lower)
        lower, upper = max(0.0, midpoint - 0.09), min(1.0, midpoint + 0.09)
    axes[0, 0].set_ylim(lower, upper)
    handles = [
        Line2D([0], [0], color=colors[model], marker="o", label=_model_label(model))
        for model in models
    ]
    figure.legend(
        handles=handles,
        title="Model",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=max(1, len(handles)),
    )
    figure.suptitle("Subtype-conditioned pCR AUROC", y=0.995)
    figure.text(
        0.5,
        0.025,
        "Lines show arithmetic means across seed×arm cells; translucent points retain every cell. "
        "Small-subgroup estimates are descriptive. T3 is late/pre-surgery.",
        ha="center",
        fontsize=8.5,
        color="#444444",
    )
    return figure


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(
        path,
        format="png",
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        metadata=PNG_METADATA,
    )
    plt.close(figure)
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"matplotlib did not create a nonempty PNG: {path}")


def _manifest_path(path: Path, experiment_root: Path) -> str:
    try:
        return path.resolve().relative_to(experiment_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _write_manifest(
    outputs: Sequence[tuple[str, Path]], *, experiment_root: Path, metrics_dir: Path
) -> Path:
    rows = [
        {
            "figure_id": figure_id,
            "path": _manifest_path(path, experiment_root),
            "sha256": _sha256(path),
            "size_bytes": int(path.stat().st_size),
        }
        for figure_id, path in outputs
    ]
    manifest = metrics_dir / "figure_manifest.csv"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{manifest.name}.", suffix=".tmp", dir=metrics_dir
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            pd.DataFrame(rows).to_csv(stream, index=False, lineterminator="\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, manifest)
    finally:
        if temporary.exists():
            temporary.unlink()
    return manifest


def render_all(experiment_root: Path) -> tuple[list[Path], Path]:
    _configure_style()
    root = experiment_root.expanduser().resolve()
    metrics_dir = root / "metrics"
    figures_dir = root / "figures"
    if not metrics_dir.is_dir():
        raise FileNotFoundError(f"metrics directory does not exist: {metrics_dir}")

    profile = _validate_profile(
        _read_aggregate(
            metrics_dir / "profile_oof_metrics.csv",
            required=PROFILE_COLUMNS,
            label="profile_oof_metrics",
        )
    )
    pcr = _validate_pcr(
        _read_aggregate(
            metrics_dir / "pcr_oof_metrics.csv",
            required=PCR_COLUMNS,
            label="pcr_oof_metrics",
        )
    )
    bootstrap = _validate_bootstrap(
        _read_aggregate(
            metrics_dir / "bootstrap_ci.csv",
            required=BOOTSTRAP_COLUMNS,
            label="bootstrap_ci",
        )
    )
    subgroup = _validate_subgroup(
        _read_aggregate(
            metrics_dir / "subgroup_metrics.csv",
            required=SUBGROUP_COLUMNS,
            label="subgroup_metrics",
        )
    )

    figures = (
        ("profile_auroc_heatmap", render_profile_heatmap(profile)),
        ("primary_pcr_auroc_by_timing", render_primary_pcr(pcr)),
        (
            "full_cohort_incremental_forest",
            render_forest(
                bootstrap,
                population="full_808",
                comparison="C+M_vs_C",
                title="Incremental MRI value beyond clinical profile (full cohort, n=808)",
            ),
        ),
        (
            "beyond_ftv_incremental_forest",
            render_forest(
                bootstrap,
                population="ftv_complete_375",
                comparison="C+F+M_vs_C+F",
                title="Incremental MRI value beyond clinical profile + FTV (n=375)",
            ),
        ),
        ("residual_mri_comparison", render_residual_comparison(pcr)),
        ("subgroup_auroc", render_subgroup(subgroup)),
    )

    figures_dir.mkdir(parents=True, exist_ok=True)
    completed: list[tuple[str, Path]] = []
    with tempfile.TemporaryDirectory(
        prefix=".figure-build-", dir=figures_dir
    ) as temporary:
        staging = Path(temporary)
        for figure_id, figure in figures:
            staged_path = staging / FIGURE_FILENAMES[figure_id]
            _save_figure(figure, staged_path)
            completed.append((figure_id, staged_path))
        final_outputs: list[tuple[str, Path]] = []
        for figure_id, staged_path in completed:
            final_path = figures_dir / staged_path.name
            os.replace(staged_path, final_path)
            final_outputs.append((figure_id, final_path))

    manifest = _write_manifest(
        final_outputs, experiment_root=root, metrics_dir=metrics_dir
    )
    return [path for _, path in final_outputs], manifest


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _configure_style()
    outputs, manifest = render_all(args.experiment_root)
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "aggregate_only": True,
                "figure_count": len(outputs),
                "figures": [path.name for path in outputs],
                "manifest": str(manifest),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render the compact-fusion audit figures from public aggregate metrics only.

The plotting contract is intentionally narrow: this module reads four named
CSV files under ``metrics/`` and rejects private files and patient-level
columns.  Every point is a seed-by-arm sensitivity cell (or, for PCA variance,
that cell's five-fold mean); thick lines/markers are descriptive means across
the four cells.  Populations are never pooled or interpreted as paired.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS_DIR = EXPERIMENT_ROOT / "metrics"
DEFAULT_FIGURES_DIR = EXPERIMENT_ROOT / "figures"
DEFAULT_MANIFEST = DEFAULT_METRICS_DIR / "figure_manifest.csv"

ALLOWED_AGGREGATE_INPUTS = frozenset(
    {
        "pcr_oof_metrics.csv",
        "paired_effects.csv",
        "profile_oof_metrics.csv",
        "pca_explained_variance.csv",
    }
)
POPULATIONS = (
    ("full_808", "Full cohort (n=808)"),
    ("ftv_complete_375", "FTV-complete cohort (n=375)"),
)
TIMINGS = ("T0", "T1", "T2", "T3")
TIMING_LABELS = {
    "T0": "T0",
    "T1": "T0–T1",
    "T2": "T0–T2",
    "T3": "T0–T3 (late)",
}
DIMENSIONS = (8, 16, 32, 64)
CELL_ORDER = (
    (2026, "LOCAL0"),
    (2026, "LOCAL3"),
    (3026, "LOCAL0"),
    (3026, "LOCAL3"),
)
CELL_MARKERS = {
    (2026, "LOCAL0"): "o",
    (2026, "LOCAL3"): "s",
    (3026, "LOCAL0"): "^",
    (3026, "LOCAL3"): "D",
}
CELL_JITTER = {
    (2026, "LOCAL0"): -0.045,
    (2026, "LOCAL3"): -0.015,
    (3026, "LOCAL0"): 0.015,
    (3026, "LOCAL3"): 0.045,
}
COLORS = {
    "M": "#5E3C99",
    "C+M": "#1B9E77",
    "C+F+M": "#D95F02",
    "C+Mk − C": "#1B9E77",
    "C+F+Mk − C+F": "#D95F02",
    "Compact Mk": "#1B9E77",
    "FTV-residualized Mk": "#7570B3",
    "MRI only": "#5E3C99",
    "C + MRI": "#1B9E77",
    "C + FTV + MRI": "#D95F02",
    "Late(C,Mk) − concat(C,Mk)": "#1B9E77",
    "Late(C+F,Mk) − concat(C+F,Mk)": "#D95F02",
    "Raw prefix": "#4D4D4D",
    "PCA-16": "#1B9E77",
    "PCA-32": "#7570B3",
}
LATE_FACE_COLOR = "#FFF4E6"
MEAN_EDGE_COLOR = "#202020"
FOOTNOTE = (
    "Small markers: seed×arm sensitivity cells (not independent patients). "
    "Thick markers/lines: descriptive mean across the four cells. "
    "T3 is late/pre-surgery."
)


@dataclass(frozen=True)
class FigureResult:
    filename: str
    title: str
    description: str
    source_files: tuple[str, ...]
    source_rows_used: int
    point_unit: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_aggregate(metrics_dir: Path, filename: str) -> pd.DataFrame:
    """Read one allowlisted aggregate table and reject patient-level schemas."""

    if filename not in ALLOWED_AGGREGATE_INPUTS:
        raise ValueError(f"aggregate input is not allowlisted: {filename}")
    if ".private." in filename:
        raise ValueError(f"private input is forbidden: {filename}")
    metrics_root = metrics_dir.resolve(strict=True)
    path = (metrics_root / filename).resolve(strict=True)
    if path.parent != metrics_root:
        raise ValueError(f"aggregate input escaped metrics directory: {path}")
    frame = pd.read_csv(path)
    forbidden = {"patient_id", "predicted_probability", "true_label"}
    overlap = forbidden.intersection(frame.columns)
    if overlap:
        raise ValueError(
            f"aggregate input {filename} has patient-level columns: {sorted(overlap)}"
        )
    if frame.empty:
        raise ValueError(f"aggregate input is empty: {filename}")
    return frame


def _require_columns(frame: pd.DataFrame, required: Iterable[str], source: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")


def _require_finite(frame: pd.DataFrame, columns: Iterable[str], source: str) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{source}.{column} contains missing/non-finite values")


def _assert_unique(frame: pd.DataFrame, keys: Sequence[str], source: str) -> None:
    duplicated = frame.duplicated(list(keys), keep=False)
    if duplicated.any():
        example = frame.loc[duplicated, list(keys)].head(2).to_dict("records")
        raise ValueError(f"{source} has duplicate aggregate cells: {example}")


def _require_seed_arm_cells(
    frame: pd.DataFrame, group_keys: Sequence[str], source: str
) -> None:
    expected = set(CELL_ORDER)
    for group, rows in frame.groupby(list(group_keys), sort=False, dropna=False):
        observed = set(zip(rows["seed"].astype(int), rows["arm"].astype(str)))
        if observed != expected:
            raise ValueError(
                f"{source} sensitivity-cell coverage for {group!r} is "
                f"{sorted(observed)!r}; expected {sorted(expected)!r}"
            )


def _validate_shared_domains(frame: pd.DataFrame, source: str) -> None:
    populations = set(frame["population"].astype(str))
    unknown_populations = populations.difference(population for population, _ in POPULATIONS)
    if unknown_populations:
        raise ValueError(f"{source} has unknown populations: {unknown_populations}")
    timings = set(frame["timing"].astype(str))
    unknown_timings = timings.difference(TIMINGS)
    if unknown_timings:
        raise ValueError(f"{source} has unknown timings: {unknown_timings}")


def _style_axis(ax: plt.Axes, *, timing: str | None = None) -> None:
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if timing == "T3":
        ax.set_facecolor(LATE_FACE_COLOR)


def _cell_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color="#6F6F6F",
            marker=CELL_MARKERS[cell],
            linestyle="None",
            markersize=6,
            label=f"{cell[0]} · {cell[1]}",
        )
        for cell in CELL_ORDER
    ]


def _series_handles(series: Mapping[str, str]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=color,
            marker="o",
            markeredgecolor=MEAN_EDGE_COLOR,
            linewidth=2.2,
            markersize=7,
            label=label,
        )
        for label, color in series.items()
    ]


def _add_figure_legend(
    fig: plt.Figure,
    series: Mapping[str, str],
    *,
    y: float = 0.925,
) -> None:
    handles = _series_handles(series) + _cell_handles()
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=min(len(handles), 7),
        frameon=False,
        fontsize=8.5,
        handlelength=2.0,
        columnspacing=1.3,
    )


def _scatter_cell_points(
    ax: plt.Axes,
    rows: pd.DataFrame,
    *,
    x: float,
    color: str,
    value_column: str,
) -> None:
    for cell in CELL_ORDER:
        selected = rows[
            rows["seed"].astype(int).eq(cell[0])
            & rows["arm"].astype(str).eq(cell[1])
        ]
        if len(selected) != 1:
            raise ValueError(
                f"expected one value for seed×arm cell {cell}; found {len(selected)}"
            )
        ax.scatter(
            x + CELL_JITTER[cell],
            float(selected.iloc[0][value_column]),
            color=color,
            marker=CELL_MARKERS[cell],
            s=26,
            alpha=0.48,
            linewidth=0.4,
            edgecolor="white",
            zorder=2,
        )


def _plot_dimension_series(
    ax: plt.Axes,
    panel: pd.DataFrame,
    *,
    series_column: str,
    series_labels: Sequence[str],
    value_column: str,
) -> None:
    x_base = np.arange(len(DIMENSIONS), dtype=float)
    offsets = dict(
        zip(series_labels, np.linspace(-0.16, 0.16, len(series_labels)))
    )
    for label in series_labels:
        color = COLORS[label]
        means: list[float] = []
        for dimension_index, dimension in enumerate(DIMENSIONS):
            rows = panel[
                panel[series_column].astype(str).eq(label)
                & panel["dimension"].astype(int).eq(dimension)
            ]
            if len(rows) != len(CELL_ORDER):
                raise ValueError(
                    f"expected four cells for {label}, dimension {dimension}; "
                    f"found {len(rows)}"
                )
            x_value = x_base[dimension_index] + offsets[label]
            _scatter_cell_points(
                ax,
                rows,
                x=x_value,
                color=color,
                value_column=value_column,
            )
            means.append(float(rows[value_column].mean()))
        ax.plot(
            x_base + offsets[label],
            means,
            color=color,
            marker="o",
            markeredgecolor=MEAN_EDGE_COLOR,
            markeredgewidth=0.7,
            markersize=6.5,
            linewidth=2.1,
            zorder=3,
        )
    ax.set_xticks(x_base, [str(value) for value in DIMENSIONS])
    ax.set_xlabel("PCA total dimension k")


def _plot_timing_series(
    ax: plt.Axes,
    panel: pd.DataFrame,
    *,
    series_column: str,
    series_labels: Sequence[str],
    value_column: str,
) -> None:
    x_base = np.arange(len(TIMINGS), dtype=float)
    offsets = dict(
        zip(series_labels, np.linspace(-0.16, 0.16, len(series_labels)))
    )
    for label in series_labels:
        color = COLORS[label]
        means: list[float] = []
        for timing_index, timing in enumerate(TIMINGS):
            rows = panel[
                panel[series_column].astype(str).eq(label)
                & panel["timing"].astype(str).eq(timing)
            ]
            if len(rows) != len(CELL_ORDER):
                raise ValueError(
                    f"expected four cells for {label}, timing {timing}; "
                    f"found {len(rows)}"
                )
            x_value = x_base[timing_index] + offsets[label]
            _scatter_cell_points(
                ax,
                rows,
                x=x_value,
                color=color,
                value_column=value_column,
            )
            means.append(float(rows[value_column].mean()))
        ax.plot(
            x_base + offsets[label],
            means,
            color=color,
            marker="o",
            markeredgecolor=MEAN_EDGE_COLOR,
            markeredgewidth=0.7,
            markersize=7,
            linewidth=2.2,
            zorder=3,
        )
    ax.set_xticks(x_base, [TIMING_LABELS[timing] for timing in TIMINGS])
    ax.axvspan(2.5, 3.5, color=LATE_FACE_COLOR, zorder=-2)


def _finish_figure(
    fig: plt.Figure,
    output_path: Path,
    *,
    title: str,
    footnote: str = FOOTNOTE,
) -> None:
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.988)
    fig.text(0.5, 0.012, footnote, ha="center", va="bottom", fontsize=8.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "compact_mri_clinical_fusion_audit"},
    )
    plt.close(fig)
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"figure was not written: {output_path}")


def figure_auroc_by_dimensionality(
    metrics: pd.DataFrame, figures_dir: Path
) -> FigureResult:
    source = "pcr_oof_metrics.csv"
    required = {
        "population",
        "seed",
        "arm",
        "timing",
        "model_family",
        "representation",
        "dimension",
        "auroc",
    }
    _require_columns(metrics, required, source)
    selected = metrics[
        metrics["representation"].astype(str).eq("pca")
        & metrics["model_family"].astype(str).isin(["M", "C+M", "C+F+M"])
        & metrics["dimension"].astype(int).isin(DIMENSIONS)
    ].copy()
    _validate_shared_domains(selected, source)
    _require_finite(selected, ["dimension", "auroc"], source)
    _assert_unique(
        selected,
        ["population", "seed", "arm", "timing", "model_family", "dimension"],
        source,
    )
    _require_seed_arm_cells(
        selected,
        ["population", "timing", "model_family", "dimension"],
        source,
    )

    fig, axes = plt.subplots(2, 4, figsize=(21, 9), sharex=True, sharey=True)
    series = {key: COLORS[key] for key in ("M", "C+M", "C+F+M")}
    _add_figure_legend(fig, series)
    y_min = max(0.0, float(selected["auroc"].min()) - 0.035)
    y_max = min(1.0, float(selected["auroc"].max()) + 0.035)
    for row_index, (population, population_label) in enumerate(POPULATIONS):
        families = ["M", "C+M"]
        if population == "ftv_complete_375":
            families.append("C+F+M")
        for column_index, timing in enumerate(TIMINGS):
            ax = axes[row_index, column_index]
            panel = selected[
                selected["population"].astype(str).eq(population)
                & selected["timing"].astype(str).eq(timing)
            ]
            _plot_dimension_series(
                ax,
                panel,
                series_column="model_family",
                series_labels=families,
                value_column="auroc",
            )
            _style_axis(ax, timing=timing)
            ax.set_ylim(y_min, y_max)
            title_color = "#A14A00" if timing == "T3" else "#202020"
            ax.set_title(TIMING_LABELS[timing], color=title_color, fontweight="bold")
            if column_index == 0:
                ax.set_ylabel(f"{population_label}\nOOF AUROC")
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.11, top=0.84, wspace=0.17)
    filename = "auroc_by_dimensionality.png"
    title = "pCR AUROC by compact MRI dimensionality"
    _finish_figure(fig, figures_dir / filename, title=title)
    return FigureResult(
        filename=filename,
        title=title,
        description=(
            "OOF pCR AUROC across PCA dimensions for MRI-only and feature-"
            "concatenation models; cohorts shown in separate rows."
        ),
        source_files=(source,),
        source_rows_used=len(selected),
        point_unit="seed_x_arm_oof_metric",
    )


def _dimension_delta_frame(metrics: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    candidates = metrics[
        metrics["representation"].astype(str).eq("pca")
        & metrics["model_family"].astype(str).isin(["C+M", "C+F+M"])
        & metrics["dimension"].astype(int).isin(DIMENSIONS)
    ].copy()
    candidates["effect"] = candidates["model_family"].map(
        {"C+M": "C+Mk − C", "C+F+M": "C+F+Mk − C+F"}
    )
    candidates["reference_family"] = candidates["model_family"].map(
        {"C+M": "C", "C+F+M": "C+F"}
    )
    references = metrics[
        metrics["representation"].astype(str).eq("none")
        & metrics["model_family"].astype(str).isin(["C", "C+F"])
    ][["population", "seed", "arm", "timing", "model_family", "auroc"]].copy()
    _assert_unique(
        references,
        ["population", "seed", "arm", "timing", "model_family"],
        "pcr_oof_metrics.csv references",
    )
    keys = ["population", "seed", "arm", "timing"]
    merged = candidates.merge(
        references,
        how="left",
        left_on=keys + ["reference_family"],
        right_on=keys + ["model_family"],
        suffixes=("_candidate", "_reference"),
        validate="many_to_one",
    )
    if merged["auroc_reference"].isna().any():
        raise ValueError("a PCA fusion candidate has no matching aggregate reference")
    merged["delta_auroc"] = (
        merged["auroc_candidate"].astype(float)
        - merged["auroc_reference"].astype(float)
    )
    return merged, len(candidates) + len(references)


def figure_delta_auroc_vs_dimension(
    metrics: pd.DataFrame, figures_dir: Path
) -> FigureResult:
    source = "pcr_oof_metrics.csv"
    required = {
        "population",
        "seed",
        "arm",
        "timing",
        "model_family",
        "representation",
        "dimension",
        "auroc",
    }
    _require_columns(metrics, required, source)
    selected, rows_used = _dimension_delta_frame(metrics)
    _validate_shared_domains(selected, source)
    _require_finite(selected, ["dimension", "delta_auroc"], source)
    _assert_unique(
        selected,
        ["population", "seed", "arm", "timing", "effect", "dimension"],
        source,
    )
    _require_seed_arm_cells(
        selected,
        ["population", "timing", "effect", "dimension"],
        source,
    )

    fig, axes = plt.subplots(2, 4, figsize=(21, 9), sharex=True, sharey=True)
    series = {
        "C+Mk − C": COLORS["C+Mk − C"],
        "C+F+Mk − C+F": COLORS["C+F+Mk − C+F"],
    }
    _add_figure_legend(fig, series)
    limit = max(0.05, float(np.abs(selected["delta_auroc"]).max()) + 0.025)
    for row_index, (population, population_label) in enumerate(POPULATIONS):
        effects = ["C+Mk − C"]
        if population == "ftv_complete_375":
            effects.append("C+F+Mk − C+F")
        for column_index, timing in enumerate(TIMINGS):
            ax = axes[row_index, column_index]
            panel = selected[
                selected["population"].astype(str).eq(population)
                & selected["timing"].astype(str).eq(timing)
            ]
            _plot_dimension_series(
                ax,
                panel,
                series_column="effect",
                series_labels=effects,
                value_column="delta_auroc",
            )
            ax.axhline(0.0, color="#303030", linewidth=1.0, linestyle="--")
            _style_axis(ax, timing=timing)
            ax.set_ylim(-limit, limit)
            title_color = "#A14A00" if timing == "T3" else "#202020"
            ax.set_title(TIMING_LABELS[timing], color=title_color, fontweight="bold")
            if column_index == 0:
                ax.set_ylabel(f"{population_label}\nΔAUROC")
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.11, top=0.84, wspace=0.17)
    filename = "delta_auroc_vs_dimension.png"
    title = "Incremental pCR AUROC versus compact dimension"
    _finish_figure(fig, figures_dir / filename, title=title)
    return FigureResult(
        filename=filename,
        title=title,
        description=(
            "Within-cell AUROC differences for PCA feature fusion versus its "
            "clinical reference; populations are never differenced or pooled."
        ),
        source_files=(source,),
        source_rows_used=rows_used,
        point_unit="within_population_seed_x_arm_aggregate_difference",
    )


def _effect_frame(
    effects: pd.DataFrame,
    comparisons: Mapping[str, str],
) -> pd.DataFrame:
    selected = effects[
        effects["comparison_name"].astype(str).isin(comparisons)
    ].copy()
    selected["effect"] = selected["comparison_name"].map(comparisons)
    _require_finite(selected, ["delta_auroc"], "paired_effects.csv")
    _validate_shared_domains(selected, "paired_effects.csv")
    _assert_unique(
        selected,
        ["population", "seed", "arm", "timing", "effect"],
        "paired_effects.csv",
    )
    _require_seed_arm_cells(
        selected,
        ["population", "timing", "effect"],
        "paired_effects.csv",
    )
    return selected


def figure_beyond_ftv_delta_auroc(
    effects: pd.DataFrame, figures_dir: Path
) -> FigureResult:
    source = "paired_effects.csv"
    _require_columns(
        effects,
        {
            "comparison_name",
            "population",
            "seed",
            "arm",
            "timing",
            "delta_auroc",
        },
        source,
    )
    comparisons = {
        "delta2_CF_plus_Mk_vs_CF": "Compact Mk",
        "residual_beyond_ftv": "FTV-residualized Mk",
    }
    selected = _effect_frame(effects, comparisons)
    if set(selected["population"].astype(str)) != {"ftv_complete_375"}:
        raise ValueError("beyond-FTV effects must be restricted to ftv_complete_375")

    fig, ax = plt.subplots(figsize=(11.5, 6.7))
    series = {label: COLORS[label] for label in comparisons.values()}
    _add_figure_legend(fig, series, y=0.89)
    _plot_timing_series(
        ax,
        selected,
        series_column="effect",
        series_labels=list(comparisons.values()),
        value_column="delta_auroc",
    )
    ax.axhline(0.0, color="#303030", linewidth=1.0, linestyle="--")
    _style_axis(ax)
    limit = max(0.05, float(np.abs(selected["delta_auroc"]).max()) + 0.025)
    ax.set_ylim(-limit, limit)
    ax.set_ylabel("ΔAUROC versus C + FTV")
    ax.set_xlabel("Available MRI timing prefix")
    ax.set_title(
        "FTV-complete cohort only — no cross-population comparison",
        fontsize=11,
        pad=10,
    )
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.15, top=0.78)
    filename = "beyond_ftv_delta_auroc.png"
    title = "Does compact MRI add pCR signal beyond FTV?"
    _finish_figure(fig, figures_dir / filename, title=title)
    return FigureResult(
        filename=filename,
        title=title,
        description=(
            "Within-cell C+F+Mk and FTV-residualized compact-MRI effects versus "
            "C+F, restricted to the FTV-complete cohort."
        ),
        source_files=(source,),
        source_rows_used=len(selected),
        point_unit="within_ftv_population_seed_x_arm_paired_effect",
    )


def figure_raw_vs_compact(effects: pd.DataFrame, figures_dir: Path) -> FigureResult:
    source = "paired_effects.csv"
    comparisons = {
        "raw_vs_compact_M": "MRI only",
        "raw_vs_compact_C_plus_M": "C + MRI",
        "raw_vs_compact_CF_plus_M": "C + FTV + MRI",
    }
    selected = _effect_frame(effects, comparisons)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True, sharey=True)
    series = {label: COLORS[label] for label in comparisons.values()}
    _add_figure_legend(fig, series)
    limit = max(0.05, float(np.abs(selected["delta_auroc"]).max()) + 0.025)
    for row_index, (population, population_label) in enumerate(POPULATIONS):
        labels = ["MRI only", "C + MRI"]
        if population == "ftv_complete_375":
            labels.append("C + FTV + MRI")
        panel = selected[selected["population"].astype(str).eq(population)]
        ax = axes[row_index]
        _plot_timing_series(
            ax,
            panel,
            series_column="effect",
            series_labels=labels,
            value_column="delta_auroc",
        )
        ax.axhline(0.0, color="#303030", linewidth=1.0, linestyle="--")
        _style_axis(ax)
        ax.set_ylim(-limit, limit)
        ax.set_ylabel("Compact − raw ΔAUROC")
        ax.set_title(population_label, loc="left", fontsize=11, fontweight="bold")
    axes[-1].set_xlabel("Available MRI timing prefix")
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.11, top=0.84, hspace=0.32)
    filename = "raw_vs_compact.png"
    title = "Validation-selected compact MRI versus raw MRI prefix"
    _finish_figure(fig, figures_dir / filename, title=title)
    return FigureResult(
        filename=filename,
        title=title,
        description=(
            "Within-cell AUROC effect of selected PCA compact MRI relative to "
            "the same model family using the raw timing prefix."
        ),
        source_files=(source,),
        source_rows_used=len(selected),
        point_unit="within_population_seed_x_arm_paired_effect",
    )


def figure_late_vs_early_fusion(
    effects: pd.DataFrame, figures_dir: Path
) -> FigureResult:
    source = "paired_effects.csv"
    comparisons = {
        "late_vs_concat_C_M": "Late(C,Mk) − concat(C,Mk)",
        "late_vs_concat_CF_M": "Late(C+F,Mk) − concat(C+F,Mk)",
    }
    selected = _effect_frame(effects, comparisons)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True, sharey=True)
    series = {label: COLORS[label] for label in comparisons.values()}
    _add_figure_legend(fig, series)
    limit = max(0.05, float(np.abs(selected["delta_auroc"]).max()) + 0.025)
    for row_index, (population, population_label) in enumerate(POPULATIONS):
        labels = ["Late(C,Mk) − concat(C,Mk)"]
        if population == "ftv_complete_375":
            labels.append("Late(C+F,Mk) − concat(C+F,Mk)")
        panel = selected[selected["population"].astype(str).eq(population)]
        ax = axes[row_index]
        _plot_timing_series(
            ax,
            panel,
            series_column="effect",
            series_labels=labels,
            value_column="delta_auroc",
        )
        ax.axhline(0.0, color="#303030", linewidth=1.0, linestyle="--")
        _style_axis(ax)
        ax.set_ylim(-limit, limit)
        ax.set_ylabel("Late − concatenation ΔAUROC")
        ax.set_title(population_label, loc="left", fontsize=11, fontweight="bold")
    axes[-1].set_xlabel("Available MRI timing prefix")
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.11, top=0.84, hspace=0.32)
    filename = "late_vs_early_fusion.png"
    title = "Strict-OOF late fusion versus feature concatenation"
    _finish_figure(fig, figures_dir / filename, title=title)
    return FigureResult(
        filename=filename,
        title=title,
        description=(
            "Within-cell AUROC difference between strict-inner-OOF late fusion "
            "and compact feature concatenation."
        ),
        source_files=(source,),
        source_rows_used=len(selected),
        point_unit="within_population_seed_x_arm_paired_effect",
    )


def figure_profile_decodability(
    profiles: pd.DataFrame, figures_dir: Path
) -> FigureResult:
    source = "profile_oof_metrics.csv"
    required = {
        "population",
        "seed",
        "arm",
        "timing",
        "representation",
        "target",
        "auroc",
    }
    _require_columns(profiles, required, source)
    selected = profiles[
        profiles["representation"].astype(str).isin(["raw", "pca16", "pca32"])
        & profiles["target"].astype(str).isin(["HR", "HER2", "subtype_4class"])
    ].copy()
    if set(selected["population"].astype(str)) != {"full_808"}:
        raise ValueError("profile decodability must use the frozen full_808 cohort")
    representation_labels = {
        "raw": "Raw prefix",
        "pca16": "PCA-16",
        "pca32": "PCA-32",
    }
    selected["representation_label"] = selected["representation"].map(
        representation_labels
    )
    _require_finite(selected, ["auroc"], source)
    _validate_shared_domains(selected, source)
    _assert_unique(
        selected,
        ["population", "seed", "arm", "timing", "target", "representation"],
        source,
    )
    _require_seed_arm_cells(
        selected,
        ["population", "timing", "target", "representation"],
        source,
    )

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), sharex=True, sharey=True)
    labels = list(representation_labels.values())
    series = {label: COLORS[label] for label in labels}
    _add_figure_legend(fig, series, y=0.88)
    target_titles = {
        "HR": "Hormone receptor (binary)",
        "HER2": "HER2 (binary)",
        "subtype_4class": "Observed HR/HER2 subtype (4-class)",
    }
    y_min = min(0.48, float(selected["auroc"].min()) - 0.02)
    y_max = max(0.62, float(selected["auroc"].max()) + 0.02)
    for index, target in enumerate(("HR", "HER2", "subtype_4class")):
        ax = axes[index]
        panel = selected[selected["target"].astype(str).eq(target)]
        _plot_timing_series(
            ax,
            panel,
            series_column="representation_label",
            series_labels=labels,
            value_column="auroc",
        )
        ax.axhline(0.5, color="#303030", linewidth=1.0, linestyle="--")
        _style_axis(ax)
        ax.set_ylim(y_min, y_max)
        ax.set_title(target_titles[target], fontsize=11, fontweight="bold")
        ax.set_xlabel("Available MRI timing prefix")
        if index == 0:
            ax.set_ylabel("OOF AUROC")
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.16, top=0.76, wspace=0.15)
    filename = "profile_decodability.png"
    title = "Phenotype decodability from raw and compact MRI states"
    footnote = (
        FOOTNOTE
        + " Profile probes use the full cohort only; dashed line is chance AUROC."
    )
    _finish_figure(fig, figures_dir / filename, title=title, footnote=footnote)
    return FigureResult(
        filename=filename,
        title=title,
        description=(
            "Full-cohort OOF AUROC for HR, HER2, and four-class subtype probes "
            "from raw, PCA-16, and PCA-32 MRI representations."
        ),
        source_files=(source,),
        source_rows_used=len(selected),
        point_unit="seed_x_arm_oof_profile_metric",
    )


def figure_pca_explained_variance(
    variance: pd.DataFrame, figures_dir: Path
) -> FigureResult:
    source = "pca_explained_variance.csv"
    required = {
        "population",
        "seed",
        "arm",
        "fold",
        "timing",
        "dimension",
        "cumulative_explained_variance_ratio",
        "validation_rows_in_fit",
        "test_rows_in_fit",
    }
    _require_columns(variance, required, source)
    selected = variance[variance["dimension"].astype(int).isin(DIMENSIONS)].copy()
    _validate_shared_domains(selected, source)
    _require_finite(
        selected,
        [
            "dimension",
            "cumulative_explained_variance_ratio",
            "validation_rows_in_fit",
            "test_rows_in_fit",
        ],
        source,
    )
    if not (
        selected["validation_rows_in_fit"].astype(int).eq(0).all()
        and selected["test_rows_in_fit"].astype(int).eq(0).all()
    ):
        raise ValueError("PCA variance ledger reports non-train rows in a PCA fit")
    _assert_unique(
        selected,
        ["population", "seed", "arm", "fold", "timing", "dimension"],
        source,
    )
    cell_means = (
        selected.groupby(
            ["population", "seed", "arm", "timing", "dimension"], as_index=False
        )
        .agg(
            cumulative_explained_variance_ratio=(
                "cumulative_explained_variance_ratio",
                "mean",
            ),
            n_outer_folds=("fold", "nunique"),
        )
        .sort_values(["population", "timing", "dimension", "seed", "arm"])
    )
    if not cell_means["n_outer_folds"].astype(int).eq(5).all():
        raise ValueError("PCA sensitivity cells must each summarize five outer folds")
    cell_means["series"] = "Cumulative explained variance"
    _require_seed_arm_cells(
        cell_means,
        ["population", "timing", "dimension"],
        source,
    )

    variance_color = "#1F78B4"
    COLORS["Cumulative explained variance"] = variance_color
    fig, axes = plt.subplots(2, 4, figsize=(21, 9), sharex=True, sharey=True)
    _add_figure_legend(
        fig, {"Cumulative explained variance": variance_color}
    )
    y_min = max(
        0.0,
        float(cell_means["cumulative_explained_variance_ratio"].min()) - 0.025,
    )
    for row_index, (population, population_label) in enumerate(POPULATIONS):
        for column_index, timing in enumerate(TIMINGS):
            ax = axes[row_index, column_index]
            panel = cell_means[
                cell_means["population"].astype(str).eq(population)
                & cell_means["timing"].astype(str).eq(timing)
            ]
            _plot_dimension_series(
                ax,
                panel,
                series_column="series",
                series_labels=["Cumulative explained variance"],
                value_column="cumulative_explained_variance_ratio",
            )
            _style_axis(ax, timing=timing)
            ax.set_ylim(y_min, 1.005)
            title_color = "#A14A00" if timing == "T3" else "#202020"
            ax.set_title(TIMING_LABELS[timing], color=title_color, fontweight="bold")
            if column_index == 0:
                ax.set_ylabel(f"{population_label}\nCumulative ratio")
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.11, top=0.84, wspace=0.17)
    filename = "pca_explained_variance.png"
    title = "Outer-train PCA cumulative explained variance"
    footnote = (
        "Small markers: seed×arm means across five outer-train PCA fits; thick "
        "lines: descriptive mean across the four sensitivity cells. No "
        "validation/test rows entered PCA fits. T3 is late/pre-surgery."
    )
    _finish_figure(fig, figures_dir / filename, title=title, footnote=footnote)
    return FigureResult(
        filename=filename,
        title=title,
        description=(
            "Cumulative explained-variance ratio by PCA dimension, with each "
            "seed×arm point averaged only across its five outer-train fits."
        ),
        source_files=(source,),
        source_rows_used=len(selected),
        point_unit="seed_x_arm_mean_of_five_outer_train_pca_fits",
    )


def _write_manifest(
    results: Sequence[FigureResult],
    *,
    metrics_dir: Path,
    figures_dir: Path,
    manifest_path: Path,
) -> None:
    if len(results) != 7 or len({result.filename for result in results}) != 7:
        raise AssertionError("exactly seven uniquely named figures are required")
    rows: list[dict[str, object]] = []
    for order, result in enumerate(results, start=1):
        figure_path = figures_dir / result.filename
        image = plt.imread(figure_path)
        height, width = image.shape[:2]
        source_paths = [metrics_dir / filename for filename in result.source_files]
        rows.append(
            {
                "figure_order": order,
                "figure_file": f"figures/{result.filename}",
                "title": result.title,
                "description": result.description,
                "source_metrics": ";".join(
                    f"metrics/{filename}" for filename in result.source_files
                ),
                "source_metrics_sha256": ";".join(
                    _sha256(path) for path in source_paths
                ),
                "source_rows_used": result.source_rows_used,
                "point_unit": result.point_unit,
                "population_handling": "separate_panels_or_ftv_only_no_pooling",
                "t3_marked_late": True,
                "private_predictions_read": False,
                "width_px": int(width),
                "height_px": int(height),
                "bytes": figure_path.stat().st_size,
                "figure_sha256": _sha256(figure_path),
                "generator": "scripts/generate_figures.py",
            }
        )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(manifest_path, index=False, lineterminator="\n")


def generate_all(
    *,
    metrics_dir: Path = DEFAULT_METRICS_DIR,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> tuple[FigureResult, ...]:
    """Generate all seven figures and their public provenance manifest."""

    metrics_dir = metrics_dir.resolve(strict=True)
    figures_dir = figures_dir.resolve(strict=False)
    manifest_path = manifest_path.resolve(strict=False)
    pcr = _read_aggregate(metrics_dir, "pcr_oof_metrics.csv")
    effects = _read_aggregate(metrics_dir, "paired_effects.csv")
    profiles = _read_aggregate(metrics_dir, "profile_oof_metrics.csv")
    variance = _read_aggregate(metrics_dir, "pca_explained_variance.csv")

    results = (
        figure_auroc_by_dimensionality(pcr, figures_dir),
        figure_delta_auroc_vs_dimension(pcr, figures_dir),
        figure_beyond_ftv_delta_auroc(effects, figures_dir),
        figure_raw_vs_compact(effects, figures_dir),
        figure_late_vs_early_fusion(effects, figures_dir),
        figure_profile_decodability(profiles, figures_dir),
        figure_pca_explained_variance(variance, figures_dir),
    )
    _write_manifest(
        results,
        metrics_dir=metrics_dir,
        figures_dir=figures_dir,
        manifest_path=manifest_path,
    )
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    results = generate_all(
        metrics_dir=args.metrics_dir,
        figures_dir=args.figures_dir,
        manifest_path=args.manifest,
    )
    for result in results:
        print(f"wrote figures/{result.filename}")
    print(f"wrote {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

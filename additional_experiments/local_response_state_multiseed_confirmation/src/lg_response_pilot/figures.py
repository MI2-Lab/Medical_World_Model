"""Render public LOCAL confirmation figures from deidentified tables."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


FIGURE_FILENAMES = (
    "01_gap_local_architecture_schematic.png",
    "02_static_ftv_spearman_comparison.png",
    "03_static_ftv_natural_r2_comparison.png",
    "04_delta_ftv_spearman_comparison.png",
    "05_delta_ftv_natural_r2_comparison.png",
    "06_prediction_target_variance_ratio.png",
    "07_descriptive_calibration_slope.png",
    "08_paired_fold_effects.png",
    "09_optimization_safety_heatmap.png",
    "10_representative_training_curves.png",
)
ARM_ORDER = ("GAP0", "GAP3", "LOCAL0", "LOCAL3")
SEED_ORDER = (2026, 3026, 4026, 5026, 6026)
ARM_COLORS = {
    "GAP0": "#6b7280",
    "GAP3": "#374151",
    "LOCAL0": "#2a9d8f",
    "LOCAL3": "#16796f",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 140,
            "savefig.dpi": 200,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save_atomic(figure: plt.Figure, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite figure: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent
    )
    os.close(descriptor)
    try:
        figure.savefig(temporary, bbox_inches="tight", facecolor="white")
        os.chmod(temporary, 0o644)
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite figure: {path}") from error
    finally:
        plt.close(figure)
        Path(temporary).unlink(missing_ok=True)


def _box(
    axis: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    color: str,
) -> None:
    axis.add_patch(
        FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.025",
            linewidth=1.4,
            edgecolor=color,
            facecolor=f"{color}18",
        )
    )
    axis.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
    )


def _arrow(
    axis: plt.Axes, start: tuple[float, float], end: tuple[float, float]
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=12, color="#334155"
        )
    )


def architecture_schematic(table1: pd.DataFrame) -> plt.Figure:
    if set(table1["arm"]) != set(ARM_ORDER) or len(table1) != len(ARM_ORDER):
        raise ValueError("architecture schematic requires all four confirmation arms")
    figure, axis = plt.subplots(figsize=(11.0, 4.8))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    _box(
        axis, (0.03, 0.40), 0.17, 0.20, "Final encoder map\n128 × D × H × W", "#4361ee"
    )
    _box(axis, (0.28, 0.68), 0.18, 0.16, "Global GAP\n128-D", "#6b7280")
    _box(
        axis,
        (0.28, 0.16),
        0.18,
        0.16,
        "Fixed 64-mm local\nfractional cell overlap\n128-D",
        "#2a9d8f",
    )
    _arrow(axis, (0.20, 0.52), (0.28, 0.76))
    _arrow(axis, (0.20, 0.48), (0.28, 0.24))
    _box(
        axis,
        (0.55, 0.68),
        0.18,
        0.16,
        "GAP0 / GAP3\nLinear 128→192\n+ LayerNorm",
        "#6b7280",
    )
    _box(
        axis,
        (0.55, 0.16),
        0.18,
        0.16,
        "LOCAL0 / LOCAL3\nLinear 128→192\n+ LayerNorm",
        "#2a9d8f",
    )
    _arrow(axis, (0.46, 0.76), (0.55, 0.76))
    _arrow(axis, (0.46, 0.24), (0.55, 0.24))
    _box(
        axis,
        (0.83, 0.40),
        0.14,
        0.20,
        "Response state\nrₜ ∈ ℝ¹⁹²\nJEPA / FTV",
        "#7b2cbf",
    )
    _arrow(axis, (0.73, 0.76), (0.86, 0.60))
    _arrow(axis, (0.73, 0.24), (0.86, 0.40))
    axis.text(
        0.5,
        0.96,
        "Frozen GAP vs fixed 64-mm LOCAL confirmation",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
    )
    axis.text(
        0.5,
        0.04,
        "T0–T3 share one frozen C1B-H coordinate convention; no lesion/FTV/outcome-adaptive pooling",
        ha="center",
        color="#475569",
    )
    return figure


def _macro_arm_plot(
    table: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    *,
    reference: float | None = None,
) -> plt.Figure:
    rows = table.loc[table["endpoint"].eq("macro")].copy()
    expected = {(seed, arm) for seed in SEED_ORDER for arm in ARM_ORDER}
    observed = set(rows[["seed_base", "arm"]].itertuples(index=False, name=None))
    if observed != expected or len(rows) != len(expected):
        raise ValueError("macro arm plot requires five seeds and four arms")
    figure, axis = plt.subplots(figsize=(8.8, 4.8))
    x = np.arange(len(ARM_ORDER))
    for index, arm in enumerate(ARM_ORDER):
        values = rows.loc[rows["arm"].eq(arm), metric].to_numpy(float)
        if len(values) != 5 or not np.isfinite(values).all():
            raise ValueError(f"macro arm plot has invalid {arm}/{metric} values")
        axis.scatter(
            np.full(len(values), index) + np.linspace(-0.14, 0.14, len(values)),
            values,
            s=48,
            color=ARM_COLORS[arm],
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        axis.plot(
            [index - 0.12, index + 0.12], [np.mean(values)] * 2, color="#111827", lw=2
        )
    if reference is not None:
        axis.axhline(reference, color="#94a3b8", linestyle="--", linewidth=1)
    axis.set_xticks(x, ARM_ORDER)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.22)
    axis.text(
        0.99,
        0.02,
        "dots: five independent training seeds; bars: seed mean",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        color="#64748b",
    )
    return figure


def _two_task_macro_plot(
    table2: pd.DataFrame,
    table3: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    *,
    reference: float | None = None,
) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=False)
    for axis, table, task in zip(
        axes, (table2, table3), ("Static FTV", "Observed ΔFTV"), strict=True
    ):
        rows = table.loc[table["endpoint"].eq("macro")]
        expected = {(seed, arm) for seed in SEED_ORDER for arm in ARM_ORDER}
        if set(
            rows[["seed_base", "arm"]].itertuples(index=False, name=None)
        ) != expected or len(rows) != len(expected):
            raise ValueError("two-task macro plot requires five seeds and four arms")
        for index, arm in enumerate(ARM_ORDER):
            values = rows.loc[rows["arm"].eq(arm), metric].to_numpy(float)
            if len(values) != 5 or not np.isfinite(values).all():
                raise ValueError(
                    f"two-task macro plot has invalid {arm}/{metric} values"
                )
            axis.scatter(
                np.full(len(values), index) + np.linspace(-0.14, 0.14, len(values)),
                values,
                s=43,
                color=ARM_COLORS[arm],
                edgecolor="white",
                linewidth=0.6,
            )
            axis.plot(
                [index - 0.11, index + 0.11],
                [np.mean(values)] * 2,
                color="#111827",
                lw=1.8,
            )
        if reference is not None:
            axis.axhline(reference, color="#94a3b8", linestyle="--", linewidth=1)
        axis.set_xticks(range(len(ARM_ORDER)), ARM_ORDER, rotation=30, ha="right")
        axis.set_title(task)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(title, fontweight="bold")
    figure.tight_layout()
    return figure


def paired_fold_plot(fold_effects: pd.DataFrame) -> plt.Figure:
    comparisons = ("LOCAL0-GAP0", "LOCAL3-LOCAL0")
    chosen = fold_effects.loc[fold_effects["comparison"].isin(comparisons)].copy()
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    for axis, task in zip(axes, ("static", "delta"), strict=True):
        rows = chosen.loc[chosen["task"].eq(task)]
        labels = ["LOCAL0−GAP0", "LOCAL3−LOCAL0"]
        for index, comparison in enumerate(comparisons):
            values = rows.loc[
                rows["comparison"].eq(comparison), "effect_spearman"
            ].to_numpy(float)
            if len(values) != 25 or not np.isfinite(values).all():
                raise ValueError(
                    "fold sensitivity requires 25 finite paired folds per comparison"
                )
            jitter = np.linspace(-0.12, 0.12, len(values))
            axis.scatter(
                np.full(len(values), index) + jitter,
                values,
                color="#2a9d8f" if index == 0 else "#16796f",
                alpha=0.85,
                s=35,
            )
            axis.plot(
                [index - 0.18, index + 0.18],
                [np.mean(values)] * 2,
                color="#111827",
                lw=2,
            )
        axis.axhline(0, color="#475569", linestyle="--", linewidth=1)
        axis.set_xticks((0, 1), labels)
        axis.set_title("Static FTV" if task == "static" else "Observed ΔFTV")
        axis.set_ylabel("Fold-level Spearman effect")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(
        "Paired fold sensitivity (descriptive; folds are not independent replicates)",
        fontweight="bold",
    )
    figure.tight_layout()
    return figure


def safety_heatmap(table6: pd.DataFrame) -> plt.Figure:
    comparisons = ("GAP3-GAP0", "LOCAL3-LOCAL0")
    columns = [f"{seed}/F{fold}" for seed in SEED_ORDER for fold in range(5)]
    matrix = np.full((len(comparisons), len(columns)), np.nan, dtype=float)
    for row_index, comparison in enumerate(comparisons):
        rows = table6.loc[table6["comparison"].eq(comparison)]
        for column_index, (seed, fold) in enumerate(
            (seed, fold) for seed in SEED_ORDER for fold in range(5)
        ):
            match = rows.loc[
                rows["seed_base"].eq(seed) & rows["fold"].eq(fold),
                "state_loss_degradation_fraction",
            ]
            if len(match) != 1:
                raise ValueError("safety heatmap lacks an exact paired fold")
            matrix[row_index, column_index] = float(match.iloc[0])
    bound = max(0.05, float(np.nanmax(np.abs(matrix))))
    figure, axis = plt.subplots(figsize=(16.0, 3.6))
    image = axis.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-bound, vmax=bound)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{100 * matrix[row, column]:+.1f}%",
                ha="center",
                va="center",
                fontsize=5.8,
                color="black",
            )
    axis.set_xticks(range(len(columns)), columns, rotation=35, ha="right")
    axis.set_yticks(range(len(comparisons)), comparisons)
    axis.set_title("Selected validation state-loss degradation (pass ≤ +5%)")
    figure.colorbar(image, ax=axis, label="degradation fraction", shrink=0.82)
    figure.tight_layout()
    return figure


def training_curves(histories: pd.DataFrame) -> plt.Figure:
    required = {"arm", "epoch", "val_state_loss", "val_ftv_loss"}
    if missing := sorted(required.difference(histories.columns)):
        raise ValueError(f"training histories miss figure columns: {missing}")
    if set(histories["arm"]) != set(ARM_ORDER):
        raise ValueError("training histories must cover all four confirmation arms")
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for arm in ARM_ORDER:
        rows = histories.loc[histories["arm"].eq(arm)]
        grouped = rows.groupby("epoch", as_index=False).agg(
            val_state_loss=("val_state_loss", "mean"),
            val_ftv_loss=("val_ftv_loss", "mean"),
        )
        axes[0].plot(
            grouped["epoch"],
            grouped["val_state_loss"],
            label=arm,
            color=ARM_COLORS[arm],
            linewidth=1.8,
        )
        if arm.endswith("3"):
            axes[1].plot(
                grouped["epoch"],
                grouped["val_ftv_loss"],
                label=arm,
                color=ARM_COLORS[arm],
                linewidth=1.8,
            )
    axes[0].set_title("Validation state loss")
    axes[1].set_title("Validation FTV loss (grounded arms)")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Mean across seed/fold cells")
        axis.grid(alpha=0.2)
        axis.legend(ncol=2)
    figure.suptitle("Representative formal training trajectories", fontweight="bold")
    figure.tight_layout()
    return figure


def render_required_figures(
    *,
    table1: pd.DataFrame,
    table2: pd.DataFrame,
    table3: pd.DataFrame,
    table4: pd.DataFrame,
    table6: pd.DataFrame,
    fold_effects: pd.DataFrame,
    histories: pd.DataFrame,
    output_dir: str | Path,
) -> tuple[Path, ...]:
    """Render exactly the ten required figures; inputs contain no patient IDs."""

    del table4  # Effects are rendered from their fold-sensitivity companion.
    _style()
    output = Path(output_dir).resolve()
    builders: tuple[Callable[[], plt.Figure], ...] = (
        lambda: architecture_schematic(table1),
        lambda: _macro_arm_plot(
            table2,
            "natural_spearman",
            "Static FTV macro Spearman",
            "Spearman",
        ),
        lambda: _macro_arm_plot(
            table2,
            "natural_r2",
            "Static FTV macro natural-scale R²",
            "Natural R²",
            reference=0.0,
        ),
        lambda: _macro_arm_plot(
            table3,
            "natural_spearman",
            "Observed ΔFTV macro Spearman",
            "Spearman",
        ),
        lambda: _macro_arm_plot(
            table3,
            "natural_r2",
            "Observed ΔFTV macro natural-scale R²",
            "Natural R²",
            reference=0.0,
        ),
        lambda: _two_task_macro_plot(
            table2,
            table3,
            "natural_prediction_target_variance_ratio",
            "Prediction/target variance ratio",
            "Variance ratio",
            reference=1.0,
        ),
        lambda: _two_task_macro_plot(
            table2,
            table3,
            "natural_calibration_slope",
            "Descriptive calibration slope: Cov(true,pred) / Var(true)",
            "Calibration slope",
            reference=1.0,
        ),
        lambda: paired_fold_plot(fold_effects),
        lambda: safety_heatmap(table6),
        lambda: training_curves(histories),
    )
    paths: list[Path] = []
    for filename, builder in zip(FIGURE_FILENAMES, builders, strict=True):
        path = output / filename
        _save_atomic(builder(), path)
        paths.append(path)
    return tuple(paths)


__all__ = [
    "ARM_ORDER",
    "FIGURE_FILENAMES",
    "architecture_schematic",
    "paired_fold_plot",
    "render_required_figures",
    "safety_heatmap",
    "training_curves",
]

#!/usr/bin/env python3
"""Create privacy-preserving figures from the audit's public aggregates only.

The plotting code deliberately does not know how to open feature matrices, OOF
predictions, labels, masks, or checkpoints.  Its input boundary is the ten
public aggregate CSV files emitted by ``run_audit.py``.  Small schema aliases
are accepted so that presentation is insulated from spelling-only changes;
ambiguous or patient-level inputs fail closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

PUBLIC_TABLES: dict[str, str] = {
    "occupancy": "region_occupancy.csv",
    "mri_only_pcr": "table_mri_only_pcr.csv",
    "clinical_pcr": "table_clinical_pcr.csv",
    "clinical_ftv_incremental": "table_clinical_ftv_incremental.csv",
    "phenotype": "table_phenotype.csv",
    "ftv": "table_ftv.csv",
    "oracle_recovery": "table_oracle_recovery.csv",
    "bootstrap": "table_bootstrap.csv",
    "seed_consistency": "table_seed_consistency.csv",
    "timing_sensitivity": "table_timing_sensitivity.csv",
}

FIGURES: tuple[str, ...] = (
    "01_region_schematic.png",
    "02_region_occupancy.png",
    "03_mri_only_pcr.png",
    "04_phenotype_probes.png",
    "05_clinical_ftv_incremental.png",
    "06_ftv_response_control.png",
    "07_oracle_recovery.png",
    "08_patient_bootstrap.png",
    "09_seed_consistency.png",
    "10_timing_sensitivity.png",
)

FIGURE_SOURCES: dict[str, tuple[str, ...]] = {
    FIGURES[0]: ("configs/audit.json",),
    FIGURES[1]: (PUBLIC_TABLES["occupancy"],),
    FIGURES[2]: (PUBLIC_TABLES["mri_only_pcr"],),
    FIGURES[3]: (PUBLIC_TABLES["phenotype"],),
    FIGURES[4]: (PUBLIC_TABLES["clinical_ftv_incremental"],),
    FIGURES[5]: (PUBLIC_TABLES["ftv"],),
    FIGURES[6]: (PUBLIC_TABLES["oracle_recovery"],),
    FIGURES[7]: (PUBLIC_TABLES["bootstrap"],),
    FIGURES[8]: (PUBLIC_TABLES["seed_consistency"],),
    FIGURES[9]: (PUBLIC_TABLES["timing_sensitivity"],),
}

FIGURE_MANIFEST_COLUMNS: tuple[str, ...] = (
    "figure",
    "source_files",
    "source_sha256",
    "size_bytes",
    "sha256",
)

PRIMARY_REGIONS = ("R0", "R1", "R2", "R3", "R4", "R5")
REGION_COLORS = {
    "R0": "#777777",
    "R1": "#4c78a8",
    "R2": "#f58518",
    "R3": "#e45756",
    "R4": "#72b7b2",
    "R5": "#54a24b",
    "FIXED_P3": "#777777",
    "PERI20": "#b279a2",
}

IDENTIFIER_COLUMN_RE = re.compile(
    r"^(?:patient|subject|participant|clinical_patient|raw_patient|mrn)(?:_?id)?$",
    flags=re.IGNORECASE,
)
PATIENT_TOKEN_RE = re.compile(r"\bACRIN[-_ ]?\d+\b", flags=re.IGNORECASE)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Audit root (primarily useful for isolated verification fixtures).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the ten declared figures and their manifest.",
    )
    return parser.parse_args(argv)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False, lineterminator="\n")
        temporary.chmod(0o644)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _normal_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def resolve_column(
    frame: pd.DataFrame,
    aliases: Iterable[str],
    *,
    required: bool = True,
) -> str | None:
    """Resolve a spelling-only schema alias, rejecting ambiguous columns."""

    normalized: dict[str, list[str]] = {}
    columns = tuple(frame.columns.astype(str))
    for column in columns:
        normalized.setdefault(_normal_name(column), []).append(column)
    for alias in aliases:
        if str(alias) in columns:
            return str(alias)
        matches = normalized.get(_normal_name(alias), [])
        if not str(alias).startswith("_"):
            matches = [match for match in matches if not str(match).startswith("_")]
        if len(matches) > 1:
            raise ValueError(f"ambiguous columns for {alias!r}: {matches}")
        if matches:
            return matches[0]
    if required:
        raise ValueError(
            f"none of the required columns {tuple(aliases)!r} is present; "
            f"observed={tuple(frame.columns.astype(str))!r}"
        )
    return None


def numeric_column(
    frame: pd.DataFrame,
    aliases: Iterable[str],
    *,
    required: bool = True,
) -> tuple[str | None, pd.Series | None]:
    column = resolve_column(frame, aliases, required=required)
    if column is None:
        return None, None
    values = pd.to_numeric(frame[column], errors="coerce")
    if required and (values.isna().all() or not np.isfinite(values.dropna()).all()):
        raise ValueError(f"column {column!r} has no finite numeric values")
    return column, values


def _reject_patient_level_table(frame: pd.DataFrame, path: Path) -> None:
    bad_columns = [
        str(column)
        for column in frame.columns
        if IDENTIFIER_COLUMN_RE.fullmatch(
            re.sub(r"[^A-Za-z0-9_]+", "_", str(column).strip())
        )
    ]
    if bad_columns:
        raise ValueError(
            f"public aggregate table exposes identifier columns {bad_columns}: {path}"
        )
    # Aggregate files are small; scanning text also catches identifiers hidden
    # in a generic column such as ``cohort``.
    if PATIENT_TOKEN_RE.search(path.read_text(encoding="utf-8", errors="strict")):
        raise ValueError(f"public aggregate table contains a patient-like token: {path}")


def read_public_table(root: Path, filename: str) -> pd.DataFrame:
    path = root / "metrics" / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    before = file_sha256(path)
    frame = pd.read_csv(path, float_precision="round_trip")
    after = file_sha256(path)
    if before != after:
        raise RuntimeError(f"aggregate changed while being read: {path}")
    if frame.empty:
        raise ValueError(f"public aggregate table is empty: {path}")
    _reject_patient_level_table(frame, path)
    frame.attrs["source_sha256"] = before
    frame.attrs["source_filename"] = filename
    return frame


def load_public_tables(root: Path = ROOT) -> dict[str, pd.DataFrame]:
    """Load the complete registered public table inventory."""

    return {
        logical_name: read_public_table(root, filename)
        for logical_name, filename in PUBLIC_TABLES.items()
    }


def _region_series(frame: pd.DataFrame) -> pd.Series:
    column = resolve_column(
        frame,
        ("variant", "region", "feature_variant", "candidate", "comparison"),
        required=False,
    )
    if column is None:
        model = resolve_column(frame, ("model", "model_name"), required=False)
        if model is None:
            raise ValueError("table has no region/variant/comparison/model column")
        values = frame[model].astype(str)
    else:
        values = frame[column].astype(str)
    extracted = values.str.upper().str.extract(r"\b(R[0-5])\b", expand=False)
    return extracted.fillna(values.str.upper())


def _timing_series(frame: pd.DataFrame) -> pd.Series:
    column = resolve_column(
        frame, ("timing", "view", "visit", "prefix", "timepoint"), required=False
    )
    if column is None:
        return pd.Series(["all"] * len(frame), index=frame.index, dtype=object)
    values = frame[column].astype(str).str.upper()
    values = values.str.replace("→", "-", regex=False).str.replace("_", "-", regex=False)
    values = values.str.replace(r"^T([0-3])-TO-T([0-3])$", r"T\1-T\2", regex=True)
    values = values.str.replace(r"^T([0-3])-?>T([0-3])$", r"T\1-T\2", regex=True)
    return values


def _target_series(frame: pd.DataFrame) -> pd.Series:
    column = resolve_column(frame, ("target", "endpoint", "outcome"), required=False)
    if column is None:
        return pd.Series(["outcome"] * len(frame), index=frame.index, dtype=object)
    return frame[column].astype(str)


def _filter_primary(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    geometry = resolve_column(
        result, ("geometry", "geometry_set", "region_set"), required=False
    )
    if geometry is not None:
        primary = result[geometry].astype(str).str.lower().isin(
            {"primary", "32/48/64", "32-48-64", "r"}
        )
        if primary.any():
            result = result.loc[primary].copy()
    result["_region"] = _region_series(result)
    selected = result["_region"].isin(PRIMARY_REGIONS)
    return result.loc[selected].copy() if selected.any() else result


def _metric_series(
    frame: pd.DataFrame,
    aliases: Sequence[str] = ("auroc", "test_auroc", "oof_auroc"),
) -> tuple[str, pd.Series]:
    column, values = numeric_column(frame, aliases)
    assert column is not None and values is not None
    if "auroc" in _normal_name(column) and (
        (values.dropna() < 0).any() or (values.dropna() > 1).any()
    ):
        raise ValueError(f"{column} lies outside [0,1]")
    return column, values


def region_delta_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return canonical region-vs-R0 AUROC deltas from long or delta schemas."""

    work = _filter_primary(frame)
    work["_timing"] = _timing_series(work)
    work["_target"] = _target_series(work)
    direct = resolve_column(
        work,
        (
            "delta_auroc_vs_r0",
            "auroc_delta_vs_r0",
            "delta_vs_r0",
            "mean_delta_auroc_vs_r0",
            "delta_auroc_vs_c_plus_f_plus_r0",
            "delta_auroc_vs_C+F+R0",
            "delta_auroc_vs_cf_r0",
            "incremental_auroc",
            "delta_auroc",
        ),
        required=False,
    )
    if direct is not None:
        values = pd.to_numeric(work[direct], errors="coerce")
        if not np.isfinite(values.dropna()).all() or values.notna().sum() == 0:
            raise ValueError(f"delta column {direct!r} has no finite values")
        result = work.copy()
        result["_delta"] = values
        return result.loc[result["_delta"].notna()].copy()

    metric, auroc = _metric_series(work)
    work["_metric"] = auroc
    possible_keys = (
        "seed",
        "seed_base",
        "arm",
        "checkpoint",
        "timing",
        "view",
        "visit",
        "target",
        "endpoint",
        "population",
        "cohort",
        "context",
        "probe_context",
        "clinical_contract",
    )
    keys: list[str] = []
    normalized_seen: set[str] = set()
    for alias in possible_keys:
        column = resolve_column(work, (alias,), required=False)
        if column is not None and _normal_name(column) not in normalized_seen:
            keys.append(column)
            normalized_seen.add(_normal_name(column))
    # Canonical display dimensions are included if the source used no direct
    # spelling for them.
    for canonical in ("_timing", "_target"):
        if canonical not in keys:
            keys.append(canonical)
    references = work.loc[work["_region"].eq("R0"), [*keys, "_metric"]].copy()
    if references.empty:
        raise ValueError("cannot compute region delta: R0 reference is absent")
    if references.duplicated(keys).any():
        # Multiple rows can differ by a dimension with an unfamiliar spelling;
        # ambiguity must not be silently averaged into a comparison.
        raise ValueError("R0 reference is ambiguous within inferred matched cells")
    references = references.rename(columns={"_metric": "_r0_metric"})
    compared = work.merge(references, on=keys, how="left", validate="many_to_one")
    if compared["_r0_metric"].isna().any():
        raise ValueError("one or more region rows lacks an exactly matched R0 reference")
    compared["_delta"] = compared["_metric"] - compared["_r0_metric"]
    compared.attrs["derived_from"] = metric
    return compared


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def region_schematic(config: Mapping[str, Any]) -> plt.Figure:
    contract = config.get("feature_contract", {})
    boundaries = contract.get("primary_boundaries_mm", [32.0, 48.0, 64.0])
    if not isinstance(boundaries, list) or len(boundaries) != 3:
        raise ValueError("config must define three primary physical boundaries")
    inner, middle, outer = [float(value) for value in boundaries]
    if not 0 < inner < middle < outer:
        raise ValueError("primary boundaries must be strictly increasing")

    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))
    axis = axes[0]
    axis.set_aspect("equal")
    half = outer / 2.0
    for size, color, label, alpha in (
        (outer, REGION_COLORS["R3"], f"R3 outer shell ({middle:g}-{outer:g} mm)", 0.32),
        (middle, REGION_COLORS["R2"], f"R2 inner shell ({inner:g}-{middle:g} mm)", 0.45),
        (inner, REGION_COLORS["R1"], f"R1 central cube ({inner:g} mm)", 0.65),
    ):
        axis.add_patch(
            Rectangle(
                (-size / 2, -size / 2),
                size,
                size,
                facecolor=color,
                edgecolor="white",
                linewidth=1.5,
                alpha=alpha,
                label=label,
            )
        )
    axis.scatter([0], [0], marker="+", s=90, color="#222222", zorder=5)
    axis.set_xlim(-half - 4, half + 4)
    axis.set_ylim(-half - 4, half + 4)
    axis.set_xlabel("physical X (mm)")
    axis.set_ylabel("physical Y (mm)")
    axis.set_title("Fixed physical partition (central slice)", weight="bold")
    axis.legend(loc="upper left", bbox_to_anchor=(0.0, -0.15), frameon=False)

    axes[1].axis("off")
    lines = (
        "R0 = full 64-mm LOCAL weighted mean",
        "R1 = central 32-mm cube",
        "R2 = 48-mm cube minus R1",
        "R3 = 64-mm cube minus 48-mm cube",
        "R4 = concat(R1, R2)",
        "R5 = concat(R1, R2, R3)",
        "",
        "Fractional feature-cell volume overlap",
        "No mask / bbox / label read by this readout",
        "Crop center is frozen upstream T0 localization",
    )
    axes[1].text(
        0.02,
        0.95,
        "\n".join(lines),
        va="top",
        ha="left",
        linespacing=1.55,
        family="DejaVu Sans",
    )
    axes[1].set_title("Representation contract", weight="bold")
    figure.suptitle("Mask-free-at-readout-only region schematic", weight="bold")
    return figure


def occupancy_figure(frame: pd.DataFrame) -> plt.Figure:
    work = _filter_primary(frame)
    candidates = (
        "mean_effective_cells",
        "effective_cell_count",
        "effective_cells",
        "mean_weight_sum",
        "weight_sum",
        "occupancy_fraction",
        "mean_occupancy",
        "physical_volume_mm3",
        "expected_volume_mm3",
        "volume_mm3",
    )
    metric, values = numeric_column(work, candidates, required=False)
    if metric is None or values is None:
        excluded = {
            _normal_name(value)
            for value in ("seed", "fold", "n", "channels", "dimension", "size_mm")
        }
        for column in work.columns:
            numeric = pd.to_numeric(work[column], errors="coerce")
            if (
                _normal_name(column) not in excluded
                and numeric.notna().all()
                and np.isfinite(numeric).all()
            ):
                metric, values = str(column), numeric
                break
    if metric is None or values is None:
        raise ValueError("occupancy table has no plottable aggregate measure")
    work["_value"] = values
    summary = work.groupby("_region", sort=False)["_value"].agg(["mean", "min", "max"])
    order = [value for value in PRIMARY_REGIONS if value in summary.index]
    if not order:
        raise ValueError("occupancy table has no registered primary regions")
    summary = summary.loc[order]
    figure, axis = plt.subplots(figsize=(7.3, 4.2))
    x = np.arange(len(summary))
    low = np.maximum(0.0, summary["mean"].to_numpy() - summary["min"].to_numpy())
    high = np.maximum(0.0, summary["max"].to_numpy() - summary["mean"].to_numpy())
    axis.bar(
        x,
        summary["mean"],
        color=[REGION_COLORS.get(value, "#999999") for value in summary.index],
        yerr=np.vstack([low, high]),
        capsize=3,
    )
    axis.set_xticks(x, summary.index)
    axis.set_ylabel(metric.replace("_", " "))
    axis.set_title("Fractional feature-cell occupancy", weight="bold")
    axis.grid(axis="y", alpha=0.2)
    return figure


def _line_by_region(
    frame: pd.DataFrame,
    *,
    title: str,
    ylabel: str,
    delta: bool = False,
) -> plt.Figure:
    work = region_delta_frame(frame) if delta else _filter_primary(frame)
    if "_timing" not in work:
        work["_timing"] = _timing_series(work)
    if delta:
        work["_value"] = pd.to_numeric(work["_delta"], errors="coerce")
    else:
        _, values = _metric_series(work)
        work["_value"] = values
    summary = work.groupby(["_timing", "_region"], sort=False)["_value"].mean()
    timings = [
        value
        for value in ("T0", "T0-T1", "T0-T2", "T0-T3", "T1", "T2", "T3", "all")
        if value in summary.index.get_level_values(0)
    ]
    timings.extend(
        sorted(set(summary.index.get_level_values(0)) - set(timings), key=str)
    )
    regions = [value for value in PRIMARY_REGIONS if value in summary.index.get_level_values(1)]
    if not timings or not regions:
        raise ValueError(f"{title}: no plottable timing/region cells")
    figure, axis = plt.subplots(figsize=(7.8, 4.4))
    x = np.arange(len(timings))
    for region in regions:
        points = [summary.get((timing, region), np.nan) for timing in timings]
        axis.plot(
            x,
            points,
            marker="o",
            linewidth=1.5,
            label=region,
            color=REGION_COLORS.get(region, "#999999"),
        )
    if delta:
        axis.axhline(0.0, color="#333333", linestyle="--", linewidth=0.9)
    else:
        axis.axhline(0.5, color="#999999", linestyle="--", linewidth=0.8)
    axis.set_xticks(x, ["T0-T3\nlate/pre-surgery" if t == "T0-T3" else t for t in timings])
    axis.set_ylabel(ylabel)
    axis.set_title(title, weight="bold")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncol=3)
    return figure


def mri_only_figure(frame: pd.DataFrame) -> plt.Figure:
    work = frame.copy()
    target = resolve_column(work, ("target", "endpoint", "outcome"), required=False)
    if target is not None:
        selected = work[target].astype(str).str.lower().eq("pcr")
        if selected.any():
            work = work.loc[selected].copy()
    population = resolve_column(work, ("population", "cohort"), required=False)
    if population is not None:
        full = work[population].astype(str).str.lower().isin(
            {"full_808", "full", "eligible_808"}
        )
        if full.any():
            work = work.loc[full].copy()
    return _line_by_region(
        work,
        title="MRI-only pCR OOF performance",
        ylabel="AUROC (mean over arm/seed)",
    )


def phenotype_figure(frame: pd.DataFrame) -> plt.Figure:
    work = _filter_primary(frame)
    work["_target"] = _target_series(work)
    _, values = _metric_series(work)
    work["_value"] = values
    pivot = work.pivot_table(
        index="_target", columns="_region", values="_value", aggfunc="mean"
    )
    columns = [value for value in PRIMARY_REGIONS if value in pivot.columns]
    pivot = pivot.reindex(columns=columns)
    if pivot.empty or not columns:
        raise ValueError("phenotype table has no registered target/region cells")
    figure, axis = plt.subplots(figsize=(7.5, max(3.3, 0.65 * len(pivot))))
    image = axis.imshow(pivot.to_numpy(float), vmin=0.45, vmax=0.85, cmap="viridis")
    axis.set_xticks(np.arange(len(columns)), columns)
    axis.set_yticks(np.arange(len(pivot.index)), pivot.index)
    axis.set_xlabel("mask-free region")
    axis.set_title("Phenotype probe AUROC", weight="bold")
    for row in range(len(pivot.index)):
        for column in range(len(columns)):
            value = pivot.iloc[row, column]
            if np.isfinite(value):
                axis.text(column, row, f"{value:.3f}", ha="center", va="center", color="white")
    figure.colorbar(image, ax=axis, fraction=0.035, pad=0.03)
    return figure


def incremental_figure(frame: pd.DataFrame) -> plt.Figure:
    return _line_by_region(
        frame,
        title="Increment beyond Clinical + FTV + R0",
        ylabel="AUROC delta vs C+F+R0",
        delta=True,
    )


def ftv_figure(frame: pd.DataFrame) -> plt.Figure:
    work = _filter_primary(frame)
    scope = resolve_column(work, ("analysis_scope", "scope"), required=False)
    if scope is not None:
        primary = work[scope].astype(str).eq("primary_measurement_valid")
        if primary.any():
            work = work.loc[primary].copy()
    work["_timing"] = _timing_series(work)
    work["_target"] = _target_series(work)
    aliases = (
        "r2",
        "test_r2",
        "spearman",
        "test_spearman",
        "pearson",
        "negative_rmse",
        "rmse",
        "mae",
    )
    metric, values = numeric_column(work, aliases)
    assert metric is not None and values is not None
    work["_value"] = values
    work["_panel"] = work["_target"].astype(str) + " · " + work["_timing"].astype(str)
    summary = work.groupby(["_panel", "_region"], sort=False)["_value"].mean().unstack()
    columns = [value for value in PRIMARY_REGIONS if value in summary.columns]
    summary = summary.reindex(columns=columns)
    if summary.empty or not columns:
        raise ValueError("FTV table has no registered endpoint/region cells")
    figure, axis = plt.subplots(figsize=(8.2, max(3.8, 0.36 * len(summary))))
    y = np.arange(len(summary.index))
    width = 0.82 / len(columns)
    for offset, region in enumerate(columns):
        axis.barh(
            y - 0.41 + width / 2 + offset * width,
            summary[region],
            height=width,
            label=region,
            color=REGION_COLORS.get(region, "#999999"),
        )
    axis.set_yticks(y, summary.index)
    axis.set_xlabel(metric.replace("_", " "))
    axis.set_title("FTV / delta-FTV response controls", weight="bold")
    axis.grid(axis="x", alpha=0.2)
    axis.legend(frameon=False, ncol=3)
    return figure


def oracle_figure(frame: pd.DataFrame) -> plt.Figure:
    work = frame.copy()
    work["_region"] = _region_series(work)
    ratio_col, ratio = numeric_column(
        work,
        ("recovery_ratio", "oracle_recovery_ratio", "mean_recovery_ratio"),
        required=False,
    )
    if ratio_col is None or ratio is None:
        _, numerator = numeric_column(
            work, ("numerator", "mask_free_uplift", "delta_auroc_vs_r0")
        )
        _, denominator = numeric_column(
            work, ("denominator", "oracle_uplift", "peri20_uplift")
        )
        assert numerator is not None and denominator is not None
        if (denominator.dropna() <= 0).any():
            raise ValueError("Oracle recovery ratio is defined only for positive denominators")
        ratio = numerator / denominator
        ratio_col = "derived recovery ratio"
    work["_ratio"] = ratio
    work = work.loc[np.isfinite(work["_ratio"])].copy()
    summary = work.groupby("_region", sort=False)["_ratio"].agg(["mean", "min", "max"])
    order = [value for value in ("R1", "R2", "R3", "R5") if value in summary.index]
    if not order:
        raise ValueError("Oracle recovery table lacks registered mask-free candidates")
    summary = summary.loc[order]
    figure, axis = plt.subplots(figsize=(6.8, 4.2))
    x = np.arange(len(summary))
    low = summary["mean"].to_numpy() - summary["min"].to_numpy()
    high = summary["max"].to_numpy() - summary["mean"].to_numpy()
    axis.bar(
        x,
        summary["mean"],
        yerr=np.vstack([low, high]),
        capsize=4,
        color=[REGION_COLORS.get(value, "#999999") for value in order],
    )
    axis.axhline(0.30, color="#222222", linestyle="--", linewidth=1.0, label="Gate C: 30%")
    axis.axhline(0.0, color="#999999", linewidth=0.8)
    axis.set_xticks(x, order)
    axis.set_ylabel("recovery ratio")
    axis.set_title("PERI20 Oracle uplift recovery (matched T0-T1)", weight="bold")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    return figure


def bootstrap_figure(frame: pd.DataFrame) -> plt.Figure:
    work = frame.copy()
    metric_column = resolve_column(work, ("metric", "measure"), required=False)
    if metric_column is not None:
        auroc_rows = work[metric_column].astype(str).str.lower().eq("auroc")
        if auroc_rows.any():
            work = work.loc[auroc_rows].copy()
    work["_region"] = _region_series(work)
    work["_timing"] = _timing_series(work)
    estimate_col, estimate = numeric_column(
        work,
        ("estimate", "observed_delta", "delta_auroc", "mean_delta", "point_estimate"),
    )
    _, lower = numeric_column(
        work, ("ci_lower", "lower", "lower_95", "ci95_lower", "percentile_lower")
    )
    _, upper = numeric_column(
        work, ("ci_upper", "upper", "upper_95", "ci95_upper", "percentile_upper")
    )
    assert estimate_col is not None and estimate is not None and lower is not None and upper is not None
    if ((lower > estimate) | (estimate > upper)).any():
        raise ValueError("bootstrap interval does not contain its point estimate")
    context = resolve_column(work, ("context", "probe_context", "comparison_type"), required=False)
    work["_context"] = work[context].astype(str) if context else "registered"
    work["_estimate"] = estimate
    work["_lower"] = lower
    work["_upper"] = upper
    work["_label"] = (
        work["_context"].astype(str)
        + " · "
        + work["_timing"].astype(str)
        + " · "
        + work["_region"].astype(str)
    )
    # Preserve every preregistered comparison up to a readable bound.  Formal
    # tables remain the authoritative full inventory.
    work = work.sort_values(["_context", "_timing", "_region"]).head(36)
    y = np.arange(len(work))
    figure, axis = plt.subplots(figsize=(8.4, max(4.0, 0.27 * len(work) + 1.2)))
    axis.errorbar(
        work["_estimate"],
        y,
        xerr=np.vstack(
            [work["_estimate"] - work["_lower"], work["_upper"] - work["_estimate"]]
        ),
        fmt="o",
        markersize=4,
        color="#4c78a8",
        ecolor="#777777",
        capsize=2,
    )
    axis.axvline(0.0, color="#333333", linestyle="--", linewidth=0.9)
    axis.set_yticks(y, work["_label"])
    axis.set_xlabel(estimate_col.replace("_", " ") + " (95% percentile CI)")
    axis.set_title("Paired patient-level bootstrap within outer fold", weight="bold")
    axis.grid(axis="x", alpha=0.2)
    return figure


def seed_consistency_figure(frame: pd.DataFrame) -> plt.Figure:
    work = frame.copy()
    work["_region"] = _region_series(work)
    work["_timing"] = _timing_series(work)
    x_col, x = numeric_column(
        work,
        ("seed_2026_delta_auroc", "delta_seed_2026", "gain_seed_2026", "seed1_gain"),
        required=False,
    )
    y_col, y = numeric_column(
        work,
        ("seed_3026_delta_auroc", "delta_seed_3026", "gain_seed_3026", "seed2_gain"),
        required=False,
    )
    if x_col is None or y_col is None or x is None or y is None:
        seed = resolve_column(work, ("seed", "seed_base"))
        delta_col, delta = numeric_column(
            work,
            ("delta_auroc", "gain", "delta_vs_r0", "delta_auroc_vs_r0"),
        )
        assert delta_col is not None and delta is not None
        work["_delta"] = delta
        other_keys = ["_region", "_timing"]
        for aliases in (("arm",), ("context", "probe_context"), ("target", "endpoint")):
            column = resolve_column(work, aliases, required=False)
            if column is not None:
                other_keys.append(column)
        pivot = work.pivot_table(index=other_keys, columns=seed, values="_delta", aggfunc="first")
        seed_columns = {_normal_name(value): value for value in pivot.columns}
        if "2026" not in seed_columns or "3026" not in seed_columns:
            raise ValueError("seed-consistency table must contain seeds 2026 and 3026")
        plot = pivot.reset_index()
        x = pd.to_numeric(plot[seed_columns["2026"]], errors="coerce")
        y = pd.to_numeric(plot[seed_columns["3026"]], errors="coerce")
        labels = plot["_timing"].astype(str) + " · " + plot["_region"].astype(str)
        x_col, y_col = "seed 2026 delta AUROC", "seed 3026 delta AUROC"
    else:
        plot = work
        labels = plot["_timing"].astype(str) + " · " + plot["_region"].astype(str)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() == 0:
        raise ValueError("seed-consistency table has no paired finite values")
    x = x.loc[valid]
    y = y.loc[valid]
    labels = labels.loc[valid]
    figure, axis = plt.subplots(figsize=(6.2, 5.4))
    axis.scatter(x, y, c=[REGION_COLORS.get(str(region), "#777777") for region in plot.loc[valid, "_region"]], s=34, alpha=0.85)
    extent = max(0.03, float(np.nanmax(np.abs(np.concatenate([x.to_numpy(), y.to_numpy()])))) * 1.15)
    axis.plot([-extent, extent], [-extent, extent], color="#999999", linestyle=":", linewidth=0.8)
    axis.axhline(0.0, color="#333333", linestyle="--", linewidth=0.8)
    axis.axvline(0.0, color="#333333", linestyle="--", linewidth=0.8)
    for xv, yv, label in zip(x, y, labels, strict=True):
        axis.annotate(str(label), (xv, yv), xytext=(3, 3), textcoords="offset points", fontsize=6)
    axis.set_xlim(-extent, extent)
    axis.set_ylim(-extent, extent)
    axis.set_xlabel(x_col.replace("_", " "))
    axis.set_ylabel(y_col.replace("_", " "))
    axis.set_title("Seed consistency of registered gains", weight="bold")
    axis.grid(alpha=0.15)
    return figure


def timing_figure(frame: pd.DataFrame) -> plt.Figure:
    return _line_by_region(
        frame,
        title="Timing sensitivity (T3 is late/pre-surgery)",
        ylabel="AUROC delta vs R0",
        delta=True,
    )


def build_figures(
    tables: Mapping[str, pd.DataFrame], config: Mapping[str, Any]
) -> tuple[plt.Figure, ...]:
    missing = set(PUBLIC_TABLES) - set(tables)
    if missing:
        raise ValueError(f"missing public tables for figures: {sorted(missing)}")
    return (
        region_schematic(config),
        occupancy_figure(tables["occupancy"]),
        mri_only_figure(tables["mri_only_pcr"]),
        phenotype_figure(tables["phenotype"]),
        incremental_figure(tables["clinical_ftv_incremental"]),
        ftv_figure(tables["ftv"]),
        oracle_figure(tables["oracle_recovery"]),
        bootstrap_figure(tables["bootstrap"]),
        seed_consistency_figure(tables["seed_consistency"]),
        timing_figure(tables["timing_sensitivity"]),
    )


def _save_figure(figure: plt.Figure, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".png", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        figure.savefig(
            temporary,
            format="png",
            bbox_inches="tight",
            facecolor="white",
            metadata={"Software": "mask_free_region_aware_audit"},
        )
        temporary.chmod(0o644)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(figure)


def generate_all(root: Path = ROOT, *, overwrite: bool = False) -> pd.DataFrame:
    root = root.resolve()
    config_path = root / "configs" / "audit.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("audit config must be a JSON object")
    tables = load_public_tables(root)
    destinations = [root / "figures" / name for name in FIGURES]
    manifest_path = root / "metrics" / "figure_manifest.csv"
    existing = [path for path in [*destinations, manifest_path] if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "figure outputs already exist; pass --overwrite to replace the declared set"
        )

    _style()
    figures = build_figures(tables, config)
    rows: list[dict[str, object]] = []
    for filename, figure, destination in zip(FIGURES, figures, destinations, strict=True):
        _save_figure(figure, destination)
        logical_sources = FIGURE_SOURCES[filename]
        source_hashes: dict[str, str] = {}
        for source in logical_sources:
            path = config_path if source == "configs/audit.json" else root / "metrics" / source
            source_hashes[source] = file_sha256(path)
        rows.append(
            {
                "figure": filename,
                "source_files": json.dumps(logical_sources, separators=(",", ":")),
                "source_sha256": json.dumps(source_hashes, sort_keys=True, separators=(",", ":")),
                "size_bytes": destination.stat().st_size,
                "sha256": file_sha256(destination),
            }
        )
    manifest = pd.DataFrame(rows).reindex(columns=FIGURE_MANIFEST_COLUMNS)
    _atomic_csv(manifest, manifest_path)
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    os.umask(0o022)
    manifest = generate_all(args.root, overwrite=args.overwrite)
    print(f"rendered {len(manifest)} aggregate-only figures")


if __name__ == "__main__":
    main()

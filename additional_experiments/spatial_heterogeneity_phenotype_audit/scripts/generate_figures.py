#!/usr/bin/env python3
"""Render the eight preregistered public figures from aggregate metrics only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from common import (
    atomic_csv,
    file_sha256,
    load_config,
    require_preregistration_lock,
)  # noqa: E402
from run_feature_matrix import (  # noqa: E402
    REPRESENTATIVE_PATH,
    validate_representative_asset,
)


FIGURES = (
    "01_pooling_schematic.png",
    "02_phenotype_auroc_by_pooling.png",
    "03_pcr_auroc.png",
    "04_beyond_ftv_delta.png",
    "05_mean_vs_std.png",
    "06_core_peritumoral_comparison.png",
    "07_longitudinal_heterogeneity.png",
    "08_representative_spatial_activation_statistics.png",
)
COLORS = {
    "P1": "#4c78a8",
    "P2": "#f58518",
    "P3": "#54a24b",
    "P4": "#e45756",
    "P5": "#b279a2",
    "CORE": "#4c78a8",
    "PERI10": "#f58518",
    "PERI20": "#e45756",
    "LOCAL_REST": "#72b7b2",
    "CORE_PERI": "#54a24b",
    "FIXED_P3": "#777777",
}
ORACLE_COMPARATORS = ("CORE", "PERI10", "PERI20", "LOCAL_REST", "CORE_PERI")
PHENOTYPE_TARGETS = ("HR", "HER2", "subtype_4class")
FIGURE_MANIFEST_COLUMNS = (
    "figure",
    "size_bytes",
    "sha256",
    "source_inputs_sha256",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Accept no formal options so help/typos stop before touching outputs."""

    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o755)
    figure.savefig(
        path,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "spatial_heterogeneity_phenotype_audit"},
    )
    path.chmod(0o644)
    plt.close(figure)


def _read(name: str) -> pd.DataFrame:
    path = ROOT / "metrics" / name
    if not path.is_file():
        raise FileNotFoundError(path)
    before = file_sha256(path)
    frame = pd.read_csv(path)
    after = file_sha256(path)
    if before != after:
        raise RuntimeError(f"aggregate table changed while loading: {path}")
    if frame.empty:
        raise ValueError(f"aggregate table is empty: {path}")
    frame.attrs["source_sha256"] = before
    return frame


def figure_source_inputs(
    *,
    config_sha256: str,
    lock_sha256: str,
    table_sha256: dict[str, str],
    representative_metadata_sha256: str,
    representative_sha256: str,
) -> dict[str, dict[str, str]]:
    """Return the exact logical source-hash map for every public figure."""

    required_tables = {
        "table2_phenotype_probes.csv",
        "table3_mri_only_pcr.csv",
        "table4_clinical_ftv_incremental.csv",
        "table6_longitudinal_heterogeneity.csv",
        "table7_oracle_regions.csv",
    }
    if set(table_sha256) != required_tables:
        raise ValueError("figure source table inventory drifted")
    return {
        FIGURES[0]: {
            "audit_config": str(config_sha256),
            "preregistration_lock": str(lock_sha256),
        },
        FIGURES[1]: {
            "table2_phenotype_probes": table_sha256["table2_phenotype_probes.csv"]
        },
        FIGURES[2]: {"table3_mri_only_pcr": table_sha256["table3_mri_only_pcr.csv"]},
        FIGURES[3]: {
            "table4_clinical_ftv_incremental": table_sha256[
                "table4_clinical_ftv_incremental.csv"
            ]
        },
        FIGURES[4]: {
            "table2_phenotype_probes": table_sha256["table2_phenotype_probes.csv"]
        },
        FIGURES[5]: {
            "table7_oracle_regions": table_sha256["table7_oracle_regions.csv"]
        },
        FIGURES[6]: {
            "table6_longitudinal_heterogeneity": table_sha256[
                "table6_longitudinal_heterogeneity.csv"
            ]
        },
        FIGURES[7]: {
            "designated_feature_metadata": str(representative_metadata_sha256),
            "representative_activation": str(representative_sha256),
        },
    }


def encode_source_inputs(value: dict[str, str]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def pooling_schematic() -> plt.Figure:
    figure, axis = plt.subplots(figsize=(10.5, 4.2))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    def box(
        x: float, y: float, width: float, height: float, text: str, color: str
    ) -> None:
        axis.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.015",
                facecolor=color,
                edgecolor="#333333",
                linewidth=1.0,
            )
        )
        axis.text(
            x + width / 2,
            y + height / 2,
            text,
            ha="center",
            va="center",
            color="white" if color != "#eeeeee" else "#222222",
            weight="bold",
        )

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        axis.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.2,
                color="#444444",
            )
        )

    box(0.02, 0.40, 0.16, 0.22, "Frozen LOCAL encoder\n128 x 14 x 22 x 20", "#3b5b92")
    box(0.24, 0.40, 0.16, 0.22, "Fixed central 64-mm\nfractional support", "#457b9d")
    arrow(0.18, 0.51, 0.24, 0.51)
    entries = (
        (0.48, 0.77, "P1 mean\n128-D", "#4c78a8"),
        (0.48, 0.59, "P2 SD\n128-D", "#f58518"),
        (0.48, 0.41, "P3 mean + SD\n256-D (primary)", "#54a24b"),
        (0.48, 0.23, "P4 Q25/Q50/Q75\n384-D", "#e45756"),
        (0.48, 0.05, "P5 all statistics\n640-D diagnostic", "#b279a2"),
    )
    for x, y, text, color in entries:
        box(x, y, 0.21, 0.12, text, color)
        arrow(0.40, 0.51, x, y + 0.06)
    box(
        0.78,
        0.40,
        0.19,
        0.22,
        "Outer-fold-isolated\nlinear probes\nHR / HER2 / subtype / pCR",
        "#eeeeee",
    )
    for _, y, _, _ in entries:
        arrow(0.69, y + 0.06, 0.78, 0.51)
    axis.set_title(
        "Frozen spatial-statistic pooling audit (no encoder retraining)",
        weight="bold",
        pad=8,
    )
    return figure


def phenotype_figure(table: pd.DataFrame) -> plt.Figure:
    selected = table.loc[table["variant"].isin(["P1", "P2", "P3", "P4", "P5"])].copy()
    targets = [
        value
        for value in ("HR", "HER2", "subtype_4class")
        if value in set(selected["target"])
    ]
    figure, axes = plt.subplots(
        1, len(targets), figsize=(4.2 * len(targets), 3.8), sharey=True, squeeze=False
    )
    for axis, target in zip(axes[0], targets, strict=True):
        current = selected.loc[selected["target"].eq(target)]
        summary = (
            current.groupby(["view", "variant"], sort=False)["auroc"]
            .mean()
            .unstack("variant")
        )
        for variant in ["P1", "P2", "P3", "P4", "P5"]:
            if variant in summary:
                axis.plot(
                    summary.index,
                    summary[variant],
                    marker="o",
                    linewidth=1.5,
                    label=variant,
                    color=COLORS[variant],
                )
        axis.axhline(0.5, color="#999999", linestyle="--", linewidth=0.8)
        axis.set_title(target)
        axis.set_xlabel("Visit")
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.2)
    axes[0, 0].set_ylabel("OOF AUROC (mean across arms/seeds)")
    axes[0, -1].legend(frameon=False, ncol=1)
    figure.suptitle("Phenotype decodability by pooling statistic", weight="bold")
    return figure


def pcr_figure(table: pd.DataFrame) -> plt.Figure:
    population = (
        "ftv_complete_375"
        if "ftv_complete_375" in set(table["population"].dropna())
        else table["population"].dropna().iloc[0]
    )
    selected = table.loc[
        table["population"].eq(population)
        & table["variant"].isin(["P1", "P2", "P3", "P4", "P5"])
    ]
    summary = (
        selected.groupby(["view", "variant"], sort=False)["auroc"]
        .mean()
        .unstack("variant")
    )
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    for variant in ["P1", "P2", "P3", "P4", "P5"]:
        if variant in summary:
            axis.plot(
                summary.index,
                summary[variant],
                marker="o",
                label=variant,
                color=COLORS[variant],
            )
    axis.axhline(0.5, color="#999999", linestyle="--", linewidth=0.8)
    axis.set_ylabel("OOF pCR AUROC")
    axis.set_xlabel("Causal image prefix")
    axis.set_title(f"MRI-only pCR by pooling ({population})", weight="bold")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncol=5)
    return figure


def beyond_ftv_figure(table: pd.DataFrame) -> plt.Figure:
    selected = table.loc[table["population"].eq("ftv_complete_375")].copy()
    index = ["seed", "arm", "view"]
    pivot = selected.pivot_table(
        index=index, columns="model", values="auroc", aggfunc="first"
    )
    baseline = pivot["C+F"]
    variants = [value for value in ("C+F+P1", "C+F+P3", "C+F+P4") if value in pivot]
    delta = (
        pivot[variants]
        .subtract(baseline, axis=0)
        .reset_index()
        .melt(index, var_name="model", value_name="delta")
    )
    summary = (
        delta.groupby(["view", "model"], sort=False)["delta"].mean().unstack("model")
    )
    figure, axis = plt.subplots(figsize=(7.4, 4.2))
    color = {"C+F+P1": COLORS["P1"], "C+F+P3": COLORS["P3"], "C+F+P4": COLORS["P4"]}
    for model in variants:
        axis.plot(
            summary.index,
            summary[model],
            marker="o",
            label=f"{model} vs C+F",
            color=color[model],
        )
    axis.axhline(0.0, color="#333333", linewidth=0.9)
    axis.set_ylabel("AUROC increment")
    axis.set_xlabel("Causal timing")
    axis.set_title("Image statistics beyond clinical + FTV", weight="bold")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    return figure


def mean_std_figure(table: pd.DataFrame) -> plt.Figure:
    selected = table.loc[table["variant"].isin(["P1", "P2"])].copy()
    group = ["target", "view", "seed", "arm"]
    pivot = selected.pivot_table(
        index=group, columns="variant", values="auroc", aggfunc="first"
    ).dropna(subset=["P1", "P2"])
    figure, axis = plt.subplots(figsize=(5.2, 5.0))
    for target, current in pivot.groupby(level="target"):
        axis.scatter(current["P1"], current["P2"], alpha=0.7, s=28, label=target)
    lower = float(min(pivot[["P1", "P2"]].min())) - 0.01
    upper = float(max(pivot[["P1", "P2"]].max())) + 0.01
    axis.plot(
        [lower, upper], [lower, upper], linestyle="--", color="#777777", linewidth=1
    )
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_xlabel("P1 mean AUROC")
    axis.set_ylabel("P2 SD AUROC")
    axis.set_title("Independent value of spatial SD", weight="bold")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    return figure


def pair_matched_oracle_deltas(table: pd.DataFrame) -> pd.DataFrame:
    """Return oracle-minus-fixed-P3 AUROC within each exact pair population."""

    keys = ["seed", "arm", "view", "target", "population"]
    required = {*keys, "variant", "n", "auroc"}
    if not required.issubset(table.columns):
        raise ValueError(
            f"oracle table lacks pair-matching columns: {sorted(required - set(table))}"
        )
    selected = table.loc[
        table["variant"].isin((*ORACLE_COMPARATORS, "FIXED_P3"))
    ].copy()
    if selected.duplicated([*keys, "variant"]).any():
        raise ValueError("oracle table repeats a pair/variant identity")
    oracle = selected.loc[selected["variant"].isin(ORACLE_COMPARATORS)].copy()
    if len(oracle) * 2 != len(selected):
        raise ValueError(
            "oracle table does not contain one fixed-P3 row per comparator row"
        )
    expected_population = "oracle_pair_" + oracle["variant"].astype(str)
    if not oracle["population"].astype(str).eq(expected_population).all():
        raise ValueError("oracle comparator appears in another pair population")
    fixed = selected.loc[
        selected["variant"].eq("FIXED_P3"), [*keys, "n", "auroc"]
    ].rename(columns={"n": "fixed_n", "auroc": "fixed_auroc"})
    paired = oracle.merge(fixed, on=keys, how="left", validate="one_to_one")
    if paired[["fixed_n", "fixed_auroc"]].isna().any().any():
        raise ValueError("oracle comparator lacks its pair-matched fixed-P3 reference")
    if not paired["n"].eq(paired["fixed_n"]).all():
        raise ValueError("oracle and fixed-P3 pair populations have different coverage")
    paired["delta_auroc"] = paired["auroc"] - paired["fixed_auroc"]
    if not np.isfinite(paired["delta_auroc"].to_numpy(dtype=float)).all():
        raise ValueError("oracle pair-matched AUROC delta is non-finite")
    return paired


def oracle_figure(table: pd.DataFrame) -> plt.Figure:
    paired = pair_matched_oracle_deltas(table)
    paired = paired.loc[paired["target"].isin(PHENOTYPE_TARGETS)].copy()
    if paired.empty:
        raise ValueError("oracle table has no pair-matched phenotype rows")
    variants = [
        value for value in ORACLE_COMPARATORS if value in set(paired["variant"])
    ]
    summary = (
        paired.groupby(["target", "variant"], sort=False)["delta_auroc"]
        .mean()
        .unstack("variant")
    )
    figure, axis = plt.subplots(figsize=(8.0, 4.5))
    x = np.arange(len(summary.index))
    width = 0.82 / max(1, len(variants))
    for offset, variant in enumerate(variants):
        axis.bar(
            x - 0.41 + width / 2 + offset * width,
            summary[variant],
            width=width,
            label=variant,
            color=COLORS.get(variant, "#999999"),
        )
    axis.axhline(0.0, color="#333333", linestyle="--", linewidth=0.8)
    axis.set_xticks(x, summary.index, rotation=20, ha="right")
    axis.set_ylabel("Pair-matched AUROC increment vs fixed P3")
    axis.set_title(
        "Phenotype localization: oracle vs same-cohort fixed P3", weight="bold"
    )
    axis.legend(frameon=False, ncol=3)
    axis.grid(axis="y", alpha=0.2)
    return figure


def longitudinal_figure(table: pd.DataFrame) -> plt.Figure:
    variants = [
        value
        for value in ("DELTA_MEAN", "DELTA_STD", "P3_PLUS_DELTA")
        if value in set(table["variant"])
    ]
    summary = (
        table.loc[table["variant"].isin(variants)]
        .groupby(["view", "variant"], sort=False)["auroc"]
        .mean()
        .unstack("variant")
    )
    figure, axis = plt.subplots(figsize=(7.0, 4.1))
    for variant in variants:
        axis.plot(
            summary.index, summary[variant], marker="o", linewidth=1.6, label=variant
        )
    axis.axhline(0.5, color="#999999", linestyle="--", linewidth=0.8)
    axis.set_xlabel("Observed treatment interval")
    axis.set_ylabel("OOF pCR AUROC")
    axis.set_title("Longitudinal heterogeneity change", weight="bold")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    return figure


def representative_figure(
    path: Path, *, expected_sha256: str | None = None
) -> plt.Figure:
    arrays = validate_representative_asset(path, expected_sha256=expected_sha256)
    activation = np.asarray(arrays["activation_mean_abs"], dtype=float)
    activation_std = np.asarray(arrays["activation_channel_std"], dtype=float)
    local = np.asarray(arrays["local_weight"], dtype=float)
    region = np.asarray(arrays["region_weight"], dtype=float)
    regions = tuple(np.asarray(arrays["regions"]).astype(str))
    maps = [
        ("Mean |activation|", activation.max(axis=0), "magma"),
        ("Across-channel SD", activation_std.max(axis=0), "viridis"),
        ("Fixed LOCAL support", local.max(axis=0), "Blues"),
        (regions[0], region[0].max(axis=0), "Reds"),
        (regions[1], region[1].max(axis=0), "Oranges"),
        (regions[2], region[2].max(axis=0), "Purples"),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(9.0, 6.0))
    for axis, (title, image, cmap) in zip(axes.flat, maps, strict=True):
        handle = axis.imshow(image, cmap=cmap, origin="lower", aspect="auto")
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(handle, ax=axis, fraction=0.046, pad=0.03)
    figure.suptitle(
        "De-identified representative spatial activation/statistics (max Z projection)",
        weight="bold",
    )
    return figure


def main() -> None:
    parse_args()
    os.umask(0o077)
    config = load_config(ROOT / "configs" / "audit.json", verify_inputs=True)
    require_preregistration_lock(config)
    output_dir = ROOT / "figures"
    manifest_path = ROOT / "metrics" / "figure_manifest.csv"
    destinations = [output_dir / name for name in FIGURES]
    if manifest_path.exists():
        raise FileExistsError("completed figure manifest already exists")
    # Safe recovery from an interrupted renderer: remove only the exact
    # preregistered figure set when no completion manifest was published.
    for destination in destinations:
        destination.unlink(missing_ok=True)
    _style()
    table2 = _read("table2_phenotype_probes.csv")
    table3 = _read("table3_mri_only_pcr.csv")
    table4 = _read("table4_clinical_ftv_incremental.csv")
    table6 = _read("table6_longitudinal_heterogeneity.csv")
    table7 = _read("table7_oracle_regions.csv")
    representative_metadata_path = (
        ROOT
        / "features"
        / "seed_2026"
        / "LOCAL3"
        / "fold_0"
        / "spatial_statistics.private.metadata.json"
    )
    representative_metadata_sha256 = file_sha256(representative_metadata_path)
    representative_metadata_payload = json.loads(
        representative_metadata_path.read_text(encoding="utf-8")
    )
    if file_sha256(representative_metadata_path) != representative_metadata_sha256:
        raise RuntimeError("designated feature metadata changed while loading")
    representative_metadata = representative_metadata_payload.get(
        "representative_activation"
    )
    if not isinstance(representative_metadata, dict):
        raise ValueError("designated feature metadata lacks Figure-8 provenance")
    if (
        Path(str(representative_metadata.get("path", ""))).resolve()
        != REPRESENTATIVE_PATH.resolve()
    ):
        raise ValueError(
            "Figure-8 source path differs from designated feature metadata"
        )
    representative_sha256 = str(representative_metadata.get("sha256", ""))
    validate_representative_asset(
        REPRESENTATIVE_PATH, expected_sha256=representative_sha256
    )
    tables_by_name = {
        "table2_phenotype_probes.csv": table2,
        "table3_mri_only_pcr.csv": table3,
        "table4_clinical_ftv_incremental.csv": table4,
        "table6_longitudinal_heterogeneity.csv": table6,
        "table7_oracle_regions.csv": table7,
    }
    source_inputs = figure_source_inputs(
        config_sha256=file_sha256(ROOT / "configs" / "audit.json"),
        lock_sha256=file_sha256(ROOT / "PREREGISTRATION_LOCK.json"),
        table_sha256={
            name: str(frame.attrs["source_sha256"])
            for name, frame in tables_by_name.items()
        },
        representative_metadata_sha256=representative_metadata_sha256,
        representative_sha256=representative_sha256,
    )
    builders = (
        pooling_schematic(),
        phenotype_figure(table2),
        pcr_figure(table3),
        beyond_ftv_figure(table4),
        mean_std_figure(table2),
        oracle_figure(table7),
        longitudinal_figure(table6),
        representative_figure(
            REPRESENTATIVE_PATH,
            expected_sha256=representative_sha256,
        ),
    )
    rows: list[dict[str, object]] = []
    for filename, figure in zip(FIGURES, builders, strict=True):
        destination = output_dir / filename
        _save(figure, destination)
        rows.append(
            {
                "figure": filename,
                "size_bytes": destination.stat().st_size,
                "sha256": file_sha256(destination),
                "source_inputs_sha256": encode_source_inputs(source_inputs[filename]),
            }
        )
    atomic_csv(
        pd.DataFrame(rows).reindex(columns=FIGURE_MANIFEST_COLUMNS), manifest_path
    )
    print(f"rendered {len(rows)} figures")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create the anonymous aggregate raw-DICOM repair QC figure."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    summary = pd.read_csv(ROOT / "metrics/dicom_pixel_rebuild_summary.csv")
    visits = pd.read_csv(ROOT / "metrics/dicom_pixel_rebuild_visit_qc.csv")
    groups = summary[summary["scope"].isin(["FORMAL_72", "BASE_ONLY_EXTENSION_74"])]

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    labels = ["Formal 72", "Base extension 74"]
    positions = range(len(groups))
    axes[0].bar(positions, groups["verified_cells"], color=("#3182bd", "#9ecae1"))
    axes[0].set_xticks(positions, labels)
    axes[0].set_ylabel("PixelData cells")
    axes[0].set_title("Decoded and independently re-compared")
    for position, (_, row) in enumerate(groups.iterrows()):
        axes[0].text(
            position,
            float(row["verified_cells"]),
            f"{int(row['verified_cells']):,}\nerror={row['max_cell_error']:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[0].grid(axis="y", alpha=0.25)

    for scope, color, label in (
        ("FORMAL_72", "#08519c", "Formal 72"),
        ("BASE_ONLY_EXTENSION_74", "#6baed6", "Base extension 74"),
    ):
        subset = visits[visits["scope"].eq(scope)]
        axes[1].scatter(
            range(len(subset)),
            subset["footprint_corner_error_mm"],
            s=18,
            alpha=0.75,
            color=color,
            label=label,
        )
    axes[1].axhline(0.1, color="#cb181d", linestyle="--", label="0.1-mm gate")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Anonymous rebuilt visit")
    axes[1].set_ylabel("Footprint-corner error (mm)")
    axes[1].set_title("Physical geometry after PixelData rebuild")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    output = ROOT / "figures/01_raw_dicom_repair_qc.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()

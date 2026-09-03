#!/usr/bin/env python3
"""Generate aggregate-only Goal C figures; no patient identifiers are used."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, default=ROOT / "metrics")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mri_path = args.metrics_dir / "mri_only_metrics.csv"
    if mri_path.exists() and mri_path.stat().st_size:
        mri = pd.read_csv(mri_path)
        if not mri.empty and {"arm", "timing", "auroc_mean"}.issubset(mri.columns):
            pivot = mri.pivot(index="timing", columns="arm", values="auroc_mean")
            ax = pivot.plot(kind="bar", figsize=(9, 5), ylim=(0.35, 0.75), title="MRI-only OOF AUROC (fold/seed mean)")
            ax.set_ylabel("AUROC"); ax.set_xlabel("Timing"); ax.grid(axis="y", alpha=0.25)
            plt.tight_layout(); plt.savefig(args.output_dir / "01_mri_only_auroc.png", dpi=180); plt.close()
    gap_path = args.metrics_dir / "generalization_gap.csv"
    if gap_path.exists() and gap_path.stat().st_size:
        gap = pd.read_csv(gap_path)
        if not gap.empty:
            summary = gap.groupby(["arm", "timing"], as_index=False)["train_minus_oof_auroc"].mean()
            pivot = summary.pivot(index="timing", columns="arm", values="train_minus_oof_auroc")
            ax = pivot.plot(kind="bar", figsize=(9, 5), title="Train–OOF AUROC gap")
            ax.set_ylabel("AUROC gap"); ax.set_xlabel("Timing"); ax.grid(axis="y", alpha=0.25)
            plt.tight_layout(); plt.savefig(args.output_dir / "02_generalization_gap.png", dpi=180); plt.close()
    attention_path = args.metrics_dir / "attention_diagnostics.csv"
    if attention_path.exists() and attention_path.stat().st_size:
        attention = pd.read_csv(attention_path)
        if not attention.empty and {"arm", "attention_entropy"}.issubset(attention.columns):
            summary = attention.groupby("arm", as_index=False)[["attention_entropy", "attention_concentration_top10"]].mean(numeric_only=True)
            summary.set_index("arm").plot(kind="bar", figsize=(8, 5), title="Attention entropy and top-10% concentration")
            plt.ylabel("Diagnostic value"); plt.grid(axis="y", alpha=0.25); plt.tight_layout(); plt.savefig(args.output_dir / "03_attention_diagnostics.png", dpi=180); plt.close()
    fusion_path = args.metrics_dir / "fusion_fold_metrics.csv"
    if fusion_path.exists() and fusion_path.stat().st_size:
        fusion = pd.read_csv(fusion_path)
        if not fusion.empty and "full_808" in set(fusion["population"]):
            full = fusion.loc[fusion["population"].eq("full_808")]
            summary = full.groupby(["arm", "model"], as_index=False)["auroc"].mean()
            pivot = summary.pivot(index="arm", columns="model", values="auroc")
            pivot.plot(kind="bar", figsize=(10, 5), title="Fold-safe clinical complementarity")
            plt.ylabel("Test AUROC"); plt.grid(axis="y", alpha=0.25); plt.tight_layout(); plt.savefig(args.output_dir / "04_clinical_complementarity.png", dpi=180); plt.close()
    print({"status": "COMPLETE", "output_dir": str(args.output_dir)})


if __name__ == "__main__":
    main()

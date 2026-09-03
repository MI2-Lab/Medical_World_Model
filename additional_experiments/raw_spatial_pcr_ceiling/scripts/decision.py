#!/usr/bin/env python3
"""Apply preregistered Goal C gates to public aggregate metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = {"T0", "T0_T1", "T0_T2"}


def _spatial_gate(frame: pd.DataFrame, candidate: str, threshold: float, reference: str = "C0") -> dict[str, object]:
    rows = frame.loc[(frame["candidate_arm"].eq(candidate)) & (frame["reference_arm"].eq(reference)) & (frame["timing"].isin(PRIMARY))].copy()
    timing_rows: list[dict[str, object]] = []
    for timing, group in rows.groupby("timing", sort=True):
        seed_effects = group.groupby("seed")["delta_auroc"].mean()
        if set(seed_effects.index) != {2026, 3026}:
            continue
        timing_rows.append({"timing": timing, "seed_2026_delta_auroc": float(seed_effects.loc[2026]), "seed_3026_delta_auroc": float(seed_effects.loc[3026]), "mean_delta_auroc": float(seed_effects.mean()), "both_seeds_positive": bool((seed_effects > 0).all())})
    passing = [row for row in timing_rows if row["mean_delta_auroc"] >= threshold and row["both_seeds_positive"]]
    return {"status": "PASS" if passing else ("FAIL" if timing_rows else "NOT_RUN"), "candidate": candidate, "reference": reference, "threshold": threshold, "timings": timing_rows, "passing_timings": passing}


def _fusion_gate(path: Path, base_model: str, augmented_model: str, population: str) -> dict[str, object]:
    if not path.exists():
        return {"status": "NOT_RUN", "population": population}
    frame = pd.read_csv(path)
    frame = frame.loc[frame["population"].eq(population)].copy()
    if frame.empty:
        return {"status": "NOT_RUN", "population": population}
    rows: list[dict[str, object]] = []
    for (arm, timing), group in frame.groupby(["arm", "timing"], sort=True):
        # Fusion metrics contain one row per outer fold.  Collapse folds
        # within each seed before comparing the paired seed-level estimates;
        # indexing the raw fold rows by seed produces duplicate labels.
        base = group.loc[group["model"].eq(base_model)].groupby("seed")["auroc"].mean()
        augmented = group.loc[group["model"].eq(augmented_model)].groupby("seed")["auroc"].mean()
        common = sorted(set(base.index) & set(augmented.index))
        if common != [2026, 3026]:
            continue
        deltas = {str(seed): float(augmented.loc[seed] - base.loc[seed]) for seed in common}
        rows.append({"arm": arm, "timing": timing, "delta_seed_2026": deltas["2026"], "delta_seed_3026": deltas["3026"], "both_seeds_positive": all(value > 0 for value in deltas.values()), "mean_delta": float(sum(deltas.values()) / len(deltas))})
    passing = [row for row in rows if row["both_seeds_positive"]]
    return {"status": "PASS" if passing else ("FAIL" if rows else "NOT_RUN"), "population": population, "base_model": base_model, "augmented_model": augmented_model, "rows": rows, "passing_rows": passing}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, default=ROOT / "metrics")
    args = parser.parse_args()
    output: dict[str, object] = {"schema_version": 1, "experiment": "raw_spatial_pcr_ceiling", "status": "NOT_RUN", "gates": {}, "decision_class": "NOT_RUN", "reporting_boundary": "empirical ceiling under the current C1B-H input contract"}
    bootstrap_path = args.metrics_dir / "paired_bootstrap.csv"
    if not bootstrap_path.exists() or bootstrap_path.stat().st_size == 0:
        (args.metrics_dir / "decision.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(output, indent=2, sort_keys=True))
        return
    bootstrap = pd.read_csv(bootstrap_path)
    if bootstrap.empty:
        (args.metrics_dir / "decision.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(output, indent=2, sort_keys=True))
        return
    c2 = _spatial_gate(bootstrap, "C2", 0.05)
    c3 = _spatial_gate(bootstrap, "C3", 0.05)
    c5 = _spatial_gate(bootstrap, "C5", 0.08)
    c4 = _spatial_gate(bootstrap, "C4", 0.03, reference="C3")
    output["gates"] = {
        "A_spatial_vs_C0": {"status": "PASS" if c2["status"] == "PASS" or c3["status"] == "PASS" else ("FAIL" if c2["status"] == "FAIL" and c3["status"] == "FAIL" else "NOT_RUN"), "C2": c2, "C3": c3},
        "B_raw_vs_C0": c5,
        "C_full_context_vs_LOCAL_patch": c4,
        "D_clinical_increment": _fusion_gate(args.metrics_dir / "fusion_fold_metrics.csv", "C", "C_PLUS_M", "full_808"),
        "E_ftv_increment": _fusion_gate(args.metrics_dir / "fusion_fold_metrics.csv", "C_PLUS_F", "C_PLUS_F_PLUS_M", "ftv_complete_375"),
    }
    gap_path = args.metrics_dir / "generalization_gap.csv"
    overfit = {"status": "NOT_RUN"}
    if gap_path.exists():
        gaps = pd.read_csv(gap_path)
        relevant = gaps.loc[gaps["timing"].isin(PRIMARY)].copy()
        if not relevant.empty:
            flags = ((relevant["train_auroc"] >= 0.80) & (relevant["oof_auroc"] <= 0.55))
            overfit = {"status": "PASS" if bool(flags.any()) else "FAIL", "n_flagged_cells": int(flags.sum()), "threshold_train_auroc": 0.80, "threshold_oof_auroc": 0.55}
    output["gates"]["overfit_generalization_audit"] = overfit
    output["status"] = "COMPLETE"
    gate_a = output["gates"]["A_spatial_vs_C0"]["status"]
    gate_b = output["gates"]["B_raw_vs_C0"]["status"]
    if gate_a == "PASS":
        output["decision_class"] = "POOLING_BOTTLENECK"
    elif gate_b == "PASS":
        output["decision_class"] = "INPUT_REPRESENTATION_BOTTLENECK"
    elif overfit.get("status") == "PASS":
        output["decision_class"] = "GENERALIZATION_BOTTLENECK"
    else:
        output["decision_class"] = "LOW_RAW_SPATIAL_CEILING_OR_NO_SUPPORTED_GAIN"
    (args.metrics_dir / "decision.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

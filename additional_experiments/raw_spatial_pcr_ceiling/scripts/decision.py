#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply frozen Goal C gates to aggregate metrics.")
    parser.add_argument("--metrics-dir", type=Path, default=ROOT / "metrics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = {"schema_version": 1, "experiment": "raw_spatial_pcr_ceiling", "status": "NOT_RUN", "gates": {}, "decision_class": "NOT_RUN", "reporting_boundary": "empirical ceiling under the current C1B-H input contract"}
    metrics_path = args.metrics_dir / "paired_bootstrap.csv"
    if not metrics_path.exists() or metrics_path.stat().st_size == 0:
        (args.metrics_dir / "decision.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(output, indent=2, sort_keys=True))
        return
    frame = pd.read_csv(metrics_path)
    if frame.empty:
        (args.metrics_dir / "decision.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(output, indent=2, sort_keys=True))
        return
    primary = frame[frame["timing"].isin(["T0", "T0_T1", "T0_T2"])]
    def gate(candidate: str, threshold: float, reference: str = "C0") -> dict:
        rows = primary[(primary["candidate_arm"] == candidate) & (primary["reference_arm"] == reference)]
        if rows.empty:
            return {"status": "NOT_RUN", "candidate": candidate, "reference": reference}
        timing_rows = []
        for timing, timing_group in rows.groupby("timing"):
            seed_effects = timing_group.groupby("seed")["delta_auroc"].mean()
            if set(seed_effects.index) == {2026, 3026}:
                timing_rows.append({"timing": timing, "mean_delta_auroc": float(seed_effects.mean()), "both_seeds_positive": bool((seed_effects > 0).all())})
        passing = [row for row in timing_rows if row["mean_delta_auroc"] >= threshold and row["both_seeds_positive"]]
        best = max(timing_rows, key=lambda row: row["mean_delta_auroc"], default=None)
        return {"status": "PASS" if passing else "FAIL", "candidate": candidate, "reference": reference, "best": best, "passing_timings": passing, "threshold": threshold}
    c2 = gate("C2", 0.05)
    c3 = gate("C3", 0.05)
    output["gates"]["A_spatial"] = {"status": "PASS" if c2["status"] == "PASS" or c3["status"] == "PASS" else ("FAIL" if c2["status"] == "FAIL" or c3["status"] == "FAIL" else "NOT_RUN"), "C2": c2, "C3": c3}
    output["gates"]["B_raw"] = gate("C5", 0.08)
    output["gates"]["C_context"] = gate("C4", 0.03, reference="C3")
    output["status"] = "COMPLETE"
    if output["gates"]["B_raw"]["status"] == "PASS" and output["gates"]["A_spatial"]["status"] != "PASS":
        output["decision_class"] = "INPUT_REPRESENTATION_BOTTLENECK"
    elif output["gates"]["A_spatial"]["status"] == "PASS":
        output["decision_class"] = "POOLING_BOTTLENECK"
    else:
        output["decision_class"] = "LOW_RAW_SPATIAL_CEILING_OR_PENDING_CONTEXT"
    (args.metrics_dir / "decision.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

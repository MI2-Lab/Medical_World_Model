#!/usr/bin/env python3
"""Normalize the previously completed supervised pooled MRI reference to Goal C."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prior = pd.read_csv(args.prior_predictions)
    manifest = pd.read_csv(args.manifest)
    selected = prior.loc[
        prior["arm"].eq("B1")
        & prior["model_family"].eq("M")
        & prior["population"].eq("full_808")
    ].copy()
    selected["patient_id"] = selected["patient_id"].astype(str)
    index = manifest.drop_duplicates("patient_id")[["patient_id", "row_index"]]
    selected = selected.merge(index, on="patient_id", how="left", validate="many_to_one")
    selected = selected.rename(columns={"predicted_probability": "probability"})
    selected["arm"] = "C0"
    selected = selected[["row_index", "patient_id", "split", "y_true", "probability", "seed", "fold", "arm", "timing"]]
    expected = 2 * 5 * 4 * 808
    if len(selected) != expected or selected.duplicated(["seed", "fold", "timing", "split", "row_index"]).any():
        raise SystemExit(f"C0 reference has wrong coverage: rows={len(selected)} expected={expected}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output, index=False)
    print({"status": "COMPLETE", "rows": len(selected), "source": str(args.prior_predictions)})


if __name__ == "__main__":
    main()

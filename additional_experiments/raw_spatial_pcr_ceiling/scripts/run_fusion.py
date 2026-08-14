#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from raw_spatial_pcr.fusion import fit_fold_safe_logistic
from raw_spatial_pcr.metrics import classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fold-safe clinical/FTV/MRI fusion using private low-dimensional scores.")
    parser.add_argument("--input-csv", type=Path, required=True, help="private row-level score table")
    parser.add_argument("--mri-column", default="mri_score")
    parser.add_argument("--clinical-columns", nargs="+", required=True)
    parser.add_argument("--ftv-column")
    parser.add_argument("--population", required=True, choices=("full_808", "ftv_complete_375"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "metrics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input_csv)
    required = {"row_index", "seed", "fold", "timing", "split", "y_true", args.mri_column, *args.clinical_columns}
    if args.ftv_column:
        required.add(args.ftv_column)
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"private fusion table missing columns: {sorted(missing)}")
    rows = []
    for keys, group in frame.groupby(["seed", "fold", "timing"], sort=True):
        by_split = {name: part.sort_values("row_index") for name, part in group.groupby("split")}
        if not {"train", "validation", "test"}.issubset(by_split):
            continue
        train, validation, test = by_split["train"], by_split["validation"], by_split["test"]
        feature_sets = {"C": args.clinical_columns, "C_PLUS_M": [*args.clinical_columns, args.mri_column]}
        if args.ftv_column:
            feature_sets["C_PLUS_F"] = [*args.clinical_columns, args.ftv_column]
            feature_sets["C_PLUS_F_PLUS_M"] = [*args.clinical_columns, args.ftv_column, args.mri_column]
        for model_name, columns in feature_sets.items():
            test_probability, selection = fit_fold_safe_logistic(train[columns].to_numpy(float), train.y_true.to_numpy(float), validation[columns].to_numpy(float), validation.y_true.to_numpy(float), test[columns].to_numpy(float))
            values = classification_metrics(test.y_true, test_probability)
            values.update({"seed": keys[0], "fold": keys[1], "timing": keys[2], "population": args.population, "model": model_name, **selection})
            rows.append(values)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / f"fusion_{args.population}.csv", index=False)
    print(f"wrote {len(rows)} fold-safe fusion rows")


if __name__ == "__main__":
    main()


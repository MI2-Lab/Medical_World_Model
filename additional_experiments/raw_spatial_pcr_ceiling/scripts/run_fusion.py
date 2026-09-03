#!/usr/bin/env python3
"""Run fold-safe clinical/FTV fusion for one MRI arm."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "conditional_pcr_contrastive_ceiling" / "src"))

from conditional_ceiling.clinical import CLINICAL_FIELDS, TrainOnlyClinicalEncoder, ftv_prefix_matrix, load_ftv_wide  # noqa: E402
from raw_spatial_pcr.fusion import fit_fold_safe_logistic  # noqa: E402
from raw_spatial_pcr.metrics import classification_metrics  # noqa: E402


def _prediction_frame(source: Path, arm: str) -> pd.DataFrame:
    paths = [source] if source.is_file() else sorted(source.rglob("*.private.csv"))
    frames = [pd.read_csv(path) for path in paths]
    if not frames:
        raise FileNotFoundError(f"no private predictions under {source}")
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.loc[frame["arm"].eq(arm)].copy()
    required = {"row_index", "patient_id", "seed", "fold", "timing", "split", "y_true", "probability"}
    if not required.issubset(frame.columns):
        raise ValueError(f"predictions miss {sorted(required - set(frame.columns))}")
    if frame.duplicated(["seed", "fold", "timing", "split", "row_index"]).any():
        raise ValueError("duplicate MRI prediction row")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--population", required=True, choices=("full_808", "ftv_complete_375"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "metrics")
    parser.add_argument("--clinical-table", type=Path)
    parser.add_argument("--ftv-table", type=Path)
    args = parser.parse_args()
    predictions = _prediction_frame(args.predictions, args.arm)
    manifest = pd.read_csv(args.manifest)
    if args.population == "ftv_complete_375" and "ftv_complete" not in manifest:
        raise SystemExit("manifest needs ftv_complete")
    join_columns = ["row_index", "fold", "patient_id", "label_hr", "label_her2", "label_mp", "age_at_screening", "race_simple", "menopausal_status_simple", "ethnicity", "arm", "ftv_complete", "FTV_T0", "FTV_T1", "FTV_T2", "FTV_T3"]
    missing = sorted(set(join_columns) - set(manifest.columns))
    if missing:
        raise SystemExit(f"manifest misses fusion fields: {missing}")
    frame = predictions.merge(manifest[join_columns], on=["row_index", "fold", "patient_id"], how="left", validate="many_to_one", suffixes=("", "_manifest"))
    if frame["label_hr"].isna().any():
        raise SystemExit("fusion join missed clinical rows")
    if args.population == "ftv_complete_375":
        frame = frame.loc[frame["ftv_complete"].eq(1)].copy()
    if frame.empty:
        raise SystemExit("fusion population is empty")
    rows: list[dict[str, float | int | str]] = []
    for (seed, fold, timing), group in frame.groupby(["seed", "fold", "timing"], sort=True):
        by_split = {name: part.copy() for name, part in group.groupby("split")}
        if not {"train", "validation", "test"}.issubset(by_split):
            raise SystemExit("fusion group lacks a complete train/validation/test split")
        train, validation, test = by_split["train"], by_split["validation"], by_split["test"]
        encoder = TrainOnlyClinicalEncoder(CLINICAL_FIELDS).fit(train)
        clinical_train = encoder.transform(train)
        clinical_validation = encoder.transform(validation)
        clinical_test = encoder.transform(test)
        feature_sets: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
            "C": (clinical_train, clinical_validation, clinical_test),
            "C_PLUS_M": (np.column_stack([clinical_train, train["probability"]]), np.column_stack([clinical_validation, validation["probability"]]), np.column_stack([clinical_test, test["probability"]])),
        }
        if args.population == "ftv_complete_375":
            ftv_train = ftv_prefix_matrix(train, str(timing))
            ftv_validation = ftv_prefix_matrix(validation, str(timing))
            ftv_test = ftv_prefix_matrix(test, str(timing))
            feature_sets["F"] = (ftv_train, ftv_validation, ftv_test)
            feature_sets["C_PLUS_F"] = (np.column_stack([clinical_train, ftv_train]), np.column_stack([clinical_validation, ftv_validation]), np.column_stack([clinical_test, ftv_test]))
            feature_sets["C_PLUS_F_PLUS_M"] = (np.column_stack([clinical_train, ftv_train, train["probability"]]), np.column_stack([clinical_validation, ftv_validation, validation["probability"]]), np.column_stack([clinical_test, ftv_test, test["probability"]]))
        for model_name, (train_x, validation_x, test_x) in feature_sets.items():
            probability, selection = fit_fold_safe_logistic(train_x, train["y_true"].to_numpy(float), validation_x, validation["y_true"].to_numpy(float), test_x)
            values = classification_metrics(test["y_true"].to_numpy(float), probability)
            values.update({"seed": int(seed), "fold": int(fold), "timing": str(timing), "population": args.population, "arm": args.arm, "model": model_name, **selection})
            rows.append(values)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"fusion_{args.population}_{args.arm}.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print({"status": "COMPLETE", "arm": args.arm, "population": args.population, "rows": len(rows), "output": str(output)})


if __name__ == "__main__":
    main()

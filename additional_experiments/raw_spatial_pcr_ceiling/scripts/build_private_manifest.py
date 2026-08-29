#!/usr/bin/env python3
"""Build the private 808-patient/five-fold manifest used by the ceiling audit.

The output is deliberately patient-level and must stay in a gitignored private
path.  It contains no image bytes; it only joins the frozen cache inventory,
outer folds, clinical covariates, and the complete-FTV indicator.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONDITIONAL_SRC = ROOT.parent / "conditional_pcr_contrastive_ceiling" / "src"
sys.path.insert(0, str(CONDITIONAL_SRC))

from conditional_ceiling.clinical import load_ftv_wide  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--clinical", type=Path, required=True)
    parser.add_argument("--ftv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache = pd.read_csv(args.cache_manifest)
    folds = pd.read_csv(args.fold_manifest)
    clinical = pd.read_csv(args.clinical)
    required_cache = {"patient_id", "cache_path", "input_kind"}
    required_fold = {"patient_id", "fold", "split", "label_pcr"}
    if not required_cache.issubset(cache.columns):
        raise SystemExit(f"cache manifest misses {sorted(required_cache - set(cache.columns))}")
    if not required_fold.issubset(folds.columns):
        raise SystemExit(f"fold manifest misses {sorted(required_fold - set(folds.columns))}")
    if "patient_id" not in clinical.columns:
        raise SystemExit("clinical table must contain patient_id")
    clinical["patient_id"] = clinical["patient_id"].astype(str)
    patient_ids = sorted(set(clinical["patient_id"]))
    if len(patient_ids) != 808 or clinical["patient_id"].duplicated().any():
        raise SystemExit("clinical table must contain exactly 808 unique patients")
    cache["patient_id"] = cache["patient_id"].astype(str)
    cache = cache.loc[cache["input_kind"].astype(str).eq("c1b")].copy()
    cache = cache.drop_duplicates("patient_id").set_index("patient_id")
    missing_cache = sorted(set(patient_ids) - set(cache.index))
    if missing_cache:
        raise SystemExit(f"cache manifest misses patients: {missing_cache[:5]}")
    folds["patient_id"] = folds["patient_id"].astype(str)
    folds = folds.loc[folds["patient_id"].isin(patient_ids)].copy()
    folds["fold"] = pd.to_numeric(folds["fold"], errors="raise").astype(int)
    if set(folds["fold"]) != set(range(5)) or len(folds) != 808 * 5:
        raise SystemExit("fold manifest must contain 808 rows for each frozen fold")
    if folds.duplicated(["patient_id", "fold"]).any():
        raise SystemExit("fold manifest repeats a patient/fold")
    if set(folds["split"].astype(str)) != {"train", "val", "test"}:
        raise SystemExit("fold manifest must use train/val/test splits")
    # The formal runner uses the explicit validation spelling internally.
    folds["split"] = folds["split"].astype(str).replace({"val": "validation"})
    folds["label_pcr"] = pd.to_numeric(folds["label_pcr"], errors="raise").astype(int)
    if not folds["label_pcr"].isin([0, 1]).all():
        raise SystemExit("fold pCR labels must be binary")
    clinical = clinical.set_index("patient_id", verify_integrity=True).loc[patient_ids].reset_index()
    label_col = pd.to_numeric(clinical["label_pcr"], errors="raise").astype(int)
    if not label_col.isin([0, 1]).all():
        raise SystemExit("clinical pCR labels must be binary")
    clinical["label_pcr"] = label_col
    ftv = load_ftv_wide(str(args.ftv), patient_ids)
    ftv["ftv_complete"] = 1
    merged = folds.merge(
        cache.reset_index()[["patient_id", "cache_path"]],
        on="patient_id",
        how="left",
        validate="many_to_one",
    ).merge(clinical, on="patient_id", how="left", suffixes=("", "_clinical"), validate="many_to_one")
    merged = merged.merge(ftv, on="patient_id", how="left", validate="many_to_one")
    merged["ftv_complete"] = merged["ftv_complete"].fillna(0).astype(int)
    merged["row_index"] = merged["patient_id"].map({patient_id: i for i, patient_id in enumerate(patient_ids)}).astype(int)
    merged["label_pcr"] = merged["label_pcr"].astype(int)
    merged = merged.sort_values(["fold", "row_index"]).reset_index(drop=True)
    if len(merged) != 4040 or merged[["cache_path", "label_pcr"]].isna().any().any():
        raise SystemExit("private manifest join is incomplete")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print({"status": "COMPLETE", "rows": len(merged), "patients": merged["patient_id"].nunique(), "ftv_complete": int(merged.drop_duplicates("patient_id")["ftv_complete"].sum()), "output": str(args.output)})


if __name__ == "__main__":
    main()

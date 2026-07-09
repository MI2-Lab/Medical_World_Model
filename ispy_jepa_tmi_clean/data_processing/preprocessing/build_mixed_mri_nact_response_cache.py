#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from path_defaults import PROJECT_ROOT, ispy2_preprocessed_root


NACT_FEATURE_NAMES = (
    "nact_log_tumor_volume_blu",
    "nact_sphericity",
    "nact_log_ld",
    "nact_log_bpe_5slice_mean",
    "nact_ftv_pch_from_t0",
    "nact_sphericity_pch_from_t0",
    "nact_ld_pch_from_t0",
    "nact_bpe_pch_from_t0",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append MRI-NACT longitudinal features to an existing response feature cache."
    )
    parser.add_argument(
        "--base-cache",
        type=Path,
        default=PROJECT_ROOT / "runs" / "mixed_ispy1_ld_response_cache.npz",
    )
    parser.add_argument(
        "--mri-nact-csv",
        type=Path,
        default=ispy2_preprocessed_root() / "mri_nact_features_complete4visits_wide.csv",
    )
    parser.add_argument(
        "--output-cache",
        type=Path,
        default=PROJECT_ROOT / "runs" / "mixed_ispy1_ld_mri_nact_response_cache.npz",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def to_float(row: dict[str, str], name: str) -> float:
    text = row.get(name, "")
    if text is None or text == "":
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def nonnegative_log1p(value: float) -> float:
    if not np.isfinite(value):
        return float("nan")
    return float(np.log1p(max(value, 0.0)))


def load_nact_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    return {row["patient_id"]: row for row in rows if row.get("patient_id")}


def nact_visit_matrix(row: dict[str, str]) -> np.ndarray:
    x = np.full((4, len(NACT_FEATURE_NAMES)), np.nan, dtype=np.float32)
    for visit_idx in range(4):
        suffix = f"t{visit_idx}"
        x[visit_idx, 0] = nonnegative_log1p(to_float(row, f"tumor_volume_blu_{suffix}"))
        x[visit_idx, 1] = to_float(row, f"sphericity_{suffix}")
        x[visit_idx, 2] = nonnegative_log1p(to_float(row, f"ld_{suffix}"))
        x[visit_idx, 3] = nonnegative_log1p(to_float(row, f"bpe_5slice_mean_{suffix}"))

    pch_prefixes = (
        ("ftv_pch", 4),
        ("sphericity_pch", 5),
        ("ld_pch", 6),
        ("bpe_pch", 7),
    )
    for _, col_idx in pch_prefixes:
        x[0, col_idx] = 0.0
    for visit_idx in (1, 2, 3):
        for prefix, col_idx in pch_prefixes:
            x[visit_idx, col_idx] = to_float(row, f"{prefix}_t0_t{visit_idx}")
    return x


def copy_optional_arrays(cache: Any, skip: set[str]) -> dict[str, np.ndarray]:
    copied: dict[str, np.ndarray] = {}
    for key in cache.files:
        if key in skip:
            continue
        copied[key] = cache[key]
    return copied


def main() -> None:
    args = parse_args()
    if args.output_cache.exists() and not args.overwrite:
        raise FileExistsError(f"Output cache already exists: {args.output_cache}")

    base = np.load(args.base_cache, allow_pickle=False)
    x_visit = base["x_visit"].astype(np.float32)
    y = base["y"].astype(np.int64)
    patient_ids = base["patient_ids"].astype(str)
    feature_names = base["feature_names"].astype(str)

    nact_by_id = load_nact_rows(args.mri_nact_csv)
    nact = np.full((x_visit.shape[0], 4, len(NACT_FEATURE_NAMES)), np.nan, dtype=np.float32)
    matched = 0
    for idx, patient_id in enumerate(patient_ids.tolist()):
        row = nact_by_id.get(patient_id)
        if row is None:
            continue
        nact[idx] = nact_visit_matrix(row)
        matched += 1

    output_x = np.concatenate([x_visit, nact], axis=2).astype(np.float32)
    output_feature_names = np.concatenate([feature_names, np.asarray(NACT_FEATURE_NAMES)])
    payload = {
        "x_visit": output_x,
        "y": y,
        "patient_ids": patient_ids,
        "feature_names": output_feature_names,
        **copy_optional_arrays(base, {"x_visit", "y", "patient_ids", "feature_names"}),
        "source_response_cache": np.asarray(str(args.base_cache)),
        "mri_nact_source_csv": np.asarray(str(args.mri_nact_csv)),
        "mri_nact_matched_patient_count": np.asarray(matched, dtype=np.int64),
        "mri_nact_feature_names": np.asarray(NACT_FEATURE_NAMES),
    }
    args.output_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_cache, **payload)
    finite_fraction = float(np.isfinite(nact).mean())
    print(
        f"wrote {args.output_cache} with x_visit={output_x.shape}, "
        f"nact_matched={matched}/{len(patient_ids)}, nact_finite_fraction={finite_fraction:.4f}"
    )


if __name__ == "__main__":
    main()

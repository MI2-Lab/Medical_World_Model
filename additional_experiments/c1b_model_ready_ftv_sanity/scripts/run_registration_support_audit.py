#!/usr/bin/env python3
"""Post-hoc physical-support audit for frozen image-only transforms.

Localization supports are deliberately absent from registration fitting.  This
script opens them only after all transforms exist, to compare C1B-H and C1B-R
exact source-domain containment, retention, and apparent lesion-centroid
displacement.  Failed registrations use the pre-frozen identity/header
fallback and remain counted as registration failures.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in os.sys.path:
    os.sys.path.insert(0, str(SRC))

from c1b_sanity.geometry import (  # noqa: E402
    audit_support_containment,
    load_nifti_ras,
    make_c1b_grid,
    support_bbox_center_ras,
    support_centroid_ras,
)


VISITS = ("T0", "T1", "T2", "T3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=ROOT / "manifests/model_input_inventory.private.csv",
    )
    parser.add_argument(
        "--registration-pairs",
        type=Path,
        default=ROOT / "metrics/registration_sensitivity_pairs.private.csv",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def atomic_text(path: Path, text: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def truth(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def matrix_from_row(row: pd.Series) -> np.ndarray:
    return np.asarray(
        [
            [row[f"source_to_anchor_ras_{i}{j}"] for j in range(4)]
            for i in range(4)
        ],
        dtype=np.float64,
    )


def transformed_point(matrix: np.ndarray, point: tuple[float, float, float]) -> np.ndarray:
    homogeneous = np.asarray((*point, 1.0), dtype=np.float64)
    return (np.asarray(matrix, dtype=np.float64) @ homogeneous)[:3]


def summarize(values: pd.Series) -> dict[str, float | int]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    return {
        "n": int(len(x)),
        "minimum": float(np.min(x)),
        "q05": float(np.quantile(x, 0.05)),
        "median": float(np.median(x)),
        "q95": float(np.quantile(x, 0.95)),
        "maximum": float(np.max(x)),
    }


def main() -> None:
    args = parse_args()
    inventory = pd.read_csv(args.inventory)
    formal = inventory.loc[inventory["formal_ftv_overlap"].astype(bool)].copy()
    if len(formal) != 1500 or formal.duplicated(["patient_id", "visit"]).any():
        raise ValueError("Formal support inventory must contain 375x4 unique rows")
    pairs = pd.read_csv(args.registration_pairs)
    if len(pairs) != 1125 or pairs.duplicated(["patient_id", "visit"]).any():
        raise ValueError("Registration output must contain exactly 375x3 unique pairs")
    if set(pairs["visit"].astype(str)) != {"T1", "T2", "T3"}:
        raise ValueError("Registration pairs contain an unexpected visit")
    pair_lookup = pairs.set_index(["patient_id", "visit"])

    records: list[dict[str, Any]] = []
    for patient_id, group in formal.groupby("patient_id", sort=True):
        if set(group["visit"]) != set(VISITS):
            raise ValueError("A formal patient lacks T0--T3 support")
        by_visit = group.set_index("visit")
        t0_support = load_nifti_ras(str(by_visit.loc["T0", "ftv_mask_nifti"]))
        grid = make_c1b_grid(support_bbox_center_ras(t0_support))
        t0_centroid = np.asarray(support_centroid_ras(t0_support), dtype=np.float64)
        for visit in VISITS:
            support = (
                t0_support
                if visit == "T0"
                else load_nifti_ras(str(by_visit.loc[visit, "ftv_mask_nifti"]))
            )
            centroid = support_centroid_ras(support)
            if visit == "T0":
                success = True
                failure_code = ""
                transform = np.eye(4, dtype=np.float64)
                fallback = False
            else:
                pair = pair_lookup.loc[(patient_id, visit)]
                success = truth(pair["success"])
                failure_code = "" if pd.isna(pair["failure_code"]) else str(pair["failure_code"])
                transform = matrix_from_row(pair) if success else np.eye(4, dtype=np.float64)
                fallback = not success
            h = audit_support_containment(support, grid)
            r = audit_support_containment(
                support,
                grid,
                source_to_anchor_ras=transform,
            )
            h_centroid = np.asarray(centroid, dtype=np.float64)
            r_centroid = transformed_point(transform, centroid)
            records.append(
                {
                    "patient_id": str(patient_id),
                    "visit": visit,
                    "registration_success": success,
                    "registration_failure_code": failure_code,
                    "registration_identity_fallback": fallback,
                    "h_exact_full_support_containment": h.exact_full_support_containment,
                    "r_exact_full_support_containment": r.exact_full_support_containment,
                    "h_physical_volume_retention": h.physical_volume_retention,
                    "r_physical_volume_retention": r.physical_volume_retention,
                    "h_minimum_margin_mm": h.minimum_margin_mm,
                    "r_minimum_margin_mm": r.minimum_margin_mm,
                    "h_target_boundary_touch": h.target_boundary_touch,
                    "r_target_boundary_touch": r.target_boundary_touch,
                    "source_boundary_touch": h.source_boundary_touch,
                    "h_lesion_centroid_displacement_from_t0_mm": float(
                        np.linalg.norm(h_centroid - t0_centroid)
                    ),
                    "r_lesion_centroid_displacement_from_t0_mm": float(
                        np.linalg.norm(r_centroid - t0_centroid)
                    ),
                }
            )

    frame = pd.DataFrame(records)
    if len(frame) != 1500:
        raise AssertionError("Physical support audit is incomplete")
    private_path = ROOT / "metrics/registration_support_patient_visit.private.csv"
    atomic_text(private_path, frame.to_csv(index=False), args.overwrite)

    followup = frame[frame["visit"].ne("T0")]
    h_exact = float(frame["h_exact_full_support_containment"].mean())
    r_exact = float(frame["r_exact_full_support_containment"].mean())
    h_q05 = float(frame["h_physical_volume_retention"].quantile(0.05))
    r_q05 = float(frame["r_physical_volume_retention"].quantile(0.05))
    summary = {
        "schema_version": 1,
        "formal_patients": 375,
        "formal_visits": 1500,
        "registration_pairs": 1125,
        "registration_success_pairs": int(followup["registration_success"].sum()),
        "registration_identity_fallback_pairs": int(
            followup["registration_identity_fallback"].sum()
        ),
        "c1b_h_exact_containment_rate": h_exact,
        "c1b_r_exact_containment_rate": r_exact,
        "c1b_r_minus_h_exact_containment_points": float(r_exact - h_exact),
        "c1b_h_ftv_retention": summarize(frame["h_physical_volume_retention"]),
        "c1b_r_ftv_retention": summarize(frame["r_physical_volume_retention"]),
        "c1b_h_ftv_retention_q05": h_q05,
        "c1b_r_ftv_retention_q05": r_q05,
        "r_exact_drop_gate_pass": bool(r_exact >= h_exact - 0.005),
        "r_retention_q05_gate_pass": bool(r_q05 >= 0.95),
        "h_lesion_centroid_displacement_followup_mm": summarize(
            followup["h_lesion_centroid_displacement_from_t0_mm"]
        ),
        "r_lesion_centroid_displacement_followup_mm": summarize(
            followup["r_lesion_centroid_displacement_from_t0_mm"]
        ),
        "registration_fitted_with_localization": False,
        "localization_opened_only_posthoc": True,
        "failed_transform_used": False,
        "contains_patient_identifiers": False,
    }
    atomic_text(
        ROOT / "metrics/registration_physical_support_summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        args.overwrite,
    )
    report = f"""# Registration physical-support sensitivity

Localization support只在全部image-only transform拟合完成后打开，用于独立QC；从未进入registration objective、ROI或初始化。

- C1B-H exact containment：{h_exact:.3%}；C1B-R（失败pair按预冻结identity/header fallback）为{r_exact:.3%}，差值{r_exact - h_exact:+.3%}。
- FTV retention Q05：H={h_q05:.3f}，R={r_q05:.3f}。
- registration失败/identity fallback：{int(followup['registration_identity_fallback'].sum())}/1125；失败transform从不采样。
- R相对H exact containment下降不超过0.5 point：{'PASS' if summary['r_exact_drop_gate_pass'] else 'FAIL'}；R retention Q05 >=0.95：{'PASS' if summary['r_retention_q05_gate_pass'] else 'FAIL'}。
- lesion centroid displacement仅为post-hoc apparent-motion audit，不用于拟合或选择phase；是否存在lesion-align pattern还需与独立whole-anatomy residual合并。
"""
    atomic_text(ROOT / "reports/registration_physical_support_audit.md", report, args.overwrite)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

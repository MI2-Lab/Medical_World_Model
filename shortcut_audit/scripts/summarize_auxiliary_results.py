#!/usr/bin/env python3
"""生成报告所需的概率变化与 donor matching 机器可读补充汇总。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


AUDIT_ROOT = Path(__file__).resolve().parents[1]
PERTURBATIONS = (
    "repeated_t0_c1_mri_only",
    "repeated_t0_c2_full_image_derived",
    "temporal_t1_t2_swap",
    "matched_followup_swap",
)


def _fresh(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"拒绝覆盖补充汇总：{path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, default=AUDIT_ROOT)
    parser.add_argument("--allow-summary", action="store_true")
    args = parser.parse_args()
    if not args.allow_summary:
        raise SystemExit("补充汇总未启动：必须显式添加 --allow-summary")
    root = args.audit_root.resolve()
    probability_output = root / "metrics" / "probability_change_summary.csv"
    donor_output = root / "metrics" / "donor_matching_summary.csv"
    donor_json = root / "metrics" / "donor_matching_summary.json"
    for path in (probability_output, donor_output, donor_json):
        _fresh(path)

    changes = pd.read_csv(root / "metrics" / "patient_probability_changes.csv")
    changes = changes.loc[changes["audit_condition"].isin(PERTURBATIONS)].copy()
    rows: list[dict[str, object]] = []
    for (condition, decision), group in changes.groupby(
        ["audit_condition", "decision_point"], sort=True, observed=True
    ):
        signed = group["probability_change"].to_numpy(float)
        absolute = group["absolute_probability_change"].to_numpy(float)
        if not np.isfinite(signed).all() or not np.isfinite(absolute).all():
            raise ValueError("patient probability change 含非有限值")
        rows.append(
            {
                "audit_condition": condition,
                "decision_point": decision,
                "n_patients": len(group),
                "mean_probability_change": signed.mean(),
                "mean_absolute_probability_change": absolute.mean(),
                "median_absolute_probability_change": np.median(absolute),
                "q25_absolute_probability_change": np.quantile(absolute, 0.25),
                "q75_absolute_probability_change": np.quantile(absolute, 0.75),
                "proportion_absolute_change_gt_0_05": np.mean(absolute > 0.05),
                "proportion_absolute_change_gt_0_10": np.mean(absolute > 0.10),
                "maximum_absolute_probability_change": absolute.max(),
                "aggregation": "one row per patient; donor repetitions averaged within patient",
            }
        )
    pd.DataFrame(rows).to_csv(probability_output, index=False)

    fold_rows: list[dict[str, object]] = []
    mappings: list[pd.DataFrame] = []
    for fold in range(5):
        directory = root / "donor_results" / f"fold_{fold:02d}"
        mapping = pd.read_csv(directory / "donor_mapping.csv")
        success = json.loads((directory / "matching_success.json").read_text())
        if (mapping["recipient_patient_id"] == mapping["donor_patient_id"]).any():
            raise ValueError(f"fold {fold} donor mapping 含 self-match")
        if not mapping["fold"].eq(fold).all():
            raise ValueError(f"fold {fold} donor mapping fold 漂移")
        mappings.append(mapping)
        fold_rows.append(
            {
                "scope": "fold",
                "fold": fold,
                "n_recipients": success["n_recipients"],
                "n_matched_recipients": success["n_matched_recipients"],
                "n_unmatched_recipients": success["n_unmatched_recipients"],
                "n_full_10_donor_recipients": success["n_full_recipients"],
                "n_partial_recipients": success["n_partial_recipients"],
                "n_pairs": len(mapping),
                "mean_donors_per_matched_recipient": success["mean_donors_per_recipient"],
                "matching_success_rate": success["success_rate"],
                "full_10_donor_match_rate": success["full_match_rate"],
                "relaxed_pair_rate": success["any_relaxed_pair_rate"],
                "subtype_match_rate": mapping["subtype_match"].astype(float).mean(),
                "treatment_family_match_rate": mapping["treatment_family_match"].astype(float).mean(),
                "visit_compatibility_rate": mapping["visit_availability_compatible"].astype(float).mean(),
                "mammaprint_match_rate": 1.0 - mapping["mammaprint_mismatch"].astype(float).mean(),
                "mean_volume_distance_z": mapping["volume_distance_z"].astype(float).mean(),
                "mean_age_distance_z": mapping["age_distance_z"].astype(float).mean(),
            }
        )
    all_mapping = pd.concat(mappings, ignore_index=True)
    n_recipients = int(sum(row["n_recipients"] for row in fold_rows))
    n_matched = int(sum(row["n_matched_recipients"] for row in fold_rows))
    pooled = {
        "scope": "pooled",
        "fold": "ALL",
        "n_recipients": n_recipients,
        "n_matched_recipients": n_matched,
        "n_unmatched_recipients": n_recipients - n_matched,
        "n_full_10_donor_recipients": int(
            sum(row["n_full_10_donor_recipients"] for row in fold_rows)
        ),
        "n_partial_recipients": int(sum(row["n_partial_recipients"] for row in fold_rows)),
        "n_pairs": len(all_mapping),
        "mean_donors_per_matched_recipient": len(all_mapping) / n_matched,
        "matching_success_rate": n_matched / n_recipients,
        "full_10_donor_match_rate": sum(
            row["n_full_10_donor_recipients"] for row in fold_rows
        )
        / n_recipients,
        "relaxed_pair_rate": all_mapping["matching_relaxed"].astype(float).mean(),
        "subtype_match_rate": all_mapping["subtype_match"].astype(float).mean(),
        "treatment_family_match_rate": all_mapping["treatment_family_match"].astype(float).mean(),
        "visit_compatibility_rate": all_mapping[
            "visit_availability_compatible"
        ].astype(float).mean(),
        "mammaprint_match_rate": 1.0
        - all_mapping["mammaprint_mismatch"].astype(float).mean(),
        "mean_volume_distance_z": all_mapping["volume_distance_z"].astype(float).mean(),
        "mean_age_distance_z": all_mapping["age_distance_z"].astype(float).mean(),
    }
    pd.DataFrame([*fold_rows, pooled]).to_csv(donor_output, index=False)
    donor_json.write_text(
        json.dumps(
            {
                **pooled,
                "matching_policy": "strict HR/HER2 subtype + treatment family + visit compatibility",
                "outcome_used_for_matching": False,
                "baseline_volume_unit": "cropped ROI voxel count",
                "matching_seed": 1729,
                "requested_donors_per_recipient": 10,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(probability_output)
    print(donor_output)
    print(donor_json)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Privacy-safe real-input smoke for Stage A loading and cache reconstruction."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "scripts"))

from run_stage_a import audit_patient, load_inputs  # noqa: E402


def required_root(variable: str) -> Path:
    value = os.environ.get(variable)
    if not value:
        raise SystemExit(f"missing required environment variable: {variable}")
    return Path(value).expanduser().resolve(strict=True)


def main() -> None:
    preprocessed_root = required_root("DGRS_DATA_ROOT") / "I-SPY2"
    raw_root = required_root("ISPY2_RAW_ROOT")
    config_path = EXPERIMENT_ROOT / "configs" / "stage_a.json"
    overlap_path = (
        REPO_ROOT
        / "additional_experiments"
        / "radiomics_next_change"
        / "data_audit"
        / "radiomics_patient_overlap.csv"
    )
    workbook_path = raw_root / "Multi-feature-MRI-NACT-Data.xlsx"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    inputs = SimpleNamespace(overlap=overlap_path, workbook=workbook_path)
    matched, _ = load_inputs(inputs, config)
    crop_size = tuple(config["input_contract"]["crop_size_zyx"])
    radius = int(config["input_contract"]["legacy_origin_search_radius_vox"])

    # Deterministic first row; no identifier or path is emitted.
    row = next(matched.itertuples(index=False))
    records = audit_patient(row, preprocessed_root, crop_size, radius)
    if len(records) != 4:
        raise AssertionError(f"expected four visits, got {len(records)}")
    if not all(record["full_support_voxels"] > 0 for record in records):
        raise AssertionError("real smoke encountered empty full support")
    if not all(0.0 <= record["containment_ratio"] <= 1.0 for record in records):
        raise AssertionError("containment ratio outside [0, 1]")
    if not any(record["origin_exact_match"] for record in records):
        raise AssertionError("real smoke found no exactly reconstructed visit")

    print(
        json.dumps(
            {
                "status": "PASS",
                "patient_count": 1,
                "visit_count": len(records),
                "exact_origin_visits": sum(
                    bool(record["origin_exact_match"]) for record in records
                ),
                "nonempty_full_support_visits": sum(
                    record["full_support_voxels"] > 0 for record in records
                ),
                "patient_identifier_emitted": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

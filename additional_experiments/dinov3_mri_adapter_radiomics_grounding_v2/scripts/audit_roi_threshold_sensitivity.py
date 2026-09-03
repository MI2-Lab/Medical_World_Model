#!/usr/bin/env python3
"""Outcome-blind ROI threshold sensitivity; cannot override the frozen NO-GO."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dinov3_rg.contracts import VISITS, atomic_json, private_patient_token  # noqa: E402
from dinov3_rg.radiomics import ftv_wide  # noqa: E402
from dinov3_rg.security import RepresentationReadSentinel  # noqa: E402


def main() -> None:
    RepresentationReadSentinel().install()
    ids = tuple(sorted(ftv_wide()["patient_id"].astype(str)))
    voxels, slices = [], []
    for patient_id in ids:
        path = ROOT / f"cache/radiomics_rois/{private_patient_token(patient_id)}.private.npz"
        with np.load(path, allow_pickle=False) as payload:
            voxels.append(np.asarray(payload["roi_voxels"], dtype=np.int64))
            slices.append(np.asarray(payload["roi_axial_slices"], dtype=np.int16))
    voxel = np.stack(voxels)
    axial = np.stack(slices)
    grid = {}
    for minimum_voxels in (64, 56, 48, 40, 35, 32, 24, 16):
        for minimum_slices in (3, 2):
            coverage = ((voxel >= minimum_voxels) & (axial >= minimum_slices)).mean(axis=0)
            grid[f"voxels_{minimum_voxels}_slices_{minimum_slices}"] = {
                visit: float(coverage[index]) for index, visit in enumerate(VISITS)
            }
    thresholds = {"T0": 0.90, "T1": 0.90, "T2": 0.90, "T3": 0.70}
    maximum_voxel_threshold = {}
    for index, visit in enumerate(VISITS):
        passing = [
            value for value in range(1, 65)
            if float(((voxel[:, index] >= value) & (axial[:, index] >= 3)).mean()) >= thresholds[visit]
        ]
        maximum_voxel_threshold[visit] = max(passing) if passing else None
    current_invalid_t2 = ~((voxel[:, 2] >= 64) & (axial[:, 2] >= 3))
    payload = {
        "schema_version": 1,
        "status": "DESCRIPTIVE_ONLY",
        "cannot_override_current_stage_a_no_go": True,
        "outcome_fields_read": [],
        "clinical_fields_read": [],
        "coverage_grid": grid,
        "maximum_voxel_threshold_passing_with_three_slices": maximum_voxel_threshold,
        "t2_current_invalid_visits": int(current_invalid_t2.sum()),
        "t2_invalid_voxel_bins": {
            "0_15": int(((voxel[:, 2] < 16) & current_invalid_t2).sum()),
            "16_31": int(((voxel[:, 2] >= 16) & (voxel[:, 2] < 32) & current_invalid_t2).sum()),
            "32_47": int(((voxel[:, 2] >= 32) & (voxel[:, 2] < 48) & current_invalid_t2).sum()),
            "48_63": int(((voxel[:, 2] >= 48) & (voxel[:, 2] < 64) & current_invalid_t2).sum()),
            "at_least_64_but_slice_fail": int(((voxel[:, 2] >= 64) & current_invalid_t2).sum()),
        },
        "interpretation": (
            "Changing the axial-slice threshold from 3 to 2 barely changes T2 coverage; "
            "the binding constraint is ROI voxel count. A minimum of 35 voxels is the "
            "largest threshold that reaches 90% at T2, and this would require a new protocol."
        ),
    }
    atomic_json(ROOT / "metrics/roi_threshold_sensitivity.json", payload)
    print(payload)


if __name__ == "__main__":
    main()

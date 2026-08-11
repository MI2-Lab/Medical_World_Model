#!/usr/bin/env python3
"""Run the header-only audit on one real singular-affine DCE visit.

The emitted JSON deliberately contains no patient/visit identifier or path.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any
import warnings


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from observable_crop.dicom_geometry import audit_dicom_geometry  # noqa: E402
from observable_crop.nifti import (  # noqa: E402
    NiftiGeometry,
    affine_max_corner_disagreement_mm,
    read_nifti_geometry,
)


def _discover_real_singular_visit(
    preprocessed_root: Path,
) -> tuple[dict[str, Any], NiftiGeometry, NiftiGeometry]:
    """Prefer a singular case whose qform also disagrees with the mask grid."""

    first_singular: (
        tuple[dict[str, Any], NiftiGeometry, NiftiGeometry] | None
    ) = None
    for manifest_path in sorted(preprocessed_root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for visit in manifest.get("visits", []):
            required = (
                "dce_nifti",
                "ftv_mask_nifti",
                "raw_dce_series",
            )
            if any(not visit.get(key) for key in required):
                continue
            dce = read_nifti_geometry(visit["dce_nifti"])
            if dce.sform_valid or dce.sform_failure_reason != "SFORM_SINGULAR":
                continue
            mask = read_nifti_geometry(visit["ftv_mask_nifti"])
            if not mask.sform_valid or mask.sform is None:
                continue
            candidate = (visit, dce, mask)
            if first_singular is None:
                first_singular = candidate
            if dce.qform_valid and dce.qform is not None:
                disagreement = affine_max_corner_disagreement_mm(
                    dce.qform,
                    mask.sform,
                    tuple(int(value) for value in dce.shape[:3]),
                )
                if disagreement > 0.1:
                    return candidate
    if first_singular is None:
        raise RuntimeError("REAL_SINGULAR_VISIT_NOT_FOUND")
    return first_singular


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Privacy-safe real-data smoke for header-only DICOM geometry"
    )
    configured_root = os.environ.get("ISPY2_PREPROCESSED_ROOT")
    parser.add_argument(
        "--preprocessed-root",
        type=Path,
        default=Path(configured_root) if configured_root else None,
        required=configured_root is None,
        help=(
            "I-SPY2 preprocessed patient-root directory; alternatively set "
            "ISPY2_PREPROCESSED_ROOT (the value is never echoed)"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    warnings.filterwarnings("ignore", module="pydicom")
    try:
        visit, dce, mask = _discover_real_singular_visit(
            args.preprocessed_root.expanduser().resolve(strict=True)
        )
        audit = audit_dicom_geometry(
            visit["raw_dce_series"],
            expected_shape_xyz_t=tuple(int(value) for value in dce.shape[:4]),
            expected_spacing_xyz_mm=dce.spacing_xyz_mm,
            mask_affine_ras=mask.sform,
            mask_shape_xyz=tuple(int(value) for value in mask.shape[:3]),
            dce_sform_ras=dce.sform,
            dce_qform_ras=dce.qform,
        )
        checks = {
            "real_singular_sform": not audit.dce_sform_valid,
            "headers_only": audit.headers_only and not audit.pixel_data_read,
            "all_headers_readable": audit.file_count == audit.readable_header_count,
            "rows_columns_match": (
                audit.dicom_shape_xyz == tuple(int(value) for value in dce.shape[:3])
            ),
            "iop_consistent_orthonormal": (
                audit.iop_orthonormal
                and audit.iop_max_abs_delta is not None
                and audit.iop_max_abs_delta <= 1e-5
            ),
            "pixel_spacing_consistent": (
                audit.pixel_spacing_max_abs_delta_mm is not None
                and audit.pixel_spacing_max_abs_delta_mm <= 1e-5
            ),
            "slice_grid_complete_regular": (
                audit.slice_count == dce.shape[2]
                and audit.slice_spacing_max_deviation_mm is not None
                and audit.slice_spacing_max_deviation_mm <= 1e-3
            ),
            "temporal_position_complete": audit.temporal_position.complete,
            "acquisition_time_complete": audit.acquisition_time.complete,
            "mask_sform_matches_dicom": audit.mask_geometry_consistent,
            "safe_decision_boundary": audit.decision
            in {
                "TRUST_DCE_QFORM",
                "MASK_SFORM_GEOMETRY_CANDIDATE",
            },
            "pixel_order_not_claimed": not audit.pixel_order_verified,
        }
        passed = bool(all(checks.values()) and audit.audit_pass)
        public = {
            "smoke": "PASS" if passed else "FAIL",
            "case": "one_real_singular_dce_visit",
            "privacy": "no_patient_visit_or_path_emitted",
            "files_scanned": audit.file_count,
            "shape_xyz_t": list(audit.expected_shape_xyz_t or ()),
            "slice_count": audit.slice_count,
            "timepoint_count": audit.temporal_position.group_count,
            "mask_center_corner_hausdorff_mm": audit.mask_center_corner_hausdorff_mm,
            "mask_footprint_corner_hausdorff_mm": (
                audit.mask_footprint_corner_hausdorff_mm
            ),
            "decision": audit.decision,
            "geometry_auto_repairable": audit.geometry_auto_repairable,
            "header_only_safe": audit.header_only_safe,
            "recommended_action": audit.recommended_action,
            "checks": checks,
        }
        print(json.dumps(public, indent=2, sort_keys=True))
        return 0 if passed else 1
    except Exception as exc:  # keep errors privacy-safe as well
        print(
            json.dumps(
                {
                    "smoke": "FAIL",
                    "case": "one_real_singular_dce_visit",
                    "privacy": "no_patient_visit_or_path_emitted",
                    "error_type": type(exc).__name__,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

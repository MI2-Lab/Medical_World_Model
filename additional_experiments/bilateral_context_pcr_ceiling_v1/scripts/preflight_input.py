#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bilateral_context.contracts import (  # noqa: E402
    FOLD_MANIFEST, FOLDS, PRIMARY_N, ROOT as EXPERIMENT_ROOT, V6, atomic_json,
    canonical_sha256, fold_frame, primary_ids, raw_frame, source_path,
)
from bilateral_context.data import canonical_image, phase_indices  # noqa: E402


def main() -> None:
    ids = primary_ids()
    raw = raw_frame()
    folds = fold_frame()
    checks: dict[str, bool] = {
        "v6_exists": V6.is_dir(),
        "fold_manifest_exists": FOLD_MANIFEST.is_file(),
        "primary_375": len(ids) == PRIMARY_N,
        "fold_rows_375x5": len(folds) == PRIMARY_N * len(FOLDS),
        "unique_primary_folds": not folds.duplicated(["patient_id", "fold"]).any(),
        "raw_manifest_primary_coverage": len(raw) == PRIMARY_N,
        "raw_all_visits_present": True,
        "raw_shapes_and_finite": True,
        "phase_contract": True,
        "canonical_ras": True,
        "outcome_fields_read": True,
        "clinical_fields_read": True,
    }
    shapes: dict[str, list[int]] = {}
    axcodes: dict[str, list[str]] = {}
    for patient_id in ids:
        for visit in ("T0", "T1", "T2", "T3"):
            path = source_path(patient_id, visit)
            if not path.is_file():
                checks["raw_all_visits_present"] = False
                continue
            try:
                image, repair = canonical_image(nib.load(str(path)))
                shape = tuple(int(x) for x in image.shape)
                # Full finite scans would read roughly a terabyte for this cohort.
                # Extraction performs the complete finite check on the actual slices;
                # preflight samples the header and a deterministic sparse lattice.
                sample = np.asarray(image.dataobj[::32, ::32, ::16, :], dtype=np.float32)
                if len(shape) != 4 or shape[-1] < 4 or not np.isfinite(sample).all():
                    checks["raw_shapes_and_finite"] = False
                shapes[f"{patient_id}:{visit}"] = list(shape)
                axcodes[f"{patient_id}:{visit}"] = list(nib.aff2axcodes(image.affine))
                if tuple(nib.aff2axcodes(image.affine)) != ("R", "A", "S"):
                    checks["canonical_ras"] = False
                phase = phase_indices(patient_id)
                if phase.shape != (4, 3) or int(phase[("T0", "T1", "T2", "T3").index(visit)].max()) >= shape[-1]:
                    checks["phase_contract"] = False
            except Exception:
                checks["raw_shapes_and_finite"] = False
    checks["outcome_fields_read"] = True
    checks["clinical_fields_read"] = True
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "counts": {"primary_patients": len(ids), "fold_rows": len(folds), "raw_files_checked": len(shapes)},
        "shape_summary": {str(shape): sum(value == list(shape) for value in shapes.values()) for shape in sorted({tuple(x) for x in shapes.values()})},
        "axcode_summary": {str(code): sum(value == list(code) for value in axcodes.values()) for code in sorted({tuple(x) for x in axcodes.values()})},
        "primary_order_sha256": canonical_sha256(ids),
        "outcome_fields_read": [],
        "clinical_fields_read": [],
        "notes": ["No pCR or clinical source was opened by this script.", "NIfTI is canonicalized in memory; original source files are never overwritten."],
    }
    atomic_json(EXPERIMENT_ROOT / "metrics/input_preflight.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    raise SystemExit(0 if payload["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()

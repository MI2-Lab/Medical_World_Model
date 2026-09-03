#!/usr/bin/env python3
"""Resample FTV support and build private observable LOCAL ROIs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dinov3_rg.cache_io import load_c1b_manifest  # noqa: E402
from dinov3_rg.radiomics import (  # noqa: E402
    build_patient_roi, finalize_roi_gate, ftv_wide, load_support_inventory,
)
from dinov3_rg.security import RepresentationReadSentinel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "cache/radiomics_rois")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    RepresentationReadSentinel().install()
    entries = load_c1b_manifest()
    inventory = load_support_inventory()
    ftv = ftv_wide().set_index("patient_id")
    patient_ids = tuple(sorted(ftv.index.astype(str)))
    if args.limit is not None:
        patient_ids = patient_ids[: args.limit]
    for index, patient_id in enumerate(patient_ids, start=1):
        result = build_patient_roi(
            entries[patient_id],
            inventory.loc[inventory["patient_id"].eq(patient_id)],
            ftv.loc[patient_id],
            args.output_dir,
            overwrite=args.overwrite,
        )
        print({"patient": index, "total": len(patient_ids), **result}, flush=True)
    if args.limit is None:
        print(finalize_roi_gate(patient_ids, args.output_dir))
    else:
        print({"status": "SMOKE_COMPLETE", "patients": len(patient_ids)})


if __name__ == "__main__":
    main()

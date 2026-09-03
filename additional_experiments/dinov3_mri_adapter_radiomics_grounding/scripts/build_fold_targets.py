#!/usr/bin/env python3
"""Fit stability/residualization/PCA transforms separately in each outer fold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dinov3_rg.contracts import FOLDS  # noqa: E402
from dinov3_rg.data import load_fold_frame  # noqa: E402
from dinov3_rg.radiomics import ftv_wide  # noqa: E402
from dinov3_rg.security import RepresentationReadSentinel  # noqa: E402
from dinov3_rg.targets import build_fold_targets, load_raw_radiomics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radiomics-dir", type=Path, default=ROOT / "cache/radiomics_raw")
    parser.add_argument("--roi-dir", type=Path, default=ROOT / "cache/radiomics_rois")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "features/private/fold_targets")
    parser.add_argument("--fold", type=int, choices=FOLDS)
    args = parser.parse_args()
    RepresentationReadSentinel().install()
    completion = ROOT / "manifests/radiomics_raw_complete.json"
    if not completion.is_file() or json.loads(completion.read_text())["status"] != "COMPLETE":
        raise SystemExit("raw radiomics extraction is not complete")
    patient_ids = tuple(sorted(ftv_wide()["patient_id"].astype(str)))
    raw = load_raw_radiomics(patient_ids, args.radiomics_dir, args.roi_dir)
    folds = load_fold_frame()
    requested = FOLDS if args.fold is None else (args.fold,)
    for fold in requested:
        train_ids = tuple(
            sorted(folds.loc[folds["fold"].eq(fold) & folds["split"].eq("train"), "patient_id"].astype(str))
        )
        print(build_fold_targets(raw, fold, train_ids, args.output_dir))


if __name__ == "__main__":
    main()

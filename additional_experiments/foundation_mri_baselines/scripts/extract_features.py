#!/usr/bin/env python3
"""Extract one pre-test-frozen foundation representation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from foundation_mri.extraction import extract_features  # noqa: E402


DEFAULT_FOLD = Path(
    "/data/data/Preprocessed/I-SPY2/"
    "_matched_breastdcedl_t0_dicomrepair_rgb224_seed2026/"
    "matched_patient_cv_splits_seed2026.csv"
)
DEFAULT_CACHE = (
    REPO_ROOT
    / "additional_experiments"
    / "c1b_overlap_eligibility_ftv_stageb"
    / "manifests"
    / "stage_b_c1b_cache.private.csv"
)
DEFAULT_CHECKPOINTS = {
    "medicalnet_resnet50_3dseg8": (
        EXPERIMENT_ROOT / "checkpoints" / "medicalnet" / "resnet_50.pth"
    ),
    "dino_vitb16_imagenet1k": (
        EXPERIMENT_ROOT
        / "checkpoints"
        / "dino"
        / "dino_vitbase16_pretrain.pth"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(DEFAULT_CHECKPOINTS))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--fold-manifest", type=Path, default=DEFAULT_FOLD)
    parser.add_argument("--cache-manifest", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-root", type=Path, default=EXPERIMENT_ROOT / "features")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--dino-batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = extract_features(
        model_name=args.model,
        checkpoint=args.checkpoint or DEFAULT_CHECKPOINTS[args.model],
        fold_manifest=args.fold_manifest,
        cache_manifest=args.cache_manifest,
        output_root=args.output_root,
        device_name=args.device,
        precision=args.precision,
        dino_batch_size=args.dino_batch_size,
        limit=args.limit,
    )
    print(f"combined_feature_file={destination}")


if __name__ == "__main__":
    main()


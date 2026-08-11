#!/usr/bin/env python3
"""Extract the single fixed DINOv3 post-hoc representation.

The formal command is intentionally the empty argument vector.  Its snapshot
comes only from ``DINOV3_SNAPSHOT_DIR``; all other paths and runtime choices are
frozen defaults.  Non-empty CLI arguments are accepted only with ``--limit``
and therefore can create private smoke shards but never a combined formal NPZ.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from foundation_mri_dinov3.extraction import (  # noqa: E402
    SNAPSHOT_ENV,
    extract_features,
)


DEFAULT_FOLD = Path(
    "/data/data/Preprocessed/I-SPY2/"
    "_matched_breastdcedl_t0_dicomrepair_rgb224_seed2026/"
    "matched_patient_cv_splits_seed2026.csv"
)
DEFAULT_CACHE = (
    REPOSITORY_ROOT
    / "additional_experiments"
    / "c1b_overlap_eligibility_ftv_stageb"
    / "manifests"
    / "stage_b_c1b_cache.private.csv"
)
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "features"
DEFAULT_DEVICE = "cuda:0"
DEFAULT_PRECISION = "bf16"
DEFAULT_BATCH_SIZE = 64


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot",
        type=Path,
        help=(
            "local hash-gated snapshot for smoke only; formal uses " f"{SNAPSHOT_ENV}"
        ),
    )
    parser.add_argument("--fold-manifest", type=Path, default=DEFAULT_FOLD)
    parser.add_argument("--cache-manifest", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument(
        "--precision", choices=("bf16", "fp32"), default=DEFAULT_PRECISION
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--limit",
        type=int,
        help="smoke-only canonical prefix in 1..807; omit for formal 808 run",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    if args.limit is None and raw_argv:
        raise ValueError(
            "formal extraction is frozen to an empty argument vector; "
            "non-empty arguments require --limit and are smoke-only"
        )
    destination = extract_features(
        snapshot_dir=args.snapshot,
        fold_manifest=args.fold_manifest,
        cache_manifest=args.cache_manifest,
        output_root=args.output_root,
        device_name=args.device,
        precision=args.precision,
        batch_size=args.batch_size,
        limit=args.limit,
    )
    # Do not put local cache/workspace paths into captured logs.
    print(
        "combined_feature_file="
        + (destination.name if destination is not None else "None")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corejepa.config import load_config
from corejepa.data.imaging import load_phase_metadata
from corejepa.data.response_targets import build_response_feature_cache
from corejepa.training.runner import load_experiment_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build raw pCR-free response-feature cache.")
    parser.add_argument("--config", default="configs/paper_v1.yaml")
    parser.add_argument("--output", help="Override data.response_cache (useful for a local smoke test).")
    parser.add_argument("--cohort", choices=("all", "ispy2", "ispy1"), default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.output:
        config.data.response_cache = args.output
    records, _ = load_experiment_records(config)
    if args.cohort != "all":
        records = [record for record in records if record.cohort.lower() == args.cohort]
    if args.limit is not None:
        records = records[: args.limit]
    path = build_response_feature_cache(
        records,
        config.data.response_cache,
        config.data.auto_roi_fallback,
        config.data.legacy_empty_ftv_full_field,
        load_phase_metadata(config.data.breastdcedl_metadata_csv),
        config.data.response_phase_policy,
        overwrite=args.overwrite,
    )
    print(path)


if __name__ == "__main__":
    main()

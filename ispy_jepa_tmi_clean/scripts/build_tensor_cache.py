#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corejepa.config import load_config
from corejepa.data.imaging import build_patient_tensor, load_phase_metadata
from corejepa.training.runner import load_experiment_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build auditable DCE8 and q_t patient caches.")
    parser.add_argument("--config", default="configs/paper_v1.yaml")
    parser.add_argument("--cache-dir", help="Override data.tensor_cache (useful for a local smoke test).")
    parser.add_argument("--cohort", choices=("all", "ispy2", "ispy1"), default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.cache_dir:
        config.data.tensor_cache = args.cache_dir
    records, _ = load_experiment_records(config)
    if args.cohort != "all":
        records = [record for record in records if record.cohort.lower() == args.cohort]
    if args.limit is not None:
        records = records[: args.limit]
    metadata = load_phase_metadata(config.data.breastdcedl_metadata_csv)
    for index, record in enumerate(records, start=1):
        path = build_patient_tensor(
            record,
            config.data.tensor_cache,
            config.data.crop_size,
            config.data.phase_policy,
            metadata.get(record.patient_id),
            config.data.auto_roi_fallback,
            config.data.min_roi_capture,
            config.data.legacy_empty_ftv_full_field,
            overwrite=args.overwrite,
        )
        print(f"[{index}/{len(records)}] {record.patient_id}: {path}")


if __name__ == "__main__":
    main()

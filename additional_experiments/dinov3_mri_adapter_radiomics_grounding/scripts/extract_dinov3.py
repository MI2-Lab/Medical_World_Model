#!/usr/bin/env python3
"""Extract hash-bound DINOv3 `[4,7,32,2304]` private caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dinov3_rg.contracts import atomic_json  # noqa: E402
from dinov3_rg.cache_io import load_c1b_manifest  # noqa: E402
from dinov3_rg.extraction import (  # noqa: E402
    extract_cache_entry, finalize_extraction_manifest, load_frozen_dino,
)
from dinov3_rg.security import RepresentationReadSentinel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "cache/dinov3_summaries")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-source-hash", action="store_true")
    args = parser.parse_args()
    sentinel = RepresentationReadSentinel().install()
    entries = load_c1b_manifest()
    patient_ids = tuple(sorted(entries))
    if args.limit is not None:
        patient_ids = patient_ids[: args.limit]
    frozen = load_frozen_dino(args.device)
    progress = ROOT / "logs/private/dinov3_extraction.private.json"
    completed: list[str] = []
    for index, patient_id in enumerate(patient_ids, start=1):
        result = extract_cache_entry(
            frozen, entries[patient_id], args.output_dir, batch_size=args.batch_size,
            verify_source_hash=not args.skip_source_hash, overwrite=args.overwrite,
        )
        completed.append(patient_id)
        atomic_json(progress, {"completed": len(completed), "requested": len(patient_ids), "last_status": result["status"]})
        print({"patient": index, "total": len(patient_ids), "status": result["status"]}, flush=True)
    if args.limit is None:
        print(finalize_extraction_manifest(patient_ids, args.output_dir, frozen.contract_sha256))
    else:
        print({"status": "SMOKE_COMPLETE", "patients": len(patient_ids)})


if __name__ == "__main__":
    main()

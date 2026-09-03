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
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-source-hash", action="store_true")
    args = parser.parse_args()
    sentinel = RepresentationReadSentinel().install()
    target_gate = ROOT / "target_feasibility.json"
    if not target_gate.is_file() or json.loads(target_gate.read_text(encoding="utf-8")).get("status") != "PASS":
        raise SystemExit("DINO extraction is gated on passing V2 target feasibility")
    entries = load_c1b_manifest()
    all_patient_ids = tuple(sorted(entries))
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("shard contract requires num-shards >= 1 and 0 <= shard-index < num-shards")
    frozen = load_frozen_dino(args.device)
    if args.finalize_only:
        if args.limit is not None or args.num_shards != 1 or args.shard_index != 0:
            raise SystemExit("finalize-only cannot be combined with limit or sharding")
        print(finalize_extraction_manifest(all_patient_ids, args.output_dir, frozen.contract_sha256))
        return
    patient_ids = all_patient_ids[args.shard_index :: args.num_shards]
    if args.limit is not None:
        patient_ids = patient_ids[: args.limit]
    progress = ROOT / f"logs/private/dinov3_extraction_shard{args.shard_index}.private.json"
    completed: list[str] = []
    for index, patient_id in enumerate(patient_ids, start=1):
        result = extract_cache_entry(
            frozen, entries[patient_id], args.output_dir, batch_size=args.batch_size,
            verify_source_hash=not args.skip_source_hash, overwrite=args.overwrite,
        )
        completed.append(patient_id)
        atomic_json(progress, {"completed": len(completed), "requested": len(patient_ids), "last_status": result["status"]})
        print({"patient": index, "total": len(patient_ids), "status": result["status"]}, flush=True)
    if args.limit is None and args.num_shards == 1:
        print(finalize_extraction_manifest(patient_ids, args.output_dir, frozen.contract_sha256))
    else:
        status = "SMOKE_COMPLETE" if args.limit is not None else "SHARD_COMPLETE"
        print({
            "status": status,
            "patients": len(patient_ids),
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
        })


if __name__ == "__main__":
    main()

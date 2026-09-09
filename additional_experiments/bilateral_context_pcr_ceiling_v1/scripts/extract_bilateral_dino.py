#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bilateral_context.contracts import (  # noqa: E402
    SUMMARY_SHAPE, SPATIAL_SHAPE, atomic_json, canonical_sha256, primary_ids,
    private_token, sha256_file,
)
from bilateral_context.dino import extract_patient, load_frozen_dino, write_patient  # noqa: E402


def validate_cache(path: Path, patient_id: str) -> None:
    with np.load(path, allow_pickle=False) as payload:
        expected = {"patient_token", "global_summary", "spatial_tokens", "source_digest", "contract_sha256"}
        if set(payload.files) != expected or str(payload["patient_token"].item()) != private_token(patient_id):
            raise ValueError(f"cache identity contract failed: {path}")
        if tuple(payload["global_summary"].shape) != SUMMARY_SHAPE or tuple(payload["spatial_tokens"].shape) != SPATIAL_SHAPE:
            raise ValueError(f"cache shape contract failed: {path}")
        if payload["global_summary"].dtype != np.float16 or payload["spatial_tokens"].dtype != np.float16:
            raise ValueError(f"cache dtype contract failed: {path}")
        if not np.isfinite(payload["global_summary"]).all() or not np.isfinite(payload["spatial_tokens"]).all():
            raise ValueError(f"cache finite contract failed: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    ids = primary_ids()
    if args.shards < 1 or not 0 <= args.shard_index < args.shards:
        raise SystemExit("invalid shard arguments")
    selected = ids[args.shard_index :: args.shards]
    model, device, dino_config = load_frozen_dino(args.device)
    contract = canonical_sha256({
        "experiment": "bilateral_context_pcr_ceiling_v1",
        "repository_id": dino_config["repository_id"],
        "revision": dino_config["revision"],
        "register_tokens_excluded": 4,
        "summary_shape": list(SUMMARY_SHAPE),
        "spatial_shape": list(SPATIAL_SHAPE),
        "input": "canonical RAS bilateral DCE7, two halves, 32 central-80pct axial slices",
    })
    output_dir = ROOT / "cache/bilateral_dino"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, patient_id in enumerate(selected, start=1):
        destination = output_dir / f"{private_token(patient_id)}.private.npz"
        if destination.exists() and not args.overwrite:
            validate_cache(destination, patient_id)
            rows.append({"patient_token": private_token(patient_id), "status": "REUSED", "sha256": sha256_file(destination)})
            continue
        global_summary, spatial_tokens, provenance = extract_patient(model, device, patient_id, batch_size=args.batch_size)
        digest = write_patient(destination, patient_id, global_summary, spatial_tokens, provenance, contract)
        validate_cache(destination, patient_id)
        rows.append({"patient_token": private_token(patient_id), "status": "WRITTEN", "sha256": digest})
        if index % 5 == 0 or index == len(selected):
            print(json.dumps({"shard": args.shard_index, "completed": index, "total": len(selected)}), flush=True)
    manifest = {
        "schema_version": 1,
        "status": "SHARD_COMPLETE",
        "shard_index": args.shard_index,
        "shards": args.shards,
        "patients": len(selected),
        "contract_sha256": contract,
        "cache_shape_global": list(SUMMARY_SHAPE),
        "cache_shape_spatial": list(SPATIAL_SHAPE),
        "cache_dtype": "float16",
        "patient_order_sha256": canonical_sha256([private_token(x) for x in selected]),
        "members": rows,
        "outcome_fields_read": [],
        "clinical_fields_read": [],
    }
    atomic_json(ROOT / f"manifests/cache_shard_{args.shard_index}.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != "members"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

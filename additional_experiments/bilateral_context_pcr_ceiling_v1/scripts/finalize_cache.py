#!/usr/bin/env python3
from __future__ import annotations

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


def validate(path: Path, patient_id: str) -> dict[str, str]:
    with np.load(path, allow_pickle=False) as payload:
        expected = {"patient_token", "global_summary", "spatial_tokens", "source_digest", "contract_sha256"}
        if set(payload.files) != expected:
            raise ValueError(f"unexpected cache keys: {path}")
        if str(payload["patient_token"].item()) != private_token(patient_id):
            raise ValueError(f"cache token mismatch: {path}")
        if tuple(payload["global_summary"].shape) != SUMMARY_SHAPE:
            raise ValueError(f"global shape mismatch: {path}")
        if tuple(payload["spatial_tokens"].shape) != SPATIAL_SHAPE:
            raise ValueError(f"spatial shape mismatch: {path}")
        if payload["global_summary"].dtype != np.float16 or payload["spatial_tokens"].dtype != np.float16:
            raise ValueError(f"cache dtype mismatch: {path}")
        if not np.isfinite(payload["global_summary"]).all() or not np.isfinite(payload["spatial_tokens"]).all():
            raise ValueError(f"cache is non-finite: {path}")
        return {"source_digest": str(payload["source_digest"].item()), "contract_sha256": str(payload["contract_sha256"].item())}


def main() -> None:
    ids = primary_ids()
    cache_dir = ROOT / "cache/bilateral_dino"
    rows = []
    for patient_id in ids:
        path = cache_dir / f"{private_token(patient_id)}.private.npz"
        if not path.is_file():
            raise SystemExit(f"missing cache for primary patient token {private_token(patient_id)}")
        details = validate(path, patient_id)
        rows.append({"patient_token": private_token(patient_id), "sha256": sha256_file(path), **details})
    contracts = sorted({row["contract_sha256"] for row in rows})
    if len(contracts) != 1:
        raise SystemExit("cache contract hashes are inconsistent")
    manifest = {
        "schema_version": 1,
        "status": "COMPLETE",
        "patients": len(rows),
        "contract_sha256": contracts[0],
        "global_shape": list(SUMMARY_SHAPE),
        "spatial_shape": list(SPATIAL_SHAPE),
        "dtype": "float16",
        "patient_tokens_sha256": canonical_sha256([row["patient_token"] for row in rows]),
        "cache_sha256": canonical_sha256([row["sha256"] for row in rows]),
        "outcome_fields_read": [],
        "clinical_fields_read": [],
        "members": rows,
    }
    atomic_json(ROOT / "manifests/cache_manifest.private.json", manifest)
    atomic_json(ROOT / "manifests/private_sha_manifest.json", {"status": "COMPLETE", "cache_sha256": manifest["cache_sha256"], "contract_sha256": manifest["contract_sha256"], "members": rows})
    public = {key: value for key, value in manifest.items() if key != "members"}
    atomic_json(ROOT / "metrics/cache_check.json", public)
    print(json.dumps(public, indent=2))


if __name__ == "__main__":
    main()

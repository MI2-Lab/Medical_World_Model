#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bilateral_context.contracts import atomic_json, canonical_sha256, primary_ids, private_token  # noqa: E402


def main() -> None:
    input_lock = json.loads((ROOT / "INPUT_LOCK.json").read_text(encoding="utf-8"))
    cache_lock = json.loads((ROOT / "manifests/cache_manifest.private.json").read_text(encoding="utf-8"))
    expected = {private_token(pid) for pid in primary_ids()}
    members = cache_lock.get("members", [])
    actual = {str(row["patient_token"]) for row in members}
    if input_lock.get("status") != "LOCKED":
        raise SystemExit("INPUT_LOCK is not locked")
    if cache_lock.get("status") != "COMPLETE" or actual != expected or len(members) != 375:
        raise SystemExit("bilateral DINO cache is not complete for the locked primary cohort")
    payload = {
        "schema_version": 1,
        "status": "LOCKED",
        "pcr_open": True,
        "purpose": "exploratory V6 state rescue; no model training",
        "input_lock_sha256": canonical_sha256(input_lock),
        "cache_manifest_sha256": canonical_sha256(cache_lock),
        "cache_contract_sha256": cache_lock["contract_sha256"],
        "state_source": "V6 seed2026 S0/S1 frozen states",
        "outcome_fields_read": [],
        "clinical_fields_read": [],
        "public_patient_ids": False,
        "public_predictions": False,
        "public_private_paths": False,
    }
    atomic_json(ROOT / "EXPLORATORY_EVALUATION_LOCK.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

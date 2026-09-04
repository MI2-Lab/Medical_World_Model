#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent


def main() -> None:
    input_lock = json.loads((PARENT / "INPUT_LOCK.json").read_text(encoding="utf-8"))
    cache_check = json.loads((PARENT / "metrics/cache_check.json").read_text(encoding="utf-8"))
    protocol = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))
    if input_lock.get("status") != "LOCKED" or cache_check.get("status") != "COMPLETE" or cache_check.get("patients") != 375:
        raise SystemExit("parent input/cache locks are not complete")
    payload = {
        "schema_version": 1,
        "status": "LOCKED",
        "pcr_open": True,
        "parent_input_lock": "LOCKED",
        "parent_cache_status": "COMPLETE",
        "protocol_sha256": __import__("hashlib").sha256((ROOT / "configs/protocol.json").read_bytes()).hexdigest(),
        "no_adapter_training": True,
        "radiomics_is_comparator_only": True,
        "outcome_fields_read_before_lock": [],
        "clinical_fields_read_before_lock": [],
        "public_patient_ids": False,
        "public_predictions": False,
        "public_private_paths": False,
    }
    (ROOT / "FROZEN_READOUT_LOCK.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

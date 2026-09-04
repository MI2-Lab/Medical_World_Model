#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bilateral_context.contracts import ROOT as EXPERIMENT_ROOT, atomic_json, canonical_sha256, sha256_file  # noqa: E402


def main() -> None:
    preflight_path = EXPERIMENT_ROOT / "metrics/input_preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS":
        raise SystemExit("input preflight must pass before INPUT_LOCK")
    protocol = EXPERIMENT_ROOT / "configs/protocol.json"
    payload = {
        "schema_version": 1,
        "experiment": "bilateral_context_pcr_ceiling_v1",
        "status": "LOCKED",
        "protocol_sha256": sha256_file(protocol),
        "input_preflight_sha256": sha256_file(preflight_path),
        "input": {
            "source": "original bilateral DCE NIfTI; known singular sform repaired in memory from valid qform",
            "geometry_mutation": "source files never modified",
            "phase_selection": "locked C1B phase_indices",
            "views": "canonical RAS left/right half views with shared encoder",
        },
        "outcome_fields_read": [],
        "clinical_fields_read": [],
        "pcr_evaluator_open": False,
    }
    payload["lock_sha256"] = canonical_sha256(payload)
    atomic_json(EXPERIMENT_ROOT / "INPUT_LOCK.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

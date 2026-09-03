#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from raw_spatial_pcr.contracts import load_contract


def main() -> None:
    contract = load_contract()
    payload = {
        "status": "PASS",
        "experiment": contract.config["experiment"],
        "branch": contract.config["branch"],
        "config_sha256": contract.config_sha256,
        "lock_sha256": contract.lock_sha256,
        "seeds": list(contract.seeds),
        "folds": list(contract.folds),
        "arms": list(contract.arms),
        "primary_timings": list(contract.primary_timings),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


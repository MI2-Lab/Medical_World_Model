#!/usr/bin/env python3
"""Produce the static pCR-firewall attestation required before representation freeze."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


os.umask(0o077)
sys.dont_write_bytecode = True
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC = EXPERIMENT_ROOT / "src"
sys.path.insert(0, str(SRC))

from residual_sph.evaluation_lock import build_pcr_firewall_audit  # noqa: E402
from residual_sph.preregistration import (  # noqa: E402
    require_lock_sha256,
    verify_preregistration,
)


OUTPUT = EXPERIMENT_ROOT / "manifests" / "pcr_firewall_audit.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration-lock-sha256", required=True)
    args = parser.parse_args()
    preregistration = verify_preregistration(EXPERIMENT_ROOT)
    require_lock_sha256(preregistration["lock_sha256"], args.preregistration_lock_sha256)
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    payload = build_pcr_firewall_audit(
        EXPERIMENT_ROOT,
        preregistration_lock_sha256=preregistration["lock_sha256"],
        implementation_lock_sha256=preregistration["implementation_lock_sha256"],
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    OUTPUT.chmod(0o644)
    print(json.dumps({"status": payload["status"], "finding_count": len(payload["findings"])}))
    if payload["status"] != "PASS":
        raise SystemExit("pCR firewall audit failed")


if __name__ == "__main__":
    main()

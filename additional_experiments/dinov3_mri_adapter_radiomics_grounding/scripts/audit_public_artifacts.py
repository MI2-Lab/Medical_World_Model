#!/usr/bin/env python3
"""Run the public-artifact privacy gate."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dinov3_rg.contracts import atomic_json  # noqa: E402
from dinov3_rg.security import public_artifact_privacy_scan  # noqa: E402


def main() -> None:
    payload = public_artifact_privacy_scan(ROOT)
    atomic_json(ROOT / "metrics/public_artifact_privacy_gate.json", payload)
    print(payload)
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

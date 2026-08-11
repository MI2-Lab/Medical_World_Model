#!/usr/bin/env python3
"""Verify all new P0 states against the 40 immutable Stage-B references."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c1b_spatial_audit.parity import verify_p0_matrix  # noqa: E402


def _atomic_csv(path: Path, frame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        frame.to_csv(temporary_path, index=False)
        os.chmod(temporary_path, 0o644)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_path, 0o644)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, default=ROOT / "features" / "final")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    frame, summary = verify_p0_matrix(args.feature_root)
    if not args.execute:
        print(json.dumps(summary, sort_keys=True))
        return
    table = ROOT / "metrics" / "p0_equivalence_by_cell.csv"
    gate = ROOT / "metrics" / "p0_equivalence_gate.json"
    if table.exists() or gate.exists():
        raise FileExistsError("refusing to overwrite P0 equivalence outputs")
    _atomic_csv(table, frame)
    _atomic_json(gate, summary)
    print(json.dumps(summary, sort_keys=True))
    if summary["status"] != "PASS":
        raise SystemExit("P0 equivalence failed; all downstream probes are forbidden")


if __name__ == "__main__":
    main()

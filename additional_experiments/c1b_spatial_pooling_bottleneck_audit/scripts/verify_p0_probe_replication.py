#!/usr/bin/env python3
"""Gate alternate pooling interpretation on exact P0 probe replication."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c1b_spatial_audit.probe_parity import verify_p0_probe_matrix  # noqa: E402


def _write_csv(path: Path, frame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        frame.to_csv(temporary_path, index=False)
        os.chmod(temporary_path, 0o644)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
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
    parser.add_argument("--probe-root", type=Path, default=ROOT / "probes" / "final")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    cells, pooled, summary = verify_p0_probe_matrix(args.probe_root)
    if not args.execute:
        print(json.dumps(summary, sort_keys=True))
        return
    outputs = (
        ROOT / "metrics" / "p0_probe_replication_by_cell.csv",
        ROOT / "metrics" / "p0_probe_replication_pooled_metrics.csv",
        ROOT / "metrics" / "p0_probe_replication_gate.json",
    )
    if any(path.exists() for path in outputs):
        raise FileExistsError("refusing to overwrite P0 probe replication outputs")
    _write_csv(outputs[0], cells)
    _write_csv(outputs[1], pooled)
    _write_json(outputs[2], summary)
    print(json.dumps(summary, sort_keys=True))
    if summary["status"] != "PASS":
        raise SystemExit("P0 probe replication failed; alternate pooling is not interpretable")


if __name__ == "__main__":
    main()

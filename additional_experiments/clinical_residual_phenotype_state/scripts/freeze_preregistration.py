#!/usr/bin/env python3
"""Write or verify the Goal-F representation-training preregistration lock."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = EXPERIMENT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from crps.preregistration import LOCK_PATH, build_payload, verify  # noqa: E402


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.write:
        if LOCK_PATH.exists():
            raise FileExistsError("refusing to overwrite an existing preregistration lock")
        payload = build_payload()
        _atomic_json(LOCK_PATH, payload)
    else:
        payload = verify()
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

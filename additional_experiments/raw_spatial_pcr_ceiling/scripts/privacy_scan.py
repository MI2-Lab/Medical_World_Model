#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


FORBIDDEN_SUFFIXES = (".private.csv", ".private.json", ".private.npz", ".pt", ".pth", ".ckpt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail closed if private artifacts are staged in the public experiment tree.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    violations = []
    for path in args.root.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.name.endswith(FORBIDDEN_SUFFIXES) and not any(part in {"predictions", "features", "checkpoints", "logs", "manifests"} for part in path.parts):
            violations.append(str(path.relative_to(args.root)))
    payload = {"status": "PASS" if not violations else "FAIL", "violations": violations, "patient_level_predictions_committed": False}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


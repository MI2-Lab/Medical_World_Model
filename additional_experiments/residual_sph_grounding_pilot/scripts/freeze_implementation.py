#!/usr/bin/env python3
"""Create the exact code/test implementation lock before formal training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
SRC = EXPERIMENT_ROOT / "src"
sys.path.insert(0, str(SRC))

from residual_sph.contracts import canonical_sha256, file_sha256  # noqa: E402
from residual_sph.preregistration import (  # noqa: E402
    implementation_files,
    verify_preregistration,
)


OUTPUT = EXPERIMENT_ROOT / "manifests" / "implementation_lock.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passed-test-count", type=int, required=True)
    args = parser.parse_args()
    if args.passed_test_count <= 0:
        raise ValueError("passed-test-count must be positive")
    scientific = verify_preregistration(
        EXPERIMENT_ROOT, require_implementation=False
    )
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    private_roots = (
        EXPERIMENT_ROOT / "checkpoints",
        EXPERIMENT_ROOT / "features",
        EXPERIMENT_ROOT / "predictions",
    )
    forbidden = [
        path
        for root in private_roots
        if root.exists()
        for path in root.rglob("*")
        if path.is_file() and path.name != ".gitkeep"
    ]
    if forbidden:
        raise RuntimeError("formal/private artifacts exist before implementation lock")
    files = implementation_files(EXPERIMENT_ROOT)
    inventory = {
        path.relative_to(REPO_ROOT).as_posix(): file_sha256(path) for path in files
    }
    commands = [
        "PYTHONDONTWRITEBYTECODE=1 python3.11 -m pytest -q additional_experiments/residual_sph_grounding_pilot/tests",
        "python3.11 -m py_compile <implementation Python inventory>",
        "git diff --check -- additional_experiments/residual_sph_grounding_pilot",
        "python3.11 additional_experiments/residual_sph_grounding_pilot/scripts/freeze_preregistration.py --scientific-only",
    ]
    attestation_body = {
        "status": "PASS",
        "passed_test_count": int(args.passed_test_count),
        "commands": commands,
        "warnings_are_failures": False,
    }
    attestation = {
        **attestation_body,
        "attestation_sha256": canonical_sha256(attestation_body),
    }
    payload = {
        "schema_version": 1,
        "experiment": "residual_sph_grounding_pilot",
        "status": "IMPLEMENTATION_FROZEN_BEFORE_FORMAL_RESULTS",
        "scientific_preregistration_sha256": scientific["lock_sha256"],
        "formal_cells_started_before_implementation_lock": 0,
        "implementation_file_policy": "all Python files under src, scripts, and tests",
        "implementation_files_sha256": dict(sorted(inventory.items())),
        "test_attestation": attestation,
    }
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    verified = verify_preregistration(EXPERIMENT_ROOT)
    print(
        json.dumps(
            {
                "status": verified["status"],
                "implementation_file_count": len(inventory),
                "implementation_lock_sha256": verified["implementation_lock_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

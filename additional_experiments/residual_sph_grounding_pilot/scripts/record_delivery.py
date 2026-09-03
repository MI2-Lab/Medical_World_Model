#!/usr/bin/env python3
"""Record the required implementation commit and non-force push attempt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
SRC = EXPERIMENT_ROOT / "src"
sys.path.insert(0, str(SRC))

from residual_sph.contracts import file_sha256  # noqa: E402
from residual_sph.preregistration import verify_preregistration  # noqa: E402


BRANCH = "feature/residual-sph-grounding-pilot"
COMMIT_MESSAGE = "Add residual SPH grounding pilot"
PUSH_COMMAND = "git push -u origin feature/residual-sph-grounding-pilot"
OUTPUT = EXPERIMENT_ROOT / "manifests" / "delivery.json"
SHA = re.compile(r"[0-9a-f]{40}")


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation-commit-sha", required=True)
    parser.add_argument(
        "--push-status", choices=("PASS", "GITHUB_PUSH_FAILED"), required=True
    )
    args = parser.parse_args()
    preregistration = verify_preregistration(EXPERIMENT_ROOT)
    commit = args.implementation_commit_sha.lower()
    if SHA.fullmatch(commit) is None:
        raise ValueError("implementation commit must be a full Git SHA")
    if _git("branch", "--show-current") != BRANCH:
        raise ValueError("delivery must be recorded on the preregistered branch")
    _git("cat-file", "-e", f"{commit}^{{commit}}")
    if _git("show", "-s", "--format=%s", commit) != COMMIT_MESSAGE:
        raise ValueError("implementation commit message differs from the required message")
    if _git("rev-parse", "HEAD") != commit:
        raise ValueError("record delivery immediately after the implementation commit")
    remote_sha: str | None = None
    if args.push_status == "PASS":
        lines = _git("ls-remote", "--heads", "origin", BRANCH).splitlines()
        if len(lines) != 1:
            raise ValueError("remote branch is missing or ambiguous after reported push")
        remote_sha = lines[0].split()[0]
        if remote_sha != commit:
            raise ValueError("remote branch does not match the implementation commit")
    artifacts = (
        EXPERIMENT_ROOT / "PREREGISTRATION_LOCK.json",
        EXPERIMENT_ROOT / "manifests/implementation_lock.json",
        EXPERIMENT_ROOT / "manifests/s0_confirmation_provenance.json",
        EXPERIMENT_ROOT / "manifests/residualizer_inventory.json",
        EXPERIMENT_ROOT / "metrics/execution_status.json",
        EXPERIMENT_ROOT / "metrics/decision.json",
    )
    missing = [path.name for path in artifacts if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"delivery evidence is incomplete: {missing}")
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    payload = {
        "schema_version": 1,
        "experiment": "residual_sph_grounding_pilot",
        "status": "DELIVERY_RECORDED",
        "branch": BRANCH,
        "starting_head": "7644e3835af6b12899c57819bedd1876572c434f",
        "implementation_commit_sha": commit,
        "required_commit_message": COMMIT_MESSAGE,
        "push_command": PUSH_COMMAND,
        "push_status": args.push_status,
        "remote_branch_sha_after_push": remote_sha,
        "force_push_used": False,
        "privacy_gate_regeneration_required_after_manifest_creation": True,
        "delivery_manifest_commit_note": (
            "This manifest is committed separately because a Git commit cannot "
            "contain its own SHA; implementation_commit_sha is the required science/code commit."
        ),
        "preregistration_lock_sha256": preregistration["lock_sha256"],
        "implementation_lock_sha256": preregistration["implementation_lock_sha256"],
        "delivery_evidence_sha256": {
            path.relative_to(EXPERIMENT_ROOT).as_posix(): file_sha256(path)
            for path in artifacts
        },
        "patient_identifiers_persisted": False,
    }
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

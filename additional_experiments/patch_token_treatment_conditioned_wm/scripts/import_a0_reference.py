#!/usr/bin/env python3
"""Copy the immutable two-seed LOCAL3 reference into private pilot storage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
DEFAULT_SOURCE = (
    REPO_ROOT.parent
    / "Medical_World_Model"
    / "additional_experiments"
    / "local_response_state_multiseed_confirmation"
)
SEEDS = (2026, 3026)
FOLDS = tuple(range(5))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _copy_once(source: Path, destination: Path) -> str:
    if not source.is_file():
        raise FileNotFoundError(source)
    expected = file_sha256(source)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if destination.exists():
        if file_sha256(destination) != expected:
            raise FileExistsError(f"existing A0 artifact differs: {destination}")
        return expected
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".copy", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        if file_sha256(temporary) != expected:
            raise OSError("A0 copy failed SHA-256 verification")
        os.chmod(temporary, 0o600)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return expected


def import_reference(source_root: Path) -> dict[str, Any]:
    checkpoint_root = EXPERIMENT_ROOT / "checkpoints" / "a0_local3_reference"
    feature_root = EXPERIMENT_ROOT / "features" / "a0_local3_reference"
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for fold in FOLDS:
            checkpoint_source = (
                source_root
                / "checkpoints"
                / "formal_4x8"
                / f"seed_{seed}"
                / "LOCAL3"
                / f"fold_{fold}"
            )
            feature_source = (
                source_root
                / "features"
                / "formal_4x8"
                / f"seed_{seed}"
                / "LOCAL3"
                / f"fold_{fold}"
            )
            checkpoint_destination = checkpoint_root / f"seed_{seed}" / f"fold_{fold}"
            feature_destination = feature_root / f"seed_{seed}" / f"fold_{fold}"
            artifacts = {
                "selected_checkpoint_sha256": _copy_once(
                    checkpoint_source / "selected.pt",
                    checkpoint_destination / "selected.pt",
                ),
                "selection_sha256": _copy_once(
                    checkpoint_source / "selection.json",
                    checkpoint_destination / "selection.json",
                ),
                "response_feature_sha256": _copy_once(
                    feature_source / "response_state.private.npz",
                    feature_destination / "response_state.private.npz",
                ),
                "response_metadata_sha256": _copy_once(
                    feature_source / "response_state.private.metadata.json",
                    feature_destination / "response_state.private.metadata.json",
                ),
            }
            selection = json.loads(
                (checkpoint_destination / "selection.json").read_text(encoding="utf-8")
            )
            if (
                selection.get("arm") != "LOCAL3"
                or int(selection.get("seed_base", -1)) != seed
                or int(selection.get("fold", -1)) != fold
                or selection.get("test_data_used") is not False
                or selection.get("pcr_used") is not False
            ):
                raise ValueError("A0 selection violates the immutable LOCAL3 identity")
            rows.append(
                {
                    "seed_base": seed,
                    "fold": fold,
                    "selected_epoch": int(selection["selected_epoch"]),
                    **artifacts,
                }
            )
    manifest = {
        "schema_version": 1,
        "status": "A0_REFERENCE_IMPORTED",
        "experiment": "local_response_state_multiseed_confirmation",
        "arm": "LOCAL3",
        "source_preregistration_lock_sha256": (
            "a4e1cd2d8b61a7130da2b2eb6dc04e9a5355f44d0a37f4ceccf2fba48b35a9ee"
        ),
        "cells": rows,
        "cell_count": len(rows),
        "patient_identifiers_in_manifest": False,
        "private_artifacts_gitignored": True,
    }
    if len(rows) != 10:
        raise AssertionError("A0 reference matrix must contain exactly ten cells")
    _atomic_json(EXPERIMENT_ROOT / "manifests" / "a0_reference.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    result = import_reference(args.source_root.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

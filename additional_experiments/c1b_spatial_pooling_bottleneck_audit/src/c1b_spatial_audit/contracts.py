"""Immutable paths and provenance helpers for the spatial pooling audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator


PACKAGE_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = PACKAGE_ROOT.parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[3]
UPSTREAM_ROOT = REPO_ROOT / "additional_experiments" / "c1b_overlap_eligibility_ftv_stageb"
MODEL_READY_ROOT = REPO_ROOT / "additional_experiments" / "c1b_model_ready_ftv_sanity"
G3_ROOT = REPO_ROOT / "additional_experiments" / "g3_multiseed_generalization"

CHECKPOINT_ROOT = UPSTREAM_ROOT / "checkpoints" / "formal_4x8_restart1"
REFERENCE_FEATURE_ROOT = UPSTREAM_ROOT / "features" / "formal_4x8_restart1"
REFERENCE_PROBE_ROOT = UPSTREAM_ROOT / "predictions" / "formal_4x8_restart1"

SEEDS = (2026, 3026)
ARMS = ("L1", "L3", "N1", "N3")
FOLDS = tuple(range(5))
POOLINGS = ("P0", "PVALID", "PLOCAL", "PLOCAL+GLOBAL", "PORACLE")
TIMEPOINTS = ("T0", "T1", "T2", "T3")
TRANSITIONS = ("T0→T1", "T1→T2", "T2→T3")

UPSTREAM_COMPLETION_SHA256 = {
    "checkpoints/formal_4x8_restart1/matrix_complete.json": (
        "0adbcf7daf74f31e70c64c1ec9a5bb259411792fb0dfa4d093ee9d3e3210b4a2"
    ),
    "features/formal_4x8_restart1/feature_export_complete.json": (
        "f8bc1a158c93c0563b11e46cb02c4b0ef5681048febd94ab6d674d3ea4fdc40d"
    ),
    "predictions/formal_4x8_restart1/postprocessing_complete.json": (
        "4a599f5d76482677056f9df11e46faa1b8d4f277eedabb63d60306531e841558"
    ),
    "metrics/stage_b_aggregation_summary.json": (
        "2c70c429d7b32640160f8ffbbf9b3f3f7b991838227a964387bd2f4090e445c4"
    ),
    "manifests/stage_b_data_contract.private.json": (
        "dd22f130043863d4fce8956061fca389894a31874567ed7929e139f32ff5ab27"
    ),
}

UPSTREAM_SOURCE_SHA256 = {
    "additional_experiments/g3_multiseed_generalization/src/dgrs/model.py": (
        "ce39878a0fef5af1f92a86811faabbe73b39f57cdaf6d7580bbd65bd855d4ed9"
    ),
    "additional_experiments/c1b_overlap_eligibility_ftv_stageb/src/c1b_stage_b/probes.py": (
        "5cb42e886204823cea2aa86cb2361592123b0d6921652fe18665ebc4eb52295e"
    ),
    "additional_experiments/c1b_overlap_eligibility_ftv_stageb/src/c1b_stage_b/targets.py": (
        "06434db46cf76e6f39ff6eb1c476933885e90ed0a4c952dcc0a3477a25996c7b"
    ),
    "additional_experiments/c1b_model_ready_ftv_sanity/src/c1b_sanity/geometry.py": (
        "4921d09eeee7b8e57ff01a2c2dc6a1a92b901cf3ccfb2b3048987c724a9c5c3c"
    ),
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def relative(path: str | Path) -> str:
    return Path(path).resolve().relative_to(REPO_ROOT).as_posix()


def cells() -> Iterator[tuple[int, str, int]]:
    for seed in SEEDS:
        for arm in ARMS:
            for fold in FOLDS:
                yield seed, arm, fold


def cell_key(seed: int, arm: str, fold: int) -> str:
    return f"seed_{int(seed)}/{str(arm).upper()}/fold_{int(fold)}"


def checkpoint_path(seed: int, arm: str, fold: int) -> Path:
    return CHECKPOINT_ROOT / cell_key(seed, arm, fold) / "selected.pt"


def reference_feature_path(seed: int, arm: str, fold: int) -> Path:
    return REFERENCE_FEATURE_ROOT / cell_key(seed, arm, fold) / "response_state.private.npz"


def reference_feature_metadata_path(seed: int, arm: str, fold: int) -> Path:
    return reference_feature_path(seed, arm, fold).with_suffix(".metadata.json")


def reference_probe_dir(seed: int, arm: str, fold: int) -> Path:
    return REFERENCE_PROBE_ROOT / cell_key(seed, arm, fold)


__all__ = [
    "ARMS",
    "CHECKPOINT_ROOT",
    "EXPERIMENT_ROOT",
    "FOLDS",
    "G3_ROOT",
    "MODEL_READY_ROOT",
    "POOLINGS",
    "REFERENCE_FEATURE_ROOT",
    "REFERENCE_PROBE_ROOT",
    "REPO_ROOT",
    "SEEDS",
    "TIMEPOINTS",
    "TRANSITIONS",
    "UPSTREAM_COMPLETION_SHA256",
    "UPSTREAM_ROOT",
    "UPSTREAM_SOURCE_SHA256",
    "canonical_sha256",
    "cell_key",
    "cells",
    "checkpoint_path",
    "file_sha256",
    "reference_feature_metadata_path",
    "reference_feature_path",
    "reference_probe_dir",
    "relative",
]


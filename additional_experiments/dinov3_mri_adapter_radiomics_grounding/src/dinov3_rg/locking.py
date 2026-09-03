"""Freeze and verify the one-way representation-to-evaluation boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import (
    ARMS,
    EXPERIMENT_ROOT,
    FOLDS,
    SEEDS,
    atomic_json,
    canonical_sha256,
    expected_cells,
    file_sha256,
)
from .data import validate_state_archive
from .training import checkpoint_feasible


LOCK_PATH = EXPERIMENT_ROOT / "EVALUATION_LOCK.json"


def implementation_manifest() -> dict[str, str]:
    paths = sorted((EXPERIMENT_ROOT / "src").rglob("*.py"))
    paths.extend(sorted((EXPERIMENT_ROOT / "scripts").glob("*.py")))
    return {
        str(path.relative_to(EXPERIMENT_ROOT)): file_sha256(path)
        for path in sorted(set(paths))
    }


def freeze_evaluation_lock(
    checkpoint_root: str | Path,
    state_root: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    if LOCK_PATH.exists() and not overwrite:
        raise FileExistsError(f"evaluation lock already exists: {LOCK_PATH}")
    checkpoint_root = Path(checkpoint_root)
    state_root = Path(state_root)
    cells: dict[str, dict[str, Any]] = {}
    shared_patient_order: str | None = None
    for seed in SEEDS:
        for fold in FOLDS:
            completions: dict[str, dict[str, Any]] = {}
            initialization: set[str] = set()
            train_orders: set[str] = set()
            validation_orders: set[str] = set()
            references: dict[str, dict[str, Any]] = {}
            for arm in ARMS:
                tag = f"seed{seed}_fold{fold}_{arm}"
                cell_dir = checkpoint_root / tag
                complete_path = cell_dir / "cell_complete.private.json"
                checkpoint_path = cell_dir / "selected.private.pt"
                state_path = state_root / f"{tag}_states.private.npz"
                if not complete_path.is_file() or not checkpoint_path.is_file() or not state_path.is_file():
                    raise FileNotFoundError(f"incomplete representation cell: {tag}")
                complete = json.loads(complete_path.read_text(encoding="utf-8"))
                if complete.get("status") != "COMPLETE" or complete.get("arm") != arm:
                    raise ValueError(f"invalid completion sentinel: {tag}")
                if file_sha256(checkpoint_path) != complete["checkpoint_sha256"]:
                    raise ValueError(f"checkpoint hash mismatch: {tag}")
                patient_ids, _ = validate_state_archive(state_path)
                order_hash = canonical_sha256(list(patient_ids))
                if shared_patient_order is None:
                    shared_patient_order = order_hash
                if order_hash != shared_patient_order:
                    raise ValueError("state patient order differs across cells")
                initialization.add(str(complete["initialization_sha256"]))
                train_orders.add(str(complete["train_patient_order_sha256"]))
                validation_orders.add(str(complete["validation_patient_order_sha256"]))
                completions[arm] = complete
                if arm == "D1":
                    references["D1"] = complete
                elif arm == "D2":
                    references["D2"] = complete
                if not checkpoint_feasible(
                    arm,
                    complete["selected_validation"],
                    {name: references[name] for name in ({"D1": (), "D2": ("D1",), "D3": ("D1", "D2")}[arm])},
                ):
                    raise ValueError(f"selected checkpoint violates paired constraints: {tag}")
                cells[tag] = {
                    "completion_sha256": file_sha256(complete_path),
                    "checkpoint_sha256": file_sha256(checkpoint_path),
                    "state_sha256": file_sha256(state_path),
                    "state_shape": [808, 4, 192],
                }
            if len(initialization) != 1 or len(train_orders) != 1 or len(validation_orders) != 1:
                raise ValueError(f"D1-D3 pairing contract failed for seed={seed}, fold={fold}")
    if tuple(sorted(cells)) != tuple(sorted(expected_cells())):
        raise ValueError("evaluation lock does not contain exactly 75 cells")
    prerequisite_paths = [
        EXPERIMENT_ROOT / "manifests/dinov3_cache_complete.json",
        EXPERIMENT_ROOT / "metrics/radiomics_stage_a_gate.json",
        *(EXPERIMENT_ROOT / f"metrics/fold_{fold}_target_gate.json" for fold in FOLDS),
    ]
    prerequisites: dict[str, str] = {}
    for path in prerequisite_paths:
        if not path.is_file():
            raise FileNotFoundError(f"evaluation prerequisite is missing: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") not in {"PASS", "COMPLETE"}:
            raise ValueError(f"evaluation prerequisite did not pass: {path}")
        prerequisites[str(path.relative_to(EXPERIMENT_ROOT))] = file_sha256(path)
    implementation = implementation_manifest()
    payload = {
        "schema_version": 1,
        "status": "LOCKED_FOR_EVALUATION",
        "cells": cells,
        "cell_count": len(cells),
        "state_patient_order_sha256": shared_patient_order,
        "prerequisites": prerequisites,
        "implementation": implementation,
        "implementation_sha256": canonical_sha256(implementation),
        "outcome_fields_read_before_lock": [],
        "clinical_fields_read_before_lock": [],
        "representation_complete_before_evaluation": True,
    }
    payload["lock_content_sha256"] = canonical_sha256(payload)
    atomic_json(LOCK_PATH, payload)
    return payload


def verify_evaluation_lock(path: str | Path = LOCK_PATH) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise PermissionError("pCR evaluation is locked: EVALUATION_LOCK.json is absent")
    payload = json.loads(source.read_text(encoding="utf-8"))
    claimed = payload.pop("lock_content_sha256", None)
    if payload.get("status") != "LOCKED_FOR_EVALUATION" or claimed != canonical_sha256(payload):
        raise PermissionError("evaluation lock content/hash is invalid")
    if int(payload.get("cell_count", -1)) != 75 or set(payload.get("cells", {})) != set(expected_cells()):
        raise PermissionError("evaluation lock cell coverage is invalid")
    if payload.get("implementation") != implementation_manifest():
        raise PermissionError("implementation changed after evaluation lock")
    payload["lock_content_sha256"] = claimed
    return payload


__all__ = ["LOCK_PATH", "freeze_evaluation_lock", "implementation_manifest", "verify_evaluation_lock"]

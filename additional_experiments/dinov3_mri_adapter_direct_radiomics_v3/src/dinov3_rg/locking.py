"""Hash-bound pilot, representation, and mechanism locks for V3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import (
    ARMS, EXPERIMENT_ROOT, FOLDS, PILOT_ARMS, SEEDS, atomic_json,
    canonical_sha256, expected_cells, file_sha256, load_protocol,
)
from .data import validate_state_archive


PILOT_LOCK_PATH = EXPERIMENT_ROOT / "PILOT_LOCK.json"
LOCK_PATH = EXPERIMENT_ROOT / "EVALUATION_LOCK.json"
MECHANISM_LOCK_PATH = EXPERIMENT_ROOT / "MECHANISM_LOCK.json"


def implementation_manifest() -> dict[str, str]:
    paths = sorted((EXPERIMENT_ROOT / "src").rglob("*.py"))
    paths.extend(sorted((EXPERIMENT_ROOT / "scripts").glob("*.py")))
    return {
        str(path.relative_to(EXPERIMENT_ROOT)): file_sha256(path)
        for path in sorted(set(paths))
    }


def _write_content_lock(path: Path, payload: dict[str, Any], overwrite: bool) -> dict[str, Any]:
    if path.exists() and not overwrite:
        raise FileExistsError(f"lock already exists: {path}")
    payload["lock_content_sha256"] = canonical_sha256(payload)
    atomic_json(path, payload)
    return payload


def _read_content_lock(path: Path, expected_status: str) -> dict[str, Any]:
    if not path.is_file():
        raise PermissionError(f"required lock is absent: {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("lock_content_sha256", None)
    if payload.get("status") != expected_status or claimed != canonical_sha256(payload):
        raise PermissionError(f"invalid lock content/hash: {path.name}")
    payload["lock_content_sha256"] = claimed
    return payload


def freeze_pilot_lock(
    gate_path: str | Path,
    metrics_path: str | Path,
    execution_path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    gate_path, metrics_path, execution_path = map(Path, (gate_path, metrics_path, execution_path))
    for path in (gate_path, metrics_path, execution_path):
        if not path.is_file():
            raise FileNotFoundError(f"pilot lock input missing: {path}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS" or not all(gate.get("selected_candidate_gates", {}).values()):
        raise PermissionError("pilot gate did not pass")
    selected_arm = str(gate["selected_arm"])
    if selected_arm not in PILOT_ARMS:
        raise ValueError("pilot selected an unregistered arm")
    implementation = implementation_manifest()
    payload = {
        "schema_version": 1,
        "status": "PILOT_WEIGHT_LOCKED",
        "selected_arm": selected_arm,
        "selected_radiomics_weight": float(gate["selected_radiomics_weight"]),
        "pilot_gate_sha256": file_sha256(gate_path),
        "pilot_metrics_sha256": file_sha256(metrics_path),
        "pilot_execution_sha256": file_sha256(execution_path),
        "implementation": implementation,
        "implementation_sha256": canonical_sha256(implementation),
        "outcome_fields_read_before_lock": [],
        "clinical_fields_read_before_lock": [],
    }
    return _write_content_lock(PILOT_LOCK_PATH, payload, overwrite)


def verify_pilot_lock(path: str | Path = PILOT_LOCK_PATH) -> dict[str, Any]:
    payload = _read_content_lock(Path(path), "PILOT_WEIGHT_LOCKED")
    protocol = load_protocol()
    arm = str(payload.get("selected_arm"))
    if arm not in PILOT_ARMS:
        raise PermissionError("pilot lock arm is invalid")
    expected_weight = float(protocol["loss"]["pilot_radiomics_weights"][arm])
    if float(payload.get("selected_radiomics_weight", -1)) != expected_weight:
        raise PermissionError("pilot lock weight is invalid")
    if payload.get("implementation") != implementation_manifest():
        raise PermissionError("implementation changed after pilot lock")
    return payload


def freeze_evaluation_lock(
    checkpoint_root: str | Path,
    state_root: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    pilot = verify_pilot_lock()
    checkpoint_root, state_root = Path(checkpoint_root), Path(state_root)
    cells: dict[str, dict[str, Any]] = {}
    shared_patient_order: str | None = None
    for seed in SEEDS:
        for fold in FOLDS:
            pair: dict[str, dict[str, Any]] = {}
            for arm in ARMS:
                tag = f"seed{seed}_fold{fold}_{arm}"
                cell_dir = checkpoint_root / tag
                complete_path = cell_dir / "cell_complete.private.json"
                checkpoint_path = cell_dir / "selected.private.pt"
                state_path = state_root / f"{tag}_states.private.npz"
                if not all(path.is_file() for path in (complete_path, checkpoint_path, state_path)):
                    raise FileNotFoundError(f"incomplete formal cell: {tag}")
                complete = json.loads(complete_path.read_text(encoding="utf-8"))
                if complete.get("status") != "COMPLETE" or complete.get("arm") != arm:
                    raise ValueError(f"invalid completion sentinel: {tag}")
                if file_sha256(checkpoint_path) != complete.get("checkpoint_sha256"):
                    raise ValueError(f"checkpoint hash mismatch: {tag}")
                patient_ids, state = validate_state_archive(state_path)
                with np.load(state_path, allow_pickle=False) as archive:
                    state_checkpoint_sha = str(archive["checkpoint_sha256"].item())
                order_hash = canonical_sha256(list(patient_ids))
                if shared_patient_order is None:
                    shared_patient_order = order_hash
                if order_hash != shared_patient_order or tuple(state.shape) != (808, 4, 192):
                    raise ValueError(f"state archive contract failed: {tag}")
                if state_checkpoint_sha != complete["checkpoint_sha256"]:
                    raise ValueError(f"state/checkpoint binding failed: {tag}")
                pair[arm] = complete
                cells[tag] = {
                    "completion_sha256": file_sha256(complete_path),
                    "checkpoint_sha256": file_sha256(checkpoint_path),
                    "state_sha256": file_sha256(state_path),
                    "state_shape": [808, 4, 192],
                }
            c0, rad = pair["C0"], pair["RAD"]
            if (
                float(c0["radiomics_weight"]) != 0.0
                or float(rad["radiomics_weight"]) != float(pilot["selected_radiomics_weight"])
                or float(c0["ftv_weight"]) != 0.0
                or float(rad["ftv_weight"]) != 0.0
                or rad["base_checkpoint_sha256"] != c0["checkpoint_sha256"]
                or rad["train_patient_order_sha256"] != c0["train_patient_order_sha256"]
                or rad["validation_patient_order_sha256"] != c0["validation_patient_order_sha256"]
            ):
                raise ValueError(f"formal pairing contract failed: seed={seed}, fold={fold}")
            allowed = 1.05 * float(c0["selected_validation"]["jepa_loss"])
            if float(rad["selected_validation"]["jepa_loss"]) > allowed:
                raise ValueError(f"paired JEPA safety failed: seed={seed}, fold={fold}")
    if set(cells) != set(expected_cells()) or len(cells) != 50:
        raise ValueError("evaluation lock requires exactly 50 formal cells")
    prerequisites = {}
    for path in (
        EXPERIMENT_ROOT / "inheritance_check.json",
        EXPERIMENT_ROOT / "metrics/preflight.json",
        EXPERIMENT_ROOT / "metrics/smoke.json",
        EXPERIMENT_ROOT / "pilot_gate.json",
        EXPERIMENT_ROOT / "metrics/representation_matrix_complete.json",
        PILOT_LOCK_PATH,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"evaluation prerequisite missing: {path}")
        prerequisites[str(path.relative_to(EXPERIMENT_ROOT))] = file_sha256(path)
    implementation = implementation_manifest()
    payload = {
        "schema_version": 1,
        "status": "LOCKED_FOR_EVALUATION",
        "cells": cells,
        "cell_count": len(cells),
        "state_patient_order_sha256": shared_patient_order,
        "pilot_lock_sha256": file_sha256(PILOT_LOCK_PATH),
        "prerequisites": prerequisites,
        "implementation": implementation,
        "implementation_sha256": canonical_sha256(implementation),
        "outcome_fields_read_before_lock": [],
        "clinical_fields_read_before_lock": [],
        "representation_complete_before_evaluation": True,
    }
    return _write_content_lock(LOCK_PATH, payload, overwrite)


def verify_representation_lock(path: str | Path = LOCK_PATH) -> dict[str, Any]:
    payload = _read_content_lock(Path(path), "LOCKED_FOR_EVALUATION")
    pilot = verify_pilot_lock()
    if (
        int(payload.get("cell_count", -1)) != 50
        or set(payload.get("cells", {})) != set(expected_cells())
        or payload.get("pilot_lock_sha256") != file_sha256(PILOT_LOCK_PATH)
        or payload.get("implementation") != implementation_manifest()
        or pilot.get("status") != "PILOT_WEIGHT_LOCKED"
    ):
        raise PermissionError("evaluation lock coverage or implementation is invalid")
    return payload


def freeze_mechanism_lock(
    gate_path: str | Path,
    metrics_path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    representation = verify_representation_lock()
    gate_path, metrics_path = Path(gate_path), Path(metrics_path)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS" or not all(gate.get("gates", {}).values()):
        raise PermissionError("mechanism gate did not pass; outcomes remain locked")
    payload = {
        "schema_version": 1,
        "status": "LOCKED_FOR_PCR_EVALUATION",
        "representation_lock_sha256": file_sha256(LOCK_PATH),
        "representation_lock_content_sha256": representation["lock_content_sha256"],
        "mechanism_gate_sha256": file_sha256(gate_path),
        "mechanism_metrics_sha256": file_sha256(metrics_path),
        "gates": gate["gates"],
        "outcome_fields_read_before_lock": [],
        "clinical_fields_read_before_lock": [],
    }
    return _write_content_lock(MECHANISM_LOCK_PATH, payload, overwrite)


def verify_mechanism_lock(path: str | Path = MECHANISM_LOCK_PATH) -> dict[str, Any]:
    representation = verify_representation_lock()
    payload = _read_content_lock(Path(path), "LOCKED_FOR_PCR_EVALUATION")
    if (
        payload.get("representation_lock_sha256") != file_sha256(LOCK_PATH)
        or payload.get("representation_lock_content_sha256") != representation["lock_content_sha256"]
        or not all(payload.get("gates", {}).values())
    ):
        raise PermissionError("mechanism lock binding is invalid")
    return payload


def verify_evaluation_lock(path: str | Path = LOCK_PATH) -> dict[str, Any]:
    representation = verify_representation_lock(path)
    representation["mechanism_lock"] = verify_mechanism_lock()
    return representation


__all__ = [
    "LOCK_PATH", "MECHANISM_LOCK_PATH", "PILOT_LOCK_PATH", "freeze_evaluation_lock",
    "freeze_mechanism_lock", "freeze_pilot_lock", "implementation_manifest",
    "verify_evaluation_lock", "verify_mechanism_lock", "verify_pilot_lock",
    "verify_representation_lock",
]

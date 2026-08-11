"""Fail-closed access to the immutable Stage-B runtime and data bundle."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

from .contracts import (
    EXPERIMENT_ROOT,
    G3_ROOT,
    MODEL_READY_ROOT,
    UPSTREAM_COMPLETION_SHA256,
    UPSTREAM_ROOT,
    file_sha256,
)


_UPSTREAM_SRC = UPSTREAM_ROOT / "src"
_MODEL_READY_SRC = MODEL_READY_ROOT / "src"
_G3_SRC = G3_ROOT / "src"

# Both the formal Stage-B experiment and its older model-ready predecessor ship
# a top-level ``c1b_stage_b`` package.  Insert dependencies in reverse priority
# so the hash-locked formal implementation is always the first import target.
# A forward ``insert(0, ...)`` loop would silently select the schema-v1 package.
for _source in (_G3_SRC, _MODEL_READY_SRC, _UPSTREAM_SRC):
    source_text = str(_source)
    if source_text in sys.path:
        sys.path.remove(source_text)
    sys.path.insert(0, source_text)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def verify_preregistration() -> dict[str, Any]:
    lock_path = EXPERIMENT_ROOT / "PREREGISTRATION_LOCK.json"
    lock = _read_json(lock_path)
    if lock.get("status") != "FROZEN_BEFORE_NEW_FEATURE_OR_PROBE":
        raise ValueError("pooling audit preregistration is absent or invalid")
    if int(lock.get("formal_cell_count", -1)) != 40:
        raise ValueError("preregistration does not bind the exact 40-cell matrix")
    expected = {
        "plan_sha256": EXPERIMENT_ROOT / "EXPERIMENT_PLAN.md",
        "config_sha256": EXPERIMENT_ROOT / "configs" / "audit.json",
    }
    for field, path in expected.items():
        if lock.get(field) != file_sha256(path):
            raise ValueError(f"preregistered {field} drifted")
    for rel, digest in UPSTREAM_COMPLETION_SHA256.items():
        path = UPSTREAM_ROOT / rel
        if file_sha256(path) != digest:
            raise ValueError(f"upstream completion drifted: {rel}")
    return lock


def load_stage_b_bundle(*, verify_cache_files: bool = False):
    """Load exactly the prior Stage-B cohort through its original adapters."""

    verify_preregistration()
    from c1b_stage_b.gate import require_stage_a_go
    from c1b_stage_b.inputs import StageBDataPaths, load_stage_b_data

    sentinel = UPSTREAM_ROOT / "STAGE_A_GO.json"
    authorization = require_stage_a_go(sentinel)
    contract = UPSTREAM_ROOT / "manifests" / "stage_b_data_contract.private.json"
    expected = UPSTREAM_COMPLETION_SHA256[
        "manifests/stage_b_data_contract.private.json"
    ]
    paths = StageBDataPaths.load(contract, expected)
    data = load_stage_b_data(
        paths, authorization, verify_cache_files=bool(verify_cache_files)
    )
    return authorization, paths, data


def load_selected_model(path: str | Path, device):
    """Use the official strict evaluation loader; never restore optimizer state."""

    verify_preregistration()
    from c1b_stage_b.upstream import (
        load_checkpoint_for_evaluation,
        validate_model_contract,
    )

    model, checkpoint = load_checkpoint_for_evaluation(path, device)
    arm = str(checkpoint.get("arm", ""))
    validate_model_contract(model, arm)
    if checkpoint.get("selected") is not True or checkpoint.get("test_data_used") is not False:
        raise ValueError("audit requires a selected, test-blind formal checkpoint")
    model.eval()
    model.requires_grad_(False)
    return model, checkpoint


__all__ = [
    "load_selected_model",
    "load_stage_b_bundle",
    "verify_preregistration",
]

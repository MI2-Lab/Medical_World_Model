"""Immutable representation-training preregistration inventory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import assert_representation_config, canonical_sha256, file_sha256
from .stageb import SOURCE_VERIFICATION as STAGEB_SOURCE_VERIFICATION
from .upstream import SOURCE_VERIFICATION as MODEL_SOURCE_VERIFICATION


PACKAGE_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = PACKAGE_ROOT.parents[1]
CONFIG_PATH = EXPERIMENT_ROOT / "configs" / "representation.json"
LOCK_PATH = EXPERIMENT_ROOT / "PREREGISTRATION_LOCK.json"

CODE_PATHS = (
    "src/crps/contracts.py",
    "src/crps/upstream.py",
    "src/crps/stageb.py",
    "src/crps/data.py",
    "src/crps/model.py",
    "src/crps/losses.py",
    "src/crps/training.py",
    "src/crps/preregistration.py",
    "scripts/build_training_profiles.py",
    "scripts/train_cell.py",
    "scripts/run_matrix.py",
    "scripts/export_features.py",
)


def build_payload() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert_representation_config(config)
    code_sha256: dict[str, str] = {}
    for relative in CODE_PATHS:
        source = EXPERIMENT_ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(f"representation preregistration source missing: {relative}")
        code_sha256[relative] = file_sha256(source)
    profile = config["profiles"]
    manifest = Path(str(profile["training_manifest_path"])).expanduser().resolve()
    expected_manifest = str(profile["training_manifest_sha256"])
    if file_sha256(manifest) != expected_manifest:
        raise ValueError("pCR-free training profile manifest hash drifted before freezing")
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "experiment": "clinical_residual_phenotype_state",
        "phase": "representation_training",
        "PCR_LABEL_ACCESS": "FORBIDDEN",
        "config_path": str(CONFIG_PATH.relative_to(EXPERIMENT_ROOT)),
        "config_sha256": file_sha256(CONFIG_PATH),
        "code_sha256": code_sha256,
        "training_profile_manifest_sha256": expected_manifest,
        "upstream_source_sha256": {
            "model": MODEL_SOURCE_VERIFICATION,
            "stage_b": STAGEB_SOURCE_VERIFICATION,
        },
        "pcr_used_for_hyperparameter_selection": False,
        "pcr_used_for_checkpoint_selection": False,
        "pcr_used_for_representation_training": False,
    }
    payload["lock_sha256"] = canonical_sha256(payload)
    return payload


def verify(path: str | Path = LOCK_PATH) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError("representation preregistration lock is missing")
    locked = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(locked, dict):
        raise ValueError("representation preregistration lock must be a JSON object")
    observed = build_payload()
    if locked != observed:
        raise PermissionError("representation preregistration lock no longer matches code/config/data")
    return observed


__all__ = [
    "CODE_PATHS",
    "CONFIG_PATH",
    "EXPERIMENT_ROOT",
    "LOCK_PATH",
    "build_payload",
    "verify",
]

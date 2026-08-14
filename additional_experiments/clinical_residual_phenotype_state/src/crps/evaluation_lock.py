"""Immutable label-free boundary between feature export and outcome evaluation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from .evaluation_contracts import (
    ARMS,
    CONFIG_PATH,
    EXPERIMENT_ROOT,
    FOLDS,
    REPO_ROOT,
    SEEDS,
    EvaluationContractError,
    canonical_sha256,
    file_sha256,
    load_evaluation_config,
    load_f0_asset,
    load_factorized_asset,
    load_factorized_export_status,
    load_fold_assignments,
    load_representation_lock,
    load_stage_b_ftv_records,
)


LOCK_PATH = EXPERIMENT_ROOT / "EVALUATION_LOCK.json"
CODE_PATHS = (
    "configs/evaluation.json",
    "src/crps/diagnostics.py",
    "src/crps/evaluation_contracts.py",
    "src/crps/evaluation_lock.py",
    "src/crps/evaluation_modeling.py",
    "src/crps/evaluation.py",
    "src/crps/reporting.py",
    "src/crps/response_probes.py",
    "scripts/evaluate_frozen.py",
    "scripts/generate_report.py",
    "scripts/freeze_evaluation.py",
)


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError as error:
        raise EvaluationContractError(
            f"evaluation asset is outside the repository: {path}"
        ) from error


def _hash_inventory(paths: list[Path]) -> dict[str, str]:
    inventory: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"evaluation lock input is missing: {path}")
        key = _relative(path)
        if key in inventory:
            raise EvaluationContractError(f"duplicate evaluation lock input: {key}")
        inventory[key] = file_sha256(path)
    return dict(sorted(inventory.items()))


def build_payload(config_path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Build the deterministic boundary exclusively from label-free inputs."""

    source = Path(config_path).expanduser().resolve()
    if source != CONFIG_PATH.resolve():
        raise EvaluationContractError("formal evaluation lock requires the canonical config")
    config = load_evaluation_config(source)
    assignments = load_fold_assignments(config)
    representation = load_representation_lock(config)
    export_status_path, export_status = load_factorized_export_status(config)
    _, stage_b_provenance = load_stage_b_ftv_records(config, assignments)

    factorized_paths: list[Path] = []
    factorized_cells: list[dict[str, int | str]] = []
    for seed in SEEDS:
        for arm in ARMS:
            for fold in FOLDS:
                asset = load_factorized_asset(config, assignments, arm, seed, fold)
                factorized_paths.extend((asset.path, asset.metadata_path))
                factorized_cells.append(
                    {"seed_base": seed, "arm": arm, "fold": fold}
                )

    f0_paths: list[Path] = []
    f0_cells: list[dict[str, int | str]] = []
    for seed in SEEDS:
        for fold in FOLDS:
            asset = load_f0_asset(config, assignments, seed, fold)
            metadata_path = asset.path.with_suffix(".metadata.json")
            f0_paths.extend((asset.path, metadata_path))
            f0_cells.append(
                {"seed_base": seed, "arm": "LOCAL3", "fold": fold}
            )

    if len(factorized_cells) != 20 or len(f0_cells) != 10:
        raise EvaluationContractError("evaluation feature matrix cardinality drifted")
    code_paths = [EXPERIMENT_ROOT / relative for relative in CODE_PATHS]
    code_inventory = _hash_inventory(code_paths)
    factorized_inventory = _hash_inventory(factorized_paths)
    f0_inventory = _hash_inventory(f0_paths)
    representation_path = Path(
        config["frozen_inputs"]["representation_preregistration_lock"]
    )
    if not representation_path.is_absolute():
        representation_path = REPO_ROOT / representation_path
    fold_path = Path(config["frozen_inputs"]["fold_manifest_path"]).expanduser()
    if not fold_path.is_absolute():
        fold_path = REPO_ROOT / fold_path
    stage_a_path = Path(config["frozen_inputs"]["stage_a_sentinel_path"]).expanduser()
    if not stage_a_path.is_absolute():
        stage_a_path = REPO_ROOT / stage_a_path
    stage_b_contract_path = Path(
        config["frozen_inputs"]["stage_b_data_contract_path"]
    ).expanduser()
    if not stage_b_contract_path.is_absolute():
        stage_b_contract_path = REPO_ROOT / stage_b_contract_path

    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "experiment": "clinical_residual_phenotype_state",
        "phase": "post_export_pre_outcome_evaluation_lock",
        "outcome_access_during_build": "FORBIDDEN",
        "outcome_labels_read_during_build": False,
        "label_free_fold_assignment": {
            "columns": ["patient_id", "fold", "split"],
            "rows": int(len(assignments)),
            "patients": int(assignments["patient_id"].nunique()),
            "source_sha256": file_sha256(fold_path),
        },
        "config_path": _relative(source),
        "config_sha256": file_sha256(source),
        "code_sha256": code_inventory,
        "representation_preregistration": {
            "path": _relative(representation_path),
            "file_sha256": file_sha256(representation_path),
            "payload_sha256": representation["lock_sha256"],
            "status": representation["status"],
            "outcome_access": representation["PCR_LABEL_ACCESS"],
        },
        "stage_b_response_inputs": {
            "authorization_path": _relative(stage_a_path),
            "authorization_sha256": file_sha256(stage_a_path),
            "data_contract_path": _relative(stage_b_contract_path),
            "data_contract_sha256": file_sha256(stage_b_contract_path),
            "adapter": stage_b_provenance["adapter"],
            "ftv_patient_count": int(stage_b_provenance["ftv_patient_count"]),
            "source_file_sha256": {
                key: value
                for key, value in sorted(stage_b_provenance.items())
                if key.endswith("_sha256") and isinstance(value, str)
            },
        },
        "factorized_export_status": {
            "path": _relative(export_status_path),
            "sha256": file_sha256(export_status_path),
            "status": export_status["status"],
            "completed_cells": export_status["completed_cells"],
            "outcome_access": export_status["PCR_LABEL_ACCESS"],
        },
        "factorized_cells": factorized_cells,
        "factorized_file_count": len(factorized_inventory),
        "factorized_file_sha256": factorized_inventory,
        "f0_cells": f0_cells,
        "f0_file_count": len(f0_inventory),
        "f0_file_sha256": f0_inventory,
    }
    if payload["factorized_file_count"] != 40 or payload["f0_file_count"] != 20:
        raise EvaluationContractError("evaluation asset inventory is incomplete")
    payload["lock_sha256"] = canonical_sha256(payload)
    return payload


def freeze(
    path: str | Path = LOCK_PATH,
    *,
    config_path: str | Path = CONFIG_PATH,
) -> dict[str, Any]:
    """Create the canonical lock exactly once; existing files are never replaced."""

    destination = Path(path).expanduser().resolve()
    if destination != LOCK_PATH.resolve():
        raise EvaluationContractError("formal evaluation lock path is not canonical")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite evaluation lock: {destination}")
    payload = build_payload(config_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
        destination.chmod(0o444)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def verify(
    path: str | Path = LOCK_PATH,
    *,
    config_path: str | Path = CONFIG_PATH,
) -> dict[str, Any]:
    """Verify the lock and every bound code, status, and feature byte."""

    source = Path(path).expanduser().resolve()
    if source != LOCK_PATH.resolve() or not source.is_file():
        raise FileNotFoundError("canonical evaluation lock is missing")
    locked = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(locked, Mapping):
        raise EvaluationContractError("evaluation lock must be a JSON object")
    expected = build_payload(config_path)
    if locked != expected:
        raise PermissionError(
            "evaluation boundary drifted; outcome access remains forbidden"
        )
    return expected


def require_before_outcome_access(
    path: str | Path = LOCK_PATH,
    *,
    config_path: str | Path = CONFIG_PATH,
) -> dict[str, Any]:
    """Named gate for the evaluation entry point before it opens outcomes."""

    return verify(path, config_path=config_path)


__all__ = [
    "CODE_PATHS",
    "LOCK_PATH",
    "build_payload",
    "freeze",
    "require_before_outcome_access",
    "verify",
]

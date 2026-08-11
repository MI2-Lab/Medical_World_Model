"""Prospective, fail-closed authorization for the conditional S3 audit.

The preregistration permits S3 only when the *final-stage* Strong Oracle
Recovery gate is false.  The main aggregation intentionally cannot publish a
final classification until a required S3 audit is complete, so this module
materializes the earlier trigger decision as a small aggregate-only gate.  No
feature extraction, model execution, or probe fitting occurs here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

# Importing runtime first establishes the same formal Stage-B/model-ready
# source priority used by every audit CLI before analysis imports sidecars.
from .runtime import verify_preregistration
from .analysis import (
    build_ftv_table,
    build_recovery_table,
    load_audit_config,
    load_probe_stage,
    pooled_stage_natural_metrics,
    strong_oracle_gate,
    transformed_fold_summaries,
)
from .contracts import EXPERIMENT_ROOT, canonical_sha256, file_sha256


TRIGGERED_STATUS = "TRIGGERED_FINAL_ORACLE_WEAK"
NOT_TRIGGERED_STATUS = "NOT_TRIGGERED_FINAL_ORACLE_STRONG"
EXPECTED_FINAL_PROBE_CELLS = 180
S3_TRIGGER_GATE_FILENAME = "s3_trigger_authorization.json"

_GATE_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "s3_execution_authorized",
        "decision_contract",
        "final_stage_strong_oracle_recovery",
        "final_probe_root",
        "final_probe_cell_count",
        "final_probe_metadata_inventory_sha256",
        "preregistration_lock_sha256",
        "plan_sha256",
        "config_sha256",
        "p0_equivalence_gate_sha256",
        "p0_probe_replication_gate_sha256",
        "trigger_implementation_sha256",
        "analysis_implementation_sha256",
        "probe_adapter_sha256",
        "new_training_performed",
        "probe_refit_performed",
        "patient_identifiers_present",
    }
)


def _read_json(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {source}")
    return payload


def _relative_to_experiment(path: Path) -> str:
    source = path.resolve()
    try:
        return source.relative_to(EXPERIMENT_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("formal final probe root must remain inside this experiment") from exc


def _probe_metadata_inventory(probe_root: Path) -> tuple[int, str]:
    paths = sorted(probe_root.resolve().rglob("probe_metadata.json"))
    inventory = {
        path.relative_to(probe_root.resolve()).as_posix(): file_sha256(path)
        for path in paths
    }
    if len(inventory) != EXPECTED_FINAL_PROBE_CELLS:
        raise ValueError(
            "conditional S3 trigger requires the exact 180-cell final probe matrix"
        )
    return len(inventory), canonical_sha256(inventory)


def _validate_p0_gates(
    equivalence_gate_path: str | Path,
    replication_gate_path: str | Path,
    *,
    preregistration_sha256: str,
) -> tuple[str, str]:
    equivalence_path = Path(equivalence_gate_path).resolve()
    replication_path = Path(replication_gate_path).resolve()
    equivalence = _read_json(equivalence_path, label="P0 equivalence gate")
    replication = _read_json(replication_path, label="P0 probe replication gate")
    if (
        equivalence.get("status") != "PASS"
        or equivalence.get("probe_execution_authorized") is not True
        or int(equivalence.get("formal_cells", -1)) != 40
        or equivalence.get("preregistration_lock_sha256")
        != preregistration_sha256
    ):
        raise PermissionError("P0 equivalence gate does not authorize interpretation")
    if (
        replication.get("status") != "PASS"
        or replication.get("alternate_pooling_interpretation_authorized") is not True
        or int(replication.get("formal_cells", -1)) != 40
    ):
        raise PermissionError("P0 probe replication gate does not authorize interpretation")
    return file_sha256(equivalence_path), file_sha256(replication_path)


def compute_s3_trigger_gate(
    *,
    final_probe_root: str | Path = EXPERIMENT_ROOT / "probes" / "final",
    config_path: str | Path = EXPERIMENT_ROOT / "configs" / "audit.json",
    equivalence_gate_path: str | Path = EXPERIMENT_ROOT
    / "metrics"
    / "p0_equivalence_gate.json",
    replication_gate_path: str | Path = EXPERIMENT_ROOT
    / "metrics"
    / "p0_probe_replication_gate.json",
) -> dict[str, Any]:
    """Compute the preregistered final-oracle trigger without fitting anything."""

    lock = verify_preregistration()
    lock_path = EXPERIMENT_ROOT / "PREREGISTRATION_LOCK.json"
    lock_sha256 = file_sha256(lock_path)
    config_source = Path(config_path).resolve()
    if config_source != (EXPERIMENT_ROOT / "configs" / "audit.json").resolve():
        raise ValueError("formal S3 trigger requires the frozen audit config")
    config = load_audit_config(config_source)
    equivalence_sha256, replication_sha256 = _validate_p0_gates(
        equivalence_gate_path,
        replication_gate_path,
        preregistration_sha256=lock_sha256,
    )

    probe_root = Path(final_probe_root).resolve()
    if probe_root != (EXPERIMENT_ROOT / "probes" / "final").resolve():
        raise ValueError("formal S3 trigger requires the canonical final probe root")
    final = load_probe_stage(probe_root, stage="final")
    if len(final.identities) != EXPECTED_FINAL_PROBE_CELLS or not final.secondary_present:
        raise ValueError("final probe stage must include all 180 preregistered cells")
    natural = pooled_stage_natural_metrics(final)
    transformed = transformed_fold_summaries(final)
    table2 = build_ftv_table(
        final,
        natural,
        transformed,
        task="static",
        config=config,
    )
    table4 = build_recovery_table(table2)
    oracle = strong_oracle_gate(table4, stage="final", config=config)
    supported = bool(oracle["supported"])
    count, inventory_sha256 = _probe_metadata_inventory(probe_root)
    return {
        "schema_version": 1,
        "status": NOT_TRIGGERED_STATUS if supported else TRIGGERED_STATUS,
        "s3_execution_authorized": not supported,
        "decision_contract": "final_stage_strong_oracle_recovery_false",
        "final_stage_strong_oracle_recovery": oracle,
        "final_probe_root": _relative_to_experiment(probe_root),
        "final_probe_cell_count": count,
        "final_probe_metadata_inventory_sha256": inventory_sha256,
        "preregistration_lock_sha256": lock_sha256,
        "plan_sha256": lock["plan_sha256"],
        "config_sha256": lock["config_sha256"],
        "p0_equivalence_gate_sha256": equivalence_sha256,
        "p0_probe_replication_gate_sha256": replication_sha256,
        "trigger_implementation_sha256": file_sha256(Path(__file__)),
        "analysis_implementation_sha256": file_sha256(
            Path(__file__).with_name("analysis.py")
        ),
        "probe_adapter_sha256": file_sha256(Path(__file__).with_name("probes.py")),
        "new_training_performed": False,
        "probe_refit_performed": False,
        "patient_identifiers_present": False,
    }


def _validate_gate_schema(gate: Mapping[str, Any]) -> None:
    if set(gate) != _GATE_FIELDS:
        raise ValueError(
            "S3 trigger gate schema drifted: "
            f"missing={sorted(_GATE_FIELDS.difference(gate))}, "
            f"extra={sorted(set(gate).difference(_GATE_FIELDS))}"
        )
    oracle = gate.get("final_stage_strong_oracle_recovery")
    if not isinstance(oracle, Mapping) or not isinstance(oracle.get("supported"), bool):
        raise ValueError("S3 trigger gate has malformed final-oracle evidence")
    supported = bool(oracle["supported"])
    expected_status = NOT_TRIGGERED_STATUS if supported else TRIGGERED_STATUS
    if (
        gate.get("schema_version") != 1
        or gate.get("status") != expected_status
        or gate.get("s3_execution_authorized") is not (not supported)
        or gate.get("decision_contract")
        != "final_stage_strong_oracle_recovery_false"
        or int(gate.get("final_probe_cell_count", -1))
        != EXPECTED_FINAL_PROBE_CELLS
        or gate.get("new_training_performed") is not False
        or gate.get("probe_refit_performed") is not False
        or gate.get("patient_identifiers_present") is not False
    ):
        raise ValueError("S3 trigger gate identity/decision contract drifted")


def require_s3_trigger_authorization(
    path: str | Path,
    *,
    verify_live: bool = True,
    require_authorized: bool = True,
) -> dict[str, Any]:
    """Validate an explicit trigger gate and optionally require weak-final status."""

    source = Path(path).resolve()
    gate = _read_json(source, label="S3 trigger authorization")
    _validate_gate_schema(gate)
    if verify_live:
        lock = verify_preregistration()
        lock_path = EXPERIMENT_ROOT / "PREREGISTRATION_LOCK.json"
        live = {
            "preregistration_lock_sha256": file_sha256(lock_path),
            "plan_sha256": lock["plan_sha256"],
            "config_sha256": lock["config_sha256"],
            "trigger_implementation_sha256": file_sha256(Path(__file__)),
            "analysis_implementation_sha256": file_sha256(
                Path(__file__).with_name("analysis.py")
            ),
            "probe_adapter_sha256": file_sha256(Path(__file__).with_name("probes.py")),
            "p0_equivalence_gate_sha256": file_sha256(
                EXPERIMENT_ROOT / "metrics" / "p0_equivalence_gate.json"
            ),
            "p0_probe_replication_gate_sha256": file_sha256(
                EXPERIMENT_ROOT / "metrics" / "p0_probe_replication_gate.json"
            ),
        }
        for field, expected in live.items():
            if gate.get(field) != expected:
                raise ValueError(f"live S3 trigger provenance drifted at {field}")
        probe_root = (EXPERIMENT_ROOT / str(gate["final_probe_root"])).resolve()
        if probe_root != (EXPERIMENT_ROOT / "probes" / "final").resolve():
            raise ValueError("S3 trigger final probe root drifted")
        # Re-run the existing read-only stage validator before trusting the
        # compact inventory digest.  This also proves every sibling CSV hash.
        final = load_probe_stage(probe_root, stage="final")
        if len(final.identities) != EXPECTED_FINAL_PROBE_CELLS:
            raise ValueError("live final probe matrix is no longer 180 cells")
        count, digest = _probe_metadata_inventory(probe_root)
        if (
            count != int(gate["final_probe_cell_count"])
            or digest != gate["final_probe_metadata_inventory_sha256"]
        ):
            raise ValueError("live final probe inventory changed after S3 authorization")
    if require_authorized and gate.get("s3_execution_authorized") is not True:
        raise PermissionError(
            "conditional S3 is not authorized because final-stage oracle is strong"
        )
    return gate


def write_s3_trigger_gate(path: str | Path, gate: Mapping[str, Any]) -> Path:
    """Atomically publish the aggregate-only gate as a public 0644 JSON file."""

    _validate_gate_schema(gate)
    destination = Path(path).resolve()
    if destination.name != S3_TRIGGER_GATE_FILENAME:
        raise ValueError(f"S3 trigger gate must be named {S3_TRIGGER_GATE_FILENAME}")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite S3 trigger gate: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(dict(gate), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_path, 0o644)
        os.link(temporary_path, destination)
        destination.chmod(0o644)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


__all__ = [
    "EXPECTED_FINAL_PROBE_CELLS",
    "NOT_TRIGGERED_STATUS",
    "S3_TRIGGER_GATE_FILENAME",
    "TRIGGERED_STATUS",
    "compute_s3_trigger_gate",
    "require_s3_trigger_authorization",
    "write_s3_trigger_gate",
]

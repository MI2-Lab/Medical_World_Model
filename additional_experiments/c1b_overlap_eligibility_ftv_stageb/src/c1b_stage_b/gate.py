"""Fail-closed Stage A authorization gate for every Stage B entry point."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .contracts import (
    EXPERIMENT_ROOT,
    REQUIRED_STAGE_A_GATES,
    TIMEPOINTS,
    file_sha256,
    require_sha256,
)


DEFAULT_STAGE_A_SENTINEL = EXPERIMENT_ROOT / "STAGE_A_GO.json"


class StageAGateError(RuntimeError):
    """Raised before Stage B can read caches, create outputs, or train."""


@dataclass(frozen=True)
class StageAAuthorization:
    path: Path
    sha256: str
    payload: Mapping[str, Any]
    eligible_population_patients: int
    eligible_population_visits: int
    technical_eligibility_manifest_sha256: str


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise StageAGateError(f"Stage B blocked: {label} must be a positive integer")
    try:
        observed = int(value)
    except (TypeError, ValueError) as error:
        raise StageAGateError(
            f"Stage B blocked: {label} must be a positive integer"
        ) from error
    if observed <= 0 or isinstance(value, float) and not value.is_integer():
        raise StageAGateError(f"Stage B blocked: {label} must be a positive integer")
    return observed


def _eligibility_manifest_sha256(payload: Mapping[str, Any]) -> str:
    candidates: list[Any] = [
        payload.get("technical_eligibility_manifest_sha256"),
        payload.get("eligibility_manifest_sha256"),
    ]
    for container_name in (
        "provenance_sha256",
        "private_manifest_sha256",
        "private_artifact_sha256",
    ):
        container = payload.get(container_name)
        if isinstance(container, Mapping):
            candidates.extend(
                value
                for key, value in container.items()
                if str(key).endswith("technical_eligibility_patients.private.csv")
            )
    for value in candidates:
        if value is None:
            continue
        try:
            return require_sha256(str(value), "technical eligibility manifest")
        except ValueError:
            continue
    raise StageAGateError(
        "Stage B blocked: Stage A does not pin the technical eligibility patient manifest"
    )


def require_stage_a_go(path: str | Path = DEFAULT_STAGE_A_SENTINEL) -> StageAAuthorization:
    sentinel = Path(path).expanduser().resolve()
    if not sentinel.is_file():
        raise StageAGateError(f"Stage B blocked: Stage A GO sentinel is missing: {sentinel}")
    try:
        payload = json.loads(sentinel.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StageAGateError("Stage B blocked: Stage A sentinel is unreadable") from error
    if not isinstance(payload, Mapping):
        raise StageAGateError("Stage B blocked: Stage A sentinel must be a JSON object")
    if int(payload.get("schema_version", -1)) != 1:
        raise StageAGateError("Stage B blocked: unsupported Stage A sentinel schema")
    if str(payload.get("stage", "")).upper() != "A":
        raise StageAGateError("Stage B blocked: sentinel is not for Stage A")
    if str(payload.get("status", "")).upper() != "GO":
        raise StageAGateError("Stage B blocked: Stage A status is not GO")
    if payload.get("stage_b_authorized") is not True:
        raise StageAGateError("Stage B blocked: sentinel does not authorize Stage B")
    if payload.get("thresholds_relaxed") is not False:
        raise StageAGateError("Stage B blocked: Stage A thresholds were relaxed or unspecified")
    if payload.get("eligibility_rule_frozen_before_stage_b") is not True:
        raise StageAGateError(
            "Stage B blocked: the new eligibility rule was not frozen before Stage B"
        )
    if str(payload.get("chosen_input_strategy", "")) != "C1B-H":
        raise StageAGateError("Stage B blocked: Stage A did not freeze C1B-H")
    eligible_patients = _positive_integer(
        payload.get("eligible_population_patients"), "eligible_population_patients"
    )
    eligible_visits = _positive_integer(
        payload.get("eligible_population_visits"), "eligible_population_visits"
    )
    if eligible_visits != len(TIMEPOINTS) * eligible_patients:
        raise StageAGateError(
            "Stage B blocked: eligible visit count is not four visits per eligible patient"
        )
    try:
        completion = float(payload.get("cache_completion_fraction"))
    except (TypeError, ValueError) as error:
        raise StageAGateError(
            "Stage B blocked: Stage A cache completion is missing"
        ) from error
    if not math.isfinite(completion) or completion != 1.0:
        raise StageAGateError(
            "Stage B blocked: Stage A cache completion is not exactly 100%"
        )
    eligibility_sha256 = _eligibility_manifest_sha256(payload)
    gates = payload.get("gates")
    if not isinstance(gates, list) or len(gates) != len(REQUIRED_STAGE_A_GATES):
        raise StageAGateError(
            "Stage B blocked: Stage A sentinel does not contain exactly the 15 "
            "preregistered gate checks"
        )
    if any(
        not isinstance(gate, Mapping) or str(gate.get("status", "")).upper() != "PASS"
        for gate in gates
    ):
        raise StageAGateError("Stage B blocked: Stage A required gates are not all PASS")
    gate_names = tuple(str(gate.get("gate", "")).strip() for gate in gates)
    gate_items = tuple(gate.get("item") for gate in gates)
    if gate_names != REQUIRED_STAGE_A_GATES or gate_items != tuple(range(1, 16)):
        raise StageAGateError(
            "Stage B blocked: Stage A gate identities/order differ from the "
            "15 preregistered checks"
        )
    return StageAAuthorization(
        sentinel,
        file_sha256(sentinel),
        dict(payload),
        eligible_patients,
        eligible_visits,
        eligibility_sha256,
    )

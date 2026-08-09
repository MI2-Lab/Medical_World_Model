"""Fail-closed Stage A authorization gate for every Stage B entry point."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import EXPERIMENT_ROOT, file_sha256


DEFAULT_STAGE_A_SENTINEL = EXPERIMENT_ROOT / "STAGE_A_GO.json"


class StageAGateError(RuntimeError):
    """Raised before Stage B can read caches, create outputs, or train."""


@dataclass(frozen=True)
class StageAAuthorization:
    path: Path
    sha256: str
    payload: Mapping[str, Any]


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
    gates = payload.get("gates")
    if not isinstance(gates, list) or not gates:
        raise StageAGateError("Stage B blocked: Stage A sentinel has no gate evidence")
    if any(
        not isinstance(gate, Mapping) or str(gate.get("status", "")).upper() != "PASS"
        for gate in gates
    ):
        raise StageAGateError("Stage B blocked: Stage A required gates are not all PASS")
    return StageAAuthorization(sentinel, file_sha256(sentinel), dict(payload))

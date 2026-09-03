from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "experiment.json"
LOCK_PATH = ROOT / "PREREGISTRATION_LOCK.json"
SEEDS = (2026, 3026)
FOLDS = (0, 1, 2, 3, 4)
TIMINGS = ("T0", "T0_T1", "T0_T2", "T0_T3")
PRIMARY_TIMINGS = ("T0", "T0_T1", "T0_T2")
ARMS = ("C0", "C1", "C2", "C3", "C4", "C5")


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class ExperimentContract:
    config: Mapping[str, Any]
    lock: Mapping[str, Any]
    config_sha256: str
    lock_sha256: str

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(self.config["seeds"])

    @property
    def folds(self) -> tuple[int, ...]:
        return tuple(self.config["folds"])

    @property
    def arms(self) -> tuple[str, ...]:
        return tuple(self.config["arms"])

    @property
    def primary_timings(self) -> tuple[str, ...]:
        return tuple(self.config["primary_timings"])

    def timing_steps(self, timing: str) -> int:
        _require(timing in TIMINGS, f"unknown timing: {timing}")
        return {"T0": 1, "T0_T1": 2, "T0_T2": 3, "T0_T3": 4}[timing]

    def resolve_private_root(self, explicit: str | Path | None = None) -> Path:
        import os

        value = explicit or os.environ.get("MWM_PRIVATE_INPUT_REPO_ROOT")
        _require(value is not None, "set MWM_PRIVATE_INPUT_REPO_ROOT or pass --private-root")
        root = Path(value).expanduser().resolve(strict=True)
        _require(root.is_dir(), "private input root must be a directory")
        return root


def validate_config(config: Mapping[str, Any]) -> None:
    required = {"schema_version", "experiment", "branch", "seeds", "folds", "timings", "primary_timings", "supplementary_timing_label", "input", "populations", "arms", "arm_contract", "attention", "transformer", "optimization", "fusion", "bootstrap", "privacy"}
    _require(set(config) == required, f"config keys drifted: expected {sorted(required)}")
    _require(config["schema_version"] == 1, "unsupported config schema")
    _require(config["experiment"] == "raw_spatial_pcr_ceiling", "wrong experiment")
    _require(config["branch"] == "feature/raw-spatial-pcr-ceiling", "wrong branch contract")
    _require(tuple(config["seeds"]) == SEEDS, "seeds are frozen to 2026 and 3026")
    _require(tuple(config["folds"]) == FOLDS, "five outer folds are required")
    _require(tuple(config["timings"]) == TIMINGS, "timing contract drifted")
    _require(tuple(config["primary_timings"]) == PRIMARY_TIMINGS, "primary timing contract drifted")
    inp = config["input"]
    _require(inp["modality"] == "C1B-H DCE7" and inp["channels"] == 7 and inp["visits"] == 4, "C1B-H DCE7 input contract drifted")
    _require(inp["shape_zyx"] == [112, 176, 160], "C1B geometry drifted")
    _require(inp["local_support_mm"] == 64, "LOCAL support drifted")
    _require(tuple(config["arms"]) == ARMS, "arm order drifted")
    _require(config["attention"] == {"query_tokens": 1, "blocks": 2, "heads": 4, "width": 128, "dropout": 0.1}, "attention contract drifted")
    _require(config["transformer"]["blocks"] == 3 and config["transformer"]["heads"] == 4 and config["transformer"]["width"] == 128, "transformer contract drifted")
    _require(config["optimization"]["loss"] == "BCEWithLogitsLoss", "primary loss drifted")
    _require(config["optimization"]["selection_scope"] == "outer_train_validation_only", "selection scope drifted")
    _require(config["fusion"]["clinical_inside_mri_branch"] is False, "clinical variables leaked into MRI branch")
    _require(config["bootstrap"]["draws"] >= 5000 and config["bootstrap"]["unit"] == "paired_patient_within_outer_fold", "bootstrap contract drifted")


def validate_lock(lock: Mapping[str, Any]) -> None:
    _require(lock["schema_version"] == 1, "unsupported lock schema")
    _require(lock["experiment"] == "raw_spatial_pcr_ceiling", "wrong lock experiment")
    _require(lock["branch"] == "feature/raw-spatial-pcr-ceiling", "wrong lock branch")
    _require(tuple(lock["population_contract"]["seeds"]) == SEEDS, "lock seed contract drifted")
    _require(tuple(lock["population_contract"]["folds"]) == FOLDS, "lock fold contract drifted")
    _require(lock["training_contract"]["loss"] == "BCEWithLogitsLoss", "lock loss drifted")
    _require(lock["training_contract"]["clinical_in_mri_branch"] is False, "lock clinical leakage contract drifted")
    _require(lock["metrics_contract"]["bootstrap_draws"] >= 5000, "lock bootstrap count too small")


def load_contract(config_path: str | Path = CONFIG_PATH, lock_path: str | Path = LOCK_PATH) -> ExperimentContract:
    config_source = Path(config_path).resolve(strict=True)
    lock_source = Path(lock_path).resolve(strict=True)
    config = json.loads(config_source.read_text(encoding="utf-8"))
    lock = json.loads(lock_source.read_text(encoding="utf-8"))
    validate_config(config)
    validate_lock(lock)
    _require(lock["branch"] == config["branch"], "config/lock branch mismatch")
    return ExperimentContract(config, lock, file_sha256(config_source), file_sha256(lock_source))

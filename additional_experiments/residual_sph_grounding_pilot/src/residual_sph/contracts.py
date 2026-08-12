"""Immutable scientific and runtime contracts for the residual-SPH pilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import operator
from pathlib import Path
from typing import Any, Mapping


SEED_BASES = (2026, 3026)
FOLDS = tuple(range(5))
PRIMARY_ARMS = ("S0", "S1", "S2")
SENSITIVITY_ARMS = ("S2_L10",)
ARMS = PRIMARY_ARMS + SENSITIVITY_ARMS
VISITS = ("T0", "T1", "T2", "T3")
RESPONSE_DIM = 192
FTV_WEIGHT = 0.25
SIGREG_WEIGHT = 0.09
SPH_HEAD_PARAMETER_COUNT = RESPONSE_DIM + 1
LOGICAL_BATCH_SIZE = 32


@dataclass(frozen=True)
class ArmSpec:
    name: str
    sph_target: str | None
    lambda_sph: float
    primary: bool

    def __post_init__(self) -> None:
        if self.name not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}")
        if self.sph_target not in (None, "raw_sph_z", "ftv_residual_sph_z"):
            raise ValueError("unknown SPH target")
        if not math.isfinite(self.lambda_sph) or self.lambda_sph < 0:
            raise ValueError("SPH loss weight must be finite and nonnegative")
        if (self.sph_target is None) != (self.lambda_sph == 0):
            raise ValueError("SPH target/head and loss weight disagree")

    @property
    def has_sph_head(self) -> bool:
        return self.sph_target is not None


ARM_SPECS: Mapping[str, ArmSpec] = {
    "S0": ArmSpec("S0", None, 0.0, True),
    "S1": ArmSpec("S1", "raw_sph_z", 0.05, True),
    "S2": ArmSpec("S2", "ftv_residual_sph_z", 0.05, True),
    "S2_L10": ArmSpec("S2_L10", "ftv_residual_sph_z", 0.10, False),
}


def arm_spec(arm: str) -> ArmSpec:
    name = str(arm).strip().upper()
    try:
        return ARM_SPECS[name]
    except KeyError as error:
        raise ValueError(f"arm must be one of {ARMS}; got {arm!r}") from error


def validate_seed_fold(seed_base: int, fold: int) -> int:
    try:
        seed = operator.index(seed_base)
        fold_value = operator.index(fold)
    except TypeError as error:
        raise ValueError("seed and fold must be exact integers") from error
    if isinstance(seed_base, bool) or seed not in SEED_BASES:
        raise ValueError(f"seed_base must be one of {SEED_BASES}")
    if isinstance(fold, bool) or fold_value not in FOLDS:
        raise ValueError(f"fold must be one of {FOLDS}")
    return int(seed + fold_value)


@dataclass(frozen=True)
class TrainHyperparameters:
    physical_batch_size: int = 4
    accumulation_steps: int = 8
    workers: int = 2
    epochs: int = 12
    patience: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 1e-4
    ema_momentum: float = 0.996
    max_grad_norm: float = 5.0
    min_representation_std: float = 0.05

    def validate(self) -> None:
        if (self.physical_batch_size, self.accumulation_steps) != (4, 8):
            raise ValueError("formal training is frozen to physical B4 x accumulation 8")
        if self.physical_batch_size * self.accumulation_steps != LOGICAL_BATCH_SIZE:
            raise ValueError("formal logical batch must be 32")
        if self.workers < 0 or self.epochs != 12 or self.patience != 4:
            raise ValueError("formal worker/epoch/patience contract drifted")
        exact = {
            "learning_rate": (self.learning_rate, 5e-5),
            "weight_decay": (self.weight_decay, 1e-4),
            "ema_momentum": (self.ema_momentum, 0.996),
            "max_grad_norm": (self.max_grad_norm, 5.0),
            "min_representation_std": (self.min_representation_std, 0.05),
        }
        for name, (observed, expected) in exact.items():
            if float(observed) != expected:
                raise ValueError(f"formal {name} is frozen to {expected}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


FORBIDDEN_REPRESENTATION_TOKENS = (
    "pcr",
    "hr_status",
    "her2",
    "clinical",
    "treatment",
    "response_label",
)


def assert_representation_schema(keys: object) -> None:
    names = tuple(str(value).casefold() for value in keys)
    bad = sorted(
        name
        for name in names
        if any(token in name for token in FORBIDDEN_REPRESENTATION_TOKENS)
    )
    if bad:
        raise PermissionError(f"forbidden representation-training fields: {bad}")


CHECKPOINT_SELECTION_RULE = (
    "minimum validation FTV loss among finite non-collapsed epochs whose "
    "validation state loss is <=1.05 times the paired S0 value; SPH, test, "
    "delta-SPH, pCR, HR, HER2, clinical and treatment values do not select"
)


__all__ = [
    "ARMS",
    "ARM_SPECS",
    "CHECKPOINT_SELECTION_RULE",
    "FOLDS",
    "FTV_WEIGHT",
    "LOGICAL_BATCH_SIZE",
    "PRIMARY_ARMS",
    "RESPONSE_DIM",
    "SEED_BASES",
    "SENSITIVITY_ARMS",
    "SIGREG_WEIGHT",
    "SPH_HEAD_PARAMETER_COUNT",
    "TrainHyperparameters",
    "VISITS",
    "ArmSpec",
    "arm_spec",
    "assert_representation_schema",
    "canonical_sha256",
    "file_sha256",
    "validate_seed_fold",
]

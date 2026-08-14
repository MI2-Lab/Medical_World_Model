"""Frozen public constants and validation helpers for Goal F."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ARMS = ("F1", "F2", "F2S")
PRIMARY_ARMS = ("F1", "F2")
SEED_BASES = (2026, 3026)
FOLDS = tuple(range(5))
VISITS = ("T0", "T1", "T2", "T3")

C1B_SHAPE_ZYX = (112, 176, 160)
C1B_SPACING_XYZ_MM = (0.9, 0.9, 2.0)
LOCAL_WINDOW_MM_XYZ = (64.0, 64.0, 64.0)
IMAGE_CHANNELS = 7
FEATURE_CHANNELS = 128
RESPONSE_DIM = 96
PHENOTYPE_DIM = 96
STATE_DIM = RESPONSE_DIM + PHENOTYPE_DIM

PCR_LABEL_ACCESS = "FORBIDDEN"
TRAINING_PROFILE_COLUMNS = (
    "patient_id",
    "label_hr",
    "label_her2",
    "label_mp",
    "arm",
)
PROFILE_SOURCE_USECOLS = TRAINING_PROFILE_COLUMNS
TECHNICAL_ELIGIBILITY_USECOLS = ("patient_id", "eligible")
FORBIDDEN_TRAINING_COLUMN_TOKENS = ("pcr", "rcb", "outcome", "response")
LOCKED_ISPY2_PROFILE_SHA256 = (
    "b3355f8ac80cf8f0fa95722b8d8a8b73d96790e9ded5c491ddb5b2e6a7793436"
)
LOCKED_ISPY1_PROFILE_SHA256 = (
    "7301e6d43ce2c8aa4f45a56fa43f065c4a5c0a119f1735e3d2a540337940e4fd"
)
LOCKED_TECHNICAL_ELIGIBILITY_SHA256 = (
    "041f11877bb55d70a611e14a811f5610a3572c4437470bbde54cbdf19121f2c7"
)
EXPECTED_TRAINING_PROFILE_PATIENTS = 947


@dataclass(frozen=True)
class ArmSpec:
    name: str
    adversarial_weight: float
    primary: bool

    @property
    def uses_adversary(self) -> bool:
        return self.adversarial_weight > 0.0


ARM_SPECS: Mapping[str, ArmSpec] = {
    "F1": ArmSpec("F1", 0.0, True),
    "F2": ArmSpec("F2", 0.05, True),
    "F2S": ArmSpec("F2S", 0.01, False),
}


def arm_spec(value: str) -> ArmSpec:
    name = str(value).strip().upper()
    if name not in ARM_SPECS:
        raise ValueError(f"arm must be one of {ARMS}; got {value!r}")
    return ARM_SPECS[name]


def validate_seed_fold(seed_base: int, fold: int) -> int:
    if isinstance(seed_base, bool) or int(seed_base) not in SEED_BASES:
        raise ValueError(f"seed_base must be one of {SEED_BASES}")
    if isinstance(fold, bool) or int(fold) not in FOLDS:
        raise ValueError(f"fold must be one of {FOLDS}")
    return int(seed_base) + int(fold)


def validate_geometry(
    shape_zyx: Sequence[int],
    spacing_xyz_mm: Sequence[float],
    local_window_mm_xyz: Sequence[float],
) -> None:
    if tuple(int(value) for value in shape_zyx) != C1B_SHAPE_ZYX:
        raise ValueError("C1B-H image shape drifted")
    if tuple(float(value) for value in spacing_xyz_mm) != C1B_SPACING_XYZ_MM:
        raise ValueError("C1B-H spacing drifted")
    if tuple(float(value) for value in local_window_mm_xyz) != LOCAL_WINDOW_MM_XYZ:
        raise ValueError("fixed LOCAL support drifted")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def assert_representation_config(config: Mapping[str, Any]) -> None:
    if config.get("experiment") != "clinical_residual_phenotype_state":
        raise ValueError("wrong representation experiment config")
    if config.get("phase") != "representation_training":
        raise ValueError("representation config phase drifted")
    if config.get("PCR_LABEL_ACCESS") != PCR_LABEL_ACCESS:
        raise PermissionError("PCR_LABEL_ACCESS must be FORBIDDEN")
    if int(config["state"]["response_dim"]) != RESPONSE_DIM:
        raise ValueError("response state dimension drifted")
    if int(config["state"]["phenotype_dim"]) != PHENOTYPE_DIM:
        raise ValueError("phenotype state dimension drifted")
    if int(config["state"]["total_dim"]) != STATE_DIM:
        raise ValueError("total state dimension drifted")
    if int(config["state"]["phenotype_queries"]) != 1:
        raise ValueError("primary pilot requires exactly one phenotype query")
    columns = tuple(config["profiles"]["allowed_columns"])
    if columns != TRAINING_PROFILE_COLUMNS:
        raise PermissionError("training profile allowlist drifted")
    lowered = tuple(column.casefold() for column in columns)
    if any(token in column for token in FORBIDDEN_TRAINING_COLUMN_TOKENS for column in lowered):
        raise PermissionError("training profile allowlist contains an outcome-like column")
    profiles = config["profiles"]
    if profiles.get("ispy2_sha256") != LOCKED_ISPY2_PROFILE_SHA256:
        raise PermissionError("I-SPY2 profile source SHA-256 lock drifted")
    if profiles.get("ispy1_sha256") != LOCKED_ISPY1_PROFILE_SHA256:
        raise PermissionError("I-SPY1 profile source SHA-256 lock drifted")
    if (
        profiles.get("technical_eligibility_sha256")
        != LOCKED_TECHNICAL_ELIGIBILITY_SHA256
    ):
        raise PermissionError("technical eligibility SHA-256 lock drifted")
    if int(profiles.get("expected_patient_count", -1)) != EXPECTED_TRAINING_PROFILE_PATIENTS:
        raise ValueError("training profile population count drifted")
    if not str(profiles.get("training_manifest_path", "")).endswith(".private.csv"):
        raise PermissionError("formal training profile path must be private")
    manifest_sha256 = profiles.get("training_manifest_sha256")
    if manifest_sha256 not in (None, "", "PENDING", "PENDING_PRIVATE_PROJECTION"):
        normalized_manifest_sha256 = str(manifest_sha256).strip().casefold()
        if len(normalized_manifest_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in normalized_manifest_sha256
        ):
            raise ValueError("training profile manifest SHA-256 is invalid")
    validate_geometry(
        config["input"]["tensor_shape"][-3:],
        config["input"]["spacing_xyz_mm"],
        config["input"]["local_window_mm_xyz"],
    )
    if tuple(config["training"]["seed_bases"]) != SEED_BASES:
        raise ValueError("training seeds drifted")
    if tuple(config["training"]["folds"]) != FOLDS:
        raise ValueError("folds drifted")
    for name, spec in ARM_SPECS.items():
        record = config["arms"][name]
        if float(record["adversarial_weight"]) != spec.adversarial_weight:
            raise ValueError(f"{name} adversarial weight drifted")


__all__ = [
    "ARMS",
    "ARM_SPECS",
    "C1B_SHAPE_ZYX",
    "C1B_SPACING_XYZ_MM",
    "FEATURE_CHANNELS",
    "FOLDS",
    "FORBIDDEN_TRAINING_COLUMN_TOKENS",
    "IMAGE_CHANNELS",
    "EXPECTED_TRAINING_PROFILE_PATIENTS",
    "LOCKED_ISPY1_PROFILE_SHA256",
    "LOCKED_ISPY2_PROFILE_SHA256",
    "LOCKED_TECHNICAL_ELIGIBILITY_SHA256",
    "LOCAL_WINDOW_MM_XYZ",
    "PCR_LABEL_ACCESS",
    "PHENOTYPE_DIM",
    "PROFILE_SOURCE_USECOLS",
    "PRIMARY_ARMS",
    "RESPONSE_DIM",
    "SEED_BASES",
    "STATE_DIM",
    "TECHNICAL_ELIGIBILITY_USECOLS",
    "TRAINING_PROFILE_COLUMNS",
    "VISITS",
    "ArmSpec",
    "arm_spec",
    "assert_representation_config",
    "canonical_sha256",
    "file_sha256",
    "load_json",
    "validate_geometry",
    "validate_seed_fold",
]

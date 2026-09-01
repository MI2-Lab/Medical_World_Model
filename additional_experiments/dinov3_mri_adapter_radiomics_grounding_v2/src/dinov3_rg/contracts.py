"""Frozen constants, paths, hashes, and fail-closed protocol validation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping


PACKAGE_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = PACKAGE_ROOT.parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[3]
PROTOCOL_PATH = EXPERIMENT_ROOT / "configs/protocol.json"

STAGE_B_ROOT = REPO_ROOT / "additional_experiments/c1b_overlap_eligibility_ftv_stageb"
C1B_SOURCE_ROOT = REPO_ROOT / "additional_experiments/c1b_model_ready_ftv_sanity"
CACHE_MANIFEST = STAGE_B_ROOT / "manifests/stage_b_c1b_cache.private.csv"
FOLD_MANIFEST = Path(
    "/data/data/Preprocessed/I-SPY2/"
    "_matched_breastdcedl_t0_dicomrepair_rgb224_seed2026/"
    "matched_patient_cv_splits_seed2026.csv"
)
TECHNICAL_ELIGIBILITY = STAGE_B_ROOT / "manifests/technical_eligibility_patients.private.csv"
TRAIN_ONLY_MANIFEST = C1B_SOURCE_ROOT / "manifests/ispy1_base_eligibility_patients.private.csv"
FTV_TRANSITIONS = (
    REPO_ROOT
    / "additional_experiments/radiomics_next_change/data_audit/"
    "radiomics_transition_targets_raw.csv"
)
SOURCE_INVENTORY = C1B_SOURCE_ROOT / "manifests/model_input_inventory.private.csv"

LOCKED_HASHES: Mapping[str, str] = {
    "cache_manifest": "672ad7436b19f30a89640a2b36504f1e7fbaaff83fd07bc058c008b204d2a3c9",
    "fold_manifest": "143e482d711225c0611006d99bd7345d2fa1a5c16c65fbaf8399341a0d26aa38",
    "ftv_transitions": "26fbde8590fde4612267f02d762af99d65926ff6d0206d0e500577ef394ff75d",
    "source_inventory": "40f7d1db7af108b2cb0c07863db208ee4db2124ffaf660771feaa7270809df0a",
    "train_only_manifest": "d9029507607dcdb327db8066f7fd7a712d10e157e4d2b8f79ab3c9f0daad6513",
    "technical_eligibility": "041f11877bb55d70a611e14a811f5610a3572c4437470bbde54cbdf19121f2c7",
}

VISITS = ("T0", "T1", "T2", "T3")
SEEDS = (2026, 3026, 4026, 5026, 6026)
FOLDS = tuple(range(5))
ARMS = ("D1", "D2", "D3")
PROJECTION_SEEDS = (260812, 260813, 260814, 260815, 260816)
SUMMARY_SHAPE = (4, 7, 32, 2304)
STATE_SHAPE = (4, 192)
PRIMARY_PATIENTS = 375
FOLD_PATIENTS = 808
TRAIN_ONLY_PATIENTS = 139
COHORT_PATIENTS = 947


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def patient_order_sha256(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode("utf-8")).hexdigest()


def private_patient_token(patient_id: str) -> str:
    return hashlib.sha256(str(patient_id).encode("utf-8")).hexdigest()


def atomic_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def load_protocol(path: str | Path = PROTOCOL_PATH) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("protocol schema must be 1")
    if payload.get("experiment") != "dinov3_mri_adapter_radiomics_grounding_v2":
        raise ValueError("V2 experiment identity drifted")
    if tuple(payload["training"]["seeds"]) != SEEDS:
        raise ValueError("training seeds drifted")
    if tuple(payload["training"]["folds"]) != FOLDS:
        raise ValueError("folds drifted")
    if tuple(payload["training"]["arms"]) != ARMS:
        raise ValueError("arms drifted")
    if tuple(payload["data"]["visits"]) != VISITS:
        raise ValueError("visit order drifted")
    if tuple(payload["data"]["local_shape_zyx"]) != (32, 72, 72):
        raise ValueError("LOCAL geometry drifted")
    if int(payload["dinov3"]["summary_dim"]) != SUMMARY_SHAPE[-1]:
        raise ValueError("DINO summary dimension drifted")
    if tuple(payload["radiomics"]["grounding_visits"]) != VISITS[:3]:
        raise ValueError("V2 radiomics grounding visits drifted")
    if int(payload["radiomics"]["minimum_voxels"]) != 64:
        raise ValueError("V2 radiomics minimum ROI drifted")
    if int(payload["radiomics"]["pyradiomics_minimum_roi_size_setting"]) != 63:
        raise ValueError("PyRadiomics inclusive-64 compatibility setting drifted")
    if float(payload["radiomics"]["coverage_minimum"]["T2"]) != 0.85:
        raise ValueError("V2 T2 coverage gate drifted")
    if payload["loss"]["ftv"] != {"D1": 0.0, "D2": 0.25, "D3": 0.25}:
        raise ValueError("FTV arm weights drifted")
    if payload["loss"]["radiomics"] != {"D1": 0.0, "D2": 0.0, "D3": 0.1}:
        raise ValueError("radiomics arm weights drifted")
    return payload


def validate_seed_fold_arm(seed: int, fold: int, arm: str) -> tuple[int, int, str]:
    if isinstance(seed, bool) or int(seed) not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    if isinstance(fold, bool) or int(fold) not in FOLDS:
        raise ValueError("fold must be 0..4")
    normalized = str(arm).upper()
    if normalized not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    return int(seed), int(fold), normalized


def verify_locked_file(path: str | Path, expected: str, label: str) -> str:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"locked {label} is missing: {source}")
    observed = file_sha256(source)
    if observed != expected:
        raise ValueError(f"locked {label} hash mismatch: {observed} != {expected}")
    return observed


def expected_cells() -> tuple[str, ...]:
    return tuple(f"seed{seed}_fold{fold}_{arm}" for seed in SEEDS for fold in FOLDS for arm in ARMS)


__all__ = [
    "ARMS", "CACHE_MANIFEST", "COHORT_PATIENTS", "EXPERIMENT_ROOT", "FOLDS",
    "FOLD_MANIFEST", "FOLD_PATIENTS", "FTV_TRANSITIONS", "LOCKED_HASHES",
    "PRIMARY_PATIENTS", "PROJECTION_SEEDS", "REPO_ROOT", "SEEDS", "TECHNICAL_ELIGIBILITY",
    "SOURCE_INVENTORY", "STATE_SHAPE", "SUMMARY_SHAPE", "TRAIN_ONLY_MANIFEST",
    "TRAIN_ONLY_PATIENTS", "VISITS", "atomic_json", "canonical_sha256",
    "expected_cells", "file_sha256", "load_protocol", "patient_order_sha256",
    "private_patient_token", "validate_seed_fold_arm", "verify_locked_file"
]

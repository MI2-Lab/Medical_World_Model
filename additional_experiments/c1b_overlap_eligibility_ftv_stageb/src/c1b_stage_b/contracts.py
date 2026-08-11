"""Frozen Stage B arm, batching, split, eligibility, and hashing contracts.

This package deliberately contains no clinical-label adapter.  Model-facing
data are DCE7 tensors; the only measurement table admitted by Stage B is FTV.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
G3_ROOT = REPO_ROOT / "additional_experiments" / "g3_multiseed_generalization"
G3_SRC = G3_ROOT / "src"
PRIOR_C1B_ROOT = REPO_ROOT / "additional_experiments" / "c1b_model_ready_ftv_sanity"
PRIOR_C1B_SRC = PRIOR_C1B_ROOT / "src"

TIMEPOINTS = ("T0", "T1", "T2", "T3")
TRANSITIONS = ("T0→T1", "T1→T2", "T2→T3")
SEED_BASES = (2026, 3026)
FOLDS = tuple(range(5))
ARMS = ("L1", "L3", "N1", "N3")
REQUIRED_STAGE_A_GATES = (
    "eligibility_rule_frozen_before_stage_b",
    "eligibility_outcome_free_and_public_private",
    "eligible_cohort_mechanically_determined",
    "all_eligible_visits_positive_overlap",
    "dicom_repair_contract",
    "true_ras_orientation_contract",
    "c1b_h_strategy_frozen",
    "formal_ftv_support_containment",
    "ftv_retention_q05",
    "grounding_observability_loss_side_only",
    "complete_dce7_cache",
    "cache_roundtrip_and_hash",
    "no_patient_specific_manual_correction",
    "no_unresolved_catastrophic_resampling",
    "geometry_metadata_excluded_from_model_tensor",
)
LOCKED_SEED_2026_FOLD_MANIFEST_SHA256 = (
    "143e482d711225c0611006d99bd7345d2fa1a5c16c65fbaf8399341a0d26aa38"
)
LOCKED_TRAIN_ONLY_SOURCE_MANIFEST_SHA256 = (
    "d9029507607dcdb327db8066f7fd7a712d10e157e4d2b8f79ab3c9f0daad6513"
)
LOCKED_FTV_TRANSITION_TABLE_SHA256 = (
    "26fbde8590fde4612267f02d762af99d65926ff6d0206d0e500577ef394ff75d"
)
LOCKED_OBSERVABILITY_MANIFEST_SHA256 = (
    "0203da10f2efee626ae948fe0f7190771b67503639694dcd49c03762f4bde39c"
)
LOCKED_G3_TARGETS_SHA256 = (
    "28fbf66f93c8541dfa5ecc7ebcf65d4143a9a605b3ce98be48355d5ab679ffac"
)
LOCKED_G3_MODEL_SHA256 = (
    "ce39878a0fef5af1f92a86811faabbe73b39f57cdaf6d7580bbd65bd855d4ed9"
)
LOCKED_G3_OBJECTIVE_SHA256 = (
    "76f9108df0ca8c0ff69e514cff3bab1d5e316d946da60c5f530dd7b9706d3815"
)
LOCKED_G3_DATA_SHA256 = (
    "15b4b68ad45c935e313b893b0ce849877311c98d6c5c0c45495e8e9200240943"
)
LOCKED_C1B_CACHE_CONTRACT_SHA256 = (
    "6bc36a4d8dac356ed4d6f0fc5858607a7bbd4845a66fc83b214025d22f1f2af7"
)

EFFECTIVE_BATCH_SIZE = 32
PREFERRED_PHYSICAL_BATCH_SIZE = 4
PREFERRED_ACCUMULATION_STEPS = 8
FALLBACK_PHYSICAL_BATCH_SIZE = 2
FALLBACK_ACCUMULATION_STEPS = 16

FOLD_USECOLS = ("patient_id", "fold", "split")
FTV_TRANSITION_USECOLS = (
    "patient_id",
    "transition",
    "start_visit",
    "end_visit",
    "ftv_start",
    "ftv_end",
    "ftv_valid",
)
OBSERVABILITY_USECOLS = (
    "patient_id",
    "visit",
    "ftv_measurement_valid",
    "grounding_observable_mask",
)
CACHE_MANIFEST_USECOLS = (
    "patient_id",
    "cache_path",
    "cache_sha256",
    "cache_size_bytes",
    "cache_mtime_ns",
    "input_kind",
)
TECHNICAL_ELIGIBILITY_USECOLS = ("patient_id", "cohort", "eligible")
TRAIN_ONLY_CANDIDATE_USECOLS = ("patient_id", "eligible")

MODEL_KWARGS: Mapping[str, Any] = {
    "image_channels": 7,
    "base_channels": 16,
    "latent_dim": 192,
    "predictor_depth": 3,
    "predictor_heads": 4,
    "predictor_mlp_dim": 512,
    "dropout": 0.1,
}

LOGICAL_OBJECTIVE_CONTRACT: Mapping[str, str] = {
    "state_loss": "exact_patient_mean_over_32",
    "sigreg": "one_nonlinear_reduction_over_32_with_exact_surrogate_gradient",
    "ftv_loss": "exact_grounded_patient_mean_over_32",
    "optimizer_clip_ema": "once_per_32_patients",
}


@dataclass(frozen=True)
class ArmSpec:
    name: str
    input_kind: str
    upstream_model_name: str
    grounded: bool


ARM_SPECS: Mapping[str, ArmSpec] = {
    "L1": ArmSpec("L1", "legacy", "G1", False),
    "L3": ArmSpec("L3", "legacy", "G3", True),
    "N1": ArmSpec("N1", "c1b", "G1", False),
    "N3": ArmSpec("N3", "c1b", "G3", True),
}


def arm_spec(arm: str) -> ArmSpec:
    name = str(arm).upper()
    if name not in ARM_SPECS:
        raise ValueError(f"Stage B arm must be one of {ARMS}; got {arm!r}")
    return ARM_SPECS[name]


def validate_seed_fold(seed_base: int, fold: int) -> int:
    if isinstance(seed_base, bool) or int(seed_base) not in SEED_BASES:
        raise ValueError(f"seed_base must be one of {SEED_BASES}")
    if isinstance(fold, bool) or int(fold) not in FOLDS:
        raise ValueError("fold must be 0..4")
    return int(seed_base) + int(fold)


def validate_batch_contract(physical_batch_size: int, accumulation_steps: int) -> None:
    pair = (int(physical_batch_size), int(accumulation_steps))
    allowed = {
        (PREFERRED_PHYSICAL_BATCH_SIZE, PREFERRED_ACCUMULATION_STEPS),
        (FALLBACK_PHYSICAL_BATCH_SIZE, FALLBACK_ACCUMULATION_STEPS),
    }
    if pair not in allowed or pair[0] * pair[1] != EFFECTIVE_BATCH_SIZE:
        raise ValueError(
            "all four arms must use either physical=4/accum=8 or the global "
            "OOM restart contract physical=2/accum=16"
        )


def validate_lambda_ftv(value: float) -> float:
    """Reject any attempt to turn this preregistered run into a lambda sweep."""

    observed = float(value)
    if observed != 0.25:
        raise ValueError("grounded Stage B arms require the frozen lambda_FTV=0.25")
    return observed


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


def ordered_patient_sha256(patient_ids: Iterable[str]) -> str:
    return hashlib.sha256(
        "\n".join(str(value) for value in patient_ids).encode("utf-8")
    ).hexdigest()


def validate_no_extra_columns(actual: Iterable[str], allowed: Iterable[str], label: str) -> None:
    actual_set = set(str(value) for value in actual)
    allowed_set = set(str(value) for value in allowed)
    if actual_set != allowed_set:
        raise ValueError(
            f"{label} adapter columns drifted: actual={sorted(actual_set)}, "
            f"expected={sorted(allowed_set)}"
        )


def require_sha256(value: str, label: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest

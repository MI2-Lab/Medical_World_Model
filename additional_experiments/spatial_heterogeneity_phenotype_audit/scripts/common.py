"""Shared fail-closed utilities for the spatial phenotype audit."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
from typing import Any, Mapping

import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "audit.json"
AMENDMENT_PATH = EXPERIMENT_ROOT / "PREREGISTRATION_AMENDMENT.json"
IMPLEMENTATION_ERRATUM_PATH = (
    EXPERIMENT_ROOT / "PREREGISTRATION_IMPLEMENTATION_ERRATUM.json"
)
LOCK_PATH = EXPERIMENT_ROOT / "PREREGISTRATION_LOCK.json"
AMENDMENT_STATUS = "AMENDED_BEFORE_LABEL_DEPENDENT_ANALYSIS"
AMENDED_LOCK_STATUS = (
    "AMENDED_AND_REFROZEN_BEFORE_REEXTRACTION_OR_LABEL_DEPENDENT_PROBING"
)
IMPLEMENTATION_ERRATUM_STATUS = "IMPLEMENTATION_ERRATUM_BEFORE_LABEL_DEPENDENT_ANALYSIS"
IMPLEMENTATION_ERRATUM_LOCK_STATUS = (
    "IMPLEMENTATION_ERRATUM_FIXED_AND_REFROZEN_BEFORE_LABEL_DEPENDENT_PROBING"
)
IMPLEMENTATION_ERRATUM_SHA256 = (
    "ea49551e49a57bb7ddc52bc5ab841fab3cc9c3e1f1a9959d4f8a621d0dae9662"
)
PRIOR_AMENDED_PREREGISTRATION_COMMIT = "cdc7a57bf1ff373d97a97f51817ea83abe75d7e3"
PRIOR_AMENDED_PREREGISTRATION_LOCK_SHA256 = (
    "12e3c046f108d601c99fd354745fc5620e3ab234a72f307a2b1529063b7be0c4"
)
AMENDMENT_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "amendment_number",
        "original_preregistration_commit",
        "original_preregistration_lock_sha256",
        "reason_code",
        "geometry_qc",
        "pre_amendment_execution",
        "discarded_artifact_sha256",
        "contains_patient_identifiers",
    }
)
AMENDMENT_GEOMETRY_QC = {
    "source_authorized_by_visit": [808, 375, 375, 375],
    "post_local_core_valid_by_visit": [808, 375, 374, 374],
    "all_four_source_authorized_patient_count": 375,
    "all_four_upstream_core_parity_patient_count": 375,
    "all_four_post_local_core_valid_patient_count": 373,
    "post_local_empty_authorized_core_visit_count": 2,
    "affected_patient_count": 2,
    "representative_candidate_count_before": 375,
    "representative_candidate_count_after": 373,
    "amendment_scope": "deidentified_figure8_representative_selection_only",
    "probe_population_or_gate_contract_changed": False,
}
AMENDMENT_PRE_EXECUTION = {
    "cache_integrity_completed": True,
    "oracle_sidecar_completed": True,
    "completed_feature_cells": [
        "seed_2026/LOCAL0/fold_0",
        "seed_2026/LOCAL0/fold_1",
        "seed_2026/LOCAL0/fold_2",
    ],
    "interrupted_feature_cells": [
        "seed_2026/LOCAL0/fold_3",
        "seed_2026/LOCAL0/fold_4",
    ],
    "failed_feature_cell": "seed_2026/LOCAL3/fold_0",
    "representative_asset_created": False,
    "clinical_label_table_parsed": False,
    "stage_a_probe_fit": False,
    "stage_a_result_artifacts_created": False,
    "stage_b_started": False,
    "discard_before_refreeze_required": True,
    "reuse_forbidden": True,
}
AMENDMENT_DISCARDED_PATHS = frozenset(
    {
        "features/seed_2026/LOCAL0/fold_0/spatial_statistics.private.metadata.json",
        "features/seed_2026/LOCAL0/fold_0/spatial_statistics.private.npz",
        "features/seed_2026/LOCAL0/fold_1/spatial_statistics.private.metadata.json",
        "features/seed_2026/LOCAL0/fold_1/spatial_statistics.private.npz",
        "features/seed_2026/LOCAL0/fold_2/spatial_statistics.private.metadata.json",
        "features/seed_2026/LOCAL0/fold_2/spatial_statistics.private.npz",
        "logs/export_seed2026_LOCAL0_fold3.private.log",
        "logs/export_seed2026_LOCAL0_fold4.private.log",
        "logs/export_seed2026_LOCAL3_fold0.private.log",
        "manifests/cache_integrity.private.json",
        "manifests/oracle_regions.private.npz",
        "metrics/cache_integrity_contract.json",
        "metrics/oracle_region_contract.json",
    }
)
IMPLEMENTATION_ERRATUM_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "erratum_number",
        "prior_amended_preregistration_commit",
        "prior_amended_preregistration_lock_sha256",
        "reason_code",
        "pre_erratum_execution",
        "contract_scope",
        "discarded_artifact_sha256",
        "contains_patient_identifiers",
    }
)
IMPLEMENTATION_ERRATUM_PRE_EXECUTION = {
    "cache_integrity_completed": True,
    "oracle_sidecar_completed": True,
    "completed_feature_cell_count": 20,
    "independently_validated_feature_cell_count": 20,
    "maximum_p1_projection_parity_absolute_difference": 0.0,
    "representative_asset_created": True,
    "feature_matrix_final_validation_started": True,
    "feature_matrix_completion_marker_created": False,
    "failure_reason": (
        "audit.load_spatial_feature_asset_called_without_keyword_only_seed_arm_fold"
    ),
    "clinical_label_table_parsed": False,
    "stage_a_probe_fit": False,
    "stage_a_result_artifacts_created": False,
    "stage_b_started": False,
    "discarded_artifact_count": 65,
    "discarded_artifact_total_bytes": 307933315,
    "discard_before_refreeze_required": True,
    "reuse_forbidden": True,
}
IMPLEMENTATION_ERRATUM_CONTRACT_SCOPE = {
    "implementation_change": (
        "pass_seed_arm_fold_as_keyword_arguments_to_spatial_feature_loader"
    ),
    "affected_stage": "feature_matrix_completion_validation_only",
    "scientific_contract_changed": False,
    "representative_contract_changed": False,
    "causal_oracle_contract_changed": False,
}
IMPLEMENTATION_ERRATUM_DISCARDED_PATHS = frozenset(
    {
        "manifests/cache_integrity.private.json",
        "manifests/oracle_regions.private.npz",
        "metrics/cache_integrity_contract.json",
        "metrics/oracle_region_contract.json",
        "features/representative_activation.private.npz",
    }
    | {
        f"logs/export_seed{seed}_{arm}_fold{fold}.private.log"
        for seed in (2026, 3026)
        for arm in ("LOCAL0", "LOCAL3")
        for fold in range(5)
    }
    | {
        f"features/seed_{seed}/{arm}/fold_{fold}/spatial_statistics.private.{suffix}"
        for seed in (2026, 3026)
        for arm in ("LOCAL0", "LOCAL3")
        for fold in range(5)
        for suffix in ("metadata.json", "npz")
    }
)
if len(IMPLEMENTATION_ERRATUM_DISCARDED_PATHS) != 65:
    raise AssertionError(
        "implementation erratum discard inventory must contain 65 paths"
    )
PRIOR_LOCK_PRESERVED_FIELDS = frozenset(
    {
        "preregistration_revision",
        "branch",
        "formal_cell_count",
        "config_sha256",
        "privacy_policy_sha256",
        "amendment_sha256",
        "superseded_preregistration_commit",
        "superseded_preregistration_lock_sha256",
        "config_canonical_sha256",
        "selected_cells",
        "upstream_code_sha256",
        "runtime_environment",
        "analysis_outputs_present_before_freeze",
        "analysis_outputs_before_freeze",
    }
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


def runtime_environment() -> dict[str, str]:
    """Return exact analysis-runtime versions recorded in the formal lock."""

    packages = (
        "torch",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "Pillow",
    )
    versions = {name: importlib.metadata.version(name) for name in packages}
    import torch

    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "torch_cuda_build": str(torch.version.cuda or "NONE"),
        "torch_cudnn_version": str(torch.backends.cudnn.version() or "NONE"),
        **versions,
    }


def file_sha256(path: str | Path) -> str:
    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"not a regular file: {source}")
    before = source.stat()
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = source.stat()
    for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns"):
        if getattr(before, field) != getattr(after, field):
            raise RuntimeError(f"file changed while hashing: {source}")
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def ordered_sha256(values: Any) -> str:
    return hashlib.sha256(
        "\n".join(str(value) for value in values).encode("utf-8")
    ).hexdigest()


def load_preregistration_amendment(
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Load the exact public, patient-free revision-1 amendment record."""

    source = Path(AMENDMENT_PATH if path is None else path).resolve(strict=True)
    try:
        amendment = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("preregistration amendment is unreadable") from error
    if not isinstance(amendment, dict) or set(amendment) != set(AMENDMENT_KEYS):
        raise ValueError("preregistration amendment schema drifted")
    expected_scalars = {
        "schema_version": 1,
        "status": AMENDMENT_STATUS,
        "amendment_number": 1,
        "reason_code": "REPRESENTATIVE_POST_LOCAL_CORE_VALIDITY_COUNT_CORRECTION",
        "contains_patient_identifiers": False,
    }
    if any(amendment.get(name) != value for name, value in expected_scalars.items()):
        raise ValueError("preregistration amendment scalar contract drifted")
    if (
        type(amendment["schema_version"]) is not int
        or type(amendment["amendment_number"]) is not int
    ):
        raise ValueError("preregistration amendment revision fields must be integers")
    original_commit = amendment.get("original_preregistration_commit")
    original_lock = amendment.get("original_preregistration_lock_sha256")
    if (
        not isinstance(original_commit, str)
        or COMMIT_PATTERN.fullmatch(original_commit) is None
    ):
        raise ValueError("amendment original commit must be a full lowercase SHA")
    if (
        not isinstance(original_lock, str)
        or SHA256_PATTERN.fullmatch(original_lock) is None
    ):
        raise ValueError("amendment original lock must be a lowercase SHA-256")
    if amendment.get("geometry_qc") != AMENDMENT_GEOMETRY_QC:
        raise ValueError("preregistration amendment geometry QC drifted")
    if amendment.get("pre_amendment_execution") != AMENDMENT_PRE_EXECUTION:
        raise ValueError("preregistration amendment execution ledger drifted")
    discarded = amendment.get("discarded_artifact_sha256")
    if not isinstance(discarded, dict) or set(discarded) != set(
        AMENDMENT_DISCARDED_PATHS
    ):
        raise ValueError("preregistration amendment discard ledger drifted")
    for relative, digest in discarded.items():
        relative_path = Path(str(relative))
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or not isinstance(digest, str)
            or SHA256_PATTERN.fullmatch(digest) is None
        ):
            raise ValueError("preregistration amendment discard record is invalid")
    return amendment


def load_preregistration_implementation_erratum(
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Load the exact public, patient-free implementation-only erratum."""

    source = Path(IMPLEMENTATION_ERRATUM_PATH if path is None else path).resolve(
        strict=True
    )
    try:
        erratum = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "preregistration implementation erratum is unreadable"
        ) from error
    if not isinstance(erratum, dict) or set(erratum) != set(
        IMPLEMENTATION_ERRATUM_KEYS
    ):
        raise ValueError("preregistration implementation erratum schema drifted")
    expected_scalars = {
        "schema_version": 1,
        "status": IMPLEMENTATION_ERRATUM_STATUS,
        "erratum_number": 1,
        "prior_amended_preregistration_commit": (PRIOR_AMENDED_PREREGISTRATION_COMMIT),
        "prior_amended_preregistration_lock_sha256": (
            PRIOR_AMENDED_PREREGISTRATION_LOCK_SHA256
        ),
        "reason_code": "FEATURE_MATRIX_VALIDATOR_KEYWORD_ONLY_CALL_CORRECTION",
        "contains_patient_identifiers": False,
    }
    if any(erratum.get(name) != value for name, value in expected_scalars.items()):
        raise ValueError("preregistration implementation erratum scalar drifted")
    if (
        type(erratum["schema_version"]) is not int
        or type(erratum["erratum_number"]) is not int
    ):
        raise ValueError("implementation erratum revision fields must be integers")
    execution = erratum.get("pre_erratum_execution")
    if execution != IMPLEMENTATION_ERRATUM_PRE_EXECUTION:
        raise ValueError("implementation erratum execution ledger drifted")
    integer_execution_fields = (
        "completed_feature_cell_count",
        "independently_validated_feature_cell_count",
        "discarded_artifact_count",
        "discarded_artifact_total_bytes",
    )
    boolean_execution_fields = (
        "cache_integrity_completed",
        "oracle_sidecar_completed",
        "representative_asset_created",
        "feature_matrix_final_validation_started",
        "feature_matrix_completion_marker_created",
        "clinical_label_table_parsed",
        "stage_a_probe_fit",
        "stage_a_result_artifacts_created",
        "stage_b_started",
        "discard_before_refreeze_required",
        "reuse_forbidden",
    )
    if any(
        type(execution[name]) is not int for name in integer_execution_fields
    ) or any(type(execution[name]) is not bool for name in boolean_execution_fields):
        raise ValueError("implementation erratum execution ledger types drifted")
    if type(execution["maximum_p1_projection_parity_absolute_difference"]) is not float:
        raise ValueError("implementation erratum parity value must be a float")
    contract_scope = erratum.get("contract_scope")
    if contract_scope != IMPLEMENTATION_ERRATUM_CONTRACT_SCOPE:
        raise ValueError("implementation erratum contract scope drifted")
    if any(
        type(contract_scope[name]) is not bool
        for name in (
            "scientific_contract_changed",
            "representative_contract_changed",
            "causal_oracle_contract_changed",
        )
    ):
        raise ValueError("implementation erratum contract-scope types drifted")
    discarded = erratum.get("discarded_artifact_sha256")
    if not isinstance(discarded, dict) or set(discarded) != set(
        IMPLEMENTATION_ERRATUM_DISCARDED_PATHS
    ):
        raise ValueError("implementation erratum discard ledger drifted")
    for relative, digest in discarded.items():
        relative_path = Path(str(relative))
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or not isinstance(digest, str)
            or SHA256_PATTERN.fullmatch(digest) is None
        ):
            raise ValueError("implementation erratum discard record is invalid")
    if file_sha256(source) != IMPLEMENTATION_ERRATUM_SHA256:
        raise ValueError("preregistration implementation erratum bytes drifted")
    return erratum


def _experiment_relative_path(filename: str) -> str:
    return (EXPERIMENT_ROOT.relative_to(REPO_ROOT) / filename).as_posix()


def historical_file_bytes(commit: str, filename: str) -> bytes:
    """Read an experiment file exactly as committed at a historical anchor."""

    if COMMIT_PATTERN.fullmatch(str(commit)) is None:
        raise ValueError("historical Git anchor must be a full lowercase SHA")
    relative_name = Path(filename)
    if relative_name.is_absolute() or ".." in relative_name.parts:
        raise ValueError("historical experiment path is unsafe")
    try:
        return subprocess.check_output(
            [
                "git",
                "show",
                f"{commit}:{_experiment_relative_path(relative_name.as_posix())}",
            ],
            cwd=REPO_ROOT,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as error:
        raise ValueError(
            f"historical Git artifact is unavailable: {filename}"
        ) from error


def historical_file_sha256(commit: str, filename: str) -> str:
    """Hash an experiment file exactly as committed at a historical anchor."""

    return hashlib.sha256(historical_file_bytes(commit, filename)).hexdigest()


def historical_json(commit: str, filename: str) -> dict[str, Any]:
    """Load an exact JSON object from a committed experiment revision."""

    try:
        value = json.loads(historical_file_bytes(commit, filename).decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"historical Git JSON is unreadable: {filename}") from error
    if not isinstance(value, dict):
        raise ValueError(f"historical Git JSON is not an object: {filename}")
    return value


def authenticate_original_preregistration(
    amendment: Mapping[str, Any],
) -> tuple[str, str]:
    """Authenticate the immutable original lock named by the amendment."""

    original_commit = str(amendment["original_preregistration_commit"])
    original_lock = str(amendment["original_preregistration_lock_sha256"])
    if (
        historical_file_sha256(original_commit, "PREREGISTRATION_LOCK.json")
        != original_lock
    ):
        raise ValueError("amendment original lock differs from historical Git bytes")
    return original_commit, original_lock


def authenticate_prior_amended_preregistration(
    erratum: Mapping[str, Any],
    amendment: Mapping[str, Any] | None = None,
) -> tuple[str, str, dict[str, Any]]:
    """Authenticate the superseded amended lock bound by the erratum."""

    value = load_preregistration_amendment() if amendment is None else dict(amendment)
    prior_commit = str(erratum["prior_amended_preregistration_commit"])
    prior_lock_sha256 = str(erratum["prior_amended_preregistration_lock_sha256"])
    if (
        prior_commit != PRIOR_AMENDED_PREREGISTRATION_COMMIT
        or prior_lock_sha256 != PRIOR_AMENDED_PREREGISTRATION_LOCK_SHA256
        or historical_file_sha256(prior_commit, "PREREGISTRATION_LOCK.json")
        != prior_lock_sha256
    ):
        raise ValueError("implementation erratum prior amended lock is unauthenticated")
    prior_lock = historical_json(prior_commit, "PREREGISTRATION_LOCK.json")
    original_commit, original_lock = authenticate_original_preregistration(value)
    if prior_commit == original_commit or (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", original_commit, prior_commit],
            cwd=REPO_ROOT,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        != 0
    ):
        raise ValueError("original preregistration is not before the amended anchor")
    if (
        prior_lock.get("schema_version") != 3
        or prior_lock.get("preregistration_revision") != 2
        or prior_lock.get("status") != AMENDED_LOCK_STATUS
        or prior_lock.get("amendment_sha256") != file_sha256(AMENDMENT_PATH)
        or prior_lock.get("superseded_preregistration_commit") != original_commit
        or prior_lock.get("superseded_preregistration_lock_sha256") != original_lock
        or historical_file_sha256(prior_commit, "PREREGISTRATION_AMENDMENT.json")
        != file_sha256(AMENDMENT_PATH)
    ):
        raise ValueError("historical amended preregistration contract drifted")
    return prior_commit, prior_lock_sha256, prior_lock


def require_preserved_prior_lock_contract(
    current: Mapping[str, Any], prior: Mapping[str, Any]
) -> None:
    """Require the implementation refreeze to preserve every scientific input."""

    for field in PRIOR_LOCK_PRESERVED_FIELDS:
        if current.get(field) != prior.get(field):
            raise ValueError(
                f"implementation erratum changed preserved lock field: {field}"
            )
    current_implementations = current.get("implementation_sha256")
    prior_implementations = prior.get("implementation_sha256")
    if (
        not isinstance(current_implementations, Mapping)
        or not isinstance(prior_implementations, Mapping)
        or set(current_implementations) != set(prior_implementations)
    ):
        raise ValueError(
            "implementation erratum changed implementation inventory paths"
        )


def require_implementation_erratum_plan_disclosure(
    prior_lock: Mapping[str, Any],
) -> None:
    """Prove the plan changed only by insertion of the public erratum disclosure."""

    prior_payload = historical_file_bytes(
        PRIOR_AMENDED_PREREGISTRATION_COMMIT, "EXPERIMENT_PLAN.md"
    )
    if hashlib.sha256(prior_payload).hexdigest() != prior_lock.get("plan_sha256"):
        raise ValueError("historical amended lock does not authenticate its plan")
    try:
        prior_text = prior_payload.decode("utf-8")
        current_text = (EXPERIMENT_ROOT / "EXPERIMENT_PLAN.md").read_text(
            encoding="utf-8"
        )
    except UnicodeError as error:
        raise ValueError("preregistration plan is not valid UTF-8") from error
    start_marker = "**Pre-probe implementation erratum 1 (no scientific change).**"
    end_marker = "## 7. Longitudinal heterogeneity"
    if current_text.count(start_marker) != 1 or current_text.count(end_marker) != 1:
        raise ValueError(
            "implementation erratum plan disclosure is missing or repeated"
        )
    start = current_text.index(start_marker)
    end = current_text.index(end_marker, start)
    disclosure = current_text[start:end]
    normalized_disclosure = " ".join(disclosure.split())
    required_fragments = (
        PRIOR_AMENDED_PREREGISTRATION_COMMIT,
        PRIOR_AMENDED_PREREGISTRATION_LOCK_SHA256,
        "maximum P1 parity absolute difference of `0.0`",
        "No clinical-label table was parsed",
        "no Stage-A probe was fit",
        "Stage B did not start",
        "all 65 experiment outputs present at failure (307,933,315 bytes)",
        "none may be reused",
    )
    if any(fragment not in normalized_disclosure for fragment in required_fragments):
        raise ValueError("implementation erratum plan disclosure facts drifted")
    if current_text[:start] + current_text[end:] != prior_text:
        raise ValueError("plan changed outside the implementation erratum disclosure")


def preregistration_provenance_anchors(
    amendment: Mapping[str, Any] | None = None,
    erratum: Mapping[str, Any] | None = None,
) -> tuple[str, str, str]:
    """Return authenticated ``(original, prior-amended, active)`` anchors."""

    amendment_value = (
        load_preregistration_amendment() if amendment is None else dict(amendment)
    )
    erratum_value = (
        load_preregistration_implementation_erratum()
        if erratum is None
        else dict(erratum)
    )
    original_commit, _original_lock = authenticate_original_preregistration(
        amendment_value
    )
    prior_commit, _prior_lock, _prior_payload = (
        authenticate_prior_amended_preregistration(erratum_value, amendment_value)
    )
    lock_relative = _experiment_relative_path("PREREGISTRATION_LOCK.json")
    active_commit = subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", lock_relative],
        cwd=REPO_ROOT,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()
    if COMMIT_PATTERN.fullmatch(active_commit) is None:
        raise ValueError("active preregistration lock has no committed Git anchor")
    if active_commit in {original_commit, prior_commit}:
        raise ValueError("active preregistration anchor is still superseded")
    for ancestor, descendant, message in (
        (
            original_commit,
            prior_commit,
            "original preregistration is not an ancestor of the amended anchor",
        ),
        (
            prior_commit,
            active_commit,
            "amended preregistration is not an ancestor of the erratum refreeze",
        ),
        (
            active_commit,
            "HEAD",
            "active erratum refreeze is not an ancestor of HEAD",
        ),
    ):
        if (
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", ancestor, descendant],
                cwd=REPO_ROOT,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            != 0
        ):
            raise ValueError(message)
    if historical_file_sha256(
        active_commit, "PREREGISTRATION_LOCK.json"
    ) != file_sha256(LOCK_PATH):
        raise ValueError("committed active lock differs from the current lock")
    if historical_file_sha256(
        active_commit, "PREREGISTRATION_AMENDMENT.json"
    ) != file_sha256(AMENDMENT_PATH):
        raise ValueError("committed amendment differs from the current amendment")
    if historical_file_sha256(
        active_commit, "PREREGISTRATION_IMPLEMENTATION_ERRATUM.json"
    ) != file_sha256(IMPLEMENTATION_ERRATUM_PATH):
        raise ValueError(
            "committed implementation erratum differs from the current erratum"
        )
    return original_commit, prior_commit, active_commit


def preregistration_anchor_commits(
    amendment: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    """Return ``(original, active)`` while preserving the established API."""

    original, _prior, active = preregistration_provenance_anchors(amendment)
    return original, active


def preregistration_chain(
    lock: Mapping[str, Any],
    amendment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the stable six-key scientific chain, transitively binding the erratum."""

    value = load_preregistration_amendment() if amendment is None else dict(amendment)
    erratum = load_preregistration_implementation_erratum()
    original_commit, prior_commit, active_commit = preregistration_provenance_anchors(
        value, erratum
    )
    original_lock = str(value["original_preregistration_lock_sha256"])
    amendment_sha256 = file_sha256(AMENDMENT_PATH)
    erratum_sha256 = file_sha256(IMPLEMENTATION_ERRATUM_PATH)
    if (
        lock.get("schema_version") != 4
        or lock.get("preregistration_revision") != 2
        or lock.get("status") != IMPLEMENTATION_ERRATUM_LOCK_STATUS
        or lock.get("amendment_sha256") != amendment_sha256
        or lock.get("superseded_preregistration_commit") != original_commit
        or lock.get("superseded_preregistration_lock_sha256") != original_lock
        or lock.get("implementation_erratum_sha256") != erratum_sha256
        or lock.get("superseded_amended_preregistration_commit") != prior_commit
        or lock.get("superseded_amended_preregistration_lock_sha256")
        != PRIOR_AMENDED_PREREGISTRATION_LOCK_SHA256
    ):
        raise ValueError("runtime preregistration/implementation-erratum chain drifted")
    return {
        "preregistration_revision": 2,
        "active_preregistration_lock_sha256": file_sha256(LOCK_PATH),
        "preregistration_amendment_sha256": amendment_sha256,
        "original_preregistration_lock_sha256": original_lock,
        "original_preregistration_commit": original_commit,
        "active_preregistration_commit": active_commit,
    }


def load_config(
    path: str | Path = DEFAULT_CONFIG, *, verify_inputs: bool = True
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("audit config must be a JSON object")
    if payload.get("schema_version") != 1:
        raise ValueError("audit config schema_version must be 1")
    if payload.get("experiment") != "spatial_heterogeneity_phenotype_audit":
        raise ValueError("audit config names a different experiment")
    paths = payload.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("audit config paths must be an object")
    hash_pairs = (
        ("local_preregistration_lock", "local_preregistration_lock_sha256"),
        ("spatial_sidecar", "spatial_sidecar_sha256"),
        ("stage_b_data_contract", "stage_b_data_contract_sha256"),
        (
            "stage_b_upstream_authorization",
            "stage_b_upstream_authorization_sha256",
        ),
        ("c1b_cache_manifest", "c1b_cache_manifest_sha256"),
        ("support_inventory", "support_inventory_sha256"),
        ("clinical_labels", "clinical_labels_sha256"),
        ("fold_manifest", "fold_manifest_sha256"),
        ("ftv_table", "ftv_table_sha256"),
    )
    resolved = dict(paths)
    for path_key, hash_key in hash_pairs:
        source_path = Path(str(paths[path_key])).expanduser().resolve()
        resolved[path_key] = source_path
        expected = str(paths[hash_key])
        if verify_inputs and file_sha256(source_path) != expected:
            raise ValueError(f"hash mismatch for {path_key}: {source_path}")
    for directory_key in ("source_repo", "checkpoint_root", "local_feature_root"):
        directory = Path(str(paths[directory_key])).expanduser().resolve()
        if verify_inputs and not directory.is_dir():
            raise FileNotFoundError(directory)
        resolved[directory_key] = directory
    output = dict(payload)
    output["paths"] = resolved
    return output


def require_preregistration_lock(config: Mapping[str, Any]) -> dict[str, Any]:
    lock_path = LOCK_PATH
    if not lock_path.is_file():
        raise FileNotFoundError("formal execution requires PREREGISTRATION_LOCK.json")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != 4:
        raise ValueError("preregistration lock schema version is invalid")
    if lock.get("preregistration_revision") != 2:
        raise ValueError("preregistration lock revision is invalid")
    if lock.get("status") != IMPLEMENTATION_ERRATUM_LOCK_STATUS:
        raise ValueError("preregistration lock status is invalid")
    if lock.get("branch") != config.get("branch"):
        raise ValueError("preregistration lock names another branch")
    if lock.get("analysis_outputs_present_before_freeze") is not False:
        raise ValueError("preregistration was not frozen before analysis outputs")
    checks = {
        "plan_sha256": EXPERIMENT_ROOT / "EXPERIMENT_PLAN.md",
        "config_sha256": EXPERIMENT_ROOT / "configs" / "audit.json",
        "privacy_policy_sha256": EXPERIMENT_ROOT / ".gitignore",
        "amendment_sha256": AMENDMENT_PATH,
        "implementation_erratum_sha256": IMPLEMENTATION_ERRATUM_PATH,
    }
    for field, path in checks.items():
        if lock.get(field) != file_sha256(path):
            raise ValueError(f"preregistered {field} drifted")
    if lock.get("config_canonical_sha256") != canonical_sha256(config):
        raise ValueError("loaded config differs from the frozen canonical config")
    amendment = load_preregistration_amendment()
    erratum = load_preregistration_implementation_erratum()
    original_commit, original_lock = authenticate_original_preregistration(amendment)
    prior_commit, prior_lock_sha256, prior_lock = (
        authenticate_prior_amended_preregistration(erratum, amendment)
    )
    if (
        lock.get("superseded_preregistration_commit") != original_commit
        or lock.get("superseded_preregistration_lock_sha256") != original_lock
        or lock.get("superseded_amended_preregistration_commit") != prior_commit
        or lock.get("superseded_amended_preregistration_lock_sha256")
        != prior_lock_sha256
    ):
        raise ValueError("active lock does not mirror its superseded Git anchors")
    require_implementation_erratum_plan_disclosure(prior_lock)
    require_preserved_prior_lock_contract(lock, prior_lock)
    provenance = lock.get("git_provenance_before_freeze")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("base_head") != prior_commit
        or provenance.get("branch") != config.get("branch")
        or provenance.get("all_dirty_paths_confined_to_new_experiment") is not True
    ):
        raise ValueError("implementation refreeze Git provenance drifted")
    implementations = lock.get("implementation_sha256")
    if not isinstance(implementations, dict) or not implementations:
        raise ValueError("preregistration lock has no implementation inventory")
    for relative, expected in implementations.items():
        relative_path = Path(str(relative))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"invalid preregistered implementation path: {relative}")
        path = EXPERIMENT_ROOT / relative_path
        if file_sha256(path) != expected:
            raise ValueError(f"preregistered implementation drifted: {relative}")
    upstream = lock.get("upstream_code_sha256")
    if not isinstance(upstream, dict) or not upstream:
        raise ValueError("preregistration lock has no upstream-code inventory")
    for source, expected in upstream.items():
        if file_sha256(Path(source)) != expected:
            raise ValueError(f"preregistered upstream code drifted: {source}")
    if lock.get("runtime_environment") != runtime_environment():
        raise ValueError("formal runtime environment differs from preregistration")
    chain = preregistration_chain(lock, amendment)
    if chain["original_preregistration_commit"] != original_commit:
        raise ValueError("amended preregistration anchor chain drifted")
    return lock


def private_directory(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    output.chmod(0o700)
    return output


def atomic_json(
    payload: Mapping[str, Any], path: str | Path, *, private: bool = False
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.parent.chmod(0o700 if private else 0o755)
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
        temporary.chmod(0o600 if private else 0o644)
        temporary.replace(destination)
        destination.chmod(0o600 if private else 0o644)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def atomic_csv(frame: pd.DataFrame, path: str | Path, *, private: bool = False) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.parent.chmod(0o700 if private else 0o755)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            frame.to_csv(stream, index=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o600 if private else 0o644)
        temporary.replace(destination)
        destination.chmod(0o600 if private else 0o644)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


__all__ = [
    "AMENDED_LOCK_STATUS",
    "AMENDMENT_PATH",
    "AMENDMENT_STATUS",
    "DEFAULT_CONFIG",
    "EXPERIMENT_ROOT",
    "IMPLEMENTATION_ERRATUM_LOCK_STATUS",
    "IMPLEMENTATION_ERRATUM_PATH",
    "IMPLEMENTATION_ERRATUM_SHA256",
    "IMPLEMENTATION_ERRATUM_STATUS",
    "LOCK_PATH",
    "PRIOR_AMENDED_PREREGISTRATION_COMMIT",
    "PRIOR_AMENDED_PREREGISTRATION_LOCK_SHA256",
    "REPO_ROOT",
    "atomic_csv",
    "atomic_json",
    "authenticate_original_preregistration",
    "authenticate_prior_amended_preregistration",
    "canonical_sha256",
    "file_sha256",
    "historical_file_bytes",
    "historical_file_sha256",
    "historical_json",
    "load_config",
    "load_preregistration_amendment",
    "load_preregistration_implementation_erratum",
    "ordered_sha256",
    "preregistration_anchor_commits",
    "preregistration_chain",
    "preregistration_provenance_anchors",
    "private_directory",
    "require_implementation_erratum_plan_disclosure",
    "require_preserved_prior_lock_contract",
    "require_preregistration_lock",
    "runtime_environment",
]

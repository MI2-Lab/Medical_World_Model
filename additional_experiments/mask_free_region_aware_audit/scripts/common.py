"""Fail-closed contracts shared by the mask-free regional audit."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
CONFIG_PATH = EXPERIMENT_ROOT / "configs" / "audit.json"
PLAN_PATH = EXPERIMENT_ROOT / "EXPERIMENT_PLAN.md"
GITIGNORE_PATH = EXPERIMENT_ROOT / ".gitignore"
LOCK_PATH = EXPERIMENT_ROOT / "PREREGISTRATION_LOCK.json"
IMPLEMENTATION_ERRATUM_PATH = (
    EXPERIMENT_ROOT / "PREREGISTRATION_IMPLEMENTATION_ERRATUM.json"
)
IMPLEMENTATION_ERRATUM_2_PATH = (
    EXPERIMENT_ROOT / "PREREGISTRATION_IMPLEMENTATION_ERRATUM_2.json"
)
FEATURE_FILENAME = "regional_features.private.npz"
METADATA_FILENAME = "regional_features.private.metadata.json"
COMPLETION_PATH = EXPERIMENT_ROOT / "features" / "feature_matrix_complete.private.json"
GEOMETRY_CONTRACT_PATH = EXPERIMENT_ROOT / "metrics" / "region_occupancy_contract.json"
ARMS = ("LOCAL0", "LOCAL3")
SEEDS = (2026, 3026)
FOLDS = (0, 1, 2, 3, 4)
VISITS = ("T0", "T1", "T2", "T3")
PATIENT_COUNT = 808
VARIANT_KEYS = (
    "R0",
    "R1",
    "R2",
    "R3",
    "R4",
    "R5",
    "R5_RP192",
    "S1",
    "S2",
    "S3",
    "S4",
    "S5",
)
VARIANT_DIMENSIONS = {
    "R0": 128,
    "R1": 128,
    "R2": 128,
    "R3": 128,
    "R4": 256,
    "R5": 384,
    "R5_RP192": 192,
    "S1": 128,
    "S2": 128,
    "S3": 128,
    "S4": 256,
    "S5": 384,
}
FEATURE_KEYS = (
    "patient_id",
    "split",
    *VARIANT_KEYS,
    "arm",
    "seed_base",
    "fold",
)
METADATA_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "experiment",
        "cell",
        "arm",
        "seed_base",
        "fold",
        "checkpoint_sha256",
        "selection_sha256",
        "reference_feature_sha256",
        "goal5_feature_sha256",
        "goal5_feature_metadata_sha256",
        "feature_path",
        "feature_sha256",
        "patient_count",
        "visit_count",
        "patient_order_sha256",
        "split_order_sha256",
        "actual_encoder_shape",
        "actual_encoder_dtype",
        "variant_shapes",
        "variant_dtypes",
        "region_weight_sha256",
        "geometry_contract_path",
        "geometry_contract_sha256",
        "projection_matrix_float32_sha256",
        "checkpoint_c1b_local_weight_bitwise_equal",
        "r0_goal5_mean_parity",
        "projected_r0_local_state_parity",
        "goal5_preregistration_lock_sha256",
        "preregistration_lock_sha256",
        "config_sha256",
        "stage_a_sentinel_sha256",
        "checkpoint_data_provenance_sha256",
        "implementation_sha256",
        "encoder_frozen",
        "training_performed",
        "streamed_raw_spatial_map_not_persisted",
        "response_projection_used_only_for_parity",
        "projector_called",
        "transition_called",
        "target_encoder_called",
        "ftv_head_called",
        "lesion_mask_read",
        "tumor_bbox_read",
        "clinical_label_table_read",
        "ftv_value_table_read",
        "phenotype_or_pcr_labels_read",
        "future_visit_used_to_define_region",
    }
)
COMPLETION_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "experiment",
        "cell_count",
        "config_sha256",
        "preregistration_lock_sha256",
        "geometry_contract_sha256",
        "cells",
    }
)
COMPLETION_CELL_KEYS = frozenset(
    {
        "cell",
        "seed_base",
        "arm",
        "fold",
        "feature_path",
        "feature_sha256",
        "metadata_path",
        "metadata_sha256",
        "patient_order_sha256",
    }
)
LOCK_STATUS = "FROZEN_BEFORE_FEATURE_EXTRACTION_OR_LABEL_DEPENDENT_ANALYSIS"
REFREEZE_LOCK_STATUS = (
    "IMPLEMENTATION_COMPATIBILITY_ERRATUM_FIXED_AND_REFROZEN_BEFORE_"
    "FORMAL_ANALYSIS_COMPLETION"
)
REFREEZE_2_LOCK_STATUS = (
    "IMPLEMENTATION_OUTPUT_AND_REPORTING_ERRATUM_FIXED_AND_REFROZEN_BEFORE_"
    "CORRECTED_FORMAL_RERUN"
)
PRIOR_PREREGISTRATION_COMMIT = "673aab146936d3890e79a9df8e8bbad8f9dec81c"
PRIOR_PREREGISTRATION_LOCK_SHA256 = (
    "d53f8d23a552edce15283c13b433830da7378413ec1e3eb52783cad3087c5d90"
)
PRIOR_COMPATIBILITY_REFREEZE_COMMIT = (
    "c781b8f4c8ff14e0439e15447af9ca21bcd88be1"
)
PRIOR_COMPATIBILITY_REFREEZE_LOCK_SHA256 = (
    "a574c905bd9780ac4260ca5e63a9a89e4db15a3217932409cda0fb00aaadd6ad"
)
IMPLEMENTATION_ERRATUM_STATUS = (
    "IMPLEMENTATION_COMPATIBILITY_ERRATUM_BEFORE_FORMAL_OUTPUT_PUBLICATION"
)
IMPLEMENTATION_ERRATUM_REASON = (
    "PANDAS_DEFAULT_FLOAT_PARSER_BREAKS_EXACT_GOAL5_P1_PARITY"
)
ERRATUM_PLAN_APPENDIX_SHA256 = (
    "2b550ac5bab32db45d5af807ca6b7316315a73670a506715d3e07689f3670ae9"
)
ERRATUM_DISCARDED_MAP_CANONICAL_SHA256 = (
    "247bcfabda1c688c9855995a8523b368a59f27faf0d0ae6cb30356ffc6e9ab76"
)
IMPLEMENTATION_ERRATUM_2_STATUS = (
    "IMPLEMENTATION_OUTPUT_AND_REPORTING_ERRATUM_AFTER_FORMAL_OUTPUT_PUBLICATION"
)
IMPLEMENTATION_ERRATUM_2_REASON = (
    "BOOTSTRAP_SUMMARY_MODEL_SEED_OVERWRITTEN_BY_RNG_SEED"
)
IMPLEMENTATION_ERRATUM_2_SHA256 = (
    "91ca3417912ad757f34ea847effd38b00464d873b8ac6936a4ce3e09bd6bf670"
)
ERRATUM_2_PLAN_APPENDIX_SHA256 = (
    "ae8d9e800932e05b30ceedd168eff87fe190c29c5473d72817ff29cc43f59f47"
)
ERRATUM_2_PRE_EXECUTION_CANONICAL_SHA256 = (
    "058960e32f1e10ed54ad5a255ef2cd4c86955d1fa81e7738923087876b9a5b66"
)
ERRATUM_2_CONTRACT_SCOPE_CANONICAL_SHA256 = (
    "d9ef0bf47a72b4549862c5a05f8c7153f6fb1112e59ff01c4e4ecc8ae6d13b6a"
)
ERRATUM_2_DISCARDED_MAP_CANONICAL_SHA256 = (
    "904d9625862b29bb0662b254bbad3b9b05c505a237b4050f5fd6b8f3225f52bb"
)
ERRATUM_2_DISCARDED_RECORD_SET_SHA256 = (
    "fc5a93acb2a0beca572c495fe6a60fecac5bce254dd74b79132911f326778853"
)
ZERO_RESULT_INVENTORY = {
    "feature_files": 0,
    "prediction_files": 0,
    "metric_files": 0,
    "figure_files": 0,
    "report_files": 0,
    "log_files": 0,
    "manifest_files": 0,
}
PRIVACY_CONTRACT = {
    "private_patient_artifacts_owner_only": True,
    "raw_spatial_maps_persisted": False,
    "region_definition_reads_masks_labels_ftv_or_clinical": False,
}
INITIAL_LOCK_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "branch",
        "parent_commit_sha",
        "created_utc",
        "config_sha256",
        "config_canonical_sha256",
        "experiment_plan_sha256",
        "gitignore_sha256",
        "implementation_sha256",
        "goal5_preregistration_lock_sha256",
        "goal5_feature_completion_sha256",
        "formal_cell_count",
        "selected_cells",
        "pre_freeze_result_inventory",
        "privacy_contract",
    }
)
REFREEZE_LOCK_KEYS = frozenset(
    {
        *INITIAL_LOCK_KEYS,
        "preregistration_revision",
        "prior_preregistration_commit",
        "prior_preregistration_lock_sha256",
        "implementation_erratum_sha256",
        "superseded_artifacts_reused",
        "scientific_contract_unchanged",
        "pre_refreeze_result_inventory",
        "git_provenance_before_refreeze",
    }
)
REFREEZE_2_LOCK_KEYS = frozenset(
    {
        *REFREEZE_LOCK_KEYS,
        "prior_compatibility_refreeze_commit",
        "prior_compatibility_refreeze_lock_sha256",
        "implementation_erratum_2_sha256",
        "superseded_formal_run_artifact_count",
        "superseded_formal_run_artifact_record_set_sha256",
        "all_twenty_feature_cells_rebuild_required",
        "pre_refreeze_2_result_inventory",
        "git_provenance_before_refreeze_2",
    }
)
IMPLEMENTATION_ERRATUM_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "erratum_number",
        "prior_preregistration_commit",
        "prior_preregistration_lock_sha256",
        "reason_code",
        "pre_erratum_execution",
        "contract_scope",
        "discarded_artifact_sha256",
        "contains_patient_identifiers",
    }
)
IMPLEMENTATION_ERRATUM_2_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "erratum_number",
        "prior_compatibility_refreeze_commit",
        "prior_compatibility_refreeze_lock_sha256",
        "prior_implementation_erratum_sha256",
        "reason_code",
        "pre_erratum_execution",
        "contract_scope",
        "discarded_artifact_sha256",
        "contains_patient_identifiers",
    }
)
IMPLEMENTATION_ERRATUM_PRE_EXECUTION = {
    "formal_run_started_utc": "2026-08-12T07:13:20Z",
    "formal_run_failed_utc": "2026-08-12T07:35:19Z",
    "failure_wall_time_seconds_approximate": 1320,
    "feature_matrix_completed": True,
    "completed_feature_cell_count": 20,
    "independently_validated_feature_cell_count": 20,
    "feature_matrix_validated_before_labels": True,
    "fold_manifest_with_pcr_labels_parsed": True,
    "clinical_phenotype_label_table_parsed": True,
    "ftv_table_parsed": True,
    "registered_probe_families_completed_in_memory": [
        "mri_only_pcr",
        "clinical_pcr",
        "clinical_ftv_pcr",
        "phenotype",
        "ftv_and_delta_ftv",
    ],
    "goal5_prediction_tables_loaded": True,
    "oracle_recovery_started": True,
    "bootstrap_started": False,
    "gates_evaluated": False,
    "failure_exception_type": "ValueError",
    "failure_function": "verify_r0_p1_parity",
    "failure_identity_without_patient_identifier": (
        "seed_2026/LOCAL0/T0/pCR/ftv_complete_375"
    ),
    "failure_reason": (
        "default_pandas_csv_float_parser_changed_goal5_p1_probability_last_bits"
    ),
    "diagnostic_fold_test_row_count": 69,
    "default_parser_probability_difference_count": 20,
    "default_parser_maximum_absolute_probability_difference": (
        5.551115123125783e-17
    ),
    "round_trip_parser_probability_difference_count": 0,
    "round_trip_parser_maximum_absolute_probability_difference": 0,
    "round_trip_patient_fold_label_equal": True,
    "round_trip_probability_label_threshold_bitwise_equal": True,
    "performance_metric_or_gate_value_inspected_or_printed": False,
    "patient_level_prediction_artifact_created": False,
    "private_analysis_artifact_count": 0,
    "public_label_derived_artifact_count": 0,
    "public_metric_csv_count": 0,
    "gate_artifact_created": False,
    "run_summary_created": False,
    "figure_artifact_count": 0,
    "report_artifact_count": 0,
    "new_encoder_or_jepa_training_performed": False,
    "in_memory_label_dependent_objects_discarded_at_process_exit": True,
    "superseded_feature_completion_sha256": (
        "70ae60c6913f92ae9b9c7bd62482ba496e3ad0850e255645ba721ee0180d2d8f"
    ),
    "superseded_geometry_contract_sha256": (
        "de1070317278599e832cd276ab808671d9f7267b051193df5af8ef8fc13ea700"
    ),
    "discarded_artifact_count": 62,
    "discarded_artifact_total_bytes": 571494370,
    "discarded_artifact_count_by_root": {
        "features": 41,
        "logs": 20,
        "metrics": 1,
        "predictions": 0,
        "figures": 0,
        "reports": 0,
    },
    "discarded_artifact_bytes_by_root": {
        "features": 571372121,
        "logs": 120060,
        "metrics": 2189,
        "predictions": 0,
        "figures": 0,
        "reports": 0,
    },
    "discarded_artifact_record_format": (
        "utf8_lines_sorted_by_audit_relative_path_as_"
        "path_tab_size_bytes_tab_sha256_newline"
    ),
    "discarded_artifact_record_set_sha256": (
        "6779442afc5d7375a141eea6c07d6740fae80a838bd99d22ec929cb33cc07244"
    ),
    "discard_before_refreeze_required": True,
    "formal_reuse_forbidden": True,
}
IMPLEMENTATION_ERRATUM_CONTRACT_SCOPE = {
    "primary_implementation_change": (
        "read_goal5_prediction_csv_with_pandas_float_precision_round_trip"
    ),
    "regression_requirement": (
        "exact_r0_p1_probability_label_threshold_parity_after_csv_round_trip"
    ),
    "affected_probe_stage": (
        "goal5_prediction_deserialization_and_exact_bridge_validation_only"
    ),
    "static_delivery_validation_changes": [
        "validate_positive_published_oracle_uplift_for_defined_recovery_ratio",
        "record_real_git_push_error_when_push_fails",
        "authenticate_experiment_commit_subject_ancestry_and_push_state",
    ],
    "run_summary_delivery_schema_adds_push_error": True,
    "config_changed": False,
    "feature_definition_or_extraction_changed": False,
    "checkpoint_or_fold_changed": False,
    "probe_estimator_or_hyperparameter_changed": False,
    "population_or_timing_changed": False,
    "metric_definition_changed": False,
    "bootstrap_changed": False,
    "oracle_denominator_or_numerator_changed": False,
    "gate_threshold_or_logic_changed": False,
    "classification_precedence_changed": False,
    "scientific_contract_changed": False,
    "all_superseded_feature_artifacts_must_be_rebuilt": True,
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


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


def ordered_sha256(values: Iterable[Any]) -> str:
    return hashlib.sha256(
        "\n".join(str(value) for value in values).encode("utf-8")
    ).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    descriptor = {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "bytes_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }
    return canonical_sha256(descriptor)


def private_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)
    return directory


def _atomic_json_payload(
    payload: Mapping[str, Any], path: Path, *, mode: int, replace: bool
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(dict(payload), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(mode)
        if replace:
            temporary.replace(path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError:
                observed = json.loads(path.read_text(encoding="utf-8"))
                if observed != dict(payload):
                    raise ValueError(f"existing immutable JSON differs: {path}")
        path.chmod(mode)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def atomic_json(
    payload: Mapping[str, Any], path: str | Path, *, private: bool = False
) -> Path:
    destination = Path(path)
    if private:
        private_directory(destination.parent)
    return _atomic_json_payload(
        payload, destination, mode=0o600 if private else 0o644, replace=True
    )


def publish_json_once(
    payload: Mapping[str, Any], path: str | Path, *, private: bool = False
) -> Path:
    destination = Path(path)
    if private:
        private_directory(destination.parent)
    return _atomic_json_payload(
        payload, destination, mode=0o600 if private else 0o644, replace=False
    )


def atomic_npz(path: str | Path, arrays: Mapping[str, np.ndarray]) -> Path:
    """Atomically publish an owner-only archive without replacing a peer."""

    destination = Path(path)
    private_directory(destination.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".npz", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.chmod(0o600)
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to replace an existing private archive: {destination}"
            ) from error
        destination.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def cells() -> list[tuple[int, str, int]]:
    return [
        (seed, arm, fold)
        for seed in SEEDS
        for arm in ARMS
        for fold in FOLDS
    ]


def cell_key(seed: int, arm: str, fold: int) -> str:
    identity = (int(seed), str(arm), int(fold))
    if identity not in set(cells()):
        raise ValueError(f"cell is outside the frozen 20-cell matrix: {identity}")
    return f"seed_{identity[0]}/{identity[1]}/fold_{identity[2]}"


def feature_path(seed: int, arm: str, fold: int) -> Path:
    cell_key(seed, arm, fold)
    return (
        EXPERIMENT_ROOT
        / "features"
        / f"seed_{int(seed)}"
        / str(arm)
        / f"fold_{int(fold)}"
        / FEATURE_FILENAME
    )


def metadata_path(seed: int, arm: str, fold: int) -> Path:
    return feature_path(seed, arm, fold).with_name(METADATA_FILENAME)


def _require_sha256(value: Any, *, name: str) -> str:
    normalized = str(value)
    if SHA256_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")
    return normalized


def load_config(
    path: str | Path = CONFIG_PATH, *, verify_extraction_inputs: bool = True
) -> dict[str, Any]:
    source = Path(path).resolve(strict=True)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("audit config must be a schema-v1 JSON object")
    if value.get("experiment") != "mask_free_region_aware_audit":
        raise ValueError("audit config experiment name drifted")
    frozen = value.get("frozen_cells")
    if not isinstance(frozen, dict) or (
        tuple(frozen.get("arms", ())) != ARMS
        or tuple(frozen.get("seed_bases", ())) != SEEDS
        or tuple(frozen.get("folds", ())) != FOLDS
        or tuple(frozen.get("visits", ())) != VISITS
        or frozen.get("patient_count") != PATIENT_COUNT
    ):
        raise ValueError("frozen cell matrix drifted")
    feature = value.get("feature_contract")
    expected_feature = {
        "input_shape_zyx": [112, 176, 160],
        "spacing_xyz_mm": [0.9, 0.9, 2.0],
        "channels": 128,
        "stage_stride_zyx": [8, 8, 8],
        "stage_center_offset_zyx": [0.0, 0.0, 0.0],
        "theoretical_receptive_field_zyx": [47, 47, 47],
        "primary_boundaries_mm": [32.0, 48.0, 64.0],
        "secondary_boundaries_mm": [24.0, 40.0, 64.0],
    }
    if not isinstance(feature, dict) or any(
        feature.get(key) != expected for key, expected in expected_feature.items()
    ):
        raise ValueError("feature geometry contract drifted")
    if feature.get("shape_policy") != "derive_from_runtime_then_validate_frozen_geometry":
        raise ValueError("runtime feature-shape policy drifted")
    projection = feature.get("projection_control")
    if projection != {
        "variant": "R5_RP192",
        "input_dim": 384,
        "output_dim": 192,
        "seed": 260812,
        "method": "fixed_gaussian_reduced_qr_orthonormal_columns",
    }:
        raise ValueError("fixed QR projection contract drifted")
    variants = value.get("variants")
    if not isinstance(variants, dict) or variants.get("dimensions") != VARIANT_DIMENSIONS:
        raise ValueError("regional feature dimensions drifted")
    forbidden = value.get("forbidden")
    if not isinstance(forbidden, list) or len(forbidden) < 10:
        raise ValueError("forbidden-input contract is absent")
    if verify_extraction_inputs:
        # This allow-list deliberately excludes clinical_labels and ftv_table.
        # Extraction authenticates their paths only indirectly through the
        # frozen upstream checkpoint provenance and never opens either table.
        paths = value.get("paths")
        if not isinstance(paths, dict):
            raise ValueError("config paths are absent")
        safe_hash_pairs = (
            ("goal5_lock", "goal5_lock_sha256"),
            ("goal5_feature_completion", "goal5_feature_completion_sha256"),
            ("spatial_sidecar", "spatial_sidecar_sha256"),
            ("stage_b_data_contract", "stage_b_data_contract_sha256"),
            ("stage_b_authorization", "stage_b_authorization_sha256"),
            ("fold_manifest", "fold_manifest_sha256"),
        )
        for path_key, hash_key in safe_hash_pairs:
            expected = _require_sha256(paths.get(hash_key), name=hash_key)
            source_path = Path(str(paths.get(path_key, ""))).resolve(strict=True)
            if file_sha256(source_path) != expected:
                raise ValueError(f"frozen extraction input drifted: {path_key}")
    return value


def implementation_inventory() -> dict[str, str]:
    files = sorted((EXPERIMENT_ROOT / "scripts").glob("*.py")) + sorted(
        (EXPERIMENT_ROOT / "tests").glob("*.py")
    )
    return {
        str(path.relative_to(EXPERIMENT_ROOT)): file_sha256(path)
        for path in files
        if path.is_file()
    }


def _git_bytes(*arguments: str) -> bytes:
    result = subprocess.run(
        ("git", *arguments),
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(message or f"git {' '.join(arguments)} failed")
    return result.stdout


def historical_file_bytes(commit: str, relative_path: str | Path) -> bytes:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("historical path must be repository-relative")
    return _git_bytes("show", f"{commit}:{relative.as_posix()}")


def _experiment_relative(path: Path) -> Path:
    return path.resolve().relative_to(REPO_ROOT.resolve())


def require_prior_preregistration() -> dict[str, Any]:
    """Authenticate the exact schema-1 lock committed before the failed run."""

    ancestry = subprocess.run(
        (
            "git",
            "merge-base",
            "--is-ancestor",
            PRIOR_PREREGISTRATION_COMMIT,
            "HEAD",
        ),
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if ancestry.returncode != 0:
        raise ValueError("prior preregistration commit is not an ancestor of HEAD")
    lock_bytes = historical_file_bytes(
        PRIOR_PREREGISTRATION_COMMIT, _experiment_relative(LOCK_PATH)
    )
    if hashlib.sha256(lock_bytes).hexdigest() != PRIOR_PREREGISTRATION_LOCK_SHA256:
        raise ValueError("historical schema-1 preregistration lock bytes drifted")
    try:
        prior = json.loads(lock_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("historical schema-1 lock is not valid UTF-8 JSON") from error
    if (
        not isinstance(prior, dict)
        or set(prior) != set(INITIAL_LOCK_KEYS)
        or prior.get("schema_version") != 1
        or prior.get("status") != LOCK_STATUS
        or prior.get("pre_freeze_result_inventory") != ZERO_RESULT_INVENTORY
        or prior.get("privacy_contract") != PRIVACY_CONTRACT
    ):
        raise ValueError("historical schema-1 preregistration contract drifted")
    historical_paths = {
        "config_sha256": CONFIG_PATH,
        "experiment_plan_sha256": PLAN_PATH,
        "gitignore_sha256": GITIGNORE_PATH,
    }
    for key, path in historical_paths.items():
        payload = historical_file_bytes(
            PRIOR_PREREGISTRATION_COMMIT, _experiment_relative(path)
        )
        if hashlib.sha256(payload).hexdigest() != prior.get(key):
            raise ValueError(f"historical schema-1 lock does not bind {path.name}")
    return prior


def require_implementation_erratum() -> dict[str, Any]:
    """Require the exact compatibility-only erratum semantics."""

    value = json.loads(IMPLEMENTATION_ERRATUM_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != set(IMPLEMENTATION_ERRATUM_KEYS):
        raise ValueError("implementation erratum schema drifted")
    discarded = value.get("discarded_artifact_sha256")
    discarded_counts = {
        root: sum(str(path).startswith(root + "/") for path in discarded)
        if isinstance(discarded, Mapping)
        else -1
        for root in ("features", "logs", "metrics", "predictions", "figures", "reports")
    }
    if (
        value.get("schema_version") != 1
        or value.get("status") != IMPLEMENTATION_ERRATUM_STATUS
        or value.get("erratum_number") != 1
        or value.get("prior_preregistration_commit")
        != PRIOR_PREREGISTRATION_COMMIT
        or value.get("prior_preregistration_lock_sha256")
        != PRIOR_PREREGISTRATION_LOCK_SHA256
        or value.get("reason_code") != IMPLEMENTATION_ERRATUM_REASON
        or value.get("pre_erratum_execution")
        != IMPLEMENTATION_ERRATUM_PRE_EXECUTION
        or value.get("contract_scope") != IMPLEMENTATION_ERRATUM_CONTRACT_SCOPE
        or not isinstance(discarded, Mapping)
        or len(discarded) != 62
        or any(
            not isinstance(path, str)
            or not isinstance(digest, str)
            or SHA256_PATTERN.fullmatch(digest) is None
            for path, digest in discarded.items()
        )
        or discarded_counts
        != IMPLEMENTATION_ERRATUM_PRE_EXECUTION["discarded_artifact_count_by_root"]
        or canonical_sha256(discarded)
        != ERRATUM_DISCARDED_MAP_CANONICAL_SHA256
        or discarded.get("features/feature_matrix_complete.private.json")
        != IMPLEMENTATION_ERRATUM_PRE_EXECUTION[
            "superseded_feature_completion_sha256"
        ]
        or discarded.get("metrics/region_occupancy_contract.json")
        != IMPLEMENTATION_ERRATUM_PRE_EXECUTION[
            "superseded_geometry_contract_sha256"
        ]
        or value.get("contains_patient_identifiers") is not False
    ):
        raise ValueError("implementation erratum content drifted")
    return value


def _require_erratum_1_plan_bytes(
    prior: Mapping[str, Any], current_bytes: bytes
) -> None:
    """Validate the exact first append-only plan disclosure bytes."""

    prior_bytes = historical_file_bytes(
        PRIOR_PREREGISTRATION_COMMIT, _experiment_relative(PLAN_PATH)
    )
    if hashlib.sha256(prior_bytes).hexdigest() != prior.get(
        "experiment_plan_sha256"
    ):
        raise ValueError("historical plan differs from the prior lock")
    if not current_bytes.startswith(prior_bytes):
        raise ValueError("pre-erratum scientific plan text changed")
    appendix = current_bytes[len(prior_bytes) :]
    if (
        hashlib.sha256(appendix).hexdigest() != ERRATUM_PLAN_APPENDIX_SHA256
        or not appendix.startswith(
            b"## 10. Implementation compatibility erratum 1 and refreeze\n"
        )
    ):
        raise ValueError("implementation erratum plan disclosure drifted")


def require_erratum_plan_disclosure(prior: Mapping[str, Any]) -> None:
    """Prove the old scientific plan is unchanged and only erratum 1 is appended."""

    _require_erratum_1_plan_bytes(prior, PLAN_PATH.read_bytes())


def require_prior_compatibility_refreeze() -> dict[str, Any]:
    """Authenticate the exact committed schema-2 lock superseded by erratum 2."""

    ancestry = subprocess.run(
        (
            "git",
            "merge-base",
            "--is-ancestor",
            PRIOR_COMPATIBILITY_REFREEZE_COMMIT,
            "HEAD",
        ),
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if ancestry.returncode != 0:
        raise ValueError("prior compatibility refreeze commit is not an ancestor")
    lock_bytes = historical_file_bytes(
        PRIOR_COMPATIBILITY_REFREEZE_COMMIT, _experiment_relative(LOCK_PATH)
    )
    if (
        hashlib.sha256(lock_bytes).hexdigest()
        != PRIOR_COMPATIBILITY_REFREEZE_LOCK_SHA256
    ):
        raise ValueError("historical schema-2 preregistration lock bytes drifted")
    try:
        prior = json.loads(lock_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("historical schema-2 lock is not valid UTF-8 JSON") from error
    original = require_prior_preregistration()
    preserved_fields = (
        "branch",
        "parent_commit_sha",
        "config_sha256",
        "config_canonical_sha256",
        "gitignore_sha256",
        "goal5_preregistration_lock_sha256",
        "goal5_feature_completion_sha256",
        "formal_cell_count",
        "selected_cells",
        "pre_freeze_result_inventory",
        "privacy_contract",
    )
    if (
        not isinstance(prior, dict)
        or set(prior) != set(REFREEZE_LOCK_KEYS)
        or prior.get("schema_version") != 2
        or prior.get("preregistration_revision") != 1
        or prior.get("status") != REFREEZE_LOCK_STATUS
        or prior.get("prior_preregistration_commit")
        != PRIOR_PREREGISTRATION_COMMIT
        or prior.get("prior_preregistration_lock_sha256")
        != PRIOR_PREREGISTRATION_LOCK_SHA256
        or prior.get("implementation_erratum_sha256")
        != file_sha256(IMPLEMENTATION_ERRATUM_PATH)
        or prior.get("superseded_artifacts_reused") is not False
        or prior.get("scientific_contract_unchanged") is not True
        or prior.get("pre_refreeze_result_inventory") != ZERO_RESULT_INVENTORY
        or any(prior.get(name) != original.get(name) for name in preserved_fields)
    ):
        raise ValueError("historical schema-2 compatibility refreeze contract drifted")
    require_implementation_erratum()
    require_refreeze_git_provenance(prior["git_provenance_before_refreeze"])
    if prior["git_provenance_before_refreeze"]["branch"] != prior["branch"]:
        raise ValueError("historical schema-2 refreeze branch record drifted")
    historical_files = {
        "config_sha256": CONFIG_PATH,
        "experiment_plan_sha256": PLAN_PATH,
        "gitignore_sha256": GITIGNORE_PATH,
        "implementation_erratum_sha256": IMPLEMENTATION_ERRATUM_PATH,
    }
    for key, path in historical_files.items():
        payload = historical_file_bytes(
            PRIOR_COMPATIBILITY_REFREEZE_COMMIT, _experiment_relative(path)
        )
        if hashlib.sha256(payload).hexdigest() != prior.get(key):
            raise ValueError(f"historical schema-2 lock does not bind {path.name}")
    plan_bytes = historical_file_bytes(
        PRIOR_COMPATIBILITY_REFREEZE_COMMIT, _experiment_relative(PLAN_PATH)
    )
    _require_erratum_1_plan_bytes(original, plan_bytes)
    implementation = prior.get("implementation_sha256")
    if (
        not isinstance(implementation, Mapping)
        or not implementation
        or any(
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or len(Path(relative).parts) != 2
            or Path(relative).parts[0] not in {"scripts", "tests"}
            or Path(relative).suffix != ".py"
            or SHA256_PATTERN.fullmatch(str(expected)) is None
            for relative, expected in implementation.items()
        )
    ):
        raise ValueError("historical schema-2 implementation inventory drifted")
    for relative, expected in implementation.items():
        payload = historical_file_bytes(
            PRIOR_COMPATIBILITY_REFREEZE_COMMIT,
            _experiment_relative(EXPERIMENT_ROOT / str(relative)),
        )
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"historical schema-2 implementation drifted: {relative}")
    return prior


def require_implementation_erratum_2() -> dict[str, Any]:
    """Require the exact output/provenance-only second erratum."""

    if file_sha256(IMPLEMENTATION_ERRATUM_2_PATH) != IMPLEMENTATION_ERRATUM_2_SHA256:
        raise ValueError("second implementation erratum bytes drifted")
    value = json.loads(IMPLEMENTATION_ERRATUM_2_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != set(IMPLEMENTATION_ERRATUM_2_KEYS):
        raise ValueError("second implementation erratum schema drifted")
    execution = value.get("pre_erratum_execution")
    scope = value.get("contract_scope")
    discarded = value.get("discarded_artifact_sha256")
    roots = ("features", "logs", "metrics", "predictions", "figures", "reports")
    discarded_counts = {
        root: sum(str(path).startswith(root + "/") for path in discarded)
        if isinstance(discarded, Mapping)
        else -1
        for root in roots
    }
    if (
        value.get("schema_version") != 1
        or value.get("status") != IMPLEMENTATION_ERRATUM_2_STATUS
        or value.get("erratum_number") != 2
        or value.get("prior_compatibility_refreeze_commit")
        != PRIOR_COMPATIBILITY_REFREEZE_COMMIT
        or value.get("prior_compatibility_refreeze_lock_sha256")
        != PRIOR_COMPATIBILITY_REFREEZE_LOCK_SHA256
        or value.get("prior_implementation_erratum_sha256")
        != file_sha256(IMPLEMENTATION_ERRATUM_PATH)
        or value.get("reason_code") != IMPLEMENTATION_ERRATUM_2_REASON
        or canonical_sha256(execution)
        != ERRATUM_2_PRE_EXECUTION_CANONICAL_SHA256
        or canonical_sha256(scope)
        != ERRATUM_2_CONTRACT_SCOPE_CANONICAL_SHA256
        or not isinstance(discarded, Mapping)
        or len(discarded) != 94
        or any(
            not isinstance(path, str)
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or not isinstance(digest, str)
            or SHA256_PATTERN.fullmatch(digest) is None
            for path, digest in discarded.items()
        )
        or discarded_counts
        != execution.get("discarded_artifact_count_by_root")
        or canonical_sha256(discarded)
        != ERRATUM_2_DISCARDED_MAP_CANONICAL_SHA256
        or execution.get("discarded_artifact_record_set_sha256")
        != ERRATUM_2_DISCARDED_RECORD_SET_SHA256
        or discarded.get("features/feature_matrix_complete.private.json")
        != execution.get("superseded_feature_completion_sha256")
        or discarded.get("metrics/table_bootstrap.csv")
        != execution.get("superseded_bootstrap_summary_sha256")
        or discarded.get("predictions/bootstrap_draws.private.csv")
        != execution.get("superseded_bootstrap_draws_sha256")
        or discarded.get("metrics/gates.json")
        != execution.get("superseded_gates_sha256")
        or discarded.get("metrics/run_summary.json")
        != execution.get("superseded_run_summary_sha256")
        or discarded.get("reports/report_manifest.json")
        != execution.get("superseded_report_manifest_sha256")
        or value.get("contains_patient_identifiers") is not False
    ):
        raise ValueError("second implementation erratum content drifted")
    return value


def require_erratum_2_plan_disclosure(prior: Mapping[str, Any]) -> None:
    """Prove schema-2 plan bytes are unchanged and only erratum 2 is appended."""

    prior_bytes = historical_file_bytes(
        PRIOR_COMPATIBILITY_REFREEZE_COMMIT, _experiment_relative(PLAN_PATH)
    )
    if hashlib.sha256(prior_bytes).hexdigest() != prior.get(
        "experiment_plan_sha256"
    ):
        raise ValueError("historical schema-2 plan differs from its lock")
    current_bytes = PLAN_PATH.read_bytes()
    if not current_bytes.startswith(prior_bytes):
        raise ValueError("schema-2 scientific plan text changed")
    appendix = current_bytes[len(prior_bytes) :]
    if (
        hashlib.sha256(appendix).hexdigest() != ERRATUM_2_PLAN_APPENDIX_SHA256
        or not appendix.startswith(
            b"\n## 11. Implementation output erratum 2 and schema-3 refreeze\n"
        )
    ):
        raise ValueError("second implementation erratum plan disclosure drifted")


def require_refreeze_git_provenance(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "base_head",
        "branch",
        "all_dirty_paths_confined_to_new_experiment",
        "tracked_paths_before_refreeze",
        "untracked_paths_before_refreeze",
    }:
        raise ValueError("schema-2 refreeze Git provenance schema drifted")
    tracked = value.get("tracked_paths_before_refreeze")
    untracked = value.get("untracked_paths_before_refreeze")
    if (
        value.get("base_head") != PRIOR_PREREGISTRATION_COMMIT
        or value.get("all_dirty_paths_confined_to_new_experiment") is not True
        or not isinstance(tracked, list)
        or not isinstance(untracked, list)
        or tracked != sorted(set(tracked))
        or untracked != sorted(set(untracked))
    ):
        raise ValueError("schema-2 refreeze Git provenance content drifted")
    prefix = _experiment_relative(EXPERIMENT_ROOT).as_posix() + "/"
    paths = [*tracked, *untracked]
    if not paths or any(not isinstance(path, str) or not path.startswith(prefix) for path in paths):
        raise ValueError("schema-2 refreeze dirty paths escaped the experiment")
    required_dirty = {
        prefix + "EXPERIMENT_PLAN.md",
        prefix + "scripts/common.py",
        prefix + "scripts/freeze_preregistration.py",
        prefix + "scripts/run_audit.py",
        prefix + "tests/test_extraction_contract.py",
        prefix + "PREREGISTRATION_IMPLEMENTATION_ERRATUM.json",
    }
    if not required_dirty.issubset(paths):
        raise ValueError("schema-2 refreeze omits required implementation changes")
    forbidden = {prefix + "configs/audit.json", prefix + ".gitignore"}
    if forbidden.intersection(paths):
        raise ValueError("schema-2 refreeze changed config or privacy policy")


def require_refreeze_2_git_provenance(value: Any) -> None:
    """Validate the exact Git-context schema captured before schema 3."""

    if not isinstance(value, Mapping) or set(value) != {
        "base_head",
        "branch",
        "all_dirty_paths_confined_to_new_experiment",
        "tracked_paths_before_refreeze",
        "untracked_paths_before_refreeze",
    }:
        raise ValueError("schema-3 refreeze Git provenance schema drifted")
    tracked = value.get("tracked_paths_before_refreeze")
    untracked = value.get("untracked_paths_before_refreeze")
    if (
        value.get("base_head") != PRIOR_COMPATIBILITY_REFREEZE_COMMIT
        or value.get("all_dirty_paths_confined_to_new_experiment") is not True
        or not isinstance(tracked, list)
        or not isinstance(untracked, list)
        or tracked != sorted(set(tracked))
        or untracked != sorted(set(untracked))
    ):
        raise ValueError("schema-3 refreeze Git provenance content drifted")
    prefix = _experiment_relative(EXPERIMENT_ROOT).as_posix() + "/"
    paths = [*tracked, *untracked]
    if not paths or any(
        not isinstance(path, str) or not path.startswith(prefix) for path in paths
    ):
        raise ValueError("schema-3 refreeze dirty paths escaped the experiment")
    required_dirty = {
        prefix + "EXPERIMENT_PLAN.md",
        prefix + "PREREGISTRATION_IMPLEMENTATION_ERRATUM_2.json",
        prefix + "scripts/common.py",
        prefix + "scripts/freeze_preregistration.py",
        prefix + "scripts/generate_figures.py",
        prefix + "scripts/generate_report.py",
        prefix + "scripts/run_audit.py",
        prefix + "scripts/validate_results.py",
        prefix + "tests/test_analysis.py",
        prefix + "tests/test_extraction_contract.py",
        prefix + "tests/test_reporting.py",
    }
    if not required_dirty.issubset(paths):
        raise ValueError("schema-3 refreeze omits required implementation changes")
    forbidden = {
        prefix + "configs/audit.json",
        prefix + ".gitignore",
        prefix + "PREREGISTRATION_IMPLEMENTATION_ERRATUM.json",
    }
    if forbidden.intersection(paths):
        raise ValueError("schema-3 refreeze changed config, privacy, or erratum 1")


def load_goal5_lock(config: Mapping[str, Any]) -> dict[str, Any]:
    paths = config["paths"]
    source = Path(str(paths["goal5_lock"])).resolve(strict=True)
    if file_sha256(source) != paths["goal5_lock_sha256"]:
        raise ValueError("Goal5 preregistration lock drifted")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("formal_cell_count") != 20:
        raise ValueError("Goal5 preregistration lock schema drifted")
    if set(value.get("selected_cells", {})) != {
        cell_key(seed, arm, fold) for seed, arm, fold in cells()
    }:
        raise ValueError("Goal5 selected-cell inventory drifted")
    return value


def require_preregistration_lock(
    config: Mapping[str, Any], path: str | Path = LOCK_PATH
) -> dict[str, Any]:
    # Analysis resolves configured paths to ``Path`` objects and appends two
    # runtime-only keys.  Authenticate that view against the on-disk config,
    # but hash only the exact frozen JSON object used by the freezer.
    frozen_config = load_config(CONFIG_PATH, verify_extraction_inputs=False)
    for name in ("schema_version", "experiment", "branch", "start"):
        if config.get(name) != frozen_config.get(name):
            raise ValueError(f"caller config differs from frozen config at {name}")
    caller_digest = config.get("config_sha256")
    if caller_digest is not None and caller_digest != file_sha256(CONFIG_PATH):
        raise ValueError("caller config SHA-256 differs from the frozen config")
    source = Path(path).resolve(strict=True)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("preregistration lock is not a JSON object")

    prior: dict[str, Any] | None = None
    if IMPLEMENTATION_ERRATUM_2_PATH.is_file():
        if set(value) != set(REFREEZE_2_LOCK_KEYS):
            raise ValueError(
                "schema-3 preregistration is required after erratum 2"
            )
        prior = require_prior_compatibility_refreeze()
        require_implementation_erratum_2()
        require_erratum_2_plan_disclosure(prior)
        if (
            value["schema_version"] != 3
            or value["preregistration_revision"] != 2
            or value["status"] != REFREEZE_2_LOCK_STATUS
            or value["prior_compatibility_refreeze_commit"]
            != PRIOR_COMPATIBILITY_REFREEZE_COMMIT
            or value["prior_compatibility_refreeze_lock_sha256"]
            != PRIOR_COMPATIBILITY_REFREEZE_LOCK_SHA256
            or value["implementation_erratum_2_sha256"]
            != file_sha256(IMPLEMENTATION_ERRATUM_2_PATH)
            or value["superseded_formal_run_artifact_count"] != 94
            or value["superseded_formal_run_artifact_record_set_sha256"]
            != ERRATUM_2_DISCARDED_RECORD_SET_SHA256
            or value["all_twenty_feature_cells_rebuild_required"] is not True
            or value["superseded_artifacts_reused"] is not False
            or value["scientific_contract_unchanged"] is not True
            or value["pre_refreeze_2_result_inventory"]
            != ZERO_RESULT_INVENTORY
        ):
            raise ValueError("schema-3 output/reporting refreeze is not active")
        require_refreeze_2_git_provenance(
            value["git_provenance_before_refreeze_2"]
        )
        if value["git_provenance_before_refreeze_2"]["branch"] != value["branch"]:
            raise ValueError("schema-3 refreeze branch record drifted")
        preserved_fields = (
            "branch",
            "parent_commit_sha",
            "config_sha256",
            "config_canonical_sha256",
            "gitignore_sha256",
            "goal5_preregistration_lock_sha256",
            "goal5_feature_completion_sha256",
            "formal_cell_count",
            "selected_cells",
            "pre_freeze_result_inventory",
            "privacy_contract",
            "prior_preregistration_commit",
            "prior_preregistration_lock_sha256",
            "implementation_erratum_sha256",
            "superseded_artifacts_reused",
            "scientific_contract_unchanged",
            "pre_refreeze_result_inventory",
            "git_provenance_before_refreeze",
        )
        if any(value[name] != prior[name] for name in preserved_fields):
            raise ValueError("schema-3 refreeze changed a frozen schema-2 field")
    elif IMPLEMENTATION_ERRATUM_PATH.is_file():
        if set(value) != set(REFREEZE_LOCK_KEYS):
            raise ValueError("schema-2 preregistration lock schema drifted")
        prior = require_prior_preregistration()
        require_implementation_erratum()
        require_erratum_plan_disclosure(prior)
        if (
            value["schema_version"] != 2
            or value["preregistration_revision"] != 1
            or value["status"] != REFREEZE_LOCK_STATUS
            or value["prior_preregistration_commit"]
            != PRIOR_PREREGISTRATION_COMMIT
            or value["prior_preregistration_lock_sha256"]
            != PRIOR_PREREGISTRATION_LOCK_SHA256
            or value["implementation_erratum_sha256"]
            != file_sha256(IMPLEMENTATION_ERRATUM_PATH)
            or value["superseded_artifacts_reused"] is not False
            or value["scientific_contract_unchanged"] is not True
            or value["pre_refreeze_result_inventory"] != ZERO_RESULT_INVENTORY
        ):
            raise ValueError("schema-2 compatibility refreeze is not active")
        require_refreeze_git_provenance(value["git_provenance_before_refreeze"])
        if value["git_provenance_before_refreeze"]["branch"] != value["branch"]:
            raise ValueError("schema-2 refreeze branch record drifted")
        preserved_fields = (
            "branch",
            "parent_commit_sha",
            "config_sha256",
            "config_canonical_sha256",
            "gitignore_sha256",
            "goal5_preregistration_lock_sha256",
            "goal5_feature_completion_sha256",
            "formal_cell_count",
            "selected_cells",
            "pre_freeze_result_inventory",
            "privacy_contract",
        )
        if any(value[name] != prior[name] for name in preserved_fields):
            raise ValueError("schema-2 refreeze changed a frozen schema-1 field")
    else:
        if set(value) != set(INITIAL_LOCK_KEYS):
            raise ValueError("schema-1 preregistration lock schema drifted")
        if value["schema_version"] != 1 or value["status"] != LOCK_STATUS:
            raise ValueError("schema-1 preregistration lock is not active")

    if (
        value["branch"] != frozen_config["branch"]
        or value["parent_commit_sha"]
        != frozen_config["start"]["parent_commit_sha"]
    ):
        raise ValueError("preregistration git provenance drifted")
    if value["config_sha256"] != file_sha256(CONFIG_PATH) or value[
        "config_canonical_sha256"
    ] != canonical_sha256(frozen_config):
        raise ValueError("config differs from preregistration lock")
    if value["experiment_plan_sha256"] != file_sha256(PLAN_PATH) or value[
        "gitignore_sha256"
    ] != file_sha256(GITIGNORE_PATH):
        raise ValueError("plan/privacy policy differs from preregistration lock")
    observed_implementation = implementation_inventory()
    if value["implementation_sha256"] != observed_implementation:
        raise ValueError("implementation differs from preregistration lock")
    if value["goal5_preregistration_lock_sha256"] != frozen_config["paths"]["goal5_lock_sha256"]:
        raise ValueError("Goal5 lock anchor drifted")
    if value["goal5_feature_completion_sha256"] != frozen_config["paths"]["goal5_feature_completion_sha256"]:
        raise ValueError("Goal5 feature-completion anchor drifted")
    goal5 = load_goal5_lock(frozen_config)
    if value["formal_cell_count"] != 20 or value["selected_cells"] != goal5["selected_cells"]:
        raise ValueError("selected checkpoints differ from the exact Goal5 lock")
    if value["pre_freeze_result_inventory"] != ZERO_RESULT_INVENTORY:
        raise ValueError("preregistration lock did not precede result generation")
    if value["privacy_contract"] != PRIVACY_CONTRACT:
        raise ValueError("preregistration privacy contract drifted")
    if prior is not None and (
        value["config_sha256"] != prior["config_sha256"]
        or value["config_canonical_sha256"] != prior["config_canonical_sha256"]
        or value["gitignore_sha256"] != prior["gitignore_sha256"]
    ):
        raise ValueError("refreeze changed config or privacy bytes")
    return value


def require_owner_only(path: str | Path) -> Path:
    source = Path(path).resolve(strict=True)
    if source.stat().st_mode & 0o077:
        raise PermissionError(f"private artifact must remain owner-only: {source}")
    return source


def validate_feature_cell(
    path: str | Path,
    config: Mapping[str, Any],
    lock: Mapping[str, Any],
    *,
    seed: int,
    arm: str,
    fold: int,
) -> dict[str, Any]:
    """Strictly authenticate one owner-only regional feature cell."""

    key = cell_key(seed, arm, fold)
    source = require_owner_only(path)
    expected_source = feature_path(seed, arm, fold).resolve()
    if source != expected_source:
        raise ValueError(f"feature cell path drifted: {source}")
    metadata_source = require_owner_only(metadata_path(seed, arm, fold))
    metadata = json.loads(metadata_source.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or set(metadata) != set(METADATA_KEYS):
        raise ValueError(f"feature metadata schema drifted: {metadata_source}")
    record = lock["selected_cells"][key]
    if (
        metadata["schema_version"] != 1
        or metadata["status"] != "COMPLETE"
        or metadata["experiment"] != config["experiment"]
        or metadata["cell"] != key
        or metadata["seed_base"] != seed
        or metadata["arm"] != arm
        or metadata["fold"] != fold
        or metadata["checkpoint_sha256"] != record["checkpoint_sha256"]
        or metadata["selection_sha256"] != record["selection_sha256"]
        or metadata["reference_feature_sha256"] != record["reference"]["sha256"]
        or Path(metadata["feature_path"]).resolve() != source
        or metadata["feature_sha256"] != file_sha256(source)
        or metadata["patient_count"] != PATIENT_COUNT
        or metadata["visit_count"] != len(VISITS)
        or metadata["config_sha256"] != file_sha256(CONFIG_PATH)
        or metadata["preregistration_lock_sha256"] != file_sha256(LOCK_PATH)
        or metadata["goal5_preregistration_lock_sha256"]
        != config["paths"]["goal5_lock_sha256"]
    ):
        raise ValueError(f"feature metadata identity/provenance drifted: {metadata_source}")
    if metadata["variant_shapes"] != {
        name: [PATIENT_COUNT, len(VISITS), dimension]
        for name, dimension in VARIANT_DIMENSIONS.items()
    }:
        raise ValueError("feature metadata variant shapes drifted")
    if metadata["variant_dtypes"] != {
        name: "float32" for name in VARIANT_DIMENSIONS
    }:
        raise ValueError("feature metadata variant dtypes drifted")
    feature_contract = config["feature_contract"]
    input_shape = tuple(int(value) for value in feature_contract["input_shape_zyx"])
    stride = tuple(int(value) for value in feature_contract["stage_stride_zyx"])
    receptive_field = tuple(
        int(value) for value in feature_contract["theoretical_receptive_field_zyx"]
    )
    padding = tuple((value - 1) // 2 for value in receptive_field)
    expected_grid = tuple(
        (size + 2 * pad - kernel) // step + 1
        for size, step, kernel, pad in zip(
            input_shape, stride, receptive_field, padding, strict=True
        )
    )
    if metadata["actual_encoder_shape"] != [
        PATIENT_COUNT,
        len(VISITS),
        128,
        *expected_grid,
    ]:
        raise ValueError("feature metadata encoder shape drifted")
    if (
        metadata["actual_encoder_dtype"] != "float32"
        or metadata["checkpoint_c1b_local_weight_bitwise_equal"] is not True
        or metadata["r0_goal5_mean_parity"].get("bitwise_equal") is not True
        or metadata["projected_r0_local_state_parity"].get("allclose") is not True
        or metadata["encoder_frozen"] is not True
        or metadata["training_performed"] is not False
        or metadata["streamed_raw_spatial_map_not_persisted"] is not True
        or metadata["response_projection_used_only_for_parity"] is not True
    ):
        raise ValueError("feature metadata parity/frozen contract failed")
    forbidden_true = (
        "projector_called",
        "transition_called",
        "target_encoder_called",
        "ftv_head_called",
        "lesion_mask_read",
        "tumor_bbox_read",
        "clinical_label_table_read",
        "ftv_value_table_read",
        "phenotype_or_pcr_labels_read",
        "future_visit_used_to_define_region",
    )
    if any(metadata[name] is not False for name in forbidden_true):
        raise ValueError("feature metadata reports forbidden execution/input access")
    geometry_path = Path(str(metadata["geometry_contract_path"])).resolve(strict=True)
    if geometry_path != GEOMETRY_CONTRACT_PATH.resolve() or file_sha256(
        geometry_path
    ) != metadata["geometry_contract_sha256"]:
        raise ValueError("geometry contract provenance drifted")
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    if (
        geometry.get("status") != "GEOMETRY_VALID"
        or geometry.get("feature_shape_zyx") != list(expected_grid)
        or geometry.get("contains_patient_data") is not False
        or geometry.get("uses_mask_bbox_ftv_clinical_treatment_phenotype_or_outcome")
        is not False
        or geometry.get("projection", {}).get("matrix_float32_sha256")
        != metadata["projection_matrix_float32_sha256"]
    ):
        raise ValueError("public geometry/occupancy contract drifted")
    weight_hashes = metadata["region_weight_sha256"]
    if not isinstance(weight_hashes, dict) or set(weight_hashes) != {
        "R0",
        "R1",
        "R2",
        "R3",
        "S1",
        "S2",
        "S3",
    } or any(SHA256_PATTERN.fullmatch(str(value)) is None for value in weight_hashes.values()):
        raise ValueError("regional weight provenance drifted")
    implementation = metadata["implementation_sha256"]
    expected_implementation = {
        name: lock["implementation_sha256"][name]
        for name in (
            "scripts/export_features.py",
            "scripts/regions.py",
            "scripts/common.py",
        )
    }
    if implementation != expected_implementation:
        raise ValueError("feature implementation provenance drifted")
    completion_path = Path(str(config["paths"]["goal5_feature_completion"])).resolve(
        strict=True
    )
    if file_sha256(completion_path) != config["paths"]["goal5_feature_completion_sha256"]:
        raise ValueError("Goal5 feature completion drifted")
    goal5_completion = json.loads(completion_path.read_text(encoding="utf-8"))
    goal5_matches = [
        item
        for item in goal5_completion.get("cells", [])
        if (item.get("seed"), item.get("arm"), item.get("fold"))
        == (seed, arm, fold)
    ]
    if (
        len(goal5_matches) != 1
        or goal5_matches[0].get("feature_sha256") != metadata["goal5_feature_sha256"]
    ):
        raise ValueError("Goal5 feature-cell provenance drifted")
    with np.load(source, allow_pickle=False) as archive:
        if tuple(archive.files) != FEATURE_KEYS:
            raise ValueError(f"feature archive schema/order drifted: {source}")
        patient_id = np.asarray(archive["patient_id"]).astype(str)
        split = np.asarray(archive["split"]).astype(str)
        if (
            patient_id.shape != (PATIENT_COUNT,)
            or split.shape != (PATIENT_COUNT,)
            or len(set(patient_id)) != PATIENT_COUNT
            or not set(split).issubset({"train", "val", "test"})
            or ordered_sha256(patient_id) != metadata["patient_order_sha256"]
            or ordered_sha256(split) != metadata["split_order_sha256"]
            or ordered_sha256(patient_id) != record["reference"]["patient_order_sha256"]
            or ordered_sha256(split) != record["reference"]["split_order_sha256"]
        ):
            raise ValueError(f"feature archive patient/split order drifted: {source}")
        for name, dimension in VARIANT_DIMENSIONS.items():
            value = np.asarray(archive[name])
            if (
                value.shape != (PATIENT_COUNT, len(VISITS), dimension)
                or value.dtype != np.float32
                or not np.isfinite(value).all()
            ):
                raise ValueError(f"feature archive {name} shape/dtype/value drifted")
        if (
            np.asarray(archive["arm"]).shape != ()
            or str(np.asarray(archive["arm"]).item()) != arm
            or np.asarray(archive["seed_base"]).shape != ()
            or int(np.asarray(archive["seed_base"]).item()) != seed
            or np.asarray(archive["fold"]).shape != ()
            or int(np.asarray(archive["fold"]).item()) != fold
        ):
            raise ValueError("feature archive cell identity drifted")
    return metadata


__all__ = [
    "ARMS",
    "COMPLETION_CELL_KEYS",
    "COMPLETION_KEYS",
    "COMPLETION_PATH",
    "CONFIG_PATH",
    "ERRATUM_2_CONTRACT_SCOPE_CANONICAL_SHA256",
    "ERRATUM_2_DISCARDED_MAP_CANONICAL_SHA256",
    "ERRATUM_2_DISCARDED_RECORD_SET_SHA256",
    "ERRATUM_2_PLAN_APPENDIX_SHA256",
    "ERRATUM_2_PRE_EXECUTION_CANONICAL_SHA256",
    "ERRATUM_DISCARDED_MAP_CANONICAL_SHA256",
    "EXPERIMENT_ROOT",
    "FEATURE_FILENAME",
    "FEATURE_KEYS",
    "FOLDS",
    "GEOMETRY_CONTRACT_PATH",
    "GITIGNORE_PATH",
    "IMPLEMENTATION_ERRATUM_2_KEYS",
    "IMPLEMENTATION_ERRATUM_2_PATH",
    "IMPLEMENTATION_ERRATUM_2_REASON",
    "IMPLEMENTATION_ERRATUM_2_SHA256",
    "IMPLEMENTATION_ERRATUM_2_STATUS",
    "IMPLEMENTATION_ERRATUM_CONTRACT_SCOPE",
    "IMPLEMENTATION_ERRATUM_KEYS",
    "IMPLEMENTATION_ERRATUM_PATH",
    "IMPLEMENTATION_ERRATUM_PRE_EXECUTION",
    "IMPLEMENTATION_ERRATUM_REASON",
    "IMPLEMENTATION_ERRATUM_STATUS",
    "INITIAL_LOCK_KEYS",
    "LOCK_PATH",
    "LOCK_STATUS",
    "METADATA_FILENAME",
    "METADATA_KEYS",
    "PATIENT_COUNT",
    "PLAN_PATH",
    "PRIOR_COMPATIBILITY_REFREEZE_COMMIT",
    "PRIOR_COMPATIBILITY_REFREEZE_LOCK_SHA256",
    "PRIOR_PREREGISTRATION_COMMIT",
    "PRIOR_PREREGISTRATION_LOCK_SHA256",
    "PRIVACY_CONTRACT",
    "REPO_ROOT",
    "REFREEZE_2_LOCK_KEYS",
    "REFREEZE_2_LOCK_STATUS",
    "REFREEZE_LOCK_KEYS",
    "REFREEZE_LOCK_STATUS",
    "SEEDS",
    "VARIANT_DIMENSIONS",
    "VARIANT_KEYS",
    "VISITS",
    "ZERO_RESULT_INVENTORY",
    "array_sha256",
    "atomic_json",
    "atomic_npz",
    "canonical_sha256",
    "cell_key",
    "cells",
    "feature_path",
    "file_sha256",
    "implementation_inventory",
    "historical_file_bytes",
    "load_config",
    "load_goal5_lock",
    "metadata_path",
    "ordered_sha256",
    "private_directory",
    "publish_json_once",
    "require_owner_only",
    "require_erratum_2_plan_disclosure",
    "require_erratum_plan_disclosure",
    "require_implementation_erratum",
    "require_implementation_erratum_2",
    "require_prior_compatibility_refreeze",
    "require_prior_preregistration",
    "require_preregistration_lock",
    "require_refreeze_2_git_provenance",
    "require_refreeze_git_provenance",
    "validate_feature_cell",
]

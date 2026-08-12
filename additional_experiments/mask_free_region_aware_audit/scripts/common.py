"""Fail-closed contracts shared by the mask-free regional audit."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
CONFIG_PATH = EXPERIMENT_ROOT / "configs" / "audit.json"
PLAN_PATH = EXPERIMENT_ROOT / "EXPERIMENT_PLAN.md"
GITIGNORE_PATH = EXPERIMENT_ROOT / ".gitignore"
LOCK_PATH = EXPERIMENT_ROOT / "PREREGISTRATION_LOCK.json"
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
    required = {
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
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("preregistration lock schema drifted")
    if value["schema_version"] != 1 or value["status"] != LOCK_STATUS:
        raise ValueError("preregistration lock is not active")
    if value["branch"] != frozen_config["branch"] or value["parent_commit_sha"] != frozen_config["start"]["parent_commit_sha"]:
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
    if value["pre_freeze_result_inventory"] != {
        "feature_files": 0,
        "prediction_files": 0,
        "metric_files": 0,
        "figure_files": 0,
        "report_files": 0,
        "log_files": 0,
        "manifest_files": 0,
    }:
        raise ValueError("preregistration lock did not precede result generation")
    privacy = value["privacy_contract"]
    if privacy != {
        "private_patient_artifacts_owner_only": True,
        "raw_spatial_maps_persisted": False,
        "region_definition_reads_masks_labels_ftv_or_clinical": False,
    }:
        raise ValueError("preregistration privacy contract drifted")
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
    "EXPERIMENT_ROOT",
    "FEATURE_FILENAME",
    "FEATURE_KEYS",
    "FOLDS",
    "GEOMETRY_CONTRACT_PATH",
    "GITIGNORE_PATH",
    "LOCK_PATH",
    "LOCK_STATUS",
    "METADATA_FILENAME",
    "METADATA_KEYS",
    "PATIENT_COUNT",
    "PLAN_PATH",
    "REPO_ROOT",
    "SEEDS",
    "VARIANT_DIMENSIONS",
    "VARIANT_KEYS",
    "VISITS",
    "array_sha256",
    "atomic_json",
    "atomic_npz",
    "canonical_sha256",
    "cell_key",
    "cells",
    "feature_path",
    "file_sha256",
    "implementation_inventory",
    "load_config",
    "load_goal5_lock",
    "metadata_path",
    "ordered_sha256",
    "private_directory",
    "publish_json_once",
    "require_owner_only",
    "require_preregistration_lock",
    "validate_feature_cell",
]

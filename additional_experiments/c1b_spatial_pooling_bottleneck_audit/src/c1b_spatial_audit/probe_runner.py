"""Fail-closed orchestration for the frozen spatial probe matrix.

The runner validates the complete requested exporter matrix before any Ridge
fit is started.  It owns only matrix discovery, immutable-input binding,
private publication, and optional process-level parallelism; all statistical
work remains in :mod:`c1b_spatial_audit.probes`.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .contracts import (
    ARMS,
    EXPERIMENT_ROOT,
    FOLDS,
    POOLINGS,
    REPO_ROOT,
    SEEDS,
    TIMEPOINTS,
    UPSTREAM_ROOT,
    canonical_sha256,
    cell_key,
    cells,
    file_sha256,
)
from .probes import (
    ProbeResult,
    combine_probe_results,
    load_frozen_state_asset,
    run_continuous_probe_cell,
    run_ftv_probe_cell,
    write_probe_outputs,
)
from .runtime import load_stage_b_bundle, verify_preregistration
from .sidecars import NUISANCE_COLUMNS


STAGES = ("final", "s3")
SECONDARY_POOLING = "PLOCAL+PVALID_SECONDARY"
FORMAL_POOLINGS = (*POOLINGS, SECONDARY_POOLING)
NUISANCE_TARGETS = tuple(NUISANCE_COLUMNS[2:])
NUISANCE_POOLINGS = frozenset({"P0", "PVALID", "PLOCAL"})
N_ONLY_POOLINGS = frozenset({"PVALID", "PORACLE", SECONDARY_POOLING})
POOLING_SLUGS = {
    "P0": "p0",
    "PVALID": "pvalid",
    "PLOCAL": "plocal",
    "PLOCAL+GLOBAL": "plocal_global",
    "PORACLE": "poracle",
    SECONDARY_POOLING: "plocal_pvalid_secondary",
}
STAGE_POOLINGS = {
    "final": FORMAL_POOLINGS,
    "s3": ("P0", "PLOCAL", "PORACLE"),
}
S3_REPRESENTATION_CONTRACT = "raw_encoder_features2_pooled_64d_no_projection"

EXPORTER_METADATA_FIELDS = frozenset(
    {
        "schema_version",
        "stage",
        "status",
        "arm",
        "seed_base",
        "fold",
        "pooling",
        "pooling_slug",
        "feature_path",
        "feature_sha256",
        "state_shape",
        "state_dtype",
        "state_valid_shape",
        "state_valid_count",
        "patient_count",
        "patient_order_sha256",
        "split_order_sha256",
        "checkpoint_path",
        "checkpoint_sha256",
        "checkpoint_lock_key",
        "reference_feature_path",
        "reference_feature_sha256",
        "reference_feature_metadata_path",
        "reference_feature_metadata_sha256",
        "preregistration_lock_sha256",
        "plan_sha256",
        "config_sha256",
        "sidecar_path",
        "sidecar_sha256",
        "sidecar_keys_used",
        "data_contract_provenance_sha256",
        "checkpoint_data_provenance_sha256",
        "stage_a_sentinel_sha256",
        "implementation_sha256",
        "device",
        "batch_size",
        "workers",
        "feature_tensor",
        "response_projection",
        "training_performed",
        "projector_called",
        "transition_called",
        "target_encoder_called",
        "ftv_head_called",
        "test_labels_used",
    }
)


@dataclass(frozen=True, order=True)
class ProbeCellKey:
    seed_base: int
    arm: str
    fold: int
    pooling: str

    @property
    def checkpoint_key(self) -> str:
        return cell_key(self.seed_base, self.arm, self.fold)

    @property
    def inventory_key(self) -> str:
        return f"{self.checkpoint_key}/{POOLING_SLUGS[self.pooling]}"


@dataclass(frozen=True)
class LockedCellBinding:
    checkpoint_path: Path
    checkpoint_sha256: str
    reference_feature_path: Path
    reference_feature_sha256: str
    reference_metadata_path: Path
    reference_metadata_sha256: str
    patient_order_sha256: str


@dataclass(frozen=True)
class ProbeCellSpec:
    key: ProbeCellKey
    stage: str
    feature_path: Path
    feature_metadata_path: Path
    feature_sha256: str
    feature_metadata_sha256: str
    private_root: Path
    output_dir: Path
    include_nuisance: bool
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class ProbeMatrixPlan:
    stage: str
    poolings: tuple[str, ...]
    cells: tuple[ProbeCellSpec, ...]
    feature_root: Path
    probe_root: Path
    preregistration_path: Path
    preregistration_sha256: str
    nuisance_path: Path | None
    nuisance_sha256: str | None
    gate_sha256: Mapping[str, str]
    exporter_completion_sha256: str

    @property
    def completion_path(self) -> Path:
        label = "_".join(POOLING_SLUGS[value] for value in self.poolings)
        return self.probe_root / self.stage / f"probe_matrix_{label}_complete.private.json"


def _read_json(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {source}")
    return payload


def _require_sha256(value: Any, label: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _resolve_path(value: Any, *, relative_to: Path = REPO_ROOT) -> Path:
    candidate = Path(str(value)).expanduser()
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else (relative_to / candidate).resolve()
    )


def _metadata_path_matches(value: Any, expected: Path, metadata_path: Path) -> bool:
    candidate = Path(str(value)).expanduser()
    expected = expected.resolve()
    if candidate.is_absolute():
        return candidate.resolve() == expected
    return expected in {
        (REPO_ROOT / candidate).resolve(),
        (metadata_path.parent / candidate).resolve(),
    }


def _cached_file_sha256(path: Path, cache: dict[Path, str]) -> str:
    source = path.resolve()
    if source not in cache:
        cache[source] = file_sha256(source)
    return cache[source]


def split_order_sha256(splits: Iterable[str]) -> str:
    """Hash ordered split labels with the exporter's no-trailing-newline rule."""

    return hashlib.sha256(
        "\n".join(str(split).lower() for split in splits).encode("utf-8")
    ).hexdigest()


def parse_poolings(value: str | Iterable[str]) -> tuple[str, ...]:
    raw = value.split(",") if isinstance(value, str) else tuple(value)
    aliases = {slug.upper(): pooling for pooling, slug in POOLING_SLUGS.items()}
    normalized = tuple(
        aliases.get(str(item).strip().upper(), str(item).strip().upper())
        for item in raw
    )
    if not normalized or any(not item for item in normalized):
        raise ValueError("--poolings must contain one or more comma-separated values")
    if len(set(normalized)) != len(normalized):
        raise ValueError("--poolings contains a duplicate pooling")
    unknown = sorted(set(normalized).difference(FORMAL_POOLINGS))
    if unknown:
        raise ValueError(f"unknown pooling values: {unknown}")
    return tuple(pooling for pooling in FORMAL_POOLINGS if pooling in normalized)


def validate_stage_poolings(stage: str, poolings: Sequence[str]) -> tuple[str, tuple[str, ...]]:
    normalized_stage = str(stage).strip().lower()
    if normalized_stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    normalized_poolings = parse_poolings(poolings)
    forbidden = sorted(set(normalized_poolings).difference(STAGE_POOLINGS[normalized_stage]))
    if forbidden:
        raise ValueError(
            f"stage {normalized_stage} does not preregister poolings {forbidden}"
        )
    return normalized_stage, normalized_poolings


def pooling_available(stage: str, arm: str, pooling: str) -> bool:
    stage, selected = validate_stage_poolings(stage, (pooling,))
    pooling = selected[0]
    arm = str(arm).upper()
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if pooling in N_ONLY_POOLINGS and not arm.startswith("N"):
        return False
    return pooling in STAGE_POOLINGS[stage]


def expected_feature_keys(stage: str, poolings: Sequence[str]) -> tuple[ProbeCellKey, ...]:
    stage, poolings = validate_stage_poolings(stage, poolings)
    return tuple(
        ProbeCellKey(seed, arm, fold, pooling)
        for seed, arm, fold in cells()
        for pooling in poolings
        if pooling_available(stage, arm, pooling)
    )


def expected_feature_dimension(stage: str, pooling: str) -> int:
    stage, selected = validate_stage_poolings(stage, (pooling,))
    pooling = selected[0]
    if stage == "final":
        return (
            384
            if pooling in {"PLOCAL+GLOBAL", SECONDARY_POOLING}
            else 192
        )
    # The preregistered S3 tap is pre-projector encoder.features[2] (64 channels).
    return 64


def expected_sidecar_keys(arm: str, pooling: str) -> tuple[str, ...]:
    arm = str(arm).upper()
    pooling = str(pooling).upper()
    if pooling == "P0":
        return ()
    if pooling == "PVALID":
        return ("c1b_valid_weight_final",)
    if pooling in {"PLOCAL", "PLOCAL+GLOBAL"}:
        return (
            ("legacy_local_weight_final",)
            if arm.startswith("L")
            else ("c1b_local_weight_final",)
        )
    if pooling == "PORACLE":
        return ("c1b_oracle_weight_final", "c1b_oracle_valid")
    if pooling == SECONDARY_POOLING:
        return ("c1b_local_weight_final", "c1b_valid_weight_final")
    raise ValueError(f"no sidecar contract for {arm}/{pooling}")


def feature_path_for(
    feature_root: str | Path, stage: str, key: ProbeCellKey
) -> Path:
    return (
        Path(feature_root).resolve()
        / stage
        / f"seed_{key.seed_base}"
        / key.arm
        / f"fold_{key.fold}"
        / f"{POOLING_SLUGS[key.pooling]}.private.npz"
    )


def output_path_for(probe_root: str | Path, stage: str, key: ProbeCellKey) -> Path:
    return (
        Path(probe_root).resolve()
        / stage
        / f"seed_{key.seed_base}"
        / key.arm
        / f"fold_{key.fold}"
        / POOLING_SLUGS[key.pooling]
    )


def validate_locked_bindings(
    lock: Mapping[str, Any],
) -> dict[tuple[int, str, int], LockedCellBinding]:
    """Validate and hash the 40 immutable checkpoint/reference bindings once."""

    expected_names = {cell_key(seed, arm, fold) for seed, arm, fold in cells()}
    if int(lock.get("formal_cell_count", -1)) != 40:
        raise ValueError("preregistration does not declare the exact 40 checkpoint cells")
    selected = lock.get("selected_checkpoints")
    references = lock.get("formal_p0_references")
    if not isinstance(selected, Mapping) or set(selected) != expected_names:
        raise ValueError("selected-checkpoint lock inventory is not the exact 40-cell matrix")
    if not isinstance(references, Mapping) or set(references) != expected_names:
        raise ValueError("P0-reference lock inventory is not the exact 40-cell matrix")

    bindings: dict[tuple[int, str, int], LockedCellBinding] = {}
    for seed, arm, fold in cells():
        name = cell_key(seed, arm, fold)
        checkpoint_record = selected[name]
        reference_record = references[name]
        if not isinstance(checkpoint_record, Mapping) or not isinstance(
            reference_record, Mapping
        ):
            raise ValueError(f"malformed preregistration binding at {name}")
        checkpoint = _resolve_path(checkpoint_record.get("path", ""))
        checkpoint_digest = _require_sha256(
            checkpoint_record.get("sha256"), f"{name} checkpoint SHA-256"
        )
        if not checkpoint.is_file() or file_sha256(checkpoint) != checkpoint_digest:
            raise ValueError(f"selected checkpoint path/SHA-256 drifted at {name}")
        if "size_bytes" in checkpoint_record and int(
            checkpoint_record["size_bytes"]
        ) != checkpoint.stat().st_size:
            raise ValueError(f"selected checkpoint size drifted at {name}")
        if "mtime_ns" in checkpoint_record and int(checkpoint_record["mtime_ns"]) != (
            checkpoint.stat().st_mtime_ns
        ):
            raise ValueError(f"selected checkpoint mtime drifted at {name}")

        reference = _resolve_path(reference_record.get("feature_path", ""))
        reference_digest = _require_sha256(
            reference_record.get("feature_sha256"), f"{name} reference SHA-256"
        )
        reference_metadata = _resolve_path(
            reference_record.get("feature_metadata_path", "")
        )
        reference_metadata_digest = _require_sha256(
            reference_record.get("feature_metadata_sha256"),
            f"{name} reference metadata SHA-256",
        )
        if not reference.is_file() or file_sha256(reference) != reference_digest:
            raise ValueError(f"immutable P0 reference path/SHA-256 drifted at {name}")
        if not reference_metadata.is_file() or file_sha256(
            reference_metadata
        ) != reference_metadata_digest:
            raise ValueError(f"immutable P0 reference metadata drifted at {name}")
        patient_digest = _require_sha256(
            reference_record.get("patient_order_sha256"),
            f"{name} reference patient-order SHA-256",
        )
        bindings[(seed, arm, fold)] = LockedCellBinding(
            checkpoint,
            checkpoint_digest,
            reference,
            reference_digest,
            reference_metadata,
            reference_metadata_digest,
            patient_digest,
        )
    return bindings


def validate_exporter_completion(
    feature_root: str | Path,
    *,
    stage: str,
    preregistration_sha256: str,
) -> str:
    """Bind formal probes to the exporter's complete, nonresumable matrix."""

    feature_root = Path(feature_root).resolve()
    stage = str(stage).lower()
    if stage != "final":
        raise ValueError("S3 needs its own trigger-bound exporter completion contract")
    completion_path = feature_root / "feature_export_complete.private.json"
    preflight_path = feature_root / "feature_export_preflight.private.json"
    completion = _read_json(completion_path, label="feature export completion")
    expected_fields = {
        "schema_version",
        "status",
        "stage",
        "run_count",
        "expected_asset_count",
        "cell_count",
        "feature_metadata_sha256",
        "preflight_sha256",
        "sidecar_sha256",
        "preregistration_lock_sha256",
    }
    if set(completion) != expected_fields:
        raise ValueError("feature export completion schema drifted")
    if (
        int(completion.get("schema_version", -1)) != 1
        or completion.get("status") != "COMPLETE"
        or completion.get("stage") != "final"
        or int(completion.get("run_count", -1)) != 40
        or int(completion.get("cell_count", -1)) != 40
        or int(completion.get("expected_asset_count", -1)) != 180
        or completion.get("preregistration_lock_sha256") != preregistration_sha256
    ):
        raise ValueError("feature export completion identity/count binding drifted")
    if not preflight_path.is_file() or file_sha256(preflight_path) != completion.get(
        "preflight_sha256"
    ):
        raise ValueError("feature export preflight binding drifted")
    preflight = _read_json(preflight_path, label="feature export preflight")
    if (
        preflight.get("status") != "PREFLIGHT_PASS"
        or preflight.get("stage") != "final"
        or int(preflight.get("cell_count", -1)) != 40
        or int(preflight.get("expected_asset_count", -1)) != 180
        or preflight.get("preregistration_lock_sha256") != preregistration_sha256
        or preflight.get("sidecar_sha256") != completion.get("sidecar_sha256")
    ):
        raise ValueError("feature export preflight contract drifted")
    inventory = completion.get("feature_metadata_sha256")
    if not isinstance(inventory, Mapping) or len(inventory) != 180:
        raise ValueError("feature export metadata inventory is not exactly 180 assets")
    expected_metadata = set((feature_root / "final").rglob("*.private.metadata.json"))
    resolved_inventory: dict[Path, str] = {}
    for raw_path, raw_digest in inventory.items():
        path = _resolve_path(raw_path)
        if not path.is_relative_to(feature_root / "final"):
            raise ValueError("feature export inventory path escaped final stage root")
        digest = _require_sha256(raw_digest, f"exporter metadata inventory {raw_path}")
        if path in resolved_inventory:
            raise ValueError("feature export inventory duplicates a metadata path")
        resolved_inventory[path] = digest
    if set(resolved_inventory) != {path.resolve() for path in expected_metadata}:
        raise ValueError("feature export completion does not inventory exact live metadata")
    for path, digest in resolved_inventory.items():
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"feature export metadata inventory drifted: {path}")
        if path.stat().st_mode & 0o077:
            raise PermissionError("feature export metadata inventory must be owner-only")
    for private_path in (completion_path, preflight_path):
        if private_path.stat().st_mode & 0o077:
            raise PermissionError("feature export control artifacts must be owner-only")
    return file_sha256(completion_path)


def _validate_fold_split(asset: Any, folds: pd.DataFrame) -> None:
    required = {"patient_id", "fold", "split"}
    if missing := sorted(required.difference(folds.columns)):
        raise ValueError(f"frozen fold manifest misses columns: {missing}")
    current = folds.loc[folds["fold"].astype(int).eq(asset.fold), list(required)].copy()
    current["patient_id"] = current["patient_id"].astype(str)
    current["split"] = current["split"].astype(str).str.lower()
    if current.empty or current["patient_id"].duplicated().any():
        raise ValueError(f"frozen fold {asset.fold} is empty or duplicates patients")
    expected = dict(zip(current["patient_id"], current["split"], strict=True))
    observed = dict(zip(asset.patient_id, asset.split, strict=True))
    if observed != expected:
        raise ValueError(
            "exported patient identities/splits disagree with the frozen fold manifest"
        )


def _validate_exporter_metadata(
    *,
    asset: Any,
    expected_key: ProbeCellKey,
    stage: str,
    metadata_path: Path,
    binding: LockedCellBinding,
    preregistration_path: Path,
    preregistration_sha256: str,
    lock: Mapping[str, Any],
    live_hashes: dict[Path, str],
) -> None:
    metadata = dict(asset.source_metadata)
    expected_fields = set(EXPORTER_METADATA_FIELDS)
    if stage == "s3":
        expected_fields.add("representation_contract")
    if set(metadata) != expected_fields:
        raise ValueError(
            "exporter metadata schema drifted: "
            f"missing={sorted(expected_fields.difference(metadata))}, "
            f"extra={sorted(set(metadata).difference(expected_fields))}"
        )
    identity = {
        "stage": stage,
        "status": "COMPLETE",
        "arm": expected_key.arm,
        "seed_base": expected_key.seed_base,
        "fold": expected_key.fold,
        "pooling": expected_key.pooling,
        "pooling_slug": POOLING_SLUGS[expected_key.pooling],
    }
    if int(metadata.get("schema_version", -1)) != 1:
        raise ValueError("exporter metadata must be schema version 1")
    for field, expected in identity.items():
        if metadata.get(field) != expected:
            raise ValueError(f"exporter metadata identity drifted at {field}")

    if metadata.get("state_shape") != list(asset.state.shape):
        raise ValueError("exporter metadata state_shape drifted")
    if metadata.get("state_dtype") != "float32":
        raise ValueError("exporter metadata state_dtype is not float32")
    if metadata.get("state_valid_shape") != list(asset.state_valid.shape):
        raise ValueError("exporter metadata state_valid_shape drifted")
    if int(metadata.get("state_valid_count", -1)) != int(asset.state_valid.sum()):
        raise ValueError("exporter metadata state_valid_count drifted")
    if int(metadata.get("patient_count", -1)) != len(asset.patient_id):
        raise ValueError("exporter metadata patient_count drifted")
    if metadata.get("patient_order_sha256") != asset.patient_order_sha256:
        raise ValueError("exporter metadata patient-order SHA-256 drifted")
    if metadata.get("split_order_sha256") != split_order_sha256(asset.split):
        raise ValueError("exporter metadata split-order SHA-256 drifted")
    if asset.patient_order_sha256 != binding.patient_order_sha256:
        raise ValueError("exported patient order differs from the locked P0 reference")
    expected_dimension = expected_feature_dimension(stage, expected_key.pooling)
    if asset.feature_dim != expected_dimension:
        raise ValueError(
            f"{stage}/{expected_key.pooling} feature dimension must be {expected_dimension}"
        )

    path_bindings = {
        "checkpoint_path": binding.checkpoint_path,
        "reference_feature_path": binding.reference_feature_path,
        "reference_feature_metadata_path": binding.reference_metadata_path,
        "feature_path": asset.source_path,
    }
    for field, expected in path_bindings.items():
        if expected is None or not _metadata_path_matches(
            metadata.get(field, ""), expected, metadata_path
        ):
            raise ValueError(f"exporter metadata path drifted at {field}")
    hash_bindings = {
        "checkpoint_sha256": binding.checkpoint_sha256,
        "reference_feature_sha256": binding.reference_feature_sha256,
        "reference_feature_metadata_sha256": binding.reference_metadata_sha256,
        "feature_sha256": file_sha256(asset.source_path),
        "preregistration_lock_sha256": preregistration_sha256,
        "plan_sha256": lock.get("plan_sha256"),
        "config_sha256": lock.get("config_sha256"),
    }
    for field, expected in hash_bindings.items():
        expected_digest = _require_sha256(expected, f"locked {field}")
        if metadata.get(field) != expected_digest:
            raise ValueError(f"exporter metadata SHA-256 drifted at {field}")
    if metadata.get("checkpoint_lock_key") != expected_key.checkpoint_key:
        raise ValueError("exporter checkpoint_lock_key drifted")
    if _cached_file_sha256(preregistration_path, live_hashes) != preregistration_sha256:
        raise ValueError("preregistration lock changed during feature validation")

    for field in (
        "sidecar_sha256",
        "data_contract_provenance_sha256",
        "checkpoint_data_provenance_sha256",
        "stage_a_sentinel_sha256",
    ):
        _require_sha256(metadata.get(field), f"exporter {field}")
    implementation = metadata.get("implementation_sha256")
    expected_implementation = {
        "exporter.py": Path(__file__).with_name("exporter.py"),
        "pooling.py": Path(__file__).with_name("pooling.py"),
        "runtime.py": Path(__file__).with_name("runtime.py"),
        "contracts.py": Path(__file__).with_name("contracts.py"),
    }
    if not isinstance(implementation, Mapping) or set(implementation) != set(
        expected_implementation
    ):
        raise ValueError("exporter implementation SHA-256 map drifted")
    for name, source in expected_implementation.items():
        digest = _require_sha256(implementation[name], f"exporter implementation {name}")
        if not source.is_file() or _cached_file_sha256(source, live_hashes) != digest:
            raise ValueError(f"exporter implementation source drifted at {name}")
    sidecar = _resolve_path(metadata.get("sidecar_path", ""))
    if not sidecar.is_file() or _cached_file_sha256(
        sidecar, live_hashes
    ) != metadata["sidecar_sha256"]:
        raise ValueError("exporter sidecar path/SHA-256 drifted")
    sentinel = UPSTREAM_ROOT / "STAGE_A_GO.json"
    if not sentinel.is_file() or _cached_file_sha256(
        sentinel, live_hashes
    ) != metadata["stage_a_sentinel_sha256"]:
        raise ValueError("exporter Stage-A sentinel binding drifted")
    if not isinstance(metadata.get("sidecar_keys_used"), list) or sorted(
        metadata["sidecar_keys_used"]
    ) != sorted(expected_sidecar_keys(expected_key.arm, expected_key.pooling)):
        raise ValueError("exporter sidecar_keys_used contract drifted")
    if not isinstance(metadata.get("device"), str) or not metadata["device"]:
        raise ValueError("exporter device metadata is empty")
    if int(metadata.get("batch_size", 0)) <= 0 or int(metadata.get("workers", -1)) < 0:
        raise ValueError("exporter batch/worker metadata is invalid")
    if not isinstance(metadata.get("feature_tensor"), str) or not metadata["feature_tensor"]:
        raise ValueError("exporter feature_tensor metadata is empty")
    if not isinstance(metadata.get("response_projection"), str) or not metadata[
        "response_projection"
    ]:
        raise ValueError("exporter response_projection metadata is empty")
    forbidden_true = (
        "training_performed",
        "transition_called",
        "target_encoder_called",
        "ftv_head_called",
        "test_labels_used",
    )
    if any(metadata.get(field) is not False for field in forbidden_true):
        raise ValueError("exporter metadata reports forbidden training/head/label use")
    if not isinstance(metadata.get("projector_called"), bool):
        raise ValueError("exporter projector_called must be boolean")
    if stage == "final" and metadata["projector_called"] is not False:
        raise ValueError("final export must not call the model projector path")
    if stage == "final" and (
        metadata["feature_tensor"] != "full_model.encoder_output_before_gap"
        or metadata["response_projection"]
        != "frozen_online_Linear128x192_plus_LayerNorm"
        or not str(metadata["device"]).startswith("cuda")
        or int(metadata["batch_size"]) != 4
        or int(metadata["workers"]) != 2
    ):
        raise ValueError("final exporter tensor/projection/runtime contract drifted")
    if stage == "s3" and (
        metadata["projector_called"] is not False
        or metadata.get("representation_contract") != S3_REPRESENTATION_CONTRACT
    ):
        raise ValueError("S3 must be raw pooled 64-D features with no projection")


def discover_feature_matrix(
    *,
    feature_root: str | Path,
    probe_root: str | Path,
    stage: str,
    poolings: Sequence[str],
    folds: pd.DataFrame,
    lock: Mapping[str, Any],
    preregistration_path: str | Path,
    common_provenance: Mapping[str, Any] | None = None,
) -> tuple[ProbeCellSpec, ...]:
    """Discover and fully validate the exact requested exporter matrix."""

    stage, poolings = validate_stage_poolings(stage, poolings)
    feature_root = Path(feature_root).resolve()
    probe_root = Path(probe_root).resolve()
    stage_root = feature_root / stage
    if not stage_root.is_dir():
        raise FileNotFoundError(f"exporter stage directory is missing: {stage_root}")
    preregistration_path = Path(preregistration_path).resolve()
    preregistration_sha256 = file_sha256(preregistration_path)
    bindings = validate_locked_bindings(lock)
    expected_keys = set(expected_feature_keys(stage, poolings))
    discovered: dict[ProbeCellKey, Path] = {}

    for path in sorted(stage_root.rglob("*.private.npz")):
        try:
            with np.load(path, allow_pickle=False) as archive:
                identity_fields = {"seed_base", "arm", "fold", "pooling"}
                if not identity_fields.issubset(archive.files):
                    raise ValueError("missing exporter identity keys")
                key = ProbeCellKey(
                    int(np.asarray(archive["seed_base"]).item()),
                    str(np.asarray(archive["arm"]).item()).upper(),
                    int(np.asarray(archive["fold"]).item()),
                    str(np.asarray(archive["pooling"]).item()).upper(),
                )
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError(f"cannot inspect exporter feature asset: {path}") from exc
        if key.pooling not in POOLING_SLUGS:
            continue
        if key.seed_base not in SEEDS or key.arm not in ARMS or key.fold not in FOLDS:
            if key.pooling in poolings:
                raise ValueError(f"requested pooling has an out-of-matrix feature: {key}")
            continue
        if not pooling_available(stage, key.arm, key.pooling):
            raise ValueError(f"fabricated unavailable legacy/stage feature detected: {key}")
        if key.pooling not in poolings:
            continue
        expected_path = feature_path_for(feature_root, stage, key)
        if path.resolve() != expected_path:
            raise ValueError(
                f"exporter feature path layout drifted for {key}: {path.resolve()}"
            )
        if key in discovered:
            raise ValueError(f"duplicate exporter feature for {key}")
        discovered[key] = path.resolve()

    if set(discovered) != expected_keys:
        missing = sorted(expected_keys.difference(discovered))
        extra = sorted(set(discovered).difference(expected_keys))
        raise ValueError(
            f"requested exporter matrix is incomplete: missing={missing}, extra={extra}"
        )

    common = {} if common_provenance is None else dict(common_provenance)
    live_hashes = {preregistration_path: preregistration_sha256}
    specifications: list[ProbeCellSpec] = []
    for key in sorted(expected_keys):
        path = discovered[key]
        asset = load_frozen_state_asset(path)
        if path.stat().st_mode & 0o077 or (
            asset.metadata_path is not None and asset.metadata_path.stat().st_mode & 0o077
        ):
            raise PermissionError("exporter feature assets and metadata must be owner-only")
        if (
            asset.seed_base,
            asset.arm,
            asset.fold,
            asset.pooling,
        ) != (key.seed_base, key.arm, key.fold, key.pooling):
            raise ValueError(f"loaded exporter identity drifted for {key}")
        _validate_fold_split(asset, folds)
        binding = bindings[(key.seed_base, key.arm, key.fold)]
        if asset.metadata_path is None:
            raise ValueError(f"exporter metadata is absent for {key}")
        _validate_exporter_metadata(
            asset=asset,
            expected_key=key,
            stage=stage,
            metadata_path=asset.metadata_path,
            binding=binding,
            preregistration_path=preregistration_path,
            preregistration_sha256=preregistration_sha256,
            lock=lock,
            live_hashes=live_hashes,
        )
        feature_digest = file_sha256(path)
        metadata_digest = file_sha256(asset.metadata_path)
        provenance = {
            **common,
            "stage": stage,
            "checkpoint_lock_key": key.checkpoint_key,
            "checkpoint_sha256": binding.checkpoint_sha256,
            "reference_feature_sha256": binding.reference_feature_sha256,
            "reference_feature_metadata_sha256": binding.reference_metadata_sha256,
            "preregistration_lock_sha256": preregistration_sha256,
            "plan_sha256": lock["plan_sha256"],
            "config_sha256": lock["config_sha256"],
            "exporter_feature_sha256": feature_digest,
            "exporter_metadata_sha256": metadata_digest,
        }
        specifications.append(
            ProbeCellSpec(
                key=key,
                stage=stage,
                feature_path=path,
                feature_metadata_path=asset.metadata_path,
                feature_sha256=feature_digest,
                feature_metadata_sha256=metadata_digest,
                private_root=probe_root,
                output_dir=output_path_for(probe_root, stage, key),
                include_nuisance=key.pooling in NUISANCE_POOLINGS,
                provenance=provenance,
            )
        )
    return tuple(specifications)


def load_nuisance_targets(
    path: str | Path,
    patient_ids: Iterable[str],
) -> dict[str, dict[str, np.ndarray]]:
    """Load exactly the ten preregistered visit-level geometry targets."""

    source = Path(path).resolve()
    if not source.name.endswith(".private.csv"):
        raise ValueError("nuisance targets must be an identifier-bearing private CSV")
    if source.stat().st_mode & 0o077:
        raise PermissionError("nuisance target CSV must be owner-only")
    frame = pd.read_csv(source, dtype={"patient_id": str, "visit": str})
    if tuple(frame.columns) != NUISANCE_COLUMNS:
        raise ValueError(
            "nuisance CSV schema drifted: "
            f"expected={list(NUISANCE_COLUMNS)}, observed={list(frame.columns)}"
        )
    if tuple(frame.columns[2:]) != NUISANCE_TARGETS or len(NUISANCE_TARGETS) != 10:
        raise AssertionError("nuisance target allow-list is not the preregistered ten")
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["visit"] = frame["visit"].astype(str)
    if frame.duplicated(["patient_id", "visit"]).any():
        raise ValueError("nuisance CSV duplicates a patient/visit row")
    patients = tuple(str(value) for value in patient_ids)
    if len(set(patients)) != len(patients) or not patients:
        raise ValueError("frozen nuisance patient population is empty or duplicated")
    expected_rows = {(patient, visit) for patient in patients for visit in TIMEPOINTS}
    observed_rows = set(zip(frame["patient_id"], frame["visit"], strict=True))
    if observed_rows != expected_rows:
        raise ValueError("nuisance CSV does not exactly cover the frozen four-visit population")
    for target in NUISANCE_TARGETS:
        frame[target] = pd.to_numeric(frame[target], errors="raise")
    values = frame[list(NUISANCE_TARGETS)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise FloatingPointError("nuisance CSV contains a non-finite target")
    indexed = frame.set_index(["patient_id", "visit"], verify_integrity=True)
    output: dict[str, dict[str, np.ndarray]] = {}
    for target in NUISANCE_TARGETS:
        output[target] = {
            patient: np.asarray(
                [indexed.loc[(patient, visit), target] for visit in TIMEPOINTS],
                dtype=np.float64,
            )
            for patient in patients
        }
    return output


def validate_alternative_gates(
    poolings: Sequence[str],
    *,
    preregistration_sha256: str,
    equivalence_gate_path: str | Path,
    probe_replication_gate_path: str | Path,
) -> dict[str, str]:
    """Require state parity for every probe and probe parity for alternatives."""

    equivalence_path = Path(equivalence_gate_path).resolve()
    equivalence = _read_json(equivalence_path, label="P0 equivalence gate")
    if (
        equivalence.get("status") != "PASS"
        or equivalence.get("probe_execution_authorized") is not True
        or int(equivalence.get("formal_cells", -1)) != 40
        or equivalence.get("preregistration_lock_sha256")
        != preregistration_sha256
    ):
        raise PermissionError("P0 state-equivalence STOP gate does not authorize probes")
    hashes = {"p0_equivalence_gate_sha256": file_sha256(equivalence_path)}
    if not any(pooling != "P0" for pooling in poolings):
        return hashes
    replication_path = Path(probe_replication_gate_path).resolve()
    replication = _read_json(replication_path, label="P0 probe replication gate")
    if (
        replication.get("status") != "PASS"
        or replication.get("alternate_pooling_interpretation_authorized") is not True
        or int(replication.get("formal_cells", -1)) != 40
    ):
        raise PermissionError("P0 probe-replication STOP gate does not authorize alternatives")
    hashes["p0_probe_replication_gate_sha256"] = file_sha256(replication_path)
    return hashes


def _ensure_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)


def _ensure_private_tree(root: Path, directory: Path) -> None:
    root = root.resolve()
    directory = directory.resolve()
    if directory != root and not directory.is_relative_to(root):
        raise ValueError("private output directory escaped its declared root")
    _ensure_private_directory(root)
    relative = directory.relative_to(root)
    current = root
    for part in relative.parts:
        current = current / part
        _ensure_private_directory(current)


def _publish_cell(
    result: ProbeResult,
    specification: ProbeCellSpec,
) -> dict[str, Any]:
    target = specification.output_dir.resolve()
    _ensure_private_tree(specification.private_root, target.parent)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite frozen probe cell: {target}")
    lock_path = target.parent / f".{target.name}.publish.lock"
    descriptor: int | None = None
    lock_acquired = False
    staging: Path | None = None
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        lock_acquired = True
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{target.name}.staging.", dir=target.parent)
        )
        os.chmod(staging, 0o700)
        metadata = write_probe_outputs(
            result,
            staging,
            provenance=specification.provenance,
            feature_path=specification.feature_path,
            feature_metadata_path=specification.feature_metadata_path,
        )
        if target.exists():
            raise FileExistsError(f"probe cell appeared during publication: {target}")
        os.rename(staging, target)
        staging = None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if staging is not None:
            shutil.rmtree(staging)
        if lock_acquired:
            lock_path.unlink(missing_ok=True)
    metadata_path = target / "probe_metadata.json"
    return {
        "seed_base": specification.key.seed_base,
        "arm": specification.key.arm,
        "fold": specification.key.fold,
        "pooling": specification.key.pooling,
        "pooling_slug": POOLING_SLUGS[specification.key.pooling],
        "feature_path": str(specification.feature_path),
        "feature_sha256": specification.feature_sha256,
        "feature_metadata_path": str(specification.feature_metadata_path),
        "feature_metadata_sha256": specification.feature_metadata_sha256,
        "output_dir": str(target),
        "probe_metadata_sha256": file_sha256(metadata_path),
        "output_sha256": metadata["output_sha256"],
        "selection_rows": int(metadata["selection_rows"]),
        "prediction_rows": int(metadata["prediction_rows"]),
        "metric_rows": int(metadata["metric_rows"]),
        "nuisance_included": specification.include_nuisance,
    }


def execute_probe_cell(
    specification: ProbeCellSpec,
    records: Mapping[str, Any],
    nuisance_targets: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Execute and atomically publish one already-validated matrix cell."""

    if file_sha256(specification.feature_path) != specification.feature_sha256:
        raise ValueError("feature asset changed after matrix preflight")
    if file_sha256(
        specification.feature_metadata_path
    ) != specification.feature_metadata_sha256:
        raise ValueError("feature metadata changed after matrix preflight")
    asset = load_frozen_state_asset(specification.feature_path)
    ftv = run_ftv_probe_cell(asset, records)
    result = ftv
    if specification.include_nuisance:
        if nuisance_targets is None or set(nuisance_targets) != set(NUISANCE_TARGETS):
            raise ValueError("the exact ten nuisance targets are required for this pooling")
        nuisance = run_continuous_probe_cell(
            asset,
            nuisance_targets,
            task="nuisance",
            analysis_scope="target_valid",
            target_semantics={
                target: f"preregistered_geometry_nuisance::{target}"
                for target in NUISANCE_TARGETS
            },
        )
        result = combine_probe_results(ftv, nuisance)
    return _publish_cell(result, specification)


_WORKER_RECORDS: Mapping[str, Any] | None = None
_WORKER_NUISANCE: Mapping[str, Mapping[str, Any]] | None = None


def _initialize_worker(
    records: Mapping[str, Any],
    nuisance_targets: Mapping[str, Mapping[str, Any]] | None,
) -> None:
    global _WORKER_RECORDS, _WORKER_NUISANCE
    _WORKER_RECORDS = records
    _WORKER_NUISANCE = nuisance_targets


def _worker_execute(specification: ProbeCellSpec) -> dict[str, Any]:
    if _WORKER_RECORDS is None:
        raise RuntimeError("probe worker was not initialized")
    return execute_probe_cell(specification, _WORKER_RECORDS, _WORKER_NUISANCE)


def _private_json_no_overwrite(path: Path, payload: Mapping[str, Any]) -> None:
    _ensure_private_directory(path.parent)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_path, 0o600)
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def plan_summary(plan: ProbeMatrixPlan, *, status: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": status,
        "stage": plan.stage,
        "representation_contract": (
            S3_REPRESENTATION_CONTRACT
            if plan.stage == "s3"
            else "final_pooled_then_frozen_response_projection"
        ),
        "requested_poolings": list(plan.poolings),
        "expected_cell_count": len(plan.cells),
        "nuisance_cell_count": sum(cell.include_nuisance for cell in plan.cells),
        "nuisance_targets": list(NUISANCE_TARGETS),
        "feature_root": str(plan.feature_root),
        "probe_root": str(plan.probe_root),
        "completion_path": str(plan.completion_path),
        "preregistration_lock_sha256": plan.preregistration_sha256,
        "nuisance_sha256": plan.nuisance_sha256,
        "gate_sha256": dict(plan.gate_sha256),
        "exporter_completion_sha256": plan.exporter_completion_sha256,
        "legacy_pvalid": "NA_no_source_authoritative_mask",
        "legacy_poracle": "NA_incomplete_source_authoritative_support_1488_of_1500",
        "fabricated_unavailable_rows": 0,
    }


def execute_probe_plan(
    plan: ProbeMatrixPlan,
    *,
    records: Mapping[str, Any],
    nuisance_targets: Mapping[str, Mapping[str, Any]] | None,
    workers: int,
) -> dict[str, Any]:
    """Execute all cells and publish one owner-only completion inventory."""

    workers = int(workers)
    if workers < 1:
        raise ValueError("workers must be a positive integer")
    if plan.completion_path.exists():
        raise FileExistsError(
            f"refusing to overwrite probe completion inventory: {plan.completion_path}"
        )
    existing = [cell.output_dir for cell in plan.cells if cell.output_dir.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite frozen probe cell: {existing[0]}")
    if any(cell.include_nuisance for cell in plan.cells):
        if nuisance_targets is None or set(nuisance_targets) != set(NUISANCE_TARGETS):
            raise ValueError("the exact ten nuisance targets were not loaded")

    rows: list[dict[str, Any]] = []
    if workers == 1:
        for specification in plan.cells:
            rows.append(execute_probe_cell(specification, records, nuisance_targets))
    else:
        future_map = {}
        with ProcessPoolExecutor(
            max_workers=min(workers, len(plan.cells)),
            initializer=_initialize_worker,
            initargs=(records, nuisance_targets),
        ) as executor:
            for specification in plan.cells:
                future_map[executor.submit(_worker_execute, specification)] = specification
            try:
                for future in as_completed(future_map):
                    rows.append(future.result())
            except BaseException:
                for future in future_map:
                    future.cancel()
                raise
    rows.sort(key=lambda row: (row["seed_base"], row["arm"], row["fold"], row["pooling"]))
    if len(rows) != len(plan.cells):
        raise AssertionError("probe execution did not return the exact planned cell count")
    by_key = {
        ProbeCellKey(
            int(row["seed_base"]),
            str(row["arm"]),
            int(row["fold"]),
            str(row["pooling"]),
        ).inventory_key: row
        for row in rows
    }
    if len(by_key) != len(plan.cells):
        raise ValueError("probe completion rows duplicate a matrix cell")
    completion = {
        **plan_summary(plan, status="COMPLETE"),
        "executed_cell_count": len(rows),
        "workers": workers,
        "probe_runner_sha256": file_sha256(Path(__file__)),
        "feature_metadata_sha256": {
            cell.key.inventory_key: cell.feature_metadata_sha256 for cell in plan.cells
        },
        "cells": by_key,
        "patient_identifiers_private": True,
        "new_training_performed": False,
    }
    _ensure_private_tree(plan.probe_root, plan.completion_path.parent)
    _private_json_no_overwrite(plan.completion_path, completion)
    return completion


def prepare_formal_probe_matrix(
    *,
    stage: str,
    poolings: Sequence[str],
    feature_root: str | Path = EXPERIMENT_ROOT / "features",
    probe_root: str | Path = EXPERIMENT_ROOT / "probes",
    nuisance_path: str | Path = EXPERIMENT_ROOT
    / "manifests"
    / "nuisance_targets.private.csv",
    equivalence_gate_path: str | Path = EXPERIMENT_ROOT
    / "metrics"
    / "p0_equivalence_gate.json",
    probe_replication_gate_path: str | Path = EXPERIMENT_ROOT
    / "metrics"
    / "p0_probe_replication_gate.json",
) -> tuple[
    ProbeMatrixPlan,
    Mapping[str, Any],
    Mapping[str, Mapping[str, Any]] | None,
]:
    """Load formal runtime data and preflight the requested matrix without fitting."""

    stage, poolings = validate_stage_poolings(stage, poolings)
    lock = verify_preregistration()
    preregistration_path = EXPERIMENT_ROOT / "PREREGISTRATION_LOCK.json"
    preregistration_sha256 = file_sha256(preregistration_path)
    gate_sha256 = validate_alternative_gates(
        poolings,
        preregistration_sha256=preregistration_sha256,
        equivalence_gate_path=equivalence_gate_path,
        probe_replication_gate_path=probe_replication_gate_path,
    )
    exporter_completion_sha256 = validate_exporter_completion(
        feature_root,
        stage=stage,
        preregistration_sha256=preregistration_sha256,
    )
    _authorization, paths, data = load_stage_b_bundle(verify_cache_files=False)
    common_provenance = {
        "fold_manifest_path": str(paths.fold_manifest),
        "fold_manifest_sha256": paths.fold_manifest_sha256,
        "data_contract_provenance_sha256": canonical_sha256(data.provenance),
        "exporter_completion_sha256": exporter_completion_sha256,
        **gate_sha256,
    }
    specifications = discover_feature_matrix(
        feature_root=feature_root,
        probe_root=probe_root,
        stage=stage,
        poolings=poolings,
        folds=data.folds,
        lock=lock,
        preregistration_path=preregistration_path,
        common_provenance=common_provenance,
    )
    needs_nuisance = any(cell.include_nuisance for cell in specifications)
    nuisance_targets = None
    nuisance_source: Path | None = None
    nuisance_sha256: str | None = None
    if needs_nuisance:
        nuisance_source = Path(nuisance_path).resolve()
        patients = tuple(
            data.folds.loc[data.folds["fold"].eq(FOLDS[0]), "patient_id"].astype(str)
        )
        nuisance_targets = load_nuisance_targets(nuisance_source, patients)
        nuisance_sha256 = file_sha256(nuisance_source)
        specifications = tuple(
            ProbeCellSpec(
                **{
                    **specification.__dict__,
                    "provenance": {
                        **dict(specification.provenance),
                        "nuisance_targets_sha256": nuisance_sha256,
                        "nuisance_target_names": list(NUISANCE_TARGETS),
                    },
                }
            )
            for specification in specifications
        )
    plan = ProbeMatrixPlan(
        stage=stage,
        poolings=poolings,
        cells=specifications,
        feature_root=Path(feature_root).resolve(),
        probe_root=Path(probe_root).resolve(),
        preregistration_path=preregistration_path.resolve(),
        preregistration_sha256=preregistration_sha256,
        nuisance_path=nuisance_source,
        nuisance_sha256=nuisance_sha256,
        gate_sha256=gate_sha256,
        exporter_completion_sha256=exporter_completion_sha256,
    )
    return plan, data.ftv, nuisance_targets


__all__ = [
    "EXPORTER_METADATA_FIELDS",
    "NUISANCE_POOLINGS",
    "NUISANCE_TARGETS",
    "POOLING_SLUGS",
    "ProbeCellKey",
    "ProbeCellSpec",
    "ProbeMatrixPlan",
    "S3_REPRESENTATION_CONTRACT",
    "SECONDARY_POOLING",
    "FORMAL_POOLINGS",
    "STAGES",
    "discover_feature_matrix",
    "execute_probe_cell",
    "execute_probe_plan",
    "expected_feature_dimension",
    "expected_feature_keys",
    "expected_sidecar_keys",
    "feature_path_for",
    "load_nuisance_targets",
    "output_path_for",
    "parse_poolings",
    "plan_summary",
    "pooling_available",
    "prepare_formal_probe_matrix",
    "split_order_sha256",
    "validate_alternative_gates",
    "validate_exporter_completion",
    "validate_locked_bindings",
    "validate_stage_poolings",
]

"""Frozen final-spatial feature export for the C1B pooling audit.

The formal entry point in this module deliberately executes only two checkpoint
modules: ``model.encoder`` and ``model.response_projection``.  It never calls
the model forward method, projector, transition, target branch, or FTV head.
All alternate poolings are deterministic functions of the full encoder output
and the preregistered, outcome-free audit sidecars.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .contracts import (
    ARMS,
    EXPERIMENT_ROOT,
    REPO_ROOT,
    cell_key,
    checkpoint_path as formal_checkpoint_path,
    file_sha256,
    reference_feature_metadata_path,
    reference_feature_path,
    relative,
)
from .pooling import (
    RESPONSE_DIM,
    apply_frozen_response_projection,
    concatenate_local_global,
    expected_feature_shape,
    global_average_pool,
    weighted_average_pool,
)
from .runtime import load_selected_model, verify_preregistration


FORMAL_PATIENT_COUNT = 808
VISIT_COUNT = 4
LEGACY_INPUT_SHAPE_ZYX = (32, 96, 96)
C1B_INPUT_SHAPE_ZYX = (112, 176, 160)
LEGACY_FEATURE_SHAPE_ZYX = (4, 12, 12)
C1B_FEATURE_SHAPE_ZYX = (14, 22, 20)
FORMAL_ORACLE_VALID_COUNT = 1500

PRIMARY_POOLINGS = ("P0", "PVALID", "PLOCAL", "PLOCAL+GLOBAL", "PORACLE")
SECONDARY_POOLING = "PLOCAL+PVALID_SECONDARY"
POOLING_SLUGS: Mapping[str, str] = {
    "P0": "p0",
    "PVALID": "pvalid",
    "PLOCAL": "plocal",
    "PLOCAL+GLOBAL": "plocal_global",
    "PORACLE": "poracle",
    SECONDARY_POOLING: "plocal_pvalid_secondary",
}
LEGACY_POOLINGS = ("P0", "PLOCAL", "PLOCAL+GLOBAL")
C1B_POOLINGS = (*PRIMARY_POOLINGS, SECONDARY_POOLING)

SIDECAR_KEYS = frozenset(
    {
        "patient_id",
        "c1b_valid_weight_final",
        "c1b_oracle_weight_final",
        "c1b_oracle_valid",
        "c1b_local_weight_final",
        "legacy_local_weight_final",
    }
)
FEATURE_ASSET_KEYS = frozenset(
    {
        "patient_id",
        "split",
        "state",
        "state_valid",
        "arm",
        "seed_base",
        "fold",
        "pooling",
    }
)


def _scalar(array: np.ndarray, *, name: str) -> Any:
    value = np.asarray(array)
    if value.shape != ():
        raise ValueError(f"{name} must be a scalar array")
    return value.item()


def _ordered_sha256(values: Sequence[str]) -> str:
    import hashlib

    return hashlib.sha256("\n".join(str(value) for value in values).encode("utf-8")).hexdigest()


def _canonical_sha256(value: Any) -> str:
    import hashlib

    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(0o700)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    _private_directory(path.parent)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        np.savez_compressed(temporary_path, **arrays)
        temporary_path.chmod(0o600)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _private_directory(path.parent)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        temporary_path.chmod(0o600)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def normalize_pooling(pooling: str) -> str:
    value = str(pooling).strip().upper()
    aliases = {slug.upper(): name for name, slug in POOLING_SLUGS.items()}
    value = aliases.get(value, value)
    if value not in POOLING_SLUGS:
        raise ValueError(f"unknown or unregistered pooling: {pooling!r}")
    return value


def pooling_slug(pooling: str) -> str:
    return POOLING_SLUGS[normalize_pooling(pooling)]


def feature_asset_path(
    feature_root: str | Path,
    seed_base: int,
    arm: str,
    fold: int,
    pooling: str,
    *,
    stage: str = "final",
) -> Path:
    """Return the one canonical private feature path for a formal cell/pooling."""

    if str(stage).lower() != "final":
        raise ValueError("this exporter implements only the preregistered final stage")
    arm_name = str(arm).upper()
    if arm_name not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    if int(seed_base) not in (2026, 3026) or int(fold) not in range(5):
        raise ValueError("formal feature identity must be seed 2026/3026 and fold 0..4")
    pooling_name = normalize_pooling(pooling)
    allowed = LEGACY_POOLINGS if arm_name.startswith("L") else C1B_POOLINGS
    if pooling_name not in allowed:
        raise ValueError(f"{pooling_name} is undefined for arm {arm_name}")
    return (
        Path(feature_root).expanduser().resolve()
        / "final"
        / f"seed_{int(seed_base)}"
        / arm_name
        / f"fold_{int(fold)}"
        / f"{pooling_slug(pooling_name)}.private.npz"
    )


def feature_metadata_path(asset_path: str | Path) -> Path:
    path = Path(asset_path)
    if not path.name.endswith(".private.npz"):
        raise ValueError("feature assets must end in .private.npz")
    return path.with_suffix(".metadata.json")


@dataclass(frozen=True)
class AuditSidecars:
    """Fully validated outcome-free spatial weights in formal patient order."""

    path: Path
    sha256: str
    patient_id: tuple[str, ...]
    c1b_valid_weight_final: np.ndarray
    c1b_oracle_weight_final: np.ndarray
    c1b_oracle_valid: np.ndarray
    c1b_local_weight_final: np.ndarray
    legacy_local_weight_final: np.ndarray


def _validate_weight_array(
    array: np.ndarray,
    *,
    name: str,
    shape: tuple[int, ...],
    require_support: np.ndarray | bool,
) -> np.ndarray:
    value = np.asarray(array)
    if value.dtype != np.float32 or value.shape != shape:
        raise ValueError(f"{name} must be float32 {shape}, got {value.dtype}/{value.shape}")
    if not np.isfinite(value).all() or np.any(value < 0.0) or np.any(value > 1.0):
        raise ValueError(f"{name} must contain only finite [0,1] weights")
    if value.ndim == 3:
        support = np.asarray(value.sum(dtype=np.float64) > 0.0)
    else:
        support = value.reshape(*value.shape[:2], -1).sum(axis=-1, dtype=np.float64) > 0.0
    required = np.asarray(require_support, dtype=bool)
    if required.shape not in {(), support.shape}:
        raise ValueError(f"{name} support-validity shape mismatch")
    required = np.broadcast_to(required, support.shape)
    if not bool(np.all(support[required])):
        raise ValueError(f"{name} has empty support for a required row")
    if bool(np.any(support[~required])):
        raise ValueError(f"{name} has nonzero support for a declared-invalid row")
    return value


def load_audit_sidecars(
    path: str | Path,
    expected_patient_ids: Sequence[str],
    *,
    c1b_feature_shape_zyx: tuple[int, int, int] = C1B_FEATURE_SHAPE_ZYX,
    legacy_feature_shape_zyx: tuple[int, int, int] = LEGACY_FEATURE_SHAPE_ZYX,
    expected_oracle_valid_count: int | None = FORMAL_ORACLE_VALID_COUNT,
) -> AuditSidecars:
    """Load the exact six-key sidecar NPZ and reject every geometry ambiguity."""

    source = Path(path).expanduser().resolve()
    if not source.is_file() or not source.name.endswith(".private.npz"):
        raise FileNotFoundError(f"audit sidecar must be an existing .private.npz: {source}")
    with np.load(source, allow_pickle=False) as archive:
        if set(archive.files) != SIDECAR_KEYS:
            raise ValueError(
                "audit sidecar keys drifted: "
                f"expected {sorted(SIDECAR_KEYS)}, got {sorted(archive.files)}"
            )
        arrays = {name: np.asarray(archive[name]) for name in archive.files}

    stored_patient_ids = tuple(np.asarray(arrays["patient_id"]).astype(str).tolist())
    expected_ids = tuple(str(value) for value in expected_patient_ids)
    if (
        len(set(stored_patient_ids)) != len(stored_patient_ids)
        or len(set(expected_ids)) != len(expected_ids)
        or set(stored_patient_ids) != set(expected_ids)
    ):
        raise ValueError("audit sidecar patient identity set differs from the formal fold population")
    # The sidecar owns one global, outcome-free patient order, whereas each
    # feature cell owns fold-local train->val->test order.  Reindex every
    # patient-axis array explicitly; never assume those orders happen to match.
    stored_lookup = {patient_id: index for index, patient_id in enumerate(stored_patient_ids)}
    reorder = np.asarray([stored_lookup[patient_id] for patient_id in expected_ids], dtype=np.int64)
    patient_ids = expected_ids
    for name in (
        "c1b_valid_weight_final",
        "c1b_oracle_weight_final",
        "c1b_oracle_valid",
        "legacy_local_weight_final",
    ):
        arrays[name] = np.asarray(arrays[name])[reorder]
    count = len(patient_ids)
    oracle_valid = np.asarray(arrays["c1b_oracle_valid"])
    if oracle_valid.dtype != np.bool_ or oracle_valid.shape != (count, VISIT_COUNT):
        raise ValueError("c1b_oracle_valid must be bool [N,4]")
    if expected_oracle_valid_count is not None and int(oracle_valid.sum()) != int(
        expected_oracle_valid_count
    ):
        raise ValueError("C1B oracle-valid population differs from the frozen formal 1500 rows")

    c1b_shape = (count, VISIT_COUNT, *c1b_feature_shape_zyx)
    legacy_shape = (count, VISIT_COUNT, *legacy_feature_shape_zyx)
    valid = _validate_weight_array(
        arrays["c1b_valid_weight_final"],
        name="c1b_valid_weight_final",
        shape=c1b_shape,
        require_support=True,
    )
    oracle = _validate_weight_array(
        arrays["c1b_oracle_weight_final"],
        name="c1b_oracle_weight_final",
        shape=c1b_shape,
        require_support=oracle_valid,
    )
    c1b_local = _validate_weight_array(
        arrays["c1b_local_weight_final"],
        name="c1b_local_weight_final",
        shape=c1b_feature_shape_zyx,
        require_support=True,
    )
    legacy_local = _validate_weight_array(
        arrays["legacy_local_weight_final"],
        name="legacy_local_weight_final",
        shape=legacy_shape,
        require_support=True,
    )
    return AuditSidecars(
        path=source,
        sha256=file_sha256(source),
        patient_id=patient_ids,
        c1b_valid_weight_final=valid,
        c1b_oracle_weight_final=oracle,
        c1b_oracle_valid=oracle_valid,
        c1b_local_weight_final=c1b_local,
        legacy_local_weight_final=legacy_local,
    )


@dataclass(frozen=True)
class FrozenFeatureAsset:
    path: Path
    patient_id: tuple[str, ...]
    split: tuple[str, ...]
    state: np.ndarray
    state_valid: np.ndarray
    arm: str
    seed_base: int
    fold: int
    pooling: str


def load_feature_asset(
    path: str | Path,
    *,
    expected_arm: str | None = None,
    expected_seed_base: int | None = None,
    expected_fold: int | None = None,
    expected_pooling: str | None = None,
    expected_patient_count: int | None = None,
) -> FrozenFeatureAsset:
    """Load and structurally validate one exported pooling asset."""

    source = Path(path).expanduser().resolve()
    if not source.is_file() or not source.name.endswith(".private.npz"):
        raise FileNotFoundError(f"feature asset is missing or not private: {source}")
    with np.load(source, allow_pickle=False) as archive:
        if set(archive.files) != FEATURE_ASSET_KEYS:
            raise ValueError(
                "feature asset keys drifted: "
                f"expected {sorted(FEATURE_ASSET_KEYS)}, got {sorted(archive.files)}"
            )
        patient_id = tuple(np.asarray(archive["patient_id"]).astype(str).tolist())
        split = tuple(np.asarray(archive["split"]).astype(str).tolist())
        state = np.asarray(archive["state"])
        state_valid = np.asarray(archive["state_valid"])
        arm = str(_scalar(archive["arm"], name="arm")).upper()
        seed_base = int(_scalar(archive["seed_base"], name="seed_base"))
        fold = int(_scalar(archive["fold"], name="fold"))
        pooling = normalize_pooling(str(_scalar(archive["pooling"], name="pooling")))

    count = len(patient_id)
    expected_dim = 2 * RESPONSE_DIM if pooling in {
        "PLOCAL+GLOBAL",
        SECONDARY_POOLING,
    } else RESPONSE_DIM
    if expected_patient_count is not None and count != int(expected_patient_count):
        raise ValueError("feature patient count differs from expectation")
    if len(set(patient_id)) != count or len(split) != count:
        raise ValueError("feature patient IDs must be unique and align with split")
    if set(split) != {"train", "val", "test"}:
        raise ValueError("feature split must contain the formal train/val/test labels")
    if state.dtype != np.float32 or state.shape != (count, VISIT_COUNT, expected_dim):
        raise ValueError(
            f"feature state must be float32 [N,4,{expected_dim}], got {state.dtype}/{state.shape}"
        )
    if state_valid.dtype != np.bool_ or state_valid.shape != (count, VISIT_COUNT):
        raise ValueError("feature state_valid must be bool [N,4]")
    if not np.isfinite(state).all():
        raise FloatingPointError("feature state contains NaN/Inf")
    if bool(np.any(state[~state_valid] != 0.0)):
        raise ValueError("invalid feature rows must use the explicit zero placeholder")
    allowed = LEGACY_POOLINGS if arm.startswith("L") else C1B_POOLINGS
    if arm not in ARMS or pooling not in allowed:
        raise ValueError(f"pooling {pooling} is not defined for arm {arm}")
    if pooling == "PORACLE":
        if arm.startswith("L") or not bool((~state_valid).any()):
            raise ValueError("PORACLE must be the C1B formal subset with explicit invalid rows")
    elif not bool(state_valid.all()):
        raise ValueError(f"{pooling} must be valid for every formal patient/visit")

    expected = {
        "arm": None if expected_arm is None else str(expected_arm).upper(),
        "seed_base": expected_seed_base,
        "fold": expected_fold,
        "pooling": None if expected_pooling is None else normalize_pooling(expected_pooling),
    }
    observed = {
        "arm": arm,
        "seed_base": seed_base,
        "fold": fold,
        "pooling": pooling,
    }
    for name, value in expected.items():
        if value is not None and observed[name] != value:
            raise ValueError(f"feature {name} differs from expectation")
    return FrozenFeatureAsset(
        source, patient_id, split, state, state_valid, arm, seed_base, fold, pooling
    )


def validate_feature_export(
    path: str | Path,
    *,
    expected_arm: str | None = None,
    expected_seed_base: int | None = None,
    expected_fold: int | None = None,
    expected_pooling: str | None = None,
    expected_patient_count: int | None = None,
    verify_live_inputs: bool = True,
) -> tuple[FrozenFeatureAsset, dict[str, Any]]:
    """Validate an asset and its same-stem provenance metadata."""

    asset = load_feature_asset(
        path,
        expected_arm=expected_arm,
        expected_seed_base=expected_seed_base,
        expected_fold=expected_fold,
        expected_pooling=expected_pooling,
        expected_patient_count=expected_patient_count,
    )
    metadata_source = feature_metadata_path(asset.path)
    metadata = _read_json(metadata_source)
    required = {
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
    if set(metadata) != required:
        raise ValueError(
            "feature metadata fields drifted: "
            f"expected {sorted(required)}, got {sorted(metadata)}"
        )
    identity = {
        "schema_version": 1,
        "stage": "final",
        "status": "COMPLETE",
        "arm": asset.arm,
        "seed_base": asset.seed_base,
        "fold": asset.fold,
        "pooling": asset.pooling,
        "pooling_slug": pooling_slug(asset.pooling),
        "feature_path": str(asset.path),
        "feature_sha256": file_sha256(asset.path),
        "state_shape": list(asset.state.shape),
        "state_dtype": "float32",
        "state_valid_shape": list(asset.state_valid.shape),
        "state_valid_count": int(asset.state_valid.sum()),
        "patient_count": len(asset.patient_id),
        "patient_order_sha256": _ordered_sha256(asset.patient_id),
        "split_order_sha256": _ordered_sha256(asset.split),
    }
    for name, value in identity.items():
        if metadata.get(name) != value:
            raise ValueError(f"feature metadata is inconsistent at {name}")
    forbidden_false = (
        "training_performed",
        "projector_called",
        "transition_called",
        "target_encoder_called",
        "ftv_head_called",
        "test_labels_used",
    )
    if any(metadata.get(name) is not False for name in forbidden_false):
        raise ValueError("feature metadata reports a forbidden training/model/label call")
    if metadata.get("feature_tensor") != "full_model.encoder_output_before_gap":
        raise ValueError("feature metadata does not bind the full pre-GAP encoder output")
    if metadata.get("response_projection") != "frozen_online_Linear128x192_plus_LayerNorm":
        raise ValueError("feature metadata response projection contract drifted")
    if sorted(metadata.get("sidecar_keys_used", [])) != sorted(
        _sidecar_keys_for(asset.arm, asset.pooling)
    ):
        raise ValueError("feature metadata sidecar dependency set drifted")

    if verify_live_inputs:
        lock = verify_preregistration()
        lock_path = EXPERIMENT_ROOT / "PREREGISTRATION_LOCK.json"
        checks = {
            "preregistration_lock_sha256": file_sha256(lock_path),
            "plan_sha256": lock["plan_sha256"],
            "config_sha256": lock["config_sha256"],
            "checkpoint_sha256": file_sha256(Path(metadata["checkpoint_path"])),
            "reference_feature_sha256": file_sha256(
                Path(metadata["reference_feature_path"])
            ),
            "reference_feature_metadata_sha256": file_sha256(
                Path(metadata["reference_feature_metadata_path"])
            ),
            "sidecar_sha256": file_sha256(Path(metadata["sidecar_path"])),
        }
        for name, value in checks.items():
            if metadata.get(name) != value:
                raise ValueError(f"live feature provenance drifted at {name}")
        key = cell_key(asset.seed_base, asset.arm, asset.fold)
        if metadata.get("checkpoint_lock_key") != key:
            raise ValueError("feature checkpoint lock key drifted")
        if lock["selected_checkpoints"][key]["sha256"] != metadata["checkpoint_sha256"]:
            raise ValueError("feature checkpoint no longer matches preregistration")
        reference = lock["formal_p0_references"][key]
        if reference["feature_sha256"] != metadata["reference_feature_sha256"]:
            raise ValueError("feature P0 reference no longer matches preregistration")
    return asset, metadata


def _as_spatial_weights(
    weights: torch.Tensor | np.ndarray,
    *,
    batch: int,
    visits: int,
    feature_shape: tuple[int, int, int],
    device: torch.device,
    shared: bool = False,
) -> torch.Tensor:
    value = torch.as_tensor(weights, device=device)
    if value.dtype != torch.float32:
        raise TypeError("formal spatial weights must retain sidecar float32 dtype")
    if shared:
        if tuple(value.shape) != feature_shape:
            raise ValueError("shared local weight shape differs from the feature grid")
        return value.reshape(1, 1, *feature_shape)
    if tuple(value.shape) != (batch, visits, *feature_shape):
        raise ValueError("per-visit spatial weights differ from [B,4,D,H,W]")
    return value.reshape(batch * visits, 1, *feature_shape)


def compute_final_pooling_states(
    model: torch.nn.Module,
    image: torch.Tensor,
    *,
    arm: str,
    local_weights: torch.Tensor | np.ndarray,
    valid_weights: torch.Tensor | np.ndarray | None = None,
    oracle_weights: torch.Tensor | np.ndarray | None = None,
    oracle_valid: torch.Tensor | np.ndarray | None = None,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Compute every registered final-stage state for one image batch.

    This low-level routine is intentionally usable with synthetic CPU modules in
    tests.  The formal writer separately requires CUDA and the official strict
    checkpoint loader.
    """

    arm_name = str(arm).upper()
    if arm_name not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    if not isinstance(image, torch.Tensor) or image.ndim != 6:
        raise ValueError("image must be a tensor [B,4,7,Z,Y,X]")
    if image.shape[1] != VISIT_COUNT or image.shape[2] != 7 or image.shape[0] <= 0:
        raise ValueError("image must have exactly four DCE7 visits")
    if image.dtype != torch.float32 or not bool(torch.isfinite(image).all()):
        raise ValueError("image must be finite float32")
    if model.training or model.encoder.training or model.response_projection.training:
        raise ValueError("encoder and response projection must be in frozen eval mode")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("all checkpoint parameters must be frozen for export")

    batch, visits = image.shape[:2]
    input_shape = tuple(int(value) for value in image.shape[-3:])
    feature_shape = expected_feature_shape(input_shape, stage="final")
    expected_input = LEGACY_INPUT_SHAPE_ZYX if arm_name.startswith("L") else C1B_INPUT_SHAPE_ZYX
    expected_feature = (
        LEGACY_FEATURE_SHAPE_ZYX if arm_name.startswith("L") else C1B_FEATURE_SHAPE_ZYX
    )
    # Synthetic tests may exercise smaller convolution-compatible grids.  Any
    # formal arm shape must nevertheless map to the exact frozen output shape.
    if input_shape == expected_input and feature_shape != expected_feature:
        raise AssertionError("frozen input-to-feature geometry drifted")

    if arm_name.startswith("L"):
        if valid_weights is not None or oracle_weights is not None or oracle_valid is not None:
            raise ValueError("legacy PVALID/PORACLE are preregistered NA; masks are forbidden")
        local = _as_spatial_weights(
            local_weights,
            batch=batch,
            visits=visits,
            feature_shape=feature_shape,
            device=image.device,
        )
    else:
        if valid_weights is None or oracle_weights is None or oracle_valid is None:
            raise ValueError("C1B export requires valid/oracle weights and oracle validity")
        local = _as_spatial_weights(
            local_weights,
            batch=batch,
            visits=visits,
            feature_shape=feature_shape,
            device=image.device,
            shared=True,
        )

    parameter_versions = tuple(parameter._version for parameter in model.parameters())
    flat = image.reshape(batch * visits, *image.shape[2:])
    with torch.inference_mode():
        spatial = model.encoder(flat)
    if not isinstance(spatial, torch.Tensor) or tuple(spatial.shape) != (
        batch * visits,
        128,
        *feature_shape,
    ):
        observed = getattr(spatial, "shape", type(spatial).__name__)
        raise ValueError(
            "full model.encoder output must be [B*4,128,D,H,W], "
            f"got {observed}"
        )
    if spatial.dtype != torch.float32 or not bool(torch.isfinite(spatial).all()):
        raise ValueError("full encoder spatial output must be finite float32")

    global_response = apply_frozen_response_projection(
        global_average_pool(spatial), model.response_projection
    ).reshape(batch, visits, RESPONSE_DIM)
    local_response = apply_frozen_response_projection(
        weighted_average_pool(spatial, local), model.response_projection
    ).reshape(batch, visits, RESPONSE_DIM)
    all_valid = torch.ones((batch, visits), dtype=torch.bool, device=image.device)
    output: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
        "P0": (global_response, all_valid),
        "PLOCAL": (local_response, all_valid),
        "PLOCAL+GLOBAL": (
            concatenate_local_global(local_response, global_response),
            all_valid,
        ),
    }

    if not arm_name.startswith("L"):
        valid = _as_spatial_weights(
            valid_weights,
            batch=batch,
            visits=visits,
            feature_shape=feature_shape,
            device=image.device,
        )
        oracle = _as_spatial_weights(
            oracle_weights,
            batch=batch,
            visits=visits,
            feature_shape=feature_shape,
            device=image.device,
        )
        oracle_valid_tensor = torch.as_tensor(oracle_valid, device=image.device)
        if oracle_valid_tensor.dtype != torch.bool or tuple(oracle_valid_tensor.shape) != (
            batch,
            visits,
        ):
            raise ValueError("oracle_valid must be bool [B,4]")
        valid_response = apply_frozen_response_projection(
            weighted_average_pool(spatial, valid), model.response_projection
        ).reshape(batch, visits, RESPONSE_DIM)
        flat_oracle_valid = oracle_valid_tensor.reshape(-1)
        oracle_response = torch.zeros(
            (batch * visits, RESPONSE_DIM),
            dtype=spatial.dtype,
            device=spatial.device,
        )
        if bool(flat_oracle_valid.any()):
            selected_spatial = spatial[flat_oracle_valid]
            selected_weights = oracle[flat_oracle_valid]
            oracle_response[flat_oracle_valid] = apply_frozen_response_projection(
                weighted_average_pool(selected_spatial, selected_weights),
                model.response_projection,
            )
        oracle_response = oracle_response.reshape(batch, visits, RESPONSE_DIM)
        output.update(
            {
                "PVALID": (valid_response, all_valid),
                "PORACLE": (oracle_response, oracle_valid_tensor),
                SECONDARY_POOLING: (
                    concatenate_local_global(local_response, valid_response),
                    all_valid,
                ),
            }
        )

    if tuple(parameter._version for parameter in model.parameters()) != parameter_versions:
        raise RuntimeError("checkpoint parameters mutated during frozen feature export")
    expected_poolings = set(LEGACY_POOLINGS if arm_name.startswith("L") else C1B_POOLINGS)
    if set(output) != expected_poolings:
        raise AssertionError("computed pooling set differs from preregistration")
    for pooling, (state, state_valid) in output.items():
        dimension = 2 * RESPONSE_DIM if pooling in {
            "PLOCAL+GLOBAL",
            SECONDARY_POOLING,
        } else RESPONSE_DIM
        if tuple(state.shape) != (batch, visits, dimension):
            raise AssertionError(f"{pooling} returned a wrong state shape")
        if tuple(state_valid.shape) != (batch, visits) or state_valid.dtype != torch.bool:
            raise AssertionError(f"{pooling} returned a wrong validity shape")
        if not bool(torch.isfinite(state).all()) or state.requires_grad:
            raise ValueError(f"{pooling} state is nonfinite or retains gradients")
        if bool((state[~state_valid] != 0).any()):
            raise ValueError(f"{pooling} invalid states must remain explicit zeros")
    return output


def _validate_checkpoint_data_contract(
    checkpoint: Mapping[str, Any], data: Any, splits: Any
) -> str:
    from c1b_stage_b.contracts import (
        LOGICAL_OBJECTIVE_CONTRACT,
        canonical_sha256,
        ordered_patient_sha256,
    )

    provenance = checkpoint.get("data_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("checkpoint has no structured Stage-B data provenance")
    digest = canonical_sha256(provenance)
    if checkpoint.get("data_provenance_sha256") != digest:
        raise ValueError("checkpoint data provenance digest is internally inconsistent")
    for name, value in data.provenance.items():
        if provenance.get(name) != value:
            raise ValueError(f"checkpoint/current data contract differ at {name}")
    expected = {
        "train_primary_order_sha256": ordered_patient_sha256(splits.train_primary),
        "train_all_order_sha256": ordered_patient_sha256(splits.train_all),
        "validation_order_sha256": ordered_patient_sha256(splits.val),
        "test_patient_count_not_loaded": len(splits.test),
        "model_forward_fields": ["image"],
        "auxiliary_fields": ["ftv_target", "ftv_mask"],
        "logical_objective_contract": dict(LOGICAL_OBJECTIVE_CONTRACT),
    }
    for name, value in expected.items():
        if provenance.get(name) != value:
            raise ValueError(f"checkpoint split/model provenance differs at {name}")
    if checkpoint.get("train_patient_sha256") != canonical_sha256(sorted(splits.train_all)):
        raise ValueError("checkpoint train-patient hash differs from the locked fold")
    if checkpoint.get("val_patient_sha256") != canonical_sha256(sorted(splits.val)):
        raise ValueError("checkpoint validation-patient hash differs from the locked fold")
    return digest


def _validate_checkpoint_selection(
    checkpoint: Mapping[str, Any], authorization: Any, *, arm: str, seed_base: int, fold: int
) -> None:
    if str(checkpoint.get("arm", "")).upper() != arm:
        raise ValueError("checkpoint arm mismatch")
    if int(checkpoint.get("seed_base", -1)) != seed_base or int(
        checkpoint.get("fold", -1)
    ) != fold:
        raise ValueError("checkpoint seed/fold mismatch")
    if checkpoint.get("selected") is not True or checkpoint.get("test_data_used") is not False:
        raise ValueError("feature export requires a selected, test-blind checkpoint")
    selection_path = Path(str(checkpoint.get("selection_path", ""))).resolve()
    if not selection_path.is_file() or checkpoint.get("selection_sha256") != file_sha256(
        selection_path
    ):
        raise ValueError("checkpoint selection binding is absent or stale")
    selection = _read_json(selection_path)
    if checkpoint.get("selection") != selection:
        raise ValueError("checkpoint embeds a different selection record")
    if int(checkpoint.get("epoch", -1)) != int(selection.get("selected_epoch", -2)):
        raise ValueError("checkpoint epoch differs from the selection record")
    if str(checkpoint.get("stage_a_sentinel_sha256", "")) != authorization.sha256:
        raise ValueError("checkpoint and current Stage-A authorization differ")


def _validate_reference_identity(
    path: Path, expected_ids: tuple[str, ...], expected_split: tuple[str, ...]
) -> None:
    with np.load(path, allow_pickle=False) as archive:
        required = {"patient_id", "split", "response_state", "arm", "seed_base", "fold"}
        if set(archive.files) != required:
            raise ValueError("immutable P0 reference schema drifted")
        ids = tuple(np.asarray(archive["patient_id"]).astype(str).tolist())
        split = tuple(np.asarray(archive["split"]).astype(str).tolist())
        response = np.asarray(archive["response_state"])
    if ids != expected_ids or split != expected_split:
        raise ValueError("formal patient order/split differs from the immutable P0 reference")
    if response.dtype != np.float32 or response.shape != (
        FORMAL_PATIENT_COUNT,
        VISIT_COUNT,
        RESPONSE_DIM,
    ):
        raise ValueError("immutable P0 reference shape/dtype drifted")


def _sidecar_keys_for(arm: str, pooling: str) -> tuple[str, ...]:
    arm_name = str(arm).upper()
    pooling_name = normalize_pooling(pooling)
    if pooling_name == "P0":
        return ()
    if pooling_name in {"PLOCAL", "PLOCAL+GLOBAL"}:
        return (
            "legacy_local_weight_final" if arm_name.startswith("L") else "c1b_local_weight_final",
        )
    if pooling_name == "PVALID":
        return ("c1b_valid_weight_final",)
    if pooling_name == "PORACLE":
        return ("c1b_oracle_weight_final", "c1b_oracle_valid")
    if pooling_name == SECONDARY_POOLING:
        return ("c1b_local_weight_final", "c1b_valid_weight_final")
    raise AssertionError("unreachable registered pooling")


def _write_feature(
    path: Path,
    *,
    patient_ids: tuple[str, ...],
    split_labels: tuple[str, ...],
    state: np.ndarray,
    state_valid: np.ndarray,
    arm: str,
    seed_base: int,
    fold: int,
    pooling: str,
    metadata_base: Mapping[str, Any],
) -> dict[str, Any]:
    arrays = {
        "patient_id": np.asarray(patient_ids, dtype=str),
        "split": np.asarray(split_labels, dtype=str),
        "state": np.asarray(state, dtype=np.float32),
        "state_valid": np.asarray(state_valid, dtype=bool),
        "arm": np.asarray(arm),
        "seed_base": np.asarray(seed_base, dtype=np.int64),
        "fold": np.asarray(fold, dtype=np.int64),
        "pooling": np.asarray(pooling),
    }
    _atomic_npz(path, arrays)
    metadata = {
        **metadata_base,
        "schema_version": 1,
        "stage": "final",
        "status": "COMPLETE",
        "arm": arm,
        "seed_base": seed_base,
        "fold": fold,
        "pooling": pooling,
        "pooling_slug": pooling_slug(pooling),
        "feature_path": str(path),
        "feature_sha256": file_sha256(path),
        "state_shape": list(state.shape),
        "state_dtype": "float32",
        "state_valid_shape": list(state_valid.shape),
        "state_valid_count": int(state_valid.sum()),
        "patient_count": len(patient_ids),
        "patient_order_sha256": _ordered_sha256(patient_ids),
        "split_order_sha256": _ordered_sha256(split_labels),
        "sidecar_keys_used": list(_sidecar_keys_for(arm, pooling)),
    }
    _atomic_json(feature_metadata_path(path), metadata)
    return metadata


@torch.no_grad()
def export_frozen_feature_cell(
    *,
    checkpoint_path: str | Path,
    arm: str,
    seed_base: int,
    fold: int,
    data: Any,
    authorization: Any,
    sidecar_path: str | Path,
    feature_root: str | Path,
    device: torch.device,
    batch_size: int = 4,
    workers: int = 2,
) -> dict[str, dict[str, Any]]:
    """Export all and only registered final poolings for one formal checkpoint."""

    lock = verify_preregistration()
    arm_name = str(arm).upper()
    seed = int(seed_base)
    fold_index = int(fold)
    if arm_name not in ARMS or seed not in (2026, 3026) or fold_index not in range(5):
        raise ValueError("invalid formal cell identity")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal frozen feature export requires CUDA")
    if int(batch_size) != 4 or int(workers) != 2:
        raise ValueError("formal P0 parity requires the frozen batch_size=4/workers=2 schedule")
    expected_feature_root = (EXPERIMENT_ROOT / "features").resolve()
    if Path(feature_root).expanduser().resolve() != expected_feature_root:
        raise ValueError("formal outputs must remain in this audit's private features root")
    expected_sidecar = (
        EXPERIMENT_ROOT / "manifests" / "audit_sidecars.private.npz"
    ).resolve()
    if Path(sidecar_path).expanduser().resolve() != expected_sidecar:
        raise ValueError("formal export requires the canonical audit sidecar path")
    if expected_sidecar.stat().st_mode & 0o077:
        raise PermissionError("formal audit sidecar must be owner-only")
    expected_checkpoint = formal_checkpoint_path(seed, arm_name, fold_index).resolve()
    source_checkpoint = Path(checkpoint_path).expanduser().resolve()
    if source_checkpoint != expected_checkpoint:
        raise ValueError("checkpoint path is not the exact preregistered formal cell")
    key = cell_key(seed, arm_name, fold_index)
    checkpoint_lock = lock["selected_checkpoints"][key]
    checkpoint_sha = file_sha256(source_checkpoint)
    if checkpoint_sha != checkpoint_lock["sha256"] or relative(source_checkpoint) != checkpoint_lock["path"]:
        raise ValueError("checkpoint hash/path differs from preregistration")

    from c1b_stage_b.data import StageBDataset, arm_cache, make_splits

    splits = make_splits(data.folds, fold_index, data.train_only_ids)
    patient_ids = tuple(splits.train_primary + splits.val + splits.test)
    split_labels = tuple(
        ("train",) * len(splits.train_primary)
        + ("val",) * len(splits.val)
        + ("test",) * len(splits.test)
    )
    if len(patient_ids) != FORMAL_PATIENT_COUNT or len(set(patient_ids)) != FORMAL_PATIENT_COUNT:
        raise ValueError("formal feature export must contain exactly 808 unique fold patients")
    reference_path = reference_feature_path(seed, arm_name, fold_index).resolve()
    reference_metadata = reference_feature_metadata_path(seed, arm_name, fold_index).resolve()
    reference_lock = lock["formal_p0_references"][key]
    if (
        file_sha256(reference_path) != reference_lock["feature_sha256"]
        or file_sha256(reference_metadata) != reference_lock["feature_metadata_sha256"]
        or _ordered_sha256(patient_ids) != reference_lock["patient_order_sha256"]
    ):
        raise ValueError("immutable P0 reference provenance drifted")
    _validate_reference_identity(reference_path, patient_ids, split_labels)

    sidecars = load_audit_sidecars(sidecar_path, patient_ids)
    allowed_poolings = LEGACY_POOLINGS if arm_name.startswith("L") else C1B_POOLINGS
    output_paths = {
        pooling: feature_asset_path(
            feature_root, seed, arm_name, fold_index, pooling, stage="final"
        )
        for pooling in allowed_poolings
    }
    collisions = [
        path
        for path in output_paths.values()
        if path.exists() or feature_metadata_path(path).exists()
    ]
    if collisions:
        raise FileExistsError(f"refusing to overwrite frozen feature outputs: {collisions[0]}")

    model, checkpoint = load_selected_model(source_checkpoint, device)
    _validate_checkpoint_selection(
        checkpoint, authorization, arm=arm_name, seed_base=seed, fold=fold_index
    )
    checkpoint_data_sha = _validate_checkpoint_data_contract(checkpoint, data, splits)
    cache = arm_cache(arm_name, data.legacy_cache, data.c1b_cache)
    dataset = StageBDataset(patient_ids, cache, transformed_ftv={})
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        drop_last=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=1,
    )
    observed_ids: list[str] = []
    state_parts: dict[str, list[np.ndarray]] = {pooling: [] for pooling in allowed_poolings}
    valid_parts: dict[str, list[np.ndarray]] = {pooling: [] for pooling in allowed_poolings}
    offset = 0
    for batch in loader:
        batch_ids = tuple(str(value) for value in batch["patient_id"])
        expected_batch_ids = patient_ids[offset : offset + len(batch_ids)]
        if batch_ids != expected_batch_ids:
            raise AssertionError("feature DataLoader changed formal patient order")
        image = batch["image"].to(device, non_blocking=True)
        indices = slice(offset, offset + len(batch_ids))
        if arm_name.startswith("L"):
            states = compute_final_pooling_states(
                model,
                image,
                arm=arm_name,
                local_weights=sidecars.legacy_local_weight_final[indices],
            )
        else:
            states = compute_final_pooling_states(
                model,
                image,
                arm=arm_name,
                local_weights=sidecars.c1b_local_weight_final,
                valid_weights=sidecars.c1b_valid_weight_final[indices],
                oracle_weights=sidecars.c1b_oracle_weight_final[indices],
                oracle_valid=sidecars.c1b_oracle_valid[indices],
            )
        if set(states) != set(allowed_poolings):
            raise AssertionError("batch pooling inventory drifted")
        for pooling, (state, state_valid) in states.items():
            state_parts[pooling].append(state.detach().float().cpu().numpy())
            valid_parts[pooling].append(state_valid.detach().cpu().numpy())
        observed_ids.extend(batch_ids)
        offset += len(batch_ids)
    if tuple(observed_ids) != patient_ids or offset != FORMAL_PATIENT_COUNT:
        raise AssertionError("formal feature export did not cover the exact 808-patient order")

    complete_states = {
        pooling: np.concatenate(parts, axis=0).astype(np.float32, copy=False)
        for pooling, parts in state_parts.items()
    }
    complete_validity = {
        pooling: np.concatenate(parts, axis=0).astype(bool, copy=False)
        for pooling, parts in valid_parts.items()
    }
    for pooling in allowed_poolings:
        dimension = 384 if pooling in {"PLOCAL+GLOBAL", SECONDARY_POOLING} else 192
        if complete_states[pooling].shape != (FORMAL_PATIENT_COUNT, 4, dimension):
            raise AssertionError(f"{pooling} final state shape drifted")
        if pooling == "PORACLE":
            if int(complete_validity[pooling].sum()) != FORMAL_ORACLE_VALID_COUNT:
                raise ValueError("PORACLE state population differs from the formal 1500 rows")
        elif not complete_validity[pooling].all():
            raise ValueError(f"{pooling} unexpectedly contains invalid formal rows")

    lock_path = EXPERIMENT_ROOT / "PREREGISTRATION_LOCK.json"
    metadata_base = {
        "checkpoint_path": str(source_checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_lock_key": key,
        "reference_feature_path": str(reference_path),
        "reference_feature_sha256": reference_lock["feature_sha256"],
        "reference_feature_metadata_path": str(reference_metadata),
        "reference_feature_metadata_sha256": reference_lock["feature_metadata_sha256"],
        "preregistration_lock_sha256": file_sha256(lock_path),
        "plan_sha256": lock["plan_sha256"],
        "config_sha256": lock["config_sha256"],
        "sidecar_path": str(sidecars.path),
        "sidecar_sha256": sidecars.sha256,
        "data_contract_provenance_sha256": _canonical_sha256(data.provenance),
        "checkpoint_data_provenance_sha256": checkpoint_data_sha,
        "stage_a_sentinel_sha256": authorization.sha256,
        "implementation_sha256": {
            "exporter.py": file_sha256(Path(__file__)),
            "pooling.py": file_sha256(Path(__file__).with_name("pooling.py")),
            "runtime.py": file_sha256(Path(__file__).with_name("runtime.py")),
            "contracts.py": file_sha256(Path(__file__).with_name("contracts.py")),
        },
        "device": str(device),
        "batch_size": 4,
        "workers": 2,
        "feature_tensor": "full_model.encoder_output_before_gap",
        "response_projection": "frozen_online_Linear128x192_plus_LayerNorm",
        "training_performed": False,
        "projector_called": False,
        "transition_called": False,
        "target_encoder_called": False,
        "ftv_head_called": False,
        "test_labels_used": False,
    }
    metadata_by_pooling: dict[str, dict[str, Any]] = {}
    for pooling in allowed_poolings:
        metadata_by_pooling[pooling] = _write_feature(
            output_paths[pooling],
            patient_ids=patient_ids,
            split_labels=split_labels,
            state=complete_states[pooling],
            state_valid=complete_validity[pooling],
            arm=arm_name,
            seed_base=seed,
            fold=fold_index,
            pooling=pooling,
            metadata_base=metadata_base,
        )

    # Re-read every just-written output, and prove that all immutable inputs and
    # the selected checkpoint remained byte-identical throughout GPU inference.
    if (
        file_sha256(source_checkpoint) != checkpoint_sha
        or file_sha256(reference_path) != reference_lock["feature_sha256"]
        or file_sha256(reference_metadata) != reference_lock["feature_metadata_sha256"]
        or file_sha256(sidecars.path) != sidecars.sha256
    ):
        raise RuntimeError("an immutable checkpoint/reference/sidecar changed during export")
    for pooling, path in output_paths.items():
        validate_feature_export(
            path,
            expected_arm=arm_name,
            expected_seed_base=seed,
            expected_fold=fold_index,
            expected_pooling=pooling,
            expected_patient_count=FORMAL_PATIENT_COUNT,
            verify_live_inputs=True,
        )
    return metadata_by_pooling


__all__ = [
    "AuditSidecars",
    "C1B_POOLINGS",
    "FEATURE_ASSET_KEYS",
    "FrozenFeatureAsset",
    "LEGACY_POOLINGS",
    "POOLING_SLUGS",
    "SECONDARY_POOLING",
    "compute_final_pooling_states",
    "export_frozen_feature_cell",
    "feature_asset_path",
    "feature_metadata_path",
    "load_audit_sidecars",
    "load_feature_asset",
    "normalize_pooling",
    "pooling_slug",
    "validate_feature_export",
]

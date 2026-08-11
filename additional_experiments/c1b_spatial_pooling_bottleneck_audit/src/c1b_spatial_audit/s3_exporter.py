"""Conditional export of raw pooled ``encoder.features[2]`` representations.

This exporter is isolated from the already-bound final-stage exporter.  It
executes the first three *full residual blocks* of the frozen online encoder,
then produces only the preregistered raw 64-D P0/PLOCAL/PORACLE states.  The
128->192 response projection and every downstream/model-target module remain
uncalled.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
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
    cell_key,
    checkpoint_path as formal_checkpoint_path,
    file_sha256,
    reference_feature_metadata_path,
    reference_feature_path,
    relative,
)
from .pooling import expected_feature_shape, global_average_pool, weighted_average_pool
from .runtime import load_selected_model, verify_preregistration
from .s3_sidecars import (
    C1B_S3_SHAPE_ZYX,
    FORMAL_ORACLE_VISIT_COUNT,
    FORMAL_PATIENT_COUNT,
    LEGACY_S3_SHAPE_ZYX,
    LoadedS3Sidecars,
    load_s3_sidecars,
)
from .s3_trigger import require_s3_trigger_authorization
from .sidecars import C1B_INPUT_SHAPE_ZYX, LEGACY_INPUT_SHAPE_ZYX


VISIT_COUNT = 4
S3_CHANNELS = 64
S3_REPRESENTATION_CONTRACT = "raw_encoder_features2_pooled_64d_no_projection"
S3_POOLING_SLUGS = {"P0": "p0", "PLOCAL": "plocal", "PORACLE": "poracle"}
S3_LEGACY_POOLINGS = ("P0", "PLOCAL")
S3_C1B_POOLINGS = ("P0", "PLOCAL", "PORACLE")
S3_FEATURE_ASSET_KEYS = frozenset(
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
S3_FEATURE_METADATA_FIELDS = frozenset(
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
        "sidecar_metadata_path",
        "sidecar_metadata_sha256",
        "sidecar_keys_used",
        "trigger_gate_path",
        "trigger_gate_sha256",
        "trigger_status",
        "data_contract_provenance_sha256",
        "checkpoint_data_provenance_sha256",
        "stage_a_sentinel_sha256",
        "implementation_sha256",
        "device",
        "batch_size",
        "workers",
        "feature_tensor",
        "stage_module",
        "feature_channels",
        "response_projection",
        "representation_contract",
        "training_performed",
        "response_projection_called",
        "projector_called",
        "transition_called",
        "target_encoder_called",
        "ftv_head_called",
        "test_labels_used",
    }
)


def _ordered_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256(
        "\n".join(str(value) for value in values).encode("utf-8")
    ).hexdigest()


def _canonical_sha256(value: Any) -> str:
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
        raise ValueError(f"expected JSON object: {path}")
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
            json.dump(dict(payload), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.chmod(0o600)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def normalize_s3_pooling(pooling: str) -> str:
    value = str(pooling).strip().upper()
    aliases = {slug.upper(): name for name, slug in S3_POOLING_SLUGS.items()}
    value = aliases.get(value, value)
    if value not in S3_POOLING_SLUGS:
        raise ValueError(f"pooling is not preregistered for S3: {pooling!r}")
    return value


def s3_feature_asset_path(
    feature_root: str | Path,
    seed_base: int,
    arm: str,
    fold: int,
    pooling: str,
) -> Path:
    arm_name = str(arm).upper()
    pooling_name = normalize_s3_pooling(pooling)
    if arm_name not in ARMS or int(seed_base) not in (2026, 3026) or int(fold) not in range(5):
        raise ValueError("invalid formal S3 feature identity")
    allowed = S3_LEGACY_POOLINGS if arm_name.startswith("L") else S3_C1B_POOLINGS
    if pooling_name not in allowed:
        raise ValueError(f"{pooling_name} is unavailable for S3 arm {arm_name}")
    return (
        Path(feature_root).expanduser().resolve()
        / "s3"
        / f"seed_{int(seed_base)}"
        / arm_name
        / f"fold_{int(fold)}"
        / f"{S3_POOLING_SLUGS[pooling_name]}.private.npz"
    )


def s3_feature_metadata_path(feature_path: str | Path) -> Path:
    path = Path(feature_path)
    if not path.name.endswith(".private.npz"):
        raise ValueError("S3 feature asset must end in .private.npz")
    return path.with_suffix(".metadata.json")


def _as_weights(
    weights: torch.Tensor | np.ndarray,
    *,
    batch: int,
    visits: int,
    feature_shape: tuple[int, int, int],
    device: torch.device,
    shared: bool,
) -> torch.Tensor:
    value = torch.as_tensor(weights, device=device)
    if value.dtype != torch.float32:
        raise TypeError("formal S3 spatial weights must retain float32 dtype")
    if shared:
        if tuple(value.shape) != feature_shape:
            raise ValueError("shared S3 local weight shape differs from feature grid")
        return value.reshape(1, 1, *feature_shape)
    if tuple(value.shape) != (batch, visits, *feature_shape):
        raise ValueError("S3 per-visit weights must have shape [B,4,D,H,W]")
    return value.reshape(batch * visits, 1, *feature_shape)


def compute_s3_pooling_states(
    model: torch.nn.Module,
    image: torch.Tensor,
    *,
    arm: str,
    local_weights: torch.Tensor | np.ndarray,
    oracle_weights: torch.Tensor | np.ndarray | None = None,
    oracle_valid: torch.Tensor | np.ndarray | None = None,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Run the first three full residual blocks and pool raw 64-D S3 states."""

    arm_name = str(arm).upper()
    if arm_name not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    if not isinstance(image, torch.Tensor) or image.ndim != 6:
        raise ValueError("image must have shape [B,4,7,Z,Y,X]")
    if image.shape[0] <= 0 or image.shape[1:3] != (VISIT_COUNT, 7):
        raise ValueError("S3 image must contain exactly four DCE7 visits")
    if image.dtype != torch.float32 or not bool(torch.isfinite(image).all()):
        raise ValueError("S3 image must be finite float32")
    if model.training or model.encoder.training:
        raise ValueError("S3 online encoder must be in frozen eval mode")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("all checkpoint parameters must be frozen for S3 export")
    features = getattr(model.encoder, "features", None)
    if not isinstance(features, torch.nn.Sequential) or len(features) != 4:
        raise ValueError("S3 requires the frozen four-block encoder.features sequence")

    batch, visits = image.shape[:2]
    input_shape = tuple(int(value) for value in image.shape[-3:])
    feature_shape = expected_feature_shape(input_shape, stage="s3")
    expected_input = LEGACY_INPUT_SHAPE_ZYX if arm_name.startswith("L") else C1B_INPUT_SHAPE_ZYX
    expected_feature = LEGACY_S3_SHAPE_ZYX if arm_name.startswith("L") else C1B_S3_SHAPE_ZYX
    if input_shape == expected_input and feature_shape != expected_feature:
        raise AssertionError("formal input-to-S3 feature geometry drifted")
    if arm_name.startswith("L"):
        if oracle_weights is not None or oracle_valid is not None:
            raise ValueError("legacy S3 PORACLE is preregistered NA and must remain absent")
        local = _as_weights(
            local_weights,
            batch=batch,
            visits=visits,
            feature_shape=feature_shape,
            device=image.device,
            shared=False,
        )
    else:
        if oracle_weights is None or oracle_valid is None:
            raise ValueError("C1B S3 export requires authoritative oracle weights/validity")
        local = _as_weights(
            local_weights,
            batch=batch,
            visits=visits,
            feature_shape=feature_shape,
            device=image.device,
            shared=True,
        )

    versions = tuple(parameter._version for parameter in model.parameters())
    flat = image.reshape(batch * visits, *image.shape[2:])
    with torch.inference_mode():
        spatial = flat
        # Calling the complete module at index 2 includes both ``main`` and
        # ``skip``.  No hook into ``.main`` and no fourth residual block.
        for index in range(3):
            spatial = features[index](spatial)
    if not isinstance(spatial, torch.Tensor) or tuple(spatial.shape) != (
        batch * visits,
        S3_CHANNELS,
        *feature_shape,
    ):
        observed = getattr(spatial, "shape", type(spatial).__name__)
        raise ValueError(
            "encoder.features[2] full residual output must be [B*4,64,D,H,W], "
            f"got {observed}"
        )
    if spatial.dtype != torch.float32 or not bool(torch.isfinite(spatial).all()):
        raise ValueError("S3 spatial output must be finite float32")

    all_valid = torch.ones((batch, visits), dtype=torch.bool, device=image.device)
    output: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
        "P0": (
            global_average_pool(spatial).reshape(batch, visits, S3_CHANNELS),
            all_valid,
        ),
        "PLOCAL": (
            weighted_average_pool(spatial, local).reshape(
                batch, visits, S3_CHANNELS
            ),
            all_valid,
        ),
    }
    if not arm_name.startswith("L"):
        oracle = _as_weights(
            oracle_weights,
            batch=batch,
            visits=visits,
            feature_shape=feature_shape,
            device=image.device,
            shared=False,
        )
        validity = torch.as_tensor(oracle_valid, device=image.device)
        if validity.dtype != torch.bool or tuple(validity.shape) != (batch, visits):
            raise ValueError("S3 oracle_valid must be bool [B,4]")
        flat_valid = validity.reshape(-1)
        oracle_state = torch.zeros(
            (batch * visits, S3_CHANNELS), dtype=spatial.dtype, device=spatial.device
        )
        if bool(flat_valid.any()):
            oracle_state[flat_valid] = weighted_average_pool(
                spatial[flat_valid], oracle[flat_valid]
            )
        output["PORACLE"] = (
            oracle_state.reshape(batch, visits, S3_CHANNELS),
            validity,
        )
    if tuple(parameter._version for parameter in model.parameters()) != versions:
        raise RuntimeError("checkpoint parameters mutated during S3 export")
    expected_poolings = S3_LEGACY_POOLINGS if arm_name.startswith("L") else S3_C1B_POOLINGS
    if tuple(output) != expected_poolings:
        raise AssertionError("S3 pooling inventory drifted")
    for pooling, (state, state_valid) in output.items():
        if tuple(state.shape) != (batch, visits, S3_CHANNELS):
            raise AssertionError(f"{pooling} returned a wrong S3 state shape")
        if state_valid.dtype != torch.bool or tuple(state_valid.shape) != (batch, visits):
            raise AssertionError(f"{pooling} returned wrong S3 validity")
        if not bool(torch.isfinite(state).all()) or state.requires_grad:
            raise ValueError(f"{pooling} S3 state is nonfinite or retains gradients")
        if bool((state[~state_valid] != 0).any()):
            raise ValueError("invalid S3 states must remain explicit zero")
    return output


def _validate_checkpoint_data_contract(checkpoint: Mapping[str, Any], data: Any, splits: Any) -> str:
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
        raise ValueError("checkpoint data provenance digest is inconsistent")
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
        raise ValueError("checkpoint train-patient hash differs from locked fold")
    if checkpoint.get("val_patient_sha256") != canonical_sha256(sorted(splits.val)):
        raise ValueError("checkpoint validation-patient hash differs from locked fold")
    return digest


def _validate_checkpoint_selection(
    checkpoint: Mapping[str, Any], authorization: Any, *, arm: str, seed: int, fold: int
) -> None:
    if (
        str(checkpoint.get("arm", "")).upper() != arm
        or int(checkpoint.get("seed_base", -1)) != seed
        or int(checkpoint.get("fold", -1)) != fold
        or checkpoint.get("selected") is not True
        or checkpoint.get("test_data_used") is not False
    ):
        raise ValueError("selected checkpoint identity/selection contract drifted")
    selection_path = Path(str(checkpoint.get("selection_path", ""))).resolve()
    if not selection_path.is_file() or checkpoint.get("selection_sha256") != file_sha256(
        selection_path
    ):
        raise ValueError("checkpoint selection binding is absent or stale")
    selection = _read_json(selection_path)
    if checkpoint.get("selection") != selection or int(checkpoint.get("epoch", -1)) != int(
        selection.get("selected_epoch", -2)
    ):
        raise ValueError("checkpoint and selection record disagree")
    if checkpoint.get("stage_a_sentinel_sha256") != authorization.sha256:
        raise ValueError("checkpoint and Stage-A authorization disagree")


def _validate_reference_identity(
    path: Path, expected_ids: tuple[str, ...], expected_splits: tuple[str, ...]
) -> None:
    with np.load(path, allow_pickle=False) as archive:
        required = {"patient_id", "split", "response_state", "arm", "seed_base", "fold"}
        if set(archive.files) != required:
            raise ValueError("immutable P0 reference schema drifted")
        ids = tuple(np.asarray(archive["patient_id"]).astype(str).tolist())
        splits = tuple(np.asarray(archive["split"]).astype(str).tolist())
        state = np.asarray(archive["response_state"])
    if ids != expected_ids or splits != expected_splits:
        raise ValueError("S3 fold patient order differs from immutable P0 reference")
    if state.dtype != np.float32 or state.shape != (FORMAL_PATIENT_COUNT, VISIT_COUNT, 192):
        raise ValueError("immutable P0 reference state shape/dtype drifted")


def _sidecar_keys_for(arm: str, pooling: str) -> tuple[str, ...]:
    arm_name = str(arm).upper()
    pooling_name = normalize_s3_pooling(pooling)
    if pooling_name == "P0":
        return ()
    if pooling_name == "PLOCAL":
        return (
            "legacy_local_weight_s3" if arm_name.startswith("L") else "c1b_local_weight_s3",
        )
    if pooling_name == "PORACLE" and arm_name.startswith("N"):
        return ("c1b_oracle_weight_s3", "c1b_oracle_valid")
    raise ValueError(f"unavailable S3 sidecar contract: {arm_name}/{pooling_name}")


@dataclass(frozen=True)
class S3FeatureAsset:
    path: Path
    patient_id: tuple[str, ...]
    split: tuple[str, ...]
    state: np.ndarray
    state_valid: np.ndarray
    arm: str
    seed_base: int
    fold: int
    pooling: str


def _load_s3_feature_asset(path: str | Path) -> S3FeatureAsset:
    source = Path(path).resolve()
    if not source.is_file() or not source.name.endswith(".private.npz"):
        raise FileNotFoundError(f"S3 feature asset is absent: {source}")
    with np.load(source, allow_pickle=False) as archive:
        if set(archive.files) != S3_FEATURE_ASSET_KEYS:
            raise ValueError("S3 feature NPZ schema drifted")
        patient_ids = tuple(np.asarray(archive["patient_id"]).astype(str).tolist())
        splits = tuple(np.asarray(archive["split"]).astype(str).tolist())
        state = np.asarray(archive["state"])
        valid = np.asarray(archive["state_valid"])
        arm = str(np.asarray(archive["arm"]).item()).upper()
        seed = int(np.asarray(archive["seed_base"]).item())
        fold = int(np.asarray(archive["fold"]).item())
        pooling = normalize_s3_pooling(str(np.asarray(archive["pooling"]).item()))
    count = len(patient_ids)
    if count != FORMAL_PATIENT_COUNT or len(set(patient_ids)) != count:
        raise ValueError("S3 feature patient population is not 808 unique patients")
    if len(splits) != count or set(splits) != {"train", "val", "test"}:
        raise ValueError("S3 feature split labels are invalid")
    if state.dtype != np.float32 or state.shape != (count, VISIT_COUNT, S3_CHANNELS):
        raise ValueError("S3 state must be float32 [808,4,64]")
    if valid.dtype != np.bool_ or valid.shape != (count, VISIT_COUNT):
        raise ValueError("S3 state_valid must be bool [808,4]")
    if not np.isfinite(state).all() or np.any(state[~valid] != 0):
        raise ValueError("S3 state is nonfinite or invalid placeholders are nonzero")
    allowed = S3_LEGACY_POOLINGS if arm.startswith("L") else S3_C1B_POOLINGS
    if arm not in ARMS or seed not in (2026, 3026) or fold not in range(5) or pooling not in allowed:
        raise ValueError("S3 feature identity is outside the preregistered matrix")
    if pooling == "PORACLE":
        if arm.startswith("L") or int(valid.sum()) != FORMAL_ORACLE_VISIT_COUNT:
            raise ValueError("S3 PORACLE must be N-only with exact 1500 valid visits")
    elif not bool(valid.all()):
        raise ValueError("S3 P0/PLOCAL must be valid for all four visits")
    return S3FeatureAsset(source, patient_ids, splits, state, valid, arm, seed, fold, pooling)


def validate_s3_feature_export(
    path: str | Path,
    *,
    expected_arm: str | None = None,
    expected_seed_base: int | None = None,
    expected_fold: int | None = None,
    expected_pooling: str | None = None,
    verify_live_inputs: bool = True,
) -> tuple[S3FeatureAsset, dict[str, Any]]:
    asset = _load_s3_feature_asset(path)
    expected = {
        "arm": None if expected_arm is None else str(expected_arm).upper(),
        "seed_base": expected_seed_base,
        "fold": expected_fold,
        "pooling": None if expected_pooling is None else normalize_s3_pooling(expected_pooling),
    }
    for field, value in expected.items():
        if value is not None and getattr(asset, field) != value:
            raise ValueError(f"S3 feature {field} differs from expectation")
    metadata_path = s3_feature_metadata_path(asset.path)
    metadata = _read_json(metadata_path)
    if set(metadata) != S3_FEATURE_METADATA_FIELDS:
        raise ValueError("S3 feature metadata schema drifted")
    identity = {
        "schema_version": 1,
        "stage": "s3",
        "status": "COMPLETE",
        "arm": asset.arm,
        "seed_base": asset.seed_base,
        "fold": asset.fold,
        "pooling": asset.pooling,
        "pooling_slug": S3_POOLING_SLUGS[asset.pooling],
        "feature_path": str(asset.path),
        "feature_sha256": file_sha256(asset.path),
        "state_shape": list(asset.state.shape),
        "state_dtype": "float32",
        "state_valid_shape": list(asset.state_valid.shape),
        "state_valid_count": int(asset.state_valid.sum()),
        "patient_count": FORMAL_PATIENT_COUNT,
        "patient_order_sha256": _ordered_sha256(asset.patient_id),
        "split_order_sha256": _ordered_sha256(asset.split),
        "sidecar_keys_used": list(_sidecar_keys_for(asset.arm, asset.pooling)),
        "feature_tensor": "model.encoder.features[2]_full_residual_output",
        "stage_module": "encoder.features[2]",
        "feature_channels": S3_CHANNELS,
        "response_projection": "not_called_raw_64d",
        "representation_contract": S3_REPRESENTATION_CONTRACT,
        "training_performed": False,
        "response_projection_called": False,
        "projector_called": False,
        "transition_called": False,
        "target_encoder_called": False,
        "ftv_head_called": False,
        "test_labels_used": False,
    }
    for field, value in identity.items():
        if metadata.get(field) != value:
            raise ValueError(f"S3 feature metadata drifted at {field}")
    if verify_live_inputs:
        lock = verify_preregistration()
        key = cell_key(asset.seed_base, asset.arm, asset.fold)
        checkpoint = Path(str(metadata["checkpoint_path"])).resolve()
        reference = Path(str(metadata["reference_feature_path"])).resolve()
        reference_metadata = Path(str(metadata["reference_feature_metadata_path"])).resolve()
        sidecar = Path(str(metadata["sidecar_path"])).resolve()
        sidecar_metadata = Path(str(metadata["sidecar_metadata_path"])).resolve()
        trigger = Path(str(metadata["trigger_gate_path"])).resolve()
        expected_paths = {
            "checkpoint_path": formal_checkpoint_path(
                asset.seed_base, asset.arm, asset.fold
            ).resolve(),
            "reference_feature_path": reference_feature_path(
                asset.seed_base, asset.arm, asset.fold
            ).resolve(),
            "reference_feature_metadata_path": reference_feature_metadata_path(
                asset.seed_base, asset.arm, asset.fold
            ).resolve(),
            "sidecar_path": (
                EXPERIMENT_ROOT / "manifests" / "audit_sidecars_s3.private.npz"
            ).resolve(),
            "sidecar_metadata_path": (
                EXPERIMENT_ROOT
                / "manifests"
                / "audit_sidecars_s3.private.metadata.json"
            ).resolve(),
            "trigger_gate_path": (
                EXPERIMENT_ROOT / "metrics" / "s3_trigger_authorization.json"
            ).resolve(),
        }
        observed_paths = {
            "checkpoint_path": checkpoint,
            "reference_feature_path": reference,
            "reference_feature_metadata_path": reference_metadata,
            "sidecar_path": sidecar,
            "sidecar_metadata_path": sidecar_metadata,
            "trigger_gate_path": trigger,
        }
        for field, expected_path in expected_paths.items():
            if observed_paths[field] != expected_path:
                raise ValueError(f"S3 feature canonical path drifted at {field}")
        checks = {
            "preregistration_lock_sha256": file_sha256(
                EXPERIMENT_ROOT / "PREREGISTRATION_LOCK.json"
            ),
            "plan_sha256": lock["plan_sha256"],
            "config_sha256": lock["config_sha256"],
            "checkpoint_sha256": file_sha256(checkpoint),
            "reference_feature_sha256": file_sha256(reference),
            "reference_feature_metadata_sha256": file_sha256(reference_metadata),
            "sidecar_sha256": file_sha256(sidecar),
            "sidecar_metadata_sha256": file_sha256(sidecar_metadata),
            "trigger_gate_sha256": file_sha256(trigger),
        }
        for field, value in checks.items():
            if metadata.get(field) != value:
                raise ValueError(f"live S3 feature provenance drifted at {field}")
        if metadata.get("checkpoint_lock_key") != key:
            raise ValueError("S3 checkpoint lock key drifted")
        if lock["selected_checkpoints"][key]["sha256"] != metadata["checkpoint_sha256"]:
            raise ValueError("S3 checkpoint no longer matches preregistration")
        if lock["formal_p0_references"][key]["feature_sha256"] != metadata[
            "reference_feature_sha256"
        ]:
            raise ValueError("S3 patient-order reference no longer matches preregistration")
        if lock["formal_p0_references"][key]["feature_metadata_sha256"] != metadata[
            "reference_feature_metadata_sha256"
        ]:
            raise ValueError("S3 reference metadata no longer matches preregistration")
        trigger_gate = require_s3_trigger_authorization(trigger, verify_live=True)
        if metadata.get("trigger_status") != trigger_gate["status"]:
            raise ValueError("S3 feature trigger status differs from authorization gate")
        implementation = metadata.get("implementation_sha256")
        expected_implementation = {
            "s3_exporter.py": Path(__file__),
            "s3_sidecars.py": Path(__file__).with_name("s3_sidecars.py"),
            "s3_trigger.py": Path(__file__).with_name("s3_trigger.py"),
            "pooling.py": Path(__file__).with_name("pooling.py"),
            "runtime.py": Path(__file__).with_name("runtime.py"),
            "contracts.py": Path(__file__).with_name("contracts.py"),
        }
        if not isinstance(implementation, Mapping) or set(implementation) != set(
            expected_implementation
        ):
            raise ValueError("S3 feature implementation hash map drifted")
        for name, source in expected_implementation.items():
            if implementation.get(name) != file_sha256(source):
                raise ValueError(f"S3 feature implementation changed at {name}")
    return asset, metadata


def _write_feature(
    path: Path,
    *,
    patient_ids: tuple[str, ...],
    splits: tuple[str, ...],
    state: np.ndarray,
    state_valid: np.ndarray,
    arm: str,
    seed: int,
    fold: int,
    pooling: str,
    metadata_base: Mapping[str, Any],
) -> dict[str, Any]:
    arrays = {
        "patient_id": np.asarray(patient_ids, dtype=str),
        "split": np.asarray(splits, dtype=str),
        "state": np.asarray(state, dtype=np.float32),
        "state_valid": np.asarray(state_valid, dtype=bool),
        "arm": np.asarray(arm),
        "seed_base": np.asarray(seed, dtype=np.int64),
        "fold": np.asarray(fold, dtype=np.int64),
        "pooling": np.asarray(pooling),
    }
    _atomic_npz(path, arrays)
    metadata = {
        **metadata_base,
        "schema_version": 1,
        "stage": "s3",
        "status": "COMPLETE",
        "arm": arm,
        "seed_base": seed,
        "fold": fold,
        "pooling": pooling,
        "pooling_slug": S3_POOLING_SLUGS[pooling],
        "feature_path": str(path),
        "feature_sha256": file_sha256(path),
        "state_shape": list(state.shape),
        "state_dtype": "float32",
        "state_valid_shape": list(state_valid.shape),
        "state_valid_count": int(state_valid.sum()),
        "patient_count": len(patient_ids),
        "patient_order_sha256": _ordered_sha256(patient_ids),
        "split_order_sha256": _ordered_sha256(splits),
        "sidecar_keys_used": list(_sidecar_keys_for(arm, pooling)),
    }
    _atomic_json(s3_feature_metadata_path(path), metadata)
    return metadata


@torch.no_grad()
def export_s3_feature_cell(
    *,
    checkpoint_path: str | Path,
    arm: str,
    seed_base: int,
    fold: int,
    data: Any,
    authorization: Any,
    trigger_gate_path: str | Path,
    sidecar_path: str | Path,
    feature_root: str | Path,
    device: torch.device,
    batch_size: int = 4,
    workers: int = 2,
) -> dict[str, dict[str, Any]]:
    """Export all and only registered S3 poolings for one frozen checkpoint."""

    gate = require_s3_trigger_authorization(trigger_gate_path, verify_live=True)
    lock = verify_preregistration()
    arm_name = str(arm).upper()
    seed = int(seed_base)
    fold_index = int(fold)
    if arm_name not in ARMS or seed not in (2026, 3026) or fold_index not in range(5):
        raise ValueError("invalid formal S3 cell identity")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal S3 feature export requires CUDA")
    if int(batch_size) != 4 or int(workers) != 2:
        raise ValueError("formal S3 export is frozen at batch_size=4/workers=2")
    if Path(feature_root).resolve() != (EXPERIMENT_ROOT / "features").resolve():
        raise ValueError("formal S3 outputs must remain in this audit's features root")
    expected_sidecar = (
        EXPERIMENT_ROOT / "manifests" / "audit_sidecars_s3.private.npz"
    ).resolve()
    if Path(sidecar_path).resolve() != expected_sidecar:
        raise ValueError("formal S3 export requires the canonical S3 sidecar")
    expected_trigger = (
        EXPERIMENT_ROOT / "metrics" / "s3_trigger_authorization.json"
    ).resolve()
    if Path(trigger_gate_path).resolve() != expected_trigger:
        raise ValueError("formal S3 export requires the canonical trigger gate")
    source_checkpoint = Path(checkpoint_path).resolve()
    expected_checkpoint = formal_checkpoint_path(seed, arm_name, fold_index).resolve()
    if source_checkpoint != expected_checkpoint:
        raise ValueError("S3 checkpoint path is not the exact formal selected cell")
    key = cell_key(seed, arm_name, fold_index)
    checkpoint_lock = lock["selected_checkpoints"][key]
    checkpoint_sha = file_sha256(source_checkpoint)
    if checkpoint_sha != checkpoint_lock["sha256"] or relative(source_checkpoint) != checkpoint_lock["path"]:
        raise ValueError("S3 checkpoint path/hash differs from preregistration")

    from c1b_stage_b.data import StageBDataset, arm_cache, make_splits

    splits = make_splits(data.folds, fold_index, data.train_only_ids)
    patient_ids = tuple(splits.train_primary + splits.val + splits.test)
    split_labels = tuple(
        ("train",) * len(splits.train_primary)
        + ("val",) * len(splits.val)
        + ("test",) * len(splits.test)
    )
    if len(patient_ids) != FORMAL_PATIENT_COUNT or len(set(patient_ids)) != FORMAL_PATIENT_COUNT:
        raise ValueError("formal S3 export must contain exactly 808 unique patients")
    reference = reference_feature_path(seed, arm_name, fold_index).resolve()
    reference_metadata = reference_feature_metadata_path(seed, arm_name, fold_index).resolve()
    reference_lock = lock["formal_p0_references"][key]
    if (
        file_sha256(reference) != reference_lock["feature_sha256"]
        or file_sha256(reference_metadata) != reference_lock["feature_metadata_sha256"]
        or _ordered_sha256(patient_ids) != reference_lock["patient_order_sha256"]
    ):
        raise ValueError("S3 immutable P0 patient-order reference drifted")
    _validate_reference_identity(reference, patient_ids, split_labels)
    sidecars: LoadedS3Sidecars = load_s3_sidecars(
        expected_sidecar, patient_ids, verify_live=True
    )
    allowed = S3_LEGACY_POOLINGS if arm_name.startswith("L") else S3_C1B_POOLINGS
    output_paths = {
        pooling: s3_feature_asset_path(
            feature_root, seed, arm_name, fold_index, pooling
        )
        for pooling in allowed
    }
    collisions = [
        path
        for path in output_paths.values()
        if path.exists() or s3_feature_metadata_path(path).exists()
    ]
    if collisions:
        raise FileExistsError(f"refusing to overwrite S3 feature output: {collisions[0]}")

    model, checkpoint = load_selected_model(source_checkpoint, device)
    _validate_checkpoint_selection(
        checkpoint, authorization, arm=arm_name, seed=seed, fold=fold_index
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
    state_parts: dict[str, list[np.ndarray]] = {pooling: [] for pooling in allowed}
    valid_parts: dict[str, list[np.ndarray]] = {pooling: [] for pooling in allowed}
    observed_ids: list[str] = []
    offset = 0
    for batch in loader:
        batch_ids = tuple(str(value) for value in batch["patient_id"])
        if batch_ids != patient_ids[offset : offset + len(batch_ids)]:
            raise AssertionError("S3 DataLoader changed formal patient order")
        image = batch["image"].to(device, non_blocking=True)
        indices = slice(offset, offset + len(batch_ids))
        if arm_name.startswith("L"):
            states = compute_s3_pooling_states(
                model,
                image,
                arm=arm_name,
                local_weights=sidecars.legacy_local_weight_s3[indices],
            )
        else:
            states = compute_s3_pooling_states(
                model,
                image,
                arm=arm_name,
                local_weights=sidecars.c1b_local_weight_s3,
                oracle_weights=sidecars.c1b_oracle_weight_s3[indices],
                oracle_valid=sidecars.c1b_oracle_valid[indices],
            )
        if tuple(states) != allowed:
            raise AssertionError("S3 batch pooling inventory drifted")
        for pooling, (state, state_valid) in states.items():
            state_parts[pooling].append(state.detach().float().cpu().numpy())
            valid_parts[pooling].append(state_valid.detach().cpu().numpy())
        observed_ids.extend(batch_ids)
        offset += len(batch_ids)
    if tuple(observed_ids) != patient_ids or offset != FORMAL_PATIENT_COUNT:
        raise AssertionError("S3 export did not cover exact formal patient order")

    complete_states = {
        pooling: np.concatenate(parts).astype(np.float32, copy=False)
        for pooling, parts in state_parts.items()
    }
    complete_valid = {
        pooling: np.concatenate(parts).astype(bool, copy=False)
        for pooling, parts in valid_parts.items()
    }
    for pooling in allowed:
        if complete_states[pooling].shape != (FORMAL_PATIENT_COUNT, VISIT_COUNT, S3_CHANNELS):
            raise AssertionError("complete S3 state shape drifted")
        if pooling == "PORACLE":
            if int(complete_valid[pooling].sum()) != FORMAL_ORACLE_VISIT_COUNT:
                raise ValueError("complete S3 oracle population is not 1500 visits")
        elif not bool(complete_valid[pooling].all()):
            raise ValueError("complete S3 P0/PLOCAL contains invalid rows")

    lock_path = EXPERIMENT_ROOT / "PREREGISTRATION_LOCK.json"
    trigger_path = Path(trigger_gate_path).resolve()
    metadata_base = {
        "checkpoint_path": str(source_checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_lock_key": key,
        "reference_feature_path": str(reference),
        "reference_feature_sha256": reference_lock["feature_sha256"],
        "reference_feature_metadata_path": str(reference_metadata),
        "reference_feature_metadata_sha256": reference_lock["feature_metadata_sha256"],
        "preregistration_lock_sha256": file_sha256(lock_path),
        "plan_sha256": lock["plan_sha256"],
        "config_sha256": lock["config_sha256"],
        "sidecar_path": str(sidecars.path),
        "sidecar_sha256": sidecars.sha256,
        "sidecar_metadata_path": str(sidecars.metadata_path),
        "sidecar_metadata_sha256": sidecars.metadata_sha256,
        "trigger_gate_path": str(trigger_path),
        "trigger_gate_sha256": file_sha256(trigger_path),
        "trigger_status": gate["status"],
        "data_contract_provenance_sha256": _canonical_sha256(data.provenance),
        "checkpoint_data_provenance_sha256": checkpoint_data_sha,
        "stage_a_sentinel_sha256": authorization.sha256,
        "implementation_sha256": {
            "s3_exporter.py": file_sha256(Path(__file__)),
            "s3_sidecars.py": file_sha256(Path(__file__).with_name("s3_sidecars.py")),
            "s3_trigger.py": file_sha256(Path(__file__).with_name("s3_trigger.py")),
            "pooling.py": file_sha256(Path(__file__).with_name("pooling.py")),
            "runtime.py": file_sha256(Path(__file__).with_name("runtime.py")),
            "contracts.py": file_sha256(Path(__file__).with_name("contracts.py")),
        },
        "device": str(device),
        "batch_size": 4,
        "workers": 2,
        "feature_tensor": "model.encoder.features[2]_full_residual_output",
        "stage_module": "encoder.features[2]",
        "feature_channels": S3_CHANNELS,
        "response_projection": "not_called_raw_64d",
        "representation_contract": S3_REPRESENTATION_CONTRACT,
        "training_performed": False,
        "response_projection_called": False,
        "projector_called": False,
        "transition_called": False,
        "target_encoder_called": False,
        "ftv_head_called": False,
        "test_labels_used": False,
    }
    metadata_by_pooling: dict[str, dict[str, Any]] = {}
    for pooling in allowed:
        metadata_by_pooling[pooling] = _write_feature(
            output_paths[pooling],
            patient_ids=patient_ids,
            splits=split_labels,
            state=complete_states[pooling],
            state_valid=complete_valid[pooling],
            arm=arm_name,
            seed=seed,
            fold=fold_index,
            pooling=pooling,
            metadata_base=metadata_base,
        )
    immutable_hashes = {
        source_checkpoint: checkpoint_sha,
        reference: reference_lock["feature_sha256"],
        reference_metadata: reference_lock["feature_metadata_sha256"],
        sidecars.path: sidecars.sha256,
        sidecars.metadata_path: sidecars.metadata_sha256,
        trigger_path: file_sha256(trigger_path),
    }
    for immutable, expected_hash in immutable_hashes.items():
        if file_sha256(immutable) != expected_hash:
            raise RuntimeError(f"immutable S3 input changed during export: {immutable}")
    for pooling, path in output_paths.items():
        validate_s3_feature_export(
            path,
            expected_arm=arm_name,
            expected_seed_base=seed,
            expected_fold=fold_index,
            expected_pooling=pooling,
            verify_live_inputs=True,
        )
    return metadata_by_pooling


__all__ = [
    "S3_C1B_POOLINGS",
    "S3_CHANNELS",
    "S3_FEATURE_ASSET_KEYS",
    "S3_FEATURE_METADATA_FIELDS",
    "S3_LEGACY_POOLINGS",
    "S3_POOLING_SLUGS",
    "S3_REPRESENTATION_CONTRACT",
    "S3FeatureAsset",
    "compute_s3_pooling_states",
    "export_s3_feature_cell",
    "normalize_s3_pooling",
    "s3_feature_asset_path",
    "s3_feature_metadata_path",
    "validate_s3_feature_export",
]

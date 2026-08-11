"""Resumable, label-blind extraction of frozen foundation representations."""

from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Iterator, Mapping, Sequence

import numpy as np

# Must be set before CUDA creates its first cuBLAS handle.  Formal extraction
# refuses nondeterministic fallback kernels.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from .models import (
    DINO_EMBED_DIM,
    MEDICALNET_EMBED_DIM,
    DINOEncoder,
    MedicalNetEncoder,
    load_dino_encoder,
    load_medicalnet_encoder,
    model_audit,
)
from .provenance import (
    atomic_json,
    atomic_private_npz,
    canonical_json_sha256,
    environment_snapshot,
    ordered_text_sha256,
    secure_directory,
    verify_file_lock,
)
from .spatial import (
    C1B_SPACING_ZYX_MM,
    MEDICALNET_FEATURE_SHAPE,
    dino_slice_stack,
    medicalnet_volume_batch,
    spatial_contract,
)
from .upstream import (
    EXPECTED_CACHE_MANIFEST_SHA256,
    EXPECTED_FOLD_SHA256,
    EXPERIMENT_ROOT,
    fixed_physical_local_weights,
    global_average_pool,
    load_dce7,
    read_cache_manifest,
    read_fold_manifest,
    upstream_contract,
    weighted_average_pool,
)


SPATIAL_AXES = ("GLOBAL", "LOCAL")
VISITS = ("T0", "T1", "T2", "T3")


def _autocast(device: torch.device, precision: str) -> contextlib.AbstractContextManager:
    normalized = str(precision).lower()
    if normalized == "fp32":
        return contextlib.nullcontext()
    if normalized == "bf16" and device.type == "cuda":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("formal bf16 extraction requested but GPU lacks bf16")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    raise ValueError("precision must be fp32, or bf16 on CUDA")


def _model_signature(
    model_name: str, precision: str, audit: Mapping[str, object]
) -> dict[str, object]:
    payload = {
        "schema_version": 1,
        "model_name": model_name,
        "precision": precision,
        "audit": dict(audit),
        "spatial_contract": spatial_contract(),
        "upstream_contract": upstream_contract(),
        "visit_order": list(VISITS),
        "spatial_axis_order": list(SPATIAL_AXES),
        "aggregation": {
            "dino": "mean_32_slice_features",
            "medicalnet": "per_channel_pool_then_DCE7_order_concat",
        },
    }
    payload["signature_sha256"] = canonical_json_sha256(payload)
    return payload


def _load_primary_population(
    fold_manifest: Path,
    cache_manifest: Path,
) -> tuple[tuple[str, ...], Mapping[str, object]]:
    folds = read_fold_manifest(fold_manifest, EXPECTED_FOLD_SHA256)
    fold_zero = folds.loc[folds["fold"].eq(0)]
    patient_ids = tuple(fold_zero["patient_id"].astype(str))
    if len(patient_ids) != 808 or len(set(patient_ids)) != 808:
        raise ValueError("formal primary population must contain exactly 808 unique patients")
    if set(folds["patient_id"].astype(str)) != set(patient_ids):
        raise ValueError("fold manifest patient membership drifted")
    cache = read_cache_manifest(
        cache_manifest,
        EXPECTED_CACHE_MANIFEST_SHA256,
        expected_input_kind="c1b",
        verify_cache_files=False,
    )
    missing = sorted(set(patient_ids).difference(cache))
    if missing:
        raise FileNotFoundError(f"C1B cache misses formal patients: {missing[:5]}")
    return patient_ids, cache


def _dino_patient(
    image: np.ndarray,
    model: DINOEncoder,
    device: torch.device,
    precision: str,
    batch_size: int,
) -> np.ndarray:
    if int(batch_size) <= 0:
        raise ValueError("DINO batch size must be positive")
    output = np.empty((4, 2, DINO_EMBED_DIM), dtype=np.float32)
    tensor = torch.from_numpy(image)
    with torch.inference_mode():
        for visit in range(4):
            visit_tensor = tensor[visit].to(device=device, non_blocking=False)
            for axis_index, axis in enumerate(SPATIAL_AXES):
                slices = dino_slice_stack(visit_tensor, axis)
                encoded: list[torch.Tensor] = []
                for start in range(0, len(slices), int(batch_size)):
                    batch = slices[start : start + int(batch_size)]
                    with _autocast(device, precision):
                        encoded.append(model(batch).float())
                pooled = torch.cat(encoded, dim=0).mean(dim=0)
                if tuple(pooled.shape) != (DINO_EMBED_DIM,):
                    raise AssertionError("DINO slice pooling shape failed")
                output[visit, axis_index] = pooled.cpu().numpy()
            del visit_tensor
    if not np.isfinite(output).all():
        raise FloatingPointError("DINO patient representation contains NaN/Inf")
    return output


def _medicalnet_patient(
    image: np.ndarray,
    model: MedicalNetEncoder,
    device: torch.device,
    precision: str,
) -> np.ndarray:
    output = np.empty((4, 2, MEDICALNET_EMBED_DIM), dtype=np.float32)
    tensor = torch.from_numpy(image)
    local_weights = fixed_physical_local_weights(
        (112, 176, 160),
        MEDICALNET_FEATURE_SHAPE,
        tuple(reversed(C1B_SPACING_ZYX_MM)),
        stage="final",
        device=device,
        dtype=torch.float32,
    )
    with torch.inference_mode():
        for visit in range(4):
            batch = medicalnet_volume_batch(tensor[visit]).to(
                device=device, non_blocking=False
            )
            with _autocast(device, precision):
                spatial = model.forward_spatial(batch)
            if tuple(spatial.shape) != (7, 2048, *MEDICALNET_FEATURE_SHAPE):
                raise AssertionError(
                    f"MedicalNet native feature map drifted: {tuple(spatial.shape)}"
                )
            global_features = global_average_pool(spatial.float())
            local_features = weighted_average_pool(spatial.float(), local_weights)
            output[visit, 0] = global_features.reshape(-1).cpu().numpy()
            output[visit, 1] = local_features.reshape(-1).cpu().numpy()
            del batch, spatial, global_features, local_features
    if not np.isfinite(output).all():
        raise FloatingPointError("MedicalNet patient representation contains NaN/Inf")
    return output


def _shard_name(patient_id: str) -> str:
    return hashlib.sha256(patient_id.encode("utf-8")).hexdigest() + ".private.npz"


def _validate_shard(
    path: Path,
    *,
    patient_id: str,
    signature_sha256: str,
    representation_dim: int,
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {
            "patient_id",
            "representation",
            "signature_sha256",
        }:
            raise ValueError(f"private shard schema drifted: {path.name}")
        observed_patient = str(np.asarray(archive["patient_id"]).item())
        observed_signature = str(np.asarray(archive["signature_sha256"]).item())
        representation = np.asarray(archive["representation"], dtype=np.float32)
    if observed_patient != patient_id:
        raise ValueError("private shard patient identity mismatch")
    if observed_signature != signature_sha256:
        raise ValueError("private shard belongs to a different frozen extraction contract")
    if representation.shape != (4, 2, representation_dim):
        raise ValueError("private shard representation shape mismatch")
    if not np.isfinite(representation).all():
        raise FloatingPointError("private shard contains NaN/Inf")
    if path.stat().st_mode & 0o077:
        raise PermissionError(f"private shard permissions are too broad: {path.name}")
    return representation


def extract_features(
    *,
    model_name: str,
    checkpoint: Path,
    fold_manifest: Path,
    cache_manifest: Path,
    output_root: Path,
    device_name: str,
    precision: str,
    dino_batch_size: int = 64,
    limit: int | None = None,
) -> Path | None:
    """Extract resumable shards, and combine only a complete 808-patient run."""

    verify_file_lock(
        EXPERIMENT_ROOT / "configs" / "MODEL_INPUT_LOCK.json", EXPERIMENT_ROOT
    )
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal extraction requires an available CUDA device")
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)

    normalized_name = str(model_name).lower()
    if normalized_name == "dino_vitb16_imagenet1k":
        model = load_dino_encoder(checkpoint).to(device)
        representation_dim = DINO_EMBED_DIM
        extractor = lambda image: _dino_patient(  # noqa: E731
            image, model, device, precision, dino_batch_size
        )
    elif normalized_name == "medicalnet_resnet50_3dseg8":
        model = load_medicalnet_encoder(checkpoint).to(device)
        representation_dim = MEDICALNET_EMBED_DIM
        extractor = lambda image: _medicalnet_patient(  # noqa: E731
            image, model, device, precision
        )
    else:
        raise ValueError(f"unknown formal model: {model_name}")
    audit = model_audit(model)
    signature = _model_signature(normalized_name, precision, audit)
    signature_sha256 = str(signature["signature_sha256"])

    patient_ids, cache = _load_primary_population(fold_manifest, cache_manifest)
    selected = patient_ids
    if limit is not None:
        parsed_limit = int(limit)
        if parsed_limit <= 0 or parsed_limit > len(patient_ids):
            raise ValueError("limit must lie in 1..808")
        selected = patient_ids[:parsed_limit]
    run_kind = "formal" if len(selected) == 808 else f"smoke_{len(selected)}"
    run_root = secure_directory(output_root / run_kind / normalized_name)
    shard_root = secure_directory(run_root / "shards")
    atomic_json(run_root / "contract.private.json", signature, private=True)

    started = time.monotonic()
    for index, patient_id in enumerate(selected, start=1):
        shard = shard_root / _shard_name(patient_id)
        if shard.exists():
            _validate_shard(
                shard,
                patient_id=patient_id,
                signature_sha256=signature_sha256,
                representation_dim=representation_dim,
            )
            action = "resume"
        else:
            image = load_dce7(cache[patient_id])
            representation = extractor(image)
            atomic_private_npz(
                shard,
                {
                    "patient_id": np.asarray(patient_id),
                    "representation": representation,
                    "signature_sha256": np.asarray(signature_sha256),
                },
            )
            del image, representation
            gc.collect()
            action = "extract"
        elapsed = time.monotonic() - started
        print(
            f"[{normalized_name}] {index}/{len(selected)} {action}; "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    if len(selected) != 808:
        return None

    representations = np.stack(
        [
            _validate_shard(
                shard_root / _shard_name(patient_id),
                patient_id=patient_id,
                signature_sha256=signature_sha256,
                representation_dim=representation_dim,
            )
            for patient_id in patient_ids
        ],
        axis=0,
    )
    if representations.shape != (808, 4, 2, representation_dim):
        raise AssertionError("combined foundation feature shape failed")
    destination = run_root / "frozen_features.private.npz"
    if destination.exists():
        raise FileExistsError(f"formal feature file already exists: {destination}")
    atomic_private_npz(
        destination,
        {
            "patient_id": np.asarray(patient_ids),
            "representation": representations,
            "spatial_axis": np.asarray(SPATIAL_AXES),
            "visits": np.asarray(VISITS),
            "model_name": np.asarray(normalized_name),
            "checkpoint_sha256": np.asarray(str(audit["checkpoint_sha256"])),
            "extraction_signature_sha256": np.asarray(signature_sha256),
            "canonical_patient_order_sha256": np.asarray(
                ordered_text_sha256(patient_ids)
            ),
        },
    )
    private_metadata = {
        "schema_version": 1,
        "model": normalized_name,
        "patient_count": len(patient_ids),
        "representation_shape": list(representations.shape),
        "canonical_patient_order_sha256": ordered_text_sha256(patient_ids),
        "extraction_signature_sha256": signature_sha256,
        "environment": environment_snapshot(),
        "elapsed_seconds": time.monotonic() - started,
        "feature_file": str(destination.resolve()),
    }
    atomic_json(run_root / "execution.private.json", private_metadata, private=True)
    return destination


__all__ = ["SPATIAL_AXES", "VISITS", "extract_features"]

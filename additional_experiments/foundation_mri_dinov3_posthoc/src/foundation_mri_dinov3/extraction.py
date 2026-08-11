"""Resumable, label-blind DINOv3 extraction for the frozen 808-patient cohort."""

from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from typing import Mapping

import numpy as np

# This must be set before CUDA creates its first cuBLAS handle.  Formal runs
# additionally reject CPU execution and nondeterministic fallback kernels.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from .model import (
    DINOV3_ARTIFACTS,
    DINOV3_EMBED_DIM,
    DINOV3_REVISION,
    DINOv3Encoder,
    MODEL_NAME,
    load_dinov3_encoder,
    model_audit,
)
from .paths import BASE_SOURCE_ROOT


# Reuse the published experiment's audited image/cache/spatial implementation
# byte-for-byte.  The post-hoc lock independently binds those source files.
_base_source = str(BASE_SOURCE_ROOT.resolve())
if _base_source not in sys.path:
    sys.path.insert(0, _base_source)

from foundation_mri.provenance import (  # noqa: E402
    atomic_json,
    atomic_private_npz,
    canonical_json_sha256,
    ordered_text_sha256,
    secure_directory,
)
from foundation_mri.spatial import dino_slice_stack, spatial_contract  # noqa: E402
from foundation_mri.upstream import (  # noqa: E402
    EXPECTED_CACHE_MANIFEST_SHA256,
    EXPECTED_FOLD_SHA256,
    load_dce7,
    read_cache_manifest,
    read_fold_manifest,
    upstream_contract,
)


SPATIAL_AXES = ("GLOBAL", "LOCAL")
VISITS = ("T0", "T1", "T2", "T3")
PRIMARY_COHORT_SIZE = 808
SNAPSHOT_ENV = "DINOV3_SNAPSHOT_DIR"


def _autocast(
    device: torch.device, precision: str
) -> contextlib.AbstractContextManager:
    normalized = str(precision).strip().lower()
    if normalized == "fp32":
        return contextlib.nullcontext()
    if normalized == "bf16" and device.type == "cuda":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "bf16 extraction requested but the GPU lacks bf16 support"
            )
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    raise ValueError("precision must be fp32, or bf16 on CUDA")


def resolve_snapshot_dir(snapshot_dir: str | Path | None, *, formal: bool) -> Path:
    """Resolve a local snapshot without ever treating a repo ID as a fallback."""

    environment_value = os.environ.get(SNAPSHOT_ENV, "").strip()
    if formal:
        if snapshot_dir is not None:
            raise ValueError(
                "formal extraction is env-only; omit --snapshot and set "
                f"{SNAPSHOT_ENV}"
            )
        if not environment_value:
            raise RuntimeError(
                f"formal extraction requires local snapshot environment variable {SNAPSHOT_ENV}"
            )
        return Path(environment_value).expanduser().resolve()
    candidate = snapshot_dir if snapshot_dir is not None else environment_value
    if not candidate:
        raise RuntimeError(f"smoke extraction requires --snapshot or {SNAPSHOT_ENV}")
    return Path(candidate).expanduser().resolve()


def _verify_formal_model_input_lock(snapshot_dir: Path) -> Mapping[str, object]:
    """Delay the import so purely synthetic smoke tests need no formal lock."""

    try:
        from .locking import verify_model_input_lock
    except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - build gate
        raise RuntimeError("formal MODEL_INPUT_LOCK verifier is unavailable") from exc
    payload = verify_model_input_lock(snapshot_dir)
    if not isinstance(payload, Mapping):
        raise TypeError("MODEL_INPUT_LOCK verifier must return a mapping")
    return payload


def _model_signature(
    precision: str,
    audit: Mapping[str, object],
    *,
    model_input_lock_sha256: str | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "experiment_kind": "user_requested_posthoc_sensitivity",
        "model_name": MODEL_NAME,
        "precision": str(precision).lower(),
        "audit": dict(audit),
        "spatial_contract": spatial_contract(),
        "upstream_contract": upstream_contract(),
        "visit_order": list(VISITS),
        "spatial_axis_order": list(SPATIAL_AXES),
        "aggregation": {
            "per_slice": (
                "final_cls_concat_mean_final_patch_tokens; "
                "four_register_tokens_excluded"
            ),
            "per_visit_axis": "arithmetic_mean_of_32_uniform_axial_slices",
        },
        "outcome_fields_consumed": [],
        "model_input_lock_sha256": model_input_lock_sha256,
    }
    payload["signature_sha256"] = canonical_json_sha256(payload)
    return payload


def _load_primary_population(
    fold_manifest: Path,
    cache_manifest: Path,
) -> tuple[tuple[str, ...], Mapping[str, object]]:
    """Load only canonical membership/folds and C1B image-cache locations."""

    folds = read_fold_manifest(fold_manifest, EXPECTED_FOLD_SHA256)
    fold_zero = folds.loc[folds["fold"].eq(0)]
    patient_ids = tuple(fold_zero["patient_id"].astype(str))
    if len(patient_ids) != PRIMARY_COHORT_SIZE:
        raise ValueError("formal primary population must contain exactly 808 patients")
    if len(set(patient_ids)) != PRIMARY_COHORT_SIZE:
        raise ValueError("formal primary population contains duplicate patients")
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


def _dinov3_patient(
    image: np.ndarray,
    model: DINOv3Encoder,
    device: torch.device,
    precision: str,
    batch_size: int,
) -> np.ndarray:
    """Encode four visits x GLOBAL/LOCAL without accepting any label input."""

    if int(batch_size) <= 0:
        raise ValueError("DINOv3 batch size must be positive")
    array = np.asarray(image)
    if array.ndim != 5 or array.shape[0] != len(VISITS):
        raise ValueError("C1B patient image must be [4,C,Z,Y,X]")
    if array.dtype != np.dtype(np.float32):
        raise TypeError("C1B patient image must be float32")
    if not np.isfinite(array).all():
        raise FloatingPointError("C1B patient image contains NaN/Inf")

    output = np.empty((4, 2, DINOV3_EMBED_DIM), dtype=np.float32)
    tensor = torch.from_numpy(array)
    with torch.inference_mode():
        for visit_index in range(len(VISITS)):
            visit = tensor[visit_index].to(device=device, non_blocking=False)
            for axis_index, axis in enumerate(SPATIAL_AXES):
                slices = dino_slice_stack(visit, axis)
                encoded: list[torch.Tensor] = []
                for start in range(0, len(slices), int(batch_size)):
                    batch = slices[start : start + int(batch_size)]
                    with _autocast(device, precision):
                        features = model(batch)
                    encoded.append(features.float())
                if not encoded:
                    raise AssertionError("DINOv3 spatial adapter produced zero slices")
                pooled = torch.cat(encoded, dim=0).mean(dim=0)
                if tuple(pooled.shape) != (DINOV3_EMBED_DIM,):
                    raise AssertionError("DINOv3 slice pooling shape contract failed")
                if not torch.isfinite(pooled).all():
                    raise FloatingPointError("DINOv3 slice pooling produced NaN/Inf")
                output[visit_index, axis_index] = pooled.cpu().numpy()
            del visit
    if output.dtype != np.dtype(np.float32) or not np.isfinite(output).all():
        raise FloatingPointError("DINOv3 patient representation is not finite float32")
    return output


def _shard_name(patient_id: str) -> str:
    return hashlib.sha256(str(patient_id).encode("utf-8")).hexdigest() + ".private.npz"


def _validate_private_mode(path: Path, *, label: str) -> None:
    if path.stat().st_mode & 0o077:
        raise PermissionError(f"{label} permissions are too broad: {path.name}")


def _validate_shard(
    path: Path,
    *,
    patient_id: str,
    signature_sha256: str,
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        expected = {"patient_id", "representation", "signature_sha256"}
        if set(archive.files) != expected:
            raise ValueError(f"private shard schema drifted: {path.name}")
        observed_patient = str(np.asarray(archive["patient_id"]).item())
        observed_signature = str(np.asarray(archive["signature_sha256"]).item())
        representation = np.asarray(archive["representation"])
    if observed_patient != patient_id:
        raise ValueError("private shard patient identity mismatch")
    if observed_signature != signature_sha256:
        raise ValueError("private shard belongs to another extraction contract")
    if representation.dtype != np.dtype(np.float32):
        raise TypeError("private shard representation must remain float32")
    if representation.shape != (4, 2, DINOV3_EMBED_DIM):
        raise ValueError("private shard representation shape mismatch")
    if not np.isfinite(representation).all():
        raise FloatingPointError("private shard contains NaN/Inf")
    _validate_private_mode(path, label="private shard")
    return representation


def _write_exclusive_private_npz(
    destination: Path, arrays: Mapping[str, object]
) -> None:
    """Atomically publish a new NPZ while refusing any existing destination."""

    secure_directory(destination.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            np.savez(stream, **dict(arrays))
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise FileExistsError(
                f"formal feature file already exists: {destination.name}"
            ) from exc
        destination.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _write_or_validate_contract(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists():
        _validate_private_mode(path, label="private extraction contract")
        observed = json.loads(path.read_text(encoding="utf-8"))
        if canonical_json_sha256(observed) != canonical_json_sha256(payload):
            raise ValueError(
                "existing extraction contract differs from frozen signature"
            )
        return
    atomic_json(path, dict(payload), private=True)
    _validate_private_mode(path, label="private extraction contract")


def _environment_snapshot(device: torch.device) -> dict[str, object]:
    try:
        import transformers

        transformers_version: str | None = transformers.__version__
    except ModuleNotFoundError:  # pragma: no cover - loader already gates this
        transformers_version = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "numpy": np.__version__,
        "transformers": transformers_version,
    }


def extract_features(
    *,
    snapshot_dir: str | Path | None,
    fold_manifest: Path,
    cache_manifest: Path,
    output_root: Path,
    device_name: str,
    precision: str,
    batch_size: int = 64,
    limit: int | None = None,
) -> Path | None:
    """Extract private shards; combine only an untruncated formal run.

    Supplying ``limit`` explicitly marks a smoke run and must select 1..807
    canonical patients.  Omitting it is the only route to the formal
    808-patient artifact and activates the model/input lock plus CUDA gate.
    """

    formal = limit is None
    parsed_limit: int | None = None
    if not formal:
        parsed_limit = int(limit)
        if parsed_limit <= 0 or parsed_limit >= PRIMARY_COHORT_SIZE:
            raise ValueError("smoke limit must lie in 1..807; omit it for formal")
    snapshot = resolve_snapshot_dir(snapshot_dir, formal=formal)
    model_input_lock_receipt: Mapping[str, object] | None = None
    if formal:
        model_input_lock_receipt = _verify_formal_model_input_lock(snapshot)

    device = torch.device(device_name)
    if formal and (device.type != "cuda" or not torch.cuda.is_available()):
        raise RuntimeError("formal DINOv3 extraction requires an available CUDA device")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("requested CUDA device is unavailable")
        torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)

    run_kind = "formal" if formal else f"smoke_{parsed_limit}"
    run_root = secure_directory(Path(output_root) / run_kind / MODEL_NAME)
    shard_root = secure_directory(run_root / "shards")
    destination = run_root / "frozen_features.private.npz"
    execution_path = run_root / "execution.private.json"
    if formal and destination.exists():
        raise FileExistsError(f"formal feature file already exists: {destination.name}")
    if formal and execution_path.exists():
        raise FileExistsError(
            f"formal execution receipt already exists: {execution_path.name}"
        )

    model = load_dinov3_encoder(snapshot).to(device)
    audit = model_audit(model)
    signature = _model_signature(
        precision,
        audit,
        model_input_lock_sha256=(
            str(model_input_lock_receipt["lock_sha256"])
            if model_input_lock_receipt is not None
            else None
        ),
    )
    signature_sha256 = str(signature["signature_sha256"])
    _write_or_validate_contract(run_root / "contract.private.json", signature)

    patient_ids, cache = _load_primary_population(fold_manifest, cache_manifest)
    selected = patient_ids if formal else patient_ids[:parsed_limit]
    started = time.monotonic()
    for index, patient_id in enumerate(selected, start=1):
        shard = shard_root / _shard_name(patient_id)
        if shard.exists():
            _validate_shard(
                shard,
                patient_id=patient_id,
                signature_sha256=signature_sha256,
            )
            action = "resume"
        else:
            image = load_dce7(cache[patient_id])
            representation = _dinov3_patient(
                image, model, device, precision, int(batch_size)
            )
            atomic_private_npz(
                shard,
                {
                    "patient_id": np.asarray(patient_id),
                    "representation": representation,
                    "signature_sha256": np.asarray(signature_sha256),
                },
            )
            _validate_private_mode(shard, label="private shard")
            del image, representation
            gc.collect()
            action = "extract"
        elapsed = time.monotonic() - started
        print(
            f"[{MODEL_NAME}] {index}/{len(selected)} {action}; elapsed={elapsed:.1f}s",
            flush=True,
        )

    if not formal:
        return None

    representations = np.stack(
        [
            _validate_shard(
                shard_root / _shard_name(patient_id),
                patient_id=patient_id,
                signature_sha256=signature_sha256,
            )
            for patient_id in patient_ids
        ],
        axis=0,
    )
    expected_shape = (PRIMARY_COHORT_SIZE, 4, 2, DINOV3_EMBED_DIM)
    if representations.dtype != np.dtype(np.float32):
        raise TypeError("combined DINOv3 representation must remain float32")
    if representations.shape != expected_shape:
        raise AssertionError(
            f"combined DINOv3 feature shape failed: {representations.shape}"
        )
    if not np.isfinite(representations).all():
        raise FloatingPointError("combined DINOv3 representation contains NaN/Inf")
    model_sha256 = DINOV3_ARTIFACTS["model.safetensors"][1]
    config_sha256 = DINOV3_ARTIFACTS["config.json"][1]
    _write_exclusive_private_npz(
        destination,
        {
            "patient_id": np.asarray(patient_ids),
            "representation": representations,
            "spatial_axis": np.asarray(SPATIAL_AXES),
            "visits": np.asarray(VISITS),
            "model_name": np.asarray(MODEL_NAME),
            "checkpoint_sha256": np.asarray(model_sha256),
            "config_sha256": np.asarray(config_sha256),
            "extraction_signature_sha256": np.asarray(signature_sha256),
            "canonical_patient_order_sha256": np.asarray(
                ordered_text_sha256(patient_ids)
            ),
        },
    )
    _validate_private_mode(destination, label="formal feature file")
    execution = {
        "schema_version": 1,
        "experiment_kind": "user_requested_posthoc_sensitivity",
        "model": MODEL_NAME,
        "revision": DINOV3_REVISION,
        "patient_count": len(patient_ids),
        "representation_shape": list(representations.shape),
        "representation_dtype": str(representations.dtype),
        "canonical_patient_order_sha256": ordered_text_sha256(patient_ids),
        "extraction_signature_sha256": signature_sha256,
        "model_input_lock_sha256": str(model_input_lock_receipt["lock_sha256"]),
        "environment": _environment_snapshot(device),
        "elapsed_seconds": time.monotonic() - started,
        # Deliberately path-free: local cache/workspace paths are private.
        "feature_file": destination.name,
    }
    atomic_json(execution_path, execution, private=True)
    _validate_private_mode(execution_path, label="private execution receipt")
    return destination


__all__ = [
    "PRIMARY_COHORT_SIZE",
    "SNAPSHOT_ENV",
    "SPATIAL_AXES",
    "VISITS",
    "extract_features",
    "resolve_snapshot_dir",
]

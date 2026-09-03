"""Versioned extraction of frozen per-slice DINOv3 summaries."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from .contracts import (
    EXPERIMENT_ROOT,
    SUMMARY_SHAPE,
    atomic_json,
    canonical_sha256,
    file_sha256,
    load_protocol,
    patient_order_sha256,
    private_patient_token,
)
from .cache_io import CacheEntry
from .data import load_summary


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def central_local_slices(image_vczyx: np.ndarray) -> np.ndarray:
    """Return the frozen central `[4,7,32,72,72]` LOCAL crop."""

    image = np.asarray(image_vczyx)
    if image.shape != (4, 7, 112, 176, 160) or image.dtype != np.float32:
        raise ValueError(f"C1B image must be float32 [4,7,112,176,160], got {image.shape}/{image.dtype}")
    starts = tuple((length - width) // 2 for length, width in zip(image.shape[-3:], (32, 72, 72)))
    z, y, x = starts
    local = image[:, :, z : z + 32, y : y + 72, x : x + 72]
    if local.shape != (4, 7, 32, 72, 72) or not np.isfinite(local).all():
        raise ValueError("LOCAL crop contract failed")
    return np.ascontiguousarray(local)


def prepare_grayscale_slices(slices: torch.Tensor) -> torch.Tensor:
    """Map frozen C1B values to DINO pixels without patient-specific fitting."""

    if slices.ndim != 3 or tuple(slices.shape[-2:]) != (72, 72):
        raise ValueError("slice tensor must be [N,72,72]")
    pixels = slices.float().clamp(-5.0, 5.0).add(5.0).div(10.0)
    pixels = pixels[:, None].expand(-1, 3, -1, -1)
    pixels = F.interpolate(pixels, size=(224, 224), mode="bicubic", align_corners=False, antialias=True)
    mean = pixels.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = pixels.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    pixels = (pixels - mean) / std
    if not bool(torch.isfinite(pixels).all()):
        raise ValueError("DINO pixel tensor is non-finite")
    return pixels


def summarize_tokens(last_hidden_state: torch.Tensor, register_tokens: int = 4) -> torch.Tensor:
    """Concatenate CLS, patch mean and population SD; exclude registers."""

    if last_hidden_state.ndim != 3 or last_hidden_state.size(-1) != 768:
        raise ValueError("DINO hidden state must be [N,tokens,768]")
    patch_start = 1 + int(register_tokens)
    if last_hidden_state.size(1) <= patch_start:
        raise ValueError("DINO output does not contain patch tokens")
    cls = last_hidden_state[:, 0]
    patches = last_hidden_state[:, patch_start:]
    result = torch.cat(
        (cls, patches.mean(dim=1), patches.std(dim=1, unbiased=False)), dim=-1
    )
    if result.size(-1) != 2304 or not bool(torch.isfinite(result).all()):
        raise ValueError("DINO summary contract failed")
    return result


@dataclass(frozen=True)
class FrozenDINO:
    model: torch.nn.Module
    device: torch.device
    checkpoint_path: Path
    contract_sha256: str


def load_frozen_dino(device: str | torch.device = "cuda") -> FrozenDINO:
    from transformers import AutoModel

    protocol = load_protocol()
    config = protocol["dinov3"]
    resolved = torch.device(device if str(device) != "cuda" or torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(
        config["repository_id"], revision=config["revision"], local_files_only=True
    ).to(resolved)
    model.eval().requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("DINO parameters must be frozen")
    if sum(parameter.numel() for parameter in model.parameters()) != 85660416:
        raise ValueError("DINO parameter count drifted")
    checkpoint = next(
        path for path in Path.home().glob(
            ".cache/huggingface/hub/models--facebook--dinov3-vitb16-pretrain-lvd1689m/"
            f"snapshots/{config['revision']}/model.safetensors"
        )
    )
    artifacts = {
        "model.safetensors": config["checkpoint_sha256"],
        "config.json": config["config_sha256"],
        "preprocessor_config.json": config["preprocessor_sha256"],
    }
    for name, expected in artifacts.items():
        path = checkpoint.parent / name
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"DINO artifact SHA-256 mismatch: {name}")
    contract = {
        "repository_id": config["repository_id"],
        "revision": config["revision"],
        "checkpoint_sha256": config["checkpoint_sha256"],
        "input_mapping": config["input_mapping"],
        "local_shape_zyx": protocol["data"]["local_shape_zyx"],
        "visit_channel_slice_order": "visit_major_channel_major_increasing_axial_z",
        "token_contract": "CLS=0; registers=1:5 excluded; patches=5:; CLS+mean+population_std",
        "summary_shape": list(SUMMARY_SHAPE),
    }
    return FrozenDINO(model, resolved, checkpoint, canonical_sha256(contract))


@torch.inference_mode()
def extract_patient_summary(
    frozen: FrozenDINO,
    image_vczyx: np.ndarray,
    *,
    batch_size: int = 64,
) -> np.ndarray:
    local = central_local_slices(image_vczyx)
    flattened = torch.from_numpy(local.reshape(-1, 72, 72))
    chunks: list[np.ndarray] = []
    for start in range(0, len(flattened), int(batch_size)):
        pixels = prepare_grayscale_slices(flattened[start : start + batch_size]).to(
            frozen.device, non_blocking=True
        )
        autocast = frozen.device.type == "cuda"
        with torch.autocast(device_type=frozen.device.type, dtype=torch.bfloat16, enabled=autocast):
            hidden = frozen.model(pixel_values=pixels).last_hidden_state
        summary = summarize_tokens(hidden, register_tokens=4)
        chunks.append(summary.float().cpu().numpy())
    result = np.concatenate(chunks, axis=0).reshape(SUMMARY_SHAPE).astype(np.float16)
    if result.shape != SUMMARY_SHAPE or not np.isfinite(result).all():
        raise ValueError("extracted DINO cache is invalid")
    return result


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def extract_cache_entry(
    frozen: FrozenDINO,
    entry: CacheEntry,
    output_dir: str | Path,
    *,
    batch_size: int = 64,
    verify_source_hash: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    destination = Path(output_dir) / f"{private_patient_token(entry.patient_id)}.private.npz"
    if destination.exists() and not overwrite:
        load_summary(destination, entry.patient_id)
        return {"status": "REUSED", "sha256": file_sha256(destination)}
    if verify_source_hash and file_sha256(entry.path) != entry.sha256:
        raise ValueError("source C1B cache SHA-256 mismatch")
    with np.load(entry.path, allow_pickle=False) as payload:
        if str(payload["patient_id"].item()) != entry.patient_id:
            raise ValueError("C1B cache identity mismatch")
        image = np.asarray(payload["image"])
    summary = extract_patient_summary(frozen, image, batch_size=batch_size)
    _atomic_npz(
        destination,
        patient_token=np.asarray(private_patient_token(entry.patient_id)),
        summary=summary,
        source_cache_sha256=np.asarray(entry.sha256),
        contract_sha256=np.asarray(frozen.contract_sha256),
    )
    load_summary(destination, entry.patient_id)
    return {"status": "WRITTEN", "sha256": file_sha256(destination)}


def finalize_extraction_manifest(
    patient_ids: Iterable[str], output_dir: str | Path, contract_sha256: str
) -> dict[str, Any]:
    ids = tuple(map(str, patient_ids))
    output = Path(output_dir)
    hashes: list[str] = []
    for patient_id in ids:
        path = output / f"{private_patient_token(patient_id)}.private.npz"
        load_summary(path, patient_id)
        with np.load(path, allow_pickle=False) as payload:
            if str(payload["contract_sha256"].item()) != contract_sha256:
                raise ValueError("DINO cache extraction contract differs across patients")
        hashes.append(file_sha256(path))
    payload = {
        "schema_version": 1,
        "status": "COMPLETE",
        "patients": len(ids),
        "patient_order_sha256": patient_order_sha256(ids),
        "summary_shape_per_patient": list(SUMMARY_SHAPE),
        "cache_dtype": "float16",
        "contract_sha256": contract_sha256,
        "ordered_cache_hashes_sha256": canonical_sha256(hashes),
        "outcome_fields_read": [],
        "clinical_fields_read": [],
    }
    atomic_json(EXPERIMENT_ROOT / "manifests/dinov3_cache_complete.json", payload)
    return payload


__all__ = [
    "FrozenDINO", "central_local_slices", "extract_cache_entry", "extract_patient_summary",
    "finalize_extraction_manifest", "load_frozen_dino", "prepare_grayscale_slices", "summarize_tokens"
]

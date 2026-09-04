from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .contracts import ROOT, SUMMARY_SHAPE, SPATIAL_SHAPE, canonical_sha256, private_token, sha256_file
from .data import dce7_visit_from_raw, full_field_visit_views, load_dce, phase_indices, slice_indices

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def load_frozen_dino(device: str = "cuda") -> tuple[torch.nn.Module, torch.device, dict[str, str]]:
    from transformers import AutoModel

    parent = json_load(ROOT.parents[1] / "additional_experiments/dinov3_mri_adapter_radiomics_grounding_v2/configs/protocol.json")
    config = parent["dinov3"]
    resolved = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(config["repository_id"], revision=config["revision"], local_files_only=True).to(resolved)
    model.eval().requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("DINO parameters must be frozen")
    return model, resolved, config


def json_load(path: Path) -> dict:
    import json
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_pixels(slices: torch.Tensor) -> torch.Tensor:
    if slices.ndim != 3 or not torch.isfinite(slices).all():
        raise ValueError("bilateral slice tensor must be finite [N,H,W]")
    pixels = slices.float().clamp(-5.0, 5.0).add(5.0).div(10.0)
    pixels = pixels[:, None].expand(-1, 3, -1, -1)
    pixels = F.interpolate(pixels, size=(224, 224), mode="bicubic", align_corners=False, antialias=True)
    mean = pixels.new_tensor(MEAN).view(1, 3, 1, 1)
    std = pixels.new_tensor(STD).view(1, 3, 1, 1)
    pixels = (pixels - mean) / std
    if not torch.isfinite(pixels).all():
        raise ValueError("DINO pixels are non-finite")
    return pixels


def summarize(hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if hidden.ndim != 3 or hidden.shape[-1] != 768 or hidden.shape[1] != 201:
        raise ValueError(f"unexpected DINO hidden shape: {tuple(hidden.shape)}")
    cls = hidden[:, 0]
    patches = hidden[:, 5:].reshape(-1, 14, 14, 768)
    global_summary = torch.cat((cls, patches.mean((1, 2)), patches.std((1, 2), unbiased=False)), dim=-1)
    # Adaptive pooling preserves the 2-D patch layout while keeping this cache small.
    spatial = F.adaptive_avg_pool2d(patches.permute(0, 3, 1, 2), (4, 4)).permute(0, 2, 3, 1)
    if global_summary.shape[-1] != 2304 or spatial.shape[1:] != (4, 4, 768):
        raise ValueError("DINO summary contract failed")
    return global_summary, spatial


def source_digest(patient_id: str, dce_paths: list[Path], phases: np.ndarray) -> str:
    return canonical_sha256({"patient_id": patient_id, "sources": [sha256_file(path) for path in dce_paths], "phases": phases.tolist()})


@torch.inference_mode()
def extract_patient(model: torch.nn.Module, device: torch.device, patient_id: str, batch_size: int = 64) -> tuple[np.ndarray, np.ndarray, dict]:
    metas: list[dict] = []
    paths: list[Path] = []
    raw_visits: list[np.ndarray] = []
    for visit in ("T0", "T1", "T2", "T3"):
        raw, meta = load_dce(patient_id, visit)
        raw_visits.append(raw)
        metas.append(meta)
        paths.append(Path(meta["path"]))
    phase = phase_indices(patient_id)
    global_summary = np.empty(SUMMARY_SHAPE, dtype=np.float16)
    spatial_tokens = np.empty(SPATIAL_SHAPE, dtype=np.float16)
    z_indices_by_visit: list[list[int]] = []
    for visit_index, visit in enumerate(("T0", "T1", "T2", "T3")):
        raw = raw_visits[visit_index]
        dce7 = dce7_visit_from_raw(raw, phase[visit_index])
        views, _ = full_field_visit_views(dce7)
        z_indices = slice_indices(dce7[None])
        z_indices_by_visit.append(z_indices.tolist())
        selected = views[:, :, z_indices, :, :]
        flattened = torch.from_numpy(selected.reshape(-1, selected.shape[-2], selected.shape[-1]))
        global_chunks: list[np.ndarray] = []
        spatial_chunks: list[np.ndarray] = []
        for start in range(0, len(flattened), batch_size):
            pixels = prepare_pixels(flattened[start : start + batch_size]).to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                hidden = model(pixel_values=pixels).last_hidden_state
            global_part, spatial_part = summarize(hidden)
            global_chunks.append(global_part.float().cpu().numpy())
            spatial_chunks.append(spatial_part.float().cpu().numpy())
        global_summary[visit_index] = np.concatenate(global_chunks).reshape(2, 7, 32, 2304).astype(np.float16)
        spatial_tokens[visit_index] = np.concatenate(spatial_chunks).reshape(2, 7, 32, 4, 4, 768).astype(np.float16)
    if not np.isfinite(global_summary).all() or not np.isfinite(spatial_tokens).all():
        raise ValueError("DINO cache contains non-finite values")
    return global_summary, spatial_tokens, {"phase_indices": phase.tolist(), "slice_indices": z_indices_by_visit, "source_digest": source_digest(patient_id, paths, phase), "raw_meta": metas}


def write_patient(path: Path, patient_id: str, global_summary: np.ndarray, spatial_tokens: np.ndarray, provenance: dict, contract_sha: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temporary, patient_token=np.asarray(private_token(patient_id)), global_summary=global_summary, spatial_tokens=spatial_tokens, source_digest=np.asarray(provenance["source_digest"]), contract_sha256=np.asarray(contract_sha))
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return sha256_file(path)

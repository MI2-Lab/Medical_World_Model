"""Test-blind export of frozen online pre-projector response states."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from .contracts import arm_spec, canonical_sha256, file_sha256, validate_seed_fold
from .model import load_checkpoint


RESPONSE_SHAPE = (4, 192)


def ordered_patient_sha256(patient_ids: Any) -> str:
    return hashlib.sha256(
        "\n".join(str(value) for value in patient_ids).encode("utf-8")
    ).hexdigest()


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(descriptor)
    os.chmod(temporary, 0o600)
    try:
        np.savez_compressed(temporary, **arrays)
        Path(temporary).replace(path)
        path.chmod(0o600)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@torch.no_grad()
def export_response_features(
    *,
    checkpoint_path: str | Path,
    experimental_arm: str,
    seed_base: int,
    fold: int,
    data: Any,
    output_path: str | Path,
    device: torch.device,
    preregistration_lock_sha256: str,
    batch_size: int = 4,
    workers: int = 2,
) -> dict[str, Any]:
    spec = arm_spec(experimental_arm)
    if spec.name == "S0":
        raise ValueError("S0 uses the hash-verified confirmation feature asset")
    effective_seed = validate_seed_fold(seed_base, fold)
    checkpoint_path = Path(checkpoint_path).resolve()
    output_path = Path(output_path).resolve()
    metadata_path = output_path.with_suffix(".metadata.json")
    if not output_path.name.endswith(".private.npz"):
        raise ValueError("identifier-bearing features must end in .private.npz")
    if output_path.exists() or metadata_path.exists():
        raise FileExistsError("refusing to overwrite feature output")
    if int(batch_size) <= 0 or int(workers) < 0:
        raise ValueError("batch size/workers are invalid")
    model, checkpoint = load_checkpoint(str(checkpoint_path), device)
    expected = {
        "arm": spec.name,
        "seed_base": int(seed_base),
        "fold": int(fold),
        "effective_seed": effective_seed,
        "selected": True,
        "test_data_used": False,
        "pcr_used": False,
        "clinical_used": False,
        "treatment_used": False,
        "preregistration_lock_sha256": str(preregistration_lock_sha256),
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"selected checkpoint differs at {key}")
    selection_path = checkpoint_path.with_name("selection.json")
    if not selection_path.is_file() or checkpoint.get("selection_sha256") != file_sha256(selection_path):
        raise ValueError("selected checkpoint/selection binding drifted")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if checkpoint.get("selection") != selection:
        raise ValueError("selected checkpoint embeds a different selection")

    from c1b_stage_b.data import StageBDataset, make_splits

    splits = make_splits(data.folds, int(fold), data.train_only_ids)
    patient_ids = splits.train_primary + splits.val + splits.test
    split_labels = (
        ("train",) * len(splits.train_primary)
        + ("val",) * len(splits.val)
        + ("test",) * len(splits.test)
    )
    if len(patient_ids) != len(set(patient_ids)):
        raise ValueError("primary feature cohort contains duplicate patients")
    if checkpoint.get("train_patient_sha256") != canonical_sha256(sorted(splits.train_all)):
        raise ValueError("checkpoint train-patient binding drifted")
    if checkpoint.get("val_patient_sha256") != canonical_sha256(sorted(splits.val)):
        raise ValueError("checkpoint validation-patient binding drifted")
    dataset = StageBDataset(patient_ids, data.c1b_cache, transformed_ftv={})
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=int(workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(workers),
        prefetch_factor=1 if workers else None,
    )
    observed_ids: list[str] = []
    responses: list[np.ndarray] = []
    model.eval()
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        state = model.encode_response(image, None)
        if state.ndim != 3 or tuple(state.shape[1:]) != RESPONSE_SHAPE:
            raise ValueError("online response must be [B,4,192]")
        if not bool(torch.isfinite(state).all()):
            raise FloatingPointError("online response contains non-finite values")
        observed_ids.extend(str(value) for value in batch["patient_id"])
        responses.append(state.detach().to(device="cpu", dtype=torch.float32).numpy())
    if tuple(observed_ids) != tuple(patient_ids):
        raise AssertionError("feature loader changed patient order")
    response = np.concatenate(responses, axis=0).astype(np.float32, copy=False)
    if response.shape != (len(patient_ids), *RESPONSE_SHAPE):
        raise AssertionError("concatenated feature shape drifted")
    _atomic_npz(
        output_path,
        patient_id=np.asarray(patient_ids, dtype=str),
        split=np.asarray(split_labels, dtype=str),
        response_state=response,
        arm=np.asarray(spec.name),
        seed_base=np.asarray(int(seed_base), dtype=np.int64),
        fold=np.asarray(int(fold), dtype=np.int64),
    )
    metadata = {
        "schema_version": 1,
        "experiment": "residual_sph_grounding_pilot",
        "arm": spec.name,
        "seed_base": int(seed_base),
        "fold": int(fold),
        "effective_seed": effective_seed,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "selection_path": str(selection_path),
        "selection_sha256": file_sha256(selection_path),
        "selected_epoch": int(selection["selected_epoch"]),
        "feature_path": str(output_path),
        "feature_sha256": file_sha256(output_path),
        "feature_tensor": "online_preprojector_response_state",
        "feature_shape": list(response.shape),
        "feature_dtype": "float32",
        "cohort": "exact_locked_primary_train_validation_test",
        "patient_order_sha256": ordered_patient_sha256(patient_ids),
        "ftv_head_called": False,
        "sph_head_called": False,
        "test_labels_used": False,
        "clinical_or_pcr_loaded": False,
        "preregistration_lock_sha256": str(preregistration_lock_sha256),
    }
    _atomic_json(metadata_path, metadata)
    return metadata


__all__ = ["RESPONSE_SHAPE", "export_response_features", "ordered_patient_sha256"]

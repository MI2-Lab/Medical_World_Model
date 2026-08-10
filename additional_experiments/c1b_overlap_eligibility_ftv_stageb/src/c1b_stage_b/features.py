"""Test-blind export of the frozen online pre-projector response state ``r``."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from .contracts import (
    LOGICAL_OBJECTIVE_CONTRACT,
    canonical_sha256,
    file_sha256,
    ordered_patient_sha256,
)
from .data import StageBDataset, arm_cache, make_splits
from .gate import StageAAuthorization
from .inputs import StageBDataBundle
from .upstream import load_checkpoint_for_evaluation, validate_model_contract


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _validate_checkpoint_data_contract(
    checkpoint: Mapping[str, Any], data: StageBDataBundle, splits: Any
) -> str:
    """Bind feature export to the exact data contract used for training."""

    checkpoint_provenance = checkpoint.get("data_provenance")
    if not isinstance(checkpoint_provenance, Mapping):
        raise ValueError("checkpoint has no structured Stage B data provenance")
    observed_digest = canonical_sha256(checkpoint_provenance)
    if checkpoint.get("data_provenance_sha256") != observed_digest:
        raise ValueError("checkpoint data provenance digest is internally inconsistent")
    for key, value in data.provenance.items():
        if checkpoint_provenance.get(key) != value:
            raise ValueError(f"checkpoint and current data contract differ at {key}")

    expected_run_fields = {
        "train_primary_order_sha256": ordered_patient_sha256(splits.train_primary),
        "train_all_order_sha256": ordered_patient_sha256(splits.train_all),
        "validation_order_sha256": ordered_patient_sha256(splits.val),
        "test_patient_count_not_loaded": len(splits.test),
        "model_forward_fields": ["image"],
        "auxiliary_fields": ["ftv_target", "ftv_mask"],
        "logical_objective_contract": dict(LOGICAL_OBJECTIVE_CONTRACT),
    }
    for key, value in expected_run_fields.items():
        if checkpoint_provenance.get(key) != value:
            raise ValueError(f"checkpoint split/model-field provenance differs at {key}")

    expected_train = canonical_sha256(sorted(splits.train_all))
    expected_validation = canonical_sha256(sorted(splits.val))
    if checkpoint.get("train_patient_sha256") != expected_train:
        raise ValueError("checkpoint train-patient hash differs from the locked fold")
    if checkpoint.get("val_patient_sha256") != expected_validation:
        raise ValueError("checkpoint validation-patient hash differs from the locked fold")
    return observed_digest


@torch.no_grad()
def export_response_features(
    *,
    checkpoint_path: str | Path,
    arm: str,
    seed_base: int,
    fold: int,
    data: StageBDataBundle,
    authorization: StageAAuthorization,
    output_path: str | Path,
    device: torch.device,
    batch_size: int = 4,
    workers: int = 2,
) -> dict[str, Any]:
    output_path = Path(output_path).resolve()
    metadata_path = output_path.with_suffix(".metadata.json")
    if not output_path.name.endswith(".private.npz"):
        raise ValueError("identifier-bearing Stage B features must end in .private.npz")
    if output_path.exists() or metadata_path.exists():
        raise FileExistsError(f"refusing to overwrite feature export or metadata: {output_path}")
    model, checkpoint = load_checkpoint_for_evaluation(checkpoint_path, device)
    if str(checkpoint.get("arm")) != str(arm).upper():
        raise ValueError("checkpoint arm mismatch")
    if int(checkpoint.get("seed_base", -1)) != int(seed_base) or int(checkpoint.get("fold", -1)) != int(fold):
        raise ValueError("checkpoint seed/fold mismatch")
    if checkpoint.get("selected") is not True or checkpoint.get("test_data_used") is not False:
        raise ValueError("feature export requires a selected, test-blind checkpoint")
    selection_path = Path(str(checkpoint.get("selection_path", ""))).resolve()
    if not selection_path.is_file() or checkpoint.get("selection_sha256") != file_sha256(
        selection_path
    ):
        raise ValueError("selected checkpoint is not bound to its selection JSON")
    selection_payload = json.loads(selection_path.read_text(encoding="utf-8"))
    if checkpoint.get("selection") != selection_payload:
        raise ValueError("selected checkpoint embeds a different selection record")
    if int(checkpoint.get("epoch", -1)) != int(selection_payload.get("selected_epoch", -2)):
        raise ValueError("selected checkpoint epoch differs from the selection record")
    if str(checkpoint.get("stage_a_sentinel_sha256", "")) != authorization.sha256:
        raise ValueError("checkpoint and current Stage A authorization differ")
    validate_model_contract(model, arm)
    splits = make_splits(data.folds, fold, data.train_only_ids)
    checkpoint_data_provenance_sha256 = _validate_checkpoint_data_contract(
        checkpoint, data, splits
    )
    patient_ids = splits.train_primary + splits.val + splits.test
    split_labels = (
        ("train",) * len(splits.train_primary)
        + ("val",) * len(splits.val)
        + ("test",) * len(splits.test)
    )
    cache = arm_cache(arm, data.legacy_cache, data.c1b_cache)
    dataset = StageBDataset(patient_ids, cache, transformed_ftv={})
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
    response: list[np.ndarray] = []
    model.eval()
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        # This is intentionally the only feature call.  The FTV head and JEPA
        # projector output are not evaluated or exported.
        state = model.encode_response(image, None)
        if state.ndim != 3 or tuple(state.shape[1:]) != (4, 192):
            raise ValueError(f"online response state must be [B,4,192], got {tuple(state.shape)}")
        observed_ids.extend(str(value) for value in batch["patient_id"])
        response.append(state.detach().float().cpu().numpy())
    if tuple(observed_ids) != patient_ids:
        raise AssertionError("feature loader changed patient order")
    response_array = np.concatenate(response, axis=0).astype(np.float32, copy=False)
    _atomic_npz(
        output_path,
        patient_id=np.asarray(patient_ids, dtype=str),
        split=np.asarray(split_labels, dtype=str),
        response_state=response_array,
        arm=np.asarray(str(arm).upper()),
        seed_base=np.asarray(int(seed_base), dtype=np.int64),
        fold=np.asarray(int(fold), dtype=np.int64),
    )
    metadata = {
        "schema_version": 1,
        "stage": "B",
        "arm": str(arm).upper(),
        "seed_base": int(seed_base),
        "fold": int(fold),
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "feature_path": str(output_path),
        "feature_sha256": file_sha256(output_path),
        "feature_tensor": "online_preprojector_r",
        "feature_implementation_sha256": file_sha256(Path(__file__)),
        "feature_shape": list(response_array.shape),
        "patient_order_sha256": ordered_patient_sha256(patient_ids),
        "current_data_contract_provenance_sha256": canonical_sha256(data.provenance),
        "checkpoint_base_data_contract_provenance_sha256": canonical_sha256(
            data.provenance
        ),
        "checkpoint_data_provenance_sha256": checkpoint_data_provenance_sha256,
        "train_patient_sha256": checkpoint["train_patient_sha256"],
        "validation_patient_sha256": checkpoint["val_patient_sha256"],
        "ftv_head_called": False,
        "test_labels_used": False,
        "stage_a_sentinel_sha256": authorization.sha256,
    }
    _atomic_json(metadata_path, metadata)
    return metadata


__all__ = ["export_response_features"]

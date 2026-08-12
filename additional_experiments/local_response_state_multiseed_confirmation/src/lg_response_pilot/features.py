"""Test-blind export of selected online pre-projector response states.

The exporter deliberately reuses the sealed Stage-B cohort/cache adapter.  It
never reads FTV values, observability, geometry, masks, or model heads while
materializing features.  The only exported patient-level asset is an
owner-private NPZ containing the exact locked primary train/validation/test
cohort for one outer fold.
"""

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


RESPONSE_SHAPE = (4, 192)
ARMS = ("GAP0", "GAP3", "LOCAL0", "LOCAL3")
SEED_BASES = (2026, 3026, 4026, 5026, 6026)
FOLDS = tuple(range(5))


def require_sha256(value: str, label: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ordered_patient_sha256(patient_ids: Any) -> str:
    return hashlib.sha256(
        "\n".join(str(value) for value in patient_ids).encode("utf-8")
    ).hexdigest()


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    os.chmod(temporary, 0o600)
    try:
        np.savez_compressed(temporary, **arrays)
        Path(temporary).replace(path)
        path.chmod(0o600)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, Any], *, private: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600 if private else 0o644)
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _validate_checkpoint_data_contract(
    checkpoint: Mapping[str, Any], data: Any, splits: Any
) -> str:
    checkpoint_provenance = checkpoint.get("data_provenance")
    if not isinstance(checkpoint_provenance, Mapping):
        raise ValueError("checkpoint has no structured data provenance")
    observed_digest = canonical_sha256(checkpoint_provenance)
    if checkpoint.get("data_provenance_sha256") != observed_digest:
        raise ValueError("checkpoint data provenance digest is inconsistent")
    for key, value in data.provenance.items():
        if checkpoint_provenance.get(key) != value:
            raise ValueError(f"checkpoint and current data contract differ at {key}")

    expected_train = canonical_sha256(sorted(splits.train_all))
    expected_validation = canonical_sha256(sorted(splits.val))
    if checkpoint.get("train_patient_sha256") != expected_train:
        raise ValueError("checkpoint train-patient hash differs from the locked fold")
    if checkpoint.get("val_patient_sha256") != expected_validation:
        raise ValueError("checkpoint validation-patient hash differs from the locked fold")
    return observed_digest


def _validate_selected_checkpoint(
    checkpoint_path: Path,
    checkpoint: Mapping[str, Any],
    *,
    arm: str,
    seed_base: int,
    fold: int,
    authorization_sha256: str,
    preregistration_lock_sha256: str,
) -> dict[str, Any]:
    identity = {
        "arm": str(arm).upper(),
        "seed_base": int(seed_base),
        "fold": int(fold),
    }
    for key, value in identity.items():
        observed = checkpoint.get(key)
        if key in {"seed_base", "fold"}:
            observed = int(observed) if observed is not None else -1
        else:
            observed = str(observed).upper()
        if observed != value:
            raise ValueError(f"checkpoint identity mismatch at {key}")
    if checkpoint.get("selected") is not True:
        raise ValueError("feature export requires a selected checkpoint")
    if checkpoint.get("test_data_used") is not False:
        raise ValueError("selected checkpoint is not test-blind")
    if checkpoint.get("stage_a_sentinel_sha256") != authorization_sha256:
        raise ValueError("checkpoint and current Stage-A authorization differ")
    lock_sha256 = require_sha256(
        preregistration_lock_sha256, "preregistration lock"
    )
    if checkpoint.get("preregistration_status") != "PASS":
        raise ValueError("checkpoint lacks a passing preregistration verification")
    if checkpoint.get("preregistration_lock_sha256") != lock_sha256:
        raise ValueError("checkpoint uses another preregistration lock")
    checkpoint_evidence = checkpoint.get("preregistration")
    if not isinstance(checkpoint_evidence, Mapping) or checkpoint_evidence != {
        "status": "PASS",
        "lock_sha256": lock_sha256,
    }:
        raise ValueError("checkpoint preregistration evidence is inconsistent")

    selection_path = Path(str(checkpoint.get("selection_path", ""))).resolve()
    if not selection_path.is_file():
        raise ValueError("selected checkpoint has no live selection record")
    if checkpoint.get("selection_sha256") != file_sha256(selection_path):
        raise ValueError("selected checkpoint selection SHA-256 drifted")
    try:
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("selected checkpoint selection record is invalid") from error
    if not isinstance(selection, dict):
        raise ValueError("checkpoint selection must be a JSON object")
    if checkpoint.get("selection") != selection:
        raise ValueError("selected checkpoint embeds a different selection record")
    if int(checkpoint.get("epoch", -1)) != int(selection.get("selected_epoch", -2)):
        raise ValueError("selected checkpoint epoch differs from selection")
    if selection.get("preregistration_status") != "PASS":
        raise ValueError("selection lacks a passing preregistration verification")
    if selection.get("preregistration_lock_sha256") != lock_sha256:
        raise ValueError("selection uses another preregistration lock")
    if selection.get("preregistration") != checkpoint_evidence:
        raise ValueError("selection/checkpoint preregistration evidence differs")
    return selection


@torch.no_grad()
def export_response_features(
    *,
    checkpoint_path: str | Path,
    arm: str,
    seed_base: int,
    fold: int,
    data: Any,
    authorization: Any,
    output_path: str | Path,
    device: torch.device,
    preregistration_lock_sha256: str,
    batch_size: int = 4,
    workers: int = 2,
) -> dict[str, Any]:
    """Export float32 ``[N,4,192]`` online response states for one cell.

    ``data`` is the sealed ``StageBDataBundle`` and ``authorization`` is the
    sealed Stage-A authorization object.  Imports of the training model and
    data adapter are deferred so this module's pure validation helpers remain
    independently testable.
    """

    from c1b_stage_b.data import StageBDataset, make_splits
    from lg_response_pilot.model import load_checkpoint_for_evaluation

    arm = str(arm).upper()
    lock_sha256 = require_sha256(
        preregistration_lock_sha256, "preregistration lock"
    )
    if arm not in ARMS:
        raise ValueError(f"unknown confirmation arm: {arm}")
    if int(seed_base) not in SEED_BASES or int(fold) not in FOLDS:
        raise ValueError("feature identity is outside the preregistered matrix")
    if int(batch_size) <= 0 or int(workers) < 0:
        raise ValueError("batch size must be positive and workers nonnegative")

    checkpoint_path = Path(checkpoint_path).resolve()
    output_path = Path(output_path).resolve()
    metadata_path = output_path.with_suffix(".metadata.json")
    if not output_path.name.endswith(".private.npz"):
        raise ValueError("identifier-bearing features must end in .private.npz")
    if output_path.exists() or metadata_path.exists():
        raise FileExistsError(f"refusing to overwrite feature output: {output_path}")

    model, checkpoint = load_checkpoint_for_evaluation(checkpoint_path, device)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint loader did not return a checkpoint mapping")
    selection = _validate_selected_checkpoint(
        checkpoint_path,
        checkpoint,
        arm=arm,
        seed_base=int(seed_base),
        fold=int(fold),
        authorization_sha256=str(authorization.sha256),
        preregistration_lock_sha256=lock_sha256,
    )
    splits = make_splits(data.folds, int(fold), data.train_only_ids)
    run_provenance_sha256 = _validate_checkpoint_data_contract(
        checkpoint, data, splits
    )

    patient_ids = splits.train_primary + splits.val + splits.test
    split_labels = (
        ("train",) * len(splits.train_primary)
        + ("val",) * len(splits.val)
        + ("test",) * len(splits.test)
    )
    if len(patient_ids) != len(set(patient_ids)):
        raise ValueError("locked primary feature cohort contains duplicate patients")
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
    response: list[np.ndarray] = []
    model.eval()
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        state = model.encode_response(image, None)
        if not isinstance(state, torch.Tensor):
            raise TypeError("model.encode_response must return a Tensor")
        if state.ndim != 3 or tuple(state.shape[1:]) != RESPONSE_SHAPE:
            raise ValueError(
                "online pre-projector response must be [B,4,192], "
                f"got {tuple(state.shape)}"
            )
        if not torch.isfinite(state).all():
            raise FloatingPointError("online response state contains non-finite values")
        observed_ids.extend(str(value) for value in batch["patient_id"])
        response.append(state.detach().to(dtype=torch.float32, device="cpu").numpy())
    if tuple(observed_ids) != tuple(patient_ids):
        raise AssertionError("feature loader changed the locked patient order")
    response_array = np.concatenate(response, axis=0).astype(np.float32, copy=False)
    if response_array.shape != (len(patient_ids), *RESPONSE_SHAPE):
        raise AssertionError("concatenated response feature shape drifted")

    _atomic_npz(
        output_path,
        patient_id=np.asarray(patient_ids, dtype=str),
        split=np.asarray(split_labels, dtype=str),
        response_state=response_array,
        arm=np.asarray(arm),
        seed_base=np.asarray(int(seed_base), dtype=np.int64),
        fold=np.asarray(int(fold), dtype=np.int64),
    )
    metadata = {
        "schema_version": 1,
        "experiment": "local_response_state_multiseed_confirmation",
        "arm": arm,
        "seed_base": int(seed_base),
        "fold": int(fold),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "selection_path": str(Path(str(checkpoint["selection_path"])).resolve()),
        "selection_sha256": checkpoint["selection_sha256"],
        "selected_epoch": int(selection["selected_epoch"]),
        "feature_path": str(output_path),
        "feature_sha256": file_sha256(output_path),
        "feature_tensor": "online_preprojector_response_state",
        "feature_implementation_sha256": file_sha256(Path(__file__)),
        "feature_shape": list(response_array.shape),
        "feature_dtype": "float32",
        "cohort": "exact_locked_primary_train_validation_test",
        "patient_order_sha256": ordered_patient_sha256(patient_ids),
        "current_data_contract_provenance_sha256": canonical_sha256(data.provenance),
        "checkpoint_data_provenance_sha256": run_provenance_sha256,
        "train_patient_sha256": checkpoint["train_patient_sha256"],
        "validation_patient_sha256": checkpoint["val_patient_sha256"],
        "ftv_head_called": False,
        "test_labels_used": False,
        "stage_a_sentinel_sha256": str(authorization.sha256),
        "preregistration_lock": "PREREGISTRATION_LOCK.json",
        "preregistration_lock_sha256": lock_sha256,
    }
    _atomic_json(metadata_path, metadata, private=True)
    return metadata


__all__ = [
    "ARMS",
    "FOLDS",
    "RESPONSE_SHAPE",
    "SEED_BASES",
    "canonical_sha256",
    "export_response_features",
    "file_sha256",
    "ordered_patient_sha256",
    "require_sha256",
]

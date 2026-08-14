#!/usr/bin/env python3
"""Export one frozen A1 token cell and pCR-free test-fold dynamics."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any
from uuid import uuid4

import numpy as np
import torch
from torch.utils.data import DataLoader


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))
sys.path.insert(
    0,
    str(
        REPO_ROOT
        / "additional_experiments"
        / "c1b_overlap_eligibility_ftv_stageb"
        / "src"
    ),
)
sys.path.insert(0, str(EXPERIMENT_ROOT / "scripts"))

from freeze_preregistration import file_sha256, verify  # noqa: E402
from patch_token_wm.data import (  # noqa: E402
    ConditionEncoder,
    ConditionedStageBDataset,
    load_authorized_condition_table,
)
from patch_token_wm.diagnostics import (  # noqa: E402
    normalized_token_mse,
    spatial_band_labels,
    token_cosine,
    token_standard_deviation,
)
from patch_token_wm.model import PatchTokenWorldModel  # noqa: E402
from patch_token_wm.training import transition_condition_to_device  # noqa: E402
from c1b_stage_b.data import StageBDataset, make_splits  # noqa: E402
from c1b_stage_b.gate import require_stage_a_go  # noqa: E402
from c1b_stage_b.inputs import StageBDataPaths, load_stage_b_data  # noqa: E402
from train_cell import (  # noqa: E402
    DEFAULT_DATA_CONTRACT,
    DEFAULT_DATA_CONTRACT_SHA256,
    DEFAULT_STAGE_A_SENTINEL,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-base", type=int, choices=(2026, 3026), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--stage-a-sentinel", type=Path, default=DEFAULT_STAGE_A_SENTINEL
    )
    parser.add_argument("--data-contract", type=Path, default=DEFAULT_DATA_CONTRACT)
    parser.add_argument("--data-contract-sha256", default=DEFAULT_DATA_CONTRACT_SHA256)
    return parser.parse_args()


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    if not path.name.endswith(".private.npz"):
        raise ValueError("identifier/token export must end in .private.npz")
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        os.chmod(temporary, 0o600)
        np.savez(temporary, **arrays)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    """Persist a same-filesystem artifact rename before publishing the cell."""

    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write the completion metadata atomically; callers invoke this last."""

    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.chmod(temporary, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_output_target(output_path: Path, *, seed_base: int, fold: int) -> Path:
    """Constrain directory promotion to the exact private export cell."""

    if output_path.name != "tokens.private.npz":
        raise ValueError("formal output must be named tokens.private.npz")
    cell_directory = output_path.parent
    if cell_directory.name != f"fold_{int(fold)}":
        raise ValueError("formal output parent differs from the requested fold")
    if cell_directory.parent.name != f"seed_{int(seed_base)}":
        raise ValueError("formal output grandparent differs from the requested seed")
    return cell_directory


def _metadata_declares_complete(cell_directory: Path) -> bool:
    metadata_path = cell_directory / "tokens.private.metadata.json"
    if not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return metadata.get("status") == "COMPLETE"


def _new_staging_directory(cell_directory: Path) -> Path:
    """Create a hidden sibling so final publication is one directory rename."""

    cell_directory.parent.mkdir(parents=True, exist_ok=True)
    working = Path(
        tempfile.mkdtemp(
            prefix=f".{cell_directory.name}.export-working.",
            dir=cell_directory.parent,
        )
    )
    os.chmod(working, 0o700)
    return working


def _promote_staged_cell(staged: Path, destination: Path) -> Path | None:
    """Atomically publish a complete cell while preserving legacy partials.

    A previous partial directory is moved, never deleted, and restored if the
    final same-filesystem rename fails.  A completion marker is never replaced.
    """

    if staged.parent != destination.parent:
        raise ValueError("staged export must be a sibling of its destination")
    if not _metadata_declares_complete(staged):
        raise ValueError("staged export has no COMPLETE metadata marker")
    preserved: Path | None = None
    if os.path.lexists(destination):
        if _metadata_declares_complete(destination):
            raise FileExistsError(f"refusing to replace complete export: {destination}")
        preserved = destination.with_name(
            f".{destination.name}.incomplete-preserved.{uuid4().hex}"
        )
        destination.replace(preserved)
        _fsync_directory(destination.parent)
    try:
        staged.replace(destination)
        _fsync_directory(destination.parent)
    except BaseException:
        if preserved is not None and not os.path.lexists(destination):
            preserved.replace(destination)
            _fsync_directory(destination.parent)
        raise
    return preserved


def _validate_staged_export(
    token_path: Path,
    dynamics_path: Path,
    metadata_path: Path,
    expected_metadata: dict[str, Any],
) -> None:
    """Check the frozen triplet and its digests before directory promotion."""

    if not all(path.is_file() for path in (token_path, dynamics_path, metadata_path)):
        raise FileNotFoundError("staged token export triplet is incomplete")
    observed = json.loads(metadata_path.read_text(encoding="utf-8"))
    if observed != expected_metadata:
        raise ValueError("staged export metadata changed after serialization")
    if observed.get("status") != "COMPLETE":
        raise ValueError("staged export metadata is not COMPLETE")
    if observed.get("token_feature_sha256") != file_sha256(token_path):
        raise ValueError("staged token export SHA-256 mismatch")
    if observed.get("dynamics_sha256") != file_sha256(dynamics_path):
        raise ValueError("staged dynamics export SHA-256 mismatch")


def _channel_moments(values: np.ndarray) -> dict[str, Any]:
    """Return exact float64 sufficient statistics for pooled channel SD."""

    array = np.asarray(values)
    if array.ndim != 4 or array.shape[1:] != (3, 250, 128):
        raise ValueError(f"masked dynamics have wrong shape: {array.shape}")
    converted = array.astype(np.float64, copy=False)
    if not np.isfinite(converted).all():
        raise FloatingPointError("masked dynamics contain non-finite values")
    return {
        "count_per_channel": int(np.prod(converted.shape[:-1])),
        "channel_sum": converted.sum(axis=(0, 1, 2), dtype=np.float64).tolist(),
        "channel_sum_squares": np.square(converted)
        .sum(axis=(0, 1, 2), dtype=np.float64)
        .tolist(),
    }


def _per_patient_metrics(
    prediction: np.ndarray,
    actual: np.ndarray,
    shuffled: np.ndarray,
    indices: np.ndarray,
    band_names: np.ndarray,
) -> dict[str, np.ndarray]:
    patients = prediction.shape[0]
    result: dict[str, np.ndarray] = {
        "actual_cosine": np.empty(patients, dtype=np.float64),
        "shuffled_cosine": np.empty(patients, dtype=np.float64),
        "actual_normalized_mse": np.empty(patients, dtype=np.float64),
        "shuffled_normalized_mse": np.empty(patients, dtype=np.float64),
        "target_std": np.empty(patients, dtype=np.float64),
        "prediction_std": np.empty(patients, dtype=np.float64),
    }
    for patient in range(patients):
        result["actual_cosine"][patient] = token_cosine(
            prediction[patient], actual[patient]
        )
        result["shuffled_cosine"][patient] = token_cosine(
            prediction[patient], shuffled[patient]
        )
        result["actual_normalized_mse"][patient] = normalized_token_mse(
            prediction[patient], actual[patient]
        )
        result["shuffled_normalized_mse"][patient] = normalized_token_mse(
            prediction[patient], shuffled[patient]
        )
        result["target_std"][patient] = token_standard_deviation(
            actual[patient : patient + 1]
        )
        result["prediction_std"][patient] = token_standard_deviation(
            prediction[patient : patient + 1]
        )

    def normalize(values: np.ndarray) -> np.ndarray:
        centered = values - values.mean(axis=-1, keepdims=True)
        return centered / np.sqrt(np.mean(centered**2, axis=-1, keepdims=True) + 1e-6)

    error = np.mean((normalize(prediction) - normalize(actual)) ** 2, axis=-1)
    selected_bands = band_names[indices]
    for transition, visit in enumerate(("T1", "T2", "T3")):
        for band in ("central", "inner_local", "outer_local"):
            output = np.full(patients, np.nan, dtype=np.float64)
            for patient in range(patients):
                selected = error[patient, transition][
                    selected_bands[patient, transition] == band
                ]
                if selected.size:
                    output[patient] = float(selected.mean())
            result[f"{visit}_{band}_normalized_mse"] = output
    if not all(np.isfinite(values).all() for values in result.values()):
        raise FloatingPointError("per-patient dynamics contain non-finite values")
    return result


def main() -> dict[str, Any]:
    args = parse_args()
    lock = verify()
    matrix = json.loads(
        (EXPERIMENT_ROOT / "metrics" / "formal_matrix_complete.json").read_text(
            encoding="utf-8"
        )
    )
    if matrix.get("status") != "COMPLETE" or int(matrix.get("run_count", -1)) != 10:
        raise RuntimeError(
            "all ten world-model cells must be frozen before token export"
        )
    if matrix.get("preregistration_lock_sha256") != lock["lock_sha256"]:
        raise ValueError("matrix and preregistration lock differ")
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    expected = {
        "arm": "A1_PATCH3",
        "seed_base": int(args.seed_base),
        "fold": int(args.fold),
        "selected": True,
        "test_data_used": False,
        "pcr_loaded": False,
        "preregistration_lock_sha256": lock["lock_sha256"],
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"selected checkpoint differs at {key}")

    authorization = require_stage_a_go(args.stage_a_sentinel)
    paths = StageBDataPaths.load(args.data_contract, args.data_contract_sha256)
    data = load_stage_b_data(paths, authorization, verify_cache_files=False)
    splits = make_splits(data.folds, args.fold, data.train_only_ids)
    primary_ids = splits.train_primary + splits.val + splits.test
    split_labels = (
        ("train",) * len(splits.train_primary)
        + ("val",) * len(splits.val)
        + ("test",) * len(splits.test)
    )
    table = load_authorized_condition_table(
        primary_patient_ids=primary_ids,
        authorized_external_train_only_patient_ids=data.train_only_ids,
    )
    encoder = ConditionEncoder.fit(
        table,
        outer_train_patient_ids=splits.train_primary,
        authorized_external_train_only_patient_ids=data.train_only_ids,
    )
    base = StageBDataset(primary_ids, data.c1b_cache, transformed_ftv={})
    dataset = ConditionedStageBDataset(
        base, encoder, split="inference", include_patient_id=True
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=int(args.workers),
        pin_memory=True,
        persistent_workers=bool(args.workers),
        prefetch_factor=1 if args.workers else None,
        multiprocessing_context="spawn" if args.workers else None,
    )
    device = resolve_device(args.device)
    model = PatchTokenWorldModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    token_parts: list[np.ndarray] = []
    test_patient_ids: list[str] = []
    prediction_parts: list[np.ndarray] = []
    actual_parts: list[np.ndarray] = []
    shuffled_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    observed_ids: list[str] = []
    effective_seed = int(args.seed_base) + int(args.fold)
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            identities = tuple(str(value) for value in batch["patient_id"])
            image = batch["image"].to(device=device, non_blocking=True)
            condition = transition_condition_to_device(batch, device)
            output = model(
                image,
                condition,
                patient_ids=identities,
                mask_seed=effective_seed,
                epoch=0,
                logical_batch_index=batch_index,
            )
            online = output.online_tokens.detach().float().cpu().numpy()
            token_parts.append(online)
            observed_ids.extend(identities)
            test_rows = [
                index
                for index, identity in enumerate(identities)
                if split_labels[len(observed_ids) - len(identities) + index] == "test"
            ]
            if test_rows:
                prediction = (
                    output.predictions[test_rows].detach().float().cpu().numpy()
                )
                actual = output.target_masked[test_rows].detach().float().cpu().numpy()
                indices = output.mask_indices[test_rows]
                shuffled_visits = output.target_tokens[test_rows][:, (2, 3, 1)]
                gather = indices.unsqueeze(-1).expand(
                    -1, -1, -1, shuffled_visits.shape[-1]
                )
                shuffled = (
                    torch.gather(shuffled_visits, 2, gather)
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )
                prediction_parts.append(prediction)
                actual_parts.append(actual)
                shuffled_parts.append(shuffled)
                index_parts.append(indices.cpu().numpy())
                test_patient_ids.extend(identities[index] for index in test_rows)
    if tuple(observed_ids) != primary_ids:
        raise AssertionError("export loader changed formal patient order")
    tokens = np.concatenate(token_parts, axis=0).astype(np.float32, copy=False)
    if tokens.shape != (808, 4, 500, 128):
        raise ValueError(f"formal token export has wrong shape: {tokens.shape}")
    prediction = np.concatenate(prediction_parts)
    actual = np.concatenate(actual_parts)
    shuffled = np.concatenate(shuffled_parts)
    indices = np.concatenate(index_parts)
    bands = spatial_band_labels(model.local_coordinates_xyz_mm.detach().cpu().numpy())
    dynamics = _per_patient_metrics(prediction, actual, shuffled, indices, bands)
    output_path = args.output.resolve()
    cell_directory = _validate_output_target(
        output_path, seed_base=int(args.seed_base), fold=int(args.fold)
    )
    staging_directory = _new_staging_directory(cell_directory)
    staged_output_path = staging_directory / output_path.name
    _atomic_npz(
        staged_output_path,
        patient_id=np.asarray(primary_ids, dtype=str),
        split=np.asarray(split_labels, dtype=str),
        tokens=tokens,
        fractional_weights=model.local_weights.detach().float().cpu().numpy(),
        coordinates_xyz_mm=model.local_coordinates_xyz_mm.detach()
        .float()
        .cpu()
        .numpy(),
        seed_base=np.asarray(int(args.seed_base), dtype=np.int64),
        fold=np.asarray(int(args.fold), dtype=np.int64),
    )
    dynamics_path = staging_directory / "dynamics.private.npz"
    _atomic_npz(
        dynamics_path,
        patient_id=np.asarray(test_patient_ids, dtype=str),
        fold=np.full(len(test_patient_ids), int(args.fold), dtype=np.int64),
        **dynamics,
    )
    metadata = {
        "schema_version": 1,
        "status": "COMPLETE",
        "arm": "A1_PATCH3",
        "seed_base": int(args.seed_base),
        "fold": int(args.fold),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "token_feature_sha256": file_sha256(staged_output_path),
        "dynamics_sha256": file_sha256(dynamics_path),
        "token_shape": list(tokens.shape),
        "test_dynamics_patients": len(test_patient_ids),
        "target_order_actual": ["T1", "T2", "T3"],
        "target_order_cyclic_shuffle": ["T2", "T3", "T1"],
        "pcr_loaded": False,
        "condition_in_exported_tokens": False,
        "export_batch_size": int(args.batch_size),
        "mask_schedule": (
            "effective_seed_epoch0_logical_batch_index_patient_sha256_transition"
        ),
        "data_loader_workers": int(args.workers),
        "multiprocessing_start_method": "spawn" if args.workers else "none",
        "target_channel_moments": _channel_moments(actual),
        "prediction_channel_moments": _channel_moments(prediction),
        "preregistration_lock_sha256": lock["lock_sha256"],
    }
    metadata_path = staging_directory / "tokens.private.metadata.json"
    _atomic_json(metadata_path, metadata)
    _validate_staged_export(staged_output_path, dynamics_path, metadata_path, metadata)
    preserved_partial = _promote_staged_cell(staging_directory, cell_directory)
    if preserved_partial is not None:
        print(
            f"preserved prior incomplete export at {preserved_partial}",
            file=sys.stderr,
        )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


if __name__ == "__main__":
    main()

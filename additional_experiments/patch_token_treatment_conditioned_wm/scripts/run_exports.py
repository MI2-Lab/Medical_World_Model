#!/usr/bin/env python3
"""Export all ten frozen A1 token cells with fail-fast GPU scheduling."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import threading
from typing import Any


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "scripts"))

from freeze_preregistration import file_sha256, verify  # noqa: E402
from run_matrix import ActiveProcesses, parse_devices  # noqa: E402
from train_cell import (  # noqa: E402
    DEFAULT_DATA_CONTRACT,
    DEFAULT_DATA_CONTRACT_SHA256,
    DEFAULT_STAGE_A_SENTINEL,
)


SEEDS = (2026, 3026)
FOLDS = tuple(range(5))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=EXPERIMENT_ROOT / "checkpoints" / "a1_formal",
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=EXPERIMENT_ROOT / "features" / "a1_formal",
    )
    parser.add_argument(
        "--stage-a-sentinel", type=Path, default=DEFAULT_STAGE_A_SENTINEL
    )
    parser.add_argument("--data-contract", type=Path, default=DEFAULT_DATA_CONTRACT)
    parser.add_argument("--data-contract-sha256", default=DEFAULT_DATA_CONTRACT_SHA256)
    return parser.parse_args()


def _paths(args: argparse.Namespace, seed: int, fold: int) -> tuple[Path, Path, Path]:
    checkpoint = (
        args.checkpoint_root.resolve() / f"seed_{seed}" / f"fold_{fold}" / "selected.pt"
    )
    feature = (
        args.feature_root.resolve()
        / f"seed_{seed}"
        / f"fold_{fold}"
        / "tokens.private.npz"
    )
    return checkpoint, feature, feature.with_suffix(".metadata.json")


def _metadata_declares_complete(metadata_path: Path) -> bool:
    """Only the last-written cell marker makes an export resumably complete."""

    if not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return metadata.get("status") == "COMPLETE"


def _validate_channel_moments(
    value: object, *, expected_patients: int, label: str
) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} channel moments are missing")
    expected_count = int(expected_patients) * 3 * 250
    if int(value.get("count_per_channel", -1)) != expected_count:
        raise ValueError(f"{label} channel-moment count differs")
    for key in ("channel_sum", "channel_sum_squares"):
        numbers = value.get(key)
        if (
            not isinstance(numbers, list)
            or len(numbers) != 128
            or not all(isinstance(number, (int, float)) for number in numbers)
            or not all(math.isfinite(float(number)) for number in numbers)
        ):
            raise ValueError(f"{label} {key} is not 128 finite values")
    if any(float(number) < 0.0 for number in value["channel_sum_squares"]):
        raise ValueError(f"{label} channel_sum_squares contains a negative value")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_export(
    args: argparse.Namespace, seed: int, fold: int, lock_sha: str
) -> dict[str, Any]:
    checkpoint, feature, metadata_path = _paths(args, seed, fold)
    dynamics = feature.with_name("dynamics.private.npz")
    if not all(
        path.is_file() for path in (checkpoint, feature, dynamics, metadata_path)
    ):
        raise FileNotFoundError(f"incomplete export seed={seed} fold={fold}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "status": "COMPLETE",
        "arm": "A1_PATCH3",
        "seed_base": int(seed),
        "fold": int(fold),
        "preregistration_lock_sha256": lock_sha,
        "pcr_loaded": False,
        "condition_in_exported_tokens": False,
        "token_shape": [808, 4, 500, 128],
        "export_batch_size": int(args.batch_size),
        "mask_schedule": (
            "effective_seed_epoch0_logical_batch_index_patient_sha256_transition"
        ),
        "data_loader_workers": int(args.workers),
        "multiprocessing_start_method": "spawn" if args.workers else "none",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"export metadata differs at {key}")
    if metadata.get("checkpoint_sha256") != file_sha256(checkpoint):
        raise ValueError("export checkpoint SHA-256 mismatch")
    if metadata.get("token_feature_sha256") != file_sha256(feature):
        raise ValueError("token export SHA-256 mismatch")
    if metadata.get("dynamics_sha256") != file_sha256(dynamics):
        raise ValueError("dynamics export SHA-256 mismatch")
    test_patients = int(metadata.get("test_dynamics_patients", -1))
    if test_patients < 1:
        raise ValueError("test dynamics patient count is invalid")
    _validate_channel_moments(
        metadata.get("target_channel_moments"),
        expected_patients=test_patients,
        label="target",
    )
    _validate_channel_moments(
        metadata.get("prediction_channel_moments"),
        expected_patients=test_patients,
        label="prediction",
    )
    return {**metadata, "export_metadata_sha256": file_sha256(metadata_path)}


def main() -> dict[str, Any]:
    args = parse_args()
    devices = parse_devices(args.devices)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    lock = verify()
    matrix = json.loads(
        (EXPERIMENT_ROOT / "metrics" / "formal_matrix_complete.json").read_text(
            encoding="utf-8"
        )
    )
    if matrix.get("status") != "COMPLETE" or int(matrix.get("run_count", -1)) != 10:
        raise RuntimeError("formal world-model matrix is incomplete")
    cells = [(seed, fold) for seed in SEEDS for fold in FOLDS]
    completed: dict[tuple[int, int], dict[str, Any]] = {}
    pending: list[tuple[int, int, str]] = []
    for index, (seed, fold) in enumerate(cells):
        _checkpoint, _feature, metadata_path = _paths(args, seed, fold)
        if _metadata_declares_complete(metadata_path):
            completed[(seed, fold)] = _validate_export(
                args, seed, fold, str(lock["lock_sha256"])
            )
        else:
            pending.append((seed, fold, devices[index % len(devices)]))
    bucket = {
        device: [cell for cell in pending if cell[2] == device] for device in devices
    }
    stop = threading.Event()
    active = ActiveProcesses()
    export_script = EXPERIMENT_ROOT / "scripts" / "export_cell.py"

    def device_worker(device: str) -> list[tuple[tuple[int, int], dict[str, Any]]]:
        local: list[tuple[tuple[int, int], dict[str, Any]]] = []
        for seed, fold, _ in bucket[device]:
            if stop.is_set():
                return local
            checkpoint, feature, _metadata = _paths(args, seed, fold)
            command = [
                sys.executable,
                str(export_script),
                "--seed-base",
                str(seed),
                "--fold",
                str(fold),
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(feature),
                "--device",
                device,
                "--workers",
                str(args.workers),
                "--batch-size",
                str(args.batch_size),
                "--stage-a-sentinel",
                str(args.stage_a_sentinel),
                "--data-contract",
                str(args.data_contract),
                "--data-contract-sha256",
                str(args.data_contract_sha256),
            ]
            log = (
                EXPERIMENT_ROOT / "logs" / f"export_seed_{seed}_fold_{fold}.private.log"
            )
            log.parent.mkdir(parents=True, exist_ok=True)
            active.run(command, log)
            local.append(
                (
                    (seed, fold),
                    _validate_export(args, seed, fold, str(lock["lock_sha256"])),
                )
            )
        return local

    try:
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            futures = [executor.submit(device_worker, device) for device in devices]
            for future in as_completed(futures):
                for key, metadata in future.result():
                    completed[key] = metadata
    except BaseException:
        stop.set()
        active.abort()
        raise
    if set(completed) != set(cells):
        raise RuntimeError("ten-cell token export matrix is incomplete")
    payload = {
        "schema_version": 1,
        "status": "COMPLETE",
        "arm": "A1_PATCH3",
        "run_count": 10,
        "seeds": list(SEEDS),
        "folds": list(FOLDS),
        "preregistration_lock_sha256": lock["lock_sha256"],
        "pcr_loaded": False,
        "all_token_shapes": [[808, 4, 500, 128]],
        "data_loader": {
            "batch_size": int(args.batch_size),
            "workers_per_cell": int(args.workers),
            "multiprocessing_start_method": "spawn" if args.workers else "none",
        },
        "cuda_allocator_config": os.environ["PYTORCH_CUDA_ALLOC_CONF"],
        "cells": [
            {
                "seed_base": seed,
                "fold": fold,
                "token_feature_sha256": completed[(seed, fold)]["token_feature_sha256"],
                "dynamics_sha256": completed[(seed, fold)]["dynamics_sha256"],
                "export_metadata_sha256": completed[(seed, fold)][
                    "export_metadata_sha256"
                ],
                "test_dynamics_patients": completed[(seed, fold)][
                    "test_dynamics_patients"
                ],
            }
            for seed, fold in cells
        ],
    }
    output = EXPERIMENT_ROOT / "metrics" / "formal_exports_complete.json"
    if output.exists():
        observed = json.loads(output.read_text(encoding="utf-8"))
        if observed != payload:
            raise ValueError("existing formal export completion record differs")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return payload
    _atomic_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    main()

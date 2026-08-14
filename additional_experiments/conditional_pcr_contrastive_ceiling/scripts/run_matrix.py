#!/usr/bin/env python3
"""Preflight or run supervised ceiling cells across available devices."""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from typing import Any, Mapping, Sequence


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CELL = EXPERIMENT_ROOT / "scripts" / "train_cell.py"
CHECKPOINT_ROOT = EXPERIMENT_ROOT / "checkpoints"
FEATURE_ROOT = EXPERIMENT_ROOT / "features"
METRICS_ROOT = EXPERIMENT_ROOT / "metrics"
LOG_ROOT = EXPERIMENT_ROOT / "logs"
CONFIG_PATH = EXPERIMENT_ROOT / "configs" / "experiment.json"
SOURCE_ROOT = EXPERIMENT_ROOT / "src" / "conditional_ceiling"
REGISTERED_ARMS = ("B1", "B2", "B3")
REGISTERED_SEEDS = (2026, 3026, 4026)
REGISTERED_FOLDS = tuple(range(5))


@dataclass(frozen=True)
class Cell:
    seed: int
    arm: str
    fold: int

    def outputs(self) -> tuple[Path, ...]:
        return (
            CHECKPOINT_ROOT
            / f"seed_{self.seed}"
            / self.arm
            / f"fold_{self.fold}"
            / "selected.private.pt",
            CHECKPOINT_ROOT
            / f"seed_{self.seed}"
            / self.arm
            / f"fold_{self.fold}"
            / "selection.private.json",
            FEATURE_ROOT
            / f"seed_{self.seed}"
            / self.arm
            / f"fold_{self.fold}"
            / "representation.private.npz",
            METRICS_ROOT
            / f"matching_audit_cell_seed_{self.seed}_{self.arm}_fold_{self.fold}.private.json",
        )

    def status(self) -> str:
        present = tuple(path.is_file() for path in self.outputs())
        if all(present):
            try:
                validate_cell_artifacts(self)
            except (OSError, ValueError, TypeError, KeyError):
                return "invalid"
            return "complete"
        if any(present):
            return "partial"
        return "missing"

    def log_path(self) -> Path:
        return LOG_ROOT / f"{self.arm}_seed{self.seed}_fold{self.fold}.private.log"


def _cell_identity(payload: Mapping[str, Any], cell: Cell, label: str) -> None:
    if (
        payload.get("arm") != cell.arm
        or payload.get("seed") != cell.seed
        or payload.get("fold") != cell.fold
    ):
        raise ValueError(f"{label} cell identity disagrees with its path")


def _current_config_sha256() -> str:
    digest = hashlib.sha256()
    with CONFIG_PATH.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _current_training_implementation_hashes() -> dict[str, str]:
    paths = {
        "scripts/train_cell.py": TRAIN_CELL,
        "src/conditional_ceiling/data.py": SOURCE_ROOT / "data.py",
        "src/conditional_ceiling/contracts.py": SOURCE_ROOT / "contracts.py",
        "src/conditional_ceiling/training.py": SOURCE_ROOT / "training.py",
        "src/conditional_ceiling/losses.py": SOURCE_ROOT / "losses.py",
        "src/conditional_ceiling/model.py": SOURCE_ROOT / "model.py",
        "src/conditional_ceiling/strata.py": SOURCE_ROOT / "strata.py",
    }
    return {name: _file_sha256(path) for name, path in paths.items()}


def validate_cell_artifacts(cell: Cell) -> None:
    """Require all four private outputs and their exact identity/schema."""

    import numpy as np
    import torch

    checkpoint_path, selection_path, feature_path, matching_path = cell.outputs()
    if not all(path.is_file() for path in cell.outputs()):
        raise FileNotFoundError("cell artifact set is incomplete")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not isinstance(selection, dict):
        raise ValueError("selection must be a JSON object")
    _cell_identity(selection, cell, "selection")
    current_config_sha256 = _current_config_sha256()
    current_implementation = _current_training_implementation_hashes()
    if (
        selection.get("status") != "SELECTED_VALIDATION_ONLY"
        or selection.get("config_sha256") != current_config_sha256
        or selection.get("feature_sha256") != _file_sha256(feature_path)
        or selection.get("training_implementation_sha256") != current_implementation
        or selection.get("test_labels_used_for_training_or_selection") is not False
        or selection.get("external_ispy1_patients_used") != 0
        or selection.get("world_model_claim_allowed") is not False
    ):
        raise ValueError("selection isolation contract failed")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("provenance"), dict
    ):
        raise ValueError("checkpoint schema is invalid")
    _cell_identity(checkpoint["provenance"], cell, "checkpoint")
    if (
        checkpoint.get("pcr_supervised") is not True
        or checkpoint.get("world_model_claim_allowed") is not False
        or checkpoint["provenance"].get("config_sha256")
        != current_config_sha256
        or checkpoint["provenance"].get("feature_sha256")
        != _file_sha256(feature_path)
        or checkpoint["provenance"].get("training_implementation_sha256")
        != current_implementation
        or checkpoint["provenance"].get(
            "test_labels_used_for_training_or_selection"
        )
        is not False
    ):
        raise ValueError("checkpoint reporting/isolation contract failed")
    if checkpoint["provenance"] != {
        key: selection[key] for key in checkpoint["provenance"]
    }:
        raise ValueError("checkpoint and selection provenance disagree")

    with np.load(feature_path, allow_pickle=False) as archive:
        if set(archive.files) != {
            "patient_id",
            "split",
            "representation",
            "arm",
            "seed",
            "fold",
        }:
            raise ValueError("feature archive schema drifted")
        identifiers = archive["patient_id"].astype(str)
        splits = archive["split"].astype(str)
        representation = archive["representation"]
        if (
            len(identifiers) != 808
            or len(set(identifiers)) != 808
            or set(splits) != {"train", "val", "test"}
            or representation.shape != (808, 4, 64)
            or representation.dtype != np.float32
            or not np.isfinite(representation).all()
            or str(archive["arm"].item()) != cell.arm
            or int(archive["seed"].item()) != cell.seed
            or int(archive["fold"].item()) != cell.fold
        ):
            raise ValueError("feature archive tensor/identity contract failed")

    matching = json.loads(matching_path.read_text(encoding="utf-8"))
    if not isinstance(matching, dict):
        raise ValueError("matching audit must be a JSON object")
    _cell_identity(matching, cell, "matching audit")
    if matching.get("contains_patient_identifiers") is not False:
        raise ValueError("matching audit is not aggregate-only")
    audit = matching.get("matching")
    if (
        not isinstance(audit, dict)
        or audit.get("scope") != "outer_train_only"
        or audit.get("unmatched_fallback_used") is not False
        or audit.get("test_patients_used") is not False
    ):
        raise ValueError("matching audit isolation contract failed")
    expected_sampling = None if cell.arm == "B1" else "all_eligible_anchors_exactly_once_per_epoch"
    expected_batch = None if cell.arm == "B1" else 4
    expected_microbatch = 1 if cell.arm == "B3" else None
    expected_anchors = None if cell.arm == "B1" else int(audit.get("usable_patients", -1))
    for provenance in (selection, checkpoint["provenance"]):
        if (
            provenance.get("anchor_sampling_strategy") != expected_sampling
            or provenance.get("logical_patient_batch_size") != expected_batch
            or provenance.get("encoder_microbatch_size") != expected_microbatch
            or provenance.get("eligible_anchors_per_epoch") != expected_anchors
        ):
            raise ValueError("cell sampling/microbatch provenance contract failed")


def _csv_values(value: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if not values or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("value must be a nonempty unique CSV list")
    return values


def _arms(value: str) -> tuple[str, ...]:
    values = tuple(part.upper() for part in _csv_values(value))
    if any(part not in REGISTERED_ARMS for part in values):
        raise argparse.ArgumentTypeError(f"arms must be selected from {REGISTERED_ARMS}")
    return values


def _integers(value: str, allowed: Sequence[int], label: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part) for part in _csv_values(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{label} must be comma-separated integers") from error
    if any(part not in allowed for part in parsed):
        raise argparse.ArgumentTypeError(f"{label} must be selected from {tuple(allowed)}")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", type=_arms, default=REGISTERED_ARMS)
    parser.add_argument(
        "--seeds",
        type=lambda value: _integers(value, REGISTERED_SEEDS, "seeds"),
        default=(2026, 3026),
    )
    parser.add_argument(
        "--folds",
        type=lambda value: _integers(value, REGISTERED_FOLDS, "folds"),
        default=REGISTERED_FOLDS,
    )
    parser.add_argument(
        "--gpus",
        type=_csv_values,
        default=("0", "1", "2"),
        help="CUDA indices, CUDA device strings, or `cpu`, comma separated.",
    )
    parser.add_argument("--workers", type=int)
    parser.add_argument("--physical-batch-size", type=int)
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-cache-content", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Without this flag, print the validated execution plan only.",
    )
    return parser.parse_args(argv)


def _devices(values: Sequence[str]) -> tuple[str, ...]:
    import torch

    devices: list[str] = []
    for raw in values:
        value = str(raw).strip().lower()
        if value == "cpu":
            device = "cpu"
        elif value.isdigit():
            device = f"cuda:{int(value)}"
        elif value.startswith("cuda:") and value.split(":", 1)[1].isdigit():
            device = f"cuda:{int(value.split(':', 1)[1])}"
        else:
            raise ValueError(f"invalid device: {raw!r}")
        if device.startswith("cuda:"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA device requested but CUDA is unavailable")
            if int(device.split(":", 1)[1]) >= torch.cuda.device_count():
                raise ValueError(f"unavailable CUDA device: {device}")
        devices.append(device)
    if len(set(devices)) != len(devices):
        raise ValueError("devices must be unique")
    return tuple(devices)


def build_cells(
    seeds: Sequence[int], arms: Sequence[str], folds: Sequence[int]
) -> tuple[Cell, ...]:
    if any(int(seed) not in REGISTERED_SEEDS for seed in seeds):
        raise ValueError("unregistered matrix seed")
    if any(str(arm) not in REGISTERED_ARMS for arm in arms):
        raise ValueError("unregistered matrix arm")
    if any(int(fold) not in REGISTERED_FOLDS for fold in folds):
        raise ValueError("unregistered matrix fold")
    return tuple(
        Cell(int(seed), str(arm), int(fold))
        for seed in seeds
        for fold in folds
        for arm in arms
    )


def build_command(cell: Cell, device: str, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(TRAIN_CELL),
        "--arm",
        cell.arm,
        "--seed",
        str(cell.seed),
        "--fold",
        str(cell.fold),
        "--device",
        device,
    ]
    if args.workers is not None:
        command.extend(("--workers", str(args.workers)))
    if args.physical_batch_size is not None:
        command.extend(("--physical-batch-size", str(args.physical_batch_size)))
    if args.overwrite:
        command.append("--overwrite")
    if args.verify_cache_content:
        command.append("--verify-cache-content")
    return command


class ProcessRegistry:
    """Track process groups so a failed cell terminates active peers."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active: dict[int, subprocess.Popen[bytes]] = {}
        self.failed = threading.Event()

    def run(self, command: Sequence[str], log_path: Path, *, overwrite: bool) -> None:
        if self.failed.is_set():
            raise RuntimeError("matrix aborted after another cell failed")
        log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        mode = "wb" if overwrite else "ab"
        with log_path.open(mode) as log_stream:
            log_path.chmod(0o600)
            process = subprocess.Popen(
                list(command),
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            with self.lock:
                self.active[process.pid] = process
            return_code = process.wait()
        with self.lock:
            self.active.pop(process.pid, None)
        if return_code:
            self.failed.set()
            self.terminate_all()
            raise subprocess.CalledProcessError(return_code, list(command))

    def terminate_all(self) -> None:
        with self.lock:
            active = tuple(self.active.values())
        for process in active:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and any(
            process.poll() is None for process in active
        ):
            time.sleep(0.05)
        for process in active:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def run_matrix(
    cells: Sequence[Cell], devices: Sequence[str], args: argparse.Namespace
) -> tuple[Cell, ...]:
    buckets: list[list[Cell]] = [[] for _ in devices]
    for index, cell in enumerate(cells):
        buckets[index % len(devices)].append(cell)
    registry = ProcessRegistry()
    completed: list[Cell] = []
    completed_lock = threading.Lock()

    def worker(device: str, assigned: Sequence[Cell]) -> None:
        for cell in assigned:
            if registry.failed.is_set():
                return
            registry.run(
                build_command(cell, device, args),
                cell.log_path(),
                overwrite=bool(args.overwrite),
            )
            validate_cell_artifacts(cell)
            with completed_lock:
                completed.append(cell)

    try:
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            futures = [
                executor.submit(worker, device, bucket)
                for device, bucket in zip(devices, buckets, strict=True)
                if bucket
            ]
            for future in as_completed(futures):
                future.result()
    except BaseException:
        registry.failed.set()
        registry.terminate_all()
        raise
    return tuple(completed)


def main(argv: Sequence[str] | None = None) -> None:
    os.umask(0o077)
    args = parse_args(argv)
    if args.workers is not None and args.workers < 0:
        raise ValueError("workers must be non-negative")
    if args.physical_batch_size is not None and args.physical_batch_size != 4:
        raise ValueError(
            "the locked logical patient batch size is 4; use encoder microbatching "
            "rather than changing the sampling budget"
        )
    if args.skip_complete and args.overwrite:
        raise ValueError("--skip-complete and --overwrite are mutually exclusive")
    devices = _devices(args.gpus)
    all_cells = build_cells(args.seeds, args.arms, args.folds)
    statuses = {cell: cell.status() for cell in all_cells}
    partial = [
        cell for cell, status in statuses.items() if status in {"partial", "invalid"}
    ]
    if partial and not args.overwrite:
        raise FileExistsError(
            "matrix contains partial private cells; rerun those cells with --overwrite"
        )
    if args.skip_complete:
        cells = tuple(cell for cell in all_cells if statuses[cell] != "complete")
    elif not args.overwrite:
        complete = [cell for cell, status in statuses.items() if status == "complete"]
        if complete:
            raise FileExistsError(
                "matrix contains complete cells; use --skip-complete or --overwrite"
            )
        cells = all_cells
    else:
        cells = all_cells
    plan = {
        "status": "PREFLIGHT_PASS",
        "arms": list(args.arms),
        "seeds": list(args.seeds),
        "folds": list(args.folds),
        "devices": list(devices),
        "total_registered_cells": len(all_cells),
        "cells_to_run": len(cells),
        "complete_cells_skipped": sum(
            status == "complete" for status in statuses.values()
        )
        if args.skip_complete
        else 0,
        "private_log_paths": [str(cell.log_path()) for cell in cells],
        "execute": bool(args.execute),
        "commands": [
            build_command(cell, devices[index % len(devices)], args)
            for index, cell in enumerate(cells)
        ],
    }
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    completed = run_matrix(cells, devices, args)
    result = {
        **{key: value for key, value in plan.items() if key != "commands"},
        "status": "COMPLETE",
        "completed_cells": len(completed),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

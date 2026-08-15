#!/usr/bin/env python3
"""Preflight and launch the exact 2-seed x 5-fold A1 matrix."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

import torch


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
from train_cell import (  # noqa: E402
    DEFAULT_DATA_CONTRACT,
    DEFAULT_DATA_CONTRACT_SHA256,
    DEFAULT_STAGE_A_SENTINEL,
)
from c1b_stage_b.gate import require_stage_a_go  # noqa: E402
from c1b_stage_b.inputs import StageBDataPaths, load_stage_b_data  # noqa: E402


SEEDS = (2026, 3026)
FOLDS = tuple(range(5))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Publish a matrix manifest only after durable, strict JSON serialization."""

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
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def parse_devices(value: str) -> tuple[str, ...]:
    devices = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("devices must be nonempty and unique")
    for device in devices:
        parsed = torch.device(device)
        if parsed.type != "cuda" or parsed.index is None:
            raise ValueError("every formal device must be an explicit cuda:N")
        if not torch.cuda.is_available() or parsed.index >= torch.cuda.device_count():
            raise ValueError(f"requested unavailable device {device}")
    return devices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=EXPERIMENT_ROOT / "checkpoints" / "a1_formal",
    )
    parser.add_argument(
        "--stage-a-sentinel", type=Path, default=DEFAULT_STAGE_A_SENTINEL
    )
    parser.add_argument("--data-contract", type=Path, default=DEFAULT_DATA_CONTRACT)
    parser.add_argument("--data-contract-sha256", default=DEFAULT_DATA_CONTRACT_SHA256)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


class ActiveProcesses:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.processes: dict[int, subprocess.Popen[bytes]] = {}
        self.aborted = False

    def run(self, command: list[str], log_path: Path) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab", buffering=0) as log:
            with self.lock:
                if self.aborted:
                    raise RuntimeError("matrix already aborted")
                process = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                self.processes[process.pid] = process
            return_code = process.wait()
            with self.lock:
                self.processes.pop(process.pid, None)
                if return_code:
                    self.aborted = True
                    peers = tuple(self.processes.values())
                else:
                    peers = ()
            if return_code:
                self._terminate(peers)
                raise subprocess.CalledProcessError(return_code, command)

    def abort(self) -> None:
        with self.lock:
            self.aborted = True
            peers = tuple(self.processes.values())
        self._terminate(peers)

    @staticmethod
    def _terminate(processes: tuple[subprocess.Popen[bytes], ...]) -> None:
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and any(
            process.poll() is None for process in processes
        ):
            time.sleep(0.05)
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def _cell_dir(root: Path, seed: int, fold: int) -> Path:
    return root / f"seed_{seed}" / f"fold_{fold}"


def _validate_complete_cell(
    path: Path, seed: int, fold: int, lock_sha: str
) -> dict[str, Any]:
    completion_path = path / "cell_complete.json"
    selection_path = path / "selection.json"
    checkpoint_path = path / "selected.pt"
    if not all(
        candidate.is_file()
        for candidate in (completion_path, selection_path, checkpoint_path)
    ):
        raise FileNotFoundError(f"formal cell is incomplete at {path}")
    payload = json.loads(completion_path.read_text(encoding="utf-8"))
    expected = {
        "status": "COMPLETE",
        "arm": "A1_PATCH3",
        "seed_base": int(seed),
        "fold": int(fold),
        "preregistration_lock_sha256": lock_sha,
        "training_is_pcr_free": True,
        "test_data_used": False,
        "finite_noncollapsed_selection": True,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"completed formal cell differs at {key}: {path}")
    if payload.get("selection_sha256") != file_sha256(selection_path):
        raise ValueError("formal selection SHA-256 mismatch")
    if payload.get("selected_checkpoint_sha256") != file_sha256(checkpoint_path):
        raise ValueError("formal checkpoint SHA-256 mismatch")
    return payload


def main() -> dict[str, Any]:
    args = parse_args()
    devices = parse_devices(args.devices)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    lock = verify()
    authorization = require_stage_a_go(args.stage_a_sentinel)
    paths = StageBDataPaths.load(args.data_contract, args.data_contract_sha256)
    # The parent performs the one expensive SHA/archive preflight. Children use
    # the already hash-pinned manifest and same-fd stat/schema checks.
    data = load_stage_b_data(paths, authorization, verify_cache_files=True)
    if (
        int(data.folds["patient_id"].nunique()) != 808
        or len(data.train_only_ids) != 139
    ):
        raise ValueError("formal 808+139 population contract drifted")
    cells = [(seed, fold) for seed in SEEDS for fold in FOLDS]
    assignments = [
        {"seed_base": seed, "fold": fold, "device": devices[index % len(devices)]}
        for index, (seed, fold) in enumerate(cells)
    ]
    preflight: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "matrix": {"arm": "A1_PATCH3", "seeds": list(SEEDS), "folds": list(FOLDS)},
        "run_count": 10,
        "devices": list(devices),
        "assignments": assignments,
        "population": {"primary": 808, "authorized_external_train_only": 139},
        "cache_preflight": {
            "full_sha256_and_archive_contract_verified": True,
            "c1b_cache_count": len(data.c1b_cache),
        },
        "batch": {"physical": 4, "accumulation": 8, "logical": 32},
        "data_loader": {
            "workers_per_cell": int(args.workers),
            "multiprocessing_start_method": "spawn" if args.workers else "none",
        },
        "cuda_allocator_config": os.environ["PYTORCH_CUDA_ALLOC_CONF"],
        "preregistration_lock_sha256": lock["lock_sha256"],
        "stage_a_sentinel_sha256": authorization.sha256,
        "data_contract_sha256": args.data_contract_sha256,
        "execution_requested": bool(args.execute),
    }
    if not args.execute:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return preflight

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    preflight_path = EXPERIMENT_ROOT / "manifests" / "formal_preflight.json"
    if preflight_path.exists():
        existing = json.loads(preflight_path.read_text(encoding="utf-8"))
        if existing != preflight:
            raise FileExistsError("existing formal preflight differs from this launch")
    else:
        _atomic_json(preflight_path, preflight)

    completed: dict[tuple[int, int], dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    for assignment in assignments:
        key = (int(assignment["seed_base"]), int(assignment["fold"]))
        path = _cell_dir(output_root, *key)
        if path.exists():
            completed[key] = _validate_complete_cell(
                path, *key, str(lock["lock_sha256"])
            )
        else:
            pending.append(assignment)

    active = ActiveProcesses()
    train_script = EXPERIMENT_ROOT / "scripts" / "train_cell.py"
    bucket = {
        device: [assignment for assignment in pending if assignment["device"] == device]
        for device in devices
    }

    def device_worker(device: str) -> list[tuple[tuple[int, int], dict[str, Any]]]:
        local: list[tuple[tuple[int, int], dict[str, Any]]] = []
        for assignment in bucket[device]:
            seed = int(assignment["seed_base"])
            fold = int(assignment["fold"])
            destination = _cell_dir(output_root, seed, fold)
            command = [
                sys.executable,
                str(train_script),
                "--seed-base",
                str(seed),
                "--fold",
                str(fold),
                "--output-dir",
                str(destination),
                "--device",
                device,
                "--workers",
                str(args.workers),
                "--stage-a-sentinel",
                str(args.stage_a_sentinel),
                "--data-contract",
                str(args.data_contract),
                "--data-contract-sha256",
                str(args.data_contract_sha256),
            ]
            log_path = (
                EXPERIMENT_ROOT / "logs" / f"train_seed_{seed}_fold_{fold}.private.log"
            )
            active.run(command, log_path)
            local.append(
                (
                    (seed, fold),
                    _validate_complete_cell(
                        destination, seed, fold, str(lock["lock_sha256"])
                    ),
                )
            )
        return local

    try:
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            futures = [executor.submit(device_worker, device) for device in devices]
            for future in as_completed(futures):
                for key, payload in future.result():
                    completed[key] = payload
    except BaseException:
        active.abort()
        raise
    if set(completed) != set(cells):
        raise RuntimeError("formal matrix ended without all ten cells")
    completion = {
        "schema_version": 1,
        "status": "COMPLETE",
        "arm": "A1_PATCH3",
        "run_count": 10,
        "seeds": list(SEEDS),
        "folds": list(FOLDS),
        "preregistration_lock_sha256": lock["lock_sha256"],
        "all_training_pcr_free": all(
            bool(payload["training_is_pcr_free"]) for payload in completed.values()
        ),
        "all_test_blind_selection": all(
            not bool(payload["test_data_used"]) for payload in completed.values()
        ),
        "all_cells_finite_noncollapsed": all(
            bool(payload["finite_noncollapsed_selection"])
            for payload in completed.values()
        ),
        "cells": [
            {
                "seed_base": seed,
                "fold": fold,
                "effective_seed": int(completed[(seed, fold)]["effective_seed"]),
                "selected_epoch": int(completed[(seed, fold)]["selected_epoch"]),
                "selection_sha256": completed[(seed, fold)]["selection_sha256"],
                "selected_checkpoint_sha256": completed[(seed, fold)][
                    "selected_checkpoint_sha256"
                ],
                "wall_seconds": float(completed[(seed, fold)]["wall_seconds"]),
            }
            for seed, fold in cells
        ],
    }
    completion_path = EXPERIMENT_ROOT / "metrics" / "formal_matrix_complete.json"
    if completion_path.exists():
        existing = json.loads(completion_path.read_text(encoding="utf-8"))
        if existing != completion:
            raise FileExistsError("existing formal matrix completion differs")
    else:
        _atomic_json(completion_path, completion)
    print(json.dumps(completion, indent=2, sort_keys=True))
    return completion


if __name__ == "__main__":
    main()

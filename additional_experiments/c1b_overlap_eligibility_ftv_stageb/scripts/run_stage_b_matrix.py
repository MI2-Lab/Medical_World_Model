#!/usr/bin/env python3
"""Launch the exact 2 seeds x 5 folds x 4 Stage B matrix after preflight."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c1b_stage_b.cli import add_data_contract_arguments, add_gate_arguments, authorize, data_paths  # noqa: E402
from c1b_stage_b.contracts import ARMS, FOLDS, SEED_BASES, validate_batch_contract  # noqa: E402
from c1b_stage_b.inputs import load_stage_b_data  # noqa: E402
from c1b_stage_b.matrix import (  # noqa: E402
    GROUP_ARM_ORDER,
    MatrixCell,
    MatrixExecutionError,
    build_matrix_groups,
    build_train_command,
    execute_matrix_groups,
    parse_multi_devices,
)


class ActiveSubprocesses:
    """Own exact training process groups so first failure can stop peers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: dict[int, subprocess.Popen[bytes]] = {}
        self._aborted = False

    def run(self, command: Sequence[str]) -> None:
        with self._lock:
            if self._aborted:
                raise MatrixExecutionError("matrix was already aborted")
            try:
                process = subprocess.Popen(list(command), start_new_session=True)
            except BaseException:
                self._aborted = True
                raise
            self._active[process.pid] = process
        return_code = process.wait()
        with self._lock:
            self._active.pop(process.pid, None)
            if return_code:
                # Close the launch race before releasing the lock: no other
                # device worker may start a dependent or subsequent run.
                self._aborted = True
                active = tuple(self._active.values())
        if return_code:
            self._terminate(active)
            raise subprocess.CalledProcessError(return_code, list(command))

    def abort(self) -> None:
        with self._lock:
            self._aborted = True
            active = tuple(self._active.values())
        self._terminate(active)

    @staticmethod
    def _terminate(active: Sequence[subprocess.Popen[bytes]]) -> None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_gate_arguments(parser)
    add_data_contract_arguments(parser)
    parser.add_argument("--output-root", type=Path, required=True)
    device = parser.add_mutually_exclusive_group()
    device.add_argument(
        "--device",
        help="Single-device compatibility mode (default: cuda).",
    )
    device.add_argument(
        "--devices",
        help="Comma-separated explicit CUDA devices, e.g. cuda:0,cuda:1,cuda:2.",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--physical-batch-size", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument(
        "--global-fallback-restart",
        action="store_true",
        help="Required for the all-arm 2/16 restart; use a new empty output root.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Without this explicit flag, perform gated read-only preflight only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    authorization = authorize(args)
    from c1b_stage_b.upstream import paired_initialization_report
    import torch

    devices = parse_multi_devices(args.devices) if args.devices else (args.device or "cuda",)
    cuda_devices = [device for device in devices if device.startswith("cuda")]
    if cuda_devices and not torch.cuda.is_available():
        raise RuntimeError("one or more CUDA devices were requested but CUDA is unavailable")
    cuda_count = torch.cuda.device_count()
    for device in cuda_devices:
        if ":" in device and int(device.split(":", 1)[1]) >= cuda_count:
            raise ValueError(f"requested device {device} but only {cuda_count} CUDA devices exist")
    validate_batch_contract(args.physical_batch_size, args.accumulation_steps)
    if (args.physical_batch_size, args.accumulation_steps) == (2, 16):
        if not args.global_fallback_restart:
            raise ValueError("2/16 is allowed only as an explicit global all-four-arm restart")
    elif args.global_fallback_restart:
        raise ValueError("--global-fallback-restart is valid only with physical=2/accumulation=16")
    paths = data_paths(args)
    # Formal launch/preflight performs the expensive SHA-256 pass once in this
    # parent process.  The forty child runs and their data workers then rely on
    # the hash-pinned manifest plus same-fd size/mtime checks.
    data = load_stage_b_data(paths, authorization, verify_cache_files=True)
    output_root = args.output_root.resolve()
    groups = build_matrix_groups(output_root, devices)
    preflight = {
        "schema_version": 1,
        "stage_a_sentinel_sha256": authorization.sha256,
        "matrix": {"arms": list(ARMS), "seed_bases": list(SEED_BASES), "folds": list(FOLDS)},
        "batch_contract": {
            "effective": 32,
            "physical": args.physical_batch_size,
            "accumulation": args.accumulation_steps,
            "global_for_all_arms": True,
        },
        "scheduler": {
            "devices": list(devices),
            "parallel_device_workers": len(devices),
            "seed_fold_groups": len(groups),
            "runs": sum(len(group.cells) for group in groups),
            "group_assignment": [
                {
                    "group_index": group.index,
                    "seed_base": group.seed_base,
                    "fold": group.fold,
                    "device": group.device,
                    "arm_order": [cell.arm for cell in group.cells],
                }
                for group in groups
            ],
            "fail_fast_active_process_group_termination": True,
        },
        "fold_eligible_patients": int(data.folds["patient_id"].nunique()),
        "upstream_authorized_train_only_patients": len(data.train_only_ids),
        "matched_eligible_patients": len(data.eligibility.eligible_ids),
        "fold_split_counts": data.provenance["fold_split_counts"],
        "cache_preflight": {
            "full_sha256_verified": True,
            "size_and_mtime_ns_pinned": True,
            "archive_envelopes_verified_without_sidecar_materialization": True,
        },
        "paired_initialization": {
            str(seed + fold): paired_initialization_report(seed + fold)
            for seed in SEED_BASES
            for fold in FOLDS
        },
        "execution_requested": bool(args.execute),
    }
    if not args.execute:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            "matrix output root must be empty; a 2/16 fallback is a new global restart, not a partial resume"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "matrix_preflight.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    train_script = ROOT / "scripts" / "train_stage_b.py"
    processes = ActiveSubprocesses()

    def run_cell(cell: MatrixCell) -> None:
        command = build_train_command(
            cell,
            python_executable=sys.executable,
            train_script=train_script,
            stage_a_sentinel=args.stage_a_sentinel,
            data_contract=args.data_contract,
            data_contract_sha256=args.data_contract_sha256,
            physical_batch_size=args.physical_batch_size,
            accumulation_steps=args.accumulation_steps,
            workers=args.workers,
            global_fallback_restart=args.global_fallback_restart,
        )
        processes.run(command)

    try:
        completed = execute_matrix_groups(groups, devices, run_cell, processes.abort)
    except BaseException as error:
        processes.abort()
        raise RuntimeError(
            "Stage B matrix stopped at first failure and preserved the output root. "
            "If this was CUDA OOM under 4/8, restart all forty runs in a new empty "
            "root with 2/16 and --global-fallback-restart."
        ) from error
    completion = {
        "schema_version": 1,
        "status": "COMPLETE",
        "run_count": len(completed),
        "devices": list(devices),
        "stage_a_sentinel_sha256": authorization.sha256,
        "batch_contract": preflight["batch_contract"],
        "runs": [
            {
                "seed_base": cell.seed_base,
                "fold": cell.fold,
                "arm": cell.arm,
                "device": cell.device,
                "selection_path": str(cell.output_dir / "selection.json"),
            }
            for cell in sorted(
                completed,
                key=lambda value: (
                    value.group_index,
                    GROUP_ARM_ORDER.index(value.arm),
                ),
            )
        ],
    }
    (output_root / "matrix_complete.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

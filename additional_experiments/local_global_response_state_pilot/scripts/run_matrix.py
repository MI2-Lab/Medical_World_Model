#!/usr/bin/env python3
"""Preflight or launch the formal 2-seed x 5-fold x 6-arm pilot matrix."""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import threading
import time
from typing import Mapping, Sequence


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
SRC_ROOT = EXPERIMENT_ROOT / "src"
SCRIPTS_ROOT = EXPERIMENT_ROOT / "scripts"
SEALED_EXPERIMENT_ROOT = (
    REPO_ROOT / "additional_experiments" / "c1b_overlap_eligibility_ftv_stageb"
)
DEFAULT_STAGE_A_SENTINEL = SEALED_EXPERIMENT_ROOT / "STAGE_A_GO.json"
DEFAULT_STAGE_A_SENTINEL_SHA256 = (
    "0b2c9e0af63ce8806525fb15ac9a27f6ab525b0259ccf16001981ca5091afbdb"
)
DEFAULT_DATA_CONTRACT = (
    SEALED_EXPERIMENT_ROOT / "manifests" / "stage_b_data_contract.private.json"
)
DEFAULT_DATA_CONTRACT_SHA256 = (
    "dd22f130043863d4fce8956061fca389894a31874567ed7929e139f32ff5ab27"
)
CHECKPOINT_ROOT = EXPERIMENT_ROOT / "checkpoints"
DEFAULT_OUTPUT_ROOT = EXPERIMENT_ROOT / "checkpoints" / "formal_4x8"
DEFAULT_DEVICES = ("cuda:0", "cuda:1", "cuda:2")
SAFE_TAG = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
for source in (SRC_ROOT, SCRIPTS_ROOT):
    value = str(source.resolve())
    while value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)

from freeze_preregistration import verify as verify_preregistration  # noqa: E402


class ActiveSubprocesses:
    """Own training process groups so the first failure stops active peers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: dict[int, subprocess.Popen[bytes]] = {}
        self._aborted = False

    def run(self, command: Sequence[str]) -> None:
        with self._lock:
            if self._aborted:
                raise RuntimeError("pilot matrix was already aborted")
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


def _atomic_private_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary).replace(path)
        path.chmod(0o600)
    finally:
        Path(temporary).unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage-a-sentinel", type=Path, default=DEFAULT_STAGE_A_SENTINEL
    )
    parser.add_argument(
        "--stage-a-sentinel-sha256", default=DEFAULT_STAGE_A_SENTINEL_SHA256
    )
    parser.add_argument("--data-contract", type=Path, default=DEFAULT_DATA_CONTRACT)
    parser.add_argument(
        "--data-contract-sha256", default=DEFAULT_DATA_CONTRACT_SHA256
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--devices", default=",".join(DEFAULT_DEVICES))
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Without this flag, perform gated read-only preflight only.",
    )
    return parser.parse_args()


def _validate_cuda_devices(devices: Sequence[str]) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("formal pilot execution requires CUDA")
    count = torch.cuda.device_count()
    for device in devices:
        if int(device.split(":", 1)[1]) >= count:
            raise ValueError(f"requested {device}, but only {count} CUDA devices exist")


def main() -> None:
    os.umask(0o077)
    preregistration = verify_preregistration()
    import lg_response_pilot.matrix as pilot_matrix_module
    import lg_response_pilot.model as pilot_model_module
    import lg_response_pilot.security as pilot_security
    import lg_response_pilot.training as pilot_training_module

    if Path(str(getattr(pilot_security, "__file__", ""))).resolve() != (
        SRC_ROOT / "lg_response_pilot" / "security.py"
    ).resolve():
        raise ImportError("pilot security module was shadowed before formal execution")
    for module, label in (
        (pilot_matrix_module, "pilot matrix"),
        (pilot_model_module, "pilot model"),
        (pilot_training_module, "pilot training"),
    ):
        pilot_security.require_module_within(
            module, SRC_ROOT / "lg_response_pilot", label=label
        )
    from lg_response_pilot.matrix import (
        FORMAL_ARM_ORDER,
        FORMAL_DEVICES,
        MatrixCell,
        build_matrix_groups,
        build_train_command,
        execute_matrix_groups,
        parse_three_devices,
    )
    from lg_response_pilot.model import paired_initialization_report
    from lg_response_pilot.security import (
        claim_private_directory,
        require_canonical_file,
        resolve_contained_path,
    )
    from lg_response_pilot.training import (
        FOLDS,
        SEALED_SRC,
        SEED_BASES,
        file_sha256,
        verify_sealed_stage_b_sources,
    )

    if tuple(FORMAL_DEVICES) != DEFAULT_DEVICES:
        raise RuntimeError("formal three-device contract drifted")
    sealed_value = str(SEALED_SRC.resolve())
    while sealed_value in sys.path:
        sys.path.remove(sealed_value)
    sys.path.insert(0, sealed_value)
    preregistration_evidence = {
        "status": preregistration["status"],
        "lock_sha256": preregistration["lock_sha256"],
        "config_sha256": preregistration["config_sha256"],
        "verified_code_files": preregistration["verified_code_files"],
    }
    args = parse_args()
    locked_upstream = preregistration["upstream_sha256"]
    args.stage_a_sentinel = require_canonical_file(
        args.stage_a_sentinel,
        DEFAULT_STAGE_A_SENTINEL,
        args.stage_a_sentinel_sha256,
        locked_upstream[str(DEFAULT_STAGE_A_SENTINEL.relative_to(REPO_ROOT))],
        label="Stage-A sentinel",
    )
    args.data_contract = require_canonical_file(
        args.data_contract,
        DEFAULT_DATA_CONTRACT,
        args.data_contract_sha256,
        locked_upstream[str(DEFAULT_DATA_CONTRACT.relative_to(REPO_ROOT))],
        label="Stage-B data contract",
    )
    output_root = resolve_contained_path(
        args.output_root,
        CHECKPOINT_ROOT,
        label="formal checkpoint output root",
    )
    output_relative = output_root.relative_to(CHECKPOINT_ROOT.resolve())
    if len(output_relative.parts) != 1 or SAFE_TAG.fullmatch(output_relative.name) is None:
        raise ValueError("formal checkpoint root must be checkpoints/<one-safe-tag>")
    devices = parse_three_devices(args.devices)
    _validate_cuda_devices(devices)
    sealed_hashes = verify_sealed_stage_b_sources()

    from c1b_stage_b.gate import require_stage_a_go
    from c1b_stage_b.inputs import StageBDataPaths, load_stage_b_data

    import c1b_stage_b.gate as sealed_gate_module
    import c1b_stage_b.inputs as sealed_inputs_module

    pilot_security.require_module_within(
        sealed_gate_module, SEALED_SRC, label="sealed Stage-B gate"
    )
    pilot_security.require_module_within(
        sealed_inputs_module, SEALED_SRC, label="sealed Stage-B inputs"
    )

    authorization = require_stage_a_go(args.stage_a_sentinel)
    paths = StageBDataPaths.load(args.data_contract, args.data_contract_sha256)
    # The parent performs the expensive content-hash verification exactly once;
    # child cells use the pinned manifest and same-fd size/mtime checks.
    data = load_stage_b_data(paths, authorization, verify_cache_files=True)
    groups = build_matrix_groups(output_root, devices)
    paired = {
        str(seed + fold): paired_initialization_report(seed + fold)
        for seed in SEED_BASES
        for fold in FOLDS
    }
    preflight: dict[str, object] = {
        "schema_version": 1,
        "status": "PREFLIGHT_PASS",
        "preregistration": preregistration_evidence,
        "stage_a_sentinel_sha256": authorization.sha256,
        "data_contract_sha256": args.data_contract_sha256,
        "config_sha256": preregistration["config_sha256"],
        "matrix": {
            "arms": list(FORMAL_ARM_ORDER),
            "seed_bases": list(SEED_BASES),
            "folds": list(FOLDS),
            "run_count": 60,
            "group_count": 10,
        },
        "batch_contract": {
            "physical": 4,
            "accumulation": 8,
            "effective": 32,
            "sigreg": "exact_logical_B32",
            "clip_optimizer_ema": "once_per_logical_batch",
        },
        "scheduler": {
            "devices": list(devices),
            "parallel_device_workers": 3,
            "one_sequential_seed_fold_stream_per_device": True,
            "within_group_arm_order": list(FORMAL_ARM_ORDER),
            "fail_fast_active_process_group_termination": True,
            "partial_resume": False,
        },
        "cohort_counts": {
            "fold_eligible_patients": int(data.folds["patient_id"].nunique()),
            "authorized_train_only_patients": len(data.train_only_ids),
            "matched_eligible_patients": len(data.eligibility.eligible_ids),
        },
        "cache_preflight": {
            "full_sha256_verified": True,
            "manifest_size_mtime_pinned": True,
        },
        "sealed_stage_b_source_sha256": sealed_hashes,
        "pilot_code_sha256": {
            "model.py": file_sha256(SRC_ROOT / "lg_response_pilot" / "model.py"),
            "pooling.py": file_sha256(SRC_ROOT / "lg_response_pilot" / "pooling.py"),
            "training.py": file_sha256(SRC_ROOT / "lg_response_pilot" / "training.py"),
            "matrix.py": file_sha256(SRC_ROOT / "lg_response_pilot" / "matrix.py"),
            "train_cell.py": file_sha256(EXPERIMENT_ROOT / "scripts" / "train_cell.py"),
            "run_matrix.py": file_sha256(Path(__file__)),
        },
        "paired_initialization": paired,
        "execution_requested": bool(args.execute),
    }
    if not args.execute:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return
    claim_private_directory(
        output_root,
        CHECKPOINT_ROOT,
        label="formal checkpoint output root",
    )
    _atomic_private_json(output_root / "matrix_preflight.json", preflight)
    train_script = EXPERIMENT_ROOT / "scripts" / "train_cell.py"
    processes = ActiveSubprocesses()

    def run_cell(cell: MatrixCell) -> None:
        command = build_train_command(
            cell,
            python_executable=sys.executable,
            train_script=train_script,
            stage_a_sentinel=args.stage_a_sentinel,
            stage_a_sentinel_sha256=args.stage_a_sentinel_sha256,
            data_contract=args.data_contract,
            data_contract_sha256=args.data_contract_sha256,
            preregistration_lock_sha256=preregistration["lock_sha256"],
        )
        processes.run(command)

    try:
        completed = execute_matrix_groups(groups, devices, run_cell, processes.abort)
    except BaseException as error:
        processes.abort()
        raise RuntimeError(
            "pilot matrix stopped at first failure; the private partial root was "
            "preserved and must not be resumed or overwritten"
        ) from error
    completion: dict[str, object] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "run_count": len(completed),
        "group_count": len(groups),
        "devices": list(devices),
        "stage_a_sentinel_sha256": authorization.sha256,
        "data_contract_sha256": args.data_contract_sha256,
        "config_sha256": preregistration["config_sha256"],
        "preregistration": preregistration_evidence,
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
                    FORMAL_ARM_ORDER.index(value.arm),
                ),
            )
        ],
    }
    _atomic_private_json(output_root / "matrix_complete.json", completion)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

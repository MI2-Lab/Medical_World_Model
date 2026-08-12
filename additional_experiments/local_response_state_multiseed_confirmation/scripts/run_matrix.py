#!/usr/bin/env python3
"""Preflight or launch the formal 5-seed x 5-fold x 4-arm confirmation."""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
import os
import platform
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
PRIVATE_INPUT_REPO_ROOT_ENV = "MWM_PRIVATE_INPUT_REPO_ROOT"
DATA_CONTRACT_REPOSITORY_RELATIVE = Path(
    "additional_experiments/c1b_overlap_eligibility_ftv_stageb/"
    "manifests/stage_b_data_contract.private.json"
)
DATA_CONTRACT_LOCK_KEY = DATA_CONTRACT_REPOSITORY_RELATIVE.as_posix()


def _resolve_default_data_contract(
    environ: Mapping[str, str] = os.environ,
) -> Path:
    configured = str(environ.get(PRIVATE_INPUT_REPO_ROOT_ENV, "")).strip()
    private_repository_root = (
        Path(configured).expanduser().resolve() if configured else REPO_ROOT.resolve()
    )
    return (private_repository_root / DATA_CONTRACT_REPOSITORY_RELATIVE).resolve()


DEFAULT_DATA_CONTRACT = _resolve_default_data_contract()
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
                raise RuntimeError("confirmation matrix was already aborted")
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
        raise RuntimeError("formal confirmation execution requires CUDA")
    count = torch.cuda.device_count()
    for device in devices:
        if int(device.split(":", 1)[1]) >= count:
            raise ValueError(f"requested {device}, but only {count} CUDA devices exist")


def _runtime_provenance(devices: Sequence[str]) -> dict[str, object]:
    """Capture the execution runtime without changing any training behavior."""

    import torch

    _validate_cuda_devices(devices)
    requested = []
    for device in devices:
        index = int(device.split(":", 1)[1])
        requested.append(
            {
                "device": device,
                "name": torch.cuda.get_device_name(index),
                "compute_capability": list(torch.cuda.get_device_capability(index)),
            }
        )
    return {
        "python_version": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "requested_gpus": requested,
    }


def _validate_initialization_reports(
    reports: Mapping[str, object], expected_effective_seeds: Sequence[int]
) -> None:
    expected_keys = tuple(str(seed) for seed in expected_effective_seeds)
    if tuple(reports) != expected_keys:
        raise ValueError("paired initialization report seed inventory/order drifted")
    for key in expected_keys:
        report = reports[key]
        if not isinstance(report, Mapping):
            raise TypeError("paired initialization report must be a mapping")
        if report.get("effective_seed") != int(key):
            raise ValueError("paired initialization report effective seed mismatch")
        if report.get("arms") != ["GAP0", "GAP3", "LOCAL0", "LOCAL3"]:
            raise ValueError("paired initialization report arm order drifted")
        per_arm = report.get("per_arm")
        if not isinstance(per_arm, Mapping) or tuple(per_arm) != (
            "GAP0",
            "GAP3",
            "LOCAL0",
            "LOCAL3",
        ):
            raise ValueError("paired initialization per-arm evidence drifted")
        checks = report.get("checks")
        if not isinstance(checks, Mapping) or not checks or not all(
            value is True for value in checks.values()
        ):
            raise AssertionError("paired initialization report contains a failed check")
        shared = report.get("shared_initialization_sha256")
        if report.get("architecture_pairs") != {
            "GAP0_LOCAL0": shared,
            "GAP3_LOCAL3": shared,
        }:
            raise ValueError("paired GAP/LOCAL initialization evidence drifted")


def main() -> None:
    os.umask(0o077)
    preregistration = verify_preregistration()
    # Historical package name retained for exact pilot-checkpoint compatibility.
    import lg_response_pilot.matrix as pilot_matrix_module
    import lg_response_pilot.model as pilot_model_module
    import lg_response_pilot.security as pilot_security
    import lg_response_pilot.training as pilot_training_module

    if Path(str(getattr(pilot_security, "__file__", ""))).resolve() != (
        SRC_ROOT / "lg_response_pilot" / "security.py"
    ).resolve():
        raise ImportError(
            "confirmation security module was shadowed before formal execution"
        )
    for module, label in (
        (pilot_matrix_module, "confirmation matrix"),
        (pilot_model_module, "confirmation model"),
        (pilot_training_module, "confirmation training"),
    ):
        pilot_security.require_module_within(
            module, SRC_ROOT / "lg_response_pilot", label=label
        )
    from lg_response_pilot.matrix import (
        FORMAL_ARM_ORDER,
        FORMAL_CELL_COUNT,
        FORMAL_DEVICES,
        FORMAL_FOLDS,
        FORMAL_GROUP_COUNT,
        FORMAL_SEED_BASES,
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
    if tuple(FORMAL_ARM_ORDER) != ("GAP0", "GAP3", "LOCAL0", "LOCAL3"):
        raise RuntimeError("formal confirmation arm order drifted")
    if tuple(SEED_BASES) != tuple(FORMAL_SEED_BASES) or tuple(FOLDS) != tuple(
        FORMAL_FOLDS
    ):
        raise RuntimeError("formal confirmation seed/fold inventory drifted")
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
        locked_upstream[DATA_CONTRACT_LOCK_KEY],
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
    runtime_provenance = _runtime_provenance(devices)
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
    expected_effective_seeds = tuple(
        seed + fold for seed in FORMAL_SEED_BASES for fold in FORMAL_FOLDS
    )
    _validate_initialization_reports(paired, expected_effective_seeds)
    preflight: dict[str, object] = {
        "schema_version": 1,
        "status": "PREFLIGHT_PASS",
        "preregistration": preregistration_evidence,
        "stage_a_sentinel_sha256": authorization.sha256,
        "data_contract_sha256": args.data_contract_sha256,
        "config_sha256": preregistration["config_sha256"],
        "runtime_provenance": runtime_provenance,
        "matrix": {
            "arms": list(FORMAL_ARM_ORDER),
            "seed_bases": list(SEED_BASES),
            "folds": list(FOLDS),
            "run_count": FORMAL_CELL_COUNT,
            "group_count": FORMAL_GROUP_COUNT,
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
        "confirmation_code_sha256": {
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
            "confirmation matrix stopped at first failure; the private partial root was "
            "preserved and must not be resumed or overwritten"
        ) from error
    if len(completed) != FORMAL_CELL_COUNT or len(groups) != FORMAL_GROUP_COUNT:
        raise RuntimeError("completed confirmation matrix cardinality drifted")
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
        "runtime_provenance": preflight["runtime_provenance"],
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

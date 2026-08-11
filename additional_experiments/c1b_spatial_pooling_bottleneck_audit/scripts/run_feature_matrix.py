#!/usr/bin/env python3
"""Preflight or execute the frozen 40-cell final-spatial feature matrix."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c1b_spatial_audit.contracts import (  # noqa: E402
    ARMS,
    EXPERIMENT_ROOT,
    checkpoint_path,
    file_sha256,
    relative,
)
from c1b_spatial_audit.exporter import (  # noqa: E402
    C1B_POOLINGS,
    LEGACY_POOLINGS,
    feature_asset_path,
    feature_metadata_path,
    validate_feature_export,
)
from c1b_spatial_audit.runtime import (  # noqa: E402
    load_stage_b_bundle,
    verify_preregistration,
)


@dataclass(frozen=True)
class MatrixCell:
    index: int
    seed_base: int
    arm: str
    fold: int
    device: str


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        temporary_path.chmod(0o600)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _devices(value: str) -> tuple[str, ...]:
    devices = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("--devices must contain unique CUDA devices")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("formal matrix requires CUDA")
    resolved: list[str] = []
    for value in devices:
        device = torch.device(value)
        if device.type != "cuda" or device.index is None:
            raise ValueError("matrix devices must be explicit cuda:<index> values")
        if device.index < 0 or device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device is unavailable: {device}")
        resolved.append(str(device))
    return tuple(resolved)


def build_matrix(devices: Sequence[str]) -> tuple[MatrixCell, ...]:
    """Use the immutable old formal round-robin order and one stream/device."""

    cells: list[MatrixCell] = []
    for seed in (2026, 3026):
        for fold in range(5):
            for arm in ARMS:
                index = len(cells)
                cells.append(
                    MatrixCell(index, seed, arm, fold, str(devices[index % len(devices)]))
                )
    if len(cells) != 40:
        raise AssertionError("formal matrix must contain exactly forty cells")
    return tuple(cells)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-root", type=Path, default=EXPERIMENT_ROOT / "features"
    )
    parser.add_argument(
        "--sidecar",
        type=Path,
        default=EXPERIMENT_ROOT / "manifests" / "audit_sidecars.private.npz",
    )
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Without this flag only a read-only, hash-complete preflight is printed.",
    )
    return parser.parse_args()


def _inventory(lock: Mapping[str, Any], cells: Sequence[MatrixCell]) -> None:
    for cell in cells:
        key = f"seed_{cell.seed_base}/{cell.arm}/fold_{cell.fold}"
        checkpoint = checkpoint_path(cell.seed_base, cell.arm, cell.fold).resolve()
        frozen = lock["selected_checkpoints"][key]
        if relative(checkpoint) != frozen["path"] or file_sha256(checkpoint) != frozen["sha256"]:
            raise ValueError(f"formal checkpoint drifted: {key}")
        reference = lock["formal_p0_references"][key]
        for path_field, hash_field in (
            ("feature_path", "feature_sha256"),
            ("feature_metadata_path", "feature_metadata_sha256"),
        ):
            path = (ROOT.parents[1] / reference[path_field]).resolve()
            if file_sha256(path) != reference[hash_field]:
                raise ValueError(f"formal P0 reference drifted: {key}/{path_field}")


def _terminate(processes: Mapping[int, subprocess.Popen[str]]) -> None:
    for process in tuple(processes.values()):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def _execute(
    cells: Sequence[MatrixCell],
    *,
    sidecar: Path,
    feature_root: Path,
    exporter_script: Path,
    log_root: Path,
) -> None:
    buckets: dict[str, list[MatrixCell]] = {}
    for cell in cells:
        buckets.setdefault(cell.device, []).append(cell)
    stop = threading.Event()
    lock = threading.Lock()
    active: dict[int, subprocess.Popen[str]] = {}

    def run_device(device: str, assigned: Sequence[MatrixCell]) -> None:
        for cell in assigned:
            if stop.is_set():
                return
            log_path = (
                log_root
                / f"seed_{cell.seed_base}"
                / cell.arm
                / f"fold_{cell.fold}.private.log"
            )
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.parent.chmod(0o700)
            command = [
                sys.executable,
                str(exporter_script),
                "--checkpoint",
                str(checkpoint_path(cell.seed_base, cell.arm, cell.fold)),
                "--arm",
                cell.arm,
                "--seed-base",
                str(cell.seed_base),
                "--fold",
                str(cell.fold),
                "--sidecar",
                str(sidecar),
                "--feature-root",
                str(feature_root),
                "--device",
                device,
                "--batch-size",
                "4",
                "--workers",
                "2",
            ]
            with log_path.open("w", encoding="utf-8") as log:
                log_path.chmod(0o600)
                process = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                with lock:
                    active[cell.index] = process
                return_code = process.wait()
                with lock:
                    active.pop(cell.index, None)
                if return_code != 0:
                    stop.set()
                    with lock:
                        snapshot = dict(active)
                    _terminate(snapshot)
                    raise RuntimeError(
                        f"feature cell failed ({cell.seed_base}/{cell.arm}/fold{cell.fold}); "
                        f"see {log_path}"
                    )

    with ThreadPoolExecutor(max_workers=len(buckets)) as executor:
        futures = [
            executor.submit(run_device, device, assigned)
            for device, assigned in buckets.items()
        ]
        for future in as_completed(futures):
            future.result()
    if stop.is_set():
        raise RuntimeError("formal feature matrix terminated after a cell failure")


def main() -> None:
    args = parse_args()
    feature_root = args.feature_root.expanduser().resolve()
    expected_feature_root = (EXPERIMENT_ROOT / "features").resolve()
    if feature_root != expected_feature_root:
        raise ValueError("formal matrix feature root must be this experiment's features/")
    sidecar = args.sidecar.expanduser().resolve()
    expected_sidecar = (
        EXPERIMENT_ROOT / "manifests" / "audit_sidecars.private.npz"
    ).resolve()
    if sidecar != expected_sidecar:
        raise ValueError("formal matrix requires the canonical audit sidecar path")
    if not sidecar.is_file() or not sidecar.name.endswith(".private.npz"):
        raise FileNotFoundError("formal audit sidecar is missing")
    if sidecar.stat().st_mode & 0o077:
        raise PermissionError("formal audit sidecar must be owner-only")
    if args.batch_size != 4 or args.workers != 2:
        raise ValueError("formal matrix is frozen at batch_size=4/workers=2")
    devices = _devices(args.devices)
    cells = build_matrix(devices)
    lock_payload = verify_preregistration()
    _inventory(lock_payload, cells)
    authorization, _, data = load_stage_b_bundle(verify_cache_files=False)
    if int(data.folds["patient_id"].nunique()) != 808 or len(data.ftv) != 375:
        raise ValueError("live Stage-B population differs from 808 formal / 375 FTV")

    exporter_script = ROOT / "scripts" / "export_frozen_features.py"
    code_paths = {
        "matrix_driver": Path(__file__).resolve(),
        "feature_cli": exporter_script.resolve(),
        "exporter": (ROOT / "src" / "c1b_spatial_audit" / "exporter.py").resolve(),
        "pooling": (ROOT / "src" / "c1b_spatial_audit" / "pooling.py").resolve(),
        "runtime": (ROOT / "src" / "c1b_spatial_audit" / "runtime.py").resolve(),
        "contracts": (ROOT / "src" / "c1b_spatial_audit" / "contracts.py").resolve(),
    }
    code_sha256 = {name: file_sha256(path) for name, path in code_paths.items()}
    lock_path = ROOT / "PREREGISTRATION_LOCK.json"
    expected_asset_count = sum(
        len(LEGACY_POOLINGS if cell.arm.startswith("L") else C1B_POOLINGS)
        for cell in cells
    )
    if expected_asset_count != 180:
        raise AssertionError("formal pooling inventory must contain 180 numeric assets")
    preflight = {
        "schema_version": 1,
        "status": "PREFLIGHT_PASS",
        "stage": "final",
        "cell_count": len(cells),
        "expected_asset_count": expected_asset_count,
        "preregistration_lock_sha256": file_sha256(lock_path),
        "sidecar_path": str(sidecar),
        "sidecar_sha256": file_sha256(sidecar),
        "stage_a_sentinel_sha256": authorization.sha256,
        "feature_root": str(feature_root),
        "scheduler": {
            "devices": list(devices),
            "parallel_processes": len(devices),
            "one_sequential_stream_per_device": True,
            "batch_size": 4,
            "workers": 2,
            "fail_fast_process_group_termination": True,
        },
        "cell_inventory": [cell.__dict__ for cell in cells],
        "code_sha256": code_sha256,
        "python_executable": str(Path(sys.executable).resolve()),
        "execution_requested": bool(args.execute),
    }
    if not args.execute:
        print(json.dumps(preflight, indent=2, sort_keys=True, allow_nan=False))
        return

    final_root = feature_root / "final"
    if final_root.exists():
        raise FileExistsError(
            "refusing to resume or overwrite a formal final-spatial feature matrix"
        )
    feature_root.mkdir(parents=True, exist_ok=True)
    feature_root.chmod(0o700)
    claim_path = feature_root / "feature_export_claim.private.json"
    preflight_path = feature_root / "feature_export_preflight.private.json"
    completion_path = feature_root / "feature_export_complete.private.json"
    for path in (claim_path, preflight_path, completion_path):
        if path.exists():
            raise FileExistsError(f"formal feature control artifact already exists: {path}")
    _atomic_json(
        claim_path,
        {
            "schema_version": 1,
            "status": "CLAIMED",
            "stage": "final",
            "nonresumable": True,
            "cell_count": 40,
            "expected_asset_count": 180,
            "preregistration_lock_sha256": file_sha256(lock_path),
            "sidecar_sha256": file_sha256(sidecar),
            "matrix_driver_sha256": code_sha256["matrix_driver"],
        },
    )
    preflight["claim_sha256"] = file_sha256(claim_path)
    _atomic_json(preflight_path, preflight)

    log_root = (EXPERIMENT_ROOT / "logs" / "feature_matrix").resolve()
    try:
        _execute(
            cells,
            sidecar=sidecar,
            feature_root=feature_root,
            exporter_script=exporter_script,
            log_root=log_root,
        )
        if {name: file_sha256(path) for name, path in code_paths.items()} != code_sha256:
            raise RuntimeError("feature implementation changed during the formal matrix")
        if file_sha256(sidecar) != preflight["sidecar_sha256"]:
            raise RuntimeError("audit sidecar changed during the formal matrix")
        if file_sha256(lock_path) != preflight["preregistration_lock_sha256"]:
            raise RuntimeError("preregistration lock changed during the formal matrix")

        metadata_hashes: dict[str, str] = {}
        for cell in cells:
            poolings = LEGACY_POOLINGS if cell.arm.startswith("L") else C1B_POOLINGS
            for pooling in poolings:
                path = feature_asset_path(
                    feature_root,
                    cell.seed_base,
                    cell.arm,
                    cell.fold,
                    pooling,
                )
                validate_feature_export(
                    path,
                    expected_arm=cell.arm,
                    expected_seed_base=cell.seed_base,
                    expected_fold=cell.fold,
                    expected_pooling=pooling,
                    expected_patient_count=808,
                    verify_live_inputs=True,
                )
                metadata = feature_metadata_path(path)
                metadata_hashes[relative(metadata)] = file_sha256(metadata)
        if len(metadata_hashes) != expected_asset_count:
            raise RuntimeError("formal feature matrix did not produce all 180 assets")
    except BaseException as error:
        raise RuntimeError(
            "formal frozen feature matrix failed closed; partial private outputs are preserved "
            "and must not be resumed or overwritten"
        ) from error

    completion = {
        "schema_version": 1,
        "status": "COMPLETE",
        "stage": "final",
        "run_count": 40,
        "expected_asset_count": 180,
        "cell_count": 40,
        "feature_metadata_sha256": metadata_hashes,
        "preflight_sha256": file_sha256(preflight_path),
        "sidecar_sha256": file_sha256(sidecar),
        "preregistration_lock_sha256": file_sha256(lock_path),
    }
    _atomic_json(completion_path, completion)
    print(json.dumps(completion, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

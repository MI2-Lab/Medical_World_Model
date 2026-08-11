#!/usr/bin/env python3
"""Preflight or execute the trigger-authorized 40-cell S3 feature matrix."""

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
from c1b_spatial_audit.runtime import load_stage_b_bundle, verify_preregistration  # noqa: E402
from c1b_spatial_audit.s3_exporter import (  # noqa: E402
    S3_C1B_POOLINGS,
    S3_LEGACY_POOLINGS,
    s3_feature_asset_path,
    s3_feature_metadata_path,
    validate_s3_feature_export,
)
from c1b_spatial_audit.s3_sidecars import (  # noqa: E402
    FORMAL_PATIENT_COUNT,
    load_s3_sidecars,
)
from c1b_spatial_audit.s3_trigger import require_s3_trigger_authorization  # noqa: E402


EXPECTED_S3_ASSET_COUNT = 100


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
            json.dump(dict(payload), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.chmod(0o600)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _devices(value: str) -> tuple[str, ...]:
    devices = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("--devices must contain unique explicit CUDA devices")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("formal S3 matrix requires CUDA")
    resolved: list[str] = []
    for raw in devices:
        device = torch.device(raw)
        if (
            device.type != "cuda"
            or device.index is None
            or device.index < 0
            or device.index >= torch.cuda.device_count()
        ):
            raise ValueError(f"unavailable explicit CUDA device: {raw}")
        resolved.append(str(device))
    return tuple(resolved)


def build_matrix(devices: Sequence[str]) -> tuple[MatrixCell, ...]:
    cells: list[MatrixCell] = []
    for seed in (2026, 3026):
        for fold in range(5):
            for arm in ARMS:
                index = len(cells)
                cells.append(
                    MatrixCell(index, seed, arm, fold, str(devices[index % len(devices)]))
                )
    if len(cells) != 40:
        raise AssertionError("formal S3 matrix must contain exactly 40 checkpoint cells")
    return tuple(cells)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, default=ROOT / "features")
    parser.add_argument(
        "--sidecar",
        type=Path,
        default=ROOT / "manifests" / "audit_sidecars_s3.private.npz",
    )
    parser.add_argument(
        "--trigger-gate",
        type=Path,
        default=ROOT / "metrics" / "s3_trigger_authorization.json",
    )
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Without this flag, perform a read-only trigger/hash preflight.",
    )
    return parser.parse_args(argv)


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
            source = (ROOT.parents[1] / reference[path_field]).resolve()
            if file_sha256(source) != reference[hash_field]:
                raise ValueError(f"formal P0 patient-order reference drifted: {key}")


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
    trigger_gate: Path,
    sidecar: Path,
    feature_root: Path,
    exporter_script: Path,
    log_root: Path,
) -> None:
    buckets: dict[str, list[MatrixCell]] = {}
    for cell in cells:
        buckets.setdefault(cell.device, []).append(cell)
    stop = threading.Event()
    active_lock = threading.Lock()
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
                "--trigger-gate",
                str(trigger_gate),
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
                with active_lock:
                    active[cell.index] = process
                return_code = process.wait()
                with active_lock:
                    active.pop(cell.index, None)
                if return_code != 0:
                    stop.set()
                    with active_lock:
                        snapshot = dict(active)
                    _terminate(snapshot)
                    raise RuntimeError(
                        f"S3 feature cell failed ({cell.seed_base}/{cell.arm}/fold{cell.fold}); "
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
        raise RuntimeError("formal S3 feature matrix terminated after a cell failure")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    feature_root = args.feature_root.resolve()
    if feature_root != (EXPERIMENT_ROOT / "features").resolve():
        raise ValueError("formal S3 matrix feature root must be this audit's features/")
    sidecar = args.sidecar.resolve()
    trigger_gate = args.trigger_gate.resolve()
    if sidecar != (ROOT / "manifests" / "audit_sidecars_s3.private.npz").resolve():
        raise ValueError("formal S3 matrix requires the canonical S3 sidecar")
    if trigger_gate != (ROOT / "metrics" / "s3_trigger_authorization.json").resolve():
        raise ValueError("formal S3 matrix requires the canonical trigger gate")
    trigger = require_s3_trigger_authorization(trigger_gate, verify_live=True)
    if args.batch_size != 4 or args.workers != 2:
        raise ValueError("formal S3 matrix is frozen at batch_size=4/workers=2")
    devices = _devices(args.devices)
    cells = build_matrix(devices)
    lock = verify_preregistration()
    _inventory(lock, cells)
    authorization, _, data = load_stage_b_bundle(verify_cache_files=False)
    if int(data.folds["patient_id"].nunique()) != FORMAL_PATIENT_COUNT or len(data.ftv) != 375:
        raise ValueError("live Stage-B population differs from 808/375")
    from c1b_stage_b.data import make_splits

    split0 = make_splits(data.folds, 0, data.train_only_ids)
    patients0 = tuple(split0.train_primary + split0.val + split0.test)
    loaded_sidecars = load_s3_sidecars(sidecar, patients0, verify_live=True)
    del loaded_sidecars

    exporter_script = ROOT / "scripts" / "export_s3_frozen_features.py"
    code_paths = {
        "matrix_driver": Path(__file__).resolve(),
        "feature_cli": exporter_script.resolve(),
        "s3_exporter": (ROOT / "src/c1b_spatial_audit/s3_exporter.py").resolve(),
        "s3_sidecars": (ROOT / "src/c1b_spatial_audit/s3_sidecars.py").resolve(),
        "s3_trigger": (ROOT / "src/c1b_spatial_audit/s3_trigger.py").resolve(),
        "pooling": (ROOT / "src/c1b_spatial_audit/pooling.py").resolve(),
        "runtime": (ROOT / "src/c1b_spatial_audit/runtime.py").resolve(),
        "contracts": (ROOT / "src/c1b_spatial_audit/contracts.py").resolve(),
    }
    code_sha256 = {name: file_sha256(path) for name, path in code_paths.items()}
    expected_assets = sum(
        len(S3_LEGACY_POOLINGS if cell.arm.startswith("L") else S3_C1B_POOLINGS)
        for cell in cells
    )
    if expected_assets != EXPECTED_S3_ASSET_COUNT:
        raise AssertionError("formal S3 matrix must contain exactly 100 feature assets")
    lock_path = ROOT / "PREREGISTRATION_LOCK.json"
    sidecar_metadata = sidecar.with_name("audit_sidecars_s3.private.metadata.json")
    preflight = {
        "schema_version": 1,
        "status": "PREFLIGHT_PASS",
        "stage": "s3",
        "representation_contract": "raw_encoder_features2_pooled_64d_no_projection",
        "cell_count": len(cells),
        "expected_asset_count": expected_assets,
        "preregistration_lock_sha256": file_sha256(lock_path),
        "trigger_gate_sha256": file_sha256(trigger_gate),
        "trigger_status": trigger["status"],
        "sidecar_sha256": file_sha256(sidecar),
        "sidecar_metadata_sha256": file_sha256(sidecar_metadata),
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
        "execution_requested": bool(args.execute),
        "legacy_poracle": "NA_incomplete_source_authoritative_support_1488_of_1500",
    }
    if not args.execute:
        print(json.dumps(preflight, indent=2, sort_keys=True, allow_nan=False))
        return

    stage_root = feature_root / "s3"
    if stage_root.exists():
        raise FileExistsError("refusing to resume or overwrite a formal S3 feature matrix")
    stage_root.mkdir(parents=True, mode=0o700)
    claim_path = stage_root / "feature_export_claim.private.json"
    preflight_path = stage_root / "feature_export_preflight.private.json"
    completion_path = stage_root / "feature_export_complete.private.json"
    _atomic_json(
        claim_path,
        {
            "schema_version": 1,
            "status": "CLAIMED",
            "stage": "s3",
            "nonresumable": True,
            "cell_count": 40,
            "expected_asset_count": EXPECTED_S3_ASSET_COUNT,
            "trigger_gate_sha256": file_sha256(trigger_gate),
            "sidecar_sha256": file_sha256(sidecar),
            "matrix_driver_sha256": code_sha256["matrix_driver"],
        },
    )
    preflight["claim_sha256"] = file_sha256(claim_path)
    _atomic_json(preflight_path, preflight)
    log_root = (ROOT / "logs" / "s3_feature_matrix").resolve()
    try:
        _execute(
            cells,
            trigger_gate=trigger_gate,
            sidecar=sidecar,
            feature_root=feature_root,
            exporter_script=exporter_script,
            log_root=log_root,
        )
        if {name: file_sha256(path) for name, path in code_paths.items()} != code_sha256:
            raise RuntimeError("S3 feature implementation changed during formal matrix")
        immutable = {
            lock_path: preflight["preregistration_lock_sha256"],
            trigger_gate: preflight["trigger_gate_sha256"],
            sidecar: preflight["sidecar_sha256"],
            sidecar_metadata: preflight["sidecar_metadata_sha256"],
        }
        for path, digest in immutable.items():
            if file_sha256(path) != digest:
                raise RuntimeError(f"immutable S3 input changed during matrix: {path}")
        metadata_hashes: dict[str, str] = {}
        for cell in cells:
            poolings = S3_LEGACY_POOLINGS if cell.arm.startswith("L") else S3_C1B_POOLINGS
            for pooling in poolings:
                feature = s3_feature_asset_path(
                    feature_root, cell.seed_base, cell.arm, cell.fold, pooling
                )
                validate_s3_feature_export(
                    feature,
                    expected_arm=cell.arm,
                    expected_seed_base=cell.seed_base,
                    expected_fold=cell.fold,
                    expected_pooling=pooling,
                    verify_live_inputs=True,
                )
                metadata = s3_feature_metadata_path(feature)
                metadata_hashes[relative(metadata)] = file_sha256(metadata)
        if len(metadata_hashes) != EXPECTED_S3_ASSET_COUNT:
            raise RuntimeError("formal S3 matrix did not produce all 100 assets")
    except BaseException as error:
        raise RuntimeError(
            "formal S3 feature matrix failed closed; partial outputs are preserved and "
            "must not be resumed or overwritten"
        ) from error
    completion = {
        "schema_version": 1,
        "status": "COMPLETE",
        "stage": "s3",
        "representation_contract": "raw_encoder_features2_pooled_64d_no_projection",
        "run_count": 40,
        "cell_count": 40,
        "expected_asset_count": EXPECTED_S3_ASSET_COUNT,
        "feature_metadata_sha256": metadata_hashes,
        "preflight_sha256": file_sha256(preflight_path),
        "trigger_gate_sha256": file_sha256(trigger_gate),
        "sidecar_sha256": file_sha256(sidecar),
        "sidecar_metadata_sha256": file_sha256(sidecar_metadata),
        "preregistration_lock_sha256": file_sha256(lock_path),
    }
    _atomic_json(completion_path, completion)
    print(json.dumps(completion, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

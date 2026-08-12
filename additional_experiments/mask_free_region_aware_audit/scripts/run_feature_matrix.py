#!/usr/bin/env python3
"""Run or validate the exact resume-safe 20-cell regional feature matrix."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable).resolve()
sys.path.insert(0, str(ROOT / "scripts"))

from common import (  # noqa: E402
    COMPLETION_CELL_KEYS,
    COMPLETION_KEYS,
    COMPLETION_PATH,
    CONFIG_PATH,
    GEOMETRY_CONTRACT_PATH,
    LOCK_PATH,
    cell_key,
    cells,
    feature_path,
    file_sha256,
    load_config,
    metadata_path,
    private_directory,
    publish_json_once,
    require_owner_only,
    require_preregistration_lock,
    validate_feature_cell,
)


def _execute_device(
    device: str,
    queue: list[tuple[int, str, int]],
    queue_lock: threading.Lock,
    *,
    batch_size: int,
    workers: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    while True:
        with queue_lock:
            if not queue:
                break
            seed, arm, fold = queue.pop(0)
        key = cell_key(seed, arm, fold)
        output = feature_path(seed, arm, fold)
        log = ROOT / "logs" / f"export_seed{seed}_{arm}_fold{fold}.private.log"
        private_directory(log.parent)
        command = [
            str(PYTHON),
            str(ROOT / "scripts" / "export_features.py"),
            "--seed-base",
            str(seed),
            "--arm",
            arm,
            "--fold",
            str(fold),
            "--device",
            device,
            "--batch-size",
            str(batch_size),
            "--workers",
            str(workers),
        ]
        with log.open("w", encoding="utf-8") as stream:
            log.chmod(0o600)
            result = subprocess.run(
                command,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        record = {
            "cell": key,
            "device": device,
            "returncode": result.returncode,
            "output": str(output),
            "log": str(log),
        }
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if result.returncode != 0:
            raise RuntimeError(f"feature cell failed; inspect owner-only log {log}")
    return records


def _authenticate_parent_context() -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate every frozen input before the runner may create output."""

    config = load_config(CONFIG_PATH, verify_extraction_inputs=True)
    lock = require_preregistration_lock(config)
    return config, lock


def pending_cells(
    config: Mapping[str, Any], lock: Mapping[str, Any]
) -> list[tuple[int, str, int]]:
    """Return absent/incomplete cells; reject corrupt complete pairs."""

    pending: list[tuple[int, str, int]] = []
    for seed, arm, fold in cells():
        feature = feature_path(seed, arm, fold)
        metadata = metadata_path(seed, arm, fold)
        if feature.is_file() and metadata.is_file():
            validate_feature_cell(
                feature, config, lock, seed=seed, arm=arm, fold=fold
            )
        else:
            pending.append((seed, arm, fold))
    return pending


def validate_complete(
    *,
    config: Mapping[str, Any] | None = None,
    lock: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if config is None and lock is None:
        config, lock = _authenticate_parent_context()
    elif config is None or lock is None:
        raise ValueError("config and preregistration lock must be supplied together")
    if not GEOMETRY_CONTRACT_PATH.is_file():
        raise FileNotFoundError("public region occupancy contract is absent")
    geometry_sha256 = file_sha256(GEOMETRY_CONTRACT_PATH)
    complete: list[dict[str, Any]] = []
    for seed, arm, fold in cells():
        feature = feature_path(seed, arm, fold)
        metadata_source = metadata_path(seed, arm, fold)
        metadata = validate_feature_cell(
            feature, config, lock, seed=seed, arm=arm, fold=fold
        )
        if metadata["geometry_contract_sha256"] != geometry_sha256:
            raise ValueError("feature cells disagree on the public geometry contract")
        record = {
            "cell": cell_key(seed, arm, fold),
            "seed_base": seed,
            "arm": arm,
            "fold": fold,
            "feature_path": str(feature.resolve()),
            "feature_sha256": file_sha256(feature),
            "metadata_path": str(metadata_source.resolve()),
            "metadata_sha256": file_sha256(metadata_source),
            "patient_order_sha256": metadata["patient_order_sha256"],
        }
        if set(record) != set(COMPLETION_CELL_KEYS):
            raise AssertionError("completion cell schema drifted")
        complete.append(record)
    payload = {
        "schema_version": 1,
        "status": "COMPLETE",
        "experiment": "mask_free_region_aware_audit",
        "cell_count": len(complete),
        "config_sha256": file_sha256(CONFIG_PATH),
        "preregistration_lock_sha256": file_sha256(LOCK_PATH),
        "geometry_contract_sha256": geometry_sha256,
        "cells": complete,
    }
    if set(payload) != set(COMPLETION_KEYS) or payload["cell_count"] != 20:
        raise AssertionError("feature-matrix completion schema/count drifted")
    return payload


def validate_completion_marker(
    path: Path, expected: Mapping[str, Any]
) -> dict[str, Any]:
    source = require_owner_only(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or set(value) != set(COMPLETION_KEYS)
        or value != dict(expected)
    ):
        raise ValueError("feature-matrix marker differs from fresh validation")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    os.umask(0o077)
    args = parse_args(argv)
    if args.batch_size <= 0 or args.workers < 0:
        raise ValueError("batch size must be positive and workers nonnegative")
    # First side-effecting action remains below this complete parent auth.
    config, lock = _authenticate_parent_context()
    devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
    if not devices:
        raise ValueError("at least one CUDA device is required")
    if any(not value.startswith("cuda:") for value in devices):
        raise ValueError("formal extraction devices must be explicit cuda:N values")
    if COMPLETION_PATH.exists():
        summary = validate_complete(config=config, lock=lock)
        validate_completion_marker(COMPLETION_PATH, summary)
        print(json.dumps({"status": "COMPLETE", "cell_count": 20}, sort_keys=True))
        return
    pending = pending_cells(config, lock)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "PREFLIGHT",
                    "completed_cells": 20 - len(pending),
                    "missing_or_incomplete_cells": len(pending),
                    "devices": list(devices),
                },
                sort_keys=True,
            )
        )
        return
    if pending:
        queue = list(pending)
        queue_lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            futures = [
                executor.submit(
                    _execute_device,
                    device,
                    queue,
                    queue_lock,
                    batch_size=args.batch_size,
                    workers=args.workers,
                )
                for device in devices
            ]
            for future in as_completed(futures):
                future.result()
    summary = validate_complete(config=config, lock=lock)
    publish_json_once(summary, COMPLETION_PATH, private=True)
    validate_completion_marker(COMPLETION_PATH, summary)
    print(json.dumps({"status": "COMPLETE", "cell_count": 20}, sort_keys=True))


if __name__ == "__main__":
    main()

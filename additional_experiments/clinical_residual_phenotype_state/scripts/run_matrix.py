#!/usr/bin/env python3
"""Run the 2-arm x 2-seed x 5-fold Goal-F training or export matrix."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping


sys.dont_write_bytecode = True
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = EXPERIMENT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from crps.contracts import PRIMARY_ARMS, SEED_BASES, FOLDS, file_sha256  # noqa: E402
from crps.preregistration import LOCK_PATH, verify as verify_preregistration  # noqa: E402


FORMAL_TAG = "formal_primary"
CHECKPOINT_ROOT = EXPERIMENT_ROOT / "checkpoints" / FORMAL_TAG
FEATURE_ROOT = EXPERIMENT_ROOT / "features" / FORMAL_TAG
LOG_ROOT = EXPERIMENT_ROOT / "logs" / FORMAL_TAG


def _cells() -> tuple[tuple[int, str, int], ...]:
    return tuple(
        (seed, arm, fold)
        for seed in SEED_BASES
        for arm in PRIMARY_ARMS
        for fold in FOLDS
    )


def _parse_devices(value: str) -> tuple[str, ...]:
    devices = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if not devices:
        raise ValueError("at least one CUDA device identifier is required")
    if len(set(devices)) != len(devices):
        raise ValueError("CUDA device identifiers must be unique")
    if any(not device.isdecimal() for device in devices):
        raise ValueError("--devices must be comma-separated nonnegative CUDA indices")
    return devices


def _checkpoint_dir(seed: int, arm: str, fold: int) -> Path:
    return CHECKPOINT_ROOT / f"seed_{seed}" / arm / f"fold_{fold}"


def _feature_dir(seed: int, arm: str, fold: int) -> Path:
    return FEATURE_ROOT / f"seed_{seed}" / arm / f"fold_{fold}"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _complete_training(seed: int, arm: str, fold: int) -> bool:
    root = _checkpoint_dir(seed, arm, fold)
    selection_path = root / "selection.json"
    selected_path = root / "selected.pt"
    if not root.exists():
        return False
    if not selection_path.is_file() or not selected_path.is_file():
        if any(root.iterdir()):
            raise RuntimeError(f"partial training cell requires manual audit: {root}")
        return False
    selection = _load_json(selection_path)
    expected = {"arm": arm, "seed_base": seed, "fold": fold}
    if any(selection.get(key) != value for key, value in expected.items()):
        raise ValueError(f"training selection identity mismatch: {selection_path}")
    if selection.get("PCR_LABEL_ACCESS") != "FORBIDDEN":
        raise PermissionError(f"pCR firewall absent from selection: {selection_path}")
    return True


def _complete_export(seed: int, arm: str, fold: int) -> bool:
    root = _feature_dir(seed, arm, fold)
    feature = root / "factorized_state.private.npz"
    metadata_path = root / "factorized_state.private.metadata.json"
    if not root.exists():
        return False
    if not feature.is_file() or not metadata_path.is_file():
        if any(root.iterdir()):
            raise RuntimeError(f"partial feature export requires manual audit: {root}")
        return False
    metadata = _load_json(metadata_path)
    expected = {"arm": arm, "seed_base": seed, "fold": fold}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(f"feature metadata identity mismatch: {metadata_path}")
    if metadata.get("export_completed") is not True:
        raise ValueError(f"feature export lacks completion sentinel: {metadata_path}")
    if metadata.get("feature_sha256") != file_sha256(feature):
        raise ValueError(f"feature export SHA-256 mismatch: {feature}")
    return True


def _command(
    mode: str,
    seed: int,
    arm: str,
    fold: int,
    lock_sha256: str,
) -> list[str]:
    if mode == "train":
        return [
            sys.executable,
            str(EXPERIMENT_ROOT / "scripts" / "train_cell.py"),
            "--arm",
            arm,
            "--seed-base",
            str(seed),
            "--fold",
            str(fold),
            "--output-dir",
            str(_checkpoint_dir(seed, arm, fold)),
            "--device",
            "cuda",
            "--preregistration-lock-sha256",
            lock_sha256,
        ]
    return [
        sys.executable,
        str(EXPERIMENT_ROOT / "scripts" / "export_features.py"),
        "--arm",
        arm,
        "--seed-base",
        str(seed),
        "--fold",
        str(fold),
        "--checkpoint",
        str(_checkpoint_dir(seed, arm, fold) / "selected.pt"),
        "--output-dir",
        str(_feature_dir(seed, arm, fold)),
        "--device",
        "cuda",
        "--preregistration-lock-sha256",
        lock_sha256,
    ]


def _run_group(
    device: str,
    mode: str,
    cells: Iterable[tuple[int, str, int]],
    lock_sha256: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = device
    for seed, arm, fold in cells:
        log_path = LOG_ROOT / f"{mode}_seed_{seed}_{arm}_fold_{fold}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        command = _command(mode, seed, arm, fold, lock_sha256)
        with log_path.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                command,
                cwd=EXPERIMENT_ROOT.parents[1],
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        log_path.chmod(0o600)
        if completed.returncode:
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
            raise RuntimeError(
                f"{mode} failed for seed={seed} arm={arm} fold={fold} "
                f"on CUDA {device} (exit {completed.returncode}):\n"
                + "\n".join(tail)
            )
        complete = (
            _complete_training(seed, arm, fold)
            if mode == "train"
            else _complete_export(seed, arm, fold)
        )
        if not complete:
            raise RuntimeError(f"{mode} subprocess returned without a complete cell")
        results.append(
            {
                "seed_base": seed,
                "arm": arm,
                "fold": fold,
                "cuda_device": device,
                "status": "COMPLETE",
            }
        )
    return results


def _atomic_public_json(path: Path, payload: Mapping[str, Any]) -> None:
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
        path.chmod(0o644)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("train", "export"))
    parser.add_argument("--devices", default="0,1,2")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    devices = _parse_devices(args.devices)
    preregistration = verify_preregistration()
    lock_sha256 = file_sha256(LOCK_PATH)
    pending: list[tuple[int, str, int]] = []
    complete_before: list[tuple[int, str, int]] = []
    for cell in _cells():
        seed, arm, fold = cell
        if args.mode == "export" and not _complete_training(seed, arm, fold):
            raise FileNotFoundError(
                f"feature export blocked until selected checkpoint exists: {cell}"
            )
        complete = (
            _complete_training(seed, arm, fold)
            if args.mode == "train"
            else _complete_export(seed, arm, fold)
        )
        (complete_before if complete else pending).append(cell)
    plan = {
        "mode": args.mode,
        "required_cells": len(_cells()),
        "complete_before": len(complete_before),
        "pending": len(pending),
        "devices": list(devices),
        "preregistration_lock_sha256": lock_sha256,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    groups: list[list[tuple[int, str, int]]] = [[] for _ in devices]
    for index, cell in enumerate(pending):
        groups[index % len(devices)].append(cell)
    launched: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [
            executor.submit(_run_group, device, args.mode, group, lock_sha256)
            for device, group in zip(devices, groups, strict=True)
            if group
        ]
        for future in futures:
            launched.extend(future.result())
    all_complete = all(
        (
            _complete_training(seed, arm, fold)
            if args.mode == "train"
            else _complete_export(seed, arm, fold)
        )
        for seed, arm, fold in _cells()
    )
    if not all_complete:
        raise RuntimeError("formal matrix returned without all 20 cells complete")
    payload = {
        "schema_version": 1,
        "experiment": "clinical_residual_phenotype_state",
        "phase": "representation_training" if args.mode == "train" else "feature_export",
        "status": "COMPLETE",
        "required_cells": len(_cells()),
        "completed_cells": len(_cells()),
        "already_complete_cells": len(complete_before),
        "launched_cells": len(launched),
        "arms": list(PRIMARY_ARMS),
        "seed_bases": list(SEED_BASES),
        "folds": list(FOLDS),
        "PCR_LABEL_ACCESS": "FORBIDDEN",
        "pcr_labels_used": False,
        "preregistration_lock_sha256": lock_sha256,
        "preregistration_payload_sha256": preregistration["lock_sha256"],
    }
    filename = (
        "representation_matrix_status.json"
        if args.mode == "train"
        else "feature_export_status.json"
    )
    _atomic_public_json(EXPERIMENT_ROOT / "metrics" / filename, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

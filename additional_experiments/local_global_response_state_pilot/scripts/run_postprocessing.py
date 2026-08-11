#!/usr/bin/env python3
"""Run exact 60-cell feature export followed by sealed Ridge probes."""

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
import tempfile
import threading
import time
from typing import Any, Callable, Mapping, Sequence


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
SRC_ROOT = EXPERIMENT_ROOT / "src"
SCRIPTS_ROOT = EXPERIMENT_ROOT / "scripts"
SEALED_ROOT = REPO_ROOT / "additional_experiments" / "c1b_overlap_eligibility_ftv_stageb"
SEALED_SRC = SEALED_ROOT / "src"
DEFAULT_SENTINEL = SEALED_ROOT / "STAGE_A_GO.json"
DEFAULT_SENTINEL_SHA256 = "0b2c9e0af63ce8806525fb15ac9a27f6ab525b0259ccf16001981ca5091afbdb"
DEFAULT_DATA_CONTRACT = SEALED_ROOT / "manifests" / "stage_b_data_contract.private.json"
DEFAULT_DATA_CONTRACT_SHA256 = "dd22f130043863d4fce8956061fca389894a31874567ed7929e139f32ff5ab27"
CONFIG_PATH = EXPERIMENT_ROOT / "configs" / "pilot.json"
DEFAULT_TAG = "formal_4x8"
PILOT_FEATURE_ROOT = EXPERIMENT_ROOT / "features"
PILOT_PREDICTION_ROOT = EXPERIMENT_ROOT / "predictions"
for source in (SRC_ROOT, SCRIPTS_ROOT):
    value = str(source.resolve())
    while value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)

from freeze_preregistration import verify as verify_preregistration  # noqa: E402


@dataclass(frozen=True)
class Cell:
    index: int
    seed: int
    arm: str
    fold: int
    device: str
    checkpoint_dir: Path
    feature_dir: Path
    probe_dir: Path

    @property
    def checkpoint(self) -> Path:
        return self.checkpoint_dir / "selected.pt"

    @property
    def selection(self) -> Path:
        return self.checkpoint_dir / "selection.json"

    @property
    def history(self) -> Path:
        return self.checkpoint_dir / "history.csv"

    @property
    def feature(self) -> Path:
        return self.feature_dir / "response_state.private.npz"


class ActiveProcesses:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: dict[int, subprocess.Popen[bytes]] = {}
        self._aborted = False

    def run(self, command: Sequence[str]) -> None:
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        with self._lock:
            if self._aborted:
                raise RuntimeError("postprocessing was already aborted")
            process = subprocess.Popen(
                list(command), env=environment, start_new_session=True
            )
            self._active[process.pid] = process
        return_code = process.wait()
        with self._lock:
            self._active.pop(process.pid, None)
        if return_code:
            self.abort()
            raise subprocess.CalledProcessError(return_code, list(command))

    def abort(self) -> None:
        with self._lock:
            self._aborted = True
            active = tuple(self._active.values())
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
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--stage-a-sentinel", type=Path, default=DEFAULT_SENTINEL)
    parser.add_argument("--stage-a-sentinel-sha256", default=DEFAULT_SENTINEL_SHA256)
    parser.add_argument("--data-contract", type=Path, default=DEFAULT_DATA_CONTRACT)
    parser.add_argument("--data-contract-sha256", default=DEFAULT_DATA_CONTRACT_SHA256)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=EXPERIMENT_ROOT / "checkpoints" / DEFAULT_TAG,
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=EXPERIMENT_ROOT / "features" / DEFAULT_TAG,
    )
    parser.add_argument(
        "--probe-root",
        type=Path,
        default=EXPERIMENT_ROOT / "predictions" / DEFAULT_TAG,
    )
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def _devices(value: str) -> tuple[str, str, str]:
    devices = tuple(part.strip() for part in value.split(",") if part.strip())
    if len(devices) != 3 or len(set(devices)) != 3:
        raise ValueError("formal postprocessing requires three unique CUDA devices")
    if any(not device.startswith("cuda:") for device in devices):
        raise ValueError("formal devices must be explicit CUDA indices")
    return devices  # type: ignore[return-value]


def _cells(
    checkpoint_root: Path,
    feature_root: Path,
    probe_root: Path,
    config: Mapping[str, Any],
    devices: Sequence[str],
) -> tuple[Cell, ...]:
    cells: list[Cell] = []
    for index, (seed, fold, arm) in enumerate(
        (seed, fold, arm)
        for seed in config["training"]["seed_bases"]
        for fold in config["training"]["folds"]
        for arm in config["arms"]
    ):
        parts = (f"seed_{seed}", arm, f"fold_{fold}")
        cells.append(
            Cell(
                index=index,
                seed=int(seed),
                arm=str(arm),
                fold=int(fold),
                device=str(devices[index % len(devices)]),
                checkpoint_dir=checkpoint_root.joinpath(*parts),
                feature_dir=feature_root.joinpath(*parts),
                probe_dir=probe_root.joinpath(*parts),
            )
        )
    if len(cells) != 60:
        raise AssertionError("postprocessing plan must contain exactly 60 cells")
    return tuple(cells)


def _validate_matrix(
    checkpoint_root: Path,
    cells: Sequence[Cell],
    lock_sha256: str,
    config_sha256: str,
    stage_a_sentinel_sha256: str,
    data_contract_sha256: str,
) -> dict[str, Any]:
    completion_path = checkpoint_root / "matrix_complete.json"
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("formal matrix completion is missing or invalid") from error
    if (
        not isinstance(completion, dict)
        or completion.get("schema_version") != 1
        or completion.get("status") != "COMPLETE"
        or int(completion.get("run_count", -1)) != 60
    ):
        raise ValueError("formal matrix completion does not authorize 60 cells")
    preregistration = completion.get("preregistration")
    if not isinstance(preregistration, Mapping) or preregistration.get(
        "lock_sha256"
    ) != lock_sha256:
        raise ValueError("matrix completion uses another preregistration lock")
    expected_sources = {
        "config_sha256": config_sha256,
        "stage_a_sentinel_sha256": stage_a_sentinel_sha256,
        "data_contract_sha256": data_contract_sha256,
    }
    if any(completion.get(key) != value for key, value in expected_sources.items()):
        raise ValueError("matrix completion canonical config/data provenance drifted")
    runs = completion.get("runs")
    if not isinstance(runs, list) or len(runs) != 60:
        raise ValueError("matrix completion run inventory is not exactly 60 rows")
    observed = {
        (int(row["seed_base"]), str(row["arm"]), int(row["fold"])): Path(
            str(row["selection_path"])
        ).resolve()
        for row in runs
        if isinstance(row, Mapping)
    }
    expected = {
        (cell.seed, cell.arm, cell.fold): cell.selection for cell in cells
    }
    if observed != expected:
        raise ValueError("matrix completion identities/selection paths drifted")
    for cell in cells:
        if not all(path.is_file() for path in (cell.selection, cell.history, cell.checkpoint)):
            raise FileNotFoundError(f"training cell is incomplete: {cell.checkpoint_dir}")
        selection = json.loads(cell.selection.read_text(encoding="utf-8"))
        if any(
            selection.get(key) != value
            for key, value in {
                "seed_base": cell.seed,
                "arm": cell.arm,
                "fold": cell.fold,
                "test_data_used": False,
            }.items()
        ):
            raise ValueError(f"selection contract drifted for {cell.checkpoint_dir}")
        evidence = selection.get("preregistration")
        if (
            selection.get("preregistration_status") != "PASS"
            or selection.get("preregistration_lock_sha256") != lock_sha256
            or not isinstance(evidence, Mapping)
            or evidence != {"status": "PASS", "lock_sha256": lock_sha256}
        ):
            raise ValueError("selection preregistration binding drifted")
    return completion


def _require_empty(path: Path, label: str) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"{label} root must be nonexistent/empty: {path}")


def _run_streams(
    cells: Sequence[Cell],
    streams: Sequence[str],
    stream_for: Callable[[Cell], str],
    command_for: Callable[[Cell], Sequence[str]],
) -> None:
    buckets = {
        stream: tuple(cell for cell in cells if stream_for(cell) == stream)
        for stream in streams
    }
    if sum(len(bucket) for bucket in buckets.values()) != len(cells):
        raise ValueError("postprocess cell does not belong to exactly one stream")
    processes = ActiveProcesses()
    stop = threading.Event()
    lock = threading.Lock()
    failure: list[tuple[Cell, BaseException]] = []

    def worker(stream: str) -> None:
        for cell in buckets[stream]:
            if stop.is_set():
                return
            try:
                processes.run(command_for(cell))
            except BaseException as error:
                with lock:
                    if not failure:
                        failure.append((cell, error))
                        stop.set()
                        processes.abort()
                return

    with ThreadPoolExecutor(max_workers=len(streams)) as executor:
        futures = [executor.submit(worker, stream) for stream in streams]
        for future in as_completed(futures):
            future.result()
    if failure:
        cell, error = failure[0]
        raise RuntimeError(
            f"postprocessing failed at seed={cell.seed}/arm={cell.arm}/fold={cell.fold}"
        ) from error


def _private_json(path: Path, payload: Mapping[str, Any]) -> None:
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
        os.chmod(temporary, 0o600)
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assert_private_outputs(cells: Sequence[Cell], *, probes: bool) -> None:
    for cell in cells:
        paths = (
            (
                cell.probe_dir / "ridge_selection.csv",
                cell.probe_dir / "ridge_predictions.private.csv",
                cell.probe_dir / "probe_metrics.csv",
                cell.probe_dir / "probe_metadata.json",
            )
            if probes
            else (cell.feature, cell.feature.with_suffix(".metadata.json"))
        )
        for path in paths:
            if not path.is_file() or (path.stat().st_mode & 0o777) != 0o600:
                raise PermissionError(f"private output is missing or not 0600: {path}")


def main() -> None:
    os.umask(0o077)
    preregistration = verify_preregistration()
    import lg_response_pilot.security as pilot_security

    if Path(str(getattr(pilot_security, "__file__", ""))).resolve() != (
        SRC_ROOT / "lg_response_pilot" / "security.py"
    ).resolve():
        raise ImportError("pilot security module was shadowed before postprocessing")
    from lg_response_pilot.security import (
        claim_private_directory,
        require_canonical_file,
        resolve_contained_path,
    )

    args = parse_args()
    locked_upstream = preregistration["upstream_sha256"]
    args.config = require_canonical_file(
        args.config,
        CONFIG_PATH,
        preregistration["config_sha256"],
        preregistration["config_sha256"],
        label="pilot config",
    )
    args.stage_a_sentinel = require_canonical_file(
        args.stage_a_sentinel,
        DEFAULT_SENTINEL,
        args.stage_a_sentinel_sha256,
        locked_upstream[str(DEFAULT_SENTINEL.relative_to(REPO_ROOT))],
        label="Stage-A sentinel",
    )
    args.data_contract = require_canonical_file(
        args.data_contract,
        DEFAULT_DATA_CONTRACT,
        args.data_contract_sha256,
        locked_upstream[str(DEFAULT_DATA_CONTRACT.relative_to(REPO_ROOT))],
        label="Stage-B data contract",
    )
    sealed_value = str(SEALED_SRC.resolve())
    while sealed_value in sys.path:
        sys.path.remove(sealed_value)
    sys.path.insert(0, sealed_value)
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
    data_paths = StageBDataPaths.load(args.data_contract, args.data_contract_sha256)
    current_data = load_stage_b_data(
        data_paths, authorization, verify_cache_files=True
    )
    data_provenance_sha256 = _canonical_sha256(current_data.provenance)
    from lg_response_pilot.analysis import load_pilot_config

    import lg_response_pilot.analysis as pilot_analysis_module

    pilot_security.require_module_within(
        pilot_analysis_module,
        SRC_ROOT / "lg_response_pilot",
        label="pilot analysis",
    )

    config = load_pilot_config(args.config)
    devices = _devices(args.devices)
    checkpoints = args.checkpoint_root.resolve()
    features = resolve_contained_path(
        args.feature_root,
        PILOT_FEATURE_ROOT,
        label="formal feature root",
    )
    probes = resolve_contained_path(
        args.probe_root,
        PILOT_PREDICTION_ROOT,
        label="formal probe root",
    )
    cells = _cells(checkpoints, features, probes, config, devices)
    matrix = _validate_matrix(
        checkpoints,
        cells,
        preregistration["lock_sha256"],
        preregistration["config_sha256"],
        args.stage_a_sentinel_sha256,
        args.data_contract_sha256,
    )
    preflight = {
        "schema_version": 1,
        "status": "PREFLIGHT_PASS",
        "execution_requested": bool(args.execute),
        "cells": 60,
        "devices": list(devices),
        "feature_streams": 3,
        "probe_streams": 3,
        "preregistration_lock": "PREREGISTRATION_LOCK.json",
        "preregistration_lock_sha256": preregistration["lock_sha256"],
        "config_sha256": preregistration["config_sha256"],
        "stage_a_sentinel_sha256": args.stage_a_sentinel_sha256,
        "data_contract_sha256": args.data_contract_sha256,
        "cache_preflight": {
            "full_sha256_verified": True,
            "current_data_provenance_sha256": data_provenance_sha256,
        },
        "matrix_completion_sha256": hashlib.sha256(
            (checkpoints / "matrix_complete.json").read_bytes()
        ).hexdigest(),
    }
    if not args.execute:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return
    claim_private_directory(
        features,
        PILOT_FEATURE_ROOT,
        label="formal feature root",
    )
    claim_private_directory(
        probes,
        PILOT_PREDICTION_ROOT,
        label="formal probe root",
    )
    _private_json(features / "postprocessing_claim.private.json", preflight)
    export_script = SCRIPTS_ROOT / "export_features.py"
    probe_script = SCRIPTS_ROOT / "run_probes.py"

    def feature_command(cell: Cell) -> tuple[str, ...]:
        return (
            sys.executable,
            str(export_script),
            "--stage-a-sentinel",
            str(args.stage_a_sentinel),
            "--stage-a-sentinel-sha256",
            args.stage_a_sentinel_sha256,
            "--data-contract",
            str(args.data_contract),
            "--data-contract-sha256",
            args.data_contract_sha256,
            "--checkpoint",
            str(cell.checkpoint),
            "--feature-root",
            str(features),
            "--preregistration-lock-sha256",
            preregistration["lock_sha256"],
            "--arm",
            cell.arm,
            "--seed-base",
            str(cell.seed),
            "--fold",
            str(cell.fold),
            "--output",
            str(cell.feature),
            "--device",
            cell.device,
        )

    _run_streams(cells, devices, lambda cell: cell.device, feature_command)
    _assert_private_outputs(cells, probes=False)
    cpu_streams = ("cpu:0", "cpu:1", "cpu:2")

    def probe_command(cell: Cell) -> tuple[str, ...]:
        return (
            sys.executable,
            str(probe_script),
            "--stage-a-sentinel",
            str(args.stage_a_sentinel),
            "--stage-a-sentinel-sha256",
            args.stage_a_sentinel_sha256,
            "--data-contract",
            str(args.data_contract),
            "--data-contract-sha256",
            args.data_contract_sha256,
            "--features",
            str(cell.feature),
            "--feature-root",
            str(features),
            "--probe-root",
            str(probes),
            "--preregistration-lock-sha256",
            preregistration["lock_sha256"],
            "--output-dir",
            str(cell.probe_dir),
        )

    _run_streams(
        cells,
        cpu_streams,
        lambda cell: cpu_streams[cell.index % len(cpu_streams)],
        probe_command,
    )
    _assert_private_outputs(cells, probes=True)
    cell_chain = [
        {
            "seed_base": cell.seed,
            "arm": cell.arm,
            "fold": cell.fold,
            "selection_sha256": _file_sha256(cell.selection),
            "checkpoint_sha256": _file_sha256(cell.checkpoint),
            "feature_sha256": _file_sha256(cell.feature),
            "feature_metadata_sha256": _file_sha256(
                cell.feature.with_suffix(".metadata.json")
            ),
            "probe_metadata_sha256": _file_sha256(
                cell.probe_dir / "probe_metadata.json"
            ),
        }
        for cell in cells
    ]
    completion = {
        "schema_version": 1,
        "status": "COMPLETE",
        "cells": 60,
        "preregistration_lock": "PREREGISTRATION_LOCK.json",
        "preregistration_lock_sha256": preregistration["lock_sha256"],
        "config_sha256": preregistration["config_sha256"],
        "stage_a_sentinel_sha256": args.stage_a_sentinel_sha256,
        "data_contract_sha256": args.data_contract_sha256,
        "data_provenance_sha256": data_provenance_sha256,
        "matrix_completion_status": matrix["status"],
        "matrix_completion_sha256": preflight["matrix_completion_sha256"],
        "cell_chain": cell_chain,
        "cell_chain_sha256": _canonical_sha256(cell_chain),
        "patient_level_outputs_private": True,
    }
    _private_json(probes / "postprocessing_complete.private.json", completion)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the exact formal 40-cell Stage B feature export, then Ridge probes."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c1b_stage_b.cli import (  # noqa: E402
    add_data_contract_arguments,
    add_gate_arguments,
    authorize,
    data_paths,
)
from c1b_stage_b.contracts import G3_SRC, canonical_sha256, file_sha256  # noqa: E402
from c1b_stage_b.inputs import load_stage_b_data  # noqa: E402
from c1b_stage_b.matrix import parse_multi_devices  # noqa: E402
from c1b_stage_b.postprocess import (  # noqa: E402
    FORMAL_DEVICES,
    FORMAL_POSTPROCESS_TAG,
    PostprocessCell,
    build_feature_command,
    build_postprocess_cells,
    build_probe_command,
    validate_feature_outputs,
    validate_probe_outputs,
    validate_training_matrix,
)


FORMAL_CHECKPOINT_ROOT = ROOT / "checkpoints" / FORMAL_POSTPROCESS_TAG
FORMAL_FEATURE_ROOT = ROOT / "features" / FORMAL_POSTPROCESS_TAG
FORMAL_PROBE_ROOT = ROOT / "predictions" / FORMAL_POSTPROCESS_TAG


class PostprocessExecutionError(RuntimeError):
    """Raised after the first child failure and active-process termination."""


class ActiveSubprocesses:
    """Own child process groups so one failure can stop every active peer."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: dict[int, subprocess.Popen[bytes]] = {}
        self._aborted = False

    def run(
        self,
        command: Sequence[str],
        *,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        with self._lock:
            if self._aborted:
                raise PostprocessExecutionError("formal postprocessing was already aborted")
            try:
                process = subprocess.Popen(
                    list(command),
                    env=None if environment is None else dict(environment),
                    start_new_session=True,
                )
            except BaseException:
                self._aborted = True
                raise
            self._active[process.pid] = process
        return_code = process.wait()
        with self._lock:
            self._active.pop(process.pid, None)
            if return_code:
                # Close the launch race while holding the lock: after this
                # point no peer can start another cell.
                self._aborted = True
        if return_code:
            # Let the calling worker record this originating cell before it
            # invokes abort().  Synchronously killing peers here would let a
            # SIGTERM-killed peer race ahead and be misreported as the first
            # failure in the formal provenance/error message.
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


@contextmanager
def _termination_guard():
    """Turn parent SIGINT/SIGTERM into catchable phase-abort exceptions."""

    if threading.current_thread() is not threading.main_thread():
        yield
        return
    watched = (signal.SIGINT, signal.SIGTERM)
    previous = {number: signal.getsignal(number) for number in watched}

    def interrupt(number: int, _frame: Any) -> None:
        raise PostprocessExecutionError(
            f"formal postprocessing parent received {signal.Signals(number).name}"
        )

    try:
        for number in watched:
            signal.signal(number, interrupt)
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    finally:
        Path(temporary).unlink(missing_ok=True)


def _claim_formal_run(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically establish one persistent owner for the frozen output roots."""

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as error:
        raise FileExistsError(
            f"formal postprocessing root is already claimed and cannot be resumed: {path}"
        ) from error
    # A write failure deliberately leaves the exclusive claim in place.  That
    # makes an interrupted formal root fail closed instead of silently reusable.
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _hash_code_inventory(paths: Mapping[str, Path]) -> dict[str, str]:
    return {name: file_sha256(path) for name, path in paths.items()}


def _require_code_unchanged(
    paths: Mapping[str, Path], expected: Mapping[str, str]
) -> None:
    observed = _hash_code_inventory(paths)
    if observed != dict(expected):
        changed = sorted(
            name for name in set(observed) | set(expected)
            if observed.get(name) != expected.get(name)
        )
        raise RuntimeError(
            "formal postprocessing code changed after preflight: " + ", ".join(changed)
        )


def _hash_selection_history_inventory(
    cells: Sequence[PostprocessCell],
) -> dict[str, dict[str, str]]:
    inventory: dict[str, dict[str, str]] = {}
    for cell in cells:
        key = f"seed_{cell.seed_base}/{cell.arm}/fold_{cell.fold}"
        paths = {
            "selection_sha256": cell.selection_path,
            "history_sha256": cell.history_path,
        }
        if any(not path.is_file() for path in paths.values()):
            raise ValueError(f"training selection/history is missing for {key}")
        inventory[key] = {
            name: file_sha256(path) for name, path in paths.items()
        }
    return inventory


def _require_selection_history_unchanged(
    cells: Sequence[PostprocessCell],
    expected: Mapping[str, Mapping[str, str]],
) -> None:
    observed = _hash_selection_history_inventory(cells)
    if observed != {key: dict(value) for key, value in expected.items()}:
        changed = sorted(
            key
            for key in set(observed) | set(expected)
            if observed.get(key) != expected.get(key)
        )
        raise RuntimeError(
            "formal training selection/history changed after preflight: "
            + ", ".join(changed)
        )


def _require_exact_formal_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    observed = (
        args.checkpoint_root.resolve(),
        args.feature_root.resolve(),
        args.probe_root.resolve(),
    )
    expected = (
        FORMAL_CHECKPOINT_ROOT.resolve(),
        FORMAL_FEATURE_ROOT.resolve(),
        FORMAL_PROBE_ROOT.resolve(),
    )
    if observed != expected:
        raise ValueError(
            "formal postprocessing paths are frozen to checkpoints/features/"
            f"predictions/{FORMAL_POSTPROCESS_TAG} under this experiment"
        )
    return observed


def _require_empty_directory(path: Path, label: str) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(
            f"{label} must be nonexistent or completely empty: {path}; "
            "partial formal outputs are never resumed or overwritten"
        )


def _validate_cuda_devices(devices: Sequence[str]) -> None:
    import torch

    if tuple(devices) != FORMAL_DEVICES:
        raise ValueError(f"formal export devices must be exactly {FORMAL_DEVICES}")
    if not torch.cuda.is_available():
        raise RuntimeError("formal feature export requires CUDA, but CUDA is unavailable")
    device_count = torch.cuda.device_count()
    for device in devices:
        index = int(device.split(":", 1)[1])
        if index >= device_count:
            raise ValueError(
                f"formal feature export requested {device}, but only {device_count} CUDA devices exist"
            )


def _execute_feature_exports(
    cells: Sequence[PostprocessCell],
    devices: Sequence[str],
    command_for: Any,
) -> tuple[PostprocessCell, ...]:
    buckets = {
        device: tuple(cell for cell in cells if cell.device == device)
        for device in devices
    }
    if sum(len(bucket) for bucket in buckets.values()) != len(cells):
        raise ValueError("every feature cell must belong to exactly one formal GPU stream")
    processes = ActiveSubprocesses()
    stop = threading.Event()
    lock = threading.Lock()
    completed: list[PostprocessCell] = []
    first_failure: list[tuple[PostprocessCell, BaseException]] = []

    def record_failure(cell: PostprocessCell, error: BaseException) -> None:
        should_abort = False
        with lock:
            if not first_failure:
                first_failure.append((cell, error))
                stop.set()
                should_abort = True
        if should_abort:
            processes.abort()

    def device_worker(device: str) -> None:
        for cell in buckets[device]:
            if stop.is_set():
                return
            try:
                processes.run(command_for(cell))
            except BaseException as error:
                record_failure(cell, error)
                return
            with lock:
                completed.append(cell)

    with _termination_guard():
        with ThreadPoolExecutor(
            max_workers=len(devices), thread_name_prefix="stage-b-feature-gpu"
        ) as executor:
            futures = {
                executor.submit(device_worker, device): device for device in devices
            }
            try:
                for future in as_completed(futures):
                    try:
                        future.result()
                    except BaseException as error:
                        fallback = buckets[futures[future]][0]
                        record_failure(fallback, error)
            except BaseException:
                stop.set()
                processes.abort()
                for future in futures:
                    future.cancel()
                raise
    if first_failure:
        cell, error = first_failure[0]
        raise PostprocessExecutionError(
            "feature export stopped at first failure for "
            f"seed={cell.seed_base}, fold={cell.fold}, arm={cell.arm}, device={cell.device}"
        ) from error
    if len(completed) != len(cells):
        raise PostprocessExecutionError(
            "feature scheduler ended without a recorded failure but did not complete 40 cells"
        )
    return tuple(completed)


def _execute_probes(
    cells: Sequence[PostprocessCell],
    process_count: int,
    command_for: Any,
) -> tuple[PostprocessCell, ...]:
    if process_count <= 0:
        raise ValueError("--probe-processes must be positive")
    processes = ActiveSubprocesses()
    environment = os.environ.copy()
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment[variable] = "1"
    completed: list[PostprocessCell] = []
    first_failure: list[tuple[PostprocessCell, BaseException]] = []
    lock = threading.Lock()

    def run_cell(cell: PostprocessCell) -> PostprocessCell:
        processes.run(command_for(cell), environment=environment)
        return cell

    with _termination_guard():
        with ThreadPoolExecutor(
            max_workers=process_count, thread_name_prefix="stage-b-probe-cpu"
        ) as executor:
            futures = {executor.submit(run_cell, cell): cell for cell in cells}
            try:
                for future in as_completed(futures):
                    cell = futures[future]
                    try:
                        completed.append(future.result())
                    except BaseException as error:
                        with lock:
                            if not first_failure:
                                first_failure.append((cell, error))
                                processes.abort()
                        for pending in futures:
                            pending.cancel()
            except BaseException:
                processes.abort()
                for future in futures:
                    future.cancel()
                raise
    if first_failure:
        cell, error = first_failure[0]
        raise PostprocessExecutionError(
            "probe execution stopped at first failure for "
            f"seed={cell.seed_base}, fold={cell.fold}, arm={cell.arm}"
        ) from error
    if len(completed) != len(cells):
        raise PostprocessExecutionError(
            "probe scheduler ended without a recorded failure but did not complete 40 cells"
        )
    return tuple(completed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_gate_arguments(parser)
    add_data_contract_arguments(parser)
    parser.add_argument("--checkpoint-root", type=Path, default=FORMAL_CHECKPOINT_ROOT)
    parser.add_argument("--feature-root", type=Path, default=FORMAL_FEATURE_ROOT)
    parser.add_argument("--probe-root", type=Path, default=FORMAL_PROBE_ROOT)
    parser.add_argument("--devices", default=",".join(FORMAL_DEVICES))
    parser.add_argument("--probe-processes", type=int, default=8)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Without this explicit flag, perform gated read-only preflight only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    authorization = authorize(args)
    checkpoint_root, feature_root, probe_root = _require_exact_formal_paths(args)
    devices = parse_multi_devices(args.devices)
    _validate_cuda_devices(devices)
    if args.probe_processes <= 0:
        raise ValueError("--probe-processes must be positive")

    # This verifies all manifest hashes and the current GO-bound cohort without
    # repeating the parent matrix launcher's expensive full cache-file hash pass.
    data = load_stage_b_data(
        data_paths(args), authorization, verify_cache_files=False
    )
    cells = build_postprocess_cells(
        checkpoint_root, feature_root, probe_root, devices
    )
    matrix = validate_training_matrix(checkpoint_root, cells, authorization)
    selection_history_sha256 = _hash_selection_history_inventory(cells)
    selection_history_inventory_sha256 = canonical_sha256(
        selection_history_sha256
    )
    export_script = ROOT / "scripts" / "export_stage_b_features.py"
    probe_script = ROOT / "scripts" / "run_stage_b_probes.py"
    source_root = ROOT / "src" / "c1b_stage_b"
    code_paths = {
        "postprocess_driver": Path(__file__),
        "feature_cli": export_script,
        "probe_cli": probe_script,
        "aggregate_cli": ROOT / "scripts" / "aggregate_stage_b.py",
        **{
            f"c1b_stage_b/{path.name}": path
            for path in sorted(source_root.glob("*.py"))
        },
        "upstream_g3/model.py": G3_SRC / "dgrs" / "model.py",
        "upstream_g3/training.py": G3_SRC / "dgrs" / "training.py",
        "upstream_g3/targets.py": G3_SRC / "dgrs" / "targets.py",
    }
    code_sha256 = _hash_code_inventory(code_paths)
    preflight = {
        "schema_version": 1,
        "status": "PREFLIGHT_PASS",
        "formal_tag": FORMAL_POSTPROCESS_TAG,
        "stage_a_sentinel_sha256": authorization.sha256,
        "data_contract_path": str(args.data_contract.resolve()),
        "data_contract_sha256": args.data_contract_sha256,
        "matrix": matrix,
        "selection_history_sha256": selection_history_sha256,
        "selection_history_inventory_sha256": (
            selection_history_inventory_sha256
        ),
        "paths": {
            "checkpoint_root": str(checkpoint_root),
            "feature_root": str(feature_root),
            "probe_root": str(probe_root),
        },
        "scheduler": {
            "run_count": len(cells),
            "feature_devices": list(devices),
            "parallel_feature_processes": len(devices),
            "one_sequential_stream_per_device": True,
            "probe_processes": args.probe_processes,
            "probes_admitted_only_after_all_feature_hashes_validate": True,
            "fail_fast_active_process_group_termination": True,
        },
        "cell_inventory": [
            {
                "index": cell.index,
                "seed_base": cell.seed_base,
                "fold": cell.fold,
                "arm": cell.arm,
                "feature_device": cell.device,
            }
            for cell in cells
        ],
        "cohort_counts": {
            "fold_eligible_patients": int(data.folds["patient_id"].nunique()),
            "authorized_train_only_patients": len(data.train_only_ids),
            "matched_eligible_patients": len(data.eligibility.eligible_ids),
            "ftv_patients": len(data.ftv),
        },
        "code_sha256": code_sha256,
        "python_executable": str(Path(sys.executable).resolve()),
        "execution_requested": bool(args.execute),
    }
    if not args.execute:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return

    _require_empty_directory(feature_root, "formal feature root")
    _require_empty_directory(probe_root, "formal prediction root")
    feature_root.mkdir(parents=True, exist_ok=True)
    claim_path = feature_root / "postprocessing_claim.json"
    _claim_formal_run(
        claim_path,
        {
            "schema_version": 1,
            "status": "CLAIMED",
            "formal_tag": FORMAL_POSTPROCESS_TAG,
            "stage_a_sentinel_sha256": authorization.sha256,
            "data_contract_sha256": args.data_contract_sha256,
            "matrix_complete_sha256": matrix["matrix_complete_sha256"],
            "selection_history_inventory_sha256": (
                selection_history_inventory_sha256
            ),
            "postprocess_driver_sha256": code_sha256["postprocess_driver"],
            "nonresumable": True,
        },
    )
    preflight["claim_sha256"] = file_sha256(claim_path)
    preflight_path = feature_root / "postprocessing_preflight.json"
    _atomic_json(preflight_path, preflight)

    def feature_command(cell: PostprocessCell) -> tuple[str, ...]:
        return build_feature_command(
            cell,
            python_executable=sys.executable,
            export_script=export_script,
            stage_a_sentinel=args.stage_a_sentinel,
            data_contract=args.data_contract,
            data_contract_sha256=args.data_contract_sha256,
        )

    try:
        exported = _execute_feature_exports(cells, devices, feature_command)
        if len(exported) != 40:
            raise PostprocessExecutionError("feature export did not return forty cells")
        feature_metadata_hashes = validate_feature_outputs(
            cells,
            authorization,
            expected_feature_implementation_sha256=code_sha256[
                "c1b_stage_b/features.py"
            ],
        )
        _require_code_unchanged(code_paths, code_sha256)
        _require_selection_history_unchanged(cells, selection_history_sha256)
    except BaseException as error:
        raise RuntimeError(
            "formal feature export failed fast; no probes were admitted. Partial outputs "
            "were preserved and must not be overwritten or resumed. A rerun requires "
            "separately authorized new empty formal roots."
        ) from error

    feature_completion = {
        "schema_version": 1,
        "status": "COMPLETE",
        "formal_tag": FORMAL_POSTPROCESS_TAG,
        "run_count": len(feature_metadata_hashes),
        "stage_a_sentinel_sha256": authorization.sha256,
        "data_contract_sha256": args.data_contract_sha256,
        "matrix_complete_sha256": matrix["matrix_complete_sha256"],
        "claim_sha256": file_sha256(claim_path),
        "preflight_sha256": file_sha256(preflight_path),
        "selection_history_sha256": selection_history_sha256,
        "selection_history_inventory_sha256": (
            selection_history_inventory_sha256
        ),
        "feature_metadata_sha256": feature_metadata_hashes,
    }
    feature_completion_path = feature_root / "feature_export_complete.json"
    _atomic_json(feature_completion_path, feature_completion)

    # Recheck immediately before the phase transition so an external or stale
    # prediction artifact cannot be mixed with the just-validated feature set.
    _require_empty_directory(probe_root, "formal prediction root")
    probe_root.mkdir(parents=True, exist_ok=True)

    def probe_command(cell: PostprocessCell) -> tuple[str, ...]:
        return build_probe_command(
            cell,
            python_executable=sys.executable,
            probe_script=probe_script,
            stage_a_sentinel=args.stage_a_sentinel,
            data_contract=args.data_contract,
            data_contract_sha256=args.data_contract_sha256,
        )

    try:
        probed = _execute_probes(cells, args.probe_processes, probe_command)
        if len(probed) != 40:
            raise PostprocessExecutionError("probe execution did not return forty cells")
        probe_metadata_hashes = validate_probe_outputs(
            cells,
            authorization,
            expected_probe_implementation_sha256=code_sha256[
                "c1b_stage_b/probes.py"
            ],
            expected_target_adapter_sha256=code_sha256[
                "c1b_stage_b/targets.py"
            ],
        )
        _require_code_unchanged(code_paths, code_sha256)
        _require_selection_history_unchanged(cells, selection_history_sha256)
    except BaseException as error:
        raise RuntimeError(
            "formal probes failed fast after the complete feature gate. Partial prediction "
            "outputs were preserved and must not be overwritten or resumed. A rerun "
            "requires separately authorized new empty formal roots."
        ) from error

    completion = {
        "schema_version": 1,
        "status": "COMPLETE",
        "formal_tag": FORMAL_POSTPROCESS_TAG,
        "run_count": len(probe_metadata_hashes),
        "stage_a_sentinel_sha256": authorization.sha256,
        "data_contract_sha256": args.data_contract_sha256,
        "matrix_complete_sha256": matrix["matrix_complete_sha256"],
        "claim_sha256": file_sha256(claim_path),
        "preflight_sha256": file_sha256(preflight_path),
        "feature_export_complete_sha256": file_sha256(feature_completion_path),
        "selection_history_sha256": selection_history_sha256,
        "selection_history_inventory_sha256": (
            selection_history_inventory_sha256
        ),
        "probe_metadata_sha256": probe_metadata_hashes,
        "code_sha256": code_sha256,
    }
    completion_path = probe_root / "postprocessing_complete.json"
    _atomic_json(completion_path, completion)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

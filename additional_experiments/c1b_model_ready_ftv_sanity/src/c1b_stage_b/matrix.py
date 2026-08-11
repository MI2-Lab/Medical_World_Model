"""Pure planning and fail-fast scheduling for the 40-run Stage B matrix."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import re
import threading
from typing import Callable, Iterable, Sequence

from .contracts import FOLDS, SEED_BASES, validate_batch_contract


GROUP_ARM_ORDER = ("L1", "N1", "L3", "N3")


@dataclass(frozen=True)
class MatrixCell:
    group_index: int
    seed_base: int
    fold: int
    arm: str
    device: str
    output_dir: Path
    paired_baseline_selection: Path | None


@dataclass(frozen=True)
class MatrixGroup:
    index: int
    seed_base: int
    fold: int
    device: str
    cells: tuple[MatrixCell, ...]


class MatrixExecutionError(RuntimeError):
    """Raised after the first cell failure and active-run cancellation."""


def parse_multi_devices(value: str) -> tuple[str, ...]:
    devices = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if not devices:
        raise ValueError("--devices must contain at least one CUDA device")
    if len(set(devices)) != len(devices):
        raise ValueError("--devices contains a duplicate device")
    if any(re.fullmatch(r"cuda:\d+", device) is None for device in devices):
        raise ValueError("--devices entries must be explicit CUDA indices such as cuda:0")
    return devices


def build_matrix_groups(
    output_root: str | Path, devices: Sequence[str]
) -> tuple[MatrixGroup, ...]:
    devices = tuple(str(device) for device in devices)
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("matrix devices must be nonempty and unique")
    root = Path(output_root).resolve()
    groups: list[MatrixGroup] = []
    for group_index, (seed, fold) in enumerate(
        (seed, fold) for seed in SEED_BASES for fold in FOLDS
    ):
        device = devices[group_index % len(devices)]
        cells: list[MatrixCell] = []
        for arm in GROUP_ARM_ORDER:
            output_dir = root / f"seed_{seed}" / arm / f"fold_{fold}"
            if arm in {"L3", "N3"}:
                baseline_arm = "L1" if arm == "L3" else "N1"
                baseline = root / f"seed_{seed}" / baseline_arm / f"fold_{fold}" / "selection.json"
            else:
                baseline = None
            cells.append(
                MatrixCell(
                    group_index=group_index,
                    seed_base=seed,
                    fold=fold,
                    arm=arm,
                    device=device,
                    output_dir=output_dir,
                    paired_baseline_selection=baseline,
                )
            )
        groups.append(MatrixGroup(group_index, seed, fold, device, tuple(cells)))
    if len(groups) != 10 or sum(len(group.cells) for group in groups) != 40:
        raise AssertionError("formal Stage B matrix must contain ten groups and forty runs")
    return tuple(groups)


def build_train_command(
    cell: MatrixCell,
    *,
    python_executable: str | Path,
    train_script: str | Path,
    stage_a_sentinel: str | Path,
    data_contract: str | Path,
    data_contract_sha256: str,
    physical_batch_size: int,
    accumulation_steps: int,
    workers: int,
    global_fallback_restart: bool,
) -> tuple[str, ...]:
    validate_batch_contract(physical_batch_size, accumulation_steps)
    fallback = (int(physical_batch_size), int(accumulation_steps)) == (2, 16)
    if bool(global_fallback_restart) != fallback:
        raise ValueError("global fallback flag disagrees with the matrix batch contract")
    if int(workers) < 0:
        raise ValueError("workers must be nonnegative")
    command = [
        str(python_executable),
        str(train_script),
        "--stage-a-sentinel", str(stage_a_sentinel),
        "--data-contract", str(data_contract),
        "--data-contract-sha256", str(data_contract_sha256),
        "--arm", cell.arm,
        "--seed-base", str(cell.seed_base),
        "--fold", str(cell.fold),
        "--output-dir", str(cell.output_dir),
        "--device", cell.device,
        "--physical-batch-size", str(physical_batch_size),
        "--accumulation-steps", str(accumulation_steps),
        "--workers", str(workers),
    ]
    if cell.paired_baseline_selection is not None:
        command.extend(
            ["--paired-baseline-selection", str(cell.paired_baseline_selection)]
        )
    if global_fallback_restart:
        command.append("--global-fallback-restart")
    return tuple(command)


def execute_matrix_groups(
    groups: Iterable[MatrixGroup],
    devices: Sequence[str],
    run_cell: Callable[[MatrixCell], None],
    abort_active: Callable[[], None] | None = None,
) -> tuple[MatrixCell, ...]:
    """Run one sequential group stream per device and stop on first failure."""

    groups = tuple(groups)
    devices = tuple(str(device) for device in devices)
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("scheduler devices must be nonempty and unique")
    expected_indices = set(range(len(groups)))
    if {group.index for group in groups} != expected_indices:
        raise ValueError("matrix group indices must be unique and contiguous")
    buckets = {
        device: tuple(group for group in groups if group.device == device)
        for device in devices
    }
    if sum(len(bucket) for bucket in buckets.values()) != len(groups):
        raise ValueError("every matrix group must be assigned to exactly one requested device")
    for device, bucket in buckets.items():
        if any(group.device != device for group in bucket):
            raise AssertionError("device bucket assignment drifted")

    stop = threading.Event()
    lock = threading.Lock()
    completed: list[MatrixCell] = []
    first_failure: list[tuple[MatrixCell, BaseException]] = []
    abort_called = False

    def record_failure(cell: MatrixCell, error: BaseException) -> None:
        nonlocal abort_called
        should_abort = False
        with lock:
            if not first_failure:
                first_failure.append((cell, error))
                stop.set()
                if not abort_called:
                    abort_called = True
                    should_abort = True
        if should_abort and abort_active is not None:
            abort_active()

    def device_worker(device: str) -> None:
        for group in buckets[device]:
            if stop.is_set():
                return
            if tuple(cell.arm for cell in group.cells) != GROUP_ARM_ORDER:
                raise ValueError("within-group arm order must be L1,N1,L3,N3")
            for cell in group.cells:
                if stop.is_set():
                    return
                try:
                    run_cell(cell)
                except BaseException as error:
                    record_failure(cell, error)
                    return
                with lock:
                    completed.append(cell)

    with ThreadPoolExecutor(max_workers=len(devices), thread_name_prefix="stage-b-gpu") as executor:
        futures = [executor.submit(device_worker, device) for device in devices]
        for future in as_completed(futures):
            try:
                future.result()
            except BaseException as error:
                # Structural worker errors occur outside run_cell. Attribute
                # them to a synthetic first cell while still aborting active jobs.
                fallback = next(group.cells[0] for group in groups if group.cells)
                record_failure(fallback, error)
    if first_failure:
        cell, error = first_failure[0]
        raise MatrixExecutionError(
            f"Stage B matrix failed at seed={cell.seed_base}, fold={cell.fold}, "
            f"arm={cell.arm}, device={cell.device}: {error}"
        ) from error
    if len(completed) != sum(len(group.cells) for group in groups):
        raise MatrixExecutionError("Stage B scheduler ended without failure but matrix is incomplete")
    return tuple(completed)


__all__ = [
    "GROUP_ARM_ORDER",
    "MatrixCell",
    "MatrixExecutionError",
    "MatrixGroup",
    "build_matrix_groups",
    "build_train_command",
    "execute_matrix_groups",
    "parse_multi_devices",
]

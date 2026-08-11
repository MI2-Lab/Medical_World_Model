"""Pure planning and fail-fast scheduling for the formal 60-cell pilot."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import re
import threading
from typing import Callable, Iterable, Sequence

from .model import ARMS, arm_spec
from .training import BASELINE_BY_GROUNDED, FOLDS, SEED_BASES


FORMAL_ARM_ORDER = ("GAP0", "GAP3", "LOCAL0", "LOCAL3", "LG0", "LG3")
FORMAL_DEVICES = ("cuda:0", "cuda:1", "cuda:2")


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


def parse_three_devices(value: str) -> tuple[str, str, str]:
    devices = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if len(devices) != 3:
        raise ValueError("formal pilot execution requires exactly three CUDA devices")
    if len(set(devices)) != 3:
        raise ValueError("--devices contains a duplicate device")
    if any(re.fullmatch(r"cuda:\d+", device) is None for device in devices):
        raise ValueError("--devices entries must be explicit CUDA indices such as cuda:0")
    return devices  # type: ignore[return-value]


def build_matrix_groups(
    output_root: str | Path,
    devices: Sequence[str] = FORMAL_DEVICES,
) -> tuple[MatrixGroup, ...]:
    devices = tuple(str(device) for device in devices)
    if len(devices) != 3 or len(set(devices)) != 3:
        raise ValueError("formal matrix requires exactly three unique devices")
    if tuple(ARMS) != FORMAL_ARM_ORDER:
        raise ValueError("model arm inventory/order differs from the frozen pilot matrix")
    root = Path(output_root).resolve()
    groups: list[MatrixGroup] = []
    for group_index, (seed, fold) in enumerate(
        (seed, fold) for seed in SEED_BASES for fold in FOLDS
    ):
        device = devices[group_index % len(devices)]
        cells: list[MatrixCell] = []
        for arm in FORMAL_ARM_ORDER:
            spec = arm_spec(arm)
            output_dir = root / f"seed_{seed}" / arm / f"fold_{fold}"
            baseline = (
                root
                / f"seed_{seed}"
                / BASELINE_BY_GROUNDED[arm]
                / f"fold_{fold}"
                / "selection.json"
                if bool(spec.grounded)
                else None
            )
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
    if len(groups) != 10 or sum(len(group.cells) for group in groups) != 60:
        raise AssertionError("formal pilot matrix must contain ten groups and sixty runs")
    return tuple(groups)


def build_train_command(
    cell: MatrixCell,
    *,
    python_executable: str | Path,
    train_script: str | Path,
    stage_a_sentinel: str | Path,
    stage_a_sentinel_sha256: str,
    data_contract: str | Path,
    data_contract_sha256: str,
    preregistration_lock_sha256: str,
) -> tuple[str, ...]:
    command = [
        str(python_executable),
        str(train_script),
        "--stage-a-sentinel",
        str(stage_a_sentinel),
        "--stage-a-sentinel-sha256",
        str(stage_a_sentinel_sha256),
        "--data-contract",
        str(data_contract),
        "--data-contract-sha256",
        str(data_contract_sha256),
        "--preregistration-lock-sha256",
        str(preregistration_lock_sha256),
        "--arm",
        cell.arm,
        "--seed-base",
        str(cell.seed_base),
        "--fold",
        str(cell.fold),
        "--output-dir",
        str(cell.output_dir),
        "--device",
        cell.device,
    ]
    if cell.paired_baseline_selection is not None:
        command.extend(
            ["--paired-baseline-selection", str(cell.paired_baseline_selection)]
        )
    return tuple(command)


def execute_matrix_groups(
    groups: Iterable[MatrixGroup],
    devices: Sequence[str],
    run_cell: Callable[[MatrixCell], None],
    abort_active: Callable[[], None] | None = None,
) -> tuple[MatrixCell, ...]:
    """Run one sequential seed/fold stream per GPU and stop on first failure."""

    groups = tuple(groups)
    devices = tuple(str(device) for device in devices)
    if len(devices) != 3 or len(set(devices)) != 3:
        raise ValueError("formal scheduler requires exactly three unique devices")
    if {group.index for group in groups} != set(range(len(groups))):
        raise ValueError("matrix group indices must be unique and contiguous")
    buckets = {
        device: tuple(group for group in groups if group.device == device)
        for device in devices
    }
    if sum(len(bucket) for bucket in buckets.values()) != len(groups):
        raise ValueError("every matrix group must be assigned to one requested device")

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
            if tuple(cell.arm for cell in group.cells) != FORMAL_ARM_ORDER:
                raise ValueError("within-group arm order differs from the frozen pilot order")
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

    with ThreadPoolExecutor(
        max_workers=len(devices), thread_name_prefix="lg-pilot-gpu"
    ) as executor:
        futures = [executor.submit(device_worker, device) for device in devices]
        for future in as_completed(futures):
            try:
                future.result()
            except BaseException as error:
                fallback = next(group.cells[0] for group in groups if group.cells)
                record_failure(fallback, error)
    if first_failure:
        cell, error = first_failure[0]
        raise MatrixExecutionError(
            f"pilot matrix failed at seed={cell.seed_base}, fold={cell.fold}, "
            f"arm={cell.arm}, device={cell.device}: {error}"
        ) from error
    expected = sum(len(group.cells) for group in groups)
    if len(completed) != expected:
        raise MatrixExecutionError(
            "pilot scheduler ended without failure but matrix is incomplete"
        )
    return tuple(completed)


__all__ = [
    "FORMAL_ARM_ORDER",
    "FORMAL_DEVICES",
    "MatrixCell",
    "MatrixExecutionError",
    "MatrixGroup",
    "build_matrix_groups",
    "build_train_command",
    "execute_matrix_groups",
    "parse_three_devices",
]

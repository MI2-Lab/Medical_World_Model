"""Pure planning and fail-fast scheduling for the 100-cell confirmation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import re
import threading
from typing import Callable, Iterable, Sequence

from .model import ARMS, arm_spec
from .training import BASELINE_BY_GROUNDED, FOLDS, SEED_BASES, validate_seed_fold


FORMAL_ARM_ORDER = ("GAP0", "GAP3", "LOCAL0", "LOCAL3")
FORMAL_DEVICES = ("cuda:0", "cuda:1", "cuda:2")
FORMAL_SEED_BASES = (2026, 3026, 4026, 5026, 6026)
FORMAL_FOLDS = tuple(range(5))
FORMAL_GROUP_COUNT = 25
FORMAL_CELL_COUNT = 100


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
        raise ValueError("formal confirmation requires exactly three CUDA devices")
    if len(set(devices)) != 3:
        raise ValueError("--devices contains a duplicate device")
    if any(re.fullmatch(r"cuda:\d+", device) is None for device in devices):
        raise ValueError("--devices entries must be explicit CUDA indices such as cuda:0")
    return devices  # type: ignore[return-value]


def _validated_devices(devices: Sequence[str]) -> tuple[str, str, str]:
    parsed = tuple(str(device).strip() for device in devices)
    if len(parsed) != 3 or len(set(parsed)) != 3:
        raise ValueError("formal confirmation requires exactly three unique devices")
    if any(re.fullmatch(r"cuda:\d+", device) is None for device in parsed):
        raise ValueError("formal devices must be explicit CUDA indices such as cuda:0")
    return parsed  # type: ignore[return-value]


def validate_matrix_groups(
    groups: Iterable[MatrixGroup], devices: Sequence[str]
) -> tuple[MatrixGroup, ...]:
    """Fail closed unless groups encode the exact frozen order and pairings."""

    groups = tuple(groups)
    devices = _validated_devices(devices)
    if tuple(ARMS) != FORMAL_ARM_ORDER:
        raise ValueError("model arm inventory/order differs from confirmation matrix")
    if tuple(SEED_BASES) != FORMAL_SEED_BASES or tuple(FOLDS) != FORMAL_FOLDS:
        raise ValueError("training seed/fold inventory differs from confirmation matrix")
    if dict(BASELINE_BY_GROUNDED) != {"GAP3": "GAP0", "LOCAL3": "LOCAL0"}:
        raise ValueError("grounded-to-baseline pairing differs from confirmation contract")
    expected_groups = tuple(
        (seed, fold) for seed in FORMAL_SEED_BASES for fold in FORMAL_FOLDS
    )
    if len(groups) != FORMAL_GROUP_COUNT:
        raise ValueError(
            f"formal confirmation requires exactly {FORMAL_GROUP_COUNT} groups"
        )
    observed_groups = tuple((group.seed_base, group.fold) for group in groups)
    if observed_groups != expected_groups:
        raise ValueError("matrix groups differ from frozen seed-major/fold-minor order")

    output_roots: set[Path] = set()
    identities: set[tuple[int, int, str]] = set()
    for expected_index, group in enumerate(groups):
        if group.index != expected_index:
            raise ValueError("matrix group indices must be unique and contiguous")
        expected_device = devices[expected_index % len(devices)]
        if group.device != expected_device:
            raise ValueError("matrix group device assignment is not frozen round-robin")
        if tuple(cell.arm for cell in group.cells) != FORMAL_ARM_ORDER:
            raise ValueError(
                "within-group arm order must be GAP0,GAP3,LOCAL0,LOCAL3"
            )
        by_arm = {cell.arm: cell for cell in group.cells}
        if set(by_arm) != set(FORMAL_ARM_ORDER) or len(group.cells) != len(
            FORMAL_ARM_ORDER
        ):
            raise ValueError("matrix group must contain each formal arm exactly once")
        for cell in group.cells:
            validate_seed_fold(cell.seed_base, cell.fold)
            if (
                cell.group_index != group.index
                or cell.seed_base != group.seed_base
                or cell.fold != group.fold
                or cell.device != group.device
            ):
                raise ValueError("matrix cell identity disagrees with its group")
            output = cell.output_dir.resolve()
            root = output.parents[2]
            output_roots.add(root)
            expected_output = (
                root
                / f"seed_{group.seed_base}"
                / cell.arm
                / f"fold_{group.fold}"
            ).resolve()
            if output != expected_output:
                raise ValueError("matrix cell output path disagrees with its identity")
            identities.add((cell.seed_base, cell.fold, cell.arm))
            spec = arm_spec(cell.arm)
            if spec.grounded:
                baseline = by_arm[BASELINE_BY_GROUNDED[cell.arm]]
                expected_selection = (baseline.output_dir / "selection.json").resolve()
                if (
                    cell.paired_baseline_selection is None
                    or cell.paired_baseline_selection.resolve() != expected_selection
                ):
                    raise ValueError("grounded cell lacks its same-seed/fold paired baseline")
            elif cell.paired_baseline_selection is not None:
                raise ValueError("ungrounded matrix cell must not have a paired baseline")
    if len(output_roots) != 1:
        raise ValueError("all formal matrix cells must share one output root")
    if len(identities) != FORMAL_CELL_COUNT:
        raise ValueError(
            f"formal confirmation requires exactly {FORMAL_CELL_COUNT} unique cells"
        )
    return groups


def build_matrix_groups(
    output_root: str | Path,
    devices: Sequence[str] = FORMAL_DEVICES,
) -> tuple[MatrixGroup, ...]:
    devices = _validated_devices(devices)
    if tuple(ARMS) != FORMAL_ARM_ORDER:
        raise ValueError("model arm inventory/order differs from confirmation matrix")
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
    return validate_matrix_groups(groups, devices)


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
    spec = arm_spec(cell.arm)
    validate_seed_fold(cell.seed_base, cell.fold)
    if re.fullmatch(r"cuda:\d+", cell.device) is None:
        raise ValueError("cell device must be an explicit CUDA index such as cuda:0")
    output = cell.output_dir.resolve()
    root = output.parents[2]
    expected_output = (
        root / f"seed_{cell.seed_base}" / cell.arm / f"fold_{cell.fold}"
    ).resolve()
    if output != expected_output:
        raise ValueError("cell output path disagrees with its formal identity")
    if spec.grounded:
        expected_baseline = (
            root
            / f"seed_{cell.seed_base}"
            / BASELINE_BY_GROUNDED[cell.arm]
            / f"fold_{cell.fold}"
            / "selection.json"
        ).resolve()
        if (
            cell.paired_baseline_selection is None
            or cell.paired_baseline_selection.resolve() != expected_baseline
        ):
            raise ValueError("grounded command lacks its exact paired baseline")
    elif cell.paired_baseline_selection is not None:
        raise ValueError("ungrounded command must not include a paired baseline")
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

    devices = _validated_devices(devices)
    groups = validate_matrix_groups(groups, devices)
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
                raise ValueError(
                    "within-group arm order differs from frozen confirmation order"
                )
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
        max_workers=len(devices), thread_name_prefix="local-confirmation-gpu"
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
            f"confirmation matrix failed at seed={cell.seed_base}, fold={cell.fold}, "
            f"arm={cell.arm}, device={cell.device}: {error}"
        ) from error
    expected = sum(len(group.cells) for group in groups)
    if len(completed) != expected:
        raise MatrixExecutionError(
            "confirmation scheduler ended without failure but matrix is incomplete"
        )
    return tuple(completed)


__all__ = [
    "FORMAL_ARM_ORDER",
    "FORMAL_CELL_COUNT",
    "FORMAL_DEVICES",
    "FORMAL_FOLDS",
    "FORMAL_GROUP_COUNT",
    "FORMAL_SEED_BASES",
    "MatrixCell",
    "MatrixExecutionError",
    "MatrixGroup",
    "build_matrix_groups",
    "build_train_command",
    "execute_matrix_groups",
    "parse_three_devices",
    "validate_matrix_groups",
]

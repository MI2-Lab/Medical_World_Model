"""Formal, complete-matrix Stage B feature/probe postprocessing contract."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import ARMS, FOLDS, SEED_BASES, file_sha256
from .gate import StageAAuthorization


FORMAL_POSTPROCESS_TAG = "formal_4x8_restart1"
FORMAL_EXPORT_BATCH_SIZE = 4
FORMAL_EXPORT_WORKERS = 2
FORMAL_DEVICES = ("cuda:0", "cuda:1", "cuda:2")
FORMAL_HYPERPARAMETERS: Mapping[str, Any] = {
    "physical_batch_size": 4,
    "accumulation_steps": 8,
    "workers": 2,
    "epochs": 12,
    "patience": 4,
    "learning_rate": 5e-5,
    "weight_decay": 1e-4,
    "ema_momentum": 0.996,
    "max_grad_norm": 5.0,
    "min_representation_std": 0.05,
}


@dataclass(frozen=True)
class PostprocessCell:
    index: int
    seed_base: int
    fold: int
    arm: str
    device: str
    checkpoint_dir: Path
    feature_dir: Path
    probe_dir: Path

    @property
    def selection_path(self) -> Path:
        return self.checkpoint_dir / "selection.json"

    @property
    def history_path(self) -> Path:
        return self.checkpoint_dir / "history.csv"

    @property
    def checkpoint_path(self) -> Path:
        return self.checkpoint_dir / "selected.pt"

    @property
    def feature_path(self) -> Path:
        return self.feature_dir / "response_state.private.npz"

    @property
    def feature_metadata_path(self) -> Path:
        return self.feature_path.with_suffix(".metadata.json")

    @property
    def probe_metadata_path(self) -> Path:
        return self.probe_dir / "probe_metadata.json"


def build_postprocess_cells(
    checkpoint_root: str | Path,
    feature_root: str | Path,
    probe_root: str | Path,
    devices: Sequence[str] = FORMAL_DEVICES,
) -> tuple[PostprocessCell, ...]:
    devices = tuple(str(value) for value in devices)
    if devices != FORMAL_DEVICES:
        raise ValueError(f"formal postprocessing devices must be {FORMAL_DEVICES}")
    checkpoints = Path(checkpoint_root).resolve()
    features = Path(feature_root).resolve()
    probes = Path(probe_root).resolve()
    cells: list[PostprocessCell] = []
    for index, (seed, fold, arm) in enumerate(
        (seed, fold, arm)
        for seed in SEED_BASES
        for fold in FOLDS
        for arm in ARMS
    ):
        cells.append(
            PostprocessCell(
                index=index,
                seed_base=seed,
                fold=fold,
                arm=arm,
                device=devices[index % len(devices)],
                checkpoint_dir=checkpoints / f"seed_{seed}" / arm / f"fold_{fold}",
                feature_dir=features / f"seed_{seed}" / arm / f"fold_{fold}",
                probe_dir=probes / f"seed_{seed}" / arm / f"fold_{fold}",
            )
        )
    if len(cells) != 40 or len(
        {(cell.seed_base, cell.fold, cell.arm) for cell in cells}
    ) != 40:
        raise AssertionError("formal postprocessing plan must contain exactly 40 cells")
    return tuple(cells)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is missing or invalid: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def validate_training_matrix(
    checkpoint_root: str | Path,
    cells: Sequence[PostprocessCell],
    authorization: StageAAuthorization,
) -> dict[str, Any]:
    """Require the exact complete 4x8 matrix before any feature file is made."""

    root = Path(checkpoint_root).resolve()
    completion_path = root / "matrix_complete.json"
    completion = _read_json(completion_path, "matrix completion")
    if (
        int(completion.get("schema_version", -1)) != 1
        or completion.get("status") != "COMPLETE"
        or int(completion.get("run_count", -1)) != 40
        or completion.get("stage_a_sentinel_sha256") != authorization.sha256
    ):
        raise ValueError("matrix completion does not authorize this exact 40-cell postprocess")
    batch = completion.get("batch_contract")
    if not isinstance(batch, Mapping) or (
        int(batch.get("effective", -1)),
        int(batch.get("physical", -1)),
        int(batch.get("accumulation", -1)),
    ) != (32, 4, 8) or batch.get("global_for_all_arms") is not True:
        raise ValueError("postprocessing requires the complete global physical=4/accum=8 matrix")
    runs = completion.get("runs")
    if not isinstance(runs, list) or len(runs) != 40:
        raise ValueError("matrix completion run inventory is not exactly 40 rows")
    completed: dict[tuple[int, int, str], Path] = {}
    for run in runs:
        if not isinstance(run, Mapping):
            raise ValueError("matrix completion contains a non-object run row")
        key = (int(run.get("seed_base", -1)), int(run.get("fold", -1)), str(run.get("arm", "")))
        if key in completed:
            raise ValueError("matrix completion contains a duplicate run identity")
        completed[key] = Path(str(run.get("selection_path", ""))).resolve()
    expected_keys = {(cell.seed_base, cell.fold, cell.arm) for cell in cells}
    if set(completed) != expected_keys:
        raise ValueError("matrix completion identities differ from the formal 40 cells")

    for cell in cells:
        key = (cell.seed_base, cell.fold, cell.arm)
        if completed[key] != cell.selection_path:
            raise ValueError(f"matrix completion selection path drifted for {key}")
        selection = _read_json(cell.selection_path, "training selection")
        expected = {
            "schema_version": 1,
            "arm": cell.arm,
            "seed_base": cell.seed_base,
            "fold": cell.fold,
            "effective_seed": cell.seed_base + cell.fold,
            "test_data_used": False,
            "stage_a_sentinel_sha256": authorization.sha256,
            "global_fallback_restart": False,
        }
        for field, value in expected.items():
            if selection.get(field) != value:
                raise ValueError(f"selection {field} drifted for {key}")
        if selection.get("selection_mode") not in {"primary", "fallback_base_gate_failed"}:
            raise ValueError(f"selection mode is invalid for {key}")
        if selection.get("finite_status") is not True or not math.isfinite(
            float(selection.get("selected_representation_std", math.nan))
        ):
            raise ValueError(f"selected checkpoint is non-finite for {key}")
        hyperparameters = selection.get("hyperparameters")
        if not isinstance(hyperparameters, Mapping) or any(
            hyperparameters.get(field) != value
            for field, value in FORMAL_HYPERPARAMETERS.items()
        ):
            raise ValueError(f"formal training hyperparameters drifted for {key}")
        if not cell.history_path.is_file() or selection.get("history_sha256") != file_sha256(
            cell.history_path
        ):
            raise ValueError(f"training history hash drifted for {key}")
        if not cell.checkpoint_path.is_file() or cell.checkpoint_path.stat().st_size <= 0:
            raise ValueError(f"selected checkpoint is missing for {key}")
    return {
        "matrix_complete_sha256": file_sha256(completion_path),
        "run_count": len(cells),
        "batch_contract": dict(batch),
    }


def build_feature_command(
    cell: PostprocessCell,
    *,
    python_executable: str | Path,
    export_script: str | Path,
    stage_a_sentinel: str | Path,
    data_contract: str | Path,
    data_contract_sha256: str,
) -> tuple[str, ...]:
    return (
        str(python_executable),
        str(export_script),
        "--stage-a-sentinel", str(stage_a_sentinel),
        "--data-contract", str(data_contract),
        "--data-contract-sha256", str(data_contract_sha256),
        "--checkpoint", str(cell.checkpoint_path),
        "--arm", cell.arm,
        "--seed-base", str(cell.seed_base),
        "--fold", str(cell.fold),
        "--output", str(cell.feature_path),
        "--device", cell.device,
        "--batch-size", str(FORMAL_EXPORT_BATCH_SIZE),
        "--workers", str(FORMAL_EXPORT_WORKERS),
    )


def build_probe_command(
    cell: PostprocessCell,
    *,
    python_executable: str | Path,
    probe_script: str | Path,
    stage_a_sentinel: str | Path,
    data_contract: str | Path,
    data_contract_sha256: str,
) -> tuple[str, ...]:
    return (
        str(python_executable),
        str(probe_script),
        "--stage-a-sentinel", str(stage_a_sentinel),
        "--data-contract", str(data_contract),
        "--data-contract-sha256", str(data_contract_sha256),
        "--features", str(cell.feature_path),
        "--output-dir", str(cell.probe_dir),
    )


def validate_feature_outputs(
    cells: Sequence[PostprocessCell],
    authorization: StageAAuthorization,
    *,
    expected_feature_implementation_sha256: str | None = None,
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for cell in cells:
        key = f"seed_{cell.seed_base}/{cell.arm}/fold_{cell.fold}"
        metadata = _read_json(cell.feature_metadata_path, "feature metadata")
        if any(
            metadata.get(field) != value
            for field, value in {
                "schema_version": 1,
                "stage": "B",
                "arm": cell.arm,
                "seed_base": cell.seed_base,
                "fold": cell.fold,
                "stage_a_sentinel_sha256": authorization.sha256,
                "feature_tensor": "online_preprojector_r",
                "ftv_head_called": False,
                "test_labels_used": False,
            }.items()
        ):
            raise ValueError(f"feature metadata identity/provenance drifted for {key}")
        if Path(str(metadata.get("feature_path", ""))).resolve() != cell.feature_path:
            raise ValueError(f"feature metadata path drifted for {key}")
        if Path(str(metadata.get("checkpoint_path", ""))).resolve() != cell.checkpoint_path:
            raise ValueError(f"feature checkpoint path drifted for {key}")
        if metadata.get("checkpoint_sha256") != file_sha256(cell.checkpoint_path):
            raise ValueError(f"feature checkpoint hash drifted for {key}")
        if (
            expected_feature_implementation_sha256 is not None
            and metadata.get("feature_implementation_sha256")
            != expected_feature_implementation_sha256
        ):
            raise ValueError(f"feature implementation hash drifted for {key}")
        shape = metadata.get("feature_shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or not isinstance(shape[0], int)
            or isinstance(shape[0], bool)
            or shape[0] <= 0
            or shape[1:] != [4, 192]
        ):
            raise ValueError(f"feature tensor shape drifted for {key}")
        if not cell.feature_path.is_file() or metadata.get("feature_sha256") != file_sha256(
            cell.feature_path
        ):
            raise ValueError(f"feature hash drifted for {key}")
        hashes[key] = file_sha256(cell.feature_metadata_path)
    return hashes


def validate_probe_outputs(
    cells: Sequence[PostprocessCell],
    authorization: StageAAuthorization,
    *,
    expected_probe_implementation_sha256: str | None = None,
    expected_target_adapter_sha256: str | None = None,
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for cell in cells:
        key = f"seed_{cell.seed_base}/{cell.arm}/fold_{cell.fold}"
        metadata = _read_json(cell.probe_metadata_path, "probe metadata")
        if any(
            metadata.get(field) != value
            for field, value in {
                "schema_version": 1,
                "arm": cell.arm,
                "seed_base": cell.seed_base,
                "fold": cell.fold,
                "stage_a_sentinel_sha256": authorization.sha256,
                "test_used_for_scaler_or_selection": False,
                "outer_test_predict_calls_per_cell": 1,
                "feature_asset_name": cell.feature_path.name,
                "feature_metadata_name": cell.feature_metadata_path.name,
            }.items()
        ):
            raise ValueError(f"probe metadata identity/provenance drifted for {key}")
        if metadata.get("feature_sha256") != file_sha256(cell.feature_path):
            raise ValueError(f"probe feature binding drifted for {key}")
        if metadata.get("feature_metadata_sha256") != file_sha256(
            cell.feature_metadata_path
        ):
            raise ValueError(f"probe feature-metadata binding drifted for {key}")
        if (
            expected_probe_implementation_sha256 is not None
            and metadata.get("probe_implementation_sha256")
            != expected_probe_implementation_sha256
        ):
            raise ValueError(f"probe implementation hash drifted for {key}")
        if (
            expected_target_adapter_sha256 is not None
            and metadata.get("target_adapter_sha256")
            != expected_target_adapter_sha256
        ):
            raise ValueError(f"probe target-adapter hash drifted for {key}")
        output_hashes = metadata.get("output_sha256")
        expected = (
            cell.probe_dir / "ridge_selection.csv",
            cell.probe_dir / "ridge_predictions.private.csv",
            cell.probe_dir / "probe_metrics.csv",
        )
        if not isinstance(output_hashes, Mapping) or any(
            not path.is_file() or output_hashes.get(path.name) != file_sha256(path)
            for path in expected
        ):
            raise ValueError(f"probe output hashes drifted for {key}")
        hashes[key] = file_sha256(cell.probe_metadata_path)
    return hashes


__all__ = [
    "FORMAL_DEVICES",
    "FORMAL_POSTPROCESS_TAG",
    "PostprocessCell",
    "build_feature_command",
    "build_postprocess_cells",
    "build_probe_command",
    "validate_feature_outputs",
    "validate_probe_outputs",
    "validate_training_matrix",
]

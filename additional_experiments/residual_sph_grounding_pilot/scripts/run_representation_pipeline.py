#!/usr/bin/env python3
"""Plan or run the outcome-free representation pipeline in frozen stage order."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
SCRIPTS = EXPERIMENT_ROOT / "scripts"
SRC = EXPERIMENT_ROOT / "src"
sys.path.insert(0, str(SRC))

from residual_sph.evaluation_lock import (  # noqa: E402
    verify_pcr_firewall_audit,
    verify_representation_freeze,
)
from residual_sph.preregistration import (  # noqa: E402
    require_lock_sha256,
    verify_preregistration,
)


ARMS = ("S0", "S1", "S2", "S2_L10")
NEW_ARMS = ("S1", "S2", "S2_L10")
SEEDS = (2026, 3026)
FOLDS = tuple(range(5))
CHECKPOINTS = EXPERIMENT_ROOT / "checkpoints/formal_4x8"
FEATURES = EXPERIMENT_ROOT / "features/formal_4x8"
PROBES = EXPERIMENT_ROOT / "predictions/formal_4x8"
S0_ROOT = (
    REPO_ROOT
    / "additional_experiments/local_response_state_multiseed_confirmation"
)
REPRESENTATION_AGGREGATES = (
    EXPERIMENT_ROOT / "metrics/representation_metrics.csv",
    EXPERIMENT_ROOT / "metrics/table_static_ftv.csv",
    EXPERIMENT_ROOT / "metrics/table_observed_delta_ftv.csv",
    EXPERIMENT_ROOT / "metrics/table_sph_and_residual.csv",
    EXPERIMENT_ROOT / "metrics/table_partial_correlations.csv",
    EXPERIMENT_ROOT / "metrics/table_state_redundancy.csv",
    EXPERIMENT_ROOT / "metrics/table_seed_consistency.csv",
    EXPERIMENT_ROOT / "metrics/optimization_safety.csv",
    EXPERIMENT_ROOT / "metrics/optimization_trajectories.csv",
    EXPERIMENT_ROOT / "metrics/representation_effects.json",
)


@dataclass(frozen=True)
class Progress:
    name: str
    script: str
    completed: int
    total: int

    @property
    def status(self) -> str:
        if self.completed == self.total:
            return "COMPLETE"
        if self.completed:
            return "RESUMABLE"
        return "PENDING"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid completed {label}: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"completed {label} is not a JSON object: {path}")
    return value


def _nonempty(path: Path, *, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"completed {label} is missing or empty: {path}")


def _single_progress(
    name: str,
    script: str,
    path: Path,
    validator: Callable[[Path], None],
) -> Progress:
    if not path.exists():
        return Progress(name, script, 0, 1)
    validator(path)
    return Progress(name, script, 1, 1)


def _validate_s0(path: Path, lock_sha256: str) -> None:
    payload = _read_json(path, label="S0 provenance")
    expected = {
        "status": "S0_CONFIRMATION_PROVENANCE_VERIFIED",
        "cell_count": 10,
        "preregistration_lock_sha256": lock_sha256,
        "patient_identifiers_in_manifest": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"completed S0 provenance differs at {key}")
    cells = payload.get("cells")
    if not isinstance(cells, list) or len(cells) != 10:
        raise RuntimeError("completed S0 provenance lacks the exact ten cells")
    identities = {
        (str(cell.get("arm")), int(cell.get("seed_base", -1)), int(cell.get("fold", -1)))
        for cell in cells
        if isinstance(cell, dict)
    }
    if identities != {("S0", seed, fold) for seed in SEEDS for fold in FOLDS}:
        raise RuntimeError("completed S0 provenance cell coverage drifted")
    indexed = {
        (int(cell["seed_base"]), int(cell["fold"])): cell
        for cell in cells
        if isinstance(cell, dict)
    }
    for seed in SEEDS:
        for fold in FOLDS:
            cell = indexed[(seed, fold)]
            run = S0_ROOT / f"checkpoints/formal_4x8/seed_{seed}/LOCAL3/fold_{fold}"
            feature_run = (
                S0_ROOT / f"features/formal_4x8/seed_{seed}/LOCAL3/fold_{fold}"
            )
            bound = (
                (run / "selection.json", "selection_sha256"),
                (run / "selected.pt", "checkpoint_sha256"),
                (run / "ftv_transform.json", "ftv_transform_sha256"),
                (feature_run / "response_state.private.npz", "feature_sha256"),
                (
                    feature_run / "response_state.private.metadata.json",
                    "feature_metadata_sha256",
                ),
            )
            for artifact, hash_key in bound:
                _nonempty(artifact, label="confirmed S0 runtime artifact")
                if cell.get(hash_key) != _sha256(artifact):
                    raise RuntimeError(
                        f"completed S0 seed {seed}/fold {fold} differs at {hash_key}"
                    )


def _residualizer_progress(lock_sha256: str) -> Progress:
    residualizer_directory = EXPERIMENT_ROOT / "manifests/residualizers"
    fold_paths = tuple(
        residualizer_directory / f"fold_{fold}.json"
        for fold in FOLDS
    )
    inventory_path = EXPERIMENT_ROOT / "manifests/residualizer_inventory.json"
    fits_path = EXPERIMENT_ROOT / "metrics/residualizer_fits.csv"
    expected = (*fold_paths, inventory_path, fits_path)
    present = [path.exists() for path in expected]
    if not any(present):
        if residualizer_directory.exists():
            raise RuntimeError(
                "residualizer output directory exists without a complete inventory"
            )
        return Progress("residualizers", "scripts/build_residualizers.py", 0, 1)
    if not all(present):
        raise RuntimeError("residualizer stage is partial; refusing an unsafe overwrite")
    inventory = _read_json(inventory_path, label="residualizer inventory")
    if (
        inventory.get("status") != "FOLD_SAFE_RESIDUALIZERS_FITTED"
        or inventory.get("preregistration_lock_sha256") != lock_sha256
        or inventory.get("patient_level_values_persisted") is not False
    ):
        raise RuntimeError("completed residualizer inventory contract drifted")
    artifacts = inventory.get("fold_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(FOLDS):
        raise RuntimeError("completed residualizer inventory lacks five folds")
    by_fold = {int(row.get("fold", -1)): row for row in artifacts if isinstance(row, dict)}
    if set(by_fold) != set(FOLDS):
        raise RuntimeError("completed residualizer fold coverage drifted")
    for fold, path in zip(FOLDS, fold_paths, strict=True):
        _nonempty(path, label="residualizer transform")
        if by_fold[fold].get("artifact_sha256") != _sha256(path):
            raise RuntimeError(f"completed residualizer fold {fold} hash drifted")
    _nonempty(fits_path, label="residualizer fit table")
    return Progress("residualizers", "scripts/build_residualizers.py", 1, 1)


def _cell_directory_state(
    directory: Path,
    required: tuple[Path, ...],
    *,
    label: str,
    validator: Callable[[], None],
) -> bool:
    if not directory.exists():
        return False
    if not directory.is_dir():
        raise RuntimeError(f"{label} path is not a directory: {directory}")
    has_any = next(directory.iterdir(), None) is not None
    if not has_any:
        return False
    if not all(path.is_file() for path in required):
        raise RuntimeError(f"{label} is partial; refusing an unsafe overwrite: {directory}")
    validator()
    return True


def _validate_training_cell(
    directory: Path, arm: str, seed: int, fold: int, lock_sha256: str
) -> None:
    selection_path = directory / "selection.json"
    selection = _read_json(selection_path, label="training selection")
    expected = {
        "arm": arm,
        "seed_base": seed,
        "fold": fold,
        "preregistration_lock_sha256": lock_sha256,
        "test_data_used": False,
        "pcr_used": False,
        "clinical_used": False,
        "treatment_used": False,
    }
    for key, value in expected.items():
        if selection.get(key) != value:
            raise RuntimeError(
                f"completed train cell {arm}/seed_{seed}/fold_{fold} differs at {key}"
            )
    _read_json(directory / "ftv_transform.json", label="FTV transform")
    _nonempty(directory / "selected.pt", label="selected checkpoint")


def _training_progress(lock_sha256: str) -> Progress:
    completed = 0
    for arm in NEW_ARMS:
        for seed in SEEDS:
            for fold in FOLDS:
                directory = CHECKPOINTS / f"seed_{seed}/{arm}/fold_{fold}"
                required = (
                    directory / "selection.json",
                    directory / "selected.pt",
                    directory / "ftv_transform.json",
                )
                completed += int(
                    _cell_directory_state(
                        directory,
                        required,
                        label=f"train cell {arm}/seed_{seed}/fold_{fold}",
                        validator=lambda directory=directory, arm=arm, seed=seed, fold=fold: _validate_training_cell(
                            directory, arm, seed, fold, lock_sha256
                        ),
                    )
                )
    return Progress("matrix_train", "scripts/run_matrix.py", completed, 30)


def _validate_feature_cell(
    feature: Path, arm: str, seed: int, fold: int, lock_sha256: str
) -> None:
    _nonempty(feature, label="feature asset")
    metadata = _read_json(feature.with_suffix(".metadata.json"), label="feature metadata")
    expected = {
        "experiment": "residual_sph_grounding_pilot",
        "arm": arm,
        "seed_base": seed,
        "fold": fold,
        "feature_sha256": _sha256(feature),
        "feature_shape": [808, 4, 192],
        "feature_dtype": "float32",
        "test_labels_used": False,
        "clinical_or_pcr_loaded": False,
        "preregistration_lock_sha256": lock_sha256,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(
                f"completed feature {arm}/seed_{seed}/fold_{fold} differs at {key}"
            )


def _export_progress(lock_sha256: str) -> Progress:
    completed = 0
    for arm in NEW_ARMS:
        for seed in SEEDS:
            for fold in FOLDS:
                directory = FEATURES / f"seed_{seed}/{arm}/fold_{fold}"
                feature = directory / "response_state.private.npz"
                metadata = feature.with_suffix(".metadata.json")
                completed += int(
                    _cell_directory_state(
                        directory,
                        (feature, metadata),
                        label=f"export cell {arm}/seed_{seed}/fold_{fold}",
                        validator=lambda feature=feature, arm=arm, seed=seed, fold=fold: _validate_feature_cell(
                            feature, arm, seed, fold, lock_sha256
                        ),
                    )
                )
    return Progress("matrix_export", "scripts/run_matrix.py", completed, 30)


def _validate_probe_cell(
    directory: Path, arm: str, seed: int, fold: int, lock_sha256: str
) -> None:
    for name in ("ridge_selection.private.csv", "ridge_predictions.private.csv"):
        _nonempty(directory / name, label="probe table")
    metadata = _read_json(
        directory / "probe_metadata.private.json", label="probe metadata"
    )
    expected = {
        "arm": arm,
        "seed_base": seed,
        "fold": fold,
        "test_used_for_fit_or_selection": False,
        "pcr_or_clinical_loaded": False,
        "preregistration_lock_sha256": lock_sha256,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(
                f"completed probe {arm}/seed_{seed}/fold_{fold} differs at {key}"
            )


def _probe_progress(lock_sha256: str) -> Progress:
    completed = 0
    for arm in ARMS:
        for seed in SEEDS:
            for fold in FOLDS:
                directory = PROBES / f"seed_{seed}/{arm}/fold_{fold}"
                required = tuple(
                    directory / name
                    for name in (
                        "ridge_selection.private.csv",
                        "ridge_predictions.private.csv",
                        "probe_metadata.private.json",
                    )
                )
                completed += int(
                    _cell_directory_state(
                        directory,
                        required,
                        label=f"probe cell {arm}/seed_{seed}/fold_{fold}",
                        validator=lambda directory=directory, arm=arm, seed=seed, fold=fold: _validate_probe_cell(
                            directory, arm, seed, fold, lock_sha256
                        ),
                    )
                )
    return Progress("representation_probes", "scripts/run_probes.py", completed, 40)


def _aggregate_progress(lock_sha256: str) -> Progress:
    present = [path.exists() for path in REPRESENTATION_AGGREGATES]
    if not any(present):
        return Progress("aggregate_representation", "scripts/aggregate_representation.py", 0, 1)
    if not all(present):
        raise RuntimeError("representation aggregates are partial; refusing overwrite")
    for path in REPRESENTATION_AGGREGATES:
        _nonempty(path, label="representation aggregate")
    effects = _read_json(
        EXPERIMENT_ROOT / "metrics/representation_effects.json",
        label="representation effects",
    )
    if effects.get("preregistration_lock_sha256") != lock_sha256:
        raise RuntimeError("representation effects are bound to a different lock")
    return Progress("aggregate_representation", "scripts/aggregate_representation.py", 1, 1)


def _inspect(
    lock_sha256: str, implementation_sha256: str
) -> list[Progress]:
    s0_path = EXPERIMENT_ROOT / "manifests/s0_confirmation_provenance.json"
    firewall_path = EXPERIMENT_ROOT / "manifests/pcr_firewall_audit.json"
    freeze_path = EXPERIMENT_ROOT / "manifests/representation_freeze.json"
    stages = [
        _single_progress(
            "s0_reference_audit",
            "scripts/audit_s0_reference.py",
            s0_path,
            lambda path: _validate_s0(path, lock_sha256),
        ),
        _residualizer_progress(lock_sha256),
        _training_progress(lock_sha256),
        _export_progress(lock_sha256),
        _probe_progress(lock_sha256),
        _aggregate_progress(lock_sha256),
        _single_progress(
            "pcr_firewall_audit",
            "scripts/audit_pcr_firewall.py",
            firewall_path,
            lambda _path: verify_pcr_firewall_audit(
                EXPERIMENT_ROOT,
                expected_preregistration_sha256=lock_sha256,
                expected_implementation_sha256=implementation_sha256,
            ),
        ),
        _single_progress(
            "representation_freeze",
            "scripts/freeze_representation.py",
            freeze_path,
            lambda _path: verify_representation_freeze(
                EXPERIMENT_ROOT,
                expected_preregistration_sha256=lock_sha256,
            ),
        ),
    ]
    incomplete_seen = False
    for stage in stages:
        if incomplete_seen and stage.completed:
            raise RuntimeError(
                f"out-of-order artifact detected at {stage.name}; pipeline is fail-closed"
            )
        if stage.completed != stage.total:
            incomplete_seen = True
    return stages


def _public_plan(stages: list[Progress], lock_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "pipeline": "representation_only",
        "mode": "plan",
        "scientific_preregistration_sha256": lock_sha256,
        "resume_policy": "skip_valid_complete_units_fail_on_partial_units",
        "stages": [
            {
                "order": index,
                "name": stage.name,
                "script": stage.script,
                "status": stage.status,
                "completed_units": stage.completed,
                "total_units": stage.total,
            }
            for index, stage in enumerate(stages, start=1)
        ],
    }


def _run_child(script_name: str, arguments: list[str]) -> None:
    script = SCRIPTS / script_name
    if not script.is_file() or script.parent != SCRIPTS:
        raise FileNotFoundError(f"pipeline stage script is missing: {script_name}")
    subprocess.run(
        [sys.executable, str(script), *arguments],
        cwd=REPO_ROOT,
        check=True,
    )


def _run_probe_cells(lock_sha256: str) -> None:
    for arm in ARMS:
        for seed in SEEDS:
            for fold in FOLDS:
                directory = PROBES / f"seed_{seed}/{arm}/fold_{fold}"
                required = tuple(
                    directory / name
                    for name in (
                        "ridge_selection.private.csv",
                        "ridge_predictions.private.csv",
                        "probe_metadata.private.json",
                    )
                )
                if _cell_directory_state(
                    directory,
                    required,
                    label=f"probe cell {arm}/seed_{seed}/fold_{fold}",
                    validator=lambda directory=directory, arm=arm, seed=seed, fold=fold: _validate_probe_cell(
                        directory, arm, seed, fold, lock_sha256
                    ),
                ):
                    continue
                _run_child(
                    "run_probes.py",
                    [
                        "--arm",
                        arm,
                        "--seed-base",
                        str(seed),
                        "--fold",
                        str(fold),
                        "--output-dir",
                        str(directory),
                        "--preregistration-lock-sha256",
                        lock_sha256,
                    ],
                )


def _execute(
    lock_sha256: str,
    implementation_sha256: str,
    *,
    devices: str,
    minimum_free_mib: int,
) -> None:
    command_by_stage: dict[str, tuple[str, list[str]]] = {
        "s0_reference_audit": ("audit_s0_reference.py", []),
        "residualizers": ("build_residualizers.py", []),
        "matrix_train": (
            "run_matrix.py",
            [
                "--mode",
                "train",
                "--devices",
                devices,
                "--arms",
                ",".join(NEW_ARMS),
                "--minimum-free-mib",
                str(minimum_free_mib),
                "--preregistration-lock-sha256",
                lock_sha256,
            ],
        ),
        "matrix_export": (
            "run_matrix.py",
            [
                "--mode",
                "export",
                "--devices",
                devices,
                "--arms",
                ",".join(NEW_ARMS),
                "--minimum-free-mib",
                str(minimum_free_mib),
                "--preregistration-lock-sha256",
                lock_sha256,
            ],
        ),
        "aggregate_representation": (
            "aggregate_representation.py",
            ["--preregistration-lock-sha256", lock_sha256],
        ),
        "pcr_firewall_audit": (
            "audit_pcr_firewall.py",
            ["--preregistration-lock-sha256", lock_sha256],
        ),
        "representation_freeze": (
            "freeze_representation.py",
            ["--preregistration-lock-sha256", lock_sha256],
        ),
    }
    for stage in _inspect(lock_sha256, implementation_sha256):
        if stage.completed == stage.total:
            continue
        if stage.name == "representation_probes":
            _run_probe_cells(lock_sha256)
        else:
            script, arguments = command_by_stage[stage.name]
            _run_child(script, arguments)
        refreshed = {
            item.name: item for item in _inspect(lock_sha256, implementation_sha256)
        }[stage.name]
        if refreshed.completed != refreshed.total:
            raise RuntimeError(f"pipeline stage did not complete: {stage.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "run"), default="plan")
    parser.add_argument("--preregistration-lock-sha256", required=True)
    parser.add_argument("--devices", default="0,1,2")
    parser.add_argument("--minimum-free-mib", type=int, default=60_000)
    args = parser.parse_args()
    if args.minimum_free_mib <= 0:
        raise ValueError("minimum-free-mib must be positive")
    preregistration = verify_preregistration(EXPERIMENT_ROOT)
    require_lock_sha256(
        preregistration["lock_sha256"], args.preregistration_lock_sha256
    )
    stages = _inspect(
        preregistration["lock_sha256"],
        preregistration["implementation_lock_sha256"],
    )
    if args.mode == "plan":
        print(json.dumps(_public_plan(stages, preregistration["lock_sha256"]), indent=2))
        return
    _execute(
        preregistration["lock_sha256"],
        preregistration["implementation_lock_sha256"],
        devices=args.devices,
        minimum_free_mib=args.minimum_free_mib,
    )
    print(
        json.dumps(
            {
                "status": "REPRESENTATION_PIPELINE_COMPLETE",
                "representation_frozen": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

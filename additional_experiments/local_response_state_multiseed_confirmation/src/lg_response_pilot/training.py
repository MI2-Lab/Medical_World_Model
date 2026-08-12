"""Formal LOCAL multi-seed confirmation on sealed Stage-B B32 primitives.

Only GAP versus fixed 64-mm LOCAL pooling and optional FTV grounding differ
from the completed C1B-H Stage-B experiment. Logical batching, SIGReg,
validation, optimizer cadence, and EMA cadence are imported from that
experiment after an exact source-hash check.
"""

from __future__ import annotations

from dataclasses import asdict
import csv
import hashlib
import inspect
import json
import math
import operator
import os
from pathlib import Path
import random
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
SEALED_EXPERIMENT_ROOT = (
    REPO_ROOT / "additional_experiments" / "c1b_overlap_eligibility_ftv_stageb"
)
SEALED_SRC = SEALED_EXPERIMENT_ROOT / "src"

SEALED_SOURCE_SHA256: Mapping[str, str] = {
    "__init__.py": "2976c6d040c506b4f1b1db5374718d1c3edf341805d0e6a9d176f0c02fa37a47",
    "contracts.py": "48d7738b6764780ba2e784f826be44ac718fdbb0beb526ec31c3c5525cba4bf9",
    "data.py": "948a25aa00eeaf68a11f5a0bcf7c4d0c7592786a36ebb9bce472361745eebb59",
    "gate.py": "babb748a71eba0c36d802a8e15c861387d506de67cf754ec58ae96f1d3341555",
    "inputs.py": "40965e509afa059ce2674c7a7fde18cd9097e1eb00d05021738d8cf9f6346177",
    "targets.py": "06434db46cf76e6f39ff6eb1c476933885e90ed0a4c952dcc0a3477a25996c7b",
    "training.py": "2edf546628e447bdd1b9715f60f105d1a5952763bd782aabddbae298fae62f52",
    "upstream.py": "dfc03ab80590d1b57240a8ce210c75245bce4dd3bad9a4d655d8d63a1f96d54f",
}

SEED_BASES = (2026, 3026, 4026, 5026, 6026)
FOLDS = tuple(range(5))
EFFECTIVE_BATCH_SIZE = 32
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
BASELINE_BY_GROUNDED: Mapping[str, str] = {
    "GAP3": "GAP0",
    "LOCAL3": "LOCAL0",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sealed_stage_b_sources() -> dict[str, str]:
    """Fail closed if any reused Stage-B implementation has changed."""

    observed: dict[str, str] = {}
    package = SEALED_SRC / "c1b_stage_b"
    for name, expected in SEALED_SOURCE_SHA256.items():
        path = package / name
        if not path.is_file():
            raise ImportError(f"sealed Stage-B source is missing: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise ImportError(
                f"sealed Stage-B source hash drifted for {name}: "
                f"expected {expected}, observed {actual}"
            )
        observed[name] = actual
    return observed


_SEALED_SOURCE_VERIFICATION = verify_sealed_stage_b_sources()
sealed_value = str(SEALED_SRC.resolve())
while sealed_value in sys.path:
    sys.path.remove(sealed_value)
sys.path.insert(0, sealed_value)

# These imports intentionally occur only after the source inventory passes.
from c1b_stage_b.contracts import (  # noqa: E402
    LOGICAL_OBJECTIVE_CONTRACT,
    canonical_sha256,
    file_sha256,
    ordered_patient_sha256,
)
from c1b_stage_b.training import (  # noqa: E402
    TrainHyperparameters,
    logical_patient_batches,
    logical_sigreg_surrogate,
    physical_patient_batches,
    run_logical_train_epoch,
    run_validation_epoch,
    scale_microbatch_components,
    select_checkpoint,
)

import c1b_stage_b as _SEALED_PACKAGE  # noqa: E402
import c1b_stage_b.contracts as _SEALED_CONTRACTS  # noqa: E402
import c1b_stage_b.training as _SEALED_TRAINING  # noqa: E402

_EXPECTED_SEALED_MODULES = {
    _SEALED_PACKAGE: (SEALED_SRC / "c1b_stage_b" / "__init__.py").resolve(),
    _SEALED_CONTRACTS: (SEALED_SRC / "c1b_stage_b" / "contracts.py").resolve(),
    _SEALED_TRAINING: (SEALED_SRC / "c1b_stage_b" / "training.py").resolve(),
}
for _module, _expected_path in _EXPECTED_SEALED_MODULES.items():
    if Path(str(getattr(_module, "__file__", ""))).resolve() != _expected_path:
        raise ImportError(f"sealed Stage-B module resolved outside canonical root: {_module}")
for _value, _expected_path in (
    (canonical_sha256, _EXPECTED_SEALED_MODULES[_SEALED_CONTRACTS]),
    (file_sha256, _EXPECTED_SEALED_MODULES[_SEALED_CONTRACTS]),
    (ordered_patient_sha256, _EXPECTED_SEALED_MODULES[_SEALED_CONTRACTS]),
    (TrainHyperparameters, _EXPECTED_SEALED_MODULES[_SEALED_TRAINING]),
    (logical_patient_batches, _EXPECTED_SEALED_MODULES[_SEALED_TRAINING]),
    (logical_sigreg_surrogate, _EXPECTED_SEALED_MODULES[_SEALED_TRAINING]),
    (physical_patient_batches, _EXPECTED_SEALED_MODULES[_SEALED_TRAINING]),
    (run_logical_train_epoch, _EXPECTED_SEALED_MODULES[_SEALED_TRAINING]),
    (run_validation_epoch, _EXPECTED_SEALED_MODULES[_SEALED_TRAINING]),
    (scale_microbatch_components, _EXPECTED_SEALED_MODULES[_SEALED_TRAINING]),
    (select_checkpoint, _EXPECTED_SEALED_MODULES[_SEALED_TRAINING]),
):
    if Path(inspect.getfile(inspect.unwrap(_value))).resolve() != _expected_path:
        raise ImportError(f"sealed Stage-B object resolved outside canonical source: {_value}")

from .model import (  # noqa: E402
    ARMS,
    arm_spec,
    shared_initialization_sha256,
    transition_sha256,
    validate_model_contract,
)


def validate_seed_fold(seed_base: int, fold: int) -> int:
    try:
        seed = operator.index(seed_base)
    except TypeError as error:
        raise ValueError(f"seed_base must be one of {SEED_BASES}") from error
    try:
        fold_index = operator.index(fold)
    except TypeError as error:
        raise ValueError(f"fold must be one of {FOLDS}") from error
    if isinstance(seed_base, bool) or seed not in SEED_BASES:
        raise ValueError(f"seed_base must be one of {SEED_BASES}")
    if isinstance(fold, bool) or fold_index not in FOLDS:
        raise ValueError(f"fold must be one of {FOLDS}")
    return seed + fold_index


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def formal_hyperparameters() -> TrainHyperparameters:
    hyperparameters = TrainHyperparameters(**FORMAL_HYPERPARAMETERS)
    validate_formal_hyperparameters(hyperparameters)
    return hyperparameters


def validate_formal_hyperparameters(hyperparameters: TrainHyperparameters) -> None:
    hyperparameters.validate()
    observed = asdict(hyperparameters)
    if observed != dict(FORMAL_HYPERPARAMETERS):
        raise ValueError(
            "formal LOCAL confirmation hyperparameters are frozen to physical=4, "
            "accumulation=8, logical B32 and the completed Stage-B optimizer budget"
        )


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
        path.chmod(0o600)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _write_history(path: Path, history: Sequence[Mapping[str, Any]]) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary).replace(path)
        path.chmod(0o600)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".pt", dir=path.parent
    )
    os.close(descriptor)
    try:
        torch.save(dict(payload), temporary)
        Path(temporary).replace(path)
        path.chmod(0o600)
    finally:
        Path(temporary).unlink(missing_ok=True)


def validate_paired_baseline(
    selection_path: str | Path,
    *,
    grounded_arm: str,
    seed_base: int,
    fold: int,
    effective_seed: int,
    paired_initialization_sha256: str,
    hyperparameters: TrainHyperparameters,
    train_patient_sha256: str,
    val_patient_sha256: str,
    data_provenance_sha256: str,
    preregistration_lock_sha256: str,
) -> tuple[float, dict[str, Any]]:
    """Validate the same-architecture no-grounding selection and safety anchor."""

    arm = str(grounded_arm).upper()
    if arm not in BASELINE_BY_GROUNDED:
        raise ValueError(f"grounded arm has no registered paired baseline: {arm}")
    path = Path(selection_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "arm": BASELINE_BY_GROUNDED[arm],
        "seed_base": int(seed_base),
        "fold": int(fold),
        "effective_seed": int(effective_seed),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"paired baseline selection {key} mismatch")
    if payload.get("selection_mode") != "primary" or payload.get("experiment_pass") is not True:
        raise ValueError("paired no-ground baseline must have a primary passing selection")
    if payload.get("test_data_used") is not False:
        raise ValueError("paired baseline selection is not test blind")
    paired_contract = {
        "paired_initialization_sha256": paired_initialization_sha256,
        "hyperparameters": asdict(hyperparameters),
        "train_patient_sha256": train_patient_sha256,
        "val_patient_sha256": val_patient_sha256,
        "data_provenance_sha256": data_provenance_sha256,
    }
    for key, value in paired_contract.items():
        if payload.get(key) != value:
            raise ValueError(f"paired baseline {key} mismatch")
    expected_preregistration = {
        "status": "PASS",
        "lock_sha256": str(preregistration_lock_sha256),
    }
    if (
        payload.get("preregistration_status") != "PASS"
        or payload.get("preregistration_lock_sha256")
        != expected_preregistration["lock_sha256"]
        or payload.get("preregistration") != expected_preregistration
    ):
        raise ValueError("paired baseline preregistration lock mismatch")
    metric = float(payload["selected_validation_state_loss"])
    if not math.isfinite(metric) or metric <= 0:
        raise ValueError("paired baseline state loss must be finite and positive")
    return metric, payload


def train_epochs(
    *,
    arm: str,
    seed_base: int,
    fold: int,
    model: torch.nn.Module,
    objective: torch.nn.Module,
    train_dataset: Any,
    val_dataset: Any,
    device: torch.device,
    output_dir: str | Path,
    authorization: Any,
    hyperparameters: TrainHyperparameters,
    paired_initialization_sha256: str,
    data_provenance: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    paired_baseline_selection: str | Path | None = None,
) -> dict[str, Any]:
    """Train one of four confirmation arms under the frozen Stage-B rule."""

    spec = arm_spec(arm)
    effective_seed = validate_seed_fold(seed_base, fold)
    validate_formal_hyperparameters(hyperparameters)
    validate_model_contract(model, spec.name)
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"refusing to mix or overwrite an existing confirmation run: {output}"
        )
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    output.chmod(0o700)
    observed_initialization = shared_initialization_sha256(model)
    if observed_initialization != paired_initialization_sha256:
        raise AssertionError(
            "run model does not match its paired architecture initialization hash"
        )
    initial_transition_sha256 = transition_sha256(model)
    train_patient_sha256 = canonical_sha256(sorted(train_dataset.patient_ids))
    val_patient_sha256 = canonical_sha256(sorted(val_dataset.patient_ids))
    data_provenance_sha256 = canonical_sha256(data_provenance)
    preregistration_status = str(preregistration.get("status", ""))
    preregistration_lock_sha256 = str(preregistration.get("lock_sha256", ""))
    if preregistration_status != "PASS" or len(preregistration_lock_sha256) != 64:
        raise ValueError("formal training requires a verified preregistration lock")
    preregistration_evidence = {
        "status": preregistration_status,
        "lock_sha256": preregistration_lock_sha256,
    }
    if tuple(
        int(value)
        for value in (
            hyperparameters.physical_batch_size,
            hyperparameters.accumulation_steps,
        )
    ) != (4, 8):
        raise ValueError(
            "formal confirmation training requires physical=4 and accumulation=8"
        )

    if bool(spec.grounded):
        if paired_baseline_selection is None:
            raise ValueError(f"{spec.name} requires its matching no-grounding selection")
        baseline_state, baseline_payload = validate_paired_baseline(
            paired_baseline_selection,
            grounded_arm=spec.name,
            seed_base=seed_base,
            fold=fold,
            effective_seed=effective_seed,
            paired_initialization_sha256=paired_initialization_sha256,
            hyperparameters=hyperparameters,
            train_patient_sha256=train_patient_sha256,
            val_patient_sha256=val_patient_sha256,
            data_provenance_sha256=data_provenance_sha256,
            preregistration_lock_sha256=preregistration_lock_sha256,
        )
    else:
        if paired_baseline_selection is not None:
            raise ValueError(f"{spec.name} must not receive a baseline selection")
        baseline_state, baseline_payload = None, None

    model.to(device)
    objective.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparameters.learning_rate,
        weight_decay=hyperparameters.weight_decay,
    )
    history: list[dict[str, Any]] = []
    stale = 0
    running_best: tuple[float, float] = (math.inf, math.inf)
    for epoch in range(1, hyperparameters.epochs + 1):
        logical = logical_patient_batches(train_dataset.patient_ids, effective_seed, epoch)
        logical_order = tuple(patient for batch in logical for patient in batch)
        train_stats = run_logical_train_epoch(
            model,
            objective,
            train_dataset,
            optimizer,
            device,
            logical,
            hyperparameters,
            effective_seed=effective_seed,
            epoch=epoch,
        )
        val_stats = run_validation_epoch(
            model,
            objective,
            val_dataset,
            device,
            hyperparameters.physical_batch_size,
            hyperparameters.workers,
        )
        row: dict[str, Any] = {
            "epoch": epoch,
            "arm": spec.name,
            "architecture": spec.architecture,
            "seed_base": int(seed_base),
            "fold": int(fold),
            "effective_seed": effective_seed,
            "patient_order_sha256": ordered_patient_sha256(logical_order),
            "dropped_logical_tail_patients": len(train_dataset) - len(logical_order),
            "train_loss": train_stats["loss"],
            "train_base_loss": train_stats["base_loss"],
            "train_state_loss": train_stats["state_loss"],
            "train_ftv_loss": train_stats["ftv_loss"],
            "train_grounded_patients": train_stats["grounded_patients"],
            "train_representation_std": train_stats["representation_std"],
            "train_optimizer_steps": train_stats["optimizer_steps"],
            "val_loss": val_stats["loss"],
            "val_base_objective": val_stats["base_loss"],
            "val_state_loss": val_stats["state_loss"],
            "val_ftv_loss": val_stats["ftv_loss"],
            "val_grounded_patients": val_stats["grounded_patients"],
            "val_representation_std": val_stats["representation_std"],
            "finite": all(
                math.isfinite(float(value))
                for value in (
                    train_stats["loss"],
                    val_stats["state_loss"],
                    val_stats["representation_std"],
                )
            ),
        }
        history.append(row)
        parameter_counts = (
            model.parameter_counts() if hasattr(model, "parameter_counts") else None
        )
        checkpoint_payload = {
            "schema_version": 1,
            "stage": "local_response_state_multiseed_confirmation",
            "arm": spec.name,
            "architecture": spec.architecture,
            "input_kind": "c1b",
            "grounded": bool(spec.grounded),
            "seed_base": int(seed_base),
            "fold": int(fold),
            "effective_seed": effective_seed,
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "model_config": model.model_config(),
            "architecture_contract": model.architecture_contract(),
            "parameter_counts": parameter_counts,
            "optimizer_state": optimizer.state_dict(),
            "hyperparameters": asdict(hyperparameters),
            "paired_initialization_sha256": paired_initialization_sha256,
            "transition_initialization_sha256": initial_transition_sha256,
            "stage_a_sentinel_path": str(authorization.path),
            "stage_a_sentinel_sha256": authorization.sha256,
            "train_patient_sha256": train_patient_sha256,
            "val_patient_sha256": val_patient_sha256,
            "data_provenance_sha256": data_provenance_sha256,
            "preregistration_status": preregistration_status,
            "preregistration_lock_sha256": preregistration_lock_sha256,
            "preregistration": preregistration_evidence,
            "test_data_used": False,
            "delta_ftv_used": False,
            "pcr_used": False,
            "data_provenance": dict(data_provenance),
            "paired_baseline_selection": baseline_payload,
            "epoch_metrics": row,
        }
        _atomic_torch_save(output / f"epoch_{epoch:02d}.pt", checkpoint_payload)
        _write_history(output / "history.csv", history)

        noncollapse = (
            row["finite"]
            and row["val_representation_std"]
            >= hyperparameters.min_representation_std
        )
        ftv_finite = (
            math.isfinite(row["val_ftv_loss"])
            and row["val_grounded_patients"] > 0
        )
        if bool(spec.grounded):
            violation = max(
                0.0,
                row["val_state_loss"] - 1.05 * float(baseline_state),
            )
            current = (
                violation,
                row["val_ftv_loss"] if ftv_finite else math.inf,
            )
        else:
            current = (0.0, row["val_state_loss"])
        improved = noncollapse and current < running_best
        if improved:
            running_best = current
            stale = 0
        else:
            stale += 1
        if stale >= hyperparameters.patience:
            break

    selection = select_checkpoint(
        history,
        grounded=bool(spec.grounded),
        min_representation_std=hyperparameters.min_representation_std,
        paired_baseline_state_loss=baseline_state,
    )
    selection.update(
        {
            "arm": spec.name,
            "architecture": spec.architecture,
            "seed_base": int(seed_base),
            "fold": int(fold),
            "effective_seed": effective_seed,
            "paired_initialization_sha256": paired_initialization_sha256,
            "stage_a_sentinel_sha256": authorization.sha256,
            "hyperparameters": asdict(hyperparameters),
            "train_patient_sha256": train_patient_sha256,
            "val_patient_sha256": val_patient_sha256,
            "data_provenance_sha256": data_provenance_sha256,
            "preregistration_status": preregistration_status,
            "preregistration_lock_sha256": preregistration_lock_sha256,
            "preregistration": preregistration_evidence,
            "history_sha256": file_sha256(output / "history.csv"),
            "test_data_used": False,
            "delta_ftv_used": False,
            "pcr_used": False,
        }
    )
    selection_path = output / "selection.json"
    _atomic_json(selection_path, selection)
    selected_epoch = int(selection["selected_epoch"])
    selected_payload = torch.load(
        output / f"epoch_{selected_epoch:02d}.pt",
        map_location="cpu",
        weights_only=True,
    )
    selected_payload["selected"] = True
    selected_payload["selection"] = selection
    selected_payload["selection_path"] = str(selection_path)
    selected_payload["selection_sha256"] = file_sha256(selection_path)
    _atomic_torch_save(output / "selected.pt", selected_payload)
    return selection


__all__ = [
    "BASELINE_BY_GROUNDED",
    "EFFECTIVE_BATCH_SIZE",
    "FOLDS",
    "FORMAL_HYPERPARAMETERS",
    "LOGICAL_OBJECTIVE_CONTRACT",
    "SEALED_SOURCE_SHA256",
    "SEED_BASES",
    "TrainHyperparameters",
    "formal_hyperparameters",
    "logical_patient_batches",
    "logical_sigreg_surrogate",
    "ordered_patient_sha256",
    "physical_patient_batches",
    "run_logical_train_epoch",
    "run_validation_epoch",
    "scale_microbatch_components",
    "seed_everything",
    "select_checkpoint",
    "train_epochs",
    "validate_formal_hyperparameters",
    "validate_paired_baseline",
    "validate_seed_fold",
    "verify_sealed_stage_b_sources",
]

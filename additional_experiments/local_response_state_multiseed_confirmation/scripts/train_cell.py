#!/usr/bin/env python3
"""Train one gated C1B LOCAL confirmation arm/seed/fold."""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Mapping


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
SRC_ROOT = EXPERIMENT_ROOT / "src"
SCRIPTS_ROOT = EXPERIMENT_ROOT / "scripts"
SEALED_EXPERIMENT_ROOT = (
    REPO_ROOT / "additional_experiments" / "c1b_overlap_eligibility_ftv_stageb"
)
DEFAULT_STAGE_A_SENTINEL = SEALED_EXPERIMENT_ROOT / "STAGE_A_GO.json"
DEFAULT_STAGE_A_SENTINEL_SHA256 = (
    "0b2c9e0af63ce8806525fb15ac9a27f6ab525b0259ccf16001981ca5091afbdb"
)
PRIVATE_INPUT_REPO_ROOT_ENV = "MWM_PRIVATE_INPUT_REPO_ROOT"
DATA_CONTRACT_REPOSITORY_RELATIVE = Path(
    "additional_experiments/c1b_overlap_eligibility_ftv_stageb/"
    "manifests/stage_b_data_contract.private.json"
)
DATA_CONTRACT_LOCK_KEY = DATA_CONTRACT_REPOSITORY_RELATIVE.as_posix()


def _resolve_default_data_contract(
    environ: Mapping[str, str] = os.environ,
) -> Path:
    configured = str(environ.get(PRIVATE_INPUT_REPO_ROOT_ENV, "")).strip()
    private_repository_root = (
        Path(configured).expanduser().resolve() if configured else REPO_ROOT.resolve()
    )
    return (private_repository_root / DATA_CONTRACT_REPOSITORY_RELATIVE).resolve()


DEFAULT_DATA_CONTRACT = _resolve_default_data_contract()
DEFAULT_DATA_CONTRACT_SHA256 = (
    "dd22f130043863d4fce8956061fca389894a31874567ed7929e139f32ff5ab27"
)
CHECKPOINT_ROOT = EXPERIMENT_ROOT / "checkpoints"
FORMAL_ARMS = ("GAP0", "GAP3", "LOCAL0", "LOCAL3")
FORMAL_SEED_BASES = (2026, 3026, 4026, 5026, 6026)
BASELINE_BY_GROUNDED = {"GAP3": "GAP0", "LOCAL3": "LOCAL0"}
SAFE_TAG = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
for source in (SRC_ROOT, SCRIPTS_ROOT):
    value = str(source.resolve())
    while value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)

from freeze_preregistration import verify as verify_preregistration  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage-a-sentinel", type=Path, default=DEFAULT_STAGE_A_SENTINEL
    )
    parser.add_argument(
        "--stage-a-sentinel-sha256", default=DEFAULT_STAGE_A_SENTINEL_SHA256
    )
    parser.add_argument("--data-contract", type=Path, default=DEFAULT_DATA_CONTRACT)
    parser.add_argument(
        "--data-contract-sha256", default=DEFAULT_DATA_CONTRACT_SHA256
    )
    parser.add_argument("--preregistration-lock-sha256", required=True)
    parser.add_argument("--arm", choices=FORMAL_ARMS, required=True)
    parser.add_argument(
        "--seed-base", type=int, choices=FORMAL_SEED_BASES, required=True
    )
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paired-baseline-selection", type=Path)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _resolve_device(value: str):
    import torch

    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _validate_initialization_report(
    report: dict[str, object],
    arm: str,
    model: object,
    effective_seed: int,
    *,
    arms: tuple[str, ...],
    shared_digest: object,
    transition_digest: object,
) -> str:
    if report.get("schema_version") != 1:
        raise ValueError("paired initialization report schema drifted")
    if report.get("effective_seed") != int(effective_seed):
        raise ValueError("paired initialization report effective seed mismatch")
    if report.get("arms") != list(arms):
        raise ValueError("paired initialization report arm inventory/order drifted")
    checks = report.get("checks")
    if not isinstance(checks, dict) or not checks or not all(
        value is True for value in checks.values()
    ):
        raise AssertionError("paired initialization report contains a failed check")
    per_arm = report.get("per_arm")
    if not isinstance(per_arm, dict) or tuple(per_arm) != arms:
        raise ValueError("paired initialization per-arm inventory/order drifted")
    if not all(isinstance(per_arm.get(name), dict) for name in arms):
        raise ValueError("paired initialization report lacks a formal arm record")
    shared = str(report.get("shared_initialization_sha256", ""))
    if re.fullmatch(r"[0-9a-f]{64}", shared) is None:
        raise ValueError("paired shared initialization digest is invalid")
    per_arm_shared = {
        str(per_arm[name].get("shared_initialization_sha256", ""))
        for name in arms
    }
    arm_shared = str(per_arm[arm].get("shared_initialization_sha256", ""))
    observed_shared = shared_digest(model)  # type: ignore[operator]
    if per_arm_shared != {shared} or shared != arm_shared or shared != observed_shared:
        raise AssertionError("paired shared initialization report/model mismatch")
    transition = str(report.get("transition_sha256", ""))
    if re.fullmatch(r"[0-9a-f]{64}", transition) is None:
        raise ValueError("paired transition initialization digest is invalid")
    per_arm_transitions = {
        str(per_arm[name].get("transition_sha256", "")) for name in arms
    }
    arm_transition = str(per_arm[arm].get("transition_sha256", ""))
    observed_transition = transition_digest(model)  # type: ignore[operator]
    if (
        per_arm_transitions != {transition}
        or transition != arm_transition
        or transition != observed_transition
    ):
        raise AssertionError("paired transition initialization report/model mismatch")
    if report.get("architecture_pairs") != {
        "GAP0_LOCAL0": shared,
        "GAP3_LOCAL3": shared,
    }:
        raise ValueError("paired GAP/LOCAL initialization evidence drifted")
    return shared


def _write_private_json(path: Path, payload: object) -> None:
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
        Path(temporary).replace(path)
        path.chmod(0o600)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _validate_cell_paths(
    output_dir: Path,
    paired_baseline_selection: Path | None,
    *,
    arm: str,
    seed_base: int,
    fold: int,
    checkpoint_root: Path = CHECKPOINT_ROOT,
) -> None:
    """Bind CLI identity and baseline pairing to one exact formal path."""

    if arm not in FORMAL_ARMS:
        raise ValueError(f"formal confirmation arm must be one of {FORMAL_ARMS}")
    if type(seed_base) is not int or seed_base not in FORMAL_SEED_BASES:
        raise ValueError(f"formal confirmation seed must be one of {FORMAL_SEED_BASES}")
    if type(fold) is not int or fold not in range(5):
        raise ValueError("formal confirmation fold must be one of 0..4")
    root = checkpoint_root.resolve()
    relative = output_dir.resolve().relative_to(root)
    expected_tail = (f"seed_{seed_base}", arm, f"fold_{fold}")
    if (
        len(relative.parts) != 4
        or SAFE_TAG.fullmatch(relative.parts[0]) is None
        or tuple(relative.parts[1:]) != expected_tail
    ):
        raise ValueError(
            "cell output must be checkpoints/<tag>/"
            f"seed_{seed_base}/{arm}/fold_{fold} with one safe tag component"
        )
    baseline_arm = BASELINE_BY_GROUNDED.get(arm)
    if baseline_arm is None:
        if paired_baseline_selection is not None:
            raise ValueError("ungrounded arms must not receive a paired baseline")
        return
    expected_baseline = (
        root
        / relative.parts[0]
        / f"seed_{seed_base}"
        / baseline_arm
        / f"fold_{fold}"
        / "selection.json"
    ).resolve()
    if paired_baseline_selection is None:
        raise ValueError(f"grounded arm {arm} requires its paired baseline selection")
    if paired_baseline_selection.resolve() != expected_baseline:
        raise ValueError(
            f"grounded arm {arm} must use {baseline_arm} from the same tag/seed/fold"
        )


def main() -> None:
    os.umask(0o077)
    preregistration = verify_preregistration()
    # Historical package name retained for exact pilot-checkpoint compatibility.
    import lg_response_pilot.model as pilot_model_module
    import lg_response_pilot.security as pilot_security
    import lg_response_pilot.training as pilot_training_module

    if Path(str(getattr(pilot_security, "__file__", ""))).resolve() != (
        SRC_ROOT / "lg_response_pilot" / "security.py"
    ).resolve():
        raise ImportError(
            "confirmation security module was shadowed before formal execution"
        )
    pilot_security.require_module_within(
        pilot_model_module,
        SRC_ROOT / "lg_response_pilot",
        label="confirmation model",
    )
    pilot_security.require_module_within(
        pilot_training_module,
        SRC_ROOT / "lg_response_pilot",
        label="confirmation training",
    )
    from lg_response_pilot.model import (
        ARMS,
        build_model,
        build_objective,
        paired_initialization_report,
        shared_initialization_sha256,
        transition_sha256,
    )
    from lg_response_pilot.security import (
        claim_private_directory,
        require_canonical_file,
        require_lock_sha256,
        resolve_contained_path,
    )
    from lg_response_pilot.training import (
        FOLDS,
        LOGICAL_OBJECTIVE_CONTRACT,
        SEALED_SRC,
        SEED_BASES,
        file_sha256,
        formal_hyperparameters,
        ordered_patient_sha256,
        seed_everything,
        train_epochs,
        validate_seed_fold,
        verify_sealed_stage_b_sources,
    )

    if tuple(ARMS) != FORMAL_ARMS:
        raise RuntimeError("formal confirmation arm inventory drifted")
    if tuple(SEED_BASES) != FORMAL_SEED_BASES or tuple(FOLDS) != tuple(range(5)):
        raise RuntimeError("formal confirmation seed/fold inventory drifted")
    sealed_value = str(SEALED_SRC.resolve())
    while sealed_value in sys.path:
        sys.path.remove(sealed_value)
    sys.path.insert(0, sealed_value)
    preregistration_evidence = {
        "status": preregistration["status"],
        "lock_sha256": preregistration["lock_sha256"],
        "config_sha256": preregistration["config_sha256"],
        "verified_code_files": preregistration["verified_code_files"],
    }
    args = parse_args()
    require_lock_sha256(
        preregistration["lock_sha256"], args.preregistration_lock_sha256
    )
    locked_upstream = preregistration["upstream_sha256"]
    args.stage_a_sentinel = require_canonical_file(
        args.stage_a_sentinel,
        DEFAULT_STAGE_A_SENTINEL,
        args.stage_a_sentinel_sha256,
        locked_upstream[str(DEFAULT_STAGE_A_SENTINEL.relative_to(REPO_ROOT))],
        label="Stage-A sentinel",
    )
    args.data_contract = require_canonical_file(
        args.data_contract,
        DEFAULT_DATA_CONTRACT,
        args.data_contract_sha256,
        locked_upstream[DATA_CONTRACT_LOCK_KEY],
        label="Stage-B data contract",
    )
    output_dir = resolve_contained_path(
        args.output_dir,
        CHECKPOINT_ROOT,
        label="confirmation cell output directory",
    )
    if args.paired_baseline_selection is not None:
        paired_baseline_selection = resolve_contained_path(
            args.paired_baseline_selection,
            CHECKPOINT_ROOT,
            label="paired baseline selection",
        )
    else:
        paired_baseline_selection = None
    _validate_cell_paths(
        output_dir,
        paired_baseline_selection,
        arm=args.arm,
        seed_base=args.seed_base,
        fold=args.fold,
    )
    sealed_hashes = verify_sealed_stage_b_sources()

    # Import sealed data/target code only after its complete source inventory is
    # re-hashed above.  No copy of these private contracts is created here.
    from c1b_stage_b.data import StageBDataset, make_splits
    from c1b_stage_b.gate import require_stage_a_go
    from c1b_stage_b.inputs import StageBDataPaths, load_stage_b_data
    from c1b_stage_b.targets import fit_grounding_transform

    import c1b_stage_b.data as sealed_data_module
    import c1b_stage_b.gate as sealed_gate_module
    import c1b_stage_b.inputs as sealed_inputs_module
    import c1b_stage_b.targets as sealed_targets_module

    for module, label in (
        (sealed_data_module, "sealed Stage-B data"),
        (sealed_gate_module, "sealed Stage-B gate"),
        (sealed_inputs_module, "sealed Stage-B inputs"),
        (sealed_targets_module, "sealed Stage-B targets"),
    ):
        pilot_security.require_module_within(module, SEALED_SRC, label=label)

    authorization = require_stage_a_go(args.stage_a_sentinel)
    paths = StageBDataPaths.load(args.data_contract, args.data_contract_sha256)
    data = load_stage_b_data(paths, authorization, verify_cache_files=False)
    effective_seed = validate_seed_fold(args.seed_base, args.fold)
    splits = make_splits(data.folds, args.fold, data.train_only_ids)
    transform, transformed_ftv = fit_grounding_transform(
        data.ftv,
        splits.train_primary,
        args.fold,
        apply_ids=splits.train_primary + splits.val,
    )
    train_dataset = StageBDataset(
        splits.train_all, data.c1b_cache, transformed_ftv
    )
    val_dataset = StageBDataset(splits.val, data.c1b_cache, transformed_ftv)
    hyperparameters = formal_hyperparameters()

    seed_everything(effective_seed)
    paired = paired_initialization_report(effective_seed)
    model = build_model(args.arm, effective_seed)
    objective = build_objective(args.arm)
    paired_shared_sha256 = _validate_initialization_report(
        paired,
        args.arm,
        model,
        effective_seed,
        arms=tuple(ARMS),
        shared_digest=shared_initialization_sha256,
        transition_digest=transition_sha256,
    )

    source_root = SRC_ROOT / "lg_response_pilot"
    provenance = {
        **data.provenance,
        "ftv_transform": transform.to_dict(),
        "sealed_stage_b_source_sha256": sealed_hashes,
        "confirmation_code_sha256": {
            "model.py": file_sha256(source_root / "model.py"),
            "pooling.py": file_sha256(source_root / "pooling.py"),
            "training.py": file_sha256(source_root / "training.py"),
            "train_cell.py": file_sha256(Path(__file__)),
        },
        "train_primary_order_sha256": ordered_patient_sha256(splits.train_primary),
        "train_all_order_sha256": ordered_patient_sha256(splits.train_all),
        "validation_order_sha256": ordered_patient_sha256(splits.val),
        "test_patient_count_not_loaded": len(splits.test),
        "input_kind": "c1b",
        "model_forward_fields": ["image"],
        "auxiliary_fields": ["ftv_target", "ftv_mask"],
        "logical_objective_contract": dict(LOGICAL_OBJECTIVE_CONTRACT),
        "physical_batch_size": 4,
        "accumulation_steps": 8,
        "effective_batch_size": 32,
        "optimizer": "AdamW",
        "response_state_only_experimental_factor": True,
        "preregistration": preregistration_evidence,
        "config_sha256": preregistration["config_sha256"],
        "data_contract_sha256": args.data_contract_sha256,
        "stage_a_sentinel_sha256": args.stage_a_sentinel_sha256,
    }
    claim_private_directory(
        output_dir,
        CHECKPOINT_ROOT,
        label="confirmation cell output directory",
    )
    selection = train_epochs(
        arm=args.arm,
        seed_base=args.seed_base,
        fold=args.fold,
        model=model,
        objective=objective,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=_resolve_device(args.device),
        output_dir=output_dir,
        authorization=authorization,
        hyperparameters=hyperparameters,
        paired_initialization_sha256=paired_shared_sha256,
        data_provenance=provenance,
        preregistration=preregistration_evidence,
        paired_baseline_selection=paired_baseline_selection,
    )
    transform.save(output_dir / "ftv_transform.json")
    (output_dir / "ftv_transform.json").chmod(0o600)
    _write_private_json(output_dir / "paired_initialization.json", paired)
    print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

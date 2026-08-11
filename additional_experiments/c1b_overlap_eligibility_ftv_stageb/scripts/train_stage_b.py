#!/usr/bin/env python3
"""Train one gated Stage B arm/seed/fold; never reads an outcome table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c1b_stage_b.cli import (  # noqa: E402
    add_data_contract_arguments,
    add_gate_arguments,
    authorize,
    data_paths,
    resolve_device,
)
from c1b_stage_b.contracts import (  # noqa: E402
    ARMS,
    G3_SRC,
    LOGICAL_OBJECTIVE_CONTRACT,
    file_sha256,
    ordered_patient_sha256,
    validate_seed_fold,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_gate_arguments(parser)
    add_data_contract_arguments(parser)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed-base", type=int, choices=(2026, 3026), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paired-baseline-selection", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--physical-batch-size", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument(
        "--global-fallback-restart",
        action="store_true",
        help="Required with 2/16; provenance assertion that every arm is restarted globally.",
    )
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Hard stop occurs before reading the data contract, any cache, or output.
    authorization = authorize(args)
    from c1b_stage_b.data import StageBDataset, arm_cache, make_splits
    from c1b_stage_b.inputs import load_stage_b_data
    from c1b_stage_b.targets import fit_grounding_transform
    from c1b_stage_b.training import TrainHyperparameters, train_epochs
    from c1b_stage_b.upstream import (
        build_model,
        build_objective,
        paired_initialization_report,
        seed_everything,
    )

    paths = data_paths(args)
    data = load_stage_b_data(paths, authorization, verify_cache_files=False)
    effective_seed = validate_seed_fold(args.seed_base, args.fold)
    splits = make_splits(data.folds, args.fold, data.train_only_ids)
    transform, transformed_ftv = fit_grounding_transform(
        data.ftv,
        splits.train_primary,
        args.fold,
        apply_ids=splits.train_primary + splits.val,
    )
    cache = arm_cache(args.arm, data.legacy_cache, data.c1b_cache)
    train_dataset = StageBDataset(splits.train_all, cache, transformed_ftv)
    val_dataset = StageBDataset(splits.val, cache, transformed_ftv)
    hyperparameters = TrainHyperparameters(
        physical_batch_size=args.physical_batch_size,
        accumulation_steps=args.accumulation_steps,
        workers=args.workers,
    )
    hyperparameters.validate()
    fallback = (args.physical_batch_size, args.accumulation_steps) == (2, 16)
    if fallback != bool(args.global_fallback_restart):
        raise ValueError(
            "2/16 requires --global-fallback-restart, and that flag is invalid for 4/8"
        )
    seed_everything(effective_seed)
    paired = paired_initialization_report(effective_seed)
    model = build_model(args.arm, effective_seed)
    objective = build_objective(args.arm)
    provenance = {
        **data.provenance,
        "ftv_transform": transform.to_dict(),
        "ftv_target_implementation_sha256": file_sha256(
            G3_SRC / "dgrs" / "targets.py"
        ),
        "upstream_model_implementation_sha256": paired["upstream_model_sha256"],
        "upstream_objective_implementation_sha256": paired[
            "upstream_objective_sha256"
        ],
        "stage_b_code_sha256": {
            name: file_sha256(ROOT / "src" / "c1b_stage_b" / f"{name}.py")
            for name in (
                "contracts",
                "data",
                "gate",
                "inputs",
                "targets",
                "training",
                "upstream",
            )
        },
        "train_primary_order_sha256": ordered_patient_sha256(splits.train_primary),
        "train_all_order_sha256": ordered_patient_sha256(splits.train_all),
        "validation_order_sha256": ordered_patient_sha256(splits.val),
        "test_patient_count_not_loaded": len(splits.test),
        "model_forward_fields": ["image"],
        "auxiliary_fields": ["ftv_target", "ftv_mask"],
        "logical_objective_contract": dict(LOGICAL_OBJECTIVE_CONTRACT),
        "global_fallback_restart": bool(args.global_fallback_restart),
    }
    selection = train_epochs(
        arm=args.arm,
        seed_base=args.seed_base,
        fold=args.fold,
        model=model,
        objective=objective,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=resolve_device(args.device),
        output_dir=args.output_dir,
        authorization=authorization,
        hyperparameters=hyperparameters,
        paired_initialization_sha256=paired["common_initialization_sha256"],
        data_provenance=provenance,
        paired_baseline_selection=args.paired_baseline_selection,
    )
    transform.save(args.output_dir / "ftv_transform.json")
    (args.output_dir / "paired_initialization.json").write_text(
        json.dumps(paired, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

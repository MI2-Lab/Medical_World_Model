#!/usr/bin/env python3
"""Train one pCR-blind S1/S2 residual-SPH pilot cell."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

import numpy as np


os.umask(0o077)
sys.dont_write_bytecode = True
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
SRC = EXPERIMENT_ROOT / "src"
sys.path.insert(0, str(SRC))

from residual_sph.contracts import (  # noqa: E402
    TrainHyperparameters,
    file_sha256,
    validate_seed_fold,
)
from residual_sph.data import StaticSPHDataset, target_mapping  # noqa: E402
from residual_sph.losses import build_objective  # noqa: E402
from residual_sph.model import (  # noqa: E402
    build_model,
    paired_initialization_report,
    shared_initialization_sha256,
)
from residual_sph.preregistration import (  # noqa: E402
    require_lock_sha256,
    verify_preregistration,
)
from residual_sph.provenance import load_s0_manifest, validate_s0_cell  # noqa: E402
from residual_sph.targets import (  # noqa: E402
    canonical_sha256,
    fit_fold_target_bundle,
    load_fold_split_map,
    load_static_sph_ftv_table,
)
from residual_sph.training import seed_everything, train_epochs  # noqa: E402
from residual_sph.upstream import verify_local_sources  # noqa: E402


SEALED_ROOT = REPO_ROOT / "additional_experiments" / "c1b_overlap_eligibility_ftv_stageb"
SEALED_SRC = SEALED_ROOT / "src"
DEFAULT_SENTINEL = SEALED_ROOT / "STAGE_A_GO.json"
DEFAULT_SENTINEL_SHA256 = "0b2c9e0af63ce8806525fb15ac9a27f6ab525b0259ccf16001981ca5091afbdb"
DEFAULT_DATA_CONTRACT = SEALED_ROOT / "manifests" / "stage_b_data_contract.private.json"
DEFAULT_DATA_CONTRACT_SHA256 = "dd22f130043863d4fce8956061fca389894a31874567ed7929e139f32ff5ab27"
DEFAULT_TARGET_TABLE = (
    REPO_ROOT
    / "additional_experiments/radiomics_next_change/data_audit/radiomics_transition_targets_raw.csv"
)
DEFAULT_TARGET_SHA256 = "26fbde8590fde4612267f02d762af99d65926ff6d0206d0e500577ef394ff75d"
DEFAULT_FOLD_SHA256 = "143e482d711225c0611006d99bd7345d2fa1a5c16c65fbaf8399341a0d26aa38"
DEFAULT_S0_ROOT = (
    REPO_ROOT
    / "additional_experiments/local_response_state_multiseed_confirmation/"
    "checkpoints/formal_4x8"
)
DEFAULT_S0_MANIFEST = EXPERIMENT_ROOT / "manifests" / "s0_confirmation_provenance.json"
CHECKPOINT_ROOT = EXPERIMENT_ROOT / "checkpoints"
SAFE_TAG = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("S1", "S2", "S2_L10"), required=True)
    parser.add_argument("--seed-base", choices=(2026, 3026), type=int, required=True)
    parser.add_argument("--fold", choices=range(5), type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preregistration-lock-sha256", required=True)
    parser.add_argument("--stage-a-sentinel", type=Path, default=DEFAULT_SENTINEL)
    parser.add_argument("--data-contract", type=Path, default=DEFAULT_DATA_CONTRACT)
    parser.add_argument("--target-table", type=Path, default=DEFAULT_TARGET_TABLE)
    parser.add_argument(
        "--fold-manifest",
        type=Path,
        help="Optional private override; otherwise use the hash-locked Stage-B contract",
    )
    parser.add_argument("--s0-root", type=Path, default=DEFAULT_S0_ROOT)
    parser.add_argument("--s0-manifest", type=Path, default=DEFAULT_S0_MANIFEST)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Verify one cell's contracts and aggregate counts without reading images or training",
    )
    return parser.parse_args()


def _require_file(path: Path, expected_sha256: str, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing {label}: {resolved}")
    observed = file_sha256(resolved)
    if observed != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 drift: expected {expected_sha256}, observed {observed}"
        )
    return resolved


def _validate_output(path: Path, *, arm: str, seed: int, fold: int) -> Path:
    resolved = path.resolve()
    root = CHECKPOINT_ROOT.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("output must be contained in the experiment checkpoint root") from error
    expected = (f"seed_{seed}", arm, f"fold_{fold}")
    if (
        len(relative.parts) != 4
        or SAFE_TAG.fullmatch(relative.parts[0]) is None
        or tuple(relative.parts[1:]) != expected
    ):
        raise ValueError(
            "output must be checkpoints/<safe-tag>/seed_<seed>/<arm>/fold_<fold>"
        )
    return resolved


def _resolve_device(value: str):
    import torch

    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def main() -> None:
    args = parse_args()
    preregistration = verify_preregistration(EXPERIMENT_ROOT)
    require_lock_sha256(
        preregistration["lock_sha256"], args.preregistration_lock_sha256
    )
    verify_local_sources()
    sentinel = _require_file(args.stage_a_sentinel, DEFAULT_SENTINEL_SHA256, "Stage-A sentinel")
    data_contract = _require_file(
        args.data_contract, DEFAULT_DATA_CONTRACT_SHA256, "Stage-B data contract"
    )
    target_table_path = _require_file(
        args.target_table, DEFAULT_TARGET_SHA256, "private FTV/SPH target table"
    )
    output = _validate_output(
        args.output_dir,
        arm=args.arm,
        seed=args.seed_base,
        fold=args.fold,
    )
    effective_seed = validate_seed_fold(args.seed_base, args.fold)

    sealed_value = str(SEALED_SRC.resolve())
    while sealed_value in sys.path:
        sys.path.remove(sealed_value)
    sys.path.insert(0, sealed_value)
    from c1b_stage_b.data import StageBDataset, make_splits
    from c1b_stage_b.gate import require_stage_a_go
    from c1b_stage_b.inputs import StageBDataPaths, load_stage_b_data
    from c1b_stage_b.targets import fit_grounding_transform

    authorization = require_stage_a_go(sentinel)
    paths = StageBDataPaths.load(data_contract, DEFAULT_DATA_CONTRACT_SHA256)
    fold_manifest = _require_file(
        args.fold_manifest if args.fold_manifest is not None else paths.fold_manifest,
        DEFAULT_FOLD_SHA256,
        "outer-fold manifest",
    )
    data = load_stage_b_data(paths, authorization, verify_cache_files=False)
    splits = make_splits(data.folds, args.fold, data.train_only_ids)
    ftv_transform, transformed_ftv = fit_grounding_transform(
        data.ftv,
        splits.train_primary,
        args.fold,
        apply_ids=splits.train_primary + splits.val,
    )

    table = load_static_sph_ftv_table(
        target_table_path,
        DEFAULT_TARGET_SHA256,
        expected_patient_count=375,
    )
    split_map = load_fold_split_map(
        fold_manifest,
        DEFAULT_FOLD_SHA256,
        fold=args.fold,
        expected_patient_ids=table.patient_ids,
    )
    bundle = fit_fold_target_bundle(table, split_map, fold=args.fold)
    full_split = {
        **{patient_id: "train" for patient_id in splits.train_primary},
        **{patient_id: "val" for patient_id in splits.val},
        **{patient_id: "test" for patient_id in splits.test},
    }
    if len(full_split) != sum(
        len(values) for values in (splits.train_primary, splits.val, splits.test)
    ):
        raise ValueError("frozen primary fold partitions overlap")
    if any(
        patient_id not in full_split or full_split[patient_id] != split
        for patient_id, split in zip(bundle.patient_ids, bundle.splits, strict=True)
    ):
        raise ValueError("SPH complete-case cohort is not a split-preserving subset")
    target_values = bundle.target_matrix(args.arm)
    valid = np.ones_like(target_values, dtype=bool)
    all_targets = target_mapping(bundle.patient_ids, target_values, valid)

    train_base = StageBDataset(splits.train_all, data.c1b_cache, transformed_ftv)
    val_base = StageBDataset(splits.val, data.c1b_cache, transformed_ftv)
    train_targets = {
        patient_id: all_targets[patient_id]
        for patient_id in splits.train_primary
        if patient_id in all_targets
    }
    val_targets = {
        patient_id: all_targets[patient_id]
        for patient_id in splits.val
        if patient_id in all_targets
    }
    train_dataset = StaticSPHDataset(train_base, train_targets, args.arm)
    val_dataset = StaticSPHDataset(val_base, val_targets, args.arm)

    s0_selection = (
        args.s0_root.expanduser().resolve()
        / f"seed_{args.seed_base}"
        / "LOCAL3"
        / f"fold_{args.fold}"
        / "selection.json"
    )
    if not s0_selection.is_file():
        raise FileNotFoundError(f"missing confirmed S0 selection: {s0_selection}")
    s0_checkpoint = s0_selection.with_name("selected.pt")
    s0_manifest = load_s0_manifest(args.s0_manifest)
    s0_cell = validate_s0_cell(
        s0_manifest,
        seed_base=args.seed_base,
        fold=args.fold,
        selection_path=s0_selection,
        checkpoint_path=s0_checkpoint,
    )

    hyperparameters = TrainHyperparameters()
    train_patient_sha256 = canonical_sha256(sorted(splits.train_all))
    val_patient_sha256 = canonical_sha256(sorted(splits.val))
    if s0_cell["train_patient_sha256"] != train_patient_sha256:
        raise ValueError("paired S0/new-arm training patient set differs")
    if s0_cell["val_patient_sha256"] != val_patient_sha256:
        raise ValueError("paired S0/new-arm validation patient set differs")
    if s0_cell["hyperparameters"] != hyperparameters.to_dict():
        raise ValueError("paired S0/new-arm optimization hyperparameters differ")
    s0_ftv_transform = s0_selection.with_name("ftv_transform.json")
    if (
        not s0_ftv_transform.is_file()
        or file_sha256(s0_ftv_transform) != s0_cell["ftv_transform_sha256"]
        or json.loads(s0_ftv_transform.read_text(encoding="utf-8"))
        != ftv_transform.to_dict()
    ):
        raise ValueError("paired S0/new-arm FTV grounding transform differs")
    seed_everything(effective_seed)
    paired = paired_initialization_report(effective_seed)
    model = build_model(args.arm, effective_seed)
    objective = build_objective(args.arm)
    if paired["shared_initialization_sha256"] != shared_initialization_sha256(model):
        raise AssertionError("paired initialization report/model mismatch")

    if args.preflight_only:
        preflight = {
            "schema_version": 1,
            "status": "CELL_PREFLIGHT_PASS",
            "arm": args.arm,
            "seed_base": args.seed_base,
            "fold": args.fold,
            "effective_seed": effective_seed,
            "train_all_count": len(splits.train_all),
            "train_primary_count": len(splits.train_primary),
            "validation_count": len(splits.val),
            "held_out_test_count_not_loaded": len(splits.test),
            "train_sph_patient_count": len(train_targets),
            "validation_sph_patient_count": len(val_targets),
            "broader_train_patients_have_sph_mask_false": (
                len(train_targets) < len(splits.train_all)
            ),
            "sph_head_parameter_count": sum(
                parameter.numel() for parameter in model.sph_head.parameters()
            ),
            "shared_initialization_sha256": shared_initialization_sha256(model),
            "s0_selection_sha256": file_sha256(s0_selection),
            "pcr_or_clinical_loaded": False,
            "image_cache_opened": False,
            "training_started": False,
            "preregistration_lock_sha256": preregistration["lock_sha256"],
        }
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return

    provenance = {
        **dict(data.provenance),
        "ftv_grounding_transform": ftv_transform.to_dict(),
        "sph_residualizer_public_sha256": canonical_sha256(bundle.to_public_dict()),
        "target_table_sha256": table.source_sha256,
        "outer_fold_manifest_sha256": DEFAULT_FOLD_SHA256,
        "train_primary_patient_set_sha256": canonical_sha256(sorted(splits.train_primary)),
        "train_all_patient_set_sha256": train_patient_sha256,
        "validation_patient_set_sha256": val_patient_sha256,
        "test_patient_count_not_loaded": len(splits.test),
        "model_forward_fields": ["image"],
        "loss_side_fields": ["ftv_target", "ftv_mask", "sph_target", "sph_mask"],
        "pcr_or_clinical_loaded": False,
        "preregistration_lock_sha256": preregistration["lock_sha256"],
        "implementation_lock_sha256": preregistration["implementation_lock_sha256"],
    }
    selection = train_epochs(
        experimental_arm=args.arm,
        seed_base=args.seed_base,
        fold=args.fold,
        model=model,
        objective=objective,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=_resolve_device(args.device),
        output_dir=output,
        hyperparameters=hyperparameters,
        s0_selection_path=s0_selection,
        preregistration_lock_sha256=preregistration["lock_sha256"],
        data_provenance=provenance,
    )
    ftv_transform.save(output / "ftv_transform.json")
    (output / "ftv_transform.json").chmod(0o600)
    print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train one preregistered, pCR-free A1 seed/fold cell."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time

import numpy as np
import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))
sys.path.insert(
    0,
    str(
        REPO_ROOT
        / "additional_experiments"
        / "c1b_overlap_eligibility_ftv_stageb"
        / "src"
    ),
)
sys.path.insert(0, str(EXPERIMENT_ROOT / "scripts"))

from freeze_preregistration import file_sha256, verify  # noqa: E402
from patch_token_wm.data import (  # noqa: E402
    ConditionEncoder,
    ConditionedStageBDataset,
    load_authorized_condition_table,
)
from patch_token_wm.model import PatchTokenWorldModel  # noqa: E402
from patch_token_wm.objective import PatchTokenObjective  # noqa: E402
from patch_token_wm.training import TrainHyperparameters, train_epochs  # noqa: E402
from c1b_stage_b.data import StageBDataset, make_splits  # noqa: E402
from c1b_stage_b.gate import require_stage_a_go  # noqa: E402
from c1b_stage_b.inputs import StageBDataPaths, load_stage_b_data  # noqa: E402
from c1b_stage_b.targets import fit_grounding_transform  # noqa: E402


DEFAULT_STAGE_A_SENTINEL = (
    REPO_ROOT
    / "additional_experiments"
    / "c1b_overlap_eligibility_ftv_stageb"
    / "STAGE_A_GO.json"
)
DEFAULT_DATA_CONTRACT = (
    REPO_ROOT.parent
    / "Medical_World_Model"
    / "additional_experiments"
    / "c1b_overlap_eligibility_ftv_stageb"
    / "manifests"
    / "stage_b_data_contract.private.json"
)
DEFAULT_DATA_CONTRACT_SHA256 = (
    "dd22f130043863d4fce8956061fca389894a31874567ed7929e139f32ff5ab27"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-base", type=int, choices=(2026, 3026), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--physical-batch-size", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument(
        "--stage-a-sentinel", type=Path, default=DEFAULT_STAGE_A_SENTINEL
    )
    parser.add_argument("--data-contract", type=Path, default=DEFAULT_DATA_CONTRACT)
    parser.add_argument("--data-contract-sha256", default=DEFAULT_DATA_CONTRACT_SHA256)
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type != "cuda" or device.index is None:
        raise ValueError(
            "formal training requires an explicit CUDA device such as cuda:0"
        )
    if not torch.cuda.is_available() or device.index >= torch.cuda.device_count():
        raise RuntimeError(f"requested unavailable device {device}")
    return device


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def _primary_patient_order(data: object) -> tuple[str, ...]:
    frame = data.folds
    first = frame.loc[frame["fold"].eq(0), "patient_id"].astype(str)
    identities = tuple(first)
    if len(identities) != 808 or len(set(identities)) != 808:
        raise ValueError(
            "formal primary fold population must contain 808 unique patients"
        )
    return identities


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> dict[str, object]:
    args = parse_args()
    lock_result = verify()
    authorization = require_stage_a_go(args.stage_a_sentinel)
    paths = StageBDataPaths.load(args.data_contract, args.data_contract_sha256)
    data = load_stage_b_data(paths, authorization, verify_cache_files=False)
    splits = make_splits(data.folds, args.fold, data.train_only_ids)
    if len(data.train_only_ids) != 139:
        raise ValueError(
            "formal external train-only population must contain 139 patients"
        )
    transform, transformed_ftv = fit_grounding_transform(
        data.ftv,
        splits.train_primary,
        args.fold,
        apply_ids=splits.train_all + splits.val,
    )
    primary_ids = _primary_patient_order(data)
    condition_table = load_authorized_condition_table(
        primary_patient_ids=primary_ids,
        authorized_external_train_only_patient_ids=data.train_only_ids,
    )
    condition_encoder = ConditionEncoder.fit(
        condition_table,
        outer_train_patient_ids=splits.train_primary,
        authorized_external_train_only_patient_ids=data.train_only_ids,
    )
    train_base = StageBDataset(splits.train_all, data.c1b_cache, transformed_ftv)
    val_base = StageBDataset(splits.val, data.c1b_cache, transformed_ftv)
    train_dataset = ConditionedStageBDataset(
        train_base, condition_encoder, split="train", include_patient_id=True
    )
    val_dataset = ConditionedStageBDataset(
        val_base, condition_encoder, split="val", include_patient_id=True
    )
    hyperparameters = TrainHyperparameters(
        physical_batch_size=args.physical_batch_size,
        accumulation_steps=args.accumulation_steps,
        workers=args.workers,
    )
    hyperparameters.validate()
    device = resolve_device(args.device)
    effective_seed = int(args.seed_base) + int(args.fold)
    seed_everything(effective_seed)
    model = PatchTokenWorldModel()
    objective = PatchTokenObjective()
    condition_metadata = condition_encoder.aggregate_metadata()
    data_provenance = {
        **data.provenance,
        "ftv_transform": transform.to_dict(),
        "condition": condition_metadata,
        "formal_test_data_loaded": False,
        "pcr_table_loaded": False,
        "model_forward_fields": ["image", "condition", "patient_id_hash_key"],
        "auxiliary_fields": ["ftv_target", "ftv_mask"],
    }

    final_output = args.output_dir.resolve()
    if final_output.exists():
        raise FileExistsError(f"refusing to overwrite formal cell {final_output}")
    working = final_output.parent / f".{final_output.name}.working.{os.getpid()}"
    if working.exists():
        raise FileExistsError(f"working directory already exists: {working}")
    started = time.monotonic()
    selection = train_epochs(
        seed_base=args.seed_base,
        fold=args.fold,
        model=model,
        objective=objective,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        output_dir=working,
        hyperparameters=hyperparameters,
        preregistration_lock_sha256=str(lock_result["lock_sha256"]),
        data_provenance=data_provenance,
        condition_metadata=condition_metadata,
    )
    transform.save(working / "ftv_transform.json")
    cell_manifest = {
        "schema_version": 1,
        "status": "COMPLETE",
        "arm": "A1_PATCH3",
        "seed_base": int(args.seed_base),
        "fold": int(args.fold),
        "effective_seed": effective_seed,
        "selected_epoch": int(selection["selected_epoch"]),
        "finite_noncollapsed_selection": bool(
            selection.get("optimization_safety_pass")
        ),
        "selected_checkpoint_sha256": file_sha256(working / "selected.pt"),
        "selection_sha256": file_sha256(working / "selection.json"),
        "preregistration_lock_sha256": str(lock_result["lock_sha256"]),
        "training_is_pcr_free": True,
        "test_data_used": False,
        "wall_seconds": time.monotonic() - started,
    }
    _atomic_json(working / "cell_complete.json", cell_manifest)
    final_output.parent.mkdir(parents=True, exist_ok=True)
    working.replace(final_output)
    directory = os.open(
        final_output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    print(json.dumps(cell_manifest, indent=2, sort_keys=True))
    return cell_manifest


if __name__ == "__main__":
    main()

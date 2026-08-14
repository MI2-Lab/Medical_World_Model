#!/usr/bin/env python3
"""Train one preregistered, pCR-blind Goal-F representation cell."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping


sys.dont_write_bytecode = True
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
SRC_ROOT = EXPERIMENT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from crps.contracts import (  # noqa: E402
    PRIMARY_ARMS,
    PCR_LABEL_ACCESS,
    arm_spec,
    assert_representation_config,
    canonical_sha256,
    file_sha256,
    load_json,
    validate_seed_fold,
)
from crps.data import ProfiledStageBDataset, load_training_profiles  # noqa: E402
from crps.losses import FactorizedObjective, LogicalLossWeights  # noqa: E402
from crps.model import build_model  # noqa: E402
from crps.preregistration import LOCK_PATH, verify as verify_preregistration  # noqa: E402
from crps.stageb import (  # noqa: E402
    StageBDataPaths,
    StageBDataset,
    fit_grounding_transform,
    load_stage_b_data,
    make_splits,
    require_stage_a_go,
)
from crps.training import (  # noqa: E402
    TrainHyperparameters,
    seed_everything,
    train_epochs,
)


CONFIG_PATH = EXPERIMENT_ROOT / "configs" / "representation.json"
CHECKPOINT_ROOT = EXPERIMENT_ROOT / "checkpoints"
FORMAL_TAG = "formal_primary"
HEX_SHA256 = re.compile(r"[0-9a-f]{64}")


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _require_digest(value: str, label: str) -> str:
    digest = str(value).strip().casefold()
    if HEX_SHA256.fullmatch(digest) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return digest


def _require_locked_file(path: Path, expected: str, label: str) -> Path:
    resolved = path.expanduser().resolve()
    digest = _require_digest(expected, f"{label} digest")
    if file_sha256(resolved) != digest:
        raise PermissionError(f"{label} SHA-256 mismatch")
    return resolved


def _validate_output_path(path: Path, arm: str, seed_base: int, fold: int) -> Path:
    resolved = path.expanduser().resolve()
    expected = (
        CHECKPOINT_ROOT
        / FORMAL_TAG
        / f"seed_{seed_base}"
        / arm
        / f"fold_{fold}"
    ).resolve()
    if resolved != expected:
        raise ValueError(f"formal cell output must be exactly {expected}")
    return resolved


def _tensor_digest(state: Mapping[str, Any]) -> str:
    import torch

    digest = hashlib.sha256()
    for name in sorted(state):
        if name.startswith(("hr_adversary.", "her2_adversary.")):
            continue
        value = state[name]
        if not torch.is_tensor(value):
            raise TypeError(f"state value is not a tensor: {name}")
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def paired_initialization(arm: str, condition_dim: int, effective_seed: int):
    """Build F1/F2 from the same seed and prove every shared tensor is paired."""

    models = {
        name: build_model(name, condition_dim, effective_seed)
        for name in PRIMARY_ARMS
    }
    common_keys = set(models["F1"].state_dict())
    f2_common = {
        key
        for key in models["F2"].state_dict()
        if not key.startswith(("hr_adversary.", "her2_adversary."))
    }
    if common_keys != f2_common:
        raise AssertionError("F1/F2 common initialization inventories differ")
    for key in sorted(common_keys):
        left = models["F1"].state_dict()[key]
        right = models["F2"].state_dict()[key]
        if not left.equal(right):
            raise AssertionError(f"F1/F2 shared initialization differs: {key}")
    digests = {name: _tensor_digest(model.state_dict()) for name, model in models.items()}
    if len(set(digests.values())) != 1:
        raise AssertionError("F1/F2 shared initialization digest differs")
    report = {
        "schema_version": 1,
        "arms": list(PRIMARY_ARMS),
        "effective_seed": int(effective_seed),
        "shared_tensor_count": len(common_keys),
        "shared_initialization_sha256": digests["F1"],
        "checks": {
            "common_key_inventory_equal": True,
            "common_tensors_bitwise_equal": True,
            "total_state_dim_paired": True,
        },
    }
    selected = models[arm]
    for name, model in models.items():
        if name != arm:
            del model
    return selected, report


def _hyperparameters(config: Mapping[str, Any]) -> TrainHyperparameters:
    training = config["training"]
    augmentation = config["augmentation"]
    return TrainHyperparameters(
        physical_batch_size=int(training["physical_batch_size"]),
        accumulation_steps=int(training["accumulation_steps"]),
        workers=int(training["workers"]),
        epochs=int(training["epochs"]),
        patience=int(training["patience"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        ema_momentum=float(training["ema_momentum"]),
        max_grad_norm=float(training["max_grad_norm"]),
        min_response_std=float(training["min_response_std"]),
        min_phenotype_std=float(training["min_phenotype_std"]),
        min_phenotype_effective_rank=float(training["min_phenotype_effective_rank"]),
        min_augmentation_cosine=float(training["min_augmentation_cosine"]),
        augmentation_scale_half_width=float(augmentation["scale_half_width"]),
        augmentation_shift_half_width=float(augmentation["shift_half_width"]),
        augmentation_noise_std=float(augmentation["gaussian_noise_std"]),
    )


def _objective(arm: str, config: Mapping[str, Any]) -> FactorizedObjective:
    loss = config["loss"]
    weights = LogicalLossWeights(
        lambda_ftv=float(loss["lambda_ftv"]),
        lambda_phenotype_future=float(loss["lambda_phenotype_future"]),
        lambda_phenotype_consistency=float(loss["lambda_phenotype_consistency"]),
        lambda_crosscov=float(loss["lambda_crosscov"]),
        sigreg_weight=float(loss["sigreg_weight"]),
        consistency_invariance=float(loss["consistency_invariance"]),
        consistency_variance=float(loss["consistency_variance"]),
        consistency_covariance=float(loss["consistency_covariance"]),
        variance_target=float(loss["variance_target"]),
    )
    return FactorizedObjective(
        arm,
        weights,
        sigreg_projections=int(loss["sigreg_projections"]),
        step_weights=tuple(float(value) for value in loss["step_weights"]),
    )


def _atomic_private_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        path.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=PRIMARY_ARMS)
    parser.add_argument("--seed-base", required=True, type=int, choices=(2026, 3026))
    parser.add_argument("--fold", required=True, type=int, choices=range(5))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preregistration-lock-sha256", required=True)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="verify every frozen input/contract and initialize the paired model without writing",
    )
    return parser.parse_args()


def main() -> None:
    if PCR_LABEL_ACCESS != "FORBIDDEN":
        raise PermissionError("PCR_LABEL_ACCESS must remain FORBIDDEN")
    args = parse_args()
    arm_spec(args.arm)
    effective_seed = validate_seed_fold(args.seed_base, args.fold)
    output_dir = _validate_output_path(
        args.output_dir, args.arm, args.seed_base, args.fold
    )
    config = load_json(CONFIG_PATH)
    assert_representation_config(config)
    preregistration = verify_preregistration()
    lock_sha256 = _require_digest(
        args.preregistration_lock_sha256, "preregistration lock digest"
    )
    if file_sha256(LOCK_PATH) != lock_sha256:
        raise PermissionError("preregistration lock file SHA-256 mismatch")

    upstream = config["upstream"]
    stage_a_path = _require_locked_file(
        _repo_path(upstream["stage_a_sentinel"]),
        upstream["stage_a_sentinel_sha256"],
        "Stage-A sentinel",
    )
    contract_path = _require_locked_file(
        _repo_path(upstream["stage_b_data_contract"]),
        upstream["stage_b_data_contract_sha256"],
        "Stage-B data contract",
    )
    authorization = require_stage_a_go(stage_a_path)
    paths = StageBDataPaths.load(
        contract_path, upstream["stage_b_data_contract_sha256"]
    )
    data = load_stage_b_data(paths, authorization, verify_cache_files=False)
    splits = make_splits(data.folds, args.fold, data.train_only_ids)

    profile_config = config["profiles"]
    profiles, condition_spec, profile_provenance = load_training_profiles(
        _repo_path(profile_config["training_manifest_path"]),
        str(profile_config["training_manifest_sha256"]),
        expected_patient_ids=data.eligibility.eligible_ids,
    )
    transform, transformed_ftv = fit_grounding_transform(
        data.ftv,
        splits.train_primary,
        args.fold,
        apply_ids=splits.train_all + splits.val,
    )
    train_dataset = ProfiledStageBDataset(
        StageBDataset(splits.train_all, data.c1b_cache, transformed_ftv),
        profiles,
        condition_spec,
    )
    val_dataset = ProfiledStageBDataset(
        StageBDataset(splits.val, data.c1b_cache, transformed_ftv),
        profiles,
        condition_spec,
    )

    seed_everything(effective_seed)
    model, initialization = paired_initialization(
        args.arm, condition_spec.dim, effective_seed
    )
    objective = _objective(args.arm, config)
    hyperparameters = _hyperparameters(config)
    device = __import__("torch").device(args.device)
    if device.type == "cuda" and not __import__("torch").cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    provenance = {
        **data.provenance,
        **profile_provenance,
        "PCR_LABEL_ACCESS": "FORBIDDEN",
        "representation_config_sha256": file_sha256(CONFIG_PATH),
        "preregistration_lock_sha256": lock_sha256,
        "stage_a_sentinel_sha256": file_sha256(stage_a_path),
        "stage_b_data_contract_sha256": file_sha256(contract_path),
        "ftv_transform": transform.to_dict(),
        "condition_spec": condition_spec.to_dict(),
        "condition_vocabulary_frozen_from_pcr_free_eligible_manifest": True,
        "paired_initialization": initialization,
        "train_primary_count": len(splits.train_primary),
        "train_only_count": len(splits.train_only),
        "train_all_count": len(splits.train_all),
        "validation_count": len(splits.val),
        "test_count_not_loaded": len(splits.test),
        "train_primary_order_sha256": canonical_sha256(splits.train_primary),
        "train_all_order_sha256": canonical_sha256(splits.train_all),
        "validation_order_sha256": canonical_sha256(splits.val),
        "model_input_fields": ["image", "condition"],
        "auxiliary_training_fields": ["ftv_target", "ftv_mask", "clinical_target"],
        "outcome_fields": [],
        "pcr_parsed": False,
        "pcr_used_for_selection": False,
    }
    if args.preflight_only:
        hyperparameters.validate()
        print(
            json.dumps(
                {
                    "status": "PREFLIGHT_PASS",
                    "arm": args.arm,
                    "seed_base": args.seed_base,
                    "fold": args.fold,
                    "effective_seed": effective_seed,
                    "condition_dim": condition_spec.dim,
                    "train_all_count": len(splits.train_all),
                    "validation_count": len(splits.val),
                    "test_count_not_loaded": len(splits.test),
                    "paired_initialization_sha256": initialization[
                        "shared_initialization_sha256"
                    ],
                    "PCR_LABEL_ACCESS": "FORBIDDEN",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    selection = train_epochs(
        model=model,
        objective=objective,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        output_dir=output_dir,
        hyperparameters=hyperparameters,
        seed_base=args.seed_base,
        fold=args.fold,
        effective_seed=effective_seed,
        provenance=provenance,
        preregistration=preregistration,
    )
    transform.save(output_dir / "ftv_transform.json")
    (output_dir / "ftv_transform.json").chmod(0o600)
    _atomic_private_json(output_dir / "paired_initialization.json", initialization)
    print(json.dumps(selection, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export frozen Goal-F z_R/z_P states before any pCR evaluation is opened."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping

import numpy as np


sys.dont_write_bytecode = True
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
SRC_ROOT = EXPERIMENT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from crps.contracts import (  # noqa: E402
    PRIMARY_ARMS,
    PCR_LABEL_ACCESS,
    canonical_sha256,
    file_sha256,
    load_json,
    validate_seed_fold,
)
from crps.data import ProfiledStageBDataset, load_training_profiles  # noqa: E402
from crps.model import load_checkpoint  # noqa: E402
from crps.preregistration import LOCK_PATH, verify as verify_preregistration  # noqa: E402
from crps.stageb import (  # noqa: E402
    StageBDataPaths,
    StageBDataset,
    load_stage_b_data,
    make_splits,
    require_stage_a_go,
)
from crps.training import weak_photometric_view  # noqa: E402


CONFIG_PATH = EXPERIMENT_ROOT / "configs" / "representation.json"
CHECKPOINT_ROOT = EXPERIMENT_ROOT / "checkpoints" / "formal_primary"
FEATURE_ROOT = EXPERIMENT_ROOT / "features" / "formal_primary"
HEX_SHA256 = re.compile(r"[0-9a-f]{64}")


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _require_digest(value: str, label: str) -> str:
    digest = str(value).strip().casefold()
    if HEX_SHA256.fullmatch(digest) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return digest


def _expected_paths(seed: int, arm: str, fold: int) -> tuple[Path, Path]:
    checkpoint = (
        CHECKPOINT_ROOT / f"seed_{seed}" / arm / f"fold_{fold}" / "selected.pt"
    ).resolve()
    output = (
        FEATURE_ROOT / f"seed_{seed}" / arm / f"fold_{fold}"
    ).resolve()
    return checkpoint, output


def _validate_paths(
    checkpoint: Path,
    output_dir: Path,
    seed: int,
    arm: str,
    fold: int,
) -> tuple[Path, Path]:
    expected_checkpoint, expected_output = _expected_paths(seed, arm, fold)
    observed_checkpoint = checkpoint.expanduser().resolve()
    observed_output = output_dir.expanduser().resolve()
    if observed_checkpoint != expected_checkpoint:
        raise ValueError(f"formal checkpoint must be exactly {expected_checkpoint}")
    if observed_output != expected_output:
        raise ValueError(f"formal feature output must be exactly {expected_output}")
    if not observed_checkpoint.is_file():
        raise FileNotFoundError(observed_checkpoint)
    if observed_output.exists() and any(observed_output.iterdir()):
        raise FileExistsError(f"refusing to overwrite/mix feature export: {observed_output}")
    return observed_checkpoint, observed_output


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        path.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_private_json(path: Path, payload: Mapping[str, Any]) -> None:
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


def _load_selection(path: Path, seed: int, arm: str, fold: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("selection must be a JSON object")
    expected = {"seed_base": seed, "arm": arm, "fold": fold}
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("checkpoint selection identity mismatch")
    if payload.get("PCR_LABEL_ACCESS") != "FORBIDDEN":
        raise PermissionError("checkpoint selection lacks pCR firewall")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=PRIMARY_ARMS)
    parser.add_argument("--seed-base", required=True, type=int, choices=(2026, 3026))
    parser.add_argument("--fold", required=True, type=int, choices=range(5))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--preregistration-lock-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    if PCR_LABEL_ACCESS != "FORBIDDEN":
        raise PermissionError("representation export pCR firewall is inactive")
    args = parse_args()
    if args.batch_size <= 0 or args.workers < 0:
        raise ValueError("batch size must be positive and workers nonnegative")
    effective_seed = validate_seed_fold(args.seed_base, args.fold)
    checkpoint_path, output_dir = _validate_paths(
        args.checkpoint,
        args.output_dir,
        args.seed_base,
        args.arm,
        args.fold,
    )
    lock_sha256 = _require_digest(
        args.preregistration_lock_sha256, "preregistration lock digest"
    )
    preregistration = verify_preregistration()
    if file_sha256(LOCK_PATH) != lock_sha256:
        raise PermissionError("preregistration lock file SHA-256 mismatch")
    config = load_json(CONFIG_PATH)
    upstream = config["upstream"]
    stage_a_path = _repo_path(upstream["stage_a_sentinel"])
    contract_path = _repo_path(upstream["stage_b_data_contract"])
    if file_sha256(stage_a_path) != upstream["stage_a_sentinel_sha256"]:
        raise PermissionError("Stage-A sentinel SHA-256 mismatch")
    if file_sha256(contract_path) != upstream["stage_b_data_contract_sha256"]:
        raise PermissionError("Stage-B data contract SHA-256 mismatch")
    authorization = require_stage_a_go(stage_a_path)
    paths = StageBDataPaths.load(
        contract_path, upstream["stage_b_data_contract_sha256"]
    )
    data = load_stage_b_data(paths, authorization, verify_cache_files=False)
    splits = make_splits(data.folds, args.fold, data.train_only_ids)
    patient_ids = splits.train_primary + splits.val + splits.test
    split = (
        ("train",) * len(splits.train_primary)
        + ("val",) * len(splits.val)
        + ("test",) * len(splits.test)
    )
    if len(patient_ids) != 808 or len(set(patient_ids)) != 808:
        raise ValueError("formal exported outer-fold population must be 808 unique patients")

    profiles_config = config["profiles"]
    profiles, condition_spec, _ = load_training_profiles(
        _repo_path(profiles_config["training_manifest_path"]),
        str(profiles_config["training_manifest_sha256"]),
        expected_patient_ids=data.eligibility.eligible_ids,
    )
    dataset = ProfiledStageBDataset(
        StageBDataset(patient_ids, data.c1b_cache), profiles, condition_spec
    )
    import torch
    from torch.utils.data import DataLoader

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    selection_path = checkpoint_path.parent / "selection.json"
    selection = _load_selection(
        selection_path, args.seed_base, args.arm, args.fold
    )
    checkpoint_sha256 = file_sha256(checkpoint_path)
    model, checkpoint = load_checkpoint(str(checkpoint_path), device)
    if checkpoint.get("arm") != args.arm:
        raise ValueError("selected checkpoint arm mismatch")
    if checkpoint.get("seed_base") != args.seed_base or checkpoint.get("fold") != args.fold:
        raise ValueError("selected checkpoint seed/fold mismatch")
    if checkpoint.get("effective_seed") != effective_seed:
        raise ValueError("selected checkpoint effective seed mismatch")
    if checkpoint.get("selection") != selection:
        raise ValueError("embedded checkpoint selection differs from selection.json")
    if checkpoint.get("preregistration") != preregistration:
        raise PermissionError("selected checkpoint preregistration does not match current lock")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=bool(args.workers),
        prefetch_factor=1 if args.workers else None,
    )
    collected: dict[str, list[np.ndarray]] = {
        "z_R": [],
        "z_P": [],
        "full": [],
        "z_P_aug": [],
        "z_P_future_pred": [],
        "z_P_future_target": [],
        "z_P_future_context": [],
    }
    observed_ids: list[str] = []
    model.eval()
    augmentation = config["augmentation"]
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if set(batch) != {
                "patient_id",
                "image",
                "ftv_target",
                "ftv_mask",
                "condition",
                "clinical_target",
            }:
                raise PermissionError("feature-export batch schema drifted")
            observed_ids.extend(str(value) for value in batch["patient_id"])
            image = batch["image"].to(device, non_blocking=True)
            condition = batch["condition"].to(device, non_blocking=True)
            augmented = weak_photometric_view(
                image,
                seed=effective_seed * 1_000_003 + batch_index,
                scale_half_width=float(augmentation["scale_half_width"]),
                shift_half_width=float(augmentation["shift_half_width"]),
                noise_std=float(augmentation["gaussian_noise_std"]),
            )
            output = model(image, condition, augmented)
            assert output.augmented_phenotype_state is not None
            values = {
                "z_R": output.response_state,
                "z_P": output.phenotype_state,
                "full": output.full_state,
                "z_P_aug": output.augmented_phenotype_state,
                "z_P_future_pred": output.predicted_phenotype_next,
                "z_P_future_target": output.target_phenotype_online[:, 1:],
                "z_P_future_context": output.phenotype_online[:, :-1],
            }
            for key, value in values.items():
                collected[key].append(value.detach().float().cpu().numpy())
    if tuple(observed_ids) != patient_ids:
        raise RuntimeError("feature export changed patient order")
    arrays = {
        key: np.ascontiguousarray(np.concatenate(parts, axis=0), dtype=np.float32)
        for key, parts in collected.items()
    }
    expected_shapes = {
        "z_R": (808, 4, 96),
        "z_P": (808, 4, 96),
        "full": (808, 4, 192),
        "z_P_aug": (808, 4, 96),
        "z_P_future_pred": (808, 3, 96),
        "z_P_future_target": (808, 3, 96),
        "z_P_future_context": (808, 3, 96),
    }
    for key, shape in expected_shapes.items():
        if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise ValueError(f"invalid exported tensor {key}: {arrays[key].shape}")
    if not np.array_equal(arrays["full"], np.concatenate((arrays["z_R"], arrays["z_P"]), axis=-1)):
        raise AssertionError("full state is not exact z_R/z_P concatenation")
    payload: dict[str, np.ndarray] = {
        "patient_id": np.asarray(patient_ids, dtype="U"),
        "split": np.asarray(split, dtype="U"),
        **arrays,
        "arm": np.asarray(args.arm),
        "seed_base": np.asarray(args.seed_base, dtype=np.int64),
        "fold": np.asarray(args.fold, dtype=np.int64),
    }
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    feature_path = output_dir / "factorized_state.private.npz"
    _atomic_npz(feature_path, payload)
    metadata = {
        "schema_version": 1,
        "experiment": "clinical_residual_phenotype_state",
        "arm": args.arm,
        "seed_base": args.seed_base,
        "fold": args.fold,
        "effective_seed": effective_seed,
        "selected_epoch": int(selection["selected_epoch"]),
        "selection_mode": selection["selection_mode"],
        "selection_experiment_pass": bool(selection["experiment_pass"]),
        "feature_sha256": file_sha256(feature_path),
        "checkpoint_sha256": checkpoint_sha256,
        "selection_sha256": file_sha256(selection_path),
        "preregistration_lock_sha256": lock_sha256,
        "preregistration_payload_sha256": preregistration["lock_sha256"],
        "patient_count": len(patient_ids),
        "patient_order_sha256": canonical_sha256(patient_ids),
        "train_patient_sha256": canonical_sha256(splits.train_primary),
        "validation_patient_sha256": canonical_sha256(splits.val),
        "test_patient_sha256": canonical_sha256(splits.test),
        "state_shapes": {key: list(value.shape) for key, value in arrays.items()},
        "augmentation": dict(augmentation),
        "PCR_LABEL_ACCESS": "FORBIDDEN",
        "pcr_labels_used": False,
        "representation_frozen_before_export": True,
        "export_completed": True,
    }
    metadata_path = output_dir / "factorized_state.private.metadata.json"
    _atomic_private_json(metadata_path, metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

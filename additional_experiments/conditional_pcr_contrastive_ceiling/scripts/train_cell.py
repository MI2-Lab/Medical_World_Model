#!/usr/bin/env python3
"""Train and export one pCR-supervised conditional-ceiling cell."""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = EXPERIMENT_ROOT / "src"
CHECKPOINT_ROOT = EXPERIMENT_ROOT / "checkpoints"
FEATURE_ROOT = EXPERIMENT_ROOT / "features"
METRICS_ROOT = EXPERIMENT_ROOT / "metrics"
for source in (SRC_ROOT,):
    value = str(source.resolve())
    while value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)

SUPERVISED_ARMS = ("B1", "B2", "B3")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=SUPERVISED_ARMS, required=True)
    parser.add_argument("--seed", type=int, choices=(2026, 3026, 4026), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--workers",
        type=int,
        help="DataLoader workers; defaults to the frozen configuration.",
    )
    parser.add_argument(
        "--physical-batch-size",
        type=int,
        help="Explicitly restate the locked logical batch size (must be 4).",
    )
    parser.add_argument(
        "--verify-cache-content",
        action="store_true",
        help="Hash every selected C1B archive before B2/B3 training.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only this cell's ignored private artifacts.",
    )
    return parser.parse_args(argv)


def _resolve_device(value: str) -> Any:
    import torch

    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        index = torch.cuda.current_device() if device.index is None else int(device.index)
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError("requested CUDA device index is unavailable")
        device = torch.device("cuda", index)
    return device


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _cell_paths(seed: int, arm: str, fold: int) -> dict[str, Path]:
    name = str(arm).upper()
    if name not in SUPERVISED_ARMS:
        raise ValueError(f"arm must be one of {SUPERVISED_ARMS}")
    if int(seed) not in (2026, 3026, 4026) or int(fold) not in range(5):
        raise ValueError("unregistered seed/fold")
    return {
        "checkpoint": CHECKPOINT_ROOT
        / f"seed_{int(seed)}"
        / name
        / f"fold_{int(fold)}"
        / "selected.private.pt",
        "selection": CHECKPOINT_ROOT
        / f"seed_{int(seed)}"
        / name
        / f"fold_{int(fold)}"
        / "selection.private.json",
        "feature": FEATURE_ROOT
        / f"seed_{int(seed)}"
        / name
        / f"fold_{int(fold)}"
        / "representation.private.npz",
        "matching_audit": METRICS_ROOT
        / f"matching_audit_cell_seed_{int(seed)}_{name}_fold_{int(fold)}.private.json",
    }


def _require_private_target(path: Path, root: Path) -> Path:
    target = path.resolve()
    authoritative = root.resolve()
    try:
        target.relative_to(authoritative)
    except ValueError as error:
        raise ValueError("private output escaped its experiment artifact root") from error
    if ".private." not in target.name:
        raise ValueError("cell outputs must carry the .private. marker")
    return target


def _prepare_outputs(paths: Mapping[str, Path], *, overwrite: bool) -> None:
    roots = {
        "checkpoint": CHECKPOINT_ROOT,
        "selection": CHECKPOINT_ROOT,
        "feature": FEATURE_ROOT,
        "matching_audit": METRICS_ROOT,
    }
    resolved = {
        name: _require_private_target(path, roots[name]) for name, path in paths.items()
    }
    existing = [path for path in resolved.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite existing private cell artifacts; pass --overwrite"
        )
    if overwrite:
        for path in existing:
            if not path.is_file():
                raise ValueError("private cell output target is not a regular file")
            path.unlink()
    for path in resolved.values():
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.parent.chmod(0o700)


def _atomic_private_json(path: Path, payload: Mapping[str, Any]) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary).replace(target)
        target.chmod(0o600)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _atomic_private_npz(path: Path, **arrays: Any) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".npz", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary).replace(target)
        target.chmod(0o600)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _ordered_sha256(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _training_implementation_hashes(file_sha256: Any) -> dict[str, str]:
    paths = {
        "scripts/train_cell.py": Path(__file__).resolve(),
        "src/conditional_ceiling/data.py": SRC_ROOT / "conditional_ceiling" / "data.py",
        "src/conditional_ceiling/contracts.py": SRC_ROOT / "conditional_ceiling" / "contracts.py",
        "src/conditional_ceiling/training.py": SRC_ROOT / "conditional_ceiling" / "training.py",
        "src/conditional_ceiling/losses.py": SRC_ROOT / "conditional_ceiling" / "losses.py",
        "src/conditional_ceiling/model.py": SRC_ROOT / "conditional_ceiling" / "model.py",
        "src/conditional_ceiling/strata.py": SRC_ROOT / "conditional_ceiling" / "strata.py",
    }
    return {name: file_sha256(path) for name, path in paths.items()}


def _labels(clinical: Any) -> dict[str, int]:
    return dict(
        zip(
            clinical["patient_id"].astype(str),
            clinical["label_pcr"].astype(int),
            strict=True,
        )
    )


def _split_vector(cohort: Any, fold: int) -> np.ndarray:
    current = cohort.folds.loc[cohort.folds["fold"].eq(int(fold))]
    mapping = dict(
        zip(current["patient_id"].astype(str), current["split"].astype(str), strict=True)
    )
    if set(mapping) != set(cohort.patient_ids):
        raise ValueError("fold split does not exactly cover full_808")
    return np.asarray([mapping[value] for value in cohort.patient_ids], dtype="U5")


def _load_b0_states(path: Path, cohort: Any, seed: int, fold: int) -> np.ndarray:
    from conditional_ceiling.contracts import file_sha256

    source = path.resolve(strict=True)
    metadata_path = source.with_name("response_state.private.metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("arm") != "LOCAL3"
        or int(metadata.get("seed_base", -1)) != int(seed)
        or int(metadata.get("fold", -1)) != int(fold)
        or metadata.get("feature_sha256") != file_sha256(source)
        or metadata.get("test_labels_used") is not False
    ):
        raise ValueError("confirmed B0 feature metadata contract failed")
    with np.load(source, allow_pickle=False) as archive:
        if set(archive.files) != {
            "patient_id",
            "split",
            "response_state",
            "arm",
            "seed_base",
            "fold",
        }:
            raise ValueError("confirmed B0 feature schema drifted")
        patient_ids = archive["patient_id"].astype(str)
        split = archive["split"].astype(str)
        state = archive["response_state"]
        if (
            str(archive["arm"].item()) != "LOCAL3"
            or int(archive["seed_base"].item()) != int(seed)
            or int(archive["fold"].item()) != int(fold)
        ):
            raise ValueError("confirmed B0 feature cell identity drifted")
    if (
        len(patient_ids) != 808
        or len(set(patient_ids)) != 808
        or state.shape != (808, 4, 192)
        or state.dtype != np.float32
        or not np.isfinite(state).all()
    ):
        raise ValueError("confirmed B0 feature tensor contract failed")
    expected_split = _split_vector(cohort, fold)
    index = {patient_id: row for row, patient_id in enumerate(patient_ids)}
    if set(index) != set(cohort.patient_ids):
        raise ValueError("confirmed B0 feature population differs from full_808")
    order = np.asarray([index[patient_id] for patient_id in cohort.patient_ids])
    if not np.array_equal(split[order], expected_split):
        raise ValueError("confirmed B0 feature split labels disagree with frozen fold")
    return np.ascontiguousarray(state[order], dtype=np.float32)


def _build_cache(cohort: Any, paths: Any, *, verify_content: bool) -> Any:
    from conditional_ceiling.data import load_cache_manifest

    cache = load_cache_manifest(
        paths.c1b_cache_manifest,
        paths.c1b_cache_manifest_sha256,
        allowed_patient_ids=cohort.patient_ids,
        verify_content=verify_content,
    )
    if set(cache) != set(cohort.patient_ids) or len(cache) != 808:
        raise ValueError("raw C1B loader must contain exactly the full_808 cohort")
    return cache


def _build_image_loaders(
    cohort: Any,
    cache: Any,
    strata: Any,
    *,
    fold: int,
    effective_seed: int,
    workers: int,
    physical_batch_size: int,
    device: Any,
) -> tuple[Any, Any]:
    import torch
    from conditional_ceiling.data import (
        CeilingImageDataset,
        ConditionalStratumBatchSampler,
        collate_ceiling_batch,
    )

    train_ids = cohort.split_patient_ids(fold, "train")
    validation_ids = cohort.split_patient_ids(fold, "val")
    labels = _labels(cohort.clinical)
    if set(strata.assignments["patient_id"].astype(str)) != set(train_ids):
        raise ValueError("strata assignments differ from the exact outer-train split")
    train_dataset = CeilingImageDataset(
        train_ids,
        cache,
        labels,
        strata.stratum_ids,
        strata.eligible_anchors,
    )
    validation_dataset = CeilingImageDataset(validation_ids, cache, labels)
    sampler = ConditionalStratumBatchSampler(
        train_ids,
        strata.stratum_ids,
        labels,
        seed=effective_seed,
        max_batch_size=physical_batch_size,
    )
    common = {
        "num_workers": workers,
        "collate_fn": collate_ceiling_batch,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=sampler, **common
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=physical_batch_size,
        shuffle=False,
        **common,
    )
    return train_loader, validation_loader


def _export_from_frozen_states(model: Any, state: np.ndarray, device: Any) -> np.ndarray:
    import torch

    rows: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(state), 128):
            batch = torch.from_numpy(state[start : start + 128]).to(device)
            rows.append(model.project_response(batch).float().cpu().numpy())
    return np.ascontiguousarray(np.concatenate(rows), dtype=np.float32)


def _export_from_images(
    model: Any,
    cohort: Any,
    cache: Any,
    device: Any,
    *,
    workers: int,
    batch_size: int,
) -> np.ndarray:
    import torch
    from conditional_ceiling.data import CeilingImageDataset, collate_ceiling_batch

    # Dummy labels are never read by the model and prevent test pCR from entering
    # feature export. Patient order is the frozen full-cohort order.
    dummy_labels = {patient_id: 0 for patient_id in cohort.patient_ids}
    dataset = CeilingImageDataset(cohort.patient_ids, cache, dummy_labels)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_ceiling_batch,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    rows: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            response = model.encode_response(batch["image"].to(device, non_blocking=True))
            rows.append(model.project_response(response).float().cpu().numpy())
    output = np.ascontiguousarray(np.concatenate(rows), dtype=np.float32)
    if output.shape != (808, 4, 64) or not np.isfinite(output).all():
        raise ValueError("exported supervised representation contract failed")
    return output


def _matching_payload(strata: Any, *, seed: int, arm: str, fold: int) -> dict[str, Any]:
    assignments = strata.assignments
    bidirectional = 0
    for _, rows in assignments.groupby("stratum_id", sort=False):
        counts = rows["label_pcr"].value_counts().reindex([0, 1], fill_value=0)
        bidirectional += int(int(counts.loc[0]) >= 2 and int(counts.loc[1]) >= 2)
    return {
        "schema_version": 1,
        "experiment": "conditional_pcr_contrastive_ceiling",
        "seed": int(seed),
        "arm": str(arm),
        "fold": int(fold),
        "matching": strata.audit.as_dict(),
        "bidirectionally_usable_strata": int(bidirectional),
        "contains_patient_identifiers": False,
    }


def run_cell(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from conditional_ceiling.contracts import (
        LOCKED_CONFIG_SHA256,
        file_sha256,
        load_aligned_full_cohort,
        load_config,
        resolve_input_paths,
        validate_seed_fold,
    )
    from conditional_ceiling.model import build_ceiling_model, load_local3_for_cell
    from conditional_ceiling.strata import build_outer_train_strata
    from conditional_ceiling.training import (
        TrainingHyperparameters,
        save_private_checkpoint,
        train_adaptation_cell,
        train_b1_from_frozen_states,
    )

    seed, fold = validate_seed_fold(args.seed, args.fold)
    arm = str(args.arm).upper()
    if arm not in SUPERVISED_ARMS:
        raise ValueError("train_cell supports only B1/B2/B3")
    device = _resolve_device(args.device)
    config = load_config()
    # Pin provenance to the already verified, process-local configuration lock.
    # Re-reading the file after a long GPU run could falsely attribute a cell to
    # a configuration changed while the process was alive.
    config_sha256 = LOCKED_CONFIG_SHA256
    implementation_sha256 = _training_implementation_hashes(file_sha256)
    workers = int(config["training"]["workers"] if args.workers is None else args.workers)
    if workers < 0:
        raise ValueError("workers must be non-negative")
    outputs = _cell_paths(seed, arm, fold)
    _prepare_outputs(outputs, overwrite=bool(args.overwrite))

    paths = resolve_input_paths(config)
    cohort = load_aligned_full_cohort(config, paths, verify_cache_files=False)
    if len(cohort.patient_ids) != 808:
        raise ValueError("supervised ceiling requires exactly full_808")
    train_ids = cohort.split_patient_ids(fold, "train")
    validation_ids = cohort.split_patient_ids(fold, "val")
    test_ids = cohort.split_patient_ids(fold, "test")
    if set(train_ids) & set(validation_ids) or set(train_ids) & set(test_ids):
        raise ValueError("frozen train/validation/test splits overlap")
    strata = build_outer_train_strata(cohort.clinical, cohort.folds, fold)
    effective_seed = int(seed + fold)
    _seed_everything(effective_seed)
    confirmed = load_local3_for_cell(paths, seed, fold, device=device)
    model = build_ceiling_model(arm, confirmed, head_seed=effective_seed).to(device)
    labels = _labels(cohort.clinical)
    temperature = float(config["projection_head"]["temperature"])

    b0_state: np.ndarray | None = None
    cache = None
    if arm == "B1":
        b0_state = _load_b0_states(paths.feature_path(seed, fold), cohort, seed, fold)
        index = {patient_id: row for row, patient_id in enumerate(cohort.patient_ids)}
        train_index = np.asarray([index[value] for value in train_ids])
        validation_index = np.asarray([index[value] for value in validation_ids])
        train_assignment = strata.assignments.set_index("patient_id").loc[list(train_ids)]
        train_config = config["training"]["B1"]
        selection = train_b1_from_frozen_states(
            model,
            torch.from_numpy(b0_state[train_index]).to(device),
            torch.as_tensor([labels[value] for value in train_ids], device=device),
            torch.as_tensor(train_assignment["stratum_id"].to_numpy(), device=device),
            torch.as_tensor(
                train_assignment["eligible_anchor"].to_numpy(),
                dtype=torch.bool,
                device=device,
            ),
            torch.from_numpy(b0_state[validation_index]).to(device),
            torch.as_tensor([labels[value] for value in validation_ids], device=device),
            epochs=int(train_config["epochs"]),
            patience=int(train_config["patience"]),
            learning_rate=float(train_config["learning_rate"]),
            weight_decay=float(train_config["weight_decay"]),
            temperature=temperature,
        )
        representation = _export_from_frozen_states(model, b0_state, device)
    else:
        cache = _build_cache(
            cohort, paths, verify_content=bool(args.verify_cache_content)
        )
        registered_batch = int(config["training"]["physical_patient_batch_max"])
        physical_batch = int(
            registered_batch
            if args.physical_batch_size is None
            else args.physical_batch_size
        )
        if physical_batch != registered_batch:
            raise ValueError(
                "the locked logical patient batch size cannot be overridden; "
                "encoder microbatching is used for memory control"
            )
        train_loader, validation_loader = _build_image_loaders(
            cohort,
            cache,
            strata,
            fold=fold,
            effective_seed=effective_seed,
            workers=workers,
            physical_batch_size=physical_batch,
            device=device,
        )
        train_config = config["training"][arm]
        hyperparameters = TrainingHyperparameters(
            epochs=int(train_config["epochs"]),
            patience=int(train_config["patience"]),
            encoder_learning_rate=float(train_config["encoder_learning_rate"]),
            head_learning_rate=float(train_config["head_learning_rate"]),
            weight_decay=float(train_config["weight_decay"]),
            bce_weight=0.25,
            max_grad_norm=float(config["training"]["max_grad_norm"]),
        )
        selection = train_adaptation_cell(
            model,
            train_loader,
            validation_loader,
            arm=arm,
            hyperparameters=hyperparameters,
            device=device,
            temperature=temperature,
            microbatch_size=1 if arm == "B3" else None,
        )
        representation = _export_from_images(
            model,
            cohort,
            cache,
            device,
            workers=workers,
            batch_size=physical_batch,
        )

    if representation.shape != (808, 4, 64) or representation.dtype != np.float32:
        raise ValueError("supervised representation must be float32 [808,4,64]")
    split = _split_vector(cohort, fold)
    # Persist the feature first so the selected checkpoint and selection ledger
    # can cryptographically bind the exact tensor later admitted to evaluation.
    _atomic_private_npz(
        outputs["feature"],
        patient_id=np.asarray(cohort.patient_ids, dtype="U17"),
        split=split,
        representation=representation,
        arm=np.asarray(arm),
        seed=np.asarray(seed, dtype=np.int64),
        fold=np.asarray(fold, dtype=np.int64),
    )
    feature_sha256 = file_sha256(outputs["feature"])
    provenance = {
        "schema_version": 1,
        "experiment": "conditional_pcr_contrastive_ceiling",
        "scientific_status": config["scientific_status"],
        "arm": arm,
        "seed": seed,
        "fold": fold,
        "effective_seed": effective_seed,
        "confirmed_checkpoint_sha256": confirmed.checkpoint_sha256,
        "confirmed_checkpoint_epoch": confirmed.epoch,
        "config_sha256": config_sha256,
        "training_implementation_sha256": implementation_sha256,
        "feature_sha256": feature_sha256,
        "train_patient_count": len(train_ids),
        "validation_patient_count": len(validation_ids),
        "test_patient_count_not_used_for_training_or_selection": len(test_ids),
        "train_patient_order_sha256": _ordered_sha256(train_ids),
        "validation_patient_order_sha256": _ordered_sha256(validation_ids),
        "test_labels_used_for_training_or_selection": False,
        "external_ispy1_patients_used": 0,
        "model_forward_fields": ["image"] if arm in {"B2", "B3"} else ["frozen_response_state"],
        "matching_fields_are_sampler_metadata_only": ["label_hr", "label_her2", "arm"],
        "logical_patient_batch_size": (
            None if arm == "B1" else int(config["training"]["physical_patient_batch_max"])
        ),
        "anchor_sampling_strategy": (
            None if arm == "B1" else config["training"]["anchors_per_epoch_strategy"]
        ),
        "eligible_anchors_per_epoch": (
            None if arm == "B1" else len(strata.eligible_anchors)
        ),
        "encoder_microbatch_size": 1 if arm == "B3" else None,
        "world_model_claim_allowed": False,
        "architecture_contract": model.architecture_contract(),
    }
    save_private_checkpoint(outputs["checkpoint"], model, selection, provenance)
    selection_payload = {
        **provenance,
        "status": "SELECTED_VALIDATION_ONLY",
        "selection": selection,
        "checkpoint_path": str(outputs["checkpoint"]),
        "feature_path": str(outputs["feature"]),
    }
    _atomic_private_json(outputs["selection"], selection_payload)
    _atomic_private_json(
        outputs["matching_audit"],
        _matching_payload(strata, seed=seed, arm=arm, fold=fold),
    )
    result = {
        "status": "COMPLETE",
        "arm": arm,
        "seed": seed,
        "fold": fold,
        "selected_epoch": int(selection["selected_epoch"]),
        "selected_validation_mean_auroc": float(
            selection["selected_validation_mean_auroc"]
        ),
        "matching": strata.audit.as_dict(),
        "checkpoint_path": str(outputs["checkpoint"]),
        "feature_path": str(outputs["feature"]),
        "selection_path": str(outputs["selection"]),
        "matching_audit_path": str(outputs["matching_audit"]),
    }
    return result


def main(argv: Sequence[str] | None = None) -> None:
    os.umask(0o077)
    result = run_cell(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

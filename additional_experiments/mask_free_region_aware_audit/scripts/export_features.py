#!/usr/bin/env python3
"""Stream one frozen LOCAL encoder cell into mask-free regional means."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from common import (  # noqa: E402
    CONFIG_PATH,
    FEATURE_FILENAME,
    FEATURE_KEYS,
    GEOMETRY_CONTRACT_PATH,
    LOCK_PATH,
    METADATA_FILENAME,
    METADATA_KEYS,
    PATIENT_COUNT,
    VARIANT_DIMENSIONS,
    VARIANT_KEYS,
    VISITS,
    array_sha256,
    atomic_json,
    atomic_npz,
    canonical_sha256,
    cell_key,
    feature_path,
    file_sha256,
    load_config,
    metadata_path,
    ordered_sha256,
    private_directory,
    publish_json_once,
    require_preregistration_lock,
    validate_feature_cell,
)
from regions import (  # noqa: E402
    CHANNELS,
    INPUT_SHAPE_ZYX,
    REGION_MEAN_KEYS,
    RegionWeights,
    SPACING_XYZ_MM,
    build_region_weights,
    extract_region_features,
    fixed_qr_projection,
    geometry_contract,
    projection_sha256,
)


# Loader-facing aliases: analysis and validation import these exact contracts.
FEATURE_KEYS = FEATURE_KEYS
METADATA_KEYS = METADATA_KEYS
FEATURE_FILENAME = FEATURE_FILENAME
METADATA_FILENAME = METADATA_FILENAME


@dataclass(frozen=True)
class ImageOnlyData:
    folds: Any
    train_only_ids: tuple[str, ...]
    c1b_cache: Mapping[str, Any]
    safe_provenance: Mapping[str, Any]
    stage_a_sentinel_sha256: str


def _load_npz_members(path: Path, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Materialize only explicitly named NPZ members.

    This is important for upstream containers that also hold Oracle statistics:
    extraction reads only identity/R0 reference members, never those arrays.
    """

    output: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as archive:
        available = set(archive.files)
        missing = set(names).difference(available)
        if missing:
            raise ValueError(f"NPZ lacks required members {sorted(missing)}: {path}")
        for name in names:
            output[name] = np.asarray(archive[name]).copy()
    return output


def _scalar(value: np.ndarray, name: str) -> Any:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{name} must be scalar")
    return array.item()


def _model_dependencies(config: Mapping[str, Any]) -> None:
    source_repo = Path(str(config["paths"]["source_repo"])).resolve()
    roots = (
        source_repo
        / "additional_experiments"
        / "c1b_overlap_eligibility_ftv_stageb"
        / "src",
        source_repo
        / "additional_experiments"
        / "local_global_response_state_pilot"
        / "src",
    )
    for path in reversed(roots):
        value = str(path)
        while value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)


def _load_image_only_data(config: Mapping[str, Any]) -> ImageOnlyData:
    """Load folds, eligibility, and C1B images without opening any FTV table."""

    _model_dependencies(config)
    from c1b_stage_b.data import (  # type: ignore
        derive_matched_stage_b_population,
        read_cache_manifest,
        read_fold_manifest,
        read_technical_eligibility,
        read_train_only_candidates,
    )
    from c1b_stage_b.gate import require_stage_a_go  # type: ignore
    from c1b_stage_b.inputs import StageBDataPaths  # type: ignore

    authorization = require_stage_a_go(config["paths"]["stage_b_authorization"])
    paths = StageBDataPaths.load(
        config["paths"]["stage_b_data_contract"],
        config["paths"]["stage_b_data_contract_sha256"],
    )
    folds_all = read_fold_manifest(paths.fold_manifest, paths.fold_manifest_sha256)
    eligibility = read_technical_eligibility(
        paths.technical_eligibility_manifest,
        paths.technical_eligibility_manifest_sha256,
    )
    train_only = read_train_only_candidates(
        paths.train_only_candidate_manifest,
        paths.train_only_candidate_manifest_sha256,
    )
    matched = derive_matched_stage_b_population(folds_all, eligibility, train_only)
    c1b_cache = read_cache_manifest(
        paths.c1b_cache_manifest,
        paths.c1b_cache_manifest_sha256,
        expected_input_kind="c1b",
        verify_cache_files=False,
    )
    required = set(matched.matched_patient_ids)
    if missing := sorted(required.difference(c1b_cache)):
        raise FileNotFoundError(f"C1B cache misses frozen eligible patients: {missing[:5]}")
    safe_provenance = {
        "fold_manifest": str(paths.fold_manifest),
        "fold_manifest_sha256": paths.fold_manifest_sha256,
        "technical_eligibility_manifest": str(paths.technical_eligibility_manifest),
        "technical_eligibility_manifest_sha256": paths.technical_eligibility_manifest_sha256,
        "train_only_candidate_manifest": str(paths.train_only_candidate_manifest),
        "train_only_candidate_manifest_sha256": paths.train_only_candidate_manifest_sha256,
        "c1b_cache_manifest": str(paths.c1b_cache_manifest),
        "c1b_cache_manifest_sha256": paths.c1b_cache_manifest_sha256,
        "eligible_patient_count": len(eligibility.eligible_ids),
        "fold_eligible_patient_count": len(set(matched.folds["patient_id"].astype(str))),
        "train_only_patient_count": len(matched.train_only_ids),
        "matched_cohort_patient_count": len(matched.matched_patient_ids),
    }
    if safe_provenance["fold_eligible_patient_count"] != PATIENT_COUNT:
        raise ValueError("image-only loader cohort differs from frozen 808 patients")
    return ImageOnlyData(
        matched.folds,
        matched.train_only_ids,
        c1b_cache,
        safe_provenance,
        authorization.sha256,
    )


def _validate_checkpoint(
    checkpoint: Mapping[str, Any],
    record: Mapping[str, Any],
    image_data: ImageOnlyData,
    config: Mapping[str, Any],
    *,
    arm: str,
    seed: int,
    fold: int,
) -> None:
    if (
        str(checkpoint.get("arm", "")),
        int(checkpoint.get("seed_base", -1)),
        int(checkpoint.get("fold", -1)),
    ) != (arm, seed, fold):
        raise ValueError("checkpoint cell identity drifted")
    if (
        checkpoint.get("selected") is not True
        or checkpoint.get("test_data_used") is not False
        or checkpoint.get("pcr_used") is not False
        or checkpoint.get("architecture") != "LOCAL"
        or checkpoint.get("input_kind") != "c1b"
    ):
        raise ValueError("feature extraction requires a selected test-blind LOCAL checkpoint")
    selection_path = Path(str(checkpoint.get("selection_path", ""))).resolve()
    if selection_path != Path(str(record["selection_path"])).resolve():
        raise ValueError("checkpoint selection path differs from Goal5 lock")
    if (
        file_sha256(selection_path) != record["selection_sha256"]
        or checkpoint.get("selection_sha256") != record["selection_sha256"]
    ):
        raise ValueError("checkpoint selection hash differs from Goal5 lock")
    provenance = checkpoint.get("data_provenance")
    if not isinstance(provenance, Mapping) or checkpoint.get(
        "data_provenance_sha256"
    ) != canonical_sha256(provenance):
        raise ValueError("checkpoint data provenance is absent or inconsistent")
    for name, expected in image_data.safe_provenance.items():
        if provenance.get(name) != expected:
            raise ValueError(f"checkpoint/image-only provenance differs at {name}")
    if (
        provenance.get("data_contract_sha256")
        != config["paths"]["stage_b_data_contract_sha256"]
        or provenance.get("stage_a_sentinel_sha256")
        != image_data.stage_a_sentinel_sha256
    ):
        raise ValueError("checkpoint frozen data authorization drifted")


def _load_local_reference(
    record: Mapping[str, Any], *, arm: str, seed: int, fold: int
) -> dict[str, np.ndarray]:
    reference = record["reference"]
    path = Path(str(reference["path"])).resolve(strict=True)
    metadata = Path(str(reference["metadata_path"])).resolve(strict=True)
    if (
        file_sha256(path) != reference["sha256"]
        or file_sha256(metadata) != reference["metadata_sha256"]
    ):
        raise ValueError("immutable LOCAL response reference drifted")
    output = _load_npz_members(
        path,
        ("patient_id", "split", "response_state", "arm", "seed_base", "fold"),
    )
    if (
        str(_scalar(output["arm"], "arm")),
        int(_scalar(output["seed_base"], "seed_base")),
        int(_scalar(output["fold"], "fold")),
    ) != (arm, seed, fold):
        raise ValueError("LOCAL response reference identity drifted")
    patient_id = output["patient_id"].astype(str)
    split = output["split"].astype(str)
    response = np.asarray(output["response_state"])
    if (
        patient_id.shape != (PATIENT_COUNT,)
        or split.shape != (PATIENT_COUNT,)
        or response.shape != (PATIENT_COUNT, len(VISITS), 192)
        or response.dtype != np.float32
        or not np.isfinite(response).all()
        or ordered_sha256(patient_id) != reference["patient_order_sha256"]
        or ordered_sha256(split) != reference["split_order_sha256"]
    ):
        raise ValueError("LOCAL response reference content drifted")
    return output


def _goal5_feature_record(
    config: Mapping[str, Any], *, arm: str, seed: int, fold: int
) -> tuple[Path, Path, Mapping[str, Any]]:
    completion_path = Path(str(config["paths"]["goal5_feature_completion"])).resolve(
        strict=True
    )
    if file_sha256(completion_path) != config["paths"]["goal5_feature_completion_sha256"]:
        raise ValueError("Goal5 feature-completion marker drifted")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("status") != "COMPLETE" or completion.get("cell_count") != 20:
        raise ValueError("Goal5 feature matrix is not complete")
    matches = [
        item
        for item in completion.get("cells", [])
        if (item.get("seed"), item.get("arm"), item.get("fold")) == (seed, arm, fold)
    ]
    if len(matches) != 1:
        raise ValueError("Goal5 completion marker lacks the exact feature cell")
    path = (
        Path(str(config["paths"]["goal5_root"])).resolve()
        / "features"
        / f"seed_{seed}"
        / arm
        / f"fold_{fold}"
        / "spatial_statistics.private.npz"
    ).resolve(strict=True)
    if file_sha256(path) != matches[0].get("feature_sha256"):
        raise ValueError("Goal5 feature cell differs from completion marker")
    metadata_path = path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("status") != "COMPLETE"
        or metadata.get("cell") != cell_key(seed, arm, fold)
        or metadata.get("feature_sha256") != matches[0]["feature_sha256"]
        or metadata.get("p1_projection_parity", {}).get("allclose") is not True
    ):
        raise ValueError("Goal5 feature metadata drifted")
    return path, metadata_path, metadata


def _load_goal5_r0(
    config: Mapping[str, Any],
    reference: Mapping[str, np.ndarray],
    *,
    arm: str,
    seed: int,
    fold: int,
) -> tuple[np.ndarray, str, str]:
    path, metadata_path, metadata = _goal5_feature_record(
        config, arm=arm, seed=seed, fold=fold
    )
    # Deliberately omit every `oracle_*` member from this allow-list.
    arrays = _load_npz_members(path, ("patient_id", "split", "mean", "arm", "seed_base", "fold"))
    if (
        str(_scalar(arrays["arm"], "arm")),
        int(_scalar(arrays["seed_base"], "seed_base")),
        int(_scalar(arrays["fold"], "fold")),
    ) != (arm, seed, fold):
        raise ValueError("Goal5 feature identity drifted")
    mean = np.asarray(arrays["mean"])
    if (
        not np.array_equal(arrays["patient_id"].astype(str), reference["patient_id"].astype(str))
        or not np.array_equal(arrays["split"].astype(str), reference["split"].astype(str))
        or mean.shape != (PATIENT_COUNT, len(VISITS), CHANNELS)
        or mean.dtype != np.float32
        or not np.isfinite(mean).all()
        or metadata.get("statistic_shapes", {}).get("mean") != list(mean.shape)
    ):
        raise ValueError("Goal5 R0 mean reference drifted")
    return mean, file_sha256(path), file_sha256(metadata_path)


def _load_c1b_local_weight(config: Mapping[str, Any]) -> np.ndarray:
    path = Path(str(config["paths"]["spatial_sidecar"])).resolve(strict=True)
    if file_sha256(path) != config["paths"]["spatial_sidecar_sha256"]:
        raise ValueError("C1B spatial sidecar drifted")
    # Only the fixed mask-free LOCAL member is materialized from this upstream
    # container.  No valid/oracle/mask-derived member is accessed.
    value = _load_npz_members(path, ("c1b_local_weight_final",))[
        "c1b_local_weight_final"
    ]
    if value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError("C1B LOCAL weight dtype/value drifted")
    return value


def export_cell(
    config: Mapping[str, Any],
    lock: Mapping[str, Any],
    *,
    arm: str,
    seed: int,
    fold: int,
    device_name: str,
    batch_size: int,
    workers: int,
    output: Path,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    _model_dependencies(config)
    from c1b_stage_b.data import StageBDataset, make_splits  # type: ignore
    from lg_response_pilot.model import load_checkpoint_for_evaluation  # type: ignore

    key = cell_key(seed, arm, fold)
    record = lock["selected_cells"][key]
    checkpoint_path = Path(str(record["checkpoint_path"])).resolve(strict=True)
    if file_sha256(checkpoint_path) != record["checkpoint_sha256"]:
        raise ValueError("selected checkpoint drifted after preregistration")
    reference = _load_local_reference(record, arm=arm, seed=seed, fold=fold)
    goal5_r0, goal5_feature_sha, goal5_metadata_sha = _load_goal5_r0(
        config, reference, arm=arm, seed=seed, fold=fold
    )
    c1b_local = _load_c1b_local_weight(config)
    image_data = _load_image_only_data(config)
    splits = make_splits(image_data.folds, fold, image_data.train_only_ids)
    patient_ids = tuple(splits.train_primary + splits.val + splits.test)
    split_labels = tuple(
        ("train",) * len(splits.train_primary)
        + ("val",) * len(splits.val)
        + ("test",) * len(splits.test)
    )
    if (
        patient_ids != tuple(reference["patient_id"].astype(str))
        or split_labels != tuple(reference["split"].astype(str))
        or len(patient_ids) != PATIENT_COUNT
    ):
        raise ValueError("live image-only cohort/order differs from Goal5 lock")

    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal regional feature extraction requires CUDA")
    model, checkpoint = load_checkpoint_for_evaluation(checkpoint_path, device)
    model.requires_grad_(False)
    model.eval()
    _validate_checkpoint(
        checkpoint,
        record,
        image_data,
        config,
        arm=arm,
        seed=seed,
        fold=fold,
    )
    dataset = StageBDataset(patient_ids, image_data.c1b_cache, transformed_ftv={})
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=bool(workers),
        prefetch_factor=1 if workers else None,
    )
    projection_numpy = fixed_qr_projection()
    projection = torch.as_tensor(projection_numpy, dtype=torch.float32, device=device)
    feature_parts: dict[str, list[np.ndarray]] = {name: [] for name in VARIANT_KEYS}
    projected_r0_parts: list[np.ndarray] = []
    observed_ids: list[str] = []
    observed_shapes: set[tuple[int, int, int]] = set()
    parameter_versions = tuple(parameter._version for parameter in model.parameters())
    region_weights = None
    region_hashes: dict[str, str] | None = None
    geometry_payload: dict[str, Any] | None = None
    checkpoint_weight_equal = False

    with torch.inference_mode():
        offset = 0
        for batch in loader:
            batch_ids = tuple(str(value) for value in batch["patient_id"])
            if batch_ids != patient_ids[offset : offset + len(batch_ids)]:
                raise AssertionError("feature loader changed patient order")
            image = batch["image"].to(device, non_blocking=True)
            if tuple(int(value) for value in image.shape[-3:]) != INPUT_SHAPE_ZYX:
                raise ValueError("runtime input shape differs from frozen C1B-H")
            flat = image.reshape(len(batch_ids) * len(VISITS), *image.shape[2:])
            spatial = model.encoder(flat)
            if (
                not isinstance(spatial, torch.Tensor)
                or spatial.dtype != torch.float32
                or tuple(spatial.shape[:2]) != (len(batch_ids) * len(VISITS), CHANNELS)
                or not bool(torch.isfinite(spatial).all())
            ):
                raise ValueError("online encoder final map dtype/batch/channel/value drifted")
            runtime_shape = tuple(int(value) for value in spatial.shape[-3:])
            observed_shapes.add(runtime_shape)
            if region_weights is None:
                # Build once on CPU, exactly like the frozen checkpoint buffer,
                # then transfer immutable weights.  This avoids device-specific
                # geometry arithmetic and makes every cell share one contract.
                cpu_region_weights = build_region_weights(
                    runtime_shape,
                    input_shape_zyx=tuple(int(value) for value in image.shape[-3:]),
                    spacing_xyz_mm=SPACING_XYZ_MM,
                    device="cpu",
                    dtype=torch.float32,
                )
                region_weights = RegionWeights(
                    runtime_shape,
                    {
                        name: weight.to(device=device, non_blocking=True)
                        for name, weight in cpu_region_weights.weights.items()
                    },
                )
                checkpoint_weight = model.local_pooling_weight
                if checkpoint_weight is None or tuple(checkpoint_weight.shape) != (
                    1,
                    1,
                    *runtime_shape,
                ):
                    raise ValueError("LOCAL checkpoint buffer/runtime grid drifted")
                generated_r0 = cpu_region_weights["R0"].numpy()[0, 0]
                checkpoint_r0 = checkpoint_weight.detach().cpu().numpy()[0, 0]
                if c1b_local.shape != runtime_shape:
                    raise ValueError("C1B sidecar/runtime grid drifted")
                checkpoint_weight_equal = bool(
                    np.array_equal(generated_r0, checkpoint_r0)
                    and np.array_equal(generated_r0, c1b_local)
                )
                if not checkpoint_weight_equal:
                    raise ValueError("generated/checkpoint/C1B R0 weights differ bitwise")
                region_hashes = {
                    name: array_sha256(value.numpy())
                    for name, value in cpu_region_weights.weights.items()
                }
                geometry_payload = geometry_contract(cpu_region_weights)
                publish_json_once(geometry_payload, GEOMETRY_CONTRACT_PATH)
            if region_weights.feature_shape_zyx != runtime_shape:
                raise ValueError("encoder final-map shape changed between batches")
            batch_features = extract_region_features(
                spatial, region_weights, projection=projection
            )
            for name in VARIANT_KEYS:
                feature_parts[name].append(
                    batch_features[name]
                    .reshape(len(batch_ids), len(VISITS), VARIANT_DIMENSIONS[name])
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )
            projected_r0 = model.response_projection(batch_features["R0"]).reshape(
                len(batch_ids), len(VISITS), 192
            )
            projected_r0_parts.append(projected_r0.detach().float().cpu().numpy())
            observed_ids.extend(batch_ids)
            offset += len(batch_ids)
            if offset == len(batch_ids) or offset % 80 == 0 or offset == PATIENT_COUNT:
                print(
                    json.dumps(
                        {"cell": key, "patients_complete": offset, "total": PATIENT_COUNT}
                    ),
                    flush=True,
                )

    if (
        tuple(observed_ids) != patient_ids
        or len(observed_shapes) != 1
        or region_weights is None
        or region_hashes is None
        or geometry_payload is None
    ):
        raise AssertionError("formal extraction coverage/runtime geometry drifted")
    if tuple(parameter._version for parameter in model.parameters()) != parameter_versions:
        raise RuntimeError("frozen checkpoint parameters mutated during extraction")
    complete = {
        name: np.concatenate(parts, axis=0).astype(np.float32, copy=False)
        for name, parts in feature_parts.items()
    }
    if tuple(complete) != REGION_MEAN_KEYS:
        raise AssertionError("regional variant order drifted")
    if not np.array_equal(complete["R0"], goal5_r0):
        maximum = float(
            np.max(np.abs(complete["R0"].astype(np.float64) - goal5_r0.astype(np.float64)))
        )
        raise ValueError(f"R0 fails bitwise Goal5 mean parity: max_abs={maximum}")
    projected_r0 = np.concatenate(projected_r0_parts, axis=0).astype(np.float32, copy=False)
    response_reference = np.asarray(reference["response_state"], dtype=np.float32)
    difference = np.abs(projected_r0.astype(np.float64) - response_reference.astype(np.float64))
    rtol = float(config["feature_contract"]["parity"]["rtol"])
    atol = float(config["feature_contract"]["parity"]["atol"])
    projected_parity = bool(np.allclose(projected_r0, response_reference, rtol=rtol, atol=atol))
    if not projected_parity:
        raise ValueError(
            "projected R0 fails immutable LOCAL-state parity: "
            f"max_abs={float(difference.max())}"
        )

    arrays = {
        "patient_id": np.asarray(patient_ids),
        "split": np.asarray(split_labels),
        **complete,
        "arm": np.asarray(arm),
        "seed_base": np.asarray(seed, dtype=np.int64),
        "fold": np.asarray(fold, dtype=np.int64),
    }
    if tuple(arrays) != FEATURE_KEYS:
        raise AssertionError("regional feature archive order drifted")
    private_directory(output.parent)
    atomic_npz(output, arrays)
    actual_shape = next(iter(observed_shapes))
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "experiment": "mask_free_region_aware_audit",
        "cell": key,
        "arm": arm,
        "seed_base": seed,
        "fold": fold,
        "checkpoint_sha256": record["checkpoint_sha256"],
        "selection_sha256": record["selection_sha256"],
        "reference_feature_sha256": record["reference"]["sha256"],
        "goal5_feature_sha256": goal5_feature_sha,
        "goal5_feature_metadata_sha256": goal5_metadata_sha,
        "feature_path": str(output.resolve()),
        "feature_sha256": file_sha256(output),
        "patient_count": PATIENT_COUNT,
        "visit_count": len(VISITS),
        "patient_order_sha256": ordered_sha256(patient_ids),
        "split_order_sha256": ordered_sha256(split_labels),
        "actual_encoder_shape": [PATIENT_COUNT, len(VISITS), CHANNELS, *actual_shape],
        "actual_encoder_dtype": "float32",
        "variant_shapes": {
            name: list(value.shape) for name, value in complete.items()
        },
        "variant_dtypes": {name: "float32" for name in complete},
        "region_weight_sha256": region_hashes,
        "geometry_contract_path": str(GEOMETRY_CONTRACT_PATH.resolve()),
        "geometry_contract_sha256": file_sha256(GEOMETRY_CONTRACT_PATH),
        "projection_matrix_float32_sha256": projection_sha256(projection_numpy),
        "checkpoint_c1b_local_weight_bitwise_equal": checkpoint_weight_equal,
        "r0_goal5_mean_parity": {
            "bitwise_equal": True,
            "max_abs_difference": 0.0,
            "reference_representation": "Goal5_raw_LOCAL_mean",
        },
        "projected_r0_local_state_parity": {
            "allclose": projected_parity,
            "rtol": rtol,
            "atol": atol,
            "max_abs_difference": float(difference.max()),
            "mean_abs_difference": float(difference.mean()),
        },
        "goal5_preregistration_lock_sha256": config["paths"]["goal5_lock_sha256"],
        "preregistration_lock_sha256": file_sha256(LOCK_PATH),
        "config_sha256": file_sha256(CONFIG_PATH),
        "stage_a_sentinel_sha256": image_data.stage_a_sentinel_sha256,
        "checkpoint_data_provenance_sha256": checkpoint["data_provenance_sha256"],
        "implementation_sha256": {
            "scripts/export_features.py": file_sha256(Path(__file__)),
            "scripts/regions.py": file_sha256(ROOT / "scripts" / "regions.py"),
            "scripts/common.py": file_sha256(ROOT / "scripts" / "common.py"),
        },
        "encoder_frozen": True,
        "training_performed": False,
        "streamed_raw_spatial_map_not_persisted": True,
        "response_projection_used_only_for_parity": True,
        "projector_called": False,
        "transition_called": False,
        "target_encoder_called": False,
        "ftv_head_called": False,
        "lesion_mask_read": False,
        "tumor_bbox_read": False,
        "clinical_label_table_read": False,
        "ftv_value_table_read": False,
        "phenotype_or_pcr_labels_read": False,
        "future_visit_used_to_define_region": False,
    }
    if set(metadata) != set(METADATA_KEYS):
        raise AssertionError("regional feature metadata schema drifted")
    atomic_json(metadata, output.with_name(METADATA_FILENAME), private=True)
    return metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-base", type=int, choices=(2026, 3026), required=True)
    parser.add_argument("--arm", choices=("LOCAL0", "LOCAL3"), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    os.umask(0o077)
    args = parse_args(argv)
    if args.batch_size <= 0 or args.workers < 0:
        raise ValueError("batch size must be positive and workers nonnegative")
    # Authenticate before creating directories or deleting an incomplete output.
    config = load_config(CONFIG_PATH, verify_extraction_inputs=True)
    lock = require_preregistration_lock(config)
    output = feature_path(args.seed_base, args.arm, args.fold)
    metadata_output = metadata_path(args.seed_base, args.arm, args.fold)
    if output.exists() and metadata_output.exists():
        validate_feature_cell(
            output,
            config,
            lock,
            seed=args.seed_base,
            arm=args.arm,
            fold=args.fold,
        )
        print(json.dumps({"status": "COMPLETE", "cell": cell_key(args.seed_base, args.arm, args.fold)}))
        return
    # Resume only an exact incomplete cell; complete pairs are immutable above.
    if output.exists() != metadata_output.exists():
        output.unlink(missing_ok=True)
        metadata_output.unlink(missing_ok=True)
    result = export_cell(
        config,
        lock,
        arm=args.arm,
        seed=args.seed_base,
        fold=args.fold,
        device_name=args.device,
        batch_size=args.batch_size,
        workers=args.workers,
        output=output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

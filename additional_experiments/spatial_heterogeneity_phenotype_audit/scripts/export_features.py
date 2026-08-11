#!/usr/bin/env python3
"""Stream one frozen LOCAL encoder map into preregistered spatial statistics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from common import (  # noqa: E402
    atomic_json,
    canonical_sha256,
    file_sha256,
    load_config,
    ordered_sha256,
    private_directory,
    require_preregistration_lock,
)
from pooling import (
    local_statistics,
    weighted_mean,
    weighted_population_std,
)  # noqa: E402
from verify_cache_integrity import (  # noqa: E402
    PRIVATE_MANIFEST as CACHE_PRIVATE_MANIFEST,
    PUBLIC_CONTRACT as CACHE_PUBLIC_CONTRACT,
    require_cache_integrity,
)


VISITS = ("T0", "T1", "T2", "T3")
REGIONS = ("CORE", "PERI10", "PERI20", "LOCAL_REST")
STATISTICS = ("mean", "std", "q25", "q50", "q75")
FEATURE_SHAPE_ZYX = (14, 22, 20)
PRIMARY_PATIENT_COUNT = 808
VISIT_SLOT_COUNT = 3232
SOURCE_AUTHORIZED_VISIT_COUNT = 1933
UPSTREAM_CORE_PARITY_VISIT_COUNT = 1500


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    private_directory(path.parent)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.chmod(0o600)
        temporary.replace(path)
        path.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _scalar(array: np.ndarray, name: str) -> Any:
    value = np.asarray(array)
    if value.shape != ():
        raise ValueError(f"{name} must be scalar")
    return value.item()


def _load_reference(
    record: Mapping[str, Any], *, arm: str, seed: int, fold: int
) -> dict[str, np.ndarray]:
    reference = record["reference"]
    path = Path(str(reference["path"])).resolve()
    if file_sha256(path) != reference["sha256"]:
        raise ValueError("immutable LOCAL response reference drifted")
    with np.load(path, allow_pickle=False) as archive:
        required = {"patient_id", "split", "response_state", "arm", "seed_base", "fold"}
        if set(archive.files) != required:
            raise ValueError("LOCAL reference schema drifted")
        output = {name: np.asarray(archive[name]).copy() for name in required}
    identity = (
        str(_scalar(output["arm"], "arm")),
        int(_scalar(output["seed_base"], "seed_base")),
        int(_scalar(output["fold"], "fold")),
    )
    if identity != (arm, seed, fold):
        raise ValueError("LOCAL reference cell identity drifted")
    return output


def _load_oracle(
    path: Path, contract_path: Path, lock: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    expected_contract = {
        "schema_version": 2,
        "status": "COMPLETE",
        "patient_count": PRIMARY_PATIENT_COUNT,
        "visit_count": VISIT_SLOT_COUNT,
        "visit_slot_count": VISIT_SLOT_COUNT,
        "source_mask_path_inventory_count": VISIT_SLOT_COUNT,
        "source_authorized_visit_count": SOURCE_AUTHORIZED_VISIT_COUNT,
        "source_hash_verified_visit_count": SOURCE_AUTHORIZED_VISIT_COUNT,
        "core_upstream_parity_checked_visit_count": UPSTREAM_CORE_PARITY_VISIT_COUNT,
        "source_authorized_by_visit": {"T0": 808, "T1": 375, "T2": 375, "T3": 375},
        "region_validity_policy": "valid iff source-authorized visit has nonempty post-LOCAL mapped support",
    }
    if any(contract.get(name) != value for name, value in expected_contract.items()):
        raise ValueError("oracle contract is absent or incomplete")
    observed_sha256 = file_sha256(path)
    if contract.get("sidecar_sha256") != observed_sha256:
        raise ValueError("oracle sidecar hash differs from its authenticated contract")
    if contract.get("preregistration_lock_sha256") != file_sha256(
        ROOT / "PREREGISTRATION_LOCK.json"
    ):
        raise ValueError("oracle contract belongs to another preregistration lock")
    if contract.get("builder_implementation_sha256") != lock[
        "implementation_sha256"
    ].get("scripts/build_oracle_sidecars.py"):
        raise ValueError("oracle contract belongs to another builder implementation")
    if contract.get("cache_integrity_contract_sha256") != file_sha256(
        CACHE_PUBLIC_CONTRACT
    ) or contract.get("cache_integrity_private_manifest_sha256") != file_sha256(
        CACHE_PRIVATE_MANIFEST
    ):
        raise ValueError("oracle contract cache-proof files drifted")
    cache_private = json.loads(CACHE_PRIVATE_MANIFEST.read_text(encoding="utf-8"))
    cache_records = cache_private.get("records")
    if not isinstance(cache_records, list) or len(cache_records) != 947:
        raise ValueError("cache-integrity private record schema drifted")
    primary_records = [
        record for record in cache_records if record.get("cohort") == "primary"
    ]
    if (
        len(primary_records) != PRIMARY_PATIENT_COUNT
        or contract.get("cache_integrity_record_set_sha256")
        != canonical_sha256(cache_records)
        or contract.get("cache_integrity_primary_record_set_sha256")
        != canonical_sha256(primary_records)
    ):
        raise ValueError("oracle contract cache record-set provenance drifted")
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "patient_id",
            "visits",
            "regions",
            "region_weight",
            "region_valid",
            "input_voxel_count",
            "local_weight",
            "source_authorized",
            "upstream_core_parity_valid",
        }
        if set(archive.files) != required:
            raise ValueError("oracle sidecar schema drifted")
        output = {name: np.asarray(archive[name]).copy() for name in required}
    if (
        tuple(output["visits"].astype(str)) != VISITS
        or tuple(output["regions"].astype(str)) != REGIONS
    ):
        raise ValueError("oracle visit/region order drifted")
    if (
        output["region_weight"].shape
        != (PRIMARY_PATIENT_COUNT, 4, 4, *FEATURE_SHAPE_ZYX)
        or output["region_weight"].dtype != np.float32
        or not np.isfinite(output["region_weight"]).all()
        or np.any(output["region_weight"] < 0)
    ):
        raise ValueError("oracle region weight shape drifted")
    if (
        output["region_valid"].shape != (PRIMARY_PATIENT_COUNT, 4, 4)
        or output["region_valid"].dtype != np.bool_
    ):
        raise ValueError("oracle validity shape/dtype drifted")
    if (
        output["patient_id"].shape != (PRIMARY_PATIENT_COUNT,)
        or len(set(output["patient_id"].astype(str))) != PRIMARY_PATIENT_COUNT
    ):
        raise ValueError("oracle patient identity contract drifted")
    source_authorized = output["source_authorized"]
    upstream_parity = output["upstream_core_parity_valid"]
    if (
        source_authorized.shape != (PRIMARY_PATIENT_COUNT, 4)
        or source_authorized.dtype != np.bool_
        or upstream_parity.shape != (PRIMARY_PATIENT_COUNT, 4)
        or upstream_parity.dtype != np.bool_
        or int(source_authorized.sum()) != SOURCE_AUTHORIZED_VISIT_COUNT
        or int(upstream_parity.sum()) != UPSTREAM_CORE_PARITY_VISIT_COUNT
        or np.any(upstream_parity & ~source_authorized)
        or not np.array_equal(source_authorized.sum(axis=0), [808, 375, 375, 375])
    ):
        raise ValueError("oracle visit-local source authority drifted")
    mapped_nonempty = np.any(output["region_weight"] > 0, axis=(-3, -2, -1))
    if not np.array_equal(output["region_valid"], mapped_nonempty):
        raise ValueError("oracle validity is not exactly nonempty mapped support")
    if np.any(output["region_valid"] & ~source_authorized[:, :, None]):
        raise ValueError("unavailable oracle visit is marked region-valid")
    observed_counts = {
        region: int(output["region_valid"][:, :, index].sum())
        for index, region in enumerate(REGIONS)
    }
    if contract.get("region_valid_visits") != observed_counts:
        raise ValueError("oracle contract region-valid counts drifted")
    if contract.get("patient_order_sha256") != ordered_sha256(
        output["patient_id"].astype(str)
    ):
        raise ValueError("oracle patient order differs from its authenticated contract")
    return output


def _model_dependencies(config: Mapping[str, Any]) -> None:
    source_repo = Path(config["paths"]["source_repo"])
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
    for root in reversed(roots):
        value = str(root.resolve())
        while value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)


def _validate_checkpoint(
    checkpoint: Mapping[str, Any],
    record: Mapping[str, Any],
    data: Any,
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
    ):
        raise ValueError(
            "feature extraction requires a selected, test-blind checkpoint"
        )
    selection_path = Path(str(checkpoint.get("selection_path", ""))).resolve()
    if selection_path != Path(str(record["selection_path"])).resolve():
        raise ValueError("checkpoint selection path differs from preregistration")
    if (
        file_sha256(selection_path) != record["selection_sha256"]
        or checkpoint.get("selection_sha256") != record["selection_sha256"]
    ):
        raise ValueError("checkpoint selection hash differs from preregistration")
    provenance = checkpoint.get("data_provenance")
    if not isinstance(provenance, Mapping) or checkpoint.get(
        "data_provenance_sha256"
    ) != canonical_sha256(provenance):
        raise ValueError("checkpoint data provenance is absent or inconsistent")
    for key, value in data.provenance.items():
        if provenance.get(key) != value:
            raise ValueError(f"checkpoint/current data provenance differ at {key}")


def _representative_payload(
    spatial: Any,
    patient_ids: tuple[str, ...],
    oracle: Mapping[str, np.ndarray],
    oracle_lookup: Mapping[str, int],
    local_weight: np.ndarray,
) -> dict[str, np.ndarray] | None:
    formal_in_batch = [
        index
        for index, patient_id in enumerate(patient_ids)
        if patient_id in oracle_lookup
    ]
    if not formal_in_batch:
        return None
    # Caller invokes this only when the globally selected representative occurs.
    patient_index = formal_in_batch[0]
    oracle_index = oracle_lookup[patient_ids[patient_index]]
    activation = spatial.reshape(len(patient_ids), 4, 128, *FEATURE_SHAPE_ZYX)[
        patient_index, 0
    ]
    return {
        "activation_mean_abs": activation.abs()
        .mean(dim=0)
        .detach()
        .float()
        .cpu()
        .numpy(),
        "activation_channel_std": activation.std(dim=0, unbiased=False)
        .detach()
        .float()
        .cpu()
        .numpy(),
        "local_weight": np.asarray(local_weight, dtype=np.float32),
        "region_weight": np.asarray(
            oracle["region_weight"][oracle_index, 0], dtype=np.float32
        ),
        "regions": np.asarray(REGIONS),
        "selection_rule": np.asarray(
            "median_total_core_voxel_count_all_four_core_valid_patient_seed2026_LOCAL3_fold0_T0"
        ),
    }


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
    oracle_path: Path,
    oracle_contract_path: Path,
    cache_integrity: Mapping[str, Any],
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    _model_dependencies(config)
    from c1b_stage_b.data import StageBDataset, make_splits  # type: ignore
    from c1b_stage_b.gate import require_stage_a_go  # type: ignore
    from c1b_stage_b.inputs import StageBDataPaths, load_stage_b_data  # type: ignore
    from lg_response_pilot.model import load_checkpoint_for_evaluation  # type: ignore

    private_directory(ROOT / "features")
    private_directory(output.parent)
    key = f"seed_{seed}/{arm}/fold_{fold}"
    if key not in lock["selected_cells"]:
        raise ValueError(f"cell is outside preregistration: {key}")
    record = lock["selected_cells"][key]
    checkpoint_path = Path(str(record["checkpoint_path"])).resolve()
    if file_sha256(checkpoint_path) != record["checkpoint_sha256"]:
        raise ValueError("selected checkpoint drifted after preregistration")
    reference = _load_reference(record, arm=arm, seed=seed, fold=fold)
    patient_ids_reference = tuple(reference["patient_id"].astype(str))
    split_reference = tuple(reference["split"].astype(str))
    response_reference = np.asarray(reference["response_state"], dtype=np.float32)
    oracle = _load_oracle(oracle_path, oracle_contract_path, lock)
    oracle_ids = tuple(oracle["patient_id"].astype(str))
    oracle_lookup = {value: index for index, value in enumerate(oracle_ids)}
    if set(oracle_ids) != set(patient_ids_reference):
        raise ValueError("oracle cohort is not the exact locked feature cohort")

    source_repo = Path(config["paths"]["source_repo"])
    sentinel = (
        source_repo
        / "additional_experiments"
        / "c1b_overlap_eligibility_ftv_stageb"
        / "STAGE_A_GO.json"
    )
    authorization = require_stage_a_go(sentinel)
    data_paths = StageBDataPaths.load(
        config["paths"]["stage_b_data_contract"],
        config["paths"]["stage_b_data_contract_sha256"],
    )
    data = load_stage_b_data(data_paths, authorization, verify_cache_files=False)
    splits = make_splits(data.folds, fold, data.train_only_ids)
    patient_ids = tuple(splits.train_primary + splits.val + splits.test)
    split_labels = tuple(
        ("train",) * len(splits.train_primary)
        + ("val",) * len(splits.val)
        + ("test",) * len(splits.test)
    )
    if patient_ids != patient_ids_reference or split_labels != split_reference:
        raise ValueError(
            "live Stage-B cohort/order differs from immutable LOCAL reference"
        )

    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal feature extraction requires CUDA")
    model, checkpoint = load_checkpoint_for_evaluation(checkpoint_path, device)
    model.requires_grad_(False)
    model.eval()
    _validate_checkpoint(checkpoint, record, data, arm=arm, seed=seed, fold=fold)
    local_weight = model.local_pooling_weight
    if local_weight is None or tuple(local_weight.shape) != (1, 1, *FEATURE_SHAPE_ZYX):
        raise ValueError("LOCAL checkpoint lacks the formal local weight")
    upstream_local = np.asarray(oracle["local_weight"], dtype=np.float32)
    if not np.array_equal(local_weight.detach().cpu().numpy()[0, 0], upstream_local):
        raise ValueError("LOCAL checkpoint and upstream spatial sidecar weights differ")

    dataset = StageBDataset(patient_ids, data.c1b_cache, transformed_ftv={})
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
    statistic_parts: dict[str, list[np.ndarray]] = {name: [] for name in STATISTICS}
    oracle_mean_parts: list[np.ndarray] = []
    oracle_std_parts: list[np.ndarray] = []
    oracle_valid_parts: list[np.ndarray] = []
    projected_parts: list[np.ndarray] = []
    observed_ids: list[str] = []
    observed_shapes: set[tuple[int, int, int]] = set()
    parameter_versions = tuple(parameter._version for parameter in model.parameters())
    representative_id: str | None = None
    representative_arrays: dict[str, np.ndarray] | None = None
    if seed == 2026 and arm == "LOCAL3" and fold == 0:
        representative_candidates = np.asarray(
            oracle["region_valid"][:, :, REGIONS.index("CORE")].all(axis=1)
        )
        if int(representative_candidates.sum()) != 375:
            raise ValueError("representative all-four-visit CORE-valid cohort drifted")
        core_counts = np.asarray(oracle["input_voxel_count"][:, :, 0]).sum(axis=1)
        candidate_indices = np.flatnonzero(representative_candidates)
        median_order = candidate_indices[
            np.argsort(core_counts[candidate_indices], kind="stable")
        ]
        representative_id = oracle_ids[int(median_order[len(median_order) // 2])]

    with torch.inference_mode():
        offset = 0
        for batch in loader:
            batch_ids = tuple(str(value) for value in batch["patient_id"])
            if batch_ids != patient_ids[offset : offset + len(batch_ids)]:
                raise AssertionError("feature loader changed patient order")
            image = batch["image"].to(device, non_blocking=True)
            flat = image.reshape(len(batch_ids) * 4, *image.shape[2:])
            spatial = model.encoder(flat)
            if not isinstance(spatial, torch.Tensor) or spatial.dtype != torch.float32:
                raise ValueError("encoder output must be float32 Tensor")
            if tuple(spatial.shape[:2]) != (len(batch_ids) * 4, 128):
                raise ValueError("encoder output batch/channel shape drifted")
            shape = tuple(int(value) for value in spatial.shape[-3:])
            observed_shapes.add(shape)
            if shape != FEATURE_SHAPE_ZYX or not bool(torch.isfinite(spatial).all()):
                raise ValueError(
                    f"actual formal encoder feature shape/value drifted: {shape}"
                )
            statistics = local_statistics(spatial, local_weight)
            for name, value in statistics.items():
                statistic_parts[name].append(
                    value.reshape(len(batch_ids), 4, 128).detach().float().cpu().numpy()
                )
            projected = model.response_projection(statistics["mean"]).reshape(
                len(batch_ids), 4, 192
            )
            projected_parts.append(projected.detach().float().cpu().numpy())

            batch_region_weight = np.zeros(
                (len(batch_ids), 4, 4, *FEATURE_SHAPE_ZYX), dtype=np.float32
            )
            batch_region_valid = np.zeros((len(batch_ids), 4, 4), dtype=bool)
            for batch_index, patient_id in enumerate(batch_ids):
                oracle_index = oracle_lookup[patient_id]
                batch_region_weight[batch_index] = oracle["region_weight"][oracle_index]
                batch_region_valid[batch_index] = oracle["region_valid"][oracle_index]
            flat_region_mean = torch.zeros(
                (len(batch_ids) * 4, 4, 128), dtype=torch.float32, device=device
            )
            flat_region_std = torch.zeros_like(flat_region_mean)
            flat_validity = batch_region_valid.reshape(len(batch_ids) * 4, 4)
            flat_weights = batch_region_weight.reshape(
                len(batch_ids) * 4, 4, *FEATURE_SHAPE_ZYX
            )
            for region_index in range(4):
                selected = np.flatnonzero(flat_validity[:, region_index])
                if not len(selected):
                    continue
                selected_tensor = torch.as_tensor(
                    selected, dtype=torch.long, device=device
                )
                region_weight = torch.as_tensor(
                    flat_weights[selected, region_index],
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(1)
                selected_spatial = spatial.index_select(0, selected_tensor)
                flat_region_mean[selected_tensor, region_index] = weighted_mean(
                    selected_spatial, region_weight
                )
                flat_region_std[selected_tensor, region_index] = (
                    weighted_population_std(selected_spatial, region_weight)
                )
            batch_region_mean = flat_region_mean.reshape(len(batch_ids), 4, 4, 128)
            batch_region_std = flat_region_std.reshape(len(batch_ids), 4, 4, 128)
            oracle_mean_parts.append(batch_region_mean.cpu().numpy())
            oracle_std_parts.append(batch_region_std.cpu().numpy())
            oracle_valid_parts.append(batch_region_valid)

            if representative_id is not None and representative_id in batch_ids:
                selected = batch_ids.index(representative_id)
                single_ids = (representative_id,)
                single_spatial = spatial[selected * 4 : (selected + 1) * 4]
                representative_arrays = _representative_payload(
                    single_spatial,
                    single_ids,
                    oracle,
                    oracle_lookup,
                    upstream_local,
                )
            observed_ids.extend(batch_ids)
            offset += len(batch_ids)
            if offset == len(batch_ids) or offset % 80 == 0 or offset == 808:
                print(
                    json.dumps(
                        {"cell": key, "patients_complete": offset, "total": 808}
                    ),
                    flush=True,
                )

    if tuple(observed_ids) != patient_ids or observed_shapes != {FEATURE_SHAPE_ZYX}:
        raise AssertionError("formal extraction coverage or runtime shape drifted")
    if (
        tuple(parameter._version for parameter in model.parameters())
        != parameter_versions
    ):
        raise RuntimeError("frozen checkpoint parameters mutated during extraction")

    complete_statistics = {
        name: np.concatenate(parts, axis=0).astype(np.float32, copy=False)
        for name, parts in statistic_parts.items()
    }
    projected = np.concatenate(projected_parts, axis=0).astype(np.float32, copy=False)
    parity_difference = np.abs(
        projected.astype(np.float64) - response_reference.astype(np.float64)
    )
    parity_max_abs = float(parity_difference.max())
    parity_mean_abs = float(parity_difference.mean())
    parity_ok = bool(
        np.allclose(
            projected,
            response_reference,
            rtol=float(config["feature_contract"]["p1_projection_parity"]["rtol"]),
            atol=float(config["feature_contract"]["p1_projection_parity"]["atol"]),
        )
    )
    if not parity_ok:
        raise ValueError(
            f"P1 projected response fails immutable LOCAL parity: max_abs={parity_max_abs}"
        )
    complete_oracle_mean = np.concatenate(oracle_mean_parts, axis=0).astype(
        np.float32, copy=False
    )
    complete_oracle_std = np.concatenate(oracle_std_parts, axis=0).astype(
        np.float32, copy=False
    )
    complete_oracle_valid = np.concatenate(oracle_valid_parts, axis=0).astype(
        bool, copy=False
    )
    expected_oracle_valid = np.stack(
        [
            oracle["region_valid"][oracle_lookup[patient_id]]
            for patient_id in patient_ids
        ]
    )
    if not np.array_equal(complete_oracle_valid, expected_oracle_valid):
        raise ValueError(
            "exported oracle validity differs from the authenticated sidecar"
        )
    if np.any(complete_oracle_mean[~complete_oracle_valid] != 0) or np.any(
        complete_oracle_std[~complete_oracle_valid] != 0
    ):
        raise ValueError("invalid oracle rows must remain explicit zeros")
    representative_cell = seed == 2026 and arm == "LOCAL3" and fold == 0
    if representative_cell and representative_arrays is None:
        raise ValueError("designated representative activation was not observed")
    if not representative_cell and representative_arrays is not None:
        raise AssertionError("representative activation escaped its designated cell")

    arrays = {
        "patient_id": np.asarray(patient_ids),
        "split": np.asarray(split_labels),
        **complete_statistics,
        "oracle_mean": complete_oracle_mean,
        "oracle_std": complete_oracle_std,
        "oracle_valid": complete_oracle_valid,
        "oracle_regions": np.asarray(REGIONS),
        "arm": np.asarray(arm),
        "seed_base": np.asarray(seed, dtype=np.int64),
        "fold": np.asarray(fold, dtype=np.int64),
    }
    _atomic_npz(output, arrays)
    representative_record: dict[str, Any] | None = None
    if representative_arrays is not None:
        representative_output = (
            ROOT / "features" / "representative_activation.private.npz"
        )
        if representative_output.exists():
            raise FileExistsError("representative activation output already exists")
        _atomic_npz(representative_output, representative_arrays)
        representative_record = {
            "path": str(representative_output.resolve()),
            "sha256": file_sha256(representative_output),
            "selection_rule": str(
                np.asarray(representative_arrays["selection_rule"]).item()
            ),
            "contains_patient_identifier": False,
        }

    metadata = {
        "schema_version": 2,
        "status": "COMPLETE",
        "experiment": "spatial_heterogeneity_phenotype_audit",
        "cell": key,
        "arm": arm,
        "seed_base": seed,
        "fold": fold,
        "checkpoint_sha256": record["checkpoint_sha256"],
        "selection_sha256": record["selection_sha256"],
        "reference_feature_sha256": record["reference"]["sha256"],
        "feature_path": str(output),
        "feature_sha256": file_sha256(output),
        "patient_count": 808,
        "patient_order_sha256": ordered_sha256(patient_ids),
        "split_order_sha256": ordered_sha256(split_labels),
        "actual_encoder_shape": [808, 4, 128, *FEATURE_SHAPE_ZYX],
        "actual_encoder_dtype": "float32",
        "streamed_raw_spatial_map_not_persisted": True,
        "statistic_shapes": {
            name: list(value.shape) for name, value in complete_statistics.items()
        },
        "oracle_mean_shape": list(complete_oracle_mean.shape),
        "oracle_sidecar_patient_count": PRIMARY_PATIENT_COUNT,
        "oracle_visit_slot_count": VISIT_SLOT_COUNT,
        "oracle_source_authorized_visits": SOURCE_AUTHORIZED_VISIT_COUNT,
        "oracle_source_authorized_by_visit": {
            "T0": 808,
            "T1": 375,
            "T2": 375,
            "T3": 375,
        },
        "oracle_core_parity_checked_visits": UPSTREAM_CORE_PARITY_VISIT_COUNT,
        "oracle_validity_policy": "valid iff source-authorized visit has nonempty post-LOCAL mapped support",
        "oracle_valid_visits": {
            REGIONS[index]: int(complete_oracle_valid[:, :, index].sum())
            for index in range(4)
        },
        "p1_projection_parity": {
            "allclose": parity_ok,
            "rtol": float(config["feature_contract"]["p1_projection_parity"]["rtol"]),
            "atol": float(config["feature_contract"]["p1_projection_parity"]["atol"]),
            "max_abs_difference": parity_max_abs,
            "mean_abs_difference": parity_mean_abs,
        },
        "oracle_sidecar_sha256": file_sha256(oracle_path),
        "oracle_contract_sha256": file_sha256(oracle_contract_path),
        "cache_integrity_contract_sha256": file_sha256(CACHE_PUBLIC_CONTRACT),
        "cache_integrity_private_manifest_sha256": file_sha256(CACHE_PRIVATE_MANIFEST),
        "cache_integrity_record_set_sha256": canonical_sha256(
            cache_integrity["records"]
        ),
        "cache_integrity_primary_record_set_sha256": canonical_sha256(
            [
                record
                for record in cache_integrity["records"]
                if record["cohort"] == "primary"
            ]
        ),
        "representative_activation": representative_record,
        "data_provenance_sha256": canonical_sha256(data.provenance),
        "stage_a_sentinel_sha256": authorization.sha256,
        "implementation_sha256": {
            "export_features.py": file_sha256(Path(__file__)),
            "pooling.py": file_sha256(ROOT / "scripts" / "pooling.py"),
        },
        "encoder_frozen": True,
        "training_performed": False,
        "response_projection_used_only_for_p1_parity": True,
        "projector_called": False,
        "transition_called": False,
        "target_encoder_called": False,
        "ftv_head_called": False,
        "phenotype_or_pcr_labels_read": False,
    }
    atomic_json(metadata, output.with_suffix(".metadata.json"), private=True)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("LOCAL0", "LOCAL3"), required=True)
    parser.add_argument("--seed-base", type=int, choices=(2026, 3026), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.umask(0o077)
    config = load_config(ROOT / "configs" / "audit.json", verify_inputs=True)
    lock = require_preregistration_lock(config)
    cache_integrity = require_cache_integrity(config, lock)
    output = (
        ROOT
        / "features"
        / f"seed_{args.seed_base}"
        / args.arm
        / f"fold_{args.fold}"
        / "spatial_statistics.private.npz"
    )
    output = output.resolve()
    metadata_path = output.with_suffix(".metadata.json")
    representative_output = ROOT / "features" / "representative_activation.private.npz"
    representative_cell = (
        args.seed_base == 2026 and args.arm == "LOCAL3" and args.fold == 0
    )
    # A killed process can leave exactly one member of the atomic output pair.
    # Remove only that incomplete cell (and its derived representative, if
    # applicable) so the matrix runner can safely resume it. A complete pair is
    # immutable and is never overwritten.
    if representative_cell:
        trio = (output, metadata_path, representative_output)
        if any(path.exists() for path in trio) and not all(
            path.exists() for path in trio
        ):
            for path in trio:
                path.unlink(missing_ok=True)
        if any(path.exists() for path in trio):
            raise FileExistsError(
                f"refusing to overwrite frozen feature/representative trio: {output}"
            )
    elif output.exists() != metadata_path.exists():
        output.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
    if not representative_cell and (output.exists() or metadata_path.exists()):
        raise FileExistsError(f"refusing to overwrite frozen feature cell: {output}")
    metadata = export_cell(
        config,
        lock,
        arm=args.arm,
        seed=args.seed_base,
        fold=args.fold,
        device_name=args.device,
        batch_size=args.batch_size,
        workers=args.workers,
        output=output,
        oracle_path=(ROOT / "manifests" / "oracle_regions.private.npz").resolve(),
        oracle_contract_path=(
            ROOT / "metrics" / "oracle_region_contract.json"
        ).resolve(),
        cache_integrity=cache_integrity,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

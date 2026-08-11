#!/usr/bin/env python3
"""Build visit-local physical core/peritumoral weights for oracle diagnostics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt


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
from verify_cache_integrity import (  # noqa: E402
    PRIVATE_MANIFEST as CACHE_PRIVATE_MANIFEST,
    PUBLIC_CONTRACT as CACHE_PUBLIC_CONTRACT,
    require_cache_integrity,
)


VISITS = ("T0", "T1", "T2", "T3")
REGIONS = ("CORE", "PERI10", "PERI20", "LOCAL_REST")
INPUT_SHAPE_ZYX = (112, 176, 160)
FEATURE_SHAPE_ZYX = (14, 22, 20)
SPACING_ZYX_MM = (2.0, 0.9, 0.9)
PRIMARY_PATIENT_COUNT = 808
VISIT_SLOT_COUNT = PRIMARY_PATIENT_COUNT * len(VISITS)
SOURCE_AUTHORIZED_VISIT_COUNT = 1933
UPSTREAM_CORE_PARITY_VISIT_COUNT = 1500


def fixed_local_voxel_centers(
    shape_zyx: tuple[int, int, int] = INPUT_SHAPE_ZYX,
    spacing_zyx_mm: tuple[float, float, float] = SPACING_ZYX_MM,
    half_width_mm: float = 32.0,
) -> np.ndarray:
    """Voxel-center indicator for the preregistered central physical cube."""

    if len(shape_zyx) != 3 or len(spacing_zyx_mm) != 3:
        raise ValueError("shape and spacing must be ZYX triplets")
    axes = [
        (np.arange(size, dtype=np.float64) - 0.5 * (size - 1)) * spacing
        for size, spacing in zip(shape_zyx, spacing_zyx_mm, strict=True)
    ]
    return (
        (np.abs(axes[0]) <= half_width_mm)[:, None, None]
        & (np.abs(axes[1]) <= half_width_mm)[None, :, None]
        & (np.abs(axes[2]) <= half_width_mm)[None, None, :]
    )


def physical_region_masks(
    lesion: np.ndarray,
    valid_source: np.ndarray,
    *,
    spacing_zyx_mm: tuple[float, float, float] = SPACING_ZYX_MM,
    local_cube: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Return mutually exclusive input-grid regions using millimetre distances."""

    core = np.asarray(lesion, dtype=bool)
    valid = np.asarray(valid_source, dtype=bool)
    if core.shape != valid.shape or core.ndim != 3:
        raise ValueError("lesion and valid-source masks must share a 3-D shape")
    # CORE follows the source-authoritative lesion support exactly, including a
    # lesion that extends into padded source space.  The image-valid mask limits
    # only the non-lesion regions.  Whether a region is usable is decided after
    # RF mapping and exact LOCAL confinement, never from a future visit or an
    # upstream all-visits eligibility flag.
    if core.any():
        distance = distance_transform_edt(~core, sampling=spacing_zyx_mm)
        peri10 = (~core) & valid & (distance > 0.0) & (distance <= 10.0)
        peri20 = (~core) & valid & (distance > 10.0) & (distance <= 20.0)
    else:
        peri10 = np.zeros_like(core)
        peri20 = np.zeros_like(core)
    cube = (
        fixed_local_voxel_centers(core.shape, spacing_zyx_mm)
        if local_cube is None
        else np.asarray(local_cube, dtype=bool)
    )
    if cube.shape != core.shape:
        raise ValueError("local cube shape differs from lesion grid")
    local_rest = cube & valid & ~core & ~peri10 & ~peri20
    output = {
        "CORE": core,
        "PERI10": peri10,
        "PERI20": peri20,
        "LOCAL_REST": local_rest,
    }
    stack = np.stack(list(output.values()), axis=0).astype(np.uint8)
    if np.any(stack.sum(axis=0) > 1):
        raise AssertionError("physical region masks overlap")
    return output


def _strict_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise ValueError(f"not a strict boolean: {value!r}")


def _load_cache_members(path: Path) -> dict[str, np.ndarray]:
    required = {
        "patient_id",
        "visits",
        "grid_shape_zyx",
        "grid_spacing_xyz_mm",
        "grid_center_ras_mm",
        "grid_affine_ras",
        "source_to_anchor_ras",
        "valid_source_mask",
        "support_canonical_sha256",
        "support_available",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"C1B cache lacks oracle members: {missing}")
        return {name: np.asarray(archive[name]).copy() for name in required}


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


def _authenticated_cache_index(
    cache_integrity: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    records = cache_integrity.get("records")
    expected_counts = {
        "patient_count": 947,
        "primary_patient_count": PRIMARY_PATIENT_COUNT,
        "train_only_patient_count": 139,
    }
    for name, expected in expected_counts.items():
        if cache_integrity.get(name) != expected:
            raise ValueError(
                f"authenticated cache proof {name} differs from {expected}"
            )
    if not isinstance(records, list) or len(records) != 947:
        raise ValueError("authenticated cache proof must contain exactly 947 records")
    output: dict[str, Mapping[str, Any]] = {}
    paths: set[Path] = set()
    train_only = 0
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("authenticated cache record is not a mapping")
        if set(record) != {
            "patient_id",
            "path",
            "sha256",
            "size_bytes",
            "mtime_ns",
            "cohort",
        }:
            raise ValueError("authenticated cache record schema drifted")
        patient_id = str(record.get("patient_id", ""))
        path = Path(str(record.get("path", ""))).resolve()
        cohort = str(record.get("cohort", ""))
        if not patient_id or path in paths or cohort not in {"primary", "train_only"}:
            raise ValueError("authenticated cache records repeat an identity or path")
        if cohort == "primary":
            if patient_id in output:
                raise ValueError("authenticated primary cache records repeat a patient")
            output[patient_id] = record
        else:
            train_only += 1
        paths.add(path)
    if len(output) != PRIMARY_PATIENT_COUNT or train_only != 139:
        raise ValueError("authenticated cache proof cohort counts drifted")
    return output


def _validate_visit_authorization(
    support_available: np.ndarray,
    upstream_core_valid: np.ndarray,
    *,
    expected_authorized: int = SOURCE_AUTHORIZED_VISIT_COUNT,
    expected_parity: int = UPSTREAM_CORE_PARITY_VISIT_COUNT,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate visit-local source authority without imposing future completeness."""

    available = np.asarray(support_available)
    parity = np.asarray(upstream_core_valid)
    if available.shape != parity.shape or available.ndim != 2:
        raise ValueError(
            "support availability and upstream parity must be paired [N,V]"
        )
    if available.dtype != np.bool_ or parity.dtype != np.bool_:
        raise ValueError("support availability and upstream parity must be boolean")
    if np.any(parity & ~available):
        raise ValueError("upstream CORE parity visit lacks source-authorized support")
    if int(available.sum()) != int(expected_authorized):
        raise ValueError("source-authorized visit count drifted")
    if int(parity.sum()) != int(expected_parity):
        raise ValueError("upstream CORE parity visit count drifted")
    return available.copy(), parity.copy()


def build(
    config: Mapping[str, Any],
    output: Path,
    lock: Mapping[str, Any],
    cache_integrity: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    paths = config["paths"]
    source_repo = Path(paths["source_repo"])
    dependency_roots = (
        source_repo
        / "additional_experiments"
        / "c1b_overlap_eligibility_ftv_stageb"
        / "src",
        source_repo / "additional_experiments" / "c1b_model_ready_ftv_sanity" / "src",
        source_repo
        / "additional_experiments"
        / "c1b_spatial_pooling_bottleneck_audit"
        / "src",
    )
    for dependency in reversed(dependency_roots):
        sys.path.insert(0, str(dependency))
    from c1b_sanity.geometry import (  # type: ignore
        PhysicalGrid,
        canonical_volume_sha256,
        load_nifti_ras,
        resample_support_nearest,
    )
    from c1b_spatial_audit.pooling import receptive_field_occupancy  # type: ignore

    with np.load(paths["spatial_sidecar"], allow_pickle=False) as archive:
        required = {
            "patient_id",
            "c1b_oracle_weight_final",
            "c1b_oracle_valid",
            "c1b_local_weight_final",
        }
        if not required.issubset(archive.files):
            raise ValueError("upstream spatial sidecar schema drifted")
        all_patient_ids = np.asarray(archive["patient_id"]).astype(str)
        prior_core = np.asarray(archive["c1b_oracle_weight_final"], dtype=np.float32)
        prior_valid = np.asarray(archive["c1b_oracle_valid"], dtype=bool)
        upstream_local = np.asarray(archive["c1b_local_weight_final"], dtype=np.float32)
    if (
        all_patient_ids.shape != (PRIMARY_PATIENT_COUNT,)
        or prior_core.shape != (PRIMARY_PATIENT_COUNT, 4, *FEATURE_SHAPE_ZYX)
        or prior_valid.shape != (PRIMARY_PATIENT_COUNT, 4)
        or prior_valid.dtype != np.bool_
        or upstream_local.shape != FEATURE_SHAPE_ZYX
        or not np.isfinite(prior_core).all()
        or not np.isfinite(upstream_local).all()
        or np.any(upstream_local < 0)
        or np.any(upstream_local > 1)
        or not np.any(upstream_local > 0)
    ):
        raise ValueError("upstream spatial sidecar shape/value contract drifted")
    if len(set(all_patient_ids)) != PRIMARY_PATIENT_COUNT or any(
        not patient_id or patient_id != patient_id.strip()
        for patient_id in all_patient_ids
    ):
        raise ValueError("upstream spatial sidecar patient identities drifted")

    folds = pd.read_csv(
        paths["fold_manifest"],
        usecols=["patient_id", "fold", "split"],
        dtype={"patient_id": str, "fold": int, "split": str},
    )
    if folds.duplicated(["fold", "patient_id"]).any():
        raise ValueError("locked fold manifest repeats a patient within a fold")
    locked_order = folds.loc[folds["fold"] == 0, "patient_id"].to_numpy(dtype=str)
    if not np.array_equal(all_patient_ids, locked_order):
        raise ValueError("upstream spatial sidecar does not use locked primary order")
    for fold in range(5):
        fold_ids = folds.loc[folds["fold"] == fold, "patient_id"].astype(str)
        if len(fold_ids) != PRIMARY_PATIENT_COUNT or set(fold_ids) != set(
            all_patient_ids
        ):
            raise ValueError("locked fold manifest primary cohort drifted")
    authenticated_cache = _authenticated_cache_index(cache_integrity)
    if set(all_patient_ids) != set(authenticated_cache):
        raise ValueError(
            "upstream spatial-sidecar cohort differs from the authenticated 808 caches"
        )
    patient_ids = all_patient_ids

    support = pd.read_csv(paths["support_inventory"])
    required_support = {"patient_id", "visit", "formal_ftv_overlap", "ftv_mask_nifti"}
    if not required_support.issubset(support.columns):
        raise ValueError("support inventory schema drifted")
    support["patient_id"] = support["patient_id"].astype(str)
    support["visit"] = support["visit"].astype(str)
    support["_formal"] = [
        _strict_bool(value) for value in support["formal_ftv_overlap"]
    ]
    support = support.loc[support["patient_id"].isin(patient_ids)].copy()
    if (
        len(support) != VISIT_SLOT_COUNT
        or support.duplicated(["patient_id", "visit"]).any()
    ):
        raise ValueError("support inventory is not the exact 808 x four-visit grid")
    expected_pairs = {
        (patient_id, visit) for patient_id in patient_ids for visit in VISITS
    }
    if set(zip(support["patient_id"], support["visit"], strict=True)) != expected_pairs:
        raise ValueError("support inventory patient/visit grid drifted")
    support_paths = [Path(str(value)).resolve() for value in support["ftv_mask_nifti"]]
    if len(set(support_paths)) != VISIT_SLOT_COUNT or any(
        not path.is_file() for path in support_paths
    ):
        raise ValueError("source-mask path inventory is incomplete or repeats a path")
    support_index = support.set_index(["patient_id", "visit"], verify_integrity=True)
    inventory_parity = np.asarray(
        [
            [
                bool(support_index.loc[(patient_id, visit), "_formal"])
                for visit in VISITS
            ]
            for patient_id in patient_ids
        ],
        dtype=bool,
    )
    if not np.array_equal(inventory_parity, prior_valid):
        raise ValueError(
            "support inventory formal flags differ from upstream CORE parity"
        )

    cache = pd.read_csv(paths["c1b_cache_manifest"])
    required_cache = {
        "patient_id",
        "cache_path",
        "cache_sha256",
        "cache_size_bytes",
        "cache_mtime_ns",
        "input_kind",
    }
    if not required_cache.issubset(cache.columns):
        raise ValueError("C1B cache manifest schema drifted")
    cache["patient_id"] = cache["patient_id"].astype(str)
    if cache["patient_id"].duplicated().any():
        raise ValueError("C1B cache manifest repeats patients")
    cache_index = cache.set_index("patient_id", verify_integrity=True)

    weights = np.zeros(
        (PRIMARY_PATIENT_COUNT, 4, len(REGIONS), *FEATURE_SHAPE_ZYX),
        dtype=np.float32,
    )
    valid = np.zeros((PRIMARY_PATIENT_COUNT, 4, len(REGIONS)), dtype=bool)
    voxel_counts = np.zeros((PRIMARY_PATIENT_COUNT, 4, len(REGIONS)), dtype=np.int64)
    source_authorized = np.zeros((PRIMARY_PATIENT_COUNT, 4), dtype=bool)
    parity_max_abs = 0.0
    parity_checked = 0
    source_hash_verified = 0
    local_cube = fixed_local_voxel_centers()

    for patient_index, patient_id in enumerate(patient_ids):
        row = cache_index.loc[patient_id]
        if str(row["input_kind"]) != "c1b":
            raise ValueError("oracle patient cache is not C1B")
        authenticated = authenticated_cache[patient_id]
        cache_path = Path(str(authenticated["path"])).resolve()
        if (
            Path(str(row["cache_path"])).resolve() != cache_path
            or str(row["cache_sha256"]) != str(authenticated["sha256"])
            or int(row["cache_size_bytes"]) != int(authenticated["size_bytes"])
            or int(row["cache_mtime_ns"]) != int(authenticated["mtime_ns"])
        ):
            raise ValueError(
                "oracle cache row differs from the authenticated exact-cache record"
            )
        stat = cache_path.stat()
        if stat.st_size != int(row["cache_size_bytes"]) or stat.st_mtime_ns != int(
            row["cache_mtime_ns"]
        ):
            raise ValueError("C1B cache stat provenance drifted")
        arrays = _load_cache_members(cache_path)
        if str(np.asarray(arrays["patient_id"]).item()) != patient_id:
            raise ValueError("C1B cache patient identity drifted")
        if tuple(np.asarray(arrays["grid_shape_zyx"]).astype(int)) != INPUT_SHAPE_ZYX:
            raise ValueError("C1B cache grid shape drifted")
        if tuple(np.asarray(arrays["grid_spacing_xyz_mm"]).astype(float)) != (
            0.9,
            0.9,
            2.0,
        ):
            raise ValueError("C1B cache physical spacing drifted")
        if tuple(np.asarray(arrays["visits"]).astype(str)) != VISITS:
            raise ValueError("C1B cache visit order drifted")
        grid = PhysicalGrid(
            shape_zyx=INPUT_SHAPE_ZYX,
            spacing_xyz_mm=(0.9, 0.9, 2.0),
            center_ras_mm=tuple(float(value) for value in arrays["grid_center_ras_mm"]),
        )
        if not np.array_equal(grid.affine_ras, arrays["grid_affine_ras"]):
            raise ValueError("C1B grid affine drifted")
        source_transforms = np.asarray(arrays["source_to_anchor_ras"], dtype=np.float64)
        source_valid = np.asarray(arrays["valid_source_mask"])
        support_available_raw = np.asarray(arrays["support_available"])
        support_hashes = np.asarray(arrays["support_canonical_sha256"])
        if (
            source_valid.shape != (4, 1, *INPUT_SHAPE_ZYX)
            or not np.isin(source_valid, [0, 1]).all()
            or support_available_raw.shape != (4,)
            or not np.isin(support_available_raw, [0, 1]).all()
            or support_hashes.shape != (4,)
        ):
            raise ValueError("C1B visit-local support schema drifted")
        support_available = support_available_raw.astype(bool)
        source_authorized[patient_index] = support_available

        for visit_index, visit in enumerate(VISITS):
            if not bool(support_available[visit_index]):
                if prior_valid[patient_index, visit_index]:
                    raise ValueError("upstream CORE parity lacks source authority")
                # All four paths are inventoried, but unavailable masks are not
                # opened because their content is outside the cache authority.
                continue
            mask_path = Path(
                str(support_index.loc[(patient_id, visit), "ftv_mask_nifti"])
            ).resolve()
            lesion_volume = load_nifti_ras(mask_path)
            observed_support_hash = canonical_volume_sha256(lesion_volume)
            expected_support_hash = str(support_hashes[visit_index])
            if len(expected_support_hash) != 64 or any(
                character not in "0123456789abcdef"
                for character in expected_support_hash
            ):
                raise ValueError("authorized source lesion support hash is invalid")
            if observed_support_hash != expected_support_hash:
                raise ValueError("source lesion support hash drifted")
            source_hash_verified += 1
            sampled = resample_support_nearest(
                lesion_volume,
                grid,
                source_to_anchor_ras=source_transforms[visit_index],
            )
            regions = physical_region_masks(
                sampled,
                source_valid[visit_index, 0],
                local_cube=local_cube,
            )
            nonempty_indices: list[int] = []
            nonempty_masks: list[np.ndarray] = []
            for region_index, region in enumerate(REGIONS):
                region_mask = np.asarray(regions[region], dtype=bool)
                voxel_counts[patient_index, visit_index, region_index] = int(
                    region_mask.sum()
                )
                if region_mask.any():
                    nonempty_indices.append(region_index)
                    nonempty_masks.append(region_mask)
            unrestricted_by_region = np.zeros(
                (len(REGIONS), *FEATURE_SHAPE_ZYX), dtype=np.float32
            )
            if nonempty_masks:
                stacked = np.stack(nonempty_masks, axis=0)[:, None].astype(np.float32)
                mapped_batch = receptive_field_occupancy(
                    torch.from_numpy(stacked),
                    FEATURE_SHAPE_ZYX,
                    stage="final",
                )[:, 0].numpy()
                for batch_index, region_index in enumerate(nonempty_indices):
                    unrestricted_by_region[region_index] = mapped_batch[batch_index]
            for region_index, region in enumerate(REGIONS):
                unrestricted = unrestricted_by_region[region_index]
                if region == "CORE" and prior_valid[patient_index, visit_index]:
                    parity_checked += 1
                    difference = float(
                        np.max(
                            np.abs(
                                unrestricted - prior_core[patient_index, visit_index]
                            )
                        )
                    )
                    parity_max_abs = max(parity_max_abs, difference)
                    if not np.array_equal(
                        unrestricted, prior_core[patient_index, visit_index]
                    ):
                        raise ValueError(
                            "reconstructed core weight is not bitwise equal to upstream oracle"
                        )
                # Receptive fields are much larger than a LOCAL sampling cell.
                # Constrain every diagnostic to the exact same 64-mm feature
                # support used by mask-free P3 before comparing their probes.
                mapped = np.asarray(unrestricted * upstream_local, dtype=np.float32)
                if np.any(mapped[upstream_local == 0] != 0):
                    raise AssertionError(f"{region} escaped the exact LOCAL support")
                if not np.any(mapped > 0):
                    continue
                weights[patient_index, visit_index, region_index] = mapped
                valid[patient_index, visit_index, region_index] = True
        if (
            patient_index == 0
            or (patient_index + 1) % 25 == 0
            or patient_index + 1 == PRIMARY_PATIENT_COUNT
        ):
            print(
                json.dumps(
                    {
                        "oracle_patients_complete": patient_index + 1,
                        "total": PRIMARY_PATIENT_COUNT,
                    }
                ),
                flush=True,
            )

    source_authorized, prior_valid = _validate_visit_authorization(
        source_authorized, prior_valid
    )
    if not np.array_equal(source_authorized.sum(axis=0), [808, 375, 375, 375]):
        raise ValueError("source-authorized visit pattern drifted")
    if source_hash_verified != SOURCE_AUTHORIZED_VISIT_COUNT:
        raise AssertionError("not every authorized source mask was hash verified")
    if parity_checked != UPSTREAM_CORE_PARITY_VISIT_COUNT:
        raise AssertionError("upstream CORE parity coverage drifted")
    mapped_nonempty = np.any(weights > 0, axis=(-3, -2, -1))
    if not np.array_equal(valid, mapped_nonempty):
        raise AssertionError("region validity is not exactly post-LOCAL mapped support")
    if np.any(valid & ~source_authorized[:, :, None]):
        raise AssertionError("an unavailable visit was marked region-valid")

    arrays = {
        "patient_id": patient_ids.astype(str),
        "visits": np.asarray(VISITS),
        "regions": np.asarray(REGIONS),
        "region_weight": weights,
        "region_valid": valid,
        "input_voxel_count": voxel_counts,
        "local_weight": upstream_local,
        "source_authorized": source_authorized,
        "upstream_core_parity_valid": prior_valid,
    }
    _atomic_npz(output, arrays)
    return {
        "schema_version": 2,
        "status": "COMPLETE",
        "patient_count": PRIMARY_PATIENT_COUNT,
        "visit_count": VISIT_SLOT_COUNT,
        "visit_slot_count": VISIT_SLOT_COUNT,
        "source_mask_path_inventory_count": VISIT_SLOT_COUNT,
        "source_authorized_visit_count": SOURCE_AUTHORIZED_VISIT_COUNT,
        "source_unavailable_visit_count": VISIT_SLOT_COUNT
        - SOURCE_AUTHORIZED_VISIT_COUNT,
        "source_hash_verified_visit_count": source_hash_verified,
        "source_authorized_by_visit": {
            visit: int(source_authorized[:, index].sum())
            for index, visit in enumerate(VISITS)
        },
        "regions": list(REGIONS),
        "region_valid_visits": {
            region: int(valid[:, :, index].sum())
            for index, region in enumerate(REGIONS)
        },
        "region_voxel_count_min": {
            region: (
                int(voxel_counts[:, :, index][valid[:, :, index]].min())
                if valid[:, :, index].any()
                else 0
            )
            for index, region in enumerate(REGIONS)
        },
        "region_voxel_count_median": {
            region: (
                float(np.median(voxel_counts[:, :, index][valid[:, :, index]]))
                if valid[:, :, index].any()
                else 0.0
            )
            for index, region in enumerate(REGIONS)
        },
        "core_upstream_bitwise_equal": True,
        "core_upstream_parity_checked_visit_count": parity_checked,
        "core_upstream_max_abs_difference": parity_max_abs,
        "patient_order_sha256": ordered_sha256(patient_ids),
        "sidecar_sha256": file_sha256(output),
        "preregistration_lock_sha256": file_sha256(ROOT / "PREREGISTRATION_LOCK.json"),
        "builder_implementation_sha256": lock["implementation_sha256"][
            "scripts/build_oracle_sidecars.py"
        ],
        "source_hashes": {
            "upstream_spatial_sidecar": config["paths"]["spatial_sidecar_sha256"],
            "support_inventory": config["paths"]["support_inventory_sha256"],
            "c1b_cache_manifest": config["paths"]["c1b_cache_manifest_sha256"],
        },
        "timing_contract": "region weight at visit t uses only lesion mask at visit t",
        "distance_definition": "scipy Euclidean distance transform with sampling ZYX [2.0,0.9,0.9] mm",
        "feature_mapping": "theoretical RF occupancy k47/s8/p23/count_include_pad",
        "feature_support_confinement": "every region RF occupancy multiplied by exact fixed LOCAL fractional sampling-cell weight",
        "region_validity_policy": "valid iff source-authorized visit has nonempty post-LOCAL mapped support",
        "core_upstream_parity_stage": "before LOCAL support confinement",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    parse_args(argv)
    os.umask(0o077)
    output = ROOT / "manifests" / "oracle_regions.private.npz"
    summary_path = ROOT / "metrics" / "oracle_region_contract.json"
    config = load_config(ROOT / "configs" / "audit.json", verify_inputs=True)
    lock = require_preregistration_lock(config)
    cache_integrity = require_cache_integrity(config, lock)
    if output.exists() != summary_path.exists():
        output.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
    if output.exists() or summary_path.exists():
        raise FileExistsError(
            "refusing to overwrite an oracle sidecar or its contract summary"
        )
    summary = build(config, output.resolve(), lock, cache_integrity)
    summary["cache_integrity_contract_sha256"] = file_sha256(CACHE_PUBLIC_CONTRACT)
    summary["cache_integrity_private_manifest_sha256"] = file_sha256(
        CACHE_PRIVATE_MANIFEST
    )
    summary["cache_integrity_record_set_sha256"] = canonical_sha256(
        cache_integrity["records"]
    )
    primary_records = [
        record for record in cache_integrity["records"] if record["cohort"] == "primary"
    ]
    summary["cache_integrity_primary_record_set_sha256"] = canonical_sha256(
        primary_records
    )
    cache_contract = json.loads(CACHE_PUBLIC_CONTRACT.read_text(encoding="utf-8"))
    if (
        cache_contract.get("canonical_record_set_sha256")
        != summary["cache_integrity_record_set_sha256"]
        or cache_contract.get("primary_record_set_sha256")
        != summary["cache_integrity_primary_record_set_sha256"]
    ):
        raise ValueError("cache-integrity public/private record digests differ")
    atomic_json(summary, summary_path)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the preregistered Stage-A spatial phenotype audit.

The runner consumes only the twenty private frozen-statistic assets produced by
``export_features.py``.  It never opens source MRI, trains an encoder, or writes
patient identifiers to a public artifact.  Every preprocessing and model fit is
isolated inside the matching outer fold; only held-out test predictions are
pooled into the public aggregate tables.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
SCRIPTS_ROOT = ROOT / "scripts"
COMPLEMENTARITY_SCRIPTS = (
    REPO_ROOT
    / "additional_experiments"
    / "mri_clinical_complementarity_audit"
    / "scripts"
)
for dependency_root in (SCRIPTS_ROOT, COMPLEMENTARITY_SCRIPTS):
    value = str(dependency_root)
    if value not in sys.path:
        sys.path.insert(0, value)

from common import (  # noqa: E402
    atomic_csv,
    atomic_json,
    canonical_sha256,
    file_sha256,
    load_config,
    ordered_sha256,
    preregistration_chain,
    private_directory,
    require_preregistration_lock,
)
from data_contracts import (  # noqa: E402
    TrainOnlyClinicalEncoder,
    ftv_timing_prefix,
    load_clinical_table,
    load_fold_manifest,
    load_ftv_wide,
)
from modeling import (  # noqa: E402
    fit_binary_logistic,
    fit_ftv_mri_residualizer,
    fit_multiclass_logistic,
    multiclass_metrics,
)
from run_feature_matrix import (  # noqa: E402
    require_representative_contract,
    validate_representative_asset,
)


VISITS = ("T0", "T1", "T2", "T3")
POOLING_VARIANTS = ("P1", "P2", "P3", "P4", "P5")
ORACLE_REGIONS = ("CORE", "PERI10", "PERI20", "LOCAL_REST")
ORACLE_COMPARATORS = (*ORACLE_REGIONS, "CORE_PERI")
ORACLE_VARIANTS = (*ORACLE_COMPARATORS, "FIXED_P3")
PHENOTYPE_TARGETS = ("HR", "HER2", "subtype_4class")
SUBTYPE_CLASSES = tuple(sorted(("HR+/HER2-", "HR-/HER2-", "HR+/HER2+", "HR-/HER2+")))
SUBTYPE_PROBABILITY_COLUMNS = tuple(f"probability_class_{index}" for index in range(4))
SUBTYPE_LABEL_COLUMNS = tuple(f"class_label_{index}" for index in range(4))
FEATURE_KEYS = frozenset(
    {
        "patient_id",
        "split",
        "mean",
        "std",
        "q25",
        "q50",
        "q75",
        "oracle_mean",
        "oracle_std",
        "oracle_valid",
        "oracle_regions",
        "arm",
        "seed_base",
        "fold",
    }
)
METADATA_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "experiment",
        "cell",
        "arm",
        "seed_base",
        "fold",
        "checkpoint_sha256",
        "selection_sha256",
        "reference_feature_sha256",
        "feature_path",
        "feature_sha256",
        "patient_count",
        "patient_order_sha256",
        "split_order_sha256",
        "actual_encoder_shape",
        "actual_encoder_dtype",
        "streamed_raw_spatial_map_not_persisted",
        "statistic_shapes",
        "oracle_mean_shape",
        "oracle_sidecar_patient_count",
        "oracle_visit_slot_count",
        "oracle_source_authorized_visits",
        "oracle_source_authorized_by_visit",
        "oracle_core_parity_checked_visits",
        "oracle_validity_policy",
        "oracle_valid_visits",
        "p1_projection_parity",
        "oracle_sidecar_sha256",
        "oracle_contract_sha256",
        "cache_integrity_contract_sha256",
        "cache_integrity_private_manifest_sha256",
        "cache_integrity_record_set_sha256",
        "cache_integrity_primary_record_set_sha256",
        "representative_activation",
        "data_provenance_sha256",
        "stage_a_sentinel_sha256",
        "implementation_sha256",
        "encoder_frozen",
        "training_performed",
        "response_projection_used_only_for_p1_parity",
        "projector_called",
        "transition_called",
        "target_encoder_called",
        "ftv_head_called",
        "phenotype_or_pcr_labels_read",
    }
)
PREDICTION_COLUMNS = (
    "patient_id",
    "fold",
    "population",
    "seed",
    "arm",
    "analysis",
    "view",
    "target",
    "variant",
    "clinical_contract",
    "y_true",
    "predicted_probability",
    "predicted_label",
    "threshold",
    *SUBTYPE_PROBABILITY_COLUMNS,
    *SUBTYPE_LABEL_COLUMNS,
)
METRIC_COLUMNS = (
    "seed",
    "arm",
    "view",
    "target",
    "variant",
    "population",
    "n",
    "n_positive",
    "n_negative",
    "n_classes",
    "auroc",
    "auprc",
    "balanced_accuracy",
    "brier",
)
HYPERPARAMETER_COLUMNS = (
    "analysis",
    "population",
    "seed",
    "arm",
    "fold",
    "view",
    "target",
    "variant",
    "clinical_contract",
    "selected_c",
    "validation_auroc",
    "validation_auprc",
    "threshold",
    "train_rows",
    "validation_rows",
    "test_rows",
    "feature_dim",
    "class_weight",
    "c_grid",
)
for _schema_name, _schema_columns in (
    ("prediction", PREDICTION_COLUMNS),
    ("metric", METRIC_COLUMNS),
    ("hyperparameter", HYPERPARAMETER_COLUMNS),
):
    if len(_schema_columns) != len(set(_schema_columns)):
        raise RuntimeError(f"{_schema_name} output schema contains duplicate columns")


@dataclass(frozen=True)
class SpatialFeatureAsset:
    """One fold-specific frozen statistic asset with validated provenance."""

    path: Path
    metadata_path: Path
    patient_id: np.ndarray
    split: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    q25: np.ndarray
    q50: np.ndarray
    q75: np.ndarray
    oracle_mean: np.ndarray
    oracle_std: np.ndarray
    oracle_valid: np.ndarray
    arm: str
    seed: int
    fold: int
    metadata: Mapping[str, Any]


def _scalar(array: np.ndarray, name: str) -> Any:
    value = np.asarray(array)
    if value.shape != ():
        raise ValueError(f"{name} must be a scalar")
    return value.item()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable or invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _expected_assignment(folds: pd.DataFrame, fold: int) -> dict[str, str]:
    current = folds.loc[folds["fold"].eq(fold), ["patient_id", "split"]]
    if len(current) != 808 or current["patient_id"].duplicated().any():
        raise ValueError(f"fold {fold} does not contain 808 unique patients")
    return dict(
        zip(
            current["patient_id"].astype(str), current["split"].astype(str), strict=True
        )
    )


def load_spatial_feature_asset(
    path: Path,
    folds: pd.DataFrame,
    config: Mapping[str, Any],
    lock: Mapping[str, Any],
    *,
    seed: int,
    arm: str,
    fold: int,
) -> SpatialFeatureAsset:
    """Load one exported cell and fail closed on any schema/provenance drift."""

    representative_contract = require_representative_contract(config)
    representative_identity = representative_contract["designated_cell"]
    source = path.expanduser().resolve(strict=True)
    expected_tail = (
        f"seed_{seed}",
        arm,
        f"fold_{fold}",
        "spatial_statistics.private.npz",
    )
    if tuple(source.parts[-4:]) != expected_tail:
        raise ValueError(f"feature path is not bound to cell {expected_tail}: {source}")
    metadata_path = source.with_suffix(".metadata.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    try:
        with np.load(source, allow_pickle=False) as archive:
            if set(archive.files) != set(FEATURE_KEYS):
                raise ValueError(
                    f"feature keys drifted for {expected_tail}: "
                    f"missing={sorted(FEATURE_KEYS - set(archive.files))}, "
                    f"extra={sorted(set(archive.files) - FEATURE_KEYS)}"
                )
            arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and "feature keys drifted" in str(error):
            raise
        raise ValueError(f"feature NPZ is unreadable: {source}") from error

    patient_id = arrays["patient_id"]
    split = arrays["split"]
    if patient_id.shape != (808,) or patient_id.dtype.kind != "U":
        raise ValueError("patient_id must be Unicode [808]")
    if (
        split.shape != (808,)
        or split.dtype.kind != "U"
        or set(split.astype(str)) != {"train", "val", "test"}
    ):
        raise ValueError("split must be Unicode [808] with train/val/test")
    patient_values = patient_id.astype(str)
    split_values = split.astype(str)
    if len(set(patient_values)) != 808 or any(
        not value or value != value.strip() for value in patient_values
    ):
        raise ValueError("feature asset has invalid or duplicate patient IDs")
    observed_assignment = dict(zip(patient_values, split_values, strict=True))
    if observed_assignment != _expected_assignment(folds, fold):
        raise ValueError("feature patients/splits differ from the locked outer fold")

    statistic_shape = (808, 4, 128)
    for name in ("mean", "std", "q25", "q50", "q75"):
        value = arrays[name]
        if (
            value.dtype != np.dtype("float32")
            or value.shape != statistic_shape
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"{name} must be finite float32 {statistic_shape}")
    if np.any(arrays["std"] < 0):
        raise ValueError("weighted population SD may not be negative")
    if np.any(arrays["q25"] > arrays["q50"]) or np.any(arrays["q50"] > arrays["q75"]):
        raise ValueError("weighted quantile ordering drifted")
    oracle_shape = (808, 4, 4, 128)
    for name in ("oracle_mean", "oracle_std"):
        value = arrays[name]
        if (
            value.dtype != np.dtype("float32")
            or value.shape != oracle_shape
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"{name} must be finite float32 {oracle_shape}")
    oracle_valid = arrays["oracle_valid"]
    if oracle_valid.dtype != np.dtype(bool) or oracle_valid.shape != (808, 4, 4):
        raise ValueError("oracle_valid must be bool [808,4,4]")
    if np.any(arrays["oracle_std"] < 0):
        raise ValueError("oracle weighted population SD may not be negative")
    if tuple(arrays["oracle_regions"].astype(str)) != ORACLE_REGIONS:
        raise ValueError("oracle region order drifted")
    if np.any(arrays["oracle_mean"][~oracle_valid] != 0) or np.any(
        arrays["oracle_std"][~oracle_valid] != 0
    ):
        raise ValueError("invalid oracle rows must remain explicit zeros")
    identity = (
        str(_scalar(arrays["arm"], "arm")),
        int(_scalar(arrays["seed_base"], "seed_base")),
        int(_scalar(arrays["fold"], "fold")),
    )
    if identity != (arm, seed, fold):
        raise ValueError(f"feature scalar identity drifted: {identity}")
    if arrays["arm"].dtype.kind != "U":
        raise ValueError("arm must be a Unicode scalar")
    if arrays["seed_base"].dtype != np.dtype("int64") or arrays[
        "fold"
    ].dtype != np.dtype("int64"):
        raise ValueError("seed_base and fold must be int64 scalars")

    metadata = _load_json(metadata_path, "feature metadata")
    if set(metadata) != set(METADATA_KEYS):
        raise ValueError(
            "feature metadata keys drifted; "
            f"missing={sorted(METADATA_KEYS - set(metadata))}, "
            f"extra={sorted(set(metadata) - METADATA_KEYS)}"
        )
    key = f"seed_{seed}/{arm}/fold_{fold}"
    record = lock.get("selected_cells", {}).get(key)
    if not isinstance(record, Mapping):
        raise ValueError(f"cell is absent from the preregistration lock: {key}")
    expected_metadata = {
        "schema_version": 2,
        "status": "COMPLETE",
        "experiment": "spatial_heterogeneity_phenotype_audit",
        "cell": key,
        "arm": arm,
        "seed_base": seed,
        "fold": fold,
        "patient_count": 808,
        "actual_encoder_shape": [808, 4, 128, 14, 22, 20],
        "actual_encoder_dtype": "float32",
        "streamed_raw_spatial_map_not_persisted": True,
        "encoder_frozen": True,
        "training_performed": False,
        "response_projection_used_only_for_p1_parity": True,
        "projector_called": False,
        "transition_called": False,
        "target_encoder_called": False,
        "ftv_head_called": False,
        "phenotype_or_pcr_labels_read": False,
    }
    for name, expected in expected_metadata.items():
        if metadata.get(name) != expected:
            raise ValueError(
                f"feature metadata differs at {name}: {metadata.get(name)!r}"
            )
    representative = metadata.get("representative_activation")
    designated_representative = (seed, arm, fold) == (
        representative_identity["seed_base"],
        representative_identity["arm"],
        representative_identity["fold"],
    )
    if designated_representative:
        if not isinstance(representative, Mapping) or set(representative) != {
            "path",
            "sha256",
            "selection_rule",
            "contains_patient_identifier",
        }:
            raise ValueError(
                "designated feature metadata lacks representative provenance"
            )
        if representative.get("contains_patient_identifier") is not False:
            raise ValueError("representative metadata privacy flag drifted")
        representative_path = (
            ROOT / "features" / "representative_activation.private.npz"
        )
        if (
            Path(str(representative.get("path", ""))).resolve()
            != representative_path.resolve()
            or representative.get("sha256") != file_sha256(representative_path)
            or representative.get("selection_rule")
            != representative_contract["selection_rule"]
        ):
            raise ValueError("representative activation source/hash drifted")
        validate_representative_asset(
            representative_path,
            expected_sha256=str(representative["sha256"]),
            representative_contract=representative_contract,
        )
    elif representative is not None:
        raise ValueError(
            "representative provenance appears outside its designated cell"
        )
    if Path(str(metadata["feature_path"])).expanduser().resolve() != source:
        raise ValueError("feature metadata path differs from loaded asset")
    if metadata["feature_sha256"] != file_sha256(source):
        raise ValueError("feature metadata SHA-256 differs from loaded asset")
    if metadata["patient_order_sha256"] != ordered_sha256(patient_values):
        raise ValueError("feature patient-order hash drifted")
    if metadata["split_order_sha256"] != ordered_sha256(split_values):
        raise ValueError("feature split-order hash drifted")
    if metadata["checkpoint_sha256"] != record.get("checkpoint_sha256"):
        raise ValueError("feature checkpoint hash differs from preregistration")
    if metadata["selection_sha256"] != record.get("selection_sha256"):
        raise ValueError("feature selection hash differs from preregistration")
    if metadata["reference_feature_sha256"] != record.get("reference", {}).get(
        "sha256"
    ):
        raise ValueError("feature reference hash differs from preregistration")
    locked_implementation = lock.get("implementation_sha256", {})
    if metadata.get("implementation_sha256") != {
        "export_features.py": locked_implementation.get("scripts/export_features.py"),
        "pooling.py": locked_implementation.get("scripts/pooling.py"),
    }:
        raise ValueError(
            "feature asset was not produced by the locked exporter/pooling implementation"
        )
    if metadata["statistic_shapes"] != {
        name: [808, 4, 128] for name in ("mean", "q25", "q50", "q75", "std")
    }:
        raise ValueError("feature metadata statistic shapes drifted")
    if metadata["oracle_mean_shape"] != [808, 4, 4, 128]:
        raise ValueError("feature metadata oracle shape drifted")
    expected_oracle_contract = {
        "oracle_sidecar_patient_count": 808,
        "oracle_visit_slot_count": 3232,
        "oracle_source_authorized_visits": 1933,
        "oracle_source_authorized_by_visit": {
            "T0": 808,
            "T1": 375,
            "T2": 375,
            "T3": 375,
        },
        "oracle_core_parity_checked_visits": 1500,
        "oracle_validity_policy": "valid iff source-authorized visit has nonempty post-LOCAL mapped support",
    }
    if any(
        metadata.get(name) != value for name, value in expected_oracle_contract.items()
    ):
        raise ValueError("feature metadata oracle source-authority contract drifted")
    observed_valid_counts = {
        region: int(oracle_valid[:, :, index].sum())
        for index, region in enumerate(ORACLE_REGIONS)
    }
    if metadata["oracle_valid_visits"] != observed_valid_counts:
        raise ValueError("feature metadata oracle validity counts drifted")
    for digest_name in (
        "feature_sha256",
        "checkpoint_sha256",
        "selection_sha256",
        "reference_feature_sha256",
        "oracle_sidecar_sha256",
        "oracle_contract_sha256",
        "cache_integrity_contract_sha256",
        "cache_integrity_private_manifest_sha256",
        "cache_integrity_record_set_sha256",
        "cache_integrity_primary_record_set_sha256",
        "data_provenance_sha256",
        "stage_a_sentinel_sha256",
    ):
        digest = metadata.get(digest_name)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"feature metadata {digest_name} is not lowercase SHA-256")
    parity = metadata.get("p1_projection_parity")
    locked_parity = config["feature_contract"]["p1_projection_parity"]
    if not isinstance(parity, Mapping) or parity.get("allclose") is not True:
        raise ValueError("P1 projection parity did not pass")
    if float(parity.get("rtol", -1)) != float(locked_parity["rtol"]) or float(
        parity.get("atol", -1)
    ) != float(locked_parity["atol"]):
        raise ValueError("P1 parity tolerances differ from the locked config")

    return SpatialFeatureAsset(
        path=source,
        metadata_path=metadata_path,
        patient_id=patient_values.copy(),
        split=split_values.copy(),
        mean=arrays["mean"],
        std=arrays["std"],
        q25=arrays["q25"],
        q50=arrays["q50"],
        q75=arrays["q75"],
        oracle_mean=arrays["oracle_mean"],
        oracle_std=arrays["oracle_std"],
        oracle_valid=oracle_valid,
        arm=arm,
        seed=seed,
        fold=fold,
        metadata=metadata,
    )


def load_all_spatial_feature_assets(
    feature_root: Path,
    folds: pd.DataFrame,
    config: Mapping[str, Any],
    lock: Mapping[str, Any],
) -> dict[tuple[int, str, int], SpatialFeatureAsset]:
    """Load all 2 x 2 x 5 assets before any label-dependent probe runs."""

    cells = config["frozen_cells"]
    assets: dict[tuple[int, str, int], SpatialFeatureAsset] = {}
    canonical_oracle_validity: dict[str, np.ndarray] | None = None
    oracle_hash: str | None = None
    oracle_contract_hash: str | None = None
    cache_integrity_hashes: tuple[str, str, str, str] | None = None
    for seed in cells["seed_bases"]:
        for arm in cells["arms"]:
            for fold in cells["folds"]:
                identity = (int(seed), str(arm), int(fold))
                path = (
                    feature_root
                    / f"seed_{identity[0]}"
                    / identity[1]
                    / f"fold_{identity[2]}"
                    / "spatial_statistics.private.npz"
                )
                asset = load_spatial_feature_asset(
                    path,
                    folds,
                    config,
                    lock,
                    seed=identity[0],
                    arm=identity[1],
                    fold=identity[2],
                )
                by_patient = {
                    patient_id: asset.oracle_valid[index].copy()
                    for index, patient_id in enumerate(asset.patient_id)
                }
                if canonical_oracle_validity is None:
                    canonical_oracle_validity = by_patient
                    oracle_hash = str(asset.metadata["oracle_sidecar_sha256"])
                    oracle_contract_hash = str(asset.metadata["oracle_contract_sha256"])
                    cache_integrity_hashes = (
                        str(asset.metadata["cache_integrity_contract_sha256"]),
                        str(asset.metadata["cache_integrity_private_manifest_sha256"]),
                        str(asset.metadata["cache_integrity_record_set_sha256"]),
                        str(
                            asset.metadata["cache_integrity_primary_record_set_sha256"]
                        ),
                    )
                else:
                    if set(by_patient) != set(canonical_oracle_validity):
                        raise ValueError(
                            "oracle patient cohort differs across feature cells"
                        )
                    if any(
                        not np.array_equal(
                            by_patient[patient_id],
                            canonical_oracle_validity[patient_id],
                        )
                        for patient_id in by_patient
                    ):
                        raise ValueError("oracle validity differs across feature cells")
                    if str(asset.metadata["oracle_sidecar_sha256"]) != oracle_hash:
                        raise ValueError(
                            "oracle sidecar hash differs across feature cells"
                        )
                    if (
                        str(asset.metadata["oracle_contract_sha256"])
                        != oracle_contract_hash
                    ):
                        raise ValueError(
                            "oracle contract hash differs across feature cells"
                        )
                    observed_cache_hashes = (
                        str(asset.metadata["cache_integrity_contract_sha256"]),
                        str(asset.metadata["cache_integrity_private_manifest_sha256"]),
                        str(asset.metadata["cache_integrity_record_set_sha256"]),
                        str(
                            asset.metadata["cache_integrity_primary_record_set_sha256"]
                        ),
                    )
                    if observed_cache_hashes != cache_integrity_hashes:
                        raise ValueError(
                            "cache-integrity provenance differs across feature cells"
                        )
                assets[identity] = asset
    if len(assets) != 20:
        raise ValueError("formal feature matrix must contain exactly 20 cells")
    oracle_sidecar = ROOT / "manifests" / "oracle_regions.private.npz"
    if not oracle_sidecar.is_file() or file_sha256(oracle_sidecar) != oracle_hash:
        raise ValueError(
            "exported cells are not bound to the current private oracle sidecar"
        )
    with np.load(oracle_sidecar, allow_pickle=False) as archive:
        required = {"patient_id", "source_authorized", "upstream_core_parity_valid"}
        if not required.issubset(archive.files):
            raise ValueError("private oracle sidecar source-authority schema drifted")
        oracle_patient_ids = np.asarray(archive["patient_id"]).astype(str)
        source_authorized = np.asarray(archive["source_authorized"])
        upstream_parity = np.asarray(archive["upstream_core_parity_valid"])
    if oracle_patient_ids.shape != (808,) or len(set(oracle_patient_ids)) != 808:
        raise ValueError("private oracle sidecar must contain 808 unique patients")
    if (
        source_authorized.shape != (808, 4)
        or source_authorized.dtype != np.bool_
        or upstream_parity.shape != (808, 4)
        or upstream_parity.dtype != np.bool_
        or int(source_authorized.sum()) != 1933
        or int(upstream_parity.sum()) != 1500
        or np.any(upstream_parity & ~source_authorized)
        or not np.array_equal(source_authorized.sum(axis=0), [808, 375, 375, 375])
    ):
        raise ValueError("private oracle visit-local source authority drifted")
    oracle_contract = ROOT / "metrics" / "oracle_region_contract.json"
    if (
        not oracle_contract.is_file()
        or file_sha256(oracle_contract) != oracle_contract_hash
    ):
        raise ValueError("exported cells are not bound to the current oracle contract")
    oracle_payload = _load_json(oracle_contract, "oracle region contract")
    expected_oracle_payload = {
        "schema_version": 2,
        "status": "COMPLETE",
        "patient_count": 808,
        "visit_count": 3232,
        "visit_slot_count": 3232,
        "source_mask_path_inventory_count": 3232,
        "source_authorized_visit_count": 1933,
        "source_hash_verified_visit_count": 1933,
        "source_authorized_by_visit": {"T0": 808, "T1": 375, "T2": 375, "T3": 375},
        "core_upstream_parity_checked_visit_count": 1500,
        "region_validity_policy": "valid iff source-authorized visit has nonempty post-LOCAL mapped support",
    }
    if any(
        oracle_payload.get(name) != value
        for name, value in expected_oracle_payload.items()
    ):
        raise ValueError("oracle contract source-authority counts drifted")
    first_asset = next(iter(assets.values()))
    if oracle_payload.get("region_valid_visits") != first_asset.metadata.get(
        "oracle_valid_visits"
    ):
        raise ValueError("oracle contract/feature valid-visit counts differ")
    if cache_integrity_hashes is None:
        raise AssertionError("cache-integrity provenance was not observed")
    cache_contract = ROOT / "metrics" / "cache_integrity_contract.json"
    cache_manifest = ROOT / "manifests" / "cache_integrity.private.json"
    if (
        not cache_contract.is_file()
        or file_sha256(cache_contract) != cache_integrity_hashes[0]
    ):
        raise ValueError(
            "exported cells are not bound to the current cache-integrity contract"
        )
    if (
        not cache_manifest.is_file()
        or file_sha256(cache_manifest) != cache_integrity_hashes[1]
    ):
        raise ValueError(
            "exported cells are not bound to the current private cache-integrity manifest"
        )
    cache_payload = _load_json(cache_manifest, "private cache-integrity manifest")
    if canonical_sha256(cache_payload.get("records")) != cache_integrity_hashes[2]:
        raise ValueError("cache-integrity record-set digest drifted")
    cache_records = cache_payload.get("records")
    if not isinstance(cache_records, list):
        raise ValueError("cache-integrity records are absent")
    primary_records = [
        record for record in cache_records if record.get("cohort") == "primary"
    ]
    if (
        len(primary_records) != 808
        or canonical_sha256(primary_records) != cache_integrity_hashes[3]
    ):
        raise ValueError("cache-integrity primary record-set digest drifted")
    oracle_cache_hashes = tuple(
        str(oracle_payload.get(name))
        for name in (
            "cache_integrity_contract_sha256",
            "cache_integrity_private_manifest_sha256",
            "cache_integrity_record_set_sha256",
            "cache_integrity_primary_record_set_sha256",
        )
    )
    if oracle_cache_hashes != cache_integrity_hashes:
        raise ValueError(
            "oracle contract and feature assets have different cache-integrity provenance"
        )
    return assets


def timing_end_index(view: str) -> int:
    """Return the latest visit in a preregistered static/prefix view."""

    if view in VISITS:
        return VISITS.index(view)
    pieces = view.split("-")
    if len(pieces) == 2 and pieces[0] == "T0" and pieces[1] in VISITS[1:]:
        return VISITS.index(pieces[1])
    raise ValueError(f"unregistered causal timing: {view!r}")


def validate_analysis_contract(config: Mapping[str, Any]) -> None:
    """Bind matrix construction to the locked visits, dimensions, and block order."""

    require_representative_contract(config)
    analysis = config["analysis"]
    cells = config["frozen_cells"]
    if tuple(cells["arms"]) != ("LOCAL0", "LOCAL3"):
        raise ValueError("formal arm order drifted")
    if tuple(cells["seed_bases"]) != (2026, 3026) or tuple(cells["folds"]) != tuple(
        range(5)
    ):
        raise ValueError("formal seed/fold matrix drifted")
    if tuple(cells["visits"]) != VISITS:
        raise ValueError("formal visit order drifted")
    if tuple(analysis["phenotype_targets"]) != PHENOTYPE_TARGETS:
        raise ValueError("phenotype target contract drifted")
    if tuple(analysis["phenotype_views"]) != VISITS:
        raise ValueError("phenotype view contract drifted")
    if tuple(analysis["pcr_timings"]) != ("T0", "T0-T1", "T0-T2", "T0-T3"):
        raise ValueError("pCR timing contract drifted")
    if tuple(analysis["pcr_populations"]) != ("full_808", "ftv_complete_375"):
        raise ValueError("pCR population contract drifted")
    if tuple(config["poolings"]) != POOLING_VARIANTS:
        raise ValueError("pooling variant contract drifted")
    if (
        tuple(config["oracle"]["regions"]) != ORACLE_REGIONS
        or tuple(config["oracle"]["variants"]) != ORACLE_COMPARATORS
    ):
        raise ValueError("oracle variant contract drifted")
    logistic = config["logistic"]
    if (
        logistic.get("penalty") != "l2"
        or logistic.get("solver") != "liblinear"
        or int(logistic.get("max_iter", -1)) != 10_000
        or logistic.get("phenotype_class_weight") != "balanced"
        or logistic.get("pcr_class_weight") is not None
        or logistic.get("selection_metric") != "validation_auroc"
        or logistic.get("tie_break") != "smaller_C"
    ):
        raise ValueError("logistic probe contract drifted")
    if max(float(value) for value in logistic["strong_c_grid"]) > 0.1:
        raise ValueError("strong-regularization grid may not exceed C=0.1")
    temporal = config["view_feature_contract"]
    expected_views = {
        "T0": ["T0"],
        "T0-T1": ["T0", "T1"],
        "T0-T2": ["T0", "T1", "T2"],
        "T0-T3": ["T0", "T1", "T2", "T3"],
    }
    if any(temporal.get(view) != visits for view, visits in expected_views.items()):
        raise ValueError("causal view-to-visit mapping drifted")
    if (
        temporal.get("prefix_order")
        != "visit_chronological_then_statistic_component_then_channel"
    ):
        raise ValueError("causal prefix ordering drifted")
    expected_dimensions = {
        variant: [
            int(config["poolings"][variant]["dimension"]) * count
            for count in range(1, 5)
        ]
        for variant in POOLING_VARIANTS
    }
    if temporal.get("mask_free_dimensions_by_visit_count") != expected_dimensions:
        raise ValueError("causal mask-free dimension contract drifted")
    if temporal.get("beyond_ftv_block_order") != [
        "clinical_C",
        "causal_log1p_FTV_prefix",
        "causal_MRI_prefix",
    ]:
        raise ValueError("beyond-FTV block ordering drifted")


def causal_prefix(features: np.ndarray, view: str) -> np.ndarray:
    """Flatten visits through ``view`` without exposing a future visit."""

    values = np.asarray(features)
    if values.ndim != 3 or values.shape[1] != len(VISITS) or values.shape[2] == 0:
        raise ValueError("causal-prefix features must have shape [N,4,D]")
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError("causal-prefix features must be finite numeric values")
    end = timing_end_index(view) + 1
    return values[:, :end, :].reshape(values.shape[0], end * values.shape[2]).copy()


def static_visit(features: np.ndarray, visit: str) -> np.ndarray:
    values = np.asarray(features)
    if visit not in VISITS or values.ndim != 3 or values.shape[1] != len(VISITS):
        raise ValueError("static-visit features must be [N,4,D] at T0..T3")
    output = values[:, VISITS.index(visit), :]
    if output.shape[1] == 0 or not np.isfinite(output).all():
        raise ValueError("static-visit features are empty or non-finite")
    return output.copy()


def oracle_complete_case_mask(
    oracle_valid: np.ndarray, view: str, variant: str, *, prefix: bool
) -> np.ndarray:
    """Return validity for exactly the oracle comparator in a fixed-P3 pair.

    Mask-free fixed P3 is valid for every feature patient, so pairwise matching
    adds only the comparator's required region(s), never CORE or another
    unrelated region implicitly.
    """

    valid = np.asarray(oracle_valid)
    if valid.ndim != 3 or valid.shape[1:] != (4, 4) or valid.dtype != np.dtype(bool):
        raise ValueError("oracle_valid must be bool [N,4,4]")
    region_indices = {
        "CORE": (0,),
        "PERI10": (1,),
        "PERI20": (2,),
        "LOCAL_REST": (3,),
        "CORE_PERI": (0, 1, 2),
    }
    if variant not in region_indices:
        raise ValueError(f"unknown oracle comparison variant: {variant}")
    end = timing_end_index(view)
    visits = slice(0, end + 1) if prefix else slice(end, end + 1)
    selected = valid[:, visits, :][:, :, region_indices[variant]]
    return np.all(selected, axis=(1, 2))


def pooling_features(asset: SpatialFeatureAsset) -> dict[str, np.ndarray]:
    """Construct the five exact preregistered mask-free variants."""

    return {
        "P1": asset.mean,
        "P2": asset.std,
        "P3": np.concatenate((asset.mean, asset.std), axis=2),
        "P4": np.concatenate((asset.q25, asset.q50, asset.q75), axis=2),
        "P5": np.concatenate(
            (asset.mean, asset.std, asset.q25, asset.q50, asset.q75), axis=2
        ),
    }


def oracle_features(asset: SpatialFeatureAsset) -> dict[str, np.ndarray]:
    """Construct region mean+SD features and the matched fixed-LOCAL P3."""

    regional = {
        region: np.concatenate(
            (asset.oracle_mean[:, :, index, :], asset.oracle_std[:, :, index, :]),
            axis=2,
        )
        for index, region in enumerate(ORACLE_REGIONS)
    }
    regional["CORE_PERI"] = np.concatenate(
        (regional["CORE"], regional["PERI10"], regional["PERI20"]), axis=2
    )
    regional["FIXED_P3"] = np.concatenate((asset.mean, asset.std), axis=2)
    return regional


def _aligned_clinical(
    clinical: pd.DataFrame, patient_ids: Sequence[str]
) -> pd.DataFrame:
    indexed = clinical.set_index("patient_id", verify_integrity=True)
    requested = [str(value) for value in patient_ids]
    missing = sorted(set(requested) - set(indexed.index))
    if missing:
        raise ValueError(f"clinical table misses feature patients: {missing[:5]}")
    return indexed.loc[requested].reset_index()


def _aligned_ftv(ftv: pd.DataFrame, patient_ids: Sequence[str]) -> pd.DataFrame:
    indexed = ftv.set_index("patient_id", verify_integrity=True)
    requested = [str(value) for value in patient_ids]
    missing = sorted(set(requested) - set(indexed.index))
    if missing:
        raise ValueError(f"FTV table misses selected patients: {missing[:5]}")
    return indexed.loc[requested].reset_index()


def _split_indices(split: np.ndarray) -> dict[str, np.ndarray]:
    labels = np.asarray(split).astype(str)
    output = {name: np.flatnonzero(labels == name) for name in ("train", "val", "test")}
    if any(len(index) == 0 for index in output.values()):
        raise ValueError(
            "every analyzed cohort needs non-empty train/val/test partitions"
        )
    if sum(len(index) for index in output.values()) != len(labels):
        raise ValueError("split contains an unknown label")
    return output


def _clinical_matrix(
    config: Mapping[str, Any],
    clinical: pd.DataFrame,
    indices: Mapping[str, np.ndarray],
    contract: str,
) -> np.ndarray:
    contracts = config["clinical_contracts"]
    if contract not in contracts:
        raise ValueError(f"unknown clinical contract: {contract}")
    encoder = TrainOnlyClinicalEncoder(tuple(contracts[contract]))
    encoder.fit(clinical.iloc[indices["train"]])
    return encoder.transform(clinical)


def _grid(config: Mapping[str, Any], variant: str) -> tuple[float, ...]:
    name = "strong_c_grid" if variant in {"P5", "CORE_PERI"} else "c_grid"
    return tuple(float(value) for value in config["logistic"][name])


def _fit_binary(
    matrix: np.ndarray,
    labels: np.ndarray,
    indices: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    *,
    grid: Iterable[float],
    class_weight: str | Mapping[int, float] | None,
) -> Any:
    logistic = config["logistic"]
    return fit_binary_logistic(
        matrix[indices["train"]],
        labels[indices["train"]],
        matrix[indices["val"]],
        labels[indices["val"]],
        grid,
        class_weight=class_weight,
        solver=str(logistic["solver"]),
        max_iter=int(logistic["max_iter"]),
        random_state=0,
    )


def _base_prediction(
    metadata: Mapping[str, Any], patient_id: str, fold: int
) -> dict[str, Any]:
    return {
        "patient_id": patient_id,
        "fold": int(fold),
        "population": str(metadata.get("population", "")),
        "seed": int(metadata["seed"]),
        "arm": str(metadata["arm"]),
        "analysis": str(metadata["analysis"]),
        "view": str(metadata["view"]),
        "target": str(metadata["target"]),
        "variant": str(metadata["variant"]),
        "clinical_contract": str(metadata.get("clinical_contract", "")),
    }


def _append_binary_fit(
    prediction_rows: list[dict[str, Any]],
    hyperparameter_rows: list[dict[str, Any]],
    *,
    patient_ids: np.ndarray,
    fold: int,
    labels: np.ndarray,
    matrix: np.ndarray,
    indices: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    grid: Sequence[float],
    class_weight: str | Mapping[int, float] | None,
    metadata: Mapping[str, Any],
) -> None:
    fit = _fit_binary(
        matrix,
        labels,
        indices,
        config,
        grid=grid,
        class_weight=class_weight,
    )
    test = indices["test"]
    probability = fit.predict_proba(matrix[test])
    prediction = (probability >= fit.threshold_selection.threshold).astype(np.int64)
    for offset, row_index in enumerate(test):
        row = _base_prediction(metadata, str(patient_ids[row_index]), fold)
        row.update(
            {
                "y_true": int(labels[row_index]),
                "predicted_probability": float(probability[offset]),
                "predicted_label": int(prediction[offset]),
                "threshold": float(fit.threshold_selection.threshold),
            }
        )
        prediction_rows.append(row)
    hyperparameter_rows.append(
        {
            **{name: metadata.get(name, "") for name in HYPERPARAMETER_COLUMNS[:9]},
            "fold": int(fold),
            "selected_c": float(fit.selected_c),
            "validation_auroc": float(fit.validation_auroc),
            "validation_auprc": math.nan,
            "threshold": float(fit.threshold_selection.threshold),
            "train_rows": int(fit.train_rows),
            "validation_rows": int(fit.validation_rows),
            "test_rows": int(len(test)),
            "feature_dim": int(fit.feature_dim),
            "class_weight": "none" if class_weight is None else str(class_weight),
            "c_grid": json.dumps(
                [float(value) for value in grid], separators=(",", ":")
            ),
        }
    )


def _append_multiclass_fit(
    prediction_rows: list[dict[str, Any]],
    hyperparameter_rows: list[dict[str, Any]],
    *,
    patient_ids: np.ndarray,
    fold: int,
    labels: np.ndarray,
    matrix: np.ndarray,
    indices: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    grid: Sequence[float],
    metadata: Mapping[str, Any],
) -> None:
    train, validation, test = (indices[name] for name in ("train", "val", "test"))
    logistic = config["logistic"]
    fit = fit_multiclass_logistic(
        matrix[train],
        labels[train],
        matrix[validation],
        labels[validation],
        grid,
        solver=str(logistic["solver"]),
        max_iter=int(logistic["max_iter"]),
        random_state=0,
    )
    if set(str(value) for value in fit.classes) != set(SUBTYPE_CLASSES):
        raise ValueError("subtype class contract drifted")
    probability = fit.predict_proba(matrix[test])
    predicted = fit.predict(matrix[test])
    class_lookup = {str(value): index for index, value in enumerate(fit.classes)}
    for offset, row_index in enumerate(test):
        row = _base_prediction(metadata, str(patient_ids[row_index]), fold)
        row.update(
            {
                "y_true": str(labels[row_index]),
                "predicted_probability": math.nan,
                "predicted_label": str(predicted[offset]),
                "threshold": math.nan,
            }
        )
        for output_index, subtype in enumerate(SUBTYPE_CLASSES):
            row[SUBTYPE_PROBABILITY_COLUMNS[output_index]] = float(
                probability[offset, class_lookup[subtype]]
            )
            row[SUBTYPE_LABEL_COLUMNS[output_index]] = subtype
        prediction_rows.append(row)
    hyperparameter_rows.append(
        {
            **{name: metadata.get(name, "") for name in HYPERPARAMETER_COLUMNS[:9]},
            "fold": int(fold),
            "selected_c": float(fit.selected_c),
            "validation_auroc": float(fit.validation_macro_ovr_auroc),
            "validation_auprc": float(fit.validation_macro_ovr_auprc),
            "threshold": math.nan,
            "train_rows": int(fit.train_rows),
            "validation_rows": int(fit.validation_rows),
            "test_rows": int(len(test)),
            "feature_dim": int(fit.feature_dim),
            "class_weight": "balanced",
            "c_grid": json.dumps(
                [float(value) for value in grid], separators=(",", ":")
            ),
        }
    )


def _prediction_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        raise ValueError("analysis produced no held-out predictions")
    return pd.DataFrame(rows).reindex(columns=PREDICTION_COLUMNS)


def aggregate_oof(predictions: pd.DataFrame) -> pd.DataFrame:
    """Pool fold-test predictions into public seed/arm/view-level metrics."""

    group_columns = ["seed", "arm", "view", "target", "variant", "population"]
    rows: list[dict[str, Any]] = []
    for key, group in predictions.groupby(group_columns, sort=True, dropna=False):
        if group["patient_id"].duplicated().any():
            raise ValueError(f"OOF group repeats patients: {key}")
        target = str(key[3])
        if target == "subtype_4class":
            probability = group.loc[:, SUBTYPE_PROBABILITY_COLUMNS].to_numpy(
                dtype=float
            )
            metrics = multiclass_metrics(
                group["y_true"].astype(str).to_numpy(),
                probability,
                classes=SUBTYPE_CLASSES,
            )
            values = {
                "n": int(metrics["n"]),
                "n_positive": math.nan,
                "n_negative": math.nan,
                "n_classes": 4,
                "auroc": float(metrics["macro_ovr_auroc"]),
                "auprc": float(metrics["macro_ovr_auprc"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "brier": math.nan,
            }
        else:
            labels = group["y_true"].to_numpy(dtype=np.int64)
            probability = group["predicted_probability"].to_numpy(dtype=float)
            prediction = group["predicted_label"].to_numpy(dtype=np.int64)
            if set(np.unique(labels)) != {0, 1}:
                raise ValueError(f"binary OOF group lacks both classes: {key}")
            values = {
                "n": int(len(group)),
                "n_positive": int(labels.sum()),
                "n_negative": int(np.sum(labels == 0)),
                "n_classes": 2,
                "auroc": float(roc_auc_score(labels, probability)),
                "auprc": float(average_precision_score(labels, probability)),
                "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
                "brier": (
                    float(np.mean(np.square(probability - labels)))
                    if target == "pCR"
                    else math.nan
                ),
            }
        rows.append({**dict(zip(group_columns, key, strict=True)), **values})
    return pd.DataFrame(rows).reindex(columns=METRIC_COLUMNS)


def _require_metric_coverage(
    metrics: pd.DataFrame, population_sizes: Mapping[str, int]
) -> None:
    for row in metrics.itertuples(index=False):
        expected = population_sizes.get(str(row.population))
        if expected is not None and int(row.n) != expected:
            raise ValueError(
                f"OOF coverage drifted for {row.population}/{row.arm}/{row.view}/"
                f"{row.target}/{row.variant}: {row.n} != {expected}"
            )


def _require_metric_rows(metrics: pd.DataFrame, expected: int, analysis: str) -> None:
    if len(metrics) != expected:
        raise ValueError(
            f"{analysis} aggregate matrix is incomplete: {len(metrics)} rows, expected {expected}"
        )


def _phenotype_labels(clinical: pd.DataFrame, target: str) -> np.ndarray:
    if target == "HR":
        return clinical["label_hr"].to_numpy(dtype=np.int64)
    if target == "HER2":
        return clinical["label_her2"].to_numpy(dtype=np.int64)
    if target == "subtype_4class":
        return clinical["hr_her2_subtype"].astype(str).to_numpy()
    raise ValueError(f"unknown phenotype target: {target}")


def run_phenotype_probes(
    config: Mapping[str, Any],
    clinical: pd.DataFrame,
    assets: Mapping[tuple[int, str, int], SpatialFeatureAsset],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """P1--P5 outer-fold-isolated HR/HER2/subtype probes at T0--T3."""

    prediction_rows: list[dict[str, Any]] = []
    hyperparameters: list[dict[str, Any]] = []
    class_weight = str(config["logistic"]["phenotype_class_weight"])
    for (seed, arm, fold), asset in sorted(assets.items()):
        aligned = _aligned_clinical(clinical, asset.patient_id)
        indices = _split_indices(asset.split)
        variants = pooling_features(asset)
        for view in config["analysis"]["phenotype_views"]:
            for variant in POOLING_VARIANTS:
                matrix = static_visit(variants[variant], str(view))
                grid = _grid(config, variant)
                for target in PHENOTYPE_TARGETS:
                    labels = _phenotype_labels(aligned, target)
                    metadata = {
                        "analysis": "phenotype",
                        "population": "full_808",
                        "seed": seed,
                        "arm": arm,
                        "view": str(view),
                        "target": target,
                        "variant": variant,
                        "clinical_contract": "",
                    }
                    if target == "subtype_4class":
                        _append_multiclass_fit(
                            prediction_rows,
                            hyperparameters,
                            patient_ids=asset.patient_id,
                            fold=fold,
                            labels=labels,
                            matrix=matrix,
                            indices=indices,
                            config=config,
                            grid=grid,
                            metadata=metadata,
                        )
                    else:
                        _append_binary_fit(
                            prediction_rows,
                            hyperparameters,
                            patient_ids=asset.patient_id,
                            fold=fold,
                            labels=labels,
                            matrix=matrix,
                            indices=indices,
                            config=config,
                            grid=grid,
                            class_weight=class_weight,
                            metadata=metadata,
                        )
    predictions = _prediction_frame(prediction_rows)
    metrics = aggregate_oof(predictions)
    _require_metric_coverage(metrics, {"full_808": 808})
    _require_metric_rows(metrics, 240, "phenotype")
    return predictions, metrics, hyperparameters


def _population_mask(
    asset: SpatialFeatureAsset, ftv_ids: set[str], population: str
) -> np.ndarray:
    if population == "full_808":
        return np.ones(len(asset.patient_id), dtype=bool)
    if population == "ftv_complete_375":
        return np.asarray(
            [patient_id in ftv_ids for patient_id in asset.patient_id], dtype=bool
        )
    raise ValueError(f"unknown population: {population}")


def run_mri_only_pcr(
    config: Mapping[str, Any],
    clinical: pd.DataFrame,
    ftv_ids: set[str],
    assets: Mapping[tuple[int, str, int], SpatialFeatureAsset],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """P1--P5 pCR probes on both full and matched cohorts at causal prefixes."""

    prediction_rows: list[dict[str, Any]] = []
    hyperparameters: list[dict[str, Any]] = []
    populations = tuple(config["analysis"]["pcr_populations"])
    for (seed, arm, fold), asset in sorted(assets.items()):
        variants = pooling_features(asset)
        for population in populations:
            mask = _population_mask(asset, ftv_ids, str(population))
            patient_ids = asset.patient_id[mask]
            split = asset.split[mask]
            aligned = _aligned_clinical(clinical, patient_ids)
            labels = aligned["label_pcr"].to_numpy(dtype=np.int64)
            indices = _split_indices(split)
            for view in config["analysis"]["pcr_timings"]:
                for variant in POOLING_VARIANTS:
                    matrix = causal_prefix(variants[variant][mask], str(view))
                    metadata = {
                        "analysis": "mri_only_pcr",
                        "population": str(population),
                        "seed": seed,
                        "arm": arm,
                        "view": str(view),
                        "target": "pCR",
                        "variant": variant,
                        "clinical_contract": "",
                    }
                    _append_binary_fit(
                        prediction_rows,
                        hyperparameters,
                        patient_ids=patient_ids,
                        fold=fold,
                        labels=labels,
                        matrix=matrix,
                        indices=indices,
                        config=config,
                        grid=_grid(config, variant),
                        class_weight=None,
                        metadata=metadata,
                    )
    predictions = _prediction_frame(prediction_rows)
    metrics = aggregate_oof(predictions)
    _require_metric_coverage(metrics, {"full_808": 808, "ftv_complete_375": 375})
    _require_metric_rows(metrics, 160, "MRI-only pCR")
    return predictions, metrics, hyperparameters


def run_beyond_ftv(
    config: Mapping[str, Any],
    clinical: pd.DataFrame,
    ftv: pd.DataFrame,
    assets: Mapping[tuple[int, str, int], SpatialFeatureAsset],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Fit the five exact C/F/incremental models on the matched 375 cohort."""

    prediction_rows: list[dict[str, Any]] = []
    hyperparameters: list[dict[str, Any]] = []
    ftv_ids = set(ftv["patient_id"].astype(str))
    contract = str(config["analysis"]["primary_clinical_contract"])
    registered_models = tuple(config["analysis"]["beyond_ftv_models"])
    expected_models = ("C", "C+F", "C+F+P1", "C+F+P3", "C+F+P4")
    if registered_models != expected_models:
        raise ValueError("beyond-FTV model contract drifted")
    general_grid = tuple(float(value) for value in config["logistic"]["c_grid"])
    for (seed, arm, fold), asset in sorted(assets.items()):
        mask = _population_mask(asset, ftv_ids, "ftv_complete_375")
        patient_ids = asset.patient_id[mask]
        split = asset.split[mask]
        aligned = _aligned_clinical(clinical, patient_ids)
        aligned_ftv = _aligned_ftv(ftv, patient_ids)
        labels = aligned["label_pcr"].to_numpy(dtype=np.int64)
        indices = _split_indices(split)
        clinical_matrix = _clinical_matrix(config, aligned, indices, contract)
        variants = pooling_features(asset)
        for view in config["analysis"]["pcr_timings"]:
            ftv_matrix = np.asarray(
                ftv_timing_prefix(aligned_ftv, timing_end_index(str(view))), dtype=float
            )
            p1 = causal_prefix(variants["P1"][mask], str(view))
            p3 = causal_prefix(variants["P3"][mask], str(view))
            p4 = causal_prefix(variants["P4"][mask], str(view))
            feature_sets = {
                "C": clinical_matrix,
                "C+F": np.concatenate((clinical_matrix, ftv_matrix), axis=1),
                "C+F+P1": np.concatenate((clinical_matrix, ftv_matrix, p1), axis=1),
                "C+F+P3": np.concatenate((clinical_matrix, ftv_matrix, p3), axis=1),
                "C+F+P4": np.concatenate((clinical_matrix, ftv_matrix, p4), axis=1),
            }
            if tuple(feature_sets) != registered_models:
                raise AssertionError("beyond-FTV feature-set ordering drifted")
            for model_name, matrix in feature_sets.items():
                metadata = {
                    "analysis": "beyond_ftv",
                    "population": "ftv_complete_375",
                    "seed": seed,
                    "arm": arm,
                    "view": str(view),
                    "target": "pCR",
                    "variant": model_name,
                    "clinical_contract": contract,
                }
                _append_binary_fit(
                    prediction_rows,
                    hyperparameters,
                    patient_ids=patient_ids,
                    fold=fold,
                    labels=labels,
                    matrix=matrix,
                    indices=indices,
                    config=config,
                    grid=general_grid,
                    class_weight=None,
                    metadata=metadata,
                )
    predictions = _prediction_frame(prediction_rows)
    metrics = aggregate_oof(predictions)
    _require_metric_coverage(metrics, {"ftv_complete_375": 375})
    _require_metric_rows(metrics, 80, "beyond-FTV")
    public = metrics.rename(columns={"variant": "model"})
    public.insert(5, "clinical_contract", contract)
    run_columns = ["seed", "arm", "view", "target", "population"]
    for reference_name, suffix in (("C+F", "cf"), ("C+F+P1", "cf_p1")):
        reference = public.loc[
            public["model"].eq(reference_name),
            [*run_columns, "auroc", "auprc", "brier"],
        ].rename(
            columns={
                "auroc": f"reference_auroc_{suffix}",
                "auprc": f"reference_auprc_{suffix}",
                "brier": f"reference_brier_{suffix}",
            }
        )
        public = public.merge(
            reference, on=run_columns, how="left", validate="many_to_one"
        )
        public[f"delta_auroc_vs_{reference_name}"] = (
            public["auroc"] - public[f"reference_auroc_{suffix}"]
        )
        public[f"delta_auprc_vs_{reference_name}"] = (
            public["auprc"] - public[f"reference_auprc_{suffix}"]
        )
        public[f"brier_improvement_vs_{reference_name}"] = (
            public[f"reference_brier_{suffix}"] - public["brier"]
        )
        public = public.drop(
            columns=[
                f"reference_auroc_{suffix}",
                f"reference_auprc_{suffix}",
                f"reference_brier_{suffix}",
            ]
        )
    return predictions, public, hyperparameters


def run_residualized_pcr(
    config: Mapping[str, Any],
    clinical: pd.DataFrame,
    ftv: pd.DataFrame,
    assets: Mapping[tuple[int, str, int], SpatialFeatureAsset],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Residualize P1/P3 against causal FTV using outer train only, then probe pCR."""

    prediction_rows: list[dict[str, Any]] = []
    hyperparameters: list[dict[str, Any]] = []
    ftv_ids = set(ftv["patient_id"].astype(str))
    contract = str(config["analysis"]["primary_clinical_contract"])
    registered_models = tuple(config["analysis"]["residual_models"])
    expected_models = ("P1_res", "P3_res", "C+F+P1_res", "C+F+P3_res")
    if registered_models != expected_models:
        raise ValueError("residual model contract drifted")
    general_grid = tuple(float(value) for value in config["logistic"]["c_grid"])
    for (seed, arm, fold), asset in sorted(assets.items()):
        mask = _population_mask(asset, ftv_ids, "ftv_complete_375")
        patient_ids = asset.patient_id[mask]
        split = asset.split[mask]
        aligned = _aligned_clinical(clinical, patient_ids)
        aligned_ftv = _aligned_ftv(ftv, patient_ids)
        labels = aligned["label_pcr"].to_numpy(dtype=np.int64)
        indices = _split_indices(split)
        clinical_matrix = _clinical_matrix(config, aligned, indices, contract)
        variants = pooling_features(asset)
        for view in config["analysis"]["pcr_timings"]:
            ftv_matrix = np.asarray(
                ftv_timing_prefix(aligned_ftv, timing_end_index(str(view))), dtype=float
            )
            residuals: dict[str, np.ndarray] = {}
            for variant in ("P1", "P3"):
                image = causal_prefix(variants[variant][mask], str(view))
                residualizer = fit_ftv_mri_residualizer(
                    ftv_matrix[indices["train"]], image[indices["train"]]
                )
                residuals[variant] = residualizer.transform(ftv_matrix, image)
            feature_sets = {
                "P1_res": residuals["P1"],
                "P3_res": residuals["P3"],
                "C+F+P1_res": np.concatenate(
                    (clinical_matrix, ftv_matrix, residuals["P1"]), axis=1
                ),
                "C+F+P3_res": np.concatenate(
                    (clinical_matrix, ftv_matrix, residuals["P3"]), axis=1
                ),
            }
            for model_name, matrix in feature_sets.items():
                metadata = {
                    "analysis": "residualized_pcr",
                    "population": "ftv_complete_375",
                    "seed": seed,
                    "arm": arm,
                    "view": str(view),
                    "target": "pCR",
                    "variant": model_name,
                    "clinical_contract": (
                        contract if model_name.startswith("C+") else ""
                    ),
                }
                _append_binary_fit(
                    prediction_rows,
                    hyperparameters,
                    patient_ids=patient_ids,
                    fold=fold,
                    labels=labels,
                    matrix=matrix,
                    indices=indices,
                    config=config,
                    grid=general_grid,
                    class_weight=None,
                    metadata=metadata,
                )
    predictions = _prediction_frame(prediction_rows)
    metrics = aggregate_oof(predictions)
    _require_metric_coverage(metrics, {"ftv_complete_375": 375})
    _require_metric_rows(metrics, 64, "residualized pCR")
    return predictions, metrics, hyperparameters


def run_longitudinal_pcr(
    config: Mapping[str, Any],
    clinical: pd.DataFrame,
    ftv_ids: set[str],
    assets: Mapping[tuple[int, str, int], SpatialFeatureAsset],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Probe the three adjacent, deterministic mean/SD change variants."""

    prediction_rows: list[dict[str, Any]] = []
    hyperparameters: list[dict[str, Any]] = []
    transitions = tuple(config["analysis"]["longitudinal_transitions"])
    registered_variants = tuple(config["analysis"]["longitudinal_variants"])
    if transitions != ("T0->T1", "T1->T2", "T2->T3"):
        raise ValueError("longitudinal transition contract drifted")
    if registered_variants != ("DELTA_MEAN", "DELTA_STD", "P3_PLUS_DELTA"):
        raise ValueError("longitudinal variant contract drifted")
    general_grid = tuple(float(value) for value in config["logistic"]["c_grid"])
    for (seed, arm, fold), asset in sorted(assets.items()):
        p3 = np.concatenate((asset.mean, asset.std), axis=2)
        for population in config["analysis"]["pcr_populations"]:
            mask = _population_mask(asset, ftv_ids, str(population))
            patient_ids = asset.patient_id[mask]
            split = asset.split[mask]
            aligned = _aligned_clinical(clinical, patient_ids)
            labels = aligned["label_pcr"].to_numpy(dtype=np.int64)
            indices = _split_indices(split)
            for transition in transitions:
                start_name, end_name = transition.split("->")
                start, end = VISITS.index(start_name), VISITS.index(end_name)
                if end != start + 1:
                    raise ValueError("longitudinal variants must use adjacent visits")
                feature_sets = {
                    "DELTA_MEAN": asset.mean[mask, end] - asset.mean[mask, start],
                    "DELTA_STD": asset.std[mask, end] - asset.std[mask, start],
                    "P3_PLUS_DELTA": np.concatenate(
                        (p3[mask, start], p3[mask, end] - p3[mask, start]), axis=1
                    ),
                }
                for variant, matrix in feature_sets.items():
                    metadata = {
                        "analysis": "longitudinal_pcr",
                        "population": str(population),
                        "seed": seed,
                        "arm": arm,
                        "view": transition,
                        "target": "pCR",
                        "variant": variant,
                        "clinical_contract": "",
                    }
                    _append_binary_fit(
                        prediction_rows,
                        hyperparameters,
                        patient_ids=patient_ids,
                        fold=fold,
                        labels=labels,
                        matrix=matrix,
                        indices=indices,
                        config=config,
                        grid=general_grid,
                        class_weight=None,
                        metadata=metadata,
                    )
    predictions = _prediction_frame(prediction_rows)
    metrics = aggregate_oof(predictions)
    _require_metric_coverage(metrics, {"full_808": 808, "ftv_complete_375": 375})
    _require_metric_rows(metrics, 72, "longitudinal pCR")
    return predictions, metrics, hyperparameters


def run_oracle_probes(
    config: Mapping[str, Any],
    clinical: pd.DataFrame,
    assets: Mapping[tuple[int, str, int], SpatialFeatureAsset],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Run region diagnostics and fixed-P3 controls on identical intersections."""

    prediction_rows: list[dict[str, Any]] = []
    hyperparameters: list[dict[str, Any]] = []
    phenotype_weight = str(config["logistic"]["phenotype_class_weight"])
    for (seed, arm, fold), asset in sorted(assets.items()):
        all_clinical = _aligned_clinical(clinical, asset.patient_id)
        variants = oracle_features(asset)

        # Each oracle variant uses only its own validity in the pair with
        # mask-free fixed P3; unrelated empty regions never shrink it.
        for view in config["analysis"]["phenotype_views"]:
            for comparator in ORACLE_COMPARATORS:
                mask = oracle_complete_case_mask(
                    asset.oracle_valid, str(view), comparator, prefix=False
                )
                patient_ids = asset.patient_id[mask]
                split = asset.split[mask]
                aligned = all_clinical.loc[mask].reset_index(drop=True)
                indices = _split_indices(split)
                for target in PHENOTYPE_TARGETS:
                    labels = _phenotype_labels(aligned, target)
                    for variant in (comparator, "FIXED_P3"):
                        matrix = static_visit(variants[variant][mask], str(view))
                        metadata = {
                            "analysis": "oracle_phenotype",
                            "population": f"oracle_pair_{comparator}",
                            "seed": seed,
                            "arm": arm,
                            "view": str(view),
                            "target": target,
                            "variant": variant,
                            "clinical_contract": "",
                        }
                        if target == "subtype_4class":
                            _append_multiclass_fit(
                                prediction_rows,
                                hyperparameters,
                                patient_ids=patient_ids,
                                fold=fold,
                                labels=labels,
                                matrix=matrix,
                                indices=indices,
                                config=config,
                                grid=_grid(config, variant),
                                metadata=metadata,
                            )
                        else:
                            _append_binary_fit(
                                prediction_rows,
                                hyperparameters,
                                patient_ids=patient_ids,
                                fold=fold,
                                labels=labels,
                                matrix=matrix,
                                indices=indices,
                                config=config,
                                grid=_grid(config, variant),
                                class_weight=phenotype_weight,
                                metadata=metadata,
                            )

        # pCR is causal: a Tk prefix may use oracle masks/statistics only through
        # Tk, never a later lesion mask.  Complete-case matching spans exactly
        # those already-observed visits.
        for view in config["analysis"]["pcr_timings"]:
            for comparator in ORACLE_COMPARATORS:
                mask = oracle_complete_case_mask(
                    asset.oracle_valid, str(view), comparator, prefix=True
                )
                patient_ids = asset.patient_id[mask]
                split = asset.split[mask]
                aligned = all_clinical.loc[mask].reset_index(drop=True)
                labels = aligned["label_pcr"].to_numpy(dtype=np.int64)
                indices = _split_indices(split)
                for variant in (comparator, "FIXED_P3"):
                    matrix = causal_prefix(variants[variant][mask], str(view))
                    metadata = {
                        "analysis": "oracle_pcr",
                        "population": f"oracle_pair_{comparator}",
                        "seed": seed,
                        "arm": arm,
                        "view": str(view),
                        "target": "pCR",
                        "variant": variant,
                        "clinical_contract": "",
                    }
                    _append_binary_fit(
                        prediction_rows,
                        hyperparameters,
                        patient_ids=patient_ids,
                        fold=fold,
                        labels=labels,
                        matrix=matrix,
                        indices=indices,
                        config=config,
                        grid=_grid(config, variant),
                        class_weight=None,
                        metadata=metadata,
                    )

    predictions = _prediction_frame(prediction_rows)
    metrics = aggregate_oof(predictions)
    _require_metric_rows(metrics, 640, "oracle")
    comparison_key = ["seed", "arm", "view", "target", "population"]
    for key, group in metrics.groupby(comparison_key, sort=True):
        comparator = str(key[-1]).removeprefix("oracle_pair_")
        if comparator not in ORACLE_COMPARATORS or set(group["variant"]) != {
            comparator,
            "FIXED_P3",
        }:
            raise ValueError(f"oracle comparison variants are incomplete for {key}")
        if group["n"].nunique() != 1:
            raise ValueError(f"oracle/fixed-P3 comparison populations differ for {key}")
    return predictions, metrics, hyperparameters


def pooling_contract_table(config: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant in POOLING_VARIANTS:
        definition = config["poolings"][variant]
        rows.append(
            {
                "variant": variant,
                "components": "+".join(
                    str(value) for value in definition["components"]
                ),
                "dimension": int(definition["dimension"]),
                "role": str(definition["role"]),
                "c_grid": json.dumps(
                    list(_grid(config, variant)), separators=(",", ":")
                ),
                "mask_free": True,
                "deployable": variant != "P5",
            }
        )
    for variant in ORACLE_VARIANTS:
        if variant == "FIXED_P3":
            continue
        rows.append(
            {
                "variant": variant,
                "components": (
                    "core_mean+core_std+peri10_mean+peri10_std+peri20_mean+peri20_std"
                    if variant == "CORE_PERI"
                    else f"{variant.lower()}_mean+{variant.lower()}_std"
                ),
                "dimension": 768 if variant == "CORE_PERI" else 256,
                "role": "oracle_diagnostic_not_deployable",
                "c_grid": json.dumps(
                    list(_grid(config, variant)), separators=(",", ":")
                ),
                "mask_free": False,
                "deployable": False,
            }
        )
    return pd.DataFrame(rows)


def _plain(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _delta_candidates(
    frame: pd.DataFrame,
    *,
    variant_column: str,
    reference: str,
    comparison: str,
    group_columns: Sequence[str],
    expected_seeds: Sequence[int],
    threshold: float,
    strict: bool,
) -> list[dict[str, Any]]:
    required = {*group_columns, "seed", variant_column, "auroc"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"gate metric table is missing columns: {missing}")
    selected = frame.loc[frame[variant_column].isin((reference, comparison))]
    identity_columns = [*group_columns, "seed", variant_column]
    if selected.duplicated(identity_columns).any():
        raise ValueError("gate metric table repeats a comparison identity")
    pivot = selected.pivot_table(
        index=[*group_columns, "seed"],
        columns=variant_column,
        values="auroc",
        aggfunc="first",
    )
    if reference not in pivot or comparison not in pivot:
        return []
    deltas = (pivot[comparison] - pivot[reference]).rename("delta").reset_index()
    candidates: list[dict[str, Any]] = []
    expected = {int(value) for value in expected_seeds}
    for key, group in deltas.groupby(list(group_columns), sort=True, dropna=False):
        values = key if isinstance(key, tuple) else (key,)
        by_seed = {
            int(row.seed): float(row.delta)
            for row in group.itertuples(index=False)
            if np.isfinite(float(row.delta))
        }
        if set(by_seed) != expected:
            continue
        passed = all(
            delta > threshold if strict else delta >= threshold
            for delta in by_seed.values()
        )
        candidates.append(
            {
                **{
                    name: _plain(value)
                    for name, value in zip(group_columns, values, strict=True)
                },
                "reference": reference,
                "comparison": comparison,
                "seed_deltas": {str(seed): by_seed[seed] for seed in sorted(by_seed)},
                "passed": passed,
            }
        )
    return candidates


def evaluate_gates(
    config: Mapping[str, Any],
    phenotype_metrics: pd.DataFrame,
    mri_pcr_metrics: pd.DataFrame,
    beyond_ftv_metrics: pd.DataFrame,
    oracle_metrics: pd.DataFrame,
) -> dict[str, Any]:
    """Apply preregistered Gates A--D to aggregate OOF metrics only."""

    seeds = tuple(int(value) for value in config["frozen_cells"]["seed_bases"])
    if seeds != (2026, 3026):
        raise ValueError("gate evaluation requires the two locked seed bases")
    gate_config = config["gates"]

    threshold_a = float(gate_config["A"]["minimum_auroc_gain_each_seed"])
    phenotype_a = phenotype_metrics.loc[
        phenotype_metrics["target"].isin(("HER2", "subtype_4class"))
    ]
    evidence_a = _delta_candidates(
        phenotype_a,
        variant_column="variant",
        reference="P1",
        comparison="P3",
        group_columns=("arm", "view", "target", "population"),
        expected_seeds=seeds,
        threshold=threshold_a,
        strict=False,
    )
    primary_population = str(config["analysis"]["primary_pcr_population"])
    pcr_a = mri_pcr_metrics.loc[
        mri_pcr_metrics["population"].eq(primary_population)
        & mri_pcr_metrics["target"].eq("pCR")
    ]
    evidence_a.extend(
        _delta_candidates(
            pcr_a,
            variant_column="variant",
            reference="P1",
            comparison="P3",
            group_columns=("arm", "view", "target", "population"),
            expected_seeds=seeds,
            threshold=threshold_a,
            strict=False,
        )
    )
    support_a = [row for row in evidence_a if bool(row["passed"])]
    gate_a_passed = bool(support_a)

    # Gate B requires both positive contrasts in the same arm/timing for both
    # seeds, not merely one positive contrast at each of two different timings.
    early = set(str(value) for value in gate_config["B"]["timings"])
    threshold_b = float(gate_config["B"]["minimum_gain_each_seed_strictly_gt"])
    selected_b = beyond_ftv_metrics.loc[beyond_ftv_metrics["view"].isin(early)]
    b_index = ["arm", "view", "target", "population", "seed"]
    if selected_b.duplicated([*b_index, "model"]).any():
        raise ValueError("Gate-B metric table repeats a model identity")
    b_pivot = selected_b.pivot_table(
        index=b_index, columns="model", values="auroc", aggfunc="first"
    )
    evidence_b: list[dict[str, Any]] = []
    required_b_models = ("C+F", "C+F+P1", "C+F+P3")
    if set(required_b_models).issubset(b_pivot.columns):
        b_values = b_pivot.reset_index()
        b_values["delta_vs_cf"] = b_values["C+F+P3"] - b_values["C+F"]
        b_values["delta_vs_cf_p1"] = b_values["C+F+P3"] - b_values["C+F+P1"]
        group_b = ("arm", "view", "target", "population")
        for key, group in b_values.groupby(list(group_b), sort=True):
            by_seed = {
                int(row.seed): {
                    "C+F+P3_minus_C+F": float(row.delta_vs_cf),
                    "C+F+P3_minus_C+F+P1": float(row.delta_vs_cf_p1),
                }
                for row in group.itertuples(index=False)
                if np.isfinite(float(row.delta_vs_cf))
                and np.isfinite(float(row.delta_vs_cf_p1))
            }
            if set(by_seed) != set(seeds):
                continue
            passed_b = all(
                values["C+F+P3_minus_C+F"] > threshold_b
                and values["C+F+P3_minus_C+F+P1"] > threshold_b
                for values in by_seed.values()
            )
            evidence_b.append(
                {
                    **dict(zip(group_b, (_plain(value) for value in key), strict=True)),
                    "seed_deltas": {
                        str(seed): by_seed[seed] for seed in sorted(by_seed)
                    },
                    "passed": passed_b,
                }
            )
    support_b = [row for row in evidence_b if bool(row["passed"])]
    gate_b_passed = bool(support_b)

    threshold_c = float(gate_config["C"]["minimum_matched_auroc_gain_each_seed"])
    evidence_c: list[dict[str, Any]] = []
    for comparison in ("CORE", "PERI10", "PERI20", "CORE_PERI"):
        evidence_c.extend(
            _delta_candidates(
                oracle_metrics,
                variant_column="variant",
                reference="FIXED_P3",
                comparison=comparison,
                group_columns=("arm", "view", "target", "population"),
                expected_seeds=seeds,
                threshold=threshold_c,
                strict=False,
            )
        )
    support_c = [row for row in evidence_c if bool(row["passed"])]
    gate_c_passed = bool(support_c)

    # Gate D uses only static phenotype endpoints and the exact registered
    # fixed/oracle variants named in the plan.  It is conjunctive with A/B/C fail.
    fixed_d = phenotype_metrics.loc[
        phenotype_metrics["target"].isin(PHENOTYPE_TARGETS)
        & phenotype_metrics["variant"].isin(("P1", "P3", "P4"))
    ].assign(source="mask_free")
    oracle_d = oracle_metrics.loc[
        oracle_metrics["target"].isin(PHENOTYPE_TARGETS)
        & oracle_metrics["variant"].isin(("CORE", "PERI10", "PERI20", "CORE_PERI"))
    ].assign(source="oracle")
    d_values = pd.concat((fixed_d, oracle_d), ignore_index=True)
    d_groups = ["source", "arm", "view", "target", "variant", "population"]
    if d_values.duplicated([*d_groups, "seed"]).any():
        raise ValueError("Gate-D metric table repeats a seed/variant identity")
    seed_counts = d_values.groupby(d_groups, dropna=False)["seed"].nunique()
    if d_values.empty or not seed_counts.eq(len(seeds)).all():
        raise ValueError("Gate D requires complete two-seed phenotype metrics")
    seed_means = d_values.groupby(d_groups, dropna=False)["auroc"].mean()
    minimum_seed_mean = float(seed_means.min())
    maximum_seed_mean = float(seed_means.max())
    threshold_d_lower = float(gate_config["D"]["minimum_seed_mean_auroc_near_chance"])
    threshold_d = float(gate_config["D"]["maximum_seed_mean_auroc_near_chance"])
    evidence_d = [
        {
            **{name: _plain(value) for name, value in zip(d_groups, key, strict=True)},
            "seed_mean_auroc": float(value),
            "within_near_chance_interval": bool(
                threshold_d_lower <= float(value) <= threshold_d
            ),
        }
        for key, value in seed_means.items()
    ]
    gate_d_passed = bool(
        not gate_a_passed
        and not gate_b_passed
        and not gate_c_passed
        and minimum_seed_mean >= threshold_d_lower
        and maximum_seed_mean <= threshold_d
    )

    if gate_d_passed:
        classification = "CURRENT ENCODER LACKS PHENOTYPE INFORMATION"
    elif (gate_a_passed or gate_b_passed) and not gate_c_passed:
        classification = "PHENOTYPE INFORMATION PRESENT BUT MEAN-POOLED AWAY"
    elif gate_c_passed and not gate_a_passed and not gate_b_passed:
        classification = "PHENOTYPE SPATIALLY LOCALIZED"
    else:
        classification = "MIXED"
    authorized = bool(gate_a_passed or gate_c_passed)
    return {
        "schema_version": 1,
        "experiment": "spatial_heterogeneity_phenotype_audit",
        "stage": "A",
        "status": "COMPLETE",
        "gates": {
            "A": {
                "name": "HETEROGENEITY_SIGNAL_SUPPORTED",
                "passed": gate_a_passed,
                "minimum_auroc_gain_each_seed": threshold_a,
                "evaluated_comparisons": evidence_a,
                "supporting_comparisons": support_a,
            },
            "B": {
                "name": "HETEROGENEITY_COMPLEMENTARITY_SUPPORTED",
                "passed": gate_b_passed,
                "minimum_gain_each_seed_strictly_gt": threshold_b,
                "evaluated_comparisons": evidence_b,
                "supporting_comparisons": support_b,
            },
            "C": {
                "name": "PHENOTYPE_IS_SPATIALLY_LOCALIZED",
                "passed": gate_c_passed,
                "minimum_matched_auroc_gain_each_seed": threshold_c,
                "evaluated_comparisons": evidence_c,
                "supporting_comparisons": support_c,
            },
            "D": {
                "name": "CURRENT_ENCODER_LACKS_PHENOTYPE_SIGNAL",
                "passed": gate_d_passed,
                "minimum_seed_mean_auroc_near_chance": threshold_d_lower,
                "maximum_seed_mean_auroc_near_chance": threshold_d,
                "observed_minimum_seed_mean_auroc": minimum_seed_mean,
                "observed_maximum_seed_mean_auroc": maximum_seed_mean,
                "requires_gates_a_b_and_c_fail": True,
                "evaluated_comparisons": evidence_d,
            },
        },
        "scientific_classification": classification,
        "stage_b_authorized": authorized,
        "contains_patient_level_data": False,
    }


def stage_b_authorization(
    config: Mapping[str, Any],
    gates: Mapping[str, Any],
    chain: Mapping[str, Any],
    config_sha256: str,
) -> dict[str, Any]:
    gate_values = gates["gates"]
    gate_a = bool(gate_values["A"]["passed"])
    gate_c = bool(gate_values["C"]["passed"])
    authorized = gate_a or gate_c
    return {
        "schema_version": 2,
        "experiment": "spatial_heterogeneity_phenotype_audit",
        "authorization_rule": "Gate A OR Gate C",
        "authorized": authorized,
        "status": (
            "AUTHORIZED_PENDING_EXECUTION" if authorized else "NOT_RUN_NOT_AUTHORIZED"
        ),
        "reason": (
            "At least one preregistered Stage-A authorization gate passed."
            if authorized
            else "Neither Gate A nor Gate C passed; Stage B must not run."
        ),
        "gate_a_passed": gate_a,
        "gate_c_passed": gate_c,
        "stage_a_scientific_classification": gates["scientific_classification"],
        "stage_b_contract": dict(config["stage_b"]),
        "config_sha256": config_sha256,
        "preregistration_lock_sha256": chain["active_preregistration_lock_sha256"],
        "preregistration_chain": dict(chain),
        "contains_patient_level_data": False,
    }


def _prepare_output_policy(paths: Mapping[str, Path]) -> tuple[Path, ...]:
    """Recover an interrupted run, while making completed results immutable."""

    summary = paths["run_summary"]
    if summary.exists():
        raise FileExistsError(
            "run_summary.json exists; completed formal Stage-A results are immutable"
        )
    partial = tuple(
        path for name, path in paths.items() if name != "run_summary" and path.exists()
    )
    for path in partial:
        if path.is_dir():
            raise IsADirectoryError(
                f"known Stage-A output path became a directory: {path}"
            )
        path.unlink()
    return partial


def _write_private_predictions(frame: pd.DataFrame, path: Path) -> None:
    if "patient_id" not in frame or frame["patient_id"].isna().any():
        raise ValueError("private OOF table lacks patient identity")
    private_directory(path.parent)
    atomic_csv(frame, path, private=True)


def _public_metric_table(frame: pd.DataFrame) -> pd.DataFrame:
    forbidden = {"patient_id", "clinical_patient_id", "raw_Patient_ID"}
    overlap = forbidden & set(frame.columns)
    if overlap:
        raise ValueError(
            f"public table contains patient-level columns: {sorted(overlap)}"
        )
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args()


def main() -> None:
    parse_args()
    os.umask(0o077)
    started = time.time()
    output_root = ROOT
    feature_root = ROOT / "features"
    config_path = ROOT / "configs" / "audit.json"
    config = load_config(config_path, verify_inputs=True)
    lock = require_preregistration_lock(config)
    chain = preregistration_chain(lock)
    config_sha256 = file_sha256(config_path)
    for private_root in ("features", "manifests", "checkpoints", "predictions"):
        private_directory(ROOT / private_root)
    validate_analysis_contract(config)
    locked_implementations = lock.get("implementation_sha256", {})
    runtime_paths = {
        "scripts/run_audit.py": Path(__file__),
        "scripts/common.py": ROOT / "scripts" / "common.py",
    }
    runtime_hashes = {name: file_sha256(path) for name, path in runtime_paths.items()}
    if any(
        runtime_hashes[name] != locked_implementations.get(name)
        for name in runtime_paths
    ):
        raise ValueError("Stage-A runner/common implementation drifted after freeze")

    # These are deliberately imported, not copied.  Refuse formal analysis if
    # either reusable implementation differs from the hash frozen in audit.json.
    reused_paths = {
        "data_contracts": COMPLEMENTARITY_SCRIPTS / "data_contracts.py",
        "modeling": COMPLEMENTARITY_SCRIPTS / "modeling.py",
    }
    upstream_hashes = config["upstream_code"]
    if (
        file_sha256(reused_paths["data_contracts"])
        != upstream_hashes["complementarity_data_contracts_sha256"]
    ):
        raise ValueError("reused complementarity data_contracts.py hash drifted")
    if (
        file_sha256(reused_paths["modeling"])
        != upstream_hashes["complementarity_modeling_sha256"]
    ):
        raise ValueError("reused complementarity modeling.py hash drifted")

    output_paths = {
        "phenotype_predictions": output_root
        / "predictions"
        / "phenotype_oof.private.csv",
        "mri_pcr_predictions": output_root
        / "predictions"
        / "mri_only_pcr_oof.private.csv",
        "beyond_predictions": output_root
        / "predictions"
        / "beyond_ftv_oof.private.csv",
        "residual_predictions": output_root
        / "predictions"
        / "residualized_pcr_oof.private.csv",
        "longitudinal_predictions": output_root
        / "predictions"
        / "longitudinal_oof.private.csv",
        "oracle_predictions": output_root / "predictions" / "oracle_oof.private.csv",
        "table1": output_root / "metrics" / "table1_pooling_contract.csv",
        "table2": output_root / "metrics" / "table2_phenotype_probes.csv",
        "table3": output_root / "metrics" / "table3_mri_only_pcr.csv",
        "table4": output_root / "metrics" / "table4_clinical_ftv_incremental.csv",
        "table5": output_root / "metrics" / "table5_residualized_mri.csv",
        "table6": output_root / "metrics" / "table6_longitudinal_heterogeneity.csv",
        "table7": output_root / "metrics" / "table7_oracle_regions.csv",
        "hyperparameters": output_root / "metrics" / "hyperparameter_selections.csv",
        "gates": output_root / "metrics" / "gates.json",
        "stage_b_authorization": output_root / "metrics" / "stage_b_authorization.json",
        "run_summary": output_root / "metrics" / "run_summary.json",
    }
    if output_paths["run_summary"].exists():
        raise FileExistsError(
            "run_summary.json exists; completed formal Stage-A results are immutable"
        )
    paths = config["paths"]
    folds = load_fold_manifest(paths["fold_manifest"], paths["fold_manifest_sha256"])
    clinical = load_clinical_table(
        paths["clinical_labels"], paths["clinical_labels_sha256"], folds
    )
    ftv = load_ftv_wide(paths["ftv_table"], paths["ftv_table_sha256"], folds)
    assets = load_all_spatial_feature_assets(feature_root, folds, config, lock)
    ftv_ids = set(ftv["patient_id"].astype(str))
    print(
        "validated 20 private feature assets before label-dependent analysis",
        flush=True,
    )
    removed_partial_outputs = _prepare_output_policy(output_paths)
    if removed_partial_outputs:
        print(
            f"removed {len(removed_partial_outputs)} exact known artifacts from an interrupted run",
            flush=True,
        )

    hyperparameters: list[dict[str, Any]] = []
    atomic_csv(
        _public_metric_table(pooling_contract_table(config)), output_paths["table1"]
    )

    phenotype_predictions, phenotype_metrics, selected = run_phenotype_probes(
        config, clinical, assets
    )
    hyperparameters.extend(selected)
    _write_private_predictions(
        phenotype_predictions, output_paths["phenotype_predictions"]
    )
    atomic_csv(_public_metric_table(phenotype_metrics), output_paths["table2"])
    del phenotype_predictions
    print("completed P1-P5 HR/HER2/subtype probes", flush=True)

    mri_pcr_predictions, mri_pcr_metrics, selected = run_mri_only_pcr(
        config, clinical, ftv_ids, assets
    )
    hyperparameters.extend(selected)
    _write_private_predictions(mri_pcr_predictions, output_paths["mri_pcr_predictions"])
    atomic_csv(_public_metric_table(mri_pcr_metrics), output_paths["table3"])
    del mri_pcr_predictions
    print("completed full-808 and matched-375 MRI-only pCR probes", flush=True)

    beyond_predictions, beyond_metrics, selected = run_beyond_ftv(
        config, clinical, ftv, assets
    )
    hyperparameters.extend(selected)
    _write_private_predictions(beyond_predictions, output_paths["beyond_predictions"])
    atomic_csv(_public_metric_table(beyond_metrics), output_paths["table4"])
    del beyond_predictions
    print("completed exact C/F/P1/P3/P4 incremental models", flush=True)

    residual_predictions, residual_metrics, selected = run_residualized_pcr(
        config, clinical, ftv, assets
    )
    hyperparameters.extend(selected)
    _write_private_predictions(
        residual_predictions, output_paths["residual_predictions"]
    )
    atomic_csv(_public_metric_table(residual_metrics), output_paths["table5"])
    del residual_predictions
    print("completed fold-train-only P1/P3 FTV residualization", flush=True)

    longitudinal_predictions, longitudinal_metrics, selected = run_longitudinal_pcr(
        config, clinical, ftv_ids, assets
    )
    hyperparameters.extend(selected)
    _write_private_predictions(
        longitudinal_predictions, output_paths["longitudinal_predictions"]
    )
    atomic_csv(_public_metric_table(longitudinal_metrics), output_paths["table6"])
    del longitudinal_predictions
    print("completed adjacent longitudinal heterogeneity probes", flush=True)

    oracle_predictions, oracle_metrics, selected = run_oracle_probes(
        config, clinical, assets
    )
    hyperparameters.extend(selected)
    _write_private_predictions(oracle_predictions, output_paths["oracle_predictions"])
    atomic_csv(_public_metric_table(oracle_metrics), output_paths["table7"])
    del oracle_predictions
    print("completed matched oracle-region and fixed-P3 probes", flush=True)

    hyperparameter_frame = pd.DataFrame(hyperparameters).reindex(
        columns=HYPERPARAMETER_COLUMNS
    )
    atomic_csv(
        _public_metric_table(hyperparameter_frame), output_paths["hyperparameters"]
    )
    gates = evaluate_gates(
        config, phenotype_metrics, mri_pcr_metrics, beyond_metrics, oracle_metrics
    )
    atomic_json(gates, output_paths["gates"])
    authorization = stage_b_authorization(config, gates, chain, config_sha256)
    authorization["stage_a_gates_sha256"] = file_sha256(output_paths["gates"])
    atomic_json(authorization, output_paths["stage_b_authorization"])

    if any(
        file_sha256(path) != runtime_hashes[name]
        for name, path in runtime_paths.items()
    ):
        raise RuntimeError(
            "Stage-A runner/common implementation changed during analysis"
        )

    summary = {
        "schema_version": 2,
        "experiment": "spatial_heterogeneity_phenotype_audit",
        "stage": "A",
        "status": "COMPLETE",
        "branch": str(config["branch"]),
        "n_feature_assets": len(assets),
        "n_full_patients": int(clinical["patient_id"].nunique()),
        "n_ftv_complete_patients": int(ftv["patient_id"].nunique()),
        "scientific_classification": gates["scientific_classification"],
        "stage_b_authorized": bool(authorization["authorized"]),
        "elapsed_seconds": float(time.time() - started),
        "config_sha256": config_sha256,
        "preregistration_lock_sha256": chain["active_preregistration_lock_sha256"],
        "preregistration_chain": dict(chain),
        "feature_asset_sha256": {
            f"seed_{seed}/{arm}/fold_{fold}": str(asset.metadata["feature_sha256"])
            for (seed, arm, fold), asset in sorted(assets.items())
        },
        "reused_implementation_sha256": {
            name: file_sha256(path) for name, path in reused_paths.items()
        },
        "runtime_implementation_sha256": runtime_hashes,
        "artifacts": {
            name: {
                "path": str(path.relative_to(output_root)),
                "sha256": file_sha256(path),
                "size_bytes": int(path.stat().st_size),
                "patient_level_private": name.endswith("predictions"),
            }
            for name, path in output_paths.items()
            if name != "run_summary" and path.exists()
        },
        "public_outputs_contain_patient_level_data": False,
    }
    atomic_json(summary, output_paths["run_summary"])
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

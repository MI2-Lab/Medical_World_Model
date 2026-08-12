#!/usr/bin/env python3
"""Run the preregistered mask-free region-aware frozen-feature audit.

The analysis starts only after the complete 2 x 2 x 5 feature matrix has been
validated.  Every scaler, clinical encoder, target transform, and model is fit
on outer train; validation selects the fixed hyperparameter; outer test is
constructed after selection and predicted exactly once.  Patient identifiers
are written only to owner-readable ``*.private.csv`` files.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import warnings

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import pearsonr, spearmanr
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm._base import _fit_liblinear
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
try:
    from sklearn.utils.validation import validate_data
except ImportError:  # sklearn<1.6 import compatibility; formal subtype stays 1.8-only.
    def validate_data(estimator: Any, X: Any, y: Any, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        return estimator._validate_data(X, y, **kwargs)


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
CONFIG_PATH = ROOT / "configs" / "audit.json"
LOCK_PATH = ROOT / "PREREGISTRATION_LOCK.json"
FEATURE_ROOT = ROOT / "features"

COMPLEMENTARITY_SCRIPTS = (
    REPO_ROOT / "additional_experiments" / "mri_clinical_complementarity_audit" / "scripts"
)
STAGE_B_SRC = (
    REPO_ROOT
    / "additional_experiments"
    / "c1b_overlap_eligibility_ftv_stageb"
    / "src"
)
for _dependency in (COMPLEMENTARITY_SCRIPTS, STAGE_B_SRC):
    _value = str(_dependency)
    if _value not in sys.path:
        sys.path.insert(0, _value)

from data_contracts import (  # noqa: E402
    TrainOnlyClinicalEncoder,
    ftv_timing_prefix,
    load_clinical_table,
    load_fold_manifest,
    load_ftv_wide,
)
from modeling import (  # noqa: E402
    SELECTION_ATOL,
    MulticlassCandidateScore,
    MulticlassLogisticFit,
    _matrix,
    _multiclass_labels,
    _positive_grid,
    _positive_integer,
    fit_binary_logistic,
    multiclass_metrics,
    paired_fold_stratified_bootstrap,
)


VISITS = ("T0", "T1", "T2", "T3")
TRANSITIONS = ("T0_to_T1", "T1_to_T2", "T2_to_T3")
PRIMARY_VARIANTS = ("R0", "R1", "R2", "R3", "R4", "R5")
GATE_A_VARIANTS = ("R2", "R3", "R5")
ORACLE_CANDIDATES = ("R1", "R2", "R3", "R5")
ALL_VARIANTS = (
    "R0", "R1", "R2", "R3", "R4", "R5", "R5_RP192",
    "S1", "S2", "S3", "S4", "S5",
)
FEATURE_FILENAME = "regional_features.private.npz"
FEATURE_KEYS = frozenset({"patient_id", "split", *ALL_VARIANTS, "arm", "seed_base", "fold"})
PHENOTYPE_TARGETS = ("HR", "HER2", "subtype_4class")
SUBTYPE_CLASSES = tuple(sorted(("HR+/HER2-", "HR-/HER2-", "HR+/HER2+", "HR-/HER2+")))
SUBTYPE_PROBABILITY_COLUMNS = tuple(f"probability_class_{index}" for index in range(4))
SUBTYPE_LABEL_COLUMNS = tuple(f"class_label_{index}" for index in range(4))
SPLITS = ("train", "val", "test")
EARLY_TIMINGS = ("T0", "T0-T1", "T0-T2")
CLASSIFICATION_ATOL = 1e-12
_EXACT_SKLEARN_VERSION = "1.8.0"
_SKLEARN_PENALTY_WARNING = (
    r"^'penalty' was deprecated in version 1\.8 and will be removed in 1\.10\."
)

PREDICTION_COLUMNS = (
    "patient_id", "fold", "population", "seed", "arm", "analysis", "context",
    "view", "target", "variant", "model", "clinical_contract", "y_true",
    "predicted_probability", "predicted_label", "threshold",
    *SUBTYPE_PROBABILITY_COLUMNS, *SUBTYPE_LABEL_COLUMNS,
)
CLASSIFICATION_METRIC_COLUMNS = (
    "seed", "arm", "analysis", "context", "view", "timing_label", "target",
    "variant", "model", "population", "clinical_contract", "n", "n_positive",
    "n_negative", "n_classes", "auroc", "auprc", "balanced_accuracy", "brier",
)
HYPERPARAMETER_COLUMNS = (
    "analysis", "context", "population", "seed", "arm", "fold", "view", "target",
    "variant", "model", "clinical_contract", "selected_c", "validation_auroc",
    "validation_auprc", "threshold", "train_rows", "validation_rows", "test_rows",
    "feature_dim", "class_weight", "c_grid", "test_used_for_selection",
    "test_predict_call_count",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ordered_sha256(values: Iterable[Any]) -> str:
    return hashlib.sha256("\n".join(str(value) for value in values).encode("utf-8")).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _load_json(path: str | Path, label: str) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable or invalid JSON: {source}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _scalar(array: Any, name: str) -> Any:
    value = np.asarray(array)
    if value.shape != ():
        raise ValueError(f"{name} must be a scalar")
    return value.item()


def _resolve_path(value: Any, *, relative_to: Path = REPO_ROOT) -> Path:
    candidate = Path(str(value)).expanduser()
    return (candidate if candidate.is_absolute() else relative_to / candidate).resolve()


def load_config(path: str | Path = CONFIG_PATH, *, verify_paths: bool = True) -> dict[str, Any]:
    """Load and fail-closed validate the exact preregistered analysis contract."""

    source = Path(path).resolve()
    config = _load_json(source, "audit config")
    if config.get("schema_version") != 1 or config.get("experiment") != "mask_free_region_aware_audit":
        raise ValueError("audit config identity/schema drifted")
    required = {
        "schema_version", "experiment", "branch", "evidence_status", "start", "paths",
        "upstream_code_sha256", "frozen_cells", "feature_contract", "variants", "analysis",
        "logistic", "ridge", "bootstrap", "oracle", "gates", "forbidden",
    }
    if set(config) != required:
        raise ValueError(
            f"audit config keys drifted: missing={sorted(required-set(config))}, "
            f"extra={sorted(set(config)-required)}"
        )
    frozen = config["frozen_cells"]
    exact_frozen = {
        "arms": ["LOCAL0", "LOCAL3"], "seed_bases": [2026, 3026],
        "folds": [0, 1, 2, 3, 4], "visits": list(VISITS),
        "patient_count": 808, "ftv_complete_patient_count": 375,
    }
    if frozen != exact_frozen:
        raise ValueError("frozen 2x2x5 cohort contract drifted")
    dimensions = {name: int(value) for name, value in config["variants"]["dimensions"].items()}
    if tuple(config["analysis"]["all_probe_variants"]) != ALL_VARIANTS or set(dimensions) != set(ALL_VARIANTS):
        raise ValueError("feature variant matrix drifted")
    if tuple(config["analysis"]["pcr_timings"]) != ("T0", "T0-T1", "T0-T2", "T0-T3"):
        raise ValueError("causal pCR timing contract drifted")
    logistic = config["logistic"]
    expected_logistic = {
        "c_grid": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "penalty": "l2", "solver": "liblinear", "max_iter": 10000,
        "selection_tolerance": 1e-12, "pcr_class_weight": None,
        "phenotype_class_weight": "balanced", "selection_metric": "validation_auroc",
        "tie_break": "smaller_C",
    }
    if logistic != expected_logistic:
        raise ValueError("logistic probe contract drifted")
    ridge = config["ridge"]
    if ridge != {
        "alpha_grid": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        "solver": "lsqr", "tol": 1e-8, "max_iter": 10000,
        "selection_metric": "validation_mse", "tie_break": "smaller_alpha",
    }:
        raise ValueError("Ridge probe contract drifted")
    bootstrap = config["bootstrap"]
    if (
        int(bootstrap.get("replicates", -1)) != 2000
        or float(bootstrap.get("confidence_level", -1)) != 0.95
        or int(bootstrap.get("seed", -1)) != 260811
        or bootstrap.get("unit") != "patient_within_outer_fold"
        or tuple(bootstrap.get("candidate_variants", ())) != GATE_A_VARIANTS
        or tuple(bootstrap.get("timings", ())) != EARLY_TIMINGS
        or tuple(bootstrap.get("contexts", ())) != ("MRI_ONLY", "C_PLUS_F")
    ):
        raise ValueError("bootstrap contract drifted")
    paths = dict(config["paths"])
    path_keys = tuple(key for key in paths if not key.endswith("_sha256"))
    for key in path_keys:
        paths[key] = _resolve_path(paths[key])
    if verify_paths:
        for key, path_value in paths.items():
            if key.endswith("_sha256") or key.endswith("_root"):
                continue
            expected_key = f"{key}_sha256"
            if expected_key in paths:
                source_path = Path(path_value)
                if not source_path.is_file():
                    raise FileNotFoundError(source_path)
                if file_sha256(source_path) != _require_sha256(paths[expected_key], expected_key):
                    raise ValueError(f"{key} SHA-256 mismatch")
    output = dict(config)
    output["paths"] = paths
    output["config_path"] = source
    output["config_sha256"] = file_sha256(source)
    return output


@dataclass(frozen=True)
class RegionalFeatureAsset:
    path: Path
    metadata_path: Path
    patient_id: np.ndarray
    split: np.ndarray
    features: Mapping[str, np.ndarray]
    arm: str
    seed: int
    fold: int
    metadata: Mapping[str, Any]

    def variant(self, name: str) -> np.ndarray:
        if name not in self.features:
            raise ValueError(f"unknown feature variant: {name}")
        return self.features[name]


def _expected_assignment(folds: pd.DataFrame, fold: int, patient_count: int) -> dict[str, str]:
    current = folds.loc[folds["fold"].eq(fold), ["patient_id", "split"]]
    if len(current) != patient_count or current["patient_id"].astype(str).duplicated().any():
        raise ValueError(f"fold {fold} does not contain {patient_count} unique patients")
    return dict(zip(current["patient_id"].astype(str), current["split"].astype(str), strict=True))


def _metadata_keys() -> frozenset[str]:
    """Share the exporter's exact metadata schema without importing MRI code."""

    from common import METADATA_KEYS

    return frozenset(str(value) for value in METADATA_KEYS)


def load_fold_assignments(
    path: str | Path,
    expected_sha256: str,
    *,
    patient_count: int = 808,
) -> pd.DataFrame:
    """Load only patient/fold/split fields before label access is authorized."""

    source = Path(path).resolve(strict=True)
    if file_sha256(source) != _require_sha256(expected_sha256, "fold manifest SHA-256"):
        raise ValueError("fold manifest SHA-256 mismatch")
    frame = pd.read_csv(source, usecols=["patient_id", "fold", "split"])
    if tuple(frame.columns) != ("patient_id", "fold", "split"):
        raise ValueError("fold assignment schema/order drifted")
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    frame["split"] = frame["split"].astype(str)
    if (
        len(frame) != patient_count * 5
        or frame.duplicated(["patient_id", "fold"]).any()
        or frame["patient_id"].nunique() != patient_count
        or set(frame["fold"]) != set(range(5))
        or set(frame["split"]) != set(SPLITS)
        or not frame.groupby("patient_id")["fold"].nunique().eq(5).all()
        or not frame["split"].eq("test").groupby(frame["patient_id"]).sum().eq(1).all()
    ):
        raise ValueError("fold assignment coverage drifted")
    return frame.reset_index(drop=True)


def load_regional_feature_asset(
    path: str | Path,
    folds: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    seed: int,
    arm: str,
    fold: int,
    lock: Mapping[str, Any] | None = None,
    inventory_record: Mapping[str, Any] | None = None,
) -> RegionalFeatureAsset:
    """Load one cell and fail closed on schema, identity, hash, and fold drift."""

    source = Path(path).expanduser().resolve(strict=True)
    expected_tail = (f"seed_{seed}", arm, f"fold_{fold}", FEATURE_FILENAME)
    if tuple(source.parts[-4:]) != expected_tail:
        raise ValueError(f"feature path is not bound to cell {expected_tail}: {source}")
    metadata_path = source.with_suffix(".metadata.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    try:
        with np.load(source, allow_pickle=False) as archive:
            observed = set(archive.files)
            if observed != set(FEATURE_KEYS):
                raise ValueError(
                    f"feature NPZ keys drifted: missing={sorted(FEATURE_KEYS-observed)}, "
                    f"extra={sorted(observed-FEATURE_KEYS)}"
                )
            arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    except OSError as error:
        raise ValueError(f"feature NPZ is unreadable: {source}") from error

    patient_count = int(config["frozen_cells"]["patient_count"])
    patient_id = arrays["patient_id"]
    split = arrays["split"]
    if patient_id.shape != (patient_count,) or patient_id.dtype.kind != "U":
        raise ValueError(f"patient_id must be Unicode [{patient_count}]")
    if split.shape != (patient_count,) or split.dtype.kind != "U" or set(split.astype(str)) != set(SPLITS):
        raise ValueError(f"split must be Unicode [{patient_count}] with train/val/test")
    patient_values = patient_id.astype(str)
    split_values = split.astype(str)
    if len(set(patient_values)) != patient_count or any(not value or value != value.strip() for value in patient_values):
        raise ValueError("feature asset has blank or duplicate patient IDs")
    if dict(zip(patient_values, split_values, strict=True)) != _expected_assignment(folds, fold, patient_count):
        raise ValueError("feature patients/splits differ from the locked outer fold")
    dimensions = config["variants"]["dimensions"]
    feature_arrays: dict[str, np.ndarray] = {}
    for variant in ALL_VARIANTS:
        expected_shape = (patient_count, len(VISITS), int(dimensions[variant]))
        value = arrays[variant]
        if value.dtype != np.dtype("float32") or value.shape != expected_shape or not np.isfinite(value).all():
            raise ValueError(f"{variant} must be finite float32 {expected_shape}")
        feature_arrays[variant] = np.ascontiguousarray(value)
    identity = (
        str(_scalar(arrays["arm"], "arm")),
        int(_scalar(arrays["seed_base"], "seed_base")),
        int(_scalar(arrays["fold"], "fold")),
    )
    if identity != (arm, seed, fold):
        raise ValueError(f"feature scalar identity drifted: {identity}")
    if arrays["arm"].dtype.kind != "U" or arrays["seed_base"].dtype != np.dtype("int64") or arrays["fold"].dtype != np.dtype("int64"):
        raise ValueError("arm must be Unicode and seed_base/fold must be int64 scalars")

    metadata = _load_json(metadata_path, "feature metadata")
    expected_metadata_keys = _metadata_keys()
    if set(metadata) != set(expected_metadata_keys):
        raise ValueError(
            f"feature metadata keys drifted: missing={sorted(expected_metadata_keys-set(metadata))}, "
            f"extra={sorted(set(metadata)-expected_metadata_keys)}"
        )
    cell = f"seed_{seed}/{arm}/fold_{fold}"
    exact_metadata = {
        "status": "COMPLETE", "experiment": "mask_free_region_aware_audit",
        "cell": cell, "arm": arm, "seed_base": seed, "fold": fold,
        "patient_count": patient_count, "encoder_frozen": True,
        "training_performed": False, "streamed_raw_spatial_map_not_persisted": True,
        "phenotype_or_pcr_labels_read": False,
    }
    for name, expected in exact_metadata.items():
        if metadata.get(name) != expected:
            raise ValueError(f"feature metadata differs at {name}: {metadata.get(name)!r}")
    if Path(str(metadata.get("feature_path", ""))).expanduser().resolve() != source:
        raise ValueError("feature metadata path differs from loaded asset")
    digest = file_sha256(source)
    if metadata.get("feature_sha256") != digest:
        raise ValueError("feature metadata SHA-256 differs from loaded asset")
    if metadata.get("patient_order_sha256") != ordered_sha256(patient_values):
        raise ValueError("feature patient-order hash drifted")
    if metadata.get("split_order_sha256") != ordered_sha256(split_values):
        raise ValueError("feature split-order hash drifted")
    if metadata.get("variant_shapes") != {
        name: [patient_count, len(VISITS), int(dimensions[name])] for name in ALL_VARIANTS
    }:
        raise ValueError("feature metadata variant shapes drifted")
    if metadata.get("variant_dtypes") != {name: "float32" for name in ALL_VARIANTS}:
        raise ValueError("feature metadata variant dtypes drifted")
    if metadata.get("config_sha256") != config.get("config_sha256"):
        raise ValueError("feature asset is not bound to the current config")
    if inventory_record is not None:
        expected_inventory = {
            "feature_path": source, "feature_sha256": digest,
            "metadata_path": metadata_path, "metadata_sha256": file_sha256(metadata_path),
        }
        for name, expected in expected_inventory.items():
            observed = inventory_record.get(name)
            if name.endswith("_path"):
                if Path(str(observed)).expanduser().resolve() != expected:
                    raise ValueError(f"completion inventory differs at {name}")
            elif observed != expected:
                raise ValueError(f"completion inventory differs at {name}")
    if lock is not None:
        selected = lock.get("selected_cells", {}).get(cell)
        if not isinstance(selected, Mapping):
            raise ValueError(f"cell is absent from preregistration lock: {cell}")
        for name in ("checkpoint_sha256", "selection_sha256"):
            if metadata.get(name) != selected.get(name):
                raise ValueError(f"feature {name} differs from preregistration")
        lock_digest = file_sha256(LOCK_PATH) if LOCK_PATH.is_file() else None
        if lock_digest is not None and metadata.get("preregistration_lock_sha256") != lock_digest:
            raise ValueError("feature preregistration-lock binding drifted")
    return RegionalFeatureAsset(
        source, metadata_path, patient_values.copy(), split_values.copy(), feature_arrays,
        arm, seed, fold, metadata,
    )


def load_all_regional_feature_assets(
    feature_root: str | Path,
    folds: pd.DataFrame,
    config: Mapping[str, Any],
    lock: Mapping[str, Any],
    completion: Mapping[str, Any],
) -> dict[tuple[int, str, int], RegionalFeatureAsset]:
    """Validate the completion inventory and load all 20 cells before labels."""

    if (
        set(completion) != {
            "schema_version", "status", "experiment", "cell_count", "config_sha256",
            "preregistration_lock_sha256", "geometry_contract_sha256", "cells",
        }
        or completion.get("schema_version") != 1
        or completion.get("status") != "COMPLETE"
        or completion.get("experiment") != config.get("experiment")
        or int(completion.get("cell_count", -1)) != 20
        or completion.get("config_sha256") != config.get("config_sha256")
        or completion.get("preregistration_lock_sha256") != file_sha256(LOCK_PATH)
    ):
        raise ValueError("feature completion inventory schema/provenance drifted")
    records = completion.get("cells")
    if not isinstance(records, list) or len(records) != 20:
        raise ValueError("feature completion inventory must contain a 20-row cell list")
    if any(not isinstance(record, Mapping) for record in records):
        raise ValueError("feature completion inventory contains a non-object cell")
    record_by_key = {str(record.get("cell")): record for record in records}
    expected_keys = {
        f"seed_{seed}/{arm}/fold_{fold}"
        for seed in config["frozen_cells"]["seed_bases"]
        for arm in config["frozen_cells"]["arms"]
        for fold in config["frozen_cells"]["folds"]
    }
    if set(record_by_key) != expected_keys or len(record_by_key) != len(records) or len(expected_keys) != 20:
        raise ValueError("feature completion inventory must contain exactly the locked 20 cells")
    root = Path(feature_root).resolve()
    assets: dict[tuple[int, str, int], RegionalFeatureAsset] = {}
    canonical_ids: set[str] | None = None
    for seed in config["frozen_cells"]["seed_bases"]:
        for arm in config["frozen_cells"]["arms"]:
            for fold in config["frozen_cells"]["folds"]:
                key = f"seed_{seed}/{arm}/fold_{fold}"
                path = root / f"seed_{seed}" / arm / f"fold_{fold}" / FEATURE_FILENAME
                asset = load_regional_feature_asset(
                    path, folds, config, seed=int(seed), arm=str(arm), fold=int(fold),
                    lock=lock, inventory_record=record_by_key[key],
                )
                observed_ids = set(asset.patient_id)
                if canonical_ids is None:
                    canonical_ids = observed_ids
                elif observed_ids != canonical_ids:
                    raise ValueError("feature patient cohort differs across cells")
                assets[(int(seed), str(arm), int(fold))] = asset
    if len(assets) != 20:
        raise AssertionError("formal analysis did not load exactly 20 feature cells")
    return assets


def timing_end_index(view: str) -> int:
    mapping = {"T0": 0, "T0-T1": 1, "T0-T2": 2, "T0-T3": 3}
    if view not in mapping:
        raise ValueError(f"unknown causal timing: {view}")
    return mapping[view]


def causal_prefix(features: np.ndarray, view: str) -> np.ndarray:
    values = np.asarray(features)
    if values.ndim != 3 or values.shape[1] != len(VISITS) or not np.isfinite(values).all():
        raise ValueError("regional features must be finite [N,4,D]")
    end = timing_end_index(view) + 1
    return values[:, :end, :].reshape(len(values), end * values.shape[2]).copy()


def static_visit(features: np.ndarray, visit: str) -> np.ndarray:
    if visit not in VISITS:
        raise ValueError(f"unknown static visit: {visit}")
    values = np.asarray(features)
    if values.ndim != 3 or values.shape[1] != len(VISITS) or not np.isfinite(values).all():
        raise ValueError("regional features must be finite [N,4,D]")
    return values[:, VISITS.index(visit), :].copy()


def _split_indices(split: np.ndarray) -> dict[str, np.ndarray]:
    labels = np.asarray(split).astype(str)
    result = {name: np.flatnonzero(labels == name) for name in SPLITS}
    if any(len(index) == 0 for index in result.values()) or sum(map(len, result.values())) != len(labels):
        raise ValueError("analyzed cohort must contain only nonempty train/val/test partitions")
    return result


class _ExactLegacyMulticlassLiblinear(LogisticRegression):
    """Exact sklearn<=1.7 binary-OvR liblinear behavior under sklearn 1.8."""

    def fit(self, X: Any, y: Any, sample_weight: Any | None = None) -> "_ExactLegacyMulticlassLiblinear":
        if sklearn.__version__ != _EXACT_SKLEARN_VERSION:
            raise RuntimeError("exact multiclass audit requires scikit-learn 1.8.0")
        if (
            self.penalty != "l2" or self.solver != "liblinear" or self.class_weight != "balanced"
            or self.dual is not False or self.fit_intercept is not True
            or float(self.intercept_scaling) != 1.0 or float(self.tol) != 1e-4
            or int(self.verbose) != 0 or self.warm_start is not False
            or self.l1_ratio != 0.0 or self.n_jobs is not None
            or not np.isfinite(float(self.C)) or float(self.C) <= 0.0
            or int(self.max_iter) != 10_000 or self.random_state != 0
            or sample_weight is not None
        ):
            raise ValueError("exact multiclass liblinear contract drifted")
        matrix, labels = validate_data(
            self, X, y, accept_sparse="csr", dtype=[np.float64, np.float32], order="C",
            accept_large_sparse=False,
        )
        check_classification_targets(labels)
        self.classes_ = np.unique(labels)
        if self.classes_.size < 3:
            raise ValueError("exact multiclass liblinear requires at least three classes")
        self.coef_, self.intercept_, self.n_iter_ = _fit_liblinear(
            matrix, labels, self.C, self.fit_intercept, self.intercept_scaling,
            self.class_weight, self.penalty, self.dual, self.verbose, self.max_iter,
            self.tol, self.random_state, multi_class="ovr", loss="logistic_regression",
            sample_weight=None,
        )
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self)
        probability = np.asarray(self.decision_function(X), dtype=np.float64)
        if probability.ndim != 2 or probability.shape[1] != len(self.classes_):
            raise RuntimeError("exact multiclass decision shape drifted")
        expit(probability, out=probability)
        denominator = probability.sum(axis=1, keepdims=True)
        if not np.isfinite(denominator).all() or np.any(denominator <= 0):
            raise RuntimeError("exact multiclass probability normalization failed")
        probability /= denominator
        return probability


def fit_multiclass_logistic_exact(
    train_features: Any,
    train_labels: Any,
    validation_features: Any,
    validation_labels: Any,
    c_grid: Iterable[float],
    *,
    solver: str = "liblinear",
    max_iter: int = 10_000,
    random_state: int = 0,
) -> MulticlassLogisticFit:
    """Fit exact balanced binary-OvR candidates and select smaller C on ties."""

    train_x = _matrix(train_features, name="train features")
    validation_x = _matrix(validation_features, name="validation features", expected_features=train_x.shape[1])
    train_y = _multiclass_labels(train_labels, name="train labels", expected_rows=len(train_x))
    validation_y = _multiclass_labels(validation_labels, name="validation labels", expected_rows=len(validation_x))
    if len(set(train_y.tolist())) < 3 or set(validation_y.tolist()) != set(train_y.tolist()):
        raise ValueError("multiclass train/validation must contain exactly the same >=3 classes")
    if solver != "liblinear":
        raise ValueError("exact multiclass solver must remain liblinear")
    grid = _positive_grid(c_grid, name="c_grid")
    max_iter = _positive_integer(max_iter, name="max_iter")
    scaler = StandardScaler().fit(train_x)
    train_scaled = scaler.transform(train_x)
    validation_scaled = scaler.transform(validation_x)
    scored: list[tuple[MulticlassCandidateScore, _ExactLegacyMulticlassLiblinear]] = []
    for c_value in grid:
        candidate = _ExactLegacyMulticlassLiblinear(
            penalty="l2", C=c_value, solver=solver, class_weight="balanced",
            max_iter=max_iter, random_state=int(random_state), l1_ratio=0.0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            candidate.fit(train_scaled, train_y)
        metrics = multiclass_metrics(validation_y, candidate.predict_proba(validation_scaled), classes=candidate.classes_)
        score = MulticlassCandidateScore(
            c_value=c_value,
            validation_macro_ovr_auroc=float(metrics["macro_ovr_auroc"]),
            validation_macro_ovr_auprc=float(metrics["macro_ovr_auprc"]),
        )
        if not np.isfinite(score.validation_macro_ovr_auroc):
            raise RuntimeError(f"non-finite validation macro AUROC for C={c_value}")
        scored.append((score, candidate))
    best = max(item[0].validation_macro_ovr_auroc for item in scored)
    selected_score, selected_model = min(
        (item for item in scored if item[0].validation_macro_ovr_auroc >= best - CLASSIFICATION_ATOL),
        key=lambda item: item[0].c_value,
    )
    return MulticlassLogisticFit(
        scaler=scaler, model=selected_model, selected_c=selected_score.c_value,
        validation_macro_ovr_auroc=selected_score.validation_macro_ovr_auroc,
        validation_macro_ovr_auprc=selected_score.validation_macro_ovr_auprc,
        grid_scores=tuple(item[0] for item in scored),
        classes=tuple(selected_model.classes_.tolist()), feature_dim=int(train_x.shape[1]),
        train_rows=int(len(train_x)), validation_rows=int(len(validation_x)),
    )


def _plain(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _two_seed_deltas(
    metrics: pd.DataFrame,
    *,
    reference: str,
    candidates: Sequence[str],
    identity: Sequence[str],
    variant_column: str = "variant",
    expected_seeds: Sequence[int] = (2026, 3026),
) -> list[dict[str, Any]]:
    required = {*identity, "seed", variant_column, "auroc"}
    if missing := sorted(required - set(metrics.columns)):
        raise ValueError(f"gate metric table misses columns: {missing}")
    selected = metrics.loc[metrics[variant_column].isin((reference, *candidates))]
    keys = [*identity, "seed", variant_column]
    if selected.duplicated(keys).any():
        raise ValueError("gate metric table repeats a seed/variant identity")
    pivot = selected.pivot(index=[*identity, "seed"], columns=variant_column, values="auroc")
    rows: list[dict[str, Any]] = []
    expected = {int(seed) for seed in expected_seeds}
    for candidate in candidates:
        if reference not in pivot or candidate not in pivot:
            continue
        deltas = (pivot[candidate] - pivot[reference]).rename("delta").reset_index()
        for key, group in deltas.groupby(list(identity), sort=True, dropna=False):
            values = key if isinstance(key, tuple) else (key,)
            by_seed = {int(row.seed): float(row.delta) for row in group.itertuples(index=False) if np.isfinite(row.delta)}
            if set(by_seed) != expected:
                continue
            rows.append({
                **{name: _plain(value) for name, value in zip(identity, values, strict=True)},
                "reference": reference, "candidate": candidate,
                "seed_deltas": {str(seed): by_seed[seed] for seed in sorted(by_seed)},
                "mean_delta": float(np.mean(list(by_seed.values()))),
                "both_strictly_positive": all(value > 0 for value in by_seed.values()),
            })
    return rows


def evaluate_gates(
    config: Mapping[str, Any],
    mri_metrics: pd.DataFrame,
    incremental_metrics: pd.DataFrame,
    phenotype_metrics: pd.DataFrame,
    oracle_recovery: pd.DataFrame,
) -> dict[str, Any]:
    """Apply Gates A--D and the exact preregistered classification precedence."""

    seeds = tuple(int(value) for value in config["frozen_cells"]["seed_bases"])
    if seeds != (2026, 3026):
        raise ValueError("gate evaluation requires the locked two seed bases")
    gate_cfg = config["gates"]

    selected_a = mri_metrics.loc[
        mri_metrics["analysis"].eq("mri_only_pcr")
        & mri_metrics["view"].isin(gate_cfg["A"]["timings"])
        & mri_metrics["target"].eq("pCR")
    ]
    evidence_a = _two_seed_deltas(
        selected_a, reference="R0", candidates=tuple(gate_cfg["A"]["candidates"]),
        identity=("arm", "view", "target", "population"), expected_seeds=seeds,
    )
    threshold_a = float(gate_cfg["A"]["minimum_two_seed_mean_gain"])
    for row in evidence_a:
        row["passed"] = bool(row["both_strictly_positive"] and row["mean_delta"] >= threshold_a)
    support_a = [row for row in evidence_a if row["passed"]]
    gate_a = bool(support_a)

    selected_b = incremental_metrics.loc[
        incremental_metrics["analysis"].eq("clinical_ftv_pcr")
        & incremental_metrics["view"].isin(gate_cfg["B"]["timings"])
        & incremental_metrics["target"].eq("pCR")
    ]
    b_keys = ["arm", "view", "target", "population", "seed"]
    if selected_b.duplicated([*b_keys, "model"]).any():
        raise ValueError("Gate-B table repeats a model identity")
    b_pivot = selected_b.pivot(index=b_keys, columns="model", values="auroc")
    evidence_b: list[dict[str, Any]] = []
    for candidate in gate_cfg["B"]["candidates"]:
        candidate_model = f"C+F+{candidate}"
        if not {"C+F", "C+F+R0", candidate_model}.issubset(b_pivot.columns):
            continue
        values = b_pivot.reset_index()
        values["delta_vs_cf_r0"] = values[candidate_model] - values["C+F+R0"]
        values["delta_vs_cf"] = values[candidate_model] - values["C+F"]
        for key, group in values.groupby(b_keys[:-1], sort=True, dropna=False):
            by_seed = {
                int(row.seed): {"vs_cf_r0": float(row.delta_vs_cf_r0), "vs_cf": float(row.delta_vs_cf)}
                for row in group.itertuples(index=False)
                if np.isfinite(row.delta_vs_cf_r0) and np.isfinite(row.delta_vs_cf)
            }
            if set(by_seed) != set(seeds):
                continue
            mean_vs_cf = float(np.mean([value["vs_cf"] for value in by_seed.values()]))
            both_negative_vs_cf = all(value["vs_cf"] < 0 for value in by_seed.values())
            passed = bool(
                all(value["vs_cf_r0"] > 0 for value in by_seed.values())
                and mean_vs_cf >= float(gate_cfg["B"]["minimum_two_seed_mean_vs_cf"])
                and not both_negative_vs_cf
            )
            evidence_b.append({
                **dict(zip(b_keys[:-1], (_plain(value) for value in key), strict=True)),
                "candidate": candidate,
                "seed_deltas": {str(seed): by_seed[seed] for seed in sorted(by_seed)},
                "mean_delta_vs_cf": mean_vs_cf, "both_seeds_negative_vs_cf": both_negative_vs_cf,
                "passed": passed,
            })
    support_b = [row for row in evidence_b if row["passed"]]
    gate_b = bool(support_b)

    c_required = {
        "seed", "arm", "view", "candidate", "numerator_auroc_uplift", "recovery_ratio",
        "published_oracle_uplift", "recovery_defined",
    }
    if missing := sorted(c_required - set(oracle_recovery.columns)):
        raise ValueError(f"Oracle recovery table misses columns: {missing}")
    selected_c = oracle_recovery.loc[
        oracle_recovery["arm"].eq(config["oracle"]["primary_arm"])
        & oracle_recovery["view"].eq(config["oracle"]["view"])
        & oracle_recovery["candidate"].isin(gate_cfg["C"]["candidates"])
    ]
    if selected_c.duplicated(["seed", "arm", "view", "candidate"]).any():
        raise ValueError("Gate-C table repeats a recovery identity")
    evidence_c: list[dict[str, Any]] = []
    for candidate, group in selected_c.groupby("candidate", sort=True):
        by_seed = {
            int(row.seed): {
                "numerator_auroc_uplift": float(row.numerator_auroc_uplift),
                "recovery_ratio": float(row.recovery_ratio),
            }
            for row in group.itertuples(index=False)
            if np.isfinite(row.numerator_auroc_uplift)
            and np.isfinite(row.recovery_ratio)
            and np.isfinite(row.published_oracle_uplift)
            and float(row.published_oracle_uplift) > 0
            and bool(row.recovery_defined)
        }
        if set(by_seed) != set(seeds):
            continue
        mean_ratio = float(np.mean([value["recovery_ratio"] for value in by_seed.values()]))
        passed = bool(
            all(value["numerator_auroc_uplift"] > 0 for value in by_seed.values())
            and mean_ratio >= float(gate_cfg["C"]["minimum_two_seed_mean_recovery_ratio"])
        )
        evidence_c.append({
            "arm": config["oracle"]["primary_arm"], "view": config["oracle"]["view"],
            "candidate": str(candidate),
            "seed_values": {str(seed): by_seed[seed] for seed in sorted(by_seed)},
            "mean_recovery_ratio": mean_ratio, "passed": passed,
        })
    support_c = [row for row in evidence_c if row["passed"]]
    gate_c = bool(support_c)

    selected_d = phenotype_metrics.loc[
        phenotype_metrics["analysis"].eq("phenotype")
        & phenotype_metrics["target"].isin(PHENOTYPE_TARGETS)
    ]
    evidence_d = _two_seed_deltas(
        selected_d, reference="R0", candidates=tuple(gate_cfg["D"]["candidates"]),
        identity=("arm", "view", "target", "population"), expected_seeds=seeds,
    )
    threshold_d = float(gate_cfg["D"]["minimum_auroc_gain_each_seed"])
    for row in evidence_d:
        row["passed"] = all(value >= threshold_d for value in row["seed_deltas"].values())
    support_d = [row for row in evidence_d if row["passed"]]
    gate_d = bool(support_d)

    any_positive = any(row["both_strictly_positive"] for row in evidence_a)
    classification = scientific_classification(
        gate_a=gate_a, gate_b=gate_b, gate_c=gate_c, any_two_seed_positive=any_positive
    )
    return {
        "schema_version": 1, "experiment": "mask_free_region_aware_audit", "status": "COMPLETE",
        "gates": {
            "A": {"name": "MASK_FREE_REGIONAL_SIGNAL_SUPPORTED", "passed": gate_a,
                  "evaluated_comparisons": evidence_a, "supporting_comparisons": support_a},
            "B": {"name": "MASK_FREE_BEYOND_FTV_SUPPORTED", "passed": gate_b,
                  "evaluated_comparisons": evidence_b, "supporting_comparisons": support_b},
            "C": {"name": "ORACLE_SIGNAL_PARTIALLY_RECOVERED", "passed": gate_c,
                  "evaluated_comparisons": evidence_c, "supporting_comparisons": support_c},
            "D": {"name": "PROFILE_ASSOCIATED_REGIONAL_SIGNAL_SUPPORTED", "passed": gate_d,
                  "evaluated_comparisons": evidence_d, "supporting_comparisons": support_d},
        },
        "any_primary_candidate_two_seed_positive": any_positive,
        "scientific_classification": classification,
        "classification_precedence": ["A_if_Gate_A_and_C", "B_if_Gate_A_and_not_B",
                                      "C_if_Gate_C_fails_and_any_two_seed_positive",
                                      "D_if_Gate_A_fails_and_no_two_seed_positive", "INDETERMINATE"],
        "contains_patient_level_data": False,
    }


def scientific_classification(
    *, gate_a: bool, gate_b: bool, gate_c: bool, any_two_seed_positive: bool
) -> str:
    """Apply the frozen A/B/C/D precedence without post-hoc fall-through."""

    if gate_a and gate_c:
        return "DEPLOYABLE_REGION_AWARE_SIGNAL_SUPPORTED"
    if gate_a and not gate_b:
        return "REGION_SIGNAL_EXISTS_BUT_NOT_BEYOND_FTV"
    if not gate_c and any_two_seed_positive:
        return "ORACLE_REQUIRES_LESION_RELATIVE_LOCALIZATION"
    if not gate_a and not any_two_seed_positive:
        return "MASK_FREE_REGIONALIZATION_NOT_SUPPORTED"
    return "INDETERMINATE_DIAGNOSTIC"


@dataclass
class _TestProbabilityGuard:
    calls: int = 0

    def binary(self, fit: Any, matrix: np.ndarray) -> np.ndarray:
        if self.calls:
            raise RuntimeError("outer-test probability prediction is single-use")
        self.calls += 1
        return np.asarray(fit.predict_proba(matrix), dtype=np.float64)

    def multiclass(self, fit: MulticlassLogisticFit, matrix: np.ndarray) -> np.ndarray:
        if self.calls:
            raise RuntimeError("outer-test probability prediction is single-use")
        self.calls += 1
        return np.asarray(fit.predict_proba(matrix), dtype=np.float64)


def _aligned(frame: pd.DataFrame, patient_ids: Sequence[str], label: str) -> pd.DataFrame:
    if "patient_id" not in frame:
        raise ValueError(f"{label} has no patient_id column")
    indexed = frame.set_index("patient_id", verify_integrity=True)
    requested = [str(value) for value in patient_ids]
    missing = sorted(set(requested) - set(indexed.index.astype(str)))
    if missing:
        raise ValueError(f"{label} misses feature patients: {missing[:5]}")
    indexed.index = indexed.index.astype(str)
    return indexed.loc[requested].reset_index()


def _clinical_matrix(
    config: Mapping[str, Any],
    clinical: pd.DataFrame,
    indices: Mapping[str, np.ndarray],
) -> np.ndarray:
    fields = tuple(str(value) for value in config["analysis"]["clinical_fields"])
    contract = str(config["analysis"]["clinical_contract"])
    if contract != "C2_full_with_treatment" or fields != (
        "label_hr", "label_her2", "label_mp", "age_at_screening", "race_simple",
        "menopausal_status_simple", "ethnicity", "arm",
    ):
        raise ValueError("clinical C2_full_with_treatment contract drifted")
    encoder = TrainOnlyClinicalEncoder(fields).fit(clinical.iloc[indices["train"]])
    return np.asarray(encoder.transform(clinical), dtype=np.float64)


def _population_mask(asset: RegionalFeatureAsset, ftv_ids: set[str], population: str) -> np.ndarray:
    if population == "full_808":
        return np.ones(len(asset.patient_id), dtype=bool)
    if population == "ftv_complete_375":
        return np.asarray([patient_id in ftv_ids for patient_id in asset.patient_id], dtype=bool)
    raise ValueError(f"unknown pCR population: {population}")


def _base_prediction(metadata: Mapping[str, Any], patient_id: str, fold: int) -> dict[str, Any]:
    return {
        "patient_id": patient_id,
        "fold": int(fold),
        "population": str(metadata["population"]),
        "seed": int(metadata["seed"]),
        "arm": str(metadata["arm"]),
        "analysis": str(metadata["analysis"]),
        "context": str(metadata["context"]),
        "view": str(metadata["view"]),
        "target": str(metadata["target"]),
        "variant": str(metadata["variant"]),
        "model": str(metadata["model"]),
        "clinical_contract": str(metadata.get("clinical_contract", "")),
    }


def _fit_binary(
    matrix: np.ndarray,
    labels: np.ndarray,
    indices: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    *,
    class_weight: str | Mapping[int, float] | None,
) -> Any:
    logistic = config["logistic"]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=_SKLEARN_PENALTY_WARNING, category=FutureWarning,
            module=r"^sklearn\.linear_model\._logistic$",
        )
        return fit_binary_logistic(
            matrix[indices["train"]], labels[indices["train"]],
            matrix[indices["val"]], labels[indices["val"]],
            logistic["c_grid"], class_weight=class_weight,
            solver=str(logistic["solver"]), max_iter=int(logistic["max_iter"]),
            random_state=0,
        )


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
    class_weight: str | Mapping[int, float] | None,
    metadata: Mapping[str, Any],
) -> None:
    fit = _fit_binary(matrix, labels, indices, config, class_weight=class_weight)
    test = indices["test"]
    guard = _TestProbabilityGuard()
    probability = guard.binary(fit, matrix[test])
    if guard.calls != 1:
        raise AssertionError("binary outer-test probability was not predicted exactly once")
    prediction = (probability >= fit.threshold_selection.threshold).astype(np.int64)
    for offset, row_index in enumerate(test):
        row = _base_prediction(metadata, str(patient_ids[row_index]), fold)
        row.update({
            "y_true": int(labels[row_index]),
            "predicted_probability": float(probability[offset]),
            "predicted_label": int(prediction[offset]),
            "threshold": float(fit.threshold_selection.threshold),
        })
        prediction_rows.append(row)
    hyperparameter_rows.append({
        **{name: metadata.get(name, "") for name in HYPERPARAMETER_COLUMNS[:11]},
        "fold": int(fold), "selected_c": float(fit.selected_c),
        "validation_auroc": float(fit.validation_auroc), "validation_auprc": math.nan,
        "threshold": float(fit.threshold_selection.threshold),
        "train_rows": int(fit.train_rows), "validation_rows": int(fit.validation_rows),
        "test_rows": int(len(test)), "feature_dim": int(fit.feature_dim),
        "class_weight": "none" if class_weight is None else str(class_weight),
        "c_grid": json.dumps([float(value) for value in config["logistic"]["c_grid"]], separators=(",", ":")),
        "test_used_for_selection": False, "test_predict_call_count": guard.calls,
    })


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
    metadata: Mapping[str, Any],
) -> None:
    train, validation, test = (indices[name] for name in SPLITS)
    logistic = config["logistic"]
    fit = fit_multiclass_logistic_exact(
        matrix[train], labels[train], matrix[validation], labels[validation],
        logistic["c_grid"], solver=str(logistic["solver"]),
        max_iter=int(logistic["max_iter"]), random_state=0,
    )
    if set(str(value) for value in fit.classes) != set(SUBTYPE_CLASSES):
        raise ValueError("subtype class contract drifted")
    guard = _TestProbabilityGuard()
    probability = guard.multiclass(fit, matrix[test])
    if guard.calls != 1:
        raise AssertionError("multiclass outer-test probability was not predicted exactly once")
    classes = np.asarray(fit.classes)
    predicted = classes[np.argmax(probability, axis=1)]
    class_lookup = {str(value): index for index, value in enumerate(fit.classes)}
    for offset, row_index in enumerate(test):
        row = _base_prediction(metadata, str(patient_ids[row_index]), fold)
        row.update({
            "y_true": str(labels[row_index]), "predicted_probability": math.nan,
            "predicted_label": str(predicted[offset]), "threshold": math.nan,
        })
        for output_index, subtype in enumerate(SUBTYPE_CLASSES):
            row[SUBTYPE_PROBABILITY_COLUMNS[output_index]] = float(probability[offset, class_lookup[subtype]])
            row[SUBTYPE_LABEL_COLUMNS[output_index]] = subtype
        prediction_rows.append(row)
    hyperparameter_rows.append({
        **{name: metadata.get(name, "") for name in HYPERPARAMETER_COLUMNS[:11]},
        "fold": int(fold), "selected_c": float(fit.selected_c),
        "validation_auroc": float(fit.validation_macro_ovr_auroc),
        "validation_auprc": float(fit.validation_macro_ovr_auprc), "threshold": math.nan,
        "train_rows": int(fit.train_rows), "validation_rows": int(fit.validation_rows),
        "test_rows": int(len(test)), "feature_dim": int(fit.feature_dim),
        "class_weight": "balanced",
        "c_grid": json.dumps([float(value) for value in logistic["c_grid"]], separators=(",", ":")),
        "test_used_for_selection": False, "test_predict_call_count": guard.calls,
    })


def _prediction_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        raise ValueError("analysis produced no held-out predictions")
    return pd.DataFrame(rows).reindex(columns=PREDICTION_COLUMNS)


def aggregate_classification_oof(predictions: pd.DataFrame) -> pd.DataFrame:
    """Pool fixed held-out folds into patient-level OOF classification metrics."""

    group_columns = [
        "seed", "arm", "analysis", "context", "view", "target", "variant", "model",
        "population", "clinical_contract",
    ]
    if missing := sorted({"patient_id", "fold", *group_columns} - set(predictions.columns)):
        raise ValueError(f"classification predictions miss columns: {missing}")
    rows: list[dict[str, Any]] = []
    for key, group in predictions.groupby(group_columns, sort=True, dropna=False):
        if group["patient_id"].astype(str).duplicated().any():
            raise ValueError(f"OOF group repeats a patient: {key}")
        if set(group["fold"].astype(int)) != set(range(5)):
            raise ValueError(f"OOF group does not cover all five test folds: {key}")
        target = str(key[5])
        if target == "subtype_4class":
            probability = group.loc[:, SUBTYPE_PROBABILITY_COLUMNS].to_numpy(dtype=float)
            metrics = multiclass_metrics(
                group["y_true"].astype(str).to_numpy(), probability, classes=SUBTYPE_CLASSES,
            )
            values = {
                "n": int(metrics["n"]), "n_positive": math.nan, "n_negative": math.nan,
                "n_classes": 4, "auroc": float(metrics["macro_ovr_auroc"]),
                "auprc": float(metrics["macro_ovr_auprc"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]), "brier": math.nan,
            }
        else:
            labels = group["y_true"].to_numpy(dtype=np.int64)
            probability = group["predicted_probability"].to_numpy(dtype=float)
            prediction = group["predicted_label"].to_numpy(dtype=np.int64)
            if set(np.unique(labels)) != {0, 1} or not np.isfinite(probability).all():
                raise ValueError(f"binary OOF group is invalid: {key}")
            values = {
                "n": int(len(group)), "n_positive": int(labels.sum()),
                "n_negative": int(np.sum(labels == 0)), "n_classes": 2,
                "auroc": float(roc_auc_score(labels, probability)),
                "auprc": float(average_precision_score(labels, probability)),
                "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
                "brier": float(np.mean(np.square(probability - labels))),
            }
        row = {**dict(zip(group_columns, key, strict=True)), **values}
        row["timing_label"] = "late/pre-surgery" if row["view"] == "T0-T3" else ""
        rows.append(row)
    return pd.DataFrame(rows).reindex(columns=CLASSIFICATION_METRIC_COLUMNS)


def _phenotype_labels(clinical: pd.DataFrame, target: str) -> np.ndarray:
    if target == "HR":
        return clinical["label_hr"].to_numpy(dtype=np.int64)
    if target == "HER2":
        return clinical["label_her2"].to_numpy(dtype=np.int64)
    if target == "subtype_4class":
        return clinical["hr_her2_subtype"].astype(str).to_numpy()
    raise ValueError(f"unknown phenotype target: {target}")


def run_mri_only_pcr(
    config: Mapping[str, Any],
    clinical: pd.DataFrame,
    ftv_ids: set[str],
    assets: Mapping[tuple[int, str, int], RegionalFeatureAsset],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    hyperparameters: list[dict[str, Any]] = []
    for (seed, arm, fold), asset in sorted(assets.items()):
        for population in config["analysis"]["pcr_populations"]:
            mask = _population_mask(asset, ftv_ids, str(population))
            patient_ids, split = asset.patient_id[mask], asset.split[mask]
            aligned = _aligned(clinical, patient_ids, "clinical table")
            labels = aligned["label_pcr"].to_numpy(dtype=np.int64)
            indices = _split_indices(split)
            for view in config["analysis"]["pcr_timings"]:
                for variant in config["analysis"]["all_probe_variants"]:
                    metadata = {
                        "analysis": "mri_only_pcr", "context": "MRI_ONLY",
                        "population": str(population), "seed": seed, "arm": arm,
                        "view": str(view), "target": "pCR", "variant": str(variant),
                        "model": str(variant), "clinical_contract": "",
                    }
                    _append_binary_fit(
                        predictions, hyperparameters, patient_ids=patient_ids, fold=fold,
                        labels=labels, matrix=causal_prefix(asset.variant(str(variant))[mask], str(view)),
                        indices=indices, config=config, class_weight=None, metadata=metadata,
                    )
    frame = _prediction_frame(predictions)
    return frame, aggregate_classification_oof(frame), hyperparameters


def run_clinical_pcr(
    config: Mapping[str, Any],
    clinical: pd.DataFrame,
    ftv_ids: set[str],
    assets: Mapping[tuple[int, str, int], RegionalFeatureAsset],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    hyperparameters: list[dict[str, Any]] = []
    contract = str(config["analysis"]["clinical_contract"])
    for (seed, arm, fold), asset in sorted(assets.items()):
        mask = _population_mask(asset, ftv_ids, "ftv_complete_375")
        patient_ids, split = asset.patient_id[mask], asset.split[mask]
        aligned = _aligned(clinical, patient_ids, "clinical table")
        labels = aligned["label_pcr"].to_numpy(dtype=np.int64)
        indices = _split_indices(split)
        clinical_matrix = _clinical_matrix(config, aligned, indices)
        for view in config["analysis"]["pcr_timings"]:
            matrices = {"C": clinical_matrix}
            for variant in config["analysis"]["all_probe_variants"]:
                matrices[f"C+{variant}"] = np.concatenate(
                    (clinical_matrix, causal_prefix(asset.variant(str(variant))[mask], str(view))), axis=1
                )
            for model, matrix in matrices.items():
                variant = "NONE" if model == "C" else model.removeprefix("C+")
                metadata = {
                    "analysis": "clinical_pcr", "context": "C_PLUS_R",
                    "population": "ftv_complete_375", "seed": seed, "arm": arm,
                    "view": str(view), "target": "pCR", "variant": variant,
                    "model": model, "clinical_contract": contract,
                }
                _append_binary_fit(
                    predictions, hyperparameters, patient_ids=patient_ids, fold=fold,
                    labels=labels, matrix=matrix, indices=indices, config=config,
                    class_weight=None, metadata=metadata,
                )
    frame = _prediction_frame(predictions)
    return frame, aggregate_classification_oof(frame), hyperparameters


def run_clinical_ftv_pcr(
    config: Mapping[str, Any],
    clinical: pd.DataFrame,
    ftv: pd.DataFrame,
    assets: Mapping[tuple[int, str, int], RegionalFeatureAsset],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    hyperparameters: list[dict[str, Any]] = []
    contract = str(config["analysis"]["clinical_contract"])
    ftv_ids = set(ftv["patient_id"].astype(str))
    for (seed, arm, fold), asset in sorted(assets.items()):
        mask = _population_mask(asset, ftv_ids, "ftv_complete_375")
        patient_ids, split = asset.patient_id[mask], asset.split[mask]
        aligned = _aligned(clinical, patient_ids, "clinical table")
        aligned_ftv = _aligned(ftv, patient_ids, "FTV table")
        labels = aligned["label_pcr"].to_numpy(dtype=np.int64)
        indices = _split_indices(split)
        clinical_matrix = _clinical_matrix(config, aligned, indices)
        for view in config["analysis"]["pcr_timings"]:
            ftv_matrix = np.asarray(ftv_timing_prefix(aligned_ftv, timing_end_index(str(view))), dtype=np.float64)
            base = np.concatenate((clinical_matrix, ftv_matrix), axis=1)
            matrices = {"C+F": base}
            for variant in config["analysis"]["all_probe_variants"]:
                matrices[f"C+F+{variant}"] = np.concatenate(
                    (base, causal_prefix(asset.variant(str(variant))[mask], str(view))), axis=1
                )
            for model, matrix in matrices.items():
                variant = "NONE" if model == "C+F" else model.removeprefix("C+F+")
                metadata = {
                    "analysis": "clinical_ftv_pcr", "context": "C_PLUS_F",
                    "population": "ftv_complete_375", "seed": seed, "arm": arm,
                    "view": str(view), "target": "pCR", "variant": variant,
                    "model": model, "clinical_contract": contract,
                }
                _append_binary_fit(
                    predictions, hyperparameters, patient_ids=patient_ids, fold=fold,
                    labels=labels, matrix=matrix, indices=indices, config=config,
                    class_weight=None, metadata=metadata,
                )
    frame = _prediction_frame(predictions)
    return frame, aggregate_classification_oof(frame), hyperparameters


def run_phenotype_probes(
    config: Mapping[str, Any],
    clinical: pd.DataFrame,
    assets: Mapping[tuple[int, str, int], RegionalFeatureAsset],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    hyperparameters: list[dict[str, Any]] = []
    class_weight = str(config["logistic"]["phenotype_class_weight"])
    for (seed, arm, fold), asset in sorted(assets.items()):
        aligned = _aligned(clinical, asset.patient_id, "clinical table")
        indices = _split_indices(asset.split)
        for view in config["analysis"]["phenotype_views"]:
            for variant in config["analysis"]["all_probe_variants"]:
                matrix = static_visit(asset.variant(str(variant)), str(view))
                for target in config["analysis"]["phenotype_targets"]:
                    labels = _phenotype_labels(aligned, str(target))
                    metadata = {
                        "analysis": "phenotype", "context": "MRI_ONLY_STATIC",
                        "population": "full_808", "seed": seed, "arm": arm,
                        "view": str(view), "target": str(target), "variant": str(variant),
                        "model": str(variant), "clinical_contract": "",
                    }
                    if target == "subtype_4class":
                        _append_multiclass_fit(
                            predictions, hyperparameters, patient_ids=asset.patient_id, fold=fold,
                            labels=labels, matrix=matrix, indices=indices, config=config, metadata=metadata,
                        )
                    else:
                        _append_binary_fit(
                            predictions, hyperparameters, patient_ids=asset.patient_id, fold=fold,
                            labels=labels, matrix=matrix, indices=indices, config=config,
                            class_weight=class_weight, metadata=metadata,
                        )
    frame = _prediction_frame(predictions)
    return frame, aggregate_classification_oof(frame), hyperparameters


def add_reference_deltas(
    metrics: pd.DataFrame,
    *,
    reference_model: str,
    secondary_reference_model: str | None = None,
) -> pd.DataFrame:
    """Add exact within-cell comparison-minus-reference metric deltas."""

    group = ["seed", "arm", "analysis", "context", "view", "target", "population", "clinical_contract"]
    output = metrics.copy()
    references = [reference_model] + ([] if secondary_reference_model is None else [secondary_reference_model])
    for reference in references:
        suffix = reference.replace("+", "_PLUS_").replace("-", "_")
        selected = metrics.loc[metrics["model"].eq(reference), [*group, "auroc", "auprc", "brier"]].rename(
            columns={"auroc": f"reference_auroc__{suffix}", "auprc": f"reference_auprc__{suffix}",
                     "brier": f"reference_brier__{suffix}"}
        )
        output = output.merge(selected, on=group, how="left", validate="many_to_one")
        output[f"delta_auroc_vs_{reference}"] = output["auroc"] - output[f"reference_auroc__{suffix}"]
        output[f"delta_auprc_vs_{reference}"] = output["auprc"] - output[f"reference_auprc__{suffix}"]
        output[f"brier_improvement_vs_{reference}"] = output[f"reference_brier__{suffix}"] - output["brier"]
        output = output.drop(columns=[f"reference_auroc__{suffix}", f"reference_auprc__{suffix}", f"reference_brier__{suffix}"])
    return output


def load_ftv_records(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Load measurement validity/observability through the SHA-pinned Stage-B API."""

    from c1b_stage_b.data import combine_ftv_observability, read_observability, read_raw_ftv
    from c1b_stage_b.inputs import StageBDataPaths

    paths = StageBDataPaths.load(
        config["paths"]["stage_b_data_contract"],
        config["paths"]["stage_b_data_contract_sha256"],
    )
    configured_ftv = Path(config["paths"]["ftv_table"]).resolve()
    if paths.ftv_transition_table.resolve() != configured_ftv:
        raise ValueError("Stage-B and audit FTV paths differ")
    if paths.ftv_transition_table_sha256 != config["paths"]["ftv_table_sha256"]:
        raise ValueError("Stage-B and audit FTV hashes differ")
    raw = read_raw_ftv(paths.ftv_transition_table, paths.ftv_transition_table_sha256)
    observable = read_observability(paths.observability_manifest, paths.observability_manifest_sha256)
    return combine_ftv_observability(raw, observable)


@dataclass(frozen=True)
class _PreparedContinuous:
    patient_ids: tuple[str, ...]
    matrix: np.ndarray
    target: np.ndarray


def _prepare_ftv_split(
    asset: RegionalFeatureAsset,
    variant: str,
    records: Mapping[str, Any],
    *,
    split: str,
    task: str,
    index: int,
    observable_only: bool,
) -> _PreparedContinuous:
    from c1b_stage_b.targets import literal_delta_targets, static_targets

    patient_ids: list[str] = []
    matrices: list[np.ndarray] = []
    targets: list[float] = []
    state = asset.variant(variant)
    for row_index in np.flatnonzero(asset.split == split):
        patient_id = str(asset.patient_id[row_index])
        record = records.get(patient_id)
        if record is None:
            continue
        if task == "static":
            values, valid = static_targets(record, observable_only=observable_only)
            feature = state[row_index, index]
        elif task == "delta":
            values, valid = literal_delta_targets(
                record.values,
                record.measurement_valid,
                record.observable if observable_only else None,
            )
            feature = state[row_index, index + 1] - state[row_index, index]
        else:
            raise ValueError("FTV task must be static or delta")
        if not bool(np.asarray(valid, dtype=bool)[index]):
            continue
        if feature.shape != (state.shape[2],) or not np.isfinite(feature).all():
            raise ValueError("selected regional FTV feature is invalid")
        patient_ids.append(patient_id)
        matrices.append(feature)
        targets.append(float(np.asarray(values, dtype=np.float64)[index]))
    if not matrices:
        raise ValueError(f"no valid {split} rows for FTV {task}/{index}")
    matrix = np.stack(matrices).astype(np.float64, copy=False)
    target = np.asarray(targets, dtype=np.float64)
    if not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise ValueError("prepared FTV split is non-finite")
    return _PreparedContinuous(tuple(patient_ids), matrix, target)


def _safe_correlation(function: Any, truth: np.ndarray, prediction: np.ndarray) -> float:
    if len(truth) < 2 or np.ptp(truth) == 0 or np.ptp(prediction) == 0:
        return math.nan
    value = float(function(truth, prediction).statistic)
    return value if np.isfinite(value) else math.nan


def _continuous_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    baseline: float | np.ndarray,
) -> dict[str, float]:
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    baseline_values = np.broadcast_to(np.asarray(baseline, dtype=np.float64), truth.shape)
    if len(truth) < 2 or not all(np.isfinite(value).all() for value in (truth, prediction, baseline_values)):
        raise ValueError("continuous metric inputs are too small or non-finite")
    rmse = float(math.sqrt(mean_squared_error(truth, prediction)))
    baseline_rmse = float(math.sqrt(mean_squared_error(truth, baseline_values)))
    variance = float(np.var(truth, ddof=0))
    predicted_variance = float(np.var(prediction, ddof=0))
    if variance > 0:
        covariance = float(np.mean((truth - truth.mean()) * (prediction - prediction.mean())))
        slope = covariance / variance
        intercept = float(prediction.mean() - slope * truth.mean())
    else:
        slope = intercept = math.nan
    return {
        "spearman": _safe_correlation(spearmanr, truth, prediction),
        "pearson": _safe_correlation(pearsonr, truth, prediction),
        "r2": float(r2_score(truth, prediction)),
        "rmse": rmse,
        "mae": float(mean_absolute_error(truth, prediction)),
        "b0_rmse": baseline_rmse,
        "rmse_gain_over_b0": (baseline_rmse - rmse) / baseline_rmse if baseline_rmse > 0 else math.nan,
        "prediction_target_variance_ratio": predicted_variance / variance if variance > 0 else math.nan,
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "calibration_mean_bias": float(np.mean(prediction - truth)),
    }


def _run_ftv_endpoint(
    config: Mapping[str, Any],
    asset: RegionalFeatureAsset,
    variant: str,
    records: Mapping[str, Any],
    *,
    task: str,
    index: int,
    observable_only: bool,
    static_transform: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Use the immutable Stage-B Ridge/target primitives for one endpoint."""

    from c1b_stage_b.probes import TestPredictGuard, select_ridge

    prepare = lambda name: _prepare_ftv_split(  # noqa: E731
        asset, variant, records, split=name, task=task, index=index,
        observable_only=observable_only,
    )
    train = prepare("train")
    validation = prepare("val")
    if task == "static":
        train_analysis, train_valid = static_transform.transform_values(
            train.target, np.ones(train.target.shape, dtype=bool)
        )
        validation_analysis, validation_valid = static_transform.transform_values(
            validation.target, np.ones(validation.target.shape, dtype=bool)
        )
        if not train_valid.all() or not validation_valid.all():
            raise AssertionError("valid static FTV became invalid during frozen transform")
        selected = select_ridge(
            train.matrix, train_analysis, validation.matrix, validation_analysis,
            config["ridge"]["alpha_grid"], standardize_target=False,
        )
        analysis_scale = "transformed_outer_train"
        transform_payload = static_transform.to_dict()
    else:
        selected = select_ridge(
            train.matrix, train.target, validation.matrix, validation.target,
            config["ridge"]["alpha_grid"], standardize_target=True,
        )
        if selected.y_scaler is None:
            raise AssertionError("delta FTV lost its outer-train target scaler")
        train_analysis = selected.y_scaler.transform(train.target[:, None]).reshape(-1)
        analysis_scale = "standardized_outer_train"
        transform_payload = {
            "value_transform": "literal_natural_delta",
            "standardization": "outer_train_standard_scaler",
            "train_rows": int(selected.y_scaler.n_samples_seen_),
        }

    # Test is constructed only after every fit/selection operation above.
    test = prepare("test")
    guard = TestPredictGuard()
    predicted_analysis = guard.predict(selected.model, selected.x_scaler.transform(test.matrix))
    if guard.calls != 1:
        raise AssertionError("Ridge outer test was not predicted exactly once")
    if task == "static":
        truth_analysis, truth_valid = static_transform.transform_values(
            test.target, np.ones(test.target.shape, dtype=bool)
        )
        if not truth_valid.all():
            raise AssertionError("valid static test FTV became invalid during transform")
        predicted_natural = np.asarray(static_transform.inverse(predicted_analysis), dtype=np.float64)
    else:
        assert selected.y_scaler is not None
        truth_analysis = selected.y_scaler.transform(test.target[:, None]).reshape(-1)
        predicted_natural = selected.y_scaler.inverse_transform(predicted_analysis[:, None]).reshape(-1)
    endpoint = VISITS[index] if task == "static" else TRANSITIONS[index]
    scope = "observable_only" if observable_only else "primary_measurement_valid"
    common = {
        "seed": asset.seed, "arm": asset.arm, "fold": asset.fold, "variant": variant,
        "feature_dim": int(asset.variant(variant).shape[2]), "task": task, "endpoint": endpoint,
        "view": endpoint, "target": "FTV", "analysis_scope": scope,
        "target_semantics": (
            "static_ftv_log_winsor_median_iqr_inverse_natural"
            if task == "static" else "literal_ftv_end_minus_ftv_start"
        ),
        "selected_alpha": float(selected.alpha), "n_train": len(train.patient_ids),
        "n_val": len(validation.patient_ids), "n_test": len(test.patient_ids),
    }
    natural_baseline = float(np.mean(train.target))
    predictions = [
        {
            "patient_id": patient_id, **common, "split": "test",
            "y_true": float(test.target[row]), "y_pred": float(predicted_natural[row]),
            "y_true_analysis": float(truth_analysis[row]), "y_pred_analysis": float(predicted_analysis[row]),
            "b0_prediction": natural_baseline, "analysis_scale": analysis_scale,
            "test_predict_call_count": guard.calls,
        }
        for row, patient_id in enumerate(test.patient_ids)
    ]
    selection = {
        "analysis": "ftv", "context": "MRI_ONLY", **common,
        "validation_mse_analysis_space": float(selected.validation_mse_standardized),
        "alpha_validation_mse_json": json.dumps(dict(selected.alpha_grid), sort_keys=True),
        "x_scaler_train_rows": int(selected.x_scaler.n_samples_seen_),
        "target_transform_json": json.dumps(transform_payload, sort_keys=True),
        "static_transform_fit_scope": (
            "outer_train_grounding_eligible_measurement_valid_and_observable_visits"
            if task == "static" else "not_applicable_delta_outer_train_standard_scaler"
        ),
        "test_used_for_selection": False, "test_predict_call_count": guard.calls,
    }
    return selection, predictions


FTV_METRIC_COLUMNS = (
    "seed", "arm", "variant", "feature_dim", "task", "endpoint", "view", "target",
    "analysis_scope", "target_semantics", "aggregation", "n_test", "spearman", "pearson",
    "r2", "rmse", "mae", "b0_rmse", "rmse_gain_over_b0",
    "prediction_target_variance_ratio", "calibration_slope", "calibration_intercept",
    "calibration_mean_bias",
)


def aggregate_ftv_oof(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {
        "patient_id", "fold", "seed", "arm", "variant", "feature_dim", "task", "endpoint",
        "view", "target", "analysis_scope", "target_semantics", "y_true", "y_pred",
        "b0_prediction", "test_predict_call_count",
    }
    if missing := sorted(required - set(predictions.columns)):
        raise ValueError(f"FTV predictions miss columns: {missing}")
    if predictions.empty or not predictions["test_predict_call_count"].eq(1).all():
        raise ValueError("FTV predictions must be nonempty single-use outer-test rows")
    group_columns = [
        "seed", "arm", "variant", "feature_dim", "task", "endpoint", "view", "target",
        "analysis_scope", "target_semantics",
    ]
    rows: list[dict[str, Any]] = []
    for key, group in predictions.groupby(group_columns, sort=True, dropna=False):
        if group["patient_id"].astype(str).duplicated().any() or set(group["fold"].astype(int)) != set(range(5)):
            raise ValueError(f"FTV OOF fold/patient coverage drifted: {key}")
        rows.append({
            **dict(zip(group_columns, key, strict=True)), "aggregation": "pooled_outer_test_folds",
            "n_test": int(len(group)),
            **_continuous_metrics(
                group["y_true"].to_numpy(dtype=float), group["y_pred"].to_numpy(dtype=float),
                group["b0_prediction"].to_numpy(dtype=float),
            ),
        })
    endpoint_frame = pd.DataFrame(rows)
    macro_group = [name for name in group_columns if name not in {"endpoint", "view"}]
    macros: list[dict[str, Any]] = []
    metric_names = FTV_METRIC_COLUMNS[12:]
    for key, group in endpoint_frame.groupby(macro_group, sort=True, dropna=False):
        expected = set(VISITS if str(group["task"].iloc[0]) == "static" else TRANSITIONS)
        if set(group["endpoint"]) != expected:
            raise ValueError(f"FTV endpoint coverage drifted for macro: {key}")
        macros.append({
            **dict(zip(macro_group, key, strict=True)), "endpoint": "macro", "view": "macro",
            "aggregation": "mean_of_pooled_endpoint_metrics", "n_test": int(group["n_test"].sum()),
            **{name: float(group[name].mean()) for name in metric_names},
        })
    return pd.concat((endpoint_frame, pd.DataFrame(macros)), ignore_index=True).reindex(columns=FTV_METRIC_COLUMNS)


def run_ftv_probes(
    config: Mapping[str, Any],
    records: Mapping[str, Any],
    assets: Mapping[tuple[int, str, int], RegionalFeatureAsset],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    from c1b_stage_b.targets import fit_static_probe_transform

    selections: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for (_seed, _arm, _fold), asset in sorted(assets.items()):
        outer_train_ids = tuple(asset.patient_id[asset.split == "train"].astype(str))
        static_transform = fit_static_probe_transform(records, outer_train_ids, asset.fold)
        for variant in config["analysis"]["all_probe_variants"]:
            for scope in ("primary_measurement_valid", "observable_only"):
                observable_only = scope == "observable_only"
                for task, endpoints in (("static", VISITS), ("delta", TRANSITIONS)):
                    for index, _endpoint in enumerate(endpoints):
                        selection, rows = _run_ftv_endpoint(
                            config, asset, str(variant), records, task=task, index=index,
                            observable_only=observable_only, static_transform=static_transform,
                        )
                        selections.append(selection)
                        prediction_rows.extend(rows)
    predictions = pd.DataFrame(prediction_rows)
    return predictions, aggregate_ftv_oof(predictions), selections


GOAL5_PREDICTION_COLUMNS = (
    "patient_id", "fold", "population", "seed", "arm", "analysis", "view", "target",
    "variant", "clinical_contract", "y_true", "predicted_probability", "predicted_label",
    "threshold", "probability_class_0", "probability_class_1", "probability_class_2",
    "probability_class_3", "class_label_0", "class_label_1", "class_label_2", "class_label_3",
)
ORACLE_TABLE_COLUMNS = (
    "row_type", "seed", "arm", "view", "target", "population", "source", "variant",
    "candidate", "reference", "n", "n_positive", "n_negative", "auroc", "auprc",
    "balanced_accuracy", "brier", "r0_auroc", "candidate_auroc",
    "numerator_auroc_uplift", "published_fixed_p3_auroc", "published_peri20_auroc",
    "published_oracle_uplift", "recovery_ratio", "recovery_defined",
    "representation_note", "matched_patient_sha256",
)


def _load_goal5_predictions(path: Path, expected_sha256: str, label: str) -> pd.DataFrame:
    if file_sha256(path) != _require_sha256(expected_sha256, f"{label} SHA-256"):
        raise ValueError(f"{label} SHA-256 mismatch")
    frame = pd.read_csv(path, float_precision="round_trip")
    if tuple(frame.columns) != GOAL5_PREDICTION_COLUMNS:
        raise ValueError(f"{label} schema/order drifted")
    if frame.empty or frame["patient_id"].isna().any():
        raise ValueError(f"{label} is empty or has missing patient IDs")
    frame["patient_id"] = frame["patient_id"].astype(str)
    return frame


def _binary_oof_values(group: pd.DataFrame) -> dict[str, Any]:
    if group.empty or group["patient_id"].astype(str).duplicated().any():
        raise ValueError("binary OOF group must have one row per patient")
    labels = group["y_true"].to_numpy(dtype=np.int64)
    probability = group["predicted_probability"].to_numpy(dtype=np.float64)
    predicted = group["predicted_label"].to_numpy(dtype=np.int64)
    if set(np.unique(labels)) != {0, 1} or not np.isfinite(probability).all():
        raise ValueError("binary OOF labels/probabilities are invalid")
    return {
        "n": int(len(group)), "n_positive": int(labels.sum()),
        "n_negative": int(np.sum(labels == 0)), "auroc": float(roc_auc_score(labels, probability)),
        "auprc": float(average_precision_score(labels, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "brier": float(np.mean(np.square(probability - labels))),
    }


def _require_exact_pair(reference: pd.DataFrame, comparison: pd.DataFrame, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = reference.sort_values("patient_id", kind="mergesort").reset_index(drop=True)
    right = comparison.sort_values("patient_id", kind="mergesort").reset_index(drop=True)
    if not np.array_equal(left["patient_id"].astype(str), right["patient_id"].astype(str)):
        raise ValueError(f"{label} patient sets differ")
    if not np.array_equal(left["fold"].to_numpy(dtype=int), right["fold"].to_numpy(dtype=int)):
        raise ValueError(f"{label} fold assignments differ")
    if not np.array_equal(left["y_true"].to_numpy(dtype=int), right["y_true"].to_numpy(dtype=int)):
        raise ValueError(f"{label} labels differ")
    return left, right


def verify_r0_p1_parity(new_predictions: pd.DataFrame, goal5_mri_predictions: pd.DataFrame) -> dict[str, Any]:
    """Require exact R0/P1 reproduction for every shared pCR cell."""

    new = new_predictions.loc[
        new_predictions["analysis"].eq("mri_only_pcr") & new_predictions["variant"].eq("R0")
    ]
    old = goal5_mri_predictions.loc[
        goal5_mri_predictions["analysis"].eq("mri_only_pcr") & goal5_mri_predictions["variant"].eq("P1")
    ]
    identities = ["seed", "arm", "view", "target", "population"]
    new_keys = set(map(tuple, new[identities].drop_duplicates().to_numpy()))
    old_keys = set(map(tuple, old[identities].drop_duplicates().to_numpy()))
    if new_keys != old_keys:
        raise ValueError("new R0 and Goal-5 P1 cell coverage differs")
    checked_rows = 0
    for key in sorted(new_keys):
        new_group = new
        old_group = old
        for name, value in zip(identities, key, strict=True):
            new_group = new_group.loc[new_group[name].eq(value)]
            old_group = old_group.loc[old_group[name].eq(value)]
        left, right = _require_exact_pair(new_group, old_group, f"R0/P1 parity {key}")
        for column, dtype in (
            ("predicted_probability", float), ("predicted_label", int), ("threshold", float)
        ):
            if not np.array_equal(left[column].to_numpy(dtype=dtype), right[column].to_numpy(dtype=dtype)):
                raise ValueError(f"new R0 does not exactly reproduce Goal-5 P1 {column}: {key}")
        checked_rows += len(left)
    return {
        "status": "PASS", "exact_probability_label_threshold_equality": True,
        "checked_cells": len(new_keys), "checked_rows": checked_rows,
    }


def build_oracle_recovery_table(
    config: Mapping[str, Any],
    new_mri_predictions: pd.DataFrame,
    new_mri_metrics: pd.DataFrame,
    goal5_oracle_predictions: pd.DataFrame,
    goal5_mri_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build matched Goal-5 tables and the preregistered recovery numerator/ratio."""

    oracle = config["oracle"]
    arm = str(oracle["primary_arm"])
    view = str(oracle["view"])
    population = str(oracle["population"])
    expected_count = int(oracle["patient_count"])
    expected_positive = int(oracle["positive_count"])
    expected_patient_hash = str(oracle["sorted_patient_sha256"])
    rows: list[dict[str, Any]] = []
    selected_goal5: dict[tuple[int, str], pd.DataFrame] = {}
    for seed in config["frozen_cells"]["seed_bases"]:
        for variant in oracle["reported_goal5_variants"]:
            pair_population = population if variant in {"PERI20", "FIXED_P3"} else f"oracle_pair_{variant}"
            group = goal5_oracle_predictions.loc[
                goal5_oracle_predictions["seed"].eq(seed)
                & goal5_oracle_predictions["arm"].eq(arm)
                & goal5_oracle_predictions["view"].eq(view)
                & goal5_oracle_predictions["target"].eq("pCR")
                & goal5_oracle_predictions["population"].eq(pair_population)
                & goal5_oracle_predictions["variant"].eq(variant)
            ].copy()
            if len(group) != expected_count:
                raise ValueError(f"Goal-5 matched Oracle row count drifted for seed={seed}/{variant}")
            patient_hash = ordered_sha256(sorted(group["patient_id"].astype(str)))
            if patient_hash != expected_patient_hash:
                raise ValueError(f"Goal-5 matched patient hash drifted for seed={seed}/{variant}")
            values = _binary_oof_values(group)
            if values["n_positive"] != expected_positive:
                raise ValueError("Goal-5 matched Oracle positive count drifted")
            selected_goal5[(int(seed), str(variant))] = group
            rows.append({
                "row_type": "matched_metric", "seed": int(seed), "arm": arm, "view": view,
                "target": "pCR", "population": pair_population, "source": "Goal5_Oracle",
                "variant": str(variant), "candidate": "", "reference": "", **values,
                "representation_note": "Goal5 mean+std lesion-mask Oracle/fixed representation",
                "matched_patient_sha256": patient_hash,
            })
        fixed, peri20 = _require_exact_pair(
            selected_goal5[(int(seed), "FIXED_P3")], selected_goal5[(int(seed), "PERI20")],
            f"Goal-5 PERI20/FIXED_P3 seed={seed}",
        )
        observed_denominator = _binary_oof_values(peri20)["auroc"] - _binary_oof_values(fixed)["auroc"]
        expected_denominator = float(oracle["published_uplift"][str(seed)])
        if not np.isclose(observed_denominator, expected_denominator, rtol=0.0, atol=1e-15):
            raise ValueError(
                f"published Goal-5 Oracle denominator drifted for seed={seed}: "
                f"{observed_denominator} != {expected_denominator}"
            )

    selected_new = new_mri_predictions.loc[
        new_mri_predictions["analysis"].eq("mri_only_pcr")
        & new_mri_predictions["arm"].eq(arm)
        & new_mri_predictions["view"].eq(view)
        & new_mri_predictions["target"].eq("pCR")
        & new_mri_predictions["population"].eq("ftv_complete_375")
        & new_mri_predictions["variant"].isin(("R0", *oracle["new_candidates"]))
    ]
    for (seed, variant), group in selected_new.groupby(["seed", "variant"], sort=True):
        patient_hash = ordered_sha256(sorted(group["patient_id"].astype(str)))
        if len(group) != expected_count or patient_hash != expected_patient_hash:
            raise ValueError(f"new matched Oracle population drifted for seed={seed}/{variant}")
        reference_group = selected_goal5[(int(seed), "PERI20")]
        _require_exact_pair(group, reference_group, f"new/Goal-5 matched population seed={seed}/{variant}")
        values = _binary_oof_values(group)
        rows.append({
            "row_type": "matched_metric", "seed": int(seed), "arm": arm, "view": view,
            "target": "pCR", "population": "ftv_complete_375", "source": "MaskFree",
            "variant": str(variant), "candidate": "", "reference": "", **values,
            "representation_note": "raw regional mean (mask-free readout)",
            "matched_patient_sha256": patient_hash,
        })

    metric_selection = new_mri_metrics.loc[
        new_mri_metrics["analysis"].eq("mri_only_pcr")
        & new_mri_metrics["arm"].eq(arm)
        & new_mri_metrics["view"].eq(view)
        & new_mri_metrics["population"].eq("ftv_complete_375")
    ]
    for seed in config["frozen_cells"]["seed_bases"]:
        by_variant = metric_selection.loc[metric_selection["seed"].eq(seed)].set_index("variant")
        if "R0" not in by_variant.index:
            raise ValueError("Oracle recovery lacks the new R0 metric")
        r0 = float(by_variant.loc["R0", "auroc"])
        fixed = _binary_oof_values(selected_goal5[(int(seed), "FIXED_P3")])["auroc"]
        peri20 = _binary_oof_values(selected_goal5[(int(seed), "PERI20")])["auroc"]
        denominator = peri20 - fixed
        for candidate in oracle["new_candidates"]:
            if candidate not in by_variant.index:
                raise ValueError(f"Oracle recovery lacks candidate {candidate}")
            candidate_auroc = float(by_variant.loc[candidate, "auroc"])
            numerator = candidate_auroc - r0
            defined = bool(denominator > 0)
            rows.append({
                "row_type": "recovery", "seed": int(seed), "arm": arm, "view": view,
                "target": "pCR", "population": "ftv_complete_375", "source": "MatchedRecovery",
                "variant": str(candidate), "candidate": str(candidate), "reference": "R0",
                "n": expected_count, "n_positive": expected_positive,
                "n_negative": expected_count - expected_positive,
                "r0_auroc": r0, "candidate_auroc": candidate_auroc,
                "numerator_auroc_uplift": numerator, "published_fixed_p3_auroc": fixed,
                "published_peri20_auroc": peri20, "published_oracle_uplift": denominator,
                "recovery_ratio": numerator / denominator if defined else math.nan,
                "recovery_defined": defined,
                "representation_note": (
                    "numerator raw regional means vs raw R0; denominator published "
                    "PERI20(mean+std) vs FIXED_P3(mean+std)"
                ),
                "matched_patient_sha256": expected_patient_hash,
            })

        p1 = goal5_mri_predictions.loc[
            goal5_mri_predictions["seed"].eq(seed)
            & goal5_mri_predictions["arm"].eq(arm)
            & goal5_mri_predictions["view"].eq(view)
            & goal5_mri_predictions["target"].eq("pCR")
            & goal5_mri_predictions["population"].eq("ftv_complete_375")
            & goal5_mri_predictions["variant"].eq("P1")
        ]
        p1, peri20_group = _require_exact_pair(p1, selected_goal5[(int(seed), "PERI20")], f"Goal-5 P1/PERI20 bridge seed={seed}")
        p1_auroc = _binary_oof_values(p1)["auroc"]
        rows.append({
            "row_type": "bridge", "seed": int(seed), "arm": arm, "view": view,
            "target": "pCR", "population": "ftv_complete_375", "source": "Goal5_Bridge",
            "variant": "PERI20_minus_P1", "candidate": "PERI20", "reference": "P1",
            "n": expected_count, "n_positive": expected_positive,
            "n_negative": expected_count - expected_positive, "r0_auroc": p1_auroc,
            "candidate_auroc": peri20, "numerator_auroc_uplift": peri20 - p1_auroc,
            "representation_note": "bridge diagnostic only; not the recovery denominator",
            "matched_patient_sha256": expected_patient_hash,
        })
    parity = verify_r0_p1_parity(new_mri_predictions, goal5_mri_predictions)
    table = pd.DataFrame(rows).reindex(columns=ORACLE_TABLE_COLUMNS)
    return table, parity


BOOTSTRAP_COLUMNS = (
    "seed", "arm", "context", "view", "timing", "target", "population", "candidate",
    "reference_model", "comparison_model", "metric", "reference", "comparison", "estimate",
    "improvement", "delta_auroc", "ci_lower", "ci_upper", "confidence_level", "n_patients",
    "n_folds", "n_bootstrap", "n_valid_bootstrap", "bootstrap_unit", "ci_method",
    "orientation", "bootstrap_seed",
)


def _prediction_cell(
    predictions: pd.DataFrame,
    *,
    seed: int,
    arm: str,
    view: str,
    population: str,
    model: str,
) -> pd.DataFrame:
    return predictions.loc[
        predictions["seed"].eq(seed)
        & predictions["arm"].eq(arm)
        & predictions["view"].eq(view)
        & predictions["population"].eq(population)
        & predictions["model"].eq(model),
        ["patient_id", "fold", "y_true", "predicted_probability"],
    ].copy()


def run_preregistered_bootstrap(
    config: Mapping[str, Any],
    mri_predictions: pd.DataFrame,
    incremental_predictions: pd.DataFrame,
    goal5_oracle_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run exactly the registered paired fold-stratified 2000-draw scope."""

    bootstrap = config["bootstrap"]
    summary_rows: list[dict[str, Any]] = []
    draw_rows: list[pd.DataFrame] = []
    pair_specs: list[tuple[str, int, str, str, str, str, str, pd.DataFrame, pd.DataFrame]] = []
    for seed in config["frozen_cells"]["seed_bases"]:
        for arm in config["frozen_cells"]["arms"]:
            for view in bootstrap["timings"]:
                for candidate in bootstrap["candidate_variants"]:
                    pair_specs.append((
                        "MRI_ONLY", int(seed), str(arm), str(view), "full_808", str(candidate),
                        "R0", _prediction_cell(mri_predictions, seed=int(seed), arm=str(arm), view=str(view),
                                               population="full_808", model="R0"),
                        _prediction_cell(mri_predictions, seed=int(seed), arm=str(arm), view=str(view),
                                         population="full_808", model=str(candidate)),
                    ))
                    pair_specs.append((
                        "C_PLUS_F", int(seed), str(arm), str(view), "ftv_complete_375", str(candidate),
                        "C+F+R0", _prediction_cell(incremental_predictions, seed=int(seed), arm=str(arm), view=str(view),
                                                   population="ftv_complete_375", model="C+F+R0"),
                        _prediction_cell(incremental_predictions, seed=int(seed), arm=str(arm), view=str(view),
                                         population="ftv_complete_375", model=f"C+F+{candidate}"),
                    ))
    oracle = config["oracle"]
    for seed in config["frozen_cells"]["seed_bases"]:
        common = (
            goal5_oracle_predictions["seed"].eq(seed)
            & goal5_oracle_predictions["arm"].eq(oracle["primary_arm"])
            & goal5_oracle_predictions["view"].eq(oracle["view"])
            & goal5_oracle_predictions["target"].eq("pCR")
            & goal5_oracle_predictions["population"].eq(oracle["population"])
        )
        reference = goal5_oracle_predictions.loc[
            common & goal5_oracle_predictions["variant"].eq("FIXED_P3"),
            ["patient_id", "fold", "y_true", "predicted_probability"],
        ]
        comparison = goal5_oracle_predictions.loc[
            common & goal5_oracle_predictions["variant"].eq("PERI20"),
            ["patient_id", "fold", "y_true", "predicted_probability"],
        ]
        pair_specs.append((
            "GOAL5_ORACLE", int(seed), str(oracle["primary_arm"]), str(oracle["view"]),
            str(oracle["population"]), "PERI20", "FIXED_P3", reference, comparison,
        ))

    for context, seed, arm, view, population, candidate, reference_name, reference, comparison in pair_specs:
        result = paired_fold_stratified_bootstrap(
            reference, comparison, n_bootstrap=int(bootstrap["replicates"]),
            confidence_level=float(bootstrap["confidence_level"]), seed=int(bootstrap["seed"]),
        )
        comparison_name = candidate if context != "C_PLUS_F" else f"C+F+{candidate}"
        summary_records = result.summary.to_dict("records")
        if len(summary_records) != 3 or {str(row.get("metric")) for row in summary_records} != {
            "auroc", "auprc", "brier",
        }:
            raise ValueError("paired bootstrap summary metric coverage drifted")
        for returned_row in summary_records:
            row = dict(returned_row)
            expected_summary_fields = {
                "metric", "reference", "comparison", "improvement", "ci_lower", "ci_upper",
                "confidence_level", "n_patients", "n_folds", "n_bootstrap",
                "n_valid_bootstrap", "bootstrap_unit", "ci_method", "orientation", "seed",
            }
            if set(row) != expected_summary_fields:
                raise ValueError("paired bootstrap summary schema drifted")
            returned_bootstrap_seed = row.pop("seed", None)
            if returned_bootstrap_seed != int(bootstrap["seed"]):
                raise ValueError("paired bootstrap RNG-seed provenance drifted")
            if (
                int(row["n_bootstrap"]) != int(bootstrap["replicates"])
                or float(row["confidence_level"]) != float(bootstrap["confidence_level"])
                or row["bootstrap_unit"] != "patient_within_outer_fold"
                or row["ci_method"] != "percentile"
            ):
                raise ValueError("paired bootstrap execution provenance drifted")
            metric = str(row["metric"])
            summary_rows.append({
                "seed": seed, "arm": arm, "context": context, "view": view, "timing": view,
                "target": "pCR", "population": population, "candidate": candidate,
                "reference_model": reference_name, "comparison_model": comparison_name,
                **row, "estimate": float(row["improvement"]),
                "delta_auroc": float(row["improvement"]) if metric == "auroc" else math.nan,
                "bootstrap_seed": int(returned_bootstrap_seed),
            })
        draws = result.draws.copy()
        if tuple(draws.columns) != (
            "bootstrap_index", "auroc_improvement", "auprc_improvement", "brier_improvement"
        ):
            raise ValueError("paired bootstrap draw schema drifted")
        draws.insert(0, "seed", seed)
        draws.insert(1, "arm", arm)
        draws.insert(2, "context", context)
        draws.insert(3, "view", view)
        draws.insert(4, "population", population)
        draws.insert(5, "candidate", candidate)
        draws.insert(6, "reference_model", reference_name)
        draws.insert(7, "comparison_model", comparison_name)
        draw_rows.append(draws)
    summary_frame = pd.DataFrame(summary_rows).reindex(columns=BOOTSTRAP_COLUMNS)
    draw_frame = pd.concat(draw_rows, ignore_index=True)
    summary_identity = (
        "seed", "arm", "context", "view", "population", "candidate",
        "reference_model", "comparison_model", "metric",
    )
    expected_summary_identities = {
        (
            seed, arm, context, view, population, candidate, reference_name,
            candidate if context != "C_PLUS_F" else f"C+F+{candidate}", metric,
        )
        for context, seed, arm, view, population, candidate, reference_name, _, _ in pair_specs
        for metric in ("auroc", "auprc", "brier")
    }
    observed_summary_identities = set(
        map(tuple, summary_frame.loc[:, summary_identity].to_numpy())
    )
    if (
        summary_frame.duplicated(list(summary_identity)).any()
        or observed_summary_identities != expected_summary_identities
    ):
        raise ValueError("bootstrap summary scientific-cell identity drifted")
    draw_identity = summary_identity[:-1] + ("bootstrap_index",)
    expected_draw_identities = {
        (
            seed, arm, context, view, population, candidate, reference_name,
            candidate if context != "C_PLUS_F" else f"C+F+{candidate}", bootstrap_index,
        )
        for context, seed, arm, view, population, candidate, reference_name, _, _ in pair_specs
        for bootstrap_index in range(int(bootstrap["replicates"]))
    }
    observed_draw_identities = set(map(tuple, draw_frame.loc[:, draw_identity].to_numpy()))
    if draw_frame.duplicated(list(draw_identity)).any() or observed_draw_identities != expected_draw_identities:
        raise ValueError("bootstrap draws scientific-cell identity drifted")
    return summary_frame, draw_frame


SEED_CONSISTENCY_COLUMNS = (
    "context", "arm", "view", "timing", "target", "population", "candidate",
    "reference", "seed_2026_delta_auroc", "seed_3026_delta_auroc", "mean_delta_auroc",
    "both_seeds_strictly_positive",
)


def build_seed_consistency(
    mri_metrics: pd.DataFrame,
    incremental_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    specs = (
        ("MRI_ONLY", mri_metrics.loc[mri_metrics["population"].eq("full_808")], "R0", lambda candidate: candidate),
        ("C_PLUS_F", incremental_metrics, "C+F+R0", lambda candidate: f"C+F+{candidate}"),
    )
    for context, frame, reference, model_name in specs:
        selected = frame.loc[frame["view"].isin(EARLY_TIMINGS) & frame["variant"].isin(("R0", *GATE_A_VARIANTS))]
        identity = ["arm", "view", "target", "population", "seed"]
        if selected.duplicated([*identity, "model"]).any():
            raise ValueError("seed consistency metric identity repeats")
        pivot = selected.pivot(index=identity, columns="model", values="auroc")
        for candidate in GATE_A_VARIANTS:
            comparison = model_name(candidate)
            if not {reference, comparison}.issubset(pivot.columns):
                continue
            delta = (pivot[comparison] - pivot[reference]).rename("delta").reset_index()
            for key, group in delta.groupby(identity[:-1], sort=True, dropna=False):
                by_seed = {int(row.seed): float(row.delta) for row in group.itertuples(index=False)}
                if set(by_seed) != {2026, 3026}:
                    raise ValueError("seed consistency lacks the exact two seeds")
                rows.append({
                    "context": context,
                    **dict(zip(identity[:-1], (_plain(value) for value in key), strict=True)),
                    "timing": key[1], "candidate": candidate, "reference": reference,
                    "seed_2026_delta_auroc": by_seed[2026],
                    "seed_3026_delta_auroc": by_seed[3026],
                    "mean_delta_auroc": float(np.mean(list(by_seed.values()))),
                    "both_seeds_strictly_positive": bool(by_seed[2026] > 0 and by_seed[3026] > 0),
                })
    return pd.DataFrame(rows).reindex(columns=SEED_CONSISTENCY_COLUMNS)


TIMING_SENSITIVITY_COLUMNS = (
    "seed", "arm", "context", "view", "timing", "timing_label", "target", "population",
    "variant", "reference", "auroc", "reference_auroc", "delta_auroc_vs_r0",
)


def build_timing_sensitivity(mri_metrics: pd.DataFrame, incremental_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    specs = (
        ("MRI_ONLY", mri_metrics, "R0", lambda variant: variant),
        ("C_PLUS_F", incremental_metrics, "C+F+R0", lambda variant: f"C+F+{variant}"),
    )
    for context, frame, reference, model_name in specs:
        identity = ["seed", "arm", "view", "target", "population"]
        for key, group in frame.groupby(identity, sort=True, dropna=False):
            by_model = group.set_index("model", verify_integrity=True)
            if reference not in by_model.index:
                raise ValueError(f"timing sensitivity lacks reference {reference}: {key}")
            reference_auroc = float(by_model.loc[reference, "auroc"])
            for variant in PRIMARY_VARIANTS:
                model = model_name(variant)
                if model not in by_model.index:
                    raise ValueError(f"timing sensitivity lacks model {model}: {key}")
                auroc = float(by_model.loc[model, "auroc"])
                rows.append({
                    **dict(zip(identity, (_plain(value) for value in key), strict=True)),
                    "context": context, "timing": key[2],
                    "timing_label": "late/pre-surgery" if key[2] == "T0-T3" else "",
                    "variant": variant, "reference": reference, "auroc": auroc,
                    "reference_auroc": reference_auroc, "delta_auroc_vs_r0": auroc - reference_auroc,
                })
    return pd.DataFrame(rows).reindex(columns=TIMING_SENSITIVITY_COLUMNS)


def _public(frame: pd.DataFrame) -> pd.DataFrame:
    forbidden = {"patient_id", "clinical_patient_id", "raw_Patient_ID"}
    if overlap := forbidden & set(frame.columns):
        raise ValueError(f"public table contains patient identifiers: {sorted(overlap)}")
    return frame


def _atomic_csv(frame: pd.DataFrame, path: Path, *, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700 if private else 0o755)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False, lineterminator="\n")
        temporary.chmod(0o600 if private else 0o644)
        os.replace(temporary, path)
        path.chmod(0o600 if private else 0o644)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: Mapping[str, Any], path: Path, *, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700 if private else 0o755)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(dict(payload), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o600 if private else 0o644)
        os.replace(temporary, path)
        path.chmod(0o600 if private else 0o644)
    finally:
        temporary.unlink(missing_ok=True)


def _prepare_outputs(paths: Mapping[str, Path]) -> None:
    if paths["run_summary"].exists():
        raise FileExistsError("run_summary.json exists; completed audit results are immutable")
    for name, path in paths.items():
        if name == "run_summary" or not path.exists():
            continue
        if path.is_dir():
            raise IsADirectoryError(path)
        path.unlink()


def output_paths(root: Path = ROOT) -> dict[str, Path]:
    return {
        "mri_predictions": root / "predictions" / "mri_only_pcr_oof.private.csv",
        "clinical_predictions": root / "predictions" / "clinical_pcr_oof.private.csv",
        "incremental_predictions": root / "predictions" / "clinical_ftv_pcr_oof.private.csv",
        "phenotype_predictions": root / "predictions" / "phenotype_oof.private.csv",
        "ftv_predictions": root / "predictions" / "ftv_oof.private.csv",
        "hyperparameters": root / "predictions" / "hyperparameters.private.csv",
        "bootstrap_draws": root / "predictions" / "bootstrap_draws.private.csv",
        "occupancy": root / "metrics" / "region_occupancy.csv",
        "mri_metrics": root / "metrics" / "table_mri_only_pcr.csv",
        "clinical_metrics": root / "metrics" / "table_clinical_pcr.csv",
        "incremental_metrics": root / "metrics" / "table_clinical_ftv_incremental.csv",
        "phenotype_metrics": root / "metrics" / "table_phenotype.csv",
        "ftv_metrics": root / "metrics" / "table_ftv.csv",
        "oracle": root / "metrics" / "table_oracle_recovery.csv",
        "bootstrap": root / "metrics" / "table_bootstrap.csv",
        "seed_consistency": root / "metrics" / "table_seed_consistency.csv",
        "timing_sensitivity": root / "metrics" / "table_timing_sensitivity.csv",
        "gates": root / "metrics" / "gates.json",
        "run_summary": root / "metrics" / "run_summary.json",
    }


def region_occupancy_table(config: Mapping[str, Any]) -> pd.DataFrame:
    """Flatten the extraction-frozen geometry contract into a public table."""

    source = ROOT / "metrics" / "region_occupancy_contract.json"
    payload = _load_json(source, "region occupancy contract")
    map_names = (
        "weight_sum_cells", "physical_volume_mm3", "expected_physical_volume_mm3",
        "nonzero_cells", "fractional_cells",
    )
    mappings = {name: payload.get(name) for name in map_names}
    expected_regions = ("R0", "R1", "R2", "R3", "S1", "S2", "S3")
    for name, values in mappings.items():
        if not isinstance(values, Mapping) or set(values) != set(expected_regions):
            raise ValueError(f"region occupancy contract {name} coverage drifted")
    primary = tuple(float(value) for value in config["feature_contract"]["primary_boundaries_mm"])
    secondary = tuple(float(value) for value in config["feature_contract"]["secondary_boundaries_mm"])
    definitions = {
        "R0": ("primary_and_secondary", 0.0, primary[2], "full_local_cube"),
        "R1": ("primary", 0.0, primary[0], "central_cube"),
        "R2": ("primary", primary[0], primary[1], "inner_shell"),
        "R3": ("primary", primary[1], primary[2], "outer_shell"),
        "S1": ("secondary", 0.0, secondary[0], "central_cube"),
        "S2": ("secondary", secondary[0], secondary[1], "inner_shell"),
        "S3": ("secondary", secondary[1], secondary[2], "outer_shell"),
    }
    rows = []
    for region in expected_regions:
        geometry, inner, outer, definition = definitions[region]
        values = {name: float(mappings[name][region]) for name in map_names[:3]}
        counts = {name: int(mappings[name][region]) for name in map_names[3:]}
        if (
            not all(np.isfinite(value) and value > 0 for value in values.values())
            or any(value < 0 for value in counts.values())
        ):
            raise ValueError(f"invalid occupancy aggregate for {region}")
        rows.append({
            "geometry": geometry, "region": region, "variant": region,
            "definition": definition, "inner_boundary_mm": inner,
            "outer_boundary_mm": outer,
            "mean_effective_cells": values["weight_sum_cells"],
            **values, **counts,
            "sampling_cell_volume_mm3": float(payload["sampling_cell_volume_mm3"]),
        })
    return pd.DataFrame(rows)


def _public_mri_table(metrics: pd.DataFrame) -> pd.DataFrame:
    output = metrics.copy()
    output["timing"] = output["view"]
    identity = ["seed", "arm", "view", "target", "population"]
    reference = output.loc[output["variant"].eq("R0"), [*identity, "auroc"]].rename(columns={"auroc": "r0_auroc"})
    output = output.merge(reference, on=identity, how="left", validate="many_to_one")
    output["delta_auroc_vs_r0"] = output["auroc"] - output["r0_auroc"]
    output["delta_auroc"] = output["delta_auroc_vs_r0"]
    return output


def _public_clinical_table(metrics: pd.DataFrame) -> pd.DataFrame:
    output = metrics.copy()
    output["timing"] = output["view"]
    identity = ["seed", "arm", "view", "target", "population", "clinical_contract"]
    c_reference = output.loc[output["model"].eq("C"), [*identity, "auroc", "auprc", "brier"]].rename(
        columns={"auroc": "c_auroc", "auprc": "c_auprc", "brier": "c_brier"}
    )
    output = output.merge(c_reference, on=identity, how="left", validate="many_to_one")
    output["delta_auroc_vs_C"] = output["auroc"] - output["c_auroc"]
    output["delta_auprc_vs_C"] = output["auprc"] - output["c_auprc"]
    output["brier_improvement_vs_C"] = output["c_brier"] - output["brier"]
    return output


def _public_incremental_table(metrics: pd.DataFrame) -> pd.DataFrame:
    output = add_reference_deltas(metrics, reference_model="C+F+R0", secondary_reference_model="C+F")
    output["timing"] = output["view"]
    output["delta_auroc_vs_cf_r0"] = output["delta_auroc_vs_C+F+R0"]
    output["delta_auroc"] = output["delta_auroc_vs_C+F+R0"]
    return output


def _public_phenotype_table(metrics: pd.DataFrame) -> pd.DataFrame:
    output = metrics.copy()
    output["visit"] = output["view"]
    identity = ["seed", "arm", "view", "target", "population"]
    reference = output.loc[output["variant"].eq("R0"), [*identity, "auroc"]].rename(columns={"auroc": "r0_auroc"})
    output = output.merge(reference, on=identity, how="left", validate="many_to_one")
    output["delta_auroc_vs_r0"] = output["auroc"] - output["r0_auroc"]
    return output


def _public_ftv_table(metrics: pd.DataFrame) -> pd.DataFrame:
    output = metrics.copy()
    output["timing"] = output["view"]
    output["target"] = np.where(output["task"].eq("delta"), "delta_FTV", "FTV")
    return output


def _git_value(arguments: Sequence[str], default: str) -> str:
    import subprocess

    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *arguments], text=True, capture_output=True, check=False
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else default


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--feature-root", type=Path, default=FEATURE_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    os.umask(0o077)
    args = parse_args(argv)
    paths = output_paths(ROOT)
    started = __import__("time").time()

    # Feature/config/lock/inventory authentication precedes every label read.
    config = load_config(args.config, verify_paths=False)
    if args.config.resolve() != CONFIG_PATH.resolve():
        raise ValueError("formal analysis must use the preregistered config path")
    from common import load_config as load_extraction_config, require_preregistration_lock
    from run_feature_matrix import validate_complete, validate_completion_marker

    extraction_config = load_extraction_config(CONFIG_PATH, verify_extraction_inputs=True)
    lock = require_preregistration_lock(extraction_config)
    expected_completion = validate_complete(config=extraction_config, lock=lock)
    completion_path = ROOT / "features" / "feature_matrix_complete.private.json"
    completion = validate_completion_marker(completion_path, expected_completion)
    fold_assignments = load_fold_assignments(
        config["paths"]["fold_manifest"], config["paths"]["fold_manifest_sha256"],
        patient_count=808,
    )
    assets = load_all_regional_feature_assets(args.feature_root, fold_assignments, config, lock, completion)
    occupancy = region_occupancy_table(config)

    # Only an authenticated config/lock/completion/all-cell matrix may clear an
    # interrupted prior attempt.  A completed run_summary remains immutable.
    _prepare_outputs(paths)

    # Label-dependent analysis begins only after the complete frozen matrix.
    fold_manifest = load_fold_manifest(
        config["paths"]["fold_manifest"], config["paths"]["fold_manifest_sha256"],
        expected_patient_count=808,
    )
    if not fold_manifest[["patient_id", "fold", "split"]].equals(fold_assignments):
        raise ValueError("label-bearing fold load differs from pre-label assignments")
    clinical = load_clinical_table(
        config["paths"]["clinical_labels"], config["paths"]["clinical_labels_sha256"],
        fold_manifest, expected_patient_count=808,
    )
    ftv = load_ftv_wide(
        config["paths"]["ftv_table"], config["paths"]["ftv_table_sha256"],
        expected_patient_ids=clinical, expected_patient_count=375,
    )
    ftv_ids = set(ftv["patient_id"].astype(str))
    if len(ftv_ids) != 375:
        raise ValueError("matched FTV-complete population must contain 375 patients")
    records = load_ftv_records(config)

    mri_pred, mri_metrics, mri_hp = run_mri_only_pcr(config, clinical, ftv_ids, assets)
    clinical_pred, clinical_metrics, clinical_hp = run_clinical_pcr(config, clinical, ftv_ids, assets)
    incremental_pred, incremental_metrics, incremental_hp = run_clinical_ftv_pcr(config, clinical, ftv, assets)
    phenotype_pred, phenotype_metrics, phenotype_hp = run_phenotype_probes(config, clinical, assets)
    ftv_pred, ftv_metrics, ftv_hp = run_ftv_probes(config, records, assets)

    goal5_oracle = _load_goal5_predictions(
        config["paths"]["goal5_oracle_predictions"],
        config["paths"]["goal5_oracle_predictions_sha256"], "Goal-5 Oracle predictions",
    )
    goal5_mri = _load_goal5_predictions(
        config["paths"]["goal5_mri_predictions"],
        config["paths"]["goal5_mri_predictions_sha256"], "Goal-5 MRI predictions",
    )
    oracle_table, r0_p1_parity = build_oracle_recovery_table(
        config, mri_pred, mri_metrics, goal5_oracle, goal5_mri,
    )
    bootstrap, bootstrap_draws = run_preregistered_bootstrap(
        config, mri_pred, incremental_pred, goal5_oracle,
    )
    seed_consistency = build_seed_consistency(mri_metrics, incremental_metrics)
    timing_sensitivity = build_timing_sensitivity(mri_metrics, incremental_metrics)

    public_mri = _public_mri_table(mri_metrics)
    public_clinical = _public_clinical_table(clinical_metrics)
    public_incremental = _public_incremental_table(incremental_metrics)
    public_phenotype = _public_phenotype_table(phenotype_metrics)
    public_ftv = _public_ftv_table(ftv_metrics)
    gates = evaluate_gates(config, public_mri, public_incremental, public_phenotype, oracle_table)

    private_frames = {
        "mri_predictions": mri_pred, "clinical_predictions": clinical_pred,
        "incremental_predictions": incremental_pred, "phenotype_predictions": phenotype_pred,
        "ftv_predictions": ftv_pred,
        "hyperparameters": pd.DataFrame([*mri_hp, *clinical_hp, *incremental_hp, *phenotype_hp, *ftv_hp]),
        "bootstrap_draws": bootstrap_draws,
    }
    for name, frame in private_frames.items():
        _atomic_csv(frame, paths[name], private=True)
        if stat.S_IMODE(paths[name].stat().st_mode) != 0o600:
            raise PermissionError(f"private output is not mode 0600: {paths[name]}")
    public_frames = {
        "occupancy": occupancy, "mri_metrics": public_mri, "clinical_metrics": public_clinical,
        "incremental_metrics": public_incremental, "phenotype_metrics": public_phenotype,
        "ftv_metrics": public_ftv, "oracle": oracle_table, "bootstrap": bootstrap,
        "seed_consistency": seed_consistency, "timing_sensitivity": timing_sensitivity,
    }
    for name, frame in public_frames.items():
        _atomic_csv(_public(frame), paths[name], private=False)

    _atomic_json(gates, paths["gates"])
    elapsed = float(__import__("time").time() - started)
    summary = {
        "schema_version": 1, "experiment": "mask_free_region_aware_audit",
        "status": "COMPLETED", "branch": _git_value(("branch", "--show-current"), config["branch"]),
        "commit_sha": "PENDING",
        "push_status": "PENDING", "push_error": None, "elapsed_seconds": elapsed,
        "feature_cells_validated_before_labels": 20,
        "formal_bootstrap_replicates": int(config["bootstrap"]["replicates"]),
        "outer_test_predicted_once_per_model": True,
        "new_encoder_or_jepa_training_performed": False,
        "architecture_intervention_started": False,
        "r0_goal5_p1_prediction_parity": r0_p1_parity,
        "public_outputs_contain_patient_level_data": False,
        "private_patient_outputs_mode": "0600",
        "scientific_classification": gates["scientific_classification"],
        "gate_results": {letter: bool(gates["gates"][letter]["passed"]) for letter in "ABCD"},
        "public_outputs": {
            paths[name].relative_to(ROOT).as_posix(): file_sha256(paths[name])
            for name in public_frames
        },
        "private_outputs": {
            paths[name].relative_to(ROOT).as_posix(): file_sha256(paths[name])
            for name in private_frames
        },
    }
    _atomic_json(summary, paths["run_summary"])
    print(json.dumps({"status": "COMPLETED", "classification": gates["scientific_classification"]}, sort_keys=True))


if __name__ == "__main__":
    main()

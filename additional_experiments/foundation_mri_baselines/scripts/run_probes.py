#!/usr/bin/env python3
"""Run frozen foundation-representation phenotype and FTV probes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
SRC_ROOT = EXPERIMENT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from foundation_mri.data import (  # noqa: E402
    COHORT_SIZE,
    FOLDS,
    HR_HER2_SUBTYPES,
    RADIOMICS_COMPLETE_CASE_SIZE,
    SPATIAL_AXES,
    load_clinical_labels,
    load_current_cnn_features,
    load_fold_manifest,
    load_foundation_features,
    load_radiomics_table,
)
from foundation_mri.evaluation import (  # noqa: E402
    C_GRID,
    EvaluationResult,
    LOGISTIC_MAX_ITER,
    MULTICLASS_LOGISTIC_TOL,
    PENALTIES,
    RIDGE_ALPHAS,
    RIDGE_MAX_ITER,
    RIDGE_TOL,
    SEED,
    _bound_transformed_ridge_predictions,
    _target_inverse,
    aggregate_binary_predictions,
    aggregate_continuous_predictions,
    aggregate_multiclass_predictions,
    configure_metric_free_progress,
    evaluate_binary_cv,
    evaluate_multiclass_cv,
    evaluate_ridge_cv,
    metric_free_progress,
    select_multiclass_logistic,
    select_ridge,
    timing_matrix,
    write_private_csv,
    write_public_csv,
)
from foundation_mri.locking import (  # noqa: E402
    file_sha256,
    verify_formal_evaluation_lock,
    write_metric_free_run_provenance,
)


DEFAULT_CLINICAL = Path(
    "/data/data/Preprocessed/I-SPY2/clinical_labels_complete4visits.csv"
)
DEFAULT_FOLDS = Path(
    "/data/data/Preprocessed/I-SPY2/"
    "_matched_breastdcedl_t0_dicomrepair_rgb224_seed2026/"
    "matched_patient_cv_splits_seed2026.csv"
)
DEFAULT_RADIOMICS = (
    REPO_ROOT
    / "additional_experiments/radiomics_next_change/data_audit/"
    "radiomics_transition_targets_raw.csv"
)
DEFAULT_EVALUATION_LOCK = EXPERIMENT_ROOT / "configs/EVALUATION_LOCK.v3.json"

FORMAL_MEDICALNET_SHA256 = (
    "ca45a46bd62e18e42b6d3f2426ce4690a4f3dbf7c2f44804ab0d19bd333ee4a2"
)
SMOKE_RECEIPT_KIND = "foundation_mri_probe_v3_medicalnet_synthetic_smoke"
SMOKE_CLASSES = ("synthetic_0", "synthetic_1", "synthetic_2", "synthetic_3")
FORMAL_PROGRESS_PATH = EXPERIMENT_ROOT / "logs/probe_v3.progress.private.jsonl"
FORMAL_RECEIPT_PATH = EXPERIMENT_ROOT / "metrics/probe_v3_run.private.provenance.json"


@dataclass(frozen=True)
class ProbeSource:
    """One imaging representation, common or independently exported by fold."""

    name: str
    spatial: str
    by_fold: Mapping[int, np.ndarray]

    def is_constant(self) -> bool:
        return all(self.by_fold[fold] is self.by_fold[FOLDS[0]] for fold in FOLDS[1:])

    def subset(
        self, canonical_ids: Sequence[str], requested_ids: Sequence[str]
    ) -> "ProbeSource":
        lookup = {patient_id: index for index, patient_id in enumerate(canonical_ids)}
        unknown = sorted(set(requested_ids).difference(lookup))
        if unknown:
            raise ValueError(f"probe subset contains {len(unknown)} unknown patients")
        indices = np.asarray([lookup[value] for value in requested_ids], dtype=np.int64)
        if self.is_constant():
            subset = np.ascontiguousarray(self.by_fold[FOLDS[0]][indices])
            by_fold = _constant_folds(subset)
        else:
            by_fold = {
                fold: np.ascontiguousarray(self.by_fold[fold][indices]) for fold in FOLDS
            }
        return ProbeSource(self.name, self.spatial, by_fold)


def _constant_folds(values: np.ndarray) -> dict[int, np.ndarray]:
    return {fold: values for fold in FOLDS}


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _write_json_exclusive(payload: Mapping[str, object], path: Path) -> None:
    """Atomically publish a receipt without ever replacing an existing one."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"smoke receipt already exists: {destination.name}")
    body = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _synthetic_smoke_labels(expected_n: int) -> np.ndarray:
    if expected_n <= 0 or expected_n % len(SMOKE_CLASSES):
        raise ValueError("synthetic smoke cohort must divide evenly into four classes")
    balanced = np.repeat(
        np.asarray(SMOKE_CLASSES, dtype=str), expected_n // len(SMOKE_CLASSES)
    )
    return np.random.default_rng(SEED).permutation(balanced)


def _synthetic_smoke_roles(expected_n: int) -> np.ndarray:
    index = np.arange(expected_n, dtype=np.int64)
    return np.where(index % 5 == 0, "reserved_test", np.where(index % 5 == 1, "val", "train"))


def _run_medicalnet_synthetic_smoke(
    feature_path: Path,
    receipt_path: Path,
    *,
    expected_sha256: str = FORMAL_MEDICALNET_SHA256,
    expected_n: int = COHORT_SIZE,
    expected_dim: int = 14_336,
    spatial_axes: Sequence[str] = SPATIAL_AXES,
    penalties: Sequence[str] = PENALTIES,
    c_grid: Sequence[float] = C_GRID,
    require_thread_contract: bool = True,
) -> dict[str, object]:
    """Run a metric-free convergence/runtime smoke on formal-shaped MRI features.

    Labels and roles are generated locally.  This path deliberately has no
    fold-manifest, clinical-label, radiomics, or outcome-loader dependency.
    """

    source = Path(feature_path).resolve(strict=True)
    observed_sha256 = file_sha256(source)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            "MedicalNet smoke feature SHA-256 drifted: "
            f"expected={expected_sha256}, observed={observed_sha256}"
        )
    if require_thread_contract:
        expected_environment = {
            "PYTHONHASHSEED": str(SEED),
            "FOUNDATION_MRI_SELECTION_WORKERS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        observed_environment = {
            key: os.environ.get(key) for key in expected_environment
        }
        if observed_environment != expected_environment:
            raise RuntimeError(
                "synthetic smoke requires the locked seed/one-worker/one-thread environment"
            )
    else:
        observed_environment = {
            key: os.environ.get(key)
            for key in (
                "PYTHONHASHSEED",
                "FOUNDATION_MRI_SELECTION_WORKERS",
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        }

    with np.load(source, allow_pickle=False) as archive:
        if "patient_id" not in archive.files:
            raise ValueError("MedicalNet smoke feature is missing patient_id")
        patient_ids = np.asarray(archive["patient_id"])
    asset = load_foundation_features(
        source,
        expected_patient_ids=patient_ids,
        expected_n=expected_n,
    )
    if asset.model_name != "medicalnet_resnet50_3dseg8":
        raise ValueError("synthetic smoke requires the locked MedicalNet model")
    if asset.representation.shape != (expected_n, 4, 2, expected_dim):
        raise ValueError("synthetic smoke MedicalNet representation shape drifted")

    labels = _synthetic_smoke_labels(expected_n)
    roles = _synthetic_smoke_roles(expected_n)
    train_indices = np.flatnonzero(roles == "train")
    validation_indices = np.flatnonzero(roles == "val")
    reserved_indices = np.flatnonzero(roles == "reserved_test")
    for role_name, indices in (
        ("train", train_indices),
        ("val", validation_indices),
        ("reserved_test", reserved_indices),
    ):
        if set(labels[indices].tolist()) != set(SMOKE_CLASSES):
            raise RuntimeError(f"synthetic {role_name} split lacks a smoke class")

    candidate_rows: list[dict[str, object]] = []
    ridge_candidate_rows: list[dict[str, object]] = []
    elapsed_by_axis: dict[str, float] = {}
    ridge_elapsed_by_axis: dict[str, float] = {}
    continuous_digests: dict[str, str] = {}
    ridge_reserved_clipped: dict[str, int] = {}
    sentinel_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for spatial in tuple(spatial_axes):
        matrix = np.ascontiguousarray(asset.spatial(str(spatial))[:, 0, :])
        axis_started = time.perf_counter()
        selected = select_multiclass_logistic(
            matrix[train_indices],
            labels[train_indices],
            matrix[validation_indices],
            labels[validation_indices],
            classes=SMOKE_CLASSES,
            penalties=penalties,
            c_grid=c_grid,
            random_state=SEED,
        )
        elapsed_by_axis[str(spatial)] = time.perf_counter() - axis_started
        candidate_rows.extend(
            {
                "spatial": str(spatial),
                "penalty": str(row["penalty"]),
                "C": float(row["C"]),
                "n_iter": int(row["n_iter"]),
                "estimator_count": int(row["estimator_count"]),
                "n_iter_by_class": json.loads(str(row["n_iter_by_class_json"])),
            }
            for row in selected.grid
        )

        # A cheap repeated one-candidate sentinel verifies deterministic
        # explicit OVR/liblinear state without doubling the full grid smoke.
        repeats = [
            select_multiclass_logistic(
                matrix[train_indices],
                labels[train_indices],
                matrix[validation_indices],
                labels[validation_indices],
                classes=SMOKE_CLASSES,
                penalties=("l2",),
                c_grid=(0.1,),
                random_state=SEED,
            )
            for _ in range(2)
        ]
        fingerprints = [
            {
                "n_iter": int(repeat.grid[0]["n_iter"]),
                "n_iter_by_class": json.loads(
                    str(repeat.grid[0]["n_iter_by_class_json"])
                ),
                "coef_sha256": _array_sha256(repeat.model.coef_),
                "intercept_sha256": _array_sha256(repeat.model.intercept_),
            }
            for repeat in repeats
        ]
        if fingerprints[0] != fingerprints[1]:
            raise RuntimeError(
                f"{spatial} deterministic OVR/liblinear sentinel drifted"
            )
        sentinel_rows.append({"spatial": str(spatial), **fingerprints[0]})

        # Metric-free Ridge smoke for the v3 static-FTV numerical path.  The
        # continuous values are synthesized only from the representation rank;
        # no fold, clinical, radiomics, or outcome loader is reachable here.
        feature_index = int(
            np.argmax(np.std(matrix[train_indices], axis=0, dtype=np.float64))
        )
        signal = matrix[:, feature_index]
        order = np.argsort(np.argsort(signal, kind="stable"), kind="stable")
        synthetic_continuous = np.expm1(
            3.0 * order.astype(np.float64) / max(1, expected_n - 1)
        )
        continuous_digests[str(spatial)] = _array_sha256(synthetic_continuous)
        ridge_started = time.perf_counter()
        selected_ridge = select_ridge(
            matrix[train_indices],
            synthetic_continuous[train_indices],
            matrix[validation_indices],
            synthetic_continuous[validation_indices],
            alphas=RIDGE_ALPHAS,
            target_transform="log1p",
        )
        reserved_matrix = np.ascontiguousarray(matrix[reserved_indices].copy())
        train_scale = float(np.std(matrix[train_indices, feature_index], dtype=np.float64))
        reserved_matrix[0, feature_index] = float(
            np.max(matrix[train_indices, feature_index]) + max(train_scale, 1.0) * 1e6
        )
        reserved_standardized = selected_ridge.x_scaler.transform(reserved_matrix)
        reserved_raw = selected_ridge.y_scaler.inverse_transform(
            np.asarray(
                selected_ridge.model.predict(reserved_standardized), dtype=np.float64
            ).reshape(-1, 1)
        ).reshape(-1)
        reserved_bounded, clipped = _bound_transformed_ridge_predictions(
            reserved_raw,
            target_transform="log1p",
            train_transformed_min=selected_ridge.train_transformed_min,
            train_transformed_max=selected_ridge.train_transformed_max,
        )
        reserved_natural = _target_inverse(reserved_bounded, "log1p")
        if not np.isfinite(reserved_natural).all():
            raise RuntimeError("synthetic Ridge smoke emitted a non-finite value")
        if not bool(clipped[0]):
            raise RuntimeError(
                f"{spatial} synthetic Ridge smoke did not exercise train-bound clipping"
            )
        ridge_reserved_clipped[str(spatial)] = int(np.count_nonzero(clipped))
        ridge_elapsed_by_axis[str(spatial)] = time.perf_counter() - ridge_started
        ridge_candidate_rows.extend(
            {
                "spatial": str(spatial),
                "alpha": float(row["alpha"]),
                "n_iter": int(row["n_iter"]),
                "validation_predictions_clipped": int(
                    row["validation_predictions_clipped"]
                ),
            }
            for row in selected_ridge.grid
        )

    expected_candidates = len(tuple(spatial_axes)) * len(set(penalties)) * len(set(c_grid))
    if len(candidate_rows) != expected_candidates:
        raise AssertionError("synthetic smoke candidate count drifted")
    n_iter_values = np.asarray(
        [
            value
            for row in candidate_rows
            for values in row["n_iter_by_class"].values()
            for value in values
        ],
        dtype=np.int64,
    )
    if len(n_iter_values) != expected_candidates * len(SMOKE_CLASSES):
        raise AssertionError("synthetic smoke underlying-estimator count drifted")
    if np.any(n_iter_values < 0) or np.any(n_iter_values >= LOGISTIC_MAX_ITER):
        raise RuntimeError("synthetic smoke accepted an invalid/nonconverged candidate")
    expected_ridge_candidates = len(tuple(spatial_axes)) * len(RIDGE_ALPHAS)
    if len(ridge_candidate_rows) != expected_ridge_candidates:
        raise AssertionError("synthetic Ridge smoke candidate count drifted")
    ridge_n_iter = np.asarray(
        [row["n_iter"] for row in ridge_candidate_rows], dtype=np.int64
    )
    if np.any(ridge_n_iter < 0) or np.any(ridge_n_iter >= RIDGE_MAX_ITER):
        raise RuntimeError("synthetic Ridge smoke accepted a nonconverged candidate")
    label_digest = _array_sha256(labels.astype("U32"))
    receipt: dict[str, object] = {
        "schema_version": 1,
        "receipt_kind": SMOKE_RECEIPT_KIND,
        "status": "PASS",
        "feature_sha256": observed_sha256,
        "model_name": asset.model_name,
        "feature_shape": list(asset.representation.shape),
        "feature_dtype": str(asset.representation.dtype),
        "spatial_axes": list(spatial_axes),
        "decision_point": "T0",
        "synthetic_label_algorithm": "seed2026_balanced_four_class_numpy_permutation",
        "synthetic_label_sha256": label_digest,
        "synthetic_class_counts": {
            value: int(np.sum(labels == value)) for value in SMOKE_CLASSES
        },
        "synthetic_split_algorithm": "row_index_mod5_fold0_reserved0_val1_train234",
        "smoke_fold": 0,
        "n_train": int(len(train_indices)),
        "n_val": int(len(validation_indices)),
        "n_reserved_test": int(len(reserved_indices)),
        "formal_manifest_read": False,
        "clinical_outcomes_read": False,
        "radiomics_read": False,
        "penalties": list(penalties),
        "C_grid": [float(value) for value in c_grid],
        "tol": MULTICLASS_LOGISTIC_TOL,
        "max_iter": LOGISTIC_MAX_ITER,
        "solver": "explicit_one_vs_rest_liblinear",
        "underlying_solver": "liblinear",
        "multiclass_strategy": "four_balanced_binary_ovr_sigmoid_then_row_normalize",
        "class_weight": "balanced",
        "random_state": SEED,
        "candidate_count": len(candidate_rows),
        "underlying_estimator_fit_count": int(len(n_iter_values)),
        "candidates": candidate_rows,
        "min_observed_n_iter": int(n_iter_values.min()),
        "max_observed_n_iter": int(n_iter_values.max()),
        "all_candidate_n_iter_nonnegative": True,
        "all_candidate_n_iter_strictly_less_than_max_iter": True,
        "synthetic_continuous_algorithm": (
            "per_axis_rank_of_max_train_variance_feature_mapped_by_expm1_0_to_3"
        ),
        "synthetic_continuous_sha256_by_axis": continuous_digests,
        "ridge_solver": "lsqr",
        "ridge_alphas": [float(value) for value in RIDGE_ALPHAS],
        "ridge_tol": RIDGE_TOL,
        "ridge_max_iter": RIDGE_MAX_ITER,
        "ridge_candidate_count": len(ridge_candidate_rows),
        "ridge_candidates": ridge_candidate_rows,
        "ridge_min_observed_n_iter": int(ridge_n_iter.min()),
        "ridge_max_observed_n_iter": int(ridge_n_iter.max()),
        "ridge_all_candidate_n_iter_nonnegative": True,
        "ridge_all_candidate_n_iter_strictly_less_than_max_iter": True,
        "ridge_prediction_bound_policy": (
            "log1p_only_outer_train_transformed_min_max_identity_unbounded"
        ),
        "ridge_reserved_rows_clipped_by_axis": ridge_reserved_clipped,
        "determinism_sentinels": sentinel_rows,
        "elapsed_seconds": time.perf_counter() - started,
        "elapsed_seconds_by_axis": elapsed_by_axis,
        "ridge_elapsed_seconds_by_axis": ridge_elapsed_by_axis,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "thread_environment": observed_environment,
        },
        "code_sha256": {
            "scripts/run_probes.py": file_sha256(Path(__file__)),
            "src/foundation_mri/evaluation.py": file_sha256(
                SRC_ROOT / "foundation_mri/evaluation.py"
            ),
        },
    }
    forbidden_fragments = (
        "patient_id",
        "y_true",
        "y_pred",
        "probability",
        "auroc",
        "auprc",
        "brier",
        "calibration",
    )
    serialized = json.dumps(receipt, sort_keys=True, allow_nan=False)
    if any(fragment in serialized.lower() for fragment in forbidden_fragments):
        raise AssertionError("synthetic smoke receipt contains a forbidden outcome/ID field")
    _write_json_exclusive(receipt, Path(receipt_path))
    return receipt


def _parse_cnn_feature_specs(
    specs: Iterable[str],
) -> dict[tuple[str, str], dict[int, Path]]:
    groups: dict[tuple[str, str], dict[int, Path]] = {}
    for raw in specs:
        parts = str(raw).split(",", maxsplit=3)
        if len(parts) != 4:
            raise ValueError(
                "--cnn-feature must use NAME,SPATIAL,FOLD,PATH (one entry per fold)"
            )
        name, spatial, fold_text, path_text = (part.strip() for part in parts)
        try:
            fold = int(fold_text)
        except ValueError as error:
            raise ValueError("--cnn-feature FOLD must be an integer") from error
        spatial = spatial.upper()
        if not name or spatial not in SPATIAL_AXES or fold not in FOLDS or not path_text:
            raise ValueError("--cnn-feature has an invalid name/spatial/fold/path")
        key = (name, spatial)
        if fold in groups.setdefault(key, {}):
            raise ValueError(f"duplicate current-CNN probe source for {key}/fold {fold}")
        groups[key][fold] = Path(path_text)
    for key, paths in groups.items():
        if set(paths) != set(FOLDS):
            raise ValueError(f"current-CNN probe source {key} must provide exactly five folds")
    return groups


def _parse_cnn_templates(
    specs: Iterable[str],
) -> dict[tuple[str, str], dict[int, Path]]:
    groups: dict[tuple[str, str], dict[int, Path]] = {}
    for raw in specs:
        parts = str(raw).split(",", maxsplit=2)
        if len(parts) != 3:
            raise ValueError("--cnn-template must use NAME,SPATIAL,PATH_WITH_{fold}")
        name, spatial, template = (part.strip() for part in parts)
        spatial = spatial.upper()
        key = (name, spatial)
        if (
            not name
            or spatial not in SPATIAL_AXES
            or "{fold}" not in template
            or key in groups
        ):
            raise ValueError("--cnn-template has an invalid or duplicate identity/template")
        groups[key] = {fold: Path(template.format(fold=fold)) for fold in FOLDS}
    return groups


def _merge_cnn_groups(
    explicit: dict[tuple[str, str], dict[int, Path]],
    templated: dict[tuple[str, str], dict[int, Path]],
) -> dict[tuple[str, str], dict[int, Path]]:
    overlap = sorted(set(explicit).intersection(templated))
    if overlap:
        raise ValueError(f"current-CNN probe source was specified twice: {overlap}")
    return {**explicit, **templated}


def _t0_matrices(source: ProbeSource):
    if source.is_constant():
        return timing_matrix(source.by_fold[FOLDS[0]], "T0")
    return lambda fold: timing_matrix(source.by_fold[fold], "T0")


def _visit_matrices(source: ProbeSource, visit_index: int):
    if source.is_constant():
        return np.ascontiguousarray(source.by_fold[FOLDS[0]][:, visit_index])
    return {
        fold: np.ascontiguousarray(source.by_fold[fold][:, visit_index]) for fold in FOLDS
    }


def _delta_matrices(source: ProbeSource, transition_index: int):
    def delta(values: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(
            values[:, transition_index + 1] - values[:, transition_index]
        )

    if source.is_constant():
        return delta(source.by_fold[FOLDS[0]])
    return {fold: delta(source.by_fold[fold]) for fold in FOLDS}


def _append_source_probes(
    source: ProbeSource,
    *,
    clinical,
    folds,
    radiomics,
    ftv: np.ndarray,
    cohort_size: int,
    radiomics_size: int,
    binary_results: list[EvaluationResult],
    subtype_results: list[EvaluationResult],
    ridge_results: list[EvaluationResult],
) -> None:
    t0 = _t0_matrices(source)
    for target_name, target in (("HR", clinical.hr), ("HER2", clinical.her2)):
        binary_results.append(
            evaluate_binary_cv(
                patient_ids=clinical.patient_ids,
                targets=target,
                fold_manifest=folds,
                matrices=t0,
                target_name=target_name,
                model_name=source.name,
                spatial=source.spatial,
                timing="T0",
                analysis_population=f"full_{cohort_size}",
            )
        )

    # The reliable four-class label is complete for all 808 patients.  T0 is
    # fixed prospectively to avoid a post-hoc phenotype timing search.
    subtype_results.append(
        evaluate_multiclass_cv(
            patient_ids=clinical.patient_ids,
            targets=clinical.subtype,
            classes=HR_HER2_SUBTYPES,
            fold_manifest=folds,
            matrices=t0,
            target_name="HR_HER2_subtype",
            model_name=source.name,
            spatial=source.spatial,
            timing="T0",
            analysis_population=f"full_{cohort_size}",
        )
    )

    complete = source.subset(clinical.patient_ids, radiomics.patient_ids)
    population = f"radiomics_complete_case_{radiomics_size}"
    for visit_index, endpoint in enumerate(("T0", "T1", "T2", "T3")):
        ridge_results.append(
            evaluate_ridge_cv(
                patient_ids=radiomics.patient_ids,
                targets=ftv[:, visit_index],
                fold_manifest=folds,
                matrices=_visit_matrices(complete, visit_index),
                target_name="FTV",
                model_name=source.name,
                spatial=source.spatial,
                task="static",
                endpoint=endpoint,
                analysis_population=population,
                target_transform="log1p",
            )
        )
    for transition_index, endpoint in enumerate(("T0-T1", "T1-T2", "T2-T3")):
        ridge_results.append(
            evaluate_ridge_cv(
                patient_ids=radiomics.patient_ids,
                targets=ftv[:, transition_index + 1] - ftv[:, transition_index],
                fold_manifest=folds,
                matrices=_delta_matrices(complete, transition_index),
                target_name="FTV",
                model_name=source.name,
                spatial=source.spatial,
                task="delta",
                endpoint=endpoint,
                analysis_population=population,
                target_transform="identity",
            )
        )


def _output_paths(output_root: Path) -> tuple[Path, ...]:
    return (
        output_root / "predictions/phenotype_predictions.private.csv",
        output_root / "metrics/phenotype_selection.private.csv",
        output_root / "metrics/phenotype_metrics.csv",
        output_root / "predictions/subtype_predictions.private.csv",
        output_root / "metrics/subtype_selection.private.csv",
        output_root / "metrics/subtype_metrics.csv",
        output_root / "predictions/ftv_probe_predictions.private.csv",
        output_root / "metrics/ftv_probe_selection.private.csv",
        output_root / "metrics/ftv_probe_metrics.csv",
    )


def run(
    args: argparse.Namespace, *, command_argv: Sequence[str] | None = None
) -> dict[str, int]:
    cnn_groups = _merge_cnn_groups(
        _parse_cnn_feature_specs(args.cnn_feature),
        _parse_cnn_templates(args.cnn_template),
    )
    if not args.foundation_feature and not cnn_groups:
        raise ValueError(
            "at least one --foundation-feature, --cnn-feature, or --cnn-template "
            "source is required for probes"
        )
    lock_receipt = None
    if not args.allow_unlocked_inputs:
        if args.overwrite:
            raise ValueError("formal v3 probes forbid --overwrite")
        if Path(args.output_dir).resolve() != EXPERIMENT_ROOT:
            raise ValueError("formal v3 probe output-dir must be the experiment root")
        if command_argv is None:
            raise ValueError("formal v3 probes require the exact command argv")
        lock_receipt = verify_formal_evaluation_lock(
            experiment_root=EXPERIMENT_ROOT,
            lock_path=args.evaluation_lock,
            foundation_features=args.foundation_feature,
            current_cnn_features=cnn_groups,
            fold_manifest=args.fold_manifest,
            clinical_labels=args.clinical_labels,
            radiomics=args.radiomics,
            expected_consumer="probe",
            command_argv=command_argv,
        )
        configure_metric_free_progress(FORMAL_PROGRESS_PATH)
        metric_free_progress(
            "formal_run_started",
            consumer="probe",
            lock_sha256=str(lock_receipt["lock_sha256"]),
        )
    fold_hash = clinical_hash = radiomics_hash = None
    if not args.allow_unlocked_inputs:
        from foundation_mri.data import (
            EXPECTED_CLINICAL_SHA256,
            EXPECTED_FOLD_MANIFEST_SHA256,
            EXPECTED_RADIOMICS_SHA256,
        )

        fold_hash = EXPECTED_FOLD_MANIFEST_SHA256
        clinical_hash = EXPECTED_CLINICAL_SHA256
        radiomics_hash = EXPECTED_RADIOMICS_SHA256
    folds = load_fold_manifest(
        args.fold_manifest,
        expected_n=args.cohort_size,
        expected_sha256=fold_hash,
    )
    clinical = load_clinical_labels(
        args.clinical_labels,
        expected_patient_ids=folds.patient_ids,
        expected_n=args.cohort_size,
        expected_sha256=clinical_hash,
    )
    if not np.array_equal(clinical.pcr, folds.labels):
        raise ValueError("clinical pCR labels disagree with the locked fold manifest")
    radiomics = load_radiomics_table(
        args.radiomics,
        cohort_patient_ids=clinical.patient_ids,
        expected_n=args.radiomics_size,
        expected_sha256=radiomics_hash,
    )
    ftv = radiomics.aligned_values(radiomics.patient_ids, ("ftv",))[:, :, 0]

    binary_results: list[EvaluationResult] = []
    subtype_results: list[EvaluationResult] = []
    ridge_results: list[EvaluationResult] = []
    source_keys: set[tuple[str, str]] = set()
    for feature_path in args.foundation_feature:
        asset = load_foundation_features(
            feature_path,
            expected_patient_ids=clinical.patient_ids,
            expected_n=args.cohort_size,
        )
        for spatial in SPATIAL_AXES:
            key = (asset.model_name, spatial)
            if key in source_keys:
                raise ValueError(f"duplicate foundation probe source: {key}")
            source_keys.add(key)
            representation = asset.spatial(spatial)
            _append_source_probes(
                ProbeSource(
                    asset.model_name,
                    spatial,
                    _constant_folds(representation),
                ),
                clinical=clinical,
                folds=folds,
                radiomics=radiomics,
                ftv=ftv,
                cohort_size=args.cohort_size,
                radiomics_size=args.radiomics_size,
                binary_results=binary_results,
                subtype_results=subtype_results,
                ridge_results=ridge_results,
            )
        del asset

    for (name, spatial), paths in sorted(cnn_groups.items()):
        key = (name, spatial)
        if key in source_keys:
            raise ValueError(f"duplicate probe source identity: {key}")
        source_keys.add(key)
        by_fold = {
            fold: load_current_cnn_features(
                paths[fold],
                fold=fold,
                expected_patient_ids=clinical.patient_ids,
                fold_manifest=folds,
                expected_labels=clinical.pcr,
                expected_n=args.cohort_size,
                model_name=name,
                spatial_axis=spatial,
            ).representation
            for fold in FOLDS
        }
        _append_source_probes(
            ProbeSource(name, spatial, by_fold),
            clinical=clinical,
            folds=folds,
            radiomics=radiomics,
            ftv=ftv,
            cohort_size=args.cohort_size,
            radiomics_size=args.radiomics_size,
            binary_results=binary_results,
            subtype_results=subtype_results,
            ridge_results=ridge_results,
        )

    binary_predictions = pd.concat(
        [result.predictions for result in binary_results], ignore_index=True
    )
    binary_selections = pd.concat(
        [result.selections for result in binary_results], ignore_index=True
    )
    binary_metrics = aggregate_binary_predictions(binary_predictions)
    subtype_predictions = pd.concat(
        [result.predictions for result in subtype_results], ignore_index=True
    )
    subtype_selections = pd.concat(
        [result.selections for result in subtype_results], ignore_index=True
    )
    subtype_metrics = aggregate_multiclass_predictions(subtype_predictions)
    ridge_predictions = pd.concat(
        [result.predictions for result in ridge_results], ignore_index=True
    )
    ridge_selections = pd.concat(
        [result.selections for result in ridge_results], ignore_index=True
    )
    ridge_metrics = aggregate_continuous_predictions(ridge_predictions)
    if binary_predictions.duplicated(
        ["target", "model", "spatial", "timing", "analysis_population", "patient_id"]
    ).any():
        raise ValueError("phenotype probe emitted duplicate patient predictions")
    if ridge_predictions.duplicated(
        ["target", "model", "spatial", "task", "endpoint", "analysis_population", "patient_id"]
    ).any():
        raise ValueError("FTV probe emitted duplicate patient predictions")
    if subtype_predictions.duplicated(
        ["target", "model", "spatial", "timing", "analysis_population", "patient_id"]
    ).any():
        raise ValueError("subtype probe emitted duplicate patient predictions")

    paths = _output_paths(Path(args.output_dir))
    existing = [path.name for path in paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"outputs already exist: {existing}")
    write_private_csv(binary_predictions, paths[0], overwrite=args.overwrite)
    write_private_csv(binary_selections, paths[1], overwrite=args.overwrite)
    write_public_csv(binary_metrics, paths[2], overwrite=args.overwrite)
    write_private_csv(subtype_predictions, paths[3], overwrite=args.overwrite)
    write_private_csv(subtype_selections, paths[4], overwrite=args.overwrite)
    write_public_csv(subtype_metrics, paths[5], overwrite=args.overwrite)
    write_private_csv(ridge_predictions, paths[6], overwrite=args.overwrite)
    write_private_csv(ridge_selections, paths[7], overwrite=args.overwrite)
    write_public_csv(ridge_metrics, paths[8], overwrite=args.overwrite)
    summary = {
        "probe_spatial_sources": len(source_keys),
        "phenotype_prediction_rows": int(len(binary_predictions)),
        "subtype_prediction_rows": int(len(subtype_predictions)),
        "ftv_prediction_rows": int(len(ridge_predictions)),
        "public_metric_rows": int(
            len(binary_metrics) + len(subtype_metrics) + len(ridge_metrics)
        ),
    }
    if lock_receipt is not None:
        metric_free_progress(
            "formal_artifacts_written", consumer="probe", artifact_count=9
        )
        configure_metric_free_progress(None)
        write_metric_free_run_provenance(
            experiment_root=EXPERIMENT_ROOT,
            lock_path=args.evaluation_lock,
            expected_consumer="probe",
            command_argv=command_argv,
            artifacts={
                "phenotype_predictions_private": paths[0],
                "phenotype_selection_private": paths[1],
                "phenotype_metrics_public": paths[2],
                "subtype_predictions_private": paths[3],
                "subtype_selection_private": paths[4],
                "subtype_metrics_public": paths[5],
                "ftv_predictions_private": paths[6],
                "ftv_selection_private": paths[7],
                "ftv_metrics_public": paths[8],
                "probe_progress_private": FORMAL_PROGRESS_PATH,
            },
            receipt_path=FORMAL_RECEIPT_PATH,
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--medicalnet-synthetic-smoke",
        type=Path,
        help=(
            "Outcome-blind v3 convergence/runtime smoke on the locked formal "
            "MedicalNet NPZ; bypasses every outcome loader."
        ),
    )
    parser.add_argument(
        "--smoke-receipt",
        type=Path,
        help="Exclusive metric-free JSON receipt for --medicalnet-synthetic-smoke.",
    )
    parser.add_argument("--clinical-labels", type=Path, default=DEFAULT_CLINICAL)
    parser.add_argument("--fold-manifest", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument(
        "--evaluation-lock", type=Path, default=DEFAULT_EVALUATION_LOCK
    )
    parser.add_argument("--radiomics", type=Path, default=DEFAULT_RADIOMICS)
    parser.add_argument(
        "--foundation-feature",
        type=Path,
        action="append",
        default=[],
        help="Unified foundation feature NPZ; repeat for each frozen encoder.",
    )
    parser.add_argument(
        "--cnn-feature",
        action="append",
        default=[],
        metavar="NAME,SPATIAL,FOLD,PATH",
        help="One fold-specific current-CNN feature; provide all five folds.",
    )
    parser.add_argument(
        "--cnn-template",
        action="append",
        default=[],
        metavar="NAME,SPATIAL,PATH_WITH_{fold}",
        help="Five-fold GAP0/LOCAL0 current-CNN path template.",
    )
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--cohort-size", type=int, default=COHORT_SIZE)
    parser.add_argument(
        "--radiomics-size", type=int, default=RADIOMICS_COMPLETE_CASE_SIZE
    )
    parser.add_argument("--allow-unlocked-inputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = tuple(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    if args.medicalnet_synthetic_smoke is not None:
        if args.smoke_receipt is None:
            raise ValueError("--smoke-receipt is required for synthetic smoke")
        if (
            args.foundation_feature
            or args.cnn_feature
            or args.cnn_template
            or args.allow_unlocked_inputs
            or args.overwrite
        ):
            raise ValueError("synthetic smoke cannot be combined with evaluation inputs/overrides")
        receipt = _run_medicalnet_synthetic_smoke(
            args.medicalnet_synthetic_smoke, args.smoke_receipt
        )
        print(
            "MedicalNet synthetic smoke PASS: "
            f"candidates={receipt['candidate_count']}, "
            f"max_n_iter={receipt['max_observed_n_iter']}, "
            f"elapsed_seconds={float(receipt['elapsed_seconds']):.3f}"
        )
        return 0
    if args.smoke_receipt is not None:
        raise ValueError("--smoke-receipt requires --medicalnet-synthetic-smoke")
    try:
        summary = run(args, command_argv=raw_argv)
    finally:
        configure_metric_free_progress(None)
    print(
        "probe evaluation complete: "
        + ", ".join(f"{key}={value}" for key, value in summary.items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

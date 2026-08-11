"""DINOv3-only post-hoc evaluation on the published foundation-MRI protocol.

This module deliberately contains no estimator implementation.  It imports the
hash-pinned ``foundation_mri.data`` and ``foundation_mri.evaluation`` modules
from the published sibling experiment and only assembles the DINOv3-specific
design matrix.  In particular it never regenerates clinical-only, radiomics-
only, DINO v1, MedicalNet, GAP0, or LOCAL0 rows.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .paths import BASE_SOURCE_ROOT
from .locking import write_producer_receipt


# The parent package is a committed, hash-pinned runtime dependency.  The
# formal lock verifies its bytes before the CLI imports this module.
if str(BASE_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASE_SOURCE_ROOT))

from foundation_mri.data import (  # noqa: E402
    COHORT_SIZE,
    FOLDS,
    HR_HER2_SUBTYPES,
    RADIOMICS_COMPLETE_CASE_SIZE,
    SPATIAL_AXES,
    ClinicalTable,
    FoldManifest,
    FoundationFeatureAsset,
    load_clinical_labels,
    load_fold_manifest,
    load_foundation_features,
    load_radiomics_table,
)
from foundation_mri.evaluation import (  # noqa: E402
    ClinicalEncoder,
    DECISION_POINTS,
    EvaluationResult,
    aggregate_binary_predictions,
    aggregate_continuous_predictions,
    aggregate_multiclass_predictions,
    configure_metric_free_progress,
    evaluate_binary_cv,
    evaluate_multiclass_cv,
    evaluate_ridge_cv,
    metric_free_progress,
    timing_matrix,
    write_private_csv,
    write_public_csv,
)


MODEL_NAME = "dinov3_vitb16_lvd1689m_posthoc"
FEATURE_DIM = 1_536

BASELINE_IDENTITY_COUNT = 36
BASELINE_PREDICTION_ROWS = 18_696
BASELINE_SELECTION_ROWS = 180
BASELINE_PUBLIC_ROWS = 72

PHENOTYPE_IDENTITY_COUNT = 4
SUBTYPE_IDENTITY_COUNT = 2
FTV_IDENTITY_COUNT = 14
PROBE_IDENTITY_COUNT = 20
PHENOTYPE_PREDICTION_ROWS = 3_232
SUBTYPE_PREDICTION_ROWS = 1_616
FTV_PREDICTION_ROWS = 5_250
PHENOTYPE_SELECTION_ROWS = 20
SUBTYPE_SELECTION_ROWS = 10
FTV_SELECTION_ROWS = 70
PHENOTYPE_PUBLIC_ROWS = 8
SUBTYPE_PUBLIC_ROWS = 4
FTV_PUBLIC_ROWS = 28

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_BASELINE_COUNTS = {
    "pcr_identities": BASELINE_IDENTITY_COUNT,
    "pcr_prediction_rows": BASELINE_PREDICTION_ROWS,
    "pcr_selection_rows": BASELINE_SELECTION_ROWS,
    "pcr_public_rows": BASELINE_PUBLIC_ROWS,
}
_PROBE_COUNTS = {
    "phenotype_identities": PHENOTYPE_IDENTITY_COUNT,
    "phenotype_prediction_rows": PHENOTYPE_PREDICTION_ROWS,
    "phenotype_selection_rows": PHENOTYPE_SELECTION_ROWS,
    "phenotype_public_rows": PHENOTYPE_PUBLIC_ROWS,
    "subtype_identities": SUBTYPE_IDENTITY_COUNT,
    "subtype_prediction_rows": SUBTYPE_PREDICTION_ROWS,
    "subtype_selection_rows": SUBTYPE_SELECTION_ROWS,
    "subtype_public_rows": SUBTYPE_PUBLIC_ROWS,
    "ftv_identities": FTV_IDENTITY_COUNT,
    "ftv_prediction_rows": FTV_PREDICTION_ROWS,
    "ftv_selection_rows": FTV_SELECTION_ROWS,
    "ftv_public_rows": FTV_PUBLIC_ROWS,
}

@dataclass(frozen=True)
class ImagingSource:
    """One GLOBAL/LOCAL representation, constant across outer folds."""

    name: str
    spatial: str
    by_fold: Mapping[int, np.ndarray]

    def subset(
        self, canonical_ids: Sequence[str], requested_ids: Sequence[str]
    ) -> "ImagingSource":
        lookup = {str(patient_id): index for index, patient_id in enumerate(canonical_ids)}
        requested = tuple(str(patient_id) for patient_id in requested_ids)
        unknown = sorted(set(requested).difference(lookup))
        if unknown:
            raise ValueError(f"imaging subset contains {len(unknown)} unknown patients")
        indices = np.asarray([lookup[patient_id] for patient_id in requested], dtype=np.int64)
        matrix = np.ascontiguousarray(self.by_fold[FOLDS[0]][indices])
        return ImagingSource(self.name, self.spatial, _constant_folds(matrix))


@dataclass(frozen=True)
class BaselineIdentity:
    model_name: str
    spatial: str
    timing: str
    population_kind: str
    include_clinical: bool
    include_ftv: bool


@dataclass(frozen=True)
class ProbeIdentity:
    family: str
    spatial: str
    target: str
    endpoint: str


@dataclass(frozen=True)
class _LoadedInputs:
    folds: FoldManifest
    clinical: ClinicalTable
    radiomics: Any
    sources: tuple[ImagingSource, ImagingSource]
    ftv: np.ndarray


def _constant_folds(values: np.ndarray) -> dict[int, np.ndarray]:
    return {fold: values for fold in FOLDS}


def baseline_identities(
    *, cohort_size: int = COHORT_SIZE, radiomics_size: int = RADIOMICS_COMPLETE_CASE_SIZE
) -> tuple[BaselineIdentity, ...]:
    """Return the exact 36-cell DINOv3 pCR design without reading data."""

    identities: list[BaselineIdentity] = []
    full_population = f"full_{cohort_size}"
    paired_population = f"radiomics_complete_case_{radiomics_size}"
    for spatial in SPATIAL_AXES:
        for timing in DECISION_POINTS:
            identities.extend(
                (
                    BaselineIdentity(
                        f"{MODEL_NAME}_mri_only",
                        spatial,
                        timing,
                        full_population,
                        False,
                        False,
                    ),
                    BaselineIdentity(
                        f"{MODEL_NAME}_mri_clinical",
                        spatial,
                        timing,
                        full_population,
                        True,
                        False,
                    ),
                )
            )
            for suffix, include_clinical, include_ftv in (
                ("mri_only_paired", False, False),
                ("mri_clinical_paired", True, False),
                ("mri_ftv", False, True),
                ("mri_clinical_ftv", True, True),
            ):
                identities.append(
                    BaselineIdentity(
                        f"{MODEL_NAME}_{suffix}",
                        spatial,
                        timing,
                        paired_population,
                        include_clinical,
                        include_ftv,
                    )
                )
    output = tuple(identities)
    if len(output) != BASELINE_IDENTITY_COUNT or len(set(output)) != len(output):
        raise AssertionError("DINOv3 baseline identity contract drifted")
    return output


def probe_identities() -> tuple[ProbeIdentity, ...]:
    """Return the exact 20-cell phenotype/subtype/FTV probe contract."""

    identities: list[ProbeIdentity] = []
    for spatial in SPATIAL_AXES:
        identities.extend(
            ProbeIdentity("phenotype", spatial, target, "T0")
            for target in ("HR", "HER2")
        )
        identities.append(
            ProbeIdentity("subtype", spatial, "HR_HER2_subtype", "T0")
        )
        identities.extend(
            ProbeIdentity("ftv_static", spatial, "FTV", endpoint)
            for endpoint in ("T0", "T1", "T2", "T3")
        )
        identities.extend(
            ProbeIdentity("ftv_delta", spatial, "FTV", endpoint)
            for endpoint in ("T0-T1", "T1-T2", "T2-T3")
        )
    output = tuple(identities)
    if len(output) != PROBE_IDENTITY_COUNT or len(set(output)) != len(output):
        raise AssertionError("DINOv3 probe identity contract drifted")
    return output


def _load_dinov3_asset(
    path: str | Path, *, expected_patient_ids: Sequence[str], expected_n: int
) -> FoundationFeatureAsset:
    asset = load_foundation_features(
        path, expected_patient_ids=expected_patient_ids, expected_n=expected_n
    )
    if asset.model_name != MODEL_NAME:
        raise ValueError(f"foundation feature model must be exactly {MODEL_NAME}")
    if asset.representation.shape != (expected_n, 4, 2, FEATURE_DIM):
        raise ValueError(
            f"DINOv3 representation must be float32 [{expected_n},4,2,{FEATURE_DIM}]"
        )
    if asset.representation.dtype != np.float32 or not np.isfinite(
        asset.representation
    ).all():
        raise FloatingPointError("DINOv3 representation must be finite float32")
    required_digests = (
        asset.checkpoint_sha256,
        asset.extraction_signature_sha256,
        asset.canonical_patient_order_sha256,
    )
    if any(value is None for value in required_digests):
        raise ValueError("DINOv3 feature must embed checkpoint/extraction/order digests")
    return asset


def _load_inputs(
    *,
    feature_path: str | Path,
    fold_manifest_path: str | Path,
    clinical_path: str | Path,
    radiomics_path: str | Path,
    cohort_size: int,
    radiomics_size: int,
) -> _LoadedInputs:
    folds = load_fold_manifest(fold_manifest_path, expected_n=cohort_size)
    clinical = load_clinical_labels(
        clinical_path,
        expected_patient_ids=folds.patient_ids,
        expected_n=cohort_size,
    )
    if not np.array_equal(clinical.pcr, folds.labels):
        raise ValueError("clinical pCR labels disagree with the locked fold manifest")
    asset = _load_dinov3_asset(
        feature_path,
        expected_patient_ids=clinical.patient_ids,
        expected_n=cohort_size,
    )
    sources = tuple(
        ImagingSource(
            MODEL_NAME,
            spatial,
            _constant_folds(asset.spatial(spatial)),
        )
        for spatial in SPATIAL_AXES
    )
    radiomics = load_radiomics_table(
        radiomics_path,
        cohort_patient_ids=clinical.patient_ids,
        expected_n=radiomics_size,
    )
    ftv = radiomics.aligned_values(radiomics.patient_ids, ("ftv",))[:, :, 0]
    if ftv.shape != (radiomics_size, 4) or not np.isfinite(ftv).all():
        raise ValueError("locked complete-case FTV matrix must be finite [375,4]")
    return _LoadedInputs(folds, clinical, radiomics, sources, ftv)


def _clinical_matrices(
    clinical: ClinicalTable, folds: FoldManifest
) -> dict[int, np.ndarray]:
    matrices: dict[int, np.ndarray] = {}
    for fold in FOLDS:
        roles = folds.roles(fold, clinical.patient_ids)
        encoder = ClinicalEncoder.fit(clinical, np.flatnonzero(roles == "train"))
        matrices[fold] = encoder.transform(clinical)
    return matrices


def _evaluate_binary_identity(
    *,
    clinical: ClinicalTable,
    folds: FoldManifest,
    components: Sequence[Mapping[int, np.ndarray]],
    clinical_by_fold: Mapping[int, np.ndarray] | None,
    identity: BaselineIdentity,
) -> EvaluationResult:
    all_constant = all(
        all(component[fold] is component[FOLDS[0]] for fold in FOLDS[1:])
        for component in components
    )
    common = (
        [timing_matrix(component[FOLDS[0]], identity.timing) for component in components]
        if all_constant
        else []
    )

    def matrix_for_fold(fold: int) -> np.ndarray:
        parts = (
            list(common)
            if all_constant
            else [timing_matrix(component[fold], identity.timing) for component in components]
        )
        if clinical_by_fold is not None:
            parts.append(clinical_by_fold[fold])
        if not parts:
            raise AssertionError("DINOv3 identity unexpectedly has no features")
        return (
            np.ascontiguousarray(parts[0])
            if len(parts) == 1
            else np.ascontiguousarray(np.concatenate(parts, axis=1))
        )

    matrices: Any
    if all_constant and clinical_by_fold is None and len(common) == 1:
        matrices = common[0]
    else:
        matrices = matrix_for_fold
    return evaluate_binary_cv(
        patient_ids=clinical.patient_ids,
        targets=clinical.pcr,
        fold_manifest=folds,
        matrices=matrices,
        target_name="pCR",
        model_name=identity.model_name,
        spatial=identity.spatial,
        timing=identity.timing,
        analysis_population=identity.population_kind,
        require_manifest_pcr_match=True,
    )


def _t0_matrix(source: ImagingSource) -> np.ndarray:
    return timing_matrix(source.by_fold[FOLDS[0]], "T0")


def _visit_matrix(source: ImagingSource, visit_index: int) -> np.ndarray:
    return np.ascontiguousarray(source.by_fold[FOLDS[0]][:, visit_index])


def _delta_matrix(source: ImagingSource, transition_index: int) -> np.ndarray:
    values = source.by_fold[FOLDS[0]]
    return np.ascontiguousarray(
        values[:, transition_index + 1] - values[:, transition_index]
    )


def _output_paths(output_root: Path, consumer: str) -> dict[str, Path]:
    if consumer == "baseline":
        return {
            "baseline_predictions_private": output_root
            / "predictions/dinov3_baseline_predictions.private.csv",
            "baseline_selection_private": output_root
            / "metrics/dinov3_baseline_selection.private.csv",
            "baseline_metrics_public": output_root
            / "metrics/dinov3_baseline_metrics.csv",
            "progress_private": output_root
            / "logs/dinov3_baseline.progress.private.jsonl",
            "receipt_private": output_root
            / "metrics/baseline_run.private.provenance.json",
        }
    if consumer == "probe":
        return {
            "phenotype_predictions_private": output_root
            / "predictions/dinov3_phenotype_predictions.private.csv",
            "phenotype_selection_private": output_root
            / "metrics/dinov3_phenotype_selection.private.csv",
            "phenotype_metrics_public": output_root
            / "metrics/dinov3_phenotype_metrics.csv",
            "subtype_predictions_private": output_root
            / "predictions/dinov3_subtype_predictions.private.csv",
            "subtype_selection_private": output_root
            / "metrics/dinov3_subtype_selection.private.csv",
            "subtype_metrics_public": output_root
            / "metrics/dinov3_subtype_metrics.csv",
            "ftv_predictions_private": output_root
            / "predictions/dinov3_ftv_predictions.private.csv",
            "ftv_selection_private": output_root
            / "metrics/dinov3_ftv_selection.private.csv",
            "ftv_metrics_public": output_root / "metrics/dinov3_ftv_metrics.csv",
            "progress_private": output_root
            / "logs/dinov3_probe.progress.private.jsonl",
            "receipt_private": output_root
            / "metrics/probe_run.private.provenance.json",
        }
    raise ValueError("consumer must be baseline or probe")


def _assert_targets_absent(paths: Mapping[str, Path]) -> None:
    existing = sorted(path.name for path in paths.values() if os.path.lexists(path))
    if existing:
        raise FileExistsError(f"DINOv3 formal outputs already exist: {existing}")


def _validate_formal_lock_receipt(
    lock_receipt: Mapping[str, Any],
    *,
    consumer: str,
    command_argv: Sequence[str],
) -> str:
    """Bind a producer to the verified exact-empty command before data load."""

    if tuple(str(value) for value in command_argv):
        raise ValueError("formal DINOv3 producer requires an exact empty argv")
    if lock_receipt.get("consumer") != consumer:
        raise ValueError("evaluation-lock consumer identity drifted")
    digest = str(lock_receipt.get("lock_sha256", ""))
    if not _DIGEST.fullmatch(digest):
        raise ValueError("evaluation-lock SHA-256 is missing or malformed")
    expected_counts = lock_receipt.get("expected_counts")
    required = _BASELINE_COUNTS if consumer == "baseline" else _PROBE_COUNTS
    if not isinstance(expected_counts, Mapping) or any(
        expected_counts.get(key) != value for key, value in required.items()
    ):
        raise ValueError(f"evaluation-lock {consumer} count contract drifted")
    return digest


def _assert_mode(path: Path, expected: int) -> None:
    if (path.stat().st_mode & 0o777) != expected:
        raise PermissionError(
            f"artifact mode drifted for {path.name}: expected {oct(expected)}"
        )


def run_baseline_evaluation(
    *,
    feature_path: str | Path,
    fold_manifest_path: str | Path,
    clinical_path: str | Path,
    radiomics_path: str | Path,
    output_root: str | Path,
    lock_receipt: Mapping[str, Any],
    command_argv: Sequence[str],
    cohort_size: int = COHORT_SIZE,
    radiomics_size: int = RADIOMICS_COMPLETE_CASE_SIZE,
) -> dict[str, int]:
    """Run and exclusively publish the 36 DINOv3 pCR identities."""

    if (cohort_size, radiomics_size) != (COHORT_SIZE, RADIOMICS_COMPLETE_CASE_SIZE):
        raise ValueError("formal DINOv3 populations must remain exactly 808/375")
    lock_sha256 = _validate_formal_lock_receipt(
        lock_receipt, consumer="baseline", command_argv=command_argv
    )
    root = Path(output_root)
    paths = _output_paths(root, "baseline")
    _assert_targets_absent(paths)
    configure_metric_free_progress(paths["progress_private"])
    try:
        metric_free_progress(
            "formal_run_started",
            consumer="baseline",
            model_name=MODEL_NAME,
            lock_sha256=lock_sha256,
        )
        loaded = _load_inputs(
            feature_path=feature_path,
            fold_manifest_path=fold_manifest_path,
            clinical_path=clinical_path,
            radiomics_path=radiomics_path,
            cohort_size=cohort_size,
            radiomics_size=radiomics_size,
        )
        source_by_spatial = {source.spatial: source for source in loaded.sources}
        complete_clinical = loaded.clinical.subset(loaded.radiomics.patient_ids)
        full_clinical = _clinical_matrices(loaded.clinical, loaded.folds)
        paired_clinical = _clinical_matrices(complete_clinical, loaded.folds)
        ftv = _constant_folds(
            loaded.radiomics.aligned_values(
                loaded.radiomics.patient_ids, ("ftv",)
            )
        )
        results: list[EvaluationResult] = []
        for identity in baseline_identities(
            cohort_size=cohort_size, radiomics_size=radiomics_size
        ):
            source = source_by_spatial[identity.spatial]
            if identity.population_kind == f"full_{cohort_size}":
                clinical = loaded.clinical
                imaging = source
                clinical_matrices = full_clinical if identity.include_clinical else None
                components: list[Mapping[int, np.ndarray]] = [imaging.by_fold]
            else:
                clinical = complete_clinical
                imaging = source.subset(
                    loaded.clinical.patient_ids, loaded.radiomics.patient_ids
                )
                clinical_matrices = paired_clinical if identity.include_clinical else None
                components = [imaging.by_fold]
                if identity.include_ftv:
                    components.append(ftv)
            results.append(
                _evaluate_binary_identity(
                    clinical=clinical,
                    folds=loaded.folds,
                    components=components,
                    clinical_by_fold=clinical_matrices,
                    identity=identity,
                )
            )
        if len(results) != BASELINE_IDENTITY_COUNT:
            raise AssertionError("DINOv3 baseline emitted the wrong identity count")
        predictions = pd.concat(
            [result.predictions for result in results], ignore_index=True
        )
        selections = pd.concat(
            [result.selections for result in results], ignore_index=True
        )
        metrics = aggregate_binary_predictions(predictions)
        if (
            len(predictions) != BASELINE_PREDICTION_ROWS
            or len(selections) != BASELINE_SELECTION_ROWS
            or len(metrics) != BASELINE_PUBLIC_ROWS
        ):
            raise AssertionError("DINOv3 baseline output row contract drifted")
        identity_columns = [
            "target",
            "model",
            "spatial",
            "timing",
            "analysis_population",
        ]
        if predictions.duplicated([*identity_columns, "patient_id"]).any():
            raise ValueError("DINOv3 baseline duplicated a patient identity")
        if selections.duplicated([*identity_columns, "fold"]).any():
            raise ValueError("DINOv3 baseline duplicated a selection identity")
        write_private_csv(predictions, paths["baseline_predictions_private"])
        write_private_csv(selections, paths["baseline_selection_private"])
        write_public_csv(metrics, paths["baseline_metrics_public"])
        metric_free_progress(
            "formal_artifacts_written", consumer="baseline", artifact_count=3
        )
    finally:
        configure_metric_free_progress(None)
    _assert_mode(paths["baseline_predictions_private"], 0o600)
    _assert_mode(paths["baseline_selection_private"], 0o600)
    _assert_mode(paths["baseline_metrics_public"], 0o644)
    _assert_mode(paths["progress_private"], 0o600)
    counts = dict(_BASELINE_COUNTS)
    write_producer_receipt(
        consumer="baseline",
        command_argv=command_argv,
        counts=counts,
        artifacts={
            "predictions": paths["baseline_predictions_private"],
            "selection": paths["baseline_selection_private"],
            "metrics": paths["baseline_metrics_public"],
            "progress": paths["progress_private"],
        },
        receipt_path=paths["receipt_private"],
    )
    _assert_mode(paths["receipt_private"], 0o600)
    return counts


def run_probe_evaluation(
    *,
    feature_path: str | Path,
    fold_manifest_path: str | Path,
    clinical_path: str | Path,
    radiomics_path: str | Path,
    output_root: str | Path,
    lock_receipt: Mapping[str, Any],
    command_argv: Sequence[str],
    cohort_size: int = COHORT_SIZE,
    radiomics_size: int = RADIOMICS_COMPLETE_CASE_SIZE,
) -> dict[str, int]:
    """Run and exclusively publish the 20 DINOv3 representation probes."""

    if (cohort_size, radiomics_size) != (COHORT_SIZE, RADIOMICS_COMPLETE_CASE_SIZE):
        raise ValueError("formal DINOv3 populations must remain exactly 808/375")
    lock_sha256 = _validate_formal_lock_receipt(
        lock_receipt, consumer="probe", command_argv=command_argv
    )
    root = Path(output_root)
    paths = _output_paths(root, "probe")
    _assert_targets_absent(paths)
    configure_metric_free_progress(paths["progress_private"])
    try:
        metric_free_progress(
            "formal_run_started",
            consumer="probe",
            model_name=MODEL_NAME,
            lock_sha256=lock_sha256,
        )
        loaded = _load_inputs(
            feature_path=feature_path,
            fold_manifest_path=fold_manifest_path,
            clinical_path=clinical_path,
            radiomics_path=radiomics_path,
            cohort_size=cohort_size,
            radiomics_size=radiomics_size,
        )
        binary_results: list[EvaluationResult] = []
        subtype_results: list[EvaluationResult] = []
        ridge_results: list[EvaluationResult] = []
        population = f"radiomics_complete_case_{radiomics_size}"
        for source in loaded.sources:
            t0 = _t0_matrix(source)
            for target_name, target in (
                ("HR", loaded.clinical.hr),
                ("HER2", loaded.clinical.her2),
            ):
                binary_results.append(
                    evaluate_binary_cv(
                        patient_ids=loaded.clinical.patient_ids,
                        targets=target,
                        fold_manifest=loaded.folds,
                        matrices=t0,
                        target_name=target_name,
                        model_name=MODEL_NAME,
                        spatial=source.spatial,
                        timing="T0",
                        analysis_population=f"full_{cohort_size}",
                    )
                )
            subtype_results.append(
                evaluate_multiclass_cv(
                    patient_ids=loaded.clinical.patient_ids,
                    targets=loaded.clinical.subtype,
                    classes=HR_HER2_SUBTYPES,
                    fold_manifest=loaded.folds,
                    matrices=t0,
                    target_name="HR_HER2_subtype",
                    model_name=MODEL_NAME,
                    spatial=source.spatial,
                    timing="T0",
                    analysis_population=f"full_{cohort_size}",
                )
            )
            complete = source.subset(
                loaded.clinical.patient_ids, loaded.radiomics.patient_ids
            )
            for visit_index, endpoint in enumerate(("T0", "T1", "T2", "T3")):
                ridge_results.append(
                    evaluate_ridge_cv(
                        patient_ids=loaded.radiomics.patient_ids,
                        targets=loaded.ftv[:, visit_index],
                        fold_manifest=loaded.folds,
                        matrices=_visit_matrix(complete, visit_index),
                        target_name="FTV",
                        model_name=MODEL_NAME,
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
                        patient_ids=loaded.radiomics.patient_ids,
                        targets=(
                            loaded.ftv[:, transition_index + 1]
                            - loaded.ftv[:, transition_index]
                        ),
                        fold_manifest=loaded.folds,
                        matrices=_delta_matrix(complete, transition_index),
                        target_name="FTV",
                        model_name=MODEL_NAME,
                        spatial=source.spatial,
                        task="delta",
                        endpoint=endpoint,
                        analysis_population=population,
                        target_transform="identity",
                    )
                )
        if (
            len(binary_results) != PHENOTYPE_IDENTITY_COUNT
            or len(subtype_results) != SUBTYPE_IDENTITY_COUNT
            or len(ridge_results) != FTV_IDENTITY_COUNT
        ):
            raise AssertionError("DINOv3 probe identity contract drifted")
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
        observed = (
            len(binary_predictions),
            len(subtype_predictions),
            len(ridge_predictions),
            len(binary_selections),
            len(subtype_selections),
            len(ridge_selections),
            len(binary_metrics),
            len(subtype_metrics),
            len(ridge_metrics),
        )
        expected = (
            PHENOTYPE_PREDICTION_ROWS,
            SUBTYPE_PREDICTION_ROWS,
            FTV_PREDICTION_ROWS,
            PHENOTYPE_SELECTION_ROWS,
            SUBTYPE_SELECTION_ROWS,
            FTV_SELECTION_ROWS,
            PHENOTYPE_PUBLIC_ROWS,
            SUBTYPE_PUBLIC_ROWS,
            FTV_PUBLIC_ROWS,
        )
        if observed != expected:
            raise AssertionError(
                f"DINOv3 probe output row contract drifted: {observed} != {expected}"
            )
        if binary_predictions.duplicated(
            ["target", "model", "spatial", "timing", "analysis_population", "patient_id"]
        ).any():
            raise ValueError("DINOv3 phenotype probe duplicated a patient identity")
        if subtype_predictions.duplicated(
            ["target", "model", "spatial", "timing", "analysis_population", "patient_id"]
        ).any():
            raise ValueError("DINOv3 subtype probe duplicated a patient identity")
        if ridge_predictions.duplicated(
            [
                "target",
                "model",
                "spatial",
                "task",
                "endpoint",
                "analysis_population",
                "patient_id",
            ]
        ).any():
            raise ValueError("DINOv3 FTV probe duplicated a patient identity")
        if binary_selections.duplicated(
            ["target", "model", "spatial", "timing", "analysis_population", "fold"]
        ).any():
            raise ValueError("DINOv3 phenotype probe duplicated a selection identity")
        if subtype_selections.duplicated(
            ["target", "model", "spatial", "timing", "analysis_population", "fold"]
        ).any():
            raise ValueError("DINOv3 subtype probe duplicated a selection identity")
        if ridge_selections.duplicated(
            [
                "target",
                "model",
                "spatial",
                "task",
                "endpoint",
                "analysis_population",
                "fold",
            ]
        ).any():
            raise ValueError("DINOv3 FTV probe duplicated a selection identity")
        write_private_csv(binary_predictions, paths["phenotype_predictions_private"])
        write_private_csv(binary_selections, paths["phenotype_selection_private"])
        write_public_csv(binary_metrics, paths["phenotype_metrics_public"])
        write_private_csv(subtype_predictions, paths["subtype_predictions_private"])
        write_private_csv(subtype_selections, paths["subtype_selection_private"])
        write_public_csv(subtype_metrics, paths["subtype_metrics_public"])
        write_private_csv(ridge_predictions, paths["ftv_predictions_private"])
        write_private_csv(ridge_selections, paths["ftv_selection_private"])
        write_public_csv(ridge_metrics, paths["ftv_metrics_public"])
        metric_free_progress(
            "formal_artifacts_written", consumer="probe", artifact_count=9
        )
    finally:
        configure_metric_free_progress(None)
    for key, path in paths.items():
        if key == "receipt_private":
            continue
        _assert_mode(path, 0o644 if key.endswith("metrics_public") else 0o600)
    counts = dict(_PROBE_COUNTS)
    write_producer_receipt(
        consumer="probe",
        command_argv=command_argv,
        counts=counts,
        artifacts={
            "phenotype_predictions": paths["phenotype_predictions_private"],
            "phenotype_selection": paths["phenotype_selection_private"],
            "phenotype_metrics": paths["phenotype_metrics_public"],
            "subtype_predictions": paths["subtype_predictions_private"],
            "subtype_selection": paths["subtype_selection_private"],
            "subtype_metrics": paths["subtype_metrics_public"],
            "ftv_predictions": paths["ftv_predictions_private"],
            "ftv_selection": paths["ftv_selection_private"],
            "ftv_metrics": paths["ftv_metrics_public"],
            "progress": paths["progress_private"],
        },
        receipt_path=paths["receipt_private"],
    )
    _assert_mode(paths["receipt_private"], 0o600)
    return counts


__all__ = [
    "BASELINE_IDENTITY_COUNT",
    "BASELINE_PREDICTION_ROWS",
    "BASELINE_PUBLIC_ROWS",
    "BASELINE_SELECTION_ROWS",
    "FEATURE_DIM",
    "FTV_IDENTITY_COUNT",
    "FTV_PREDICTION_ROWS",
    "FTV_PUBLIC_ROWS",
    "FTV_SELECTION_ROWS",
    "MODEL_NAME",
    "PHENOTYPE_IDENTITY_COUNT",
    "PHENOTYPE_PREDICTION_ROWS",
    "PHENOTYPE_PUBLIC_ROWS",
    "PHENOTYPE_SELECTION_ROWS",
    "PROBE_IDENTITY_COUNT",
    "SUBTYPE_IDENTITY_COUNT",
    "SUBTYPE_PREDICTION_ROWS",
    "SUBTYPE_PUBLIC_ROWS",
    "SUBTYPE_SELECTION_ROWS",
    "baseline_identities",
    "probe_identities",
    "run_baseline_evaluation",
    "run_probe_evaluation",
]

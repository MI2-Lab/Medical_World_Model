#!/usr/bin/env python3
"""Run pCR baselines from frozen features and locked tabular inputs.

The script trains no image encoder.  It produces two identifier-bearing files
(``.private.csv``) and one public aggregate metrics table with no paths or IDs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
SRC_ROOT = EXPERIMENT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from foundation_mri.data import (  # noqa: E402
    COHORT_SIZE,
    FOLDS,
    RADIOMICS_COMPLETE_CASE_SIZE,
    ClinicalTable,
    FoldManifest,
    load_clinical_labels,
    load_current_cnn_features,
    load_fold_manifest,
    load_foundation_features,
    load_radiomics_table,
)
from foundation_mri.evaluation import (  # noqa: E402
    ClinicalEncoder,
    DECISION_POINTS,
    EvaluationResult,
    aggregate_binary_predictions,
    configure_metric_free_progress,
    evaluate_binary_cv,
    metric_free_progress,
    timing_matrix,
    write_private_csv,
    write_public_csv,
)
from foundation_mri.locking import (  # noqa: E402
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
DEFAULT_EVALUATION_LOCK = EXPERIMENT_ROOT / "configs/EVALUATION_LOCK.v2.json"
FORMAL_PROGRESS_PATH = EXPERIMENT_ROOT / "logs/baseline_v2.progress.private.jsonl"
FORMAL_RECEIPT_PATH = EXPERIMENT_ROOT / "metrics/baseline_v2_run.private.provenance.json"


@dataclass(frozen=True)
class ImagingSource:
    name: str
    spatial: str
    by_fold: Mapping[int, np.ndarray]

    def subset(self, canonical_ids: Sequence[str], requested_ids: Sequence[str]) -> "ImagingSource":
        lookup = {patient_id: index for index, patient_id in enumerate(canonical_ids)}
        unknown = sorted(set(requested_ids).difference(lookup))
        if unknown:
            raise ValueError(f"imaging subset contains {len(unknown)} unknown patients")
        indices = np.asarray([lookup[value] for value in requested_ids], dtype=np.int64)
        if all(self.by_fold[fold] is self.by_fold[FOLDS[0]] for fold in FOLDS[1:]):
            subset = np.ascontiguousarray(self.by_fold[FOLDS[0]][indices])
            by_fold = _constant_folds(subset)
        else:
            by_fold = {
                fold: np.ascontiguousarray(self.by_fold[fold][indices]) for fold in FOLDS
            }
        return ImagingSource(
            self.name,
            self.spatial,
            by_fold,
        )


def _constant_folds(values: np.ndarray) -> dict[int, np.ndarray]:
    return {fold: values for fold in FOLDS}


def _parse_cnn_feature_specs(
    specs: Iterable[str],
) -> dict[tuple[str, str], dict[int, Path]]:
    """Parse ``NAME,SPATIAL,FOLD,PATH`` current-CNN specifications."""

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
        if fold not in FOLDS or not name or spatial.upper() not in {"GLOBAL", "LOCAL"}:
            raise ValueError("--cnn-feature has an invalid name/spatial/fold")
        key = (name, spatial.upper())
        if fold in groups.setdefault(key, {}):
            raise ValueError(f"duplicate current-CNN feature for {key}/fold {fold}")
        groups[key][fold] = Path(path_text)
    for key, paths in groups.items():
        if set(paths) != set(FOLDS):
            raise ValueError(f"current-CNN source {key} must provide exactly five folds")
    return groups


def _parse_cnn_templates(
    specs: Iterable[str],
) -> dict[tuple[str, str], dict[int, Path]]:
    """Parse ``NAME,SPATIAL,PATH_WITH_{fold}`` templates."""

    groups: dict[tuple[str, str], dict[int, Path]] = {}
    for raw in specs:
        parts = str(raw).split(",", maxsplit=2)
        if len(parts) != 3:
            raise ValueError("--cnn-template must use NAME,SPATIAL,PATH_WITH_{fold}")
        name, spatial, template = (part.strip() for part in parts)
        if "{fold}" not in template:
            raise ValueError("--cnn-template path must contain the literal {fold}")
        key = (name, spatial.upper())
        if not name or key[1] not in {"GLOBAL", "LOCAL"} or key in groups:
            raise ValueError("--cnn-template has a duplicate/invalid name or spatial value")
        groups[key] = {fold: Path(template.format(fold=fold)) for fold in FOLDS}
    return groups


def _merge_cnn_groups(
    explicit: dict[tuple[str, str], dict[int, Path]],
    templated: dict[tuple[str, str], dict[int, Path]],
) -> dict[tuple[str, str], dict[int, Path]]:
    overlap = sorted(set(explicit).intersection(templated))
    if overlap:
        raise ValueError(f"current-CNN source was specified twice: {overlap}")
    return {**explicit, **templated}


def _clinical_matrices(clinical: ClinicalTable, folds: FoldManifest) -> dict[int, np.ndarray]:
    matrices: dict[int, np.ndarray] = {}
    for fold in FOLDS:
        roles = folds.roles(fold, clinical.patient_ids)
        encoder = ClinicalEncoder.fit(clinical, np.flatnonzero(roles == "train"))
        matrices[fold] = encoder.transform(clinical)
    return matrices


def _evaluate_design(
    *,
    clinical: ClinicalTable,
    folds: FoldManifest,
    visit_components: Sequence[Mapping[int, np.ndarray]],
    include_clinical: bool,
    model_name: str,
    spatial: str,
    population: str,
) -> list[EvaluationResult]:
    if not visit_components and not include_clinical:
        raise ValueError("baseline design has no features")
    clinical_by_fold = _clinical_matrices(clinical, folds) if include_clinical else {}
    results: list[EvaluationResult] = []
    for timing in DECISION_POINTS:
        all_constant = all(
            all(component[fold] is component[FOLDS[0]] for fold in FOLDS[1:])
            for component in visit_components
        )
        common_parts = (
            [timing_matrix(component[FOLDS[0]], timing) for component in visit_components]
            if all_constant
            else []
        )

        def matrix_for_fold(fold: int) -> np.ndarray:
            parts = (
                list(common_parts)
                if all_constant
                else [timing_matrix(component[fold], timing) for component in visit_components]
            )
            if include_clinical:
                parts.append(clinical_by_fold[fold])
            if len(parts) == 1:
                return parts[0]
            return np.ascontiguousarray(np.concatenate(parts, axis=1))

        matrices = (
            common_parts[0]
            if all_constant and not include_clinical and len(common_parts) == 1
            else matrix_for_fold
        )
        results.append(
            evaluate_binary_cv(
                patient_ids=clinical.patient_ids,
                targets=clinical.pcr,
                fold_manifest=folds,
                matrices=matrices,
                target_name="pCR",
                model_name=model_name,
                spatial=spatial,
                timing=timing,
                analysis_population=population,
                require_manifest_pcr_match=True,
            )
        )
    return results


def _append_design(
    destination: list[EvaluationResult],
    *,
    clinical: ClinicalTable,
    folds: FoldManifest,
    visit_components: Sequence[Mapping[int, np.ndarray]] = (),
    include_clinical: bool = False,
    model_name: str,
    spatial: str,
    population: str,
) -> None:
    destination.extend(
        _evaluate_design(
            clinical=clinical,
            folds=folds,
            visit_components=visit_components,
            include_clinical=include_clinical,
            model_name=model_name,
            spatial=spatial,
            population=population,
        )
    )


def _output_paths(output_root: Path) -> tuple[Path, Path, Path]:
    return (
        output_root / "predictions/baseline_predictions.private.csv",
        output_root / "metrics/baseline_selection.private.csv",
        output_root / "metrics/baseline_metrics.csv",
    )


def run(
    args: argparse.Namespace, *, command_argv: Sequence[str] | None = None
) -> dict[str, int]:
    cnn_groups = _merge_cnn_groups(
        _parse_cnn_feature_specs(args.cnn_feature),
        _parse_cnn_templates(args.cnn_template),
    )
    lock_receipt = None
    if not args.allow_unlocked_inputs:
        if args.overwrite:
            raise ValueError("formal v2 baseline forbids --overwrite")
        if Path(args.output_dir).resolve() != EXPERIMENT_ROOT:
            raise ValueError("formal v2 baseline output-dir must be the experiment root")
        if command_argv is None:
            raise ValueError("formal v2 baseline requires the exact command argv")
        lock_receipt = verify_formal_evaluation_lock(
            experiment_root=EXPERIMENT_ROOT,
            lock_path=args.evaluation_lock,
            foundation_features=args.foundation_feature,
            current_cnn_features=cnn_groups,
            fold_manifest=args.fold_manifest,
            clinical_labels=args.clinical_labels,
            radiomics=args.radiomics,
            expected_consumer="baseline",
            command_argv=command_argv,
        )
        configure_metric_free_progress(FORMAL_PROGRESS_PATH)
        metric_free_progress(
            "formal_run_started",
            consumer="baseline",
            lock_sha256=str(lock_receipt["lock_sha256"]),
        )
    fold_hash = None if args.allow_unlocked_inputs else None  # overwritten below for clarity
    clinical_hash = None if args.allow_unlocked_inputs else None
    radiomics_hash = None if args.allow_unlocked_inputs else None
    if not args.allow_unlocked_inputs:
        from foundation_mri.data import (  # local import keeps constants singular
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

    sources: list[ImagingSource] = []
    source_keys: set[tuple[str, str]] = set()
    for feature_path in args.foundation_feature:
        asset = load_foundation_features(
            feature_path,
            expected_patient_ids=clinical.patient_ids,
            expected_n=args.cohort_size,
        )
        for spatial in ("GLOBAL", "LOCAL"):
            key = (asset.model_name, spatial)
            if key in source_keys:
                raise ValueError(f"duplicate foundation source identity: {key}")
            source_keys.add(key)
            sources.append(
                ImagingSource(asset.model_name, spatial, _constant_folds(asset.spatial(spatial)))
            )
        del asset

    for (name, spatial), paths in sorted(cnn_groups.items()):
        key = (name, spatial)
        if key in source_keys:
            raise ValueError(f"duplicate imaging source identity: {key}")
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
        sources.append(ImagingSource(name, spatial, by_fold))

    results: list[EvaluationResult] = []
    _append_design(
        results,
        clinical=clinical,
        folds=folds,
        include_clinical=True,
        model_name="clinical_only",
        spatial="NONE",
        population=f"full_{args.cohort_size}",
    )
    for source in sources:
        _append_design(
            results,
            clinical=clinical,
            folds=folds,
            visit_components=(source.by_fold,),
            model_name=f"{source.name}_mri_only",
            spatial=source.spatial,
            population=f"full_{args.cohort_size}",
        )
        _append_design(
            results,
            clinical=clinical,
            folds=folds,
            visit_components=(source.by_fold,),
            include_clinical=True,
            model_name=f"{source.name}_mri_clinical",
            spatial=source.spatial,
            population=f"full_{args.cohort_size}",
        )

    if args.radiomics is not None:
        radiomics = load_radiomics_table(
            args.radiomics,
            cohort_patient_ids=clinical.patient_ids,
            expected_n=args.radiomics_size,
            expected_sha256=radiomics_hash,
        )
        complete_clinical = clinical.subset(radiomics.patient_ids)
        ftv = _constant_folds(radiomics.aligned_values(radiomics.patient_ids, ("ftv",)))
        all_radiomics = _constant_folds(radiomics.aligned_values(radiomics.patient_ids))
        population = f"radiomics_complete_case_{args.radiomics_size}"
        # Explicit paired comparators: the full-cohort rows above remain intact.
        _append_design(
            results,
            clinical=complete_clinical,
            folds=folds,
            include_clinical=True,
            model_name="clinical_only_paired",
            spatial="NONE",
            population=population,
        )
        for name, component, include_clinical in (
            ("ftv_only", ftv, False),
            ("radiomics_only", all_radiomics, False),
            ("clinical_ftv", ftv, True),
            ("clinical_radiomics", all_radiomics, True),
        ):
            _append_design(
                results,
                clinical=complete_clinical,
                folds=folds,
                visit_components=(component,),
                include_clinical=include_clinical,
                model_name=name,
                spatial="TABULAR",
                population=population,
            )
        for source in sources:
            paired = source.subset(clinical.patient_ids, radiomics.patient_ids)
            designs = (
                (f"{source.name}_mri_only_paired", (paired.by_fold,), False),
                (f"{source.name}_mri_clinical_paired", (paired.by_fold,), True),
                (f"{source.name}_mri_ftv", (paired.by_fold, ftv), False),
                (f"{source.name}_mri_clinical_ftv", (paired.by_fold, ftv), True),
            )
            for name, components, include_clinical in designs:
                _append_design(
                    results,
                    clinical=complete_clinical,
                    folds=folds,
                    visit_components=components,
                    include_clinical=include_clinical,
                    model_name=name,
                    spatial=source.spatial,
                    population=population,
                )

    predictions = pd.concat([result.predictions for result in results], ignore_index=True)
    selections = pd.concat([result.selections for result in results], ignore_index=True)
    metrics = aggregate_binary_predictions(predictions)
    identity = ["target", "model", "spatial", "timing", "analysis_population"]
    if predictions.duplicated([*identity, "patient_id"]).any():
        raise ValueError("baseline matrix emitted duplicate patient-level predictions")
    if selections.duplicated([*identity, "fold"]).any():
        raise ValueError("baseline matrix emitted duplicate selection rows")
    prediction_path, selection_path, metric_path = _output_paths(Path(args.output_dir))
    existing = [path.name for path in (prediction_path, selection_path, metric_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"outputs already exist: {existing}")
    write_private_csv(predictions, prediction_path, overwrite=args.overwrite)
    write_private_csv(selections, selection_path, overwrite=args.overwrite)
    write_public_csv(metrics, metric_path, overwrite=args.overwrite)
    summary = {
        "patient_prediction_rows": int(len(predictions)),
        "private_selection_rows": int(len(selections)),
        "public_metric_rows": int(len(metrics)),
        "imaging_sources": int(len(sources)),
    }
    if lock_receipt is not None:
        metric_free_progress(
            "formal_artifacts_written", consumer="baseline", artifact_count=3
        )
        configure_metric_free_progress(None)
        write_metric_free_run_provenance(
            experiment_root=EXPERIMENT_ROOT,
            lock_path=args.evaluation_lock,
            expected_consumer="baseline",
            command_argv=command_argv,
            artifacts={
                "baseline_predictions_private": prediction_path,
                "baseline_selection_private": selection_path,
                "baseline_metrics_public": metric_path,
                "baseline_progress_private": FORMAL_PROGRESS_PATH,
            },
            receipt_path=FORMAL_RECEIPT_PATH,
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinical-labels", type=Path, default=DEFAULT_CLINICAL)
    parser.add_argument("--fold-manifest", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument(
        "--evaluation-lock", type=Path, default=DEFAULT_EVALUATION_LOCK
    )
    parser.add_argument(
        "--foundation-feature",
        type=Path,
        action="append",
        default=[],
        help="Unified foundation feature NPZ; repeat for each preregistered encoder.",
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
        help="Five-fold current-CNN path template.",
    )
    parser.add_argument(
        "--radiomics",
        type=Path,
        default=DEFAULT_RADIOMICS,
        help="Set to an empty string only through the Python API to skip complete-case baselines.",
    )
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--cohort-size", type=int, default=COHORT_SIZE)
    parser.add_argument(
        "--radiomics-size", type=int, default=RADIOMICS_COMPLETE_CASE_SIZE
    )
    parser.add_argument(
        "--allow-unlocked-inputs",
        action="store_true",
        help="Explicit synthetic/prospective override of the audited input SHA locks.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = tuple(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    summary = run(args, command_argv=raw_argv)
    print(
        "baseline evaluation complete: "
        + ", ".join(f"{key}={value}" for key, value in summary.items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

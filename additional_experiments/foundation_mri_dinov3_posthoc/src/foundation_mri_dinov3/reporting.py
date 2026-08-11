"""Outcome-blind expansion and fail-closed public reporting for DINOv3.

This module is intentionally isolated from the locked
``foundation_mri_baselines`` experiment.  It imports that experiment's
``ComparisonSpec`` and paired patient-bootstrap implementation, but it never
changes any base file and never performs model selection.  The comparison
matrix is resolved from model identities alone and has an invariant size of
84 specifications (252 public metric rows).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .paths import (
    BASE_EXPERIMENT_ROOT,
    BASE_SOURCE_ROOT,
    EXPERIMENT_ROOT,
    REPOSITORY_ROOT,
)


if str(BASE_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASE_SOURCE_ROOT))

from foundation_mri.data import FOLDS, file_sha256  # noqa: E402
from foundation_mri.evaluation import (  # noqa: E402
    aggregate_binary_predictions,
    aggregate_continuous_predictions,
    aggregate_multiclass_predictions,
    ensure_public_safe,
)
from foundation_mri.reporting import (  # noqa: E402
    BINARY_IRLS_COLUMNS,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CI_LEVEL,
    ComparisonSpec,
    INFERENCE_SCOPE,
    _assert_public_matches,
    paired_bootstrap_comparisons,
)


MODEL_NAME = "dinov3_vitb16_lvd1689m_posthoc"
DINO_V1_MODEL = "dino_vitb16_imagenet1k"
DECISION_POINTS = ("T0", "T0-T1", "T0-T2")
SPATIAL_AXES = ("GLOBAL", "LOCAL")
METRICS = ("auroc", "auprc", "brier")
FULL_SIZE = 808
COMPLETE_SIZE = 375
FULL_POPULATION = f"full_{FULL_SIZE}"
COMPLETE_POPULATION = f"radiomics_complete_case_{COMPLETE_SIZE}"
SUMMARY_SCHEMA_VERSION = "foundation_mri_dinov3_posthoc_results_summary_v1"
REPORTING_MARKER_SCHEMA_VERSION = (
    "foundation_mri_dinov3_posthoc_reporting_provenance_v1"
)
COMPARISON_CONTRACT_VERSION = "foundation_mri_dinov3_comparison_contract_v1"
PRODUCER_RECEIPT_SCHEMA_VERSION = "foundation_mri_dinov3_producer_receipt_v1"

FAMILY_COUNTS = {
    "dinov3_vs_dinov1": 36,
    "local_vs_global": 18,
    "dinov3_vs_current_cnn_full": 12,
    "clinical_gain": 12,
    "beyond_ftv": 6,
}
EXPECTED_COMPARISON_COUNT = sum(FAMILY_COUNTS.values())
EXPECTED_PAIRED_METRIC_ROWS = EXPECTED_COMPARISON_COUNT * len(METRICS)

# The order is a frozen scientific order, not a result-dependent order.
_PCR_VARIANTS = (
    ("mri_only", "full"),
    ("mri_clinical", "full"),
    ("mri_only_paired", "complete"),
    ("mri_clinical_paired", "complete"),
    ("mri_ftv", "complete"),
    ("mri_clinical_ftv", "complete"),
)
_FULL_VARIANTS = ("mri_only", "mri_clinical")
_BINARY_GROUP = (
    "target",
    "model",
    "spatial",
    "timing",
    "analysis_population",
    "split_seed",
    "fold_manifest_sha256",
)
_CONTINUOUS_GROUP = (
    "target",
    "model",
    "spatial",
    "task",
    "endpoint",
    "analysis_population",
    "split_seed",
    "fold_manifest_sha256",
)
_LOGICAL_BINARY_IDENTITY = (
    "target",
    "model",
    "spatial",
    "timing",
    "analysis_population",
)
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ReportingInputs:
    """The new and hash-pinned parent private/public evaluation pairs."""

    new_baseline_private: Path
    new_baseline_public: Path
    new_phenotype_private: Path
    new_phenotype_public: Path
    new_subtype_private: Path
    new_subtype_public: Path
    new_ftv_private: Path
    new_ftv_public: Path
    old_baseline_private: Path
    old_baseline_public: Path
    old_phenotype_public: Path
    old_subtype_public: Path
    old_ftv_public: Path


@dataclass(frozen=True)
class ReportingOutputs:
    paired_csv: Path
    summary_json: Path
    final_report: Path
    timing_figure: Path
    comparison_figure: Path
    reporting_marker: Path


@dataclass(frozen=True)
class _LoadedInputs:
    new_baseline_private: pd.DataFrame
    new_baseline_public: pd.DataFrame
    phenotype_public: pd.DataFrame
    subtype_public: pd.DataFrame
    ftv_public: pd.DataFrame
    old_phenotype_public: pd.DataFrame
    old_subtype_public: pd.DataFrame
    old_ftv_public: pd.DataFrame
    combined_baseline: pd.DataFrame
    full_patient_order: tuple[str, ...]
    complete_patient_order: tuple[str, ...]
    input_sha256: Mapping[str, str]


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def static_comparison_contract() -> dict[str, Any]:
    """Return the complete outcome-independent comparison contract."""

    return {
        "version": COMPARISON_CONTRACT_VERSION,
        "model": MODEL_NAME,
        "reference_dino": DINO_V1_MODEL,
        "timings": list(DECISION_POINTS),
        "spatial_axes": list(SPATIAL_AXES),
        "ordered_pcr_variants": [variant for variant, _ in _PCR_VARIANTS],
        "families": [
            {
                "name": family,
                "spec_count": count,
            }
            for family, count in FAMILY_COUNTS.items()
        ],
        "resolved_spec_count": EXPECTED_COMPARISON_COUNT,
        "metrics_per_spec": list(METRICS),
        "paired_metric_row_count": EXPECTED_PAIRED_METRIC_ROWS,
        "bootstrap": {
            "unit": "patient",
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "ci_level": CI_LEVEL,
            "method": "paired outer-fold OOF nonparametric percentile bootstrap",
        },
        "candidate_policy": (
            "display every frozen DINOv3 axis, timing, population and model variant; "
            "no best filtering and no performance-based ordering"
        ),
        "interpretation": {
            "status": "posthoc sensitivity baseline",
            "pretraining_contamination": "unknown; LVD-1689M is non-enumerable",
            "license": "Meta DINOv3 custom license; institutional acceptance required",
            "original_conclusions": "unchanged",
        },
    }


def comparison_contract_sha256() -> str:
    return hashlib.sha256(_canonical_json_bytes(static_comparison_contract())).hexdigest()


def _population(kind: str, *, full_population: str, complete_population: str) -> str:
    if kind == "full":
        return full_population
    if kind == "complete":
        return complete_population
    raise AssertionError(f"unknown frozen population kind: {kind}")


def _comparison_id(
    *,
    family: str,
    estimand: str,
    population: str,
    timing: str,
    reference_model: str,
    reference_spatial: str,
    candidate_model: str,
    candidate_spatial: str,
) -> str:
    payload = {
        "family": family,
        "estimand": estimand,
        "analysis_population": population,
        "timing": timing,
        "reference_model": reference_model,
        "reference_spatial": reference_spatial,
        "candidate_model": candidate_model,
        "candidate_spatial": candidate_spatial,
    }
    suffix = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()[:16]
    return f"{family}:{suffix}"


def build_comparison_specs(
    prediction_identities: pd.DataFrame,
    *,
    full_population: str = FULL_POPULATION,
    complete_population: str = COMPLETE_POPULATION,
) -> tuple[ComparisonSpec, ...]:
    """Resolve exactly 84 specs while reading identity columns only.

    The function deliberately accepts an identity-only DataFrame; callers and
    tests need not make any score or metric column available to it.
    """

    missing_columns = sorted(
        set(_LOGICAL_BINARY_IDENTITY).difference(prediction_identities.columns)
    )
    if missing_columns:
        raise ValueError(f"comparison identity table is missing columns: {missing_columns}")
    identities = set(
        prediction_identities.loc[:, _LOGICAL_BINARY_IDENTITY]
        .drop_duplicates()
        .astype(str)
        .itertuples(index=False, name=None)
    )

    def require(model: str, spatial: str, timing: str, population: str) -> None:
        identity = ("pCR", model, spatial, timing, population)
        if identity not in identities:
            raise ValueError(
                "missing frozen comparison cell: "
                f"model={model}, spatial={spatial}, timing={timing}, "
                f"population={population}"
            )

    specs: list[ComparisonSpec] = []

    def add(
        family: str,
        estimand: str,
        population: str,
        timing: str,
        reference_model: str,
        reference_spatial: str,
        candidate_model: str,
        candidate_spatial: str,
    ) -> None:
        require(reference_model, reference_spatial, timing, population)
        require(candidate_model, candidate_spatial, timing, population)
        specs.append(
            ComparisonSpec(
                comparison_id=_comparison_id(
                    family=family,
                    estimand=estimand,
                    population=population,
                    timing=timing,
                    reference_model=reference_model,
                    reference_spatial=reference_spatial,
                    candidate_model=candidate_model,
                    candidate_spatial=candidate_spatial,
                ),
                family=family,
                estimand=estimand,
                analysis_population=population,
                timing=timing,
                reference_model=reference_model,
                reference_spatial=reference_spatial,
                candidate_model=candidate_model,
                candidate_spatial=candidate_spatial,
            )
        )

    # 36: DINOv3 versus DINOv1, every pCR variant/axis/timing.
    for timing in DECISION_POINTS:
        for variant, population_kind in _PCR_VARIANTS:
            population = _population(
                population_kind,
                full_population=full_population,
                complete_population=complete_population,
            )
            for spatial in SPATIAL_AXES:
                add(
                    "dinov3_vs_dinov1",
                    f"dinov3_minus_dinov1_{variant}",
                    population,
                    timing,
                    f"{DINO_V1_MODEL}_{variant}",
                    spatial,
                    f"{MODEL_NAME}_{variant}",
                    spatial,
                )

    # 18: DINOv3 LOCAL versus GLOBAL, every pCR variant/timing.
    for timing in DECISION_POINTS:
        for variant, population_kind in _PCR_VARIANTS:
            population = _population(
                population_kind,
                full_population=full_population,
                complete_population=complete_population,
            )
            model = f"{MODEL_NAME}_{variant}"
            add(
                    "local_vs_global",
                f"dinov3_LOCAL_minus_GLOBAL_{variant}",
                population,
                timing,
                model,
                "GLOBAL",
                model,
                "LOCAL",
            )

    # 12: full-cohort DINOv3 versus the spatially matched current CNN.
    for timing in DECISION_POINTS:
        for variant in _FULL_VARIANTS:
            for spatial, current in (("GLOBAL", "GAP0"), ("LOCAL", "LOCAL0")):
                add(
                    "dinov3_vs_current_cnn_full",
                    f"dinov3_minus_current_CNN_{variant}",
                    full_population,
                    timing,
                    f"{current}_{variant}",
                    spatial,
                    f"{MODEL_NAME}_{variant}",
                    spatial,
                )

    # 12: clinical-only versus clinical+DINOv3, full and paired populations.
    for timing in DECISION_POINTS:
        for population, reference, variant in (
            (full_population, "clinical_only", "mri_clinical"),
            (complete_population, "clinical_only_paired", "mri_clinical_paired"),
        ):
            for spatial in SPATIAL_AXES:
                add(
                    "clinical_gain",
                    "clinical_plus_dinov3_minus_clinical",
                    population,
                    timing,
                    reference,
                    "NONE",
                    f"{MODEL_NAME}_{variant}",
                    spatial,
                )

    # 6: beyond-FTV complementarity in the complete-case population.
    for timing in DECISION_POINTS:
        for spatial in SPATIAL_AXES:
            add(
                "beyond_ftv",
                "clinical_FTV_dinov3_minus_clinical_FTV",
                complete_population,
                timing,
                "clinical_ftv",
                "TABULAR",
                f"{MODEL_NAME}_mri_clinical_ftv",
                spatial,
            )

    counts = Counter(spec.family for spec in specs)
    if dict(counts) != FAMILY_COUNTS:
        raise AssertionError(f"comparison family counts drifted: {dict(counts)}")
    if len(specs) != EXPECTED_COMPARISON_COUNT:
        raise AssertionError("comparison contract did not resolve exactly 84 specs")
    if len({spec.comparison_id for spec in specs}) != len(specs):
        raise AssertionError("comparison contract contains duplicate IDs")
    return tuple(specs)


def _read_csv(path: Path, *, private: bool, label: str) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"missing {label} input: {source.name}")
    if private and not source.name.endswith(".private.csv"):
        raise ValueError(f"{label} private input must end in .private.csv")
    if not private and ".private." in source.name:
        raise ValueError(f"{label} public input cannot have a private filename")
    frame = pd.read_csv(source, dtype={"patient_id": str})
    if frame.empty:
        raise ValueError(f"{label} input is empty")
    return frame


def _normalise_private(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    required = {"patient_id", "fold", "split"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} private input is missing columns: {missing}")
    output = frame.copy()
    output["patient_id"] = output["patient_id"].astype(str).str.strip()
    invalid_id = output["patient_id"].eq("") | output["patient_id"].str.lower().isin(
        {"nan", "none", "<na>"}
    )
    if invalid_id.any():
        raise ValueError(f"{label} contains an empty patient ID")
    if set(output["split"].astype(str)) != {"test"}:
        raise ValueError(f"{label} must contain test-only OOF rows")
    fold = pd.to_numeric(output["fold"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(fold).all() or not np.equal(fold, np.floor(fold)).all():
        raise ValueError(f"{label} fold values must be finite integers")
    output["fold"] = fold.astype(np.int64)
    return output


def _expected_new_baseline_identities(
    *, full_population: str, complete_population: str
) -> tuple[tuple[str, str, str, str, str], ...]:
    return tuple(
        (
            "pCR",
            f"{MODEL_NAME}_{variant}",
            spatial,
            timing,
            _population(
                population_kind,
                full_population=full_population,
                complete_population=complete_population,
            ),
        )
        for timing in DECISION_POINTS
        for variant, population_kind in _PCR_VARIANTS
        for spatial in SPATIAL_AXES
    )


def _expected_old_identities(
    *, full_population: str, complete_population: str
) -> tuple[tuple[str, str, str, str, str], ...]:
    identities: list[tuple[str, str, str, str, str]] = []
    for timing in DECISION_POINTS:
        for variant, population_kind in _PCR_VARIANTS:
            population = _population(
                population_kind,
                full_population=full_population,
                complete_population=complete_population,
            )
            for spatial in SPATIAL_AXES:
                identities.append(
                    (
                        "pCR",
                        f"{DINO_V1_MODEL}_{variant}",
                        spatial,
                        timing,
                        population,
                    )
                )
        for variant in _FULL_VARIANTS:
            identities.extend(
                (
                    ("pCR", f"GAP0_{variant}", "GLOBAL", timing, full_population),
                    ("pCR", f"LOCAL0_{variant}", "LOCAL", timing, full_population),
                )
            )
        identities.extend(
            (
                ("pCR", "clinical_only", "NONE", timing, full_population),
                (
                    "pCR",
                    "clinical_only_paired",
                    "NONE",
                    timing,
                    complete_population,
                ),
                ("pCR", "clinical_ftv", "TABULAR", timing, complete_population),
            )
        )
    return tuple(identities)


def _subset_identities(
    frame: pd.DataFrame,
    expected: Sequence[tuple[str, str, str, str, str]],
) -> pd.DataFrame:
    expected_index = pd.MultiIndex.from_tuples(expected, names=_LOGICAL_BINARY_IDENTITY)
    observed_index = pd.MultiIndex.from_frame(frame.loc[:, _LOGICAL_BINARY_IDENTITY].astype(str))
    selected = frame.loc[observed_index.isin(expected_index)].copy()
    observed = set(
        selected.loc[:, _LOGICAL_BINARY_IDENTITY]
        .astype(str)
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    missing = sorted(set(expected).difference(observed))
    if missing:
        raise ValueError(f"old comparator is missing {len(missing)} frozen cells: {missing[:3]}")
    return selected


def _validate_binary_groups(
    frame: pd.DataFrame,
    *,
    expected_identities: Sequence[tuple[str, str, str, str, str]],
    full_size: int,
    complete_size: int,
    full_population: str,
    complete_population: str,
    label: str,
    exact_identity_set: bool,
) -> dict[str, tuple[str, ...]]:
    required = {
        "patient_id",
        "fold",
        "split",
        "y_true",
        "y_score",
        *_BINARY_GROUP,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")
    if frame.duplicated([*_BINARY_GROUP, "patient_id"]).any():
        raise ValueError(f"{label} contains duplicate prediction identities/patients")
    logical = set(
        frame.loc[:, _LOGICAL_BINARY_IDENTITY]
        .astype(str)
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    expected = set(expected_identities)
    if exact_identity_set and logical != expected:
        raise ValueError(
            f"{label} frozen cell set drifted: missing={sorted(expected-logical)[:3]}, "
            f"extra={sorted(logical-expected)[:3]}"
        )
    if not expected.issubset(logical):
        raise ValueError(f"{label} is missing frozen prediction cells")
    if set(pd.to_numeric(frame["split_seed"], errors="coerce")) != {2026}:
        raise ValueError(f"{label} split seed drifted")
    fold_hashes = set(frame["fold_manifest_sha256"].astype(str))
    if len(fold_hashes) != 1 or not _DIGEST.fullmatch(next(iter(fold_hashes))):
        raise ValueError(f"{label} fold-manifest identity drifted")
    truth = pd.to_numeric(frame["y_true"], errors="coerce").to_numpy(dtype=np.float64)
    score = pd.to_numeric(frame["y_score"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(truth).all() or not np.isin(truth, (0.0, 1.0)).all():
        raise ValueError(f"{label} binary truth must be complete 0/1")
    if not np.isfinite(score).all() or np.any((score < 0.0) | (score > 1.0)):
        raise ValueError(f"{label} contains invalid probabilities")
    sizes = {full_population: full_size, complete_population: complete_size}
    orders: dict[str, tuple[str, ...]] = {}
    for keys, group in frame.groupby(list(_BINARY_GROUP), sort=False, dropna=False):
        population = str(group["analysis_population"].iloc[0])
        if population not in sizes:
            raise ValueError(f"{label} contains unexpected population {population}")
        expected_n = sizes[population]
        if len(group) != expected_n or group["patient_id"].duplicated().any():
            raise ValueError(f"{label} group {keys} does not have {expected_n} patients")
        if set(group["fold"].astype(int)) != set(FOLDS):
            raise ValueError(f"{label} group {keys} does not cover five outer folds")
        order = tuple(sorted(group["patient_id"].astype(str)))
        prior = orders.setdefault(population, order)
        if prior != order:
            raise ValueError(f"{label} silently changed patients in {population}")
    # Truth is invariant within one target/population, but HR and HER2 are
    # distinct probe targets and are not required to agree with each other.
    for (target, population), group in frame.groupby(
        ["target", "analysis_population"], sort=False
    ):
        identity = group.loc[:, ["patient_id", "fold", "y_true"]].drop_duplicates()
        if identity["patient_id"].duplicated().any():
            raise ValueError(
                f"{label} patient truth/fold changes in {target}/{population}"
            )
    return orders


def _patient_folds(
    frame: pd.DataFrame, *, population: str, label: str
) -> dict[str, int]:
    selected = frame.loc[
        frame["analysis_population"].astype(str).eq(population),
        ["patient_id", "fold"],
    ].drop_duplicates()
    if selected.empty or selected["patient_id"].duplicated().any():
        raise ValueError(f"{label} patient/fold assignment is not invariant")
    return {
        str(row.patient_id): int(row.fold)
        for row in selected.itertuples(index=False)
    }


def _assert_group_fold_map(
    group: pd.DataFrame, expected: Mapping[str, int], *, label: str
) -> None:
    observed = {
        str(row.patient_id): int(row.fold)
        for row in group.loc[:, ["patient_id", "fold"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    if observed != dict(expected):
        raise ValueError(f"{label} outer-fold assignment differs from the pCR baseline")


def _expected_probe_identities(
    *, full_population: str, complete_population: str
) -> tuple[
    set[tuple[str, str, str, str, str]],
    set[tuple[str, str, str, str, str]],
    set[tuple[str, str, str, str, str]],
]:
    phenotype = {
        (target, MODEL_NAME, spatial, "T0", full_population)
        for target in ("HR", "HER2")
        for spatial in SPATIAL_AXES
    }
    subtype = {
        ("HR_HER2_subtype", MODEL_NAME, spatial, "T0", full_population)
        for spatial in SPATIAL_AXES
    }
    ftv = {
        ("FTV", MODEL_NAME, spatial, task, endpoint)
        for spatial in SPATIAL_AXES
        for task, endpoints in (
            ("static", ("T0", "T1", "T2", "T3")),
            ("delta", ("T0-T1", "T1-T2", "T2-T3")),
        )
        for endpoint in endpoints
    }
    return phenotype, subtype, ftv


def _validate_probe_groups(
    phenotype: pd.DataFrame,
    subtype: pd.DataFrame,
    ftv: pd.DataFrame,
    *,
    full_size: int,
    complete_size: int,
    full_population: str,
    complete_population: str,
    full_order: Sequence[str],
    complete_order: Sequence[str],
    full_folds: Mapping[str, int],
    complete_folds: Mapping[str, int],
) -> None:
    expected_phenotype, expected_subtype, expected_ftv = _expected_probe_identities(
        full_population=full_population,
        complete_population=complete_population,
    )
    phenotype_orders = _validate_binary_groups(
        phenotype,
        expected_identities=tuple(expected_phenotype),
        full_size=full_size,
        complete_size=complete_size,
        full_population=full_population,
        complete_population=complete_population,
        label="DINOv3 phenotype",
        exact_identity_set=True,
    )
    if set(phenotype_orders) != {full_population}:
        raise ValueError("phenotype probes must cover the full population only")
    if phenotype_orders[full_population] != tuple(full_order):
        raise ValueError("phenotype and pCR patient coverage differs")
    for keys, group in phenotype.groupby(list(_BINARY_GROUP), sort=False, dropna=False):
        _assert_group_fold_map(group, full_folds, label=f"phenotype {keys}")

    subtype_required = {
        "patient_id",
        "fold",
        "split",
        "target",
        "model",
        "spatial",
        "timing",
        "analysis_population",
        "split_seed",
        "fold_manifest_sha256",
        "y_true",
        "classes_json",
        "probabilities_json",
    }
    missing = sorted(subtype_required.difference(subtype.columns))
    if missing:
        raise ValueError(f"DINOv3 subtype input is missing columns: {missing}")
    subtype_identity = set(
        subtype.loc[:, _LOGICAL_BINARY_IDENTITY]
        .astype(str)
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if subtype_identity != expected_subtype:
        raise ValueError("DINOv3 subtype frozen cell set drifted")
    if subtype.duplicated([*_BINARY_GROUP, "patient_id"]).any():
        raise ValueError("DINOv3 subtype contains duplicate patients")
    if set(pd.to_numeric(subtype["split_seed"], errors="coerce")) != {2026}:
        raise ValueError("DINOv3 subtype split seed drifted")
    subtype_hashes = set(subtype["fold_manifest_sha256"].astype(str))
    if len(subtype_hashes) != 1 or not _DIGEST.fullmatch(next(iter(subtype_hashes))):
        raise ValueError("DINOv3 subtype fold-manifest identity drifted")
    for keys, group in subtype.groupby(list(_BINARY_GROUP), sort=False, dropna=False):
        if len(group) != full_size or tuple(sorted(group["patient_id"].astype(str))) != tuple(
            full_order
        ):
            raise ValueError("DINOv3 subtype patient coverage drifted")
        if set(group["fold"].astype(int)) != set(FOLDS):
            raise ValueError("DINOv3 subtype does not cover five folds")
        _assert_group_fold_map(group, full_folds, label=f"subtype {keys}")
        class_payloads = {str(value) for value in group["classes_json"]}
        if len(class_payloads) != 1:
            raise ValueError("DINOv3 subtype class order differs across patients")
        try:
            classes = tuple(str(value) for value in json.loads(next(iter(class_payloads))))
            probabilities = np.asarray(
                [json.loads(str(value)) for value in group["probabilities_json"]],
                dtype=np.float64,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("DINOv3 subtype probability payload is invalid") from error
        if (
            len(classes) != 4
            or len(set(classes)) != 4
            or probabilities.shape != (full_size, 4)
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0.0)
            or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-7, atol=1e-7)
            or set(group["y_true"].astype(str)) != set(classes)
        ):
            raise ValueError("DINOv3 subtype truth/probability contract drifted")

    ftv_required = {
        "patient_id",
        "fold",
        "split",
        "target",
        "model",
        "spatial",
        "task",
        "endpoint",
        "analysis_population",
        "split_seed",
        "fold_manifest_sha256",
        "y_true",
        "y_pred",
    }
    missing = sorted(ftv_required.difference(ftv.columns))
    if missing:
        raise ValueError(f"DINOv3 FTV input is missing columns: {missing}")
    observed_ftv = set(
        ftv.loc[:, ("target", "model", "spatial", "task", "endpoint")]
        .astype(str)
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if observed_ftv != expected_ftv:
        raise ValueError("DINOv3 FTV frozen cell set drifted")
    if set(ftv["analysis_population"].astype(str)) != {complete_population}:
        raise ValueError("DINOv3 FTV probes must use the complete-case population")
    if set(pd.to_numeric(ftv["split_seed"], errors="coerce")) != {2026}:
        raise ValueError("DINOv3 FTV split seed drifted")
    ftv_hashes = set(ftv["fold_manifest_sha256"].astype(str))
    if len(ftv_hashes) != 1 or not _DIGEST.fullmatch(next(iter(ftv_hashes))):
        raise ValueError("DINOv3 FTV fold-manifest identity drifted")
    if ftv.duplicated([*_CONTINUOUS_GROUP, "patient_id"]).any():
        raise ValueError("DINOv3 FTV contains duplicate patients")
    for keys, group in ftv.groupby(list(_CONTINUOUS_GROUP), sort=False, dropna=False):
        if len(group) != complete_size or tuple(sorted(group["patient_id"].astype(str))) != tuple(
            complete_order
        ):
            raise ValueError("DINOv3 FTV patient coverage drifted")
        if set(group["fold"].astype(int)) != set(FOLDS):
            raise ValueError("DINOv3 FTV does not cover five folds")
        _assert_group_fold_map(group, complete_folds, label=f"FTV {keys}")
        truth = pd.to_numeric(group["y_true"], errors="coerce").to_numpy(dtype=float)
        pred = pd.to_numeric(group["y_pred"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(truth).all() or not np.isfinite(pred).all():
            raise ValueError("DINOv3 FTV contains non-finite values")


def _input_hashes(inputs: ReportingInputs) -> dict[str, str]:
    roles = {
        "new_baseline_private": inputs.new_baseline_private,
        "new_baseline_public": inputs.new_baseline_public,
        "new_phenotype_private": inputs.new_phenotype_private,
        "new_phenotype_public": inputs.new_phenotype_public,
        "new_subtype_private": inputs.new_subtype_private,
        "new_subtype_public": inputs.new_subtype_public,
        "new_ftv_private": inputs.new_ftv_private,
        "new_ftv_public": inputs.new_ftv_public,
        "old_baseline_private": inputs.old_baseline_private,
        "old_baseline_public": inputs.old_baseline_public,
        "old_phenotype_public": inputs.old_phenotype_public,
        "old_subtype_public": inputs.old_subtype_public,
        "old_ftv_public": inputs.old_ftv_public,
    }
    return {role: file_sha256(path) for role, path in roles.items()}


def _load_and_validate(
    inputs: ReportingInputs,
    *,
    full_size: int,
    complete_size: int,
) -> _LoadedInputs:
    if full_size <= complete_size or complete_size <= 0:
        raise ValueError("cohort sizes must satisfy full > complete > 0")
    full_population = f"full_{full_size}"
    complete_population = f"radiomics_complete_case_{complete_size}"

    new_baseline = _normalise_private(
        _read_csv(inputs.new_baseline_private, private=True, label="DINOv3 baseline"),
        label="DINOv3 baseline",
    )
    phenotype = _normalise_private(
        _read_csv(inputs.new_phenotype_private, private=True, label="DINOv3 phenotype"),
        label="DINOv3 phenotype",
    )
    subtype = _normalise_private(
        _read_csv(inputs.new_subtype_private, private=True, label="DINOv3 subtype"),
        label="DINOv3 subtype",
    )
    ftv = _normalise_private(
        _read_csv(inputs.new_ftv_private, private=True, label="DINOv3 FTV"),
        label="DINOv3 FTV",
    )
    old_baseline = _normalise_private(
        _read_csv(inputs.old_baseline_private, private=True, label="old baseline"),
        label="old baseline",
    )
    new_baseline_public = _read_csv(
        inputs.new_baseline_public, private=False, label="DINOv3 baseline public"
    )
    phenotype_public = _read_csv(
        inputs.new_phenotype_public, private=False, label="DINOv3 phenotype public"
    )
    subtype_public = _read_csv(
        inputs.new_subtype_public, private=False, label="DINOv3 subtype public"
    )
    ftv_public = _read_csv(
        inputs.new_ftv_public, private=False, label="DINOv3 FTV public"
    )
    old_baseline_public = _read_csv(
        inputs.old_baseline_public, private=False, label="old baseline public"
    )
    old_phenotype_public = _read_csv(
        inputs.old_phenotype_public, private=False, label="old phenotype public"
    )
    old_subtype_public = _read_csv(
        inputs.old_subtype_public, private=False, label="old subtype public"
    )
    old_ftv_public = _read_csv(
        inputs.old_ftv_public, private=False, label="old FTV public"
    )

    new_orders = _validate_binary_groups(
        new_baseline,
        expected_identities=_expected_new_baseline_identities(
            full_population=full_population, complete_population=complete_population
        ),
        full_size=full_size,
        complete_size=complete_size,
        full_population=full_population,
        complete_population=complete_population,
        label="DINOv3 baseline",
        exact_identity_set=True,
    )
    if set(new_orders) != {full_population, complete_population}:
        raise ValueError("DINOv3 baseline must cover full and complete populations")

    old_expected = _expected_old_identities(
        full_population=full_population, complete_population=complete_population
    )
    old_required = _subset_identities(old_baseline, old_expected)
    old_orders = _validate_binary_groups(
        old_required,
        expected_identities=old_expected,
        full_size=full_size,
        complete_size=complete_size,
        full_population=full_population,
        complete_population=complete_population,
        label="old comparator",
        exact_identity_set=True,
    )
    if old_orders != new_orders:
        raise ValueError("old and DINOv3 comparator patient coverage differs")
    full_folds = _patient_folds(
        new_baseline, population=full_population, label="DINOv3 baseline/full"
    )
    complete_folds = _patient_folds(
        new_baseline,
        population=complete_population,
        label="DINOv3 baseline/complete-case",
    )
    if _patient_folds(
        old_required, population=full_population, label="old comparator/full"
    ) != full_folds or _patient_folds(
        old_required,
        population=complete_population,
        label="old comparator/complete-case",
    ) != complete_folds:
        raise ValueError("old and DINOv3 comparator outer-fold assignments differ")

    _validate_probe_groups(
        phenotype,
        subtype,
        ftv,
        full_size=full_size,
        complete_size=complete_size,
        full_population=full_population,
        complete_population=complete_population,
        full_order=new_orders[full_population],
        complete_order=new_orders[complete_population],
        full_folds=full_folds,
        complete_folds=complete_folds,
    )

    binary_keys = [*_BINARY_GROUP, "aggregation"]
    _assert_public_matches(
        new_baseline_public,
        aggregate_binary_predictions(new_baseline),
        key_columns=binary_keys,
        label="baseline",
        relaxed_numeric_columns=BINARY_IRLS_COLUMNS,
    )
    _assert_public_matches(
        phenotype_public,
        aggregate_binary_predictions(phenotype),
        key_columns=binary_keys,
        label="phenotype",
        relaxed_numeric_columns=BINARY_IRLS_COLUMNS,
    )
    _assert_public_matches(
        subtype_public,
        aggregate_multiclass_predictions(subtype),
        key_columns=binary_keys,
        label="subtype",
    )
    _assert_public_matches(
        ftv_public,
        aggregate_continuous_predictions(ftv),
        key_columns=[*_CONTINUOUS_GROUP, "aggregation"],
        label="FTV",
    )
    _assert_public_matches(
        old_baseline_public,
        aggregate_binary_predictions(old_baseline),
        key_columns=binary_keys,
        label="baseline",
        relaxed_numeric_columns=BINARY_IRLS_COLUMNS,
    )
    # The old probe inputs are public, hash-pinned aggregate comparators.  They
    # are matched exhaustively to the new public probes; no old patient-level
    # probe predictions are needed or read.
    _matched_probe_public(phenotype_public, old_phenotype_public, kind="phenotype")
    _matched_probe_public(subtype_public, old_subtype_public, kind="subtype")
    _matched_probe_public(ftv_public, old_ftv_public, kind="ftv")

    combined = pd.concat((new_baseline, old_required), ignore_index=True)
    if combined.duplicated([*_BINARY_GROUP, "patient_id"]).any():
        raise ValueError("combined DINOv3/old predictions contain duplicate identities")
    # Every patient/fold/truth must remain identical across both producers.
    for population, group in combined.groupby("analysis_population", sort=False):
        identity = group.loc[:, ["patient_id", "fold", "y_true"]].drop_duplicates()
        if identity["patient_id"].duplicated().any():
            raise ValueError(f"combined truth/fold differs in {population}")

    return _LoadedInputs(
        new_baseline_private=new_baseline,
        new_baseline_public=new_baseline_public,
        phenotype_public=phenotype_public,
        subtype_public=subtype_public,
        ftv_public=ftv_public,
        old_phenotype_public=old_phenotype_public,
        old_subtype_public=old_subtype_public,
        old_ftv_public=old_ftv_public,
        combined_baseline=combined,
        full_patient_order=new_orders[full_population],
        complete_patient_order=new_orders[complete_population],
        input_sha256=_input_hashes(inputs),
    )


def _checked_digest(value: object, *, label: str) -> str:
    digest = str(value)
    if not _DIGEST.fullmatch(digest):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _checked_relative_path(root: Path, raw: object, *, label: str) -> Path:
    text = str(raw)
    path = Path(text)
    if not text or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be a root-relative path")
    resolved = (root / path).resolve(strict=True)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise ValueError(f"{label} escapes the experiment root") from error
    return resolved


def _validate_mode(path: Path, expected: int, *, label: str) -> None:
    observed = stat.S_IMODE(path.stat().st_mode)
    if observed != expected:
        raise ValueError(f"{label} mode must be {expected:04o}, observed {observed:04o}")


def _formal_artifacts(
    inputs: ReportingInputs, consumer: str
) -> dict[str, tuple[Path, int]]:
    if consumer == "baseline":
        return {
            "predictions": (Path(inputs.new_baseline_private), 0o600),
            "selection": (
                EXPERIMENT_ROOT / "metrics/dinov3_baseline_selection.private.csv",
                0o600,
            ),
            "metrics": (Path(inputs.new_baseline_public), 0o644),
            "progress": (
                EXPERIMENT_ROOT / "logs/dinov3_baseline.progress.private.jsonl",
                0o600,
            ),
        }
    if consumer == "probe":
        return {
            "phenotype_predictions": (Path(inputs.new_phenotype_private), 0o600),
            "phenotype_selection": (
                EXPERIMENT_ROOT / "metrics/dinov3_phenotype_selection.private.csv",
                0o600,
            ),
            "phenotype_metrics": (Path(inputs.new_phenotype_public), 0o644),
            "subtype_predictions": (Path(inputs.new_subtype_private), 0o600),
            "subtype_selection": (
                EXPERIMENT_ROOT / "metrics/dinov3_subtype_selection.private.csv",
                0o600,
            ),
            "subtype_metrics": (Path(inputs.new_subtype_public), 0o644),
            "ftv_predictions": (Path(inputs.new_ftv_private), 0o600),
            "ftv_selection": (
                EXPERIMENT_ROOT / "metrics/dinov3_ftv_selection.private.csv",
                0o600,
            ),
            "ftv_metrics": (Path(inputs.new_ftv_public), 0o644),
            "progress": (
                EXPERIMENT_ROOT / "logs/dinov3_probe.progress.private.jsonl",
                0o600,
            ),
        }
    raise ValueError("producer consumer must be baseline or probe")


def _sanitize_producer_receipt(
    payload: Mapping[str, Any],
    *,
    consumer: str,
    report_lock: Mapping[str, Any],
    inputs: ReportingInputs,
    strict_modes: bool,
) -> dict[str, Any]:
    """Recheck a receipt verified by ``locking.verify_producer_receipt``.

    This function never parses an artifact.  Paths are resolved against the
    repository root because that is the producer receipt's frozen convention.
    Only hashes/byte counts, never paths, are returned for public provenance.
    """

    required = {
        "schema_version",
        "consumer",
        "evaluation_lock_sha256",
        "argv_sha256",
        "metric_values_viewed",
        "artifact_hash_method",
        "artifacts",
        "counts",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ValueError(f"DINOv3 {consumer} producer receipt schema drifted")
    if payload["schema_version"] != PRODUCER_RECEIPT_SCHEMA_VERSION:
        raise ValueError(f"DINOv3 {consumer} producer receipt version drifted")
    if payload["consumer"] != consumer or payload["metric_values_viewed"] is not False:
        raise ValueError(f"DINOv3 {consumer} receipt identity/visibility drifted")
    if payload["artifact_hash_method"] != "sha256_binary_stream_no_parse":
        raise ValueError(f"DINOv3 {consumer} artifact hash method drifted")
    lock_digest = _checked_digest(
        payload["evaluation_lock_sha256"], label=f"{consumer} evaluation lock"
    )
    if lock_digest != _checked_digest(report_lock["lock_sha256"], label="report lock"):
        raise ValueError(f"DINOv3 {consumer} receipt belongs to another lock")
    argv_digest = _checked_digest(payload["argv_sha256"], label=f"{consumer} argv")
    if argv_digest != _checked_digest(report_lock["argv_sha256"], label="report argv"):
        # All three formal commands are frozen to the exact empty argv.
        raise ValueError(f"DINOv3 {consumer} argv differs from the report argv")

    counts = payload["counts"]
    expected_counts = report_lock.get("expected_counts")
    if (
        not isinstance(counts, Mapping)
        or not counts
        or not isinstance(expected_counts, Mapping)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in counts.values()
        )
        or any(key not in expected_counts or expected_counts[key] != value for key, value in counts.items())
    ):
        raise ValueError(f"DINOv3 {consumer} producer counts drifted from the lock")

    expected_artifacts = _formal_artifacts(inputs, consumer)
    records = payload["artifacts"]
    if not isinstance(records, Mapping) or set(records) != set(expected_artifacts):
        raise ValueError(f"DINOv3 {consumer} producer artifact set drifted")
    verified: dict[str, dict[str, Any]] = {}
    root = REPOSITORY_ROOT.resolve(strict=True)
    for role, (expected_path, expected_mode) in expected_artifacts.items():
        record = records[role]
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256", "bytes"}:
            raise ValueError(f"DINOv3 {consumer}/{role} receipt record drifted")
        actual_path = _checked_relative_path(
            root, record["path"], label=f"{consumer}/{role}"
        )
        if actual_path.is_symlink() or actual_path != expected_path.resolve(strict=True):
            raise ValueError(f"DINOv3 {consumer}/{role} artifact path drifted")
        byte_count = record["bytes"]
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count <= 0:
            raise ValueError(f"DINOv3 {consumer}/{role} byte count is invalid")
        if actual_path.stat().st_size != byte_count:
            raise ValueError(f"DINOv3 {consumer}/{role} byte count drifted")
        digest = _checked_digest(record["sha256"], label=f"{consumer}/{role}")
        if digest != file_sha256(actual_path):
            raise ValueError(f"DINOv3 {consumer}/{role} artifact hash drifted")
        if strict_modes:
            _validate_mode(actual_path, expected_mode, label=f"{consumer}/{role}")
        verified[str(role)] = {"sha256": digest, "bytes": byte_count}

    receipt_path = EXPERIMENT_ROOT / f"metrics/{consumer}_run.private.provenance.json"
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise FileNotFoundError(f"missing DINOv3 {consumer} receipt")
    if strict_modes:
        _validate_mode(receipt_path, 0o600, label=f"{consumer} receipt")
    return {
        "schema_version": PRODUCER_RECEIPT_SCHEMA_VERSION,
        "consumer": consumer,
        "evaluation_lock_sha256": lock_digest,
        "argv_sha256": argv_digest,
        "receipt_sha256": file_sha256(receipt_path),
        "artifacts": dict(sorted(verified.items())),
        "counts": dict(sorted((str(key), int(value)) for key, value in counts.items())),
    }


def _sanitize_parent_comparator_artifacts(
    records: Mapping[str, Any], inputs: ReportingInputs
) -> dict[str, Any]:
    """Require all five parsed parent CSVs to be hash-pinned by the active lock."""

    if not isinstance(records, Mapping) or not records:
        raise ValueError("parent comparator artifact lock is missing")
    required_paths = {
        Path(inputs.old_baseline_private).resolve(strict=True),
        Path(inputs.old_baseline_public).resolve(strict=True),
        Path(inputs.old_phenotype_public).resolve(strict=True),
        Path(inputs.old_subtype_public).resolve(strict=True),
        Path(inputs.old_ftv_public).resolve(strict=True),
    }
    expected_paths = {
        (BASE_EXPERIMENT_ROOT / "predictions/baseline_predictions.private.csv").resolve(
            strict=True
        ),
        (BASE_EXPERIMENT_ROOT / "metrics/baseline_metrics.csv").resolve(strict=True),
        (BASE_EXPERIMENT_ROOT / "metrics/phenotype_metrics.csv").resolve(strict=True),
        (BASE_EXPERIMENT_ROOT / "metrics/subtype_metrics.csv").resolve(strict=True),
        (BASE_EXPERIMENT_ROOT / "metrics/ftv_probe_metrics.csv").resolve(strict=True),
    }
    if required_paths != expected_paths:
        raise ValueError("parent comparator inputs are not the frozen baseline CSV pair")
    seen: set[Path] = set()
    sanitized: dict[str, dict[str, Any]] = {}
    root = REPOSITORY_ROOT.resolve(strict=True)
    for role, raw in sorted(records.items()):
        if not isinstance(raw, Mapping) or set(raw) not in (
            {"path", "sha256"},
            {"path", "sha256", "bytes"},
        ):
            raise ValueError(f"parent comparator record drifted for {role}")
        path = _checked_relative_path(root, raw["path"], label=f"parent/{role}")
        if path.is_symlink():
            raise ValueError(f"parent comparator artifact is a symlink: {role}")
        digest = _checked_digest(raw["sha256"], label=f"parent/{role}")
        if digest != file_sha256(path):
            raise ValueError(f"parent comparator artifact hash drifted for {role}")
        record: dict[str, Any] = {"sha256": digest}
        if "bytes" in raw:
            byte_count = raw["bytes"]
            if (
                isinstance(byte_count, bool)
                or not isinstance(byte_count, int)
                or byte_count <= 0
                or path.stat().st_size != byte_count
            ):
                raise ValueError(f"parent comparator byte count drifted for {role}")
            record["bytes"] = byte_count
        sanitized[str(role)] = record
        if path in required_paths:
            seen.add(path)
    if seen != required_paths:
        raise ValueError("active lock does not pin all five parent comparator CSVs")
    return {"artifacts": sanitized, "required_csv_count": len(seen)}


def _public_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    ensure_public_safe(frame)
    records: list[dict[str, Any]] = []
    for raw in frame.to_dict(orient="records"):
        record: dict[str, Any] = {}
        for key, value in raw.items():
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, float) and not math.isfinite(value):
                value = None
            record[str(key)] = value
        records.append(record)
    return records


def _pooled(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["aggregation"].astype(str).eq("pooled_oof")].copy()


def _matched_probe_public(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    kind: str,
) -> pd.DataFrame:
    """Match public DINOv3 and DINOv1 probe rows without inferential reuse."""

    ensure_public_safe(candidate)
    ensure_public_safe(reference)
    if kind == "phenotype":
        identity = [
            "target",
            "spatial",
            "timing",
            "analysis_population",
            "split_seed",
            "fold_manifest_sha256",
            "aggregation",
        ]
        metrics = ("auroc", "auprc", "brier")
        expected = 4
    elif kind == "subtype":
        identity = [
            "target",
            "spatial",
            "timing",
            "analysis_population",
            "split_seed",
            "fold_manifest_sha256",
            "aggregation",
        ]
        metrics = ("macro_ovr_auroc", "macro_ovr_auprc", "multiclass_brier", "accuracy")
        expected = 2
    elif kind == "ftv":
        identity = [
            "target",
            "spatial",
            "task",
            "endpoint",
            "analysis_population",
            "split_seed",
            "fold_manifest_sha256",
            "aggregation",
        ]
        metrics = ("spearman", "r2", "rmse", "mae")
        expected = 14
    else:
        raise AssertionError(f"unknown probe comparison kind: {kind}")
    required = {"model", "n", *identity, *metrics}
    for frame, label in ((candidate, "DINOv3"), (reference, "DINOv1")):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{label} {kind} public metrics are missing {missing}")

    candidate_rows = _pooled(candidate).loc[
        _pooled(candidate)["model"].astype(str).eq(MODEL_NAME),
        [*identity, "n", *metrics],
    ].copy()
    reference_rows = _pooled(reference).loc[
        _pooled(reference)["model"].astype(str).eq(DINO_V1_MODEL),
        [*identity, "n", *metrics],
    ].copy()
    if len(candidate_rows) != expected or len(reference_rows) != expected:
        raise ValueError(
            f"matched {kind} probe coverage drifted: "
            f"DINOv3={len(candidate_rows)}, DINOv1={len(reference_rows)}"
        )
    if candidate_rows.duplicated(identity).any() or reference_rows.duplicated(identity).any():
        raise ValueError(f"matched {kind} probe identities are duplicated")
    merged = reference_rows.merge(
        candidate_rows,
        on=identity,
        how="inner",
        validate="one_to_one",
        suffixes=("_reference", "_candidate"),
    )
    if len(merged) != expected or not np.array_equal(
        pd.to_numeric(merged["n_reference"]).to_numpy(),
        pd.to_numeric(merged["n_candidate"]).to_numpy(),
    ):
        raise ValueError(f"matched {kind} probe population/pairing drifted")
    output = merged.loc[:, identity].copy()
    output["reference_model"] = DINO_V1_MODEL
    output["candidate_model"] = MODEL_NAME
    output["n"] = pd.to_numeric(merged["n_reference"]).astype(np.int64)
    for metric in metrics:
        reference_value = pd.to_numeric(
            merged[f"{metric}_reference"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        candidate_value = pd.to_numeric(
            merged[f"{metric}_candidate"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        if not np.isfinite(reference_value).all() or not np.isfinite(candidate_value).all():
            raise ValueError(f"matched {kind}/{metric} public values must be finite")
        output[f"reference_{metric}"] = reference_value
        output[f"candidate_{metric}"] = candidate_value
        output[f"delta_candidate_minus_reference_{metric}"] = (
            candidate_value - reference_value
        )
    ensure_public_safe(output)
    return output


def _fixed_public_order(
    frame: pd.DataFrame,
    *,
    kind: str,
    full_population: str,
    complete_population: str,
) -> pd.DataFrame:
    output = frame.copy()
    timing_order = {value: index for index, value in enumerate(DECISION_POINTS)}
    spatial_order = {value: index for index, value in enumerate(SPATIAL_AXES)}
    variant_order = {variant: index for index, (variant, _) in enumerate(_PCR_VARIANTS)}
    if kind == "pcr":
        output["_timing"] = output["timing"].astype(str).map(timing_order)
        output["_spatial"] = output["spatial"].astype(str).map(spatial_order)
        output["_variant"] = output["model"].astype(str).map(
            lambda value: next(
                (
                    variant_order[variant]
                    for variant, _ in _PCR_VARIANTS
                    if value == f"{MODEL_NAME}_{variant}"
                ),
                999,
            )
        )
        return output.sort_values(
            ["_timing", "_variant", "_spatial"], kind="stable"
        ).drop(columns=["_timing", "_variant", "_spatial"])
    if kind in {"phenotype", "subtype"}:
        target_order = {"HR": 0, "HER2": 1, "HR_HER2_subtype": 0}
        output["_target"] = output["target"].astype(str).map(target_order)
        output["_spatial"] = output["spatial"].astype(str).map(spatial_order)
        return output.sort_values(["_target", "_spatial"], kind="stable").drop(
            columns=["_target", "_spatial"]
        )
    if kind == "ftv":
        endpoint_order = {
            "T0": 0,
            "T1": 1,
            "T2": 2,
            "T3": 3,
            "T0-T1": 4,
            "T1-T2": 5,
            "T2-T3": 6,
        }
        output["_spatial"] = output["spatial"].astype(str).map(spatial_order)
        output["_endpoint"] = output["endpoint"].astype(str).map(endpoint_order)
        return output.sort_values(["_spatial", "_endpoint"], kind="stable").drop(
            columns=["_spatial", "_endpoint"]
        )
    raise AssertionError(f"unknown public ordering kind: {kind}")


def _ordered_comparisons(
    comparisons: pd.DataFrame, specs: Sequence[ComparisonSpec]
) -> pd.DataFrame:
    spec_order = {spec.comparison_id: index for index, spec in enumerate(specs)}
    metric_order = {metric: index for index, metric in enumerate(METRICS)}
    output = comparisons.copy()
    output["_spec_order"] = output["comparison_id"].astype(str).map(spec_order)
    output["_metric_order"] = output["metric"].astype(str).map(metric_order)
    if output[["_spec_order", "_metric_order"]].isna().any().any():
        raise AssertionError("paired results contain an unknown spec or metric")
    return output.sort_values(["_spec_order", "_metric_order"], kind="stable").drop(
        columns=["_spec_order", "_metric_order"]
    ).reset_index(drop=True)


def build_public_summary(
    loaded: _LoadedInputs,
    comparisons: pd.DataFrame,
    specs: Sequence[ComparisonSpec],
    *,
    full_size: int,
    complete_size: int,
) -> dict[str, Any]:
    full_population = f"full_{full_size}"
    complete_population = f"radiomics_complete_case_{complete_size}"
    pcr = _fixed_public_order(
        _pooled(loaded.new_baseline_public),
        kind="pcr",
        full_population=full_population,
        complete_population=complete_population,
    )
    phenotype = _fixed_public_order(
        _pooled(loaded.phenotype_public),
        kind="phenotype",
        full_population=full_population,
        complete_population=complete_population,
    )
    subtype = _fixed_public_order(
        _pooled(loaded.subtype_public),
        kind="subtype",
        full_population=full_population,
        complete_population=complete_population,
    )
    ftv = _fixed_public_order(
        _pooled(loaded.ftv_public),
        kind="ftv",
        full_population=full_population,
        complete_population=complete_population,
    )
    matched_phenotype = _fixed_public_order(
        _matched_probe_public(
            loaded.phenotype_public, loaded.old_phenotype_public, kind="phenotype"
        ),
        kind="phenotype",
        full_population=full_population,
        complete_population=complete_population,
    )
    matched_subtype = _fixed_public_order(
        _matched_probe_public(
            loaded.subtype_public, loaded.old_subtype_public, kind="subtype"
        ),
        kind="subtype",
        full_population=full_population,
        complete_population=complete_population,
    )
    matched_ftv = _fixed_public_order(
        _matched_probe_public(loaded.ftv_public, loaded.old_ftv_public, kind="ftv"),
        kind="ftv",
        full_population=full_population,
        complete_population=complete_population,
    )
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "model_name": MODEL_NAME,
        "posthoc": True,
        "changes_original_foundation_mri_conclusions": False,
        "reported_candidate_policy": (
            "all frozen candidates/axes/timings; no best filtering or metric sorting"
        ),
        "comparison_contract": static_comparison_contract(),
        "comparison_contract_sha256": comparison_contract_sha256(),
        "coverage": {
            "comparison_specs": len(specs),
            "paired_metric_rows": len(comparisons),
            "family_spec_counts": dict(FAMILY_COUNTS),
            "pcr_pooled_cells": len(pcr),
            "phenotype_pooled_cells": len(phenotype),
            "subtype_pooled_cells": len(subtype),
            "ftv_pooled_cells": len(ftv),
            "matched_phenotype_cells": len(matched_phenotype),
            "matched_subtype_cells": len(matched_subtype),
            "matched_ftv_cells": len(matched_ftv),
            "full_n": full_size,
            "complete_case_n": complete_size,
        },
        "inference_scope": INFERENCE_SCOPE,
        "caveats": {
            "pretraining_contamination": (
                "unknown: LVD-1689M is non-enumerable and cannot prove patient-level "
                "I-SPY exclusion"
            ),
            "license": "Meta DINOv3 custom license; institutional acceptance required",
            "register_tokens": (
                "four register tokens [1:5] are excluded; representation is CLS [0] "
                "concatenated with mean patch tokens [5:201]"
            ),
        },
        "input_sha256": dict(sorted(loaded.input_sha256.items())),
        "paired_comparisons": _public_records(comparisons),
        "pcr_pooled_metrics": _public_records(pcr),
        "phenotype_pooled_metrics": _public_records(phenotype),
        "subtype_pooled_metrics": _public_records(subtype),
        "ftv_pooled_metrics": _public_records(ftv),
        "dinov3_vs_dinov1_probe_descriptive": {
            "inference": "matched public pooled aggregates; absolute values and deltas only; no CI",
            "phenotype": _public_records(matched_phenotype),
            "subtype": _public_records(matched_subtype),
            "ftv": _public_records(matched_ftv),
        },
    }


def _format_number(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| "
        + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in row)
        + " |"
        for row in rows
    ]
    return "\n".join((header, separator, *body))


def _pcr_table(frame: pd.DataFrame) -> str:
    return _markdown_table(
        ("模型 variant", "空间", "时点", "人群", "n", "AUROC", "AUPRC", "Brier", "校准斜率", "ECE10"),
        (
            (
                row.model,
                row.spatial,
                row.timing,
                row.analysis_population,
                int(row.n),
                _format_number(row.auroc),
                _format_number(row.auprc),
                _format_number(row.brier),
                _format_number(row.calibration_slope),
                _format_number(row.ece_10bin),
            )
            for row in frame.itertuples(index=False)
        ),
    )


def _comparison_table(
    comparisons: pd.DataFrame, specs: Sequence[ComparisonSpec]
) -> str:
    lookup = {
        (str(row.comparison_id), str(row.metric)): row
        for row in comparisons.itertuples(index=False)
    }

    def interval(row: Any) -> str:
        return (
            f"{_format_number(row.delta_candidate_minus_reference)} "
            f"[{_format_number(row.ci_low)}, {_format_number(row.ci_high)}]"
        )

    rows = []
    for spec in specs:
        metric_rows = [lookup[(spec.comparison_id, metric)] for metric in METRICS]
        rows.append(
            (
                spec.family,
                spec.analysis_population,
                spec.timing,
                f"{spec.reference_model}@{spec.reference_spatial}",
                f"{spec.candidate_model}@{spec.candidate_spatial}",
                int(metric_rows[0].n_paired),
                interval(metric_rows[0]),
                interval(metric_rows[1]),
                interval(metric_rows[2]),
            )
        )
    return _markdown_table(
        (
            "比较族",
            "人群",
            "时点",
            "参照",
            "候选",
            "n",
            "ΔAUROC [95% CI]",
            "ΔAUPRC [95% CI]",
            "ΔBrier [95% CI]",
        ),
        rows,
    )


def _phenotype_table(frame: pd.DataFrame) -> str:
    def triplet(row: Any, metric: str) -> str:
        return (
            f"{_format_number(getattr(row, f'reference_{metric}'))} → "
            f"{_format_number(getattr(row, f'candidate_{metric}'))} "
            f"(Δ {_format_number(getattr(row, f'delta_candidate_minus_reference_{metric}'))})"
        )

    return _markdown_table(
        ("任务", "空间", "n", "AUROC DINOv1→v3 (Δ)", "AUPRC DINOv1→v3 (Δ)", "Brier DINOv1→v3 (Δ)"),
        (
            (
                row.target,
                row.spatial,
                int(row.n),
                triplet(row, "auroc"),
                triplet(row, "auprc"),
                triplet(row, "brier"),
            )
            for row in frame.itertuples(index=False)
        ),
    )


def _subtype_table(frame: pd.DataFrame) -> str:
    def triplet(row: Any, metric: str) -> str:
        return (
            f"{_format_number(getattr(row, f'reference_{metric}'))} → "
            f"{_format_number(getattr(row, f'candidate_{metric}'))} "
            f"(Δ {_format_number(getattr(row, f'delta_candidate_minus_reference_{metric}'))})"
        )

    return _markdown_table(
        (
            "空间",
            "n",
            "macro AUROC DINOv1→v3 (Δ)",
            "macro AUPRC DINOv1→v3 (Δ)",
            "Brier DINOv1→v3 (Δ)",
            "准确率 DINOv1→v3 (Δ)",
        ),
        (
            (
                row.spatial,
                int(row.n),
                triplet(row, "macro_ovr_auroc"),
                triplet(row, "macro_ovr_auprc"),
                triplet(row, "multiclass_brier"),
                triplet(row, "accuracy"),
            )
            for row in frame.itertuples(index=False)
        ),
    )


def _ftv_table(frame: pd.DataFrame) -> str:
    def triplet(row: Any, metric: str) -> str:
        return (
            f"{_format_number(getattr(row, f'reference_{metric}'))} → "
            f"{_format_number(getattr(row, f'candidate_{metric}'))} "
            f"(Δ {_format_number(getattr(row, f'delta_candidate_minus_reference_{metric}'))})"
        )

    return _markdown_table(
        (
            "空间",
            "任务",
            "终点",
            "n",
            "Spearman DINOv1→v3 (Δ)",
            "R² DINOv1→v3 (Δ)",
            "RMSE DINOv1→v3 (Δ)",
            "MAE DINOv1→v3 (Δ)",
        ),
        (
            (
                row.spatial,
                row.task,
                row.endpoint,
                int(row.n),
                triplet(row, "spearman"),
                triplet(row, "r2"),
                triplet(row, "rmse"),
                triplet(row, "mae"),
            )
            for row in frame.itertuples(index=False)
        ),
    )


def render_final_report(
    template_path: Path,
    loaded: _LoadedInputs,
    comparisons: pd.DataFrame,
    specs: Sequence[ComparisonSpec],
    *,
    full_size: int,
    complete_size: int,
) -> str:
    source = Path(template_path)
    if not source.is_file():
        raise FileNotFoundError("DINOv3 final report template is missing")
    full_population = f"full_{full_size}"
    complete_population = f"radiomics_complete_case_{complete_size}"
    pcr = _fixed_public_order(
        _pooled(loaded.new_baseline_public),
        kind="pcr",
        full_population=full_population,
        complete_population=complete_population,
    )
    phenotype = _fixed_public_order(
        _matched_probe_public(
            loaded.phenotype_public, loaded.old_phenotype_public, kind="phenotype"
        ),
        kind="phenotype",
        full_population=full_population,
        complete_population=complete_population,
    )
    subtype = _fixed_public_order(
        _matched_probe_public(
            loaded.subtype_public, loaded.old_subtype_public, kind="subtype"
        ),
        kind="subtype",
        full_population=full_population,
        complete_population=complete_population,
    )
    ftv = _fixed_public_order(
        _matched_probe_public(loaded.ftv_public, loaded.old_ftv_public, kind="ftv"),
        kind="ftv",
        full_population=full_population,
        complete_population=complete_population,
    )
    replacements = {
        "{{MODEL_NAME}}": MODEL_NAME,
        "{{FULL_SIZE}}": str(full_size),
        "{{COMPLETE_SIZE}}": str(complete_size),
        "{{SPEC_COUNT}}": str(len(specs)),
        "{{METRIC_ROW_COUNT}}": str(len(comparisons)),
        "{{CONTRACT_SHA256}}": comparison_contract_sha256(),
        "{{PCR_TABLE}}": _pcr_table(pcr),
        "{{COMPARISON_TABLE}}": _comparison_table(comparisons, specs),
        "{{PHENOTYPE_TABLE}}": _phenotype_table(phenotype),
        "{{SUBTYPE_TABLE}}": _subtype_table(subtype),
        "{{FTV_TABLE}}": _ftv_table(ftv),
    }
    text = source.read_text(encoding="utf-8")
    for placeholder, value in replacements.items():
        if placeholder not in text:
            raise ValueError(f"final report template is missing {placeholder}")
        text = text.replace(placeholder, value)
    if re.search(r"\{\{[A-Z0-9_]+\}\}", text):
        raise ValueError("final report template contains an unresolved placeholder")
    return text.rstrip() + "\n"


def _matplotlib_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("matplotlib is required for DINOv3 reporting") from error
    return plt


def render_timing_figure(
    baseline_public: pd.DataFrame,
    destination: Path,
    *,
    full_population: str,
) -> None:
    plt = _matplotlib_pyplot()
    pooled = _pooled(baseline_public)
    selected = pooled.loc[
        pooled["analysis_population"].astype(str).eq(full_population)
        & pooled["model"].astype(str).isin(
            f"{MODEL_NAME}_{variant}" for variant in _FULL_VARIANTS
        )
    ].copy()
    expected = len(_FULL_VARIANTS) * len(SPATIAL_AXES) * len(DECISION_POINTS)
    if len(selected) != expected:
        raise ValueError("timing figure does not have every frozen DINOv3 cell")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True)
    x = np.arange(len(DECISION_POINTS))
    for axis, metric, title in zip(
        axes, ("auroc", "auprc"), ("pCR AUROC", "pCR AUPRC"), strict=True
    ):
        for variant in _FULL_VARIANTS:
            for spatial in SPATIAL_AXES:
                model = f"{MODEL_NAME}_{variant}"
                group = selected.loc[
                    selected["model"].astype(str).eq(model)
                    & selected["spatial"].astype(str).eq(spatial)
                ]
                values = {
                    str(row.timing): float(getattr(row, metric))
                    for row in group.itertuples(index=False)
                }
                if tuple(values) != DECISION_POINTS:
                    values = {timing: values[timing] for timing in DECISION_POINTS}
                axis.plot(
                    x,
                    [values[timing] for timing in DECISION_POINTS],
                    marker="o",
                    linewidth=1.7,
                    label=f"{variant} · {spatial}",
                )
        axis.set_title(title)
        axis.set_xticks(x, DECISION_POINTS)
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel("Prediction timing")
        axis.set_ylabel(metric.upper())
        axis.grid(alpha=0.25, linewidth=0.7)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.suptitle("DINOv3 post-hoc sensitivity: every full-cohort pCR arm")
    fig.text(
        0.01,
        0.01,
        "Fixed protocol order; no best-cell filtering or performance sorting.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 0.82, 0.95))
    fig.savefig(destination, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_comparison_figure(
    comparisons: pd.DataFrame,
    specs: Sequence[ComparisonSpec],
    destination: Path,
) -> None:
    """Plot all 84 specs for all three metrics in frozen protocol order."""

    plt = _matplotlib_pyplot()
    lookup = {
        (str(row.comparison_id), str(row.metric)): row
        for row in comparisons.itertuples(index=False)
    }
    x = np.arange(len(specs))
    family_colors = {
        "dinov3_vs_dinov1": "#1f77b4",
        "local_vs_global": "#ff7f0e",
        "dinov3_vs_current_cnn_full": "#2ca02c",
        "clinical_gain": "#9467bd",
        "beyond_ftv": "#8c564b",
    }
    fig, axes = plt.subplots(3, 1, figsize=(19, 12), sharex=True)
    for axis, metric in zip(axes, METRICS, strict=True):
        rows = [lookup[(spec.comparison_id, metric)] for spec in specs]
        delta = np.asarray(
            [row.delta_candidate_minus_reference for row in rows], dtype=float
        )
        low = np.asarray([row.ci_low for row in rows], dtype=float)
        high = np.asarray([row.ci_high for row in rows], dtype=float)
        colors = [family_colors[spec.family] for spec in specs]
        axis.vlines(x, low, high, color=colors, alpha=0.55, linewidth=0.8)
        axis.scatter(x, delta, c=colors, s=13, zorder=3)
        axis.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        axis.set_ylabel(f"Δ{metric.upper()}")
        axis.grid(axis="y", alpha=0.2, linewidth=0.6)
    boundaries = []
    cursor = 0
    for family, count in FAMILY_COUNTS.items():
        midpoint = cursor + (count - 1) / 2
        boundaries.append((family, midpoint))
        cursor += count
        if cursor < len(specs):
            for axis in axes:
                axis.axvline(cursor - 0.5, color="#999999", linewidth=0.6)
    axes[-1].set_xticks(
        [midpoint for _, midpoint in boundaries],
        [family for family, _ in boundaries],
        rotation=15,
        ha="right",
    )
    axes[-1].set_xlabel("All 84 comparison specs in frozen protocol order")
    fig.suptitle("Paired DINOv3 post-hoc comparisons: all 252 metric intervals")
    fig.text(
        0.01,
        0.01,
        "Candidate minus reference; descriptive paired patient bootstrap only.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(destination, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _assert_output_targets(outputs: ReportingOutputs) -> None:
    paths = [Path(value) for value in asdict(outputs).values()]
    if len(set(paths)) != len(paths):
        raise ValueError("reporting output paths must be distinct")
    existing = [path.name for path in paths if os.path.lexists(path)]
    if existing:
        raise FileExistsError(f"reporting outputs already exist: {existing}")


def _publish_no_overwrite_with_rollback(
    staged: Mapping[str, Path],
    destinations: Mapping[str, Path],
    *,
    marker_field: str,
) -> None:
    if set(staged) != set(destinations) or marker_field not in staged:
        raise ValueError("staged/destination reporting fields drifted")
    order = [field for field in destinations if field != marker_field] + [marker_field]
    published: list[Path] = []
    try:
        for field in order:
            destination = Path(destinations[field])
            destination.parent.mkdir(parents=True, exist_ok=True)
            if os.path.lexists(destination):
                raise FileExistsError(destination)
            os.link(staged[field], destination)
            os.chmod(destination, 0o644)
            published.append(destination)
    except Exception:
        for destination in reversed(published):
            destination.unlink(missing_ok=True)
        raise


def _public_artifact_hashes(staged: Mapping[str, Path]) -> dict[str, str]:
    roles = {
        "paired_comparisons": "paired_csv",
        "results_summary": "summary_json",
        "final_report": "final_report",
        "timing_figure": "timing_figure",
        "comparison_figure": "comparison_figure",
    }
    return {role: file_sha256(staged[field]) for role, field in roles.items()}


def _sanitize_report_lock(lineage: Mapping[str, Any]) -> dict[str, str]:
    required = {"lock_sha256", "argv_sha256", "consumer"}
    if not required.issubset(lineage):
        raise ValueError("report lock receipt lacks lock/argv identity")
    if lineage["consumer"] != "report":
        raise ValueError("report lock receipt has the wrong consumer")
    empty_argv_sha256 = hashlib.sha256(b"[]\n").hexdigest()
    argv_digest = _checked_digest(lineage["argv_sha256"], label="report argv")
    if argv_digest != empty_argv_sha256:
        raise ValueError("formal report command must have the exact empty argv")
    return {
        "lock_sha256": _checked_digest(lineage["lock_sha256"], label="report lock"),
        "argv_sha256": argv_digest,
    }


def _build_reporting_marker(
    *,
    staged: Mapping[str, Path],
    input_sha256: Mapping[str, str],
    report_lock: Mapping[str, Any] | None,
    producer_lineage: Mapping[str, Any] | None,
) -> dict[str, Any]:
    marker: dict[str, Any] = {
        "schema_version": REPORTING_MARKER_SCHEMA_VERSION,
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "model_name": MODEL_NAME,
        "posthoc": True,
        "comparison_contract_sha256": comparison_contract_sha256(),
        "comparison_spec_count": EXPECTED_COMPARISON_COUNT,
        "paired_metric_row_count": EXPECTED_PAIRED_METRIC_ROWS,
        "input_sha256": dict(sorted(input_sha256.items())),
        "public_artifact_sha256": _public_artifact_hashes(staged),
        "published_last": True,
    }
    if report_lock is None:
        marker["lineage_mode"] = "synthetic"
    else:
        if producer_lineage is None or set(producer_lineage) != {
            "baseline",
            "probe",
            "parent_comparator",
        }:
            raise ValueError("formal producer lineage is incomplete")
        marker["lineage_mode"] = "formal"
        marker["report_lock"] = _sanitize_report_lock(report_lock)
        marker["producers"] = dict(producer_lineage)
    return marker


def _assert_no_public_leakage(
    *,
    comparisons: pd.DataFrame,
    summary_text: str,
    report_text: str,
    marker_text: str,
    patient_ids: Sequence[str],
) -> None:
    ensure_public_safe(comparisons)
    if "patient_id" in comparisons.columns:
        raise ValueError("public paired CSV contains patient_id")
    combined_text = "\n".join((summary_text, report_text, marker_text))
    lowered = combined_text.lower()
    if "patient_id" in lowered or ".private." in lowered:
        raise ValueError("public reporting text leaks a private schema/path")
    if re.search(r"(?:^|[\s\"'])/(?:home|data|tmp|mnt|root)/", combined_text):
        raise ValueError("public reporting text contains an absolute path")
    leaked = [patient for patient in patient_ids if patient and patient in combined_text]
    if leaked:
        raise ValueError("public reporting text contains a patient identifier")


def summarize_results(
    inputs: ReportingInputs,
    outputs: ReportingOutputs,
    *,
    template_path: Path,
    full_size: int = FULL_SIZE,
    complete_size: int = COMPLETE_SIZE,
    report_lock: Mapping[str, Any] | None = None,
    producer_receipts: Mapping[str, Mapping[str, Any]] | None = None,
    parent_comparator_artifacts: Mapping[str, Any] | None = None,
    strict_modes: bool = True,
) -> dict[str, int]:
    """Verify, summarize and atomically publish six identifier-free outputs."""

    _assert_output_targets(outputs)
    producer_lineage: dict[str, Any] | None = None
    if report_lock is not None:
        sanitized_lock = _sanitize_report_lock(report_lock)
        if producer_receipts is None or set(producer_receipts) != {"baseline", "probe"}:
            raise ValueError("formal reporting requires both verified producer receipts")
        if parent_comparator_artifacts is None:
            raise ValueError("formal reporting requires hash-pinned parent comparators")
        producer_lineage = {
            consumer: _sanitize_producer_receipt(
                producer_receipts[consumer],
                consumer=consumer,
                report_lock=report_lock,
                inputs=inputs,
                strict_modes=strict_modes,
            )
            for consumer in ("baseline", "probe")
        }
        producer_lineage["parent_comparator"] = _sanitize_parent_comparator_artifacts(
            parent_comparator_artifacts, inputs
        )
        # Keep the sanitized value alive for the marker; validation above also
        # establishes the exact empty-argv report gate before any CSV parse.
        if sanitized_lock["lock_sha256"] != producer_lineage["baseline"]["evaluation_lock_sha256"]:
            raise AssertionError("sanitized formal lineage drifted")
    elif producer_receipts is not None or parent_comparator_artifacts is not None:
        raise ValueError("formal lineage cannot be supplied without a report lock")

    # No outcome CSV is parsed before every applicable receipt/hash gate above.
    loaded = _load_and_validate(inputs, full_size=full_size, complete_size=complete_size)
    full_population = f"full_{full_size}"
    complete_population = f"radiomics_complete_case_{complete_size}"
    specs = build_comparison_specs(
        loaded.combined_baseline.loc[:, _LOGICAL_BINARY_IDENTITY],
        full_population=full_population,
        complete_population=complete_population,
    )
    comparisons = paired_bootstrap_comparisons(
        loaded.combined_baseline,
        specs,
        patient_orders={
            full_population: loaded.full_patient_order,
            complete_population: loaded.complete_patient_order,
        },
    )
    comparisons = _ordered_comparisons(comparisons, specs)
    if len(comparisons) != EXPECTED_PAIRED_METRIC_ROWS:
        raise AssertionError("paired comparison output must contain exactly 252 rows")
    summary = build_public_summary(
        loaded,
        comparisons,
        specs,
        full_size=full_size,
        complete_size=complete_size,
    )
    report = render_final_report(
        template_path,
        loaded,
        comparisons,
        specs,
        full_size=full_size,
        complete_size=complete_size,
    )

    output_paths = [Path(value) for value in asdict(outputs).values()]
    common_parent = Path(
        os.path.commonpath([str(path.parent.resolve()) for path in output_paths])
    )
    common_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dinov3-reporting-", dir=common_parent) as name:
        staging = Path(name)
        staged = {
            "paired_csv": staging / "paired.csv",
            "summary_json": staging / "summary.json",
            "final_report": staging / "final_report.md",
            "timing_figure": staging / "timing.png",
            "comparison_figure": staging / "comparisons.png",
        }
        comparisons.to_csv(staged["paired_csv"], index=False)
        summary_text = _json_text(summary)
        staged["summary_json"].write_text(summary_text, encoding="utf-8")
        staged["final_report"].write_text(report, encoding="utf-8")
        render_timing_figure(
            loaded.new_baseline_public,
            staged["timing_figure"],
            full_population=full_population,
        )
        render_comparison_figure(comparisons, specs, staged["comparison_figure"])
        marker = _build_reporting_marker(
            staged=staged,
            input_sha256=loaded.input_sha256,
            report_lock=report_lock,
            producer_lineage=producer_lineage,
        )
        marker_text = _json_text(marker)
        staged["reporting_marker"] = staging / "reporting_run_provenance.json"
        staged["reporting_marker"].write_text(marker_text, encoding="utf-8")
        _assert_no_public_leakage(
            comparisons=comparisons,
            summary_text=summary_text,
            report_text=report,
            marker_text=marker_text,
            patient_ids=(*loaded.full_patient_order, *loaded.complete_patient_order),
        )
        destinations = {
            field: Path(destination)
            for field, destination in asdict(outputs).items()
        }
        _publish_no_overwrite_with_rollback(
            staged,
            destinations,
            marker_field="reporting_marker",
        )
    return {
        "comparison_specs": len(specs),
        "paired_metric_rows": len(comparisons),
        "pcr_pooled_cells": len(_pooled(loaded.new_baseline_public)),
        "public_outputs": len(ReportingOutputs.__dataclass_fields__),
        "reporting_marker_published_last": 1,
    }


__all__ = [
    "COMPARISON_CONTRACT_VERSION",
    "COMPLETE_POPULATION",
    "COMPLETE_SIZE",
    "DINO_V1_MODEL",
    "EXPECTED_COMPARISON_COUNT",
    "EXPECTED_PAIRED_METRIC_ROWS",
    "FAMILY_COUNTS",
    "FULL_POPULATION",
    "FULL_SIZE",
    "MODEL_NAME",
    "PRODUCER_RECEIPT_SCHEMA_VERSION",
    "ReportingInputs",
    "ReportingOutputs",
    "build_comparison_specs",
    "comparison_contract_sha256",
    "static_comparison_contract",
    "summarize_results",
]

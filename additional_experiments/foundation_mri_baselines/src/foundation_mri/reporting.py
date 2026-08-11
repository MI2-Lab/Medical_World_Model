"""Fail-closed public reporting for the frozen DCE-MRI baseline study.

The module consumes the private outer-fold test predictions written by
``run_baselines.py``/``run_probes.py`` and their public aggregate companions.
It never fits or selects a model.  Every comparison is generated from a
predeclared naming/estimand contract, and every formal candidate is retained.
Patient identifiers are used only in memory to align paired resamples and are
never included in a returned public table or report payload.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .data import FOLDS, file_sha256
from .evaluation import (
    aggregate_binary_predictions,
    aggregate_continuous_predictions,
    aggregate_multiclass_predictions,
    ensure_public_safe,
)


BOOTSTRAP_SEED = 2026
BOOTSTRAP_REPLICATES = 5000
CI_LEVEL = 0.95
COMPARISON_CONTRACT_VERSION = "foundation_mri_paired_comparisons_v1"
COMPARISON_CONTRACT_CANONICAL_SHA256 = (
    "f99fd76bd35b784500194347c4b363725616ca6adab2ba830a006cc7cc4a7e13"
)
SUMMARY_SCHEMA_VERSION = "foundation_mri_results_summary_v1"
REPORTING_PROVENANCE_SCHEMA_VERSION = "foundation_mri_reporting_run_provenance_v1"
MIN_PUBLIC_CALIBRATION_BIN_N = 10
STRICT_PUBLIC_RECOMPUTE_RTOL = 1e-10
STRICT_PUBLIC_RECOMPUTE_ATOL = 1e-12
# Binary calibration is obtained by IRLS whose locked stopping threshold is
# 1e-9.  Recomputing from serialized OOF values plus IRLS/linear-algebra
# variation can perturb the final iterate below that scale, so only the two
# IRLS identities receive this one-order relative safety margin.
BINARY_IRLS_RECOMPUTE_RTOL = 1e-8
BINARY_IRLS_RECOMPUTE_ATOL = 1e-9
BINARY_IRLS_COLUMNS = frozenset({"calibration_slope", "calibration_intercept"})
INFERENCE_SCOPE = (
    "descriptive paired outer-fold OOF patient bootstrap; not confirmatory and "
    "not used for model or checkpoint selection"
)
DECISION_POINTS = ("T0", "T0-T1", "T0-T2")
METRICS = ("auroc", "auprc", "brier")
HIGHER_IS_BETTER = {"auroc": True, "auprc": True, "brier": False}

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
_SOURCE_SUFFIXES = (
    "_mri_clinical_radiomics",
    "_mri_clinical_ftv",
    "_mri_clinical_paired",
    "_mri_radiomics",
    "_mri_only_paired",
    "_mri_ftv",
    "_mri_clinical",
    "_mri_only",
)


@dataclass(frozen=True)
class ReportingInputs:
    """The eight outputs required from the two locked evaluation CLIs."""

    baseline_private: Path
    baseline_public: Path
    phenotype_private: Path
    phenotype_public: Path
    subtype_private: Path
    subtype_public: Path
    ftv_private: Path
    ftv_public: Path


@dataclass(frozen=True)
class ReportingOutputs:
    paired_csv: Path
    summary_json: Path
    summary_markdown: Path
    timing_figure: Path
    calibration_figure: Path


@dataclass(frozen=True)
class ComparisonSpec:
    """One test-independent, paired OOF comparison."""

    comparison_id: str
    family: str
    estimand: str
    analysis_population: str
    timing: str
    reference_model: str
    reference_spatial: str
    candidate_model: str
    candidate_spatial: str


@dataclass(frozen=True)
class _LoadedInputs:
    baseline_private: pd.DataFrame
    baseline_public: pd.DataFrame
    phenotype_private: pd.DataFrame
    phenotype_public: pd.DataFrame
    subtype_private: pd.DataFrame
    subtype_public: pd.DataFrame
    ftv_private: pd.DataFrame
    ftv_public: pd.DataFrame
    input_sha256: Mapping[str, str]
    full_patient_order: tuple[str, ...]
    complete_patient_order: tuple[str, ...]
    foundation_models: tuple[str, ...]


def static_comparison_contract() -> dict[str, Any]:
    """Return the serialisable, pre-test comparison-generation contract."""

    return {
        "version": COMPARISON_CONTRACT_VERSION,
        "bootstrap": {
            "unit": "patient",
            "pairing": "identical OOF patients and identical resample weights",
            "method": "nonparametric multinomial resampling with replacement",
            "master_seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "ci_level": CI_LEVEL,
            "interval": "percentile",
        },
        "metrics": {
            "auroc": "candidate minus reference; higher is better",
            "auprc": "candidate minus reference; higher is better",
            "brier": "candidate minus reference; lower is better",
        },
        "candidate_policy": (
            "infer every formal foundation source from full-cohort *_mri_only rows; "
            "require and report every source at GLOBAL and LOCAL without test filtering"
        ),
        "families": [
            {
                "name": "clinical_gain",
                "populations": ["full", "radiomics_complete_case"],
                "rule": (
                    "clinical-only versus clinical plus each imaging source; includes all "
                    "foundation axes and the preregistered GAP0/LOCAL0 comparators"
                ),
            },
            {
                "name": "local_vs_global",
                "populations": ["full", "radiomics_complete_case"],
                "rule": (
                    "LOCAL minus GLOBAL for each foundation source, and LOCAL0 minus GAP0 "
                    "for current CNN; MRI-only and MRI+clinical estimands"
                ),
            },
            {
                "name": "foundation_vs_current_cnn",
                "populations": ["full", "radiomics_complete_case"],
                "rule": (
                    "each foundation GLOBAL versus GAP0 and each foundation LOCAL versus "
                    "LOCAL0; MRI-only and MRI+clinical estimands"
                ),
            },
            {
                "name": "beyond_ftv",
                "populations": ["radiomics_complete_case"],
                "rule": (
                    "clinical+FTV versus clinical+FTV+foundation MRI for every foundation "
                    "source and both spatial axes"
                ),
            },
        ],
        "probe_policy": (
            "require HR, HER2, T0 four-class subtype, static FTV, and delta-FTV "
            "for every foundation GLOBAL/LOCAL source and for GAP0@GLOBAL/LOCAL0@LOCAL"
        ),
        "timings": list(DECISION_POINTS),
        "inference_scope": INFERENCE_SCOPE,
    }


def _read_csv(path: Path, *, private: bool) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"required reporting input is missing: {source.name}")
    if private and not source.name.endswith(".private.csv"):
        raise ValueError(f"private reporting input must end in .private.csv: {source.name}")
    frame = pd.read_csv(source, dtype={"patient_id": str})
    if frame.empty:
        raise ValueError(f"required reporting input is empty: {source.name}")
    return frame


def _assert_public_matches(
    observed: pd.DataFrame,
    recomputed: pd.DataFrame,
    *,
    key_columns: Sequence[str],
    label: str,
    relaxed_numeric_columns: Iterable[str] = (),
) -> None:
    """Match public CSV to private OOF under locked per-column tolerances."""

    ensure_public_safe(observed)
    ensure_public_safe(recomputed)
    if set(observed.columns) != set(recomputed.columns):
        raise ValueError(
            f"{label} public schema drifted: expected={sorted(recomputed.columns)}, "
            f"observed={sorted(observed.columns)}"
        )
    if observed.duplicated(list(key_columns)).any() or recomputed.duplicated(
        list(key_columns)
    ).any():
        raise ValueError(f"{label} public aggregate has duplicate identities")
    left = observed.sort_values(list(key_columns), kind="stable").reset_index(drop=True)
    right = recomputed.sort_values(list(key_columns), kind="stable").reset_index(drop=True)
    if len(left) != len(right):
        raise ValueError(f"{label} public aggregate row count drifted")
    relaxed = frozenset(str(value) for value in relaxed_numeric_columns)
    unsupported_relaxations = sorted(relaxed.difference(BINARY_IRLS_COLUMNS))
    if unsupported_relaxations:
        raise ValueError(
            f"{label} requested non-IRLS relaxed columns: {unsupported_relaxations}"
        )
    if relaxed and label not in {"baseline", "phenotype"}:
        raise ValueError(
            "IRLS numeric relaxation is restricted to baseline/phenotype binary aggregates"
        )
    unknown_relaxations = sorted(relaxed.difference(right.columns))
    if unknown_relaxations:
        raise ValueError(
            f"{label} requested unknown relaxed numeric columns: {unknown_relaxations}"
        )
    for column in right.columns:
        if pd.api.types.is_numeric_dtype(right[column]):
            actual = pd.to_numeric(left[column], errors="coerce").to_numpy(dtype=np.float64)
            expected = pd.to_numeric(right[column], errors="coerce").to_numpy(dtype=np.float64)
            if column in relaxed:
                rtol, atol = BINARY_IRLS_RECOMPUTE_RTOL, BINARY_IRLS_RECOMPUTE_ATOL
            else:
                rtol, atol = STRICT_PUBLIC_RECOMPUTE_RTOL, STRICT_PUBLIC_RECOMPUTE_ATOL
            if not np.allclose(
                actual, expected, rtol=rtol, atol=atol, equal_nan=True
            ):
                raise ValueError(f"{label} public aggregate drifted in numeric column {column}")
        else:
            actual = left[column].fillna("<NA>").astype(str).to_numpy()
            expected = right[column].fillna("<NA>").astype(str).to_numpy()
            if not np.array_equal(actual, expected):
                raise ValueError(f"{label} public aggregate drifted in column {column}")


def _normalise_private_ids(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if "patient_id" not in frame.columns:
        raise ValueError(f"{label} private OOF is missing patient_id")
    output = frame.copy()
    output["patient_id"] = output["patient_id"].astype(str).str.strip()
    if output["patient_id"].eq("").any() or output["patient_id"].str.lower().isin(
        {"nan", "none", "<na>"}
    ).any():
        raise ValueError(f"{label} private OOF contains an empty patient ID")
    if "split" not in output or set(output["split"].astype(str)) != {"test"}:
        raise ValueError(f"{label} private OOF must contain test-only rows")
    fold = pd.to_numeric(output.get("fold"), errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(fold).all() or not np.equal(fold, np.floor(fold)).all():
        raise ValueError(f"{label} private OOF fold values must be finite integers")
    output["fold"] = fold.astype(np.int64)
    return output


def _population_size(population: str, full_size: int, complete_size: int) -> int:
    sizes = {
        f"full_{full_size}": full_size,
        f"radiomics_complete_case_{complete_size}": complete_size,
    }
    if population not in sizes:
        raise ValueError(f"unexpected analysis population in reporting input: {population}")
    return sizes[population]


def _validate_group_coverage(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    full_size: int,
    complete_size: int,
    label: str,
) -> dict[str, tuple[str, ...]]:
    required = {"patient_id", "fold", "split", "analysis_population", *group_columns}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} private OOF is missing columns: {missing}")
    population_orders: dict[str, tuple[str, ...]] = {}
    for keys, group in frame.groupby(list(group_columns), sort=True, dropna=False):
        population = str(group["analysis_population"].iloc[0])
        expected_n = _population_size(population, full_size, complete_size)
        if len(group) != expected_n or group["patient_id"].duplicated().any():
            raise ValueError(
                f"{label} OOF group {keys} must contain {expected_n} unique patients"
            )
        if set(group["fold"].astype(int)) != set(FOLDS):
            raise ValueError(f"{label} OOF group {keys} does not cover all five folds")
        order = tuple(sorted(group["patient_id"].astype(str)))
        prior = population_orders.setdefault(population, order)
        if prior != order:
            raise ValueError(f"{label} silently changed patients within {population}")
    return population_orders


def _binary_truth_map(frame: pd.DataFrame, population: str) -> dict[str, int]:
    current = frame.loc[frame["analysis_population"].astype(str).eq(population)]
    pairs = current.loc[:, ["patient_id", "y_true"]].drop_duplicates()
    if pairs["patient_id"].duplicated().any():
        raise ValueError(f"binary truth changes across models in {population}")
    truth = pd.to_numeric(pairs["y_true"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(truth).all() or not np.isin(truth, (0.0, 1.0)).all():
        raise ValueError(f"binary truth is not complete 0/1 in {population}")
    return dict(zip(pairs["patient_id"].astype(str), truth.astype(int), strict=True))


def _validate_probe_contract(
    *,
    phenotype: pd.DataFrame,
    subtype: pd.DataFrame,
    ftv: pd.DataFrame,
    foundation_models: Sequence[str],
    full_population: str,
    complete_population: str,
) -> None:
    source_axes = [
        *((source, spatial) for source in foundation_models for spatial in ("GLOBAL", "LOCAL")),
        ("GAP0", "GLOBAL"),
        ("LOCAL0", "LOCAL"),
    ]
    for source, spatial in source_axes:
        for target in ("HR", "HER2"):
            count = len(
                phenotype.loc[
                    phenotype["target"].astype(str).eq(target)
                    & phenotype["model"].astype(str).eq(source)
                    & phenotype["spatial"].astype(str).eq(spatial)
                    & phenotype["timing"].astype(str).eq("T0")
                    & phenotype["analysis_population"].astype(str).eq(full_population)
                ]
            )
            if count == 0:
                raise ValueError(f"missing preregistered {target} probe for {source}/{spatial}")
        subtype_count = len(
            subtype.loc[
                subtype["target"].astype(str).eq("HR_HER2_subtype")
                & subtype["model"].astype(str).eq(source)
                & subtype["spatial"].astype(str).eq(spatial)
                & subtype["timing"].astype(str).eq("T0")
                & subtype["analysis_population"].astype(str).eq(full_population)
            ]
        )
        if subtype_count == 0:
            raise ValueError(f"missing preregistered subtype probe for {source}/{spatial}")
        required_ftv = {
            ("static", endpoint) for endpoint in ("T0", "T1", "T2", "T3")
        } | {("delta", endpoint) for endpoint in ("T0-T1", "T1-T2", "T2-T3")}
        observed_ftv = set(
            zip(
                ftv.loc[
                    ftv["target"].astype(str).eq("FTV")
                    & ftv["model"].astype(str).eq(source)
                    & ftv["spatial"].astype(str).eq(spatial)
                    & ftv["analysis_population"].astype(str).eq(complete_population),
                    "task",
                ].astype(str),
                ftv.loc[
                    ftv["target"].astype(str).eq("FTV")
                    & ftv["model"].astype(str).eq(source)
                    & ftv["spatial"].astype(str).eq(spatial)
                    & ftv["analysis_population"].astype(str).eq(complete_population),
                    "endpoint",
                ].astype(str),
                strict=True,
            )
        )
        if not required_ftv.issubset(observed_ftv):
            missing = sorted(required_ftv.difference(observed_ftv))
            raise ValueError(f"missing preregistered FTV probes for {source}/{spatial}: {missing}")


def _source_from_model(model: str) -> str | None:
    for suffix in _SOURCE_SUFFIXES:
        if model.endswith(suffix):
            source = model[: -len(suffix)]
            return source or None
    return None


def _foundation_sources(baseline: pd.DataFrame, full_population: str) -> tuple[str, ...]:
    rows = baseline.loc[
        baseline["target"].astype(str).eq("pCR")
        & baseline["analysis_population"].astype(str).eq(full_population)
        & baseline["model"].astype(str).str.endswith("_mri_only")
    ]
    by_source: dict[str, set[str]] = {}
    for model, spatial in rows.loc[:, ["model", "spatial"]].drop_duplicates().itertuples(
        index=False, name=None
    ):
        source = _source_from_model(str(model))
        if source is not None:
            by_source.setdefault(source, set()).add(str(spatial))
    for current, expected_spatial in (("GAP0", "GLOBAL"), ("LOCAL0", "LOCAL")):
        if by_source.get(current) != {expected_spatial}:
            raise ValueError(f"missing or spatially invalid preregistered {current} comparator")
    foundation = tuple(sorted(set(by_source).difference({"GAP0", "LOCAL0"})))
    if not foundation:
        raise ValueError("no formal foundation candidate was found")
    for source in foundation:
        if by_source[source] != {"GLOBAL", "LOCAL"}:
            raise ValueError(f"foundation candidate {source} must contain GLOBAL and LOCAL")
    return foundation


def _load_and_validate(
    inputs: ReportingInputs, *, full_size: int, complete_size: int
) -> _LoadedInputs:
    if full_size <= complete_size or complete_size <= 0:
        raise ValueError("reporting cohort sizes must satisfy full > complete > 0")
    baseline = _normalise_private_ids(
        _read_csv(inputs.baseline_private, private=True), "baseline"
    )
    phenotype = _normalise_private_ids(
        _read_csv(inputs.phenotype_private, private=True), "phenotype"
    )
    subtype = _normalise_private_ids(_read_csv(inputs.subtype_private, private=True), "subtype")
    ftv = _normalise_private_ids(_read_csv(inputs.ftv_private, private=True), "FTV")
    baseline_public = _read_csv(inputs.baseline_public, private=False)
    phenotype_public = _read_csv(inputs.phenotype_public, private=False)
    subtype_public = _read_csv(inputs.subtype_public, private=False)
    ftv_public = _read_csv(inputs.ftv_public, private=False)

    baseline_orders = _validate_group_coverage(
        baseline,
        group_columns=_BINARY_GROUP,
        full_size=full_size,
        complete_size=complete_size,
        label="baseline",
    )
    phenotype_orders = _validate_group_coverage(
        phenotype,
        group_columns=_BINARY_GROUP,
        full_size=full_size,
        complete_size=complete_size,
        label="phenotype",
    )
    subtype_orders = _validate_group_coverage(
        subtype,
        group_columns=_BINARY_GROUP,
        full_size=full_size,
        complete_size=complete_size,
        label="subtype",
    )
    ftv_orders = _validate_group_coverage(
        ftv,
        group_columns=_CONTINUOUS_GROUP,
        full_size=full_size,
        complete_size=complete_size,
        label="FTV",
    )
    full_population = f"full_{full_size}"
    complete_population = f"radiomics_complete_case_{complete_size}"
    if set(baseline_orders) != {full_population, complete_population}:
        raise ValueError("baseline predictions must include full and complete-case populations")
    if set(phenotype_orders) != {full_population} or set(subtype_orders) != {full_population}:
        raise ValueError("phenotype/subtype probes must cover exactly the full population")
    if set(ftv_orders) != {complete_population}:
        raise ValueError("FTV probes must cover exactly the complete-case population")
    if phenotype_orders[full_population] != baseline_orders[full_population]:
        raise ValueError("phenotype and pCR full-cohort patient coverage differs")
    if subtype_orders[full_population] != baseline_orders[full_population]:
        raise ValueError("subtype and pCR full-cohort patient coverage differs")
    if ftv_orders[complete_population] != baseline_orders[complete_population]:
        raise ValueError("FTV and pCR complete-case patient coverage differs")
    if not set(baseline_orders[complete_population]).issubset(
        baseline_orders[full_population]
    ):
        raise ValueError("complete-case patients are not a subset of the full cohort")
    full_truth = _binary_truth_map(baseline, full_population)
    complete_truth = _binary_truth_map(baseline, complete_population)
    if any(full_truth[patient] != truth for patient, truth in complete_truth.items()):
        raise ValueError("pCR truth differs between full and complete-case populations")
    if set(baseline["target"].astype(str)) != {"pCR"}:
        raise ValueError("baseline private input must contain only the pCR target")

    recomputed_baseline = aggregate_binary_predictions(baseline)
    recomputed_phenotype = aggregate_binary_predictions(phenotype)
    recomputed_subtype = aggregate_multiclass_predictions(subtype)
    recomputed_ftv = aggregate_continuous_predictions(ftv)
    binary_keys = [*_BINARY_GROUP, "aggregation"]
    _assert_public_matches(
        baseline_public,
        recomputed_baseline,
        key_columns=binary_keys,
        label="baseline",
        relaxed_numeric_columns=BINARY_IRLS_COLUMNS,
    )
    _assert_public_matches(
        phenotype_public,
        recomputed_phenotype,
        key_columns=binary_keys,
        label="phenotype",
        relaxed_numeric_columns=BINARY_IRLS_COLUMNS,
    )
    _assert_public_matches(
        subtype_public, recomputed_subtype, key_columns=binary_keys, label="subtype"
    )
    _assert_public_matches(
        ftv_public,
        recomputed_ftv,
        key_columns=[*_CONTINUOUS_GROUP, "aggregation"],
        label="FTV",
    )
    foundation = _foundation_sources(baseline, full_population)
    _validate_probe_contract(
        phenotype=phenotype,
        subtype=subtype,
        ftv=ftv,
        foundation_models=foundation,
        full_population=full_population,
        complete_population=complete_population,
    )
    hashes = {
        field: file_sha256(getattr(inputs, field))
        for field in ReportingInputs.__dataclass_fields__
    }
    return _LoadedInputs(
        baseline,
        baseline_public,
        phenotype,
        phenotype_public,
        subtype,
        subtype_public,
        ftv,
        ftv_public,
        hashes,
        baseline_orders[full_population],
        baseline_orders[complete_population],
        foundation,
    )


def _comparison_id(
    family: str,
    population: str,
    timing: str,
    reference_model: str,
    reference_spatial: str,
    candidate_model: str,
    candidate_spatial: str,
) -> str:
    raw = "|".join(
        (
            family,
            population,
            timing,
            f"{reference_model}@{reference_spatial}",
            f"{candidate_model}@{candidate_spatial}",
        )
    )
    suffix = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"{family}:{suffix}"


def build_comparison_contract(
    baseline: pd.DataFrame,
    *,
    foundation_models: Sequence[str],
    full_population: str,
    complete_population: str,
) -> tuple[ComparisonSpec, ...]:
    """Resolve the static contract against all candidates, without metric access."""

    identities = set(
        baseline.loc[:, ["target", "model", "spatial", "timing", "analysis_population"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    def require(model: str, spatial: str, timing: str, population: str) -> None:
        identity = ("pCR", model, spatial, timing, population)
        if identity not in identities:
            raise ValueError(
                "missing preregistered comparison cell: "
                f"model={model}, spatial={spatial}, timing={timing}, population={population}"
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
                _comparison_id(
                    family,
                    population,
                    timing,
                    reference_model,
                    reference_spatial,
                    candidate_model,
                    candidate_spatial,
                ),
                family,
                estimand,
                population,
                timing,
                reference_model,
                reference_spatial,
                candidate_model,
                candidate_spatial,
            )
        )

    source_axes = [
        *((source, spatial) for source in foundation_models for spatial in ("GLOBAL", "LOCAL")),
        ("GAP0", "GLOBAL"),
        ("LOCAL0", "LOCAL"),
    ]
    for timing in DECISION_POINTS:
        for source, spatial in source_axes:
            add(
                "clinical_gain",
                "clinical_plus_MRI_minus_clinical",
                full_population,
                timing,
                "clinical_only",
                "NONE",
                f"{source}_mri_clinical",
                spatial,
            )
            add(
                "clinical_gain",
                "clinical_plus_MRI_minus_clinical",
                complete_population,
                timing,
                "clinical_only_paired",
                "NONE",
                f"{source}_mri_clinical_paired",
                spatial,
            )

        for source in foundation_models:
            for population, suffixes in (
                (full_population, ("mri_only", "mri_clinical")),
                (complete_population, ("mri_only_paired", "mri_clinical_paired")),
            ):
                for suffix in suffixes:
                    add(
                        "local_vs_global",
                        f"LOCAL_minus_GLOBAL_{suffix}",
                        population,
                        timing,
                        f"{source}_{suffix}",
                        "GLOBAL",
                        f"{source}_{suffix}",
                        "LOCAL",
                    )
        for population, suffixes in (
            (full_population, ("mri_only", "mri_clinical")),
            (complete_population, ("mri_only_paired", "mri_clinical_paired")),
        ):
            for suffix in suffixes:
                add(
                    "local_vs_global",
                    f"LOCAL_minus_GLOBAL_current_CNN_{suffix}",
                    population,
                    timing,
                    f"GAP0_{suffix}",
                    "GLOBAL",
                    f"LOCAL0_{suffix}",
                    "LOCAL",
                )

        for source in foundation_models:
            for spatial, current in (("GLOBAL", "GAP0"), ("LOCAL", "LOCAL0")):
                for population, suffixes in (
                    (full_population, ("mri_only", "mri_clinical")),
                    (complete_population, ("mri_only_paired", "mri_clinical_paired")),
                ):
                    for suffix in suffixes:
                        add(
                            "foundation_vs_current_cnn",
                            f"foundation_minus_current_CNN_{suffix}",
                            population,
                            timing,
                            f"{current}_{suffix}",
                            spatial,
                            f"{source}_{suffix}",
                            spatial,
                        )
            for spatial in ("GLOBAL", "LOCAL"):
                add(
                    "beyond_ftv",
                    "clinical_FTV_foundation_minus_clinical_FTV",
                    complete_population,
                    timing,
                    "clinical_ftv",
                    "TABULAR",
                    f"{source}_mri_clinical_ftv",
                    spatial,
                )
    if len({spec.comparison_id for spec in specs}) != len(specs):
        raise AssertionError("resolved comparison contract contains duplicate IDs")
    return tuple(specs)


def _prediction_group(frame: pd.DataFrame, spec: ComparisonSpec, *, candidate: bool) -> pd.DataFrame:
    model = spec.candidate_model if candidate else spec.reference_model
    spatial = spec.candidate_spatial if candidate else spec.reference_spatial
    selected = frame.loc[
        frame["target"].astype(str).eq("pCR")
        & frame["model"].astype(str).eq(model)
        & frame["spatial"].astype(str).eq(spatial)
        & frame["timing"].astype(str).eq(spec.timing)
        & frame["analysis_population"].astype(str).eq(spec.analysis_population)
    ].copy()
    if selected.empty:
        raise ValueError(f"resolved comparison cell disappeared: {spec.comparison_id}")
    return selected.sort_values("patient_id", kind="stable").reset_index(drop=True)


def _bootstrap_weights(n: int, rng: np.random.Generator) -> np.ndarray:
    probabilities = np.full(n, 1.0 / n, dtype=np.float64)
    weights = rng.multinomial(n, probabilities, size=BOOTSTRAP_REPLICATES)
    if weights.max(initial=0) > np.iinfo(np.uint16).max:
        raise AssertionError("bootstrap multiplicity exceeds uint16")
    return weights.astype(np.uint16, copy=False)


def _weighted_metric_distribution(
    truth: np.ndarray, score: np.ndarray, weights: np.ndarray, *, chunk_size: int = 256
) -> dict[str, np.ndarray]:
    """Vectorised exact metrics for multinomial patient-bootstrap weights."""

    y = np.asarray(truth, dtype=np.int8)
    probability = np.asarray(score, dtype=np.float64)
    if y.shape != probability.shape or weights.shape[1] != len(y):
        raise ValueError("bootstrap metric inputs are not aligned")
    if set(y.tolist()) != {0, 1}:
        raise ValueError("bootstrap metrics require both classes")
    output = {
        metric: np.full(len(weights), np.nan, dtype=np.float64) for metric in METRICS
    }
    ascending = np.argsort(probability, kind="stable")
    asc_score = probability[ascending]
    asc_y = y[ascending]
    starts = np.r_[0, np.flatnonzero(np.diff(asc_score) != 0.0) + 1]
    descending_group_order = np.arange(len(starts) - 1, -1, -1)
    squared_error = (y - probability) ** 2
    for begin in range(0, len(weights), chunk_size):
        end = min(begin + chunk_size, len(weights))
        current = weights[begin:end].astype(np.float64, copy=False)
        total_positive = current @ y.astype(np.float64)
        total_negative = current.sum(axis=1) - total_positive
        valid = (total_positive > 0) & (total_negative > 0)
        output["brier"][begin:end] = current @ squared_error / current.sum(axis=1)

        sorted_weight = current[:, ascending]
        positive_group = np.add.reduceat(sorted_weight * asc_y, starts, axis=1)
        negative_group = np.add.reduceat(sorted_weight * (1 - asc_y), starts, axis=1)
        negative_before = np.cumsum(negative_group, axis=1) - negative_group
        auc_numerator = np.sum(
            positive_group * (negative_before + 0.5 * negative_group), axis=1
        )
        auc = np.full(len(current), np.nan, dtype=np.float64)
        auc[valid] = auc_numerator[valid] / (
            total_positive[valid] * total_negative[valid]
        )
        output["auroc"][begin:end] = auc

        positive_desc = positive_group[:, descending_group_order]
        total_desc = (positive_group + negative_group)[:, descending_group_order]
        cumulative_positive = np.cumsum(positive_desc, axis=1)
        cumulative_total = np.cumsum(total_desc, axis=1)
        precision = np.divide(
            cumulative_positive,
            cumulative_total,
            out=np.zeros_like(cumulative_positive),
            where=cumulative_total > 0,
        )
        ap = np.full(len(current), np.nan, dtype=np.float64)
        ap[valid] = np.sum(
            positive_desc[valid] * precision[valid], axis=1
        ) / total_positive[valid]
        output["auprc"][begin:end] = ap
    return output


def _observed_metrics(truth: np.ndarray, score: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(truth, score)),
        "auprc": float(average_precision_score(truth, score)),
        "brier": float(brier_score_loss(truth, score)),
    }


def paired_bootstrap_comparisons(
    baseline: pd.DataFrame,
    specs: Sequence[ComparisonSpec],
    *,
    patient_orders: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Compute all locked deltas using identical patient bootstrap weights."""

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    weights_by_population = {
        population: _bootstrap_weights(len(order), rng)
        for population, order in sorted(patient_orders.items())
    }
    cache: dict[
        tuple[str, str, str, str], tuple[dict[str, float], dict[str, np.ndarray], np.ndarray]
    ] = {}

    def metrics_for(group: pd.DataFrame, population: str) -> tuple[
        dict[str, float], dict[str, np.ndarray], np.ndarray
    ]:
        identity = (
            str(group["model"].iloc[0]),
            str(group["spatial"].iloc[0]),
            str(group["timing"].iloc[0]),
            population,
        )
        if identity in cache:
            return cache[identity]
        expected = tuple(patient_orders[population])
        if tuple(group["patient_id"].astype(str)) != expected:
            raise ValueError(f"patient order/coverage drifted for comparison cell {identity}")
        truth = pd.to_numeric(group["y_true"], errors="coerce").to_numpy(dtype=np.float64)
        score = pd.to_numeric(group["y_score"], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(truth).all() or not np.isin(truth, (0.0, 1.0)).all():
            raise ValueError(f"invalid binary truth for comparison cell {identity}")
        if not np.isfinite(score).all() or np.any((score < 0.0) | (score > 1.0)):
            raise ValueError(f"invalid probability for comparison cell {identity}")
        value = (
            _observed_metrics(truth.astype(np.int8), score),
            _weighted_metric_distribution(
                truth.astype(np.int8), score, weights_by_population[population]
            ),
            truth.astype(np.int8),
        )
        cache[identity] = value
        return value

    rows: list[dict[str, Any]] = []
    alpha = (1.0 - CI_LEVEL) / 2.0
    for spec in specs:
        reference = _prediction_group(baseline, spec, candidate=False)
        candidate = _prediction_group(baseline, spec, candidate=True)
        if not np.array_equal(reference["patient_id"], candidate["patient_id"]):
            raise ValueError(f"paired patients differ for {spec.comparison_id}")
        if not np.array_equal(reference["fold"], candidate["fold"]):
            raise ValueError(f"paired outer folds differ for {spec.comparison_id}")
        if not np.array_equal(reference["y_true"], candidate["y_true"]):
            raise ValueError(f"paired pCR truth differs for {spec.comparison_id}")
        ref_value, ref_bootstrap, ref_truth = metrics_for(
            reference, spec.analysis_population
        )
        cand_value, cand_bootstrap, cand_truth = metrics_for(
            candidate, spec.analysis_population
        )
        if not np.array_equal(ref_truth, cand_truth):
            raise AssertionError("paired truth cache drifted")
        for metric in METRICS:
            delta = cand_bootstrap[metric] - ref_bootstrap[metric]
            valid = np.isfinite(delta)
            valid_count = int(np.count_nonzero(valid))
            if valid_count < math.ceil(0.95 * BOOTSTRAP_REPLICATES):
                raise ValueError(
                    f"too few valid bootstrap replicates for {spec.comparison_id}/{metric}"
                )
            lower, upper = np.quantile(delta[valid], (alpha, 1.0 - alpha))
            rows.append(
                {
                    **asdict(spec),
                    "metric": metric,
                    "higher_is_better": HIGHER_IS_BETTER[metric],
                    "reference_value": ref_value[metric],
                    "candidate_value": cand_value[metric],
                    "delta_candidate_minus_reference": cand_value[metric] - ref_value[metric],
                    "ci_level": CI_LEVEL,
                    "ci_low": float(lower),
                    "ci_high": float(upper),
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "valid_bootstrap_replicates": valid_count,
                    "n_paired": int(len(reference)),
                    "positive": int(ref_truth.sum()),
                    "inference_scope": INFERENCE_SCOPE,
                }
            )
    output = pd.DataFrame(rows)
    if len(output) != len(specs) * len(METRICS):
        raise AssertionError("paired comparison output row count drifted")
    ensure_public_safe(output)
    return output.sort_values(
        ["family", "analysis_population", "timing", "candidate_model", "candidate_spatial", "metric"],
        kind="stable",
    ).reset_index(drop=True)


def _public_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    ensure_public_safe(frame)
    records: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        clean: dict[str, Any] = {}
        for key, value in record.items():
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, float) and not math.isfinite(value):
                value = None
            clean[str(key)] = value
        records.append(clean)
    return records


def _pooled(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["aggregation"].astype(str).eq("pooled_oof")].copy()


def build_public_summary(
    loaded: _LoadedInputs,
    comparisons: pd.DataFrame,
    specs: Sequence[ComparisonSpec],
    *,
    full_size: int,
    complete_size: int,
) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "comparison_contract": static_comparison_contract(),
        "resolved_comparison_count": len(specs),
        "reported_candidate_policy": "all preregistered candidates; no best-model filtering",
        "inference_scope": INFERENCE_SCOPE,
        "cohorts": {
            "full": {"analysis_population": f"full_{full_size}", "n": full_size},
            "complete_case": {
                "analysis_population": f"radiomics_complete_case_{complete_size}",
                "n": complete_size,
            },
        },
        "foundation_models": list(loaded.foundation_models),
        "current_cnn_models": ["GAP0", "LOCAL0"],
        "input_sha256": dict(sorted(loaded.input_sha256.items())),
        "paired_comparisons": _public_records(comparisons),
        "pcr_pooled_metrics": _public_records(_pooled(loaded.baseline_public)),
        "phenotype_pooled_metrics": _public_records(_pooled(loaded.phenotype_public)),
        "subtype_pooled_metrics": _public_records(_pooled(loaded.subtype_public)),
        "ftv_pooled_metrics": _public_records(_pooled(loaded.ftv_public)),
    }


def _format_number(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "NA" if not math.isfinite(number) else f"{number:.{digits}f}"


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |"
        for row in rows
    ]
    return "\n".join((header, separator, *body))


def render_markdown_summary(
    loaded: _LoadedInputs,
    comparisons: pd.DataFrame,
    *,
    full_size: int,
    complete_size: int,
) -> str:
    full_population = f"full_{full_size}"
    pcr = _pooled(loaded.baseline_public)
    pcr = pcr.loc[
        pcr["analysis_population"].astype(str).eq(full_population)
        & (
            pcr["model"].astype(str).eq("clinical_only")
            | pcr["model"].astype(str).str.endswith("_mri_only")
            | pcr["model"].astype(str).str.endswith("_mri_clinical")
        )
    ].sort_values(["model", "spatial", "timing"], kind="stable")
    pcr_table = _markdown_table(
        ("模型", "空间", "时点", "n", "AUROC", "AUPRC", "Brier", "校准斜率", "ECE10"),
        (
            (
                row.model,
                row.spatial,
                row.timing,
                int(row.n),
                _format_number(row.auroc),
                _format_number(row.auprc),
                _format_number(row.brier),
                _format_number(row.calibration_slope),
                _format_number(row.ece_10bin),
            )
            for row in pcr.itertuples(index=False)
        ),
    )
    compact = comparisons.pivot_table(
        index=[
            "comparison_id",
            "family",
            "analysis_population",
            "timing",
            "reference_model",
            "reference_spatial",
            "candidate_model",
            "candidate_spatial",
            "n_paired",
        ],
        columns="metric",
        values=["delta_candidate_minus_reference", "ci_low", "ci_high"],
        aggfunc="first",
    ).reset_index()
    paired_rows = []
    for row in compact.itertuples(index=False, name=None):
        identity = row[:9]
        values = row[9:]
        # pandas orders the three-level value columns alphabetically by first level.
        lookup = {
            tuple(column): value
            for column, value in zip(compact.columns[9:], values, strict=True)
        }
        def interval(metric: str) -> str:
            delta = lookup[("delta_candidate_minus_reference", metric)]
            lower = lookup[("ci_low", metric)]
            upper = lookup[("ci_high", metric)]
            return f"{_format_number(delta)} [{_format_number(lower)}, {_format_number(upper)}]"
        paired_rows.append(
            (
                identity[1],
                identity[2],
                identity[3],
                f"{identity[4]}@{identity[5]}",
                f"{identity[6]}@{identity[7]}",
                int(identity[8]),
                interval("auroc"),
                interval("auprc"),
                interval("brier"),
            )
        )
    paired_table = _markdown_table(
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
        paired_rows,
    )

    phenotype = _pooled(loaded.phenotype_public).sort_values(
        ["target", "model", "spatial", "timing"], kind="stable"
    )
    phenotype_table = _markdown_table(
        ("任务", "模型", "空间", "时点", "AUROC", "AUPRC", "Brier"),
        (
            (
                row.target,
                row.model,
                row.spatial,
                row.timing,
                _format_number(row.auroc),
                _format_number(row.auprc),
                _format_number(row.brier),
            )
            for row in phenotype.itertuples(index=False)
        ),
    )
    subtype = _pooled(loaded.subtype_public).sort_values(
        ["model", "spatial", "timing"], kind="stable"
    )
    subtype_table = _markdown_table(
        ("模型", "空间", "时点", "macro AUROC", "macro AUPRC", "Brier", "准确率"),
        (
            (
                row.model,
                row.spatial,
                row.timing,
                _format_number(row.macro_ovr_auroc),
                _format_number(row.macro_ovr_auprc),
                _format_number(row.multiclass_brier),
                _format_number(row.accuracy),
            )
            for row in subtype.itertuples(index=False)
        ),
    )
    ftv = _pooled(loaded.ftv_public).sort_values(
        ["model", "spatial", "task", "endpoint"], kind="stable"
    )
    ftv_table = _markdown_table(
        ("模型", "空间", "任务", "终点", "Spearman", "R²", "RMSE", "MAE"),
        (
            (
                row.model,
                row.spatial,
                row.task,
                row.endpoint,
                _format_number(row.spearman),
                _format_number(row.r2),
                _format_number(row.rmse),
                _format_number(row.mae),
            )
            for row in ftv.itertuples(index=False)
        ),
    )
    models = "、".join(loaded.foundation_models)
    return f"""# Foundation MRI 结果汇总（描述性）

正式 foundation 候选为：{models}。本页逐一保留全部预注册候选，没有按 test 指标筛选“最佳模型”。

推断边界：{INFERENCE_SCOPE}。配对区间采用同一患者非参数 bootstrap，固定 seed={BOOTSTRAP_SEED}、{BOOTSTRAP_REPLICATES} 次、percentile 95% CI。所有差值均为“候选减参照”；AUROC/AUPRC 越高越好，Brier 越低越好。区间仅用于描述不确定性，不作确认性显著性检验。

## pCR pooled OOF（完整 {full_size} 人）

{pcr_table}

## 预注册配对比较（完整 {full_size} 人与 complete-case {complete_size} 人）

{paired_table}

其中 `foundation_vs_current_cnn` 中 LOCAL 比较直接对应每个 foundation LOCAL 与 LOCAL0 的同患者差值；这些结果只用于描述 current World Model 是否可能 underuse MRI，不用于事后过滤 foundation 候选。

## HR/HER2 phenotype probes

{phenotype_table}

## HR/HER2 subtype probe

{subtype_table}

## FTV / ΔFTV decodability

{ftv_table}
"""


def _matplotlib_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - environment diagnostic
        raise RuntimeError("matplotlib is required for locked publication figures") from error
    return plt


def _series_label(model: str, spatial: str) -> str:
    return model if spatial in {"NONE", "TABULAR"} else f"{model} [{spatial}]"


def render_timing_figure(
    baseline_public: pd.DataFrame, destination: Path, *, full_population: str
) -> None:
    plt = _matplotlib_pyplot()
    pooled = _pooled(baseline_public)
    selected = pooled.loc[
        pooled["analysis_population"].astype(str).eq(full_population)
        & (
            pooled["model"].astype(str).eq("clinical_only")
            | pooled["model"].astype(str).str.endswith("_mri_only")
            | pooled["model"].astype(str).str.endswith("_mri_clinical")
        )
    ].copy()
    if selected.empty:
        raise ValueError("no full-cohort pCR rows are available for timing figure")
    selected["series"] = [
        _series_label(model, spatial)
        for model, spatial in selected.loc[:, ["model", "spatial"]].itertuples(
            index=False, name=None
        )
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    x = np.arange(len(DECISION_POINTS))
    for axis, metric, title in zip(
        axes, ("auroc", "auprc"), ("pCR AUROC", "pCR AUPRC"), strict=True
    ):
        for series, group in selected.groupby("series", sort=True):
            lookup = dict(zip(group["timing"].astype(str), group[metric], strict=True))
            if set(lookup) != set(DECISION_POINTS):
                raise ValueError(f"timing figure series is incomplete: {series}")
            axis.plot(
                x,
                [lookup[timing] for timing in DECISION_POINTS],
                marker="o",
                linewidth=1.7,
                markersize=4,
                label=series,
            )
        axis.set_title(title)
        axis.set_xticks(x, DECISION_POINTS)
        axis.set_xlabel("Prediction timing")
        axis.set_ylabel(metric.upper())
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.25, linewidth=0.7)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.suptitle("Frozen representations: pooled outer-fold pCR performance")
    fig.text(
        0.01,
        0.01,
        "All preregistered candidates shown; descriptive OOF estimates only.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 0.82, 0.95))
    fig.savefig(destination, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _reliability_points(truth: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bins = np.minimum((score * 10).astype(np.int64), 9)
    mean_score, fraction_positive = [], []
    for index in range(10):
        selected = bins == index
        if int(np.count_nonzero(selected)) >= MIN_PUBLIC_CALIBRATION_BIN_N:
            mean_score.append(float(np.mean(score[selected])))
            fraction_positive.append(float(np.mean(truth[selected])))
    return np.asarray(mean_score), np.asarray(fraction_positive)


def render_calibration_complementarity_figure(
    baseline_private: pd.DataFrame,
    comparisons: pd.DataFrame,
    destination: Path,
    *,
    full_population: str,
) -> None:
    plt = _matplotlib_pyplot()
    final_timing = "T0-T2"
    calibration = baseline_private.loc[
        baseline_private["analysis_population"].astype(str).eq(full_population)
        & baseline_private["timing"].astype(str).eq(final_timing)
        & (
            baseline_private["model"].astype(str).eq("clinical_only")
            | baseline_private["model"].astype(str).str.endswith("_mri_clinical")
        )
    ].copy()
    gains = comparisons.loc[
        comparisons["family"].astype(str).eq("clinical_gain")
        & comparisons["analysis_population"].astype(str).eq(full_population)
        & comparisons["timing"].astype(str).eq(final_timing)
        & comparisons["metric"].astype(str).isin(("auroc", "auprc"))
    ].copy()
    if calibration.empty or gains.empty:
        raise ValueError("calibration/complementarity figure inputs are incomplete")
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes[0].plot((0, 1), (0, 1), color="black", linestyle="--", linewidth=1, label="Ideal")
    for (model, spatial), group in calibration.groupby(["model", "spatial"], sort=True):
        truth = pd.to_numeric(group["y_true"]).to_numpy(dtype=np.float64)
        score = pd.to_numeric(group["y_score"]).to_numpy(dtype=np.float64)
        x, y = _reliability_points(truth, score)
        if len(x) == 0:
            raise ValueError(
                "calibration series has no public bin meeting minimum n=10"
            )
        axes[0].plot(x, y, marker="o", linewidth=1.4, markersize=3, label=_series_label(model, spatial))
    axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed pCR fraction")
    axes[0].set_title(
        f"Calibration at {final_timing} (fixed 10-bin; public bins n>=10)"
    )
    axes[0].grid(alpha=0.25, linewidth=0.7)
    axes[0].legend(fontsize=7, frameon=False, loc="best")

    gains["label"] = [
        f"{_series_label(model, spatial)} · {metric.upper()}"
        for model, spatial, metric in gains.loc[:, ["candidate_model", "candidate_spatial", "metric"]].itertuples(index=False, name=None)
    ]
    gains = gains.sort_values(["candidate_model", "candidate_spatial", "metric"], kind="stable").reset_index(drop=True)
    y_position = np.arange(len(gains))
    colors = np.asarray(
        ["#1f77b4" if metric == "auroc" else "#ff7f0e" for metric in gains["metric"]]
    )
    for metric, color in (("auroc", "#1f77b4"), ("auprc", "#ff7f0e")):
        selected = gains["metric"].astype(str).eq(metric).to_numpy()
        delta = gains.loc[selected, "delta_candidate_minus_reference"].to_numpy()
        axes[1].errorbar(
            delta,
            y_position[selected],
            xerr=np.vstack(
                (
                    delta - gains.loc[selected, "ci_low"].to_numpy(),
                    gains.loc[selected, "ci_high"].to_numpy() - delta,
                )
            ),
            fmt="none",
            ecolor=color,
            elinewidth=1.2,
            capsize=2.5,
        )
    axes[1].scatter(
        gains["delta_candidate_minus_reference"], y_position, c=colors, s=28, zorder=3
    )
    axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_yticks(y_position, gains["label"], fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Clinical+MRI minus clinical-only")
    axes[1].set_title(f"Clinical complementarity at {final_timing}\npaired 95% bootstrap CI")
    axes[1].grid(axis="x", alpha=0.25, linewidth=0.7)
    fig.suptitle("Calibration and clinical complementarity (full cohort)")
    fig.text(
        0.01,
        0.01,
        "Fixed seed 2026, 5,000 patient-bootstrap replicates; descriptive inference only.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(destination, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _check_output_targets(
    outputs: ReportingOutputs,
    overwrite: bool,
    *,
    reporting_marker: Path | None = None,
) -> None:
    paths = [Path(value) for value in asdict(outputs).values()]
    if reporting_marker is not None:
        paths.append(Path(reporting_marker))
    if len(set(paths)) != len(paths):
        raise ValueError("reporting output paths must be distinct")
    existing = [path.name for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"reporting outputs already exist: {existing}")


def _build_reporting_marker(
    lineage: Mapping[str, Any], staged: Mapping[str, Path]
) -> dict[str, Any]:
    if set(lineage) != {"baseline_v2", "probe_v3", "summarizer"}:
        raise ValueError("reporting lineage must contain baseline_v2/probe_v3/summarizer")
    stage_schema = {
        "protocol_version",
        "evaluation_lock_sha256",
        "run_receipt_sha256",
        "argv_sha256",
        "artifact_sha256",
    }
    expected_artifacts = {
        "baseline_v2": {"predictions", "selection", "metrics", "progress"},
        "probe_v3": {
            "phenotype_predictions",
            "phenotype_selection",
            "phenotype_metrics",
            "subtype_predictions",
            "subtype_selection",
            "subtype_metrics",
            "ftv_predictions",
            "ftv_selection",
            "ftv_metrics",
            "progress",
        },
    }
    digest_pattern = re.compile(r"^[0-9a-f]{64}$")
    for stage, version in (("baseline_v2", "v2"), ("probe_v3", "v3")):
        record = lineage[stage]
        if not isinstance(record, Mapping) or set(record) != stage_schema:
            raise ValueError(f"reporting lineage {stage} schema drifted")
        if record["protocol_version"] != version:
            raise ValueError(f"reporting lineage {stage} protocol drifted")
        if any(
            not digest_pattern.fullmatch(str(record[key]))
            for key in (
                "evaluation_lock_sha256",
                "run_receipt_sha256",
                "argv_sha256",
            )
        ):
            raise ValueError(f"reporting lineage {stage} digest drifted")
        artifacts = record["artifact_sha256"]
        if (
            not isinstance(artifacts, Mapping)
            or set(artifacts) != expected_artifacts[stage]
            or any(not digest_pattern.fullmatch(str(value)) for value in artifacts.values())
        ):
            raise ValueError(f"reporting lineage {stage} artifact schema drifted")
    summarizer = lineage["summarizer"]
    if not isinstance(summarizer, Mapping) or set(summarizer) != {
        "protocol_version",
        "argv_sha256",
        "code_lock_sha256",
        "finalization_lock_sha256",
    }:
        raise ValueError("reporting lineage summarizer schema drifted")
    if summarizer["protocol_version"] != "v3" or any(
        not digest_pattern.fullmatch(str(summarizer[key]))
        for key in ("argv_sha256", "code_lock_sha256", "finalization_lock_sha256")
    ):
        raise ValueError("reporting lineage summarizer identity drifted")
    roles = {
        "paired_public": "paired_csv",
        "summary_json": "summary_json",
        "results_summary_markdown": "summary_markdown",
        "timing_figure": "timing_figure",
        "calibration_figure": "calibration_figure",
    }
    return {
        "schema_version": REPORTING_PROVENANCE_SCHEMA_VERSION,
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "comparison_contract_canonical_sha256": (
            COMPARISON_CONTRACT_CANONICAL_SHA256
        ),
        "baseline_v2": dict(lineage["baseline_v2"]),
        "probe_v3": dict(lineage["probe_v3"]),
        "summarizer": dict(summarizer),
        "public_artifact_sha256": {
            role: file_sha256(staged[field]) for role, field in roles.items()
        },
    }


def _publish_no_overwrite_with_rollback(
    staged: Mapping[str, Path],
    destinations: Mapping[str, Path],
    *,
    marker_field: str | None,
) -> None:
    """Hard-link staged bytes, publish marker last, and roll back on failure."""

    order = [field for field in destinations if field != marker_field]
    if marker_field is not None:
        order.append(marker_field)
    published: list[Path] = []
    try:
        for field in order:
            destination = destinations[field]
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(staged[field], destination)
            published.append(destination)
            os.chmod(destination, 0o644)
    except Exception:
        for destination in reversed(published):
            try:
                destination.unlink()
            except FileNotFoundError:
                pass
        raise


def summarize_results(
    inputs: ReportingInputs,
    outputs: ReportingOutputs,
    *,
    full_size: int = 808,
    complete_size: int = 375,
    overwrite: bool = False,
    expected_foundation_models: Sequence[str] | None = None,
    reporting_lineage: Mapping[str, Any] | None = None,
    reporting_marker: Path | None = None,
) -> dict[str, int]:
    """Validate, compare, render, and atomically publish identifier-free outputs."""

    if (reporting_lineage is None) != (reporting_marker is None):
        raise ValueError("reporting lineage and commit marker must be supplied together")
    if reporting_marker is not None and overwrite:
        raise ValueError("formal reporting marker publication forbids overwrite")
    _check_output_targets(outputs, overwrite, reporting_marker=reporting_marker)
    loaded = _load_and_validate(inputs, full_size=full_size, complete_size=complete_size)
    if expected_foundation_models is not None:
        expected = tuple(str(value) for value in expected_foundation_models)
        if len(expected) != 2 or len(set(expected)) != 2:
            raise ValueError("formal reporting requires exactly two foundation models")
        if set(loaded.foundation_models) != set(expected):
            raise ValueError("formal foundation set is not the exact frozen two-model set")
    full_population = f"full_{full_size}"
    complete_population = f"radiomics_complete_case_{complete_size}"
    specs = build_comparison_contract(
        loaded.baseline_private,
        foundation_models=loaded.foundation_models,
        full_population=full_population,
        complete_population=complete_population,
    )
    comparisons = paired_bootstrap_comparisons(
        loaded.baseline_private,
        specs,
        patient_orders={
            full_population: loaded.full_patient_order,
            complete_population: loaded.complete_patient_order,
        },
    )
    summary = build_public_summary(
        loaded,
        comparisons,
        specs,
        full_size=full_size,
        complete_size=complete_size,
    )
    markdown = render_markdown_summary(
        loaded, comparisons, full_size=full_size, complete_size=complete_size
    )
    ensure_public_safe(comparisons)
    output_paths = [Path(value) for value in asdict(outputs).values()]
    common_parent = Path(os.path.commonpath([str(path.parent.resolve()) for path in output_paths]))
    common_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".foundation-reporting-", dir=common_parent) as temp_name:
        staging = Path(temp_name)
        staged = {
            "paired_csv": staging / "paired.csv",
            "summary_json": staging / "summary.json",
            "summary_markdown": staging / "summary.md",
            "timing_figure": staging / "timing.png",
            "calibration_figure": staging / "calibration.png",
        }
        comparisons.to_csv(staged["paired_csv"], index=False)
        staged["summary_json"].write_text(_json_text(summary), encoding="utf-8")
        staged["summary_markdown"].write_text(markdown, encoding="utf-8")
        render_timing_figure(
            loaded.baseline_public, staged["timing_figure"], full_population=full_population
        )
        render_calibration_complementarity_figure(
            loaded.baseline_private,
            comparisons,
            staged["calibration_figure"],
            full_population=full_population,
        )
        if reporting_marker is None:
            for field, destination in zip(asdict(outputs), output_paths, strict=True):
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged[field], destination)
                os.chmod(destination, 0o644)
        else:
            marker = _build_reporting_marker(reporting_lineage, staged)
            staged["reporting_marker"] = staging / "reporting_run_provenance.json"
            staged["reporting_marker"].write_text(
                _json_text(marker), encoding="utf-8"
            )
            destinations = {
                **{
                    field: destination
                    for field, destination in zip(
                        asdict(outputs), output_paths, strict=True
                    )
                },
                "reporting_marker": Path(reporting_marker),
            }
            _publish_no_overwrite_with_rollback(
                staged,
                destinations,
                marker_field="reporting_marker",
            )
    result = {
        "foundation_models": len(loaded.foundation_models),
        "resolved_comparisons": len(specs),
        "paired_metric_rows": len(comparisons),
        "public_outputs": len(output_paths),
    }
    if reporting_marker is not None:
        result["reporting_marker"] = 1
    return result


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "CI_LEVEL",
    "COMPARISON_CONTRACT_VERSION",
    "ComparisonSpec",
    "INFERENCE_SCOPE",
    "ReportingInputs",
    "ReportingOutputs",
    "build_comparison_contract",
    "paired_bootstrap_comparisons",
    "static_comparison_contract",
    "summarize_results",
]

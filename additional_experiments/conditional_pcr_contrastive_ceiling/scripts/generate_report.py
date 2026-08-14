#!/usr/bin/env python3
"""Generate the required Chinese scientific report from aggregate outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = EXPERIMENT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from conditional_ceiling.gates import evaluate_gates  # noqa: E402


BOUNDARY = (
    "This experiment intentionally uses pCR supervision and estimates a supervised "
    "representation ceiling. It is not evidence that the pCR-free World Model learned "
    "this information."
)
TIMING_LABEL = {"T0": "T0", "T0_T1": "T0–T1", "T0_T2": "T0–T2", "T0_T3": "T0–T3"}
SEEDS = (2026, 3026)
ARMS = ("B0", "B1", "B2", "B3")
SUPERVISED_ARMS = ("B1", "B2", "B3")
TIMINGS = ("T0", "T0_T1", "T0_T2", "T0_T3")
PRIMARY_TIMINGS = ("T0", "T0_T1", "T0_T2")
FOLDS = (0, 1, 2, 3, 4)
POPULATION_FAMILIES = {
    "full_808": ("C", "M", "C+M"),
    "ftv_complete_375": ("F", "C+F", "C+F+M"),
}
POPULATION_COUNTS = {
    "full_808": (808, 275, 533),
    "ftv_complete_375": (375, 110, 265),
}
SUBGROUP_COUNTS = {"HR-/HER2-": 287, "HR+/HER2-": 320, "HER2+": 201}
METRICS = ("auroc", "auprc", "brier", "calibration_slope", "ece10")
MODEL_KEYS = {"population", "seed", "arm", "timing", "model_family"}
PUBLIC_CSV_SCHEMAS = {
    "cache_integrity_audit.csv": {
        "population", "expected_files", "stat_verified_files", "sha256_verified_files",
        "mismatches", "external_files_hashed", "cache_manifest_sha256",
        "stage_b_manifest_sha256", "content_digest_aggregate_sha256",
        "content_sha256_verified",
    },
    "aggregate_metrics.csv": {
        *MODEL_KEYS, "n", "n_positive", "n_negative", *METRICS, "supplementary",
    },
    "paired_bootstrap.csv": {
        "comparison", "population", "seed", "arm", "timing", "reference_arm",
        "reference_family", "comparison_family", "delta_auroc", "ci_lower", "ci_upper",
        "reference_auroc", "comparison_auroc", "n_patients", "n_folds", "n_bootstrap",
        "n_valid_bootstrap", "confidence_level", "bootstrap_unit", "ci_method",
        "orientation", "bootstrap_seed",
    },
    "matching_audit.csv": {
        "fold", "scope", "training_patients", "total_strata", "usable_strata",
        "bidirectionally_usable_strata", "usable_patients", "dropped_anchors",
        "dropped_no_same_class_partner", "dropped_no_opposite_class",
        "unmatched_fallback_used", "test_patients_used", "pcr_negative", "pcr_positive",
        "usable_pcr_negative", "usable_pcr_positive", "natural_batch_size_3",
        "natural_batch_size_4", "max_unique_patients_per_logical_batch",
    },
    "generalization_gaps.csv": {
        *MODEL_KEYS, "supplementary",
        *(f"{split}_{metric}" for split in ("train", "validation", "test") for metric in METRICS),
        *(f"{split}_test_{metric}_gap" for split in ("train", "validation") for metric in METRICS),
    },
    "clinical_profile_probes.csv": {
        "seed", "arm", "timing", "target", "metric", "value", "n", "n_folds",
        "fold_isolated", "status",
    },
    "subgroup_refits.csv": {
        "seed", "arm", "timing", "subgroup", "eligible", "status", "n_folds",
        "n", "n_positive", "n_negative", *METRICS,
    },
    "fold_diagnostics.csv": {
        *MODEL_KEYS, "fold", "split", "selected_dimension", "selected_C",
        "validation_selection_auroc", "n", "n_positive", "n_negative", *METRICS,
        "supplementary",
    },
    "training_summary.csv": {
        "seed", "arm", "fold", "selection_status", "selected_epoch", "epochs_run",
        "selected_validation_mean_auroc", "anchor_sampling_strategy",
        "logical_patient_batch_size", "encoder_microbatch_size",
        "eligible_anchors_per_epoch", "feature_sha256", "config_sha256",
        "test_labels_used", "external_ispy1_patients_used", "world_model_claim_allowed",
    },
}


def _read_csv(path: Path, required: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    observed = set(frame.columns)
    if frame.empty or observed != required:
        raise ValueError(
            f"aggregate file schema invalid: {path.name}; "
            f"missing={sorted(required - observed)}, extra={sorted(observed - required)}"
        )
    forbidden = {"patient_id", "predicted_probability", "y_true"} & {
        str(column).strip().lower() for column in frame.columns
    }
    if forbidden:
        raise ValueError(f"report input is patient-level: {path.name}")
    return frame


def _booleans(values: pd.Series, *, name: str) -> pd.Series:
    mapping = {
        True: True, False: False, 1: True, 0: False,
        "true": True, "false": False, "1": True, "0": False,
    }
    normalized = values.map(
        lambda value: mapping.get(value, mapping.get(str(value).strip().lower()))
    )
    if normalized.isna().any():
        raise ValueError(f"{name} contains non-boolean values")
    return normalized.astype(bool)


def _exact_registry(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    expected: set[tuple[Any, ...]],
    *,
    name: str,
    integer_columns: tuple[str, ...] = (),
) -> None:
    if frame[list(columns)].isna().any().any():
        raise ValueError(f"{name} registry contains missing keys")
    normalized = frame.loc[:, columns].copy()
    for column in integer_columns:
        numeric = pd.to_numeric(normalized[column], errors="raise").to_numpy(float)
        if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError(f"{name}.{column} must contain exact integers")
        normalized[column] = numeric.astype(np.int64)
    for column in set(columns) - set(integer_columns):
        normalized[column] = normalized[column].astype(str)
    observed = set(normalized.itertuples(index=False, name=None))
    if len(frame) != len(expected) or normalized.duplicated().any() or observed != expected:
        missing = sorted(expected - observed, key=str)[:3]
        extra = sorted(observed - expected, key=str)[:3]
        raise ValueError(
            f"{name} registry is not the exact registered Cartesian product; "
            f"rows={len(frame)}, expected={len(expected)}, missing={missing}, extra={extra}"
        )


def _finite(frame: pd.DataFrame, columns: tuple[str, ...], *, name: str) -> None:
    values = frame.loc[:, columns].apply(pd.to_numeric, errors="raise").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN/infinite registered values")


def _exact_integers(frame: pd.DataFrame, columns: tuple[str, ...], *, name: str) -> None:
    values = frame.loc[:, columns].apply(pd.to_numeric, errors="raise").to_numpy(float)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{name} contains non-integral count/identity values")


def _model_registry() -> set[tuple[Any, ...]]:
    return {
        (population, seed, arm, timing, family)
        for population, families in POPULATION_FAMILIES.items()
        for seed in SEEDS
        for arm in ARMS
        for timing in TIMINGS
        for family in families
    }


def _stable_seed(*values: Any, base: int = 0) -> int:
    payload = "|".join(str(value) for value in values).encode("utf-8")
    return int(
        (int(base) + int.from_bytes(hashlib.sha256(payload).digest()[:4], "little"))
        % (2**32 - 1)
    )


def _same_value(observed: Any, expected: Any, *, path: str = "decision") -> None:
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or {str(key) for key in observed} != {str(key) for key in expected}:
            raise ValueError(f"{path} keys disagree with recomputed gate decision")
        observed_by_key = {str(key): value for key, value in observed.items()}
        for key, value in expected.items():
            _same_value(observed_by_key[str(key)], value, path=f"{path}.{key}")
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(observed, (list, tuple)) or len(observed) != len(expected):
            raise ValueError(f"{path} disagrees with recomputed gate decision")
        for index, (left, right) in enumerate(zip(observed, expected, strict=True)):
            _same_value(left, right, path=f"{path}[{index}]")
        return
    if isinstance(expected, (float, np.floating)):
        try:
            matches = math.isfinite(float(observed)) and np.isclose(
                float(observed), float(expected), rtol=1e-12, atol=1e-12
            )
        except (TypeError, ValueError):
            matches = False
        if not matches:
            raise ValueError(f"{path} disagrees with recomputed gate decision")
        return
    if observed != expected:
        raise ValueError(f"{path} disagrees with recomputed gate decision")


def _load_and_validate_inputs(root: Path) -> dict[str, Any]:
    """Load a complete, exact aggregate bundle before any report claim is made."""

    metrics_dir = root / "metrics"
    cache_audit = _read_csv(
        metrics_dir / "cache_integrity_audit.csv",
        PUBLIC_CSV_SCHEMAS["cache_integrity_audit.csv"],
    )
    if len(cache_audit) != 1:
        raise ValueError("cache integrity audit must contain exactly one aggregate row")
    _exact_integers(
        cache_audit,
        ("expected_files", "stat_verified_files", "sha256_verified_files", "mismatches", "external_files_hashed"),
        name="cache integrity audit",
    )
    cache_row = cache_audit.iloc[0]
    if (
        str(cache_row["population"]) != "full_808"
        or int(cache_row["expected_files"]) != 808
        or int(cache_row["stat_verified_files"]) != 808
        or int(cache_row["sha256_verified_files"]) != 808
        or int(cache_row["mismatches"]) != 0
        or int(cache_row["external_files_hashed"]) != 0
        or not bool(_booleans(pd.Series([cache_row["content_sha256_verified"]]), name="cache content verification").iloc[0])
        or not all(
            len(str(cache_row[column])) == 64
            and all(character in "0123456789abcdef" for character in str(cache_row[column]))
            for column in ("cache_manifest_sha256", "stage_b_manifest_sha256")
        )
        or str(cache_row["content_digest_aggregate_sha256"])
        != "f1c9965e8ae5456a899735a5462b76277ba0ec97a229dedc5faf9c380ce94c89"
    ):
        raise ValueError("cache integrity aggregate contract failed")
    aggregate = _read_csv(
        metrics_dir / "aggregate_metrics.csv",
        PUBLIC_CSV_SCHEMAS["aggregate_metrics.csv"],
    )
    model_registry = _model_registry()
    model_keys = ("population", "seed", "arm", "timing", "model_family")
    _exact_registry(
        aggregate, model_keys, model_registry, name="aggregate metrics",
        integer_columns=("seed",),
    )
    _finite(aggregate, ("n", "n_positive", "n_negative", *METRICS), name="aggregate metrics")
    _exact_integers(aggregate, ("seed", "n", "n_positive", "n_negative"), name="aggregate metrics")
    if not _booleans(aggregate["supplementary"], name="aggregate supplementary").eq(
        aggregate["timing"].astype(str).eq("T0_T3")
    ).all():
        raise ValueError("aggregate supplementary flags drifted")
    for population, counts in POPULATION_COUNTS.items():
        rows = aggregate.loc[aggregate["population"].eq(population)]
        observed = rows[["n", "n_positive", "n_negative"]].apply(pd.to_numeric, errors="raise")
        if not (observed.to_numpy(np.int64) == np.asarray(counts, dtype=np.int64)).all():
            raise ValueError(f"{population} aggregate class counts drifted")

    bootstrap = _read_csv(
        metrics_dir / "paired_bootstrap.csv",
        PUBLIC_CSV_SCHEMAS["paired_bootstrap.csv"],
    )
    bootstrap_registry = {
        (comparison, seed, arm, timing)
        for comparison in ("MRI_ceiling", "clinical_complementarity", "beyond_ftv")
        for seed in SEEDS for arm in SUPERVISED_ARMS for timing in PRIMARY_TIMINGS
    }
    _exact_registry(
        bootstrap, ("comparison", "seed", "arm", "timing"), bootstrap_registry,
        name="paired bootstrap", integer_columns=("seed",),
    )
    _finite(
        bootstrap,
        ("delta_auroc", "ci_lower", "ci_upper", "reference_auroc", "comparison_auroc", "n_patients", "n_folds", "n_bootstrap", "n_valid_bootstrap", "confidence_level", "bootstrap_seed"),
        name="paired bootstrap",
    )
    _exact_integers(
        bootstrap,
        ("seed", "n_patients", "n_folds", "n_bootstrap", "n_valid_bootstrap", "bootstrap_seed"),
        name="paired bootstrap",
    )
    aggregate_lookup = {
        key: float(row.auroc)
        for key, row in zip(
            aggregate[list(model_keys)].itertuples(index=False, name=None),
            aggregate.itertuples(index=False), strict=True,
        )
    }
    pair_specs = {
        "MRI_ceiling": ("full_808", "B0", "M", "M", 808),
        "clinical_complementarity": ("full_808", None, "C", "C+M", 808),
        "beyond_ftv": ("ftv_complete_375", None, "C+F", "C+F+M", 375),
    }
    for row in bootstrap.itertuples(index=False):
        population, fixed_reference_arm, reference_family, comparison_family, n_patients = pair_specs[str(row.comparison)]
        reference_arm = fixed_reference_arm or str(row.arm)
        if (
            str(row.population) != population
            or str(row.reference_arm) != reference_arm
            or str(row.reference_family) != reference_family
            or str(row.comparison_family) != comparison_family
            or int(row.n_patients) != n_patients
            or int(row.n_folds) != 5
            or int(row.n_bootstrap) != 5000
            or int(row.n_valid_bootstrap) != 5000
            or not np.isclose(float(row.confidence_level), 0.95, rtol=0.0, atol=1e-15)
            or str(row.bootstrap_unit) != "patient_within_outer_fold"
            or str(row.ci_method) != "percentile"
            or str(row.orientation) != "comparison - reference"
        ):
            raise ValueError(f"paired bootstrap endpoint semantics drifted for {row.comparison}")
        reference = aggregate_lookup[(population, int(row.seed), reference_arm, str(row.timing), reference_family)]
        comparison = aggregate_lookup[(population, int(row.seed), str(row.arm), str(row.timing), comparison_family)]
        seed_tag = {
            "MRI_ceiling": "mri",
            "clinical_complementarity": "cm",
            "beyond_ftv": "cfm",
        }[str(row.comparison)]
        expected_bootstrap_seed = _stable_seed(
            int(row.seed), str(row.arm), str(row.timing), seed_tag, base=260_812
        )
        if not (
            np.isclose(float(row.reference_auroc), reference, rtol=1e-12, atol=1e-12)
            and np.isclose(float(row.comparison_auroc), comparison, rtol=1e-12, atol=1e-12)
            and np.isclose(float(row.delta_auroc), comparison - reference, rtol=1e-12, atol=1e-12)
            and float(row.ci_lower) <= float(row.ci_upper)
            and int(row.bootstrap_seed) == expected_bootstrap_seed
        ):
            raise ValueError("paired bootstrap values do not match their aggregate endpoints")

    matching = _read_csv(
        metrics_dir / "matching_audit.csv",
        PUBLIC_CSV_SCHEMAS["matching_audit.csv"],
    )
    _exact_registry(matching, ("fold",), {(fold,) for fold in FOLDS}, name="matching audit", integer_columns=("fold",))
    _exact_integers(
        matching,
        (
            "fold", "training_patients", "total_strata", "usable_strata",
            "bidirectionally_usable_strata", "usable_patients", "dropped_anchors",
            "dropped_no_same_class_partner", "dropped_no_opposite_class", "pcr_negative",
            "pcr_positive", "usable_pcr_negative", "usable_pcr_positive",
            "natural_batch_size_3", "natural_batch_size_4",
            "max_unique_patients_per_logical_batch",
        ),
        name="matching audit",
    )
    locked_matching = {
        0: (525, 28, 26, 24, 506, 19, 347, 178, 347, 159),
        1: (525, 28, 26, 22, 513, 12, 346, 179, 344, 169),
        2: (525, 28, 27, 24, 520, 5, 347, 178, 345, 175),
        3: (526, 28, 27, 24, 515, 11, 347, 179, 346, 169),
        4: (526, 28, 27, 26, 524, 2, 347, 179, 345, 179),
    }
    matching_columns = (
        "training_patients", "total_strata", "usable_strata", "bidirectionally_usable_strata",
        "usable_patients", "dropped_anchors", "pcr_negative", "pcr_positive",
        "usable_pcr_negative", "usable_pcr_positive",
    )
    for row in matching.itertuples(index=False):
        if tuple(int(getattr(row, column)) for column in matching_columns) != locked_matching[int(row.fold)]:
            raise ValueError("exact-arm matching aggregate disagrees with the locked audit")
        expected_drops = {0: (2, 17), 1: (6, 6), 2: (5, 0), 3: (3, 8), 4: (2, 1)}[int(row.fold)]
        if (
            str(row.scope) != "outer_train_only"
            or (int(row.dropped_no_same_class_partner), int(row.dropped_no_opposite_class)) != expected_drops
        ):
            raise ValueError("matching scope/drop-reason audit disagrees with frozen strata")
        expected_batches = {0: (2, 504), 1: (0, 513), 2: (0, 520), 3: (2, 513), 4: (0, 524)}[int(row.fold)]
        observed_batches = (int(row.natural_batch_size_3), int(row.natural_batch_size_4))
        if (
            observed_batches != expected_batches
            or sum(observed_batches) != int(row.usable_patients)
            or int(row.max_unique_patients_per_logical_batch) != 4
        ):
            raise ValueError("natural logical batch-size audit disagrees with the locked sampler")
    if _booleans(matching["unmatched_fallback_used"], name="matching fallback").any() or _booleans(
        matching["test_patients_used"], name="matching test use"
    ).any():
        raise ValueError("matching audit records fallback or test-patient use")

    gaps = _read_csv(
        metrics_dir / "generalization_gaps.csv",
        PUBLIC_CSV_SCHEMAS["generalization_gaps.csv"],
    )
    _exact_registry(gaps, model_keys, model_registry, name="generalization gaps", integer_columns=("seed",))
    gap_numeric = tuple(
        column for column in gaps.columns
        if any(column.startswith(prefix) for prefix in ("train_", "validation_", "test_"))
        and column != "supplementary"
    )
    _finite(gaps, gap_numeric, name="generalization gaps")
    if not _booleans(gaps["supplementary"], name="gap supplementary").eq(
        gaps["timing"].astype(str).eq("T0_T3")
    ).all():
        raise ValueError("generalization-gap supplementary flags drifted")
    aggregate_by_key = aggregate.set_index(list(model_keys))
    for row in gaps.itertuples(index=False):
        key = (str(row.population), int(row.seed), str(row.arm), str(row.timing), str(row.model_family))
        aggregate_row = aggregate_by_key.loc[key]
        for metric in METRICS:
            train = float(getattr(row, f"train_{metric}"))
            validation = float(getattr(row, f"validation_{metric}"))
            test = float(getattr(row, f"test_{metric}"))
            if not (
                np.isclose(test, float(aggregate_row[metric]), rtol=1e-12, atol=1e-12)
                and np.isclose(float(getattr(row, f"train_test_{metric}_gap")), train - test, rtol=1e-12, atol=1e-12)
                and np.isclose(float(getattr(row, f"validation_test_{metric}_gap")), validation - test, rtol=1e-12, atol=1e-12)
            ):
                raise ValueError("generalization gap or held-out metric is internally inconsistent")

    probes = _read_csv(
        metrics_dir / "clinical_profile_probes.csv",
        PUBLIC_CSV_SCHEMAS["clinical_profile_probes.csv"],
    )
    registered_probe = {
        (seed, arm, timing, target)
        for seed in SEEDS for arm in ("B0", "B2", "B3")
        for timing in PRIMARY_TIMINGS for target in ("HR", "HER2", "subtype")
    }
    main_probes = probes.loc[probes["target"].astype(str).ne("treatment")].copy()
    _exact_registry(
        main_probes, ("seed", "arm", "timing", "target"), registered_probe,
        name="clinical profile probes", integer_columns=("seed",),
    )
    treatment = probes.loc[probes["target"].astype(str).eq("treatment")]
    if len(probes) != 55 or len(treatment) != 1:
        raise ValueError("clinical profile probe registry must contain 54 probes and one treatment sentinel")
    sentinel = treatment.iloc[0]
    if (
        int(sentinel["seed"]) != -1 or str(sentinel["arm"]) != "ALL"
        or str(sentinel["timing"]) != "ALL" or str(sentinel["metric"]) != "not_run"
        or int(sentinel["n"]) != 808 or int(sentinel["n_folds"]) != 0
        or pd.notna(sentinel["value"])
        or str(sentinel.get("status", "")) != "unsuitable_exact_13_arm_target_due_to_sparse_fold_classes"
        or not bool(_booleans(pd.Series([sentinel["fold_isolated"]]), name="treatment isolation").iloc[0])
    ):
        raise ValueError("treatment-unsuitability sentinel drifted")
    _finite(main_probes, ("value", "n", "n_folds"), name="clinical profile probes")
    _exact_integers(main_probes, ("seed", "n", "n_folds"), name="clinical profile probes")
    expected_metric = main_probes["target"].map({"HR": "auroc", "HER2": "auroc", "subtype": "macro_ovr_auroc"})
    if (
        not main_probes["metric"].astype(str).eq(expected_metric).all()
        or not pd.to_numeric(main_probes["n"], errors="raise").eq(808).all()
        or not pd.to_numeric(main_probes["n_folds"], errors="raise").eq(5).all()
        or not _booleans(main_probes["fold_isolated"], name="profile isolation").all()
    ):
        raise ValueError("profile probes do not cover all 808 patients/all five folds")

    subgroups = _read_csv(
        metrics_dir / "subgroup_refits.csv",
        PUBLIC_CSV_SCHEMAS["subgroup_refits.csv"],
    )
    subgroup_registry = {
        (seed, arm, timing, subgroup)
        for seed in SEEDS for arm in ARMS for timing in PRIMARY_TIMINGS
        for subgroup in SUBGROUP_COUNTS
    }
    _exact_registry(
        subgroups, ("seed", "arm", "timing", "subgroup"), subgroup_registry,
        name="subgroup refits", integer_columns=("seed",),
    )
    _finite(subgroups, ("n", "n_folds", *METRICS), name="subgroup refits")
    _exact_integers(
        subgroups, ("seed", "n", "n_positive", "n_negative", "n_folds"),
        name="subgroup refits",
    )
    if (
        not _booleans(subgroups["eligible"], name="subgroup eligibility").all()
        or not subgroups["status"].astype(str).eq("ok").all()
        or not pd.to_numeric(subgroups["n_folds"], errors="raise").eq(5).all()
        or any(int(row.n) != SUBGROUP_COUNTS[str(row.subgroup)] for row in subgroups.itertuples(index=False))
    ):
        raise ValueError("registered subgroup refits are incomplete or ineligible")

    diagnostics = _read_csv(
        metrics_dir / "fold_diagnostics.csv",
        PUBLIC_CSV_SCHEMAS["fold_diagnostics.csv"],
    )
    diagnostic_registry = {
        (*key, fold, split)
        for key in model_registry for fold in FOLDS for split in ("train", "validation", "test")
    }
    _exact_registry(
        diagnostics,
        ("population", "seed", "arm", "timing", "model_family", "fold", "split"),
        diagnostic_registry, name="fold diagnostics", integer_columns=("seed", "fold"),
    )
    _finite(diagnostics, ("n", "n_positive", "n_negative", *METRICS), name="fold diagnostics")
    _exact_integers(
        diagnostics, ("seed", "fold", "n", "n_positive", "n_negative"),
        name="fold diagnostics",
    )
    if not _booleans(diagnostics["supplementary"], name="diagnostic supplementary").eq(
        diagnostics["timing"].astype(str).eq("T0_T3")
    ).all():
        raise ValueError("fold-diagnostic supplementary flags drifted")
    diagnostic_compact = diagnostics["model_family"].isin(("M", "C+M", "C+F+M"))
    diagnostic_dimension = pd.to_numeric(diagnostics["selected_dimension"], errors="coerce")
    diagnostic_c = pd.to_numeric(diagnostics["selected_C"], errors="raise")
    diagnostic_validation = pd.to_numeric(diagnostics["validation_selection_auroc"], errors="raise")
    if (
        not diagnostic_dimension.loc[diagnostic_compact].isin((8, 16, 32, 64)).all()
        or diagnostic_dimension.loc[~diagnostic_compact].notna().any()
        or not np.isfinite(diagnostic_c.to_numpy(float)).all()
        or not np.isfinite(diagnostic_validation.to_numpy(float)).all()
    ):
        raise ValueError("fold diagnostic selection metadata drifted")

    training = _read_csv(
        metrics_dir / "training_summary.csv",
        PUBLIC_CSV_SCHEMAS["training_summary.csv"],
    )
    training_registry = {(seed, arm, fold) for seed in SEEDS for arm in SUPERVISED_ARMS for fold in FOLDS}
    _exact_registry(
        training, ("seed", "arm", "fold"), training_registry,
        name="training summary", integer_columns=("seed", "fold"),
    )
    _finite(training, ("selected_epoch", "epochs_run", "selected_validation_mean_auroc"), name="training summary")
    _exact_integers(training, ("seed", "fold", "selected_epoch", "epochs_run"), name="training summary")
    if (
        not training["selection_status"].astype(str).eq("SELECTED_VALIDATION_ONLY").all()
        or _booleans(training["test_labels_used"], name="training test-label use").any()
        or pd.to_numeric(training["external_ispy1_patients_used"], errors="raise").ne(0).any()
        or _booleans(training["world_model_claim_allowed"], name="training claim boundary").any()
    ):
        raise ValueError("training summary isolation contract failed")
    hash_columns = ("feature_sha256", "config_sha256")
    for column in hash_columns:
        values = training[column].astype(str)
        if not values.str.fullmatch(r"[0-9a-f]{64}").all():
            raise ValueError(f"training summary {column} binding is invalid")
    if training["config_sha256"].astype(str).nunique() != 1 or training["feature_sha256"].astype(str).nunique() != 30:
        raise ValueError("training summary artifact bindings are incomplete")
    usable_by_fold = {
        int(row.fold): int(row.usable_patients) for row in matching.itertuples(index=False)
    }
    b1 = training["arm"].astype(str).eq("B1")
    adapted = ~b1
    b3 = training["arm"].astype(str).eq("B3")
    expected_eligible = training.loc[adapted, "fold"].map(usable_by_fold).astype(float)
    if (
        training.loc[b1, ["anchor_sampling_strategy", "logical_patient_batch_size", "encoder_microbatch_size", "eligible_anchors_per_epoch"]].notna().any().any()
        or not training.loc[adapted, "anchor_sampling_strategy"].astype(str).eq(
            "all_eligible_anchors_exactly_once_per_epoch"
        ).all()
        or not pd.to_numeric(training.loc[adapted, "logical_patient_batch_size"], errors="raise").eq(4).all()
        or not pd.to_numeric(training.loc[adapted, "eligible_anchors_per_epoch"], errors="raise").reset_index(drop=True).eq(
            expected_eligible.reset_index(drop=True)
        ).all()
        or training.loc[~b3, "encoder_microbatch_size"].notna().any()
        or not pd.to_numeric(training.loc[b3, "encoder_microbatch_size"], errors="raise").eq(1).all()
    ):
        raise ValueError("training summary sampling/microbatch contract failed")

    decision = json.loads((metrics_dir / "decision_summary.json").read_text(encoding="utf-8"))
    expected_decision_keys = {
        "schema_version", "reporting_boundary", "primary_seeds",
        "folds_are_biological_replicates", "headline_bootstrap_draws",
        "private_predictions_sha256", "decision",
    }
    prediction_sha = str(decision.get("private_predictions_sha256", ""))
    if (
        set(decision) != expected_decision_keys
        or type(decision.get("schema_version")) is not int
        or decision.get("schema_version") != 1
        or decision.get("reporting_boundary") != BOUNDARY
        or decision.get("primary_seeds") != list(SEEDS)
        or decision.get("folds_are_biological_replicates") is not False
        or type(decision.get("headline_bootstrap_draws")) is not int
        or decision.get("headline_bootstrap_draws") != 5000
        or len(prediction_sha) != 64
        or any(character not in "0123456789abcdef" for character in prediction_sha)
    ):
        raise ValueError("decision summary wrapper contract drifted")
    recomputed = evaluate_gates(
        bootstrap.loc[bootstrap["comparison"].eq("MRI_ceiling")],
        bootstrap.loc[bootstrap["comparison"].eq("clinical_complementarity")],
        bootstrap.loc[bootstrap["comparison"].eq("beyond_ftv")],
    ).as_dict()
    _same_value(decision.get("decision"), recomputed)

    return {
        "metrics": aggregate,
        "cache_audit": cache_audit,
        "bootstrap": bootstrap,
        "matching": matching,
        "gaps": gaps,
        "probes": probes,
        "subgroups": subgroups,
        "diagnostics": diagnostics,
        "training": training,
        "decision": decision,
    }


def _f(value: Any, digits: int = 3, signed: bool = False) -> str:
    number = float(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:+.{digits}f}" if signed else f"{number:.{digits}f}"


def _mean_cells(metrics: pd.DataFrame, family: str, arms: tuple[str, ...], primary: bool = True) -> pd.DataFrame:
    frame = metrics.loc[
        metrics["population"].eq("full_808")
        & metrics["model_family"].eq(family)
        & metrics["arm"].isin(arms)
    ].copy()
    if primary:
        frame = frame.loc[frame["timing"].isin(("T0", "T0_T1", "T0_T2"))]
    return frame.groupby(["arm", "timing"], as_index=False).agg(
        mean_auroc=("auroc", "mean"),
        mean_auprc=("auprc", "mean"),
        mean_brier=("brier", "mean"),
        mean_calibration_slope=("calibration_slope", "mean"),
        mean_ece10=("ece10", "mean"),
        seeds=("seed", "nunique"),
    )


def _delta_table(frame: pd.DataFrame, comparison: str) -> pd.DataFrame:
    selected = frame.loc[frame["comparison"].eq(comparison)].copy()
    return selected.groupby(["arm", "timing"], as_index=False).agg(
        mean_delta=("delta_auroc", "mean"),
        positive_seeds=("delta_auroc", lambda values: int((values > 0).sum())),
        ci_positive_seeds=("ci_lower", lambda values: int((values > 0).sum())),
        min_ci_lower=("ci_lower", "min"),
        max_ci_upper=("ci_upper", "max"),
    )


def _markdown_table(frame: pd.DataFrame, columns: list[tuple[str, str]], *, signed: set[str] = set()) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    lines = [header, separator]
    for row in frame.itertuples(index=False):
        values: list[str] = []
        record = row._asdict()
        for key, _ in columns:
            value = record[key]
            if key == "timing":
                values.append(TIMING_LABEL.get(str(value), str(value)))
            elif isinstance(value, (float, np.floating)):
                values.append(_f(value, signed=key in signed))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def render_report(root: Path = EXPERIMENT_ROOT) -> str:
    """Return the fully validated deterministic report body without writing."""

    root = root.resolve()
    bundle = _load_and_validate_inputs(root)
    metrics = bundle["metrics"]
    bootstrap = bundle["bootstrap"]
    matching = bundle["matching"]
    cache_audit = bundle["cache_audit"]
    gaps = bundle["gaps"]
    probes = bundle["probes"]
    subgroups = bundle["subgroups"]
    decision = bundle["decision"]
    gate = decision["decision"]

    mri = _mean_cells(metrics, "M", ("B0", "B1", "B2", "B3"))
    mri_t3 = _mean_cells(
        metrics, "M", ("B0", "B1", "B2", "B3"), primary=False
    ).loc[lambda frame: frame["timing"].eq("T0_T3")]
    best = mri.loc[mri["arm"].isin(("B1", "B2", "B3"))].sort_values(
        ["mean_auroc", "arm", "timing"], ascending=[False, True, True]
    ).iloc[0]
    # Goal B defines the named ceiling gap literally as B3 - B0.  B1/B2 may
    # still be the best supervised diagnostic cell, but can never redefine it.
    b3_best = mri.loc[mri["arm"].eq("B3")].sort_values(
        ["mean_auroc", "timing"], ascending=[False, True]
    ).iloc[0]
    b0_at_b3 = mri.loc[
        mri["arm"].eq("B0") & mri["timing"].eq(b3_best["timing"])
    ].iloc[0]
    ceiling_gap = float(b3_best["mean_auroc"] - b0_at_b3["mean_auroc"])

    mri_delta = _delta_table(bootstrap, "MRI_ceiling")
    cm_delta = _delta_table(bootstrap, "clinical_complementarity")
    ftv_delta = _delta_table(bootstrap, "beyond_ftv")
    b1_best = mri_delta.loc[mri_delta["arm"].eq("B1")].sort_values("mean_delta", ascending=False).iloc[0]
    adapted_best = mri_delta.loc[mri_delta["arm"].isin(("B2", "B3"))].sort_values("mean_delta", ascending=False).iloc[0]

    primary_probe = probes.loc[
        probes["arm"].isin(("B0", "B2", "B3"))
        & probes["timing"].isin(("T0", "T0_T1", "T0_T2"))
        & probes["metric"].isin(("auroc", "macro_ovr_auroc"))
    ].copy()
    probe_summary = primary_probe.groupby(["target", "arm"], as_index=False)["value"].mean()
    subgroup_summary = subgroups.loc[subgroups["eligible"].astype(str).str.lower().isin(("true", "1"))].groupby(
        ["subgroup", "arm", "timing"], as_index=False
    )["auroc"].mean()
    best_subgroup = subgroup_summary.sort_values(["subgroup", "auroc"], ascending=[True, False]).groupby("subgroup", as_index=False).head(1)
    gap_summary = gaps.loc[
        gaps["population"].eq("full_808") & gaps["model_family"].eq("M")
        & gaps["timing"].isin(("T0", "T0_T1", "T0_T2"))
    ].groupby("arm", as_index=False).agg(
        train_auroc=("train_auroc", "mean"), validation_auroc=("validation_auroc", "mean"),
        test_auroc=("test_auroc", "mean"), train_test_gap=("train_test_auroc_gap", "mean")
    )

    gate_lines = []
    for key, label in (("gate_a", "Gate A"), ("gate_b", "Gate B"), ("gate_c", "Gate C")):
        item = gate[key]
        gate_lines.append(
            f"- {label}: **{'PASS' if item['passed'] else 'FAIL'}**；最佳 {item.get('best_arm') or 'NA'} / "
            f"{TIMING_LABEL.get(str(item.get('best_timing')), str(item.get('best_timing') or 'NA'))}；"
            f"均值 ΔAUROC {_f(item.get('mean_delta_auroc', math.nan), signed=True)}；"
            f"代码 `{item.get('pass_code') or 'NO_PASS_CODE'}`。"
        )

    b1_sufficient = float(b1_best["mean_delta"]) >= 0.08 and float(adapted_best["mean_delta"]) - float(b1_best["mean_delta"]) < 0.03
    adaptation_required = float(adapted_best["mean_delta"]) > float(b1_best["mean_delta"]) + 0.03
    justification = gate["gate_a"]["passed"] or gate["gate_b"]["passed"] or gate["gate_c"]["passed"]

    report = f"""# 条件 pCR 对比学习上限实验：最终报告

> **关键报告边界**
> {BOUNDARY}

本实验是刻意使用 pCR 标签的 oracle / ceiling / discovery 分析，不是可部署模型，也不是无结局监督 World Model 的证据。若出现正结果，后续问题只能表述为：能否用不含 pCR 的目标复现这个有监督上限结构？

## 前置证据

本实验在启动前读取并冻结了六项既有证据：MRI–clinical complementarity audit 显示当前 MRI-only pCR 较弱；compact fusion audit 显示直接 192-D/纵向拼接过拟合，而 train-only PCA 可降低但未消除泛化差距；LOCAL multi-seed confirmation 锁定 LOCAL3 为起点；classical DCE phenotype complementarity 显示非 FTV 的传统 DCE 特征存在部分互补性；foundation MRI baselines 与 DINOv3 post-hoc 均未稳定解决 pCR 互补性。因此此处检验的是“信号存在但无结局监督目标没有组织出来”的上限假设，不重复选择基础架构。

冻结 full_808 的 808 个 I-SPY2 C1B cache 文件均完成 size、mtime 与逐文件 SHA-256 内容复核，mismatch=0；排序后的内容摘要聚合 SHA-256 为 `{cache_audit.iloc[0]['content_digest_aggregate_sha256']}`。139 个外部患者文件不在本实验输入范围，未被哈希或使用。公开审计仅含汇总计数与摘要，不含患者标识或路径。

## 执行摘要

在两个预注册训练种子（2026、3026）和同一组冻结外层折上，最佳 MRI-only 有监督单元为 **{best['arm']} / {TIMING_LABEL[str(best['timing'])]}**，跨种子平均 AUROC 为 **{_f(best['mean_auroc'])}**（AUPRC {_f(best['mean_auprc'])}）；按预注册定义，B3 最佳时间点 {TIMING_LABEL[str(b3_best['timing'])]} 相对同时间 B0 的 ceiling gap（B3−B0）为 **{_f(ceiling_gap, signed=True)} AUROC**。最终解释类别为 **{gate['interpretation_class']} — `{gate['interpretation_code']}`**。

{chr(10).join(gate_lines)}

## 预注册设计与隔离边界

- MRI 输入固定为 C1B-H / DCE7；从每个 seed×fold 对应的 confirmed LOCAL3 检查点启动。
- B0 不接触 pCR；B1 仅训练 192→128→64 投影；B2 训练末级编码器、LOCAL response projection、小投影及训练期线性 pCR 头；B3 以低学习率微调整个编码器，且仅是诊断上限。
- B2/B3 的损失严格为 `L_condSupCon + 0.25 L_pCR`。HR、HER2 与精确 assigned arm 只用于外层训练集内的配对；它们从不进入 MRI 前向或 pCR 分类器。
- B2/B3 每轮恰好访问每个 eligible anchor 一次，并逐轮确定性重排；每个 exact-stratum 逻辑 batch 含 3–4 个不重复患者（4 是上限，2-v-1 层无法合法补足第四人）。编码器 microbatch 只用于降低显存，不改变 anchor 暴露次数。
- BCE 在每个采样逻辑 batch 的全部行上计算（anchor、同类 support 与异类 support）；因此 support 患者可在同一 epoch 被重复抽到。处于不可用 exact strata 的患者不进入 B2/B3 优化。临床字段只定义配对层，绝不进入 MRI 表征或 pCR logits。
- `full_808` 只用于 M/C/C+M；`ftv_complete_375` 只用于 F/C+F/C+F+M。PCA、临床编码和 L2 logistic 均在每个外层训练折单独拟合，超参数只看 validation。
- 折不是独立生物学重复；所有 headline CI 都按患者在冻结外层折内成对重采样 5,000 次。

### 精确 HR×HER2×assigned-arm 匹配审计

{_markdown_table(matching.sort_values('fold'), [('fold','折'),('training_patients','训练 n'),('total_strata','全部层'),('usable_strata','任一方向可用层'),('bidirectionally_usable_strata','双向可用层'),('usable_patients','可用 anchors'),('dropped_anchors','丢弃 anchors'),('natural_batch_size_3','natural batch=3'),('natural_batch_size_4','natural batch=4'),('max_unique_patients_per_logical_batch','配置上限'),('pcr_negative','pCR−'),('pcr_positive','pCR+')])}

没有使用 test 患者配对，也没有 unmatched negative fallback。`usable_strata` 表示至少有一类 anchor 可用；另列 `bidirectionally_usable_strata` 避免把 1-v-1 层误报为双向可用。

## MRI-only 主结果

{_markdown_table(mri.sort_values(['timing','arm']), [('arm','模型臂'),('timing','时间'),('mean_auroc','AUROC'),('mean_auprc','AUPRC'),('mean_brier','Brier'),('mean_calibration_slope','校准斜率'),('mean_ece10','ECE10'),('seeds','种子数')])}

### 相对 B0 的有监督 MRI ceiling

{_markdown_table(mri_delta.sort_values(['timing','arm']), [('arm','模型臂'),('timing','时间'),('mean_delta','平均 ΔAUROC'),('positive_seeds','正向种子'),('ci_positive_seeds','CI>0 种子'),('min_ci_lower','最低 CI 下界'),('max_ci_upper','最高 CI 上界')], signed={'mean_delta','min_ci_lower','max_ci_upper'})}

## 临床条件价值与 FTV 之外价值

### C+M − C（full_808）

{_markdown_table(cm_delta.sort_values(['timing','arm']), [('arm','模型臂'),('timing','时间'),('mean_delta','平均 ΔAUROC'),('positive_seeds','正向种子'),('ci_positive_seeds','CI>0 种子'),('min_ci_lower','最低 CI 下界'),('max_ci_upper','最高 CI 上界')], signed={'mean_delta','min_ci_lower','max_ci_upper'})}

### C+F+M − (C+F)（ftv_complete_375）

{_markdown_table(ftv_delta.sort_values(['timing','arm']), [('arm','模型臂'),('timing','时间'),('mean_delta','平均 ΔAUROC'),('positive_seeds','正向种子'),('ci_positive_seeds','CI>0 种子'),('min_ci_lower','最低 CI 下界'),('max_ci_upper','最高 CI 上界')], signed={'mean_delta','min_ci_lower','max_ci_upper'})}

### T0–T3 补充结果（不进入 Gates A–C）

{_markdown_table(mri_t3.sort_values('arm'), [('arm','模型臂'),('timing','时间'),('mean_auroc','AUROC'),('mean_auprc','AUPRC'),('mean_brier','Brier'),('mean_calibration_slope','校准斜率'),('mean_ece10','ECE10')])}

## 临床捷径与亚组审计

HR/HER2/subtype 的 fold-isolated 线性 probe 汇总如下；精确 13-arm treatment target 因部分折类别稀疏，被预先标为不适合而未静默合并治疗组。

{_markdown_table(probe_summary.sort_values(['target','arm']), [('target','目标'),('arm','MRI 臂'),('value','平均 probe AUROC')])}

每个临床亚组都重新拟合 MRI-only 分类器，而不是仅对总体概率做事后分层：

{_markdown_table(best_subgroup.sort_values('subgroup'), [('subgroup','亚组'),('arm','最佳臂'),('timing','最佳时间'),('auroc','平均 AUROC')])}

## 泛化差距

{_markdown_table(gap_summary.sort_values('arm'), [('arm','模型臂'),('train_auroc','Train AUROC'),('validation_auroc','Validation AUROC'),('test_auroc','Test/OOF AUROC'),('train_test_gap','Train−test')], signed={'train_test_gap'})}

## 九个核心问题的直接回答

1. **直接监督下 MRI-only 能到多高？** 最佳主时间单元为 {best['arm']} / {TIMING_LABEL[str(best['timing'])]}，平均 AUROC {_f(best['mean_auroc'])}、AUPRC {_f(best['mean_auprc'])}。
2. **是否揭示 HR/HER2 之外的信息？** {'是' if gate['gate_b']['passed'] else '未达到预注册的肯定标准'}；Gate B {'通过' if gate['gate_b']['passed'] else '未通过'}，其最佳平均 C+M−C 为 {_f(gate['gate_b']['mean_delta_auroc'], signed=True)}。
3. **是否揭示 FTV 之外的信息？** {'是' if gate['gate_c']['passed'] else '未达到预注册的肯定标准'}；Gate C {'通过' if gate['gate_c']['passed'] else '未通过'}，最佳平均 C+F+M−(C+F) 为 {_f(gate['gate_c']['mean_delta_auroc'], signed=True)}。
4. **信号最强在何时？** 在预注册主时间中为 {TIMING_LABEL[str(best['timing'])]}。T0–T3 仅列作补充，不参与成功门槛。
5. **冻结状态的非线性组织是否足够？** {'按预定义“B1 强、适配增益小”标准，足够。' if b1_sufficient else '不足以单独解释全部上限；B1 的最佳平均增益为 ' + _f(b1_best['mean_delta'], signed=True) + '。'}
6. **是否需要编码器适配？** {'需要；B2/B3 相对 B1 的额外最佳增益超过 0.03。' if adaptation_required else '未显示出超过 0.03 AUROC 的明确额外必要性。'}
7. **增益是否经受 clinical-matched 对比训练？** {'是；Gate A/B 的相应条件达到预注册门槛。' if gate['gate_a']['passed'] and gate['gate_b']['passed'] else '没有同时达到 Gate A 与 Gate B，因此不能作强肯定。'}
8. **上限是否足以支持继续研究 pCR-free 表征目标？** {'是；至少一个预注册 ceiling/complementarity 门槛通过，值得把该结构作为无结局监督目标的待复现上限。' if justification else '当前结果不足以用性能门槛支持大规模追加，但负结果仍限定了可实现上限。'}
9. **当前 World Model 与有监督上限相差多少？** 按字面 B3−B0 定义，在 B3 最佳主时间 {TIMING_LABEL[str(b3_best['timing'])]} 比较约为 {_f(ceiling_gap, signed=True)} AUROC（B3 {_f(b3_best['mean_auroc'])} 对 B0 {_f(b0_at_b3['mean_auroc'])}）。

## 结论与限制

最终分类为 **{gate['interpretation_class']} / `{gate['interpretation_code']}`**。这是在固定输入、固定 LOCAL 架构、精确临床/治疗匹配及低容量读出下的经验上限，不是 MRI 的信息论上限，也不是治疗效应估计。两个训练种子共享患者与外层折；种子一致性是优化稳定性证据，不能当作独立人群复现。校准斜率与 ECE 在样本较小的 FTV/亚组分析中不稳定，应与 Brier 和区分度共同阅读。

可选的治疗臂内 AUROC / arm-balanced weighting 未作为主分析运行：精确 13-arm 在折内过于稀疏，任意合并都会改变已冻结的 matching estimand；本实验也不作因果治疗效应解释。

## 可复核产物

- 汇总指标：`metrics/aggregate_metrics.csv`
- 5,000 次成对患者 bootstrap：`metrics/paired_bootstrap.csv`
- 匹配审计：`metrics/matching_audit.csv`
- 训练/validation-only 选择审计：`metrics/training_summary.csv`
- 临床 probe、亚组重拟合与泛化差距：`metrics/clinical_profile_probes.csv`、`metrics/subgroup_refits.csv`、`metrics/generalization_gaps.csv`
- 判定：`metrics/decision_summary.json`
- 七张规定图：`figures/01_*.png` 至 `figures/07_*.png`
- 患者级预测、特征与检查点仅存于 gitignored 私有目录，未进入 Git。
"""
    return report


def generate_report(root: Path = EXPERIMENT_ROOT) -> Path:
    root = root.resolve()
    report = render_report(root)
    output = root / "reports" / "final_report.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=EXPERIMENT_ROOT)
    arguments = parser.parse_args()
    try:
        output = generate_report(arguments.experiment_root)
    except (FileNotFoundError, ValueError, KeyError, IndexError) as error:
        print(f"REPORT_GENERATION_FAILED: {error}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

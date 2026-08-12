#!/usr/bin/env python3
"""Generate the required Chinese report from completed aggregate artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "metrics"
DECISION = METRICS / "decision.json"
REPORT = ROOT / "reports/final_report.md"
SEEDS = (2026, 3026)
ARMS = ("S0", "S1", "S2", "S2_L10")
VISITS = ("T0", "T1", "T2", "T3")
DELTA_ENDPOINTS = ("T0_to_T1", "T1_to_T2", "T2_to_T3")
PCR_TIMINGS = ("T0", "T0-T1", "T0-T2", "T0-T3")
PCR_MODELS = ("M", "C", "C+M", "C+F", "C+F+M")
PCR_COMPARISONS = (
    "E5_S2_minus_S0_M",
    "E6_S2_CFM_minus_CF",
    "E6_S0_CFM_minus_CF",
    "S2_CM_minus_C",
    "S0_CM_minus_C",
)
AGGREGATE_ARTIFACTS = (
    ("Residualizer fits", METRICS / "residualizer_fits.csv"),
    ("Residualizer inventory", ROOT / "manifests/residualizer_inventory.json"),
    ("All representation metrics", METRICS / "representation_metrics.csv"),
    ("Static FTV", METRICS / "table_static_ftv.csv"),
    ("Observed delta FTV", METRICS / "table_observed_delta_ftv.csv"),
    ("SPH and residual-SPH", METRICS / "table_sph_and_residual.csv"),
    ("State redundancy", METRICS / "table_state_redundancy.csv"),
    ("Seed consistency", METRICS / "table_seed_consistency.csv"),
    ("Partial correlations", METRICS / "table_partial_correlations.csv"),
    ("Optimization safety", METRICS / "optimization_safety.csv"),
    ("Optimization trajectories", METRICS / "optimization_trajectories.csv"),
    ("Representation effects", METRICS / "representation_effects.json"),
    ("pCR complementarity", METRICS / "table_pcr_complementarity.csv"),
    ("Paired bootstrap", METRICS / "paired_bootstrap.csv"),
    ("pCR effects", METRICS / "pcr_effects.json"),
    ("Gate decision", DECISION),
    ("Execution status", METRICS / "execution_status.json"),
)
AGGREGATE_FIGURES = (
    ("Primary representation effects", ROOT / "figures/representation_effects.svg"),
    ("Residual-SPH organization", ROOT / "figures/sph_res_organization.svg"),
    ("Post-freeze pCR effects", ROOT / "figures/pcr_effects.svg"),
)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"formal report is missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"formal report found invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _read_csv(path: Path, required: set[str], *, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"formal report is missing {label}: {path}")
    frame = pd.read_csv(path)
    if missing := required.difference(frame.columns):
        raise ValueError(f"{label} misses columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{label} is empty")
    return frame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_markdown_links(
    artifacts: tuple[tuple[str, Path], ...],
    *,
    report_directory: Path,
    label: str,
    require_svg: bool = False,
) -> str:
    """Validate public report artifacts and return portable relative links."""

    lines: list[str] = []
    for title, path in artifacts:
        if not path.is_file():
            raise FileNotFoundError(f"formal report is missing {label}: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"formal report found empty {label}: {path}")
        if require_svg:
            if path.suffix.lower() != ".svg":
                raise ValueError(f"formal report expected an SVG {label}: {path}")
            if "<svg" not in path.read_text(encoding="utf-8"):
                raise ValueError(f"formal report found invalid SVG {label}: {path}")
        relative = Path(os.path.relpath(path, start=report_directory)).as_posix()
        if Path(relative).is_absolute():
            raise ValueError(f"formal report artifact link is not relative: {relative}")
        lines.append(f"- [{title}]({relative})")
    return "\n".join(lines)


def _format_cell(value: Any, *, signed: bool = False) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return "NA"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            return "NA"
        return f"{number:+.4f}" if signed else f"{number:.4f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    rename: Mapping[str, str] | None = None,
    signed: set[str] | None = None,
) -> str:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"report table misses required columns: {missing}")
    if frame.empty:
        raise ValueError("refusing to render an empty formal report table")
    signed = signed or set()
    selected = frame.loc[:, columns]
    headers = [rename.get(column, column) if rename else column for column in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in selected.itertuples(index=False, name=None):
        cells = [
            _format_cell(value, signed=column in signed)
            for column, value in zip(columns, row, strict=True)
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _coerce_bool(series: pd.Series, *, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    if unexpected := sorted(set(normalized) - set(mapping)):
        raise ValueError(f"{label} has non-boolean values: {unexpected}")
    return normalized.map(mapping).astype(bool)


def _ordered(
    frame: pd.DataFrame,
    *,
    arm: bool = True,
    endpoint_order: tuple[str, ...] | None = None,
    timing_order: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    output = frame.copy()
    sort_columns: list[str] = []
    if arm and "arm" in output:
        output["arm"] = pd.Categorical(output["arm"], ARMS, ordered=True)
        sort_columns.append("arm")
    if "seed_base" in output:
        sort_columns.append("seed_base")
    if endpoint_order is not None and "endpoint" in output:
        output["endpoint"] = pd.Categorical(
            output["endpoint"], endpoint_order, ordered=True
        )
        sort_columns.append("endpoint")
    if timing_order is not None and "timing" in output:
        output["timing"] = pd.Categorical(
            output["timing"], timing_order, ordered=True
        )
        sort_columns.append("timing")
    return output.sort_values(sort_columns).reset_index(drop=True)


def _one(frame: pd.DataFrame, **identity: object) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column, value in identity.items():
        if column not in frame:
            raise ValueError(f"lookup column is missing: {column}")
        mask &= frame[column].eq(value)
    selected = frame.loc[mask]
    if len(selected) != 1:
        raise ValueError(f"expected one aggregate row for {identity}, observed {len(selected)}")
    return selected.iloc[0]


def _seed_values_text(values: Mapping[int, float]) -> str:
    if set(values) != set(SEEDS):
        raise ValueError(f"paired seed map differs from {SEEDS}: {sorted(values)}")
    mean = sum(float(values[seed]) for seed in SEEDS) / len(SEEDS)
    return ", ".join(f"seed {seed} {_format_cell(values[seed], signed=True)}" for seed in SEEDS) + f"；均值 {_format_cell(mean, signed=True)}"


def _effect_values(effects: Mapping[str, Any], effect: str) -> dict[int, float]:
    raw = effects.get("by_seed")
    if not isinstance(raw, Mapping) or set(raw) != {str(seed) for seed in SEEDS}:
        raise ValueError("representation effects do not contain the two frozen seeds")
    values = {seed: float(raw[str(seed)][effect]) for seed in SEEDS}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError(f"{effect} contains a non-finite value")
    return values


def _nested_effect_values(
    effects: Mapping[str, Any], key: str, timing: str
) -> dict[int, float]:
    raw = effects.get(key)
    if not isinstance(raw, Mapping) or timing not in raw:
        raise ValueError(f"pCR effects miss {key}/{timing}")
    timing_values = raw[timing]
    if not isinstance(timing_values, Mapping) or set(timing_values) != {
        str(seed) for seed in SEEDS
    }:
        raise ValueError(f"pCR effects have wrong seed coverage for {key}/{timing}")
    values = {seed: float(timing_values[str(seed)]) for seed in SEEDS}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError(f"pCR effects contain a non-finite value for {key}/{timing}")
    return values


def _timing_effect_text(effects: Mapping[str, Any], key: str) -> str:
    return "；".join(
        f"{timing}: {_seed_values_text(_nested_effect_values(effects, key, timing))}"
        for timing in PCR_TIMINGS
    )


def _consistent_positive_timings(
    effects: Mapping[str, Any], key: str, timings: tuple[str, ...] = PCR_TIMINGS
) -> list[str]:
    return [
        timing
        for timing in timings
        if all(value > 0.0 for value in _nested_effect_values(effects, key, timing).values())
    ]


def _task_effect(
    frame: pd.DataFrame,
    *,
    task: str,
    endpoint: str,
    space: str,
    comparison: str,
    reference: str,
    metric: str,
) -> dict[int, float]:
    values: dict[int, float] = {}
    for seed in SEEDS:
        comparison_row = _one(
            frame,
            arm=comparison,
            seed_base=seed,
            task=task,
            endpoint=endpoint,
            space=space,
        )
        reference_row = _one(
            frame,
            arm=reference,
            seed_base=seed,
            task=task,
            endpoint=endpoint,
            space=space,
        )
        if metric not in comparison_row or metric not in reference_row:
            raise ValueError(f"task effect metric is missing: {metric}")
        values[seed] = float(comparison_row[metric] - reference_row[metric])
    return values


def _paired_metric_values(
    effects: Mapping[str, Any],
    effect_name: str,
    timing: str,
    metric: str,
) -> dict[int, float]:
    summaries = effects.get("paired_metric_effect_summaries")
    if not isinstance(summaries, Mapping) or effect_name not in summaries:
        raise ValueError(f"pCR paired summaries miss {effect_name}")
    effect = summaries[effect_name]
    if not isinstance(effect, Mapping) or timing not in effect:
        raise ValueError(f"pCR paired summaries miss {effect_name}/{timing}")
    summary = effect[timing]
    if not isinstance(summary, Mapping) or not isinstance(summary.get("by_seed"), Mapping):
        raise ValueError(f"invalid pCR paired summary for {effect_name}/{timing}")
    by_seed = summary["by_seed"]
    if set(by_seed) != {str(seed) for seed in SEEDS}:
        raise ValueError(f"pCR paired summary has wrong seeds for {effect_name}/{timing}")
    values = {seed: float(by_seed[str(seed)][metric]) for seed in SEEDS}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError(f"pCR paired summary is non-finite for {effect_name}/{timing}/{metric}")
    mean = float(summary["two_seed_mean"][metric])
    positive = bool(summary["both_seeds_positive"][metric])
    if not math.isclose(mean, sum(values.values()) / len(values), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"pCR paired summary mean disagrees for {effect_name}/{timing}/{metric}")
    if positive != all(value > 0.0 for value in values.values()):
        raise ValueError(f"pCR paired summary direction disagrees for {effect_name}/{timing}/{metric}")
    if metric == "delta_auroc":
        legacy = _nested_effect_values(effects, effect_name, timing)
        if any(
            not math.isclose(values[seed], legacy[seed], rel_tol=0.0, abs_tol=1e-12)
            for seed in SEEDS
        ):
            raise ValueError(f"legacy and paired AUROC effects disagree for {effect_name}/{timing}")
    return values


def _map_timing_text(values: Mapping[str, Mapping[int, float]]) -> str:
    return "；".join(
        f"{timing}: {_seed_values_text(values[timing])}" for timing in PCR_TIMINGS
    )


def _paired_bootstrap_audit(frame: pd.DataFrame) -> pd.DataFrame:
    expected = {
        (comparison, timing, seed)
        for comparison in PCR_COMPARISONS
        for timing in PCR_TIMINGS
        for seed in SEEDS
    }
    observed = set(
        zip(
            frame["comparison"].astype(str),
            frame["timing"].astype(str),
            frame["seed_base"].astype(int),
            strict=True,
        )
    )
    if observed != expected or len(frame) != len(expected):
        raise ValueError("paired bootstrap table has incomplete formal coverage")
    exact_strings = {
        "orientation": "comparison_minus_reference",
        "auroc_orientation": "comparison_minus_reference",
        "auprc_orientation": "comparison_minus_reference",
        "brier_orientation": "reference_minus_comparison_lower_is_better",
        "aggregation": "pooled_oof_paired_patient_bootstrap",
        "stratification": "patient_within_outer_fold",
        "bootstrap_unit": "patient_within_outer_fold",
        "ci_method": "percentile",
    }
    for column, expected_value in exact_strings.items():
        if set(frame[column].astype(str)) != {expected_value}:
            raise ValueError(f"paired bootstrap {column} contract drifted")
    if (
        not frame["n_bootstrap"].eq(2_000).all()
        or not frame["confidence_level"].eq(0.95).all()
        or not frame["fold_count"].eq(5).all()
    ):
        raise ValueError("paired bootstrap draw/CI/fold contract drifted")
    numeric_columns = (
        "reference_auroc", "comparison_auroc", "delta_auroc",
        "delta_auroc_ci_lower", "delta_auroc_ci_upper",
        "delta_auroc_bootstrap_probability_positive",
        "reference_auprc", "comparison_auprc", "delta_auprc",
        "delta_auprc_ci_lower", "delta_auprc_ci_upper",
        "delta_auprc_bootstrap_probability_positive",
        "reference_brier", "comparison_brier", "brier_improvement",
        "brier_improvement_ci_lower", "brier_improvement_ci_upper",
        "brier_improvement_bootstrap_probability_positive",
    )
    for column in numeric_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if not values.map(math.isfinite).all():
            raise ValueError(f"paired bootstrap column is non-finite: {column}")
    for column in (
        "delta_auroc_bootstrap_probability_positive",
        "delta_auprc_bootstrap_probability_positive",
        "brier_improvement_bootstrap_probability_positive",
    ):
        if not pd.to_numeric(frame[column]).between(0.0, 1.0).all():
            raise ValueError(f"paired bootstrap probability is outside [0,1]: {column}")
    output = frame.copy()
    output["comparison"] = pd.Categorical(
        output["comparison"], PCR_COMPARISONS, ordered=True
    )
    output["timing"] = pd.Categorical(output["timing"], PCR_TIMINGS, ordered=True)
    return output.sort_values(["comparison", "timing", "seed_base"]).reset_index(drop=True)


def _bootstrap_effect_text(
    frame: pd.DataFrame,
    *,
    comparison: str,
    point: str,
    lower: str,
    upper: str,
) -> str:
    pieces: list[str] = []
    for timing in PCR_TIMINGS:
        seeds: list[str] = []
        for seed in SEEDS:
            row = _one(
                frame,
                comparison=comparison,
                timing=timing,
                seed_base=seed,
            )
            seeds.append(
                f"seed {seed} {_format_cell(row[point], signed=True)} "
                f"[{_format_cell(row[lower], signed=True)}, {_format_cell(row[upper], signed=True)}]"
            )
        pieces.append(f"{timing}: " + ", ".join(seeds))
    return "；".join(pieces)


def _residualizer_audit(
    inventory: Mapping[str, Any], fits: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if inventory.get("status") != "FOLD_SAFE_RESIDUALIZERS_FITTED":
        raise ValueError("residualizer inventory is not formally complete")
    if inventory.get("patient_level_values_persisted") is not False:
        raise ValueError("residualizer inventory persisted patient-level values")
    artifacts = inventory.get("fold_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 5:
        raise ValueError("residualizer inventory must contain five fold artifacts")
    inventory_rows: list[dict[str, object]] = []
    root = ROOT.resolve()
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise ValueError("invalid residualizer artifact record")
        path = (ROOT / str(artifact["file"])).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("residualizer artifact escaped the experiment root") from error
        if not path.is_file() or _sha256(path) != artifact.get("artifact_sha256"):
            raise ValueError(f"residualizer artifact is missing or hash-mismatched: {path}")
        payload = _read_json(path, label="fold residualizer")
        if (
            payload.get("fit_scope") != "outer_train_only_per_fold_and_visit"
            or payload.get("dynamic_sph_supervision") is not False
            or len(payload.get("residualizers", [])) != 4
        ):
            raise ValueError("fold residualizer contract drifted")
        split_counts = artifact.get("split_counts")
        if not isinstance(split_counts, Mapping):
            raise ValueError("residualizer split counts are missing")
        inventory_rows.append(
            {
                "fold": int(artifact["fold"]),
                "train": int(split_counts["train"]),
                "val": int(split_counts["val"]),
                "test": int(split_counts["test"]),
                "visits": len(payload["residualizers"]),
                "artifact_sha256": str(artifact["artifact_sha256"]),
            }
        )
    expected = {(fold, visit) for fold in range(5) for visit in VISITS}
    observed = set(zip(fits["fold"].astype(int), fits["visit"].astype(str), strict=True))
    if observed != expected or len(fits) != 20:
        raise ValueError("residualizer coefficient table is not exact fold-by-visit coverage")
    return (
        pd.DataFrame(inventory_rows).sort_values("fold").reset_index(drop=True),
        fits.sort_values(["fold", "visit"]).reset_index(drop=True),
    )


def _sph_combined_table(sph: pd.DataFrame) -> pd.DataFrame:
    keys = ["arm", "seed_base", "endpoint"]
    expected_identity = {
        (arm, seed, endpoint, task, space)
        for arm in ARMS
        for seed in SEEDS
        for endpoint in VISITS
        for task, space in (
            ("raw_sph", "natural"),
            ("raw_sph", "transformed"),
            ("sph_res", "residual"),
            ("sph_res", "reconstructed_natural"),
        )
    }
    observed_identity = set(
        zip(
            sph["arm"].astype(str),
            sph["seed_base"].astype(int),
            sph["endpoint"].astype(str),
            sph["task"].astype(str),
            sph["space"].astype(str),
            strict=True,
        )
    )
    if observed_identity != expected_identity or len(sph) != len(expected_identity):
        raise ValueError("SPH table does not have exact arm/seed/visit/space coverage")
    expected_aggregation = {
        ("raw_sph", "natural"): "pooled_5fold_oof_within_seed",
        ("raw_sph", "transformed"): (
            "outer_test_n_weighted_fold_transformed_metrics"
        ),
        ("sph_res", "residual"): (
            "pooled_5fold_oof_residual_rank_and_fold_weighted_scale_metrics"
        ),
        ("sph_res", "reconstructed_natural"): (
            "pooled_5fold_oof_conditional_target_reconstruction"
        ),
    }
    for (task, space), aggregation in expected_aggregation.items():
        selected = sph.loc[sph["task"].eq(task) & sph["space"].eq(space)]
        if set(selected["aggregation"].astype(str)) != {aggregation}:
            raise ValueError(f"SPH aggregation contract drifted for {task}/{space}")
    residual_contract = sph.loc[
        sph["task"].eq("sph_res") & sph["space"].eq("residual")
    ]
    if set(residual_contract["rank_aggregation"].astype(str)) != {
        "pooled_5fold_oof_analysis_coordinate_within_seed"
    }:
        raise ValueError("SPH_res primary rank aggregation contract drifted")
    if set(residual_contract["scale_metric_aggregation"].astype(str)) != {
        "outer_test_n_weighted_fold_metric_with_rmse_from_weighted_fold_mse"
    }:
        raise ValueError("SPH_res scale-metric aggregation contract drifted")
    raw = sph.loc[sph["task"].eq("raw_sph") & sph["space"].eq("natural")].copy()
    raw = raw.loc[
        :,
        keys
        + [
            "n",
            "raw_natural_spearman",
            "raw_natural_pearson",
            "raw_natural_r2",
            "raw_natural_rmse",
            "raw_natural_mae",
        ],
    ]
    raw = raw.rename(
        columns={
            "raw_natural_spearman": "raw_sph_spearman",
            "raw_natural_pearson": "raw_sph_pearson",
            "raw_natural_r2": "raw_sph_natural_r2",
            "raw_natural_rmse": "raw_sph_rmse",
            "raw_natural_mae": "raw_sph_mae",
        }
    )
    residual = sph.loc[
        sph["task"].eq("sph_res") & sph["space"].eq("residual")
    ].copy()
    residual = residual.loc[
        :,
        keys
        + [
            "residual_space_spearman",
            "residual_space_pearson",
            "residual_space_r2",
            "residual_space_rmse",
            "residual_space_mae",
            "fold_weighted_residual_space_spearman",
            "fold_weighted_residual_space_pearson",
        ],
    ]
    residual = residual.rename(
        columns={
            "residual_space_spearman": "sph_res_spearman",
            "residual_space_pearson": "sph_res_pearson",
        }
    )
    reconstructed = sph.loc[
        sph["task"].eq("sph_res") & sph["space"].eq("reconstructed_natural")
    ].copy()
    reconstructed = reconstructed.loc[
        :,
        keys
        + [
            "reconstructed_natural_spearman",
            "reconstructed_natural_pearson",
            "reconstructed_natural_r2",
            "reconstructed_natural_rmse",
            "reconstructed_natural_mae",
            "reconstructed_natural_variance_ratio",
        ],
    ]
    reconstructed = reconstructed.rename(
        columns={
            "reconstructed_natural_spearman": "reconstructed_sph_spearman",
            "reconstructed_natural_pearson": "reconstructed_sph_pearson",
            "reconstructed_natural_r2": "reconstructed_sph_natural_r2",
            "reconstructed_natural_rmse": "reconstructed_sph_rmse",
            "reconstructed_natural_mae": "reconstructed_sph_mae",
            "reconstructed_natural_variance_ratio": "reconstructed_sph_variance_ratio",
        }
    )
    combined = raw.merge(residual, on=keys, how="inner", validate="one_to_one")
    combined = combined.merge(reconstructed, on=keys, how="inner", validate="one_to_one")
    expected_rows = len(ARMS) * len(SEEDS) * len(VISITS)
    if len(combined) != expected_rows:
        raise ValueError(
            f"SPH combined table expected {expected_rows} rows, observed {len(combined)}"
        )
    numeric = combined.select_dtypes(include="number")
    if numeric.isna().any().any() or not numeric.map(math.isfinite).all().all():
        raise ValueError("SPH combined table contains a missing/non-finite selected metric")
    return _ordered(combined, endpoint_order=VISITS)


def _state_redundancy_audit(frame: pd.DataFrame) -> pd.DataFrame:
    expected = {
        (arm, seed, task, endpoint)
        for arm in ARMS
        for seed in SEEDS
        for task in ("ftv_to_state", "sph_to_state", "sph_res_to_state")
        for endpoint in VISITS
    }
    observed = set(
        zip(
            frame["arm"].astype(str),
            frame["seed_base"].astype(int),
            frame["task"].astype(str),
            frame["endpoint"].astype(str),
            strict=True,
        )
    )
    if observed != expected or len(frame) != len(expected):
        raise ValueError("target-to-state redundancy table has incomplete formal coverage")
    if set(frame["direction"].astype(str)) != {
        "scalar_phenotype_to_192D_response_state"
    }:
        raise ValueError("target-to-state redundancy direction drifted")
    if set(frame["primary_metric"].astype(str)) != {
        "state_variance_weighted_r2"
    }:
        raise ValueError("target-to-state redundancy primary metric drifted")
    pooled = _coerce_bool(
        frame["cross_fold_state_vectors_pooled"], label="cross-fold state pooling"
    )
    if pooled.any() or not frame["fold_count"].eq(5).all():
        raise ValueError("target-to-state redundancy aggregation is not fold-safe")
    return _ordered(frame, endpoint_order=VISITS)


def _seed_consistency_audit(frame: pd.DataFrame) -> pd.DataFrame:
    expected: set[tuple[str, str, str, str]] = set()
    for arm in ARMS:
        for endpoint in VISITS:
            expected.update(
                {
                    (arm, "raw_sph", endpoint, "natural_spearman"),
                    (arm, "raw_sph", endpoint, "natural_r2"),
                    (arm, "sph_res", endpoint, "residual_space_spearman"),
                    (arm, "sph_res", endpoint, "residual_space_r2"),
                    (arm, "sph_res", endpoint, "reconstructed_natural_r2"),
                    (arm, "ftv_to_state", endpoint, "state_variance_weighted_r2"),
                    (arm, "sph_to_state", endpoint, "state_variance_weighted_r2"),
                    (arm, "sph_res_to_state", endpoint, "state_variance_weighted_r2"),
                }
            )
    observed = set(
        zip(
            frame["arm"].astype(str),
            frame["task"].astype(str),
            frame["endpoint"].astype(str),
            frame["metric"].astype(str),
            strict=True,
        )
    )
    if observed != expected or len(frame) != len(expected):
        raise ValueError("seed-consistency table has incomplete formal coverage")
    if not frame["seed_count"].eq(2).all() or set(
        frame["independent_unit"].astype(str)
    ) != {"training_seed"}:
        raise ValueError("seed-consistency independent-unit contract drifted")
    for column in (
        "seed_2026",
        "seed_3026",
        "seed_mean",
        "seed_minimum",
        "seed_maximum",
        "absolute_seed_difference",
    ):
        if not pd.to_numeric(frame[column], errors="coerce").map(math.isfinite).all():
            raise ValueError(f"seed-consistency column is non-finite: {column}")
    return _ordered(frame, endpoint_order=VISITS)


def _effect_tables(
    effects: Mapping[str, Any], pcr_effects: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    representation_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        row = {"seed_base": seed}
        row.update({effect: _effect_values(effects, effect)[seed] for effect in ("E1", "E2", "E3", "E4")})
        representation_rows.append(row)
    representation_rows.append(
        {
            "seed_base": "mean",
            **{
                effect: sum(_effect_values(effects, effect).values()) / len(SEEDS)
                for effect in ("E1", "E2", "E3", "E4")
            },
        }
    )

    names = (
        ("E5: S2-S0 MRI-only", "E5_S2_minus_S0_MRI_only"),
        ("E6: S2 C+F+M-(C+F)", "E6_S2_C_plus_F_plus_M_minus_C_plus_F"),
        ("S0 C+F+M-(C+F)", "S0_C_plus_F_plus_M_minus_C_plus_F"),
        ("S2 C+M-C", "S2_C_plus_M_minus_C"),
        ("S0 C+M-C", "S0_C_plus_M_minus_C"),
    )
    pcr_rows: list[dict[str, object]] = []
    for label, key in names:
        for timing in PCR_TIMINGS:
            values = _nested_effect_values(pcr_effects, key, timing)
            pcr_rows.append(
                {
                    "comparison": label,
                    "timing": timing,
                    "seed_2026": values[2026],
                    "seed_3026": values[3026],
                    "mean": sum(values.values()) / len(values),
                }
            )
    return pd.DataFrame(representation_rows), pd.DataFrame(pcr_rows)


def _pcr_pivot(pcr: pd.DataFrame) -> pd.DataFrame:
    observed_models = set(pcr["model"].astype(str))
    if observed_models != set(PCR_MODELS):
        raise ValueError(f"pCR model coverage drifted: {sorted(observed_models)}")
    counts = pcr.groupby(["arm", "seed_base", "timing"], observed=True)["model"].nunique()
    if not counts.eq(len(PCR_MODELS)).all():
        raise ValueError("one or more pCR cells lack all five models")
    pivot = pcr.pivot(
        index=["arm", "seed_base", "timing", "n"], columns="model", values="auroc"
    ).reset_index()
    pivot.columns.name = None
    return _ordered(pivot, timing_order=PCR_TIMINGS)


def main() -> None:
    decision = _read_json(DECISION, label="decision")
    if decision.get("status") != "FORMAL_TWO_SEED_PILOT_COMPLETE":
        raise RuntimeError("formal report requires a completed two-seed decision")
    if decision.get("pcr_evaluation_was_post_freeze") is not True:
        raise RuntimeError("formal decision does not attest post-freeze pCR evaluation")

    artifact_links = _artifact_markdown_links(
        AGGREGATE_ARTIFACTS,
        report_directory=REPORT.parent,
        label="aggregate CSV/JSON artifact",
    )
    figure_links = _artifact_markdown_links(
        AGGREGATE_FIGURES,
        report_directory=REPORT.parent,
        label="aggregate figure",
        require_svg=True,
    )

    static = _read_csv(
        METRICS / "table_static_ftv.csv",
        {"arm", "seed_base", "task", "endpoint", "space", "n", "spearman", "pearson", "natural_r2", "rmse", "mae", "variance_ratio"},
        label="static FTV table",
    )
    delta = _read_csv(
        METRICS / "table_observed_delta_ftv.csv",
        {"arm", "seed_base", "task", "endpoint", "space", "n", "spearman", "pearson", "natural_r2", "rmse", "mae", "variance_ratio"},
        label="observed delta-FTV table",
    )
    sph = _read_csv(
        METRICS / "table_sph_and_residual.csv",
        {
            "arm", "seed_base", "task", "endpoint", "space", "n", "aggregation",
            "rank_aggregation", "scale_metric_aggregation",
            "raw_natural_spearman", "raw_natural_pearson", "raw_natural_r2",
            "raw_natural_rmse", "raw_natural_mae",
            "residual_space_spearman", "residual_space_pearson",
            "residual_space_r2", "residual_space_rmse", "residual_space_mae",
            "fold_weighted_residual_space_spearman",
            "fold_weighted_residual_space_pearson",
            "reconstructed_natural_spearman", "reconstructed_natural_pearson",
            "reconstructed_natural_r2", "reconstructed_natural_rmse",
            "reconstructed_natural_mae", "reconstructed_natural_variance_ratio",
        },
        label="SPH table",
    )
    redundancy = _read_csv(
        METRICS / "table_state_redundancy.csv",
        {
            "arm", "seed_base", "task", "endpoint", "direction",
            "target_coordinate", "state_coordinate", "primary_metric",
            "state_variance_weighted_r2", "state_uniform_average_r2",
            "state_standardized_rmse", "state_standardized_mae", "n",
            "fold_count", "aggregation", "cross_fold_state_vectors_pooled",
        },
        label="target-to-state redundancy table",
    )
    seed_consistency = _read_csv(
        METRICS / "table_seed_consistency.csv",
        {
            "family", "arm", "task", "endpoint", "metric", "seed_2026",
            "seed_3026", "seed_mean", "seed_minimum", "seed_maximum",
            "absolute_seed_difference", "same_sign", "both_positive",
            "independent_unit", "seed_count",
        },
        label="seed-consistency table",
    )
    partial = _read_csv(
        METRICS / "table_partial_correlations.csv",
        {"arm", "seed_base", "endpoint", "n", "control_dimension", "partial_pearson", "partial_spearman"},
        label="partial-correlation table",
    )
    safety = _read_csv(
        METRICS / "optimization_safety.csv",
        {"seed_base", "fold", "selected_epoch", "selection_mode", "paired_s0_state_loss", "allowed_state_loss", "selected_validation_state_loss", "selected_validation_ftv_loss", "selected_validation_sph_loss", "state_loss_degradation_fraction", "optimization_safety_pass", "test_or_pcr_used"},
        label="optimization-safety table",
    )
    residualizer_fits = _read_csv(
        METRICS / "residualizer_fits.csv",
        {"fold", "visit", "n_train", "coefficient", "intercept", "residual_train_mean", "residual_train_population_scale", "residualizer_id"},
        label="residualizer-fit table",
    )
    pcr = _read_csv(
        METRICS / "table_pcr_complementarity.csv",
        {"arm", "seed_base", "timing", "model", "n", "n_positive", "auroc", "auprc", "brier", "aggregation"},
        label="pCR complementarity table",
    )
    bootstrap = _read_csv(
        METRICS / "paired_bootstrap.csv",
        {
            "comparison", "timing", "seed_base", "metrics", "orientation",
            "auroc_orientation", "auprc_orientation", "brier_orientation",
            "aggregation", "stratification", "bootstrap_unit", "ci_method",
            "n", "n_positive", "n_negative", "fold_count", "n_bootstrap",
            "n_valid_auroc_bootstrap", "n_valid_auprc_bootstrap",
            "n_valid_brier_bootstrap", "confidence_level", "bootstrap_seed",
            "reference_auroc", "comparison_auroc", "delta_auroc",
            "delta_auroc_bootstrap_mean", "delta_auroc_ci_lower",
            "delta_auroc_ci_upper",
            "delta_auroc_bootstrap_probability_positive",
            "reference_auprc", "comparison_auprc", "delta_auprc",
            "delta_auprc_bootstrap_mean", "delta_auprc_ci_lower",
            "delta_auprc_ci_upper",
            "delta_auprc_bootstrap_probability_positive",
            "reference_brier", "comparison_brier", "brier_improvement",
            "brier_improvement_bootstrap_mean", "brier_improvement_ci_lower",
            "brier_improvement_ci_upper",
            "brier_improvement_bootstrap_probability_positive",
        },
        label="paired bootstrap table",
    )
    effects = _read_json(METRICS / "representation_effects.json", label="representation effects")
    pcr_effects = _read_json(METRICS / "pcr_effects.json", label="pCR effects")
    inventory = _read_json(
        ROOT / "manifests/residualizer_inventory.json",
        label="residualizer inventory",
    )
    if decision.get("pcr_effects") != pcr_effects:
        raise ValueError("decision and standalone pCR-effect artifacts differ")

    gates = decision.get("gates")
    classification = decision.get("classification")
    if not isinstance(gates, Mapping) or set(gates) != set("ABCD"):
        raise ValueError("decision does not contain exact Gates A-D")
    if not isinstance(classification, Mapping):
        raise ValueError("decision classification must be an object")
    a, b, c, d = (bool(gates[name]["passed"]) for name in "ABCD")

    safety = safety.copy()
    safety["optimization_safety_pass"] = _coerce_bool(
        safety["optimization_safety_pass"], label="optimization safety"
    )
    safety["test_or_pcr_used"] = _coerce_bool(
        safety["test_or_pcr_used"], label="optimization test/pCR flag"
    )
    if len(safety) != 10 or safety["test_or_pcr_used"].any():
        raise ValueError("optimization-safety audit has wrong coverage or outcome access")
    safety_count = int(safety["optimization_safety_pass"].sum())
    if (
        safety_count != int(effects["optimization_safety"]["pass_count"])
        or len(safety) != int(effects["optimization_safety"]["total"])
        or safety_count != int(gates["A"]["optimization_safety_pass_count"])
    ):
        raise ValueError("optimization-safety aggregate artifacts disagree")

    residual_inventory, residual_fits = _residualizer_audit(
        inventory, residualizer_fits
    )
    sph_combined = _sph_combined_table(sph)
    redundancy = _state_redundancy_audit(redundancy)
    seed_consistency = _seed_consistency_audit(seed_consistency)
    representation_effect_table, pcr_effect_table = _effect_tables(effects, pcr_effects)
    pcr_auroc = _pcr_pivot(pcr)
    bootstrap = _paired_bootstrap_audit(bootstrap)

    e1 = _effect_values(effects, "E1")
    e2 = _effect_values(effects, "E2")
    e3 = _effect_values(effects, "E3")
    e4 = _effect_values(effects, "E4")
    s1_raw_gain = _task_effect(
        sph,
        task="raw_sph",
        endpoint="T0",
        space="natural",
        comparison="S1",
        reference="S0",
        metric="raw_natural_spearman",
    )
    s1_residual_gain = _task_effect(
        sph,
        task="sph_res",
        endpoint="T0",
        space="residual",
        comparison="S1",
        reference="S0",
        metric="residual_space_spearman",
    )
    raw_consistent = all(value > 0.0 for value in s1_raw_gain.values())

    s2_residual_absolute = {
        seed: float(
            _one(
                sph,
                arm="S2",
                seed_base=seed,
                task="sph_res",
                endpoint="T0",
                space="residual",
            )["residual_space_spearman"]
        )
        for seed in SEEDS
    }
    s2_reconstructed_r2 = {
        seed: float(
            _one(
                sph,
                arm="S2",
                seed_base=seed,
                task="sph_res",
                endpoint="T0",
                space="reconstructed_natural",
            )["reconstructed_natural_r2"]
        )
        for seed in SEEDS
    }

    e5_positive = _consistent_positive_timings(
        pcr_effects, "E5_S2_minus_S0_MRI_only"
    )
    gate_d_positive = list(gates["D"].get("qualifying_timings", []))
    clinical_increment = {
        timing: _paired_metric_values(
            pcr_effects, "S2_C_plus_M_minus_C", timing, "delta_auroc"
        )
        for timing in PCR_TIMINGS
    }
    clinical_auprc_increment = {
        timing: _paired_metric_values(
            pcr_effects, "S2_C_plus_M_minus_C", timing, "delta_auprc"
        )
        for timing in PCR_TIMINGS
    }
    clinical_brier_improvement = {
        timing: _paired_metric_values(
            pcr_effects, "S2_C_plus_M_minus_C", timing, "brier_improvement"
        )
        for timing in PCR_TIMINGS
    }
    clinical_positive = [
        timing
        for timing in PCR_TIMINGS
        if all(value > 0.0 for value in clinical_increment[timing].values())
    ]

    answers = [
        (
            f"1. **LOCAL response 性能是否保留？** {'是' if a else '否'}。"
            f"E1（S2-S0 静态 FTV macro Spearman）为 {_seed_values_text(e1)}；"
            f"E2（观测 ΔFTV macro）为 {_seed_values_text(e2)}；优化安全 {safety_count}/10。"
            f"Gate A {'通过' if a else '未通过'}。"
        ),
        (
            f"2. **raw-SPH grounding 是否有效？** 描述性结论为"
            f"{'两种子一致改善 raw SPH' if raw_consistent else '没有两种子一致的 raw-SPH 改善'}。"
            f"T0 raw-SPH Spearman 的 S1-S0 为 {_seed_values_text(s1_raw_gain)}；"
            f"同一状态对 SPH_res 的 S1-S0 为 {_seed_values_text(s1_residual_gain)}。"
            "S1 是 mechanistic control，没有独立预注册 Gate，因此此处不把方向性结果升级为正式验证。"
        ),
        (
            f"3. **residual-SPH grounding 是否有效？** {'是' if b else '否'}。"
            f"E3（S2-S0 T0 SPH_res Spearman）为 {_seed_values_text(e3)}；"
            f"Gate B {'通过' if b else '未通过'}，强形式"
            f"{'通过' if gates['B'].get('strong_form_passed') else '未通过'}。"
        ),
        (
            f"4. **residual grounding 是否优于 raw grounding？** {'是' if c else '否'}。"
            f"E4（S2-S1 T0 SPH_res Spearman）为 {_seed_values_text(e4)}；"
            f"Gate C {'通过' if c else '未通过'}。"
        ),
        (
            f"5. **S2 是否改善 FTV-independent morphology 表征？** "
            f"{'获得预注册的两种子支持' if b and c else '未获得预注册支持'}。"
            f"S2 的绝对 T0 SPH_res Spearman 为 {_seed_values_text(s2_residual_absolute)}；"
            f"重建 natural-SPH R2 为 {_seed_values_text(s2_reconstructed_r2)}。"
            "这里的 independent 只指冻结线性残差定义，不是因果独立。"
        ),
        (
            f"6. **是否影响静态 FTV？** E1 为 {_seed_values_text(e1)}；"
            f"两种子均不低于 -0.03：{'是' if gates['A'].get('static_ftv_both_seeds_ge_minus_0_03') else '否'}。"
        ),
        (
            f"7. **是否影响观测 ΔFTV？** E2 为 {_seed_values_text(e2)}；"
            f"两种子系统性下降：{'是' if gates['A'].get('delta_ftv_systematic_degradation') else '否'}。"
            "ΔFTV 从未用于监督或 checkpoint selection。"
        ),
        (
            "8. **是否改善 MRI-only pCR？** "
            f"E5（S2-S0 MRI-only ΔAUROC）为 {_timing_effect_text(pcr_effects, 'E5_S2_minus_S0_MRI_only')}。"
            f"两种子在同一 timing 均为正的 timing：{', '.join(e5_positive) if e5_positive else '无'}。"
            "E5 没有单独的通过 Gate，因此该答案是冻结后的描述性配对结论。"
        ),
        (
            "9. **是否增加 clinical 之外的信息？** "
            f"S2 的配对 C+M-C ΔAUROC 为 {_map_timing_text(clinical_increment)}；"
            f"ΔAUPRC 为 {_map_timing_text(clinical_auprc_increment)}；"
            f"Brier improvement（C Brier-C+M Brier，正值更好）为 {_map_timing_text(clinical_brier_improvement)}。"
            f"两种子在同一 timing 均为正的 timing：{', '.join(clinical_positive) if clinical_positive else '无'}。"
            "配对 ΔAUROC 的 95% bootstrap CI 为 "
            f"{_bootstrap_effect_text(bootstrap, comparison='S2_CM_minus_C', point='delta_auroc', lower='delta_auroc_ci_lower', upper='delta_auroc_ci_upper')}。"
            "该比较没有独立预注册 Gate，不能替代 Gate D。"
        ),
        (
            "10. **是否增加 clinical+FTV 之外的信息？** "
            f"{'是' if d else '否'}。S2 的 E6 为 "
            f"{_timing_effect_text(pcr_effects, 'E6_S2_C_plus_F_plus_M_minus_C_plus_F')}。"
            f"Gate D qualifying timing：{', '.join(gate_d_positive) if gate_d_positive else '无'}；"
            f"强形式 {'通过' if gates['D'].get('strong_form_passed') else '未通过'}。"
            f"S0 对照增量为 {_timing_effect_text(pcr_effects, 'S0_C_plus_F_plus_M_minus_C_plus_F')}。"
        ),
        (
            "11. **SPH 是否值得保留为 auxiliary target？** "
            + (
                "是，residual-SPH 可作为 morphology auxiliary 进入五种子确认；但若 Gate D 未通过，不可作为 headline pCR phenotype solution。"
                if a and b and c
                else (
                    "SPH 可至多保留为 raw-SPH 探索性 auxiliary；Gate C 未支持 residual target 的独特优势。"
                    if a and b and not c
                    else "否；当前 Gate A/B 组合不足以支持继续保留该 grounding auxiliary。"
                )
            )
        ),
        (
            f"12. **是否值得做五种子确认？** "
            f"{'是' if classification.get('five_seed_confirmation_justified') else '否'}。"
            f"正式 representation 分类为 `{classification.get('representation')}`；"
            f"downstream 分类为 `{classification.get('downstream')}`。"
        ),
    ]

    static_natural = _ordered(
        static.loc[static["space"].eq("natural")],
        endpoint_order=VISITS + ("macro",),
    )
    delta_natural = _ordered(
        delta.loc[delta["space"].eq("natural")],
        endpoint_order=DELTA_ENDPOINTS + ("macro",),
    )
    partial = _ordered(partial, endpoint_order=VISITS)
    safety = safety.sort_values(["seed_base", "fold"]).reset_index(drop=True)
    gate_rows = pd.DataFrame(
        [
            {
                "gate": "A",
                "passed": a,
                "strong": "NA",
                "evidence": f"E1={_seed_values_text(e1)}; E2={_seed_values_text(e2)}; safety={safety_count}/10",
            },
            {
                "gate": "B",
                "passed": b,
                "strong": bool(gates["B"].get("strong_form_passed")),
                "evidence": f"E3={_seed_values_text(e3)}",
            },
            {
                "gate": "C",
                "passed": c,
                "strong": "NA",
                "evidence": f"E4={_seed_values_text(e4)}",
            },
            {
                "gate": "D",
                "passed": d,
                "strong": bool(gates["D"].get("strong_form_passed")),
                "evidence": f"qualifying={','.join(gate_d_positive) if gate_d_positive else 'none'}",
            },
        ]
    )

    text = [
        "# FTV + residual-SPH grounding pilot 最终报告",
        "",
        "## 结论",
        "",
        f"正式 representation 分类：`{classification.get('representation')}`。下游分类：`{classification.get('downstream')}`。",
        f"Gate A/B/C/D 分别为 {'PASS' if a else 'FAIL'} / {'PASS' if b else 'FAIL'} / {'PASS' if c else 'FAIL'} / {'PASS' if d else 'FAIL'}。所有阈值是预注册的描述性决策阈值，不是 p-value。",
        "",
        "## 12 个问题",
        "",
        *answers,
        "",
        "## Target residualization audit",
        "",
        f"Residualizer inventory 状态为 `{inventory['status']}`：5/5 outer folds、20/20 fold×visit fits，source target SHA-256 `{inventory['source_target_table_sha256']}`，outer-fold SHA-256 `{inventory['outer_fold_manifest_sha256']}`。每个 visit 均在 outer-train 上执行 1/99 winsor；SPH identity/population-z；FTV log1p/population-z；Ridge(alpha=1)；epsilon 再 population-z。动态 SPH supervision=False，且没有持久化 patient-level target 值。",
        "",
        _table(
            residual_inventory,
            ["fold", "train", "val", "test", "visits", "artifact_sha256"],
        ),
        "",
        _table(
            residual_fits,
            ["fold", "visit", "n_train", "coefficient", "intercept", "residual_train_mean", "residual_train_population_scale"],
        ),
        "",
        "## Optimization safety",
        "",
        f"Primary S2 safety 为 {safety_count}/10；预注册要求至少 9/10。下表中的 SPH validation loss 仅被记录，未参与 selection；`test_or_pcr_used` 必须全部为 `false`。",
        "",
        _table(
            safety,
            ["seed_base", "fold", "selected_epoch", "selection_mode", "paired_s0_state_loss", "allowed_state_loss", "selected_validation_state_loss", "selected_validation_ftv_loss", "selected_validation_sph_loss", "state_loss_degradation_fraction", "optimization_safety_pass", "test_or_pcr_used"],
        ),
        "",
        "## Seed-level E1-E4",
        "",
        _table(
            representation_effect_table,
            ["seed_base", "E1", "E2", "E3", "E4"],
            signed={"E1", "E2", "E3", "E4"},
        ),
        "",
        "E1=静态 FTV macro Spearman S2-S0；E2=观测 ΔFTV macro Spearman S2-S0；E3=T0 SPH_res Spearman S2-S0；E4=T0 SPH_res Spearman S2-S1。",
        "",
        "## Static FTV（natural space）",
        "",
        _table(
            static_natural,
            ["arm", "seed_base", "endpoint", "n", "spearman", "pearson", "natural_r2", "rmse", "mae", "variance_ratio"],
        ),
        "",
        "## Observed ΔFTV（literal natural difference）",
        "",
        _table(
            delta_natural,
            ["arm", "seed_base", "endpoint", "n", "spearman", "pearson", "natural_r2", "rmse", "mae", "variance_ratio"],
        ),
        "",
        "## SPH / SPH_res organization",
        "",
        "`sph_res_spearman/pearson` 是五个 held-out fold 的 residual-z OOF 值在每个训练种子内 pooled 后计算，也是 Gate B/C 的 primary rank 指标。R2/RMSE/MAE/variance ratio 是各 fold 自身 residual coordinate 内计算后按 test n 汇总（RMSE 由加权 fold MSE 开方）；`fold_weighted_residual_space_*` 仅保留为 rank sensitivity。`reconstructed_sph_*` 是将预测 residual 与 observed same-visit FTV 的冻结 conditional component 合并后，pool 五折 OOF 在 natural SPH 上计算的条件重建指标。两者不得互称，尤其 reconstructed-natural R2 不是 residual natural R2。",
        "",
        _table(
            sph_combined,
            ["arm", "seed_base", "endpoint", "n", "raw_sph_spearman", "raw_sph_natural_r2", "sph_res_spearman", "fold_weighted_residual_space_spearman", "residual_space_r2", "residual_space_rmse", "residual_space_mae", "reconstructed_sph_spearman", "reconstructed_sph_natural_r2", "reconstructed_sph_rmse", "reconstructed_sph_mae", "reconstructed_sph_variance_ratio"],
        ),
        "",
        "## Redundancy and partial correlation",
        "",
        "预注册的 redundancy 方向是 scalar phenotype→192D response state。每个 outer fold 只用 train split 拟合 target scaler、逐维 state scaler 与 ridge；validation 选 alpha；test 只预测一次。主指标 `state_variance_weighted_r2` 是各 test-fold 内的总 SSE/总 SST 定义，再按 test n 汇总；不同 fold 的 state 坐标从不拼接。Partial correlation 表则对 state-derived raw SPH 与 target SPH 同时控制 same-visit FTV；partial Spearman 使用 rank-residual convention。",
        "",
        _table(
            redundancy,
            ["task", "endpoint", "arm", "seed_base", "n", "state_variance_weighted_r2", "state_uniform_average_r2", "state_standardized_rmse", "state_standardized_mae"],
        ),
        "",
        _table(
            partial,
            ["arm", "seed_base", "endpoint", "n", "control_dimension", "partial_spearman", "partial_pearson"],
        ),
        "",
        "## Seed consistency",
        "",
        "下表保留两个训练种子的独立点估计、均值、最小值、绝对种子差与方向一致性；fold 不是独立重复单位。",
        "",
        _table(
            seed_consistency,
            ["family", "task", "endpoint", "metric", "arm", "seed_2026", "seed_3026", "seed_mean", "seed_minimum", "absolute_seed_difference", "same_sign", "both_positive"],
        ),
        "",
        "## pCR complementarity（post-freeze）",
        "",
        "下表为每个训练种子的五折 pooled OOF AUROC。T0-T3 包含 late/pre-surgery T3 信息。pCR、clinical 与 treatment fields 仅在 representation freeze 后读取。",
        "",
        _table(
            pcr_auroc,
            ["arm", "seed_base", "timing", "n", "M", "C", "C+M", "C+F", "C+F+M"],
        ),
        "",
        "### Seed-level E5/E6 effects",
        "",
        _table(
            pcr_effect_table,
            ["comparison", "timing", "seed_2026", "seed_3026", "mean"],
            signed={"seed_2026", "seed_3026", "mean"},
        ),
        "",
        "### Paired patient bootstrap",
        "",
        "所有表均为 patient-within-outer-fold 的 2,000 次 paired percentile bootstrap。ΔAUROC 与 ΔAUPRC 定义为 comparison-reference；Brier improvement 定义为 reference-comparison（comparison 的 Brier 更低时为正）。因此三类 effect 都是正值有利于 comparison，`probability_positive` 是 bootstrap draw 严格大于 0 的比例。",
        "",
        "#### AUROC",
        "",
        _table(
            bootstrap,
            ["comparison", "timing", "seed_base", "n", "reference_auroc", "comparison_auroc", "delta_auroc", "delta_auroc_ci_lower", "delta_auroc_ci_upper", "delta_auroc_bootstrap_probability_positive", "n_bootstrap"],
            signed={"delta_auroc", "delta_auroc_ci_lower", "delta_auroc_ci_upper"},
        ),
        "",
        "#### AUPRC",
        "",
        _table(
            bootstrap,
            ["comparison", "timing", "seed_base", "n", "reference_auprc", "comparison_auprc", "delta_auprc", "delta_auprc_ci_lower", "delta_auprc_ci_upper", "delta_auprc_bootstrap_probability_positive", "n_bootstrap"],
            signed={"delta_auprc", "delta_auprc_ci_lower", "delta_auprc_ci_upper"},
        ),
        "",
        "#### Brier score",
        "",
        _table(
            bootstrap,
            ["comparison", "timing", "seed_base", "n", "reference_brier", "comparison_brier", "brier_improvement", "brier_improvement_ci_lower", "brier_improvement_ci_upper", "brier_improvement_bootstrap_probability_positive", "n_bootstrap"],
            signed={"brier_improvement", "brier_improvement_ci_lower", "brier_improvement_ci_upper"},
        ),
        "",
        "## Gate decision",
        "",
        _table(gate_rows, ["gate", "passed", "strong", "evidence"]),
        "",
        "所有 pCR 结果均在 representation freeze 后产生；pCR 没有选择 arm、epoch、target 或 lambda。S2-L10 仅为预注册 sensitivity，未替代 primary S2。SPH 仅解释为 non-volume morphological measurement；SPH_res 仅解释为冻结线性 residualization 下的 FTV-independent morphological component。",
        "",
        "## Aggregate artifacts",
        "",
        "以下均为 aggregate-only 公共产物；链接相对于本报告，不包含 private runtime path。",
        "",
        artifact_links,
        "",
        "## Aggregate figures",
        "",
        figure_links,
        "",
    ]

    current = REPORT.read_text(encoding="utf-8") if REPORT.exists() else ""
    pending_markers = (
        "尚未产生正式实验结果",
        "FORMAL_EXECUTION_NOT_STARTED_RESOURCE_GUARD",
    )
    if current and not any(marker in current for marker in pending_markers):
        raise FileExistsError("refusing to overwrite a non-pending final report")
    REPORT.write_text("\n".join(text), encoding="utf-8")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Pool five OOF folds per seed and publish aggregate representation results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC = EXPERIMENT_ROOT / "src"
sys.path.insert(0, str(SRC))

from residual_sph.evaluation import (  # noqa: E402
    partial_correlations_controlling_ftv,
    regression_metrics,
    seed_level_effects,
)
from residual_sph.preregistration import require_lock_sha256, verify_preregistration  # noqa: E402


PREDICTIONS = EXPERIMENT_ROOT / "predictions" / "formal_4x8"
CHECKPOINTS = EXPERIMENT_ROOT / "checkpoints" / "formal_4x8"
METRICS = EXPERIMENT_ROOT / "metrics"
PRIMARY_ARMS = ("S0", "S1", "S2")
ALL_ARMS = PRIMARY_ARMS + ("S2_L10",)
SEEDS = (2026, 3026)
FOLDS = tuple(range(5))
STATIC = ("T0", "T1", "T2", "T3")
DELTA = ("T0_to_T1", "T1_to_T2", "T2_to_T3")
STATE_DIAGNOSTIC_TASKS = (
    "ftv_to_state",
    "sph_to_state",
    "sph_res_to_state",
)


def _load_predictions() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for arm in ALL_ARMS:
        for seed in SEEDS:
            for fold in FOLDS:
                path = (
                    PREDICTIONS / f"seed_{seed}" / arm / f"fold_{fold}"
                    / "ridge_predictions.private.csv"
                )
                if not path.is_file():
                    raise FileNotFoundError(f"missing formal probe output: {path}")
                frame = pd.read_csv(path)
                if set(frame["arm"].astype(str)) != {arm} or set(frame["seed_base"].astype(int)) != {seed} or set(frame["fold"].astype(int)) != {fold}:
                    raise ValueError(f"probe identity drifted: {path}")
                frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    required = {
        "record_type", "patient_id", "arm", "seed_base", "fold", "task", "endpoint",
        "y_true_analysis", "y_pred_analysis", "y_true_natural", "y_pred_natural",
        "ftv_control", "test_predict_call_count", "refit_after_selection",
    }
    if missing := required.difference(combined.columns):
        raise ValueError(f"probe predictions miss columns: {sorted(missing)}")
    if not combined["test_predict_call_count"].eq(1).all() or combined["refit_after_selection"].astype(bool).any():
        raise ValueError("probe test/refit contract drifted")
    if set(combined["record_type"].astype(str)) != {"state_to_target_probe"}:
        raise ValueError("unexpected prediction record type")
    return combined


def _load_state_diagnostics() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for arm in ALL_ARMS:
        for seed in SEEDS:
            for fold in FOLDS:
                path = (
                    PREDICTIONS / f"seed_{seed}" / arm / f"fold_{fold}"
                    / "ridge_selection.private.csv"
                )
                if not path.is_file():
                    raise FileNotFoundError(f"missing formal probe selection: {path}")
                frame = pd.read_csv(path)
                required = {
                    "record_type",
                    "arm",
                    "seed_base",
                    "fold",
                    "task",
                    "endpoint",
                    "target_coordinate",
                    "state_coordinate",
                    "selected_alpha",
                    "n_test",
                    "test_predict_call_count",
                    "refit_after_selection",
                    "test_used_for_selection",
                    "state_dimension",
                    "nonconstant_test_state_dimensions",
                    "state_variance_weighted_r2",
                    "state_uniform_average_r2",
                    "state_standardized_rmse",
                    "state_standardized_mae",
                }
                if missing := required.difference(frame.columns):
                    raise ValueError(
                        f"state diagnostic selection misses columns: {sorted(missing)}"
                    )
                frame = frame.loc[
                    frame["record_type"].astype(str).eq("target_to_state_diagnostic")
                ].copy()
                if len(frame) != len(STATE_DIAGNOSTIC_TASKS) * len(STATIC):
                    raise ValueError(f"state diagnostic cell coverage drifted: {path}")
                if (
                    set(frame["arm"].astype(str)) != {arm}
                    or set(frame["seed_base"].astype(int)) != {seed}
                    or set(frame["fold"].astype(int)) != {fold}
                ):
                    raise ValueError(f"state diagnostic identity drifted: {path}")
                observed = set(
                    zip(
                        frame["task"].astype(str),
                        frame["endpoint"].astype(str),
                        strict=True,
                    )
                )
                expected = {
                    (task, endpoint)
                    for task in STATE_DIAGNOSTIC_TASKS
                    for endpoint in STATIC
                }
                if observed != expected:
                    raise ValueError(f"state diagnostic endpoint coverage drifted: {path}")
                frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if (
        not combined["test_predict_call_count"].eq(1).all()
        or combined["refit_after_selection"].astype(bool).any()
        or combined["test_used_for_selection"].astype(bool).any()
    ):
        raise ValueError("state diagnostic test/refit contract drifted")
    if not combined["state_dimension"].eq(192).all():
        raise ValueError("state diagnostic dimensionality drifted")
    return combined


def _metric_row(group: pd.DataFrame, *, space: str) -> dict[str, object]:
    truth = group[f"y_true_{space}"].to_numpy(dtype=np.float64)
    prediction = group[f"y_pred_{space}"].to_numpy(dtype=np.float64)
    values = regression_metrics(truth, prediction)
    return {
        "arm": str(group["arm"].iloc[0]),
        "seed_base": int(group["seed_base"].iloc[0]),
        "task": str(group["task"].iloc[0]),
        "endpoint": str(group["endpoint"].iloc[0]),
        "space": space,
        "fold_count": int(group["fold"].nunique()),
        "aggregation": "pooled_5fold_oof_within_seed",
        **values,
    }


def _fold_weighted_space_metrics(
    group: pd.DataFrame,
    *,
    space: str,
) -> dict[str, float | int | str]:
    """Aggregate a fold-specific target coordinate without concatenating it."""

    fold_values: list[dict[str, float | int]] = []
    weights: list[int] = []
    fold_sizes: dict[str, int] = {}
    for fold, fold_frame in group.groupby("fold", sort=True):
        values = regression_metrics(
            fold_frame[f"y_true_{space}"].to_numpy(dtype=np.float64),
            fold_frame[f"y_pred_{space}"].to_numpy(dtype=np.float64),
        )
        fold_values.append(values)
        weights.append(int(values["n"]))
        fold_sizes[str(int(fold))] = int(values["n"])
    if set(int(value) for value in group["fold"].unique()) != set(FOLDS):
        raise ValueError("fold-specific target metric lacks five folds")
    weight_array = np.asarray(weights, dtype=np.float64)

    def weighted(name: str) -> float:
        return float(
            np.average(
                np.asarray([float(value[name]) for value in fold_values]),
                weights=weight_array,
            )
        )

    return {
        "n": int(sum(weights)),
        "fold_count": len(fold_values),
        "fold_sizes_json": json.dumps(fold_sizes, sort_keys=True, separators=(",", ":")),
        "spearman": weighted("spearman"),
        "pearson": weighted("pearson"),
        "r2": weighted("natural_r2"),
        "rmse": float(
            np.sqrt(
                np.average(
                    np.asarray([float(value["rmse"]) ** 2 for value in fold_values]),
                    weights=weight_array,
                )
            )
        ),
        "mae": weighted("mae"),
        "variance_ratio": weighted("variance_ratio"),
    }


def _raw_sph_metric_rows(group: pd.DataFrame) -> list[dict[str, object]]:
    natural = _metric_row(group, space="natural")
    natural.update(
        {
            "raw_natural_spearman": natural["spearman"],
            "raw_natural_pearson": natural["pearson"],
            "raw_natural_r2": natural["natural_r2"],
            "raw_natural_rmse": natural["rmse"],
            "raw_natural_mae": natural["mae"],
            "raw_natural_variance_ratio": natural["variance_ratio"],
        }
    )
    transformed = _fold_weighted_space_metrics(group, space="analysis")
    identity = {
        "arm": str(group["arm"].iloc[0]),
        "seed_base": int(group["seed_base"].iloc[0]),
        "task": "raw_sph",
        "endpoint": str(group["endpoint"].iloc[0]),
    }
    transformed_row: dict[str, object] = {
        **identity,
        "space": "transformed",
        "aggregation": "outer_test_n_weighted_fold_transformed_metrics",
        "n": transformed["n"],
        "fold_count": transformed["fold_count"],
        "fold_sizes_json": transformed["fold_sizes_json"],
        "spearman": transformed["spearman"],
        "pearson": transformed["pearson"],
        "rmse": transformed["rmse"],
        "mae": transformed["mae"],
        "variance_ratio": transformed["variance_ratio"],
        "transformed_space_spearman": transformed["spearman"],
        "transformed_space_pearson": transformed["pearson"],
        "transformed_space_r2": transformed["r2"],
        "transformed_space_rmse": transformed["rmse"],
        "transformed_space_mae": transformed["mae"],
        "transformed_space_variance_ratio": transformed["variance_ratio"],
    }
    return [natural, transformed_row]


def _residual_metric_rows(group: pd.DataFrame) -> list[dict[str, object]]:
    """Expose residual and conditional-reconstruction metrics separately."""

    # SPH_res is standardized independently from each fold's outer-training
    # partition, but it has one preregistered residual-z interpretation.  The
    # primary rank endpoints therefore follow the frozen five-fold pooled-OOF
    # rule.  Scale-sensitive diagnostics remain within-fold calculations so
    # fold-specific residual scale cannot leak into R2/error aggregation.
    pooled_residual = _metric_row(group, space="analysis")
    fold_residual = _fold_weighted_space_metrics(group, space="analysis")
    reconstructed = _metric_row(group, space="natural")
    identity = {
        "arm": str(group["arm"].iloc[0]),
        "seed_base": int(group["seed_base"].iloc[0]),
        "task": "sph_res",
        "endpoint": str(group["endpoint"].iloc[0]),
    }
    residual_row: dict[str, object] = {
        **identity,
        "space": "residual",
        "aggregation": (
            "pooled_5fold_oof_residual_rank_and_fold_weighted_scale_metrics"
        ),
        "rank_aggregation": "pooled_5fold_oof_analysis_coordinate_within_seed",
        "scale_metric_aggregation": (
            "outer_test_n_weighted_fold_metric_with_rmse_from_weighted_fold_mse"
        ),
        "n": fold_residual["n"],
        "fold_count": fold_residual["fold_count"],
        "fold_sizes_json": fold_residual["fold_sizes_json"],
        # Generic aliases remain scoped by space=residual for downstream gate
        # code and are the pooled rank endpoints used by E3/E4.  R2 is
        # intentionally not called natural_r2 here.
        "spearman": pooled_residual["spearman"],
        "pearson": pooled_residual["pearson"],
        "rmse": fold_residual["rmse"],
        "mae": fold_residual["mae"],
        "variance_ratio": fold_residual["variance_ratio"],
        "residual_space_spearman": pooled_residual["spearman"],
        "residual_space_pearson": pooled_residual["pearson"],
        "residual_space_r2": fold_residual["r2"],
        "residual_space_rmse": fold_residual["rmse"],
        "residual_space_mae": fold_residual["mae"],
        "residual_space_variance_ratio": fold_residual["variance_ratio"],
        "fold_weighted_residual_space_spearman": fold_residual["spearman"],
        "fold_weighted_residual_space_pearson": fold_residual["pearson"],
    }
    reconstructed_row: dict[str, object] = {
        **identity,
        "space": "reconstructed_natural",
        "aggregation": "pooled_5fold_oof_conditional_target_reconstruction",
        "n": reconstructed["n"],
        "fold_count": reconstructed["fold_count"],
        "natural_r2": reconstructed["natural_r2"],
        "rmse": reconstructed["rmse"],
        "mae": reconstructed["mae"],
        "variance_ratio": reconstructed["variance_ratio"],
        "reconstructed_natural_spearman": reconstructed["spearman"],
        "reconstructed_natural_pearson": reconstructed["pearson"],
        "reconstructed_natural_r2": reconstructed["natural_r2"],
        "reconstructed_natural_rmse": reconstructed["rmse"],
        "reconstructed_natural_mae": reconstructed["mae"],
        "reconstructed_natural_variance_ratio": reconstructed["variance_ratio"],
        "natural_metric_interpretation": (
            "conditional_SPH_reconstruction_from_actual_FTV_plus_predicted_residual"
        ),
    }
    return [residual_row, reconstructed_row]


def _aggregate_probe_group(group: pd.DataFrame) -> list[dict[str, object]]:
    """Apply only aggregation rules valid for a probe's target coordinates."""

    tasks = set(group["task"].astype(str))
    if len(tasks) != 1:
        raise ValueError("probe aggregation group contains multiple tasks")
    task = next(iter(tasks))
    if task in {"static_ftv", "observed_delta_ftv"}:
        # Public primary metrics are common-unit natural values.  Fold-specific
        # transformed/standardized coordinates are deliberately omitted.
        return [_metric_row(group, space="natural")]
    if task == "raw_sph":
        return _raw_sph_metric_rows(group)
    if task == "sph_res":
        return _residual_metric_rows(group)
    raise ValueError(f"unknown representation probe task: {task}")


def _aggregate_state_diagnostics(source: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = source.groupby(["arm", "seed_base", "task", "endpoint"], sort=True)
    for (arm, seed, task, endpoint), group in grouped:
        if set(group["fold"].astype(int)) != set(FOLDS) or len(group) != len(FOLDS):
            raise ValueError(
                f"state diagnostic five-fold coverage drifted: {arm}/{seed}/{task}/{endpoint}"
            )
        weights = group["n_test"].to_numpy(dtype=np.float64)
        if np.any(weights <= 0.0):
            raise ValueError("state diagnostic fold has no held-out rows")
        target_coordinates = set(group["target_coordinate"].astype(str))
        state_coordinates = set(group["state_coordinate"].astype(str))
        if len(target_coordinates) != 1 or len(state_coordinates) != 1:
            raise ValueError("state diagnostic coordinate definition drifted across folds")

        def weighted(column: str) -> float:
            values = group[column].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"state diagnostic {column} is non-finite")
            return float(np.average(values, weights=weights))

        rows.append(
            {
                "arm": arm,
                "seed_base": int(seed),
                "task": task,
                "endpoint": endpoint,
                "direction": "scalar_phenotype_to_192D_response_state",
                "target_coordinate": next(iter(target_coordinates)),
                "state_coordinate": next(iter(state_coordinates)),
                "primary_metric": "state_variance_weighted_r2",
                "state_variance_weighted_r2": weighted(
                    "state_variance_weighted_r2"
                ),
                "state_uniform_average_r2": weighted("state_uniform_average_r2"),
                "state_standardized_rmse": float(
                    np.sqrt(
                        np.average(
                            group["state_standardized_rmse"].to_numpy(
                                dtype=np.float64
                            )
                            ** 2,
                            weights=weights,
                        )
                    )
                ),
                "state_standardized_mae": weighted("state_standardized_mae"),
                "n": int(np.sum(weights)),
                "fold_count": len(FOLDS),
                "minimum_nonconstant_test_state_dimensions": int(
                    group["nonconstant_test_state_dimensions"].min()
                ),
                "selected_alpha_by_fold_json": json.dumps(
                    {
                        str(int(row.fold)): float(row.selected_alpha)
                        for row in group.itertuples(index=False)
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "aggregation": (
                    "outer_test_n_weighted_fold_metrics_in_fold_train_standardized_state_space"
                ),
                "cross_fold_state_vectors_pooled": False,
            }
        )
    return pd.DataFrame(rows)


def _seed_consistency_rows(
    metrics: pd.DataFrame,
    state_diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    def add(
        source: pd.DataFrame,
        *,
        family: str,
        metric: str,
        value_column: str,
    ) -> None:
        for (arm, task, endpoint), group in source.groupby(
            ["arm", "task", "endpoint"], sort=True
        ):
            values = {
                int(row.seed_base): float(getattr(row, value_column))
                for row in group.itertuples(index=False)
            }
            if set(values) != set(SEEDS):
                raise ValueError(
                    f"seed-consistency coverage drifted: {arm}/{task}/{endpoint}/{metric}"
                )
            ordered = [values[seed] for seed in SEEDS]
            if not np.isfinite(np.asarray(ordered)).all():
                raise ValueError("seed-consistency input is non-finite")
            records.append(
                {
                    "family": family,
                    "arm": arm,
                    "task": task,
                    "endpoint": endpoint,
                    "metric": metric,
                    "seed_2026": ordered[0],
                    "seed_3026": ordered[1],
                    "seed_mean": float(np.mean(ordered)),
                    "seed_minimum": float(np.min(ordered)),
                    "seed_maximum": float(np.max(ordered)),
                    "absolute_seed_difference": float(abs(ordered[0] - ordered[1])),
                    "same_sign": bool(np.sign(ordered[0]) == np.sign(ordered[1])),
                    "both_positive": bool(ordered[0] > 0.0 and ordered[1] > 0.0),
                    "independent_unit": "training_seed",
                    "seed_count": len(SEEDS),
                }
            )

    raw = metrics.loc[
        metrics["task"].eq("raw_sph") & metrics["space"].eq("natural")
    ]
    add(raw, family="state_to_phenotype", metric="natural_spearman", value_column="spearman")
    add(raw, family="state_to_phenotype", metric="natural_r2", value_column="natural_r2")
    residual = metrics.loc[
        metrics["task"].eq("sph_res") & metrics["space"].eq("residual")
    ]
    add(
        residual,
        family="state_to_phenotype",
        metric="residual_space_spearman",
        value_column="residual_space_spearman",
    )
    add(
        residual,
        family="state_to_phenotype",
        metric="residual_space_r2",
        value_column="residual_space_r2",
    )
    reconstruction = metrics.loc[
        metrics["task"].eq("sph_res")
        & metrics["space"].eq("reconstructed_natural")
    ]
    add(
        reconstruction,
        family="state_to_phenotype",
        metric="reconstructed_natural_r2",
        value_column="reconstructed_natural_r2",
    )
    add(
        state_diagnostics,
        family="phenotype_to_state_redundancy",
        metric="state_variance_weighted_r2",
        value_column="state_variance_weighted_r2",
    )
    return pd.DataFrame(records)


def _macro_rows(metrics: pd.DataFrame) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    metric_names = ("spearman", "pearson", "natural_r2", "rmse", "mae", "variance_ratio")
    for task, endpoints in (("static_ftv", STATIC), ("observed_delta_ftv", DELTA)):
        current = metrics.loc[
            metrics["task"].eq(task)
            & metrics["endpoint"].isin(endpoints)
            & metrics["space"].eq("natural")
        ]
        for (arm, seed), group in current.groupby(["arm", "seed_base"], sort=True):
            if set(group["endpoint"]) != set(endpoints) or len(group) != len(endpoints):
                raise ValueError(f"macro endpoint coverage drifted for {arm}/{seed}/{task}")
            output.append(
                {
                    "arm": arm,
                    "seed_base": int(seed),
                    "task": task,
                    "endpoint": "macro",
                    "space": "natural",
                    "fold_count": 5,
                    "aggregation": "unweighted_mean_of_pooled_endpoint_metrics",
                    "n": int(group["n"].sum()),
                    **{name: float(group[name].mean()) for name in metric_names},
                }
            )
    return output


def _optimization_safety() -> tuple[pd.DataFrame, dict[int, dict[int, bool]]]:
    rows: list[dict[str, object]] = []
    flags: dict[int, dict[int, bool]] = {seed: {} for seed in SEEDS}
    for seed in SEEDS:
        for fold in FOLDS:
            path = CHECKPOINTS / f"seed_{seed}" / "S2" / f"fold_{fold}" / "selection.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing S2 selection: {path}")
            selection = json.loads(path.read_text(encoding="utf-8"))
            passed = bool(selection.get("optimization_safety_pass"))
            flags[seed][fold] = passed
            rows.append(
                {
                    "seed_base": seed,
                    "fold": fold,
                    "selected_epoch": int(selection["selected_epoch"]),
                    "selection_mode": selection["selection_mode"],
                    "paired_s0_state_loss": float(selection["paired_s0_state_loss"]),
                    "allowed_state_loss": float(selection["allowed_state_loss"]),
                    "selected_validation_state_loss": float(selection["selected_validation_state_loss"]),
                    "selected_validation_ftv_loss": float(selection["selected_validation_ftv_loss"]),
                    "selected_validation_sph_loss": float(selection["selected_validation_sph_loss"]),
                    "state_loss_degradation_fraction": float(selection["state_loss_degradation_fraction"]),
                    "optimization_safety_pass": passed,
                    "test_or_pcr_used": bool(selection.get("test_data_used")) or bool(selection.get("pcr_used")),
                }
            )
    return pd.DataFrame(rows), flags


def _trajectory_rows_from_selection(
    selection: dict[str, object], *, arm: str, seed: int, fold: int
) -> list[dict[str, object]]:
    """Return identifier-free epoch rows from one selected private run."""

    expected = {
        "arm": arm,
        "seed_base": seed,
        "fold": fold,
        "effective_seed": seed + fold,
        "test_data_used": False,
        "pcr_used": False,
        "clinical_used": False,
        "treatment_used": False,
    }
    for key, value in expected.items():
        if selection.get(key) != value:
            raise ValueError(f"optimization trajectory selection differs at {key}")
    raw_epochs = selection.get("epochs")
    if not isinstance(raw_epochs, list) or not raw_epochs or len(raw_epochs) > 12:
        raise ValueError("optimization trajectory has invalid epoch coverage")
    selected_epoch = int(selection["selected_epoch"])
    observed_epochs = [int(dict(row)["epoch"]) for row in raw_epochs]
    if observed_epochs != list(range(1, len(raw_epochs) + 1)):
        raise ValueError("optimization trajectory epochs are not contiguous")
    if selected_epoch not in observed_epochs:
        raise ValueError("selected epoch is absent from optimization trajectory")
    rows: list[dict[str, object]] = []
    numeric_fields = (
        "train_loss",
        "train_base_loss",
        "train_state_loss",
        "train_ftv_loss",
        "train_sph_loss",
        "train_representation_std",
        "val_loss",
        "val_base_objective",
        "val_state_loss",
        "val_ftv_loss",
        "val_sph_loss",
        "val_representation_std",
    )
    for raw in raw_epochs:
        epoch = dict(raw)
        values = {name: float(epoch[name]) for name in numeric_fields}
        if not np.isfinite(np.asarray(list(values.values()), dtype=np.float64)).all():
            raise ValueError("optimization trajectory contains non-finite metrics")
        rows.append(
            {
                "arm": arm,
                "seed_base": seed,
                "fold": fold,
                "effective_seed": seed + fold,
                "epoch": int(epoch["epoch"]),
                "selected_epoch": selected_epoch,
                "is_selected_epoch": int(epoch["epoch"]) == selected_epoch,
                "selection_mode": str(selection["selection_mode"]),
                "experiment_pass": bool(selection["experiment_pass"]),
                "checkpoint_eligible": bool(epoch["checkpoint_eligible"]),
                "base_gate_pass": bool(epoch["base_gate_pass"]),
                "noncollapse": bool(epoch["noncollapse"]),
                "finite": bool(epoch["finite"]),
                **values,
            }
        )
    return rows


def _optimization_trajectories() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for arm in ("S1", "S2", "S2_L10"):
        for seed in SEEDS:
            for fold in FOLDS:
                path = (
                    CHECKPOINTS
                    / f"seed_{seed}"
                    / arm
                    / f"fold_{fold}"
                    / "selection.json"
                )
                if not path.is_file():
                    raise FileNotFoundError(f"missing new-arm selection: {path}")
                selection = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(selection, dict):
                    raise ValueError(f"selection is not an object: {path}")
                rows.extend(
                    _trajectory_rows_from_selection(
                        selection, arm=arm, seed=seed, fold=fold
                    )
                )
    frame = pd.DataFrame(rows)
    identities = set(
        zip(
            frame["arm"].astype(str),
            frame["seed_base"].astype(int),
            frame["fold"].astype(int),
            strict=True,
        )
    )
    expected = {
        (arm, seed, fold)
        for arm in ("S1", "S2", "S2_L10")
        for seed in SEEDS
        for fold in FOLDS
    }
    if identities != expected or int(frame["is_selected_epoch"].sum()) != 30:
        raise ValueError("optimization trajectory matrix coverage drifted")
    return frame.sort_values(["arm", "seed_base", "fold", "epoch"]).reset_index(
        drop=True
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration-lock-sha256", required=True)
    args = parser.parse_args()
    preregistration = verify_preregistration(EXPERIMENT_ROOT)
    require_lock_sha256(preregistration["lock_sha256"], args.preregistration_lock_sha256)
    outputs = {
        "all": METRICS / "representation_metrics.csv",
        "static": METRICS / "table_static_ftv.csv",
        "delta": METRICS / "table_observed_delta_ftv.csv",
        "sph": METRICS / "table_sph_and_residual.csv",
        "partial": METRICS / "table_partial_correlations.csv",
        "redundancy": METRICS / "table_state_redundancy.csv",
        "seed_consistency": METRICS / "table_seed_consistency.csv",
        "safety": METRICS / "optimization_safety.csv",
        "trajectories": METRICS / "optimization_trajectories.csv",
        "effects": METRICS / "representation_effects.json",
    }
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError("refusing to overwrite representation aggregate outputs")
    predictions = _load_predictions()
    diagnostic_folds = _load_state_diagnostics()
    rows: list[dict[str, object]] = []
    grouped = predictions.groupby(["arm", "seed_base", "task", "endpoint"], sort=True)
    for identity, group in grouped:
        if set(group["fold"].astype(int)) != set(FOLDS):
            raise ValueError(f"five-fold coverage drifted: {identity}")
        if group["patient_id"].astype(str).duplicated().any() or len(group) != 375:
            raise ValueError(f"pooled OOF patient coverage drifted: {identity}")
        rows.extend(_aggregate_probe_group(group))
    metrics = pd.DataFrame(rows)
    metrics = pd.concat([metrics, pd.DataFrame(_macro_rows(metrics))], ignore_index=True)
    state_diagnostics = _aggregate_state_diagnostics(diagnostic_folds)
    seed_consistency = _seed_consistency_rows(metrics, state_diagnostics)

    partial_rows: list[dict[str, object]] = []
    raw = predictions.loc[predictions["task"].eq("raw_sph")]
    for (arm, seed, endpoint), group in raw.groupby(["arm", "seed_base", "endpoint"], sort=True):
        result = partial_correlations_controlling_ftv(
            group["y_pred_natural"].to_numpy(dtype=np.float64),
            group["y_true_natural"].to_numpy(dtype=np.float64),
            group["ftv_control"].to_numpy(dtype=np.float64),
        )
        partial_rows.append(
            {"arm": arm, "seed_base": int(seed), "endpoint": endpoint, **result}
        )
    partial = pd.DataFrame(partial_rows)

    def seed_map(task: str, endpoint: str, *, arms: tuple[str, ...]) -> dict[str, dict[int, float]]:
        source = metrics.loc[
            metrics["task"].eq(task)
            & metrics["endpoint"].eq(endpoint)
            & metrics["space"].eq("residual" if task == "sph_res" else "natural")
        ]
        return {
            arm: {
                seed: float(
                    source.loc[source["arm"].eq(arm) & source["seed_base"].eq(seed), "spearman"].iloc[0]
                )
                for seed in SEEDS
            }
            for arm in arms
        }

    effects = seed_level_effects(
        static_ftv_macro_spearman=seed_map("static_ftv", "macro", arms=("S0", "S2")),
        delta_ftv_macro_spearman=seed_map("observed_delta_ftv", "macro", arms=("S0", "S2")),
        sph_res_t0_spearman=seed_map("sph_res", "T0", arms=("S0", "S1", "S2")),
    )
    safety, flags = _optimization_safety()
    trajectories = _optimization_trajectories()
    effects["optimization_safety"] = {
        "by_seed_fold": {str(seed): {str(fold): value for fold, value in folds.items()} for seed, folds in flags.items()},
        "pass_count": int(safety["optimization_safety_pass"].sum()),
        "total": len(safety),
    }
    effects["state_redundancy_diagnostics"] = {
        "artifact": "metrics/table_state_redundancy.csv",
        "direction": "scalar_phenotype_to_192D_response_state",
        "primary_metric": "state_variance_weighted_r2",
        "aggregation": (
            "outer_test_n_weighted_fold_metrics_in_fold_train_standardized_state_space"
        ),
        "cross_fold_state_vectors_pooled": False,
    }
    effects["seed_consistency"] = {
        "artifact": "metrics/table_seed_consistency.csv",
        "independent_unit": "training_seed",
        "seed_count": len(SEEDS),
    }
    effects["preregistration_lock_sha256"] = preregistration["lock_sha256"]
    METRICS.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(outputs["all"], index=False)
    metrics.loc[metrics["task"].eq("static_ftv")].to_csv(outputs["static"], index=False)
    metrics.loc[metrics["task"].eq("observed_delta_ftv")].to_csv(outputs["delta"], index=False)
    metrics.loc[metrics["task"].isin(("raw_sph", "sph_res"))].to_csv(outputs["sph"], index=False)
    partial.to_csv(outputs["partial"], index=False)
    state_diagnostics.to_csv(outputs["redundancy"], index=False)
    seed_consistency.to_csv(outputs["seed_consistency"], index=False)
    safety.to_csv(outputs["safety"], index=False)
    trajectories.to_csv(outputs["trajectories"], index=False)
    outputs["effects"].write_text(
        json.dumps(effects, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(effects, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

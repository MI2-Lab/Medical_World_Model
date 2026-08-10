"""Deterministic aggregation for the frozen spatial-pooling audit.

This module never fits a model.  Natural-scale metrics are recomputed only by
pooling the five immutable outer-test prediction files through the probe
adapter's ``pooled_oof_natural_metrics`` implementation.  Fold-specific
transformed metrics are summarized as fold statistics and are never pooled as
if their fold-specific transforms shared one scale.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .contracts import (
    ARMS,
    FOLDS,
    POOLINGS,
    REFERENCE_PROBE_ROOT,
    SEEDS,
    TIMEPOINTS,
    TRANSITIONS,
    cell_key,
    file_sha256,
)
from .pooling import expected_feature_shape
from .probes import (
    CALIBRATION_NAMES,
    METRIC_NAMES,
    pooled_oof_natural_metrics,
)
from .sidecars import (
    NUISANCE_COLUMNS,
    OCCUPANCY_COLUMNS,
    assign_occupancy_quartiles,
)


PRIMARY_SCOPE = "primary_measurement_valid"
OBSERVABLE_SCOPE = "observable_only"
NUISANCE_SCOPE = "target_valid"
FTV_TARGET = "FTV"
SECONDARY_POOLING = "PLOCAL+PVALID_SECONDARY"

FINAL_CORE_POOLINGS = {
    "L1": ("P0", "PLOCAL", "PLOCAL+GLOBAL"),
    "L3": ("P0", "PLOCAL", "PLOCAL+GLOBAL"),
    "N1": tuple(POOLINGS),
    "N3": tuple(POOLINGS),
}
S3_CORE_POOLINGS = {
    "L1": ("P0", "PLOCAL"),
    "L3": ("P0", "PLOCAL"),
    "N1": ("P0", "PLOCAL", "PORACLE"),
    "N3": ("P0", "PLOCAL", "PORACLE"),
}
NUISANCE_POOLINGS = {
    "L1": ("P0", "PLOCAL"),
    "L3": ("P0", "PLOCAL"),
    "N1": ("P0", "PVALID", "PLOCAL"),
    "N3": ("P0", "PVALID", "PLOCAL"),
}
NUISANCE_TARGETS = tuple(NUISANCE_COLUMNS[2:])

PUBLIC_FILENAMES = {
    "table1": "table1_feature_map_contract.csv",
    "table2": "table2_static_ftv.csv",
    "table3": "table3_delta_ftv.csv",
    "table4": "table4_legacy_deficit_recovery.csv",
    "table5": "table5_nuisance_decodability.csv",
    "table6": "table6_occupancy_downsampling.csv",
    "gates": "prospective_gates.json",
}
PRIVATE_JOINED_FILENAME = "table6_joined_diagnostics.private.csv"

TABLE1_COLUMNS = (
    "stage",
    "analysis_role",
    "input_contract",
    "input_shape_zyx",
    "feature_channels",
    "feature_shape_zyx",
    "jump_input_voxels",
    "center_offset_input_voxels",
    "theoretical_receptive_field_input_voxels",
    "local_window_mm_xyz",
    "jump_x_mm_median",
    "jump_x_mm_q25",
    "jump_x_mm_q75",
    "jump_y_mm_median",
    "jump_y_mm_q25",
    "jump_y_mm_q75",
    "jump_z_mm_median",
    "jump_z_mm_q25",
    "jump_z_mm_q75",
    "spacing_basis",
)

FTV_TABLE_COLUMNS = (
    "stage",
    "analysis_role",
    "seed_base",
    "arm",
    "pooling",
    "endpoint",
    "analysis_scope",
    "availability",
    "status_reason",
    "feature_dim",
    "aggregation",
    "n_test",
    "spearman",
    "pearson",
    "natural_r2",
    "rmse",
    "mae",
    "b0_rmse",
    "rmse_gain_over_b0",
    "prediction_target_variance_ratio",
    "calibration_slope",
    "calibration_intercept",
    "calibration_mean_bias",
    "transformed_scale",
    "transformed_fold_count",
    "transformed_spearman_fold_mean",
    "transformed_spearman_fold_sd",
    "transformed_r2_fold_mean",
    "transformed_r2_fold_sd",
    "transformed_rmse_fold_mean",
    "transformed_mae_fold_mean",
)

TABLE4_COLUMNS = (
    "stage",
    "analysis_role",
    "seed_base",
    "new_arm",
    "matched_legacy_arm",
    "pooling",
    "legacy_p0_spearman",
    "new_p0_spearman",
    "legacy_deficit",
    "pooling_spearman",
    "absolute_gain_vs_new_p0",
    "recovery_ratio",
    "recovery_defined",
    "status_reason",
)

TABLE5_COLUMNS = (
    "stage",
    "seed_base",
    "arm",
    "pooling",
    "target_name",
    "endpoint",
    "availability",
    "status_reason",
    "feature_dim",
    "aggregation",
    "n_test",
    "spearman",
    "pearson",
    "natural_r2",
    "rmse",
    "mae",
    "standardized_scale",
    "standardized_fold_count",
    "standardized_spearman_fold_mean",
    "standardized_r2_fold_mean",
    "standardized_r2_fold_sd",
)

TABLE6_COLUMNS = (
    "analysis",
    "seed_base",
    "endpoint",
    "stratum",
    "n",
    "l1_spearman",
    "n1_spearman",
    "n1_minus_l1_spearman",
    "l1_mae",
    "n1_mae",
    "n1_minus_l1_mae",
    "mean_paired_abs_error_difference",
    "stratifier_error_difference_spearman",
    "status_reason",
)

PRIVATE_JOINED_COLUMNS = (
    "patient_id",
    "seed_base",
    "fold",
    "endpoint",
    "y_true",
    "l1_prediction",
    "n1_prediction",
    "l1_abs_error",
    "n1_abs_error",
    "paired_abs_error_difference",
    "lesion_occupancy",
    "occupancy_quartile",
    "max_resample_factor",
    "downsampling_bin",
)


@dataclass(frozen=True)
class ProbeStageData:
    stage: str
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    selections: pd.DataFrame
    identities: pd.DataFrame
    secondary_present: bool


@dataclass(frozen=True)
class AggregationResult:
    table1: pd.DataFrame
    table2: pd.DataFrame
    table3: pd.DataFrame
    table4: pd.DataFrame
    table5: pd.DataFrame
    table6: pd.DataFrame
    gates: Mapping[str, Any]
    private_joined: pd.DataFrame

    def public_tables(self) -> dict[str, pd.DataFrame]:
        return {
            "table1": self.table1,
            "table2": self.table2,
            "table3": self.table3,
            "table4": self.table4,
            "table5": self.table5,
            "table6": self.table6,
        }


def load_audit_config(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("audit config must be a JSON object")
    exact = {
        "seeds": list(SEEDS),
        "folds": list(FOLDS),
        "arms": list(ARMS),
        "primary_poolings": list(POOLINGS),
    }
    for field, expected in exact.items():
        if payload.get(field) != expected:
            raise ValueError(f"audit config matrix drifted at {field}")
    required = {
        "feature_contract",
        "secondary_s3_contract",
        "strong_oracle",
        "deployable_local",
        "padding_geometry",
        "downsampling_bins",
        "undertraining",
        "legacy_pvalid",
        "legacy_poracle",
    }
    if missing := sorted(required - set(payload)):
        raise ValueError(f"audit config is missing frozen fields: {missing}")
    if payload.get("secondary_poolings") != [SECONDARY_POOLING]:
        raise ValueError("secondary pooling config drifted")
    if payload.get("training_forbidden") is not True:
        raise ValueError("aggregation requires the frozen no-training contract")
    return payload


def _expected_cells(stage: str) -> set[tuple[int, str, int, str]]:
    poolings = FINAL_CORE_POOLINGS if stage == "final" else S3_CORE_POOLINGS
    if stage not in {"final", "s3"}:
        raise ValueError("probe stage must be final or s3")
    return {
        (seed, arm, fold, pooling)
        for seed in SEEDS
        for arm in ARMS
        for fold in FOLDS
        for pooling in poolings[arm]
    }


def _secondary_cells() -> set[tuple[int, str, int, str]]:
    return {
        (seed, arm, fold, SECONDARY_POOLING)
        for seed in SEEDS
        for arm in ("N1", "N3")
        for fold in FOLDS
    }


def _required_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    if missing := sorted(set(required) - set(frame.columns)):
        raise ValueError(f"{label} is missing required columns: {missing}")


def _identity_tuple(frame: pd.DataFrame, label: str) -> tuple[int, str, int, str, int]:
    required = ("seed_base", "arm", "fold", "pooling", "feature_dim")
    _required_columns(frame, required, label)
    values: list[Any] = []
    for column in required:
        unique = frame[column].drop_duplicates()
        if len(unique) != 1:
            raise ValueError(f"{label} contains multiple {column} identities")
        values.append(unique.iloc[0])
    return (
        int(values[0]),
        str(values[1]).upper(),
        int(values[2]),
        str(values[3]).upper(),
        int(values[4]),
    )


def _expected_feature_dim(stage: str, pooling: str) -> int:
    if stage == "s3":
        return 64
    return 384 if pooling in {"PLOCAL+GLOBAL", SECONDARY_POOLING} else 192


def _expected_selection_groups(
    stage: str, arm: str, pooling: str
) -> set[tuple[str, str, str, str]]:
    groups = {
        ("static", FTV_TARGET, endpoint, scope)
        for endpoint in TIMEPOINTS
        for scope in (PRIMARY_SCOPE, OBSERVABLE_SCOPE)
    }
    groups |= {
        ("delta", FTV_TARGET, endpoint, scope)
        for endpoint in TRANSITIONS
        for scope in (PRIMARY_SCOPE, OBSERVABLE_SCOPE)
    }
    if stage == "final" and pooling in NUISANCE_POOLINGS[arm]:
        groups |= {
            ("nuisance", target, endpoint, NUISANCE_SCOPE)
            for target in NUISANCE_TARGETS
            for endpoint in TIMEPOINTS
        }
    return groups


def _validate_probe_cell_content(
    stage: str,
    identity: tuple[int, str, int, str, int],
    selection: pd.DataFrame,
    prediction: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    _, arm, _, pooling, _ = identity
    group_columns = ("task", "target_name", "endpoint", "analysis_scope")
    for frame, label in (
        (selection, "selection"),
        (prediction, "prediction"),
        (metrics, "metrics"),
    ):
        _required_columns(frame, group_columns, f"probe {label}")
    expected = _expected_selection_groups(stage, arm, pooling)
    selected = set(map(tuple, selection.loc[:, group_columns].astype(str).to_numpy()))
    if selected != expected:
        missing = sorted(expected - selected)
        extra = sorted(selected - expected)
        raise ValueError(
            f"probe selection coverage drift for {identity}: missing={missing}, extra={extra}"
        )
    predicted = set(map(tuple, prediction.loc[:, group_columns].astype(str).to_numpy()))
    if predicted != expected:
        raise ValueError(f"probe prediction coverage drift for {identity}")
    metric_groups = set(
        map(
            tuple,
            metrics.loc[metrics["endpoint"].astype(str).ne("macro"), group_columns]
            .astype(str)
            .to_numpy(),
        )
    )
    if metric_groups != expected:
        raise ValueError(f"probe metric coverage drift for {identity}")

    selection_key = list(group_columns)
    prediction_key = [*group_columns, "patient_id"]
    metric_key = [*group_columns, "scale"]
    if selection.duplicated(selection_key).any():
        raise ValueError(f"duplicate Ridge selection in {identity}")
    if prediction.duplicated(prediction_key).any():
        raise ValueError(f"duplicate OOF prediction in {identity}")
    if metrics.duplicated(metric_key).any():
        raise ValueError(f"duplicate probe metric in {identity}")
    if not prediction["split"].astype(str).eq("test").all():
        raise ValueError("probe prediction file contains non-test rows")
    if not prediction["test_predict_call_count"].eq(1).all():
        raise ValueError("probe prediction file violates single-test-predict contract")


def load_probe_stage(root: str | Path, *, stage: str) -> ProbeStageData:
    """Discover a complete stage matrix and validate every sibling hash/schema."""

    stage = str(stage).lower()
    if stage not in {"final", "s3"}:
        raise ValueError("probe stage must be final or s3")
    source = Path(root).resolve()
    prediction_paths = sorted(source.rglob("ridge_predictions.private.csv"))
    if not prediction_paths:
        raise FileNotFoundError(f"no frozen {stage} probe outputs found under {source}")
    predictions: list[pd.DataFrame] = []
    metrics_rows: list[pd.DataFrame] = []
    selections: list[pd.DataFrame] = []
    identities: list[dict[str, Any]] = []
    observed_cells: set[tuple[int, str, int, str]] = set()

    for prediction_path in prediction_paths:
        directory = prediction_path.parent
        paths = {
            "ridge_selection.csv": directory / "ridge_selection.csv",
            "ridge_predictions.private.csv": prediction_path,
            "probe_metrics.csv": directory / "probe_metrics.csv",
        }
        metadata_path = directory / "probe_metadata.json"
        if not metadata_path.is_file() or any(not path.is_file() for path in paths.values()):
            raise FileNotFoundError(f"incomplete probe output set: {directory}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError(f"probe metadata is not an object: {metadata_path}")
        hashes = metadata.get("output_sha256")
        if not isinstance(hashes, Mapping):
            raise ValueError(f"probe metadata has no output hash inventory: {directory}")
        for name, path in paths.items():
            if hashes.get(name) != file_sha256(path):
                raise ValueError(f"probe output hash drift: {path}")

        selection = pd.read_csv(paths["ridge_selection.csv"])
        prediction = pd.read_csv(prediction_path)
        metrics = pd.read_csv(paths["probe_metrics.csv"])
        if selection.empty or prediction.empty or metrics.empty:
            raise ValueError(f"probe output may not be empty: {directory}")
        identity = _identity_tuple(prediction, "probe prediction")
        if _identity_tuple(selection, "probe selection") != identity:
            raise ValueError("probe selection/prediction identities disagree")
        if _identity_tuple(metrics, "probe metrics") != identity:
            raise ValueError("probe metric/prediction identities disagree")
        seed, arm, fold, pooling, feature_dim = identity
        for field, expected in (
            ("seed_base", seed),
            ("arm", arm),
            ("fold", fold),
            ("pooling", pooling),
            ("feature_dim", feature_dim),
        ):
            if field in metadata and metadata[field] != expected:
                raise ValueError(f"probe metadata identity drift at {field}: {directory}")
        if "stage" in metadata and str(metadata["stage"]).lower() != stage:
            raise ValueError(f"probe metadata stage drift: {directory}")
        for frame in (selection, prediction, metrics):
            if "stage" in frame.columns and not frame["stage"].astype(str).str.lower().eq(stage).all():
                raise ValueError(f"probe CSV stage drift: {directory}")
            frame["stage"] = stage
        if feature_dim != _expected_feature_dim(stage, pooling):
            raise ValueError(f"probe feature dimension drift for {identity}")
        cell = (seed, arm, fold, pooling)
        if cell in observed_cells:
            raise ValueError(f"duplicate probe identity discovered: {cell}")
        observed_cells.add(cell)
        _validate_probe_cell_content(stage, identity, selection, prediction, metrics)
        predictions.append(prediction)
        metrics_rows.append(metrics)
        selections.append(selection)
        identities.append(
            {
                "stage": stage,
                "seed_base": seed,
                "arm": arm,
                "fold": fold,
                "pooling": pooling,
                "feature_dim": feature_dim,
                "prediction_sha256": file_sha256(prediction_path),
                "metrics_sha256": file_sha256(paths["probe_metrics.csv"]),
                "selection_sha256": file_sha256(paths["ridge_selection.csv"]),
                "metadata_sha256": file_sha256(metadata_path),
            }
        )

    core = _expected_cells(stage)
    allowed = set(core)
    secondary = _secondary_cells() if stage == "final" else set()
    allowed |= secondary
    if missing := sorted(core - observed_cells):
        raise ValueError(f"{stage} probe matrix is incomplete: {missing}")
    if extra := sorted(observed_cells - allowed):
        raise ValueError(f"{stage} probe matrix has unexpected identities: {extra}")
    secondary_observed = observed_cells & secondary
    if secondary_observed and secondary_observed != secondary:
        raise ValueError("optional final secondary pooling matrix is only partially complete")
    return ProbeStageData(
        stage=stage,
        predictions=pd.concat(predictions, ignore_index=True),
        metrics=pd.concat(metrics_rows, ignore_index=True),
        selections=pd.concat(selections, ignore_index=True),
        identities=pd.DataFrame(identities),
        secondary_present=bool(secondary_observed),
    )


def pooled_stage_natural_metrics(data: ProbeStageData) -> pd.DataFrame:
    result = pooled_oof_natural_metrics(data.predictions, expected_folds=FOLDS)
    result.insert(0, "stage", data.stage)
    return result


def transformed_fold_summaries(data: ProbeStageData) -> pd.DataFrame:
    metrics = data.metrics.loc[data.metrics["scale"].astype(str).ne("natural")].copy()
    keys = (
        "stage",
        "seed_base",
        "arm",
        "pooling",
        "feature_dim",
        "task",
        "target_name",
        "endpoint",
        "analysis_scope",
        "target_semantics",
        "scale",
    )
    _required_columns(metrics, (*keys, "fold", *METRIC_NAMES), "transformed metrics")
    if metrics.duplicated([*keys, "fold"]).any():
        raise ValueError("transformed metrics contain duplicate outer-fold rows")
    rows: list[dict[str, Any]] = []
    for values, group in metrics.groupby(list(keys), sort=False, dropna=False):
        if set(group["fold"].astype(int)) != set(FOLDS):
            raise ValueError(f"transformed metric group misses an outer fold: {values}")
        rows.append(
            {
                **dict(zip(keys, values, strict=True)),
                "transformed_fold_count": int(len(group)),
                "transformed_spearman_fold_mean": float(group["spearman"].mean()),
                "transformed_spearman_fold_sd": float(group["spearman"].std(ddof=1)),
                "transformed_r2_fold_mean": float(group["r2"].mean()),
                "transformed_r2_fold_sd": float(group["r2"].std(ddof=1)),
                "transformed_rmse_fold_mean": float(group["rmse"].mean()),
                "transformed_mae_fold_mean": float(group["mae"].mean()),
            }
        )
    result = pd.DataFrame(rows)
    identity = [key for key in keys if key != "scale"]
    if result.duplicated(identity).any():
        raise ValueError("a result group has multiple transformed/standardized scales")
    return result


def _stage_poolings(
    stage: str, *, secondary_present: bool
) -> Mapping[str, Sequence[str]]:
    if stage == "s3":
        return S3_CORE_POOLINGS
    if not secondary_present:
        return FINAL_CORE_POOLINGS
    return {
        arm: (*FINAL_CORE_POOLINGS[arm], SECONDARY_POOLING)
        if arm in {"N1", "N3"}
        else FINAL_CORE_POOLINGS[arm]
        for arm in ARMS
    }


def build_ftv_table(
    data: ProbeStageData,
    natural: pd.DataFrame,
    transformed: pd.DataFrame,
    *,
    task: str,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    endpoints = TIMEPOINTS if task == "static" else TRANSITIONS
    endpoint_order = (*endpoints, "macro")
    rows = natural.loc[
        natural["task"].astype(str).eq(task)
        & natural["target_name"].astype(str).eq(FTV_TARGET)
        & natural["analysis_scope"].astype(str).eq(PRIMARY_SCOPE)
    ].copy()
    poolings = _stage_poolings(data.stage, secondary_present=data.secondary_present)
    expected = {
        (seed, arm, pooling, endpoint)
        for seed in SEEDS
        for arm in ARMS
        for pooling in poolings[arm]
        for endpoint in endpoint_order
    }
    observed = set(
        zip(
            rows["seed_base"].astype(int),
            rows["arm"].astype(str),
            rows["pooling"].astype(str),
            rows["endpoint"].astype(str),
        )
    )
    if observed != expected:
        raise ValueError(
            f"{data.stage}/{task} pooled natural matrix drift: "
            f"missing={sorted(expected-observed)}, extra={sorted(observed-expected)}"
        )
    join_keys = [
        "stage",
        "seed_base",
        "arm",
        "pooling",
        "feature_dim",
        "task",
        "target_name",
        "endpoint",
        "analysis_scope",
        "target_semantics",
    ]
    transformed_selected = transformed.loc[
        transformed["task"].astype(str).eq(task)
        & transformed["target_name"].astype(str).eq(FTV_TARGET)
        & transformed["analysis_scope"].astype(str).eq(PRIMARY_SCOPE)
    ].copy()
    merged = rows.merge(
        transformed_selected,
        on=join_keys,
        how="left",
        validate="one_to_one",
        indicator=True,
        suffixes=("", "_fold"),
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError(f"{data.stage}/{task} transformed fold summary is incomplete")
    merged = merged.drop(columns="_merge")
    merged["analysis_role"] = np.where(
        merged["pooling"].eq(SECONDARY_POOLING),
        "secondary_sensitivity",
        "conditional_s3" if data.stage == "s3" else "primary",
    )
    merged["availability"] = "AVAILABLE"
    merged["status_reason"] = ""
    merged["natural_r2"] = merged["r2"]
    merged["transformed_scale"] = merged["scale_fold"]

    unavailable: list[dict[str, Any]] = []
    unavailable_poolings = (
        (("PVALID", str(config["legacy_pvalid"])), ("PORACLE", str(config["legacy_poracle"])))
        if data.stage == "final"
        else (("PORACLE", str(config["legacy_poracle"])),)
    )
    for seed in SEEDS:
        for arm in ("L1", "L3"):
            for pooling, reason in unavailable_poolings:
                for endpoint in endpoint_order:
                    unavailable.append(
                        {
                            "stage": data.stage,
                            "analysis_role": "conditional_s3" if data.stage == "s3" else "primary",
                            "seed_base": seed,
                            "arm": arm,
                            "pooling": pooling,
                            "endpoint": endpoint,
                            "analysis_scope": PRIMARY_SCOPE,
                            "availability": "NA",
                            "status_reason": reason,
                        }
                    )
    result = pd.concat([merged, pd.DataFrame(unavailable)], ignore_index=True, sort=False)
    for column in FTV_TABLE_COLUMNS:
        if column not in result:
            result[column] = np.nan
    result = result.loc[:, FTV_TABLE_COLUMNS]
    if result.duplicated(["stage", "seed_base", "arm", "pooling", "endpoint"]).any():
        raise ValueError("FTV public table contains duplicate rectangular cells")
    order = {value: index for index, value in enumerate(endpoint_order)}
    result["_endpoint_order"] = result["endpoint"].map(order)
    result = result.sort_values(
        ["stage", "seed_base", "arm", "pooling", "_endpoint_order"]
    ).drop(columns="_endpoint_order")
    return result.reset_index(drop=True)


def build_recovery_table(table2: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stage in sorted(table2["stage"].dropna().astype(str).unique()):
        poolings = tuple(POOLINGS) if stage == "final" else ("P0", "PLOCAL", "PORACLE")
        available = table2.loc[
            table2["stage"].eq(stage)
            & table2["endpoint"].eq("macro")
            & table2["availability"].eq("AVAILABLE")
        ]
        for seed in SEEDS:
            for new_arm, legacy_arm, role in (
                ("N1", "L1", "primary"),
                ("N3", "L3", "secondary_replication"),
            ):
                def metric(arm: str, pooling: str) -> float:
                    selected = available.loc[
                        available["seed_base"].eq(seed)
                        & available["arm"].eq(arm)
                        & available["pooling"].eq(pooling),
                        "spearman",
                    ]
                    if len(selected) != 1 or not np.isfinite(float(selected.iloc[0])):
                        raise ValueError(
                            f"gate-relevant static macro Spearman is missing: "
                            f"{stage}/{seed}/{arm}/{pooling}"
                        )
                    return float(selected.iloc[0])

                legacy_p0 = metric(legacy_arm, "P0")
                new_p0 = metric(new_arm, "P0")
                deficit = legacy_p0 - new_p0
                for pooling in poolings:
                    pooled = metric(new_arm, pooling)
                    gain = pooled - new_p0
                    defined = deficit > 0
                    rows.append(
                        {
                            "stage": stage,
                            "analysis_role": role,
                            "seed_base": seed,
                            "new_arm": new_arm,
                            "matched_legacy_arm": legacy_arm,
                            "pooling": pooling,
                            "legacy_p0_spearman": legacy_p0,
                            "new_p0_spearman": new_p0,
                            "legacy_deficit": deficit,
                            "pooling_spearman": pooled,
                            "absolute_gain_vs_new_p0": gain,
                            "recovery_ratio": gain / deficit if defined else math.nan,
                            "recovery_defined": defined,
                            "status_reason": "" if defined else "NA_nonpositive_legacy_deficit",
                        }
                    )
    result = pd.DataFrame(rows, columns=TABLE4_COLUMNS)
    if result.duplicated(["stage", "seed_base", "new_arm", "pooling"]).any():
        raise ValueError("recovery table contains duplicate cells")
    return result.sort_values(
        ["stage", "seed_base", "new_arm", "pooling"]
    ).reset_index(drop=True)


def build_nuisance_table(
    data: ProbeStageData,
    natural: pd.DataFrame,
    transformed: pd.DataFrame,
    *,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    if data.stage != "final":
        raise ValueError("nuisance decodability is preregistered for final states only")
    rows = natural.loc[
        natural["task"].astype(str).eq("nuisance")
        & natural["analysis_scope"].astype(str).eq(NUISANCE_SCOPE)
    ].copy()
    endpoint_order = (*TIMEPOINTS, "macro")
    expected = {
        (seed, arm, pooling, target, endpoint)
        for seed in SEEDS
        for arm in ARMS
        for pooling in NUISANCE_POOLINGS[arm]
        for target in NUISANCE_TARGETS
        for endpoint in endpoint_order
    }
    observed = set(
        zip(
            rows["seed_base"].astype(int),
            rows["arm"].astype(str),
            rows["pooling"].astype(str),
            rows["target_name"].astype(str),
            rows["endpoint"].astype(str),
        )
    )
    if observed != expected:
        raise ValueError("nuisance pooled OOF matrix is incomplete or contains extras")
    join_keys = [
        "stage",
        "seed_base",
        "arm",
        "pooling",
        "feature_dim",
        "task",
        "target_name",
        "endpoint",
        "analysis_scope",
        "target_semantics",
    ]
    transformed_rows = transformed.loc[
        transformed["task"].astype(str).eq("nuisance")
        & transformed["analysis_scope"].astype(str).eq(NUISANCE_SCOPE)
    ]
    merged = rows.merge(
        transformed_rows,
        on=join_keys,
        how="left",
        validate="one_to_one",
        indicator=True,
        suffixes=("", "_fold"),
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("nuisance standardized fold summaries are incomplete")
    merged["availability"] = "AVAILABLE"
    merged["status_reason"] = ""
    merged["natural_r2"] = merged["r2"]
    merged["standardized_scale"] = merged["scale_fold"]
    merged["standardized_fold_count"] = merged["transformed_fold_count"]
    merged["standardized_spearman_fold_mean"] = merged[
        "transformed_spearman_fold_mean"
    ]
    merged["standardized_r2_fold_mean"] = merged["transformed_r2_fold_mean"]
    merged["standardized_r2_fold_sd"] = merged["transformed_r2_fold_sd"]

    unavailable = [
        {
            "stage": "final",
            "seed_base": seed,
            "arm": arm,
            "pooling": "PVALID",
            "target_name": target,
            "endpoint": endpoint,
            "availability": "NA",
            "status_reason": str(config["legacy_pvalid"]),
        }
        for seed in SEEDS
        for arm in ("L1", "L3")
        for target in NUISANCE_TARGETS
        for endpoint in endpoint_order
    ]
    result = pd.concat([merged, pd.DataFrame(unavailable)], ignore_index=True, sort=False)
    for column in TABLE5_COLUMNS:
        if column not in result:
            result[column] = np.nan
    result = result.loc[:, TABLE5_COLUMNS]
    if result.duplicated(
        ["stage", "seed_base", "arm", "pooling", "target_name", "endpoint"]
    ).any():
        raise ValueError("nuisance table contains duplicate rectangular cells")
    return result.sort_values(
        ["seed_base", "arm", "pooling", "target_name", "endpoint"]
    ).reset_index(drop=True)


def build_feature_contract_table(
    nuisance: pd.DataFrame, config: Mapping[str, Any]
) -> pd.DataFrame:
    _required_columns(nuisance, NUISANCE_COLUMNS, "nuisance sidecar")
    if tuple(nuisance.columns) != NUISANCE_COLUMNS:
        raise ValueError("nuisance sidecar contains non-allow-listed columns")
    if len(nuisance) != 808 * len(TIMEPOINTS):
        raise ValueError("nuisance sidecar is not the frozen 808x4 population")
    if nuisance.duplicated(["patient_id", "visit"]).any():
        raise ValueError("nuisance sidecar has duplicate patient/visit rows")
    rows: list[dict[str, Any]] = []
    stage_specs = {
        "final": {
            "channels": int(config["feature_contract"]["channels"]),
            "stride": int(config["feature_contract"]["jump_input_voxels"]),
            "offset": int(config["feature_contract"]["center_offset_input_voxels"]),
            "rf": int(
                config["feature_contract"]["theoretical_receptive_field_input_voxels"]
            ),
            "role": "primary",
        },
        "s3": {
            "channels": int(config["secondary_s3_contract"]["channels"]),
            "stride": int(config["secondary_s3_contract"]["jump_input_voxels"]),
            "offset": int(config["secondary_s3_contract"]["center_offset_input_voxels"]),
            "rf": int(
                config["secondary_s3_contract"][
                    "theoretical_receptive_field_input_voxels"
                ]
            ),
            "role": "conditional_secondary",
        },
    }
    for stage, spec in stage_specs.items():
        for contract, input_shape, spacing_basis in (
            ("legacy", (32, 96, 96), "visit_native_spacing_pooled_distribution"),
            ("c1b", (112, 176, 160), "frozen_0.9x0.9x2.0_mm"),
        ):
            feature_shape = expected_feature_shape(input_shape, stage=stage)
            if contract == "legacy":
                spacings = nuisance[
                    [
                        "native_spacing_x_mm",
                        "native_spacing_y_mm",
                        "native_spacing_z_mm",
                    ]
                ].to_numpy(dtype=np.float64)
            else:
                spacings = np.broadcast_to(
                    np.asarray((0.9, 0.9, 2.0), dtype=np.float64),
                    (len(nuisance), 3),
                )
            jumps = spacings * spec["stride"]
            rows.append(
                {
                    "stage": stage,
                    "analysis_role": spec["role"],
                    "input_contract": contract,
                    "input_shape_zyx": "x".join(map(str, input_shape)),
                    "feature_channels": spec["channels"],
                    "feature_shape_zyx": "x".join(map(str, feature_shape)),
                    "jump_input_voxels": spec["stride"],
                    "center_offset_input_voxels": spec["offset"],
                    "theoretical_receptive_field_input_voxels": spec["rf"],
                    "local_window_mm_xyz": "64x64x64",
                    "jump_x_mm_median": float(np.median(jumps[:, 0])),
                    "jump_x_mm_q25": float(np.quantile(jumps[:, 0], 0.25)),
                    "jump_x_mm_q75": float(np.quantile(jumps[:, 0], 0.75)),
                    "jump_y_mm_median": float(np.median(jumps[:, 1])),
                    "jump_y_mm_q25": float(np.quantile(jumps[:, 1], 0.25)),
                    "jump_y_mm_q75": float(np.quantile(jumps[:, 1], 0.75)),
                    "jump_z_mm_median": float(np.median(jumps[:, 2])),
                    "jump_z_mm_q25": float(np.quantile(jumps[:, 2], 0.25)),
                    "jump_z_mm_q75": float(np.quantile(jumps[:, 2], 0.75)),
                    "spacing_basis": spacing_basis,
                }
            )
    result = pd.DataFrame(rows, columns=TABLE1_COLUMNS)
    return result.sort_values(["stage", "input_contract"]).reset_index(drop=True)


def load_frozen_p0_predictions(
    *,
    prediction_root: str | Path = REFERENCE_PROBE_ROOT,
    preregistration_lock: str | Path,
) -> pd.DataFrame:
    """Load only immutable old L1/N1 P0 OOF rows; never refit a probe."""

    lock = json.loads(Path(preregistration_lock).read_text(encoding="utf-8"))
    references = lock.get("formal_p0_references")
    if not isinstance(references, Mapping) or len(references) != 40:
        raise ValueError("preregistration has no exact 40-cell old P0 inventory")
    root = Path(prediction_root).resolve()
    frames: list[pd.DataFrame] = []
    for seed in SEEDS:
        for arm in ("L1", "N1"):
            for fold in FOLDS:
                key = cell_key(seed, arm, fold)
                record = references.get(key)
                if not isinstance(record, Mapping):
                    raise ValueError(f"old P0 inventory is missing {key}")
                hashes = record.get("probe_outputs_sha256")
                if not isinstance(hashes, Mapping):
                    raise ValueError(f"old P0 probe hash inventory is missing {key}")
                path = root / key / "ridge_predictions.private.csv"
                if hashes.get(path.name) != file_sha256(path):
                    raise ValueError(f"immutable old P0 prediction drift at {key}")
                frame = pd.read_csv(path)
                _required_columns(
                    frame,
                    (
                        "patient_id",
                        "task",
                        "endpoint",
                        "analysis_scope",
                        "split",
                        "y_true",
                        "y_pred",
                        "test_predict_call_count",
                        "arm",
                        "seed_base",
                        "fold",
                    ),
                    "old P0 predictions",
                )
                if (
                    set(frame["arm"].astype(str)) != {arm}
                    or set(frame["seed_base"].astype(int)) != {seed}
                    or set(frame["fold"].astype(int)) != {fold}
                ):
                    raise ValueError(f"old P0 prediction identity drift at {key}")
                frames.append(
                    frame.loc[
                        frame["task"].astype(str).eq("static")
                        & frame["analysis_scope"].astype(str).eq(PRIMARY_SCOPE)
                        & frame["split"].astype(str).eq("test")
                    ].copy()
                )
    result = pd.concat(frames, ignore_index=True)
    if not result["test_predict_call_count"].eq(1).all():
        raise ValueError("old P0 rows violate single test prediction")
    if result.duplicated(["seed_base", "arm", "endpoint", "patient_id"]).any():
        raise ValueError("old P0 OOF rows contain a cross-fold duplicate patient")
    for keys, group in result.groupby(["seed_base", "arm", "endpoint"]):
        if set(group["fold"].astype(int)) != set(FOLDS):
            raise ValueError(f"old P0 OOF group misses a fold: {keys}")
    expected_groups = {
        (seed, arm, endpoint)
        for seed in SEEDS
        for arm in ("L1", "N1")
        for endpoint in TIMEPOINTS
    }
    observed_groups = set(
        zip(
            result["seed_base"].astype(int),
            result["arm"].astype(str),
            result["endpoint"].astype(str),
        )
    )
    if observed_groups != expected_groups:
        raise ValueError("old P0 static OOF endpoint matrix drifted")
    return result


def pair_frozen_p0_errors(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = ["seed_base", "fold", "endpoint", "patient_id"]
    legacy = predictions.loc[predictions["arm"].eq("L1"), [*keys, "y_true", "y_pred"]]
    c1b = predictions.loc[predictions["arm"].eq("N1"), [*keys, "y_true", "y_pred"]]
    paired = legacy.merge(
        c1b,
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_l1", "_n1"),
    )
    if not paired["_merge"].eq("both").all():
        raise ValueError("old L1/N1 P0 OOF populations do not pair exactly")
    if not np.allclose(
        paired["y_true_l1"], paired["y_true_n1"], rtol=0, atol=1e-12
    ):
        raise ValueError("paired old P0 rows disagree on natural target")
    result = paired.loc[:, keys].copy()
    result["y_true"] = paired["y_true_l1"].astype(float)
    result["l1_prediction"] = paired["y_pred_l1"].astype(float)
    result["n1_prediction"] = paired["y_pred_n1"].astype(float)
    result["l1_abs_error"] = np.abs(result["l1_prediction"] - result["y_true"])
    result["n1_abs_error"] = np.abs(result["n1_prediction"] - result["y_true"])
    result["paired_abs_error_difference"] = (
        result["n1_abs_error"] - result["l1_abs_error"]
    )
    return result


def build_private_diagnostic_join(
    paired: pd.DataFrame,
    occupancy: pd.DataFrame,
    nuisance: pd.DataFrame,
    *,
    downsampling_bins: Sequence[str],
) -> pd.DataFrame:
    _required_columns(occupancy, OCCUPANCY_COLUMNS, "occupancy sidecar")
    if tuple(occupancy.columns) != OCCUPANCY_COLUMNS:
        raise ValueError("occupancy sidecar schema drift")
    if len(occupancy) != 1500 or occupancy.duplicated(["patient_id", "visit"]).any():
        raise ValueError("occupancy sidecar is not the exact 1500-visit cohort")
    checked = assign_occupancy_quartiles(
        occupancy.drop(columns="occupancy_quartile")
    )
    if not checked["occupancy_quartile"].astype(str).equals(
        occupancy["occupancy_quartile"].astype(str)
    ):
        raise ValueError("stored occupancy quartile disagrees with frozen pooled qcut")
    _required_columns(nuisance, NUISANCE_COLUMNS, "nuisance sidecar")
    if tuple(nuisance.columns) != NUISANCE_COLUMNS:
        raise ValueError("nuisance sidecar schema drift")
    if len(nuisance) != 808 * 4 or nuisance.duplicated(["patient_id", "visit"]).any():
        raise ValueError("nuisance sidecar is not the exact 808x4 population")

    occupancy_join = occupancy[
        ["patient_id", "visit", "lesion_occupancy", "occupancy_quartile"]
    ].rename(columns={"visit": "endpoint"})
    nuisance_join = nuisance[
        ["patient_id", "visit", "max_resample_factor"]
    ].rename(columns={"visit": "endpoint"})
    result = paired.merge(
        occupancy_join,
        on=["patient_id", "endpoint"],
        how="left",
        validate="many_to_one",
        indicator="_occupancy_merge",
    ).merge(
        nuisance_join,
        on=["patient_id", "endpoint"],
        how="left",
        validate="many_to_one",
        indicator="_nuisance_merge",
    )
    if not result["_occupancy_merge"].eq("both").all():
        raise ValueError("old P0 OOF rows are not fully covered by occupancy sidecar")
    if not result["_nuisance_merge"].eq("both").all():
        raise ValueError("old P0 OOF rows are not fully covered by nuisance sidecar")
    labels = tuple(str(value) for value in downsampling_bins)
    if labels != ("<=1.5", "(1.5,2]", ">2"):
        raise ValueError("downsampling bin labels drifted from preregistration")
    factors = result["max_resample_factor"].to_numpy(dtype=np.float64)
    if not np.isfinite(factors).all() or np.any(factors <= 0):
        raise ValueError("max_resample_factor must be finite and positive")
    downsampling = pd.cut(
        factors,
        bins=(-np.inf, 1.5, 2.0, np.inf),
        labels=labels,
        right=True,
        include_lowest=True,
    )
    if downsampling.isna().any():
        raise ValueError("downsampling bin assignment produced missing rows")
    result["downsampling_bin"] = downsampling.astype(str)
    if set(result["downsampling_bin"].unique()) - set(labels):
        raise ValueError("downsampling bin assignment produced an unknown label")
    result = result.drop(columns=["_occupancy_merge", "_nuisance_merge"])
    result = result.loc[:, PRIVATE_JOINED_COLUMNS]
    if result.duplicated(["seed_base", "endpoint", "patient_id"]).any():
        raise ValueError("private diagnostic join contains duplicate OOF rows")
    return result.sort_values(
        ["seed_base", "endpoint", "patient_id"]
    ).reset_index(drop=True)


def _safe_spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if (
        len(x_array) < 2
        or not np.isfinite(x_array).all()
        or not np.isfinite(y_array).all()
        or np.ptp(x_array) == 0
        or np.ptp(y_array) == 0
    ):
        return math.nan
    value = float(spearmanr(x_array, y_array).statistic)
    return value if math.isfinite(value) else math.nan


def _diagnostic_metric_row(
    group: pd.DataFrame,
    *,
    analysis: str,
    seed: int,
    endpoint: str,
    stratum: str,
    stratifier: str | None = None,
) -> dict[str, Any]:
    n = int(len(group))
    if n == 0:
        return {
            "analysis": analysis,
            "seed_base": seed,
            "endpoint": endpoint,
            "stratum": stratum,
            "n": 0,
            "status_reason": "NA_empty_stratum",
        }
    l1_spearman = _safe_spearman(group["y_true"], group["l1_prediction"])
    n1_spearman = _safe_spearman(group["y_true"], group["n1_prediction"])
    correlation = (
        math.nan
        if stratifier is None
        else _safe_spearman(group[stratifier], group["paired_abs_error_difference"])
    )
    return {
        "analysis": analysis,
        "seed_base": seed,
        "endpoint": endpoint,
        "stratum": stratum,
        "n": n,
        "l1_spearman": l1_spearman,
        "n1_spearman": n1_spearman,
        "n1_minus_l1_spearman": n1_spearman - l1_spearman,
        "l1_mae": float(group["l1_abs_error"].mean()),
        "n1_mae": float(group["n1_abs_error"].mean()),
        "n1_minus_l1_mae": float(
            group["n1_abs_error"].mean() - group["l1_abs_error"].mean()
        ),
        "mean_paired_abs_error_difference": float(
            group["paired_abs_error_difference"].mean()
        ),
        "stratifier_error_difference_spearman": correlation,
        "status_reason": "",
    }


def build_table6(
    joined: pd.DataFrame, *, downsampling_bins: Sequence[str]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for endpoint in TIMEPOINTS:
            group = joined.loc[
                joined["seed_base"].eq(seed) & joined["endpoint"].eq(endpoint)
            ]
            if group.empty:
                raise ValueError(f"private diagnostic rows missing {seed}/{endpoint}")
            for quartile in ("Q1", "Q2", "Q3", "Q4"):
                rows.append(
                    _diagnostic_metric_row(
                        group.loc[group["occupancy_quartile"].eq(quartile)],
                        analysis="occupancy_quartile",
                        seed=seed,
                        endpoint=endpoint,
                        stratum=quartile,
                    )
                )
            rows.append(
                _diagnostic_metric_row(
                    group,
                    analysis="occupancy_correlation",
                    seed=seed,
                    endpoint=endpoint,
                    stratum="ALL",
                    stratifier="lesion_occupancy",
                )
            )
            for label in downsampling_bins:
                rows.append(
                    _diagnostic_metric_row(
                        group.loc[group["downsampling_bin"].eq(str(label))],
                        analysis="downsampling_bin",
                        seed=seed,
                        endpoint=endpoint,
                        stratum=str(label),
                    )
                )
            rows.append(
                _diagnostic_metric_row(
                    group,
                    analysis="downsampling_correlation",
                    seed=seed,
                    endpoint=endpoint,
                    stratum="ALL",
                    stratifier="max_resample_factor",
                )
            )
    result = pd.DataFrame(rows)
    for column in TABLE6_COLUMNS:
        if column not in result:
            result[column] = np.nan
    result = result.loc[:, TABLE6_COLUMNS]
    if result.duplicated(["analysis", "seed_base", "endpoint", "stratum"]).any():
        raise ValueError("Table 6 contains duplicate aggregate rows")
    return result


def _gate_recovery_rows(
    table4: pd.DataFrame, *, stage: str, pooling: str
) -> pd.DataFrame:
    rows = table4.loc[
        table4["stage"].eq(stage)
        & table4["new_arm"].eq("N1")
        & table4["pooling"].eq(pooling)
    ].sort_values("seed_base")
    if list(rows["seed_base"].astype(int)) != list(SEEDS):
        raise ValueError(f"gate recovery rows are incomplete for {stage}/{pooling}")
    return rows


def _at_least(value: float, threshold: float) -> bool:
    """Inclusive prospective threshold robust to one subtraction roundoff."""

    return bool(np.isfinite(value) and value >= float(threshold) - 1e-12)


def _at_most(value: float, threshold: float) -> bool:
    return bool(np.isfinite(value) and value <= float(threshold) + 1e-12)


def strong_oracle_gate(
    table4: pd.DataFrame, *, stage: str, config: Mapping[str, Any]
) -> dict[str, Any]:
    thresholds = config["strong_oracle"]
    rows = _gate_recovery_rows(table4, stage=stage, pooling="PORACLE")
    per_seed: dict[str, Any] = {}
    for row in rows.itertuples(index=False):
        gain = float(row.absolute_gain_vs_new_p0)
        recovery = float(row.recovery_ratio)
        passed = bool(
            bool(row.recovery_defined)
            and np.isfinite(gain)
            and np.isfinite(recovery)
            and _at_least(gain, thresholds["minimum_spearman_gain_each_seed"])
            and _at_least(recovery, thresholds["minimum_recovery_each_seed"])
        )
        per_seed[str(int(row.seed_base))] = {
            "absolute_spearman_gain": gain,
            "recovery_ratio": recovery if np.isfinite(recovery) else None,
            "recovery_defined": bool(row.recovery_defined),
            "pass": passed,
        }
    supported = all(value["pass"] for value in per_seed.values())
    return {
        "status": "SUPPORTED_IN_PILOT" if supported else "NOT_SUPPORTED_IN_PILOT",
        "supported": supported,
        "thresholds": dict(thresholds),
        "per_seed": per_seed,
    }


def deployable_local_gate(
    table4: pd.DataFrame, *, stage: str, config: Mapping[str, Any]
) -> dict[str, Any]:
    thresholds = config["deployable_local"]
    methods = ("PLOCAL", "PLOCAL+GLOBAL") if stage == "final" else ("PLOCAL",)
    evidence: dict[str, Any] = {}
    qualifying: list[str] = []
    for method in methods:
        rows = _gate_recovery_rows(table4, stage=stage, pooling=method)
        per_seed: dict[str, Any] = {}
        for row in rows.itertuples(index=False):
            gain = float(row.absolute_gain_vs_new_p0)
            recovery = float(row.recovery_ratio)
            passed = bool(
                _at_least(gain, thresholds["minimum_spearman_gain_each_seed"])
                or (
                    bool(row.recovery_defined)
                    and np.isfinite(recovery)
                    and _at_least(recovery, thresholds["minimum_recovery_each_seed"])
                )
            )
            per_seed[str(int(row.seed_base))] = {
                "absolute_spearman_gain": gain,
                "recovery_ratio": recovery if np.isfinite(recovery) else None,
                "pass": passed,
            }
        method_pass = all(item["pass"] for item in per_seed.values())
        if method_pass:
            qualifying.append(method)
        evidence[method] = {"pass": method_pass, "per_seed": per_seed}
    supported = bool(qualifying)
    return {
        "status": "SUPPORTED_IN_PILOT" if supported else "NOT_SUPPORTED_IN_PILOT",
        "supported": supported,
        "thresholds": dict(thresholds),
        "qualifying_poolings": qualifying,
        "methods": evidence,
    }


def padding_geometry_gate(
    table4: pd.DataFrame,
    table5: pd.DataFrame,
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    thresholds = config["padding_geometry"]
    recovery = _gate_recovery_rows(table4, stage="final", pooling="PVALID")
    ftv_per_seed: dict[str, Any] = {}
    for row in recovery.itertuples(index=False):
        ratio = float(row.recovery_ratio)
        passed = bool(
            _at_least(
                float(row.absolute_gain_vs_new_p0),
                thresholds["minimum_spearman_gain_each_seed"],
            )
            or (
                bool(row.recovery_defined)
                and np.isfinite(ratio)
                and _at_least(ratio, thresholds["minimum_recovery_each_seed"])
            )
        )
        ftv_per_seed[str(int(row.seed_base))] = {
            "absolute_spearman_gain": float(row.absolute_gain_vs_new_p0),
            "recovery_ratio": ratio if np.isfinite(ratio) else None,
            "pass": passed,
        }
    ftv_pass = all(value["pass"] for value in ftv_per_seed.values())
    candidates: dict[str, Any] = {}
    qualifying: list[str] = []
    for target in ("padding_fraction", "valid_source_fraction"):
        per_seed: dict[str, Any] = {}
        for seed in SEEDS:
            selected = table5.loc[
                table5["stage"].eq("final")
                & table5["seed_base"].eq(seed)
                & table5["arm"].eq("N1")
                & table5["target_name"].eq(target)
                & table5["endpoint"].eq("macro")
                & table5["pooling"].isin(["P0", "PVALID"])
                & table5["availability"].eq("AVAILABLE")
            ]
            if set(selected["pooling"]) != {"P0", "PVALID"} or len(selected) != 2:
                raise ValueError(f"padding nuisance evidence is incomplete for {seed}/{target}")
            values = selected.set_index("pooling")["natural_r2"].astype(float)
            p0 = float(values["P0"])
            pvalid = float(values["PVALID"])
            reduction = p0 - pvalid
            passed = bool(
                np.isfinite(p0)
                and np.isfinite(pvalid)
                and _at_least(
                    p0, thresholds["minimum_p0_nuisance_r2_each_seed"]
                )
                and _at_least(
                    reduction,
                    thresholds["minimum_nuisance_r2_reduction_each_seed"],
                )
            )
            per_seed[str(seed)] = {
                "p0_natural_r2": p0,
                "pvalid_natural_r2": pvalid,
                "r2_reduction": reduction,
                "pass": passed,
            }
        candidate_pass = all(value["pass"] for value in per_seed.values())
        if candidate_pass:
            qualifying.append(target)
        candidates[target] = {"pass": candidate_pass, "per_seed": per_seed}
    supported = bool(ftv_pass and qualifying)
    return {
        "status": "SUPPORTED_IN_PILOT" if supported else "NOT_SUPPORTED_IN_PILOT",
        "supported": supported,
        "thresholds": dict(thresholds),
        "pvalid_ftv": {"pass": ftv_pass, "per_seed": ftv_per_seed},
        "qualifying_nuisance_targets": qualifying,
        "nuisance_targets": candidates,
    }


def undertraining_gate(
    table7: pd.DataFrame,
    summary: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "seed",
        "arm",
        "fold",
        "configured_max_epoch",
        "hit_configured_max_epoch",
        "selected_in_last_two_observed_epochs",
        "last_three_normalized_validation_state_slope",
    }
    _required_columns(table7, required, "Table 7")
    expected = {(seed, arm, fold) for seed in SEEDS for arm in ARMS for fold in FOLDS}
    observed = set(
        zip(table7["seed"].astype(int), table7["arm"].astype(str), table7["fold"].astype(int))
    )
    if observed != expected or len(table7) != 40:
        raise ValueError("Table 7 is not the exact frozen 40-cell matrix")
    configured_max = pd.to_numeric(
        table7["configured_max_epoch"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if not np.isfinite(configured_max).all() or not np.equal(configured_max, 12).all():
        raise ValueError("Table 7 configured max epoch drifted from frozen epoch 12")

    def strict_booleans(column: str) -> pd.Series:
        def parse(value: Any) -> bool:
            if isinstance(value, (bool, np.bool_)):
                return bool(value)
            if isinstance(value, str) and value in {"True", "False"}:
                return value == "True"
            raise ValueError(f"Table 7 {column} contains a non-boolean value")

        return table7[column].map(parse)

    hit_flags = strict_booleans("hit_configured_max_epoch")
    last_two_flags = strict_booleans("selected_in_last_two_observed_epochs")
    slopes = pd.to_numeric(
        table7["last_three_normalized_validation_state_slope"], errors="coerce"
    )
    if not np.isfinite(slopes.to_numpy(dtype=np.float64)).all():
        raise ValueError("Table 7 normalized validation slopes must be finite")
    thresholds = config["undertraining"]
    if (
        summary.get("schema_version") != 1
        or summary.get("status") != "COMPLETE"
        or summary.get("new_training_performed") is not False
        or summary.get("undertraining_thresholds") != thresholds
    ):
        raise ValueError("training-budget summary thresholds/status drifted")
    summary_flags = summary.get("undertraining_plausible")
    summary_arms = summary.get("arm_summary")
    if not isinstance(summary_flags, Mapping) or set(summary_flags) != {"N1", "N3"}:
        raise ValueError("training-budget summary N-arm flags drifted")
    if not isinstance(summary_arms, Mapping):
        raise ValueError("training-budget arm summary is missing")
    evidence: dict[str, Any] = {}
    for arm in ("N1", "N3"):
        group = table7.loc[table7["arm"].eq(arm)]
        group_indices = group.index
        hit_rate = float(hit_flags.loc[group_indices].mean())
        last_two_rate = float(last_two_flags.loc[group_indices].mean())
        median_slope = float(slopes.loc[group_indices].median())
        plausible = bool(
            _at_least(hit_rate, thresholds["minimum_hit_max_rate"])
            and _at_least(
                last_two_rate, thresholds["minimum_selected_last_two_rate"]
            )
            and _at_most(
                median_slope,
                thresholds["maximum_median_normalized_last3_slope"],
            )
        )
        if not isinstance(summary_flags[arm], (bool, np.bool_)):
            raise ValueError(f"training-budget summary flag is not boolean for {arm}")
        if bool(summary_flags[arm]) != plausible:
            raise ValueError(f"training-budget summary flag drift for {arm}")
        arm_summary = summary_arms.get(arm)
        if not isinstance(arm_summary, Mapping):
            raise ValueError(f"training-budget arm summary is missing for {arm}")
        expected_summary = {
            "cells": 10,
            "hit_configured_max_rate": hit_rate,
            "selected_last_two_rate": last_two_rate,
            "median_last_three_normalized_slope": median_slope,
        }
        for field, expected_value in expected_summary.items():
            observed_value = arm_summary.get(field)
            if isinstance(expected_value, int):
                matches = observed_value == expected_value
            else:
                try:
                    matches = math.isclose(
                        float(observed_value), expected_value, rel_tol=0, abs_tol=1e-12
                    )
                except (TypeError, ValueError):
                    matches = False
            if not matches:
                raise ValueError(
                    f"training-budget arm summary drift for {arm}/{field}"
                )
        evidence[arm] = {
            "cells": int(len(group)),
            "hit_configured_max_rate": hit_rate,
            "selected_last_two_rate": last_two_rate,
            "median_last_three_normalized_slope": median_slope,
            "undertraining_plausible": plausible,
        }
    any_plausible = any(item["undertraining_plausible"] for item in evidence.values())
    if summary.get("any_n_arm_undertraining_plausible") is not any_plausible:
        raise ValueError("training-budget any-N-arm flag drifted")
    return {
        "status": "SECONDARY_CONFOUND_ONLY",
        "thresholds": dict(thresholds),
        "arms": evidence,
        "any_n_arm_undertraining_plausible": any_plausible,
    }


def unique_classification(
    *,
    final_oracle_strong: bool,
    s3_oracle_strong: bool | None,
    padding_geometry_supported: bool,
    deployable_local_supported: bool,
) -> dict[str, str]:
    if not final_oracle_strong and s3_oracle_strong is None:
        raise ValueError("weak final oracle requires completed conditional S3 evidence")
    if not final_oracle_strong and not bool(s3_oracle_strong):
        return {
            "code": "C",
            "classification": "C ENCODER BOTTLENECK",
            "next": "Stronger Pretrained 3-D Encoder Pilot",
        }
    if padding_geometry_supported:
        return {
            "code": "B",
            "classification": "B PADDING / GEOMETRY DILUTION",
            "next": "Valid-source-aware + Localized Response Pooling Pilot",
        }
    if final_oracle_strong and deployable_local_supported:
        return {
            "code": "A",
            "classification": "A POOLING BOTTLENECK",
            "next": "Local–Global Response State Pilot",
        }
    if final_oracle_strong and not deployable_local_supported:
        next_step = "Learned Spatial Response Aggregation Pilot"
    elif bool(s3_oracle_strong) and not final_oracle_strong:
        next_step = "Preserve Higher-Resolution Spatial Features Pilot"
    else:
        next_step = "Local–Global Response State Minimal Pilot"
    return {
        "code": "D",
        "classification": "D MIXED BOTTLENECK",
        "next": next_step,
    }


def build_prospective_gates(
    table4: pd.DataFrame,
    table5: pd.DataFrame,
    table7: pd.DataFrame,
    training_summary: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    s3_executed: bool,
) -> dict[str, Any]:
    final_oracle = strong_oracle_gate(table4, stage="final", config=config)
    final_local = deployable_local_gate(table4, stage="final", config=config)
    padding = padding_geometry_gate(table4, table5, config=config)
    budget = undertraining_gate(table7, training_summary, config=config)
    if final_oracle["supported"]:
        if s3_executed:
            raise ValueError("S3 outputs exist despite a strong final oracle gate")
        s3_status = "NOT_TRIGGERED_FINAL_ORACLE_STRONG"
        s3_oracle = None
        s3_local = None
        s3_strong_value: bool | None = None
    else:
        if not s3_executed:
            raise ValueError("weak final oracle requires the conditional S3 audit")
        s3_status = "TRIGGERED_FINAL_ORACLE_WEAK_COMPLETED"
        s3_oracle = strong_oracle_gate(table4, stage="s3", config=config)
        s3_local = deployable_local_gate(table4, stage="s3", config=config)
        s3_strong_value = bool(s3_oracle["supported"])
    classification = unique_classification(
        final_oracle_strong=bool(final_oracle["supported"]),
        s3_oracle_strong=s3_strong_value,
        padding_geometry_supported=bool(padding["supported"]),
        deployable_local_supported=bool(final_local["supported"]),
    )
    return {
        "schema_version": 1,
        "status": "COMPLETE",
        "natural_metrics": "pooled_five_outer_test_folds_before_metric",
        "transformed_metrics": "outer_fold_summaries_only",
        "new_training_performed": False,
        "probe_refit_during_aggregation": False,
        "final_stage": {
            "strong_oracle_recovery": final_oracle,
            "deployable_local_recovery": final_local,
            "padding_geometry_evidence": padding,
        },
        "conditional_s3": {
            "trigger_status": s3_status,
            "strong_oracle_recovery": s3_oracle,
            "deployable_local_recovery": s3_local,
        },
        "training_budget": budget,
        "classification": classification,
    }


def _has_probe_outputs(path: Path) -> bool:
    return path.is_dir() and any(path.rglob("ridge_predictions.private.csv"))


def aggregate_frozen_results(
    *,
    final_probe_root: str | Path,
    s3_probe_root: str | Path,
    occupancy_path: str | Path,
    nuisance_path: str | Path,
    old_prediction_root: str | Path,
    table7_path: str | Path,
    training_summary_path: str | Path,
    preregistration_lock: str | Path,
    config_path: str | Path,
) -> AggregationResult:
    config = load_audit_config(config_path)
    nuisance = pd.read_csv(nuisance_path)
    occupancy = pd.read_csv(occupancy_path)
    table1 = build_feature_contract_table(nuisance, config)

    final = load_probe_stage(final_probe_root, stage="final")
    final_natural = pooled_stage_natural_metrics(final)
    final_transformed = transformed_fold_summaries(final)
    table2_final = build_ftv_table(
        final, final_natural, final_transformed, task="static", config=config
    )
    table3_final = build_ftv_table(
        final, final_natural, final_transformed, task="delta", config=config
    )
    table5 = build_nuisance_table(
        final, final_natural, final_transformed, config=config
    )
    table4_final = build_recovery_table(table2_final)
    final_oracle = strong_oracle_gate(table4_final, stage="final", config=config)

    s3_root = Path(s3_probe_root).resolve()
    s3_present = _has_probe_outputs(s3_root)
    if final_oracle["supported"]:
        if s3_present:
            raise ValueError("conditional S3 results exist although final oracle is strong")
        table2 = table2_final
        table3 = table3_final
        table4 = table4_final
    else:
        if not s3_present:
            raise FileNotFoundError("final oracle is weak but conditional S3 outputs are absent")
        s3 = load_probe_stage(s3_root, stage="s3")
        s3_natural = pooled_stage_natural_metrics(s3)
        s3_transformed = transformed_fold_summaries(s3)
        table2_s3 = build_ftv_table(
            s3, s3_natural, s3_transformed, task="static", config=config
        )
        table3_s3 = build_ftv_table(
            s3, s3_natural, s3_transformed, task="delta", config=config
        )
        table2 = pd.concat([table2_final, table2_s3], ignore_index=True)
        table3 = pd.concat([table3_final, table3_s3], ignore_index=True)
        table4 = build_recovery_table(table2)

    old_predictions = load_frozen_p0_predictions(
        prediction_root=old_prediction_root,
        preregistration_lock=preregistration_lock,
    )
    paired = pair_frozen_p0_errors(old_predictions)
    private_joined = build_private_diagnostic_join(
        paired,
        occupancy,
        nuisance,
        downsampling_bins=config["downsampling_bins"],
    )
    table6 = build_table6(
        private_joined, downsampling_bins=config["downsampling_bins"]
    )
    table7 = pd.read_csv(table7_path)
    training_summary = json.loads(Path(training_summary_path).read_text(encoding="utf-8"))
    gates = build_prospective_gates(
        table4,
        table5,
        table7,
        training_summary,
        config=config,
        s3_executed=not bool(final_oracle["supported"]),
    )
    return AggregationResult(
        table1=table1,
        table2=table2,
        table3=table3,
        table4=table4,
        table5=table5,
        table6=table6,
        gates=gates,
        private_joined=private_joined,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _temporary(destination: Path, suffix: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=suffix, dir=destination.parent
    )
    os.close(descriptor)
    return Path(name)


def validate_aggregation_result(result: AggregationResult) -> None:
    """Fail closed before any aggregate artifact is created."""

    schemas = {
        "table1": TABLE1_COLUMNS,
        "table2": FTV_TABLE_COLUMNS,
        "table3": FTV_TABLE_COLUMNS,
        "table4": TABLE4_COLUMNS,
        "table5": TABLE5_COLUMNS,
        "table6": TABLE6_COLUMNS,
    }
    identities = {
        "table1": ("stage", "input_contract"),
        "table2": ("stage", "seed_base", "arm", "pooling", "endpoint"),
        "table3": ("stage", "seed_base", "arm", "pooling", "endpoint"),
        "table4": ("stage", "seed_base", "new_arm", "pooling"),
        "table5": (
            "stage",
            "seed_base",
            "arm",
            "pooling",
            "target_name",
            "endpoint",
        ),
        "table6": ("analysis", "seed_base", "endpoint", "stratum"),
    }
    for name, table in result.public_tables().items():
        if not isinstance(table, pd.DataFrame) or tuple(table.columns) != schemas[name]:
            raise ValueError(f"public {name} schema drift")
        forbidden = {"patient_id", "cache_path", "ftv_mask_nifti"} & set(
            table.columns
        )
        if forbidden:
            raise ValueError(
                f"public {name} contains private columns: {sorted(forbidden)}"
            )
        if table.duplicated(list(identities[name])).any():
            raise ValueError(f"public {name} contains duplicate aggregate identities")
        numeric = table.select_dtypes(include=[np.number])
        if not numeric.empty and np.isinf(numeric.to_numpy(dtype=np.float64)).any():
            raise ValueError(f"public {name} contains an infinite numeric value")
    expected_table1 = {
        (stage, contract)
        for stage in ("final", "s3")
        for contract in ("legacy", "c1b")
    }
    observed_table1 = set(
        zip(result.table1["stage"].astype(str), result.table1["input_contract"].astype(str))
    )
    if observed_table1 != expected_table1 or len(result.table1) != 4:
        raise ValueError("public table1 is not the exact four-row feature contract")

    if tuple(result.private_joined.columns) != PRIVATE_JOINED_COLUMNS:
        raise ValueError("private joined diagnostic schema drift")
    if result.private_joined.duplicated(
        ["seed_base", "endpoint", "patient_id"]
    ).any():
        raise ValueError("private joined diagnostic contains duplicate OOF identities")
    private_numeric = result.private_joined.select_dtypes(include=[np.number])
    if not private_numeric.empty and not np.isfinite(
        private_numeric.to_numpy(dtype=np.float64)
    ).all():
        raise ValueError("private joined diagnostic contains a non-finite numeric value")

    gates = result.gates
    required_gate_fields = {
        "schema_version",
        "status",
        "natural_metrics",
        "transformed_metrics",
        "new_training_performed",
        "probe_refit_during_aggregation",
        "final_stage",
        "conditional_s3",
        "training_budget",
        "classification",
    }
    if not isinstance(gates, Mapping) or set(gates) != required_gate_fields:
        raise ValueError("prospective gate schema drift")
    if (
        gates["schema_version"] != 1
        or gates["status"] != "COMPLETE"
        or gates["natural_metrics"]
        != "pooled_five_outer_test_folds_before_metric"
        or gates["transformed_metrics"] != "outer_fold_summaries_only"
        or gates["new_training_performed"] is not False
        or gates["probe_refit_during_aggregation"] is not False
    ):
        raise ValueError("prospective gate contract drift")
    conditional = gates["conditional_s3"]
    if not isinstance(conditional, Mapping) or conditional.get("trigger_status") not in {
        "NOT_TRIGGERED_FINAL_ORACLE_STRONG",
        "TRIGGERED_FINAL_ORACLE_WEAK_COMPLETED",
    }:
        raise ValueError("conditional S3 trigger status drift")
    classification = gates["classification"]
    if not isinstance(classification, Mapping) or set(classification) != {
        "code",
        "classification",
        "next",
    }:
        raise ValueError("prospective result must contain exactly one A-D classification")
    names = {
        "A": "A POOLING BOTTLENECK",
        "B": "B PADDING / GEOMETRY DILUTION",
        "C": "C ENCODER BOTTLENECK",
        "D": "D MIXED BOTTLENECK",
    }
    next_steps = {
        "A": {"Local–Global Response State Pilot"},
        "B": {"Valid-source-aware + Localized Response Pooling Pilot"},
        "C": {"Stronger Pretrained 3-D Encoder Pilot"},
        "D": {
            "Learned Spatial Response Aggregation Pilot",
            "Preserve Higher-Resolution Spatial Features Pilot",
            "Local–Global Response State Minimal Pilot",
        },
    }
    code = classification.get("code")
    if (
        code not in names
        or classification.get("classification") != names[code]
        or classification.get("next") not in next_steps[code]
    ):
        raise ValueError("prospective A-D classification/NEXT mapping drift")


def write_aggregation_outputs(
    result: AggregationResult, *, output_dir: str | Path
) -> dict[str, Path]:
    """Publish public aggregates as 0644 and the identifier join as 0600."""

    root = Path(output_dir).resolve()
    outputs = {
        name: root / filename for name, filename in PUBLIC_FILENAMES.items()
    }
    outputs["private_joined"] = root / PRIVATE_JOINED_FILENAME
    if existing := [path for path in outputs.values() if path.exists()]:
        raise FileExistsError(f"refusing to overwrite aggregate outputs: {existing}")
    validate_aggregation_result(result)

    temporaries: dict[str, Path] = {}
    published: list[Path] = []
    try:
        for name, table in result.public_tables().items():
            temp = _temporary(outputs[name], ".csv")
            table.to_csv(temp, index=False)
            os.chmod(temp, 0o644)
            temporaries[name] = temp
        gate_temp = _temporary(outputs["gates"], ".json")
        gate_temp.write_text(
            json.dumps(
                _json_safe(result.gates),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.chmod(gate_temp, 0o644)
        temporaries["gates"] = gate_temp
        private_temp = _temporary(outputs["private_joined"], ".csv")
        result.private_joined.to_csv(private_temp, index=False)
        os.chmod(private_temp, 0o600)
        temporaries["private_joined"] = private_temp
        for name, destination in outputs.items():
            temporary = temporaries[name]
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.link(temporary, destination)
            temporary.unlink()
            destination.chmod(0o600 if name == "private_joined" else 0o644)
            published.append(destination)
        return outputs
    except Exception:
        for path in published:
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in temporaries.values():
            path.unlink(missing_ok=True)


__all__ = [
    "AggregationResult",
    "FINAL_CORE_POOLINGS",
    "FTV_TABLE_COLUMNS",
    "NUISANCE_TARGETS",
    "PRIVATE_JOINED_COLUMNS",
    "ProbeStageData",
    "S3_CORE_POOLINGS",
    "TABLE1_COLUMNS",
    "TABLE4_COLUMNS",
    "TABLE5_COLUMNS",
    "TABLE6_COLUMNS",
    "aggregate_frozen_results",
    "build_feature_contract_table",
    "build_ftv_table",
    "build_nuisance_table",
    "build_private_diagnostic_join",
    "build_prospective_gates",
    "build_recovery_table",
    "build_table6",
    "deployable_local_gate",
    "load_audit_config",
    "load_frozen_p0_predictions",
    "load_probe_stage",
    "padding_geometry_gate",
    "pair_frozen_p0_errors",
    "pooled_stage_natural_metrics",
    "strong_oracle_gate",
    "transformed_fold_summaries",
    "undertraining_gate",
    "unique_classification",
    "validate_aggregation_result",
    "write_aggregation_outputs",
]

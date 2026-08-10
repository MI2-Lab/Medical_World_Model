"""Replication gate for P0 Ridge probes against immutable Stage-B outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .contracts import FOLDS, cell_key, cells, file_sha256, reference_probe_dir
from .probes import (
    CALIBRATION_NAMES,
    METRIC_NAMES,
    pooled_oof_natural_metrics,
)
from .runtime import verify_preregistration


PREDICTION_RTOL = 1e-5
PREDICTION_ATOL = 1e-6
POOLED_METRIC_ATOL = 1e-6
_CELL_KEYS = ["task", "endpoint", "analysis_scope", "target_semantics"]
_PREDICTION_KEYS = [*_CELL_KEYS, "patient_id"]


def _metadata_output_paths(path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("pooling") != "P0":
        raise ValueError("probe metadata is not a P0 result")
    output = path.parent
    paths = {
        "selection": output / "ridge_selection.csv",
        "prediction": output / "ridge_predictions.private.csv",
        "metrics": output / "probe_metrics.csv",
    }
    for key, candidate in paths.items():
        if not candidate.is_file():
            raise FileNotFoundError(f"P0 probe output is missing: {candidate}")
        expected = payload.get("output_sha256", {}).get(candidate.name)
        if expected != file_sha256(candidate):
            raise ValueError(f"P0 probe output SHA-256 drifted at {key}")
    return payload, paths


def discover_p0_probe_outputs(
    probe_root: str | Path,
) -> dict[tuple[int, str, int], dict[str, Path]]:
    root = Path(probe_root).resolve()
    discovered: dict[tuple[int, str, int], dict[str, Path]] = {}
    for metadata_path in sorted(root.rglob("probe_metadata.json")):
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if payload.get("pooling") != "P0":
            continue
        key = (
            int(payload["seed_base"]),
            str(payload["arm"]).upper(),
            int(payload["fold"]),
        )
        validated, paths = _metadata_output_paths(metadata_path)
        if key in discovered:
            raise ValueError(f"duplicate P0 probe result for {key}")
        if int(validated.get("feature_dim", -1)) != 192:
            raise ValueError("P0 probe metadata feature dimension drifted")
        discovered[key] = paths
    expected = set(cells())
    if set(discovered) != expected:
        raise ValueError(
            "P0 probe matrix is incomplete: "
            f"missing={sorted(expected.difference(discovered))}, "
            f"extra={sorted(set(discovered).difference(expected))}"
        )
    return discovered


def _ftv_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "target_name" in frame:
        frame = frame.loc[frame["target_name"].eq("FTV")]
    if "pooling" in frame:
        frame = frame.loc[frame["pooling"].eq("P0")]
    return frame.copy()


def _exact_key_order(
    new: pd.DataFrame, old: pd.DataFrame, keys: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for label, frame in (("new", new), ("old", old)):
        if missing := sorted(set(keys).difference(frame.columns)):
            raise ValueError(f"{label} P0 probe rows miss keys: {missing}")
        if frame.duplicated(keys).any():
            raise ValueError(f"{label} P0 probe rows duplicate keys")
    new_keys = set(new[keys].itertuples(index=False, name=None))
    old_keys = set(old[keys].itertuples(index=False, name=None))
    if new_keys != old_keys:
        raise ValueError("new and immutable P0 probe row keys differ")
    return (
        new.sort_values(keys).reset_index(drop=True),
        old.sort_values(keys).reset_index(drop=True),
    )


def _max_absolute_difference(left: Iterable[float], right: Iterable[float]) -> float:
    left_values = np.asarray(list(left), dtype=np.float64)
    right_values = np.asarray(list(right), dtype=np.float64)
    if left_values.shape != right_values.shape:
        raise ValueError("numeric comparison shapes differ")
    both_nan = np.isnan(left_values) & np.isnan(right_values)
    if np.any(np.isnan(left_values) ^ np.isnan(right_values)):
        return float("inf")
    finite = ~both_nan
    return 0.0 if not finite.any() else float(
        np.max(np.abs(left_values[finite] - right_values[finite]))
    )


def compare_p0_probe_cell(
    new_paths: dict[str, Path], old_dir: str | Path
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    old_root = Path(old_dir).resolve()
    new_selection = _ftv_rows(pd.read_csv(new_paths["selection"]))
    old_selection = pd.read_csv(old_root / "ridge_selection.csv")
    new_selection, old_selection = _exact_key_order(
        new_selection, old_selection, _CELL_KEYS
    )
    if len(new_selection) != 14:
        raise ValueError("P0 FTV selection must contain exactly 14 task cells")
    exact_selection_columns = [
        "selected_alpha",
        "n_train",
        "n_val",
        "n_test",
        "target_transform_json",
        "test_used_for_scaler",
        "test_used_for_alpha_selection",
        "test_predict_call_count",
    ]
    selection_exact = True
    for column in exact_selection_columns:
        left = new_selection[column].to_numpy()
        right = old_selection[column].to_numpy()
        if left.dtype.kind in "fc" or right.dtype.kind in "fc":
            equal = np.array_equal(left, right, equal_nan=True)
        else:
            equal = np.array_equal(left, right)
        selection_exact = selection_exact and bool(equal)

    new_predictions = _ftv_rows(pd.read_csv(new_paths["prediction"]))
    old_predictions = pd.read_csv(old_root / "ridge_predictions.private.csv")
    new_predictions, old_predictions = _exact_key_order(
        new_predictions, old_predictions, _PREDICTION_KEYS
    )
    exact_prediction_columns = [
        "split",
        "selected_alpha",
        "n_train",
        "n_val",
        "n_test",
        "analysis_scale",
        "test_predict_call_count",
    ]
    prediction_contract_exact = all(
        np.array_equal(
            new_predictions[column].to_numpy(), old_predictions[column].to_numpy()
        )
        for column in exact_prediction_columns
    )
    numeric_columns = [
        "y_true",
        "y_pred",
        "y_true_analysis",
        "y_pred_analysis",
        "b0_prediction",
        "b0_prediction_analysis",
    ]
    close_by_column = {
        column: bool(
            np.allclose(
                new_predictions[column].to_numpy(dtype=np.float64),
                old_predictions[column].to_numpy(dtype=np.float64),
                rtol=PREDICTION_RTOL,
                atol=PREDICTION_ATOL,
                equal_nan=False,
            )
        )
        for column in numeric_columns
    }
    difference_by_column = {
        column: _max_absolute_difference(
            new_predictions[column], old_predictions[column]
        )
        for column in numeric_columns
    }
    row = {
        "selection_rows": len(new_selection),
        "prediction_rows": len(new_predictions),
        "selection_contract_exact": bool(selection_exact),
        "prediction_keys_exact": True,
        "prediction_contract_exact": bool(prediction_contract_exact),
        "prediction_allclose": bool(all(close_by_column.values())),
        "maximum_prediction_absolute_difference": float(
            max(difference_by_column.values())
        ),
        **{
            f"{column}_max_abs_difference": value
            for column, value in difference_by_column.items()
        },
    }
    row["status"] = "PASS" if all(
        (
            row["selection_contract_exact"],
            row["prediction_contract_exact"],
            row["prediction_allclose"],
        )
    ) else "FAIL"
    return row, new_predictions, old_predictions


def _augment_for_pooling(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["pooling"] = "P0"
    output["feature_dim"] = 192
    output["target_name"] = "FTV"
    return output


def verify_p0_probe_matrix(
    probe_root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    lock = verify_preregistration()
    outputs = discover_p0_probe_outputs(probe_root)
    rows: list[dict[str, Any]] = []
    new_predictions: list[pd.DataFrame] = []
    old_predictions: list[pd.DataFrame] = []
    for seed, arm, fold in cells():
        key = cell_key(seed, arm, fold)
        old_root = reference_probe_dir(seed, arm, fold)
        for filename, expected in lock["formal_p0_references"][key][
            "probe_outputs_sha256"
        ].items():
            if file_sha256(old_root / filename) != expected:
                raise ValueError(f"immutable P0 probe output drifted for {key}/{filename}")
        row, new_frame, old_frame = compare_p0_probe_cell(
            outputs[(seed, arm, fold)], old_root
        )
        rows.append({"seed_base": seed, "arm": arm, "fold": fold, **row})
        new_predictions.append(_augment_for_pooling(new_frame))
        old_predictions.append(_augment_for_pooling(old_frame))
    cell_frame = pd.DataFrame(rows).sort_values(
        ["seed_base", "arm", "fold"]
    ).reset_index(drop=True)
    new_pooled = pooled_oof_natural_metrics(pd.concat(new_predictions, ignore_index=True))
    old_pooled = pooled_oof_natural_metrics(pd.concat(old_predictions, ignore_index=True))
    pooled_keys = [
        "seed_base",
        "arm",
        "task",
        "target_name",
        "endpoint",
        "analysis_scope",
        "target_semantics",
        "scale",
    ]
    new_pooled, old_pooled = _exact_key_order(new_pooled, old_pooled, pooled_keys)
    metric_columns = [*METRIC_NAMES, *CALIBRATION_NAMES]
    pooled_rows: list[dict[str, Any]] = []
    for index, new_row in new_pooled.iterrows():
        old_row = old_pooled.iloc[index]
        differences = {
            metric: _max_absolute_difference([new_row[metric]], [old_row[metric]])
            for metric in metric_columns
        }
        pooled_rows.append(
            {
                **{key: new_row[key] for key in pooled_keys},
                "maximum_metric_absolute_difference": max(differences.values()),
                "status": (
                    "PASS"
                    if max(differences.values()) <= POOLED_METRIC_ATOL
                    else "FAIL"
                ),
            }
        )
    pooled_comparison = pd.DataFrame(pooled_rows)
    status = bool(
        cell_frame["status"].eq("PASS").all()
        and pooled_comparison["status"].eq("PASS").all()
    )
    summary = {
        "schema_version": 1,
        "status": "PASS" if status else "FAIL",
        "formal_cells": 40,
        "selection_cells": int(cell_frame["selection_rows"].sum()),
        "outer_test_prediction_rows": int(cell_frame["prediction_rows"].sum()),
        "selection_contract_exact_fraction": float(
            cell_frame["selection_contract_exact"].mean()
        ),
        "prediction_contract_exact_fraction": float(
            cell_frame["prediction_contract_exact"].mean()
        ),
        "prediction_allclose_fraction": float(cell_frame["prediction_allclose"].mean()),
        "maximum_prediction_absolute_difference": float(
            cell_frame["maximum_prediction_absolute_difference"].max()
        ),
        "pooled_natural_metric_rows": len(pooled_comparison),
        "maximum_pooled_metric_absolute_difference": float(
            pooled_comparison["maximum_metric_absolute_difference"].max()
        ),
        "prediction_rtol": PREDICTION_RTOL,
        "prediction_atol": PREDICTION_ATOL,
        "pooled_metric_atol": POOLED_METRIC_ATOL,
        "alternate_pooling_interpretation_authorized": status,
    }
    return cell_frame, pooled_comparison, summary


__all__ = [
    "POOLED_METRIC_ATOL",
    "PREDICTION_ATOL",
    "PREDICTION_RTOL",
    "compare_p0_probe_cell",
    "discover_p0_probe_outputs",
    "verify_p0_probe_matrix",
]

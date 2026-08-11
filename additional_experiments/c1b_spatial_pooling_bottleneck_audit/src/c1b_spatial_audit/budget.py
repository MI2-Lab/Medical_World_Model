"""Read-only audit of the already-completed Stage-B training histories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import CHECKPOINT_ROOT, cell_key, cells, file_sha256


def _normalized_last_three_slope(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return float("nan")
    tail = values[-3:]
    slope = float(np.polyfit(np.arange(3, dtype=np.float64), tail, 1)[0])
    scale = max(abs(float(np.mean(tail))), 1e-12)
    return slope / scale


def audit_training_budget(
    checkpoint_root: str | Path = CHECKPOINT_ROOT,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    root = Path(checkpoint_root).resolve()
    rows: list[dict[str, Any]] = []
    trajectory: list[pd.DataFrame] = []
    for seed, arm, fold in cells():
        cell = root / cell_key(seed, arm, fold)
        selection_path = cell / "selection.json"
        history_path = cell / "history.csv"
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("history_sha256") != file_sha256(history_path):
            raise ValueError(f"history binding drifted for {cell_key(seed, arm, fold)}")
        history = pd.read_csv(history_path)
        required = {"epoch", "train_state_loss", "val_state_loss", "finite"}
        if not required.issubset(history.columns) or history.empty:
            raise ValueError(f"history schema drifted for {cell_key(seed, arm, fold)}")
        if not history["finite"].astype(str).str.lower().isin({"true", "1"}).all():
            raise FloatingPointError("budget audit encountered a non-finite epoch")
        observed_max = int(history["epoch"].max())
        configured_max = int(selection["hyperparameters"]["epochs"])
        selected_epoch = int(selection["selected_epoch"])
        selected_rows = history.loc[history["epoch"].eq(selected_epoch)]
        if len(selected_rows) != 1:
            raise ValueError("selected epoch is absent or duplicated in history")
        selected_state = float(selected_rows.iloc[0]["val_state_loss"])
        if not np.isclose(
            selected_state,
            float(selection["selected_validation_state_loss"]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("selection JSON and history state loss disagree")
        final_state = float(history.sort_values("epoch").iloc[-1]["val_state_loss"])
        hit_max = observed_max >= configured_max
        patience = int(selection["hyperparameters"]["patience"])
        if hit_max:
            stop_reason = "configured_max_epoch"
        elif observed_max - selected_epoch >= patience:
            stop_reason = "patience_exhausted_after_selected_epoch"
        else:
            stop_reason = "stopped_before_configured_max_other"
        slope = _normalized_last_three_slope(history["val_state_loss"].to_numpy())
        rows.append(
            {
                "seed": seed,
                "arm": arm,
                "fold": fold,
                "selected_epoch": selected_epoch,
                "observed_max_epoch": observed_max,
                "configured_max_epoch": configured_max,
                "hit_configured_max_epoch": hit_max,
                "selected_in_last_two_observed_epochs": selected_epoch >= observed_max - 1,
                "selected_validation_state_loss": selected_state,
                "final_validation_state_loss": final_state,
                "final_minus_selected_state_loss": final_state - selected_state,
                "last_three_normalized_validation_state_slope": slope,
                "early_stopping_reason": stop_reason,
                "selection_mode": str(selection.get("selection_mode")),
                "optimization_safety_pass": bool(selection.get("optimization_safety_pass")),
                "history_sha256": file_sha256(history_path),
                "selection_sha256": file_sha256(selection_path),
            }
        )
        current = history[["epoch", "train_state_loss", "val_state_loss"]].copy()
        current.insert(0, "fold", fold)
        current.insert(0, "arm", arm)
        current.insert(0, "seed", seed)
        trajectory.append(current)
    frame = pd.DataFrame(rows).sort_values(["seed", "arm", "fold"]).reset_index(drop=True)
    if len(frame) != 40:
        raise AssertionError("training-budget audit is not exactly 40 cells")
    arm_summary: dict[str, Any] = {}
    for arm, group in frame.groupby("arm", sort=True):
        arm_summary[str(arm)] = {
            "cells": int(len(group)),
            "hit_configured_max_rate": float(group["hit_configured_max_epoch"].mean()),
            "selected_last_two_rate": float(
                group["selected_in_last_two_observed_epochs"].mean()
            ),
            "median_last_three_normalized_slope": float(
                group["last_three_normalized_validation_state_slope"].median()
            ),
            "selected_epoch_median": float(group["selected_epoch"].median()),
        }
    flags: dict[str, bool] = {}
    for arm in ("N1", "N3"):
        values = arm_summary[arm]
        flags[arm] = bool(
            values["hit_configured_max_rate"] >= 0.6
            and values["selected_last_two_rate"] >= 0.6
            and values["median_last_three_normalized_slope"] <= -0.005
        )
    summary = {
        "schema_version": 1,
        "status": "COMPLETE",
        "normalization_for_last_three_slope": "OLS slope divided by absolute mean of last 3 finite validation state losses",
        "undertraining_thresholds": {
            "minimum_hit_max_rate": 0.6,
            "minimum_selected_last_two_rate": 0.6,
            "maximum_median_normalized_last3_slope": -0.005,
        },
        "arm_summary": arm_summary,
        "undertraining_plausible": flags,
        "any_n_arm_undertraining_plausible": any(flags.values()),
        "new_training_performed": False,
    }
    trajectories = pd.concat(trajectory, ignore_index=True)
    return frame, summary, trajectories


__all__ = ["audit_training_budget"]


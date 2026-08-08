"""把既有 25 个 seed×fold 结果闭环为 conflict audit 的 run-level 基表。"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .assets import history_path, validate_checkpoint_grid
from .contracts import FOLDS, SEED_BASES, SOURCE_ROOT, assert_source_hashes, atomic_csv
from .freeze import assert_plan_freeze


def _strict_bool(value: Any, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ValueError(f"{name} 非严格 boolean: {value!r}")


def _slope(frame: pd.DataFrame, column: str) -> float:
    x = frame["epoch"].to_numpy(dtype=float)
    y = frame[column].to_numpy(dtype=float)
    if len(x) < 2 or not np.isfinite(y).all():
        return math.nan
    return float(np.polyfit(x, y, 1)[0])


def _post_selected_slope(
    frame: pd.DataFrame, column: str, selected_epoch: int
) -> float:
    window = frame.loc[frame["epoch"].ge(selected_epoch), ["epoch", column]].copy()
    if len(window) < 3:
        raise ValueError(f"post-selected slope {column} 少于3 epochs")
    x = window["epoch"].to_numpy(dtype=float)
    y = window[column].to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError(f"post-selected slope {column} nonfinite")
    centered_x = x - x.mean()
    denominator = float(np.square(centered_x).sum())
    if denominator <= 0:
        raise ValueError(f"post-selected slope {column} epoch variance 为0")
    return float((centered_x * (y - y.mean())).sum() / denominator)


def build_existing_metrics() -> (
    tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]
):
    assert_plan_freeze()
    assert_source_hashes()
    validate_checkpoint_grid()
    stability = pd.read_csv(
        SOURCE_ROOT / "metrics" / "final" / "training_stability_seed_fold.csv"
    )
    effects = pd.read_csv(SOURCE_ROOT / "metrics" / "final" / "seed_fold_effects.csv")
    if len(stability) != 50 or len(effects) != 25:
        raise ValueError("existing final grid row count 漂移")
    if (
        stability.duplicated(["seed_base", "fold", "model"]).any()
        or effects.duplicated(["seed_base", "fold"]).any()
    ):
        raise ValueError("existing final grid key 重复")

    rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    for seed in SEED_BASES:
        for fold in FOLDS:
            g1_match = stability.loc[
                stability["seed_base"].eq(seed)
                & stability["fold"].eq(fold)
                & stability["model"].eq("G1")
            ]
            g3_match = stability.loc[
                stability["seed_base"].eq(seed)
                & stability["fold"].eq(fold)
                & stability["model"].eq("G3")
            ]
            effect_match = effects.loc[
                effects["seed_base"].eq(seed) & effects["fold"].eq(fold)
            ]
            if len(g1_match) != 1 or len(g3_match) != 1 or len(effect_match) != 1:
                raise ValueError(f"existing grid cell 缺失: {seed}/{fold}")
            g1 = g1_match.iloc[0]
            g3 = g3_match.iloc[0]
            effect = effect_match.iloc[0]
            history = pd.read_csv(history_path(seed, fold))
            if (
                history.empty
                or history["epoch"].duplicated().any()
                or not history["epoch"].is_monotonic_increasing
                or int(history.iloc[0]["epoch"]) != 1
            ):
                raise ValueError(f"history {seed}/{fold} epoch coverage/order 非法")
            for column, expected in (("seed_base", seed), ("fold", fold)):
                if column in history and set(history[column]) != {expected}:
                    raise ValueError(f"history {seed}/{fold} {column} metadata 漂移")
            if "model" in history and set(history["model"].astype(str).str.upper()) != {
                "G3"
            }:
                raise ValueError(f"history {seed}/{fold} model metadata 漂移")
            selected_mask = history["is_selected_checkpoint"].map(
                lambda value: _strict_bool(value, f"history {seed}/{fold}.selected")
            )
            if selected_mask.sum() != 1:
                raise ValueError(f"history {seed}/{fold} selected row 不唯一")
            selected = history.loc[selected_mask].iloc[0]
            selected_epoch = int(selected["epoch"])
            last = history.iloc[-1]
            degradation = (
                float(g3["val_state_loss"]) - float(g1["val_state_loss"])
            ) / float(g1["val_state_loss"])
            if not math.isclose(
                degradation,
                float(g3["base_degradation_fraction"]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(f"base degradation 重算不一致: {seed}/{fold}")
            base_pass = _strict_bool(
                g3["base_pass"], f"stability {seed}/{fold}.base_pass"
            )
            if base_pass != (degradation <= 0.05):
                raise ValueError(f"base gate threshold 不一致: {seed}/{fold}")
            cumulative = history.loc[
                history["epoch"].le(selected_epoch), "train_grounded_patients"
            ].sum()
            row = {
                "seed_base": seed,
                "fold": fold,
                "effective_seed": seed + fold,
                "selected_epoch": selected_epoch,
                "last_epoch": int(last["epoch"]),
                "last_minus_selected_epoch": int(last["epoch"]) - selected_epoch,
                "selection_mode": str(g3["selection_mode"]),
                "base_degradation": degradation,
                "base_gate_pass": base_pass,
                "base_gate": "PASS" if base_pass else "FAIL",
                "static_ftv_delta_spearman": float(effect["dS_sf"]),
                "delta_ftv_delta_spearman": float(effect["dD_sf"]),
                "delta_ftv_delta_r2": float(effect["D"]),
                "representation_std": float(g3["representation_std"]),
                "g1_val_state_loss": float(g1["val_state_loss"]),
                "g3_val_state_loss": float(g3["val_state_loss"]),
                "available_train_loss": float(selected["train_loss"]),
                "available_train_base_loss": float(selected["train_base_loss"]),
                "available_train_state_loss": float(selected["train_state_loss"]),
                "available_train_sigreg_loss": float(selected["train_sigreg_loss"]),
                "available_val_loss": float(selected["val_loss"]),
                "available_val_state_loss": float(selected["val_state_loss"]),
                "available_val_base_objective": float(selected["val_base_objective"]),
                "available_ftv_loss": float(selected["train_ftv_loss"]),
                "available_val_ftv_loss": float(selected["val_ftv_loss"]),
                "selected_epoch_grounded_exposure": float(
                    selected["train_grounded_patients"]
                ),
                "cumulative_grounded_exposure_to_selected": float(cumulative),
                "history_grounded_exposure_mean": float(
                    history["train_grounded_patients"].mean()
                ),
                "history_grounded_exposure_sd": float(
                    history["train_grounded_patients"].std(ddof=1)
                ),
                "history_train_total_slope": _slope(history, "train_loss"),
                "history_train_base_slope": _slope(history, "train_base_loss"),
                "history_train_ftv_slope": _slope(history, "train_ftv_loss"),
                "post_selected_val_state_slope": _post_selected_slope(
                    history, "val_state_loss", selected_epoch
                ),
                "post_selected_val_ftv_slope": _post_selected_slope(
                    history, "val_ftv_loss", selected_epoch
                ),
                "selected_to_last_val_state_change": float(last["val_state_loss"])
                - float(selected["val_state_loss"]),
                "selected_to_last_val_ftv_change": float(last["val_ftv_loss"])
                - float(selected["val_ftv_loss"]),
                "history_rows": len(history),
            }
            if not all(
                math.isfinite(float(value))
                for name, value in row.items()
                if name
                not in {
                    "selection_mode",
                    "base_gate",
                    "base_gate_pass",
                }
            ):
                raise ValueError(f"existing run-level row nonfinite: {seed}/{fold}")
            rows.append(row)
            for epoch_row in history.itertuples(index=False):
                history_rows.append(
                    {
                        "seed_base": seed,
                        "fold": fold,
                        "base_gate": row["base_gate"],
                        "base_degradation": degradation,
                        "epoch": int(epoch_row.epoch),
                        "is_selected_checkpoint": int(epoch_row.epoch)
                        == selected_epoch,
                        "train_total_loss": float(epoch_row.train_loss),
                        "train_base_loss": float(epoch_row.train_base_loss),
                        "train_state_loss": float(epoch_row.train_state_loss),
                        "train_sigreg_loss": float(epoch_row.train_sigreg_loss),
                        "train_ftv_loss": float(epoch_row.train_ftv_loss),
                        "train_weighted_ftv_loss": float(
                            epoch_row.train_weighted_ftv_loss
                        ),
                        "val_state_loss": float(epoch_row.val_state_loss),
                        "val_base_objective": float(epoch_row.val_base_objective),
                        "val_ftv_loss": float(epoch_row.val_ftv_loss),
                        "representation_std": float(epoch_row.representation_std),
                        "grounded_exposure": float(epoch_row.train_grounded_patients),
                    }
                )

    if len(rows) != 25 or sum(bool(row["base_gate_pass"]) for row in rows) != 17:
        raise ValueError("existing run-level grid/base gate count 错误")
    frame = pd.DataFrame(rows).sort_values(
        ["base_gate_pass", "base_degradation"], ascending=[False, True]
    )
    representatives: list[dict[str, Any]] = []
    for gate, expected in (("PASS", 17), ("FAIL", 8)):
        group = (
            frame.loc[frame["base_gate"].eq(gate)]
            .sort_values("base_degradation")
            .reset_index(drop=True)
        )
        if len(group) != expected:
            raise ValueError(f"{gate} group count 错误")
        ranks = [0, 8, 16] if gate == "PASS" else [0, 3, 7]
        labels = ["minimum", "median" if gate == "PASS" else "lower_median", "maximum"]
        for rank, label in zip(ranks, labels):
            item = group.iloc[rank]
            representatives.append(
                {
                    "base_gate": gate,
                    "selection_rule": label,
                    "within_group_rank_zero_based": rank,
                    "seed_base": int(item["seed_base"]),
                    "fold": int(item["fold"]),
                    "base_degradation": float(item["base_degradation"]),
                    "selected_epoch": int(item["selected_epoch"]),
                    "last_epoch": int(item["last_epoch"]),
                    "gradient_result_used_for_selection": False,
                }
            )
    return rows, history_rows, representatives


def write_existing_metrics(root, *, overwrite: bool = False) -> dict[str, int]:
    rows, histories, representatives = build_existing_metrics()
    atomic_csv(
        root / "metrics" / "run_level_existing_metrics.csv", rows, overwrite=overwrite
    )
    atomic_csv(
        root / "metrics" / "training_history_audit.csv", histories, overwrite=overwrite
    )
    atomic_csv(
        root / "metrics" / "representative_runs.csv",
        representatives,
        overwrite=overwrite,
    )
    return {
        "run_rows": len(rows),
        "history_rows": len(histories),
        "representative_rows": len(representatives),
    }


def synthetic_self_test() -> dict[str, bool]:
    frame = pd.DataFrame({"epoch": [1, 2, 3], "x": [3.0, 2.0, 1.0]})
    checks = {
        "strict_true": _strict_bool("true", "test") is True,
        "strict_false": _strict_bool("false", "test") is False,
        "slope_negative_one": math.isclose(_slope(frame, "x"), -1.0, abs_tol=1e-12),
    }
    try:
        _strict_bool("yes", "test")
    except ValueError:
        checks["invalid_boolean_rejected"] = True
    else:
        checks["invalid_boolean_rejected"] = False
    if not all(checks.values()):
        raise AssertionError(f"existing metrics self-test 失败: {checks}")
    return checks


__all__ = ["build_existing_metrics", "synthetic_self_test", "write_existing_metrics"]

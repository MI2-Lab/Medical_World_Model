"""把 batch-level gradient geometry 聚合为冻结的 run/layer/trajectory 表。

本模块只做确定性的描述性聚合。batch 不是推断单位；后续统计模块只能使用
这里产生的 run-level 行。公开聚合表不会保留 patient ID、batch ID 或 batch
membership HMAC。
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .assets import load_data_context
from .batches import PRIVATE_MEMBERSHIP, PUBLIC_MANIFEST, validate_manifests
from .contracts import (
    AUDIT_ROOT,
    BATCHES_PER_SPLIT,
    BATCH_SIZE,
    FOLDS,
    GROUPS,
    LAMBDA_FTV,
    SEED_BASES,
    SPLITS,
    atomic_csv,
    ensure_no_patient_columns,
    file_sha256,
)
from .gradients import BATCH_GRADIENT_COLUMNS, EXPECTED_GROUP_COUNTS
from .source_contract import SOURCE_CONTRACT, assert_source_contract


SELECTED_BATCH_ROWS = 2_800
TRAJECTORY_LAST_BATCH_ROWS = 672
LAYER_LEVEL_ROWS = 350
RUN_LEVEL_ROWS = 50
TRAJECTORY_ROWS = 168
TRAJECTORY_CHANGE_ROWS = 84
COVERAGE_ROWS = 10

AGGREGATE_VALUE_COLUMNS = (
    "gradient_cosine",
    "gradient_cosine_mean",
    "negative_fraction",
    "strong_negative_fraction",
    "very_strong_negative_fraction",
    "weighted_gradient_norm_ratio",
    "weighted_gradient_norm_ratio_mean",
    "base_descent_margin",
    "base_descent_margin_mean",
    "base_descent_failure_fraction",
    "ftv_descent_margin",
    "ftv_descent_margin_mean",
)

LAYER_LEVEL_COLUMNS = (
    "schema_version",
    "seed_base",
    "fold",
    "effective_seed",
    "base_gate",
    "base_degradation",
    "checkpoint_kind",
    "checkpoint_epoch",
    "checkpoint_sha256",
    "split",
    "group",
    "parameter_tensors",
    "parameter_count",
    *AGGREGATE_VALUE_COLUMNS,
    "n_batches",
    "n_undefined",
    "source_contract_sha256",
    "public_manifest_sha256",
    "contains_patient_ids",
)

EXISTING_METRIC_COLUMNS = (
    "seed_base",
    "fold",
    "effective_seed",
    "selected_epoch",
    "last_epoch",
    "last_minus_selected_epoch",
    "selection_mode",
    "base_degradation",
    "base_gate_pass",
    "base_gate",
    "static_ftv_delta_spearman",
    "delta_ftv_delta_spearman",
    "delta_ftv_delta_r2",
    "representation_std",
    "g1_val_state_loss",
    "g3_val_state_loss",
    "available_train_loss",
    "available_train_base_loss",
    "available_train_state_loss",
    "available_train_sigreg_loss",
    "available_val_loss",
    "available_val_state_loss",
    "available_val_base_objective",
    "available_ftv_loss",
    "available_val_ftv_loss",
    "selected_epoch_grounded_exposure",
    "cumulative_grounded_exposure_to_selected",
    "history_grounded_exposure_mean",
    "history_grounded_exposure_sd",
    "history_train_total_slope",
    "history_train_base_slope",
    "history_train_ftv_slope",
    "post_selected_val_state_slope",
    "post_selected_val_ftv_slope",
    "selected_to_last_val_state_change",
    "selected_to_last_val_ftv_change",
    "history_rows",
)

EXISTING_AUXILIARY_COLUMNS = tuple(
    column
    for column in EXISTING_METRIC_COLUMNS
    if column
    not in {"seed_base", "fold", "effective_seed", "base_degradation", "base_gate"}
)

RUN_LEVEL_COLUMNS = (*LAYER_LEVEL_COLUMNS, *EXISTING_AUXILIARY_COLUMNS)

REPRESENTATIVE_COLUMNS = (
    "base_gate",
    "selection_rule",
    "within_group_rank_zero_based",
    "seed_base",
    "fold",
    "base_degradation",
    "selected_epoch",
    "last_epoch",
    "gradient_result_used_for_selection",
)

TRAJECTORY_COLUMNS = LAYER_LEVEL_COLUMNS

TRAJECTORY_CHANGE_COLUMNS = (
    "schema_version",
    "seed_base",
    "fold",
    "effective_seed",
    "base_gate",
    "base_degradation",
    "split",
    "group",
    "parameter_tensors",
    "parameter_count",
    "selected_epoch",
    "last_epoch",
    "selected_checkpoint_sha256",
    "last_checkpoint_sha256",
    *(f"last_minus_selected_{column}" for column in AGGREGATE_VALUE_COLUMNS),
    "selected_n_batches",
    "last_n_batches",
    "selected_n_undefined",
    "last_n_undefined",
    "source_contract_sha256",
    "public_manifest_sha256",
    "contains_patient_ids",
)

PUBLIC_MANIFEST_COLUMNS = (
    "schema_version",
    "plan_freeze_sha256",
    "audit_config_sha256",
    "source_fold_manifest_sha256",
    "source_raw_ftv_semantic_sha256",
    "batch_id",
    "fold",
    "split",
    "batch_index",
    "ordered_members_hmac_sha256",
    "hmac_key_id",
    "n_total",
    "n_ftv_available",
    "n_unavailable",
    "n_ispy2",
    "n_ispy1",
    "pool_n",
    "pool_ftv_available",
    "pool_ftv_proportion",
    "batch_ftv_proportion",
    "applies_to_seed_count",
    "within_batch_replacement",
    "contains_patient_ids",
    "contains_patient_level_rows",
    "private_mapping_hmac_sha256",
)

PRIVATE_MANIFEST_COLUMNS = (
    "batch_id",
    "fold",
    "split",
    "batch_index",
    "position",
    "patient_id",
    "cohort_role",
    "has_ftv",
)

COVERAGE_COLUMNS = (
    "schema_version",
    "fold",
    "split",
    "plan_freeze_sha256",
    "audit_config_sha256",
    "source_fold_manifest_sha256",
    "source_raw_ftv_semantic_sha256",
    "n_batches",
    "batch_size",
    "pool_n",
    "pool_ftv_available",
    "pool_ftv_proportion",
    "total_draws",
    "ftv_draws",
    "ftv_draw_proportion",
    "batch_ftv_min",
    "batch_ftv_median",
    "batch_ftv_mean",
    "batch_ftv_sd",
    "batch_ftv_max",
    "unique_members_drawn",
    "unique_ftv_members_drawn",
    "pool_member_coverage_fraction",
    "pool_ftv_member_coverage_fraction",
    "member_exposure_min",
    "member_exposure_median",
    "member_exposure_mean",
    "member_exposure_sd",
    "member_exposure_max",
    "ftv_member_exposure_min",
    "ftv_member_exposure_median",
    "ftv_member_exposure_mean",
    "ftv_member_exposure_sd",
    "ftv_member_exposure_max",
    "contains_patient_ids",
)

_CORE_BATCH_NUMERIC = (
    "base_gradient_norm",
    "ftv_gradient_norm_raw",
    "weighted_ftv_gradient_norm",
    "gradient_dot_raw",
    "gradient_cosine",
    "weighted_gradient_norm_ratio",
    "base_descent_margin",
    "ftv_descent_margin",
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _strict_bool(value: Any, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ValueError(f"{name} 非严格 boolean: {value!r}")


def _strict_bool_series(series: pd.Series, name: str) -> pd.Series:
    return series.map(lambda value: _strict_bool(value, name))


def _require_columns(frame: pd.DataFrame, expected: Sequence[str], name: str) -> None:
    if tuple(frame.columns) != tuple(expected):
        raise ValueError(f"{name} exact column schema 漂移: {tuple(frame.columns)}")


def _require_finite(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    selected = frame[list(columns)]
    try:
        values = selected.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} numeric schema 非法") from error
    if not np.isfinite(values).all():
        raise ValueError(f"{name} 含 nonfinite 核心值")


def _require_sha(series: pd.Series, name: str) -> None:
    if not series.map(lambda value: bool(_HEX64.fullmatch(str(value)))).all():
        raise ValueError(f"{name} 含非法 SHA-256")


def _assert_public_frame(
    frame: pd.DataFrame,
    name: str,
    *,
    private_identifiers: Iterable[str] = (),
) -> None:
    ensure_no_patient_columns(frame.columns)
    lowered = {str(column).lower() for column in frame.columns}
    if any("batch_id" in column or "hmac" in column for column in lowered):
        raise ValueError(f"{name} 意外公开 batch identifier/HMAC")
    if (
        "contains_patient_ids" not in frame
        or _strict_bool_series(
            frame["contains_patient_ids"], f"{name}.contains_patient_ids"
        ).any()
    ):
        raise ValueError(f"{name} privacy flag 失败")
    identifiers = {str(value) for value in private_identifiers}
    for column in frame.select_dtypes(include=["object", "string"]).columns:
        values = frame[column].dropna().astype(str)
        if values.map(
            lambda value: value.startswith(("/", "file://"))
            or bool(re.match(r"^[A-Za-z]:\\", value))
        ).any():
            raise ValueError(f"{name} 含 absolute path: {column}")
        if identifiers and values.isin(identifiers).any():
            raise ValueError(f"{name} 泄漏 private identifier: {column}")


def _expected_run_set() -> set[tuple[int, int]]:
    return {(seed, fold) for seed in SEED_BASES for fold in FOLDS}


def _validate_existing(existing: pd.DataFrame) -> pd.DataFrame:
    _require_columns(existing, EXISTING_METRIC_COLUMNS, "existing metrics")
    expected = _expected_run_set()
    observed = set(existing[["seed_base", "fold"]].itertuples(index=False, name=None))
    if (
        len(existing) != 25
        or existing.duplicated(["seed_base", "fold"]).any()
        or observed != expected
    ):
        raise ValueError("existing metrics 25-run key coverage 错误")
    if (
        int(_strict_bool_series(existing["base_gate_pass"], "base_gate_pass").sum())
        != 17
    ):
        raise ValueError("existing metrics PASS/FAIL 不是17/8")
    numeric = [
        column
        for column in EXISTING_METRIC_COLUMNS
        if column not in {"selection_mode", "base_gate", "base_gate_pass"}
    ]
    _require_finite(existing, numeric, "existing metrics")
    for row in existing.itertuples(index=False):
        expected_gate = "PASS" if float(row.base_degradation) <= 0.05 else "FAIL"
        if (
            int(row.effective_seed) != int(row.seed_base) + int(row.fold)
            or str(row.base_gate) != expected_gate
            or _strict_bool(row.base_gate_pass, "base_gate_pass")
            != (expected_gate == "PASS")
        ):
            raise ValueError("existing metrics seed/base-gate contract 失败")
    return existing.copy()


def _validate_representatives(
    representatives: pd.DataFrame, existing: pd.DataFrame
) -> pd.DataFrame:
    _require_columns(representatives, REPRESENTATIVE_COLUMNS, "representative runs")
    if (
        len(representatives) != 6
        or representatives.duplicated(["seed_base", "fold"]).any()
    ):
        raise ValueError("representative runs row/key coverage 错误")
    if set(representatives["base_gate"]) != {"PASS", "FAIL"} or not all(
        (representatives["base_gate"] == gate).sum() == 3 for gate in ("PASS", "FAIL")
    ):
        raise ValueError("representative runs PASS/FAIL 不是3/3")
    if _strict_bool_series(
        representatives["gradient_result_used_for_selection"],
        "gradient_result_used_for_selection",
    ).any():
        raise ValueError("representative selection 意外使用 gradient result")
    expected_runs = _expected_run_set()
    observed_runs = set(
        representatives[["seed_base", "fold"]].itertuples(index=False, name=None)
    )
    if not observed_runs.issubset(expected_runs):
        raise ValueError("representative runs 含 grid 外 key")
    lookup = existing.set_index(["seed_base", "fold"])
    for row in representatives.itertuples(index=False):
        source = lookup.loc[(int(row.seed_base), int(row.fold))]
        if (
            str(row.base_gate) != str(source.base_gate)
            or not math.isclose(
                float(row.base_degradation),
                float(source.base_degradation),
                abs_tol=1e-12,
            )
            or int(row.selected_epoch) != int(source.selected_epoch)
            or int(row.last_epoch) != int(source.last_epoch)
        ):
            raise ValueError("representative runs 与 existing metrics 不闭环")
    return representatives.copy()


def _validate_gradient_geometry(frame: pd.DataFrame, name: str) -> None:
    base = frame["base_gradient_norm"].to_numpy(dtype=float)
    ftv = frame["ftv_gradient_norm_raw"].to_numpy(dtype=float)
    dot = frame["gradient_dot_raw"].to_numpy(dtype=float)
    if (base <= 0).any() or (ftv <= 0).any():
        raise ValueError(f"{name} gradient norm 非正")
    expected = {
        "weighted_ftv_gradient_norm": LAMBDA_FTV * ftv,
        "gradient_cosine": dot / (base * ftv),
        "weighted_gradient_norm_ratio": LAMBDA_FTV * ftv / base,
        "base_descent_margin": 1.0 + LAMBDA_FTV * dot / np.square(base),
        "ftv_descent_margin": (dot + LAMBDA_FTV * np.square(ftv))
        / (LAMBDA_FTV * np.square(ftv)),
    }
    for column, values in expected.items():
        observed = frame[column].to_numpy(dtype=float)
        if not np.allclose(observed, values, rtol=1e-9, atol=1e-12):
            raise ValueError(f"{name} {column} 不可由 raw norm/dot 重算")
    if not frame["gradient_cosine"].between(-1.0000001, 1.0000001).all():
        raise ValueError(f"{name} cosine 越界")
    flags = {
        "negative_cosine": frame["gradient_cosine"].lt(0),
        "strong_negative_cosine": frame["gradient_cosine"].lt(-0.1),
        "very_strong_negative_cosine": frame["gradient_cosine"].lt(-0.25),
        "base_descent_failure": frame["base_descent_margin"].lt(0),
    }
    for column, expected_flag in flags.items():
        observed = _strict_bool_series(frame[column], f"{name}.{column}")
        if not observed.equals(expected_flag):
            raise ValueError(f"{name} {column} flag 不可重算")


def _validate_batch_matrix(
    frame: pd.DataFrame,
    *,
    checkpoint_kind: str,
    expected_runs: set[tuple[int, int]],
    expected_rows: int,
    name: str,
    expected_public_manifest_sha256: str | None = None,
    expected_source_contract_sha256: str | None = None,
) -> pd.DataFrame:
    _require_columns(frame, BATCH_GRADIENT_COLUMNS, name)
    if len(frame) != expected_rows:
        raise ValueError(f"{name} row count {len(frame)} != {expected_rows}")
    keys = ["seed_base", "fold", "checkpoint_kind", "split", "batch_id", "group"]
    if frame.duplicated(keys).any():
        raise ValueError(f"{name} batch key 重复")
    if set(frame["checkpoint_kind"]) != {checkpoint_kind}:
        raise ValueError(f"{name} checkpoint_kind 非 {checkpoint_kind}")
    observed_runs = set(frame[["seed_base", "fold"]].itertuples(index=False, name=None))
    if observed_runs != expected_runs:
        raise ValueError(f"{name} run grid coverage 错误")
    expected_keys = {
        (seed, fold, split, batch_index, group)
        for seed, fold in expected_runs
        for split in SPLITS
        for batch_index in range(BATCHES_PER_SPLIT)
        for group in GROUPS
    }
    observed_keys = set(
        frame[["seed_base", "fold", "split", "batch_index", "group"]].itertuples(
            index=False, name=None
        )
    )
    if observed_keys != expected_keys:
        raise ValueError(f"{name} Cartesian key coverage 错误")
    if (
        set(frame["schema_version"]) != {1}
        or set(frame["split"]) != set(SPLITS)
        or set(frame["group"]) != set(GROUPS)
    ):
        raise ValueError(f"{name} schema/split/group contract 错误")
    _require_finite(frame, _CORE_BATCH_NUMERIC, name)
    _require_finite(
        frame,
        (
            "base_objective",
            "state_loss",
            "sigreg_loss",
            "ftv_loss_raw",
            "base_degradation",
        ),
        name,
    )
    if (frame["base_gradient_none_count"] != 0).any() or (
        frame["ftv_gradient_none_count"] != 0
    ).any():
        raise ValueError(f"{name} shared gradient 含 None")
    if (frame["n_total"] != BATCH_SIZE).any() or (frame["n_ftv_available"] < 8).any():
        raise ValueError(f"{name} batch size/grounding quota 失败")
    if set(frame["lambda_ftv"].astype(float)) != {LAMBDA_FTV}:
        raise ValueError(f"{name} lambda 漂移")
    if set(frame["model_mode"].astype(str)) != {"train_fixed_rng"}:
        raise ValueError(f"{name} model mode 漂移")
    if (frame["component_forward_backward_count"] != 2).any():
        raise ValueError(f"{name} forward/backward count 漂移")
    required_true = ("deterministic_algorithms", "paired_forward_outputs_exact")
    required_false = (
        "optimizer_created",
        "optimizer_step",
        "pcr_signal_used",
        "contains_patient_ids",
    )
    if any(
        not _strict_bool_series(frame[column], f"{name}.{column}").all()
        for column in required_true
    ):
        raise ValueError(f"{name} deterministic/paired-forward flag 失败")
    if any(
        _strict_bool_series(frame[column], f"{name}.{column}").any()
        for column in required_false
    ):
        raise ValueError(f"{name} optimizer/pCR/privacy flag 失败")
    _require_sha(frame["checkpoint_sha256"], f"{name}.checkpoint_sha256")
    _require_sha(
        frame["ordered_members_hmac_sha256"], f"{name}.ordered_members_hmac_sha256"
    )
    _require_sha(
        frame["model_state_sha256_before"], f"{name}.model_state_sha256_before"
    )
    _require_sha(frame["model_state_sha256_after"], f"{name}.model_state_sha256_after")
    _require_sha(frame["source_contract_sha256"], f"{name}.source_contract_sha256")
    _require_sha(frame["public_manifest_sha256"], f"{name}.public_manifest_sha256")
    if expected_public_manifest_sha256 is not None and set(
        frame["public_manifest_sha256"]
    ) != {expected_public_manifest_sha256}:
        raise ValueError(f"{name} public manifest SHA 不一致")
    if expected_source_contract_sha256 is not None and set(
        frame["source_contract_sha256"]
    ) != {expected_source_contract_sha256}:
        raise ValueError(f"{name} source contract SHA 不一致")
    if not (
        frame["model_state_sha256_before"].astype(str)
        == frame["model_state_sha256_after"].astype(str)
    ).all():
        raise ValueError(f"{name} checkpoint state 被修改")
    _validate_gradient_geometry(frame, name)

    for row in frame.itertuples(index=False):
        expected_batch_id = f"f{int(row.fold)}_{'tr' if row.split == 'train' else 'va'}_{int(row.batch_index):02d}"
        tensors, parameters = EXPECTED_GROUP_COUNTS[str(row.group)]
        expected_gate = "PASS" if float(row.base_degradation) <= 0.05 else "FAIL"
        if (
            str(row.batch_id) != expected_batch_id
            or int(row.effective_seed) != int(row.seed_base) + int(row.fold)
            or int(row.parameter_tensors) != tensors
            or int(row.parameter_count) != parameters
            or str(row.base_gate) != expected_gate
        ):
            raise ValueError(f"{name} row metadata contract 失败")

    batch_invariants = (
        "batch_id",
        "ordered_members_hmac_sha256",
        "n_total",
        "n_ftv_available",
        "n_ftv_valid_visits",
        "stochastic_seed",
        "public_manifest_sha256",
    )
    # 同一 fold 的固定 batch 必须跨五个 selected seeds 完全一致；last 是其子集。
    for (_, split, batch_index), batch in frame.groupby(
        ["fold", "split", "batch_index"], sort=False
    ):
        for column in batch_invariants:
            if batch[column].nunique(dropna=False) != 1:
                raise ValueError(f"{name} 固定 batch invariant 跨 run 漂移: {column}")
    run_invariants = (
        "effective_seed",
        "base_gate",
        "base_degradation",
        "checkpoint_epoch",
        "checkpoint_sha256",
        "model_state_sha256_before",
        "model_state_sha256_after",
        "source_contract_sha256",
        "public_manifest_sha256",
    )
    for key, run in frame.groupby(["seed_base", "fold"], sort=False):
        if any(run[column].nunique(dropna=False) != 1 for column in run_invariants):
            raise ValueError(f"{name} checkpoint/run invariant 漂移: {key}")
    return frame.copy()


def _aggregate_groups(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["seed_base", "fold", "checkpoint_kind", "split", "group"]
    invariant_columns = (
        "effective_seed",
        "base_gate",
        "base_degradation",
        "checkpoint_epoch",
        "checkpoint_sha256",
        "parameter_tensors",
        "parameter_count",
        "source_contract_sha256",
        "public_manifest_sha256",
    )
    for key, batch in frame.groupby(keys, sort=True):
        if len(batch) != BATCHES_PER_SPLIT or set(
            batch["batch_index"].astype(int)
        ) != set(range(BATCHES_PER_SPLIT)):
            raise ValueError(f"gradient aggregate cell 不是恰8 batches: {key}")
        if any(
            batch[column].nunique(dropna=False) != 1 for column in invariant_columns
        ):
            raise ValueError(f"gradient aggregate cell metadata 漂移: {key}")
        undefined = int(
            (~np.isfinite(batch[list(_CORE_BATCH_NUMERIC)].to_numpy(dtype=float)))
            .any(axis=1)
            .sum()
        )
        if undefined != 0:
            raise ValueError(f"gradient aggregate cell 含 undefined: {key}")
        first = batch.iloc[0]
        rows.append(
            {
                "schema_version": 1,
                "seed_base": int(key[0]),
                "fold": int(key[1]),
                "effective_seed": int(first["effective_seed"]),
                "base_gate": str(first["base_gate"]),
                "base_degradation": float(first["base_degradation"]),
                "checkpoint_kind": str(key[2]),
                "checkpoint_epoch": int(first["checkpoint_epoch"]),
                "checkpoint_sha256": str(first["checkpoint_sha256"]),
                "split": str(key[3]),
                "group": str(key[4]),
                "parameter_tensors": int(first["parameter_tensors"]),
                "parameter_count": int(first["parameter_count"]),
                "gradient_cosine": float(batch["gradient_cosine"].median()),
                "gradient_cosine_mean": float(batch["gradient_cosine"].mean()),
                "negative_fraction": float(
                    _strict_bool_series(
                        batch["negative_cosine"], "negative_cosine"
                    ).mean()
                ),
                "strong_negative_fraction": float(
                    _strict_bool_series(
                        batch["strong_negative_cosine"], "strong_negative_cosine"
                    ).mean()
                ),
                "very_strong_negative_fraction": float(
                    _strict_bool_series(
                        batch["very_strong_negative_cosine"],
                        "very_strong_negative_cosine",
                    ).mean()
                ),
                "weighted_gradient_norm_ratio": float(
                    batch["weighted_gradient_norm_ratio"].median()
                ),
                "weighted_gradient_norm_ratio_mean": float(
                    batch["weighted_gradient_norm_ratio"].mean()
                ),
                "base_descent_margin": float(batch["base_descent_margin"].median()),
                "base_descent_margin_mean": float(batch["base_descent_margin"].mean()),
                "base_descent_failure_fraction": float(
                    _strict_bool_series(
                        batch["base_descent_failure"], "base_descent_failure"
                    ).mean()
                ),
                "ftv_descent_margin": float(batch["ftv_descent_margin"].median()),
                "ftv_descent_margin_mean": float(batch["ftv_descent_margin"].mean()),
                "n_batches": len(batch),
                "n_undefined": undefined,
                "source_contract_sha256": str(first["source_contract_sha256"]),
                "public_manifest_sha256": str(first["public_manifest_sha256"]),
                "contains_patient_ids": False,
            }
        )
    result = (
        pd.DataFrame(rows, columns=LAYER_LEVEL_COLUMNS)
        .sort_values(["seed_base", "fold", "checkpoint_kind", "split", "group"])
        .reset_index(drop=True)
    )
    return result


def _validate_aggregate_table(
    frame: pd.DataFrame,
    *,
    expected_rows: int,
    expected_kinds: set[str],
    expected_runs: set[tuple[int, int]],
    name: str,
) -> pd.DataFrame:
    _require_columns(frame, LAYER_LEVEL_COLUMNS, name)
    if (
        len(frame) != expected_rows
        or frame.duplicated(
            ["seed_base", "fold", "checkpoint_kind", "split", "group"]
        ).any()
    ):
        raise ValueError(f"{name} row/key coverage 错误")
    expected_keys = {
        (seed, fold, kind, split, group)
        for seed, fold in expected_runs
        for kind in expected_kinds
        for split in SPLITS
        for group in GROUPS
    }
    observed_keys = set(
        frame[["seed_base", "fold", "checkpoint_kind", "split", "group"]].itertuples(
            index=False, name=None
        )
    )
    if observed_keys != expected_keys:
        raise ValueError(f"{name} Cartesian product 错误")
    _require_finite(frame, (*AGGREGATE_VALUE_COLUMNS, "base_degradation"), name)
    if set(frame["n_batches"]) != {BATCHES_PER_SPLIT} or set(frame["n_undefined"]) != {
        0
    }:
        raise ValueError(f"{name} n_batches/n_undefined contract 失败")
    fraction_columns = (
        "negative_fraction",
        "strong_negative_fraction",
        "very_strong_negative_fraction",
        "base_descent_failure_fraction",
    )
    if any(not frame[column].between(0, 1).all() for column in fraction_columns):
        raise ValueError(f"{name} fraction 越界")
    _require_sha(frame["checkpoint_sha256"], f"{name}.checkpoint_sha256")
    _require_sha(frame["source_contract_sha256"], f"{name}.source_contract_sha256")
    _require_sha(frame["public_manifest_sha256"], f"{name}.public_manifest_sha256")
    _assert_public_frame(frame, name)
    return frame


def aggregate_selected(
    selected: pd.DataFrame,
    existing: pd.DataFrame,
    *,
    expected_public_manifest_sha256: str | None = None,
    expected_source_contract_sha256: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """验证 2800 行 selected gradients，并产生 350 layer + 50 all-shared run 行。"""

    existing = _validate_existing(existing)
    selected = _validate_batch_matrix(
        selected,
        checkpoint_kind="selected",
        expected_runs=_expected_run_set(),
        expected_rows=SELECTED_BATCH_ROWS,
        name="selected batch gradients",
        expected_public_manifest_sha256=expected_public_manifest_sha256,
        expected_source_contract_sha256=expected_source_contract_sha256,
    )
    layer = _aggregate_groups(selected)
    _validate_aggregate_table(
        layer,
        expected_rows=LAYER_LEVEL_ROWS,
        expected_kinds={"selected"},
        expected_runs=_expected_run_set(),
        name="layer-level conflict metrics",
    )

    existing_lookup = existing.set_index(["seed_base", "fold"])
    run_rows: list[dict[str, Any]] = []
    for row in layer.loc[layer["group"].eq("all_shared")].itertuples(index=False):
        source = existing_lookup.loc[(int(row.seed_base), int(row.fold))]
        if (
            int(row.checkpoint_epoch) != int(source.selected_epoch)
            or int(row.effective_seed) != int(source.effective_seed)
            or str(row.base_gate) != str(source.base_gate)
            or not math.isclose(
                float(row.base_degradation),
                float(source.base_degradation),
                abs_tol=1e-12,
            )
        ):
            raise ValueError("all-shared aggregate 与 existing metrics 不闭环")
        payload = row._asdict()
        payload.update(
            {column: source[column] for column in EXISTING_AUXILIARY_COLUMNS}
        )
        run_rows.append(payload)
    run = (
        pd.DataFrame(run_rows, columns=RUN_LEVEL_COLUMNS)
        .sort_values(["seed_base", "fold", "split"])
        .reset_index(drop=True)
    )
    _require_columns(run, RUN_LEVEL_COLUMNS, "run-level conflict metrics")
    if (
        len(run) != RUN_LEVEL_ROWS
        or run.duplicated(["seed_base", "fold", "split"]).any()
    ):
        raise ValueError("run-level all-shared 50-row key coverage 错误")
    expected_keys = {
        (seed, fold, split) for seed, fold in _expected_run_set() for split in SPLITS
    }
    if (
        set(run[["seed_base", "fold", "split"]].itertuples(index=False, name=None))
        != expected_keys
    ):
        raise ValueError("run-level all-shared Cartesian product 错误")
    if set(run["group"]) != {"all_shared"} or set(run["checkpoint_kind"]) != {
        "selected"
    }:
        raise ValueError("run-level 表不是 selected/all_shared")
    _require_finite(
        run,
        [
            column
            for column in RUN_LEVEL_COLUMNS
            if column
            not in {
                "selection_mode",
                "base_gate",
                "checkpoint_kind",
                "checkpoint_sha256",
                "split",
                "group",
                "source_contract_sha256",
                "public_manifest_sha256",
                "contains_patient_ids",
                "base_gate_pass",
            }
        ],
        "run-level conflict metrics",
    )
    _assert_public_frame(run, "run-level conflict metrics")
    return layer, run


def aggregate_trajectory(
    selected: pd.DataFrame,
    last: pd.DataFrame,
    representatives: pd.DataFrame,
    existing: pd.DataFrame,
    *,
    expected_public_manifest_sha256: str | None = None,
    expected_source_contract_sha256: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """聚合六个预选代表 run 的 selected+last，并计算 84 个 last-selected change。"""

    existing = _validate_existing(existing)
    representatives = _validate_representatives(representatives, existing)
    representative_runs = set(
        representatives[["seed_base", "fold"]].itertuples(index=False, name=None)
    )
    selected = _validate_batch_matrix(
        selected,
        checkpoint_kind="selected",
        expected_runs=_expected_run_set(),
        expected_rows=SELECTED_BATCH_ROWS,
        name="selected batch gradients",
        expected_public_manifest_sha256=expected_public_manifest_sha256,
        expected_source_contract_sha256=expected_source_contract_sha256,
    )
    last = _validate_batch_matrix(
        last,
        checkpoint_kind="last",
        expected_runs=representative_runs,
        expected_rows=TRAJECTORY_LAST_BATCH_ROWS,
        name="trajectory last batch gradients",
        expected_public_manifest_sha256=expected_public_manifest_sha256,
        expected_source_contract_sha256=expected_source_contract_sha256,
    )
    selected_subset = selected.merge(
        representatives[["seed_base", "fold"]],
        on=["seed_base", "fold"],
        how="inner",
        validate="many_to_one",
    )
    if len(selected_subset) != TRAJECTORY_LAST_BATCH_ROWS:
        raise ValueError("representative selected batch subset 不是672行")

    pairing = ["seed_base", "fold", "split", "batch_index", "group"]
    selected_pair = selected_subset.set_index(pairing)
    last_pair = last.set_index(pairing)
    if set(selected_pair.index) != set(last_pair.index):
        raise ValueError("trajectory selected/last batch pairing 不完整")
    for column in (
        "batch_id",
        "ordered_members_hmac_sha256",
        "n_total",
        "n_ftv_available",
        "stochastic_seed",
        "public_manifest_sha256",
        "source_contract_sha256",
    ):
        if (
            not selected_pair[column]
            .astype(str)
            .sort_index()
            .equals(last_pair[column].astype(str).sort_index())
        ):
            raise ValueError(
                f"trajectory selected/last 固定 batch invariant 漂移: {column}"
            )

    combined = pd.concat([selected_subset, last], ignore_index=True)
    trajectory = _aggregate_groups(combined)
    _validate_aggregate_table(
        trajectory,
        expected_rows=TRAJECTORY_ROWS,
        expected_kinds={"selected", "last"},
        expected_runs=representative_runs,
        name="trajectory conflict metrics",
    )
    representative_lookup = representatives.set_index(["seed_base", "fold"])
    for row in trajectory.itertuples(index=False):
        source = representative_lookup.loc[(int(row.seed_base), int(row.fold))]
        expected_epoch = int(
            source.selected_epoch
            if row.checkpoint_kind == "selected"
            else source.last_epoch
        )
        if int(row.checkpoint_epoch) != expected_epoch:
            raise ValueError(
                "trajectory checkpoint epoch 与 representative contract 不一致"
            )

    index = ["seed_base", "fold", "split", "group"]
    selected_level = trajectory.loc[
        trajectory["checkpoint_kind"].eq("selected")
    ].set_index(index)
    last_level = trajectory.loc[trajectory["checkpoint_kind"].eq("last")].set_index(
        index
    )
    if set(selected_level.index) != set(last_level.index):
        raise ValueError("trajectory aggregate selected/last pairing 不完整")
    changes: list[dict[str, Any]] = []
    for key in sorted(selected_level.index):
        before = selected_level.loc[key]
        after = last_level.loc[key]
        invariant = (
            "effective_seed",
            "base_gate",
            "base_degradation",
            "parameter_tensors",
            "parameter_count",
            "source_contract_sha256",
            "public_manifest_sha256",
        )
        if any(str(before[column]) != str(after[column]) for column in invariant):
            raise ValueError(f"trajectory aggregate invariant 漂移: {key}")
        payload: dict[str, Any] = {
            "schema_version": 1,
            "seed_base": int(key[0]),
            "fold": int(key[1]),
            "effective_seed": int(before["effective_seed"]),
            "base_gate": str(before["base_gate"]),
            "base_degradation": float(before["base_degradation"]),
            "split": str(key[2]),
            "group": str(key[3]),
            "parameter_tensors": int(before["parameter_tensors"]),
            "parameter_count": int(before["parameter_count"]),
            "selected_epoch": int(before["checkpoint_epoch"]),
            "last_epoch": int(after["checkpoint_epoch"]),
            "selected_checkpoint_sha256": str(before["checkpoint_sha256"]),
            "last_checkpoint_sha256": str(after["checkpoint_sha256"]),
        }
        payload.update(
            {
                f"last_minus_selected_{column}": float(after[column])
                - float(before[column])
                for column in AGGREGATE_VALUE_COLUMNS
            }
        )
        payload.update(
            {
                "selected_n_batches": int(before["n_batches"]),
                "last_n_batches": int(after["n_batches"]),
                "selected_n_undefined": int(before["n_undefined"]),
                "last_n_undefined": int(after["n_undefined"]),
                "source_contract_sha256": str(before["source_contract_sha256"]),
                "public_manifest_sha256": str(before["public_manifest_sha256"]),
                "contains_patient_ids": False,
            }
        )
        changes.append(payload)
    change = (
        pd.DataFrame(changes, columns=TRAJECTORY_CHANGE_COLUMNS)
        .sort_values(index)
        .reset_index(drop=True)
    )
    _require_columns(change, TRAJECTORY_CHANGE_COLUMNS, "trajectory change metrics")
    if len(change) != TRAJECTORY_CHANGE_ROWS or change.duplicated(index).any():
        raise ValueError("trajectory change 84-row key coverage 错误")
    expected_change_keys = {
        (seed, fold, split, group)
        for seed, fold in representative_runs
        for split in SPLITS
        for group in GROUPS
    }
    if set(change[index].itertuples(index=False, name=None)) != expected_change_keys:
        raise ValueError("trajectory change Cartesian product 错误")
    _require_finite(
        change,
        (
            "base_degradation",
            *(f"last_minus_selected_{column}" for column in AGGREGATE_VALUE_COLUMNS),
        ),
        "trajectory change metrics",
    )
    if (
        set(change["selected_n_batches"]) != {8}
        or set(change["last_n_batches"]) != {8}
        or set(change["selected_n_undefined"]) != {0}
        or set(change["last_n_undefined"]) != {0}
    ):
        raise ValueError("trajectory change batch/undefined contract 失败")
    _assert_public_frame(change, "trajectory change metrics")
    return trajectory, change


def _exposure_summary(
    counts: pd.Series, population_n: int
) -> tuple[float, float, float, float, float]:
    if population_n <= 0 or len(counts) > population_n:
        raise ValueError("coverage exposure population 非法")
    values = counts.to_numpy(dtype=float)
    if len(values) < population_n:
        values = np.concatenate(
            [values, np.zeros(population_n - len(values), dtype=float)]
        )
    if not np.isfinite(values).all():
        raise ValueError("coverage exposure nonfinite")
    return (
        float(np.min(values)),
        float(np.median(values)),
        float(np.mean(values)),
        float(np.std(values, ddof=1)),
        float(np.max(values)),
    )


def aggregate_coverage(public: pd.DataFrame, private: pd.DataFrame) -> pd.DataFrame:
    """由 manifest pair 产生10行无标识 coverage 表；不输出 batch/member HMAC。"""

    _require_columns(public, PUBLIC_MANIFEST_COLUMNS, "public batch manifest")
    _require_columns(private, PRIVATE_MANIFEST_COLUMNS, "private batch membership")
    if len(public) != 80 or len(private) != 2_560:
        raise ValueError("manifest row count 不是80/2560")
    expected_public_keys = {
        (fold, split, batch_index)
        for fold in FOLDS
        for split in SPLITS
        for batch_index in range(BATCHES_PER_SPLIT)
    }
    if (
        set(public[["fold", "split", "batch_index"]].itertuples(index=False, name=None))
        != expected_public_keys
    ):
        raise ValueError("public manifest Cartesian key coverage 错误")
    if (
        public.duplicated(["fold", "split", "batch_index"]).any()
        or private.duplicated(["batch_id", "position"]).any()
    ):
        raise ValueError("manifest key 重复")
    if (
        _strict_bool_series(
            public["contains_patient_ids"], "public.contains_patient_ids"
        ).any()
        or _strict_bool_series(
            public["contains_patient_level_rows"], "public.contains_patient_level_rows"
        ).any()
        or _strict_bool_series(
            public["within_batch_replacement"], "within_batch_replacement"
        ).any()
    ):
        raise ValueError("public manifest privacy/replacement flag 失败")
    _require_sha(public["ordered_members_hmac_sha256"], "ordered_members_hmac_sha256")
    _require_sha(public["private_mapping_hmac_sha256"], "private_mapping_hmac_sha256")
    for column in (
        "plan_freeze_sha256",
        "audit_config_sha256",
        "source_fold_manifest_sha256",
        "source_raw_ftv_semantic_sha256",
    ):
        _require_sha(public[column], column)
    has_ftv = _strict_bool_series(private["has_ftv"], "private.has_ftv")
    private = private.copy().assign(has_ftv=has_ftv)
    if (
        private["patient_id"].isna().any()
        or private["patient_id"].astype(str).eq("").any()
    ):
        raise ValueError("private manifest patient_id 缺失")
    if private.groupby("patient_id")["has_ftv"].nunique(dropna=False).gt(1).any():
        raise ValueError("private manifest 同一 member 的 FTV availability 漂移")

    rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        for split in SPLITS:
            selected_public = public.loc[
                public["fold"].eq(fold) & public["split"].eq(split)
            ].sort_values("batch_index")
            selected_private = private.loc[
                private["fold"].eq(fold) & private["split"].eq(split)
            ].copy()
            if len(selected_public) != 8 or len(selected_private) != 256:
                raise ValueError(f"manifest fold/split coverage 错误: {fold}/{split}")
            invariant = (
                "plan_freeze_sha256",
                "audit_config_sha256",
                "source_fold_manifest_sha256",
                "source_raw_ftv_semantic_sha256",
                "pool_n",
                "pool_ftv_available",
                "pool_ftv_proportion",
            )
            if any(
                selected_public[column].nunique(dropna=False) != 1
                for column in invariant
            ):
                raise ValueError(
                    f"public manifest pool/source invariant 漂移: {fold}/{split}"
                )
            for manifest in selected_public.itertuples(index=False):
                batch = selected_private.loc[
                    selected_private["batch_id"].eq(str(manifest.batch_id))
                ]
                if (
                    len(batch) != BATCH_SIZE
                    or batch["patient_id"].duplicated().any()
                    or set(batch["position"].astype(int)) != set(range(BATCH_SIZE))
                    or set(batch["batch_index"].astype(int))
                    != {int(manifest.batch_index)}
                ):
                    raise ValueError(
                        f"private batch size/key/uniqueness 错误: {manifest.batch_id}"
                    )
                grounded = int(batch["has_ftv"].sum())
                if (
                    grounded != int(manifest.n_ftv_available)
                    or int(manifest.n_total) != BATCH_SIZE
                    or int(manifest.n_unavailable) != BATCH_SIZE - grounded
                    or not math.isclose(
                        float(manifest.batch_ftv_proportion),
                        grounded / BATCH_SIZE,
                        abs_tol=1e-12,
                    )
                ):
                    raise ValueError(
                        f"public/private grounded counts 不闭环: {manifest.batch_id}"
                    )
            member_counts = selected_private["patient_id"].astype(str).value_counts()
            ftv_counts = (
                selected_private.loc[selected_private["has_ftv"], "patient_id"]
                .astype(str)
                .value_counts()
            )
            pool_n = int(selected_public.iloc[0]["pool_n"])
            pool_ftv = int(selected_public.iloc[0]["pool_ftv_available"])
            if (
                len(member_counts) > pool_n
                or len(ftv_counts) > pool_ftv
                or pool_ftv <= 0
            ):
                raise ValueError(
                    f"private unique members 超出 public pool: {fold}/{split}"
                )
            member_summary = _exposure_summary(member_counts, pool_n)
            ftv_summary = _exposure_summary(ftv_counts, pool_ftv)
            batch_ftv = selected_public["n_ftv_available"].to_numpy(dtype=float)
            first = selected_public.iloc[0]
            ftv_draws = int(batch_ftv.sum())
            rows.append(
                {
                    "schema_version": 1,
                    "fold": fold,
                    "split": split,
                    "plan_freeze_sha256": str(first["plan_freeze_sha256"]),
                    "audit_config_sha256": str(first["audit_config_sha256"]),
                    "source_fold_manifest_sha256": str(
                        first["source_fold_manifest_sha256"]
                    ),
                    "source_raw_ftv_semantic_sha256": str(
                        first["source_raw_ftv_semantic_sha256"]
                    ),
                    "n_batches": len(selected_public),
                    "batch_size": BATCH_SIZE,
                    "pool_n": pool_n,
                    "pool_ftv_available": pool_ftv,
                    "pool_ftv_proportion": float(first["pool_ftv_proportion"]),
                    "total_draws": len(selected_private),
                    "ftv_draws": ftv_draws,
                    "ftv_draw_proportion": ftv_draws / len(selected_private),
                    "batch_ftv_min": float(np.min(batch_ftv)),
                    "batch_ftv_median": float(np.median(batch_ftv)),
                    "batch_ftv_mean": float(np.mean(batch_ftv)),
                    "batch_ftv_sd": float(np.std(batch_ftv, ddof=1)),
                    "batch_ftv_max": float(np.max(batch_ftv)),
                    "unique_members_drawn": len(member_counts),
                    "unique_ftv_members_drawn": len(ftv_counts),
                    "pool_member_coverage_fraction": len(member_counts) / pool_n,
                    "pool_ftv_member_coverage_fraction": len(ftv_counts) / pool_ftv,
                    "member_exposure_min": member_summary[0],
                    "member_exposure_median": member_summary[1],
                    "member_exposure_mean": member_summary[2],
                    "member_exposure_sd": member_summary[3],
                    "member_exposure_max": member_summary[4],
                    "ftv_member_exposure_min": ftv_summary[0],
                    "ftv_member_exposure_median": ftv_summary[1],
                    "ftv_member_exposure_mean": ftv_summary[2],
                    "ftv_member_exposure_sd": ftv_summary[3],
                    "ftv_member_exposure_max": ftv_summary[4],
                    "contains_patient_ids": False,
                }
            )
    result = (
        pd.DataFrame(rows, columns=COVERAGE_COLUMNS)
        .sort_values(["fold", "split"])
        .reset_index(drop=True)
    )
    _require_columns(result, COVERAGE_COLUMNS, "FTV coverage metrics")
    if len(result) != COVERAGE_ROWS or result.duplicated(["fold", "split"]).any():
        raise ValueError("FTV coverage 10-row key coverage 错误")
    _require_finite(
        result,
        [
            column
            for column in COVERAGE_COLUMNS
            if column
            not in {
                "split",
                "plan_freeze_sha256",
                "audit_config_sha256",
                "source_fold_manifest_sha256",
                "source_raw_ftv_semantic_sha256",
                "contains_patient_ids",
            }
        ],
        "FTV coverage metrics",
    )
    if not np.allclose(
        result["pool_ftv_proportion"],
        result["pool_ftv_available"] / result["pool_n"],
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError("FTV coverage pool proportion 不可重算")
    _assert_public_frame(
        result,
        "FTV coverage metrics",
        private_identifiers=private["patient_id"].astype(str),
    )
    return result


def build_aggregate_tables(root: str | Path = AUDIT_ROOT) -> dict[str, pd.DataFrame]:
    """读取并严格验证正式输入；仅返回表，不写文件。"""

    root = Path(root).resolve()
    if root != AUDIT_ROOT.resolve():
        raise ValueError("formal aggregate root 必须是冻结 audit root")
    assert_source_contract(full_content_hash=False, full_checkpoint_hash=False)
    public, private = validate_manifests(load_data_context())
    expected_public_sha = file_sha256(PUBLIC_MANIFEST)
    expected_source_sha = file_sha256(SOURCE_CONTRACT)
    selected = pd.read_csv(root / "metrics" / "batch_gradient_metrics.csv")
    last = pd.read_csv(root / "metrics" / "trajectory_batch_gradient_metrics.csv")
    existing = pd.read_csv(root / "metrics" / "run_level_existing_metrics.csv")
    representatives = pd.read_csv(root / "metrics" / "representative_runs.csv")
    layer, run = aggregate_selected(
        selected,
        existing,
        expected_public_manifest_sha256=expected_public_sha,
        expected_source_contract_sha256=expected_source_sha,
    )
    trajectory, change = aggregate_trajectory(
        selected,
        last,
        representatives,
        existing,
        expected_public_manifest_sha256=expected_public_sha,
        expected_source_contract_sha256=expected_source_sha,
    )
    coverage = aggregate_coverage(public, private)
    return {
        "layer_level_conflict_metrics": layer,
        "run_level_conflict_metrics": run,
        "trajectory_conflict_metrics": trajectory,
        "trajectory_change_metrics": change,
        "ftv_coverage_metrics": coverage,
    }


def write_aggregate_tables(
    root: str | Path = AUDIT_ROOT, *, overwrite: bool = False
) -> dict[str, int]:
    """验证全部输入后原子写出五张公开聚合表。"""

    root = Path(root).resolve()
    tables = build_aggregate_tables(root)
    destinations = {name: root / "metrics" / f"{name}.csv" for name in tables}
    if not overwrite and any(path.exists() for path in destinations.values()):
        existing = [path.name for path in destinations.values() if path.exists()]
        raise FileExistsError(f"拒绝部分写入；聚合输出已存在: {existing}")
    for stem, frame in tables.items():
        atomic_csv(
            destinations[stem],
            frame.to_dict(orient="records"),
            fieldnames=tuple(frame.columns),
            overwrite=overwrite,
        )
    return {name: len(frame) for name, frame in tables.items()}


def _synthetic_existing() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    runs = sorted(_expected_run_set())
    for index, (seed, fold) in enumerate(runs):
        passed = index < 17
        degradation = -0.02 + index * 0.002 if passed else 0.06 + (index - 17) * 0.01
        row: dict[str, Any] = {
            "seed_base": seed,
            "fold": fold,
            "effective_seed": seed + fold,
            "selected_epoch": 2 + index % 2,
            "last_epoch": 7,
            "last_minus_selected_epoch": 7 - (2 + index % 2),
            "selection_mode": "primary" if passed else "fallback",
            "base_degradation": degradation,
            "base_gate_pass": passed,
            "base_gate": "PASS" if passed else "FAIL",
        }
        numeric_tail = [
            column
            for column in EXISTING_METRIC_COLUMNS
            if column not in row
            and column not in {"selection_mode", "base_gate", "base_gate_pass"}
        ]
        row.update(
            {
                column: 0.1 + index * 0.001 + offset * 0.0001
                for offset, column in enumerate(numeric_tail)
            }
        )
        row["history_rows"] = 6 + index % 2
        rows.append(row)
    existing = pd.DataFrame(rows, columns=EXISTING_METRIC_COLUMNS)
    representatives = pd.DataFrame(
        [
            {
                "base_gate": existing.iloc[index]["base_gate"],
                "selection_rule": f"synthetic_{rank}",
                "within_group_rank_zero_based": rank,
                "seed_base": int(existing.iloc[index]["seed_base"]),
                "fold": int(existing.iloc[index]["fold"]),
                "base_degradation": float(existing.iloc[index]["base_degradation"]),
                "selected_epoch": int(existing.iloc[index]["selected_epoch"]),
                "last_epoch": int(existing.iloc[index]["last_epoch"]),
                "gradient_result_used_for_selection": False,
            }
            for rank, index in enumerate((0, 8, 16, 17, 20, 24))
        ],
        columns=REPRESENTATIVE_COLUMNS,
    )
    return existing, representatives


def _synthetic_gradient_rows(
    existing: pd.DataFrame,
    runs: Sequence[tuple[int, int]],
    checkpoint_kind: str,
) -> pd.DataFrame:
    lookup = existing.set_index(["seed_base", "fold"])
    rows: list[dict[str, Any]] = []
    for run_index, (seed, fold) in enumerate(runs):
        source = lookup.loc[(seed, fold)]
        for split_index, split in enumerate(SPLITS):
            for batch_index in range(BATCHES_PER_SPLIT):
                batch_hash = f"{fold * 16 + split_index * 8 + batch_index + 1:064x}"
                for group_index, group in enumerate(GROUPS):
                    cosine = (
                        -0.30
                        + 0.02 * batch_index
                        + 0.005 * group_index
                        + 0.0001 * run_index
                    )
                    if checkpoint_kind == "last":
                        cosine -= 0.05
                    base_norm = 2.0 + 0.01 * group_index
                    ftv_norm = 3.0 + 0.01 * batch_index
                    dot = cosine * base_norm * ftv_norm
                    weighted = LAMBDA_FTV * ftv_norm
                    ratio = weighted / base_norm
                    m_base = 1.0 + LAMBDA_FTV * dot / base_norm**2
                    m_ftv = (dot + LAMBDA_FTV * ftv_norm**2) / (
                        LAMBDA_FTV * ftv_norm**2
                    )
                    tensors, parameters = EXPECTED_GROUP_COUNTS[group]
                    epoch = int(
                        source.selected_epoch
                        if checkpoint_kind == "selected"
                        else source.last_epoch
                    )
                    row = {column: 0 for column in BATCH_GRADIENT_COLUMNS}
                    row.update(
                        {
                            "schema_version": 1,
                            "seed_base": seed,
                            "fold": fold,
                            "effective_seed": seed + fold,
                            "base_gate": str(source.base_gate),
                            "base_degradation": float(source.base_degradation),
                            "checkpoint_kind": checkpoint_kind,
                            "checkpoint_epoch": epoch,
                            "checkpoint_sha256": f"{seed + fold + (1 if checkpoint_kind == 'selected' else 10):064x}",
                            "split": split,
                            "batch_id": f"f{fold}_{'tr' if split == 'train' else 'va'}_{batch_index:02d}",
                            "batch_index": batch_index,
                            "ordered_members_hmac_sha256": batch_hash,
                            "n_total": BATCH_SIZE,
                            "n_ftv_available": 12,
                            "n_ftv_valid_visits": 24,
                            "group": group,
                            "parameter_tensors": tensors,
                            "parameter_count": parameters,
                            "base_gradient_none_count": 0,
                            "ftv_gradient_none_count": 0,
                            "base_gradient_norm": base_norm,
                            "ftv_gradient_norm_raw": ftv_norm,
                            "weighted_ftv_gradient_norm": weighted,
                            "gradient_dot_raw": dot,
                            "gradient_cosine": cosine,
                            "weighted_gradient_norm_ratio": ratio,
                            "base_descent_margin": m_base,
                            "ftv_descent_margin": m_ftv,
                            "negative_cosine": cosine < 0,
                            "strong_negative_cosine": cosine < -0.1,
                            "very_strong_negative_cosine": cosine < -0.25,
                            "base_descent_failure": m_base < 0,
                            "base_objective": 1.0,
                            "state_loss": 0.9,
                            "sigreg_loss": 1.0,
                            "ftv_loss_raw": 0.5,
                            "lambda_ftv": LAMBDA_FTV,
                            "stochastic_seed": 20260808
                            + fold * 100_000
                            + split_index * 10_000
                            + batch_index,
                            "model_mode": "train_fixed_rng",
                            "deterministic_algorithms": True,
                            "paired_forward_outputs_exact": True,
                            "component_forward_backward_count": 2,
                            "optimizer_created": False,
                            "optimizer_step": False,
                            "pcr_signal_used": False,
                            "model_state_sha256_before": f"{seed + fold + 100:064x}",
                            "model_state_sha256_after": f"{seed + fold + 100:064x}",
                            "source_contract_sha256": "a" * 64,
                            "public_manifest_sha256": "b" * 64,
                            "contains_patient_ids": False,
                        }
                    )
                    rows.append(row)
    return pd.DataFrame(rows, columns=BATCH_GRADIENT_COLUMNS)


def _synthetic_manifests() -> tuple[pd.DataFrame, pd.DataFrame]:
    public_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        for split_index, split in enumerate(SPLITS):
            pool_n = 100
            pool_ftv = 40
            for batch_index in range(BATCHES_PER_SPLIT):
                batch_id = (
                    f"f{fold}_{'tr' if split == 'train' else 'va'}_{batch_index:02d}"
                )
                for position in range(BATCH_SIZE):
                    has_ftv = position < 12
                    member_index = (
                        (batch_index * 12 + position) % pool_ftv
                        if has_ftv
                        else pool_ftv
                        + (batch_index * 20 + position - 12) % (pool_n - pool_ftv)
                    )
                    private_rows.append(
                        {
                            "batch_id": batch_id,
                            "fold": fold,
                            "split": split,
                            "batch_index": batch_index,
                            "position": position,
                            "patient_id": f"SYN_F{fold}_{split_index}_{member_index:03d}",
                            "cohort_role": "ispy2_ftv" if has_ftv else "ispy2_noftv",
                            "has_ftv": has_ftv,
                        }
                    )
                public_rows.append(
                    {
                        "schema_version": 1,
                        "plan_freeze_sha256": "1" * 64,
                        "audit_config_sha256": "2" * 64,
                        "source_fold_manifest_sha256": "3" * 64,
                        "source_raw_ftv_semantic_sha256": "4" * 64,
                        "batch_id": batch_id,
                        "fold": fold,
                        "split": split,
                        "batch_index": batch_index,
                        "ordered_members_hmac_sha256": f"{fold * 16 + split_index * 8 + batch_index + 1:064x}",
                        "hmac_key_id": "synthetic-key-id",
                        "n_total": 32,
                        "n_ftv_available": 12,
                        "n_unavailable": 20,
                        "n_ispy2": 32,
                        "n_ispy1": 0,
                        "pool_n": pool_n,
                        "pool_ftv_available": pool_ftv,
                        "pool_ftv_proportion": pool_ftv / pool_n,
                        "batch_ftv_proportion": 12 / 32,
                        "applies_to_seed_count": 5,
                        "within_batch_replacement": False,
                        "contains_patient_ids": False,
                        "contains_patient_level_rows": False,
                        "private_mapping_hmac_sha256": "5" * 64,
                    }
                )
    return (
        pd.DataFrame(public_rows, columns=PUBLIC_MANIFEST_COLUMNS),
        pd.DataFrame(private_rows, columns=PRIVATE_MANIFEST_COLUMNS),
    )


def synthetic_self_test() -> dict[str, Any]:
    """只使用合成表验证 row/key、聚合数学、trajectory pairing 与 privacy fail-closed。"""

    existing, representatives = _synthetic_existing()
    selected = _synthetic_gradient_rows(
        existing, sorted(_expected_run_set()), "selected"
    )
    representative_runs = list(
        representatives[["seed_base", "fold"]].itertuples(index=False, name=None)
    )
    last = _synthetic_gradient_rows(existing, representative_runs, "last")
    public, private = _synthetic_manifests()
    layer, run = aggregate_selected(selected, existing)
    trajectory, change = aggregate_trajectory(selected, last, representatives, existing)
    coverage = aggregate_coverage(public, private)
    probe = layer.loc[
        layer["seed_base"].eq(SEED_BASES[0])
        & layer["fold"].eq(FOLDS[0])
        & layer["split"].eq("train")
        & layer["group"].eq("all_shared")
    ].iloc[0]
    source_probe = selected.loc[
        selected["seed_base"].eq(SEED_BASES[0])
        & selected["fold"].eq(FOLDS[0])
        & selected["split"].eq("train")
        & selected["group"].eq("all_shared")
    ]
    change_probe = change.loc[
        change["seed_base"].eq(int(representatives.iloc[0]["seed_base"]))
        & change["fold"].eq(int(representatives.iloc[0]["fold"]))
        & change["split"].eq("train")
        & change["group"].eq("all_shared")
    ].iloc[0]
    checks = {
        "selected_input_rows_2800": len(selected) == SELECTED_BATCH_ROWS,
        "last_input_rows_672": len(last) == TRAJECTORY_LAST_BATCH_ROWS,
        "layer_rows_350": len(layer) == LAYER_LEVEL_ROWS,
        "run_rows_50_all_shared": len(run) == RUN_LEVEL_ROWS
        and set(run["group"]) == {"all_shared"},
        "trajectory_rows_168": len(trajectory) == TRAJECTORY_ROWS,
        "trajectory_change_rows_84": len(change) == TRAJECTORY_CHANGE_ROWS,
        "coverage_rows_10": len(coverage) == COVERAGE_ROWS,
        "coverage_sample_sd_present": "batch_ftv_sd" in coverage
        and np.isfinite(coverage["batch_ftv_sd"]).all()
        and np.allclose(coverage["batch_ftv_sd"], 0.0),
        "median_exact": math.isclose(
            float(probe.gradient_cosine),
            float(source_probe["gradient_cosine"].median()),
            abs_tol=1e-12,
        ),
        "mean_exact": math.isclose(
            float(probe.gradient_cosine_mean),
            float(source_probe["gradient_cosine"].mean()),
            abs_tol=1e-12,
        ),
        "trajectory_change_exact": math.isclose(
            float(change_probe.last_minus_selected_gradient_cosine),
            -0.05,
            abs_tol=1e-12,
        ),
        "undefined_zero": set(layer["n_undefined"]) == {0},
        "public_outputs_no_hmac": all(
            not any("hmac" in column.lower() for column in frame.columns)
            for frame in (layer, run, trajectory, change, coverage)
        ),
        "public_outputs_no_patient_ids": all(
            not _strict_bool_series(
                frame["contains_patient_ids"], "self-test privacy"
            ).any()
            for frame in (layer, run, trajectory, change, coverage)
        ),
    }
    malformed = selected.iloc[:-1].copy()
    try:
        aggregate_selected(malformed, existing)
    except ValueError:
        checks["missing_row_rejected"] = True
    else:
        checks["missing_row_rejected"] = False
    leaked = coverage.copy()
    leaked["patient_id"] = private.iloc[0]["patient_id"]
    try:
        _assert_public_frame(
            leaked, "synthetic leak", private_identifiers=private["patient_id"]
        )
    except ValueError:
        checks["patient_leak_rejected"] = True
    else:
        checks["patient_leak_rejected"] = False
    if not all(checks.values()):
        raise AssertionError(f"aggregation synthetic self-test 失败: {checks}")
    return {"status": "ok", "checks": checks}


__all__ = [
    "AGGREGATE_VALUE_COLUMNS",
    "COVERAGE_COLUMNS",
    "EXISTING_METRIC_COLUMNS",
    "LAYER_LEVEL_COLUMNS",
    "RUN_LEVEL_COLUMNS",
    "TRAJECTORY_CHANGE_COLUMNS",
    "TRAJECTORY_COLUMNS",
    "aggregate_coverage",
    "aggregate_selected",
    "aggregate_trajectory",
    "build_aggregate_tables",
    "synthetic_self_test",
    "write_aggregate_tables",
]

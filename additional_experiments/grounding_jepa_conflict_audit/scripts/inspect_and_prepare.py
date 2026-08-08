#!/usr/bin/env python3
"""核验冻结资产，并生成 patient-free 固定 audit batch manifest。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gjca.aggregation import (
    EXISTING_METRIC_COLUMNS,
    REPRESENTATIVE_COLUMNS,
)  # noqa: E402
from gjca.assets import (  # noqa: E402
    fold_transform,
    load_data_context,
    validate_checkpoint_grid,
    validate_trajectory_assets,
)
from gjca.batches import (  # noqa: E402
    PRIVATE_HMAC_KEY,
    PRIVATE_MEMBERSHIP,
    PUBLIC_MANIFEST,
    synthetic_self_test,
    write_manifests,
)
from gjca.contracts import atomic_csv, file_sha256  # noqa: E402
from gjca.freeze import SOURCE_CONTRACT, assert_plan_freeze  # noqa: E402
from gjca.phase_a import (  # noqa: E402
    PHASE_A_COLUMNS,
    PHASE_A_OUTPUT,
    RESAMPLING_BUNDLE,
    RESAMPLING_MANIFEST,
    load_resampling_bundle,
)
from gjca.source_contract import (  # noqa: E402
    PRIVATE_CACHE_MANIFEST,
    PUBLIC_CACHE_CONTRACT,
    SOURCE_MANIFEST,
    write_cache_input_contract,
    write_source_contract,
    write_source_manifest,
)


EXISTING_METRICS = ROOT / "metrics" / "run_level_existing_metrics.csv"
TRAINING_HISTORY = ROOT / "metrics" / "training_history_audit.csv"
REPRESENTATIVE_RUNS = ROOT / "metrics" / "representative_runs.csv"
ASSET_MANIFEST = ROOT / "metrics" / "asset_manifest.csv"

TRAINING_HISTORY_COLUMNS = (
    "seed_base",
    "fold",
    "base_gate",
    "base_degradation",
    "epoch",
    "is_selected_checkpoint",
    "train_total_loss",
    "train_base_loss",
    "train_state_loss",
    "train_sigreg_loss",
    "train_ftv_loss",
    "train_weighted_ftv_loss",
    "val_state_loss",
    "val_base_objective",
    "val_ftv_loss",
    "representation_std",
    "grounded_exposure",
)

PHASE_A_ENDPOINTS = {
    "degradation_vs_static_delta_spearman",
    "degradation_vs_observed_delta_spearman",
    "degradation_vs_observed_delta_r2",
}

INSPECTION_DESTINATIONS = (
    ASSET_MANIFEST,
    SOURCE_MANIFEST,
    PUBLIC_MANIFEST,
    PRIVATE_MEMBERSHIP,
    PRIVATE_HMAC_KEY,
    PRIVATE_CACHE_MANIFEST,
    PUBLIC_CACHE_CONTRACT,
    SOURCE_CONTRACT,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def _strict_bool(value: object, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ValueError(f"{name} 非严格 boolean: {value!r}")


def _preflight_phase_a() -> pd.DataFrame:
    prerequisites = (
        EXISTING_METRICS,
        TRAINING_HISTORY,
        REPRESENTATIVE_RUNS,
        RESAMPLING_BUNDLE,
        RESAMPLING_MANIFEST,
        PHASE_A_OUTPUT,
    )
    missing = [
        path.relative_to(ROOT).as_posix()
        for path in prerequisites
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Phase A prerequisites 不完整；正确顺序为 freeze_plan -> "
            "build_existing_metrics -> run_phase_a -> inspect_and_prepare；"
            f"missing={missing}"
        )

    existing = pd.read_csv(EXISTING_METRICS)
    expected_runs = {
        (seed, fold) for seed in (2026, 3026, 4026, 5026, 6026) for fold in range(5)
    }
    observed_runs = (
        set(existing[["seed_base", "fold"]].itertuples(index=False, name=None))
        if {"seed_base", "fold"}.issubset(existing.columns)
        else set()
    )
    if (
        tuple(existing.columns) != EXISTING_METRIC_COLUMNS
        or len(existing) != 25
        or existing.duplicated(["seed_base", "fold"]).any()
        or observed_runs != expected_runs
    ):
        raise ValueError("Phase A existing metrics exact schema/25-run grid 失败")
    pass_count = 0
    for row in existing.itertuples(index=False):
        expected_gate = "PASS" if float(row.base_degradation) <= 0.05 else "FAIL"
        gate_pass = _strict_bool(row.base_gate_pass, "existing.base_gate_pass")
        if (
            str(row.base_gate) != expected_gate
            or gate_pass != (expected_gate == "PASS")
            or int(row.effective_seed) != int(row.seed_base) + int(row.fold)
        ):
            raise ValueError("Phase A existing metrics gate/effective-seed 失败")
        pass_count += int(gate_pass)
    if pass_count != 17:
        raise ValueError("Phase A existing metrics PASS/FAIL 不是17/8")

    history = pd.read_csv(TRAINING_HISTORY)
    history_runs = (
        set(history[["seed_base", "fold"]].itertuples(index=False, name=None))
        if {"seed_base", "fold"}.issubset(history.columns)
        else set()
    )
    if (
        tuple(history.columns) != TRAINING_HISTORY_COLUMNS
        or len(history) != 161
        or history.duplicated(["seed_base", "fold", "epoch"]).any()
        or history_runs != expected_runs
    ):
        raise ValueError("Phase A training history exact schema/161-row grid 失败")
    history_numeric = [
        column
        for column in TRAINING_HISTORY_COLUMNS
        if column not in {"base_gate", "is_selected_checkpoint"}
    ]
    if not np.isfinite(history[history_numeric].to_numpy(dtype=float)).all():
        raise ValueError("Phase A training history 含 nonfinite")
    existing_lookup = existing.set_index(["seed_base", "fold"])
    for key, group in history.groupby(["seed_base", "fold"], sort=False):
        ordered = group.sort_values("epoch")
        selected = group["is_selected_checkpoint"].map(
            lambda value: _strict_bool(value, "history.is_selected_checkpoint")
        )
        source = existing_lookup.loc[key]
        selected_epoch = (
            int(group.loc[selected, "epoch"].iloc[0])
            if int(selected.sum()) == 1
            else -1
        )
        if (
            not group["epoch"].is_monotonic_increasing
            or int(ordered.iloc[0]["epoch"]) != 1
            or int(selected.sum()) != 1
            or selected_epoch != int(source.selected_epoch)
            or int(ordered.iloc[-1]["epoch"]) != int(source.last_epoch)
            or len(group) != int(source.history_rows)
            or set(group["base_gate"].astype(str)) != {str(source.base_gate)}
            or not np.allclose(
                group["base_degradation"].to_numpy(dtype=float),
                float(source.base_degradation),
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise ValueError(f"Phase A training history run contract 失败: {key}")

    representatives = pd.read_csv(REPRESENTATIVE_RUNS)
    representative_keys = (
        set(representatives[["seed_base", "fold"]].itertuples(index=False, name=None))
        if {"seed_base", "fold"}.issubset(representatives.columns)
        else set()
    )
    if (
        tuple(representatives.columns) != REPRESENTATIVE_COLUMNS
        or len(representatives) != 6
        or representatives.duplicated(["seed_base", "fold"]).any()
        or not representative_keys.issubset(expected_runs)
        or set(representatives["base_gate"].astype(str)) != {"PASS", "FAIL"}
        or any(
            int(representatives["base_gate"].eq(gate).sum()) != 3
            for gate in ("PASS", "FAIL")
        )
        or representatives["gradient_result_used_for_selection"]
        .map(
            lambda value: _strict_bool(
                value, "representatives.gradient_result_used_for_selection"
            )
        )
        .any()
    ):
        raise ValueError("Phase A representative runs exact schema/3+3 contract 失败")
    expected_representatives: list[tuple[object, ...]] = []
    for gate, ranks, labels in (
        ("PASS", (0, 8, 16), ("minimum", "median", "maximum")),
        ("FAIL", (0, 3, 7), ("minimum", "lower_median", "maximum")),
    ):
        group = (
            existing.loc[existing["base_gate"].eq(gate)]
            .sort_values("base_degradation")
            .reset_index(drop=True)
        )
        for rank, label in zip(ranks, labels):
            source = group.iloc[rank]
            expected_representatives.append(
                (
                    gate,
                    label,
                    rank,
                    int(source["seed_base"]),
                    int(source["fold"]),
                    float(source["base_degradation"]),
                    int(source["selected_epoch"]),
                    int(source["last_epoch"]),
                    False,
                )
            )
    observed_representatives = [
        (
            str(row.base_gate),
            str(row.selection_rule),
            int(row.within_group_rank_zero_based),
            int(row.seed_base),
            int(row.fold),
            float(row.base_degradation),
            int(row.selected_epoch),
            int(row.last_epoch),
            _strict_bool(
                row.gradient_result_used_for_selection,
                "representatives.gradient_result_used_for_selection",
            ),
        )
        for row in representatives.itertuples(index=False)
    ]
    for observed, expected in zip(observed_representatives, expected_representatives):
        if (
            observed[:5] != expected[:5]
            or not np.isclose(observed[5], expected[5], rtol=0.0, atol=1e-12)
            or observed[6:] != expected[6:]
        ):
            raise ValueError("Phase A representative rank/source mapping 漂移")

    _, bundle_manifest = load_resampling_bundle()
    phase_a = pd.read_csv(PHASE_A_OUTPUT)
    if (
        tuple(phase_a.columns) != PHASE_A_COLUMNS
        or len(phase_a) != 3
        or phase_a["endpoint"].duplicated().any()
        or set(phase_a["endpoint"].astype(str)) != PHASE_A_ENDPOINTS
        or set(phase_a["family_id"].astype(str)) != {"phase_a_gain_associations"}
        or set(phase_a["family_size"].astype(int)) != {3}
        or set(phase_a["n"].astype(int)) != {25}
        or set(phase_a["source_existing_metrics_sha256"].astype(str))
        != {file_sha256(EXISTING_METRICS)}
        or set(phase_a["resampling_bundle_sha256"].astype(str))
        != {str(bundle_manifest["bundle_sha256"])}
        or phase_a["contains_patient_ids"]
        .map(lambda value: _strict_bool(value, "phase_a.contains_patient_ids"))
        .any()
    ):
        raise ValueError("Phase A correlation/resampling exact contract 失败")
    for column in ("p_holm", "p_raw_two_sided"):
        values = phase_a[column].dropna().to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"Phase A {column} 非法")
    return representatives


def _preflight_destinations() -> None:
    existing = [
        path.relative_to(ROOT).as_posix()
        for path in INSPECTION_DESTINATIONS
        if path.exists() or path.is_symlink()
    ]
    if existing:
        raise FileExistsError(
            "inspect destinations 必须在首次 inspect 前全部不存在；"
            "检测到 partial/既有产物，拒绝覆盖或自动清理："
            f"{existing}"
        )
    invalid_parents = [
        path.parent.relative_to(ROOT).as_posix()
        for path in INSPECTION_DESTINATIONS
        if not path.parent.is_dir() or not os.access(path.parent, os.W_OK)
    ]
    if invalid_parents:
        raise PermissionError(
            f"inspect destination parent 不存在或不可写: {sorted(set(invalid_parents))}"
        )


def main() -> None:
    args = parse_args()
    if args.self_test:
        print(json.dumps(synthetic_self_test(), ensure_ascii=False, indent=2))
        return
    assert_plan_freeze()
    representatives = _preflight_phase_a()
    _preflight_destinations()
    context = load_data_context()
    for fold in range(5):
        fold_transform(context, fold)
    selected_assets = validate_checkpoint_grid()
    trajectory_assets = validate_trajectory_assets(representatives)
    assets = selected_assets + trajectory_assets
    asset_path = ASSET_MANIFEST
    if len(assets) != 31:
        raise ValueError("inspect asset preflight 不是25 selected + 6 last")
    atomic_csv(asset_path, assets, overwrite=False)
    source_manifest = write_source_manifest(representatives)
    public, private = write_manifests(context, overwrite=False)
    cache_contract = write_cache_input_contract(context)
    source_contract = write_source_contract()
    print(
        json.dumps(
            {
                "status": "ok",
                "selected_checkpoint_cells": len(selected_assets),
                "representative_last_checkpoint_cells": len(trajectory_assets),
                "asset_manifest": str(asset_path.relative_to(ROOT)),
                "asset_manifest_sha256": file_sha256(asset_path),
                "source_manifest": str(source_manifest.relative_to(ROOT)),
                "source_manifest_sha256": file_sha256(source_manifest),
                "public_batch_manifest": str(public.relative_to(ROOT)),
                "public_batch_manifest_sha256": file_sha256(public),
                "private_batch_manifest": "configs/private/[ignored]",
                "private_mapping_present": private.is_file(),
                "cache_input_contract": str(cache_contract.relative_to(ROOT)),
                "cache_input_contract_sha256": file_sha256(cache_contract),
                "source_contract": str(source_contract.relative_to(ROOT)),
                "source_contract_sha256": file_sha256(source_contract),
                "commit_marker": "SOURCE_CONTRACT.json:contract_complete=true",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

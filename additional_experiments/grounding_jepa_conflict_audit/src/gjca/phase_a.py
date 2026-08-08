"""结果盲冻结重采样索引，并完成既有 gain/degradation 相关。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import (
    AUDIT_ROOT,
    FOLDS,
    SEED_BASES,
    assert_audit_config,
    atomic_csv,
    atomic_json,
    file_sha256,
)
from .freeze import PLAN_FREEZE, assert_plan_freeze
from .statistics import (
    ResamplingIndices,
    exact_crossed_spearman_permutation,
    generate_resampling_indices,
    holm_adjust,
    load_resampling_indices_npz,
    save_resampling_indices_npz,
    spearman_crossed_ci,
)


RESAMPLING_BUNDLE = AUDIT_ROOT / "metrics" / "resampling_indices.npz"
RESAMPLING_MANIFEST = AUDIT_ROOT / "metrics" / "resampling_manifest.json"
PHASE_A_OUTPUT = AUDIT_ROOT / "metrics" / "phase_a_gain_correlations.csv"

PHASE_A_COLUMNS = (
    "schema_version",
    "endpoint",
    "family_id",
    "family_size",
    "analysis_unit",
    "x",
    "y",
    "n",
    "spearman_rho",
    "ci_low",
    "ci_high",
    "p_raw_two_sided",
    "permutation_method",
    "permutation_replicates",
    "permutation_extreme_count",
    "permutation_identity_included",
    "permutation_status",
    "p_holm",
    "holm_rank",
    "status",
    "bootstrap_method",
    "bootstrap_requested",
    "bootstrap_finite",
    "bootstrap_nonfinite",
    "bootstrap_finite_fraction",
    "confidence_level",
    "resampling_bundle_sha256",
    "source_existing_metrics_sha256",
    "contains_patient_ids",
)


def create_resampling_bundle() -> tuple[ResamplingIndices, dict[str, Any]]:
    assert_plan_freeze()
    if RESAMPLING_BUNDLE.exists() or RESAMPLING_MANIFEST.exists():
        raise FileExistsError("拒绝覆盖已有结果盲 resampling bundle")
    config = assert_audit_config()["statistics"]
    indices = generate_resampling_indices(
        crossed_replicates=int(config["crossed_bootstrap"]["replicates"]),
        n_seeds=len(SEED_BASES),
        n_folds=len(FOLDS),
        crossed_seed=int(config["rng"]["crossed_bootstrap_seed"]),
    )
    if len(indices.crossed_seed_draws) != int(
        config["group_bootstrap"]["replicates"]
    ) or len(indices.crossed_permutation_seed_orders) != int(
        config["permutation"]["replicates"]
    ):
        raise ValueError("resampling config 与生成的 crossed indices 数量分叉")
    bundle_manifest = save_resampling_indices_npz(RESAMPLING_BUNDLE, indices)
    public_manifest = {
        "schema_version": 1,
        "created_before_new_gradient_forward": True,
        "plan_freeze_sha256": file_sha256(PLAN_FREEZE),
        "bundle": bundle_manifest,
        "contains_patient_ids": False,
    }
    atomic_json(RESAMPLING_MANIFEST, public_manifest)
    loaded, observed = load_resampling_bundle()
    if observed != bundle_manifest:
        raise ValueError("resampling manifest 写入后不闭环")
    return loaded, observed


def load_resampling_bundle() -> tuple[ResamplingIndices, dict[str, Any]]:
    assert_plan_freeze()
    if not RESAMPLING_MANIFEST.is_file():
        raise FileNotFoundError("resampling_manifest.json 缺失")
    public = json.loads(RESAMPLING_MANIFEST.read_text(encoding="utf-8"))
    if (
        int(public.get("schema_version", -1)) != 1
        or public.get("created_before_new_gradient_forward") is not True
        or public.get("contains_patient_ids") is not False
        or str(public.get("plan_freeze_sha256")) != file_sha256(PLAN_FREEZE)
    ):
        raise ValueError("resampling public manifest contract 失败")
    indices, observed = load_resampling_indices_npz(RESAMPLING_BUNDLE)
    if public.get("bundle") != observed:
        raise ValueError("resampling bundle 与 public manifest 不一致")
    return indices, observed


def _grid(frame: pd.DataFrame, column: str) -> np.ndarray:
    pivot = frame.pivot(index="seed_base", columns="fold", values=column).reindex(
        index=SEED_BASES, columns=FOLDS
    )
    if pivot.shape != (5, 5) or pivot.isna().any().any():
        raise ValueError(f"Phase A {column} 不是完整5×5 grid")
    values = pivot.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Phase A {column} nonfinite")
    return values


def write_phase_a_correlations() -> Path:
    assert_plan_freeze()
    if PHASE_A_OUTPUT.exists():
        raise FileExistsError("拒绝覆盖已有 Phase A correlation")
    existing_path = AUDIT_ROOT / "metrics" / "run_level_existing_metrics.csv"
    frame = pd.read_csv(existing_path)
    if (
        len(frame) != 25
        or frame.duplicated(["seed_base", "fold"]).any()
        or int(frame["base_gate"].eq("PASS").sum()) != 17
        or int(frame["base_gate"].eq("FAIL").sum()) != 8
    ):
        raise ValueError("Phase A existing 25-run/17-PASS contract 失败")
    if RESAMPLING_BUNDLE.exists() or RESAMPLING_MANIFEST.exists():
        indices, bundle_manifest = load_resampling_bundle()
    else:
        indices, bundle_manifest = create_resampling_bundle()
    x = _grid(frame, "base_degradation")
    endpoints = (
        ("degradation_vs_static_delta_spearman", "static_ftv_delta_spearman"),
        ("degradation_vs_observed_delta_spearman", "delta_ftv_delta_spearman"),
        ("degradation_vs_observed_delta_r2", "delta_ftv_delta_r2"),
    )
    calculated: dict[str, tuple[Any, Any]] = {}
    for endpoint, column in endpoints:
        y = _grid(frame, column)
        interval = spearman_crossed_ci(
            x,
            y,
            indices.crossed_seed_draws,
            indices.crossed_fold_draws,
            confidence_level=0.95,
            minimum_finite_fraction=0.95,
        )
        permutation = exact_crossed_spearman_permutation(
            x,
            y,
            indices.crossed_permutation_seed_orders,
            indices.crossed_permutation_fold_orders,
        )
        calculated[endpoint] = (interval, permutation)
    adjusted = holm_adjust(
        {endpoint: calculated[endpoint][1].p_value for endpoint, _ in endpoints}
    )
    rows: list[dict[str, Any]] = []
    for endpoint, column in endpoints:
        result, permutation = calculated[endpoint]
        holm = adjusted[endpoint]
        row = {
            "schema_version": 1,
            "endpoint": endpoint,
            "family_id": "phase_a_gain_associations",
            "family_size": holm.family_size,
            "analysis_unit": "seed_fold_run",
            "x": "base_degradation",
            "y": column,
            "n": 25,
            "spearman_rho": result.estimate,
            "ci_low": result.ci_low,
            "ci_high": result.ci_high,
            "p_raw_two_sided": permutation.p_value,
            "permutation_method": "exact_crossed_seed_order_x_fold_order",
            "permutation_replicates": permutation.replicates,
            "permutation_extreme_count": permutation.extreme_count,
            "permutation_identity_included": permutation.includes_identity,
            "permutation_status": permutation.status,
            "p_holm": holm.p_holm,
            "holm_rank": holm.rank,
            "status": result.status,
            "bootstrap_method": "crossed_seed_fold_cartesian_percentile",
            "bootstrap_requested": result.bootstrap_requested,
            "bootstrap_finite": result.bootstrap_finite,
            "bootstrap_nonfinite": result.bootstrap_requested - result.bootstrap_finite,
            "bootstrap_finite_fraction": result.bootstrap_finite_fraction,
            "confidence_level": result.confidence_level,
            "resampling_bundle_sha256": bundle_manifest["bundle_sha256"],
            "source_existing_metrics_sha256": file_sha256(existing_path),
            "contains_patient_ids": False,
        }
        if tuple(row) != PHASE_A_COLUMNS:
            raise AssertionError("Phase A row schema/order 漂移")
        rows.append(row)
    atomic_csv(PHASE_A_OUTPUT, rows)
    return PHASE_A_OUTPUT


__all__ = [
    "PHASE_A_COLUMNS",
    "PHASE_A_OUTPUT",
    "RESAMPLING_BUNDLE",
    "RESAMPLING_MANIFEST",
    "create_resampling_bundle",
    "load_resampling_bundle",
    "write_phase_a_correlations",
]

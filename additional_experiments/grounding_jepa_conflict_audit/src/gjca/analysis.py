"""25-run conflict audit 的预注册统计、定位与唯一 H1–H4 判定。"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata

from .aggregation import (
    AGGREGATE_VALUE_COLUMNS,
    COVERAGE_COLUMNS,
    EXISTING_METRIC_COLUMNS,
    LAYER_LEVEL_COLUMNS,
    RUN_LEVEL_COLUMNS,
    TRAJECTORY_CHANGE_COLUMNS,
    TRAJECTORY_COLUMNS,
    build_aggregate_tables,
)
from .batches import PUBLIC_MANIFEST
from .contracts import (
    AUDIT_ROOT,
    FOLDS,
    GROUPS,
    SEED_BASES,
    SPLITS,
    atomic_csv,
    atomic_json,
    canonical_json_sha256,
    ensure_no_patient_columns,
    file_sha256,
)
from .gradients import BATCH_GRADIENT_COLUMNS
from .diagnosis import DYNAMICS_ENDPOINTS, evaluate_hypotheses
from .phase_a import PHASE_A_COLUMNS, RESAMPLING_BUNDLE, load_resampling_bundle
from .source_contract import SOURCE_CONTRACT, assert_source_contract
from .statistics import (
    ResamplingIndices,
    crossed_group_bootstrap_contrasts,
    exact_crossed_group_permutation,
    exact_crossed_spearman_permutation,
    generate_resampling_indices,
    holm_adjust,
    spearman_crossed_ci,
)


PASS_FAIL_METRICS = (
    ("gradient_cosine", "median"),
    ("negative_fraction", "mean"),
    ("strong_negative_fraction", "mean"),
    ("weighted_gradient_norm_ratio", "median"),
    ("base_descent_margin", "median"),
    ("base_descent_failure_fraction", "mean"),
)
CORRELATION_ENDPOINTS = (
    "gradient_cosine",
    "negative_fraction",
    "weighted_gradient_norm_ratio",
    "base_descent_margin",
)
LOCALIZATION_GROUPS = (
    "encoder_stage_1",
    "encoder_stage_2",
    "encoder_stage_3",
    "encoder_stage_4",
    "response_projection",
)

PASS_FAIL_COLUMNS = (
    "schema_version",
    "split",
    "group",
    "metric",
    "primary_contrast",
    "pass_n",
    "fail_n",
    "pass_mean",
    "pass_sd",
    "pass_median",
    "pass_q1",
    "pass_q3",
    "fail_mean",
    "fail_sd",
    "fail_median",
    "fail_q1",
    "fail_q3",
    "mean_difference_fail_minus_pass",
    "mean_ci_low",
    "mean_ci_high",
    "median_difference_fail_minus_pass",
    "median_ci_low",
    "median_ci_high",
    "fail_over_pass_median_ratio",
    "ratio_ci_low",
    "ratio_ci_high",
    "ratio_status",
    "permutation_method",
    "permutation_statistic",
    "permutation_p_raw_two_sided",
    "permutation_extreme_count",
    "permutation_replicates",
    "permutation_identity_included",
    "permutation_status",
    "mann_whitney_u",
    "mann_whitney_p_two_sided",
    "decision_endpoint",
    "decision_family",
    "decision_family_size",
    "p_holm",
    "holm_rank",
    "bootstrap_method",
    "mean_bootstrap_requested",
    "mean_bootstrap_finite",
    "mean_bootstrap_nonfinite",
    "mean_bootstrap_finite_fraction",
    "mean_bootstrap_status",
    "median_bootstrap_requested",
    "median_bootstrap_finite",
    "median_bootstrap_nonfinite",
    "median_bootstrap_finite_fraction",
    "median_bootstrap_status",
    "ratio_bootstrap_requested",
    "ratio_bootstrap_finite",
    "ratio_bootstrap_nonfinite",
    "ratio_bootstrap_finite_fraction",
    "ratio_bootstrap_status",
    "status",
    "resampling_bundle_sha256",
    "contains_patient_ids",
)

CORRELATION_COLUMNS = (
    "schema_version",
    "split",
    "group",
    "endpoint",
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
    "decision_endpoint",
    "decision_family",
    "decision_family_size",
    "p_holm",
    "holm_rank",
    "status",
    "bootstrap_method",
    "bootstrap_requested",
    "bootstrap_finite",
    "bootstrap_nonfinite",
    "bootstrap_finite_fraction",
    "resampling_bundle_sha256",
    "contains_patient_ids",
)

DYNAMICS_COLUMNS = (
    "schema_version",
    "endpoint",
    "source_variable",
    "risk_orientation_multiplier",
    "risk_interpretation",
    "analysis_unit",
    "n",
    "spearman_rho_oriented",
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
    "family_id",
    "family_size",
    "status",
    "bootstrap_method",
    "bootstrap_requested",
    "bootstrap_finite",
    "bootstrap_nonfinite",
    "bootstrap_finite_fraction",
    "resampling_bundle_sha256",
    "contains_patient_ids",
)

COVERAGE_CORRELATION_COLUMNS = (
    "schema_version",
    "endpoint",
    "split",
    "coverage_variable",
    "analysis_unit",
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
    "status",
    "bootstrap_method",
    "bootstrap_requested",
    "bootstrap_finite",
    "bootstrap_nonfinite",
    "bootstrap_finite_fraction",
    "resampling_bundle_sha256",
    "contains_patient_ids",
)

LOCALIZATION_COLUMNS = (
    "schema_version",
    "group",
    "validation_D_cos",
    "validation_D_mbase",
    "validation_D_ratio",
    "train_D_cos",
    "train_D_mbase",
    "train_D_ratio",
    "train_cosine_direction_replicated",
    "train_mbase_direction_replicated",
    "train_ratio_direction_replicated",
    "cosine_rank",
    "mbase_rank",
    "ratio_rank",
    "localization_score",
    "localization_rank",
    "family_id",
    "validation_cosine_p_raw",
    "validation_cosine_p_holm",
    "validation_cosine_holm_rank",
    "validation_cosine_holm_status",
    "validation_mbase_p_raw",
    "validation_mbase_p_holm",
    "validation_mbase_holm_rank",
    "validation_mbase_holm_status",
    "validation_ratio_p_raw",
    "validation_ratio_p_holm",
    "validation_ratio_holm_rank",
    "validation_ratio_holm_status",
    "holm_family_size",
    "layer_crosses_direction_threshold",
    "widespread_layer_count",
    "widespread_conflict",
    "contains_patient_ids",
)

FOLD_SIGNATURE_COLUMNS = (
    "schema_version",
    "fold",
    "split",
    "n_runs",
    "pass_n",
    "fail_n",
    "median_gradient_cosine",
    "median_base_descent_margin",
    "median_norm_ratio",
    "cosine_rank_ascending",
    "mbase_rank_ascending",
    "ratio_rank_descending",
    "cosine_minus_other_runs",
    "mbase_minus_other_runs",
    "ratio_minus_other_runs",
    "is_fold3",
    "fold3_strictly_worst_cosine",
    "fold3_strictly_worst_mbase",
    "fold3_crosses_practical_threshold",
    "fold3_special_signature",
    "contains_patient_ids",
)

ANALYSIS_INPUT_MANIFEST_COLUMNS = (
    "schema_version",
    "artifact_role",
    "path",
    "sha256",
    "bytes",
    "rows",
    "column_count",
    "columns_sha256",
    "contains_patient_ids",
)
HYPOTHESIS_DECISION_COLUMNS = (
    "schema_version",
    "hypothesis",
    "hierarchy_reached",
    "rule_eligible",
    "rule_satisfied",
    "selected",
    "decision_status",
    "selected_hypothesis",
    "first_recommendation",
    "second_recommendation",
    "condition_details_json",
    "contains_patient_ids",
)
FINAL_ANALYSIS_MANIFEST = AUDIT_ROOT / "metrics" / "final_analysis_manifest.json"
FINAL_ANALYSIS_MARKER = AUDIT_ROOT / "metrics" / "FINAL_ANALYSIS_COMPLETE.json"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")


def _fixed(frame: pd.DataFrame, split: str, group: str) -> pd.DataFrame:
    selected = frame.loc[frame["split"].eq(split) & frame["group"].eq(group)].copy()
    selected = selected.sort_values(["seed_base", "fold"]).reset_index(drop=True)
    expected = {(seed, fold) for seed in SEED_BASES for fold in FOLDS}
    observed = set(selected[["seed_base", "fold"]].itertuples(index=False, name=None))
    if len(selected) != 25 or observed != expected:
        raise ValueError(f"run grid 不完整: {split}/{group}")
    if (
        list(selected["base_gate"]).count("PASS") != 17
        or list(selected["base_gate"]).count("FAIL") != 8
    ):
        raise ValueError(f"PASS/FAIL 17/8 contract 失败: {split}/{group}")
    return selected


def _grid(frame: pd.DataFrame, column: str) -> np.ndarray:
    pivot = frame.pivot(index="seed_base", columns="fold", values=column).reindex(
        index=SEED_BASES, columns=FOLDS
    )
    values = pivot.to_numpy(dtype=np.float64)
    if values.shape != (5, 5) or not np.isfinite(values).all():
        raise ValueError(f"{column} 不是 finite 5×5 grid")
    return values


def _summary(values: np.ndarray) -> dict[str, float]:
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("group descriptive values 非法")
    q1, q3 = np.quantile(values, [0.25, 0.75], method="linear")
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)),
        "median": float(np.median(values)),
        "q1": float(q1),
        "q3": float(q3),
    }


def _validate_written_aggregates(
    frames: Mapping[str, pd.DataFrame],
    representatives: pd.DataFrame,
    *,
    expected_source_sha: str | None = None,
    expected_public_sha: str | None = None,
) -> None:
    """正式分析前重新验证五张落盘聚合表的 grid、finite 与 provenance。"""

    contracts = {
        "layer_level_conflict_metrics": (LAYER_LEVEL_COLUMNS, 350),
        "run_level_conflict_metrics": (RUN_LEVEL_COLUMNS, 50),
        "trajectory_conflict_metrics": (TRAJECTORY_COLUMNS, 168),
        "trajectory_change_metrics": (TRAJECTORY_CHANGE_COLUMNS, 84),
        "ftv_coverage_metrics": (COVERAGE_COLUMNS, 10),
    }
    for name, (columns, row_count) in contracts.items():
        frame = frames[name]
        if tuple(frame.columns) != columns or len(frame) != row_count:
            raise ValueError(f"{name} schema/rows 漂移")
        privacy = frame["contains_patient_ids"].astype(str).str.strip().str.lower()
        if set(privacy) != {"false"}:
            raise ValueError(f"{name} privacy flag 漂移")

    run_keys = {(seed, fold) for seed in SEED_BASES for fold in FOLDS}
    layer_keys = {
        (seed, fold, split, group)
        for seed, fold in run_keys
        for split in SPLITS
        for group in GROUPS
    }
    layer = frames["layer_level_conflict_metrics"]
    if (
        layer.duplicated(["seed_base", "fold", "split", "group"]).any()
        or set(
            layer[["seed_base", "fold", "split", "group"]].itertuples(
                index=False, name=None
            )
        )
        != layer_keys
        or set(layer["checkpoint_kind"]) != {"selected"}
    ):
        raise ValueError("layer aggregate Cartesian grid 漂移")
    run = frames["run_level_conflict_metrics"]
    expected_run_keys = {
        (seed, fold, split) for seed, fold in run_keys for split in SPLITS
    }
    if (
        run.duplicated(["seed_base", "fold", "split"]).any()
        or set(run[["seed_base", "fold", "split"]].itertuples(index=False, name=None))
        != expected_run_keys
        or set(run["group"]) != {"all_shared"}
        or set(run["checkpoint_kind"]) != {"selected"}
    ):
        raise ValueError("run aggregate Cartesian grid 漂移")

    if (
        len(representatives) != 6
        or representatives.duplicated(["seed_base", "fold"]).any()
    ):
        raise ValueError("representative run grid 漂移")
    representative_keys = set(
        representatives[["seed_base", "fold"]].itertuples(index=False, name=None)
    )
    trajectory = frames["trajectory_conflict_metrics"]
    expected_trajectory = {
        (seed, fold, kind, split, group)
        for seed, fold in representative_keys
        for kind in ("selected", "last")
        for split in SPLITS
        for group in GROUPS
    }
    trajectory_key_columns = ["seed_base", "fold", "checkpoint_kind", "split", "group"]
    if (
        trajectory.duplicated(trajectory_key_columns).any()
        or set(trajectory[trajectory_key_columns].itertuples(index=False, name=None))
        != expected_trajectory
    ):
        raise ValueError("trajectory aggregate Cartesian grid 漂移")
    change = frames["trajectory_change_metrics"]
    expected_change = {
        (seed, fold, split, group)
        for seed, fold in representative_keys
        for split in SPLITS
        for group in GROUPS
    }
    change_key_columns = ["seed_base", "fold", "split", "group"]
    if (
        change.duplicated(change_key_columns).any()
        or set(change[change_key_columns].itertuples(index=False, name=None))
        != expected_change
    ):
        raise ValueError("trajectory change Cartesian grid 漂移")
    coverage = frames["ftv_coverage_metrics"]
    if coverage.duplicated(["fold", "split"]).any() or set(
        coverage[["fold", "split"]].itertuples(index=False, name=None)
    ) != {(fold, split) for fold in FOLDS for split in SPLITS}:
        raise ValueError("coverage fold/split grid 漂移")

    numeric_checks = {
        "layer_level_conflict_metrics": AGGREGATE_VALUE_COLUMNS,
        "run_level_conflict_metrics": AGGREGATE_VALUE_COLUMNS,
        "trajectory_conflict_metrics": AGGREGATE_VALUE_COLUMNS,
        "trajectory_change_metrics": tuple(
            f"last_minus_selected_{column}" for column in AGGREGATE_VALUE_COLUMNS
        ),
        "ftv_coverage_metrics": (
            "pool_ftv_proportion",
            "ftv_draw_proportion",
            "batch_ftv_sd",
            "member_exposure_sd",
            "ftv_member_exposure_sd",
        ),
    }
    for name, columns in numeric_checks.items():
        if not np.isfinite(frames[name][list(columns)].to_numpy(dtype=float)).all():
            raise ValueError(f"{name} 含 nonfinite core aggregate")
    for name in (
        "layer_level_conflict_metrics",
        "run_level_conflict_metrics",
        "trajectory_conflict_metrics",
    ):
        frame = frames[name]
        if set(frame["n_batches"]) != {8} or set(frame["n_undefined"]) != {0}:
            raise ValueError(f"{name} batch/undefined contract 漂移")
    if (
        set(change["selected_n_batches"]) != {8}
        or set(change["last_n_batches"]) != {8}
        or set(change["selected_n_undefined"]) != {0}
        or set(change["last_n_undefined"]) != {0}
    ):
        raise ValueError("trajectory change batch/undefined contract 漂移")

    source_sha = expected_source_sha or file_sha256(SOURCE_CONTRACT)
    public_sha = expected_public_sha or file_sha256(PUBLIC_MANIFEST)
    for name in (
        "layer_level_conflict_metrics",
        "run_level_conflict_metrics",
        "trajectory_conflict_metrics",
        "trajectory_change_metrics",
    ):
        frame = frames[name]
        if set(frame["source_contract_sha256"].astype(str)) != {source_sha} or set(
            frame["public_manifest_sha256"].astype(str)
        ) != {public_sha}:
            raise ValueError(f"{name} provenance SHA 漂移")


def _assert_aggregate_equivalence(
    written: Mapping[str, pd.DataFrame], rebuilt: Mapping[str, pd.DataFrame]
) -> None:
    """证明落盘 aggregate 与当前两张 gradient matrix 的严格重聚合等价。"""

    if set(written) != set(rebuilt):
        raise ValueError("written/rebuilt aggregate table set 漂移")
    for name in written:
        try:
            pd.testing.assert_frame_equal(
                written[name],
                rebuilt[name],
                check_dtype=False,
                check_exact=False,
                rtol=1e-10,
                atol=1e-12,
                check_like=False,
            )
        except AssertionError as error:
            raise ValueError(
                f"{name} 与当前 gradient matrix 重聚合结果不等价"
            ) from error


def _analysis_input_rows(root: Path) -> list[dict[str, Any]]:
    specs = (
        (
            "selected_batch_gradients",
            "metrics/batch_gradient_metrics.csv",
            BATCH_GRADIENT_COLUMNS,
            2800,
        ),
        (
            "trajectory_batch_gradients",
            "metrics/trajectory_batch_gradient_metrics.csv",
            BATCH_GRADIENT_COLUMNS,
            672,
        ),
        (
            "layer_aggregate",
            "metrics/layer_level_conflict_metrics.csv",
            LAYER_LEVEL_COLUMNS,
            350,
        ),
        (
            "run_aggregate",
            "metrics/run_level_conflict_metrics.csv",
            RUN_LEVEL_COLUMNS,
            50,
        ),
        (
            "trajectory_aggregate",
            "metrics/trajectory_conflict_metrics.csv",
            TRAJECTORY_COLUMNS,
            168,
        ),
        (
            "trajectory_change",
            "metrics/trajectory_change_metrics.csv",
            TRAJECTORY_CHANGE_COLUMNS,
            84,
        ),
        ("ftv_coverage", "metrics/ftv_coverage_metrics.csv", COVERAGE_COLUMNS, 10),
        (
            "existing_run_metrics",
            "metrics/run_level_existing_metrics.csv",
            EXISTING_METRIC_COLUMNS,
            25,
        ),
        (
            "phase_a_correlations",
            "metrics/phase_a_gain_correlations.csv",
            PHASE_A_COLUMNS,
            3,
        ),
    )
    rows: list[dict[str, Any]] = []
    for role, relative, columns, expected_rows in specs:
        path = root / relative
        frame = pd.read_csv(path)
        if tuple(frame.columns) != tuple(columns) or len(frame) != expected_rows:
            raise ValueError(f"analysis input schema/rows 漂移: {role}")
        ensure_no_patient_columns(frame.columns)
        if "contains_patient_ids" not in frame.columns or set(
            frame["contains_patient_ids"].astype(str).str.strip().str.lower()
        ) != {"false"}:
            raise ValueError(f"analysis input privacy flag 漂移: {role}")
        row = {
            "schema_version": 1,
            "artifact_role": role,
            "path": relative,
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
            "rows": len(frame),
            "column_count": len(frame.columns),
            "columns_sha256": canonical_json_sha256(list(frame.columns)),
            "contains_patient_ids": False,
        }
        if tuple(row) != ANALYSIS_INPUT_MANIFEST_COLUMNS:
            raise AssertionError("analysis input manifest row schema 漂移")
        rows.append(row)
    if len(rows) != 9 or len({row["artifact_role"] for row in rows}) != 9:
        raise AssertionError("analysis input manifest 不是9个唯一输入")
    return rows


def _git_head() -> str:
    value = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=AUDIT_ROOT.parents[1], text=True
    ).strip()
    if not _HEX40.fullmatch(value):
        raise ValueError("analysis commit 不是40位 git SHA")
    return value


def _csv_artifact(path: Path, relative: str) -> dict[str, Any]:
    frame = pd.read_csv(path)
    ensure_no_patient_columns(frame.columns)
    return {
        "artifact_kind": "csv",
        "path": f"metrics/{relative}",
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "rows": len(frame),
        "columns": list(frame.columns),
    }


def _publish_analysis_bundle(
    root: Path,
    *,
    csv_outputs: Mapping[str, tuple[list[dict[str, Any]], tuple[str, ...]]],
    diagnosis: Mapping[str, Any],
) -> dict[str, Any]:
    """先完整 staging，最后发布 completion marker；任何旧目标均拒绝覆盖。"""

    metrics = root / "metrics"
    input_name = "analysis_input_manifest.csv"
    diagnosis_name = "diagnosis.json"
    manifest_name = FINAL_ANALYSIS_MANIFEST.name
    marker_name = FINAL_ANALYSIS_MARKER.name
    publication_names = (
        input_name,
        *csv_outputs.keys(),
        diagnosis_name,
        manifest_name,
        marker_name,
    )
    existing = [name for name in publication_names if (metrics / name).exists()]
    if existing:
        raise FileExistsError(f"拒绝覆盖/补写已有 analysis bundle: {existing}")
    input_rows = _analysis_input_rows(root)
    source_sha = file_sha256(SOURCE_CONTRACT)
    with tempfile.TemporaryDirectory(
        prefix=".analysis-staging-", dir=metrics
    ) as temporary:
        stage = Path(temporary)
        atomic_csv(
            stage / input_name,
            input_rows,
            fieldnames=ANALYSIS_INPUT_MANIFEST_COLUMNS,
        )
        input_sha = file_sha256(stage / input_name)
        for name, (rows, columns) in csv_outputs.items():
            atomic_csv(stage / name, rows, fieldnames=columns)
        diagnosis_payload = {
            **dict(diagnosis),
            "analysis_input_manifest_sha256": input_sha,
        }
        atomic_json(stage / diagnosis_name, diagnosis_payload)
        artifacts = [_csv_artifact(stage / input_name, input_name)]
        artifacts.extend(_csv_artifact(stage / name, name) for name in csv_outputs)
        artifacts.append(
            {
                "artifact_kind": "json",
                "path": f"metrics/{diagnosis_name}",
                "sha256": file_sha256(stage / diagnosis_name),
                "bytes": (stage / diagnosis_name).stat().st_size,
                "rows": None,
                "columns": [],
            }
        )
        final_manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "source_contract_sha256": source_sha,
            "analysis_input_manifest_sha256": input_sha,
            "artifacts": artifacts,
            "contains_patient_ids": False,
        }
        final_manifest["payload_sha256"] = canonical_json_sha256(final_manifest)
        atomic_json(stage / manifest_name, final_manifest)
        marker = {
            "schema_version": 1,
            "status": "complete",
            "final_analysis_manifest_sha256": file_sha256(stage / manifest_name),
            "source_contract_sha256": source_sha,
            "diagnosis_sha256": file_sha256(stage / diagnosis_name),
            "analysis_commit": _git_head(),
            "contains_patient_ids": False,
        }
        atomic_json(stage / marker_name, marker)

        # Hard-link publish 不覆盖竞态出现的目标；marker 永远最后发布。
        for name in publication_names[:-1]:
            os.link(stage / name, metrics / name)
        os.link(stage / marker_name, metrics / marker_name)
    return {
        "analysis_input_manifest_sha256": input_sha,
        "final_analysis_manifest_sha256": file_sha256(metrics / manifest_name),
        "completion_marker_sha256": file_sha256(metrics / marker_name),
    }


def validate_final_analysis_bundle(
    root: str | Path = AUDIT_ROOT,
) -> dict[str, Any]:
    """只读验证 committed analysis bundle、输入闭包与所有登记 SHA。"""

    root = Path(root).resolve()
    if root != AUDIT_ROOT.resolve():
        raise ValueError("formal validation root 必须是冻结 audit root")
    assert_source_contract(full_content_hash=False, full_checkpoint_hash=False)
    manifest_path = root / "metrics" / FINAL_ANALYSIS_MANIFEST.name
    marker_path = root / "metrics" / FINAL_ANALYSIS_MARKER.name
    diagnosis_path = root / "metrics" / "diagnosis.json"
    for path in (manifest_path, marker_path, diagnosis_path):
        if not path.is_file():
            raise FileNotFoundError(f"analysis bundle 缺文件: {path.name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    unsigned = dict(manifest)
    payload_sha = str(unsigned.pop("payload_sha256", ""))
    expected_paths = {
        "metrics/analysis_input_manifest.csv",
        "metrics/pass_fail_comparison.csv",
        "metrics/gradient_correlation_metrics.csv",
        "metrics/dynamics_correlations.csv",
        "metrics/coverage_correlations.csv",
        "metrics/layer_localization_metrics.csv",
        "metrics/fold_signature_metrics.csv",
        "metrics/hypothesis_decision.csv",
        "metrics/diagnosis.json",
    }
    artifacts = manifest.get("artifacts")
    if (
        int(manifest.get("schema_version", -1)) != 1
        or manifest.get("status") != "complete"
        or manifest.get("contains_patient_ids") is not False
        or canonical_json_sha256(unsigned) != payload_sha
        or str(manifest.get("source_contract_sha256")) != file_sha256(SOURCE_CONTRACT)
        or not isinstance(artifacts, list)
        or {str(item.get("path")) for item in artifacts} != expected_paths
    ):
        raise ValueError("final analysis manifest schema/payload/artifact set 漂移")
    for item in artifacts:
        if set(item) != {"artifact_kind", "path", "sha256", "bytes", "rows", "columns"}:
            raise ValueError("final analysis artifact schema 漂移")
        path = (root / str(item["path"])).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("final analysis artifact path 越界") from error
        if (
            not path.is_file()
            or file_sha256(path) != str(item["sha256"])
            or path.stat().st_size != int(item["bytes"])
        ):
            raise ValueError(f"final analysis artifact SHA/bytes 漂移: {item['path']}")
        if item["artifact_kind"] == "csv":
            frame = pd.read_csv(path)
            if (
                len(frame) != int(item["rows"])
                or list(frame.columns) != item["columns"]
            ):
                raise ValueError(f"final analysis CSV rows/schema 漂移: {item['path']}")
            ensure_no_patient_columns(frame.columns)
        elif (
            item["artifact_kind"] != "json"
            or item["rows"] is not None
            or item["columns"] != []
        ):
            raise ValueError("final analysis JSON artifact schema 漂移")

    input_path = root / "metrics" / "analysis_input_manifest.csv"
    if file_sha256(input_path) != str(manifest.get("analysis_input_manifest_sha256")):
        raise ValueError("analysis input manifest SHA 未闭环")
    input_frame = pd.read_csv(input_path)
    if (
        tuple(input_frame.columns) != ANALYSIS_INPUT_MANIFEST_COLUMNS
        or len(input_frame) != 9
        or input_frame["artifact_role"].duplicated().any()
    ):
        raise ValueError("analysis input manifest schema/grid 漂移")
    for item in input_frame.itertuples(index=False):
        path = (root / str(item.path)).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("analysis input path 越界") from error
        frame = pd.read_csv(path)
        if (
            file_sha256(path) != str(item.sha256)
            or path.stat().st_size != int(item.bytes)
            or len(frame) != int(item.rows)
            or len(frame.columns) != int(item.column_count)
            or canonical_json_sha256(list(frame.columns)) != str(item.columns_sha256)
            or str(item.contains_patient_ids).strip().lower() != "false"
        ):
            raise ValueError(f"analysis input artifact 漂移: {item.artifact_role}")
        ensure_no_patient_columns(frame.columns)

    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected_marker_keys = {
        "schema_version",
        "status",
        "final_analysis_manifest_sha256",
        "source_contract_sha256",
        "diagnosis_sha256",
        "analysis_commit",
        "contains_patient_ids",
    }
    if (
        set(marker) != expected_marker_keys
        or int(marker.get("schema_version", -1)) != 1
        or marker.get("status") != "complete"
        or marker.get("contains_patient_ids") is not False
        or str(marker.get("final_analysis_manifest_sha256"))
        != file_sha256(manifest_path)
        or str(marker.get("source_contract_sha256")) != file_sha256(SOURCE_CONTRACT)
        or str(marker.get("diagnosis_sha256")) != file_sha256(diagnosis_path)
        or not _HEX40.fullmatch(str(marker.get("analysis_commit", "")))
    ):
        raise ValueError("FINAL_ANALYSIS_COMPLETE marker 漂移")
    decisions = pd.read_csv(root / "metrics" / "hypothesis_decision.csv")
    selected_text = (
        decisions["selected"].astype(str).str.strip().str.lower()
        if "selected" in decisions
        else pd.Series(dtype=str)
    )
    if (
        tuple(decisions.columns) != HYPOTHESIS_DECISION_COLUMNS
        or len(decisions) != 4
        or decisions["hypothesis"].duplicated().any()
        or not set(selected_text).issubset({"true", "false"})
        or int(selected_text.eq("true").sum()) != 1
    ):
        raise ValueError("committed hypothesis decision 不是唯一4行")
    diagnosis = json.loads(diagnosis_path.read_text(encoding="utf-8"))
    if diagnosis.get("contains_patient_ids") is not False or str(
        diagnosis.get("analysis_input_manifest_sha256")
    ) != file_sha256(input_path):
        raise ValueError("diagnosis privacy/input provenance 漂移")
    return {
        "status": "ok",
        "selected_hypothesis": diagnosis["selected_hypothesis"],
        "artifacts": len(artifacts),
        "inputs": len(input_frame),
        "final_analysis_manifest_sha256": file_sha256(manifest_path),
    }


def build_pass_fail_table(
    layer: pd.DataFrame, indices: ResamplingIndices, bundle_sha: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        for group in GROUPS:
            run = _fixed(layer, split, group)
            passed = run["base_gate"].eq("PASS").to_numpy(dtype=bool)
            gate_grid = run.pivot(
                index="seed_base", columns="fold", values="base_gate"
            ).reindex(index=SEED_BASES, columns=FOLDS)
            passed_grid = gate_grid.eq("PASS").to_numpy(dtype=bool)
            if int(passed_grid.sum()) != 17 or not np.array_equal(
                passed_grid.ravel(), passed
            ):
                raise ValueError("PASS mask vector/grid 不闭环")
            for metric, contrast in PASS_FAIL_METRICS:
                values_grid = _grid(run, metric)
                values = values_grid.ravel()
                if not np.isfinite(values).all():
                    raise ValueError(
                        f"PASS/FAIL core nonfinite: {split}/{group}/{metric}"
                    )
                pass_values, fail_values = values[passed], values[~passed]
                descriptive_pass = _summary(pass_values)
                descriptive_fail = _summary(fail_values)
                bootstrap = crossed_group_bootstrap_contrasts(
                    values_grid,
                    passed_grid,
                    indices.crossed_seed_draws,
                    indices.crossed_fold_draws,
                )
                permutation = exact_crossed_group_permutation(
                    values_grid,
                    passed_grid,
                    indices.crossed_permutation_seed_orders,
                    indices.crossed_permutation_fold_orders,
                    contrast=contrast,
                )
                mann = mannwhitneyu(
                    fail_values,
                    pass_values,
                    alternative="two-sided",
                    method="asymptotic",
                    use_continuity=True,
                )
                decision_endpoint = ""
                decision_family = ""
                if split == "validation" and group == "all_shared":
                    mapping = {
                        "gradient_cosine": (
                            "D_cos",
                            "h1_primary_validation_all_shared",
                        ),
                        "negative_fraction": (
                            "D_neg",
                            "h1_primary_validation_all_shared",
                        ),
                        "weighted_gradient_norm_ratio": (
                            "D_ratio",
                            "h2_primary_validation_all_shared",
                        ),
                        "base_descent_margin": (
                            "D_mbase",
                            "h1_primary_validation_all_shared",
                        ),
                        "base_descent_failure_fraction": (
                            "D_mfail",
                            "h1_primary_validation_all_shared",
                        ),
                    }
                    decision_endpoint, decision_family = mapping.get(metric, ("", ""))
                ratio_applicable = metric == "weighted_gradient_norm_ratio"
                relevant_bootstrap = [
                    bootstrap.mean_difference.status,
                    bootstrap.median_difference.status,
                    *([bootstrap.median_ratio.status] if ratio_applicable else []),
                ]
                row = {
                    "schema_version": 1,
                    "split": split,
                    "group": group,
                    "metric": metric,
                    "primary_contrast": contrast,
                    "pass_n": 17,
                    "fail_n": 8,
                    "pass_mean": descriptive_pass["mean"],
                    "pass_sd": descriptive_pass["sd"],
                    "pass_median": descriptive_pass["median"],
                    "pass_q1": descriptive_pass["q1"],
                    "pass_q3": descriptive_pass["q3"],
                    "fail_mean": descriptive_fail["mean"],
                    "fail_sd": descriptive_fail["sd"],
                    "fail_median": descriptive_fail["median"],
                    "fail_q1": descriptive_fail["q1"],
                    "fail_q3": descriptive_fail["q3"],
                    "mean_difference_fail_minus_pass": bootstrap.mean_difference.estimate,
                    "mean_ci_low": bootstrap.mean_difference.ci_low,
                    "mean_ci_high": bootstrap.mean_difference.ci_high,
                    "median_difference_fail_minus_pass": bootstrap.median_difference.estimate,
                    "median_ci_low": bootstrap.median_difference.ci_low,
                    "median_ci_high": bootstrap.median_difference.ci_high,
                    "fail_over_pass_median_ratio": (
                        bootstrap.median_ratio.estimate if ratio_applicable else None
                    ),
                    "ratio_ci_low": (
                        bootstrap.median_ratio.ci_low if ratio_applicable else None
                    ),
                    "ratio_ci_high": (
                        bootstrap.median_ratio.ci_high if ratio_applicable else None
                    ),
                    "ratio_status": (
                        bootstrap.median_ratio.status
                        if ratio_applicable
                        else "not_applicable"
                    ),
                    "permutation_method": "exact_crossed_seed_order_x_fold_order",
                    "permutation_statistic": permutation.estimate,
                    "permutation_p_raw_two_sided": permutation.p_value,
                    "permutation_extreme_count": permutation.extreme_count,
                    "permutation_replicates": permutation.replicates,
                    "permutation_identity_included": permutation.includes_identity,
                    "permutation_status": permutation.status,
                    "mann_whitney_u": float(mann.statistic),
                    "mann_whitney_p_two_sided": float(mann.pvalue),
                    "decision_endpoint": decision_endpoint,
                    "decision_family": decision_family,
                    "decision_family_size": 0,
                    "p_holm": None,
                    "holm_rank": None,
                    "bootstrap_method": "crossed_seed_fold_cartesian_metric_gate_pairs",
                    "mean_bootstrap_requested": bootstrap.mean_difference.bootstrap_requested,
                    "mean_bootstrap_finite": bootstrap.mean_difference.bootstrap_finite,
                    "mean_bootstrap_nonfinite": (
                        bootstrap.mean_difference.bootstrap_requested
                        - bootstrap.mean_difference.bootstrap_finite
                    ),
                    "mean_bootstrap_finite_fraction": bootstrap.mean_difference.bootstrap_finite_fraction,
                    "mean_bootstrap_status": bootstrap.mean_difference.status,
                    "median_bootstrap_requested": bootstrap.median_difference.bootstrap_requested,
                    "median_bootstrap_finite": bootstrap.median_difference.bootstrap_finite,
                    "median_bootstrap_nonfinite": (
                        bootstrap.median_difference.bootstrap_requested
                        - bootstrap.median_difference.bootstrap_finite
                    ),
                    "median_bootstrap_finite_fraction": bootstrap.median_difference.bootstrap_finite_fraction,
                    "median_bootstrap_status": bootstrap.median_difference.status,
                    "ratio_bootstrap_requested": (
                        bootstrap.median_ratio.bootstrap_requested
                        if ratio_applicable
                        else 0
                    ),
                    "ratio_bootstrap_finite": (
                        bootstrap.median_ratio.bootstrap_finite
                        if ratio_applicable
                        else 0
                    ),
                    "ratio_bootstrap_nonfinite": (
                        bootstrap.median_ratio.bootstrap_requested
                        - bootstrap.median_ratio.bootstrap_finite
                        if ratio_applicable
                        else 0
                    ),
                    "ratio_bootstrap_finite_fraction": (
                        bootstrap.median_ratio.bootstrap_finite_fraction
                        if ratio_applicable
                        else None
                    ),
                    "ratio_bootstrap_status": (
                        bootstrap.median_ratio.status
                        if ratio_applicable
                        else "not_applicable"
                    ),
                    "status": (
                        "ok"
                        if all(value == "ok" for value in relevant_bootstrap)
                        else "ci_unavailable"
                    ),
                    "resampling_bundle_sha256": bundle_sha,
                    "contains_patient_ids": False,
                }
                if tuple(row) != PASS_FAIL_COLUMNS:
                    raise AssertionError("PASS/FAIL row schema/order 漂移")
                rows.append(row)
    if len(rows) != 84:
        raise AssertionError("PASS/FAIL row count 不是84")
    return rows


def build_gradient_correlations(
    layer: pd.DataFrame, indices: ResamplingIndices, bundle_sha: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        for group in GROUPS:
            run = _fixed(layer, split, group)
            x = _grid(run, "base_degradation")
            for endpoint in CORRELATION_ENDPOINTS:
                y = _grid(run, endpoint)
                result = spearman_crossed_ci(
                    x, y, indices.crossed_seed_draws, indices.crossed_fold_draws
                )
                permutation = exact_crossed_spearman_permutation(
                    x,
                    y,
                    indices.crossed_permutation_seed_orders,
                    indices.crossed_permutation_fold_orders,
                )
                decision_endpoint = ""
                decision_family = ""
                if split == "validation" and group == "all_shared":
                    mapping = {
                        "gradient_cosine": (
                            "rho_cos",
                            "h1_primary_validation_all_shared",
                        ),
                        "base_descent_margin": (
                            "rho_mbase",
                            "h1_primary_validation_all_shared",
                        ),
                        "weighted_gradient_norm_ratio": (
                            "rho_ratio",
                            "h2_primary_validation_all_shared",
                        ),
                    }
                    decision_endpoint, decision_family = mapping.get(endpoint, ("", ""))
                row = {
                    "schema_version": 1,
                    "split": split,
                    "group": group,
                    "endpoint": endpoint,
                    "analysis_unit": "seed_fold_run",
                    "x": "base_degradation",
                    "y": endpoint,
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
                    "decision_endpoint": decision_endpoint,
                    "decision_family": decision_family,
                    "decision_family_size": 0,
                    "p_holm": None,
                    "holm_rank": None,
                    "status": result.status,
                    "bootstrap_method": "crossed_seed_fold_cartesian_percentile",
                    "bootstrap_requested": result.bootstrap_requested,
                    "bootstrap_finite": result.bootstrap_finite,
                    "bootstrap_nonfinite": result.bootstrap_requested
                    - result.bootstrap_finite,
                    "bootstrap_finite_fraction": result.bootstrap_finite_fraction,
                    "resampling_bundle_sha256": bundle_sha,
                    "contains_patient_ids": False,
                }
                if tuple(row) != CORRELATION_COLUMNS:
                    raise AssertionError("gradient correlation row schema/order 漂移")
                rows.append(row)
    if len(rows) != 56:
        raise AssertionError("gradient correlation row count 不是56")
    return rows


def apply_decision_holm(
    pass_rows: list[dict[str, Any]], correlation_rows: list[dict[str, Any]]
) -> None:
    for family, size in (
        ("h1_primary_validation_all_shared", 6),
        ("h2_primary_validation_all_shared", 2),
    ):
        targets = [
            row
            for row in (*pass_rows, *correlation_rows)
            if row["decision_family"] == family
        ]
        raw = {
            str(row["decision_endpoint"]): row.get(
                "permutation_p_raw_two_sided", row.get("p_raw_two_sided")
            )
            for row in targets
        }
        if len(raw) != size:
            raise AssertionError(f"{family} family size 不是{size}")
        adjusted = holm_adjust(raw)
        for row in targets:
            result = adjusted[str(row["decision_endpoint"])]
            row["decision_family_size"] = result.family_size
            row["p_holm"] = result.p_holm
            row["holm_rank"] = result.rank


def build_dynamics(
    existing: pd.DataFrame, indices: ResamplingIndices, bundle_sha: str
) -> list[dict[str, Any]]:
    definitions = (
        ("selected_epoch", "selected_epoch", 1, "更晚 selected 表示更高累计优化暴露"),
        ("last_epoch", "last_epoch", 1, "更晚 last 表示更长训练"),
        (
            "last_minus_selected_epoch",
            "last_minus_selected_epoch",
            1,
            "更大 selection gap 表示更多 post-selected 更新",
        ),
        (
            "representation_collapse",
            "representation_std",
            -1,
            "更低 representation std 表示 collapse risk",
        ),
        (
            "ftv_pressure",
            "available_val_ftv_loss",
            -1,
            "更低 validation FTV loss 表示更强 FTV pressure",
        ),
        (
            "cumulative_grounded_exposure",
            "cumulative_grounded_exposure_to_selected",
            1,
            "更多 selected 前 grounded exposure",
        ),
        (
            "post_selected_base_deterioration",
            "post_selected_val_state_slope",
            1,
            "更正的 post-selected base slope 表示恶化",
        ),
        (
            "post_selected_ftv_improvement",
            "post_selected_val_ftv_slope",
            -1,
            "更负的 post-selected FTV slope 表示持续改善",
        ),
    )
    fixed = existing.sort_values(["seed_base", "fold"]).reset_index(drop=True)
    x = _grid(fixed, "base_degradation")
    calculated: dict[str, tuple[Any, Any]] = {}
    for endpoint, source, multiplier, _ in definitions:
        oriented = fixed.copy()
        oriented["oriented"] = oriented[source].astype(float) * multiplier
        y = _grid(oriented, "oriented")
        interval = spearman_crossed_ci(
            x, y, indices.crossed_seed_draws, indices.crossed_fold_draws
        )
        permutation = exact_crossed_spearman_permutation(
            x,
            y,
            indices.crossed_permutation_seed_orders,
            indices.crossed_permutation_fold_orders,
        )
        calculated[endpoint] = (interval, permutation)
    adjusted = holm_adjust(
        {endpoint: calculated[endpoint][1].p_value for endpoint, *_ in definitions}
    )
    rows: list[dict[str, Any]] = []
    for endpoint, source, multiplier, interpretation in definitions:
        result, permutation = calculated[endpoint]
        holm = adjusted[endpoint]
        row = {
            "schema_version": 1,
            "endpoint": endpoint,
            "source_variable": source,
            "risk_orientation_multiplier": multiplier,
            "risk_interpretation": interpretation,
            "analysis_unit": "seed_fold_run",
            "n": 25,
            "spearman_rho_oriented": result.estimate,
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
            "family_id": "h3_dynamics",
            "family_size": holm.family_size,
            "status": result.status,
            "bootstrap_method": "crossed_seed_fold_cartesian_percentile",
            "bootstrap_requested": result.bootstrap_requested,
            "bootstrap_finite": result.bootstrap_finite,
            "bootstrap_nonfinite": result.bootstrap_requested - result.bootstrap_finite,
            "bootstrap_finite_fraction": result.bootstrap_finite_fraction,
            "resampling_bundle_sha256": bundle_sha,
            "contains_patient_ids": False,
        }
        if tuple(row) != DYNAMICS_COLUMNS:
            raise AssertionError("dynamics row schema/order 漂移")
        rows.append(row)
    if len(rows) != 8:
        raise AssertionError("dynamics rows 不是8")
    return rows


def _rowwise_spearman(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("rowwise Spearman shape 非法")
    x_rank = rankdata(x, axis=1, method="average")
    y_rank = rankdata(y, axis=1, method="average")
    x_centered = x_rank - x_rank.mean(axis=1, keepdims=True)
    y_centered = y_rank - y_rank.mean(axis=1, keepdims=True)
    denominator = np.sqrt(
        np.sum(np.square(x_centered), axis=1) * np.sum(np.square(y_centered), axis=1)
    )
    result = np.full(x.shape[0], np.nan, dtype=np.float64)
    valid = denominator > 0
    result[valid] = (
        np.sum(x_centered[valid] * y_centered[valid], axis=1) / denominator[valid]
    )
    return result


def _fold_spearman(
    x: np.ndarray, y: np.ndarray, indices: ResamplingIndices
) -> dict[str, Any]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if (
        x.shape != (5,)
        or y.shape != (5,)
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("fold Spearman 必须是 finite n=5")
    draws = np.asarray(indices.crossed_fold_draws, dtype=np.int64)
    orders = np.unique(
        np.asarray(indices.crossed_permutation_fold_orders, dtype=np.int64), axis=0
    )
    identity = np.arange(5, dtype=np.int64)
    if (
        draws.ndim != 2
        or draws.shape[1] != 5
        or orders.shape != (120, 5)
        or not np.any(np.all(orders == identity, axis=1))
    ):
        raise ValueError("fold resampling/permutation contract 漂移")
    requested = len(draws)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return {
            "estimate": None,
            "ci_low": None,
            "ci_high": None,
            "p_value": None,
            "permutation_replicates": 120,
            "permutation_extreme_count": 0,
            "permutation_status": "constant_input",
            "bootstrap_requested": requested,
            "bootstrap_finite": 0,
            "bootstrap_finite_fraction": 0.0,
            "status": "constant_input",
        }
    estimate = float(_rowwise_spearman(x[None, :], y[None, :])[0])
    boot = _rowwise_spearman(x[draws], y[draws])
    finite = boot[np.isfinite(boot)]
    fraction = float(len(finite) / requested)
    if fraction >= 0.95:
        ci_low, ci_high = (
            float(value)
            for value in np.quantile(finite, [0.025, 0.975], method="linear")
        )
        status = "ok"
    else:
        ci_low = ci_high = None
        status = "insufficient_finite_bootstrap"
    permuted = _rowwise_spearman(np.broadcast_to(x, (len(orders), 5)), y[orders])
    if not np.isfinite(permuted).all():
        raise ValueError("fold exact permutation 产生 nonfinite")
    extreme = int(np.count_nonzero(np.abs(permuted) >= abs(estimate)))
    return {
        "estimate": estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": float(extreme / len(orders)),
        "permutation_replicates": len(orders),
        "permutation_extreme_count": extreme,
        "permutation_status": "ok",
        "bootstrap_requested": requested,
        "bootstrap_finite": len(finite),
        "bootstrap_finite_fraction": fraction,
        "status": status,
    }


def build_coverage_correlations(
    existing: pd.DataFrame,
    coverage: pd.DataFrame,
    indices: ResamplingIndices,
    bundle_sha: str,
) -> list[dict[str, Any]]:
    degradation = (
        existing.groupby("fold", sort=True)["base_degradation"].median().reindex(FOLDS)
    )
    definitions = (
        ("train_pool_ftv_proportion", "train", "pool_ftv_proportion"),
        ("validation_pool_ftv_proportion", "validation", "pool_ftv_proportion"),
        ("train_batch_ftv_count_sd", "train", "batch_ftv_sd"),
        ("validation_batch_ftv_count_sd", "validation", "batch_ftv_sd"),
    )
    rows: list[dict[str, Any]] = []
    y5 = degradation.to_numpy(dtype=float)
    for endpoint, split, column in definitions:
        current = (
            coverage.loc[coverage["split"].eq(split)].set_index("fold").reindex(FOLDS)
        )
        x5 = current[column].to_numpy(dtype=float)
        if not np.isfinite(x5).all() or not np.isfinite(y5).all():
            raise ValueError("coverage correlation nonfinite")
        result = _fold_spearman(x5, y5, indices)
        row = {
            "schema_version": 1,
            "endpoint": endpoint,
            "split": split,
            "coverage_variable": column,
            "analysis_unit": "fold",
            "n": 5,
            "spearman_rho": result["estimate"],
            "ci_low": result["ci_low"],
            "ci_high": result["ci_high"],
            "p_raw_two_sided": result["p_value"],
            "permutation_method": "exact_fold_label_orders",
            "permutation_replicates": result["permutation_replicates"],
            "permutation_extreme_count": result["permutation_extreme_count"],
            "permutation_identity_included": True,
            "permutation_status": result["permutation_status"],
            "status": result["status"],
            "bootstrap_method": "fold_level_percentile_synchronized_fold_draws",
            "bootstrap_requested": result["bootstrap_requested"],
            "bootstrap_finite": result["bootstrap_finite"],
            "bootstrap_nonfinite": result["bootstrap_requested"]
            - result["bootstrap_finite"],
            "bootstrap_finite_fraction": result["bootstrap_finite_fraction"],
            "resampling_bundle_sha256": bundle_sha,
            "contains_patient_ids": False,
        }
        if tuple(row) != COVERAGE_CORRELATION_COLUMNS:
            raise AssertionError("coverage correlation row schema/order 漂移")
        rows.append(row)
    return rows


def _pass_row(
    rows: Iterable[dict[str, Any]], split: str, group: str, metric: str
) -> dict[str, Any]:
    matched = [
        row
        for row in rows
        if row["split"] == split and row["group"] == group and row["metric"] == metric
    ]
    if len(matched) != 1:
        raise ValueError(f"PASS/FAIL row 缺失: {split}/{group}/{metric}")
    return matched[0]


def build_localization(pass_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    endpoint_metrics = {
        "cosine": "gradient_cosine",
        "mbase": "base_descent_margin",
        "ratio": "weighted_gradient_norm_ratio",
    }
    family_raw: dict[str, float] = {}
    base: list[dict[str, Any]] = []
    for group in LOCALIZATION_GROUPS:
        val = {
            name: _pass_row(pass_rows, "validation", group, metric)
            for name, metric in endpoint_metrics.items()
        }
        train = {
            name: _pass_row(pass_rows, "train", group, metric)
            for name, metric in endpoint_metrics.items()
        }
        item = {
            "group": group,
            "validation_D_cos": val["cosine"]["median_difference_fail_minus_pass"],
            "validation_D_mbase": val["mbase"]["median_difference_fail_minus_pass"],
            "validation_D_ratio": val["ratio"]["median_difference_fail_minus_pass"],
            "train_D_cos": train["cosine"]["median_difference_fail_minus_pass"],
            "train_D_mbase": train["mbase"]["median_difference_fail_minus_pass"],
            "train_D_ratio": train["ratio"]["median_difference_fail_minus_pass"],
            "validation_cosine_p_raw": val["cosine"]["permutation_p_raw_two_sided"],
            "validation_mbase_p_raw": val["mbase"]["permutation_p_raw_two_sided"],
            "validation_ratio_p_raw": val["ratio"]["permutation_p_raw_two_sided"],
        }
        for name in endpoint_metrics:
            family_raw[f"{group}::{name}"] = float(item[f"validation_{name}_p_raw"])
        base.append(item)
    adjusted = holm_adjust(family_raw)
    frame = pd.DataFrame(base)
    frame["cosine_rank"] = frame["validation_D_cos"].rank(
        method="average", ascending=True
    )
    frame["mbase_rank"] = frame["validation_D_mbase"].rank(
        method="average", ascending=True
    )
    frame["ratio_rank"] = frame["validation_D_ratio"].rank(
        method="average", ascending=False
    )
    frame["localization_score"] = frame[
        ["cosine_rank", "mbase_rank", "ratio_rank"]
    ].mean(axis=1)
    frame["localization_rank"] = frame["localization_score"].rank(
        method="average", ascending=True
    )
    crosses = (frame["validation_D_cos"] <= -0.10) | (
        frame["validation_D_mbase"] <= -0.10
    )
    widespread_count = int(crosses.sum())
    rows: list[dict[str, Any]] = []
    for index, item in frame.iterrows():
        group = str(item["group"])
        row = {
            "schema_version": 1,
            "group": group,
            "validation_D_cos": float(item["validation_D_cos"]),
            "validation_D_mbase": float(item["validation_D_mbase"]),
            "validation_D_ratio": float(item["validation_D_ratio"]),
            "train_D_cos": float(item["train_D_cos"]),
            "train_D_mbase": float(item["train_D_mbase"]),
            "train_D_ratio": float(item["train_D_ratio"]),
            "train_cosine_direction_replicated": float(item["train_D_cos"]) < 0,
            "train_mbase_direction_replicated": float(item["train_D_mbase"]) < 0,
            "train_ratio_direction_replicated": float(item["train_D_ratio"]) > 0,
            "cosine_rank": float(item["cosine_rank"]),
            "mbase_rank": float(item["mbase_rank"]),
            "ratio_rank": float(item["ratio_rank"]),
            "localization_score": float(item["localization_score"]),
            "localization_rank": float(item["localization_rank"]),
            "family_id": "layer_localization_validation_5x3",
            "validation_cosine_p_raw": float(item["validation_cosine_p_raw"]),
            "validation_cosine_p_holm": adjusted[f"{group}::cosine"].p_holm,
            "validation_cosine_holm_rank": adjusted[f"{group}::cosine"].rank,
            "validation_cosine_holm_status": adjusted[f"{group}::cosine"].status,
            "validation_mbase_p_raw": float(item["validation_mbase_p_raw"]),
            "validation_mbase_p_holm": adjusted[f"{group}::mbase"].p_holm,
            "validation_mbase_holm_rank": adjusted[f"{group}::mbase"].rank,
            "validation_mbase_holm_status": adjusted[f"{group}::mbase"].status,
            "validation_ratio_p_raw": float(item["validation_ratio_p_raw"]),
            "validation_ratio_p_holm": adjusted[f"{group}::ratio"].p_holm,
            "validation_ratio_holm_rank": adjusted[f"{group}::ratio"].rank,
            "validation_ratio_holm_status": adjusted[f"{group}::ratio"].status,
            "holm_family_size": 15,
            "layer_crosses_direction_threshold": bool(crosses.iloc[index]),
            "widespread_layer_count": widespread_count,
            "widespread_conflict": widespread_count >= 4,
            "contains_patient_ids": False,
        }
        if tuple(row) != LOCALIZATION_COLUMNS:
            raise AssertionError("localization row schema/order 漂移")
        rows.append(row)
    return rows


def build_fold_signature(run: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for split in SPLITS:
        current = _fixed(run, split, "all_shared")
        fold_rows: list[dict[str, Any]] = []
        for fold in FOLDS:
            own = current.loc[current["fold"].eq(fold)]
            other = current.loc[current["fold"].ne(fold)]
            fold_rows.append(
                {
                    "fold": fold,
                    "split": split,
                    "n_runs": len(own),
                    "pass_n": int(own["base_gate"].eq("PASS").sum()),
                    "fail_n": int(own["base_gate"].eq("FAIL").sum()),
                    "median_gradient_cosine": float(own["gradient_cosine"].median()),
                    "median_base_descent_margin": float(
                        own["base_descent_margin"].median()
                    ),
                    "median_norm_ratio": float(
                        own["weighted_gradient_norm_ratio"].median()
                    ),
                    "cosine_minus_other_runs": float(
                        own["gradient_cosine"].median()
                        - other["gradient_cosine"].median()
                    ),
                    "mbase_minus_other_runs": float(
                        own["base_descent_margin"].median()
                        - other["base_descent_margin"].median()
                    ),
                    "ratio_minus_other_runs": float(
                        own["weighted_gradient_norm_ratio"].median()
                        - other["weighted_gradient_norm_ratio"].median()
                    ),
                }
            )
        temp = pd.DataFrame(fold_rows)
        temp["cosine_rank_ascending"] = temp["median_gradient_cosine"].rank(
            method="average", ascending=True
        )
        temp["mbase_rank_ascending"] = temp["median_base_descent_margin"].rank(
            method="average", ascending=True
        )
        temp["ratio_rank_descending"] = temp["median_norm_ratio"].rank(
            method="average", ascending=False
        )
        fold3 = temp.loc[temp["fold"].eq(3)].iloc[0]
        others = temp.loc[temp["fold"].ne(3)]
        worst_cos = bool(
            float(fold3["median_gradient_cosine"])
            < float(others["median_gradient_cosine"].min())
        )
        worst_mbase = bool(
            float(fold3["median_base_descent_margin"])
            < float(others["median_base_descent_margin"].min())
        )
        crosses = bool(
            float(fold3["cosine_minus_other_runs"]) <= -0.10
            or float(fold3["mbase_minus_other_runs"]) <= -0.10
        )
        special = split == "validation" and worst_cos and worst_mbase and crosses
        for _, item in temp.iterrows():
            is_fold3 = int(item["fold"]) == 3
            row = {
                "schema_version": 1,
                "fold": int(item["fold"]),
                "split": split,
                "n_runs": int(item["n_runs"]),
                "pass_n": int(item["pass_n"]),
                "fail_n": int(item["fail_n"]),
                "median_gradient_cosine": float(item["median_gradient_cosine"]),
                "median_base_descent_margin": float(item["median_base_descent_margin"]),
                "median_norm_ratio": float(item["median_norm_ratio"]),
                "cosine_rank_ascending": float(item["cosine_rank_ascending"]),
                "mbase_rank_ascending": float(item["mbase_rank_ascending"]),
                "ratio_rank_descending": float(item["ratio_rank_descending"]),
                "cosine_minus_other_runs": float(item["cosine_minus_other_runs"]),
                "mbase_minus_other_runs": float(item["mbase_minus_other_runs"]),
                "ratio_minus_other_runs": float(item["ratio_minus_other_runs"]),
                "is_fold3": is_fold3,
                "fold3_strictly_worst_cosine": worst_cos if is_fold3 else False,
                "fold3_strictly_worst_mbase": worst_mbase if is_fold3 else False,
                "fold3_crosses_practical_threshold": crosses if is_fold3 else False,
                "fold3_special_signature": special if is_fold3 else False,
                "contains_patient_ids": False,
            }
            if tuple(row) != FOLD_SIGNATURE_COLUMNS:
                raise AssertionError("fold signature row schema/order 漂移")
            records.append(row)
    return records


def _decision_signals(
    pass_rows: list[dict[str, Any]],
    corr_rows: list[dict[str, Any]],
    dynamics_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    def corr(split: str, endpoint: str) -> dict[str, Any]:
        match = [
            row
            for row in corr_rows
            if row["split"] == split
            and row["group"] == "all_shared"
            and row["endpoint"] == endpoint
        ]
        if len(match) != 1:
            raise ValueError("decision correlation row 缺失")
        return match[0]

    blocks: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        cosine = _pass_row(pass_rows, split, "all_shared", "gradient_cosine")
        negative = _pass_row(pass_rows, split, "all_shared", "negative_fraction")
        mbase = _pass_row(pass_rows, split, "all_shared", "base_descent_margin")
        mfail = _pass_row(
            pass_rows, split, "all_shared", "base_descent_failure_fraction"
        )
        ratio = _pass_row(
            pass_rows, split, "all_shared", "weighted_gradient_norm_ratio"
        )
        blocks[split] = {
            "D_cos": cosine["median_difference_fail_minus_pass"],
            "D_neg": negative["mean_difference_fail_minus_pass"],
            "D_mbase": mbase["median_difference_fail_minus_pass"],
            "D_mfail": mfail["mean_difference_fail_minus_pass"],
            "D_ratio": ratio["median_difference_fail_minus_pass"],
            "Q_ratio": ratio["fail_over_pass_median_ratio"],
            "rho_cos": corr(split, "gradient_cosine")["spearman_rho"],
            "rho_mbase": corr(split, "base_descent_margin")["spearman_rho"],
            "rho_ratio": corr(split, "weighted_gradient_norm_ratio")["spearman_rho"],
        }
    h1 = {
        endpoint: next(
            row["p_holm"]
            for row in (*pass_rows, *corr_rows)
            if row.get("decision_endpoint") == endpoint
        )
        for endpoint in ("D_cos", "D_neg", "D_mbase", "rho_cos", "rho_mbase", "D_mfail")
    }
    h2 = {
        endpoint: next(
            row["p_holm"]
            for row in (*pass_rows, *corr_rows)
            if row.get("decision_endpoint") == endpoint
        )
        for endpoint in ("D_ratio", "rho_ratio")
    }
    dynamics = {
        str(row["endpoint"]): {
            "oriented_rho": row["spearman_rho_oriented"],
            "p_holm": row["p_holm"],
        }
        for row in dynamics_rows
    }
    if set(dynamics) != set(DYNAMICS_ENDPOINTS):
        raise ValueError("H3 dynamics endpoint set 漂移")
    return {
        "core_eligible": True,
        "validation": blocks["validation"],
        "train": blocks["train"],
        "h1_holm_p": h1,
        "h2_holm_p": h2,
        "dynamics": dynamics,
    }


def write_full_analysis(root: str | Path = AUDIT_ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    if root != AUDIT_ROOT.resolve():
        raise ValueError("formal analysis root 必须是冻结 audit root")
    assert_source_contract(full_content_hash=False, full_checkpoint_hash=False)
    aggregate_stems = (
        "layer_level_conflict_metrics",
        "run_level_conflict_metrics",
        "trajectory_conflict_metrics",
        "trajectory_change_metrics",
        "ftv_coverage_metrics",
    )
    aggregate_paths = [root / "metrics" / f"{stem}.csv" for stem in aggregate_stems]
    if not all(path.is_file() for path in aggregate_paths):
        raise FileNotFoundError("必须先完整运行 aggregate_metrics；拒绝静默生成/补写")
    aggregate_frames = {
        stem: pd.read_csv(path)
        for stem, path in zip(aggregate_stems, aggregate_paths, strict=True)
    }
    representatives = pd.read_csv(root / "metrics" / "representative_runs.csv")
    _validate_written_aggregates(aggregate_frames, representatives)
    _assert_aggregate_equivalence(aggregate_frames, build_aggregate_tables(root))
    layer = aggregate_frames["layer_level_conflict_metrics"]
    run = aggregate_frames["run_level_conflict_metrics"]
    coverage = aggregate_frames["ftv_coverage_metrics"]
    existing = pd.read_csv(root / "metrics" / "run_level_existing_metrics.csv")
    indices, bundle_manifest = load_resampling_bundle()
    bundle_sha = str(bundle_manifest["bundle_sha256"])
    pass_rows = build_pass_fail_table(layer, indices, bundle_sha)
    corr_rows = build_gradient_correlations(layer, indices, bundle_sha)
    apply_decision_holm(pass_rows, corr_rows)
    dynamics_rows = build_dynamics(existing, indices, bundle_sha)
    coverage_rows = build_coverage_correlations(existing, coverage, indices, bundle_sha)
    localization_rows = build_localization(pass_rows)
    fold_rows = build_fold_signature(run)
    signals = _decision_signals(pass_rows, corr_rows, dynamics_rows)
    decision = evaluate_hypotheses(signals)
    outputs: dict[str, tuple[list[dict[str, Any]], tuple[str, ...], int]] = {
        "pass_fail_comparison.csv": (pass_rows, PASS_FAIL_COLUMNS, 84),
        "gradient_correlation_metrics.csv": (corr_rows, CORRELATION_COLUMNS, 56),
        "dynamics_correlations.csv": (dynamics_rows, DYNAMICS_COLUMNS, 8),
        "coverage_correlations.csv": (coverage_rows, COVERAGE_CORRELATION_COLUMNS, 4),
        "layer_localization_metrics.csv": (localization_rows, LOCALIZATION_COLUMNS, 5),
        "fold_signature_metrics.csv": (fold_rows, FOLD_SIGNATURE_COLUMNS, 10),
    }
    for filename, (rows, columns, expected) in outputs.items():
        if len(rows) != expected or any(tuple(row) != columns for row in rows):
            raise AssertionError(f"analysis output contract 失败: {filename}")
    decision_rows: list[dict[str, Any]] = []
    for item in decision["decision_rows"]:
        row = {"schema_version": 1, **item, "contains_patient_ids": False}
        if tuple(row) != HYPOTHESIS_DECISION_COLUMNS:
            raise AssertionError("hypothesis decision schema 漂移")
        decision_rows.append(row)
    if (
        len(decision_rows) != 4
        or {row["hypothesis"] for row in decision_rows} != {"H1", "H2", "H3", "H4"}
        or sum(bool(row["selected"]) for row in decision_rows) != 1
    ):
        raise AssertionError("hypothesis decision 4-row/唯一性合同失败")
    diagnosis = {
        **decision,
        "signals": signals,
        "source_contract_sha256": file_sha256(SOURCE_CONTRACT),
        "resampling_bundle_sha256": bundle_sha,
        "resampling_npz_sha256": file_sha256(RESAMPLING_BUNDLE),
        "fix_executed": False,
        "causal_limitation": (
            "固定 checkpoint 上的 post-hoc gradient geometry 是 objective-level association，"
            "不是优化干预或原训练 RNG 的 bit-exact replay。"
        ),
        "contains_patient_ids": False,
    }
    csv_outputs = {
        filename: (rows, columns) for filename, (rows, columns, _) in outputs.items()
    }
    csv_outputs["hypothesis_decision.csv"] = (
        decision_rows,
        HYPOTHESIS_DECISION_COLUMNS,
    )
    provenance = _publish_analysis_bundle(
        root,
        csv_outputs=csv_outputs,
        diagnosis=diagnosis,
    )
    return {
        "rows": {
            **{filename: expected for filename, (_, _, expected) in outputs.items()},
            "hypothesis_decision.csv": 4,
        },
        "selected_hypothesis": decision["selected_hypothesis"],
        "first_recommendation": decision["first_recommendation"],
        "second_recommendation": decision["second_recommendation"],
        **provenance,
    }


def synthetic_self_test() -> dict[str, Any]:
    """用合成 25-run grid 覆盖统计表、Holm family 与唯一判定闭环。"""

    # 复用 aggregation 自测 fixture，避免另造一套可能与正式列契约脱节的假数据。
    from .aggregation import (  # noqa: PLC0415
        _expected_run_set,
        _synthetic_existing,
        _synthetic_gradient_rows,
        _synthetic_manifests,
        aggregate_coverage,
        aggregate_selected,
        aggregate_trajectory,
    )

    existing, representatives = _synthetic_existing()
    selected = _synthetic_gradient_rows(
        existing, sorted(_expected_run_set()), "selected"
    )
    public, private = _synthetic_manifests()
    layer, run = aggregate_selected(selected, existing)
    representative_runs = list(
        representatives[["seed_base", "fold"]].itertuples(index=False, name=None)
    )
    last = _synthetic_gradient_rows(existing, representative_runs, "last")
    trajectory, change = aggregate_trajectory(selected, last, representatives, existing)
    coverage = aggregate_coverage(public, private)
    _validate_written_aggregates(
        {
            "layer_level_conflict_metrics": layer,
            "run_level_conflict_metrics": run,
            "trajectory_conflict_metrics": trajectory,
            "trajectory_change_metrics": change,
            "ftv_coverage_metrics": coverage,
        },
        representatives,
        expected_source_sha="a" * 64,
        expected_public_sha="b" * 64,
    )
    indices = generate_resampling_indices(
        crossed_replicates=100,
        crossed_seed=2026080801,
    )
    bundle_sha = "f" * 64
    pass_rows = build_pass_fail_table(layer, indices, bundle_sha)
    correlation_rows = build_gradient_correlations(layer, indices, bundle_sha)
    apply_decision_holm(pass_rows, correlation_rows)
    dynamics_rows = build_dynamics(existing, indices, bundle_sha)
    coverage_rows = build_coverage_correlations(existing, coverage, indices, bundle_sha)
    localization_rows = build_localization(pass_rows)
    fold_rows = build_fold_signature(run)
    signals = _decision_signals(pass_rows, correlation_rows, dynamics_rows)
    decision = evaluate_hypotheses(signals)
    fold_probe = _fold_spearman(
        np.arange(5, dtype=np.float64),
        np.arange(5, dtype=np.float64),
        indices,
    )
    checks = {
        "pass_fail_rows_84": len(pass_rows) == 84,
        "gradient_correlation_rows_56": len(correlation_rows) == 56,
        "dynamics_rows_8": len(dynamics_rows) == 8,
        "coverage_correlation_rows_4": len(coverage_rows) == 4,
        "localization_rows_5": len(localization_rows) == 5,
        "fold_signature_rows_10": len(fold_rows) == 10,
        "all_written_aggregates_revalidated": True,
        "crossed_exact_replicates_14400": all(
            row["permutation_replicates"] == 14_400
            for rows in (pass_rows, correlation_rows, dynamics_rows)
            for row in rows
        ),
        "crossed_bootstrap_requested_100": all(
            row["mean_bootstrap_requested"] == 100 for row in pass_rows
        )
        and all(
            row["bootstrap_requested"] == 100
            for rows in (correlation_rows, dynamics_rows)
            for row in rows
        ),
        "fold_exact_120": (
            fold_probe["permutation_replicates"] == 120
            and fold_probe["estimate"] == 1.0
            and fold_probe["permutation_status"] == "ok"
        ),
        "h1_family_six": sum(
            row["decision_family"] == "h1_primary_validation_all_shared"
            for row in (*pass_rows, *correlation_rows)
        )
        == 6,
        "h2_family_two": sum(
            row["decision_family"] == "h2_primary_validation_all_shared"
            for row in (*pass_rows, *correlation_rows)
        )
        == 2,
        "one_selected_hypothesis": sum(
            bool(row["selected"]) for row in decision["decision_rows"]
        )
        == 1,
        "recommendations_distinct": (
            decision["first_recommendation"] != decision["second_recommendation"]
        ),
        "public_privacy_flags_false": all(
            not bool(row["contains_patient_ids"])
            for rows in (
                pass_rows,
                correlation_rows,
                dynamics_rows,
                coverage_rows,
                localization_rows,
                fold_rows,
            )
            for row in rows
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"analysis synthetic self-test 失败: {checks}")
    return {
        "status": "ok",
        "selected_hypothesis": decision["selected_hypothesis"],
        "checks": checks,
    }


__all__ = [
    "ANALYSIS_INPUT_MANIFEST_COLUMNS",
    "CORRELATION_COLUMNS",
    "COVERAGE_CORRELATION_COLUMNS",
    "DYNAMICS_COLUMNS",
    "FINAL_ANALYSIS_MANIFEST",
    "FINAL_ANALYSIS_MARKER",
    "FOLD_SIGNATURE_COLUMNS",
    "HYPOTHESIS_DECISION_COLUMNS",
    "LOCALIZATION_COLUMNS",
    "PASS_FAIL_COLUMNS",
    "apply_decision_holm",
    "build_coverage_correlations",
    "build_dynamics",
    "build_fold_signature",
    "build_gradient_correlations",
    "build_localization",
    "build_pass_fail_table",
    "validate_final_analysis_bundle",
    "write_full_analysis",
    "synthetic_self_test",
]

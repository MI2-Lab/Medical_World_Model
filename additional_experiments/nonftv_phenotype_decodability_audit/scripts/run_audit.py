#!/usr/bin/env python3
"""Run the complete frozen non-FTV decodability probe matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import pandas as pd

from audit_core import (
    AUDIT_ROOT,
    INTERVALS,
    MAIN_REPRESENTATIONS,
    OOFAccumulator,
    PrivatePredictionWriter,
    REPO_ROOT,
    VISITS,
    atomic_csv,
    atomic_json,
    authenticate,
    build_outcomes,
    dynamic_views,
    file_sha256,
    load_config,
    load_feature_cell,
    load_fold_splits,
    load_targets,
    localization_comparisons,
    probe_outcome_batch,
    representation_pair_comparisons,
    resolve_path,
    static_views,
)
from freeze_preregistration import require_preregistration_lock


def git_value(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPO_ROOT, text=True
    ).strip()


def _sort(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    available = [column for column in columns if column in frame.columns]
    return frame.sort_values(available, kind="stable").reset_index(drop=True)


def target_contract(config: dict[str, Any]) -> pd.DataFrame:
    definitions = {
        "FTV": "满足 PE/SER 阈值的增强组织功能性肿瘤体积（cc）",
        "LD": "Goal 6 工作簿最长径；工作簿未声明单位",
        "SPH": "等体积球表面积 / 3D FTV tumor-mask 表面积（无量纲）",
        "BPE": "对侧乳腺中央连续五层纤维腺体组织平均早期 percent enhancement",
    }
    rows = []
    for family, fields in config["targets"].items():
        for timing, field in zip(VISITS, fields, strict=True):
            rows.append(
                {
                    "family": family,
                    "timing": timing,
                    "workbook_field": field,
                    "definition": definitions[family],
                    "static_family_transform": "log1p" if family in config["target_transforms"]["log1p_families"] else "identity",
                    "winsorization": "outer_train_1pct_99pct",
                    "scaling": "outer_train_population_standardization",
                    "dynamic_formula": config["target_transforms"]["change_formula"],
                    "dynamic_status": "new_adjacent_interval_instantiation_of_goal6_formula",
                    "T3_timing_note": "late/pre-surgery" if timing == "T3" else "",
                }
            )
    return pd.DataFrame(rows)


def representation_contract(config: dict[str, Any]) -> pd.DataFrame:
    descriptions = {
        "Z1": "JEPA projector(r); transition 使用的 projected state",
        "Z2": "response_projection(fixed LOCAL mean) 后、JEPA projector 前的 r",
        "Z3": "final 128-channel spatial map 在固定 64-mm LOCAL support 的 weighted mean",
        "Z4": "同一 LOCAL support 的 [weighted mean, weighted population std]",
        "Z5": "Oracle CORE [mean,std]；mask-dependent diagnostic only",
        "Z6": "Oracle PERI10 (0,10] mm [mean,std]；mask-dependent diagnostic only",
        "Z7": "Oracle PERI20 (10,20] mm [mean,std]；mask-dependent diagnostic only",
    }
    rows = []
    for name, payload in config["representations"].items():
        rows.append(
            {
                "representation": name,
                "label": payload["label"],
                "dimension_static": int(payload["dimension"]),
                "dimension_difference": int(payload["dimension"]),
                "dimension_prefix_secondary": 3 * int(payload["dimension"]),
                "deployable": bool(payload["deployable"]),
                "description": descriptions[name],
                "encoder_retrained": False,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=AUDIT_ROOT / "configs" / "audit.json",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="authenticate target/fold/external completion contracts without fitting probes",
    )
    arguments = parser.parse_args()
    start_time = time.time()
    config = load_config(arguments.config)
    lock_evidence = require_preregistration_lock(require_exact_parent=True)

    branch = git_value("branch", "--show-current")
    if branch != config["branch"]:
        raise ValueError(f"formal audit must run on {config['branch']}; current={branch}")
    parent = git_value("rev-parse", "HEAD")
    if parent != config["parent_sha"]:
        raise ValueError(
            "formal audit parent drifted before first experiment commit: "
            f"expected {config['parent_sha']}, observed {parent}"
        )

    authenticate(
        config["paths"]["spatial_feature_root"] + "/feature_matrix_complete.private.json",
        config["paths"]["spatial_completion_sha256"],
        "spatial feature completion marker",
    )
    authenticate(
        config["paths"]["goal6_root"] + "/reports/final_report.md",
        config["paths"]["goal6_final_report_sha256"],
        "Goal 6 final report",
    )
    targets = load_targets(config)
    fold_splits = load_fold_splits(config, targets)
    static_outcomes, dynamic_outcomes, transform_rows, residualizer_rows = build_outcomes(
        config, targets, fold_splits
    )
    if arguments.preflight_only:
        payload = {
            "status": "PASS",
            "parent_sha": parent,
            "branch": branch,
            "patient_count": len(targets.patient_ids),
            "patient_set_sha256": targets.patient_set_sha256,
            "workbook_max_abs_difference": targets.workbook_max_abs_difference,
            "fold_count": len(fold_splits),
            "pcr_parsed": False,
            "preregistration_lock_sha256": lock_evidence["lock_sha256"],
            "elapsed_seconds": time.time() - start_time,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    run_summary_path = AUDIT_ROOT / "metrics" / "run_summary.json"
    prediction_path = AUDIT_ROOT / "predictions" / "oof_predictions.private.csv.gz"
    if run_summary_path.exists() or prediction_path.exists():
        raise FileExistsError("refusing to overwrite an existing formal audit run")

    writer = PrivatePredictionWriter(prediction_path)
    accumulator = OOFAccumulator()
    selection_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    try:
        cell_count = 0
        for seed in config["frozen"]["seeds"]:
            for arm in config["frozen"]["arms"]:
                for fold in config["frozen"]["folds"]:
                    cell = load_feature_cell(
                        config,
                        targets,
                        fold_splits,
                        seed=int(seed),
                        arm=str(arm),
                        fold=int(fold),
                    )
                    cell_count += 1
                    cell_rows.append(dict(cell.provenance))
                    print(
                        f"[{cell_count}/20] seed={seed} arm={arm} fold={fold} "
                        f"epoch={cell.selected_epoch}",
                        flush=True,
                    )
                    for visit_index, timing in enumerate(VISITS):
                        outcomes = static_outcomes[(int(fold), timing)]
                        for representation, (matrix, valid, matched_for) in static_views(
                            cell, visit_index
                        ).items():
                            probe_outcome_batch(
                                x=matrix,
                                feature_valid=valid,
                                outcomes=outcomes,
                                cell=cell,
                                representation=representation,
                                matched_reference_for=matched_for,
                                input_variant="current",
                                config=config,
                                writer=writer,
                                accumulator=accumulator,
                                selection_rows=selection_rows,
                                fold_metric_rows=fold_metric_rows,
                                coverage_rows=coverage_rows,
                            )
                    for interval_index, interval in enumerate(INTERVALS):
                        outcomes = dynamic_outcomes[(int(fold), interval)]
                        for input_variant in ("difference", "prefix"):
                            views = dynamic_views(
                                cell,
                                interval_index,
                                input_variant,
                                include_matched_references=input_variant == "difference",
                            )
                            for representation, (matrix, valid, matched_for) in views.items():
                                probe_outcome_batch(
                                    x=matrix,
                                    feature_valid=valid,
                                    outcomes=outcomes,
                                    cell=cell,
                                    representation=representation,
                                    matched_reference_for=matched_for,
                                    input_variant=input_variant,
                                    config=config,
                                    writer=writer,
                                    accumulator=accumulator,
                                    selection_rows=selection_rows,
                                    fold_metric_rows=fold_metric_rows,
                                    coverage_rows=coverage_rows,
                                )
        writer.close()
    except Exception:
        writer.abort()
        raise

    oof = accumulator.metrics_frame()
    oof = _sort(
        oof,
        [
            "task_type",
            "target_kind",
            "target",
            "timing",
            "interval",
            "input_variant",
            "seed",
            "arm",
            "representation",
        ],
    )
    # Main matrix: (4 raw + 5 residual) * 7 reps * 4 visits * 4 seed/arm
    # cells = 1,008 static rows, plus 9 * 7 * 3 intervals * 2 dynamic
    # inputs * 4 cells = 1,512.  Matched Z4 oracle references contribute
    # 9 * 3 * 4 * 4 = 432 static and 9 * 3 * 3 * 4 = 324 dynamic rows.
    expected_oof_rows = 3276
    if len(oof) != expected_oof_rows or not (oof["n_folds"] == 5).all():
        raise ValueError(
            f"OOF matrix completeness failed: rows={len(oof)} expected={expected_oof_rows}"
        )
    raw_oof = oof.loc[oof["target_kind"] == "raw"]
    residual_oof = oof.loc[oof["target_kind"] != "raw"]
    finite_contracts = {
        "raw": (
            raw_oof,
            (
                "spearman",
                "pearson",
                "natural_r2",
                "transformed_r2",
                "rmse",
                "mae",
                "prediction_target_variance_ratio",
                "calibration_slope",
            ),
        ),
        "residual": (
            residual_oof,
            (
                "residual_spearman",
                "residual_pearson",
                "residual_transformed_r2",
                "residual_rmse",
                "residual_mae",
                "prediction_target_variance_ratio",
                "calibration_slope",
                "reconstructed_natural_r2",
                "reconstructed_natural_rmse",
                "reconstructed_natural_mae",
            ),
        ),
    }
    for label, (frame, columns) in finite_contracts.items():
        values = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce").to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"{label} OOF metrics contain undefined/non-finite values")
    expected_private_prediction_rows = int(pd.to_numeric(oof["n"]).sum())
    if writer.rows != expected_private_prediction_rows:
        raise ValueError(
            "private OOF prediction completeness failed: "
            f"rows={writer.rows} expected={expected_private_prediction_rows}"
        )
    fold_metrics = _sort(
        pd.DataFrame(fold_metric_rows),
        ["task_type", "target_kind", "target", "timing", "interval", "input_variant", "seed", "arm", "representation", "fold"],
    )
    selections = _sort(
        pd.DataFrame(selection_rows),
        ["task_type", "target_kind", "target", "timing", "interval", "input_variant", "seed", "arm", "representation", "fold"],
    )
    coverage = _sort(
        pd.DataFrame(coverage_rows),
        ["task_type", "target_kind", "target", "timing", "interval", "input_variant", "seed", "arm", "representation", "fold"],
    )
    if len(fold_metrics) != expected_oof_rows * 5 or len(selections) != len(fold_metrics):
        raise ValueError("fold-level probe matrix is incomplete")

    main = oof.loc[oof["representation"].isin(MAIN_REPRESENTATIONS)].copy()
    static_matrix = main.loc[(main["task_type"] == "static") & (main["target_kind"] == "raw")].copy()
    residual_matrix = main.loc[(main["task_type"] == "static") & (main["target_kind"] != "raw")].copy()
    dynamic_matrix = main.loc[main["task_type"] == "dynamic"].copy()
    if (len(static_matrix), len(residual_matrix), len(dynamic_matrix)) != (448, 560, 1512):
        raise ValueError("public target matrices have unexpected dimensions")
    location = representation_pair_comparisons(oof)
    localization = localization_comparisons(oof)

    outputs = {
        AUDIT_ROOT / "metrics" / "oof_metrics.csv": oof,
        AUDIT_ROOT / "metrics" / "fold_metrics.csv": fold_metrics,
        AUDIT_ROOT / "metrics" / "hyperparameter_selections.csv": selections,
        AUDIT_ROOT / "metrics" / "coverage.csv": coverage,
        AUDIT_ROOT / "metrics" / "target_transform_fits.csv": pd.DataFrame(transform_rows),
        AUDIT_ROOT / "metrics" / "residualizer_fits.csv": pd.DataFrame(residualizer_rows),
        AUDIT_ROOT / "metrics" / "static_target_matrix.csv": static_matrix,
        AUDIT_ROOT / "metrics" / "residual_target_matrix.csv": residual_matrix,
        AUDIT_ROOT / "metrics" / "dynamic_target_matrix.csv": dynamic_matrix,
        AUDIT_ROOT / "metrics" / "representation_location_comparison.csv": location,
        AUDIT_ROOT / "metrics" / "oracle_localization_comparison.csv": localization,
        AUDIT_ROOT / "manifests" / "frozen_cell_manifest.csv": pd.DataFrame(cell_rows),
        AUDIT_ROOT / "manifests" / "target_contract.csv": target_contract(config),
        AUDIT_ROOT / "manifests" / "representation_contract.csv": representation_contract(config),
    }
    for path, frame in outputs.items():
        atomic_csv(path, frame)

    source_manifest = {
        "schema_version": 1,
        "status": "COMPLETE",
        "branch": branch,
        "parent_sha": parent,
        "source_commits": config["source_commits"],
        "patient_count": len(targets.patient_ids),
        "patient_set_sha256": targets.patient_set_sha256,
        "workbook_target_parity_max_abs_difference": targets.workbook_max_abs_difference,
        "fold_count": len(fold_splits),
        "feature_cell_count": len(cell_rows),
        "source_hashes": {
            "workbook": file_sha256(resolve_path(config["paths"]["workbook"])),
            "fold_manifest": file_sha256(resolve_path(config["paths"]["fold_manifest"])),
            "eligible_target_table": file_sha256(resolve_path(config["paths"]["eligible_target_table"])),
            "spatial_completion": file_sha256(resolve_path(config["paths"]["spatial_feature_root"] + "/feature_matrix_complete.private.json")),
            "goal6_final_report": file_sha256(resolve_path(config["paths"]["goal6_root"] + "/reports/final_report.md")),
        },
        "privacy": {
            "pcr_column_parsed": False,
            "pcr_used_for_target_representation_residualizer_timing_or_alpha_selection": False,
            "patient_identifiers_in_public_outputs": False,
            "private_prediction_path": "predictions/oof_predictions.private.csv.gz",
        },
        "preregistration_lock_sha256": lock_evidence["lock_sha256"],
    }
    source_manifest_path = AUDIT_ROOT / "manifests" / "input_provenance.json"
    atomic_json(source_manifest_path, source_manifest)

    core_output_hashes = {
        str(path.relative_to(AUDIT_ROOT)): file_sha256(path)
        for path in [*outputs, source_manifest_path]
    }
    run_summary = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "status": "COMPLETE",
        "branch": branch,
        "parent_sha": parent,
        "elapsed_seconds": time.time() - start_time,
        "patient_count": len(targets.patient_ids),
        "seeds": config["frozen"]["seeds"],
        "arms": config["frozen"]["arms"],
        "folds": config["frozen"]["folds"],
        "feature_cells": len(cell_rows),
        "oof_metric_rows": len(oof),
        "fold_metric_rows": len(fold_metrics),
        "private_prediction_rows": writer.rows,
        "expected_private_prediction_rows_from_oof_n_sum": expected_private_prediction_rows,
        "private_prediction_sha256": file_sha256(prediction_path),
        "private_prediction_committed": False,
        "encoder_retrained": False,
        "pcr_read": False,
        "pcr_used_for_selection": False,
        "test_used_for_alpha_selection": False,
        "test_predict_calls_per_probe": 1,
        "preregistration_lock_sha256": lock_evidence["lock_sha256"],
        "core_output_sha256": core_output_hashes,
    }
    atomic_json(run_summary_path, run_summary)
    print(json.dumps(run_summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

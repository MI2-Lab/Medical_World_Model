#!/usr/bin/env bash
# Post-freeze semantic erratum for the observed delta-FTV R2 gain mapping.
# Frozen Python sources remain byte-identical; invalid public artifacts are
# archived under ignored logs before replacement, with rollback on failure.
set -euo pipefail

script_path=$(cd -- "$(dirname -- "$0")" && pwd)/$(basename -- "$0")
audit_root=$(cd -- "$(dirname -- "$0")/.." && pwd)

python - "$audit_root" "$script_path" "$@" <<'PY'
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


root = Path(sys.argv[1]).resolve()
wrapper_path = Path(sys.argv[2]).resolve()
parser = argparse.ArgumentParser()
mode = parser.add_mutually_exclusive_group()
mode.add_argument("--preflight", action="store_true")
mode.add_argument("--validate-applied", action="store_true")
args = parser.parse_args(sys.argv[3:])
sys.path.insert(0, str(root / "src"))

import gjca.analysis as analysis  # noqa: E402
from gjca.aggregation import write_aggregate_tables  # noqa: E402
from gjca.contracts import (  # noqa: E402
    SEED_BASES,
    FOLDS,
    SOURCE_ROOT,
    atomic_csv,
    atomic_json,
    canonical_json_sha256,
    ensure_no_patient_columns,
    file_sha256,
)
from gjca.delivery import make_deliverables, validate_acceptance  # noqa: E402
from gjca.freeze import assert_plan_freeze  # noqa: E402
from gjca.gradients import validate_gradient_file  # noqa: E402
from gjca.phase_a import (  # noqa: E402
    load_resampling_bundle,
    write_phase_a_correlations,
)
from gjca.source_contract import (  # noqa: E402
    assert_source_contract,
    write_source_contract,
)
from gjca.statistics import (  # noqa: E402
    exact_crossed_spearman_permutation,
    holm_adjust,
    spearman_crossed_ci,
)


AMENDMENT_PATH = root / "POST_FREEZE_SEMANTIC_ERRATUM.json"
APPLIED_PATH = root / "POST_FREEZE_SEMANTIC_ERRATUM_APPLIED.json"
ARCHIVE_ROOT = root / "logs" / "delta_r2_semantic_erratum_invalid_v1"
PLAN_SHA256 = "a8c3dc736b9f31d3b0d9f4efeaf5efa1605388b19a0604087e15dc9987287cf1"
FROZEN_ANALYSIS_SHA256 = (
    "4e654f169b5fc281c070f6193d51ceffa704c8b6aa8c5439cb6c173cd35d47ff"
)
UPSTREAM_PROBE_SHA256 = (
    "4fe0afb68ec28dc57ae30063516ec1ad9fdaf362e241e722b943ca9df15efa1b"
)
CORRECT_VECTOR_SHA256 = (
    "7b310c7b41ed4547bd424d80f9bf8057a06fc0369f0e27dd45043988575cf0ea"
)
OLD_SOURCE_CONTRACT_SHA256 = (
    "5462d8e6196d423f24aacd3d981e7bac31b8f0ec0584702834d972ff626b66f3"
)

AGGREGATE_FILES = (
    "metrics/layer_level_conflict_metrics.csv",
    "metrics/run_level_conflict_metrics.csv",
    "metrics/trajectory_conflict_metrics.csv",
    "metrics/trajectory_change_metrics.csv",
    "metrics/ftv_coverage_metrics.csv",
)
ANALYSIS_FILES = (
    "metrics/analysis_input_manifest.csv",
    "metrics/pass_fail_comparison.csv",
    "metrics/gradient_correlation_metrics.csv",
    "metrics/dynamics_correlations.csv",
    "metrics/coverage_correlations.csv",
    "metrics/layer_localization_metrics.csv",
    "metrics/fold_signature_metrics.csv",
    "metrics/hypothesis_decision.csv",
    "metrics/diagnosis.json",
    "metrics/final_analysis_manifest.json",
    "metrics/FINAL_ANALYSIS_COMPLETE.json",
)
FIGURE_FILES = tuple(
    f"figures/{name}"
    for name in (
        "01_base_degradation_vs_static_ftv_gain.png",
        "02_base_degradation_vs_observed_delta_ftv_gain.png",
        "03_base_degradation_vs_all_shared_cosine.png",
        "04_base_degradation_vs_mbase.png",
        "05_pass_fail_cosine.png",
        "06_pass_fail_mbase.png",
        "07_pass_fail_norm_ratio.png",
        "08_layerwise_cosine_heatmap.png",
        "09_layerwise_mbase_heatmap.png",
        "10_seed_fold_conflict_heatmap.png",
        "11_representative_loss_trajectories.png",
        "12_conflict_vs_selected_epoch.png",
        "13_ftv_coverage_exposure_vs_degradation.png",
        "14_selected_to_last_conflict_change.png",
    )
)
PUBLIC_REPLACED_FILES = (
    "SOURCE_CONTRACT.json",
    "metrics/run_level_existing_metrics.csv",
    "metrics/phase_a_gain_correlations.csv",
    "metrics/batch_gradient_metrics.csv",
    "metrics/trajectory_batch_gradient_metrics.csv",
    *AGGREGATE_FILES,
    *ANALYSIS_FILES,
    "metrics/figure_manifest.csv",
    "reports/final_report.md",
    *FIGURE_FILES,
)


def _strict_false(series: pd.Series, role: str) -> None:
    values = set(series.astype(str).str.strip().str.lower())
    if values != {"false"}:
        raise ValueError(f"{role} privacy flag 非全 false: {values}")


def _atomic_text(path: Path, text: str, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_amendment() -> dict[str, Any]:
    if not AMENDMENT_PATH.is_file():
        raise FileNotFoundError("缺 POST_FREEZE_SEMANTIC_ERRATUM.json")
    amendment = json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version",
        "kind",
        "created_at_utc",
        "original_plan_freeze_sha256",
        "frozen_core_analysis_sha256",
        "first_provenance_erratum_sha256",
        "invalid_artifact_sha256",
        "buggy_source_column",
        "buggy_source_semantics",
        "correct_source_path",
        "correct_source_sha256",
        "correct_derivation",
        "correct_vector_sha256",
        "expected_corrected_phase_a",
        "formal_analysis_bundle_published_before_erratum",
        "gradient_numeric_values_observed_before_erratum",
        "input_values_modified",
        "formal_acceptance_written_before_erratum",
        "frozen_core_files_modified",
        "statistical_functions_modified",
        "threshold_logic_modified",
        "gradient_forward_backward_recomputed",
        "gradient_numeric_values_modified",
        "gradient_raw_combined_exact_before_erratum",
        "gradient_raw_combined_max_ulp_before_erratum",
        "gradient_raw_combined_nonfloat_and_rank_exact_before_erratum",
        "hypothesis_logic_modified",
        "expected_selected_hypothesis",
        "expected_first_recommendation",
        "expected_second_recommendation",
        "wrapper_path",
        "wrapper_sha256",
        "payload_sha256",
    }
    if set(amendment) != expected_keys:
        raise ValueError("semantic erratum exact key set 漂移")
    unsigned = dict(amendment)
    payload_sha = str(unsigned.pop("payload_sha256", ""))
    if (
        int(amendment["schema_version"]) != 1
        or amendment["kind"] != "postfreeze_delta_ftv_r2_semantic_erratum"
        or amendment["original_plan_freeze_sha256"] != PLAN_SHA256
        or amendment["frozen_core_analysis_sha256"] != FROZEN_ANALYSIS_SHA256
        or amendment["buggy_source_column"] != "seed_fold_effects.csv:D"
        or amendment["buggy_source_semantics"] != "base_degradation_fraction"
        or amendment["correct_source_path"]
        != "additional_experiments/g3_multiseed_generalization/metrics/final/probe_seed_fold_cell_metrics.csv"
        or amendment["correct_source_sha256"] != UPSTREAM_PROBE_SHA256
        or amendment["correct_derivation"]
        != "per seed×fold mean(change-cell G3 r2) - mean(change-cell G1 r2)"
        or amendment["correct_vector_sha256"] != CORRECT_VECTOR_SHA256
        or amendment["formal_analysis_bundle_published_before_erratum"] is not True
        or amendment["gradient_numeric_values_observed_before_erratum"] is not True
        or amendment["input_values_modified"] is not True
        or amendment["formal_acceptance_written_before_erratum"] is not False
        or amendment["frozen_core_files_modified"] is not False
        or amendment["statistical_functions_modified"] is not False
        or amendment["threshold_logic_modified"] is not False
        or amendment["gradient_forward_backward_recomputed"] is not False
        or amendment["gradient_numeric_values_modified"] is not False
        or amendment["gradient_raw_combined_exact_before_erratum"] is not False
        or int(amendment["gradient_raw_combined_max_ulp_before_erratum"]) != 2
        or amendment[
            "gradient_raw_combined_nonfloat_and_rank_exact_before_erratum"
        ]
        is not True
        or amendment["hypothesis_logic_modified"] is not False
        or amendment["expected_selected_hypothesis"] != "H4"
        or amendment["expected_first_recommendation"]
        != "fixed batch order/composition stochastic replicate"
        or amendment["expected_second_recommendation"] != "checkpoint averaging"
        or amendment["wrapper_path"]
        != "scripts/apply_delta_r2_semantic_erratum.sh"
        or amendment["wrapper_sha256"] != file_sha256(wrapper_path)
        or canonical_json_sha256(unsigned) != payload_sha
    ):
        raise ValueError("semantic erratum payload/SHA 漂移")
    return amendment


def _corrected_existing() -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    existing_path = root / "metrics" / "run_level_existing_metrics.csv"
    original = pd.read_csv(existing_path)
    if (
        tuple(original.columns) != tuple(analysis.EXISTING_METRIC_COLUMNS)
        or len(original) != 25
        or original.duplicated(["seed_base", "fold"]).any()
        or set(
            original[["seed_base", "fold"]].itertuples(index=False, name=None)
        )
        != {(seed, fold) for seed in SEED_BASES for fold in FOLDS}
    ):
        raise ValueError("invalid existing table 不是冻结25格精确schema")
    ensure_no_patient_columns(original.columns)
    mislabeled = original["delta_ftv_delta_r2"].to_numpy(dtype=np.float64)
    degradation = original["base_degradation"].to_numpy(dtype=np.float64)
    maximum_roundtrip_error = float(np.max(np.abs(mislabeled - degradation)))
    mislabeled_ranks = pd.Series(mislabeled).rank(method="average").to_numpy()
    degradation_ranks = pd.Series(degradation).rank(method="average").to_numpy()
    if (
        not np.isfinite(maximum_roundtrip_error)
        or maximum_roundtrip_error > 3e-15
        or not np.array_equal(mislabeled_ranks, degradation_ranks)
    ):
        raise ValueError(
            "待纠正列与base_degradation非仅CSV浮点回转误差，"
            "或Spearman秩不再完全相同；拒绝泛化修补"
        )

    source_path = (
        SOURCE_ROOT / "metrics" / "final" / "probe_seed_fold_cell_metrics.csv"
    )
    if file_sha256(source_path) != UPSTREAM_PROBE_SHA256:
        raise ValueError("正确R2上游源SHA漂移")
    probe = pd.read_csv(source_path)
    expected_probe_columns = (
        "seed_base",
        "fold",
        "model",
        "task",
        "cell",
        "n",
        "n_patients",
        "n_folds",
        "spearman",
        "pearson",
        "r2",
        "mae",
        "rmse",
        "b0_rmse",
        "rmse_gain_over_b0",
        "target_variance",
        "prediction_variance",
        "prediction_target_variance_ratio",
    )
    if tuple(probe.columns) != expected_probe_columns or len(probe) != 350:
        raise ValueError("正确R2上游源schema/rows漂移")
    change = probe.loc[probe["task"].eq("change")].copy()
    if (
        len(change) != 150
        or change.duplicated(["seed_base", "fold", "model", "cell"]).any()
        or set(change["model"].astype(str)) != {"G1", "G3"}
        or set(change["cell"].astype(str)) != {"T0→T1", "T1→T2", "T2→T3"}
        or not np.isfinite(change["r2"].to_numpy(dtype=np.float64)).all()
    ):
        raise ValueError("正确R2 change-cell grid漂移")
    grouped = (
        change.groupby(["seed_base", "fold", "model"], sort=True)
        .agg(r2=("r2", "mean"), cell_n=("r2", "size"))
        .reset_index()
    )
    if len(grouped) != 50 or set(grouped["cell_n"].astype(int)) != {3}:
        raise ValueError("正确R2每run/model必须恰有3个change cells")
    pivot = grouped.pivot(
        index=["seed_base", "fold"], columns="model", values="r2"
    ).sort_index()
    if pivot.shape != (25, 2) or list(pivot.columns) != ["G1", "G3"]:
        raise ValueError("正确R2 25×2 pivot漂移")
    mapping = {
        (int(seed), int(fold)): float(row["G3"] - row["G1"])
        for (seed, fold), row in pivot.iterrows()
    }
    records = [
        {
            "seed_base": seed,
            "fold": fold,
            "delta_ftv_delta_r2": mapping[(seed, fold)],
        }
        for seed in SEED_BASES
        for fold in FOLDS
    ]
    if canonical_json_sha256(records) != CORRECT_VECTOR_SHA256:
        raise ValueError("正确R2 25格向量SHA漂移")
    corrected = original.copy()
    corrected["delta_ftv_delta_r2"] = [
        mapping[(int(seed), int(fold))]
        for seed, fold in corrected[["seed_base", "fold"]].itertuples(
            index=False, name=None
        )
    ]
    unchanged_columns = [
        column for column in original.columns if column != "delta_ftv_delta_r2"
    ]
    pd.testing.assert_frame_equal(
        original[unchanged_columns],
        corrected[unchanged_columns],
        check_exact=True,
    )
    return original, corrected, records


def _phase_a_expected(
    corrected: pd.DataFrame,
) -> dict[str, dict[str, float | int]]:
    indices, _ = load_resampling_bundle()

    def grid(column: str) -> np.ndarray:
        return (
            corrected.pivot(index="seed_base", columns="fold", values=column)
            .reindex(index=SEED_BASES, columns=FOLDS)
            .to_numpy(dtype=np.float64)
        )

    x = grid("base_degradation")
    endpoints = {
        "degradation_vs_static_delta_spearman": "static_ftv_delta_spearman",
        "degradation_vs_observed_delta_spearman": "delta_ftv_delta_spearman",
        "degradation_vs_observed_delta_r2": "delta_ftv_delta_r2",
    }
    calculated: dict[str, tuple[Any, Any]] = {}
    for endpoint, column in endpoints.items():
        interval = spearman_crossed_ci(
            x,
            grid(column),
            indices.crossed_seed_draws,
            indices.crossed_fold_draws,
            confidence_level=0.95,
            minimum_finite_fraction=0.95,
        )
        permutation = exact_crossed_spearman_permutation(
            x,
            grid(column),
            indices.crossed_permutation_seed_orders,
            indices.crossed_permutation_fold_orders,
        )
        calculated[endpoint] = (interval, permutation)
    adjusted = holm_adjust(
        {endpoint: value[1].p_value for endpoint, value in calculated.items()}
    )
    return {
        endpoint: {
            "spearman_rho": float(interval.estimate),
            "ci_low": float(interval.ci_low),
            "ci_high": float(interval.ci_high),
            "p_raw_two_sided": float(permutation.p_value),
            "permutation_extreme_count": int(permutation.extreme_count),
            "p_holm": float(adjusted[endpoint].p_holm),
            "holm_rank": int(adjusted[endpoint].rank),
        }
        for endpoint, (interval, permutation) in calculated.items()
    }


def _assert_expected_phase(
    observed: Mapping[str, Mapping[str, Any]],
    expected: Mapping[str, Mapping[str, Any]],
) -> None:
    if set(observed) != set(expected):
        raise ValueError("corrected Phase A endpoint set漂移")
    for endpoint in expected:
        if set(observed[endpoint]) != set(expected[endpoint]):
            raise ValueError(f"corrected Phase A field set漂移: {endpoint}")
        for key, target in expected[endpoint].items():
            value = observed[endpoint][key]
            if isinstance(target, int):
                if int(value) != target:
                    raise ValueError(f"corrected Phase A integer漂移: {endpoint}/{key}")
            elif not math.isclose(
                float(value), float(target), rel_tol=0.0, abs_tol=1e-15
            ):
                raise ValueError(f"corrected Phase A value漂移: {endpoint}/{key}")


def _raw_gradient_paths() -> list[Path]:
    selected = sorted((root / "metrics" / "raw").glob("gradient_*_selected.csv"))
    last = sorted((root / "metrics" / "raw").glob("gradient_*_last.csv"))
    if len(selected) != 25 or len(last) != 6:
        raise ValueError(
            f"raw gradient exact set漂移: selected={len(selected)}, last={len(last)}"
        )
    return [*selected, *last]


def _preflight() -> tuple[
    dict[str, Any],
    pd.DataFrame,
    list[str],
    dict[str, dict[str, Any]],
]:
    amendment = _load_amendment()
    freeze = assert_plan_freeze()
    if (
        file_sha256(root / "PLAN_FREEZE.json") != PLAN_SHA256
        or freeze["core_protocol_sha256"]["src/gjca/analysis.py"]
        != FROZEN_ANALYSIS_SHA256
        or file_sha256(root / "src" / "gjca" / "analysis.py")
        != FROZEN_ANALYSIS_SHA256
    ):
        raise ValueError("semantic erratum plan/core SHA漂移")
    first_erratum = root / "POST_FREEZE_ERRATUM.json"
    if file_sha256(first_erratum) != amendment["first_provenance_erratum_sha256"]:
        raise ValueError("first provenance erratum SHA漂移")
    if file_sha256(root / "SOURCE_CONTRACT.json") != OLD_SOURCE_CONTRACT_SHA256:
        raise ValueError("invalid source contract SHA不符合唯一前提")
    assert_source_contract(full_content_hash=False, full_checkpoint_hash=False)
    invalid_hashes = amendment["invalid_artifact_sha256"]
    if (
        not isinstance(invalid_hashes, dict)
        or set(invalid_hashes) != set(PUBLIC_REPLACED_FILES)
        or len(invalid_hashes) != 37
    ):
        raise ValueError("invalid artifact hash map不是精确37-file public set")
    for relative, expected_sha in invalid_hashes.items():
        path = root / relative
        if not path.is_file() or path.is_symlink() or file_sha256(path) != expected_sha:
            raise ValueError(f"invalid artifact precondition漂移: {relative}")
    if APPLIED_PATH.exists() or ARCHIVE_ROOT.exists():
        raise FileExistsError("semantic erratum已应用或archive已存在")
    for relative in (
        "reports/acceptance.json",
        "metrics/acceptance_check.json",
    ):
        if (root / relative).exists():
            raise FileExistsError("正式acceptance已写；拒绝事后语义勘误")
    original, corrected, _ = _corrected_existing()
    expected_phase = amendment["expected_corrected_phase_a"]
    calculated_phase = _phase_a_expected(corrected)
    _assert_expected_phase(calculated_phase, expected_phase)
    diagnosis = json.loads((root / "metrics" / "diagnosis.json").read_text())
    if (
        diagnosis.get("selected_hypothesis") != "H4"
        or diagnosis.get("fix_executed") is not False
    ):
        raise ValueError("invalid bundle diagnosis前提漂移")
    raw = _raw_gradient_paths()
    pre_erratum_gradient_closure = _validate_gradient_closure(raw)
    raw_relatives = [path.relative_to(root).as_posix() for path in raw]
    all_targets = [*PUBLIC_REPLACED_FILES, *raw_relatives]
    if len(all_targets) != len(set(all_targets)):
        raise AssertionError("semantic erratum archive target重复")
    for relative in all_targets:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"semantic erratum archive target缺失: {relative}")
    if len(list((root / "figures").glob("*.png"))) != 14:
        raise ValueError("invalid figure exact count不是14")
    if len(original) != 25:
        raise AssertionError("invalid existing row count漂移")
    return (
        amendment,
        corrected,
        sorted(all_targets),
        pre_erratum_gradient_closure,
    )


def _archive(targets: Sequence[str]) -> tuple[list[dict[str, Any]], str]:
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    for relative in targets:
        source = root / relative
        destination = ARCHIVE_ROOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        rows.append(
            {
                "schema_version": 1,
                "path": relative,
                "sha256": file_sha256(source),
                "bytes": source.stat().st_size,
            }
        )
    manifest = ARCHIVE_ROOT / "archive_manifest.csv"
    atomic_csv(
        manifest,
        rows,
        fieldnames=("schema_version", "path", "sha256", "bytes"),
    )
    return rows, file_sha256(manifest)


def _restore(rows: Sequence[Mapping[str, Any]]) -> Path:
    for item in rows:
        relative = str(item["path"])
        archived = ARCHIVE_ROOT / relative
        destination = root / relative
        if (
            not archived.is_file()
            or file_sha256(archived) != str(item["sha256"])
            or archived.stat().st_size != int(item["bytes"])
        ):
            raise RuntimeError(f"rollback archive损坏: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(archived, destination)
    APPLIED_PATH.unlink(missing_ok=True)
    for item in rows:
        destination = root / str(item["path"])
        if (
            not destination.is_file()
            or file_sha256(destination) != str(item["sha256"])
            or destination.stat().st_size != int(item["bytes"])
        ):
            raise RuntimeError(f"rollback后文件未精确恢复: {item['path']}")
    assert_plan_freeze()
    assert_source_contract(full_content_hash=False, full_checkpoint_hash=False)
    analysis.validate_final_analysis_bundle(root)
    restored_acceptance = validate_acceptance(root, write=False)
    if restored_acceptance["status"] != "PASS":
        raise RuntimeError("rollback后check-only acceptance未PASS")
    failed_at = dt.datetime.now(dt.timezone.utc)
    failure_record = {
        "schema_version": 1,
        "status": "failed_application_rolled_back_and_validated",
        "created_at_utc": failed_at.isoformat(),
        "restored_file_count": len(rows),
        "archive_manifest_sha256": file_sha256(
            ARCHIVE_ROOT / "archive_manifest.csv"
        ),
        "source_contract_sha256": file_sha256(root / "SOURCE_CONTRACT.json"),
        "final_analysis_manifest_sha256": file_sha256(
            root / "metrics" / "final_analysis_manifest.json"
        ),
        "check_only_acceptance": "PASS",
    }
    failure_record["payload_sha256"] = canonical_json_sha256(failure_record)
    atomic_json(ARCHIVE_ROOT / "FAILED_ROLLBACK_VERIFIED.json", failure_record)
    failure_destination = ARCHIVE_ROOT.with_name(
        f"{ARCHIVE_ROOT.name}_failed_{failed_at.strftime('%Y%m%dT%H%M%S%fZ')}"
    )
    os.replace(ARCHIVE_ROOT, failure_destination)
    return failure_destination


def _atomic_bytes(path: Path, payload: bytes, mode: int) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, stat.S_IMODE(mode))
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_corrected_existing_exact(
    path: Path, corrected: pd.DataFrame
) -> dict[str, Any]:
    before = path.read_bytes()
    before_sha = file_sha256(path)
    before_mode = path.stat().st_mode
    before_frame = pd.read_csv(path)
    lines = before.splitlines(keepends=True)
    if lines and all(line.endswith(b"\r\n") for line in lines):
        newline = b"\r\n"
        newline_name = "CRLF"
    elif lines and all(
        line.endswith(b"\n") and not line.endswith(b"\r\n") for line in lines
    ):
        newline = b"\n"
        newline_name = "LF"
    else:
        raise ValueError("existing CSV非统一LF/CRLF换行")
    if (
        len(lines) != 26
        or b'"' in before
    ):
        raise ValueError("existing CSV非预期无引号25-row格式")
    header = lines[0][: -len(newline)].decode("utf-8").split(",")
    if tuple(header) != tuple(analysis.EXISTING_METRIC_COLUMNS):
        raise ValueError("existing CSV byte header漂移")
    target_index = header.index("delta_ftv_delta_r2")
    seed_index = header.index("seed_base")
    fold_index = header.index("fold")
    mapping = {
        (int(row.seed_base), int(row.fold)): float(row.delta_ftv_delta_r2)
        for row in corrected.itertuples(index=False)
    }
    rewritten = [lines[0]]
    token_replacements: list[dict[str, Any]] = []
    for line in lines[1:]:
        fields = line[: -len(newline)].decode("utf-8").split(",")
        if len(fields) != len(header):
            raise ValueError("existing CSV byte row字段数漂移")
        key = (int(fields[seed_index]), int(fields[fold_index]))
        if key not in mapping:
            raise ValueError(f"existing CSV byte row key漂移: {key}")
        old_target = fields[target_index]
        new_target = repr(mapping[key])
        fields[target_index] = new_target
        rewritten.append(",".join(fields).encode("utf-8") + newline)
        token_replacements.append(
            {
                "seed_base": key[0],
                "fold": key[1],
                "old_token": old_target,
                "new_token": new_target,
            }
        )
    after = b"".join(rewritten)
    for old_line, new_line in zip(lines[1:], rewritten[1:], strict=True):
        old_fields = old_line[: -len(newline)].split(b",")
        new_fields = new_line[: -len(newline)].split(b",")
        if any(
            old_fields[index] != new_fields[index]
            for index in range(len(header))
            if index != target_index
        ):
            raise AssertionError("existing CSV非目标token被修改")
    _atomic_bytes(path, after, before_mode)
    observed = pd.read_csv(path)
    observed_roundtrip = pd.read_csv(path, float_precision="round_trip")
    unchanged = [
        column for column in before_frame.columns if column != "delta_ftv_delta_r2"
    ]
    pd.testing.assert_frame_equal(
        observed[unchanged], before_frame[unchanged], check_exact=True
    )
    pd.testing.assert_series_equal(
        observed_roundtrip["delta_ftv_delta_r2"],
        corrected["delta_ftv_delta_r2"],
        check_exact=True,
    )
    if (
        float(
            np.max(
                np.abs(
                    observed["delta_ftv_delta_r2"].to_numpy(dtype=np.float64)
                    - corrected["delta_ftv_delta_r2"].to_numpy(dtype=np.float64)
                )
            )
        )
        > 1e-15
        or not np.array_equal(
            observed["delta_ftv_delta_r2"].rank(method="average").to_numpy(),
            corrected["delta_ftv_delta_r2"].rank(method="average").to_numpy(),
        )
    ):
        raise AssertionError("existing corrected target CSV读取语义漂移")
    return {
        "path": path.relative_to(root).as_posix(),
        "before_sha256": before_sha,
        "after_sha256": file_sha256(path),
        "row_count": 25,
        "target_column": "delta_ftv_delta_r2",
        "target_token_replacement_count": len(token_replacements),
        "non_target_token_count_preserved": 25 * (len(header) - 1),
        "non_target_tokens_exact": True,
        "target_decimal_tokens_roundtrip_exact": True,
        "line_endings_preserved": newline_name,
        "permissions_preserved": True,
        "target_token_replacements_sha256": canonical_json_sha256(
            token_replacements
        ),
    }


def _replace_gradient_source_sha_bytes(
    path: Path, new_sha: str, expected_replacements: int
) -> dict[str, Any]:
    old_token = OLD_SOURCE_CONTRACT_SHA256.encode("ascii")
    new_token = new_sha.encode("ascii")
    if len(old_token) != 64 or len(new_token) != 64 or old_token == new_token:
        raise ValueError("gradient provenance SHA token非合法64-byte等长替换")
    before = path.read_bytes()
    before_sha = file_sha256(path)
    before_size = len(before)
    before_mode = path.stat().st_mode
    replacements = before.count(old_token)
    if replacements != expected_replacements or before.count(new_token) != 0:
        raise ValueError(
            f"gradient provenance token次数漂移: {path.name}="
            f"{replacements} != {expected_replacements}"
        )
    expected_after = before.replace(old_token, new_token)
    if len(expected_after) != before_size:
        raise AssertionError("gradient provenance替换改变文件字节数")
    _atomic_bytes(path, expected_after, before_mode)
    after = path.read_bytes()
    if (
        after != expected_after
        or len(after) != before_size
        or stat.S_IMODE(path.stat().st_mode) != stat.S_IMODE(before_mode)
        or after.count(old_token) != 0
        or after.count(new_token) != expected_replacements
        or after.replace(new_token, old_token) != before
    ):
        raise AssertionError(f"gradient provenance非纯字节替换: {path.name}")
    return {
        "path": path.relative_to(root).as_posix(),
        "before_sha256": before_sha,
        "after_sha256": file_sha256(path),
        "bytes_before": before_size,
        "bytes_after": len(after),
        "replacement_count": replacements,
        "old_token_absent_after": True,
        "new_token_count_after": expected_replacements,
        "all_other_bytes_exact": True,
        "permissions_preserved": True,
    }


def _gradient_identity(path: Path) -> tuple[int, int, str]:
    match = re.fullmatch(
        r"gradient_seed_(\d+)_fold_(\d+)_(selected|last)\.csv", path.name
    )
    if match is None:
        raise ValueError(f"raw gradient filename非法: {path.name}")
    return int(match.group(1)), int(match.group(2)), str(match.group(3))


def _float_ulp_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_bits = np.asarray(left, dtype=np.float64).view(np.uint64)
    right_bits = np.asarray(right, dtype=np.float64).view(np.uint64)
    sign = np.uint64(1 << 63)
    left_ordered = np.where(
        (left_bits & sign) != 0, ~left_bits, left_bits | sign
    )
    right_ordered = np.where(
        (right_bits & sign) != 0, ~right_bits, right_bits | sign
    )
    return np.maximum(left_ordered, right_ordered) - np.minimum(
        left_ordered, right_ordered
    )


def _validate_gradient_closure(
    raw_paths: Sequence[Path],
) -> dict[str, dict[str, Any]]:
    frames: dict[str, list[pd.DataFrame]] = {"selected": [], "last": []}
    for path in raw_paths:
        seed, fold, kind = _gradient_identity(path)
        frames[kind].append(validate_gradient_file(path, seed, fold, kind))
    specifications = (
        ("selected", root / "metrics" / "batch_gradient_metrics.csv", 2800),
        (
            "last",
            root / "metrics" / "trajectory_batch_gradient_metrics.csv",
            672,
        ),
    )
    sort_columns = [
        "seed_base",
        "fold",
        "checkpoint_kind",
        "split",
        "batch_index",
        "group",
    ]
    evidence: dict[str, dict[str, Any]] = {}
    for kind, combined_path, expected_rows in specifications:
        reconstructed = (
            pd.concat(frames[kind], ignore_index=True)
            .sort_values(sort_columns)
            .reset_index(drop=True)
        )
        combined = pd.read_csv(combined_path).reset_index(drop=True)
        if (
            len(reconstructed) != expected_rows
            or tuple(reconstructed.columns) != tuple(combined.columns)
            or tuple(reconstructed.dtypes.astype(str))
            != tuple(combined.dtypes.astype(str))
        ):
            raise ValueError(f"{kind} raw gradient重建行数漂移")
        float_columns: dict[str, dict[str, int | bool]] = {}
        differing_float_cells = 0
        maximum_ulp = 0
        for column in combined.columns:
            left = combined[column]
            right = reconstructed[column]
            if pd.api.types.is_float_dtype(left.dtype):
                left_values = left.to_numpy(dtype=np.float64)
                right_values = right.to_numpy(dtype=np.float64)
                if (
                    not np.isfinite(left_values).all()
                    or not np.isfinite(right_values).all()
                ):
                    raise ValueError(f"{kind}/{column} raw-combined nonfinite")
                ulp = _float_ulp_distance(left_values, right_values)
                column_maximum = int(ulp.max(initial=np.uint64(0)))
                differing = int(np.count_nonzero(ulp))
                rank_exact = np.array_equal(
                    pd.Series(left_values).rank(method="average").to_numpy(),
                    pd.Series(right_values).rank(method="average").to_numpy(),
                )
                if column_maximum > 2 or not rank_exact:
                    raise ValueError(
                        f"{kind}/{column} raw-combined超出预存CSV解析差: "
                        f"max_ulp={column_maximum}, rank_exact={rank_exact}"
                    )
                differing_float_cells += differing
                maximum_ulp = max(maximum_ulp, column_maximum)
                if differing:
                    float_columns[column] = {
                        "differing_cells": differing,
                        "max_ulp": column_maximum,
                        "rank_exact": rank_exact,
                    }
            else:
                pd.testing.assert_series_equal(
                    left,
                    right,
                    check_exact=True,
                    check_dtype=True,
                    check_names=True,
                )
        relative = combined_path.relative_to(root).as_posix()
        evidence[relative] = {
            "rows": expected_rows,
            "columns": len(combined.columns),
            "key_and_nonfloat_fields_exact": True,
            "float_ranks_exact": True,
            "preexisting_csv_serialization_max_ulp": maximum_ulp,
            "preexisting_csv_serialization_differing_float_cells": (
                differing_float_cells
            ),
            "differing_float_columns": float_columns,
        }
    if max(
        int(item["preexisting_csv_serialization_max_ulp"])
        for item in evidence.values()
    ) != 2:
        raise ValueError("raw-combined预存CSV serialization max ULP非锁定2")
    return evidence


def _provenance_rows_with_erratum(path_root: Path) -> list[dict[str, Any]]:
    specs = (
        (
            "selected_batch_gradients",
            "metrics/batch_gradient_metrics.csv",
            analysis.BATCH_GRADIENT_COLUMNS,
            2800,
        ),
        (
            "trajectory_batch_gradients",
            "metrics/trajectory_batch_gradient_metrics.csv",
            analysis.BATCH_GRADIENT_COLUMNS,
            672,
        ),
        (
            "layer_aggregate",
            "metrics/layer_level_conflict_metrics.csv",
            analysis.LAYER_LEVEL_COLUMNS,
            350,
        ),
        (
            "run_aggregate",
            "metrics/run_level_conflict_metrics.csv",
            analysis.RUN_LEVEL_COLUMNS,
            50,
        ),
        (
            "trajectory_aggregate",
            "metrics/trajectory_conflict_metrics.csv",
            analysis.TRAJECTORY_COLUMNS,
            168,
        ),
        (
            "trajectory_change",
            "metrics/trajectory_change_metrics.csv",
            analysis.TRAJECTORY_CHANGE_COLUMNS,
            84,
        ),
        (
            "ftv_coverage",
            "metrics/ftv_coverage_metrics.csv",
            analysis.COVERAGE_COLUMNS,
            10,
        ),
        (
            "existing_run_metrics",
            "metrics/run_level_existing_metrics.csv",
            analysis.EXISTING_METRIC_COLUMNS,
            25,
        ),
        (
            "phase_a_correlations",
            "metrics/phase_a_gain_correlations.csv",
            analysis.PHASE_A_COLUMNS,
            3,
        ),
    )
    rows: list[dict[str, Any]] = []
    missing_flag_roles: list[str] = []
    for role, relative, columns, expected_rows in specs:
        path = path_root / relative
        frame = pd.read_csv(path)
        if tuple(frame.columns) != tuple(columns) or len(frame) != expected_rows:
            raise ValueError(f"semantic erratum analysis input漂移: {role}")
        ensure_no_patient_columns(frame.columns)
        if "contains_patient_ids" in frame.columns:
            _strict_false(frame["contains_patient_ids"], role)
        else:
            missing_flag_roles.append(role)
            if role != "existing_run_metrics":
                raise ValueError(f"非预期缺privacy flag: {role}")
            objects = frame.select_dtypes(
                include=["object", "string"]
            ).columns.tolist()
            if objects != ["selection_mode", "base_gate"]:
                raise ValueError("existing object schema漂移")
            if set(frame["selection_mode"].astype(str)) != {
                "primary",
                "fallback_base_gate_failed",
            } or set(frame["base_gate"].astype(str)) != {"PASS", "FAIL"}:
                raise ValueError("existing object values漂移")
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
        if tuple(row) != analysis.ANALYSIS_INPUT_MANIFEST_COLUMNS:
            raise AssertionError("semantic erratum input manifest row schema漂移")
        rows.append(row)
    if missing_flag_roles != ["existing_run_metrics"] or len(rows) != 9:
        raise AssertionError("semantic erratum privacy豁免范围漂移")
    return rows


def _correct_phase_table(
    expected: Mapping[str, Mapping[str, Any]],
) -> None:
    frame = pd.read_csv(root / "metrics" / "phase_a_gain_correlations.csv")
    observed = {
        str(row.endpoint): {
            "spearman_rho": float(row.spearman_rho),
            "ci_low": float(row.ci_low),
            "ci_high": float(row.ci_high),
            "p_raw_two_sided": float(row.p_raw_two_sided),
            "permutation_extreme_count": int(row.permutation_extreme_count),
            "p_holm": float(row.p_holm),
            "holm_rank": int(row.holm_rank),
        }
        for row in frame.itertuples(index=False)
    }
    _assert_expected_phase(observed, expected)


def _report_erratum_note() -> None:
    path = root / "reports" / "final_report.md"
    text = path.read_text(encoding="utf-8")
    anchor = "# Grounding–JEPA Conflict Audit 最终报告\n\n"
    if not text.startswith(anchor) or "冻结后语义勘误" in text:
        raise ValueError("final report erratum note anchor漂移")
    note = (
        "> **冻结后语义勘误：** 冻结实现曾把上游 D（base degradation）误作"
        " delta_ftv_delta_r2，产生rho=1的伪自相关。本版已从冻结的"
        " probe_seed_fold_cell_metrics.csv 按每个seed×fold的三个change cell"
        "重算R² gain；没有重算梯度或改变H1–H4规则。详见"
        "[POST_FREEZE_SEMANTIC_ERRATUM.json](../POST_FREEZE_SEMANTIC_ERRATUM.json)。\n\n"
    )
    _atomic_text(path, anchor + note + text[len(anchor) :], overwrite=True)


def _acceptance_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "status",
        "checks",
        "row_counts",
        "selected_hypothesis",
        "first_recommendation",
        "second_recommendation",
        "source_contract_sha256",
        "final_analysis_manifest_sha256",
        "final_analysis_marker_sha256",
        "diagnosis_sha256",
        "figure_manifest_sha256",
        "final_report_sha256",
        "non_table_file_sha256",
        "analysis_manifest_artifact_count",
        "analysis_commit",
        "contains_patient_ids",
    )
    return {key: payload[key] for key in keys}


def _validate_applied_marker() -> dict[str, Any]:
    amendment = _load_amendment()
    if not APPLIED_PATH.is_file() or APPLIED_PATH.is_symlink():
        raise FileNotFoundError("semantic erratum applied marker缺失")
    payload = json.loads(APPLIED_PATH.read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version",
        "status",
        "created_at_utc",
        "semantic_erratum_sha256",
        "wrapper_sha256",
        "archive_manifest_sha256",
        "archive_file_count",
        "old_artifact_sha256",
        "old_artifact_bytes",
        "new_artifact_sha256",
        "new_artifact_bytes",
        "plan_freeze_sha256",
        "frozen_core_sha256",
        "upstream_probe_path",
        "upstream_probe_sha256",
        "resampling_indices_sha256",
        "resampling_manifest_sha256",
        "old_source_contract_sha256",
        "new_source_contract_sha256",
        "corrected_existing_metrics_sha256",
        "corrected_existing_token_rewrite",
        "corrected_phase_a_sha256",
        "correct_vector_sha256",
        "gradient_byte_replacement_file_count",
        "gradient_byte_replacement_count",
        "gradient_byte_replacements",
        "gradient_raw_combined_exact_before_erratum",
        "gradient_raw_combined_exact_after_erratum",
        "gradient_raw_combined_closure_before_erratum",
        "gradient_raw_combined_closure_after_erratum",
        "gradient_raw_combined_key_nonfloat_exact",
        "gradient_raw_combined_float_ranks_exact",
        "gradient_raw_combined_max_ulp",
        "aggregate_rows",
        "final_analysis_manifest_sha256",
        "final_analysis_artifact_sha256",
        "figure_manifest_sha256",
        "final_report_sha256",
        "final_delivery_artifact_sha256",
        "old_diagnosis",
        "new_diagnosis",
        "selected_hypothesis",
        "first_recommendation",
        "second_recommendation",
        "formal_analysis_bundle_published_before_erratum",
        "gradient_numeric_values_observed_before_erratum",
        "input_values_modified",
        "formal_acceptance_written_before_erratum",
        "formal_acceptance_written_when_marker_created",
        "frozen_core_files_modified",
        "statistical_functions_modified",
        "threshold_logic_modified",
        "gradient_forward_backward_recomputed",
        "gradient_numeric_values_modified",
        "hypothesis_logic_modified",
        "check_only_acceptance_summary",
        "semantic_erratum_chain_validator_available",
        "contains_patient_ids",
        "payload_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError("semantic applied marker exact key set漂移")
    unsigned = dict(payload)
    digest = str(unsigned.pop("payload_sha256", ""))
    if (
        int(payload["schema_version"]) != 1
        or payload["status"]
        != "applied_and_check_only_acceptance_passed"
        or canonical_json_sha256(unsigned) != digest
        or payload["semantic_erratum_sha256"] != file_sha256(AMENDMENT_PATH)
        or payload["wrapper_sha256"] != file_sha256(wrapper_path)
        or payload["plan_freeze_sha256"] != PLAN_SHA256
        or payload["upstream_probe_sha256"] != UPSTREAM_PROBE_SHA256
        or payload["correct_vector_sha256"] != CORRECT_VECTOR_SHA256
        or payload["old_source_contract_sha256"]
        != OLD_SOURCE_CONTRACT_SHA256
        or payload["new_source_contract_sha256"]
        != file_sha256(root / "SOURCE_CONTRACT.json")
        or payload["corrected_existing_metrics_sha256"]
        != file_sha256(root / "metrics" / "run_level_existing_metrics.csv")
        or payload["corrected_phase_a_sha256"]
        != file_sha256(root / "metrics" / "phase_a_gain_correlations.csv")
        or payload["resampling_indices_sha256"]
        != file_sha256(root / "metrics" / "resampling_indices.npz")
        or payload["resampling_manifest_sha256"]
        != file_sha256(root / "metrics" / "resampling_manifest.json")
    ):
        raise ValueError("semantic applied marker payload/current source chain漂移")
    expected_flags = {
        "formal_analysis_bundle_published_before_erratum": True,
        "gradient_numeric_values_observed_before_erratum": True,
        "input_values_modified": True,
        "formal_acceptance_written_before_erratum": False,
        "formal_acceptance_written_when_marker_created": False,
        "frozen_core_files_modified": False,
        "statistical_functions_modified": False,
        "threshold_logic_modified": False,
        "gradient_forward_backward_recomputed": False,
        "gradient_numeric_values_modified": False,
        "hypothesis_logic_modified": False,
        "gradient_raw_combined_exact_before_erratum": False,
        "gradient_raw_combined_exact_after_erratum": False,
        "gradient_raw_combined_key_nonfloat_exact": True,
        "gradient_raw_combined_float_ranks_exact": True,
        "semantic_erratum_chain_validator_available": True,
        "contains_patient_ids": False,
    }
    if any(payload[key] is not value for key, value in expected_flags.items()):
        raise ValueError("semantic applied marker honesty/safety flags漂移")
    freeze = assert_plan_freeze()
    if payload["frozen_core_sha256"] != freeze["core_protocol_sha256"]:
        raise ValueError("semantic applied marker frozen core map漂移")
    if file_sha256(SOURCE_ROOT / "metrics/final/probe_seed_fold_cell_metrics.csv") != (
        UPSTREAM_PROBE_SHA256
    ):
        raise ValueError("semantic applied marker upstream probe漂移")
    raw_paths = _raw_gradient_paths()
    raw_relatives = [path.relative_to(root).as_posix() for path in raw_paths]
    targets = sorted([*PUBLIC_REPLACED_FILES, *raw_relatives])
    old_hashes = payload["old_artifact_sha256"]
    old_bytes = payload["old_artifact_bytes"]
    new_hashes = payload["new_artifact_sha256"]
    new_bytes = payload["new_artifact_bytes"]
    if not all(isinstance(item, dict) for item in (old_hashes, old_bytes, new_hashes, new_bytes)):
        raise ValueError("semantic applied marker old/new maps非object")
    if any(set(item) != set(targets) for item in (old_hashes, old_bytes, new_hashes, new_bytes)):
        raise ValueError("semantic applied marker old/new maps非exact 68-set")
    if {
        relative: old_hashes[relative] for relative in PUBLIC_REPLACED_FILES
    } != amendment["invalid_artifact_sha256"]:
        raise ValueError("semantic applied marker old public map与amendment不闭环")
    for relative in targets:
        current = root / relative
        if (
            not current.is_file()
            or file_sha256(current) != str(new_hashes[relative])
            or current.stat().st_size != int(new_bytes[relative])
        ):
            raise ValueError(f"semantic applied current artifact漂移: {relative}")
    if ARCHIVE_ROOT.is_dir():
        if file_sha256(ARCHIVE_ROOT / "archive_manifest.csv") != payload[
            "archive_manifest_sha256"
        ]:
            raise ValueError("semantic applied archive manifest漂移")
        for relative in targets:
            archived = ARCHIVE_ROOT / relative
            if (
                not archived.is_file()
                or file_sha256(archived) != str(old_hashes[relative])
                or archived.stat().st_size != int(old_bytes[relative])
            ):
                raise ValueError(f"semantic applied archive artifact漂移: {relative}")
    records = payload["gradient_byte_replacements"]
    gradient_relatives = {
        "metrics/batch_gradient_metrics.csv",
        "metrics/trajectory_batch_gradient_metrics.csv",
        *raw_relatives,
    }
    if (
        not isinstance(records, list)
        or len(records) != 33
        or int(payload["gradient_byte_replacement_file_count"]) != 33
        or int(payload["gradient_byte_replacement_count"]) != 6944
        or {str(record["path"]) for record in records} != gradient_relatives
        or sum(int(record["replacement_count"]) for record in records) != 6944
    ):
        raise ValueError("semantic applied gradient replacement ledger漂移")
    new_token = str(payload["new_source_contract_sha256"]).encode("ascii")
    old_token = OLD_SOURCE_CONTRACT_SHA256.encode("ascii")
    for record in records:
        relative = str(record["path"])
        content = (root / relative).read_bytes()
        if (
            str(record["before_sha256"]) != str(old_hashes[relative])
            or str(record["after_sha256"]) != str(new_hashes[relative])
            or int(record["bytes_before"]) != int(old_bytes[relative])
            or int(record["bytes_after"]) != int(new_bytes[relative])
            or int(record["bytes_before"]) != int(record["bytes_after"])
            or content.count(old_token) != 0
            or content.count(new_token) != int(record["replacement_count"])
            or record["all_other_bytes_exact"] is not True
        ):
            raise ValueError(f"semantic applied gradient byte ledger漂移: {relative}")
    observed_gradient_closure = _validate_gradient_closure(raw_paths)
    if (
        int(payload["gradient_raw_combined_max_ulp"]) != 2
        or payload["gradient_raw_combined_closure_before_erratum"]
        != payload["gradient_raw_combined_closure_after_erratum"]
        or payload["gradient_raw_combined_closure_after_erratum"]
        != observed_gradient_closure
    ):
        raise ValueError("semantic applied raw-combined closure ledger漂移")
    _correct_phase_table(amendment["expected_corrected_phase_a"])
    assert_source_contract(full_content_hash=False, full_checkpoint_hash=False)
    validated_analysis = analysis.validate_final_analysis_bundle(root)
    if validated_analysis["selected_hypothesis"] != "H4":
        raise ValueError("semantic applied final analysis非H4")
    for mapping_key in (
        "final_analysis_artifact_sha256",
        "final_delivery_artifact_sha256",
    ):
        for relative, expected_sha in payload[mapping_key].items():
            if file_sha256(root / relative) != str(expected_sha):
                raise ValueError(f"semantic applied final hash漂移: {relative}")
    diagnosis = json.loads(
        (root / "metrics" / "diagnosis.json").read_text(encoding="utf-8")
    )
    if payload["new_diagnosis"] != {
        key: diagnosis[key]
        for key in (
            "selected_hypothesis",
            "first_recommendation",
            "second_recommendation",
            "fix_executed",
        )
    } or payload["old_diagnosis"] != payload["new_diagnosis"]:
        raise ValueError("semantic applied diagnosis ledger漂移")
    acceptance = validate_acceptance(root, write=False)
    if _acceptance_summary(acceptance) != payload["check_only_acceptance_summary"]:
        raise ValueError("semantic applied check-only acceptance summary漂移")
    return {
        "status": "PASS",
        "mode": "validate-applied",
        "applied_marker_sha256": file_sha256(APPLIED_PATH),
        "archive_file_count": int(payload["archive_file_count"]),
        "gradient_byte_replacement_count": 6944,
        "selected_hypothesis": "H4",
        "check_only_acceptance": acceptance["status"],
    }


def _apply(
    amendment: Mapping[str, Any],
    corrected: pd.DataFrame,
    targets: Sequence[str],
    pre_erratum_gradient_closure: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    archive_rows, archive_manifest_sha = _archive(targets)
    try:
        existing_path = root / "metrics" / "run_level_existing_metrics.csv"
        existing_rewrite = _write_corrected_existing_exact(existing_path, corrected)

        phase_path = root / "metrics" / "phase_a_gain_correlations.csv"
        phase_path.unlink()
        write_phase_a_correlations()
        _correct_phase_table(amendment["expected_corrected_phase_a"])

        contract_path = root / "SOURCE_CONTRACT.json"
        contract_path.unlink()
        write_source_contract()
        new_source_sha = file_sha256(contract_path)
        assert_source_contract(full_content_hash=False, full_checkpoint_hash=False)

        raw_paths = _raw_gradient_paths()
        gradient_replacements: list[dict[str, Any]] = []
        gradient_replacements.append(
            _replace_gradient_source_sha_bytes(
                root / "metrics" / "batch_gradient_metrics.csv",
                new_source_sha,
                2800,
            )
        )
        gradient_replacements.append(
            _replace_gradient_source_sha_bytes(
                root / "metrics" / "trajectory_batch_gradient_metrics.csv",
                new_source_sha,
                672,
            )
        )
        gradient_replacements.extend(
            _replace_gradient_source_sha_bytes(path, new_source_sha, 112)
            for path in raw_paths
        )
        if (
            len(gradient_replacements) != 33
            or sum(
                int(record["replacement_count"])
                for record in gradient_replacements
            )
            != 6944
        ):
            raise AssertionError("gradient provenance精确替换总数漂移")
        post_erratum_gradient_closure = _validate_gradient_closure(raw_paths)
        if post_erratum_gradient_closure != pre_erratum_gradient_closure:
            raise AssertionError(
                "gradient raw-combined预存序列化差在勘误前后改变"
            )

        aggregate_rows = write_aggregate_tables(root, overwrite=True)

        for relative in ANALYSIS_FILES:
            (root / relative).unlink()
        analysis._analysis_input_rows = _provenance_rows_with_erratum
        analysis_result = analysis.write_full_analysis(root)
        validated_analysis = analysis.validate_final_analysis_bundle(root)
        if (
            analysis_result["selected_hypothesis"] != "H4"
            or validated_analysis["selected_hypothesis"] != "H4"
        ):
            raise ValueError("semantic correction意外改变H1-H4唯一判定")

        for relative in (*FIGURE_FILES, "metrics/figure_manifest.csv"):
            (root / relative).unlink()
        (root / "reports" / "final_report.md").unlink()
        deliverables = make_deliverables(root)
        _report_erratum_note()
        deliverables["final_report_sha256"] = file_sha256(
            root / "reports" / "final_report.md"
        )
        acceptance = validate_acceptance(root, write=False)
        if acceptance["status"] != "PASS":
            raise ValueError("semantic erratum check-only acceptance未PASS")

        current_freeze = assert_plan_freeze()
        if (
            current_freeze["core_protocol_sha256"]["src/gjca/analysis.py"]
            != FROZEN_ANALYSIS_SHA256
        ):
            raise ValueError("semantic erratum后frozen core SHA漂移")
        old_artifact_sha256 = {
            str(row["path"]): str(row["sha256"]) for row in archive_rows
        }
        old_artifact_bytes = {
            str(row["path"]): int(row["bytes"]) for row in archive_rows
        }
        new_artifact_sha256 = {
            relative: file_sha256(root / relative) for relative in targets
        }
        new_artifact_bytes = {
            relative: (root / relative).stat().st_size for relative in targets
        }
        if (
            set(old_artifact_sha256) != set(targets)
            or set(new_artifact_sha256) != set(targets)
            or len(old_artifact_sha256) != 68
        ):
            raise AssertionError("semantic erratum old/new full hash map漂移")
        old_diagnosis = json.loads(
            (ARCHIVE_ROOT / "metrics" / "diagnosis.json").read_text(
                encoding="utf-8"
            )
        )
        new_diagnosis = json.loads(
            (root / "metrics" / "diagnosis.json").read_text(encoding="utf-8")
        )
        diagnosis_keys = (
            "selected_hypothesis",
            "first_recommendation",
            "second_recommendation",
            "fix_executed",
        )
        old_diagnosis_summary = {
            key: old_diagnosis[key] for key in diagnosis_keys
        }
        new_diagnosis_summary = {
            key: new_diagnosis[key] for key in diagnosis_keys
        }
        if old_diagnosis_summary != new_diagnosis_summary or (
            new_diagnosis_summary
            != {
                "selected_hypothesis": "H4",
                "first_recommendation": amendment[
                    "expected_first_recommendation"
                ],
                "second_recommendation": amendment[
                    "expected_second_recommendation"
                ],
                "fix_executed": False,
            }
        ):
            raise ValueError("semantic erratum意外改变诊断/推荐")
        final_analysis_hashes = {
            relative: file_sha256(root / relative) for relative in ANALYSIS_FILES
        }
        final_delivery_hashes = {
            relative: file_sha256(root / relative)
            for relative in (
                "metrics/figure_manifest.csv",
                "reports/final_report.md",
                *FIGURE_FILES,
            )
        }
        payload: dict[str, Any] = {
            "schema_version": 1,
            "status": "applied_and_check_only_acceptance_passed",
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "semantic_erratum_sha256": file_sha256(AMENDMENT_PATH),
            "wrapper_sha256": file_sha256(wrapper_path),
            "archive_manifest_sha256": archive_manifest_sha,
            "archive_file_count": len(archive_rows),
            "old_artifact_sha256": old_artifact_sha256,
            "old_artifact_bytes": old_artifact_bytes,
            "new_artifact_sha256": new_artifact_sha256,
            "new_artifact_bytes": new_artifact_bytes,
            "plan_freeze_sha256": PLAN_SHA256,
            "frozen_core_sha256": current_freeze["core_protocol_sha256"],
            "upstream_probe_path": amendment["correct_source_path"],
            "upstream_probe_sha256": UPSTREAM_PROBE_SHA256,
            "resampling_indices_sha256": file_sha256(
                root / "metrics" / "resampling_indices.npz"
            ),
            "resampling_manifest_sha256": file_sha256(
                root / "metrics" / "resampling_manifest.json"
            ),
            "old_source_contract_sha256": OLD_SOURCE_CONTRACT_SHA256,
            "new_source_contract_sha256": new_source_sha,
            "corrected_existing_metrics_sha256": file_sha256(existing_path),
            "corrected_existing_token_rewrite": existing_rewrite,
            "corrected_phase_a_sha256": file_sha256(phase_path),
            "correct_vector_sha256": CORRECT_VECTOR_SHA256,
            "gradient_byte_replacement_file_count": len(gradient_replacements),
            "gradient_byte_replacement_count": sum(
                int(record["replacement_count"])
                for record in gradient_replacements
            ),
            "gradient_byte_replacements": gradient_replacements,
            "gradient_raw_combined_exact_before_erratum": False,
            "gradient_raw_combined_exact_after_erratum": False,
            "gradient_raw_combined_closure_before_erratum": (
                pre_erratum_gradient_closure
            ),
            "gradient_raw_combined_closure_after_erratum": (
                post_erratum_gradient_closure
            ),
            "gradient_raw_combined_key_nonfloat_exact": True,
            "gradient_raw_combined_float_ranks_exact": True,
            "gradient_raw_combined_max_ulp": 2,
            "aggregate_rows": aggregate_rows,
            "final_analysis_manifest_sha256": validated_analysis[
                "final_analysis_manifest_sha256"
            ],
            "final_analysis_artifact_sha256": final_analysis_hashes,
            "figure_manifest_sha256": file_sha256(
                root / "metrics" / "figure_manifest.csv"
            ),
            "final_report_sha256": file_sha256(
                root / "reports" / "final_report.md"
            ),
            "final_delivery_artifact_sha256": final_delivery_hashes,
            "old_diagnosis": old_diagnosis_summary,
            "new_diagnosis": new_diagnosis_summary,
            "selected_hypothesis": "H4",
            "first_recommendation": amendment["expected_first_recommendation"],
            "second_recommendation": amendment["expected_second_recommendation"],
            "formal_analysis_bundle_published_before_erratum": True,
            "gradient_numeric_values_observed_before_erratum": True,
            "input_values_modified": True,
            "formal_acceptance_written_before_erratum": False,
            "formal_acceptance_written_when_marker_created": False,
            "frozen_core_files_modified": False,
            "statistical_functions_modified": False,
            "threshold_logic_modified": False,
            "gradient_forward_backward_recomputed": False,
            "gradient_numeric_values_modified": False,
            "hypothesis_logic_modified": False,
            "check_only_acceptance_summary": _acceptance_summary(acceptance),
            "semantic_erratum_chain_validator_available": True,
            "contains_patient_ids": False,
        }
        payload["payload_sha256"] = canonical_json_sha256(payload)
        atomic_json(APPLIED_PATH, payload)
        applied_validation = _validate_applied_marker()
        return {
            "status": "ok",
            "archive_file_count": len(archive_rows),
            "archive_manifest_sha256": archive_manifest_sha,
            "new_source_contract_sha256": new_source_sha,
            "corrected_phase_a": amendment["expected_corrected_phase_a"],
            "selected_hypothesis": "H4",
            "first_recommendation": amendment["expected_first_recommendation"],
            "second_recommendation": amendment["expected_second_recommendation"],
            "applied_marker_sha256": file_sha256(APPLIED_PATH),
            "check_only_acceptance": acceptance["status"],
            "semantic_erratum_chain_validation": applied_validation["status"],
            "deliverables": deliverables,
        }
    except BaseException:
        _restore(archive_rows)
        raise


if args.validate_applied:
    print(
        json.dumps(
            _validate_applied_marker(),
            ensure_ascii=False,
            indent=2,
        )
    )
else:
    (
        amendment,
        corrected_existing,
        archive_targets,
        pre_erratum_gradient_closure,
    ) = _preflight()
    if args.preflight:
        print(
            json.dumps(
                {
                    "status": "ok",
                    "mode": "preflight",
                    "archive_target_count": len(archive_targets),
                    "correct_vector_sha256": CORRECT_VECTOR_SHA256,
                    "corrected_phase_a": amendment[
                        "expected_corrected_phase_a"
                    ],
                    "gradient_raw_combined_closure": (
                        pre_erratum_gradient_closure
                    ),
                    "frozen_core_files_modified": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(
            json.dumps(
                _apply(
                    amendment,
                    corrected_existing,
                    archive_targets,
                    pre_erratum_gradient_closure,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
PY

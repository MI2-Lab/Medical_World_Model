#!/usr/bin/env bash
# Post-freeze erratum: repair one provenance-only precondition without changing
# any frozen Python source, statistical calculation, threshold, or input value.
set -euo pipefail

script_path=$(cd -- "$(dirname -- "$0")" && pwd)/$(basename -- "$0")
audit_root=$(cd -- "$(dirname -- "$0")/.." && pwd)

python - "$audit_root" "$script_path" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


root = Path(sys.argv[1]).resolve()
wrapper_path = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(root / "src"))

import gjca.analysis as analysis  # noqa: E402
from gjca.contracts import (  # noqa: E402
    canonical_json_sha256,
    ensure_no_patient_columns,
    file_sha256,
)
from gjca.freeze import assert_plan_freeze  # noqa: E402


EXPECTED_PLAN_FILE_SHA256 = (
    "a8c3dc736b9f31d3b0d9f4efeaf5efa1605388b19a0604087e15dc9987287cf1"
)
EXPECTED_FROZEN_ANALYSIS_SHA256 = (
    "4e654f169b5fc281c070f6193d51ceffa704c8b6aa8c5439cb6c173cd35d47ff"
)
EXPECTED_FAILURE_SIGNATURE = (
    "ValueError: analysis input privacy flag 漂移: existing_run_metrics"
)
AMENDMENT_PATH = root / "POST_FREEZE_ERRATUM.json"


def validate_erratum_contract() -> None:
    if root != analysis.AUDIT_ROOT.resolve():
        raise ValueError("erratum 只允许正式冻结 audit root")
    freeze = assert_plan_freeze()
    plan_path = root / "PLAN_FREEZE.json"
    if (
        file_sha256(plan_path) != EXPECTED_PLAN_FILE_SHA256
        or freeze["core_protocol_sha256"]["src/gjca/analysis.py"]
        != EXPECTED_FROZEN_ANALYSIS_SHA256
        or file_sha256(root / "src/gjca/analysis.py")
        != EXPECTED_FROZEN_ANALYSIS_SHA256
    ):
        raise ValueError("erratum 冻结计划或 analysis.py SHA 漂移")
    if not AMENDMENT_PATH.is_file():
        raise FileNotFoundError("缺 POST_FREEZE_ERRATUM.json")
    amendment = json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))
    unsigned = dict(amendment)
    payload_sha = str(unsigned.pop("payload_sha256", ""))
    expected_keys = {
        "schema_version",
        "kind",
        "created_at_utc",
        "original_plan_freeze_sha256",
        "frozen_core_analysis_sha256",
        "failure_signature",
        "failure_stage",
        "formal_analysis_bundle_published_before_erratum",
        "numeric_gradient_values_inspected_before_erratum",
        "frozen_core_files_modified",
        "statistical_functions_modified",
        "thresholds_or_hypothesis_logic_modified",
        "input_values_modified",
        "scope",
        "wrapper_path",
        "wrapper_sha256",
        "payload_sha256",
    }
    if set(amendment) != expected_keys:
        raise ValueError("POST_FREEZE_ERRATUM schema 漂移")
    if (
        int(amendment["schema_version"]) != 1
        or amendment["kind"] != "postfreeze_provenance_only_erratum"
        or amendment["original_plan_freeze_sha256"]
        != EXPECTED_PLAN_FILE_SHA256
        or amendment["frozen_core_analysis_sha256"]
        != EXPECTED_FROZEN_ANALYSIS_SHA256
        or amendment["failure_signature"] != EXPECTED_FAILURE_SIGNATURE
        or amendment["failure_stage"] != "prepublication_analysis_input_manifest"
        or amendment["formal_analysis_bundle_published_before_erratum"] is not False
        or amendment["numeric_gradient_values_inspected_before_erratum"] is not False
        or amendment["frozen_core_files_modified"] is not False
        or amendment["statistical_functions_modified"] is not False
        or amendment["thresholds_or_hypothesis_logic_modified"] is not False
        or amendment["input_values_modified"] is not False
        or amendment["scope"]
        != "existing_run_metrics privacy provenance inference only"
        or amendment["wrapper_path"]
        != "scripts/run_analysis_with_provenance_erratum.sh"
        or amendment["wrapper_sha256"] != file_sha256(wrapper_path)
        or canonical_json_sha256(unsigned) != payload_sha
    ):
        raise ValueError("POST_FREEZE_ERRATUM payload/SHA 漂移")
    if (root / "metrics" / "FINAL_ANALYSIS_COMPLETE.json").exists():
        raise FileExistsError("正式 analysis bundle 已存在；erratum 拒绝覆盖")


def provenance_rows_with_erratum(path_root: Path) -> list[dict[str, Any]]:
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
            raise ValueError(f"erratum analysis input schema/rows 漂移: {role}")
        ensure_no_patient_columns(frame.columns)
        if "contains_patient_ids" in frame.columns:
            privacy = set(
                frame["contains_patient_ids"].astype(str).str.strip().str.lower()
            )
            if privacy != {"false"}:
                raise ValueError(f"erratum analysis input privacy flag 漂移: {role}")
        else:
            missing_flag_roles.append(role)
            if role != "existing_run_metrics":
                raise ValueError(f"erratum 非预期缺 privacy flag: {role}")
            object_columns = frame.select_dtypes(
                include=["object", "string"]
            ).columns.tolist()
            if object_columns != ["selection_mode", "base_gate"]:
                raise ValueError("existing_run_metrics object schema 漂移")
            if set(frame["selection_mode"].astype(str)) != {
                "primary",
                "fallback_base_gate_failed",
            }:
                raise ValueError("existing_run_metrics selection_mode 漂移")
            if set(frame["base_gate"].astype(str)) != {"PASS", "FAIL"}:
                raise ValueError("existing_run_metrics base_gate 漂移")
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
            raise AssertionError("erratum input manifest row schema 漂移")
        rows.append(row)
    if (
        missing_flag_roles != ["existing_run_metrics"]
        or len(rows) != 9
        or len({row["artifact_role"] for row in rows}) != 9
    ):
        raise AssertionError("erratum 必须只豁免 existing_run_metrics 一个 flag")
    return rows


validate_erratum_contract()
analysis._analysis_input_rows = provenance_rows_with_erratum
result = {
    "status": "ok",
    "postfreeze_erratum": "POST_FREEZE_ERRATUM.json",
    "frozen_core_files_modified": False,
    **analysis.write_full_analysis(root),
}
print(json.dumps(result, ensure_ascii=False, indent=2))
PY

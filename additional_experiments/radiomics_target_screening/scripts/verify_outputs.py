#!/usr/bin/env python3
"""Fail-closed verification for radiomics target screening deliverables."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


EXP_ROOT = Path(__file__).resolve().parents[1]
METRICS = EXP_ROOT / "metrics"
FIGURES = EXP_ROOT / "figures"
REPORTS = EXP_ROOT / "reports"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(name: str, passed: bool, detail: object) -> dict[str, object]:
    return {"check": name, "passed": bool(passed), "detail": detail}


def main() -> None:
    checks: list[dict[str, object]] = []
    manifest = pd.read_csv(METRICS / "output_manifest.csv")
    missing = []
    hash_mismatch = []
    for row in manifest.itertuples(index=False):
        path = EXP_ROOT / row.path
        if not path.is_file() or path.stat().st_size <= 0:
            missing.append(row.path)
        elif sha256(path) != row.sha256:
            hash_mismatch.append(row.path)
    checks.append(check("manifest files exist and are non-empty", not missing, missing))
    checks.append(check("manifest SHA256 values match", not hash_mismatch, hash_mismatch))

    expected_rows = {
        "table_schema.csv": 29,
        "candidate_coverage.csv": 4,
        "static_ftv_redundancy.csv": 72,
        "delta_ftv_redundancy.csv": 54,
        "residual_information.csv": 105,
        "longitudinal_variability.csv": 72,
        "feature_distribution_summary.csv": 16,
        "pairwise_static_spearman.csv": 16,
        "pairwise_delta_spearman.csv": 12,
        "shortcut_risk_summary.csv": 4,
        "observability_summary.csv": 4,
        "prior_decodability_summary.csv": 10,
        "fold_level_candidate_metrics.csv": 15,
        "candidate_decision_matrix.csv": 3,
    }
    observed_rows = {name: len(pd.read_csv(METRICS / name)) for name in expected_rows}
    checks.append(check("CSV row counts match registered structure", observed_rows == expected_rows, observed_rows))

    fold_metrics = pd.read_csv(METRICS / "fold_level_candidate_metrics.csv")
    expected_train_counts = {0: 247, 1: 239, 2: 240, 3: 242, 4: 225}
    actual_train_counts = (
        fold_metrics.groupby("fold")["n_fold_train_measurement_patients"].first().astype(int).to_dict()
    )
    checks.append(check("fold-train overlap counts match", actual_train_counts == expected_train_counts, actual_train_counts))
    checks.append(
        check(
            "formal metrics contain folds 0-4 and candidates exactly once",
            set(fold_metrics["fold"]) == set(range(5))
            and set(fold_metrics["candidate"]) == {"LD", "SPH", "BPE"}
            and not fold_metrics.duplicated(["fold", "candidate"]).any(),
            {"rows": len(fold_metrics)},
        )
    )
    numeric_columns = [
        "static_median_abs_spearman_with_ftv",
        "delta_median_abs_spearman_with_delta_ftv",
        "static_median_residual_variance_ratio",
        "delta_median_residual_variance_ratio",
        "within_to_total_variance_ratio",
    ]
    checks.append(
        check(
            "formal screening metrics are finite",
            np.isfinite(fold_metrics[numeric_columns].to_numpy(dtype=float)).all(),
            numeric_columns,
        )
    )
    checks.append(
        check(
            "no pCR used in fold metrics",
            "pcr_used" in fold_metrics.columns and not fold_metrics["pcr_used"].astype(bool).any(),
            fold_metrics["pcr_used"].unique().tolist(),
        )
    )

    decision = pd.read_csv(METRICS / "candidate_decision_matrix.csv").set_index("candidate")
    expected_decisions = {
        "LD": "RECOMMENDED",
        "SPH": "POSSIBLE SECOND CHOICE",
        "BPE": "NOT RECOMMENDED WITH CURRENT INPUT",
    }
    actual_decisions = decision["overall"].to_dict()
    checks.append(check("decision classes match registered selection", actual_decisions == expected_decisions, actual_decisions))
    checks.append(
        check(
            "observability gate excludes BPE only",
            decision.loc["BPE", "observability_gate"] == "FAIL_INPUT_MISMATCH"
            and decision.loc["LD", "observability_gate"] == "PASS"
            and str(decision.loc["SPH", "observability_gate"]).startswith("PASS"),
            decision["observability_gate"].to_dict(),
        )
    )

    selection = json.loads((METRICS / "final_target_selection.json").read_text(encoding="utf-8"))
    checks.append(
        check(
            "final JSON selection is outcome-free and complete",
            selection["recommended_target"] == "LD"
            and selection["second_choice"] == "SPH"
            and selection["statistically_attractive_but_input_mismatched"] == "BPE"
            and selection["recommendation_status"] == "conditional_pragmatic_first"
            and selection["selection_is_statistically_unique"] is False
            and selection["pareto_nondominated_current_input_candidates"] == ["LD", "SPH"]
            and selection["pcr_read_or_used_for_selection"] is False
            and selection["no_model_training_performed"] is True,
            {
                "recommended": selection["recommended_target"],
                "second": selection["second_choice"],
                "status": selection["recommendation_status"],
                "pcr_used": selection["pcr_read_or_used_for_selection"],
            },
        )
    )

    report = (REPORTS / "final_report.md").read_text(encoding="utf-8")
    schema_report = (REPORTS / "table_schema_report.md").read_text(encoding="utf-8")
    required_report_phrases = [
        "最终结论",
        "Outcome-free screening",
        "Excel 真实结构",
        "Patient coverage",
        "Static FTV redundancy",
        "Longitudinal raw-Δ redundancy",
        "Residual information beyond FTV",
        "Longitudinal responsiveness",
        "Shortcut / mask-geometry audit",
        "Current MRI input observability gate",
        "Prior frozen representation decodability",
        "Candidate Decision Matrix",
        "必答 A–J",
        "LD — RECOMMENDED",
        "H2: DCE7 → JEPA + FTV + LD",
    ]
    absent_phrases = [phrase for phrase in required_report_phrases if phrase not in report]
    checks.append(check("final report contains registered sections", not absent_phrases, absent_phrases))
    checks.append(
        check(
            "schema report contains verified workbook facts",
            all(
                phrase in schema_report
                for phrase in ("384×29", "datawith4visits", "formal candidate pool 恰为 LD、SPH、BPE", "不做 fuzzy matching")
            ),
            len(schema_report),
        )
    )

    image_details = {}
    image_ok = True
    for index in range(1, 11):
        matches = list(FIGURES.glob(f"{index:02d}_*.png"))
        if len(matches) != 1:
            image_ok = False
            image_details[f"{index:02d}"] = f"matches={len(matches)}"
            continue
        with Image.open(matches[0]) as image:
            image.verify()
        with Image.open(matches[0]) as image:
            width, height = image.size
        image_details[matches[0].name] = [width, height]
        if width < 800 or height < 500:
            image_ok = False
    checks.append(check("ten PNG figures are valid and readable", image_ok, image_details))

    status = "PASS" if all(bool(item["passed"]) for item in checks) else "FAIL"
    payload = {
        "status": status,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
    }
    output = METRICS / "verification.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

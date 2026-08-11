from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
VERIFY_PATH = EXPERIMENT_ROOT / "scripts" / "verify_audit.py"
SPEC = importlib.util.spec_from_file_location("compact_verify_under_test", VERIFY_PATH)
assert SPEC is not None and SPEC.loader is not None
verify = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verify
SPEC.loader.exec_module(verify)


CELL = ("comparison", "full_808", 2026, "LOCAL0", "T0")
CELL_COLUMNS = ["comparison_name", "population", "seed", "arm", "timing"]


def _bootstrap_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    effects = {
        "auroc_improvement": np.asarray([0.10, 0.20, 0.30, 0.40]),
        "auprc_improvement": np.asarray([-0.10, 0.00, 0.10, 0.20]),
        "brier_improvement": np.asarray([0.01, 0.02, 0.03, 0.04]),
    }
    draw_rows = []
    for index in range(4):
        draw_rows.append(
            {
                **dict(zip(CELL_COLUMNS, CELL, strict=True)),
                "bootstrap_index": index,
                **{name: float(values[index]) for name, values in effects.items()},
                "delta_brier": -float(effects["brier_improvement"][index]),
                "reference_selector": '{"model_key":"C"}',
                "comparison_selector": '{"model_key":"C+M|PCA_SELECTED"}',
                "bootstrap_seed": 12345,
            }
        )
    draws = pd.DataFrame(draw_rows)

    points = {
        "auroc": (0.50, 0.70, 0.20, np.nan),
        "auprc": (0.40, 0.50, 0.10, np.nan),
        "brier": (0.25, 0.20, 0.05, -0.05),
    }
    summary_rows = []
    for metric, (reference, comparison, improvement, delta_brier) in points.items():
        values = effects[f"{metric}_improvement"]
        lower, upper = np.quantile(values, [0.025, 0.975])
        summary_rows.append(
            {
                **dict(zip(CELL_COLUMNS, CELL, strict=True)),
                "metric": metric,
                "reference_value": reference,
                "comparison_value": comparison,
                "improvement": improvement,
                "ci_lower": lower,
                "ci_upper": upper,
                "confidence_level": 0.95,
                "n_patients": 4,
                "n_folds": 5,
                "n_bootstrap": 4,
                "n_valid_bootstrap": 4,
                "bootstrap_unit": "patient_within_outer_fold",
                "ci_method": "percentile",
                "orientation": (
                    "reference - comparison (lower Brier is better)"
                    if metric == "brier"
                    else "comparison - reference"
                ),
                "delta": comparison - reference,
                "delta_brier": delta_brier,
                "reference_selector": '{"model_key":"C"}',
                "comparison_selector": '{"model_key":"C+M|PCA_SELECTED"}',
                "bootstrap_seed": 12345,
            }
        )
    return pd.DataFrame(summary_rows), draws


def test_status_distinguishes_pending_final_from_core_failure() -> None:
    passed = [{"status": "PASS", "category": "core"}]
    pending = [*passed, {"status": "FAIL", "category": "final_deliverable"}]
    failed = [*pending, {"status": "FAIL", "category": "core"}]
    assert verify.classify_status(passed) == "PASS"
    assert verify.classify_status(pending) == "PRE_FINAL_FAIL"
    assert verify.classify_status(failed) == "FAIL"


def test_bootstrap_requires_exact_draws_and_brier_orientation() -> None:
    summary, draws = _bootstrap_frames()
    details = verify.validate_bootstrap_frames(
        summary,
        draws,
        expected_cells={CELL},
        population_counts={"full_808": 4},
        replicates=4,
    )
    assert details == {"comparison_cells": 1, "draw_rows": 4}


def test_bootstrap_rejects_duplicate_draw_index_and_wrong_delta_brier() -> None:
    summary, draws = _bootstrap_frames()
    duplicated = draws.copy()
    duplicated.loc[3, "bootstrap_index"] = 2
    with pytest.raises(verify.AuditVerificationError, match="duplicate bootstrap draw"):
        verify.validate_bootstrap_frames(
            summary,
            duplicated,
            expected_cells={CELL},
            population_counts={"full_808": 4},
            replicates=4,
        )

    wrong_brier = draws.copy()
    wrong_brier.loc[0, "delta_brier"] = 0.01
    with pytest.raises(verify.AuditVerificationError, match="delta_brier"):
        verify.validate_bootstrap_frames(
            summary,
            wrong_brier,
            expected_cells={CELL},
            population_counts={"full_808": 4},
            replicates=4,
        )


def test_atomic_json_replaces_complete_document(tmp_path: Path) -> None:
    output = tmp_path / "metrics" / "verification.json"
    verify._atomic_json(output, {"status": "PRE_FINAL_FAIL", "checks": []})
    verify._atomic_json(output, {"status": "PASS", "checks": [{"status": "PASS"}]})
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "status": "PASS",
        "checks": [{"status": "PASS"}],
    }
    assert not list(output.parent.glob("*.tmp"))

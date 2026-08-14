from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


SRC = Path(__file__).parents[1] / "src"
sys.path.insert(0, str(SRC))

from conditional_ceiling.gates import (  # noqa: E402
    GATE_A_PASS_CODE,
    GATE_B_PASS_CODE,
    GATE_C_PASS_CODE,
    classify_interpretation,
    evaluate_gate_a,
    evaluate_gate_b,
    evaluate_gate_c,
    evaluate_gates,
)


def _rows(
    arm: str,
    timing: str,
    deltas: tuple[float, float],
    population: str,
    ci: tuple[float, float] | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "seed": [2026, 3026],
            "arm": [arm, arm],
            "timing": [timing, timing],
            "population": [population, population],
            "delta_auroc": list(deltas),
        }
    )
    if ci is not None:
        frame["ci_lower"] = list(ci)
    return frame


def test_gate_a_uses_two_seed_mean_at_registered_mid_timings() -> None:
    passing = _rows("B2", "T0-T1", (0.07, 0.09), "full_808")
    decision = evaluate_gate_a(passing)
    assert decision.passed
    assert decision.pass_code == GATE_A_PASS_CODE
    assert decision.mean_delta_auroc == 0.08

    t0_only = _rows("B3", "T0", (0.2, 0.2), "full_808")
    assert not evaluate_gate_a(t0_only).passed
    one_high_one_low = _rows("B3", "T0_T2", (0.20, -0.05), "full_808")
    assert not evaluate_gate_a(one_high_one_low).passed


def test_gate_b_requires_both_positive_and_at_least_one_positive_ci_lower() -> None:
    passing = _rows(
        "B3", "T0_T2", (0.02, 0.01), "full_808", ci=(0.001, -0.01)
    )
    decision = evaluate_gate_b(passing)
    assert decision.passed
    assert decision.pass_code == GATE_B_PASS_CODE

    ci_crosses = passing.assign(ci_lower=[-0.001, -0.01])
    assert not evaluate_gate_b(ci_crosses).passed
    one_negative = passing.assign(delta_auroc=[0.02, -0.001])
    assert not evaluate_gate_b(one_negative).passed


def test_gate_c_requires_ftv_population_both_positive_and_mean_point03() -> None:
    passing = _rows("B2", "T0_T1", (0.02, 0.04), "ftv_complete_375")
    decision = evaluate_gate_c(passing)
    assert decision.passed
    assert decision.pass_code == GATE_C_PASS_CODE

    wrong_population = passing.assign(population="full_808")
    assert not evaluate_gate_c(wrong_population).passed
    below_mean = passing.assign(delta_auroc=[0.01, 0.02])
    assert not evaluate_gate_c(below_mean).passed


def test_b1_is_eligible_for_complementarity_gates_b_and_c_but_not_gate_a() -> None:
    b1_full = _rows(
        "B1", "T0_T1", (0.03, 0.04), "full_808", ci=(0.001, -0.01)
    )
    b1_ftv = _rows("B1", "T0_T1", (0.03, 0.04), "ftv_complete_375")

    assert evaluate_gate_b(b1_full).passed
    assert evaluate_gate_b(b1_full).best_arm == "B1"
    assert evaluate_gate_c(b1_ftv).passed
    assert evaluate_gate_c(b1_ftv).best_arm == "B1"
    assert not evaluate_gate_a(b1_full).passed


def test_interpretation_classes_follow_literal_goal_logic() -> None:
    assert classify_interpretation(True, True, True) == "A"
    assert classify_interpretation(True, True, False) == "B"
    assert (
        classify_interpretation(
            False,
            False,
            False,
            b1_improves_strongly=True,
            adaptation_adds_little=True,
        )
        == "C"
    )
    assert classify_interpretation(False, False, False, b3_low_ceiling=True) == "D"
    assert classify_interpretation(False, True, False) == "UNRESOLVED"


def test_evaluate_gates_returns_strong_hidden_ceiling_when_a_b_c_pass() -> None:
    a = pd.concat(
        [
            _rows("B1", "T0_T1", (0.01, 0.01), "full_808"),
            _rows("B3", "T0_T2", (0.08, 0.10), "full_808"),
        ],
        ignore_index=True,
    )
    b = _rows("B3", "T0_T2", (0.02, 0.03), "full_808", ci=(0.001, -0.01))
    c = _rows("B3", "T0_T2", (0.03, 0.05), "ftv_complete_375")
    result = evaluate_gates(a, b, c)

    assert result.interpretation_class == "A"
    assert result.interpretation_code == "STRONG_HIDDEN_MRI_CEILING"
    assert result.passed_codes == (
        GATE_A_PASS_CODE,
        GATE_B_PASS_CODE,
        GATE_C_PASS_CODE,
    )

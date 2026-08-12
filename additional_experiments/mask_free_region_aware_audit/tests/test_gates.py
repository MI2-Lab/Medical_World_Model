from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_audit as audit  # noqa: E402


def _config() -> dict:
    return {
        "frozen_cells": {"seed_bases": [2026, 3026]},
        "oracle": {"primary_arm": "LOCAL0", "view": "T0-T1"},
        "gates": {
            "A": {
                "candidates": ["R2", "R3", "R5"],
                "timings": ["T0", "T0-T1", "T0-T2"],
                "minimum_two_seed_mean_gain": 0.02,
            },
            "B": {
                "candidates": ["R1", "R2", "R3", "R4", "R5"],
                "timings": ["T0", "T0-T1", "T0-T2"],
                "minimum_two_seed_mean_vs_cf": 0.0,
            },
            "C": {
                "candidates": ["R1", "R2", "R3", "R5"],
                "minimum_two_seed_mean_recovery_ratio": 0.30,
            },
            "D": {
                "candidates": ["R1", "R2", "R3", "R4", "R5"],
                "minimum_auroc_gain_each_seed": 0.03,
            },
        },
    }


def _gate_tables(
    *,
    a_deltas: tuple[float, float] = (0.021, 0.021),
    b_vs_r0: tuple[float, float] = (0.01, 0.01),
    b_vs_cf: tuple[float, float] = (0.01, -0.005),
    c_numerator: tuple[float, float] = (0.012, 0.012),
    c_ratio: tuple[float, float] = (0.30, 0.30),
    d_deltas: tuple[float, float] = (0.03, 0.03),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mri = []
    incremental = []
    phenotype = []
    oracle = []
    for index, seed in enumerate((2026, 3026)):
        mri.extend(
            [
                {
                    "seed": seed, "arm": "LOCAL0", "analysis": "mri_only_pcr",
                    "view": "T0-T1", "target": "pCR", "population": "full_808",
                    "variant": "R0", "auroc": 0.60,
                },
                {
                    "seed": seed, "arm": "LOCAL0", "analysis": "mri_only_pcr",
                    "view": "T0-T1", "target": "pCR", "population": "full_808",
                    "variant": "R2", "auroc": 0.60 + a_deltas[index],
                },
            ]
        )
        cf = 0.61
        cf_r0 = cf + b_vs_cf[index] - b_vs_r0[index]
        candidate = cf_r0 + b_vs_r0[index]
        incremental.extend(
            [
                {
                    "seed": seed, "arm": "LOCAL0", "analysis": "clinical_ftv_pcr",
                    "view": "T0-T1", "target": "pCR", "population": "ftv_complete_375",
                    "variant": "NONE", "model": "C+F", "auroc": cf,
                },
                {
                    "seed": seed, "arm": "LOCAL0", "analysis": "clinical_ftv_pcr",
                    "view": "T0-T1", "target": "pCR", "population": "ftv_complete_375",
                    "variant": "R0", "model": "C+F+R0", "auroc": cf_r0,
                },
                {
                    "seed": seed, "arm": "LOCAL0", "analysis": "clinical_ftv_pcr",
                    "view": "T0-T1", "target": "pCR", "population": "ftv_complete_375",
                    "variant": "R1", "model": "C+F+R1", "auroc": candidate,
                },
            ]
        )
        phenotype.extend(
            [
                {
                    "seed": seed, "arm": "LOCAL0", "analysis": "phenotype", "view": "T0",
                    "target": "HR", "population": "full_808", "variant": "R0", "auroc": 0.60,
                },
                {
                    "seed": seed, "arm": "LOCAL0", "analysis": "phenotype", "view": "T0",
                    "target": "HR", "population": "full_808", "variant": "R1",
                    "auroc": 0.60 + d_deltas[index],
                },
            ]
        )
        oracle.append(
            {
                "seed": seed, "arm": "LOCAL0", "view": "T0-T1", "candidate": "R1",
                "numerator_auroc_uplift": c_numerator[index],
                "published_oracle_uplift": 0.04,
                "recovery_ratio": c_ratio[index], "recovery_defined": True,
            }
        )
    return pd.DataFrame(mri), pd.DataFrame(incremental), pd.DataFrame(phenotype), pd.DataFrame(oracle)


def test_gates_a_to_d_pass_exact_registered_conditions() -> None:
    gates = audit.evaluate_gates(_config(), *_gate_tables())
    assert {letter: gates["gates"][letter]["passed"] for letter in "ABCD"} == {
        "A": True, "B": True, "C": True, "D": True,
    }
    assert gates["scientific_classification"] == "DEPLOYABLE_REGION_AWARE_SIGNAL_SUPPORTED"
    assert gates["gates"]["A"]["supporting_comparisons"][0]["mean_delta"] == pytest.approx(0.021)


@pytest.mark.parametrize(
    ("a_deltas", "expected"),
    [
        ((0.0, 0.05), False),
        ((0.019, 0.019), False),
        ((0.020, 0.020), True),
    ],
)
def test_gate_a_requires_each_seed_positive_and_mean_at_least_point02(
    a_deltas: tuple[float, float], expected: bool
) -> None:
    result = audit.evaluate_gates(_config(), *_gate_tables(a_deltas=a_deltas))
    assert result["gates"]["A"]["passed"] is expected


def test_gate_b_requires_each_seed_vs_cf_r0_and_nonnegative_two_seed_mean_vs_cf() -> None:
    result = audit.evaluate_gates(
        _config(), *_gate_tables(b_vs_r0=(0.01, 0.0), b_vs_cf=(0.02, 0.02))
    )
    assert result["gates"]["B"]["passed"] is False
    result = audit.evaluate_gates(
        _config(), *_gate_tables(b_vs_r0=(0.01, 0.01), b_vs_cf=(-0.02, 0.01))
    )
    assert result["gates"]["B"]["passed"] is False
    result = audit.evaluate_gates(
        _config(), *_gate_tables(b_vs_r0=(0.01, 0.01), b_vs_cf=(-0.01, 0.01))
    )
    assert result["gates"]["B"]["passed"] is True


def test_gate_c_requires_positive_denominator_positive_numerators_and_mean_ratio() -> None:
    result = audit.evaluate_gates(
        _config(), *_gate_tables(c_numerator=(0.0, 0.03), c_ratio=(0.6, 0.6))
    )
    assert result["gates"]["C"]["passed"] is False
    result = audit.evaluate_gates(_config(), *_gate_tables(c_ratio=(0.29, 0.30)))
    assert result["gates"]["C"]["passed"] is False
    tables = list(_gate_tables())
    tables[3].loc[tables[3]["seed"].eq(3026), "published_oracle_uplift"] = -0.04
    tables[3].loc[tables[3]["seed"].eq(3026), "recovery_defined"] = False
    result = audit.evaluate_gates(_config(), *tables)
    assert result["gates"]["C"]["passed"] is False


def test_gate_d_threshold_is_inclusive_for_each_seed() -> None:
    result = audit.evaluate_gates(_config(), *_gate_tables(d_deltas=(0.03, 0.029999)))
    assert result["gates"]["D"]["passed"] is False
    result = audit.evaluate_gates(_config(), *_gate_tables(d_deltas=(0.03, 0.03)))
    assert result["gates"]["D"]["passed"] is True


@pytest.mark.parametrize(
    ("gate_a", "gate_b", "gate_c", "positive", "expected"),
    [
        (True, False, True, True, "DEPLOYABLE_REGION_AWARE_SIGNAL_SUPPORTED"),
        (True, False, False, True, "REGION_SIGNAL_EXISTS_BUT_NOT_BEYOND_FTV"),
        (False, True, False, True, "ORACLE_REQUIRES_LESION_RELATIVE_LOCALIZATION"),
        (False, False, False, False, "MASK_FREE_REGIONALIZATION_NOT_SUPPORTED"),
        (False, True, True, True, "INDETERMINATE_DIAGNOSTIC"),
    ],
)
def test_scientific_classification_precedence(
    gate_a: bool, gate_b: bool, gate_c: bool, positive: bool, expected: str
) -> None:
    assert audit.scientific_classification(
        gate_a=gate_a, gate_b=gate_b, gate_c=gate_c,
        any_two_seed_positive=positive,
    ) == expected


def test_gate_tables_reject_duplicate_metric_identity() -> None:
    tables = list(_gate_tables())
    tables[0] = pd.concat((tables[0], tables[0].iloc[[0]]), ignore_index=True)
    with pytest.raises(ValueError, match="repeats"):
        audit.evaluate_gates(_config(), *tables)


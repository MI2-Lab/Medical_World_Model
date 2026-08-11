from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_audit import (  # noqa: E402
    _prepare_output_policy,
    causal_prefix,
    evaluate_gates,
    oracle_complete_case_mask,
    timing_end_index,
)


SEEDS = (2026, 3026)


def _config() -> dict:
    return {
        "frozen_cells": {"seed_bases": list(SEEDS)},
        "analysis": {"primary_pcr_population": "ftv_complete_375"},
        "gates": {
            "A": {"minimum_auroc_gain_each_seed": 0.03},
            "B": {
                "timings": ["T0", "T0-T1", "T0-T2"],
                "minimum_gain_each_seed_strictly_gt": 0.0,
            },
            "C": {"minimum_matched_auroc_gain_each_seed": 0.03},
            "D": {
                "minimum_seed_mean_auroc_near_chance": 0.45,
                "maximum_seed_mean_auroc_near_chance": 0.55,
            },
        },
    }


def _metric_rows(
    *,
    target: str,
    population: str,
    values: dict[str, float],
    view: str = "T0",
) -> list[dict]:
    return [
        {
            "seed": seed,
            "arm": "LOCAL0",
            "view": view,
            "target": target,
            "variant": variant,
            "population": population,
            "auroc": auroc,
        }
        for seed in SEEDS
        for variant, auroc in values.items()
    ]


def _synthetic_tables(
    *, phenotype_p3: float = 0.50, phenotype_p4: float = 0.50, oracle_core: float = 0.50
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    phenotype = pd.DataFrame(
        _metric_rows(
            target="HER2",
            population="full_808",
            values={"P1": 0.50, "P3": phenotype_p3, "P4": phenotype_p4},
        )
    )
    mri_pcr = pd.DataFrame(
        _metric_rows(
            target="pCR",
            population="ftv_complete_375",
            values={"P1": 0.50, "P3": 0.50},
        )
    )
    beyond = pd.DataFrame(
        [
            {
                "seed": seed,
                "arm": "LOCAL0",
                "view": "T0",
                "target": "pCR",
                "population": "ftv_complete_375",
                "model": model,
                "auroc": 0.50,
            }
            for seed in SEEDS
            for model in ("C+F", "C+F+P1", "C+F+P3")
        ]
    )
    oracle_rows: list[dict] = []
    for comparator in ("CORE", "PERI10", "PERI20", "CORE_PERI"):
        oracle_rows.extend(
            _metric_rows(
                target="HER2",
                population=f"oracle_pair_{comparator}",
                values={
                    "FIXED_P3": 0.50,
                    comparator: oracle_core if comparator == "CORE" else 0.50,
                },
            )
        )
    return phenotype, mri_pcr, beyond, pd.DataFrame(oracle_rows)


def test_causal_prefix_excludes_future_visits_and_rejects_nonregistered_view() -> None:
    features = np.arange(8, dtype=np.float32).reshape(2, 4, 1)
    prefix = causal_prefix(features, "T0-T2")
    assert prefix.shape == (2, 3)
    assert np.array_equal(prefix, features[:, :3].reshape(2, 3))
    assert timing_end_index("T0") == 0
    assert timing_end_index("T0-T3") == 3
    with pytest.raises(ValueError, match="unregistered"):
        causal_prefix(features, "T1-T3")


def test_oracle_masks_are_pair_specific_and_prefix_causal() -> None:
    valid = np.ones((3, 4, 4), dtype=bool)
    valid[0, 0, 3] = False  # unrelated LOCAL_REST invalidity
    valid[0, 0, 0] = False  # unrelated CORE invalidity for a PERI10-only pair
    valid[1, 0, 1] = False  # PERI10 invalidity
    valid[2, 3, 1] = False  # future PERI10 invalidity

    peri_t0 = oracle_complete_case_mask(valid, "T0", "PERI10", prefix=False)
    local_t0 = oracle_complete_case_mask(valid, "T0", "LOCAL_REST", prefix=False)
    assert np.array_equal(peri_t0, np.asarray([True, False, True]))
    assert np.array_equal(local_t0, np.asarray([False, True, True]))
    assert not oracle_complete_case_mask(valid, "T0", "CORE", prefix=False)[0]
    assert not oracle_complete_case_mask(valid, "T0", "CORE_PERI", prefix=False)[0]

    # A T0-T2 prediction cannot inspect the invalid T3 mask.
    assert oracle_complete_case_mask(valid, "T0-T2", "PERI10", prefix=True)[2]
    assert not oracle_complete_case_mask(valid, "T0-T3", "PERI10", prefix=True)[2]


def test_t0_only_patient_is_retained_at_t0_without_future_selection() -> None:
    valid = np.zeros((2, 4, 4), dtype=bool)
    valid[0, 0] = True  # T0-authorized patient with no later support.
    valid[1] = True
    assert np.array_equal(
        oracle_complete_case_mask(valid, "T0", "CORE", prefix=False),
        [True, True],
    )
    assert np.array_equal(
        oracle_complete_case_mask(valid, "T0-T1", "CORE", prefix=True),
        [False, True],
    )


def test_gate_a_requires_both_seed_deltas_and_drives_classification_a() -> None:
    tables = _synthetic_tables(phenotype_p3=0.54)
    gates = evaluate_gates(_config(), *tables)
    assert gates["gates"]["A"]["passed"]
    evidence = gates["gates"]["A"]["supporting_comparisons"]
    assert evidence[0]["seed_deltas"] == {
        "2026": pytest.approx(0.04),
        "3026": pytest.approx(0.04),
    }
    assert (
        gates["scientific_classification"]
        == "PHENOTYPE INFORMATION PRESENT BUT MEAN-POOLED AWAY"
    )
    assert gates["stage_b_authorized"]
    json.dumps(gates, allow_nan=False)


def test_gate_c_uses_pair_matched_fixed_p3_and_drives_classification_b() -> None:
    tables = _synthetic_tables(oracle_core=0.54)
    gates = evaluate_gates(_config(), *tables)
    assert not gates["gates"]["A"]["passed"]
    assert gates["gates"]["C"]["passed"]
    support = gates["gates"]["C"]["supporting_comparisons"]
    assert support[0]["population"] == "oracle_pair_CORE"
    assert support[0]["reference"] == "FIXED_P3"
    assert gates["scientific_classification"] == "PHENOTYPE SPATIALLY LOCALIZED"


def test_gate_d_requires_two_sided_near_chance_interval() -> None:
    near_chance = _synthetic_tables()
    passed = evaluate_gates(_config(), *near_chance)
    assert passed["gates"]["D"]["passed"]
    assert (
        passed["scientific_classification"]
        == "CURRENT ENCODER LACKS PHENOTYPE INFORMATION"
    )

    inverted = _synthetic_tables(phenotype_p4=0.40)
    failed = evaluate_gates(_config(), *inverted)
    assert not failed["gates"]["D"]["passed"]
    assert failed["gates"]["D"]["observed_minimum_seed_mean_auroc"] == pytest.approx(
        0.40
    )
    assert failed["scientific_classification"] == "MIXED"


def test_gate_b_requires_both_positive_contrasts_in_the_same_timing() -> None:
    phenotype, mri_pcr, beyond, oracle = _synthetic_tables()
    # Seed 3026 fails the P3-vs-P1 contrast, so direction is not consistent.
    beyond.loc[(beyond["seed"] == 2026) & (beyond["model"] == "C+F+P3"), "auroc"] = 0.53
    beyond.loc[(beyond["seed"] == 3026) & (beyond["model"] == "C+F+P3"), "auroc"] = 0.49
    failed = evaluate_gates(_config(), phenotype, mri_pcr, beyond, oracle)
    assert not failed["gates"]["B"]["passed"]

    beyond.loc[beyond["model"] == "C+F+P3", "auroc"] = 0.53
    passed = evaluate_gates(_config(), phenotype, mri_pcr, beyond, oracle)
    assert passed["gates"]["B"]["passed"]
    assert not passed["gates"]["D"]["passed"]
    assert (
        passed["scientific_classification"]
        == "PHENOTYPE INFORMATION PRESENT BUT MEAN-POOLED AWAY"
    )


def test_output_policy_recovers_only_partial_run_and_freezes_completed_run(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "metrics" / "table.csv"
    summary = tmp_path / "metrics" / "run_summary.json"
    artifact.parent.mkdir()
    artifact.write_text("partial\n", encoding="utf-8")
    paths = {"table": artifact, "run_summary": summary}

    removed = _prepare_output_policy(paths)
    assert removed == (artifact,)
    assert not artifact.exists()

    artifact.write_text("complete\n", encoding="utf-8")
    summary.write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="immutable"):
        _prepare_output_policy(paths)
    assert artifact.exists()
    assert summary.exists()

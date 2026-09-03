from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raw_spatial_pcr.contracts import FOLDS, PRIMARY_TIMINGS, SEEDS, load_contract


def test_frozen_contract_loads() -> None:
    contract = load_contract()
    assert contract.seeds == SEEDS
    assert contract.folds == FOLDS
    assert contract.primary_timings == PRIMARY_TIMINGS
    assert contract.config["input"]["modality"] == "C1B-H DCE7"


def test_timing_steps_are_causal() -> None:
    contract = load_contract()
    assert [contract.timing_steps(value) for value in ("T0", "T0_T1", "T0_T2", "T0_T3")] == [1, 2, 3, 4]


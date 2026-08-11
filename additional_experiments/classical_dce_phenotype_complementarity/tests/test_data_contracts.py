from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from data_contracts import (  # noqa: E402
    FAMILIES,
    RAW_COLUMNS,
    VISITS,
    build_feature_frame,
    canonical_mri_trial_id,
    canonical_trial_id,
)


def _synthetic_cohort() -> pd.DataFrame:
    data = {}
    for family in FAMILIES:
        for index, visit in enumerate(VISITS):
            data[RAW_COLUMNS[family][visit]] = np.array([1.0, 2.0, 4.0]) * (index + 1)
    return pd.DataFrame(data)


def test_exact_identifier_contract() -> None:
    assert canonical_trial_id(123456) == "123456"
    assert canonical_mri_trial_id("ISPY2-123456") == "123456"
    assert canonical_mri_trial_id("ACRIN-6698-123456") == "123456"
    with pytest.raises(ValueError):
        canonical_trial_id("12345")
    with pytest.raises(ValueError):
        canonical_mri_trial_id("subject-123456")


@pytest.mark.parametrize("timing,expected", [("T0", 4), ("T1", 16), ("T2", 28), ("T3", 40)])
def test_longitudinal_prefix_has_only_visible_information(timing: str, expected: int) -> None:
    built = build_feature_frame(_synthetic_cohort(), timing, "longitudinal", FAMILIES)
    assert built.values.shape == (3, expected)
    end_indices = built.metadata["end_visit"].map(VISITS.index)
    assert end_indices.max() <= VISITS.index(timing)


def test_static_uses_current_visit_only_and_recomputes_change() -> None:
    cohort = _synthetic_cohort()
    static = build_feature_frame(cohort, "T2", "static", ("FTV", "LD"))
    assert list(static.values) == ["FTV__absolute__T2", "LD__absolute__T2"]
    longitudinal = build_feature_frame(cohort, "T1", "longitudinal", ("FTV",))
    np.testing.assert_allclose(longitudinal.values["FTV__delta__T0_T1"], [1.0, 2.0, 4.0])
    np.testing.assert_allclose(longitudinal.values["FTV__relative_pct__T0_T1"], [100.0] * 3)

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_core import (  # noqa: E402
    audit_axis_aligned_roi_against_fov,
    decide_scientific_classification,
    deterministic_case_sample,
    physical_bounds_xyz,
)


def affine(spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    result = np.eye(4, dtype=float)
    result[:3, :3] = np.diag(spacing)
    result[:3, 3] = origin
    return result


def test_voxel_footprint_bounds_use_half_voxels():
    low, high = physical_bounds_xyz((4, 6, 8), affine((2.0, 3.0, 4.0)))
    np.testing.assert_allclose(low, (-1.0, -1.5, -2.0))
    np.testing.assert_allclose(high, (7.0, 16.5, 30.0))


def test_physical_roi_audit_reports_occupancy_touch_and_margin():
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[1:3, 1:3, 1:3] = True
    audit = audit_axis_aligned_roi_against_fov(
        mask, affine(), (4, 4, 4), affine()
    )
    assert audit.occupancy == pytest.approx(1.0)
    assert not audit.boundary_touch_any
    assert audit.centroid_inside
    assert audit.physical_margin_mm == pytest.approx(1.0)

    clipped = audit_axis_aligned_roi_against_fov(
        mask, affine(), (2, 4, 4), affine()
    )
    assert clipped.occupancy == pytest.approx(0.5)
    assert clipped.boundary_touch_x
    assert clipped.physical_margin_mm == pytest.approx(-1.0)


def test_oblique_roi_fails_closed():
    mask = np.ones((2, 2, 2), dtype=bool)
    oblique = affine()
    oblique[0, 1] = 0.1
    with pytest.raises(ValueError, match="axis-aligned"):
        audit_axis_aligned_roi_against_fov(mask, oblique, (4, 4, 4), affine())


def test_source_unavailable_has_priority_over_all_fov_guesses():
    assert decide_scientific_classification(
        source_roi_available=False,
        local_gate_pass=True,
        c1b_gate_pass=True,
        acquisition_source_complete=True,
    ) == ("D", "BPE_SOURCE_NOT_RELIABLY_AUDITABLE")


@pytest.mark.parametrize(
    "local,c1b,acquisition,expected",
    [
        (True, None, None, "C"),
        (False, True, None, "A"),
        (False, False, True, "B"),
        (False, False, False, "E"),
    ],
)
def test_evaluable_classification_tree(local, c1b, acquisition, expected):
    code, _ = decide_scientific_classification(
        source_roi_available=True,
        local_gate_pass=local,
        c1b_gate_pass=c1b,
        acquisition_source_complete=acquisition,
    )
    assert code == expected


def test_case_sampling_is_order_invariant_and_deterministic():
    first, digest_first = deterministic_case_sample(
        ["p3", "p1", "p2"], 2, salt="fixed"
    )
    second, digest_second = deterministic_case_sample(
        ["p2", "p3", "p1"], 2, salt="fixed"
    )
    assert first == second
    assert digest_first == digest_second

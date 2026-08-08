#!/usr/bin/env python3
"""Synthetic-only smoke checks for Stage A geometry utilities."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from ftv_ld_pilot.geometry import (  # noqa: E402
    approx_max_extent_mm,
    bbox_xyz,
    crop_or_pad_from_start,
    geometry_metrics,
    recover_origin,
)


def _assert_close(actual: float, expected: float, *, atol: float = 1e-12) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=atol):
        raise AssertionError(f"expected {expected}, got {actual}")


def smoke_fully_contained_and_orientation() -> dict[str, object]:
    mask = np.zeros((18, 19, 20), dtype=bool)  # raw XYZ
    mask[5:9, 6:11, 7:10] = True
    start_xyz = (3, 4, 5)
    crop_size_zyx = (8, 10, 10)

    cropped = crop_or_pad_from_start(mask, start_xyz, crop_size_zyx)
    assert cropped.dtype == np.bool_
    assert cropped.shape == crop_size_zyx
    assert int(cropped.sum()) == int(mask.sum())

    # A raw XYZ point must land at the corresponding transposed ZYX offset.
    point_mask = np.zeros((9, 9, 9), dtype=bool)
    point_mask[2, 3, 4] = True
    point_crop = crop_or_pad_from_start(point_mask, (1, 1, 1), (5, 5, 5))
    assert point_crop[3, 2, 1]
    assert int(point_crop.sum()) == 1

    metrics = geometry_metrics(mask, start_xyz, (0.7, 0.8, 2.0), crop_size_zyx)
    assert metrics["bbox_xyz"]["min_xyz"] == [5, 6, 7]
    assert metrics["bbox_xyz"]["max_xyz"] == [8, 10, 9]
    assert metrics["signed_margins_voxel"] == {
        "x_low": 2,
        "x_high": 4,
        "y_low": 2,
        "y_high": 3,
        "z_low": 2,
        "z_high": 3,
    }
    assert metrics["min_margin_voxel"] == 2
    assert metrics["containment_ratio"] == 1.0
    assert not metrics["any_boundary_touch"]
    assert metrics["crop_physical_extent_xyz_mm"] == [7.0, 8.0, 16.0]
    return {
        "contained_voxels": metrics["contained_voxels"],
        "min_margin_voxel": metrics["min_margin_voxel"],
        "orientation": "XYZ_to_ZYX",
    }


def smoke_boundary_touch() -> dict[str, object]:
    mask = np.zeros((14, 14, 14), dtype=bool)
    mask[3:6, 5:8, 5:8] = True
    metrics = geometry_metrics(mask, (3, 4, 4), (1.0, 1.0, 1.0), (8, 8, 8))
    assert metrics["containment_ratio"] == 1.0
    assert metrics["signed_margins_voxel"]["x_low"] == 0
    assert metrics["boundary_touch"]["x_low"]
    assert metrics["any_boundary_touch"]
    return {
        "touch_face": "x_low",
        "min_margin_voxel": metrics["min_margin_voxel"],
    }


def smoke_truncation() -> dict[str, object]:
    mask = np.zeros((16, 16, 16), dtype=bool)
    mask[10:15, 5:8, 5:8] = True
    metrics = geometry_metrics(mask, (4, 4, 4), (1.0, 1.5, 2.0), (8, 8, 8))
    _assert_close(metrics["containment_ratio"], 18.0 / 45.0)
    assert metrics["signed_margins_voxel"]["x_high"] == -3
    _assert_close(metrics["signed_margins_mm"]["x_high"], -3.0)
    assert metrics["boundary_touch"]["x_high"]
    assert metrics["min_margin_voxel"] == -3
    return {
        "containment_ratio": metrics["containment_ratio"],
        "signed_x_high_margin_voxel": metrics["signed_margins_voxel"]["x_high"],
    }


def smoke_padding() -> dict[str, object]:
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[0:3, 0:3, 0:2] = True
    crop_size_zyx = (8, 9, 10)
    cropped = crop_or_pad_from_start(mask, (-2, -1, -3), crop_size_zyx)
    metrics = geometry_metrics(mask, (-2, -1, -3), (0.5, 0.75, 2.5), crop_size_zyx)
    assert cropped.shape == crop_size_zyx
    assert int(cropped.sum()) == int(mask.sum())
    assert metrics["padding_voxel"] == {
        "x_low": 2,
        "x_high": 0,
        "y_low": 1,
        "y_high": 0,
        "z_low": 3,
        "z_high": 0,
    }
    assert metrics["containment_ratio"] == 1.0
    return {"padding_voxel": metrics["padding_voxel"]}


def smoke_origin_recovery() -> dict[str, object]:
    mask = np.zeros((18, 18, 18), dtype=bool)
    mask[6:10, 7:11, 6:9] = True
    mask[10, 8, 7] = True  # break symmetries deterministically
    clean_start = (3, 4, 4)
    true_start = (4, 3, 4)  # includes both +1 and -1 offsets
    crop_size_zyx = (7, 9, 8)
    actual = crop_or_pad_from_start(mask, true_start, crop_size_zyx)

    recovered = recover_origin(mask, actual, clean_start, crop_size_zyx, radius=1)
    assert recovered == {
        "status": "unique",
        "chosen_start": [4, 3, 4],
        "candidate_count": 1,
        "method": "exact_mask_unique",
    }

    # Uniform support intentionally has multiple exact origins.  The clean
    # start is nearest and must win deterministically.
    uniform = np.ones((12, 12, 12), dtype=bool)
    uniform_crop = crop_or_pad_from_start(uniform, (4, 4, 4), (4, 4, 4))
    ambiguous = recover_origin(uniform, uniform_crop, (4, 4, 4), (4, 4, 4), radius=1)
    assert ambiguous["status"] == "multiple"
    assert ambiguous["candidate_count"] == 27
    assert ambiguous["chosen_start"] == [4, 4, 4]
    assert ambiguous["method"] == "exact_mask_multiple_nearest_then_lexicographic"

    empty_actual = np.zeros(crop_size_zyx, dtype=bool)
    unresolved = recover_origin(
        mask, empty_actual, clean_start, crop_size_zyx, radius=2
    )
    assert unresolved == {
        "status": "no_match",
        "chosen_start": None,
        "candidate_count": 0,
        "method": "empty_actual_roi_nonidentifying",
    }
    return {
        "unique": recovered,
        "multiple_candidate_count": ambiguous["candidate_count"],
        "empty_cached_status": unresolved["status"],
    }


def smoke_bbox_and_approx_extent() -> dict[str, object]:
    empty = np.zeros((5, 5, 5), dtype=bool)
    assert bbox_xyz(empty) is None
    empty_extent = approx_max_extent_mm(empty, (1.0, 1.0, 1.0))
    assert empty_extent["component_count"] == 0
    assert empty_extent["whole_union_approx_max_extent_mm"] is None

    mask = np.zeros((12, 12, 12), dtype=bool)
    mask[1:5, 1, 1] = True  # largest component: four voxels
    mask[8:11, 8, 8] = True  # second component: three voxels
    extent = approx_max_extent_mm(mask, (2.0, 1.0, 0.5))
    assert extent["component_count"] == 2
    assert extent["whole_union_voxel_count"] == 7
    assert extent["largest_component_voxel_count"] == 4
    _assert_close(extent["largest_component_approx_max_extent_mm"], 6.0)
    assert (
        extent["whole_union_approx_max_extent_mm"]
        > extent["largest_component_approx_max_extent_mm"]
    )
    assert (
        extent["whole_union_approx_max_extent_mm"]
        <= extent["whole_union_bbox_diagonal_mm"] + 1e-12
    )
    return {
        "component_count": extent["component_count"],
        "whole_union_approx_max_extent_mm": extent["whole_union_approx_max_extent_mm"],
        "largest_component_approx_max_extent_mm": extent[
            "largest_component_approx_max_extent_mm"
        ],
    }


def main() -> None:
    checks = {
        "fully_contained_and_orientation": smoke_fully_contained_and_orientation(),
        "boundary_touch": smoke_boundary_touch(),
        "truncation": smoke_truncation(),
        "padding": smoke_padding(),
        "origin_recovery": smoke_origin_recovery(),
        "bbox_and_approx_extent": smoke_bbox_and_approx_extent(),
    }
    print(json.dumps({"status": "PASS", "checks": checks}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

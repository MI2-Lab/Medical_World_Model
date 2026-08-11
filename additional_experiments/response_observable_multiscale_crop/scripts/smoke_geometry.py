#!/usr/bin/env python3
"""Synthetic smoke tests for physical NIfTI and crop geometry contracts.

The checks are deliberately independent of the private I-SPY inputs.  They
exercise the exact coordinate, containment, temporal-anchor, morphology, and
multiscale invariants that Stage A relies on and emit one machine-readable JSON
summary.  Any failed assertion makes the process exit non-zero.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import struct
import sys
import tempfile
from typing import Any, Iterable

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from observable_crop.geometry import (  # noqa: E402
    PhysicalWindow,
    audit_support,
    bbox_footprint_in_frame,
    make_fixed_expand_window,
    make_union_window,
    orthonormal_index_basis,
)
from observable_crop.geometry import make_tight_resize_window  # noqa: E402
from observable_crop.nifti import (  # noqa: E402
    affine_max_corner_disagreement_mm,
    read_nifti_geometry,
)


def _assert_close(
    actual: Any,
    expected: Any,
    *,
    atol: float = 1e-7,
) -> None:
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=atol)


def _expect_raises(exception: type[BaseException], function: Any, *args: Any, **kwargs: Any) -> None:
    try:
        function(*args, **kwargs)
    except exception:
        return
    raise AssertionError(f"expected {exception.__name__} from {function.__name__}")


def _write_nifti_header(
    path: Path,
    *,
    shape_xyz: tuple[int, int, int] = (9, 10, 11),
    spacing_xyz: tuple[float, float, float] = (1.0, 1.5, 2.0),
    qform_code: int = 0,
    sform_code: int = 0,
    quaternion_bcd: tuple[float, float, float] = (0.0, 0.0, 0.0),
    qoffset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    qfac: float = 1.0,
    sform: np.ndarray | None = None,
    endian: str = "<",
) -> None:
    """Write a data-free but standards-shaped NIfTI-1 header for unit tests."""

    if endian not in {"<", ">"}:
        raise ValueError("endian must be '<' or '>'")
    header = bytearray(348)
    struct.pack_into(endian + "i", header, 0, 348)
    struct.pack_into(endian + "8h", header, 40, 3, *shape_xyz, 1, 1, 1, 1)
    struct.pack_into(endian + "h", header, 70, 2)  # uint8
    struct.pack_into(endian + "h", header, 72, 8)
    pixdim = (float(qfac), *map(float, spacing_xyz), 1.0, 0.0, 0.0, 0.0)
    struct.pack_into(endian + "8f", header, 76, *pixdim)
    struct.pack_into(endian + "f", header, 108, 352.0)
    struct.pack_into(endian + "2h", header, 252, int(qform_code), int(sform_code))
    struct.pack_into(endian + "3f", header, 256, *map(float, quaternion_bcd))
    struct.pack_into(endian + "3f", header, 268, *map(float, qoffset_xyz))
    if sform is not None:
        matrix = np.asarray(sform, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError("sform must be 4x4")
        for offset, row in zip((280, 296, 312), matrix[:3], strict=True):
            struct.pack_into(endian + "4f", header, offset, *map(float, row))
    header[344:348] = b"n+1\x00"
    path.write_bytes(header)


def _window(
    *,
    center_xyz: Iterable[float],
    fov_xyz: Iterable[float],
    shape_zyx: tuple[int, int, int] = (8, 8, 8),
    contract: str = "SYNTHETIC",
    view: str = "detail",
) -> PhysicalWindow:
    return PhysicalWindow(
        contract=contract,
        view=view,
        frame_basis=np.eye(3),
        center_frame_mm=np.asarray(tuple(center_xyz), dtype=np.float64),
        fov_xyz_mm=np.asarray(tuple(fov_xyz), dtype=np.float64),
        output_shape_zyx=shape_zyx,
        anchor_policy="synthetic",
        reference_visit="T0",
        causal_deployability="SYNTHETIC",
    )


def smoke_nifti_affines() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="observable_crop_nifti_") as temporary:
        root = Path(temporary)
        spacing = (1.0, 1.5, 2.0)
        theta = math.radians(30.0)
        rotation = np.asarray(
            [
                [math.cos(theta), -math.sin(theta), 0.0],
                [math.sin(theta), math.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        sform = np.eye(4)
        sform[:3, :3] = rotation @ np.diag(spacing)
        sform[:3, 3] = (11.0, -7.0, 3.0)
        primary = root / "sform_primary.nii"
        _write_nifti_header(
            primary,
            spacing_xyz=spacing,
            qform_code=1,
            sform_code=1,
            qoffset_xyz=(101.0, 102.0, 103.0),
            sform=sform,
        )
        geometry = read_nifti_geometry(primary)
        assert geometry.affine_source == "sform"
        assert geometry.sform_valid and geometry.qform_valid
        assert geometry.orientation == "RAS"
        _assert_close(geometry.selected_affine, sform, atol=2e-6)
        _assert_close(geometry.max_obliquity_deg, 30.0, atol=2e-5)
        _assert_close(abs(float(geometry.determinant)), np.prod(spacing), atol=2e-6)

        # Invalid sform spacing must fail closed to a valid qform, rather than
        # silently accepting the contradictory srow geometry.
        invalid_sform = np.eye(4)
        invalid_sform[:3, :3] = np.diag((9.0, spacing[1], spacing[2]))
        fallback = root / "qform_fallback.nii"
        _write_nifti_header(
            fallback,
            spacing_xyz=spacing,
            qform_code=1,
            sform_code=1,
            qoffset_xyz=(4.0, 5.0, 6.0),
            sform=invalid_sform,
        )
        fallback_geometry = read_nifti_geometry(fallback)
        assert fallback_geometry.affine_source == "qform_fallback"
        assert fallback_geometry.sform_failure_reason == "SFORM_SPACING_MISMATCH"
        expected_qform = np.diag((spacing[0], spacing[1], spacing[2], 1.0))
        expected_qform[:3, 3] = (4.0, 5.0, 6.0)
        _assert_close(fallback_geometry.selected_affine, expected_qform)

        # Exercise the quaternion rotation and negative qfac path independently.
        qform_only = root / "qform_only_big_endian.nii"
        half_angle = math.sqrt(0.5)
        _write_nifti_header(
            qform_only,
            spacing_xyz=(2.0, 3.0, 4.0),
            qform_code=1,
            quaternion_bcd=(0.0, 0.0, half_angle),
            qoffset_xyz=(7.0, 8.0, 9.0),
            qfac=-1.0,
            endian=">",
        )
        qgeometry = read_nifti_geometry(qform_only)
        assert qgeometry.affine_source == "qform_fallback"
        assert qgeometry.orientation == "ALI"
        assert float(qgeometry.determinant) < 0.0
        _assert_close(qgeometry.selected_affine[:3, 3], (7.0, 8.0, 9.0))

        unavailable = root / "no_affine.nii"
        _write_nifti_header(unavailable, qform_code=0, sform_code=0)
        missing = read_nifti_geometry(unavailable)
        assert missing.affine_source == "unavailable"
        assert missing.selected_affine is None
        assert missing.orientation == "UNKNOWN"

        shifted = sform.copy()
        shifted[:3, 3] += (1.0, 2.0, 2.0)
        disagreement = affine_max_corner_disagreement_mm(sform, shifted, (9, 10, 11))
        _assert_close(disagreement, 3.0)

        return {
            "sform_precedence": geometry.affine_source,
            "qform_fallback_reason": fallback_geometry.sform_failure_reason,
            "qform_orientation": qgeometry.orientation,
            "max_obliquity_deg": geometry.max_obliquity_deg,
            "corner_disagreement_mm": disagreement,
        }


def smoke_xyz_zyx_and_affine_footprints() -> dict[str, object]:
    affine = np.eye(4)
    affine[:3, :3] = np.diag((-2.0, 3.0, 4.0))
    affine[:3, 3] = (20.0, -5.0, 7.0)
    basis = orthonormal_index_basis(affine)
    _assert_close(basis.T @ basis, np.eye(3))
    assert np.linalg.det(basis) < 0.0

    low, high = bbox_footprint_in_frame(affine, basis, (1, 2, 3), (1, 2, 3))
    _assert_close(high - low, (2.0, 3.0, 4.0))

    window = _window(
        center_xyz=(0.0, 0.0, 0.0),
        fov_xyz=(18.0, 14.0, 10.0),
        shape_zyx=(5, 7, 9),
    )
    np.testing.assert_array_equal(window.output_shape_xyz, (9, 7, 5))
    _assert_close(window.effective_spacing_xyz_mm, (2.0, 2.0, 2.0))

    _expect_raises(
        ValueError,
        _window,
        center_xyz=(0.0, 0.0, 0.0),
        fov_xyz=(8.0, 8.0, 8.0),
        shape_zyx=(8, 8, 0),
    )
    _expect_raises(
        ValueError,
        PhysicalWindow,
        contract="BAD",
        view="detail",
        frame_basis=np.eye(3),
        center_frame_mm=np.zeros(3),
        fov_xyz_mm=np.ones(3),
        output_shape_zyx=(8, 8, 8.5),
        anchor_policy="bad",
        reference_visit="T0",
        causal_deployability="bad",
    )
    return {
        "source_order": "XYZ",
        "tensor_order": "ZYX",
        "shape_xyz": window.output_shape_xyz.tolist(),
        "reflected_basis_determinant": float(np.linalg.det(basis)),
    }


def smoke_containment_states() -> dict[str, object]:
    mask = np.zeros((12, 12, 12), dtype=bool)
    mask[4:7, 4:7, 4:7] = True
    affine = np.eye(4)

    full = audit_support(mask, affine, _window(center_xyz=(5, 5, 5), fov_xyz=(8, 8, 8)))
    assert full.full_support_voxels == 27
    assert full.retained_support_voxels == 27
    assert full.exact_full_support_containment
    assert full.sufficient_containment
    assert not full.boundary_touch
    _assert_close(full.retained_ftv_fraction, 1.0)
    _assert_close(full.surface_voxel_retention, 1.0)

    # The low X crop face is exactly the lesion's voxel-footprint boundary.
    touch = audit_support(
        mask,
        affine,
        _window(center_xyz=(7.5, 5, 5), fov_xyz=(8, 8, 8)),
    )
    assert touch.exact_full_support_containment
    assert touch.boundary_touch and touch.suspected_truncation
    assert not touch.sufficient_containment
    _assert_close(touch.minimum_margin_mm, 0.0)

    truncated = audit_support(
        mask,
        affine,
        _window(center_xyz=(8.5, 5, 5), fov_xyz=(8, 8, 8)),
    )
    _assert_close(truncated.retained_ftv_fraction, 2.0 / 3.0)
    assert truncated.boundary_touch
    assert truncated.suspected_truncation and truncated.severe_truncation
    assert not truncated.exact_full_support_containment
    assert truncated.cut_component_count == 1
    _assert_close(truncated.extent_retention_x, 2.0 / 3.0)

    return {
        "full_retention": full.retained_ftv_fraction,
        "touch_minimum_margin_mm": touch.minimum_margin_mm,
        "truncated_retention": truncated.retained_ftv_fraction,
        "truncated_surface_retention": truncated.surface_voxel_retention,
    }


def smoke_padding_and_expansion_semantics() -> dict[str, object]:
    expanded = make_fixed_expand_window(
        contract="C1A",
        view="detail",
        frame_basis=np.eye(3),
        support_low_frame_mm=(0.0, 0.0, 0.0),
        support_high_frame_mm=(12.0, 4.0, 4.0),
        nominal_fov_xyz_mm=(8.0, 8.0, 8.0),
        output_shape_zyx=(8, 8, 8),
        margin_mm=1.0,
        anchor_policy="CURRENT_VISIT_SUPPORT",
        reference_visit="T1",
        causal_deployability="CURRENT_VISIT_CAUSAL",
    )
    assert expanded.expanded_from_nominal
    _assert_close(expanded.fov_xyz_mm, (14.0, 8.0, 8.0))
    _assert_close(expanded.effective_spacing_xyz_mm, (1.75, 1.0, 1.0))

    no_expand = make_fixed_expand_window(
        contract="SENSITIVITY",
        view="detail",
        frame_basis=np.eye(3),
        support_low_frame_mm=(0.0, 0.0, 0.0),
        support_high_frame_mm=(12.0, 4.0, 4.0),
        nominal_fov_xyz_mm=(8.0, 8.0, 8.0),
        output_shape_zyx=(8, 8, 8),
        margin_mm=1.0,
        anchor_policy="SYNTHETIC",
        reference_visit="T1",
        causal_deployability="SYNTHETIC",
        allow_expand=False,
    )
    mask = np.zeros((16, 8, 8), dtype=bool)
    mask[0:12, 2:6, 2:6] = True
    truncated = audit_support(mask, np.eye(4), no_expand)
    assert truncated.suspected_truncation
    assert truncated.retained_ftv_fraction < 1.0

    tight = make_tight_resize_window(
        contract="BBOX_RESIZE_SENSITIVITY",
        view="detail",
        frame_basis=np.eye(3),
        support_low_frame_mm=(0.0, 0.0, 0.0),
        support_high_frame_mm=(12.0, 4.0, 4.0),
        output_shape_zyx=(8, 8, 8),
        margin_mm=1.0,
        reference_visit="T1",
    )
    assert tight.direct_bbox_resize
    _assert_close(tight.fov_xyz_mm, (14.0, 6.0, 6.0))
    _expect_raises(
        ValueError,
        make_tight_resize_window,
        contract="BAD",
        view="detail",
        frame_basis=np.eye(3),
        support_low_frame_mm=(0.0, 0.0, 0.0),
        support_high_frame_mm=(1.0, 1.0, 1.0),
        output_shape_zyx=(8, 8, 8),
        margin_mm=-1.0,
        reference_visit="T0",
    )
    return {
        "expanded_fov_xyz_mm": expanded.fov_xyz_mm.tolist(),
        "expanded_scale_xyz": expanded.effective_spacing_xyz_mm.tolist(),
        "no_expand_retention": truncated.retained_ftv_fraction,
        "tight_resize_flag": tight.direct_bbox_resize,
    }


def smoke_temporal_anchor_and_union() -> dict[str, object]:
    basis = np.eye(3)
    t0_bounds = (np.asarray((0.0, 0.0, 0.0)), np.asarray((4.0, 4.0, 4.0)))
    t1_bounds = (np.asarray((8.0, -1.0, 1.0)), np.asarray((10.0, 5.0, 3.0)))

    adaptive_t0 = make_fixed_expand_window(
        contract="C1A",
        view="detail",
        frame_basis=basis,
        support_low_frame_mm=t0_bounds[0],
        support_high_frame_mm=t0_bounds[1],
        nominal_fov_xyz_mm=(6.0, 6.0, 6.0),
        output_shape_zyx=(6, 6, 6),
        margin_mm=1.0,
        anchor_policy="CURRENT_VISIT_SUPPORT",
        reference_visit="T0",
        causal_deployability="CURRENT_VISIT_CAUSAL",
    )
    adaptive_t1 = make_fixed_expand_window(
        contract="C1A",
        view="detail",
        frame_basis=basis,
        support_low_frame_mm=t1_bounds[0],
        support_high_frame_mm=t1_bounds[1],
        nominal_fov_xyz_mm=(6.0, 6.0, 6.0),
        output_shape_zyx=(6, 6, 6),
        margin_mm=1.0,
        anchor_policy="CURRENT_VISIT_SUPPORT",
        reference_visit="T1",
        causal_deployability="CURRENT_VISIT_CAUSAL",
    )
    assert not np.allclose(adaptive_t0.center_frame_mm, adaptive_t1.center_frame_mm)

    t0_center = 0.5 * (t0_bounds[0] + t0_bounds[1])
    anchored_t0 = make_fixed_expand_window(
        contract="C1B",
        view="detail",
        frame_basis=basis,
        support_low_frame_mm=t0_bounds[0],
        support_high_frame_mm=t0_bounds[1],
        nominal_fov_xyz_mm=(6.0, 6.0, 6.0),
        output_shape_zyx=(6, 6, 6),
        margin_mm=1.0,
        anchor_policy="T0_SUPPORT",
        reference_visit="T0",
        causal_deployability="T0_CAUSAL",
        center_frame_mm=t0_center,
    )
    anchored_t1 = make_fixed_expand_window(
        contract="C1B",
        view="detail",
        frame_basis=basis,
        support_low_frame_mm=t0_bounds[0],
        support_high_frame_mm=t0_bounds[1],
        nominal_fov_xyz_mm=(6.0, 6.0, 6.0),
        output_shape_zyx=(6, 6, 6),
        margin_mm=1.0,
        anchor_policy="T0_SUPPORT",
        reference_visit="T0",
        causal_deployability="T0_CAUSAL",
        center_frame_mm=t0_center,
    )
    _assert_close(anchored_t0.center_frame_mm, anchored_t1.center_frame_mm)
    _assert_close(anchored_t0.low_frame_mm, anchored_t1.low_frame_mm)
    _assert_close(anchored_t0.high_frame_mm, anchored_t1.high_frame_mm)

    union = make_union_window(
        contract="C1C",
        view="detail",
        frame_basis=basis,
        visit_bounds_frame_mm=(t0_bounds, t1_bounds),
        nominal_fov_xyz_mm=(6.0, 6.0, 6.0),
        output_shape_zyx=(6, 8, 12),
        margin_mm=1.0,
    )
    assert union.audit_only
    assert union.causal_deployability == "AUDIT_ONLY_FUTURE_INFORMATION"
    _assert_close(union.center_frame_mm, (5.0, 2.0, 2.0))
    _assert_close(union.fov_xyz_mm, (12.0, 8.0, 6.0))
    assert np.all(union.low_frame_mm <= t0_bounds[0] - 1.0 + 1e-9)
    assert np.all(union.high_frame_mm >= t1_bounds[1] + 1.0 - 1e-9)

    # A translated current support is fully visible adaptively but entirely
    # missed by the unchanged T0 physical window.
    current_mask = np.zeros((4, 4, 4), dtype=bool)
    current_mask[1:3, 1:3, 1:3] = True
    translated_affine = np.eye(4)
    translated_affine[0, 3] = 8.0
    adaptive_current = _window(center_xyz=(9.5, 1.5, 1.5), fov_xyz=(6, 6, 6), shape_zyx=(6, 6, 6))
    adaptive_audit = audit_support(current_mask, translated_affine, adaptive_current)
    anchored_audit = audit_support(current_mask, translated_affine, anchored_t1)
    assert adaptive_audit.exact_full_support_containment
    assert anchored_audit.retained_support_voxels == 0
    assert anchored_audit.severe_truncation

    return {
        "visit_adaptive_center_drift_mm": float(
            np.linalg.norm(adaptive_t1.center_frame_mm - adaptive_t0.center_frame_mm)
        ),
        "t0_anchor_center_drift_mm": float(
            np.linalg.norm(anchored_t1.center_frame_mm - anchored_t0.center_frame_mm)
        ),
        "union_fov_xyz_mm": union.fov_xyz_mm.tolist(),
        "anchored_translated_retention": anchored_audit.retained_ftv_fraction,
    }


def smoke_surface_and_components() -> dict[str, object]:
    mask = np.zeros((20, 20, 20), dtype=bool)
    mask[2:6, 4:7, 4:7] = True
    mask[14:17, 4:7, 4:7] = True
    window = _window(
        center_xyz=(8.0, 9.5, 9.5),
        fov_xyz=(9.0, 20.0, 20.0),
        shape_zyx=(20, 20, 9),
    )
    audit = audit_support(mask, np.eye(4), window)
    assert audit.component_count == 2
    assert audit.cut_component_count == 1
    assert audit.missed_component_count == 1
    assert 0 < audit.retained_surface_voxels < audit.surface_voxels
    assert 0.0 < audit.surface_voxel_retention < 1.0
    assert audit.extent_retention_x < 1.0
    assert audit.context_to_lesion_volume_ratio > 0.0

    # The audit contract deliberately uses 26-connectivity: diagonal voxels
    # form one component.
    diagonal = np.zeros((5, 5, 5), dtype=bool)
    diagonal[1, 1, 1] = True
    diagonal[2, 2, 2] = True
    diagonal_audit = audit_support(
        diagonal,
        np.eye(4),
        _window(center_xyz=(2, 2, 2), fov_xyz=(5, 5, 5), shape_zyx=(5, 5, 5)),
    )
    assert diagonal_audit.component_count == 1

    return {
        "component_count": audit.component_count,
        "cut_component_count": audit.cut_component_count,
        "missed_component_count": audit.missed_component_count,
        "surface_retention": audit.surface_voxel_retention,
        "diagonal_26_connected_components": diagonal_audit.component_count,
    }


def smoke_c2_nesting() -> dict[str, object]:
    common = dict(
        frame_basis=np.eye(3),
        support_low_frame_mm=(0.0, 0.0, 0.0),
        support_high_frame_mm=(4.0, 4.0, 4.0),
        output_shape_zyx=(14, 12, 10),
        margin_mm=1.0,
        anchor_policy="T0_SUPPORT",
        reference_visit="T0",
        causal_deployability="T0_CAUSAL",
        center_frame_mm=(2.0, 2.0, 2.0),
    )
    detail = make_fixed_expand_window(
        contract="C2B",
        view="detail",
        nominal_fov_xyz_mm=(10.0, 12.0, 14.0),
        **common,
    )
    context = make_fixed_expand_window(
        contract="C2B",
        view="context",
        nominal_fov_xyz_mm=(20.0, 24.0, 28.0),
        **common,
    )
    _assert_close(detail.center_frame_mm, context.center_frame_mm)
    assert np.all(context.low_frame_mm <= detail.low_frame_mm)
    assert np.all(context.high_frame_mm >= detail.high_frame_mm)
    _assert_close(detail.effective_spacing_xyz_mm, (1.0, 1.0, 1.0))
    _assert_close(context.effective_spacing_xyz_mm, (2.0, 2.0, 2.0))
    return {
        "same_center": True,
        "detail_fov_xyz_mm": detail.fov_xyz_mm.tolist(),
        "context_fov_xyz_mm": context.fov_xyz_mm.tolist(),
        "context_contains_detail": True,
    }


def main() -> None:
    checks = {
        "nifti_affines": smoke_nifti_affines(),
        "xyz_zyx_affine_footprints": smoke_xyz_zyx_and_affine_footprints(),
        "containment_states": smoke_containment_states(),
        "padding_expansion": smoke_padding_and_expansion_semantics(),
        "temporal_anchor_union": smoke_temporal_anchor_and_union(),
        "surface_components": smoke_surface_and_components(),
        "c2_nesting": smoke_c2_nesting(),
    }
    print(
        json.dumps(
            {"status": "PASS", "synthetic_only": True, "checks": checks},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

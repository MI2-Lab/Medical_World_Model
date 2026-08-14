from __future__ import annotations

import inspect

import pytest
import torch

from patch_token_wm.contracts import (
    AUDITED_POOLING_SHA256,
    G3_MODEL_PATH,
    G3_MODEL_SHA256,
    SpatialVisitEncoder3D,
    file_sha256,
)
from patch_token_wm.geometry import (
    build_local_token_geometry,
    derived_feature_shape,
    feature_cell_coordinates_xyz_mm,
    sinusoidal_physical_position_encoding,
)


def test_exact_hash_locked_g3_encoder_and_audited_pooling() -> None:
    assert file_sha256(G3_MODEL_PATH) == G3_MODEL_SHA256
    assert inspect.getfile(SpatialVisitEncoder3D) == str(G3_MODEL_PATH)
    # The digest value is part of the public source contract, not a mutable
    # digest computed from whatever implementation happens to be importable.
    assert AUDITED_POOLING_SHA256 == (
        "630a717a98a7e80d69d3a462dd3086c2de81449c91910312cbc0bfce0fd58d54"
    )


def test_formal_positive_overlap_support_is_exactly_500_audited_cells() -> None:
    geometry = build_local_token_geometry()
    assert derived_feature_shape() == (14, 22, 20)
    assert geometry.feature_shape_zyx == (14, 22, 20)
    assert geometry.dense_weights.shape == (1, 1, 14, 22, 20)
    assert geometry.token_count == 500
    assert torch.all(geometry.weights > 0)
    assert int(((geometry.weights > 0) & (geometry.weights < 1)).sum()) == 308
    assert float(geometry.weights.sum()) == pytest.approx(316.0493815, abs=3e-5)
    assert torch.equal(
        geometry.weights,
        geometry.dense_weights.reshape(-1).index_select(0, geometry.flat_indices),
    )


def test_physical_coordinates_are_actual_crop_centered_xyz_cell_centers() -> None:
    geometry = build_local_token_geometry()
    dense = feature_cell_coordinates_xyz_mm(
        geometry.input_shape_zyx,
        geometry.feature_shape_zyx,
        geometry.spacing_xyz_mm,
    )
    assert torch.equal(
        geometry.coordinates_xyz_mm,
        dense.reshape(-1, 3).index_select(0, geometry.flat_indices),
    )
    # Tensor cell (z=7,y=11,x=10) is physically (x=.45,y=.45,z=1) mm:
    # ((j*8) - (input_size-1)/2) * spacing, returned in XYZ order.
    flat_index = 7 * 22 * 20 + 11 * 20 + 10
    selected_position = int(torch.nonzero(geometry.flat_indices == flat_index).item())
    torch.testing.assert_close(
        geometry.coordinates_xyz_mm[selected_position],
        torch.tensor((0.45, 0.45, 1.0)),
        rtol=0,
        atol=1e-5,
    )
    torch.testing.assert_close(
        geometry.coordinates_xyz_mm.amin(dim=0),
        torch.tensor((-35.55, -35.55, -31.0)),
        rtol=0,
        atol=1e-4,
    )
    torch.testing.assert_close(
        geometry.coordinates_xyz_mm.amax(dim=0),
        torch.tensor((29.25, 29.25, 33.0)),
        rtol=0,
        atol=1e-4,
    )


def test_position_encoding_is_deterministic_and_geometry_dependent() -> None:
    geometry = build_local_token_geometry()
    first = sinusoidal_physical_position_encoding(geometry.coordinates_xyz_mm)
    second = sinusoidal_physical_position_encoding(geometry.coordinates_xyz_mm)
    shifted = sinusoidal_physical_position_encoding(
        geometry.coordinates_xyz_mm + torch.tensor((1.0, 0.0, 0.0))
    )
    assert first.shape == (500, 128)
    assert torch.equal(first, second)
    assert not torch.equal(first, shifted)
    assert not first.requires_grad


def test_small_synthetic_geometry_is_explicitly_nonformal() -> None:
    geometry = build_local_token_geometry(
        (16, 16, 16), (4.0, 4.0, 4.0), require_formal_count=False
    )
    assert geometry.feature_shape_zyx == (2, 2, 2)
    assert geometry.token_count == 8
    with pytest.raises(ValueError, match="exactly 500"):
        build_local_token_geometry(
            (16, 16, 16), (4.0, 4.0, 4.0), require_formal_count=True
        )

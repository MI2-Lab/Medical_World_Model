from __future__ import annotations

import torch

from foundation_mri.spatial import (
    DINO_IMAGE_SIZE,
    dino_slice_stack,
    fixed_physical_center_crop,
    medicalnet_volume_batch,
    spatial_contract,
)


def test_dino_global_and_local_shapes_are_fixed() -> None:
    visit = torch.zeros(7, 112, 176, 160, dtype=torch.float32)
    for axis in ("GLOBAL", "LOCAL"):
        stack = dino_slice_stack(visit, axis)
        assert stack.shape == (32, 3, DINO_IMAGE_SIZE, DINO_IMAGE_SIZE)
        assert stack.dtype == torch.float32
        assert torch.isfinite(stack).all()


def test_physical_crop_is_central_and_mask_free() -> None:
    z = torch.linspace(-1.0, 1.0, 112).view(1, 112, 1, 1)
    visit = z.expand(7, 112, 176, 160).contiguous()
    local = fixed_physical_center_crop(visit, target_shape_zyx=(32, 72, 72))
    assert local.shape == (7, 32, 72, 72)
    assert abs(float(local.mean())) < 1e-6
    assert torch.allclose(local[:, 0], -local[:, -1], atol=1e-6, rtol=0.0)


def test_medicalnet_preserves_native_c1b_grid() -> None:
    visit = torch.zeros(7, 112, 176, 160, dtype=torch.float32)
    batch = medicalnet_volume_batch(visit)
    assert batch.shape == (7, 1, 112, 176, 160)
    assert batch.data_ptr() == visit.data_ptr()
    contract = spatial_contract()
    assert contract["local_uses_lesion_mask"] is False
    assert contract["medicalnet_feature_shape_zyx"] == [14, 22, 20]


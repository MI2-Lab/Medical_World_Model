from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(
    0,
    str(
        ROOT.parents[1]
        / "additional_experiments"
        / "c1b_spatial_pooling_bottleneck_audit"
        / "src"
    ),
)

import build_oracle_sidecars as oracle_builder  # noqa: E402
from build_oracle_sidecars import (  # noqa: E402
    _authenticated_cache_index,
    _validate_visit_authorization,
    fixed_local_voxel_centers,
    physical_region_masks,
)


@pytest.mark.parametrize(
    ("arguments", "exit_code"),
    [(["--help"], 0), (["--unexpected"], 2)],
)
def test_oracle_cli_parses_before_any_formal_action(
    arguments: list[str], exit_code: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("formal action ran before argument parsing")

    monkeypatch.setattr(oracle_builder, "load_config", forbidden)
    monkeypatch.setattr(oracle_builder, "require_preregistration_lock", forbidden)
    monkeypatch.setattr(oracle_builder, "require_cache_integrity", forbidden)
    with pytest.raises(SystemExit) as stopped:
        oracle_builder.main(arguments)
    assert stopped.value.code == exit_code


def test_authenticated_cache_index_filters_exact_primary_808_from_947() -> None:
    records = [
        {
            "patient_id": f"P{index:04d}",
            "path": f"/cache/P{index:04d}.npz",
            "sha256": "a" * 64,
            "size_bytes": 1,
            "mtime_ns": 1,
            "cohort": "primary",
        }
        for index in range(808)
    ]
    records.extend(
        {
            "patient_id": f"X{index:04d}",
            "path": f"/cache/X{index:04d}.npz",
            "sha256": "b" * 64,
            "size_bytes": 1,
            "mtime_ns": 1,
            "cohort": "train_only",
        }
        for index in range(139)
    )
    index = _authenticated_cache_index(
        {
            "patient_count": 947,
            "primary_patient_count": 808,
            "train_only_patient_count": 139,
            "records": records,
        }
    )
    assert len(index) == 808
    assert set(index) == {f"P{patient_index:04d}" for patient_index in range(808)}


def test_physical_rings_use_anisotropic_millimetres_and_are_disjoint() -> None:
    lesion = np.zeros((21, 41, 41), dtype=bool)
    lesion[10, 20, 20] = True
    valid = np.ones_like(lesion)
    regions = physical_region_masks(
        lesion,
        valid,
        spacing_zyx_mm=(2.0, 1.0, 1.0),
        local_cube=np.ones_like(lesion),
    )
    assert regions["CORE"][10, 20, 20]
    assert regions["PERI10"][15, 20, 20]  # 10 mm through-plane
    assert not regions["PERI10"][16, 20, 20]  # 12 mm
    assert regions["PERI20"][20, 20, 20]  # 20 mm
    stack = np.stack(list(regions.values()))
    assert np.all(stack.sum(axis=0) <= 1)


def test_local_rest_is_limited_to_cube_and_valid_source() -> None:
    lesion = np.zeros((9, 9, 9), dtype=bool)
    lesion[4, 4, 4] = True
    valid = np.ones_like(lesion)
    valid[0] = False
    cube = np.zeros_like(lesion)
    cube[2:7, 2:7, 2:7] = True
    regions = physical_region_masks(
        lesion,
        valid,
        spacing_zyx_mm=(10.0, 10.0, 10.0),
        local_cube=cube,
    )
    assert np.all(regions["LOCAL_REST"] <= cube)
    assert np.all(regions["LOCAL_REST"] <= valid)


def test_core_preserves_source_lesion_outside_image_valid_support() -> None:
    lesion = np.zeros((3, 3, 3), dtype=bool)
    lesion[0, 0, 0] = True
    valid = np.ones_like(lesion)
    valid[0, 0, 0] = False
    regions = physical_region_masks(lesion, valid)
    assert regions["CORE"][0, 0, 0]
    assert not regions["PERI10"][0, 0, 0]
    assert not regions["PERI20"][0, 0, 0]
    assert not regions["LOCAL_REST"][0, 0, 0]


def test_empty_lesion_yields_empty_core_and_peri_without_prefilter() -> None:
    lesion = np.zeros((5, 5, 5), dtype=bool)
    valid = np.ones_like(lesion)
    regions = physical_region_masks(
        lesion,
        valid,
        local_cube=np.ones_like(lesion),
    )
    assert not regions["CORE"].any()
    assert not regions["PERI10"].any()
    assert not regions["PERI20"].any()
    assert regions["LOCAL_REST"].all()


def test_visit_authority_is_808_patient_local_and_parity_is_a_subset() -> None:
    available = np.zeros((808, 4), dtype=bool)
    available[:, 0] = True
    available[:375, 1:] = True
    parity = np.zeros_like(available)
    parity[:375] = True
    observed_available, observed_parity = _validate_visit_authorization(
        available, parity
    )
    assert observed_available.shape == (808, 4)
    assert np.array_equal(observed_available.sum(axis=0), [808, 375, 375, 375])
    assert int(observed_available.sum()) == 1933
    assert int(observed_parity.sum()) == 1500


def test_batched_region_rf_mapping_is_bitwise_equal_to_separate_mapping() -> None:
    torch = pytest.importorskip("torch")
    from c1b_spatial_audit.pooling import receptive_field_occupancy

    masks = np.zeros((2, 1, 112, 176, 160), dtype=np.float32)
    masks[0, 0, 50:53, 80:84, 70:75] = 1
    masks[1, 0, 60:64, 90:94, 82:86] = 1
    batched = receptive_field_occupancy(
        torch.from_numpy(masks), (14, 22, 20), stage="final"
    )
    separate = torch.cat(
        [
            receptive_field_occupancy(
                torch.from_numpy(masks[index : index + 1]),
                (14, 22, 20),
                stage="final",
            )
            for index in range(2)
        ],
        dim=0,
    )
    assert torch.equal(batched, separate)


def test_formal_local_voxel_cube_is_centered_and_nonempty() -> None:
    cube = fixed_local_voxel_centers()
    assert cube.shape == (112, 176, 160)
    assert cube.any()
    assert np.array_equal(cube, cube[::-1, ::-1, ::-1])

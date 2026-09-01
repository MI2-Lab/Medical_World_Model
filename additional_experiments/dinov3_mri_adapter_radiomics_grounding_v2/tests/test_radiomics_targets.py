from pathlib import Path

import numpy as np

import dinov3_rg.targets as target_module
from dinov3_rg.data import FoldTargets
from dinov3_rg.radiomics import local_bounds, morphology_variants
from dinov3_rg.targets import RawRadiomics, build_fold_targets, select_stable_features


def test_local_bounds_and_in_plane_morphology():
    assert local_bounds() == (slice(40, 72), slice(52, 124), slice(44, 116))
    mask = np.zeros((32, 72, 72), dtype=bool)
    mask[10, 30:35, 30:35] = True
    original, eroded, dilated = morphology_variants(mask)
    assert original.sum() == 25
    assert eroded.sum() == 9
    assert dilated.sum() == 49
    assert not dilated[9].any() and not dilated[11].any()
    valid_source = np.ones_like(mask)
    valid_source[10, 29, 30:35] = False
    _, _, bounded = morphology_variants(mask, valid_source)
    assert not (bounded & ~valid_source).any()


def synthetic_raw() -> RawRadiomics:
    generator = np.random.default_rng(7)
    n, features = 20, 20
    patient_ids = tuple(f"P{index:03d}" for index in range(n))
    base = generator.normal(size=(n, 4, features)).astype(np.float32)
    values = np.stack((base, base * 0.99 + 0.01, base * 1.01 - 0.01), axis=2)
    ftv = generator.uniform(0.1, 20.0, size=(n, 4)).astype(np.float32)
    volume = generator.uniform(100, 5000, size=(n, 4)).astype(np.float32)
    valid = np.ones((n, 4), dtype=bool)
    return RawRadiomics(
        patient_ids=patient_ids,
        feature_names=tuple(f"feature_{index:02d}" for index in range(features)),
        values=values,
        ftv=ftv,
        ftv_mask=valid,
        local_volume_mm3=volume,
        roi_mask=valid,
        variant_mask=np.ones((n, 4, 3), dtype=bool),
        source_hashes=tuple("a" * 64 for _ in range(n * 2)),
    )


def test_selection_and_fold_transform_are_train_only(tmp_path, monkeypatch):
    raw = synthetic_raw()
    train = raw.patient_ids[:15]
    selected, audit = select_stable_features(raw, train)
    assert len(selected) == 20
    assert all(row["passed"] for row in audit)
    monkeypatch.setattr(target_module, "EXPERIMENT_ROOT", tmp_path)
    summary = build_fold_targets(raw, 0, train, tmp_path / "private")
    assert summary["status"] == "PASS"
    targets = FoldTargets.load(tmp_path / "private/fold_0_targets.private.npz")
    assert targets.radiomics.shape == (20, 4, 16)
    assert targets.radiomics_mask[:, :3].all()
    assert not targets.radiomics_mask[:, 3].any()
    assert (tmp_path / "private/fold_0_transform.private.json").is_file()

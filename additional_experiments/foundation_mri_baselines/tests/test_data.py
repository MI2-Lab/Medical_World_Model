from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foundation_mri.data import (  # noqa: E402
    FOLDS,
    RADIOMIC_FEATURES,
    file_sha256,
    load_clinical_labels,
    load_current_cnn_features,
    load_fold_manifest,
    load_foundation_features,
    load_radiomics_table,
    ordered_text_sha256,
)


def _patient_ids(n: int = 30) -> np.ndarray:
    return np.asarray([f"P{i:03d}" for i in range(n)], dtype=str)


def _write_folds(path: Path, patient_ids: np.ndarray) -> np.ndarray:
    rows = []
    labels = np.arange(len(patient_ids)) % 2
    for fold in FOLDS:
        for index, patient_id in enumerate(patient_ids):
            if index % 5 == fold:
                split = "test"
            elif (index + 1) % 5 == fold:
                split = "val"
            else:
                split = "train"
            rows.append(
                {
                    "patient_id": patient_id,
                    "fold": fold,
                    "split": split,
                    "label_pcr": int(labels[index]),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    return labels.astype(np.int64)


def _write_clinical(path: Path, patient_ids: np.ndarray, labels: np.ndarray) -> None:
    n = len(patient_ids)
    age = 30.0 + np.arange(n, dtype=np.float64)
    age[3] = np.nan
    hr = np.arange(n) % 2
    her2 = (np.arange(n) // 2) % 2
    subtype = np.asarray(
        [
            f"HR{'+' if hr_value else '-'}/HER2{'+' if her2_value else '-'}"
            for hr_value, her2_value in zip(hr, her2, strict=True)
        ]
    )
    pd.DataFrame(
        {
            "patient_id": patient_ids[::-1],
            "label_pcr": labels[::-1],
            "label_hr": hr[::-1],
            "label_her2": her2[::-1],
            "label_mp": ((np.arange(n) // 3) % 2)[::-1],
            "age_at_screening": age[::-1],
            "arm": np.asarray([f"arm_{index % 3}" for index in range(n)])[::-1],
            "hr_her2_subtype": subtype[::-1],
            "complete_4visits": True,
        }
    ).to_csv(path, index=False)


def _write_current_cnn_metadata(
    feature_path: Path,
    *,
    patient_ids: np.ndarray,
    arm: str,
    fold: int,
    feature_shape: tuple[int, ...],
) -> Path:
    implementation_sha = "1" * 64
    current_provenance_sha = "2" * 64
    checkpoint_provenance_sha = "3" * 64
    stage_a_sha = "4" * 64
    train_sha = "5" * 64
    validation_sha = "6" * 64
    lock_path = feature_path.parents[0] / "PREREGISTRATION_LOCK.json"
    lock = {
        "schema_version": 1,
        "experiment": "local_global_response_state_pilot",
        "status": "FROZEN_BEFORE_FORMAL_RESULTS",
        "matrix": {"arms": [arm], "seeds": [2026], "folds": list(FOLDS)},
        "upstream_sha256": {"stage_a": stage_a_sha},
        "code_and_plan_sha256": {"features.py": implementation_sha},
    }
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    lock_sha = file_sha256(lock_path)
    checkpoint_path = feature_path.with_suffix(".checkpoint.bin")
    checkpoint_path.write_bytes(b"synthetic-checkpoint")
    selection_path = feature_path.with_suffix(".selection.json")
    selection = {
        "schema_version": 1,
        "arm": arm,
        "seed_base": 2026,
        "fold": fold,
        "selected_epoch": 3,
        "data_provenance_sha256": checkpoint_provenance_sha,
        "preregistration_lock_sha256": lock_sha,
        "preregistration": {"lock_sha256": lock_sha, "status": "PASS"},
        "stage_a_sentinel_sha256": stage_a_sha,
        "train_patient_sha256": train_sha,
        "val_patient_sha256": validation_sha,
        "test_data_used": False,
        "pcr_used": False,
        "delta_ftv_used": False,
    }
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    metadata = {
        "arm": arm,
        "checkpoint_data_provenance_sha256": checkpoint_provenance_sha,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "cohort": "exact_locked_primary_train_validation_test",
        "current_data_contract_provenance_sha256": current_provenance_sha,
        "experiment": "local_global_response_state_pilot",
        "feature_dtype": "float32",
        "feature_implementation_sha256": implementation_sha,
        "feature_path": str(feature_path.resolve()),
        "feature_sha256": file_sha256(feature_path),
        "feature_shape": list(feature_shape),
        "feature_tensor": "online_preprojector_response_state",
        "fold": fold,
        "ftv_head_called": False,
        "patient_order_sha256": hashlib.sha256(
            "\n".join(patient_ids.astype(str)).encode("utf-8")
        ).hexdigest(),
        "preregistration_lock": "PREREGISTRATION_LOCK.json",
        "preregistration_lock_sha256": lock_sha,
        "schema_version": 1,
        "seed_base": 2026,
        "selected_epoch": 3,
        "selection_path": str(selection_path.resolve()),
        "selection_sha256": file_sha256(selection_path),
        "stage_a_sentinel_sha256": stage_a_sha,
        "test_labels_used": False,
        "train_patient_sha256": train_sha,
        "validation_patient_sha256": validation_sha,
    }
    metadata_path = feature_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return metadata_path


def test_locked_tables_and_unified_foundation_asset_align_without_deletion(
    tmp_path: Path,
) -> None:
    ids = _patient_ids()
    fold_path = tmp_path / "folds.csv"
    clinical_path = tmp_path / "clinical.csv"
    labels = _write_folds(fold_path, ids)
    _write_clinical(clinical_path, ids, labels)
    folds = load_fold_manifest(fold_path, expected_n=len(ids), expected_sha256=None)
    clinical = load_clinical_labels(
        clinical_path,
        expected_patient_ids=folds.patient_ids,
        expected_n=len(ids),
        expected_sha256=None,
    )
    assert np.array_equal(clinical.patient_ids, ids)
    assert np.array_equal(clinical.pcr, labels)
    assert len(clinical.patient_ids) == len(ids)
    assert np.isnan(clinical.age).sum() == 1
    for fold in FOLDS:
        assert set(folds.roles(fold)) == {"train", "val", "test"}

    rng = np.random.default_rng(2026)
    representation = rng.normal(size=(len(ids), 4, 2, 7)).astype(np.float32)
    feature_path = tmp_path / "features.private.npz"
    np.savez_compressed(
        feature_path,
        patient_id=ids[::-1],
        representation=representation[::-1],
        spatial_axis=np.asarray(["GLOBAL", "LOCAL"]),
        visits=np.asarray(["T0", "T1", "T2", "T3"]),
        model_name=np.asarray("synthetic_encoder"),
        checkpoint_sha256=np.asarray("a" * 64),
    )
    asset = load_foundation_features(
        feature_path, expected_patient_ids=ids, expected_n=len(ids)
    )
    assert asset.representation.shape == (len(ids), 4, 2, 7)
    assert asset.spatial("LOCAL").shape == (len(ids), 4, 7)
    assert np.allclose(asset.representation, representation)


def test_foundation_asset_fails_closed_on_schema_patient_and_numeric_drift(
    tmp_path: Path,
) -> None:
    ids = _patient_ids()
    base = {
        "patient_id": ids,
        "representation": np.zeros((len(ids), 4, 2, 3), dtype=np.float32),
        "spatial_axis": np.asarray(["GLOBAL", "LOCAL"]),
        "visits": np.asarray(["T0", "T1", "T2", "T3"]),
        "model_name": np.asarray("encoder"),
    }
    locked = tmp_path / "locked.npz"
    np.savez(
        locked,
        **base,
        checkpoint_sha256=np.asarray("a" * 64),
        extraction_signature_sha256=np.asarray("b" * 64),
        canonical_patient_order_sha256=np.asarray(ordered_text_sha256(ids)),
    )
    asset = load_foundation_features(
        locked, expected_patient_ids=ids, expected_n=len(ids)
    )
    assert asset.checkpoint_sha256 == "a" * 64
    assert asset.extraction_signature_sha256 == "b" * 64
    assert asset.canonical_patient_order_sha256 == ordered_text_sha256(ids)
    extra = tmp_path / "extra.npz"
    np.savez(extra, **base, unexpected=np.asarray(1))
    with pytest.raises(ValueError, match="schema drifted"):
        load_foundation_features(extra, expected_patient_ids=ids, expected_n=len(ids))

    uppercase_digest = tmp_path / "uppercase_digest.npz"
    np.savez(uppercase_digest, **base, checkpoint_sha256=np.asarray("A" * 64))
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        load_foundation_features(
            uppercase_digest, expected_patient_ids=ids, expected_n=len(ids)
        )

    wrong_locked_dimension = tmp_path / "wrong_locked_dimension.npz"
    locked_model = dict(base)
    locked_model["model_name"] = np.asarray("dino_vitb16_imagenet1k")
    np.savez(wrong_locked_dimension, **locked_model)
    with pytest.raises(ValueError, match="D must be 1536"):
        load_foundation_features(
            wrong_locked_dimension, expected_patient_ids=ids, expected_n=len(ids)
        )

    missing = tmp_path / "missing.npz"
    dropped = dict(base)
    dropped["patient_id"] = ids[:-1]
    dropped["representation"] = base["representation"][:-1]
    np.savez(missing, **dropped)
    with pytest.raises(ValueError, match="exactly 30"):
        load_foundation_features(missing, expected_patient_ids=ids, expected_n=len(ids))

    nonfinite = tmp_path / "nonfinite.npz"
    corrupted = dict(base)
    corrupted["representation"] = base["representation"].copy()
    corrupted["representation"][0, 0, 0, 0] = np.nan
    np.savez(nonfinite, **corrupted)
    with pytest.raises(FloatingPointError, match="NaN/Inf"):
        load_foundation_features(nonfinite, expected_patient_ids=ids, expected_n=len(ids))


def test_current_cnn_fold_schema_and_embedded_split_are_verified(tmp_path: Path) -> None:
    ids = _patient_ids()
    fold_path = tmp_path / "folds.csv"
    labels = _write_folds(fold_path, ids)
    folds = load_fold_manifest(fold_path, expected_n=len(ids), expected_sha256=None)
    state = np.arange(len(ids) * 4 * 192, dtype=np.float32).reshape(len(ids), 4, 192)
    path = tmp_path / "cnn.private.npz"
    np.savez_compressed(
        path,
        patient_id=ids[::-1],
        split=folds.roles(2, ids)[::-1],
        response_state=state[::-1],
        arm=np.asarray("GAP0"),
        seed_base=np.asarray(2026),
        fold=np.asarray(2),
    )
    metadata_path = _write_current_cnn_metadata(
        path,
        patient_ids=ids[::-1],
        arm="GAP0",
        fold=2,
        feature_shape=state.shape,
    )
    asset = load_current_cnn_features(
        path,
        fold=2,
        expected_patient_ids=ids,
        fold_manifest=folds,
        expected_labels=labels,
        expected_n=len(ids),
        model_name="GAP0",
        spatial_axis="GLOBAL",
    )
    assert np.array_equal(asset.representation, state)
    assert asset.metadata_sha256 == file_sha256(metadata_path)
    assert asset.checkpoint_sha256 == json.loads(
        metadata_path.read_text(encoding="utf-8")
    )["checkpoint_sha256"]
    with pytest.raises(ValueError, match="embedded arm"):
        load_current_cnn_features(
            path,
            fold=2,
            expected_patient_ids=ids,
            fold_manifest=folds,
            expected_labels=labels,
            expected_n=len(ids),
            model_name="LOCAL0",
            spatial_axis="LOCAL",
        )
    with pytest.raises(ValueError, match="must use spatial=GLOBAL"):
        load_current_cnn_features(
            path,
            fold=2,
            expected_patient_ids=ids,
            fold_manifest=folds,
            expected_labels=labels,
            expected_n=len(ids),
            model_name="GAP0",
            spatial_axis="LOCAL",
        )
    wrong_seed_path = tmp_path / "cnn_wrong_seed.private.npz"
    np.savez_compressed(
        wrong_seed_path,
        patient_id=ids,
        split=folds.roles(2, ids),
        response_state=state,
        arm=np.asarray("GAP0"),
        seed_base=np.asarray(2025),
        fold=np.asarray(2),
    )
    with pytest.raises(ValueError, match="locked seed 2026"):
        load_current_cnn_features(
            wrong_seed_path,
            fold=2,
            expected_patient_ids=ids,
            fold_manifest=folds,
            expected_n=len(ids),
            model_name="GAP0",
            spatial_axis="GLOBAL",
        )
    missing_metadata_path = tmp_path / "cnn_missing_metadata.private.npz"
    np.savez_compressed(
        missing_metadata_path,
        patient_id=ids,
        split=folds.roles(2, ids),
        response_state=state,
        arm=np.asarray("GAP0"),
        seed_base=np.asarray(2026),
        fold=np.asarray(2),
    )
    with pytest.raises(ValueError, match="metadata.*missing|missing or invalid"):
        load_current_cnn_features(
            missing_metadata_path,
            fold=2,
            expected_patient_ids=ids,
            fold_manifest=folds,
            expected_n=len(ids),
            model_name="GAP0",
            spatial_axis="GLOBAL",
        )

    bad_path = tmp_path / "cnn_bad.private.npz"
    np.savez_compressed(
        bad_path,
        patient_id=ids,
        split=np.asarray(["train"] * len(ids)),
        response_state=state,
        arm=np.asarray("current_cnn"),
        seed_base=np.asarray(2026),
        fold=np.asarray(2),
    )
    with pytest.raises(ValueError, match="split assignment drifted"):
        load_current_cnn_features(
            bad_path,
            fold=2,
            expected_patient_ids=ids,
            fold_manifest=folds,
            expected_n=len(ids),
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["test_labels_used"] = True
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="test_labels_used=false"):
        load_current_cnn_features(
            path,
            fold=2,
            expected_patient_ids=ids,
            fold_manifest=folds,
            expected_labels=labels,
            expected_n=len(ids),
            model_name="GAP0",
            spatial_axis="GLOBAL",
        )


def _write_radiomics(path: Path, patient_ids: Sequence[str]) -> None:
    rows = []
    pairs = (("T0", "T1"), ("T1", "T2"), ("T2", "T3"))
    for patient_index, patient_id in enumerate(patient_ids):
        for transition, (start_visit, end_visit) in enumerate(pairs):
            row: dict[str, object] = {
                "patient_id": patient_id,
                "start_visit": start_visit,
                "end_visit": end_visit,
            }
            for feature_index, feature in enumerate(RADIOMIC_FEATURES):
                start = 1.0 + patient_index + feature_index + transition
                end = start + 1.0
                row[f"{feature}_start"] = start
                row[f"{feature}_end"] = end
                row[f"{feature}_absolute_change"] = end - start
                row[f"{feature}_valid"] = True
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_radiomics_requires_exact_complete_case_visit_chains(tmp_path: Path) -> None:
    cohort = _patient_ids()
    complete = cohort[:10]
    path = tmp_path / "radiomics.csv"
    _write_radiomics(path, complete)
    table = load_radiomics_table(
        path,
        cohort_patient_ids=cohort,
        expected_n=len(complete),
        expected_sha256=None,
    )
    assert table.values.shape == (10, 4, 4)
    assert np.array_equal(table.patient_ids, complete)
    assert table.aligned_values(complete, ("ftv",)).shape == (10, 4, 1)

    frame = pd.read_csv(path)
    frame.loc[1, "ftv_start"] += 0.25
    broken = tmp_path / "broken.csv"
    frame.to_csv(broken, index=False)
    with pytest.raises(ValueError, match="change identity|discontinuous"):
        load_radiomics_table(
            broken,
            cohort_patient_ids=cohort,
            expected_n=len(complete),
            expected_sha256=None,
        )

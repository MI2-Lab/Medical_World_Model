from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

AUDIT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AUDIT_ROOT / "scripts"))

from data_contracts import (  # noqa: E402
    CLINICAL_COLUMNS,
    FTV_COLUMNS,
    ContractError,
    TrainOnlyClinicalEncoder,
    canonical_sha256,
    file_sha256,
    ftv_timing_prefix,
    load_clinical_table,
    load_config,
    load_fold_manifest,
    load_ftv_wide,
    load_local_feature_asset,
    mri_timing_prefix,
    ordered_patient_sha256,
    require_file_sha256,
)


def _write_csv(path: Path, frame: pd.DataFrame) -> str:
    frame.to_csv(path, index=False)
    return file_sha256(path)


def _fold_frame(patient_count: int = 5) -> pd.DataFrame:
    patient_ids = [f"P{index}" for index in range(patient_count)]
    rows = []
    for fold in range(5):
        for index, patient_id in enumerate(patient_ids):
            if index == fold:
                split = "test"
            elif index == (fold + 1) % patient_count:
                split = "val"
            else:
                split = "train"
            rows.append(
                {
                    "patient_id": patient_id,
                    "fold": fold,
                    "split": split,
                    "label_pcr": index % 2,
                }
            )
    return pd.DataFrame(rows, columns=("patient_id", "fold", "split", "label_pcr"))


def _load_synthetic_folds(tmp_path: Path) -> tuple[Path, pd.DataFrame]:
    path = tmp_path / "folds.csv"
    digest = _write_csv(path, _fold_frame())
    loaded = load_fold_manifest(
        path,
        digest,
        expected_patient_count=5,
        expected_split_counts={
            fold: {"train": 3, "val": 1, "test": 1} for fold in range(5)
        },
    )
    return path, loaded


def _clinical_frame() -> pd.DataFrame:
    rows = []
    for index in range(5):
        hr = index % 2
        her2 = (index // 2) % 2
        subtype = {
            (1, 0): "HR+/HER2-",
            (0, 0): "HR-/HER2-",
            (1, 1): "HR+/HER2+",
            (0, 1): "HR-/HER2+",
        }[(hr, her2)]
        rows.append(
            {
                "clinical_patient_id": str(100 + index),
                "patient_id": f"P{index}",
                "preprocessed_dir": f"/private/P{index}",
                "label_pcr": index % 2,
                "label_hr": hr,
                "label_her2": her2,
                "label_mp": (index + 1) % 2,
                "age_at_screening": np.nan if index == 1 else 40 + index,
                "arm": "ARM_A" if index < 3 else "ARM_B",
                "hr_her2_subtype": subtype,
                "race_raw": "R",
                "race_simple": np.nan if index == 2 else "Race",
                "menopausal_status_raw": "M",
                "menopausal_status_simple": "pre",
                "ethnicity": "E",
                "raw_Patient_ID": str(100 + index),
                "raw_HR": hr,
                "raw_HER2": her2,
                "raw_MP": (index + 1) % 2,
                "raw_pCR": index % 2,
                "audit_status": "complete",
                "n_visits": 4,
                "complete_4visits": True,
                "missing": np.nan,
                "failed_visits": np.nan,
                "aligned_dce_visits": 0,
            }
        )
    return pd.DataFrame(rows, columns=CLINICAL_COLUMNS)


def _ftv_frame(patient_ids: tuple[str, ...] = ("P0", "P1")) -> pd.DataFrame:
    rows = []
    transitions = (
        ("T0→T1", "T0", "T1", 1.0, 2.0),
        ("T1→T2", "T1", "T2", 2.0, 4.0),
        ("T2→T3", "T2", "T3", 4.0, 8.0),
    )
    for patient_number, patient_id in enumerate(patient_ids):
        scale = patient_number + 1
        for transition, start, end, value_start, value_end in transitions:
            row = {
                "patient_id": patient_id,
                "trial_id": 100 + patient_number,
                "transition": transition,
                "start_visit": start,
                "end_visit": end,
                "ftv_start": value_start * scale,
                "ftv_end": value_end * scale,
                "ftv_absolute_change": (value_end - value_start) * scale,
                "ftv_valid": True,
                "sphericity_start": 0.1,
                "sphericity_end": 0.2,
                "sphericity_absolute_change": 0.1,
                "sphericity_valid": True,
                "ld_start": 1.0,
                "ld_end": 2.0,
                "ld_absolute_change": 1.0,
                "ld_valid": True,
                "bpe_start": 0.2,
                "bpe_end": 0.3,
                "bpe_absolute_change": 0.1,
                "bpe_valid": True,
            }
            rows.append(row)
    return pd.DataFrame(rows, columns=FTV_COLUMNS)


def _full_config(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    assets = tmp_path / "assets"
    assets.mkdir()
    files = {
        "clinical_labels": assets / "clinical.csv",
        "fold_manifest": assets / "folds.csv",
        "ftv_table": assets / "ftv.csv",
        "local_preregistration_lock": assets / "lock.json",
    }
    for name, path in files.items():
        path.write_text(f"{name}\n", encoding="utf-8")
    feature_root = assets / "features"
    feature_root.mkdir()
    paths = {
        "clinical_labels": "assets/clinical.csv",
        "clinical_labels_sha256": file_sha256(files["clinical_labels"]),
        "fold_manifest": "assets/folds.csv",
        "fold_manifest_sha256": file_sha256(files["fold_manifest"]),
        "ftv_table": "assets/ftv.csv",
        "ftv_table_sha256": file_sha256(files["ftv_table"]),
        "local_feature_root": "assets/features",
        "local_preregistration_lock": "assets/lock.json",
        "local_preregistration_lock_sha256": file_sha256(
            files["local_preregistration_lock"]
        ),
    }
    config = {
        "schema_version": 1,
        "experiment": "mri_clinical_complementarity_audit",
        "paths": paths,
        "local_cells": {
            "arms": ["LOCAL0", "LOCAL3"],
            "seed_bases": [2026, 3026],
            "folds": [0, 1, 2, 3, 4],
            "visits": ["T0", "T1", "T2", "T3"],
            "state_dim": 192,
        },
        "clinical_contracts": {
            "C1": ["label_hr", "label_her2"],
            "C2": ["label_hr", "age_at_screening", "race_simple", "arm"],
        },
        "primary_clinical_contract": "C2",
    }
    config_path = tmp_path / "audit.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path, files | {"local_feature_root": feature_root}


def _selection_payload(
    *,
    arm: str,
    seed_base: int,
    fold: int,
    lock_digest: str,
    train_digest: str,
    val_digest: str,
    data_digest: str,
    stage_a_digest: str,
) -> dict[str, object]:
    return {
        "allowed_state_loss": None,
        "architecture": "LOCAL",
        "arm": arm,
        "data_provenance_sha256": data_digest,
        "delta_ftv_used": False,
        "effective_seed": seed_base + fold,
        "epochs": [],
        "experiment_pass": True,
        "fallback_rule": "unused synthetic fixture",
        "finite_status": True,
        "fold": fold,
        "history_sha256": "7" * 64,
        "hyperparameters": {},
        "optimization_safety_pass": True,
        "paired_baseline_state_loss": None,
        "paired_initialization_sha256": "8" * 64,
        "pcr_used": False,
        "preregistration": {"status": "PASS", "lock_sha256": lock_digest},
        "preregistration_lock_sha256": lock_digest,
        "preregistration_status": "PASS",
        "schema_version": 1,
        "seed_base": seed_base,
        "selected_epoch": 2,
        "selected_representation_std": 0.5,
        "selected_validation_base_loss": 1.0,
        "selected_validation_ftv_loss": 0.0,
        "selected_validation_state_loss": 0.5,
        "selected_validation_total_loss": 1.0,
        "selection_mode": "primary",
        "selection_rule": "synthetic",
        "stage_a_sentinel_sha256": stage_a_digest,
        "state_loss_degradation_fraction": None,
        "test_data_used": False,
        "train_patient_sha256": train_digest,
        "val_patient_sha256": val_digest,
    }


def _local_asset(
    tmp_path: Path, folds: pd.DataFrame, *, state_dim: int = 3
) -> tuple[Path, str]:
    arm = "LOCAL0"
    seed_base = 2026
    fold = 0
    feature_dir = (
        tmp_path
        / "features"
        / "formal_4x8"
        / f"seed_{seed_base}"
        / arm
        / f"fold_{fold}"
    )
    checkpoint_dir = (
        tmp_path
        / "checkpoints"
        / "formal_4x8"
        / f"seed_{seed_base}"
        / arm
        / f"fold_{fold}"
    )
    feature_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    feature_path = feature_dir / "response_state.private.npz"
    checkpoint_path = checkpoint_dir / "selected.pt"
    selection_path = checkpoint_dir / "selection.json"

    current = folds.loc[folds["fold"].eq(fold), ["patient_id", "split"]]
    patient_ids = current["patient_id"].to_numpy(dtype=str)
    split = current["split"].to_numpy(dtype=str)
    response = np.arange(len(current) * 4 * state_dim, dtype=np.float32).reshape(
        len(current), 4, state_dim
    )
    np.savez_compressed(
        feature_path,
        patient_id=patient_ids,
        split=split,
        response_state=response,
        arm=np.asarray(arm),
        seed_base=np.asarray(seed_base, dtype=np.int64),
        fold=np.asarray(fold, dtype=np.int64),
    )
    checkpoint_path.write_bytes(b"selected checkpoint fixture")

    lock_digest = "1" * 64
    train_digest = "2" * 64
    val_digest = canonical_sha256(
        sorted(current.loc[current["split"].eq("val"), "patient_id"])
    )
    data_digest = "3" * 64
    stage_a_digest = "4" * 64
    selection = _selection_payload(
        arm=arm,
        seed_base=seed_base,
        fold=fold,
        lock_digest=lock_digest,
        train_digest=train_digest,
        val_digest=val_digest,
        data_digest=data_digest,
        stage_a_digest=stage_a_digest,
    )
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    metadata = {
        "arm": arm,
        "checkpoint_data_provenance_sha256": data_digest,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "cohort": "exact_locked_primary_train_validation_test",
        "current_data_contract_provenance_sha256": "5" * 64,
        "experiment": "local_global_response_state_pilot",
        "feature_dtype": "float32",
        "feature_implementation_sha256": "6" * 64,
        "feature_path": str(feature_path.resolve()),
        "feature_sha256": file_sha256(feature_path),
        "feature_shape": list(response.shape),
        "feature_tensor": "online_preprojector_response_state",
        "fold": fold,
        "ftv_head_called": False,
        "patient_order_sha256": ordered_patient_sha256(patient_ids),
        "preregistration_lock": "PREREGISTRATION_LOCK.json",
        "preregistration_lock_sha256": lock_digest,
        "schema_version": 1,
        "seed_base": seed_base,
        "selected_epoch": 2,
        "selection_path": str(selection_path.resolve()),
        "selection_sha256": file_sha256(selection_path),
        "stage_a_sentinel_sha256": stage_a_digest,
        "test_labels_used": False,
        "train_patient_sha256": train_digest,
        "validation_patient_sha256": val_digest,
    }
    feature_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return feature_path, lock_digest


def test_config_paths_resolve_against_repo_and_hashes_fail_closed(
    tmp_path: Path,
) -> None:
    config_path, files = _full_config(tmp_path)
    loaded = load_config(config_path, repo_root=tmp_path)
    assert loaded["paths"]["clinical_labels"] == files["clinical_labels"].resolve()
    assert (
        loaded["paths"]["local_feature_root"] == files["local_feature_root"].resolve()
    )

    files["clinical_labels"].write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ContractError, match="SHA-256 mismatch"):
        load_config(config_path, repo_root=tmp_path)
    with pytest.raises(ContractError, match="SHA-256 mismatch"):
        require_file_sha256(
            files["clinical_labels"], loaded["paths"]["clinical_labels_sha256"]
        )


def test_fold_manifest_requires_long_exact_stable_contract(tmp_path: Path) -> None:
    path, folds = _load_synthetic_folds(tmp_path)
    assert folds.shape == (25, 4)
    assert (
        folds.groupby("patient_id")["split"]
        .apply(lambda value: (value == "test").sum())
        .eq(1)
        .all()
    )

    broken = _fold_frame()
    broken.loc[(broken["patient_id"] == "P0") & (broken["fold"] == 4), "label_pcr"] = 1
    digest = _write_csv(path, broken)
    with pytest.raises(ContractError, match="stable"):
        load_fold_manifest(path, digest, expected_patient_count=5)


def test_clinical_schema_patient_equality_and_label_consistency(tmp_path: Path) -> None:
    _, folds = _load_synthetic_folds(tmp_path)
    path = tmp_path / "clinical.csv"
    digest = _write_csv(path, _clinical_frame())
    clinical = load_clinical_table(path, digest, folds, expected_patient_count=5)
    assert set(clinical["patient_id"]) == set(folds["patient_id"])
    assert clinical["label_pcr"].dtype == np.dtype("int8")

    wrong = _clinical_frame()
    wrong.loc[0, "patient_id"] = "OUTSIDE"
    digest = _write_csv(path, wrong)
    with pytest.raises(ContractError, match="patient equality"):
        load_clinical_table(path, digest, folds, expected_patient_count=5)


def test_ftv_reconstruction_consistency_and_prefix_no_future_mixing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ftv.csv"
    frame = _ftv_frame()
    digest = _write_csv(path, frame)
    wide = load_ftv_wide(
        path,
        digest,
        expected_patient_ids=[f"P{index}" for index in range(5)],
        expected_patient_count=2,
    )
    assert wide.loc[
        wide["patient_id"].eq("P0"), list(wide.columns[1:])
    ].to_numpy().tolist() == [[1.0, 2.0, 4.0, 8.0]]
    early = ftv_timing_prefix(wide, "T1", log1p=False)
    altered = wide.copy()
    altered[["FTV_T2", "FTV_T3"]] = 99999.0
    np.testing.assert_array_equal(early, ftv_timing_prefix(altered, "T1", log1p=False))

    broken = frame.copy()
    broken.loc[
        (broken["patient_id"] == "P0") & (broken["transition"] == "T1→T2"), "ftv_start"
    ] = 2.5
    broken.loc[
        (broken["patient_id"] == "P0") & (broken["transition"] == "T1→T2"),
        "ftv_absolute_change",
    ] = 1.5
    digest = _write_csv(path, broken)
    with pytest.raises(ContractError, match="FTV_T1 is inconsistent"):
        load_ftv_wide(path, digest, expected_patient_count=2)


def test_local_npz_metadata_checkpoint_and_fold_binding(tmp_path: Path) -> None:
    _, folds = _load_synthetic_folds(tmp_path)
    feature_path, lock_digest = _local_asset(tmp_path, folds)
    asset = load_local_feature_asset(
        feature_path,
        folds,
        arm="LOCAL0",
        seed_base=2026,
        fold=0,
        preregistration_lock_sha256=lock_digest,
        expected_patient_count=5,
        state_dim=3,
    )
    assert asset.response_state.shape == (5, 4, 3)
    assert asset.selection["selected_epoch"] == asset.metadata["selected_epoch"]

    metadata_path = feature_path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["feature_sha256"] = "f" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ContractError, match="metadata hash"):
        load_local_feature_asset(
            feature_path,
            folds,
            arm="LOCAL0",
            seed_base=2026,
            fold=0,
            expected_lock_sha256=lock_digest,
            expected_patient_count=5,
            state_dim=3,
        )


def test_train_only_encoder_median_missing_sorted_vocab_and_unknown_zero() -> None:
    train = pd.DataFrame(
        {
            "age_at_screening": [20.0, np.nan, 40.0],
            "race_simple": ["B", None, "A"],
        }
    )
    encoder = TrainOnlyClinicalEncoder(("age_at_screening", "race_simple"))
    encoded_train = encoder.fit_transform(train)
    assert encoder.numeric_medians_["age_at_screening"] == 30.0
    assert encoder.categories_["race_simple"] == tuple(
        sorted(("A", "B", "__MISSING__"))
    )
    assert np.isfinite(encoded_train).all()

    evaluation = pd.DataFrame(
        {"age_at_screening": [np.nan, 50.0], "race_simple": ["NEVER_SEEN", None]}
    )
    encoded = encoder.transform(evaluation)
    race_start = 1
    assert encoded[0, 0] == 30.0
    assert encoded[0, race_start:].sum() == 0.0
    missing_column = encoder.feature_names_.index("race_simple=__MISSING__")
    assert encoded[1, missing_column] == 1.0
    assert "NEVER_SEEN" not in encoder.categories_["race_simple"]


def test_mri_prefix_excludes_future_visits() -> None:
    states = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
    prefix = mri_timing_prefix(states, "T1", state_dim=3)
    assert prefix.shape == (2, 6)
    np.testing.assert_array_equal(prefix, states[:, :2].reshape(2, 6))
    altered = states.copy()
    altered[:, 2:] = -999.0
    np.testing.assert_array_equal(prefix, mri_timing_prefix(altered, 1, state_dim=3))

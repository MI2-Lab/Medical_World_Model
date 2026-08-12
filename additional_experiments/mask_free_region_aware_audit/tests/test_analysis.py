from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_audit as audit  # noqa: E402


def _mini_feature_config(n: int = 12) -> dict:
    return {
        "experiment": "mask_free_region_aware_audit",
        "frozen_cells": {"patient_count": n},
        "variants": {"dimensions": {name: (3 if name != "R5" else 5) for name in audit.ALL_VARIANTS}},
        "config_sha256": "c" * 64,
    }


def _write_asset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, pd.DataFrame, dict, dict, dict]:
    n = 12
    config = _mini_feature_config(n)
    directory = tmp_path / "seed_2026" / "LOCAL0" / "fold_0"
    directory.mkdir(parents=True)
    path = directory / audit.FEATURE_FILENAME
    patients = np.asarray([f"subject-{index:03d}" for index in range(n)])
    split = np.asarray(["train"] * 6 + ["val"] * 3 + ["test"] * 3)
    arrays = {
        "patient_id": patients,
        "split": split,
        **{
            name: np.full((n, 4, int(config["variants"]["dimensions"][name])), index + 0.25, dtype=np.float32)
            for index, name in enumerate(audit.ALL_VARIANTS)
        },
        "arm": np.asarray("LOCAL0"),
        "seed_base": np.asarray(2026, dtype=np.int64),
        "fold": np.asarray(0, dtype=np.int64),
    }
    np.savez(path, **arrays)
    metadata = {
        "status": "COMPLETE",
        "experiment": "mask_free_region_aware_audit",
        "cell": "seed_2026/LOCAL0/fold_0",
        "arm": "LOCAL0",
        "seed_base": 2026,
        "fold": 0,
        "patient_count": n,
        "encoder_frozen": True,
        "training_performed": False,
        "streamed_raw_spatial_map_not_persisted": True,
        "phenotype_or_pcr_labels_read": False,
        "feature_path": str(path.resolve()),
        "feature_sha256": audit.file_sha256(path),
        "patient_order_sha256": audit.ordered_sha256(patients),
        "split_order_sha256": audit.ordered_sha256(split),
        "variant_shapes": {
            name: [n, 4, int(config["variants"]["dimensions"][name])]
            for name in audit.ALL_VARIANTS
        },
        "variant_dtypes": {name: "float32" for name in audit.ALL_VARIANTS},
        "config_sha256": config["config_sha256"],
        "checkpoint_sha256": "a" * 64,
        "selection_sha256": "b" * 64,
    }
    metadata_path = path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(audit, "_metadata_keys", lambda: frozenset(metadata))
    monkeypatch.setattr(audit, "LOCK_PATH", tmp_path / "absent-lock.json")
    inventory = {
        "feature_path": str(path.resolve()),
        "feature_sha256": audit.file_sha256(path),
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": audit.file_sha256(metadata_path),
    }
    lock = {
        "selected_cells": {
            "seed_2026/LOCAL0/fold_0": {
                "checkpoint_sha256": "a" * 64,
                "selection_sha256": "b" * 64,
            }
        }
    }
    folds = pd.DataFrame({"patient_id": patients, "fold": 0, "split": split})
    return path, folds, config, lock, inventory


def test_regional_loader_is_strict_and_binds_fold_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, folds, config, lock, inventory = _write_asset(tmp_path, monkeypatch)
    asset = audit.load_regional_feature_asset(
        path, folds, config, seed=2026, arm="LOCAL0", fold=0,
        lock=lock, inventory_record=inventory,
    )
    assert asset.variant("R0").dtype == np.float32
    assert asset.variant("R5").shape == (12, 4, 5)
    assert asset.metadata["phenotype_or_pcr_labels_read"] is False

    metadata_path = path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["unexpected"] = True
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata keys drifted"):
        audit.load_regional_feature_asset(
            path, folds, config, seed=2026, arm="LOCAL0", fold=0,
            lock=lock, inventory_record=inventory,
        )


def test_regional_loader_rejects_locked_split_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, folds, config, lock, inventory = _write_asset(tmp_path, monkeypatch)
    folds = folds.copy()
    folds.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="locked outer fold"):
        audit.load_regional_feature_asset(
            path, folds, config, seed=2026, arm="LOCAL0", fold=0,
            lock=lock, inventory_record=inventory,
        )


def test_timing_contract_is_causal_and_static() -> None:
    values = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
    np.testing.assert_array_equal(audit.causal_prefix(values, "T0"), values[:, :1].reshape(2, 3))
    np.testing.assert_array_equal(audit.causal_prefix(values, "T0-T2"), values[:, :3].reshape(2, 9))
    np.testing.assert_array_equal(audit.static_visit(values, "T3"), values[:, 3])
    with pytest.raises(ValueError, match="unknown causal"):
        audit.causal_prefix(values, "T1-T3")


def _binary_config() -> dict:
    return {
        "logistic": {
            "c_grid": [100.0, 0.01, 1.0],
            "solver": "liblinear",
            "max_iter": 10_000,
        }
    }


def test_binary_probe_uses_train_scaler_smaller_c_and_one_test_call() -> None:
    train_x = np.r_[np.linspace(-4, -1, 8), np.linspace(1, 4, 8)][:, None]
    train_y = np.asarray([0] * 8 + [1] * 8)
    val_x = np.asarray([[-3.0], [-2.0], [2.0], [3.0]])
    val_y = np.asarray([0, 0, 1, 1])
    test_x = np.asarray([[-2.5], [-1.5], [1.5], [2.5]])
    test_y = np.asarray([0, 0, 1, 1])
    matrix = np.vstack((train_x, val_x, test_x))
    labels = np.concatenate((train_y, val_y, test_y))
    indices = {
        "train": np.arange(len(train_x)),
        "val": np.arange(len(train_x), len(train_x) + len(val_x)),
        "test": np.arange(len(train_x) + len(val_x), len(matrix)),
    }
    fit = audit._fit_binary(matrix, labels, indices, _binary_config(), class_weight=None)
    np.testing.assert_allclose(fit.scaler.mean_, train_x.mean(axis=0))
    assert fit.selected_c == pytest.approx(0.01)

    predictions: list[dict] = []
    hyperparameters: list[dict] = []
    audit._append_binary_fit(
        predictions, hyperparameters,
        patient_ids=np.asarray([f"p-{index}" for index in range(len(matrix))]),
        fold=0, labels=labels, matrix=matrix, indices=indices, config=_binary_config(),
        class_weight=None,
        metadata={
            "analysis": "mri_only_pcr", "context": "MRI_ONLY", "population": "synthetic",
            "seed": 2026, "arm": "LOCAL0", "view": "T0", "target": "pCR",
            "variant": "R0", "model": "R0", "clinical_contract": "",
        },
    )
    assert len(predictions) == len(test_x)
    assert hyperparameters[0]["test_predict_call_count"] == 1
    assert hyperparameters[0]["test_used_for_selection"] is False
    assert {row["patient_id"] for row in predictions} == {f"p-{index}" for index in indices["test"]}


def test_exact_multiclass_binary_ovr_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    centers = np.asarray([[-4.0, -4.0], [4.0, -4.0], [-4.0, 4.0], [4.0, 4.0]])
    classes = np.asarray(audit.SUBTYPE_CLASSES)
    train = np.vstack([center + offset for center in centers for offset in (-0.2, 0.0, 0.2)])
    train_y = np.repeat(classes, 3)
    validation = np.vstack([center + offset for center in centers for offset in (-0.1, 0.1)])
    validation_y = np.repeat(classes, 2)
    monkeypatch.setattr(audit.sklearn, "__version__", "1.8.0")
    fit = audit.fit_multiclass_logistic_exact(
        train, train_y, validation, validation_y, [100.0, 0.01, 1.0]
    )
    assert fit.selected_c == pytest.approx(0.01)
    assert fit.validation_macro_ovr_auroc == pytest.approx(1.0)
    probability = fit.predict_proba(validation)
    np.testing.assert_allclose(probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-15)
    decision = fit.model.decision_function(fit.scaler.transform(validation))
    expected = 1.0 / (1.0 + np.exp(-decision))
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(probability, expected, rtol=0.0, atol=1e-15)


def test_classification_aggregation_requires_five_unique_oof_folds() -> None:
    rows = []
    for index in range(20):
        label = index % 2
        rows.append({
            "patient_id": f"p{index}", "fold": index % 5, "population": "full_808",
            "seed": 2026, "arm": "LOCAL0", "analysis": "mri_only_pcr", "context": "MRI_ONLY",
            "view": "T0", "target": "pCR", "variant": "R0", "model": "R0",
            "clinical_contract": "", "y_true": label,
            "predicted_probability": 0.8 if label else 0.2,
            "predicted_label": label, "threshold": 0.5,
        })
    predictions = pd.DataFrame(rows).reindex(columns=audit.PREDICTION_COLUMNS)
    metrics = audit.aggregate_classification_oof(predictions)
    assert metrics.loc[0, "auroc"] == pytest.approx(1.0)
    assert metrics.loc[0, "n"] == 20
    duplicated = pd.concat((predictions, predictions.iloc[[0]]), ignore_index=True)
    with pytest.raises(ValueError, match="repeats a patient"):
        audit.aggregate_classification_oof(duplicated)


class _IdentityStaticTransform:
    fold = 0

    def transform_values(self, values: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(values, dtype=float), np.asarray(valid, dtype=bool)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float)

    def to_dict(self) -> dict:
        return {"synthetic": "identity"}


def test_ftv_endpoint_reuses_upstream_ridge_and_predicts_test_once(monkeypatch: pytest.MonkeyPatch) -> None:
    import c1b_stage_b.probes as upstream

    n = 18
    patient_ids = np.asarray([f"p{index}" for index in range(n)])
    split = np.asarray(["train"] * 10 + ["val"] * 4 + ["test"] * 4)
    state = np.zeros((n, 4, 2), dtype=np.float32)
    state[:, :, 0] = np.arange(n)[:, None]
    state[:, :, 1] = np.arange(4)[None, :]
    asset = audit.RegionalFeatureAsset(
        Path("synthetic.npz"), Path("synthetic.json"), patient_ids, split,
        {"R0": state}, "LOCAL0", 2026, 0, {},
    )
    records = {
        patient_id: SimpleNamespace(
            values=np.asarray([index + 1.0, index + 2.0, index + 4.0, index + 7.0]),
            measurement_valid=np.ones(4, dtype=bool),
            observable=np.ones(4, dtype=bool),
        )
        for index, patient_id in enumerate(patient_ids)
    }
    original = upstream.select_ridge
    calls = {"count": 0}

    def counted(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(upstream, "select_ridge", counted)
    selection, predictions = audit._run_ftv_endpoint(
        {"ridge": {"alpha_grid": [0.0001, 0.1, 10.0]}}, asset, "R0", records,
        task="static", index=0, observable_only=False,
        static_transform=_IdentityStaticTransform(),
    )
    assert calls["count"] == 1
    assert selection["test_predict_call_count"] == 1
    assert selection["test_used_for_selection"] is False
    assert len(predictions) == 4
    assert {row["patient_id"] for row in predictions} == set(patient_ids[-4:])


def _probability_rows(
    patient_ids: np.ndarray, labels: np.ndarray, probabilities: np.ndarray, *,
    seed: int, variant: str, population: str, analysis: str = "oracle_pcr",
) -> pd.DataFrame:
    predicted = (probabilities >= 0.5).astype(int)
    return pd.DataFrame({
        "patient_id": patient_ids, "fold": np.arange(len(patient_ids)) % 5,
        "population": population, "seed": seed, "arm": "LOCAL0", "analysis": analysis,
        "view": "T0-T1", "target": "pCR", "variant": variant,
        "clinical_contract": "", "y_true": labels,
        "predicted_probability": probabilities, "predicted_label": predicted,
        "threshold": 0.5,
        **{name: np.nan for name in audit.SUBTYPE_PROBABILITY_COLUMNS},
        **{name: np.nan for name in audit.SUBTYPE_LABEL_COLUMNS},
    }).reindex(columns=audit.GOAL5_PREDICTION_COLUMNS)


def test_goal5_loader_round_trips_exact_r0_p1_predictions(tmp_path: Path) -> None:
    patient_ids = np.asarray(["synthetic-0", "synthetic-1"])
    labels = np.asarray([0, 1])
    parser_sensitive = float(
        np.nextafter(np.float64(0.3), np.float64(1.0))
    )
    probabilities = np.asarray([parser_sensitive, 0.7], dtype=np.float64)
    goal5_p1 = _probability_rows(
        patient_ids,
        labels,
        probabilities,
        seed=2026,
        variant="P1",
        population="ftv_complete_375",
        analysis="mri_only_pcr",
    )
    goal5_p1.loc[0, "threshold"] = parser_sensitive
    path = tmp_path / "goal5_p1.private.csv"
    goal5_p1.to_csv(path, index=False)

    default_loaded = pd.read_csv(path)
    assert default_loaded.loc[0, "predicted_probability"] != parser_sensitive
    assert default_loaded.loc[0, "threshold"] != parser_sensitive

    new_r0 = goal5_p1.copy()
    new_r0["variant"] = "R0"
    with pytest.raises(ValueError, match="predicted_probability"):
        audit.verify_r0_p1_parity(new_r0, default_loaded)

    round_trip_loaded = audit._load_goal5_predictions(
        path, audit.file_sha256(path), "synthetic Goal-5 P1 predictions"
    )
    assert np.array_equal(
        round_trip_loaded["predicted_probability"].to_numpy(dtype=np.float64),
        goal5_p1["predicted_probability"].to_numpy(dtype=np.float64),
    )
    assert np.array_equal(
        round_trip_loaded["threshold"].to_numpy(dtype=np.float64),
        goal5_p1["threshold"].to_numpy(dtype=np.float64),
    )
    parity = audit.verify_r0_p1_parity(new_r0, round_trip_loaded)
    assert parity == {
        "status": "PASS",
        "exact_probability_label_threshold_equality": True,
        "checked_cells": 1,
        "checked_rows": 2,
    }

    truly_corrupted = round_trip_loaded.copy()
    truly_corrupted.loc[0, "predicted_probability"] = np.nextafter(
        truly_corrupted.loc[0, "predicted_probability"], np.inf
    )
    with pytest.raises(ValueError, match="predicted_probability"):
        audit.verify_r0_p1_parity(new_r0, truly_corrupted)


def test_oracle_recovery_uses_exact_matched_population_and_positive_denominator() -> None:
    n = 20
    patient_ids = np.asarray([f"p{index:02d}" for index in range(n)])
    labels = np.asarray([0, 1] * (n // 2))
    fixed_probability = np.linspace(0.1, 0.9, n)
    peri_probability = fixed_probability.copy()
    peri_probability[labels == 1] += 0.2
    peri_probability[labels == 0] -= 0.2
    r0_probability = fixed_probability.copy()
    candidate_probability = r0_probability.copy()
    candidate_probability[labels == 1] += 0.1
    candidate_probability[labels == 0] -= 0.1
    frames = []
    for variant in ("FIXED_P3", "PERI10", "PERI20", "CORE", "CORE_PERI"):
        population = "oracle_pair_PERI20" if variant in {"FIXED_P3", "PERI20"} else f"oracle_pair_{variant}"
        probability = peri_probability if variant == "PERI20" else fixed_probability
        frames.append(_probability_rows(patient_ids, labels, probability, seed=2026, variant=variant, population=population))
    goal5_oracle = pd.concat(frames, ignore_index=True)
    new_predictions = pd.concat(
        [_probability_rows(patient_ids, labels, r0_probability if variant == "R0" else candidate_probability,
                           seed=2026, variant=variant, population="ftv_complete_375", analysis="mri_only_pcr")
         for variant in ("R0", "R1", "R2", "R3", "R5")],
        ignore_index=True,
    ).rename(columns={})
    new_predictions["context"] = "MRI_ONLY"
    new_predictions["model"] = new_predictions["variant"]
    new_predictions = new_predictions.reindex(columns=audit.PREDICTION_COLUMNS)
    goal5_mri = _probability_rows(
        patient_ids, labels, r0_probability, seed=2026, variant="P1",
        population="ftv_complete_375", analysis="mri_only_pcr",
    )
    new_metrics = audit.aggregate_classification_oof(new_predictions)
    denominator = (
        audit._binary_oof_values(frames[2])["auroc"]
        - audit._binary_oof_values(frames[0])["auroc"]
    )
    config = {
        "frozen_cells": {"seed_bases": [2026]},
        "oracle": {
            "primary_arm": "LOCAL0", "view": "T0-T1", "population": "oracle_pair_PERI20",
            "patient_count": n, "positive_count": int(labels.sum()),
            "sorted_patient_sha256": audit.ordered_sha256(sorted(patient_ids)),
            "reported_goal5_variants": ["FIXED_P3", "PERI10", "PERI20", "CORE", "CORE_PERI"],
            "new_candidates": ["R1", "R2", "R3", "R5"],
            "published_uplift": {"2026": denominator},
        },
    }
    table, parity = audit.build_oracle_recovery_table(
        config, new_predictions, new_metrics, goal5_oracle, goal5_mri
    )
    recovery = table.loc[table["row_type"].eq("recovery")]
    assert len(recovery) == 4
    assert recovery["recovery_defined"].all()
    assert np.isfinite(recovery["recovery_ratio"]).all()
    assert parity["status"] == "PASS"
    corrupted = goal5_oracle.loc[~(
        goal5_oracle["variant"].eq("PERI20") & goal5_oracle["patient_id"].eq(patient_ids[0])
    )]
    with pytest.raises(ValueError, match="row count drifted"):
        audit.build_oracle_recovery_table(config, new_predictions, new_metrics, corrupted, goal5_mri)


def test_region_occupancy_flattens_frozen_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    metric_dir = tmp_path / "metrics"
    metric_dir.mkdir()
    regions = ("R0", "R1", "R2", "R3", "S1", "S2", "S3")
    payload = {
        "weight_sum_cells": {name: 10.0 + index for index, name in enumerate(regions)},
        "physical_volume_mm3": {name: 1000.0 + index for index, name in enumerate(regions)},
        "expected_physical_volume_mm3": {name: 1000.0 + index for index, name in enumerate(regions)},
        "nonzero_cells": {name: 20 + index for index, name in enumerate(regions)},
        "fractional_cells": {name: 5 + index for index, name in enumerate(regions)},
        "sampling_cell_volume_mm3": 128.0,
    }
    (metric_dir / "region_occupancy_contract.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    table = audit.region_occupancy_table({
        "feature_contract": {
            "primary_boundaries_mm": [32.0, 48.0, 64.0],
            "secondary_boundaries_mm": [24.0, 40.0, 64.0],
        }
    })
    assert tuple(table["region"]) == regions
    assert set(table["geometry"]) == {"primary", "secondary", "primary_and_secondary"}
    assert table["mean_effective_cells"].gt(0).all()


def test_private_writer_is_owner_only(tmp_path: Path) -> None:
    path = tmp_path / "predictions" / "synthetic.private.csv"
    audit._atomic_csv(pd.DataFrame({"patient_id": ["secret"], "value": [1]}), path, private=True)
    assert path.stat().st_mode & 0o077 == 0


def test_actual_config_contract_loads_without_opening_labels() -> None:
    config = audit.load_config(ROOT / "configs" / "audit.json", verify_paths=False)
    assert config["logistic"]["solver"] == "liblinear"
    assert config["bootstrap"]["replicates"] == 2000
    assert config["analysis"]["pcr_populations"] == ["full_808", "ftv_complete_375"]

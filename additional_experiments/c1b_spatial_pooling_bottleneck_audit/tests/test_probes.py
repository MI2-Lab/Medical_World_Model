from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import stat
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c1b_spatial_audit.contracts import file_sha256  # noqa: E402
from c1b_spatial_audit.probes import (  # noqa: E402
    ALPHAS,
    FrozenStateAsset,
    TestPredictGuard,
    load_frozen_state_asset,
    ordered_patient_sha256,
    pooled_oof_natural_metrics,
    run_continuous_probe_cell,
    run_ftv_probe_cell,
    select_ridge,
    write_probe_outputs,
)


@dataclass(frozen=True)
class SyntheticFTVRecord:
    values: np.ndarray
    measurement_valid: np.ndarray
    observable: np.ndarray

    @property
    def grounding_eligible(self) -> np.ndarray:
        return (
            np.asarray(self.measurement_valid, dtype=bool)
            & np.asarray(self.observable, dtype=bool)
            & np.isfinite(self.values)
        )


def synthetic_asset(*, feature_dim: int = 7) -> tuple[FrozenStateAsset, dict[str, SyntheticFTVRecord]]:
    patient_ids = np.asarray([f"p{index:02d}" for index in range(18)])
    split = np.asarray(["train"] * 8 + ["val"] * 5 + ["test"] * 5)
    state = np.empty((len(patient_ids), 4, feature_dim), dtype=np.float32)
    records: dict[str, SyntheticFTVRecord] = {}
    for patient_index, patient_id in enumerate(patient_ids):
        visit = np.arange(4, dtype=np.float64)
        values = (
            8.0
            + 0.7 * patient_index
            + (1.0 + 0.04 * patient_index) * visit
            + 0.15 * visit**2
        )
        measurement_valid = np.ones(4, dtype=bool)
        observable = np.ones(4, dtype=bool)
        if patient_id == "p14":
            observable[2] = False
        records[str(patient_id)] = SyntheticFTVRecord(
            values=values,
            measurement_valid=measurement_valid,
            observable=observable,
        )
        for visit_index in range(4):
            basis = np.asarray(
                [
                    values[visit_index],
                    patient_index,
                    visit_index,
                    values[visit_index] * (patient_index + 1) / 20.0,
                    np.sin(patient_index + visit_index),
                    np.cos(0.3 * patient_index - visit_index),
                    (patient_index + 1) ** 0.5 + visit_index,
                ],
                dtype=np.float32,
            )
            if feature_dim <= len(basis):
                state[patient_index, visit_index] = basis[:feature_dim]
            else:
                state[patient_index, visit_index] = np.resize(basis, feature_dim)
    state_valid = np.ones((len(patient_ids), 4), dtype=bool)
    state_valid[13, 3] = False
    return (
        FrozenStateAsset(
            patient_id=patient_ids,
            split=split,
            state=state,
            state_valid=state_valid,
            arm="N1",
            seed_base=2026,
            fold=0,
            pooling="PLOCAL+GLOBAL",
        ),
        records,
    )


class FrozenProbeContractTests(unittest.TestCase):
    def test_exact_stage_b_ridge_api_is_reused(self) -> None:
        self.assertEqual(
            ALPHAS,
            (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0),
        )
        self.assertEqual(select_ridge.__module__, "c1b_stage_b.probes")
        self.assertIs(ALPHAS, sys.modules[select_ridge.__module__].ALPHAS)
        self.assertIn(
            "c1b_overlap_eligibility_ftv_stageb",
            str(Path(sys.modules[select_ridge.__module__].__file__).resolve()),
        )

    def test_dimension_agnostic_ftv_static_and_literal_delta(self) -> None:
        asset, records = synthetic_asset(feature_dim=7)
        result = run_ftv_probe_cell(asset, records)

        self.assertEqual(asset.feature_dim, 7)
        self.assertEqual(len(result.selection), 14)
        self.assertEqual(len(result.metrics), 36)
        self.assertEqual(set(result.predictions["split"]), {"test"})
        self.assertTrue(result.predictions["test_predict_call_count"].eq(1).all())
        self.assertTrue(result.selection["test_predict_call_count"].eq(1).all())
        self.assertFalse(result.selection["test_used_for_scaler"].any())
        self.assertFalse(result.selection["test_used_for_alpha_selection"].any())
        self.assertTrue(result.selection["feature_dim"].eq(7).all())
        self.assertTrue(
            result.selection["x_scaler_train_rows"].eq(result.selection["n_train"]).all()
        )

        primary_t3 = result.predictions.loc[
            result.predictions["task"].eq("static")
            & result.predictions["analysis_scope"].eq("primary_measurement_valid")
            & result.predictions["endpoint"].eq("T3")
        ]
        self.assertNotIn("p13", set(primary_t3["patient_id"]))
        observable_t2 = result.predictions.loc[
            result.predictions["task"].eq("static")
            & result.predictions["analysis_scope"].eq("observable_only")
            & result.predictions["endpoint"].eq("T2")
        ]
        self.assertNotIn("p14", set(observable_t2["patient_id"]))

        delta_row = result.predictions.loc[
            result.predictions["task"].eq("delta")
            & result.predictions["analysis_scope"].eq("primary_measurement_valid")
            & result.predictions["endpoint"].eq("T0→T1")
            & result.predictions["patient_id"].eq("p15")
        ].iloc[0]
        expected_delta = records["p15"].values[1] - records["p15"].values[0]
        self.assertAlmostEqual(float(delta_row["y_true"]), float(expected_delta))
        delta_transform = json.loads(
            result.selection.loc[
                result.selection["task"].eq("delta"), "target_transform_json"
            ].iloc[0]
        )
        self.assertEqual(delta_transform["value_transform"], "literal_natural_delta")
        self.assertEqual(
            delta_transform["standardization"], "outer_train_standard_scaler"
        )

    def test_test_prediction_guard_is_single_use(self) -> None:
        class ConstantModel:
            @staticmethod
            def predict(matrix: np.ndarray) -> np.ndarray:
                return np.zeros(len(matrix), dtype=np.float64)

        guard = TestPredictGuard()
        matrix = np.ones((3, 2), dtype=np.float64)
        np.testing.assert_array_equal(guard.predict(ConstantModel(), matrix), np.zeros(3))
        self.assertEqual(guard.calls, 1)
        with self.assertRaisesRegex(RuntimeError, "single-use"):
            guard.predict(ConstantModel(), matrix)

    def test_continuous_nuisance_targets_use_same_isolated_ridge(self) -> None:
        asset, _ = synthetic_asset(feature_dim=5)
        targets: dict[str, dict[str, object]] = {
            "padding_fraction": {},
            "native_spacing_x_mm": {},
        }
        for patient_index, patient_id in enumerate(asset.patient_id):
            visits = np.arange(4, dtype=np.float64)
            targets["padding_fraction"][str(patient_id)] = (
                0.05 + patient_index / 100.0 + visits / 50.0
            )
            valid = np.ones(4, dtype=bool)
            if patient_id == "p15":
                valid[1] = False
            targets["native_spacing_x_mm"][str(patient_id)] = (
                0.7 + patient_index / 80.0 + visits / 100.0,
                valid,
            )

        result = run_continuous_probe_cell(
            asset,
            targets,
            target_semantics={
                "padding_fraction": "source_padding_fraction",
                "native_spacing_x_mm": "source_affine_column_norm_x_mm",
            },
        )
        self.assertEqual(len(result.selection), 8)
        self.assertEqual(len(result.metrics), 20)
        self.assertEqual(set(result.selection["task"]), {"nuisance"})
        self.assertTrue(result.selection["feature_dim"].eq(5).all())
        self.assertTrue(result.selection["y_scaler_mean"].notna().all())
        self.assertTrue(result.predictions["test_predict_call_count"].eq(1).all())
        invalid_visit = result.predictions.loc[
            result.predictions["target_name"].eq("native_spacing_x_mm")
            & result.predictions["endpoint"].eq("T1")
        ]
        self.assertNotIn("p15", set(invalid_visit["patient_id"]))
        for payload in result.selection["target_transform_json"]:
            self.assertEqual(json.loads(payload)["value_transform"], "identity_natural")


class PooledOOFMetricTests(unittest.TestCase):
    @staticmethod
    def predictions() -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for fold in (0, 1):
            for endpoint_index, endpoint in enumerate(("T0", "T1", "T2", "T3")):
                for patient_index in range(3):
                    truth = float(10 * fold + patient_index + 2 * endpoint_index)
                    rows.append(
                        {
                            "patient_id": f"f{fold}_p{patient_index}",
                            "split": "test",
                            "arm": "N1",
                            "seed_base": 2026,
                            "fold": fold,
                            "pooling": "P0",
                            "feature_dim": 192,
                            "task": "nuisance",
                            "target_name": "padding_fraction",
                            "endpoint": endpoint,
                            "analysis_scope": "target_valid",
                            "target_semantics": "source_padding_fraction",
                            "y_true": truth,
                            "y_pred": 0.8 * truth + 0.5,
                            "b0_prediction": float(4 + fold + endpoint_index),
                            "test_predict_call_count": 1,
                        }
                    )
        return pd.DataFrame(rows)

    def test_pools_natural_predictions_before_metrics_and_calibration(self) -> None:
        predictions = self.predictions()
        metrics = pooled_oof_natural_metrics(predictions, expected_folds=(0, 1))
        self.assertEqual(len(metrics), 5)
        self.assertEqual(set(metrics["scale"]), {"natural"})
        self.assertEqual(
            set(metrics["aggregation"]),
            {"pooled_outer_test_folds", "mean_of_pooled_endpoint_metrics"},
        )
        t0 = metrics.loc[metrics["endpoint"].eq("T0")].iloc[0]
        self.assertAlmostEqual(float(t0["calibration_slope"]), 0.8, places=12)
        self.assertAlmostEqual(float(t0["calibration_intercept"]), 0.5, places=12)
        truth = predictions.loc[predictions["endpoint"].eq("T0"), "y_true"].to_numpy()
        prediction = predictions.loc[
            predictions["endpoint"].eq("T0"), "y_pred"
        ].to_numpy()
        expected_r2 = 1.0 - np.square(truth - prediction).sum() / np.square(
            truth - truth.mean()
        ).sum()
        self.assertAlmostEqual(float(t0["r2"]), float(expected_r2), places=12)
        macro = metrics.loc[metrics["endpoint"].eq("macro")].iloc[0]
        endpoint_rows = metrics.loc[~metrics["endpoint"].eq("macro")]
        self.assertAlmostEqual(
            float(macro["spearman"]), float(endpoint_rows["spearman"].mean())
        )
        self.assertAlmostEqual(float(macro["calibration_slope"]), 0.8, places=12)

    def test_rejects_missing_fold_or_cross_fold_duplicate_patient(self) -> None:
        predictions = self.predictions()
        with self.assertRaisesRegex(ValueError, "exact expected folds"):
            pooled_oof_natural_metrics(
                predictions.loc[predictions["fold"].eq(0)], expected_folds=(0, 1)
            )
        duplicate = predictions.copy()
        fold_one_t0 = duplicate.index[
            duplicate["fold"].eq(1) & duplicate["endpoint"].eq("T0")
        ][0]
        duplicate.loc[fold_one_t0, "patient_id"] = "f0_p0"
        with self.assertRaisesRegex(ValueError, "multiple outer-test folds"):
            pooled_oof_natural_metrics(duplicate, expected_folds=(0, 1))


class FrozenAssetAndOutputTests(unittest.TestCase):
    @staticmethod
    def write_asset(directory: Path, *, use_seed_base: bool = True) -> tuple[Path, Path]:
        directory.mkdir(parents=True, exist_ok=True)
        asset, _ = synthetic_asset(feature_dim=3)
        feature_path = directory / "plocal.private.npz"
        arrays = {
            "patient_id": asset.patient_id,
            "split": asset.split,
            "state": asset.state,
            "state_valid": asset.state_valid,
            "arm": np.asarray(asset.arm),
            "fold": np.asarray(asset.fold),
            "pooling": np.asarray(asset.pooling),
            ("seed_base" if use_seed_base else "seed"): np.asarray(asset.seed_base),
        }
        np.savez_compressed(feature_path, **arrays)
        metadata_path = feature_path.with_suffix(".metadata.json")
        metadata = {
            **asset.identity,
            "feature_path": str(feature_path.resolve()),
            "feature_sha256": file_sha256(feature_path),
            "state_shape": list(asset.state.shape),
            "state_valid_shape": list(asset.state_valid.shape),
            "patient_order_sha256": ordered_patient_sha256(asset.patient_id),
            "checkpoint_sha256": "a" * 64,
            "provenance": {"checkpoint_selected": True, "test_labels_used": False},
        }
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        return feature_path, metadata_path

    def test_loader_requires_seed_base_and_validates_same_stem_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            feature_path, metadata_path = self.write_asset(directory)
            asset = load_frozen_state_asset(feature_path)
            self.assertEqual(asset.seed_base, 2026)
            self.assertEqual(asset.feature_dim, 3)
            self.assertEqual(asset.metadata_path, metadata_path.resolve())

            wrong_path, _ = self.write_asset(directory / "wrong", use_seed_base=False)
            with self.assertRaisesRegex(ValueError, "schema drifted"):
                load_frozen_state_asset(wrong_path)

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["pooling"] = "P0"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity drifted"):
                load_frozen_state_asset(feature_path)

    def test_private_output_and_provenance_are_hash_bound_and_non_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            feature_path, _ = self.write_asset(directory)
            asset = load_frozen_state_asset(feature_path)
            targets = {
                "padding_fraction": {
                    str(patient_id): np.linspace(0.1, 0.4, 4)
                    + patient_index / 100.0
                    for patient_index, patient_id in enumerate(asset.patient_id)
                }
            }
            result = run_continuous_probe_cell(asset, targets)
            output = directory / "probe"
            provenance = {
                "preregistration_sha256": "b" * 64,
                "feature_extractor_sha256": "c" * 64,
            }
            metadata = write_probe_outputs(result, output, provenance=provenance)

            expected_names = {
                "ridge_selection.csv",
                "ridge_predictions.private.csv",
                "probe_metrics.csv",
                "probe_metadata.json",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected_names)
            for path in output.iterdir():
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            prediction_path = output / "ridge_predictions.private.csv"
            self.assertIn("patient_id", pd.read_csv(prediction_path).columns)
            self.assertTrue(metadata["patient_identifiers_private"])
            self.assertTrue(metadata["prediction_asset_private"])
            self.assertEqual(metadata["feature_sha256"], file_sha256(feature_path))
            self.assertEqual(metadata["provenance"], provenance)
            for name, digest in metadata["output_sha256"].items():
                self.assertEqual(digest, file_sha256(output / name))
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                write_probe_outputs(result, output, provenance=provenance)


if __name__ == "__main__":
    unittest.main()

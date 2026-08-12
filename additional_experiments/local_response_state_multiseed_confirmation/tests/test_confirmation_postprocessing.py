from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import stat
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


sys.dont_write_bytecode = True
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = EXPERIMENT_ROOT / "src"
SEALED_SRC = EXPERIMENT_ROOT.parent / "c1b_overlap_eligibility_ftv_stageb" / "src"
for source in (SRC_ROOT, SEALED_SRC):
    value = str(source.resolve())
    while value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)

from lg_response_pilot import features, probes  # noqa: E402


ARMS = ("GAP0", "GAP3", "LOCAL0", "LOCAL3")
SEEDS = (2026, 3026, 4026, 5026, 6026)
FOLDS = tuple(range(5))
DATA_CONTRACT_KEY = (
    "additional_experiments/c1b_overlap_eligibility_ftv_stageb/"
    "manifests/stage_b_data_contract.private.json"
)


def load_script(name: str):
    path = EXPERIMENT_ROOT / "scripts" / name
    module_name = f"confirmation_{path.stem}_test"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class ConfirmationFeatureProbeContractTest(unittest.TestCase):
    def test_feature_and_probe_identity_is_exactly_four_arms_by_five_seeds(self) -> None:
        self.assertEqual(features.ARMS, ARMS)
        self.assertEqual(features.SEED_BASES, SEEDS)
        self.assertEqual(features.FOLDS, FOLDS)
        self.assertEqual(probes.ARMS, ARMS)
        self.assertEqual(probes.SEED_BASES, SEEDS)
        self.assertEqual(probes.FOLDS, FOLDS)

    def test_feature_asset_accepts_fifth_seed_and_rejects_retired_global_arm(self) -> None:
        patient_ids = np.asarray(["train", "validation", "test"])
        split = np.asarray(["train", "val", "test"])
        response = np.zeros((3, 4, 192), dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            accepted = Path(directory) / "accepted.private.npz"
            np.savez_compressed(
                accepted,
                patient_id=patient_ids,
                split=split,
                response_state=response,
                arm=np.asarray("LOCAL3"),
                seed_base=np.asarray(6026, dtype=np.int64),
                fold=np.asarray(4, dtype=np.int64),
            )
            loaded = probes.load_feature_asset(accepted)
            self.assertEqual(str(loaded["arm"].item()), "LOCAL3")
            self.assertEqual(int(loaded["seed_base"].item()), 6026)

            retired = Path(directory) / "retired.private.npz"
            np.savez_compressed(
                retired,
                patient_id=patient_ids,
                split=split,
                response_state=response,
                arm=np.asarray("LG3"),
                seed_base=np.asarray(2026, dtype=np.int64),
                fold=np.asarray(0, dtype=np.int64),
            )
            with self.assertRaisesRegex(ValueError, "confirmation matrix"):
                probes.load_feature_asset(retired)

    def test_feature_metadata_requires_confirmation_experiment_identity(self) -> None:
        arrays = {
            "patient_id": np.asarray(["train", "validation", "test"]),
            "split": np.asarray(["train", "val", "test"]),
            "response_state": np.zeros((3, 4, 192), dtype=np.float32),
            "arm": np.asarray("GAP0"),
            "seed_base": np.asarray(4026, dtype=np.int64),
            "fold": np.asarray(2, dtype=np.int64),
        }
        lock_sha256 = "a" * 64
        provenance = {"contract": "sealed"}

        class Authorization:
            sha256 = "b" * 64

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature_path = root / "response_state.private.npz"
            checkpoint_path = root / "selected.pt"
            selection_path = root / "selection.json"
            feature_path.write_bytes(b"feature")
            checkpoint_path.write_bytes(b"checkpoint")
            selection_path.write_text("{}", encoding="utf-8")
            metadata_path = feature_path.with_suffix(".metadata.json")
            payload = {
                "schema_version": 1,
                "experiment": "local_response_state_multiseed_confirmation",
                "arm": "GAP0",
                "seed_base": 4026,
                "fold": 2,
                "feature_tensor": "online_preprojector_response_state",
                "feature_dtype": "float32",
                "feature_shape": [3, 4, 192],
                "cohort": "exact_locked_primary_train_validation_test",
                "stage_a_sentinel_sha256": Authorization.sha256,
                "ftv_head_called": False,
                "test_labels_used": False,
                "preregistration_lock": "PREREGISTRATION_LOCK.json",
                "preregistration_lock_sha256": lock_sha256,
                "feature_path": str(feature_path.resolve()),
                "feature_sha256": features.file_sha256(feature_path),
                "patient_order_sha256": features.ordered_patient_sha256(
                    arrays["patient_id"]
                ),
                "current_data_contract_provenance_sha256": features.canonical_sha256(
                    provenance
                ),
                "checkpoint_path": str(checkpoint_path.resolve()),
                "checkpoint_sha256": features.file_sha256(checkpoint_path),
                "selection_path": str(selection_path.resolve()),
                "selection_sha256": features.file_sha256(selection_path),
            }
            metadata_path.write_text(json.dumps(payload), encoding="utf-8")
            observed_path, observed = probes.validate_feature_metadata(
                feature_path,
                arrays,
                Authorization(),
                provenance,
                lock_sha256,
            )
            self.assertEqual(observed_path, metadata_path)
            self.assertEqual(
                observed["experiment"],
                "local_response_state_multiseed_confirmation",
            )
            payload["experiment"] = "local_global_response_state_pilot"
            metadata_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs at experiment"):
                probes.validate_feature_metadata(
                    feature_path,
                    arrays,
                    Authorization(),
                    provenance,
                    lock_sha256,
                )

    def test_ridge_selection_and_test_prediction_remain_outer_split_sealed(self) -> None:
        train_matrix = np.arange(24, dtype=np.float64).reshape(8, 3)
        train_target = np.linspace(-1.0, 1.0, 8)
        validation_matrix = np.arange(12, dtype=np.float64).reshape(4, 3) + 0.5
        validation_target = np.linspace(-0.5, 0.5, 4)
        selected = probes.select_ridge(
            train_matrix,
            train_target,
            validation_matrix,
            validation_target,
            probes.ALPHAS,
            standardize_target=True,
        )
        self.assertIn(selected.alpha, probes.ALPHAS)
        self.assertEqual(int(selected.x_scaler.n_samples_seen_), len(train_matrix))
        guard = probes.TestPredictGuard()
        transformed = selected.x_scaler.transform(validation_matrix)
        guard.predict(selected.model, transformed)
        with self.assertRaisesRegex(RuntimeError, "single-use"):
            guard.predict(selected.model, transformed)

    def test_metric_formulas_preserve_natural_r2_and_calibration_orientation(self) -> None:
        truth = np.asarray([0.0, 1.0, 2.0, 3.0])
        prediction = 0.25 + 0.5 * truth
        observed = probes.metric_values(truth, prediction, np.mean(truth))
        self.assertAlmostEqual(observed["calibration_slope"], 0.5)
        self.assertAlmostEqual(observed["calibration_intercept"], 0.25)
        self.assertAlmostEqual(observed["prediction_target_variance_ratio"], 0.25)
        self.assertTrue(np.isfinite(observed["r2"]))

    def test_feature_and_probe_writers_remain_private(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = (
                root / "response_state.private.npz",
                root / "response_state.private.metadata.json",
                root / "ridge_predictions.private.csv",
                root / "probe_metadata.json",
            )
            features._atomic_npz(paths[0], value=np.asarray([1], dtype=np.int64))
            features._atomic_json(paths[1], {"status": "ok"}, private=True)
            probes._atomic_csv(paths[2], pd.DataFrame({"value": [1]}))
            probes._atomic_json(paths[3], {"status": "ok"})
            for path in paths:
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_feature_export_refuses_to_overwrite_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "response_state.private.npz"
            output.write_bytes(b"existing")
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                features.export_response_features(
                    checkpoint_path=root / "selected.pt",
                    arm="LOCAL3",
                    seed_base=6026,
                    fold=4,
                    data=None,
                    authorization=None,
                    output_path=output,
                    device=features.torch.device("cpu"),
                    preregistration_lock_sha256="a" * 64,
                    workers=0,
                )

    def test_probe_export_refuses_to_overwrite_any_existing_output(self) -> None:
        lock_sha256 = "a" * 64
        provenance = {"contract": "sealed"}

        class Authorization:
            sha256 = "b" * 64

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature_path = root / "response_state.private.npz"
            checkpoint_path = root / "selected.pt"
            selection_path = root / "selection.json"
            patient_ids = np.asarray(["train", "validation", "test"])
            splits = np.asarray(["train", "val", "test"])
            response = np.zeros((3, 4, 192), dtype=np.float32)
            np.savez_compressed(
                feature_path,
                patient_id=patient_ids,
                split=splits,
                response_state=response,
                arm=np.asarray("GAP0"),
                seed_base=np.asarray(5026, dtype=np.int64),
                fold=np.asarray(3, dtype=np.int64),
            )
            checkpoint_path.write_bytes(b"checkpoint")
            selection_path.write_text("{}", encoding="utf-8")
            metadata = {
                "schema_version": 1,
                "experiment": "local_response_state_multiseed_confirmation",
                "arm": "GAP0",
                "seed_base": 5026,
                "fold": 3,
                "feature_tensor": "online_preprojector_response_state",
                "feature_dtype": "float32",
                "feature_shape": [3, 4, 192],
                "cohort": "exact_locked_primary_train_validation_test",
                "stage_a_sentinel_sha256": Authorization.sha256,
                "ftv_head_called": False,
                "test_labels_used": False,
                "preregistration_lock": "PREREGISTRATION_LOCK.json",
                "preregistration_lock_sha256": lock_sha256,
                "feature_path": str(feature_path.resolve()),
                "feature_sha256": features.file_sha256(feature_path),
                "patient_order_sha256": features.ordered_patient_sha256(patient_ids),
                "current_data_contract_provenance_sha256": features.canonical_sha256(
                    provenance
                ),
                "checkpoint_path": str(checkpoint_path.resolve()),
                "checkpoint_sha256": features.file_sha256(checkpoint_path),
                "selection_path": str(selection_path.resolve()),
                "selection_sha256": features.file_sha256(selection_path),
            }
            feature_path.with_suffix(".metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            folds = pd.DataFrame(
                {
                    "patient_id": patient_ids,
                    "fold": [3, 3, 3],
                    "split": splits,
                }
            )
            output_dir = root / "probes"
            output_dir.mkdir()
            (output_dir / "probe_metrics.csv").write_text(
                "existing\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                probes.run_ftv_probes(
                    feature_path=feature_path,
                    records={},
                    folds=folds,
                    authorization=Authorization(),
                    data_provenance=provenance,
                    output_dir=output_dir,
                    preregistration_lock_sha256=lock_sha256,
                )


class ConfirmationPostprocessingContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scripts = tuple(
            load_script(name)
            for name in (
                "export_features.py",
                "run_probes.py",
                "run_postprocessing.py",
            )
        )

    def test_external_private_contract_resolution_keeps_relative_lock_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            external_root = Path(directory).resolve()
            for module in self.scripts:
                self.assertEqual(module.DATA_CONTRACT_LOCK_KEY, DATA_CONTRACT_KEY)
                self.assertEqual(
                    module._resolve_default_data_contract(
                        {module.PRIVATE_INPUT_REPO_ROOT_ENV: str(external_root)}
                    ),
                    external_root / DATA_CONTRACT_KEY,
                )
                self.assertEqual(
                    module._resolve_default_data_contract(
                        {module.PRIVATE_INPUT_REPO_ROOT_ENV: f"  {external_root}  "}
                    ),
                    external_root / DATA_CONTRACT_KEY,
                )
                self.assertEqual(
                    module._resolve_default_data_contract({}),
                    module.REPO_ROOT.resolve() / DATA_CONTRACT_KEY,
                )
                source = Path(module.__file__).read_text(encoding="utf-8")
                self.assertIn("locked_upstream[DATA_CONTRACT_LOCK_KEY]", source)
                self.assertNotIn("DEFAULT_DATA_CONTRACT.relative_to(REPO_ROOT)", source)

    def test_postprocessing_grid_is_exactly_one_hundred_fresh_cells(self) -> None:
        postprocessing = self.scripts[-1]
        config = json.loads(
            (EXPERIMENT_ROOT / "configs" / "confirmation.json").read_text(
                encoding="utf-8"
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cells = postprocessing._cells(
                root / "checkpoints",
                root / "features",
                root / "predictions",
                config,
                ("cuda:0", "cuda:1", "cuda:2"),
            )
        identities = {(cell.seed, cell.arm, cell.fold) for cell in cells}
        expected = {
            (seed, arm, fold)
            for seed in SEEDS
            for arm in ARMS
            for fold in FOLDS
        }
        self.assertEqual(len(cells), 100)
        self.assertEqual(identities, expected)
        self.assertEqual(len(identities), 100)

    def test_matrix_completion_gate_requires_all_one_hundred_cells(self) -> None:
        postprocessing = self.scripts[-1]
        config = json.loads(
            (EXPERIMENT_ROOT / "configs" / "confirmation.json").read_text(
                encoding="utf-8"
            )
        )
        lock_sha256 = "a" * 64
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_root = root / "checkpoints"
            cells = postprocessing._cells(
                checkpoint_root,
                root / "features",
                root / "predictions",
                config,
                ("cuda:0", "cuda:1", "cuda:2"),
            )
            runs: list[dict[str, object]] = []
            for cell in cells:
                cell.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                selection = {
                    "seed_base": cell.seed,
                    "arm": cell.arm,
                    "fold": cell.fold,
                    "test_data_used": False,
                    "preregistration_status": "PASS",
                    "preregistration_lock_sha256": lock_sha256,
                    "preregistration": {
                        "status": "PASS",
                        "lock_sha256": lock_sha256,
                    },
                }
                cell.selection.write_text(json.dumps(selection), encoding="utf-8")
                cell.history.write_text("epoch\n1\n", encoding="utf-8")
                cell.checkpoint.write_bytes(b"checkpoint")
                runs.append(
                    {
                        "seed_base": cell.seed,
                        "arm": cell.arm,
                        "fold": cell.fold,
                        "selection_path": str(cell.selection.resolve()),
                    }
                )
            completion = {
                "schema_version": 1,
                "status": "COMPLETE",
                "run_count": 100,
                "preregistration": {
                    "status": "PASS",
                    "lock_sha256": lock_sha256,
                },
                "config_sha256": "b" * 64,
                "stage_a_sentinel_sha256": "c" * 64,
                "data_contract_sha256": "d" * 64,
                "runs": runs,
            }
            completion_path = checkpoint_root / "matrix_complete.json"
            completion_path.write_text(json.dumps(completion), encoding="utf-8")
            observed = postprocessing._validate_matrix(
                checkpoint_root,
                cells,
                lock_sha256,
                "b" * 64,
                "c" * 64,
                "d" * 64,
            )
            self.assertEqual(observed["run_count"], 100)

            completion["run_count"] = 60
            completion_path.write_text(json.dumps(completion), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not authorize 100 cells"):
                postprocessing._validate_matrix(
                    checkpoint_root,
                    cells,
                    lock_sha256,
                    "b" * 64,
                    "c" * 64,
                    "d" * 64,
                )

    def test_postprocessing_rejects_any_non_confirmation_matrix(self) -> None:
        postprocessing = self.scripts[-1]
        config = json.loads(
            (EXPERIMENT_ROOT / "configs" / "confirmation.json").read_text(
                encoding="utf-8"
            )
        )
        config["training"]["seed_bases"] = list(SEEDS[:-1])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "exact confirmation matrix"):
                postprocessing._cells(
                    root / "checkpoints",
                    root / "features",
                    root / "predictions",
                    config,
                    ("cuda:0", "cuda:1", "cuda:2"),
                )
        config = json.loads(
            (EXPERIMENT_ROOT / "configs" / "confirmation.json").read_text(
                encoding="utf-8"
            )
        )
        config["arms"] = {
            **{arm: config["arms"][arm] for arm in ARMS[:-1]},
            "LG3": config["arms"]["LOCAL3"],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "exact confirmation matrix"):
                postprocessing._cells(
                    root / "checkpoints",
                    root / "features",
                    root / "predictions",
                    config,
                    ("cuda:0", "cuda:1", "cuda:2"),
                )


if __name__ == "__main__":
    unittest.main()

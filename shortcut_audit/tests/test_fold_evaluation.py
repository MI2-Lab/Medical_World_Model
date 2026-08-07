from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ispy_jepa_tmi_clean.corejepa.config import ExperimentConfig
from ispy_jepa_tmi_clean.corejepa.data.condition import ConditionEncoder
from ispy_jepa_tmi_clean.corejepa.data.records import PatientRecord
from ispy_jepa_tmi_clean.corejepa.models import CoReJEPA

from shortcut_audit.auditlib.baseline_models import BaselineReadoutConfig
from shortcut_audit.auditlib import fold_evaluation as fold_evaluation_module
from shortcut_audit.auditlib.fold_evaluation import (
    PERTURBATIONS,
    evaluate_retrained_fold,
    load_validated_fold_inputs,
)
from shortcut_audit.auditlib.provenance import file_sha256
from shortcut_audit.auditlib.readouts import AuditReadoutConfig
from shortcut_audit.auditlib.training import AUDIT_PROTOCOL, LegacyXCacheDataset


class SyntheticFold:
    """A complete tiny fold artifact set; no training is run by these tests."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.fold = 2
        self.fold_dir = root / "fold_02"
        self.cache_dir = root / "legacy_x"
        self.output_dir = root / "evaluation"
        self.fold_dir.mkdir()
        self.cache_dir.mkdir()
        self.records = self._records()
        self.n_primary = len(self.records)
        self.splits = {
            "primary_train": list(range(0, 6)),
            "pretrain_train": list(range(0, 6)),
            "validation": list(range(6, 10)),
            "test": list(range(10, 14)),
        }
        self.config = self._config()
        self.encoder = ConditionEncoder(
            [self.records[index] for index in self.splits["primary_train"]]
        )
        self._write_cache()
        torch.manual_seed(20260806)
        self.model = CoReJEPA(self.config.model, self.encoder.spec.dim).eval()
        self._write_artifacts()

    @staticmethod
    def _records() -> list[PatientRecord]:
        return [
            PatientRecord(
                patient_id=f"P{index:02d}",
                cohort="ispy2",
                arm="Paclitaxel",
                hr=index % 2,
                her2=(index // 2) % 2,
                mp=(index // 3) % 2,
                age=35.0 + index,
                manifest_path=Path(f"/unused/P{index:02d}/manifest.json"),
                pcr=index % 2,
            )
            for index in range(14)
        ]

    def _config(self) -> ExperimentConfig:
        config = ExperimentConfig()
        config.data.crop_size = (8, 8, 8)
        config.model.base_channels = 1
        config.model.latent_dim = 8
        config.model.predictor_depth = 1
        config.model.predictor_heads = 2
        config.model.predictor_mlp_dim = 16
        config.model.response_dim = 4
        config.model.response_hidden_dim = 8
        config.model.response_depth = 1
        config.model.expert_hidden_dim = 8
        config.model.expert_gate_hidden_dim = 8
        config.model.dropout = 0.0
        config.train.output_dir = str(self.fold_dir)
        config.train.seed = 2026 + self.fold
        config.train.batch_size = 4
        config.train.workers = 0
        config.readout.penalties = ("l2",)
        config.readout.c_grid = (0.1,)
        config.readout.max_iter = 200
        return config

    def _write_cache(self) -> None:
        rng = np.random.default_rng(20260806)
        for index, record in enumerate(self.records):
            image = rng.normal(0.0, 0.2, size=(4, 8, 8, 8, 8)).astype(np.float32)
            image[:, 7] = 0.0
            for visit in range(4):
                width = 2 + ((index + visit) % 4)
                start = (index + 2 * visit) % (8 - width + 1)
                image[visit, 7, 1:4, 2:6, start : start + width] = 1.0
            np.savez_compressed(
                self.cache_dir / f"{record.patient_id}_dce8_smoke.npz",
                x=image,
            )

    def _write_artifacts(self) -> None:
        manifest = self.root / "fold_manifest.csv"
        test_fold = {
            record.patient_id: (self.fold if index >= 10 else (0, 1, 3, 4)[index % 4])
            for index, record in enumerate(self.records)
        }
        manifest_rows: list[dict[str, object]] = []
        for current_fold in range(5):
            for index, record in enumerate(self.records):
                if test_fold[record.patient_id] == current_fold:
                    split = "test"
                elif current_fold == self.fold:
                    split = "val" if index in self.splits["validation"] else "train"
                else:
                    split = "val" if (index + current_fold) % 3 == 0 else "train"
                manifest_rows.append(
                    {
                        "patient_id": record.patient_id,
                        "fold": current_fold,
                        "split": split,
                        "label_pcr": int(record.pcr),
                    }
                )
        pd.DataFrame(manifest_rows).to_csv(manifest, index=False)
        split_ids = {
            "train": [self.records[index].patient_id for index in self.splits["primary_train"]],
            "val": [self.records[index].patient_id for index in self.splits["validation"]],
            "test": [self.records[index].patient_id for index in self.splits["test"]],
        }
        payload = {
            "model": self.model.state_dict(),
            "config": self.config.to_dict(),
            "condition": {
                "feature_names": list(self.encoder.spec.feature_names),
                "arm_vocab": dict(self.encoder.spec.arm_vocab),
                "age_mean": self.encoder.spec.age_mean,
                "age_std": self.encoder.spec.age_std,
            },
            "response_transform": {},
            "patient_ids": [record.patient_id for record in self.records],
            "n_primary": self.n_primary,
            "splits": self.splits,
            "epoch": 1,
            "validation": {"prediction": 1.0},
            "audit_provenance": {
                "schema_version": 1,
                "protocol": AUDIT_PROTOCOL,
                "fold": self.fold,
                "manifest": {
                    "path": str(manifest),
                    "sha256": file_sha256(manifest),
                    "hash_kind": "file_sha256",
                    "n_rows": len(manifest_rows),
                    "n_patients": self.n_primary,
                },
                "primary_split_patient_ids": split_ids,
                "extra_pretraining_patient_ids": [],
                "fit_scopes": {},
                "fit_indices": {
                    "condition_encoder": list(self.splits["pretrain_train"]),
                    "response_transform": list(self.splits["pretrain_train"]),
                },
                "base_dataset_source": {
                    "mode": "verified_legacy_x_adapter",
                    "cache_dir": str(self.cache_dir.resolve()),
                    "n_patient_files": self.n_primary,
                    "image_key": "x",
                    "geometry": "clean mask_geometry(x[:,7])",
                },
                "response_target_source": {"mode": "unit_test"},
                "implementation": {},
                "output_dir": str(self.fold_dir),
                "seed": 2026 + self.fold,
            },
        }
        torch.save(payload, self.fold_dir / "best_corejepa.pt")
        (self.fold_dir / "splits.json").write_text(json.dumps(self.splits))

        dataset = LegacyXCacheDataset(
            self.records,
            self.encoder,
            self.cache_dir,
            expected_crop_size=self.config.data.crop_size,
        )
        states: list[np.ndarray] = []
        image_predictions: list[np.ndarray] = []
        ids: list[str] = []
        with torch.no_grad():
            for batch in DataLoader(dataset, batch_size=4, shuffle=False):
                state = self.model.forecast_response(batch["geometry"], batch["condition"])
                visits = self.model.encode_visits(batch["image"], batch["geometry"])
                prediction = self.model.image_transition(visits[:, :-1], batch["condition"])
                states.append(state.numpy())
                image_predictions.append(prediction.numpy())
                ids.extend(str(value) for value in batch["patient_id"])
        np.savez_compressed(
            self.fold_dir / "frozen_states.npz",
            patient_ids=np.asarray(ids),
            future_response_state=np.concatenate(states).astype(np.float32),
            image_prediction=np.concatenate(image_predictions).astype(np.float32),
            pcr=np.asarray([record.pcr for record in self.records], dtype=np.int64),
            n_primary=np.asarray(self.n_primary, dtype=np.int64),
        )

    @contextmanager
    def patch_records(self):
        with mock.patch(
            "shortcut_audit.auditlib.fold_evaluation.clean_runner.load_experiment_records",
            return_value=(list(self.records), self.n_primary),
        ):
            yield


def _readout_config() -> AuditReadoutConfig:
    return AuditReadoutConfig(
        penalties=("l2",),
        c_grid=(0.1,),
        max_iter=200,
        random_state=7,
    )


def _baseline_config() -> BaselineReadoutConfig:
    return BaselineReadoutConfig(
        penalties=("l2",),
        c_grid=(0.1,),
        max_iter=200,
        random_state=7,
    )


class FoldEvaluationSmokeTest(unittest.TestCase):
    def test_full_fold_orchestration_outputs_and_donor_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticFold(Path(directory))
            fold_files_before = sorted(path.name for path in fixture.fold_dir.iterdir())
            with fixture.patch_records():
                result = evaluate_retrained_fold(
                    fixture.fold_dir,
                    fold=fixture.fold,
                    legacy_x_cache_dir=fixture.cache_dir,
                    output_dir=fixture.output_dir,
                    device="cpu",
                    batch_size=4,
                    workers=0,
                    readout_config=_readout_config(),
                    baseline_config=_baseline_config(),
                )

            self.assertEqual(result.fold, fixture.fold)
            self.assertEqual(set(result.baseline_bundles), {"F1", "F2", "F3", "F4", "F5"})
            self.assertEqual(set(result.paired_metrics), set(PERTURBATIONS))
            self.assertEqual(len(result.predictions["native"]), 4 * 3)
            self.assertEqual(len(result.predictions["perturbations"]), 4 * 3 * 3)
            self.assertEqual(len(result.predictions["baselines"]), 4 * 3 * 5)
            self.assertEqual(len(result.copy_current_metrics), 4 * 3)
            self.assertTrue(all(path.is_file() for path in result.artifact_paths.values()))
            self.assertTrue((fixture.output_dir / "evaluation_provenance.json").is_file())
            self.assertEqual(
                result.baseline_bundles["F5"].feature_provenance["appearance_extractor"],
                "model.projector(model.encoder(image[:,0]))",
            )
            self.assertFalse(
                result.baseline_bundles["F5"].feature_provenance["geometry_in_t0_representation"]
            )

            native = result.predictions["native"].sort_values(
                ["patient_id", "decision_point"]
            )
            c1 = result.predictions["perturbations"].loc[
                lambda frame: frame["audit_condition"].eq("repeated_t0_c1_mri_only")
            ].sort_values(["patient_id", "decision_point"])
            np.testing.assert_allclose(
                native["predicted_probability"],
                c1["predicted_probability"],
                rtol=1e-5,
                atol=1e-6,
            )
            self.assertTrue(
                result.paired_metrics["repeated_t0_c1_mri_only"][
                    "response_state_l2_change"
                ].eq(0.0).all()
            )

            heldout = list(result.donor_context.patient_index)
            mapping = pd.DataFrame(
                {
                    "recipient_patient_id": heldout,
                    "donor_patient_id": heldout[1:] + heldout[:1],
                    "fold": fixture.fold,
                    "audit_repetition": 1,
                    "matching_distance": 0.5,
                }
            )
            pairs = result.donor_context.build_pair_dataset(mapping)
            item = pairs[0]
            self.assertEqual(len(pairs), len(heldout))
            self.assertNotIn("pcr", item)
            self.assertEqual(set(result.donor_context.labels_by_patient.values()), {0, 1})

            self.assertEqual(
                fold_files_before,
                sorted(path.name for path in fixture.fold_dir.iterdir()),
            )

            failed_output = fixture.root / "failed_evaluation"
            with fixture.patch_records():
                validated = load_validated_fold_inputs(
                    fixture.fold_dir,
                    fold=fixture.fold,
                    legacy_x_cache_dir=fixture.cache_dir,
                )
            with mock.patch.object(
                fold_evaluation_module,
                "_write_json",
                side_effect=RuntimeError("injected write failure"),
            ), self.assertRaisesRegex(RuntimeError, "injected"):
                fold_evaluation_module._write_outputs(
                    failed_output,
                    validated=validated,
                    readout=result.readout_bundle,
                    baselines=result.baseline_bundles,
                    predictions=result.predictions,
                    copy_current=result.copy_current_metrics,
                    paired=result.paired_metrics,
                )
            self.assertFalse(failed_output.exists())
            self.assertEqual(
                list(fixture.root.glob(".failed_evaluation.staging-*")),
                [],
            )

            with fixture.patch_records(), self.assertRaisesRegex(FileExistsError, "覆盖"):
                evaluate_retrained_fold(
                    fixture.fold_dir,
                    fold=fixture.fold,
                    legacy_x_cache_dir=fixture.cache_dir,
                    output_dir=fixture.output_dir,
                )

    def test_split_fold_and_label_drift_fail_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticFold(Path(directory))
            checkpoint = fixture.fold_dir / "best_corejepa.pt"
            original_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)

            wrong_fold = dict(original_payload)
            wrong_fold["audit_provenance"] = dict(original_payload["audit_provenance"])
            wrong_fold["audit_provenance"]["fold"] = 1
            torch.save(wrong_fold, checkpoint)
            with fixture.patch_records(), self.assertRaisesRegex(ValueError, "fold"):
                load_validated_fold_inputs(
                    fixture.fold_dir,
                    fold=fixture.fold,
                    legacy_x_cache_dir=fixture.cache_dir,
                )
            self.assertFalse(fixture.output_dir.exists())

            torch.save(original_payload, checkpoint)
            split_path = fixture.fold_dir / "splits.json"
            wrong_splits = {key: list(value) for key, value in fixture.splits.items()}
            wrong_splits["test"] = list(reversed(wrong_splits["test"]))
            split_path.write_text(json.dumps(wrong_splits))
            with fixture.patch_records(), self.assertRaisesRegex(ValueError, "顺序或内容"):
                load_validated_fold_inputs(
                    fixture.fold_dir,
                    fold=fixture.fold,
                    legacy_x_cache_dir=fixture.cache_dir,
                )

            split_path.write_text(json.dumps(fixture.splits))
            frozen_path = fixture.fold_dir / "frozen_states.npz"
            with np.load(frozen_path, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            arrays["pcr"] = arrays["pcr"].copy()
            arrays["pcr"][0] = 1
            np.savez_compressed(frozen_path, **arrays)
            with fixture.patch_records(), self.assertRaisesRegex(ValueError, "label/order"):
                load_validated_fold_inputs(
                    fixture.fold_dir,
                    fold=fixture.fold,
                    legacy_x_cache_dir=fixture.cache_dir,
                )
            self.assertFalse(fixture.output_dir.exists())

            arrays["pcr"][0] = 0
            np.savez_compressed(frozen_path, **arrays)
            cache_path = fixture.cache_dir / "P00_dce8_smoke.npz"
            with np.load(cache_path, allow_pickle=False) as archive:
                image = archive["x"].copy()
            image[:, 0] += 10.0
            np.savez_compressed(cache_path, x=image)
            with fixture.patch_records(), self.assertRaisesRegex(
                RuntimeError, "image_prediction"
            ):
                evaluate_retrained_fold(
                    fixture.fold_dir,
                    fold=fixture.fold,
                    legacy_x_cache_dir=fixture.cache_dir,
                    output_dir=fixture.output_dir,
                    readout_config=_readout_config(),
                    baseline_config=_baseline_config(),
                )
            self.assertFalse(fixture.output_dir.exists())


if __name__ == "__main__":
    unittest.main()

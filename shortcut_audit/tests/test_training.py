from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset

from ispy_jepa_tmi_clean.corejepa.config import ExperimentConfig, load_config
from ispy_jepa_tmi_clean.corejepa.data.condition import ConditionEncoder
from ispy_jepa_tmi_clean.corejepa.data.imaging import mask_geometry
from ispy_jepa_tmi_clean.corejepa.data.records import PatientRecord
from ispy_jepa_tmi_clean.corejepa.models import CoReJEPA
from ispy_jepa_tmi_clean.corejepa.training import runner as clean_runner

from shortcut_audit.auditlib.provenance import validate_checkpoint_payload
from shortcut_audit.auditlib.training import (
    DEFAULT_SEED2026_MANIFEST,
    SEED2026_MANIFEST_SHA256,
    LegacyXCacheDataset,
    PreparedFoldTraining,
    build_fold_training_plan,
    checkpoint_payload_with_provenance,
    fit_fold_preprocessors,
    make_smoke_config,
    prepare_fold_training,
    train_explicit_fold,
    train_fivefold,
)


def record(
    patient_id: str,
    *,
    cohort: str = "ispy2",
    pcr: int | None = 0,
    arm: str = "Paclitaxel",
    age: float = 40.0,
) -> PatientRecord:
    return PatientRecord(
        patient_id=patient_id,
        cohort=cohort,
        arm=arm,
        hr=1,
        her2=0,
        mp=1,
        age=age,
        manifest_path=Path(f"/{patient_id}/manifest.json"),
        pcr=pcr,
    )


def records() -> tuple[list[PatientRecord], int]:
    primary = [record(f"P{index}", pcr=index % 2, age=20.0 + 10.0 * index) for index in range(5)]
    extra = [
        record("E0", cohort="ispy1", pcr=None, arm="ISPY1_NACT", age=70.0),
        record("E1", cohort="ispy1", pcr=None, arm="ISPY1_NACT", age=80.0),
    ]
    return primary + extra, len(primary)


def manifest() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold in range(5):
        for index in range(5):
            split = "test" if index == fold else "val" if index == (fold + 1) % 5 else "train"
            rows.append(
                {
                    "patient_id": f"P{index}",
                    "fold": fold,
                    "split": split,
                    "label_pcr": index % 2,
                }
            )
    return pd.DataFrame(rows)


def raw_response(n: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 3, 18)).astype(np.float32)


class EmptyBase(Dataset):
    def __init__(self, source_records: list[PatientRecord]) -> None:
        self.records = source_records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        raise AssertionError("unit test should not materialize image tensors")


def prepared_fixture(output_root: Path) -> PreparedFoldTraining:
    source, n_primary = records()
    plan = build_fold_training_plan(
        source,
        n_primary,
        manifest(),
        0,
        output_root=output_root,
    )
    blinded, encoder, transform, vector, score = fit_fold_preprocessors(
        source,
        plan,
        raw_response(len(source)),
    )
    from ispy_jepa_tmi_clean.corejepa.data.dataset import PretrainingDataset

    config = ExperimentConfig()
    config.train.output_dir = str(plan.output_dir)
    dataset = PretrainingDataset(EmptyBase(blinded), vector, score)  # type: ignore[arg-type]
    return PreparedFoldTraining(
        config=config,
        dataset=dataset,
        records=source,
        pretraining_records=blinded,
        n_primary=n_primary,
        plan=plan,
        condition_encoder=encoder,
        response_transform=transform,
        base_dataset_source={"mode": "unit_test"},
        response_target_source={"mode": "unit_test"},
    )


class FoldTrainingPlanTest(unittest.TestCase):
    def test_explicit_fold_and_ispy1_pretraining_indices(self) -> None:
        source, n_primary = records()
        plan = build_fold_training_plan(
            source,
            n_primary,
            manifest(),
            0,
            output_root="audit-runs",
        )
        self.assertEqual(plan.primary_train, (2, 3, 4))
        self.assertEqual(plan.validation, (1,))
        self.assertEqual(plan.test, (0,))
        self.assertEqual(plan.pretrain_train, (2, 3, 4, 5, 6))
        self.assertEqual(plan.extra_pretraining_indices, (5, 6))
        self.assertEqual(plan.output_dir, Path("audit-runs/fold_00"))
        self.assertEqual(plan.split_patient_ids()["test"], ["P0"])

    def test_fold_output_directories_are_independent(self) -> None:
        source, n_primary = records()
        paths = {
            build_fold_training_plan(source, n_primary, manifest(), fold, output_root="root").output_dir
            for fold in range(5)
        }
        self.assertEqual(len(paths), 5)
        self.assertEqual(paths, {Path(f"root/fold_{fold:02d}") for fold in range(5)})

    def test_manifest_label_mismatch_is_rejected(self) -> None:
        source, n_primary = records()
        bad = manifest()
        bad.loc[(bad.patient_id == "P0") & (bad.fold == 0), "label_pcr"] = 1
        with self.assertRaisesRegex(ValueError, "label"):
            build_fold_training_plan(source, n_primary, bad, 0, output_root="root")

    def test_non_ispy1_extra_is_rejected(self) -> None:
        source, n_primary = records()
        source[-1] = record("E1", cohort="ispy2", pcr=None)
        with self.assertRaisesRegex(ValueError, "仅允许 I-SPY1"):
            build_fold_training_plan(source, n_primary, manifest(), 0, output_root="root")

    def test_file_manifest_hash_is_enforced(self) -> None:
        source, n_primary = records()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "folds.csv"
            manifest().to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "SHA256"):
                build_fold_training_plan(
                    source,
                    n_primary,
                    path,
                    0,
                    output_root=directory,
                    expected_manifest_sha256="0" * 64,
                )


class LeakageBarrierTest(unittest.TestCase):
    def test_fit_scope_is_pretrain_only_and_outcomes_are_redacted(self) -> None:
        source, n_primary = records()
        plan = build_fold_training_plan(source, n_primary, manifest(), 0, output_root="root")
        blinded, encoder, _, vector, score = fit_fold_preprocessors(
            source,
            plan,
            raw_response(len(source)),
        )
        self.assertTrue(all(item.pcr is None for item in blinded))
        expected_age_mean = np.mean([40.0, 50.0, 60.0, 70.0, 80.0])
        self.assertAlmostEqual(encoder.spec.age_mean, expected_age_mean)
        self.assertNotAlmostEqual(encoder.spec.age_mean, np.mean([item.age for item in source]))
        self.assertEqual(vector.shape, (7, 3, 18))
        self.assertEqual(score.shape, (7, 3, 1))

    def test_test_response_rows_do_not_change_transform_fit(self) -> None:
        source, n_primary = records()
        plan = build_fold_training_plan(source, n_primary, manifest(), 0, output_root="root")
        first = raw_response(len(source))
        second = first.copy()
        second[plan.test] = 1_000_000.0
        transform_a = fit_fold_preprocessors(source, plan, first)[2]
        transform_b = fit_fold_preprocessors(source, plan, second)[2]
        for key in transform_a.state_dict():
            np.testing.assert_allclose(
                transform_a.state_dict()[key],
                transform_b.state_dict()[key],
                rtol=0,
                atol=0,
            )

    def test_unseen_validation_or_test_arm_fails_closed(self) -> None:
        source, n_primary = records()
        source[0] = replace(source[0], arm="test-only-arm")
        plan = build_fold_training_plan(source, n_primary, manifest(), 0, output_root="root")
        with self.assertRaisesRegex(ValueError, "未见 treatment arm"):
            fit_fold_preprocessors(source, plan, raw_response(len(source)))

    def test_wrong_response_shape_is_rejected(self) -> None:
        source, n_primary = records()
        plan = build_fold_training_plan(source, n_primary, manifest(), 0, output_root="root")
        with self.assertRaisesRegex(ValueError, r"\[N,3,18\]"):
            fit_fold_preprocessors(source, plan, np.zeros((len(source), 4, 18), dtype=np.float32))


class LegacyXCacheDatasetTest(unittest.TestCase):
    def test_loads_x_and_recomputes_clean_geometry(self) -> None:
        source = [replace(record("P0"), pcr=None)]
        encoder = ConditionEncoder(source)
        image = np.zeros((4, 8, 4, 6, 8), dtype=np.float32)
        image[:, 7, 1:3, 2:5, 3:7] = 1.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "P0_dce8_test.npz"
            np.savez_compressed(path, x=image)
            dataset = LegacyXCacheDataset(source, encoder, directory, expected_crop_size=(4, 6, 8))
            item = dataset[0]
        self.assertEqual(item["patient_id"], "P0")
        torch.testing.assert_close(item["image"], torch.from_numpy(image))
        expected = np.stack([mask_geometry(image[visit, 7]) for visit in range(4)])
        torch.testing.assert_close(item["geometry"], torch.from_numpy(expected))
        self.assertNotIn("pcr", item)

    def test_missing_patient_and_duplicate_aggregate_fail_closed(self) -> None:
        source = [replace(record("P0"), pcr=None)]
        encoder = ConditionEncoder(source)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "缺少"):
                LegacyXCacheDataset(source, encoder, directory)
            image = np.zeros((4, 8, 2, 2, 2), dtype=np.float32)
            np.savez_compressed(Path(directory) / "P0_dce8_a.npz", x=image)
            np.savez_compressed(Path(directory) / "P0_dce8_b.npz", x=image)
            with self.assertRaisesRegex(ValueError, "恰好一个"):
                LegacyXCacheDataset(source, encoder, directory)


class PrepareAndProvenanceTest(unittest.TestCase):
    def test_prepare_injects_outcome_blind_records_and_fold_config(self) -> None:
        source, n_primary = records()
        captured: list[PatientRecord] = []

        def factory(
            factory_records: list[PatientRecord],
            _encoder: ConditionEncoder,
            _config: ExperimentConfig,
        ) -> Dataset:
            captured.extend(factory_records)
            return EmptyBase(factory_records)

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            clean_runner,
            "load_experiment_records",
            return_value=(source, n_primary),
        ):
            prepared = prepare_fold_training(
                ExperimentConfig(),
                2,
                manifest=manifest(),
                expected_manifest_sha256=None,
                output_root=directory,
                raw_response=raw_response(len(source)),
                base_dataset_factory=factory,
            )
        self.assertTrue(captured)
        self.assertTrue(all(item.pcr is None for item in captured))
        self.assertEqual(prepared.plan.output_dir, Path(directory) / "fold_02")
        self.assertEqual(prepared.config.train.output_dir, str(Path(directory) / "fold_02"))
        self.assertEqual(prepared.base_dataset_source["mode"], "injected_base_dataset_factory")
        self.assertEqual(prepared.response_target_source["mode"], "in_memory_raw_response")
        self.assertEqual(len(prepared.response_target_source["sha256"]), 64)

    def test_checkpoint_keeps_clean_schema_and_audit_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prepared = prepared_fixture(Path(directory))
            payload = checkpoint_payload_with_provenance(
                nn.Linear(2, 2),
                prepared,
                epoch=3,
                validation={"prediction": 0.25, "visit_state_std": 0.8},
            )
            summary = validate_checkpoint_payload(
                payload,
                expected_primary_ids=[f"P{index}" for index in range(5)],
                expected_fold_ids=prepared.plan.split_patient_ids(),
            )
        self.assertEqual(summary["epoch"], 3)
        provenance = payload["audit_provenance"]
        self.assertEqual(provenance["fold"], 0)
        self.assertEqual(
            provenance["fit_indices"]["condition_encoder"],
            list(prepared.plan.pretrain_train),
        )
        self.assertIn("pCR redacted", provenance["fit_scopes"]["condition_encoder"])
        self.assertEqual(payload["splits"], prepared.plan.checkpoint_splits())
        self.assertEqual(provenance["response_target_source"]["mode"], "unit_test")
        self.assertEqual(len(provenance["implementation"]["audit_wrapper"]["sha256"]), 64)


class TrainingSafetyAndSmokeTest(unittest.TestCase):
    def test_dead_man_switch_precedes_preparation(self) -> None:
        with mock.patch(
            "shortcut_audit.auditlib.training.prepare_fold_training"
        ) as prepare:
            with self.assertRaisesRegex(RuntimeError, "allow_training=True"):
                train_explicit_fold(ExperimentConfig(), 0)
            prepare.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "allow_training=True"):
            train_fivefold(ExperimentConfig())

    def test_mocked_one_epoch_writes_provenance_without_real_training(self) -> None:
        class TinyModel(nn.Module):
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.ones(()))

        class TinyObjective(nn.Module):
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                super().__init__()

        stats = {"loss": 1.0, "prediction": 0.5, "visit_state_std": 0.8}
        with tempfile.TemporaryDirectory() as directory:
            prepared = prepared_fixture(Path(directory))
            prepared.config.train.epochs = 1
            with (
                mock.patch(
                    "shortcut_audit.auditlib.training.prepare_fold_training",
                    return_value=prepared,
                ),
                mock.patch("shortcut_audit.auditlib.training.CoReJEPA", TinyModel),
                mock.patch("shortcut_audit.auditlib.training.PretrainingObjective", TinyObjective),
                mock.patch.object(clean_runner, "make_loader", return_value=object()),
                mock.patch.object(clean_runner, "select_device", return_value=(torch.device("cpu"), [])),
                mock.patch.object(clean_runner, "run_epoch", return_value=stats),
                mock.patch.object(clean_runner, "export_frozen_states") as export,
            ):
                best = train_explicit_fold(ExperimentConfig(), 0, allow_training=True)
            payload = torch.load(best, map_location="cpu", weights_only=False)
            self.assertEqual(payload["audit_provenance"]["fold"], 0)
            self.assertTrue((prepared.plan.output_dir / "last_corejepa.pt").exists())
            self.assertTrue((prepared.plan.output_dir / "history.csv").exists())
            export.assert_called_once()
            with mock.patch(
                "shortcut_audit.auditlib.training.prepare_fold_training",
                return_value=prepared,
            ):
                with self.assertRaisesRegex(FileExistsError, "拒绝覆盖"):
                    train_explicit_fold(ExperimentConfig(), 0, allow_training=True)

    def test_smoke_config_is_small_valid_and_does_not_mutate_input(self) -> None:
        source = ExperimentConfig()
        smoke = make_smoke_config(source, "smoke-output")
        self.assertEqual(source.model.latent_dim, 192)
        self.assertEqual(source.train.epochs, 12)
        self.assertEqual(smoke.model.latent_dim, 16)
        self.assertEqual(smoke.train.epochs, 1)
        self.assertEqual(smoke.train.gpus, ())
        model = CoReJEPA(smoke.model, condition_dim=12)
        self.assertEqual(model.config.latent_dim, 16)

    def test_shared_seed2026_manifest_plan_when_available(self) -> None:
        if not DEFAULT_SEED2026_MANIFEST.exists():
            self.skipTest("共享 seed2026 manifest 在当前机器不存在")
        config = load_config("ispy_jepa_tmi_clean/configs/paper_v1.yaml")
        source, n_primary = clean_runner.load_experiment_records(config)
        plan = build_fold_training_plan(
            source,
            n_primary,
            DEFAULT_SEED2026_MANIFEST,
            0,
            output_root="audit-runs",
            expected_manifest_sha256=SEED2026_MANIFEST_SHA256,
        )
        self.assertEqual(len(plan.primary_train), 525)
        self.assertEqual(len(plan.validation), 121)
        self.assertEqual(len(plan.test), 162)
        self.assertEqual(len(plan.extra_pretraining_indices), 156)
        dummy_response = raw_response(len(source))
        _, encoder, _, _, _ = fit_fold_preprocessors(source, plan, dummy_response)
        self.assertEqual(encoder.spec.dim, 25)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import numpy as np

from ispy_jepa_tmi_clean.corejepa.config import ExperimentConfig, ModelConfig
from ispy_jepa_tmi_clean.corejepa.data.condition import ConditionEncoder
from ispy_jepa_tmi_clean.corejepa.data.records import PatientRecord
from ispy_jepa_tmi_clean.corejepa.models import CoReJEPA
from shortcut_audit.auditlib.runtime import (
    StoredConditionEncoder,
    experiment_config_from_payload,
    records_in_checkpoint_order,
)


def record(patient_id: str, arm: str = "A") -> PatientRecord:
    return PatientRecord(
        patient_id=patient_id,
        cohort="ispy2",
        arm=arm,
        hr=1,
        her2=0,
        mp=1,
        age=60.0,
        manifest_path=None,  # type: ignore[arg-type]
        pcr=0,
    )


class RuntimeTest(unittest.TestCase):
    def test_stored_condition_matches_clean_encoder(self) -> None:
        records = [record("P0", "A"), record("P1", "B")]
        native = ConditionEncoder(records)
        stored = StoredConditionEncoder(
            {
                "feature_names": native.spec.feature_names,
                "arm_vocab": native.spec.arm_vocab,
                "age_mean": native.spec.age_mean,
                "age_std": native.spec.age_std,
            }
        )
        np.testing.assert_array_equal(stored.encode(records[0]), native.encode(records[0]))

    def test_patient_order_rejects_extra_records(self) -> None:
        with self.assertRaisesRegex(ValueError, "extra=1"):
            records_in_checkpoint_order([record("P0"), record("P1")], ["P1"])

    def test_config_roundtrip_preserves_tuple_fields(self) -> None:
        config = ExperimentConfig(model=ModelConfig(base_channels=4, latent_dim=32))
        restored = experiment_config_from_payload({"config": config.to_dict()})
        self.assertEqual(restored.model.latent_dim, 32)
        self.assertIsInstance(restored.train.gpus, tuple)

    def test_model_contract_import_is_live(self) -> None:
        model = CoReJEPA(ModelConfig(base_channels=4, latent_dim=32), condition_dim=25)
        self.assertEqual(model.config.latent_dim, 32)


if __name__ == "__main__":
    unittest.main()

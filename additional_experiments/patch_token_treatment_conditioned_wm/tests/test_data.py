from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
import math
from pathlib import Path
import pickle
import stat
import sys
import types
from typing import Any

import numpy as np
import pandas as pd
import pytest


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from patch_token_wm import data  # noqa: E402


@pytest.fixture()
def clinical_assets(tmp_path: Path) -> dict[str, Any]:
    primary = pd.DataFrame(
        {
            "patient_id": ["P0", "P1", "P2", "P3", "P4"],
            "arm": [
                "T-DM1 + Pertuzumab",  # absent from age-fit rows on purpose
                "Paclitaxel + Pembrolizumab",
                "Paclitaxel",
                "Paclitaxel + AMG 386",
                "Paclitaxel + Neratinib",
            ],
            "label_hr": [0, 1, 0, 1, 0],
            "label_her2": [1, 0, 0, 1, 0],
            "label_mp": [1, 0, 1, 0, 1],
            "age_at_screening": [1000.0, 900.0, 40.0, 60.0, np.nan],
            # Forbidden fields exist in the real source and must not be loaded.
            "label_pcr": ["DO_NOT_PARSE"] * 5,
            "raw_pCR": [object().__repr__()] * 5,
            "unrelated_column": [1, 2, 3, 4, 5],
        }
    )
    external = pd.DataFrame(
        {
            "patient_id": ["E0", "E1"],
            "arm": ["ISPY1_NACT", "ISPY1_NACT"],
            "label_hr": [1, 0],
            "label_her2": [0, 1],
            "label_mp": [0, 0],
            "age_at_screening": [50.0, 70.0],
            "label_pcr": ["DO_NOT_PARSE", "DO_NOT_PARSE"],
            "rcb_class": [3, 1],
        }
    )
    ispy2_path = tmp_path / "ispy2_clinical.csv"
    ispy1_path = tmp_path / "ispy1_clinical.csv"
    primary.to_csv(ispy2_path, index=False)
    external.to_csv(ispy1_path, index=False)
    return {
        "ispy2_path": ispy2_path,
        "ispy1_path": ispy1_path,
        "ispy2_sha256": data.file_sha256(ispy2_path),
        "ispy1_sha256": data.file_sha256(ispy1_path),
        "primary_ids": tuple(primary["patient_id"]),
        "external_ids": ("E0",),
        "primary_frame": primary,
    }


def load_table(assets: dict[str, Any]) -> data.AuthorizedConditionTable:
    return data.load_authorized_condition_table(
        primary_patient_ids=assets["primary_ids"],
        authorized_external_train_only_patient_ids=assets["external_ids"],
        ispy2_path=assets["ispy2_path"],
        ispy2_sha256=assets["ispy2_sha256"],
        ispy1_path=assets["ispy1_path"],
        ispy1_sha256=assets["ispy1_sha256"],
        expected_primary_patient_sha256=data.patient_set_sha256(assets["primary_ids"]),
        expected_external_patient_sha256=data.patient_set_sha256(
            assets["external_ids"]
        ),
    )


def fit_encoder(table: data.AuthorizedConditionTable) -> data.ConditionEncoder:
    return data.ConditionEncoder.fit(
        table,
        outer_train_patient_ids=("P2", "P3", "P4"),
        authorized_external_train_only_patient_ids=("E0",),
    )


def test_authorized_table_and_encoder_are_spawn_picklable(
    clinical_assets: dict[str, Any],
) -> None:
    table = load_table(clinical_assets)
    encoder = fit_encoder(table)
    restored = pickle.loads(pickle.dumps(encoder, protocol=5))
    assert isinstance(restored.table.records, types.MappingProxyType)
    assert (
        restored.encode_numpy("P2")["arm_index"].item()
        == encoder.encode_numpy("P2")["arm_index"].item()
    )


def test_clinical_reads_are_hash_pinned_and_pcr_free(
    clinical_assets: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    observed_usecols: list[tuple[str, ...]] = []
    original = data.pd.read_csv

    def guarded_read_csv(*args: Any, **kwargs: Any) -> pd.DataFrame:
        observed_usecols.append(tuple(kwargs["usecols"]))
        assert "label_pcr" not in kwargs["usecols"]
        assert "raw_pCR" not in kwargs["usecols"]
        return original(*args, **kwargs)

    monkeypatch.setattr(data.pd, "read_csv", guarded_read_csv)
    table = load_table(clinical_assets)
    assert observed_usecols == [data.CLINICAL_CSV_USECOLS, data.CLINICAL_CSV_USECOLS]
    assert set(table.records) == {"P0", "P1", "P2", "P3", "P4", "E0"}
    assert set(vars(table.records["P0"])) == {
        "patient_id",
        "cohort",
        "arm",
        "hr",
        "her2",
        "mp",
        "age",
    }
    assert not any("pcr" in name.casefold() for name in vars(table.records["P0"]))

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        data.read_clinical_condition_csv(
            clinical_assets["ispy2_path"], "0" * 64, cohort="ispy2"
        )


def test_coverage_and_patient_hashes_fail_closed(
    clinical_assets: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="misses 1 required"):
        data.load_authorized_condition_table(
            primary_patient_ids=clinical_assets["primary_ids"] + ("MISSING",),
            authorized_external_train_only_patient_ids=("E0",),
            ispy2_path=clinical_assets["ispy2_path"],
            ispy2_sha256=clinical_assets["ispy2_sha256"],
            ispy1_path=clinical_assets["ispy1_path"],
            ispy1_sha256=clinical_assets["ispy1_sha256"],
        )
    with pytest.raises(ValueError, match="patient-set SHA-256 mismatch"):
        data.load_authorized_condition_table(
            primary_patient_ids=clinical_assets["primary_ids"],
            authorized_external_train_only_patient_ids=("E0",),
            ispy2_path=clinical_assets["ispy2_path"],
            ispy2_sha256=clinical_assets["ispy2_sha256"],
            ispy1_path=clinical_assets["ispy1_path"],
            ispy1_sha256=clinical_assets["ispy1_sha256"],
            expected_primary_patient_sha256="0" * 64,
        )


def test_fixed_vocab_temporal_contract_age_missing_and_nominal_delta(
    clinical_assets: dict[str, Any],
) -> None:
    encoder = fit_encoder(load_table(clinical_assets))
    assert encoder.arm_vocab == data.FIXED_ARM_VOCAB
    assert len(encoder.arm_vocab) == 14
    assert encoder.arm_vocab[0] == "ISPY1_NACT"
    assert encoder.normalization.mean == pytest.approx(50.0)
    assert encoder.normalization.std == pytest.approx(math.sqrt(200.0 / 3.0))
    assert encoder.normalization.fit_patient_count == 4
    assert encoder.normalization.observed_count == 3
    assert encoder.normalization.missing_count == 1

    encoded = encoder.encode_numpy("P4")
    np.testing.assert_array_equal(
        encoded["temporal_bits"],
        np.asarray(
            [
                [1, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 1, 0],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(encoded["delta_t"], np.ones(3, dtype=np.float32))
    np.testing.assert_array_equal(encoded["clinical"][3:], [0.0, 1.0])
    assert encoded["clinical"].shape == (5,)
    matrix = encoder.encode_matrix("P0")
    assert matrix.shape == (3, 27)
    assert matrix.dtype == np.float32
    assert int(encoded["arm_index"]) == data.ARM_TO_INDEX["Paclitaxel + Neratinib"]
    # P0's arm was absent from fit rows, yet its preregistered ID remains fixed.
    assert int(encoder.encode_numpy("P0")["arm_index"]) == 13


def test_age_fit_is_independent_of_validation_and_test(
    clinical_assets: dict[str, Any]
) -> None:
    first = fit_encoder(load_table(clinical_assets))
    changed = clinical_assets["primary_frame"].copy()
    changed.loc[changed["patient_id"].eq("P0"), "age_at_screening"] = -999999.0
    changed.loc[changed["patient_id"].eq("P1"), "age_at_screening"] = 999999.0
    changed.to_csv(clinical_assets["ispy2_path"], index=False)
    clinical_assets["ispy2_sha256"] = data.file_sha256(clinical_assets["ispy2_path"])
    second = fit_encoder(load_table(clinical_assets))
    assert second.normalization == first.normalization
    np.testing.assert_array_equal(
        second.encode_numpy("P2")["clinical"], first.encode_numpy("P2")["clinical"]
    )


def test_age_fit_requires_every_authorized_external_record(
    clinical_assets: dict[str, Any]
) -> None:
    table = load_table(clinical_assets)
    with pytest.raises(ValueError, match="complete authorized external"):
        data.ConditionEncoder.fit(
            table,
            outer_train_patient_ids=("P2", "P3", "P4"),
            authorized_external_train_only_patient_ids=("E1",),
        )


def test_dataset_seals_base_and_default_collation_matches_model_api(
    clinical_assets: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    encoder = fit_encoder(load_table(clinical_assets))

    class FakeStageBDataset:
        def __init__(
            self, patient_ids: tuple[str, ...], add_label: bool = False
        ) -> None:
            self.patient_ids = patient_ids
            self.add_label = add_label

        def __len__(self) -> int:
            return len(self.patient_ids)

        def __getitem__(self, index: int) -> dict[str, Any]:
            item: dict[str, Any] = {
                "patient_id": self.patient_ids[index],
                "image": torch.zeros((4, 7, 2, 2, 2), dtype=torch.float32),
                "ftv_target": torch.zeros(4, dtype=torch.float32),
                "ftv_mask": torch.zeros(4, dtype=torch.bool),
            }
            if self.add_label:
                item["label_pcr"] = 1
            return item

    training_ids = ("P2", "P3", "P4", "E0")
    wrapped = data.ConditionedStageBDataset(
        FakeStageBDataset(training_ids), encoder, split="train"
    )
    item = wrapped[0]
    assert set(item) == {"patient_id", "image", "ftv_target", "ftv_mask", "condition"}
    assert set(item["condition"]) == {
        "arm_index",
        "clinical",
        "temporal_bits",
        "delta_t",
    }
    assert item["condition"]["arm_index"].shape == torch.Size([])
    assert item["condition"]["clinical"].shape == (5,)
    assert item["condition"]["temporal_bits"].shape == (3, 7)
    assert item["condition"]["delta_t"].shape == (3,)
    assert not any("pcr" in key.casefold() or "label" in key.casefold() for key in item)

    batch = next(
        iter(torch.utils.data.DataLoader(wrapped, batch_size=4, shuffle=False))
    )
    assert batch["condition"]["arm_index"].shape == (4,)
    assert batch["condition"]["clinical"].shape == (4, 5)
    assert batch["condition"]["temporal_bits"].shape == (4, 3, 7)
    assert batch["condition"]["delta_t"].shape == (4, 3)

    fake_model = types.ModuleType("patch_token_wm.model")

    @dataclass
    class TransitionCondition:
        arm_index: Any
        clinical: Any
        temporal_bits: Any
        delta_t: Any

    fake_model.TransitionCondition = TransitionCondition
    monkeypatch.setitem(sys.modules, "patch_token_wm.model", fake_model)
    condition = data.transition_condition_from_batch(batch)
    assert isinstance(condition, TransitionCondition)
    assert condition.temporal_bits.shape == (4, 3, 7)

    poisoned = data.ConditionedStageBDataset(
        FakeStageBDataset(training_ids, add_label=True), encoder, split="train"
    )
    with pytest.raises(ValueError, match="item keys drifted"):
        poisoned[0]
    with pytest.raises(ValueError, match="overlaps age-fit"):
        data.ConditionedStageBDataset(FakeStageBDataset(("P2",)), encoder, split="test")


@pytest.fixture()
def manifest_module() -> Any:
    script = EXPERIMENT_ROOT / "scripts" / "build_condition_manifest.py"
    spec = importlib.util.spec_from_file_location(
        "build_condition_manifest_test", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_fold_fixture(path: Path) -> None:
    rows: list[dict[str, Any]] = []
    patients = ["P0", "P1", "P2", "P3", "P4"]
    for fold in range(5):
        for index, patient_id in enumerate(patients):
            split = (
                "test"
                if index == fold
                else "val" if index == (fold + 1) % 5 else "train"
            )
            rows.append(
                {
                    "patient_id": patient_id,
                    "fold": fold,
                    "split": split,
                    "label_pcr": "FORBIDDEN_OUTCOME",
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_manifest_cli_writes_private_aggregate_only_metadata(
    tmp_path: Path, clinical_assets: dict[str, Any], manifest_module: Any
) -> None:
    fold_path = tmp_path / "folds.csv"
    _write_fold_fixture(fold_path)
    external_path = tmp_path / "external_authorization.private.csv"
    pd.DataFrame(
        {
            "patient_id": ["E0", "E1"],
            "eligible": [True, False],
            "label_pcr": ["FORBIDDEN", "FORBIDDEN"],
        }
    ).to_csv(external_path, index=False)
    output = tmp_path / "condition_fold_0.private.json"
    manifest_module.main(
        [
            "--fold",
            "0",
            "--fold-manifest",
            str(fold_path),
            "--fold-manifest-sha256",
            data.file_sha256(fold_path),
            "--external-authorization-manifest",
            str(external_path),
            "--external-authorization-manifest-sha256",
            data.file_sha256(external_path),
            "--external-manifest-has-eligible-column",
            "--ispy2-clinical",
            str(clinical_assets["ispy2_path"]),
            "--ispy2-clinical-sha256",
            clinical_assets["ispy2_sha256"],
            "--ispy1-clinical",
            str(clinical_assets["ispy1_path"]),
            "--ispy1-clinical-sha256",
            clinical_assets["ispy1_sha256"],
            "--expected-primary-count",
            "5",
            "--expected-external-count",
            "1",
            "--output",
            str(output),
        ]
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    encoded = json.dumps(payload, sort_keys=True)
    assert payload["privacy"]["contains_patient_identifiers"] is False
    assert payload["population"]["primary_count"] == 5
    assert payload["population"]["authorized_external_train_only_count"] == 1
    assert payload["condition"]["age_normalization"]["fit_scope"].endswith("_only")
    assert payload["assertions"]["age_fit_uses_test"] is False
    assert payload["condition"]["pcr_column_loaded"] is False
    assert "FORBIDDEN_OUTCOME" not in encoded
    assert not manifest_module._PATIENT_VALUE_RE.search(encoded)
    assert stat.S_IMODE(output.stat().st_mode) == 0o600

    with pytest.raises(ValueError, match="must end in .private.json"):
        manifest_module.write_private_manifest(tmp_path / "tracked.json", payload)

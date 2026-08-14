from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from crps import evaluation_contracts as contracts  # noqa: E402
from crps import evaluation_lock  # noqa: E402


def _config(tmp_path: Path, fold_path: Path) -> dict:
    return {
        "frozen_inputs": {
            "fold_manifest_path": str(fold_path),
            "fold_manifest_sha256": contracts.file_sha256(fold_path),
            "expected_primary_patient_count": 2,
        }
    }


def test_label_free_fold_loader_passes_exact_usecols_and_never_materializes_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "folds.csv"
    rows = []
    for fold in range(5):
        rows.extend(
            [
                {"patient_id": "P0", "fold": fold, "split": "test" if fold == 0 else "val", "label_pcr": "SECRET0"},
                {"patient_id": "P1", "fold": fold, "split": "test" if fold == 1 else "train", "label_pcr": "SECRET1"},
            ]
        )
    pd.DataFrame(rows).to_csv(source, index=False)
    real_read_csv = contracts.pd.read_csv
    calls: list[object] = []

    def guarded_read_csv(path: object, *args: object, **kwargs: object) -> pd.DataFrame:
        calls.append(kwargs.get("usecols"))
        assert kwargs.get("usecols") == ["patient_id", "fold", "split"]
        frame = real_read_csv(path, *args, **kwargs)
        assert "label_pcr" not in frame.columns
        return frame

    monkeypatch.setattr(contracts.pd, "read_csv", guarded_read_csv)
    assignments = contracts.load_fold_assignments(_config(tmp_path, source))
    assert assignments.shape == (10, 3)
    assert list(assignments.columns) == ["patient_id", "fold", "split"]
    assert calls == [["patient_id", "fold", "split"]]


def test_evaluation_lock_build_orders_label_free_boundary_before_any_outcome_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    fake_config = {
        "frozen_inputs": {
            "representation_preregistration_lock": "PREREGISTRATION_LOCK.json",
            "fold_manifest_path": "fold.csv",
            "stage_a_sentinel_path": "stage_a.json",
            "stage_b_data_contract_path": "stage_b.json",
        }
    }
    monkeypatch.setattr(evaluation_lock, "load_evaluation_config", lambda _: fake_config)

    def assignments(_: object) -> pd.DataFrame:
        calls.append("label_free_assignments")
        return pd.DataFrame({"patient_id": [], "fold": [], "split": []})

    monkeypatch.setattr(evaluation_lock, "load_fold_assignments", assignments)
    monkeypatch.setattr(
        evaluation_lock,
        "load_representation_lock",
        lambda _: calls.append("representation") or {"lock_sha256": "a" * 64},
    )
    monkeypatch.setattr(
        evaluation_lock,
        "load_factorized_export_status",
        lambda _: (_ for _ in ()).throw(RuntimeError("stop after first boundary calls")),
    )
    with pytest.raises(RuntimeError, match="stop"):
        evaluation_lock.build_payload()
    assert calls == ["label_free_assignments", "representation"]


def test_verify_refuses_tampered_lock_without_rewriting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = tmp_path / "EVALUATION_LOCK.json"
    payload = {"status": "PASS", "lock_sha256": "a" * 64}
    lock.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(evaluation_lock, "LOCK_PATH", lock)
    monkeypatch.setattr(
        evaluation_lock,
        "build_payload",
        lambda _=None: {"status": "PASS", "lock_sha256": "b" * 64},
    )
    before = lock.read_bytes()
    with pytest.raises(PermissionError, match="drifted"):
        evaluation_lock.verify(lock)
    assert lock.read_bytes() == before


def test_evaluation_code_inventory_includes_diagnostics_and_matched_response_probe() -> None:
    required = {
        "src/crps/diagnostics.py",
        "src/crps/response_probes.py",
        "src/crps/evaluation_contracts.py",
        "src/crps/evaluation_modeling.py",
        "src/crps/evaluation.py",
        "src/crps/reporting.py",
    }
    assert required.issubset(set(evaluation_lock.CODE_PATHS))


def test_load_clinical_verifies_boundary_before_opening_label_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crps import evaluation_lock as lock_module

    calls: list[str] = []

    def fail_lock() -> object:
        calls.append("verify")
        raise PermissionError("locked first")

    monkeypatch.setattr(lock_module, "verify", fail_lock)
    monkeypatch.setattr(
        contracts,
        "_locked_file",
        lambda *_: (_ for _ in ()).throw(AssertionError("label file was opened")),
    )
    with pytest.raises(PermissionError, match="locked first"):
        contracts.load_clinical({"labels": {}}, pd.DataFrame())
    assert calls == ["verify"]


def test_freeze_refuses_overwrite_before_payload_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = tmp_path / "EVALUATION_LOCK.json"
    lock.write_text("sentinel", encoding="utf-8")
    monkeypatch.setattr(evaluation_lock, "LOCK_PATH", lock)
    monkeypatch.setattr(
        evaluation_lock,
        "build_payload",
        lambda _=None: (_ for _ in ()).throw(AssertionError("must not build")),
    )
    with pytest.raises(FileExistsError, match="overwrite"):
        evaluation_lock.freeze(lock)
    assert lock.read_text(encoding="utf-8") == "sentinel"

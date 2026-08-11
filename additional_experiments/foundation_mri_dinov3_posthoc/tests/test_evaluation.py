from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

from foundation_mri.data import (
    ClinicalTable,
    FoundationFeatureAsset,
    HR_HER2_SUBTYPES,
    RadiomicsTable,
)
from foundation_mri.evaluation import EvaluationResult
from foundation_mri_dinov3 import evaluation as subject


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_evaluation.py"


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _lock_receipt(consumer: str) -> dict[str, Any]:
    return {
        "consumer": consumer,
        "lock_sha256": ("a" if consumer == "baseline" else "b") * 64,
        "expected_counts": {**subject._BASELINE_COUNTS, **subject._PROBE_COUNTS},
    }


def _synthetic_loaded() -> subject._LoadedInputs:
    patient_ids = np.asarray([f"P{index:04d}" for index in range(808)], dtype=str)
    hr = np.arange(808, dtype=np.int64) % 2
    her2 = (np.arange(808, dtype=np.int64) // 2) % 2
    subtype = np.asarray(
        [
            f"HR{'+' if hr_value else '-'}/HER2{'+' if her2_value else '-'}"
            for hr_value, her2_value in zip(hr, her2, strict=True)
        ],
        dtype=str,
    )
    assert set(subtype) == set(HR_HER2_SUBTYPES)
    clinical = ClinicalTable(
        patient_ids=patient_ids,
        pcr=np.arange(808, dtype=np.int64) % 2,
        hr=hr,
        her2=her2,
        mp=(np.arange(808, dtype=np.int64) // 3) % 2,
        age=np.linspace(30.0, 79.0, 808),
        arm=np.asarray([f"arm_{index % 3}" for index in range(808)], dtype=str),
        subtype=subtype,
        sha256="1" * 64,
    )
    radiomics_ids = patient_ids[:375]
    ftv = np.arange(375 * 4, dtype=np.float64).reshape(375, 4, 1) / 10.0
    radiomics = RadiomicsTable(
        patient_ids=radiomics_ids,
        values=ftv,
        feature_names=("ftv",),
        sha256="2" * 64,
    )
    values = np.arange(808 * 4, dtype=np.float32).reshape(808, 4, 1)
    sources = tuple(
        subject.ImagingSource(
            subject.MODEL_NAME,
            spatial,
            subject._constant_folds(values + spatial_index),
        )
        for spatial_index, spatial in enumerate(("GLOBAL", "LOCAL"))
    )
    return subject._LoadedInputs(
        folds=object(),  # all estimator entry points are synthetic stubs below
        clinical=clinical,
        radiomics=radiomics,
        sources=sources,
        ftv=ftv[:, :, 0],
    )


def _binary_result(
    *,
    patient_ids: np.ndarray,
    target: str,
    model: str,
    spatial: str,
    timing: str,
    population: str,
) -> EvaluationResult:
    n = len(patient_ids)
    predictions = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "target": target,
            "model": model,
            "spatial": spatial,
            "timing": timing,
            "analysis_population": population,
            "fold": np.arange(n, dtype=np.int64) % 5,
            "y_true": np.arange(n, dtype=np.int64) % 2,
            "y_probability": np.linspace(0.1, 0.9, n),
        }
    )
    selections = pd.DataFrame(
        {
            "target": [target] * 5,
            "model": [model] * 5,
            "spatial": [spatial] * 5,
            "timing": [timing] * 5,
            "analysis_population": [population] * 5,
            "fold": list(range(5)),
            "selected": ["synthetic"] * 5,
        }
    )
    return EvaluationResult(predictions, selections)


def _ridge_result(
    *,
    patient_ids: np.ndarray,
    model: str,
    spatial: str,
    task: str,
    endpoint: str,
    population: str,
) -> EvaluationResult:
    n = len(patient_ids)
    predictions = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "target": "FTV",
            "model": model,
            "spatial": spatial,
            "task": task,
            "endpoint": endpoint,
            "analysis_population": population,
            "fold": np.arange(n, dtype=np.int64) % 5,
            "y_true": np.arange(n, dtype=np.float64),
            "y_pred": np.arange(n, dtype=np.float64),
        }
    )
    selections = pd.DataFrame(
        {
            "target": ["FTV"] * 5,
            "model": [model] * 5,
            "spatial": [spatial] * 5,
            "task": [task] * 5,
            "endpoint": [endpoint] * 5,
            "analysis_population": [population] * 5,
            "fold": list(range(5)),
            "selected": ["synthetic"] * 5,
        }
    )
    return EvaluationResult(predictions, selections)


def _two_public_rows(frame: pd.DataFrame, identity_columns: list[str]) -> pd.DataFrame:
    identities = frame.loc[:, identity_columns].drop_duplicates().reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for identity in identities.to_dict(orient="records"):
        for aggregation in ("pooled_oof", "outer_fold_macro"):
            rows.append({**identity, "aggregation": aggregation, "synthetic_value": 0.0})
    return pd.DataFrame(rows)


def _fake_receipt_writer(calls: list[dict[str, Any]]):
    def write(**kwargs: Any) -> dict[str, Any]:
        destination = Path(kwargs["receipt_path"])
        assert not os.path.lexists(destination)
        assert tuple(kwargs["command_argv"]) == ()
        assert all(Path(path).is_file() for path in kwargs["artifacts"].values())
        payload = {
            "schema_version": "synthetic_producer_receipt",
            "consumer": kwargs["consumer"],
            "counts": kwargs["counts"],
            "artifact_roles": sorted(kwargs["artifacts"]),
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        destination.chmod(0o600)
        calls.append(dict(kwargs))
        return payload

    return write


def test_identity_contract_is_exact_and_dinov3_only() -> None:
    baseline = subject.baseline_identities()
    assert len(baseline) == 36 == len(set(baseline))
    assert {cell.spatial for cell in baseline} == {"GLOBAL", "LOCAL"}
    assert {cell.timing for cell in baseline} == {"T0", "T0-T1", "T0-T2"}
    assert sum(cell.population_kind == "full_808" for cell in baseline) == 12
    assert sum(cell.population_kind == "radiomics_complete_case_375" for cell in baseline) == 24
    assert {
        cell.model_name.removeprefix(subject.MODEL_NAME + "_") for cell in baseline
    } == {
        "mri_only",
        "mri_clinical",
        "mri_only_paired",
        "mri_clinical_paired",
        "mri_ftv",
        "mri_clinical_ftv",
    }
    assert all(cell.model_name.startswith(subject.MODEL_NAME + "_") for cell in baseline)

    probes = subject.probe_identities()
    assert len(probes) == 20 == len(set(probes))
    assert {family: sum(cell.family == family for cell in probes) for family in {
        "phenotype", "subtype", "ftv_static", "ftv_delta"
    }} == {"phenotype": 4, "subtype": 2, "ftv_static": 8, "ftv_delta": 6}


def test_dinov3_feature_gate_and_adapter_are_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    representation = np.zeros((3, 4, 2, subject.FEATURE_DIM), dtype=np.float32)
    asset = FoundationFeatureAsset(
        patient_ids=np.asarray(["P0", "P1", "P2"], dtype=str),
        representation=representation,
        spatial_axis=("GLOBAL", "LOCAL"),
        visits=("T0", "T1", "T2", "T3"),
        model_name=subject.MODEL_NAME,
        checkpoint_sha256="a" * 64,
        config_sha256=None,
        extraction_signature_sha256="b" * 64,
        canonical_patient_order_sha256="c" * 64,
        source_sha256="d" * 64,
    )
    monkeypatch.setattr(subject, "load_foundation_features", lambda *args, **kwargs: asset)
    assert subject._load_dinov3_asset(
        "synthetic.npz", expected_patient_ids=asset.patient_ids, expected_n=3
    ) is asset

    monkeypatch.setattr(
        subject,
        "load_foundation_features",
        lambda *args, **kwargs: replace(asset, model_name="dinov1"),
    )
    with pytest.raises(ValueError, match="model must be exactly"):
        subject._load_dinov3_asset(
            "synthetic.npz", expected_patient_ids=asset.patient_ids, expected_n=3
        )
    monkeypatch.setattr(
        subject,
        "load_foundation_features",
        lambda *args, **kwargs: replace(asset, checkpoint_sha256=None),
    )
    with pytest.raises(ValueError, match="embed checkpoint/extraction/order"):
        subject._load_dinov3_asset(
            "synthetic.npz", expected_patient_ids=asset.patient_ids, expected_n=3
        )

    visits = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
    source = subject.ImagingSource("synthetic", "GLOBAL", subject._constant_folds(visits))
    np.testing.assert_array_equal(subject._t0_matrix(source), visits[:, 0])
    np.testing.assert_array_equal(subject._visit_matrix(source, 2), visits[:, 2])
    np.testing.assert_array_equal(subject._delta_matrix(source, 1), visits[:, 2] - visits[:, 1])


def test_formal_lock_receipt_is_gated_before_data_load() -> None:
    receipt = _lock_receipt("baseline")
    assert subject._validate_formal_lock_receipt(
        receipt, consumer="baseline", command_argv=()
    ) == "a" * 64
    with pytest.raises(ValueError, match="empty argv"):
        subject._validate_formal_lock_receipt(
            receipt, consumer="baseline", command_argv=("--override",)
        )
    with pytest.raises(ValueError, match="consumer identity"):
        subject._validate_formal_lock_receipt(
            receipt, consumer="probe", command_argv=()
        )
    drifted = {**receipt, "expected_counts": {**receipt["expected_counts"]}}
    drifted["expected_counts"]["pcr_prediction_rows"] -= 1
    with pytest.raises(ValueError, match="count contract"):
        subject._validate_formal_lock_receipt(
            drifted, consumer="baseline", command_argv=()
        )


def test_synthetic_baseline_exact_rows_modes_receipt_and_no_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded = _synthetic_loaded()
    monkeypatch.setattr(subject, "_load_inputs", lambda **kwargs: loaded)
    monkeypatch.setattr(
        subject,
        "_clinical_matrices",
        lambda clinical, folds: {
            fold: np.zeros((len(clinical.patient_ids), 1), dtype=np.float64)
            for fold in range(5)
        },
    )

    def evaluate(**kwargs: Any) -> EvaluationResult:
        identity = kwargs["identity"]
        clinical = kwargs["clinical"]
        return _binary_result(
            patient_ids=clinical.patient_ids,
            target="pCR",
            model=identity.model_name,
            spatial=identity.spatial,
            timing=identity.timing,
            population=identity.population_kind,
        )

    monkeypatch.setattr(subject, "_evaluate_binary_identity", evaluate)
    monkeypatch.setattr(
        subject,
        "aggregate_binary_predictions",
        lambda frame: _two_public_rows(
            frame, ["target", "model", "spatial", "timing", "analysis_population"]
        ),
    )
    receipt_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        subject, "write_producer_receipt", _fake_receipt_writer(receipt_calls)
    )

    counts = subject.run_baseline_evaluation(
        feature_path="feature",
        fold_manifest_path="folds",
        clinical_path="clinical",
        radiomics_path="radiomics",
        output_root=tmp_path,
        lock_receipt=_lock_receipt("baseline"),
        command_argv=(),
    )
    assert counts == {
        "pcr_identities": 36,
        "pcr_prediction_rows": 18_696,
        "pcr_selection_rows": 180,
        "pcr_public_rows": 72,
    }
    paths = subject._output_paths(tmp_path, "baseline")
    assert {_mode(path) for key, path in paths.items() if "metrics_public" not in key} == {0o600}
    assert _mode(paths["baseline_metrics_public"]) == 0o644
    assert len(receipt_calls) == 1
    assert receipt_calls[0]["consumer"] == "baseline"
    assert sorted(receipt_calls[0]["artifacts"]) == [
        "metrics",
        "predictions",
        "progress",
        "selection",
    ]
    with pytest.raises(FileExistsError, match="already exist"):
        subject.run_baseline_evaluation(
            feature_path="feature",
            fold_manifest_path="folds",
            clinical_path="clinical",
            radiomics_path="radiomics",
            output_root=tmp_path,
            lock_receipt=_lock_receipt("baseline"),
            command_argv=(),
        )


def test_synthetic_probe_exact_rows_modes_receipt_and_no_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded = _synthetic_loaded()
    monkeypatch.setattr(subject, "_load_inputs", lambda **kwargs: loaded)

    def binary(**kwargs: Any) -> EvaluationResult:
        return _binary_result(
            patient_ids=np.asarray(kwargs["patient_ids"], dtype=str),
            target=kwargs["target_name"],
            model=kwargs["model_name"],
            spatial=kwargs["spatial"],
            timing=kwargs["timing"],
            population=kwargs["analysis_population"],
        )

    monkeypatch.setattr(subject, "evaluate_binary_cv", binary)
    monkeypatch.setattr(subject, "evaluate_multiclass_cv", binary)
    monkeypatch.setattr(
        subject,
        "evaluate_ridge_cv",
        lambda **kwargs: _ridge_result(
            patient_ids=np.asarray(kwargs["patient_ids"], dtype=str),
            model=kwargs["model_name"],
            spatial=kwargs["spatial"],
            task=kwargs["task"],
            endpoint=kwargs["endpoint"],
            population=kwargs["analysis_population"],
        ),
    )
    classification_public = lambda frame: _two_public_rows(
        frame, ["target", "model", "spatial", "timing", "analysis_population"]
    )
    monkeypatch.setattr(subject, "aggregate_binary_predictions", classification_public)
    monkeypatch.setattr(subject, "aggregate_multiclass_predictions", classification_public)
    monkeypatch.setattr(
        subject,
        "aggregate_continuous_predictions",
        lambda frame: _two_public_rows(
            frame,
            ["target", "model", "spatial", "task", "endpoint", "analysis_population"],
        ),
    )
    receipt_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        subject, "write_producer_receipt", _fake_receipt_writer(receipt_calls)
    )

    counts = subject.run_probe_evaluation(
        feature_path="feature",
        fold_manifest_path="folds",
        clinical_path="clinical",
        radiomics_path="radiomics",
        output_root=tmp_path,
        lock_receipt=_lock_receipt("probe"),
        command_argv=(),
    )
    assert counts == {
        "phenotype_identities": 4,
        "phenotype_prediction_rows": 3_232,
        "phenotype_selection_rows": 20,
        "phenotype_public_rows": 8,
        "subtype_identities": 2,
        "subtype_prediction_rows": 1_616,
        "subtype_selection_rows": 10,
        "subtype_public_rows": 4,
        "ftv_identities": 14,
        "ftv_prediction_rows": 5_250,
        "ftv_selection_rows": 70,
        "ftv_public_rows": 28,
    }
    paths = subject._output_paths(tmp_path, "probe")
    for key, path in paths.items():
        assert _mode(path) == (0o644 if key.endswith("metrics_public") else 0o600)
    assert len(receipt_calls) == 1
    assert receipt_calls[0]["consumer"] == "probe"
    assert len(receipt_calls[0]["artifacts"]) == 10
    with pytest.raises(FileExistsError, match="already exist"):
        subject.run_probe_evaluation(
            feature_path="feature",
            fold_manifest_path="folds",
            clinical_path="clinical",
            radiomics_path="radiomics",
            output_root=tmp_path,
            lock_receipt=_lock_receipt("probe"),
            command_argv=(),
        )


def _load_cli_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("dinov3_run_evaluation_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_verifies_both_empty_commands_before_baseline_then_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _load_cli_module()
    monkeypatch.setattr(cli, "EXPERIMENT_ROOT", tmp_path)
    events: list[tuple[Any, ...]] = []
    locking = ModuleType("foundation_mri_dinov3.locking")

    def verify_lock(consumer: str, argv: Any) -> dict[str, Any]:
        events.append(("verify_lock", consumer, tuple(argv)))
        return {"lock_sha256": consumer[0] * 64, "consumer": consumer}

    def verify_receipt(consumer: str) -> dict[str, Any]:
        events.append(("verify_receipt", consumer))
        return {"consumer": consumer}

    locking.verify_evaluation_lock = verify_lock  # type: ignore[attr-defined]
    locking.verify_producer_receipt = verify_receipt  # type: ignore[attr-defined]
    evaluation = ModuleType("foundation_mri_dinov3.evaluation")

    def runner(consumer: str):
        def run(**kwargs: Any) -> dict[str, int]:
            events.append(("run", consumer, tuple(kwargs["command_argv"])))
            receipt = tmp_path / f"metrics/{consumer}_run.private.provenance.json"
            receipt.parent.mkdir(parents=True, exist_ok=True)
            receipt.write_text("{}\n", encoding="utf-8")
            receipt.chmod(0o600)
            return {"identity_count": 36 if consumer == "baseline" else 20}

        return run

    evaluation.run_baseline_evaluation = runner("baseline")  # type: ignore[attr-defined]
    evaluation.run_probe_evaluation = runner("probe")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "foundation_mri_dinov3.locking", locking)
    monkeypatch.setitem(sys.modules, "foundation_mri_dinov3.evaluation", evaluation)

    assert cli.main(()) == 0
    assert events[:3] == [
        ("verify_lock", "baseline", ()),
        ("verify_lock", "probe", ()),
        ("run", "baseline", ()),
    ]
    events.clear()
    assert cli.main(()) == 0
    assert events == [
        ("verify_lock", "baseline", ()),
        ("verify_lock", "probe", ()),
        ("verify_receipt", "baseline"),
        ("run", "probe", ()),
    ]
    with pytest.raises(FileExistsError, match="both locked"):
        cli.main(())
    with pytest.raises(ValueError, match="exact empty argv"):
        cli.main(("--consumer", "baseline"))


def test_cli_lock_failure_prevents_evaluator_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _load_cli_module()
    monkeypatch.setattr(cli, "EXPERIMENT_ROOT", tmp_path)
    locking = ModuleType("foundation_mri_dinov3.locking")

    def fail(consumer: str, argv: Any) -> None:
        raise RuntimeError("synthetic lock failure")

    locking.verify_evaluation_lock = fail  # type: ignore[attr-defined]
    locking.verify_producer_receipt = lambda consumer: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "foundation_mri_dinov3.locking", locking)
    monkeypatch.delitem(sys.modules, "foundation_mri_dinov3.evaluation", raising=False)
    with pytest.raises(RuntimeError, match="synthetic lock failure"):
        cli.main(())
    assert "foundation_mri_dinov3.evaluation" not in sys.modules

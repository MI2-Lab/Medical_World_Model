from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from residual_sph.contracts import file_sha256  # noqa: E402
import residual_sph.evaluation_lock as evaluation_lock  # noqa: E402
from residual_sph.evaluation_lock import (  # noqa: E402
    PROBE_SPECIFICATION_FILES,
    REPRESENTATION_AGGREGATES,
    build_pcr_firewall_audit,
    expected_representation_artifact_groups,
    verify_pcr_firewall_audit,
    verify_representation_freeze,
)
from residual_sph.pcr_evaluation import (  # noqa: E402
    MODEL_NAMES,
    TrainOnlyClinicalEncoder,
    feature_sets,
    fit_logistic,
    timing_prefix,
)


def _clinical() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "label_hr": [0, 1, 1, np.nan],
            "label_her2": [0, 0, 1, 1],
            "label_mp": [1, 0, 1, 0],
            "age_at_screening": [40.0, 50.0, np.nan, 60.0],
            "race_simple": ["A", "B", "C", None],
            "menopausal_status_simple": ["pre", "post", "pre", "post"],
            "ethnicity": ["x", "x", "y", None],
            "arm": ["a", "b", "unseen", "a"],
        }
    )


def test_clinical_encoder_fits_imputation_and_vocabulary_on_train_only() -> None:
    frame = _clinical()
    encoder = TrainOnlyClinicalEncoder().fit(frame.iloc[:2])
    before = dict(encoder.categories)
    matrix = encoder.transform(frame)
    assert np.isfinite(matrix).all()
    assert encoder.numeric_medians["age_at_screening"] == 45.0
    assert "C" not in encoder.categories["race_simple"]
    assert "unseen" not in encoder.categories["arm"]
    assert encoder.categories == before


def test_logistic_selects_from_frozen_grid_without_test_signature() -> None:
    x = np.arange(80, dtype=np.float64).reshape(40, 2)
    y = np.tile([0, 1], 20)
    fit = fit_logistic(x[:24], y[:24], x[24:32], y[24:32])
    probability = fit.predict_probability(x[32:])
    assert len(probability) == 8
    assert fit.selected_c in {1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0}


def test_timing_prefix_and_model_inventory() -> None:
    response = np.ones((3, 4, 192), dtype=np.float32)
    ftv = np.arange(12, dtype=np.float64).reshape(3, 4)
    mri, f = timing_prefix(response, ftv, "T0-T2")
    clinical = np.ones((3, 5))
    matrices = feature_sets(clinical, mri, f)
    assert mri.shape == (3, 576) and f.shape == (3, 3)
    assert tuple(matrices) == MODEL_NAMES


def test_pcr_boundary_fails_closed_without_freeze(tmp_path: Path) -> None:
    root = tmp_path / "experiment"
    root.mkdir()
    with pytest.raises(PermissionError, match="forbidden before representation freeze"):
        verify_representation_freeze(root, expected_preregistration_sha256="a" * 64)


def test_representation_freeze_inventory_shape_is_exact() -> None:
    groups = expected_representation_artifact_groups(ROOT)
    assert set(groups) == {
        "implementation_lock",
        "s0_provenance",
        "selection_records",
        "selected_checkpoints",
        "ftv_transforms",
        "feature_assets",
        "feature_metadata",
        "probe_outputs",
        "residualizer_transforms",
        "representation_aggregates",
        "probe_specification",
        "pcr_firewall_audit",
    }
    assert {key: len(value) for key, value in groups.items()} == {
        "implementation_lock": 1,
        "s0_provenance": 1,
        "selection_records": 40,
        "selected_checkpoints": 40,
        "ftv_transforms": 40,
        "feature_assets": 40,
        "feature_metadata": 40,
        "probe_outputs": 120,
        "residualizer_transforms": 5,
        "representation_aggregates": len(REPRESENTATION_AGGREGATES),
        "probe_specification": len(PROBE_SPECIFICATION_FILES),
        "pcr_firewall_audit": 1,
    }
    s0_ftv = [
        path
        for path in groups["ftv_transforms"]
        if "local_response_state_multiseed_confirmation" in path
    ]
    assert len(s0_ftv) == 10
    assert all(path.endswith("/ftv_transform.json") for path in s0_ftv)
    assert set(Path(path).name for path in groups["residualizer_transforms"]) == {
        f"fold_{fold}.json" for fold in range(5)
    }
    assert set(Path(path).name for path in groups["representation_aggregates"]) == {
        Path(path).name for path in REPRESENTATION_AGGREGATES
    }
    assert set(Path(path).name for path in groups["probe_specification"]) == {
        Path(path).name for path in PROBE_SPECIFICATION_FILES
    }
    flattened = [path for values in groups.values() for path in values]
    assert len(flattened) == len(set(flattened))


def _write_synthetic_freeze(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    repo = tmp_path / "repo"
    root = repo / "additional_experiments" / "residual_sph_grounding_pilot"
    groups = expected_representation_artifact_groups(root)
    for relative in (path for values in groups.values() for path in values):
        artifact = repo / relative
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(f"synthetic:{relative}\n".encode())
    implementation = root / "manifests" / "implementation_lock.json"
    payload: dict[str, object] = {
        "schema_version": 2,
        "experiment": "residual_sph_grounding_pilot",
        "status": "REPRESENTATION_FROZEN_PCR_EVALUATION_AUTHORIZED",
        "preregistration_lock_sha256": "a" * 64,
        "implementation_lock_sha256": file_sha256(implementation),
        "pcr_or_clinical_read_before_freeze": False,
        "selected_checkpoint_count": 40,
        "selection_record_count": 40,
        "ftv_transform_count": 40,
        "feature_asset_count": 40,
        "feature_metadata_count": 40,
        "probe_cell_count": 40,
        "probe_artifact_count": 120,
        "residualizer_fold_count": 5,
        "representation_aggregate_count": len(REPRESENTATION_AGGREGATES),
        "probe_specification_file_count": len(PROBE_SPECIFICATION_FILES),
        "pcr_firewall_audit_status": "PASS",
        "artifact_groups": {key: list(value) for key, value in groups.items()},
        "artifact_sha256": {
            relative: file_sha256(repo / relative)
            for values in groups.values()
            for relative in values
        },
    }
    freeze = root / "manifests" / "representation_freeze.json"
    freeze.write_text(json.dumps(payload), encoding="utf-8")
    return root, payload


def test_freeze_verifier_accepts_only_the_exact_inventory_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, payload = _write_synthetic_freeze(tmp_path)
    monkeypatch.setattr(
        evaluation_lock,
        "verify_pcr_firewall_audit",
        lambda *args, **kwargs: {"status": "PASS"},
    )
    observed = verify_representation_freeze(
        root, expected_preregistration_sha256="a" * 64
    )
    assert observed["ftv_transform_count"] == 40

    groups = payload["artifact_groups"]
    assert isinstance(groups, dict)
    hashes = payload["artifact_sha256"]
    assert isinstance(hashes, dict)
    s0_transform = next(
        path
        for path in groups["ftv_transforms"]
        if "local_response_state_multiseed_confirmation" in path
    )
    missing = copy.deepcopy(payload)
    missing["artifact_groups"]["ftv_transforms"].remove(s0_transform)
    missing["artifact_sha256"].pop(s0_transform)
    freeze = root / "manifests" / "representation_freeze.json"
    freeze.write_text(json.dumps(missing), encoding="utf-8")
    with pytest.raises(PermissionError, match="inventory shape drifted"):
        verify_representation_freeze(root, expected_preregistration_sha256="a" * 64)

    extra = copy.deepcopy(payload)
    extra_path = (
        "additional_experiments/residual_sph_grounding_pilot/metrics/unregistered.csv"
    )
    extra["artifact_groups"]["representation_aggregates"].append(extra_path)
    extra["artifact_sha256"][extra_path] = "0" * 64
    freeze.write_text(json.dumps(extra), encoding="utf-8")
    with pytest.raises(PermissionError, match="inventory shape drifted"):
        verify_representation_freeze(root, expected_preregistration_sha256="a" * 64)


def test_static_pcr_firewall_audit_is_complete_and_excludes_postfreeze_code() -> None:
    audit = build_pcr_firewall_audit(
        ROOT,
        preregistration_lock_sha256="a" * 64,
        implementation_lock_sha256="b" * 64,
    )
    assert audit["status"] == "PASS"
    assert audit["findings"] == []
    assert all(audit["checks"].values())
    assert "residual_sph.pcr_evaluation" not in audit["audited_local_modules"]
    assert not any(
        path.endswith("evaluate_pcr_postfreeze.py")
        or path.endswith("pcr_evaluation.py")
        for path in audit["audited_source_sha256"]
    )
    assert any(
        path.endswith("scripts/audit_pcr_firewall.py")
        for path in audit["audited_entrypoints"]
    )
    assert any(
        path.endswith("scripts/record_resource_guard.py")
        for path in audit["audited_entrypoints"]
    )
    assert any(
        path.endswith("scripts/run_representation_pipeline.py")
        for path in audit["audited_entrypoints"]
    )


def test_static_firewall_detects_dynamic_postfreeze_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    root = repo / "additional_experiments" / "residual_sph_grounding_pilot"
    script = root / "scripts" / "representation.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "from importlib import import_module\n"
        "import_module('residual_sph.pcr_evaluation')\n",
        encoding="utf-8",
    )
    package = root / "src" / "residual_sph"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "pcr_evaluation.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(
        evaluation_lock, "REPRESENTATION_ENTRYPOINTS", ("scripts/representation.py",)
    )
    audit = build_pcr_firewall_audit(
        root,
        preregistration_lock_sha256="a" * 64,
        implementation_lock_sha256="b" * 64,
    )
    assert audit["status"] == "FAIL"
    assert {
        finding["rule"] for finding in audit["findings"]
    } == {"postfreeze_module_imported_by_representation_phase"}


def test_firewall_attestation_rejects_source_drift(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    root = repo / "additional_experiments" / "residual_sph_grounding_pilot"
    shutil.copytree(ROOT / "scripts", root / "scripts")
    shutil.copytree(ROOT / "src", root / "src")
    manifest = root / "manifests" / "pcr_firewall_audit.json"
    manifest.parent.mkdir(parents=True)
    payload = build_pcr_firewall_audit(
        root,
        preregistration_lock_sha256="a" * 64,
        implementation_lock_sha256="b" * 64,
    )
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    assert verify_pcr_firewall_audit(
        root,
        expected_preregistration_sha256="a" * 64,
        expected_implementation_sha256="b" * 64,
    )["status"] == "PASS"
    probe = root / "scripts" / "run_probes.py"
    probe.write_text(probe.read_text(encoding="utf-8") + "\n# drift\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="stale or has an invalid shape"):
        verify_pcr_firewall_audit(
            root,
            expected_preregistration_sha256="a" * 64,
            expected_implementation_sha256="b" * 64,
        )


def test_postfreeze_executable_verifies_boundary_before_clinical_read() -> None:
    source = (ROOT / "scripts" / "evaluate_pcr_postfreeze.py").read_text(encoding="utf-8")
    freeze_position = source.index("verify_representation_freeze(")
    clinical_read_position = source.index("pd.read_csv(args.clinical_table")
    assert freeze_position < clinical_read_position
    assert "Representation-freeze" not in source  # exact call is executable, not a comment-only claim


def test_postfreeze_executable_reports_question_9_and_three_metric_effects() -> None:
    source = (ROOT / "scripts" / "evaluate_pcr_postfreeze.py").read_text(
        encoding="utf-8"
    )

    assert '"S2_CM_minus_C"' in source
    assert '"S0_CM_minus_C"' in source
    assert '"S2_C_plus_M_minus_C"' in source
    assert '"S0_C_plus_M_minus_C"' in source
    assert '"delta_auprc": float(result["delta_auprc"])' in source
    assert '"brier_improvement": float(result["brier_improvement"])' in source
    assert '"paired_metric_effect_summaries"' in source
    assert "bootstrap_seed=260_811 + bootstrap_counter" in source
    assert "bootstrap_counter += 1" in source

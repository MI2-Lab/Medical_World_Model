from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import common as common_module  # noqa: E402
from common import file_sha256  # noqa: E402
import freeze_preregistration as freeze  # noqa: E402
import generate_figures as figures  # noqa: E402
from generate_figures import pair_matched_oracle_deltas  # noqa: E402
from generate_report import (  # noqa: E402
    STAGE_A_AUTHORIZATION_KEYS,
    STAGE_A_RUN_SUMMARY_KEYS,
    _stage_b_text,
    two_seed_endpoint_support,
)
import run_feature_matrix as feature_matrix  # noqa: E402
from run_feature_matrix import (  # noqa: E402
    COMPLETION_KEYS,
    FEATURE_SHAPE_ZYX,
    ORACLE_REGIONS,
    REPRESENTATIVE_SELECTION_RULE,
    publish_completion_marker,
    validate_completion_marker,
    validate_representative_asset,
)
import validate_results as validator  # noqa: E402


def _synthetic_run_summary(
    root: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    (root / "configs").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "metrics").mkdir()
    config_path = root / "configs" / "audit.json"
    config_path.write_text('{"synthetic":true}\n', encoding="utf-8")
    for name in ("run_audit.py", "common.py"):
        (root / "scripts" / name).write_text(f"# synthetic {name}\n", encoding="utf-8")

    selected_cells: dict[str, object] = {}
    feature_hashes: dict[str, str] = {}
    feature_records: list[dict[str, object]] = []
    for seed in (2026, 3026):
        for arm in ("LOCAL0", "LOCAL3"):
            for fold in range(5):
                key = f"seed_{seed}/{arm}/fold_{fold}"
                feature = root / "features" / key / "spatial_statistics.private.npz"
                feature.parent.mkdir(parents=True)
                feature.write_bytes(f"synthetic-{key}".encode("ascii"))
                digest = file_sha256(feature)
                selected_cells[key] = {}
                feature_hashes[key] = digest
                feature_records.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "fold": fold,
                        "feature_sha256": digest,
                        "max_parity_abs": 0.0,
                    }
                )

    artifact_records: dict[str, object] = {}
    for index, (name, relative) in enumerate(
        validator.RUN_SUMMARY_ARTIFACT_PATHS.items()
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"synthetic artifact {index}\n", encoding="utf-8")
        private = name in validator.RUN_SUMMARY_PRIVATE_ARTIFACTS
        path.chmod(0o600 if private else 0o644)
        artifact_records[name] = {
            "path": relative,
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
            "patient_level_private": private,
        }

    lock_path = root / "PREREGISTRATION_LOCK.json"
    lock = {
        "selected_cells": selected_cells,
        "implementation_sha256": {
            "scripts/run_audit.py": file_sha256(root / "scripts" / "run_audit.py"),
            "scripts/common.py": file_sha256(root / "scripts" / "common.py"),
        },
    }
    lock_path.write_text(json.dumps(lock, sort_keys=True), encoding="utf-8")
    config = {
        "branch": "feature/spatial-heterogeneity-phenotype-audit",
        "upstream_code": {
            "complementarity_data_contracts_sha256": "a" * 64,
            "complementarity_modeling_sha256": "b" * 64,
        },
    }
    gates = {
        "scientific_classification": "MIXED",
        "stage_b_authorized": False,
    }
    feature_summary = {"cells": feature_records}
    summary: dict[str, object] = {
        "schema_version": 2,
        "experiment": "spatial_heterogeneity_phenotype_audit",
        "stage": "A",
        "status": "COMPLETE",
        "branch": config["branch"],
        "n_feature_assets": 20,
        "n_full_patients": 808,
        "n_ftv_complete_patients": 375,
        "scientific_classification": "MIXED",
        "stage_b_authorized": False,
        "elapsed_seconds": 1.25,
        "config_sha256": file_sha256(config_path),
        "preregistration_lock_sha256": file_sha256(lock_path),
        "preregistration_chain": {
            "preregistration_revision": 2,
            "active_preregistration_lock_sha256": file_sha256(lock_path),
            "preregistration_amendment_sha256": "c" * 64,
            "original_preregistration_lock_sha256": "d" * 64,
            "original_preregistration_commit": "e" * 40,
            "active_preregistration_commit": "f" * 40,
        },
        "feature_asset_sha256": feature_hashes,
        "reused_implementation_sha256": {
            "data_contracts": "a" * 64,
            "modeling": "b" * 64,
        },
        "runtime_implementation_sha256": dict(lock["implementation_sha256"]),
        "artifacts": artifact_records,
        "public_outputs_contain_patient_level_data": False,
    }
    (root / "metrics" / "run_summary.json").write_text(
        json.dumps(summary, sort_keys=True), encoding="utf-8"
    )
    return config, lock, gates, feature_summary


def test_oracle_delta_is_pair_matched_and_rejects_coverage_drift() -> None:
    rows = [
        {
            "seed": seed,
            "arm": "LOCAL3",
            "view": "T0",
            "target": "HER2",
            "population": "oracle_pair_CORE",
            "variant": variant,
            "n": 101,
            "auroc": value,
        }
        for seed in (2026, 3026)
        for variant, value in (("CORE", 0.72), ("FIXED_P3", 0.64))
    ]
    frame = pd.DataFrame(rows)
    paired = pair_matched_oracle_deltas(frame)
    assert len(paired) == 2
    np.testing.assert_allclose(paired["delta_auroc"], 0.08)
    frame.loc[frame["variant"].eq("FIXED_P3"), "n"] = 100
    with pytest.raises(ValueError, match="different coverage"):
        pair_matched_oracle_deltas(frame)


def test_gate_a_support_is_endpoint_specific() -> None:
    rows = []
    for seed in (2026, 3026):
        for target, gain in (("HER2", 0.04), ("subtype_4class", 0.02)):
            for variant, value in (("P1", 0.60), ("P3", 0.60 + gain)):
                rows.append(
                    {
                        "seed": seed,
                        "arm": "LOCAL3",
                        "view": "T0",
                        "target": target,
                        "variant": variant,
                        "population": "full_808",
                        "auroc": value,
                    }
                )
    support = two_seed_endpoint_support(
        pd.DataFrame(rows),
        targets=("HER2", "subtype_4class"),
        population="full_808",
        expected_seeds=(2026, 3026),
        threshold=0.03,
    )
    assert len(support["HER2"]) == 1
    assert support["subtype_4class"] == []


def test_stage_b_report_summarizes_actual_paired_deltas() -> None:
    table = pd.DataFrame(
        {
            "status": ["COMPLETE", "COMPLETE"],
            "target": ["HER2", "pCR"],
            "delta_auroc": [0.05, -0.01],
            "brier_improvement": [np.nan, 0.02],
        }
    )
    text = _stage_b_text(table, {"authorized": True})
    assert "phenotype ΔAUROC" in text
    assert "pCR ΔAUROC" in text
    assert "Brier" in text


def test_stage_a_provenance_schemas_are_synchronized() -> None:
    chain = {
        "preregistration_revision": 2,
        "active_preregistration_lock_sha256": "a" * 64,
        "preregistration_amendment_sha256": "b" * 64,
        "original_preregistration_lock_sha256": "c" * 64,
        "original_preregistration_commit": "d" * 40,
        "active_preregistration_commit": "e" * 40,
    }
    gates = {
        "gates": {"A": {"passed": False}, "C": {"passed": True}},
        "scientific_classification": "PHENOTYPE_IS_SPATIALLY_LOCALIZED",
    }
    authorization = validator.stage_b_authorization(
        {"stage_b": {"enabled": True}}, gates, chain, "f" * 64
    )
    authorization["stage_a_gates_sha256"] = "0" * 64
    assert set(authorization) == set(STAGE_A_AUTHORIZATION_KEYS)
    assert set(STAGE_A_RUN_SUMMARY_KEYS) == set(validator.RUN_SUMMARY_KEYS)


def test_representative_npz_has_strict_schema_dtype_and_hash(tmp_path: Path) -> None:
    shape = FEATURE_SHAPE_ZYX
    path = tmp_path / "representative.private.npz"
    np.savez(
        path,
        activation_mean_abs=np.ones(shape, dtype=np.float32),
        activation_channel_std=np.ones(shape, dtype=np.float32),
        local_weight=np.ones(shape, dtype=np.float32),
        region_weight=np.ones((4, *shape), dtype=np.float32),
        regions=np.asarray(ORACLE_REGIONS),
        selection_rule=np.asarray(REPRESENTATIVE_SELECTION_RULE),
    )
    path.chmod(0o600)
    validate_representative_asset(path, expected_sha256=file_sha256(path))
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: np.asarray(archive[name]) for name in archive.files}
    payload["activation_mean_abs"] = payload["activation_mean_abs"].astype(np.float64)
    np.savez(path, **payload)
    path.chmod(0o600)
    with pytest.raises(ValueError, match="shape/dtype/value"):
        validate_representative_asset(path)


def test_privacy_scan_is_a_closed_delivery_allowlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(validator, "ROOT", tmp_path)
    undeclared = tmp_path / "figures" / "raw_patient.png"
    undeclared.parent.mkdir()
    undeclared.write_bytes(b"raw image bytes")
    with pytest.raises(ValueError, match="undeclared file"):
        validator._privacy_scan([undeclared])

    public_csv = tmp_path / "metrics" / "hyperparameter_selections.csv"
    public_csv.parent.mkdir()
    public_csv.write_text("subject-id,value\npatient-001,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="patient identifier column"):
        validator._privacy_scan([public_csv])


def test_amendment_schema_is_exact_and_patient_free(tmp_path: Path) -> None:
    source = ROOT / "PREREGISTRATION_AMENDMENT.json"
    amendment = json.loads(source.read_text(encoding="utf-8"))
    path = tmp_path / source.name
    path.write_text(json.dumps(amendment), encoding="utf-8")
    observed = common_module.load_preregistration_amendment(path)
    assert (
        observed["geometry_qc"]["all_four_post_local_core_valid_patient_count"] == 373
    )
    assert observed["contains_patient_identifiers"] is False

    amendment["geometry_qc"]["representative_candidate_count_after"] = 375
    path.write_text(json.dumps(amendment), encoding="utf-8")
    with pytest.raises(ValueError, match="geometry QC"):
        common_module.load_preregistration_amendment(path)


def test_refreeze_historical_parity_allows_only_representative_amendment() -> None:
    historical_config = {
        "branch": "feature/spatial-heterogeneity-phenotype-audit",
        "oracle": {"comparison_population": "prefix_specific"},
        "analysis": {"gates": ["A", "B", "C", "D"]},
    }
    current_config = copy.deepcopy(historical_config)
    current_config["oracle"]["representative"] = {"candidate_count": 373}
    selected_cells = {"seed_2026/LOCAL3/fold_0": {"checkpoint_sha256": "a" * 64}}
    upstream = {"/sealed/module.py": "b" * 64}
    runtime = {"python": "3.11.14"}
    privacy = "c" * 64
    historical_lock = {
        "schema_version": 2,
        "status": freeze.ORIGINAL_LOCK_STATUS,
        "analysis_outputs_present_before_freeze": False,
        "analysis_outputs_before_freeze": [],
        "branch": current_config["branch"],
        "formal_cell_count": 1,
        "privacy_policy_sha256": privacy,
        "runtime_environment": runtime,
        "selected_cells": copy.deepcopy(selected_cells),
        "upstream_code_sha256": dict(upstream),
    }
    arguments = {
        "historical_lock": historical_lock,
        "historical_config": historical_config,
        "current_config": current_config,
        "selected_cells": selected_cells,
        "upstream_code_sha256": upstream,
        "runtime": runtime,
        "privacy_policy_sha256": privacy,
    }
    freeze._require_historical_parity(**arguments)

    drifted_cells = copy.deepcopy(selected_cells)
    drifted_cells["seed_2026/LOCAL3/fold_0"]["checkpoint_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="selected-cell assets"):
        freeze._require_historical_parity(
            **{**arguments, "selected_cells": drifted_cells}
        )

    with pytest.raises(ValueError, match="upstream code"):
        freeze._require_historical_parity(
            **{**arguments, "upstream_code_sha256": {"/sealed/module.py": "e" * 64}}
        )

    drifted_config = copy.deepcopy(current_config)
    drifted_config["analysis"]["gates"].append("UNDECLARED")
    with pytest.raises(ValueError, match="outside the representative-only amendment"):
        freeze._require_historical_parity(
            **{**arguments, "current_config": drifted_config}
        )

    with pytest.raises(ValueError, match="runtime_environment"):
        freeze._require_historical_parity(
            **{**arguments, "runtime": {"python": "3.12.0"}}
        )


def test_preregistration_chain_authenticates_both_committed_locks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    experiment = repo / "audit"
    experiment.mkdir(parents=True)

    def git(*arguments: str) -> str:
        return subprocess.check_output(["git", *arguments], cwd=repo, text=True).strip()

    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    git("config", "user.email", "audit@example.invalid")
    git("config", "user.name", "Audit Test")
    original_lock_path = experiment / "PREREGISTRATION_LOCK.json"
    original_lock_path.write_text('{"revision":1}\n', encoding="utf-8")
    git("add", ".")
    git("commit", "-q", "-m", "original preregistration")
    original_commit = git("rev-parse", "HEAD")
    original_lock_sha256 = file_sha256(original_lock_path)

    amendment = json.loads(
        (ROOT / "PREREGISTRATION_AMENDMENT.json").read_text(encoding="utf-8")
    )
    amendment["original_preregistration_commit"] = original_commit
    amendment["original_preregistration_lock_sha256"] = original_lock_sha256
    amendment_path = experiment / "PREREGISTRATION_AMENDMENT.json"
    amendment_path.write_text(json.dumps(amendment, sort_keys=True), encoding="utf-8")
    amended_lock = {
        "schema_version": 3,
        "preregistration_revision": 2,
        "status": common_module.AMENDED_LOCK_STATUS,
        "amendment_sha256": file_sha256(amendment_path),
        "superseded_preregistration_commit": original_commit,
        "superseded_preregistration_lock_sha256": original_lock_sha256,
    }
    original_lock_path.write_text(
        json.dumps(amended_lock, sort_keys=True), encoding="utf-8"
    )
    git("add", ".")
    git("commit", "-q", "-m", "amended preregistration")
    active_commit = git("rev-parse", "HEAD")

    monkeypatch.setattr(common_module, "REPO_ROOT", repo)
    monkeypatch.setattr(common_module, "EXPERIMENT_ROOT", experiment)
    monkeypatch.setattr(common_module, "AMENDMENT_PATH", amendment_path)
    monkeypatch.setattr(common_module, "LOCK_PATH", original_lock_path)
    chain = common_module.preregistration_chain(amended_lock)
    assert chain == {
        "preregistration_revision": 2,
        "active_preregistration_lock_sha256": file_sha256(original_lock_path),
        "preregistration_amendment_sha256": file_sha256(amendment_path),
        "original_preregistration_lock_sha256": original_lock_sha256,
        "original_preregistration_commit": original_commit,
        "active_preregistration_commit": active_commit,
    }

    original_lock_path.write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="committed amended lock"):
        common_module.preregistration_chain(amended_lock)


def test_run_summary_is_exact_and_authenticates_features_and_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, lock, gates, feature_summary = _synthetic_run_summary(tmp_path)
    monkeypatch.setattr(validator, "ROOT", tmp_path)
    path = tmp_path / "metrics" / "run_summary.json"
    expected_chain = json.loads(path.read_text(encoding="utf-8"))[
        "preregistration_chain"
    ]
    monkeypatch.setattr(
        validator, "preregistration_chain", lambda _lock: expected_chain
    )

    validated = validator._validate_run_summary(config, lock, gates, feature_summary)
    assert validated["status"] == "COMPLETE"

    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["undeclared"] = True
    path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="top-level schema"):
        validator._validate_run_summary(config, lock, gates, feature_summary)

    del summary["undeclared"]
    key = next(iter(summary["feature_asset_sha256"]))
    original_feature_hash = summary["feature_asset_sha256"][key]
    summary["feature_asset_sha256"][key] = "0" * 64
    path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="feature hashes"):
        validator._validate_run_summary(config, lock, gates, feature_summary)

    summary["feature_asset_sha256"][key] = original_feature_hash
    summary["runtime_implementation_sha256"]["scripts/common.py"] = "0" * 64
    path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="runtime implementation provenance"):
        validator._validate_run_summary(config, lock, gates, feature_summary)

    summary["runtime_implementation_sha256"]["scripts/common.py"] = lock[
        "implementation_sha256"
    ]["scripts/common.py"]
    summary["artifacts"]["phenotype_predictions"]["patient_level_private"] = False
    path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="privacy flag"):
        validator._validate_run_summary(config, lock, gates, feature_summary)


def test_formal_zero_option_clis_stop_on_help_or_unknown_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(freeze, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["freeze_preregistration.py", "--help"])
    with pytest.raises(SystemExit) as freeze_help:
        freeze.main()
    assert freeze_help.value.code == 0
    assert not (tmp_path / "PREREGISTRATION_LOCK.json").exists()

    monkeypatch.setattr(figures, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["generate_figures.py", "--unknown"])
    with pytest.raises(SystemExit) as figure_unknown:
        figures.main()
    assert figure_unknown.value.code != 0
    assert not (tmp_path / "figures").exists()
    assert not (tmp_path / "metrics").exists()


def test_freeze_rejects_preexisting_private_data_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data" / "patient.private.csv"
    data.parent.mkdir()
    data.write_text("patient_id\npatient-001\n", encoding="utf-8")
    monkeypatch.setattr(freeze, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["freeze_preregistration.py"])
    with pytest.raises(RuntimeError, match="outputs exist before preregistration"):
        freeze.main()
    assert not (tmp_path / "PREREGISTRATION_LOCK.json").exists()


def test_completion_marker_is_private_atomic_and_never_replaced(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "features" / "feature_matrix_complete.private.json"
    payload = {
        "schema_version": 1,
        "status": "COMPLETE",
        "cell_count": 20,
        "config_sha256": "a" * 64,
        "preregistration_lock_sha256": "b" * 64,
        "cells": [],
    }
    assert set(payload) == set(COMPLETION_KEYS)
    publish_completion_marker(marker, payload)
    before = (marker.stat().st_ino, marker.stat().st_mtime_ns, marker.read_bytes())
    assert marker.stat().st_mode & 0o777 == 0o600
    assert validate_completion_marker(marker, payload) == payload

    # Exact validation is allowed, but the complete marker is not rewritten.
    publish_completion_marker(marker, payload)
    after = (marker.stat().st_ino, marker.stat().st_mtime_ns, marker.read_bytes())
    assert after == before

    drifted = dict(payload)
    drifted["cell_count"] = 19
    with pytest.raises(ValueError, match="differs from current assets"):
        publish_completion_marker(marker, drifted)
    assert (
        marker.stat().st_ino,
        marker.stat().st_mtime_ns,
        marker.read_bytes(),
    ) == before


def test_feature_parent_authenticates_before_inspecting_or_creating_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    def authenticate() -> tuple[dict, dict]:
        events.append("authenticate")
        return {}, {}

    def path_for_cell(*_identity: object) -> Path:
        events.append("inspect_cell")
        return tmp_path / "absent.private.npz"

    monkeypatch.setattr(feature_matrix, "_authenticate_parent_context", authenticate)
    monkeypatch.setattr(feature_matrix, "feature_path", path_for_cell)
    monkeypatch.setattr(feature_matrix, "COMPLETION_PATH", tmp_path / "complete.json")
    monkeypatch.setattr(sys, "argv", ["run_feature_matrix.py"])
    feature_matrix.main()
    assert events[0] == "authenticate"
    assert "inspect_cell" in events[1:]
    assert not list(tmp_path.iterdir())


def test_feature_parent_authentication_failure_has_no_output_side_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def rejected() -> tuple[dict, dict]:
        raise ValueError("lock rejected")

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("output operation occurred before authentication")

    monkeypatch.setattr(feature_matrix, "_authenticate_parent_context", rejected)
    monkeypatch.setattr(feature_matrix, "feature_path", forbidden)
    monkeypatch.setattr(feature_matrix, "private_directory", forbidden)
    monkeypatch.setattr(feature_matrix, "COMPLETION_PATH", tmp_path / "complete.json")
    monkeypatch.setattr(sys, "argv", ["run_feature_matrix.py", "--execute"])
    with pytest.raises(ValueError, match="lock rejected"):
        feature_matrix.main()
    assert not list(tmp_path.iterdir())

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


sys.dont_write_bytecode = True
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
CONTRACT_PATH = EXPERIMENT_ROOT / "scripts" / "contracts.py"
SPEC = importlib.util.spec_from_file_location("compact_contracts_under_test", CONTRACT_PATH)
assert SPEC is not None and SPEC.loader is not None
contracts = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = contracts
SPEC.loader.exec_module(contracts)


def _write_config(tmp_path: Path, mutate=None) -> Path:
    payload = json.loads(json.dumps(contracts.EXPECTED_CONFIG))
    if mutate is not None:
        mutate(payload)
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _copy_pinned_sources(destination_root: Path) -> None:
    for key in contracts.SOURCE_PATH_KEYS:
        relative = contracts.EXPECTED_CONFIG["source_goal2"][key]
        source = REPO_ROOT / relative
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def test_default_config_resolves_and_verifies_all_goal2_sources() -> None:
    config = contracts.load_config()
    source = config["source_goal2"]
    for key in contracts.SOURCE_PATH_KEYS:
        assert isinstance(source[key], Path)
        assert source[key].is_file()
        assert contracts.file_sha256(source[key]) == source[f"{key}_sha256"]
    assert config["pca"]["dimensions"] == [8, 16, 32, 64]
    assert config["late_fusion"]["train_predictions"] == "strict_inner_oof"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.update({"unexpected": True}),
        lambda payload: payload["pca"].update({"dimensions": [8, 16, 32]}),
        lambda payload: payload["bootstrap"].update({"replicates": 2000.0}),
        lambda payload: payload["late_fusion"].update({"inner_seed": 1}),
    ],
)
def test_config_rejects_schema_setting_and_type_drift(tmp_path: Path, mutate) -> None:
    path = _write_config(tmp_path, mutate)
    with pytest.raises(contracts.ContractError, match="drifted"):
        contracts.load_config(path, repo_root=REPO_ROOT, verify_sources=False)


def test_config_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version": 1, "schema_version": 1}\n', encoding="utf-8")
    with pytest.raises(contracts.ContractError, match="duplicate key"):
        contracts.load_config(path, repo_root=REPO_ROOT, verify_sources=False)


def test_source_hash_drift_is_detected_in_temporary_repository(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _copy_pinned_sources(repo)
    config_path = _write_config(tmp_path)
    report = repo / contracts.EXPECTED_CONFIG["source_goal2"]["final_report"]
    report.write_text(report.read_text(encoding="utf-8") + "drift\n", encoding="utf-8")
    with pytest.raises(contracts.ContractError, match="SHA-256 mismatch"):
        contracts.load_config(config_path, repo_root=repo, verify_sources=True)


def test_alignment_preserves_requested_order_and_rejects_bad_ids() -> None:
    clinical = pd.DataFrame(
        {"patient_id": ["p2", "p1", "p3"], "label_pcr": [0, 1, 0]}
    )
    aligned = contracts.align_clinical(clinical, ["p3", "p1"])
    assert aligned["patient_id"].tolist() == ["p3", "p1"]
    assert aligned["label_pcr"].tolist() == [0, 1]
    with pytest.raises(contracts.ContractError, match="misses requested"):
        contracts.align_clinical(clinical, ["p4"])
    with pytest.raises(contracts.ContractError, match="duplicate"):
        contracts.align_clinical(clinical, ["p1", "p1"])


def test_population_mask_and_split_indices_are_fail_closed() -> None:
    asset = SimpleNamespace(patient_id=np.asarray(["p3", "p1", "p2"]))
    ftv = pd.DataFrame({"patient_id": ["p1", "p3"]})
    assert contracts.population_mask(asset, ftv, "full_808").tolist() == [True] * 3
    assert contracts.population_mask(asset, ftv, "ftv_complete_375").tolist() == [
        True,
        True,
        False,
    ]
    indices = contracts.split_indices(np.asarray(["train", "test", "val", "train"]))
    assert indices["train"].tolist() == [0, 3]
    assert indices["val"].tolist() == [2]
    assert indices["test"].tolist() == [1]
    with pytest.raises(contracts.ContractError, match="exactly train/val/test"):
        contracts.split_indices(np.asarray(["train", "test", "holdout"]))
    with pytest.raises(contracts.ContractError, match="population must be"):
        contracts.population_mask(asset, ftv, "all")


def test_atomic_writers_and_known_output_guard(tmp_path: Path) -> None:
    csv_path = tmp_path / "metrics" / "values.csv"
    json_path = tmp_path / "metrics" / "values.json"
    contracts.atomic_write_csv(pd.DataFrame({"value": [1, 2]}), csv_path)
    contracts.atomic_write_json({"status": "PASS", "value": 2}, json_path)
    assert pd.read_csv(csv_path)["value"].tolist() == [1, 2]
    assert json.loads(json_path.read_text(encoding="utf-8"))["status"] == "PASS"
    with pytest.raises(FileExistsError, match="already exist"):
        contracts.require_known_output_policy(
            [csv_path, json_path], False, output_root=tmp_path
        )
    assert contracts.require_known_output_policy(
        [csv_path, json_path], True, output_root=tmp_path
    ) == (csv_path.resolve(), json_path.resolve())
    with pytest.raises(contracts.ContractError, match="escapes"):
        contracts.require_known_output_policy(
            [tmp_path.parent / "outside.csv"], True, output_root=tmp_path
        )


def test_real_goal2_tables_and_one_local_asset_smoke() -> None:
    frozen = contracts.load_frozen_goal2_inputs(load_assets=False)
    assert len(frozen.fold_manifest) == 4040
    assert len(frozen.clinical) == 808
    assert len(frozen.ftv_wide) == 375
    assert frozen.assets == {}

    goal2 = contracts.load_goal2_contract_module()
    asset = goal2.load_local_feature_cell(
        frozen.goal2_config,
        frozen.fold_manifest,
        arm="LOCAL0",
        seed_base=2026,
        fold=0,
    )
    full = contracts.build_population_view(
        asset, frozen.clinical, frozen.ftv_wide, "full_808"
    )
    selected = contracts.build_population_view(
        asset, frozen.clinical, frozen.ftv_wide, "ftv_complete_375"
    )
    assert full.response_state.shape == (808, 4, 192)
    assert selected.response_state.shape == (375, 4, 192)
    assert selected.ftv_wide is not None
    assert selected.clinical["patient_id"].tolist() == selected.patient_id.tolist()

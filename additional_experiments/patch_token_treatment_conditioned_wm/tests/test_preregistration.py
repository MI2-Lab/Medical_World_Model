from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "freeze_preregistration.py"


def _module():
    spec = importlib.util.spec_from_file_location("patch_freeze", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_payload_has_pcr_firewall_and_complete_matrix() -> None:
    payload = _module().build_payload()
    assert (
        payload["outcome_firewall"]["pcr_loaded_during_world_model_training"] is False
    )
    contract = payload["scientific_contract"]
    assert contract["formal_new_cells"] == 10
    assert contract["condition_method"] == "condition_token"
    assert contract["token_mask_ratio"] == 0.5
    assert contract["bootstrap_draws"] >= 2000


def test_lock_hash_is_canonical() -> None:
    module = _module()
    payload = module.build_payload()
    contract = payload["scientific_contract"]
    assert payload["scientific_contract_sha256"] == module.canonical_sha256(contract)

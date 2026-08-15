from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_a0_import_matrix_and_privacy_contract_are_literal() -> None:
    source = (ROOT / "scripts" / "import_a0_reference.py").read_text()
    assert "SEEDS = (2026, 3026)" in source
    assert "FOLDS = tuple(range(5))" in source
    assert "local_response_state_multiseed_confirmation" in source
    assert "local_global_response_state_pilot" not in source
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "checkpoints/**" in ignore
    assert "features/**" in ignore

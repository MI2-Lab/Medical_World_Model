from __future__ import annotations

from pathlib import Path


def test_private_artifact_rules_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    ignore = (root / ".gitignore").read_text(encoding="utf-8")
    assert "*.private" in ignore
    assert "*.pt" in ignore
    assert "predictions/**/*.csv" in ignore


from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUN_MATRIX = _module("patch_atomic_run_matrix", "run_matrix.py")
TRAIN_CELL = _module("patch_atomic_train_cell", "train_cell.py")


@pytest.mark.parametrize("module", (RUN_MATRIX, TRAIN_CELL))
def test_atomic_json_replaces_only_after_strict_serialization(
    tmp_path: Path, module: object
) -> None:
    output = tmp_path / "manifest.json"
    output.write_text('{"status": "old"}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        module._atomic_json(output, {"value": float("nan")})
    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "old"}

    expected = {"status": "COMPLETE", "count": 10}
    module._atomic_json(output, expected)
    assert json.loads(output.read_text(encoding="utf-8")) == expected
    assert not tuple(tmp_path.glob(".manifest.json.*.tmp"))


def test_run_matrix_source_resumes_identical_terminal_manifest() -> None:
    source = (ROOT / "scripts" / "run_matrix.py").read_text(encoding="utf-8")
    assert "if existing != completion:" in source
    assert "_atomic_json(completion_path, completion)" in source
    assert ".write_text(" not in source


def test_train_cell_publishes_completion_marker_before_directory_rename() -> None:
    source = (ROOT / "scripts" / "train_cell.py").read_text(encoding="utf-8")
    marker = source.index('_atomic_json(working / "cell_complete.json", cell_manifest)')
    promotion = source.index("working.replace(final_output)")
    assert marker < promotion

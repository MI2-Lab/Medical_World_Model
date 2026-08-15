from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_public_artifacts.py"


def _module():
    spec = importlib.util.spec_from_file_location("patch_privacy", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gitignore_covers_all_private_roots() -> None:
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for name in ("checkpoints", "features", "predictions", "logs"):
        assert f"{name}/**" in ignore
    assert "manifests/*private*" in ignore
    assert "metrics/*private*" in ignore


def test_explicit_private_candidate_is_rejected(tmp_path: Path) -> None:
    module = _module()
    bad = ROOT / "metrics" / "unit_test.private.csv"
    bad.write_text("patient_id,y_true\nabc,1\n", encoding="utf-8")
    relative = bad.relative_to(module.REPO_ROOT).as_posix()
    try:
        with pytest.raises(RuntimeError):
            module.audit([relative])
    finally:
        bad.unlink(missing_ok=True)


def test_personal_absolute_path_is_rejected() -> None:
    module = _module()
    bad = ROOT / "metrics" / "unit_test_path.json"
    personal_path = "/data/" + "mi2-interns/example/private"
    bad.write_text(f'{{"source": "{personal_path}"}}\n', encoding="utf-8")
    relative = bad.relative_to(module.REPO_ROOT).as_posix()
    try:
        with pytest.raises(RuntimeError, match="personal absolute path"):
            module.audit([relative])
    finally:
        bad.unlink(missing_ok=True)


def test_probable_patient_identifier_is_rejected() -> None:
    module = _module()
    bad = ROOT / "metrics" / "unit_test_patient.json"
    patient_id = "ISPY2_" + "0001"
    bad.write_text(f'{{"patient": "{patient_id}"}}\n', encoding="utf-8")
    relative = bad.relative_to(module.REPO_ROOT).as_posix()
    try:
        with pytest.raises(RuntimeError, match="patient identifier"):
            module.audit([relative])
    finally:
        bad.unlink(missing_ok=True)

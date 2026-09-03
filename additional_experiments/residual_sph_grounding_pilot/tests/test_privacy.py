from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_public_artifacts.py"
SPEC = importlib.util.spec_from_file_location("residual_sph_privacy", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PRIVACY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PRIVACY)


def test_public_tree_excludes_private_runtime_directories() -> None:
    relative = [path.relative_to(PRIVACY.EXPERIMENT_ROOT) for path in PRIVACY.public_artifacts()]
    assert all(
        not any(part in PRIVACY.PRIVATE_DIRECTORIES for part in path.parts)
        for path in relative
    )


def test_scan_detects_identifier_path_and_restricted_payload(tmp_path: Path) -> None:
    identifier = "PRIVATE-SUBJECT-0007"
    leaking_text = tmp_path / "report.md"
    leaking_text.write_text(
        "source=" + "/" + "data" + "/private/cohort.csv subject=" + identifier + "\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_text("placeholder", encoding="utf-8")
    findings = PRIVACY.scan_paths([leaking_text, checkpoint], {identifier})
    assert len(findings["absolute_private_path_findings"]) == 1
    assert len(findings["identifier_findings"]) == 1
    assert len(findings["restricted_extension_findings"]) == 1


def test_relative_repository_paths_and_shebang_are_allowed(tmp_path: Path) -> None:
    source = tmp_path / "safe.py"
    source.write_text(
        "#!/usr/bin/env python3\npath='additional_experiments/example/config.json'\n",
        encoding="utf-8",
    )
    findings = PRIVACY.scan_paths([source], {"PRIVATE-SUBJECT-0007"})
    assert all(not rows for rows in findings.values())

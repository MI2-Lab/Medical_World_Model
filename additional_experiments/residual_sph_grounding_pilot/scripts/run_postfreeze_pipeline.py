#!/usr/bin/env python3
"""Plan or run the post-freeze evaluation and public-report pipeline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
SCRIPTS = EXPERIMENT_ROOT / "scripts"
SRC = EXPERIMENT_ROOT / "src"
sys.path.insert(0, str(SRC))

from residual_sph.evaluation_lock import verify_representation_freeze  # noqa: E402
from residual_sph.preregistration import (  # noqa: E402
    require_lock_sha256,
    verify_preregistration,
)


PRIVATE_DIRECTORIES = {"checkpoints", "features", "predictions", "logs", "__pycache__"}
PCR_OUTPUTS = (
    EXPERIMENT_ROOT / "predictions/pcr_oof.private.csv",
    EXPERIMENT_ROOT / "predictions/pcr_hyperparameters.private.csv",
    EXPERIMENT_ROOT / "metrics/table_pcr_complementarity.csv",
    EXPERIMENT_ROOT / "metrics/paired_bootstrap.csv",
    EXPERIMENT_ROOT / "metrics/pcr_effects.json",
)
FIGURES = (
    EXPERIMENT_ROOT / "figures/representation_effects.svg",
    EXPERIMENT_ROOT / "figures/sph_res_organization.svg",
    EXPERIMENT_ROOT / "figures/pcr_effects.svg",
)
REPORT = EXPERIMENT_ROOT / "reports/final_report.md"
PRIVACY_GATE = EXPERIMENT_ROOT / "metrics/public_artifact_privacy_gate.json"


@dataclass(frozen=True)
class Progress:
    name: str
    script: str
    completed: int
    total: int = 1

    @property
    def status(self) -> str:
        return "COMPLETE" if self.completed == self.total else "PENDING"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return value


def _pcr_progress(freeze_sha256: str) -> Progress:
    decision_path = EXPERIMENT_ROOT / "metrics/decision.json"
    execution_path = EXPERIMENT_ROOT / "metrics/execution_status.json"
    decision = _read_json(decision_path, label="decision sentinel")
    execution = _read_json(execution_path, label="execution sentinel")
    formal_decision = decision.get("status") == "FORMAL_TWO_SEED_PILOT_COMPLETE"
    formal_execution = execution.get("status") == "FORMAL_EXECUTION_COMPLETE"
    present = [path.exists() for path in PCR_OUTPUTS]
    if not any(present) and not formal_decision and not formal_execution:
        return Progress("postfreeze_pcr_evaluation", "scripts/evaluate_pcr_postfreeze.py", 0)
    if not all(present) or not formal_decision or not formal_execution:
        raise RuntimeError("post-freeze evaluation outputs are partial; refusing overwrite")
    for path in PCR_OUTPUTS:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"post-freeze output is missing or empty: {path}")
    effects = _read_json(
        EXPERIMENT_ROOT / "metrics/pcr_effects.json", label="pCR effects"
    )
    if (
        decision.get("pcr_evaluation_was_post_freeze") is not True
        or decision.get("representation_freeze_sha256") != freeze_sha256
        or decision.get("pcr_effects") != effects
        or execution.get("pcr_evaluation_was_post_freeze") is not True
        or execution.get("representation_freeze_sha256") != freeze_sha256
    ):
        raise RuntimeError("completed post-freeze evaluation is not bound to this freeze")
    return Progress("postfreeze_pcr_evaluation", "scripts/evaluate_pcr_postfreeze.py", 1)


def _figure_progress() -> Progress:
    present = [path.exists() for path in FIGURES]
    if not any(present):
        return Progress("aggregate_figures", "scripts/generate_figures.py", 0)
    if not all(present):
        raise RuntimeError("aggregate figures are partial; refusing overwrite")
    for path in FIGURES:
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise RuntimeError(f"aggregate figure is unreadable: {path}") from error
        if "<svg" not in source:
            raise RuntimeError(f"aggregate figure is not valid SVG text: {path}")
    return Progress("aggregate_figures", "scripts/generate_figures.py", 1)


def _report_progress(decision: dict[str, Any]) -> Progress:
    if not REPORT.is_file():
        return Progress("chinese_final_report", "scripts/generate_report.py", 0)
    text = REPORT.read_text(encoding="utf-8")
    pending_markers = (
        "尚未产生正式实验结果",
        "FORMAL_EXECUTION_NOT_STARTED_RESOURCE_GUARD",
    )
    if any(marker in text for marker in pending_markers):
        return Progress("chinese_final_report", "scripts/generate_report.py", 0)
    classification = decision.get("classification")
    required = ["# FTV + residual-SPH grounding pilot 最终报告"] + [
        f"{index}. **" for index in range(1, 13)
    ]
    if (
        not isinstance(classification, dict)
        or any(token not in text for token in required)
        or str(classification.get("representation")) not in text
        or str(classification.get("downstream")) not in text
    ):
        raise RuntimeError("non-pending final report does not satisfy its formal contract")
    return Progress("chinese_final_report", "scripts/generate_report.py", 1)


def _public_artifacts() -> list[Path]:
    paths: list[Path] = []
    for path in EXPERIMENT_ROOT.rglob("*"):
        if not path.is_file() or path == PRIVACY_GATE:
            continue
        relative = path.relative_to(EXPERIMENT_ROOT)
        if any(part in PRIVATE_DIRECTORIES for part in relative.parts):
            continue
        if path.suffix in {".pyc", ".pyo"} or ".private." in path.name:
            continue
        paths.append(path)
    return sorted(paths)


def _privacy_progress(*, report_complete: bool) -> Progress:
    if not report_complete or not PRIVACY_GATE.is_file():
        return Progress("public_privacy_audit", "scripts/audit_public_artifacts.py", 0)
    try:
        payload = _read_json(PRIVACY_GATE, label="public privacy audit")
    except RuntimeError:
        return Progress("public_privacy_audit", "scripts/audit_public_artifacts.py", 0)
    inventory = payload.get("artifact_sha256")
    if (
        payload.get("status") != "PASS"
        or payload.get("finding_count") != 0
        or not isinstance(inventory, dict)
    ):
        return Progress("public_privacy_audit", "scripts/audit_public_artifacts.py", 0)
    artifacts = _public_artifacts()
    current = {
        path.relative_to(REPO_ROOT).as_posix(): _sha256(path) for path in artifacts
    }
    if inventory != current:
        return Progress("public_privacy_audit", "scripts/audit_public_artifacts.py", 0)
    return Progress("public_privacy_audit", "scripts/audit_public_artifacts.py", 1)


def _inspect(freeze: dict[str, Any]) -> list[Progress]:
    freeze_sha256 = str(freeze["freeze_sha256"])
    pcr = _pcr_progress(freeze_sha256)
    figures = _figure_progress()
    decision = _read_json(
        EXPERIMENT_ROOT / "metrics/decision.json", label="decision"
    )
    report = _report_progress(decision)
    privacy = _privacy_progress(report_complete=report.completed == report.total)
    stages = [pcr, figures, report, privacy]
    incomplete_seen = False
    for stage in stages:
        if incomplete_seen and stage.completed:
            raise RuntimeError(
                f"out-of-order artifact detected at {stage.name}; pipeline is fail-closed"
            )
        if not stage.completed:
            incomplete_seen = True
    return stages


def _public_plan(
    stages: list[Progress], lock_sha256: str, freeze_sha256: str
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "pipeline": "postfreeze_evaluation_and_reporting",
        "mode": "plan",
        "scientific_preregistration_sha256": lock_sha256,
        "representation_freeze_sha256": freeze_sha256,
        "private_runtime_input": "supplied_but_path_redacted",
        "stages": [
            {
                "order": index,
                "name": stage.name,
                "script": stage.script,
                "status": stage.status,
            }
            for index, stage in enumerate(stages, start=1)
        ],
    }


def _run_child(script_name: str, arguments: list[str]) -> None:
    script = SCRIPTS / script_name
    if not script.is_file() or script.parent != SCRIPTS:
        raise FileNotFoundError(f"pipeline stage script is missing: {script_name}")
    subprocess.run(
        [sys.executable, str(script), *arguments],
        cwd=REPO_ROOT,
        check=True,
    )


def _execute(
    lock_sha256: str,
    freeze: dict[str, Any],
    *,
    clinical_table: Path,
) -> None:
    command_by_stage = {
        "postfreeze_pcr_evaluation": (
            "evaluate_pcr_postfreeze.py",
            [
                "--preregistration-lock-sha256",
                lock_sha256,
                "--clinical-table",
                str(clinical_table),
            ],
        ),
        "aggregate_figures": (
            "generate_figures.py",
            ["--preregistration-lock-sha256", lock_sha256],
        ),
        "chinese_final_report": ("generate_report.py", []),
        "public_privacy_audit": ("audit_public_artifacts.py", []),
    }
    for stage in _inspect(freeze):
        if stage.completed:
            continue
        script, arguments = command_by_stage[stage.name]
        _run_child(script, arguments)
        refreshed = {item.name: item for item in _inspect(freeze)}[stage.name]
        if not refreshed.completed:
            raise RuntimeError(f"pipeline stage did not complete: {stage.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "run"), default="plan")
    parser.add_argument("--preregistration-lock-sha256", required=True)
    parser.add_argument("--clinical-table", type=Path, required=True)
    args = parser.parse_args()

    preregistration = verify_preregistration(EXPERIMENT_ROOT)
    require_lock_sha256(
        preregistration["lock_sha256"], args.preregistration_lock_sha256
    )
    freeze = verify_representation_freeze(
        EXPERIMENT_ROOT,
        expected_preregistration_sha256=preregistration["lock_sha256"],
    )
    stages = _inspect(freeze)
    if args.mode == "plan":
        print(
            json.dumps(
                _public_plan(
                    stages,
                    preregistration["lock_sha256"],
                    str(freeze["freeze_sha256"]),
                ),
                indent=2,
            )
        )
        return
    _execute(
        preregistration["lock_sha256"],
        freeze,
        clinical_table=args.clinical_table,
    )
    print(json.dumps({"status": "POSTFREEZE_PIPELINE_COMPLETE"}, sort_keys=True))


if __name__ == "__main__":
    main()

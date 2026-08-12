from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


REPRESENTATION = _module("run_representation_pipeline.py")
POSTFREEZE = _module("run_postfreeze_pipeline.py")


def _all_strings(value: object):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _all_strings(key)
            yield from _all_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_strings(item)


def test_representation_plan_is_ordered_relative_and_complete() -> None:
    names = (
        "s0_reference_audit",
        "residualizers",
        "matrix_train",
        "matrix_export",
        "representation_probes",
        "aggregate_representation",
        "pcr_firewall_audit",
        "representation_freeze",
    )
    scripts = (
        "scripts/audit_s0_reference.py",
        "scripts/build_residualizers.py",
        "scripts/run_matrix.py",
        "scripts/run_matrix.py",
        "scripts/run_probes.py",
        "scripts/aggregate_representation.py",
        "scripts/audit_pcr_firewall.py",
        "scripts/freeze_representation.py",
    )
    totals = (1, 1, 30, 30, 40, 1, 1, 1)
    stages = [
        REPRESENTATION.Progress(name, script, 0, total)
        for name, script, total in zip(names, scripts, totals, strict=True)
    ]
    plan = REPRESENTATION._public_plan(stages, "a" * 64)
    assert [stage["name"] for stage in plan["stages"]] == list(names)
    assert [stage["total_units"] for stage in plan["stages"]] == list(totals)
    assert not any(value.startswith("/") for value in _all_strings(plan))
    json.dumps(plan, allow_nan=False)


def test_representation_runner_has_no_postfreeze_import_or_input_cli() -> None:
    path = ROOT / "scripts/run_representation_pipeline.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()
    cli_arguments: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            cli_arguments.update(
                str(argument.value)
                for argument in node.args
                if isinstance(argument, ast.Constant)
            )
    assert not any("clinical" in module.lower() for module in imports)
    assert "residual_sph.pcr_evaluation" not in imports
    assert not any("clinical" in argument.lower() for argument in cli_arguments)
    assert "evaluate_pcr_postfreeze.py" not in source
    assert "[sys.executable, str(script), *arguments]" in source
    assert "shell=True" not in source


def test_representation_runner_rejects_half_written_cell(tmp_path: Path) -> None:
    directory = tmp_path / "fold_0"
    directory.mkdir()
    one_of_three = directory / "selection.json"
    one_of_three.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="partial; refusing an unsafe overwrite"):
        REPRESENTATION._cell_directory_state(
            directory,
            (
                one_of_three,
                directory / "selected.pt",
                directory / "ftv_transform.json",
            ),
            label="train cell",
            validator=lambda: None,
        )


def test_postfreeze_plan_redacts_private_path_and_preserves_order() -> None:
    stages = [
        POSTFREEZE.Progress("postfreeze_pcr_evaluation", "scripts/evaluate_pcr_postfreeze.py", 0),
        POSTFREEZE.Progress("aggregate_figures", "scripts/generate_figures.py", 0),
        POSTFREEZE.Progress("chinese_final_report", "scripts/generate_report.py", 0),
        POSTFREEZE.Progress("public_privacy_audit", "scripts/audit_public_artifacts.py", 0),
    ]
    plan = POSTFREEZE._public_plan(stages, "a" * 64, "b" * 64)
    assert [stage["name"] for stage in plan["stages"]] == [
        "postfreeze_pcr_evaluation",
        "aggregate_figures",
        "chinese_final_report",
        "public_privacy_audit",
    ]
    assert not any(value.startswith("/") for value in _all_strings(plan))
    assert "supplied_but_path_redacted" in set(_all_strings(plan))


def test_postfreeze_main_verifies_freeze_before_execution() -> None:
    source = (ROOT / "scripts/run_postfreeze_pipeline.py").read_text(encoding="utf-8")
    main_source = source[source.index("def main()") :]
    assert main_source.index("verify_representation_freeze(") < main_source.index("_execute(")
    assert 'parser.add_argument("--clinical-table", type=Path, required=True)' in source
    assert "[sys.executable, str(script), *arguments]" in source
    assert "shell=True" not in source


def test_firewall_audits_representation_orchestrator() -> None:
    source = (ROOT / "src/residual_sph/evaluation_lock.py").read_text(encoding="utf-8")
    assert '"scripts/run_representation_pipeline.py"' in source

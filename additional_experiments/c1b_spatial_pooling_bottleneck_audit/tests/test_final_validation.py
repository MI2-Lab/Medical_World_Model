from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c1b_spatial_audit.final_validation import (  # noqa: E402
    ARMS,
    FEATURE_POOLINGS,
    FIGURE_FILES,
    FOLDS,
    FORMAL_PROBE_POOLINGS,
    POOLING_SLUGS,
    SEEDS,
    TABLE_SCHEMAS,
    UPSTREAM_RELATIVE,
    atomic_json,
    audit_conditional_s3,
    audit_final_inventories,
    audit_locked_inputs,
    audit_no_new_training,
    audit_p0_gates,
    audit_permissions,
    audit_public_deliverables,
    audit_report_links,
    canonical_sha256,
    file_sha256,
    scan_public_artifacts,
)


def _write(path: Path, payload: str | bytes, mode: int = 0o644) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, bytes):
        path.write_bytes(payload)
    else:
        path.write_text(payload, encoding="utf-8")
    os.chmod(path, mode)
    return path


def _minimal_public_root(tmp_path: Path) -> Path:
    root = tmp_path / "experiment"
    for name in ("configs", "manifests", "metrics", "reports"):
        (root / name).mkdir(parents=True, mode=0o755)
    _write(root / "EXPERIMENT_PLAN.md", "frozen plan\n")
    return root


def _init_git_for_privacy(tmp_path: Path, root: Path, *, private_rules: bool = True) -> Path:
    repo = tmp_path
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.org")
    _git(repo, "config", "user.name", "Synthetic Test")
    rules = "__pycache__/\n*.py[cod]\n"
    if private_rules:
        rules += (
            "features/**\nprobes/**\nlogs/**\n*private*\n"
            "!features/.gitkeep\n!probes/.gitkeep\n!logs/.gitkeep\n"
        )
    _write(root / ".gitignore", rules)
    return repo


def test_privacy_scan_exactly_cross_checks_private_ids_without_disclosure(
    tmp_path: Path,
) -> None:
    root = _minimal_public_root(tmp_path)
    _init_git_for_privacy(tmp_path, root)
    secret = "I-SPY2-987654"
    _write(root / "manifests/cohort.private.csv", f"patient_id,value\n{secret},1\n", 0o600)
    _write(root / "reports/final_report.md", f"leaked token: {secret}\n")

    result = scan_public_artifacts(root)

    assert result["status"] == "FAIL"
    assert result["private_identifier_values_checked"] == 1
    assert any(
        row["finding"] == "exact_private_patient_identifier"
        for row in result["identifier_path_or_column_findings"]
    )
    assert secret not in json.dumps(result, sort_keys=True)
    assert "cohort.private.csv" not in result["scanned_files_sha256"]


def test_privacy_scan_rejects_uid_path_sensitive_columns_and_bad_private_npz(
    tmp_path: Path,
) -> None:
    root = _minimal_public_root(tmp_path)
    _init_git_for_privacy(tmp_path, root)
    _write(root / "metrics/rows.csv", "patient_id,value\nredacted,1\n")
    _write(
        root / "reports/final_report.md",
        "debug /data/secret/input.nii.gz and UID 1.2.840.113619.2.55.3\n",
    )
    _write(root / "manifests/broken.private.npz", b"not-an-npz", 0o600)

    result = scan_public_artifacts(root)
    categories = {
        row["finding"] for row in result["identifier_path_or_column_findings"]
    }

    assert result["status"] == "FAIL"
    assert {"sensitive_csv_column", "absolute_workspace_path", "dicom_uid"} <= categories
    assert result["private_identifier_source_errors"] == 1


def test_privacy_scan_reads_npz_patient_ids_but_passes_safe_aggregates(
    tmp_path: Path,
) -> None:
    root = _minimal_public_root(tmp_path)
    _init_git_for_privacy(tmp_path, root)
    private = root / "manifests/audit.private.npz"
    np.savez(private, patient_id=np.asarray(["TOKEN-0001", "TOKEN-0002"]))
    os.chmod(private, 0o600)
    _write(root / "metrics/table.csv", "arm,count\nN1,2\n")

    result = scan_public_artifacts(root)

    assert result["status"] == "PASS"
    assert result["private_identifier_values_checked"] == 2


def test_privacy_scan_strips_web_urls_but_rejects_general_local_paths(
    tmp_path: Path,
) -> None:
    root = _minimal_public_root(tmp_path)
    _init_git_for_privacy(tmp_path, root)
    _write(
        root / "reports/final_report.md",
        "safe https://example.org/data/example.nii and http://host/Users/demo/file\n",
    )
    assert scan_public_artifacts(root)["status"] == "PASS"

    sensitive = (
        "/custom/workspace/results/file.txt C:\\Users\\name\\file.txt "
        "\\\\server\\share\\file.txt file:///secret/path ~/secret ${HOME}/secret"
    )
    _write(root / "reports/final_report.md", sensitive)
    failed = scan_public_artifacts(root)
    categories = {
        row["finding"] for row in failed["identifier_path_or_column_findings"]
    }
    assert failed["status"] == "FAIL"
    assert {"absolute_workspace_path", "home_or_file_uri_path"} <= categories
    serialized = json.dumps(failed, sort_keys=True)
    assert "/custom/workspace/results/file.txt" not in serialized
    assert "C:\\Users\\name" not in serialized


def test_privacy_scan_rejects_structured_private_paths_and_raw_report_paths(
    tmp_path: Path,
) -> None:
    root = _minimal_public_root(tmp_path)
    _init_git_for_privacy(tmp_path, root)
    private = _write(root / "features/final/secret.private.npz", b"x", 0o600)
    os.chmod(root / "features", 0o700)
    os.chmod(root / "features/final", 0o700)
    raw_value = "features/final/secret.private.npz"
    _write(root / "metrics/public.json", json.dumps({"checkpoint_path": raw_value}))
    _write(root / "reports/final_report.md", f"debug {raw_value}\n")

    result = scan_public_artifacts(root)
    categories = {
        row["finding"] for row in result["identifier_path_or_column_findings"]
    }
    assert result["status"] == "FAIL"
    assert {"unsafe_public_json_path_value", "raw_private_path_in_final_report"} <= categories
    serialized = json.dumps(result, sort_keys=True)
    assert raw_value not in serialized
    assert private.name not in serialized


def test_privacy_scan_fails_when_private_file_is_not_ignored_and_tokens_stale_names(
    tmp_path: Path,
) -> None:
    root = _minimal_public_root(tmp_path)
    _init_git_for_privacy(tmp_path, root, private_rules=False)
    secret_name = "patient_rows.private.csv"
    _write(root / "manifests" / secret_name, "patient_id\nTOKEN\n", 0o600)
    stale_name = "aggregate_smoke_sensitive.csv"
    _write(root / "metrics" / stale_name, "arm,count\nN1,1\n")
    result = scan_public_artifacts(root)
    assert result["status"] == "FAIL"
    serialized = json.dumps(result, sort_keys=True)
    assert secret_name not in serialized
    assert stale_name not in serialized
    assert all(set(row) == {"file_token"} for row in result["stale_smoke_or_limited_public_artifacts"])


def test_atomic_json_is_0644_and_refuses_unapproved_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "metrics/result.json"
    atomic_json(path, {"status": "PASS"})
    assert stat.S_IMODE(path.stat().st_mode) == 0o644
    try:
        atomic_json(path, {"status": "FAIL"})
    except FileExistsError:
        pass
    else:
        raise AssertionError("atomic_json silently overwrote an existing file")
    atomic_json(path, {"status": "REPLACED"}, overwrite=True)
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "REPLACED"
    assert stat.S_IMODE(path.stat().st_mode) == 0o644


def test_report_links_must_exist_and_remain_inside_experiment(tmp_path: Path) -> None:
    root = _minimal_public_root(tmp_path)
    _init_git_for_privacy(tmp_path, root)
    _write(root / "figures/plot.png", b"png")
    _write(
        root / "reports/final_report.md",
        "[plot](../figures/plot.png) [section](#answer) "
        "[source](https://example.org/paper)\n",
    )
    passed = audit_report_links(root)
    assert passed["status"] == "PASS"
    assert passed["local_links_checked"] == 1

    _write(
        root / "reports/final_report.md",
        "[missing](../figures/missing.png) [escape](../../outside.txt)\n",
    )
    failed = audit_report_links(root)
    categories = {row["finding"] for row in failed["findings"]}
    assert failed["status"] == "FAIL"
    assert {"markdown_link_target_missing", "markdown_link_escaped_experiment"} <= categories


def test_report_links_reject_private_symlink_directory_and_nonweb_schemes(
    tmp_path: Path,
) -> None:
    root = _minimal_public_root(tmp_path)
    _init_git_for_privacy(tmp_path, root)
    private = _write(root / "features/state.private.npz", b"private", 0o600)
    os.chmod(root / "features", 0o700)
    public = _write(root / "metrics/public.csv", "a\n1\n")
    link = root / "reports/alias.csv"
    link.symlink_to(public)
    _write(
        root / "reports/final_report.md",
        " ".join(
            (
                "[private](../features/state.private.npz)",
                "[directory](../metrics)",
                "[symlink](alias.csv)",
                "[file](file:///tmp/x)",
                "[data](data:text/plain,x)",
                "[ssh](ssh://host/x)",
                "[vscode](vscode://file/x)",
                "[root](/metrics/public.csv)",
            )
        ),
    )
    result = audit_report_links(root)
    categories = {row["finding"] for row in result["findings"]}
    assert result["status"] == "FAIL"
    assert {
        "markdown_link_target_private",
        "markdown_link_target_not_regular",
        "markdown_link_target_symlink",
        "forbidden_markdown_link_scheme",
        "non_relative_local_markdown_link",
    } <= categories
    assert private.name not in json.dumps(result)


def test_permission_audit_distinguishes_public_and_private_assets(tmp_path: Path) -> None:
    root = tmp_path / "experiment"
    root.mkdir(mode=0o755)
    _init_git_for_privacy(tmp_path, root)
    report = _write(root / "reports/final_report.md", "safe\n", 0o644)
    os.chmod(root / "reports", 0o755)
    (root / "features").mkdir(mode=0o700)
    private = _write(root / "features/state.private.npz", b"state", 0o600)

    passed = audit_permissions(root, tmp_path, public_files=[report])
    assert passed["status"] == "PASS"

    os.chmod(private, 0o644)
    failed = audit_permissions(root, tmp_path, public_files=[report])
    assert failed["status"] == "FAIL"
    serialized = json.dumps(failed, sort_keys=True)
    assert "state.private.npz" not in serialized
    assert failed["private_paths_disclosed"] is False


def test_private_git_hygiene_rejects_nonignored_and_cached_private_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / "experiment"
    root.mkdir(mode=0o755)
    _init_git_for_privacy(tmp_path, root, private_rules=False)
    report = _write(root / "reports/final_report.md", "safe\n")
    private = _write(root / "manifests/patient_rows.private.csv", "patient_id\nX\n", 0o600)

    nonignored = audit_permissions(root, tmp_path, public_files=[report])
    assert nonignored["status"] == "FAIL"
    assert any(
        row["finding"] == "private_path_not_git_ignored"
        for row in nonignored["findings"]
    )
    assert private.name not in json.dumps(nonignored, sort_keys=True)

    _write(root / ".gitignore", "*private*\n")
    _git(tmp_path, "add", "-f", private.relative_to(tmp_path).as_posix())
    cached = audit_permissions(root, tmp_path, public_files=[report])
    assert cached["status"] == "FAIL"
    assert any(
        row["finding"] == "private_path_cached_by_git" for row in cached["findings"]
    )
    assert private.name not in json.dumps(cached, sort_keys=True)


def test_no_training_audit_requires_false_declarations(tmp_path: Path) -> None:
    root = _minimal_public_root(tmp_path)
    _write(root / "configs/audit.json", '{"training_forbidden": true}\n')
    _write(root / "PREREGISTRATION_LOCK.json", '{"new_training_performed": false}\n')
    _write(root / "features/cell.private.metadata.json", '{"training_performed": false}\n', 0o600)

    assert audit_no_new_training(root)["status"] == "PASS"
    _write(root / "features/cell.private.metadata.json", '{"training_performed": true}\n', 0o600)
    failed = audit_no_new_training(root)
    assert failed["status"] == "FAIL"
    assert any(row["finding"] == "training_declaration_not_false" for row in failed["findings"])


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _cell_names() -> list[str]:
    return [
        f"seed_{seed}/{arm}/fold_{fold}"
        for seed in SEEDS
        for arm in ARMS
        for fold in FOLDS
    ]


def _write_csv(path: Path, columns: list[str] | tuple[str, ...], rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)
    os.chmod(path, 0o644)
    return path


def _build_p0_gate_fixture(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    experiment = repo / "additional_experiments/c1b_spatial_pooling_bottleneck_audit"
    references: dict[str, dict[str, str]] = {}
    equivalence_rows: list[dict[str, object]] = []
    for index, cell in enumerate(_cell_names()):
        reference = _write(
            repo / UPSTREAM_RELATIVE / "features/formal_4x8_restart1" / cell
            / "response_state.private.npz",
            f"reference-{index}",
            0o600,
        )
        metadata = _write(reference.with_suffix(".metadata.json"), "{}\n", 0o600)
        candidate = _write(
            experiment / "features/final" / cell / "p0.private.npz",
            f"candidate-{index}",
            0o600,
        )
        references[cell] = {
            "feature_path": reference.relative_to(repo).as_posix(),
            "feature_sha256": file_sha256(reference),
            "feature_metadata_path": metadata.relative_to(repo).as_posix(),
            "feature_metadata_sha256": file_sha256(metadata),
        }
        seed, arm, fold = cell.split("/")
        equivalence_rows.append(
            {
                "arm": arm,
                "seed_base": seed.removeprefix("seed_"),
                "fold": fold.removeprefix("fold_"),
                "pooling": "P0",
                "patients": 808,
                "visits": 3232,
                "feature_dim": 192,
                "elements": 620544,
                "allclose_fraction": 1.0,
                "bitwise_equal_fraction": 1.0,
                "max_absolute_error": 0.0,
                "mean_absolute_error": 0.0,
                "rmse": 0.0,
                "finite_fraction": 1.0,
                "identity_exact": True,
                "split_exact": True,
                "state_valid_fraction": 1.0,
                "rtol": 1e-5,
                "atol": 1e-6,
                "status": "PASS",
                "candidate_sha256": file_sha256(candidate),
                "reference_sha256": file_sha256(reference),
            }
        )
    lock_path = _write(
        experiment / "PREREGISTRATION_LOCK.json",
        json.dumps({"formal_p0_references": references}, sort_keys=True),
    )
    lock_sha = file_sha256(lock_path)
    eq_columns = (
        "arm", "seed_base", "fold", "pooling", "patients", "visits", "feature_dim",
        "elements", "allclose_fraction", "bitwise_equal_fraction", "max_absolute_error",
        "mean_absolute_error", "rmse", "finite_fraction", "identity_exact", "split_exact",
        "state_valid_fraction", "rtol", "atol", "status", "candidate_sha256",
        "reference_sha256",
    )
    _write_csv(experiment / "metrics/p0_equivalence_by_cell.csv", eq_columns, equivalence_rows)
    _write(
        experiment / "metrics/p0_equivalence_gate.json",
        json.dumps(
            {
                "schema_version": 1, "status": "PASS", "formal_cells": 40,
                "patients_per_cell": 808, "visits_per_patient": 4,
                "feature_dimension": 192, "compared_elements": 24821760,
                "allclose_required_fraction": 1.0, "allclose_observed_fraction": 1.0,
                "bitwise_equal_fraction": 1.0, "maximum_absolute_error": 0.0,
                "mean_absolute_error": 0.0, "rtol": 1e-5, "atol": 1e-6,
                "preregistration_lock_sha256": lock_sha,
                "probe_execution_authorized": True,
            },
            sort_keys=True,
        ),
    )
    replication_columns = (
        "seed_base", "arm", "fold", "selection_rows", "prediction_rows",
        "selection_contract_exact", "prediction_keys_exact", "prediction_contract_exact",
        "prediction_allclose", "maximum_prediction_absolute_difference",
        "y_true_max_abs_difference", "y_pred_max_abs_difference",
        "y_true_analysis_max_abs_difference", "y_pred_analysis_max_abs_difference",
        "b0_prediction_max_abs_difference", "b0_prediction_analysis_max_abs_difference",
        "status",
    )
    replication_rows = []
    for index, cell in enumerate(_cell_names()):
        seed, arm, fold = cell.split("/")
        replication_rows.append(
            {
                "seed_base": seed.removeprefix("seed_"), "arm": arm,
                "fold": fold.removeprefix("fold_"), "selection_rows": 14,
                "prediction_rows": 1066 if index == 39 else 1042,
                "selection_contract_exact": True, "prediction_keys_exact": True,
                "prediction_contract_exact": True, "prediction_allclose": True,
                "maximum_prediction_absolute_difference": 0.0,
                "y_true_max_abs_difference": 0.0, "y_pred_max_abs_difference": 0.0,
                "y_true_analysis_max_abs_difference": 0.0,
                "y_pred_analysis_max_abs_difference": 0.0,
                "b0_prediction_max_abs_difference": 0.0,
                "b0_prediction_analysis_max_abs_difference": 0.0, "status": "PASS",
            }
        )
    _write_csv(
        experiment / "metrics/p0_probe_replication_by_cell.csv",
        replication_columns,
        replication_rows,
    )
    pooled_columns = (
        "seed_base", "arm", "task", "target_name", "endpoint", "analysis_scope",
        "target_semantics", "scale", "maximum_metric_absolute_difference", "status",
    )
    pooled_rows = [
        {
            "seed_base": 2026 if index < 72 else 3026, "arm": ARMS[index % 4],
            "task": "static", "target_name": "FTV", "endpoint": f"E{index}",
            "analysis_scope": "primary", "target_semantics": "natural", "scale": "natural",
            "maximum_metric_absolute_difference": 0.0, "status": "PASS",
        }
        for index in range(144)
    ]
    _write_csv(
        experiment / "metrics/p0_probe_replication_pooled_metrics.csv",
        pooled_columns,
        pooled_rows,
    )
    _write(
        experiment / "metrics/p0_probe_replication_gate.json",
        json.dumps(
            {
                "schema_version": 1, "status": "PASS", "formal_cells": 40,
                "selection_cells": 560, "outer_test_prediction_rows": 41704,
                "selection_contract_exact_fraction": 1.0,
                "prediction_contract_exact_fraction": 1.0,
                "prediction_allclose_fraction": 1.0,
                "maximum_prediction_absolute_difference": 0.0,
                "pooled_natural_metric_rows": 144,
                "maximum_pooled_metric_absolute_difference": 0.0,
                "prediction_rtol": 1e-5, "prediction_atol": 1e-6,
                "pooled_metric_atol": 1e-6,
                "alternate_pooling_interpretation_authorized": True,
            },
            sort_keys=True,
        ),
    )
    return experiment, repo


def test_p0_stop_gates_require_exact_json_csv_counts_and_live_hashes(tmp_path: Path) -> None:
    experiment, repo = _build_p0_gate_fixture(tmp_path)
    passed = audit_p0_gates(experiment, repo)
    assert passed["status"] == "PASS"
    assert passed["equivalence_cells_checked"] == 40
    assert passed["replication_pooled_rows_checked"] == 144

    gate_path = experiment / "metrics/p0_probe_replication_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["outer_test_prediction_rows"] = 41703
    _write(gate_path, json.dumps(gate, sort_keys=True))
    failed = audit_p0_gates(experiment, repo)
    assert failed["status"] == "FAIL"
    assert any(
        row["finding"] == "p0_probe_replication_gate_or_csv_invalid"
        for row in failed["findings"]
    )


def _prospective_payload(*, strong: bool) -> dict[str, object]:
    oracle = {"status": "SUPPORTED_IN_PILOT" if strong else "NOT_SUPPORTED_IN_PILOT", "supported": strong}
    if strong:
        conditional = {
            "trigger_status": "NOT_TRIGGERED_FINAL_ORACLE_STRONG",
            "strong_oracle_recovery": None,
            "deployable_local_recovery": None,
        }
        classification = {
            "code": "A", "classification": "A POOLING BOTTLENECK",
            "next": "Local–Global Response State Pilot",
        }
    else:
        conditional = {
            "trigger_status": "TRIGGERED_FINAL_ORACLE_WEAK_COMPLETED",
            "strong_oracle_recovery": {"status": "NOT_SUPPORTED_IN_PILOT", "supported": False},
            "deployable_local_recovery": {"status": "NOT_SUPPORTED_IN_PILOT", "supported": False},
        }
        classification = {
            "code": "C", "classification": "C ENCODER BOTTLENECK",
            "next": "Stronger Pretrained 3-D Encoder Pilot",
        }
    return {
        "schema_version": 1,
        "status": "COMPLETE",
        "natural_metrics": "pooled_five_outer_test_folds_before_metric",
        "transformed_metrics": "outer_fold_summaries_only",
        "new_training_performed": False,
        "probe_refit_during_aggregation": False,
        "final_stage": {
            "strong_oracle_recovery": oracle,
            "deployable_local_recovery": {"status": "SUPPORTED_IN_PILOT", "supported": True},
            "padding_geometry_evidence": {"status": "NOT_SUPPORTED_IN_PILOT", "supported": False},
        },
        "conditional_s3": conditional,
        "training_budget": {"status": "SECONDARY_CONFOUND_ONLY"},
        "classification": classification,
    }


def _build_public_deliverables(root: Path) -> None:
    for name, columns in TABLE_SCHEMAS.items():
        if name == "table1_feature_map_contract.csv":
            count = 4
        elif name == "table7_training_budget.csv":
            count = 40
        else:
            count = 1
        rows: list[dict[str, object]] = []
        for index in range(count):
            row: dict[str, object] = {column: "x" for column in columns}
            if name == "table7_training_budget.csv":
                cell = _cell_names()[index]
                seed, arm, fold = cell.split("/")
                row.update(
                    seed=seed.removeprefix("seed_"),
                    arm=arm,
                    fold=fold.removeprefix("fold_"),
                )
            rows.append(row)
        _write_csv(root / "metrics" / name, columns, rows)
    prospective = _prospective_payload(strong=True)
    _write(root / "metrics/prospective_gates.json", json.dumps(prospective, sort_keys=True))
    classification = prospective["classification"]
    report = (
        f"FINAL_CLASSIFICATION: {classification['classification']}\n\n"
        f"NEXT: {classification['next']}\n\n"
        + "\n".join(f"### {index}. Answer {index}\n\nSynthetic answer." for index in range(1, 15))
        + "\n"
    )
    _write(root / "reports/final_report.md", report)
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo

    for index, name in enumerate(FIGURE_FILES):
        image = Image.new("RGB", (512, 256), (index, 20, 30))
        image.putpixel((0, 0), (255, 255, 255))
        metadata = PngInfo()
        metadata.add_text("Software", "c1b_spatial_audit")
        metadata.add_text("Title", f"Synthetic Figure {index + 1}")
        path = root / "figures" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path, format="PNG", dpi=(200, 200), pnginfo=metadata)
        os.chmod(path, 0o644)


def test_public_deliverables_require_exact_tables_report_and_decodable_pngs(
    tmp_path: Path,
) -> None:
    root = _minimal_public_root(tmp_path)
    _build_public_deliverables(root)
    passed = audit_public_deliverables(root)
    assert passed["status"] == "PASS"
    assert passed["validated_registered_png_figures"] == 12
    assert passed["numbered_report_answers"] == 14

    broken = root / "figures" / FIGURE_FILES[0]
    _write(broken, b"", 0o644)
    failed = audit_public_deliverables(root)
    assert failed["status"] == "FAIL"
    assert any(
        row["finding"] == "registered_png_decode_or_contract_invalid"
        for row in failed["findings"]
    )


def _build_s3_trigger_fixture(tmp_path: Path, *, strong: bool) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    experiment = repo / "additional_experiments/c1b_spatial_pooling_bottleneck_audit"
    for relative in (
        "src/c1b_spatial_audit/s3_trigger.py",
        "src/c1b_spatial_audit/analysis.py",
        "src/c1b_spatial_audit/probes.py",
        "src/c1b_spatial_audit/s3_probe_runner.py",
        "src/c1b_spatial_audit/probe_runner.py",
        "src/c1b_spatial_audit/s3_exporter.py",
        "src/c1b_spatial_audit/s3_sidecars.py",
        "src/c1b_spatial_audit/pooling.py",
        "src/c1b_spatial_audit/runtime.py",
        "src/c1b_spatial_audit/contracts.py",
        "scripts/run_s3_feature_matrix.py",
        "scripts/export_s3_frozen_features.py",
    ):
        _write(experiment / relative, relative + "\n")
    equivalence = _write(experiment / "metrics/p0_equivalence_gate.json", "{}\n")
    replication = _write(experiment / "metrics/p0_probe_replication_gate.json", "{}\n")
    plan_sha = "a" * 64
    config_sha = "b" * 64
    lock_path = _write(
        experiment / "PREREGISTRATION_LOCK.json",
        json.dumps({"plan_sha256": plan_sha, "config_sha256": config_sha}),
    )
    final_root = experiment / "probes/final"
    metadata_inventory: dict[str, str] = {}
    for seed in SEEDS:
        for arm in ARMS:
            for fold in FOLDS:
                for pooling in FORMAL_PROBE_POOLINGS:
                    if pooling in {"PVALID", "PORACLE", "PLOCAL+PVALID_SECONDARY"} and arm.startswith("L"):
                        continue
                    path = (
                        final_root / f"seed_{seed}" / arm / f"fold_{fold}"
                        / POOLING_SLUGS[pooling] / "probe_metadata.json"
                    )
                    _write(path, "{}\n", 0o600)
                    metadata_inventory[path.relative_to(final_root).as_posix()] = file_sha256(path)
    assert len(metadata_inventory) == 180
    prospective = _prospective_payload(strong=strong)
    _write(experiment / "metrics/prospective_gates.json", json.dumps(prospective, sort_keys=True))
    oracle = prospective["final_stage"]["strong_oracle_recovery"]
    trigger = {
        "schema_version": 1,
        "status": (
            "NOT_TRIGGERED_FINAL_ORACLE_STRONG" if strong else "TRIGGERED_FINAL_ORACLE_WEAK"
        ),
        "s3_execution_authorized": not strong,
        "decision_contract": "final_stage_strong_oracle_recovery_false",
        "final_stage_strong_oracle_recovery": oracle,
        "final_probe_root": "probes/final",
        "final_probe_cell_count": 180,
        "final_probe_metadata_inventory_sha256": canonical_sha256(metadata_inventory),
        "preregistration_lock_sha256": file_sha256(lock_path),
        "plan_sha256": plan_sha,
        "config_sha256": config_sha,
        "p0_equivalence_gate_sha256": file_sha256(equivalence),
        "p0_probe_replication_gate_sha256": file_sha256(replication),
        "trigger_implementation_sha256": file_sha256(experiment / "src/c1b_spatial_audit/s3_trigger.py"),
        "analysis_implementation_sha256": file_sha256(experiment / "src/c1b_spatial_audit/analysis.py"),
        "probe_adapter_sha256": file_sha256(experiment / "src/c1b_spatial_audit/probes.py"),
        "new_training_performed": False,
        "probe_refit_performed": False,
        "patient_identifiers_present": False,
    }
    _write(
        experiment / "metrics/s3_trigger_authorization.json",
        json.dumps(trigger, sort_keys=True),
    )
    return experiment, repo


def test_s3_strong_branch_requires_exact_zero_assets(tmp_path: Path) -> None:
    experiment, repo = _build_s3_trigger_fixture(tmp_path, strong=True)
    passed = audit_conditional_s3(experiment, repo)
    assert passed["status"] == "PASS"
    leak = _write(experiment / "features/s3/unexpected.private.npz", b"x", 0o600)
    failed = audit_conditional_s3(experiment, repo)
    assert failed["status"] == "FAIL"
    assert leak.name not in json.dumps(failed, sort_keys=True)


def test_s3_weak_branch_fails_closed_until_exact_100_assets_complete(tmp_path: Path) -> None:
    experiment, repo = _build_s3_trigger_fixture(tmp_path, strong=False)
    failed = audit_conditional_s3(experiment, repo)
    assert failed["status"] == "FAIL"
    categories = {row["finding"] for row in failed["findings"]}
    assert {
        "s3_feature_controls_invalid",
        "s3_feature_inventory_not_exact_100",
        "s3_probe_completion_or_asset_invalid",
    } <= categories


def test_locked_input_audit_checks_every_live_hash_and_git_tree(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.org")
    _git(repo, "config", "user.name", "Synthetic Test")
    old = repo / UPSTREAM_RELATIVE
    selected: dict[str, dict[str, object]] = {}
    references: dict[str, dict[str, object]] = {}
    for index, cell in enumerate(_cell_names()):
        checkpoint = _write(old / "checkpoints/formal_4x8_restart1" / cell / "selected.pt", f"ckpt-{index}")
        feature = _write(old / "features/formal_4x8_restart1" / cell / "response_state.private.npz", f"feature-{index}", 0o600)
        feature_metadata = _write(feature.with_suffix(".metadata.json"), f'{{"cell": {index}}}\n', 0o600)
        probe_hashes: dict[str, str] = {}
        for filename in (
            "probe_metadata.json",
            "probe_metrics.csv",
            "ridge_predictions.private.csv",
            "ridge_selection.csv",
        ):
            output = _write(old / "predictions/formal_4x8_restart1" / cell / filename, f"{filename}-{index}", 0o600)
            probe_hashes[filename] = file_sha256(output)
        selected[cell] = {
            "path": checkpoint.relative_to(repo).as_posix(),
            "sha256": file_sha256(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
            "mtime_ns": checkpoint.stat().st_mtime_ns,
        }
        references[cell] = {
            "feature_path": feature.relative_to(repo).as_posix(),
            "feature_sha256": file_sha256(feature),
            "feature_metadata_path": feature_metadata.relative_to(repo).as_posix(),
            "feature_metadata_sha256": file_sha256(feature_metadata),
            "patient_order_sha256": "a" * 64,
            "probe_outputs_sha256": probe_hashes,
        }

    completion_names = (
        "checkpoints/formal_4x8_restart1/matrix_complete.json",
        "features/formal_4x8_restart1/feature_export_complete.json",
        "manifests/stage_b_data_contract.private.json",
        "metrics/stage_b_aggregation_summary.json",
        "predictions/formal_4x8_restart1/postprocessing_complete.json",
    )
    completion = {}
    for index, relative in enumerate(completion_names):
        path = _write(old / relative, f"completion-{index}", 0o600)
        completion[relative] = file_sha256(path)

    source_names = (
        "additional_experiments/g3_multiseed_generalization/src/dgrs/model.py",
        "additional_experiments/c1b_overlap_eligibility_ftv_stageb/src/c1b_stage_b/probes.py",
        "additional_experiments/c1b_overlap_eligibility_ftv_stageb/src/c1b_stage_b/targets.py",
        "additional_experiments/c1b_model_ready_ftv_sanity/src/c1b_sanity/geometry.py",
    )
    sources = {}
    for index, relative in enumerate(source_names):
        path = _write(repo / relative, f"source-{index}\n")
        sources[relative] = file_sha256(path)

    _git(repo, "add", "additional_experiments")
    _git(repo, "commit", "-m", "synthetic upstream")
    source_commit = _git(repo, "rev-parse", "HEAD")
    old_tree = _git(repo, "rev-parse", f"HEAD:{UPSTREAM_RELATIVE.as_posix()}")

    experiment = repo / "additional_experiments/c1b_spatial_pooling_bottleneck_audit"
    plan = _write(experiment / "EXPERIMENT_PLAN.md", "frozen\n")
    config = _write(experiment / "configs/audit.json", "{}\n")
    lock = {
        "formal_cell_count": 40,
        "plan_sha256": file_sha256(plan),
        "config_sha256": file_sha256(config),
        "selected_checkpoints": selected,
        "formal_p0_references": references,
        "upstream_completion_sha256": completion,
        "upstream_source_sha256": sources,
        "source_commit": source_commit,
        "upstream_tracked_tree": old_tree,
    }
    _write(experiment / "PREREGISTRATION_LOCK.json", json.dumps(lock, sort_keys=True))

    passed = audit_locked_inputs(experiment, repo)
    assert passed["status"] == "PASS"
    assert passed["checkpoint_files"] == 40
    assert passed["reference_feature_files"] == 80
    assert passed["reference_probe_files"] == 160
    privacy_passed = scan_public_artifacts(experiment, repo)
    assert privacy_passed["status"] == "PASS"
    mutated_lock = json.loads(json.dumps(lock))
    mutated_lock["selected_checkpoints"][_cell_names()[0]]["path"] = (
        "features/relative_checkpoint.private.pt"
    )
    _write(experiment / "PREREGISTRATION_LOCK.json", json.dumps(mutated_lock, sort_keys=True))
    privacy_failed = scan_public_artifacts(experiment, repo)
    assert privacy_failed["status"] == "FAIL"
    assert any(
        row["finding"] == "preregistration_lock_checkpoint_path_drift"
        for row in privacy_failed["identifier_path_or_column_findings"]
    )
    _write(experiment / "PREREGISTRATION_LOCK.json", json.dumps(lock, sort_keys=True))

    first_checkpoint = repo / selected[_cell_names()[0]]["path"]
    os.utime(first_checkpoint, ns=(first_checkpoint.stat().st_atime_ns, first_checkpoint.stat().st_mtime_ns + 1))
    mtime_only = audit_locked_inputs(experiment, repo)
    assert mtime_only["status"] == "PASS"
    assert mtime_only["checkpoint_mtime_mismatches_informational"] == 1

    untracked = _write(old / "unexpected_public.txt", "drift\n")
    untracked_result = audit_locked_inputs(experiment, repo)
    assert untracked_result["status"] == "FAIL"
    assert any(
        row["finding"] == "no_nonignored_untracked_upstream_paths"
        for row in untracked_result["findings"]
    )
    untracked.unlink()

    first_feature = repo / references[_cell_names()[0]]["feature_path"]
    _write(first_feature, "drift", 0o600)
    failed = audit_locked_inputs(experiment, repo)
    assert failed["status"] == "FAIL"
    assert any(row["finding"] == "reference_feature_files_hash_mismatch" for row in failed["findings"])


def test_loose_or_combined_feature_probe_completion_fails_closed(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    experiment = repo / "additional_experiments/c1b_spatial_pooling_bottleneck_audit"
    plan = _write(experiment / "EXPERIMENT_PLAN.md", "frozen\n")
    config = _write(experiment / "configs/audit.json", "{}\n")
    selected = {}
    references = {}
    for index, cell in enumerate(_cell_names()):
        selected[cell] = {"sha256": f"{index + 1:064x}"}
        references[cell] = {
            "feature_sha256": f"{index + 101:064x}",
            "feature_metadata_sha256": f"{index + 201:064x}",
        }
    lock = {
        "selected_checkpoints": selected,
        "formal_p0_references": references,
        "plan_sha256": file_sha256(plan),
        "config_sha256": file_sha256(config),
    }
    lock_path = _write(experiment / "PREREGISTRATION_LOCK.json", json.dumps(lock, sort_keys=True))
    lock_sha = file_sha256(lock_path)

    feature_inventory = {}
    feature_bindings: dict[tuple[str, str], tuple[Path, Path]] = {}
    for seed in SEEDS:
        for arm in ARMS:
            for fold in FOLDS:
                cell = f"seed_{seed}/{arm}/fold_{fold}"
                for pooling in FEATURE_POOLINGS[arm]:
                    slug = POOLING_SLUGS[pooling]
                    asset = _write(
                        experiment / "features/final" / f"seed_{seed}" / arm / f"fold_{fold}" / f"{slug}.private.npz",
                        f"asset::{cell}::{pooling}",
                        0o600,
                    )
                    metadata = {
                        "status": "COMPLETE",
                        "stage": "final",
                        "seed_base": seed,
                        "arm": arm,
                        "fold": fold,
                        "pooling": pooling,
                        "feature_path": str(asset),
                        "feature_sha256": file_sha256(asset),
                        "checkpoint_sha256": selected[cell]["sha256"],
                        "reference_feature_sha256": references[cell]["feature_sha256"],
                        "reference_feature_metadata_sha256": references[cell]["feature_metadata_sha256"],
                        "preregistration_lock_sha256": lock_sha,
                        "plan_sha256": lock["plan_sha256"],
                        "config_sha256": lock["config_sha256"],
                        "training_performed": False,
                        "projector_called": False,
                        "transition_called": False,
                        "target_encoder_called": False,
                        "ftv_head_called": False,
                        "test_labels_used": False,
                    }
                    metadata_path = _write(
                        asset.with_suffix(".metadata.json"),
                        json.dumps(metadata, sort_keys=True),
                        0o600,
                    )
                    relative = metadata_path.relative_to(repo).as_posix()
                    feature_inventory[relative] = file_sha256(metadata_path)
                    feature_bindings[(cell, pooling)] = (asset, metadata_path)
    feature_completion = {
        "status": "COMPLETE",
        "stage": "final",
        "run_count": 40,
        "cell_count": 40,
        "expected_asset_count": 180,
        "preregistration_lock_sha256": lock_sha,
        "feature_metadata_sha256": feature_inventory,
    }
    _write(
        experiment / "features/feature_export_complete.private.json",
        json.dumps(feature_completion, sort_keys=True),
        0o600,
    )

    probe_cells = {}
    probe_feature_hashes = {}
    for seed in SEEDS:
        for arm in ARMS:
            for fold in FOLDS:
                cell = f"seed_{seed}/{arm}/fold_{fold}"
                for pooling in FORMAL_PROBE_POOLINGS:
                    if pooling in {
                        "PVALID",
                        "PORACLE",
                        "PLOCAL+PVALID_SECONDARY",
                    } and arm.startswith("L"):
                        continue
                    slug = POOLING_SLUGS[pooling]
                    output = experiment / "probes/final" / f"seed_{seed}" / arm / f"fold_{fold}" / slug
                    feature_asset, feature_metadata = feature_bindings[(cell, pooling)]
                    hashes = {}
                    for filename in (
                        "probe_metrics.csv",
                        "ridge_predictions.private.csv",
                        "ridge_selection.csv",
                    ):
                        hashes[filename] = file_sha256(
                            _write(output / filename, f"{cell},{pooling},{filename}\n", 0o600)
                        )
                    metadata = {
                        "seed_base": seed,
                        "arm": arm,
                        "fold": fold,
                        "pooling": pooling,
                        "patient_identifiers_private": True,
                        "test_used_for_scaler_or_selection": False,
                        "feature_sha256": file_sha256(feature_asset),
                        "feature_metadata_sha256": file_sha256(feature_metadata),
                        "output_sha256": hashes,
                    }
                    metadata_path = _write(
                        output / "probe_metadata.json",
                        json.dumps(metadata, sort_keys=True),
                        0o600,
                    )
                    key = f"{cell}/{slug}"
                    probe_cells[key] = {
                        "seed_base": seed,
                        "arm": arm,
                        "fold": fold,
                        "pooling": pooling,
                        "feature_path": str(feature_asset),
                        "feature_sha256": file_sha256(feature_asset),
                        "feature_metadata_path": str(feature_metadata),
                        "feature_metadata_sha256": file_sha256(feature_metadata),
                        "output_dir": str(output),
                        "probe_metadata_sha256": file_sha256(metadata_path),
                        "output_sha256": hashes,
                        "nuisance_included": pooling in {"P0", "PVALID", "PLOCAL"},
                    }
                    probe_feature_hashes[key] = file_sha256(feature_metadata)
    probe_completion = {
        "status": "COMPLETE",
        "stage": "final",
        "new_training_performed": False,
        "requested_poolings": list(FORMAL_PROBE_POOLINGS),
        "executed_cell_count": 180,
        "expected_cell_count": 180,
        "preregistration_lock_sha256": lock_sha,
        "feature_metadata_sha256": probe_feature_hashes,
        "cells": probe_cells,
    }
    _write(
        experiment / "probes/final/probe_matrix_all_complete.private.json",
        json.dumps(probe_completion, sort_keys=True),
        0o600,
    )

    result = audit_final_inventories(experiment, repo)
    assert result["status"] == "FAIL"
    categories = {row["finding"] for row in result["findings"]}
    assert "probe_completion_inventory_not_exact_two" in categories
    assert "feature_control_schema_or_live_binding_invalid" in categories

    arbitrary_probe = next(iter(probe_cells.values()))
    _write(Path(arbitrary_probe["output_dir"]) / "probe_metrics.csv", "drift\n", 0o600)
    assert audit_final_inventories(experiment, repo)["status"] == "FAIL"


def load_tests(
    _loader: unittest.TestLoader,
    _tests: unittest.TestSuite,
    _pattern: str | None,
) -> unittest.TestSuite:
    """Adapt the synthetic path-based cases to the repository's unittest suite."""

    suite = unittest.TestSuite()
    for name, function in sorted(globals().items()):
        if not name.startswith("test_") or not callable(function):
            continue

        def run_case(function=function) -> None:
            with tempfile.TemporaryDirectory() as directory:
                function(Path(directory))

        suite.addTest(unittest.FunctionTestCase(run_case, description=name))
    return suite


if __name__ == "__main__":
    unittest.main()

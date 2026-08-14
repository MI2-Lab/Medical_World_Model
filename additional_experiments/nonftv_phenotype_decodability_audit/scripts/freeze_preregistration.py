#!/usr/bin/env python3
"""Create and verify the immutable non-FTV audit preregistration lock.

The lock must be created after the complete analysis implementation exists and
before any retained formal output exists.  Creation is fail-closed: it checks
the Git parent, authenticates the frozen clinical sources, validates all 20
feature/checkpoint cells, and writes with atomic no-replace semantics.  The
result contains hashes and aggregate identities only; it never serializes a
patient or trial identifier.

This module intentionally exposes :func:`require_preregistration_lock` so the
formal runner and aggregate-only postprocessor can verify their code and input
contract immediately before use.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping

from audit_core import (
    AUDIT_ROOT,
    REPO_ROOT,
    authenticate,
    canonical_sha256,
    file_sha256,
    load_config,
    load_feature_cell,
    load_fold_splits,
    load_json,
    load_targets,
    resolve_path,
)


LOCK_PATH = AUDIT_ROOT / "PREREGISTRATION_LOCK.json"
MANDATORY_SCRIPTS = {
    "analyze_results.py",
    "audit_core.py",
    "freeze_preregistration.py",
    "run_audit.py",
    "validate_audit.py",
}
OUTPUT_DIRECTORIES = (
    "features",
    "predictions",
    "metrics",
    "figures",
    "logs",
    "manifests",
    "reports",
)


def _git(*arguments: str, cwd: Path = REPO_ROOT) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=cwd, text=True, stderr=subprocess.STDOUT
    ).strip()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _file_binding(path: str | Path, *, role: str) -> dict[str, str]:
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"missing preregistration binding ({role}): {resolved}")
    return {
        "role": role,
        "path": _display_path(resolved),
        "sha256": file_sha256(resolved),
    }


def _binding_path(binding: Mapping[str, Any]) -> Path:
    raw = Path(str(binding["path"]))
    return raw.resolve() if raw.is_absolute() else (REPO_ROOT / raw).resolve()


def _assert_no_retained_formal_outputs() -> None:
    """Refuse to lock after any experiment output has been retained.

    The scan is deliberately broader than the runner's two sentinel files.  A
    partial/aborted run that happened to miss those sentinels must not be
    relabelled as prospectively preregistered.  Only repository placeholders
    are allowed in output directories before the lock is created.
    """

    retained: list[str] = []
    for relative_directory in OUTPUT_DIRECTORIES:
        directory = AUDIT_ROOT / relative_directory
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            if (path.is_file() or path.is_symlink()) and path.name != ".gitkeep":
                retained.append(str(path.relative_to(AUDIT_ROOT)))
    if retained:
        raise FileExistsError(
            "refusing preregistration after retained formal/partial outputs: "
            + ", ".join(retained)
        )


def _atomic_json_no_replace(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish a complete read-only JSON file atomically without replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable lock: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        # Hard-link publication is atomic and, unlike Path.replace(), fails if
        # another process created the destination after our initial check.
        os.link(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _authenticate_git(config: Mapping[str, Any], *, exact_parent: bool) -> dict[str, Any]:
    branch = _git("branch", "--show-current")
    head = _git("rev-parse", "HEAD")
    expected_branch = str(config["branch"])
    expected_parent = str(config["parent_sha"])
    if branch != expected_branch:
        raise ValueError(
            f"preregistration branch drift: expected {expected_branch}, observed {branch}"
        )
    if exact_parent:
        if head != expected_parent:
            raise ValueError(
                f"preregistration parent drift: expected {expected_parent}, observed {head}"
            )
    else:
        status = subprocess.run(
            ["git", "merge-base", "--is-ancestor", expected_parent, head],
            cwd=REPO_ROOT,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        if status != 0:
            raise ValueError(f"locked parent {expected_parent} is not an ancestor of {head}")

    source_commits: dict[str, str] = {}
    for label, commit in sorted(config["source_commits"].items()):
        commit = str(commit)
        resolved = _git("rev-parse", f"{commit}^{{commit}}")
        if resolved != commit:
            raise ValueError(f"source commit identity drift for {label}: {resolved}")
        source_commits[str(label)] = commit
    return {
        "branch": branch,
        "head_at_verification": head,
        "parent_sha": expected_parent,
        "source_commits": source_commits,
    }


def _verify_source_worktree_heads(config: Mapping[str, Any]) -> dict[str, str]:
    goal6_root = resolve_path(config["paths"]["goal6_root"])
    spatial_root = resolve_path(config["paths"]["spatial_feature_root"])
    goal6_worktree = goal6_root.parents[1]
    spatial_worktree = spatial_root.parents[2]
    observed = {
        "goal6_classical_dce": _git("rev-parse", "HEAD", cwd=goal6_worktree),
        "spatial_heterogeneity": _git("rev-parse", "HEAD", cwd=spatial_worktree),
    }
    for label, head in observed.items():
        expected = str(config["source_commits"][label])
        if head != expected:
            raise ValueError(
                f"source worktree HEAD drift for {label}: expected {expected}, observed {head}"
            )
    return observed


def _source_bindings(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    paths = config["paths"]
    definitions = {
        "workbook": (
            paths["workbook"],
            paths["workbook_sha256"],
            {"sheet": str(paths["workbook_sheet"])},
        ),
        "eligible_target_table": (
            paths["eligible_target_table"],
            paths["eligible_target_table_sha256"],
            {},
        ),
        "fold_manifest": (
            paths["fold_manifest"],
            paths["fold_manifest_sha256"],
            {},
        ),
        "goal6_final_report": (
            str(paths["goal6_root"]) + "/reports/final_report.md",
            paths["goal6_final_report_sha256"],
            {},
        ),
        "spatial_completion": (
            str(paths["spatial_feature_root"])
            + "/feature_matrix_complete.private.json",
            paths["spatial_completion_sha256"],
            {},
        ),
    }
    output: dict[str, dict[str, Any]] = {}
    for role, (path, expected, extra) in definitions.items():
        observed = authenticate(path, str(expected), role.replace("_", " "))
        binding: dict[str, Any] = _file_binding(path, role=role)
        if observed != binding["sha256"]:
            raise AssertionError(f"non-deterministic hash observation for {role}")
        binding["expected_sha256"] = str(expected)
        binding.update(extra)
        output[role] = binding
    return output


def _code_bindings() -> list[dict[str, str]]:
    scripts = sorted((AUDIT_ROOT / "scripts").glob("*.py"))
    observed_names = {path.name for path in scripts}
    missing = sorted(MANDATORY_SCRIPTS - observed_names)
    if missing:
        raise FileNotFoundError(
            "analysis implementation is not finalized; missing scripts: "
            + ", ".join(missing)
        )
    bindings = [
        _file_binding(AUDIT_ROOT / "EXPERIMENT_PLAN.md", role="experiment_plan"),
        _file_binding(AUDIT_ROOT / "configs" / "audit.json", role="audit_config"),
    ]
    bindings.extend(_file_binding(path, role=f"script:{path.name}") for path in scripts)
    bindings.extend(
        _file_binding(path, role=f"test:{path.relative_to(AUDIT_ROOT / 'tests')}")
        for path in sorted((AUDIT_ROOT / "tests").rglob("*.py"))
    )
    return bindings


def _cell_paths(config: Mapping[str, Any], seed: int, arm: str, fold: int) -> dict[str, Path]:
    relative = Path(f"seed_{seed}") / arm / f"fold_{fold}"
    local_feature = resolve_path(config["paths"]["local_feature_root"]) / relative / "response_state.private.npz"
    checkpoint = resolve_path(config["paths"]["checkpoint_root"]) / relative / str(
        config["frozen"]["checkpoint_filename"]
    )
    spatial_feature = resolve_path(config["paths"]["spatial_feature_root"]) / relative / "spatial_statistics.private.npz"
    return {
        "local_feature": local_feature,
        "local_metadata": local_feature.with_suffix(".metadata.json"),
        "checkpoint": checkpoint,
        "selection": checkpoint.with_name("selection.json"),
        "spatial_feature": spatial_feature,
        "spatial_metadata": spatial_feature.with_suffix(".metadata.json"),
    }


def _cell_bindings(
    config: Mapping[str, Any], targets: Any, fold_splits: Mapping[int, Any]
) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    expected_identities = {
        (int(seed), str(arm), int(fold))
        for seed in config["frozen"]["seeds"]
        for arm in config["frozen"]["arms"]
        for fold in config["frozen"]["folds"]
    }
    for seed, arm, fold in sorted(expected_identities):
        # This validates tensor schemas/alignment, checkpoint identity and
        # projector parity without fitting or evaluating any phenotype probe.
        cell = load_feature_cell(
            config,
            targets,
            fold_splits,
            seed=seed,
            arm=arm,
            fold=fold,
        )
        paths = _cell_paths(config, seed, arm, fold)
        local_metadata = load_json(paths["local_metadata"])
        spatial_metadata = load_json(paths["spatial_metadata"])
        selection = load_json(paths["selection"])
        identity = (cell.seed, cell.arm, cell.fold)
        if identity != (seed, arm, fold):
            raise ValueError(f"loaded cell identity drift: {identity}")
        if any(
            (int(payload.get("seed_base", -1)), str(payload.get("arm", "")), int(payload.get("fold", -1)))
            != (seed, arm, fold)
            for payload in (local_metadata, spatial_metadata, selection)
        ):
            raise ValueError(f"metadata/selection identity drift for {seed}/{arm}/{fold}")
        if int(selection.get("selected_epoch", -1)) != cell.selected_epoch:
            raise ValueError(f"selection epoch drift for {seed}/{arm}/{fold}")
        if selection.get("test_data_used") is not False or selection.get("pcr_used") is not False:
            raise ValueError(f"selection firewall failed for {seed}/{arm}/{fold}")

        files = {
            role: _file_binding(path, role=role)
            for role, path in sorted(paths.items())
        }
        provenance = dict(cell.provenance)
        expected_hashes = {
            "local_feature": provenance["local_feature_sha256"],
            "local_metadata": provenance["local_metadata_sha256"],
            "checkpoint": provenance["checkpoint_sha256"],
            "selection": provenance["selection_sha256"],
            "spatial_feature": provenance["spatial_feature_sha256"],
            "spatial_metadata": provenance["spatial_metadata_sha256"],
        }
        for role, expected_hash in expected_hashes.items():
            if files[role]["sha256"] != expected_hash:
                raise ValueError(f"cell provenance hash mismatch for {seed}/{arm}/{fold}/{role}")
        cells.append(
            {
                "cell": f"seed_{seed}/{arm}/fold_{fold}",
                "seed": seed,
                "arm": arm,
                "fold": fold,
                "selected_epoch": int(cell.selected_epoch),
                "patient_order_sha256": str(provenance["patient_order_sha256"]),
                "full_feature_patient_count": int(config["frozen"]["full_feature_patient_count"]),
                "local_feature_shape": list(local_metadata.get("feature_shape", [])),
                "spatial_statistic_shapes": spatial_metadata.get("statistic_shapes", {}),
                "oracle_mean_shape": spatial_metadata.get("oracle_mean_shape", []),
                "selection_mode": str(selection.get("selection_mode", "")),
                "selection_experiment_pass": bool(selection.get("experiment_pass")),
                "selected": bool(provenance["selected"]),
                "test_data_used": bool(provenance["test_data_used"]),
                "pcr_used": bool(provenance["pcr_used"]),
                "delta_ftv_used": bool(provenance["delta_ftv_used"]),
                "encoder_frozen": bool(provenance["encoder_frozen"]),
                "encoder_training_performed": bool(provenance["training_performed"]),
                "z1_derived_from_z2_without_encoder": bool(
                    provenance["z1_derived_from_z2_without_encoder"]
                ),
                "z3_to_z2_projection_parity_max_abs": float(
                    provenance["z3_to_z2_projection_parity_max_abs"]
                ),
                "files": files,
            }
        )
    observed_identities = {
        (int(row["seed"]), str(row["arm"]), int(row["fold"])) for row in cells
    }
    if len(cells) != 20 or observed_identities != expected_identities:
        raise ValueError("feature-cell Cartesian product is not the exact frozen 20 cells")
    return cells


def _runtime_environment() -> dict[str, Any]:
    distributions = {
        "matplotlib": "matplotlib",
        "numpy": "numpy",
        "openpyxl": "openpyxl",
        "pandas": "pandas",
        "scikit_learn": "scikit-learn",
        "scipy": "scipy",
        "torch": "torch",
    }
    versions: dict[str, str | None] = {}
    for key, distribution in distributions.items():
        try:
            versions[key] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            versions[key] = None
    torch_runtime: dict[str, Any]
    try:
        import torch

        torch_runtime = {
            "version": str(torch.__version__),
            "cuda_build": str(torch.version.cuda) if torch.version.cuda is not None else None,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
        }
    except Exception as error:  # pragma: no cover - environment inventory only
        torch_runtime = {"import_error": f"{type(error).__name__}: {error}"}
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "git_version": _git("--version"),
        "package_versions": versions,
        "torch_runtime": torch_runtime,
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "PYTHONHASHSEED",
            )
        },
    }


def _iter_file_bindings(lock: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    yield from lock["analysis_contract"]["files"]
    yield from lock["source_contract"].values()
    for cell in lock["feature_cells"]:
        yield from cell["files"].values()


def verify_preregistration_lock(
    lock_path: str | Path = LOCK_PATH,
    *,
    require_exact_parent: bool = True,
) -> dict[str, Any]:
    """Verify every file hash and identity bound by an existing lock."""

    resolved_lock = Path(lock_path).resolve()
    if not resolved_lock.is_file():
        raise FileNotFoundError(f"missing preregistration lock: {resolved_lock}")
    lock = load_json(resolved_lock)
    if lock.get("schema_version") != 1 or lock.get("status") != "LOCKED_BEFORE_FORMAL_RUN":
        raise ValueError("invalid preregistration lock schema/status")
    if resolved_lock.stat().st_mode & 0o222:
        raise PermissionError("preregistration lock is writable; immutable mode required")

    config_binding = next(
        binding
        for binding in lock["analysis_contract"]["files"]
        if binding.get("role") == "audit_config"
    )
    config = load_config(_binding_path(config_binding))
    git_contract = _authenticate_git(config, exact_parent=require_exact_parent)
    if git_contract["branch"] != lock["git_contract"]["branch"]:
        raise ValueError("current branch differs from preregistration lock")
    if str(config["parent_sha"]) != lock["git_contract"]["parent_sha"]:
        raise ValueError("config parent differs from preregistration lock")
    source_heads = _verify_source_worktree_heads(config)
    if source_heads != lock["git_contract"]["source_worktree_heads"]:
        raise ValueError("source worktree identities differ from preregistration lock")
    if canonical_sha256(config) != lock["analysis_contract"]["config_canonical_sha256"]:
        raise ValueError("canonical config contract differs from preregistration lock")

    seen: set[tuple[str, str]] = set()
    for binding in _iter_file_bindings(lock):
        role = str(binding["role"])
        path = _binding_path(binding)
        key = (role, str(path))
        if key in seen:
            raise ValueError(f"duplicate preregistration binding: {role}/{path}")
        seen.add(key)
        if not path.is_file():
            raise FileNotFoundError(f"locked input missing: {role}: {path}")
        observed = file_sha256(path)
        if observed != str(binding["sha256"]):
            raise ValueError(
                f"preregistration drift for {role}: expected {binding['sha256']}, observed {observed}"
            )
    identities = {
        (int(cell["seed"]), str(cell["arm"]), int(cell["fold"]))
        for cell in lock["feature_cells"]
    }
    expected = {
        (int(seed), str(arm), int(fold))
        for seed in config["frozen"]["seeds"]
        for arm in config["frozen"]["arms"]
        for fold in config["frozen"]["folds"]
    }
    if len(lock["feature_cells"]) != 20 or identities != expected:
        raise ValueError("locked feature-cell identities are not the exact 20-cell matrix")
    if lock["diagnostic_disclosure"].get("thresholds_changed_based_on_exposed_values") is not False:
        raise ValueError("diagnostic-threshold disclosure is absent or invalid")
    return {
        "status": "PASS",
        "lock_path": _display_path(resolved_lock),
        "lock_sha256": file_sha256(resolved_lock),
        "binding_count": len(seen),
        "feature_cell_count": len(lock["feature_cells"]),
        "branch": git_contract["branch"],
        "head": git_contract["head_at_verification"],
    }


def require_preregistration_lock(*, require_exact_parent: bool = True) -> dict[str, Any]:
    """Fail closed unless the experiment's immutable lock verifies in full."""

    return verify_preregistration_lock(
        LOCK_PATH,
        require_exact_parent=require_exact_parent,
    )


def create_preregistration_lock(config_path: str | Path) -> dict[str, Any]:
    if LOCK_PATH.exists():
        raise FileExistsError(f"refusing to overwrite immutable lock: {LOCK_PATH}")
    _assert_no_retained_formal_outputs()
    config = load_config(config_path)
    git_contract = _authenticate_git(config, exact_parent=True)
    source_worktree_heads = _verify_source_worktree_heads(config)
    sources = _source_bindings(config)
    code = _code_bindings()

    # Re-run the target/fold preflight and all cell-level alignment checks.  No
    # target-model fit, prediction, or metric computation occurs here.
    targets = load_targets(config)
    fold_splits = load_fold_splits(config, targets)
    cells = _cell_bindings(config, targets, fold_splits)
    if len(targets.patient_ids) != int(config["frozen"]["patient_count"]):
        raise ValueError("preflight patient count differs from frozen config")

    payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment": str(config["experiment"]),
        "status": "LOCKED_BEFORE_FORMAL_RUN",
        "immutability": {
            "creation_semantics": "atomic_hard_link_no_replace",
            "file_mode": "0444",
            "overwrite_refused": True,
            "formal_outputs_present_at_lock": False,
        },
        "git_contract": {
            **git_contract,
            "source_worktree_heads": source_worktree_heads,
        },
        "analysis_contract": {
            "config_canonical_sha256": canonical_sha256(config),
            "files": code,
            "all_current_scripts_bound": True,
            "all_current_python_tests_bound": True,
        },
        "source_contract": sources,
        "preflight": {
            "status": "PASS",
            "patient_count": len(targets.patient_ids),
            "patient_set_sha256": targets.patient_set_sha256,
            "fold_count": len(fold_splits),
            "workbook_max_abs_difference": targets.workbook_max_abs_difference,
            "pcr_parsed": False,
            "feature_cell_count": len(cells),
        },
        "feature_cells": cells,
        "diagnostic_disclosure": {
            "pristine_before_any_target_result_inspection": False,
            "preflight": "PASS; no phenotype probe metric was produced by preflight",
            "one_cell_smoke": {
                "performed": True,
                "scope": "first enumerated cell: seed=2026, arm=LOCAL0, fold=0",
                "retained_artifacts": False,
                "diagnostic_metrics_exposed": True,
            },
            "aborted_first_cell_formal_attempt": {
                "performed": True,
                "scope": "first enumerated cell: seed=2026, arm=LOCAL0, fold=0",
                "completed_formal_matrix": False,
                "retained_artifacts": False,
                "diagnostic_metrics_exposed": True,
            },
            "metric_values_recorded_in_lock": False,
            "thresholds_changed_based_on_exposed_values": False,
            "prelock_numeric_parity_tolerance_amendment": {
                "performed": True,
                "target_or_probe_metric_driven": False,
                "trigger": "strict CPU recomputation atol=1e-6 failed for two of 20 GPU-exported float32 cells near zero",
                "observed_global_max_abs_range": [
                    2.0265579223632812e-6,
                    2.6226043701171875e-6
                ],
                "frozen_rtol": 1e-5,
                "frozen_atol": 5e-6,
                "frozen_global_max_abs_lte": 5e-6,
                "retained_formal_outputs_before_amendment": False
            },
            "declaration": (
                "This is a prospective lock after limited diagnostic metric exposure, not a "
                "claim of pristine preregistration. No diagnostic artifact was retained and no "
                "gate threshold, target, representation, timing, fold, or analysis rule was "
                "changed based on the exposed values."
            ),
        },
        "privacy_and_training_firewall": {
            "patient_or_trial_identifiers_serialized": False,
            "pcr_parsed": False,
            "pcr_used_for_any_selection": False,
            "encoder_retrained": False,
            "test_used_for_checkpoint_or_probe_alpha_selection": False,
        },
        "runtime_environment": _runtime_environment(),
    }
    _atomic_json_no_replace(LOCK_PATH, payload)
    return verify_preregistration_lock(LOCK_PATH, require_exact_parent=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=AUDIT_ROOT / "configs" / "audit.json",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify the existing immutable lock instead of creating it",
    )
    parser.add_argument(
        "--allow-descendant-head",
        action="store_true",
        help="during verification, allow HEAD to be a descendant of the locked parent",
    )
    arguments = parser.parse_args()
    if arguments.verify_only:
        result = verify_preregistration_lock(
            LOCK_PATH,
            require_exact_parent=not arguments.allow_descendant_head,
        )
    else:
        if arguments.allow_descendant_head:
            parser.error("--allow-descendant-head is valid only with --verify-only")
        result = create_preregistration_lock(arguments.config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

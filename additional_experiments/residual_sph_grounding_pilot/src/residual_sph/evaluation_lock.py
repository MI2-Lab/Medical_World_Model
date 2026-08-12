"""Immutable boundary between pCR-free representation work and pCR evaluation."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .contracts import file_sha256


EXPERIMENT = "residual_sph_grounding_pilot"
SEEDS = (2026, 3026)
FOLDS = tuple(range(5))
ARMS = ("S0", "S1", "S2", "S2_L10")
PROBE_OUTPUT_NAMES = (
    "ridge_selection.private.csv",
    "ridge_predictions.private.csv",
    "probe_metadata.private.json",
)
REPRESENTATION_AGGREGATES = (
    "manifests/residualizer_inventory.json",
    "metrics/residualizer_fits.csv",
    "metrics/representation_metrics.csv",
    "metrics/table_static_ftv.csv",
    "metrics/table_observed_delta_ftv.csv",
    "metrics/table_sph_and_residual.csv",
    "metrics/table_partial_correlations.csv",
    "metrics/table_state_redundancy.csv",
    "metrics/table_seed_consistency.csv",
    "metrics/optimization_safety.csv",
    "metrics/optimization_trajectories.csv",
    "metrics/representation_effects.json",
)
PROBE_SPECIFICATION_FILES = (
    "configs/pilot.json",
    "scripts/run_probes.py",
    "scripts/aggregate_representation.py",
    "src/residual_sph/contracts.py",
    "src/residual_sph/evaluation.py",
    "src/residual_sph/probes.py",
    "src/residual_sph/targets.py",
)
REPRESENTATION_ENTRYPOINTS = (
    "scripts/run_representation_pipeline.py",
    "scripts/audit_pcr_firewall.py",
    "scripts/audit_s0_reference.py",
    "scripts/build_residualizers.py",
    "scripts/train_cell.py",
    "scripts/export_features.py",
    "scripts/run_probes.py",
    "scripts/aggregate_representation.py",
    "scripts/freeze_representation.py",
    "scripts/record_resource_guard.py",
    "scripts/run_matrix.py",
)
POSTFREEZE_LOCAL_MODULES = frozenset({"residual_sph.pcr_evaluation"})
FIREWALL_AUDIT_RELATIVE = "manifests/pcr_firewall_audit.json"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
PRIVATE_TABLE_LITERAL_RE = re.compile(
    r"(?:clinical|pcr)[^\n\r\"']{0,100}\.(?:csv|tsv|xls|xlsx)",
    flags=re.IGNORECASE,
)
# Split the digest so the audit implementation itself does not contain the
# forbidden clinical-input fingerprint as one source literal.
CLINICAL_INPUT_SHA256 = (
    "b3355f8ac80cf8f0fa95722b8d8a8b73"
    "d96790e9ded5c491ddb5b2e6a7793436"
)


def _repository_layout(root: Path) -> tuple[Path, str]:
    repo_root = root.parents[1]
    try:
        prefix = root.relative_to(repo_root).as_posix()
    except ValueError as error:  # pragma: no cover - defensive for exotic callers
        raise PermissionError("experiment root escaped repository layout") from error
    return repo_root, prefix


def _prefixed(prefix: str, relative: str) -> str:
    return f"{prefix}/{relative}"


def expected_representation_artifact_groups(
    experiment_root: str | Path,
) -> dict[str, tuple[str, ...]]:
    """Return the exact repo-relative artifact paths authorized for a freeze."""

    root = Path(experiment_root).resolve()
    _, prefix = _repository_layout(root)
    confirmation = "additional_experiments/local_response_state_multiseed_confirmation"
    selections: list[str] = []
    checkpoints: list[str] = []
    ftv_transforms: list[str] = []
    feature_assets: list[str] = []
    feature_metadata: list[str] = []
    probe_outputs: list[str] = []
    for arm in ARMS:
        for seed in SEEDS:
            for fold in FOLDS:
                if arm == "S0":
                    run = f"{confirmation}/checkpoints/formal_4x8/seed_{seed}/LOCAL3/fold_{fold}"
                    feature = (
                        f"{confirmation}/features/formal_4x8/seed_{seed}/LOCAL3/"
                        f"fold_{fold}/response_state.private.npz"
                    )
                else:
                    run = f"{prefix}/checkpoints/formal_4x8/seed_{seed}/{arm}/fold_{fold}"
                    feature = (
                        f"{prefix}/features/formal_4x8/seed_{seed}/{arm}/"
                        f"fold_{fold}/response_state.private.npz"
                    )
                selections.append(f"{run}/selection.json")
                checkpoints.append(f"{run}/selected.pt")
                ftv_transforms.append(f"{run}/ftv_transform.json")
                feature_assets.append(feature)
                feature_metadata.append(feature.removesuffix(".npz") + ".metadata.json")
                probe = (
                    f"{prefix}/predictions/formal_4x8/seed_{seed}/{arm}/fold_{fold}"
                )
                probe_outputs.extend(f"{probe}/{name}" for name in PROBE_OUTPUT_NAMES)

    groups = {
        "implementation_lock": (
            _prefixed(prefix, "manifests/implementation_lock.json"),
        ),
        "s0_provenance": (
            _prefixed(prefix, "manifests/s0_confirmation_provenance.json"),
        ),
        "selection_records": tuple(sorted(selections)),
        "selected_checkpoints": tuple(sorted(checkpoints)),
        "ftv_transforms": tuple(sorted(ftv_transforms)),
        "feature_assets": tuple(sorted(feature_assets)),
        "feature_metadata": tuple(sorted(feature_metadata)),
        "probe_outputs": tuple(sorted(probe_outputs)),
        "residualizer_transforms": tuple(
            _prefixed(prefix, f"manifests/residualizers/fold_{fold}.json")
            for fold in FOLDS
        ),
        "representation_aggregates": tuple(
            _prefixed(prefix, relative) for relative in REPRESENTATION_AGGREGATES
        ),
        "probe_specification": tuple(
            _prefixed(prefix, relative) for relative in PROBE_SPECIFICATION_FILES
        ),
        "pcr_firewall_audit": (
            _prefixed(prefix, FIREWALL_AUDIT_RELATIVE),
        ),
    }
    flattened = [path for paths in groups.values() for path in paths]
    if len(flattened) != len(set(flattened)):
        raise AssertionError("representation freeze groups contain duplicate paths")
    return groups


def _local_imports(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError) as error:
        raise PermissionError(f"cannot audit representation source: {path.name}") from error
    imports: set[str] = set()
    in_package = path.parent.name == "residual_sph"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(
                alias.name
                for alias in node.names
                if alias.name == "residual_sph" or alias.name.startswith("residual_sph.")
            )
        elif isinstance(node, ast.ImportFrom):
            if node.level and in_package:
                if node.module:
                    imports.add(f"residual_sph.{node.module}")
                else:
                    imports.update(f"residual_sph.{alias.name}" for alias in node.names)
            elif node.module and (
                node.module == "residual_sph" or node.module.startswith("residual_sph.")
            ):
                imports.add(node.module)
        elif isinstance(node, ast.Call) and node.args:
            dynamic_import = (
                isinstance(node.func, ast.Name)
                and node.func.id in {"__import__", "import_module"}
            ) or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "import_module"
            )
            first = node.args[0]
            if (
                dynamic_import
                and isinstance(first, ast.Constant)
                and isinstance(first.value, str)
                and (
                    first.value == "residual_sph"
                    or first.value.startswith("residual_sph.")
                )
            ):
                imports.add(first.value)
    return imports


def _cli_arguments(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        values.extend(
            argument.value
            for argument in node.args
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
        )
    return tuple(values)


def _representation_source_closure(root: Path) -> tuple[tuple[Path, ...], set[str]]:
    entrypoints = tuple(root / relative for relative in REPRESENTATION_ENTRYPOINTS)
    package = root / "src" / "residual_sph"
    pending = list(entrypoints)
    sources = set(entrypoints)
    package_init = package / "__init__.py"
    sources.add(package_init)
    local_modules: set[str] = set()
    while pending:
        source = pending.pop()
        if not source.is_file():
            raise PermissionError(f"firewall audit source is missing: {source.name}")
        for module in _local_imports(source):
            if module == "residual_sph":
                continue
            local_modules.add(module)
            suffix = module.removeprefix("residual_sph.").replace(".", "/")
            module_path = package / f"{suffix}.py"
            if not module_path.is_file():
                raise PermissionError(f"unresolved representation-phase module: {module}")
            if module_path not in sources:
                sources.add(module_path)
                pending.append(module_path)
    return tuple(sorted(sources)), local_modules


def build_pcr_firewall_audit(
    experiment_root: str | Path,
    *,
    preregistration_lock_sha256: str,
    implementation_lock_sha256: str,
) -> dict[str, Any]:
    """Build a deterministic static firewall attestation without clinical I/O."""

    root = Path(experiment_root).resolve()
    repo_root, _ = _repository_layout(root)
    sources, local_modules = _representation_source_closure(root)
    findings: list[dict[str, str]] = []
    imported_postfreeze = sorted(POSTFREEZE_LOCAL_MODULES.intersection(local_modules))
    for module in imported_postfreeze:
        findings.append(
            {
                "artifact": module,
                "rule": "postfreeze_module_imported_by_representation_phase",
            }
        )
    for relative in REPRESENTATION_ENTRYPOINTS:
        path = root / relative
        for argument in _cli_arguments(path):
            lowered = argument.lower()
            if "clinical" in lowered or "pcr" in lowered:
                findings.append(
                    {
                        "artifact": path.relative_to(repo_root).as_posix(),
                        "rule": "clinical_or_pcr_cli_argument",
                        "value": argument,
                    }
                )
    for path in sources:
        source = path.read_text(encoding="utf-8")
        token = path.relative_to(repo_root).as_posix()
        if CLINICAL_INPUT_SHA256 in source:
            findings.append({"artifact": token, "rule": "clinical_input_hash_literal"})
        if PRIVATE_TABLE_LITERAL_RE.search(source):
            findings.append({"artifact": token, "rule": "clinical_input_path_literal"})

    audited = {
        path.relative_to(repo_root).as_posix(): file_sha256(path) for path in sources
    }
    checks = {
        "local_import_closure_complete": True,
        "postfreeze_module_import_absent": not imported_postfreeze,
        "clinical_or_pcr_cli_argument_absent": not any(
            row["rule"] == "clinical_or_pcr_cli_argument" for row in findings
        ),
        "clinical_input_hash_literal_absent": not any(
            row["rule"] == "clinical_input_hash_literal" for row in findings
        ),
        "clinical_input_path_literal_absent": not any(
            row["rule"] == "clinical_input_path_literal" for row in findings
        ),
    }
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": "PASS" if not findings and all(checks.values()) else "FAIL",
        "audit": "STATIC_PCR_FIREWALL_BEFORE_REPRESENTATION_FREEZE",
        "preregistration_lock_sha256": str(preregistration_lock_sha256),
        "implementation_lock_sha256": str(implementation_lock_sha256),
        "representation_phase_can_read_pcr": False,
        "representation_phase_can_read_clinical_or_treatment_fields": False,
        "postfreeze_modules_excluded": sorted(POSTFREEZE_LOCAL_MODULES),
        "audited_entrypoints": [
            (root / relative).relative_to(repo_root).as_posix()
            for relative in REPRESENTATION_ENTRYPOINTS
        ],
        "audited_local_modules": sorted(local_modules),
        "audited_source_sha256": dict(sorted(audited.items())),
        "checks": checks,
        "findings": findings,
    }


def verify_pcr_firewall_audit(
    experiment_root: str | Path,
    *,
    expected_preregistration_sha256: str,
    expected_implementation_sha256: str,
) -> dict[str, Any]:
    root = Path(experiment_root).resolve()
    path = root / FIREWALL_AUDIT_RELATIVE
    try:
        observed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PermissionError("pCR firewall audit is missing or invalid") from error
    expected = build_pcr_firewall_audit(
        root,
        preregistration_lock_sha256=expected_preregistration_sha256,
        implementation_lock_sha256=expected_implementation_sha256,
    )
    if expected["status"] != "PASS":
        raise PermissionError("current representation source fails the pCR firewall audit")
    if observed != expected:
        raise PermissionError("pCR firewall audit is stale or has an invalid shape")
    return {**expected, "audit_sha256": file_sha256(path)}


def _exact_artifact_groups(
    payload: Mapping[str, Any], root: Path
) -> dict[str, tuple[str, ...]]:
    expected = expected_representation_artifact_groups(root)
    observed = payload.get("artifact_groups")
    if not isinstance(observed, Mapping):
        raise PermissionError("representation freeze has no exact artifact groups")
    normalized: dict[str, tuple[str, ...]] = {}
    for group, values in observed.items():
        if not isinstance(group, str) or not isinstance(values, Sequence) or isinstance(
            values, (str, bytes)
        ):
            raise PermissionError("representation freeze artifact groups are invalid")
        normalized[group] = tuple(str(value) for value in values)
    if normalized != expected:
        raise PermissionError("representation freeze artifact inventory shape drifted")
    return expected


def verify_representation_freeze(
    experiment_root: str | Path,
    *,
    expected_preregistration_sha256: str,
) -> dict[str, Any]:
    root = Path(experiment_root).resolve()
    path = root / "manifests" / "representation_freeze.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PermissionError("pCR access is forbidden before representation freeze") from error
    implementation_path = root / "manifests" / "implementation_lock.json"
    if not implementation_path.is_file():
        raise PermissionError("representation freeze implementation lock is missing")
    implementation_sha256 = file_sha256(implementation_path)
    expected = {
        "schema_version": 2,
        "experiment": EXPERIMENT,
        "status": "REPRESENTATION_FROZEN_PCR_EVALUATION_AUTHORIZED",
        "preregistration_lock_sha256": str(expected_preregistration_sha256),
        "implementation_lock_sha256": implementation_sha256,
        "pcr_or_clinical_read_before_freeze": False,
        "selected_checkpoint_count": 40,
        "selection_record_count": 40,
        "ftv_transform_count": 40,
        "feature_asset_count": 40,
        "feature_metadata_count": 40,
        "probe_cell_count": 40,
        "probe_artifact_count": 120,
        "residualizer_fold_count": 5,
        "representation_aggregate_count": len(REPRESENTATION_AGGREGATES),
        "probe_specification_file_count": len(PROBE_SPECIFICATION_FILES),
        "pcr_firewall_audit_status": "PASS",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise PermissionError(f"representation freeze differs at {key}")
    groups = _exact_artifact_groups(payload, root)
    expected_paths = {relative for paths in groups.values() for relative in paths}
    inventory = payload.get("artifact_sha256")
    if not isinstance(inventory, Mapping) or set(map(str, inventory)) != expected_paths:
        raise PermissionError("representation freeze artifact hashes have an invalid shape")
    repo_root, _ = _repository_layout(root)
    for relative in sorted(expected_paths):
        expected_hash = str(inventory[relative]).lower()
        if SHA256_RE.fullmatch(expected_hash) is None:
            raise PermissionError(f"representation freeze has an invalid hash: {relative}")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise PermissionError("representation freeze contains unsafe paths")
        artifact = (repo_root / relative_path).resolve()
        try:
            artifact.relative_to(repo_root)
        except ValueError as error:
            raise PermissionError("representation artifact escaped repository") from error
        if not artifact.is_file() or file_sha256(artifact) != expected_hash:
            raise PermissionError(f"representation artifact hash drifted: {relative}")
    verify_pcr_firewall_audit(
        root,
        expected_preregistration_sha256=str(expected_preregistration_sha256),
        expected_implementation_sha256=implementation_sha256,
    )
    return {**payload, "freeze_sha256": file_sha256(path)}


__all__ = [
    "build_pcr_firewall_audit",
    "expected_representation_artifact_groups",
    "verify_pcr_firewall_audit",
    "verify_representation_freeze",
]

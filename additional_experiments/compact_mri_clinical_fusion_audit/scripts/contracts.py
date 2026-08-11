"""Fail-closed contracts for the compact MRI--clinical fusion audit.

This experiment is a downstream audit.  It deliberately imports Goal 2's
validated loaders instead of duplicating or weakening their patient, fold,
clinical, FTV, and LOCAL-feature contracts.  The compact configuration pins the
four human/machine-readable Goal 2 source documents that define the reuse
boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
DEFAULT_CONFIG_PATH = EXPERIMENT_ROOT / "configs" / "audit.json"
GOAL2_EXPERIMENT_RELATIVE = Path(
    "additional_experiments/mri_clinical_complementarity_audit"
)
GOAL2_CONTRACT_SCRIPT_RELATIVE = GOAL2_EXPERIMENT_RELATIVE / "scripts/data_contracts.py"

POPULATIONS = ("full_808", "ftv_complete_375")
EXPECTED_POPULATION_COUNTS: Mapping[str, int] = {
    "full_808": 808,
    "ftv_complete_375": 375,
}
EXPECTED_PCR_POSITIVES: Mapping[str, int] = {
    "full_808": 275,
    "ftv_complete_375": 110,
}
TIMINGS = ("T0", "T1", "T2", "T3")
SPLITS = ("train", "val", "test")

SOURCE_PATH_KEYS = (
    "config",
    "timing_contract",
    "final_report",
    "clinical_inventory",
)

# Every scientific setting is intentionally exact.  In particular, a caller
# cannot silently add a PCA dimension, change a stacking split, or repoint a
# source document while retaining the same schema version.
EXPECTED_CONFIG: Mapping[str, Any] = {
    "schema_version": 1,
    "experiment": "compact_mri_clinical_fusion_audit",
    "branch": "feature/compact-mri-clinical-fusion-audit",
    "parent_commit": "064e0596348f0972decc39774336580f58e8da61",
    "evidence_status": "diagnostic_exploratory_two_seed_compact_fusion_audit",
    "source_goal2": {
        "config": (
            "additional_experiments/mri_clinical_complementarity_audit/"
            "configs/audit.json"
        ),
        "config_sha256": (
            "e95c4971bf6d4c4c39cda9560a6350d5a8eace59235da73941d0334bdd7f5c98"
        ),
        "timing_contract": (
            "additional_experiments/mri_clinical_complementarity_audit/"
            "information_timing_contract.csv"
        ),
        "timing_contract_sha256": (
            "e8d36252dd0d19c0542e1fb061242e886b028ff68f039e908776f72929e293c4"
        ),
        "final_report": (
            "additional_experiments/mri_clinical_complementarity_audit/"
            "reports/final_report.md"
        ),
        "final_report_sha256": (
            "0236fd20d4d1170720b4bdc9e3af913a139be38bfcd711df316b9d21de142f3f"
        ),
        "clinical_inventory": (
            "additional_experiments/mri_clinical_complementarity_audit/"
            "reports/clinical_feature_inventory.md"
        ),
        "clinical_inventory_sha256": (
            "50cbc417b3c60d67067f3d8c673ee9ac0e0ee200cfba9aff40a787e0ca36f262"
        ),
    },
    "populations": ["full_808", "ftv_complete_375"],
    "timings": ["T0", "T1", "T2", "T3"],
    "pca": {
        "dimensions": [8, 16, 32, 64],
        "max_components": 64,
        "svd_solver": "full",
        "whiten": False,
        "scope": "outer_train_timing_prefix",
        "output_dimension_semantics": "k_total_dimensions_per_timing_prefix",
        "selection_metric": "validation_auroc",
        "tie_break": "smaller_dimension_then_smaller_C",
    },
    "random_projection": {
        "dimensions": [16, 32],
        "distribution": "gaussian_N_0_1_over_sqrt_k",
        "seed": 260812,
        "selection": "none_prespecified_sensitivity",
    },
    "late_fusion": {
        "inner_folds": 5,
        "inner_seed": 260813,
        "probability_clip": 1e-6,
        "meta_features": ["clinical_logit", "mri_logit"],
        "train_predictions": "strict_inner_oof",
        "mri_dimension_source": "joint_outer_validation_late_fusion",
    },
    "bootstrap": {
        "replicates": 2000,
        "confidence_level": 0.95,
        "unit": "patient",
        "stratify_by": "outer_fold",
        "random_seed": 260814,
    },
    "profile": {
        "representations": ["raw", "pca16", "pca32"],
        "targets": ["HR", "HER2", "subtype_4class"],
    },
}


class ContractError(ValueError):
    """Raised when an input or artifact violates the frozen compact contract."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _regular_file(path: str | Path, label: str) -> Path:
    try:
        source = Path(path).expanduser().resolve(strict=True)
        mode = source.stat().st_mode
    except OSError as error:
        raise ContractError(f"{label} is missing or inaccessible: {path}") from error
    if not stat.S_ISREG(mode):
        raise ContractError(f"{label} is not a regular file: {source}")
    return source


def _read_json(path: str | Path, label: str) -> dict[str, Any]:
    source = _regular_file(path, label)

    def reject_constant(value: str) -> None:
        raise ContractError(f"{label} contains non-finite JSON constant {value}")

    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"{label} is unreadable or invalid JSON: {source}") from error
    if not isinstance(value, dict):
        raise ContractError(f"{label} must contain one JSON object")
    return value


def _exact_value(actual: Any, expected: Any, label: str) -> None:
    """Require recursive value and type equality with useful drift errors."""

    if type(actual) is not type(expected):  # noqa: E721 - intentional strict type check
        raise ContractError(
            f"{label} type drifted: expected {type(expected).__name__}, "
            f"got {type(actual).__name__}"
        )
    if isinstance(expected, Mapping):
        actual_keys = set(actual)
        expected_keys = set(expected)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            raise ContractError(f"{label} keys drifted; missing={missing}, extra={extra}")
        for key in expected:
            _exact_value(actual[key], expected[key], f"{label}.{key}")
        return
    if isinstance(expected, list):
        if len(actual) != len(expected):
            raise ContractError(
                f"{label} length drifted: expected {len(expected)}, got {len(actual)}"
            )
        for index, (observed, wanted) in enumerate(zip(actual, expected, strict=True)):
            _exact_value(observed, wanted, f"{label}[{index}]")
        return
    if actual != expected:
        raise ContractError(f"{label} drifted: expected {expected!r}, got {actual!r}")


def require_sha256(value: Any, label: str = "SHA-256") -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"{label} must be a 64-character lowercase SHA-256")
    return value


def file_sha256(path: str | Path) -> str:
    source = _regular_file(path, "hashed file")
    before = source.stat()
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = source.stat()
    stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable):
        raise ContractError(f"file changed while being hashed: {source}")
    return digest.hexdigest()


def require_file_sha256(
    path: str | Path, expected_sha256: Any, label: str = "file"
) -> Path:
    source = _regular_file(path, label)
    expected = require_sha256(expected_sha256, f"{label} expected SHA-256")
    observed = file_sha256(source)
    if observed != expected:
        raise ContractError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    return source


def _resolve_repo_source(repo_root: Path, relative: str, label: str) -> Path:
    if not isinstance(relative, str) or not relative or relative != relative.strip():
        raise ContractError(f"{label} must be a non-empty relative path")
    path = Path(relative)
    if path.is_absolute():
        raise ContractError(f"{label} must be repository-relative")
    root = repo_root.expanduser().resolve(strict=True)
    source = _regular_file(root / path, label)
    try:
        source.relative_to(root)
    except ValueError as error:
        raise ContractError(f"{label} escapes the repository root: {source}") from error
    return source


def load_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    repo_root: str | Path | None = None,
    verify_sources: bool = True,
) -> dict[str, Any]:
    """Load the exact compact config and resolve its four pinned Goal 2 sources."""

    payload = _read_json(path, "compact audit config")
    _exact_value(payload, EXPECTED_CONFIG, "compact audit config")
    root = Path(repo_root).expanduser().resolve(strict=True) if repo_root else REPO_ROOT

    source = payload["source_goal2"]
    resolved_source = dict(source)
    for key in SOURCE_PATH_KEYS:
        resolved_path = _resolve_repo_source(root, source[key], f"Goal 2 {key}")
        digest_key = f"{key}_sha256"
        digest = require_sha256(source[digest_key], f"source_goal2.{digest_key}")
        if verify_sources:
            require_file_sha256(resolved_path, digest, f"Goal 2 {key}")
        resolved_source[key] = resolved_path

    result = dict(payload)
    result["source_goal2"] = resolved_source
    return result


load_compact_config = load_config


_GOAL2_MODULES: dict[Path, ModuleType] = {}


def load_goal2_contract_module(
    *, repo_root: str | Path | None = None
) -> ModuleType:
    """Import Goal 2's data contracts by absolute file path, without editing them."""

    root = Path(repo_root).expanduser().resolve(strict=True) if repo_root else REPO_ROOT
    script = _regular_file(root / GOAL2_CONTRACT_SCRIPT_RELATIVE, "Goal 2 contracts")
    cached = _GOAL2_MODULES.get(script)
    if cached is not None:
        return cached

    suffix = hashlib.sha256(str(script).encode("utf-8")).hexdigest()[:16]
    module_name = f"_compact_audit_goal2_contracts_{suffix}"
    specification = importlib.util.spec_from_file_location(module_name, script)
    if specification is None or specification.loader is None:
        raise ContractError(f"cannot import Goal 2 contracts from {script}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    required = (
        "load_config",
        "load_fold_manifest",
        "load_clinical_table",
        "load_ftv_wide",
        "load_all_local_feature_assets",
    )
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        sys.modules.pop(module_name, None)
        raise ContractError(f"Goal 2 contracts lack required callables: {missing}")
    _GOAL2_MODULES[script] = module
    return module


@dataclass(frozen=True)
class FrozenGoal2Inputs:
    """Validated Goal 2 tables and, when requested, all 20 LOCAL assets."""

    compact_config: Mapping[str, Any]
    goal2_config: Mapping[str, Any]
    fold_manifest: pd.DataFrame
    clinical: pd.DataFrame
    ftv_wide: pd.DataFrame
    assets: Mapping[tuple[int, str, int], Any]

    @property
    def folds(self) -> pd.DataFrame:
        return self.fold_manifest

    @property
    def ftv(self) -> pd.DataFrame:
        return self.ftv_wide


def load_frozen_goal2_inputs(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    repo_root: str | Path | None = None,
    load_assets: bool = True,
) -> FrozenGoal2Inputs:
    """Validate Goal 2 provenance, tables, labels, folds, and frozen LOCAL cells."""

    if type(load_assets) is not bool:  # noqa: E721 - reject truthy non-booleans
        raise ContractError("load_assets must be a boolean")
    root = Path(repo_root).expanduser().resolve(strict=True) if repo_root else REPO_ROOT
    compact = load_config(config_path, repo_root=root, verify_sources=True)
    goal2 = load_goal2_contract_module(repo_root=root)
    source = compact["source_goal2"]
    goal2_config = goal2.load_config(source["config"], repo_root=root, verify_paths=True)
    paths = goal2_config["paths"]
    folds = goal2.load_fold_manifest(paths["fold_manifest"], paths["fold_manifest_sha256"])
    clinical = goal2.load_clinical_table(
        paths["clinical_labels"], paths["clinical_labels_sha256"], folds
    )
    ftv = goal2.load_ftv_wide(
        paths["ftv_table"], paths["ftv_table_sha256"], folds
    )
    assets = goal2.load_all_local_feature_assets(goal2_config, folds) if load_assets else {}

    if len(clinical) != 808 or clinical["patient_id"].nunique() != 808:
        raise ContractError("Goal 2 full population must contain 808 unique patients")
    if int(clinical["label_pcr"].sum()) != 275:
        raise ContractError("Goal 2 full population pCR-positive count drifted")
    if len(ftv) != 375 or ftv["patient_id"].nunique() != 375:
        raise ContractError("Goal 2 FTV population must contain 375 unique patients")
    selected = clinical.loc[clinical["patient_id"].isin(set(ftv["patient_id"]))]
    if len(selected) != 375 or int(selected["label_pcr"].sum()) != 110:
        raise ContractError("Goal 2 FTV-complete population or labels drifted")
    if load_assets and set(assets) != {
        (seed, arm, fold)
        for seed in (2026, 3026)
        for arm in ("LOCAL0", "LOCAL3")
        for fold in range(5)
    }:
        raise ContractError("Goal 2 LOCAL asset matrix must be exactly 20 cells")
    return FrozenGoal2Inputs(compact, goal2_config, folds, clinical, ftv, assets)


def _identifier_array(values: Iterable[Any], label: str) -> np.ndarray:
    if isinstance(values, (str, bytes)):
        raise ContractError(f"{label} must be an iterable of patient IDs")
    array = np.asarray(list(values), dtype=object)
    if array.ndim != 1 or array.size == 0:
        raise ContractError(f"{label} must be a non-empty one-dimensional sequence")
    identifiers: list[str] = []
    for value in array:
        if value is None or pd.isna(value):
            raise ContractError(f"{label} contains a missing patient ID")
        text = str(value)
        if not text or text != text.strip():
            raise ContractError(f"{label} contains a blank or padded patient ID")
        identifiers.append(text)
    if len(identifiers) != len(set(identifiers)):
        raise ContractError(f"{label} contains duplicate patient IDs")
    return np.asarray(identifiers, dtype=str)


def _align_frame(
    frame: pd.DataFrame, patient_ids: Iterable[Any], label: str
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or "patient_id" not in frame.columns:
        raise ContractError(f"{label} must be a DataFrame with patient_id")
    identifiers = _identifier_array(patient_ids, f"{label} requested IDs")
    if frame["patient_id"].isna().any():
        raise ContractError(f"{label} contains missing patient IDs")
    indexed = frame.copy()
    indexed["patient_id"] = indexed["patient_id"].astype(str)
    if indexed["patient_id"].duplicated().any():
        raise ContractError(f"{label} contains duplicate patient IDs")
    indexed = indexed.set_index("patient_id", verify_integrity=True)
    missing = [value for value in identifiers if value not in indexed.index]
    if missing:
        raise ContractError(f"{label} misses requested patients: {missing[:5]}")
    return indexed.loc[identifiers.tolist()].reset_index()


def align_clinical(
    clinical: pd.DataFrame, patient_ids: Iterable[Any]
) -> pd.DataFrame:
    return _align_frame(clinical, patient_ids, "clinical table")


def align_ftv(ftv_wide: pd.DataFrame, patient_ids: Iterable[Any]) -> pd.DataFrame:
    return _align_frame(ftv_wide, patient_ids, "FTV table")


def population_mask(
    asset_or_patient_ids: Any,
    ftv_wide_or_ids: pd.DataFrame | Iterable[Any],
    population: str,
) -> np.ndarray:
    """Return a row mask for one frozen estimand without reordering patients."""

    values = (
        asset_or_patient_ids.patient_id
        if hasattr(asset_or_patient_ids, "patient_id")
        else asset_or_patient_ids
    )
    patient_ids = _identifier_array(values, "LOCAL patient IDs")
    if population not in POPULATIONS:
        raise ContractError(f"population must be one of {POPULATIONS}, got {population!r}")
    if population == "full_808":
        return np.ones(len(patient_ids), dtype=bool)
    ftv_values = (
        ftv_wide_or_ids["patient_id"]
        if isinstance(ftv_wide_or_ids, pd.DataFrame)
        and "patient_id" in ftv_wide_or_ids.columns
        else ftv_wide_or_ids
    )
    ftv_ids = set(_identifier_array(ftv_values, "FTV patient IDs"))
    return np.asarray([value in ftv_ids for value in patient_ids], dtype=bool)


def split_indices(split: Sequence[Any] | np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(split)
    if values.ndim != 1 or values.size == 0:
        raise ContractError("split must be a non-empty one-dimensional array")
    labels = values.astype(str)
    if set(labels) != set(SPLITS):
        raise ContractError("split labels must be exactly train/val/test")
    output = {name: np.flatnonzero(labels == name) for name in SPLITS}
    if any(indices.size == 0 for indices in output.values()):
        raise ContractError("train/val/test partitions must all be non-empty")
    if sum(indices.size for indices in output.values()) != labels.size:
        raise ContractError("split rows were not partitioned exactly once")
    return output


@dataclass(frozen=True)
class PopulationView:
    population: str
    patient_id: np.ndarray
    split: np.ndarray
    response_state: np.ndarray
    clinical: pd.DataFrame
    ftv_wide: pd.DataFrame | None

    @property
    def indices(self) -> dict[str, np.ndarray]:
        return split_indices(self.split)


def build_population_view(
    asset: Any,
    clinical: pd.DataFrame,
    ftv_wide: pd.DataFrame,
    population: str,
) -> PopulationView:
    """Align one fold-specific LOCAL cell to an exact Goal 2 population."""

    for attribute in ("patient_id", "split", "response_state"):
        if not hasattr(asset, attribute):
            raise ContractError(f"LOCAL asset lacks {attribute}")
    mask = population_mask(asset, ftv_wide, population)
    patient_ids = np.asarray(asset.patient_id).astype(str)[mask]
    split = np.asarray(asset.split).astype(str)[mask]
    state = np.asarray(asset.response_state)[mask]
    expected = EXPECTED_POPULATION_COUNTS[population]
    if len(patient_ids) != expected:
        raise ContractError(
            f"{population} must contain {expected} patients, got {len(patient_ids)}"
        )
    if state.ndim != 3 or state.shape != (expected, 4, 192):
        raise ContractError(f"{population} LOCAL state must have shape [{expected},4,192]")
    aligned_clinical = align_clinical(clinical, patient_ids)
    positives = int(aligned_clinical["label_pcr"].sum())
    if positives != EXPECTED_PCR_POSITIVES[population]:
        raise ContractError(f"{population} pCR-positive count drifted")
    selected_ftv = align_ftv(ftv_wide, patient_ids) if population == "ftv_complete_375" else None
    split_indices(split)
    return PopulationView(
        population=population,
        patient_id=patient_ids.copy(),
        split=split.copy(),
        response_state=state.copy(),
        clinical=aligned_clinical,
        ftv_wide=selected_ftv,
    )


def atomic_write_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("atomic CSV payload must be a pandas DataFrame")
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            frame.to_csv(stream, index=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def atomic_write_json(payload: Any, path: str | Path) -> Path:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def require_known_output_policy(
    paths: Iterable[str | Path],
    overwrite: bool,
    *,
    output_root: str | Path = EXPERIMENT_ROOT,
) -> tuple[Path, ...]:
    """Validate explicit output targets and refuse replacement without opt-in."""

    if type(overwrite) is not bool:  # noqa: E721
        raise ContractError("overwrite must be a boolean")
    root = Path(output_root).expanduser().resolve()
    resolved: list[Path] = []
    for value in paths:
        candidate = Path(value).expanduser()
        candidate = candidate if candidate.is_absolute() else root / candidate
        target = candidate.resolve(strict=False)
        try:
            target.relative_to(root)
        except ValueError as error:
            raise ContractError(f"output target escapes the known output root: {target}") from error
        resolved.append(target)
    if len(resolved) != len(set(resolved)):
        raise ContractError("known output targets contain duplicates")
    existing = [path for path in resolved if path.exists()]
    if existing and not overwrite:
        preview = "\n".join(str(path) for path in existing[:5])
        raise FileExistsError(
            "formal outputs already exist; pass --overwrite to replace only the "
            f"explicit known artifacts:\n{preview}"
        )
    return tuple(resolved)


require_output_policy = require_known_output_policy


__all__ = [
    "ContractError",
    "DEFAULT_CONFIG_PATH",
    "EXPECTED_CONFIG",
    "EXPECTED_PCR_POSITIVES",
    "EXPECTED_POPULATION_COUNTS",
    "EXPERIMENT_ROOT",
    "FrozenGoal2Inputs",
    "POPULATIONS",
    "PopulationView",
    "REPO_ROOT",
    "SPLITS",
    "TIMINGS",
    "align_clinical",
    "align_ftv",
    "atomic_write_csv",
    "atomic_write_json",
    "build_population_view",
    "file_sha256",
    "load_compact_config",
    "load_config",
    "load_frozen_goal2_inputs",
    "load_goal2_contract_module",
    "population_mask",
    "require_file_sha256",
    "require_known_output_policy",
    "require_output_policy",
    "require_sha256",
    "split_indices",
]

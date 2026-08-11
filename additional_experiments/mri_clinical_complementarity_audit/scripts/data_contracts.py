"""Fail-closed input contracts for the MRI--clinical complementarity audit.

The functions in this module are deliberately independent of the upstream
training packages.  The audit consumes frozen artifacts, so it validates the
on-disk schemas and provenance chain directly instead of importing mutable
training code.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


AUDIT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = AUDIT_ROOT / "configs" / "audit.json"

VISITS = ("T0", "T1", "T2", "T3")
TRANSITIONS: Mapping[str, tuple[str, str]] = {
    "T0→T1": ("T0", "T1"),
    "T1→T2": ("T1", "T2"),
    "T2→T3": ("T2", "T3"),
}
FOLDS = tuple(range(5))
SPLITS = ("train", "val", "test")
EXPECTED_SPLIT_COUNTS: Mapping[int, Mapping[str, int]] = {
    0: {"train": 525, "val": 121, "test": 162},
    1: {"train": 525, "val": 121, "test": 162},
    2: {"train": 525, "val": 121, "test": 162},
    3: {"train": 526, "val": 121, "test": 161},
    4: {"train": 526, "val": 121, "test": 161},
}

FOLD_COLUMNS = ("patient_id", "fold", "split", "label_pcr")
CLINICAL_COLUMNS = (
    "clinical_patient_id",
    "patient_id",
    "preprocessed_dir",
    "label_pcr",
    "label_hr",
    "label_her2",
    "label_mp",
    "age_at_screening",
    "arm",
    "hr_her2_subtype",
    "race_raw",
    "race_simple",
    "menopausal_status_raw",
    "menopausal_status_simple",
    "ethnicity",
    "raw_Patient_ID",
    "raw_HR",
    "raw_HER2",
    "raw_MP",
    "raw_pCR",
    "audit_status",
    "n_visits",
    "complete_4visits",
    "missing",
    "failed_visits",
    "aligned_dce_visits",
)
FTV_COLUMNS = (
    "patient_id",
    "trial_id",
    "transition",
    "start_visit",
    "end_visit",
    "ftv_start",
    "ftv_end",
    "ftv_absolute_change",
    "ftv_valid",
    "sphericity_start",
    "sphericity_end",
    "sphericity_absolute_change",
    "sphericity_valid",
    "ld_start",
    "ld_end",
    "ld_absolute_change",
    "ld_valid",
    "bpe_start",
    "bpe_end",
    "bpe_absolute_change",
    "bpe_valid",
)
FTV_WIDE_COLUMNS = tuple(f"FTV_{visit}" for visit in VISITS)

LOCAL_NPZ_KEYS = frozenset(
    {"patient_id", "split", "response_state", "arm", "seed_base", "fold"}
)
LOCAL_METADATA_KEYS = frozenset(
    {
        "arm",
        "checkpoint_data_provenance_sha256",
        "checkpoint_path",
        "checkpoint_sha256",
        "cohort",
        "current_data_contract_provenance_sha256",
        "experiment",
        "feature_dtype",
        "feature_implementation_sha256",
        "feature_path",
        "feature_sha256",
        "feature_shape",
        "feature_tensor",
        "fold",
        "ftv_head_called",
        "patient_order_sha256",
        "preregistration_lock",
        "preregistration_lock_sha256",
        "schema_version",
        "seed_base",
        "selected_epoch",
        "selection_path",
        "selection_sha256",
        "stage_a_sentinel_sha256",
        "test_labels_used",
        "train_patient_sha256",
        "validation_patient_sha256",
    }
)
LOCAL_SELECTION_KEYS = frozenset(
    {
        "allowed_state_loss",
        "architecture",
        "arm",
        "data_provenance_sha256",
        "delta_ftv_used",
        "effective_seed",
        "epochs",
        "experiment_pass",
        "fallback_rule",
        "finite_status",
        "fold",
        "history_sha256",
        "hyperparameters",
        "optimization_safety_pass",
        "paired_baseline_state_loss",
        "paired_initialization_sha256",
        "pcr_used",
        "preregistration",
        "preregistration_lock_sha256",
        "preregistration_status",
        "schema_version",
        "seed_base",
        "selected_epoch",
        "selected_representation_std",
        "selected_validation_base_loss",
        "selected_validation_ftv_loss",
        "selected_validation_state_loss",
        "selected_validation_total_loss",
        "selection_mode",
        "selection_rule",
        "stage_a_sentinel_sha256",
        "state_loss_degradation_fraction",
        "test_data_used",
        "train_patient_sha256",
        "val_patient_sha256",
    }
)

NUMERIC_CLINICAL_FIELDS = frozenset(
    {"label_hr", "label_her2", "label_mp", "age_at_screening"}
)
CATEGORICAL_CLINICAL_FIELDS = frozenset(
    {"arm", "race_simple", "menopausal_status_simple", "ethnicity"}
)
ALLOWED_CLINICAL_FIELDS = NUMERIC_CLINICAL_FIELDS | CATEGORICAL_CLINICAL_FIELDS
MISSING_CATEGORY = "__MISSING__"


class ContractError(ValueError):
    """Raised when an input no longer satisfies the frozen audit contract."""


def _duplicates_rejected_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _read_json(path: str | Path, label: str) -> dict[str, Any]:
    source = _regular_file(path, label)

    def reject_constant(value: str) -> None:
        raise ContractError(f"{label} contains non-finite JSON constant {value}")

    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_duplicates_rejected_object,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(
            f"{label} is unreadable or invalid JSON: {source}"
        ) from error
    if not isinstance(payload, dict):
        raise ContractError(f"{label} must be a JSON object")
    return payload


def _regular_file(path: str | Path, label: str) -> Path:
    try:
        source = Path(path).expanduser().resolve(strict=True)
        mode = source.stat().st_mode
    except OSError as error:
        raise ContractError(f"{label} is missing or inaccessible: {path}") from error
    if not stat.S_ISREG(mode):
        raise ContractError(f"{label} is not a regular file: {source}")
    return source


def _directory(path: str | Path, label: str) -> Path:
    try:
        source = Path(path).expanduser().resolve(strict=True)
    except OSError as error:
        raise ContractError(f"{label} directory is missing: {path}") from error
    if not source.is_dir():
        raise ContractError(f"{label} is not a directory: {source}")
    return source


def require_sha256(value: Any, label: str = "SHA-256") -> str:
    """Return a canonical digest or reject malformed/non-lowercase input."""

    if not isinstance(value, str) or len(value) != 64:
        raise ContractError(f"{label} must be a 64-character lowercase SHA-256")
    if value != value.lower() or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ContractError(f"{label} must be a 64-character lowercase SHA-256")
    return value


def file_sha256(path: str | Path) -> str:
    """Hash a stable regular file, detecting replacement during the read."""

    source = _regular_file(path, "hashed file")
    before = source.stat()
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = source.stat()
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise ContractError(f"file changed while being hashed: {source}")
    return digest.hexdigest()


def require_file_sha256(
    path: str | Path, expected_sha256: Any, label: str = "file"
) -> Path:
    """Resolve a regular file and require its exact expected SHA-256."""

    source = _regular_file(path, label)
    expected = require_sha256(expected_sha256, f"{label} expected SHA-256")
    observed = file_sha256(source)
    if observed != expected:
        raise ContractError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    return source


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ordered_patient_sha256(patient_ids: Iterable[Any]) -> str:
    return hashlib.sha256(
        "\n".join(str(value) for value in patient_ids).encode("utf-8")
    ).hexdigest()


def _exact_keys(
    value: Mapping[str, Any], expected: set[str] | frozenset[str], label: str
) -> None:
    observed = set(value)
    if observed != set(expected):
        missing = sorted(set(expected) - observed)
        extra = sorted(observed - set(expected))
        raise ContractError(f"{label} keys drifted; missing={missing}, extra={extra}")


def _exact_int(value: Any, expected: int, label: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) != expected
    ):
        raise ContractError(f"{label} must equal integer {expected}")


def _resolve_config_path(value: Any, repo_root: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ContractError(f"{label} must be a non-empty path string")
    expanded = Path(os.path.expandvars(value)).expanduser()
    return (expanded if expanded.is_absolute() else repo_root / expanded).resolve()


def load_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    repo_root: str | Path | None = None,
    verify_paths: bool = True,
) -> dict[str, Any]:
    """Load the audit config and resolve every relative data path from the repo.

    When ``verify_paths`` is true (the default), all hash-pinned inputs are
    immediately checked.  The returned ``paths`` values are absolute ``Path``
    objects; digest values remain strings.
    """

    payload = _read_json(path, "audit config")
    if payload.get("schema_version") != 1:
        raise ContractError("audit config must have schema_version 1")
    if payload.get("experiment") != "mri_clinical_complementarity_audit":
        raise ContractError("audit config names a different experiment")

    root = (
        Path(repo_root).expanduser().resolve() if repo_root is not None else REPO_ROOT
    )
    paths = payload.get("paths")
    if not isinstance(paths, dict):
        raise ContractError("audit config paths must be an object")
    expected_path_keys = {
        "clinical_labels",
        "clinical_labels_sha256",
        "fold_manifest",
        "fold_manifest_sha256",
        "ftv_table",
        "ftv_table_sha256",
        "local_feature_root",
        "local_preregistration_lock",
        "local_preregistration_lock_sha256",
    }
    _exact_keys(paths, expected_path_keys, "audit config paths")
    resolved = dict(paths)
    for key in (
        "clinical_labels",
        "fold_manifest",
        "ftv_table",
        "local_feature_root",
        "local_preregistration_lock",
    ):
        resolved[key] = _resolve_config_path(paths[key], root, f"paths.{key}")
    for key in (
        "clinical_labels_sha256",
        "fold_manifest_sha256",
        "ftv_table_sha256",
        "local_preregistration_lock_sha256",
    ):
        resolved[key] = require_sha256(paths[key], f"paths.{key}")

    cells = payload.get("local_cells")
    if not isinstance(cells, dict):
        raise ContractError("local_cells must be an object")
    _exact_keys(
        cells, {"arms", "seed_bases", "folds", "visits", "state_dim"}, "local_cells"
    )
    if tuple(cells.get("arms", ())) != ("LOCAL0", "LOCAL3"):
        raise ContractError("local_cells.arms must be exactly LOCAL0/LOCAL3")
    if tuple(cells.get("seed_bases", ())) != (2026, 3026):
        raise ContractError("local_cells.seed_bases must be exactly 2026/3026")
    if tuple(cells.get("folds", ())) != FOLDS:
        raise ContractError("local_cells.folds must be exactly 0..4")
    if tuple(cells.get("visits", ())) != VISITS:
        raise ContractError("local_cells.visits must be exactly T0..T3")
    _exact_int(cells.get("state_dim"), 192, "local_cells.state_dim")

    contracts = payload.get("clinical_contracts")
    if not isinstance(contracts, dict) or not contracts:
        raise ContractError("clinical_contracts must be a non-empty object")
    for name, fields in contracts.items():
        if not isinstance(name, str) or not name:
            raise ContractError("clinical contract names must be non-empty strings")
        if (
            not isinstance(fields, list)
            or not fields
            or any(not isinstance(field, str) for field in fields)
        ):
            raise ContractError(f"clinical contract {name!r} must be a non-empty list")
        if len(fields) != len(set(fields)):
            raise ContractError(f"clinical contract {name!r} repeats a field")
        unknown = sorted(set(fields) - ALLOWED_CLINICAL_FIELDS)
        if unknown:
            raise ContractError(
                f"clinical contract {name!r} has unsupported fields: {unknown}"
            )
    primary = payload.get("primary_clinical_contract")
    if primary not in contracts:
        raise ContractError(
            "primary_clinical_contract does not name a configured contract"
        )

    if verify_paths:
        require_file_sha256(
            resolved["clinical_labels"],
            resolved["clinical_labels_sha256"],
            "clinical labels",
        )
        require_file_sha256(
            resolved["fold_manifest"], resolved["fold_manifest_sha256"], "fold manifest"
        )
        require_file_sha256(
            resolved["ftv_table"], resolved["ftv_table_sha256"], "FTV transition table"
        )
        require_file_sha256(
            resolved["local_preregistration_lock"],
            resolved["local_preregistration_lock_sha256"],
            "LOCAL preregistration lock",
        )
        _directory(resolved["local_feature_root"], "LOCAL feature root")

    output = dict(payload)
    output["paths"] = resolved
    return output


load_audit_config = load_config


def _read_csv(path: str | Path, expected_sha256: Any, label: str) -> pd.DataFrame:
    source = require_file_sha256(path, expected_sha256, label)
    try:
        frame = pd.read_csv(source)
    except (OSError, UnicodeError, pd.errors.ParserError) as error:
        raise ContractError(f"{label} is not a readable CSV: {source}") from error
    if frame.columns.duplicated().any():
        raise ContractError(f"{label} contains duplicate columns")
    return frame


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    if tuple(frame.columns) != tuple(columns):
        raise ContractError(
            f"{label} schema/order drifted: expected {list(columns)}, got {list(frame.columns)}"
        )


def _identifiers(series: pd.Series, label: str) -> pd.Series:
    values = series.astype("string")
    if values.isna().any():
        raise ContractError(f"{label} contains missing identifiers")
    stripped = values.str.strip()
    if (stripped == "").any() or not stripped.equals(values):
        raise ContractError(f"{label} contains blank or whitespace-padded identifiers")
    return stripped.astype(str)


def _integers(series: pd.Series, label: str) -> pd.Series:
    try:
        numeric = pd.to_numeric(series, errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ContractError(f"{label} must contain integers") from error
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ContractError(f"{label} must contain finite integers")
    return pd.Series(numeric.astype(np.int64), index=series.index, name=series.name)


def _binary(series: pd.Series, label: str) -> pd.Series:
    values = _integers(series, label)
    if not values.isin((0, 1)).all():
        raise ContractError(f"{label} must contain only binary 0/1 values")
    return values.astype(np.int8)


def _booleans(series: pd.Series, label: str) -> pd.Series:
    if series.isna().any():
        raise ContractError(f"{label} contains missing boolean values")
    mapping = {
        True: True,
        False: False,
        1: True,
        0: False,
        "True": True,
        "False": False,
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }
    mapped = series.map(mapping)
    if mapped.isna().any():
        raise ContractError(f"{label} must contain only boolean values")
    return mapped.astype(bool)


def _patient_set(value: pd.DataFrame | Iterable[Any], label: str) -> set[str]:
    if isinstance(value, pd.DataFrame):
        if "patient_id" not in value.columns:
            raise ContractError(f"{label} has no patient_id column")
        series = _identifiers(value["patient_id"], f"{label}.patient_id")
        return set(series)
    result = [str(item) for item in value]
    if any(not item or item != item.strip() for item in result):
        raise ContractError(f"{label} contains invalid patient IDs")
    return set(result)


def load_fold_manifest(
    path: str | Path,
    expected_sha256: Any,
    *,
    expected_patient_count: int = 808,
    expected_folds: Sequence[int] = FOLDS,
    expected_split_counts: Mapping[int, Mapping[str, int]] | None = None,
) -> pd.DataFrame:
    """Load the exact long outer-fold manifest and validate all invariants."""

    frame = _read_csv(path, expected_sha256, "fold manifest")
    _require_columns(frame, FOLD_COLUMNS, "fold manifest")
    if isinstance(expected_patient_count, bool) or int(expected_patient_count) <= 0:
        raise ContractError("expected_patient_count must be positive")
    folds = tuple(int(value) for value in expected_folds)
    if len(folds) != len(set(folds)) or not folds:
        raise ContractError("expected_folds must contain distinct fold integers")
    expected_rows = int(expected_patient_count) * len(folds)
    if len(frame) != expected_rows:
        raise ContractError(f"fold manifest must contain exactly {expected_rows} rows")

    frame = frame.copy()
    frame["patient_id"] = _identifiers(frame["patient_id"], "fold manifest patient_id")
    frame["fold"] = _integers(frame["fold"], "fold manifest fold")
    frame["split"] = _identifiers(frame["split"], "fold manifest split")
    frame["label_pcr"] = _binary(frame["label_pcr"], "fold manifest label_pcr")
    if set(frame["fold"]) != set(folds):
        raise ContractError("fold manifest fold set differs from the configured folds")
    if set(frame["split"]) != set(SPLITS):
        raise ContractError("fold manifest split labels must be exactly train/val/test")
    if frame.duplicated(["patient_id", "fold"]).any():
        raise ContractError("fold manifest repeats a patient within a fold")

    per_patient = frame.groupby("patient_id", sort=False)
    if frame["patient_id"].nunique() != int(expected_patient_count):
        raise ContractError("fold manifest patient count drifted")
    if not per_patient.size().eq(len(folds)).all():
        raise ContractError("every patient must have exactly one row per fold")
    if not per_patient["fold"].nunique().eq(len(folds)).all():
        raise ContractError("every patient must occur in every configured fold")
    if not per_patient["label_pcr"].nunique().eq(1).all():
        raise ContractError("pCR labels must be stable across folds")
    test_counts = frame["split"].eq("test").groupby(frame["patient_id"]).sum()
    if not test_counts.eq(1).all():
        raise ContractError("every patient must be assigned to test exactly once")

    counts_contract = expected_split_counts
    if (
        counts_contract is None
        and int(expected_patient_count) == 808
        and folds == FOLDS
    ):
        counts_contract = EXPECTED_SPLIT_COUNTS
    if counts_contract is not None:
        if set(int(key) for key in counts_contract) != set(folds):
            raise ContractError("expected split-count contract has the wrong folds")
        for fold in folds:
            observed = (
                frame.loc[frame["fold"].eq(fold), "split"].value_counts().to_dict()
            )
            expected = {
                name: int(counts_contract[fold].get(name, -1)) for name in SPLITS
            }
            if set(counts_contract[fold]) != set(SPLITS) or observed != expected:
                raise ContractError(
                    f"fold {fold} split counts drifted: expected {expected}, observed {observed}"
                )
    return frame.reset_index(drop=True)


def load_clinical_table(
    path: str | Path,
    expected_sha256: Any,
    fold_manifest: pd.DataFrame | None = None,
    *,
    expected_patient_ids: Iterable[Any] | None = None,
    expected_patient_count: int = 808,
) -> pd.DataFrame:
    """Load clinical labels and require exact equality with the fold cohort."""

    if fold_manifest is not None and expected_patient_ids is not None:
        raise ContractError("provide fold_manifest or expected_patient_ids, not both")
    reference = fold_manifest if fold_manifest is not None else expected_patient_ids
    if reference is None:
        raise ContractError(
            "clinical loading requires the locked expected patient cohort"
        )

    frame = _read_csv(path, expected_sha256, "clinical table")
    _require_columns(frame, CLINICAL_COLUMNS, "clinical table")
    if len(frame) != int(expected_patient_count):
        raise ContractError(
            f"clinical table must contain exactly {expected_patient_count} rows"
        )
    frame = frame.copy()
    frame["patient_id"] = _identifiers(frame["patient_id"], "clinical patient_id")
    if frame["patient_id"].duplicated().any():
        raise ContractError("clinical table contains duplicate patient IDs")
    observed_ids = set(frame["patient_id"])
    expected_ids = _patient_set(reference, "expected clinical cohort")
    if observed_ids != expected_ids:
        missing = sorted(expected_ids - observed_ids)[:5]
        extra = sorted(observed_ids - expected_ids)[:5]
        raise ContractError(
            f"clinical/fold patient equality failed; missing={missing}, extra={extra}"
        )

    for column in ("label_pcr", "label_hr", "label_her2", "label_mp"):
        frame[column] = _binary(frame[column], f"clinical {column}")
    try:
        age = pd.to_numeric(frame["age_at_screening"], errors="raise").astype(float)
    except (TypeError, ValueError) as error:
        raise ContractError(
            "clinical age_at_screening must be numeric or missing"
        ) from error
    if np.isinf(age.to_numpy(dtype=float)).any():
        raise ContractError("clinical age_at_screening contains infinity")
    frame["age_at_screening"] = age
    frame["arm"] = _identifiers(frame["arm"], "clinical treatment arm")

    subtype = _identifiers(frame["hr_her2_subtype"], "clinical HR/HER2 subtype")
    expected_subtype = np.select(
        [
            frame["label_hr"].eq(1) & frame["label_her2"].eq(0),
            frame["label_hr"].eq(0) & frame["label_her2"].eq(0),
            frame["label_hr"].eq(1) & frame["label_her2"].eq(1),
            frame["label_hr"].eq(0) & frame["label_her2"].eq(1),
        ],
        ["HR+/HER2-", "HR-/HER2-", "HR+/HER2+", "HR-/HER2+"],
        default="INVALID",
    )
    if not np.array_equal(subtype.to_numpy(), expected_subtype):
        raise ContractError("clinical HR/HER2 subtype disagrees with binary labels")
    frame["hr_her2_subtype"] = subtype

    if fold_manifest is not None:
        labels = fold_manifest[["patient_id", "label_pcr"]].drop_duplicates(
            "patient_id"
        )
        if len(labels) != len(frame):
            raise ContractError(
                "fold manifest does not have one stable pCR label per patient"
            )
        expected_labels = dict(
            zip(labels["patient_id"].astype(str), labels["label_pcr"], strict=True)
        )
        observed_labels = dict(
            zip(frame["patient_id"], frame["label_pcr"], strict=True)
        )
        if observed_labels != expected_labels:
            raise ContractError("clinical pCR labels disagree with the fold manifest")
    return frame.reset_index(drop=True)


def load_ftv_wide(
    path: str | Path,
    expected_sha256: Any,
    expected_patient_ids: pd.DataFrame | Iterable[Any] | None = None,
    *,
    expected_patient_count: int = 375,
    rtol: float = 1e-10,
    atol: float = 1e-10,
) -> pd.DataFrame:
    """Validate the transition table and reconstruct observed T0--T3 FTV.

    Each output visit is obtained only from the adjacent observed transition;
    repeated interior visits must match exactly within the supplied numerical
    tolerance.  This prevents accidental use of later transition targets.
    """

    frame = _read_csv(path, expected_sha256, "FTV transition table")
    _require_columns(frame, FTV_COLUMNS, "FTV transition table")
    expected_rows = int(expected_patient_count) * len(TRANSITIONS)
    if len(frame) != expected_rows:
        raise ContractError(
            f"FTV transition table must contain exactly {expected_rows} rows"
        )
    frame = frame.copy()
    frame["patient_id"] = _identifiers(frame["patient_id"], "FTV patient_id")
    frame["transition"] = _identifiers(frame["transition"], "FTV transition")
    frame["start_visit"] = _identifiers(frame["start_visit"], "FTV start_visit")
    frame["end_visit"] = _identifiers(frame["end_visit"], "FTV end_visit")
    frame["trial_id"] = _integers(frame["trial_id"], "FTV trial_id")
    frame["ftv_valid"] = _booleans(frame["ftv_valid"], "FTV ftv_valid")
    if not frame["ftv_valid"].all():
        raise ContractError("FTV reconstruction requires ftv_valid=True for every row")
    for column in ("ftv_start", "ftv_end", "ftv_absolute_change"):
        try:
            values = pd.to_numeric(frame[column], errors="raise").astype(float)
        except (TypeError, ValueError) as error:
            raise ContractError(f"FTV {column} must be numeric") from error
        if not np.isfinite(values.to_numpy()).all():
            raise ContractError(f"FTV {column} contains non-finite values")
        frame[column] = values
    if (frame[["ftv_start", "ftv_end"]] < 0).any(axis=None):
        raise ContractError("FTV measurements must be non-negative")
    if not np.allclose(
        frame["ftv_absolute_change"].to_numpy(),
        (frame["ftv_end"] - frame["ftv_start"]).to_numpy(),
        rtol=rtol,
        atol=atol,
    ):
        raise ContractError(
            "FTV absolute-change values disagree with transition endpoints"
        )
    if frame.duplicated(["patient_id", "transition"]).any():
        raise ContractError("FTV table repeats a patient transition")
    if frame["patient_id"].nunique() != int(expected_patient_count):
        raise ContractError("FTV patient count drifted")
    grouped = frame.groupby("patient_id", sort=False)
    if not grouped.size().eq(len(TRANSITIONS)).all():
        raise ContractError("every FTV patient must have exactly three transitions")
    if not grouped["trial_id"].nunique().eq(1).all():
        raise ContractError("FTV trial_id changes within patient")
    if any(set(group["transition"]) != set(TRANSITIONS) for _, group in grouped):
        raise ContractError("every FTV patient must have exactly T0→T1, T1→T2, T2→T3")
    for transition, (start, end) in TRANSITIONS.items():
        rows = frame["transition"].eq(transition)
        if (
            not frame.loc[rows, "start_visit"].eq(start).all()
            or not frame.loc[rows, "end_visit"].eq(end).all()
        ):
            raise ContractError(
                f"FTV transition {transition} has inconsistent visit labels"
            )

    if expected_patient_ids is not None:
        allowed = _patient_set(expected_patient_ids, "allowed FTV cohort")
        extra = set(frame["patient_id"]) - allowed
        if extra:
            raise ContractError(
                f"FTV table contains patients outside the locked cohort: {sorted(extra)[:5]}"
            )

    rows: list[dict[str, Any]] = []
    for patient_id, patient in grouped:
        by_transition = patient.set_index("transition", verify_integrity=True)
        first = by_transition.loc["T0→T1"]
        second = by_transition.loc["T1→T2"]
        third = by_transition.loc["T2→T3"]
        if not np.isclose(first["ftv_end"], second["ftv_start"], rtol=rtol, atol=atol):
            raise ContractError(
                f"FTV_T1 is inconsistent across transitions for {patient_id}"
            )
        if not np.isclose(second["ftv_end"], third["ftv_start"], rtol=rtol, atol=atol):
            raise ContractError(
                f"FTV_T2 is inconsistent across transitions for {patient_id}"
            )
        rows.append(
            {
                "patient_id": str(patient_id),
                "FTV_T0": float(first["ftv_start"]),
                "FTV_T1": float(first["ftv_end"]),
                "FTV_T2": float(second["ftv_end"]),
                "FTV_T3": float(third["ftv_end"]),
            }
        )
    wide = pd.DataFrame(rows, columns=("patient_id", *FTV_WIDE_COLUMNS))
    return wide.sort_values("patient_id", kind="stable").reset_index(drop=True)


load_ftv_table = load_ftv_wide
reconstruct_ftv_wide = load_ftv_wide


@dataclass(frozen=True)
class LocalFeatureAsset:
    path: Path
    metadata_path: Path
    patient_id: np.ndarray
    split: np.ndarray
    response_state: np.ndarray
    arm: str
    seed_base: int
    fold: int
    metadata: Mapping[str, Any]
    checkpoint_path: Path
    selection_path: Path
    selection: Mapping[str, Any]

    @property
    def arrays(self) -> Mapping[str, np.ndarray]:
        return {
            "patient_id": self.patient_id,
            "split": self.split,
            "response_state": self.response_state,
            "arm": np.asarray(self.arm),
            "seed_base": np.asarray(self.seed_base, dtype=np.int64),
            "fold": np.asarray(self.fold, dtype=np.int64),
        }


def _identity_path(
    path: Path, arm: str, seed_base: int, fold: int, filename: str, label: str
) -> None:
    expected_tail = (f"seed_{seed_base}", arm, f"fold_{fold}", filename)
    if tuple(path.parts[-4:]) != expected_tail:
        raise ContractError(
            f"{label} path is not bound to cell {expected_tail}: {path}"
        )


def _json_identity(
    payload: Mapping[str, Any], arm: str, seed_base: int, fold: int, label: str
) -> None:
    if payload.get("arm") != arm:
        raise ContractError(f"{label} arm identity drifted")
    _exact_int(payload.get("seed_base"), seed_base, f"{label}.seed_base")
    _exact_int(payload.get("fold"), fold, f"{label}.fold")


def load_local_feature_asset(
    path: str | Path,
    fold_manifest: pd.DataFrame,
    *,
    arm: str,
    seed_base: int,
    fold: int,
    preregistration_lock_sha256: Any | None = None,
    expected_lock_sha256: Any | None = None,
    metadata_path: str | Path | None = None,
    expected_patient_count: int = 808,
    expected_visits: int = 4,
    state_dim: int = 192,
) -> LocalFeatureAsset:
    """Load one LOCAL feature cell and validate its complete provenance chain."""

    if preregistration_lock_sha256 is not None and expected_lock_sha256 is not None:
        raise ContractError("provide one preregistration-lock digest, not two")
    lock_value = (
        preregistration_lock_sha256
        if preregistration_lock_sha256 is not None
        else expected_lock_sha256
    )
    if lock_value is None:
        raise ContractError("a preregistration-lock SHA-256 is required")
    lock_digest = require_sha256(lock_value, "LOCAL preregistration lock")
    if arm not in {"LOCAL0", "LOCAL3"}:
        raise ContractError("LOCAL feature arm must be LOCAL0 or LOCAL3")
    if isinstance(seed_base, bool) or not isinstance(seed_base, (int, np.integer)):
        raise ContractError("LOCAL seed_base must be an integer")
    if isinstance(fold, bool) or int(fold) not in FOLDS:
        raise ContractError("LOCAL fold must be 0..4")
    source = _regular_file(path, "LOCAL feature NPZ")
    _identity_path(
        source, arm, int(seed_base), int(fold), "response_state.private.npz", "feature"
    )

    try:
        with np.load(source, allow_pickle=False) as archive:
            if set(archive.files) != set(LOCAL_NPZ_KEYS):
                raise ContractError(
                    f"LOCAL feature NPZ keys drifted: expected {sorted(LOCAL_NPZ_KEYS)}, got {sorted(archive.files)}"
                )
            arrays = {key: archive[key].copy() for key in archive.files}
    except (OSError, ValueError) as error:
        if isinstance(error, ContractError):
            raise
        raise ContractError(f"LOCAL feature NPZ is unreadable: {source}") from error

    patient_raw = arrays["patient_id"]
    split_raw = arrays["split"]
    response = arrays["response_state"]
    arm_raw = arrays["arm"]
    seed_raw = arrays["seed_base"]
    fold_raw = arrays["fold"]
    if patient_raw.ndim != 1 or patient_raw.dtype.kind != "U":
        raise ContractError("LOCAL patient_id must be a one-dimensional Unicode array")
    if split_raw.ndim != 1 or split_raw.dtype.kind != "U":
        raise ContractError("LOCAL split must be a one-dimensional Unicode array")
    if (
        len(patient_raw) != int(expected_patient_count)
        or split_raw.shape != patient_raw.shape
    ):
        raise ContractError("LOCAL patient/split shapes or patient count drifted")
    if response.dtype != np.dtype("float32") or response.shape != (
        int(expected_patient_count),
        int(expected_visits),
        int(state_dim),
    ):
        raise ContractError(
            f"LOCAL response_state must be float32 [{expected_patient_count},{expected_visits},{state_dim}]"
        )
    if not np.isfinite(response).all():
        raise ContractError("LOCAL response_state contains non-finite values")
    if arm_raw.shape != () or arm_raw.dtype.kind != "U" or str(arm_raw.item()) != arm:
        raise ContractError("LOCAL NPZ arm scalar/dtype/identity drifted")
    if (
        seed_raw.shape != ()
        or seed_raw.dtype != np.dtype("int64")
        or int(seed_raw.item()) != int(seed_base)
    ):
        raise ContractError("LOCAL NPZ seed_base must be the exact int64 cell scalar")
    if (
        fold_raw.shape != ()
        or fold_raw.dtype != np.dtype("int64")
        or int(fold_raw.item()) != int(fold)
    ):
        raise ContractError("LOCAL NPZ fold must be the exact int64 cell scalar")

    patient_ids = patient_raw.astype(str, copy=False)
    split_labels = split_raw.astype(str, copy=False)
    if any(not value or value != value.strip() for value in patient_ids):
        raise ContractError(
            "LOCAL patient IDs contain blanks or surrounding whitespace"
        )
    if len(set(patient_ids)) != len(patient_ids):
        raise ContractError("LOCAL feature NPZ contains duplicate patients")
    if set(split_labels) != set(SPLITS):
        raise ContractError("LOCAL feature split labels must be exactly train/val/test")

    required_fold_columns = {"patient_id", "fold", "split"}
    if not required_fold_columns.issubset(fold_manifest.columns):
        raise ContractError(
            "fold manifest lacks patient_id/fold/split for feature validation"
        )
    current = fold_manifest.loc[
        fold_manifest["fold"].eq(int(fold)), ["patient_id", "split"]
    ].copy()
    if len(current) != int(expected_patient_count):
        raise ContractError("locked fold patient count differs from the LOCAL asset")
    current["patient_id"] = _identifiers(
        current["patient_id"], "locked fold patient_id"
    )
    current["split"] = _identifiers(current["split"], "locked fold split")
    if current["patient_id"].duplicated().any():
        raise ContractError("locked fold contains duplicate patients")
    expected_assignment = dict(
        zip(current["patient_id"], current["split"], strict=True)
    )
    observed_assignment = dict(zip(patient_ids, split_labels, strict=True))
    if observed_assignment != expected_assignment:
        raise ContractError(
            "LOCAL patients/splits do not exactly match the locked fold"
        )

    sidecar = (
        _regular_file(metadata_path, "LOCAL feature metadata")
        if metadata_path is not None
        else source.with_suffix(".metadata.json")
    )
    sidecar = _regular_file(sidecar, "LOCAL feature metadata")
    if (
        sidecar.name != "response_state.private.metadata.json"
        or sidecar.parent != source.parent
    ):
        raise ContractError(
            "LOCAL metadata must be the canonical sidecar beside the NPZ"
        )
    metadata = _read_json(sidecar, "LOCAL feature metadata")
    _exact_keys(metadata, LOCAL_METADATA_KEYS, "LOCAL feature metadata")
    _exact_int(metadata.get("schema_version"), 1, "feature metadata.schema_version")
    _json_identity(metadata, arm, int(seed_base), int(fold), "feature metadata")
    expected_metadata = {
        "experiment": "local_global_response_state_pilot",
        "feature_tensor": "online_preprojector_response_state",
        "feature_dtype": "float32",
        "cohort": "exact_locked_primary_train_validation_test",
        "ftv_head_called": False,
        "test_labels_used": False,
        "preregistration_lock": "PREREGISTRATION_LOCK.json",
        "preregistration_lock_sha256": lock_digest,
    }
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            raise ContractError(f"LOCAL feature metadata differs at {key}")
    for key in LOCAL_METADATA_KEYS:
        if key.endswith("_sha256"):
            require_sha256(metadata.get(key), f"feature metadata.{key}")
    if Path(str(metadata.get("feature_path", ""))).expanduser().resolve() != source:
        raise ContractError("LOCAL feature metadata path differs from the loaded NPZ")
    if metadata["feature_sha256"] != file_sha256(source):
        raise ContractError("LOCAL feature metadata hash differs from the loaded NPZ")
    if metadata.get("feature_shape") != list(response.shape):
        raise ContractError("LOCAL feature metadata shape differs from response_state")
    if metadata["patient_order_sha256"] != ordered_patient_sha256(patient_ids):
        raise ContractError("LOCAL feature patient-order SHA-256 drifted")

    checkpoint = _regular_file(
        metadata.get("checkpoint_path", ""), "selected LOCAL checkpoint"
    )
    selection_path = _regular_file(
        metadata.get("selection_path", ""), "LOCAL selection"
    )
    _identity_path(
        checkpoint, arm, int(seed_base), int(fold), "selected.pt", "checkpoint"
    )
    _identity_path(
        selection_path, arm, int(seed_base), int(fold), "selection.json", "selection"
    )
    if checkpoint.parent != selection_path.parent:
        raise ContractError(
            "LOCAL checkpoint and selection do not belong to the same cell"
        )
    if checkpoint.stat().st_size <= 0:
        raise ContractError("selected LOCAL checkpoint is empty")
    if metadata["checkpoint_sha256"] != file_sha256(checkpoint):
        raise ContractError("selected LOCAL checkpoint SHA-256 drifted")
    if metadata["selection_sha256"] != file_sha256(selection_path):
        raise ContractError("LOCAL selection SHA-256 drifted")

    selection = _read_json(selection_path, "LOCAL selection")
    _exact_keys(selection, LOCAL_SELECTION_KEYS, "LOCAL selection")
    _exact_int(selection.get("schema_version"), 1, "selection.schema_version")
    _json_identity(selection, arm, int(seed_base), int(fold), "selection")
    _exact_int(
        selection.get("effective_seed"),
        int(seed_base) + int(fold),
        "selection.effective_seed",
    )
    if selection.get("architecture") != "LOCAL":
        raise ContractError("selected checkpoint architecture must be LOCAL")
    for key, expected in {
        "delta_ftv_used": False,
        "pcr_used": False,
        "test_data_used": False,
        "experiment_pass": True,
        "finite_status": True,
        "optimization_safety_pass": True,
        "preregistration_status": "PASS",
        "preregistration_lock_sha256": lock_digest,
    }.items():
        if selection.get(key) != expected:
            raise ContractError(f"LOCAL selection differs at {key}")
    if selection.get("preregistration") != {
        "status": "PASS",
        "lock_sha256": lock_digest,
    }:
        raise ContractError("LOCAL selection preregistration object drifted")
    for key in (
        "data_provenance_sha256",
        "history_sha256",
        "paired_initialization_sha256",
        "preregistration_lock_sha256",
        "stage_a_sentinel_sha256",
        "train_patient_sha256",
        "val_patient_sha256",
    ):
        require_sha256(selection.get(key), f"selection.{key}")
    _exact_int(
        selection.get("selected_epoch"),
        int(metadata["selected_epoch"]),
        "selection.selected_epoch",
    )
    if (
        selection["data_provenance_sha256"]
        != metadata["checkpoint_data_provenance_sha256"]
    ):
        raise ContractError("selection/checkpoint data-provenance binding drifted")
    if selection["train_patient_sha256"] != metadata["train_patient_sha256"]:
        raise ContractError("selection/feature train-patient binding drifted")
    if selection["val_patient_sha256"] != metadata["validation_patient_sha256"]:
        raise ContractError("selection/feature validation-patient binding drifted")
    if selection["stage_a_sentinel_sha256"] != metadata["stage_a_sentinel_sha256"]:
        raise ContractError("selection/feature Stage-A binding drifted")
    expected_val_hash = canonical_sha256(
        sorted(current.loc[current["split"].eq("val"), "patient_id"])
    )
    if metadata["validation_patient_sha256"] != expected_val_hash:
        raise ContractError(
            "LOCAL validation-patient SHA-256 differs from the locked fold"
        )

    return LocalFeatureAsset(
        path=source,
        metadata_path=sidecar,
        patient_id=patient_ids.copy(),
        split=split_labels.copy(),
        response_state=response.copy(),
        arm=arm,
        seed_base=int(seed_base),
        fold=int(fold),
        metadata=metadata,
        checkpoint_path=checkpoint,
        selection_path=selection_path,
        selection=selection,
    )


def load_local_feature_cell(
    config: Mapping[str, Any],
    fold_manifest: pd.DataFrame,
    *,
    arm: str,
    seed_base: int,
    fold: int,
) -> LocalFeatureAsset:
    """Load one configured feature cell from ``local_feature_root``."""

    cells = config.get("local_cells")
    paths = config.get("paths")
    if not isinstance(cells, Mapping) or not isinstance(paths, Mapping):
        raise ContractError("validated config must contain local_cells and paths")
    if (
        arm not in cells.get("arms", ())
        or seed_base not in cells.get("seed_bases", ())
        or fold not in cells.get("folds", ())
    ):
        raise ContractError("requested LOCAL feature cell is outside the config matrix")
    root = _directory(paths.get("local_feature_root", ""), "LOCAL feature root")
    feature = (
        root / f"seed_{seed_base}" / arm / f"fold_{fold}" / "response_state.private.npz"
    )
    return load_local_feature_asset(
        feature,
        fold_manifest,
        arm=arm,
        seed_base=seed_base,
        fold=fold,
        preregistration_lock_sha256=paths.get("local_preregistration_lock_sha256"),
        expected_patient_count=int(fold_manifest["patient_id"].nunique()),
        expected_visits=len(tuple(cells.get("visits", ()))),
        state_dim=int(cells.get("state_dim", -1)),
    )


def load_all_local_feature_assets(
    config: Mapping[str, Any], fold_manifest: pd.DataFrame
) -> dict[tuple[int, str, int], LocalFeatureAsset]:
    """Load and validate every configured seed/arm/fold feature cell."""

    cells = config.get("local_cells")
    if not isinstance(cells, Mapping):
        raise ContractError("validated config has no local_cells object")
    result: dict[tuple[int, str, int], LocalFeatureAsset] = {}
    for seed_base in cells.get("seed_bases", ()):
        for arm in cells.get("arms", ()):
            for fold in cells.get("folds", ()):
                identity = (int(seed_base), str(arm), int(fold))
                result[identity] = load_local_feature_cell(
                    config,
                    fold_manifest,
                    arm=identity[1],
                    seed_base=identity[0],
                    fold=identity[2],
                )
    if len(result) != 20:
        raise ContractError("configured LOCAL matrix must contain exactly 20 cells")
    return result


class TrainOnlyClinicalEncoder:
    """Deterministic fold-train imputation and one-hot encoding.

    Categorical levels are sorted levels observed on train, with an explicit
    missing token.  A non-missing validation/test level absent from train maps
    to an all-zero block.  The encoder never mutates its fitted vocabulary.
    """

    def __init__(self, fields: Sequence[str], *, missing_token: str = MISSING_CATEGORY):
        self.fields = tuple(fields)
        if not self.fields or len(self.fields) != len(set(self.fields)):
            raise ContractError("clinical encoder fields must be non-empty and unique")
        unknown = sorted(set(self.fields) - ALLOWED_CLINICAL_FIELDS)
        if unknown:
            raise ContractError(f"clinical encoder has unsupported fields: {unknown}")
        if not isinstance(missing_token, str) or not missing_token:
            raise ContractError("clinical missing token must be a non-empty string")
        self.missing_token = missing_token
        self.numeric_medians_: dict[str, float] = {}
        self.categories_: dict[str, tuple[str, ...]] = {}
        self.feature_names_: tuple[str, ...] = ()
        self._fitted = False

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], contract_name: str | None = None
    ) -> "TrainOnlyClinicalEncoder":
        contracts = config.get("clinical_contracts")
        if not isinstance(contracts, Mapping):
            raise ContractError("config has no clinical_contracts mapping")
        selected = contract_name or config.get("primary_clinical_contract")
        if selected not in contracts:
            raise ContractError(f"unknown configured clinical contract: {selected!r}")
        fields = contracts[selected]
        if not isinstance(fields, Sequence) or isinstance(fields, (str, bytes)):
            raise ContractError(f"configured clinical contract {selected!r} is invalid")
        return cls(tuple(str(field) for field in fields))

    @staticmethod
    def _category_values(series: pd.Series, missing_token: str) -> np.ndarray:
        values: list[str] = []
        for value in series.to_numpy(dtype=object):
            if pd.isna(value) or (isinstance(value, str) and not value.strip()):
                values.append(missing_token)
            else:
                text = str(value)
                if text != text.strip():
                    raise ContractError(
                        f"categorical field {series.name} contains padded whitespace"
                    )
                values.append(text)
        return np.asarray(values, dtype=object)

    def _require_columns(self, frame: pd.DataFrame) -> None:
        missing = [field for field in self.fields if field not in frame.columns]
        if missing:
            raise ContractError(f"clinical frame is missing encoder fields: {missing}")

    def fit(self, train_frame: pd.DataFrame) -> "TrainOnlyClinicalEncoder":
        if self._fitted:
            raise ContractError("clinical encoder may only be fitted once")
        if not isinstance(train_frame, pd.DataFrame) or train_frame.empty:
            raise ContractError("clinical encoder requires non-empty fold-train rows")
        self._require_columns(train_frame)
        feature_names: list[str] = []
        for field in self.fields:
            if field in NUMERIC_CLINICAL_FIELDS:
                try:
                    values = pd.to_numeric(train_frame[field], errors="raise").to_numpy(
                        dtype=float
                    )
                except (TypeError, ValueError) as error:
                    raise ContractError(
                        f"numeric clinical field {field} is not numeric"
                    ) from error
                if np.isinf(values).any():
                    raise ContractError(
                        f"numeric clinical field {field} contains infinity"
                    )
                finite = values[np.isfinite(values)]
                if finite.size == 0:
                    raise ContractError(
                        f"numeric clinical field {field} has no train value for imputation"
                    )
                self.numeric_medians_[field] = float(np.median(finite))
                feature_names.append(field)
            else:
                values = self._category_values(train_frame[field], self.missing_token)
                levels = tuple(sorted(set(values.tolist()) | {self.missing_token}))
                self.categories_[field] = levels
                feature_names.extend(f"{field}={level}" for level in levels)
        self.feature_names_ = tuple(feature_names)
        self._fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise ContractError("clinical encoder must be fitted on fold-train first")
        if not isinstance(frame, pd.DataFrame):
            raise ContractError("clinical encoder input must be a DataFrame")
        self._require_columns(frame)
        blocks: list[np.ndarray] = []
        for field in self.fields:
            if field in NUMERIC_CLINICAL_FIELDS:
                try:
                    values = pd.to_numeric(frame[field], errors="raise").to_numpy(
                        dtype=float
                    )
                except (TypeError, ValueError) as error:
                    raise ContractError(
                        f"numeric clinical field {field} is not numeric"
                    ) from error
                if np.isinf(values).any():
                    raise ContractError(
                        f"numeric clinical field {field} contains infinity"
                    )
                values = np.where(
                    np.isnan(values), self.numeric_medians_[field], values
                )
                blocks.append(values[:, None])
            else:
                values = self._category_values(frame[field], self.missing_token)
                levels = self.categories_[field]
                block = np.zeros((len(frame), len(levels)), dtype=np.float64)
                level_index = {level: index for index, level in enumerate(levels)}
                for row, value in enumerate(values):
                    index = level_index.get(str(value))
                    if index is not None:
                        block[row, index] = 1.0
                blocks.append(block)
        output = np.concatenate(blocks, axis=1).astype(np.float32, copy=False)
        if output.shape != (len(frame), len(self.feature_names_)):
            raise ContractError("clinical encoder output shape drifted")
        if not np.isfinite(output).all():
            raise ContractError("clinical encoder output contains non-finite values")
        return output

    def fit_transform(self, train_frame: pd.DataFrame) -> np.ndarray:
        return self.fit(train_frame).transform(train_frame)

    def get_feature_names_out(self) -> np.ndarray:
        if not self._fitted:
            raise ContractError("clinical encoder is not fitted")
        return np.asarray(self.feature_names_, dtype=str)


def fit_clinical_encoder(
    config: Mapping[str, Any], contract_name: str, train_frame: pd.DataFrame
) -> TrainOnlyClinicalEncoder:
    return TrainOnlyClinicalEncoder.from_config(config, contract_name).fit(train_frame)


def timing_index(timing: str | int) -> int:
    if isinstance(timing, bool):
        raise ContractError("timing must be T0..T3 or integer 0..3")
    if isinstance(timing, (int, np.integer)):
        index = int(timing)
    elif isinstance(timing, str) and timing.upper() in VISITS:
        index = VISITS.index(timing.upper())
    else:
        raise ContractError("timing must be T0..T3 or integer 0..3")
    if index not in range(len(VISITS)):
        raise ContractError("timing must be T0..T3 or integer 0..3")
    return index


def mri_timing_prefix(
    response_state: np.ndarray,
    timing: str | int,
    *,
    state_dim: int | None = 192,
) -> np.ndarray:
    """Concatenate only MRI states observed through the requested timing."""

    values = np.asarray(response_state)
    if values.ndim != 3 or values.shape[1] != len(VISITS):
        raise ContractError("MRI response_state must have shape [N,4,D]")
    if state_dim is not None and values.shape[2] != int(state_dim):
        raise ContractError(f"MRI response_state final dimension must be {state_dim}")
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ContractError("MRI response_state must be finite numeric values")
    end = timing_index(timing) + 1
    return values[:, :end, :].reshape(values.shape[0], end * values.shape[2]).copy()


def ftv_timing_prefix(
    ftv: pd.DataFrame | np.ndarray,
    timing: str | int,
    *,
    log1p: bool = True,
) -> np.ndarray:
    """Return only observed FTV values through timing, optionally log1p."""

    if isinstance(ftv, pd.DataFrame):
        missing = [column for column in FTV_WIDE_COLUMNS if column not in ftv.columns]
        if missing:
            raise ContractError(f"FTV wide table is missing columns: {missing}")
        values = ftv.loc[:, FTV_WIDE_COLUMNS].to_numpy(dtype=float)
    else:
        values = np.asarray(ftv)
        if values.ndim != 2 or values.shape[1] != len(VISITS):
            raise ContractError("FTV values must have shape [N,4]")
        if not np.issubdtype(values.dtype, np.number):
            raise ContractError("FTV values must be numeric")
        values = values.astype(float, copy=False)
    if values.ndim != 2 or values.shape[1] != len(VISITS):
        raise ContractError("FTV values must have shape [N,4]")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ContractError("FTV prefix values must be finite and non-negative")
    end = timing_index(timing) + 1
    prefix = values[:, :end].copy()
    return np.log1p(prefix) if log1p else prefix


mri_prefix = mri_timing_prefix
mri_prefix_for_timing = mri_timing_prefix
ftv_prefix = ftv_timing_prefix
ftv_prefix_for_timing = ftv_timing_prefix


__all__ = [
    "ALLOWED_CLINICAL_FIELDS",
    "AUDIT_ROOT",
    "CLINICAL_COLUMNS",
    "ContractError",
    "DEFAULT_CONFIG_PATH",
    "EXPECTED_SPLIT_COUNTS",
    "FOLD_COLUMNS",
    "FTV_COLUMNS",
    "FTV_WIDE_COLUMNS",
    "LocalFeatureAsset",
    "MISSING_CATEGORY",
    "REPO_ROOT",
    "TrainOnlyClinicalEncoder",
    "canonical_sha256",
    "file_sha256",
    "fit_clinical_encoder",
    "ftv_prefix",
    "ftv_prefix_for_timing",
    "ftv_timing_prefix",
    "load_all_local_feature_assets",
    "load_audit_config",
    "load_clinical_table",
    "load_config",
    "load_fold_manifest",
    "load_ftv_table",
    "load_ftv_wide",
    "load_local_feature_asset",
    "load_local_feature_cell",
    "mri_prefix",
    "mri_prefix_for_timing",
    "mri_timing_prefix",
    "ordered_patient_sha256",
    "reconstruct_ftv_wide",
    "require_file_sha256",
    "require_sha256",
    "timing_index",
]

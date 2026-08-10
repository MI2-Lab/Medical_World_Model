"""Leak-resistant Stage B tables, cohort construction, and DCE7 loading.

Every CSV call in this module names an exact ``usecols`` allow-list.  The
training dataset exposes only the image, FTV target/mask, and patient ID; no
clinical, treatment, support, affine, crop, or mask sidecar is loaded.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
import os
from pathlib import Path
import stat
from typing import BinaryIO, Iterable, Mapping
import zipfile

import numpy as np
import pandas as pd
try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # Schema/gate tests do not require the training runtime.
    torch = None  # type: ignore[assignment]

    class Dataset:  # type: ignore[no-redef]
        pass

from .contracts import (
    CACHE_MANIFEST_USECOLS,
    FOLD_USECOLS,
    FTV_TRANSITION_USECOLS,
    LOCKED_SEED_2026_FOLD_MANIFEST_SHA256,
    LOCKED_FTV_TRANSITION_TABLE_SHA256,
    LOCKED_C1B_CACHE_CONTRACT_SHA256,
    LOCKED_OBSERVABILITY_MANIFEST_SHA256,
    LOCKED_TRAIN_ONLY_SOURCE_MANIFEST_SHA256,
    OBSERVABILITY_USECOLS,
    PRIOR_C1B_SRC,
    TECHNICAL_ELIGIBILITY_USECOLS,
    TIMEPOINTS,
    TRAIN_ONLY_CANDIDATE_USECOLS,
    TRANSITIONS,
    arm_spec,
    file_sha256,
    require_sha256,
    validate_no_extra_columns,
)


def _read_allowlisted_csv(path: str | Path, usecols: tuple[str, ...], label: str) -> pd.DataFrame:
    frame = pd.read_csv(Path(path), usecols=list(usecols))
    validate_no_extra_columns(frame.columns, usecols, label)
    return frame


def _boolean(series: pd.Series, label: str) -> pd.Series:
    if series.dtype == bool:
        return series
    normalized = series.astype(str).str.strip().str.lower()
    allowed = {"true": True, "false": False, "1": True, "0": False}
    if not normalized.isin(allowed).all():
        bad = sorted(normalized.loc[~normalized.isin(allowed)].unique())[:5]
        raise ValueError(f"{label} contains non-boolean values: {bad}")
    return normalized.map(allowed).astype(bool)


def _verify_table_hash(path: str | Path, expected_sha256: str, label: str) -> Path:
    source = Path(path).expanduser().resolve()
    expected = require_sha256(expected_sha256, f"{label} SHA-256")
    actual = file_sha256(source)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, observed {actual}")
    return source


def read_fold_manifest(
    path: str | Path,
    expected_sha256: str,
    *,
    enforce_seed2026_lock: bool = True,
) -> pd.DataFrame:
    expected = require_sha256(expected_sha256, "fold manifest SHA-256")
    if enforce_seed2026_lock and expected != LOCKED_SEED_2026_FOLD_MANIFEST_SHA256:
        raise ValueError("fold manifest is not the frozen seed-2026 patient split")
    source = _verify_table_hash(path, expected, "fold manifest")
    frame = _read_allowlisted_csv(source, FOLD_USECOLS, "fold manifest")
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    frame["split"] = frame["split"].astype(str).str.lower()
    if set(frame["fold"].unique()) != set(range(5)):
        raise ValueError("fold manifest must contain exactly folds 0..4")
    if not set(frame["split"]).issubset({"train", "val", "test"}):
        raise ValueError("fold manifest contains an unknown split")
    patient_sets: list[set[str]] = []
    for fold in range(5):
        current = frame.loc[frame["fold"].eq(fold)]
        if current["patient_id"].duplicated().any():
            raise ValueError(f"fold {fold} contains duplicate patients")
        patient_sets.append(set(current["patient_id"]))
    if not patient_sets or any(values != patient_sets[0] for values in patient_sets[1:]):
        raise ValueError("each fold must cover the exact same primary patient set")
    if not patient_sets[0]:
        raise ValueError("fold manifest contains no patients")
    for fold in range(5):
        observed_splits = set(frame.loc[frame["fold"].eq(fold), "split"])
        if observed_splits != {"train", "val", "test"}:
            raise ValueError(f"fold {fold} must contain nonempty train/val/test splits")
    test_counts = frame.assign(_test=frame["split"].eq("test")).groupby("patient_id")["_test"].sum()
    if not test_counts.eq(1).all():
        raise ValueError("every primary patient must be test exactly once")
    return frame


@dataclass(frozen=True)
class TechnicalEligibilityPopulation:
    candidate_ids: tuple[str, ...]
    eligible_ids: tuple[str, ...]
    excluded_ids: tuple[str, ...]
    cohort_by_patient: Mapping[str, str]


def read_technical_eligibility(
    path: str | Path,
    expected_sha256: str,
) -> TechnicalEligibilityPopulation:
    source = _verify_table_hash(
        path, expected_sha256, "technical eligibility patient manifest"
    )
    if not source.name.endswith(".private.csv"):
        raise ValueError("technical eligibility must be read from a private manifest")
    frame = _read_allowlisted_csv(
        source, TECHNICAL_ELIGIBILITY_USECOLS, "technical eligibility"
    )
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["cohort"] = frame["cohort"].astype(str).str.strip()
    frame["eligible"] = _boolean(frame["eligible"], "technical eligibility")
    if frame.empty or frame["patient_id"].duplicated().any():
        raise ValueError("technical eligibility must uniquely cover a nonempty candidate population")
    if frame["patient_id"].eq("").any() or frame["cohort"].eq("").any():
        raise ValueError("technical eligibility patient/cohort fields must be nonempty")
    candidate_ids = tuple(frame["patient_id"])
    eligible_ids = tuple(frame.loc[frame["eligible"], "patient_id"])
    excluded_ids = tuple(frame.loc[~frame["eligible"], "patient_id"])
    if not eligible_ids:
        raise ValueError("technical eligibility produced an empty eligible population")
    return TechnicalEligibilityPopulation(
        candidate_ids,
        eligible_ids,
        excluded_ids,
        dict(zip(frame["patient_id"], frame["cohort"], strict=True)),
    )


def read_train_only_candidates(
    path: str | Path,
    expected_sha256: str,
    *,
    enforce_upstream_lock: bool = True,
) -> tuple[str, ...]:
    """Read the upstream source-qualified train-only cohort without fixed counts."""

    expected = require_sha256(expected_sha256, "train-only candidate manifest SHA-256")
    if enforce_upstream_lock and expected != LOCKED_TRAIN_ONLY_SOURCE_MANIFEST_SHA256:
        raise ValueError("train-only candidates are not the frozen upstream source cohort")
    source = _verify_table_hash(path, expected, "train-only candidate manifest")
    if not source.name.endswith(".private.csv"):
        raise ValueError("train-only candidates must be read from a private manifest")
    frame = _read_allowlisted_csv(
        source, TRAIN_ONLY_CANDIDATE_USECOLS, "train-only candidates"
    )
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["eligible"] = _boolean(frame["eligible"], "upstream train-only source eligibility")
    if frame.empty or frame["patient_id"].duplicated().any():
        raise ValueError("train-only candidate manifest must be nonempty and unique")
    return tuple(frame.loc[frame["eligible"], "patient_id"])


def intersect_eligible_folds(
    folds: pd.DataFrame,
    eligible_ids: Iterable[str],
) -> pd.DataFrame:
    """Mechanically intersect the locked seed-2026 rows; never refill a split."""

    eligible = set(str(value) for value in eligible_ids)
    filtered = folds.loc[folds["patient_id"].astype(str).isin(eligible)].copy()
    if filtered.empty:
        raise ValueError("technical eligibility has no overlap with the locked fold manifest")
    patient_sets = [
        set(filtered.loc[filtered["fold"].eq(fold), "patient_id"].astype(str))
        for fold in range(5)
    ]
    if any(values != patient_sets[0] for values in patient_sets[1:]):
        raise AssertionError("eligibility intersection changed patient membership across folds")
    test_counts = (
        filtered.assign(_test=filtered["split"].eq("test"))
        .groupby("patient_id")["_test"]
        .sum()
    )
    if not test_counts.eq(1).all():
        raise ValueError("each eligible fold patient must remain test exactly once")
    return filtered.reset_index(drop=True)


@dataclass(frozen=True)
class MatchedStageBPopulation:
    folds: pd.DataFrame
    fold_candidate_ids: frozenset[str]
    fold_eligible_ids: frozenset[str]
    train_only_ids: tuple[str, ...]
    matched_patient_ids: tuple[str, ...]


def derive_matched_stage_b_population(
    folds_all: pd.DataFrame,
    eligibility: TechnicalEligibilityPopulation,
    upstream_train_only_ids: Iterable[str],
) -> MatchedStageBPopulation:
    """Apply eligibility to frozen folds and classify fold-external train-only IDs."""

    fold_candidates = frozenset(folds_all["patient_id"].astype(str))
    candidate_ids = set(eligibility.candidate_ids)
    eligible_ids = set(eligibility.eligible_ids)
    if not fold_candidates.issubset(candidate_ids):
        raise ValueError("locked fold manifest contains patients outside Stage A candidates")
    upstream_train_only = set(str(value) for value in upstream_train_only_ids)
    if upstream_train_only & fold_candidates:
        raise ValueError("upstream train-only cohort overlaps the locked fold cohort")
    if not upstream_train_only.issubset(candidate_ids):
        raise ValueError(
            "upstream source-qualified train-only IDs are outside Stage A candidates"
        )
    unassigned_eligible = eligible_ids.difference(fold_candidates)
    if not unassigned_eligible.issubset(upstream_train_only):
        raise ValueError(
            "an eligible patient outside seed-2026 folds lacks upstream train-only authorization"
        )
    train_only_ids = tuple(
        patient_id
        for patient_id in eligibility.eligible_ids
        if patient_id in unassigned_eligible
    )
    folds = intersect_eligible_folds(folds_all, eligibility.eligible_ids)
    fold_eligible = frozenset(folds["patient_id"].astype(str))
    required = fold_eligible | set(train_only_ids)
    if required != eligible_ids:
        raise AssertionError("Stage B matched cohort does not exactly equal Stage A eligibility")
    matched_patient_ids = tuple(
        patient_id for patient_id in eligibility.eligible_ids if patient_id in required
    )
    return MatchedStageBPopulation(
        folds,
        fold_candidates,
        fold_eligible,
        train_only_ids,
        matched_patient_ids,
    )


@dataclass(frozen=True)
class CacheEntry:
    patient_id: str
    path: Path
    sha256: str
    size_bytes: int
    mtime_ns: int
    input_kind: str


LEGACY_NPZ_MEMBERS = frozenset({"x.npy"})
C1B_IMAGE_SHAPE = (4, 7, 112, 176, 160)
_TINY_NPY_MEMBER_LIMIT = 4096
_NPY_HEADER_LIMIT = 4096


def _sha256_stream(stream: BinaryIO) -> str:
    stream.seek(0)
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def _assert_stable_stat(before: os.stat_result, after: os.stat_result, path: Path) -> None:
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in fields):
        raise ValueError(f"cache changed while it was being verified: {path}")


def _assert_pinned_stat(entry: CacheEntry, observed: os.stat_result) -> None:
    if not stat.S_ISREG(observed.st_mode):
        raise ValueError(f"cache is not a regular file: {entry.path}")
    if (
        int(observed.st_size) != entry.size_bytes
        or int(observed.st_mtime_ns) != entry.mtime_ns
    ):
        raise ValueError(
            "cache stat no longer matches the SHA-pinned manifest for "
            f"{entry.patient_id}: expected size/mtime_ns="
            f"{entry.size_bytes}/{entry.mtime_ns}, observed="
            f"{observed.st_size}/{observed.st_mtime_ns}"
        )


@lru_cache(maxsize=1)
def _expected_c1b_members() -> frozenset[str]:
    # The exact member contract remains owned by the schema-3 Stage A
    # validator. Parse that hash-recorded source declaration without importing
    # imaging dependencies into a Stage-B data worker.
    source = (PRIOR_C1B_SRC / "c1b_sanity" / "cache.py").resolve()
    if file_sha256(source) != LOCKED_C1B_CACHE_CONTRACT_SHA256:
        raise ValueError("frozen C1B schema-3 cache contract hash drifted")
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    required_keys: object | None = None
    string_constants: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        if len(names) == 1 and names[0] != "_REQUIRED_KEYS":
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                value = None
            if isinstance(value, str):
                string_constants[names[0]] = value
        if "_REQUIRED_KEYS" not in names:
            continue
        if (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "frozenset"
            and len(node.value.args) == 1
            and isinstance(node.value.args[0], ast.Set)
        ):
            values: set[str] = set()
            for element in node.value.args[0].elts:
                if isinstance(element, ast.Constant) and isinstance(element.value, str):
                    values.add(element.value)
                elif isinstance(element, ast.Name) and element.id in string_constants:
                    values.add(string_constants[element.id])
                else:
                    raise ValueError("frozen C1B schema uses a nonliteral member key")
            required_keys = values
            break
    if not isinstance(required_keys, set) or not required_keys:
        raise ValueError("cannot recover frozen C1B schema member contract")

    return frozenset(f"{key}.npy" for key in required_keys)


def _validate_c1b_filename(path: Path, patient_id: str) -> None:
    expected = hashlib.sha256(patient_id.encode("utf-8")).hexdigest() + ".npz"
    if path.name != expected:
        raise ValueError(
            f"C1B cache filename is not bound to patient {patient_id!r}: "
            f"expected {expected}, got {path.name}"
        )


def _validate_member_names(
    archive: zipfile.ZipFile, expected: frozenset[str], label: str
) -> None:
    members = archive.infolist()
    names = [member.filename for member in members]
    if len(names) != len(set(names)):
        raise ValueError(f"{label} cache contains duplicate NPZ member names")
    if set(names) != expected:
        missing = sorted(expected.difference(names))
        unexpected = sorted(set(names).difference(expected))
        raise ValueError(
            f"{label} cache NPZ members drifted: missing={missing}, "
            f"unexpected={unexpected}"
        )
    if any(member.compress_type != zipfile.ZIP_STORED for member in members):
        raise ValueError(f"{label} cache must use the frozen uncompressed NPZ layout")
    if any(member.flag_bits & 0x1 for member in members):
        raise ValueError(f"{label} cache must not contain encrypted NPZ members")


def _read_npy_member(
    archive: zipfile.ZipFile,
    name: str,
    *,
    expected_shape: tuple[int, ...],
    expected_dtype: np.dtype | None = None,
    expected_dtype_kind: str | None = None,
    max_member_size: int,
) -> np.ndarray:
    """Read one NPY member only after a bounded header/byte-layout preflight."""

    info = archive.getinfo(name)
    if info.file_size > int(max_member_size):
        raise ValueError(
            f"NPZ member {name!r} exceeds its frozen size bound: {info.file_size}"
        )
    with archive.open(info, mode="r") as member:
        version = np.lib.format.read_magic(member)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                member, max_header_size=_NPY_HEADER_LIMIT
            )
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(
                member, max_header_size=_NPY_HEADER_LIMIT
            )
        else:
            raise ValueError(f"NPZ member {name!r} uses unsupported NPY version {version}")
        dtype = np.dtype(dtype)
        shape = tuple(int(value) for value in shape)
        if dtype.hasobject:
            raise ValueError(f"NPZ member {name!r} must not contain an object array")
        if fortran_order:
            raise ValueError(f"NPZ member {name!r} must use C order")
        if shape != expected_shape:
            raise ValueError(
                f"NPZ member {name!r} has shape {shape}, expected {expected_shape}"
            )
        if expected_dtype is not None and dtype != np.dtype(expected_dtype):
            raise ValueError(
                f"NPZ member {name!r} has dtype {dtype}, expected {np.dtype(expected_dtype)}"
            )
        if expected_dtype_kind is not None and dtype.kind not in expected_dtype_kind:
            raise ValueError(
                f"NPZ member {name!r} has forbidden dtype {dtype}"
            )
        data_bytes = math.prod(shape) * dtype.itemsize if shape else dtype.itemsize
        remaining = int(info.file_size) - int(member.tell())
        if data_bytes != remaining:
            raise ValueError(
                f"NPZ member {name!r} header/data byte count is inconsistent: "
                f"declared={data_bytes}, stored={remaining}"
            )
        array = np.empty(shape, dtype=dtype, order="C")
        target = memoryview(array).cast("B")
        offset = 0
        while offset < len(target):
            count = member.readinto(target[offset:])
            if not count:
                raise ValueError(f"NPZ member {name!r} ended before its declared data")
            offset += int(count)
        if member.read(1):
            raise ValueError(f"NPZ member {name!r} contains trailing array data")
    return array


def _validate_c1b_identity(archive: zipfile.ZipFile, patient_id: str) -> None:
    _validate_member_names(archive, _expected_c1b_members(), "C1B schema-3")
    schema_version = _read_npy_member(
        archive,
        "schema_version.npy",
        expected_shape=(),
        expected_dtype_kind="iu",
        max_member_size=_TINY_NPY_MEMBER_LIMIT,
    )
    if (
        schema_version.shape != ()
        or schema_version.dtype.kind not in "iu"
        or int(schema_version.item()) != 3
    ):
        raise ValueError("C1B cache schema_version must be an integer scalar 3")
    embedded = _read_npy_member(
        archive,
        "patient_id.npy",
        expected_shape=(),
        expected_dtype_kind="U",
        max_member_size=_TINY_NPY_MEMBER_LIMIT,
    )
    if str(embedded.item()) != patient_id:
        raise ValueError(
            f"C1B embedded patient identity mismatch: manifest={patient_id!r}, "
            f"cache={str(embedded.item())!r}"
        )


def fingerprint_cache_file(
    path: str | Path,
    patient_id: str,
    input_kind: str,
    *,
    expected_sha256: str | None = None,
) -> CacheEntry:
    """Hash one cache once, pin stable stat metadata, and inspect its envelope.

    This is the cache-manifest creation path.  For schema-3 C1B it reads only
    the ZIP central directory plus the tiny schema/identity members; the full
    Stage A validation remains represented by the pinned cache-file digest.
    """

    source = Path(path).expanduser().resolve()
    kind = str(input_kind).lower()
    if kind not in {"legacy", "c1b"}:
        raise ValueError(f"unknown cache input kind: {input_kind}")
    if kind == "c1b" and expected_sha256 is None:
        raise ValueError("C1B manifest creation requires the trusted Stage A file SHA-256")
    identity = str(patient_id)
    if not identity:
        raise ValueError("cache manifest patient_id must be nonempty")
    with source.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"cache is not a regular file: {source}")
        digest = _sha256_stream(stream)
        after = os.fstat(stream.fileno())
        _assert_stable_stat(before, after, source)
        if expected_sha256 is not None:
            expected = require_sha256(expected_sha256, f"cache {identity}")
            if digest != expected:
                raise ValueError(f"cache SHA-256 mismatch for {identity}")
        entry = CacheEntry(
            identity,
            source,
            digest,
            int(after.st_size),
            int(after.st_mtime_ns),
            kind,
        )
        if kind == "c1b":
            _validate_c1b_filename(source, identity)
        stream.seek(0)
        with zipfile.ZipFile(stream, mode="r") as archive:
            if kind == "legacy":
                _validate_member_names(archive, LEGACY_NPZ_MEMBERS, "legacy")
            else:
                _validate_c1b_identity(archive, identity)
        final = os.fstat(stream.fileno())
        _assert_stable_stat(before, final, source)
        _assert_pinned_stat(entry, final)
    return entry


def verify_cache_entry(entry: CacheEntry, *, verify_sha256: bool) -> None:
    """Verify a manifest entry without materializing any image or sidecar."""

    with entry.path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        _assert_pinned_stat(entry, before)
        if entry.input_kind == "c1b":
            _validate_c1b_filename(entry.path, entry.patient_id)
        if verify_sha256:
            observed = _sha256_stream(stream)
            after = os.fstat(stream.fileno())
            _assert_stable_stat(before, after, entry.path)
            _assert_pinned_stat(entry, after)
            if observed != entry.sha256:
                raise ValueError(f"cache SHA-256 mismatch for {entry.patient_id}")
        stream.seek(0)
        with zipfile.ZipFile(stream, mode="r") as archive:
            if entry.input_kind == "legacy":
                _validate_member_names(archive, LEGACY_NPZ_MEMBERS, "legacy")
            elif entry.input_kind == "c1b":
                _validate_c1b_identity(archive, entry.patient_id)
            else:
                raise ValueError(f"unknown cache input kind: {entry.input_kind}")
        final = os.fstat(stream.fileno())
        _assert_stable_stat(before, final, entry.path)
        _assert_pinned_stat(entry, final)


def read_cache_manifest(
    path: str | Path,
    expected_sha256: str,
    *,
    expected_input_kind: str,
    verify_cache_files: bool = True,
) -> dict[str, CacheEntry]:
    source = _verify_table_hash(path, expected_sha256, f"{expected_input_kind} cache manifest")
    if not source.name.endswith(".private.csv"):
        raise ValueError("cache inventory must be a private manifest ending in .private.csv")
    frame = _read_allowlisted_csv(source, CACHE_MANIFEST_USECOLS, "cache manifest")
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["input_kind"] = frame["input_kind"].astype(str).str.lower()
    for column in ("cache_size_bytes", "cache_mtime_ns"):
        numeric = pd.to_numeric(frame[column], errors="raise")
        if numeric.isna().any() or any(
            isinstance(value, (float, np.floating)) and not float(value).is_integer()
            for value in numeric
        ):
            raise ValueError(f"cache manifest {column} must contain exact integers")
        frame[column] = numeric.astype(np.int64)
    if frame["cache_size_bytes"].le(0).any() or frame["cache_mtime_ns"].lt(0).any():
        raise ValueError("cache manifest size must be positive and mtime_ns nonnegative")
    if frame["patient_id"].duplicated().any() or frame["cache_path"].astype(str).duplicated().any():
        raise ValueError("cache manifest contains duplicate patients or paths")
    if set(frame["input_kind"]) != {expected_input_kind}:
        raise ValueError(f"cache manifest must contain only input_kind={expected_input_kind!r}")
    entries: dict[str, CacheEntry] = {}
    for row in frame.itertuples(index=False):
        cache_path = Path(str(row.cache_path)).expanduser()
        if not cache_path.is_absolute():
            cache_path = (source.parent / cache_path).resolve()
        else:
            cache_path = cache_path.resolve()
        digest = require_sha256(str(row.cache_sha256), f"cache {row.patient_id}")
        patient_id = str(row.patient_id)
        entry = CacheEntry(
            patient_id,
            cache_path,
            digest,
            int(row.cache_size_bytes),
            int(row.cache_mtime_ns),
            expected_input_kind,
        )
        if verify_cache_files:
            verify_cache_entry(entry, verify_sha256=True)
        entries[patient_id] = entry
    if not entries:
        raise ValueError("cache manifest is empty")
    return entries


@dataclass(frozen=True)
class FTVRecord:
    values: np.ndarray
    measurement_valid: np.ndarray
    observable: np.ndarray

    @property
    def grounding_eligible(self) -> np.ndarray:
        return np.asarray(self.measurement_valid & self.observable, dtype=bool)


def read_raw_ftv(
    path: str | Path,
    expected_sha256: str,
    *,
    enforce_frozen_target: bool = True,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    expected = require_sha256(expected_sha256, "FTV transition table SHA-256")
    if enforce_frozen_target and expected != LOCKED_FTV_TRANSITION_TABLE_SHA256:
        raise ValueError("FTV transition table is not the frozen formal target asset")
    source = _verify_table_hash(path, expected, "FTV transition table")
    frame = _read_allowlisted_csv(source, FTV_TRANSITION_USECOLS, "FTV transitions")
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["transition"] = frame["transition"].astype(str)
    frame["ftv_valid"] = _boolean(frame["ftv_valid"], "FTV validity")
    output: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for patient_id, rows in frame.groupby("patient_id", sort=False):
        if rows["transition"].duplicated().any() or set(rows["transition"]) != set(TRANSITIONS):
            raise ValueError(f"{patient_id} does not contain exactly three adjacent transitions")
        rows = rows.set_index("transition").reindex(TRANSITIONS)
        for index, transition in enumerate(TRANSITIONS):
            expected_start, expected_end = transition.split("→")
            row = rows.loc[transition]
            if str(row["start_visit"]) != expected_start or str(row["end_visit"]) != expected_end:
                raise ValueError(f"{patient_id}/{transition} visit alignment mismatch")
            if index < 2 and bool(row["ftv_valid"]) and bool(rows.iloc[index + 1]["ftv_valid"]):
                if not np.isclose(
                    float(row["ftv_end"]), float(rows.iloc[index + 1]["ftv_start"]),
                    rtol=0.0, atol=1e-10,
                ):
                    raise ValueError(f"{patient_id}/{expected_end} shared FTV endpoint mismatch")
        values = np.asarray(
            [rows.iloc[0]["ftv_start"], rows.iloc[0]["ftv_end"], rows.iloc[1]["ftv_end"], rows.iloc[2]["ftv_end"]],
            dtype=np.float64,
        )
        transition_valid = rows["ftv_valid"].to_numpy(dtype=bool)
        valid = np.asarray(
            [transition_valid[0], transition_valid[0] and transition_valid[1],
             transition_valid[1] and transition_valid[2], transition_valid[2]],
            dtype=bool,
        )
        valid &= np.isfinite(values)
        if np.any(values[valid] < 0):
            raise ValueError("FTV values must be nonnegative")
        output[str(patient_id)] = (values, valid)
    return output


def read_observability(
    path: str | Path,
    expected_sha256: str,
    *,
    enforce_frozen_manifest: bool = True,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    expected = require_sha256(expected_sha256, "grounding observability SHA-256")
    if enforce_frozen_manifest and expected != LOCKED_OBSERVABILITY_MANIFEST_SHA256:
        raise ValueError("grounding observability is not the frozen loss-side manifest")
    source = _verify_table_hash(path, expected, "grounding observability manifest")
    if not source.name.endswith(".private.csv"):
        raise ValueError("grounding observability must be read from a private manifest")
    frame = _read_allowlisted_csv(source, OBSERVABILITY_USECOLS, "grounding observability")
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["visit"] = frame["visit"].astype(str)
    frame["ftv_measurement_valid"] = _boolean(
        frame["ftv_measurement_valid"], "FTV measurement validity"
    )
    frame["grounding_observable_mask"] = _boolean(
        frame["grounding_observable_mask"], "grounding observability"
    )
    output: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for patient_id, rows in frame.groupby("patient_id", sort=False):
        if rows["visit"].duplicated().any() or set(rows["visit"]) != set(TIMEPOINTS):
            raise ValueError(f"{patient_id} observability does not cover T0..T3 exactly")
        rows = rows.set_index("visit").reindex(TIMEPOINTS)
        measurement = rows["ftv_measurement_valid"].to_numpy(dtype=bool)
        observable = rows["grounding_observable_mask"].to_numpy(dtype=bool)
        if np.any(observable & ~measurement):
            raise ValueError("grounding observability cannot be true without a valid FTV measurement")
        output[str(patient_id)] = (measurement, observable)
    return output


def combine_ftv_observability(
    raw_ftv: Mapping[str, tuple[np.ndarray, np.ndarray]],
    observability: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, FTVRecord]:
    if set(raw_ftv) != set(observability):
        raise ValueError("FTV targets and observability must cover the exact same patients")
    output: dict[str, FTVRecord] = {}
    for patient_id in raw_ftv:
        values, target_valid = raw_ftv[patient_id]
        measurement_valid, observable = observability[patient_id]
        values = np.asarray(values, dtype=np.float64)
        target_valid = np.asarray(target_valid, dtype=bool)
        measurement_valid = np.asarray(measurement_valid, dtype=bool)
        observable = np.asarray(observable, dtype=bool)
        if any(array.shape != (4,) for array in (values, target_valid, measurement_valid, observable)):
            raise ValueError(f"{patient_id} FTV/observability shape must be [4]")
        if not np.array_equal(target_valid, measurement_valid & np.isfinite(values)):
            raise ValueError(f"{patient_id} target and measurement validity disagree")
        output[patient_id] = FTVRecord(values, measurement_valid, observable)
    return output


@dataclass(frozen=True)
class StageBSplits:
    train_fold: tuple[str, ...]
    train_all: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]
    train_only: tuple[str, ...]

    @property
    def train_primary(self) -> tuple[str, ...]:
        """Compatibility alias for probe/transform code: fold-assigned train only."""

        return self.train_fold


def make_splits(
    folds: pd.DataFrame, fold: int, train_only_ids: Iterable[str]
) -> StageBSplits:
    if int(fold) not in range(5):
        raise ValueError("fold must be 0..4")
    current = folds.loc[folds["fold"].eq(int(fold))]
    primary = {
        name: tuple(current.loc[current["split"].eq(name), "patient_id"].astype(str))
        for name in ("train", "val", "test")
    }
    extras = tuple(str(value) for value in train_only_ids)
    if len(set(extras)) != len(extras):
        raise ValueError("train-only patient IDs must be unique")
    fold_union = set(primary["train"]) | set(primary["val"]) | set(primary["test"])
    if set(extras) & fold_union:
        raise ValueError("train-only patients overlap the locked fold cohort")
    if not all(primary.values()):
        raise ValueError("eligible fold intersection must retain train/val/test rows")
    return StageBSplits(
        train_fold=primary["train"],
        train_all=primary["train"] + extras,
        val=primary["val"],
        test=primary["test"],
        train_only=extras,
    )


def validate_cache_coverage(
    legacy: Mapping[str, CacheEntry],
    c1b: Mapping[str, CacheEntry],
    required_patient_ids: Iterable[str],
) -> None:
    required = set(str(value) for value in required_patient_ids)
    for name, cache in (("legacy", legacy), ("c1b", c1b)):
        missing = sorted(required.difference(cache))
        if missing:
            raise FileNotFoundError(f"{name} cache manifest misses required patients: {missing[:5]}")


def load_dce7(entry: CacheEntry) -> np.ndarray:
    # Bind the cheap stat check and all selected member reads to the same open
    # file descriptor.  No worker computes a file/content hash or materializes
    # any schema-3 support, mask, affine, provenance, or normalization sidecar.
    with entry.path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        _assert_pinned_stat(entry, before)
        if entry.input_kind == "c1b":
            _validate_c1b_filename(entry.path, entry.patient_id)
        with zipfile.ZipFile(stream, mode="r") as archive:
            if entry.input_kind == "legacy":
                _validate_member_names(archive, LEGACY_NPZ_MEMBERS, "legacy")
                legacy_shape = (4, 8, 32, 96, 96)
                cached = _read_npy_member(
                    archive,
                    "x.npy",
                    expected_shape=legacy_shape,
                    expected_dtype=np.dtype(np.float32),
                    max_member_size=math.prod(legacy_shape) * 4 + _NPY_HEADER_LIMIT,
                )
                if cached.dtype != np.dtype(np.float32) or cached.shape != (
                    4,
                    8,
                    32,
                    96,
                    96,
                ):
                    raise ValueError(
                        "legacy cache must be float32 [4,8,32,96,96], "
                        f"got {cached.dtype}/{cached.shape}"
                    )
                image = np.ascontiguousarray(cached[:, :7], dtype=np.float32)
            elif entry.input_kind == "c1b":
                _validate_c1b_identity(archive, entry.patient_id)
                image = _read_npy_member(
                    archive,
                    "image.npy",
                    expected_shape=C1B_IMAGE_SHAPE,
                    expected_dtype=np.dtype(np.float32),
                    max_member_size=(
                        math.prod(C1B_IMAGE_SHAPE) * np.dtype(np.float32).itemsize
                        + _NPY_HEADER_LIMIT
                    ),
                )
                if image.dtype != np.dtype(np.float32) or image.shape != C1B_IMAGE_SHAPE:
                    raise ValueError(
                        f"C1B cache must be float32 {list(C1B_IMAGE_SHAPE)}, "
                        f"got {image.dtype}/{image.shape}"
                    )
                image = np.ascontiguousarray(image, dtype=np.float32)
            else:
                raise ValueError(f"unknown cache input kind: {entry.input_kind}")
        after = os.fstat(stream.fileno())
        _assert_stable_stat(before, after, entry.path)
        _assert_pinned_stat(entry, after)
    if image.dtype != np.float32 or not np.isfinite(image).all():
        raise ValueError("model tensor must be finite float32 DCE7")
    if entry.input_kind == "c1b" and (
        np.any(image < -5.000001) or np.any(image > 5.000001)
    ):
        raise ValueError("C1B model tensor violates the frozen [-5,5] output clip")
    return image


class StageBDataset(Dataset):
    """Patient-indexed image/FTV dataset; sidecars are inaccessible by design."""

    def __init__(
        self,
        patient_ids: Iterable[str],
        cache: Mapping[str, CacheEntry],
        transformed_ftv: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> None:
        self.patient_ids = tuple(str(value) for value in patient_ids)
        self.cache = cache
        self.transformed_ftv = transformed_ftv or {}
        if len(set(self.patient_ids)) != len(self.patient_ids):
            raise ValueError("dataset patient IDs must be unique")
        if missing := sorted(set(self.patient_ids).difference(cache)):
            raise KeyError(f"dataset cache misses patients: {missing[:5]}")

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, index: int) -> dict[str, object]:
        if torch is None:
            raise RuntimeError("PyTorch is required to materialize Stage B model batches")
        patient_id = self.patient_ids[index]
        image = load_dce7(self.cache[patient_id])
        target, mask = self.transformed_ftv.get(
            patient_id, (np.zeros(4, dtype=np.float32), np.zeros(4, dtype=bool))
        )
        target = np.asarray(target, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)
        if target.shape != (4,) or mask.shape != (4,):
            raise ValueError("transformed FTV target and mask must be [4]")
        return {
            "patient_id": patient_id,
            "image": torch.from_numpy(image),
            "ftv_target": torch.from_numpy(target),
            "ftv_mask": torch.from_numpy(mask),
        }


def arm_cache(arm: str, legacy: Mapping[str, CacheEntry], c1b: Mapping[str, CacheEntry]) -> Mapping[str, CacheEntry]:
    return legacy if arm_spec(arm).input_kind == "legacy" else c1b


__all__ = [
    "CacheEntry",
    "FTVRecord",
    "MatchedStageBPopulation",
    "StageBDataset",
    "StageBSplits",
    "TechnicalEligibilityPopulation",
    "arm_cache",
    "combine_ftv_observability",
    "derive_matched_stage_b_population",
    "fingerprint_cache_file",
    "intersect_eligible_folds",
    "load_dce7",
    "make_splits",
    "read_cache_manifest",
    "read_fold_manifest",
    "read_observability",
    "read_raw_ftv",
    "read_technical_eligibility",
    "read_train_only_candidates",
    "validate_cache_coverage",
    "verify_cache_entry",
]

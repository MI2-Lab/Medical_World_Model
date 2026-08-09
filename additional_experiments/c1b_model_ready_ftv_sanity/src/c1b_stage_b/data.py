"""Leak-resistant Stage B tables, cohort construction, and DCE7 loading.

Every CSV call in this module names an exact ``usecols`` allow-list.  The
training dataset exposes only the image, FTV target/mask, and patient ID; no
clinical, treatment, support, affine, crop, or mask sidecar is loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    ISPY1_CANDIDATE_COUNT,
    ISPY1_ELIGIBLE_COUNT,
    ISPY1_ELIGIBILITY_USECOLS,
    OBSERVABILITY_USECOLS,
    PRIMARY_PATIENT_COUNT,
    TIMEPOINTS,
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
    expected_patient_count: int = PRIMARY_PATIENT_COUNT,
) -> pd.DataFrame:
    source = _verify_table_hash(path, expected_sha256, "fold manifest")
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
    if len(patient_sets[0]) != int(expected_patient_count):
        raise ValueError(
            f"primary cohort must contain {expected_patient_count} patients, got {len(patient_sets[0])}"
        )
    test_counts = frame.assign(_test=frame["split"].eq("test")).groupby("patient_id")["_test"].sum()
    if not test_counts.eq(1).all():
        raise ValueError("every primary patient must be test exactly once")
    return frame


def read_ispy1_eligibility(
    path: str | Path,
    expected_sha256: str,
    *,
    expected_candidate_count: int = ISPY1_CANDIDATE_COUNT,
    expected_eligible_count: int = ISPY1_ELIGIBLE_COUNT,
) -> tuple[str, ...]:
    source = _verify_table_hash(path, expected_sha256, "I-SPY1 eligibility manifest")
    frame = _read_allowlisted_csv(source, ISPY1_ELIGIBILITY_USECOLS, "I-SPY1 eligibility")
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["eligible"] = _boolean(frame["eligible"], "I-SPY1 eligible")
    if frame["patient_id"].duplicated().any() or len(frame) != int(expected_candidate_count):
        raise ValueError(
            "I-SPY1 eligibility must uniquely cover exactly "
            f"{expected_candidate_count} candidate patients"
        )
    eligible = tuple(frame.loc[frame["eligible"], "patient_id"])
    if len(eligible) != int(expected_eligible_count):
        raise ValueError(
            "I-SPY1 eligibility must contain exactly "
            f"{expected_eligible_count} eligible patients, got {len(eligible)}"
        )
    return eligible


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


def _expected_c1b_members() -> frozenset[str]:
    # The exact member contract remains owned by the schema-3 Stage A
    # validator.  Importing it here avoids a second, drifting schema copy.
    from c1b_sanity.cache import _REQUIRED_KEYS

    return frozenset(f"{key}.npy" for key in _REQUIRED_KEYS)


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


def read_raw_ftv(path: str | Path, expected_sha256: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    source = _verify_table_hash(path, expected_sha256, "FTV transition table")
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
    path: str | Path, expected_sha256: str
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    source = _verify_table_hash(path, expected_sha256, "grounding observability manifest")
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
    train_primary: tuple[str, ...]
    train_all: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]


def make_splits(folds: pd.DataFrame, fold: int, eligible_ispy1: Iterable[str]) -> StageBSplits:
    if int(fold) not in range(5):
        raise ValueError("fold must be 0..4")
    current = folds.loc[folds["fold"].eq(int(fold))]
    primary = {
        name: tuple(current.loc[current["split"].eq(name), "patient_id"].astype(str))
        for name in ("train", "val", "test")
    }
    extras = tuple(str(value) for value in eligible_ispy1)
    primary_union = set(primary["train"]) | set(primary["val"]) | set(primary["test"])
    if set(extras) & primary_union:
        raise ValueError("I-SPY1 base-only patients overlap the primary cohort")
    return StageBSplits(
        train_primary=primary["train"],
        train_all=primary["train"] + extras,
        val=primary["val"],
        test=primary["test"],
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
    "StageBDataset",
    "StageBSplits",
    "arm_cache",
    "combine_ftv_observability",
    "fingerprint_cache_file",
    "load_dce7",
    "make_splits",
    "read_cache_manifest",
    "read_fold_manifest",
    "read_ispy1_eligibility",
    "read_observability",
    "read_raw_ftv",
    "validate_cache_coverage",
    "verify_cache_entry",
]

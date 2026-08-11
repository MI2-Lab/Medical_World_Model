#!/usr/bin/env python3
"""One-time content-integrity preflight for all hash-bound C1B caches.

The proof covers the 808 locked primary patients plus the 139 manifest-only
patients needed by the canonical Stage-B ``train_all`` training population.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import platform
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from common import (  # noqa: E402
    DEFAULT_CONFIG,
    atomic_json,
    canonical_sha256,
    file_sha256,
    load_config,
    ordered_sha256,
    private_directory,
    require_preregistration_lock,
)


PRIVATE_MANIFEST = ROOT / "manifests" / "cache_integrity.private.json"
PUBLIC_CONTRACT = ROOT / "metrics" / "cache_integrity_contract.json"
PREREGISTRATION_LOCK = ROOT / "PREREGISTRATION_LOCK.json"
IMPLEMENTATION_KEY = "scripts/verify_cache_integrity.py"
EXPECTED_CELL_COUNT = 20
EXPECTED_PRIMARY_PATIENT_COUNT = 808
EXPECTED_TRAIN_ONLY_PATIENT_COUNT = 139
EXPECTED_PATIENT_COUNT = 947
PURPOSE = "one_time_all_hash_bound_c1b_cache_content_integrity_preflight"
MANIFEST_COLUMNS = (
    "patient_id",
    "cache_path",
    "cache_sha256",
    "cache_size_bytes",
    "cache_mtime_ns",
    "input_kind",
)
PRIVATE_RECORD_COLUMNS = (
    "patient_id",
    "path",
    "sha256",
    "size_bytes",
    "mtime_ns",
    "cohort",
)
PRIVATE_COLUMNS = (
    "schema_version",
    "status",
    "patient_count",
    "primary_patient_count",
    "train_only_patient_count",
    "records",
)
PUBLIC_COLUMNS = (
    "schema_version",
    "status",
    "purpose",
    "patient_count",
    "primary_patient_count",
    "train_only_patient_count",
    "total_bytes",
    "upstream_manifest_sha256",
    "private_artifact_sha256",
    "canonical_record_set_sha256",
    "primary_record_set_sha256",
    "train_only_record_set_sha256",
    "preregistration_lock_sha256",
    "implementation_sha256",
    "environment",
    "contains_patient_identifiers",
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class CacheIntegrityError(RuntimeError):
    """Raised when a frozen cache-integrity contract cannot be authenticated."""


def _require_sha256(value: Any, label: str) -> str:
    digest = str(value)
    if SHA256_PATTERN.fullmatch(digest) is None:
        raise CacheIntegrityError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise CacheIntegrityError(f"{label} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise CacheIntegrityError(f"{label} must be a positive integer") from error
    if number <= 0 or isinstance(value, float) and not value.is_integer():
        raise CacheIntegrityError(f"{label} must be a positive integer")
    return number


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise CacheIntegrityError(f"{label} must be a nonnegative integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise CacheIntegrityError(f"{label} must be a nonnegative integer") from error
    if number < 0 or isinstance(value, float) and not value.is_integer():
        raise CacheIntegrityError(f"{label} must be a nonnegative integer")
    return number


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CacheIntegrityError(f"{label} is absent or unreadable: {path}") from error
    if not isinstance(value, dict):
        raise CacheIntegrityError(f"{label} must be a JSON object")
    return value


def _current_lock_context(
    config: Mapping[str, Any], lock: Mapping[str, Any]
) -> tuple[str, str]:
    """Authenticate the supplied lock against the current formal lock and this script."""

    current = require_preregistration_lock(config)
    if canonical_sha256(current) != canonical_sha256(lock):
        raise CacheIntegrityError(
            "supplied preregistration lock is not the current lock"
        )
    on_disk = _read_json(PREREGISTRATION_LOCK, "preregistration lock")
    if canonical_sha256(on_disk) != canonical_sha256(current):
        raise CacheIntegrityError(
            "current preregistration lock changed after validation"
        )
    lock_sha256 = file_sha256(PREREGISTRATION_LOCK)

    inventory = lock.get("implementation_sha256")
    if not isinstance(inventory, Mapping) or IMPLEMENTATION_KEY not in inventory:
        raise CacheIntegrityError(
            f"preregistration lock does not bind {IMPLEMENTATION_KEY}"
        )
    expected = _require_sha256(
        inventory[IMPLEMENTATION_KEY], f"locked {IMPLEMENTATION_KEY} digest"
    )
    observed = file_sha256(Path(__file__))
    if observed != expected:
        raise CacheIntegrityError("cache-integrity implementation drifted after freeze")
    return lock_sha256, observed


def _expected_cell_keys(config: Mapping[str, Any]) -> set[str]:
    try:
        frozen = config["frozen_cells"]
        seeds = tuple(int(value) for value in frozen["seed_bases"])
        arms = tuple(str(value) for value in frozen["arms"])
        folds = tuple(int(value) for value in frozen["folds"])
        configured_count = int(frozen["patient_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise CacheIntegrityError("frozen-cell config is incomplete") from error
    if configured_count != EXPECTED_PRIMARY_PATIENT_COUNT:
        raise CacheIntegrityError(
            "formal primary cohort must contain exactly "
            f"{EXPECTED_PRIMARY_PATIENT_COUNT} patients"
        )
    keys = {
        f"seed_{seed}/{arm}/fold_{fold}"
        for seed in seeds
        for arm in arms
        for fold in folds
    }
    if len(keys) != EXPECTED_CELL_COUNT:
        raise CacheIntegrityError(
            f"formal preregistration must contain exactly {EXPECTED_CELL_COUNT} cells"
        )
    return keys


def _locked_primary_ids(
    config: Mapping[str, Any], lock: Mapping[str, Any]
) -> tuple[str, ...]:
    """Derive the exact primary set independently from every locked reference asset."""

    cells = lock.get("selected_cells")
    if not isinstance(cells, Mapping):
        raise CacheIntegrityError("preregistration lock has no selected-cell inventory")
    expected_keys = _expected_cell_keys(config)
    if set(str(key) for key in cells) != expected_keys:
        raise CacheIntegrityError(
            "locked selected cells differ from the exact 20-cell grid"
        )

    canonical_ids: set[str] | None = None
    reference_paths: set[Path] = set()
    for key in sorted(expected_keys):
        cell = cells[key]
        if not isinstance(cell, Mapping) or not isinstance(
            cell.get("reference"), Mapping
        ):
            raise CacheIntegrityError(f"locked cell has no reference asset: {key}")
        reference = cell["reference"]
        try:
            path = Path(str(reference["path"])).expanduser().resolve(strict=True)
        except (KeyError, OSError) as error:
            raise CacheIntegrityError(
                f"locked reference asset is missing: {key}"
            ) from error
        if path in reference_paths:
            raise CacheIntegrityError("20 locked cells must name 20 reference assets")
        reference_paths.add(path)
        expected_sha256 = _require_sha256(
            reference.get("sha256"), f"reference digest for {key}"
        )
        if file_sha256(path) != expected_sha256:
            raise CacheIntegrityError(f"locked reference asset drifted: {key}")
        try:
            with np.load(path, allow_pickle=False) as archive:
                if "patient_id" not in archive.files:
                    raise CacheIntegrityError(
                        f"locked reference has no patient_id member: {key}"
                    )
                values = np.asarray(archive["patient_id"])
        except (OSError, ValueError) as error:
            raise CacheIntegrityError(
                f"locked reference is unreadable: {key}"
            ) from error
        if values.shape != (EXPECTED_PRIMARY_PATIENT_COUNT,):
            raise CacheIntegrityError(
                "locked reference does not contain "
                f"{EXPECTED_PRIMARY_PATIENT_COUNT} patients: {key}"
            )
        patient_ids = tuple(str(value) for value in values)
        if any(not value for value in patient_ids) or len(set(patient_ids)) != len(
            patient_ids
        ):
            raise CacheIntegrityError(
                f"locked reference patient IDs are invalid: {key}"
            )
        expected_order = reference.get("patient_order_sha256")
        if expected_order is not None and ordered_sha256(
            patient_ids
        ) != _require_sha256(
            expected_order, f"reference patient-order digest for {key}"
        ):
            raise CacheIntegrityError(f"locked reference patient order drifted: {key}")
        observed_set = set(patient_ids)
        if canonical_ids is None:
            canonical_ids = observed_set
        elif observed_set != canonical_ids:
            raise CacheIntegrityError(
                "the 20 locked reference patient sets are not identical"
            )

    if canonical_ids is None or len(canonical_ids) != EXPECTED_PRIMARY_PATIENT_COUNT:
        raise CacheIntegrityError(
            "could not derive the exact primary cohort from references"
        )
    return tuple(sorted(canonical_ids))


def _manifest_records(
    config: Mapping[str, Any], primary_patient_ids: Sequence[str]
) -> tuple[list[dict[str, Any]], str]:
    """Read all 947 rows and label the exact locked-primary/complement split."""

    try:
        paths = config["paths"]
        source = (
            Path(str(paths["c1b_cache_manifest"])).expanduser().resolve(strict=True)
        )
        expected_manifest_sha256 = _require_sha256(
            paths["c1b_cache_manifest_sha256"], "configured C1B cache manifest digest"
        )
    except (KeyError, OSError, TypeError) as error:
        raise CacheIntegrityError("C1B cache manifest config is incomplete") from error
    observed_manifest_sha256 = file_sha256(source)
    if observed_manifest_sha256 != expected_manifest_sha256:
        raise CacheIntegrityError("hash-bound C1B cache manifest drifted")

    try:
        with source.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = tuple(reader.fieldnames or ())
            if len(columns) != len(set(columns)) or set(columns) != set(
                MANIFEST_COLUMNS
            ):
                raise CacheIntegrityError(
                    f"C1B cache manifest columns must be exactly {MANIFEST_COLUMNS}"
                )
            rows = list(reader)
    except (OSError, csv.Error) as error:
        raise CacheIntegrityError("C1B cache manifest is unreadable") from error
    if not rows:
        raise CacheIntegrityError("C1B cache manifest is empty")

    index: dict[str, dict[str, Any]] = {}
    seen_paths: set[Path] = set()
    for number, row in enumerate(rows, start=2):
        if None in row or set(row) != set(MANIFEST_COLUMNS):
            raise CacheIntegrityError(f"malformed C1B cache manifest row {number}")
        patient_id = str(row["patient_id"])
        if not patient_id or patient_id in index:
            raise CacheIntegrityError("C1B cache manifest patient rows are not unique")
        if str(row["input_kind"]) != "c1b":
            raise CacheIntegrityError("C1B cache manifest contains a non-c1b row")
        raw_path = Path(str(row["cache_path"])).expanduser()
        if not raw_path.is_absolute():
            raise CacheIntegrityError("C1B cache manifest paths must be absolute")
        path = raw_path.resolve()
        if path in seen_paths:
            raise CacheIntegrityError("C1B cache manifest paths are not unique")
        seen_paths.add(path)
        digest = _require_sha256(row["cache_sha256"], f"cache digest at row {number}")
        size = _nonnegative_integer(
            row["cache_size_bytes"], f"cache size at row {number}"
        )
        mtime = _nonnegative_integer(
            row["cache_mtime_ns"], f"cache mtime at row {number}"
        )
        index[patient_id] = {
            "patient_id": patient_id,
            "path": str(path),
            "sha256": digest,
            "size_bytes": size,
            "mtime_ns": mtime,
        }

    if len(index) != EXPECTED_PATIENT_COUNT:
        raise CacheIntegrityError(
            "C1B cache manifest must contain exactly "
            f"{EXPECTED_PATIENT_COUNT} unique patient rows"
        )
    primary = set(str(value) for value in primary_patient_ids)
    if len(primary) != EXPECTED_PRIMARY_PATIENT_COUNT:
        raise CacheIntegrityError(
            "integrity preflight did not receive the exact primary set"
        )
    missing = sorted(primary.difference(index))
    if missing:
        raise CacheIntegrityError(
            f"C1B cache manifest misses locked primary patients: {missing[:5]}"
        )
    train_only = set(index).difference(primary)
    if len(train_only) != EXPECTED_TRAIN_ONLY_PATIENT_COUNT:
        raise CacheIntegrityError(
            "C1B cache manifest primary complement must contain exactly "
            f"{EXPECTED_TRAIN_ONLY_PATIENT_COUNT} train-only patients"
        )
    records: list[dict[str, Any]] = []
    for patient_id in sorted(index):
        record = dict(index[patient_id])
        record["cohort"] = "primary" if patient_id in primary else "train_only"
        records.append(record)
    return records, observed_manifest_sha256


def _stat_identity(stat: os.stat_result) -> tuple[int, int, int, int]:
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _hash_one(record: Mapping[str, Any]) -> dict[str, Any]:
    patient_id = str(record["patient_id"])
    try:
        path = Path(str(record["path"])).resolve(strict=True)
    except OSError as error:
        raise CacheIntegrityError(f"C1B cache is missing for {patient_id}") from error
    if not path.is_file():
        raise CacheIntegrityError(f"C1B cache is not a regular file for {patient_id}")
    before = path.stat()
    if before.st_size != int(record["size_bytes"]):
        raise CacheIntegrityError(
            f"C1B cache size drifted before hashing for {patient_id}"
        )
    if before.st_mtime_ns != int(record["mtime_ns"]):
        raise CacheIntegrityError(
            f"C1B cache mtime drifted before hashing for {patient_id}"
        )
    observed_sha256 = file_sha256(path)
    after = path.stat()
    if _stat_identity(after) != _stat_identity(before):
        raise CacheIntegrityError(f"C1B cache changed while hashing for {patient_id}")
    if observed_sha256 != record["sha256"]:
        raise CacheIntegrityError(f"C1B cache content hash mismatched for {patient_id}")
    return {
        "patient_id": patient_id,
        "path": str(path),
        "sha256": observed_sha256,
        "size_bytes": int(after.st_size),
        "mtime_ns": int(after.st_mtime_ns),
        "cohort": str(record["cohort"]),
    }


def _record_set_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    normalized = [dict(record) for record in records]
    if [str(record.get("patient_id")) for record in normalized] != sorted(
        str(record.get("patient_id")) for record in normalized
    ):
        raise CacheIntegrityError("private cache records must be sorted by patient_id")
    return canonical_sha256(normalized)


def _normalize_private_records(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != EXPECTED_PATIENT_COUNT:
        raise CacheIntegrityError(
            f"private cache manifest must contain exactly {EXPECTED_PATIENT_COUNT} records"
        )
    records: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping) or set(item) != set(PRIVATE_RECORD_COLUMNS):
            raise CacheIntegrityError(
                f"private cache record {index} has an invalid schema"
            )
        patient_id = str(item["patient_id"])
        raw_path = Path(str(item["path"])).expanduser()
        if not patient_id or not raw_path.is_absolute():
            raise CacheIntegrityError(
                f"private cache record {index} has an invalid identity"
            )
        records.append(
            {
                "patient_id": patient_id,
                "path": str(raw_path.resolve()),
                "sha256": _require_sha256(
                    item["sha256"], f"private cache record {index} digest"
                ),
                "size_bytes": _nonnegative_integer(
                    item["size_bytes"], f"private cache record {index} size"
                ),
                "mtime_ns": _nonnegative_integer(
                    item["mtime_ns"], f"private cache record {index} mtime"
                ),
                "cohort": str(item["cohort"]),
            }
        )
    patient_ids = [record["patient_id"] for record in records]
    paths = [record["path"] for record in records]
    if patient_ids != sorted(patient_ids) or len(set(patient_ids)) != len(patient_ids):
        raise CacheIntegrityError(
            "private cache records are not uniquely patient-sorted"
        )
    if len(set(paths)) != len(paths):
        raise CacheIntegrityError("private cache records do not name unique paths")
    cohorts = [record["cohort"] for record in records]
    if set(cohorts) != {"primary", "train_only"}:
        raise CacheIntegrityError("private cache records have invalid cohort labels")
    if cohorts.count("primary") != EXPECTED_PRIMARY_PATIENT_COUNT:
        raise CacheIntegrityError("private cache primary count drifted")
    if cohorts.count("train_only") != EXPECTED_TRAIN_ONLY_PATIENT_COUNT:
        raise CacheIntegrityError("private cache train-only count drifted")
    return records


def _environment() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "machine": platform.machine(),
    }


def _authenticate_artifacts(
    config: Mapping[str, Any],
    lock: Mapping[str, Any],
    *,
    verify_live_stats: bool,
    private_path: Path,
    contract_path: Path,
) -> dict[str, Any]:
    lock_sha256, implementation_sha256 = _current_lock_context(config, lock)
    patient_ids = _locked_primary_ids(config, lock)
    expected_records, manifest_sha256 = _manifest_records(config, patient_ids)
    private = _read_json(private_path, "private cache-integrity manifest")
    contract = _read_json(contract_path, "public cache-integrity contract")

    if private_path.stat().st_mode & 0o777 != 0o600:
        raise CacheIntegrityError("private cache-integrity artifact is not mode 0600")
    if private_path.parent.stat().st_mode & 0o777 != 0o700:
        raise CacheIntegrityError("private cache-integrity directory is not mode 0700")
    if contract_path.stat().st_mode & 0o777 != 0o644:
        raise CacheIntegrityError("public cache-integrity contract is not mode 0644")
    if set(private) != set(PRIVATE_COLUMNS):
        raise CacheIntegrityError("private cache-integrity manifest schema drifted")
    if set(contract) != set(PUBLIC_COLUMNS):
        raise CacheIntegrityError("public cache-integrity contract schema drifted")
    if contract.get("schema_version") != 2 or contract.get("status") != "COMPLETE":
        raise CacheIntegrityError("public cache-integrity contract is incomplete")
    if contract.get("purpose") != PURPOSE:
        raise CacheIntegrityError("public cache-integrity purpose drifted")
    if contract.get("private_artifact_sha256") != file_sha256(private_path):
        raise CacheIntegrityError(
            "public contract does not authenticate its private manifest"
        )
    if private.get("schema_version") != 2 or private.get("status") != "COMPLETE":
        raise CacheIntegrityError("private cache-integrity manifest is incomplete")
    if private.get("patient_count") != EXPECTED_PATIENT_COUNT:
        raise CacheIntegrityError("private cache-integrity patient count drifted")
    if private.get("primary_patient_count") != EXPECTED_PRIMARY_PATIENT_COUNT:
        raise CacheIntegrityError("private cache-integrity primary count drifted")
    if private.get("train_only_patient_count") != EXPECTED_TRAIN_ONLY_PATIENT_COUNT:
        raise CacheIntegrityError("private cache-integrity train-only count drifted")
    records = _normalize_private_records(private.get("records"))
    primary_records = [record for record in records if record["cohort"] == "primary"]
    train_only_records = [
        record for record in records if record["cohort"] == "train_only"
    ]
    if tuple(record["patient_id"] for record in primary_records) != patient_ids:
        raise CacheIntegrityError(
            "private cache-integrity cohort differs from locked references"
        )
    if records != expected_records:
        raise CacheIntegrityError(
            "private cache records differ from the hash-bound manifest"
        )

    if contract.get("patient_count") != EXPECTED_PATIENT_COUNT:
        raise CacheIntegrityError("public cache-integrity patient count drifted")
    if contract.get("primary_patient_count") != EXPECTED_PRIMARY_PATIENT_COUNT:
        raise CacheIntegrityError("public cache-integrity primary count drifted")
    if contract.get("train_only_patient_count") != EXPECTED_TRAIN_ONLY_PATIENT_COUNT:
        raise CacheIntegrityError("public cache-integrity train-only count drifted")
    if contract.get("total_bytes") != sum(record["size_bytes"] for record in records):
        raise CacheIntegrityError("public cache-integrity byte count drifted")
    if contract.get("upstream_manifest_sha256") != manifest_sha256:
        raise CacheIntegrityError(
            "public contract is not bound to the current cache manifest"
        )
    if contract.get("preregistration_lock_sha256") != lock_sha256:
        raise CacheIntegrityError(
            "public contract is not bound to the current preregistration"
        )
    if contract.get("implementation_sha256") != implementation_sha256:
        raise CacheIntegrityError(
            "public contract is not bound to the frozen implementation"
        )
    record_set_sha256 = _record_set_sha256(records)
    if contract.get("canonical_record_set_sha256") != record_set_sha256:
        raise CacheIntegrityError("public canonical record-set digest drifted")
    if contract.get("primary_record_set_sha256") != _record_set_sha256(primary_records):
        raise CacheIntegrityError("public primary record-set digest drifted")
    if contract.get("train_only_record_set_sha256") != _record_set_sha256(
        train_only_records
    ):
        raise CacheIntegrityError("public train-only record-set digest drifted")
    expected_environment = _environment()
    if contract.get("environment") != expected_environment:
        raise CacheIntegrityError(
            "public cache-integrity environment differs from the current exact runtime"
        )
    if contract.get("contains_patient_identifiers") is not False:
        raise CacheIntegrityError(
            "public cache-integrity contract has a privacy flag error"
        )

    if verify_live_stats:
        for record in records:
            patient_id = record["patient_id"]
            try:
                path = Path(record["path"]).resolve(strict=True)
            except OSError as error:
                raise CacheIntegrityError(
                    f"authenticated C1B cache is missing for {patient_id}"
                ) from error
            if not path.is_file():
                raise CacheIntegrityError(
                    f"authenticated C1B cache is not a regular file for {patient_id}"
                )
            stat = path.stat()
            if (
                stat.st_size != record["size_bytes"]
                or stat.st_mtime_ns != record["mtime_ns"]
            ):
                raise CacheIntegrityError(
                    f"authenticated C1B cache live stat drifted for {patient_id}"
                )

    result = dict(private)
    result["records"] = records
    return result


def require_cache_integrity(
    config: Mapping[str, Any],
    lock: Mapping[str, Any],
    *,
    verify_live_stats: bool = True,
) -> dict[str, Any]:
    """Require the authenticated 947-cache preflight without rehashing content.

    ``verify_live_stats=True`` checks each current cache size and nanosecond mtime
    against the one-time content-verified private manifest. It deliberately does
    not recompute any cache SHA-256 digest.
    """

    return _authenticate_artifacts(
        config,
        lock,
        verify_live_stats=verify_live_stats,
        private_path=PRIVATE_MANIFEST,
        contract_path=PUBLIC_CONTRACT,
    )


def _existing_pair_is_complete(
    config: Mapping[str, Any],
    lock: Mapping[str, Any],
    private_path: Path,
    contract_path: Path,
) -> bool:
    if not private_path.exists() or not contract_path.exists():
        return False
    try:
        _authenticate_artifacts(
            config,
            lock,
            verify_live_stats=False,
            private_path=private_path,
            contract_path=contract_path,
        )
    except Exception:
        return False
    return True


def build_cache_integrity(
    config: Mapping[str, Any],
    lock: Mapping[str, Any],
    *,
    workers: int = 1,
    private_path: Path | None = None,
    contract_path: Path | None = None,
) -> dict[str, Any]:
    """Hash all 947 manifest caches once and atomically publish the paired proof."""

    worker_count = _positive_integer(workers, "workers")
    private_path = (
        Path(PRIVATE_MANIFEST if private_path is None else private_path)
        .expanduser()
        .resolve()
    )
    contract_path = (
        Path(PUBLIC_CONTRACT if contract_path is None else contract_path)
        .expanduser()
        .resolve()
    )

    # This happens before inspecting, deleting, or creating either output.
    lock_sha256, implementation_sha256 = _current_lock_context(config, lock)
    patient_ids = _locked_primary_ids(config, lock)
    manifest_records, manifest_sha256 = _manifest_records(config, patient_ids)

    if _existing_pair_is_complete(config, lock, private_path, contract_path):
        raise FileExistsError("refusing to overwrite complete cache-integrity pair")
    # Only these two experiment-local targets are recoverable partial outputs.
    private_path.unlink(missing_ok=True)
    contract_path.unlink(missing_ok=True)

    if worker_count == 1:
        records = [_hash_one(record) for record in manifest_records]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            records = list(executor.map(_hash_one, manifest_records))
    records.sort(key=lambda record: record["patient_id"])
    if records != manifest_records:
        raise CacheIntegrityError(
            "content-verified records differ from the frozen manifest"
        )

    private_payload = {
        "schema_version": 2,
        "status": "COMPLETE",
        "patient_count": EXPECTED_PATIENT_COUNT,
        "primary_patient_count": EXPECTED_PRIMARY_PATIENT_COUNT,
        "train_only_patient_count": EXPECTED_TRAIN_ONLY_PATIENT_COUNT,
        "records": records,
    }
    public_payload: dict[str, Any]
    try:
        private_directory(private_path.parent)
        atomic_json(private_payload, private_path, private=True)
        public_payload = {
            "schema_version": 2,
            "status": "COMPLETE",
            "purpose": PURPOSE,
            "patient_count": EXPECTED_PATIENT_COUNT,
            "primary_patient_count": EXPECTED_PRIMARY_PATIENT_COUNT,
            "train_only_patient_count": EXPECTED_TRAIN_ONLY_PATIENT_COUNT,
            "total_bytes": int(sum(record["size_bytes"] for record in records)),
            "upstream_manifest_sha256": manifest_sha256,
            "private_artifact_sha256": file_sha256(private_path),
            "canonical_record_set_sha256": _record_set_sha256(records),
            "primary_record_set_sha256": _record_set_sha256(
                [record for record in records if record["cohort"] == "primary"]
            ),
            "train_only_record_set_sha256": _record_set_sha256(
                [record for record in records if record["cohort"] == "train_only"]
            ),
            "preregistration_lock_sha256": lock_sha256,
            "implementation_sha256": implementation_sha256,
            "environment": _environment(),
            "contains_patient_identifiers": False,
        }
        atomic_json(public_payload, contract_path, private=False)
        _authenticate_artifacts(
            config,
            lock,
            verify_live_stats=True,
            private_path=private_path,
            contract_path=contract_path,
        )
        if private_path.stat().st_mode & 0o777 != 0o600:
            raise CacheIntegrityError(
                "private cache-integrity manifest is not mode 0600"
            )
    except Exception:
        private_path.unlink(missing_ok=True)
        contract_path.unlink(missing_ok=True)
        raise
    return public_payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel cache hash readers (default: 1, conservative)",
    )
    return parser.parse_args()


def main() -> None:
    os.umask(0o077)
    args = _parse_args()
    # The formal CLI intentionally accepts no path/config overrides.
    config = load_config(DEFAULT_CONFIG, verify_inputs=True)
    lock = require_preregistration_lock(config)
    result = build_cache_integrity(config, lock, workers=args.workers)
    print(
        json.dumps(
            {
                "status": result["status"],
                "patient_count": result["patient_count"],
                "primary_patient_count": result["primary_patient_count"],
                "train_only_patient_count": result["train_only_patient_count"],
                "total_bytes": result["total_bytes"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "CacheIntegrityError",
    "build_cache_integrity",
    "require_cache_integrity",
]

"""Pinned Stage B artifact bundle shared by training and feature export."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .contracts import (
    ISPY1_ELIGIBLE_COUNT,
    PRIMARY_PATIENT_COUNT,
    STAGE_B_PATIENT_COUNT,
    file_sha256,
    require_sha256,
)
from .data import (
    CacheEntry,
    FTVRecord,
    combine_ftv_observability,
    read_cache_manifest,
    read_fold_manifest,
    read_ispy1_eligibility,
    read_observability,
    read_raw_ftv,
    validate_cache_coverage,
)


@dataclass(frozen=True)
class StageBDataPaths:
    fold_manifest: Path
    fold_manifest_sha256: str
    ispy1_eligibility_manifest: Path
    ispy1_eligibility_manifest_sha256: str
    legacy_cache_manifest: Path
    legacy_cache_manifest_sha256: str
    c1b_cache_manifest: Path
    c1b_cache_manifest_sha256: str
    ftv_transition_table: Path
    ftv_transition_table_sha256: str
    observability_manifest: Path
    observability_manifest_sha256: str

    def provenance(self) -> dict[str, Any]:
        return {
            field: str(getattr(self, field)) if isinstance(getattr(self, field), Path) else getattr(self, field)
            for field in self.__dataclass_fields__
        }

    @classmethod
    def load(cls, path: str | Path, expected_sha256: str) -> "StageBDataPaths":
        source = Path(path).expanduser().resolve()
        expected = require_sha256(expected_sha256, "Stage B data contract")
        if file_sha256(source) != expected:
            raise ValueError("Stage B data-contract SHA-256 mismatch")
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or int(payload.pop("schema_version", -1)) != 1:
            raise ValueError("Stage B data contract must be a schema-v1 JSON object")
        expected_fields = set(cls.__dataclass_fields__)
        if set(payload) != expected_fields:
            raise ValueError(
                f"Stage B data-contract fields drifted: expected {sorted(expected_fields)}, got {sorted(payload)}"
            )
        for field in expected_fields:
            if field.endswith("_sha256"):
                payload[field] = require_sha256(payload[field], field)
            else:
                value = Path(str(payload[field])).expanduser()
                payload[field] = value.resolve() if value.is_absolute() else (source.parent / value).resolve()
        return cls(**payload)


@dataclass(frozen=True)
class StageBDataBundle:
    folds: Any
    eligible_ispy1: tuple[str, ...]
    legacy_cache: dict[str, CacheEntry]
    c1b_cache: dict[str, CacheEntry]
    ftv: dict[str, FTVRecord]
    provenance: dict[str, Any]


def load_stage_b_data(
    paths: StageBDataPaths, *, verify_cache_files: bool = True
) -> StageBDataBundle:
    folds = read_fold_manifest(paths.fold_manifest, paths.fold_manifest_sha256)
    eligible = read_ispy1_eligibility(
        paths.ispy1_eligibility_manifest,
        paths.ispy1_eligibility_manifest_sha256,
    )
    legacy = read_cache_manifest(
        paths.legacy_cache_manifest,
        paths.legacy_cache_manifest_sha256,
        expected_input_kind="legacy",
        verify_cache_files=verify_cache_files,
    )
    c1b = read_cache_manifest(
        paths.c1b_cache_manifest,
        paths.c1b_cache_manifest_sha256,
        expected_input_kind="c1b",
        verify_cache_files=verify_cache_files,
    )
    raw = read_raw_ftv(paths.ftv_transition_table, paths.ftv_transition_table_sha256)
    observability = read_observability(
        paths.observability_manifest, paths.observability_manifest_sha256
    )
    ftv = combine_ftv_observability(raw, observability)
    primary = set(folds["patient_id"].astype(str))
    required = primary | set(eligible)
    if (
        len(primary) != PRIMARY_PATIENT_COUNT
        or len(eligible) != ISPY1_ELIGIBLE_COUNT
        or len(required) != STAGE_B_PATIENT_COUNT
    ):
        raise ValueError(
            "Stage B cohort must be exactly 808 primary + 140 eligible I-SPY1 "
            f"= 948 unique patients; observed {len(primary)} + {len(eligible)} "
            f"= {len(required)}"
        )
    validate_cache_coverage(legacy, c1b, required)
    if not set(ftv).issubset(primary):
        raise ValueError("FTV/observability contains patients outside the primary cohort")
    provenance = paths.provenance()
    provenance.update(
        {
            "primary_patient_count": len(primary),
            "eligible_ispy1_count": len(eligible),
            "ftv_patient_count": len(ftv),
            "adapter": "exact_usecols_split_ftv_observability_only",
            "cache_files_verified": bool(verify_cache_files),
        }
    )
    return StageBDataBundle(folds, eligible, legacy, c1b, ftv, provenance)


__all__ = ["StageBDataBundle", "StageBDataPaths", "load_stage_b_data"]

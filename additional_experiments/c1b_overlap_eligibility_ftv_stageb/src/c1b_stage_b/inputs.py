"""SHA-pinned Stage B artifact bundle and dynamic matched-cohort contract."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .contracts import (
    LOCKED_C1B_CACHE_CONTRACT_SHA256,
    LOCKED_G3_DATA_SHA256,
    G3_SRC,
    PRIOR_C1B_SRC,
    file_sha256,
    ordered_patient_sha256,
    require_sha256,
)
from .data import (
    CacheEntry,
    FTVRecord,
    TechnicalEligibilityPopulation,
    combine_ftv_observability,
    derive_matched_stage_b_population,
    read_cache_manifest,
    read_fold_manifest,
    read_observability,
    read_raw_ftv,
    read_technical_eligibility,
    read_train_only_candidates,
    validate_cache_coverage,
)
from .gate import StageAAuthorization


@dataclass(frozen=True)
class StageBDataPaths:
    fold_manifest: Path
    fold_manifest_sha256: str
    technical_eligibility_manifest: Path
    technical_eligibility_manifest_sha256: str
    train_only_candidate_manifest: Path
    train_only_candidate_manifest_sha256: str
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
            field: (
                str(getattr(self, field))
                if isinstance(getattr(self, field), Path)
                else getattr(self, field)
            )
            for field in self.__dataclass_fields__
        }

    @classmethod
    def load(cls, path: str | Path, expected_sha256: str) -> "StageBDataPaths":
        source = Path(path).expanduser().resolve()
        expected = require_sha256(expected_sha256, "Stage B data contract")
        if file_sha256(source) != expected:
            raise ValueError("Stage B data-contract SHA-256 mismatch")
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or int(payload.pop("schema_version", -1)) != 2:
            raise ValueError("Stage B data contract must be a schema-v2 JSON object")
        expected_fields = set(cls.__dataclass_fields__)
        if set(payload) != expected_fields:
            raise ValueError(
                "Stage B data-contract fields drifted: "
                f"expected {sorted(expected_fields)}, got {sorted(payload)}"
            )
        for field in expected_fields:
            if field.endswith("_sha256"):
                payload[field] = require_sha256(payload[field], field)
            else:
                value = Path(str(payload[field])).expanduser()
                payload[field] = (
                    value.resolve()
                    if value.is_absolute()
                    else (source.parent / value).resolve()
                )
        return cls(**payload)


@dataclass(frozen=True)
class StageBDataBundle:
    folds: Any
    eligibility: TechnicalEligibilityPopulation
    train_only_ids: tuple[str, ...]
    legacy_cache: dict[str, CacheEntry]
    c1b_cache: dict[str, CacheEntry]
    ftv: dict[str, FTVRecord]
    provenance: dict[str, Any]


def load_stage_b_data(
    paths: StageBDataPaths,
    authorization: StageAAuthorization,
    *,
    verify_cache_files: bool = True,
) -> StageBDataBundle:
    """Load only the cohort authorized by the exact new Stage-A sentinel."""

    cache_contract_sha256 = file_sha256(PRIOR_C1B_SRC / "c1b_sanity" / "cache.py")
    if cache_contract_sha256 != LOCKED_C1B_CACHE_CONTRACT_SHA256:
        raise ValueError("frozen C1B schema-3 cache contract hash drifted")
    legacy_data_contract_sha256 = file_sha256(G3_SRC / "dgrs" / "data.py")
    if legacy_data_contract_sha256 != LOCKED_G3_DATA_SHA256:
        raise ValueError("frozen legacy DCE7 data/input contract hash drifted")
    if paths.technical_eligibility_manifest_sha256 != (
        authorization.technical_eligibility_manifest_sha256
    ):
        raise ValueError(
            "data contract eligibility manifest disagrees with Stage A authorization"
        )
    folds_all = read_fold_manifest(paths.fold_manifest, paths.fold_manifest_sha256)
    eligibility = read_technical_eligibility(
        paths.technical_eligibility_manifest,
        paths.technical_eligibility_manifest_sha256,
    )
    if len(eligibility.eligible_ids) != authorization.eligible_population_patients:
        raise ValueError(
            "technical eligibility count disagrees with the Stage A GO sentinel"
        )

    upstream_train_only = read_train_only_candidates(
        paths.train_only_candidate_manifest,
        paths.train_only_candidate_manifest_sha256,
    )
    matched = derive_matched_stage_b_population(
        folds_all, eligibility, upstream_train_only
    )
    folds = matched.folds
    fold_candidates = set(matched.fold_candidate_ids)
    fold_eligible = set(matched.fold_eligible_ids)
    train_only_ids = matched.train_only_ids
    required = set(matched.matched_patient_ids)

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
    validate_cache_coverage(legacy, c1b, required)

    raw = read_raw_ftv(paths.ftv_transition_table, paths.ftv_transition_table_sha256)
    observability = read_observability(
        paths.observability_manifest, paths.observability_manifest_sha256
    )
    ftv = combine_ftv_observability(raw, observability)
    if not set(ftv).issubset(set(eligibility.candidate_ids)):
        raise ValueError(
            "FTV/observability contains patients outside the technical candidate population"
        )
    dropped_nonprobe_ftv = len(set(ftv).difference(fold_eligible))
    ftv = {
        patient_id: record
        for patient_id, record in ftv.items()
        if patient_id in fold_eligible
    }

    provenance = paths.provenance()
    fold_split_counts: dict[str, dict[str, int]] = {}
    for fold in range(5):
        before = folds_all.loc[folds_all["fold"].eq(fold)]
        after = folds.loc[folds["fold"].eq(fold)]
        counts = {
            split: int(after["split"].eq(split).sum())
            for split in ("train", "val", "test")
        }
        counts["train_only_added"] = len(train_only_ids)
        counts["train_total"] = counts["train"] + len(train_only_ids)
        counts["technical_excluded"] = int(len(before) - len(after))
        fold_split_counts[str(fold)] = counts
    provenance.update(
        {
            "candidate_patient_count": len(eligibility.candidate_ids),
            "eligible_patient_count": len(eligibility.eligible_ids),
            "technical_excluded_patient_count": len(eligibility.excluded_ids),
            "fold_candidate_patient_count": len(fold_candidates),
            "fold_eligible_patient_count": len(fold_eligible),
            "fold_technical_excluded_patient_count": len(
                fold_candidates.difference(fold_eligible)
            ),
            "train_only_patient_count": len(train_only_ids),
            "matched_cohort_patient_count": len(required),
            "eligible_patient_order_sha256": ordered_patient_sha256(
                eligibility.eligible_ids
            ),
            "ftv_patient_count": len(ftv),
            "ftv_rows_excluded_from_probes_by_population_contract": dropped_nonprobe_ftv,
            "cohort_derivation": (
                "technical_eligibility_intersection_seed2026_folds_plus_"
                "upstream_authorized_train_only"
            ),
            "fold_split_counts": fold_split_counts,
            "adapter": "exact_usecols_split_ftv_observability_only",
            "frozen_c1b_cache_contract_sha256": cache_contract_sha256,
            "frozen_legacy_data_contract_sha256": legacy_data_contract_sha256,
            "cache_files_verified": bool(verify_cache_files),
        }
    )
    return StageBDataBundle(
        folds, eligibility, train_only_ids, legacy, c1b, ftv, provenance
    )


__all__ = ["StageBDataBundle", "StageBDataPaths", "load_stage_b_data"]

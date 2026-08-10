#!/usr/bin/env python3
"""Create SHA-pinned private legacy/C1B cache inventories after Stage A GO."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c1b_stage_b.cli import add_gate_arguments, authorize  # noqa: E402
from c1b_stage_b.contracts import (  # noqa: E402
    LOCKED_FTV_TRANSITION_TABLE_SHA256,
    LOCKED_OBSERVABILITY_MANIFEST_SHA256,
    file_sha256,
    require_sha256,
)
from c1b_stage_b.data import (  # noqa: E402
    CacheEntry,
    derive_matched_stage_b_population,
    fingerprint_cache_file,
    read_fold_manifest,
    read_technical_eligibility,
    read_train_only_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_gate_arguments(parser)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--fold-manifest-sha256", required=True)
    parser.add_argument("--technical-eligibility-manifest", type=Path, required=True)
    parser.add_argument("--technical-eligibility-manifest-sha256", required=True)
    parser.add_argument("--train-only-candidate-manifest", type=Path, required=True)
    parser.add_argument("--train-only-candidate-manifest-sha256", required=True)
    parser.add_argument("--legacy-cache-root", type=Path, required=True)
    parser.add_argument("--c1b-stage-a-cache-table", type=Path, required=True)
    parser.add_argument("--c1b-stage-a-cache-table-sha256", required=True)
    parser.add_argument("--legacy-output", type=Path, required=True)
    parser.add_argument("--c1b-output", type=Path, required=True)
    parser.add_argument("--ftv-transition-table", type=Path, required=True)
    parser.add_argument("--ftv-transition-table-sha256", required=True)
    parser.add_argument("--observability-manifest", type=Path, required=True)
    parser.add_argument("--observability-manifest-sha256", required=True)
    parser.add_argument("--data-contract-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    return parser.parse_args()


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path = path.resolve()
    if not path.name.endswith(".private.csv"):
        raise ValueError("identifier/path-bearing cache manifests must end in .private.csv")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False)
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _legacy_patient_id(path: Path) -> str:
    marker = "_dce8_"
    if marker not in path.name:
        raise ValueError(f"legacy cache filename has no {marker!r} patient delimiter: {path}")
    return path.name.split(marker, 1)[0]


def _manifest_row(entry: CacheEntry) -> dict[str, object]:
    return {
        "patient_id": entry.patient_id,
        "cache_path": str(entry.path),
        "cache_sha256": entry.sha256,
        "cache_size_bytes": entry.size_bytes,
        "cache_mtime_ns": entry.mtime_ns,
        "input_kind": entry.input_kind,
    }


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    authorization = authorize(args)
    requested_outputs = (
        args.legacy_output.resolve(),
        args.c1b_output.resolve(),
        args.data_contract_output.resolve(),
        args.summary_output.resolve(),
    )
    if any(path.exists() for path in requested_outputs):
        raise FileExistsError("refusing to overwrite any Stage B cache-manifest output")
    folds_all = read_fold_manifest(args.fold_manifest, args.fold_manifest_sha256)
    eligibility_sha256 = require_sha256(
        args.technical_eligibility_manifest_sha256,
        "technical eligibility manifest",
    )
    eligibility = read_technical_eligibility(
        args.technical_eligibility_manifest,
        eligibility_sha256,
    )
    if eligibility_sha256 != authorization.technical_eligibility_manifest_sha256:
        raise ValueError("technical eligibility hash disagrees with Stage A GO")
    if len(eligibility.eligible_ids) != authorization.eligible_population_patients:
        raise ValueError("technical eligibility count disagrees with Stage A GO")
    upstream_train_only = read_train_only_candidates(
        args.train_only_candidate_manifest,
        args.train_only_candidate_manifest_sha256,
    )
    matched = derive_matched_stage_b_population(
        folds_all, eligibility, upstream_train_only
    )
    folds = matched.folds
    extras = matched.train_only_ids
    required = set(matched.matched_patient_ids)
    legacy_paths = sorted(args.legacy_cache_root.resolve().glob("*.npz"))
    legacy_index = {_legacy_patient_id(path): path for path in legacy_paths}
    if len(legacy_index) != len(legacy_paths):
        raise ValueError("legacy cache root contains duplicate patient IDs")
    if missing := sorted(required.difference(legacy_index)):
        raise FileNotFoundError(f"legacy cache root misses required patients: {missing[:5]}")
    legacy = pd.DataFrame(
        [
            _manifest_row(
                fingerprint_cache_file(
                    legacy_index[patient_id], patient_id, "legacy"
                )
            )
            for patient_id in sorted(required)
        ]
    )
    c1b_source = args.c1b_stage_a_cache_table.resolve()
    expected = require_sha256(args.c1b_stage_a_cache_table_sha256, "Stage A C1B cache table")
    chosen_strategy = str(authorization.payload.get("chosen_input_strategy", ""))
    if chosen_strategy != "C1B-H":
        raise ValueError("Stage A sentinel must freeze C1B-H")
    pinned_cache_hashes: set[str] = set()
    for key in ("eligible_cache_manifest_sha256", "c1b_cache_table_sha256"):
        value = authorization.payload.get(key)
        if value is not None:
            pinned_cache_hashes.add(require_sha256(str(value), key))
    for container_name in (
        "provenance_sha256",
        "private_manifest_sha256",
        "private_artifact_sha256",
    ):
        container = authorization.payload.get(container_name)
        if isinstance(container, dict):
            for key, value in container.items():
                normalized_key = str(key).lower()
                if "cache" in normalized_key or "model_input" in normalized_key:
                    try:
                        pinned_cache_hashes.add(
                            require_sha256(str(value), f"Stage A provenance {key}")
                        )
                    except ValueError:
                        pass
    if expected not in pinned_cache_hashes:
        raise ValueError("Stage A GO does not SHA-pin the requested eligible C1B cache table")
    if file_sha256(c1b_source) != expected:
        raise ValueError("Stage A C1B cache table SHA-256 mismatch")
    c1b_raw = pd.read_csv(
        c1b_source, usecols=["patient_id", "cache_path", "cache_file_sha256"]
    )
    c1b_raw["patient_id"] = c1b_raw["patient_id"].astype(str)
    if c1b_raw["patient_id"].duplicated().any():
        raise ValueError("Stage A C1B cache table contains duplicate patients")
    c1b_raw = c1b_raw.set_index("patient_id")
    if missing := sorted(required.difference(c1b_raw.index)):
        raise FileNotFoundError(f"Stage A C1B cache table misses required patients: {missing[:5]}")
    c1b_rows = []
    for patient_id in sorted(required):
        row = c1b_raw.loc[patient_id]
        path = Path(str(row["cache_path"])).resolve()
        digest = require_sha256(str(row["cache_file_sha256"]), f"C1B cache {patient_id}")
        entry = fingerprint_cache_file(
            path,
            patient_id,
            "c1b",
            expected_sha256=digest,
        )
        c1b_rows.append(_manifest_row(entry))
    c1b = pd.DataFrame(c1b_rows)
    ftv_path = args.ftv_transition_table.resolve()
    ftv_sha256 = require_sha256(args.ftv_transition_table_sha256, "FTV transition table")
    observable_path = args.observability_manifest.resolve()
    observable_sha256 = require_sha256(
        args.observability_manifest_sha256, "grounding observability manifest"
    )
    if ftv_sha256 != LOCKED_FTV_TRANSITION_TABLE_SHA256:
        raise ValueError("FTV transition table is not the frozen formal asset")
    if observable_sha256 != LOCKED_OBSERVABILITY_MANIFEST_SHA256:
        raise ValueError("observability manifest is not the frozen loss-side asset")
    if file_sha256(ftv_path) != ftv_sha256:
        raise ValueError("FTV transition table SHA-256 mismatch")
    if file_sha256(observable_path) != observable_sha256:
        raise ValueError("grounding observability manifest SHA-256 mismatch")
    data_contract_path = args.data_contract_output.resolve()
    if not data_contract_path.name.endswith(".private.json"):
        raise ValueError("path-bearing Stage B data contract must end in .private.json")
    _atomic_csv(args.legacy_output, legacy)
    _atomic_csv(args.c1b_output, c1b)
    data_contract = {
        "schema_version": 2,
        "fold_manifest": str(args.fold_manifest.resolve()),
        "fold_manifest_sha256": args.fold_manifest_sha256,
        "technical_eligibility_manifest": str(
            args.technical_eligibility_manifest.resolve()
        ),
        "technical_eligibility_manifest_sha256": (
            eligibility_sha256
        ),
        "train_only_candidate_manifest": str(
            args.train_only_candidate_manifest.resolve()
        ),
        "train_only_candidate_manifest_sha256": (
            args.train_only_candidate_manifest_sha256
        ),
        "legacy_cache_manifest": str(args.legacy_output.resolve()),
        "legacy_cache_manifest_sha256": file_sha256(args.legacy_output),
        "c1b_cache_manifest": str(args.c1b_output.resolve()),
        "c1b_cache_manifest_sha256": file_sha256(args.c1b_output),
        "ftv_transition_table": str(ftv_path),
        "ftv_transition_table_sha256": ftv_sha256,
        "observability_manifest": str(observable_path),
        "observability_manifest_sha256": observable_sha256,
    }
    _atomic_json(data_contract_path, data_contract)
    summary = {
        "schema_version": 2,
        "stage_a_sentinel_sha256": authorization.sha256,
        "patients": len(required),
        "candidate_patients": len(eligibility.candidate_ids),
        "eligible_patients": len(eligibility.eligible_ids),
        "technical_excluded_patients": len(eligibility.excluded_ids),
        "fold_eligible_patients": int(folds["patient_id"].nunique()),
        "upstream_authorized_train_only_patients": len(extras),
        "legacy_manifest_sha256": file_sha256(args.legacy_output),
        "c1b_manifest_sha256": file_sha256(args.c1b_output),
        "cache_rows_pin": ["sha256", "size_bytes", "mtime_ns"],
        "c1b_envelope_checks": [
            "exact_schema3_npz_members",
            "integer_scalar_schema_version_3",
            "embedded_patient_identity",
        ],
        "data_contract_sha256": file_sha256(data_contract_path),
        "private_manifests_contain_identifiers_and_paths": True,
    }
    summary_path = args.summary_output.resolve()
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite {summary_path}")
    _atomic_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

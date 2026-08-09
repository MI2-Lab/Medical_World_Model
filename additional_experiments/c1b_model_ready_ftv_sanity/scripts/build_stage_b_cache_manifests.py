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
    ISPY1_ELIGIBLE_COUNT,
    PRIMARY_PATIENT_COUNT,
    STAGE_B_PATIENT_COUNT,
    file_sha256,
    require_sha256,
)
from c1b_stage_b.data import (  # noqa: E402
    CacheEntry,
    fingerprint_cache_file,
    read_fold_manifest,
    read_ispy1_eligibility,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_gate_arguments(parser)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--fold-manifest-sha256", required=True)
    parser.add_argument("--ispy1-eligibility-manifest", type=Path, required=True)
    parser.add_argument("--ispy1-eligibility-manifest-sha256", required=True)
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
    folds = read_fold_manifest(args.fold_manifest, args.fold_manifest_sha256)
    extras = read_ispy1_eligibility(
        args.ispy1_eligibility_manifest, args.ispy1_eligibility_manifest_sha256
    )
    required = set(folds["patient_id"].astype(str)) | set(extras)
    if (
        int(folds["patient_id"].nunique()) != PRIMARY_PATIENT_COUNT
        or len(extras) != ISPY1_ELIGIBLE_COUNT
        or len(required) != STAGE_B_PATIENT_COUNT
    ):
        raise ValueError(
            "Stage B cache manifests require exactly 808 primary + 140 eligible "
            f"I-SPY1 = 948 unique patients; observed "
            f"{folds['patient_id'].nunique()} + {len(extras)} = {len(required)}"
        )
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
    if chosen_strategy not in {"C1B-H", "C1B-R"}:
        raise ValueError("Stage A sentinel does not freeze a supported C1B strategy")
    expected_c1b_source = (
        ROOT
        / "metrics"
        / f"model_input_pipeline_{chosen_strategy[-1].lower()}_all.private.csv"
    ).resolve()
    if c1b_source != expected_c1b_source:
        raise ValueError(
            "C1B cache manifest must be built only from the full table selected "
            f"and validated by Stage A: {expected_c1b_source}"
        )
    provenance = authorization.payload.get("provenance_sha256")
    provenance_key = str(c1b_source.relative_to(ROOT))
    if not isinstance(provenance, dict) or provenance_key not in provenance:
        raise ValueError("Stage A sentinel does not pin the selected full C1B table")
    sentinel_table_sha256 = require_sha256(
        str(provenance[provenance_key]), "Stage A sentinel C1B table SHA-256"
    )
    if expected != sentinel_table_sha256:
        raise ValueError("requested C1B table SHA-256 disagrees with the Stage A sentinel")
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
        "schema_version": 1,
        "fold_manifest": str(args.fold_manifest.resolve()),
        "fold_manifest_sha256": args.fold_manifest_sha256,
        "ispy1_eligibility_manifest": str(args.ispy1_eligibility_manifest.resolve()),
        "ispy1_eligibility_manifest_sha256": args.ispy1_eligibility_manifest_sha256,
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
        "schema_version": 1,
        "stage_a_sentinel_sha256": authorization.sha256,
        "patients": len(required),
        "primary_patients": int(folds["patient_id"].nunique()),
        "eligible_ispy1_train_only_patients": len(extras),
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

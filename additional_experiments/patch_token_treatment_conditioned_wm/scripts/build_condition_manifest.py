#!/usr/bin/env python3
"""Build one fold's aggregate-only private transition-condition manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Iterable

import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from patch_token_wm.data import (  # noqa: E402
    CLINICAL_CSV_USECOLS,
    ConditionEncoder,
    ISPY1_CLINICAL_SHA256,
    ISPY1_CLINICAL_TABLE,
    ISPY2_CLINICAL_SHA256,
    ISPY2_CLINICAL_TABLE,
    file_sha256,
    load_authorized_condition_table,
    patient_set_sha256,
    require_sha256,
)


DEFAULT_FOLD_MANIFEST = Path(
    "/data/data/Preprocessed/I-SPY2/"
    "_matched_breastdcedl_t0_dicomrepair_rgb224_seed2026/"
    "matched_patient_cv_splits_seed2026.csv"
)
DEFAULT_FOLD_MANIFEST_SHA256 = (
    "143e482d711225c0611006d99bd7345d2fa1a5c16c65fbaf8399341a0d26aa38"
)
FOLD_USECOLS = ("patient_id", "fold", "split")
EXTERNAL_USECOLS = ("patient_id",)
EXTERNAL_ELIGIBILITY_USECOLS = ("patient_id", "eligible")

_PATIENT_VALUE_RE = re.compile(r"(?:ISPY[12][-_]\d|ACRIN-6698-\d)", re.IGNORECASE)


def _boolean(series: pd.Series, label: str) -> pd.Series:
    if series.dtype == bool:
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    allowed = {"true": True, "false": False, "1": True, "0": False}
    if not normalized.isin(allowed).all():
        raise ValueError(f"{label} contains a non-boolean value")
    return normalized.map(allowed).astype(bool)


def _verify_source(
    path: str | Path, expected_sha256: str, label: str
) -> tuple[Path, str]:
    source = Path(path).expanduser().resolve()
    expected = require_sha256(expected_sha256, f"{label} SHA-256")
    if file_sha256(source) != expected:
        raise ValueError(f"{label} SHA-256 mismatch")
    return source, expected


def read_fold_manifest(
    path: str | Path,
    expected_sha256: str,
    *,
    fold: int,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], str, Path]:
    """Read only identity/fold/split; an outcome column is never parsed."""

    if int(fold) not in range(5):
        raise ValueError("fold must be 0..4")
    source, digest = _verify_source(path, expected_sha256, "fold manifest")
    frame = pd.read_csv(source, usecols=list(FOLD_USECOLS))
    if len(frame.columns) != len(FOLD_USECOLS) or set(frame.columns) != set(
        FOLD_USECOLS
    ):
        raise ValueError("fold parser materialized a column outside its allow-list")
    frame = frame.loc[:, list(FOLD_USECOLS)].copy()
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    frame["split"] = frame["split"].astype(str).str.strip().str.lower()
    if set(frame["fold"]) != set(range(5)):
        raise ValueError("fold manifest must contain exactly folds 0..4")
    if not set(frame["split"]).issubset({"train", "val", "test"}):
        raise ValueError("fold manifest contains an unknown split")
    patient_sets: list[set[str]] = []
    for current_fold in range(5):
        current = frame.loc[frame["fold"].eq(current_fold)]
        if current["patient_id"].duplicated().any():
            raise ValueError(f"fold {current_fold} contains duplicate patients")
        if set(current["split"]) != {"train", "val", "test"}:
            raise ValueError(f"fold {current_fold} must have nonempty train/val/test")
        patient_sets.append(set(current["patient_id"]))
    if not patient_sets[0] or any(
        values != patient_sets[0] for values in patient_sets[1:]
    ):
        raise ValueError("all folds must cover the same nonempty primary population")
    test_counts = (
        frame.assign(_test=frame["split"].eq("test"))
        .groupby("patient_id")["_test"]
        .sum()
    )
    if not test_counts.eq(1).all():
        raise ValueError("every primary patient must be test in exactly one fold")

    current = frame.loc[frame["fold"].eq(int(fold))]
    split_ids = {
        split: tuple(
            sorted(current.loc[current["split"].eq(split), "patient_id"].astype(str))
        )
        for split in ("train", "val", "test")
    }
    return split_ids["train"], split_ids["val"], split_ids["test"], digest, source


def read_external_authorization_manifest(
    path: str | Path,
    expected_sha256: str,
    *,
    has_eligibility_column: bool,
) -> tuple[tuple[str, ...], str, Path, int]:
    """Read a SHA-pinned private authorization input through an exact schema."""

    source, digest = _verify_source(
        path, expected_sha256, "external authorization manifest"
    )
    if not source.name.endswith(".private.csv"):
        raise ValueError("external authorization input must end in .private.csv")
    usecols = (
        EXTERNAL_ELIGIBILITY_USECOLS if has_eligibility_column else EXTERNAL_USECOLS
    )
    frame = pd.read_csv(source, usecols=list(usecols))
    if len(frame.columns) != len(usecols) or set(frame.columns) != set(usecols):
        raise ValueError(
            "external authorization parser materialized a non-allowlisted column"
        )
    frame = frame.loc[:, list(usecols)].copy()
    frame["patient_id"] = frame["patient_id"].astype(str)
    if frame.empty or frame["patient_id"].duplicated().any():
        raise ValueError("external authorization must be nonempty and patient-unique")
    source_rows = len(frame)
    if has_eligibility_column:
        frame = frame.loc[_boolean(frame["eligible"], "external eligibility")]
    patient_ids = tuple(sorted(frame["patient_id"].astype(str)))
    if not patient_ids:
        raise ValueError("external authorization selected no patients")
    if any(not value or value != value.strip() for value in patient_ids):
        raise ValueError("external authorization contains an invalid patient identity")
    return patient_ids, digest, source, source_rows


def _assert_aggregate_only(value: Any, path: str = "manifest") -> None:
    """Fail closed if a serializable payload contains a patient-level identity."""

    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).casefold().replace("-", "_")
            if normalized in {"patient_id", "patient_ids", "clinical_patient_id"}:
                raise ValueError(f"{path} contains a patient-identifier field")
            _assert_aggregate_only(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_aggregate_only(child, f"{path}[{index}]")
    elif isinstance(value, str) and _PATIENT_VALUE_RE.search(value):
        raise ValueError(f"{path} contains a patient identifier value")


def build_manifest(
    *,
    fold: int,
    outer_train_ids: Iterable[str],
    validation_ids: Iterable[str],
    test_ids: Iterable[str],
    external_ids: Iterable[str],
    ispy2_path: str | Path,
    ispy2_sha256: str,
    ispy1_path: str | Path,
    ispy1_sha256: str,
    fold_manifest_path: str | Path,
    fold_manifest_sha256: str,
    external_manifest_path: str | Path,
    external_manifest_sha256: str,
    external_manifest_source_rows: int,
    expected_primary_count: int | None = 808,
    expected_external_count: int | None = 139,
) -> dict[str, Any]:
    train = tuple(outer_train_ids)
    validation = tuple(validation_ids)
    test = tuple(test_ids)
    external = tuple(external_ids)
    primary = train + validation + test
    if len(set(primary)) != len(primary):
        raise ValueError("fold train/validation/test patient sets overlap")
    if set(primary) & set(external):
        raise ValueError(
            "authorized external train-only patients overlap primary folds"
        )
    if expected_primary_count is not None and len(primary) != int(
        expected_primary_count
    ):
        raise ValueError("primary patient count differs from the frozen contract")
    if expected_external_count is not None and len(external) != int(
        expected_external_count
    ):
        raise ValueError("external train-only count differs from the frozen contract")

    table = load_authorized_condition_table(
        primary_patient_ids=primary,
        authorized_external_train_only_patient_ids=external,
        ispy2_path=ispy2_path,
        ispy2_sha256=ispy2_sha256,
        ispy1_path=ispy1_path,
        ispy1_sha256=ispy1_sha256,
        expected_primary_patient_sha256=patient_set_sha256(primary),
        expected_external_patient_sha256=patient_set_sha256(external),
    )
    encoder = ConditionEncoder.fit(
        table,
        outer_train_patient_ids=train,
        authorized_external_train_only_patient_ids=external,
        expected_outer_train_patient_sha256=patient_set_sha256(train),
        expected_external_patient_sha256=patient_set_sha256(external),
        expected_fit_patient_sha256=patient_set_sha256(train + external),
    )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "private_aggregate_transition_condition_metadata",
        "privacy": {
            "aggregate_only": True,
            "contains_patient_identifiers": False,
            "contains_patient_level_rows": False,
            "safe_for_tracked_output": False,
        },
        "fold": int(fold),
        "population": {
            "primary_count": len(primary),
            "outer_train_count": len(train),
            "validation_count": len(validation),
            "test_count": len(test),
            "authorized_external_train_only_count": len(external),
            "age_fit_count": len(train) + len(external),
            "primary_patient_set_sha256": patient_set_sha256(primary),
            "outer_train_patient_set_sha256": patient_set_sha256(train),
            "validation_patient_set_sha256": patient_set_sha256(validation),
            "test_patient_set_sha256": patient_set_sha256(test),
            "external_patient_set_sha256": patient_set_sha256(external),
            "age_fit_patient_set_sha256": patient_set_sha256(train + external),
            "external_enters_validation_or_test": False,
        },
        "sources": {
            "clinical": table.aggregate_metadata(),
            "fold_manifest": {
                "path": str(Path(fold_manifest_path).resolve()),
                "sha256": require_sha256(fold_manifest_sha256, "fold manifest"),
                "csv_allowlist": list(FOLD_USECOLS),
            },
            "external_authorization_manifest": {
                "path": str(Path(external_manifest_path).resolve()),
                "sha256": require_sha256(
                    external_manifest_sha256, "external authorization manifest"
                ),
                "source_row_count": int(external_manifest_source_rows),
                "authorized_count": len(external),
            },
        },
        "condition": encoder.aggregate_metadata(),
        "assertions": {
            "training_is_pcr_free": True,
            "clinical_csv_read_is_usecols_restricted": True,
            "clinical_csv_allowlist_exact": list(CLINICAL_CSV_USECOLS),
            "clinical_csv_extra_columns_materialized": False,
            "arm_vocabulary_is_preregistered_not_fitted": True,
            "age_fit_uses_validation": False,
            "age_fit_uses_test": False,
            "age_fit_uses_unauthorized_external": False,
            "measured_elapsed_time_source_exists": False,
        },
    }
    _assert_aggregate_only(payload)
    return payload


def write_private_manifest(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path).expanduser().resolve()
    if not output.name.endswith(".private.json"):
        raise ValueError("condition metadata output must end in .private.json")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    _assert_aggregate_only(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--fold-manifest", type=Path, default=DEFAULT_FOLD_MANIFEST)
    parser.add_argument("--fold-manifest-sha256", default=DEFAULT_FOLD_MANIFEST_SHA256)
    parser.add_argument("--external-authorization-manifest", type=Path, required=True)
    parser.add_argument("--external-authorization-manifest-sha256", required=True)
    parser.add_argument(
        "--external-manifest-has-eligible-column",
        action="store_true",
        help="Select only eligible=true rows from a patient_id,eligible private input.",
    )
    parser.add_argument("--ispy2-clinical", type=Path, default=ISPY2_CLINICAL_TABLE)
    parser.add_argument("--ispy2-clinical-sha256", default=ISPY2_CLINICAL_SHA256)
    parser.add_argument("--ispy1-clinical", type=Path, default=ISPY1_CLINICAL_TABLE)
    parser.add_argument("--ispy1-clinical-sha256", default=ISPY1_CLINICAL_SHA256)
    parser.add_argument("--expected-primary-count", type=int, default=808)
    parser.add_argument("--expected-external-count", type=int, default=139)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    train, validation, test, fold_digest, fold_path = read_fold_manifest(
        args.fold_manifest, args.fold_manifest_sha256, fold=args.fold
    )
    external, external_digest, external_path, external_source_rows = (
        read_external_authorization_manifest(
            args.external_authorization_manifest,
            args.external_authorization_manifest_sha256,
            has_eligibility_column=args.external_manifest_has_eligible_column,
        )
    )
    payload = build_manifest(
        fold=args.fold,
        outer_train_ids=train,
        validation_ids=validation,
        test_ids=test,
        external_ids=external,
        ispy2_path=args.ispy2_clinical,
        ispy2_sha256=args.ispy2_clinical_sha256,
        ispy1_path=args.ispy1_clinical,
        ispy1_sha256=args.ispy1_clinical_sha256,
        fold_manifest_path=fold_path,
        fold_manifest_sha256=fold_digest,
        external_manifest_path=external_path,
        external_manifest_sha256=external_digest,
        external_manifest_source_rows=external_source_rows,
        expected_primary_count=args.expected_primary_count,
        expected_external_count=args.expected_external_count,
    )
    return write_private_manifest(args.output, payload)


if __name__ == "__main__":
    main()

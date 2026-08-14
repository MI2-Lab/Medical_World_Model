#!/usr/bin/env python3
"""Project locked clinical sources into the private Goal-F training profile.

This is the only representation-stage utility allowed to open the source
clinical CSVs.  It verifies every input hash before parsing, requests exactly
the five preregistered non-outcome fields, and publishes only aggregate counts
and cryptographic digests.  The row-level result is always a ``.private.csv``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping

import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from crps.contracts import (  # noqa: E402
    EXPECTED_TRAINING_PROFILE_PATIENTS,
    LOCKED_ISPY1_PROFILE_SHA256,
    LOCKED_ISPY2_PROFILE_SHA256,
    LOCKED_TECHNICAL_ELIGIBILITY_SHA256,
    PROFILE_SOURCE_USECOLS,
    TECHNICAL_ELIGIBILITY_USECOLS,
    TRAINING_PROFILE_COLUMNS,
    assert_representation_config,
    canonical_sha256,
    file_sha256,
    load_json,
)


DEFAULT_ISPY2_SOURCE = Path(
    "/data/data/Preprocessed/I-SPY2/clinical_labels_complete4visits.csv"
)
DEFAULT_ISPY1_SOURCE = Path(
    "/data/data/Preprocessed/I-SPY1/clinical_labels_complete4visits.csv"
)
DEFAULT_TECHNICAL_ELIGIBILITY = (
    REPO_ROOT
    / "additional_experiments"
    / "c1b_overlap_eligibility_ftv_stageb"
    / "manifests"
    / "technical_eligibility_patients.private.csv"
)
DEFAULT_PRIVATE_OUTPUT = EXPERIMENT_ROOT / "manifests" / "training_profiles.private.csv"
DEFAULT_PUBLIC_PROVENANCE = (
    EXPERIMENT_ROOT / "manifests" / "training_profiles_provenance.json"
)
DEFAULT_REPRESENTATION_CONFIG = EXPERIMENT_ROOT / "configs" / "representation.json"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
UNSET_MANIFEST_HASHES = (None, "", "PENDING", "PENDING_PRIVATE_PROJECTION")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ispy2-source", type=Path, default=DEFAULT_ISPY2_SOURCE)
    parser.add_argument(
        "--ispy2-sha256", default=LOCKED_ISPY2_PROFILE_SHA256
    )
    parser.add_argument("--ispy1-source", type=Path, default=DEFAULT_ISPY1_SOURCE)
    parser.add_argument(
        "--ispy1-sha256", default=LOCKED_ISPY1_PROFILE_SHA256
    )
    parser.add_argument(
        "--technical-eligibility",
        type=Path,
        default=DEFAULT_TECHNICAL_ELIGIBILITY,
    )
    parser.add_argument(
        "--technical-eligibility-sha256",
        default=LOCKED_TECHNICAL_ELIGIBILITY_SHA256,
    )
    parser.add_argument(
        "--expected-patient-count",
        type=int,
        default=EXPECTED_TRAINING_PROFILE_PATIENTS,
    )
    parser.add_argument("--output-private", type=Path, default=DEFAULT_PRIVATE_OUTPUT)
    parser.add_argument(
        "--public-provenance", type=Path, default=DEFAULT_PUBLIC_PROVENANCE
    )
    parser.add_argument(
        "--representation-config", type=Path, default=DEFAULT_REPRESENTATION_CONFIG
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _require_sha256(value: str, label: str) -> str:
    normalized = str(value).strip().casefold()
    if SHA256_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"{label} must be a lowercase 64-character SHA-256")
    return normalized


def _verify_file(path: str | Path, expected_sha256: str, label: str) -> tuple[Path, str]:
    source = Path(path).expanduser().resolve()
    expected = _require_sha256(expected_sha256, f"{label} expected digest")
    actual = file_sha256(source)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch")
    return source, actual


def _binary(series: pd.Series, label: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="raise")
    if numeric.isna().any() or not numeric.isin((0, 1)).all():
        raise ValueError(f"{label} must be complete binary 0/1")
    return numeric.astype("int64")


def _boolean(series: pd.Series, label: str) -> pd.Series:
    normalized = series.astype("string").str.strip().str.casefold()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    parsed = normalized.map(mapping)
    if parsed.isna().any():
        raise ValueError(f"{label} must contain only true/false or 1/0")
    return parsed.astype(bool)


def _read_verified_profile_source(source: Path, label: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(
            source,
            usecols=PROFILE_SOURCE_USECOLS,
            dtype={"patient_id": "string", "arm": "string"},
        )
    except ValueError as error:
        raise PermissionError(f"{label} lacks the exact profile projection fields") from error
    frame = frame.loc[:, list(TRAINING_PROFILE_COLUMNS)].copy()
    frame["patient_id"] = frame["patient_id"].str.strip()
    frame["arm"] = frame["arm"].str.strip()
    if frame.empty or frame["patient_id"].isna().any() or frame["arm"].isna().any():
        raise ValueError(f"{label} profiles must be nonempty and complete")
    if frame["patient_id"].eq("").any() or frame["arm"].eq("").any():
        raise ValueError(f"{label} patient IDs and treatment arms must be nonempty")
    if frame["patient_id"].duplicated().any():
        raise ValueError(f"{label} contains duplicate patient IDs")
    for column in ("label_hr", "label_her2", "label_mp"):
        frame[column] = _binary(frame[column], column)
    return frame


def _read_verified_eligibility(source: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(
            source,
            usecols=TECHNICAL_ELIGIBILITY_USECOLS,
            dtype={"patient_id": "string"},
        )
    except ValueError as error:
        raise PermissionError("technical eligibility lacks its exact projection fields") from error
    frame = frame.loc[:, list(TECHNICAL_ELIGIBILITY_USECOLS)].copy()
    frame["patient_id"] = frame["patient_id"].str.strip()
    if frame.empty or frame["patient_id"].isna().any() or frame["patient_id"].eq("").any():
        raise ValueError("technical eligibility must contain nonempty patient IDs")
    if frame["patient_id"].duplicated().any():
        raise ValueError("technical eligibility contains duplicate patient IDs")
    frame["eligible"] = _boolean(frame["eligible"], "technical eligibility")
    return frame


def project_training_profiles(
    *,
    ispy2_source: str | Path,
    ispy2_sha256: str,
    ispy1_source: str | Path,
    ispy1_sha256: str,
    technical_eligibility: str | Path,
    technical_eligibility_sha256: str,
    expected_patient_count: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return the private projection and identifier-free public provenance.

    Hash verification for all three inputs finishes before the first CSV is
    parsed.  Callers therefore cannot accidentally process a partially locked
    set of sources.
    """

    ispy2_path, ispy2_actual = _verify_file(
        ispy2_source, ispy2_sha256, "I-SPY2 clinical source"
    )
    ispy1_path, ispy1_actual = _verify_file(
        ispy1_source, ispy1_sha256, "I-SPY1 clinical source"
    )
    eligibility_path, eligibility_actual = _verify_file(
        technical_eligibility,
        technical_eligibility_sha256,
        "technical eligibility manifest",
    )
    if not eligibility_path.name.endswith(".private.csv"):
        raise ValueError("technical eligibility must be an owner-private CSV")

    ispy2 = _read_verified_profile_source(ispy2_path, "I-SPY2")
    ispy1 = _read_verified_profile_source(ispy1_path, "I-SPY1")
    combined = pd.concat((ispy2, ispy1), ignore_index=True)
    if combined["patient_id"].duplicated().any():
        raise ValueError("clinical sources contain duplicate patient IDs across cohorts")

    eligibility = _read_verified_eligibility(eligibility_path)
    selected_ids = set(eligibility.loc[eligibility["eligible"], "patient_id"])
    if len(selected_ids) != int(expected_patient_count):
        raise ValueError(
            "technical eligibility does not select the preregistered patient count"
        )
    source_ids = set(combined["patient_id"])
    if missing := selected_ids.difference(source_ids):
        raise ValueError(
            f"technical eligibility selects {len(missing)} patients absent from locked sources"
        )

    projected = combined.loc[combined["patient_id"].isin(selected_ids)].copy()
    projected = projected.sort_values("patient_id", kind="stable").reset_index(drop=True)
    projected = projected.loc[:, list(TRAINING_PROFILE_COLUMNS)]
    if len(projected) != int(expected_patient_count):
        raise AssertionError("profile projection changed eligible population cardinality")
    if set(projected["patient_id"]) != selected_ids:
        raise AssertionError("profile projection changed eligible population membership")

    provenance: dict[str, Any] = {
        "schema_version": 1,
        "source_file_count": 2,
        "source_row_counts": {
            "ispy2": int(len(ispy2)),
            "ispy1": int(len(ispy1)),
        },
        "source_sha256": {
            "ispy2": ispy2_actual,
            "ispy1": ispy1_actual,
        },
        "projected_source_counts": {
            "ispy2": int(ispy2["patient_id"].isin(selected_ids).sum()),
            "ispy1": int(ispy1["patient_id"].isin(selected_ids).sum()),
        },
        "eligibility_manifest_sha256": eligibility_actual,
        "eligibility_candidate_count": int(len(eligibility)),
        "eligibility_selected_count": int(len(selected_ids)),
        "eligibility_excluded_count": int((~eligibility["eligible"]).sum()),
        "projected_patient_count": int(len(projected)),
        "projected_patient_order_sha256": canonical_sha256(
            tuple(projected["patient_id"])
        ),
    }
    return projected, provenance


def _atomic_write(path: Path, payload: bytes, *, mode: int, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_projection(
    profiles: pd.DataFrame,
    public_provenance: Mapping[str, Any],
    *,
    private_output: str | Path,
    provenance_output: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    private_path = Path(private_output).expanduser().resolve()
    public_path = Path(provenance_output).expanduser().resolve()
    if not private_path.name.endswith(".private.csv"):
        raise ValueError("formal training profiles must use the .private.csv suffix")
    if public_path.suffix.casefold() != ".json" or ".private." in public_path.name:
        raise ValueError("public provenance must be a non-private JSON artifact")
    if tuple(profiles.columns) != TRAINING_PROFILE_COLUMNS:
        raise PermissionError("refusing to write a non-allowlisted profile schema")
    if private_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {private_path}")
    if public_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {public_path}")

    csv_payload = _profile_csv_payload(profiles)
    _atomic_write(private_path, csv_payload, mode=0o600, overwrite=overwrite)

    provenance = dict(public_provenance)
    provenance["private_manifest_sha256"] = file_sha256(private_path)
    json_payload = (json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    _atomic_write(public_path, json_payload, mode=0o644, overwrite=overwrite)
    return provenance


def _profile_csv_payload(profiles: pd.DataFrame) -> bytes:
    if tuple(profiles.columns) != TRAINING_PROFILE_COLUMNS:
        raise PermissionError("refusing to serialize a non-allowlisted profile schema")
    csv_payload = profiles.to_csv(
        index=False, columns=list(TRAINING_PROFILE_COLUMNS)
    ).encode("utf-8")
    expected_header = (",".join(TRAINING_PROFILE_COLUMNS) + "\n").encode("utf-8")
    if not csv_payload.startswith(expected_header):
        raise AssertionError("private profile serialization changed its exact schema")
    return csv_payload


def _configured_manifest_path(config: Mapping[str, Any]) -> Path:
    configured = Path(str(config["profiles"]["training_manifest_path"])).expanduser()
    if configured.is_absolute():
        return configured.resolve()
    return (REPO_ROOT / configured).resolve()


def bind_training_manifest_sha256(
    representation_config: str | Path,
    private_manifest: str | Path,
    manifest_sha256: str,
    *,
    commit: bool,
) -> dict[str, Any]:
    """Validate and optionally atomically bind the private manifest digest."""

    config_path = Path(representation_config).expanduser().resolve()
    config = load_json(config_path)
    assert_representation_config(config)
    private_path = Path(private_manifest).expanduser().resolve()
    if _configured_manifest_path(config) != private_path:
        raise PermissionError("private output does not match the configured training manifest")
    expected = _require_sha256(manifest_sha256, "training manifest digest")
    current = config["profiles"].get("training_manifest_sha256")
    if current not in UNSET_MANIFEST_HASHES:
        current_digest = _require_sha256(str(current), "configured training manifest digest")
        if current_digest != expected:
            raise PermissionError("configured training manifest SHA-256 conflicts with projection")
        if commit and (
            not private_path.is_file() or file_sha256(private_path) != expected
        ):
            raise ValueError("bound private training manifest is absent or changed")
        return config
    if not commit:
        return config
    if not private_path.is_file() or file_sha256(private_path) != expected:
        raise ValueError("private training manifest is absent or changed before config binding")
    config["profiles"]["training_manifest_sha256"] = expected
    assert_representation_config(config)
    payload = (json.dumps(config, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    _atomic_write(config_path, payload, mode=0o644, overwrite=True)
    return config


def write_projection_and_bind(
    profiles: pd.DataFrame,
    public_provenance: Mapping[str, Any],
    *,
    private_output: str | Path,
    provenance_output: str | Path,
    representation_config: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write projection artifacts and seal their digest into the prereg config."""

    private_path = Path(private_output).expanduser().resolve()
    prospective_sha256 = hashlib.sha256(_profile_csv_payload(profiles)).hexdigest()
    config = bind_training_manifest_sha256(
        representation_config,
        private_path,
        prospective_sha256,
        commit=False,
    )
    expected_count = int(config["profiles"]["expected_patient_count"])
    if len(profiles) != expected_count:
        raise ValueError("private projection does not have the configured patient count")
    if int(public_provenance.get("projected_patient_count", -1)) != expected_count:
        raise ValueError("public projection count disagrees with the private manifest")
    expected_sources = {
        "ispy2": config["profiles"]["ispy2_sha256"],
        "ispy1": config["profiles"]["ispy1_sha256"],
    }
    if public_provenance.get("source_sha256") != expected_sources:
        raise PermissionError("projection provenance does not contain the locked source hashes")
    if (
        public_provenance.get("eligibility_manifest_sha256")
        != config["profiles"]["technical_eligibility_sha256"]
    ):
        raise PermissionError("projection provenance has the wrong eligibility hash")
    written = write_projection(
        profiles,
        public_provenance,
        private_output=private_path,
        provenance_output=provenance_output,
        overwrite=overwrite,
    )
    if written["private_manifest_sha256"] != prospective_sha256:
        raise AssertionError("written private manifest digest differs from serialization digest")
    bind_training_manifest_sha256(
        representation_config,
        private_path,
        prospective_sha256,
        commit=True,
    )
    return written


def _assert_preregistered_locks(args: argparse.Namespace) -> None:
    observed = (
        _require_sha256(args.ispy2_sha256, "I-SPY2 expected digest"),
        _require_sha256(args.ispy1_sha256, "I-SPY1 expected digest"),
        _require_sha256(
            args.technical_eligibility_sha256,
            "technical eligibility expected digest",
        ),
        int(args.expected_patient_count),
    )
    expected = (
        LOCKED_ISPY2_PROFILE_SHA256,
        LOCKED_ISPY1_PROFILE_SHA256,
        LOCKED_TECHNICAL_ELIGIBILITY_SHA256,
        EXPECTED_TRAINING_PROFILE_PATIENTS,
    )
    if observed != expected:
        raise PermissionError("formal projection must use the preregistered hashes/count")


def main() -> None:
    args = parse_args()
    _assert_preregistered_locks(args)
    profiles, provenance = project_training_profiles(
        ispy2_source=args.ispy2_source,
        ispy2_sha256=args.ispy2_sha256,
        ispy1_source=args.ispy1_source,
        ispy1_sha256=args.ispy1_sha256,
        technical_eligibility=args.technical_eligibility,
        technical_eligibility_sha256=args.technical_eligibility_sha256,
        expected_patient_count=args.expected_patient_count,
    )
    written = write_projection_and_bind(
        profiles,
        provenance,
        private_output=args.output_private,
        provenance_output=args.public_provenance,
        representation_config=args.representation_config,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "projected_patient_count": written["projected_patient_count"],
                "private_manifest_sha256": written["private_manifest_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

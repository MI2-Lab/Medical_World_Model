#!/usr/bin/env python3
"""Close the preregistered Stage-A gate and emit the sole Stage-B sentinel.

This script is intentionally fail closed.  Missing, partial, smoke-scope, or
identifier-bearing public artifacts cannot produce ``STAGE_A_GO.json``.
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
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
from audit_public_artifacts import scan_public_artifacts


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from c1b_sanity.builder import builder_contract_sha256  # noqa: E402
from c1b_sanity.cache import _REQUIRED_KEYS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(relative: str) -> tuple[Path, dict[str, Any]]:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(f"required Stage-A artifact is missing: {relative}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Stage-A JSON is not an object: {relative}")
    return path, payload


def atomic_text(path: Path, content: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "pass"}


def is_sha256(value: Any) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", str(value).strip().lower()))


def cache_contract_matches(
    patient_id: Any,
    cache_path: Any,
    *,
    cohort: str,
    strategy: str,
    formal_ftv_overlap: bool,
) -> bool:
    """Bind each row to the exact schema-3 identity/provenance envelope."""

    expected = str(patient_id)
    path = Path(str(cache_path))
    expected_name = f"{hashlib.sha256(expected.encode('utf-8')).hexdigest()}.npz"
    if not path.is_file() or path.name != expected_name:
        return False
    try:
        # NpzFile members are lazy: this reads only the tiny scalar identity
        # member, not the roughly 337-MiB model tensor.
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(_REQUIRED_KEYS):
                return False
            stored = np.asarray(archive["patient_id"])
            schema = np.asarray(archive["schema_version"])
            source_hashes = np.asarray(archive["source_canonical_sha256"])
            support_hashes = np.asarray(archive["support_canonical_sha256"])
            phase_hashes = np.asarray(archive["phase_metadata_sha256"])
            support_scope = np.asarray(archive["support_scope"])
            support_available = np.asarray(archive["support_available"])
            embedded_formal = np.asarray(archive["formal_ftv_overlap"])
            return bool(
                stored.shape == ()
                and str(stored.item()) == expected
                and schema.shape == ()
                and schema.dtype.kind in "iu"
                and int(schema.item()) == 3
                and str(np.asarray(archive["cohort"]).item()) == str(cohort)
                and str(np.asarray(archive["registration_strategy"]).item())
                == f"C1B-{strategy}"
                and embedded_formal.shape == ()
                and embedded_formal.dtype == np.dtype(np.uint8)
                and bool(embedded_formal.item()) == bool(formal_ftv_overlap)
                and source_hashes.shape == (4,)
                and all(is_sha256(value) for value in source_hashes.tolist())
                and support_hashes.shape == (4,)
                and all(
                    str(value) == "NONE" or is_sha256(value)
                    for value in support_hashes.tolist()
                )
                and phase_hashes.shape == (4,)
                and all(is_sha256(value) for value in phase_hashes.tolist())
                and support_scope.shape == (4,)
                and str(np.asarray(archive["builder_contract_sha256"]).item())
                == builder_contract_sha256()
                and is_sha256(np.asarray(archive["input_provenance_sha256"]).item())
                and support_available.shape == (4,)
                and np.isin(support_available, (0, 1)).all()
                and (
                    support_available.astype(bool).all()
                    if formal_ftv_overlap
                    else not support_available[1:].astype(bool).any()
                )
            )
    except (OSError, KeyError, TypeError, ValueError):
        return False


def check(name: str, passed: bool, observed: Any, requirement: str) -> dict[str, Any]:
    return {
        "gate": name,
        "status": "PASS" if bool(passed) else "FAIL",
        "observed": observed,
        "requirement": requirement,
    }


def _validated_catastrophic_overlap_failure(
    *, root: Path = ROOT
) -> tuple[Path, dict[str, Any], Path, Path, Path, Path] | None:
    """Recognize only the complete frozen-population zero-overlap evidence."""

    candidates = sorted(
        (root / "metrics").glob("model_input_pipeline_?_validation_gate.json")
    )
    failures: list[tuple[Path, dict[str, Any], Path, Path, Path, Path]] = []
    registration_path = root / "metrics/registration_strategy_decision.json"
    if not registration_path.is_file():
        return None
    try:
        registration = json.loads(registration_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if (
        not isinstance(registration, dict)
        or not as_bool(registration.get("decision_frozen"))
        or str(registration.get("chosen_strategy", "")).upper() not in {"H", "R"}
    ):
        return None
    required = {
        "schema_version",
        "cache_schema_version",
        "builder_contract_sha256",
        "strategy",
        "scope",
        "status",
        "failure_stage",
        "failure_codes",
        "frozen_population_patients",
        "frozen_population_visits",
        "header_audit_rows",
        "header_audit_exact_population_coverage",
        "selected_patients",
        "selected_visits",
        "zero_valid_source_overlap_visits",
        "minimum_valid_source_voxels",
        "atomic_caches_present_for_selected_patients",
        "cache_contract_completed",
        "full_scope_cache_build_forbidden",
        "stage_b_authorized",
        "thresholds_relaxed",
        "private_evidence_sha256",
        "contains_patient_identifiers",
    }
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or not required.issubset(payload):
            continue
        strategy = str(payload.get("strategy", "")).removeprefix("C1B-").lower()
        if strategy not in {"h", "r"}:
            continue
        if strategy.upper() != str(registration["chosen_strategy"]).upper():
            continue
        failure_path = (
            root
            / "metrics"
            / f"model_input_pipeline_{strategy}_validation_failures.private.csv"
        )
        selection_path = root / "manifests/cache_validation_selection.private.csv"
        audit_path = root / "metrics/orientation_resampling_patient_visit.private.csv"
        if not all(item.is_file() for item in (failure_path, selection_path, audit_path)):
            continue
        hashes = payload.get("private_evidence_sha256", {})
        if not isinstance(hashes, dict) or hashes != {
            "failure_table": sha256(failure_path),
            "selection_manifest": sha256(selection_path),
            "complete_header_audit": sha256(audit_path),
        }:
            continue
        try:
            failure_rows = pd.read_csv(
                failure_path,
                usecols=[
                    "patient_id",
                    "cohort",
                    "visit",
                    "padding_fraction_bbox",
                    "failure_code",
                    "valid_source_voxels",
                    "target_grid_voxels",
                    "evidence",
                ],
            )
            selection = pd.read_csv(
                selection_path,
                usecols=[
                    "patient_id",
                    "cohort",
                    "selected_for_validation",
                    "selection_reasons",
                    "cache_path",
                ],
            )
            audit = pd.read_csv(
                audit_path,
                usecols=["patient_id", "visit", "cohort", "padding_fraction_bbox"],
            )
        except (OSError, TypeError, ValueError):
            continue
        audit_zero = audit.loc[audit["padding_fraction_bbox"].eq(1.0)]
        failure_keys = set(
            zip(
                failure_rows["patient_id"].astype(str),
                failure_rows["visit"].astype(str),
                strict=True,
            )
        )
        audit_zero_keys = set(
            zip(
                audit_zero["patient_id"].astype(str),
                audit_zero["visit"].astype(str),
                strict=True,
            )
        )
        valid = bool(
            int(payload["schema_version"]) == 1
            and int(payload["cache_schema_version"]) == 3
            and str(payload["builder_contract_sha256"])
            == builder_contract_sha256()
            and payload["scope"] == "validation"
            and payload["status"] == "FAIL"
            and payload["failure_stage"]
            == "catastrophic_source_overlap_preflight"
            and payload["failure_codes"]
            == {"ZERO_VALID_SOURCE_OVERLAP": 1}
            and int(payload["frozen_population_patients"]) == 948
            and int(payload["frozen_population_visits"]) == 3792
            and int(payload["header_audit_rows"]) == 3792
            and as_bool(payload["header_audit_exact_population_coverage"])
            and int(payload["selected_patients"]) == 263
            and int(payload["selected_visits"]) == 1052
            and int(payload["zero_valid_source_overlap_visits"]) == 1
            and int(payload["minimum_valid_source_voxels"]) == 0
            and 0 <= int(payload["atomic_caches_present_for_selected_patients"]) < 263
            and not as_bool(payload["cache_contract_completed"])
            and as_bool(payload["full_scope_cache_build_forbidden"])
            and not as_bool(payload["stage_b_authorized"])
            and not as_bool(payload["thresholds_relaxed"])
            and not as_bool(payload["contains_patient_identifiers"])
            and len(audit) == 3792
            and audit["patient_id"].astype(str).nunique() == 948
            and not audit.duplicated(["patient_id", "visit"]).any()
            and len(audit_zero) == 1
            and len(failure_rows) == 1
            and not failure_rows.duplicated(["patient_id", "visit"]).any()
            and failure_keys == audit_zero_keys
            and failure_rows["failure_code"].eq("ZERO_VALID_SOURCE_OVERLAP").all()
            and failure_rows["padding_fraction_bbox"].eq(1.0).all()
            and failure_rows["valid_source_voxels"].eq(0).all()
            and failure_rows["target_grid_voxels"].eq(112 * 176 * 160).all()
            and failure_rows["evidence"].eq(
                "header_padding_fraction_bbox_eq_1_and_builder_zero_valid_source"
            ).all()
            and len(selection) == 263
            and selection["patient_id"].astype(str).nunique() == 263
            and selection["selected_for_validation"].map(as_bool).all()
            and failure_keys.issubset(
                set(
                    (str(value), visit)
                    for value in selection["patient_id"]
                    for visit in ("T0", "T1", "T2", "T3")
                )
            )
        )
        if valid:
            failures.append(
                (
                    path,
                    payload,
                    failure_path,
                    selection_path,
                    audit_path,
                    registration_path,
                )
            )
    if len(failures) > 1:
        raise ValueError("Multiple catastrophic cache-failure artifacts are ambiguous")
    return failures[0] if failures else None


def _close_catastrophic_overlap_no_go(
    evidence: tuple[Path, dict[str, Any], Path, Path, Path, Path],
    *,
    overwrite: bool,
    root: Path = ROOT,
) -> None:
    (
        gate_path,
        cache_failure,
        failure_path,
        selection_path,
        audit_path,
        registration_path,
    ) = evidence
    strategy = str(cache_failure["strategy"])
    cache_count = int(cache_failure["atomic_caches_present_for_selected_patients"])
    evidence_relatives = (
        "metrics/dicom_pixel_rebuild_gate.json",
        "metrics/orientation_validation_gate.json",
        "metrics/registration_strategy_decision.json",
        "metrics/registration_physical_support_summary.json",
        "metrics/support_containment_h_summary.json",
        "metrics/ispy1_base_eligibility_summary.json",
        "manifests/grounding_observability_summary.json",
    )
    public_evidence: dict[str, tuple[Path, dict[str, Any]]] = {}
    for relative in evidence_relatives:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"required public Stage-A evidence is missing: {relative}"
            )
        item = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(item, dict):
            raise ValueError(f"public Stage-A evidence is not an object: {relative}")
        public_evidence[relative] = (path, item)
    dicom = public_evidence["metrics/dicom_pixel_rebuild_gate.json"][1]
    orientation = public_evidence["metrics/orientation_validation_gate.json"][1]
    registration = public_evidence["metrics/registration_strategy_decision.json"][1]
    registration_support = public_evidence[
        "metrics/registration_physical_support_summary.json"
    ][1]
    support = public_evidence["metrics/support_containment_h_summary.json"][1]
    source = public_evidence["metrics/ispy1_base_eligibility_summary.json"][1]
    observable = public_evidence[
        "manifests/grounding_observability_summary.json"
    ][1]
    repairs = dicom.get("all_model_input_singular_visits", {})
    gates = [
        check(
            "repaired_dicom_pixel_geometry",
            dicom.get("status") == "PASS"
            and int(repairs.get("visits", -1)) == 146
            and int(repairs.get("passed", -1)) == 146
            and int(repairs.get("failed", -1)) == 0
            and float(repairs.get("max_cell_error", 1.0)) == 0.0,
            {
                "repaired_visits": repairs.get("passed"),
                "verified_cells": repairs.get("verified_cells"),
                "max_cell_error": repairs.get("max_cell_error"),
            },
            "146/146 singular model-input visits pass exact pixel/geometry rebuild",
        ),
        check(
            "true_canonical_orientation",
            float(orientation.get("canonical_ras_fraction", -1)) == 1.0
            and int(orientation.get("model_input_patients", -1)) == 948
            and int(orientation.get("model_input_visits", -1)) == 3792
            and as_bool(orientation.get("array_reordering_implemented_and_unit_tested"))
            and not as_bool(orientation.get("header_only_label_change", True)),
            {
                "patients": orientation.get("model_input_patients"),
                "visits": orientation.get("model_input_visits"),
                "canonical_ras_fraction": orientation.get("canonical_ras_fraction"),
            },
            "948x4 true RAS+ array reorientation, not header-only relabeling",
        ),
        check(
            "registration_success_and_strategy",
            as_bool(registration.get("decision_frozen"))
            and str(registration.get("chosen_strategy")) == "H"
            and int(registration.get("formal_registration_pairs", -1)) == 1125
            and int(registration.get("registration_success_pairs", -1)) == 858
            and int(registration.get("registration_failure_pairs", -1)) == 267
            and as_bool(registration.get("manual_review_pass")),
            {
                "chosen_strategy": registration.get("chosen_strategy"),
                "successful_pairs": registration.get("registration_success_pairs"),
                "formal_pairs": registration.get("formal_registration_pairs"),
                "success_rate": registration.get("registration_success_rate"),
            },
            "complete 1,125-pair sensitivity audit and uniquely frozen C1B-H decision",
        ),
        check(
            "formal_support_containment",
            str(support.get("strategy")) == "C1B-H"
            and int(support.get("formal_visits", -1)) == 1500
            and float(support.get("exact_full_support_containment_rate", -1)) >= 0.95,
            {
                "formal_visits": support.get("formal_visits"),
                "exact_containment_rate": support.get(
                    "exact_full_support_containment_rate"
                ),
            },
            ">=0.95 exact containment over all 1,500 formal visits",
        ),
        check(
            "formal_ftv_retention_q05",
            float(support.get("physical_volume_retention", {}).get("q05", -1))
            >= 0.95
            and float(registration_support.get("c1b_h_ftv_retention_q05", -1))
            >= 0.95,
            {
                "q05": support.get("physical_volume_retention", {}).get("q05"),
                "posthoc_registration_audit_q05": registration_support.get(
                    "c1b_h_ftv_retention_q05"
                ),
            },
            ">=0.95 physical FTV volume retention at Q05",
        ),
        check(
            "resampling_and_source_overlap",
            False,
            {
                "failure_code": "ZERO_VALID_SOURCE_OVERLAP",
                "failed_visits": 1,
                "minimum_valid_source_voxels": 0,
                "header_audit_rows": 3792,
                "exact_frozen_population_coverage": True,
            },
            "all frozen visits must have at least one valid source voxel",
        ),
        check(
            "complete_dce7_builder_and_cache",
            False,
            {
                "atomic_caches_present": cache_count,
                "validation_patients": 263,
                "cohort_cache_contract_completed": False,
            },
            "263/263 validation caches and complete schema-3 cohort contract",
        ),
        check(
            "leakage_exclusion_contract",
            source.get("outcome_fields_read") == []
            and source.get("clinical_or_outcome_tables_read") == []
            and not as_bool(source.get("public_artifact_contains_identifiers_or_paths", True))
            and not as_bool(cache_failure.get("contains_patient_identifiers", True)),
            {
                "outcome_fields_read": source.get("outcome_fields_read"),
                "clinical_or_outcome_tables_read": source.get(
                    "clinical_or_outcome_tables_read"
                ),
                "public_identifiers": False,
            },
            "no clinical, treatment, pCR, LD, identifier, or path leakage",
        ),
        check(
            "geometry_and_mask_scope_contract",
            as_bool(support.get("anchor_uses_t0_only"))
            and not as_bool(support.get("future_support_used_for_grid", True))
            and not as_bool(observable.get("is_model_input", True))
            and as_bool(observable.get("does_not_filter_base_training"))
            and int(observable.get("formal_visits", -1)) == 1500,
            {
                "anchor_uses_t0_only": support.get("anchor_uses_t0_only"),
                "future_support_used_for_grid": support.get(
                    "future_support_used_for_grid"
                ),
                "grounding_mask_is_model_input": observable.get("is_model_input"),
                "grounding_observable_visits": observable.get("observable_visits"),
            },
            "T0-only anchor; future support excluded; grounding mask is loss-only",
        ),
        check(
            "stage_a_model_ready",
            False,
            {"frozen_patients": 948, "frozen_visits": 3792},
            "every Stage-A hard gate must pass without eligibility changes",
        ),
    ]
    payload = {
        "schema_version": 1,
        "stage": "A",
        "status": "NO-GO",
        "chosen_input_strategy": strategy,
        "failure_code": "ZERO_VALID_SOURCE_OVERLAP",
        "thresholds_relaxed": False,
        "frozen_population_changed": False,
        "stage_b_authorized": False,
        "gates": gates,
        "provenance_sha256": {
            str(gate_path.relative_to(root)): sha256(gate_path),
            str(failure_path.relative_to(root)): sha256(failure_path),
            str(selection_path.relative_to(root)): sha256(selection_path),
            str(audit_path.relative_to(root)): sha256(audit_path),
            str(registration_path.relative_to(root)): sha256(registration_path),
            **{
                str(path.relative_to(root)): sha256(path)
                for path, _payload in public_evidence.values()
            },
        },
        "contains_patient_identifiers": False,
    }
    gate_json = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    atomic_text(root / "metrics/stage_a_model_ready_gate.json", gate_json, overwrite)
    atomic_text(
        root / "metrics/table1_model_ready_preprocessing_qc.csv",
        pd.DataFrame(gates).to_csv(index=False),
        overwrite,
    )
    rows = "\n".join(
        f"| {row['gate']} | {row['status']} | `{json.dumps(row['observed'], ensure_ascii=False, sort_keys=True)}` | {row['requirement']} |"
        for row in gates
    )
    report = f"""# Stage A C1B model-ready hard gate

## 结论

`STAGE_A = NO-GO`。完整 948 人、3792 visit header audit发现1个冻结visit为 `ZERO_VALID_SOURCE_OVERLAP`（valid-source voxel = 0）。{cache_count}/263个validation cache虽已原子完成，但cohort-level schema-3 contract未闭合；不改变人口、不放宽门槛，禁止启动Stage B。

| 子门 | 状态 | 观测 | 冻结要求 |
|---|---|---|---|
{rows}

公开产物只含聚合计数和私有证据SHA-256，不含patient identifier或路径。
"""
    atomic_text(root / "reports/stage_a_gate_report.md", report, overwrite)
    go_path = root / "STAGE_A_GO.json"
    if go_path.exists():
        go_path.unlink()
    atomic_text(root / "STAGE_A_NO_GO.json", gate_json, overwrite)
    print(gate_json, end="")


def main() -> None:
    args = parse_args()
    catastrophic = _validated_catastrophic_overlap_failure()
    if catastrophic is not None:
        # A complete frozen-population zero-overlap proof is terminal.  It is
        # the only early path; partial/malformed evidence falls through to the
        # ordinary fail-closed finalizer, where missing all-scope proof raises.
        _close_catastrophic_overlap_no_go(catastrophic, overwrite=args.overwrite)
        return
    dicom_path, dicom = load_json("metrics/dicom_pixel_rebuild_gate.json")
    orientation_path, orientation = load_json("metrics/orientation_validation_gate.json")
    support_h_path, support_h = load_json("metrics/support_containment_h_summary.json")
    registration_path, registration = load_json("metrics/registration_strategy_decision.json")
    registration_summary_path, registration_summary = load_json(
        "metrics/registration_sensitivity_summary.json"
    )
    registration_support_path, registration_support = load_json(
        "metrics/registration_physical_support_summary.json"
    )
    registration_residual_path, registration_residual = load_json(
        "metrics/registration_residual_audit.json"
    )
    registration_pairs_path = ROOT / "metrics/registration_sensitivity_pairs.private.csv"
    registration_support_private_path = (
        ROOT / "metrics/registration_support_patient_visit.private.csv"
    )
    registration_residual_private_path = (
        ROOT / "metrics/registration_residuals.private.csv"
    )
    for path in (
        registration_pairs_path,
        registration_support_private_path,
        registration_residual_private_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"complete private registration audit is missing: {path.name}"
            )
    registration_pairs = pd.read_csv(
        registration_pairs_path,
        usecols=["patient_id", "visit", "success"],
    )
    registration_support_private = pd.read_csv(
        registration_support_private_path,
        usecols=[
            "patient_id",
            "visit",
            "registration_success",
            "registration_identity_fallback",
        ],
    )
    registration_residual_private = pd.read_csv(
        registration_residual_private_path,
        usecols=["patient_id", "visit", "registration_success", "audit_success"],
    )
    source_path, source = load_json("metrics/ispy1_base_eligibility_summary.json")
    observable_path, observable = load_json("manifests/grounding_observability_summary.json")
    privacy_path, privacy = load_json("metrics/public_artifact_privacy_gate.json")
    live_privacy = scan_public_artifacts()

    chosen = str(registration.get("chosen_strategy", "")).upper()
    if chosen not in {"H", "R"}:
        raise ValueError("registration strategy must be uniquely frozen as H or R")
    if chosen == "H":
        support = support_h
        support_path = support_h_path
    else:
        support = {
            "strategy": "C1B-R",
            "formal_patients": registration_support.get("formal_patients"),
            "formal_visits": registration_support.get("formal_visits"),
            "exact_full_support_containment_rate": registration_support.get(
                "c1b_r_exact_containment_rate"
            ),
            "physical_volume_retention": registration_support.get(
                "c1b_r_ftv_retention", {}
            ),
            "contains_patient_identifiers": registration_support.get(
                "contains_patient_identifiers"
            ),
        }
        support_path = registration_support_path
    cache_path, cache = load_json(
        f"metrics/model_input_pipeline_{chosen.lower()}_all_gate.json"
    )
    if str(cache.get("scope")) != "all":
        raise ValueError("a validation/smoke cache artifact cannot close Stage A")

    formal = dicom.get("formal", {})
    all_repairs = dicom.get("all_model_input_singular_visits", {})
    eligible_ispy1 = int(source.get("counts", {}).get("patients_eligible", -1))
    expected_model_patients = 808 + eligible_ispy1
    expected_model_visits = 4 * expected_model_patients

    cache_private = ROOT / f"metrics/model_input_pipeline_{chosen.lower()}_all.private.csv"
    if not cache_private.is_file():
        raise FileNotFoundError("full-scope private cache validation table is missing")
    cache_rows = pd.read_csv(
        cache_private,
        usecols=[
            "patient_id",
            "cohort",
            "strategy",
            "scope",
            "cache_path",
            "cache_content_sha256",
            "cache_file_sha256",
            "shape_valid",
            "dtype_float32",
            "finite",
            "whole_visit_nonconstant",
            "phase_indices_in_range",
            "cache_patient_identity_match",
            "cache_current_source_hash_match",
            "cache_complete_input_contract_match",
            "cache_schema_version",
            "builder_contract_sha256",
            "input_provenance_sha256",
            "base_only_later_support_loaded_count",
            "deterministic_duplicate_match",
        ],
    )
    required_cache_columns = [
        "shape_valid",
        "dtype_float32",
        "finite",
        "whole_visit_nonconstant",
        "phase_indices_in_range",
        "cache_patient_identity_match",
        "cache_current_source_hash_match",
        "cache_complete_input_contract_match",
        "deterministic_duplicate_match",
    ]
    cache_row_pass = pd.Series(True, index=cache_rows.index)
    for column in required_cache_columns:
        cache_row_pass &= cache_rows[column].map(as_bool)
    cache_row_pass &= cache_rows["cache_schema_version"].eq(3)
    cache_row_pass &= cache_rows["builder_contract_sha256"].astype(str).eq(
        builder_contract_sha256()
    )
    cache_row_pass &= cache_rows["input_provenance_sha256"].map(is_sha256)
    cache_row_pass &= cache_rows["base_only_later_support_loaded_count"].eq(0)
    actual_cohort_counts = {
        str(key): int(value) for key, value in cache_rows["cohort"].value_counts().items()
    }
    formal_cache_ids = set(registration_support_private["patient_id"].astype(str))
    cache_archive_contract_match = pd.Series(
        (
            cache_contract_matches(
                patient_id,
                cache_path,
                cohort=str(cohort),
                strategy=chosen,
                formal_ftv_overlap=str(patient_id) in formal_cache_ids,
            )
            for patient_id, cache_path, cohort in zip(
                cache_rows["patient_id"],
                cache_rows["cache_path"],
                cache_rows["cohort"],
                strict=True,
            )
        ),
        index=cache_rows.index,
        dtype=bool,
    )
    expected_cache_root = (ROOT / "cache" / f"c1b_{chosen.lower()}").resolve()
    cache_identity_complete = bool(
        not cache_rows["patient_id"].astype(str).duplicated().any()
        and not cache_rows["cache_path"].astype(str).duplicated().any()
        and cache_rows["strategy"].astype(str).eq(chosen).all()
        and cache_rows["scope"].astype(str).eq("all").all()
        and cache_rows["cache_path"].map(lambda value: Path(str(value)).is_file()).all()
        and cache_rows["cache_path"].map(
            lambda value: Path(str(value)).resolve().parent == expected_cache_root
        ).all()
        and cache_rows["cache_content_sha256"].map(is_sha256).all()
        and cache_rows["cache_file_sha256"].map(is_sha256).all()
        and cache_archive_contract_match.all()
    )

    selection_path = ROOT / "manifests/cache_validation_selection.private.csv"
    if not selection_path.is_file():
        raise FileNotFoundError("private cache-validation selection manifest is missing")
    selection = pd.read_csv(
        selection_path,
        usecols=[
            "patient_id",
            "cohort",
            "selected_for_validation",
            "selection_reasons",
        ],
    )
    if selection["patient_id"].astype(str).duplicated().any():
        raise ValueError("cache-validation selection has duplicate patients")
    selected = selection.loc[selection["selected_for_validation"].map(as_bool)].copy()
    reason_counts = {
        reason: int(selected["selection_reasons"].str.contains(reason, regex=False).sum())
        for reason in (
            "geometry_pixel_repair",
            "source_edge",
            "large_support_top_decile",
            "ispy1_base_fallback",
            "ispy1_safe_phase_resampling",
            *(f"fold_{fold}_deterministic_random" for fold in range(5)),
        )
    }

    resampling_path = ROOT / "metrics/orientation_resampling_patient_visit.private.csv"
    if not resampling_path.is_file():
        raise FileNotFoundError("private per-visit resampling audit is missing")
    resampling = pd.read_csv(
        resampling_path,
        usecols=[
            "patient_id",
            "visit",
            "cohort",
            "formal_ftv_overlap",
            "orientation_after",
            "anti_alias_required",
            "extreme_axis_factor_gt2",
            "max_resample_factor",
        ],
    )
    if resampling.duplicated(["patient_id", "visit"]).any():
        raise ValueError("per-visit resampling audit has duplicate patient/visit rows")
    formal_resampling = resampling.loc[resampling["formal_ftv_overlap"].map(as_bool)]
    formal_extreme = int(formal_resampling["extreme_axis_factor_gt2"].map(as_bool).sum())
    formal_max_factor = float(formal_resampling["max_resample_factor"].max())

    ispy1_patient_path = (
        ROOT / "manifests/ispy1_base_eligibility_patients.private.csv"
    )
    ispy1_visit_path = ROOT / "manifests/ispy1_base_eligibility_visits.private.csv"
    ispy1_phase_path = (
        ROOT / "manifests/ispy1_base_eligibility_phase_contract.private.csv"
    )
    for path in (ispy1_patient_path, ispy1_visit_path, ispy1_phase_path):
        if not path.is_file():
            raise FileNotFoundError(f"strict I-SPY1 private artifact is missing: {path.name}")
    ispy1_patients = pd.read_csv(
        ispy1_patient_path,
        usecols=["patient_id", "eligible", "passing_visit_count"],
    )
    ispy1_visits = pd.read_csv(
        ispy1_visit_path,
        usecols=[
            "patient_id",
            "visit",
            "status",
            "raw_pixel_cells_verified",
            "resampled_phase_count",
            "private_cell_audit",
        ],
    )
    eligible_private_ids = set(
        ispy1_patients.loc[
            ispy1_patients["eligible"].map(as_bool)
            & ispy1_patients["passing_visit_count"].eq(4),
            "patient_id",
        ].astype(str)
    )
    safe_phase_private_ids = set(
        ispy1_visits.loc[
            ispy1_visits["resampled_phase_count"].fillna(0).astype(int).gt(0),
            "patient_id",
        ].astype(str)
    )
    selected_safe_phase_ids = set(
        selected.loc[
            selected["selection_reasons"].str.contains(
                "ispy1_safe_phase_resampling", regex=False
            ),
            "patient_id",
        ].astype(str)
    )
    safe_phase_private_contract_complete = True
    safe_cell_root = (
        ROOT / "manifests/ispy1_base_eligibility_cells.private"
    ).resolve()
    for row in ispy1_visits.loc[
        ispy1_visits["resampled_phase_count"].fillna(0).astype(int).gt(0)
    ].to_dict("records"):
        cell_path = Path(str(row["private_cell_audit"]))
        try:
            cell_payload = json.loads(cell_path.read_text(encoding="utf-8"))
            safe_phase_private_contract_complete &= bool(
                cell_path.resolve().parent == safe_cell_root
                and str(cell_payload.get("patient_id")) == str(row["patient_id"])
                and str(cell_payload.get("visit")) == str(row["visit"])
                and str(
                    cell_payload.get("safe_phase_resample_boundary_mode", "")
                ).lower()
                == "reflect"
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            safe_phase_private_contract_complete = False
    ispy1_hashes = source.get("private_manifest_sha256", {})
    ispy1_private_complete = bool(
        len(ispy1_patients) == 156
        and not ispy1_patients["patient_id"].astype(str).duplicated().any()
        and len(ispy1_visits) == 624
        and not ispy1_visits.duplicated(["patient_id", "visit"]).any()
        and len(eligible_private_ids) == eligible_ispy1
        and len(safe_phase_private_ids) == 1
        and int(ispy1_visits["resampled_phase_count"].fillna(0).astype(int).sum()) == 2
        and safe_phase_private_contract_complete
        and ispy1_visits.loc[
            ispy1_visits["patient_id"].astype(str).isin(eligible_private_ids),
            "status",
        ].eq("PASS").all()
        and ispy1_visits.loc[
            ispy1_visits["patient_id"].astype(str).isin(eligible_private_ids),
            "raw_pixel_cells_verified",
        ].gt(0).all()
        and sha256(ispy1_patient_path) == str(ispy1_hashes.get("patients", ""))
        and sha256(ispy1_visit_path) == str(ispy1_hashes.get("visits", ""))
        and sha256(ispy1_phase_path) == str(ispy1_hashes.get("phase_contract", ""))
    )

    observable_private_path = (
        ROOT / "manifests/grounding_observability_manifest.private.csv"
    )
    if not observable_private_path.is_file():
        raise FileNotFoundError("private grounding-observability manifest is missing")
    observable_private = pd.read_csv(
        observable_private_path,
        usecols=[
            "patient_id",
            "visit",
            "source_boundary_touch",
            "grounding_observable_mask",
        ],
    )
    support_private_path = (
        ROOT / "metrics/support_containment_patient_visit.private.csv"
    )
    if not support_private_path.is_file():
        raise FileNotFoundError("private formal support audit is missing")
    support_private = pd.read_csv(
        support_private_path,
        usecols=["patient_id", "visit", "source_boundary_touch"],
    )
    observable_support_match = observable_private.merge(
        support_private,
        on=["patient_id", "visit"],
        how="outer",
        validate="one_to_one",
        suffixes=("_manifest", "_support_audit"),
        indicator=True,
    )
    observable_private_complete = bool(
        len(observable_private) == 1500
        and not observable_private.duplicated(["patient_id", "visit"]).any()
        and observable_private["patient_id"].astype(str).nunique() == 375
        and observable_private["source_boundary_touch"].map(as_bool).sum() == 14
        and observable_private["grounding_observable_mask"].map(as_bool).sum()
        == 1486
        and (
            observable_private["source_boundary_touch"].map(as_bool)
            == ~observable_private["grounding_observable_mask"].map(as_bool)
        ).all()
        and len(support_private) == 1500
        and observable_support_match["_merge"].eq("both").all()
        and (
            observable_support_match["source_boundary_touch_manifest"].map(as_bool)
            == observable_support_match["source_boundary_touch_support_audit"].map(
                as_bool
            )
        ).all()
        and sha256(observable_private_path)
        == str(observable.get("private_manifest_sha256", ""))
    )

    registration_cohort = registration_summary.get("cohort", {})
    registration_outcomes = registration_summary.get("outcomes", {})
    registration_criteria = registration_summary.get("gate", {}).get("criteria", {})
    registration_residual_cohort = registration_residual.get("cohort", {})
    registration_safe_resample = registration.get("safe_phase_resample", {})
    registration_pair_ids = set(registration_pairs["patient_id"].astype(str))
    registration_support_ids = set(
        registration_support_private["patient_id"].astype(str)
    )
    registration_residual_ids = set(
        registration_residual_private["patient_id"].astype(str)
    )
    registration_support_followup = registration_support_private.loc[
        registration_support_private["visit"].astype(str).ne("T0")
    ]
    registration_private_complete = bool(
        len(registration_pairs) == 1125
        and not registration_pairs.duplicated(["patient_id", "visit"]).any()
        and registration_pairs["visit"].astype(str).isin(("T1", "T2", "T3")).all()
        and len(registration_pair_ids) == 375
        and len(registration_support_private) == 1500
        and not registration_support_private.duplicated(["patient_id", "visit"]).any()
        and registration_support_private["visit"]
        .astype(str)
        .isin(("T0", "T1", "T2", "T3"))
        .all()
        and len(registration_support_ids) == 375
        and len(registration_support_followup) == 1125
        and len(registration_residual_private) == 1125
        and not registration_residual_private.duplicated(["patient_id", "visit"]).any()
        and registration_residual_private["visit"]
        .astype(str)
        .isin(("T1", "T2", "T3"))
        .all()
        and len(registration_residual_ids) == 375
        and registration_pair_ids == registration_support_ids == registration_residual_ids
        and registration_pairs["success"].map(as_bool).sum()
        == registration_support_followup["registration_success"].map(as_bool).sum()
        == registration_residual_private["registration_success"].map(as_bool).sum()
        and (~registration_pairs["success"].map(as_bool)).sum()
        == registration_support_followup["registration_identity_fallback"]
        .map(as_bool)
        .sum()
        and registration_residual_private["audit_success"].map(as_bool).sum() == 1125
        and sha256(registration_pairs_path)
        == str(
            registration_summary.get("provenance", {}).get(
                "private_pair_metrics_sha256", ""
            )
        )
        and sha256(registration_support_private_path)
        == str(registration_support.get("private_support_metrics_sha256", ""))
        and int(registration_support.get("private_support_metrics_rows", -1)) == 1500
        and sha256(registration_residual_private_path)
        == str(
            registration_residual.get("provenance", {}).get(
                "private_residual_metrics_sha256", ""
            )
        )
    )
    registration_full_complete = bool(
        int(registration_cohort.get("expected_patients", -1)) == 375
        and int(registration_cohort.get("expected_pairs", -1)) == 1125
        and int(registration_cohort.get("observed_pairs", -1)) == 1125
        and as_bool(registration_cohort.get("complete"))
        and int(registration_support.get("formal_patients", -1)) == 375
        and int(registration_support.get("formal_visits", -1)) == 1500
        and int(registration_support.get("registration_pairs", -1)) == 1125
        and int(registration_residual_cohort.get("pairs", -1)) == 1125
        and int(registration_residual_cohort.get("audit_successes", -1)) == 1125
        and int(registration_residual_cohort.get("audit_failures", -1)) == 0
        and int(registration_support.get("registration_success_pairs", -1))
        == int(registration_outcomes.get("successes", -2))
        and int(registration_support.get("registration_identity_fallback_pairs", -1))
        == int(registration_outcomes.get("failures", -2))
        and registration_private_complete
    )
    registration_decision_coherent = bool(
        int(registration.get("formal_registration_pairs", -1)) == 1125
        and int(registration.get("formal_registration_pairs", -1))
        == int(registration_cohort.get("observed_pairs", -2))
        and int(registration.get("registration_success_pairs", -1))
        == int(registration_outcomes.get("successes", -2))
        and int(registration.get("registration_failure_pairs", -1))
        == int(registration_outcomes.get("failures", -2))
        and int(registration.get("registration_success_pairs", -1))
        + int(registration.get("registration_failure_pairs", -1))
        == 1125
        and int(registration.get("physical_support_audit_visits", -1)) == 1500
        and int(registration.get("physical_support_audit_visits", -1))
        == int(registration_support.get("formal_visits", -2))
        and int(registration.get("physical_support_audit_pairs", -1)) == 1125
        and int(registration.get("physical_support_audit_pairs", -1))
        == int(registration_support.get("registration_pairs", -2))
        and int(registration.get("residual_audit_pairs", -1)) == 1125
        and int(registration.get("residual_audit_pairs", -1))
        == int(registration_residual_cohort.get("pairs", -2))
        and int(registration.get("residual_audit_successes", -1)) == 1125
        and int(registration.get("residual_audit_successes", -1))
        == int(registration_residual_cohort.get("audit_successes", -2))
        and int(registration.get("residual_audit_failures", -1)) == 0
        and int(registration.get("residual_audit_failures", -1))
        == int(registration_residual_cohort.get("audit_failures", -2))
        and int(registration.get("manual_review_case_count", -1)) == 4
        and as_bool(registration.get("manual_review_complete"))
        and str(registration.get("registration_sensitivity_summary_sha256", ""))
        == sha256(registration_summary_path)
        and str(
            registration.get("registration_physical_support_summary_sha256", "")
        )
        == sha256(registration_support_path)
        and str(registration.get("registration_residual_audit_sha256", ""))
        == sha256(registration_residual_path)
    )
    hard_registration_statuses = {
        name: str(registration_criteria.get(name, {}).get("status", ""))
        for name in (
            "finite_transform_success_rate",
            "catastrophic_transform_rate",
            "median_whole_anatomy_similarity_gain",
            "nonworse_moving_visit_fraction",
            "padding_increase",
        )
    }
    all_registration_statuses = {
        str(name): str(item.get("status", ""))
        for name, item in registration_criteria.items()
        if isinstance(item, dict)
    }
    registration_criteria_complete = bool(
        all_registration_statuses
        and all(status in {"PASS", "FAIL"} for status in all_registration_statuses.values())
    )
    if chosen == "R":
        chosen_decision_consistent = bool(
            registration_summary.get("gate", {}).get("decision") == "C1B-R"
            and registration_criteria_complete
            and all(status == "PASS" for status in all_registration_statuses.values())
            and as_bool(registration_support.get("r_exact_drop_gate_pass"))
            and as_bool(registration_support.get("r_retention_q05_gate_pass"))
            and as_bool(registration.get("residual_audit_pass"))
        )
    else:
        chosen_decision_consistent = bool(
            registration_summary.get("gate", {}).get("decision") == "C1B-H"
            and registration_criteria_complete
            and any(status == "FAIL" for status in all_registration_statuses.values())
        )

    gates = [
        check(
            "formal_raw_dicom_pixel_rebuild",
            dicom.get("status") == "PASS"
            and int(formal.get("visits", -1)) == 72
            and int(formal.get("passed", -1)) == 72
            and int(formal.get("failed", -1)) == 0
            and float(formal.get("pixel_order_verified_fraction", -1)) == 1.0
            and float(formal.get("max_cell_error", 1)) == 0.0
            and float(formal.get("max_footprint_corner_error_mm", 1)) <= 0.1
            and as_bool(formal.get("all_finite_nonconstant"))
            and as_bool(formal.get("all_qform_sform_valid"))
            and int(formal.get("decoded_cells", -1))
            == int(formal.get("verified_cells", -2)),
            {
                "passed_visits": formal.get("passed"),
                "verified_cells": formal.get("verified_cells"),
                "max_cell_error": formal.get("max_cell_error"),
                "max_footprint_corner_error_mm": formal.get(
                    "max_footprint_corner_error_mm"
                ),
            },
            "72/72 pass; every cell exact; physical-corner error <=0.1 mm",
        ),
        check(
            "complete_base_cohort_singular_rebuild",
            int(all_repairs.get("visits", -1)) == 146
            and int(all_repairs.get("passed", -1)) == 146
            and int(all_repairs.get("failed", -1)) == 0
            and float(all_repairs.get("pixel_order_verified_fraction", -1)) == 1.0
            and float(all_repairs.get("max_cell_error", 1)) == 0.0
            and float(all_repairs.get("max_footprint_corner_error_mm", 1)) <= 0.1
            and as_bool(all_repairs.get("all_finite_nonconstant"))
            and as_bool(all_repairs.get("all_qform_sform_valid"))
            and int(all_repairs.get("decoded_cells", -1))
            == int(all_repairs.get("verified_cells", -2)),
            {
                "visits": all_repairs.get("visits"),
                "verified_cells": all_repairs.get("verified_cells"),
                "failed": all_repairs.get("failed"),
            },
            "all singular visits entering the candidate base population pass",
        ),
        check(
            "true_canonical_orientation",
            float(orientation.get("canonical_ras_fraction", -1)) == 1.0
            and int(orientation.get("model_input_patients", -1))
            == expected_model_patients
            and int(orientation.get("model_input_visits", -1))
            == expected_model_visits
            and int(orientation.get("strict_ispy1_eligible_patients", -1))
            == eligible_ispy1
            and as_bool(orientation.get("array_reordering_implemented_and_unit_tested"))
            and not as_bool(orientation.get("header_only_label_change", True))
            and as_bool(orientation.get("left_right_consistent"))
            and as_bool(orientation.get("anterior_posterior_consistent"))
            and as_bool(orientation.get("superior_inferior_consistent"))
            and float(orientation.get("canonical_roundtrip_corner_error_mm_max", 1))
            <= 0.1
            and float(orientation.get("dce_mask_footprint_corner_error_mm_max", 1))
            <= 0.1,
            {
                "canonical_ras_fraction": orientation.get("canonical_ras_fraction"),
                "roundtrip_error_mm_max": orientation.get(
                    "canonical_roundtrip_corner_error_mm_max"
                ),
                "dce_mask_error_mm_max": orientation.get(
                    "dce_mask_footprint_corner_error_mm_max"
                ),
            },
            "true RAS+ reorientation; all anatomical axes and <=0.1-mm round trips pass",
        ),
        check(
            "registration_strategy_frozen",
            as_bool(registration.get("decision_frozen"))
            and chosen in {"H", "R"}
            and as_bool(registration.get("manual_review_pass"))
            and registration_full_complete
            and registration_decision_coherent
            and chosen_decision_consistent,
            {
                "chosen_strategy": chosen,
                "formal_pairs": registration_cohort.get("observed_pairs"),
                "registration_success_rate": registration.get(
                    "registration_success_rate"
                ),
                "catastrophic_rate": registration.get("catastrophic_rate"),
                "identity_fallback_pairs": registration_support.get(
                    "registration_identity_fallback_pairs"
                ),
                "hard_gate_statuses": hard_registration_statuses,
                "all_gate_statuses": all_registration_statuses,
                "manual_review_status": registration.get("manual_review_status"),
                "residual_audit_successes": registration_residual_cohort.get(
                    "audit_successes"
                ),
                "decision_hashes_match": registration_decision_coherent,
                "private_audit_rows_and_hashes_match": registration_private_complete,
            },
            "375x3 formal pairs; H/R decision coherent; manual, support, and residual audits complete",
        ),
        check(
            "extreme_resampling_disposition",
            formal_extreme == 35
            and formal_max_factor <= 5.7
            and len(resampling) == expected_model_visits
            and resampling["patient_id"].astype(str).nunique()
            == expected_model_patients
            and resampling["orientation_after"].astype(str).eq("RAS").all()
            and cache_row_pass.all()
            and cache_identity_complete
            and int(cache.get("constant_derived_channels_total", -1)) >= 0,
            {
                "formal_extreme_visits": formal_extreme,
                "formal_max_axis_factor": formal_max_factor,
                "policy": "source-domain Gaussian anti-alias then fixed-grid linear sampling",
                "full_cache_failures": int((~cache_row_pass).sum()),
            },
            "all >2-axis cases audited and successfully traverse fixed preregistered builder",
        ),
        check(
            "complete_dce7_builder_and_cache",
            int(cache.get("patients", -1)) == expected_model_patients
            and int(cache.get("visits", -1)) == expected_model_visits
            and int(cache.get("cache_schema_version", -1)) == 3
            and str(cache.get("builder_contract_sha256"))
            == builder_contract_sha256()
            and str(cache.get("strategy")) == f"C1B-{chosen}"
            and int(cache.get("eligible_ispy1_base_patients", -1))
            == eligible_ispy1
            and len(cache_rows) == expected_model_patients
            and actual_cohort_counts.get("I-SPY2", 0) == 808
            and actual_cohort_counts.get("I-SPY1", 0) == eligible_ispy1
            and cache_identity_complete
            and cache_row_pass.all()
            and float(cache.get("cache_exact_roundtrip_pass_fraction", -1)) == 1.0
            and float(cache.get("byte_deterministic_validation_fraction", -1)) == 1.0
            and float(cache.get("phase_indices_in_range_fraction", -1)) == 1.0
            and float(cache.get("cache_patient_identity_match_fraction", -1)) == 1.0
            and float(cache.get("cache_current_source_hash_match_fraction", -1))
            == 1.0
            and float(
                cache.get("cache_complete_input_contract_match_fraction", -1)
            )
            == 1.0
            and int(cache.get("safe_phase_resampling_patients_in_validation", -1))
            == 1
            and str(cache.get("padding_mode")) == "reflect"
            and not as_bool(cache.get("zero_or_fixed_sentinel_padding", True))
            and as_bool(cache.get("model_loader_returns_only_dce7"))
            and str(cache.get("anchor_support_scope")) == "T0_only"
            and str(cache.get("later_visit_support_scope"))
            == "formal_ftv_overlap_containment_qc_only"
            and not as_bool(
                cache.get("later_visit_support_affects_grid_or_tensor", True)
            )
            and int(cache.get("base_only_later_visit_supports_loaded", -1)) == 0
            and not as_bool(cache.get("sidecars_are_model_inputs", True))
            and not as_bool(cache.get("clinical_treatment_pcr_ld_columns_read", True)),
            {
                "patients": cache.get("patients"),
                "visits": cache.get("visits"),
                "cache_schema_version": cache.get("cache_schema_version"),
                "cohort_counts": actual_cohort_counts,
                "cache_schema_identity_source_contract_failures": int(
                    (~cache_archive_contract_match).sum()
                ),
                "roundtrip_fraction": cache.get("cache_exact_roundtrip_pass_fraction"),
                "byte_deterministic_validation_fraction": cache.get(
                    "byte_deterministic_validation_fraction"
                ),
                "cache_patient_identity_match_fraction": cache.get(
                    "cache_patient_identity_match_fraction"
                ),
                "cache_current_source_hash_match_fraction": cache.get(
                    "cache_current_source_hash_match_fraction"
                ),
                "cache_complete_input_contract_match_fraction": cache.get(
                    "cache_complete_input_contract_match_fraction"
                ),
                "safe_phase_resampling_patients_in_validation": cache.get(
                    "safe_phase_resampling_patients_in_validation"
                ),
            },
            "all 808 I-SPY2 plus strict eligible I-SPY1; schema-3 complete input provenance, identity, exact round-trip, and repeat hashes",
        ),
        check(
            "cache_validation_strata_coverage",
            reason_counts["geometry_pixel_repair"] == 42
            and reason_counts["source_edge"] == 12
            and reason_counts["large_support_top_decile"] == 38
            and reason_counts["ispy1_base_fallback"] >= 1
            and reason_counts["ispy1_safe_phase_resampling"] == 1
            and selected_safe_phase_ids == safe_phase_private_ids
            and all(
                reason_counts[f"fold_{fold}_deterministic_random"] == 5
                for fold in range(5)
            ),
            {
                "validation_patients": int(len(selected)),
                "reason_counts": reason_counts,
            },
            "all 42 repair, 12 source-edge, 38 large-support patients; five per fold; strict I-SPY1 sample",
        ),
        check(
            "registration_physical_support_audit",
            int(registration_support.get("formal_patients", -1)) == 375
            and int(registration_support.get("formal_visits", -1)) == 1500
            and int(registration_support.get("registration_pairs", -1)) == 1125
            and abs(
                float(registration_support.get("c1b_h_exact_containment_rate", -1))
                - float(support_h.get("exact_full_support_containment_rate", -2))
            )
            <= 1e-12
            and abs(
                float(registration_support.get("c1b_h_ftv_retention_q05", -1))
                - float(
                    support_h.get("physical_volume_retention", {}).get("q05", -2)
                )
            )
            <= 1e-12
            and not as_bool(
                registration_support.get("registration_fitted_with_localization", True)
            )
            and as_bool(registration_support.get("localization_opened_only_posthoc"))
            and not as_bool(registration_support.get("failed_transform_used", True)),
            {
                "h_exact": registration_support.get(
                    "c1b_h_exact_containment_rate"
                ),
                "r_exact": registration_support.get(
                    "c1b_r_exact_containment_rate"
                ),
                "h_q05": registration_support.get("c1b_h_ftv_retention_q05"),
                "r_q05": registration_support.get("c1b_r_ftv_retention_q05"),
                "fallback_pairs": registration_support.get(
                    "registration_identity_fallback_pairs"
                ),
            },
            "complete post-hoc H/R source-domain support audit; localization never fits transform",
        ),
        check(
            "formal_available_support_containment",
            str(support.get("strategy")) == f"C1B-{chosen}"
            and int(support.get("formal_patients", -1)) == 375
            and int(support.get("formal_visits", -1)) == 1500
            and float(support.get("exact_full_support_containment_rate", -1)) >= 0.95,
            support.get("exact_full_support_containment_rate"),
            ">=0.95 exact available-support containment on 1,500 formal visits",
        ),
        check(
            "formal_ftv_retention_q05",
            float(support.get("physical_volume_retention", {}).get("q05", -1))
            >= 0.95,
            support.get("physical_volume_retention", {}).get("q05"),
            ">=0.95",
        ),
        check(
            "strict_ispy1_source_eligibility",
            int(source.get("counts", {}).get("patients_total", -1)) == 156
            and int(source.get("counts", {}).get("visits_total", -1)) == 624
            and eligible_ispy1 >= 0
            and int(source.get("counts", {}).get("patients_ineligible", -1))
            == 156 - eligible_ispy1
            and int(source.get("counts", {}).get("visits_pass", -1))
            + int(source.get("counts", {}).get("visits_fail", -1))
            == 624
            and int(source.get("counts", {}).get("raw_pixel_cells_verified", 0)) > 0
            and int(
                source.get("counts", {}).get(
                    "visits_with_safe_phase_resampling", -1
                )
            )
            == 1
            and int(source.get("counts", {}).get("resampled_phases", -1)) == 2
            and ispy1_private_complete
            and source.get("outcome_fields_read") == []
            and source.get("clinical_or_outcome_tables_read") == []
            and str(
                source.get("thresholds", {}).get(
                    "safe_resample_boundary_mode", ""
                )
            ).lower()
            == "reflect"
            and not as_bool(source.get("public_artifact_contains_identifiers_or_paths", True)),
            {
                "eligible_patients": eligible_ispy1,
                "passing_visits": source.get("counts", {}).get("visits_pass"),
                "failing_visits": source.get("counts", {}).get("visits_fail"),
                "safe_resampled_visits": source.get("counts", {}).get(
                    "visits_with_safe_phase_resampling"
                ),
                "safe_resample_boundary_mode": source.get("thresholds", {}).get(
                    "safe_resample_boundary_mode"
                ),
            },
            "source/pixel-only eligibility; no outcome or clinical table read",
        ),
        check(
            "grounding_observability_contract",
            int(observable.get("formal_visits", -1)) == 1500
            and int(observable.get("ineligible_visits", -1)) == 14
            and int(observable.get("observable_visits", -1)) == 1486
            and observable_private_complete
            and not as_bool(observable.get("is_model_input", True))
            and as_bool(observable.get("does_not_filter_base_training")),
            {
                "observable_visits": observable.get("observable_visits"),
                "ineligible_visits": observable.get("ineligible_visits"),
                "affected_patients": observable.get("affected_patients"),
            },
            "frozen loss-only mask; 14/1,500 ineligible; never model input or base filter",
        ),
        check(
            "no_future_or_forbidden_information",
            as_bool(support_h.get("anchor_uses_t0_only"))
            and not as_bool(support_h.get("future_support_used_for_grid", True))
            and str(cache.get("anchor_support_scope")) == "T0_only"
            and str(cache.get("later_visit_support_scope"))
            == "formal_ftv_overlap_containment_qc_only"
            and not as_bool(
                cache.get("later_visit_support_affects_grid_or_tensor", True)
            )
            and int(cache.get("base_only_later_visit_supports_loaded", -1)) == 0
            and str(registration_safe_resample.get("phase_policy", ""))
            == "legacy_adaptive_early_late_outcome_free"
            and str(registration_safe_resample.get("registration_channel", ""))
            == "selected_precontrast_only"
            and str(
                registration_safe_resample.get("model_resampling_padding_mode", "")
            ).lower()
            == "reflect"
            and not as_bool(
                registration_safe_resample.get(
                    "registration_qc_outside_value_is_model_padding", True
                )
            )
            and not as_bool(
                registration_safe_resample.get(
                    "valid_source_mask_is_model_input", True
                )
            )
            and as_bool(cache.get("model_loader_returns_only_dce7"))
            and not as_bool(cache.get("sidecars_are_model_inputs", True))
            and not as_bool(cache.get("clinical_treatment_pcr_ld_columns_read", True)),
            {
                "anchor_uses_t0_only": support_h.get("anchor_uses_t0_only"),
                "future_support_used_for_grid": support_h.get(
                    "future_support_used_for_grid"
                ),
                "cache_support_scope": {
                    "anchor": cache.get("anchor_support_scope"),
                    "later": cache.get("later_visit_support_scope"),
                    "later_affects_grid_or_tensor": cache.get(
                        "later_visit_support_affects_grid_or_tensor"
                    ),
                    "base_only_later_supports_loaded": cache.get(
                        "base_only_later_visit_supports_loaded"
                    ),
                },
                "safe_phase_resample": registration_safe_resample,
                "ispy1_safe_resample_boundary_mode": source.get(
                    "thresholds", {}
                ).get("safe_resample_boundary_mode"),
                "model_loader_returns_only_dce7": cache.get(
                    "model_loader_returns_only_dce7"
                ),
                "sidecars_are_model_inputs": cache.get("sidecars_are_model_inputs"),
            },
            "T0-only grid; tensor is DCE7 only; geometry/clinical/outcome fields excluded",
        ),
        check(
            "public_artifact_privacy",
            int(privacy.get("schema_version", -1)) >= 2
            and privacy.get("status") == "PASS"
            and isinstance(privacy.get("scanned_files_sha256"), dict)
            and bool(privacy.get("scanned_files_sha256"))
            and not privacy.get("identifier_or_path_findings")
            and not privacy.get("stale_smoke_or_limited_public_artifacts")
            and live_privacy.get("status") == "PASS"
            and not live_privacy.get("identifier_or_path_findings")
            and not live_privacy.get("stale_smoke_or_limited_public_artifacts")
            and all(
                not as_bool(payload.get("contains_patient_identifiers", False))
                for payload in (
                    orientation,
                    support,
                    registration,
                    registration_summary,
                    registration_support,
                    registration_residual,
                    observable,
                    cache,
                )
            )
            and not as_bool(dicom.get("patient_identifiers_in_public_outputs", True)),
            {
                "saved_privacy_scan_status": privacy.get("status"),
                "live_privacy_scan_status": live_privacy.get("status"),
                "scanned_public_text_artifacts": live_privacy.get(
                    "scanned_public_text_artifacts"
                ),
                "findings": len(
                    live_privacy.get("identifier_or_path_findings", [])
                ),
            },
            "saved and live scans find no identifiers, UIDs, absolute paths, or stale debug outputs",
        ),
    ]

    status = "GO" if all(row["status"] == "PASS" for row in gates) else "NO-GO"
    provenance = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in (
            dicom_path,
            orientation_path,
            support_h_path,
            support_path,
            registration_path,
            registration_summary_path,
            registration_support_path,
            registration_residual_path,
            registration_pairs_path,
            registration_support_private_path,
            registration_residual_private_path,
            source_path,
            observable_path,
            privacy_path,
            cache_path,
            cache_private,
            selection_path,
            resampling_path,
            ispy1_patient_path,
            ispy1_visit_path,
            ispy1_phase_path,
            observable_private_path,
            support_private_path,
        )
    }
    payload = {
        "schema_version": 1,
        "stage": "A",
        "status": status,
        "chosen_input_strategy": f"C1B-{chosen}",
        "thresholds_relaxed": False,
        "stage_b_authorized": status == "GO",
        "gates": gates,
        "live_privacy_scan": {
            "status": live_privacy.get("status"),
            "scanned_public_text_artifacts": live_privacy.get(
                "scanned_public_text_artifacts"
            ),
            "scanned_files_sha256": live_privacy.get("scanned_files_sha256"),
        },
        "provenance_sha256": provenance,
        "contains_patient_identifiers": False,
    }
    gate_json = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    atomic_text(ROOT / "metrics/stage_a_model_ready_gate.json", gate_json, args.overwrite)
    table = pd.DataFrame(gates)
    atomic_text(
        ROOT / "metrics/table1_model_ready_preprocessing_qc.csv",
        table.to_csv(index=False),
        args.overwrite,
    )

    rows = "\n".join(
        f"| {row['gate']} | {row['status']} | `{json.dumps(row['observed'], ensure_ascii=False, sort_keys=True)}` | {row['requirement']} |"
        for row in gates
    )
    report = f"""# Stage A C1B model-ready hard gate

## 结论

`STAGE_A = {status}`。正式输入唯一冻结为 `C1B-{chosen}`。本判定未放宽任何预注册阈值；{'允许' if status == 'GO' else '禁止'}启动 Stage B。

| 子门 | 状态 | 观测 | 冻结要求 |
|---|---|---|---|
{rows}

所有公开汇总均不含patient identifier；逐病例路径、hash和几何sidecar只保存在`.private`资产中，且不进入模型tensor。
"""
    atomic_text(ROOT / "reports/stage_a_gate_report.md", report, args.overwrite)

    go_path = ROOT / "STAGE_A_GO.json"
    no_go_path = ROOT / "STAGE_A_NO_GO.json"
    if status == "GO":
        if no_go_path.exists():
            no_go_path.unlink()
        atomic_text(go_path, gate_json, args.overwrite)
    else:
        if go_path.exists():
            go_path.unlink()
        atomic_text(no_go_path, gate_json, args.overwrite)
    print(gate_json, end="")


if __name__ == "__main__":
    main()

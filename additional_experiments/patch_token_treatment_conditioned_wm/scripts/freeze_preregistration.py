#!/usr/bin/env python3
"""Create or verify the immutable pre-outcome experiment lock.

The write mode is intentionally one-shot and refuses to run after any formal
decision artifact exists.  Verification is used by every formal training and
evaluation entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
CONFIG = EXPERIMENT_ROOT / "configs" / "pilot.json"
PLAN = EXPERIMENT_ROOT / "EXPERIMENT_PLAN.md"
LOCK = EXPERIMENT_ROOT / "PREREGISTRATION_LOCK.json"
IMPLEMENTATION_ERRATA = (
    EXPERIMENT_ROOT / "PREREGISTRATION_IMPLEMENTATION_ERRATUM_1.json",
    EXPERIMENT_ROOT / "PREREGISTRATION_IMPLEMENTATION_ERRATUM_2.json",
)
RESULT_SENTINELS = (
    EXPERIMENT_ROOT / "metrics" / "decision.json",
    EXPERIMENT_ROOT / "metrics" / "formal_matrix_complete.json",
    EXPERIMENT_ROOT / "reports" / "final_report.md",
)

UPSTREAM_FILES = (
    REPO_ROOT
    / "additional_experiments"
    / "g3_multiseed_generalization"
    / "src"
    / "dgrs"
    / "model.py",
    REPO_ROOT
    / "additional_experiments"
    / "g3_multiseed_generalization"
    / "src"
    / "dgrs"
    / "training.py",
    REPO_ROOT
    / "additional_experiments"
    / "c1b_spatial_pooling_bottleneck_audit"
    / "src"
    / "c1b_spatial_audit"
    / "pooling.py",
    REPO_ROOT
    / "additional_experiments"
    / "c1b_overlap_eligibility_ftv_stageb"
    / "src"
    / "c1b_stage_b"
    / "data.py",
    REPO_ROOT
    / "additional_experiments"
    / "c1b_overlap_eligibility_ftv_stageb"
    / "src"
    / "c1b_stage_b"
    / "targets.py",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _locked_code_files() -> tuple[Path, ...]:
    paths: list[Path] = []
    for relative in ("src", "scripts", "tests"):
        root = EXPERIMENT_ROOT / relative
        paths.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.name != "PREREGISTRATION_LOCK.json"
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        )
    return tuple(sorted(paths))


def _relative_hashes(paths: Iterable[Path], *, root: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    for path in paths:
        source = path.resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        output[source.relative_to(root.resolve()).as_posix()] = file_sha256(source)
    return output


def build_payload() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config.get("status_before_formal_results") != "PREREGISTERED_PENDING_CODE_LOCK":
        raise ValueError("pilot config is not in the pre-result lock state")
    code = _relative_hashes(_locked_code_files(), root=EXPERIMENT_ROOT)
    upstream = _relative_hashes(UPSTREAM_FILES, root=REPO_ROOT)
    scientific_contract = {
        "config_sha256": file_sha256(CONFIG),
        "plan_sha256": file_sha256(PLAN),
        "code_sha256": code,
        "upstream_sha256": upstream,
        "pcr_free_world_model": True,
        "formal_new_cells": 10,
        "seed_bases": [2026, 3026],
        "folds": [0, 1, 2, 3, 4],
        "condition_method": "condition_token",
        "token_mask_ratio": 0.5,
        "token_width": 128,
        "pca_components": 64,
        "bootstrap_draws": 2000,
    }
    return {
        "schema_version": 1,
        "experiment": "patch_token_treatment_conditioned_wm",
        "status": "LOCKED_BEFORE_FORMAL_TRAINING_AND_OUTCOME_EVALUATION",
        "parent_commit": "7644e3835af6b12899c57819bedd1876572c434f",
        "scientific_contract": scientific_contract,
        "scientific_contract_sha256": canonical_sha256(scientific_contract),
        "outcome_firewall": {
            "pcr_loaded_during_world_model_training": False,
            "pcr_used_for_checkpoint_selection": False,
            "test_ftv_used_for_checkpoint_selection": False,
            "formal_results_existed_when_written": False,
        },
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_lock() -> dict[str, Any]:
    if LOCK.exists():
        raise FileExistsError(f"refusing to overwrite preregistration lock: {LOCK}")
    if any(path.exists() for path in RESULT_SENTINELS):
        raise RuntimeError(
            "formal result artifacts already exist; lock cannot be created"
        )
    payload = build_payload()
    _atomic_json(LOCK, payload)
    return {"status": "PASS", "lock_sha256": file_sha256(LOCK), **payload}


def verify() -> dict[str, Any]:
    if not LOCK.is_file():
        raise FileNotFoundError("preregistration lock does not exist")
    observed = json.loads(LOCK.read_text(encoding="utf-8"))
    contract = observed.get("scientific_contract")
    if not isinstance(contract, dict):
        raise ValueError("preregistration lock has no scientific contract")
    if observed.get("scientific_contract_sha256") != canonical_sha256(contract):
        raise ValueError("preregistration scientific-contract digest drifted")
    if contract.get("config_sha256") != file_sha256(CONFIG):
        raise ValueError("locked formal config drifted")
    if contract.get("plan_sha256") != file_sha256(PLAN):
        raise ValueError("locked experiment plan drifted")
    if contract.get("upstream_sha256") != _relative_hashes(
        UPSTREAM_FILES, root=REPO_ROOT
    ):
        raise ValueError("locked upstream implementation drifted")

    locked_code = contract.get("code_sha256")
    if not isinstance(locked_code, dict):
        raise ValueError("preregistration lock has no code digest mapping")
    expected_code = dict(locked_code)
    errata_sha256: list[str] = []
    previous_erratum_sha256: str | None = None
    for index, erratum_path in enumerate(IMPLEMENTATION_ERRATA, start=1):
        if not erratum_path.is_file():
            if any(path.is_file() for path in IMPLEMENTATION_ERRATA[index:]):
                raise ValueError("implementation erratum chain has a gap")
            break
        erratum = json.loads(erratum_path.read_text(encoding="utf-8"))
        allowed_status = (
            "LOCKED_BEFORE_FORMAL_TRAINING_AND_OUTCOME_EVALUATION"
            if index == 1
            else "LOCKED_AFTER_ABORTED_PREOUTCOME_FORMAL_ATTEMPT"
        )
        if (
            not isinstance(erratum, dict)
            or int(erratum.get("schema_version", -1)) != 1
            or erratum.get("status") != allowed_status
        ):
            raise ValueError("implementation erratum schema/status is invalid")
        if erratum.get("original_lock_sha256") != file_sha256(LOCK):
            raise ValueError("implementation erratum names a different original lock")
        if (
            index > 1
            and erratum.get("parent_erratum_sha256") != previous_erratum_sha256
        ):
            raise ValueError("implementation erratum parent digest drifted")
        replacements = dict(erratum.get("replacement_code_sha256", {}))
        additions = dict(erratum.get("added_code_sha256", {}))
        if not replacements and not additions:
            raise ValueError("implementation erratum names no code changes")
        if not set(replacements).issubset(expected_code):
            raise ValueError("implementation erratum replaces an unknown path")
        if set(additions) & set(expected_code):
            raise ValueError("implementation erratum addition overlaps locked code")
        if erratum.get("scientific_config_or_plan_changed") is not False:
            raise ValueError(
                "implementation erratum may not change the scientific design"
            )
        if erratum.get("formal_outcomes_existed_when_written") is not False:
            raise ValueError("implementation erratum was not pre-outcome")
        if erratum.get("formal_training_started_when_written") is True:
            if (
                int(erratum.get("completed_formal_cells_when_written", -1)) != 0
                or erratum.get("pcr_accessed_when_written") is not False
            ):
                raise ValueError("post-attempt erratum crossed the outcome firewall")
        elif erratum.get("formal_training_started_when_written") is not False:
            raise ValueError("implementation erratum has invalid training-start status")
        expected_code.update(replacements)
        expected_code.update(additions)
        previous_erratum_sha256 = file_sha256(erratum_path)
        errata_sha256.append(previous_erratum_sha256)

    expected_paths = set(expected_code)
    current = _relative_hashes(_locked_code_files(), root=EXPERIMENT_ROOT)
    if set(current) != expected_paths:
        missing = sorted(expected_paths - set(current))
        extra = sorted(set(current) - expected_paths)
        raise ValueError(
            f"locked implementation path set drifted; missing={missing}, extra={extra}"
        )
    for relative, expected_digest in expected_code.items():
        if current.get(relative) != expected_digest:
            raise ValueError(f"locked implementation drifted at {relative}")
    return {
        "status": "PASS",
        "lock_sha256": file_sha256(LOCK),
        "scientific_contract_sha256": observed["scientific_contract_sha256"],
        "implementation_erratum_sha256": (errata_sha256[-1] if errata_sha256 else None),
        "implementation_errata_sha256": errata_sha256,
        "verified_code_files": len(current),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = write_lock() if args.write else verify()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

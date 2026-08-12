#!/usr/bin/env python3
"""Bind the ten private confirmed LOCAL3 S0 cells by public aggregate hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
SRC = EXPERIMENT_ROOT / "src"
sys.path.insert(0, str(SRC))

from residual_sph.contracts import (  # noqa: E402
    TrainHyperparameters,
    file_sha256,
    validate_seed_fold,
)
from residual_sph.model import build_model, shared_initialization_sha256  # noqa: E402
from residual_sph.preregistration import verify_preregistration  # noqa: E402


DEFAULT_CONFIRMATION = (
    REPO_ROOT / "additional_experiments" / "local_response_state_multiseed_confirmation"
)
OUTPUT = EXPERIMENT_ROOT / "manifests" / "s0_confirmation_provenance.json"
SOURCE_DELIVERY_COMMIT = "b4ec0c1473da513f2b19baa58d54c0fd5382e52f"
SOURCE_EVIDENCE_PATHS = {
    "local_confirmation_report": "reports/final_report.md",
    "local_confirmation_decision": "metrics/decision_summary.json",
    "local_confirmation_static_ftv": "metrics/table2_static_ftv.csv",
    "local_confirmation_delta_ftv": "metrics/table3_observed_delta_ftv.csv",
}


def _load_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _git_blob_sha256(commit: str, relative: str) -> str:
    repository_path = (
        "additional_experiments/local_response_state_multiseed_confirmation/"
        + relative
    )
    result = subprocess.run(
        ["git", "show", f"{commit}:{repository_path}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(result.stdout).hexdigest()


def _verify_source_ancestry(
    root: Path, preregistration_lock: dict[str, object]
) -> dict[str, object]:
    source_lock_path = root / "PREREGISTRATION_LOCK.json"
    source_lock = _load_json(source_lock_path, "source preregistration lock")
    ancestry = preregistration_lock.get("s0_confirmation_ancestry")
    prior_hashes = preregistration_lock.get("prior_evidence_sha256")
    if not isinstance(ancestry, dict) or not isinstance(prior_hashes, dict):
        raise ValueError("pilot lock lacks S0 ancestry/evidence contracts")
    expected_head = str(ancestry["source_head_recorded_by_source_lock"])
    expected_source_lock = {
        "experiment": "local_response_state_multiseed_confirmation",
        "status": "FROZEN_BEFORE_FORMAL_RESULTS",
        "config_sha256": "97394658c6834c98c9ccdfa017e75ab733654c99f5f37271339421fdecbf9eb3",
    }
    for key, value in expected_source_lock.items():
        if source_lock.get(key) != value:
            raise ValueError(f"source preregistration differs at {key}")
    source_git = source_lock.get("git")
    if not isinstance(source_git, dict) or source_git.get("head") != expected_head:
        raise ValueError("source preregistration Git head differs from pilot lock")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", expected_head, SOURCE_DELIVERY_COMMIT],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    verified_evidence: dict[str, str] = {}
    for key, relative in SOURCE_EVIDENCE_PATHS.items():
        path = root / relative
        expected = str(prior_hashes[key])
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"source evidence is missing or hash-mismatched: {key}")
        if _git_blob_sha256(SOURCE_DELIVERY_COMMIT, relative) != expected:
            raise ValueError(f"source evidence is not bound to delivery commit: {key}")
        verified_evidence[key] = expected
    decision = _load_json(root / SOURCE_EVIDENCE_PATHS["local_confirmation_decision"], "source decision")
    if decision.get("classification") != ancestry.get("required_classification"):
        raise ValueError("source decision is not LOCAL_MULTISEED_CONFIRMED")
    return {
        "source_lock_sha256": file_sha256(source_lock_path),
        "source_lock_head": expected_head,
        "source_delivery_commit": SOURCE_DELIVERY_COMMIT,
        "source_lock_is_ancestor_of_delivery_commit": True,
        "required_classification": decision["classification"],
        "verified_prior_evidence_sha256": verified_evidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation-root", type=Path, default=DEFAULT_CONFIRMATION)
    args = parser.parse_args()
    preregistration = verify_preregistration(EXPERIMENT_ROOT)
    pilot_lock = _load_json(
        EXPERIMENT_ROOT / "PREREGISTRATION_LOCK.json", "pilot preregistration lock"
    )
    root = args.confirmation_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"confirmation runtime root is missing: {root}")
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    source_ancestry = _verify_source_ancestry(root, pilot_lock)
    cells: list[dict[str, object]] = []
    for seed in (2026, 3026):
        for fold in range(5):
            effective_seed = validate_seed_fold(seed, fold)
            expected_initialization = shared_initialization_sha256(
                build_model("S0", effective_seed)
            )
            run = root / "checkpoints" / "formal_4x8" / f"seed_{seed}" / "LOCAL3" / f"fold_{fold}"
            feature_run = root / "features" / "formal_4x8" / f"seed_{seed}" / "LOCAL3" / f"fold_{fold}"
            selection_path = run / "selection.json"
            checkpoint_path = run / "selected.pt"
            ftv_transform_path = run / "ftv_transform.json"
            feature_path = feature_run / "response_state.private.npz"
            metadata_path = feature_run / "response_state.private.metadata.json"
            for path in (
                selection_path,
                checkpoint_path,
                ftv_transform_path,
                feature_path,
                metadata_path,
            ):
                if not path.is_file():
                    raise FileNotFoundError(f"missing S0 runtime asset: {path}")
            selection = _load_json(selection_path, "S0 selection")
            metadata = _load_json(metadata_path, "S0 feature metadata")
            expected = {
                "arm": "LOCAL3",
                "seed_base": seed,
                "fold": fold,
                "effective_seed": effective_seed,
                "selection_mode": "primary",
                "experiment_pass": True,
                "optimization_safety_pass": True,
                "test_data_used": False,
                "pcr_used": False,
                "delta_ftv_used": False,
                "paired_initialization_sha256": expected_initialization,
                "hyperparameters": TrainHyperparameters().to_dict(),
            }
            for key, value in expected.items():
                if selection.get(key) != value:
                    raise ValueError(f"S0 seed {seed}/fold {fold} selection differs at {key}")
            expected_metadata = {
                "experiment": "local_response_state_multiseed_confirmation",
                "arm": "LOCAL3",
                "seed_base": seed,
                "fold": fold,
                "selected_epoch": int(selection["selected_epoch"]),
                "feature_shape": [808, 4, 192],
                "feature_dtype": "float32",
                "preregistration_lock_sha256": source_ancestry["source_lock_sha256"],
                "ftv_head_called": False,
                "test_labels_used": False,
            }
            for key, value in expected_metadata.items():
                if metadata.get(key) != value:
                    raise ValueError(f"S0 feature metadata differs at {key}")
            if selection.get("preregistration_lock_sha256") != source_ancestry["source_lock_sha256"]:
                raise ValueError("S0 selection is not bound to the source preregistration")
            hashes = {
                "selection_sha256": file_sha256(selection_path),
                "checkpoint_sha256": file_sha256(checkpoint_path),
                "feature_sha256": file_sha256(feature_path),
                "feature_metadata_sha256": file_sha256(metadata_path),
                "ftv_transform_sha256": file_sha256(ftv_transform_path),
            }
            expected_hashes = {
                "selection_sha256": metadata.get("selection_sha256"),
                "checkpoint_sha256": metadata.get("checkpoint_sha256"),
                "feature_sha256": metadata.get("feature_sha256"),
            }
            for key, value in expected_hashes.items():
                if hashes[key] != value:
                    raise ValueError(f"S0 feature metadata differs at {key}")
            cells.append(
                {
                    "arm": "S0",
                    "runtime_source_arm": "LOCAL3",
                    "seed_base": seed,
                    "fold": fold,
                    "effective_seed": effective_seed,
                    "selected_epoch": int(selection["selected_epoch"]),
                    "selected_validation_state_loss": float(selection["selected_validation_state_loss"]),
                    "selected_validation_ftv_loss": float(selection["selected_validation_ftv_loss"]),
                    "paired_initialization_sha256": expected_initialization,
                    "train_patient_sha256": str(selection["train_patient_sha256"]),
                    "val_patient_sha256": str(selection["val_patient_sha256"]),
                    "data_provenance_sha256": str(selection["data_provenance_sha256"]),
                    "hyperparameters": dict(selection["hyperparameters"]),
                    **hashes,
                }
            )
    payload = {
        "schema_version": 1,
        "experiment": "residual_sph_grounding_pilot",
        "artifact": "confirmed_LOCAL3_S0_runtime_provenance",
        "status": "S0_CONFIRMATION_PROVENANCE_VERIFIED",
        "cell_count": len(cells),
        "source_experiment": "local_response_state_multiseed_confirmation",
        "source_commit": SOURCE_DELIVERY_COMMIT,
        "source_tree_tracked_in_current_branch": False,
        "provenance_note": "private runtime copies are accepted only through the per-file hashes below",
        "source_ancestry": source_ancestry,
        "preregistration_lock_sha256": preregistration["lock_sha256"],
        "patient_identifiers_in_manifest": False,
        "cells": cells,
    }
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

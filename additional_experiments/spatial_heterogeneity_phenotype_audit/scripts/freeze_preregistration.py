#!/usr/bin/env python3
"""Freeze the no-results plan, config, code, and exact 20 selected checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from common import (  # noqa: E402
    atomic_json,
    canonical_sha256,
    file_sha256,
    load_config,
    ordered_sha256,
    runtime_environment,
)


RESULT_ROOT_NAMES = (
    "features",
    "predictions",
    "metrics",
    "figures",
    "reports",
    "manifests",
    "logs",
    "checkpoints",
    "data",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Accept no formal options so help/typos stop before any filesystem work."""

    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def _scalar(value: Any, name: str) -> Any:
    import numpy as np

    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{name} must be scalar")
    return array.item()


def _reference_identity(
    path: Path,
    *,
    arm: str,
    seed: int,
    fold: int,
    authoritative_split: dict[str, str],
) -> dict[str, Any]:
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        required = {"patient_id", "split", "response_state", "arm", "seed_base", "fold"}
        if set(archive.files) != required:
            raise ValueError(f"reference feature schema drifted: {path}")
        patient_id = np.asarray(archive["patient_id"]).astype(str)
        split = np.asarray(archive["split"]).astype(str)
        state = np.asarray(archive["response_state"])
        observed_arm = str(_scalar(archive["arm"], "arm"))
        observed_seed = int(_scalar(archive["seed_base"], "seed_base"))
        observed_fold = int(_scalar(archive["fold"], "fold"))
    if (observed_arm, observed_seed, observed_fold) != (arm, seed, fold):
        raise ValueError("reference feature cell identity drifted")
    if (
        patient_id.shape != (808,)
        or split.shape != (808,)
        or state.shape != (808, 4, 192)
    ):
        raise ValueError("reference feature shape drifted")
    if state.dtype != np.float32 or not np.isfinite(state).all():
        raise ValueError("reference feature must be finite float32")
    if len(set(patient_id)) != 808 or set(split) != {"train", "val", "test"}:
        raise ValueError("reference patient/split contract drifted")
    if set(patient_id) != set(authoritative_split):
        raise ValueError(
            "reference patients differ from the authoritative fold manifest"
        )
    expected_split = np.asarray([authoritative_split[value] for value in patient_id])
    if not np.array_equal(split, expected_split):
        raise ValueError("reference splits differ from the authoritative fold manifest")
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "metadata_path": str(path.with_suffix(".metadata.json")),
        "metadata_sha256": file_sha256(path.with_suffix(".metadata.json")),
        "patient_order_sha256": ordered_sha256(patient_id),
        "split_order_sha256": ordered_sha256(split),
    }


def _git_provenance(config: dict[str, Any]) -> dict[str, Any]:
    repo = ROOT.parents[1]

    def run(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", *arguments], cwd=repo, text=True, stderr=subprocess.STDOUT
        ).strip()

    branch = run("branch", "--show-current")
    if branch != config["branch"]:
        raise ValueError(f"formal branch differs from config: {branch}")
    changed = sorted(
        set(filter(None, run("diff", "--name-only").splitlines()))
        | set(filter(None, run("diff", "--cached", "--name-only").splitlines()))
    )
    untracked = sorted(
        filter(None, run("ls-files", "--others", "--exclude-standard").splitlines())
    )
    allowed_prefix = "additional_experiments/spatial_heterogeneity_phenotype_audit/"
    outside = [
        path for path in changed + untracked if not path.startswith(allowed_prefix)
    ]
    if outside:
        raise RuntimeError(f"worktree changes outside the new experiment: {outside}")
    return {
        "branch": branch,
        "base_head": run("rev-parse", "HEAD"),
        "tracked_paths_before_freeze": changed,
        "untracked_paths_before_freeze": untracked,
        "all_dirty_paths_confined_to_new_experiment": True,
    }


def main() -> None:
    parse_args()
    config_path = ROOT / "configs" / "audit.json"
    output_path = ROOT / "PREREGISTRATION_LOCK.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen lock: {output_path}")
    result_roots = tuple(ROOT / name for name in RESULT_ROOT_NAMES)
    preexisting_results = [
        path
        for directory in result_roots
        for path in directory.rglob("*")
        if path.is_file() and path.name != ".gitkeep"
    ]
    if preexisting_results:
        display = [str(path.relative_to(ROOT)) for path in preexisting_results]
        raise RuntimeError(f"analysis outputs exist before preregistration: {display}")

    config = load_config(config_path, verify_inputs=True)
    paths = config["paths"]
    import pandas as pd

    fold_frame = pd.read_csv(
        paths["fold_manifest"],
        usecols=["patient_id", "fold", "split"],
        dtype={"patient_id": str},
    )
    fold_frame["fold"] = pd.to_numeric(fold_frame["fold"], errors="raise").astype(int)
    fold_frame["split"] = fold_frame["split"].astype(str).str.lower()
    authoritative_split: dict[int, dict[str, str]] = {}
    for fold in config["frozen_cells"]["folds"]:
        current = fold_frame.loc[fold_frame["fold"].eq(int(fold))]
        if len(current) != 808 or current["patient_id"].duplicated().any():
            raise ValueError(f"authoritative fold {fold} is not 808 unique patients")
        if set(current["split"]) != {"train", "val", "test"}:
            raise ValueError(f"authoritative fold {fold} lacks train/val/test")
        authoritative_split[int(fold)] = dict(
            zip(current["patient_id"].astype(str), current["split"], strict=True)
        )
    cells: dict[str, Any] = {}
    per_fold_identity: dict[int, tuple[str, str]] = {}
    for seed in config["frozen_cells"]["seed_bases"]:
        for arm in config["frozen_cells"]["arms"]:
            for fold in config["frozen_cells"]["folds"]:
                key = f"seed_{seed}/{arm}/fold_{fold}"
                checkpoint = (
                    paths["checkpoint_root"]
                    / f"seed_{seed}"
                    / arm
                    / f"fold_{fold}"
                    / config["frozen_cells"]["selected_checkpoint_filename"]
                )
                selection = checkpoint.with_name("selection.json")
                reference = (
                    paths["local_feature_root"]
                    / f"seed_{seed}"
                    / arm
                    / f"fold_{fold}"
                    / "response_state.private.npz"
                )
                if (
                    not checkpoint.is_file()
                    or not selection.is_file()
                    or not reference.is_file()
                ):
                    raise FileNotFoundError(f"incomplete formal cell {key}")
                identity = _reference_identity(
                    reference,
                    arm=arm,
                    seed=int(seed),
                    fold=int(fold),
                    authoritative_split=authoritative_split[int(fold)],
                )
                current_identity = (
                    identity["patient_order_sha256"],
                    identity["split_order_sha256"],
                )
                if (
                    int(fold) in per_fold_identity
                    and per_fold_identity[int(fold)] != current_identity
                ):
                    raise ValueError(
                        f"patient/split order differs across cells for fold {fold}"
                    )
                per_fold_identity[int(fold)] = current_identity
                cells[key] = {
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": file_sha256(checkpoint),
                    "selection_path": str(selection),
                    "selection_sha256": file_sha256(selection),
                    "reference": identity,
                }
    if len(cells) != 20:
        raise AssertionError("preregistration must bind exactly 20 LOCAL cells")

    implementation_paths = (
        ROOT / "scripts" / "common.py",
        ROOT / "scripts" / "pooling.py",
        ROOT / "scripts" / "build_oracle_sidecars.py",
        ROOT / "scripts" / "export_features.py",
        ROOT / "scripts" / "run_feature_matrix.py",
        ROOT / "scripts" / "run_audit.py",
        ROOT / "scripts" / "verify_cache_integrity.py",
        ROOT / "scripts" / "stage_b_pilot.py",
        ROOT / "scripts" / "generate_figures.py",
        ROOT / "scripts" / "generate_report.py",
        ROOT / "scripts" / "validate_results.py",
        Path(__file__),
    )
    missing_implementations = [
        str(path) for path in implementation_paths if not path.is_file()
    ]
    if missing_implementations:
        raise FileNotFoundError(
            f"formal implementation is incomplete: {missing_implementations}"
        )
    implementation_sha256 = {
        str(path.relative_to(ROOT)): file_sha256(path) for path in implementation_paths
    }

    required_upstream = (
        (
            paths["source_repo"]
            / "additional_experiments"
            / "c1b_spatial_pooling_bottleneck_audit"
            / "src"
            / "c1b_spatial_audit"
            / "pooling.py",
            config["upstream_code"]["spatial_pooling_sha256"],
        ),
        (
            ROOT.parent
            / "mri_clinical_complementarity_audit"
            / "scripts"
            / "data_contracts.py",
            config["upstream_code"]["complementarity_data_contracts_sha256"],
        ),
        (
            ROOT.parent
            / "mri_clinical_complementarity_audit"
            / "scripts"
            / "modeling.py",
            config["upstream_code"]["complementarity_modeling_sha256"],
        ),
    )
    for path, expected in required_upstream:
        if file_sha256(path) != expected:
            raise ValueError(f"configured upstream-code hash drifted: {path}")
    upstream_roots = (
        paths["source_repo"]
        / "additional_experiments"
        / "c1b_overlap_eligibility_ftv_stageb"
        / "src"
        / "c1b_stage_b",
        paths["source_repo"]
        / "additional_experiments"
        / "c1b_model_ready_ftv_sanity"
        / "src"
        / "c1b_sanity",
        paths["source_repo"]
        / "additional_experiments"
        / "c1b_spatial_pooling_bottleneck_audit"
        / "src"
        / "c1b_spatial_audit",
        paths["source_repo"]
        / "additional_experiments"
        / "local_global_response_state_pilot"
        / "src"
        / "lg_response_pilot",
        paths["source_repo"]
        / "additional_experiments"
        / "g3_multiseed_generalization"
        / "src"
        / "dgrs",
    )
    upstream_paths = sorted(
        {path.resolve() for root in upstream_roots for path in root.rglob("*.py")}
        | {path.resolve() for path, _expected in required_upstream}
    )
    upstream_code_sha256: dict[str, str] = {}
    for path in upstream_paths:
        upstream_code_sha256[str(path)] = file_sha256(path)

    git_provenance = _git_provenance(config)

    payload = {
        "schema_version": 2,
        "status": "FROZEN_BEFORE_FEATURE_EXTRACTION_OR_PROBING",
        "branch": config["branch"],
        "formal_cell_count": len(cells),
        "plan_sha256": file_sha256(ROOT / "EXPERIMENT_PLAN.md"),
        "config_sha256": file_sha256(config_path),
        "privacy_policy_sha256": file_sha256(ROOT / ".gitignore"),
        "config_canonical_sha256": canonical_sha256(config),
        "selected_cells": cells,
        "implementation_sha256": implementation_sha256,
        "upstream_code_sha256": upstream_code_sha256,
        "runtime_environment": runtime_environment(),
        "git_provenance_before_freeze": git_provenance,
        "analysis_outputs_present_before_freeze": bool(preexisting_results),
        "analysis_outputs_before_freeze": [],
    }
    atomic_json(payload, output_path)
    print(
        json.dumps(
            {"status": payload["status"], "formal_cell_count": 20}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()

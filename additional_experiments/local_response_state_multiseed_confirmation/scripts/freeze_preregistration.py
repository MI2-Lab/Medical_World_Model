#!/usr/bin/env python3
"""Freeze or verify the LOCAL multi-seed confirmation before formal results."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Mapping


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
CONFIG_PATH = EXPERIMENT_ROOT / "configs" / "confirmation.json"
LOCK_PATH = EXPERIMENT_ROOT / "PREREGISTRATION_LOCK.json"
PRIVATE_INPUT_ROOT_ENV = "MWM_PRIVATE_INPUT_REPO_ROOT"
RESULT_DIRS = (
    "checkpoints",
    "features",
    "predictions",
    "metrics",
    "figures",
    "logs",
    "reports",
)
CODE_PATTERNS = (
    ".gitignore",
    "EXPERIMENT_PLAN.md",
    "configs/*.json",
    "scripts/*.py",
    "src/**/*.py",
    "tests/**/*.py",
)
DATA_CONTRACT_RELATIVE = (
    "additional_experiments/c1b_overlap_eligibility_ftv_stageb/"
    "manifests/stage_b_data_contract.private.json"
)
UPSTREAM_PATHS: Mapping[str, str] = {
    "stage_a_sentinel_sha256": "additional_experiments/c1b_overlap_eligibility_ftv_stageb/STAGE_A_GO.json",
    "data_contract_sha256": DATA_CONTRACT_RELATIVE,
    "g3_config_sha256": "additional_experiments/g3_multiseed_generalization/src/dgrs/config.py",
    "g3_data_sha256": "additional_experiments/g3_multiseed_generalization/src/dgrs/data.py",
    "g3_model_sha256": "additional_experiments/g3_multiseed_generalization/src/dgrs/model.py",
    "g3_package_init_sha256": "additional_experiments/g3_multiseed_generalization/src/dgrs/__init__.py",
    "g3_objective_sha256": "additional_experiments/g3_multiseed_generalization/src/dgrs/training.py",
    "g3_ftv_transform_sha256": "additional_experiments/g3_multiseed_generalization/src/dgrs/targets.py",
    "stage_b_logical_training_sha256": "additional_experiments/c1b_overlap_eligibility_ftv_stageb/src/c1b_stage_b/training.py",
    "stage_b_package_init_sha256": "additional_experiments/c1b_overlap_eligibility_ftv_stageb/src/c1b_stage_b/__init__.py",
    "stage_b_contracts_sha256": "additional_experiments/c1b_overlap_eligibility_ftv_stageb/src/c1b_stage_b/contracts.py",
    "stage_b_data_sha256": "additional_experiments/c1b_overlap_eligibility_ftv_stageb/src/c1b_stage_b/data.py",
    "stage_b_gate_sha256": "additional_experiments/c1b_overlap_eligibility_ftv_stageb/src/c1b_stage_b/gate.py",
    "stage_b_inputs_sha256": "additional_experiments/c1b_overlap_eligibility_ftv_stageb/src/c1b_stage_b/inputs.py",
    "stage_b_targets_adapter_sha256": "additional_experiments/c1b_overlap_eligibility_ftv_stageb/src/c1b_stage_b/targets.py",
    "stage_b_upstream_sha256": "additional_experiments/c1b_overlap_eligibility_ftv_stageb/src/c1b_stage_b/upstream.py",
    "audited_pooling_package_init_sha256": "additional_experiments/c1b_spatial_pooling_bottleneck_audit/src/c1b_spatial_audit/__init__.py",
    "audited_pooling_sha256": "additional_experiments/c1b_spatial_pooling_bottleneck_audit/src/c1b_spatial_audit/pooling.py",
}
EXPECTED_ARMS = ("GAP0", "GAP3", "LOCAL0", "LOCAL3")
EXPECTED_SEEDS = (2026, 3026, 4026, 5026, 6026)
EXPECTED_FOLDS = tuple(range(5))
EXPECTED_ARM_SPECS = {
    "GAP0": {"pooling": "GAP", "projection": "Linear(128,192)+LayerNorm(192)", "grounded": False},
    "GAP3": {"pooling": "GAP", "projection": "Linear(128,192)+LayerNorm(192)", "grounded": True},
    "LOCAL0": {"pooling": "fixed_64mm_LOCAL", "projection": "Linear(128,192)+LayerNorm(192)", "grounded": False},
    "LOCAL3": {"pooling": "fixed_64mm_LOCAL", "projection": "Linear(128,192)+LayerNorm(192)", "grounded": True},
}
EXPECTED_SCOPE = {
    "only_changed_segment": "encoder final spatial feature map -> response state",
    "input": "C1B-H DCE7",
    "grounding_target": "FTV_t only",
    "forbidden": [
        "new_crop", "new_encoder", "LD", "SPH", "BPE",
        "delta_FTV_supervision", "pCR_supervision",
        "clinical_or_treatment_supervision",
        "attention_or_learned_spatial_pooling",
        "LOCAL_GLOBAL_or_global_branch", "window_size_sweep",
        "lesion_mask_recentering", "visit_adaptive_crop",
        "PCGrad_or_optimization_stabilization",
    ],
}
EXPECTED_POPULATION = {
    "technical_eligibility_patients": 947,
    "formal_primary_patients": 375,
    "formal_primary_visits": 1500,
    "grounding_observable_visits": 1486,
    "fold_manifest": "reuse_exact_stage_b_seed2026_patient_folds_without_refill_or_movement",
    "fold_external_train_only_patients": 139,
}
EXPECTED_INPUT_CONTRACT = {
    "tensor": "float32 [B,4,7,112,176,160]",
    "shape_zyx": [112, 176, 160],
    "spacing_xyz_mm": [0.9, 0.9, 2.0],
    "model_fields": ["image"],
    "loss_side_fields": ["ftv_target", "ftv_mask"],
    "geometry_or_mask_is_model_input": False,
}
EXPECTED_SPATIAL_STATE = {
    "source_module": "encoder.features[3] full residual-block output",
    "channels": 128,
    "feature_shape_policy": "derive_and_validate_at_runtime; never hard-code 14x22x20",
    "global": "exact mean over final D,H,W",
    "local_window_mm_xyz": [64.0, 64.0, 64.0],
    "local_center": "frozen C1B-H crop center",
    "local_weighting": "fractional overlap between final feature sampling cells and the fixed physical cube",
    "local_forbidden_adaptation": [
        "visit_lesion_mask", "FTV", "outcome", "performance", "window_sweep"
    ],
    "valid_source_pooling": False,
}
EXPECTED_PAIRED_INITIALIZATION = {
    "gap_and_local": "identical baseline Linear W,b and LayerNorm initialization",
    "paired_shared_modules": ["encoder", "projector", "transition", "target copies"],
    "paired_randomness": ["patient order", "dropout", "SIGReg directions"],
}
EXPECTED_OBJECTIVE = {
    "formula": "L_JEPA + 0.25 * L_FTV for grounded arms; L_JEPA otherwise",
    "lambda_ftv": 0.25,
    "sigreg_weight": 0.09,
    "step_weights": [2.0, 1.0, 0.5],
    "grounding_mask": "measurement_valid AND frozen grounding_observable_mask; loss-side only",
    "ftv_transform": "outer-train observable log(FTV+epsilon), 1/99 winsorization, median/IQR",
    "delta_supervision": False,
}
EXPECTED_TRAINING = {
    "seed_bases": list(EXPECTED_SEEDS),
    "folds": list(EXPECTED_FOLDS),
    "formal_cells": 100,
    "physical_batch_size": 4,
    "accumulation_steps": 8,
    "logical_batch_size": 32,
    "sigreg": "one exact nonlinear B32 reduction with exact surrogate gradient",
    "per_logical_batch": ["one_gradient_clip", "one_AdamW_step", "one_EMA_update"],
    "epochs": 12,
    "patience": 4,
    "learning_rate": 5e-5,
    "weight_decay": 1e-4,
    "ema_momentum": 0.996,
    "max_grad_norm": 5.0,
    "min_representation_std": 0.05,
    "workers": 2,
    "global_oom_fallback": "not_authorized; stop and require a new preregistration and code revision",
}
EXPECTED_CHECKPOINT_SELECTION = {
    "no_grounding": "minimum finite validation state loss among non-collapsed epochs",
    "grounded": "minimum validation FTV loss among finite non-collapsed epochs whose validation state loss is <=1.05 times paired no-grounding selection",
    "fallback": "minimum state-gate violation then validation FTV; marks cell failed",
    "forbidden_selection_inputs": ["test_FTV", "delta_FTV", "pCR"],
}
EXPECTED_PROBES = {
    "feature": "selected frozen online pre-projector r_t [N,4,192]",
    "ridge_alphas": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    "selection": "outer train fit and validation analysis-space MSE; smallest-alpha tie break",
    "static_endpoints": ["T0", "T1", "T2", "T3", "macro"],
    "delta_endpoints": ["T0_to_T1", "T1_to_T2", "T2_to_T3", "macro"],
    "delta_definition": "literal natural FTV_(t+1)-FTV_t from delta r",
    "natural_aggregation": "pool five outer-test folds before each endpoint metric; macro is unweighted endpoint mean",
    "transformed_aggregation": "fold summaries only; never pool incompatible fold transforms",
    "primary_scope": "measurement_valid",
    "sensitivity_scope": "measurement_valid AND grounding_observable",
}
EXPECTED_OPERATIONALIZATION = {
    "independent_unit": "training_seed",
    "seed_result": "five-fold pooled OOF result",
    "strictly_greater_thresholds_are_not_greater_or_equal": True,
    "positive_direction": "effect is strictly greater than zero",
    "natural_r2_systematic_worsening": "at least 4 of 5 seed effects are strictly negative",
    "thresholds_are_descriptive_not_statistical_significance": True,
    "folds_are_paired_sensitivity_not_independent_replicates": True,
}
EXPECTED_GATES = {
    "LOCAL_CONFIRMATION": {
        "local0_gap0_static_macro_spearman_seed_effect_strictly_gt": 0.10,
        "local0_gap0_static_macro_spearman_seeds_required": 4,
        "local0_gap0_static_macro_spearman_seed_mean_strictly_gt": 0.10,
        "local0_gap0_delta_macro_spearman_positive_seeds_required": 4,
        "local0_gap0_static_natural_r2_systematic_worsening_forbidden": True,
        "local3_local0_static_macro_spearman_positive_seeds_required": 4,
        "local3_local0_delta_macro_spearman_positive_seeds_required": 4,
        "local3_local0_optimization_safety_pass_fraction_min": 0.90,
        "local3_local0_optimization_safety_paired_folds_total": 25,
        "local3_local0_optimization_safety_paired_folds_required": 23,
        "maximum_state_loss_degradation_fraction": 0.05,
    }
}
EXPECTED_STATISTICS = {
    "seed_summary": ["mean", "median", "sample_SD", "bootstrap_95pct_CI", "min", "max", "direction_count"],
    "bootstrap_unit": "training_seed",
    "bootstrap_resamples": 10000,
    "bootstrap_seed": 20260811,
    "bootstrap_interval": "percentile_2.5_97.5",
    "fold_analysis": "paired sensitivity only",
}
EXPECTED_SELECTION_RULE = {
    "all_confirmation_criteria_pass": "LOCAL_MULTISEED_CONFIRMED",
    "otherwise": "LOCAL_MULTISEED_NOT_CONFIRMED",
}
EXPECTED_NEXT_STAGE_POLICY = {
    "lock_local_as_image_state_architecture_if_confirmed": True,
    "ftv_plus_ld_condition": "only after LOCAL_MULTISEED_CONFIRMED",
    "confirmation_scope_limit": "FTV grounding addresses tumor burden/response information only; confirmation does not establish full MRI utilization or complementary tumor phenotype beyond Patient Profile",
}
EXPECTED_TOP_LEVEL_KEYS = {
    "schema_version", "experiment", "status_before_formal_results", "scope",
    "upstream", "population", "input_contract", "spatial_state", "arms",
    "paired_initialization", "objective", "training", "checkpoint_selection",
    "probes", "gate_operationalization", "gates", "statistics",
    "selection_rule", "next_stage_policy",
}
EXPECTED_LOCK_KEYS = {
    "schema_version", "status", "frozen_at_utc", "experiment", "git",
    "formal_result_file_count_before_lock", "code_and_plan_sha256",
    "upstream_sha256", "config_sha256", "decision_rules",
    "gate_operationalization", "statistics", "matrix",
    "thresholds_are_not_statistical_significance",
    "ftv_plus_ld_authorized_by_this_lock",
}
EXPERIMENT_REPOSITORY_RELATIVE = Path(
    "additional_experiments/local_response_state_multiseed_confirmation"
)


def _exact(value: Any, expected: Any, label: str) -> None:
    observed = json.dumps(value, sort_keys=True, separators=(",", ":"))
    frozen = json.dumps(expected, sort_keys=True, separators=(",", ":"))
    if observed != frozen:
        raise ValueError(f"confirmation config {label} drifted from the frozen contract")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_inventory() -> dict[str, str]:
    paths: set[Path] = set()
    for pattern in CODE_PATTERNS:
        paths.update(path for path in EXPERIMENT_ROOT.glob(pattern) if path.is_file())
    paths.discard(LOCK_PATH)
    return {
        str(path.relative_to(REPO_ROOT)): file_sha256(path)
        for path in sorted(paths)
    }


def result_files() -> list[str]:
    files: list[str] = []
    for directory in RESULT_DIRS:
        root = EXPERIMENT_ROOT / directory
        if not root.exists():
            continue
        files.extend(
            str(path.relative_to(REPO_ROOT))
            for path in root.rglob("*")
            if path.is_file() and path != root / ".gitkeep"
        )
    return sorted(files)


def git_value(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def require_old_experiment_trees_unchanged(base_sha: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", str(base_sha)) is None:
        raise ValueError("old-tree integrity base SHA is invalid")
    excluded = f":(exclude,glob){EXPERIMENT_REPOSITORY_RELATIVE.as_posix()}/**"
    commands = (
        ("diff", "--exit-code", str(base_sha), "--", "additional_experiments", excluded),
        ("diff", "--cached", "--exit-code", str(base_sha), "--", "additional_experiments", excluded),
    )
    for command in commands:
        subprocess.run(
            ["git", *command], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True,
        )
    for ignored in (False, True):
        command = [
            "git", "status", "--porcelain", "--untracked-files=all",
        ]
        if ignored:
            command.append("--ignored=matching")
        command.extend(("--", "additional_experiments", excluded))
        observed = subprocess.run(
            command, cwd=REPO_ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
        if observed:
            raise RuntimeError(
                "old additional_experiments tree is not pristine: "
                + observed.splitlines()[0]
            )


def require_current_git_matches_lock(lock_git: Mapping[str, Any]) -> None:
    current_branch = git_value("branch", "--show-current")
    current_head = git_value("rev-parse", "HEAD")
    if current_branch != lock_git.get("branch"):
        raise RuntimeError("current branch differs from the preregistration lock")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", str(lock_git.get("head")), current_head],
        cwd=REPO_ROOT,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("preregistration HEAD is not an ancestor of current HEAD")


def resolve_upstream_path(relative: str) -> Path:
    raw_root = os.environ.get(PRIVATE_INPUT_ROOT_ENV, "").strip()
    if relative != DATA_CONTRACT_RELATIVE:
        candidate = (REPO_ROOT / relative).resolve()
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return candidate
    if raw_root:
        external_root = Path(raw_root).expanduser().resolve()
    else:
        external_root = REPO_ROOT.resolve()
    candidate = (external_root / relative).resolve()
    if not raw_root and not candidate.is_file():
        raise FileNotFoundError(
            f"{candidate}; set {PRIVATE_INPUT_ROOT_ENV} to the authorized source repository"
        )
    try:
        candidate.relative_to(external_root)
    except ValueError as error:
        raise ValueError("private input escaped the authorized repository root") from error
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def load_config() -> dict[str, Any]:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != EXPECTED_TOP_LEVEL_KEYS:
        raise ValueError("confirmation config top-level schema drifted")
    if payload.get("schema_version") != 1:
        raise ValueError("confirmation config must be schema version 1")
    if payload.get("experiment") != "local_response_state_multiseed_confirmation":
        raise ValueError("confirmation experiment identity drifted")
    if payload.get("status_before_formal_results") != "PREREGISTERED_PENDING_LOCK":
        raise ValueError("confirmation preregistration status drifted")
    _exact(payload.get("arms"), EXPECTED_ARM_SPECS, "arm specifications")
    _exact(payload.get("scope"), EXPECTED_SCOPE, "scope")
    _exact(payload.get("population"), EXPECTED_POPULATION, "population")
    _exact(payload.get("input_contract"), EXPECTED_INPUT_CONTRACT, "input contract")
    _exact(payload.get("spatial_state"), EXPECTED_SPATIAL_STATE, "spatial state")
    _exact(payload.get("paired_initialization"), EXPECTED_PAIRED_INITIALIZATION, "pairing")
    _exact(payload.get("objective"), EXPECTED_OBJECTIVE, "FTV-only objective")
    _exact(payload.get("training"), EXPECTED_TRAINING, "training")
    _exact(payload.get("checkpoint_selection"), EXPECTED_CHECKPOINT_SELECTION, "selection")
    _exact(payload.get("probes"), EXPECTED_PROBES, "probes")
    _exact(payload.get("gate_operationalization"), EXPECTED_OPERATIONALIZATION, "operationalization")
    _exact(payload.get("gates"), EXPECTED_GATES, "confirmation gate")
    _exact(payload.get("statistics"), EXPECTED_STATISTICS, "statistics")
    _exact(payload.get("selection_rule"), EXPECTED_SELECTION_RULE, "selection rule")
    _exact(payload.get("next_stage_policy"), EXPECTED_NEXT_STAGE_POLICY, "next-stage policy")
    upstream = payload.get("upstream")
    if not isinstance(upstream, dict) or set(upstream) != set(UPSTREAM_PATHS) | {
        "stage_a_sentinel", "data_contract"
    }:
        raise ValueError("upstream declaration inventory drifted")
    if upstream.get("stage_a_sentinel") != UPSTREAM_PATHS["stage_a_sentinel_sha256"]:
        raise ValueError("Stage-A sentinel path drifted")
    if upstream.get("data_contract") != DATA_CONTRACT_RELATIVE:
        raise ValueError("private data-contract identity drifted")
    return payload


def lock_declarations(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_FORMAL_RESULTS",
        "experiment": "local_response_state_multiseed_confirmation",
        "formal_result_file_count_before_lock": 0,
        "decision_rules": config["gates"],
        "gate_operationalization": config["gate_operationalization"],
        "statistics": config["statistics"],
        "matrix": {
            "arms": list(EXPECTED_ARMS),
            "seeds": list(EXPECTED_SEEDS),
            "folds": list(EXPECTED_FOLDS),
            "cells": 100,
        },
        "thresholds_are_not_statistical_significance": True,
        "ftv_plus_ld_authorized_by_this_lock": False,
    }


def upstream_inventory(config: Mapping[str, Any]) -> dict[str, str]:
    declared = config.get("upstream")
    if not isinstance(declared, Mapping):
        raise ValueError("confirmation config lacks upstream declarations")
    observed: dict[str, str] = {}
    for key, relative in UPSTREAM_PATHS.items():
        path = resolve_upstream_path(relative)
        digest = file_sha256(path)
        if declared.get(key) != digest:
            raise ValueError(f"upstream hash mismatch for {relative}")
        observed[relative] = digest
    return observed


def validate_lock_payload(payload: Any, config: Mapping[str, Any]) -> None:
    if not isinstance(payload, dict) or set(payload) != EXPECTED_LOCK_KEYS:
        raise ValueError("preregistration lock schema drifted")
    declarations = lock_declarations(config)
    for key, expected in declarations.items():
        _exact(payload.get(key), expected, f"lock {key}")
    if payload.get("formal_result_file_count_before_lock") != 0:
        raise ValueError("formal results existed before lock")
    git = payload.get("git")
    if not isinstance(git, Mapping):
        raise ValueError("lock git provenance is invalid")
    if git.get("branch") != "feature/local-response-state-multiseed-confirmation":
        raise ValueError("lock was created on the wrong branch")
    if re.fullmatch(r"[0-9a-f]{40}", str(git.get("head", ""))) is None:
        raise ValueError("lock git HEAD is invalid")
    frozen_at = payload.get("frozen_at_utc")
    if not isinstance(frozen_at, str):
        raise ValueError("lock freeze timestamp is invalid")
    try:
        parsed_timestamp = datetime.fromisoformat(frozen_at)
    except ValueError as error:
        raise ValueError("lock freeze timestamp is invalid") from error
    if parsed_timestamp.tzinfo is None:
        raise ValueError("lock freeze timestamp must be timezone-aware")
    if not isinstance(payload.get("code_and_plan_sha256"), dict):
        raise ValueError("lock code inventory is invalid")
    if not isinstance(payload.get("upstream_sha256"), dict):
        raise ValueError("lock upstream inventory is invalid")
    if re.fullmatch(r"[0-9a-f]{64}", str(payload.get("config_sha256", ""))) is None:
        raise ValueError("lock config hash is invalid")


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def freeze() -> dict[str, Any]:
    if LOCK_PATH.exists():
        raise FileExistsError("preregistration lock already exists; use --verify")
    existing = result_files()
    if existing:
        raise RuntimeError(
            "formal/result artifacts already exist before preregistration: "
            + ", ".join(existing[:5])
        )
    config = load_config()
    branch = git_value("branch", "--show-current")
    head = git_value("rev-parse", "HEAD")
    if branch != "feature/local-response-state-multiseed-confirmation":
        raise RuntimeError("refusing to freeze on the wrong branch")
    require_old_experiment_trees_unchanged(head)
    payload: dict[str, Any] = {
        **lock_declarations(config),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {"branch": branch, "head": head},
        "code_and_plan_sha256": code_inventory(),
        "upstream_sha256": upstream_inventory(config),
        "config_sha256": file_sha256(CONFIG_PATH),
    }
    validate_lock_payload(payload, config)
    atomic_json(LOCK_PATH, payload)
    return payload


def verify() -> dict[str, Any]:
    if not LOCK_PATH.is_file():
        raise FileNotFoundError("preregistration lock does not exist")
    payload = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    config = load_config()
    validate_lock_payload(payload, config)
    lock_git = payload["git"]
    require_current_git_matches_lock(lock_git)
    require_old_experiment_trees_unchanged(lock_git["head"])
    if payload.get("code_and_plan_sha256") != code_inventory():
        raise RuntimeError("preregistered code/plan inventory drifted")
    if payload.get("config_sha256") != file_sha256(CONFIG_PATH):
        raise RuntimeError("preregistered config drifted")
    if payload.get("upstream_sha256") != upstream_inventory(config):
        raise RuntimeError("preregistered upstream inventory drifted")
    if result_files() and payload.get("formal_result_file_count_before_lock") != 0:
        raise RuntimeError("preregistration result-file declaration drifted")
    return {
        "status": "PASS",
        "lock_path": str(LOCK_PATH),
        "lock_sha256": file_sha256(LOCK_PATH),
        "config_sha256": payload["config_sha256"],
        "upstream_sha256": dict(payload["upstream_sha256"]),
        "verified_code_files": len(payload["code_and_plan_sha256"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    payload = verify() if args.verify else freeze()
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

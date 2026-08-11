#!/usr/bin/env python3
"""Freeze or verify the pilot before any formal result is created."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
CONFIG_PATH = EXPERIMENT_ROOT / "configs" / "pilot.json"
LOCK_PATH = EXPERIMENT_ROOT / "PREREGISTRATION_LOCK.json"
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
UPSTREAM_PATHS = {
    "stage_a_sentinel_sha256": "additional_experiments/c1b_overlap_eligibility_ftv_stageb/STAGE_A_GO.json",
    "data_contract_sha256": "additional_experiments/c1b_overlap_eligibility_ftv_stageb/manifests/stage_b_data_contract.private.json",
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
EXPECTED_ARMS = ("GAP0", "GAP3", "LOCAL0", "LOCAL3", "LG0", "LG3")
EXPECTED_SEEDS = (2026, 3026)
EXPECTED_FOLDS = tuple(range(5))
EXPECTED_BATCH_CONTRACT = {
    "physical_batch_size": 4,
    "accumulation_steps": 8,
    "logical_batch_size": 32,
}
EXPECTED_CONFIG_KEYS = {
    "schema_version",
    "experiment",
    "status_before_formal_results",
    "scope",
    "upstream",
    "population",
    "input_contract",
    "spatial_state",
    "arms",
    "paired_initialization",
    "objective",
    "training",
    "checkpoint_selection",
    "probes",
    "gate_operationalization",
    "gates",
    "selection_rule",
    "next_stage_policy",
}
EXPECTED_ARM_SPECS = {
    "GAP0": {
        "pooling": "GAP",
        "projection": "Linear(128,192)+LayerNorm(192)",
        "grounded": False,
    },
    "GAP3": {
        "pooling": "GAP",
        "projection": "Linear(128,192)+LayerNorm(192)",
        "grounded": True,
    },
    "LOCAL0": {
        "pooling": "fixed_64mm_LOCAL",
        "projection": "Linear(128,192)+LayerNorm(192)",
        "grounded": False,
    },
    "LOCAL3": {
        "pooling": "fixed_64mm_LOCAL",
        "projection": "Linear(128,192)+LayerNorm(192)",
        "grounded": True,
    },
    "LG0": {
        "pooling": "concat(fixed_64mm_LOCAL,GAP)",
        "projection": "Linear(256,192)+LayerNorm(192)",
        "grounded": False,
    },
    "LG3": {
        "pooling": "concat(fixed_64mm_LOCAL,GAP)",
        "projection": "Linear(256,192)+LayerNorm(192)",
        "grounded": True,
    },
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
    "seed_bases": [2026, 3026],
    "folds": [0, 1, 2, 3, 4],
    "formal_cells": 60,
    "physical_batch_size": 4,
    "accumulation_steps": 8,
    "logical_batch_size": 32,
    "sigreg": "one exact nonlinear B32 reduction with exact surrogate gradient",
    "per_logical_batch": [
        "one_gradient_clip",
        "one_AdamW_step",
        "one_EMA_update",
    ],
    "epochs": 12,
    "patience": 4,
    "learning_rate": 5e-5,
    "weight_decay": 1e-4,
    "ema_momentum": 0.996,
    "max_grad_norm": 5.0,
    "min_representation_std": 0.05,
    "workers": 2,
    "global_oom_fallback": (
        "not_authorized; stop and require a new preregistration and code revision"
    ),
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
    "natural_r2_systematic_worsening": "effect is strictly negative in both seeds",
    "meaningful_delta_spearman_gain": 0.02,
    "thresholds_are_descriptive_not_statistical_significance": True,
    "folds_are_paired_sensitivity_not_independent_replicates": True,
}
EXPECTED_GATES = {
    "A_LOCAL_STATE_WORKS": {
        "static_macro_spearman_gain_each_seed_min": 0.1,
        "delta_macro_spearman_gain_each_seed_strictly_gt": 0.0,
        "static_natural_r2_systematic_worsening_forbidden": True,
    },
    "B_LOCAL_GLOBAL_ADDS_VALUE": {
        "static_macro_spearman_gain_each_seed_min": 0.0,
        "static_macro_spearman_gain_at_least_one_seed_min": 0.02,
        "static_natural_r2_systematic_worsening_forbidden": True,
    },
    "C_GROUNDING_COMPATIBILITY": {
        "candidate_static_macro_spearman_gain_each_seed_strictly_gt": 0.0,
        "candidate_delta_macro_spearman_gain_at_least_one_seed_min": 0.02,
        "static_natural_r2_systematic_worsening_forbidden": True,
    },
    "D_OPTIMIZATION_SAFETY": {
        "candidate_paired_folds_required": 9,
        "candidate_paired_folds_total": 10,
        "maximum_state_loss_degradation_fraction": 0.05,
    },
}
EXPECTED_SELECTION_RULE = {
    "if_A_pass_and_B_fail": "LOCAL",
    "if_A_pass_and_B_pass": "LOCAL_GLOBAL",
    "if_A_fail": "GAP and classification C",
    "classification_D_override": "A passes but grounded final candidate fails Gate D",
}
EXPECTED_NEXT_STAGE_POLICY = {
    "direct_FTV_plus_LD_from_this_pilot": False,
    "after_A_or_B": "first perform larger multi-seed confirmation of the selected architecture",
    "FTV_plus_LD_condition": "only after confirmation, and only if pilot grounding compatibility and safety are supported",
}
EXPECTED_LOCK_KEYS = {
    "schema_version",
    "status",
    "frozen_at_utc",
    "experiment",
    "git",
    "formal_result_file_count_before_lock",
    "code_and_plan_sha256",
    "upstream_sha256",
    "config_sha256",
    "decision_rules",
    "gate_operationalization",
    "matrix",
    "thresholds_are_not_statistical_significance",
    "ftv_plus_ld_authorized_by_this_lock",
}


def _require_exact(value: Any, expected: Any, label: str) -> None:
    observed_json = json.dumps(value, sort_keys=True, separators=(",", ":"))
    expected_json = json.dumps(expected, sort_keys=True, separators=(",", ":"))
    if observed_json != expected_json:
        raise ValueError(f"pilot config {label} drifted from the exact frozen contract")


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
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def load_config() -> dict[str, Any]:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("pilot config must be a schema-v1 JSON object")
    if set(payload) != EXPECTED_CONFIG_KEYS:
        raise ValueError("pilot config top-level schema drifted")
    if payload.get("experiment") != "local_global_response_state_pilot":
        raise ValueError("pilot config experiment identity drifted")
    if payload.get("status_before_formal_results") != "PREREGISTERED_PENDING_LOCK":
        raise ValueError("pilot config preregistration status drifted")
    arms = payload.get("arms")
    if not isinstance(arms, dict) or tuple(arms) != EXPECTED_ARMS:
        raise ValueError("pilot config arm inventory/order drifted")
    _require_exact(arms, EXPECTED_ARM_SPECS, "arm specifications")
    upstream = payload.get("upstream")
    if not isinstance(upstream, dict) or set(upstream) != set(UPSTREAM_PATHS) | {
        "stage_a_sentinel",
        "data_contract",
    }:
        raise ValueError("pilot config upstream hash/key inventory drifted")
    if upstream.get("stage_a_sentinel") != UPSTREAM_PATHS["stage_a_sentinel_sha256"]:
        raise ValueError("pilot config Stage-A sentinel path drifted")
    if upstream.get("data_contract") != UPSTREAM_PATHS["data_contract_sha256"]:
        raise ValueError("pilot config data-contract path drifted")
    training = payload.get("training")
    if not isinstance(training, dict):
        raise ValueError("pilot config lacks a training contract")
    if tuple(training.get("seed_bases", ())) != EXPECTED_SEEDS:
        raise ValueError("pilot config seed bases drifted")
    if tuple(training.get("folds", ())) != EXPECTED_FOLDS:
        raise ValueError("pilot config folds drifted")
    if training.get("formal_cells") != 60:
        raise ValueError("pilot config no longer describes the frozen 60-cell matrix")
    for key, expected in EXPECTED_BATCH_CONTRACT.items():
        if training.get(key) != expected:
            raise ValueError(f"pilot config {key} is frozen to {expected}")
    if (
        training.get("global_oom_fallback")
        != "not_authorized; stop and require a new preregistration and code revision"
    ):
        raise ValueError("pilot config must not authorize an OOM batch fallback")
    _require_exact(training, EXPECTED_TRAINING, "training")
    _require_exact(payload.get("objective"), EXPECTED_OBJECTIVE, "FTV-only objective")
    _require_exact(
        payload.get("checkpoint_selection"),
        EXPECTED_CHECKPOINT_SELECTION,
        "checkpoint selection",
    )
    _require_exact(payload.get("probes"), EXPECTED_PROBES, "probe protocol")
    _require_exact(
        payload.get("gate_operationalization"),
        EXPECTED_OPERATIONALIZATION,
        "gate operationalization",
    )
    _require_exact(payload.get("gates"), EXPECTED_GATES, "Gate A-D thresholds")
    _require_exact(
        payload.get("selection_rule"), EXPECTED_SELECTION_RULE, "selection rule"
    )
    _require_exact(
        payload.get("next_stage_policy"),
        EXPECTED_NEXT_STAGE_POLICY,
        "next-stage policy",
    )
    scope = payload.get("scope")
    if not isinstance(scope, dict) or scope.get("grounding_target") != "FTV_t only":
        raise ValueError("pilot config must remain FTV-only")
    if len(EXPECTED_ARMS) * len(EXPECTED_SEEDS) * len(EXPECTED_FOLDS) != 60:
        raise AssertionError("internal formal matrix cardinality drifted")
    return payload


def lock_declarations(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_FORMAL_RESULTS",
        "experiment": "local_global_response_state_pilot",
        "formal_result_file_count_before_lock": 0,
        "decision_rules": config["gates"],
        "gate_operationalization": config["gate_operationalization"],
        "matrix": {
            "arms": list(EXPECTED_ARMS),
            "seeds": list(EXPECTED_SEEDS),
            "folds": list(EXPECTED_FOLDS),
            "cells": 60,
        },
        "thresholds_are_not_statistical_significance": True,
        "ftv_plus_ld_authorized_by_this_lock": False,
    }


def validate_lock_payload(payload: Any, config: dict[str, Any]) -> None:
    if not isinstance(payload, dict) or set(payload) != EXPECTED_LOCK_KEYS:
        raise ValueError("preregistration lock schema/field inventory is invalid")
    declarations = lock_declarations(config)
    for key, expected in declarations.items():
        if payload.get(key) != expected:
            raise ValueError(f"preregistration lock declaration drifted at {key}")
    timestamp = payload.get("frozen_at_utc")
    if not isinstance(timestamp, str):
        raise ValueError("preregistration lock timestamp is invalid")
    try:
        frozen_at = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("preregistration lock timestamp is invalid") from error
    if frozen_at.tzinfo is None or frozen_at.utcoffset() is None:
        raise ValueError("preregistration lock timestamp must be timezone-aware")
    if frozen_at > datetime.now(timezone.utc) + timedelta(minutes=5):
        raise ValueError("preregistration lock timestamp is implausibly in the future")
    git = payload.get("git")
    if not isinstance(git, dict) or set(git) != {"branch", "head"}:
        raise ValueError("preregistration lock git metadata is invalid")
    if not isinstance(git.get("branch"), str):
        raise ValueError("preregistration lock git branch is invalid")
    if re.fullmatch(r"[0-9a-f]{40,64}", str(git.get("head", ""))) is None:
        raise ValueError("preregistration lock git HEAD is invalid")
    current_git = {
        "branch": git_value("branch", "--show-current"),
        "head": git_value("rev-parse", "HEAD"),
    }
    if git != current_git:
        raise RuntimeError("preregistration lock git revision/branch drifted")
    if not isinstance(payload.get("code_and_plan_sha256"), dict):
        raise ValueError("preregistration lock code inventory is invalid")
    if not isinstance(payload.get("upstream_sha256"), dict):
        raise ValueError("preregistration lock upstream inventory is invalid")
    if re.fullmatch(r"[0-9a-f]{64}", str(payload.get("config_sha256", ""))) is None:
        raise ValueError("preregistration lock config hash is invalid")


def upstream_inventory(config: dict[str, Any]) -> dict[str, str]:
    declared = config.get("upstream")
    if not isinstance(declared, dict):
        raise ValueError("pilot config lacks upstream hash declarations")
    observed: dict[str, str] = {}
    for key, relative in UPSTREAM_PATHS.items():
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = file_sha256(path)
        if declared.get(key) != digest:
            raise ValueError(f"upstream hash mismatch for {relative}")
        observed[relative] = digest
    return observed


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with open(descriptor, "w", encoding="utf-8", closefd=True) as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def freeze() -> dict[str, Any]:
    if LOCK_PATH.exists():
        raise FileExistsError("preregistration lock already exists; use --verify")
    existing_results = result_files()
    if existing_results:
        raise RuntimeError(
            "formal/result artifacts already exist before preregistration: "
            + ", ".join(existing_results[:5])
        )
    config = load_config()
    payload = {
        **lock_declarations(config),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "branch": git_value("branch", "--show-current"),
            "head": git_value("rev-parse", "HEAD"),
        },
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
    expected_code = payload.get("code_and_plan_sha256")
    observed_code = code_inventory()
    if expected_code != observed_code:
        missing = sorted(set(expected_code or {}).difference(observed_code))
        added = sorted(set(observed_code).difference(expected_code or {}))
        changed = sorted(
            path
            for path in set(expected_code or {}).intersection(observed_code)
            if expected_code[path] != observed_code[path]
        )
        raise RuntimeError(
            f"preregistered code drifted; missing={missing}, added={added}, changed={changed}"
        )
    if payload.get("config_sha256") != file_sha256(CONFIG_PATH):
        raise RuntimeError("preregistered config drifted")
    if payload.get("upstream_sha256") != upstream_inventory(config):
        raise RuntimeError("preregistered upstream inventory drifted")
    return {
        "status": "PASS",
        "lock_path": str(LOCK_PATH),
        "lock_sha256": file_sha256(LOCK_PATH),
        "config_sha256": payload["config_sha256"],
        "upstream_sha256": dict(payload["upstream_sha256"]),
        "verified_code_files": len(observed_code),
        "current_result_files": len(result_files()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    payload = verify() if args.verify else freeze()
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

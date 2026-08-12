#!/usr/bin/env python3
"""Validate or aggregate the complete 100-cell LOCAL confirmation matrix."""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path
import re


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
SRC_ROOT = EXPERIMENT_ROOT / "src"
SCRIPTS_ROOT = EXPERIMENT_ROOT / "scripts"
CONFIG_PATH = EXPERIMENT_ROOT / "configs" / "confirmation.json"
FORMAL_FEATURE_ROOT = EXPERIMENT_ROOT / "features"
FORMAL_PREDICTION_ROOT = EXPERIMENT_ROOT / "predictions"
FORMAL_CHECKPOINT_ROOT = EXPERIMENT_ROOT / "checkpoints"
FORMAL_METRICS_ROOT = EXPERIMENT_ROOT / "metrics"
FORMAL_FIGURE_ROOT = EXPERIMENT_ROOT / "figures"
STAGE_A_RELATIVE = (
    "additional_experiments/c1b_overlap_eligibility_ftv_stageb/STAGE_A_GO.json"
)
DATA_CONTRACT_RELATIVE = "additional_experiments/c1b_overlap_eligibility_ftv_stageb/manifests/stage_b_data_contract.private.json"
for source in (SRC_ROOT, SCRIPTS_ROOT):
    value = str(source.resolve())
    while value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)

from freeze_preregistration import verify as verify_preregistration  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_ROOT / "metrics")
    parser.add_argument("--figure-dir", type=Path, default=EXPERIMENT_ROOT / "figures")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Without this flag validate the exact matrix and aggregation inputs only.",
    )
    return parser.parse_args()


def main() -> None:
    preregistration = verify_preregistration()
    import lg_response_pilot.security as confirmation_security

    if (
        Path(str(getattr(confirmation_security, "__file__", ""))).resolve()
        != (SRC_ROOT / "lg_response_pilot" / "security.py").resolve()
    ):
        raise ImportError(
            "confirmation security module was shadowed before aggregation"
        )
    from lg_response_pilot.security import (
        require_canonical_file,
        resolve_contained_path,
    )

    args = parse_args()
    args.config = require_canonical_file(
        args.config,
        CONFIG_PATH,
        preregistration["config_sha256"],
        preregistration["config_sha256"],
        label="confirmation config",
    )
    from lg_response_pilot.analysis import (
        aggregate_confirmation,
        collect_complete_matrix,
        load_confirmation_config,
    )

    import lg_response_pilot.analysis as confirmation_analysis_module

    confirmation_security.require_module_within(
        confirmation_analysis_module,
        SRC_ROOT / "lg_response_pilot",
        label="confirmation analysis",
    )

    config = load_confirmation_config(args.config)
    checkpoint_root = resolve_contained_path(
        args.checkpoint_root,
        FORMAL_CHECKPOINT_ROOT,
        label="formal checkpoint root",
    )
    feature_root = resolve_contained_path(
        args.feature_root,
        FORMAL_FEATURE_ROOT,
        label="formal feature root",
    )
    probe_root = resolve_contained_path(
        args.probe_root,
        FORMAL_PREDICTION_ROOT,
        label="formal probe root",
    )
    postprocessing_completion_path = probe_root / "postprocessing_complete.private.json"
    try:
        postprocessing_completion = json.loads(
            postprocessing_completion_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("postprocessing completion is missing or invalid") from error
    data_provenance_sha256 = str(
        postprocessing_completion.get("data_provenance_sha256", "")
    )
    if re.fullmatch(r"[0-9a-f]{64}", data_provenance_sha256) is None:
        raise ValueError("postprocessing data provenance SHA-256 is invalid")
    output_dir = resolve_contained_path(
        args.output_dir,
        FORMAL_METRICS_ROOT,
        label="public metrics output directory",
        allow_root=True,
    )
    figure_dir = resolve_contained_path(
        args.figure_dir,
        FORMAL_FIGURE_ROOT,
        label="public figure output directory",
        allow_root=True,
    )
    source_provenance = {
        "config_sha256": preregistration["config_sha256"],
        "stage_a_sentinel_sha256": preregistration["upstream_sha256"][STAGE_A_RELATIVE],
        "data_contract_sha256": preregistration["upstream_sha256"][
            DATA_CONTRACT_RELATIVE
        ],
        "data_provenance_sha256": data_provenance_sha256,
    }
    if not args.execute:
        selections, histories, metrics, predictions = collect_complete_matrix(
            checkpoint_root=checkpoint_root,
            feature_root=feature_root,
            probe_root=probe_root,
            config=config,
            preregistration_lock_sha256=preregistration["lock_sha256"],
            source_provenance=source_provenance,
        )
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "PREFLIGHT_PASS",
                    "execution_requested": False,
                    "preregistration_lock": "PREREGISTRATION_LOCK.json",
                    "preregistration_lock_sha256": preregistration["lock_sha256"],
                    **source_provenance,
                    "cells": len(selections),
                    "history_rows": len(histories),
                    "fold_metric_rows": len(metrics),
                    "private_oof_rows": len(predictions),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    summary = aggregate_confirmation(
        checkpoint_root=checkpoint_root,
        feature_root=feature_root,
        probe_root=probe_root,
        config_path=args.config,
        output_dir=output_dir,
        figure_dir=figure_dir,
        preregistration_lock_sha256=preregistration["lock_sha256"],
        source_provenance=source_provenance,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Aggregate frozen probe outputs; never fit, refit, or train a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
FORMAL_STAGE_B_SRC = (
    REPO_ROOT
    / "additional_experiments"
    / "c1b_overlap_eligibility_ftv_stageb"
    / "src"
)
MODEL_READY_SRC = (
    REPO_ROOT / "additional_experiments" / "c1b_model_ready_ftv_sanity" / "src"
)
sys.path[:0] = [str(FORMAL_STAGE_B_SRC), str(MODEL_READY_SRC), str(ROOT / "src")]

from c1b_spatial_audit.analysis import (  # noqa: E402
    aggregate_frozen_results,
    write_aggregation_outputs,
)
from c1b_spatial_audit.contracts import REFERENCE_PROBE_ROOT, file_sha256  # noqa: E402
from c1b_spatial_audit.runtime import verify_preregistration  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--final-probes", type=Path, default=ROOT / "probes/final")
    parser.add_argument("--s3-probes", type=Path, default=ROOT / "probes/s3")
    parser.add_argument(
        "--occupancy",
        type=Path,
        default=ROOT / "manifests/lesion_occupancy.private.csv",
    )
    parser.add_argument(
        "--nuisance",
        type=Path,
        default=ROOT / "manifests/nuisance_targets.private.csv",
    )
    parser.add_argument(
        "--old-predictions", type=Path, default=REFERENCE_PROBE_ROOT
    )
    parser.add_argument(
        "--table7", type=Path, default=ROOT / "metrics/table7_training_budget.csv"
    )
    parser.add_argument(
        "--training-summary",
        type=Path,
        default=ROOT / "metrics/training_budget_summary.json",
    )
    parser.add_argument(
        "--preregistration-lock",
        type=Path,
        default=ROOT / "PREREGISTRATION_LOCK.json",
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/audit.json"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "metrics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_preregistration()
    if not args.execute:
        raise SystemExit(
            "validated preregistration; pass --execute only after the formal probe matrix is complete"
        )
    result = aggregate_frozen_results(
        final_probe_root=args.final_probes,
        s3_probe_root=args.s3_probes,
        occupancy_path=args.occupancy,
        nuisance_path=args.nuisance,
        old_prediction_root=args.old_predictions,
        table7_path=args.table7,
        training_summary_path=args.training_summary,
        preregistration_lock=args.preregistration_lock,
        config_path=args.config,
    )
    outputs = write_aggregation_outputs(result, output_dir=args.output_dir)
    classification = result.gates["classification"]
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "classification": classification["classification"],
                "next": classification["next"],
                "s3_trigger_status": result.gates["conditional_s3"][
                    "trigger_status"
                ],
                "output_sha256": {
                    path.name: file_sha256(path) for path in outputs.values()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

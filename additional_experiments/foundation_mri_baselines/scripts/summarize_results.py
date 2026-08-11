#!/usr/bin/env python3
"""Create locked, identifier-free summaries and figures from formal OOF outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = EXPERIMENT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from foundation_mri.reporting import (  # noqa: E402
    ReportingInputs,
    ReportingOutputs,
    static_comparison_contract,
    summarize_results,
)
from foundation_mri.locking import (  # noqa: E402
    file_sha256,
    verify_evaluation_code_lock,
    verify_historical_metric_free_run_provenance,
)


DEFAULT_EVALUATION_LOCK = EXPERIMENT_ROOT / "configs/REPORTING_LOCK.v1.json"
REPORTING_MARKER = EXPERIMENT_ROOT / "metrics/reporting_run_provenance.json"
FINALIZATION_LOCK = EXPERIMENT_ROOT / "configs/FINALIZATION_LOCK.v1.json"
FORMAL_FOUNDATION_MODELS = (
    "medicalnet_resnet50_3dseg8",
    "dino_vitb16_imagenet1k",
)


def _default_inputs(root: Path) -> ReportingInputs:
    return ReportingInputs(
        baseline_private=root / "predictions/baseline_predictions.private.csv",
        baseline_public=root / "metrics/baseline_metrics.csv",
        phenotype_private=root / "predictions/phenotype_predictions.private.csv",
        phenotype_public=root / "metrics/phenotype_metrics.csv",
        subtype_private=root / "predictions/subtype_predictions.private.csv",
        subtype_public=root / "metrics/subtype_metrics.csv",
        ftv_private=root / "predictions/ftv_probe_predictions.private.csv",
        ftv_public=root / "metrics/ftv_probe_metrics.csv",
    )


def _default_outputs(root: Path) -> ReportingOutputs:
    return ReportingOutputs(
        paired_csv=root / "metrics/paired_bootstrap_comparisons.csv",
        summary_json=root / "metrics/results_summary.json",
        summary_markdown=root / "reports/results_summary.md",
        timing_figure=root / "figures/pcr_timing_performance.png",
        calibration_figure=root / "figures/calibration_clinical_complementarity.png",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=EXPERIMENT_ROOT,
        help="Root containing predictions/ and metrics/ outputs from both evaluation CLIs.",
    )
    parser.add_argument(
        "--evaluation-lock", type=Path, default=DEFAULT_EVALUATION_LOCK
    )
    parser.add_argument(
        "--allow-unlocked-inputs",
        action="store_true",
        help="Synthetic/prospective override; forbidden for the formal report.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXPERIMENT_ROOT,
        help="Root for public metrics, reports, and figures.",
    )
    parser.add_argument(
        "--print-comparison-contract",
        action="store_true",
        help="Print the static pre-test comparison contract as JSON and exit without reading data.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace all five existing public reporting outputs after full validation.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = tuple(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    if args.print_comparison_contract:
        print(
            json.dumps(
                static_comparison_contract(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0
    if not args.allow_unlocked_inputs:
        if raw_argv:
            raise ValueError("formal v3 summarization requires exact empty argv")
        if args.overwrite:
            raise ValueError("formal v3 summarization forbids --overwrite")
        if Path(args.input_dir).resolve() != EXPERIMENT_ROOT:
            raise ValueError("formal v3 summary input-dir must be the experiment root")
        if Path(args.output_dir).resolve() != EXPERIMENT_ROOT:
            raise ValueError("formal v3 summary output-dir must be the experiment root")
        code_receipt = verify_evaluation_code_lock(
            experiment_root=EXPERIMENT_ROOT,
            lock_path=args.evaluation_lock,
            expected_consumer="summarizer",
            command_argv=raw_argv,
        )
        baseline_receipt = verify_historical_metric_free_run_provenance(
            experiment_root=EXPERIMENT_ROOT,
            active_lock_path=args.evaluation_lock,
            producer_key="baseline_v2",
            expected_artifacts={
                "predictions": EXPERIMENT_ROOT
                / "predictions/baseline_predictions.private.csv",
                "selection": EXPERIMENT_ROOT
                / "metrics/baseline_selection.private.csv",
                "metrics": EXPERIMENT_ROOT / "metrics/baseline_metrics.csv",
                "progress": EXPERIMENT_ROOT
                / "logs/baseline_v2.progress.private.jsonl",
            },
        )
        probe_receipt = verify_historical_metric_free_run_provenance(
            experiment_root=EXPERIMENT_ROOT,
            active_lock_path=args.evaluation_lock,
            producer_key="probe_v3",
            expected_artifacts={
                "phenotype_predictions": EXPERIMENT_ROOT
                / "predictions/phenotype_predictions.private.csv",
                "phenotype_selection": EXPERIMENT_ROOT
                / "metrics/phenotype_selection.private.csv",
                "phenotype_metrics": EXPERIMENT_ROOT
                / "metrics/phenotype_metrics.csv",
                "subtype_predictions": EXPERIMENT_ROOT
                / "predictions/subtype_predictions.private.csv",
                "subtype_selection": EXPERIMENT_ROOT
                / "metrics/subtype_selection.private.csv",
                "subtype_metrics": EXPERIMENT_ROOT
                / "metrics/subtype_metrics.csv",
                "ftv_predictions": EXPERIMENT_ROOT
                / "predictions/ftv_probe_predictions.private.csv",
                "ftv_selection": EXPERIMENT_ROOT
                / "metrics/ftv_probe_selection.private.csv",
                "ftv_metrics": EXPERIMENT_ROOT
                / "metrics/ftv_probe_metrics.csv",
                "progress": EXPERIMENT_ROOT
                / "logs/probe_v3.progress.private.jsonl",
            },
        )
        reporting_lineage = {
            "baseline_v2": {
                "protocol_version": "v2",
                "evaluation_lock_sha256": baseline_receipt["lock_sha256"],
                "run_receipt_sha256": baseline_receipt["receipt_sha256"],
                "argv_sha256": baseline_receipt["argument_vector_sha256"],
                "artifact_sha256": baseline_receipt["artifact_sha256"],
            },
            "probe_v3": {
                "protocol_version": "v3",
                "evaluation_lock_sha256": probe_receipt["lock_sha256"],
                "run_receipt_sha256": probe_receipt["receipt_sha256"],
                "argv_sha256": probe_receipt["argument_vector_sha256"],
                "artifact_sha256": probe_receipt["artifact_sha256"],
            },
            "summarizer": {
                "protocol_version": "v3",
                "argv_sha256": code_receipt["argument_vector_sha256"],
                "code_lock_sha256": code_receipt["lock_sha256"],
                "finalization_lock_sha256": file_sha256(FINALIZATION_LOCK),
            },
        }
    else:
        reporting_lineage = None
    summary = summarize_results(
        _default_inputs(args.input_dir),
        _default_outputs(args.output_dir),
        overwrite=args.overwrite,
        expected_foundation_models=(
            FORMAL_FOUNDATION_MODELS if reporting_lineage is not None else None
        ),
        reporting_lineage=reporting_lineage,
        reporting_marker=(REPORTING_MARKER if reporting_lineage is not None else None),
    )
    print(
        "public result summary complete: "
        + ", ".join(f"{key}={value}" for key, value in summary.items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

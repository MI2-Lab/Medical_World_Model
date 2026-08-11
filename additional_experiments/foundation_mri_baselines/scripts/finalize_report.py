#!/usr/bin/env python3
"""Generate the complete public-only formal report under a frozen identity contract."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = EXPERIMENT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from foundation_mri.finalization import (  # noqa: E402
    FinalizationInputs,
    FinalizationOutputs,
    finalize_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=EXPERIMENT_ROOT / "metrics/results_summary.json",
    )
    parser.add_argument(
        "--baseline-public",
        type=Path,
        default=EXPERIMENT_ROOT / "metrics/baseline_metrics.csv",
    )
    parser.add_argument(
        "--phenotype-public",
        type=Path,
        default=EXPERIMENT_ROOT / "metrics/phenotype_metrics.csv",
    )
    parser.add_argument(
        "--subtype-public",
        type=Path,
        default=EXPERIMENT_ROOT / "metrics/subtype_metrics.csv",
    )
    parser.add_argument(
        "--ftv-public",
        type=Path,
        default=EXPERIMENT_ROOT / "metrics/ftv_probe_metrics.csv",
    )
    parser.add_argument(
        "--paired-public",
        type=Path,
        default=EXPERIMENT_ROOT / "metrics/paired_bootstrap_comparisons.csv",
    )
    parser.add_argument(
        "--reporting-run-provenance",
        type=Path,
        default=EXPERIMENT_ROOT / "metrics/reporting_run_provenance.json",
    )
    parser.add_argument(
        "--results-summary-markdown",
        type=Path,
        default=EXPERIMENT_ROOT / "reports/results_summary.md",
    )
    parser.add_argument(
        "--timing-figure",
        type=Path,
        default=EXPERIMENT_ROOT / "figures/pcr_timing_performance.png",
    )
    parser.add_argument(
        "--calibration-figure",
        type=Path,
        default=EXPERIMENT_ROOT / "figures/calibration_clinical_complementarity.png",
    )
    parser.add_argument(
        "--model-execution-ledger",
        type=Path,
        default=EXPERIMENT_ROOT / "reports/model_execution_ledger.md",
    )
    parser.add_argument(
        "--foundation-model-selection",
        type=Path,
        default=EXPERIMENT_ROOT / "reports/foundation_model_selection.md",
    )
    parser.add_argument(
        "--current-cnn-provenance-audit",
        type=Path,
        default=EXPERIMENT_ROOT / "reports/current_cnn_provenance_audit.md",
    )
    parser.add_argument(
        "--contract-json",
        type=Path,
        default=EXPERIMENT_ROOT / "configs/final_report_contract.json",
    )
    parser.add_argument(
        "--template-markdown",
        type=Path,
        default=EXPERIMENT_ROOT / "reports/final_report.template.md",
    )
    parser.add_argument(
        "--finalization-lock",
        type=Path,
        default=EXPERIMENT_ROOT / "configs/FINALIZATION_LOCK.v1.json",
    )
    parser.add_argument(
        "--git-handoff-json",
        type=Path,
        default=EXPERIMENT_ROOT / "reports/git_handoff.json",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=EXPERIMENT_ROOT / "reports/final_report.md",
    )
    parser.add_argument(
        "--coverage-receipt",
        type=Path,
        default=EXPERIMENT_ROOT / "metrics/final_report_coverage.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = tuple(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    if raw_argv:
        raise ValueError("formal finalization lock requires exact empty argv")
    summary = finalize_report(
        FinalizationInputs(
            summary_json=args.summary_json,
            baseline_public=args.baseline_public,
            phenotype_public=args.phenotype_public,
            subtype_public=args.subtype_public,
            ftv_public=args.ftv_public,
            paired_public=args.paired_public,
            reporting_run_provenance=args.reporting_run_provenance,
            results_summary_markdown=args.results_summary_markdown,
            timing_figure=args.timing_figure,
            calibration_figure=args.calibration_figure,
            model_execution_ledger=args.model_execution_ledger,
            foundation_model_selection=args.foundation_model_selection,
            current_cnn_provenance_audit=args.current_cnn_provenance_audit,
            contract_json=args.contract_json,
            template_markdown=args.template_markdown,
            finalization_lock=args.finalization_lock,
            git_handoff_json=args.git_handoff_json,
        ),
        FinalizationOutputs(
            final_report=args.output_report,
            coverage_receipt=args.coverage_receipt,
        ),
    )
    print(
        "public-only final report complete: "
        + ", ".join(f"{key}={value}" for key, value in summary.items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

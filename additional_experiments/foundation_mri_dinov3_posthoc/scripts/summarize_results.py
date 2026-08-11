#!/usr/bin/env python3
"""Publish the locked DINOv3 post-hoc public report with an exact empty argv."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]
SOURCE_ROOT = EXPERIMENT_ROOT / "src"
BASE_EXPERIMENT_ROOT = REPOSITORY_ROOT / "additional_experiments/foundation_mri_baselines"
_SIBLING = "additional_experiments/foundation_mri_dinov3_posthoc"
EXCLUSIVE_OUTPUTS = {
    "baseline_predictions": f"{_SIBLING}/predictions/dinov3_baseline_predictions.private.csv",
    "baseline_selection": f"{_SIBLING}/metrics/dinov3_baseline_selection.private.csv",
    "baseline_metrics": f"{_SIBLING}/metrics/dinov3_baseline_metrics.csv",
    "baseline_progress": f"{_SIBLING}/logs/dinov3_baseline.progress.private.jsonl",
    "baseline_receipt": f"{_SIBLING}/metrics/baseline_run.private.provenance.json",
    "probe_phenotype_predictions": f"{_SIBLING}/predictions/dinov3_phenotype_predictions.private.csv",
    "probe_phenotype_selection": f"{_SIBLING}/metrics/dinov3_phenotype_selection.private.csv",
    "probe_phenotype_metrics": f"{_SIBLING}/metrics/dinov3_phenotype_metrics.csv",
    "probe_subtype_predictions": f"{_SIBLING}/predictions/dinov3_subtype_predictions.private.csv",
    "probe_subtype_selection": f"{_SIBLING}/metrics/dinov3_subtype_selection.private.csv",
    "probe_subtype_metrics": f"{_SIBLING}/metrics/dinov3_subtype_metrics.csv",
    "probe_ftv_predictions": f"{_SIBLING}/predictions/dinov3_ftv_predictions.private.csv",
    "probe_ftv_selection": f"{_SIBLING}/metrics/dinov3_ftv_selection.private.csv",
    "probe_ftv_metrics": f"{_SIBLING}/metrics/dinov3_ftv_metrics.csv",
    "probe_progress": f"{_SIBLING}/logs/dinov3_probe.progress.private.jsonl",
    "probe_receipt": f"{_SIBLING}/metrics/probe_run.private.provenance.json",
    "report_paired_comparisons": f"{_SIBLING}/metrics/paired_bootstrap_comparisons.csv",
    "report_results_summary": f"{_SIBLING}/metrics/results_summary.json",
    "report_final_report": f"{_SIBLING}/reports/final_report.md",
    "report_pcr_timing_figure": f"{_SIBLING}/figures/pcr_timing_performance.png",
    "report_paired_comparison_figure": f"{_SIBLING}/figures/paired_comparison_deltas.png",
    "report_reporting_marker": f"{_SIBLING}/metrics/reporting_run_provenance.json",
}


def _verify_exclusive_outputs(observed: object) -> None:
    if not isinstance(observed, dict) or observed != EXCLUSIVE_OUTPUTS:
        raise ValueError("evaluation-lock exclusive output contract drifted")


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = tuple(sys.argv[1:] if argv is None else argv)
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))

    # This is deliberately the first experiment-owned API call.  In
    # particular, no private/public outcome CSV is imported or parsed before
    # the report command, both producers, and parent comparator bytes pass
    # their frozen hash gates.
    from foundation_mri_dinov3.locking import (  # noqa: PLC0415
        load_json,
        verify_evaluation_lock,
        verify_producer_receipt,
    )

    report_lock = verify_evaluation_lock("report", raw_argv)
    _verify_exclusive_outputs(report_lock.get("exclusive_outputs"))
    producer_receipts = {
        "baseline": verify_producer_receipt("baseline"),
        "probe": verify_producer_receipt("probe"),
    }
    lock_document = load_json(report_lock["lock_path"])
    parent_comparator_artifacts = lock_document["parent_comparator_artifacts"]

    from foundation_mri_dinov3.reporting import (  # noqa: PLC0415
        ReportingInputs,
        ReportingOutputs,
        summarize_results,
    )

    inputs = ReportingInputs(
        new_baseline_private=EXPERIMENT_ROOT
        / "predictions/dinov3_baseline_predictions.private.csv",
        new_baseline_public=EXPERIMENT_ROOT / "metrics/dinov3_baseline_metrics.csv",
        new_phenotype_private=EXPERIMENT_ROOT
        / "predictions/dinov3_phenotype_predictions.private.csv",
        new_phenotype_public=EXPERIMENT_ROOT / "metrics/dinov3_phenotype_metrics.csv",
        new_subtype_private=EXPERIMENT_ROOT
        / "predictions/dinov3_subtype_predictions.private.csv",
        new_subtype_public=EXPERIMENT_ROOT / "metrics/dinov3_subtype_metrics.csv",
        new_ftv_private=EXPERIMENT_ROOT
        / "predictions/dinov3_ftv_predictions.private.csv",
        new_ftv_public=EXPERIMENT_ROOT / "metrics/dinov3_ftv_metrics.csv",
        old_baseline_private=BASE_EXPERIMENT_ROOT
        / "predictions/baseline_predictions.private.csv",
        old_baseline_public=BASE_EXPERIMENT_ROOT / "metrics/baseline_metrics.csv",
        old_phenotype_public=BASE_EXPERIMENT_ROOT / "metrics/phenotype_metrics.csv",
        old_subtype_public=BASE_EXPERIMENT_ROOT / "metrics/subtype_metrics.csv",
        old_ftv_public=BASE_EXPERIMENT_ROOT / "metrics/ftv_probe_metrics.csv",
    )
    outputs = ReportingOutputs(
        paired_csv=EXPERIMENT_ROOT / "metrics/paired_bootstrap_comparisons.csv",
        summary_json=EXPERIMENT_ROOT / "metrics/results_summary.json",
        final_report=EXPERIMENT_ROOT / "reports/final_report.md",
        timing_figure=EXPERIMENT_ROOT / "figures/pcr_timing_performance.png",
        comparison_figure=EXPERIMENT_ROOT / "figures/paired_comparison_deltas.png",
        reporting_marker=EXPERIMENT_ROOT / "metrics/reporting_run_provenance.json",
    )
    counts = summarize_results(
        inputs,
        outputs,
        template_path=EXPERIMENT_ROOT / "reports/final_report.template.md",
        report_lock=report_lock,
        producer_receipts=producer_receipts,
        parent_comparator_artifacts=parent_comparator_artifacts,
        strict_modes=True,
    )
    rendered = ",".join(f"{key}={counts[key]}" for key in sorted(counts))
    print(f"consumer=report;status=complete;{rendered}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

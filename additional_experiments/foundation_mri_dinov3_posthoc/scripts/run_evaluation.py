#!/usr/bin/env python3
"""Run the two locked DINOv3 post-hoc evaluation producers.

The formal argument vector is empty.  Invoke this file twice: the first
successful invocation publishes the baseline receipt and the second verifies
that receipt before publishing the probe receipt.  Existing or partial output
sets always fail closed.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Sequence


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]
SOURCE_ROOT = EXPERIMENT_ROOT / "src"

MODEL_NAME = "dinov3_vitb16_lvd1689m_posthoc"
DEFAULT_FEATURE = (
    EXPERIMENT_ROOT
    / "features"
    / "formal"
    / MODEL_NAME
    / "frozen_features.private.npz"
)
DEFAULT_CLINICAL = Path(
    "/data/data/Preprocessed/I-SPY2/clinical_labels_complete4visits.csv"
)
DEFAULT_FOLDS = Path(
    "/data/data/Preprocessed/I-SPY2/"
    "_matched_breastdcedl_t0_dicomrepair_rgb224_seed2026/"
    "matched_patient_cv_splits_seed2026.csv"
)
DEFAULT_RADIOMICS = (
    REPOSITORY_ROOT
    / "additional_experiments/radiomics_next_change/data_audit/"
    "radiomics_transition_targets_raw.csv"
)


def _receipt_path(consumer: str) -> Path:
    return EXPERIMENT_ROOT / f"metrics/{consumer}_run.private.provenance.json"


def _select_consumer(verify_producer_receipt: object) -> str:
    baseline = _receipt_path("baseline")
    probe = _receipt_path("probe")
    baseline_present = os.path.lexists(baseline)
    probe_present = os.path.lexists(probe)
    if probe_present and not baseline_present:
        raise RuntimeError("probe receipt exists without the required baseline receipt")
    if not baseline_present:
        return "baseline"
    # The historical producer is verified without parsing any metric artifact.
    verify_producer_receipt("baseline")  # type: ignore[operator]
    if not probe_present:
        return "probe"
    verify_producer_receipt("probe")  # type: ignore[operator]
    raise FileExistsError("both locked DINOv3 producer receipts already exist")


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = tuple(sys.argv[1:] if argv is None else argv)
    if raw_argv:
        raise ValueError("formal DINOv3 evaluation requires an exact empty argv")

    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))

    # This is intentionally the first experiment-owned API call.  Both exact
    # empty-argv commands are verified before output state is inspected or the
    # hash-pinned parent estimator/data modules are imported.
    from foundation_mri_dinov3.locking import (  # noqa: PLC0415
        verify_evaluation_lock,
        verify_producer_receipt,
    )

    verified = {
        consumer: verify_evaluation_lock(consumer, raw_argv)
        for consumer in ("baseline", "probe")
    }
    consumer = _select_consumer(verify_producer_receipt)

    from foundation_mri_dinov3.evaluation import (  # noqa: PLC0415
        run_baseline_evaluation,
        run_probe_evaluation,
    )

    runner = (
        run_baseline_evaluation if consumer == "baseline" else run_probe_evaluation
    )
    counts = runner(
        feature_path=DEFAULT_FEATURE,
        fold_manifest_path=DEFAULT_FOLDS,
        clinical_path=DEFAULT_CLINICAL,
        radiomics_path=DEFAULT_RADIOMICS,
        output_root=EXPERIMENT_ROOT,
        lock_receipt=verified[consumer],
        command_argv=raw_argv,
    )
    rendered_counts = ",".join(
        f"{key}={counts[key]}" for key in sorted(counts)
    )
    print(f"consumer={consumer};status=complete;{rendered_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

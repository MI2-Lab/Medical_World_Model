#!/usr/bin/env python3
"""校验五折候选 manifest 与 clean I-SPY2 cohort/label 的一致性。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


AUDIT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = AUDIT_ROOT.parent
CLEAN_ROOT = REPOSITORY_ROOT / "ispy_jepa_tmi_clean"
DEFAULT_MANIFEST = Path(
    "/data/data/Preprocessed/I-SPY2/"
    "_matched_breastdcedl_t0_dicomrepair_rgb224_seed2026/"
    "matched_patient_cv_splits_seed2026.csv"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--config",
        type=Path,
        default=CLEAN_ROOT / "configs" / "paper_v1.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=AUDIT_ROOT / "metrics" / "fold_manifest_validation.json",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(CLEAN_ROOT))
    sys.path.insert(0, str(REPOSITORY_ROOT))
    from corejepa.config import load_config  # pylint: disable=import-outside-toplevel
    from corejepa.training.runner import load_experiment_records  # pylint: disable=import-outside-toplevel
    from shortcut_audit.auditlib.folds import load_fold_manifest  # pylint: disable=import-outside-toplevel

    records, n_primary = load_experiment_records(load_config(args.config))
    primary = records[:n_primary]
    expected_labels = {record.patient_id: int(record.pcr) for record in primary}
    _, summary = load_fold_manifest(
        args.manifest,
        expected_patient_ids=list(expected_labels),
        expected_labels=expected_labels,
    )
    result = {
        "status": "valid_candidate_copy",
        "warning": (
            "该文件与 clean cohort/label 一致，但不在 clean 分支内，且原始 run 目录缺失；"
            "在 checkpoint provenance 核对前不能认定为 native fold。"
        ),
        "clean_primary_patients": n_primary,
        **summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

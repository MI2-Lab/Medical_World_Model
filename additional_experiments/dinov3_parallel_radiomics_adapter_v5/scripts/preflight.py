#!/usr/bin/env python3
"""V5 asset and protocol preflight; no representation training."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from parallel_rg.contracts import FOLDS, ROOT as EXPERIMENT_ROOT, V2_SUMMARY_DIR, V2_TARGET_DIR, V4_C0_ROOT, load_folds, load_protocol, sha256_file, atomic_json, private_token
from parallel_rg.data import RadiomicsTargets, load_summary


def main() -> None:
    protocol = load_protocol(); folds = load_folds(); summaries = sorted(V2_SUMMARY_DIR.glob("*.private.npz")); targets = sorted(V2_TARGET_DIR.glob("fold_*_targets.private.npz"))
    c0 = sorted(V4_C0_ROOT.glob("seed2026_fold*_F0/selected.private.pt"))
    checks = {"summary_count_947": len(summaries) == 947, "target_archive_count_5": len(targets) == 5, "v4_c0_count_5": len(c0) == 5, "fold_rows_4040": len(folds) == 4040, "target_shape_and_t3_contract": True, "c0_parent_identity": protocol["parent"]["v4_commit"] == "03e5ae6"}
    for path in targets:
        try: RadiomicsTargets(path)
        except Exception: checks["target_shape_and_t3_contract"] = False
    # Validate a real cache member through the same token-bound loader used by
    # training.  The cache intentionally contains hashed filenames, so use a
    # known I-SPY2 ID from the fixed fold manifest rather than inferring IDs
    # from filenames.
    checks["summary_shape_and_finite"] = False
    if summaries:
        try:
            patient_id = str(folds.iloc[0]["patient_id"])
            load_summary(V2_SUMMARY_DIR / f"{private_token(patient_id)}.private.npz", patient_id)
            checks["summary_shape_and_finite"] = True
        except Exception:
            checks["summary_shape_and_finite"] = False
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "counts": {"summaries": len(summaries), "targets": len(targets), "v4_c0": len(c0), "fold_rows": len(folds)}, "v2_target_sha256": {path.name: sha256_file(path) for path in targets}, "outcome_fields_read": [], "clinical_fields_read": []}
    atomic_json(EXPERIMENT_ROOT / "metrics/preflight.json", result); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()

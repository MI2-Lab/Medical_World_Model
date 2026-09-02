#!/usr/bin/env python3
"""Outcome-blind V4 inheritance and cache preflight."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factorized_rg.contracts import FOLDS, PILOT_ARMS, V2_D1_CHECKPOINT_ROOT, V2_SUMMARY_DIR, V2_TARGET_DIR, load_folds, load_protocol, load_train_only_ids, atomic_json
from factorized_rg.data import RadiomicsTargets, load_summary


def main() -> None:
    protocol = load_protocol(); folds = load_folds(); train_only = load_train_only_ids()
    summaries = sorted(V2_SUMMARY_DIR.glob("*.private.npz")); targets = sorted(V2_TARGET_DIR.glob("fold_*_targets.private.npz")); d1 = sorted(V2_D1_CHECKPOINT_ROOT.glob("seed2026_fold*_D1/selected.private.pt"))
    if len(summaries) != 947 or len(targets) != 5 or len(d1) != 5:
        raise SystemExit(f"V2 asset coverage failed: summaries={len(summaries)}, targets={len(targets)}, d1={len(d1)}")
    load_summary(summaries[0], "not-used") if False else None
    for path in targets: RadiomicsTargets(path)
    if len(folds) != 4040 or len(train_only) != 139: raise SystemExit("fold/train-only contract failed")
    if protocol["objective"]["pilot_radiomics_weights"] != {"F005": 0.05, "F010": 0.1, "F025": 0.25}: raise SystemExit("pilot weights drifted")
    result = {"status": "PASS", "summary_files": len(summaries), "target_files": len(targets), "v2_d1_checkpoints": len(d1), "fold_rows": len(folds), "train_only_patients": len(train_only), "pilot_arms": list(PILOT_ARMS), "clinical_fields_read": [], "outcome_fields_read": []}
    atomic_json(ROOT / "metrics/preflight.json", result); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()

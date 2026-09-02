#!/usr/bin/env python3
"""Outcome-blind frozen-state radiomics probe sanity check."""
from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from factorized_rg.contracts import V2_ROOT, V2_SUMMARY_DIR, atomic_json
from factorized_rg.data import RadiomicsTargets
from factorized_rg.contracts import load_folds


def rho(a: np.ndarray, b: np.ndarray) -> float:
    ar = np.argsort(np.argsort(a)); br = np.argsort(np.argsort(b))
    return float(np.corrcoef(ar, br)[0, 1]) if len(a) > 1 else float("nan")


def main() -> None:
    folds = load_folds(); rows = []
    target_ids = None
    for fold in range(5):
        state_path = V2_ROOT / f"features/private/states/seed2026_fold{fold}_D1_states.private.npz"
        with np.load(state_path, allow_pickle=False) as state_payload:
            ids = tuple(state_payload["patient_id"].astype(str).tolist()); full = np.asarray(state_payload["state"], dtype=np.float32)
        target = RadiomicsTargets(V2_ROOT / f"features/private/fold_targets/fold_{fold}_targets.private.npz")
        target_ids = set(target.patient_ids) if target_ids is None else target_ids.intersection(target.patient_ids)
        by_id = target.by_patient
        test_ids = set(folds.loc[(folds.fold == fold) & (folds.split == "test"), "patient_id"].astype(str))
        usable = [pid for pid in ids if pid in by_id]
        train = [pid for pid in usable if pid not in test_ids]; test = [pid for pid in usable if pid in test_ids]
        # V2 states are not factorized. This is only a pre-pilot target
        # learnability check using the complete frozen 192-D state; it is not
        # evidence for the V4 phenotype branch.
        x_train = np.asarray([full[ids.index(pid)].mean(0) for pid in train]); y_train = np.asarray([by_id[pid][0][0] for pid in train])
        x_test = np.asarray([full[ids.index(pid)].mean(0) for pid in test]); y_test = np.asarray([by_id[pid][0][0] for pid in test])
        scaler = StandardScaler().fit(x_train); model = Ridge(alpha=1.0).fit(scaler.transform(x_train), y_train)
        pred = model.predict(scaler.transform(x_test)); rows.append({"fold": fold, "n_test": len(test), "pc": 0, "spearman": rho(pred, y_test)})
    values = [r["spearman"] for r in rows if np.isfinite(r["spearman"])]
    out = ROOT / "metrics/head_only_metrics.csv"; out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    result = {"status": "PASS", "folds": len(rows), "v2_frozen_full_state_pc0_test_spearman_mean": float(np.mean(values)), "outcome_fields_read": [], "clinical_fields_read": [], "purpose": "target/state sanity only; not a pCR result"}
    atomic_json(ROOT / "metrics/head_only_gate.json", result); print(result)


if __name__ == "__main__": main()

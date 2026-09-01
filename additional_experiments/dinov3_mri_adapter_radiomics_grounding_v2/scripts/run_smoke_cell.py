#!/usr/bin/env python3
"""Run one non-formal paired D1-D3 cell without using metrics for tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dinov3_rg.contracts import TRAIN_ONLY_MANIFEST, atomic_json  # noqa: E402
from dinov3_rg.data import FoldTargets, load_train_only_ids, split_patient_ids  # noqa: E402
from dinov3_rg.security import RepresentationReadSentinel  # noqa: E402
from dinov3_rg.training import train_cell  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    RepresentationReadSentinel().install()
    for prerequisite in (
        ROOT / "target_feasibility.json",
        ROOT / "manifests/dinov3_cache_complete.json",
    ):
        if not prerequisite.is_file() or json.loads(prerequisite.read_text(encoding="utf-8")).get("status") not in {"PASS", "COMPLETE"}:
            raise SystemExit(f"smoke prerequisite did not pass: {prerequisite}")
    targets_path = ROOT / "features/private/fold_targets/fold_0_targets.private.npz"
    targets = FoldTargets.load(targets_path)
    if targets.radiomics_mask[:, 3].any():
        raise RuntimeError("T3 entered the smoke radiomics target")
    splits = split_patient_ids(0, load_train_only_ids(TRAIN_ONLY_MANIFEST))
    completions = []
    for arm in ("D1", "D2", "D3"):
        completions.append(
            train_cell(
                seed=2026,
                fold=0,
                arm=arm,
                train_ids=splits["train"],
                validation_ids=splits["val"],
                summary_dir=ROOT / "cache/dinov3_summaries",
                targets_path=targets_path,
                checkpoint_root=ROOT / "checkpoints/smoke",
                device=args.device,
                workers=args.workers,
                epochs=args.epochs,
            )
        )
    initializations = {value["initialization_sha256"] for value in completions}
    d3_history = json.loads(
        (ROOT / "checkpoints/smoke/seed2026_fold0_D3/history.private.json").read_text(encoding="utf-8")
    )["history"]
    gate = {
        "schema_version": 1,
        "status": "PASS",
        "cells": 3,
        "epochs": int(args.epochs),
        "paired_initialization": len(initializations) == 1,
        "finite_checkpoints": all(value["status"] == "COMPLETE" for value in completions),
        "d3_radiomics_head_gradient_positive": all(
            float(epoch["train"]["first_batch_radiomics_head_gradient_norm"]) > 0
            for epoch in d3_history
        ),
        "t3_radiomics_mask_false": True,
        "selection_metrics_not_used_for_tuning": True,
        "outcome_fields_read": [],
        "clinical_fields_read": [],
    }
    gate["status"] = "PASS" if all(
        gate[name]
        for name in (
            "paired_initialization",
            "finite_checkpoints",
            "d3_radiomics_head_gradient_positive",
            "t3_radiomics_mask_false",
        )
    ) else "FAIL"
    atomic_json(ROOT / "metrics/smoke_cell.json", gate)
    print(gate)
    if gate["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

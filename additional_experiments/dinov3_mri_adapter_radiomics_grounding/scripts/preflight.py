#!/usr/bin/env python3
"""Run code/data/model gates without opening outcomes or clinical data."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dinov3_rg.contracts import (  # noqa: E402
    CACHE_MANIFEST, FOLD_MANIFEST, FTV_TRANSITIONS, LOCKED_HASHES, SOURCE_INVENTORY,
    TECHNICAL_ELIGIBILITY,
    TRAIN_ONLY_MANIFEST, atomic_json, file_sha256, load_protocol, verify_locked_file,
)
from dinov3_rg.cache_io import load_c1b_manifest  # noqa: E402
from dinov3_rg.data import load_fold_frame, load_train_only_ids  # noqa: E402
from dinov3_rg.extraction import central_local_slices, summarize_tokens  # noqa: E402
from dinov3_rg.model import MRIAdapterWorldModel, initialization_sha256  # noqa: E402
from dinov3_rg.objective import masked_patient_smooth_l1  # noqa: E402
from dinov3_rg.security import scan_representation_sources  # noqa: E402


def main() -> None:
    protocol = load_protocol()
    if os.environ.get("CONDA_DEFAULT_ENV") != "bowen":
        raise SystemExit("preflight must run in Anaconda environment 'bowen'")
    locked = {
        "cache_manifest": verify_locked_file(CACHE_MANIFEST, LOCKED_HASHES["cache_manifest"], "cache manifest"),
        "fold_manifest": verify_locked_file(FOLD_MANIFEST, LOCKED_HASHES["fold_manifest"], "fold manifest"),
        "ftv_transitions": verify_locked_file(FTV_TRANSITIONS, LOCKED_HASHES["ftv_transitions"], "FTV transitions"),
        "source_inventory": verify_locked_file(SOURCE_INVENTORY, LOCKED_HASHES["source_inventory"], "source inventory"),
        "train_only_manifest": verify_locked_file(TRAIN_ONLY_MANIFEST, LOCKED_HASHES["train_only_manifest"], "train-only manifest"),
        "technical_eligibility": verify_locked_file(TECHNICAL_ELIGIBILITY, LOCKED_HASHES["technical_eligibility"], "technical eligibility"),
    }
    cache = load_c1b_manifest()
    folds = load_fold_frame()
    train_only = load_train_only_ids(TRAIN_ONLY_MANIFEST)
    first = cache[sorted(cache)[0]]
    with np.load(first.path, allow_pickle=False) as payload:
        image = np.asarray(payload["image"])
        local = central_local_slices(image)
        channel_names = tuple(payload["channel_names"].astype(str))
    mock_hidden = torch.randn(2, 201, 768)
    mock_summary = summarize_tokens(mock_hidden)
    # Paired construction must produce exact common initial states.
    hashes = []
    for _ in range(3):
        torch.manual_seed(2026)
        hashes.append(initialization_sha256(MRIAdapterWorldModel()))
    model = MRIAdapterWorldModel(dropout=0.0)
    synthetic = torch.randn(2, 4, 7, 32, 2304)
    output = model(synthetic)
    target = torch.randn(2, 4, 16)
    mask = torch.ones(2, 4, dtype=torch.bool)
    rad_loss, _, _ = masked_patient_smooth_l1(output.radiomics_prediction, target, mask)
    rad_loss.backward()
    adapter_gradient = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.adapter.parameters()
        if parameter.grad is not None
    )
    head_gradient = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.radiomics_head.parameters()
        if parameter.grad is not None
    )
    try:
        model(synthetic, torch.ones(1))  # type: ignore[call-arg]
    except TypeError:
        mask_rejected = True
    else:
        mask_rejected = False
    representation_paths = [
        ROOT / "src/dinov3_rg/data.py",
        ROOT / "src/dinov3_rg/cache_io.py",
        ROOT / "src/dinov3_rg/extraction.py",
        ROOT / "src/dinov3_rg/model.py",
        ROOT / "src/dinov3_rg/objective.py",
        ROOT / "src/dinov3_rg/radiomics.py",
        ROOT / "src/dinov3_rg/targets.py",
        ROOT / "src/dinov3_rg/training.py",
        ROOT / "scripts/extract_dinov3.py",
        ROOT / "scripts/build_radiomics_rois.py",
        ROOT / "scripts/extract_radiomics.py",
        ROOT / "scripts/build_fold_targets.py",
        ROOT / "scripts/train_cell.py",
        ROOT / "scripts/run_matrix.py",
    ]
    source_scan = scan_representation_sources(representation_paths)
    gates = {
        "bowen_environment": True,
        "locked_inputs": len(locked) == 6,
        "cohort_947": len(cache) == 947,
        "folds_808x5": len(folds) == 4040,
        "train_only_139": len(train_only) == 139,
        "c1b_shape": image.shape == (4, 7, 112, 176, 160),
        "local_shape": local.shape == (4, 7, 32, 72, 72),
        "channel_order_count": len(channel_names) == 7,
        "token_summary_2304": tuple(mock_summary.shape) == (2, 2304),
        "paired_initialization": len(set(hashes)) == 1,
        "effective_batch_32": int(protocol["training"]["physical_batch"]) * int(protocol["training"]["accumulation_steps"]) == 32,
        "model_state_shape": tuple(output.response_state.shape) == (2, 4, 192),
        "radiomics_gradient_adapter": adapter_gradient > 0,
        "radiomics_gradient_head": head_gradient > 0,
        "mask_not_forward_input": mask_rejected,
        "representation_source_scan": source_scan["status"] == "PASS",
    }
    payload = {
        "schema_version": 1,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "locked_hashes": locked,
        "protocol_sha256": file_sha256(ROOT / "configs/protocol.json"),
        "source_scan": source_scan,
        "model_architecture": model.architecture_contract(),
        "outcome_fields_read": [],
        "clinical_fields_read": [],
    }
    atomic_json(ROOT / "metrics/preflight.json", payload)
    print(payload)
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

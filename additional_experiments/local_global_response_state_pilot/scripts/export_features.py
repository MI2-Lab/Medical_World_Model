#!/usr/bin/env python3
"""Export one selected pilot checkpoint's online pre-projector response state."""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
import os
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
SRC_ROOT = EXPERIMENT_ROOT / "src"
SCRIPTS_ROOT = EXPERIMENT_ROOT / "scripts"
SEALED_ROOT = REPO_ROOT / "additional_experiments" / "c1b_overlap_eligibility_ftv_stageb"
SEALED_SRC = SEALED_ROOT / "src"
DEFAULT_SENTINEL = SEALED_ROOT / "STAGE_A_GO.json"
DEFAULT_SENTINEL_SHA256 = "0b2c9e0af63ce8806525fb15ac9a27f6ab525b0259ccf16001981ca5091afbdb"
DEFAULT_DATA_CONTRACT = SEALED_ROOT / "manifests" / "stage_b_data_contract.private.json"
DEFAULT_DATA_CONTRACT_SHA256 = "dd22f130043863d4fce8956061fca389894a31874567ed7929e139f32ff5ab27"
PILOT_FEATURE_ROOT = EXPERIMENT_ROOT / "features"
for source in (SRC_ROOT, SCRIPTS_ROOT):
    value = str(source.resolve())
    while value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)

from freeze_preregistration import verify as verify_preregistration  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-a-sentinel", type=Path, default=DEFAULT_SENTINEL)
    parser.add_argument("--stage-a-sentinel-sha256", default=DEFAULT_SENTINEL_SHA256)
    parser.add_argument("--data-contract", type=Path, default=DEFAULT_DATA_CONTRACT)
    parser.add_argument("--data-contract-sha256", default=DEFAULT_DATA_CONTRACT_SHA256)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--preregistration-lock-sha256", required=True)
    parser.add_argument(
        "--arm",
        choices=("GAP0", "GAP3", "LOCAL0", "LOCAL3", "LG0", "LG3"),
        required=True,
    )
    parser.add_argument("--seed-base", type=int, choices=(2026, 3026), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    os.umask(0o077)
    preregistration = verify_preregistration()
    import lg_response_pilot.security as pilot_security

    if Path(str(getattr(pilot_security, "__file__", ""))).resolve() != (
        SRC_ROOT / "lg_response_pilot" / "security.py"
    ).resolve():
        raise ImportError("pilot security module was shadowed before feature export")
    from lg_response_pilot.security import (
        require_canonical_file,
        require_lock_sha256,
        resolve_contained_path,
    )

    args = parse_args()
    require_lock_sha256(
        preregistration["lock_sha256"], args.preregistration_lock_sha256
    )
    locked_upstream = preregistration["upstream_sha256"]
    args.stage_a_sentinel = require_canonical_file(
        args.stage_a_sentinel,
        DEFAULT_SENTINEL,
        args.stage_a_sentinel_sha256,
        locked_upstream[str(DEFAULT_SENTINEL.relative_to(REPO_ROOT))],
        label="Stage-A sentinel",
    )
    args.data_contract = require_canonical_file(
        args.data_contract,
        DEFAULT_DATA_CONTRACT,
        args.data_contract_sha256,
        locked_upstream[str(DEFAULT_DATA_CONTRACT.relative_to(REPO_ROOT))],
        label="Stage-B data contract",
    )
    feature_root = resolve_contained_path(
        args.feature_root,
        PILOT_FEATURE_ROOT,
        label="formal feature root",
    )
    output = resolve_contained_path(
        args.output,
        feature_root,
        label="per-cell feature output",
    )

    sealed_value = str(SEALED_SRC.resolve())
    while sealed_value in sys.path:
        sys.path.remove(sealed_value)
    sys.path.insert(0, sealed_value)

    from c1b_stage_b.gate import require_stage_a_go
    from c1b_stage_b.inputs import StageBDataPaths, load_stage_b_data
    from lg_response_pilot.features import export_response_features, file_sha256
    import torch

    import c1b_stage_b.gate as sealed_gate_module
    import c1b_stage_b.inputs as sealed_inputs_module
    import lg_response_pilot.features as pilot_features_module

    for module, root, label in (
        (sealed_gate_module, SEALED_SRC, "sealed Stage-B gate"),
        (sealed_inputs_module, SEALED_SRC, "sealed Stage-B inputs"),
        (pilot_features_module, SRC_ROOT / "lg_response_pilot", "pilot features"),
    ):
        pilot_security.require_module_within(module, root, label=label)

    authorization = require_stage_a_go(args.stage_a_sentinel)
    paths = StageBDataPaths.load(args.data_contract, args.data_contract_sha256)
    data = load_stage_b_data(paths, authorization, verify_cache_files=False)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    metadata = export_response_features(
        checkpoint_path=args.checkpoint,
        arm=args.arm,
        seed_base=args.seed_base,
        fold=args.fold,
        data=data,
        authorization=authorization,
        output_path=output,
        device=device,
        preregistration_lock_sha256=preregistration["lock_sha256"],
        batch_size=args.batch_size,
        workers=args.workers,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

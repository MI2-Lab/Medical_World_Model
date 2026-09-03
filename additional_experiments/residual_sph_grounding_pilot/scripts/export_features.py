#!/usr/bin/env python3
"""Export one selected S1/S2 online response-state asset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


os.umask(0o077)
sys.dont_write_bytecode = True
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
SRC = EXPERIMENT_ROOT / "src"
sys.path.insert(0, str(SRC))

from residual_sph.contracts import file_sha256  # noqa: E402
from residual_sph.features import export_response_features  # noqa: E402
from residual_sph.preregistration import require_lock_sha256, verify_preregistration  # noqa: E402
from residual_sph.upstream import verify_local_sources  # noqa: E402


SEALED_ROOT = REPO_ROOT / "additional_experiments" / "c1b_overlap_eligibility_ftv_stageb"
SEALED_SRC = SEALED_ROOT / "src"
SENTINEL = SEALED_ROOT / "STAGE_A_GO.json"
SENTINEL_SHA256 = "0b2c9e0af63ce8806525fb15ac9a27f6ab525b0259ccf16001981ca5091afbdb"
DATA_CONTRACT = SEALED_ROOT / "manifests" / "stage_b_data_contract.private.json"
DATA_CONTRACT_SHA256 = "dd22f130043863d4fce8956061fca389894a31874567ed7929e139f32ff5ab27"
FEATURE_ROOT = EXPERIMENT_ROOT / "features"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("S1", "S2", "S2_L10"), required=True)
    parser.add_argument("--seed-base", choices=(2026, 3026), type=int, required=True)
    parser.add_argument("--fold", choices=range(5), type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--preregistration-lock-sha256", required=True)
    args = parser.parse_args()
    preregistration = verify_preregistration(EXPERIMENT_ROOT)
    require_lock_sha256(preregistration["lock_sha256"], args.preregistration_lock_sha256)
    verify_local_sources()
    for path, digest, label in (
        (SENTINEL, SENTINEL_SHA256, "Stage-A sentinel"),
        (DATA_CONTRACT, DATA_CONTRACT_SHA256, "Stage-B data contract"),
    ):
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"{label} is missing or hash-mismatched")
    output = args.output.resolve()
    try:
        output.relative_to(FEATURE_ROOT.resolve())
    except ValueError as error:
        raise ValueError("feature output must be inside the experiment feature root") from error
    expected_tail = (
        f"seed_{args.seed_base}", args.arm, f"fold_{args.fold}", "response_state.private.npz"
    )
    if tuple(output.relative_to(FEATURE_ROOT.resolve()).parts[-4:]) != expected_tail:
        raise ValueError("feature output identity/path mismatch")
    sealed_value = str(SEALED_SRC.resolve())
    while sealed_value in sys.path:
        sys.path.remove(sealed_value)
    sys.path.insert(0, sealed_value)
    from c1b_stage_b.gate import require_stage_a_go
    from c1b_stage_b.inputs import StageBDataPaths, load_stage_b_data
    import torch

    authorization = require_stage_a_go(SENTINEL)
    paths = StageBDataPaths.load(DATA_CONTRACT, DATA_CONTRACT_SHA256)
    data = load_stage_b_data(paths, authorization, verify_cache_files=False)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    metadata = export_response_features(
        checkpoint_path=args.checkpoint,
        experimental_arm=args.arm,
        seed_base=args.seed_base,
        fold=args.fold,
        data=data,
        output_path=output,
        device=device,
        preregistration_lock_sha256=preregistration["lock_sha256"],
        batch_size=args.batch_size,
        workers=args.workers,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export one trigger-authorized frozen S3 checkpoint cell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c1b_spatial_audit.contracts import ARMS, checkpoint_path  # noqa: E402
from c1b_spatial_audit.runtime import load_stage_b_bundle  # noqa: E402
from c1b_spatial_audit.s3_exporter import export_s3_feature_cell  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed-base", type=int, choices=(2026, 3026), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument(
        "--trigger-gate",
        type=Path,
        default=ROOT / "metrics" / "s3_trigger_authorization.json",
    )
    parser.add_argument(
        "--sidecar",
        type=Path,
        default=ROOT / "manifests" / "audit_sidecars_s3.private.npz",
    )
    parser.add_argument("--feature-root", type=Path, default=ROOT / "features")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    import torch

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal S3 feature export requires an available CUDA device")
    expected_checkpoint = checkpoint_path(args.seed_base, args.arm, args.fold).resolve()
    if args.checkpoint.resolve() != expected_checkpoint:
        raise ValueError("--checkpoint must be the exact formal selected.pt")
    authorization, _, data = load_stage_b_bundle(verify_cache_files=False)
    metadata = export_s3_feature_cell(
        checkpoint_path=expected_checkpoint,
        arm=args.arm,
        seed_base=args.seed_base,
        fold=args.fold,
        data=data,
        authorization=authorization,
        trigger_gate_path=args.trigger_gate,
        sidecar_path=args.sidecar,
        feature_root=args.feature_root,
        device=device,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                pooling: {
                    "feature_path": payload["feature_path"],
                    "feature_sha256": payload["feature_sha256"],
                    "state_shape": payload["state_shape"],
                    "state_valid_count": payload["state_valid_count"],
                }
                for pooling, payload in metadata.items()
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

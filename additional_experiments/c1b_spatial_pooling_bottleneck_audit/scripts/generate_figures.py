#!/usr/bin/env python3
"""Build the private activation aggregate or render the 12 public audit PNGs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
FORMAL_STAGE_B_SRC = (
    REPO_ROOT
    / "additional_experiments"
    / "c1b_overlap_eligibility_ftv_stageb"
    / "src"
)
MODEL_READY_SRC = (
    REPO_ROOT / "additional_experiments" / "c1b_model_ready_ftv_sanity" / "src"
)
sys.path[:0] = [str(FORMAL_STAGE_B_SRC), str(MODEL_READY_SRC), str(ROOT / "src")]

from c1b_spatial_audit.contracts import file_sha256  # noqa: E402
from c1b_spatial_audit.figures import (  # noqa: E402
    FigureInputPaths,
    build_formal_activation_aggregate,
    render_public_figures,
)
from c1b_spatial_audit.runtime import verify_preregistration  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--build-activation-aggregate",
        action="store_true",
        help=(
            "Use only the fixed seed-2026/N1/fold-0 encoder to create the "
            "owner-only, identifier-free aggregate activation NPZ."
        ),
    )
    mode.add_argument(
        "--render-public",
        action="store_true",
        help="Render exactly the 12 registered aggregate public PNGs.",
    )
    parser.add_argument(
        "--activation-aggregate",
        type=Path,
        default=ROOT / "manifests/activation_aggregate.private.npz",
    )
    parser.add_argument(
        "--sidecar",
        type=Path,
        default=ROOT / "manifests/audit_sidecars.private.npz",
    )
    for number, filename in (
        (1, "table1_feature_map_contract.csv"),
        (2, "table2_static_ftv.csv"),
        (3, "table3_delta_ftv.csv"),
        (4, "table4_legacy_deficit_recovery.csv"),
        (5, "table5_nuisance_decodability.csv"),
        (6, "table6_occupancy_downsampling.csv"),
        (7, "table7_training_budget.csv"),
    ):
        parser.add_argument(
            f"--table{number}",
            type=Path,
            default=ROOT / "metrics" / filename,
        )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_preregistration()
    if args.build_activation_aggregate:
        output = build_formal_activation_aggregate(
            args.activation_aggregate,
            device=args.device,
            batch_size=args.batch_size,
            workers=args.workers,
        )
        print(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "mode": "PRIVATE_ACTIVATION_AGGREGATE",
                    "training_performed": False,
                    "outcomes_used": False,
                    "output": output.name,
                    "sha256": file_sha256(output),
                },
                sort_keys=True,
            )
        )
        return

    paths = FigureInputPaths(
        table1=args.table1,
        table2=args.table2,
        table3=args.table3,
        table4=args.table4,
        table5=args.table5,
        table6=args.table6,
        table7=args.table7,
        sidecar=args.sidecar,
        activation_aggregate=args.activation_aggregate,
    )
    outputs = render_public_figures(paths, output_dir=args.output_dir)
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "mode": "PUBLIC_FIGURES",
                "figure_count": len(outputs),
                "output_sha256": {
                    path.name: file_sha256(path) for path in outputs.values()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build hash-checked, owner-only geometry sidecars for the frozen audit."""

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
# Preserve the runtime import priority: formal Stage B precedes the older
# model-ready package.  This script imports c1b_sanity explicitly, but keeping
# the same order prevents future transitive imports from resolving schema-v1.
sys.path[:0] = [str(FORMAL_STAGE_B_SRC), str(MODEL_READY_SRC), str(ROOT / "src")]

from c1b_spatial_audit.contracts import MODEL_READY_ROOT, UPSTREAM_ROOT, file_sha256  # noqa: E402
from c1b_spatial_audit.sidecars import (  # noqa: E402
    build_audit_sidecars,
    require_outputs_absent,
    write_private_sidecars,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preregistration-lock",
        type=Path,
        default=ROOT / "PREREGISTRATION_LOCK.json",
    )
    parser.add_argument(
        "--stage-a-go", type=Path, default=UPSTREAM_ROOT / "STAGE_A_GO.json"
    )
    parser.add_argument(
        "--data-contract",
        type=Path,
        default=UPSTREAM_ROOT / "manifests/stage_b_data_contract.private.json",
    )
    parser.add_argument(
        "--cache-manifest",
        type=Path,
        default=UPSTREAM_ROOT / "manifests/stage_b_c1b_cache.private.csv",
    )
    parser.add_argument(
        "--geometry-inventory",
        type=Path,
        default=UPSTREAM_ROOT
        / "manifests/eligible_model_input_inventory.private.csv",
    )
    parser.add_argument(
        "--support-inventory",
        type=Path,
        default=MODEL_READY_ROOT / "manifests/model_input_inventory.private.csv",
    )
    parser.add_argument(
        "--support-reference",
        type=Path,
        default=MODEL_READY_ROOT
        / "metrics/support_containment_patient_visit.private.csv",
    )
    parser.add_argument(
        "--sidecar-output",
        type=Path,
        default=ROOT / "manifests/audit_sidecars.private.npz",
    )
    parser.add_argument(
        "--nuisance-output",
        type=Path,
        default=ROOT / "manifests/nuisance_targets.private.csv",
    )
    parser.add_argument(
        "--occupancy-output",
        type=Path,
        default=ROOT / "manifests/lesion_occupancy.private.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Fail before reading hundreds of GB if a prior formal result is present.
    require_outputs_absent(
        args.sidecar_output, args.nuisance_output, args.occupancy_output
    )

    def progress(done: int, total: int) -> None:
        if done == 1 or done % 25 == 0 or done == total:
            print(
                json.dumps(
                    {"sidecar_patients_complete": done, "sidecar_patients_total": total}
                ),
                file=sys.stderr,
                flush=True,
            )

    bundle = build_audit_sidecars(
        preregistration_lock=args.preregistration_lock,
        stage_a_go=args.stage_a_go,
        data_contract=args.data_contract,
        cache_manifest=args.cache_manifest,
        geometry_inventory=args.geometry_inventory,
        support_inventory=args.support_inventory,
        support_reference=args.support_reference,
        # Formal execution always verifies every selected C1B archive hash.
        verify_cache_archive_sha256=True,
        progress=progress,
    )
    outputs = write_private_sidecars(
        bundle,
        sidecar_output=args.sidecar_output,
        nuisance_output=args.nuisance_output,
        occupancy_output=args.occupancy_output,
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "patient_count": int(len(bundle.patient_id)),
                "visit_count": int(len(bundle.nuisance)),
                "formal_support_visit_count": int(bundle.c1b_oracle_valid.sum()),
                "legacy_pvalid": "NA_no_source_authoritative_mask",
                "legacy_poracle": (
                    "NA_incomplete_source_authoritative_support_1488_of_1500"
                ),
                "private_outputs_sha256": {
                    path.name: file_sha256(path) for path in outputs
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

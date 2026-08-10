#!/usr/bin/env python3
"""Build trigger-gated, authoritative S3 pooling sidecars (no model execution)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
FORMAL_STAGE_B_SRC = (
    REPO_ROOT / "additional_experiments" / "c1b_overlap_eligibility_ftv_stageb" / "src"
)
MODEL_READY_SRC = (
    REPO_ROOT / "additional_experiments" / "c1b_model_ready_ftv_sanity" / "src"
)
sys.path[:0] = [str(FORMAL_STAGE_B_SRC), str(MODEL_READY_SRC), str(ROOT / "src")]

from c1b_spatial_audit.contracts import (  # noqa: E402
    MODEL_READY_ROOT,
    UPSTREAM_ROOT,
    file_sha256,
)
from c1b_spatial_audit.s3_sidecars import (  # noqa: E402
    build_s3_audit_sidecars,
    write_s3_sidecars,
)
from c1b_spatial_audit.s3_trigger import require_s3_trigger_authorization  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trigger-gate",
        type=Path,
        default=ROOT / "metrics" / "s3_trigger_authorization.json",
    )
    parser.add_argument(
        "--preregistration-lock", type=Path, default=ROOT / "PREREGISTRATION_LOCK.json"
    )
    parser.add_argument("--stage-a-go", type=Path, default=UPSTREAM_ROOT / "STAGE_A_GO.json")
    parser.add_argument(
        "--data-contract",
        type=Path,
        default=UPSTREAM_ROOT / "manifests" / "stage_b_data_contract.private.json",
    )
    parser.add_argument(
        "--cache-manifest",
        type=Path,
        default=UPSTREAM_ROOT / "manifests" / "stage_b_c1b_cache.private.csv",
    )
    parser.add_argument(
        "--geometry-inventory",
        type=Path,
        default=UPSTREAM_ROOT / "manifests" / "eligible_model_input_inventory.private.csv",
    )
    parser.add_argument(
        "--support-inventory",
        type=Path,
        default=MODEL_READY_ROOT / "manifests" / "model_input_inventory.private.csv",
    )
    parser.add_argument(
        "--support-reference",
        type=Path,
        default=MODEL_READY_ROOT
        / "metrics"
        / "support_containment_patient_visit.private.csv",
    )
    parser.add_argument(
        "--sidecar-output",
        type=Path,
        default=ROOT / "manifests" / "audit_sidecars_s3.private.npz",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=ROOT / "manifests" / "audit_sidecars_s3.private.metadata.json",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Without this flag, validate only the explicit S3 trigger and write nothing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    gate = require_s3_trigger_authorization(args.trigger_gate, verify_live=True)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "TRIGGER_VALIDATED_NOT_EXECUTED",
                    "trigger_status": gate["status"],
                    "s3_execution_authorized": True,
                },
                sort_keys=True,
            )
        )
        return
    expected_sidecar = (ROOT / "manifests" / "audit_sidecars_s3.private.npz").resolve()
    expected_metadata = (
        ROOT / "manifests" / "audit_sidecars_s3.private.metadata.json"
    ).resolve()
    if args.sidecar_output.resolve() != expected_sidecar or args.metadata_output.resolve() != expected_metadata:
        raise ValueError("formal S3 sidecars must use the canonical manifests paths")
    if expected_sidecar.exists() or expected_metadata.exists():
        raise FileExistsError("refusing to overwrite conditional S3 sidecars")

    def progress(done: int, total: int) -> None:
        if done == 1 or done % 25 == 0 or done == total:
            print(
                json.dumps({"s3_support_patients_complete": done, "total": total}),
                file=sys.stderr,
                flush=True,
            )

    bundle = build_s3_audit_sidecars(
        trigger_gate=args.trigger_gate,
        preregistration_lock=args.preregistration_lock,
        stage_a_go=args.stage_a_go,
        data_contract=args.data_contract,
        cache_manifest=args.cache_manifest,
        geometry_inventory=args.geometry_inventory,
        support_inventory=args.support_inventory,
        support_reference=args.support_reference,
        verify_cache_archive_sha256=True,
        progress=progress,
    )
    source_paths = {
        "preregistration_lock": args.preregistration_lock,
        "stage_a_go": args.stage_a_go,
        "data_contract": args.data_contract,
        "cache_manifest": args.cache_manifest,
        "geometry_inventory": args.geometry_inventory,
        "support_inventory": args.support_inventory,
        "support_reference": args.support_reference,
    }
    outputs = write_s3_sidecars(
        bundle,
        sidecar_output=expected_sidecar,
        metadata_output=expected_metadata,
        trigger_gate=args.trigger_gate,
        source_paths=source_paths,
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "patient_count": len(bundle.patient_id),
                "formal_oracle_visit_count": int(bundle.c1b_oracle_valid.sum()),
                "legacy_poracle": "NA_incomplete_source_authoritative_support_1488_of_1500",
                "private_outputs_sha256": {
                    path.name: file_sha256(path) for path in outputs
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Hash every completed representation artifact before the first pCR read."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
SRC = EXPERIMENT_ROOT / "src"
sys.path.insert(0, str(SRC))

from residual_sph.contracts import file_sha256  # noqa: E402
from residual_sph.evaluation_lock import (  # noqa: E402
    expected_representation_artifact_groups,
    verify_pcr_firewall_audit,
)
from residual_sph.preregistration import require_lock_sha256, verify_preregistration  # noqa: E402
from residual_sph.provenance import load_s0_manifest, validate_s0_cell  # noqa: E402


OUTPUT = EXPERIMENT_ROOT / "manifests" / "representation_freeze.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration-lock-sha256", required=True)
    args = parser.parse_args()
    preregistration = verify_preregistration(EXPERIMENT_ROOT)
    require_lock_sha256(preregistration["lock_sha256"], args.preregistration_lock_sha256)
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    firewall = verify_pcr_firewall_audit(
        EXPERIMENT_ROOT,
        expected_preregistration_sha256=preregistration["lock_sha256"],
        expected_implementation_sha256=preregistration["implementation_lock_sha256"],
    )

    s0_manifest_path = EXPERIMENT_ROOT / "manifests/s0_confirmation_provenance.json"
    s0_manifest = load_s0_manifest(s0_manifest_path)
    confirmation = REPO_ROOT / "additional_experiments/local_response_state_multiseed_confirmation"
    selected_count = feature_count = probe_count = 0
    for arm in ("S0", "S1", "S2", "S2_L10"):
        for seed in (2026, 3026):
            for fold in range(5):
                if arm == "S0":
                    run = (
                        confirmation / "checkpoints/formal_4x8" / f"seed_{seed}"
                        / "LOCAL3" / f"fold_{fold}"
                    )
                    feature = (
                        confirmation / "features/formal_4x8" / f"seed_{seed}" / "LOCAL3"
                        / f"fold_{fold}" / "response_state.private.npz"
                    )
                    validate_s0_cell(
                        s0_manifest,
                        seed_base=seed,
                        fold=fold,
                        selection_path=run / "selection.json",
                        checkpoint_path=run / "selected.pt",
                        feature_path=feature,
                        feature_metadata_path=feature.with_suffix(".metadata.json"),
                    )
                selected_count += 1
                feature_count += 1
                probe_count += 1
    if (selected_count, feature_count, probe_count) != (40, 40, 40):
        raise AssertionError("formal freeze matrix must contain 40 complete cells")

    groups = expected_representation_artifact_groups(EXPERIMENT_ROOT)
    inventory: dict[str, str] = {}
    for group, relative_paths in groups.items():
        for relative in relative_paths:
            path = REPO_ROOT / relative
            if not path.is_file():
                raise FileNotFoundError(
                    f"missing representation-freeze artifact in {group}: {relative}"
                )
            inventory[relative] = file_sha256(path)

    payload = {
        "schema_version": 2,
        "experiment": "residual_sph_grounding_pilot",
        "status": "REPRESENTATION_FROZEN_PCR_EVALUATION_AUTHORIZED",
        "preregistration_lock_sha256": preregistration["lock_sha256"],
        "implementation_lock_sha256": preregistration["implementation_lock_sha256"],
        "pcr_or_clinical_read_before_freeze": False,
        "selected_checkpoint_count": selected_count,
        "selection_record_count": len(groups["selection_records"]),
        "ftv_transform_count": len(groups["ftv_transforms"]),
        "feature_asset_count": feature_count,
        "feature_metadata_count": len(groups["feature_metadata"]),
        "probe_cell_count": probe_count,
        "probe_artifact_count": len(groups["probe_outputs"]),
        "residualizer_fold_count": len(groups["residualizer_transforms"]),
        "representation_aggregate_count": len(groups["representation_aggregates"]),
        "probe_specification_file_count": len(groups["probe_specification"]),
        "pcr_firewall_audit_status": firewall["status"],
        "artifact_groups": {group: list(paths) for group, paths in groups.items()},
        "artifact_sha256": dict(sorted(inventory.items())),
    }
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run sealed static-FTV and literal observed-delta probes for one pilot cell."""

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
PILOT_PREDICTION_ROOT = EXPERIMENT_ROOT / "predictions"
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
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--preregistration-lock-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    os.umask(0o077)
    preregistration = verify_preregistration()
    import lg_response_pilot.security as pilot_security

    if Path(str(getattr(pilot_security, "__file__", ""))).resolve() != (
        SRC_ROOT / "lg_response_pilot" / "security.py"
    ).resolve():
        raise ImportError("pilot security module was shadowed before probe execution")
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
    feature_path = resolve_contained_path(
        args.features,
        feature_root,
        label="per-cell feature input",
    )
    probe_root = resolve_contained_path(
        args.probe_root,
        PILOT_PREDICTION_ROOT,
        label="formal probe root",
    )
    output_dir = resolve_contained_path(
        args.output_dir,
        probe_root,
        label="per-cell probe output",
    )

    sealed_value = str(SEALED_SRC.resolve())
    while sealed_value in sys.path:
        sys.path.remove(sealed_value)
    sys.path.insert(0, sealed_value)

    from c1b_stage_b.gate import require_stage_a_go
    from c1b_stage_b.inputs import StageBDataPaths, load_stage_b_data
    from lg_response_pilot.features import file_sha256
    from lg_response_pilot.probes import run_ftv_probes

    import c1b_stage_b.gate as sealed_gate_module
    import c1b_stage_b.inputs as sealed_inputs_module
    import lg_response_pilot.features as pilot_features_module
    import lg_response_pilot.probes as pilot_probes_module

    for module, root, label in (
        (sealed_gate_module, SEALED_SRC, "sealed Stage-B gate"),
        (sealed_inputs_module, SEALED_SRC, "sealed Stage-B inputs"),
        (pilot_features_module, SRC_ROOT / "lg_response_pilot", "pilot features"),
        (pilot_probes_module, SRC_ROOT / "lg_response_pilot", "pilot probes"),
    ):
        pilot_security.require_module_within(module, root, label=label)

    authorization = require_stage_a_go(args.stage_a_sentinel)
    paths = StageBDataPaths.load(args.data_contract, args.data_contract_sha256)
    data = load_stage_b_data(paths, authorization, verify_cache_files=False)
    metadata = run_ftv_probes(
        feature_path=feature_path,
        records=data.ftv,
        folds=data.folds,
        authorization=authorization,
        data_provenance=data.provenance,
        output_dir=output_dir,
        preregistration_lock_sha256=preregistration["lock_sha256"],
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

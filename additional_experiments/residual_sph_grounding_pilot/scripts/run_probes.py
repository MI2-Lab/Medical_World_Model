#!/usr/bin/env python3
"""Run all frozen representation probes for one arm/seed/fold cell."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import pandas as pd


os.umask(0o077)
sys.dont_write_bytecode = True
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
SRC = EXPERIMENT_ROOT / "src"
sys.path.insert(0, str(SRC))

from residual_sph.contracts import file_sha256  # noqa: E402
from residual_sph.preregistration import require_lock_sha256, verify_preregistration  # noqa: E402
from residual_sph.probes import load_feature_asset, run_fold_probes  # noqa: E402
from residual_sph.provenance import load_s0_manifest, validate_s0_cell  # noqa: E402
from residual_sph.targets import (  # noqa: E402
    fit_fold_target_bundle,
    load_fold_split_map,
    load_static_sph_ftv_table,
)
from residual_sph.upstream import verify_local_sources  # noqa: E402


SEALED_ROOT = REPO_ROOT / "additional_experiments" / "c1b_overlap_eligibility_ftv_stageb"
SEALED_SRC = SEALED_ROOT / "src"
SENTINEL = SEALED_ROOT / "STAGE_A_GO.json"
SENTINEL_SHA256 = "0b2c9e0af63ce8806525fb15ac9a27f6ab525b0259ccf16001981ca5091afbdb"
DATA_CONTRACT = SEALED_ROOT / "manifests" / "stage_b_data_contract.private.json"
DATA_CONTRACT_SHA256 = "dd22f130043863d4fce8956061fca389894a31874567ed7929e139f32ff5ab27"
TARGET = REPO_ROOT / "additional_experiments/radiomics_next_change/data_audit/radiomics_transition_targets_raw.csv"
TARGET_SHA256 = "26fbde8590fde4612267f02d762af99d65926ff6d0206d0e500577ef394ff75d"
FOLD_SHA256 = "143e482d711225c0611006d99bd7345d2fa1a5c16c65fbaf8399341a0d26aa38"
CONFIRMATION = REPO_ROOT / "additional_experiments/local_response_state_multiseed_confirmation"
S0_MANIFEST = EXPERIMENT_ROOT / "manifests/s0_confirmation_provenance.json"
PREDICTION_ROOT = EXPERIMENT_ROOT / "predictions" / "formal_4x8"


def _private_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    pd.DataFrame(rows).to_csv(path, index=False)
    path.chmod(0o600)


def _validate_new_feature_metadata(
    feature: Path, *, arm: str, seed_base: int, fold: int, lock_sha256: str
) -> dict[str, object]:
    metadata_path = feature.with_suffix(".metadata.json")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("new-arm feature metadata is missing or invalid") from error
    expected = {
        "experiment": "residual_sph_grounding_pilot",
        "arm": arm,
        "seed_base": int(seed_base),
        "fold": int(fold),
        "feature_sha256": file_sha256(feature),
        "feature_shape": [808, 4, 192],
        "feature_dtype": "float32",
        "ftv_head_called": False,
        "sph_head_called": False,
        "test_labels_used": False,
        "clinical_or_pcr_loaded": False,
        "preregistration_lock_sha256": lock_sha256,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"new-arm feature metadata differs at {key}")
    for path_key, hash_key in (
        ("checkpoint_path", "checkpoint_sha256"),
        ("selection_path", "selection_sha256"),
    ):
        bound = Path(str(metadata.get(path_key, ""))).resolve()
        if not bound.is_file() or file_sha256(bound) != metadata.get(hash_key):
            raise ValueError(f"new-arm feature {path_key} binding drifted")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("S0", "S1", "S2", "S2_L10"), required=True)
    parser.add_argument("--seed-base", choices=(2026, 3026), type=int, required=True)
    parser.add_argument("--fold", choices=range(5), type=int, required=True)
    parser.add_argument("--feature", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preregistration-lock-sha256", required=True)
    args = parser.parse_args()
    preregistration = verify_preregistration(EXPERIMENT_ROOT)
    require_lock_sha256(preregistration["lock_sha256"], args.preregistration_lock_sha256)
    verify_local_sources()
    for path, digest, label in (
        (SENTINEL, SENTINEL_SHA256, "Stage-A sentinel"),
        (DATA_CONTRACT, DATA_CONTRACT_SHA256, "Stage-B data contract"),
        (TARGET, TARGET_SHA256, "FTV/SPH target table"),
    ):
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"{label} is missing or hash-mismatched")
    expected_output = (
        PREDICTION_ROOT / f"seed_{args.seed_base}" / args.arm / f"fold_{args.fold}"
    ).resolve()
    if args.output_dir.resolve() != expected_output:
        raise ValueError("probe output identity/path mismatch")
    if expected_output.exists() and any(expected_output.iterdir()):
        raise FileExistsError("refusing to overwrite a probe cell")
    if args.feature is None:
        if args.arm == "S0":
            feature = (
                CONFIRMATION / "features/formal_4x8" / f"seed_{args.seed_base}"
                / "LOCAL3" / f"fold_{args.fold}" / "response_state.private.npz"
            )
        else:
            feature = (
                EXPERIMENT_ROOT / "features/formal_4x8" / f"seed_{args.seed_base}"
                / args.arm / f"fold_{args.fold}" / "response_state.private.npz"
            )
    else:
        feature = args.feature.resolve()
    if args.arm == "S0":
        selection = (
            CONFIRMATION / "checkpoints/formal_4x8" / f"seed_{args.seed_base}"
            / "LOCAL3" / f"fold_{args.fold}" / "selection.json"
        )
        manifest = load_s0_manifest(S0_MANIFEST)
        validate_s0_cell(
            manifest,
            seed_base=args.seed_base,
            fold=args.fold,
            selection_path=selection,
            feature_path=feature,
            feature_metadata_path=feature.with_suffix(".metadata.json"),
        )
    else:
        _validate_new_feature_metadata(
            feature,
            arm=args.arm,
            seed_base=args.seed_base,
            fold=args.fold,
            lock_sha256=preregistration["lock_sha256"],
        )

    sealed_value = str(SEALED_SRC.resolve())
    while sealed_value in sys.path:
        sys.path.remove(sealed_value)
    sys.path.insert(0, sealed_value)
    from c1b_stage_b.gate import require_stage_a_go
    from c1b_stage_b.inputs import StageBDataPaths, load_stage_b_data

    authorization = require_stage_a_go(SENTINEL)
    paths = StageBDataPaths.load(DATA_CONTRACT, DATA_CONTRACT_SHA256)
    if not paths.fold_manifest.is_file() or file_sha256(paths.fold_manifest) != FOLD_SHA256:
        raise ValueError("outer-fold manifest is missing or hash-mismatched")
    data = load_stage_b_data(paths, authorization, verify_cache_files=False)
    table = load_static_sph_ftv_table(TARGET, TARGET_SHA256, expected_patient_count=375)
    split_map = load_fold_split_map(
        paths.fold_manifest,
        FOLD_SHA256,
        fold=args.fold,
        expected_patient_ids=table.patient_ids,
    )
    bundle = fit_fold_target_bundle(table, split_map, fold=args.fold)
    asset = load_feature_asset(
        feature,
        analysis_arm=args.arm,
        seed_base=args.seed_base,
        fold=args.fold,
    )
    current_fold = data.folds.loc[data.folds["fold"].eq(args.fold)]
    expected_split = dict(
        zip(
            current_fold["patient_id"].astype(str),
            current_fold["split"].astype(str),
            strict=True,
        )
    )
    observed_split = dict(zip(asset.patient_ids, asset.splits.astype(str), strict=True))
    if observed_split != expected_split:
        raise ValueError("feature asset does not match the exact frozen primary fold cohort")
    selections, predictions = run_fold_probes(asset, bundle, ftv_records=data.ftv)
    _private_csv(expected_output / "ridge_selection.private.csv", selections)
    _private_csv(expected_output / "ridge_predictions.private.csv", predictions)
    metadata = {
        "schema_version": 1,
        "arm": args.arm,
        "seed_base": args.seed_base,
        "fold": args.fold,
        "feature_sha256": file_sha256(feature),
        "selection_rows": len(selections),
        "private_prediction_rows": len(predictions),
        "test_used_for_fit_or_selection": False,
        "pcr_or_clinical_loaded": False,
        "preregistration_lock_sha256": preregistration["lock_sha256"],
    }
    metadata_path = expected_output / "probe_metadata.private.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata_path.chmod(0o600)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

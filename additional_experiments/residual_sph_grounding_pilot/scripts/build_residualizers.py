#!/usr/bin/env python3
"""Fit and publish identifier-free fold/visit SPH residualizer coefficients."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
SRC = EXPERIMENT_ROOT / "src"
sys.path.insert(0, str(SRC))

from residual_sph.preregistration import verify_preregistration  # noqa: E402
from residual_sph.targets import (  # noqa: E402
    file_sha256,
    fit_fold_target_bundle,
    load_fold_split_map,
    load_static_sph_ftv_table,
    save_public_residualizer_json,
)


DEFAULT_TARGET = (
    REPO_ROOT
    / "additional_experiments/radiomics_next_change/data_audit/radiomics_transition_targets_raw.csv"
)
TARGET_SHA256 = "26fbde8590fde4612267f02d762af99d65926ff6d0206d0e500577ef394ff75d"
DEFAULT_DATA_CONTRACT = (
    REPO_ROOT
    / "additional_experiments/c1b_overlap_eligibility_ftv_stageb/manifests/"
    "stage_b_data_contract.private.json"
)
DATA_CONTRACT_SHA256 = "dd22f130043863d4fce8956061fca389894a31874567ed7929e139f32ff5ab27"
FOLD_SHA256 = "143e482d711225c0611006d99bd7345d2fa1a5c16c65fbaf8399341a0d26aa38"


def _contract_fold_manifest(path: Path) -> Path:
    source = path.resolve()
    if not source.is_file() or file_sha256(source) != DATA_CONTRACT_SHA256:
        raise ValueError("Stage-B data contract is missing or hash-mismatched")
    payload = json.loads(source.read_text(encoding="utf-8"))
    value = Path(str(payload["fold_manifest"])).expanduser()
    return value.resolve() if value.is_absolute() else (source.parent / value).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-table", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--fold-manifest", type=Path)
    parser.add_argument("--data-contract", type=Path, default=DEFAULT_DATA_CONTRACT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXPERIMENT_ROOT / "manifests" / "residualizers",
    )
    args = parser.parse_args()
    preregistration = verify_preregistration(EXPERIMENT_ROOT)
    fold_manifest = (
        args.fold_manifest.resolve()
        if args.fold_manifest is not None
        else _contract_fold_manifest(args.data_contract)
    )
    if not fold_manifest.is_file() or file_sha256(fold_manifest) != FOLD_SHA256:
        raise ValueError("outer-fold manifest is missing or hash-mismatched")
    table = load_static_sph_ftv_table(
        args.target_table, TARGET_SHA256, expected_patient_count=375
    )
    output = args.output_dir.resolve()
    expected_output = (EXPERIMENT_ROOT / "manifests" / "residualizers").resolve()
    if output != expected_output:
        raise ValueError("public residualizers must use manifests/residualizers")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite an existing residualizer inventory")
    output.mkdir(parents=True, exist_ok=False)

    rows: list[dict[str, object]] = []
    fold_artifacts: list[dict[str, object]] = []
    for fold in range(5):
        split_map = load_fold_split_map(
            fold_manifest,
            FOLD_SHA256,
            fold=fold,
            expected_patient_ids=table.patient_ids,
        )
        bundle = fit_fold_target_bundle(table, split_map, fold=fold)
        path = output / f"fold_{fold}.json"
        save_public_residualizer_json(path, bundle)
        fold_artifacts.append(
            {
                "fold": fold,
                "file": path.relative_to(EXPERIMENT_ROOT).as_posix(),
                "artifact_sha256": file_sha256(path),
                "split_counts": bundle.to_public_dict()["split_counts"],
            }
        )
        for fitted in bundle.residualizers:
            rows.append(
                {
                    "fold": fitted.fold,
                    "visit": fitted.visit,
                    "n_train": fitted.n_train,
                    "coefficient": fitted.coefficient,
                    "intercept": fitted.intercept,
                    "residual_train_mean": fitted.residual_center,
                    "residual_train_population_scale": fitted.residual_scale,
                    "residualizer_id": fitted.residualizer_id,
                }
            )
    csv_path = EXPERIMENT_ROOT / "metrics" / "residualizer_fits.csv"
    if csv_path.exists():
        raise FileExistsError(f"refusing to overwrite {csv_path}")
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    inventory = {
        "schema_version": 1,
        "experiment": "residual_sph_grounding_pilot",
        "status": "FOLD_SAFE_RESIDUALIZERS_FITTED",
        "source_target_table_sha256": TARGET_SHA256,
        "outer_fold_manifest_sha256": FOLD_SHA256,
        "preregistration_lock_sha256": preregistration["lock_sha256"],
        "patient_level_values_persisted": False,
        "fold_artifacts": fold_artifacts,
    }
    inventory_path = EXPERIMENT_ROOT / "manifests" / "residualizer_inventory.json"
    inventory_path.write_text(
        json.dumps(inventory, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(inventory, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

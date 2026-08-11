#!/usr/bin/env python3
"""Run or validate the exact 20-cell frozen LOCAL feature matrix."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable).resolve()
sys.path.insert(0, str(ROOT / "scripts"))

from common import (  # noqa: E402
    canonical_sha256,
    file_sha256,
    load_config,
    ordered_sha256,
    private_directory,
    require_preregistration_lock,
)


FEATURE_SHAPE_ZYX = (14, 22, 20)
ORACLE_REGIONS = ("CORE", "PERI10", "PERI20", "LOCAL_REST")
REPRESENTATIVE_SELECTION_RULE = (
    "upper_median_total_core_input_voxel_count_all_four_post_local_core_valid_373_"
    "locked_order_seed2026_LOCAL3_fold0_display_T0"
)
REPRESENTATIVE_CONTRACT_KEYS = frozenset(
    {
        "designated_cell",
        "display_visit",
        "candidate_region",
        "candidate_visits",
        "candidate_validity",
        "candidate_count",
        "ranking",
        "tie_break",
        "median",
        "selection_rule",
        "role",
    }
)
EXPECTED_REPRESENTATIVE_CONTRACT = {
    "designated_cell": {"seed_base": 2026, "arm": "LOCAL3", "fold": 0},
    "display_visit": "T0",
    "candidate_region": "CORE",
    "candidate_visits": ["T0", "T1", "T2", "T3"],
    "candidate_validity": "post_LOCAL_region_valid_true_at_every_candidate_visit",
    "candidate_count": 373,
    "ranking": "ascending_total_CORE_input_voxel_count_across_candidate_visits",
    "tie_break": "stable_locked_oracle_patient_order",
    "median": "upper_median_rank_floor_n_over_2",
    "selection_rule": REPRESENTATIVE_SELECTION_RULE,
    "role": "deidentified_descriptive_only_never_analytic_population",
}
REPRESENTATIVE_PATH = ROOT / "features" / "representative_activation.private.npz"
CONFIG_PATH = ROOT / "configs" / "audit.json"
LOCK_PATH = ROOT / "PREREGISTRATION_LOCK.json"
COMPLETION_PATH = ROOT / "features" / "feature_matrix_complete.private.json"
COMPLETION_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "cell_count",
        "config_sha256",
        "preregistration_lock_sha256",
        "cells",
    }
)


def validate_representative_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    """Require the exact label-free, post-LOCAL representative amendment."""

    if not isinstance(value, Mapping) or set(value) != set(
        REPRESENTATIVE_CONTRACT_KEYS
    ):
        raise ValueError("representative config schema drifted")
    observed = dict(value)
    if canonical_sha256(observed) != canonical_sha256(EXPECTED_REPRESENTATIVE_CONTRACT):
        raise ValueError("representative config contract drifted")
    return observed


def require_representative_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Extract and authenticate the representative contract from audit.json."""

    oracle = config.get("oracle")
    if not isinstance(oracle, Mapping):
        raise ValueError("oracle config contract is absent")
    representative = oracle.get("representative")
    if not isinstance(representative, Mapping):
        raise ValueError("representative config contract is absent")
    return validate_representative_contract(representative)


def cells() -> list[tuple[int, str, int]]:
    return [
        (seed, arm, fold)
        for seed in (2026, 3026)
        for arm in ("LOCAL0", "LOCAL3")
        for fold in range(5)
    ]


def feature_path(seed: int, arm: str, fold: int) -> Path:
    return (
        ROOT
        / "features"
        / f"seed_{seed}"
        / arm
        / f"fold_{fold}"
        / "spatial_statistics.private.npz"
    )


def validate_representative_asset(
    path: Path = REPRESENTATIVE_PATH,
    *,
    expected_sha256: str | None = None,
    representative_contract: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Authenticate and strictly load the de-identified Figure-8 source."""

    contract = (
        require_representative_contract(load_config(CONFIG_PATH, verify_inputs=True))
        if representative_contract is None
        else validate_representative_contract(representative_contract)
    )
    source = Path(path).resolve(strict=True)
    if source.stat().st_mode & 0o077:
        raise PermissionError("representative activation source must remain owner-only")
    observed_sha256 = file_sha256(source)
    if expected_sha256 is not None and observed_sha256 != str(expected_sha256):
        raise ValueError("representative activation hash differs from feature metadata")
    required = {
        "activation_mean_abs",
        "activation_channel_std",
        "local_weight",
        "region_weight",
        "regions",
        "selection_rule",
    }
    with np.load(source, allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise ValueError("representative activation source schema drifted")
        arrays = {name: np.asarray(archive[name]).copy() for name in required}
    for name in ("activation_mean_abs", "activation_channel_std", "local_weight"):
        value = arrays[name]
        if (
            value.shape != FEATURE_SHAPE_ZYX
            or value.dtype != np.float32
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"representative {name} shape/dtype/value drifted")
    region_weight = arrays["region_weight"]
    if (
        region_weight.shape != (4, *FEATURE_SHAPE_ZYX)
        or region_weight.dtype != np.float32
        or not np.isfinite(region_weight).all()
    ):
        raise ValueError("representative region weights shape/dtype/value drifted")
    if np.any(arrays["activation_mean_abs"] < 0) or np.any(
        arrays["activation_channel_std"] < 0
    ):
        raise ValueError("representative activation magnitudes/SD must be nonnegative")
    if np.any((arrays["local_weight"] < 0) | (arrays["local_weight"] > 1)) or np.any(
        (region_weight < 0) | (region_weight > 1)
    ):
        raise ValueError("representative spatial weights must lie in [0,1]")
    if tuple(arrays["regions"].astype(str)) != ORACLE_REGIONS:
        raise ValueError("representative region order drifted")
    selection = np.asarray(arrays["selection_rule"])
    if selection.shape != () or str(selection.item()) != contract["selection_rule"]:
        raise ValueError("representative selection rule drifted")
    return arrays


def _execute_device(
    device: str, queue: list[tuple[int, str, int]], lock: threading.Lock
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    while True:
        with lock:
            if not queue:
                break
            seed, arm, fold = queue.pop(0)
        output = feature_path(seed, arm, fold)
        log = ROOT / "logs" / f"export_seed{seed}_{arm}_fold{fold}.private.log"
        private_directory(ROOT / "logs")
        private_directory(log.parent)
        command = [
            str(PYTHON),
            str(ROOT / "scripts" / "export_features.py"),
            "--seed-base",
            str(seed),
            "--arm",
            arm,
            "--fold",
            str(fold),
            "--device",
            device,
        ]
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                command, stdout=stream, stderr=subprocess.STDOUT, text=True, check=False
            )
        record = {
            "seed": seed,
            "arm": arm,
            "fold": fold,
            "device": device,
            "returncode": result.returncode,
            "output": str(output),
            "log": str(log),
        }
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if result.returncode != 0:
            raise RuntimeError(f"feature cell failed; inspect {log}")
    return records


def _authenticate_parent_context() -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate all frozen inputs before the parent may create outputs."""

    config = load_config(CONFIG_PATH, verify_inputs=True)
    lock = require_preregistration_lock(config)
    require_representative_contract(config)
    return config, lock


def validate_complete(
    *,
    config: Mapping[str, Any] | None = None,
    lock: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    # Reuse the formal runner's exact archive/metadata contract so exporter,
    # matrix validation, and analysis cannot silently diverge.
    import run_audit as audit

    if config is None and lock is None:
        config, lock = _authenticate_parent_context()
    elif config is None or lock is None:
        raise ValueError("config and preregistration lock must be supplied together")
    representative_contract = require_representative_contract(config)
    representative_identity = representative_contract["designated_cell"]
    folds = audit.load_fold_manifest(
        config["paths"]["fold_manifest"],
        config["paths"]["fold_manifest_sha256"],
    )
    complete: list[dict[str, Any]] = []
    for seed, arm, fold in cells():
        path = feature_path(seed, arm, fold)
        metadata_path = path.with_suffix(".metadata.json")
        if not path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"missing feature cell: {path}")
        if path.stat().st_mode & 0o077 or metadata_path.stat().st_mode & 0o077:
            raise PermissionError(
                f"feature cell artifacts must remain owner-only: {path}"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or set(metadata) != set(audit.METADATA_KEYS):
            raise ValueError(f"feature cell metadata schema drifted: {metadata_path}")
        if (
            metadata.get("status") != "COMPLETE"
            or metadata.get("p1_projection_parity", {}).get("allclose") is not True
        ):
            raise ValueError(f"feature cell metadata failed: {metadata_path}")
        key = f"seed_{seed}/{arm}/fold_{fold}"
        record = lock["selected_cells"][key]
        if (
            metadata.get("cell") != key
            or metadata.get("seed_base") != seed
            or metadata.get("arm") != arm
            or metadata.get("fold") != fold
            or metadata.get("checkpoint_sha256") != record["checkpoint_sha256"]
            or metadata.get("selection_sha256") != record["selection_sha256"]
        ):
            raise ValueError(
                f"feature cell identity/provenance drifted: {metadata_path}"
            )
        representative = metadata.get("representative_activation")
        designated = (seed, arm, fold) == (
            representative_identity["seed_base"],
            representative_identity["arm"],
            representative_identity["fold"],
        )
        if designated:
            if not isinstance(representative, dict) or set(representative) != {
                "path",
                "sha256",
                "selection_rule",
                "contains_patient_identifier",
            }:
                raise ValueError(
                    "designated feature metadata lacks representative provenance"
                )
            if (
                Path(str(representative["path"])).resolve()
                != REPRESENTATIVE_PATH.resolve()
                or representative["selection_rule"]
                != representative_contract["selection_rule"]
                or representative["contains_patient_identifier"] is not False
            ):
                raise ValueError("representative activation provenance drifted")
            validate_representative_asset(
                REPRESENTATIVE_PATH,
                expected_sha256=str(representative["sha256"]),
                representative_contract=representative_contract,
            )
        elif representative is not None:
            raise ValueError(
                "representative provenance appears outside its designated cell"
            )
        observed_feature_sha256 = file_sha256(path)
        if metadata.get("feature_sha256") != observed_feature_sha256:
            raise ValueError(f"feature hash differs from metadata: {path}")
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "patient_id",
                "split",
                "mean",
                "std",
                "q25",
                "q50",
                "q75",
                "oracle_mean",
                "oracle_std",
                "oracle_valid",
                "oracle_regions",
                "arm",
                "seed_base",
                "fold",
            }
            if set(archive.files) != required:
                raise ValueError(f"feature cell schema drifted: {path}")
            if any(
                archive[name].shape != (808, 4, 128)
                for name in ("mean", "std", "q25", "q50", "q75")
            ):
                raise ValueError(f"mask-free statistic shape drifted: {path}")
            if archive["oracle_mean"].shape != (808, 4, 4, 128) or archive[
                "oracle_std"
            ].shape != (808, 4, 4, 128):
                raise ValueError(f"oracle statistic shape drifted: {path}")
            if (
                str(np.asarray(archive["arm"]).item()) != arm
                or int(np.asarray(archive["seed_base"]).item()) != seed
                or int(np.asarray(archive["fold"]).item()) != fold
            ):
                raise ValueError(f"feature archive cell identity drifted: {path}")
            patient_id = np.asarray(archive["patient_id"]).astype(str)
            split = np.asarray(archive["split"]).astype(str)
            if (
                patient_id.shape != (808,)
                or split.shape != (808,)
                or len(set(patient_id)) != 808
            ):
                raise ValueError(f"feature archive patient/split shape drifted: {path}")
            if (
                ordered_sha256(patient_id) != metadata.get("patient_order_sha256")
                or ordered_sha256(split) != metadata.get("split_order_sha256")
                or ordered_sha256(patient_id)
                != record["reference"]["patient_order_sha256"]
                or ordered_sha256(split) != record["reference"]["split_order_sha256"]
            ):
                raise ValueError(f"feature archive patient/split order drifted: {path}")
            for name in (
                "mean",
                "std",
                "q25",
                "q50",
                "q75",
                "oracle_mean",
                "oracle_std",
            ):
                values = np.asarray(archive[name])
                if values.dtype != np.float32 or not np.isfinite(values).all():
                    raise ValueError(
                        f"feature archive {name} dtype/value drifted: {path}"
                    )
        audit.load_spatial_feature_asset(path, folds, config, lock)
        complete.append(
            {
                "seed": seed,
                "arm": arm,
                "fold": fold,
                "feature_sha256": observed_feature_sha256,
                "max_parity_abs": metadata["p1_projection_parity"][
                    "max_abs_difference"
                ],
            }
        )
    return {
        "schema_version": 1,
        "status": "COMPLETE",
        "cell_count": len(complete),
        "config_sha256": file_sha256(CONFIG_PATH),
        "preregistration_lock_sha256": file_sha256(LOCK_PATH),
        "cells": complete,
    }


def validate_completion_marker(
    path: Path, expected: Mapping[str, Any]
) -> dict[str, Any]:
    """Require an owner-only existing marker to equal fresh validation exactly."""

    source = Path(path).resolve(strict=True)
    if source.parent.stat().st_mode & 0o077 or source.stat().st_mode & 0o077:
        raise PermissionError("feature-matrix completion marker must remain owner-only")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("feature-matrix completion marker is unreadable") from error
    if (
        not isinstance(payload, dict)
        or set(payload) != set(COMPLETION_KEYS)
        or payload != dict(expected)
    ):
        raise ValueError("feature-matrix completion marker differs from current assets")
    return payload


def publish_completion_marker(path: Path, payload: Mapping[str, Any]) -> Path:
    """Atomically create, but never replace, the private completion marker."""

    destination = Path(path).resolve()
    private_directory(destination.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(dict(payload), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o600)
        try:
            # Same-directory hard-link publication is atomic and fails if the
            # immutable destination appeared concurrently; it never replaces.
            os.link(temporary, destination)
        except FileExistsError:
            validate_completion_marker(destination, payload)
        else:
            destination.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def main() -> None:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    # This is deliberately the first operation after side-effect-free argument
    # parsing.  No feature directory, log, marker, or subprocess may be created
    # until both the hash-bound config and the frozen lock authenticate here in
    # the parent process.
    config, preregistration_lock = _authenticate_parent_context()
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    if not devices:
        raise ValueError("at least one device is required")
    if COMPLETION_PATH.exists():
        summary = validate_complete(config=config, lock=preregistration_lock)
        validate_completion_marker(COMPLETION_PATH, summary)
        print(
            json.dumps(
                {"status": "COMPLETE", "cell_count": summary["cell_count"]},
                sort_keys=True,
            )
        )
        return
    missing = [
        cell
        for cell in cells()
        if not feature_path(*cell).is_file()
        or not feature_path(*cell).with_suffix(".metadata.json").is_file()
    ]
    if args.execute and missing:
        queue = list(missing)
        queue_lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            futures = [
                executor.submit(_execute_device, device, queue, queue_lock)
                for device in devices
            ]
            for future in as_completed(futures):
                future.result()
    elif missing:
        print(
            json.dumps(
                {
                    "status": "PREFLIGHT",
                    "missing_cells": len(missing),
                    "devices": devices,
                },
                sort_keys=True,
            )
        )
        return
    summary = validate_complete(config=config, lock=preregistration_lock)
    if COMPLETION_PATH.exists():
        # Another parent may have completed between validation and publication;
        # only an exact marker is acceptable and it is never overwritten.
        validate_completion_marker(COMPLETION_PATH, summary)
    else:
        publish_completion_marker(COMPLETION_PATH, summary)
    print(
        json.dumps(
            {"status": "COMPLETE", "cell_count": summary["cell_count"]}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()

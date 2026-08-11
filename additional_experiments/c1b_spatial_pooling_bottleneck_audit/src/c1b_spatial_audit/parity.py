"""Fail-closed P0 equivalence against immutable formal Stage-B states."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import (
    EXPERIMENT_ROOT,
    REPO_ROOT,
    cell_key,
    cells,
    file_sha256,
    reference_feature_path,
)
from .exporter import validate_feature_export
from .probes import FrozenStateAsset, load_frozen_state_asset
from .runtime import verify_preregistration


P0_RTOL = 1e-5
P0_ATOL = 1e-6


def _reference_state(path: Path) -> dict[str, np.ndarray]:
    expected = {"patient_id", "split", "response_state", "arm", "seed_base", "fold"}
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected:
            raise ValueError("immutable P0 reference schema drifted")
        arrays = {key: archive[key].copy() for key in archive.files}
    state = np.asarray(arrays["response_state"])
    patient_id = np.asarray(arrays["patient_id"]).astype(str)
    split = np.asarray(arrays["split"]).astype(str)
    if state.shape != (808, 4, 192) or state.dtype != np.dtype(np.float32):
        raise ValueError("immutable P0 reference must be float32 [808,4,192]")
    if patient_id.shape != (808,) or split.shape != (808,):
        raise ValueError("immutable P0 reference identity vectors drifted")
    if not np.isfinite(state).all():
        raise FloatingPointError("immutable P0 reference contains nonfinite values")
    return arrays


def compare_p0_asset(
    candidate: FrozenStateAsset,
    reference_path: str | Path,
    *,
    rtol: float = P0_RTOL,
    atol: float = P0_ATOL,
) -> dict[str, Any]:
    """Compare all 808x4x192 candidate elements, identities, and validity."""

    if candidate.pooling != "P0":
        raise ValueError("P0 parity can only consume a P0 frozen state")
    if candidate.state.shape != (808, 4, 192):
        raise ValueError("candidate P0 must be float32 [808,4,192]")
    if not bool(candidate.state_valid.all()):
        raise ValueError("candidate P0 must be valid for every patient/visit")
    reference_source = Path(reference_path).resolve()
    reference = _reference_state(reference_source)
    expected_identity = {
        "arm": str(np.asarray(reference["arm"]).item()),
        "seed_base": int(np.asarray(reference["seed_base"]).item()),
        "fold": int(np.asarray(reference["fold"]).item()),
    }
    for key, value in expected_identity.items():
        if candidate.identity[key] != value:
            raise ValueError(f"candidate/reference identity mismatch at {key}")
    if not np.array_equal(candidate.patient_id, reference["patient_id"].astype(str)):
        raise ValueError("candidate/reference patient order differs")
    if not np.array_equal(candidate.split, reference["split"].astype(str)):
        raise ValueError("candidate/reference split labels differ")

    observed = candidate.state.astype(np.float64)
    expected = np.asarray(reference["response_state"], dtype=np.float64)
    difference = observed - expected
    absolute = np.abs(difference)
    close = np.isclose(observed, expected, rtol=float(rtol), atol=float(atol))
    exact = candidate.state.view(np.uint32) == np.asarray(
        reference["response_state"]
    ).view(np.uint32)
    row = {
        **expected_identity,
        "pooling": "P0",
        "patients": 808,
        "visits": 3232,
        "feature_dim": 192,
        "elements": int(observed.size),
        "allclose_fraction": float(close.mean()),
        "bitwise_equal_fraction": float(exact.mean()),
        "max_absolute_error": float(absolute.max()),
        "mean_absolute_error": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "finite_fraction": float(np.isfinite(observed).mean()),
        "identity_exact": True,
        "split_exact": True,
        "state_valid_fraction": 1.0,
        "rtol": float(rtol),
        "atol": float(atol),
        "status": "PASS" if bool(close.all()) else "FAIL",
        "candidate_sha256": (
            None if candidate.source_path is None else file_sha256(candidate.source_path)
        ),
        "reference_sha256": file_sha256(reference_source),
    }
    return row


def _discover_p0_assets(feature_root: str | Path) -> dict[tuple[int, str, int], Path]:
    root = Path(feature_root).resolve()
    discovered: dict[tuple[int, str, int], Path] = {}
    for path in sorted(root.rglob("*.private.npz")):
        try:
            with np.load(path, allow_pickle=False) as archive:
                if "pooling" not in archive.files:
                    continue
                pooling = str(np.asarray(archive["pooling"]).item()).upper()
                if pooling != "P0":
                    continue
                key = (
                    int(np.asarray(archive["seed_base"]).item()),
                    str(np.asarray(archive["arm"]).item()).upper(),
                    int(np.asarray(archive["fold"]).item()),
                )
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError(f"cannot inspect pooled-state asset: {path}") from exc
        if key in discovered:
            raise ValueError(f"duplicate P0 asset for {key}")
        discovered[key] = path
    expected = set(cells())
    if set(discovered) != expected:
        missing = sorted(expected.difference(discovered))
        extra = sorted(set(discovered).difference(expected))
        raise ValueError(f"P0 matrix is incomplete: missing={missing}, extra={extra}")
    return discovered


def verify_p0_matrix(feature_root: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Verify preregistration, locked reference hashes, and all 40 P0 cells."""

    lock = verify_preregistration()
    assets = _discover_p0_assets(feature_root)
    rows: list[dict[str, Any]] = []
    for seed, arm, fold in cells():
        key = cell_key(seed, arm, fold)
        reference = reference_feature_path(seed, arm, fold)
        locked = lock["formal_p0_references"][key]
        if locked["feature_path"] != reference.relative_to(REPO_ROOT).as_posix():
            raise ValueError(f"locked reference path drifted for {key}")
        if locked["feature_sha256"] != file_sha256(reference):
            raise ValueError(f"locked reference SHA-256 drifted for {key}")
        candidate_path = assets[(seed, arm, fold)]
        # Structural equality alone is insufficient: bind every candidate to
        # the selected checkpoint, sidecar, immutable reference, and frozen
        # implementation before comparing numerical values.
        validate_feature_export(
            candidate_path,
            expected_arm=arm,
            expected_seed_base=seed,
            expected_fold=fold,
            expected_pooling="P0",
            expected_patient_count=808,
            verify_live_inputs=True,
        )
        candidate = load_frozen_state_asset(candidate_path)
        rows.append(compare_p0_asset(candidate, reference))
    frame = pd.DataFrame(rows).sort_values(["seed_base", "arm", "fold"]).reset_index(
        drop=True
    )
    if len(frame) != 40:
        raise AssertionError("P0 parity did not produce exactly 40 cell rows")
    all_pass = bool(frame["status"].eq("PASS").all())
    summary = {
        "schema_version": 1,
        "status": "PASS" if all_pass else "FAIL",
        "formal_cells": 40,
        "patients_per_cell": 808,
        "visits_per_patient": 4,
        "feature_dimension": 192,
        "compared_elements": int(frame["elements"].sum()),
        "allclose_required_fraction": 1.0,
        "allclose_observed_fraction": float(
            np.average(frame["allclose_fraction"], weights=frame["elements"])
        ),
        "bitwise_equal_fraction": float(
            np.average(frame["bitwise_equal_fraction"], weights=frame["elements"])
        ),
        "maximum_absolute_error": float(frame["max_absolute_error"].max()),
        "mean_absolute_error": float(
            np.average(frame["mean_absolute_error"], weights=frame["elements"])
        ),
        "rtol": P0_RTOL,
        "atol": P0_ATOL,
        "preregistration_lock_sha256": file_sha256(
            EXPERIMENT_ROOT / "PREREGISTRATION_LOCK.json"
        ),
        "probe_execution_authorized": all_pass,
    }
    return frame, summary


__all__ = [
    "P0_ATOL",
    "P0_RTOL",
    "compare_p0_asset",
    "verify_p0_matrix",
]

#!/usr/bin/env python3
"""Run frozen A0/A1 FTV, pCR, dynamics, bootstrap, gates, and reporting."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))
sys.path.insert(
    0,
    str(
        REPO_ROOT
        / "additional_experiments"
        / "c1b_overlap_eligibility_ftv_stageb"
        / "src"
    ),
)
sys.path.insert(0, str(EXPERIMENT_ROOT / "scripts"))

from freeze_preregistration import file_sha256, verify  # noqa: E402
from patch_token_wm.downstream import (  # noqa: E402
    FoldClinicalPreprocessor,
    build_pcr_feature_sets,
    load_pcr_labels_downstream_only,
)
from patch_token_wm.evaluation import (  # noqa: E402
    FoldSafeTokenSummarizer,
    evaluate_all_gates,
    paired_pcr_bootstrap,
    pcr_metrics,
    regression_metrics,
    run_outer_fold_logistic,
    run_outer_fold_ridge,
)
from c1b_stage_b.data import make_splits  # noqa: E402
from c1b_stage_b.gate import require_stage_a_go  # noqa: E402
from c1b_stage_b.inputs import StageBDataPaths, load_stage_b_data  # noqa: E402
from c1b_stage_b.targets import fit_static_probe_transform  # noqa: E402
from train_cell import (  # noqa: E402
    DEFAULT_DATA_CONTRACT,
    DEFAULT_DATA_CONTRACT_SHA256,
    DEFAULT_STAGE_A_SENTINEL,
)


SEEDS = (2026, 3026)
FOLDS = tuple(range(5))
ARMS = ("A0_LOCAL3", "A1_PATCH3")
VISITS = ("T0", "T1", "T2", "T3")
TRANSITIONS = ("T0_to_T1", "T1_to_T2", "T2_to_T3")
TIMING_LABELS = ("T0", "T0-T1", "T0-T2", "T0-T3_late")
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 260812
CLINICAL_TABLE = Path(
    "/data/data/Preprocessed/I-SPY2/clinical_labels_complete4visits.csv"
)
CLINICAL_TABLE_SHA256 = (
    "b3355f8ac80cf8f0fa95722b8d8a8b73d96790e9ded5c491ddb5b2e6a7793436"
)
CLINICAL_USECOLS = (
    "patient_id",
    "label_pcr",
    "label_hr",
    "label_her2",
    "label_mp",
    "age_at_screening",
    "race_simple",
    "menopausal_status_simple",
    "ethnicity",
    "arm",
)


def _atomic_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        raise ValueError(f"refusing to write empty table {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        os.chmod(temporary, 0o600)
        np.savez(temporary, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _cell_set(rows: Any, *, label: str) -> dict[tuple[int, int], dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError(f"{label} cells must be a list")
    output: dict[tuple[int, int], dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, dict):
            raise ValueError(f"{label} cell must be an object")
        key = (int(raw.get("seed_base", -1)), int(raw.get("fold", -1)))
        if key in output:
            raise ValueError(f"{label} contains a duplicate cell")
        output[key] = raw
    expected = {(seed, fold) for seed in SEEDS for fold in FOLDS}
    if set(output) != expected:
        raise ValueError(f"{label} does not contain the exact formal 2x5 matrix")
    return output


def _validated_channel_moments(
    value: Any, *, expected_patients: int, label: str
) -> tuple[int, np.ndarray, np.ndarray]:
    """Validate the exact float64 sufficient statistics used by Gate A."""

    if not isinstance(value, dict):
        raise ValueError(f"{label} channel moments are missing")
    count = int(value.get("count_per_channel", -1))
    if count != int(expected_patients) * 3 * 250:
        raise ValueError(f"{label} channel-moment count differs")
    channel_sum = np.asarray(value.get("channel_sum"), dtype=np.float64)
    channel_sum_squares = np.asarray(value.get("channel_sum_squares"), dtype=np.float64)
    if (
        channel_sum.shape != (128,)
        or channel_sum_squares.shape != (128,)
        or not np.isfinite(channel_sum).all()
        or not np.isfinite(channel_sum_squares).all()
        or np.any(channel_sum_squares < 0.0)
    ):
        raise ValueError(f"{label} channel moments are not 128 finite values")
    return count, channel_sum, channel_sum_squares


def _validate_formal_inputs(lock_sha: str) -> dict[tuple[int, int], dict[str, Any]]:
    """Authenticate every frozen artifact before fitting a representation transform."""

    matrix_path = EXPERIMENT_ROOT / "metrics" / "formal_matrix_complete.json"
    exports_path = EXPERIMENT_ROOT / "metrics" / "formal_exports_complete.json"
    a0_path = EXPERIMENT_ROOT / "manifests" / "a0_reference.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    exports = json.loads(exports_path.read_text(encoding="utf-8"))
    a0 = json.loads(a0_path.read_text(encoding="utf-8"))
    for label, payload in (("matrix", matrix), ("exports", exports)):
        if (
            payload.get("status") != "COMPLETE"
            or int(payload.get("run_count", -1)) != 10
            or payload.get("preregistration_lock_sha256") != lock_sha
            or payload.get("seeds") != list(SEEDS)
            or payload.get("folds") != list(FOLDS)
        ):
            raise ValueError(f"{label} completion contract drifted")
    if matrix.get("all_training_pcr_free") is not True:
        raise ValueError("formal matrix is not certified pCR-free")
    if matrix.get("all_test_blind_selection") is not True:
        raise ValueError("formal matrix selection is not test-blind")
    if matrix.get("all_cells_finite_noncollapsed") is not True:
        raise ValueError(
            "formal matrix contains a nonfinite or collapsed selected cell"
        )
    if exports.get("pcr_loaded") is not False:
        raise ValueError("formal exports are not certified pCR-free")
    if exports.get("all_token_shapes") != [[808, 4, 500, 128]]:
        raise ValueError("formal export token shape contract drifted")
    export_loader = exports.get("data_loader")
    if (
        not isinstance(export_loader, dict)
        or export_loader.get("batch_size") != 4
        or export_loader.get("multiprocessing_start_method") not in {"spawn", "none"}
    ):
        raise ValueError("formal export loader contract drifted")
    if exports.get("cuda_allocator_config") != "expandable_segments:True":
        raise ValueError("formal export CUDA allocator provenance drifted")
    matrix_cells = _cell_set(matrix.get("cells"), label="matrix")
    export_cells = _cell_set(exports.get("cells"), label="exports")
    if (
        a0.get("status") != "A0_REFERENCE_IMPORTED"
        or a0.get("experiment") != "local_response_state_multiseed_confirmation"
        or a0.get("arm") != "LOCAL3"
        or int(a0.get("cell_count", -1)) != 10
        or a0.get("source_preregistration_lock_sha256")
        != "a4e1cd2d8b61a7130da2b2eb6dc04e9a5355f44d0a37f4ceccf2fba48b35a9ee"
    ):
        raise ValueError("A0 confirmed-reference manifest drifted")
    a0_cells = _cell_set(a0.get("cells"), label="A0 reference")
    authenticated: dict[tuple[int, int], dict[str, Any]] = {}
    for key in sorted(matrix_cells):
        seed, fold = key
        checkpoint = (
            EXPERIMENT_ROOT
            / "checkpoints"
            / "a1_formal"
            / f"seed_{seed}"
            / f"fold_{fold}"
            / "selected.pt"
        )
        a1_selection = checkpoint.with_name("selection.json")
        token = (
            EXPERIMENT_ROOT
            / "features"
            / "a1_formal"
            / f"seed_{seed}"
            / f"fold_{fold}"
            / "tokens.private.npz"
        )
        dynamics = token.with_name("dynamics.private.npz")
        token_metadata = token.with_suffix(".metadata.json")
        a0_checkpoint = (
            EXPERIMENT_ROOT
            / "checkpoints"
            / "a0_local3_reference"
            / f"seed_{seed}"
            / f"fold_{fold}"
            / "selected.pt"
        )
        a0_selection = a0_checkpoint.with_name("selection.json")
        a0_feature = (
            EXPERIMENT_ROOT
            / "features"
            / "a0_local3_reference"
            / f"seed_{seed}"
            / f"fold_{fold}"
            / "response_state.private.npz"
        )
        a0_metadata = a0_feature.with_name("response_state.private.metadata.json")
        required = (
            checkpoint,
            a1_selection,
            token,
            dynamics,
            token_metadata,
            a0_checkpoint,
            a0_selection,
            a0_feature,
            a0_metadata,
        )
        if not all(path.is_file() for path in required):
            raise FileNotFoundError(
                f"formal cell artifact is missing: seed={seed} fold={fold}"
            )
        if matrix_cells[key].get("selected_checkpoint_sha256") != file_sha256(
            checkpoint
        ):
            raise ValueError("A1 selected checkpoint SHA-256 mismatch")
        if matrix_cells[key].get("selection_sha256") != file_sha256(a1_selection):
            raise ValueError("A1 selection SHA-256 mismatch")
        if export_cells[key].get("token_feature_sha256") != file_sha256(token):
            raise ValueError("A1 token SHA-256 mismatch")
        if export_cells[key].get("dynamics_sha256") != file_sha256(dynamics):
            raise ValueError("A1 dynamics SHA-256 mismatch")
        if export_cells[key].get("export_metadata_sha256") != file_sha256(
            token_metadata
        ):
            raise ValueError("A1 export metadata SHA-256 mismatch")
        a1_selection_payload = json.loads(a1_selection.read_text(encoding="utf-8"))
        if (
            a1_selection_payload.get("arm") != "A1_PATCH3"
            or int(a1_selection_payload.get("seed_base", -1)) != seed
            or int(a1_selection_payload.get("fold", -1)) != fold
            or a1_selection_payload.get("preregistration_lock_sha256") != lock_sha
            or a1_selection_payload.get("test_data_used") is not False
            or a1_selection_payload.get("pcr_loaded") is not False
            or a1_selection_payload.get("optimization_safety_pass") is not True
        ):
            raise ValueError("A1 selection identity/firewall drifted")
        a1_metadata_payload = json.loads(token_metadata.read_text(encoding="utf-8"))
        expected_metadata = {
            "status": "COMPLETE",
            "arm": "A1_PATCH3",
            "seed_base": seed,
            "fold": fold,
            "preregistration_lock_sha256": lock_sha,
            "pcr_loaded": False,
            "condition_in_exported_tokens": False,
            "token_shape": [808, 4, 500, 128],
            "export_batch_size": 4,
            "mask_schedule": (
                "effective_seed_epoch0_logical_batch_index_patient_sha256_transition"
            ),
        }
        for field, expected_value in expected_metadata.items():
            if a1_metadata_payload.get(field) != expected_value:
                raise ValueError(f"A1 export metadata drifted at {field}")
        if (
            a1_metadata_payload.get("checkpoint_sha256") != file_sha256(checkpoint)
            or a1_metadata_payload.get("token_feature_sha256") != file_sha256(token)
            or a1_metadata_payload.get("dynamics_sha256") != file_sha256(dynamics)
        ):
            raise ValueError("A1 export metadata artifact chain drifted")
        test_patients = int(a1_metadata_payload.get("test_dynamics_patients", -1))
        if (
            test_patients < 1
            or export_cells[key].get("test_dynamics_patients") != test_patients
        ):
            raise ValueError("A1 test-dynamics patient count drifted")
        _validated_channel_moments(
            a1_metadata_payload.get("target_channel_moments"),
            expected_patients=test_patients,
            label="target",
        )
        _validated_channel_moments(
            a1_metadata_payload.get("prediction_channel_moments"),
            expected_patients=test_patients,
            label="prediction",
        )
        a0_row = a0_cells[key]
        for path, field in (
            (a0_checkpoint, "selected_checkpoint_sha256"),
            (a0_selection, "selection_sha256"),
            (a0_feature, "response_feature_sha256"),
            (a0_metadata, "response_metadata_sha256"),
        ):
            if a0_row.get(field) != file_sha256(path):
                raise ValueError(f"A0 artifact SHA-256 mismatch at {field}")
        selection = json.loads(a0_selection.read_text(encoding="utf-8"))
        metadata = json.loads(a0_metadata.read_text(encoding="utf-8"))
        if (
            selection.get("arm") != "LOCAL3"
            or int(selection.get("seed_base", -1)) != seed
            or int(selection.get("fold", -1)) != fold
            or selection.get("test_data_used") is not False
            or selection.get("pcr_used") is not False
        ):
            raise ValueError("A0 selection identity/firewall drifted")
        if (
            metadata.get("experiment") != "local_response_state_multiseed_confirmation"
            or metadata.get("arm") != "LOCAL3"
            or int(metadata.get("seed_base", -1)) != seed
            or int(metadata.get("fold", -1)) != fold
            or metadata.get("feature_shape") != [808, 4, 192]
            or metadata.get("test_labels_used") is not False
            or metadata.get("ftv_head_called") is not False
        ):
            raise ValueError("A0 feature metadata identity/firewall drifted")
        authenticated[key] = {
            "matrix": matrix_cells[key],
            "export": export_cells[key],
            "a0": a0_row,
        }
    return authenticated


def _clinical_table() -> pd.DataFrame:
    if file_sha256(CLINICAL_TABLE) != CLINICAL_TABLE_SHA256:
        raise ValueError("downstream clinical table SHA-256 mismatch")
    frame = pd.read_csv(CLINICAL_TABLE, usecols=list(CLINICAL_USECOLS))
    if set(frame) != set(CLINICAL_USECOLS) or len(frame) != 808:
        raise ValueError("downstream clinical table schema/population drifted")
    frame["patient_id"] = frame["patient_id"].astype(str)
    if frame["patient_id"].duplicated().any():
        raise ValueError("clinical table patient IDs are not unique")
    return frame.set_index("patient_id", drop=False)


def _load_a0(seed: int, fold: int) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    path = (
        EXPERIMENT_ROOT
        / "features"
        / "a0_local3_reference"
        / f"seed_{seed}"
        / f"fold_{fold}"
        / "response_state.private.npz"
    )
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {
            "patient_id",
            "split",
            "response_state",
            "arm",
            "seed_base",
            "fold",
        }:
            raise ValueError("A0 feature schema drifted")
        patient_ids = tuple(archive["patient_id"].astype(str))
        split = archive["split"].astype(str)
        state = archive["response_state"].astype(np.float64)
        if str(archive["arm"].item()) != "LOCAL3":
            raise ValueError("A0 arm identity drifted")
        if (
            int(archive["seed_base"].item()) != seed
            or int(archive["fold"].item()) != fold
        ):
            raise ValueError("A0 feature cell identity drifted")
    if (
        len(patient_ids) != 808
        or len(set(patient_ids)) != 808
        or set(split) != {"train", "val", "test"}
        or split.shape != (808,)
        or state.shape != (808, 4, 192)
        or not np.isfinite(state).all()
    ):
        raise ValueError("A0 response-state tensor is invalid")
    return patient_ids, split, state


def _load_a1(
    seed: int, fold: int
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, dict[str, Any]]:
    path = (
        EXPERIMENT_ROOT
        / "features"
        / "a1_formal"
        / f"seed_{seed}"
        / f"fold_{fold}"
        / "tokens.private.npz"
    )
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {
            "patient_id",
            "split",
            "tokens",
            "fractional_weights",
            "coordinates_xyz_mm",
            "seed_base",
            "fold",
        }:
            raise ValueError("A1 token feature schema drifted")
        patient_ids = tuple(archive["patient_id"].astype(str))
        split = archive["split"].astype(str)
        tokens = archive["tokens"].copy()
        weights = archive["fractional_weights"].copy()
        coordinates = archive["coordinates_xyz_mm"].copy()
        if (
            int(archive["seed_base"].item()) != seed
            or int(archive["fold"].item()) != fold
        ):
            raise ValueError("A1 token cell identity drifted")
    if (
        len(patient_ids) != 808
        or len(set(patient_ids)) != 808
        or split.shape != (808,)
        or set(split) != {"train", "val", "test"}
        or tokens.shape != (808, 4, 500, 128)
        or not np.isfinite(tokens).all()
        or weights.shape != (500,)
        or not np.isfinite(weights).all()
        or np.any(weights <= 0.0)
        or np.any(weights > 1.0)
        or coordinates.shape != (500, 3)
        or not np.isfinite(coordinates).all()
    ):
        raise ValueError("A1 token shape drifted")
    train = np.flatnonzero(split == "train")
    summarizer = FoldSafeTokenSummarizer(outer_fold=fold, random_state=seed + fold).fit(
        tokens[train],
        weights,
        train_patient_ids=[patient_ids[index] for index in train],
        split="train",
    )
    state = summarizer.transform(tokens, weights)
    if state.shape != (808, 4, 192):
        raise AssertionError("primary A1 state is not 192-D")
    pca_path = path.with_name("pca64.private.npz")
    expected_pca = {
        "mean": summarizer.pca_mean_.astype(np.float32),
        "components": summarizer.pca_components_.astype(np.float32),
        "seed_base": np.asarray(seed, dtype=np.int64),
        "fold": np.asarray(fold, dtype=np.int64),
        "token_feature_sha256": np.asarray(file_sha256(path)),
        "labels_used": np.asarray(False),
    }
    if not pca_path.exists():
        _atomic_npz(pca_path, **expected_pca)
    with np.load(pca_path, allow_pickle=False) as archive:
        if set(archive.files) != set(expected_pca):
            raise ValueError("frozen PCA artifact schema drifted")
        for name, expected in expected_pca.items():
            observed = archive[name]
            if observed.shape != expected.shape or not np.array_equal(
                observed, expected
            ):
                raise ValueError(f"frozen PCA artifact differs at {name}")
    provenance = dict(summarizer.provenance)
    provenance["pca_artifact_sha256"] = file_sha256(pca_path)
    provenance["token_feature_sha256"] = file_sha256(path)
    return patient_ids, split, state, provenance


def _validate_fold_labels(
    patient_ids: tuple[str, ...], split: np.ndarray, data: Any, fold: int
) -> None:
    current = data.folds.loc[data.folds["fold"].eq(fold), ["patient_id", "split"]]
    expected = dict(
        zip(
            current["patient_id"].astype(str), current["split"].astype(str), strict=True
        )
    )
    observed = dict(zip(patient_ids, split.astype(str), strict=True))
    if observed != expected:
        raise ValueError("feature split labels differ from the locked fold")


def _prepare_label_free_states(
    data: Any,
    *,
    preregistration_lock_sha256: str,
) -> tuple[
    dict[tuple[int, int, str], tuple[tuple[str, ...], np.ndarray, np.ndarray]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Freeze all ten PCA transforms and run FTV before pCR can be loaded."""

    states: dict[
        tuple[int, int, str], tuple[tuple[str, ...], np.ndarray, np.ndarray]
    ] = {}
    pca_rows: list[dict[str, Any]] = []
    ftv_prediction_rows: list[dict[str, Any]] = []
    ftv_selection_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for fold in FOLDS:
            a0_ids, a0_split, a0_state = _load_a0(seed, fold)
            a1_ids, a1_split, a1_state, pca_provenance = _load_a1(seed, fold)
            if a0_ids != a1_ids or not np.array_equal(a0_split, a1_split):
                raise ValueError("A0/A1 feature patient/split pairing differs")
            _validate_fold_labels(a0_ids, a0_split, data, fold)
            pca_rows.append(
                {
                    "seed_base": seed,
                    "fold": fold,
                    "pca_components": 64,
                    "pca_components_sha256": pca_provenance["pca_components_sha256"],
                    "pca_artifact_sha256": pca_provenance["pca_artifact_sha256"],
                    "token_feature_sha256": pca_provenance["token_feature_sha256"],
                    "n_train_patients": pca_provenance["n_train_patients"],
                    "labels_used": False,
                }
            )
            for arm, state in (("A0_LOCAL3", a0_state), ("A1_PATCH3", a1_state)):
                key = (seed, fold, arm)
                states[key] = (a0_ids, a0_split.copy(), state)
                ftv_rows, ftv_select = _run_ftv_cell(
                    seed=seed,
                    fold=fold,
                    arm=arm,
                    patient_ids=a0_ids,
                    split=a0_split,
                    state=state,
                    data=data,
                )
                ftv_prediction_rows.extend(ftv_rows)
                ftv_selection_rows.extend(ftv_select)
    if len(states) != 20 or len(pca_rows) != 10:
        raise RuntimeError("all ten label-free PCA cells were not frozen")
    marker = {
        "schema_version": 1,
        "status": "COMPLETE_BEFORE_PCR_ACCESS",
        "cell_count": 10,
        "seed_bases": list(SEEDS),
        "folds": list(FOLDS),
        "labels_used": False,
        "pcr_loaded": False,
        "preregistration_lock_sha256": str(preregistration_lock_sha256),
        "cells": pca_rows,
    }
    marker_path = (
        EXPERIMENT_ROOT / "metrics" / "representation_transforms_complete.json"
    )
    if marker_path.exists():
        if json.loads(marker_path.read_text(encoding="utf-8")) != marker:
            raise ValueError("existing representation-transform marker differs")
    else:
        _atomic_json(marker_path, marker)
    return states, pca_rows, ftv_prediction_rows, ftv_selection_rows


def _ftv_arrays(
    patient_ids: tuple[str, ...], data: Any
) -> tuple[np.ndarray, np.ndarray]:
    values = np.full((len(patient_ids), 4), np.nan, dtype=np.float64)
    valid = np.zeros((len(patient_ids), 4), dtype=bool)
    for index, patient_id in enumerate(patient_ids):
        record = data.ftv.get(patient_id)
        if record is not None:
            values[index] = np.asarray(record.values, dtype=np.float64)
            valid[index] = np.asarray(record.measurement_valid, dtype=bool)
    if int(valid.all(axis=1).sum()) != 375:
        raise ValueError("formal complete-FTV population must contain 375 patients")
    return values, valid


def _run_ftv_cell(
    *,
    seed: int,
    fold: int,
    arm: str,
    patient_ids: tuple[str, ...],
    split: np.ndarray,
    state: np.ndarray,
    data: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    values, valid = _ftv_arrays(patient_ids, data)
    predictions: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    for task, endpoints in (("static", VISITS), ("delta", TRANSITIONS)):
        for endpoint_index, endpoint in enumerate(endpoints):
            row_valid = (
                valid[:, endpoint_index]
                if task == "static"
                else valid[:, endpoint_index] & valid[:, endpoint_index + 1]
            )
            feature = (
                state[:, endpoint_index]
                if task == "static"
                else state[:, endpoint_index + 1] - state[:, endpoint_index]
            )
            target = (
                values[:, endpoint_index]
                if task == "static"
                else values[:, endpoint_index + 1] - values[:, endpoint_index]
            )
            indices = {
                name: np.flatnonzero((split == name) & row_valid)
                for name in ("train", "val", "test")
            }
            if any(len(index) < 2 for index in indices.values()):
                raise ValueError(f"insufficient {task}/{endpoint} rows")
            if task == "static":
                outer_train_ids = tuple(
                    patient_ids[index] for index in np.flatnonzero(split == "train")
                )
                transform = fit_static_probe_transform(data.ftv, outer_train_ids, fold)
                train_y, train_mask = transform.transform_values(
                    target[indices["train"]],
                    np.ones(len(indices["train"]), dtype=bool),
                )
                val_y, val_mask = transform.transform_values(
                    target[indices["val"]],
                    np.ones(len(indices["val"]), dtype=bool),
                )
                if not train_mask.all() or not val_mask.all():
                    raise AssertionError("static target transform rejected valid rows")
                result = run_outer_fold_ridge(
                    feature[indices["train"]],
                    train_y,
                    feature[indices["val"]],
                    val_y,
                    feature[indices["test"]],
                    outer_fold=fold,
                )
                predicted = transform.inverse(result.predictions)
                target_transform = "outer_train_log_winsor_median_iqr_inverse_natural"
            else:
                scaler = StandardScaler().fit(target[indices["train"], None])
                train_y = scaler.transform(target[indices["train"], None]).reshape(-1)
                val_y = scaler.transform(target[indices["val"], None]).reshape(-1)
                result = run_outer_fold_ridge(
                    feature[indices["train"]],
                    train_y,
                    feature[indices["val"]],
                    val_y,
                    feature[indices["test"]],
                    outer_fold=fold,
                )
                predicted = scaler.inverse_transform(
                    result.predictions[:, None]
                ).reshape(-1)
                target_transform = (
                    "literal_delta_outer_train_standardized_inverse_natural"
                )
            selections.append(
                {
                    "seed_base": seed,
                    "fold": fold,
                    "arm": arm,
                    "task": task,
                    "endpoint": endpoint,
                    "selected_alpha": result.selected_hyperparameter,
                    "validation_score": result.validation_score,
                    "n_train": len(indices["train"]),
                    "n_val": len(indices["val"]),
                    "n_test": len(indices["test"]),
                    "target_transform": target_transform,
                }
            )
            for position, row in enumerate(indices["test"]):
                predictions.append(
                    {
                        "patient_id": patient_ids[row],
                        "fold": fold,
                        "seed_base": seed,
                        "arm": arm,
                        "task": task,
                        "endpoint": endpoint,
                        "y_true": float(target[row]),
                        "y_pred": float(predicted[position]),
                    }
                )
    return predictions, selections


def _pool_ftv_metrics(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouping = ["seed_base", "arm", "task", "endpoint"]
    for keys, group in frame.groupby(grouping, sort=True):
        rows.append(
            dict(zip(grouping, keys, strict=True))
            | regression_metrics(group.y_true, group.y_pred)
        )
    endpoint = pd.DataFrame(rows)
    for keys, group in endpoint.groupby(["seed_base", "arm", "task"], sort=True):
        numeric = (
            "spearman",
            "pearson",
            "natural_r2",
            "rmse",
            "mae",
            "prediction_target_variance_ratio",
            "calibration_slope",
        )
        row = {
            "seed_base": keys[0],
            "arm": keys[1],
            "task": keys[2],
            "endpoint": "macro",
            "n": int(group["n"].sum()),
        }
        row.update({name: float(group[name].mean()) for name in numeric})
        rows.append(row)
    return rows


def _macro_spearman(group: pd.DataFrame) -> float:
    values: list[float] = []
    for _endpoint, current in group.groupby("endpoint", sort=True):
        if current.y_true.nunique() < 2 or current.y_pred.nunique() < 2:
            return math.nan
        values.append(float(spearmanr(current.y_true, current.y_pred).statistic))
    return float(np.mean(values))


def _bootstrap_ftv_macro(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    *,
    seed: int,
    task: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    keys = ["patient_id", "fold", "endpoint"]
    ref = reference.loc[reference.task.eq(task), keys + ["y_true", "y_pred"]].rename(
        columns={"y_true": "truth_ref", "y_pred": "prediction_ref"}
    )
    comp = comparison.loc[comparison.task.eq(task), keys + ["y_true", "y_pred"]].rename(
        columns={"y_true": "truth_comp", "y_pred": "prediction_comp"}
    )
    paired = ref.merge(comp, on=keys, validate="one_to_one")
    if len(paired) != len(ref) or not np.array_equal(
        paired.truth_ref, paired.truth_comp
    ):
        raise ValueError("FTV paired prediction rows do not match exactly")
    patients = paired[["patient_id", "fold"]].drop_duplicates()
    if patients.patient_id.duplicated().any():
        raise ValueError("one patient appears in multiple folds")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    grouped_patients = {
        fold: values.patient_id.to_numpy()
        for fold, values in patients.groupby("fold", sort=True)
    }
    point_ref = _macro_spearman(
        paired.rename(columns={"truth_ref": "y_true", "prediction_ref": "y_pred"})
    )
    point_comp = _macro_spearman(
        paired.rename(columns={"truth_comp": "y_true", "prediction_comp": "y_pred"})
    )
    draws = np.full(BOOTSTRAP_DRAWS, np.nan, dtype=np.float64)
    by_patient = {key: value for key, value in paired.groupby("patient_id", sort=False)}
    for draw in range(BOOTSTRAP_DRAWS):
        blocks: list[pd.DataFrame] = []
        for fold in sorted(grouped_patients):
            values = grouped_patients[fold]
            sampled = rng.choice(values, size=len(values), replace=True)
            for sample_index, patient_id in enumerate(sampled):
                block = by_patient[str(patient_id)].copy()
                block["patient_id"] = f"draw{sample_index}_{patient_id}"
                blocks.append(block)
        sampled_rows = pd.concat(blocks, ignore_index=True)
        ref_value = _macro_spearman(
            sampled_rows.rename(
                columns={"truth_ref": "y_true", "prediction_ref": "y_pred"}
            )
        )
        comp_value = _macro_spearman(
            sampled_rows.rename(
                columns={"truth_comp": "y_true", "prediction_comp": "y_pred"}
            )
        )
        draws[draw] = comp_value - ref_value
    finite = draws[np.isfinite(draws)]
    if len(finite) != BOOTSTRAP_DRAWS:
        raise RuntimeError(
            f"{task} FTV bootstrap produced {len(finite)}/{BOOTSTRAP_DRAWS} valid draws"
        )
    summary = {
        "effect": f"A1_minus_A0_{task}_macro_spearman",
        "seed_base": int(seed),
        "reference": point_ref,
        "comparison": point_comp,
        "improvement": point_comp - point_ref,
        "ci_lower": float(np.quantile(finite, 0.025)),
        "ci_upper": float(np.quantile(finite, 0.975)),
        "n_patients": int(len(patients)),
        "n_bootstrap": BOOTSTRAP_DRAWS,
        "n_valid_bootstrap": int(len(finite)),
        "bootstrap_unit": "patient_within_outer_fold",
    }
    return summary, pd.DataFrame(
        {
            "effect": summary["effect"],
            "seed_base": seed,
            "bootstrap_index": np.arange(BOOTSTRAP_DRAWS),
            "improvement": draws,
        }
    )


def _run_logistic(
    features: dict[str, np.ndarray],
    labels: np.ndarray,
    split: np.ndarray,
    *,
    fold: int,
    random_state: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    indices = {name: np.flatnonzero(split == name) for name in ("train", "val", "test")}
    output: dict[str, np.ndarray] = {}
    selections: list[dict[str, Any]] = []
    for model_name, matrix in features.items():
        result = run_outer_fold_logistic(
            matrix[indices["train"]],
            labels[indices["train"]],
            matrix[indices["val"]],
            labels[indices["val"]],
            matrix[indices["test"]],
            outer_fold=fold,
            random_state=random_state,
        )
        output[model_name] = result.predictions
        selections.append(
            {
                "model": model_name,
                "selected_c": result.selected_hyperparameter,
                "validation_auroc": result.validation_score,
                "n_train": len(indices["train"]),
                "n_val": len(indices["val"]),
                "n_test": len(indices["test"]),
            }
        )
    return output, selections


def _run_pcr_cell(
    *,
    seed: int,
    fold: int,
    arm: str,
    patient_ids: tuple[str, ...],
    split: np.ndarray,
    state: np.ndarray,
    data: Any,
    clinical: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frame = clinical.loc[list(patient_ids)].copy()
    labels = load_pcr_labels_downstream_only(
        frame.reset_index(drop=True),
        patient_ids,
        purpose="frozen_downstream_probe",
    )
    train_index = np.flatnonzero(split == "train")
    preprocessor = FoldClinicalPreprocessor(outer_fold=fold).fit(
        frame.iloc[train_index],
        train_patient_ids=[patient_ids[index] for index in train_index],
        split="train",
    )
    # The probe's train-only StandardScaler scales the complete concatenated
    # feature matrix.  Use the deterministic imputed/one-hot design here.
    clinical_design = preprocessor.encode(frame)
    matched_mask = np.asarray([patient_id in data.ftv for patient_id in patient_ids])
    predictions: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    for timing_index, timing in enumerate(TIMING_LABELS):
        full_features = build_pcr_feature_sets(
            clinical_design, state, timing_index, state_dim=192
        )
        selected_full = {name: full_features[name] for name in ("M", "C", "C+M")}
        full_prediction, full_selection = _run_logistic(
            selected_full,
            labels,
            split,
            fold=fold,
            random_state=seed + fold,
        )
        test_rows = np.flatnonzero(split == "test")
        for model_name, probability in full_prediction.items():
            for position, row in enumerate(test_rows):
                predictions.append(
                    {
                        "patient_id": patient_ids[row],
                        "fold": fold,
                        "seed_base": seed,
                        "arm": arm,
                        "population": "full_808",
                        "timing": timing,
                        "model": model_name,
                        "y_true": int(labels[row]),
                        "predicted_probability": float(probability[position]),
                    }
                )
        selections.extend(
            {
                **row,
                "seed_base": seed,
                "fold": fold,
                "arm": arm,
                "population": "full_808",
                "timing": timing,
            }
            for row in full_selection
        )

        matched_rows = np.flatnonzero(matched_mask)
        matched_ids = tuple(patient_ids[index] for index in matched_rows)
        matched_split = split[matched_rows]
        matched_frame = frame.iloc[matched_rows]
        matched_train = np.flatnonzero(matched_split == "train")
        matched_preprocessor = FoldClinicalPreprocessor(outer_fold=fold).fit(
            matched_frame.iloc[matched_train],
            train_patient_ids=[matched_ids[index] for index in matched_train],
            split="train",
        )
        matched_clinical = matched_preprocessor.encode(matched_frame)
        ftv = np.stack([data.ftv[patient_id].values for patient_id in matched_ids])
        matched_features = build_pcr_feature_sets(
            matched_clinical,
            state[matched_rows],
            timing_index,
            ftv=ftv,
            state_dim=192,
        )
        selected_matched = {name: matched_features[name] for name in ("C+F", "C+F+M")}
        matched_labels = labels[matched_rows]
        matched_prediction, matched_selection = _run_logistic(
            selected_matched,
            matched_labels,
            matched_split,
            fold=fold,
            random_state=seed + fold,
        )
        matched_test = np.flatnonzero(matched_split == "test")
        for model_name, probability in matched_prediction.items():
            for position, row in enumerate(matched_test):
                predictions.append(
                    {
                        "patient_id": matched_ids[row],
                        "fold": fold,
                        "seed_base": seed,
                        "arm": arm,
                        "population": "ftv_complete_375",
                        "timing": timing,
                        "model": model_name,
                        "y_true": int(matched_labels[row]),
                        "predicted_probability": float(probability[position]),
                    }
                )
        selections.extend(
            {
                **row,
                "seed_base": seed,
                "fold": fold,
                "arm": arm,
                "population": "ftv_complete_375",
                "timing": timing,
            }
            for row in matched_selection
        )
    return predictions, selections


def _pool_pcr_metrics(frame: pd.DataFrame) -> list[dict[str, Any]]:
    grouping = ["seed_base", "arm", "population", "timing", "model"]
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(grouping, sort=True):
        rows.append(
            dict(zip(grouping, keys, strict=True))
            | pcr_metrics(group.y_true, group.predicted_probability)
        )
    return rows


def _pcr_bootstraps(
    predictions: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    summaries: list[dict[str, Any]] = []
    draws: list[pd.DataFrame] = []
    comparisons = (
        ("E3_A1_minus_A0_MRI", "full_808", "A0_LOCAL3", "M", "A1_PATCH3", "M"),
        ("E4_A1_CplusM_minus_C", "full_808", "A1_PATCH3", "C", "A1_PATCH3", "C+M"),
        (
            "E5_A1_CplusFplusM_minus_CplusF",
            "ftv_complete_375",
            "A1_PATCH3",
            "C+F",
            "A1_PATCH3",
            "C+F+M",
        ),
    )
    for seed in SEEDS:
        for timing in TIMING_LABELS:
            for (
                effect,
                population,
                reference_arm,
                reference_model,
                comparison_arm,
                comparison_model,
            ) in comparisons:
                reference = predictions.loc[
                    predictions.seed_base.eq(seed)
                    & predictions.population.eq(population)
                    & predictions.timing.eq(timing)
                    & predictions.arm.eq(reference_arm)
                    & predictions.model.eq(reference_model)
                ]
                comparison = predictions.loc[
                    predictions.seed_base.eq(seed)
                    & predictions.population.eq(population)
                    & predictions.timing.eq(timing)
                    & predictions.arm.eq(comparison_arm)
                    & predictions.model.eq(comparison_model)
                ]
                result = paired_pcr_bootstrap(
                    reference,
                    comparison,
                    n_bootstrap=BOOTSTRAP_DRAWS,
                    seed=BOOTSTRAP_SEED,
                )
                if (
                    len(result.draws) != BOOTSTRAP_DRAWS
                    or not result.summary["n_valid_bootstrap"].eq(BOOTSTRAP_DRAWS).all()
                ):
                    raise RuntimeError(
                        "pCR bootstrap did not produce 2,000 valid draws"
                    )
                for _, row in result.summary.iterrows():
                    summaries.append(
                        {
                            "effect": effect,
                            "seed_base": seed,
                            "timing": timing,
                            **row.to_dict(),
                        }
                    )
                draw = result.draws.copy()
                draw.insert(0, "timing", timing)
                draw.insert(0, "seed_base", seed)
                draw.insert(0, "effect", effect)
                draws.append(draw)
    return summaries, draws


def _dynamics(
    data: Any,
    authenticated: dict[tuple[int, int], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame]:
    """Pool authenticated OOF dynamics and exact channel moments by seed."""

    metric_fields = (
        "actual_cosine",
        "shuffled_cosine",
        "actual_normalized_mse",
        "shuffled_normalized_mse",
        "target_std",
        "prediction_std",
    )
    spatial_fields = tuple(
        f"{visit}_{band}_normalized_mse"
        for visit in ("T1", "T2", "T3")
        for band in ("central", "inner_local", "outer_local")
    )
    expected_schema = {"patient_id", "fold", *metric_fields, *spatial_fields}
    seed_rows: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []
    patient_frames: list[pd.DataFrame] = []
    for seed in SEEDS:
        fold_frames: list[pd.DataFrame] = []
        target_count = 0
        target_sum = np.zeros(128, dtype=np.float64)
        target_sum_squares = np.zeros(128, dtype=np.float64)
        prediction_count = 0
        prediction_sum = np.zeros(128, dtype=np.float64)
        prediction_sum_squares = np.zeros(128, dtype=np.float64)
        for fold in FOLDS:
            path = (
                EXPERIMENT_ROOT
                / "features"
                / "a1_formal"
                / f"seed_{seed}"
                / f"fold_{fold}"
                / "dynamics.private.npz"
            )
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != expected_schema:
                    raise ValueError("A1 dynamics artifact schema drifted")
                frame = pd.DataFrame({name: archive[name] for name in archive.files})
            frame["patient_id"] = frame["patient_id"].astype(str)
            expected_ids = tuple(
                str(value)
                for value in make_splits(data.folds, fold, data.train_only_ids).test
            )
            if (
                tuple(frame["patient_id"]) != expected_ids
                or frame["patient_id"].duplicated().any()
                or len(frame) != len(expected_ids)
                or not np.array_equal(
                    frame["fold"].to_numpy(),
                    np.full(len(frame), fold, dtype=np.int64),
                )
                or not np.isfinite(
                    frame.loc[:, list(metric_fields + spatial_fields)].to_numpy(
                        dtype=np.float64
                    )
                ).all()
            ):
                raise ValueError("A1 dynamics fold identity/content drifted")
            export_row = authenticated[(seed, fold)]["export"]
            if int(export_row.get("test_dynamics_patients", -1)) != len(frame):
                raise ValueError("A1 dynamics count differs from its manifest")
            metadata_path = path.with_name("tokens.private.metadata.json")
            if export_row.get("export_metadata_sha256") != file_sha256(metadata_path):
                raise ValueError("A1 dynamics metadata SHA-256 mismatch")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            count, channel_sum, channel_sum_squares = _validated_channel_moments(
                metadata.get("target_channel_moments"),
                expected_patients=len(frame),
                label="target",
            )
            target_count += count
            target_sum += channel_sum
            target_sum_squares += channel_sum_squares
            count, channel_sum, channel_sum_squares = _validated_channel_moments(
                metadata.get("prediction_channel_moments"),
                expected_patients=len(frame),
                label="prediction",
            )
            prediction_count += count
            prediction_sum += channel_sum
            prediction_sum_squares += channel_sum_squares
            fold_frames.append(frame)
        patients = pd.concat(fold_frames, ignore_index=True)
        if len(patients) != 808 or patients.patient_id.duplicated().any():
            raise ValueError("seed dynamics must have one OOF row per primary patient")
        patients["seed_base"] = seed
        patient_frames.append(patients)
        actual_cosine = float(patients.actual_cosine.mean())
        shuffled_cosine = float(patients.shuffled_cosine.mean())
        actual_mse = float(patients.actual_normalized_mse.mean())
        shuffled_mse = float(patients.shuffled_normalized_mse.mean())
        if target_count <= 0 or prediction_count != target_count:
            raise ValueError("seed channel-moment counts are invalid")

        def pooled_channel_std(
            count: int, channel_sum: np.ndarray, channel_sum_squares: np.ndarray
        ) -> float:
            mean = channel_sum / float(count)
            variance = channel_sum_squares / float(count) - np.square(mean)
            # Tiny negative roundoff is possible after summing float32 values;
            # a material negative variance is an authenticated-artifact error.
            if np.any(variance < -1e-10):
                raise ValueError("channel moments imply a negative variance")
            return float(np.sqrt(np.maximum(variance, 0.0)).mean())

        target_std = pooled_channel_std(target_count, target_sum, target_sum_squares)
        prediction_std = pooled_channel_std(
            prediction_count, prediction_sum, prediction_sum_squares
        )
        cosine_gain = actual_cosine - shuffled_cosine
        mse_improvement = (
            (shuffled_mse - actual_mse) / shuffled_mse
            if shuffled_mse != 0.0
            else math.nan
        )
        seed_rows.append(
            {
                "seed_base": seed,
                "finite_cell_count": len(fold_frames),
                "n_patients": len(patients),
                "actual_cosine": actual_cosine,
                "shuffled_cosine": shuffled_cosine,
                "cosine_gain": cosine_gain,
                "actual_normalized_mse": actual_mse,
                "shuffled_normalized_mse": shuffled_mse,
                "normalized_mse_relative_improvement": mse_improvement,
                "target_std": target_std,
                "prediction_std": prediction_std,
                "token_variance_ratio": (
                    prediction_std / target_std if target_std != 0.0 else math.nan
                ),
                "noncollapsed": target_std >= 0.05 and prediction_std >= 0.05,
                "materially_exceeds_shuffle": cosine_gain >= 0.05
                or mse_improvement >= 0.05,
            }
        )
        for visit in ("T1", "T2", "T3"):
            for band in ("central", "inner_local", "outer_local"):
                column = f"{visit}_{band}_normalized_mse"
                spatial_rows.append(
                    {
                        "seed_base": seed,
                        "visit": visit,
                        "band": band,
                        "normalized_mse": float(patients[column].mean()),
                        "n_patients": len(patients),
                    }
                )
    return seed_rows, spatial_rows, pd.concat(patient_frames, ignore_index=True)


def _dynamics_bootstrap(
    patients: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    summaries: list[dict[str, Any]] = []
    draws: list[pd.DataFrame] = []

    def mean_metric(_truth: np.ndarray, prediction: np.ndarray) -> float:
        return float(np.mean(prediction))

    for seed in SEEDS:
        current = patients.loc[patients.seed_base.eq(seed)].copy()
        for effect, reference_column, comparison_column, direction in (
            (
                "E6_actual_minus_shuffle_cosine",
                "shuffled_cosine",
                "actual_cosine",
                "higher",
            ),
            (
                "E6_actual_minus_shuffle_normalized_mse",
                "shuffled_normalized_mse",
                "actual_normalized_mse",
                "lower",
            ),
        ):
            reference = current.loc[:, ["patient_id", "fold", reference_column]].rename(
                columns={reference_column: "y_pred"}
            )
            comparison = current.loc[
                :, ["patient_id", "fold", comparison_column]
            ].rename(columns={comparison_column: "y_pred"})
            reference["y_true"] = 0.0
            comparison["y_true"] = 0.0
            from patch_token_wm.evaluation import paired_metric_bootstrap

            result = paired_metric_bootstrap(
                reference,
                comparison,
                metric_functions={"mean": mean_metric},
                metric_directions={"mean": direction},
                n_bootstrap=BOOTSTRAP_DRAWS,
                seed=BOOTSTRAP_SEED,
            )
            if (
                len(result.draws) != BOOTSTRAP_DRAWS
                or not result.summary["n_valid_bootstrap"].eq(BOOTSTRAP_DRAWS).all()
            ):
                raise RuntimeError(
                    "dynamics bootstrap did not produce 2,000 valid draws"
                )
            row = result.summary.iloc[0].to_dict()
            summaries.append({"effect": effect, "seed_base": seed, **row})
            draw = result.draws.copy()
            draw.insert(0, "seed_base", seed)
            draw.insert(0, "effect", effect)
            draws.append(draw)
    return summaries, draws


def _metric_lookup(rows: pd.DataFrame, **conditions: Any) -> pd.Series:
    selected = rows.copy()
    for key, value in conditions.items():
        selected = selected.loc[selected[key].eq(value)]
    if len(selected) != 1:
        raise ValueError(f"metric lookup did not select one row: {conditions}")
    return selected.iloc[0]


def _json_safe(value: Any) -> Any:
    """Replace non-finite numeric evidence before strict JSON serialization."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def _decision(
    ftv_metrics: pd.DataFrame,
    pcr_metrics_frame: pd.DataFrame,
    dynamics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    *,
    formal_cell_count: int,
) -> dict[str, Any]:
    effects_by_seed: dict[str, dict[str, float]] = {}
    for seed in SEEDS:
        response: dict[str, float] = {}
        for task in ("static", "delta"):
            a0 = _metric_lookup(
                ftv_metrics,
                seed_base=seed,
                arm="A0_LOCAL3",
                task=task,
                endpoint="macro",
            )
            a1 = _metric_lookup(
                ftv_metrics,
                seed_base=seed,
                arm="A1_PATCH3",
                task=task,
                endpoint="macro",
            )
            response[f"{task}_ftv_spearman_delta"] = float(a1.spearman - a0.spearman)
        early_effects: list[float] = []
        for timing in TIMING_LABELS[:3]:
            a0 = _metric_lookup(
                pcr_metrics_frame,
                seed_base=seed,
                arm="A0_LOCAL3",
                population="full_808",
                timing=timing,
                model="M",
            )
            a1 = _metric_lookup(
                pcr_metrics_frame,
                seed_base=seed,
                arm="A1_PATCH3",
                population="full_808",
                timing=timing,
                model="M",
            )
            early_effects.append(float(a1.auroc - a0.auroc))
        response["mri_pcr_auroc_delta"] = float(np.mean(early_effects))
        effects_by_seed[str(seed)] = response

    dynamics_by_seed: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        row = _metric_lookup(dynamics, seed_base=seed)
        dynamics_by_seed[seed] = {
            "finite_cell_count": int(row.finite_cell_count),
            "target_std": float(row.target_std),
            "prediction_std": float(row.prediction_std),
            "actual_cosine": float(row.actual_cosine),
            "shuffled_cosine": float(row.shuffled_cosine),
            "cosine_gain": float(row.cosine_gain),
            "normalized_mse_relative_improvement": float(
                row.normalized_mse_relative_improvement
            ),
        }
    # A missing formal cell is absence of evidence, not a failed scientific
    # gate.  The tested gate helper converts it to INCOMPLETE classification.
    if int(formal_cell_count) != 10:
        dynamics_by_seed = {}

    complementarity: dict[str, dict[str, Any]] = {}
    complementarity_ci_by_seed: dict[str, dict[str, float]] = {}
    for timing in TIMING_LABELS[1:3]:
        values: dict[int, float] = {}
        ci_lower: dict[str, float] = {}
        for seed in SEEDS:
            baseline = _metric_lookup(
                pcr_metrics_frame,
                seed_base=seed,
                arm="A1_PATCH3",
                population="ftv_complete_375",
                timing=timing,
                model="C+F",
            )
            joint = _metric_lookup(
                pcr_metrics_frame,
                seed_base=seed,
                arm="A1_PATCH3",
                population="ftv_complete_375",
                timing=timing,
                model="C+F+M",
            )
            values[seed] = float(joint.auroc - baseline.auroc)
            ci = bootstrap.loc[
                bootstrap.effect.eq("E5_A1_CplusFplusM_minus_CplusF")
                & bootstrap.seed_base.eq(seed)
                & bootstrap.timing.eq(timing)
                & bootstrap.metric.eq("auroc")
            ]
            if len(ci) == 1:
                ci_lower[str(seed)] = float(ci.iloc[0].ci_lower)
        complementarity[timing] = {
            "seed_effects": values,
            "bootstrap_ci_lower": (
                min(ci_lower.values())
                if len(ci_lower) == 2
                and all(math.isfinite(value) for value in ci_lower.values())
                else None
            ),
        }
        complementarity_ci_by_seed[timing] = ci_lower

    evaluated = evaluate_all_gates(
        dynamics_by_seed=dynamics_by_seed,
        effects_by_seed=effects_by_seed,
        complementarity_by_timing=complementarity,
        expected_seeds=SEEDS,
    )
    gate_names = {
        "A": "A_PATCH_DYNAMICS_VALID",
        "B": "B_RESPONSE_PRESERVATION",
        "C": "C_PATCH_STATE_ADDS_INFORMATION",
        "D": "D_PATCH_STATE_COMPLEMENTARITY_SUPPORTED",
    }
    gates = {gate_names[letter]: dict(evaluated["gates"][letter]) for letter in "ABCD"}
    gates["D_PATCH_STATE_COMPLEMENTARITY_SUPPORTED"][
        "bootstrap_ci_lower_by_seed"
    ] = complementarity_ci_by_seed
    final = evaluated["final"]
    return {
        "schema_version": 1,
        "status": final["status"],
        "classification": final["label"],
        "classification_reason": final["reason"],
        "classification_evidence": final["evidence"],
        "gates": _json_safe(gates),
        "effects_by_seed": _json_safe(effects_by_seed),
        "scientific_boundary": {
            "world_model_training_pcr_free": True,
            "causal_treatment_claim": False,
            "assigned_regimen_conditioning_only": True,
            "t3_label": "late/pre-surgery",
        },
    }


def _figures(ftv: pd.DataFrame, pcr: pd.DataFrame, spatial: pd.DataFrame) -> None:
    figure_root = EXPERIMENT_ROOT / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    macro = ftv.loc[ftv.endpoint.eq("macro")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for axis, task in zip(axes, ("static", "delta"), strict=True):
        current = macro.loc[macro.task.eq(task)]
        for arm, offset, color in (
            ("A0_LOCAL3", -0.12, "#4c78a8"),
            ("A1_PATCH3", 0.12, "#f58518"),
        ):
            values = [
                float(_metric_lookup(current, seed_base=seed, arm=arm).spearman)
                for seed in SEEDS
            ]
            axis.bar(np.arange(2) + offset, values, width=0.24, label=arm, color=color)
        axis.set_xticks(range(2), [str(seed) for seed in SEEDS])
        axis.set_title(f"{task} FTV macro")
        axis.set_xlabel("training seed")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Spearman")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_root / "01_ftv_macro_spearman.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, seed in zip(axes, SEEDS, strict=True):
        for arm, marker in (("A0_LOCAL3", "o"), ("A1_PATCH3", "s")):
            values = [
                float(
                    _metric_lookup(
                        pcr,
                        seed_base=seed,
                        arm=arm,
                        population="full_808",
                        timing=timing,
                        model="M",
                    ).auroc
                )
                for timing in TIMING_LABELS
            ]
            axis.plot(range(4), values, marker=marker, label=arm)
        axis.set_xticks(range(4), ("T0", "T0–T1", "T0–T2", "T0–T3"), rotation=20)
        axis.set_title(f"MRI-only pCR, seed {seed}")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("AUROC")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_root / "02_mri_only_pcr.png", dpi=180)
    plt.close(fig)

    aggregate = (
        spatial.groupby(["visit", "band"], sort=False)
        .normalized_mse.mean()
        .reset_index()
    )
    fig, axis = plt.subplots(figsize=(8, 4))
    x = np.arange(3)
    for band, offset in (
        ("central", -0.22),
        ("inner_local", 0.0),
        ("outer_local", 0.22),
    ):
        values = [
            float(
                aggregate.loc[
                    aggregate.visit.eq(visit) & aggregate.band.eq(band),
                    "normalized_mse",
                ].iloc[0]
            )
            for visit in ("T1", "T2", "T3")
        ]
        axis.bar(x + offset, values, width=0.22, label=band)
    axis.set_xticks(x, ("T1", "T2", "T3 late/pre-surgery"))
    axis.set_ylabel("normalized masked-token MSE")
    axis.set_title("Outcome-blind spatial error bands")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_root / "03_spatial_token_error.png", dpi=180)
    plt.close(fig)


def _report(
    decision: dict[str, Any],
    ftv: pd.DataFrame,
    pcr: pd.DataFrame,
    dynamics: pd.DataFrame,
    spatial: pd.DataFrame,
) -> str:
    classification = decision["classification"]
    gates = decision["gates"]
    lines = [
        "# Patch-token 治疗条件世界模型：最终报告",
        "",
        f"**最终分类：`{classification}`。**",
        "",
        "本实验完成了 2 个独立训练种子 × 5 个外层折的 A1 PATCH3 矩阵，并与已确认的 A0 LOCAL3 冻结基线做成对比较。世界模型训练、掩码、模型选择和 PCA 均未读取 pCR；pCR 只在十个检查点和无标签表征变换全部冻结后进入下游探针。治疗变量仅表示 assigned-regimen conditioning，不构成因果治疗效应。",
        "",
        "## 结论摘要",
        "",
    ]
    for seed in SEEDS:
        row = _metric_lookup(dynamics, seed_base=seed)
        lines.append(
            f"- 种子 {seed}：实际时间余弦 {row.actual_cosine:.4f}，循环乱序 {row.shuffled_cosine:.4f}，差值 {row.cosine_gain:+.4f}；归一化 MSE 相对改善 {row.normalized_mse_relative_improvement:+.1%}；目标/预测 token SD 为 {row.target_std:.4f}/{row.prediction_std:.4f}。"
        )
    lines.extend(["", "## 十个问题的直接回答", ""])
    stable = gates["A_PATCH_DYNAMICS_VALID"]["status"] == "PASS"
    lines.append(
        f"1. **Patch-token JEPA 是否稳定训练？** {'是' if stable else '未达到预注册稳定性门槛'}；Gate A = {gates['A_PATCH_DYNAMICS_VALID']['status']}。"
    )
    for number, task, label in ((2, "static", "静态 FTV"), (3, "delta", "ΔFTV")):
        details = []
        for seed in SEEDS:
            a0 = _metric_lookup(
                ftv, seed_base=seed, arm="A0_LOCAL3", task=task, endpoint="macro"
            )
            a1 = _metric_lookup(
                ftv, seed_base=seed, arm="A1_PATCH3", task=task, endpoint="macro"
            )
            details.append(
                f"{seed}: {a0.spearman:.3f}→{a1.spearman:.3f} (Δ {a1.spearman-a0.spearman:+.3f})"
            )
        lines.append(
            f"{number}. **是否优于 LOCAL3 的{label}？** " + "；".join(details) + "。"
        )
    early: list[str] = []
    for seed in SEEDS:
        a0_values = []
        a1_values = []
        for timing in TIMING_LABELS[:3]:
            a0_values.append(
                float(
                    _metric_lookup(
                        pcr,
                        seed_base=seed,
                        arm="A0_LOCAL3",
                        population="full_808",
                        timing=timing,
                        model="M",
                    ).auroc
                )
            )
            a1_values.append(
                float(
                    _metric_lookup(
                        pcr,
                        seed_base=seed,
                        arm="A1_PATCH3",
                        population="full_808",
                        timing=timing,
                        model="M",
                    ).auroc
                )
            )
        early.append(
            f"{seed}: {np.mean(a0_values):.3f}→{np.mean(a1_values):.3f} (Δ {np.mean(a1_values)-np.mean(a0_values):+.3f})"
        )
    lines.append(
        "4. **是否改善 MRI-only pCR？** 早期三前缀 AUROC 宏平均："
        + "；".join(early)
        + "。"
    )
    for number, population, base_model, joint_model, wording in (
        (5, "full_808", "C", "C+M", "临床 C"),
        (6, "ftv_complete_375", "C+F", "C+F+M", "临床+因果前缀 FTV"),
    ):
        details = []
        for seed in SEEDS:
            timing_values = []
            for timing in TIMING_LABELS[:3]:
                base = _metric_lookup(
                    pcr,
                    seed_base=seed,
                    arm="A1_PATCH3",
                    population=population,
                    timing=timing,
                    model=base_model,
                )
                joint = _metric_lookup(
                    pcr,
                    seed_base=seed,
                    arm="A1_PATCH3",
                    population=population,
                    timing=timing,
                    model=joint_model,
                )
                timing_values.append(float(joint.auroc - base.auroc))
            details.append(f"{seed} 早期均值 ΔAUROC {np.mean(timing_values):+.3f}")
        lines.append(
            f"{number}. **是否增加超越{wording}的信息？** " + "；".join(details) + "。"
        )
    band_mean = (
        spatial.groupby("band").normalized_mse.mean().sort_values(ascending=False)
    )
    lines.append(
        "7. **未来 token 误差集中在哪里？** 按固定坐标带跨种子/访视均值，"
        + "，".join(f"{band}={value:.4f}" for band, value in band_mean.items())
        + f"；最高为 `{band_mean.index[0]}`。这些带不是病灶或瘤周区域。"
    )
    gate_c = gates["C_PATCH_STATE_ADDS_INFORMATION"]
    lines.append(
        f"8. **空间 token 是否找回 pooling 丢失的信息？** Gate C = {gate_c['status']}；仅在同一端点、双种子方向一致且平均增益 ≥0.03 时才回答“是”。"
    )
    lines.append(
        f"9. **增益是 response-only 还是 phenotype-complementary？** 最终分类为 `{classification}`；Gate D = {gates['D_PATCH_STATE_COMPLEMENTARITY_SUPPORTED']['status']}。"
    )
    replace = classification == "PATCH_WORLD_MODEL_BREAKTHROUGH"
    lines.append(
        f"10. **是否应替换 pooled LOCAL state？** {'应进入多种子确认，暂不直接生产替换' if replace else '不应；按预注册规则保留 pooled LOCAL3'}。"
    )
    lines.extend(
        [
            "",
            "## 门控",
            "",
            f"- Gate A `PATCH_DYNAMICS_VALID`: {gates['A_PATCH_DYNAMICS_VALID']['status']}",
            f"- Gate B response preservation: {gates['B_RESPONSE_PRESERVATION']['status']}",
            f"- Gate C `PATCH_STATE_ADDS_INFORMATION`: {gates['C_PATCH_STATE_ADDS_INFORMATION']['status']}",
            f"- Gate D `PATCH_STATE_COMPLEMENTARITY_SUPPORTED`: {gates['D_PATCH_STATE_COMPLEMENTARITY_SUPPORTED']['status']}",
            "",
            "## 解释边界",
            "",
            "- T3 始终是 late/pre-surgery；早期结论仅限 T0、T0–T1、T0–T2。",
            "- 808 人均为 complete-4-visit 选择队列；FTV 分析的 375 人又是测量完整子集，不能把两个人群的绝对指标作增量比较。",
            "- 500 个 token 来自固定 64-mm LOCAL 正重叠支持，边界权重用于精确均值；单 token 理论感受野约 42.3×42.3×94.0 mm，并非细粒度独立病理块。",
            "- C1B-H 使用 T0 定位中心和 header-based 纵向策略；残余运动可表现为 token 预测误差。固定中心/内/外带仅为描述性坐标带。",
            "- `delta_t=1` 是名义相邻访视间隔，不是实测扫描日差。assigned-treatment-conditioned longitudinal latent modeling 不等于因果治疗效应。",
            "- 折内预测先合并为 OOF 后评分；bootstrap 在外层折内按患者重采样 2,000 次。折、访视和端点均不作为独立重复，训练种子才是独立重复。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> dict[str, Any]:
    lock = verify()
    authenticated = _validate_formal_inputs(str(lock["lock_sha256"]))
    if len(authenticated) != 10:
        raise RuntimeError("formal artifact authentication did not cover ten cells")
    decision_path = EXPERIMENT_ROOT / "metrics" / "decision.json"
    report_path = EXPERIMENT_ROOT / "reports" / "final_report.md"
    if decision_path.exists():
        raise FileExistsError("refusing to overwrite completed formal decision")

    authorization = require_stage_a_go(DEFAULT_STAGE_A_SENTINEL)
    paths = StageBDataPaths.load(DEFAULT_DATA_CONTRACT, DEFAULT_DATA_CONTRACT_SHA256)
    data = load_stage_b_data(paths, authorization, verify_cache_files=False)
    (
        frozen_states,
        pca_rows,
        ftv_prediction_rows,
        ftv_selection_rows,
    ) = _prepare_label_free_states(
        data, preregistration_lock_sha256=str(lock["lock_sha256"])
    )

    # This is the first point at which the current pCR column may be opened:
    # all ten checkpoints, exports, and outer-train PCA transforms are frozen.
    clinical = _clinical_table()
    pcr_prediction_rows: list[dict[str, Any]] = []
    pcr_selection_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for fold in FOLDS:
            for arm in ARMS:
                patient_ids, split, state = frozen_states[(seed, fold, arm)]
                pcr_rows, pcr_select = _run_pcr_cell(
                    seed=seed,
                    fold=fold,
                    arm=arm,
                    patient_ids=patient_ids,
                    split=split,
                    state=state,
                    data=data,
                    clinical=clinical,
                )
                pcr_prediction_rows.extend(pcr_rows)
                pcr_selection_rows.extend(pcr_select)
    ftv_predictions = pd.DataFrame(ftv_prediction_rows)
    pcr_predictions = pd.DataFrame(pcr_prediction_rows)
    ftv_metrics = pd.DataFrame(_pool_ftv_metrics(ftv_predictions))
    pcr_metrics_frame = pd.DataFrame(_pool_pcr_metrics(pcr_predictions))

    bootstrap_rows: list[dict[str, Any]] = []
    private_draws: list[pd.DataFrame] = []
    for seed in SEEDS:
        reference = ftv_predictions.loc[
            ftv_predictions.seed_base.eq(seed) & ftv_predictions.arm.eq("A0_LOCAL3")
        ]
        comparison = ftv_predictions.loc[
            ftv_predictions.seed_base.eq(seed) & ftv_predictions.arm.eq("A1_PATCH3")
        ]
        for task in ("static", "delta"):
            summary, draws = _bootstrap_ftv_macro(
                reference, comparison, seed=seed, task=task
            )
            bootstrap_rows.append(summary)
            private_draws.append(draws)
    pcr_bootstrap_rows, pcr_draws = _pcr_bootstraps(pcr_predictions)
    bootstrap_rows.extend(pcr_bootstrap_rows)
    private_draws.extend(pcr_draws)
    dynamics_rows, spatial_rows, dynamics_patients = _dynamics(data, authenticated)
    dynamics_bootstrap_rows, dynamics_draws = _dynamics_bootstrap(dynamics_patients)
    bootstrap_rows.extend(dynamics_bootstrap_rows)
    private_draws.extend(dynamics_draws)

    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    dynamics_frame = pd.DataFrame(dynamics_rows)
    spatial_frame = pd.DataFrame(spatial_rows)
    decision = _decision(
        ftv_metrics,
        pcr_metrics_frame,
        dynamics_frame,
        bootstrap_frame,
        formal_cell_count=len(authenticated),
    )
    decision["preregistration_lock_sha256"] = lock["lock_sha256"]
    decision["bootstrap"] = {
        "draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "unit": "patient_within_outer_fold",
        "folds_are_replicates": False,
    }

    prediction_root = EXPERIMENT_ROOT / "predictions"
    _atomic_csv(prediction_root / "ftv_predictions.private.csv", ftv_prediction_rows)
    _atomic_csv(prediction_root / "pcr_predictions.private.csv", pcr_prediction_rows)
    _atomic_csv(
        prediction_root / "bootstrap_draws.private.csv",
        pd.concat(private_draws, ignore_index=True).to_dict("records"),
    )
    _atomic_csv(
        EXPERIMENT_ROOT / "metrics" / "table_ftv.csv", ftv_metrics.to_dict("records")
    )
    _atomic_csv(
        EXPERIMENT_ROOT / "metrics" / "table_pcr.csv",
        pcr_metrics_frame.to_dict("records"),
    )
    _atomic_csv(EXPERIMENT_ROOT / "metrics" / "table_dynamics.csv", dynamics_rows)
    _atomic_csv(EXPERIMENT_ROOT / "metrics" / "table_spatial_errors.csv", spatial_rows)
    _atomic_csv(EXPERIMENT_ROOT / "metrics" / "table_bootstrap.csv", bootstrap_rows)
    _atomic_csv(
        EXPERIMENT_ROOT / "metrics" / "probe_selections.csv",
        ftv_selection_rows + pcr_selection_rows,
    )
    _atomic_csv(EXPERIMENT_ROOT / "metrics" / "pca_contract.csv", pca_rows)
    _figures(ftv_metrics, pcr_metrics_frame, spatial_frame)
    _atomic_text(
        report_path,
        _report(
            decision, ftv_metrics, pcr_metrics_frame, dynamics_frame, spatial_frame
        ),
    )
    # The decision is the atomic terminal sentinel and is published only after
    # every table, figure, private draw, and report completed successfully.
    _atomic_json(decision_path, decision)
    print(json.dumps(decision, indent=2, sort_keys=True, allow_nan=False))
    return decision


if __name__ == "__main__":
    main()

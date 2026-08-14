"""Strict, post-freeze data contracts for the Goal-F evaluation.

This module is deliberately separate from representation training.  It is the
first code allowed to read pCR, and every representation asset it accepts must
carry an explicit outcome-free export assertion.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
CONFIG_PATH = EXPERIMENT_ROOT / "configs" / "evaluation.json"
SEEDS = (2026, 3026)
FOLDS = (0, 1, 2, 3, 4)
ARMS = ("F1", "F2")
SPLITS = ("train", "val", "test")
EXPECTED_TIMINGS = {"T0": [0], "T0-T1": [0, 1], "T0-T2": [0, 1, 2]}
EXPECTED_STATES = ["z_R", "z_P", "full"]
EXPECTED_PROFILE_TARGETS = ["HR", "HER2", "subtype"]
FACTORIZED_METADATA_KEYS = {
    "schema_version",
    "experiment",
    "arm",
    "seed_base",
    "fold",
    "effective_seed",
    "selected_epoch",
    "selection_mode",
    "selection_experiment_pass",
    "feature_sha256",
    "checkpoint_sha256",
    "selection_sha256",
    "preregistration_lock_sha256",
    "preregistration_payload_sha256",
    "patient_count",
    "patient_order_sha256",
    "train_patient_sha256",
    "validation_patient_sha256",
    "test_patient_sha256",
    "state_shapes",
    "augmentation",
    "PCR_LABEL_ACCESS",
    "pcr_labels_used",
    "representation_frozen_before_export",
    "export_completed",
}


class EvaluationContractError(ValueError):
    """Raised when a frozen input or feature asset drifts."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ordered_patient_sha256(patient_ids: Sequence[Any]) -> str:
    return hashlib.sha256(
        "\n".join(str(value) for value in patient_ids).encode("utf-8")
    ).hexdigest()


def _exact_positive_grid(value: Any, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise EvaluationContractError(f"{label} must be a nonempty JSON list")
    try:
        grid = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise EvaluationContractError(f"{label} must contain numbers") from error
    if (
        not np.isfinite(grid).all()
        or any(item <= 0 for item in grid)
        or tuple(sorted(set(grid))) != grid
    ):
        raise EvaluationContractError(
            f"{label} must be finite, positive, unique, and strictly increasing"
        )
    return grid


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _locked_file(value: str | Path, digest: str, label: str) -> Path:
    path = _resolve(value)
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    if not isinstance(digest, str) or len(digest) != 64:
        raise EvaluationContractError(f"{label} requires a SHA-256 lock")
    observed = file_sha256(path)
    if observed != digest:
        raise EvaluationContractError(
            f"{label} SHA-256 drifted: expected {digest}, observed {observed}"
        )
    return path


def load_evaluation_config(
    path: str | Path = CONFIG_PATH, *, require_representation_lock: bool = True
) -> dict[str, Any]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise EvaluationContractError("evaluation schema version drifted")
    if payload.get("experiment") != "clinical_residual_phenotype_state":
        raise EvaluationContractError("evaluation experiment identity drifted")
    if payload.get("phase") != "post_export_evaluation":
        raise EvaluationContractError("evaluation phase must be post_export_evaluation")
    if payload.get("representation_must_be_frozen") is not True:
        raise EvaluationContractError("representation freeze assertion is required")
    if int(payload.get("required_primary_feature_cells", -1)) != 20:
        raise EvaluationContractError("exactly 20 factorized feature cells are required")
    if payload.get("timings") != EXPECTED_TIMINGS:
        raise EvaluationContractError("evaluation timings must be exactly T0/T0-T1/T0-T2")
    if payload.get("states") != EXPECTED_STATES:
        raise EvaluationContractError("evaluation states must be exactly z_R/z_P/full")
    if payload.get("profile_targets") != EXPECTED_PROFILE_TARGETS:
        raise EvaluationContractError("profile targets must be exactly HR/HER2/subtype")
    if payload.get("clinical_features") != [
        "label_hr",
        "label_her2",
        "label_mp",
        "age_at_screening",
        "arm",
    ]:
        raise EvaluationContractError("clinical feature contract drifted")
    _exact_positive_grid(payload.get("ridge_alphas"), "ridge_alphas")
    _exact_positive_grid(payload.get("logistic_c_grid"), "logistic_c_grid")
    bootstrap = payload.get("bootstrap", {})
    if int(bootstrap.get("replicates", 0)) != 2000:
        raise EvaluationContractError("paired patient bootstrap requires exactly 2,000 draws")
    if (
        bootstrap.get("unit") != "patient"
        or bootstrap.get("stratify_by") != "outer_fold_x_outcome"
        or float(bootstrap.get("confidence_level", 0.0)) != 0.95
        or isinstance(bootstrap.get("random_seed"), bool)
        or not isinstance(bootstrap.get("random_seed"), int)
    ):
        raise EvaluationContractError("bootstrap unit/stratification contract drifted")
    expected_diagnostics = {
        "effective_rank_floor": 10.0,
        "phenotype_mean_std_floor": 0.05,
        "augmentation_cosine_floor": 0.5,
        "nearest_neighbors_k": 10,
        "cca_components": 10,
        "future_prediction_baseline": (
            "ema_target_projected_persistence_T1_to_T2_and_T2_to_T3"
        ),
        "future_prediction_must_beat_baseline": True,
    }
    if payload.get("diagnostics") != expected_diagnostics:
        raise EvaluationContractError("representation diagnostic/gate safeguard drifted")
    expected_gates = {
        "response_static_ftv_spearman_degradation_floor": -0.03,
        "response_delta_systematic_degradation_forbidden": True,
        "clinical_redundancy_targets": ["HR", "HER2"],
        "clinical_redundancy_primary_view": "static_T0_T1_T2_macro",
        "clinical_redundancy_decodability": (
            "0.5_plus_absolute_AUROC_minus_0.5"
        ),
        "phenotype_complementarity_timings": ["T0-T1", "T0-T2"],
        "phenotype_complementarity_each_seed_strictly_gt": 0.0,
        "phenotype_complementarity_strong_mean": 0.03,
    }
    if payload.get("gates") != expected_gates:
        raise EvaluationContractError("primary Gate A-D contract drifted")
    selection = payload.get("model_selection")
    expected_selection = {
        "outer_train_fit": True,
        "outer_validation_hyperparameter_selection": True,
        "outer_test_single_prediction": True,
        "logistic_penalty": "l2",
        "binary_logistic_solver": "liblinear",
        "multiclass_logistic_solver": "lbfgs",
        "multiclass_training": "balanced_multinomial",
        "logistic_max_iter": 10000,
        "tie_break": "smaller_regularization_parameter",
    }
    if selection != expected_selection:
        raise EvaluationContractError("fold-safe model/solver selection contract drifted")
    if payload.get("optional_f3_linear_residualization") is not False:
        raise EvaluationContractError("primary immutable boundary requires optional F3 disabled")
    labels = payload.get("labels")
    if (
        not isinstance(labels, Mapping)
        or set(labels) != {"ispy2_path", "ispy2_sha256", "pcr_column"}
        or labels.get("pcr_column") != "label_pcr"
        or not isinstance(labels.get("ispy2_path"), str)
        or not isinstance(labels.get("ispy2_sha256"), str)
        or len(labels["ispy2_sha256"]) != 64
    ):
        raise EvaluationContractError("post-lock pCR label source contract drifted")
    inputs = payload.get("frozen_inputs")
    if not isinstance(inputs, Mapping):
        raise EvaluationContractError("evaluation frozen_inputs are missing")
    required_inputs = {
        "representation_preregistration_lock",
        "representation_preregistration_lock_sha256",
        "fold_manifest_path",
        "fold_manifest_sha256",
        "stage_a_sentinel_path",
        "stage_a_sentinel_sha256",
        "stage_b_data_contract_path",
        "stage_b_data_contract_sha256",
        "ftv_transition_path",
        "ftv_transition_sha256",
        "f0_feature_root",
        "f0_preregistration_lock",
        "f0_preregistration_lock_sha256",
        "factorized_feature_root",
        "factorized_checkpoint_root",
        "factorized_export_status_path",
        "expected_primary_patient_count",
        "expected_ftv_patient_count",
    }
    if set(inputs) != required_inputs:
        raise EvaluationContractError("frozen input inventory keys drifted")
    digest_pairs = (
        ("representation_preregistration_lock", "representation_preregistration_lock_sha256"),
        ("fold_manifest_path", "fold_manifest_sha256"),
        ("stage_a_sentinel_path", "stage_a_sentinel_sha256"),
        ("stage_b_data_contract_path", "stage_b_data_contract_sha256"),
        ("ftv_transition_path", "ftv_transition_sha256"),
        ("f0_preregistration_lock", "f0_preregistration_lock_sha256"),
    )
    for path_key, digest_key in digest_pairs:
        digest = inputs.get(digest_key)
        if not isinstance(digest, str) or len(digest) != 64:
            raise EvaluationContractError(f"{digest_key} must be a SHA-256 lock")
        if not isinstance(inputs.get(path_key), str) or not str(inputs[path_key]):
            raise EvaluationContractError(f"{path_key} must be a nonempty path")
    if int(inputs["expected_primary_patient_count"]) != 808:
        raise EvaluationContractError("primary evaluation cohort must contain exactly 808 patients")
    if int(inputs["expected_ftv_patient_count"]) != 375:
        raise EvaluationContractError("FTV-complete cohort must contain exactly 375 patients")
    if require_representation_lock:
        load_representation_lock(payload)
    return payload


def load_representation_lock(config: Mapping[str, Any]) -> dict[str, Any]:
    """Load and authenticate the pCR-free representation lock and its config."""

    frozen = config["frozen_inputs"]
    path = _locked_file(
        frozen["representation_preregistration_lock"],
        frozen["representation_preregistration_lock_sha256"],
        "representation preregistration lock",
    )
    expected_path = (EXPERIMENT_ROOT / "PREREGISTRATION_LOCK.json").resolve()
    if path != expected_path:
        raise EvaluationContractError("representation lock must be the canonical Goal-F lock")
    locked = json.loads(path.read_text(encoding="utf-8"))
    if (
        locked.get("schema_version") != 1
        or locked.get("status") != "PASS"
        or locked.get("experiment") != "clinical_residual_phenotype_state"
        or locked.get("phase") != "representation_training"
        or locked.get("PCR_LABEL_ACCESS") != "FORBIDDEN"
        or locked.get("pcr_used_for_hyperparameter_selection") is not False
        or locked.get("pcr_used_for_representation_training") is not False
        or locked.get("pcr_used_for_checkpoint_selection") is not False
    ):
        raise EvaluationContractError(
            "representation preregistration lock is not a frozen pCR-free PASS"
        )
    unsigned = dict(locked)
    claimed = unsigned.pop("lock_sha256", None)
    if claimed != canonical_sha256(unsigned):
        raise EvaluationContractError("representation preregistration payload digest drifted")
    representation_config = EXPERIMENT_ROOT / str(locked.get("config_path", ""))
    if representation_config.resolve() != (EXPERIMENT_ROOT / "configs/representation.json").resolve():
        raise EvaluationContractError("representation config path is not canonical")
    if file_sha256(representation_config) != locked.get("config_sha256"):
        raise EvaluationContractError("representation config hash differs from its lock")
    return locked


def load_fold_assignments(config: Mapping[str, Any]) -> pd.DataFrame:
    """Read only label-free outer-fold assignments from the locked manifest.

    ``usecols`` is intentionally literal: the source manifest also contains an
    outcome column, which must not be parsed while the evaluation boundary is
    being constructed or verified.
    """

    frozen = config["frozen_inputs"]
    path = _locked_file(
        frozen["fold_manifest_path"], frozen["fold_manifest_sha256"], "fold manifest"
    )
    columns = ["patient_id", "fold", "split"]
    frame = pd.read_csv(path, usecols=columns)
    if list(frame.columns) != columns:
        raise EvaluationContractError("label-free fold assignment columns drifted")
    expected_n = int(frozen["expected_primary_patient_count"])
    if len(frame) != expected_n * len(FOLDS):
        raise EvaluationContractError("fold manifest row count drifted")
    frame = frame.copy()
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    frame["split"] = frame["split"].astype(str)
    if set(frame["fold"]) != set(FOLDS) or set(frame["split"]) != set(SPLITS):
        raise EvaluationContractError("fold/split vocabulary drifted")
    if frame.duplicated(["patient_id", "fold"]).any():
        raise EvaluationContractError("patient repeats within an outer fold")
    grouped = frame.groupby("patient_id", sort=False)
    if frame["patient_id"].nunique() != expected_n or not grouped.size().eq(5).all():
        raise EvaluationContractError("fold manifest patient coverage drifted")
    if not frame["split"].eq("test").groupby(frame["patient_id"]).sum().eq(1).all():
        raise EvaluationContractError("each patient must be outer-test exactly once")
    return frame.sort_values(["fold", "patient_id"], kind="stable").reset_index(drop=True)


def load_fold_manifest(config: Mapping[str, Any]) -> pd.DataFrame:
    """Compatibility name for the now strictly label-free assignment loader."""

    return load_fold_assignments(config)


def load_stage_b_ftv_records(
    config: Mapping[str, Any], assignments: pd.DataFrame
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Load the exact sealed Stage-B FTVRecord mapping for matched probes.

    The adapter reads split assignments and FTV/observability inputs but never
    parses an outcome label.  All source bytes are transitively SHA-pinned by
    the Stage-B contract and the representation preregistration lock.
    """

    frozen = config["frozen_inputs"]
    sentinel = _locked_file(
        frozen["stage_a_sentinel_path"],
        frozen["stage_a_sentinel_sha256"],
        "Stage-A authorization",
    )
    contract = _locked_file(
        frozen["stage_b_data_contract_path"],
        frozen["stage_b_data_contract_sha256"],
        "Stage-B data contract",
    )
    expected_sentinel = (
        REPO_ROOT
        / "additional_experiments/c1b_overlap_eligibility_ftv_stageb/STAGE_A_GO.json"
    ).resolve()
    expected_contract = (
        REPO_ROOT
        / "additional_experiments/c1b_overlap_eligibility_ftv_stageb/manifests/"
        "stage_b_data_contract.private.json"
    ).resolve()
    if sentinel != expected_sentinel or contract != expected_contract:
        raise EvaluationContractError("Stage-B response input path is not canonical")
    from .stageb import StageBDataPaths, load_stage_b_data, require_stage_a_go

    authorization = require_stage_a_go(sentinel)
    paths = StageBDataPaths.load(contract, frozen["stage_b_data_contract_sha256"])
    if (
        paths.fold_manifest.resolve()
        != _resolve(frozen["fold_manifest_path"])
        or paths.fold_manifest_sha256 != frozen["fold_manifest_sha256"]
        or paths.ftv_transition_table.resolve()
        != _resolve(frozen["ftv_transition_path"])
        or paths.ftv_transition_table_sha256 != frozen["ftv_transition_sha256"]
    ):
        raise EvaluationContractError("Stage-B response sources differ from evaluation locks")
    data = load_stage_b_data(paths, authorization, verify_cache_files=False)
    observed = data.folds[["patient_id", "fold", "split"]].copy()
    observed["patient_id"] = observed["patient_id"].astype(str)
    observed["fold"] = pd.to_numeric(observed["fold"], errors="raise").astype(int)
    observed["split"] = observed["split"].astype(str)
    observed = observed.sort_values(["fold", "patient_id"], kind="stable").reset_index(drop=True)
    if not observed.equals(assignments.reset_index(drop=True)):
        raise EvaluationContractError("Stage-B/frozen fold assignment equality failed")
    expected_n = int(frozen["expected_ftv_patient_count"])
    if len(data.ftv) != expected_n or not set(data.ftv).issubset(set(assignments.patient_id)):
        raise EvaluationContractError("Stage-B FTVRecord population drifted")
    if data.provenance.get("adapter") != "exact_usecols_split_ftv_observability_only":
        raise EvaluationContractError("Stage-B response adapter contract drifted")
    return data.ftv, data.provenance


def load_clinical(config: Mapping[str, Any], folds: pd.DataFrame) -> pd.DataFrame:
    # This local import avoids a module cycle while making the data boundary
    # self-enforcing for every caller, not just the top-level evaluation CLI.
    from .evaluation_lock import verify as verify_evaluation_lock

    verify_evaluation_lock()
    labels = config["labels"]
    path = _locked_file(labels["ispy2_path"], labels["ispy2_sha256"], "clinical labels")
    frame = pd.read_csv(path)
    required = {
        "patient_id",
        "label_pcr",
        "label_hr",
        "label_her2",
        "label_mp",
        "age_at_screening",
        "arm",
        "hr_her2_subtype",
    }
    if not required.issubset(frame.columns):
        raise EvaluationContractError("clinical table lacks required columns")
    frame = frame.loc[:, sorted(required)].copy()
    frame["patient_id"] = frame["patient_id"].astype(str)
    if frame["patient_id"].duplicated().any():
        raise EvaluationContractError("clinical patient IDs are duplicated")
    expected_ids = set(folds["patient_id"])
    if set(frame["patient_id"]) != expected_ids:
        raise EvaluationContractError("clinical/fold cohort equality failed")
    for column in ("label_pcr", "label_hr", "label_her2", "label_mp"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
        if not set(frame[column]).issubset({0, 1}):
            raise EvaluationContractError(f"{column} is not binary")
    frame["age_at_screening"] = pd.to_numeric(
        frame["age_at_screening"], errors="coerce"
    ).astype(float)
    if np.isinf(frame["age_at_screening"].to_numpy()).any():
        raise EvaluationContractError("clinical age contains infinity")
    frame["arm"] = frame["arm"].astype(str)
    frame["hr_her2_subtype"] = frame["hr_her2_subtype"].astype(str)
    expected_subtype = np.select(
        [
            frame.label_hr.eq(1) & frame.label_her2.eq(0),
            frame.label_hr.eq(0) & frame.label_her2.eq(0),
            frame.label_hr.eq(1) & frame.label_her2.eq(1),
            frame.label_hr.eq(0) & frame.label_her2.eq(1),
        ],
        ["HR+/HER2-", "HR-/HER2-", "HR+/HER2+", "HR-/HER2+"],
        default="INVALID",
    )
    if not np.array_equal(frame["hr_her2_subtype"].to_numpy(), expected_subtype):
        raise EvaluationContractError("clinical subtype disagrees with HR/HER2 labels")
    return frame.sort_values("patient_id", kind="stable").reset_index(drop=True)


def load_ftv_wide(config: Mapping[str, Any], patient_ids: set[str]) -> pd.DataFrame:
    frozen = config["frozen_inputs"]
    path = _locked_file(
        frozen["ftv_transition_path"], frozen["ftv_transition_sha256"], "FTV table"
    )
    frame = pd.read_csv(path)
    required = {
        "patient_id",
        "transition",
        "start_visit",
        "end_visit",
        "ftv_start",
        "ftv_end",
        "ftv_absolute_change",
        "ftv_valid",
    }
    if not required.issubset(frame.columns):
        raise EvaluationContractError("FTV transition table lacks required columns")
    frame = frame.copy()
    frame["patient_id"] = frame["patient_id"].astype(str)
    if not set(frame["patient_id"]).issubset(patient_ids):
        raise EvaluationContractError("FTV table contains patients outside primary cohort")
    expected_n = int(frozen["expected_ftv_patient_count"])
    if frame["patient_id"].nunique() != expected_n or len(frame) != expected_n * 3:
        raise EvaluationContractError("FTV cohort/transition count drifted")
    valid = frame["ftv_valid"].astype(str).str.lower().isin(("true", "1"))
    if not valid.all():
        raise EvaluationContractError("invalid FTV rows are forbidden")
    for column in ("ftv_start", "ftv_end", "ftv_absolute_change"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
        if not np.isfinite(frame[column]).all():
            raise EvaluationContractError("FTV contains non-finite values")
    if not np.allclose(frame["ftv_end"] - frame["ftv_start"], frame["ftv_absolute_change"]):
        raise EvaluationContractError("FTV transition delta disagrees with endpoints")
    rows: list[dict[str, Any]] = []
    expected = ("T0→T1", "T1→T2", "T2→T3")
    for patient_id, group in frame.groupby("patient_id", sort=False):
        by = group.set_index("transition", verify_integrity=True)
        if set(by.index) != set(expected):
            raise EvaluationContractError("FTV transition vocabulary drifted")
        a, b, c = (by.loc[value] for value in expected)
        if not np.isclose(a.ftv_end, b.ftv_start) or not np.isclose(b.ftv_end, c.ftv_start):
            raise EvaluationContractError("FTV interior visit is inconsistent")
        rows.append(
            {
                "patient_id": patient_id,
                "FTV_T0": float(a.ftv_start),
                "FTV_T1": float(a.ftv_end),
                "FTV_T2": float(b.ftv_end),
                "FTV_T3": float(c.ftv_end),
            }
        )
    return pd.DataFrame(rows).sort_values("patient_id", kind="stable").reset_index(drop=True)


def _validate_assignments(
    patient_id: np.ndarray, split: np.ndarray, folds: pd.DataFrame, fold: int
) -> None:
    expected = folds.loc[folds["fold"].eq(fold), ["patient_id", "split"]]
    expected_map = dict(zip(expected.patient_id.astype(str), expected.split.astype(str), strict=True))
    observed_map = dict(zip(patient_id.astype(str), split.astype(str), strict=True))
    if observed_map != expected_map:
        raise EvaluationContractError("feature split assignment differs from frozen fold")


@dataclass(frozen=True)
class FactorizedAsset:
    path: Path
    metadata_path: Path
    patient_id: np.ndarray
    split: np.ndarray
    z_R: np.ndarray
    z_P: np.ndarray
    full: np.ndarray
    arm: str
    seed_base: int
    fold: int
    z_P_aug: np.ndarray | None
    z_P_future_pred: np.ndarray | None
    z_P_future_target: np.ndarray | None
    z_P_future_context: np.ndarray | None
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class F0Asset:
    path: Path
    patient_id: np.ndarray
    split: np.ndarray
    state: np.ndarray
    seed_base: int
    fold: int
    metadata: Mapping[str, Any]


def load_factorized_export_status(config: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    frozen = config["frozen_inputs"]
    path = _resolve(frozen["factorized_export_status_path"])
    expected_path = (EXPERIMENT_ROOT / "metrics/feature_export_status.json").resolve()
    if path != expected_path or not path.is_file():
        raise EvaluationContractError("canonical factorized export status is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version",
        "experiment",
        "phase",
        "status",
        "required_cells",
        "completed_cells",
        "already_complete_cells",
        "launched_cells",
        "arms",
        "seed_bases",
        "folds",
        "PCR_LABEL_ACCESS",
        "pcr_labels_used",
        "preregistration_lock_sha256",
        "preregistration_payload_sha256",
    }
    if set(payload) != expected_keys:
        raise EvaluationContractError("factorized export status schema drifted")
    representation = load_representation_lock(config)
    expected = {
        "schema_version": 1,
        "experiment": "clinical_residual_phenotype_state",
        "phase": "feature_export",
        "status": "COMPLETE",
        "required_cells": 20,
        "completed_cells": 20,
        "arms": list(ARMS),
        "seed_bases": list(SEEDS),
        "folds": list(FOLDS),
        "PCR_LABEL_ACCESS": "FORBIDDEN",
        "pcr_labels_used": False,
        "preregistration_lock_sha256": file_sha256(
            _resolve(frozen["representation_preregistration_lock"])
        ),
        "preregistration_payload_sha256": representation["lock_sha256"],
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise EvaluationContractError("factorized export status identity/firewall drifted")
    already = payload.get("already_complete_cells")
    launched = payload.get("launched_cells")
    if (
        isinstance(already, bool)
        or isinstance(launched, bool)
        or not isinstance(already, int)
        or not isinstance(launched, int)
        or already < 0
        or launched < 0
        or already + launched != 20
    ):
        raise EvaluationContractError("factorized export status completion accounting drifted")
    return path, payload


def _validate_factorized_provenance(
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    patient_id: np.ndarray,
    split: np.ndarray,
    arm: str,
    seed: int,
    fold: int,
) -> None:
    if set(metadata) != FACTORIZED_METADATA_KEYS:
        missing = sorted(FACTORIZED_METADATA_KEYS - set(metadata))
        extra = sorted(set(metadata) - FACTORIZED_METADATA_KEYS)
        raise EvaluationContractError(
            f"factorized metadata schema drifted (missing={missing}, extra={extra})"
        )
    frozen = config["frozen_inputs"]
    representation_path = _resolve(frozen["representation_preregistration_lock"])
    representation = load_representation_lock(config)
    representation_config_path = EXPERIMENT_ROOT / str(representation["config_path"])
    representation_config = json.loads(representation_config_path.read_text(encoding="utf-8"))
    identity = {
        "schema_version": 1,
        "experiment": "clinical_residual_phenotype_state",
        "arm": arm,
        "seed_base": seed,
        "fold": fold,
        "effective_seed": seed + fold,
        "patient_count": len(patient_id),
        "PCR_LABEL_ACCESS": "FORBIDDEN",
        "pcr_labels_used": False,
        "representation_frozen_before_export": True,
        "export_completed": True,
        "preregistration_lock_sha256": file_sha256(representation_path),
        "preregistration_payload_sha256": representation["lock_sha256"],
        "augmentation": representation_config["augmentation"],
    }
    if any(metadata.get(key) != value for key, value in identity.items()):
        raise EvaluationContractError("factorized metadata identity/firewall drifted")
    expected_shapes = {
        "z_R": [len(patient_id), 4, 96],
        "z_P": [len(patient_id), 4, 96],
        "full": [len(patient_id), 4, 192],
        "z_P_aug": [len(patient_id), 4, 96],
        "z_P_future_pred": [len(patient_id), 3, 96],
        "z_P_future_target": [len(patient_id), 3, 96],
        "z_P_future_context": [len(patient_id), 3, 96],
    }
    if metadata.get("state_shapes") != expected_shapes:
        raise EvaluationContractError("factorized metadata state shapes drifted")
    patient_list = patient_id.astype(str).tolist()
    patient_hashes = {
        "patient_order_sha256": canonical_sha256(patient_list),
        "train_patient_sha256": canonical_sha256(
            patient_id[split == "train"].astype(str).tolist()
        ),
        "validation_patient_sha256": canonical_sha256(
            patient_id[split == "val"].astype(str).tolist()
        ),
        "test_patient_sha256": canonical_sha256(
            patient_id[split == "test"].astype(str).tolist()
        ),
    }
    if any(metadata.get(key) != value for key, value in patient_hashes.items()):
        raise EvaluationContractError("factorized metadata patient binding drifted")

    checkpoint_root = _resolve(frozen["factorized_checkpoint_root"])
    canonical_root = (EXPERIMENT_ROOT / "checkpoints/formal_primary").resolve()
    if checkpoint_root != canonical_root:
        raise EvaluationContractError("factorized checkpoint root is not canonical")
    checkpoint_path = checkpoint_root / f"seed_{seed}" / arm / f"fold_{fold}" / "selected.pt"
    selection_path = checkpoint_path.parent / "selection.json"
    if not checkpoint_path.is_file() or not selection_path.is_file():
        raise EvaluationContractError("canonical selected checkpoint/selection is missing")
    if metadata.get("checkpoint_sha256") != file_sha256(checkpoint_path):
        raise EvaluationContractError("selected checkpoint hash binding drifted")
    if metadata.get("selection_sha256") != file_sha256(selection_path):
        raise EvaluationContractError("checkpoint selection hash binding drifted")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection_identity = {
        "schema_version": 1,
        "arm": arm,
        "seed_base": seed,
        "fold": fold,
        "effective_seed": seed + fold,
        "PCR_LABEL_ACCESS": "FORBIDDEN",
        "pcr_used": False,
        "test_data_used": False,
    }
    if any(selection.get(key) != value for key, value in selection_identity.items()):
        raise EvaluationContractError("checkpoint selection identity/firewall drifted")
    if selection.get("preregistration") != representation:
        raise EvaluationContractError("checkpoint selection representation lock drifted")
    if (
        metadata.get("selected_epoch") != selection.get("selected_epoch")
        or metadata.get("selection_mode") != selection.get("selection_mode")
        or metadata.get("selection_experiment_pass") != selection.get("experiment_pass")
    ):
        raise EvaluationContractError("factorized metadata/selection binding drifted")
    if (
        isinstance(selection.get("selected_epoch"), bool)
        or not isinstance(selection.get("selected_epoch"), int)
        or selection["selected_epoch"] < 1
    ):
        raise EvaluationContractError("selected epoch is invalid")

    import torch

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True, mmap=True
    )
    checkpoint_identity = {
        "schema_version": 1,
        "stage": "clinical_residual_phenotype_state",
        "arm": arm,
        "seed_base": seed,
        "fold": fold,
        "effective_seed": seed + fold,
        "epoch": selection["selected_epoch"],
        "selected": True,
        "PCR_LABEL_ACCESS": "FORBIDDEN",
        "pcr_used": False,
        "pcr_parsed": False,
        "test_data_used": False,
        "delta_ftv_used_for_selection": False,
    }
    if any(checkpoint.get(key) != value for key, value in checkpoint_identity.items()):
        raise EvaluationContractError("selected checkpoint identity/firewall drifted")
    if checkpoint.get("selection") != selection or checkpoint.get("preregistration") != representation:
        raise EvaluationContractError("selected checkpoint lock/selection binding drifted")


def load_factorized_asset(
    config: Mapping[str, Any], folds: pd.DataFrame, arm: str, seed: int, fold: int
) -> FactorizedAsset:
    if arm not in ARMS or seed not in SEEDS or fold not in FOLDS:
        raise EvaluationContractError("invalid factorized cell identity")
    root = _resolve(config["frozen_inputs"]["factorized_feature_root"])
    if root != (EXPERIMENT_ROOT / "features/formal_primary").resolve():
        raise EvaluationContractError("factorized feature root is not canonical")
    path = root / f"seed_{seed}" / arm / f"fold_{fold}" / "factorized_state.private.npz"
    if not path.is_file():
        raise FileNotFoundError(f"factorized feature asset missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "patient_id",
            "split",
            "z_R",
            "z_P",
            "full",
            "arm",
            "seed_base",
            "fold",
            "z_P_aug",
            "z_P_future_pred",
            "z_P_future_target",
            "z_P_future_context",
        }
        if set(archive.files) != required:
            raise EvaluationContractError("factorized NPZ key contract drifted")
        arrays = {key: archive[key].copy() for key in archive.files}
    patient_id = arrays["patient_id"]
    split = arrays["split"]
    expected_n = int(config["frozen_inputs"]["expected_primary_patient_count"])
    if patient_id.dtype.kind != "U" or patient_id.shape != (expected_n,):
        raise EvaluationContractError("factorized patient_id must be Unicode [808]")
    if split.dtype.kind != "U" or split.shape != patient_id.shape or set(split) != set(SPLITS):
        raise EvaluationContractError("factorized split must be Unicode train/val/test [808]")
    if len(set(patient_id.astype(str))) != expected_n:
        raise EvaluationContractError("factorized patient IDs are duplicated")
    expected_shapes = {"z_R": (expected_n, 4, 96), "z_P": (expected_n, 4, 96), "full": (expected_n, 4, 192)}
    for key, shape in expected_shapes.items():
        value = arrays[key]
        if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
            raise EvaluationContractError(f"{key} must be finite float32 {shape}")
    if not np.array_equal(arrays["full"], np.concatenate((arrays["z_R"], arrays["z_P"]), axis=-1)):
        raise EvaluationContractError("full state is not exact [z_R,z_P] concatenation")
    if arrays["arm"].shape != () or str(arrays["arm"].item()) != arm:
        raise EvaluationContractError("factorized arm scalar drifted")
    if arrays["seed_base"].shape != () or int(arrays["seed_base"].item()) != seed:
        raise EvaluationContractError("factorized seed scalar drifted")
    if arrays["fold"].shape != () or int(arrays["fold"].item()) != fold:
        raise EvaluationContractError("factorized fold scalar drifted")
    _validate_assignments(patient_id, split, folds, fold)
    diagnostic_shapes = {
        "z_P_aug": (expected_n, 4, 96),
        "z_P_future_pred": (expected_n, 3, 96),
        "z_P_future_target": (expected_n, 3, 96),
        "z_P_future_context": (expected_n, 3, 96),
    }
    for key, shape in diagnostic_shapes.items():
        if (
            arrays[key].dtype != np.float32
            or arrays[key].shape != shape
            or not np.isfinite(arrays[key]).all()
        ):
            raise EvaluationContractError(f"diagnostic {key} must be finite float32 {shape}")
    metadata_path = path.with_suffix(".metadata.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"factorized metadata missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("feature_sha256") != file_sha256(path):
        raise EvaluationContractError("factorized feature hash binding drifted")
    _validate_factorized_provenance(
        config, metadata, arrays, patient_id, split, arm, seed, fold
    )
    load_factorized_export_status(config)
    return FactorizedAsset(
        path=path,
        metadata_path=metadata_path,
        patient_id=patient_id.astype(str),
        split=split.astype(str),
        z_R=arrays["z_R"],
        z_P=arrays["z_P"],
        full=arrays["full"],
        arm=arm,
        seed_base=seed,
        fold=fold,
        z_P_aug=arrays.get("z_P_aug"),
        z_P_future_pred=arrays.get("z_P_future_pred"),
        z_P_future_target=arrays.get("z_P_future_target"),
        z_P_future_context=arrays.get("z_P_future_context"),
        metadata=metadata,
    )


def load_f0_asset(
    config: Mapping[str, Any], folds: pd.DataFrame, seed: int, fold: int
) -> F0Asset:
    if seed not in SEEDS or fold not in FOLDS:
        raise EvaluationContractError("invalid F0 cell identity")
    frozen = config["frozen_inputs"]
    _locked_file(
        frozen["f0_preregistration_lock"],
        frozen["f0_preregistration_lock_sha256"],
        "F0 preregistration lock",
    )
    root = _resolve(frozen["f0_feature_root"])
    expected_root = (
        REPO_ROOT
        / "additional_experiments/local_response_state_multiseed_confirmation/features/formal_4x8"
    ).resolve()
    if root != expected_root:
        raise EvaluationContractError("F0 must use the confirmed LOCAL3 formal_4x8 feature root")
    path = root / f"seed_{seed}" / "LOCAL3" / f"fold_{fold}" / "response_state.private.npz"
    if not path.is_file():
        raise FileNotFoundError(f"F0 feature asset missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required = {"patient_id", "split", "response_state", "arm", "seed_base", "fold"}
        if set(archive.files) != required:
            raise EvaluationContractError("F0 NPZ key contract drifted")
        arrays = {key: archive[key].copy() for key in archive.files}
    n = int(frozen["expected_primary_patient_count"])
    patient_id, split, state = arrays["patient_id"], arrays["split"], arrays["response_state"]
    if patient_id.dtype.kind != "U" or patient_id.shape != (n,) or len(set(patient_id)) != n:
        raise EvaluationContractError("F0 patient contract drifted")
    if split.dtype.kind != "U" or split.shape != (n,) or set(split) != set(SPLITS):
        raise EvaluationContractError("F0 split contract drifted")
    if state.dtype != np.float32 or state.shape != (n, 4, 192) or not np.isfinite(state).all():
        raise EvaluationContractError("F0 response state must be finite float32 [808,4,192]")
    if str(arrays["arm"].item()) != "LOCAL3" or int(arrays["seed_base"].item()) != seed or int(arrays["fold"].item()) != fold:
        raise EvaluationContractError("F0 cell scalar identity drifted")
    _validate_assignments(patient_id, split, folds, fold)
    metadata_path = path.with_suffix(".metadata.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"F0 feature metadata missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_metadata_keys = {
        "arm",
        "checkpoint_data_provenance_sha256",
        "checkpoint_path",
        "checkpoint_sha256",
        "cohort",
        "current_data_contract_provenance_sha256",
        "experiment",
        "feature_dtype",
        "feature_implementation_sha256",
        "feature_path",
        "feature_sha256",
        "feature_shape",
        "feature_tensor",
        "fold",
        "ftv_head_called",
        "patient_order_sha256",
        "preregistration_lock",
        "preregistration_lock_sha256",
        "schema_version",
        "seed_base",
        "selected_epoch",
        "selection_path",
        "selection_sha256",
        "stage_a_sentinel_sha256",
        "test_labels_used",
        "train_patient_sha256",
        "validation_patient_sha256",
    }
    if set(metadata) != expected_metadata_keys:
        raise EvaluationContractError("F0 metadata schema drifted")
    identity = {
        "schema_version": 1,
        "experiment": "local_response_state_multiseed_confirmation",
        "arm": "LOCAL3",
        "seed_base": seed,
        "fold": fold,
        "cohort": "exact_locked_primary_train_validation_test",
        "feature_dtype": "float32",
        "feature_shape": [n, 4, 192],
        "feature_tensor": "online_preprojector_response_state",
        "preregistration_lock": "PREREGISTRATION_LOCK.json",
        "preregistration_lock_sha256": frozen["f0_preregistration_lock_sha256"],
        "test_labels_used": False,
        "ftv_head_called": False,
    }
    if any(metadata.get(key) != value for key, value in identity.items()):
        raise EvaluationContractError("F0 export provenance drifted")
    if metadata.get("feature_sha256") != file_sha256(path):
        raise EvaluationContractError("F0 feature hash binding drifted")
    expected_patient_hashes = {
        "patient_order_sha256": ordered_patient_sha256(patient_id.astype(str).tolist()),
        "validation_patient_sha256": canonical_sha256(
            sorted(patient_id[split == "val"].astype(str).tolist())
        ),
    }
    if any(metadata.get(key) != value for key, value in expected_patient_hashes.items()):
        raise EvaluationContractError("F0 patient provenance binding drifted")
    digest_fields = (
        "checkpoint_data_provenance_sha256",
        "checkpoint_sha256",
        "current_data_contract_provenance_sha256",
        "feature_implementation_sha256",
        "selection_sha256",
        "stage_a_sentinel_sha256",
        "train_patient_sha256",
    )
    if any(
        not isinstance(metadata.get(key), str) or len(str(metadata[key])) != 64
        for key in digest_fields
    ):
        raise EvaluationContractError("F0 provenance digest field drifted")
    return F0Asset(
        path=path,
        patient_id=patient_id.astype(str),
        split=split.astype(str),
        state=state,
        seed_base=seed,
        fold=fold,
        metadata=metadata,
    )


__all__ = [
    "ARMS",
    "CONFIG_PATH",
    "EvaluationContractError",
    "EXPERIMENT_ROOT",
    "F0Asset",
    "FOLDS",
    "FactorizedAsset",
    "REPO_ROOT",
    "SEEDS",
    "file_sha256",
    "load_clinical",
    "load_evaluation_config",
    "load_f0_asset",
    "load_factorized_asset",
    "load_factorized_export_status",
    "load_fold_assignments",
    "load_fold_manifest",
    "load_representation_lock",
    "load_stage_b_ftv_records",
    "load_ftv_wide",
]

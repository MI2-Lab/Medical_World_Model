#!/usr/bin/env python3
"""Run the frozen, fold-isolated downstream ceiling evaluation.

Patient-level predictions are written only below the gitignored ``predictions``
directory.  Every tracked output produced by this script is aggregate-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = EXPERIMENT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from conditional_ceiling.clinical import (  # noqa: E402
    TrainOnlyClinicalEncoder,
    ftv_prefix_matrix,
    load_clinical_table,
    load_ftv_wide,
    prefix_matrix,
)
from conditional_ceiling.contracts import (  # noqa: E402
    ARMS,
    FOLDS,
    PRIMARY_TIMINGS,
    SEEDS,
    TIMINGS,
    file_sha256,
    load_aligned_full_cohort,
    load_config,
    resolve_input_paths,
)
from conditional_ceiling.evaluation import (  # noqa: E402
    aggregate_oof_metrics,
    evaluate_feature_families,
    fit_compact_logistic,
    fit_profile_probe,
    generalization_gap_table,
)
from conditional_ceiling.gates import evaluate_gates  # noqa: E402
from conditional_ceiling.metrics import binary_metrics, paired_fold_stratified_bootstrap  # noqa: E402
from conditional_ceiling.strata import build_outer_train_strata  # noqa: E402


PRIVATE_PREDICTIONS = EXPERIMENT_ROOT / "predictions" / "oof_predictions.private.csv"
PUBLIC_METRICS = EXPERIMENT_ROOT / "metrics"
SUPERVISED_ARMS = ("B1", "B2", "B3")
SUBGROUPS = ("HR-/HER2-", "HR+/HER2-", "HER2+")
CACHE_CONTENT_DIGEST_AGGREGATE_SHA256 = (
    "f1c9965e8ae5456a899735a5462b76277ba0ec97a229dedc5faf9c380ce94c89"
)
CONFIRMED_LOCAL3_PREREGISTRATION_LOCK_SHA256 = (
    "a4e1cd2d8b61a7130da2b2eb6dc04e9a5355f44d0a37f4ceccf2fba48b35a9ee"
)


def _atomic_csv(frame: pd.DataFrame, path: Path, *, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if private:
        # The containing directory and every inode that ever contains patient
        # rows are private *before* pandas writes the first byte.  A later chmod
        # is insufficient because the old implementation exposed its temp file
        # under the process umask until the atomic rename.
        path.parent.chmod(0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            descriptor = -1
            frame.to_csv(stream, index=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        path.chmod(0o600 if private else 0o644)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _stable_seed(*values: Any, base: int = 0) -> int:
    payload = "|".join(str(value) for value in values).encode("utf-8")
    return int((int(base) + int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")) % (2**32 - 1))


def _split_indices(folds: pd.DataFrame, patient_ids: Sequence[str], fold: int) -> dict[str, np.ndarray]:
    rows = folds.loc[folds["fold"].eq(int(fold)), ["patient_id", "split"]].copy()
    requested = {str(value) for value in patient_ids}
    if len(requested) != len(patient_ids):
        raise ValueError("requested fold population contains duplicate identifiers")
    rows["patient_id"] = rows["patient_id"].astype(str)
    rows = rows.loc[rows["patient_id"].isin(requested)]
    mapping = dict(zip(rows["patient_id"].astype(str), rows["split"].astype(str), strict=True))
    if set(mapping) != requested:
        raise ValueError("fold population does not align")
    values = np.asarray([mapping[value] for value in patient_ids], dtype=object)
    output = {
        "train": np.flatnonzero(values == "train"),
        "validation": np.flatnonzero(values == "val"),
        "test": np.flatnonzero(values == "test"),
    }
    if sum(map(len, output.values())) != len(patient_ids):
        raise ValueError("fold splits do not cover the population")
    return output


def _load_b0_state(
    path: Path,
    expected_ids: Sequence[str],
    *,
    seed: int,
    fold: int,
    folds: pd.DataFrame,
    checkpoint_path: Path,
) -> np.ndarray:
    source = path.resolve(strict=True)
    metadata_path = source.with_name("response_state.private.metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    authoritative_checkpoint = Path(checkpoint_path).resolve(strict=True)
    authoritative_selection = authoritative_checkpoint.with_name("selection.json")
    selection = json.loads(authoritative_selection.read_text(encoding="utf-8"))
    try:
        metadata_epoch = int(metadata["selected_epoch"])
        selection_epoch = int(selection["selected_epoch"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("B0 selected-epoch provenance is missing or invalid") from error
    if (
        metadata.get("arm") != "LOCAL3"
        or int(metadata.get("seed_base", -1)) != int(seed)
        or int(metadata.get("fold", -1)) != int(fold)
        or metadata.get("feature_sha256") != file_sha256(source)
        or metadata.get("checkpoint_sha256") != file_sha256(authoritative_checkpoint)
        or metadata.get("selection_sha256") != file_sha256(authoritative_selection)
        or metadata_epoch != selection_epoch
        or selection.get("arm") != "LOCAL3"
        or int(selection.get("seed_base", -1)) != int(seed)
        or int(selection.get("fold", -1)) != int(fold)
        or selection.get("test_data_used") is not False
        or selection.get("pcr_used") is not False
        or metadata.get("preregistration_lock_sha256")
        != CONFIRMED_LOCAL3_PREREGISTRATION_LOCK_SHA256
        or selection.get("preregistration_lock_sha256")
        != CONFIRMED_LOCAL3_PREREGISTRATION_LOCK_SHA256
        or metadata.get("checkpoint_data_provenance_sha256")
        != selection.get("data_provenance_sha256")
        or metadata.get("test_labels_used") is not False
    ):
        raise ValueError("B0 feature/checkpoint/selection provenance contract failed")
    with np.load(source, allow_pickle=False) as payload:
        required = {"patient_id", "split", "response_state", "arm", "seed_base", "fold"}
        if set(payload.files) != required:
            raise ValueError(f"B0 feature schema mismatch: {sorted(payload.files)}")
        ids = payload["patient_id"].astype(str)
        split = payload["split"].astype(str)
        raw_state = np.asarray(payload["response_state"])
        state = np.asarray(raw_state, dtype=np.float32)
        observed_arm = np.asarray(payload["arm"]).astype(str).reshape(-1)
        observed_seed = np.asarray(payload["seed_base"]).reshape(-1)
        observed_fold = np.asarray(payload["fold"]).reshape(-1)
    if len(ids) != 808 or len(set(ids)) != 808 or set(ids) != set(expected_ids):
        raise ValueError("B0 feature population does not exactly equal full_808")
    if state.shape != (808, 4, 192) or raw_state.dtype != np.float32 or not np.isfinite(state).all():
        raise ValueError("B0 response state must be finite float32 [808,4,192]")
    if (
        set(observed_arm) != {"LOCAL3"}
        or set(observed_seed.astype(int)) != {int(seed)}
        or set(observed_fold.astype(int)) != {int(fold)}
    ):
        raise ValueError("B0 feature arm/seed/fold metadata disagrees")
    expected_index = _split_indices(folds, [str(value) for value in expected_ids], fold)
    expected_split = np.full(len(expected_ids), "", dtype=object)
    for name, indices in expected_index.items():
        expected_split[indices] = "val" if name == "validation" else name
    order = {value: index for index, value in enumerate(ids)}
    aligned = np.asarray([order[str(value)] for value in expected_ids], dtype=np.int64)
    if not np.array_equal(split[aligned], expected_split.astype(str)):
        raise ValueError("B0 feature split metadata disagrees with the frozen fold manifest")
    return np.ascontiguousarray(state[aligned], dtype=np.float32)


def _preflight_supervised_cells() -> None:
    """Validate all 30 trained cells before labels are loaded or outputs touched."""

    scripts_root = EXPERIMENT_ROOT / "scripts"
    if str(scripts_root) not in sys.path:
        sys.path.insert(0, str(scripts_root))
    from run_matrix import Cell, validate_cell_artifacts

    for seed in SEEDS:
        for arm in SUPERVISED_ARMS:
            for fold in FOLDS:
                validate_cell_artifacts(Cell(seed=int(seed), arm=str(arm), fold=int(fold)))


def _load_supervised_state(
    path: Path,
    expected_ids: Sequence[str],
    *,
    seed: int,
    fold: int,
    arm: str,
    folds: pd.DataFrame,
) -> np.ndarray:
    source = path.resolve(strict=True)
    selection_path = (
        EXPERIMENT_ROOT / "checkpoints" / f"seed_{seed}" / arm / f"fold_{fold}"
        / "selection.private.json"
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("feature_sha256") != file_sha256(source):
        raise ValueError("supervised feature no longer matches its validated selection/checkpoint binding")
    with np.load(source, allow_pickle=False) as payload:
        required = {"patient_id", "representation", "seed", "fold", "arm"}
        required.add("split")
        if set(payload.files) != required:
            raise ValueError(f"supervised feature schema mismatch: {sorted(payload.files)}")
        ids = payload["patient_id"].astype(str)
        split = payload["split"].astype(str)
        raw_state = np.asarray(payload["representation"])
        state = np.asarray(raw_state, dtype=np.float32)
        observed_seed = np.asarray(payload["seed"]).reshape(-1)
        observed_fold = np.asarray(payload["fold"]).reshape(-1)
        observed_arm = np.asarray(payload["arm"]).astype(str).reshape(-1)
    if len(ids) != 808 or len(set(ids)) != 808 or set(ids) != set(expected_ids):
        raise ValueError("supervised feature population does not exactly equal full_808")
    if state.shape != (808, 4, 64) or raw_state.dtype != np.float32 or not np.isfinite(state).all():
        raise ValueError("supervised representation must be finite float32 [808,4,64]")
    if set(observed_seed.astype(int)) != {int(seed)} or set(observed_fold.astype(int)) != {int(fold)} or set(observed_arm) != {arm}:
        raise ValueError("supervised feature seed/fold/arm metadata disagrees")
    expected_index = _split_indices(folds, [str(value) for value in expected_ids], fold)
    expected_split = np.full(len(expected_ids), "", dtype=object)
    for name, indices in expected_index.items():
        expected_split[indices] = "val" if name == "validation" else name
    order = {value: index for index, value in enumerate(ids)}
    aligned = np.asarray([order[str(value)] for value in expected_ids], dtype=np.int64)
    if not np.array_equal(split[aligned], expected_split.astype(str)):
        raise ValueError("supervised feature split metadata disagrees with the frozen fold manifest")
    return np.ascontiguousarray(state[aligned], dtype=np.float32)


def _representation_path(seed: int, arm: str, fold: int) -> Path:
    return EXPERIMENT_ROOT / "features" / f"seed_{seed}" / arm / f"fold_{fold}" / "representation.private.npz"


def _training_summary() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for arm in SUPERVISED_ARMS:
            for fold in FOLDS:
                path = EXPERIMENT_ROOT / "checkpoints" / f"seed_{seed}" / arm / f"fold_{fold}" / "selection.private.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                selection = payload["selection"]
                history = selection["history"]
                if (
                    payload.get("test_labels_used_for_training_or_selection") is not False
                    or payload.get("external_ispy1_patients_used") != 0
                    or not history
                ):
                    raise ValueError("training selection isolation/history contract failed")
                rows.append({
                    "seed": seed,
                    "arm": arm,
                    "fold": fold,
                    "selection_status": str(payload["status"]),
                    "selected_epoch": int(selection["selected_epoch"]),
                    "epochs_run": len(history),
                    "selected_validation_mean_auroc": float(selection["selected_validation_mean_auroc"]),
                    "anchor_sampling_strategy": payload.get("anchor_sampling_strategy"),
                    "logical_patient_batch_size": payload.get("logical_patient_batch_size"),
                    "encoder_microbatch_size": payload.get("encoder_microbatch_size"),
                    "eligible_anchors_per_epoch": payload.get("eligible_anchors_per_epoch"),
                    "feature_sha256": str(payload["feature_sha256"]),
                    "config_sha256": str(payload["config_sha256"]),
                    "test_labels_used": False,
                    "external_ispy1_patients_used": 0,
                    "world_model_claim_allowed": False,
                })
    return pd.DataFrame(rows)


def _cache_integrity_audit(
    cohort: Any,
    paths: Any,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Hash every frozen I-SPY2 C1B cache file and disclose counts only."""

    verified = 0
    content_records: list[tuple[str, str]] = []
    manifest_parent = Path(paths.c1b_cache_manifest).resolve().parent
    for row in cohort.cache.itertuples(index=False):
        raw = Path(str(row.cache_path)).expanduser()
        source = raw.resolve() if raw.is_absolute() else (manifest_parent / raw).resolve()
        observed = source.stat()
        actual_digest = file_sha256(source)
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_size != int(row.cache_size_bytes)
            or observed.st_mtime_ns != int(row.cache_mtime_ns)
            or actual_digest != str(row.cache_sha256)
        ):
            raise ValueError("frozen I-SPY2 cache content/stat audit failed")
        verified += 1
        content_records.append((str(row.patient_id), actual_digest))
    if verified != 808:
        raise ValueError(f"cache content audit expected 808 files, observed {verified}")
    aggregate = hashlib.sha256()
    for patient_id, digest in sorted(content_records):
        aggregate.update(patient_id.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    aggregate_digest = aggregate.hexdigest()
    if aggregate_digest != CACHE_CONTENT_DIGEST_AGGREGATE_SHA256:
        raise ValueError("frozen I-SPY2 cache aggregate content digest drifted")
    return pd.DataFrame([{
        "population": "full_808",
        "expected_files": 808,
        "stat_verified_files": verified,
        "sha256_verified_files": verified,
        "mismatches": 0,
        "external_files_hashed": 0,
        "cache_manifest_sha256": str(paths.c1b_cache_manifest_sha256),
        "stage_b_manifest_sha256": str(config["paths"]["stage_b_data_contract_sha256"]),
        "content_digest_aggregate_sha256": aggregate_digest,
        "content_sha256_verified": True,
    }])


def _matching_audit(clinical: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        strata = build_outer_train_strata(clinical, folds, fold)
        audit = strata.audit.as_dict()
        group_sizes = strata.assignments.groupby("stratum_id")["patient_id"].size()
        natural_sizes = (
            strata.assignments.loc[strata.assignments["eligible_anchor"], "stratum_id"]
            .map(group_sizes).clip(upper=4).astype(int)
        )
        if set(natural_sizes) - {3, 4}:
            raise ValueError("unexpected exact-stratum natural batch size")
        size3, size4 = int(natural_sizes.eq(3).sum()), int(natural_sizes.eq(4).sum())
        if size3 + size4 != int(audit["usable_patients"]):
            raise ValueError("natural logical batch counts disagree with usable anchors")
        rows.append({
            "fold": fold,
            **{key: value for key, value in audit.items() if not isinstance(value, (dict, list))},
            "pcr_negative": audit["pcr_class_distribution"]["0"],
            "pcr_positive": audit["pcr_class_distribution"]["1"],
            "usable_pcr_negative": audit["usable_pcr_class_distribution"]["0"],
            "usable_pcr_positive": audit["usable_pcr_class_distribution"]["1"],
            "natural_batch_size_3": size3,
            "natural_batch_size_4": size4,
            "max_unique_patients_per_logical_batch": 4,
        })
    return pd.DataFrame(rows)


def _fit_one_fold(
    *,
    labels: np.ndarray,
    state: np.ndarray,
    clinical: pd.DataFrame,
    ftv: pd.DataFrame,
    folds: pd.DataFrame,
    patient_ids: np.ndarray,
    seed: int,
    arm: str,
    fold: int,
    timing: str,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = _split_indices(folds, patient_ids.tolist(), fold)
    encoder = TrainOnlyClinicalEncoder().fit(clinical.iloc[index["train"]])
    clinical_matrix = encoder.transform(clinical)
    mri = prefix_matrix(state, timing)
    downstream = config["downstream"]
    full = evaluate_feature_families(
        labels=labels,
        mri_features=mri,
        clinical_features=clinical_matrix,
        train_indices=index["train"],
        validation_indices=index["validation"],
        test_indices=index["test"],
        population="full_808",
        patient_ids=patient_ids,
        outer_fold=fold,
        dimensions=downstream["pca_dimensions"],
        c_grid=downstream["c_grid"],
        random_state=_stable_seed(seed, arm, fold, timing, "full"),
    )

    ftv_ids = ftv["patient_id"].astype(str).to_numpy()
    full_position = {value: offset for offset, value in enumerate(patient_ids)}
    positions = np.asarray([full_position[value] for value in ftv_ids], dtype=np.int64)
    ftv_clinical = clinical.iloc[positions].reset_index(drop=True)
    ftv_labels = labels[positions]
    ftv_index = _split_indices(folds, ftv_ids.tolist(), fold)
    ftv_encoder = TrainOnlyClinicalEncoder().fit(ftv_clinical.iloc[ftv_index["train"]])
    ftv_evaluation = evaluate_feature_families(
        labels=ftv_labels,
        mri_features=mri[positions],
        clinical_features=ftv_encoder.transform(ftv_clinical),
        ftv_features=ftv_prefix_matrix(ftv, timing),
        train_indices=ftv_index["train"],
        validation_indices=ftv_index["validation"],
        test_indices=ftv_index["test"],
        population="ftv_complete_375",
        patient_ids=ftv_ids,
        outer_fold=fold,
        dimensions=downstream["pca_dimensions"],
        c_grid=downstream["c_grid"],
        random_state=_stable_seed(seed, arm, fold, timing, "ftv"),
    )
    predictions = pd.concat((full.predictions, ftv_evaluation.predictions), ignore_index=True)
    diagnostics = pd.concat((full.diagnostics, ftv_evaluation.diagnostics), ignore_index=True)
    for frame in (predictions, diagnostics):
        frame.insert(0, "timing", timing)
        frame.insert(0, "arm", arm)
        frame.insert(0, "seed", seed)
    diagnostics["supplementary"] = timing == "T0_T3"
    return predictions, diagnostics


def _pair(
    predictions: pd.DataFrame,
    *,
    seed: int,
    comparison_arm: str,
    timing: str,
    population: str,
    reference_family: str,
    comparison_family: str,
    reference_arm: str | None,
    label: str,
    draws: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    common = predictions.loc[
        predictions["split"].eq("test")
        & predictions["seed"].eq(seed)
        & predictions["timing"].eq(timing)
        & predictions["population"].eq(population)
    ]
    reference = common.loc[
        common["arm"].eq(reference_arm or comparison_arm)
        & common["model_family"].eq(reference_family)
    ]
    comparison = common.loc[
        common["arm"].eq(comparison_arm)
        & common["model_family"].eq(comparison_family)
    ]
    result = paired_fold_stratified_bootstrap(
        reference,
        comparison,
        n_bootstrap=draws,
        confidence_level=0.95,
        seed=bootstrap_seed,
        metrics=("auroc",),
    ).iloc[0]
    return {
        "comparison": label,
        "population": population,
        "seed": seed,
        "arm": comparison_arm,
        "timing": timing,
        "reference_arm": reference_arm or comparison_arm,
        "reference_family": reference_family,
        "comparison_family": comparison_family,
        "delta_auroc": float(result["delta"]),
        "ci_lower": float(result["ci_lower"]),
        "ci_upper": float(result["ci_upper"]),
        "reference_auroc": float(result["reference"]),
        "comparison_auroc": float(result["comparison_value"]),
        "n_patients": int(result["n_patients"]),
        "n_folds": int(result["n_folds"]),
        "n_bootstrap": int(result["n_bootstrap"]),
        "n_valid_bootstrap": int(result["n_valid_bootstrap"]),
        "confidence_level": float(result["confidence_level"]),
        "bootstrap_unit": str(result["bootstrap_unit"]),
        "ci_method": str(result["ci_method"]),
        "orientation": str(result["orientation"]),
        "bootstrap_seed": int(result["seed"]),
    }


def _clinical_labels(clinical: pd.DataFrame, target: str) -> np.ndarray:
    if target == "HR":
        return clinical["label_hr"].to_numpy(np.int64)
    if target == "HER2":
        return clinical["label_her2"].to_numpy(np.int64)
    if target == "subtype":
        return np.asarray([
            f"HR{'+' if int(hr) else '-'}/HER2{'+' if int(her2) else '-'}"
            for hr, her2 in zip(clinical["label_hr"], clinical["label_her2"], strict=True)
        ], dtype=object)
    if target == "treatment":
        return clinical["arm"].astype(str).to_numpy()
    raise ValueError(target)


def _profile_probes(
    *,
    states: Mapping[tuple[int, str, int], np.ndarray],
    clinical: pd.DataFrame,
    folds: pd.DataFrame,
    patient_ids: np.ndarray,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for arm in ("B0", "B2", "B3"):
            for timing in PRIMARY_TIMINGS:
                test_y: dict[str, list[Any]] = {target: [] for target in ("HR", "HER2", "subtype")}
                test_probability: dict[str, list[np.ndarray]] = {target: [] for target in test_y}
                fold_ids: dict[str, list[int]] = {target: [] for target in test_y}
                test_patient_ids: dict[str, list[str]] = {target: [] for target in test_y}
                for fold in FOLDS:
                    x = prefix_matrix(states[(seed, arm, fold)], timing)
                    index = _split_indices(folds, patient_ids.tolist(), fold)
                    # The shortcut probe uses the same low-capacity, outer-train
                    # compression boundary as the pCR analysis.  This keeps B0
                    # (192-D per visit) and projected supervised arms (64-D per
                    # visit) comparable without fitting on validation/test rows.
                    projector = PCA(n_components=64, svd_solver="full", whiten=False)
                    projector.fit(x[index["train"]])
                    x = projector.transform(x)
                    for target in tuple(test_y):
                        y = _clinical_labels(clinical, target)
                        try:
                            fit = fit_profile_probe(
                                x[index["train"]], y[index["train"]],
                                x[index["validation"]], y[index["validation"]],
                                x[index["test"]], c_grid=config["downstream"]["c_grid"],
                                random_state=_stable_seed(seed, arm, fold, timing, target),
                            )
                        except ValueError as error:
                            raise ValueError(
                                "profile probe failed; broad fold skipping is forbidden "
                                f"(seed={seed}, arm={arm}, fold={fold}, timing={timing}, target={target})"
                            ) from error
                        test_y[target].extend(y[index["test"]].tolist())
                        assert fit.test_probabilities is not None
                        test_probability[target].extend(fit.test_probabilities)
                        fold_ids[target].extend([fold] * len(index["test"]))
                        test_patient_ids[target].extend(patient_ids[index["test"]].astype(str).tolist())
                for target in tuple(test_y):
                    y = np.asarray(test_y[target])
                    probability = np.asarray(test_probability[target])
                    if (
                        len(y) != 808
                        or len(test_patient_ids[target]) != 808
                        or len(set(test_patient_ids[target])) != 808
                        or set(test_patient_ids[target]) != set(patient_ids.astype(str))
                        or set(fold_ids[target]) != set(FOLDS)
                    ):
                        raise ValueError(
                            "profile probe must contain every full-cohort patient exactly once "
                            f"across all five test folds ({seed}, {arm}, {timing}, {target})"
                        )
                    classes = np.unique(y)
                    if len(classes) == 2:
                        score = roc_auc_score(y, probability[:, 1])
                        metric = "auroc"
                    else:
                        score = roc_auc_score(y, probability, labels=classes, multi_class="ovr", average="macro")
                        metric = "macro_ovr_auroc"
                    rows.append({
                        "seed": seed, "arm": arm, "timing": timing, "target": target,
                        "metric": metric, "value": float(score), "n": int(len(y)),
                        "n_folds": len(set(fold_ids[target])), "fold_isolated": True,
                    })
    # Exact 13-arm treatment has sparse validation cells, so suitability is an
    # explicit negative audit rather than a silently collapsed target.
    rows.append({
        "seed": -1, "arm": "ALL", "timing": "ALL", "target": "treatment",
        "metric": "not_run", "value": math.nan, "n": 808, "n_folds": 0,
        "fold_isolated": True,
        "status": "unsuitable_exact_13_arm_target_due_to_sparse_fold_classes",
    })
    return pd.DataFrame(rows)


def _subgroup_refits(
    *,
    states: Mapping[tuple[int, str, int], np.ndarray],
    clinical: pd.DataFrame,
    folds: pd.DataFrame,
    patient_ids: np.ndarray,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    hr = clinical["label_hr"].to_numpy(np.int64)
    her2 = clinical["label_her2"].to_numpy(np.int64)
    group = np.where(her2 == 1, "HER2+", np.where(hr == 1, "HR+/HER2-", "HR-/HER2-"))
    labels = clinical["label_pcr"].to_numpy(np.int64)
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for arm in ARMS:
            for timing in PRIMARY_TIMINGS:
                for subgroup in SUBGROUPS:
                    held_label: list[int] = []
                    held_probability: list[float] = []
                    held_patient_ids: list[str] = []
                    member = np.flatnonzero(group == subgroup)
                    for fold in FOLDS:
                        fold_index = _split_indices(folds, patient_ids.tolist(), fold)
                        split = {
                            key: np.intersect1d(value, member, assume_unique=True)
                            for key, value in fold_index.items()
                        }
                        if any(set(np.unique(labels[value])) != {0, 1} for value in split.values()):
                            raise ValueError(
                                "registered subgroup lacks required class support; partial-fold "
                                f"aggregation is forbidden ({seed}, {arm}, {timing}, {subgroup}, fold={fold})"
                            )
                        x = prefix_matrix(states[(seed, arm, fold)], timing)
                        fit = fit_compact_logistic(
                            x[split["train"]], labels[split["train"]],
                            x[split["validation"]], labels[split["validation"]],
                            x[split["test"]], dimensions=config["downstream"]["pca_dimensions"],
                            c_grid=config["downstream"]["c_grid"],
                            random_state=_stable_seed(seed, arm, fold, timing, subgroup),
                        )
                        assert fit.test_probabilities is not None
                        held_label.extend(labels[split["test"]].tolist())
                        held_probability.extend(fit.test_probabilities.tolist())
                        held_patient_ids.extend(patient_ids[split["test"]].astype(str).tolist())
                    expected_members = set(patient_ids[member].astype(str))
                    if (
                        set(held_label) != {0, 1}
                        or len(held_label) != len(member)
                        or len(set(held_patient_ids)) != len(member)
                        or set(held_patient_ids) != expected_members
                    ):
                        raise ValueError(
                            "subgroup OOF rows must cover each eligible member exactly once "
                            f"across all five folds ({seed}, {arm}, {timing}, {subgroup})"
                        )
                    metric = binary_metrics(np.asarray(held_label), np.asarray(held_probability))
                    rows.append({
                        "seed": seed, "arm": arm, "timing": timing, "subgroup": subgroup,
                        "eligible": True, "status": "ok", "n_folds": len(FOLDS), **metric,
                    })
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-probes", action="store_true")
    parser.add_argument("--skip-subgroups", action="store_true")
    parser.add_argument("--bootstrap-draws", type=int, default=None)
    return parser.parse_args()


def _registered_bootstrap_draws(arguments: argparse.Namespace, config: Mapping[str, Any]) -> int:
    raw_draws = (
        config["bootstrap"]["draws"]
        if arguments.bootstrap_draws is None
        else arguments.bootstrap_draws
    )
    if type(raw_draws) is not int or raw_draws != 5000:
        raise ValueError("headline bootstrap requires exactly 5000 draws")
    return raw_draws


def main() -> None:
    args = parse_args()
    # This deliberately precedes configuration/cohort loading (which exposes
    # test labels) and every output write.  Evaluation is fail-closed unless all
    # 30 supervised seed/arm/fold cells satisfy the training artifact contract.
    _preflight_supervised_cells()
    _atomic_csv(_training_summary(), PUBLIC_METRICS / "training_summary.csv")
    config = load_config()
    paths = resolve_input_paths(config)
    cohort = load_aligned_full_cohort(config, paths, verify_cache_files=False)
    patient_ids = np.asarray(cohort.patient_ids)
    clinical = load_clinical_table(str(paths.clinical_labels), patient_ids.tolist())
    ftv = load_ftv_wide(str(paths.ftv_table), patient_ids.tolist())
    labels = clinical["label_pcr"].to_numpy(np.int64)

    _atomic_csv(
        _cache_integrity_audit(cohort, paths, config),
        PUBLIC_METRICS / "cache_integrity_audit.csv",
    )
    _atomic_csv(_matching_audit(clinical, cohort.folds), PUBLIC_METRICS / "matching_audit.csv")

    states: dict[tuple[int, str, int], np.ndarray] = {}
    prediction_blocks: list[pd.DataFrame] = []
    diagnostic_blocks: list[pd.DataFrame] = []
    for seed in SEEDS:
        for fold in FOLDS:
            b0 = _load_b0_state(
                paths.feature_path(seed, fold), patient_ids, seed=seed, fold=fold,
                folds=cohort.folds, checkpoint_path=paths.checkpoint_path(seed, fold),
            )
            states[(seed, "B0", fold)] = b0
            for arm in SUPERVISED_ARMS:
                states[(seed, arm, fold)] = _load_supervised_state(
                    _representation_path(seed, arm, fold), patient_ids, seed=seed, fold=fold,
                    arm=arm, folds=cohort.folds,
                )
            for arm in ARMS:
                for timing in TIMINGS:
                    prediction, diagnostic = _fit_one_fold(
                        labels=labels, state=states[(seed, arm, fold)], clinical=clinical,
                        ftv=ftv, folds=cohort.folds, patient_ids=patient_ids, seed=seed,
                        arm=arm, fold=fold, timing=timing, config=config,
                    )
                    prediction_blocks.append(prediction)
                    diagnostic_blocks.append(diagnostic)
    predictions = pd.concat(prediction_blocks, ignore_index=True)
    diagnostics = pd.concat(diagnostic_blocks, ignore_index=True)
    _atomic_csv(predictions, PRIVATE_PREDICTIONS, private=True)
    _atomic_csv(diagnostics, PUBLIC_METRICS / "fold_diagnostics.csv")
    aggregate = aggregate_oof_metrics(predictions)
    aggregate["supplementary"] = aggregate["timing"].eq("T0_T3")
    _atomic_csv(aggregate, PUBLIC_METRICS / "aggregate_metrics.csv")
    gaps = generalization_gap_table(
        predictions,
        group_cols=("population", "seed", "arm", "timing", "model_family"),
    )
    gaps["supplementary"] = gaps["timing"].eq("T0_T3")
    _atomic_csv(gaps, PUBLIC_METRICS / "generalization_gaps.csv")

    draws = _registered_bootstrap_draws(args, config)
    comparison_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for arm in SUPERVISED_ARMS:
            for timing in PRIMARY_TIMINGS:
                common = dict(seed=seed, comparison_arm=arm, timing=timing, draws=draws)
                comparison_rows.append(_pair(
                    predictions, population="full_808", reference_family="M", comparison_family="M",
                    reference_arm="B0", label="MRI_ceiling", bootstrap_seed=_stable_seed(seed, arm, timing, "mri", base=config["bootstrap"]["seed"]), **common,
                ))
                comparison_rows.append(_pair(
                    predictions, population="full_808", reference_family="C", comparison_family="C+M",
                    reference_arm=None, label="clinical_complementarity", bootstrap_seed=_stable_seed(seed, arm, timing, "cm", base=config["bootstrap"]["seed"]), **common,
                ))
                comparison_rows.append(_pair(
                    predictions, population="ftv_complete_375", reference_family="C+F", comparison_family="C+F+M",
                    reference_arm=None, label="beyond_ftv", bootstrap_seed=_stable_seed(seed, arm, timing, "cfm", base=config["bootstrap"]["seed"]), **common,
                ))
    comparisons = pd.DataFrame(comparison_rows)
    _atomic_csv(comparisons, PUBLIC_METRICS / "paired_bootstrap.csv")

    if not args.skip_probes:
        _atomic_csv(_profile_probes(states=states, clinical=clinical, folds=cohort.folds,
                                    patient_ids=patient_ids, config=config),
                    PUBLIC_METRICS / "clinical_profile_probes.csv")
    else:
        # A deliberately partial run must not leave a stale complete-looking
        # registry for report/verification consumers.
        (PUBLIC_METRICS / "clinical_profile_probes.csv").unlink(missing_ok=True)
    if not args.skip_subgroups:
        _atomic_csv(_subgroup_refits(states=states, clinical=clinical, folds=cohort.folds,
                                     patient_ids=patient_ids, config=config),
                    PUBLIC_METRICS / "subgroup_refits.csv")
    else:
        (PUBLIC_METRICS / "subgroup_refits.csv").unlink(missing_ok=True)

    gate_rows = {
        label: comparisons.loc[comparisons["comparison"].eq(label)].copy()
        for label in ("MRI_ceiling", "clinical_complementarity", "beyond_ftv")
    }
    decision = evaluate_gates(
        gate_rows["MRI_ceiling"], gate_rows["clinical_complementarity"], gate_rows["beyond_ftv"]
    )
    _atomic_json({
        "schema_version": 1,
        "reporting_boundary": config["reporting_boundary"],
        "primary_seeds": list(SEEDS),
        "folds_are_biological_replicates": False,
        "headline_bootstrap_draws": draws,
        "private_predictions_sha256": file_sha256(PRIVATE_PREDICTIONS),
        "decision": decision.as_dict(),
    }, PUBLIC_METRICS / "decision_summary.json")
    print(json.dumps(decision.as_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

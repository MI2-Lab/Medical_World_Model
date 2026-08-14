#!/usr/bin/env python3
"""Fail-closed verification for the completed public/private experiment bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
SOURCE_ROOT = EXPERIMENT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
SCRIPTS_ROOT = EXPERIMENT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from conditional_ceiling.contracts import load_config, resolve_input_paths  # noqa: E402
from conditional_ceiling.evaluation import (  # noqa: E402
    aggregate_oof_metrics,
    generalization_gap_table,
)
from conditional_ceiling.metrics import binary_metrics, paired_fold_stratified_bootstrap  # noqa: E402
from generate_report import (  # noqa: E402
    ARMS,
    FOLDS,
    METRICS,
    POPULATION_COUNTS,
    POPULATION_FAMILIES,
    PRIMARY_TIMINGS,
    PUBLIC_CSV_SCHEMAS,
    SEEDS,
    TIMINGS,
    _load_and_validate_inputs,
    _stable_seed,
    render_report,
)
from generate_figures import generate_figures  # noqa: E402


BOUNDARY = (
    "This experiment intentionally uses pCR supervision and estimates a supervised "
    "representation ceiling. It is not evidence that the pCR-free World Model learned "
    "this information."
)
PUBLIC_CSVS = (
    "cache_integrity_audit.csv",
    "aggregate_metrics.csv",
    "fold_diagnostics.csv",
    "paired_bootstrap.csv",
    "matching_audit.csv",
    "clinical_profile_probes.csv",
    "subgroup_refits.csv",
    "generalization_gaps.csv",
    "training_summary.csv",
)
FORBIDDEN_PUBLIC_COLUMNS = {
    "patient_id", "y_true", "predicted_probability", "predicted_label",
    "cache_path", "checkpoint_path", "feature_path",
}
FIGURE_FILENAMES = {
    "01_mri_only_auroc.png",
    "02_clinical_complementarity.png",
    "03_beyond_ftv_complementarity.png",
    "04_hr_her2_subgroups.png",
    "05_generalization_gap.png",
    "06_profile_decodability.png",
    "07_supervised_ceiling_gap.png",
}
STATIC_DELIVERY_FILES = {
    ".gitignore",
    "EXPERIMENT_PLAN.md",
    "README.md",
    "checkpoints/.gitkeep",
    "configs/experiment.json",
    "features/.gitkeep",
    "figures/.gitkeep",
    "logs/.gitkeep",
    "manifests/.gitkeep",
    "metrics/.gitkeep",
    "predictions/.gitkeep",
    "reports/.gitkeep",
    "scripts/generate_figures.py",
    "scripts/generate_report.py",
    "scripts/run_evaluation.py",
    "scripts/run_matrix.py",
    "scripts/train_cell.py",
    "scripts/verify_experiment.py",
    "src/conditional_ceiling/__init__.py",
    "src/conditional_ceiling/clinical.py",
    "src/conditional_ceiling/contracts.py",
    "src/conditional_ceiling/data.py",
    "src/conditional_ceiling/evaluation.py",
    "src/conditional_ceiling/gates.py",
    "src/conditional_ceiling/losses.py",
    "src/conditional_ceiling/metrics.py",
    "src/conditional_ceiling/model.py",
    "src/conditional_ceiling/strata.py",
    "src/conditional_ceiling/training.py",
    "tests/test_data_sampler.py",
    "tests/test_evaluation.py",
    "tests/test_gates.py",
    "tests/test_generate_figures.py",
    "tests/test_integration_hardening.py",
    "tests/test_losses.py",
    "tests/test_metrics.py",
    "tests/test_model_contract.py",
    "tests/test_run_evaluation.py",
    "tests/test_strata.py",
    "tests/test_training_contract.py",
}
FIGURE_TITLES = {
    "01_mri_only_auroc.png": "B0/B1/B2/B3 MRI-only AUROC",
    "02_clinical_complementarity.png": "Clinical complementarity: C+M - C",
    "03_beyond_ftv_complementarity.png": "Beyond-FTV complementarity",
    "04_hr_her2_subgroups.png": "pCR performance within HR/HER2 subgroups",
    "05_generalization_gap.png": "Train/validation/test generalization",
    "06_profile_decodability.png": "Clinical-profile decodability",
    "07_supervised_ceiling_gap.png": "Supervised ceiling gap vs current World Model",
}
VERIFICATION_TOP_LEVEL_KEYS = {
    "schema_version", "status", "branch", "base_commit", "reporting_boundary",
    "config_sha256", "input_contracts_resolved", "confirmed_local_root",
    "public_results", "public_files", "private_results",
    "patient_level_artifacts_tracked", "folds_treated_as_biological_replicates",
    "private_predictions_sha256",
}
VERIFICATION_PUBLIC_RESULT_KEYS = {
    "cache_audit_rows", "aggregate_rows", "bootstrap_rows", "matching_rows",
    "probe_rows", "subgroup_rows", "gap_rows", "diagnostic_rows", "training_rows",
    "interpretation",
}
VERIFICATION_PUBLIC_FILE_KEYS = {"public_files", "figures", "outside_folder_changes"}
VERIFICATION_PRIVATE_RESULT_KEYS = {
    "validated_supervised_cells", "private_prediction_rows", "private_held_out_rows",
    "private_predictions_mode", "private_directory_mode", "private_predictions_sha256",
    "public_outputs_recomputed_from_private",
}


def _run(*command: str) -> str:
    result = subprocess.run(
        command, cwd=REPO_ROOT, check=True, text=True, capture_output=True
    )
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frozen_sensitive_signatures() -> tuple[set[bytes], set[bytes]]:
    """Return exact patient IDs and patient-specific private artifact paths.

    Generic numeric constants and configured aggregate input locations are
    intentionally not signatures: they are legitimate public source/config
    content.  Exact frozen identifiers and expanded patient/artifact paths are
    never legitimate in a deliverable payload.
    """

    config = load_config()
    paths = resolve_input_paths(config)
    identifiers = set(
        pd.read_csv(paths.fold_manifest, usecols=["patient_id"], dtype={"patient_id": str})[
            "patient_id"
        ].astype(str)
    )
    if len(identifiers) != 808:
        raise ValueError("frozen patient identifier registry is not exactly 808")
    cache = pd.read_csv(
        paths.c1b_cache_manifest,
        usecols=["patient_id", "cache_path"],
        dtype={"patient_id": str, "cache_path": str},
    )
    cache = cache.loc[cache["patient_id"].isin(identifiers)]
    if len(cache) != 808 or cache["patient_id"].nunique() != 808:
        raise ValueError("frozen cache-path registry is not exactly full_808")
    manifest_parent = paths.c1b_cache_manifest.resolve().parent
    private_paths: set[str] = set()
    for raw_value in cache["cache_path"].astype(str):
        raw = Path(raw_value).expanduser()
        resolved = raw.resolve() if raw.is_absolute() else (manifest_parent / raw).resolve()
        private_paths.update((raw_value, str(resolved)))
    for seed in SEEDS:
        for fold in FOLDS:
            private_paths.update(
                (
                    str(paths.feature_path(seed, fold).resolve()),
                    str(paths.checkpoint_path(seed, fold).resolve()),
                )
            )
    return (
        {value.encode("utf-8") for value in identifiers},
        {value.encode("utf-8") for value in private_paths if value},
    )


def _scan_deliverable_content(candidate_paths: Iterable[Path]) -> None:
    """Reject frozen identities/private patient paths anywhere in delivery bytes."""

    patient_ids, private_paths = _frozen_sensitive_signatures()
    signatures = (*sorted(patient_ids), *sorted(private_paths))
    for path in candidate_paths:
        if path.is_symlink():
            raise ValueError(f"deliverable may not contain symbolic links: {path}")
        if not path.is_file():
            continue
        payload = path.read_bytes()
        if any(signature in payload for signature in signatures):
            raise ValueError(
                f"deliverable contains a frozen patient identifier or private artifact path: {path}"
            )


def _validate_verification_payload(payload: Any) -> None:
    if not isinstance(payload, dict) or set(payload) != VERIFICATION_TOP_LEVEL_KEYS:
        raise ValueError("verification JSON top-level schema drifted")
    nested = (
        ("public_results", VERIFICATION_PUBLIC_RESULT_KEYS),
        ("public_files", VERIFICATION_PUBLIC_FILE_KEYS),
        ("private_results", VERIFICATION_PRIVATE_RESULT_KEYS),
    )
    if any(
        not isinstance(payload.get(name), dict) or set(payload[name]) != expected
        for name, expected in nested
    ):
        raise ValueError("verification JSON nested schema drifted")
    if (
        type(payload.get("schema_version")) is not int
        or payload["schema_version"] != 1
        or payload.get("status") != "PASS"
        or payload.get("branch") != "feature/conditional-pcr-contrastive-ceiling"
        or payload.get("base_commit") != "7644e38"
        or payload.get("reporting_boundary") != BOUNDARY
        or payload.get("input_contracts_resolved") is not True
        or payload.get("patient_level_artifacts_tracked") is not False
        or payload.get("folds_treated_as_biological_replicates") is not False
        or payload.get("private_predictions_sha256")
        != payload["private_results"].get("private_predictions_sha256")
    ):
        raise ValueError("verification JSON value contract drifted")


def _assert_recomputed_frame(
    public: pd.DataFrame,
    recomputed: pd.DataFrame,
    *,
    keys: list[str],
    name: str,
) -> None:
    if set(public.columns) != set(recomputed.columns):
        raise ValueError(
            f"{name} schema differs from deterministic recomputation; "
            f"public_only={sorted(set(public) - set(recomputed))}, "
            f"private_only={sorted(set(recomputed) - set(public))}"
        )
    if public.duplicated(keys).any() or recomputed.duplicated(keys).any():
        raise ValueError(f"{name} repeats a recomputation key")
    columns = list(recomputed.columns)
    left = public.loc[:, columns].sort_values(keys, kind="mergesort").reset_index(drop=True)
    right = recomputed.loc[:, columns].sort_values(keys, kind="mergesort").reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            left, right, check_dtype=False, check_exact=False,
            rtol=1e-10, atol=1e-12,
        )
    except AssertionError as error:
        raise ValueError(f"{name} differs from deterministic recomputation") from error


def _assert_private_predictions_match(
    published: pd.DataFrame,
    recomputed: pd.DataFrame,
) -> None:
    keys = ["patient_id", "population", "seed", "arm", "timing", "fold", "model_family"]
    if set(published.columns) != set(recomputed.columns) or len(published) != len(recomputed):
        raise ValueError("private predictions differ from deterministic feature-based regeneration")
    left = published.sort_values(keys, kind="mergesort").reset_index(drop=True)
    right = recomputed.loc[:, left.columns].sort_values(keys, kind="mergesort").reset_index(drop=True)
    text_columns = ["patient_id", "population", "arm", "timing", "model_family", "split"]
    if any(
        not np.array_equal(left[column].astype(str).to_numpy(), right[column].astype(str).to_numpy())
        for column in text_columns
    ):
        raise ValueError("private prediction identities/splits differ from deterministic regeneration")
    numeric_columns = sorted(set(left.columns) - set(text_columns))
    left_numeric = left[numeric_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    right_numeric = right[numeric_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.array_equal(left_numeric, right_numeric, equal_nan=True):
        raise ValueError("private probabilities/selections differ from deterministic feature-based regeneration")


def _csv(name: str, required: set[str]) -> pd.DataFrame:
    path = EXPERIMENT_ROOT / "metrics" / name
    frame = pd.read_csv(path)
    missing = sorted(required - set(frame.columns))
    if frame.empty or missing:
        raise ValueError(f"{name} is empty or misses {missing}")
    if forbidden := sorted(FORBIDDEN_PUBLIC_COLUMNS & set(frame.columns)):
        raise ValueError(f"{name} exposes patient/private columns: {forbidden}")
    return frame


def _validate_public_results() -> dict[str, Any]:
    bundle = _load_and_validate_inputs(EXPERIMENT_ROOT)
    diagnostics = bundle["diagnostics"]

    # Bind every declared diagnostic n/class count to the hash-pinned frozen
    # fold manifest (and the exact FTV-complete identity set), rather than
    # trusting a producer-supplied split name/count.
    paths = resolve_input_paths(load_config())
    folds = pd.read_csv(
        paths.fold_manifest, usecols=["patient_id", "fold", "split", "label_pcr"]
    )
    folds["patient_id"] = folds["patient_id"].astype(str)
    ftv_ids = set(
        pd.read_csv(paths.ftv_table, usecols=["patient_id"])["patient_id"].astype(str)
    )
    expected_counts: dict[tuple[str, int, str], tuple[int, int, int]] = {}
    for population in POPULATION_FAMILIES:
        population_rows = folds if population == "full_808" else folds.loc[folds["patient_id"].isin(ftv_ids)]
        for fold in FOLDS:
            for output_split, source_split in (("train", "train"), ("validation", "val"), ("test", "test")):
                rows = population_rows.loc[
                    population_rows["fold"].eq(fold) & population_rows["split"].eq(source_split)
                ]
                positive = int(pd.to_numeric(rows["label_pcr"], errors="raise").sum())
                expected_counts[(population, fold, output_split)] = (
                    len(rows), positive, len(rows) - positive
                )
    for row in diagnostics.itertuples(index=False):
        expected = expected_counts[(str(row.population), int(row.fold), str(row.split))]
        observed = (int(row.n), int(row.n_positive), int(row.n_negative))
        if observed != expected:
            raise ValueError("fold diagnostics disagree with the frozen population/split manifest")

    return {
        "cache_audit_rows": len(bundle["cache_audit"]),
        "aggregate_rows": len(bundle["metrics"]),
        "bootstrap_rows": len(bundle["bootstrap"]),
        "matching_rows": len(bundle["matching"]),
        "probe_rows": len(bundle["probes"]),
        "subgroup_rows": len(bundle["subgroups"]),
        "gap_rows": len(bundle["gaps"]),
        "diagnostic_rows": len(diagnostics),
        "training_rows": len(bundle["training"]),
        "interpretation": bundle["decision"]["decision"]["interpretation_code"],
    }


def _validate_bootstrap_against_predictions(
    held_out: pd.DataFrame,
    bootstrap: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    pair_specs = {
        "MRI_ceiling": ("full_808", "B0", "M", "M", "mri"),
        "clinical_complementarity": ("full_808", None, "C", "C+M", "cm"),
        "beyond_ftv": ("ftv_complete_375", None, "C+F", "C+F+M", "cfm"),
    }
    for row in bootstrap.itertuples(index=False):
        population, fixed_reference, reference_family, comparison_family, seed_tag = pair_specs[
            str(row.comparison)
        ]
        reference_arm = fixed_reference or str(row.arm)
        common = held_out.loc[
            held_out["population"].eq(population)
            & held_out["seed"].eq(int(row.seed))
            & held_out["timing"].eq(str(row.timing))
        ]
        reference_rows = common.loc[
            common["arm"].eq(reference_arm)
            & common["model_family"].eq(reference_family)
        ]
        comparison_rows = common.loc[
            common["arm"].eq(str(row.arm))
            & common["model_family"].eq(comparison_family)
        ]
        expected_seed = _stable_seed(
            int(row.seed), str(row.arm), str(row.timing), seed_tag,
            base=int(config["bootstrap"]["seed"]),
        )
        recomputed = paired_fold_stratified_bootstrap(
            reference_rows,
            comparison_rows,
            n_bootstrap=int(config["bootstrap"]["draws"]),
            confidence_level=float(config["bootstrap"]["confidence_level"]),
            seed=expected_seed,
            metrics=("auroc",),
        ).iloc[0]
        if not (
            np.isclose(float(row.reference_auroc), float(recomputed["reference"]), rtol=1e-10, atol=1e-12)
            and np.isclose(float(row.comparison_auroc), float(recomputed["comparison_value"]), rtol=1e-10, atol=1e-12)
            and np.isclose(float(row.delta_auroc), float(recomputed["delta"]), rtol=1e-10, atol=1e-12)
            and np.isclose(float(row.ci_lower), float(recomputed["ci_lower"]), rtol=1e-10, atol=1e-12)
            and np.isclose(float(row.ci_upper), float(recomputed["ci_upper"]), rtol=1e-10, atol=1e-12)
            and int(row.n_patients) == int(recomputed["n_patients"])
            and int(row.n_folds) == int(recomputed["n_folds"])
            and int(row.n_bootstrap) == int(recomputed["n_bootstrap"])
            and int(row.n_valid_bootstrap) == int(recomputed["n_valid_bootstrap"])
            and np.isclose(float(row.confidence_level), float(recomputed["confidence_level"]), rtol=0.0, atol=1e-15)
            and str(row.bootstrap_unit) == str(recomputed["bootstrap_unit"])
            and str(row.ci_method) == str(recomputed["ci_method"])
            and str(row.orientation) == str(recomputed["orientation"])
            and int(row.bootstrap_seed) == int(recomputed["seed"])
        ):
            raise ValueError(
                "paired bootstrap CI/point/metadata do not match deterministic private recomputation"
            )


def _validate_supervised_selection_provenance(cell: Any, paths: Any) -> None:
    """Bind a trained cell to its exact validation selection and LOCAL3 source."""

    import torch

    checkpoint_path, selection_path, _, _ = cell.outputs()
    selection_payload = json.loads(selection_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    selection = selection_payload.get("selection")
    if not isinstance(selection, dict) or checkpoint.get("selection") != selection:
        raise ValueError("checkpoint selection payload does not exactly equal selection JSON")
    history = selection.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("selection history is missing")
    try:
        scores = np.asarray(
            [float(row["validation_mean_auroc"]) for row in history], dtype=float
        )
        raw_epochs = np.asarray([float(row["epoch"]) for row in history], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("selection history epoch/AUROC values are invalid") from error
    if (
        not np.isfinite(scores).all()
        or not np.isfinite(raw_epochs).all()
        or not np.equal(raw_epochs, np.floor(raw_epochs)).all()
        or not np.array_equal(
            raw_epochs.astype(np.int64),
            np.arange(1, len(history) + 1, dtype=np.int64),
        )
        or selection.get("arm") != cell.arm
        or selection.get("selection_timings") != ["T0", "T0_T1", "T0_T2"]
        or selection.get("test_labels_used") is not False
    ):
        raise ValueError("selection history/order/identity/isolation contract is invalid")
    # Replay the frozen trainer's strict improvement rule.  Differences of at
    # most 1e-12 are numerical ties, so the earlier selected epoch wins.
    best_score = -np.inf
    earliest_max_epoch = -1
    for epoch, score in zip(raw_epochs.astype(np.int64), scores, strict=True):
        if float(score) > float(best_score) + 1e-12:
            best_score = float(score)
            earliest_max_epoch = int(epoch)
    if (
        type(selection.get("selected_epoch")) is not int
        or int(selection["selected_epoch"]) != earliest_max_epoch
        or not np.isclose(
            float(selection.get("selected_validation_mean_auroc")),
            best_score,
            rtol=0.0,
            atol=0.0,
        )
    ):
        raise ValueError("selected epoch/score is not the earliest maximum validation AUROC")
    authoritative_checkpoint_sha = _sha256(paths.checkpoint_path(cell.seed, cell.fold))
    if (
        selection_payload.get("confirmed_checkpoint_sha256")
        != authoritative_checkpoint_sha
        or checkpoint.get("provenance", {}).get("confirmed_checkpoint_sha256")
        != authoritative_checkpoint_sha
    ):
        raise ValueError("supervised cell is not bound to the authoritative LOCAL3 checkpoint")


def _validate_private_results() -> dict[str, Any]:
    # Reuse the matrix runner's strict identity/isolation/schema validation.
    scripts = EXPERIMENT_ROOT / "scripts"
    sys.path.insert(0, str(scripts))
    from run_matrix import Cell, validate_cell_artifacts  # type: ignore

    count = 0
    paths = resolve_input_paths(load_config())
    for seed in (2026, 3026):
        for arm in ("B1", "B2", "B3"):
            for fold in range(5):
                cell = Cell(seed, arm, fold)
                validate_cell_artifacts(cell)
                _validate_supervised_selection_provenance(cell, paths)
                count += 1
    predictions = EXPERIMENT_ROOT / "predictions" / "oof_predictions.private.csv"
    if (
        not predictions.is_file()
        or predictions.is_symlink()
        or predictions.parent.is_symlink()
        or stat.S_IMODE(predictions.stat().st_mode) != 0o600
        or stat.S_IMODE(predictions.parent.stat().st_mode) != 0o700
    ):
        raise ValueError("private predictions require a real 0600 file in a 0700 directory")
    expected_columns = {
        "patient_id", "fold", "population", "seed", "arm", "timing",
        "model_family", "split", "y_true", "predicted_probability",
        "selected_dimension", "selected_C",
    }
    frame = pd.read_csv(
        predictions,
        dtype={"patient_id": str},
        low_memory=False,
        float_precision="round_trip",
    )
    if set(frame.columns) != expected_columns or len(frame) != 567_840:
        raise ValueError(
            f"private prediction schema/row registry drifted: columns={sorted(frame.columns)}, rows={len(frame)}"
        )
    key_columns = (
        "patient_id", "population", "seed", "arm", "timing", "fold", "model_family"
    )
    if frame[list(key_columns)].isna().any().any() or frame.duplicated(list(key_columns)).any():
        raise ValueError("private predictions repeat or omit a patient/model/fold key")
    numeric = frame[["seed", "fold", "y_true", "predicted_probability", "selected_C"]].apply(
        pd.to_numeric, errors="raise"
    )
    if (
        not np.isfinite(numeric.to_numpy(float)).all()
        or not numeric["seed"].isin(SEEDS).all()
        or not numeric["fold"].isin(FOLDS).all()
        or not numeric["y_true"].isin((0, 1)).all()
        or not numeric["predicted_probability"].between(0.0, 1.0, inclusive="both").all()
    ):
        raise ValueError("private prediction numeric values are invalid")
    if not frame["arm"].isin(ARMS).all() or not frame["timing"].isin(TIMINGS).all():
        raise ValueError("private predictions contain an unregistered arm/timing")

    compact_families = {"M", "C+M", "C+F+M"}
    compact = frame["model_family"].isin(compact_families)
    dimensions = pd.to_numeric(frame["selected_dimension"], errors="coerce")
    if (
        not dimensions.loc[compact].isin((8, 16, 32, 64)).all()
        or dimensions.loc[~compact].notna().any()
    ):
        raise ValueError("private prediction compact-dimension selections drifted")
    config = load_config()
    allowed_c = {float(value) for value in config["downstream"]["c_grid"]}
    if not set(numeric["selected_C"].astype(float)).issubset(allowed_c):
        raise ValueError("private prediction C selections left the registered grid")

    expected_cells = {
        (population, seed, arm, timing, fold, family)
        for population, families in POPULATION_FAMILIES.items()
        for seed in SEEDS for arm in ARMS for timing in TIMINGS for fold in FOLDS
        for family in families
    }
    cell_columns = ["population", "seed", "arm", "timing", "fold", "model_family"]
    observed_cells = set(frame[cell_columns].itertuples(index=False, name=None))
    if observed_cells != expected_cells:
        raise ValueError("private prediction fold-level model registry is not the exact 960-cell product")
    grouped = frame.groupby(cell_columns, sort=False).agg(
        rows=("patient_id", "size"), patients=("patient_id", "nunique")
    ).reset_index()
    expected_n = grouped["population"].map({name: values[0] for name, values in POPULATION_COUNTS.items()})
    if not grouped["rows"].eq(expected_n).all() or not grouped["patients"].eq(expected_n).all():
        raise ValueError("a private fold-level model cell lacks its exact population")

    paths = resolve_input_paths(config)
    frozen = pd.read_csv(
        paths.fold_manifest, usecols=["patient_id", "fold", "split", "label_pcr"],
        dtype={"patient_id": str},
    ).rename(columns={"split": "expected_split", "label_pcr": "expected_label"})
    frozen["expected_split"] = frozen["expected_split"].replace({"val": "validation"})
    aligned = frame.merge(
        frozen, on=["patient_id", "fold"], how="left", validate="many_to_one", indicator=True
    )
    if (
        not aligned["_merge"].eq("both").all()
        or not aligned["split"].astype(str).eq(aligned["expected_split"].astype(str)).all()
        or not pd.to_numeric(aligned["y_true"], errors="raise").eq(
            pd.to_numeric(aligned["expected_label"], errors="raise")
        ).all()
    ):
        raise ValueError("private prediction IDs/labels/splits disagree with the frozen manifest")
    full_ids = set(frozen["patient_id"])
    ftv_ids = set(
        pd.read_csv(paths.ftv_table, usecols=["patient_id"], dtype={"patient_id": str})["patient_id"]
    )
    observed_full = set(frame.loc[frame["population"].eq("full_808"), "patient_id"])
    observed_ftv = set(frame.loc[frame["population"].eq("ftv_complete_375"), "patient_id"])
    if observed_full != full_ids or observed_ftv != ftv_ids:
        raise ValueError("private prediction population identities drifted")
    held_out = frame.loc[frame["split"].eq("test")]
    nonfold_columns = ["population", "seed", "arm", "timing", "model_family"]
    held_counts = held_out.groupby(nonfold_columns, sort=False).agg(
        rows=("patient_id", "size"), patients=("patient_id", "nunique")
    ).reset_index()
    held_expected = held_counts["population"].map(
        {name: values[0] for name, values in POPULATION_COUNTS.items()}
    )
    if (
        len(held_out) != 113_568
        or len(held_counts) != 192
        or not held_counts["rows"].eq(held_expected).all()
        or not held_counts["patients"].eq(held_expected).all()
    ):
        raise ValueError("private held-out OOF registry is incomplete")

    prediction_sha256 = _sha256(predictions)
    decision = json.loads(
        (EXPERIMENT_ROOT / "metrics" / "decision_summary.json").read_text(encoding="utf-8")
    )
    if decision.get("private_predictions_sha256") != prediction_sha256:
        raise ValueError("decision/public bundle is not bound to the current private prediction file")

    model_keys = ["population", "seed", "arm", "timing", "model_family"]

    def compare_public(
        public_name: str,
        recomputed: pd.DataFrame,
        value_columns: list[str],
    ) -> None:
        public = pd.read_csv(EXPERIMENT_ROOT / "metrics" / public_name)
        merged = public.merge(
            recomputed,
            on=model_keys,
            how="outer",
            validate="one_to_one",
            suffixes=("_public", "_private"),
            indicator=True,
        )
        if not merged["_merge"].eq("both").all():
            raise ValueError(f"{public_name} registry does not match private predictions")
        for column in value_columns:
            public_values = pd.to_numeric(merged[f"{column}_public"], errors="raise").to_numpy(float)
            private_values = pd.to_numeric(merged[f"{column}_private"], errors="raise").to_numpy(float)
            if not np.allclose(public_values, private_values, rtol=1e-10, atol=1e-12, equal_nan=True):
                raise ValueError(f"{public_name}.{column} does not match private predictions")

    recomputed_aggregate = aggregate_oof_metrics(frame, group_cols=model_keys)
    compare_public(
        "aggregate_metrics.csv",
        recomputed_aggregate,
        ["n", "n_positive", "n_negative", *METRICS],
    )
    recomputed_gaps = generalization_gap_table(frame, group_cols=model_keys)
    gap_columns = [
        *(f"{split}_{metric}" for split in ("train", "validation", "test") for metric in METRICS),
        *(f"{split}_test_{metric}_gap" for split in ("train", "validation") for metric in METRICS),
    ]
    compare_public("generalization_gaps.csv", recomputed_gaps, gap_columns)

    diagnostic_keys = [*model_keys, "fold", "split"]
    diagnostic_rows: list[dict[str, Any]] = []
    for cell_key, cell in frame.groupby([*model_keys, "fold"], sort=True):
        cell_selected_c = pd.to_numeric(cell["selected_C"], errors="raise").drop_duplicates()
        cell_selected_dimension = (
            pd.to_numeric(cell["selected_dimension"], errors="coerce").dropna().drop_duplicates()
        )
        if len(cell_selected_c) != 1 or len(cell_selected_dimension) > 1:
            raise ValueError(
                "private selected dimension/C must be identical across train/validation/test rows"
            )
        validation = cell.loc[cell["split"].eq("validation")]
        validation_auroc = float(
            binary_metrics(
                validation["y_true"].to_numpy(),
                validation["predicted_probability"].to_numpy(),
            )["auroc"]
        )
        for split in ("train", "validation", "test"):
            rows = cell.loc[cell["split"].eq(split)]
            diagnostic_rows.append({
                **dict(zip([*model_keys, "fold"], cell_key, strict=True)),
                "split": split,
                "selected_dimension": (
                    float(cell_selected_dimension.iloc[0]) if len(cell_selected_dimension) else np.nan
                ),
                "selected_C": float(cell_selected_c.iloc[0]),
                "validation_selection_auroc": validation_auroc,
                **binary_metrics(
                    rows["y_true"].to_numpy(), rows["predicted_probability"].to_numpy()
                ),
            })
    recomputed_diagnostics = pd.DataFrame(diagnostic_rows)
    public_diagnostics = pd.read_csv(EXPERIMENT_ROOT / "metrics" / "fold_diagnostics.csv")
    compared_diagnostics = public_diagnostics.merge(
        recomputed_diagnostics,
        on=diagnostic_keys,
        how="outer",
        validate="one_to_one",
        suffixes=("_public", "_private"),
        indicator=True,
    )
    if not compared_diagnostics["_merge"].eq("both").all():
        raise ValueError("fold_diagnostics.csv registry does not match private predictions")
    for column in (
        "selected_dimension", "selected_C", "validation_selection_auroc",
        "n", "n_positive", "n_negative", *METRICS,
    ):
        public_values = pd.to_numeric(
            compared_diagnostics[f"{column}_public"], errors="coerce"
        ).to_numpy(float)
        private_values = pd.to_numeric(
            compared_diagnostics[f"{column}_private"], errors="coerce"
        ).to_numpy(float)
        if not np.allclose(public_values, private_values, rtol=1e-10, atol=1e-12, equal_nan=True):
            raise ValueError(f"fold_diagnostics.csv.{column} does not match private predictions")

    # Recompute the exact registered 5,000-draw intervals.  Gate B consumes a
    # CI boundary, so validating point estimates alone would permit a forged CI
    # to change the scientific interpretation.
    bootstrap = pd.read_csv(EXPERIMENT_ROOT / "metrics" / "paired_bootstrap.csv")
    _validate_bootstrap_against_predictions(held_out, bootstrap, config)

    from run_evaluation import (  # type: ignore
        _cache_integrity_audit,
        _fit_one_fold,
        _load_b0_state,
        _load_supervised_state,
        _matching_audit,
        _profile_probes,
        _representation_path,
        _subgroup_refits,
        _training_summary,
        load_aligned_full_cohort,
        load_clinical_table,
        load_ftv_wide,
    )

    recomputed_training = _training_summary()
    _assert_recomputed_frame(
        pd.read_csv(EXPERIMENT_ROOT / "metrics" / "training_summary.csv"),
        recomputed_training,
        keys=["seed", "arm", "fold"],
        name="training_summary.csv",
    )

    cohort = load_aligned_full_cohort(config, paths, verify_cache_files=False)
    patient_ids = np.asarray(cohort.patient_ids)
    clinical = load_clinical_table(str(paths.clinical_labels), patient_ids.tolist())
    recomputed_cache_audit = _cache_integrity_audit(cohort, paths, config)
    _assert_recomputed_frame(
        pd.read_csv(EXPERIMENT_ROOT / "metrics" / "cache_integrity_audit.csv"),
        recomputed_cache_audit,
        keys=["population"],
        name="cache_integrity_audit.csv",
    )
    _assert_recomputed_frame(
        pd.read_csv(EXPERIMENT_ROOT / "metrics" / "matching_audit.csv"),
        _matching_audit(clinical, cohort.folds),
        keys=["fold"],
        name="matching_audit.csv",
    )
    ftv = load_ftv_wide(str(paths.ftv_table), patient_ids.tolist())
    states: dict[tuple[int, str, int], np.ndarray] = {}
    for seed in SEEDS:
        for fold in FOLDS:
            states[(seed, "B0", fold)] = _load_b0_state(
                paths.feature_path(seed, fold), patient_ids, seed=seed, fold=fold,
                folds=cohort.folds, checkpoint_path=paths.checkpoint_path(seed, fold),
            )
            for arm in ("B1", "B2", "B3"):
                states[(seed, arm, fold)] = _load_supervised_state(
                    _representation_path(seed, arm, fold), patient_ids,
                    seed=seed, fold=fold, arm=arm, folds=cohort.folds,
                )
    regenerated_predictions: list[pd.DataFrame] = []
    regenerated_diagnostics: list[pd.DataFrame] = []
    labels = clinical["label_pcr"].to_numpy(np.int64)
    for seed in SEEDS:
        for fold in FOLDS:
            for arm in ARMS:
                for timing in TIMINGS:
                    prediction, diagnostic = _fit_one_fold(
                        labels=labels,
                        state=states[(seed, arm, fold)],
                        clinical=clinical,
                        ftv=ftv,
                        folds=cohort.folds,
                        patient_ids=patient_ids,
                        seed=seed,
                        arm=arm,
                        fold=fold,
                        timing=timing,
                        config=config,
                    )
                    regenerated_predictions.append(prediction)
                    regenerated_diagnostics.append(diagnostic)
    recomputed_predictions = pd.concat(regenerated_predictions, ignore_index=True)
    _assert_private_predictions_match(frame, recomputed_predictions)
    _assert_recomputed_frame(
        pd.read_csv(EXPERIMENT_ROOT / "metrics" / "fold_diagnostics.csv"),
        pd.concat(regenerated_diagnostics, ignore_index=True),
        keys=["population", "seed", "arm", "timing", "model_family", "fold", "split"],
        name="fold_diagnostics.csv",
    )
    recomputed_probes = _profile_probes(
        states=states, clinical=clinical, folds=cohort.folds,
        patient_ids=patient_ids, config=config,
    )
    _assert_recomputed_frame(
        pd.read_csv(EXPERIMENT_ROOT / "metrics" / "clinical_profile_probes.csv"),
        recomputed_probes,
        keys=["seed", "arm", "timing", "target"],
        name="clinical_profile_probes.csv",
    )
    recomputed_subgroups = _subgroup_refits(
        states=states, clinical=clinical, folds=cohort.folds,
        patient_ids=patient_ids, config=config,
    )
    _assert_recomputed_frame(
        pd.read_csv(EXPERIMENT_ROOT / "metrics" / "subgroup_refits.csv"),
        recomputed_subgroups,
        keys=["seed", "arm", "timing", "subgroup"],
        name="subgroup_refits.csv",
    )
    return {
        "validated_supervised_cells": count,
        "private_prediction_rows": len(frame),
        "private_held_out_rows": len(held_out),
        "private_predictions_mode": "0600",
        "private_directory_mode": "0700",
        "private_predictions_sha256": prediction_sha256,
        "public_outputs_recomputed_from_private": True,
    }


def _validate_public_files() -> dict[str, Any]:
    report = EXPERIMENT_ROOT / "reports" / "final_report.md"
    text = report.read_text(encoding="utf-8")
    decision = json.loads(
        (EXPERIMENT_ROOT / "metrics" / "decision_summary.json").read_text(encoding="utf-8")
    )["decision"]
    if (
        BOUNDARY not in text
        or len(text) < 4000
        or str(decision["interpretation_code"]) not in text
        or "B3−B0" not in text
    ):
        raise ValueError("Chinese final report lacks the prominent mandatory boundary/content")
    if text != render_report(EXPERIMENT_ROOT):
        raise ValueError("final report is stale or differs from deterministic current-metrics rendering")
    figure_root = (EXPERIMENT_ROOT / "figures").resolve()
    figures = pd.read_csv(figure_root / "figure_manifest.csv")
    required_manifest_columns = {
        "filename", "title", "relative_path", "sha256", "bytes", "source_csvs",
        "public_aggregate_only", "contains_patient_rows",
    }
    if (
        set(figures.columns) != required_manifest_columns
        or len(figures) != 7
        or set(figures["filename"].astype(str)) != FIGURE_FILENAMES
        or figures["filename"].duplicated().any()
        or figures["relative_path"].astype(str).duplicated().any()
    ):
        raise ValueError("figure manifest must list the seven registered figures exactly once")
    exact_figure_sources = {
        "01_mri_only_auroc.png": {"metrics/aggregate_metrics.csv"},
        "02_clinical_complementarity.png": {"metrics/paired_bootstrap.csv"},
        "03_beyond_ftv_complementarity.png": {"metrics/paired_bootstrap.csv"},
        "04_hr_her2_subgroups.png": {"metrics/subgroup_refits.csv"},
        "05_generalization_gap.png": {"metrics/generalization_gaps.csv"},
        "06_profile_decodability.png": {"metrics/clinical_profile_probes.csv"},
        "07_supervised_ceiling_gap.png": {"metrics/aggregate_metrics.csv"},
    }
    resolved_figure_paths: set[Path] = set()
    for row in figures.itertuples():
        path = (EXPERIMENT_ROOT / str(row.relative_path)).resolve()
        try:
            path.relative_to(figure_root)
        except ValueError as error:
            raise ValueError("figure manifest path escapes the registered figures directory") from error
        if (
            path.parent != figure_root
            or path.name != str(row.filename)
            or path in resolved_figure_paths
            or not path.is_file()
            or path.is_symlink()
            or _sha256(path) != str(row.sha256)
            or path.stat().st_size != int(row.bytes)
            or str(row.public_aggregate_only).strip().lower() not in {"true", "1"}
            or str(row.contains_patient_rows).strip().lower() not in {"false", "0"}
        ):
            raise ValueError("figure manifest digest/size mismatch")
        resolved_figure_paths.add(path)
        sources = {value for value in str(row.source_csvs).split(";") if value}
        if sources != exact_figure_sources[str(row.filename)]:
            raise ValueError("figure manifest source mapping differs from the registered figure contract")
        if str(row.title) != FIGURE_TITLES[str(row.filename)]:
            raise ValueError("figure manifest title differs from the registered figure contract")
        for source in sources:
            source_path = (EXPERIMENT_ROOT / source).resolve()
            try:
                source_path.relative_to(EXPERIMENT_ROOT.resolve())
            except ValueError as error:
                raise ValueError("figure source path escapes the experiment root") from error
            if not source_path.is_file():
                raise ValueError("figure manifest source does not exist")
        try:
            with Image.open(path) as decoded:
                if decoded.format != "PNG":
                    raise ValueError("registered figure payload is not PNG")
                decoded.verify()
        except Exception as error:
            raise ValueError(f"registered figure is not a valid decodable PNG: {path}") from error

    # Re-render into a confined temporary directory and compare exact bytes.
    # This binds every figure to the current aggregate inputs even if an attacker
    # also rewrites the manifest hash and byte count around an arbitrary PNG.
    with tempfile.TemporaryDirectory(prefix=".verify_figures.", dir=EXPERIMENT_ROOT) as directory:
        regenerated_root = Path(directory)
        generate_figures(EXPERIMENT_ROOT, output_dir=regenerated_root)
        for filename in FIGURE_FILENAMES:
            published = figure_root / filename
            regenerated = regenerated_root / filename
            if published.read_bytes() != regenerated.read_bytes():
                raise ValueError(f"figure bytes differ from deterministic PNG regeneration: {filename}")

    # Generated public directories are a closed registry.  Inspect actual
    # filesystem outputs (not merely Git candidates) so an untracked patient
    # dump cannot hide from verification.
    registered_csvs = {
        *(EXPERIMENT_ROOT / "metrics" / name for name in PUBLIC_CSVS),
        EXPERIMENT_ROOT / "figures" / "figure_manifest.csv",
    }
    allowed_public_outputs = {
        *registered_csvs,
        EXPERIMENT_ROOT / "metrics" / "decision_summary.json",
        *(EXPERIMENT_ROOT / "figures" / name for name in FIGURE_FILENAMES),
        EXPERIMENT_ROOT / "reports" / "final_report.md",
        EXPERIMENT_ROOT / "reports" / "verification.json",
        *(EXPERIMENT_ROOT / directory / ".gitkeep" for directory in ("metrics", "figures", "reports", "manifests")),
    }
    observed_public_csvs: set[Path] = set()
    observed_public_figures: set[Path] = set()
    public_roots = tuple(EXPERIMENT_ROOT / name for name in ("metrics", "figures", "reports", "manifests"))
    forbidden_lower = {value.lower() for value in FORBIDDEN_PUBLIC_COLUMNS}
    for public_root in public_roots:
        if not public_root.exists():
            continue
        for path in public_root.rglob("*"):
            if not path.is_file() or ".private." in path.name:
                continue
            if path not in allowed_public_outputs:
                raise ValueError(f"unexpected file in closed public-output registry: {path}")
            if path.suffix.lower() == ".csv":
                observed_public_csvs.add(path)
                columns = {
                    str(column).strip().lower()
                    for column in pd.read_csv(path, nrows=0).columns
                }
                if columns & forbidden_lower:
                    raise ValueError(f"public CSV exposes patient-level columns: {path}")
            elif path.suffix.lower() == ".png" and path.parent.resolve() == figure_root:
                observed_public_figures.add(path.resolve())
            elif path.suffix.lower() == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                if path.name == "verification.json":
                    _validate_verification_payload(payload)

                def keys(value: Any) -> set[str]:
                    if isinstance(value, dict):
                        return {
                            *(str(key).strip().lower() for key in value),
                            *(item for nested in value.values() for item in keys(nested)),
                        }
                    if isinstance(value, list):
                        return {item for nested in value for item in keys(nested)}
                    return set()

                if keys(payload) & forbidden_lower:
                    raise ValueError(f"public JSON exposes patient-level keys: {path}")
    if observed_public_csvs != registered_csvs:
        missing = sorted(str(path) for path in registered_csvs - observed_public_csvs)
        extra = sorted(str(path) for path in observed_public_csvs - registered_csvs)
        raise ValueError(f"public CSV registry drifted; missing={missing[:3]}, extra={extra[:3]}")
    if observed_public_figures != resolved_figure_paths:
        raise ValueError("public figure files do not equal the seven registered manifest paths")

    allowed_root_files = {EXPERIMENT_ROOT / name for name in (".gitignore", "EXPERIMENT_PLAN.md", "README.md")}
    unexpected_root_files = [
        path for path in EXPERIMENT_ROOT.iterdir()
        if path.is_file() and path not in allowed_root_files
    ]
    if unexpected_root_files:
        raise ValueError(f"unexpected root-level delivery file: {unexpected_root_files[0]}")

    candidates = set(_run("git", "ls-files").splitlines()) | set(
        _run("git", "ls-files", "--others", "--exclude-standard").splitlines()
    )
    prefix = str(EXPERIMENT_ROOT.relative_to(REPO_ROOT)) + "/"
    public = sorted(value for value in candidates if value.startswith(prefix))
    allowed_delivery = {
        *(prefix + value for value in STATIC_DELIVERY_FILES),
        *(prefix + f"metrics/{name}" for name in PUBLIC_CSVS),
        prefix + "metrics/decision_summary.json",
        *(prefix + f"figures/{name}" for name in FIGURE_FILENAMES),
        prefix + "figures/figure_manifest.csv",
        prefix + "reports/final_report.md",
        prefix + "reports/verification.json",
    }
    unexpected_delivery = sorted(set(public) - allowed_delivery)
    if unexpected_delivery:
        raise ValueError(f"delivery contains files outside the closed allowlist: {unexpected_delivery[:5]}")
    _scan_deliverable_content(
        REPO_ROOT / value for value in public if (REPO_ROOT / value).exists()
    )
    changed = set(_run("git", "diff", "--name-only", "7644e38").splitlines())
    untracked = set(_run("git", "ls-files", "--others", "--exclude-standard").splitlines())
    outside = sorted(
        value for value in changed | untracked
        if value and not value.startswith(prefix)
    )
    if outside:
        raise ValueError(f"prior experiment paths changed: {outside[:5]}")
    return {"public_files": len(public), "figures": len(figures), "outside_folder_changes": 0}


def verify() -> dict[str, Any]:
    # A previous PASS must never survive a failed or interrupted fresh audit.
    # The canonical result is recreated only after every check below succeeds.
    (EXPERIMENT_ROOT / "reports" / "verification.json").unlink(missing_ok=True)
    if _run("git", "branch", "--show-current") != "feature/conditional-pcr-contrastive-ceiling":
        raise ValueError("wrong delivery branch")
    config = load_config()
    paths = resolve_input_paths(config)
    public_results = _validate_public_results()
    public_files = _validate_public_files()
    private_results = _validate_private_results()
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "branch": "feature/conditional-pcr-contrastive-ceiling",
        "base_commit": "7644e38",
        "reporting_boundary": BOUNDARY,
        "config_sha256": _sha256(EXPERIMENT_ROOT / "configs" / "experiment.json"),
        "input_contracts_resolved": True,
        "confirmed_local_root": paths.confirmation_root.name,
        "public_results": public_results,
        "public_files": public_files,
        "private_results": private_results,
        "patient_level_artifacts_tracked": False,
        "folds_treated_as_biological_replicates": False,
    }
    payload["private_predictions_sha256"] = private_results["private_predictions_sha256"]
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    output = EXPERIMENT_ROOT / "reports" / "verification.json"
    output.unlink(missing_ok=True)
    try:
        payload = verify()
    except Exception as error:
        output.unlink(missing_ok=True)
        print(f"VERIFICATION_FAILED: {error}", file=sys.stderr)
        return 2
    try:
        _validate_verification_payload(payload)
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                descriptor = -1
                stream.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            _scan_deliverable_content([temporary])
            temporary.replace(output)
            output.chmod(0o644)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)
    except Exception as error:
        output.unlink(missing_ok=True)
        print(f"VERIFICATION_FAILED: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

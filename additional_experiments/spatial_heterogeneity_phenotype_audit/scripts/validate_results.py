#!/usr/bin/env python3
"""Fail-closed validation and staged-file privacy scan for the completed audit."""

from __future__ import annotations

import argparse
from itertools import product
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from common import (  # noqa: E402
    atomic_json,
    canonical_sha256,
    file_sha256,
    load_config,
    load_preregistration_implementation_erratum,
    preregistration_chain,
    preregistration_provenance_anchors,
    require_preregistration_lock,
)
from export_features import _load_oracle  # noqa: E402
from generate_figures import (  # noqa: E402
    FIGURES,
    FIGURE_MANIFEST_COLUMNS,
    encode_source_inputs,
    figure_source_inputs,
    pair_matched_oracle_deltas,
)
from generate_report import (  # noqa: E402
    TABLES,
    preregistration_commit_sha,
    validate_final_git_provenance,
)
from run_feature_matrix import (  # noqa: E402
    REPRESENTATIVE_PATH,
    validate_complete,
    validate_representative_asset,
)
from run_audit import (  # noqa: E402
    METRIC_COLUMNS,
    evaluate_gates,
    pooling_contract_table,
    stage_b_authorization,
)
from stage_b_pilot import (  # noqa: E402
    FEATURE_VARIANT as STAGE_B_VARIANT,
    MODEL_ARM as STAGE_B_ARM,
    TABLE8_COLUMNS,
    unauthorized_table,
)
from verify_cache_integrity import require_cache_integrity  # noqa: E402


EXPECTED_ROWS = {
    "table1_pooling_contract.csv": 10,
    "table2_phenotype_probes.csv": 240,
    "table3_mri_only_pcr.csv": 160,
    "table4_clinical_ftv_incremental.csv": 80,
    "table5_residualized_mri.csv": 64,
    "table6_longitudinal_heterogeneity.csv": 72,
    "table7_oracle_regions.csv": 640,
}
TABLE1_COLUMNS = (
    "variant",
    "components",
    "dimension",
    "role",
    "c_grid",
    "mask_free",
    "deployable",
)
TABLE4_COLUMNS = (
    "seed",
    "arm",
    "view",
    "target",
    "model",
    "clinical_contract",
    "population",
    "n",
    "n_positive",
    "n_negative",
    "n_classes",
    "auroc",
    "auprc",
    "balanced_accuracy",
    "brier",
    "delta_auroc_vs_C+F",
    "delta_auprc_vs_C+F",
    "brier_improvement_vs_C+F",
    "delta_auroc_vs_C+F+P1",
    "delta_auprc_vs_C+F+P1",
    "brier_improvement_vs_C+F+P1",
)
FORBIDDEN_SUFFIXES = {
    ".nii",
    ".gz",
    ".dcm",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".ckpt",
    ".h5",
    ".hdf5",
}
PRIVATE_DIRECTORIES = {
    "features",
    "predictions",
    "logs",
    "manifests",
    "checkpoints",
    "data",
}
MAX_TRACKED_BYTES = 20 * 1024 * 1024
PUBLIC_IDENTIFIER_COLUMNS = frozenset(
    {
        "patient_id",
        "clinical_patient_id",
        "raw_patient_id",
        "subject_id",
        "participant_id",
    }
)
DELIVERY_ALLOWED_PATHS = frozenset(
    {
        ".gitignore",
        "EXPERIMENT_PLAN.md",
        "PREREGISTRATION_AMENDMENT.json",
        "PREREGISTRATION_IMPLEMENTATION_ERRATUM.json",
        "PREREGISTRATION_LOCK.json",
        "configs/audit.json",
        "features/.gitkeep",
        "figures/.gitkeep",
        "logs/.gitkeep",
        "manifests/.gitkeep",
        "metrics/.gitkeep",
        "predictions/.gitkeep",
        "reports/.gitkeep",
        "scripts/build_oracle_sidecars.py",
        "scripts/common.py",
        "scripts/export_features.py",
        "scripts/freeze_preregistration.py",
        "scripts/generate_figures.py",
        "scripts/generate_report.py",
        "scripts/pooling.py",
        "scripts/run_audit.py",
        "scripts/run_feature_matrix.py",
        "scripts/stage_b_pilot.py",
        "scripts/validate_results.py",
        "scripts/verify_cache_integrity.py",
        "tests/test_cache_integrity.py",
        "tests/test_delivery_contracts.py",
        "tests/test_gates.py",
        "tests/test_oracle_regions.py",
        "tests/test_pooling.py",
        "tests/test_stage_b_pilot.py",
        "metrics/cache_integrity_contract.json",
        "metrics/oracle_region_contract.json",
        "metrics/hyperparameter_selections.csv",
        "metrics/gates.json",
        "metrics/stage_b_authorization.json",
        "metrics/run_summary.json",
        "metrics/figure_manifest.csv",
        "metrics/final_validation.json",
        "reports/final_report.md",
        "reports/report_manifest.json",
        *(f"metrics/{name}" for name in TABLES),
        *(f"figures/{name}" for name in FIGURES),
    }
)
RUN_SUMMARY_KEYS = frozenset(
    {
        "schema_version",
        "experiment",
        "stage",
        "status",
        "branch",
        "n_feature_assets",
        "n_full_patients",
        "n_ftv_complete_patients",
        "scientific_classification",
        "stage_b_authorized",
        "elapsed_seconds",
        "config_sha256",
        "preregistration_lock_sha256",
        "preregistration_chain",
        "feature_asset_sha256",
        "reused_implementation_sha256",
        "runtime_implementation_sha256",
        "artifacts",
        "public_outputs_contain_patient_level_data",
    }
)
PREREGISTRATION_CHAIN_KEYS = frozenset(
    {
        "preregistration_revision",
        "active_preregistration_lock_sha256",
        "preregistration_amendment_sha256",
        "original_preregistration_lock_sha256",
        "original_preregistration_commit",
        "active_preregistration_commit",
    }
)
RUN_SUMMARY_ARTIFACT_KEYS = frozenset(
    {"path", "sha256", "size_bytes", "patient_level_private"}
)
RUN_SUMMARY_ARTIFACT_PATHS = {
    "phenotype_predictions": "predictions/phenotype_oof.private.csv",
    "mri_pcr_predictions": "predictions/mri_only_pcr_oof.private.csv",
    "beyond_predictions": "predictions/beyond_ftv_oof.private.csv",
    "residual_predictions": "predictions/residualized_pcr_oof.private.csv",
    "longitudinal_predictions": "predictions/longitudinal_oof.private.csv",
    "oracle_predictions": "predictions/oracle_oof.private.csv",
    "table1": "metrics/table1_pooling_contract.csv",
    "table2": "metrics/table2_phenotype_probes.csv",
    "table3": "metrics/table3_mri_only_pcr.csv",
    "table4": "metrics/table4_clinical_ftv_incremental.csv",
    "table5": "metrics/table5_residualized_mri.csv",
    "table6": "metrics/table6_longitudinal_heterogeneity.csv",
    "table7": "metrics/table7_oracle_regions.csv",
    "hyperparameters": "metrics/hyperparameter_selections.csv",
    "gates": "metrics/gates.json",
    "stage_b_authorization": "metrics/stage_b_authorization.json",
}
RUN_SUMMARY_PRIVATE_ARTIFACTS = frozenset(
    {
        "phenotype_predictions",
        "mri_pcr_predictions",
        "beyond_predictions",
        "residual_predictions",
        "longitudinal_predictions",
        "oracle_predictions",
    }
)


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPO, text=True, stderr=subprocess.STDOUT
    ).strip()


def _tracked_candidate_paths() -> list[Path]:
    prefix = str(ROOT.relative_to(REPO)) + "/"
    names = set(filter(None, _git("ls-files", prefix).splitlines()))
    names.update(
        filter(
            None,
            _git(
                "diff", "--cached", "--name-only", "--diff-filter=ACMR", "--", prefix
            ).splitlines(),
        )
    )
    names.update(
        filter(
            None,
            _git(
                "ls-files", "--others", "--exclude-standard", "--", prefix
            ).splitlines(),
        )
    )
    return [REPO / name for name in sorted(names)]


def _privacy_scan(paths: list[Path]) -> dict[str, Any]:
    scanned = 0
    for path in paths:
        try:
            relative = path.relative_to(ROOT)
        except ValueError as error:
            raise ValueError(f"delivery path escaped the experiment: {path}") from error
        relative_name = relative.as_posix()
        if relative_name not in DELIVERY_ALLOWED_PATHS:
            raise ValueError(f"undeclared file is not allowed in delivery: {relative}")
        if path.is_symlink():
            raise ValueError(f"delivery may not contain symbolic links: {relative}")
        if not path.is_file():
            continue
        if path.stat().st_size > MAX_TRACKED_BYTES:
            raise ValueError(f"tracked audit file is unexpectedly large: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or path.name.endswith(".nii.gz"):
            raise ValueError(f"tracked prohibited binary/source artifact: {relative}")
        if relative.parts[0] in PRIVATE_DIRECTORIES and path.name != ".gitkeep":
            raise ValueError(f"tracked file in private directory: {relative}")
        if "private" in path.name.lower():
            raise ValueError(f"tracked filename is marked private: {relative}")
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path, nrows=5)
            normalized = {
                re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
                for value in frame.columns
            }
            if PUBLIC_IDENTIFIER_COLUMNS & normalized:
                raise ValueError(
                    f"tracked CSV exposes patient identifier column: {relative}"
                )
        if relative.parts[0] in {"metrics", "reports"} and path.suffix.lower() in {
            ".csv",
            ".json",
            ".md",
        }:
            text = path.read_text(encoding="utf-8", errors="strict")
            if re.search(r"ACRIN[-_ ]?\d", text, flags=re.IGNORECASE):
                raise ValueError(
                    f"public result contains a patient-like identifier: {relative}"
                )
            if "/data/" in text:
                raise ValueError(
                    f"public result contains an absolute source-data path: {relative}"
                )
        scanned += 1
    return {
        "tracked_files_scanned": scanned,
        "maximum_allowed_bytes": MAX_TRACKED_BYTES,
    }


def _require_exact_columns(
    frame: pd.DataFrame, expected: tuple[str, ...], name: str
) -> None:
    if tuple(frame.columns.astype(str)) != expected:
        raise ValueError(
            f"{name} schema drifted: observed={tuple(frame.columns)}, expected={expected}"
        )


def _require_identity_grid(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    expected: set[tuple[Any, ...]],
    name: str,
) -> None:
    if frame.duplicated(list(columns)).any():
        raise ValueError(f"{name} repeats an aggregate identity")
    observed = set(frame.loc[:, columns].itertuples(index=False, name=None))
    if observed != expected:
        missing = sorted(expected - observed, key=str)
        extra = sorted(observed - expected, key=str)
        raise ValueError(
            f"{name} identity grid drifted; missing={missing[:3]}, extra={extra[:3]}"
        )


def _require_metric_contract(frame: pd.DataFrame, name: str) -> None:
    for column in ("n", "n_classes", "auroc", "auprc", "balanced_accuracy"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{name} has missing/non-finite {column}")
    for column in ("auroc", "auprc", "balanced_accuracy"):
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"{name} has out-of-range {column}")
    n = pd.to_numeric(frame["n"], errors="raise").to_numpy(dtype=float)
    classes = pd.to_numeric(frame["n_classes"], errors="raise").to_numpy(dtype=float)
    if np.any(n <= 0) or np.any(n != np.floor(n)):
        raise ValueError(f"{name} has invalid sample counts")
    subtype = frame["target"].astype(str).eq("subtype_4class").to_numpy()
    pcr = frame["target"].astype(str).eq("pCR").to_numpy()
    if np.any(classes[subtype] != 4) or np.any(classes[~subtype] != 2):
        raise ValueError(f"{name} has invalid class counts")
    positive = pd.to_numeric(frame["n_positive"], errors="coerce").to_numpy(dtype=float)
    negative = pd.to_numeric(frame["n_negative"], errors="coerce").to_numpy(dtype=float)
    if np.any(np.isfinite(positive[subtype])) or np.any(np.isfinite(negative[subtype])):
        raise ValueError(f"{name} subtype rows must not expose binary counts")
    binary = ~subtype
    if (
        not np.isfinite(positive[binary]).all()
        or not np.isfinite(negative[binary]).all()
        or not np.array_equal(positive[binary] + negative[binary], n[binary])
    ):
        raise ValueError(f"{name} binary positive/negative counts drifted")
    brier = pd.to_numeric(frame["brier"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(brier[pcr]).all() or np.any((brier[pcr] < 0) | (brier[pcr] > 1)):
        raise ValueError(f"{name} pCR Brier values are invalid")
    if np.any(np.isfinite(brier[~pcr])):
        raise ValueError(f"{name} non-pCR Brier values must be unavailable")


def _validate_table8(
    frame: pd.DataFrame,
    *,
    authorized: bool,
    stage_a_phenotype: pd.DataFrame,
    stage_a_pcr: pd.DataFrame,
) -> None:
    _require_exact_columns(frame, TABLE8_COLUMNS, "table8_stage_b.csv")
    if not authorized:
        try:
            pd.testing.assert_frame_equal(
                frame.reset_index(drop=True),
                unauthorized_table().reset_index(drop=True),
                check_dtype=False,
            )
        except AssertionError as error:
            raise ValueError(
                "unauthorized Table 8 differs from its canonical row"
            ) from error
        return
    if len(frame) != 20:
        raise ValueError("authorized Table 8 must contain exactly 20 aggregate rows")
    if not frame["status"].eq("COMPLETE").all() or not frame["stage"].eq("B").all():
        raise ValueError("authorized Table 8 status/stage drifted")
    if not frame["seed"].eq(2026).all() or not frame["arm"].eq(STAGE_B_ARM).all():
        raise ValueError("authorized Table 8 seed/arm drifted")
    if not frame["variant"].eq(STAGE_B_VARIANT).all():
        raise ValueError("authorized Table 8 feature variant drifted")
    if (
        not frame["stage_a_baseline_seed"].eq(2026).all()
        or not frame["stage_a_baseline_arm"].eq("LOCAL3").all()
        or not frame["stage_a_baseline_variant"].eq("P1").all()
    ):
        raise ValueError("authorized Table 8 Stage-A baseline identity drifted")
    identities: set[tuple[Any, ...]] = set()
    for view, target in product(
        ("T0", "T1", "T2", "T3"), ("HR", "HER2", "subtype_4class")
    ):
        identities.add(("phenotype", view, target, "full_808"))
    for view, population in product(
        ("T0", "T0-T1", "T0-T2", "T0-T3"),
        ("full_808", "ftv_complete_375"),
    ):
        identities.add(("mri_only_pcr", view, "pCR", population))
    _require_identity_grid(
        frame,
        ("analysis", "view", "target", "population"),
        identities,
        "table8_stage_b.csv",
    )
    expected_n = frame["population"].map({"full_808": 808, "ftv_complete_375": 375})
    if not frame["n"].eq(expected_n).all():
        raise ValueError("authorized Table 8 OOF coverage drifted")
    _require_metric_contract(frame, "table8_stage_b.csv")
    baseline = pd.concat((stage_a_phenotype, stage_a_pcr), ignore_index=True)
    baseline = baseline.loc[
        baseline["seed"].eq(2026)
        & baseline["arm"].eq("LOCAL3")
        & baseline["variant"].eq("P1")
    ].copy()
    baseline_identity = ["view", "target", "population"]
    if len(baseline) != 20 or baseline.duplicated(baseline_identity).any():
        raise ValueError("Stage-A Table-8 baseline grid is incomplete/non-unique")
    baseline = baseline.rename(
        columns={
            "n": "expected_baseline_n",
            "n_positive": "expected_baseline_n_positive",
            "n_negative": "expected_baseline_n_negative",
            "n_classes": "expected_baseline_n_classes",
            "auroc": "expected_baseline_auroc",
            "auprc": "expected_baseline_auprc",
            "balanced_accuracy": "expected_baseline_balanced_accuracy",
            "brier": "expected_baseline_brier",
        }
    )
    paired = frame.merge(
        baseline.loc[
            :,
            [
                *baseline_identity,
                "expected_baseline_n",
                "expected_baseline_n_positive",
                "expected_baseline_n_negative",
                "expected_baseline_n_classes",
                "expected_baseline_auroc",
                "expected_baseline_auprc",
                "expected_baseline_balanced_accuracy",
                "expected_baseline_brier",
            ],
        ],
        on=baseline_identity,
        how="left",
        validate="one_to_one",
    )
    if paired["expected_baseline_n"].isna().any():
        raise ValueError("authorized Table 8 lacks a Stage-A P1 baseline")
    for current, expected in (
        ("n", "expected_baseline_n"),
        ("n_positive", "expected_baseline_n_positive"),
        ("n_negative", "expected_baseline_n_negative"),
        ("n_classes", "expected_baseline_n_classes"),
        ("baseline_auroc", "expected_baseline_auroc"),
        ("baseline_auprc", "expected_baseline_auprc"),
        ("baseline_balanced_accuracy", "expected_baseline_balanced_accuracy"),
    ):
        left = pd.to_numeric(paired[current], errors="coerce").to_numpy(dtype=float)
        right = pd.to_numeric(paired[expected], errors="coerce").to_numpy(dtype=float)
        if not np.allclose(left, right, rtol=0.0, atol=1e-12, equal_nan=True):
            raise ValueError(
                f"authorized Table 8 differs from Stage-A baseline at {current}"
            )
    for delta, current, baseline_column in (
        ("delta_auroc", "auroc", "baseline_auroc"),
        ("delta_auprc", "auprc", "baseline_auprc"),
        (
            "delta_balanced_accuracy",
            "balanced_accuracy",
            "baseline_balanced_accuracy",
        ),
    ):
        observed = pd.to_numeric(paired[delta], errors="coerce").to_numpy(dtype=float)
        expected_delta = pd.to_numeric(paired[current], errors="coerce").to_numpy(
            dtype=float
        ) - pd.to_numeric(paired[baseline_column], errors="coerce").to_numpy(
            dtype=float
        )
        if not np.isfinite(observed).all() or not np.allclose(
            observed, expected_delta, rtol=0.0, atol=1e-12
        ):
            raise ValueError(f"authorized Table 8 has invalid {delta}")
    pcr = paired["target"].eq("pCR").to_numpy()
    baseline_brier = pd.to_numeric(paired["baseline_brier"], errors="coerce").to_numpy(
        dtype=float
    )
    expected_brier = pd.to_numeric(
        paired["expected_baseline_brier"], errors="coerce"
    ).to_numpy(dtype=float)
    improvement = pd.to_numeric(paired["brier_improvement"], errors="coerce").to_numpy(
        dtype=float
    )
    current_brier = pd.to_numeric(paired["brier"], errors="coerce").to_numpy(
        dtype=float
    )
    if (
        not np.allclose(baseline_brier[pcr], expected_brier[pcr], rtol=0.0, atol=1e-12)
        or not np.isfinite(improvement[pcr]).all()
        or not np.allclose(
            improvement[pcr],
            baseline_brier[pcr] - current_brier[pcr],
            rtol=0.0,
            atol=1e-12,
        )
        or np.isfinite(baseline_brier[~pcr]).any()
        or np.isfinite(improvement[~pcr]).any()
    ):
        raise ValueError("authorized Table 8 Brier baseline/delta contract drifted")


def _validate_public_tables(
    config: Mapping[str, Any], *, stage_b_authorized: bool
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    output: dict[str, Any] = {}
    frames: dict[str, pd.DataFrame] = {}
    for name in TABLES:
        path = ROOT / "metrics" / name
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        if frame.empty:
            raise ValueError(f"public table is empty: {name}")
        if name in EXPECTED_ROWS and len(frame) != EXPECTED_ROWS[name]:
            raise ValueError(
                f"{name} has {len(frame)} rows; expected {EXPECTED_ROWS[name]}"
            )
        normalized = {str(column).strip().lower() for column in frame.columns}
        if {"patient_id", "clinical_patient_id", "raw_patient_id"} & normalized:
            raise ValueError(f"public table contains patient identity: {name}")
        frames[name] = frame
        output[name] = {"rows": int(len(frame)), "sha256": file_sha256(path)}

    table1 = frames[TABLES[0]]
    _require_exact_columns(table1, TABLE1_COLUMNS, TABLES[0])
    try:
        pd.testing.assert_frame_equal(
            table1.reset_index(drop=True),
            pooling_contract_table(config).reset_index(drop=True),
            check_dtype=False,
        )
    except AssertionError as error:
        raise ValueError(
            "public pooling-contract table differs from frozen config"
        ) from error

    seeds = tuple(int(value) for value in config["frozen_cells"]["seed_bases"])
    arms = tuple(str(value) for value in config["frozen_cells"]["arms"])
    metric_tables = (TABLES[1], TABLES[2], TABLES[4], TABLES[5], TABLES[6])
    for name in metric_tables:
        _require_exact_columns(frames[name], METRIC_COLUMNS, name)
        _require_metric_contract(frames[name], name)
    key = ("seed", "arm", "view", "target", "variant", "population")
    _require_identity_grid(
        frames[TABLES[1]],
        key,
        set(
            product(
                seeds,
                arms,
                ("T0", "T1", "T2", "T3"),
                ("HR", "HER2", "subtype_4class"),
                ("P1", "P2", "P3", "P4", "P5"),
                ("full_808",),
            )
        ),
        TABLES[1],
    )
    _require_identity_grid(
        frames[TABLES[2]],
        key,
        set(
            product(
                seeds,
                arms,
                ("T0", "T0-T1", "T0-T2", "T0-T3"),
                ("pCR",),
                ("P1", "P2", "P3", "P4", "P5"),
                ("full_808", "ftv_complete_375"),
            )
        ),
        TABLES[2],
    )
    _require_identity_grid(
        frames[TABLES[4]],
        key,
        set(
            product(
                seeds,
                arms,
                ("T0", "T0-T1", "T0-T2", "T0-T3"),
                ("pCR",),
                ("P1_res", "P3_res", "C+F+P1_res", "C+F+P3_res"),
                ("ftv_complete_375",),
            )
        ),
        TABLES[4],
    )
    _require_identity_grid(
        frames[TABLES[5]],
        key,
        set(
            product(
                seeds,
                arms,
                ("T0->T1", "T1->T2", "T2->T3"),
                ("pCR",),
                ("DELTA_MEAN", "DELTA_STD", "P3_PLUS_DELTA"),
                ("full_808", "ftv_complete_375"),
            )
        ),
        TABLES[5],
    )
    for name in (TABLES[1], TABLES[2], TABLES[4], TABLES[5]):
        expected_n = frames[name]["population"].map(
            {"full_808": 808, "ftv_complete_375": 375}
        )
        if not frames[name]["n"].eq(expected_n).all():
            raise ValueError(f"{name} OOF population coverage drifted")

    table4 = frames[TABLES[3]]
    _require_exact_columns(table4, TABLE4_COLUMNS, TABLES[3])
    metric4 = table4.rename(columns={"model": "variant"}).loc[:, METRIC_COLUMNS]
    _require_metric_contract(metric4, TABLES[3])
    _require_identity_grid(
        table4,
        ("seed", "arm", "view", "target", "model", "population"),
        set(
            product(
                seeds,
                arms,
                ("T0", "T0-T1", "T0-T2", "T0-T3"),
                ("pCR",),
                ("C", "C+F", "C+F+P1", "C+F+P3", "C+F+P4"),
                ("ftv_complete_375",),
            )
        ),
        TABLES[3],
    )
    if (
        not table4["clinical_contract"].eq("C2_full_with_treatment").all()
        or not table4["n"].eq(375).all()
    ):
        raise ValueError("table4 clinical/population contract drifted")
    delta_columns = TABLE4_COLUMNS[-6:]
    if not np.isfinite(table4.loc[:, delta_columns].to_numpy(dtype=float)).all():
        raise ValueError("table4 contains missing/non-finite paired increments")

    table7 = frames[TABLES[6]]
    comparators = ("CORE", "PERI10", "PERI20", "LOCAL_REST", "CORE_PERI")
    expected7: set[tuple[Any, ...]] = set()
    for seed, arm, comparator in product(seeds, arms, comparators):
        population = f"oracle_pair_{comparator}"
        for view, target, variant in product(
            ("T0", "T1", "T2", "T3"),
            ("HR", "HER2", "subtype_4class"),
            (comparator, "FIXED_P3"),
        ):
            expected7.add((seed, arm, view, target, variant, population))
        for view, variant in product(
            ("T0", "T0-T1", "T0-T2", "T0-T3"),
            (comparator, "FIXED_P3"),
        ):
            expected7.add((seed, arm, view, "pCR", variant, population))
    _require_identity_grid(table7, key, expected7, TABLES[6])
    pair_matched_oracle_deltas(table7)

    _validate_table8(
        frames[TABLES[7]],
        authorized=stage_b_authorized,
        stage_a_phenotype=frames[TABLES[1]],
        stage_a_pcr=frames[TABLES[2]],
    )
    return output, frames


def _validate_run_summary(
    config: Mapping[str, Any],
    lock: Mapping[str, Any],
    gates: Mapping[str, Any],
    feature_summary: Mapping[str, Any],
) -> dict[str, Any]:
    path = ROOT / "metrics" / "run_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict) or set(summary) != set(RUN_SUMMARY_KEYS):
        raise ValueError("Stage-A run summary top-level schema drifted")
    expected_scalars = {
        "schema_version": 2,
        "experiment": "spatial_heterogeneity_phenotype_audit",
        "stage": "A",
        "status": "COMPLETE",
        "branch": config["branch"],
        "n_feature_assets": 20,
        "n_full_patients": 808,
        "n_ftv_complete_patients": 375,
        "scientific_classification": gates["scientific_classification"],
        "stage_b_authorized": bool(gates["stage_b_authorized"]),
        "config_sha256": file_sha256(ROOT / "configs" / "audit.json"),
        "preregistration_lock_sha256": file_sha256(ROOT / "PREREGISTRATION_LOCK.json"),
        "public_outputs_contain_patient_level_data": False,
    }
    for name, expected in expected_scalars.items():
        if summary.get(name) != expected:
            raise ValueError(f"Stage-A run summary differs at {name}")
    for name in (
        "schema_version",
        "n_feature_assets",
        "n_full_patients",
        "n_ftv_complete_patients",
    ):
        if type(summary[name]) is not int:
            raise ValueError(f"Stage-A run summary {name} must be an integer")
    if (
        type(summary["stage_b_authorized"]) is not bool
        or type(summary["public_outputs_contain_patient_level_data"]) is not bool
    ):
        raise ValueError("Stage-A run summary Boolean fields have invalid types")
    expected_chain = preregistration_chain(lock)
    observed_chain = summary.get("preregistration_chain")
    if (
        not isinstance(observed_chain, Mapping)
        or set(observed_chain) != set(PREREGISTRATION_CHAIN_KEYS)
        or observed_chain != expected_chain
        or summary["preregistration_lock_sha256"]
        != expected_chain["active_preregistration_lock_sha256"]
    ):
        raise ValueError("Stage-A run summary preregistration chain drifted")
    elapsed = summary.get("elapsed_seconds")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not np.isfinite(float(elapsed))
        or float(elapsed) < 0.0
    ):
        raise ValueError("Stage-A run summary elapsed_seconds is invalid")

    selected_cells = lock.get("selected_cells")
    completed_cells = feature_summary.get("cells")
    if (
        not isinstance(selected_cells, Mapping)
        or len(selected_cells) != 20
        or not isinstance(completed_cells, list)
        or len(completed_cells) != 20
    ):
        raise ValueError("Stage-A run summary lacks an exact 20-cell feature closure")
    expected_features: dict[str, str] = {}
    cell_keys = {"seed", "arm", "fold", "feature_sha256", "max_parity_abs"}
    for record in completed_cells:
        if not isinstance(record, Mapping) or set(record) != cell_keys:
            raise ValueError("validated feature completion record schema drifted")
        key = (
            f"seed_{int(record['seed'])}/{str(record['arm'])}/"
            f"fold_{int(record['fold'])}"
        )
        if key in expected_features:
            raise ValueError("validated feature completion repeats a cell")
        feature_path = ROOT / "features" / key / "spatial_statistics.private.npz"
        observed_feature_sha256 = file_sha256(feature_path)
        if record.get("feature_sha256") != observed_feature_sha256:
            raise ValueError(f"validated feature bytes changed after loading: {key}")
        expected_features[key] = observed_feature_sha256
    if set(expected_features) != set(selected_cells):
        raise ValueError("validated feature cells differ from the preregistration lock")
    if summary.get("feature_asset_sha256") != expected_features:
        raise ValueError("Stage-A run summary feature hashes drifted")

    upstream_hashes = config.get("upstream_code")
    if not isinstance(upstream_hashes, Mapping):
        raise ValueError("Stage-A upstream implementation config is invalid")
    expected_reused = {
        "data_contracts": upstream_hashes.get("complementarity_data_contracts_sha256"),
        "modeling": upstream_hashes.get("complementarity_modeling_sha256"),
    }
    if summary.get("reused_implementation_sha256") != expected_reused:
        raise ValueError("Stage-A reused implementation provenance drifted")
    runtime_paths = {
        "scripts/run_audit.py": ROOT / "scripts" / "run_audit.py",
        "scripts/common.py": ROOT / "scripts" / "common.py",
    }
    expected_runtime = {
        name: file_sha256(runtime_path) for name, runtime_path in runtime_paths.items()
    }
    locked_implementations = lock.get("implementation_sha256")
    if not isinstance(locked_implementations, Mapping) or any(
        locked_implementations.get(name) != digest
        for name, digest in expected_runtime.items()
    ):
        raise ValueError("Stage-A runtime implementation differs from the lock")
    if summary.get("runtime_implementation_sha256") != expected_runtime:
        raise ValueError("Stage-A runtime implementation provenance drifted")

    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(
        RUN_SUMMARY_ARTIFACT_PATHS
    ):
        raise ValueError("Stage-A run summary artifact inventory drifted")
    for name, expected_relative in RUN_SUMMARY_ARTIFACT_PATHS.items():
        record = artifacts[name]
        if not isinstance(record, Mapping) or set(record) != set(
            RUN_SUMMARY_ARTIFACT_KEYS
        ):
            raise ValueError(f"Stage-A run summary artifact schema drifted: {name}")
        if record.get("path") != expected_relative:
            raise ValueError(f"Stage-A run summary artifact path drifted: {name}")
        artifact = ROOT / expected_relative
        if artifact.is_symlink():
            raise ValueError(f"Stage-A run summary artifact is a symlink: {name}")
        artifact = artifact.resolve(strict=True)
        if ROOT.resolve() not in artifact.parents:
            raise ValueError(
                f"Stage-A run summary artifact escaped the experiment: {name}"
            )
        private = name in RUN_SUMMARY_PRIVATE_ARTIFACTS
        if record.get("patient_level_private") is not private:
            raise ValueError(
                f"Stage-A run summary artifact privacy flag drifted: {name}"
            )
        expected_mode = 0o600 if private else 0o644
        if artifact.stat().st_mode & 0o777 != expected_mode:
            raise PermissionError(f"Stage-A run summary artifact mode drifted: {name}")
        size = record.get("size_bytes")
        if (
            type(size) is not int
            or size < 0
            or record.get("sha256") != file_sha256(artifact)
            or size != artifact.stat().st_size
        ):
            raise ValueError(f"Stage-A run summary artifact hash/size drifted: {name}")
    return summary


def _validate_report(
    *,
    final: bool,
    gates: Mapping[str, Any],
    authorization_path: Path,
    branch: str,
    chain: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any]]:
    report_path = ROOT / "reports" / "final_report.md"
    manifest_path = ROOT / "reports" / "report_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest_columns = {
        "schema_version",
        "status",
        "language",
        "scientific_classification",
        "preregistration_revision",
        "original_preregistration_commit",
        "original_preregistration_lock_sha256",
        "preregistration_amendment_sha256",
        "prior_amended_preregistration_commit",
        "prior_amended_preregistration_lock_sha256",
        "implementation_erratum_sha256",
        "preregistration_commit",
        "active_amended_preregistration_commit",
        "experiment_commit",
        "push_status",
        "input_sha256",
        "contains_patient_level_data",
        "report_sha256",
    }
    if (
        set(manifest) != expected_manifest_columns
        or manifest.get("schema_version") != 1
        or manifest.get("status") != "COMPLETE"
        or manifest.get("language") != "zh-CN"
        or manifest.get("preregistration_revision") != 2
        or manifest.get("contains_patient_level_data") is not False
    ):
        raise ValueError("report manifest contract drifted")
    if manifest.get("report_sha256") != file_sha256(report_path):
        raise ValueError("final report hash differs from report manifest")
    if manifest.get("scientific_classification") != gates["scientific_classification"]:
        raise ValueError("report classification differs from current Stage-A gates")
    input_paths = [
        ROOT / "PREREGISTRATION_LOCK.json",
        ROOT / "PREREGISTRATION_AMENDMENT.json",
        ROOT / "PREREGISTRATION_IMPLEMENTATION_ERRATUM.json",
        ROOT / "metrics" / "gates.json",
        authorization_path,
        ROOT / "metrics" / "run_summary.json",
        *(ROOT / "metrics" / name for name in TABLES),
        *(ROOT / "figures" / name for name in FIGURES),
    ]
    expected_inputs = {
        str(path.relative_to(ROOT)): file_sha256(path) for path in input_paths
    }
    if manifest.get("input_sha256") != expected_inputs:
        raise ValueError(
            "report manifest is stale relative to current scientific inputs"
        )
    commit = str(manifest.get("experiment_commit", ""))
    preregistration_commit = str(manifest.get("preregistration_commit", ""))
    original_preregistration_commit = str(
        manifest.get("original_preregistration_commit", "")
    )
    prior_amended_preregistration_commit = str(
        manifest.get("prior_amended_preregistration_commit", "")
    )
    implementation_erratum = load_preregistration_implementation_erratum()
    original_anchor, prior_amended_anchor, active_anchor = (
        preregistration_provenance_anchors()
    )
    if (
        preregistration_commit != preregistration_commit_sha()
        or preregistration_commit != chain["active_preregistration_commit"]
        or preregistration_commit != active_anchor
        or manifest.get("active_amended_preregistration_commit")
        != chain["active_preregistration_commit"]
        or original_preregistration_commit != chain["original_preregistration_commit"]
        or original_preregistration_commit != original_anchor
        or prior_amended_preregistration_commit != prior_amended_anchor
        or manifest.get("original_preregistration_lock_sha256")
        != chain["original_preregistration_lock_sha256"]
        or manifest.get("preregistration_amendment_sha256")
        != chain["preregistration_amendment_sha256"]
        or manifest.get("prior_amended_preregistration_lock_sha256")
        != implementation_erratum["prior_amended_preregistration_lock_sha256"]
        or manifest.get("implementation_erratum_sha256")
        != file_sha256(ROOT / "PREREGISTRATION_IMPLEMENTATION_ERRATUM.json")
    ):
        raise ValueError(
            "report preregistration amendment chain differs from current anchors"
        )
    push = str(manifest.get("push_status", ""))
    pending = commit == "PENDING" and push == "PENDING"
    completed = bool(
        re.fullmatch(r"[0-9a-f]{40}", commit)
        and push in {"PUSHED", "GITHUB_PUSH_FAILED"}
    )
    if not (pending or completed):
        raise ValueError("report manifest has an invalid mixed provenance state")
    if final and not completed:
        raise ValueError("final report lacks completed commit/push provenance")
    if completed:
        validate_final_git_provenance(
            commit,
            push,
            branch,
            preregistration_commit,
            original_preregistration_commit,
            prior_amended_preregistration_commit,
        )
    text = report_path.read_text(encoding="utf-8")
    if (
        f"Preregistration commit SHA：`{preregistration_commit}`" not in text
        or f"Original preregistration commit SHA：`{original_preregistration_commit}`"
        not in text
        or f"Original preregistration lock：`{chain['original_preregistration_lock_sha256']}`"
        not in text
        or f"Prior amended preregistration commit SHA：`{prior_amended_preregistration_commit}`"
        not in text
        or f"Prior amended preregistration lock：`{implementation_erratum['prior_amended_preregistration_lock_sha256']}`"
        not in text
        or f"Preregistration implementation erratum：`{file_sha256(ROOT / 'PREREGISTRATION_IMPLEMENTATION_ERRATUM.json')}`"
        not in text
        or f"Active implementation-refrozen preregistration commit SHA：`{preregistration_commit}`"
        not in text
        or f"Preregistration amendment：`{chain['preregistration_amendment_sha256']}`"
        not in text
        or f"Experiment commit SHA：`{commit}`" not in text
        or f"GitHub push status：`{push}`" not in text
    ):
        raise ValueError("report text and report-manifest provenance differ")
    return text, manifest


def validate(*, final: bool) -> dict[str, Any]:
    config = load_config(ROOT / "configs" / "audit.json", verify_inputs=True)
    lock = require_preregistration_lock(config)
    chain = preregistration_chain(lock)
    implementation_erratum = load_preregistration_implementation_erratum()
    _original_anchor, prior_amended_anchor, _active_anchor = (
        preregistration_provenance_anchors()
    )
    if _git("branch", "--show-current") != config["branch"]:
        raise ValueError("current Git branch differs from the formal branch")

    base = str(lock["git_provenance_before_freeze"]["base_head"])
    changed = sorted(
        set(filter(None, _git("diff", "--name-only", base, "--").splitlines()))
        | set(
            filter(
                None, _git("ls-files", "--others", "--exclude-standard").splitlines()
            )
        )
    )
    allowed_prefix = str(ROOT.relative_to(REPO)) + "/"
    outside = [path for path in changed if not path.startswith(allowed_prefix)]
    if outside:
        raise ValueError(f"pre-existing repository paths changed: {outside}")

    for name in PRIVATE_DIRECTORIES:
        directory = ROOT / name
        if directory.exists() and directory.stat().st_mode & 0o777 != 0o700:
            raise PermissionError(
                f"private output directory is not mode 0700: {directory}"
            )
    for name in ("metrics", "figures", "reports"):
        directory = ROOT / name
        if not directory.is_dir() or directory.stat().st_mode & 0o777 != 0o755:
            raise PermissionError(
                f"public output directory is not mode 0755: {directory}"
            )

    cache_integrity = require_cache_integrity(config, lock)
    cache_contract_path = ROOT / "metrics" / "cache_integrity_contract.json"
    cache_private_path = ROOT / "manifests" / "cache_integrity.private.json"
    oracle_contract = json.loads(
        (ROOT / "metrics" / "oracle_region_contract.json").read_text(encoding="utf-8")
    )
    if oracle_contract.get("status") != "COMPLETE":
        raise ValueError("oracle-region contract is incomplete")
    oracle_sidecar = ROOT / "manifests" / "oracle_regions.private.npz"
    _load_oracle(
        oracle_sidecar,
        ROOT / "metrics" / "oracle_region_contract.json",
        lock,
    )
    expected_cache_provenance = {
        "cache_integrity_contract_sha256": file_sha256(cache_contract_path),
        "cache_integrity_private_manifest_sha256": file_sha256(cache_private_path),
        "cache_integrity_record_set_sha256": canonical_sha256(
            cache_integrity["records"]
        ),
        "cache_integrity_primary_record_set_sha256": canonical_sha256(
            [
                record
                for record in cache_integrity["records"]
                if record["cohort"] == "primary"
            ]
        ),
    }
    for name, expected in expected_cache_provenance.items():
        if oracle_contract.get(name) != expected:
            raise ValueError(
                f"oracle contract cache-integrity provenance drifted: {name}"
            )
    feature_summary = validate_complete()
    if feature_summary.get("cell_count") != 20:
        raise ValueError("feature matrix is not exactly 20 complete cells")

    gates_path = ROOT / "metrics" / "gates.json"
    gates = json.loads(gates_path.read_text(encoding="utf-8"))
    authorization_path = ROOT / "metrics" / "stage_b_authorization.json"
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    if gates.get("status") != "COMPLETE" or set(gates.get("gates", {})) != {
        "A",
        "B",
        "C",
        "D",
    }:
        raise ValueError("Stage-A gates contract is incomplete")
    expected_authorized = bool(
        gates["gates"]["A"]["passed"] or gates["gates"]["C"]["passed"]
    )
    expected_authorization_provenance = {
        "schema_version": 2,
        "config_sha256": file_sha256(ROOT / "configs" / "audit.json"),
        "preregistration_lock_sha256": chain["active_preregistration_lock_sha256"],
        "preregistration_chain": chain,
        "stage_a_gates_sha256": file_sha256(gates_path),
    }
    if any(
        authorization.get(name) != expected
        for name, expected in expected_authorization_provenance.items()
    ):
        raise ValueError("Stage-B authorization provenance chain drifted")
    if bool(authorization.get("authorized")) != expected_authorized:
        raise ValueError("Stage-B authorization differs from Gate A OR Gate C")
    tables, table_frames = _validate_public_tables(
        config, stage_b_authorized=expected_authorized
    )
    recomputed_gates = evaluate_gates(
        config,
        table_frames[TABLES[1]],
        table_frames[TABLES[2]],
        table_frames[TABLES[3]],
        table_frames[TABLES[6]],
    )
    if canonical_sha256(gates) != canonical_sha256(recomputed_gates):
        raise ValueError(
            "published Stage-A gates differ from canonical table recomputation"
        )
    recomputed_authorization = stage_b_authorization(
        config,
        recomputed_gates,
        chain,
        expected_authorization_provenance["config_sha256"],
    )
    recomputed_authorization["stage_a_gates_sha256"] = file_sha256(gates_path)
    if canonical_sha256(authorization) != canonical_sha256(recomputed_authorization):
        raise ValueError("published Stage-B authorization differs from canonical gates")
    run_summary = _validate_run_summary(config, lock, gates, feature_summary)

    representative_metadata_path = (
        ROOT
        / "features"
        / "seed_2026"
        / "LOCAL3"
        / "fold_0"
        / "spatial_statistics.private.metadata.json"
    )
    representative_metadata_sha256 = file_sha256(representative_metadata_path)
    representative_metadata_payload = json.loads(
        representative_metadata_path.read_text(encoding="utf-8")
    )
    if file_sha256(representative_metadata_path) != representative_metadata_sha256:
        raise RuntimeError("designated feature metadata changed while loading")
    representative = representative_metadata_payload.get("representative_activation")
    if not isinstance(representative, Mapping):
        raise ValueError("designated feature metadata lacks representative provenance")
    representative_sha256 = str(representative.get("sha256", ""))
    validate_representative_asset(
        REPRESENTATIVE_PATH, expected_sha256=representative_sha256
    )
    expected_figure_sources = figure_source_inputs(
        config_sha256=file_sha256(ROOT / "configs" / "audit.json"),
        lock_sha256=file_sha256(ROOT / "PREREGISTRATION_LOCK.json"),
        table_sha256={
            "table2_phenotype_probes.csv": file_sha256(
                ROOT / "metrics" / "table2_phenotype_probes.csv"
            ),
            "table3_mri_only_pcr.csv": file_sha256(
                ROOT / "metrics" / "table3_mri_only_pcr.csv"
            ),
            "table4_clinical_ftv_incremental.csv": file_sha256(
                ROOT / "metrics" / "table4_clinical_ftv_incremental.csv"
            ),
            "table6_longitudinal_heterogeneity.csv": file_sha256(
                ROOT / "metrics" / "table6_longitudinal_heterogeneity.csv"
            ),
            "table7_oracle_regions.csv": file_sha256(
                ROOT / "metrics" / "table7_oracle_regions.csv"
            ),
        },
        representative_metadata_sha256=representative_metadata_sha256,
        representative_sha256=representative_sha256,
    )
    figure_manifest = pd.read_csv(ROOT / "metrics" / "figure_manifest.csv")
    _require_exact_columns(
        figure_manifest, FIGURE_MANIFEST_COLUMNS, "figure_manifest.csv"
    )
    if tuple(figure_manifest["figure"].astype(str)) != FIGURES:
        raise ValueError("figure manifest order/coverage drifted")
    figure_hashes: dict[str, str] = {}
    for row in figure_manifest.itertuples(index=False):
        path = ROOT / "figures" / str(row.figure)
        observed = file_sha256(path)
        if str(row.sha256) != observed or int(row.size_bytes) != path.stat().st_size:
            raise ValueError(f"figure hash differs from manifest: {path.name}")
        if str(row.source_inputs_sha256) != encode_source_inputs(
            expected_figure_sources[path.name]
        ):
            raise ValueError(f"figure source hashes are stale: {path.name}")
        figure_hashes[path.name] = observed

    if authorization.get("stage_a_gates_sha256") != file_sha256(gates_path):
        raise ValueError("Stage-B authorization is not hash-bound to gates")

    report_text, report_manifest = _validate_report(
        final=final,
        gates=gates,
        authorization_path=authorization_path,
        branch=str(config["branch"]),
        chain=chain,
    )
    for index in range(1, 13):
        if f"{index}. **" not in report_text:
            raise ValueError(f"final report omits explicit answer {index}")
    privacy = _privacy_scan(_tracked_candidate_paths())
    return {
        "schema_version": 1,
        "status": "PASS",
        "branch": config["branch"],
        "preregistration_chain": chain,
        "implementation_erratum_sha256": file_sha256(
            ROOT / "PREREGISTRATION_IMPLEMENTATION_ERRATUM.json"
        ),
        "prior_amended_preregistration_commit": prior_amended_anchor,
        "prior_amended_preregistration_lock_sha256": implementation_erratum[
            "prior_amended_preregistration_lock_sha256"
        ],
        "pre_erratum_run_state_verified": True,
        "discarded_pre_erratum_artifact_count": implementation_erratum[
            "pre_erratum_execution"
        ]["discarded_artifact_count"],
        "discarded_pre_erratum_artifact_total_bytes": implementation_erratum[
            "pre_erratum_execution"
        ]["discarded_artifact_total_bytes"],
        "base_head": base,
        "old_repository_paths_unchanged": True,
        "changed_paths": changed,
        "cache_contract_sha256": file_sha256(cache_contract_path),
        "oracle_contract_sha256": file_sha256(
            ROOT / "metrics" / "oracle_region_contract.json"
        ),
        "feature_cell_count": int(feature_summary["cell_count"]),
        "tables": tables,
        "figures": figure_hashes,
        "scientific_classification": gates["scientific_classification"],
        "stage_b_authorized": expected_authorized,
        "run_summary_sha256": file_sha256(ROOT / "metrics" / "run_summary.json"),
        "run_summary_artifact_count": len(run_summary["artifacts"]),
        "report_sha256": file_sha256(ROOT / "reports" / "final_report.md"),
        "report_experiment_commit": report_manifest.get("experiment_commit"),
        "report_push_status": report_manifest.get("push_status"),
        "privacy_scan": privacy,
        "raw_mri_or_checkpoint_tracked": False,
        "patient_level_public_output": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    result = validate(final=not args.dry_run)
    output = ROOT / "metrics" / "final_validation.json"
    if args.dry_run:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if output.exists():
        raise FileExistsError(f"refusing to overwrite final validation: {output}")
    atomic_json(result, output)
    print(
        json.dumps({"status": result["status"], "output": str(output)}, sort_keys=True)
    )


if __name__ == "__main__":
    main()

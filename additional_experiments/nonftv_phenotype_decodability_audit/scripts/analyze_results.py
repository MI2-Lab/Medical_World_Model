#!/usr/bin/env python3
"""Aggregate the frozen non-FTV probe matrix into gates, figures, and report.

This program is intentionally a public-aggregate-only consumer.  It never opens
``predictions/oof_predictions.private.csv.gz`` and rejects identifier-bearing
input schemas.  All thresholds and tie breaks come from EXPERIMENT_PLAN.md and
configs/audit.json.  The only derived threshold is the explicitly labelled
Gate-A-style raw-support descriptor used to make the categorical SPH/BPE
scorecard deterministic; it is not promoted to an additional primary gate.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping, Sequence
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from audit_core import (
    AUDIT_ROOT,
    INTERVALS,
    MAIN_REPRESENTATIONS,
    REPO_ROOT,
    VISITS,
    atomic_csv,
    atomic_json,
    file_sha256,
    load_config,
)
from freeze_preregistration import require_preregistration_lock


DEPLOYABLE_REPRESENTATIONS = ("Z1", "Z2", "Z3", "Z4")
ORACLE_REPRESENTATIONS = ("Z5", "Z6", "Z7")
ARMS = ("LOCAL0", "LOCAL3")
SEEDS = (2026, 3026)
EARLY_TIMINGS = ("T0", "T1", "T2")
EARLY_INTERVALS = ("T0->T1", "T1->T2")
TARGETS = ("FTV", "LD", "SPH", "BPE")
NONFTV_TARGETS = ("LD", "SPH", "BPE")

# Frozen in EXPERIMENT_PLAN.md section 3.2.  The array is Z,Y,X and the spacing
# is X,Y,Z, so nominal X,Y,Z coverage is (160*.9, 176*.9, 112*2) mm.  This is
# only an acquisition/crop inventory item; it is not evidence that the BPE ROI
# lies inside the crop.
C1B_SHAPE_ZYX = (112, 176, 160)
C1B_SPACING_XYZ_MM = (0.9, 0.9, 2.0)
C1B_NOMINAL_EXTENT_XYZ_MM = (
    C1B_SHAPE_ZYX[2] * C1B_SPACING_XYZ_MM[0],
    C1B_SHAPE_ZYX[1] * C1B_SPACING_XYZ_MM[1],
    C1B_SHAPE_ZYX[0] * C1B_SPACING_XYZ_MM[2],
)

PUBLIC_INPUTS = {
    "oof": Path("metrics/oof_metrics.csv"),
    "fold": Path("metrics/fold_metrics.csv"),
    "selection": Path("metrics/hyperparameter_selections.csv"),
    "coverage": Path("metrics/coverage.csv"),
    "location": Path("metrics/representation_location_comparison.csv"),
    "localization": Path("metrics/oracle_localization_comparison.csv"),
    "run_summary": Path("metrics/run_summary.json"),
    "provenance": Path("manifests/input_provenance.json"),
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=True)
    forbidden = {
        "patient_id",
        "patientid",
        "subject_id",
        "study_id",
        "label_pcr",
        "pcr",
        "y_true",
        "y_pred",
        "y_true_natural",
        "y_pred_natural",
    }
    observed = {str(column).strip().lower() for column in frame.columns}
    overlap = sorted(observed & forbidden)
    if overlap:
        raise ValueError(f"public aggregate input contains forbidden columns: {overlap}")
    return frame


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _finite(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{label}.{column} contains non-finite values")


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _as_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("attempted to serialize a non-finite gate statistic")
    return result


def _as_int(value: Any) -> int:
    return int(value)


def _candidate_order(arm: str, representation: str) -> tuple[int, int]:
    return DEPLOYABLE_REPRESENTATIONS.index(representation), ARMS.index(arm)


def _cell_label(row: Mapping[str, Any]) -> str:
    endpoint = row.get("timing") or row.get("interval") or ""
    return f"{row['arm']}/{row['representation']}@{endpoint}"


def _metric_r2_column(target_kind: str, frame: pd.DataFrame) -> str:
    if target_kind == "raw":
        return "natural_r2"
    # Gate B's amplitude guardrail is explicitly defined in reconstructed raw
    # target space.  Substituting residual-space R2 would change the gate.
    if "reconstructed_natural_r2" not in frame.columns:
        raise ValueError(
            "residual gate aggregation requires reconstructed_natural_r2; "
            "do not substitute residual-space natural_r2"
        )
    return "reconstructed_natural_r2"


def _metric_spearman_column(target_kind: str, frame: pd.DataFrame) -> str:
    if target_kind == "raw":
        return "spearman"
    if "residual_spearman" not in frame.columns:
        raise ValueError(
            "residual gate aggregation requires residual_spearman; "
            "do not infer it from an ambiguously labelled metric"
        )
    return "residual_spearman"


def _primary_rows(oof: pd.DataFrame) -> pd.Series:
    return (
        ((oof["task_type"] == "static") & (oof["input_variant"] == "current"))
        | ((oof["task_type"] == "dynamic") & (oof["input_variant"] == "difference"))
    )


def validate_inputs(
    config: Mapping[str, Any],
    frames: Mapping[str, pd.DataFrame],
    run_summary: Mapping[str, Any],
    provenance: Mapping[str, Any],
    lock_verification: Mapping[str, Any],
) -> None:
    if run_summary.get("status") != "COMPLETE" or provenance.get("status") != "COMPLETE":
        raise ValueError("formal run/provenance is not COMPLETE")
    if bool(run_summary.get("pcr_read")) or bool(run_summary.get("pcr_used_for_selection")):
        raise ValueError("pCR firewall failed")
    if bool(run_summary.get("test_used_for_alpha_selection")):
        raise ValueError("test data was used for alpha selection")
    if int(run_summary.get("oof_metric_rows", -1)) != 3276:
        raise ValueError("formal OOF matrix row count is not the frozen 3,276")
    if int(run_summary.get("feature_cells", -1)) != 20:
        raise ValueError("formal feature-cell count is not 20")
    if int(provenance.get("patient_count", -1)) != int(config["frozen"]["patient_count"]):
        raise ValueError("provenance cohort size drifted")
    observed_lock_sha = str(lock_verification.get("lock_sha256", ""))
    if not observed_lock_sha:
        raise ValueError("preregistration lock verification did not return a SHA-256")
    if str(run_summary.get("preregistration_lock_sha256", "")) != observed_lock_sha:
        raise ValueError("run summary preregistration lock hash mismatch")
    if str(provenance.get("preregistration_lock_sha256", "")) != observed_lock_sha:
        raise ValueError("input provenance preregistration lock hash mismatch")
    privacy = provenance.get("privacy", {})
    if privacy.get("pcr_column_parsed") is not False:
        raise ValueError("input provenance does not affirm that pCR was not parsed")

    oof = frames["oof"]
    _require_columns(
        oof,
        (
            "seed",
            "arm",
            "representation",
            "task_type",
            "target_definition",
            "target_kind",
            "target",
            "timing",
            "interval",
            "input_variant",
            "n",
            "n_folds",
            "spearman",
            "pearson",
            "natural_r2",
            "transformed_r2",
            "rmse",
            "mae",
            "prediction_target_variance_ratio",
            "calibration_slope",
        ),
        "oof_metrics",
    )
    if len(oof) != 3276 or not (pd.to_numeric(oof["n_folds"]) == 5).all():
        raise ValueError("OOF metric matrix is incomplete")
    if set(pd.to_numeric(oof["seed"]).astype(int)) != set(SEEDS):
        raise ValueError("OOF seed set drifted")
    if set(oof["arm"].astype(str)) != set(ARMS):
        raise ValueError("OOF arm set drifted")
    if not set(MAIN_REPRESENTATIONS).issubset(set(oof["representation"].astype(str))):
        raise ValueError("OOF representation set is incomplete")
    _finite(oof, ("n", "n_folds", "spearman", "transformed_r2"), "oof_metrics")
    raw_mask = oof["target_kind"] == "raw"
    _finite(
        oof.loc[raw_mask],
        (
            "natural_r2",
            "rmse",
            "mae",
            "prediction_target_variance_ratio",
            "calibration_slope",
        ),
        "oof raw metrics",
    )
    residual = oof["target_kind"].isin(("residual_ftv", "residual_ftv_ld"))
    if residual.any():
        _require_columns(
            oof,
            (
                "residual_spearman",
                "residual_transformed_r2",
                "residual_rmse",
                "residual_mae",
                "reconstructed_natural_r2",
                "reconstructed_natural_rmse",
                "reconstructed_natural_mae",
            ),
            "oof residual metrics",
        )
        _finite(
            oof.loc[residual],
            (
                "residual_spearman",
                "residual_transformed_r2",
                "residual_rmse",
                "residual_mae",
                "reconstructed_natural_r2",
                "reconstructed_natural_rmse",
                "reconstructed_natural_mae",
            ),
            "oof residual metrics",
        )

    selection = frames["selection"]
    _require_columns(
        selection,
        (
            "selected_alpha",
            "feature_constant_columns",
            "test_used_for_scaler",
            "test_used_for_alpha_selection",
            "test_predict_call_count",
        ),
        "hyperparameter_selections",
    )
    if selection["test_used_for_scaler"].astype(str).str.lower().isin(("true", "1")).any():
        raise ValueError("test rows entered a feature scaler")
    if selection["test_used_for_alpha_selection"].astype(str).str.lower().isin(("true", "1")).any():
        raise ValueError("test rows entered alpha selection")
    if not (pd.to_numeric(selection["test_predict_call_count"]) == 1).all():
        raise ValueError("test prediction call count is not exactly one")
    expected_alphas = {float(value) for value in config["probe"]["alphas"]}
    if not set(pd.to_numeric(selection["selected_alpha"]).astype(float)).issubset(expected_alphas):
        raise ValueError("selected alpha escaped the frozen grid")

    _require_columns(
        frames["location"],
        (
            "comparison",
            "seed",
            "arm",
            "task_type",
            "target_kind",
            "target",
            "timing",
            "interval",
            "input_variant",
            "delta_spearman",
        ),
        "representation_location_comparison",
    )
    _require_columns(
        frames["localization"],
        (
            "oracle_representation",
            "seed",
            "arm",
            "task_type",
            "target_kind",
            "target",
            "timing",
            "interval",
            "input_variant",
            "n_oracle",
            "n_full_local_matched",
            "delta_spearman",
        ),
        "oracle_localization_comparison",
    )
    if not (
        pd.to_numeric(frames["localization"]["n_oracle"])
        == pd.to_numeric(frames["localization"]["n_full_local_matched"])
    ).all():
        raise ValueError("oracle comparison is not population matched")


def authenticate_public_inputs(run_summary: Mapping[str, Any]) -> None:
    hashes = run_summary.get("core_output_sha256", {})
    for relative in (
        "metrics/oof_metrics.csv",
        "metrics/fold_metrics.csv",
        "metrics/hyperparameter_selections.csv",
        "metrics/coverage.csv",
        "metrics/representation_location_comparison.csv",
        "metrics/oracle_localization_comparison.csv",
        "manifests/input_provenance.json",
    ):
        expected = hashes.get(relative)
        if not isinstance(expected, str):
            raise ValueError(f"run summary lacks frozen hash for {relative}")
        observed = file_sha256(AUDIT_ROOT / relative)
        if observed != expected:
            raise ValueError(f"aggregate artifact hash drifted: {relative}")


def load_public_inputs() -> tuple[dict[str, pd.DataFrame], dict[str, Any], dict[str, Any]]:
    paths = {name: AUDIT_ROOT / relative for name, relative in PUBLIC_INPUTS.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"formal aggregate outputs are incomplete: {missing}")
    run_summary = _read_json(paths["run_summary"])
    authenticate_public_inputs(run_summary)
    provenance = _read_json(paths["provenance"])
    frames = {
        name: _read_csv(path)
        for name, path in paths.items()
        if name not in {"run_summary", "provenance"}
    }
    return frames, run_summary, provenance


def _evaluate_candidates(
    rows: pd.DataFrame,
    *,
    gate_id: str,
    gate_name: str,
    target: str,
    endpoint_column: str,
    endpoints: Sequence[str],
    target_kind: str,
    required_count: int,
    threshold: float,
    inclusive: bool,
    r2_guard: Callable[[np.ndarray], bool] | None,
    patient_set_sha256: str,
    is_primary_gate: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    subset = rows.loc[
        (rows["target"] == target)
        & (rows["target_kind"] == target_kind)
        & rows["arm"].isin(ARMS)
        & rows["representation"].isin(DEPLOYABLE_REPRESENTATIONS)
        & rows[endpoint_column].isin(endpoints)
    ].copy()
    r2_column = _metric_r2_column(target_kind, subset)
    spearman_column = _metric_spearman_column(target_kind, subset)
    expected = len(ARMS) * len(DEPLOYABLE_REPRESENTATIONS) * len(endpoints) * len(SEEDS)
    if len(subset) != expected:
        raise ValueError(
            f"{gate_id} matrix incomplete: observed={len(subset)} expected={expected}"
        )
    candidate_rows: list[dict[str, Any]] = []
    for arm in ARMS:
        for representation in DEPLOYABLE_REPRESENTATIONS:
            candidate = subset.loc[
                (subset["arm"] == arm) & (subset["representation"] == representation)
            ]
            endpoint_records: list[dict[str, Any]] = []
            for endpoint in endpoints:
                cell = candidate.loc[candidate[endpoint_column] == endpoint].sort_values("seed")
                observed_seeds = tuple(pd.to_numeric(cell["seed"]).astype(int))
                if observed_seeds != SEEDS:
                    raise ValueError(f"{gate_id} {arm}/{representation}/{endpoint} seed mismatch")
                spearman = cell[spearman_column].to_numpy(dtype=np.float64)
                r2 = cell[r2_column].to_numpy(dtype=np.float64)
                rank_pass = bool(np.all(spearman >= threshold)) if inclusive else bool(np.all(spearman > threshold))
                guard_pass = True if r2_guard is None else bool(r2_guard(r2))
                endpoint_records.append(
                    {
                        "endpoint": endpoint,
                        "seed_spearman": {str(seed): _as_float(value) for seed, value in zip(SEEDS, spearman, strict=True)},
                        "seed_natural_r2": {str(seed): _as_float(value) for seed, value in zip(SEEDS, r2, strict=True)},
                        "minimum_seed_spearman": _as_float(np.min(spearman)),
                        "minimum_seed_natural_r2": _as_float(np.min(r2)),
                        "mean_seed_natural_r2": _as_float(np.mean(r2)),
                        "rank_threshold_pass": rank_pass,
                        "r2_guard_pass": guard_pass,
                        "qualifies": bool(rank_pass and guard_pass),
                        "n_min": _as_int(cell["n"].min()),
                        "n_max": _as_int(cell["n"].max()),
                    }
                )
            qualifying = [record["endpoint"] for record in endpoint_records if record["qualifies"]]
            candidate_rows.append(
                {
                    "gate_id": gate_id,
                    "gate_name": gate_name,
                    "is_primary_gate": is_primary_gate,
                    "target": target,
                    "target_kind": target_kind,
                    "arm": arm,
                    "representation": representation,
                    "endpoint_type": endpoint_column,
                    "registered_endpoints": "|".join(endpoints),
                    "qualifying_endpoint_count": len(qualifying),
                    "required_endpoint_count": required_count,
                    "candidate_pass": len(qualifying) >= required_count,
                    "minimum_seed_spearman_macro": _as_float(
                        np.mean([record["minimum_seed_spearman"] for record in endpoint_records])
                    ),
                    "minimum_seed_natural_r2_macro": _as_float(
                        np.mean([record["minimum_seed_natural_r2"] for record in endpoint_records])
                    ),
                    "qualifying_endpoints": "|".join(qualifying),
                    "endpoint_evidence_json": _json_text(endpoint_records),
                    "spearman_threshold": threshold,
                    "spearman_operator": ">=" if inclusive else ">",
                    "spearman_metric": spearman_column,
                    "natural_r2_metric": r2_column,
                    "n_min": min(record["n_min"] for record in endpoint_records),
                    "n_max": max(record["n_max"] for record in endpoint_records),
                    "maximum_exclusions_from_375": 375 - min(record["n_min"] for record in endpoint_records),
                    "patient_set_sha256": patient_set_sha256,
                }
            )
    candidate_frame = pd.DataFrame(candidate_rows)
    ranked = candidate_frame.assign(
        representation_order=candidate_frame["representation"].map(
            {value: index for index, value in enumerate(DEPLOYABLE_REPRESENTATIONS)}
        ),
        arm_order=candidate_frame["arm"].map({value: index for index, value in enumerate(ARMS)}),
    ).sort_values(
        [
            "qualifying_endpoint_count",
            "minimum_seed_spearman_macro",
            "minimum_seed_natural_r2_macro",
            "representation_order",
            "arm_order",
        ],
        ascending=[False, False, False, True, True],
        kind="stable",
    )
    selected = ranked.iloc[0]
    candidate_frame["selected_candidate"] = (
        (candidate_frame["arm"] == selected["arm"])
        & (candidate_frame["representation"] == selected["representation"])
    )
    selected_records = json.loads(str(selected["endpoint_evidence_json"]))
    payload = {
        "gate_id": gate_id,
        "name": gate_name,
        "is_primary_gate": is_primary_gate,
        "target": target,
        "target_kind": target_kind,
        "passed": bool(int(selected["qualifying_endpoint_count"]) >= required_count),
        "required_endpoint_count": required_count,
        "selected_candidate": {
            "arm": str(selected["arm"]),
            "representation": str(selected["representation"]),
            "qualifying_endpoint_count": int(selected["qualifying_endpoint_count"]),
            "qualifying_endpoints": [
                value for value in str(selected["qualifying_endpoints"]).split("|") if value
            ],
            "minimum_seed_spearman_macro": _as_float(selected["minimum_seed_spearman_macro"]),
            "minimum_seed_natural_r2_macro": _as_float(selected["minimum_seed_natural_r2_macro"]),
            "endpoint_evidence": selected_records,
        },
    }
    return candidate_frame, payload


def compute_gates(
    config: Mapping[str, Any], oof: pd.DataFrame, patient_set_sha256: str
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    static = oof.loc[
        (oof["task_type"] == "static") & (oof["input_variant"] == "current")
    ].copy()
    dynamic = oof.loc[
        (oof["task_type"] == "dynamic") & (oof["input_variant"] == "difference")
    ].copy()
    candidate_frames: list[pd.DataFrame] = []
    raw_support: dict[str, dict[str, Any]] = {}
    residual_gates: dict[str, dict[str, Any]] = {}

    gate_a_config = config["gates"]["ld_observable"]
    for target in TARGETS:
        gate_id = "A" if target == "LD" else f"RAW_SUPPORT_{target}"
        gate_name = "LD_IMAGE_OBSERVABLE" if target == "LD" else f"{target}_GATE_A_STYLE_RAW_SUPPORT"
        frame, payload = _evaluate_candidates(
            static,
            gate_id=gate_id,
            gate_name=gate_name,
            target=target,
            endpoint_column="timing",
            endpoints=EARLY_TIMINGS,
            target_kind="raw",
            required_count=int(gate_a_config["minimum_early_mid_timings"]),
            threshold=float(gate_a_config["spearman_gte"]),
            inclusive=True,
            r2_guard=lambda values: bool(np.all(values > 0.0)),
            patient_set_sha256=patient_set_sha256,
            is_primary_gate=target == "LD",
        )
        candidate_frames.append(frame)
        raw_support[target] = payload

    static_gate_specs = (
        ("B", "LD_BEYOND_FTV_DECODABLE", "LD"),
        ("C", "SPH_BEYOND_FTV_DECODABLE", "SPH"),
        ("D", "BPE_BEYOND_FTV_DECODABLE", "BPE"),
    )
    for gate_id, gate_name, target in static_gate_specs:
        gate_config = config["gates"][{"LD": "ld_beyond_ftv", "SPH": "sph_beyond_ftv", "BPE": "bpe_beyond_ftv"}[target]]
        severe = float(config["gates"]["ld_beyond_ftv"]["systematically_severe_natural_r2_lte"])

        def guard(values: np.ndarray, *, family: str = target) -> bool:
            if family != "LD":
                return True
            systematically_severe = bool(np.all(values < 0.0) and np.mean(values) <= severe)
            return not systematically_severe

        frame, payload = _evaluate_candidates(
            static,
            gate_id=gate_id,
            gate_name=gate_name,
            target=target,
            endpoint_column="timing",
            endpoints=EARLY_TIMINGS,
            target_kind="residual_ftv",
            required_count=int(gate_config["minimum_early_mid_timings"]),
            threshold=float(gate_config["spearman_gt"]),
            inclusive=False,
            r2_guard=guard,
            patient_set_sha256=patient_set_sha256,
            is_primary_gate=True,
        )
        candidate_frames.append(frame)
        residual_gates[target] = payload

    dynamic_gates: dict[str, dict[str, Any]] = {}
    dynamic_config = config["gates"]["dynamic"]
    for target in NONFTV_TARGETS:
        frame, payload = _evaluate_candidates(
            dynamic,
            gate_id=f"E_{target}",
            gate_name=f"NONFTV_DYNAMIC_SIGNAL_SUPPORTED_{target}",
            target=target,
            endpoint_column="interval",
            endpoints=EARLY_INTERVALS,
            target_kind="residual_ftv",
            required_count=1,
            threshold=float(dynamic_config["spearman_gt"]),
            inclusive=False,
            r2_guard=None,
            patient_set_sha256=patient_set_sha256,
            is_primary_gate=False,
        )
        candidate_frames.append(frame)
        dynamic_gates[target] = payload

    dynamic_passes = [target for target in NONFTV_TARGETS if dynamic_gates[target]["passed"]]
    if dynamic_passes:
        target_order = {value: index for index, value in enumerate(NONFTV_TARGETS)}
        chosen_target = sorted(
            dynamic_passes,
            key=lambda target: (
                -dynamic_gates[target]["selected_candidate"]["qualifying_endpoint_count"],
                -dynamic_gates[target]["selected_candidate"]["minimum_seed_spearman_macro"],
                -dynamic_gates[target]["selected_candidate"]["minimum_seed_natural_r2_macro"],
                target_order[target],
            ),
        )[0]
    else:
        chosen_target = sorted(
            NONFTV_TARGETS,
            key=lambda target: (
                -dynamic_gates[target]["selected_candidate"]["qualifying_endpoint_count"],
                -dynamic_gates[target]["selected_candidate"]["minimum_seed_spearman_macro"],
                -dynamic_gates[target]["selected_candidate"]["minimum_seed_natural_r2_macro"],
                NONFTV_TARGETS.index(target),
            ),
        )[0]
    gate_e = {
        "gate_id": "E",
        "name": "NONFTV_DYNAMIC_SIGNAL_SUPPORTED",
        "is_primary_gate": True,
        "passed": bool(dynamic_passes),
        "qualifying_targets": dynamic_passes,
        "selected_target": chosen_target,
        "selected_candidate": dynamic_gates[chosen_target]["selected_candidate"],
        "eligible_intervals": list(EARLY_INTERVALS),
        "input_variant": "literal_difference",
    }
    gates = {
        "A": raw_support["LD"],
        "B": residual_gates["LD"],
        "C": residual_gates["SPH"],
        "D": residual_gates["BPE"],
        "E": gate_e,
    }
    result = {
        "schema_version": 1,
        "status": "COMPLETE",
        "gate_semantics": {
            "seed_rule": "same arm/representation/target/endpoint must pass in seeds 2026 and 3026",
            "undefined_metric_rule": "fail_closed",
            "candidate_tie_break": [
                "max_qualifying_endpoint_count",
                "max_early_minimum_over_seed_spearman_macro",
                "max_early_minimum_over_seed_natural_r2_macro",
                "Z1_Z2_Z3_Z4_order",
                "LOCAL0_LOCAL3_order",
            ],
            "dynamic_primary_input": "literal_z_end_minus_z_start",
            "T3_gate_use": False,
            "secondary_ftv_ld_residual_gate_use": False,
            "raw_support_note": (
                "FTV/SPH/BPE raw descriptors adapt Gate A solely for deterministic scorecard "
                "classification and are not additional primary gates"
            ),
        },
        "gates": gates,
        "per_target_dynamic": dynamic_gates,
        "per_target_raw_support": raw_support,
    }
    candidate_matrix = pd.concat(candidate_frames, ignore_index=True)
    return candidate_matrix, result, raw_support, {**residual_gates, **{f"dynamic_{k}": v for k, v in dynamic_gates.items()}}


def _two_seed_effect_rows(
    frame: pd.DataFrame,
    *,
    diagnostic_type: str,
    comparison_column: str,
    comparison_values: Sequence[str],
    patient_set_sha256: str,
) -> pd.DataFrame:
    primary = frame.loc[
        frame["target"].isin(NONFTV_TARGETS)
        & frame["target_kind"].isin(("raw", "residual_ftv"))
        & (
            ((frame["task_type"] == "static") & (frame["input_variant"] == "current"))
            | ((frame["task_type"] == "dynamic") & (frame["input_variant"] == "difference"))
        )
        & frame[comparison_column].isin(comparison_values)
    ].copy()
    identity = [
        comparison_column,
        "arm",
        "task_type",
        "target_kind",
        "target",
        "timing",
        "interval",
        "input_variant",
    ]
    rows: list[dict[str, Any]] = []
    for key, cell in primary.groupby(identity, dropna=False, sort=True):
        cell = cell.sort_values("seed")
        observed_seeds = tuple(pd.to_numeric(cell["seed"]).astype(int))
        if observed_seeds != SEEDS:
            raise ValueError(f"{diagnostic_type} effect cell seed mismatch: {key}")
        residual = key[identity.index("target_kind")] == "residual_ftv"
        delta_column = "delta_residual_spearman" if residual else "delta_spearman"
        if delta_column not in cell.columns:
            raise ValueError(f"{diagnostic_type} lacks {delta_column}")
        delta = cell[delta_column].to_numpy(dtype=np.float64)
        if diagnostic_type == "oracle_localization":
            candidate_column = "residual_spearman_oracle" if residual else "spearman_oracle"
            reference_column = (
                "residual_spearman_full_local_matched"
                if residual
                else "spearman_full_local_matched"
            )
        else:
            candidate_column = "residual_spearman_candidate" if residual else "spearman_candidate"
            reference_column = "residual_spearman_reference" if residual else "spearman_reference"
        if candidate_column not in cell.columns or reference_column not in cell.columns:
            raise ValueError(f"{diagnostic_type} lacks paired rank columns")
        candidate_rank = cell[candidate_column].to_numpy(dtype=np.float64)
        reference_rank = cell[reference_column].to_numpy(dtype=np.float64)
        n_column = "n_oracle" if "n_oracle" in cell.columns else "n_candidate"
        n = pd.to_numeric(cell[n_column]).to_numpy(dtype=np.int64)
        record = dict(zip(identity, key, strict=True))
        record.update(
            {
                "diagnostic_type": diagnostic_type,
                "evidence_scope": (
                    "primary_ftv_residual" if record["target_kind"] == "residual_ftv" else "raw_only"
                ),
                "seed_2026_delta_spearman": _as_float(delta[0]),
                "seed_3026_delta_spearman": _as_float(delta[1]),
                "minimum_seed_delta_spearman": _as_float(np.min(delta)),
                "rank_metric": "residual_spearman" if residual else "spearman",
                "seed_2026_candidate_spearman": _as_float(candidate_rank[0]),
                "seed_3026_candidate_spearman": _as_float(candidate_rank[1]),
                "seed_2026_reference_spearman": _as_float(reference_rank[0]),
                "seed_3026_reference_spearman": _as_float(reference_rank[1]),
                "candidate_both_seed_gt_0_20": bool(np.all(candidate_rank > 0.20)),
                "reference_both_seed_gt_0_20": bool(np.all(reference_rank > 0.20)),
                "spearman_gain_threshold": 0.10,
                "threshold_operator": ">=",
                "both_seed_pass": bool(np.all(delta >= 0.10)),
                "n_min": _as_int(np.min(n)),
                "n_max": _as_int(np.max(n)),
                "maximum_exclusions_from_375": 375 - _as_int(np.min(n)),
                "patient_set_sha256": patient_set_sha256,
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def compute_bottleneck_diagnostics(
    location: pd.DataFrame,
    localization: pd.DataFrame,
    patient_set_sha256: str,
) -> pd.DataFrame:
    location_rows = _two_seed_effect_rows(
        location,
        diagnostic_type="representation_location",
        comparison_column="comparison",
        comparison_values=("PREPROJECTOR_MINUS_PROJECTED", "MEAN_STD_MINUS_MEAN"),
        patient_set_sha256=patient_set_sha256,
    )
    oracle_rows = _two_seed_effect_rows(
        localization,
        diagnostic_type="oracle_localization",
        comparison_column="oracle_representation",
        comparison_values=ORACLE_REPRESENTATIONS,
        patient_set_sha256=patient_set_sha256,
    )
    union = sorted(set(location_rows.columns) | set(oracle_rows.columns))
    output = pd.concat(
        [location_rows.reindex(columns=union), oracle_rows.reindex(columns=union)],
        ignore_index=True,
    )
    order = [
        "diagnostic_type",
        "comparison",
        "oracle_representation",
        "evidence_scope",
        "target",
        "task_type",
        "timing",
        "interval",
        "arm",
    ]
    return output.sort_values([column for column in order if column in output], kind="stable").reset_index(drop=True)


def bpe_observability_audit(
    config: Mapping[str, Any], patient_set_sha256: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    contract = config["bpe_observability"]
    if contract.get("a_priori_fov_mismatch") is not True:
        raise ValueError("BPE a-priori FOV mismatch lock drifted")
    # Neither the frozen config nor the public provenance exposes a BPE source
    # ROI, side/coordinate transform, or occupancy map.  Per section 12 this is
    # therefore UNVERIFIED, not a post-hoc claim of encoder failure or disjointness.
    status = "FOV_OBSERVABILITY_UNVERIFIED"
    row = {
        "target": "BPE",
        "target_source_definition": str(contract["source"]),
        "target_source_anatomy": "contralateral_breast_fibroglandular_tissue",
        "target_source_slice_definition": "central_five_contiguous_axial_slices",
        "target_measurement": "mean_early_percent_enhancement",
        "model_input": str(contract["current_input"]),
        "c1b_tensor_shape_zyx": "x".join(str(value) for value in C1B_SHAPE_ZYX),
        "c1b_spacing_xyz_mm": "x".join(format(value, ".6g") for value in C1B_SPACING_XYZ_MM),
        "c1b_nominal_extent_xyz_mm": "x".join(format(value, ".6g") for value in C1B_NOMINAL_EXTENT_XYZ_MM),
        "c1b_crop_center": "frozen_C1B-H_crop_center_lesion_centered",
        "c1b_crop_laterality_mapping": "NOT_AVAILABLE_IN_PUBLIC_LOCK",
        "local_support_xyz_mm": "x".join(format(float(value), ".6g") for value in config["frozen"]["local_support_mm_xyz"]),
        "oracle_regions_extend_local_fov": False,
        "source_roi_coordinate_mapping_available": False,
        "source_roi_to_input_overlap_computable": False,
        "source_occupancy_threshold_for_observable": 0.99,
        "support_boundary_touch_check_available": False,
        "a_priori_fov_mismatch": True,
        "observability_status": status,
        "status_reason": (
            "target is defined in contralateral central breast tissue while all Z1-Z7 arise from "
            "a lesion-centered 64-mm support; no hash-bound BPE ROI/laterality coordinate mapping "
            "is present, so >=99% occupancy and boundary touch cannot be tested"
        ),
        "grounding_implication": "BPE_grounding_blocked_pending_broader_context_or_verified_overlap",
        "low_decodability_interpretation": "must_not_be_called_encoder_failure",
        "patient_count": int(config["frozen"]["patient_count"]),
        "patient_set_sha256": patient_set_sha256,
    }
    return pd.DataFrame([row]), row


def oracle_validity_summary(oof: pd.DataFrame, patient_set_sha256: str) -> pd.DataFrame:
    subset = oof.loc[
        oof["representation"].isin(ORACLE_REPRESENTATIONS)
        & (oof["target_kind"] == "raw")
        & (oof["target"] == "FTV")
        & _primary_rows(oof)
    ].copy()
    rows: list[dict[str, Any]] = []
    for representation in ORACLE_REPRESENTATIONS:
        for task_type, endpoints, endpoint_column in (
            ("static", VISITS, "timing"),
            ("dynamic", INTERVALS, "interval"),
        ):
            for endpoint in endpoints:
                cell = subset.loc[
                    (subset["representation"] == representation)
                    & (subset["task_type"] == task_type)
                    & (subset[endpoint_column] == endpoint)
                ]
                values = sorted(set(pd.to_numeric(cell["n"]).astype(int)))
                if len(values) != 1:
                    raise ValueError(
                        f"oracle validity is not invariant across seed/arm: {representation}/{endpoint}"
                    )
                rows.append(
                    {
                        "representation": representation,
                        "region": {"Z5": "CORE", "Z6": "PERI10", "Z7": "PERI20"}[representation],
                        "task_type": task_type,
                        "timing": endpoint if task_type == "static" else "",
                        "interval": endpoint if task_type == "dynamic" else "",
                        "valid_n": values[0],
                        "excluded_n": 375 - values[0],
                        "validity_rule": (
                            "current_visit_oracle_valid"
                            if task_type == "static"
                            else "both_interval_endpoints_oracle_valid_and_FTV_change_eligible"
                        ),
                        "patient_set_sha256": patient_set_sha256,
                    }
                )
    output = pd.DataFrame(rows)
    expected_static = {
        "Z5": [375, 375, 374, 374],
        "Z6": [375, 375, 374, 375],
        "Z7": [375, 375, 375, 375],
    }
    expected_dynamic = {
        "Z5": [375, 374, 373],
        "Z6": [375, 374, 374],
        "Z7": [375, 375, 375],
    }
    for representation in ORACLE_REPRESENTATIONS:
        observed_static = output.loc[
            (output["representation"] == representation) & (output["task_type"] == "static"),
            "valid_n",
        ].tolist()
        observed_dynamic = output.loc[
            (output["representation"] == representation) & (output["task_type"] == "dynamic"),
            "valid_n",
        ].tolist()
        if observed_static != expected_static[representation] or observed_dynamic != expected_dynamic[representation]:
            raise ValueError(f"oracle validity coverage drifted for {representation}")
    return output


def probe_integrity_summary(
    coverage: pd.DataFrame,
    selection: pd.DataFrame,
    patient_set_sha256: str,
) -> pd.DataFrame:
    identity = ["task_type", "target_kind", "input_variant", "representation"]
    _require_columns(
        coverage,
        (*identity, "fold", "n_train", "n_validation", "n_test", "joint_valid_total"),
        "coverage",
    )
    _require_columns(selection, (*identity, "selected_alpha", "feature_constant_columns"), "selection")
    rows: list[dict[str, Any]] = []
    for key, group in coverage.groupby(identity, dropna=False, sort=True):
        select = selection
        for column, value in zip(identity, key, strict=True):
            select = select.loc[select[column].fillna("") == ("" if pd.isna(value) else value)]
        if len(select) != len(group):
            raise ValueError(f"coverage/selection row mismatch for {key}")
        alpha_counts = (
            pd.to_numeric(select["selected_alpha"])
            .map(lambda value: format(float(value), ".10g"))
            .value_counts(sort=False)
            .sort_index()
            .to_dict()
        )
        rows.append(
            {
                **dict(zip(identity, key, strict=True)),
                "probe_fold_rows": len(group),
                "distinct_outer_folds": int(pd.to_numeric(group["fold"]).nunique()),
                "n_train_min": int(pd.to_numeric(group["n_train"]).min()),
                "n_validation_min": int(pd.to_numeric(group["n_validation"]).min()),
                "n_test_min": int(pd.to_numeric(group["n_test"]).min()),
                "n_test_max": int(pd.to_numeric(group["n_test"]).max()),
                "joint_valid_total_min": int(pd.to_numeric(group["joint_valid_total"]).min()),
                "joint_valid_total_max": int(pd.to_numeric(group["joint_valid_total"]).max()),
                "maximum_exclusions_from_375": 375 - int(pd.to_numeric(group["joint_valid_total"]).min()),
                "selected_alpha_counts_json": _json_text(alpha_counts),
                "feature_constant_columns_max": int(pd.to_numeric(select["feature_constant_columns"]).max()),
                "test_used_for_scaler": False,
                "test_used_for_alpha_selection": False,
                "test_predict_call_count": 1,
                "patient_set_sha256": patient_set_sha256,
            }
        )
    return pd.DataFrame(rows).sort_values(identity, kind="stable").reset_index(drop=True)


def _effect_pass(
    diagnostics: pd.DataFrame,
    target: str,
    *,
    comparison: str | None = None,
    oracle: bool = False,
    residual_only: bool = True,
) -> bool:
    rows = diagnostics.loc[(diagnostics["target"] == target) & diagnostics["both_seed_pass"]].copy()
    if residual_only:
        rows = rows.loc[rows["evidence_scope"] == "primary_ftv_residual"]
    if oracle:
        rows = rows.loc[rows["diagnostic_type"] == "oracle_localization"]
    elif comparison is not None:
        rows = rows.loc[
            (rows["diagnostic_type"] == "representation_location")
            & (rows["comparison"] == comparison)
        ]
    return not rows.empty


def _effect_bottleneck_pass(
    diagnostics: pd.DataFrame,
    target: str,
    *,
    comparison: str | None = None,
    oracle: bool = False,
) -> bool:
    rows = diagnostics.loc[
        (diagnostics["target"] == target)
        & diagnostics["both_seed_pass"]
        & (diagnostics["evidence_scope"] == "primary_ftv_residual")
        & diagnostics["candidate_both_seed_gt_0_20"]
        & ~diagnostics["reference_both_seed_gt_0_20"]
    ].copy()
    if oracle:
        rows = rows.loc[rows["diagnostic_type"] == "oracle_localization"]
    elif comparison is not None:
        rows = rows.loc[
            (rows["diagnostic_type"] == "representation_location")
            & (rows["comparison"] == comparison)
        ]
    return not rows.empty


def _passed_effect_labels(diagnostics: pd.DataFrame, target: str) -> str:
    rows = diagnostics.loc[
        (diagnostics["target"] == target)
        & diagnostics["both_seed_pass"]
        & (diagnostics["evidence_scope"] == "primary_ftv_residual")
    ].copy()
    labels: list[str] = []
    for _, row in rows.iterrows():
        endpoint = str(row["timing"]) if pd.notna(row.get("timing")) and str(row.get("timing")) else str(row.get("interval", ""))
        comparison = (
            str(row["oracle_representation"])
            if row["diagnostic_type"] == "oracle_localization"
            else str(row["comparison"])
        )
        labels.append(f"{comparison}:{row['arm']}:{endpoint}")
    return "|".join(sorted(set(labels)))


def _all_locations_residual_weak(oof: pd.DataFrame, target: str) -> bool:
    """Fail-closed absolute-weakness check across Z1-Z7.

    This is diagnostic, not Gate C/D: a location counts as non-weak if any
    static early/mid or early literal-difference cell has both frozen seeds with
    residual Spearman > .20.  Encoder/Class-D language is authorized only when
    every Z1-Z7 arm/location is weak under that explicit rule.
    """

    subset = oof.loc[
        (oof["target"] == target)
        & (oof["target_kind"] == "residual_ftv")
        & oof["representation"].isin(MAIN_REPRESENTATIONS)
        & (
            (
                (oof["task_type"] == "static")
                & (oof["input_variant"] == "current")
                & oof["timing"].isin(EARLY_TIMINGS)
            )
            | (
                (oof["task_type"] == "dynamic")
                & (oof["input_variant"] == "difference")
                & oof["interval"].isin(EARLY_INTERVALS)
            )
        )
    ].copy()
    endpoint = np.where(subset["task_type"] == "static", subset["timing"], subset["interval"])
    subset = subset.assign(_endpoint=endpoint)
    expected_cells = len(ARMS) * len(MAIN_REPRESENTATIONS) * (len(EARLY_TIMINGS) + len(EARLY_INTERVALS))
    groups = list(subset.groupby(["arm", "representation", "task_type", "_endpoint"], sort=True))
    if len(groups) != expected_cells:
        raise ValueError(f"absolute Z1-Z7 residual weakness matrix incomplete for {target}")
    for _, cell in groups:
        cell = cell.sort_values("seed")
        if tuple(pd.to_numeric(cell["seed"]).astype(int)) != SEEDS:
            raise ValueError(f"absolute residual weakness seed mismatch for {target}")
        if bool(np.all(cell["residual_spearman"].to_numpy(dtype=np.float64) > 0.20)):
            return False
    return True


def make_scorecard(
    oof: pd.DataFrame,
    gate_results: Mapping[str, Any],
    raw_support: Mapping[str, Mapping[str, Any]],
    residual_and_dynamic: Mapping[str, Mapping[str, Any]],
    diagnostics: pd.DataFrame,
    bpe_status: str,
    patient_set_sha256: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    classifications: dict[str, str] = {}
    bottlenecks: dict[str, str] = {}
    for target in NONFTV_TARGETS:
        raw_pass = bool(raw_support[target]["passed"])
        residual_pass = bool(residual_and_dynamic[target]["passed"])
        dynamic_pass = bool(residual_and_dynamic[f"dynamic_{target}"]["passed"])
        projection = _effect_pass(
            diagnostics, target, comparison="PREPROJECTOR_MINUS_PROJECTED"
        )
        pooling = _effect_pass(diagnostics, target, comparison="MEAN_STD_MINUS_MEAN")
        localization = _effect_pass(diagnostics, target, oracle=True)
        projection_bottleneck = _effect_bottleneck_pass(
            diagnostics, target, comparison="PREPROJECTOR_MINUS_PROJECTED"
        )
        pooling_bottleneck = _effect_bottleneck_pass(
            diagnostics, target, comparison="MEAN_STD_MINUS_MEAN"
        )
        localization_bottleneck = _effect_bottleneck_pass(
            diagnostics, target, oracle=True
        )
        all_locations_weak = _all_locations_residual_weak(oof, target)
        if target == "BPE" and bpe_status != "OBSERVABLE_IN_FOV":
            bottleneck = "input_observability"
        elif localization_bottleneck:
            bottleneck = "localization"
        elif pooling_bottleneck:
            bottleneck = "pooling/statistic"
        elif projection_bottleneck:
            bottleneck = "projection"
        elif not residual_pass and all_locations_weak and not any((projection, pooling, localization)):
            bottleneck = "encoder/current_feature_map"
        else:
            bottleneck = "mixed_or_unresolved"

        fov_compatible = target != "BPE" or bpe_status == "OBSERVABLE_IN_FOV"
        if raw_pass and residual_pass and dynamic_pass and fov_compatible:
            classification = "Class A — STRONG GROUNDING CANDIDATE"
        elif raw_pass and not residual_pass:
            classification = "Class B — RESPONSE-ONLY CANDIDATE"
        elif not residual_pass and localization and fov_compatible:
            classification = "Class C — SPATIALLY LOCALIZED TARGET"
        elif (
            not raw_pass
            and not residual_pass
            and all_locations_weak
            and not any((projection, pooling, localization))
            and fov_compatible
        ):
            classification = "Class D — CURRENTLY NOT IMAGE-OBSERVABLE"
        elif target == "BPE" and not fov_compatible:
            classification = "FOV-BLOCKED — A/D CLASSIFICATION NOT AUTHORIZED"
        else:
            classification = "MIXED OR UNRESOLVED"
        classifications[target] = classification
        bottlenecks[target] = bottleneck
        rows.append(
            {
                "target": target,
                "joint_N_evidence": "Goal6_joint_N_and_joint_N_res_increment_supported_n384",
                "family_specific_evidence": {
                    "LD": "descriptive_strongest_family_not_individually_pCR_tested",
                    "SPH": "descriptive_weaker_family_not_individually_pCR_tested",
                    "BPE": "descriptive_weakest_family_not_individually_pCR_tested",
                }[target],
                "raw_image_observability_rule": "RAW_IMAGE_SUPPORT_FOR_CLASSIFICATION_Gate-A-style_not_primary_gate",
                "raw_image_observable": raw_pass,
                "raw_best_candidate": (
                    f"{raw_support[target]['selected_candidate']['arm']}/"
                    f"{raw_support[target]['selected_candidate']['representation']}"
                ),
                "raw_qualifying_timings": "|".join(raw_support[target]["selected_candidate"]["qualifying_endpoints"]),
                "beyond_ftv_gate": {"LD": "B", "SPH": "C", "BPE": "D"}[target],
                "beyond_ftv_decodable": residual_pass,
                "residual_best_candidate": (
                    f"{residual_and_dynamic[target]['selected_candidate']['arm']}/"
                    f"{residual_and_dynamic[target]['selected_candidate']['representation']}"
                ),
                "residual_qualifying_timings": "|".join(
                    residual_and_dynamic[target]["selected_candidate"]["qualifying_endpoints"]
                ),
                "longitudinal_literal_difference_supported": dynamic_pass,
                "dynamic_best_candidate": (
                    f"{residual_and_dynamic[f'dynamic_{target}']['selected_candidate']['arm']}/"
                    f"{residual_and_dynamic[f'dynamic_{target}']['selected_candidate']['representation']}"
                ),
                "dynamic_qualifying_intervals": "|".join(
                    residual_and_dynamic[f"dynamic_{target}"]["selected_candidate"]["qualifying_endpoints"]
                ),
                "both_seed_stability": bool(raw_pass or residual_pass or dynamic_pass),
                "input_observability": (
                    bpe_status if target == "BPE" else "LESION_DERIVED_TARGET_COMPATIBLE_WITH_LOCAL_INPUT"
                ),
                "projection_bottleneck_flag": projection,
                "pooling_bottleneck_flag": pooling,
                "localization_flag": localization,
                "projection_primary_mapping": projection_bottleneck,
                "pooling_primary_mapping": pooling_bottleneck,
                "localization_primary_mapping": localization_bottleneck,
                "passed_residual_effect_cells": _passed_effect_labels(diagnostics, target),
                "all_Z1_to_Z7_primary_residual_weak": all_locations_weak,
                "primary_bottleneck": bottleneck,
                "scientific_classification": classification,
                "grounding_eligibility": (
                    "eligible_for_next_stage_design_not_automatic_training"
                    if classification.startswith("Class A")
                    else "not_authorized_for_direct_phenotype_grounding"
                ),
                "patient_count": 375,
                "patient_set_sha256": patient_set_sha256,
            }
        )

    if classifications["LD"].startswith("Class A"):
        recommendation = "FTV + LD"
        rule = 1
        rationale = "LD satisfies raw, beyond-FTV, early dynamic, same-candidate two-seed, and FOV-compatible criteria"
    elif any(
        _effect_bottleneck_pass(diagnostics, target, oracle=True)
        for target in ("LD", "SPH")
    ):
        recommendation = "FTV + regional LD/SPH"
        rule = 2
        rationale = "No Class-A LD; LD/SPH primary residual has a two-seed Oracle localization gain"
    elif bpe_status in {"TARGET_NOT_OBSERVABLE_IN_LOCAL_FOV", "FOV_OBSERVABILITY_UNVERIFIED"}:
        recommendation = "Need broader-context phenotype branch"
        rule = 3
        rationale = "Higher-priority rules failed and BPE source observability is not established in lesion-centered LOCAL FOV"
    else:
        recommendation = "FTV only; phenotype target not yet image-observable"
        rule = 4
        rationale = "Higher-priority rules failed and no residual phenotype target is sufficiently observable"
    recommendation_frame = pd.DataFrame(
        [
            {
                "recommendation": recommendation,
                "priority_rule_applied": rule,
                "rationale": rationale,
                "automatically_start_training": False,
                "ld_classification": classifications["LD"],
                "sph_classification": classifications["SPH"],
                "bpe_classification": classifications["BPE"],
                "ld_bottleneck": bottlenecks["LD"],
                "sph_bottleneck": bottlenecks["SPH"],
                "bpe_bottleneck": bottlenecks["BPE"],
                "patient_count": 375,
                "patient_set_sha256": patient_set_sha256,
            }
        ]
    )
    return pd.DataFrame(rows), recommendation_frame


def descriptive_best_cells(oof: pd.DataFrame, patient_set_sha256: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    specifications = []
    for target in TARGETS:
        specifications.append(("static_raw_early", "static", "raw", target, "timing", EARLY_TIMINGS, "current"))
    for target in NONFTV_TARGETS:
        specifications.append(("static_residual_early", "static", "residual_ftv", target, "timing", EARLY_TIMINGS, "current"))
        specifications.append(("dynamic_residual_early", "dynamic", "residual_ftv", target, "interval", EARLY_INTERVALS, "difference"))
    for target in TARGETS:
        specifications.append(("dynamic_raw_early", "dynamic", "raw", target, "interval", EARLY_INTERVALS, "difference"))
    for analysis, task_type, target_kind, target, endpoint_column, endpoints, input_variant in specifications:
        subset = oof.loc[
            (oof["task_type"] == task_type)
            & (oof["target_kind"] == target_kind)
            & (oof["target"] == target)
            & (oof["input_variant"] == input_variant)
            & oof["representation"].isin(DEPLOYABLE_REPRESENTATIONS)
            & oof["arm"].isin(ARMS)
            & oof[endpoint_column].isin(endpoints)
        ].copy()
        spearman_column = _metric_spearman_column(target_kind, subset)
        r2_column = _metric_r2_column(target_kind, subset)
        cell_rows: list[dict[str, Any]] = []
        for (arm, representation, endpoint), cell in subset.groupby(
            ["arm", "representation", endpoint_column], sort=True
        ):
            cell = cell.sort_values("seed")
            if tuple(pd.to_numeric(cell["seed"]).astype(int)) != SEEDS:
                raise ValueError(f"descriptive cell seed mismatch: {analysis}/{target}")
            rank = cell[spearman_column].to_numpy(dtype=np.float64)
            r2 = cell[r2_column].to_numpy(dtype=np.float64)
            cell_rows.append(
                {
                    "arm": arm,
                    "representation": representation,
                    "endpoint": endpoint,
                    "minimum_seed_spearman": _as_float(np.min(rank)),
                    "mean_seed_spearman": _as_float(np.mean(rank)),
                    "minimum_seed_natural_r2": _as_float(np.min(r2)),
                    "mean_seed_natural_r2": _as_float(np.mean(r2)),
                    "seed_2026_spearman": _as_float(rank[0]),
                    "seed_3026_spearman": _as_float(rank[1]),
                    "seed_2026_natural_r2": _as_float(r2[0]),
                    "seed_3026_natural_r2": _as_float(r2[1]),
                    "n_min": int(cell["n"].min()),
                    "n_max": int(cell["n"].max()),
                }
            )
        best = sorted(
            cell_rows,
            key=lambda row: (
                -row["minimum_seed_spearman"],
                -row["minimum_seed_natural_r2"],
                *_candidate_order(str(row["arm"]), str(row["representation"])),
                list(endpoints).index(str(row["endpoint"])),
            ),
        )[0]
        rows.append(
            {
                "analysis": analysis,
                "task_type": task_type,
                "target_kind": target_kind,
                "target": target,
                "endpoint": best["endpoint"],
                "arm": best["arm"],
                "representation": best["representation"],
                "spearman_metric": spearman_column,
                "natural_r2_metric": r2_column,
                **{key: value for key, value in best.items() if key not in {"endpoint", "arm", "representation"}},
                "maximum_exclusions_from_375": 375 - best["n_min"],
                "patient_set_sha256": patient_set_sha256,
                "selection_scope": "descriptive_summary_only_not_checkpoint_or_probe_selection",
            }
        )
    return pd.DataFrame(rows)


def dynamic_interval_best_cells(oof: pd.DataFrame, patient_set_sha256: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target_kind, targets in (("raw", TARGETS), ("residual_ftv", NONFTV_TARGETS)):
        metric = _metric_spearman_column(target_kind, oof)
        r2_metric = _metric_r2_column(target_kind, oof)
        for target in targets:
            for interval in INTERVALS:
                subset = oof.loc[
                    (oof["task_type"] == "dynamic")
                    & (oof["target_definition"] == "adjacent_percent_change_new_extension")
                    & (oof["target_kind"] == target_kind)
                    & (oof["target"] == target)
                    & (oof["interval"] == interval)
                    & (oof["input_variant"] == "difference")
                    & oof["arm"].isin(ARMS)
                    & oof["representation"].isin(DEPLOYABLE_REPRESENTATIONS)
                ].copy()
                candidates: list[dict[str, Any]] = []
                for arm in ARMS:
                    for representation in DEPLOYABLE_REPRESENTATIONS:
                        cell = subset.loc[
                            (subset["arm"] == arm)
                            & (subset["representation"] == representation)
                        ].sort_values("seed")
                        if tuple(pd.to_numeric(cell["seed"]).astype(int)) != SEEDS:
                            raise ValueError(
                                f"dynamic interval summary seed mismatch: {target_kind}/{target}/{interval}"
                            )
                        rank = cell[metric].to_numpy(dtype=np.float64)
                        r2 = cell[r2_metric].to_numpy(dtype=np.float64)
                        candidates.append(
                            {
                                "arm": arm,
                                "representation": representation,
                                "minimum_seed_spearman": _as_float(np.min(rank)),
                                "mean_seed_spearman": _as_float(np.mean(rank)),
                                "minimum_seed_natural_r2": _as_float(np.min(r2)),
                                "mean_seed_natural_r2": _as_float(np.mean(r2)),
                                "seed_2026_spearman": _as_float(rank[0]),
                                "seed_3026_spearman": _as_float(rank[1]),
                                "seed_2026_natural_r2": _as_float(r2[0]),
                                "seed_3026_natural_r2": _as_float(r2[1]),
                                "n_min": int(cell["n"].min()),
                                "n_max": int(cell["n"].max()),
                            }
                        )
                best = sorted(
                    candidates,
                    key=lambda row: (
                        -row["minimum_seed_spearman"],
                        -row["minimum_seed_natural_r2"],
                        *_candidate_order(str(row["arm"]), str(row["representation"])),
                    ),
                )[0]
                rows.append(
                    {
                        "task_type": "dynamic",
                        "target_definition": "adjacent_percent_change_new_extension",
                        "target_kind": target_kind,
                        "target": target,
                        "interval": interval,
                        "input_variant": "difference",
                        "spearman_metric": metric,
                        "natural_r2_metric": r2_metric,
                        **best,
                        "maximum_exclusions_from_375": 375 - best["n_min"],
                        "patient_set_sha256": patient_set_sha256,
                        "selection_scope": "descriptive_best_fixed_candidate_per_interval_not_a_gate",
                    }
                )
    return pd.DataFrame(rows)


def dynamic_macro_table(oof: pd.DataFrame, patient_set_sha256: str) -> pd.DataFrame:
    dynamic = oof.loc[oof["task_type"] == "dynamic"].copy()
    if "target_definition" not in dynamic.columns:
        raise ValueError("dynamic OOF metrics lacks target_definition")
    identity = [
        "seed",
        "arm",
        "representation",
        "matched_reference_for",
        "task_type",
        "target_kind",
        "target",
        "input_variant",
        "metric_space",
        "feature_dim",
        "target_definition",
    ]
    metrics = [
        column
        for column in (
            "spearman",
            "residual_spearman",
            "pearson",
            "natural_r2",
            "transformed_r2",
            "residual_transformed_r2",
            "reconstructed_natural_r2",
            "rmse",
            "mae",
            "reconstructed_natural_rmse",
            "reconstructed_natural_mae",
            "prediction_target_variance_ratio",
            "calibration_slope",
            "calibration_intercept",
        )
        if column in dynamic.columns
    ]
    rows: list[dict[str, Any]] = []
    for key, group in dynamic.groupby(identity, dropna=False, sort=True):
        observed = tuple(
            value
            for value in INTERVALS
            if value in set(group["interval"].astype(str))
        )
        if observed != INTERVALS or len(group) != len(INTERVALS):
            raise ValueError(f"dynamic macro interval matrix incomplete for {key}")
        row: dict[str, Any] = {
            **dict(zip(identity, key, strict=True)),
            "interval": "macro_all_3_intervals",
            "macro_definition": "unweighted_mean_of_3_preregistered_interval_metrics",
            "interval_count": 3,
            "intervals": "|".join(INTERVALS),
            "n_sum_descriptive": int(pd.to_numeric(group["n"]).sum()),
            "n_min": int(pd.to_numeric(group["n"]).min()),
            "n_max": int(pd.to_numeric(group["n"]).max()),
            "maximum_exclusions_from_375": 375 - int(pd.to_numeric(group["n"]).min()),
            "patient_set_sha256": patient_set_sha256,
            "gate_e_use": False,
        }
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            # Residual-only fields are intentionally empty on raw rows.
            row[metric] = float(values.mean()) if values.notna().all() else np.nan
        rows.append(row)
        early = group.loc[group["interval"].isin(EARLY_INTERVALS)]
        early_row = dict(row)
        early_row.update(
            {
                "interval": "macro_early_2_intervals",
                "macro_definition": "unweighted_mean_of_T0toT1_and_T1toT2_metrics_descriptive_only",
                "interval_count": 2,
                "intervals": "|".join(EARLY_INTERVALS),
                "n_sum_descriptive": int(pd.to_numeric(early["n"]).sum()),
                "n_min": int(pd.to_numeric(early["n"]).min()),
                "n_max": int(pd.to_numeric(early["n"]).max()),
                "maximum_exclusions_from_375": 375 - int(pd.to_numeric(early["n"]).min()),
            }
        )
        for metric in metrics:
            values = pd.to_numeric(early[metric], errors="coerce")
            early_row[metric] = float(values.mean()) if values.notna().all() else np.nan
        rows.append(early_row)
    return pd.DataFrame(rows).sort_values(
        ["target_kind", "target", "input_variant", "seed", "arm", "representation", "interval"],
        kind="stable",
    ).reset_index(drop=True)


def _heatmap_matrix(
    oof: pd.DataFrame,
    *,
    task_type: str,
    target_kind: str,
    targets: Sequence[str],
    endpoints: Sequence[str],
    endpoint_column: str,
    input_variant: str,
) -> tuple[np.ndarray, list[str], list[str]]:
    subset = oof.loc[
        (oof["task_type"] == task_type)
        & (oof["target_kind"] == target_kind)
        & oof["target"].isin(targets)
        & oof[endpoint_column].isin(endpoints)
        & (oof["input_variant"] == input_variant)
        & oof["representation"].isin(MAIN_REPRESENTATIONS)
    ].copy()
    metric = _metric_spearman_column(target_kind, subset)
    row_labels = [f"{target} {endpoint}" for target in targets for endpoint in endpoints]
    column_labels = [f"{arm}/{representation}" for arm in ARMS for representation in MAIN_REPRESENTATIONS]
    matrix = np.full((len(row_labels), len(column_labels)), np.nan, dtype=np.float64)
    for row_index, (target, endpoint) in enumerate(
        (pair for target in targets for pair in ((target, endpoint) for endpoint in endpoints))
    ):
        for column_index, (arm, representation) in enumerate(
            (pair for arm in ARMS for pair in ((arm, representation) for representation in MAIN_REPRESENTATIONS))
        ):
            cell = subset.loc[
                (subset["target"] == target)
                & (subset[endpoint_column] == endpoint)
                & (subset["arm"] == arm)
                & (subset["representation"] == representation)
            ]
            if len(cell) != len(SEEDS):
                raise ValueError(f"figure matrix cell incomplete: {target}/{endpoint}/{arm}/{representation}")
            matrix[row_index, column_index] = float(cell[metric].mean())
    return matrix, row_labels, column_labels


def _atomic_figure(path: Path, figure: plt.Figure) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        figure.savefig(
            temporary,
            dpi=180,
            bbox_inches="tight",
            metadata={"Software": "nonftv_phenotype_decodability_audit"},
        )
        temporary.replace(path)
        path.chmod(0o644)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(figure)


def _draw_heatmap(
    matrix: np.ndarray,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    *,
    title: str,
    path: Path,
    vmin: float,
    vmax: float,
) -> None:
    figure, axis = plt.subplots(
        figsize=(max(11.0, 0.72 * len(column_labels)), max(4.8, 0.40 * len(row_labels)))
    )
    masked = np.ma.masked_invalid(matrix)
    palette = plt.get_cmap("coolwarm").copy()
    palette.set_bad("#d9d9d9")
    image = axis.imshow(masked, aspect="auto", cmap=palette, vmin=vmin, vmax=vmax)
    axis.set_xticks(np.arange(len(column_labels)), labels=column_labels, rotation=55, ha="right")
    axis.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    axis.set_title(title)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            if np.isfinite(matrix[row, column]):
                value = matrix[row, column]
                color = "white" if abs(value) >= 0.34 else "black"
                axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=6.5, color=color)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.027, pad=0.02)
    colorbar.set_label("Spearman rho (mean across frozen seeds)")
    axis.set_xlabel("arm / frozen representation")
    axis.set_ylabel("target / endpoint")
    figure.text(
        0.5,
        0.005,
        "Z5-Z7 are mask-dependent Oracle diagnostics; T3 is late/pre-surgery.",
        ha="center",
        fontsize=8,
    )
    _atomic_figure(path, figure)


def generate_figures(oof: pd.DataFrame, diagnostics: pd.DataFrame) -> list[Path]:
    outputs: list[Path] = []
    specifications = (
        (
            "static_raw_decodability.png",
            "Static raw target decodability",
            "static",
            "raw",
            TARGETS,
            VISITS,
            "timing",
            "current",
            -0.30,
            0.80,
        ),
        (
            "static_ftv_residual_decodability.png",
            "Static primary FTV-residual decodability",
            "static",
            "residual_ftv",
            NONFTV_TARGETS,
            VISITS,
            "timing",
            "current",
            -0.30,
            0.50,
        ),
        (
            "dynamic_ftv_residual_difference_decodability.png",
            "Dynamic primary FTV-residual decodability (literal latent difference)",
            "dynamic",
            "residual_ftv",
            NONFTV_TARGETS,
            INTERVALS,
            "interval",
            "difference",
            -0.30,
            0.50,
        ),
    )
    for filename, title, task_type, target_kind, targets, endpoints, endpoint_column, input_variant, vmin, vmax in specifications:
        matrix, row_labels, column_labels = _heatmap_matrix(
            oof,
            task_type=task_type,
            target_kind=target_kind,
            targets=targets,
            endpoints=endpoints,
            endpoint_column=endpoint_column,
            input_variant=input_variant,
        )
        path = AUDIT_ROOT / "figures" / filename
        _draw_heatmap(
            matrix,
            row_labels,
            column_labels,
            title=title,
            path=path,
            vmin=vmin,
            vmax=vmax,
        )
        outputs.append(path)

    diagnostic_columns = (
        ("PREPROJECTOR_MINUS_PROJECTED", "Z2-Z1"),
        ("MEAN_STD_MINUS_MEAN", "Z4-Z3"),
        ("Z5", "CORE-Z4"),
        ("Z6", "PERI10-Z4"),
        ("Z7", "PERI20-Z4"),
    )
    matrix = np.full((len(NONFTV_TARGETS), len(diagnostic_columns)), np.nan, dtype=np.float64)
    residual = diagnostics.loc[diagnostics["evidence_scope"] == "primary_ftv_residual"]
    for target_index, target in enumerate(NONFTV_TARGETS):
        for column_index, (identifier, _) in enumerate(diagnostic_columns):
            if identifier.startswith("Z"):
                cells = residual.loc[
                    (residual["target"] == target)
                    & (residual["diagnostic_type"] == "oracle_localization")
                    & (residual["oracle_representation"] == identifier)
                ]
            else:
                cells = residual.loc[
                    (residual["target"] == target)
                    & (residual["diagnostic_type"] == "representation_location")
                    & (residual["comparison"] == identifier)
                ]
            if not cells.empty:
                matrix[target_index, column_index] = float(cells["minimum_seed_delta_spearman"].max())
    path = AUDIT_ROOT / "figures" / "bottleneck_maximum_two_seed_gain.png"
    _draw_heatmap(
        matrix,
        list(NONFTV_TARGETS),
        [label for _, label in diagnostic_columns],
        title="Maximum minimum-over-seeds residual Spearman gain (diagnostic)",
        path=path,
        vmin=-0.20,
        vmax=0.30,
    )
    outputs.append(path)
    return outputs


def _metric_sentence(row: pd.Series) -> str:
    r2_label = str(row["natural_r2_metric"])
    r2_display = (
        "reconstructed target R²"
        if r2_label == "reconstructed_natural_r2"
        else "natural R²"
    )


def _interval_sentence(row: pd.Series) -> str:
    return (
        f"{row['interval']} {row['arm']}/{row['representation']} "
        f"rho={row['seed_2026_spearman']:.3f}/{row['seed_3026_spearman']:.3f}, "
        f"R²={row['seed_2026_natural_r2']:.3f}/{row['seed_3026_natural_r2']:.3f}, "
        f"n={int(row['n_min'])}–{int(row['n_max'])}"
    )
    return (
        f"{row['arm']}/{row['representation']}，{row['endpoint']}："
        f"两 seed rho={row['seed_2026_spearman']:.3f}/{row['seed_3026_spearman']:.3f}，"
        f"{r2_display}={row['seed_2026_natural_r2']:.3f}/{row['seed_3026_natural_r2']:.3f}，"
        f"n={int(row['n_min'])}–{int(row['n_max'])}"
    )


def _best_lookup(best: pd.DataFrame, analysis: str, target: str) -> pd.Series:
    rows = best.loc[(best["analysis"] == analysis) & (best["target"] == target)]
    if len(rows) != 1:
        raise ValueError(f"best-cell lookup failed: {analysis}/{target}")
    return rows.iloc[0]


def _effect_summary(diagnostics: pd.DataFrame, target: str, mode: str) -> str:
    rows = diagnostics.loc[
        (diagnostics["target"] == target)
        & (diagnostics["evidence_scope"] == "primary_ftv_residual")
    ].copy()
    if mode == "projection":
        rows = rows.loc[
            (rows["diagnostic_type"] == "representation_location")
            & (rows["comparison"] == "PREPROJECTOR_MINUS_PROJECTED")
        ]
        label_column = "comparison"
    elif mode == "pooling":
        rows = rows.loc[
            (rows["diagnostic_type"] == "representation_location")
            & (rows["comparison"] == "MEAN_STD_MINUS_MEAN")
        ]
        label_column = "comparison"
    else:
        rows = rows.loc[rows["diagnostic_type"] == "oracle_localization"]
        label_column = "oracle_representation"
    passed = rows.loc[rows["both_seed_pass"]]
    if passed.empty:
        maximum = float(rows["minimum_seed_delta_spearman"].max()) if not rows.empty else float("nan")
        return f"未通过双 seed +0.10 规则；最大 minimum-over-seeds Δrho={maximum:.3f}。"
    cells = []
    for _, row in passed.iterrows():
        endpoint = row["timing"] if pd.notna(row["timing"]) and str(row["timing"]) else row["interval"]
        cells.append(
            f"{row[label_column]}/{row['arm']}@{endpoint} "
            f"({row['seed_2026_delta_spearman']:+.3f}/{row['seed_3026_delta_spearman']:+.3f}, "
            f"n={int(row['n_min'])}–{int(row['n_max'])})"
        )
    return "通过：" + "；".join(cells) + "。"


def _gate_text(gate: Mapping[str, Any]) -> str:
    selected = gate["selected_candidate"]
    status = "PASS" if gate["passed"] else "FAIL"
    return (
        f"{status}；固定 candidate={selected['arm']}/{selected['representation']}，"
        f"qualifying endpoints={selected['qualifying_endpoints'] or '无'}，"
        f"minimum-seed rho macro={selected['minimum_seed_spearman_macro']:.3f}，"
        f"minimum-seed R² macro={selected['minimum_seed_natural_r2_macro']:.3f}"
    )


def render_report(
    config: Mapping[str, Any],
    run_summary: Mapping[str, Any],
    provenance: Mapping[str, Any],
    gates: Mapping[str, Any],
    raw_support: Mapping[str, Mapping[str, Any]],
    residual_and_dynamic: Mapping[str, Mapping[str, Any]],
    diagnostics: pd.DataFrame,
    scorecard: pd.DataFrame,
    recommendation: pd.DataFrame,
    best: pd.DataFrame,
    dynamic_interval_best: pd.DataFrame,
    validity: pd.DataFrame,
    bpe: Mapping[str, Any],
    artifact_rows: Sequence[Mapping[str, Any]],
    lock_verification: Mapping[str, Any],
    *,
    branch: str,
    commit_sha: str,
    push_status: str,
    push_error: str,
) -> str:
    score = scorecard.set_index("target")
    ftv = _metric_sentence(_best_lookup(best, "static_raw_early", "FTV"))
    ld_raw = _metric_sentence(_best_lookup(best, "static_raw_early", "LD"))
    ld_res = _metric_sentence(_best_lookup(best, "static_residual_early", "LD"))
    sph_raw = _metric_sentence(_best_lookup(best, "static_raw_early", "SPH"))
    sph_res = _metric_sentence(_best_lookup(best, "static_residual_early", "SPH"))
    bpe_raw = _metric_sentence(_best_lookup(best, "static_raw_early", "BPE"))
    bpe_res = _metric_sentence(_best_lookup(best, "static_residual_early", "BPE"))
    rec = recommendation.iloc[0]
    dynamic_lines = []
    for target in NONFTV_TARGETS:
        dynamic_lines.append(
            f"- {target}: {_gate_text(residual_and_dynamic[f'dynamic_{target}'])}; "
            f"descriptive best={_metric_sentence(_best_lookup(best, 'dynamic_residual_early', target))}."
        )
    dynamic_endpoint_lines = []
    for target_kind, targets, label in (
        ("raw", TARGETS, "raw"),
        ("residual_ftv", NONFTV_TARGETS, "FTV-residual"),
    ):
        for target in targets:
            target_rows = dynamic_interval_best.loc[
                (dynamic_interval_best["target_kind"] == target_kind)
                & (dynamic_interval_best["target"] == target)
            ].copy()
            target_rows["_order"] = target_rows["interval"].map(
                {value: index for index, value in enumerate(INTERVALS)}
            )
            target_rows = target_rows.sort_values("_order")
            dynamic_endpoint_lines.append(
                f"- {target} {label}: "
                + "；".join(_interval_sentence(row) for _, row in target_rows.iterrows())
                + "。"
            )
    validity_lines = []
    for representation in ORACLE_REPRESENTATIONS:
        region = {"Z5": "CORE", "Z6": "PERI10", "Z7": "PERI20"}[representation]
        static_n = validity.loc[
            (validity["representation"] == representation) & (validity["task_type"] == "static"),
            "valid_n",
        ].tolist()
        dynamic_n = validity.loc[
            (validity["representation"] == representation) & (validity["task_type"] == "dynamic"),
            "valid_n",
        ].tolist()
        validity_lines.append(f"- {region} ({representation}): static {static_n}；adjacent pairs {dynamic_n}。")
    score_lines = []
    for target in NONFTV_TARGETS:
        row = score.loc[target]
        score_lines.append(
            f"| {target} | {row['scientific_classification']} | {row['raw_image_observable']} | "
            f"{row['beyond_ftv_decodable']} | {row['longitudinal_literal_difference_supported']} | "
            f"{row['input_observability']} | {row['primary_bottleneck']} |"
        )
    artifact_lines = [
        f"- `{row['path']}` — {row['sha256']} ({row['size_bytes']} bytes)"
        for row in artifact_rows
    ]
    push_disclosure = push_status
    if push_status == "GITHUB_PUSH_FAILED":
        push_disclosure += f"；真实错误：`{push_error}`"
    return f"""# Non-FTV Phenotype Target Decodability Audit 最终报告

## 执行结论

唯一主推荐：**`{rec['recommendation']}`**（优先级规则 {int(rec['priority_rule_applied'])}）。理由：{rec['rationale']}。本审计没有启动、也不授权自动启动 multi-target training。

Primary gates：Gate A `{_gate_text(gates['gates']['A'])}`；Gate B `{_gate_text(gates['gates']['B'])}`；Gate C `{_gate_text(gates['gates']['C'])}`；Gate D `{_gate_text(gates['gates']['D'])}`；Gate E `{'PASS' if gates['gates']['E']['passed'] else 'FAIL'}`，qualifying targets={gates['gates']['E']['qualifying_targets']}。

## 设计、证据边界与完整性

本实验冻结 C1B-H/DCE7、375 人、四访视、seed-2026 五折、两个 checkpoint seed、LOCAL0/LOCAL3、Z1–Z7 与 64-mm LOCAL support；encoder/JEPA 均未重训。20 个 frozen feature cells、3,276 个 OOF aggregate endpoints 均完成。Ridge scaler/residualizer 仅在 outer train 拟合，alpha 仅由 validation 选择，test 每 probe 只预测一次；pCR 没有被解析或用于 target、representation、timing、residualizer、alpha、gate 或推荐选择。

Goal 6 是 SHA-locked sibling evidence：384 人 primary 支持 joint `N=LD+SPH+BPE` 与 joint `N_res` 在 `Clinical+FTV` 后的 increment；它没有分别证明 LD/SPH/BPE 各自的 residual pCR increment。family-specific 证据只能写成描述性排序（LD strongest、SPH weaker、BPE weakest），不能外推为三项独立 clinical proof。当前审计则固定在 375 人 image-observability estimand。

相邻变化 `100*(x_end-x_start)/abs(x_start)` 是将 Goal 6 的 signed-percent formula 新实例化到 adjacent intervals；Goal 6 原本冻结的是 baseline-referenced change，因此不能把本审计说成复现了 Goal 6 adjacent target。动态 macro 是三个 interval endpoint metric 的无权均值；Gate E 始终逐 early interval 读取 literal `z_end-z_start`，没有用 macro、T2→T3、prefix 或 FTV+LD residual 替代。

对 SPH/BPE raw 的原 brief 没有给分类阈值。为使分类 deterministic，本报告预先统一采用明确标记的 `RAW_IMAGE_SUPPORT_FOR_CLASSIFICATION`：同一固定 deployable candidate 在 T0/T1/T2 至少两个 timing、两 seed 均 rho≥0.40 且 natural R²>0。它不是新增 primary gate，也不改变 A–E。

## 十五个问题逐项回答

### 1. FTV decodability control 如何？

Gate-A-style raw control 为 `{_gate_text(raw_support['FTV'])}`。描述性最佳 early cell：{ftv}。FTV 是 response control，不是 non-FTV candidate。

### 2. LD raw 是否稳定可解码？

Gate A：`{_gate_text(gates['gates']['A'])}`。描述性最佳 early cell：{ld_raw}。

### 3. LD 去除 FTV component 后是否仍可解码？

Gate B：`{_gate_text(gates['gates']['B'])}`。这里 rho 是 outer-fold-isolated Goal-6 transformed-standardized residual rank；R² guardrail 是 FTV conditional baseline + MRI residual readout 的 raw-target reconstruction，不是“自然单位 residual R²”。描述性最佳 cell：{ld_res}。

### 4. SPH raw 是否可解码？

Raw classification support：`{_gate_text(raw_support['SPH'])}`；最佳：{sph_raw}。该 support 是 scorecard operationalization，不是新增 primary gate。

### 5. SPH residual 是否可解码？

Primary FTV-residual Gate C：`{_gate_text(gates['gates']['C'])}`；最佳：{sph_res}。FTV+LD residual 仅列在 residual matrix，未进入 Gate C 或分类。

### 6. BPE raw 与 residual 是否可解码？

Raw classification support：`{_gate_text(raw_support['BPE'])}`；最佳：{bpe_raw}。Primary FTV-residual Gate D：`{_gate_text(gates['gates']['D'])}`；最佳：{bpe_res}。即使数值 gate 通过，也不越过下一节 FOV firewall。

### 7. BPE 是否存在 input/FOV observability 问题？

是。冻结 target 是对侧乳腺中央连续五层 fibroglandular tissue early enhancement；所有 Z1–Z7 都只来自 lesion-centered 64-mm LOCAL，Oracle 也不扩 FOV。C1B nominal tensor extent 仅可由 shape/spacing 记为 {C1B_NOMINAL_EXTENT_XYZ_MM} mm；public lock 中没有 hash-bound BPE source ROI、laterality coordinate mapping、occupancy 或 boundary-touch audit，因此状态必须是 **`{bpe['observability_status']}`**，而不是擅自称 disjoint 或 encoder failure。BPE grounding 被阻断，直到验证 ≥99% source occupancy 且无 boundary touch，或开发 broader-context branch。

### 8. 哪些 adjacent Delta target/residual 可由 literal longitudinal latent difference 解码？

Gate E 只看 primary FTV-residual 与 literal difference：

{os.linesep.join(dynamic_lines)}

全部三个 interval 的 literal-difference 描述性最佳固定 candidate（按 minimum-over-seeds rho，再按 R²与注册顺序汇总；这不是 gate）为：

{os.linesep.join(dynamic_endpoint_lines)}

Prefix sensitivity 与 dynamic macro 仍完整公开，但不授权 Gate E。

### 9. Z2 `r` 是否优于 Z1 `projector(r)`？

按同 arm/target/endpoint、两 seed 各自 `Z2-Z1 >= +0.10` 判断：LD {_effect_summary(diagnostics, 'LD', 'projection')} SPH {_effect_summary(diagnostics, 'SPH', 'projection')} BPE {_effect_summary(diagnostics, 'BPE', 'projection')} Raw-only effect 在诊断表中另列，不能冒充 beyond-FTV evidence。

同一 representation-location audit 的 pooling 补充结果（两 seed `Z4-Z3 >= +0.10`）为：LD {_effect_summary(diagnostics, 'LD', 'pooling')} SPH {_effect_summary(diagnostics, 'SPH', 'pooling')} BPE {_effect_summary(diagnostics, 'BPE', 'pooling')}

### 10. Oracle CORE/PERI10/PERI20 是否显著改善 target，并基于何种 validity cohort？

这里“显著改善”严格指预注册 effect threshold，不是 p-value：同一 matched eligible set 上 Oracle mean+SD 相对 matched Z4、两 seed 都 Δrho≥+0.10。LD {_effect_summary(diagnostics, 'LD', 'localization')} SPH {_effect_summary(diagnostics, 'SPH', 'localization')} BPE {_effect_summary(diagnostics, 'BPE', 'localization')}

Validity（static T0/T1/T2/T3；dynamic T0→T1/T1→T2/T2→T3）：

{os.linesep.join(validity_lines)}

Oracle 是 mask-dependent diagnostic，不能成为 deployment input。

### 11. 每个 target 的主要 bottleneck 是什么？

- LD：`{score.loc['LD', 'primary_bottleneck']}`；classification=`{score.loc['LD', 'scientific_classification']}`。
- SPH：`{score.loc['SPH', 'primary_bottleneck']}`；classification=`{score.loc['SPH', 'scientific_classification']}`。
- BPE：`{score.loc['BPE', 'primary_bottleneck']}`；classification=`{score.loc['BPE', 'scientific_classification']}`。

只有在 Z1–Z7 绝对 residual signal 均弱且无 projection/pooling/localization effect 时才使用 encoder/current-feature-map；其余组合保留 mixed/unresolved。BPE FOV 未验证时永远先归 input observability。

### 12. 哪些 target 值得进入下一轮 grounding？

| target | classification | raw support | beyond FTV | early dynamic | input observability | bottleneck |
|---|---|---:|---:|---:|---|---|
{os.linesep.join(score_lines)}

Scorecard 没有计算 weighted total。Class A 仍只是下一阶段设计候选，不等于已单 family 证明 pCR increment，也不自动启动训练。

### 13. `FTV+LD` 是否得到充分的 image-observability/beyond-FTV 支持？

结论由 LD classification 决定：**`{score.loc['LD', 'scientific_classification']}`**。只有 Class A 才称 `FTV+LD` 得到充分支持；否则 LD 只可称 morphology/extent response target。最终 recommendation=`{rec['recommendation']}`。

### 14. SPH/BPE 是否应等待 region-aware 或 broader-context architecture？

SPH classification=`{score.loc['SPH', 'scientific_classification']}`；若 residual Oracle localization 通过，应先开发 mask-free region-aware representation。BPE status=`{bpe['observability_status']}`，故无论 probe 数值如何，都应等待可覆盖/验证对侧 tissue 的 broader-context pathway，不应在现有 LOCAL 中直接 grounding。

### 15. 下一轮正式 training 必须继续冻结什么？

必须继续冻结：C1B-H/DCE7 与 exact 375-person eligible population；canonical six-digit exact join；seed-2026 outer folds；FTV/LD/SPH/BPE workbook fields 与 hashes；T0–T3 timing（T3 始终 late/pre-surgery）；adjacent formula/zero-denominator fail-closed rule；train-only 1/99% winsor、family transform、scaler 与 fixed-alpha residualizer；selected LOCAL0/LOCAL3 checkpoints、online pathway、64-mm physical support 与 representation definitions；validation-only alpha selection/test-once contract；pCR firewall；patient-level artifact private/ignored/0600 contract。任何新 target、FOV、region-aware state 或 multi-target loss都需另立 preregistration，不能由本审计自动触发。

## Scorecard 与推荐

| target | classification | raw support | beyond FTV | early dynamic | input observability | bottleneck |
|---|---|---:|---:|---:|---|---|
{os.linesep.join(score_lines)}

唯一 recommendation：`{rec['recommendation']}`。没有 weighted total，没有用 test pCR 作 tie-break。

## Reproducibility、privacy 与交付状态

- Branch：`{branch}`
- Experiment parent：`{config['parent_sha']}`
- Preregistration lock SHA-256：`{lock_verification['lock_sha256']}`；verification=`{lock_verification['status']}`；binding count={lock_verification['binding_count']}。
- Reported experiment commit SHA：`{commit_sha}`
- Push status：`{push_disclosure}`
- Formal run status：`{run_summary['status']}`；encoder retrained=`{run_summary['encoder_retrained']}`；pCR read=`{run_summary['pcr_read']}`；test used for alpha=`{run_summary['test_used_for_alpha_selection']}`。
- Patient set SHA-256：`{provenance['patient_set_sha256']}`；公开表不含 patient identifiers，private OOF renderer 从未打开。

本报告生成时公开 aggregate artifact hashes：

{os.linesep.join(artifact_lines)}

Commit/push 字段通过命令行注入，便于在 scientific commit 后记录 delivery provenance；若 push 失败必须保留 local commit、写 `GITHUB_PUSH_FAILED` 与原始错误，禁止 force push。
"""


def write_report(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
        path.chmod(0o644)
    finally:
        temporary.unlink(missing_ok=True)


def public_artifact_rows(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(paths):
        rows.append(
            {
                "path": str(path.relative_to(AUDIT_ROOT)),
                "size_bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
                "contains_patient_identifiers": False,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=AUDIT_ROOT / "configs" / "audit.json")
    parser.add_argument("--commit-sha", default="PENDING_LOCAL_COMMIT")
    parser.add_argument(
        "--push-status",
        choices=("NOT_ATTEMPTED", "PUSHED", "GITHUB_PUSH_FAILED"),
        default="NOT_ATTEMPTED",
    )
    parser.add_argument("--push-error", default="")
    parser.add_argument(
        "--allow-descendant-head",
        action="store_true",
        help="verify the immutable lock on a descendant HEAD for delivery-only report rerender",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="regenerate only final_report.md from already generated aggregate analysis artifacts",
    )
    arguments = parser.parse_args()
    if arguments.push_status == "GITHUB_PUSH_FAILED" and not arguments.push_error.strip():
        raise ValueError("GITHUB_PUSH_FAILED requires --push-error with the real error")
    if arguments.push_status != "GITHUB_PUSH_FAILED" and arguments.push_error:
        raise ValueError("--push-error is allowed only with GITHUB_PUSH_FAILED")

    if arguments.allow_descendant_head and not arguments.report_only:
        raise ValueError("--allow-descendant-head is allowed only with --report-only")
    lock_verification = require_preregistration_lock(
        require_exact_parent=not arguments.allow_descendant_head
    )
    config = load_config(arguments.config)
    frames, run_summary, provenance = load_public_inputs()
    validate_inputs(config, frames, run_summary, provenance, lock_verification)
    patient_set_sha256 = str(provenance["patient_set_sha256"])

    if arguments.report_only:
        analysis_paths = {
            "candidate": AUDIT_ROOT / "metrics" / "gate_candidate_matrix.csv",
            "gates": AUDIT_ROOT / "metrics" / "primary_gates.json",
            "diagnostics": AUDIT_ROOT / "metrics" / "bottleneck_diagnostics.csv",
            "scorecard": AUDIT_ROOT / "metrics" / "grounding_candidate_scorecard.csv",
            "recommendation": AUDIT_ROOT / "metrics" / "final_target_recommendation.csv",
            "best": AUDIT_ROOT / "metrics" / "descriptive_best_cells.csv",
            "dynamic_interval_best": AUDIT_ROOT / "metrics" / "dynamic_interval_best_cells.csv",
            "validity": AUDIT_ROOT / "metrics" / "oracle_validity_summary.csv",
            "bpe": AUDIT_ROOT / "metrics" / "bpe_fov_observability_audit.csv",
            "manifest": AUDIT_ROOT / "manifests" / "public_analysis_artifacts.csv",
        }
        missing = [str(path) for path in analysis_paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"report-only analysis inputs missing: {missing}")
        gates = _read_json(analysis_paths["gates"])
        candidate = _read_csv(analysis_paths["candidate"])
        diagnostics = _read_csv(analysis_paths["diagnostics"])
        scorecard = _read_csv(analysis_paths["scorecard"])
        recommendation = _read_csv(analysis_paths["recommendation"])
        best = _read_csv(analysis_paths["best"])
        dynamic_interval_best = _read_csv(analysis_paths["dynamic_interval_best"])
        validity = _read_csv(analysis_paths["validity"])
        bpe_frame = _read_csv(analysis_paths["bpe"])
        artifact_rows = _read_csv(analysis_paths["manifest"]).to_dict("records")
        artifact_rows.append(
            public_artifact_rows([analysis_paths["manifest"]])[0]
        )
        raw_support = gates["per_target_raw_support"]
        residual_and_dynamic = {
            "LD": gates["gates"]["B"],
            "SPH": gates["gates"]["C"],
            "BPE": gates["gates"]["D"],
            **{f"dynamic_{target}": gates["per_target_dynamic"][target] for target in NONFTV_TARGETS},
        }
        del candidate
    else:
        candidate, gates, raw_support, residual_and_dynamic = compute_gates(
            config, frames["oof"], patient_set_sha256
        )
        diagnostics = compute_bottleneck_diagnostics(
            frames["location"], frames["localization"], patient_set_sha256
        )
        bpe_frame, bpe_row = bpe_observability_audit(config, patient_set_sha256)
        validity = oracle_validity_summary(frames["oof"], patient_set_sha256)
        integrity = probe_integrity_summary(
            frames["coverage"], frames["selection"], patient_set_sha256
        )
        scorecard, recommendation = make_scorecard(
            frames["oof"],
            gates,
            raw_support,
            residual_and_dynamic,
            diagnostics,
            str(bpe_row["observability_status"]),
            patient_set_sha256,
        )
        best = descriptive_best_cells(frames["oof"], patient_set_sha256)
        dynamic_interval_best = dynamic_interval_best_cells(
            frames["oof"], patient_set_sha256
        )
        dynamic_macro = dynamic_macro_table(frames["oof"], patient_set_sha256)
        outputs = {
            AUDIT_ROOT / "metrics" / "gate_candidate_matrix.csv": candidate,
            AUDIT_ROOT / "metrics" / "bottleneck_diagnostics.csv": diagnostics,
            AUDIT_ROOT / "metrics" / "bpe_fov_observability_audit.csv": bpe_frame,
            AUDIT_ROOT / "metrics" / "oracle_validity_summary.csv": validity,
            AUDIT_ROOT / "metrics" / "probe_integrity_summary.csv": integrity,
            AUDIT_ROOT / "metrics" / "grounding_candidate_scorecard.csv": scorecard,
            AUDIT_ROOT / "metrics" / "final_target_recommendation.csv": recommendation,
            AUDIT_ROOT / "metrics" / "descriptive_best_cells.csv": best,
            AUDIT_ROOT / "metrics" / "dynamic_interval_best_cells.csv": dynamic_interval_best,
            AUDIT_ROOT / "metrics" / "dynamic_macro.csv": dynamic_macro,
        }
        for path, frame in outputs.items():
            atomic_csv(path, frame)
        gate_path = AUDIT_ROOT / "metrics" / "primary_gates.json"
        atomic_json(gate_path, gates)
        figure_paths = generate_figures(frames["oof"], diagnostics)
        artifact_paths = [*outputs, gate_path, *figure_paths]
        artifact_rows = public_artifact_rows(artifact_paths)
        manifest_path = AUDIT_ROOT / "manifests" / "public_analysis_artifacts.csv"
        atomic_csv(manifest_path, pd.DataFrame(artifact_rows))
        artifact_rows = public_artifact_rows([*artifact_paths, manifest_path])

    bpe_row = bpe_frame.iloc[0].to_dict()
    report = render_report(
        config,
        run_summary,
        provenance,
        gates,
        raw_support,
        residual_and_dynamic,
        diagnostics,
        scorecard,
        recommendation,
        best,
        dynamic_interval_best,
        validity,
        bpe_row,
        artifact_rows,
        lock_verification,
        branch=str(config["branch"]),
        commit_sha=arguments.commit_sha,
        push_status=arguments.push_status,
        push_error=arguments.push_error,
    )
    report_path = AUDIT_ROOT / "reports" / "final_report.md"
    write_report(report_path, report)
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "report": str(report_path.relative_to(REPO_ROOT)),
                "recommendation": str(recommendation.iloc[0]["recommendation"]),
                "gates": {
                    gate: bool(gates["gates"][gate]["passed"])
                    for gate in ("A", "B", "C", "D", "E")
                },
                "bpe_observability": bpe_row["observability_status"],
                "private_predictions_opened": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

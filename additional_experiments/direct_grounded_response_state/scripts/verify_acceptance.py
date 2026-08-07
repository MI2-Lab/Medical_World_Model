#!/usr/bin/env python3
"""逐条验证 Goal 的 19 项验收标准，并生成不含患者行的公开机器证据。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
MODELS = ("G0", "G1", "G2", "G3", "G4")
TRAINED = ("G1", "G2", "G3", "G4")
FOLDS = tuple(range(5))
TIMEPOINTS = ("T0", "T1", "T2", "T3")
TRANSITIONS = ("T0→T1", "T1→T2", "T2→T3")
DECISION_POINTS = ("T0", "T0-T1", "T0-T2")
EXPECTED_MANIFEST_SHA = "143e482d711225c0611006d99bd7345d2fa1a5c16c65fbaf8399341a0d26aa38"
EXPECTED_TRAINING_PLAN_SHA = "fd43c11d9855c62d97dcd89f3ab4c46c6292be31faf7c66fa92ff1477c24dfd9"
EXPECTED_TRAINING_IMPLEMENTATION_SHA = "fb308f8a3cfe735ca1ef2e17e66367b11d4e6edc424bd01725cad200b780e750"
HISTORY_COLUMNS = {
    "epoch",
    "fold",
    "model",
    "total_loss",
    "base_loss",
    "ftv_loss",
    "weighted_ftv_loss",
    "val_base_loss",
    "val_ftv_metric",
    "representation_std",
    "learning_rate",
    "is_selected_checkpoint",
}
PROBE_FALSE_FLAGS = {
    "test_used_for_target_transform",
    "test_used_for_checkpoint_selection",
    "test_used_for_lambda_selection",
    "test_used_for_scaler",
    "test_used_for_alpha_selection",
}
PCR_FALSE_FLAGS = {
    "test_used_for_checkpoint_selection",
    "test_used_for_lambda_selection",
    "test_used_for_scaler",
    "test_used_for_hyperparameter_selection",
    "test_used_for_threshold_selection",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON 顶层不是 object: {path}")
    return payload


def safe_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint 顶层不是 mapping: {path}")
    return payload


def check(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"criterion": name, "passed": bool(passed), "evidence": evidence}


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    return None


def _series_all(frame: pd.DataFrame, columns: set[str], expected: bool) -> bool:
    if not columns.issubset(frame.columns):
        return False
    return all(_as_bool(value) is expected for column in columns for value in frame[column])


def _prediction_files(kind: str) -> list[Path]:
    return sorted((ROOT / "predictions" / kind).glob("G*/fold_*/test_predictions.csv"))


def _fold_manifest() -> Path:
    explicit = os.environ.get("DGRS_FOLD_MANIFEST")
    if explicit:
        return Path(explicit).expanduser()
    data_root = Path(os.environ.get("DGRS_DATA_ROOT", "/path/to/preprocessed"))
    return (
        data_root
        / "I-SPY2"
        / "_matched_breastdcedl_t0_dicomrepair_rgb224_seed2026"
        / "matched_patient_cv_splits_seed2026.csv"
    )


def _expected_paths(base: Path, models: tuple[str, ...], leaf: str) -> list[Path]:
    return [base / model.lower() / f"fold_{fold}" / leaf for model in models for fold in FOLDS]


def _live_manifest_audit(manifest: pd.DataFrame, paths: list[Path], kind: str) -> bool:
    if len(manifest) != len(paths) or set(manifest["kind"]) != {kind}:
        return False
    expected = {str(path.relative_to(REPO)): path for path in paths}
    if set(manifest["path"].astype(str)) != set(expected):
        return False
    for row in manifest.itertuples(index=False):
        path = expected[str(row.path)]
        frame = pd.read_csv(path)
        if sha256(path) != str(row.sha256) or len(frame) != int(row.rows):
            return False
        if frame["patient_id"].nunique() != int(row.patients):
            return False
    return True


def _prediction_key_set(frame: pd.DataFrame, columns: list[str]) -> set[tuple[str, ...]]:
    return set(
        frame[columns]
        .fillna("<NA>")
        .astype(str)
        .itertuples(index=False, name=None)
    )


def _recompute_decision(
    macro: pd.DataFrame, paired: pd.DataFrame, stability: pd.DataFrame
) -> tuple[str, dict[str, dict[str, bool]], pd.DataFrame]:
    results: dict[str, dict[str, bool]] = {}
    rows: list[dict[str, Any]] = []
    for grounded, baseline in (("G3", "G1"), ("G4", "G2")):
        comparison = f"{grounded}-{baseline}"
        subset = macro.loc[macro["comparison"].eq(comparison)]

        def one(scope: str, metric: str) -> pd.Series:
            part = subset.loc[subset["scope"].eq(scope) & subset["metric"].eq(metric)]
            if len(part) != 1:
                raise ValueError(f"缺唯一 macro row: {comparison}/{scope}/{metric}")
            return part.iloc[0]

        static_s, static_r = one("A_static_ftv", "spearman"), one("A_static_ftv", "r2")
        pass_s = bool(static_s.estimate >= 0.05 and static_s.ci_low > 0)
        pass_r = bool(static_r.estimate >= 0.05 and static_r.ci_low > 0)
        reverse_s = bool(static_s.estimate <= -0.05 and static_s.ci_high < 0)
        reverse_r = bool(static_r.estimate <= -0.05 and static_r.ci_high < 0)
        gate_a = (pass_s and not reverse_r) or (pass_r and not reverse_s)

        change_s, change_r = one("B_change_ftv", "spearman"), one("B_change_ftv", "r2")
        macro_pass = bool(
            (change_s.estimate >= 0.05 and change_s.ci_low > 0)
            or (change_r.estimate >= 0.05 and change_r.ci_low > 0)
        )
        change_cells = paired.loc[
            paired["comparison"].eq(comparison)
            & paired["kind"].eq("probe")
            & paired["task"].eq("change")
            & paired["target"].eq("ftv")
            & paired["metric"].isin({"spearman", "r2"})
        ]
        transition_pass = False
        for metric in ("spearman", "r2"):
            part = change_cells.loc[change_cells["metric"].eq(metric)]
            positive = part["estimate"].ge(0.05) & part["ci_low"].gt(0)
            reverse = part["estimate"].le(-0.05) & part["ci_high"].lt(0)
            transition_pass = transition_pass or bool(positive.any() and not reverse.any())
        gate_b = macro_pass or transition_pass

        pcr_macro = one("C_longitudinal_pcr", "auroc")
        pcr_cells = paired.loc[
            paired["comparison"].eq(comparison)
            & paired["kind"].eq("pcr")
            & paired["metric"].eq("auroc")
            & paired["decision_point"].isin({"T0-T1", "T0-T2"})
        ]
        both_positive = len(pcr_cells) == 2 and bool(pcr_cells["estimate"].gt(0).all())
        gate_c = bool(both_positive and pcr_macro.estimate >= 0.02 and pcr_macro.ci_low > 0)
        clear_decline = bool(
            ((pcr_cells["estimate"] < 0) & (pcr_cells["ci_high"] < 0)).any()
        )

        grounded_rows = stability.loc[stability["model"].eq(grounded)]
        stable = bool(
            len(grounded_rows) == 5
            and _series_all(grounded_rows, {"finite"}, True)
            and grounded_rows["representation_std"].ge(0.05).all()
            and grounded_rows["base_degradation_fraction"].le(0.05 + 1e-12).all()
        )
        eligible = stable and not clear_decline
        result = {
            "A_static_grounding": gate_a,
            "B_observed_delta_ftv": gate_b,
            "C_longitudinal_pcr": gate_c,
            "no_clear_pcr_decline": not clear_decline,
            "stability_and_base_gate": stable,
            "eligible": eligible,
            "go": eligible and gate_a and (gate_b or gate_c),
            "partial_go": eligible and gate_a and not (gate_b or gate_c),
        }
        results[comparison] = result
        rows.extend(
            {"comparison": comparison, "gate": gate, "passed": passed}
            for gate, passed in result.items()
        )
    decision = (
        "GO"
        if any(item["go"] for item in results.values())
        else "PARTIAL GO"
        if any(item["partial_go"] for item in results.values())
        else "NO-GO"
    )
    return decision, results, pd.DataFrame(rows)


def verify() -> tuple[dict[str, Any], dict[str, Any]]:
    final = ROOT / "metrics" / "final"
    analysis = read_json(final / "analysis_acceptance_evidence.json")
    summary = read_json(final / "aggregation_summary.json")
    decision = read_json(final / "decision.json")
    lambda_selection = read_json(ROOT / "metrics" / "lambda_selection" / "lambda_selection.json")
    smoke = read_json(ROOT / "metrics" / "smoke" / "smoke_checks.json")
    plan = ROOT / "EXPERIMENT_PLAN.md"
    plan_redaction = read_json(ROOT / "reports" / "plan_redaction_provenance.json")
    report = ROOT / "reports" / "final_report.md"
    manifest = _fold_manifest()

    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=REPO, text=True
    ).strip()
    conda_name = Path(sys.prefix).name

    checkpoint_paths = _expected_paths(ROOT / "checkpoints" / "formal", TRAINED, "best.pt")
    history_paths = [
        ROOT / "metrics" / "training" / "formal" / model.lower() / f"fold_{fold}.csv"
        for model in TRAINED
        for fold in FOLDS
    ]
    selection_paths = _expected_paths(
        ROOT / "checkpoints" / "formal", TRAINED, "selection.json"
    )
    all_training_files_exist = all(
        path.is_file() for path in checkpoint_paths + history_paths + selection_paths
    )
    payloads = [safe_checkpoint(path) for path in checkpoint_paths]
    payload_by_key = {
        (str(payload.get("model_name")), int(payload.get("fold", -1))): payload
        for payload in payloads
    }
    selections = [read_json(path) for path in selection_paths]
    selection_by_key = {
        (str(item.get("model_name")), int(item.get("fold", -1))): item
        for item in selections
    }
    trained_keys = {(model, fold) for model in TRAINED for fold in FOLDS}
    implementation_hashes = {str(payload.get("implementation_sha256")) for payload in payloads}
    plan_hashes = {str(payload.get("plan_sha256")) for payload in payloads}

    history_ok = True
    selected_history_rows = 0
    for path, model, fold in zip(
        history_paths,
        (model for model in TRAINED for _ in FOLDS),
        (fold for _ in TRAINED for fold in FOLDS),
        strict=True,
    ):
        frame = pd.read_csv(path)
        selected = frame["is_selected_checkpoint"].map(_as_bool) if "is_selected_checkpoint" in frame else pd.Series(dtype=bool)
        selection = selection_by_key.get((model, fold), {})
        payload = payload_by_key.get((model, fold), {})
        this_ok = (
            HISTORY_COLUMNS.issubset(frame.columns)
            and set(frame["model"].astype(str)) == {model}
            and set(pd.to_numeric(frame["fold"], errors="coerce")) == {fold}
            and sum(value is True for value in selected) == 1
            and int(frame.loc[selected.eq(True), "epoch"].iloc[0]) == int(selection.get("selected_epoch", -1))
            and int(payload.get("epoch", -1)) == int(selection.get("selected_epoch", -2))
            and str(payload.get("history_sha256")) == sha256(path)
            and str(payload.get("selection_sha256"))
            == sha256(ROOT / "checkpoints" / "formal" / model.lower() / f"fold_{fold}" / "selection.json")
        )
        history_ok = history_ok and this_ok
        selected_history_rows += sum(value is True for value in selected)

    exact_checkpoint_glob = set((ROOT / "checkpoints" / "formal").glob("g*/fold_*/best.pt")) == set(checkpoint_paths)
    exact_history_glob = set((ROOT / "metrics" / "training" / "formal").glob("g*/fold_*.csv")) == set(history_paths)
    checkpoint_ok = (
        all_training_files_exist
        and exact_checkpoint_glob
        and exact_history_glob
        and set(payload_by_key) == trained_keys
        and set(selection_by_key) == trained_keys
        and all(bool(payload.get("finalized")) for payload in payloads)
        and all(item.get("test_data_used") is False for item in selections)
        and implementation_hashes == {EXPECTED_TRAINING_IMPLEMENTATION_SHA}
        and history_ok
        and selected_history_rows == 20
    )

    shared_initialization_ok = all(
        isinstance(payload_by_key[(model, fold)].get("shared_initialization_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", payload_by_key[(model, fold)]["shared_initialization_sha256"])
        for model in TRAINED
        for fold in FOLDS
    ) and all(
        payload_by_key[("G1", fold)]["shared_initialization_sha256"]
        == payload_by_key[("G3", fold)]["shared_initialization_sha256"]
        and payload_by_key[("G2", fold)]["shared_initialization_sha256"]
        == payload_by_key[("G4", fold)]["shared_initialization_sha256"]
        for fold in FOLDS
    )

    public_plan_sha = sha256(plan)
    exact_plan = plan_hashes == {public_plan_sha} and all(
        plan.stat().st_mtime <= path.stat().st_mtime for path in checkpoint_paths
    )
    certified_redaction = (
        plan_hashes == {EXPECTED_TRAINING_PLAN_SHA}
        and plan_redaction.get("status") == "certified_public_path_redaction"
        and plan_redaction.get("scientific_content_changed") is False
        and plan_redaction.get("training_plan_sha256") == EXPECTED_TRAINING_PLAN_SHA
        and plan_redaction.get("public_plan_sha256") == public_plan_sha
        and plan_redaction.get("replacement_token") in plan.read_text(encoding="utf-8")
        and plan_redaction.get("contains_original_private_path") is False
    )
    plan_ok = plan.is_file() and (exact_plan or certified_redaction)

    feature_paths = [
        ROOT / "features" / model / f"fold_{fold}" / "observed_features.npz"
        for model in MODELS
        for fold in FOLDS
    ]
    metadata_paths = [path.with_name("extraction_metadata.json") for path in feature_paths]
    expected_feature_keys = {(model, fold) for model in MODELS for fold in FOLDS}
    feature_ok = (
        set((ROOT / "features").glob("G*/fold_*/observed_features.npz")) == set(feature_paths)
        and all(path.is_file() for path in feature_paths + metadata_paths)
    )
    feature_metadata_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for path, metadata_path, model, fold in zip(
        feature_paths,
        metadata_paths,
        (model for model in MODELS for _ in FOLDS),
        (fold for _ in MODELS for fold in FOLDS),
        strict=True,
    ):
        metadata = read_json(metadata_path)
        feature_metadata_by_key[(model, fold)] = metadata
        with np.load(path, allow_pickle=False) as asset:
            required = {"patient_ids", "splits", "response_state", "timepoints", "model", "fold", "label_pcr"}
            response = asset["response_state"] if required.issubset(asset.files) else np.empty(0)
            this_ok = (
                required.issubset(asset.files)
                and response.shape == (808, 4, 192)
                and np.isfinite(response).all()
                and len(np.unique(asset["patient_ids"].astype(str))) == 808
                and tuple(asset["timepoints"].astype(str)) == TIMEPOINTS
                and str(asset["model"].item()) == model
                and int(asset["fold"].item()) == fold
                and metadata.get("model") == model
                and int(metadata.get("fold", -1)) == fold
                and metadata.get("feature_shape") == [808, 4, 192]
                and metadata.get("patient_count") == 808
                and metadata.get("feature_file_sha256") == sha256(path)
                and metadata.get("fold_manifest_sha256") == EXPECTED_MANIFEST_SHA
                and metadata.get("canonical_manifest_rows_verified") is True
                and metadata.get("canonical_label_rows_verified") is True
                and metadata.get("measurement_targets_read_during_extraction") is False
                and metadata.get("coverage", {}).get("formal_complete") is True
            )
        feature_ok = feature_ok and this_ok
    feature_ok = feature_ok and set(feature_metadata_by_key) == expected_feature_keys

    smoke_models = smoke.get("models", [])
    smoke_model_names = [str(item.get("model")) for item in smoke_models]
    smoke_impl: set[str] = set()
    smoke_assets_ok = True
    for item in smoke_models:
        path = ROOT / str(item.get("checkpoint", ""))
        if not path.is_file() or sha256(path) != str(item.get("checkpoint_sha256")):
            smoke_assets_ok = False
            continue
        smoke_impl.add(str(safe_checkpoint(path).get("implementation_sha256")))
    pooling = smoke.get("pooling_contract", {})
    smoke_ok = (
        smoke.get("status") == "passed"
        and smoke.get("real_cache_smoke") is True
        and len(smoke_models) == 4
        and set(smoke_model_names) == set(TRAINED)
        and len(set(smoke_model_names)) == 4
        and smoke_assets_ok
        and smoke_impl == implementation_hashes
        and smoke.get("paired_shared_initialization") == {"G1-G3": True, "G2-G4": True}
        and smoke.get("mask_routing_rejection") == {model: True for model in TRAINED}
        and smoke.get("encoder_spatial_map_bitwise_stable_without_mask_argument") is True
        and smoke.get("world_model_forward_has_no_ftv_input") is True
        and smoke.get("ftv_head_removal_cannot_change_encode_online_graph") is True
        and smoke.get("split_disjoint") is True
        and smoke.get("test_data_used_for_smoke_selection") is False
        and pooling.get("empty_mask_finite") is True
        and all(abs(float(pooling.get(key, math.inf))) <= 1e-6 for key in (
            "occupancy_scale_max_abs_error",
            "constant_feature_different_support_max_abs_error",
            "all_ones_vs_gap_max_abs_error",
            "empty_mask_vs_gap_max_abs_error",
        ))
        and all(
            float(item.get("ftv_encoder_gradient_norm_raw", 0)) > 0
            and float(item.get("ftv_head_gradient_norm_raw", 0)) > 0
            for item in smoke_models
            if item.get("model") in {"G3", "G4"}
        )
    )

    candidate_lambdas = [float(value) for value in lambda_selection.get("candidate_lambdas", [])]
    pilot_pairs = {
        (str(item.get("grounded_model")), str(item.get("baseline_model")))
        for item in lambda_selection.get("selected_pairing_evidence", [])
    }
    access = lambda_selection.get("data_access_contract", {})
    lambda_ok = (
        lambda_selection.get("status") == "selected"
        and math.isclose(float(lambda_selection.get("selected_lambda_ftv", -1)), 0.25)
        and candidate_lambdas == [0.02, 0.05, 0.1, 0.25]
        and lambda_selection.get("selected_same_lambda_for_g3_and_g4") is True
        and len(lambda_selection.get("source_assets", [])) == 10
        and pilot_pairs == {("G3", "G1"), ("G4", "G2")}
        and all(item.get("joint_safe") is True and item.get("joint_effective") is True for item in lambda_selection.get("selected_pairing_evidence", []))
        and access.get("allowed_splits") == ["train", "val"]
        and set(access.get("observed_splits", [])) == {"train", "val"}
        and all(access.get(key) is False for key in (
            "test_features_loaded",
            "test_ftv_loaded",
            "pcr_loaded",
            "test_auroc_loaded",
        ))
        and access.get("lambda_selected_on_validation_only") is True
    )

    probe_files = _prediction_files("representation_probes")
    pcr_files = _prediction_files("pcr_readouts")
    probe_frames: dict[tuple[str, int], pd.DataFrame] = {}
    pcr_frames: dict[tuple[str, int], pd.DataFrame] = {}
    predictions_ok = len(probe_files) == 25 and len(pcr_files) == 25
    for path, kind, store in (
        [(path, "probe", probe_frames) for path in probe_files]
        + [(path, "pcr", pcr_frames) for path in pcr_files]
    ):
        frame = pd.read_csv(path)
        models = set(frame["model"].astype(str))
        folds = set(pd.to_numeric(frame["fold"], errors="coerce"))
        model = next(iter(models)) if len(models) == 1 else ""
        fold = int(next(iter(folds))) if len(folds) == 1 else -1
        key = (model, fold)
        false_flags = PROBE_FALSE_FLAGS if kind == "probe" else PCR_FALSE_FLAGS
        guard_col = "test_prediction_guard_enforced"
        call_col = "test_predict_call_count" if kind == "probe" else "test_predict_proba_call_count"
        unique_cols = (
            ["patient_id", "model", "fold", "task", "timepoint", "transition", "representation", "input_variant", "target"]
            if kind == "probe"
            else ["patient_id", "model", "fold", "decision_point"]
        )
        this_ok = (
            key in expected_feature_keys
            and set(frame["split"].astype(str)) == {"test"}
            and _series_all(frame, false_flags, False)
            and _series_all(frame, {guard_col}, True)
            and call_col in frame
            and pd.to_numeric(frame[call_col], errors="coerce").eq(1).all()
            and not frame.duplicated(unique_cols).any()
            and set(frame["fold_manifest_sha256"].astype(str)) == {EXPECTED_MANIFEST_SHA}
            and set(frame["source_feature_sha256"].astype(str))
            == {feature_metadata_by_key[key]["feature_file_sha256"]}
        )
        predictions_ok = predictions_ok and this_ok and key not in store
        store[key] = frame
    predictions_ok = predictions_ok and set(probe_frames) == expected_feature_keys and set(pcr_frames) == expected_feature_keys
    for fold in FOLDS:
        for grounded, baseline in (("G3", "G1"), ("G4", "G2")):
            probe_columns = ["patient_id", "task", "timepoint", "transition", "representation", "input_variant", "target"]
            pcr_columns = ["patient_id", "decision_point"]
            predictions_ok = predictions_ok and (
                _prediction_key_set(probe_frames[(grounded, fold)], probe_columns)
                == _prediction_key_set(probe_frames[(baseline, fold)], probe_columns)
                and _prediction_key_set(pcr_frames[(grounded, fold)], pcr_columns)
                == _prediction_key_set(pcr_frames[(baseline, fold)], pcr_columns)
            )
    probe_rows = sum(len(frame) for frame in probe_frames.values())
    pcr_rows = sum(len(frame) for frame in pcr_frames.values())
    probe_patients = len(set().union(*(set(frame["patient_id"].astype(str)) for frame in probe_frames.values())))
    pcr_patients = len(set().union(*(set(frame["patient_id"].astype(str)) for frame in pcr_frames.values())))
    predictions_ok = predictions_ok and (probe_rows, pcr_rows, probe_patients, pcr_patients) == (39375, 12120, 375, 808)

    prediction_manifest = pd.read_csv(final / "prediction_file_manifest.csv")
    manifest_ok = (
        len(prediction_manifest) == 50
        and _live_manifest_audit(
            prediction_manifest.loc[prediction_manifest["kind"].eq("representation_probe")],
            probe_files,
            "representation_probe",
        )
        and _live_manifest_audit(
            prediction_manifest.loc[prediction_manifest["kind"].eq("pcr_readout")],
            pcr_files,
            "pcr_readout",
        )
    )

    probe_metrics = pd.read_csv(final / "probe_oof_metrics.csv")
    pcr_metrics = pd.read_csv(final / "pcr_oof_metrics.csv")
    static_ftv = probe_metrics.loc[probe_metrics["task"].eq("static") & probe_metrics["target"].eq("ftv")]
    change_ftv = probe_metrics.loc[probe_metrics["task"].eq("change") & probe_metrics["target"].eq("ftv")]
    transfer = probe_metrics.loc[probe_metrics["target"].isin({"ld", "sphericity"})]
    probe_metric_keys = set(
        probe_metrics[["model", "task", "target", "timepoint", "transition"]]
        .fillna("<NA>")
        .astype(str)
        .itertuples(index=False, name=None)
    )
    expected_probe_metric_keys = {
        (model, task, target, cell if task == "static" else "<NA>", cell if task == "change" else "<NA>")
        for model in MODELS
        for target in ("ftv", "ld", "sphericity")
        for task, cells in (("static", TIMEPOINTS), ("change", TRANSITIONS))
        for cell in cells
    }
    metrics_ok = (
        len(probe_metrics) == 105
        and probe_metric_keys == expected_probe_metric_keys
        and probe_metrics["n_patients"].eq(375).all()
        and probe_metrics["n_folds"].eq(5).all()
        and len(pcr_metrics) == 15
        and set(pcr_metrics[["model", "decision_point"]].itertuples(index=False, name=None))
        == {(model, point) for model in MODELS for point in DECISION_POINTS}
        and pcr_metrics["n_patients"].eq(808).all()
        and pcr_metrics["n_folds"].eq(5).all()
    )

    coverage = pd.read_csv(final / "coverage.csv")
    coverage_ok = (
        len(coverage) == 120
        and len(coverage.loc[coverage["kind"].eq("probe")]) == 105
        and len(coverage.loc[coverage["kind"].eq("pcr")]) == 15
        and coverage.loc[coverage["kind"].eq("probe"), "rows"].eq(375).all()
        and coverage.loc[coverage["kind"].eq("probe"), "patients"].eq(375).all()
        and coverage.loc[coverage["kind"].eq("pcr"), "rows"].eq(808).all()
        and coverage.loc[coverage["kind"].eq("pcr"), "patients"].eq(808).all()
        and coverage["fold_count"].eq(5).all()
    )
    issues = pd.read_csv(final / "input_issues.csv")

    ci_ok = True
    for name in (
        "probe_bootstrap_ci.csv",
        "pcr_bootstrap_ci.csv",
        "paired_differences_bootstrap_ci.csv",
        "paired_macro_bootstrap_ci.csv",
    ):
        frame = pd.read_csv(final / name)
        numeric = frame[["estimate", "ci_low", "ci_high"]].apply(pd.to_numeric, errors="coerce")
        ci_ok = (
            ci_ok
            and np.isfinite(numeric.to_numpy()).all()
            and frame["bootstrap_replicates"].eq(2000).all()
            and ("finite_replicates" not in frame or frame["finite_replicates"].eq(2000).all())
        )

    analysis_checks = analysis.get("checks", {})
    all_analysis_checks_pass = bool(analysis_checks) and all(
        item.get("passed") is True for item in analysis_checks.values()
    )
    analysis_fresh = (
        analysis.get("formal_analysis") is True
        and analysis.get("eligible_for_independent_acceptance_verifier") is True
        and all_analysis_checks_pass
        and analysis.get("analysis_source_sha256") == sha256(ROOT / "src" / "dgrs" / "analysis.py")
        and summary.get("analysis_source_sha256") == sha256(ROOT / "src" / "dgrs" / "analysis.py")
        and analysis.get("aggregate_script_sha256") == sha256(ROOT / "scripts" / "aggregate_results.py")
        and summary.get("status") == "complete"
        and summary.get("formal_analysis") is True
        and summary.get("eligible_for_independent_acceptance_verifier") is True
        and summary.get("registered_issues") == 0
        and summary.get("models") == list(MODELS)
        and summary.get("folds") == list(FOLDS)
        and summary.get("probe_rows") == 39375
        and summary.get("pcr_rows") == 12120
        and summary.get("figures") == 12
        and issues.empty
        and coverage_ok
        and ci_ok
    )

    figure_manifest = pd.read_csv(final / "figure_manifest.csv")
    pngs = sorted((ROOT / "figures" / "final").glob("*.png"))
    figure_ok = len(figure_manifest) == 12 and len(pngs) == 12
    expected_png_paths = {str(path.relative_to(REPO)): path for path in pngs}
    figure_ok = figure_ok and set(figure_manifest["path"].astype(str)) == set(expected_png_paths)
    for row in figure_manifest.itertuples(index=False):
        path = expected_png_paths.get(str(row.path))
        if path is None:
            figure_ok = False
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.load()
                nonempty = image.width > 0 and image.height > 0
        except Exception:
            nonempty = False
        figure_ok = figure_ok and (
            nonempty
            and sha256(path) == str(row.sha256)
            and path.stat().st_size == int(row.bytes)
            and _as_bool(row.decodable) is True
        )

    macro = pd.read_csv(final / "paired_macro_bootstrap_ci.csv")
    paired = pd.read_csv(final / "paired_differences_bootstrap_ci.csv")
    stability = pd.read_csv(final / "training_stability.csv")
    recomputed_decision, recomputed_comparisons, recomputed_gates = _recompute_decision(
        macro, paired, stability
    )
    stored_gates = pd.read_csv(final / "decision_gates.csv").sort_values(["comparison", "gate"]).reset_index(drop=True)
    recomputed_gates = recomputed_gates.sort_values(["comparison", "gate"]).reset_index(drop=True)
    gates_equal = (
        stored_gates[["comparison", "gate"]].equals(recomputed_gates[["comparison", "gate"]])
        and [_as_bool(value) for value in stored_gates["passed"]]
        == [_as_bool(value) for value in recomputed_gates["passed"]]
    )
    metric_hashes = decision.get("input_metric_sha256", {})
    expected_metric_files = {
        "probe_oof_metrics.csv",
        "pcr_oof_metrics.csv",
        "paired_differences_bootstrap_ci.csv",
        "paired_macro_bootstrap_ci.csv",
        "training_stability.csv",
    }
    metric_hashes_ok = set(metric_hashes) == expected_metric_files and all(
        metric_hashes[name] == sha256(final / name) for name in expected_metric_files
    )
    expected_thresholds = {
        "A_macro_gain": 0.05,
        "B_macro_or_transition_gain": 0.05,
        "C_macro_auroc_gain": 0.02,
        "ci_requirement": "95% paired patient bootstrap lower bound > 0",
        "max_validation_base_degradation": 0.05,
        "minimum_representation_std": 0.05,
    }
    decision_ok = (
        decision.get("formal_analysis") is True
        and decision.get("decision") == recomputed_decision
        and decision.get("comparisons") == recomputed_comparisons
        and decision.get("thresholds") == expected_thresholds
        and decision.get("bootstrap_conditional_on_single_training_seed") is True
        and decision.get("multiple_comparison_adjustment")
        == "none; pre-registered cells, interpret individual CIs conditionally"
        and gates_equal
        and metric_hashes_ok
    )

    report_text = report.read_text(encoding="utf-8") if report.is_file() else ""
    numbered_sections = len(re.findall(r"^##\s+\d+\.", report_text, flags=re.MULTILINE))
    cjk = len(re.findall(r"[\u3400-\u9fff]", report_text))
    answers = {
        letter
        for letter in "abcdefghi"
        if re.search(rf"^(?:###\s+|[-*]\s+|\*\*){letter}[\.、)]", report_text, flags=re.MULTILINE | re.IGNORECASE)
    }
    report_figure_links = set(re.findall(r"\((?:\.\./)?figures/final/([^\s)]+\.png)\)", report_text))
    report_ok = (
        report.is_file()
        and numbered_sections >= 18
        and cjk >= 500
        and answers == set("abcdefghi")
        and report_figure_links == {path.name for path in pngs}
        and "/home/" not in report_text
        and "/data/" not in report_text
    )

    architecture = {
        key: payload["architecture_contract"] for key, payload in payload_by_key.items()
    }
    g3 = [architecture[("G3", fold)] for fold in FOLDS]
    g4 = [architecture[("G4", fold)] for fold in FOLDS]
    criteria: list[dict[str, Any]] = []
    criteria.append(check("1. 使用 feature/ispy-clean-corejepa 分支", branch == "feature/ispy-clean-corejepa", branch))
    criteria.append(check("2. 使用 conda bowen", conda_name == "bowen", {"environment": conda_name, "python": sys.version.split()[0]}))
    criteria.append(check("3. 使用原五折 patient split", manifest.is_file() and sha256(manifest) == EXPECTED_MANIFEST_SHA, {"manifest_present": manifest.is_file(), "fold_manifest_sha256": sha256(manifest) if manifest.is_file() else "missing"}))
    criteria.append(check("4. 先完成 EXPERIMENT_PLAN.md", plan_ok, {"training_plan_sha256": sorted(plan_hashes), "public_plan_sha256": public_plan_sha, "certified_path_redaction": certified_redaction}))
    criteria.append(check("5. 完成 G1–G4 smoke tests", smoke_ok, {"models": sorted(smoke_model_names), "status": smoke.get("status"), "implementation_sha256": sorted(smoke_impl)}))
    criteria.append(check("6. 完成 lambda pilot", lambda_ok, {"candidate_lambdas": candidate_lambdas, "selected_lambda_ftv": lambda_selection.get("selected_lambda_ftv"), "validation_only": access.get("lambda_selected_on_validation_only")}))
    criteria.append(check("7. 完成 G0–G4 五折比较", checkpoint_ok and shared_initialization_ok and feature_ok, {"formal_checkpoints": len(checkpoint_paths), "histories": len(history_paths), "selected_history_rows": selected_history_rows, "frozen_feature_assets": len(feature_paths), "feature_shape": [808, 4, 192], "shared_initialization_pairs": 10}))
    criteria.append(check("8. 不使用 test 选择 lambda/checkpoint", all(item.get("test_data_used") is False for item in selections) and access.get("lambda_selected_on_validation_only") is True and predictions_ok and analysis_checks.get("formal_selections_20_test_blind", {}).get("passed") is True, {"formal_selection_assets": len(selections), "prediction_guards": predictions_ok, "lambda_validation_only": access.get("lambda_selected_on_validation_only")}))
    criteria.append(check("9. G3/G4 inference 不使用 FTV", all(item.get("ftv_is_forward_input") is False and item.get("observed_response_state") == "online_preprojector_r" for item in g3 + g4), {"grounded_contracts": 10, "ftv_forward_input": False}))
    criteria.append(check("10. G3 不使用 mask", all(item.get("backbone_input") == "DCE7" and item.get("roi_mask_use") == "absent" and item.get("roi_mask_backbone_input") is False for item in g3), {"folds": 5, "roi_mask_use": "absent"}))
    criteria.append(check("11. G4 mask 仅用于 normalized ROI pooling", all(item.get("backbone_input") == "DCE7" and item.get("roi_mask_use") == "normalized_occupancy_roi_mean_only" and item.get("roi_mask_backbone_input") is False for item in g4), {"folds": 5, "roi_mask_use": "normalized_occupancy_roi_mean_only"}))
    criteria.append(check("12. G4 state 无 mask volume/geometry", all({"mask_geometry", "voxel_count", "explicit_volume"}.issubset(item.get("forbidden_inputs_absent", [])) for item in g4), {"folds": 5, "explicit_geometry_scalar_route": False}))
    criteria.append(check("13. 完成 static FTV decodability", metrics_ok and len(static_ftv) == 20, {"oof_cells": len(static_ftv), "patients_per_cell": 375}))
    criteria.append(check("14. 完成 observed ΔFTV decodability", metrics_ok and len(change_ftv) == 15, {"oof_cells": len(change_ftv), "patients_per_cell": 375}))
    criteria.append(check("15. 完成 image-only pCR readout", metrics_ok and len(pcr_metrics) == 15, {"oof_cells": len(pcr_metrics), "patients_per_cell": 808, "decision_points": list(DECISION_POINTS)}))
    criteria.append(check("16. 完成 LD/sphericity secondary probes", metrics_ok and len(transfer) == 70 and set(transfer["target"]) == {"ld", "sphericity"}, {"oof_cells": len(transfer), "patients_per_cell": 375}))
    criteria.append(check("17. 保存 prediction-level results", predictions_ok and manifest_ok and coverage_ok, {"probe_files": len(probe_files), "pcr_files": len(pcr_files), "probe_rows": probe_rows, "pcr_rows": pcr_rows, "probe_patients": probe_patients, "pcr_patients": pcr_patients, "paired_patient_sets_exact": predictions_ok}))
    criteria.append(check("18. 输出完整中文报告", report_ok, {"numbered_sections": numbered_sections, "answers_a_to_i": sorted(answers), "cjk_characters": cjk, "figure_links": len(report_figure_links), "report_sha256": sha256(report) if report.is_file() else "missing"}))
    criteria.append(check("19. 最终做 GO/PARTIAL GO/NO-GO 决策", analysis_fresh and figure_ok and decision_ok, {"formal_analysis_eligible": analysis.get("eligible_for_independent_acceptance_verifier"), "analysis_checks_all_pass": all_analysis_checks_pass, "fresh_source_hashes": analysis_fresh, "figures_verified": len(pngs), "decision_recomputed": recomputed_decision, "decision_gates_recomputed": gates_equal, "input_metric_hashes_match": metric_hashes_ok}))

    all_passed = all(item["passed"] for item in criteria)
    acceptance = {
        "schema_version": 1,
        "status": "passed" if all_passed else "failed",
        "criteria_passed": sum(item["passed"] for item in criteria),
        "criteria_total": 19,
        "all_19_acceptance_criteria_passed": all_passed,
        "criteria": criteria,
        "formal_analysis_eligible": analysis.get("eligible_for_independent_acceptance_verifier") is True,
        "formal_analysis_checks_all_pass": all_analysis_checks_pass,
        "training_implementation_sha256": sorted(implementation_hashes),
        "analysis_evidence_sha256": sha256(final / "analysis_acceptance_evidence.json"),
        "contains_patient_level_rows": False,
        "contains_absolute_local_paths": False,
    }
    public_decision = {
        **decision,
        "source": "metrics/final/decision.json",
        "source_sha256": sha256(final / "decision.json"),
        "contains_patient_level_rows": False,
        "contains_absolute_local_paths": False,
    }
    return acceptance, public_decision


def atomic_write(path: Path, payload: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"输出已存在，默认拒绝覆盖: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    acceptance, decision = verify()
    atomic_write(ROOT / "metrics" / "acceptance_check.json", acceptance, args.overwrite)
    atomic_write(ROOT / "metrics" / "decision.json", decision, args.overwrite)
    print(json.dumps(acceptance, ensure_ascii=False, indent=2))
    if not acceptance["all_19_acceptance_criteria_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

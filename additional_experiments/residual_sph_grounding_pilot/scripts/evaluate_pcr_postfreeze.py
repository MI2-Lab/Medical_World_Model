#!/usr/bin/env python3
"""Open pCR only after freeze, then run the registered complementarity models."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys


os.umask(0o077)
sys.dont_write_bytecode = True
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
SRC = EXPERIMENT_ROOT / "src"
sys.path.insert(0, str(SRC))

# This import is boundary-only and performs no clinical or pCR I/O.
from residual_sph.contracts import file_sha256  # noqa: E402
from residual_sph.evaluation_lock import verify_representation_freeze  # noqa: E402
from residual_sph.preregistration import require_lock_sha256, verify_preregistration  # noqa: E402


CLINICAL_SHA256 = "b3355f8ac80cf8f0fa95722b8d8a8b73d96790e9ded5c491ddb5b2e6a7793436"
TARGET_SHA256 = "26fbde8590fde4612267f02d762af99d65926ff6d0206d0e500577ef394ff75d"
FOLD_SHA256 = "143e482d711225c0611006d99bd7345d2fa1a5c16c65fbaf8399341a0d26aa38"
DEFAULT_TARGET = REPO_ROOT / "additional_experiments/radiomics_next_change/data_audit/radiomics_transition_targets_raw.csv"
DEFAULT_DATA_CONTRACT = (
    REPO_ROOT
    / "additional_experiments/c1b_overlap_eligibility_ftv_stageb/manifests/"
    "stage_b_data_contract.private.json"
)
DATA_CONTRACT_SHA256 = "dd22f130043863d4fce8956061fca389894a31874567ed7929e139f32ff5ab27"
CONFIRMATION = REPO_ROOT / "additional_experiments/local_response_state_multiseed_confirmation"
PRIVATE_OUTPUT = EXPERIMENT_ROOT / "predictions" / "pcr_oof.private.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration-lock-sha256", required=True)
    parser.add_argument("--clinical-table", type=Path, required=True)
    parser.add_argument("--target-table", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--fold-manifest", type=Path)
    parser.add_argument("--data-contract", type=Path, default=DEFAULT_DATA_CONTRACT)
    args = parser.parse_args()

    # Mandatory first phase: no clinical path is resolved, hashed, imported, or
    # opened until both immutable locks pass.
    preregistration = verify_preregistration(EXPERIMENT_ROOT)
    require_lock_sha256(preregistration["lock_sha256"], args.preregistration_lock_sha256)
    representation_freeze = verify_representation_freeze(
        EXPERIMENT_ROOT,
        expected_preregistration_sha256=preregistration["lock_sha256"],
    )

    # The clinical/pCR boundary begins here.
    import numpy as np
    import pandas as pd
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    from residual_sph.evaluation import (
        evaluate_decision_gates,
        paired_fold_stratified_auroc_bootstrap,
    )
    from residual_sph.pcr_evaluation import (
        CLINICAL_FIELDS,
        MODEL_NAMES,
        TIMINGS,
        TrainOnlyClinicalEncoder,
        feature_sets,
        fit_logistic,
        timing_prefix,
    )
    from residual_sph.probes import load_feature_asset
    from residual_sph.provenance import load_s0_manifest, validate_s0_cell
    from residual_sph.targets import load_static_sph_ftv_table

    data_contract = args.data_contract.resolve()
    if not data_contract.is_file() or file_sha256(data_contract) != DATA_CONTRACT_SHA256:
        raise ValueError("Stage-B data contract is missing or hash-mismatched")
    contract = json.loads(data_contract.read_text(encoding="utf-8"))
    fold_value = Path(str(contract["fold_manifest"])).expanduser()
    contract_fold_manifest = (
        fold_value.resolve()
        if fold_value.is_absolute()
        else (data_contract.parent / fold_value).resolve()
    )
    fold_manifest = (
        args.fold_manifest.resolve()
        if args.fold_manifest is not None
        else contract_fold_manifest
    )

    for path, expected, label in (
        (args.clinical_table, CLINICAL_SHA256, "clinical/pCR table"),
        (args.target_table, TARGET_SHA256, "FTV target table"),
        (fold_manifest, FOLD_SHA256, "outer-fold manifest"),
    ):
        resolved = path.resolve()
        if not resolved.is_file() or file_sha256(resolved) != expected:
            raise ValueError(f"{label} is missing or hash-mismatched")
    public_outputs = (
        EXPERIMENT_ROOT / "metrics/table_pcr_complementarity.csv",
        EXPERIMENT_ROOT / "metrics/paired_bootstrap.csv",
        EXPERIMENT_ROOT / "metrics/pcr_effects.json",
        EXPERIMENT_ROOT / "metrics/decision.json",
    )
    preexisting_decision = None
    if public_outputs[3].is_file():
        preexisting_decision = json.loads(public_outputs[3].read_text(encoding="utf-8"))
        if preexisting_decision.get("status") != "FORMAL_EXECUTION_NOT_STARTED_RESOURCE_GUARD":
            raise FileExistsError("refusing to overwrite a completed decision")
    if PRIVATE_OUTPUT.exists() or any(path.exists() for path in public_outputs[:3]):
        raise FileExistsError("refusing to overwrite post-freeze pCR outputs")

    table = load_static_sph_ftv_table(
        args.target_table, TARGET_SHA256, expected_patient_count=375
    )
    clinical_columns = ("patient_id", "label_pcr") + CLINICAL_FIELDS
    clinical = pd.read_csv(args.clinical_table, usecols=list(clinical_columns))
    clinical["patient_id"] = clinical["patient_id"].astype(str)
    if clinical["patient_id"].duplicated().any():
        raise ValueError("clinical table has duplicate patients")
    clinical = clinical.set_index("patient_id", drop=False)
    if not set(table.patient_ids).issubset(clinical.index):
        raise ValueError("clinical table misses complete-case target patients")
    target_index = table.patient_to_index
    rows: list[dict[str, object]] = []
    hyperparameter_rows: list[dict[str, object]] = []
    for arm in ("S0", "S1", "S2"):
        for seed in (2026, 3026):
            for fold in range(5):
                if arm == "S0":
                    feature_path = (
                        CONFIRMATION / "features/formal_4x8" / f"seed_{seed}" / "LOCAL3"
                        / f"fold_{fold}" / "response_state.private.npz"
                    )
                else:
                    feature_path = (
                        EXPERIMENT_ROOT / "features/formal_4x8" / f"seed_{seed}" / arm
                        / f"fold_{fold}" / "response_state.private.npz"
                    )
                asset = load_feature_asset(
                    feature_path,
                    analysis_arm=arm,
                    seed_base=seed,
                    fold=fold,
                )
                if arm == "S0":
                    s0_manifest = load_s0_manifest(
                        EXPERIMENT_ROOT / "manifests/s0_confirmation_provenance.json"
                    )
                    s0_run = (
                        CONFIRMATION / "checkpoints/formal_4x8" / f"seed_{seed}"
                        / "LOCAL3" / f"fold_{fold}"
                    )
                    validate_s0_cell(
                        s0_manifest,
                        seed_base=seed,
                        fold=fold,
                        selection_path=s0_run / "selection.json",
                        checkpoint_path=s0_run / "selected.pt",
                        feature_path=feature_path,
                        feature_metadata_path=feature_path.with_suffix(".metadata.json"),
                    )
                keep = np.asarray([patient_id in target_index for patient_id in asset.patient_ids], dtype=bool)
                patient_ids = np.asarray(asset.patient_ids, dtype=str)[keep]
                split = asset.splits[keep]
                response = asset.response_state[keep]
                if len(patient_ids) != 375 or set(split) != {"train", "val", "test"}:
                    raise ValueError("pCR complete-case feature cohort drifted")
                clinical_aligned = clinical.loc[patient_ids]
                labels = clinical_aligned["label_pcr"].to_numpy(dtype=np.int64)
                if not np.isin(labels, (0, 1)).all():
                    raise ValueError("pCR label is not binary")
                ftv = np.stack([table.ftv[target_index[patient_id]] for patient_id in patient_ids])
                indices = {name: np.flatnonzero(split == name) for name in ("train", "val", "test")}
                encoder = TrainOnlyClinicalEncoder().fit(clinical_aligned.iloc[indices["train"]])
                clinical_matrix = encoder.transform(clinical_aligned)
                for timing in TIMINGS:
                    mri, ftv_prefix = timing_prefix(response, ftv, timing)
                    matrices = feature_sets(clinical_matrix, mri, ftv_prefix)
                    if tuple(matrices) != MODEL_NAMES:
                        raise AssertionError("post-freeze model-set order drifted")
                    for model_name, matrix in matrices.items():
                        fit = fit_logistic(
                            matrix[indices["train"]], labels[indices["train"]],
                            matrix[indices["val"]], labels[indices["val"]],
                        )
                        probability = fit.predict_probability(matrix[indices["test"]])
                        for offset, row_index in enumerate(indices["test"]):
                            rows.append(
                                {
                                    "patient_id": patient_ids[row_index],
                                    "fold": fold,
                                    "label_pcr": int(labels[row_index]),
                                    "seed_base": seed,
                                    "arm": arm,
                                    "timing": timing,
                                    "model": model_name,
                                    "probability": float(probability[offset]),
                                }
                            )
                        hyperparameter_rows.append(
                            {
                                "seed_base": seed,
                                "arm": arm,
                                "fold": fold,
                                "timing": timing,
                                "model": model_name,
                                "selected_c": fit.selected_c,
                                "validation_auroc": fit.validation_auroc,
                                "train_rows": len(indices["train"]),
                                "validation_rows": len(indices["val"]),
                                "refit_after_validation": False,
                                "test_predict_calls": 1,
                            }
                        )
    predictions = pd.DataFrame(rows)
    PRIVATE_OUTPUT.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    predictions.to_csv(PRIVATE_OUTPUT, index=False)
    PRIVATE_OUTPUT.chmod(0o600)
    hyper_path = EXPERIMENT_ROOT / "predictions/pcr_hyperparameters.private.csv"
    pd.DataFrame(hyperparameter_rows).to_csv(hyper_path, index=False)
    hyper_path.chmod(0o600)

    metric_rows: list[dict[str, object]] = []
    for identity, group in predictions.groupby(["seed_base", "arm", "timing", "model"], sort=True):
        seed, arm, timing, model_name = identity
        if len(group) != 375 or group["patient_id"].duplicated().any() or set(group["fold"]) != set(range(5)):
            raise ValueError(f"pCR OOF coverage drifted: {identity}")
        labels = group["label_pcr"].to_numpy(dtype=np.int64)
        probability = group["probability"].to_numpy(dtype=np.float64)
        metric_rows.append(
            {
                "seed_base": int(seed),
                "arm": arm,
                "timing": timing,
                "model": model_name,
                "n": len(group),
                "n_positive": int(labels.sum()),
                "auroc": float(roc_auc_score(labels, probability)),
                "auprc": float(average_precision_score(labels, probability)),
                "brier": float(brier_score_loss(labels, probability)),
                "aggregation": "pooled_5fold_oof_within_seed",
            }
        )
    metrics = pd.DataFrame(metric_rows)

    def paired(
        seed: int,
        timing: str,
        ref_arm: str,
        ref_model: str,
        cmp_arm: str,
        cmp_model: str,
        *,
        bootstrap_seed: int,
    ) -> dict[str, object]:
        reference = predictions.loc[
            predictions["seed_base"].eq(seed) & predictions["arm"].eq(ref_arm)
            & predictions["timing"].eq(timing) & predictions["model"].eq(ref_model)
        ].sort_values("patient_id")
        comparison = predictions.loc[
            predictions["seed_base"].eq(seed) & predictions["arm"].eq(cmp_arm)
            & predictions["timing"].eq(timing) & predictions["model"].eq(cmp_model)
        ].sort_values("patient_id")
        if not reference[["patient_id", "fold", "label_pcr"]].reset_index(drop=True).equals(
            comparison[["patient_id", "fold", "label_pcr"]].reset_index(drop=True)
        ):
            raise ValueError("paired pCR comparison is not patient-aligned")
        return paired_fold_stratified_auroc_bootstrap(
            reference["fold"].to_numpy(),
            reference["label_pcr"].to_numpy(),
            reference["probability"].to_numpy(),
            comparison["probability"].to_numpy(),
            n_bootstrap=2_000,
            seed=bootstrap_seed,
        )

    bootstrap_rows: list[dict[str, object]] = []
    e5: dict[str, dict[str, float]] = {timing: {} for timing in TIMINGS}
    e6_s2: dict[str, dict[str, float]] = {timing: {} for timing in TIMINGS}
    e6_s0: dict[str, dict[str, float]] = {timing: {} for timing in TIMINGS}
    cm_s2: dict[str, dict[str, float]] = {timing: {} for timing in TIMINGS}
    cm_s0: dict[str, dict[str, float]] = {timing: {} for timing in TIMINGS}
    paired_metric_effects: dict[
        str, dict[str, dict[str, dict[str, float]]]
    ] = {
        effect_name: {timing: {} for timing in TIMINGS}
        for effect_name in (
            "E5_S2_minus_S0_MRI_only",
            "E6_S2_C_plus_F_plus_M_minus_C_plus_F",
            "S0_C_plus_F_plus_M_minus_C_plus_F",
            "S2_C_plus_M_minus_C",
            "S0_C_plus_M_minus_C",
        )
    }
    comparison_specs = (
        (
            "E5_S2_minus_S0_M",
            "E5_S2_minus_S0_MRI_only",
            "S0",
            "M",
            "S2",
            "M",
            e5,
        ),
        (
            "E6_S2_CFM_minus_CF",
            "E6_S2_C_plus_F_plus_M_minus_C_plus_F",
            "S2",
            "C+F",
            "S2",
            "C+F+M",
            e6_s2,
        ),
        (
            "E6_S0_CFM_minus_CF",
            "S0_C_plus_F_plus_M_minus_C_plus_F",
            "S0",
            "C+F",
            "S0",
            "C+F+M",
            e6_s0,
        ),
        (
            "S2_CM_minus_C",
            "S2_C_plus_M_minus_C",
            "S2",
            "C",
            "S2",
            "C+M",
            cm_s2,
        ),
        (
            "S0_CM_minus_C",
            "S0_C_plus_M_minus_C",
            "S0",
            "C",
            "S0",
            "C+M",
            cm_s0,
        ),
    )
    bootstrap_counter = 0
    for timing in TIMINGS:
        for seed in (2026, 3026):
            for (
                comparison_name,
                effect_name,
                ref_arm,
                ref_model,
                cmp_arm,
                cmp_model,
                destination,
            ) in comparison_specs:
                result = paired(
                    seed,
                    timing,
                    ref_arm,
                    ref_model,
                    cmp_arm,
                    cmp_model,
                    bootstrap_seed=260_811 + bootstrap_counter,
                )
                bootstrap_counter += 1
                destination[timing][str(seed)] = float(result["delta_auroc"])
                paired_metric_effects[effect_name][timing][str(seed)] = {
                    "delta_auroc": float(result["delta_auroc"]),
                    "delta_auprc": float(result["delta_auprc"]),
                    "brier_improvement": float(result["brier_improvement"]),
                }
                bootstrap_rows.append(
                    {
                        "comparison": comparison_name,
                        "timing": timing,
                        "seed_base": seed,
                        **result,
                    }
                )

    paired_effect_summaries: dict[str, dict[str, object]] = {}
    effect_metrics = ("delta_auroc", "delta_auprc", "brier_improvement")
    expected_seed_keys = {"2026", "3026"}
    for effect_name, timing_values in paired_metric_effects.items():
        paired_effect_summaries[effect_name] = {}
        for timing, by_seed in timing_values.items():
            if set(by_seed) != expected_seed_keys:
                raise ValueError(
                    f"paired pCR effect lacks both formal seeds: {effect_name}/{timing}"
                )
            paired_effect_summaries[effect_name][timing] = {
                "by_seed": by_seed,
                "two_seed_mean": {
                    metric: float(
                        sum(by_seed[seed][metric] for seed in sorted(by_seed))
                        / len(by_seed)
                    )
                    for metric in effect_metrics
                },
                "both_seeds_positive": {
                    metric: bool(
                        all(by_seed[seed][metric] > 0.0 for seed in by_seed)
                    )
                    for metric in effect_metrics
                },
            }
    effects = {
        "E5_S2_minus_S0_MRI_only": e5,
        "E6_S2_C_plus_F_plus_M_minus_C_plus_F": e6_s2,
        "S0_C_plus_F_plus_M_minus_C_plus_F": e6_s0,
        "S2_C_plus_M_minus_C": cm_s2,
        "S0_C_plus_M_minus_C": cm_s0,
        "paired_metric_effect_summaries": paired_effect_summaries,
    }
    representation_effects = json.loads(
        (EXPERIMENT_ROOT / "metrics/representation_effects.json").read_text(encoding="utf-8")
    )
    gate_d = {timing: e6_s2[timing] for timing in ("T0", "T0-T1", "T0-T2")}
    safety = {
        int(seed): {int(fold): bool(value) for fold, value in folds.items()}
        for seed, folds in representation_effects["optimization_safety"]["by_seed_fold"].items()
    }
    decision = evaluate_decision_gates(
        effects=representation_effects,
        optimization_safety=safety,
        downstream_delta_auroc=gate_d,
    )
    decision.update(
        {
            "schema_version": 1,
            "experiment": "residual_sph_grounding_pilot",
            "status": "FORMAL_TWO_SEED_PILOT_COMPLETE",
            "pcr_evaluation_was_post_freeze": True,
            "representation_freeze_sha256": representation_freeze["freeze_sha256"],
            "pcr_effects": effects,
        }
    )
    metrics.to_csv(public_outputs[0], index=False)
    pd.DataFrame(bootstrap_rows).to_csv(public_outputs[1], index=False)
    public_outputs[2].write_text(json.dumps(effects, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    public_outputs[3].write_text(
        json.dumps(decision, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    execution_status = {
        "schema_version": 1,
        "experiment": "residual_sph_grounding_pilot",
        "status": "FORMAL_EXECUTION_COMPLETE",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_new_training_cells_required": 30,
        "formal_new_training_cells_completed": 30,
        "s0_reference_cells_verified": 10,
        "representation_cells_frozen": 40,
        "pcr_evaluation_was_post_freeze": True,
        "representation_freeze_sha256": representation_freeze["freeze_sha256"],
        "scientific_gates_evaluated": True,
        "classification_assigned": True,
        "decision_status": decision["status"],
    }
    (EXPERIMENT_ROOT / "metrics/execution_status.json").write_text(
        json.dumps(execution_status, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

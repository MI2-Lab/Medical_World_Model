#!/usr/bin/env python3
"""只用 fold-0 train/validation response state 选择统一 lambda_FTV。

输入必须是训练器 ``--export-pilot-features`` 产生的 train+val-only NPZ；只要
NPZ 中出现 test split，本脚本立即拒绝。脚本不会打开正式 808-patient feature、
outer-test target、pCR label 或任何 test prediction。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from dgrs.config import file_sha256, load_config, resolve_path  # noqa: E402
from dgrs.targets import patient_hash  # noqa: E402


LAMBDA_CANDIDATES = (0.02, 0.05, 0.1, 0.25)
TIMEPOINTS = ("T0", "T1", "T2", "T3")
FIXED_RIDGE_ALPHA = 1.0
MIN_REPRESENTATION_STD = 0.05
MAX_BASE_DEGRADATION = 0.05
EFFECTIVE_GAIN = 0.03
MAX_OTHER_PAIRING_DROP = -0.02


class PilotInputError(RuntimeError):
    """Pilot 输入违反 train/val-only 或完整矩阵契约。"""


@dataclass(frozen=True)
class PilotAsset:
    path: Path
    run_name: str
    model: str
    lambda_ftv: float
    response: np.ndarray
    patient_ids: np.ndarray
    splits: np.ndarray
    target: np.ndarray
    target_valid: np.ndarray
    val_base_loss: float
    declared_representation_std: float
    computed_representation_std: float
    fold: int
    fold_manifest_sha256: str
    canonical_train_patient_hash: str
    canonical_val_patient_hash: str
    canonical_test_patient_hash: str
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_implementation_sha256: str
    transform_path: Path
    transform_sha256: str


def _scalar(npz: Mapping[str, np.ndarray], names: tuple[str, ...], label: str) -> Any:
    for name in names:
        if name in npz:
            value = np.asarray(npz[name])
            if value.size != 1:
                raise PilotInputError(f"{label} 必须为标量，实际 shape={value.shape}")
            return value.reshape(-1)[0].item()
    raise PilotInputError(f"pilot NPZ 缺少 {label}（候选字段 {names}）")


def _normalise_model(value: Any) -> str:
    text = str(value).strip().upper().replace("MODEL_", "")
    if text.startswith("G") and text[1:].isdigit():
        text = f"G{int(text[1:])}"
    return text


def _hash_text_or_file(value: Any) -> str:
    text = str(value).strip().lower()
    if len(text) == 64 and all(character in "0123456789abcdef" for character in text):
        return text
    path = Path(text)
    if path.is_file():
        return file_sha256(path)
    return ""


def load_pilot_asset(path: Path) -> PilotAsset:
    """加载一个 train+val-only NPZ，并在任何拟合前完成泄漏检查。"""

    try:
        archive = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise PilotInputError(f"无法安全读取 pilot NPZ {path}: {exc}") from exc
    with archive as npz:
        response_key = next((item for item in ("response", "responses", "r", "features") if item in npz), None)
        if response_key is None:
            raise PilotInputError(f"pilot NPZ 缺 response: {path}")
        response = np.asarray(npz[response_key], dtype=np.float64)
        patient_ids = np.asarray(npz["patient_ids"]).astype(str)
        split_key = "splits" if "splits" in npz else "split" if "split" in npz else None
        if split_key is None:
            raise PilotInputError(f"pilot NPZ 必须显式含 splits: {path}")
        splits = np.char.lower(np.char.strip(np.asarray(npz[split_key]).astype(str)))
        target_key = next(
            (item for item in ("ftv_standardized", "target_standardized", "ftv_target") if item in npz),
            None,
        )
        if target_key is None:
            raise PilotInputError(f"pilot NPZ 缺 train-fold transformed FTV target: {path}")
        target = np.asarray(npz[target_key], dtype=np.float64)
        valid_key = "ftv_valid" if "ftv_valid" in npz else "target_valid" if "target_valid" in npz else None
        target_valid = np.asarray(npz[valid_key], dtype=bool) if valid_key else np.isfinite(target)
        model = _normalise_model(_scalar(npz, ("model", "model_name"), "model"))
        lambda_ftv = float(_scalar(npz, ("lambda_ftv", "lambda"), "lambda_ftv"))
        val_base_loss = float(_scalar(npz, ("val_base_loss", "validation_base_loss"), "val_base_loss"))
        declared_std = float(
            _scalar(
                npz,
                ("representation_std", "val_representation_std", "visit_feature_std"),
                "representation_std",
            )
        )
        fold = int(_scalar(npz, ("fold",), "fold"))
        manifest_sha = str(_scalar(npz, ("fold_manifest_sha256",), "fold_manifest_sha256"))
        train_hash = str(_scalar(npz, ("canonical_train_patient_hash",), "canonical train hash"))
        val_hash = str(_scalar(npz, ("canonical_val_patient_hash",), "canonical val hash"))
        test_hash = str(_scalar(npz, ("canonical_test_patient_hash",), "canonical test hash"))
        checkpoint_path = Path(str(_scalar(npz, ("checkpoint",), "checkpoint path"))).resolve()
        checkpoint_sha = str(_scalar(npz, ("checkpoint_sha256",), "checkpoint SHA-256")).lower()
        implementation_sha = str(
            _scalar(npz, ("checkpoint_implementation_sha256",), "checkpoint implementation SHA-256")
        ).lower()
        transform_path = Path(str(_scalar(npz, ("transform_path",), "transform path"))).resolve()
        transform_sha = str(_scalar(npz, ("transform_sha256",), "transform SHA-256")).lower()

    if response.ndim != 3 or response.shape[1:] != (4, 192):
        raise PilotInputError(f"response 必须为 [N,4,192]，实际 {response.shape}: {path}")
    n = response.shape[0]
    if patient_ids.shape != (n,) or splits.shape != (n,):
        raise PilotInputError(f"patient_ids/splits 必须与 response 第一维一致: {path}")
    if target.shape != (n, 4) or target_valid.shape != (n, 4):
        raise PilotInputError(f"FTV target/valid 必须为 [N,4]，实际 {target.shape}/{target_valid.shape}: {path}")
    observed_splits = set(splits.tolist())
    if "test" in observed_splits:
        raise PilotInputError(f"严重：pilot NPZ 含 test feature，拒绝读取/选择: {path}")
    if observed_splits != {"train", "val"}:
        raise PilotInputError(f"pilot splits 必须恰为 train/val，实际 {sorted(observed_splits)}: {path}")
    if len(set(patient_ids.tolist())) != n:
        raise PilotInputError(f"pilot patient_ids 重复: {path}")
    if not np.isfinite(response).all():
        raise PilotInputError(f"pilot response 含 NaN/Inf: {path}")
    if not np.isfinite(target[target_valid]).all():
        raise PilotInputError(f"pilot valid FTV 含 NaN/Inf: {path}")
    if not math.isfinite(val_base_loss) or val_base_loss <= 0:
        raise PilotInputError(f"val_base_loss 必须有限且 >0: {path}")
    if not math.isfinite(declared_std) or declared_std < 0:
        raise PilotInputError(f"representation_std 非法: {path}")
    if model not in {"G1", "G2", "G3", "G4"}:
        raise PilotInputError(f"pilot model 必须为 G1–G4，实际 {model}: {path}")
    if model in {"G1", "G2"} and not math.isclose(lambda_ftv, 0.0, abs_tol=1e-12):
        raise PilotInputError(f"{model} baseline lambda 必须为 0: {path}")
    if model in {"G3", "G4"} and not any(math.isclose(lambda_ftv, item, abs_tol=1e-12) for item in LAMBDA_CANDIDATES):
        raise PilotInputError(f"{model} lambda 不在锁定集合 {LAMBDA_CANDIDATES}: {path}")
    if fold != 0:
        raise PilotInputError(f"lambda pilot 只接受 fold 0，实际 {fold}: {path}")
    hashes = {
        "fold_manifest_sha256": manifest_sha,
        "canonical_train_patient_hash": train_hash,
        "canonical_val_patient_hash": val_hash,
        "canonical_test_patient_hash": test_hash,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_implementation_sha256": implementation_sha,
        "transform_sha256": transform_sha,
    }
    for label, value in hashes.items():
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
            raise PilotInputError(f"{label} 非法: {path}")
    if not checkpoint_path.is_file() or file_sha256(checkpoint_path) != checkpoint_sha:
        raise PilotInputError(f"checkpoint path/SHA 不一致: {path}")
    if not transform_path.is_file() or file_sha256(transform_path) != transform_sha:
        raise PilotInputError(f"transform path/SHA 不一致: {path}")
    val = splits == "val"
    if val.sum() < 3:
        raise PilotInputError(f"validation 患者不足 3: {path}")
    # 对每个 visit/dimension 跨 validation patient 求 std，再取平均；它直接对应
    # 跨患者 response-state spread，不把 4 visits 当独立患者扩大样本数。
    computed_std = float(np.std(response[val], axis=0, ddof=0).mean())
    return PilotAsset(
        path=path.resolve(),
        run_name=path.parent.name,
        model=model,
        lambda_ftv=lambda_ftv,
        response=response,
        patient_ids=patient_ids,
        splits=splits,
        target=target,
        target_valid=target_valid,
        val_base_loss=val_base_loss,
        declared_representation_std=declared_std,
        computed_representation_std=computed_std,
        fold=fold,
        fold_manifest_sha256=manifest_sha,
        canonical_train_patient_hash=train_hash,
        canonical_val_patient_hash=val_hash,
        canonical_test_patient_hash=test_hash,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha,
        checkpoint_implementation_sha256=implementation_sha,
        transform_path=transform_path,
        transform_sha256=transform_sha,
    )


def _safe_spearman(target: np.ndarray, prediction: np.ndarray) -> float:
    if len(target) < 3 or np.std(target) <= 0 or np.std(prediction) <= 0:
        return math.nan
    value = float(spearmanr(target, prediction).statistic)
    return value if math.isfinite(value) else math.nan


def fixed_ridge_validation(asset: PilotAsset) -> dict[str, Any]:
    """四时点独立固定 alpha=1 Ridge；scaler/Ridge 只 fit fold0 train。"""

    train = asset.splits == "train"
    val = asset.splits == "val"
    rows = []
    for index, timepoint in enumerate(TIMEPOINTS):
        train_valid = train & asset.target_valid[:, index]
        val_valid = val & asset.target_valid[:, index]
        if train_valid.sum() < 3 or val_valid.sum() < 3:
            raise PilotInputError(
                f"{asset.run_name}/{timepoint} 有效 train/val FTV 不足: "
                f"{train_valid.sum()}/{val_valid.sum()}"
            )
        model = make_pipeline(StandardScaler(), Ridge(alpha=FIXED_RIDGE_ALPHA))
        model.fit(asset.response[train_valid, index], asset.target[train_valid, index])
        prediction = model.predict(asset.response[val_valid, index])
        truth = asset.target[val_valid, index]
        rows.append(
            {
                "timepoint": timepoint,
                "train_patients": int(train_valid.sum()),
                "validation_patients": int(val_valid.sum()),
                "spearman": _safe_spearman(truth, prediction),
                "r2": float(r2_score(truth, prediction)) if np.var(truth) > 0 else math.nan,
                "prediction_variance": float(np.var(prediction, ddof=1)),
                "target_variance": float(np.var(truth, ddof=1)),
            }
        )
    macro_spearman = float(np.nanmean([row["spearman"] for row in rows]))
    macro_r2 = float(np.nanmean([row["r2"] for row in rows]))
    return {
        "timepoints": rows,
        "macro_spearman": macro_spearman,
        "macro_r2": macro_r2,
        "fixed_alpha": FIXED_RIDGE_ALPHA,
        "train_only_scaler_and_ridge": True,
        "validation_predict_only": True,
    }


def _same_patient_target_contract(assets: list[PilotAsset]) -> None:
    reference = assets[0]
    reference_order = np.argsort(reference.patient_ids)
    for asset in assets[1:]:
        order = np.argsort(asset.patient_ids)
        if not np.array_equal(asset.patient_ids[order], reference.patient_ids[reference_order]):
            raise PilotInputError("pilot 各 run train/val patient set 不一致")
        if not np.array_equal(asset.splits[order], reference.splits[reference_order]):
            raise PilotInputError("pilot 各 run patient split 不一致")
        if not np.array_equal(asset.target_valid[order], reference.target_valid[reference_order]):
            raise PilotInputError("pilot 各 run FTV valid mask 不一致")
        valid = asset.target_valid[order]
        if not np.allclose(
            asset.target[order][valid], reference.target[reference_order][valid], rtol=0, atol=1e-9
        ):
            raise PilotInputError("pilot 各 run standardized FTV target 不一致")
    hashes = {asset.transform_sha256 for asset in assets if asset.transform_sha256}
    if len(hashes) > 1:
        raise PilotInputError("pilot 各 run FTV transform hash 不一致")


def canonical_fold0_contract() -> dict[str, Any]:
    config = load_config(EXPERIMENT_ROOT / "configs" / "base.yaml")
    manifest_path = resolve_path(config["data"]["fold_manifest"])
    expected_sha = str(config["data"]["fold_manifest_sha256"])
    if file_sha256(manifest_path) != expected_sha:
        raise PilotInputError("锁定 fold manifest SHA-256 不匹配")
    frame = pd.read_csv(manifest_path)
    required = {"patient_id", "fold", "split"}
    if missing := required.difference(frame.columns):
        raise PilotInputError(f"fold manifest 缺列: {sorted(missing)}")
    fold0 = frame.loc[frame["fold"].eq(0)].copy()
    fold0["patient_id"] = fold0["patient_id"].astype(str)
    if len(fold0) != 808 or fold0["patient_id"].duplicated().any():
        raise PilotInputError("fold 0 manifest 必须唯一覆盖 808 名患者")
    train = fold0.loc[fold0["split"].eq("train"), "patient_id"].tolist()
    val = fold0.loc[fold0["split"].eq("val"), "patient_id"].tolist()
    test = fold0.loc[fold0["split"].eq("test"), "patient_id"].tolist()
    if set(train) & set(val) or (set(train) | set(val)) & set(test):
        raise PilotInputError("fold 0 canonical splits 不互斥")
    return {
        "fold_manifest_sha256": expected_sha,
        "patient_ids": np.asarray(train + val),
        "splits": np.asarray(["train"] * len(train) + ["val"] * len(val)),
        "train_patient_hash": patient_hash(train),
        "val_patient_hash": patient_hash(val),
        "test_patient_hash": patient_hash(test),
    }


def _validate_canonical_fold0(
    assets: list[PilotAsset], contract: Mapping[str, Any]
) -> None:
    expected_ids = np.asarray(contract["patient_ids"]).astype(str)
    expected_splits = np.asarray(contract["splits"]).astype(str)
    for asset in assets:
        if not np.array_equal(asset.patient_ids, expected_ids) or not np.array_equal(
            asset.splits, expected_splits
        ):
            raise PilotInputError(
                f"{asset.path} 的 patient/split 不是 canonical fold0 train+val；"
                "可能包含被误标为 validation 的 test patient"
            )
        expected = {
            "fold_manifest_sha256": str(contract["fold_manifest_sha256"]),
            "canonical_train_patient_hash": str(contract["train_patient_hash"]),
            "canonical_val_patient_hash": str(contract["val_patient_hash"]),
            "canonical_test_patient_hash": str(contract["test_patient_hash"]),
        }
        for field, value in expected.items():
            if str(getattr(asset, field)) != value:
                raise PilotInputError(f"{asset.path} 的 {field} 与 canonical manifest 不一致")
    implementation_hashes = {asset.checkpoint_implementation_sha256 for asset in assets}
    if len(implementation_hashes) != 1:
        raise PilotInputError("pilot checkpoint training implementation SHA 不一致")


def discover_assets(
    input_root: Path, canonical_contract: Mapping[str, Any] | None = None
) -> list[PilotAsset]:
    paths = sorted(input_root.glob("*/fold_0_train_val_features.npz"))
    if not paths:
        paths = sorted(input_root.rglob("fold_0_train_val_features.npz")) if input_root.is_dir() else []
    if not paths:
        raise PilotInputError(f"未发现 train+val-only pilot NPZ: {input_root}")
    assets = [load_pilot_asset(path) for path in paths]
    keys = [(asset.model, round(asset.lambda_ftv, 10)) for asset in assets]
    if len(keys) != len(set(keys)):
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        raise PilotInputError(f"同一 model/lambda 有多个 pilot asset: {duplicates}")
    expected = {("G1", 0.0), ("G2", 0.0)} | {
        (model, value) for model in ("G3", "G4") for value in LAMBDA_CANDIDATES
    }
    observed = set(keys)
    if observed != expected:
        raise PilotInputError(
            f"pilot 必须恰有 G1/G2 paired baseline + G3/G4×4 候选；"
            f"缺 {sorted(expected-observed)}，多 {sorted(observed-expected)}"
        )
    _same_patient_target_contract(assets)
    _validate_canonical_fold0(
        assets, canonical_contract if canonical_contract is not None else canonical_fold0_contract()
    )
    return assets


def select_lambda(assets: list[PilotAsset]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """执行 joint G3/G4 gate，返回候选证据与锁定选择。"""

    ridge = {(asset.model, asset.lambda_ftv): fixed_ridge_validation(asset) for asset in assets}
    by_key = {(asset.model, asset.lambda_ftv): asset for asset in assets}
    baselines = {"G3": "G1", "G4": "G2"}
    rows = []
    for value in LAMBDA_CANDIDATES:
        pairing_rows = []
        for grounded, baseline in baselines.items():
            asset = by_key[(grounded, value)]
            baseline_asset = by_key[(baseline, 0.0)]
            base_degradation = (
                asset.val_base_loss - baseline_asset.val_base_loss
            ) / max(baseline_asset.val_base_loss, 1e-12)
            gain = ridge[(grounded, value)]["macro_spearman"] - ridge[(baseline, 0.0)]["macro_spearman"]
            finite = bool(
                np.isfinite(asset.response).all()
                and math.isfinite(asset.val_base_loss)
                and math.isfinite(asset.declared_representation_std)
                and math.isfinite(asset.computed_representation_std)
                and math.isfinite(ridge[(grounded, value)]["macro_spearman"])
            )
            stable = bool(
                finite
                and asset.declared_representation_std >= MIN_REPRESENTATION_STD
                and asset.computed_representation_std >= MIN_REPRESENTATION_STD
            )
            base_pass = bool(base_degradation <= MAX_BASE_DEGRADATION + 1e-12)
            pairing_rows.append(
                {
                    "lambda_ftv": value,
                    "grounded_model": grounded,
                    "baseline_model": baseline,
                    "run_name": asset.run_name,
                    "source_npz": str(asset.path),
                    "source_npz_sha256": file_sha256(asset.path),
                    "checkpoint_sha256": asset.checkpoint_sha256,
                    "transform_sha256": asset.transform_sha256,
                    "val_base_loss": asset.val_base_loss,
                    "paired_baseline_val_base_loss": baseline_asset.val_base_loss,
                    "base_degradation_fraction": base_degradation,
                    "declared_representation_std": asset.declared_representation_std,
                    "computed_validation_representation_std": asset.computed_representation_std,
                    "validation_macro_ftv_spearman": ridge[(grounded, value)]["macro_spearman"],
                    "paired_baseline_macro_ftv_spearman": ridge[(baseline, 0.0)]["macro_spearman"],
                    "macro_spearman_gain": gain,
                    "validation_macro_ftv_r2": ridge[(grounded, value)]["macro_r2"],
                    "finite": finite,
                    "representation_stability_pass": stable,
                    "base_degradation_pass": base_pass,
                }
            )
        gains = [item["macro_spearman_gain"] for item in pairing_rows]
        joint_safe = all(
            item["representation_stability_pass"] and item["base_degradation_pass"]
            for item in pairing_rows
        )
        joint_effective = bool(
            joint_safe and max(gains) >= EFFECTIVE_GAIN and min(gains) >= MAX_OTHER_PAIRING_DROP
        )
        for item in pairing_rows:
            item["joint_safe"] = joint_safe
            item["joint_effective"] = joint_effective
            item["joint_pairing_mean_gain"] = float(np.mean(gains))
            rows.append(item)
    evidence = pd.DataFrame(rows)
    effective = sorted(evidence.loc[evidence["joint_effective"], "lambda_ftv"].unique())
    if effective:
        selected = float(effective[0])
        mode = "smallest_effective_lambda"
        status = "selected"
        grounding_evidence = "sufficient_by_preregistered_validation_gate"
    else:
        safe = evidence.loc[evidence["joint_safe"]].groupby("lambda_ftv", as_index=False)[
            "joint_pairing_mean_gain"
        ].first()
        if safe.empty:
            selected = math.nan
            mode = "no_safe_candidate"
            status = "blocked"
            grounding_evidence = "insufficient_and_no_candidate_passed_stability_plus_base_gate"
        else:
            # fallback 先最大 mean gain，再以较小 lambda 打破平局。
            chosen = safe.sort_values(
                ["joint_pairing_mean_gain", "lambda_ftv"], ascending=[False, True]
            ).iloc[0]
            selected = float(chosen["lambda_ftv"])
            mode = "fallback_highest_joint_mean_gain_among_safe_candidates"
            status = "selected_with_insufficient_grounding_evidence"
            grounding_evidence = "insufficient_by_preregistered_effectiveness_gate"
    selected_rows = evidence.loc[evidence["lambda_ftv"].eq(selected)] if math.isfinite(selected) else evidence.iloc[:0]
    result = {
        "schema_version": 1,
        "status": status,
        "selected_lambda_ftv": selected,
        "selection_mode": mode,
        "pilot_grounding_evidence": grounding_evidence,
        "fold": 0,
        "models": {
            "paired_baselines": ["G1", "G2"],
            "grounded_candidates": ["G3", "G4"],
        },
        "candidate_lambdas": list(LAMBDA_CANDIDATES),
        "selected_same_lambda_for_g3_and_g4": bool(math.isfinite(selected)),
        "fixed_ridge_alpha": FIXED_RIDGE_ALPHA,
        "thresholds": {
            "minimum_representation_std": MIN_REPRESENTATION_STD,
            "maximum_validation_base_degradation": MAX_BASE_DEGRADATION,
            "at_least_one_pairing_macro_spearman_gain": EFFECTIVE_GAIN,
            "other_pairing_minimum_gain": MAX_OTHER_PAIRING_DROP,
        },
        "selected_pairing_evidence": selected_rows.to_dict(orient="records"),
        "data_access_contract": {
            "allowed_splits": ["train", "val"],
            "observed_splits": sorted(set(assets[0].splits.tolist())),
            "test_features_loaded": False,
            "test_ftv_loaded": False,
            "pcr_loaded": False,
            "test_auroc_loaded": False,
            "lambda_selected_on_validation_only": True,
        },
        "ridge_protocol": {
            "per_timepoint": True,
            "feature_scaler_fit": "fold0 train only",
            "ridge_fit": "fold0 train only",
            "validation": "predict once per timepoint",
            "test": "not loaded",
        },
    }
    return evidence, result


def run_pilot(
    input_root: Path,
    output_dir: Path,
    overwrite: bool = False,
    canonical_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"lambda pilot 输出已存在，默认拒绝覆盖: {output_dir}")
    assets = discover_assets(input_root, canonical_contract)
    evidence, selection = select_lambda(assets)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".lambda-pilot-", dir=output_dir.parent))
    try:
        evidence_path = stage / "candidate_metrics.csv"
        evidence.to_csv(evidence_path, index=False)
        selection["candidate_metrics_sha256"] = file_sha256(evidence_path)
        selection["source_assets"] = [
            {
                "path": str(asset.path),
                "sha256": file_sha256(asset.path),
                "model": asset.model,
                "lambda_ftv": asset.lambda_ftv,
                "patients": len(asset.patient_ids),
                "train_patients": int(np.sum(asset.splits == "train")),
                "validation_patients": int(np.sum(asset.splits == "val")),
            }
            for asset in assets
        ]
        (stage / "lambda_selection.json").write_text(
            json.dumps(selection, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if output_dir.exists():
            backup = output_dir.with_name(f".{output_dir.name}.backup")
            if backup.exists():
                raise FileExistsError(f"lambda pilot backup 已存在: {backup}")
            output_dir.replace(backup)
            try:
                stage.replace(output_dir)
            except Exception:
                backup.replace(output_dir)
                raise
            shutil.rmtree(backup)
        else:
            stage.replace(output_dir)
        return selection
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def _write_synthetic_asset(
    path: Path,
    model: str,
    lambda_ftv: float,
    response: np.ndarray,
    patient_ids: np.ndarray,
    splits: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    val_base_loss: float,
    canonical_contract: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = path.with_name("synthetic_checkpoint.pt")
    transform = path.with_name("synthetic_transform.json")
    checkpoint.write_bytes(b"synthetic checkpoint for lambda-pilot contract\n")
    transform.write_text('{"synthetic":true}\n', encoding="utf-8")
    np.savez_compressed(
        path,
        response=response,
        patient_ids=patient_ids,
        splits=splits,
        ftv_standardized=target,
        ftv_valid=valid,
        model=np.asarray(model),
        lambda_ftv=np.asarray(lambda_ftv),
        fold=np.asarray(0),
        val_base_loss=np.asarray(val_base_loss),
        representation_std=np.asarray(float(np.std(response[splits == "val"], axis=0).mean())),
        checkpoint=np.asarray(str(checkpoint.resolve())),
        checkpoint_sha256=np.asarray(file_sha256(checkpoint)),
        checkpoint_implementation_sha256=np.asarray("c" * 64),
        fold_manifest_sha256=np.asarray(str(canonical_contract["fold_manifest_sha256"])),
        canonical_train_patient_hash=np.asarray(str(canonical_contract["train_patient_hash"])),
        canonical_val_patient_hash=np.asarray(str(canonical_contract["val_patient_hash"])),
        canonical_test_patient_hash=np.asarray(str(canonical_contract["test_patient_hash"])),
        transform_path=np.asarray(str(transform.resolve())),
        transform_sha256=np.asarray(file_sha256(transform)),
    )


def run_self_test() -> dict[str, Any]:
    rng = np.random.default_rng(123)
    n = 80
    patient_ids = np.asarray([f"SYN-{index:03d}" for index in range(n)])
    splits = np.asarray(["train"] * 55 + ["val"] * 25)
    latent = rng.normal(size=(n, 4, 1))
    target = latent[..., 0] + rng.normal(scale=0.08, size=(n, 4))
    valid = np.ones((n, 4), dtype=bool)
    baseline_response = rng.normal(scale=0.5, size=(n, 4, 192))
    canonical_contract = {
        "fold_manifest_sha256": "d" * 64,
        "patient_ids": patient_ids,
        "splits": splits,
        "train_patient_hash": patient_hash(patient_ids[splits == "train"]),
        "val_patient_hash": patient_hash(patient_ids[splits == "val"]),
        "test_patient_hash": patient_hash(["SYN-TEST-ONLY"]),
    }
    with tempfile.TemporaryDirectory(prefix="dgrs-lambda-selftest-") as name:
        root = Path(name)
        input_root = root / "inputs"
        for model in ("G1", "G2"):
            _write_synthetic_asset(
                input_root / f"{model.lower()}_pilot" / "fold_0_train_val_features.npz",
                model,
                0.0,
                baseline_response + rng.normal(scale=0.02, size=baseline_response.shape),
                patient_ids,
                splits,
                target,
                valid,
                1.0,
                canonical_contract,
            )
        for model in ("G3", "G4"):
            for value in LAMBDA_CANDIDATES:
                # 0.02 不足，0.05 起注入稳定可解码 signal，确保最小有效规则可测试。
                strength = 0.0 if value == 0.02 else 0.8
                response = baseline_response.copy()
                response[:, :, 0] += strength * latent[..., 0]
                response += rng.normal(scale=0.015, size=response.shape)
                token = str(value).replace(".", "p")
                _write_synthetic_asset(
                    input_root / f"{model.lower()}_lambda_{token}" / "fold_0_train_val_features.npz",
                    model,
                    value,
                    response,
                    patient_ids,
                    splits,
                    target,
                    valid,
                    1.02,
                    canonical_contract,
                )
        selection = run_pilot(
            input_root, root / "output", canonical_contract=canonical_contract
        )
        if not math.isclose(selection["selected_lambda_ftv"], 0.05):
            raise AssertionError(f"最小有效 lambda 自测失败: {selection['selected_lambda_ftv']}")
        # 泄漏负测：任何 test split 都必须在 Ridge fit 前拒绝。
        bad = root / "bad" / "fold_0_train_val_features.npz"
        bad_splits = splits.copy()
        bad_splits[-1] = "test"
        _write_synthetic_asset(
            bad,
            "G3",
            0.05,
            baseline_response,
            patient_ids,
            bad_splits,
            target,
            valid,
            1.0,
            {
                **canonical_contract,
                "splits": bad_splits,
            },
        )
        rejected_test = False
        try:
            load_pilot_asset(bad)
        except PilotInputError:
            rejected_test = True
        if not rejected_test:
            raise AssertionError("pilot 未拒绝 test split")
        # 更强负测：把 canonical 外（代表 outer-test）的患者伪装成 validation，
        # split 字符串本身仍只有 train/val，也必须在任何 Ridge fit 前拒绝。
        mislabeled = root / "mislabeled" / "fold_0_train_val_features.npz"
        mislabeled_ids = patient_ids.copy()
        mislabeled_ids[-1] = "SYN-TEST-ONLY"
        _write_synthetic_asset(
            mislabeled,
            "G3",
            0.05,
            baseline_response,
            mislabeled_ids,
            splits,
            target,
            valid,
            1.0,
            canonical_contract,
        )
        rejected_mislabeled = False
        try:
            _validate_canonical_fold0([load_pilot_asset(mislabeled)], canonical_contract)
        except PilotInputError:
            rejected_mislabeled = True
        if not rejected_mislabeled:
            raise AssertionError("pilot 未拒绝被误标为 validation 的 test/canonical 外患者")
        return {
            "status": "self-test passed",
            "selected_lambda_ftv": selection["selected_lambda_ftv"],
            "smallest_effective_rule_verified": True,
            "test_split_rejection_verified": True,
            "mislabeled_test_patient_rejection_verified": True,
            "fixed_ridge_alpha": FIXED_RIDGE_ALPHA,
            "temporary_outputs_removed": True,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=EXPERIMENT_ROOT / "metrics" / "lambda_pilot",
        help="包含十个 run 的 fold_0_train_val_features.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXPERIMENT_ROOT / "metrics" / "lambda_selection",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true", help="只在系统临时目录运行合成测试")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = run_self_test() if args.self_test else run_pilot(
            args.input_root.resolve(), args.output_dir.resolve(), args.overwrite
        )
    except (PilotInputError, FileExistsError, ValueError) as exc:
        print(f"lambda pilot 失败：{exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if result.get("status") == "blocked":
        raise SystemExit(3)


if __name__ == "__main__":
    main()

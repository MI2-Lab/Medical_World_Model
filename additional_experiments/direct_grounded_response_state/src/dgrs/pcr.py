"""严格 image-state-only 的 frozen response-state pCR logistic readout。

三个 decision point 分别独立训练；输入恰为 EXPERIMENT_PLAN §12 的 ``r``
组合。StandardScaler 与 class-balanced LogisticRegression 只在 outer-train
拟合，penalty/C 与 Youden threshold 只由 validation 选择，锁定后才对 test
调用一次 ``predict_proba``。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .features import (
    DGRS_ROOT,
    EXPECTED_FOLD_MANIFEST_SHA256,
    MODELS,
    REPO_ROOT,
    RESPONSE_DIM,
    file_sha256,
)
from .probes import FeatureAsset, _load_feature_asset


DECISION_POINTS = ("T0", "T0-T1", "T0-T2")
C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
PENALTIES = ("l1", "l2")
FEATURE_SCHEMAS = {
    "T0": "r0",
    "T0-T1": "concat(r0,r1,r1-r0)",
    "T0-T2": "concat(r0,r1,r2,r1-r0,r2-r1,r2-r0)",
}
FEATURE_DIMS = {"T0": RESPONSE_DIM, "T0-T1": 3 * RESPONSE_DIM, "T0-T2": 6 * RESPONSE_DIM}
LOGISTIC_SOLVER = "liblinear"
LOGISTIC_MAX_ITER = 20_000
LOGISTIC_TOL = 1e-7


PREDICTION_COLUMNS = (
    "patient_id",
    "fold",
    "split",
    "model",
    "decision_point",
    "y_true",
    "probability",
    "predicted_label",
    "threshold",
    "penalty",
    "C",
    "readout",
    "class_weight",
    "feature_schema",
    "feature_schema_sha256",
    "feature_dim",
    "val_auroc",
    "val_auprc",
    "val_youden",
    "source_feature_file",
    "source_feature_sha256",
    "feature_extractor_sha256",
    "source_checkpoint",
    "source_checkpoint_sha256",
    "fold_manifest_sha256",
    "canonical_patient_order_sha256",
    "canonical_patient_label_sha256",
    "test_used_for_checkpoint_selection",
    "test_used_for_lambda_selection",
    "test_used_for_scaler",
    "test_used_for_hyperparameter_selection",
    "test_used_for_threshold_selection",
    "test_feature_matrix_constructed_after_selection_lock",
    "test_prediction_guard_enforced",
    "test_predict_proba_call_count",
)


SELECTION_COLUMNS = (
    "fold",
    "model",
    "decision_point",
    "feature_schema",
    "feature_schema_sha256",
    "feature_dim",
    "n_train",
    "n_val",
    "n_test",
    "train_positive",
    "val_positive",
    "test_positive",
    "selected_penalty",
    "selected_C",
    "val_auroc",
    "val_auprc",
    "selected_threshold",
    "val_youden",
    "val_sensitivity",
    "val_specificity",
    "grid_validation_metrics_json",
    "selection_rule",
    "threshold_tie_rule",
    "solver",
    "class_weight",
    "max_iter",
    "tol",
    "random_state",
    "logistic_intercept_json",
    "logistic_coef_json",
    "feature_scaler_mean_json",
    "feature_scaler_scale_json",
    "feature_scaler_n_samples_seen",
    "source_feature_file",
    "source_feature_sha256",
    "feature_extractor_sha256",
    "source_checkpoint",
    "source_checkpoint_sha256",
    "fold_manifest_sha256",
    "canonical_patient_order_sha256",
    "canonical_patient_label_sha256",
    "sklearn_version",
    "test_used_for_checkpoint_selection",
    "test_used_for_lambda_selection",
    "test_used_for_scaler",
    "test_used_for_hyperparameter_selection",
    "test_used_for_threshold_selection",
    "test_feature_matrix_constructed_after_selection_lock",
    "test_prediction_guard_enforced",
    "test_predict_proba_call_count",
)


@dataclass(frozen=True)
class SelectedLogistic:
    scaler: StandardScaler
    model: LogisticRegression
    penalty: str
    c_value: float
    validation_auroc: float
    validation_auprc: float
    threshold: float
    youden: float
    sensitivity: float
    specificity: float
    grid: tuple[dict[str, Any], ...]


@dataclass
class _LogisticTestPredictGuard:
    """把 outer-test ``predict_proba`` 约束为单次可消费操作。"""

    call_count: int = 0

    def predict_positive_probability(
        self, model: LogisticRegression, matrix: np.ndarray
    ) -> np.ndarray:
        if self.call_count != 0:
            raise RuntimeError("outer-test predict_proba 已调用；拒绝第二次调用")
        self.call_count += 1
        return np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float64)


def _source_sha256(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(value).resolve() for value in paths):
        label = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
        digest.update(str(label).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def pcr_implementation_sha256() -> str:
    paths = [
        Path(__file__),
        Path(__file__).with_name("features.py"),
        Path(__file__).with_name("probes.py"),
    ]
    script = DGRS_ROOT / "scripts" / "run_pcr_readouts.py"
    if script.is_file():
        paths.append(script)
    return _source_sha256(paths)


def _temporary_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(handle)
    return Path(name)


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = _temporary_path(path)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = _temporary_path(path)
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _refuse_existing(paths: Sequence[Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("输出已存在，默认拒绝覆盖：" + ", ".join(existing))


def response_readout_matrix(response_state: np.ndarray, decision_point: str) -> np.ndarray:
    """构造唯一允许的 pCR readout feature；不接收任何表格输入。"""

    response = np.asarray(response_state, dtype=np.float64)
    if response.ndim != 3 or response.shape[1:] != (4, RESPONSE_DIM):
        raise ValueError(f"response_state shape 非法：{response.shape}")
    if not np.isfinite(response).all():
        raise FloatingPointError("response_state 含 NaN/Inf")
    if decision_point == "T0":
        matrix = response[:, 0]
    elif decision_point == "T0-T1":
        r0, r1 = response[:, 0], response[:, 1]
        matrix = np.concatenate((r0, r1, r1 - r0), axis=1)
    elif decision_point == "T0-T2":
        r0, r1, r2 = response[:, 0], response[:, 1], response[:, 2]
        matrix = np.concatenate(
            (r0, r1, r2, r1 - r0, r2 - r1, r2 - r0), axis=1
        )
    else:
        raise ValueError(f"未知 decision point：{decision_point}")
    if matrix.shape != (len(response), FEATURE_DIMS[decision_point]):
        raise AssertionError("pCR feature dimension contract 失效")
    return np.ascontiguousarray(matrix, dtype=np.float64)


def _youden_threshold(
    y_true: np.ndarray, probabilities: np.ndarray
) -> tuple[float, float, float, float]:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if y_true.shape != probabilities.shape or set(y_true.tolist()) != {0, 1}:
        raise ValueError("Youden validation label/probability 非法")
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0) | (probabilities > 1)
    ):
        raise ValueError("validation probability 不在 [0,1]")
    candidates = np.unique(np.concatenate(([0.0, 1.0], probabilities)))
    rows: list[tuple[float, float, float, float]] = []
    positives = int(np.count_nonzero(y_true == 1))
    negatives = int(np.count_nonzero(y_true == 0))
    for threshold in candidates:
        prediction = probabilities >= threshold
        sensitivity = float(np.count_nonzero(prediction & (y_true == 1)) / positives)
        specificity = float(np.count_nonzero(~prediction & (y_true == 0)) / negatives)
        youden = sensitivity + specificity - 1.0
        rows.append((float(threshold), youden, sensitivity, specificity))
    best_youden = max(row[1] for row in rows)
    eligible = [row for row in rows if row[1] >= best_youden - 1e-12]
    # 等 Youden 时，先选最接近 0.5，再选较小 threshold。
    return min(eligible, key=lambda row: (abs(row[0] - 0.5), row[0]))


def select_logistic_readout(
    train_matrix: np.ndarray,
    train_labels: np.ndarray,
    validation_matrix: np.ndarray,
    validation_labels: np.ndarray,
    *,
    penalties: Sequence[str] = PENALTIES,
    c_grid: Sequence[float] = C_GRID,
    random_state: int,
) -> SelectedLogistic:
    """不接受 test 参数，从接口上阻断 test 参与超参数/threshold 选择。"""

    train_matrix = np.asarray(train_matrix, dtype=np.float64)
    validation_matrix = np.asarray(validation_matrix, dtype=np.float64)
    train_labels = np.asarray(train_labels, dtype=np.int64).reshape(-1)
    validation_labels = np.asarray(validation_labels, dtype=np.int64).reshape(-1)
    if train_matrix.ndim != 2 or validation_matrix.ndim != 2:
        raise ValueError("logistic feature 必须为二维")
    if train_matrix.shape[1] != validation_matrix.shape[1]:
        raise ValueError("train/validation feature_dim 不一致")
    if len(train_matrix) != len(train_labels) or len(validation_matrix) != len(
        validation_labels
    ):
        raise ValueError("logistic X/y 行数不一致")
    if set(train_labels.tolist()) != {0, 1} or set(validation_labels.tolist()) != {
        0,
        1,
    }:
        raise ValueError("train/validation 必须均含两个 pCR class")
    if not np.isfinite(train_matrix).all() or not np.isfinite(validation_matrix).all():
        raise FloatingPointError("logistic feature 含 NaN/Inf")
    penalties = tuple(dict.fromkeys(str(value) for value in penalties))
    if not penalties or not set(penalties).issubset(PENALTIES):
        raise ValueError(f"penalties 必须为 {PENALTIES} 的非空子集")
    c_grid = tuple(sorted(set(float(value) for value in c_grid)))
    if not c_grid or any(value <= 0 or not math.isfinite(value) for value in c_grid):
        raise ValueError("C grid 必须为有限正数")
    scaler = StandardScaler().fit(train_matrix)
    train_scaled = scaler.transform(train_matrix)
    validation_scaled = scaler.transform(validation_matrix)
    candidates: list[tuple[float, float, float, int, str, LogisticRegression, np.ndarray]] = []
    grid_rows: list[dict[str, Any]] = []
    penalty_order = {value: index for index, value in enumerate(PENALTIES)}
    for penalty in penalties:
        for c_value in c_grid:
            model = LogisticRegression(
                l1_ratio=1.0 if penalty == "l1" else 0.0,
                C=c_value,
                solver=LOGISTIC_SOLVER,
                class_weight="balanced",
                max_iter=LOGISTIC_MAX_ITER,
                tol=LOGISTIC_TOL,
                random_state=int(random_state),
            )
            model.fit(train_scaled, train_labels)
            probability = model.predict_proba(validation_scaled)[:, 1]
            auroc = float(roc_auc_score(validation_labels, probability))
            auprc = float(average_precision_score(validation_labels, probability))
            if not math.isfinite(auroc) or not math.isfinite(auprc):
                raise FloatingPointError("validation AUROC/AUPRC 非有限")
            grid_rows.append(
                {
                    "penalty": penalty,
                    "C": c_value,
                    "val_auroc": auroc,
                    "val_auprc": auprc,
                    "n_iter": int(np.max(model.n_iter_)),
                }
            )
            candidates.append(
                (
                    auroc,
                    auprc,
                    c_value,
                    penalty_order[penalty],
                    penalty,
                    model,
                    probability,
                )
            )
    best_auroc = max(item[0] for item in candidates)
    auroc_tied = [item for item in candidates if item[0] >= best_auroc - 1e-12]
    best_auprc = max(item[1] for item in auroc_tied)
    metric_tied = [item for item in auroc_tied if item[1] >= best_auprc - 1e-12]
    # 继续平手时优先更小 C，再按预注册 penalty 顺序 l1、l2。
    chosen = min(metric_tied, key=lambda item: (item[2], item[3]))
    auroc, auprc, c_value, _, penalty, model, probability = chosen
    threshold, youden, sensitivity, specificity = _youden_threshold(
        validation_labels, probability
    )
    return SelectedLogistic(
        scaler=scaler,
        model=model,
        penalty=penalty,
        c_value=float(c_value),
        validation_auroc=float(auroc),
        validation_auprc=float(auprc),
        threshold=float(threshold),
        youden=float(youden),
        sensitivity=float(sensitivity),
        specificity=float(specificity),
        grid=tuple(grid_rows),
    )


def _json_vector(values: np.ndarray) -> str:
    return json.dumps(
        [float(value) for value in np.asarray(values, dtype=np.float64).reshape(-1)],
        separators=(",", ":"),
    )


def _labels_from_asset(asset: FeatureAsset) -> np.ndarray:
    labels = np.asarray(asset.label_pcr)
    if labels.shape != (808,) or labels.dtype.kind not in {"i", "u"}:
        raise ValueError("feature asset pCR label dtype/shape contract 非法")
    if not set(labels.tolist()).issubset({0, 1}):
        raise ValueError("feature asset pCR label contract 非法")
    if asset.metadata.get("canonical_label_rows_verified") is not True:
        raise ValueError("feature asset 缺 canonical pCR label 逐行闭环证据")
    return labels.astype(np.int64, copy=True)


def _run_decision_point(
    *,
    asset: FeatureAsset,
    labels: np.ndarray,
    model_name: str,
    decision_point: str,
    penalties: Sequence[str],
    c_grid: Sequence[float],
    random_state: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split_indices = {
        split: np.flatnonzero(asset.splits == split)
        for split in ("train", "val", "test")
    }
    train_indices = split_indices["train"]
    validation_indices = split_indices["val"]
    train_matrix = response_readout_matrix(
        asset.response_state[train_indices], decision_point
    )
    validation_matrix = response_readout_matrix(
        asset.response_state[validation_indices], decision_point
    )
    selected = select_logistic_readout(
        train_matrix,
        labels[train_indices],
        validation_matrix,
        labels[validation_indices],
        penalties=penalties,
        c_grid=c_grid,
        random_state=random_state,
    )
    # 只有 penalty/C/threshold 锁定以后才构造 test matrix，并单次消费 predict_proba。
    test_indices = split_indices["test"]
    test_matrix = response_readout_matrix(
        asset.response_state[test_indices], decision_point
    )
    test_scaled = selected.scaler.transform(test_matrix)
    test_guard = _LogisticTestPredictGuard()
    probability = test_guard.predict_positive_probability(selected.model, test_scaled)
    if test_guard.call_count != 1:
        raise RuntimeError("outer-test predict_proba call count 非 1")
    if probability.shape != (len(test_indices),) or not np.isfinite(probability).all():
        raise FloatingPointError("test pCR probability 非法")
    predicted_label = (probability >= selected.threshold).astype(np.int64)
    schema = FEATURE_SCHEMAS[decision_point]
    schema_sha = hashlib.sha256(schema.encode("utf-8")).hexdigest()
    common = {
        "fold": int(asset.metadata["fold"]),
        "model": model_name,
        "decision_point": decision_point,
        "threshold": selected.threshold,
        "penalty": selected.penalty,
        "C": selected.c_value,
        "readout": "class-balanced LogisticRegression",
        "class_weight": "balanced",
        "feature_schema": schema,
        "feature_schema_sha256": schema_sha,
        "feature_dim": FEATURE_DIMS[decision_point],
        "val_auroc": selected.validation_auroc,
        "val_auprc": selected.validation_auprc,
        "val_youden": selected.youden,
        "source_feature_file": str(asset.path),
        "source_feature_sha256": asset.sha256,
        "feature_extractor_sha256": asset.metadata["extractor_sha256"],
        "source_checkpoint": asset.metadata["checkpoint"],
        "source_checkpoint_sha256": asset.metadata["checkpoint_sha256"],
        "fold_manifest_sha256": asset.metadata["fold_manifest_sha256"],
        "canonical_patient_order_sha256": asset.metadata[
            "canonical_patient_order_sha256"
        ],
        "canonical_patient_label_sha256": asset.metadata[
            "canonical_patient_label_sha256"
        ],
    }
    rows = []
    for output_index, patient_index in enumerate(test_indices):
        rows.append(
            {
                "patient_id": str(asset.patient_ids[patient_index]),
                **common,
                "split": "test",
                "y_true": int(labels[patient_index]),
                "probability": float(probability[output_index]),
                "predicted_label": int(predicted_label[output_index]),
                "test_used_for_checkpoint_selection": False,
                "test_used_for_lambda_selection": False,
                "test_used_for_scaler": False,
                "test_used_for_hyperparameter_selection": False,
                "test_used_for_threshold_selection": False,
                "test_feature_matrix_constructed_after_selection_lock": True,
                "test_prediction_guard_enforced": True,
                "test_predict_proba_call_count": test_guard.call_count,
            }
        )
    selection = {
        "fold": int(asset.metadata["fold"]),
        "model": model_name,
        "decision_point": decision_point,
        "feature_schema": schema,
        "feature_schema_sha256": schema_sha,
        "feature_dim": FEATURE_DIMS[decision_point],
        "n_train": len(train_indices),
        "n_val": len(validation_indices),
        "n_test": len(test_indices),
        "train_positive": int(labels[train_indices].sum()),
        "val_positive": int(labels[validation_indices].sum()),
        "test_positive": int(labels[test_indices].sum()),
        "selected_penalty": selected.penalty,
        "selected_C": selected.c_value,
        "val_auroc": selected.validation_auroc,
        "val_auprc": selected.validation_auprc,
        "selected_threshold": selected.threshold,
        "val_youden": selected.youden,
        "val_sensitivity": selected.sensitivity,
        "val_specificity": selected.specificity,
        "grid_validation_metrics_json": json.dumps(
            selected.grid, separators=(",", ":")
        ),
        "selection_rule": (
            "max validation AUROC; <=1e-12 tie max AUPRC; then smaller C; "
            "then penalty order l1,l2"
        ),
        "threshold_tie_rule": (
            "max validation Youden J; <=1e-12 tie closest to 0.5; then smaller threshold"
        ),
        "solver": LOGISTIC_SOLVER,
        "class_weight": "balanced",
        "max_iter": LOGISTIC_MAX_ITER,
        "tol": LOGISTIC_TOL,
        "random_state": int(random_state),
        "logistic_intercept_json": _json_vector(selected.model.intercept_),
        "logistic_coef_json": _json_vector(selected.model.coef_),
        "feature_scaler_mean_json": _json_vector(selected.scaler.mean_),
        "feature_scaler_scale_json": _json_vector(selected.scaler.scale_),
        "feature_scaler_n_samples_seen": int(selected.scaler.n_samples_seen_),
        "sklearn_version": sklearn.__version__,
        "source_feature_file": str(asset.path),
        "source_feature_sha256": asset.sha256,
        "feature_extractor_sha256": asset.metadata["extractor_sha256"],
        "source_checkpoint": asset.metadata["checkpoint"],
        "source_checkpoint_sha256": asset.metadata["checkpoint_sha256"],
        "fold_manifest_sha256": asset.metadata["fold_manifest_sha256"],
        "canonical_patient_order_sha256": asset.metadata[
            "canonical_patient_order_sha256"
        ],
        "canonical_patient_label_sha256": asset.metadata[
            "canonical_patient_label_sha256"
        ],
        "test_used_for_checkpoint_selection": False,
        "test_used_for_lambda_selection": False,
        "test_used_for_scaler": False,
        "test_used_for_hyperparameter_selection": False,
        "test_used_for_threshold_selection": False,
        "test_feature_matrix_constructed_after_selection_lock": True,
        "test_prediction_guard_enforced": True,
        "test_predict_proba_call_count": test_guard.call_count,
    }
    return rows, selection


def run_pcr_readouts(
    *,
    model_name: str,
    fold: int,
    feature_root: Path = DGRS_ROOT / "features",
    prediction_root: Path = DGRS_ROOT / "predictions" / "pcr_readouts",
    metric_root: Path = DGRS_ROOT / "metrics" / "pcr_readouts",
    penalties: Sequence[str] = PENALTIES,
    c_grid: Sequence[float] = C_GRID,
    seed: int = 2026,
    overwrite: bool = False,
) -> dict[str, Any]:
    """运行一个 model×fold 的三个独立 pCR readout。"""

    model_name = str(model_name).upper()
    if model_name not in MODELS or fold not in range(5):
        raise ValueError("model/fold 非法")
    prediction_dir = Path(prediction_root) / model_name / f"fold_{fold}"
    metric_dir = Path(metric_root) / model_name / f"fold_{fold}"
    prediction_path = prediction_dir / "test_predictions.csv"
    selection_path = metric_dir / "selection_records.csv"
    summary_path = metric_dir / "summary.json"
    _refuse_existing([prediction_path, selection_path, summary_path], overwrite)
    asset = _load_feature_asset(feature_root, model_name, fold)
    labels = _labels_from_asset(asset)
    prediction_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for decision_index, decision_point in enumerate(DECISION_POINTS):
        rows, selection = _run_decision_point(
            asset=asset,
            labels=labels,
            model_name=model_name,
            decision_point=decision_point,
            penalties=penalties,
            c_grid=c_grid,
            random_state=int(seed + fold * 100 + decision_index),
        )
        prediction_rows.extend(rows)
        selection_rows.append(selection)
    predictions = pd.DataFrame(prediction_rows, columns=PREDICTION_COLUMNS)
    selections = pd.DataFrame(selection_rows, columns=SELECTION_COLUMNS)
    prediction_key = ["patient_id", "fold", "model", "decision_point"]
    selection_key = prediction_key[1:]
    if predictions.empty or predictions.duplicated(prediction_key).any():
        raise ValueError("pCR predictions 为空或 key 重复")
    if selections.empty or selections.duplicated(selection_key).any():
        raise ValueError("pCR selections 为空或 key 重复")
    expected_test = int(np.count_nonzero(asset.splits == "test"))
    if len(predictions) != expected_test * len(DECISION_POINTS):
        raise ValueError("pCR test prediction coverage 错误")
    if set(predictions["decision_point"]) != set(DECISION_POINTS):
        raise ValueError("pCR decision point 不完整")
    if set(predictions["split"]) != {"test"}:
        raise ValueError("pCR predictions 只能含 test")
    if not np.isfinite(
        predictions[["probability", "threshold", "C"]].to_numpy(dtype=np.float64)
    ).all():
        raise FloatingPointError("pCR prediction 核心数值含 NaN/Inf")
    if not predictions["probability"].between(0, 1).all():
        raise ValueError("pCR probability 超出 [0,1]")
    selection_flags = (
        "test_used_for_checkpoint_selection",
        "test_used_for_lambda_selection",
        "test_used_for_scaler",
        "test_used_for_hyperparameter_selection",
        "test_used_for_threshold_selection",
    )
    if predictions[list(selection_flags)].to_numpy(dtype=bool).any() or selections[
        list(selection_flags)
    ].to_numpy(dtype=bool).any():
        raise ValueError("pCR 输出声称 test 参与了选择/拟合")
    for frame in (predictions, selections):
        if not frame["test_feature_matrix_constructed_after_selection_lock"].eq(
            True
        ).all():
            raise ValueError("pCR test matrix 在 selection lock 前构造")
        if not frame["test_prediction_guard_enforced"].eq(True).all():
            raise ValueError("pCR test single-use guard 未执行")
        if not frame["test_predict_proba_call_count"].eq(1).all():
            raise ValueError("pCR test predict_proba call count 非 1")
    _atomic_csv(prediction_path, predictions)
    _atomic_csv(selection_path, selections)
    summary = {
        "schema_version": 1,
        "status": "strict image-state-only pCR readouts complete",
        "model": model_name,
        "fold": fold,
        "decision_points": list(DECISION_POINTS),
        "feature_schemas": FEATURE_SCHEMAS,
        "feature_dims": FEATURE_DIMS,
        "test_prediction_rows": len(predictions),
        "test_patient_count": expected_test,
        "prediction_columns": list(PREDICTION_COLUMNS),
        "selection_columns": list(SELECTION_COLUMNS),
        "prediction_file": str(prediction_path.resolve()),
        "prediction_file_sha256": file_sha256(prediction_path),
        "selection_file": str(selection_path.resolve()),
        "selection_file_sha256": file_sha256(selection_path),
        "source_feature_file": str(asset.path),
        "source_feature_sha256": asset.sha256,
        "source_checkpoint": asset.metadata["checkpoint"],
        "source_checkpoint_sha256": asset.metadata["checkpoint_sha256"],
        "fold_manifest_sha256": EXPECTED_FOLD_MANIFEST_SHA256,
        "canonical_patient_order_sha256": asset.metadata[
            "canonical_patient_order_sha256"
        ],
        "canonical_patient_label_sha256": asset.metadata[
            "canonical_patient_label_sha256"
        ],
        "pcr_implementation_sha256": pcr_implementation_sha256(),
        "sklearn_version": sklearn.__version__,
        "logistic": {
            "penalties": list(penalties),
            "C_grid": [float(value) for value in c_grid],
            "solver": LOGISTIC_SOLVER,
            "class_weight": "balanced",
            "scaler": "StandardScaler fit on outer fold train only",
            "hyperparameter_selection": "outer fold validation AUROC/AUPRC only",
            "threshold_selection": "outer fold validation Youden J only",
        },
        "leakage_guards": {
            "feature_inputs": "frozen observed response_state only",
            "clinical_used": False,
            "treatment_used": False,
            "radiomics_used": False,
            "mask_geometry_used": False,
            "ground_truth_ftv_used": False,
            "predicted_ftv_or_ftv_head_used": False,
            "feature_scaler_fit_scope": "outer fold train only",
            "logistic_fit_scope": "outer fold train only",
            "penalty_C_selection_scope": "outer fold validation only",
            "threshold_selection_scope": "outer fold validation only",
            "test_feature_matrix_constructed_after_selection_lock": True,
            "test_predict_proba_calls_per_decision": 1,
            "test_prediction_guard_enforced": True,
            "test_used_for_checkpoint_selection": False,
            "test_used_for_lambda_selection": False,
            "test_used_for_any_selection": False,
            "world_model_trained_or_finetuned": False,
        },
    }
    _atomic_json(summary_path, summary)
    return summary


def synthetic_self_test() -> dict[str, Any]:
    generator = np.random.default_rng(20260807)
    response = generator.normal(size=(117, 4, RESPONSE_DIM))
    shapes = {
        decision: list(response_readout_matrix(response, decision).shape)
        for decision in DECISION_POINTS
    }
    train = response_readout_matrix(response[:64], "T0")
    validation = response_readout_matrix(response[64:94], "T0") + 0.4
    test = response_readout_matrix(response[94:], "T0")
    train_labels = np.asarray([0, 1] * 32, dtype=np.int64)
    validation_labels = np.asarray([0, 1] * 15, dtype=np.int64)
    selected = select_logistic_readout(
        train,
        train_labels,
        validation,
        validation_labels,
        penalties=("l1", "l2"),
        c_grid=(0.01, 0.1, 1.0),
        random_state=2026,
    )
    if not np.allclose(selected.scaler.mean_, train.mean(axis=0)):
        raise AssertionError("pCR StandardScaler 不是 train-only")
    if np.allclose(selected.scaler.mean_, validation.mean(axis=0)):
        raise AssertionError("synthetic train/validation 均值意外相等")
    guard = _LogisticTestPredictGuard()
    probability = guard.predict_positive_probability(
        selected.model, selected.scaler.transform(test)
    )
    if probability.shape != (23,) or not np.isfinite(probability).all():
        raise AssertionError("synthetic pCR test prediction 非法")
    try:
        guard.predict_positive_probability(selected.model, selected.scaler.transform(test))
    except RuntimeError:
        second_test_predict_rejected = True
    else:
        raise AssertionError("pCR outer-test 第二次 predict_proba 未被拒绝")
    return {
        "status": "synthetic pCR readout self-test passed",
        "feature_shapes": shapes,
        "strict_response_state_only": True,
        "train_only_scaler_verified": True,
        "validation_only_selection_enforced_by_signature": True,
        "test_matrix_constructed_after_selection_lock": True,
        "test_predict_single_use_guard_verified": second_test_predict_rejected,
        "test_predict_proba_call_count": guard.call_count,
        "sklearn_version": sklearn.__version__,
        "selected_penalty": selected.penalty,
        "selected_C": selected.c_value,
        "threshold": selected.threshold,
    }

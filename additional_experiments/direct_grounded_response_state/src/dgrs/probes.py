"""Direct Grounded Response State 的 frozen representation Ridge probes。

每个 outer fold/cell 都只在 train 拟合 feature ``StandardScaler`` 与单输出
``Ridge``；alpha 仅按 validation standardized MSE 选择，锁定后才构造 test
矩阵并恰好调用一次 ``predict``。本模块只评估 observed response state：
static 使用 ``r_t``，longitudinal 使用 ``r_(t+1)-r_t``。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from .features import (
    DGRS_ROOT,
    DEFAULT_FOLD_MANIFEST,
    EXPECTED_FOLD_MANIFEST_SHA256,
    MODELS,
    REPO_ROOT,
    RESPONSE_DIM,
    TIMEPOINTS,
    file_sha256,
    patient_hash,
    validate_feature_against_canonical,
    validate_feature_arrays,
)


TARGETS = ("ftv", "ld", "sphericity")
RAW_FEATURE_ORDER = ("ftv", "sphericity", "ld", "bpe")
TRANSITIONS = ("T0→T1", "T1→T2", "T2→T3")
ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
RIDGE_SOLVER = "lsqr"
RIDGE_TOL = 1e-8
RIDGE_MAX_ITER = 10_000
OSRA_CONFIG_ROOT = (
    REPO_ROOT
    / "additional_experiments"
    / "observed_state_radiomics_audit"
    / "configs"
)
RNC_ROOT = REPO_ROOT / "additional_experiments" / "radiomics_next_change"
RAW_TARGET_PATH = RNC_ROOT / "data_audit" / "radiomics_transition_targets_raw.csv"
CHANGE_CONFIG_ROOT = RNC_ROOT / "configs"


PREDICTION_COLUMNS = (
    "patient_id",
    "fold",
    "split",
    "model",
    "task",
    "timepoint",
    "transition",
    "representation",
    "input_variant",
    "target",
    "feature_dim",
    "y_true",
    "y_pred",
    "y_true_standardized",
    "y_pred_standardized",
    "b0_prediction",
    "b0_prediction_standardized",
    "selected_alpha",
    "target_transform",
    "source_feature_file",
    "source_feature_sha256",
    "feature_extractor_sha256",
    "source_checkpoint",
    "source_checkpoint_sha256",
    "fold_manifest_sha256",
    "canonical_patient_order_sha256",
    "canonical_patient_label_sha256",
    "static_transform_sha256",
    "change_transform_sha256",
    "raw_targets_sha256",
    "test_used_for_target_transform",
    "test_used_for_checkpoint_selection",
    "test_used_for_lambda_selection",
    "test_used_for_scaler",
    "test_used_for_alpha_selection",
    "test_prediction_guard_enforced",
    "test_predict_call_count",
)


SELECTION_COLUMNS = (
    "fold",
    "model",
    "task",
    "timepoint",
    "transition",
    "representation",
    "input_variant",
    "target",
    "feature_dim",
    "n_train",
    "n_val",
    "n_test",
    "selected_alpha",
    "val_mse_standardized",
    "alpha_validation_mse_json",
    "ridge_solver",
    "ridge_tol",
    "ridge_max_iter",
    "ridge_fit_intercept",
    "ridge_intercept",
    "ridge_coef_json",
    "feature_scaler_mean_json",
    "feature_scaler_scale_json",
    "feature_scaler_n_samples_seen",
    "train_target_mean_standardized",
    "b0_val_mse_standardized",
    "target_transform",
    "source_feature_file",
    "source_feature_sha256",
    "feature_extractor_sha256",
    "source_checkpoint",
    "source_checkpoint_sha256",
    "fold_manifest_sha256",
    "canonical_patient_order_sha256",
    "canonical_patient_label_sha256",
    "static_transform_file",
    "static_transform_sha256",
    "change_transform_file",
    "change_transform_sha256",
    "raw_target_file",
    "raw_target_file_sha256",
    "raw_targets_sha256",
    "sklearn_version",
    "test_used_for_target_transform",
    "test_used_for_checkpoint_selection",
    "test_used_for_lambda_selection",
    "test_used_for_scaler",
    "test_used_for_alpha_selection",
    "test_prediction_guard_enforced",
    "test_predict_call_count",
)


@dataclass(frozen=True)
class FeatureAsset:
    path: Path
    sha256: str
    metadata_path: Path
    metadata: Mapping[str, Any]
    patient_ids: np.ndarray
    splits: np.ndarray
    label_pcr: np.ndarray
    response_state: np.ndarray
    patient_to_index: Mapping[str, int]


@dataclass(frozen=True)
class TargetAssets:
    raw: Mapping[str, np.ndarray]
    raw_targets_sha256: str
    raw_file_sha256: str
    static_payload: Mapping[str, Any]
    static_path: Path
    static_sha256: str
    change_payload: Mapping[str, Any]
    change_path: Path
    change_sha256: str


@dataclass(frozen=True)
class Cell:
    task: str
    target: str
    index: int

    @property
    def timepoint(self) -> str:
        return TIMEPOINTS[self.index] if self.task == "static" else ""

    @property
    def transition(self) -> str:
        return TRANSITIONS[self.index] if self.task == "change" else ""

    @property
    def input_variant(self) -> str:
        return "current" if self.task == "static" else "observed_difference"


@dataclass(frozen=True)
class PreparedSplit:
    patient_ids: tuple[str, ...]
    matrix: np.ndarray
    y_standardized: np.ndarray
    y_natural: np.ndarray


@dataclass(frozen=True)
class SelectedRidge:
    scaler: StandardScaler
    model: Ridge
    alpha: float
    validation_mse: float
    grid: tuple[tuple[float, float], ...]


@dataclass
class _RidgeTestPredictGuard:
    """把 outer-test ``predict`` 约束为单次可消费操作。"""

    call_count: int = 0

    def predict(self, model: Ridge, matrix: np.ndarray) -> np.ndarray:
        if self.call_count != 0:
            raise RuntimeError("outer-test Ridge predict 已调用；拒绝第二次调用")
        self.call_count += 1
        return np.asarray(model.predict(matrix), dtype=np.float64).reshape(-1)


def _source_sha256(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(value).resolve() for value in paths):
        label = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
        digest.update(str(label).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def probe_implementation_sha256() -> str:
    paths = [Path(__file__), Path(__file__).with_name("features.py")]
    script = DGRS_ROOT / "scripts" / "run_representation_probes.py"
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


def _load_feature_asset(feature_root: Path, model_name: str, fold: int) -> FeatureAsset:
    model_name = str(model_name).upper()
    if model_name not in MODELS or fold not in range(5):
        raise ValueError("model/fold 非法")
    directory = Path(feature_root) / model_name / f"fold_{fold}"
    path = directory / "observed_features.npz"
    metadata_path = directory / "extraction_metadata.json"
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"缺 frozen feature asset：{directory}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_hash = file_sha256(path)
    if metadata.get("feature_file_sha256") != feature_hash:
        raise ValueError("feature SHA 与 extraction metadata 不一致")
    if metadata.get("max_patients_per_split") is not None:
        raise ValueError("正式 probe 拒绝 smoke/partial feature")
    if str(metadata.get("model", "")).upper() != model_name or int(
        metadata.get("fold", -1)
    ) != fold:
        raise ValueError("feature metadata model/fold 错位")
    if metadata.get("fold_manifest_sha256") != EXPECTED_FOLD_MANIFEST_SHA256:
        raise ValueError("feature fold manifest SHA 漂移")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    validate_feature_arrays(arrays, expected_n=808)
    if str(np.asarray(arrays["model"]).reshape(()).item()).upper() != model_name:
        raise ValueError("feature NPZ model 错位")
    if int(np.asarray(arrays["fold"]).reshape(()).item()) != fold:
        raise ValueError("feature NPZ fold 错位")
    canonical_evidence = validate_feature_against_canonical(
        arrays,
        fold_manifest=DEFAULT_FOLD_MANIFEST,
        fold=fold,
        max_patients_per_split=None,
    )
    for key, expected in canonical_evidence.items():
        if metadata.get(key) != expected:
            raise ValueError(f"feature metadata canonical evidence 漂移：{key}")
    patient_ids = arrays["patient_ids"].astype(str)
    splits = arrays["splits"].astype(str)
    return FeatureAsset(
        path=path.resolve(),
        sha256=feature_hash,
        metadata_path=metadata_path.resolve(),
        metadata=metadata,
        patient_ids=patient_ids,
        splits=splits,
        label_pcr=arrays["label_pcr"].astype(np.int64, copy=True),
        response_state=arrays["response_state"].astype(np.float64),
        patient_to_index={value: index for index, value in enumerate(patient_ids)},
    )


def _load_raw_targets(path: Path) -> dict[str, np.ndarray]:
    frame = pd.read_csv(path)
    required = {"patient_id", "transition", "start_visit", "end_visit"}
    for feature in RAW_FEATURE_ORDER:
        required.update(
            {f"{feature}_start", f"{feature}_end", f"{feature}_valid"}
        )
    if missing := required.difference(frame.columns):
        raise ValueError(f"raw target CSV 缺列：{sorted(missing)}")
    output: dict[str, np.ndarray] = {}
    for patient_id, group in frame.groupby("patient_id", sort=False):
        if group["transition"].duplicated().any():
            raise ValueError(f"{patient_id} transition 重复")
        group = group.set_index("transition").reindex(TRANSITIONS)
        if group.isna().all(axis=1).any():
            raise ValueError(f"{patient_id} transition 不完整")
        values = np.full((3, len(RAW_FEATURE_ORDER), 3), np.nan, dtype=np.float64)
        for transition_index, transition in enumerate(TRANSITIONS):
            row = group.loc[transition]
            expected_start, expected_end = transition.split("→")
            if str(row["start_visit"]) != expected_start or str(row["end_visit"]) != expected_end:
                raise ValueError(f"{patient_id}/{transition} endpoint 错位")
            for feature_index, feature in enumerate(RAW_FEATURE_ORDER):
                values[transition_index, feature_index] = (
                    float(row[f"{feature}_start"]),
                    float(row[f"{feature}_end"]),
                    float(bool(row[f"{feature}_valid"])),
                )
        output[str(patient_id)] = values
    return output


def _raw_targets_hash(raw: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for patient_id in sorted(raw):
        digest.update(patient_id.encode("utf-8"))
        digest.update(np.asarray(raw[patient_id], dtype="<f8").tobytes())
    return digest.hexdigest()


def _load_target_assets(asset: FeatureAsset, fold: int) -> TargetAssets:
    raw_path = RAW_TARGET_PATH.resolve(strict=True)
    static_path = (OSRA_CONFIG_ROOT / f"static_target_transform_fold_{fold}.json").resolve(
        strict=True
    )
    change_path = (CHANGE_CONFIG_ROOT / f"radiomics_transform_fold_{fold}.json").resolve(
        strict=True
    )
    raw = _load_raw_targets(raw_path)
    raw_hash = _raw_targets_hash(raw)
    static_payload = json.loads(static_path.read_text(encoding="utf-8"))
    change_payload = json.loads(change_path.read_text(encoding="utf-8"))
    if int(static_payload.get("fold", -1)) != fold or int(
        change_payload.get("fold", -1)
    ) != fold:
        raise ValueError("target transform fold 错位")
    if tuple(static_payload.get("feature_order", ())) != RAW_FEATURE_ORDER:
        raise ValueError("static transform feature order 漂移")
    if tuple(change_payload.get("feature_order", ())) != RAW_FEATURE_ORDER:
        raise ValueError("change transform feature order 漂移")
    if static_payload.get("raw_targets_sha256") != raw_hash or change_payload.get(
        "raw_targets_sha256"
    ) != raw_hash:
        raise ValueError("target transform raw target hash 漂移")
    train_ids = asset.patient_ids[asset.splits == "train"].astype(str).tolist()
    train_hash = patient_hash(train_ids)
    if static_payload.get("train_patient_hash") != train_hash or change_payload.get(
        "train_patient_hash"
    ) != train_hash:
        raise ValueError("target transform 不是由当前 fold train patients 拟合")
    if len(raw) != 375:
        raise ValueError(f"measurement-matched patient 数漂移：{len(raw)}")
    return TargetAssets(
        raw=raw,
        raw_targets_sha256=raw_hash,
        raw_file_sha256=file_sha256(raw_path),
        static_payload=static_payload,
        static_path=static_path,
        static_sha256=file_sha256(static_path),
        change_payload=change_payload,
        change_path=change_path,
        change_sha256=file_sha256(change_path),
    )


def _static_specs(targets: TargetAssets) -> dict[tuple[str, str], Mapping[str, Any]]:
    specs = {
        (str(item["timepoint"]), str(item["feature_name"])): item
        for item in targets.static_payload["specs"]
    }
    expected = {(timepoint, target) for timepoint in TIMEPOINTS for target in RAW_FEATURE_ORDER}
    if set(specs) != expected:
        raise ValueError("static transform cell 不完整")
    return specs


def _change_specs(targets: TargetAssets) -> dict[str, Mapping[str, Any]]:
    specs = {str(item["name"]): item for item in targets.change_payload["features"]}
    if set(specs) != set(RAW_FEATURE_ORDER):
        raise ValueError("change transform cell 不完整")
    return specs


def _reconstruct_static(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(raw, dtype=np.float64)
    if raw.shape != (3, len(RAW_FEATURE_ORDER), 3):
        raise ValueError(f"raw target shape 非法：{raw.shape}")
    values = np.stack(
        (raw[0, :, 0], raw[0, :, 1], raw[1, :, 1], raw[2, :, 1]), axis=0
    )
    valid = np.stack(
        (
            raw[0, :, 2].astype(bool),
            raw[0, :, 2].astype(bool) & raw[1, :, 2].astype(bool),
            raw[1, :, 2].astype(bool) & raw[2, :, 2].astype(bool),
            raw[2, :, 2].astype(bool),
        ),
        axis=0,
    )
    for transition_index in (0, 1):
        left = raw[transition_index, :, 1]
        right = raw[transition_index + 1, :, 0]
        both = raw[transition_index, :, 2].astype(bool) & raw[
            transition_index + 1, :, 2
        ].astype(bool)
        if not np.allclose(left[both], right[both], atol=1e-10, rtol=0.0):
            raise ValueError("shared visit radiomics endpoint 不一致")
    valid &= np.isfinite(values)
    return values, valid


def _analysis_value(values: np.ndarray, spec: Mapping[str, Any]) -> np.ndarray:
    transform = str(spec["value_transform"])
    values = np.asarray(values, dtype=np.float64)
    if transform == "log_epsilon":
        epsilon = float(spec["epsilon"])
        if np.any(values + epsilon <= 0):
            raise ValueError("log_epsilon target 非正")
        return np.log(values + epsilon)
    if transform == "log1p":
        return np.log1p(values)
    if transform == "identity":
        return values
    raise ValueError(f"未知 target value_transform：{transform}")


def _cell_target(
    patient_id: str,
    cell: Cell,
    targets: TargetAssets,
    static_specs: Mapping[tuple[str, str], Mapping[str, Any]],
    change_specs: Mapping[str, Mapping[str, Any]],
) -> tuple[float, float, bool, str]:
    raw = targets.raw.get(patient_id)
    if raw is None:
        return math.nan, math.nan, False, ""
    feature_index = RAW_FEATURE_ORDER.index(cell.target)
    if cell.task == "static":
        values, valid = _reconstruct_static(raw)
        if not valid[cell.index, feature_index]:
            return math.nan, math.nan, False, ""
        natural = float(values[cell.index, feature_index])
        spec = static_specs[(cell.timepoint, cell.target)]
        analysis = float(_analysis_value(np.asarray([natural]), spec)[0])
        clipped = np.clip(
            analysis, float(spec["winsor_low"]), float(spec["winsor_high"])
        )
        standardized = (clipped - float(spec["center"])) / float(spec["scale"])
        transform_label = (
            f"static:{spec['value_transform']}+winsor01_99+median_iqr:"
            f"{cell.timepoint}"
        )
        return float(standardized), natural, True, transform_label
    values = np.asarray(raw[cell.index, feature_index], dtype=np.float64)
    valid = bool(values[2]) and np.isfinite(values[:2]).all()
    if not valid:
        return math.nan, math.nan, False, ""
    spec = change_specs[cell.target]
    transformed_endpoints = _analysis_value(values[:2], spec)
    natural = float(transformed_endpoints[1] - transformed_endpoints[0])
    clipped = np.clip(
        natural, float(spec["winsor_low"]), float(spec["winsor_high"])
    )
    standardized = (clipped - float(spec["center"])) / float(spec["scale"])
    transform_label = (
        f"delta:{spec['value_transform']}_difference+"
        "winsor01_99+median_iqr"
    )
    return float(standardized), natural, True, transform_label


def _inverse_prediction(
    values: np.ndarray,
    cell: Cell,
    static_specs: Mapping[tuple[str, str], Mapping[str, Any]],
    change_specs: Mapping[str, Mapping[str, Any]],
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if cell.task == "static":
        spec = static_specs[(cell.timepoint, cell.target)]
        analysis = values * float(spec["scale"]) + float(spec["center"])
        transform = str(spec["value_transform"])
        if transform == "log_epsilon":
            return np.exp(analysis) - float(spec["epsilon"])
        if transform == "log1p":
            return np.expm1(analysis)
        return analysis
    spec = change_specs[cell.target]
    return values * float(spec["scale"]) + float(spec["center"])


def _prepare_split(
    split: str,
    cell: Cell,
    asset: FeatureAsset,
    targets: TargetAssets,
    static_specs: Mapping[tuple[str, str], Mapping[str, Any]],
    change_specs: Mapping[str, Mapping[str, Any]],
) -> PreparedSplit:
    patient_ids: list[str] = []
    features: list[np.ndarray] = []
    y_standardized: list[float] = []
    y_natural: list[float] = []
    for index in np.flatnonzero(asset.splits == split):
        patient_id = str(asset.patient_ids[index])
        standardized, natural, valid, _ = _cell_target(
            patient_id, cell, targets, static_specs, change_specs
        )
        if not valid:
            continue
        if cell.task == "static":
            feature = asset.response_state[index, cell.index]
        else:
            feature = (
                asset.response_state[index, cell.index + 1]
                - asset.response_state[index, cell.index]
            )
        if feature.shape != (RESPONSE_DIM,) or not np.isfinite(feature).all():
            raise FloatingPointError(f"response feature 非法：{patient_id}/{cell}")
        patient_ids.append(patient_id)
        features.append(feature)
        y_standardized.append(standardized)
        y_natural.append(natural)
    if not features:
        raise ValueError(f"{split} 无有效 rows：{cell}")
    matrix = np.stack(features).astype(np.float64, copy=False)
    standardized_array = np.asarray(y_standardized, dtype=np.float64)
    natural_array = np.asarray(y_natural, dtype=np.float64)
    if not np.isfinite(matrix).all() or not np.isfinite(standardized_array).all():
        raise FloatingPointError(f"{split} probe matrix/target 含 NaN/Inf")
    return PreparedSplit(
        patient_ids=tuple(patient_ids),
        matrix=matrix,
        y_standardized=standardized_array,
        y_natural=natural_array,
    )


def select_single_output_ridge(
    train_matrix: np.ndarray,
    train_target: np.ndarray,
    validation_matrix: np.ndarray,
    validation_target: np.ndarray,
    alphas: Iterable[float] = ALPHAS,
) -> SelectedRidge:
    """不接受 test 参数，从函数签名上阻断 test 参与选择。"""

    train_matrix = np.asarray(train_matrix, dtype=np.float64)
    validation_matrix = np.asarray(validation_matrix, dtype=np.float64)
    train_target = np.asarray(train_target, dtype=np.float64).reshape(-1)
    validation_target = np.asarray(validation_target, dtype=np.float64).reshape(-1)
    if train_matrix.ndim != 2 or validation_matrix.ndim != 2:
        raise ValueError("Ridge feature 必须为二维")
    if train_matrix.shape[1] != validation_matrix.shape[1]:
        raise ValueError("train/val feature_dim 不一致")
    if len(train_matrix) != len(train_target) or len(validation_matrix) != len(
        validation_target
    ):
        raise ValueError("Ridge X/y 行数不一致")
    if len(train_target) < 2 or len(validation_target) < 2:
        raise ValueError("Ridge train/val 样本不足")
    if not all(
        np.isfinite(value).all()
        for value in (train_matrix, validation_matrix, train_target, validation_target)
    ):
        raise FloatingPointError("Ridge train/val 含 NaN/Inf")
    alpha_grid = tuple(sorted(set(float(value) for value in alphas)))
    if not alpha_grid or any(value <= 0 or not math.isfinite(value) for value in alpha_grid):
        raise ValueError("alpha grid 必须为有限正数")
    scaler = StandardScaler().fit(train_matrix)
    train_scaled = scaler.transform(train_matrix)
    validation_scaled = scaler.transform(validation_matrix)
    results: list[tuple[float, float, Ridge]] = []
    for alpha in alpha_grid:
        model = Ridge(
            alpha=alpha,
            fit_intercept=True,
            solver=RIDGE_SOLVER,
            tol=RIDGE_TOL,
            max_iter=RIDGE_MAX_ITER,
        )
        model.fit(train_scaled, train_target)
        prediction = model.predict(validation_scaled)
        mse = float(mean_squared_error(validation_target, prediction))
        if not math.isfinite(mse):
            raise FloatingPointError("validation MSE 非有限")
        results.append((alpha, mse, model))
    best_mse = min(item[1] for item in results)
    eligible = [item for item in results if item[1] <= best_mse + 1e-12]
    alpha, mse, model = min(eligible, key=lambda item: item[0])
    return SelectedRidge(
        scaler=scaler,
        model=model,
        alpha=float(alpha),
        validation_mse=float(mse),
        grid=tuple((float(a), float(score)) for a, score, _ in results),
    )


def _json_vector(values: np.ndarray | Sequence[float]) -> str:
    return json.dumps(
        [float(value) for value in np.asarray(values, dtype=np.float64).reshape(-1)],
        separators=(",", ":"),
    )


def _run_cell(
    *,
    cell: Cell,
    asset: FeatureAsset,
    targets: TargetAssets,
    model_name: str,
    alphas: Sequence[float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    static_specs = _static_specs(targets)
    change_specs = _change_specs(targets)
    train = _prepare_split(
        "train", cell, asset, targets, static_specs, change_specs
    )
    validation = _prepare_split(
        "val", cell, asset, targets, static_specs, change_specs
    )
    selected = select_single_output_ridge(
        train.matrix,
        train.y_standardized,
        validation.matrix,
        validation.y_standardized,
        alphas,
    )
    b0_standardized = float(np.mean(train.y_standardized))
    b0_validation_mse = float(
        mean_squared_error(
            validation.y_standardized,
            np.full(validation.y_standardized.shape, b0_standardized),
        )
    )
    # 只有超参数锁定后才构造 test，并恰好调用一次 predict。
    test = _prepare_split("test", cell, asset, targets, static_specs, change_specs)
    test_scaled = selected.scaler.transform(test.matrix)
    test_guard = _RidgeTestPredictGuard()
    predicted_standardized = test_guard.predict(selected.model, test_scaled)
    if test_guard.call_count != 1:
        raise RuntimeError("outer-test Ridge predict call count 非 1")
    if predicted_standardized.shape != test.y_standardized.shape or not np.isfinite(
        predicted_standardized
    ).all():
        raise FloatingPointError("test Ridge prediction 非法")
    predicted_natural = _inverse_prediction(
        predicted_standardized, cell, static_specs, change_specs
    )
    b0_natural = float(
        _inverse_prediction(
            np.asarray([b0_standardized]), cell, static_specs, change_specs
        )[0]
    )
    _, _, _, transform_label = _cell_target(
        train.patient_ids[0], cell, targets, static_specs, change_specs
    )
    common = {
        "fold": int(asset.metadata["fold"]),
        "model": model_name,
        "task": cell.task,
        "timepoint": cell.timepoint,
        "transition": cell.transition,
        "representation": "response_state",
        "input_variant": cell.input_variant,
        "target": cell.target,
        "feature_dim": RESPONSE_DIM,
        "target_transform": transform_label,
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
        "static_transform_sha256": targets.static_sha256,
        "change_transform_sha256": targets.change_sha256,
        "raw_targets_sha256": targets.raw_targets_sha256,
    }
    prediction_rows = []
    for index, patient_id in enumerate(test.patient_ids):
        prediction_rows.append(
            {
                "patient_id": patient_id,
                **common,
                "split": "test",
                "y_true": float(test.y_natural[index]),
                "y_pred": float(predicted_natural[index]),
                "y_true_standardized": float(test.y_standardized[index]),
                "y_pred_standardized": float(predicted_standardized[index]),
                "b0_prediction": b0_natural,
                "b0_prediction_standardized": b0_standardized,
                "selected_alpha": selected.alpha,
                "test_used_for_target_transform": False,
                "test_used_for_checkpoint_selection": False,
                "test_used_for_lambda_selection": False,
                "test_used_for_scaler": False,
                "test_used_for_alpha_selection": False,
                "test_prediction_guard_enforced": True,
                "test_predict_call_count": test_guard.call_count,
            }
        )
    alpha_scores = {format(a, ".17g"): score for a, score in selected.grid}
    selection_row = {
        **common,
        "n_train": len(train.patient_ids),
        "n_val": len(validation.patient_ids),
        "n_test": len(test.patient_ids),
        "selected_alpha": selected.alpha,
        "val_mse_standardized": selected.validation_mse,
        "alpha_validation_mse_json": json.dumps(alpha_scores, separators=(",", ":")),
        "ridge_solver": RIDGE_SOLVER,
        "ridge_tol": RIDGE_TOL,
        "ridge_max_iter": RIDGE_MAX_ITER,
        "ridge_fit_intercept": True,
        "ridge_intercept": float(selected.model.intercept_),
        "ridge_coef_json": _json_vector(selected.model.coef_),
        "feature_scaler_mean_json": _json_vector(selected.scaler.mean_),
        "feature_scaler_scale_json": _json_vector(selected.scaler.scale_),
        "feature_scaler_n_samples_seen": int(selected.scaler.n_samples_seen_),
        "train_target_mean_standardized": b0_standardized,
        "b0_val_mse_standardized": b0_validation_mse,
        "static_transform_file": str(targets.static_path),
        "change_transform_file": str(targets.change_path),
        "raw_target_file": str(RAW_TARGET_PATH.resolve()),
        "raw_target_file_sha256": targets.raw_file_sha256,
        "sklearn_version": sklearn.__version__,
        "test_used_for_target_transform": False,
        "test_used_for_checkpoint_selection": False,
        "test_used_for_lambda_selection": False,
        "test_used_for_scaler": False,
        "test_used_for_alpha_selection": False,
        "test_prediction_guard_enforced": True,
        "test_predict_call_count": test_guard.call_count,
    }
    return prediction_rows, selection_row


def run_representation_probes(
    *,
    model_name: str,
    fold: int,
    feature_root: Path = DGRS_ROOT / "features",
    prediction_root: Path = DGRS_ROOT / "predictions" / "representation_probes",
    metric_root: Path = DGRS_ROOT / "metrics" / "representation_probes",
    targets: Sequence[str] = TARGETS,
    alphas: Sequence[float] = ALPHAS,
    overwrite: bool = False,
) -> dict[str, Any]:
    """运行一个 model×fold 的 21 个预注册 cell。"""

    model_name = str(model_name).upper()
    targets = tuple(dict.fromkeys(str(value).lower() for value in targets))
    if not targets or not set(targets).issubset(TARGETS):
        raise ValueError(f"targets 必须为 {TARGETS} 的非空子集")
    prediction_dir = Path(prediction_root) / model_name / f"fold_{fold}"
    metric_dir = Path(metric_root) / model_name / f"fold_{fold}"
    prediction_path = prediction_dir / "test_predictions.csv"
    selection_path = metric_dir / "selection_records.csv"
    summary_path = metric_dir / "summary.json"
    _refuse_existing([prediction_path, selection_path, summary_path], overwrite)
    asset = _load_feature_asset(feature_root, model_name, fold)
    target_assets = _load_target_assets(asset, fold)
    cells = [Cell("static", target, index) for index in range(4) for target in targets]
    cells += [Cell("change", target, index) for index in range(3) for target in targets]
    prediction_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for cell in cells:
        rows, selection = _run_cell(
            cell=cell,
            asset=asset,
            targets=target_assets,
            model_name=model_name,
            alphas=alphas,
        )
        prediction_rows.extend(rows)
        selection_rows.append(selection)
    predictions = pd.DataFrame(prediction_rows, columns=PREDICTION_COLUMNS)
    selections = pd.DataFrame(selection_rows, columns=SELECTION_COLUMNS)
    prediction_key = [
        "patient_id",
        "fold",
        "model",
        "task",
        "timepoint",
        "transition",
        "target",
    ]
    selection_key = prediction_key[1:]
    if predictions.empty or predictions.duplicated(prediction_key).any():
        raise ValueError("probe predictions 为空或 key 重复")
    if selections.empty or selections.duplicated(selection_key).any():
        raise ValueError("probe selections 为空或 key 重复")
    if len(selections) != (4 + 3) * len(targets):
        raise ValueError("probe cell 数错误")
    if set(predictions["split"]) != {"test"}:
        raise ValueError("probe prediction 只能含 test")
    expected_test_patients = len(
        set(asset.patient_ids[asset.splits == "test"]) & set(target_assets.raw)
    )
    if predictions["patient_id"].nunique() != expected_test_patients:
        raise ValueError("probe test paired patient coverage 不一致")
    expected_rows = expected_test_patients * len(selections)
    if len(predictions) != expected_rows:
        raise ValueError(f"probe test prediction 行数错误：{len(predictions)} != {expected_rows}")
    selection_flags = (
        "test_used_for_target_transform",
        "test_used_for_checkpoint_selection",
        "test_used_for_lambda_selection",
        "test_used_for_scaler",
        "test_used_for_alpha_selection",
    )
    if predictions[list(selection_flags)].to_numpy(dtype=bool).any() or selections[
        list(selection_flags)
    ].to_numpy(dtype=bool).any():
        raise ValueError("probe 输出声称 test 参与了选择/拟合")
    if not predictions["test_prediction_guard_enforced"].eq(True).all() or not selections[
        "test_prediction_guard_enforced"
    ].eq(True).all():
        raise ValueError("probe test single-use guard 未执行")
    if not predictions["test_predict_call_count"].eq(1).all() or not selections[
        "test_predict_call_count"
    ].eq(1).all():
        raise ValueError("probe test predict call count 非 1")
    numeric = predictions[
        [
            "y_true",
            "y_pred",
            "y_true_standardized",
            "y_pred_standardized",
            "b0_prediction",
            "b0_prediction_standardized",
            "selected_alpha",
        ]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise FloatingPointError("probe prediction 核心数值含 NaN/Inf")
    _atomic_csv(prediction_path, predictions)
    _atomic_csv(selection_path, selections)
    summary = {
        "schema_version": 1,
        "status": "frozen response-state Ridge probes complete",
        "model": model_name,
        "fold": fold,
        "targets": list(targets),
        "tasks": ["static", "change"],
        "representation": "response_state",
        "static_feature": "r_t",
        "delta_feature": "r_(t+1)-r_t",
        "probe_cells": len(selections),
        "test_prediction_rows": len(predictions),
        "test_patient_count": int(predictions["patient_id"].nunique()),
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
        "raw_targets_sha256": target_assets.raw_targets_sha256,
        "static_transform_sha256": target_assets.static_sha256,
        "change_transform_sha256": target_assets.change_sha256,
        "probe_implementation_sha256": probe_implementation_sha256(),
        "sklearn_version": sklearn.__version__,
        "ridge": {
            "type": "single_output",
            "alphas": [float(value) for value in alphas],
            "solver": RIDGE_SOLVER,
            "tol": RIDGE_TOL,
            "max_iter": RIDGE_MAX_ITER,
            "feature_scaler": "StandardScaler fit on fold train only",
            "alpha_selection": "validation standardized MSE; <=1e-12 tie chooses smaller alpha",
        },
        "leakage_guards": {
            "feature_scaler_fit_scope": "outer fold train only",
            "ridge_fit_scope": "outer fold train only",
            "alpha_selection_scope": "outer fold validation only",
            "test_constructed_after_alpha_lock": True,
            "test_predict_calls_per_cell": 1,
            "test_prediction_guard_enforced": True,
            "test_used_for_target_transform": False,
            "test_used_for_checkpoint_selection": False,
            "test_used_for_lambda_selection": False,
            "test_used_for_scaler": False,
            "test_used_for_alpha_selection": False,
            "radiomics_used_as_input": False,
            "ftv_head_output_used_as_input": False,
            "transition_prediction_used_as_input": False,
            "world_model_trained_or_finetuned": False,
        },
    }
    _atomic_json(summary_path, summary)
    return summary


def synthetic_self_test() -> dict[str, Any]:
    generator = np.random.default_rng(20260807)
    train_matrix = generator.normal(size=(64, 11))
    validation_matrix = generator.normal(loc=0.7, size=(31, 11))
    test_matrix = generator.normal(size=(23, 11))
    weights = generator.normal(size=11)
    train_target = train_matrix @ weights + generator.normal(scale=0.05, size=64)
    validation_target = validation_matrix @ weights + generator.normal(
        scale=0.05, size=31
    )
    selected = select_single_output_ridge(
        train_matrix,
        train_target,
        validation_matrix,
        validation_target,
        (1e-4, 1e-2, 1.0, 100.0),
    )
    if not np.allclose(selected.scaler.mean_, train_matrix.mean(axis=0)):
        raise AssertionError("StandardScaler 不是 train-only")
    if np.allclose(selected.scaler.mean_, validation_matrix.mean(axis=0)):
        raise AssertionError("synthetic train/validation 均值意外相等")
    guard = _RidgeTestPredictGuard()
    prediction = guard.predict(selected.model, selected.scaler.transform(test_matrix))
    if prediction.shape != (23,) or not np.isfinite(prediction).all():
        raise AssertionError("synthetic Ridge prediction 非法")
    try:
        guard.predict(selected.model, selected.scaler.transform(test_matrix))
    except RuntimeError:
        second_test_predict_rejected = True
    else:
        raise AssertionError("Ridge outer-test 第二次 predict 未被拒绝")
    return {
        "status": "synthetic representation-probe self-test passed",
        "single_output": True,
        "train_only_scaler_verified": True,
        "validation_only_alpha_selection_enforced_by_signature": True,
        "test_predict_single_use_guard_verified": second_test_predict_rejected,
        "test_predict_call_count": guard.call_count,
        "sklearn_version": sklearn.__version__,
        "selected_alpha": selected.alpha,
        "test_shape": list(prediction.shape),
    }

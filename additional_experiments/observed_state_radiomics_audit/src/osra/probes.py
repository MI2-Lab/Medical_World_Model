"""Frozen observed-state representation 的单输出 Ridge probe。

本模块严格保持 outer-fold 隔离：feature scaler 与 Ridge 只在 train 拟合，
alpha 只由 validation standardized MSE 选择，选择锁定后才对 test 调用一次
``predict``。Radiomics 仅作为监督 target；唯一例外是显式标记的 B1
``current_radiomics`` table baseline。
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from .common import (
    AUDIT_ROOT,
    REPO_ROOT,
    atomic_csv,
    atomic_json,
    file_sha256,
    load_yaml,
    refuse_existing,
    source_sha256,
)
from .extraction import CORE_REPRESENTATIONS, REPRESENTATION_STREAM, TIMEPOINTS
from .targets import (
    FEATURE_NAMES,
    TRANSITIONS,
    StaticTargetTransform,
    change_target,
    load_target_assets,
    static_target,
)


TASK_TYPES = ("static", "change")
CHANGE_VARIANTS = (
    "current_only",
    "observed_pair",
    "observed_difference",
    "observed_combined",
)
TRANSITION_REPRESENTATION = "transition_predicted_delta"
TRANSITION_VARIANT = "predicted_next_delta"
CURRENT_RADIOMICS_REPRESENTATION = "current_radiomics"
STATIC_VARIANT = "current"
ROI_DEPENDENT_REPRESENTATIONS = frozenset(
    {
        "online_roi_mean",
        "ema_roi_mean",
        "mask_geometry",
        "raw_roi_intensity",
    }
)
RIDGE_SOLVER = "lsqr"
RIDGE_TOL = 1e-8
RIDGE_MAX_ITER = 10_000
OUTPUT_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


PREDICTION_COLUMNS = (
    "patient_id",
    "fold",
    "split",
    "task_type",
    "model",
    "run_name",
    "encoder_stream",
    "representation",
    "input_variant",
    "target_name",
    "timepoint",
    "transition",
    "probe_type",
    "feature_dim",
    "y_true_natural",
    "y_pred_natural",
    "y_true_standardized",
    "y_pred_standardized",
    "b0_prediction_natural",
    "b0_prediction_standardized",
    "zero_change_prediction_natural",
    "zero_change_prediction_standardized",
    "selected_alpha",
    "target_value_transform",
    "target_is_exploratory",
    "radiomics_used_as_input",
    "future_radiomics_used_as_input",
    "target_endpoint_mri_used_as_input",
    "observed_visits_used",
    "source_feature_file",
    "source_feature_sha256",
    "source_checkpoint",
    "source_checkpoint_sha256",
)


SELECTION_COLUMNS = (
    "fold",
    "task_type",
    "model",
    "run_name",
    "encoder_stream",
    "representation",
    "input_variant",
    "target_name",
    "timepoint",
    "transition",
    "probe_type",
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
    "zero_change_standardized",
    "target_value_transform",
    "static_transform_sha256",
    "change_transform_sha256",
    "raw_targets_sha256",
    "source_feature_file",
    "source_feature_sha256",
    "source_checkpoint",
    "source_checkpoint_sha256",
    "fold_manifest_sha256",
    "test_used_for_scaler",
    "test_used_for_alpha_selection",
    "test_predict_call_count",
)


@dataclass(frozen=True)
class FeatureAsset:
    path: Path
    sha256: str
    metadata_path: Path
    metadata: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]
    patient_ids: np.ndarray
    splits: np.ndarray
    patient_to_index: Mapping[str, int]


@dataclass(frozen=True)
class TargetAssets:
    raw_targets: Mapping[str, np.ndarray]
    raw_targets_sha256: str
    static_transform: StaticTargetTransform
    static_transform_path: Path
    static_transform_sha256: str
    change_transform: Any
    change_transform_path: Path
    change_transform_sha256: str
    exploratory_targets: frozenset[str]
    source: Any


@dataclass(frozen=True)
class CellSpec:
    task_type: str
    representation: str
    input_variant: str
    target_name: str
    time_index: int | None = None
    transition_index: int | None = None

    @property
    def timepoint(self) -> str:
        return TIMEPOINTS[self.time_index] if self.time_index is not None else ""

    @property
    def transition(self) -> str:
        return (
            TRANSITIONS[self.transition_index]
            if self.transition_index is not None
            else ""
        )


@dataclass(frozen=True)
class PreparedSplit:
    patient_ids: tuple[str, ...]
    matrix: np.ndarray
    target_standardized: np.ndarray
    target_natural: np.ndarray


@dataclass(frozen=True)
class SelectedRidge:
    scaler: StandardScaler
    model: Ridge
    selected_alpha: float
    selected_validation_mse: float
    validation_mse: tuple[tuple[float, float], ...]


def probe_implementation_sha256() -> str:
    """锁定 probe、target/common helper 与 CLI 实现。"""

    return source_sha256(
        [
            Path(__file__),
            Path(__file__).with_name("targets.py"),
            Path(__file__).with_name("common.py"),
            AUDIT_ROOT / "scripts" / "run_probes.py",
        ]
    )


def _json_vector(values: np.ndarray | Sequence[float]) -> str:
    return json.dumps(
        [float(value) for value in np.asarray(values, dtype=np.float64).reshape(-1)],
        separators=(",", ":"),
    )


def _load_feature_asset(
    config: Mapping[str, Any], feature_root: Path, model_label: str, fold: int
) -> FeatureAsset:
    if model_label not in config["models"]:
        raise ValueError(f"未知 model: {model_label}")
    if fold not in range(5):
        raise ValueError(f"fold 必须为 0–4: {fold}")
    directory = feature_root / model_label / f"fold_{fold}"
    path = directory / "observed_features.npz"
    metadata_path = directory / "extraction_metadata.json"
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"缺少正式 feature/metadata: {directory}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_sha = file_sha256(path)
    if feature_sha != metadata.get("feature_file_sha256"):
        raise ValueError(f"feature SHA 与 extraction metadata 不一致: {path}")
    if metadata.get("model") != model_label or int(metadata.get("fold", -1)) != fold:
        raise ValueError("feature metadata model/fold 与请求不一致")
    if metadata.get("max_patients_per_split") is not None:
        raise ValueError("probe 正式入口拒绝 smoke/partial extraction feature")
    expected_run = config["models"][model_label]["run_name"]
    if metadata.get("run_name") != expected_run:
        raise ValueError("feature run_name 与 audit config 不一致")

    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: payload[name].copy() for name in payload.files}
    required = {
        *CORE_REPRESENTATIONS,
        "patient_ids",
        "splits",
        "roi_valid",
        "timepoints",
    }
    if missing := required.difference(arrays):
        raise ValueError(f"feature NPZ 缺字段: {sorted(missing)}")
    patient_ids = arrays["patient_ids"].astype(str)
    splits = arrays["splits"].astype(str)
    if patient_ids.ndim != 1 or splits.shape != patient_ids.shape:
        raise ValueError("patient_ids/splits shape 非法")
    if len(patient_ids) != len(set(patient_ids.tolist())):
        raise ValueError("feature NPZ patient_id 重复")
    if set(splits.tolist()) != {"train", "val", "test"}:
        raise ValueError("feature NPZ 未恰好包含 train/val/test")
    if tuple(arrays["timepoints"].astype(str)) != TIMEPOINTS:
        raise ValueError("feature NPZ timepoint 顺序漂移")
    for representation in CORE_REPRESENTATIONS:
        value = np.asarray(arrays[representation])
        if value.ndim != 3 or value.shape[:2] != (len(patient_ids), len(TIMEPOINTS)):
            raise ValueError(f"{representation} shape 非法: {value.shape}")
    roi_valid = np.asarray(arrays["roi_valid"])
    if roi_valid.shape != (len(patient_ids), len(TIMEPOINTS)):
        raise ValueError(f"roi_valid shape 非法: {roi_valid.shape}")
    if roi_valid.dtype != np.bool_:
        raise TypeError(f"roi_valid 必须为 bool，实际 {roi_valid.dtype}")
    for representation in CORE_REPRESENTATIONS:
        finite = np.isfinite(np.asarray(arrays[representation])).all(axis=-1)
        if representation in ROI_DEPENDENT_REPRESENTATIONS:
            if not np.array_equal(finite, roi_valid):
                raise ValueError(
                    f"{representation} 的 finite rows 未严格对应 roi_valid"
                )
        elif not finite.all():
            raise FloatingPointError(
                f"global representation 含 NaN/Inf: {representation}"
            )
    patient_to_index = {
        patient_id: index for index, patient_id in enumerate(patient_ids)
    }
    return FeatureAsset(
        path=path.resolve(),
        sha256=feature_sha,
        metadata_path=metadata_path.resolve(),
        metadata=metadata,
        arrays=arrays,
        patient_ids=patient_ids,
        splits=splits,
        patient_to_index=patient_to_index,
    )


def _load_target_assets(
    config_path: Path, config: Mapping[str, Any], asset: FeatureAsset, fold: int
) -> TargetAssets:
    bundle, raw_targets, raw_targets_hash, source = load_target_assets(config_path)
    # load_target_assets 已把 source src 加入 sys.path；这里导入锁定实现。
    from rnc.data import patient_hash, split_ids  # type: ignore
    from rnc.transforms import RadiomicsChangeTransform  # type: ignore

    canonical = split_ids(bundle, fold)
    expected_patient_ids = tuple(
        canonical["train"] + canonical["val"] + canonical["test"]
    )
    expected_splits = tuple(
        ["train"] * len(canonical["train"])
        + ["val"] * len(canonical["val"])
        + ["test"] * len(canonical["test"])
    )
    if tuple(asset.patient_ids.tolist()) != expected_patient_ids:
        raise ValueError("feature NPZ patient IDs/顺序与 canonical fold split 不一致")
    if tuple(asset.splits.tolist()) != expected_splits:
        raise ValueError("feature NPZ split labels/顺序与 canonical fold split 不一致")
    source_config = source.load_config(
        source.source_root / config["models"]["m0"]["config"]
    )
    fold_manifest = Path(source_config["data"]["fold_manifest"]).resolve(strict=True)
    actual_manifest_sha = file_sha256(fold_manifest)
    configured_manifest_sha = str(source_config["data"]["fold_manifest_sha256"])
    if actual_manifest_sha != configured_manifest_sha:
        raise ValueError("当前 fold manifest SHA 与 source config 锁定值不一致")
    if asset.metadata.get("fold_manifest_sha256") != actual_manifest_sha:
        raise ValueError("feature metadata 与当前 canonical fold manifest SHA 不一致")
    expected_split_counts = {
        split: len(canonical[split]) for split in ("train", "val", "test")
    }
    if asset.metadata.get("split_counts") != expected_split_counts:
        raise ValueError("feature metadata split counts 与 canonical fold 不一致")

    static_path = AUDIT_ROOT / "configs" / f"static_target_transform_fold_{fold}.json"
    change_path = (
        source.source_root / "configs" / f"radiomics_transform_fold_{fold}.json"
    )
    static = StaticTargetTransform.load(static_path)
    change = RadiomicsChangeTransform.load(change_path)
    current_hash = raw_targets_hash(raw_targets)
    train_ids = asset.patient_ids[asset.splits == "train"].tolist()
    train_hash = patient_hash(train_ids)
    if static.fold != fold or change.fold != fold:
        raise ValueError("target transform fold 与 feature fold 不一致")
    if (
        static.train_patient_hash != train_hash
        or change.train_patient_hash != train_hash
    ):
        raise ValueError("target transform 不是由 feature fold train patient 拟合")
    if (
        static.raw_targets_sha256 != current_hash
        or change.raw_targets_sha256 != current_hash
    ):
        raise ValueError("当前 raw target 与 static/change transform 来源不一致")
    if asset.metadata.get("raw_radiomics_sha256") != current_hash:
        raise ValueError("feature extraction 与 probe raw target SHA 不一致")
    return TargetAssets(
        raw_targets=raw_targets,
        raw_targets_sha256=current_hash,
        static_transform=static,
        static_transform_path=static_path.resolve(),
        static_transform_sha256=file_sha256(static_path),
        change_transform=change,
        change_transform_path=change_path.resolve(),
        change_transform_sha256=file_sha256(change_path),
        exploratory_targets=frozenset(config["targets"].get("exploratory", ())),
        source=source,
    )


def _validate_choices(
    config: Mapping[str, Any],
    task_types: Sequence[str],
    representations: Sequence[str],
    input_variants: Sequence[str],
    target_names: Sequence[str],
) -> None:
    if not task_types or not set(task_types).issubset(TASK_TYPES):
        raise ValueError(f"task_types 必须是 {TASK_TYPES} 的非空子集")
    configured = tuple(config["feature_extraction"]["representations"])
    allowed_representations = {*configured, TRANSITION_REPRESENTATION}
    if not representations or not set(representations).issubset(
        allowed_representations
    ):
        raise ValueError(f"representation 非法；允许 {sorted(allowed_representations)}")
    if not input_variants or not set(input_variants).issubset(CHANGE_VARIANTS):
        raise ValueError(f"input_variants 必须是 {CHANGE_VARIANTS} 的非空子集")
    configured_targets = tuple(config["targets"]["primary"]) + tuple(
        config["targets"].get("exploratory", ())
    )
    if not target_names or not set(target_names).issubset(configured_targets):
        raise ValueError(f"target_names 必须是 {configured_targets} 的非空子集")


def _cell_target(
    patient_id: str, spec: CellSpec, targets: TargetAssets
) -> tuple[float, float, bool]:
    raw = targets.raw_targets.get(patient_id)
    if raw is None:
        return float("nan"), float("nan"), False
    feature_index = FEATURE_NAMES.index(spec.target_name)
    if spec.task_type == "static":
        if spec.time_index is None:
            raise AssertionError("static cell 缺 time_index")
        standardized, natural, _, valid = static_target(
            raw, targets.static_transform, spec.time_index, feature_index
        )
        return standardized, natural, valid
    if spec.transition_index is None:
        raise AssertionError("change cell 缺 transition_index")
    standardized, natural, valid = change_target(
        raw, targets.change_transform, spec.transition_index, feature_index
    )
    return standardized, natural, valid


def _current_radiomics_feature(
    patient_id: str, spec: CellSpec, targets: TargetAssets
) -> tuple[np.ndarray, bool]:
    if spec.task_type != "change" or spec.transition_index is None:
        raise AssertionError("B1 只用于 change")
    raw = targets.raw_targets.get(patient_id)
    if raw is None:
        return np.empty(1, dtype=np.float64), False
    values: list[float] = []
    for feature_index in range(len(FEATURE_NAMES)):
        standardized, _, _, valid = static_target(
            raw,
            targets.static_transform,
            spec.transition_index,
            feature_index,
        )
        if not valid or not math.isfinite(standardized):
            return np.empty(len(FEATURE_NAMES), dtype=np.float64), False
        values.append(standardized)
    return np.asarray(values, dtype=np.float64), True


def _representation_feature(
    asset: FeatureAsset,
    patient_index: int,
    spec: CellSpec,
    transition_features: np.ndarray | None,
) -> tuple[np.ndarray, bool]:
    if spec.representation == TRANSITION_REPRESENTATION:
        if spec.task_type != "change" or spec.transition_index is None:
            raise AssertionError("transition predicted delta 只用于 change")
        if transition_features is None:
            raise AssertionError("缺少 transition predicted delta feature")
        value = np.asarray(
            transition_features[patient_index, spec.transition_index], dtype=np.float64
        )
        return value, bool(np.isfinite(value).all())

    values = np.asarray(asset.arrays[spec.representation], dtype=np.float64)
    roi_valid = np.asarray(asset.arrays["roi_valid"], dtype=bool)
    if spec.task_type == "static":
        if spec.time_index is None:
            raise AssertionError("static cell 缺 time_index")
        if (
            spec.representation in ROI_DEPENDENT_REPRESENTATIONS
            and not roi_valid[patient_index, spec.time_index]
        ):
            return np.empty(values.shape[-1], dtype=np.float64), False
        feature = values[patient_index, spec.time_index]
    else:
        if spec.transition_index is None:
            raise AssertionError("change cell 缺 transition_index")
        if spec.representation in ROI_DEPENDENT_REPRESENTATIONS:
            required_visits = (
                (spec.transition_index,)
                if spec.input_variant == "current_only"
                else (spec.transition_index, spec.transition_index + 1)
            )
            if not all(roi_valid[patient_index, visit] for visit in required_visits):
                multiplier = {
                    "current_only": 1,
                    "observed_pair": 2,
                    "observed_difference": 1,
                    "observed_combined": 3,
                }[spec.input_variant]
                return np.empty(values.shape[-1] * multiplier, dtype=np.float64), False
        start = values[patient_index, spec.transition_index]
        end = values[patient_index, spec.transition_index + 1]
        if spec.input_variant == "current_only":
            feature = start
        elif spec.input_variant == "observed_pair":
            feature = np.concatenate((start, end))
        elif spec.input_variant == "observed_difference":
            feature = end - start
        elif spec.input_variant == "observed_combined":
            feature = np.concatenate((start, end, end - start))
        else:
            raise ValueError(f"未知 change input variant: {spec.input_variant}")
    feature = np.asarray(feature, dtype=np.float64).reshape(-1)
    return feature, bool(np.isfinite(feature).all())


def _prepare_split(
    split: str,
    spec: CellSpec,
    asset: FeatureAsset,
    targets: TargetAssets,
    transition_features: np.ndarray | None,
) -> PreparedSplit:
    patient_ids: list[str] = []
    features: list[np.ndarray] = []
    standardized_targets: list[float] = []
    natural_targets: list[float] = []
    indices = np.flatnonzero(asset.splits == split)
    for patient_index in indices:
        patient_id = str(asset.patient_ids[patient_index])
        target_standardized, target_natural, target_valid = _cell_target(
            patient_id, spec, targets
        )
        if not target_valid:
            continue
        if spec.representation == CURRENT_RADIOMICS_REPRESENTATION:
            feature, feature_valid = _current_radiomics_feature(
                patient_id, spec, targets
            )
        else:
            feature, feature_valid = _representation_feature(
                asset, patient_index, spec, transition_features
            )
        if not feature_valid:
            continue
        if not math.isfinite(target_standardized) or not math.isfinite(target_natural):
            raise FloatingPointError(f"有效 target 却非有限: {patient_id}/{spec}")
        patient_ids.append(patient_id)
        features.append(feature)
        standardized_targets.append(float(target_standardized))
        natural_targets.append(float(target_natural))
    if not features:
        raise ValueError(f"{split} 无有效 probe rows: {spec}")
    matrix = np.stack(features).astype(np.float64, copy=False)
    target_standardized = np.asarray(standardized_targets, dtype=np.float64)
    target_natural = np.asarray(natural_targets, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise FloatingPointError(f"{split} feature matrix 非法: {spec}")
    if len(patient_ids) != len(set(patient_ids)):
        raise ValueError(f"{split} probe patient 重复: {spec}")
    return PreparedSplit(
        patient_ids=tuple(patient_ids),
        matrix=matrix,
        target_standardized=target_standardized,
        target_natural=target_natural,
    )


def select_single_output_ridge(
    train_matrix: np.ndarray,
    train_target: np.ndarray,
    validation_matrix: np.ndarray,
    validation_target: np.ndarray,
    alphas: Iterable[float],
) -> SelectedRidge:
    """只用 train 拟合，按 validation MSE 选 alpha；不接收 test 参数。"""

    train_matrix = np.asarray(train_matrix, dtype=np.float64)
    validation_matrix = np.asarray(validation_matrix, dtype=np.float64)
    train_target = np.asarray(train_target, dtype=np.float64).reshape(-1)
    validation_target = np.asarray(validation_target, dtype=np.float64).reshape(-1)
    if train_matrix.ndim != 2 or validation_matrix.ndim != 2:
        raise ValueError("Ridge input 必须为二维")
    if train_matrix.shape[1] != validation_matrix.shape[1]:
        raise ValueError("train/validation feature_dim 不一致")
    if train_matrix.shape[0] != train_target.size:
        raise ValueError("train X/y 行数不一致")
    if validation_matrix.shape[0] != validation_target.size:
        raise ValueError("validation X/y 行数不一致")
    if train_target.size < 2 or validation_target.size < 2:
        raise ValueError("train/validation 每个 probe cell 至少需要 2 名患者")
    if not all(
        np.isfinite(value).all()
        for value in (train_matrix, validation_matrix, train_target, validation_target)
    ):
        raise FloatingPointError("Ridge train/validation 含 NaN/Inf")
    alpha_values = tuple(sorted({float(value) for value in alphas}))
    if not alpha_values or any(
        not math.isfinite(value) or value <= 0 for value in alpha_values
    ):
        raise ValueError("Ridge alpha 必须是非空正有限值")

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_matrix)
    validation_scaled = scaler.transform(validation_matrix)
    candidates: list[tuple[float, float, Ridge]] = []
    for alpha in alpha_values:
        model = Ridge(
            alpha=alpha,
            fit_intercept=True,
            solver=RIDGE_SOLVER,
            tol=RIDGE_TOL,
            max_iter=RIDGE_MAX_ITER,
        )
        model.fit(train_scaled, train_target)
        prediction = np.asarray(model.predict(validation_scaled), dtype=np.float64)
        mse = float(mean_squared_error(validation_target, prediction))
        if not math.isfinite(mse):
            raise FloatingPointError(f"validation MSE 非有限: alpha={alpha}")
        candidates.append((alpha, mse, model))
    minimum = min(item[1] for item in candidates)
    eligible = [item for item in candidates if item[1] <= minimum + 1e-12]
    alpha, mse, model = min(eligible, key=lambda item: item[0])
    return SelectedRidge(
        scaler=scaler,
        model=model,
        selected_alpha=alpha,
        selected_validation_mse=mse,
        validation_mse=tuple((item[0], item[1]) for item in candidates),
    )


def _inverse_predictions(
    values: np.ndarray, spec: CellSpec, targets: TargetAssets
) -> np.ndarray:
    feature_index = FEATURE_NAMES.index(spec.target_name)
    if spec.task_type == "static":
        transform = targets.static_transform.spec(spec.timepoint, spec.target_name)
        return np.asarray(transform.inverse_prediction(values), dtype=np.float64)
    return np.asarray(
        targets.change_transform.inverse_feature(feature_index, values),
        dtype=np.float64,
    )


def _zero_change_standardized(spec: CellSpec, targets: TargetAssets) -> float:
    if spec.task_type != "change":
        return float("nan")
    feature_index = FEATURE_NAMES.index(spec.target_name)
    transform = targets.change_transform.features[feature_index]
    clipped_zero = float(np.clip(0.0, transform.winsor_low, transform.winsor_high))
    return float((clipped_zero - transform.center) / transform.scale)


def _encoder_stream(representation: str) -> str:
    if representation == CURRENT_RADIOMICS_REPRESENTATION:
        return "table_baseline"
    if representation == TRANSITION_REPRESENTATION:
        return "transition"
    return REPRESENTATION_STREAM[representation]


def _probe_type(representation: str) -> str:
    if representation == CURRENT_RADIOMICS_REPRESENTATION:
        return "current_radiomics_single_output_ridge"
    return "single_output_ridge"


def _observed_visits(spec: CellSpec) -> str:
    if spec.task_type == "static":
        return spec.timepoint
    if spec.transition_index is None:
        raise AssertionError("change cell 缺 transition index")
    if spec.representation == TRANSITION_REPRESENTATION:
        return "+".join(TIMEPOINTS[: spec.transition_index + 1])
    if spec.representation == CURRENT_RADIOMICS_REPRESENTATION:
        return TIMEPOINTS[spec.transition_index]
    if spec.input_variant == "current_only":
        return TIMEPOINTS[spec.transition_index]
    return "+".join(TIMEPOINTS[spec.transition_index : spec.transition_index + 2])


def _target_endpoint_mri_used(spec: CellSpec) -> bool:
    return bool(
        spec.task_type == "change"
        and spec.representation
        not in {TRANSITION_REPRESENTATION, CURRENT_RADIOMICS_REPRESENTATION}
        and spec.input_variant
        in {"observed_pair", "observed_difference", "observed_combined"}
    )


def _target_value_transform(spec: CellSpec, targets: TargetAssets) -> str:
    feature_index = FEATURE_NAMES.index(spec.target_name)
    if spec.task_type == "static":
        return targets.static_transform.spec(
            spec.timepoint, spec.target_name
        ).value_transform
    return targets.change_transform.features[feature_index].value_transform


def _transition_features(
    config: Mapping[str, Any],
    targets: TargetAssets,
    asset: FeatureAsset,
    model_label: str,
    fold: int,
    device_name: str,
    batch_size: int,
) -> np.ndarray:
    """仅从 frozen projected prefix 计算 canonical predicted delta。"""

    if batch_size <= 0:
        raise ValueError("transition batch_size 必须为正")
    import torch

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA 运行 transition，但 CUDA 不可用")
    model_spec = config["models"][model_label]
    checkpoint = (
        targets.source.source_root
        / "checkpoints"
        / model_spec["run_name"]
        / f"fold_{fold}"
        / "best.pt"
    )
    source_config = targets.source.load_config(
        targets.source.source_root / model_spec["config"]
    )
    evaluation = targets.source.load_evaluation(checkpoint, source_config, device)
    if evaluation.fold != fold or evaluation.mode != model_label:
        raise ValueError("transition checkpoint model/fold 与 feature 不一致")
    if evaluation.checkpoint_sha256 != asset.metadata.get("checkpoint_sha256"):
        raise ValueError("transition checkpoint SHA 与 feature extraction 不一致")
    model = evaluation.model
    model.requires_grad_(False).eval()
    online = np.asarray(asset.arrays["online_projected"], dtype=np.float32)
    ema = np.asarray(asset.arrays["ema_projected"], dtype=np.float32)
    output = np.empty(
        (len(asset.patient_ids), len(TRANSITIONS), online.shape[-1]), dtype=np.float32
    )
    with torch.inference_mode():
        for start in range(0, len(asset.patient_ids), batch_size):
            stop = min(start + batch_size, len(asset.patient_ids))
            for transition_index in range(len(TRANSITIONS)):
                # 复制成独立 contiguous prefix；future visit 连底层 storage 都不传入模型。
                prefix = np.ascontiguousarray(
                    online[start:stop, : transition_index + 1]
                )
                transition_output = model.transition(
                    torch.from_numpy(prefix).to(device)
                )[:, -1]
                if model_label == "m0":
                    current = np.ascontiguousarray(ema[start:stop, transition_index])
                    predicted_delta = transition_output - torch.from_numpy(current).to(
                        device
                    )
                else:
                    predicted_delta = transition_output
                output[start:stop, transition_index] = (
                    predicted_delta.float().cpu().numpy()
                )
    if not np.isfinite(output).all():
        raise FloatingPointError("canonical transition predicted delta 含 NaN/Inf")
    del evaluation, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output


def _cell_specs(
    task_types: Sequence[str],
    representations: Sequence[str],
    input_variants: Sequence[str],
    target_names: Sequence[str],
    include_b1: bool,
) -> list[CellSpec]:
    specs: list[CellSpec] = []
    if "static" in task_types:
        for representation in representations:
            if representation == TRANSITION_REPRESENTATION:
                continue
            for time_index in range(len(TIMEPOINTS)):
                for target_name in target_names:
                    specs.append(
                        CellSpec(
                            task_type="static",
                            representation=representation,
                            input_variant=STATIC_VARIANT,
                            target_name=target_name,
                            time_index=time_index,
                        )
                    )
    if "change" in task_types:
        for representation in representations:
            variants = (
                (TRANSITION_VARIANT,)
                if representation == TRANSITION_REPRESENTATION
                else input_variants
            )
            for input_variant in variants:
                for transition_index in range(len(TRANSITIONS)):
                    for target_name in target_names:
                        specs.append(
                            CellSpec(
                                task_type="change",
                                representation=representation,
                                input_variant=input_variant,
                                target_name=target_name,
                                transition_index=transition_index,
                            )
                        )
        if include_b1:
            for transition_index in range(len(TRANSITIONS)):
                for target_name in target_names:
                    specs.append(
                        CellSpec(
                            task_type="change",
                            representation=CURRENT_RADIOMICS_REPRESENTATION,
                            input_variant="current_only",
                            target_name=target_name,
                            transition_index=transition_index,
                        )
                    )
    return specs


def _run_cell(
    spec: CellSpec,
    asset: FeatureAsset,
    targets: TargetAssets,
    transition_features: np.ndarray | None,
    model_label: str,
    alphas: Sequence[float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # 按顺序先完成 train/validation 的 scaler、拟合和 alpha 选择。
    train = _prepare_split("train", spec, asset, targets, transition_features)
    validation = _prepare_split("val", spec, asset, targets, transition_features)
    if train.matrix.shape[1] != validation.matrix.shape[1]:
        raise ValueError(f"train/val feature_dim 漂移: {spec}")
    selected = select_single_output_ridge(
        train.matrix,
        train.target_standardized,
        validation.matrix,
        validation.target_standardized,
        alphas,
    )
    b0_standardized = float(np.mean(train.target_standardized))
    b0_validation_mse = float(
        mean_squared_error(
            validation.target_standardized,
            np.full(validation.target_standardized.shape, b0_standardized),
        )
    )

    # 只有 alpha 锁定以后才构造 test cell，且只调用一次 test predict。
    test = _prepare_split("test", spec, asset, targets, transition_features)
    if test.matrix.shape[1] != train.matrix.shape[1]:
        raise ValueError(f"test feature_dim 漂移: {spec}")
    test_scaled = selected.scaler.transform(test.matrix)
    predicted_standardized = np.asarray(
        selected.model.predict(test_scaled), dtype=np.float64
    ).reshape(-1)
    if predicted_standardized.shape != test.target_standardized.shape:
        raise ValueError("test prediction shape 与 target 不一致")
    if not np.isfinite(predicted_standardized).all():
        raise FloatingPointError(f"test prediction 含 NaN/Inf: {spec}")
    predicted_natural = _inverse_predictions(predicted_standardized, spec, targets)
    b0_natural = float(
        _inverse_predictions(np.asarray([b0_standardized]), spec, targets)[0]
    )
    zero_standardized = _zero_change_standardized(spec, targets)
    feature_dim = int(train.matrix.shape[1])
    encoder_stream = _encoder_stream(spec.representation)
    probe_type = _probe_type(spec.representation)
    target_transform = _target_value_transform(spec, targets)
    radiomics_input = spec.representation == CURRENT_RADIOMICS_REPRESENTATION
    prediction_rows: list[dict[str, Any]] = []
    for patient_index, patient_id in enumerate(test.patient_ids):
        prediction_rows.append(
            {
                "patient_id": patient_id,
                "fold": int(asset.metadata["fold"]),
                "split": "test",
                "task_type": spec.task_type,
                "model": model_label,
                "run_name": asset.metadata["run_name"],
                "encoder_stream": encoder_stream,
                "representation": spec.representation,
                "input_variant": spec.input_variant,
                "target_name": spec.target_name,
                "timepoint": spec.timepoint,
                "transition": spec.transition,
                "probe_type": probe_type,
                "feature_dim": feature_dim,
                "y_true_natural": float(test.target_natural[patient_index]),
                "y_pred_natural": float(predicted_natural[patient_index]),
                "y_true_standardized": float(test.target_standardized[patient_index]),
                "y_pred_standardized": float(predicted_standardized[patient_index]),
                "b0_prediction_natural": b0_natural,
                "b0_prediction_standardized": b0_standardized,
                "zero_change_prediction_natural": (
                    0.0 if spec.task_type == "change" else float("nan")
                ),
                "zero_change_prediction_standardized": zero_standardized,
                "selected_alpha": selected.selected_alpha,
                "target_value_transform": target_transform,
                "target_is_exploratory": spec.target_name
                in targets.exploratory_targets,
                "radiomics_used_as_input": radiomics_input,
                "future_radiomics_used_as_input": False,
                "target_endpoint_mri_used_as_input": _target_endpoint_mri_used(spec),
                "observed_visits_used": _observed_visits(spec),
                "source_feature_file": str(asset.path.relative_to(REPO_ROOT)),
                "source_feature_sha256": asset.sha256,
                "source_checkpoint": asset.metadata["checkpoint"],
                "source_checkpoint_sha256": asset.metadata["checkpoint_sha256"],
            }
        )

    alpha_mse = {format(alpha, ".17g"): mse for alpha, mse in selected.validation_mse}
    selection_row = {
        "fold": int(asset.metadata["fold"]),
        "task_type": spec.task_type,
        "model": model_label,
        "run_name": asset.metadata["run_name"],
        "encoder_stream": encoder_stream,
        "representation": spec.representation,
        "input_variant": spec.input_variant,
        "target_name": spec.target_name,
        "timepoint": spec.timepoint,
        "transition": spec.transition,
        "probe_type": probe_type,
        "feature_dim": feature_dim,
        "n_train": len(train.patient_ids),
        "n_val": len(validation.patient_ids),
        "n_test": len(test.patient_ids),
        "selected_alpha": selected.selected_alpha,
        "val_mse_standardized": selected.selected_validation_mse,
        "alpha_validation_mse_json": json.dumps(alpha_mse, separators=(",", ":")),
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
        "zero_change_standardized": zero_standardized,
        "target_value_transform": target_transform,
        "static_transform_sha256": targets.static_transform_sha256,
        "change_transform_sha256": targets.change_transform_sha256,
        "raw_targets_sha256": targets.raw_targets_sha256,
        "source_feature_file": str(asset.path.relative_to(REPO_ROOT)),
        "source_feature_sha256": asset.sha256,
        "source_checkpoint": asset.metadata["checkpoint"],
        "source_checkpoint_sha256": asset.metadata["checkpoint_sha256"],
        "fold_manifest_sha256": asset.metadata["fold_manifest_sha256"],
        "test_used_for_scaler": False,
        "test_used_for_alpha_selection": False,
        "test_predict_call_count": 1,
    }
    return prediction_rows, selection_row


def run_probe_suite(
    config_path: Path,
    model_label: str,
    fold: int,
    *,
    feature_root: Path,
    prediction_root: Path,
    metric_root: Path,
    output_name: str,
    task_types: Sequence[str] = TASK_TYPES,
    representations: Sequence[str] | None = None,
    input_variants: Sequence[str] = CHANGE_VARIANTS,
    target_names: Sequence[str] | None = None,
    include_b1: bool = True,
    transition_device: str = "cpu",
    transition_batch_size: int = 256,
    overwrite: bool = False,
) -> dict[str, Any]:
    """运行单个 model×fold；默认覆盖完整预注册 probe 矩阵。"""

    config_path = config_path.expanduser().resolve(strict=True)
    config = load_yaml(config_path)
    if not OUTPUT_NAME_PATTERN.fullmatch(output_name):
        raise ValueError(
            "output_name 只能含字母、数字、点、下划线、短横线，且不超过 128 字符"
        )
    configured_representations = tuple(config["feature_extraction"]["representations"])
    representations = (
        (*configured_representations, TRANSITION_REPRESENTATION)
        if representations is None
        else tuple(dict.fromkeys(representations))
    )
    configured_targets = tuple(config["targets"]["primary"]) + tuple(
        config["targets"].get("exploratory", ())
    )
    target_names = (
        configured_targets
        if target_names is None
        else tuple(dict.fromkeys(target_names))
    )
    task_types = tuple(dict.fromkeys(task_types))
    input_variants = tuple(dict.fromkeys(input_variants))
    _validate_choices(config, task_types, representations, input_variants, target_names)
    alphas = tuple(float(value) for value in config["probe"]["alphas"])

    prediction_dir = prediction_root / output_name / model_label / f"fold_{fold}"
    metric_dir = metric_root / output_name / model_label / f"fold_{fold}"
    prediction_path = prediction_dir / "test_predictions.csv"
    selection_path = metric_dir / "selection_records.csv"
    summary_path = metric_dir / "summary.json"
    refuse_existing([prediction_path, selection_path, summary_path], overwrite)

    asset = _load_feature_asset(config, feature_root, model_label, fold)
    targets = _load_target_assets(config_path, config, asset, fold)
    needs_transition = (
        "change" in task_types and TRANSITION_REPRESENTATION in representations
    )
    transition_features = (
        _transition_features(
            config,
            targets,
            asset,
            model_label,
            fold,
            transition_device,
            transition_batch_size,
        )
        if needs_transition
        else None
    )
    specs = _cell_specs(
        task_types, representations, input_variants, target_names, include_b1
    )
    if not specs:
        raise ValueError("筛选后没有 probe cell")
    prediction_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for spec in specs:
        rows, selection = _run_cell(
            spec,
            asset,
            targets,
            transition_features,
            model_label,
            alphas,
        )
        prediction_rows.extend(rows)
        selection_rows.append(selection)

    predictions = pd.DataFrame(prediction_rows, columns=PREDICTION_COLUMNS)
    selections = pd.DataFrame(selection_rows, columns=SELECTION_COLUMNS)
    prediction_key = [
        "patient_id",
        "fold",
        "task_type",
        "model",
        "representation",
        "input_variant",
        "target_name",
        "timepoint",
        "transition",
    ]
    selection_key = prediction_key[1:]
    if predictions.empty or predictions.duplicated(prediction_key).any():
        raise ValueError("prediction rows 为空或 key 重复")
    if selections.empty or selections.duplicated(selection_key).any():
        raise ValueError("selection rows 为空或 key 重复")
    if set(predictions["split"]) != {"test"}:
        raise ValueError("prediction CSV 只能包含 test")
    if predictions[list(PREDICTION_COLUMNS[:14])].isna().any().any():
        raise ValueError("prediction canonical identity columns 意外缺失")
    if not np.isfinite(
        predictions[
            [
                "y_true_natural",
                "y_pred_natural",
                "y_true_standardized",
                "y_pred_standardized",
                "b0_prediction_natural",
                "b0_prediction_standardized",
                "selected_alpha",
            ]
        ].to_numpy(dtype=np.float64)
    ).all():
        raise FloatingPointError("prediction 核心数值列含 NaN/Inf")

    atomic_csv(prediction_path, predictions)
    atomic_csv(selection_path, selections)
    summary = {
        "schema_version": 1,
        "status": "single-output frozen Ridge probes complete",
        "output_name": output_name,
        "model": model_label,
        "run_name": asset.metadata["run_name"],
        "fold": fold,
        "task_types": list(task_types),
        "representations": list(representations),
        "input_variants": list(input_variants),
        "target_names": list(target_names),
        "include_b1_current_radiomics": include_b1 and "change" in task_types,
        "transition_predicted_delta_included": needs_transition,
        "probe_cells": len(selections),
        "test_prediction_rows": len(predictions),
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
        "fold_manifest_sha256": asset.metadata["fold_manifest_sha256"],
        "audit_config": str(config_path),
        "audit_config_sha256": file_sha256(config_path),
        "raw_targets_sha256": targets.raw_targets_sha256,
        "static_transform_sha256": targets.static_transform_sha256,
        "change_transform_sha256": targets.change_transform_sha256,
        "probe_implementation_sha256": probe_implementation_sha256(),
        "ridge": {
            "type": "single_output",
            "alphas": list(alphas),
            "solver": RIDGE_SOLVER,
            "tol": RIDGE_TOL,
            "max_iter": RIDGE_MAX_ITER,
            "feature_scaler": "StandardScaler fit on fold train only",
            "alpha_selection": "fold validation standardized MSE; <=1e-12 tie chooses smaller alpha",
        },
        "leakage_guards": {
            "scaler_fit_scope": "fold train only",
            "ridge_fit_scope": "fold train only",
            "alpha_selection_scope": "fold validation only",
            "test_prediction_after_alpha_lock": True,
            "test_predict_calls_per_cell": 1,
            "test_used_for_scaler": False,
            "test_used_for_alpha_selection": False,
            "future_radiomics_used_as_input": False,
            "transition_uses_only_observed_online_prefix": True,
            "transition_target_delta_or_target_next_used": False,
            "world_model_trained_or_finetuned": False,
            "feature_patient_ids_and_split_order_match_canonical_manifest": True,
            "roi_valid_explicitly_applied": True,
        },
        "b0": "exact-cell train target mean repeated on the same valid test rows",
        "b1": (
            "fold-train static-transformed 4-D current radiomics "
            "[ftv,sphericity,ld,bpe] -> each adjacent change target; explicitly table-dependent"
        ),
        "zero_change": "change rows additionally carry natural zero and its fold-transform standardized value",
    }
    atomic_json(summary_path, summary)
    return summary


def synthetic_self_test() -> dict[str, Any]:
    """无磁盘快速验证 train scaler、validation alpha 与单输出契约。"""

    generator = np.random.default_rng(20260807)
    train_matrix = generator.normal(size=(64, 9))
    validation_matrix = generator.normal(size=(31, 9))
    test_matrix = generator.normal(size=(23, 9))
    weights = generator.normal(size=9)
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
        raise AssertionError("synthetic validation 均值意外等于 train")
    prediction = selected.model.predict(selected.scaler.transform(test_matrix))
    if prediction.shape != (len(test_matrix),) or not np.isfinite(prediction).all():
        raise AssertionError("single-output Ridge test prediction 非法")
    second = select_single_output_ridge(
        train_matrix,
        train_target,
        validation_matrix,
        validation_target,
        (1e-4, 1e-2, 1.0, 100.0),
    )
    if selected.selected_alpha != second.selected_alpha or not np.array_equal(
        selected.model.coef_, second.model.coef_
    ):
        raise AssertionError("固定 lsqr Ridge 不确定")
    return {
        "status": "synthetic probe self-test passed",
        "single_output": True,
        "ridge_solver": RIDGE_SOLVER,
        "selected_alpha": selected.selected_alpha,
        "train_only_scaler_verified": True,
        "validation_only_alpha_verified_by_signature": True,
        "test_shape": list(prediction.shape),
        "finite": True,
    }

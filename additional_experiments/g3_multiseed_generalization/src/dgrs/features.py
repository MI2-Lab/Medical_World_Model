"""G3 multi-seed 实验的冻结 observed response-state 特征提取。

本模块只接受 G1/G3 的显式 ``best.pt``，且不读取任何表型 target。
正式资产统一为 ``[808,4,192]``，患者顺序严格是每个 outer fold 的
train、validation、test 顺序，并锁定 ``effective_seed=seed_base+fold``。
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


DGRS_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = DGRS_ROOT.parents[1]
_PUBLIC_DATA_ROOT = Path(os.environ.get("DGRS_DATA_ROOT", "/path/to/preprocessed"))
DEFAULT_CACHE_ROOT = Path(
    os.environ.get(
        "DGRS_CACHE_ROOT",
        str(
            _PUBLIC_DATA_ROOT
            / "I-SPY2"
            / "_mixed_ispy1_train_cache_dce8_adaptivephase_axiscanonv1_autoroi_"
            "t0fallback_minfrac05_z32_y96_x96"
        ),
    )
)
DEFAULT_FOLD_MANIFEST = Path(
    os.environ.get(
        "DGRS_FOLD_MANIFEST",
        str(
            _PUBLIC_DATA_ROOT
            / "I-SPY2"
            / "_matched_breastdcedl_t0_dicomrepair_rgb224_seed2026"
            / "matched_patient_cv_splits_seed2026.csv"
        ),
    )
)
EXPECTED_FOLD_MANIFEST_SHA256 = (
    "143e482d711225c0611006d99bd7345d2fa1a5c16c65fbaf8399341a0d26aa38"
)
MODELS = ("G1", "G3")
TRAINED_MODELS = MODELS
SEED_BASES = (2026, 3026, 4026, 5026, 6026)
TIMEPOINTS = ("T0", "T1", "T2", "T3")
RESPONSE_DIM = 192


def file_sha256(path: Path) -> str:
    """流式计算 SHA-256。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def patient_hash(patient_ids: Sequence[str]) -> str:
    content = "\n".join(sorted(str(value) for value in patient_ids)).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def ordered_patient_hash(patient_ids: Sequence[str]) -> str:
    """对 patient 顺序敏感的 SHA；与 transform 使用的集合哈希分开。"""

    content = "\n".join(str(value) for value in patient_ids).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def patient_label_hash(patient_ids: Sequence[str], labels: Sequence[int]) -> str:
    """锁定逐行 patient→pCR label 映射，避免二元但错位的 label 静默通过。"""

    if len(patient_ids) != len(labels):
        raise ValueError("patient_ids/labels 长度不一致")
    content = "\n".join(
        f"{patient_id}\t{int(label)}"
        for patient_id, label in zip(patient_ids, labels, strict=True)
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _source_sha256(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(value).resolve() for value in paths):
        label = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
        digest.update(str(label).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def extraction_implementation_sha256() -> str:
    paths = [Path(__file__)]
    model_path = Path(__file__).with_name("model.py")
    if model_path.is_file():
        paths.append(model_path)
    script = DGRS_ROOT / "scripts" / "extract_features.py"
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


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = _temporary_path(path)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _refuse_existing(paths: Sequence[Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("输出已存在，默认拒绝覆盖：" + ", ".join(existing))


def _canonical_fold(
    fold_manifest: Path, fold: int
) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...], np.ndarray]:
    fold_manifest = Path(fold_manifest).expanduser().resolve(strict=True)
    if fold not in range(5):
        raise ValueError(f"fold 必须为 0--4，实际 {fold}")
    actual_hash = file_sha256(fold_manifest)
    if actual_hash != EXPECTED_FOLD_MANIFEST_SHA256:
        raise ValueError(
            "fold manifest SHA 漂移："
            f"{actual_hash} != {EXPECTED_FOLD_MANIFEST_SHA256}"
        )
    frame = pd.read_csv(fold_manifest)
    required = {"patient_id", "fold", "split", "label_pcr"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"fold manifest 缺列：{sorted(missing)}")
    subset = frame.loc[frame["fold"].eq(fold)].copy()
    if len(subset) != 808 or subset["patient_id"].astype(str).duplicated().any():
        raise ValueError(f"fold {fold} 未恰好覆盖 808 名唯一患者")
    if set(subset["split"].astype(str)) != {"train", "val", "test"}:
        raise ValueError(f"fold {fold} split 集合非法")
    ordered_parts = [
        subset.loc[subset["split"].eq(split)].copy()
        for split in ("train", "val", "test")
    ]
    ordered = pd.concat(ordered_parts, ignore_index=True)
    patient_ids = tuple(ordered["patient_id"].astype(str))
    splits = tuple(ordered["split"].astype(str))
    labels = ordered["label_pcr"].to_numpy(dtype=np.int64)
    if not set(labels.tolist()).issubset({0, 1}):
        raise ValueError("pCR label 必须为 0/1")
    split_sets = [
        set(ordered.loc[ordered["split"].eq(split), "patient_id"].astype(str))
        for split in ("train", "val", "test")
    ]
    if any(split_sets[i] & split_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("canonical train/val/test patient overlap")
    return ordered, patient_ids, splits, labels


def _cache_index(cache_root: Path) -> dict[str, Path]:
    cache_root = Path(cache_root).expanduser().resolve(strict=True)
    if not cache_root.is_dir():
        raise NotADirectoryError(cache_root)
    paths = sorted(cache_root.glob("*.npz"))
    output: dict[str, Path] = {}
    marker = "_dce8_"
    for path in paths:
        if marker not in path.name:
            continue
        patient_id = path.name.split(marker, 1)[0]
        if patient_id in output:
            raise ValueError(f"cache patient_id 重复：{patient_id}")
        output[patient_id] = path
    return output


class _FrozenDCE7Dataset(Dataset[dict[str, Any]]):
    """只读 DCE8 cache，并把 DCE7 与 ROI mask 始终分开返回。"""

    def __init__(self, patient_ids: Sequence[str], cache: Mapping[str, Path]) -> None:
        self.patient_ids = tuple(str(value) for value in patient_ids)
        missing = [
            patient_id for patient_id in self.patient_ids if patient_id not in cache
        ]
        if missing:
            raise FileNotFoundError(f"缺少 DCE cache：{missing[:5]}")
        self.paths = tuple(cache[patient_id] for patient_id in self.patient_ids)

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.paths[index]
        with np.load(path, allow_pickle=False) as archive:
            if "x" not in archive:
                raise KeyError(f"cache 缺 x：{path}")
            image = np.asarray(archive["x"], dtype=np.float32)
        if image.shape != (4, 8, 32, 96, 96):
            raise ValueError(f"DCE8 cache shape 漂移：{path} -> {image.shape}")
        dce7 = np.ascontiguousarray(image[:, :7])
        roi_mask = np.ascontiguousarray(image[:, 7:8])
        if not np.isfinite(dce7).all() or not np.isfinite(roi_mask).all():
            raise FloatingPointError(f"cache 含 NaN/Inf：{path}")
        return {
            "patient_id": self.patient_ids[index],
            "image": torch.from_numpy(dce7),
            "roi_mask": torch.from_numpy(roi_mask),
        }


def _safe_payload(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise RuntimeError(
            f"checkpoint 无法通过 weights_only 安全加载：{path}"
        ) from error
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint payload 必须为 mapping")
    return payload


def _load_model_and_payload(
    checkpoint: Path, device: torch.device
) -> tuple[torch.nn.Module, Mapping[str, Any]]:
    """仅用 weights-only payload 与白名单 ``model_config`` 安全重建模型。"""

    checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
    payload = _safe_payload(checkpoint)
    model_config = payload.get("model_config")
    if not isinstance(model_config, Mapping):
        raise ValueError("checkpoint 缺可直接重建的 model_config")
    model_module = importlib.import_module("dgrs.model")
    model_class = getattr(model_module, "DGRSWorldModel")
    model = model_class(**dict(model_config))
    state = payload.get("state_dict", payload.get("model_state"))
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint 缺 state_dict/model_state")
    if not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise TypeError("checkpoint state 必须为 string -> Tensor")
    model.load_state_dict(state, strict=True)
    model.to(device)
    if not isinstance(model, torch.nn.Module) or not isinstance(payload, Mapping):
        raise TypeError("core checkpoint loader 返回类型非法")
    stored_contract = payload.get("architecture_contract")
    contract_method = getattr(model, "architecture_contract", None)
    if not isinstance(stored_contract, Mapping) or not callable(contract_method):
        raise ValueError("checkpoint/model 缺 architecture contract")
    if dict(stored_contract) != dict(contract_method()):
        raise ValueError("checkpoint architecture_contract 与白名单模型不一致")
    model.requires_grad_(False).eval().to(device)
    for name, value in model.state_dict().items():
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(
            value
        ).all():
            raise FloatingPointError(f"checkpoint 参数含 NaN/Inf：{name}")
    return model, payload


def _payload_model_name(payload: Mapping[str, Any]) -> str:
    model_config = payload.get("model_config", {})
    if isinstance(model_config, Mapping):
        value = model_config.get("model_name", model_config.get("model"))
        if value is not None:
            return str(value).upper()
    for key in ("model_name", "model"):
        value = payload.get(key)
        if isinstance(value, str):
            return value.upper()
    return ""


def _validate_checkpoint_contract(
    payload: Mapping[str, Any],
    checkpoint: Path,
    model_name: str,
    fold: int,
    patient_ids: Sequence[str],
    splits: Sequence[str],
    seed_base: int,
    formal: bool,
) -> int:
    if (
        int(payload.get("schema_version", -1)) != 2
        or payload.get("finalized") is not True
    ):
        raise ValueError("特征提取只接受 finalized schema-v2 checkpoint")
    stored_fold = payload.get("fold")
    if int(stored_fold) != fold:
        raise ValueError(f"checkpoint fold 错位：{stored_fold} != {fold}")
    stored_model = _payload_model_name(payload)
    if stored_model and stored_model != model_name:
        raise ValueError(f"checkpoint model 错位：{stored_model} != {model_name}")
    model_config = payload.get("model_config")
    if not isinstance(model_config, Mapping):
        raise ValueError("checkpoint 缺 model_config")
    if int(model_config.get("image_channels", 7)) != 7:
        raise ValueError("G1/G3 checkpoint 必须锁定 7-channel backbone")
    architecture = payload.get("architecture_contract")
    if not isinstance(architecture, Mapping):
        raise ValueError("checkpoint 缺 architecture_contract")
    stored_splits = payload.get("splits")
    if not isinstance(stored_splits, Mapping):
        raise ValueError("checkpoint 缺 canonical splits")
    expected_by_split = {
        split: [
            patient_id
            for patient_id, current_split in zip(patient_ids, splits, strict=True)
            if current_split == split
        ]
        for split in ("train", "val", "test")
    }
    for split, expected in expected_by_split.items():
        actual = stored_splits.get(split)
        if not isinstance(actual, Sequence) or isinstance(actual, (str, bytes)):
            raise TypeError(f"checkpoint.splits.{split} 非 sequence")
        if [str(value) for value in actual] != expected:
            raise ValueError(
                f"checkpoint {split} patient 顺序与 canonical manifest 不一致"
            )
    data_contract = payload.get("data_contract")
    if not isinstance(data_contract, Mapping):
        raise ValueError("checkpoint 缺 data_contract")
    manifest_sha = data_contract.get("fold_manifest_sha256")
    if str(manifest_sha) != EXPECTED_FOLD_MANIFEST_SHA256:
        raise ValueError("checkpoint fold manifest SHA 漂移或缺失")
    if checkpoint.name != "best.pt":
        raise ValueError("正式 frozen extraction 只接受 best.pt")
    stored_seed_base = payload.get("seed_base")
    stored_effective_seed = payload.get("effective_seed")
    if isinstance(stored_seed_base, bool) or not isinstance(stored_seed_base, int):
        raise TypeError("checkpoint 缺显式整数 seed_base")
    if isinstance(stored_effective_seed, bool) or not isinstance(
        stored_effective_seed, int
    ):
        raise TypeError("checkpoint 缺显式整数 effective_seed")
    if stored_seed_base not in SEED_BASES or stored_seed_base != seed_base:
        raise ValueError(
            f"checkpoint seed_base 错位：{stored_seed_base} != {seed_base}"
        )
    expected_effective_seed = seed_base + fold
    if stored_effective_seed != expected_effective_seed:
        raise ValueError(
            "checkpoint effective_seed 违反 seed_base+fold："
            f"{stored_effective_seed} != {expected_effective_seed}"
        )
    train_config = payload.get("train_config")
    if not isinstance(train_config, Mapping) or train_config.get("seed") != seed_base:
        raise ValueError("checkpoint train_config.seed 与 seed_base 不一致")
    runtime = payload.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("checkpoint 缺 runtime seed contract")
    if (
        runtime.get("seed_base") != seed_base
        or runtime.get("fold") != fold
        or runtime.get("effective_seed") != stored_effective_seed
        or runtime.get("seed") != stored_effective_seed
    ):
        raise ValueError("checkpoint runtime seed contract 与顶层字段不一致")
    if formal and runtime.get("smoke") is not False:
        raise ValueError("正式 feature 拒绝 smoke checkpoint")
    determinism = payload.get("determinism")
    if not isinstance(determinism, Mapping):
        raise ValueError("checkpoint 缺 determinism seed contract")
    if (
        determinism.get("seed_base") != seed_base
        or determinism.get("fold") != fold
        or determinism.get("effective_seed") != stored_effective_seed
    ):
        raise ValueError("checkpoint determinism seed contract 与顶层字段不一致")
    plan_path = DGRS_ROOT / "EXPERIMENT_PLAN.md"
    if not plan_path.is_file() or payload.get("plan_sha256") != file_sha256(plan_path):
        raise ValueError("checkpoint plan SHA 与当前冻结实验计划不一致")
    for path_key, hash_key in (
        ("history_path", "history_sha256"),
        ("selection_path", "selection_sha256"),
    ):
        evidence_path = Path(str(payload.get(path_key, "")))
        if not evidence_path.is_file() or payload.get(hash_key) != file_sha256(
            evidence_path
        ):
            raise ValueError(f"checkpoint {path_key}/{hash_key} 终结证据漂移")
    return stored_effective_seed


def _checkpoint_cache_root(
    payload: Mapping[str, Any], requested_cache_root: Path | None
) -> tuple[Path, bool]:
    """解析并锁死 checkpoint 训练 cache；显式 override 只能指向同一目录。"""

    data_contract = payload.get("data_contract")
    if not isinstance(data_contract, Mapping):
        raise ValueError("checkpoint 缺 data_contract，无法验证 cache provenance")
    stored_value = data_contract.get("cache_root")
    if not isinstance(stored_value, (str, os.PathLike)) or not str(stored_value):
        raise ValueError("checkpoint data_contract 缺 cache_root")
    stored_path = Path(stored_value).expanduser()
    if not stored_path.is_absolute():
        raise ValueError("checkpoint data_contract.cache_root 必须为绝对路径")
    stored_path = stored_path.resolve(strict=True)
    if not stored_path.is_dir():
        raise NotADirectoryError(stored_path)
    override_supplied = requested_cache_root is not None
    if requested_cache_root is not None:
        requested = Path(requested_cache_root).expanduser().resolve(strict=True)
        if not requested.is_dir():
            raise NotADirectoryError(requested)
        if requested != stored_path:
            raise ValueError(
                "cache_root 与 checkpoint data_contract 不一致："
                f"{requested} != {stored_path}"
            )
    return stored_path, override_supplied


def _encode_response(
    model: torch.nn.Module,
    image: torch.Tensor,
    roi_mask: torch.Tensor,
    model_name: str,
) -> torch.Tensor:
    encoder = getattr(model, "encode_response", None)
    if not callable(encoder):
        raise AttributeError("DGRSWorldModel 缺 encode_response")
    if model_name not in MODELS:
        raise ValueError(model_name)
    response = encoder(image)
    if not isinstance(response, torch.Tensor):
        raise TypeError("encode_response 必须返回 Tensor")
    if response.shape != (image.shape[0], len(TIMEPOINTS), RESPONSE_DIM):
        raise ValueError(f"response shape 非法：{tuple(response.shape)}")
    if not torch.isfinite(response).all():
        raise FloatingPointError("response 含 NaN/Inf")
    return response


def _normalise_checkpoint_path(
    checkpoint: Path | None, model_name: str, fold: int
) -> Path:
    del model_name, fold
    if checkpoint is None:
        raise ValueError("本实验禁止 checkpoint 自动发现；必须显式传入 best.pt")
    return Path(checkpoint).expanduser().resolve(strict=True)


def _validate_seed_scoped_root(root: Path, seed_base: int, label: str) -> Path:
    resolved = Path(root).expanduser().resolve()
    expected_name = f"seed_{seed_base}"
    if resolved.name != expected_name:
        raise ValueError(f"{label} 必须指向隔离目录 {expected_name}：{resolved}")
    return resolved


def _split_counts(splits: np.ndarray) -> dict[str, int]:
    return {
        split: int(np.count_nonzero(splits == split))
        for split in ("train", "val", "test")
    }


def validate_feature_arrays(
    arrays: Mapping[str, np.ndarray], *, expected_n: int | None = 808
) -> None:
    required = {
        "patient_ids",
        "splits",
        "response_state",
        "timepoints",
        "model",
        "fold",
        "seed_base",
        "effective_seed",
        "label_pcr",
    }
    if missing := required.difference(arrays):
        raise ValueError(f"feature NPZ 缺字段：{sorted(missing)}")
    patient_ids = np.asarray(arrays["patient_ids"]).astype(str)
    splits = np.asarray(arrays["splits"]).astype(str)
    response = np.asarray(arrays["response_state"])
    labels = np.asarray(arrays["label_pcr"])
    n_patients = len(patient_ids)
    if expected_n is not None and n_patients != expected_n:
        raise ValueError(f"feature patient 数错误：{n_patients} != {expected_n}")
    if patient_ids.ndim != 1 or len(set(patient_ids.tolist())) != n_patients:
        raise ValueError("feature patient_ids 非一维唯一序列")
    if splits.shape != patient_ids.shape or labels.shape != patient_ids.shape:
        raise ValueError("feature splits/label_pcr shape 与 patient_ids 不一致")
    if set(splits.tolist()) != {"train", "val", "test"}:
        raise ValueError("feature 未覆盖 train/val/test")
    order = {"train": 0, "val": 1, "test": 2}
    split_order = [order[value] for value in splits.tolist()]
    if split_order != sorted(split_order):
        raise ValueError("feature 患者顺序不是 train+val+test")
    if response.shape != (n_patients, len(TIMEPOINTS), RESPONSE_DIM):
        raise ValueError(f"response_state shape 非法：{response.shape}")
    if response.dtype != np.float32 or not np.isfinite(response).all():
        raise FloatingPointError("response_state 必须为 finite float32")
    if tuple(np.asarray(arrays["timepoints"]).astype(str)) != TIMEPOINTS:
        raise ValueError("timepoint 顺序漂移")
    model_name = str(np.asarray(arrays["model"]).reshape(()).item()).upper()
    if model_name not in MODELS:
        raise ValueError(f"feature model 非法：{model_name}")
    for seed_field in ("fold", "seed_base", "effective_seed"):
        seed_array = np.asarray(arrays[seed_field])
        if seed_array.shape != () or seed_array.dtype.kind not in {"i", "u"}:
            raise TypeError(f"feature {seed_field} 必须为整数 scalar")
    fold = int(np.asarray(arrays["fold"]).reshape(()).item())
    if fold not in range(5):
        raise ValueError(f"feature fold 非法：{fold}")
    seed_base = int(np.asarray(arrays["seed_base"]).reshape(()).item())
    effective_seed = int(np.asarray(arrays["effective_seed"]).reshape(()).item())
    if seed_base not in SEED_BASES:
        raise ValueError(f"feature seed_base 非法：{seed_base}")
    if effective_seed != seed_base + fold:
        raise ValueError("feature effective_seed 违反 seed_base+fold")
    if labels.dtype.kind not in {"i", "u"}:
        raise TypeError("feature label_pcr 必须为整数 dtype，禁止静默截断")
    if not set(labels.tolist()).issubset({0, 1}):
        raise ValueError("feature pCR label 非 0/1")


def _validate_feature_rows(
    arrays: Mapping[str, np.ndarray],
    expected_ids: Sequence[str],
    expected_splits: Sequence[str],
    expected_labels: Sequence[int],
) -> None:
    """逐行闭环 patient、split 与 pCR label，而非只核 shape/集合。"""

    actual_ids = np.asarray(arrays["patient_ids"]).astype(str)
    actual_splits = np.asarray(arrays["splits"]).astype(str)
    actual_labels = np.asarray(arrays["label_pcr"])
    expected_ids_array = np.asarray(expected_ids, dtype=str)
    expected_splits_array = np.asarray(expected_splits, dtype=str)
    expected_labels_array = np.asarray(expected_labels, dtype=np.int64)
    if not np.array_equal(actual_ids, expected_ids_array):
        raise ValueError("feature patient_ids/顺序与 canonical manifest 不一致")
    if not np.array_equal(actual_splits, expected_splits_array):
        raise ValueError("feature splits/顺序与 canonical manifest 不一致")
    if not np.array_equal(
        actual_labels.astype(np.int64, copy=False), expected_labels_array
    ):
        raise ValueError("feature pCR labels 与 canonical manifest 逐行不一致")


def validate_feature_against_canonical(
    arrays: Mapping[str, np.ndarray],
    *,
    fold_manifest: Path = DEFAULT_FOLD_MANIFEST,
    fold: int,
    max_patients_per_split: int | None = None,
) -> dict[str, Any]:
    """对正式或 smoke subset 执行锁定 manifest 的逐行闭环验证。"""

    _, canonical_ids, canonical_splits, canonical_labels = _canonical_fold(
        fold_manifest, fold
    )
    indices = np.arange(len(canonical_ids), dtype=np.int64)
    if max_patients_per_split is not None:
        if max_patients_per_split <= 0:
            raise ValueError("max_patients_per_split 必须为正整数")
        split_array = np.asarray(canonical_splits, dtype=str)
        indices = np.concatenate(
            [
                np.flatnonzero(split_array == split)[:max_patients_per_split]
                for split in ("train", "val", "test")
            ]
        )
    expected_ids = np.asarray(canonical_ids, dtype=str)[indices]
    expected_splits = np.asarray(canonical_splits, dtype=str)[indices]
    expected_labels = canonical_labels[indices].astype(np.int64, copy=False)
    validate_feature_arrays(arrays, expected_n=len(indices))
    _validate_feature_rows(arrays, expected_ids, expected_splits, expected_labels)
    return {
        "canonical_manifest_rows_verified": True,
        "canonical_label_rows_verified": True,
        "canonical_patient_order_sha256": ordered_patient_hash(expected_ids.tolist()),
        "canonical_patient_label_sha256": patient_label_hash(
            expected_ids.tolist(), expected_labels.tolist()
        ),
    }


def _write_asset(
    *,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    output_root: Path,
    model_name: str,
    fold: int,
    seed_base: int,
    overwrite: bool,
) -> dict[str, Any]:
    output_root = _validate_seed_scoped_root(output_root, seed_base, "output_root")
    effective_seed = seed_base + fold
    array_seed_base = int(np.asarray(arrays["seed_base"]).reshape(()).item())
    array_effective_seed = int(np.asarray(arrays["effective_seed"]).reshape(()).item())
    if (array_seed_base, array_effective_seed) != (seed_base, effective_seed):
        raise ValueError("feature NPZ seed contract 与 extraction 请求不一致")
    if (
        metadata.get("seed_base") != seed_base
        or metadata.get("effective_seed") != effective_seed
    ):
        raise ValueError("feature metadata seed contract 与 extraction 请求不一致")
    output_dir = output_root / model_name / f"fold_{fold}"
    feature_path = output_dir / "observed_features.npz"
    metadata_path = output_dir / "extraction_metadata.json"
    manifest_path = output_dir / "feature_manifest_fragment.csv"
    _refuse_existing([feature_path, metadata_path, manifest_path], overwrite)
    fold_manifest_value = metadata.get("fold_manifest")
    if not isinstance(fold_manifest_value, str) or not fold_manifest_value:
        raise ValueError("feature metadata 缺 fold_manifest，无法闭环验证")
    canonical_evidence = validate_feature_against_canonical(
        arrays,
        fold_manifest=Path(fold_manifest_value),
        fold=fold,
        max_patients_per_split=metadata.get("max_patients_per_split"),
    )
    _atomic_npz(feature_path, arrays)
    feature_hash = file_sha256(feature_path)
    metadata.update(
        {
            "schema_version": 2,
            "status": "frozen observed response-state extraction complete",
            "model": model_name,
            "seed_base": seed_base,
            "fold": fold,
            "effective_seed": effective_seed,
            "patient_count": int(len(arrays["patient_ids"])),
            "split_counts": _split_counts(arrays["splits"].astype(str)),
            "feature_shape": list(arrays["response_state"].shape),
            "feature_dtype": str(arrays["response_state"].dtype),
            "feature_file": str(feature_path.resolve()),
            "feature_file_sha256": feature_hash,
            "patient_hash": patient_hash(arrays["patient_ids"].astype(str).tolist()),
            **canonical_evidence,
            "extractor_sha256": extraction_implementation_sha256(),
            "coverage": {
                "expected_primary_patients": 808,
                "observed_primary_patients": int(len(arrays["patient_ids"])),
                "all_four_visits_present": True,
                "response_rows_finite": True,
                "patient_ids_unique": True,
                "formal_complete": metadata.get("max_patients_per_split") is None
                and len(arrays["patient_ids"]) == 808,
            },
            "inference_inputs": ["DCE7"],
            "measurement_targets_read_during_extraction": False,
            "pcr_labels_attached_from_locked_manifest_for_downstream_only": True,
            "world_model_trained_or_finetuned": False,
        }
    )
    patient_ids = arrays["patient_ids"].astype(str)
    splits = arrays["splits"].astype(str)
    labels = arrays["label_pcr"].astype(int)
    try:
        feature_label = str(feature_path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        feature_label = str(feature_path.resolve())
    rows = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "seed_base": seed_base,
            "fold": fold,
            "effective_seed": effective_seed,
            "split": splits,
            "model": model_name,
            "patient_index": np.arange(len(patient_ids), dtype=int),
            "label_pcr": labels,
            "visits": len(TIMEPOINTS),
            "representation": "response_state",
            "feature_dim": RESPONSE_DIM,
            "feature_file": feature_label,
            "feature_file_sha256": feature_hash,
            "source_checkpoint": metadata.get("checkpoint", ""),
            "source_checkpoint_sha256": metadata.get("checkpoint_sha256", ""),
            "fold_manifest_sha256": metadata["fold_manifest_sha256"],
            "canonical_patient_order_sha256": metadata[
                "canonical_patient_order_sha256"
            ],
            "canonical_patient_label_sha256": metadata[
                "canonical_patient_label_sha256"
            ],
            "extractor_sha256": metadata["extractor_sha256"],
        }
    )
    if rows.duplicated(["patient_id", "seed_base", "fold", "model"]).any():
        raise ValueError("feature manifest fragment 出现重复 patient key")
    _atomic_csv(manifest_path, rows)
    metadata["metadata_file"] = str(metadata_path.resolve())
    metadata["manifest_file"] = str(manifest_path.resolve())
    metadata["manifest_file_sha256"] = file_sha256(manifest_path)
    # metadata 不能可靠地在自身内容中记录自身 SHA；其 SHA 由上层 manifest 计算。
    _atomic_json(metadata_path, metadata)
    return metadata


def extract_trained_model(
    *,
    model_name: str,
    fold: int,
    seed_base: int,
    checkpoint: Path,
    output_root: Path,
    device_name: str = "cuda",
    cache_root: Path | None = None,
    fold_manifest: Path = DEFAULT_FOLD_MANIFEST,
    batch_size: int = 16,
    workers: int = 4,
    max_patients_per_split: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """从 G1/G3 显式 best checkpoint 提取全部 primary patient 的 frozen ``r``。"""

    model_name = str(model_name).upper()
    if model_name not in TRAINED_MODELS:
        raise ValueError(f"extract_trained_model 只接受 {TRAINED_MODELS}")
    if (
        isinstance(seed_base, bool)
        or not isinstance(seed_base, int)
        or seed_base not in SEED_BASES
    ):
        raise ValueError(f"seed_base 必须为 {SEED_BASES}")
    output_root = _validate_seed_scoped_root(output_root, seed_base, "output_root")
    if batch_size <= 0 or workers < 0:
        raise ValueError("batch_size 必须为正且 workers 非负")
    if max_patients_per_split is not None and max_patients_per_split <= 0:
        raise ValueError("max_patients_per_split 必须为正整数")
    _, canonical_ids, canonical_splits, canonical_labels = _canonical_fold(
        fold_manifest, fold
    )
    checkpoint_path = _normalise_checkpoint_path(checkpoint, model_name, fold)
    device = torch.device(device_name)
    model, payload = _load_model_and_payload(checkpoint_path, device)
    effective_seed = _validate_checkpoint_contract(
        payload,
        checkpoint_path,
        model_name,
        fold,
        canonical_ids,
        canonical_splits,
        seed_base,
        max_patients_per_split is None,
    )
    cache_root, cache_override_supplied = _checkpoint_cache_root(payload, cache_root)
    cache = _cache_index(cache_root)

    all_ids: list[str] = []
    all_splits: list[str] = []
    all_labels: list[int] = []
    response_parts: list[np.ndarray] = []
    label_lookup = dict(zip(canonical_ids, canonical_labels.tolist(), strict=True))
    for split in ("train", "val", "test"):
        ids = [
            patient_id
            for patient_id, value in zip(canonical_ids, canonical_splits, strict=True)
            if value == split
        ]
        if max_patients_per_split is not None:
            ids = ids[:max_patients_per_split]
        dataset = _FrozenDCE7Dataset(ids, cache)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=device.type == "cuda",
            persistent_workers=workers > 0,
            drop_last=False,
        )
        observed_ids: list[str] = []
        with torch.inference_mode():
            for batch in loader:
                image = batch["image"].to(device, non_blocking=True)
                roi_mask = batch["roi_mask"].to(device, non_blocking=True)
                if image.shape[2] != 7 or roi_mask.shape[2] != 1:
                    raise AssertionError("DCE7/mask channel separation contract 失效")
                response = _encode_response(model, image, roi_mask, model_name)
                response_parts.append(response.float().cpu().numpy())
                batch_ids = [str(value) for value in batch["patient_id"]]
                observed_ids.extend(batch_ids)
                all_ids.extend(batch_ids)
                all_splits.extend([split] * len(batch_ids))
                all_labels.extend(label_lookup[patient_id] for patient_id in batch_ids)
        if observed_ids != ids:
            raise RuntimeError(f"{split} extraction patient 顺序漂移")
    response_state = np.concatenate(response_parts, axis=0).astype(
        np.float32, copy=False
    )
    arrays = {
        "patient_ids": np.asarray(all_ids, dtype=str),
        "splits": np.asarray(all_splits, dtype=str),
        "response_state": response_state,
        "timepoints": np.asarray(TIMEPOINTS, dtype=str),
        "model": np.asarray(model_name),
        "seed_base": np.asarray(seed_base, dtype=np.int64),
        "fold": np.asarray(fold, dtype=np.int64),
        "effective_seed": np.asarray(effective_seed, dtype=np.int64),
        "label_pcr": np.asarray(all_labels, dtype=np.int64),
    }
    contract = payload.get("architecture_contract")
    contract_hash = hashlib.sha256(
        json.dumps(contract, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    metadata = {
        "source_kind": "frozen_dgrs_best_checkpoint",
        "seed_base": seed_base,
        "effective_seed": effective_seed,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_schema_version": payload.get("schema_version"),
        "checkpoint_epoch": payload.get("epoch"),
        "run_name": payload.get("run_name", checkpoint_path.parents[1].name),
        "model_config": dict(payload.get("model_config", {})),
        "architecture_contract_sha256": contract_hash,
        "checkpoint_plan_sha256": payload.get("plan_sha256"),
        "checkpoint_training_implementation_sha256": payload.get(
            "implementation_sha256"
        ),
        "checkpoint_resolved_config_sha256": payload.get("resolved_config_sha256"),
        "shared_initialization_sha256": payload.get("shared_initialization_sha256"),
        "checkpoint_source_commit": payload.get("source_commit"),
        "checkpoint_git": payload.get("git"),
        "checkpoint_split_hashes": payload.get("split_hashes"),
        "ftv_transform_path": payload.get("ftv_transform_path"),
        "ftv_transform_sha256": payload.get("ftv_transform_sha256"),
        "fold_manifest": str(Path(fold_manifest).resolve()),
        "fold_manifest_sha256": EXPECTED_FOLD_MANIFEST_SHA256,
        "cache_root": str(cache_root.resolve()),
        "checkpoint_cache_contract_match": True,
        "cache_override_supplied": cache_override_supplied,
        "max_patients_per_split": max_patients_per_split,
        "device": str(device),
        "online_encoder": True,
        "ftv_head_loaded_but_not_called": model_name == "G3",
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return _write_asset(
        arrays=arrays,
        metadata=metadata,
        output_root=output_root,
        model_name=model_name,
        fold=fold,
        seed_base=seed_base,
        overwrite=overwrite,
    )


def extract_response_subset(
    *,
    model_name: str,
    checkpoint: Path,
    patient_ids: Sequence[str],
    device_name: str = "cuda",
    cache_root: Path | None = None,
    batch_size: int = 16,
    workers: int = 4,
) -> np.ndarray:
    """为 validation/pilot 提取显式 patient list，不读取 manifest 或 target。

    调用者负责只传允许的 train/validation IDs；本函数保持给定顺序，返回
    ``[N,4,192]``，且不写正式 feature 资产。它故意不接受 split/test/FTV
    参数，因此 lambda pilot 可在不触碰 test target 的前提下复用同一编码逻辑。
    """

    model_name = str(model_name).upper()
    if model_name not in TRAINED_MODELS:
        raise ValueError(f"subset extraction 只接受 {TRAINED_MODELS}")
    patient_ids = tuple(str(value) for value in patient_ids)
    if not patient_ids or len(patient_ids) != len(set(patient_ids)):
        raise ValueError("patient_ids 必须为非空唯一序列")
    if batch_size <= 0 or workers < 0:
        raise ValueError("batch_size 必须为正且 workers 非负")
    checkpoint_path = Path(checkpoint).expanduser().resolve(strict=True)
    device = torch.device(device_name)
    model, payload = _load_model_and_payload(checkpoint_path, device)
    stored_model = _payload_model_name(payload)
    if stored_model and stored_model != model_name:
        raise ValueError(f"checkpoint model 错位：{stored_model} != {model_name}")
    resolved_cache_root, _ = _checkpoint_cache_root(payload, cache_root)
    cache = _cache_index(resolved_cache_root)
    dataset = _FrozenDCE7Dataset(patient_ids, cache)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        drop_last=False,
    )
    parts: list[np.ndarray] = []
    observed_ids: list[str] = []
    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            roi_mask = batch["roi_mask"].to(device, non_blocking=True)
            response = _encode_response(model, image, roi_mask, model_name)
            parts.append(response.float().cpu().numpy())
            observed_ids.extend(str(value) for value in batch["patient_id"])
    if tuple(observed_ids) != patient_ids:
        raise RuntimeError("subset extraction patient 顺序漂移")
    output = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
    if output.shape != (len(patient_ids), len(TIMEPOINTS), RESPONSE_DIM):
        raise ValueError(f"subset response shape 非法：{output.shape}")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output


def extract_model(
    *,
    model_name: str,
    fold: int,
    seed_base: int,
    checkpoint: Path,
    output_root: Path,
    device_name: str = "cuda",
    cache_root: Path | None = None,
    fold_manifest: Path = DEFAULT_FOLD_MANIFEST,
    batch_size: int = 16,
    workers: int = 4,
    max_patients_per_split: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    model_name = str(model_name).upper()
    return extract_trained_model(
        model_name=model_name,
        fold=fold,
        seed_base=seed_base,
        checkpoint=checkpoint,
        device_name=device_name,
        output_root=output_root,
        cache_root=cache_root,
        fold_manifest=fold_manifest,
        batch_size=batch_size,
        workers=workers,
        max_patients_per_split=max_patients_per_split,
        overwrite=overwrite,
    )


def synthetic_self_test() -> dict[str, Any]:
    generator = np.random.default_rng(20260807)
    counts = (13, 5, 7)
    splits = np.asarray(
        ["train"] * counts[0] + ["val"] * counts[1] + ["test"] * counts[2]
    )
    arrays = {
        "patient_ids": np.asarray([f"P{index:03d}" for index in range(sum(counts))]),
        "splits": splits,
        "response_state": generator.normal(size=(sum(counts), 4, RESPONSE_DIM)).astype(
            np.float32
        ),
        "timepoints": np.asarray(TIMEPOINTS),
        "model": np.asarray("G3"),
        "seed_base": np.asarray(2026, dtype=np.int64),
        "fold": np.asarray(0, dtype=np.int64),
        "effective_seed": np.asarray(2026, dtype=np.int64),
        "label_pcr": generator.integers(0, 2, size=sum(counts), dtype=np.int64),
    }
    validate_feature_arrays(arrays, expected_n=sum(counts))
    _validate_feature_rows(
        arrays,
        arrays["patient_ids"].astype(str),
        arrays["splits"].astype(str),
        arrays["label_pcr"].astype(np.int64),
    )
    corrupted = dict(arrays)
    corrupted["label_pcr"] = arrays["label_pcr"].copy()
    corrupted["label_pcr"][0] = 1 - corrupted["label_pcr"][0]
    try:
        _validate_feature_rows(
            corrupted,
            arrays["patient_ids"].astype(str),
            arrays["splits"].astype(str),
            arrays["label_pcr"].astype(np.int64),
        )
    except ValueError:
        canonical_label_mismatch_rejected = True
    else:
        raise AssertionError("canonical pCR label 错位未被拒绝")
    float_labels = dict(arrays)
    float_labels["label_pcr"] = arrays["label_pcr"].astype(np.float64)
    try:
        validate_feature_arrays(float_labels, expected_n=sum(counts))
    except TypeError:
        float_label_rejected = True
    else:
        raise AssertionError("非整数 label dtype 未被拒绝")
    float_seed = dict(arrays)
    float_seed["seed_base"] = np.asarray(2026.0, dtype=np.float64)
    try:
        validate_feature_arrays(float_seed, expected_n=sum(counts))
    except TypeError:
        non_integer_seed_rejected = True
    else:
        raise AssertionError("非整数 seed scalar 未被拒绝")
    try:
        _normalise_checkpoint_path(None, "G1", 0)
    except ValueError:
        implicit_checkpoint_rejected = True
    else:
        raise AssertionError("checkpoint 自动发现未被禁用")
    with tempfile.TemporaryDirectory(prefix="dgrs-feature-selftest-") as directory:
        contract_cache = Path(directory) / "contract_cache"
        other_cache = Path(directory) / "other_cache"
        contract_cache.mkdir()
        other_cache.mkdir()
        payload = {"data_contract": {"cache_root": str(contract_cache.resolve())}}
        resolved, override = _checkpoint_cache_root(payload, contract_cache)
        if resolved != contract_cache.resolve() or not override:
            raise AssertionError("matching checkpoint cache contract 未通过")
        try:
            _checkpoint_cache_root(payload, other_cache)
        except ValueError:
            cache_mismatch_rejected = True
        else:
            raise AssertionError("checkpoint cache override 漂移未被拒绝")
    return {
        "status": "synthetic feature contract self-test passed",
        "shape": list(arrays["response_state"].shape),
        "dtype": str(arrays["response_state"].dtype),
        "dce_channels": 7,
        "mask_stored_in_feature_asset": False,
        "target_values_required": False,
        "canonical_row_label_closure_verified": canonical_label_mismatch_rejected,
        "non_integer_label_rejected": float_label_rejected,
        "non_integer_seed_rejected": non_integer_seed_rejected,
        "implicit_checkpoint_rejected": implicit_checkpoint_rejected,
        "checkpoint_cache_mismatch_rejected": cache_mismatch_rejected,
    }

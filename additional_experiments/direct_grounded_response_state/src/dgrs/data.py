"""DCE7 与分离 ROI mask 的 locked-cohort data adapter。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .targets import TRANSITIONS, patient_hash


@dataclass(frozen=True)
class PatientRecord:
    patient_id: str
    cache_path: Path
    source: str
    pcr: int | None
    has_ftv: bool


@dataclass(frozen=True)
class CohortBundle:
    primary: tuple[PatientRecord, ...]
    extra_pretrain: tuple[PatientRecord, ...]
    folds: pd.DataFrame
    raw_ftv: Mapping[str, tuple[np.ndarray, np.ndarray]]
    primary_labels: pd.DataFrame

    @property
    def by_id(self) -> dict[str, PatientRecord]:
        return {item.patient_id: item for item in (*self.primary, *self.extra_pretrain)}


def _cache_patient_id(path: Path) -> str:
    marker = "_dce8_"
    if marker not in path.name:
        raise ValueError(f"无法解析 DCE8 cache patient ID: {path}")
    return path.name.split(marker, 1)[0]


def cache_index(cache_root: str | Path) -> dict[str, Path]:
    paths = sorted(Path(cache_root).glob("*.npz"))
    index = {_cache_patient_id(path): path for path in paths}
    if not paths:
        raise FileNotFoundError(f"DCE8 cache 为空: {cache_root}")
    if len(index) != len(paths):
        raise ValueError("DCE8 cache 存在重复 patient ID")
    return index


def read_raw_ftv(path: str | Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """由三个相邻 transition 严格恢复四访 static FTV。"""

    frame = pd.read_csv(path)
    required = {
        "patient_id",
        "transition",
        "start_visit",
        "end_visit",
        "ftv_start",
        "ftv_end",
        "ftv_valid",
    }
    if missing := required.difference(frame.columns):
        raise ValueError(f"FTV raw target 缺列: {sorted(missing)}")
    output: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for patient_id, rows in frame.groupby("patient_id", sort=False):
        if rows["transition"].duplicated().any():
            raise ValueError(f"{patient_id} transition 重复")
        rows = rows.set_index("transition").reindex(TRANSITIONS)
        if rows[list(required - {"patient_id", "transition"})].isna().all(axis=1).any():
            raise ValueError(f"{patient_id} transition 不完整")
        for index, transition in enumerate(TRANSITIONS):
            expected_start, expected_end = transition.split("→")
            row = rows.loc[transition]
            if str(row["start_visit"]) != expected_start or str(row["end_visit"]) != expected_end:
                raise ValueError(f"{patient_id}/{transition} visit 对齐错误")
            if index < 2:
                left = float(row["ftv_end"])
                right = float(rows.iloc[index + 1]["ftv_start"])
                both = bool(row["ftv_valid"]) and bool(rows.iloc[index + 1]["ftv_valid"])
                if both and not np.isclose(left, right, rtol=0.0, atol=1e-10):
                    raise ValueError(f"{patient_id}/{expected_end} 的共享 FTV endpoint 不一致")
        values = np.asarray(
            [rows.iloc[0]["ftv_start"], rows.iloc[0]["ftv_end"], rows.iloc[1]["ftv_end"], rows.iloc[2]["ftv_end"]],
            dtype=np.float64,
        )
        transition_valid = rows["ftv_valid"].astype(bool).to_numpy()
        valid = np.asarray(
            [transition_valid[0], transition_valid[0] and transition_valid[1], transition_valid[1] and transition_valid[2], transition_valid[2]],
            dtype=bool,
        )
        valid &= np.isfinite(values)
        output[str(patient_id)] = (values, valid)
    return output


def validate_fold_manifest(folds: pd.DataFrame, primary_ids: Iterable[str]) -> None:
    required = {"patient_id", "fold", "split", "label_pcr"}
    if missing := required.difference(folds.columns):
        raise ValueError(f"fold manifest 缺列: {sorted(missing)}")
    primary_ids = set(str(value) for value in primary_ids)
    if set(folds["fold"].unique()) != set(range(5)):
        raise ValueError("fold manifest 必须恰好含 fold 0–4")
    if not set(folds["split"]).issubset({"train", "val", "test"}):
        raise ValueError("fold manifest 含未知 split")
    for fold in range(5):
        current = folds.loc[folds["fold"] == fold].copy()
        ids = current["patient_id"].astype(str)
        if ids.duplicated().any() or set(ids) != primary_ids:
            raise ValueError(f"fold {fold} 未唯一覆盖 primary cohort")
        split_sets = [set(ids[current["split"].eq(name)]) for name in ("train", "val", "test")]
        if any(split_sets[i] & split_sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError(f"fold {fold} split 不互斥")
    test_counts = folds.assign(is_test=folds["split"].eq("test")).groupby("patient_id")["is_test"].sum()
    if not test_counts.eq(1).all():
        raise ValueError("每位 primary 患者必须恰好一次进入 test")


def load_cohort_bundle(
    primary_labels_path: str | Path,
    extra_labels_path: str | Path,
    folds_path: str | Path,
    ftv_targets_path: str | Path,
    cache_root: str | Path,
    overlap_path: str | Path | None = None,
) -> CohortBundle:
    labels = pd.read_csv(primary_labels_path)
    extra_labels = pd.read_csv(extra_labels_path)
    folds = pd.read_csv(folds_path)
    raw_ftv = read_raw_ftv(ftv_targets_path)
    cache = cache_index(cache_root)
    labels["patient_id"] = labels["patient_id"].astype(str)
    folds["patient_id"] = folds["patient_id"].astype(str)
    if labels["patient_id"].duplicated().any() or len(labels) != 808:
        raise ValueError("primary labels 必须是 808 名唯一患者")
    validate_fold_manifest(folds, labels["patient_id"])
    manifest_labels = folds.groupby("patient_id")["label_pcr"].nunique()
    if not manifest_labels.eq(1).all():
        raise ValueError("同一患者在五折 manifest 的 pCR 不一致")
    canonical_manifest = folds.drop_duplicates("patient_id").set_index("patient_id")["label_pcr"].astype(int)
    canonical_labels = labels.set_index("patient_id")["label_pcr"].astype(int)
    if not canonical_manifest.equals(canonical_labels.loc[canonical_manifest.index]):
        raise ValueError("manifest 与 clinical labels 的 pCR 不一致")
    if overlap_path is not None:
        overlap = pd.read_csv(overlap_path)
        overlap["patient_id"] = overlap["patient_id"].astype(str)
        if overlap["patient_id"].duplicated().any() or set(overlap["patient_id"]) != set(labels["patient_id"]):
            raise ValueError("FTV overlap 未唯一覆盖 primary cohort")
        expected = set(overlap.loc[overlap["has_radiomics"].astype(bool), "patient_id"])
        if expected != set(raw_ftv):
            raise ValueError("FTV overlap 与 raw targets patient set 不一致")
    primary: list[PatientRecord] = []
    for row in labels.itertuples(index=False):
        patient_id = str(row.patient_id)
        if patient_id not in cache:
            raise FileNotFoundError(f"primary patient 缺 cache: {patient_id}")
        primary.append(PatientRecord(patient_id, cache[patient_id], "ispy2", int(row.label_pcr), patient_id in raw_ftv))
    complete_extra = extra_labels.loc[extra_labels["complete_4visits"].astype(bool)].copy()
    extra: list[PatientRecord] = []
    for row in complete_extra.itertuples(index=False):
        patient_id = str(row.patient_id)
        if patient_id not in cache:
            raise FileNotFoundError(f"extra pretrain patient 缺 cache: {patient_id}")
        extra.append(PatientRecord(patient_id, cache[patient_id], "ispy1", None, False))
    if len(extra) != 156:
        raise ValueError(f"I-SPY1 complete-four-visit 应为 156，实际 {len(extra)}")
    if {item.patient_id for item in primary} & {item.patient_id for item in extra}:
        raise ValueError("I-SPY1/I-SPY2 patient ID 有交集")
    return CohortBundle(tuple(primary), tuple(extra), folds, raw_ftv, labels)


def split_ids(bundle: CohortBundle, fold: int) -> dict[str, list[str]]:
    if fold not in range(5):
        raise ValueError("fold 必须为 0–4")
    current = bundle.folds.loc[bundle.folds["fold"] == fold]
    output = {
        name: current.loc[current["split"] == name, "patient_id"].astype(str).tolist()
        for name in ("train", "val", "test")
    }
    output["pretrain_train"] = output["train"] + [item.patient_id for item in bundle.extra_pretrain]
    return output


def records_for_ids(bundle: CohortBundle, patient_ids: Iterable[str]) -> list[PatientRecord]:
    lookup = bundle.by_id
    ids = [str(value) for value in patient_ids]
    if missing := [patient_id for patient_id in ids if patient_id not in lookup]:
        raise KeyError(f"未知 patient IDs: {missing[:5]}")
    return [lookup[patient_id] for patient_id in ids]


class LongitudinalDGRSDataset(Dataset):
    """每次只读一次 NPZ，永远分开返回 DCE7 与命名明确的 ROI mask。"""

    def __init__(
        self,
        records: Iterable[PatientRecord],
        transformed_ftv: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None,
        raw_ftv: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> None:
        self.records = tuple(records)
        self.transformed_ftv = transformed_ftv or {}
        self.raw_ftv = raw_ftv or {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        with np.load(record.cache_path) as archive:
            cached = np.asarray(archive["x"], dtype=np.float32)
        if cached.shape != (4, 8, 32, 96, 96):
            raise ValueError(f"DCE8 cache shape 错误: {record.cache_path} -> {cached.shape}")
        image = np.ascontiguousarray(cached[:, :7])
        roi_mask = np.ascontiguousarray(cached[:, 7:8])
        if not np.isfinite(image).all() or not np.isfinite(roi_mask).all():
            raise ValueError(f"cache 出现非有限值: {record.cache_path}")
        target, valid = self.transformed_ftv.get(
            record.patient_id,
            (np.zeros(4, dtype=np.float32), np.zeros(4, dtype=bool)),
        )
        raw_target, raw_valid = self.raw_ftv.get(
            record.patient_id,
            (np.full(4, np.nan, dtype=np.float64), np.zeros(4, dtype=bool)),
        )
        target = np.asarray(target, dtype=np.float32)
        valid = np.asarray(valid, dtype=bool)
        raw_target = np.asarray(raw_target, dtype=np.float32)
        raw_valid = np.asarray(raw_valid, dtype=bool)
        if any(value.shape != (4,) for value in (target, valid, raw_target, raw_valid)):
            raise ValueError(f"{record.patient_id} 的 FTV target shape 非法")
        if not np.array_equal(valid, raw_valid):
            raise ValueError(f"{record.patient_id} transformed/raw FTV mask 不一致")
        roi_valid = roi_mask.reshape(4, -1).sum(axis=1) > 0
        pcr = -1 if record.pcr is None else int(record.pcr)
        return {
            "patient_id": record.patient_id,
            "source": record.source,
            "image": torch.from_numpy(image),
            "roi_mask": torch.from_numpy(roi_mask),
            "roi_valid": torch.from_numpy(roi_valid),
            "ftv_target": torch.from_numpy(target),
            "ftv_raw": torch.from_numpy(raw_target),
            "ftv_mask": torch.from_numpy(valid),
            "has_ftv": bool(valid.any()),
            "pcr": pcr,
            "label_pcr": pcr,
        }


# 兼容评估代码中更短的名字，同时保持唯一实现。
LongitudinalCacheDataset = LongitudinalDGRSDataset


__all__ = [
    "CohortBundle",
    "LongitudinalCacheDataset",
    "LongitudinalDGRSDataset",
    "PatientRecord",
    "load_cohort_bundle",
    "patient_hash",
    "read_raw_ftv",
    "records_for_ids",
    "split_ids",
]

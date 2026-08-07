"""只从锁定 patient manifest 与只读 DCE8 cache 构建纵向数据集。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


FEATURE_NAMES = ("ftv", "sphericity", "ld", "bpe")
TRANSITIONS = ("T0→T1", "T1→T2", "T2→T3")


@dataclass(frozen=True)
class PatientRecord:
    patient_id: str
    cache_path: Path
    source: str
    pcr: int | None
    has_radiomics: bool


@dataclass(frozen=True)
class CohortBundle:
    primary: tuple[PatientRecord, ...]
    extra_pretrain: tuple[PatientRecord, ...]
    folds: pd.DataFrame
    raw_radiomics: Mapping[str, np.ndarray]
    primary_labels: pd.DataFrame

    @property
    def by_id(self) -> dict[str, PatientRecord]:
        return {record.patient_id: record for record in (*self.primary, *self.extra_pretrain)}


def _cache_patient_id(path: Path) -> str:
    marker = "_dce8_"
    if marker not in path.name:
        raise ValueError(f"无法从 cache 文件名解析患者 ID: {path}")
    return path.name.split(marker, 1)[0]


def cache_index(cache_root: Path) -> dict[str, Path]:
    paths = sorted(cache_root.glob("*.npz"))
    output = {_cache_patient_id(path): path for path in paths}
    if len(output) != len(paths):
        raise ValueError("DCE8 cache 存在重复 patient_id")
    return output


def read_raw_radiomics(path: Path) -> dict[str, np.ndarray]:
    frame = pd.read_csv(path)
    required = {"patient_id", "transition", "start_visit", "end_visit"}
    for feature in FEATURE_NAMES:
        required.update((f"{feature}_start", f"{feature}_end", f"{feature}_valid"))
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"radiomics target 文件缺列: {sorted(missing)}")
    output: dict[str, np.ndarray] = {}
    for patient_id, group in frame.groupby("patient_id", sort=False):
        group = group.set_index("transition").reindex(TRANSITIONS)
        if group.index.has_duplicates or len(group) != 3:
            raise ValueError(f"{patient_id} 的 transition 不唯一/不完整")
        values = np.full((3, len(FEATURE_NAMES), 3), np.nan, dtype=np.float64)
        for step, (_, row) in enumerate(group.iterrows()):
            expected_start, expected_end = TRANSITIONS[step].split("→")
            if str(row["start_visit"]) != expected_start or str(row["end_visit"]) != expected_end:
                raise ValueError(f"{patient_id}/{TRANSITIONS[step]} 的 start/end visit 错位")
            for feature_index, feature in enumerate(FEATURE_NAMES):
                values[step, feature_index, 0] = float(row[f"{feature}_start"])
                values[step, feature_index, 1] = float(row[f"{feature}_end"])
                values[step, feature_index, 2] = float(bool(row[f"{feature}_valid"]))
        output[str(patient_id)] = values
    return output


def validate_fold_manifest(folds: pd.DataFrame, patient_ids: Iterable[str]) -> None:
    patient_ids = set(patient_ids)
    required = {"patient_id", "fold", "split", "label_pcr"}
    if missing := required.difference(folds.columns):
        raise ValueError(f"五折 manifest 缺列: {sorted(missing)}")
    if set(folds["fold"].unique()) != set(range(5)):
        raise ValueError("五折 manifest 必须恰好包含 fold 0–4")
    if not set(folds["split"]).issubset({"train", "val", "test"}):
        raise ValueError("五折 manifest 含未知 split")
    for fold in range(5):
        subset = folds.loc[folds["fold"] == fold]
        if subset["patient_id"].duplicated().any() or set(subset["patient_id"]) != patient_ids:
            raise ValueError(f"fold {fold} 未恰好覆盖锁定 primary cohort")
        split_sets = [set(subset.loc[subset["split"] == split, "patient_id"]) for split in ("train", "val", "test")]
        if any(split_sets[i] & split_sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError(f"fold {fold} 的 train/val/test 有交集")
    test_count = folds.assign(is_test=folds["split"].eq("test")).groupby("patient_id")["is_test"].sum()
    if not test_count.eq(1).all():
        raise ValueError("并非每位 primary 患者恰好一次进入 test")


def load_cohort_bundle(
    primary_labels_path: Path,
    extra_labels_path: Path,
    folds_path: Path,
    overlap_path: Path,
    radiomics_targets_path: Path,
    cache_root: Path,
) -> CohortBundle:
    labels = pd.read_csv(primary_labels_path)
    extra_labels = pd.read_csv(extra_labels_path)
    folds = pd.read_csv(folds_path)
    overlap = pd.read_csv(overlap_path)
    raw_radiomics = read_raw_radiomics(radiomics_targets_path)
    cache = cache_index(cache_root)

    if labels["patient_id"].duplicated().any() or len(labels) != 808:
        raise ValueError("primary labels 必须是 808 名唯一患者")
    if overlap["patient_id"].duplicated().any() or len(overlap) != 808:
        raise ValueError("radiomics overlap 必须是 808 名唯一患者")
    if set(overlap["patient_id"].astype(str)) != set(labels["patient_id"].astype(str)):
        raise ValueError("radiomics overlap 未恰好覆盖 primary cohort")
    validate_fold_manifest(folds, labels["patient_id"])
    fold_labels = folds.groupby("patient_id")["label_pcr"].nunique()
    if not fold_labels.eq(1).all():
        raise ValueError("manifest 中同一患者的 pCR 标签不一致")
    manifest_label = folds.drop_duplicates("patient_id").set_index("patient_id")["label_pcr"]
    label_lookup = labels.set_index("patient_id")["label_pcr"]
    if not manifest_label.astype(int).equals(label_lookup.loc[manifest_label.index].astype(int)):
        raise ValueError("fold manifest 与 primary labels 的 pCR 不一致")

    overlap_lookup = overlap.set_index("patient_id")["has_radiomics"].astype(bool)
    primary: list[PatientRecord] = []
    for row in labels.itertuples(index=False):
        patient_id = str(row.patient_id)
        if patient_id not in cache:
            raise FileNotFoundError(f"primary 患者缺 DCE8 cache: {patient_id}")
        has_radiomics = bool(overlap_lookup.loc[patient_id])
        if has_radiomics != (patient_id in raw_radiomics):
            raise ValueError(f"radiomics overlap 与 target 不一致: {patient_id}")
        primary.append(PatientRecord(patient_id, cache[patient_id], "ispy2", int(row.label_pcr), has_radiomics))

    extra: list[PatientRecord] = []
    extra_complete = extra_labels.loc[extra_labels["complete_4visits"].astype(bool)].copy()
    for row in extra_complete.itertuples(index=False):
        patient_id = str(row.patient_id)
        if patient_id not in cache:
            raise FileNotFoundError(f"I-SPY1 额外预训练患者缺 DCE8 cache: {patient_id}")
        extra.append(PatientRecord(patient_id, cache[patient_id], "ispy1", None, False))
    if len(extra) != 156:
        raise ValueError(f"期望 156 名 I-SPY1 额外预训练患者，实际 {len(extra)}")
    if {record.patient_id for record in primary} & {record.patient_id for record in extra}:
        raise ValueError("I-SPY1 与 I-SPY2 patient_id 发生交集")

    return CohortBundle(tuple(primary), tuple(extra), folds, raw_radiomics, labels)


def split_ids(bundle: CohortBundle, fold: int) -> dict[str, list[str]]:
    subset = bundle.folds.loc[bundle.folds["fold"] == fold]
    output = {
        split: subset.loc[subset["split"] == split, "patient_id"].astype(str).tolist()
        for split in ("train", "val", "test")
    }
    output["pretrain_train"] = output["train"] + [record.patient_id for record in bundle.extra_pretrain]
    return output


def patient_hash(patient_ids: Iterable[str]) -> str:
    content = "\n".join(sorted(str(value) for value in patient_ids)).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


class LongitudinalCacheDataset(Dataset):
    """输出 MRI 与可选 radiomics target；不输出 clinical/treatment/geometry。"""

    def __init__(
        self,
        records: Iterable[PatientRecord],
        transformed_radiomics: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None,
        image_channels: int = 8,
    ) -> None:
        self.records = tuple(records)
        self.transformed_radiomics = transformed_radiomics or {}
        if image_channels not in (7, 8):
            raise ValueError("image_channels 只能为 7（严格 image-only）或 8（ROI 辅助）")
        self.image_channels = int(image_channels)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        with np.load(record.cache_path) as archive:
            image = np.asarray(archive["x"][:, : self.image_channels], dtype=np.float32)
        if image.shape[:2] != (4, self.image_channels) or image.shape[2:] != (32, 96, 96):
            raise ValueError(f"DCE cache shape 错误: {record.cache_path} -> {image.shape}")
        target, mask = self.transformed_radiomics.get(
            record.patient_id,
            (
                np.zeros((3, len(FEATURE_NAMES)), dtype=np.float32),
                np.zeros((3, len(FEATURE_NAMES)), dtype=bool),
            ),
        )
        return {
            "patient_id": record.patient_id,
            "source": record.source,
            "image": torch.from_numpy(image),
            "radiomics_target": torch.from_numpy(np.asarray(target, dtype=np.float32)),
            "radiomics_mask": torch.from_numpy(np.asarray(mask, dtype=bool)),
            "has_radiomics": bool(np.asarray(mask).any()),
        }

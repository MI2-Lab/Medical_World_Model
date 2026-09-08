from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from rnc.data import CohortBundle, PatientRecord, load_cohort_bundle, split_ids


@dataclass(frozen=True)
class FTVTransform:
    median: float
    iqr: float

    @classmethod
    def fit(cls, raw: dict[str, np.ndarray], ids: Iterable[str]) -> "FTVTransform":
        vals = [raw[p][:, 0, :2].reshape(-1) for p in ids if p in raw]
        values = np.log1p(np.concatenate(vals))
        return cls(float(np.median(values)), max(float(np.subtract(*np.percentile(values, [75, 25]))), 1e-6))

    def apply(self, values: np.ndarray) -> np.ndarray:
        return ((np.log1p(values) - self.median) / self.iqr).astype(np.float32)


class ChangeDataset(Dataset):
    """Image plus FTV-only targets. No pCR, clinical, treatment, mask or geometry is exposed."""
    def __init__(self, records: Iterable[PatientRecord], raw: dict[str, np.ndarray], transform: FTVTransform) -> None:
        self.records = tuple(records)
        self.raw, self.transform = raw, transform
        self.patient_ids = tuple(record.patient_id for record in self.records)

    def __len__(self) -> int: return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        with np.load(record.cache_path) as archive:
            image = np.asarray(archive["x"][:, :7], dtype=np.float32)
        target = np.zeros(4, np.float32); valid = np.zeros(4, bool)
        if record.patient_id in self.raw:
            rows = self.raw[record.patient_id][:, 0, :]
            values = np.full(4, np.nan); values[0] = rows[0, 0]; values[1:] = rows[:, 1]
            valid = np.isfinite(values) & (values >= 0)
            target[valid] = self.transform.apply(values[valid])
        change = target[1:] - target[:-1]
        cmask = valid[1:] & valid[:-1]
        return {"patient_id": record.patient_id, "image": torch.from_numpy(image), "static": torch.from_numpy(target), "static_mask": torch.from_numpy(valid), "change": torch.from_numpy(change), "change_mask": torch.from_numpy(cmask)}


def bundle_from_config(config: dict) -> CohortBundle:
    data = config["data"]
    return load_cohort_bundle(Path(data["primary_labels"]), Path(data["extra_labels"]), Path(data["fold_manifest"]), Path(data["radiomics_overlap"]), Path(data["radiomics_targets"]), Path(data["cache_root"]))


def records(bundle: CohortBundle, ids: Iterable[str]) -> list[PatientRecord]:
    lookup = bundle.by_id
    return [lookup[str(patient)] for patient in ids]

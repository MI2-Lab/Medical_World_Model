from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .contracts import SUMMARY_SHAPE, V2_TARGET_DIR, private_token


def load_summary(path: str | Path, patient_id: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"patient_token", "summary", "source_cache_sha256", "contract_sha256"}:
            raise ValueError("DINO cache member contract drifted")
        if str(payload["patient_token"].item()) != private_token(patient_id):
            raise ValueError("DINO cache patient binding failed")
        summary = np.asarray(payload["summary"])
    if summary.shape != SUMMARY_SHAPE or summary.dtype != np.float16 or not np.isfinite(summary).all():
        raise ValueError("invalid DINO summary")
    return summary


class RadiomicsTargets:
    """Load only the V2 residual-radiomics target fields; never parse FTV."""

    def __init__(self, path: str | Path) -> None:
        with np.load(path, allow_pickle=False) as payload:
            required = {"patient_id", "radiomics", "radiomics_mask"}
            if not required.issubset(payload.files):
                raise ValueError("V2 fold target archive lacks radiomics fields")
            self.patient_ids = tuple(payload["patient_id"].astype(str).tolist())
            self.radiomics = np.asarray(payload["radiomics"], dtype=np.float32)
            self.mask = np.asarray(payload["radiomics_mask"], dtype=bool)
        if len(set(self.patient_ids)) != len(self.patient_ids):
            raise ValueError("duplicate target patient IDs")
        if self.radiomics.shape != (len(self.patient_ids), 4, 16) or self.mask.shape != (len(self.patient_ids), 4):
            raise ValueError("radiomics target shape contract failed")
        if self.mask[:, 3].any() or not np.isfinite(self.radiomics[self.mask]).all():
            raise ValueError("radiomics target finite/T3 contract failed")
        self.by_patient = {
            pid: (self.radiomics[i], self.mask[i]) for i, pid in enumerate(self.patient_ids)
        }


class SummaryDataset(Dataset[dict[str, object]]):
    def __init__(self, patient_ids: Iterable[str], summary_dir: str | Path, targets_path: str | Path) -> None:
        self.patient_ids = tuple(map(str, patient_ids))
        if len(set(self.patient_ids)) != len(self.patient_ids):
            raise ValueError("dataset patient IDs must be unique")
        self.summary_dir = Path(summary_dir)
        self.targets = RadiomicsTargets(targets_path).by_patient

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, index: int) -> dict[str, object]:
        patient_id = self.patient_ids[index]
        target, mask = self.targets.get(patient_id, (np.zeros((4, 16), np.float32), np.zeros(4, bool)))
        return {
            "patient_id": patient_id,
            "summary": torch.from_numpy(load_summary(self.summary_dir / f"{private_token(patient_id)}.private.npz", patient_id)),
            "radiomics": torch.from_numpy(np.asarray(target, dtype=np.float32)),
            "radiomics_mask": torch.from_numpy(np.asarray(mask, dtype=bool)),
        }


def fold_target_path(fold: int) -> Path:
    return V2_TARGET_DIR / f"fold_{int(fold)}_targets.private.npz"

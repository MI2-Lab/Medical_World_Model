"""Strict private-cache and fold-target readers for representation learning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .contracts import (
    COHORT_PATIENTS,
    FOLD_MANIFEST,
    FOLD_PATIENTS,
    LOCKED_HASHES,
    STATE_SHAPE,
    SUMMARY_SHAPE,
    TECHNICAL_ELIGIBILITY,
    TRAIN_ONLY_PATIENTS,
    file_sha256,
    private_patient_token,
    verify_locked_file,
)


from .cache_io import CacheEntry, load_c1b_manifest


def load_fold_frame(path: str | Path = FOLD_MANIFEST, *, verify_hash: bool = True) -> pd.DataFrame:
    source = Path(path)
    if verify_hash:
        verify_locked_file(source, LOCKED_HASHES["fold_manifest"], "seed-2026 folds")
    frame = pd.read_csv(source, usecols=["patient_id", "fold", "split"], dtype={"patient_id": str})
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    frame["split"] = frame["split"].replace({"validation": "val"}).astype(str)
    if len(frame) != FOLD_PATIENTS * 5 or frame.duplicated(["patient_id", "fold"]).any():
        raise ValueError("fold manifest must contain 808 patients in each of five folds")
    if set(frame["fold"]) != set(range(5)) or not set(frame["split"]).issubset({"train", "val", "test"}):
        raise ValueError("fold/split contract drifted")
    if not frame.groupby("fold").size().eq(FOLD_PATIENTS).all():
        raise ValueError("every fold must contain exactly 808 patients")
    return frame


def split_patient_ids(
    fold: int,
    train_only_ids: Iterable[str],
    frame: pd.DataFrame | None = None,
) -> dict[str, tuple[str, ...]]:
    frame = load_fold_frame() if frame is None else frame
    current = frame.loc[frame["fold"].eq(int(fold))]
    result = {
        split: tuple(sorted(current.loc[current["split"].eq(split), "patient_id"].astype(str)))
        for split in ("train", "val", "test")
    }
    extras = tuple(sorted(map(str, train_only_ids)))
    if len(extras) != TRAIN_ONLY_PATIENTS or set(extras).intersection(set(current["patient_id"])):
        raise ValueError("train-only population must be 139 disjoint patients")
    result["train"] = tuple(sorted((*result["train"], *extras)))
    return result


def load_train_only_ids(
    path: str | Path,
    technical_eligibility_path: str | Path = TECHNICAL_ELIGIBILITY,
) -> tuple[str, ...]:
    source = Path(path)
    frame = pd.read_csv(source, dtype={"patient_id": str})
    if "eligible" in frame:
        frame = frame.loc[frame["eligible"].astype(bool)]
    technical_source = Path(technical_eligibility_path)
    verify_locked_file(
        technical_source, LOCKED_HASHES["technical_eligibility"], "technical eligibility"
    )
    technical = pd.read_csv(
        technical_source,
        usecols=["patient_id", "cohort", "eligible"],
        dtype={"patient_id": str, "cohort": str},
    )
    technical_ids = set(
        technical.loc[
            technical["cohort"].eq("I-SPY1") & technical["eligible"].astype(bool),
            "patient_id",
        ].astype(str)
    )
    values = tuple(sorted(set(frame["patient_id"].astype(str)).intersection(technical_ids)))
    if len(values) != TRAIN_ONLY_PATIENTS:
        raise ValueError(f"expected 139 I-SPY1 train-only patients, got {len(values)}")
    return values


def load_summary(path: str | Path, expected_patient_id: str | None = None) -> np.ndarray:
    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        if set(payload.files) != {"patient_token", "summary", "source_cache_sha256", "contract_sha256"}:
            raise ValueError(f"DINO cache member contract drifted: {source}")
        summary = np.asarray(payload["summary"])
        token = str(payload["patient_token"].item())
    if summary.shape != SUMMARY_SHAPE or summary.dtype != np.float16 or not np.isfinite(summary).all():
        raise ValueError(f"invalid DINO summary cache: {source}")
    if expected_patient_id is not None and token != private_patient_token(expected_patient_id):
        raise ValueError("DINO cache patient binding failed")
    return summary


@dataclass(frozen=True)
class FoldTargets:
    patient_ids: tuple[str, ...]
    ftv: np.ndarray
    ftv_mask: np.ndarray
    radiomics: np.ndarray
    radiomics_mask: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> "FoldTargets":
        with np.load(path, allow_pickle=False) as payload:
            required = {"patient_id", "ftv", "ftv_mask", "radiomics", "radiomics_mask"}
            if not required.issubset(payload.files):
                raise ValueError("fold target archive is incomplete")
            patient_ids = tuple(payload["patient_id"].astype(str).tolist())
            ftv = np.asarray(payload["ftv"], dtype=np.float32)
            ftv_mask = np.asarray(payload["ftv_mask"], dtype=bool)
            radiomics = np.asarray(payload["radiomics"], dtype=np.float32)
            radiomics_mask = np.asarray(payload["radiomics_mask"], dtype=bool)
        n = len(patient_ids)
        if len(set(patient_ids)) != n or ftv.shape != (n, 4) or ftv_mask.shape != (n, 4):
            raise ValueError("fold FTV target shape/identity contract failed")
        if radiomics.shape != (n, 4, 16) or radiomics_mask.shape != (n, 4):
            raise ValueError("fold radiomics target shape contract failed")
        if not np.isfinite(ftv[ftv_mask]).all() or not np.isfinite(radiomics[radiomics_mask]).all():
            raise ValueError("valid fold targets contain non-finite values")
        return cls(patient_ids, ftv, ftv_mask, radiomics, radiomics_mask)

    def mapping(self) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        return {
            patient_id: (self.ftv[i], self.ftv_mask[i], self.radiomics[i], self.radiomics_mask[i])
            for i, patient_id in enumerate(self.patient_ids)
        }


class SummaryDataset(Dataset[dict[str, object]]):
    """Model-visible input is only a frozen DINO summary tensor."""

    def __init__(
        self,
        patient_ids: Iterable[str],
        summary_dir: str | Path,
        targets: FoldTargets | None = None,
    ) -> None:
        self.patient_ids = tuple(map(str, patient_ids))
        if len(set(self.patient_ids)) != len(self.patient_ids):
            raise ValueError("dataset patient IDs must be unique")
        self.summary_dir = Path(summary_dir)
        self.targets = {} if targets is None else targets.mapping()

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, index: int) -> dict[str, object]:
        patient_id = self.patient_ids[index]
        summary = load_summary(
            self.summary_dir / f"{private_patient_token(patient_id)}.private.npz",
            patient_id,
        )
        target = self.targets.get(patient_id)
        if target is None:
            ftv = np.zeros(4, dtype=np.float32)
            ftv_mask = np.zeros(4, dtype=bool)
            radiomics = np.zeros((4, 16), dtype=np.float32)
            radiomics_mask = np.zeros(4, dtype=bool)
        else:
            ftv, ftv_mask, radiomics, radiomics_mask = target
        return {
            "patient_id": patient_id,
            "summary": torch.from_numpy(summary),
            "ftv": torch.from_numpy(np.asarray(ftv, dtype=np.float32)),
            "ftv_mask": torch.from_numpy(np.asarray(ftv_mask, dtype=bool)),
            "radiomics": torch.from_numpy(np.asarray(radiomics, dtype=np.float32)),
            "radiomics_mask": torch.from_numpy(np.asarray(radiomics_mask, dtype=bool)),
        }


def validate_state_archive(path: str | Path, expected_patients: int = FOLD_PATIENTS) -> tuple[tuple[str, ...], np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        if not {"patient_id", "state"}.issubset(payload.files):
            raise ValueError("state archive is incomplete")
        patient_ids = tuple(payload["patient_id"].astype(str).tolist())
        state = np.asarray(payload["state"], dtype=np.float32)
    if len(patient_ids) != expected_patients or len(set(patient_ids)) != expected_patients:
        raise ValueError("state archive patient coverage failed")
    if state.shape != (expected_patients, *STATE_SHAPE) or not np.isfinite(state).all():
        raise ValueError("state archive tensor contract failed")
    return patient_ids, state


__all__ = [
    "CacheEntry", "FoldTargets", "SummaryDataset", "load_c1b_manifest", "load_fold_frame",
    "load_summary", "load_train_only_ids", "split_patient_ids", "validate_state_archive"
]

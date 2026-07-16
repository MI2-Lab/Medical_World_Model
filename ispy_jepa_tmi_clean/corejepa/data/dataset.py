from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .condition import ConditionEncoder
from .imaging import build_patient_tensor
from .records import PatientRecord


class LongitudinalDCEDataset(Dataset):
    """Load cached four-visit DCE8 trajectories.

    One item contains:
        ``image: float32 [4,8,Z,Y,X]``
        ``geometry: float32 [4,9]``
        ``condition: float32 [3,Cc]``
        ``routing_target: int64 scalar``
        ``record_index: int64 scalar``
    """

    def __init__(
        self,
        records: list[PatientRecord],
        condition_encoder: ConditionEncoder,
        cache_dir: str | Path,
        crop_size: tuple[int, int, int],
        phase_policy: str,
        phase_metadata: dict[str, dict[str, Any]],
        automatic_roi_fallback: bool,
        minimum_roi_capture: float,
        legacy_empty_ftv_full_field: bool,
        build_missing: bool = True,
    ) -> None:
        self.records = records
        self.condition_encoder = condition_encoder
        self.cache_dir = Path(cache_dir)
        self.crop_size = crop_size
        self.phase_policy = phase_policy
        self.phase_metadata = phase_metadata
        self.automatic_roi_fallback = automatic_roi_fallback
        self.minimum_roi_capture = minimum_roi_capture
        self.legacy_empty_ftv_full_field = legacy_empty_ftv_full_field
        self.build_missing = build_missing

    def __len__(self) -> int:
        return len(self.records)

    def cache_path(self, record: PatientRecord) -> Path:
        return self.cache_dir / f"{record.patient_id}.npz"

    def _ensure_cache(self, record: PatientRecord) -> Path:
        path = self.cache_path(record)
        if path.exists():
            return path
        if not self.build_missing:
            raise FileNotFoundError(f"Missing tensor cache: {path}")
        return build_patient_tensor(
            record=record,
            cache_dir=self.cache_dir,
            crop_size_zyx=self.crop_size,
            phase_policy=self.phase_policy,
            phase_metadata=self.phase_metadata.get(record.patient_id),
            automatic_roi_fallback=self.automatic_roi_fallback,
            minimum_roi_capture=self.minimum_roi_capture,
            legacy_empty_ftv_full_field=self.legacy_empty_ftv_full_field,
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        with np.load(self._ensure_cache(record), allow_pickle=False) as cache:
            image = cache["image"].astype(np.float32)
            geometry = cache["geometry"].astype(np.float32)
        return {
            "patient_id": record.patient_id,
            "record_index": torch.tensor(index, dtype=torch.long),
            "image": torch.from_numpy(image),
            "geometry": torch.from_numpy(geometry),
            "condition": torch.from_numpy(self.condition_encoder.encode(record)),
            "routing_target": torch.tensor(self.condition_encoder.routing_target(record), dtype=torch.long),
        }


class PretrainingDataset(Dataset):
    """Attach pCR-free guidance targets to the image dataset.

    This wrapper intentionally exposes no pCR field.
    """

    def __init__(
        self,
        base: LongitudinalDCEDataset,
        response_vector: np.ndarray,
        response_score: np.ndarray,
    ) -> None:
        if response_vector.shape[:2] != (len(base), 3) or response_vector.shape[-1] != 18:
            raise ValueError(f"response_vector must be [N,3,18], got {response_vector.shape}")
        if response_score.shape != (len(base), 3, 1):
            raise ValueError(f"response_score must be [N,3,1], got {response_score.shape}")
        self.base = base
        self.response_vector = response_vector.astype(np.float32)
        self.response_score = response_score.astype(np.float32)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        item = self.base[index]
        item["response_vector"] = torch.from_numpy(self.response_vector[index])
        item["response_score"] = torch.from_numpy(self.response_score[index])
        return item

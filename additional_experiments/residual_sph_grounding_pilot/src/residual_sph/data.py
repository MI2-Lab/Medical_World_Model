"""Loss-side SPH augmentation for the sealed patient-indexed Stage-B dataset."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .contracts import arm_spec, assert_representation_schema


class StaticSPHDataset(Dataset):
    """Wrap a Stage-B image/FTV dataset without exposing any outcome sidecars."""

    def __init__(
        self,
        base_dataset: Dataset,
        sph_targets: Mapping[str, tuple[np.ndarray, np.ndarray]],
        experimental_arm: str,
    ) -> None:
        self.base_dataset = base_dataset
        self.experimental_arm = arm_spec(experimental_arm).name
        self.patient_ids = tuple(str(value) for value in getattr(base_dataset, "patient_ids"))
        self.transformed_ftv = getattr(base_dataset, "transformed_ftv")
        self.sph_targets = {
            str(patient_id): (
                np.asarray(pair[0], dtype=np.float32),
                np.asarray(pair[1], dtype=bool),
            )
            for patient_id, pair in sph_targets.items()
        }
        assert_representation_schema(("image", "ftv_target", "ftv_mask", "sph_target", "sph_mask"))
        unknown = sorted(set(self.sph_targets).difference(self.patient_ids))
        if unknown:
            raise ValueError("SPH mapping contains patients outside the wrapped split")
        for target, mask in self.sph_targets.values():
            if target.shape != (4,) or mask.shape != (4,):
                raise ValueError("every SPH target and mask must have four visits")
            if not np.isfinite(target[mask]).all():
                raise ValueError("valid SPH targets must be finite")
        if not arm_spec(self.experimental_arm).has_sph_head and any(mask.any() for _, mask in self.sph_targets.values()):
            raise ValueError("S0 cannot receive SPH loss-side targets")

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = dict(self.base_dataset[index])
        if set(record) != {"patient_id", "image", "ftv_target", "ftv_mask"}:
            raise PermissionError("base Stage-B dataset schema drifted")
        patient_id = str(record["patient_id"])
        target, mask = self.sph_targets.get(
            patient_id,
            (np.zeros(4, dtype=np.float32), np.zeros(4, dtype=bool)),
        )
        record["sph_target"] = torch.from_numpy(np.asarray(target, dtype=np.float32).copy())
        record["sph_mask"] = torch.from_numpy(np.asarray(mask, dtype=bool).copy())
        assert_representation_schema(record)
        return record


def target_mapping(
    patient_ids: tuple[str, ...] | list[str],
    target: np.ndarray,
    mask: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    ids = tuple(str(value) for value in patient_ids)
    values = np.asarray(target, dtype=np.float32)
    valid = np.asarray(mask, dtype=bool)
    if values.shape != (len(ids), 4) or valid.shape != values.shape:
        raise ValueError("target arrays must be [patients,4]")
    return {
        patient_id: (values[index].copy(), valid[index].copy())
        for index, patient_id in enumerate(ids)
    }


__all__ = ["StaticSPHDataset", "target_mapping"]

"""pCR-free clinical conditions attached to the sealed C1B-H dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .contracts import (
    FORBIDDEN_TRAINING_COLUMN_TOKENS,
    PCR_LABEL_ACCESS,
    TRAINING_PROFILE_COLUMNS,
    canonical_sha256,
    file_sha256,
)


TEMPORAL_FEATURES = (
    "forecast_T1",
    "forecast_T2",
    "forecast_T3",
    "exposed_through_T0",
    "exposed_through_T1",
    "exposed_through_T2",
)


@dataclass(frozen=True)
class TrainingProfile:
    patient_id: str
    hr: int
    her2: int
    mp: int
    arm: str


@dataclass(frozen=True)
class ConditionSpec:
    arm_vocab: Mapping[str, int]
    feature_names: tuple[str, ...]

    @property
    def dim(self) -> int:
        return len(self.feature_names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_vocab": dict(self.arm_vocab),
            "feature_names": list(self.feature_names),
            "dim": self.dim,
        }


def _binary(series: pd.Series, label: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="raise")
    if numeric.isna().any() or not numeric.isin((0, 1)).all():
        raise ValueError(f"{label} must be complete binary 0/1")
    return numeric.astype(np.int64)


def load_training_profiles(
    path: str | Path,
    expected_sha256: str,
    *,
    expected_patient_ids: Iterable[str] | None = None,
) -> tuple[dict[str, TrainingProfile], ConditionSpec, dict[str, Any]]:
    """Load only an already-projected five-column private manifest.

    Formal training never opens the source clinical workbooks/CSVs.  This
    function rejects every extra column, including outcome-like columns.
    """

    if PCR_LABEL_ACCESS != "FORBIDDEN":
        raise PermissionError("representation-training pCR firewall is not active")
    source = Path(path).expanduser().resolve()
    if not source.name.endswith(".private.csv"):
        raise ValueError("training profiles must be an owner-private projected CSV")
    actual = file_sha256(source)
    if actual != str(expected_sha256).lower():
        raise ValueError("training profile manifest SHA-256 mismatch")
    header = tuple(pd.read_csv(source, nrows=0).columns)
    if header != TRAINING_PROFILE_COLUMNS:
        raise PermissionError(
            "training profile manifest must contain exactly the pCR-free allowlist"
        )
    lowered = tuple(column.casefold() for column in header)
    if any(token in column for token in FORBIDDEN_TRAINING_COLUMN_TOKENS for column in lowered):
        raise PermissionError("outcome-like field entered training profile manifest")
    frame = pd.read_csv(
        source,
        usecols=list(TRAINING_PROFILE_COLUMNS),
        dtype={"patient_id": "string", "arm": "string"},
    )
    frame["patient_id"] = frame["patient_id"].str.strip()
    frame["arm"] = frame["arm"].str.strip()
    if frame["patient_id"].isna().any() or frame["arm"].isna().any():
        raise ValueError("training patient IDs and treatment arms must be complete")
    if frame.empty or frame["patient_id"].duplicated().any():
        raise ValueError("training profiles must be nonempty and patient-unique")
    for column in ("label_hr", "label_her2", "label_mp"):
        frame[column] = _binary(frame[column], column)
    if frame["patient_id"].eq("").any() or frame["arm"].eq("").any():
        raise ValueError("training patient IDs and treatment arms must be nonempty")
    if expected_patient_ids is not None:
        expected = set(str(value) for value in expected_patient_ids)
        observed = set(frame["patient_id"])
        if observed != expected:
            raise ValueError(
                "pCR-free training profiles do not exactly cover the model-input cohort"
            )
    arms = tuple(sorted(frame["arm"].unique(), key=str.casefold))
    vocab = {arm: index for index, arm in enumerate(arms)}
    names = TEMPORAL_FEATURES + tuple(f"arm={arm}" for arm in arms) + (
        "HR",
        "HER2",
        "MammaPrint",
    )
    spec = ConditionSpec(vocab, names)
    profiles = {
        row.patient_id: TrainingProfile(
            patient_id=row.patient_id,
            hr=int(row.label_hr),
            her2=int(row.label_her2),
            mp=int(row.label_mp),
            arm=str(row.arm),
        )
        for row in frame.itertuples(index=False)
    }
    provenance = {
        "profile_manifest_sha256": actual,
        "profile_patient_count": len(profiles),
        "profile_patient_order_sha256": canonical_sha256(tuple(frame["patient_id"])),
        "profile_arm_count": len(arms),
    }
    return profiles, spec, provenance


def encode_condition(profile: TrainingProfile, spec: ConditionSpec) -> np.ndarray:
    output = np.zeros((3, spec.dim), dtype=np.float32)
    arm_offset = len(TEMPORAL_FEATURES)
    clinical_offset = arm_offset + len(spec.arm_vocab)
    if profile.arm not in spec.arm_vocab:
        raise ValueError(f"unknown treatment arm: {profile.arm}")
    for transition in range(3):
        output[transition, transition] = 1.0
        output[transition, 3 : 4 + transition] = 1.0
        output[transition, arm_offset + spec.arm_vocab[profile.arm]] = 1.0
        output[transition, clinical_offset:] = (
            float(profile.hr),
            float(profile.her2),
            float(profile.mp),
        )
    return output


class ProfiledStageBDataset(Dataset):
    """Add condition/adversary targets without exposing any outcome field."""

    def __init__(
        self,
        base: Dataset,
        profiles: Mapping[str, TrainingProfile],
        condition_spec: ConditionSpec,
    ) -> None:
        self.base = base
        self.profiles = profiles
        self.condition_spec = condition_spec
        patient_ids = tuple(str(value) for value in getattr(base, "patient_ids"))
        if missing := sorted(set(patient_ids).difference(profiles)):
            raise KeyError(f"training profiles miss model patients: {missing[:5]}")
        self.patient_ids = patient_ids
        self.transformed_ftv = getattr(base, "transformed_ftv", {})

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.base[index])
        if set(item) != {"patient_id", "image", "ftv_target", "ftv_mask"}:
            raise PermissionError("sealed Stage-B item schema unexpectedly changed")
        patient_id = str(item["patient_id"])
        profile = self.profiles[patient_id]
        item["condition"] = torch.from_numpy(
            encode_condition(profile, self.condition_spec)
        )
        item["clinical_target"] = torch.tensor(
            (profile.hr, profile.her2), dtype=torch.long
        )
        return item


__all__ = [
    "ConditionSpec",
    "ProfiledStageBDataset",
    "TEMPORAL_FEATURES",
    "TrainingProfile",
    "encode_condition",
    "load_training_profiles",
]

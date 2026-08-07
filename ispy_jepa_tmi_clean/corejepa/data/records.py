from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class PatientRecord:
    """One complete longitudinal trajectory and its baseline context."""

    patient_id: str
    cohort: str
    arm: str
    hr: int
    her2: int
    mp: int
    age: float
    manifest_path: Path
    pcr: int | None = None
    longest_diameter: tuple[float, float, float, float] | None = None


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_records(
    root: str | Path,
    labels_csv: str | Path,
    cohort: str,
    require_pcr: bool = False,
) -> list[PatientRecord]:
    """Load patients with four visits and an existing preprocessing manifest."""

    root = Path(root)
    frame = pd.read_csv(labels_csv)
    if "complete_4visits" in frame:
        frame = frame[frame["complete_4visits"].map(_truthy)]
    records: list[PatientRecord] = []
    for row in frame.sort_values("patient_id").itertuples(index=False):
        patient_id = str(row.patient_id)
        manifest = root / patient_id / "manifest.json"
        if not manifest.exists():
            continue
        raw_pcr = getattr(row, "label_pcr", np.nan)
        pcr = None if pd.isna(raw_pcr) else int(raw_pcr)
        if require_pcr and pcr is None:
            continue
        diameter_columns = ("mri_ld_baseline", "mri_ld_1_3dac", "mri_ld_interreg", "mri_ld_presurg")
        diameter_values = tuple(float(getattr(row, name, np.nan)) for name in diameter_columns)
        longest_diameter = diameter_values if any(np.isfinite(value) and value > 0 for value in diameter_values) else None
        records.append(
            PatientRecord(
                patient_id=patient_id,
                cohort=cohort,
                arm=str(getattr(row, "arm", "ISPY1_NACT" if cohort.lower() == "ispy1" else "Unknown")),
                hr=int(getattr(row, "label_hr", 0)),
                her2=int(getattr(row, "label_her2", 0)),
                mp=int(getattr(row, "label_mp", 0)),
                age=float(getattr(row, "age_at_screening", np.nan)),
                manifest_path=manifest,
                pcr=pcr,
                longest_diameter=longest_diameter,
            )
        )
    return records


def treatment_family(record: PatientRecord) -> str:
    """Map an exact arm to one of the six routing strata used by paper v1."""

    if record.cohort.lower() == "ispy1":
        return "ispy1_nact"
    arm = record.arm.lower()
    if "pembrolizumab" in arm:
        return "io"
    if "carboplatin" in arm or "abt 888" in arm:
        return "platinum_parp"
    if any(name in arm for name in ("trastuzumab", "pertuzumab", "t-dm1", "neratinib")):
        return "her2_targeted"
    if arm.strip() == "paclitaxel":
        return "taxane"
    return "targeted_other"


def stratified_split(records: list[PatientRecord], seed: int) -> tuple[list[int], list[int], list[int]]:
    """Return the locked 70/15/15 patient split used for I-SPY2."""

    if any(record.pcr is None for record in records):
        raise ValueError("A stratified pCR split requires labels for every primary record")
    indices = np.arange(len(records))
    labels = np.asarray([int(record.pcr) for record in records], dtype=np.int64)
    train, temporary = train_test_split(indices, test_size=0.30, random_state=seed, stratify=labels)
    validation, test = train_test_split(
        temporary,
        test_size=0.50,
        random_state=seed,
        stratify=labels[temporary],
    )
    return train.tolist(), validation.tolist(), test.tolist()

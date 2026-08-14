"""Fold-safe clinical and FTV adapters for downstream logistic fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


CLINICAL_FIELDS = (
    "label_hr",
    "label_her2",
    "label_mp",
    "age_at_screening",
    "race_simple",
    "menopausal_status_simple",
    "ethnicity",
    "arm",
)
NUMERIC_FIELDS = frozenset(
    {"label_hr", "label_her2", "label_mp", "age_at_screening"}
)
MISSING_TOKEN = "__MISSING__"
VISITS = ("T0", "T1", "T2", "T3")
TRANSITIONS = ("T0→T1", "T1→T2", "T2→T3")


class TrainOnlyClinicalEncoder:
    """Median-impute numeric fields and one-hot train-observed categories."""

    def __init__(self, fields: Sequence[str] = CLINICAL_FIELDS) -> None:
        self.fields = tuple(str(value) for value in fields)
        if not self.fields or len(self.fields) != len(set(self.fields)):
            raise ValueError("clinical fields must be nonempty and unique")
        unknown = sorted(set(self.fields) - set(CLINICAL_FIELDS))
        if unknown:
            raise ValueError(f"unknown clinical fields: {unknown}")
        self.medians: dict[str, float] = {}
        self.categories: dict[str, tuple[str, ...]] = {}
        self.feature_names: tuple[str, ...] = ()
        self.fitted = False

    @staticmethod
    def _categories(series: pd.Series) -> np.ndarray:
        values: list[str] = []
        for value in series.to_numpy(dtype=object):
            if pd.isna(value) or (isinstance(value, str) and not value.strip()):
                values.append(MISSING_TOKEN)
            else:
                text = str(value)
                if text != text.strip():
                    raise ValueError("categorical clinical value contains padded whitespace")
                values.append(text)
        return np.asarray(values, dtype=object)

    def fit(self, train: pd.DataFrame) -> "TrainOnlyClinicalEncoder":
        if self.fitted:
            raise ValueError("clinical encoder may only be fitted once")
        missing = [field for field in self.fields if field not in train]
        if train.empty or missing:
            raise ValueError(
                f"invalid clinical training frame; missing={missing if not train.empty else self.fields}"
            )
        names: list[str] = []
        for field in self.fields:
            if field in NUMERIC_FIELDS:
                values = pd.to_numeric(train[field], errors="raise").to_numpy(float)
                finite = values[np.isfinite(values)]
                if not finite.size:
                    raise ValueError(f"numeric clinical field {field} has no finite train values")
                self.medians[field] = float(np.median(finite))
                names.append(field)
            else:
                values = self._categories(train[field])
                levels = tuple(sorted(set(values.tolist()) | {MISSING_TOKEN}))
                self.categories[field] = levels
                names.extend(f"{field}={level}" for level in levels)
        self.feature_names = tuple(names)
        self.fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise ValueError("clinical encoder is not fitted")
        blocks: list[np.ndarray] = []
        for field in self.fields:
            if field not in frame:
                raise ValueError(f"clinical frame misses {field}")
            if field in NUMERIC_FIELDS:
                values = pd.to_numeric(frame[field], errors="raise").to_numpy(float)
                if np.isinf(values).any():
                    raise ValueError("clinical numeric field contains infinity")
                blocks.append(np.where(np.isnan(values), self.medians[field], values)[:, None])
            else:
                values = self._categories(frame[field])
                levels = self.categories[field]
                lookup = {level: index for index, level in enumerate(levels)}
                block = np.zeros((len(frame), len(levels)), dtype=np.float64)
                for row, value in enumerate(values):
                    column = lookup.get(str(value))
                    if column is not None:
                        block[row, column] = 1.0
                blocks.append(block)
        output = np.concatenate(blocks, axis=1)
        if output.shape != (len(frame), len(self.feature_names)) or not np.isfinite(output).all():
            raise RuntimeError("clinical encoder produced an invalid matrix")
        return output


def load_clinical_table(path: str, patient_ids: Sequence[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"patient_id", "label_pcr", "label_hr", "label_her2", *CLINICAL_FIELDS}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"clinical table misses columns: {missing}")
    frame["patient_id"] = frame["patient_id"].astype(str)
    if frame["patient_id"].duplicated().any() or set(frame["patient_id"]) != set(patient_ids):
        raise ValueError("clinical table does not exactly equal full_808")
    for field in ("label_pcr", "label_hr", "label_her2", "label_mp"):
        numeric = pd.to_numeric(frame[field], errors="raise")
        if numeric.isna().any() or not numeric.isin((0, 1)).all():
            raise ValueError(f"clinical {field} must be complete binary")
        frame[field] = numeric.astype(np.int64)
    return frame.set_index("patient_id", verify_integrity=True).loc[list(patient_ids)].reset_index()


def load_ftv_wide(path: str, allowed_patient_ids: Sequence[str]) -> pd.DataFrame:
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
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"FTV table misses columns: {missing}")
    frame["patient_id"] = frame["patient_id"].astype(str)
    if not set(frame["patient_id"]).issubset(set(allowed_patient_ids)):
        raise ValueError("FTV table contains patients outside full_808")
    rows: list[dict[str, Any]] = []
    for patient_id, group in frame.groupby("patient_id", sort=False):
        if len(group) != 3 or set(group["transition"].astype(str)) != set(TRANSITIONS):
            raise ValueError("every FTV patient must contain three adjacent transitions")
        by = group.set_index("transition", verify_integrity=True).loc[list(TRANSITIONS)]
        valid = by["ftv_valid"].astype(str).str.lower().isin(("true", "1"))
        if not valid.all():
            raise ValueError("FTV-complete table contains an invalid transition")
        values = np.asarray(
            [
                float(by.iloc[0]["ftv_start"]),
                float(by.iloc[0]["ftv_end"]),
                float(by.iloc[1]["ftv_end"]),
                float(by.iloc[2]["ftv_end"]),
            ],
            dtype=np.float64,
        )
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("FTV values must be finite and nonnegative")
        if not np.isclose(values[1], float(by.iloc[1]["ftv_start"]), atol=1e-10, rtol=0):
            raise ValueError("FTV T1 endpoints disagree")
        if not np.isclose(values[2], float(by.iloc[2]["ftv_start"]), atol=1e-10, rtol=0):
            raise ValueError("FTV T2 endpoints disagree")
        rows.append({"patient_id": patient_id, **{f"FTV_{visit}": value for visit, value in zip(VISITS, values)}})
    output = pd.DataFrame(rows)
    if len(output) != 375 or output["patient_id"].duplicated().any():
        raise ValueError("FTV complete estimand must contain exactly 375 unique patients")
    return output


def prefix_matrix(state: np.ndarray, timing: str) -> np.ndarray:
    values = np.asarray(state)
    mapping = {"T0": 1, "T0_T1": 2, "T0_T2": 3, "T0_T3": 4}
    if timing not in mapping or values.ndim != 3 or values.shape[1] != 4:
        raise ValueError("state must be [N,4,D] and timing registered")
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError("state must be finite numeric")
    end = mapping[timing]
    return values[:, :end].reshape(len(values), end * values.shape[2]).astype(np.float64, copy=True)


def ftv_prefix_matrix(ftv: pd.DataFrame, timing: str) -> np.ndarray:
    mapping = {"T0": 1, "T0_T1": 2, "T0_T2": 3, "T0_T3": 4}
    if timing not in mapping:
        raise ValueError("unregistered timing")
    columns = [f"FTV_{visit}" for visit in VISITS[: mapping[timing]]]
    values = ftv.loc[:, columns].to_numpy(np.float64)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("FTV prefix is invalid")
    return np.log1p(values)


__all__ = [
    "CLINICAL_FIELDS",
    "TrainOnlyClinicalEncoder",
    "ftv_prefix_matrix",
    "load_clinical_table",
    "load_ftv_wide",
    "prefix_matrix",
]

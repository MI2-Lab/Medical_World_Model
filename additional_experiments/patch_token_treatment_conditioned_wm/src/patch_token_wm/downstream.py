"""Fold-isolated downstream feature and natural-target construction.

Nothing in this module participates in world-model training.  The clinical
preprocessor reproduces the prior primary ``C2_full_with_treatment`` contract,
and every fitted statistic (numeric median, categorical vocabulary, and feature
scaling) comes from one outer training fold.  Validation and test data are
transform-only.

pCR is intentionally absent from all general construction APIs.  It can be
materialized only through :func:`load_pcr_labels_downstream_only`, whose caller
must explicitly attest the frozen-downstream-probe purpose.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


VISITS = ("T0", "T1", "T2", "T3")
TRANSITIONS: Mapping[str, tuple[int, int]] = {
    "T0_to_T1": (0, 1),
    "T1_to_T2": (1, 2),
    "T2_to_T3": (2, 3),
}

C2_FULL_WITH_TREATMENT_FIELDS = (
    "label_hr",
    "label_her2",
    "label_mp",
    "age_at_screening",
    "race_simple",
    "menopausal_status_simple",
    "ethnicity",
    "arm",
)
NUMERIC_CLINICAL_FIELDS = (
    "label_hr",
    "label_her2",
    "label_mp",
    "age_at_screening",
)
CATEGORICAL_CLINICAL_FIELDS = (
    "race_simple",
    "menopausal_status_simple",
    "ethnicity",
    "arm",
)
MISSING_CATEGORY = "__MISSING__"
PCR_DOWNSTREAM_PURPOSE = "frozen_downstream_probe"


class DownstreamContractError(ValueError):
    """Raised when downstream arrays violate the locked construction contract."""


def _timing_index(timing: str | int) -> int:
    if isinstance(timing, (bool, np.bool_)):
        raise DownstreamContractError("timing must be T0..T3 or integer 0..3")
    if isinstance(timing, (int, np.integer)):
        index = int(timing)
    elif isinstance(timing, str) and timing.upper() in VISITS:
        index = VISITS.index(timing.upper())
    else:
        raise DownstreamContractError("timing must be T0..T3 or integer 0..3")
    if index not in range(len(VISITS)):
        raise DownstreamContractError("timing must be T0..T3 or integer 0..3")
    return index


def _transition_indices(transition: str | int) -> tuple[str, int, int]:
    if isinstance(transition, (bool, np.bool_)):
        raise DownstreamContractError(
            "transition must be T0_to_T1, T1_to_T2, T2_to_T3, or integer 0..2"
        )
    if isinstance(transition, (int, np.integer)):
        index = int(transition)
        if index not in range(3):
            raise DownstreamContractError("transition integer must be in 0..2")
        name = tuple(TRANSITIONS)[index]
        start, end = TRANSITIONS[name]
        return name, start, end
    if not isinstance(transition, str):
        raise DownstreamContractError("transition has an unsupported type")
    normalized = (
        transition.strip()
        .replace("→", "_to_")
        .replace("–", "_to_")
        .replace("-", "_to_")
    )
    # Avoid accepting accidental repeated separators after normalization.
    normalized = normalized.replace("__", "_")
    if normalized not in TRANSITIONS:
        raise DownstreamContractError(
            "transition must be T0_to_T1, T1_to_T2, or T2_to_T3"
        )
    start, end = TRANSITIONS[normalized]
    return normalized, start, end


def _unique_patient_ids(
    patient_ids: Sequence[Any], *, expected_rows: int | None = None, name: str
) -> tuple[str, ...]:
    if isinstance(patient_ids, (str, bytes)):
        raise DownstreamContractError(f"{name} must be a sequence of patient IDs")
    raw_ids = tuple(patient_ids)
    if any(pd.isna(value) for value in raw_ids):
        raise DownstreamContractError(f"{name} may not contain missing values")
    ids = tuple(str(value) for value in raw_ids)
    if expected_rows is not None and len(ids) != expected_rows:
        raise DownstreamContractError(
            f"{name} contains {len(ids)} IDs; expected {expected_rows}"
        )
    if not ids or any(not value for value in ids) or len(set(ids)) != len(ids):
        raise DownstreamContractError(f"{name} must be non-empty and unique")
    return ids


def _state_tensor(values: Any, *, name: str = "MRI states") -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind not in "fiu" or raw.ndim != 3 or raw.shape[1] != 4:
        raise DownstreamContractError(f"{name} must be numeric [N,4,D]")
    if raw.shape[0] == 0 or raw.shape[2] == 0:
        raise DownstreamContractError(
            f"{name} must have non-empty patient/feature axes"
        )
    states = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(states).all():
        raise DownstreamContractError(f"{name} contains NaN or infinity")
    return states


def _ftv_matrix(values: Any, *, allow_missing: bool, name: str = "FTV") -> np.ndarray:
    if isinstance(values, pd.DataFrame):
        columns = tuple(f"FTV_{visit}" for visit in VISITS)
        if missing := [column for column in columns if column not in values.columns]:
            raise DownstreamContractError(f"{name} table misses columns: {missing}")
        raw = values.loc[:, columns].to_numpy()
    else:
        raw = np.asarray(values)
    if raw.dtype.kind not in "fiu" or raw.ndim != 2 or raw.shape[1] != 4:
        raise DownstreamContractError(f"{name} must be numeric [N,4]")
    matrix = np.asarray(raw, dtype=np.float64)
    if np.isinf(matrix).any():
        raise DownstreamContractError(f"{name} contains infinity")
    if not allow_missing and not np.isfinite(matrix).all():
        raise DownstreamContractError(f"{name} contains missing/non-finite values")
    finite = matrix[np.isfinite(matrix)]
    if np.any(finite < 0.0):
        raise DownstreamContractError(f"{name} must be non-negative")
    return matrix


def _validity_matrix(
    valid: Any | None, *, ftv: np.ndarray, expected_rows: int
) -> np.ndarray:
    finite = np.isfinite(ftv)
    if valid is None:
        return finite
    raw = np.asarray(valid)
    if raw.dtype.kind != "b" or raw.shape != (expected_rows, 4):
        raise DownstreamContractError("FTV validity must be boolean [N,4]")
    if np.any(raw & ~finite):
        raise DownstreamContractError("FTV validity marks a non-finite target as valid")
    return raw.copy()


def _state_validity_matrix(valid: Any | None, *, expected_rows: int) -> np.ndarray:
    if valid is None:
        return np.ones((expected_rows, 4), dtype=bool)
    raw = np.asarray(valid)
    if raw.dtype.kind != "b" or raw.shape != (expected_rows, 4):
        raise DownstreamContractError("MRI state validity must be boolean [N,4]")
    return raw.copy()


class FoldClinicalPreprocessor:
    """Train-only C2 clinical encoding followed by train-only StandardScaler.

    ``encode`` exposes the deterministic pre-scaling design.  In that design a
    non-missing held-out category not observed on train maps to an all-zero
    block.  ``transform`` applies the scaler fitted on that encoded train
    design; it never changes the vocabulary or imputation medians.
    """

    fields = C2_FULL_WITH_TREATMENT_FIELDS

    def __init__(
        self,
        *,
        outer_fold: int | str,
        missing_token: str = MISSING_CATEGORY,
    ) -> None:
        if isinstance(outer_fold, str) and not outer_fold.strip():
            raise DownstreamContractError("outer_fold may not be empty")
        if not isinstance(missing_token, str) or not missing_token:
            raise DownstreamContractError("missing_token must be a non-empty string")
        self.outer_fold = outer_fold
        self.missing_token = missing_token
        self.numeric_medians_: dict[str, float] = {}
        self.categories_: dict[str, tuple[str, ...]] = {}
        self.feature_names_: tuple[str, ...] = ()
        self.scaler_: StandardScaler | None = None
        self._train_patient_order_sha256: str | None = None

    @property
    def fitted(self) -> bool:
        return self.scaler_ is not None

    @staticmethod
    def _require_frame(frame: pd.DataFrame, fields: Sequence[str]) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise DownstreamContractError("clinical input must be a pandas DataFrame")
        if missing := [field for field in fields if field not in frame.columns]:
            raise DownstreamContractError(f"clinical input misses C2 fields: {missing}")

    def _categories(self, series: pd.Series) -> np.ndarray:
        values: list[str] = []
        for value in series.to_numpy(dtype=object):
            if pd.isna(value) or (isinstance(value, str) and not value.strip()):
                values.append(self.missing_token)
                continue
            text = str(value)
            if text != text.strip():
                raise DownstreamContractError(
                    f"categorical field {series.name} contains padded whitespace"
                )
            values.append(text)
        return np.asarray(values, dtype=object)

    @staticmethod
    def _numeric(series: pd.Series) -> np.ndarray:
        try:
            values = pd.to_numeric(series, errors="raise").to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise DownstreamContractError(
                f"numeric clinical field {series.name} is not numeric"
            ) from error
        if np.isinf(values).any():
            raise DownstreamContractError(
                f"numeric clinical field {series.name} contains infinity"
            )
        return values

    def fit(
        self,
        train_frame: pd.DataFrame,
        *,
        train_patient_ids: Sequence[Any] | None = None,
        split: str = "train",
    ) -> "FoldClinicalPreprocessor":
        if split != "train":
            raise DownstreamContractError(
                "clinical preprocessing may be fitted only for split='train'"
            )
        if self.fitted:
            raise DownstreamContractError(
                "clinical preprocessor is single-fit; create one instance per fold"
            )
        self._require_frame(train_frame, self.fields)
        if train_frame.empty:
            raise DownstreamContractError("clinical train frame may not be empty")
        if train_patient_ids is not None:
            ids = _unique_patient_ids(
                train_patient_ids,
                expected_rows=len(train_frame),
                name="train_patient_ids",
            )
            self._train_patient_order_sha256 = hashlib.sha256(
                json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest()

        feature_names: list[str] = []
        for field in self.fields:
            if field in NUMERIC_CLINICAL_FIELDS:
                values = self._numeric(train_frame[field])
                finite = values[np.isfinite(values)]
                if not finite.size:
                    raise DownstreamContractError(
                        f"numeric clinical field {field} has no train value"
                    )
                self.numeric_medians_[field] = float(np.median(finite))
                feature_names.append(field)
            else:
                values = self._categories(train_frame[field])
                levels = tuple(sorted(set(values.tolist()) | {self.missing_token}))
                self.categories_[field] = levels
                feature_names.extend(f"{field}={level}" for level in levels)
        self.feature_names_ = tuple(feature_names)
        encoded = self._encode_fitted(train_frame)
        self.scaler_ = StandardScaler().fit(encoded)
        return self

    def _encode_fitted(self, frame: pd.DataFrame) -> np.ndarray:
        self._require_frame(frame, self.fields)
        blocks: list[np.ndarray] = []
        for field in self.fields:
            if field in NUMERIC_CLINICAL_FIELDS:
                values = self._numeric(frame[field])
                values = np.where(
                    np.isnan(values), self.numeric_medians_[field], values
                )
                blocks.append(values[:, None])
                continue
            values = self._categories(frame[field])
            levels = self.categories_[field]
            level_index = {value: index for index, value in enumerate(levels)}
            block = np.zeros((len(frame), len(levels)), dtype=np.float64)
            for row, value in enumerate(values):
                index = level_index.get(str(value))
                if index is not None:
                    block[row, index] = 1.0
            blocks.append(block)
        encoded = np.concatenate(blocks, axis=1)
        if encoded.shape != (len(frame), len(self.feature_names_)):
            raise AssertionError("clinical encoded feature shape drifted")
        if not np.isfinite(encoded).all():
            raise DownstreamContractError("clinical encoded features are non-finite")
        return encoded

    def encode(self, frame: pd.DataFrame) -> np.ndarray:
        """Return median-imputed/one-hot features before StandardScaler."""

        if not self.fitted:
            raise DownstreamContractError("clinical preprocessor must be fitted first")
        return self._encode_fitted(frame).copy()

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Encode and apply the outer-train-fitted StandardScaler."""

        if self.scaler_ is None:
            raise DownstreamContractError("clinical preprocessor must be fitted first")
        transformed = np.asarray(
            self.scaler_.transform(self._encode_fitted(frame)), dtype=np.float64
        )
        if not np.isfinite(transformed).all():
            raise DownstreamContractError("scaled clinical features are non-finite")
        return transformed

    def fit_transform(
        self,
        train_frame: pd.DataFrame,
        *,
        train_patient_ids: Sequence[Any] | None = None,
        split: str = "train",
    ) -> np.ndarray:
        self.fit(
            train_frame,
            train_patient_ids=train_patient_ids,
            split=split,
        )
        return self.transform(train_frame)

    def get_feature_names_out(self) -> np.ndarray:
        if not self.fitted:
            raise DownstreamContractError("clinical preprocessor must be fitted first")
        return np.asarray(self.feature_names_, dtype=str)

    @property
    def provenance(self) -> dict[str, Any]:
        if self.scaler_ is None:
            raise DownstreamContractError("clinical preprocessor must be fitted first")
        return {
            "schema_version": 1,
            "clinical_contract": "C2_full_with_treatment",
            "fields": list(self.fields),
            "outer_fold": self.outer_fold,
            "fit_scope": "outer_train_only",
            "numeric_imputation": "outer_train_median",
            "categorical_levels": "sorted_outer_train_levels_plus_missing_token",
            "heldout_unseen_nonmissing": "all_zero_category_block_before_scaling",
            "scaler": "StandardScaler_fit_outer_train_encoded_features_only",
            "missing_token": self.missing_token,
            "numeric_medians": dict(self.numeric_medians_),
            "categories": {
                field: list(levels) for field, levels in self.categories_.items()
            },
            "feature_names": list(self.feature_names_),
            "train_patient_order_sha256": self._train_patient_order_sha256,
        }


def chronological_mri_prefix(
    states: Any, timing: str | int, *, state_dim: int | None = 192
) -> np.ndarray:
    """Flatten MRI states in strict ``T0,T1,...,timing`` order."""

    values = _state_tensor(states)
    if state_dim is not None and values.shape[2] != int(state_dim):
        raise DownstreamContractError(
            f"MRI state dimension must be {int(state_dim)}; got {values.shape[2]}"
        )
    end = _timing_index(timing) + 1
    return values[:, :end, :].reshape(len(values), end * values.shape[2]).copy()


def causal_ftv_prefix(ftv: Any, timing: str | int) -> np.ndarray:
    """Return only ``log1p(FTV_T0)..log1p(FTV_timing)``."""

    values = _ftv_matrix(ftv, allow_missing=False)
    end = _timing_index(timing) + 1
    return np.log1p(values[:, :end]).copy()


def build_pcr_feature_sets(
    clinical_features: Any,
    states: Any,
    timing: str | int,
    *,
    ftv: Any | None = None,
    state_dim: int | None = 192,
) -> dict[str, np.ndarray]:
    """Build chronologically matched C/M/F combinations for frozen pCR probes.

    The function accepts *already train-fold-preprocessed* clinical features;
    it contains no outcome argument.  If FTV is omitted, only ``C``, ``M``, and
    ``C+M`` are returned.
    """

    clinical = np.asarray(clinical_features)
    if (
        clinical.dtype.kind not in "fiu"
        or clinical.ndim != 2
        or not clinical.shape[0]
        or not clinical.shape[1]
    ):
        raise DownstreamContractError("clinical_features must be numeric [N,C]")
    clinical = np.asarray(clinical, dtype=np.float64)
    if not np.isfinite(clinical).all():
        raise DownstreamContractError("clinical_features contain NaN or infinity")
    mri = chronological_mri_prefix(states, timing, state_dim=state_dim)
    if len(mri) != len(clinical):
        raise DownstreamContractError("clinical and MRI patient counts differ")
    output = {
        "C": clinical.copy(),
        "M": mri,
        "C+M": np.concatenate((clinical, mri), axis=1),
    }
    if ftv is not None:
        ftv_prefix = causal_ftv_prefix(ftv, timing)
        if len(ftv_prefix) != len(clinical):
            raise DownstreamContractError("clinical and FTV patient counts differ")
        output.update(
            {
                "F": ftv_prefix,
                "C+F": np.concatenate((clinical, ftv_prefix), axis=1),
                "C+F+M": np.concatenate((clinical, ftv_prefix, mri), axis=1),
            }
        )
    return output


@dataclass(frozen=True)
class NaturalProbeRows:
    """Aligned natural-scale rows for one static or literal-delta endpoint."""

    patient_ids: tuple[str, ...]
    features: np.ndarray
    targets: np.ndarray
    task: str
    endpoint: str
    target_semantics: str
    source_row_indices: np.ndarray
    excluded_invalid_targets: int

    def __post_init__(self) -> None:
        n_rows = len(self.patient_ids)
        if self.features.ndim != 2 or self.features.shape[0] != n_rows:
            raise DownstreamContractError("probe row feature shape is inconsistent")
        if self.targets.shape != (n_rows,) or self.source_row_indices.shape != (
            n_rows,
        ):
            raise DownstreamContractError(
                "probe row target/index shape is inconsistent"
            )
        if not np.isfinite(self.features).all() or not np.isfinite(self.targets).all():
            raise DownstreamContractError("probe rows contain NaN or infinity")


def build_static_rows(
    patient_ids: Sequence[Any],
    states: Any,
    ftv: Any,
    visit: str | int,
    *,
    ftv_valid: Any | None = None,
    state_valid: Any | None = None,
) -> NaturalProbeRows:
    """Build ``state_t -> natural FTV_t`` rows, excluding invalid targets."""

    state_array = _state_tensor(states)
    ids = _unique_patient_ids(
        patient_ids, expected_rows=len(state_array), name="patient_ids"
    )
    ftv_array = _ftv_matrix(ftv, allow_missing=True)
    if len(ftv_array) != len(state_array):
        raise DownstreamContractError("MRI and FTV patient counts differ")
    valid = _validity_matrix(ftv_valid, ftv=ftv_array, expected_rows=len(ids))
    state_is_valid = _state_validity_matrix(state_valid, expected_rows=len(ids))
    index = _timing_index(visit)
    selected = np.flatnonzero(valid[:, index] & state_is_valid[:, index])
    if not selected.size:
        raise DownstreamContractError(f"no valid static rows for {VISITS[index]}")
    return NaturalProbeRows(
        patient_ids=tuple(ids[row] for row in selected),
        features=state_array[selected, index, :].copy(),
        targets=ftv_array[selected, index].copy(),
        task="static",
        endpoint=VISITS[index],
        target_semantics="natural_FTV_at_observed_visit",
        source_row_indices=selected.astype(np.int64, copy=False),
        excluded_invalid_targets=int(len(ids) - len(selected)),
    )


def build_literal_delta_rows(
    patient_ids: Sequence[Any],
    states: Any,
    ftv: Any,
    transition: str | int,
    *,
    ftv_valid: Any | None = None,
    state_valid: Any | None = None,
) -> NaturalProbeRows:
    """Build ``state_end-state_start -> FTV_end-FTV_start`` natural rows."""

    state_array = _state_tensor(states)
    ids = _unique_patient_ids(
        patient_ids, expected_rows=len(state_array), name="patient_ids"
    )
    ftv_array = _ftv_matrix(ftv, allow_missing=True)
    if len(ftv_array) != len(state_array):
        raise DownstreamContractError("MRI and FTV patient counts differ")
    valid = _validity_matrix(ftv_valid, ftv=ftv_array, expected_rows=len(ids))
    state_is_valid = _state_validity_matrix(state_valid, expected_rows=len(ids))
    name, start, end = _transition_indices(transition)
    row_valid = (
        valid[:, start]
        & valid[:, end]
        & state_is_valid[:, start]
        & state_is_valid[:, end]
    )
    selected = np.flatnonzero(row_valid)
    if not selected.size:
        raise DownstreamContractError(f"no valid literal-delta rows for {name}")
    return NaturalProbeRows(
        patient_ids=tuple(ids[row] for row in selected),
        features=(state_array[selected, end, :] - state_array[selected, start, :]),
        targets=(ftv_array[selected, end] - ftv_array[selected, start]),
        task="delta",
        endpoint=name,
        target_semantics="literal_natural_FTV_end_minus_FTV_start",
        source_row_indices=selected.astype(np.int64, copy=False),
        excluded_invalid_targets=int(len(ids) - len(selected)),
    )


def load_pcr_labels_downstream_only(
    clinical_table: pd.DataFrame,
    patient_ids: Sequence[Any],
    *,
    purpose: str,
    patient_column: str = "patient_id",
    label_column: str = "label_pcr",
) -> np.ndarray:
    """Align binary pCR labels for frozen probes only.

    Callers must pass ``purpose='frozen_downstream_probe'``.  No training data
    loader or representation constructor calls this function.
    """

    if purpose != PCR_DOWNSTREAM_PURPOSE:
        raise PermissionError(f"pCR access requires purpose={PCR_DOWNSTREAM_PURPOSE!r}")
    if not isinstance(clinical_table, pd.DataFrame):
        raise DownstreamContractError("clinical_table must be a pandas DataFrame")
    if missing := [
        column
        for column in (patient_column, label_column)
        if column not in clinical_table.columns
    ]:
        raise DownstreamContractError(f"pCR table misses columns: {missing}")
    ids = _unique_patient_ids(patient_ids, name="patient_ids")
    source = clinical_table.loc[:, [patient_column, label_column]].copy()
    if source.isna().any().any():
        raise DownstreamContractError("pCR source contains missing IDs or labels")
    source[patient_column] = source[patient_column].astype(str)
    if source[patient_column].eq("").any() or source[patient_column].duplicated().any():
        raise DownstreamContractError(
            "pCR source patient IDs must be non-empty and unique"
        )
    indexed = source.set_index(patient_column, verify_integrity=True)
    missing_ids = sorted(set(ids) - set(indexed.index))
    if missing_ids:
        raise DownstreamContractError(
            f"pCR source misses requested patients: {missing_ids[:5]}"
        )
    try:
        numeric = pd.to_numeric(
            indexed.loc[list(ids), label_column], errors="raise"
        ).to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise DownstreamContractError(
            "pCR labels must be numeric binary 0/1"
        ) from error
    if not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all():
        raise DownstreamContractError("pCR labels must be finite binary 0/1")
    return numeric.astype(np.int64)


__all__ = [
    "C2_FULL_WITH_TREATMENT_FIELDS",
    "CATEGORICAL_CLINICAL_FIELDS",
    "DownstreamContractError",
    "FoldClinicalPreprocessor",
    "MISSING_CATEGORY",
    "NUMERIC_CLINICAL_FIELDS",
    "NaturalProbeRows",
    "PCR_DOWNSTREAM_PURPOSE",
    "TRANSITIONS",
    "VISITS",
    "build_literal_delta_rows",
    "build_pcr_feature_sets",
    "build_static_rows",
    "causal_ftv_prefix",
    "chronological_mri_prefix",
    "load_pcr_labels_downstream_only",
]

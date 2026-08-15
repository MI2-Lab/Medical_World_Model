"""Leak-resistant treatment-condition data for the patch-token pilot.

The clinical tables used by this module contain outcome columns, but those
columns are never parsed.  Every clinical CSV read is constrained by
``CLINICAL_CSV_USECOLS`` and the materialized frame is checked to contain
exactly that pCR-free schema.  File digests are opaque provenance checks; they
are not parsed as tabular data.

Age normalization is deliberately a two-step contract: load the authorized
population, then fit :class:`ConditionEncoder` with one outer-fold training set
and the complete authorized external train-only set.  Validation and test
records remain transform-only.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # Contract/manifest checks do not need PyTorch.
    torch = None  # type: ignore[assignment]

    class Dataset:  # type: ignore[no-redef]
        pass


ISPY2_CLINICAL_TABLE = Path(
    "/data/data/Preprocessed/I-SPY2/clinical_labels_complete4visits.csv"
)
ISPY1_CLINICAL_TABLE = Path(
    "/data/data/Preprocessed/I-SPY1/clinical_labels_complete4visits.csv"
)

# These hashes bind the exact clinical assets inspected before preregistration.
ISPY2_CLINICAL_SHA256 = (
    "b3355f8ac80cf8f0fa95722b8d8a8b73d96790e9ded5c491ddb5b2e6a7793436"
)
ISPY1_CLINICAL_SHA256 = (
    "7301e6d43ce2c8aa4f45a56fa43f065c4a5c0a119f1735e3d2a540337940e4fd"
)

# Keep this tuple literal, fixed, and in the same order as configs/pilot.json.
# It is never inferred from any fold or evaluation table.
FIXED_ARM_VOCAB: tuple[str, ...] = (
    "ISPY1_NACT",
    "Paclitaxel",
    "Paclitaxel + ABT 888 + Carboplatin",
    "Paclitaxel + AMG 386",
    "Paclitaxel + AMG 386 + Trastuzumab",
    "Paclitaxel + Ganetespib",
    "Paclitaxel + Ganitumab",
    "Paclitaxel + MK-2206",
    "Paclitaxel + MK-2206 + Trastuzumab",
    "Paclitaxel + Neratinib",
    "Paclitaxel + Pembrolizumab",
    "Paclitaxel + Pertuzumab + Trastuzumab",
    "Paclitaxel + Trastuzumab",
    "T-DM1 + Pertuzumab",
)
ARM_TO_INDEX: Mapping[str, int] = MappingProxyType(
    {arm: index for index, arm in enumerate(FIXED_ARM_VOCAB)}
)
ISPY2_ARMS = frozenset(FIXED_ARM_VOCAB[1:])

# This is the complete and only training-time clinical CSV allow-list.  In
# particular, neither label_pcr nor raw_pCR is present.
CLINICAL_CSV_USECOLS: tuple[str, ...] = (
    "patient_id",
    "arm",
    "label_hr",
    "label_her2",
    "label_mp",
    "age_at_screening",
)
if any("pcr" in name.casefold() for name in CLINICAL_CSV_USECOLS):  # pragma: no cover
    raise RuntimeError("training clinical allow-list must remain pCR-free")

TEMPORAL_FEATURE_NAMES: tuple[str, ...] = (
    "target_T1",
    "target_T2",
    "target_T3",
    "observed_T0",
    "observed_T1",
    "observed_T2",
    "observed_T3",
)
CLINICAL_FEATURE_NAMES: tuple[str, ...] = (
    "HR",
    "HER2",
    "MP_as_provided",
    "age_z",
    "age_missing",
)

_TEMPORAL_BITS = np.asarray(
    (
        (1, 0, 0, 1, 0, 0, 0),  # T0 -> T1
        (0, 1, 0, 1, 1, 0, 0),  # T1 -> T2
        (0, 0, 1, 1, 1, 1, 0),  # T2 -> T3
    ),
    dtype=np.float32,
)
_TEMPORAL_BITS.setflags(write=False)
_NOMINAL_DELTA_T = np.ones(3, dtype=np.float32)
_NOMINAL_DELTA_T.setflags(write=False)

CONDITION_MATRIX_FEATURE_NAMES: tuple[str, ...] = (
    TEMPORAL_FEATURE_NAMES
    + tuple(f"arm={arm}" for arm in FIXED_ARM_VOCAB)
    + CLINICAL_FEATURE_NAMES
    + ("delta_t_unitless",)
)
CONDITION_MATRIX_SHAPE = (3, len(CONDITION_MATRIX_FEATURE_NAMES))

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def file_sha256(path: str | Path) -> str:
    """Return an opaque byte-level provenance digest for a regular file."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(value: str, label: str = "SHA-256") -> str:
    digest = str(value).strip().lower()
    if _SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"{label} must be a lowercase 64-character SHA-256")
    return digest


def patient_set_sha256(patient_ids: Iterable[str]) -> str:
    """Hash a unique patient set without publishing its members.

    The canonical representation is a compact JSON array sorted by Unicode
    codepoint.  Empty sets and duplicate-bearing inputs are rejected so the
    digest is also a useful coverage assertion.
    """

    values = tuple(str(value) for value in patient_ids)
    if not values:
        raise ValueError("patient set must be nonempty")
    if any(not value or value != value.strip() for value in values):
        raise ValueError("patient identifiers must be nonempty and whitespace-stable")
    if len(set(values)) != len(values):
        raise ValueError("patient identifiers must be unique")
    payload = json.dumps(sorted(values), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _checked_ids(
    values: Iterable[str], label: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    patient_ids = tuple(str(value) for value in values)
    if not patient_ids and not allow_empty:
        raise ValueError(f"{label} must be nonempty")
    if any(not value or value != value.strip() for value in patient_ids):
        raise ValueError(f"{label} contains an empty or whitespace-unstable identifier")
    if len(set(patient_ids)) != len(patient_ids):
        raise ValueError(f"{label} contains duplicate identifiers")
    return patient_ids


def _validate_expected_patient_hash(
    patient_ids: Sequence[str], expected_sha256: str | None, label: str
) -> str:
    observed = patient_set_sha256(patient_ids)
    if expected_sha256 is not None:
        expected = require_sha256(expected_sha256, f"{label} patient-set SHA-256")
        if observed != expected:
            raise ValueError(f"{label} patient-set SHA-256 mismatch")
    return observed


@dataclass(frozen=True)
class ClinicalConditionRecord:
    """Only baseline-known fields authorized for transition conditioning."""

    patient_id: str
    cohort: str
    arm: str
    hr: int
    her2: int
    mp: int
    age: float


@dataclass(frozen=True)
class ClinicalSourceAudit:
    cohort: str
    path: Path
    sha256: str
    source_row_count: int

    def aggregate_dict(
        self, selected_count: int, selected_patient_sha256: str
    ) -> dict[str, Any]:
        return {
            "cohort": self.cohort,
            "path": str(self.path),
            "sha256": self.sha256,
            "source_row_count": int(self.source_row_count),
            "selected_count": int(selected_count),
            "selected_patient_set_sha256": selected_patient_sha256,
        }


def _binary_column(frame: pd.DataFrame, column: str, label: str) -> np.ndarray:
    numeric = pd.to_numeric(frame[column], errors="raise")
    if numeric.isna().any() or not numeric.isin((0, 1)).all():
        raise ValueError(
            f"{label}.{column} must contain only complete binary 0/1 values"
        )
    return numeric.astype(np.int8).to_numpy(copy=True)


def read_clinical_condition_csv(
    path: str | Path,
    expected_sha256: str,
    *,
    cohort: str,
) -> tuple[tuple[ClinicalConditionRecord, ...], ClinicalSourceAudit]:
    """Read one clinical source through the exact pCR-free allow-list.

    ``cohort`` is source provenance, not a value learned from the CSV.  The
    I-SPY2 source accepts only the 13 preregistered trial arms and the I-SPY1
    source accepts only ``ISPY1_NACT``.
    """

    normalized_cohort = str(cohort).strip().lower().replace("-", "")
    if normalized_cohort not in {"ispy1", "ispy2"}:
        raise ValueError("cohort must be ispy1 or ispy2")
    source = Path(path).expanduser().resolve()
    expected = require_sha256(expected_sha256, f"{cohort} clinical table SHA-256")
    observed = file_sha256(source)
    if observed != expected:
        raise ValueError(f"{cohort} clinical table SHA-256 mismatch")

    # Do not replace this with an unrestricted read followed by column slicing:
    # the source contains pCR and other fields forbidden during world-model
    # training.
    frame = pd.read_csv(
        source,
        usecols=list(CLINICAL_CSV_USECOLS),
        dtype={"patient_id": "string", "arm": "string"},
    )
    if len(frame.columns) != len(CLINICAL_CSV_USECOLS) or set(frame.columns) != set(
        CLINICAL_CSV_USECOLS
    ):
        raise ValueError(
            "clinical parser materialized a column outside the fixed allow-list"
        )
    frame = frame.loc[:, list(CLINICAL_CSV_USECOLS)].copy()
    if frame.empty:
        raise ValueError(f"{cohort} clinical condition table is empty")
    if frame["patient_id"].isna().any() or frame["arm"].isna().any():
        raise ValueError(f"{cohort} clinical condition table has missing identity/arm")

    patient_ids = frame["patient_id"].astype(str)
    arms = frame["arm"].astype(str)
    if patient_ids.str.strip().ne(patient_ids).any() or patient_ids.eq("").any():
        raise ValueError(f"{cohort} clinical table has invalid patient identifiers")
    if patient_ids.duplicated().any():
        raise ValueError(f"{cohort} clinical table has duplicate patient identifiers")
    if arms.str.strip().ne(arms).any() or arms.eq("").any():
        raise ValueError(f"{cohort} clinical table has invalid assigned arms")

    observed_arms = frozenset(arms)
    expected_arms = {"ISPY1_NACT"} if normalized_cohort == "ispy1" else ISPY2_ARMS
    unknown_arms = observed_arms.difference(expected_arms)
    if unknown_arms:
        # Values are intentionally omitted from the exception to avoid copying
        # potentially sensitive source contents into tracked logs.
        raise ValueError(
            f"{cohort} clinical table contains {len(unknown_arms)} unknown arm value(s)"
        )
    if normalized_cohort == "ispy1" and observed_arms != {"ISPY1_NACT"}:
        raise ValueError("I-SPY1 condition rows must use only ISPY1_NACT")

    hr = _binary_column(frame, "label_hr", cohort)
    her2 = _binary_column(frame, "label_her2", cohort)
    mp = _binary_column(frame, "label_mp", cohort)
    age = pd.to_numeric(frame["age_at_screening"], errors="raise").to_numpy(
        dtype=np.float64
    )
    if np.isinf(age).any():
        raise ValueError(f"{cohort}.age_at_screening must be finite or missing")

    canonical_cohort = "I-SPY1" if normalized_cohort == "ispy1" else "I-SPY2"
    records = tuple(
        ClinicalConditionRecord(
            patient_id=str(patient_id),
            cohort=canonical_cohort,
            arm=str(arm),
            hr=int(hr[index]),
            her2=int(her2[index]),
            mp=int(mp[index]),
            age=float(age[index]),
        )
        for index, (patient_id, arm) in enumerate(zip(patient_ids, arms, strict=True))
    )
    return records, ClinicalSourceAudit(canonical_cohort, source, observed, len(frame))


@dataclass(frozen=True)
class AuthorizedConditionTable:
    """Authorized records plus source/coverage provenance.

    Patient identifiers live only in memory and private inputs.  Use
    :meth:`aggregate_metadata` for serialization.
    """

    records: Mapping[str, ClinicalConditionRecord]
    primary_patient_ids: tuple[str, ...]
    external_train_only_patient_ids: tuple[str, ...]
    ispy2_source: ClinicalSourceAudit
    ispy1_source: ClinicalSourceAudit
    primary_patient_sha256: str
    external_patient_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", MappingProxyType(dict(self.records)))
        expected = set(self.primary_patient_ids) | set(
            self.external_train_only_patient_ids
        )
        if set(self.records) != expected:
            raise ValueError(
                "authorized condition records do not exactly match authorized coverage"
            )

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Preserve the read-only mapping while allowing spawn-safe workers."""

        return (
            type(self),
            (
                dict(self.records),
                self.primary_patient_ids,
                self.external_train_only_patient_ids,
                self.ispy2_source,
                self.ispy1_source,
                self.primary_patient_sha256,
                self.external_patient_sha256,
            ),
        )

    def aggregate_metadata(self) -> dict[str, Any]:
        return {
            "primary": self.ispy2_source.aggregate_dict(
                len(self.primary_patient_ids), self.primary_patient_sha256
            ),
            "authorized_external_train_only": self.ispy1_source.aggregate_dict(
                len(self.external_train_only_patient_ids), self.external_patient_sha256
            ),
            "total_authorized_count": len(self.records),
        }


def _select_records(
    records: Sequence[ClinicalConditionRecord],
    required_ids: Sequence[str],
    label: str,
) -> tuple[ClinicalConditionRecord, ...]:
    by_id = {record.patient_id: record for record in records}
    missing_count = len(set(required_ids).difference(by_id))
    if missing_count:
        raise ValueError(
            f"{label} clinical coverage misses {missing_count} required patient(s)"
        )
    return tuple(by_id[patient_id] for patient_id in required_ids)


def load_authorized_condition_table(
    *,
    primary_patient_ids: Iterable[str],
    authorized_external_train_only_patient_ids: Iterable[str],
    ispy2_path: str | Path = ISPY2_CLINICAL_TABLE,
    ispy2_sha256: str = ISPY2_CLINICAL_SHA256,
    ispy1_path: str | Path = ISPY1_CLINICAL_TABLE,
    ispy1_sha256: str = ISPY1_CLINICAL_SHA256,
    expected_primary_patient_sha256: str | None = None,
    expected_external_patient_sha256: str | None = None,
) -> AuthorizedConditionTable:
    """Load exactly the fold population and authorized external population."""

    primary_ids = _checked_ids(primary_patient_ids, "primary patient IDs")
    external_ids = _checked_ids(
        authorized_external_train_only_patient_ids,
        "authorized external train-only patient IDs",
    )
    if set(primary_ids) & set(external_ids):
        raise ValueError("primary and external train-only patient sets overlap")
    primary_hash = _validate_expected_patient_hash(
        primary_ids, expected_primary_patient_sha256, "primary"
    )
    external_hash = _validate_expected_patient_hash(
        external_ids, expected_external_patient_sha256, "external train-only"
    )

    ispy2_records, ispy2_audit = read_clinical_condition_csv(
        ispy2_path, ispy2_sha256, cohort="ispy2"
    )
    ispy1_records, ispy1_audit = read_clinical_condition_csv(
        ispy1_path, ispy1_sha256, cohort="ispy1"
    )
    selected_primary = _select_records(ispy2_records, primary_ids, "primary")
    selected_external = _select_records(
        ispy1_records, external_ids, "external train-only"
    )
    combined = selected_primary + selected_external
    by_id: MutableMapping[str, ClinicalConditionRecord] = {
        record.patient_id: record for record in combined
    }
    if len(by_id) != len(combined):
        raise ValueError("clinical sources contain a cross-cohort patient collision")
    return AuthorizedConditionTable(
        records=MappingProxyType(dict(by_id)),
        primary_patient_ids=primary_ids,
        external_train_only_patient_ids=external_ids,
        ispy2_source=ispy2_audit,
        ispy1_source=ispy1_audit,
        primary_patient_sha256=primary_hash,
        external_patient_sha256=external_hash,
    )


@dataclass(frozen=True)
class AgeNormalization:
    mean: float
    std: float
    fit_patient_count: int
    observed_count: int
    missing_count: int
    fit_patient_sha256: str
    primary_train_patient_sha256: str
    external_train_only_patient_sha256: str

    def aggregate_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean,
            "std": self.std,
            "fit_patient_count": self.fit_patient_count,
            "observed_count": self.observed_count,
            "missing_count": self.missing_count,
            "fit_patient_set_sha256": self.fit_patient_sha256,
            "primary_train_patient_set_sha256": self.primary_train_patient_sha256,
            "external_train_only_patient_set_sha256": self.external_train_only_patient_sha256,
            "fit_scope": "outer_train_plus_authorized_external_train_only_only",
            "missing_policy": "mean_impute_to_age_z_zero_plus_explicit_indicator",
            "std_definition": "population_ddof_0_with_unit_fallback_if_constant",
        }


class ConditionEncoder:
    """Fold-fitted, pCR-free transition condition encoder.

    The model-facing decomposition is:

    * ``arm_index``: scalar int64 (batched shape ``[B]``),
    * ``clinical``: float32 ``[5]`` (batched ``[B,5]``),
    * ``temporal_bits``: float32 ``[3,7]`` (batched ``[B,3,7]``), and
    * ``delta_t``: float32 ``[3]`` (batched ``[B,3]``).

    :meth:`encode_matrix` provides the equivalent fixed-one-hot ``[3,C]``
    representation for contract tests and audits.
    """

    def __init__(
        self,
        table: AuthorizedConditionTable,
        normalization: AgeNormalization,
        primary_train_patient_ids: Sequence[str],
        external_train_only_patient_ids: Sequence[str],
    ) -> None:
        self.table = table
        self.normalization = normalization
        self._primary_train_ids = frozenset(primary_train_patient_ids)
        self._external_train_only_ids = frozenset(external_train_only_patient_ids)
        self._fit_ids = self._primary_train_ids | self._external_train_only_ids

    @classmethod
    def fit(
        cls,
        table: AuthorizedConditionTable,
        *,
        outer_train_patient_ids: Iterable[str],
        authorized_external_train_only_patient_ids: Iterable[str],
        expected_outer_train_patient_sha256: str | None = None,
        expected_external_patient_sha256: str | None = None,
        expected_fit_patient_sha256: str | None = None,
    ) -> "ConditionEncoder":
        primary_train = _checked_ids(outer_train_patient_ids, "outer-train patient IDs")
        external = _checked_ids(
            authorized_external_train_only_patient_ids,
            "authorized external train-only patient IDs",
        )
        if set(primary_train) & set(external):
            raise ValueError("outer-train and external train-only patient sets overlap")
        if not set(primary_train).issubset(table.primary_patient_ids):
            raise ValueError("age-fit outer-train set contains a non-primary patient")
        if set(external) != set(table.external_train_only_patient_ids):
            raise ValueError(
                "age fit must include the complete authorized external train-only set"
            )
        primary_hash = _validate_expected_patient_hash(
            primary_train, expected_outer_train_patient_sha256, "outer train"
        )
        external_hash = _validate_expected_patient_hash(
            external, expected_external_patient_sha256, "external train-only"
        )
        if external_hash != table.external_patient_sha256:
            raise ValueError(
                "age-fit external patient hash differs from authorization hash"
            )
        fit_ids = primary_train + external
        fit_hash = _validate_expected_patient_hash(
            fit_ids, expected_fit_patient_sha256, "age fit"
        )
        ages = np.asarray(
            [table.records[patient_id].age for patient_id in fit_ids], dtype=np.float64
        )
        finite = ages[np.isfinite(ages)]
        if finite.size == 0:
            raise ValueError("age normalization has no observed training age")
        mean = float(np.mean(finite))
        raw_std = float(np.std(finite, ddof=0))
        std = raw_std if math.isfinite(raw_std) and raw_std > 1e-6 else 1.0
        normalization = AgeNormalization(
            mean=mean,
            std=std,
            fit_patient_count=len(fit_ids),
            observed_count=int(finite.size),
            missing_count=int(len(fit_ids) - finite.size),
            fit_patient_sha256=fit_hash,
            primary_train_patient_sha256=primary_hash,
            external_train_only_patient_sha256=external_hash,
        )
        return cls(table, normalization, primary_train, external)

    @property
    def arm_vocab(self) -> tuple[str, ...]:
        return FIXED_ARM_VOCAB

    @property
    def fit_patient_ids(self) -> frozenset[str]:
        return self._fit_ids

    @property
    def primary_train_patient_ids(self) -> frozenset[str]:
        return self._primary_train_ids

    @property
    def external_train_only_patient_ids(self) -> frozenset[str]:
        return self._external_train_only_ids

    def _record(self, patient_id: str) -> ClinicalConditionRecord:
        identity = str(patient_id)
        try:
            return self.table.records[identity]
        except KeyError as error:
            raise KeyError(
                "patient is outside the authorized condition population"
            ) from error

    def encode_numpy(self, patient_id: str) -> dict[str, np.ndarray]:
        record = self._record(patient_id)
        age_missing = not math.isfinite(record.age)
        age_z = (
            0.0
            if age_missing
            else (record.age - self.normalization.mean) / self.normalization.std
        )
        clinical = np.asarray(
            (record.hr, record.her2, record.mp, age_z, float(age_missing)),
            dtype=np.float32,
        )
        return {
            "arm_index": np.asarray(ARM_TO_INDEX[record.arm], dtype=np.int64),
            "clinical": clinical,
            "temporal_bits": _TEMPORAL_BITS.copy(),
            "delta_t": _NOMINAL_DELTA_T.copy(),
        }

    def encode_torch(self, patient_id: str) -> dict[str, Any]:
        if torch is None:
            raise RuntimeError(
                "PyTorch is required to materialize a training condition"
            )
        encoded = self.encode_numpy(patient_id)
        return {
            "arm_index": torch.as_tensor(encoded["arm_index"], dtype=torch.long),
            "clinical": torch.from_numpy(encoded["clinical"]),
            "temporal_bits": torch.from_numpy(encoded["temporal_bits"]),
            "delta_t": torch.from_numpy(encoded["delta_t"]),
        }

    def encode_matrix(self, patient_id: str) -> np.ndarray:
        """Return fixed-one-hot condition rows with shape ``[3,27]``."""

        encoded = self.encode_numpy(patient_id)
        arm = np.zeros((3, len(FIXED_ARM_VOCAB)), dtype=np.float32)
        arm[:, int(encoded["arm_index"])] = 1.0
        clinical = np.repeat(encoded["clinical"][None, :], 3, axis=0)
        matrix = np.concatenate(
            (
                encoded["temporal_bits"],
                arm,
                clinical,
                encoded["delta_t"][:, None],
            ),
            axis=1,
        ).astype(np.float32, copy=False)
        if matrix.shape != CONDITION_MATRIX_SHAPE:
            raise AssertionError("condition matrix contract drifted")
        return matrix

    def aggregate_metadata(self) -> dict[str, Any]:
        return {
            "fixed_arm_vocabulary": list(FIXED_ARM_VOCAB),
            "arm_count": len(FIXED_ARM_VOCAB),
            "clinical_features": list(CLINICAL_FEATURE_NAMES),
            "temporal_features": list(TEMPORAL_FEATURE_NAMES),
            "temporal_shape": [3, 7],
            "delta_t": [1.0, 1.0, 1.0],
            "delta_t_semantics": (
                "unitless_nominal_adjacent_visit_interval_no_measured_elapsed_time_source"
            ),
            "one_hot_matrix_shape": list(CONDITION_MATRIX_SHAPE),
            "age_normalization": self.normalization.aggregate_dict(),
            "training_clinical_csv_allowlist": list(CLINICAL_CSV_USECOLS),
            "pcr_column_loaded": False,
        }


class ConditionedStageBDataset(Dataset):
    """Seal a StageBDataset item and attach only an authorized condition.

    The wrapped dataset must expose ordered ``patient_ids`` and return exactly
    ``patient_id``, ``image``, ``ftv_target``, and ``ftv_mask``.  Any added
    label, sidecar, mask, geometry, or pCR field is rejected rather than silently
    propagated.  ``patient_id`` remains runtime-only because deterministic token
    masking needs its hash; scripts in this experiment never serialize it.
    """

    BASE_ITEM_KEYS = frozenset(("patient_id", "image", "ftv_target", "ftv_mask"))
    CONDITION_KEYS = frozenset(("arm_index", "clinical", "temporal_bits", "delta_t"))

    def __init__(
        self,
        base: Dataset,
        condition_encoder: ConditionEncoder,
        *,
        split: str = "inference",
        require_exact_train_coverage: bool = True,
        include_patient_id: bool = True,
    ) -> None:
        if not hasattr(base, "patient_ids"):
            raise TypeError("sealed StageBDataset must expose ordered patient_ids")
        patient_ids = _checked_ids(getattr(base, "patient_ids"), "Stage B patient IDs")
        if len(base) != len(patient_ids):
            raise ValueError("Stage B length and patient_ids coverage disagree")
        missing_count = len(
            set(patient_ids).difference(condition_encoder.table.records)
        )
        if missing_count:
            raise ValueError(
                f"condition table misses {missing_count} Stage B patient(s)"
            )

        normalized_split = str(split).strip().lower()
        if normalized_split not in {"train", "val", "test", "inference"}:
            raise ValueError("split must be train, val, test, or inference")
        patient_set = set(patient_ids)
        if normalized_split == "train":
            if not patient_set.issubset(condition_encoder.fit_patient_ids):
                raise ValueError("training dataset contains a non-age-fit patient")
            if require_exact_train_coverage and patient_set != set(
                condition_encoder.fit_patient_ids
            ):
                raise ValueError(
                    "training dataset does not exactly cover the age-fit population"
                )
        elif normalized_split in {"val", "test"}:
            if patient_set & set(condition_encoder.fit_patient_ids):
                raise ValueError(
                    f"{normalized_split} dataset overlaps age-fit patients"
                )
            if patient_set & set(condition_encoder.external_train_only_patient_ids):
                raise ValueError(
                    f"external train-only patient entered {normalized_split}"
                )

        self.base = base
        self.condition_encoder = condition_encoder
        self.patient_ids = patient_ids
        self.split = normalized_split
        self.include_patient_id = bool(include_patient_id)

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.base[index]
        if not isinstance(item, Mapping):
            raise TypeError("sealed StageBDataset item must be a mapping")
        if set(item) != set(self.BASE_ITEM_KEYS):
            raise ValueError(
                "sealed StageBDataset item keys drifted from the pCR-free contract"
            )
        expected_patient_id = self.patient_ids[index]
        if str(item["patient_id"]) != expected_patient_id:
            raise ValueError(
                "Stage B item identity differs from its sealed patient order"
            )
        condition = self.condition_encoder.encode_torch(expected_patient_id)
        if set(condition) != set(self.CONDITION_KEYS):
            raise AssertionError("condition output contract drifted")
        output: dict[str, Any] = {
            "image": item["image"],
            "ftv_target": item["ftv_target"],
            "ftv_mask": item["ftv_mask"],
            "condition": condition,
        }
        if self.include_patient_id:
            output["patient_id"] = expected_patient_id
        return output


# A descriptive alias makes the scientific role explicit while retaining a
# short, unsurprising class name for callers.
TreatmentConditionedStageBDataset = ConditionedStageBDataset


def transition_condition_from_batch(batch: Mapping[str, Any]) -> Any:
    """Construct ``model.TransitionCondition`` without importing model eagerly.

    ``batch`` may be a default-collated dataset batch (with a nested
    ``condition`` mapping) or that nested mapping itself.  Keeping the import in
    this function lets data/manifest contract tests run before the model module
    or GPU runtime is available.
    """

    from .model import TransitionCondition

    values: Any = batch.get("condition", batch)
    if isinstance(values, TransitionCondition):
        return values
    if not isinstance(values, Mapping):
        raise TypeError("batch condition must be a mapping")
    required = ("arm_index", "clinical", "temporal_bits", "delta_t")
    if set(values) != set(required):
        raise ValueError("batch condition keys do not match TransitionCondition")
    return TransitionCondition(
        arm_index=values["arm_index"],
        clinical=values["clinical"],
        temporal_bits=values["temporal_bits"],
        delta_t=values["delta_t"],
    )


__all__ = [
    "ARM_TO_INDEX",
    "AgeNormalization",
    "AuthorizedConditionTable",
    "CLINICAL_CSV_USECOLS",
    "CLINICAL_FEATURE_NAMES",
    "CONDITION_MATRIX_FEATURE_NAMES",
    "CONDITION_MATRIX_SHAPE",
    "ClinicalConditionRecord",
    "ClinicalSourceAudit",
    "ConditionEncoder",
    "ConditionedStageBDataset",
    "FIXED_ARM_VOCAB",
    "ISPY1_CLINICAL_SHA256",
    "ISPY1_CLINICAL_TABLE",
    "ISPY2_CLINICAL_SHA256",
    "ISPY2_CLINICAL_TABLE",
    "TEMPORAL_FEATURE_NAMES",
    "TreatmentConditionedStageBDataset",
    "file_sha256",
    "load_authorized_condition_table",
    "patient_set_sha256",
    "read_clinical_condition_csv",
    "require_sha256",
    "transition_condition_from_batch",
]

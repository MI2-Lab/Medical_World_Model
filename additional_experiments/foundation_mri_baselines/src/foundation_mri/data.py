"""Fail-closed tabular and frozen-feature loaders for foundation MRI baselines.

The public experiment has one canonical population (808 I-SPY2 patients) and
one explicitly named complete-case population (375 patients with all four
radiomics visits).  This module never performs a permissive inner join: every
loader proves the expected patient coverage before it returns an array.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


SEED = 2026
FOLDS = (0, 1, 2, 3, 4)
SPLITS = ("train", "val", "test")
VISITS = ("T0", "T1", "T2", "T3")
SPATIAL_AXES = ("GLOBAL", "LOCAL")
COHORT_SIZE = 808
RADIOMICS_COMPLETE_CASE_SIZE = 375
RADIOMIC_FEATURES = ("ftv", "sphericity", "ld", "bpe")
HR_HER2_SUBTYPES = ("HR-/HER2-", "HR-/HER2+", "HR+/HER2-", "HR+/HER2+")

# These hashes bind the default command-line run to the audited seed-2026 data.
# Unit tests and prospective, separately locked data may opt out explicitly by
# passing ``expected_sha256=None``.
EXPECTED_FOLD_MANIFEST_SHA256 = (
    "143e482d711225c0611006d99bd7345d2fa1a5c16c65fbaf8399341a0d26aa38"
)
EXPECTED_CLINICAL_SHA256 = (
    "b3355f8ac80cf8f0fa95722b8d8a8b73d96790e9ded5c491ddb5b2e6a7793436"
)
EXPECTED_RADIOMICS_SHA256 = (
    "26fbde8590fde4612267f02d762af99d65926ff6d0206d0e500577ef394ff75d"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+\-]*$")


def file_sha256(path: str | Path) -> str:
    """Return a streaming SHA-256 digest without materialising a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_text_sha256(values: Sequence[str]) -> str:
    """Hash an ordered string sequence using the extractor's length framing."""

    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _newline_ordered_text_sha256(values: Sequence[str]) -> str:
    """Hash patient order with the current-CNN exporter's exact framing."""

    return hashlib.sha256(
        "\n".join(str(value) for value in values).encode("utf-8")
    ).hexdigest()


def _check_file_hash(path: Path, expected: str | None, label: str) -> str:
    observed = file_sha256(path)
    if expected is not None:
        expected = str(expected).strip().lower()
        if not _SHA256_RE.fullmatch(expected):
            raise ValueError(f"{label} expected SHA-256 is malformed")
        if observed != expected:
            raise ValueError(
                f"{label} SHA-256 drifted: expected={expected}, observed={observed}"
            )
    return observed


def _as_strings(values: np.ndarray | Sequence[object], label: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in {"U", "S", "O"}:
        raise ValueError(f"{label} must be a one-dimensional string array")
    output = array.astype(str)
    missing_tokens = {"nan", "none", "<na>", "nat"}
    if any(
        not value.strip() or value.strip().lower() in missing_tokens for value in output
    ):
        raise ValueError(f"{label} contains an empty/missing value")
    return output


def _unique_patient_ids(
    values: np.ndarray | Sequence[object], *, expected_n: int, label: str
) -> np.ndarray:
    patient_ids = _as_strings(values, label)
    if len(patient_ids) != expected_n:
        raise ValueError(f"{label} must contain exactly {expected_n} patients")
    if len(set(patient_ids.tolist())) != len(patient_ids):
        raise ValueError(f"{label} contains duplicate patients")
    return patient_ids


def _alignment_indices(
    observed: Sequence[str], expected: Sequence[str], *, label: str
) -> np.ndarray:
    observed_array = np.asarray(observed, dtype=str)
    expected_array = np.asarray(expected, dtype=str)
    if len(set(observed_array.tolist())) != len(observed_array):
        raise ValueError(f"{label} has duplicate patient IDs")
    if len(set(expected_array.tolist())) != len(expected_array):
        raise ValueError("canonical patient order has duplicate IDs")
    observed_set = set(observed_array.tolist())
    expected_set = set(expected_array.tolist())
    if observed_set != expected_set:
        missing = sorted(expected_set.difference(observed_set))
        extra = sorted(observed_set.difference(expected_set))
        raise ValueError(
            f"{label} patient coverage drifted: missing={len(missing)}, "
            f"extra={len(extra)}"
        )
    lookup = {patient_id: index for index, patient_id in enumerate(observed_array)}
    return np.asarray([lookup[patient_id] for patient_id in expected_array], dtype=np.int64)


def _subset_indices(
    observed: Sequence[str], requested: Sequence[str], *, label: str
) -> np.ndarray:
    observed_array = np.asarray(observed, dtype=str)
    requested_array = np.asarray(requested, dtype=str)
    if len(set(observed_array.tolist())) != len(observed_array):
        raise ValueError(f"{label} source patient IDs are duplicated")
    if len(set(requested_array.tolist())) != len(requested_array):
        raise ValueError(f"{label} requested patient IDs are duplicated")
    lookup = {patient_id: index for index, patient_id in enumerate(observed_array)}
    unknown = sorted(set(requested_array.tolist()).difference(lookup))
    if unknown:
        raise ValueError(f"{label} contains {len(unknown)} unknown patients")
    return np.asarray([lookup[patient_id] for patient_id in requested_array], dtype=np.int64)


def _strict_binary(series: pd.Series, label: str) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all():
        raise ValueError(f"{label} must be complete binary 0/1")
    return numeric.astype(np.int64)


def _strict_bool(series: pd.Series, label: str) -> np.ndarray:
    values: list[bool] = []
    for value in series.tolist():
        if isinstance(value, (bool, np.bool_)):
            values.append(bool(value))
            continue
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y"}:
            values.append(True)
        elif text in {"0", "false", "no", "n"}:
            values.append(False)
        else:
            raise ValueError(f"{label} contains a non-boolean value")
    return np.asarray(values, dtype=bool)


@dataclass(frozen=True)
class FoldManifest:
    """The locked five-fold assignment in canonical patient order."""

    patient_ids: np.ndarray
    labels: np.ndarray
    frame: pd.DataFrame
    sha256: str

    def roles(self, fold: int, patient_ids: Sequence[str] | None = None) -> np.ndarray:
        if int(fold) not in FOLDS:
            raise ValueError(f"fold must be one of {FOLDS}")
        requested = self.patient_ids if patient_ids is None else np.asarray(patient_ids, dtype=str)
        if len(set(requested.tolist())) != len(requested):
            raise ValueError("requested fold patient IDs are duplicated")
        current = self.frame.loc[self.frame["fold"].eq(int(fold))]
        mapping = dict(zip(current["patient_id"], current["split"], strict=True))
        unknown = sorted(set(requested.tolist()).difference(mapping))
        if unknown:
            raise ValueError(f"fold request contains {len(unknown)} unknown patients")
        roles = np.asarray([mapping[patient_id] for patient_id in requested], dtype=str)
        if not set(roles.tolist()).issubset(SPLITS):
            raise AssertionError("validated fold produced an invalid split role")
        return roles

    def labels_for(self, patient_ids: Sequence[str]) -> np.ndarray:
        requested = np.asarray(patient_ids, dtype=str)
        mapping = dict(zip(self.patient_ids, self.labels, strict=True))
        unknown = sorted(set(requested.tolist()).difference(mapping))
        if unknown:
            raise ValueError(f"label request contains {len(unknown)} unknown patients")
        return np.asarray([mapping[value] for value in requested], dtype=np.int64)


def load_fold_manifest(
    path: str | Path,
    *,
    expected_n: int = COHORT_SIZE,
    expected_sha256: str | None = EXPECTED_FOLD_MANIFEST_SHA256,
) -> FoldManifest:
    """Load and prove the exact seed-2026, five-fold patient split contract."""

    source = Path(path).resolve()
    digest = _check_file_hash(source, expected_sha256, "fold manifest")
    frame = pd.read_csv(source, dtype={"patient_id": str})
    required = {"patient_id", "fold", "split", "label_pcr"}
    if set(frame.columns) != required:
        raise ValueError(
            f"fold manifest schema drifted: expected={sorted(required)}, "
            f"observed={sorted(frame.columns)}"
        )
    if len(frame) != expected_n * len(FOLDS):
        raise ValueError("fold manifest must contain N patients in each of five folds")
    frame = frame.loc[:, ["patient_id", "fold", "split", "label_pcr"]].copy()
    frame["patient_id"] = _as_strings(frame["patient_id"].to_numpy(), "fold patient_id")
    fold_numeric = pd.to_numeric(frame["fold"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(fold_numeric).all() or not np.equal(fold_numeric, np.floor(fold_numeric)).all():
        raise ValueError("fold values must be finite integers")
    frame["fold"] = fold_numeric.astype(np.int64)
    if set(frame["fold"].tolist()) != set(FOLDS):
        raise ValueError(f"fold manifest must contain exactly folds {FOLDS}")
    frame["split"] = frame["split"].astype(str).str.strip().str.lower()
    if set(frame["split"].tolist()) != set(SPLITS):
        raise ValueError(f"fold manifest split values must be exactly {SPLITS}")
    frame["label_pcr"] = _strict_binary(frame["label_pcr"], "fold label_pcr")
    if frame.duplicated(["patient_id", "fold"]).any():
        raise ValueError("fold manifest duplicates a patient within a fold")

    patient_ids = np.asarray(sorted(frame["patient_id"].unique()), dtype=str)
    if len(patient_ids) != expected_n:
        raise ValueError(f"fold manifest must contain exactly {expected_n} unique patients")
    canonical_set = set(patient_ids.tolist())
    for fold in FOLDS:
        current = frame.loc[frame["fold"].eq(fold)]
        if len(current) != expected_n or set(current["patient_id"]) != canonical_set:
            raise ValueError(f"fold {fold} does not cover the complete cohort once")
        for split in SPLITS:
            split_rows = current.loc[current["split"].eq(split)]
            if split_rows.empty:
                raise ValueError(f"fold {fold}/{split} is empty")
            if set(split_rows["label_pcr"].tolist()) != {0, 1}:
                raise ValueError(f"fold {fold}/{split} does not contain both pCR classes")

    label_counts = frame.groupby("patient_id", sort=False)["label_pcr"].nunique()
    if not label_counts.eq(1).all():
        raise ValueError("pCR labels change across fold rows")
    test_counts = frame.loc[frame["split"].eq("test")].groupby("patient_id").size()
    if len(test_counts) != expected_n or not test_counts.eq(1).all():
        raise ValueError("each patient must be outer-test in exactly one of five folds")
    labels_by_id = frame.groupby("patient_id", sort=False)["label_pcr"].first().to_dict()
    labels = np.asarray([labels_by_id[value] for value in patient_ids], dtype=np.int64)
    frame = frame.sort_values(["fold", "patient_id"], kind="stable").reset_index(drop=True)
    return FoldManifest(patient_ids, labels, frame, digest)


@dataclass(frozen=True)
class ClinicalTable:
    """Only the preregistered pCR/biomarker/age/exact-arm fields."""

    patient_ids: np.ndarray
    pcr: np.ndarray
    hr: np.ndarray
    her2: np.ndarray
    mp: np.ndarray
    age: np.ndarray
    arm: np.ndarray
    subtype: np.ndarray
    sha256: str

    def subset(self, patient_ids: Sequence[str]) -> "ClinicalTable":
        requested = np.asarray(patient_ids, dtype=str)
        indices = _subset_indices(self.patient_ids, requested, label="clinical subset")
        return ClinicalTable(
            requested.copy(),
            self.pcr[indices].copy(),
            self.hr[indices].copy(),
            self.her2[indices].copy(),
            self.mp[indices].copy(),
            self.age[indices].copy(),
            self.arm[indices].copy(),
            self.subtype[indices].copy(),
            self.sha256,
        )


def load_clinical_labels(
    path: str | Path,
    *,
    expected_patient_ids: Sequence[str] | None = None,
    expected_n: int = COHORT_SIZE,
    expected_sha256: str | None = EXPECTED_CLINICAL_SHA256,
) -> ClinicalTable:
    """Load all 808 labels, rejecting missing biomarkers or patient drift."""

    source = Path(path).resolve()
    digest = _check_file_hash(source, expected_sha256, "clinical labels")
    frame = pd.read_csv(source, dtype={"patient_id": str})
    required = {
        "patient_id",
        "label_pcr",
        "label_hr",
        "label_her2",
        "label_mp",
        "age_at_screening",
        "arm",
        "hr_her2_subtype",
    }
    missing_columns = sorted(required.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"clinical labels are missing columns: {missing_columns}")
    patient_ids = _unique_patient_ids(
        frame["patient_id"].to_numpy(), expected_n=expected_n, label="clinical patient_id"
    )
    frame = frame.loc[:, list(required)].copy()
    frame["patient_id"] = patient_ids
    if "complete_4visits" in pd.read_csv(source, nrows=0).columns:
        complete = _strict_bool(
            pd.read_csv(source, usecols=["complete_4visits"])["complete_4visits"],
            "complete_4visits",
        )
        if not complete.all():
            raise ValueError("clinical table contains a patient without four complete visits")

    if expected_patient_ids is None:
        canonical = np.asarray(sorted(patient_ids.tolist()), dtype=str)
    else:
        canonical = _unique_patient_ids(
            expected_patient_ids, expected_n=expected_n, label="canonical patient IDs"
        )
    order = _alignment_indices(patient_ids, canonical, label="clinical labels")
    frame = frame.iloc[order].reset_index(drop=True)
    patient_ids = canonical.copy()
    pcr = _strict_binary(frame["label_pcr"], "clinical label_pcr")
    hr = _strict_binary(frame["label_hr"], "clinical label_hr")
    her2 = _strict_binary(frame["label_her2"], "clinical label_her2")
    mp = _strict_binary(frame["label_mp"], "clinical label_mp")
    age = pd.to_numeric(frame["age_at_screening"], errors="coerce").to_numpy(dtype=np.float64)
    if np.isinf(age).any() or not np.isfinite(age).any():
        raise ValueError("clinical age must contain finite values and no infinities")
    arm = _as_strings(frame["arm"].to_numpy(), "clinical exact treatment arm")
    subtype = _as_strings(frame["hr_her2_subtype"].to_numpy(), "HR/HER2 subtype")
    if set(subtype.tolist()) != set(HR_HER2_SUBTYPES):
        raise ValueError(f"HR/HER2 subtype classes must be exactly {HR_HER2_SUBTYPES}")
    expected_subtype = np.asarray(
        [
            f"HR{'+' if hr_value else '-'}/HER2{'+' if her2_value else '-'}"
            for hr_value, her2_value in zip(hr, her2, strict=True)
        ],
        dtype=str,
    )
    if not np.array_equal(subtype, expected_subtype):
        raise ValueError("HR/HER2 subtype disagrees with the binary HR/HER2 labels")
    return ClinicalTable(patient_ids, pcr, hr, her2, mp, age, arm, subtype, digest)


@dataclass(frozen=True)
class FoundationFeatureAsset:
    patient_ids: np.ndarray
    representation: np.ndarray
    spatial_axis: tuple[str, str]
    visits: tuple[str, str, str, str]
    model_name: str
    checkpoint_sha256: str | None
    config_sha256: str | None
    extraction_signature_sha256: str | None
    canonical_patient_order_sha256: str | None
    source_sha256: str

    @property
    def feature_dim(self) -> int:
        return int(self.representation.shape[-1])

    def spatial(self, spatial_axis: str) -> np.ndarray:
        axis = str(spatial_axis).strip().upper()
        if axis not in self.spatial_axis:
            raise ValueError(f"spatial axis must be one of {self.spatial_axis}")
        index = self.spatial_axis.index(axis)
        return np.ascontiguousarray(self.representation[:, :, index, :])


def _scalar_string(value: np.ndarray, label: str) -> str:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in {"U", "S", "O"}:
        raise ValueError(f"{label} must be a scalar string")
    text = str(array.item()).strip()
    if not text:
        raise ValueError(f"{label} must not be empty")
    return text


def _optional_digest(arrays: Mapping[str, np.ndarray], key: str) -> str | None:
    if key not in arrays:
        return None
    value = _scalar_string(arrays[key], key)
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{key} must be a lowercase SHA-256 digest")
    return value


def load_foundation_features(
    path: str | Path,
    *,
    expected_patient_ids: Sequence[str],
    expected_n: int = COHORT_SIZE,
) -> FoundationFeatureAsset:
    """Load the unified ``[N,4,2,D]`` GLOBAL/LOCAL foundation NPZ."""

    source = Path(path).resolve()
    required = {"patient_id", "representation", "spatial_axis", "visits", "model_name"}
    optional = {
        "checkpoint_sha256",
        "config_sha256",
        "extraction_signature_sha256",
        "canonical_patient_order_sha256",
    }
    with np.load(source, allow_pickle=False) as archive:
        observed = set(archive.files)
        if not required.issubset(observed) or not observed.issubset(required | optional):
            raise ValueError(
                "foundation feature NPZ schema drifted: "
                f"required={sorted(required)}, optional={sorted(optional)}, "
                f"observed={sorted(observed)}"
            )
        arrays = {key: archive[key] for key in archive.files}
    observed_ids = _unique_patient_ids(
        arrays["patient_id"], expected_n=expected_n, label="foundation patient_id"
    )
    expected_ids = _unique_patient_ids(
        expected_patient_ids, expected_n=expected_n, label="canonical patient IDs"
    )
    order = _alignment_indices(observed_ids, expected_ids, label="foundation features")
    representation = arrays["representation"]
    if (
        representation.ndim != 4
        or representation.shape[:3] != (expected_n, 4, 2)
        or representation.shape[3] <= 0
        or representation.dtype != np.dtype(np.float32)
    ):
        raise ValueError("representation must be float32 [N,4,2,D] with D > 0")
    if not np.isfinite(representation).all():
        raise FloatingPointError("foundation representation contains NaN/Inf")
    spatial_axis = tuple(_as_strings(arrays["spatial_axis"], "spatial_axis").tolist())
    visits = tuple(_as_strings(arrays["visits"], "visits").tolist())
    if spatial_axis != SPATIAL_AXES:
        raise ValueError(f"spatial_axis must be exactly {SPATIAL_AXES}")
    if visits != VISITS:
        raise ValueError(f"visits must be exactly {VISITS}")
    model_name = _scalar_string(arrays["model_name"], "model_name")
    if not _SAFE_MODEL_RE.fullmatch(model_name):
        raise ValueError("model_name must be a path-free safe identifier")
    locked_dimensions = {
        "medicalnet_resnet50_3dseg8": 14_336,
        "dino_vitb16_imagenet1k": 1_536,
    }
    if (
        model_name in locked_dimensions
        and representation.shape[3] != locked_dimensions[model_name]
    ):
        raise ValueError(
            f"{model_name} representation D must be {locked_dimensions[model_name]}"
        )
    canonical_digest = _optional_digest(arrays, "canonical_patient_order_sha256")
    if canonical_digest is not None:
        if canonical_digest != ordered_text_sha256(observed_ids):
            raise ValueError("foundation embedded canonical patient-order digest drifted")
        if canonical_digest != ordered_text_sha256(expected_ids):
            raise ValueError("foundation patient order is not the locked canonical order")
    aligned_representation = (
        np.ascontiguousarray(representation)
        if np.array_equal(order, np.arange(expected_n, dtype=np.int64))
        else np.ascontiguousarray(representation[order])
    )
    return FoundationFeatureAsset(
        expected_ids.copy(),
        aligned_representation,
        SPATIAL_AXES,
        VISITS,
        model_name,
        _optional_digest(arrays, "checkpoint_sha256"),
        _optional_digest(arrays, "config_sha256"),
        _optional_digest(arrays, "extraction_signature_sha256"),
        canonical_digest,
        file_sha256(source),
    )


@dataclass(frozen=True)
class CurrentCNNFeatureAsset:
    patient_ids: np.ndarray
    representation: np.ndarray
    model_name: str
    spatial_axis: str
    fold: int
    source_sha256: str
    metadata_sha256: str
    checkpoint_sha256: str
    selection_sha256: str
    current_data_contract_provenance_sha256: str


_CURRENT_CNN_METADATA_KEYS = {
    "arm",
    "checkpoint_data_provenance_sha256",
    "checkpoint_path",
    "checkpoint_sha256",
    "cohort",
    "current_data_contract_provenance_sha256",
    "experiment",
    "feature_dtype",
    "feature_implementation_sha256",
    "feature_path",
    "feature_sha256",
    "feature_shape",
    "feature_tensor",
    "fold",
    "ftv_head_called",
    "patient_order_sha256",
    "preregistration_lock",
    "preregistration_lock_sha256",
    "schema_version",
    "seed_base",
    "selected_epoch",
    "selection_path",
    "selection_sha256",
    "stage_a_sentinel_sha256",
    "test_labels_used",
    "train_patient_sha256",
    "validation_patient_sha256",
}

_CURRENT_CNN_METADATA_DIGEST_KEYS = {
    "checkpoint_data_provenance_sha256",
    "checkpoint_sha256",
    "current_data_contract_provenance_sha256",
    "feature_implementation_sha256",
    "feature_sha256",
    "patient_order_sha256",
    "preregistration_lock_sha256",
    "selection_sha256",
    "stage_a_sentinel_sha256",
    "train_patient_sha256",
    "validation_patient_sha256",
}


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is missing or invalid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _metadata_bound_file(payload: Mapping[str, Any], path_key: str, hash_key: str) -> Path:
    raw_path = payload.get(path_key)
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"current CNN metadata {path_key} must be a nonempty path")
    target = Path(raw_path).resolve()
    if not target.is_file():
        raise ValueError(f"current CNN metadata {path_key} does not exist")
    if file_sha256(target) != payload[hash_key]:
        raise ValueError(f"current CNN metadata {hash_key} binding drifted")
    return target


def _find_preregistration_lock(source: Path, reference: str) -> Path:
    if reference != "PREREGISTRATION_LOCK.json":
        raise ValueError("current CNN metadata preregistration_lock drifted")
    for parent in source.parents:
        candidate = parent / reference
        if candidate.is_file():
            return candidate.resolve()
    raise ValueError("current CNN preregistration lock is missing")


def _validate_current_cnn_metadata(
    source: Path,
    *,
    arrays: Mapping[str, np.ndarray],
    state_key: str,
    patient_id_key: str,
    split_key: str,
    model_name: str,
    spatial_axis: str,
    fold: int,
) -> tuple[str, str, str, str, str]:
    """Validate the adjacent export receipt and its linked provenance chain."""

    metadata_path = source.with_suffix(".metadata.json")
    payload = _read_json_object(metadata_path, "current CNN feature metadata")
    if set(payload) != _CURRENT_CNN_METADATA_KEYS:
        missing = sorted(_CURRENT_CNN_METADATA_KEYS.difference(payload))
        extra = sorted(set(payload).difference(_CURRENT_CNN_METADATA_KEYS))
        raise ValueError(
            "current CNN metadata schema drifted: "
            f"missing={missing}, extra={extra}"
        )
    for key in _CURRENT_CNN_METADATA_DIGEST_KEYS:
        value = payload[key]
        if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
            raise ValueError(f"current CNN metadata {key} must be a lowercase SHA-256")
    exact = {
        "schema_version": 1,
        "experiment": "local_global_response_state_pilot",
        "arm": model_name,
        "seed_base": SEED,
        "fold": fold,
        "feature_tensor": "online_preprojector_response_state",
        "feature_dtype": "float32",
        "cohort": "exact_locked_primary_train_validation_test",
    }
    for key, expected in exact.items():
        if type(payload[key]) is not type(expected) or payload[key] != expected:
            raise ValueError(f"current CNN metadata identity drifted at {key}")
    if payload["ftv_head_called"] is not False:
        raise ValueError("current CNN metadata must prove ftv_head_called=false")
    if payload["test_labels_used"] is not False:
        raise ValueError("current CNN metadata must prove test_labels_used=false")
    if (
        type(payload["selected_epoch"]) is not int
        or int(payload["selected_epoch"]) <= 0
    ):
        raise ValueError("current CNN metadata selected_epoch must be a positive integer")

    raw_feature_path = payload["feature_path"]
    if not isinstance(raw_feature_path, str) or Path(raw_feature_path).resolve() != source:
        raise ValueError("current CNN metadata feature_path differs from the input")
    source_sha256 = file_sha256(source)
    if payload["feature_sha256"] != source_sha256:
        raise ValueError("current CNN metadata feature_sha256 differs from the input")
    representation = np.asarray(arrays[state_key])
    if payload["feature_shape"] != list(representation.shape):
        raise ValueError("current CNN metadata feature_shape differs from the tensor")
    patient_ids = _as_strings(arrays[patient_id_key], "current CNN patient_id")
    if payload["patient_order_sha256"] != _newline_ordered_text_sha256(patient_ids):
        raise ValueError("current CNN metadata patient-order hash drifted")

    # GAP0/LOCAL0 encode the spatial identity in the arm name; the formal
    # receipt has no separate spatial field, so this is the fail-closed check.
    locked_spatial = {"GAP0": "GLOBAL", "LOCAL0": "LOCAL"}
    if model_name in locked_spatial and spatial_axis != locked_spatial[model_name]:
        raise ValueError(
            f"current CNN metadata arm {model_name} implies {locked_spatial[model_name]}"
        )

    checkpoint_path = _metadata_bound_file(
        payload, "checkpoint_path", "checkpoint_sha256"
    )
    selection_path = _metadata_bound_file(payload, "selection_path", "selection_sha256")
    selection = _read_json_object(selection_path, "current CNN linked selection")
    selection_exact = {
        "schema_version": 1,
        "arm": model_name,
        "seed_base": SEED,
        "fold": fold,
        "selected_epoch": payload["selected_epoch"],
        "data_provenance_sha256": payload["checkpoint_data_provenance_sha256"],
        "preregistration_lock_sha256": payload["preregistration_lock_sha256"],
        "stage_a_sentinel_sha256": payload["stage_a_sentinel_sha256"],
        "train_patient_sha256": payload["train_patient_sha256"],
        "val_patient_sha256": payload["validation_patient_sha256"],
        "test_data_used": False,
        "pcr_used": False,
        "delta_ftv_used": False,
    }
    for key, expected in selection_exact.items():
        if key not in selection or selection[key] != expected:
            raise ValueError(f"current CNN selection provenance drifted at {key}")
    nested_lock = selection.get("preregistration")
    if not isinstance(nested_lock, dict) or nested_lock.get("lock_sha256") != payload[
        "preregistration_lock_sha256"
    ]:
        raise ValueError("current CNN selection preregistration chain drifted")

    lock_path = _find_preregistration_lock(source, payload["preregistration_lock"])
    for linked_path, label in (
        (checkpoint_path, "checkpoint_path"),
        (selection_path, "selection_path"),
    ):
        try:
            linked_path.relative_to(lock_path.parent)
        except ValueError as error:
            raise ValueError(
                f"current CNN metadata {label} escapes the locked experiment root"
            ) from error
    if file_sha256(lock_path) != payload["preregistration_lock_sha256"]:
        raise ValueError("current CNN preregistration-lock hash binding drifted")
    lock = _read_json_object(lock_path, "current CNN preregistration lock")
    if (
        lock.get("schema_version") != 1
        or lock.get("experiment") != "local_global_response_state_pilot"
        or lock.get("status") != "FROZEN_BEFORE_FORMAL_RESULTS"
    ):
        raise ValueError("current CNN preregistration lock identity drifted")
    matrix = lock.get("matrix")
    if not isinstance(matrix, dict):
        raise ValueError("current CNN preregistration lock matrix is missing")
    if (
        model_name not in matrix.get("arms", [])
        or SEED not in matrix.get("seeds", [])
        or fold not in matrix.get("folds", [])
    ):
        raise ValueError("current CNN arm/fold/seed is outside the frozen matrix")
    upstream_hashes = lock.get("upstream_sha256")
    if (
        not isinstance(upstream_hashes, dict)
        or payload["stage_a_sentinel_sha256"] not in upstream_hashes.values()
    ):
        raise ValueError("current CNN Stage-A sentinel is not bound by the lock")
    code_hashes = lock.get("code_and_plan_sha256")
    if (
        not isinstance(code_hashes, dict)
        or payload["feature_implementation_sha256"] not in code_hashes.values()
    ):
        raise ValueError("current CNN feature implementation is not bound by the lock")

    # The split array is already compared against the locked fivefold manifest
    # by the caller.  Reassert its exact vocabulary here before trusting the
    # provenance receipt.
    splits = _as_strings(arrays[split_key], "current CNN split")
    if set(np.char.lower(splits).tolist()) != set(SPLITS):
        raise ValueError("current CNN metadata binds an invalid split vocabulary")
    return (
        source_sha256,
        file_sha256(metadata_path),
        payload["checkpoint_sha256"],
        payload["selection_sha256"],
        payload["current_data_contract_provenance_sha256"],
    )


def load_current_cnn_features(
    path: str | Path,
    *,
    fold: int,
    expected_patient_ids: Sequence[str],
    fold_manifest: FoldManifest | None = None,
    expected_labels: Sequence[int] | None = None,
    expected_n: int = COHORT_SIZE,
    model_name: str = "current_cnn",
    spatial_axis: str = "LOCAL",
) -> CurrentCNNFeatureAsset:
    """Load one fold-specific current-CNN ``[N,4,192]`` feature asset.

    Both established repository schemas (``patient_id/response_state/split``
    and ``patient_ids/response_state/splits``) are accepted, as is the new
    neutral ``patient_id/representation`` schema.  All embedded identities are
    checked when present.
    """

    fold = int(fold)
    if fold not in FOLDS:
        raise ValueError(f"fold must be one of {FOLDS}")
    requested_name = str(model_name).strip()
    source = Path(path).resolve()
    allowed = {
        "patient_id", "patient_ids", "representation", "response_state", "state",
        "split", "splits", "fold", "visits", "timepoints", "model", "model_name",
        "arm", "seed_base", "label_pcr", "state_valid", "checkpoint_sha256",
        "config_sha256",
    }
    with np.load(source, allow_pickle=False) as archive:
        observed = set(archive.files)
        if not observed.issubset(allowed):
            raise ValueError(f"current CNN NPZ has unexpected keys: {sorted(observed - allowed)}")
        formal_keys = {
            "patient_id",
            "split",
            "response_state",
            "arm",
            "seed_base",
            "fold",
        }
        if requested_name in {"GAP0", "LOCAL0"} and observed != formal_keys:
            raise ValueError(
                "formal GAP0/LOCAL0 current CNN NPZ schema drifted: "
                f"expected={sorted(formal_keys)}, observed={sorted(observed)}"
            )
        arrays = {key: archive[key] for key in archive.files}
    if "arm" not in arrays or "seed_base" not in arrays or "fold" not in arrays:
        raise ValueError("current CNN NPZ must embed arm, seed_base, and fold identities")
    id_keys = [key for key in ("patient_id", "patient_ids") if key in arrays]
    state_keys = [key for key in ("representation", "response_state", "state") if key in arrays]
    if len(id_keys) != 1 or len(state_keys) != 1:
        raise ValueError("current CNN NPZ must have exactly one patient-ID and state key")
    observed_ids = _unique_patient_ids(
        arrays[id_keys[0]], expected_n=expected_n, label="current CNN patient_id"
    )
    expected_ids = _unique_patient_ids(
        expected_patient_ids, expected_n=expected_n, label="canonical patient IDs"
    )
    order = _alignment_indices(observed_ids, expected_ids, label="current CNN features")
    representation = arrays[state_keys[0]]
    if representation.shape != (expected_n, 4, 192) or representation.dtype != np.float32:
        raise ValueError("current CNN representation must be float32 [N,4,192]")
    if not np.isfinite(representation).all():
        raise FloatingPointError("current CNN representation contains NaN/Inf")
    if "fold" in arrays and int(np.asarray(arrays["fold"]).item()) != fold:
        raise ValueError("current CNN embedded fold does not match requested fold")
    if int(np.asarray(arrays["seed_base"]).item()) != SEED:
        raise ValueError("current CNN seed_base must be the locked seed 2026")
    visit_key = "visits" if "visits" in arrays else "timepoints" if "timepoints" in arrays else None
    if visit_key is not None and tuple(_as_strings(arrays[visit_key], visit_key)) != VISITS:
        raise ValueError(f"current CNN {visit_key} must be exactly {VISITS}")
    split_keys = [key for key in ("split", "splits") if key in arrays]
    if len(split_keys) != 1:
        raise ValueError("current CNN NPZ must have exactly one embedded split key")
    embedded = _as_strings(arrays[split_keys[0]], "current CNN split")[order]
    if fold_manifest is None:
        raise ValueError("embedded CNN splits require a FoldManifest for verification")
    expected_roles = fold_manifest.roles(fold, expected_ids)
    if not np.array_equal(np.char.lower(embedded), expected_roles):
        raise ValueError("current CNN embedded split assignment drifted")
    if "label_pcr" in arrays:
        embedded_labels = np.asarray(arrays["label_pcr"])[order]
        if expected_labels is None:
            raise ValueError("embedded CNN labels require expected_labels for verification")
        labels = np.asarray(expected_labels, dtype=np.int64)
        if labels.shape != (expected_n,) or not np.array_equal(embedded_labels, labels):
            raise ValueError("current CNN embedded pCR labels drifted")
    if "state_valid" in arrays:
        valid = np.asarray(arrays["state_valid"])[order]
        if valid.dtype != np.bool_ or valid.shape != (expected_n, 4) or not valid.all():
            raise ValueError("current CNN state_valid must be all-true boolean [N,4]")
    spatial = str(spatial_axis).strip().upper()
    if spatial not in SPATIAL_AXES:
        raise ValueError(f"spatial_axis must be one of {SPATIAL_AXES}")
    name = requested_name
    if not _SAFE_MODEL_RE.fullmatch(name):
        raise ValueError("current CNN model_name must be a path-free safe identifier")
    embedded_arm = _scalar_string(arrays["arm"], "current CNN arm")
    if embedded_arm != name:
        raise ValueError("current CNN embedded arm must exactly match model_name")
    locked_spatial = {"GAP0": "GLOBAL", "LOCAL0": "LOCAL"}
    if name in locked_spatial and spatial != locked_spatial[name]:
        raise ValueError(f"current CNN {name} must use spatial={locked_spatial[name]}")
    (
        source_sha256,
        metadata_sha256,
        checkpoint_sha256,
        selection_sha256,
        current_data_contract_provenance_sha256,
    ) = _validate_current_cnn_metadata(
        source,
        arrays=arrays,
        state_key=state_keys[0],
        patient_id_key=id_keys[0],
        split_key=split_keys[0],
        model_name=name,
        spatial_axis=spatial,
        fold=fold,
    )
    aligned_representation = (
        np.ascontiguousarray(representation)
        if np.array_equal(order, np.arange(expected_n, dtype=np.int64))
        else np.ascontiguousarray(representation[order])
    )
    return CurrentCNNFeatureAsset(
        expected_ids.copy(),
        aligned_representation,
        name,
        spatial,
        fold,
        source_sha256,
        metadata_sha256,
        checkpoint_sha256,
        selection_sha256,
        current_data_contract_provenance_sha256,
    )


@dataclass(frozen=True)
class RadiomicsTable:
    patient_ids: np.ndarray
    values: np.ndarray
    feature_names: tuple[str, ...]
    sha256: str

    def aligned_values(
        self, patient_ids: Sequence[str], feature_names: Sequence[str] | None = None
    ) -> np.ndarray:
        requested = np.asarray(patient_ids, dtype=str)
        indices = _subset_indices(self.patient_ids, requested, label="radiomics subset")
        names = self.feature_names if feature_names is None else tuple(feature_names)
        unknown = sorted(set(names).difference(self.feature_names))
        if unknown:
            raise ValueError(f"unknown radiomics features: {unknown}")
        columns = [self.feature_names.index(name) for name in names]
        return np.ascontiguousarray(self.values[indices][:, :, columns])


def load_radiomics_table(
    path: str | Path,
    *,
    cohort_patient_ids: Sequence[str],
    expected_n: int = RADIOMICS_COMPLETE_CASE_SIZE,
    expected_sha256: str | None = EXPECTED_RADIOMICS_SHA256,
) -> RadiomicsTable:
    """Reconstruct the complete four-visit FTV/SPH/LD/BPE table.

    A patient is never dropped for a malformed row.  Instead, any missing
    transition, invalid flag, discontinuous endpoint, or non-finite value
    aborts the complete-case analysis.
    """

    source = Path(path).resolve()
    digest = _check_file_hash(source, expected_sha256, "radiomics table")
    frame = pd.read_csv(source, dtype={"patient_id": str})
    required = {"patient_id", "start_visit", "end_visit"}
    for name in RADIOMIC_FEATURES:
        required.update(
            {f"{name}_start", f"{name}_end", f"{name}_absolute_change", f"{name}_valid"}
        )
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"radiomics table is missing columns: {missing}")
    frame = frame.loc[:, sorted(required)].copy()
    frame["patient_id"] = _as_strings(frame["patient_id"].to_numpy(), "radiomics patient_id")
    cohort_ids = np.asarray(cohort_patient_ids, dtype=str)
    if len(set(cohort_ids.tolist())) != len(cohort_ids):
        raise ValueError("cohort patient IDs are duplicated")
    patient_set = set(frame["patient_id"].tolist())
    if not patient_set.issubset(set(cohort_ids.tolist())):
        raise ValueError("radiomics table contains a patient outside the 808-person cohort")
    if len(patient_set) != expected_n:
        raise ValueError(f"radiomics table must contain exactly {expected_n} patients")
    counts = frame.groupby("patient_id").size()
    if not counts.eq(3).all() or len(frame) != expected_n * 3:
        raise ValueError("each radiomics patient must have exactly three transitions")

    ordered_ids = np.asarray([value for value in cohort_ids if value in patient_set], dtype=str)
    if len(ordered_ids) != expected_n:
        raise AssertionError("radiomics canonical ordering lost a complete-case patient")
    values = np.empty((expected_n, 4, len(RADIOMIC_FEATURES)), dtype=np.float64)
    pairs = tuple(zip(VISITS[:-1], VISITS[1:], strict=True))
    for patient_index, patient_id in enumerate(ordered_ids):
        rows = frame.loc[frame["patient_id"].eq(patient_id)]
        lookup = {
            (str(row.start_visit), str(row.end_visit)): row
            for row in rows.itertuples(index=False)
        }
        if set(lookup) != set(pairs) or len(lookup) != len(pairs):
            raise ValueError(f"radiomics transition coverage drifted for {patient_id}")
        for feature_index, name in enumerate(RADIOMIC_FEATURES):
            prior_end: float | None = None
            for transition_index, pair in enumerate(pairs):
                row = lookup[pair]
                if not bool(_strict_bool(pd.Series([getattr(row, f"{name}_valid")]), name)[0]):
                    raise ValueError(f"radiomics {name} is invalid for {patient_id}/{pair}")
                start = float(getattr(row, f"{name}_start"))
                end = float(getattr(row, f"{name}_end"))
                change = float(getattr(row, f"{name}_absolute_change"))
                if not all(math.isfinite(item) for item in (start, end, change)):
                    raise FloatingPointError(f"radiomics {name} is non-finite")
                if not math.isclose(change, end - start, rel_tol=1e-7, abs_tol=1e-9):
                    raise ValueError(f"radiomics {name} change identity drifted")
                if prior_end is not None and not math.isclose(
                    start, prior_end, rel_tol=1e-7, abs_tol=1e-9
                ):
                    raise ValueError(f"radiomics {name} visit chain is discontinuous")
                if transition_index == 0:
                    values[patient_index, 0, feature_index] = start
                values[patient_index, transition_index + 1, feature_index] = end
                prior_end = end
    if not np.isfinite(values).all():
        raise FloatingPointError("reconstructed radiomics values contain NaN/Inf")
    return RadiomicsTable(ordered_ids, values, RADIOMIC_FEATURES, digest)


__all__ = [
    "COHORT_SIZE",
    "ClinicalTable",
    "CurrentCNNFeatureAsset",
    "EXPECTED_CLINICAL_SHA256",
    "EXPECTED_FOLD_MANIFEST_SHA256",
    "EXPECTED_RADIOMICS_SHA256",
    "FOLDS",
    "FoldManifest",
    "FoundationFeatureAsset",
    "HR_HER2_SUBTYPES",
    "RADIOMIC_FEATURES",
    "RADIOMICS_COMPLETE_CASE_SIZE",
    "RadiomicsTable",
    "SEED",
    "SPATIAL_AXES",
    "SPLITS",
    "VISITS",
    "file_sha256",
    "load_clinical_labels",
    "load_current_cnn_features",
    "load_fold_manifest",
    "load_foundation_features",
    "load_radiomics_table",
    "ordered_text_sha256",
]

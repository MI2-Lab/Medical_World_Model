"""Fail-closed data and timing contracts for the classical DCE experiment.

This module deliberately reads the two authoritative workbooks instead of a
pre-merged table containing outcomes beside predictors.  Patient identifiers
are retained only in memory and in gitignored ``*.private`` artifacts.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
VISITS = ("T0", "T1", "T2", "T3")
FAMILIES = ("FTV", "LD", "SPH", "BPE")
NONFTV_FAMILIES = ("LD", "SPH", "BPE")
SUBTYPE_ORDER = ("HR-/HER2-", "HR-/HER2+", "HR+/HER2-", "HR+/HER2+")

RAW_COLUMNS: dict[str, dict[str, str]] = {
    "FTV": {
        "T0": "VOLUME_TUM_BLU_V10",
        "T1": "VOLUME_TUM_BLU_V20",
        "T2": "VOLUME_TUM_BLU_V30",
        "T3": "VOLUME_TUM_BLU_V40",
    },
    "LD": {visit: f"LD_{visit}" for visit in VISITS},
    "SPH": {visit: f"SPHERICITY_{visit}" for visit in VISITS},
    "BPE": {visit: f"BPE_5slice_mean_{visit}" for visit in VISITS},
}

DERIVED_COLUMNS: dict[str, dict[str, str]] = {
    "FTV": {visit: f"FTV_pch_T0_{visit}" for visit in VISITS[1:]},
    "LD": {visit: f"LD_pch_T0_{visit}" for visit in VISITS[1:]},
    "SPH": {visit: f"Sphericity_pch_T0_{visit}" for visit in VISITS[1:]},
    "BPE": {visit: f"BPE_pch_T0_{visit}" for visit in VISITS[1:]},
}

EXPECTED_RADIOMICS_COLUMNS = (
    ["CLINICAL-TRIAL-SUBJECT-ID"]
    + [RAW_COLUMNS[family][visit] for family in ("FTV", "SPH", "LD", "BPE") for visit in VISITS]
    + [DERIVED_COLUMNS[family][visit] for family in ("FTV", "SPH", "LD", "BPE") for visit in VISITS[1:]]
)


@dataclass(frozen=True)
class SplitSpec:
    fold: int
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    protocol: str


@dataclass(frozen=True)
class FeatureFrame:
    values: pd.DataFrame
    metadata: pd.DataFrame


def load_config(path: str | Path | None = None) -> dict:
    config_path = Path(path) if path else EXPERIMENT_ROOT / "configs" / "experiment.json"
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sha256(path: str | Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if expected and actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, observed {actual}")
    return actual


def canonical_trial_id(value: object) -> str:
    """Return an exact six-digit trial ID; reject approximate representations."""

    if pd.isna(value):
        raise ValueError("missing trial ID")
    if isinstance(value, (int, np.integer)):
        text = str(int(value))
    elif isinstance(value, (float, np.floating)) and float(value).is_integer():
        text = str(int(value))
    else:
        text = str(value).strip()
    if not re.fullmatch(r"\d{6}", text):
        raise ValueError(f"trial ID is not exactly six digits: {value!r}")
    return text


def canonical_mri_trial_id(value: object) -> str:
    text = str(value).strip()
    match = re.fullmatch(r"(?:ISPY2-|ACRIN-6698-)(\d{6})", text)
    if not match:
        raise ValueError(f"MRI patient ID violates the exact prefix/six-digit contract: {value!r}")
    return match.group(1)


def subtype_label(hr: int, her2: int) -> str:
    return f"HR{'+' if int(hr) else '-'}/HER2{'+' if int(her2) else '-'}"


def simplify_race(value: object) -> object:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if ";" in text or "," in text:
        return "Multiple"
    if text == "Native Hawaiian or Other Pacific Islande":
        return "Native Hawaiian or Pacific Islander"
    return text


def simplify_menopausal_status(value: object) -> object:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text.startswith("Premenopausal"):
        return "Premenopausal"
    if text.startswith("Perimenopausal"):
        return "Perimenopausal"
    if text.startswith("Postmenopausal"):
        return "Postmenopausal"
    if text == "Above categories not applicable AND Age > 50":
        return "Other_age_gt_50"
    if text == "Above categories not applicable AND Age < 50":
        return "Other_age_lt_50"
    raise ValueError(f"unrecognized menopausal-status value: {value!r}")


def _assert_finite_or_missing(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    numeric = frame[list(columns)].apply(pd.to_numeric, errors="raise")
    values = numeric.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError(f"{label} contains infinite values")


def load_primary_cohort(config: Mapping) -> tuple[pd.DataFrame, dict]:
    """Load and one-to-one join all 384 workbook patients to raw clinical data."""

    source = config["source"]
    rad_path = Path(source["radiomics_workbook"])
    clinical_path = Path(source["clinical_workbook"])
    rad_sha = verify_sha256(rad_path, source.get("radiomics_sha256", ""), "radiomics workbook")
    clinical_sha = verify_sha256(clinical_path, source.get("clinical_sha256", ""), "clinical workbook")

    xls = pd.ExcelFile(rad_path)
    if xls.sheet_names != [source["radiomics_sheet"]]:
        raise ValueError(f"unexpected radiomics sheets: {xls.sheet_names}")
    radiomics = pd.read_excel(rad_path, sheet_name=source["radiomics_sheet"])
    if list(radiomics.columns) != EXPECTED_RADIOMICS_COLUMNS:
        raise ValueError("radiomics schema/order differs from the preregistered 29-column contract")
    if len(radiomics) != 384:
        raise ValueError(f"expected 384 radiomics rows, observed {len(radiomics)}")
    radiomics = radiomics.copy()
    radiomics["trial_id"] = radiomics["CLINICAL-TRIAL-SUBJECT-ID"].map(canonical_trial_id)
    if radiomics["trial_id"].duplicated().any():
        raise ValueError("duplicate radiomics patient ID")
    _assert_finite_or_missing(radiomics, EXPECTED_RADIOMICS_COLUMNS[1:], "radiomics workbook")

    clinical = pd.read_excel(clinical_path, sheet_name=source["clinical_sheet"]).copy()
    expected_clinical = {
        "Patient_ID", "Arm", "HR", "HER2", "MP", "pCR", "Age_at_Screening",
        "Race", "menopausal_status", "ethnicity",
    }
    if set(clinical.columns) != expected_clinical:
        raise ValueError(f"clinical schema differs from contract: {list(clinical.columns)}")
    clinical["trial_id"] = clinical["Patient_ID"].map(canonical_trial_id)
    if clinical["trial_id"].duplicated().any():
        raise ValueError("duplicate clinical patient ID")

    cohort = radiomics.merge(
        clinical.drop(columns=["Patient_ID"]), on="trial_id", how="left", validate="one_to_one", indicator=True
    )
    if not cohort["_merge"].eq("both").all():
        raise ValueError("not every radiomics patient has an exact clinical match")
    cohort = cohort.drop(columns=["_merge"]).sort_values("trial_id").reset_index(drop=True)

    for target in ("pCR", "HR", "HER2", "MP"):
        if cohort[target].isna().any() or not set(cohort[target].astype(int).unique()).issubset({0, 1}):
            raise ValueError(f"invalid or missing binary target {target}")
        cohort[target] = cohort[target].astype(int)
    if cohort["Age_at_Screening"].isna().any():
        raise ValueError("age is unexpectedly missing in the 384-patient matched cohort")
    cohort["subtype"] = [subtype_label(hr, her2) for hr, her2 in zip(cohort["HR"], cohort["HER2"])]
    cohort["race_simple"] = cohort["Race"].map(simplify_race)
    cohort["menopausal_status_simple"] = cohort["menopausal_status"].map(simplify_menopausal_status)
    if not set(cohort["subtype"]).issubset(SUBTYPE_ORDER):
        raise ValueError("unexpected subtype")

    provenance = {
        "radiomics_path": str(rad_path),
        "radiomics_sha256": rad_sha,
        "clinical_path": str(clinical_path),
        "clinical_sha256": clinical_sha,
        "n": int(len(cohort)),
        "n_pcr_positive": int(cohort["pCR"].sum()),
        "n_hr_positive": int(cohort["HR"].sum()),
        "n_her2_positive": int(cohort["HER2"].sum()),
    }
    return cohort, provenance


def make_primary_splits(cohort: pd.DataFrame, config: Mapping) -> list[SplitSpec]:
    cv = config["outer_cv"]
    seed = int(config["seed"])
    n_splits = int(cv["n_splits"])
    strata = (
        cohort["pCR"].astype(str)
        + "_" + cohort["HR"].astype(str)
        + "_" + cohort["HER2"].astype(str)
    )
    outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits: list[SplitSpec] = []
    for fold, (development, test) in enumerate(outer.split(np.zeros(len(cohort)), strata)):
        inner = StratifiedShuffleSplit(
            n_splits=1,
            test_size=float(cv["validation_fraction"]),
            random_state=seed + 1009 * (fold + 1),
        )
        local_train, local_validation = next(
            inner.split(np.zeros(len(development)), strata.iloc[development])
        )
        train = development[local_train]
        validation = development[local_validation]
        _validate_disjoint_cover(train, validation, test, len(cohort), fold)
        splits.append(SplitSpec(fold, train, validation, test, "primary_stratified_384"))
    _validate_oof_test_once(splits, len(cohort))
    return splits


def make_mri_matched_splits(cohort: pd.DataFrame, config: Mapping) -> tuple[pd.DataFrame, list[SplitSpec]]:
    """Apply the existing 808-patient manifest to the exact 375-patient overlap."""

    path = Path(config["source"]["mri_fold_manifest"])
    verify_sha256(path, config["source"].get("mri_fold_manifest_sha256", ""), "MRI fold manifest")
    manifest = pd.read_csv(path)
    if list(manifest.columns) != ["patient_id", "fold", "split", "label_pcr"]:
        raise ValueError("unexpected MRI fold manifest schema")
    manifest = manifest.copy()
    manifest["trial_id"] = manifest["patient_id"].map(canonical_mri_trial_id)
    overlap_ids = sorted(set(cohort["trial_id"]) & set(manifest["trial_id"]))
    matched = cohort.set_index("trial_id").loc[overlap_ids].reset_index()
    if len(matched) != 375:
        raise ValueError(f"expected MRI-matched n=375, observed {len(matched)}")
    position = {trial_id: i for i, trial_id in enumerate(matched["trial_id"])}
    splits: list[SplitSpec] = []
    for fold in sorted(manifest["fold"].unique()):
        current = manifest[(manifest["fold"] == fold) & manifest["trial_id"].isin(overlap_ids)].copy()
        if len(current) != len(matched) or current["trial_id"].duplicated().any():
            raise ValueError(f"invalid MRI matched manifest rows in fold {fold}")
        indices: dict[str, np.ndarray] = {}
        for split_name in ("train", "val", "test"):
            ids = current.loc[current["split"] == split_name, "trial_id"]
            indices[split_name] = np.array([position[x] for x in ids], dtype=int)
        _validate_disjoint_cover(indices["train"], indices["val"], indices["test"], len(matched), int(fold))
        observed = matched.iloc[[position[x] for x in current["trial_id"]]]["pCR"].to_numpy()
        if not np.array_equal(observed.astype(int), current["label_pcr"].to_numpy(dtype=int)):
            raise ValueError(f"pCR labels disagree with MRI manifest in fold {fold}")
        splits.append(
            SplitSpec(int(fold), indices["train"], indices["val"], indices["test"], "locked_mri_manifest_375")
        )
    _validate_oof_test_once(splits, len(matched))
    return matched, splits


def _validate_disjoint_cover(
    train: Sequence[int], validation: Sequence[int], test: Sequence[int], n: int, fold: int
) -> None:
    sets = [set(map(int, values)) for values in (train, validation, test)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError(f"split overlap in fold {fold}")
    if set.union(*sets) != set(range(n)):
        raise ValueError(f"split does not cover cohort in fold {fold}")


def _validate_oof_test_once(splits: Sequence[SplitSpec], n: int) -> None:
    counts = np.zeros(n, dtype=int)
    for split in splits:
        counts[split.test] += 1
    if not np.all(counts == 1):
        raise ValueError("each patient must occur in outer test exactly once")


def visible_visits(timing: str) -> tuple[str, ...]:
    if timing not in VISITS:
        raise ValueError(f"unknown timing {timing}")
    return VISITS[: VISITS.index(timing) + 1]


def build_feature_frame(
    cohort: pd.DataFrame,
    timing: str,
    view: str,
    families: Sequence[str],
) -> FeatureFrame:
    """Build an allowlisted timing-safe feature matrix from absolute endpoints."""

    requested = tuple(families)
    if not requested or not set(requested).issubset(FAMILIES):
        raise ValueError(f"invalid feature families: {requested}")
    visits = visible_visits(timing)
    if view not in {"static", "longitudinal"}:
        raise ValueError(f"invalid view {view}")

    values: dict[str, pd.Series] = {}
    metadata: list[dict] = []
    for family in requested:
        absolute_visits = (timing,) if view == "static" else visits
        for visit in absolute_visits:
            source = RAW_COLUMNS[family][visit]
            name = f"{family}__absolute__{visit}"
            values[name] = pd.to_numeric(cohort[source], errors="raise")
            metadata.append(
                {
                    "feature": name,
                    "family": family,
                    "role": "absolute",
                    "start_visit": visit,
                    "end_visit": visit,
                    "source_columns": source,
                    "transform": "log1p" if family in {"FTV", "LD", "BPE"} else "identity",
                }
            )
        if view == "longitudinal":
            baseline = pd.to_numeric(cohort[RAW_COLUMNS[family]["T0"]], errors="raise")
            for visit in visits[1:]:
                current = pd.to_numeric(cohort[RAW_COLUMNS[family][visit]], errors="raise")
                absolute_name = f"{family}__delta__T0_{visit}"
                relative_name = f"{family}__relative_pct__T0_{visit}"
                values[absolute_name] = current - baseline
                values[relative_name] = 100.0 * (current - baseline) / baseline.abs()
                common = {
                    "family": family,
                    "start_visit": "T0",
                    "end_visit": visit,
                    "source_columns": f"{RAW_COLUMNS[family]['T0']}|{RAW_COLUMNS[family][visit]}",
                    "transform": "identity",
                }
                metadata.append({"feature": absolute_name, "role": "absolute_change", **common})
                metadata.append({"feature": relative_name, "role": "relative_change_pct", **common})

    frame = pd.DataFrame(values, index=cohort.index)
    meta = pd.DataFrame(metadata)
    if not np.isfinite(frame.to_numpy(dtype=float)[~frame.isna().to_numpy()]).all():
        raise ValueError("constructed features contain infinity")
    max_visit = VISITS.index(timing)
    if any(VISITS.index(end) > max_visit for end in meta["end_visit"]):
        raise AssertionError("future feature leaked through timing builder")
    return FeatureFrame(frame, meta)


def feature_set_families(model_name: str) -> tuple[str, ...]:
    mapping = {
        "F": ("FTV",),
        "N": NONFTV_FAMILIES,
        "FULL": FAMILIES,
        "D": ("LD",),
        "S": ("SPH",),
        "B": ("BPE",),
    }
    if model_name not in mapping:
        raise ValueError(f"unknown radiomics feature set {model_name}")
    return mapping[model_name]


def clinical_frame(cohort: pd.DataFrame, config: Mapping) -> pd.DataFrame:
    contract = config["clinical_contract"]
    columns = list(contract["numeric"]) + list(contract["categorical"])
    frame = cohort[columns].copy()
    for column in contract["numeric"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in contract["categorical"]:
        frame[column] = frame[column].where(frame[column].notna(), "__MISSING__").astype(str)
    return frame


def make_clinical_preprocessor(config: Mapping) -> ColumnTransformer:
    numeric = list(config["clinical_contract"]["numeric"])
    categorical = list(config["clinical_contract"]["categorical"])
    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)],
        remainder="drop",
        sparse_threshold=0.0,
    )


def private_split_frame(cohort: pd.DataFrame, splits: Sequence[SplitSpec]) -> pd.DataFrame:
    rows: list[dict] = []
    for split in splits:
        for label, indices in (("train", split.train), ("validation", split.validation), ("test", split.test)):
            for index in indices:
                rows.append(
                    {
                        "trial_id": cohort.iloc[int(index)]["trial_id"],
                        "fold": split.fold,
                        "split": label,
                        "protocol": split.protocol,
                    }
                )
    return pd.DataFrame(rows)


def aggregate_split_manifest(cohort: pd.DataFrame, splits: Sequence[SplitSpec]) -> pd.DataFrame:
    rows: list[dict] = []
    for split in splits:
        for label, indices in (("train", split.train), ("validation", split.validation), ("test", split.test)):
            subset = cohort.iloc[indices]
            rows.append(
                {
                    "protocol": split.protocol,
                    "fold": split.fold,
                    "split": label,
                    "n": len(subset),
                    "pCR_positive": int(subset["pCR"].sum()),
                    "HR_positive": int(subset["HR"].sum()),
                    "HER2_positive": int(subset["HER2"].sum()),
                }
            )
    return pd.DataFrame(rows)

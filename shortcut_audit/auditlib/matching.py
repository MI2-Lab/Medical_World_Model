"""Outcome-blind donor matching for the matched follow-up shortcut audit.

The matcher intentionally works from a small, explicitly selected set of
baseline columns.  Outcome columns may be present in the input frame, but are
never copied or inspected.  They are rejected if a caller attempts to use one
as a matching feature (or aliases one as a required matching column).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


_OUTCOME_EXACT_NAMES = frozenset(
    {
        "pcr",
        "label",
        "target",
        "outcome",
        "endpoint",
        "y",
        "y_true",
        "ytrue",
        "rcb",
        "pathologic_complete_response",
        "pathological_complete_response",
        "overall_survival",
        "disease_free_survival",
        "progression_free_survival",
    }
)
_OUTCOME_TOKENS = frozenset(
    {
        "pcr",
        "label",
        "target",
        "outcome",
        "endpoint",
        "response",
        "rcb",
        "survival",
        "death",
        "mortality",
        "recurrence",
        "relapse",
        "event",
        "prognosis",
    }
)


MAPPING_COLUMNS = (
    "recipient_patient_id",
    "donor_patient_id",
    "fold",
    "subtype",
    "donor_subtype",
    "treatment_family",
    "donor_treatment_family",
    "baseline_lesion_volume",
    "recipient_baseline_lesion_volume",
    "donor_baseline_lesion_volume",
    "matching_distance",
    "volume_distance_z",
    "age_distance_z",
    "mammaprint_mismatch",
    "extra_feature_distance",
    "recipient_age",
    "donor_age",
    "recipient_mammaprint",
    "donor_mammaprint",
    "subtype_match",
    "treatment_family_match",
    "visit_availability_compatible",
    "matching_level",
    "audit_repetition",
    "matching_seed",
)

DIAGNOSTIC_COLUMNS = (
    "recipient_patient_id",
    "fold",
    "status",
    "failure_reason",
    "requested_donors",
    "selected_donors",
    "n_same_fold_nonself",
    "n_visit_compatible",
    "n_hard_candidates",
    "n_eligible_candidates",
)

BALANCE_COLUMNS = (
    "scope",
    "fold",
    "matching_level",
    "n_pairs",
    "mean_abs_baseline_volume_difference",
    "median_abs_baseline_volume_difference",
    "mean_volume_distance_z",
    "baseline_volume_standardized_mean_difference",
    "mean_abs_age_difference",
    "age_standardized_mean_difference",
    "mammaprint_match_rate",
    "subtype_match_rate",
    "treatment_family_match_rate",
)


@dataclass(frozen=True)
class MatchingConfig:
    """Configuration for held-out-fold donor matching.

    Strict hard matching is the default: HR/HER2 subtype, treatment family,
    and visit availability must match.  ``allow_relaxed_matches=True`` is an
    explicit opt-in to fill remaining repetitions from documented lower
    matching levels.  Even then, visit compatibility and fold isolation are
    never relaxed.
    """

    patient_id_col: str = "patient_id"
    fold_col: str = "fold"
    hr_col: str = "hr"
    her2_col: str = "her2"
    treatment_family_col: str = "treatment_family"
    baseline_volume_col: str = "baseline_lesion_volume"
    visit_availability_cols: tuple[str, ...] = ("has_t1", "has_t2")
    subtype_col: str | None = None
    age_col: str | None = "age"
    mammaprint_col: str | None = "mammaprint"
    matching_features: tuple[str, ...] = ()
    max_donors: int = 10
    seed: int = 1729
    allow_relaxed_matches: bool = False
    age_weight: float = 0.05
    mammaprint_weight: float = 0.05
    extra_feature_weight: float = 0.05


@dataclass(frozen=True)
class MatchingResult:
    """Complete matching output, including non-silent failure reporting."""

    mapping: pd.DataFrame
    failures: pd.DataFrame
    recipient_diagnostics: pd.DataFrame
    balance_stats: pd.DataFrame
    success_stats: Mapping[str, Any]

    @property
    def donor_mapping(self) -> pd.DataFrame:
        """Alias useful for callers that prefer the audit terminology."""

        return self.mapping


def _normalise_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def _looks_like_outcome(name: str) -> bool:
    normalised = _normalise_column_name(name)
    tokens = frozenset(token for token in normalised.split("_") if token)
    collapsed = normalised.replace("_", "")
    return (
        normalised in _OUTCOME_EXACT_NAMES
        or bool(tokens & _OUTCOME_TOKENS)
        or "pcr" in collapsed
    )


def validate_matching_features(matching_features: Sequence[str] | None) -> tuple[str, ...]:
    """Validate and de-duplicate caller-provided, outcome-blind features.

    The check is intentionally based on names rather than values, so it never
    needs to inspect a pCR/outcome column in the input frame.
    """

    if matching_features is None:
        return ()
    if isinstance(matching_features, (str, bytes)):
        raise TypeError("matching_features 必须是列名序列，不能是单个字符串")

    validated: list[str] = []
    for feature in matching_features:
        if not isinstance(feature, str) or not feature.strip():
            raise TypeError("matching_features 中的每一项都必须是非空字符串")
        if _looks_like_outcome(feature):
            raise ValueError(
                f"禁止将 outcome/label 列 {feature!r} 用于 donor matching；"
                "匹配只能使用 baseline 可观测变量"
            )
        if feature not in validated:
            validated.append(feature)
    return tuple(validated)


def _validate_config(config: MatchingConfig, features: Sequence[str]) -> None:
    if not isinstance(config.max_donors, int) or isinstance(config.max_donors, bool):
        raise TypeError("max_donors 必须是正整数")
    if config.max_donors <= 0:
        raise ValueError("max_donors 必须大于 0")
    if not isinstance(config.seed, int) or isinstance(config.seed, bool):
        raise TypeError("seed 必须是整数")
    for name, weight in (
        ("age_weight", config.age_weight),
        ("mammaprint_weight", config.mammaprint_weight),
        ("extra_feature_weight", config.extra_feature_weight),
    ):
        if not math.isfinite(float(weight)) or float(weight) < 0:
            raise ValueError(f"{name} 必须是有限的非负数")

    consumed_columns = [
        config.patient_id_col,
        config.fold_col,
        config.hr_col,
        config.her2_col,
        config.treatment_family_col,
        config.baseline_volume_col,
        *config.visit_availability_cols,
        *features,
    ]
    for optional in (config.subtype_col, config.age_col, config.mammaprint_col):
        if optional is not None:
            consumed_columns.append(optional)
    for column in consumed_columns:
        if not isinstance(column, str) or not column.strip():
            raise TypeError("所有 matching 列名都必须是非空字符串")
        if _looks_like_outcome(column):
            raise ValueError(
                f"配置试图把 outcome/label 列 {column!r} 用于 donor matching"
            )


def _ordered_unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _is_missing(value: Any) -> bool:
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def _same_observed_value(left: Any, right: Any) -> bool:
    if _is_missing(left) or _is_missing(right):
        return False
    try:
        equal = left == right
    except (TypeError, ValueError):
        return False
    return bool(equal) if isinstance(equal, (bool, np.bool_)) else False


def _finite_float(value: Any) -> float | None:
    if _is_missing(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _scale(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    numeric = numeric[np.isfinite(numeric)]
    if numeric.size < 2:
        return 1.0
    standard_deviation = float(np.std(numeric, ddof=0))
    return standard_deviation if standard_deviation > 1e-12 else 1.0


def _availability_flag(value: Any) -> bool | None:
    if _is_missing(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = _finite_float(value)
        if number == 0.0:
            return False
        if number == 1.0:
            return True
        return None
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised in {"1", "true", "yes", "y", "available", "present"}:
            return True
        if normalised in {"0", "false", "no", "n", "unavailable", "missing", "absent"}:
            return False
    return None


def _availability_compatible(
    recipient: Mapping[str, Any],
    donor: Mapping[str, Any],
    columns: Sequence[str],
) -> bool:
    """Return whether the donor contains every visit needed by the recipient."""

    for column in columns:
        recipient_value = recipient[column]
        donor_value = donor[column]
        if _is_missing(recipient_value) or _is_missing(donor_value):
            return False
        recipient_flag = _availability_flag(recipient_value)
        donor_flag = _availability_flag(donor_value)
        if recipient_flag is not None and donor_flag is not None:
            if recipient_flag and not donor_flag:
                return False
        elif not _same_observed_value(recipient_value, donor_value):
            # Non-boolean availability encodings (for example a visit-pattern
            # category) must match exactly.
            return False
    return True


def _format_receptor(prefix: str, value: Any) -> str:
    if _is_missing(value):
        return f"{prefix}?"
    flag = _availability_flag(value)
    if flag is not None:
        return f"{prefix}{'+' if flag else '-'}"
    text = str(value).strip()
    lower = text.lower()
    if lower in {"positive", "pos", "+"}:
        text = "+"
    elif lower in {"negative", "neg", "-"}:
        text = "-"
    return f"{prefix}{text}"


def _subtype(row: Mapping[str, Any], config: MatchingConfig) -> str:
    if config.subtype_col is not None and config.subtype_col in row:
        value = row[config.subtype_col]
        if not _is_missing(value):
            return str(value)
    return "/".join(
        (
            _format_receptor("HR", row[config.hr_col]),
            _format_receptor("HER2", row[config.her2_col]),
        )
    )


def _stable_tie_break(seed: int, recipient_id: Any, donor_id: Any) -> int:
    payload = f"{seed}\x1f{recipient_id!r}\x1f{donor_id!r}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _matching_level(
    same_subtype: bool,
    same_treatment: bool,
    allow_relaxed_matches: bool,
) -> tuple[int, str] | None:
    if same_subtype and same_treatment:
        return 0, "hard_subtype_treatment_visit"
    if not allow_relaxed_matches:
        return None
    if same_subtype:
        return 1, "relaxed_subtype_visit"
    if same_treatment:
        return 2, "relaxed_treatment_visit"
    return 3, "relaxed_visit_only"


def _numeric_feature_flags(frame: pd.DataFrame, features: Sequence[str]) -> dict[str, bool]:
    flags: dict[str, bool] = {}
    for feature in features:
        nonmissing = frame[feature].notna()
        converted = pd.to_numeric(frame[feature], errors="coerce")
        flags[feature] = bool(nonmissing.any() and converted[nonmissing].notna().all())
    return flags


def _soft_distance(
    recipient_value: Any,
    donor_value: Any,
    *,
    numeric: bool,
    scale: float,
) -> float | None:
    if numeric:
        recipient_number = _finite_float(recipient_value)
        donor_number = _finite_float(donor_value)
        if recipient_number is None or donor_number is None:
            return None
        return abs(recipient_number - donor_number) / scale
    if _is_missing(recipient_value) or _is_missing(donor_value):
        return None
    return 0.0 if _same_observed_value(recipient_value, donor_value) else 1.0


def match_follow_up_donors(
    frame: pd.DataFrame,
    config: MatchingConfig | None = None,
    *,
    matching_features: Sequence[str] | None = None,
) -> MatchingResult:
    """Build an outcome-blind donor map within each held-out fold.

    Every row is treated as a recipient.  Candidates are restricted to rows in
    the same fold with a different patient ID and compatible follow-up visits.
    Baseline lesion-volume distance is standardised using that fold only.
    Age, MammaPrint, and extra baseline features are optional soft terms.

    ``matching_features`` overrides ``config.matching_features`` when supplied.
    The returned ``failures`` frame includes both unmatched recipients and
    partial matches with fewer than ``max_donors`` candidates.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame 必须是 pandas.DataFrame")
    config = config or MatchingConfig()
    features = validate_matching_features(
        config.matching_features if matching_features is None else matching_features
    )
    _validate_config(config, features)

    required_columns = _ordered_unique(
        [
            config.patient_id_col,
            config.fold_col,
            config.hr_col,
            config.her2_col,
            config.treatment_family_col,
            config.baseline_volume_col,
            *config.visit_availability_cols,
            *features,
        ]
    )
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise KeyError(f"donor matching 缺少必要列: {missing_columns}")

    optional_columns = [
        column
        for column in (config.subtype_col, config.age_col, config.mammaprint_col)
        if column is not None and column in frame.columns
    ]
    safe_columns = _ordered_unique([*required_columns, *optional_columns])
    # This is the only input-frame value selection.  It deliberately excludes
    # any label/outcome columns that merely coexist in ``frame``.
    working = frame.loc[:, safe_columns].copy()

    if working[config.patient_id_col].isna().any():
        raise ValueError("patient ID 不能缺失")
    if working[config.fold_col].isna().any():
        raise ValueError("fold 不能缺失")
    if working.duplicated([config.fold_col, config.patient_id_col]).any():
        raise ValueError("每个 fold 内 patient ID 必须唯一")
    fold_counts = working.groupby(config.patient_id_col, dropna=False)[config.fold_col].nunique()
    if bool((fold_counts > 1).any()):
        raise ValueError("同一 patient ID 不能出现在多个 fold；需要 patient-level fold manifest")

    known_algorithm_features = {
        config.patient_id_col,
        config.fold_col,
        config.hr_col,
        config.her2_col,
        config.treatment_family_col,
        config.baseline_volume_col,
        *config.visit_availability_cols,
        config.subtype_col,
        config.age_col,
        config.mammaprint_col,
    }
    extra_features = tuple(
        feature for feature in features if feature not in known_algorithm_features
    )
    numeric_extra = _numeric_feature_flags(working, extra_features)

    rows_by_fold: dict[Any, list[dict[str, Any]]] = {}
    volume_scales: dict[Any, float] = {}
    age_scales: dict[Any, float] = {}
    extra_scales: dict[tuple[Any, str], float] = {}
    for fold, fold_frame in working.groupby(config.fold_col, sort=False, dropna=False):
        rows_by_fold[fold] = fold_frame.to_dict(orient="records")
        volume_scales[fold] = _scale(fold_frame[config.baseline_volume_col])
        if config.age_col is not None and config.age_col in working.columns:
            age_scales[fold] = _scale(fold_frame[config.age_col])
        for feature in extra_features:
            if numeric_extra[feature]:
                extra_scales[(fold, feature)] = _scale(fold_frame[feature])

    mapping_records: list[dict[str, Any]] = []
    diagnostic_records: list[dict[str, Any]] = []

    for recipient in working.to_dict(orient="records"):
        recipient_id = recipient[config.patient_id_col]
        fold = recipient[config.fold_col]
        same_fold_rows = rows_by_fold[fold]
        nonself_rows = [
            donor
            for donor in same_fold_rows
            if not _same_observed_value(donor[config.patient_id_col], recipient_id)
        ]

        recipient_volume = _finite_float(recipient[config.baseline_volume_col])
        recipient_hard_missing = any(
            _is_missing(recipient[column])
            for column in (config.hr_col, config.her2_col, config.treatment_family_col)
        )
        recipient_visit_missing = any(
            _is_missing(recipient[column]) for column in config.visit_availability_cols
        )

        visit_compatible_count = 0
        hard_candidate_count = 0
        candidates: list[dict[str, Any]] = []
        if (
            recipient_volume is not None
            and not recipient_hard_missing
            and not recipient_visit_missing
        ):
            for donor in nonself_rows:
                donor_volume = _finite_float(donor[config.baseline_volume_col])
                if donor_volume is None:
                    continue
                if not _availability_compatible(
                    recipient, donor, config.visit_availability_cols
                ):
                    continue
                visit_compatible_count += 1

                same_subtype = _same_observed_value(
                    recipient[config.hr_col], donor[config.hr_col]
                ) and _same_observed_value(
                    recipient[config.her2_col], donor[config.her2_col]
                )
                same_treatment = _same_observed_value(
                    recipient[config.treatment_family_col],
                    donor[config.treatment_family_col],
                )
                if same_subtype and same_treatment:
                    hard_candidate_count += 1
                level = _matching_level(
                    same_subtype, same_treatment, config.allow_relaxed_matches
                )
                if level is None:
                    continue

                level_rank, level_name = level
                volume_distance = abs(recipient_volume - donor_volume) / volume_scales[fold]

                recipient_age = (
                    _finite_float(recipient[config.age_col])
                    if config.age_col is not None and config.age_col in working.columns
                    else None
                )
                donor_age = (
                    _finite_float(donor[config.age_col])
                    if config.age_col is not None and config.age_col in working.columns
                    else None
                )
                age_distance = (
                    abs(recipient_age - donor_age) / age_scales[fold]
                    if recipient_age is not None and donor_age is not None
                    else None
                )

                recipient_mp = (
                    recipient[config.mammaprint_col]
                    if config.mammaprint_col is not None
                    and config.mammaprint_col in working.columns
                    else np.nan
                )
                donor_mp = (
                    donor[config.mammaprint_col]
                    if config.mammaprint_col is not None
                    and config.mammaprint_col in working.columns
                    else np.nan
                )
                mp_mismatch = (
                    0.0 if _same_observed_value(recipient_mp, donor_mp) else 1.0
                ) if not _is_missing(recipient_mp) and not _is_missing(donor_mp) else None

                extra_distance = 0.0
                for feature in extra_features:
                    feature_distance = _soft_distance(
                        recipient[feature],
                        donor[feature],
                        numeric=numeric_extra[feature],
                        scale=extra_scales.get((fold, feature), 1.0),
                    )
                    if feature_distance is not None:
                        extra_distance += feature_distance

                matching_distance = volume_distance
                if age_distance is not None:
                    matching_distance += config.age_weight * age_distance
                if mp_mismatch is not None:
                    matching_distance += config.mammaprint_weight * mp_mismatch
                matching_distance += config.extra_feature_weight * extra_distance

                donor_id = donor[config.patient_id_col]
                candidates.append(
                    {
                        "donor": donor,
                        "level_rank": level_rank,
                        "matching_level": level_name,
                        "same_subtype": same_subtype,
                        "same_treatment": same_treatment,
                        "volume_distance_z": volume_distance,
                        "age_distance_z": age_distance,
                        "mammaprint_mismatch": mp_mismatch,
                        "extra_feature_distance": extra_distance,
                        "matching_distance": matching_distance,
                        "recipient_age": recipient_age,
                        "donor_age": donor_age,
                        "recipient_mp": recipient_mp,
                        "donor_mp": donor_mp,
                        "tie_break": _stable_tie_break(config.seed, recipient_id, donor_id),
                    }
                )

        candidates.sort(
            key=lambda item: (
                item["level_rank"],
                item["matching_distance"],
                item["volume_distance_z"],
                item["tie_break"],
            )
        )
        selected = candidates[: config.max_donors]

        if not selected:
            if recipient_volume is None:
                failure_reason = "missing_recipient_baseline_volume"
            elif recipient_hard_missing:
                failure_reason = "missing_recipient_hard_match_value"
            elif recipient_visit_missing:
                failure_reason = "missing_recipient_visit_availability"
            elif not nonself_rows:
                failure_reason = "no_same_fold_nonself_donor"
            elif visit_compatible_count == 0:
                failure_reason = "no_visit_compatible_donor"
            elif hard_candidate_count == 0 and not config.allow_relaxed_matches:
                failure_reason = "no_hard_match_candidate"
            else:
                failure_reason = "no_eligible_candidate"
            status = "unmatched"
        elif len(selected) < config.max_donors:
            failure_reason = "fewer_than_requested_donors"
            status = "partial"
        else:
            failure_reason = None
            status = "full"

        diagnostic_records.append(
            {
                "recipient_patient_id": recipient_id,
                "fold": fold,
                "status": status,
                "failure_reason": failure_reason,
                "requested_donors": config.max_donors,
                "selected_donors": len(selected),
                "n_same_fold_nonself": len(nonself_rows),
                "n_visit_compatible": visit_compatible_count,
                "n_hard_candidates": hard_candidate_count,
                "n_eligible_candidates": len(candidates),
            }
        )

        recipient_subtype = _subtype(recipient, config)
        for repetition, candidate in enumerate(selected, start=1):
            donor = candidate["donor"]
            donor_volume = _finite_float(donor[config.baseline_volume_col])
            mapping_records.append(
                {
                    "recipient_patient_id": recipient_id,
                    "donor_patient_id": donor[config.patient_id_col],
                    "fold": fold,
                    "subtype": recipient_subtype,
                    "donor_subtype": _subtype(donor, config),
                    "treatment_family": recipient[config.treatment_family_col],
                    "donor_treatment_family": donor[config.treatment_family_col],
                    "baseline_lesion_volume": recipient_volume,
                    "recipient_baseline_lesion_volume": recipient_volume,
                    "donor_baseline_lesion_volume": donor_volume,
                    "matching_distance": candidate["matching_distance"],
                    "volume_distance_z": candidate["volume_distance_z"],
                    "age_distance_z": candidate["age_distance_z"],
                    "mammaprint_mismatch": candidate["mammaprint_mismatch"],
                    "extra_feature_distance": candidate["extra_feature_distance"],
                    "recipient_age": candidate["recipient_age"],
                    "donor_age": candidate["donor_age"],
                    "recipient_mammaprint": candidate["recipient_mp"],
                    "donor_mammaprint": candidate["donor_mp"],
                    "subtype_match": candidate["same_subtype"],
                    "treatment_family_match": candidate["same_treatment"],
                    "visit_availability_compatible": True,
                    "matching_level": candidate["matching_level"],
                    "audit_repetition": repetition,
                    "matching_seed": config.seed,
                }
            )

    mapping = pd.DataFrame.from_records(mapping_records, columns=MAPPING_COLUMNS)
    diagnostics = pd.DataFrame.from_records(
        diagnostic_records, columns=DIAGNOSTIC_COLUMNS
    )
    failures = diagnostics.loc[diagnostics["status"] != "full"].reset_index(drop=True)
    balance_stats = summarize_matching_balance(mapping)
    success_stats = summarize_matching_success(mapping, diagnostics)
    return MatchingResult(
        mapping=mapping,
        failures=failures,
        recipient_diagnostics=diagnostics,
        balance_stats=balance_stats,
        success_stats=success_stats,
    )


def _safe_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = numeric[np.isfinite(numeric)]
    return float(np.mean(finite)) if finite.size else math.nan


def _safe_median(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = numeric[np.isfinite(numeric)]
    return float(np.median(finite)) if finite.size else math.nan


def _standardized_mean_difference(left: pd.Series, right: pd.Series) -> float:
    left_values = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    right_values = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    left_values = left_values[np.isfinite(left_values)]
    right_values = right_values[np.isfinite(right_values)]
    if not left_values.size or not right_values.size:
        return math.nan
    difference = float(np.mean(left_values) - np.mean(right_values))
    pooled_scale = math.sqrt(
        (float(np.var(left_values, ddof=0)) + float(np.var(right_values, ddof=0))) / 2.0
    )
    if pooled_scale <= 1e-12:
        return 0.0 if abs(difference) <= 1e-12 else math.nan
    return difference / pooled_scale


def _balance_row(
    pairs: pd.DataFrame,
    *,
    scope: str,
    fold: Any = None,
    matching_level: str | None = None,
) -> dict[str, Any]:
    volume_difference = (
        pd.to_numeric(pairs["recipient_baseline_lesion_volume"], errors="coerce")
        - pd.to_numeric(pairs["donor_baseline_lesion_volume"], errors="coerce")
    ).abs()
    age_difference = (
        pd.to_numeric(pairs["recipient_age"], errors="coerce")
        - pd.to_numeric(pairs["donor_age"], errors="coerce")
    ).abs()
    mp_mismatch = pd.to_numeric(pairs["mammaprint_mismatch"], errors="coerce")
    mp_match_rate = 1.0 - _safe_mean(mp_mismatch) if mp_mismatch.notna().any() else math.nan
    return {
        "scope": scope,
        "fold": fold,
        "matching_level": matching_level,
        "n_pairs": int(len(pairs)),
        "mean_abs_baseline_volume_difference": _safe_mean(volume_difference),
        "median_abs_baseline_volume_difference": _safe_median(volume_difference),
        "mean_volume_distance_z": _safe_mean(pairs["volume_distance_z"]),
        "baseline_volume_standardized_mean_difference": _standardized_mean_difference(
            pairs["recipient_baseline_lesion_volume"],
            pairs["donor_baseline_lesion_volume"],
        ),
        "mean_abs_age_difference": _safe_mean(age_difference),
        "age_standardized_mean_difference": _standardized_mean_difference(
            pairs["recipient_age"], pairs["donor_age"]
        ),
        "mammaprint_match_rate": mp_match_rate,
        "subtype_match_rate": _safe_mean(pairs["subtype_match"]),
        "treatment_family_match_rate": _safe_mean(
            pairs["treatment_family_match"]
        ),
    }


def summarize_matching_balance(mapping: pd.DataFrame) -> pd.DataFrame:
    """Summarize pairwise baseline balance overall, by fold, and by level."""

    missing = [column for column in MAPPING_COLUMNS if column not in mapping.columns]
    if missing:
        raise KeyError(f"matching mapping 缺少列: {missing}")
    if mapping.empty:
        return pd.DataFrame.from_records(
            [_balance_row(mapping, scope="overall")], columns=BALANCE_COLUMNS
        )

    records = [_balance_row(mapping, scope="overall")]
    for fold, pairs in mapping.groupby("fold", sort=False, dropna=False):
        records.append(_balance_row(pairs, scope="fold", fold=fold))
    for level, pairs in mapping.groupby("matching_level", sort=False, dropna=False):
        records.append(
            _balance_row(
                pairs,
                scope="matching_level",
                matching_level=str(level),
            )
        )
    return pd.DataFrame.from_records(records, columns=BALANCE_COLUMNS)


def summarize_matching_success(
    mapping: pd.DataFrame,
    recipient_diagnostics: pd.DataFrame,
) -> dict[str, Any]:
    """Return JSON-friendly matching success and shortfall statistics."""

    diagnostic_missing = [
        column for column in DIAGNOSTIC_COLUMNS if column not in recipient_diagnostics.columns
    ]
    if diagnostic_missing:
        raise KeyError(f"recipient diagnostics 缺少列: {diagnostic_missing}")
    mapping_missing = [column for column in MAPPING_COLUMNS if column not in mapping.columns]
    if mapping_missing:
        raise KeyError(f"matching mapping 缺少列: {mapping_missing}")

    n_recipients = int(len(recipient_diagnostics))
    status_counts = recipient_diagnostics["status"].value_counts().to_dict()
    n_full = int(status_counts.get("full", 0))
    n_partial = int(status_counts.get("partial", 0))
    n_unmatched = int(status_counts.get("unmatched", 0))
    n_matched = n_full + n_partial

    per_fold: list[dict[str, Any]] = []
    for fold, group in recipient_diagnostics.groupby("fold", sort=False, dropna=False):
        fold_status = group["status"].value_counts().to_dict()
        fold_matched = int(
            fold_status.get("full", 0) + fold_status.get("partial", 0)
        )
        per_fold.append(
            {
                "fold": fold,
                "n_recipients": int(len(group)),
                "n_matched_recipients": fold_matched,
                "n_full_recipients": int(fold_status.get("full", 0)),
                "n_partial_recipients": int(fold_status.get("partial", 0)),
                "n_unmatched_recipients": int(fold_status.get("unmatched", 0)),
                "success_rate": fold_matched / len(group) if len(group) else math.nan,
            }
        )

    return {
        "n_recipients": n_recipients,
        "n_matched_recipients": n_matched,
        "n_full_recipients": n_full,
        "n_partial_recipients": n_partial,
        "n_unmatched_recipients": n_unmatched,
        "n_failed_recipients": n_unmatched,
        "success_rate": n_matched / n_recipients if n_recipients else math.nan,
        "full_match_rate": n_full / n_recipients if n_recipients else math.nan,
        "n_mappings": int(len(mapping)),
        "mean_donors_per_recipient": (
            float(recipient_diagnostics["selected_donors"].mean())
            if n_recipients
            else math.nan
        ),
        "counts_by_matching_level": {
            str(level): int(count)
            for level, count in mapping["matching_level"].value_counts().items()
        },
        "failure_reason_counts": {
            str(reason): int(count)
            for reason, count in recipient_diagnostics["failure_reason"]
            .dropna()
            .value_counts()
            .items()
        },
        "per_fold": per_fold,
    }


# Readable aliases for scripts that use either naming convention.
build_donor_mapping = match_follow_up_donors
build_matched_followup_mapping = match_follow_up_donors


__all__ = [
    "BALANCE_COLUMNS",
    "DIAGNOSTIC_COLUMNS",
    "MAPPING_COLUMNS",
    "MatchingConfig",
    "MatchingResult",
    "build_donor_mapping",
    "build_matched_followup_mapping",
    "match_follow_up_donors",
    "summarize_matching_balance",
    "summarize_matching_success",
    "validate_matching_features",
]

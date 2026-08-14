"""Exact outer-train HR x HER2 x assigned-arm matching strata.

There is deliberately no fallback path in this module.  Every contrastive
positive and negative is drawn from the anchor's exact clinical/treatment
stratum, and an anchor is usable only when it has both a different-patient
same-pCR partner and an opposite-pCR patient in that stratum.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .contracts import FOLDS, MATCHING_FIELDS


@dataclass(frozen=True)
class StrataAudit:
    """Aggregate-only audit safe to include in public logs."""

    scope: str
    matching_fields: tuple[str, ...]
    training_patients: int
    total_strata: int
    # A stratum is ``usable`` when it contributes at least one eligible anchor;
    # this includes asymmetric 1-v-N strata whose majority-class anchors are
    # eligible. ``bidirectionally_usable_strata`` is the stricter both-classes-
    # have-a-partner count and removes any 1-v-N ambiguity in reporting.
    usable_strata: int
    bidirectionally_usable_strata: int
    usable_patients: int
    dropped_anchors: int
    dropped_no_same_class_partner: int
    dropped_no_opposite_class: int
    pcr_class_distribution: Mapping[int, int]
    usable_pcr_class_distribution: Mapping[int, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "matching_fields": list(self.matching_fields),
            "training_patients": self.training_patients,
            "total_strata": self.total_strata,
            "usable_strata": self.usable_strata,
            "bidirectionally_usable_strata": self.bidirectionally_usable_strata,
            "usable_patients": self.usable_patients,
            "dropped_anchors": self.dropped_anchors,
            "dropped_no_same_class_partner": self.dropped_no_same_class_partner,
            "dropped_no_opposite_class": self.dropped_no_opposite_class,
            "pcr_class_distribution": {
                str(key): int(value)
                for key, value in self.pcr_class_distribution.items()
            },
            "usable_pcr_class_distribution": {
                str(key): int(value)
                for key, value in self.usable_pcr_class_distribution.items()
            },
            "unmatched_fallback_used": False,
            "test_patients_used": False,
        }


@dataclass(frozen=True)
class StrataResult:
    """Private per-patient assignments plus an aggregate audit."""

    assignments: pd.DataFrame
    audit: StrataAudit

    @property
    def stratum_ids(self) -> dict[str, int]:
        return dict(
            zip(
                self.assignments["patient_id"].astype(str),
                self.assignments["stratum_id"].astype(int),
                strict=True,
            )
        )

    @property
    def eligible_anchors(self) -> tuple[str, ...]:
        return tuple(
            self.assignments.loc[
                self.assignments["eligible_anchor"], "patient_id"
            ].astype(str)
        )

    @property
    def eligible_mask(self) -> Any:
        return self.assignments["eligible_anchor"].to_numpy(dtype=bool, copy=True)


def _binary_integer(series: pd.Series, label: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="raise")
    if numeric.isna().any() or not numeric.isin((0, 1)).all():
        raise ValueError(f"{label} must contain complete binary values")
    if any(float(value) != int(value) for value in numeric):
        raise ValueError(f"{label} must contain exact binary integers")
    return numeric.astype("int8")


def build_exact_strata(
    training_rows: pd.DataFrame | Sequence[Mapping[str, Any]],
    *,
    patient_column: str = "patient_id",
    label_column: str = "label_pcr",
    matching_fields: Sequence[str] = MATCHING_FIELDS,
    split_column: str = "split",
) -> StrataResult:
    """Construct deterministic exact strata from outer-training rows only.

    If a ``split`` column is present, every row must be explicitly marked
    ``train``.  Callers therefore cannot accidentally pass a complete fold and
    let this function silently discard validation or test patients.
    """

    frame = (
        training_rows.copy()
        if isinstance(training_rows, pd.DataFrame)
        else pd.DataFrame(list(training_rows))
    )
    fields = tuple(str(value) for value in matching_fields)
    if fields != MATCHING_FIELDS:
        raise ValueError(f"matching fields are frozen to exact {MATCHING_FIELDS}")
    required = {patient_column, label_column, *fields}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"training strata input lacks required fields: {missing}")
    if frame.empty:
        raise ValueError("training strata input is empty")
    if split_column in frame.columns:
        split = frame[split_column].astype(str).str.lower()
        if not split.eq("train").all():
            raise ValueError("exact strata may be constructed from outer-train rows only")
    if "fold" in frame.columns and frame["fold"].nunique(dropna=False) != 1:
        raise ValueError("strata input must represent exactly one outer fold")

    selected = frame[[patient_column, label_column, *fields]].copy()
    selected.columns = ["patient_id", "label_pcr", *MATCHING_FIELDS]
    selected["patient_id"] = selected["patient_id"].astype(str)
    if selected["patient_id"].eq("").any() or selected["patient_id"].duplicated().any():
        raise ValueError("outer-training patient identifiers must be nonempty and unique")
    selected["label_pcr"] = _binary_integer(selected["label_pcr"], "pCR")
    selected["label_hr"] = _binary_integer(selected["label_hr"], "HR")
    selected["label_her2"] = _binary_integer(selected["label_her2"], "HER2")
    if selected["arm"].isna().any():
        raise ValueError("assigned treatment arm must be complete")
    selected["arm"] = selected["arm"].astype(str)
    if selected["arm"].str.len().eq(0).any():
        raise ValueError("assigned treatment arm must be nonempty")

    # Sorting the clinical/treatment tuples makes IDs deterministic while
    # keeping the literal arm categories exact (no collapsing or fallback).
    keys = list(
        zip(
            selected["label_hr"].astype(int),
            selected["label_her2"].astype(int),
            selected["arm"],
            strict=True,
        )
    )
    unique_keys = sorted(set(keys), key=lambda value: (value[0], value[1], value[2]))
    identifier = {key: index for index, key in enumerate(unique_keys)}
    selected["stratum_id"] = [identifier[key] for key in keys]

    counts = (
        selected.groupby(["stratum_id", "label_pcr"], sort=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[0, 1], fill_value=0)
    )
    same_counts = [
        int(counts.loc[stratum_id, int(label)]) - 1
        for stratum_id, label in zip(
            selected["stratum_id"], selected["label_pcr"], strict=True
        )
    ]
    opposite_counts = [
        int(counts.loc[stratum_id, 1 - int(label)])
        for stratum_id, label in zip(
            selected["stratum_id"], selected["label_pcr"], strict=True
        )
    ]
    selected["same_class_partner_count"] = same_counts
    selected["opposite_class_count"] = opposite_counts
    selected["eligible_anchor"] = (
        selected["same_class_partner_count"].gt(0)
        & selected["opposite_class_count"].gt(0)
    )
    usable_strata = set(
        selected.loc[selected["eligible_anchor"], "stratum_id"].astype(int)
    )
    bidirectionally_usable_strata = set(
        counts.index[(counts[0] >= 2) & (counts[1] >= 2)].astype(int)
    )
    selected["usable_stratum"] = selected["stratum_id"].isin(usable_strata)

    all_distribution = selected["label_pcr"].value_counts().reindex([0, 1], fill_value=0)
    usable_distribution = (
        selected.loc[selected["eligible_anchor"], "label_pcr"]
        .value_counts()
        .reindex([0, 1], fill_value=0)
    )
    usable_count = int(selected["eligible_anchor"].sum())
    audit = StrataAudit(
        scope="outer_train_only",
        matching_fields=MATCHING_FIELDS,
        training_patients=len(selected),
        total_strata=len(unique_keys),
        usable_strata=len(usable_strata),
        bidirectionally_usable_strata=len(bidirectionally_usable_strata),
        usable_patients=usable_count,
        dropped_anchors=len(selected) - usable_count,
        dropped_no_same_class_partner=int(
            selected["same_class_partner_count"].eq(0).sum()
        ),
        dropped_no_opposite_class=int(selected["opposite_class_count"].eq(0).sum()),
        pcr_class_distribution={
            0: int(all_distribution.loc[0]), 1: int(all_distribution.loc[1])
        },
        usable_pcr_class_distribution={
            0: int(usable_distribution.loc[0]),
            1: int(usable_distribution.loc[1]),
        },
    )
    return StrataResult(selected.reset_index(drop=True), audit)


def build_outer_train_strata(
    clinical: pd.DataFrame,
    folds: pd.DataFrame,
    fold: int,
) -> StrataResult:
    """Select one locked outer-train split, then build exact strata."""

    if isinstance(fold, bool) or int(fold) not in FOLDS:
        raise ValueError("fold must be one of 0..4")
    required_clinical = {"patient_id", "label_pcr", *MATCHING_FIELDS}
    required_folds = {"patient_id", "fold", "split", "label_pcr"}
    if missing := sorted(required_clinical.difference(clinical.columns)):
        raise ValueError(f"clinical table lacks required fields: {missing}")
    if missing := sorted(required_folds.difference(folds.columns)):
        raise ValueError(f"fold table lacks required fields: {missing}")
    split = folds.loc[
        folds["fold"].eq(int(fold)) & folds["split"].astype(str).str.lower().eq("train"),
        ["patient_id", "fold", "split", "label_pcr"],
    ].copy()
    if split.empty or split["patient_id"].astype(str).duplicated().any():
        raise ValueError("outer-train split must be nonempty and unique")
    clinical_columns = clinical[["patient_id", "label_pcr", *MATCHING_FIELDS]].copy()
    clinical_columns["patient_id"] = clinical_columns["patient_id"].astype(str)
    split["patient_id"] = split["patient_id"].astype(str)
    merged = split.merge(
        clinical_columns,
        on="patient_id",
        how="left",
        validate="one_to_one",
        suffixes=("_fold", ""),
    )
    if merged[list(MATCHING_FIELDS)].isna().any().any() or merged["label_pcr"].isna().any():
        raise ValueError("outer-train patients do not align with clinical matching fields")
    if not _binary_integer(merged["label_pcr_fold"], "fold pCR").eq(
        _binary_integer(merged["label_pcr"], "clinical pCR")
    ).all():
        raise ValueError("clinical and fold pCR labels disagree")
    return build_exact_strata(
        merged[["patient_id", "label_pcr", *MATCHING_FIELDS, "fold", "split"]]
    )


@dataclass(frozen=True)
class PairMasks:
    """Exact masks used by the conditional supervised contrastive loss."""

    positives: Any
    negatives: Any
    denominator: Any
    eligible_anchor: Any

    def __iter__(self):
        yield self.positives
        yield self.negatives
        yield self.denominator
        yield self.eligible_anchor


def conditional_pair_masks(
    labels: Any,
    stratum_ids: Any,
    eligible_anchor: Any | None = None,
) -> PairMasks:
    """Build self-excluding, strictly within-stratum pair masks.

    Explicitly marked eligible anchors are checked against the mathematical
    eligibility rule.  False entries may intentionally select a subset of
    otherwise eligible anchors, but a true entry can never enable fallback.
    """

    import torch

    labels = torch.as_tensor(labels)
    stratum_ids = torch.as_tensor(stratum_ids, device=labels.device)
    if labels.ndim != 1 or stratum_ids.ndim != 1 or labels.shape != stratum_ids.shape:
        raise ValueError("labels and stratum_ids must be aligned one-dimensional tensors")
    if labels.numel() == 0:
        raise ValueError("contrastive batch is empty")
    if labels.dtype == torch.bool:
        labels = labels.to(dtype=torch.long)
    if labels.is_floating_point() and not torch.equal(labels, labels.round()):
        raise ValueError("pCR labels must be exact binary integers")
    if not bool(torch.logical_or(labels == 0, labels == 1).all()):
        raise ValueError("pCR labels must be binary")
    if stratum_ids.dtype == torch.bool:
        raise ValueError("stratum_ids must be nonnegative exact integers")
    if stratum_ids.is_floating_point() and not torch.equal(
        stratum_ids, stratum_ids.round()
    ):
        raise ValueError("stratum_ids must be nonnegative exact integers")
    if bool((stratum_ids < 0).any()):
        raise ValueError("stratum_ids must be nonnegative exact integers")
    count = labels.numel()
    nonself = ~torch.eye(count, dtype=torch.bool, device=labels.device)
    same_stratum = stratum_ids[:, None].eq(stratum_ids[None, :])
    same_label = labels[:, None].eq(labels[None, :])
    denominator = same_stratum & nonself
    positives = denominator & same_label
    negatives = denominator & ~same_label
    derived = positives.any(dim=1) & negatives.any(dim=1)
    if eligible_anchor is None:
        eligible = derived
    else:
        eligible = torch.as_tensor(eligible_anchor, device=labels.device)
        if eligible.ndim != 1 or eligible.shape != labels.shape:
            raise ValueError("eligible_anchor must align with labels")
        if eligible.dtype != torch.bool:
            if eligible.is_floating_point() and not torch.equal(eligible, eligible.round()):
                raise ValueError("eligible_anchor must be boolean")
            if not bool(torch.logical_or(eligible == 0, eligible == 1).all()):
                raise ValueError("eligible_anchor must be boolean")
            eligible = eligible.bool()
        if bool((eligible & ~derived).any()):
            raise ValueError(
                "an eligible anchor lacks a same-class partner or opposite-class negative"
            )
    if not bool(eligible.any()):
        raise ValueError("batch contains no eligible exact-stratum anchors")
    return PairMasks(positives, negatives, denominator, eligible)


# Compatibility names used in analysis and orchestration code.
build_exact_training_strata = build_exact_strata
construct_exact_strata = build_exact_strata
build_pair_masks = conditional_pair_masks


__all__ = [
    "PairMasks",
    "StrataAudit",
    "StrataResult",
    "build_exact_strata",
    "build_exact_training_strata",
    "build_outer_train_strata",
    "build_pair_masks",
    "conditional_pair_masks",
    "construct_exact_strata",
]

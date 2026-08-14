"""Pre-registered success gates and final interpretation classes.

Folds are never treated as biological replicates here.  Gate decisions operate
on one paired AUROC delta per training seed and headline configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


PRIMARY_SEEDS = (2026, 3026)
SUPERVISED_ARMS = ("B1", "B2", "B3")
ADAPTED_ARMS = ("B2", "B3")
EARLY_MID_TIMINGS = ("T0", "T0_T1", "T0_T2")
CEILING_TIMINGS = ("T0_T1", "T0_T2")

GATE_A_PASS_CODE = "SUPERVISED_MRI_SIGNAL_CEILING_FOUND"
GATE_B_PASS_CODE = "CLINICAL_COMPLEMENTARY_MRI_SIGNAL_EXISTS"
GATE_C_PASS_CODE = "BEYOND_FTV_MRI_SIGNAL_EXISTS"

INTERPRETATION_CODES = {
    "A": "STRONG_HIDDEN_MRI_CEILING",
    "B": "MRI_SIGNAL_EXISTS_BUT_IS_FTV_REDUNDANT",
    "C": "CLASSIFIER_OBJECTIVE_BOTTLENECK_ONLY",
    "D": "LOW_CONDITIONAL_MRI_CEILING",
    "UNRESOLVED": "UNRESOLVED_INTERMEDIATE_PATTERN",
}


@dataclass(frozen=True)
class GateDecision:
    gate: str
    passed: bool
    pass_code: str | None
    threshold: float | None
    best_arm: str | None
    best_timing: str | None
    mean_delta_auroc: float
    seed_deltas: tuple[tuple[int, float], ...]
    positive_in_both_seeds: bool
    ci_excludes_zero: bool | None
    criterion: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "passed": self.passed,
            "pass_code": self.pass_code,
            "threshold": self.threshold,
            "best_arm": self.best_arm,
            "best_timing": self.best_timing,
            "mean_delta_auroc": self.mean_delta_auroc,
            "seed_deltas": dict(self.seed_deltas),
            "positive_in_both_seeds": self.positive_in_both_seeds,
            "ci_excludes_zero": self.ci_excludes_zero,
            "criterion": self.criterion,
        }


@dataclass(frozen=True)
class GateEvaluation:
    gate_a: GateDecision
    gate_b: GateDecision
    gate_c: GateDecision
    interpretation_class: str
    interpretation_code: str

    @property
    def passed_codes(self) -> tuple[str, ...]:
        return tuple(
            decision.pass_code
            for decision in (self.gate_a, self.gate_b, self.gate_c)
            if decision.pass_code is not None
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "gate_a": self.gate_a.as_dict(),
            "gate_b": self.gate_b.as_dict(),
            "gate_c": self.gate_c.as_dict(),
            "interpretation_class": self.interpretation_class,
            "interpretation_code": self.interpretation_code,
            "passed_codes": self.passed_codes,
        }


def _frame(rows: Any, *, name: str, require_ci: bool = False) -> pd.DataFrame:
    if isinstance(rows, pd.DataFrame):
        frame = rows.copy()
    else:
        try:
            frame = pd.DataFrame(rows)
        except Exception as error:  # pragma: no cover - pandas exception types vary
            raise TypeError(f"{name} must be a DataFrame or records") from error
    aliases = {
        "delta": "delta_auroc",
        "auroc_delta": "delta_auroc",
        "delta_AUROC": "delta_auroc",
        "auroc_ci_lower": "ci_lower",
        "bootstrap_ci_lower": "ci_lower",
    }
    for source, destination in aliases.items():
        if destination not in frame.columns and source in frame.columns:
            frame[destination] = frame[source]
    required = {"seed", "arm", "timing", "delta_auroc"}
    if require_ci:
        required.add("ci_lower")
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} misses required columns: {missing}")
    if frame.empty:
        raise ValueError(f"{name} must not be empty")
    frame["arm"] = frame["arm"].astype(str).str.upper()
    frame["timing"] = (
        frame["timing"]
        .astype(str)
        .str.upper()
        .str.replace("–", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    try:
        seed_numeric = pd.to_numeric(frame["seed"], errors="raise").to_numpy(float)
        delta_numeric = pd.to_numeric(frame["delta_auroc"], errors="raise").to_numpy(float)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} seed/delta values must be numeric") from error
    if (
        not np.isfinite(seed_numeric).all()
        or not np.equal(seed_numeric, np.floor(seed_numeric)).all()
        or not np.isfinite(delta_numeric).all()
    ):
        raise ValueError(f"{name} seed/delta values must be finite (seeds integral)")
    frame["seed"] = seed_numeric.astype(np.int64)
    frame["delta_auroc"] = delta_numeric
    if require_ci:
        ci = pd.to_numeric(frame["ci_lower"], errors="coerce").to_numpy(float)
        if not np.isfinite(ci).all():
            raise ValueError(f"{name} ci_lower values must be finite")
        frame["ci_lower"] = ci
    return frame


def _eligible_groups(
    rows: pd.DataFrame,
    *,
    seeds: Sequence[int],
    timings: Sequence[str],
    population: str,
    arms: Sequence[str] = SUPERVISED_ARMS,
) -> list[tuple[str, str, pd.DataFrame]]:
    selected = rows.loc[
        rows["arm"].isin(tuple(arms)) & rows["timing"].isin(timings)
    ].copy()
    if "population" in selected.columns:
        selected = selected.loc[selected["population"].astype(str).eq(population)]
    required_seeds = tuple(int(seed) for seed in seeds)
    selected = selected.loc[selected["seed"].isin(required_seeds)]
    if selected.duplicated(["arm", "timing", "seed"]).any():
        raise ValueError("gate evidence repeats an arm/timing/seed cell")
    groups: list[tuple[str, str, pd.DataFrame]] = []
    for (arm, timing), group in selected.groupby(["arm", "timing"], sort=True):
        if set(group["seed"].astype(int)) == set(required_seeds) and len(group) == len(required_seeds):
            groups.append((str(arm), str(timing), group.sort_values("seed")))
    return groups


def _failed(gate: str, *, threshold: float | None, criterion: str) -> GateDecision:
    return GateDecision(
        gate=gate,
        passed=False,
        pass_code=None,
        threshold=threshold,
        best_arm=None,
        best_timing=None,
        mean_delta_auroc=math.nan,
        seed_deltas=(),
        positive_in_both_seeds=False,
        ci_excludes_zero=None,
        criterion=criterion,
    )


def evaluate_gate_a(
    rows: Any,
    *,
    seeds: Sequence[int] = PRIMARY_SEEDS,
    threshold: float = 0.08,
) -> GateDecision:
    """Gate A: B2/B3 MRI-only mean delta >= .08 at T0-T1 or T0-T2."""

    criterion = "B2_or_B3_M_minus_B0_mean_delta_at_T0_T1_or_T0_T2_ge_0.08"
    frame = _frame(rows, name="Gate A evidence")
    groups = _eligible_groups(
        frame, seeds=seeds, timings=CEILING_TIMINGS, population="full_808",
        arms=ADAPTED_ARMS,
    )
    if not groups:
        return _failed("A", threshold=threshold, criterion=criterion)
    arm, timing, best = max(
        groups,
        key=lambda item: (
            float(item[2]["delta_auroc"].mean()),
            -ADAPTED_ARMS.index(item[0]),
            -CEILING_TIMINGS.index(item[1]),
        ),
    )
    mean_delta = float(best["delta_auroc"].mean())
    passed = mean_delta >= float(threshold)
    return GateDecision(
        gate="A",
        passed=passed,
        pass_code=GATE_A_PASS_CODE if passed else None,
        threshold=float(threshold),
        best_arm=arm,
        best_timing=timing,
        mean_delta_auroc=mean_delta,
        seed_deltas=tuple(
            (int(row.seed), float(row.delta_auroc)) for row in best.itertuples()
        ),
        positive_in_both_seeds=bool((best["delta_auroc"] > 0.0).all()),
        ci_excludes_zero=None,
        criterion=criterion,
    )


def evaluate_gate_b(
    rows: Any,
    *,
    seeds: Sequence[int] = PRIMARY_SEEDS,
) -> GateDecision:
    """Gate B: C+M-C positive in both seeds and one paired AUROC CI > 0."""

    criterion = "C_plus_M_minus_C_positive_both_seeds_and_at_least_one_CI_lower_gt_0"
    frame = _frame(rows, name="Gate B evidence", require_ci=True)
    groups = _eligible_groups(
        frame, seeds=seeds, timings=EARLY_MID_TIMINGS, population="full_808"
    )
    if not groups:
        return _failed("B", threshold=0.0, criterion=criterion)
    ranked = sorted(
        groups,
        key=lambda item: (
            bool((item[2]["delta_auroc"] > 0.0).all() and (item[2]["ci_lower"] > 0.0).any()),
            float(item[2]["delta_auroc"].mean()),
        ),
        reverse=True,
    )
    arm, timing, best = ranked[0]
    positive = bool((best["delta_auroc"] > 0.0).all())
    ci_positive = bool((best["ci_lower"] > 0.0).any())
    passed = positive and ci_positive
    return GateDecision(
        gate="B",
        passed=passed,
        pass_code=GATE_B_PASS_CODE if passed else None,
        threshold=0.0,
        best_arm=arm,
        best_timing=timing,
        mean_delta_auroc=float(best["delta_auroc"].mean()),
        seed_deltas=tuple(
            (int(row.seed), float(row.delta_auroc)) for row in best.itertuples()
        ),
        positive_in_both_seeds=positive,
        ci_excludes_zero=ci_positive,
        criterion=criterion,
    )


def evaluate_gate_c(
    rows: Any,
    *,
    seeds: Sequence[int] = PRIMARY_SEEDS,
    threshold: float = 0.03,
) -> GateDecision:
    """Gate C: C+F+M-(C+F) positive both seeds with mean delta >= .03."""

    criterion = "C_plus_F_plus_M_minus_C_plus_F_positive_both_seeds_and_mean_ge_0.03"
    frame = _frame(rows, name="Gate C evidence")
    groups = _eligible_groups(
        frame,
        seeds=seeds,
        timings=EARLY_MID_TIMINGS,
        population="ftv_complete_375",
    )
    if not groups:
        return _failed("C", threshold=threshold, criterion=criterion)
    ranked = sorted(
        groups,
        key=lambda item: (
            bool((item[2]["delta_auroc"] > 0.0).all()),
            float(item[2]["delta_auroc"].mean()),
        ),
        reverse=True,
    )
    arm, timing, best = ranked[0]
    positive = bool((best["delta_auroc"] > 0.0).all())
    mean_delta = float(best["delta_auroc"].mean())
    passed = positive and mean_delta >= float(threshold)
    ci_positive = (
        bool((pd.to_numeric(best["ci_lower"], errors="coerce") > 0.0).any())
        if "ci_lower" in best.columns
        else None
    )
    return GateDecision(
        gate="C",
        passed=passed,
        pass_code=GATE_C_PASS_CODE if passed else None,
        threshold=float(threshold),
        best_arm=arm,
        best_timing=timing,
        mean_delta_auroc=mean_delta,
        seed_deltas=tuple(
            (int(row.seed), float(row.delta_auroc)) for row in best.itertuples()
        ),
        positive_in_both_seeds=positive,
        ci_excludes_zero=ci_positive,
        criterion=criterion,
    )


def _passed(value: bool | GateDecision) -> bool:
    return bool(value.passed if isinstance(value, GateDecision) else value)


def classify_interpretation(
    gate_a: bool | GateDecision,
    gate_b: bool | GateDecision,
    gate_c: bool | GateDecision,
    *,
    b1_improves_strongly: bool = False,
    adaptation_adds_little: bool = False,
    b3_low_ceiling: bool = False,
) -> str:
    """Return the literal final interpretation class (A/B/C/D or unresolved).

    Classes C and D contain qualitative clauses in the goal ("strongly",
    "little").  They therefore require explicit audited predicates instead of
    silently inventing a threshold inside this classifier.
    """

    a, b, c = _passed(gate_a), _passed(gate_b), _passed(gate_c)
    if a and b and c:
        return "A"
    if a and b and not c:
        return "B"
    if bool(b1_improves_strongly) and bool(adaptation_adds_little):
        return "C"
    if bool(b3_low_ceiling):
        return "D"
    return "UNRESOLVED"


def _mean_by_arm(
    rows: Any,
    arm: str,
    *,
    seeds: Sequence[int],
    timings: Sequence[str],
    population: str,
) -> list[float]:
    frame = _frame(rows, name="interpretation evidence")
    selected = frame.loc[frame["arm"].eq(arm) & frame["timing"].isin(timings)].copy()
    if "population" in selected.columns:
        selected = selected.loc[selected["population"].astype(str).eq(population)]
    output: list[float] = []
    for _, group in selected.groupby("timing", sort=True):
        group = group.loc[group["seed"].isin(tuple(int(seed) for seed in seeds))]
        if set(group["seed"].astype(int)) == set(int(seed) for seed in seeds) and not group["seed"].duplicated().any():
            output.append(float(group["delta_auroc"].mean()))
    return output


def evaluate_gates(
    mri_ceiling_rows: Any,
    clinical_complementarity_rows: Any,
    beyond_ftv_rows: Any,
    *,
    seeds: Sequence[int] = PRIMARY_SEEDS,
    b1_improves_strongly: bool | None = None,
    adaptation_adds_little: bool | None = None,
    b3_low_ceiling: bool | None = None,
    strong_threshold: float = 0.08,
    little_threshold: float = 0.03,
) -> GateEvaluation:
    """Evaluate Gates A/B/C and apply the pre-registered interpretation logic."""

    gate_a = evaluate_gate_a(mri_ceiling_rows, seeds=seeds, threshold=strong_threshold)
    gate_b = evaluate_gate_b(clinical_complementarity_rows, seeds=seeds)
    gate_c = evaluate_gate_c(beyond_ftv_rows, seeds=seeds, threshold=little_threshold)

    b1_means = _mean_by_arm(
        mri_ceiling_rows,
        "B1",
        seeds=seeds,
        timings=CEILING_TIMINGS,
        population="full_808",
    )
    adapted_means = [
        *_mean_by_arm(
            mri_ceiling_rows,
            "B2",
            seeds=seeds,
            timings=CEILING_TIMINGS,
            population="full_808",
        ),
        *_mean_by_arm(
            mri_ceiling_rows,
            "B3",
            seeds=seeds,
            timings=CEILING_TIMINGS,
            population="full_808",
        ),
    ]
    b3_mri = _mean_by_arm(
        mri_ceiling_rows,
        "B3",
        seeds=seeds,
        timings=CEILING_TIMINGS,
        population="full_808",
    )
    b3_ftv = _mean_by_arm(
        beyond_ftv_rows,
        "B3",
        seeds=seeds,
        timings=EARLY_MID_TIMINGS,
        population="ftv_complete_375",
    )
    inferred_b1 = bool(b1_means and max(b1_means) >= float(strong_threshold))
    inferred_adaptation_little = bool(
        inferred_b1
        and adapted_means
        and max(adapted_means) - max(b1_means) < float(little_threshold)
    )
    inferred_b3_low = bool(
        b3_mri
        and b3_ftv
        and max(b3_mri) < float(strong_threshold)
        and max(b3_ftv) < float(little_threshold)
    )
    interpretation = classify_interpretation(
        gate_a,
        gate_b,
        gate_c,
        b1_improves_strongly=inferred_b1 if b1_improves_strongly is None else b1_improves_strongly,
        adaptation_adds_little=(
            inferred_adaptation_little if adaptation_adds_little is None else adaptation_adds_little
        ),
        b3_low_ceiling=inferred_b3_low if b3_low_ceiling is None else b3_low_ceiling,
    )
    return GateEvaluation(
        gate_a=gate_a,
        gate_b=gate_b,
        gate_c=gate_c,
        interpretation_class=interpretation,
        interpretation_code=INTERPRETATION_CODES[interpretation],
    )


__all__ = [
    "ADAPTED_ARMS",
    "CEILING_TIMINGS",
    "EARLY_MID_TIMINGS",
    "GATE_A_PASS_CODE",
    "GATE_B_PASS_CODE",
    "GATE_C_PASS_CODE",
    "INTERPRETATION_CODES",
    "PRIMARY_SEEDS",
    "SUPERVISED_ARMS",
    "GateDecision",
    "GateEvaluation",
    "classify_interpretation",
    "evaluate_gate_a",
    "evaluate_gate_b",
    "evaluate_gate_c",
    "evaluate_gates",
]

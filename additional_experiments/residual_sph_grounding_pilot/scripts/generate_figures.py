#!/usr/bin/env python3
"""Generate aggregate-only SVG figures after the formal two-seed decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC = EXPERIMENT_ROOT / "src"
sys.path.insert(0, str(SRC))

from residual_sph.preregistration import (  # noqa: E402
    require_lock_sha256,
    verify_preregistration,
)


SEEDS = (2026, 3026)
ARMS = ("S0", "S1", "S2")
TIMINGS = ("T0", "T0-T1", "T0-T2", "T0-T3")
FIGURE_NAMES = (
    "representation_effects.svg",
    "sph_res_organization.svg",
    "pcr_effects.svg",
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"missing or invalid aggregate JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"aggregate JSON must be an object: {path}")
    return value


def _paired_seed_values(container: Mapping[str, Any], key: str) -> list[float]:
    by_seed = container.get("by_seed")
    if not isinstance(by_seed, Mapping) or set(by_seed) != {
        str(seed) for seed in SEEDS
    }:
        raise ValueError("representation effect seed coverage drifted")
    values = [float(by_seed[str(seed)][key]) for seed in SEEDS]
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite representation effect: {key}")
    return values


def _pcr_seed_values(
    effects: Mapping[str, Any], key: str, timing: str
) -> list[float]:
    source = effects.get(key)
    if not isinstance(source, Mapping) or timing not in source:
        raise ValueError(f"pCR effects miss {key}/{timing}")
    by_seed = source[timing]
    if not isinstance(by_seed, Mapping) or set(by_seed) != {
        str(seed) for seed in SEEDS
    }:
        raise ValueError(f"pCR effect seed coverage drifted: {key}/{timing}")
    values = [float(by_seed[str(seed)]) for seed in SEEDS]
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite pCR effect: {key}/{timing}")
    return values


def _save_svg(figure: plt.Figure, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite aggregate figure: {path}")
    figure.savefig(
        path,
        format="svg",
        bbox_inches="tight",
        metadata={"Date": None, "Creator": "residual_sph_grounding_pilot"},
    )
    plt.close(figure)


def render_aggregate_figures(
    *,
    representation_effects: Mapping[str, Any],
    pcr_effects: Mapping[str, Any],
    sph_metrics: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, ...]:
    """Render only two-seed aggregates; no patient-level input is accepted."""

    required = {
        "arm",
        "seed_base",
        "task",
        "endpoint",
        "space",
        "residual_space_spearman",
    }
    if missing := required.difference(sph_metrics.columns):
        raise ValueError(f"SPH aggregate table misses columns: {sorted(missing)}")
    residual = sph_metrics.loc[
        sph_metrics["task"].eq("sph_res")
        & sph_metrics["endpoint"].eq("T0")
        & sph_metrics["space"].eq("residual")
        & sph_metrics["arm"].isin(ARMS)
    ].copy()
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    observed = set(
        zip(
            residual["arm"].astype(str),
            residual["seed_base"].astype(int),
            strict=True,
        )
    )
    if observed != expected or len(residual) != len(expected):
        raise ValueError("T0 residual-SPH aggregate coverage drifted")
    if "patient_id" in sph_metrics.columns:
        raise PermissionError("figure generator refuses patient-level input")

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "residual_sph_grounding_pilot",
        }
    )

    effect_names = ("E1", "E2", "E3", "E4")
    effect_matrix = np.asarray(
        [_paired_seed_values(representation_effects, name) for name in effect_names]
    )
    figure, axis = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(len(effect_names), dtype=np.float64)
    for seed_index, seed in enumerate(SEEDS):
        axis.plot(
            x,
            effect_matrix[:, seed_index],
            marker="o",
            linewidth=1.7,
            label=f"seed {seed}",
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.axhline(0.05, color="#777777", linewidth=0.8, linestyle="--")
    axis.set_xticks(x, effect_names)
    axis.set_ylabel("Paired Spearman difference")
    axis.set_title("Primary representation effects")
    axis.legend(frameon=False)
    first = output_dir / FIGURE_NAMES[0]
    _save_svg(figure, first)

    figure, axis = plt.subplots(figsize=(6.4, 3.8))
    width = 0.24
    x = np.arange(len(ARMS), dtype=np.float64)
    for seed_index, seed in enumerate(SEEDS):
        values = [
            float(
                residual.loc[
                    residual["arm"].eq(arm)
                    & residual["seed_base"].eq(seed),
                    "residual_space_spearman",
                ].iloc[0]
            )
            for arm in ARMS
        ]
        if not np.isfinite(values).all():
            raise ValueError("T0 residual-SPH figure values are non-finite")
        axis.bar(
            x + (seed_index - 0.5) * width,
            values,
            width=width,
            label=f"seed {seed}",
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x, ARMS)
    axis.set_ylabel("T0 SPH_res Spearman")
    axis.set_title("Residual-SPH organization")
    axis.legend(frameon=False)
    second = output_dir / FIGURE_NAMES[1]
    _save_svg(figure, second)

    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), sharex=True)
    pcr_specs = (
        ("E5_S2_minus_S0_MRI_only", "E5: S2-S0 MRI-only"),
        (
            "E6_S2_C_plus_F_plus_M_minus_C_plus_F",
            "E6: S2 C+F+M minus C+F",
        ),
    )
    x = np.arange(len(TIMINGS), dtype=np.float64)
    for axis, (key, title) in zip(axes, pcr_specs, strict=True):
        for seed_index, seed in enumerate(SEEDS):
            values = [
                _pcr_seed_values(pcr_effects, key, timing)[seed_index]
                for timing in TIMINGS
            ]
            axis.plot(x, values, marker="o", linewidth=1.7, label=f"seed {seed}")
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks(x, TIMINGS, rotation=25)
        axis.set_title(title)
        axis.set_ylabel("Paired AUROC difference")
    axes[1].legend(frameon=False)
    third = output_dir / FIGURE_NAMES[2]
    _save_svg(figure, third)
    return first, second, third


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration-lock-sha256", required=True)
    args = parser.parse_args()
    preregistration = verify_preregistration(EXPERIMENT_ROOT)
    require_lock_sha256(
        preregistration["lock_sha256"], args.preregistration_lock_sha256
    )
    decision = _read_json(EXPERIMENT_ROOT / "metrics" / "decision.json")
    if (
        decision.get("status") != "FORMAL_TWO_SEED_PILOT_COMPLETE"
        or decision.get("pcr_evaluation_was_post_freeze") is not True
    ):
        raise RuntimeError("aggregate figures require a completed post-freeze decision")
    representation = _read_json(
        EXPERIMENT_ROOT / "metrics" / "representation_effects.json"
    )
    pcr = _read_json(EXPERIMENT_ROOT / "metrics" / "pcr_effects.json")
    if decision.get("pcr_effects") != pcr:
        raise ValueError("decision and pCR effect artifacts differ")
    sph = pd.read_csv(EXPERIMENT_ROOT / "metrics" / "table_sph_and_residual.csv")
    outputs = render_aggregate_figures(
        representation_effects=representation,
        pcr_effects=pcr,
        sph_metrics=sph,
        output_dir=EXPERIMENT_ROOT / "figures",
    )
    print(json.dumps({"status": "PASS", "figures": [path.name for path in outputs]}))


if __name__ == "__main__":
    main()

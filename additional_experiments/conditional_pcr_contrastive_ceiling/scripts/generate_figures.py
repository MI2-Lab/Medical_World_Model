#!/usr/bin/env python3
"""Generate the seven registered figures from public aggregate CSVs only.

The script never reads the private ``predictions/``, ``features/``, checkpoint,
or patient-manifest trees.  Every resolved input and output path must remain
inside the supplied experiment root, and patient-level schemas are rejected.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = {
    "aggregate": Path("metrics/aggregate_metrics.csv"),
    "bootstrap": Path("metrics/paired_bootstrap.csv"),
    "profile": Path("metrics/clinical_profile_probes.csv"),
    "gaps": Path("metrics/generalization_gaps.csv"),
    "subgroups": Path("metrics/subgroup_refits.csv"),
}
PRIMARY_TIMINGS = ("T0", "T0_T1", "T0_T2")
ARMS = ("B0", "B1", "B2", "B3")
SUPERVISED_ARMS = ("B1", "B2", "B3")
PRIMARY_SEEDS = (2026, 3026)
BOUNDARY_NOTE = (
    "pCR-supervised representation ceiling; not evidence for the pCR-free World Model"
)
FORBIDDEN_PATIENT_COLUMNS = {
    "patient",
    "patient_id",
    "predicted_probability",
    "predicted_label",
    "y_true",
    "bootstrap_index",
    "bootstrap_draw",
    "draw_id",
}
FIGURE_SPECS = (
    ("01_mri_only_auroc.png", "B0/B1/B2/B3 MRI-only AUROC"),
    ("02_clinical_complementarity.png", "Clinical complementarity: C+M - C"),
    ("03_beyond_ftv_complementarity.png", "Beyond-FTV complementarity"),
    ("04_hr_her2_subgroups.png", "pCR performance within HR/HER2 subgroups"),
    ("05_generalization_gap.png", "Train/validation/test generalization"),
    ("06_profile_decodability.png", "Clinical-profile decodability"),
    ("07_supervised_ceiling_gap.png", "Supervised ceiling gap vs current World Model"),
)


class FigureContractError(RuntimeError):
    """Raised when an aggregate input cannot support a registered figure."""


def _inside(root: Path, path: str | Path, *, must_exist: bool) -> Path:
    root = root.resolve()
    raw = Path(path)
    candidate = (raw if raw.is_absolute() else root / raw).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise FigureContractError(
            f"path escapes experiment root {root}: {candidate}"
        ) from error
    if must_exist and (not candidate.is_file()):
        raise FileNotFoundError(f"required aggregate CSV does not exist: {candidate}")
    return candidate


def _require_columns(frame: pd.DataFrame, required: Iterable[str], *, name: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise FigureContractError(f"{name} misses required columns: {missing}")


def _numeric(frame: pd.DataFrame, columns: Iterable[str], *, name: str) -> None:
    for column in columns:
        try:
            values = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
        except (TypeError, ValueError) as error:
            raise FigureContractError(f"{name}.{column} must be numeric") from error
        if not np.isfinite(values).all():
            raise FigureContractError(f"{name}.{column} contains NaN or infinity")
        frame[column] = values


def _normalize_timing(values: pd.Series) -> pd.Series:
    return (
        values.astype(str)
        .str.upper()
        .str.strip()
        .str.replace("–", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )


def _load_csv(
    root: Path,
    relative_path: str | Path,
    *,
    name: str,
    required: Iterable[str],
    aliases: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, Path]:
    path = _inside(root, relative_path, must_exist=True)
    try:
        frame = pd.read_csv(path)
    except Exception as error:
        raise FigureContractError(f"could not read {name}: {path}") from error
    if frame.empty:
        raise FigureContractError(f"{name} must not be empty: {path}")
    lowered = {str(column).strip().lower() for column in frame.columns}
    forbidden = sorted(lowered & FORBIDDEN_PATIENT_COLUMNS)
    if forbidden:
        raise FigureContractError(
            f"{name} appears patient-level; forbidden columns present: {forbidden}"
        )
    if aliases:
        for source, destination in aliases.items():
            if destination not in frame.columns and source in frame.columns:
                frame[destination] = frame[source]
    _require_columns(frame, required, name=name)
    if "timing" in frame.columns:
        frame["timing"] = _normalize_timing(frame["timing"])
    if "arm" in frame.columns:
        frame["arm"] = frame["arm"].astype(str).str.upper().str.strip()
    if "model_family" in frame.columns:
        frame["model_family"] = frame["model_family"].astype(str).str.upper().str.strip()
    return frame, path


def _primary(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    selected = frame.loc[frame["timing"].isin(PRIMARY_TIMINGS)].copy()
    if selected.empty:
        raise FigureContractError(
            f"{name} has no primary timing rows; expected {PRIMARY_TIMINGS}"
        )
    return selected


def _save(figure: plt.Figure, path: Path) -> None:
    figure.text(0.5, 0.005, BOUNDARY_NOTE, ha="center", va="bottom", fontsize=7)
    figure.tight_layout(rect=(0.0, 0.035, 1.0, 1.0))
    figure.savefig(path, dpi=170, bbox_inches="tight", metadata={"Software": "conditional_ceiling"})
    plt.close(figure)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"failed to create non-empty figure: {path}")


def _timing_axis(axis: plt.Axes) -> None:
    axis.set_xticks(range(len(PRIMARY_TIMINGS)))
    axis.set_xticklabels(("T0", "T0-T1", "T0-T2"))


def _figure_mri_only(aggregate: pd.DataFrame, output: Path) -> None:
    frame = _primary(aggregate, name="aggregate metrics")
    frame = frame.loc[
        frame["population"].astype(str).eq("full_808")
        & frame["model_family"].eq("M")
        & frame["arm"].isin(ARMS)
    ].copy()
    if set(frame["arm"]) != set(ARMS):
        raise FigureContractError("MRI-only figure requires B0/B1/B2/B3 full_808 M rows")
    summary = frame.groupby(["arm", "timing"], sort=False)["auroc"].agg(["mean", "std", "count"])
    expected = pd.MultiIndex.from_product((ARMS, PRIMARY_TIMINGS), names=("arm", "timing"))
    if not expected.isin(summary.index).all():
        missing = expected[~expected.isin(summary.index)].tolist()
        raise FigureContractError(f"MRI-only figure misses arm/timing cells: {missing}")

    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    colors = dict(zip(ARMS, ("#666666", "#4c78a8", "#f58518", "#e45756"), strict=True))
    for arm in ARMS:
        cell = summary.loc[arm].reindex(PRIMARY_TIMINGS)
        error = cell["std"].fillna(0.0).to_numpy(float)
        axis.errorbar(
            range(len(PRIMARY_TIMINGS)),
            cell["mean"],
            yerr=error,
            marker="o",
            capsize=3,
            linewidth=1.8,
            label=arm,
            color=colors[arm],
        )
    _timing_axis(axis)
    axis.set_ylabel("Pooled OOF AUROC (mean across model seeds)")
    axis.set_title("MRI-only supervised ceiling")
    axis.set_ylim(0.35, 1.0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(title="Arm", ncol=4)
    _save(figure, output)


def _comparison_key(value: Any) -> str:
    return (
        str(value)
        .upper()
        .replace(" ", "")
        .replace("–", "-")
        .replace("−", "-")
        .replace("_PLUS_", "+")
        .replace("_MINUS_", "-")
        .replace("(", "")
        .replace(")", "")
    )


def _forest_delta(
    bootstrap: pd.DataFrame,
    output: Path,
    *,
    aliases: set[str],
    title: str,
    expected_population: str,
) -> None:
    frame = _primary(bootstrap, name="bootstrap summary")
    keys = frame["comparison"].map(_comparison_key)
    normalized_aliases = {_comparison_key(value) for value in aliases}
    frame = frame.loc[
        keys.isin(normalized_aliases)
        & frame["metric"].astype(str).str.lower().eq("auroc")
        & frame["population"].astype(str).eq(expected_population)
        & frame["arm"].isin(SUPERVISED_ARMS)
    ].copy()
    keys = ["arm", "timing", "seed"]
    expected = pd.MultiIndex.from_product(
        (SUPERVISED_ARMS, PRIMARY_TIMINGS, PRIMARY_SEEDS), names=keys
    )
    observed = pd.MultiIndex.from_frame(frame.loc[:, keys])
    if observed.has_duplicates or len(frame) != len(expected) or set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        raise FigureContractError(
            f"{title} requires the exact B1/B2/B3 x timing x seed registry; "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    frame = frame.sort_values(["timing", "arm", "seed"], kind="stable").reset_index(drop=True)
    labels = [f"{row.arm} | {row.timing.replace('_', '-')} | seed {int(row.seed)}" for row in frame.itertuples()]
    y = np.arange(len(frame))
    lower = frame["delta"].to_numpy(float) - frame["ci_lower"].to_numpy(float)
    upper = frame["ci_upper"].to_numpy(float) - frame["delta"].to_numpy(float)
    if np.any(lower < 0.0) or np.any(upper < 0.0):
        raise FigureContractError(f"{title} has CI bounds that do not contain delta")

    figure, axis = plt.subplots(figsize=(8.5, max(4.0, 0.38 * len(frame) + 1.7)))
    color_by_arm = {"B1": "#4c78a8", "B2": "#f58518", "B3": "#e45756"}
    colors = frame["arm"].map(color_by_arm).to_numpy()
    for index in range(len(frame)):
        axis.errorbar(
            frame.loc[index, "delta"],
            y[index],
            xerr=np.asarray([[lower[index]], [upper[index]]]),
            fmt="o",
            color=colors[index],
            capsize=3,
        )
    axis.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
    axis.set_yticks(y)
    axis.set_yticklabels(labels, fontsize=8)
    axis.invert_yaxis()
    axis.set_xlabel("Paired ΔAUROC (augmented - reference), 95% bootstrap CI")
    axis.set_title(title)
    axis.grid(axis="x", alpha=0.25)
    _save(figure, output)


def _figure_subgroups(subgroups: pd.DataFrame, output: Path) -> None:
    frame = _primary(subgroups, name="subgroup metrics")
    frame = frame.loc[
        frame["population"].astype(str).eq("full_808")
        & frame["model_family"].eq("M")
        & frame["arm"].isin(("B0", "B2", "B3"))
    ].copy()
    if "eligible" in frame.columns:
        eligibility = frame["eligible"].astype(str).str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        )
        if eligibility.isna().any():
            raise FigureContractError("subgroup metrics.eligible must be boolean")
        frame = frame.loc[eligibility].copy()
    required_groups = {"HR-/HER2-", "HR+/HER2-", "HER2+"}
    if not required_groups.issubset(set(frame["subgroup"].astype(str))):
        raise FigureContractError(
            f"subgroup figure requires eligible groups {sorted(required_groups)}"
        )
    summary = frame.groupby(["subgroup", "arm"], sort=False)["auroc"].mean()
    missing = [
        (subgroup, arm)
        for subgroup in sorted(required_groups)
        for arm in ("B0", "B2", "B3")
        if (subgroup, arm) not in summary.index
    ]
    if missing:
        raise FigureContractError(f"subgroup figure misses subgroup/arm cells: {missing}")

    groups = ("HR-/HER2-", "HR+/HER2-", "HER2+")
    x = np.arange(len(groups), dtype=float)
    width = 0.24
    figure, axis = plt.subplots(figsize=(8.3, 4.8))
    for offset, arm, color in zip(
        (-width, 0.0, width),
        ("B0", "B2", "B3"),
        ("#666666", "#f58518", "#e45756"),
        strict=True,
    ):
        values = [float(summary.loc[(group, arm)]) for group in groups]
        axis.bar(x + offset, values, width=width, label=arm, color=color)
    axis.set_xticks(x)
    axis.set_xticklabels(groups)
    axis.set_ylabel("MRI-only AUROC (mean across timings/seeds)")
    axis.set_title("Clinical-matched response discrimination within HR/HER2 profiles")
    axis.set_ylim(0.35, 1.0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(title="Arm")
    _save(figure, output)


def _figure_generalization(gaps: pd.DataFrame, output: Path) -> None:
    frame = _primary(gaps, name="generalization gaps")
    frame = frame.loc[
        frame["population"].astype(str).eq("full_808")
        & frame["model_family"].eq("M")
        & frame["arm"].isin(ARMS)
    ].copy()
    if set(frame["arm"]) != set(ARMS):
        raise FigureContractError("generalization figure requires full_808 M rows for all arms")
    summary = frame.groupby("arm", sort=False)[
        ["train_auroc", "validation_auroc", "test_auroc"]
    ].mean().reindex(ARMS)
    if summary.isna().any().any():
        raise FigureContractError("generalization figure has incomplete arm/split cells")

    x = np.arange(len(ARMS), dtype=float)
    width = 0.24
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    for offset, column, label, color in zip(
        (-width, 0.0, width),
        ("train_auroc", "validation_auroc", "test_auroc"),
        ("Train", "Validation", "Test/OOF"),
        ("#4c78a8", "#f2cf5b", "#e45756"),
        strict=True,
    ):
        axis.bar(x + offset, summary[column], width=width, label=label, color=color)
    axis.set_xticks(x)
    axis.set_xticklabels(ARMS)
    axis.set_ylabel("AUROC (mean across primary timings/seeds)")
    axis.set_title("Generalization gap by supervised ceiling arm")
    axis.set_ylim(0.35, 1.0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    _save(figure, output)


def _figure_profiles(profile: pd.DataFrame, output: Path) -> None:
    frame = _primary(profile, name="profile probe metrics")
    frame = frame.loc[frame["arm"].isin(("B0", "B2", "B3"))].copy()
    required_targets = {"HR", "HER2", "SUBTYPE"}
    normalized = frame["target"].astype(str).str.upper().str.strip()
    frame["target"] = normalized
    if not required_targets.issubset(set(frame["target"])):
        raise FigureContractError(
            f"profile figure requires at least targets {sorted(required_targets)}"
        )
    summary = frame.groupby(["target", "arm"], sort=False)["auroc"].mean()
    targets = [target for target in ("HR", "HER2", "SUBTYPE", "TREATMENT") if target in set(frame["target"])]
    missing = [
        (target, arm)
        for target in targets
        for arm in ("B0", "B2", "B3")
        if (target, arm) not in summary.index
    ]
    if missing:
        raise FigureContractError(f"profile figure misses target/arm cells: {missing}")

    x = np.arange(len(targets), dtype=float)
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    for arm, color, marker in (
        ("B0", "#666666", "o"),
        ("B2", "#f58518", "s"),
        ("B3", "#e45756", "^"),
    ):
        values = [float(summary.loc[(target, arm)]) for target in targets]
        axis.plot(x, values, label=arm, color=color, marker=marker, linewidth=1.8)
    axis.set_xticks(x)
    axis.set_xticklabels(["Subtype" if target == "SUBTYPE" else target.title() for target in targets])
    axis.set_ylabel("Probe AUROC (mean across primary timings/seeds)")
    axis.set_title("Clinical-profile decodability before/after pCR supervision")
    axis.set_ylim(0.35, 1.0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(title="MRI arm")
    _save(figure, output)


def _figure_ceiling_gap(aggregate: pd.DataFrame, output: Path) -> None:
    frame = _primary(aggregate, name="aggregate metrics")
    frame = frame.loc[
        frame["population"].astype(str).eq("full_808")
        & frame["model_family"].eq("M")
        & frame["arm"].isin(("B0", "B3"))
    ].copy()
    keys = ["seed", "timing", "arm"]
    if frame.duplicated(keys).any():
        raise FigureContractError("ceiling-gap aggregate repeats a seed/timing/arm cell")
    wide = frame.pivot(index=["seed", "timing"], columns="arm", values="auroc")
    if not {"B0", "B3"}.issubset(wide.columns) or wide[["B0", "B3"]].isna().any().any():
        raise FigureContractError("ceiling-gap figure requires paired B0 and B3 seed/timing cells")
    wide["delta"] = wide["B3"] - wide["B0"]
    summary = wide.groupby(level="timing")["delta"].agg(["mean", "std", "count"]).reindex(PRIMARY_TIMINGS)
    if summary["mean"].isna().any():
        raise FigureContractError("ceiling-gap figure misses a primary timing")

    figure, axis = plt.subplots(figsize=(8.0, 4.6))
    axis.bar(
        range(len(PRIMARY_TIMINGS)),
        summary["mean"],
        yerr=summary["std"].fillna(0.0),
        capsize=4,
        color=("#9ecae9", "#fdae6b", "#e45756"),
    )
    axis.axhline(0.0, color="black", linewidth=1.0)
    _timing_axis(axis)
    axis.set_ylabel("ΔAUROC: B3 supervised ceiling - B0 World Model")
    axis.set_title("Outcome structure left unorganized by the current pCR-free representation")
    axis.grid(axis="y", alpha=0.25)
    _save(figure, output)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_figures(
    experiment_root: str | Path = EXPERIMENT_ROOT,
    *,
    aggregate_csv: str | Path = DEFAULT_INPUTS["aggregate"],
    bootstrap_csv: str | Path = DEFAULT_INPUTS["bootstrap"],
    profile_csv: str | Path = DEFAULT_INPUTS["profile"],
    gaps_csv: str | Path = DEFAULT_INPUTS["gaps"],
    subgroups_csv: str | Path = DEFAULT_INPUTS["subgroups"],
    output_dir: str | Path = "figures",
) -> pd.DataFrame:
    """Validate five public aggregate tables and generate all seven figures."""

    root = Path(experiment_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"experiment root does not exist: {root}")
    aggregate, aggregate_path = _load_csv(
        root,
        aggregate_csv,
        name="aggregate metrics",
        required=("population", "seed", "arm", "timing", "model_family", "auroc"),
    )
    bootstrap, bootstrap_path = _load_csv(
        root,
        bootstrap_csv,
        name="bootstrap summary",
        required=(
            "comparison",
            "population",
            "seed",
            "arm",
            "timing",
            "delta",
            "ci_lower",
            "ci_upper",
        ),
        aliases={
            "comparison_name": "comparison",
            "point": "delta",
            "delta_auroc": "delta",
        },
    )
    if "metric" not in bootstrap.columns:
        bootstrap["metric"] = "auroc"
    profile, profile_path = _load_csv(
        root,
        profile_csv,
        name="profile probe metrics",
        required=("seed", "arm", "timing", "target"),
    )
    if {"metric", "value"}.issubset(profile.columns):
        pass
    elif "auroc" in profile.columns:
        profile["metric"] = "auroc"
        profile["value"] = profile["auroc"]
    elif "probe_auroc" in profile.columns:
        profile["metric"] = "auroc"
        profile["value"] = profile["probe_auroc"]
    else:
        raise FigureContractError(
            "profile probe metrics requires metric/value (or an auroc alias)"
        )
    gaps, gaps_path = _load_csv(
        root,
        gaps_csv,
        name="generalization gaps",
        required=(
            "population",
            "seed",
            "arm",
            "timing",
            "model_family",
            "train_auroc",
            "validation_auroc",
            "test_auroc",
        ),
    )
    subgroups, subgroup_path = _load_csv(
        root,
        subgroups_csv,
        name="subgroup metrics",
        required=(
            "seed",
            "arm",
            "timing",
            "subgroup",
            "auroc",
        ),
    )
    _numeric(aggregate, ("seed", "auroc"), name="aggregate metrics")
    _numeric(
        bootstrap,
        ("seed", "delta", "ci_lower", "ci_upper"),
        name="bootstrap summary",
    )
    _numeric(profile, ("seed",), name="profile probe metrics")
    profile_metric = profile["metric"].astype(str).str.lower()
    plotted_profile = profile_metric.isin(("auroc", "macro_ovr_auroc"))
    profile_values = pd.to_numeric(
        profile.loc[plotted_profile, "value"], errors="coerce"
    ).to_numpy(float)
    if not np.isfinite(profile_values).all():
        raise FigureContractError("plotted profile probe metrics.value must be finite")
    profile["auroc"] = np.nan
    profile.loc[plotted_profile, "auroc"] = profile_values
    _numeric(
        gaps,
        ("seed", "train_auroc", "validation_auroc", "test_auroc"),
        name="generalization gaps",
    )
    # Ineligible subgroups may legitimately carry NaN metrics.  Validate only
    # rows the registered figure will display.
    if "eligible" not in subgroups.columns:
        _numeric(subgroups, ("seed", "auroc"), name="subgroup metrics")
    else:
        _numeric(subgroups, ("seed",), name="subgroup metrics")
        eligible = subgroups["eligible"].astype(str).str.lower().isin(("true", "1"))
        displayed = pd.to_numeric(subgroups.loc[eligible, "auroc"], errors="coerce").to_numpy(float)
        if not np.isfinite(displayed).all():
            raise FigureContractError("eligible subgroup metrics.auroc must be finite")
        subgroups.loc[eligible, "auroc"] = displayed
    # ``subgroup_refits.csv`` has an explicit MRI-only/full-cohort contract, so
    # those redundant columns are deliberately absent from its public schema.
    if "population" not in subgroups.columns:
        subgroups["population"] = "full_808"
    if "model_family" not in subgroups.columns:
        subgroups["model_family"] = "M"

    figures = _inside(root, output_dir, must_exist=False)
    figures.mkdir(parents=True, exist_ok=True)
    functions = (
        lambda path: _figure_mri_only(aggregate, path),
        lambda path: _forest_delta(
            bootstrap,
            path,
            aliases={"C+M-C", "CLINICAL_COMPLEMENTARITY"},
            title="Clinical complementarity: C+M - C",
            expected_population="full_808",
        ),
        lambda path: _forest_delta(
            bootstrap,
            path,
            aliases={"C+F+M-(C+F)", "BEYOND_FTV"},
            title="Beyond-FTV complementarity: C+F+M - (C+F)",
            expected_population="ftv_complete_375",
        ),
        lambda path: _figure_subgroups(subgroups, path),
        lambda path: _figure_generalization(gaps, path),
        lambda path: _figure_profiles(profile, path),
        lambda path: _figure_ceiling_gap(aggregate, path),
    )
    manifest_rows: list[dict[str, Any]] = []
    source_map = (
        (aggregate_path,),
        (bootstrap_path,),
        (bootstrap_path,),
        (subgroup_path,),
        (gaps_path,),
        (profile_path,),
        (aggregate_path,),
    )
    for (filename, title), function, sources in zip(
        FIGURE_SPECS, functions, source_map, strict=True
    ):
        path = _inside(root, figures / filename, must_exist=False)
        function(path)
        manifest_rows.append(
            {
                "filename": filename,
                "title": title,
                "relative_path": str(path.relative_to(root)),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
                "source_csvs": ";".join(str(source.relative_to(root)) for source in sources),
                "public_aggregate_only": True,
                "contains_patient_rows": False,
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = _inside(root, figures / "figure_manifest.csv", must_exist=False)
    manifest.to_csv(manifest_path, index=False)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", default=str(EXPERIMENT_ROOT))
    parser.add_argument("--aggregate-csv", default=str(DEFAULT_INPUTS["aggregate"]))
    parser.add_argument("--bootstrap-csv", default=str(DEFAULT_INPUTS["bootstrap"]))
    parser.add_argument("--profile-csv", default=str(DEFAULT_INPUTS["profile"]))
    parser.add_argument("--gaps-csv", default=str(DEFAULT_INPUTS["gaps"]))
    parser.add_argument("--subgroups-csv", default=str(DEFAULT_INPUTS["subgroups"]))
    parser.add_argument("--output-dir", default="figures")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        manifest = generate_figures(
            arguments.experiment_root,
            aggregate_csv=arguments.aggregate_csv,
            bootstrap_csv=arguments.bootstrap_csv,
            profile_csv=arguments.profile_csv,
            gaps_csv=arguments.gaps_csv,
            subgroups_csv=arguments.subgroups_csv,
            output_dir=arguments.output_dir,
        )
    except (FileNotFoundError, FigureContractError, ValueError) as error:
        print(f"FIGURE_GENERATION_FAILED: {error}", file=sys.stderr)
        return 2
    print(f"generated {len(manifest)} aggregate-only figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

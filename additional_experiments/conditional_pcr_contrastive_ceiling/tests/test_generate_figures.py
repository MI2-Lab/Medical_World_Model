from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd
import pytest


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "generate_figures.py"
SPEC = importlib.util.spec_from_file_location("conditional_generate_figures", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
generate_figures_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = generate_figures_module
SPEC.loader.exec_module(generate_figures_module)


def _write_synthetic_aggregates(root: Path) -> None:
    metrics = root / "metrics"
    metrics.mkdir(parents=True)
    seeds = (2026, 3026)
    timings = ("T0", "T0_T1", "T0_T2")

    aggregate_rows = []
    for seed_index, seed in enumerate(seeds):
        for arm_index, arm in enumerate(("B0", "B1", "B2", "B3")):
            for timing_index, timing in enumerate(timings):
                aggregate_rows.append(
                    {
                        "population": "full_808",
                        "seed": seed,
                        "arm": arm,
                        "timing": timing,
                        "model_family": "M",
                        "auroc": 0.52 + 0.025 * arm_index + 0.015 * timing_index + 0.005 * seed_index,
                    }
                )
    pd.DataFrame(aggregate_rows).to_csv(metrics / "aggregate_metrics.csv", index=False)

    bootstrap_rows = []
    for comparison, population, base in (
        ("C+M-C", "full_808", 0.025),
        ("C+F+M-(C+F)", "ftv_complete_375", 0.035),
    ):
        for seed_index, seed in enumerate(seeds):
            for arm_index, arm in enumerate(("B1", "B2", "B3")):
                for timing_index, timing in enumerate(timings):
                    delta = base + 0.005 * seed_index + 0.008 * arm_index + 0.003 * timing_index
                    bootstrap_rows.append(
                        {
                            "comparison": comparison,
                            "population": population,
                            "seed": seed,
                            "arm": arm,
                            "timing": timing,
                            "delta_auroc": delta,
                            "ci_lower": delta - 0.02,
                            "ci_upper": delta + 0.02,
                        }
                    )
    pd.DataFrame(bootstrap_rows).to_csv(metrics / "paired_bootstrap.csv", index=False)

    profile_rows = []
    for seed_index, seed in enumerate(seeds):
        for arm_index, arm in enumerate(("B0", "B2", "B3")):
            for timing in timings:
                for target_index, target in enumerate(("HR", "HER2", "subtype")):
                    profile_rows.append(
                        {
                            "seed": seed,
                            "arm": arm,
                            "timing": timing,
                            "target": target,
                            "metric": "macro_ovr_auroc" if target == "subtype" else "auroc",
                            "value": 0.58 + 0.03 * arm_index + 0.01 * target_index + 0.005 * seed_index,
                        }
                    )
    # Mirror the runner's explicit treatment-unsuitability audit row.
    profile_rows.append(
        {
            "seed": -1,
            "arm": "ALL",
            "timing": "ALL",
            "target": "treatment",
            "metric": "not_run",
            "value": float("nan"),
        }
    )
    pd.DataFrame(profile_rows).to_csv(metrics / "clinical_profile_probes.csv", index=False)

    gap_rows = []
    for seed in seeds:
        for arm_index, arm in enumerate(("B0", "B1", "B2", "B3")):
            for timing_index, timing in enumerate(timings):
                test = 0.52 + 0.025 * arm_index + 0.01 * timing_index
                gap_rows.append(
                    {
                        "population": "full_808",
                        "seed": seed,
                        "arm": arm,
                        "timing": timing,
                        "model_family": "M",
                        "train_auroc": test + 0.12,
                        "validation_auroc": test + 0.03,
                        "test_auroc": test,
                    }
                )
    pd.DataFrame(gap_rows).to_csv(metrics / "generalization_gaps.csv", index=False)

    subgroup_rows = []
    for seed in seeds:
        for arm_index, arm in enumerate(("B0", "B2", "B3")):
            for timing in timings:
                for group_index, subgroup in enumerate(("HR-/HER2-", "HR+/HER2-", "HER2+")):
                    subgroup_rows.append(
                        {
                            "seed": seed,
                            "arm": arm,
                            "timing": timing,
                            "subgroup": subgroup,
                            "n": 50 + 10 * group_index,
                            "eligible": True,
                            "auroc": 0.51 + 0.035 * arm_index + 0.015 * group_index,
                        }
                    )
    pd.DataFrame(subgroup_rows).to_csv(metrics / "subgroup_refits.csv", index=False)


def test_generate_all_seven_figures_from_public_aggregates(tmp_path: Path) -> None:
    root = tmp_path / "conditional_pcr_contrastive_ceiling"
    root.mkdir()
    _write_synthetic_aggregates(root)

    manifest = generate_figures_module.generate_figures(root)

    assert len(manifest) == 7
    assert manifest["public_aggregate_only"].all()
    assert not manifest["contains_patient_rows"].any()
    assert set(manifest["filename"]) == {
        specification[0] for specification in generate_figures_module.FIGURE_SPECS
    }
    for row in manifest.itertuples():
        path = root / row.relative_path
        assert path.is_file()
        assert path.stat().st_size == row.bytes
        assert len(row.sha256) == 64
    assert (root / "figures" / "figure_manifest.csv").is_file()


def test_generator_rejects_patient_level_schema_and_path_escape(tmp_path: Path) -> None:
    root = tmp_path / "experiment"
    root.mkdir()
    _write_synthetic_aggregates(root)
    aggregate = pd.read_csv(root / "metrics" / "aggregate_metrics.csv")
    aggregate["patient_id"] = [f"P{index}" for index in range(len(aggregate))]
    aggregate.to_csv(root / "metrics" / "aggregate_metrics.csv", index=False)

    with pytest.raises(generate_figures_module.FigureContractError, match="patient-level"):
        generate_figures_module.generate_figures(root)
    with pytest.raises(generate_figures_module.FigureContractError, match="escapes"):
        generate_figures_module.generate_figures(
            root, aggregate_csv=tmp_path / "outside.csv"
        )


def test_complementarity_figures_require_every_supervised_arm_cell(
    tmp_path: Path,
) -> None:
    root = tmp_path / "experiment"
    root.mkdir()
    _write_synthetic_aggregates(root)
    bootstrap_path = root / "metrics" / "paired_bootstrap.csv"
    bootstrap = pd.read_csv(bootstrap_path)
    bootstrap = bootstrap.loc[
        ~(
            bootstrap["comparison"].eq("C+M-C")
            & bootstrap["arm"].eq("B1")
            & bootstrap["seed"].eq(2026)
            & bootstrap["timing"].eq("T0")
        )
    ]
    bootstrap.to_csv(bootstrap_path, index=False)

    with pytest.raises(generate_figures_module.FigureContractError, match="exact B1/B2/B3"):
        generate_figures_module.generate_figures(root)

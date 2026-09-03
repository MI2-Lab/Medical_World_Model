from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _module():
    source = ROOT / "scripts" / "generate_figures.py"
    spec = importlib.util.spec_from_file_location("residual_sph_figures_test", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FIGURES = _module()


def test_aggregate_figures_are_svg_and_identifier_free(tmp_path: Path) -> None:
    representation = {
        "by_seed": {
            str(seed): {
                "E1": 0.01 + index * 0.01,
                "E2": -0.01 + index * 0.01,
                "E3": 0.07 + index * 0.01,
                "E4": 0.04 + index * 0.01,
            }
            for index, seed in enumerate(FIGURES.SEEDS)
        }
    }
    pcr = {
        key: {
            timing: {
                str(seed): 0.005 * (timing_index + seed_index + 1)
                for seed_index, seed in enumerate(FIGURES.SEEDS)
            }
            for timing_index, timing in enumerate(FIGURES.TIMINGS)
        }
        for key in (
            "E5_S2_minus_S0_MRI_only",
            "E6_S2_C_plus_F_plus_M_minus_C_plus_F",
        )
    }
    rows = [
        {
            "arm": arm,
            "seed_base": seed,
            "task": "sph_res",
            "endpoint": "T0",
            "space": "residual",
            "residual_space_spearman": 0.1 + 0.02 * arm_index,
        }
        for arm_index, arm in enumerate(FIGURES.ARMS)
        for seed in FIGURES.SEEDS
    ]
    outputs = FIGURES.render_aggregate_figures(
        representation_effects=representation,
        pcr_effects=pcr,
        sph_metrics=pd.DataFrame(rows),
        output_dir=tmp_path,
    )
    assert tuple(path.name for path in outputs) == FIGURES.FIGURE_NAMES
    for path in outputs:
        text = path.read_text(encoding="utf-8")
        assert text.lstrip().startswith("<?xml")
        assert "patient_id" not in text


def test_figure_generator_rejects_patient_level_table(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "arm": arm,
                "seed_base": seed,
                "task": "sph_res",
                "endpoint": "T0",
                "space": "residual",
                "residual_space_spearman": 0.1,
                "patient_id": "forbidden",
            }
            for arm in FIGURES.ARMS
            for seed in FIGURES.SEEDS
        ]
    )
    effects = {
        "by_seed": {
            str(seed): {name: 0.1 for name in ("E1", "E2", "E3", "E4")}
            for seed in FIGURES.SEEDS
        }
    }
    pcr = {
        key: {
            timing: {str(seed): 0.1 for seed in FIGURES.SEEDS}
            for timing in FIGURES.TIMINGS
        }
        for key in (
            "E5_S2_minus_S0_MRI_only",
            "E6_S2_C_plus_F_plus_M_minus_C_plus_F",
        )
    }
    try:
        FIGURES.render_aggregate_figures(
            representation_effects=effects,
            pcr_effects=pcr,
            sph_metrics=frame,
            output_dir=tmp_path,
        )
    except PermissionError:
        pass
    else:
        raise AssertionError("patient-level figure input was accepted")

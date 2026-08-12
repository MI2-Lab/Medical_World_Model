#!/usr/bin/env python3
"""Render the ten LOCAL confirmation figures from deidentified tables."""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = EXPERIMENT_ROOT / "src"
SCRIPTS_ROOT = EXPERIMENT_ROOT / "scripts"
FORMAL_METRICS_ROOT = EXPERIMENT_ROOT / "metrics"
FORMAL_FIGURE_ROOT = EXPERIMENT_ROOT / "figures"
for source in (SRC_ROOT, SCRIPTS_ROOT):
    value = str(source.resolve())
    while value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)

from freeze_preregistration import verify as verify_preregistration  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, default=EXPERIMENT_ROOT / "metrics")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_ROOT / "figures")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    preregistration = verify_preregistration()
    import lg_response_pilot.security as confirmation_security

    if (
        Path(str(getattr(confirmation_security, "__file__", ""))).resolve()
        != (SRC_ROOT / "lg_response_pilot" / "security.py").resolve()
    ):
        raise ImportError(
            "confirmation security module was shadowed before figure rendering"
        )
    from lg_response_pilot.security import resolve_contained_path

    args = parse_args()
    from lg_response_pilot.analysis import TABLE_FILENAMES
    from lg_response_pilot.figures import FIGURE_FILENAMES, render_required_figures

    import lg_response_pilot.analysis as confirmation_analysis_module
    import lg_response_pilot.figures as confirmation_figures_module

    for module, label in (
        (confirmation_analysis_module, "confirmation analysis"),
        (confirmation_figures_module, "confirmation figures"),
    ):
        confirmation_security.require_module_within(
            module, SRC_ROOT / "lg_response_pilot", label=label
        )

    metrics_dir = resolve_contained_path(
        args.metrics_dir,
        FORMAL_METRICS_ROOT,
        label="public metrics input directory",
        allow_root=True,
    )
    output_dir = resolve_contained_path(
        args.output_dir,
        FORMAL_FIGURE_ROOT,
        label="public figure output directory",
        allow_root=True,
    )

    required = {
        "table1": metrics_dir / TABLE_FILENAMES["table1"],
        "table2": metrics_dir / TABLE_FILENAMES["table2"],
        "table3": metrics_dir / TABLE_FILENAMES["table3"],
        "table4": metrics_dir / TABLE_FILENAMES["table4"],
        "table6": metrics_dir / TABLE_FILENAMES["table6"],
        "fold_effects": metrics_dir / "paired_fold_effects.csv",
        "histories": metrics_dir / "training_trajectories.csv",
    }
    summary_path = metrics_dir / "aggregation_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            "aggregation_summary.json is required for figure provenance"
        )
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("aggregation summary is invalid") from error
    if (
        not isinstance(summary, dict)
        or summary.get("status") != "COMPLETE"
        or summary.get("preregistration_lock") != "PREREGISTRATION_LOCK.json"
        or summary.get("preregistration_lock_sha256") != preregistration["lock_sha256"]
    ):
        raise ValueError("figure inputs use another/incomplete aggregation lock")
    if absent := [str(path) for path in required.values() if not path.is_file()]:
        raise FileNotFoundError(f"required figure input is missing: {absent[0]}")
    hashes = summary.get("artifact_sha256")
    if not isinstance(hashes, dict) or any(
        hashes.get(path.name) != hashlib.sha256(path.read_bytes()).hexdigest()
        for path in required.values()
    ):
        raise ValueError("figure input hash differs from the completed aggregation")
    frames = {name: pd.read_csv(path) for name, path in required.items()}
    if not args.execute:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "PREFLIGHT_PASS",
                    "execution_requested": False,
                    "preregistration_lock": "PREREGISTRATION_LOCK.json",
                    "preregistration_lock_sha256": preregistration["lock_sha256"],
                    "figure_targets": list(FIGURE_FILENAMES),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    paths = render_required_figures(
        table1=frames["table1"],
        table2=frames["table2"],
        table3=frames["table3"],
        table4=frames["table4"],
        table6=frames["table6"],
        fold_effects=frames["fold_effects"],
        histories=frames["histories"],
        output_dir=output_dir,
    )
    print(
        json.dumps(
            {
                "schema_version": 1,
                "status": "COMPLETE",
                "preregistration_lock": "PREREGISTRATION_LOCK.json",
                "preregistration_lock_sha256": preregistration["lock_sha256"],
                "figures": [path.name for path in paths],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

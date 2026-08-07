#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corejepa.config import load_config
from corejepa.readout import fit_frozen_landmark_readout


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and evaluate the Frozen Landmark Readout.")
    parser.add_argument("--config", default="configs/paper_v1.yaml")
    parser.add_argument("--states")
    parser.add_argument("--splits")
    args = parser.parse_args()
    config = load_config(args.config)
    run_dir = Path(config.train.output_dir)
    summary = fit_frozen_landmark_readout(
        args.states or run_dir / "frozen_states.npz",
        args.splits or run_dir / "splits.json",
        run_dir,
        config.readout,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

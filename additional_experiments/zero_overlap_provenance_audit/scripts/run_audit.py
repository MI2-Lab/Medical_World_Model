#!/usr/bin/env python3
"""Run the anonymous, outcome-free single-case geometry provenance audit."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PRIOR_ROOT = EXPERIMENT_ROOT.parent / "c1b_model_ready_ftv_sanity"
sys.path.insert(0, str(PRIOR_ROOT / "src"))
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from zero_overlap_audit.runner import run_audit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_audit(experiment_root=EXPERIMENT_ROOT, overwrite=args.overwrite)
    print(
        "[6/7] audit classification sealed: "
        f"{result['decision']['decision']} / {result['decision']['root_cause_class']}"
    )
    print("[7/7] Stage B not run; prior Stage-A decision unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

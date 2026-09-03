#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from raw_spatial_pcr.metrics import attention_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate descriptive attention diagnostics from private arrays.")
    parser.add_argument("--attention-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "metrics" / "attention_diagnostics.csv")
    args = parser.parse_args()
    rows = []
    for path in sorted(args.attention_dir.rglob("*.private.npz")):
        payload = np.load(path, allow_pickle=False)
        if "attention" not in payload:
            continue
        values = attention_diagnostics(payload["attention"])
        values["artifact_kind"] = path.parent.name
        rows.append(values)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"wrote {len(rows)} attention diagnostic rows")


if __name__ == "__main__":
    main()


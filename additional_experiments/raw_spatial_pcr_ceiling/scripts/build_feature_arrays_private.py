#!/usr/bin/env python3
"""Pack verified per-patient maps into a private memory-mapped feature array."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="must end in .private.npy")
    args = parser.parse_args()
    if not args.output.name.endswith(".private.npy"):
        raise SystemExit("feature arrays must remain private")
    manifest = pd.read_csv(args.manifest).drop_duplicates("patient_id").sort_values("row_index")
    if len(manifest) != 808:
        raise SystemExit("feature array requires exactly 808 patients")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    first = args.feature_dir / f"{manifest.iloc[0]['patient_id']}.private.npz"
    with np.load(first, allow_pickle=False) as payload:
        shape = tuple(int(value) for value in payload["feature_map"].shape)
    if shape != (4, 128, 14, 22, 20):
        raise SystemExit(f"unexpected feature map shape {shape}")
    array = np.lib.format.open_memmap(args.output, mode="w+", dtype=np.float16, shape=(808, *shape))
    for index, row in enumerate(manifest.itertuples(index=False)):
        source = args.feature_dir / f"{row.patient_id}.private.npz"
        if not source.is_file():
            raise SystemExit(f"feature map missing: {source}")
        with np.load(source, allow_pickle=False) as payload:
            value = np.asarray(payload["feature_map"], dtype=np.float16)
        if value.shape != shape:
            raise SystemExit(f"feature map shape drifted at {source}: {value.shape}")
        array[index] = value
        if (index + 1) % 50 == 0:
            print({"patients_packed": index + 1}, flush=True)
    array.flush()
    del array
    print({"status": "COMPLETE", "patients": 808, "shape": [808, *shape], "output": str(args.output)})


if __name__ == "__main__":
    main()

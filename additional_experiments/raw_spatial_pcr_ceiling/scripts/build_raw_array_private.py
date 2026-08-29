#!/usr/bin/env python3
"""Materialize one private raw C1B-H array to amortize ZIP/archive overhead."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from raw_spatial_pcr.training import CacheRecord, load_c1b_image  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="must end in .private.npy")
    args = parser.parse_args()
    if not args.output.name.endswith(".private.npy"):
        raise SystemExit("raw arrays must remain private")
    frame = pd.read_csv(args.manifest).drop_duplicates("patient_id").sort_values("row_index")
    if len(frame) != 808:
        raise SystemExit("raw array requires exactly 808 patients")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    array = np.lib.format.open_memmap(args.output, mode="w+", dtype=np.float32, shape=(808, 4, 7, 112, 176, 160))
    for index, row in enumerate(frame.itertuples(index=False)):
        path = Path(str(row.cache_path)).expanduser().resolve(strict=True)
        record = CacheRecord(str(row.patient_id), path, "", int(path.stat().st_size), int(path.stat().st_mtime_ns))
        array[index] = load_c1b_image(record)
        if (index + 1) % 10 == 0:
            print({"patients_packed": index + 1}, flush=True)
    array.flush()
    del array
    print({"status": "COMPLETE", "patients": 808, "shape": [808, 4, 7, 112, 176, 160], "output": str(args.output)})


if __name__ == "__main__":
    main()

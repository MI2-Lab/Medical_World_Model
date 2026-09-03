#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate private C1B-H cache geometry without reading clinical labels.")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "metrics" / "input_contract_preflight.json")
    args = parser.parse_args()
    files = sorted(args.cache_root.glob("*.npz"))
    if not files:
        raise SystemExit("no private C1B cache NPZ files found")
    first = np.load(files[0], allow_pickle=False)
    if "image" not in first or "valid_source_mask" not in first:
        raise SystemExit("C1B cache must contain image and valid_source_mask")
    image_shape = list(first["image"].shape)
    mask_shape = list(first["valid_source_mask"].shape)
    if image_shape != [4, 7, 112, 176, 160] or mask_shape != [4, 1, 112, 176, 160]:
        raise SystemExit(f"unexpected private C1B shapes: image={image_shape}, valid_source_mask={mask_shape}")
    payload = {"status": "PASS", "cache_file_count": len(files), "image_shape": image_shape, "valid_source_mask_shape": mask_shape, "feature_map_contract": [128, 14, 22, 20], "raw_images_read": 1, "clinical_labels_read": 0, "pcr_labels_read": 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


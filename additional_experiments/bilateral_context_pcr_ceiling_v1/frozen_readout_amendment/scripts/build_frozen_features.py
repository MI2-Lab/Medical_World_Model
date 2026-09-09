#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "src"))
from bilateral_context.contracts import atomic_json, canonical_sha256, primary_ids, private_token, sha256_file  # noqa: E402


def main() -> None:
    parent = ROOT.parent
    cache_dir = parent / "cache/bilateral_dino"
    ids = primary_ids()
    global_rows = []
    spatial_rows = []
    for patient_id in ids:
        path = cache_dir / f"{private_token(patient_id)}.private.npz"
        with np.load(path, allow_pickle=False) as payload:
            global_summary = np.asarray(payload["global_summary"], dtype=np.float32)
            spatial_tokens = np.asarray(payload["spatial_tokens"], dtype=np.float32)
        if global_summary.shape != (4, 2, 7, 32, 2304) or spatial_tokens.shape != (4, 2, 7, 32, 4, 4, 768):
            raise ValueError(f"cache shape mismatch for {patient_id}")
        global_side = global_summary.mean(axis=(2, 3))
        spatial_side = spatial_tokens.mean(axis=(2, 3, 4, 5))
        global_rows.append(np.concatenate((global_side.mean(axis=1), np.abs(global_side[:, 0] - global_side[:, 1])), axis=-1))
        spatial_rows.append(np.concatenate((spatial_side.mean(axis=1), np.abs(spatial_side[:, 0] - spatial_side[:, 1])), axis=-1))
    global_features = np.asarray(global_rows, dtype=np.float32)
    spatial_features = np.asarray(spatial_rows, dtype=np.float32)
    if not np.isfinite(global_features).all() or not np.isfinite(spatial_features).all():
        raise FloatingPointError("frozen feature aggregate is non-finite")
    output = ROOT / "features/frozen_bilateral_features.private.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, patient_token=np.asarray([private_token(pid) for pid in ids]), global_features=global_features, spatial_features=spatial_features, source="locked bilateral DINO cache", outcome_used=np.asarray(False), clinical_used=np.asarray(False))
    manifest = {"status": "COMPLETE", "patients": len(ids), "global_shape": list(global_features.shape), "spatial_shape": list(spatial_features.shape), "feature_sha256": sha256_file(output), "patient_tokens_sha256": canonical_sha256([private_token(pid) for pid in ids]), "outcome_fields_read": [], "clinical_fields_read": []}
    atomic_json(ROOT / "metrics/frozen_feature_check.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

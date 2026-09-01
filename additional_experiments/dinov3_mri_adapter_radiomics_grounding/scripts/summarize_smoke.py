#!/usr/bin/env python3
"""Publish de-identified round-trip facts from the three private smoke caches."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dinov3_rg.contracts import atomic_json  # noqa: E402


def only_npz(directory: Path) -> Path:
    paths = sorted(directory.glob("*.private.npz"))
    if len(paths) != 1:
        raise RuntimeError(f"smoke directory must contain exactly one archive: {directory}")
    return paths[0]


def main() -> None:
    with np.load(only_npz(ROOT / "cache/smoke_dinov3_summaries"), allow_pickle=False) as payload:
        dino = np.asarray(payload["summary"])
    with np.load(only_npz(ROOT / "cache/smoke_radiomics_rois"), allow_pickle=False) as payload:
        roi_valid = np.asarray(payload["radiomics_mask"], dtype=bool)
    with np.load(only_npz(ROOT / "cache/smoke_radiomics_raw"), allow_pickle=False) as payload:
        radiomics = np.asarray(payload["value"])
        names = payload["feature_name"].astype(str)
    forbidden = [
        name for name in names
        if any(token in name.lower() for token in ("shape", "wavelet", "log-"))
    ]
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "patients": 1,
        "dinov3_summary_shape": list(dino.shape),
        "dinov3_summary_dtype": str(dino.dtype),
        "dinov3_summary_finite": bool(np.isfinite(dino).all()),
        "roi_valid_visits": int(roi_valid.sum()),
        "radiomics_shape": list(radiomics.shape),
        "radiomics_feature_count": int(len(names)),
        "forbidden_radiomics_features": forbidden,
        "patient_identity_disclosed": False,
        "outcome_fields_read": [],
        "clinical_fields_read": [],
    }
    if dino.shape != (4, 7, 32, 2304) or dino.dtype != np.float16:
        raise RuntimeError("DINO smoke contract failed")
    if radiomics.shape != (4, 3, 651) or forbidden:
        raise RuntimeError("radiomics smoke contract failed")
    atomic_json(ROOT / "metrics/smoke_validation.json", payload)
    print(payload)


if __name__ == "__main__":
    main()

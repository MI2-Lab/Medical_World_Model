from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT.parents[1]
V2_ROOT = REPO_ROOT / "additional_experiments/dinov3_mri_adapter_radiomics_grounding_v2"
V4_ROOT = REPO_ROOT / "additional_experiments/dinov3_mri_adapter_factorized_radiomics_v4"
V2_SUMMARY_DIR = V2_ROOT / "cache/dinov3_summaries"
V2_TARGET_DIR = V2_ROOT / "features/private/fold_targets"
V4_C0_ROOT = V4_ROOT / "checkpoints/pilot"
FOLD_MANIFEST = Path("/data/data/Preprocessed/I-SPY2/_matched_breastdcedl_t0_dicomrepair_rgb224_seed2026/matched_patient_cv_splits_seed2026.csv")
SUMMARY_SHAPE = (4, 7, 32, 2304)
FOLDS = tuple(range(5))
PILOT_SEED = 2026
FORMAL_SEEDS = (7026, 8026, 9026, 10026, 11026)
VERSION = "v5-parallel-radiomics-20260903-r1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def private_token(patient_id: str) -> str:
    return hashlib.sha256(str(patient_id).encode()).hexdigest()


def load_protocol() -> dict[str, Any]:
    payload = json.loads((ROOT / "configs/protocol.json").read_text())
    if payload.get("experiment") != "dinov3_parallel_radiomics_adapter_v5":
        raise ValueError("V5 protocol identity drifted")
    if tuple(payload["data"]["summary_shape"]) != SUMMARY_SHAPE:
        raise ValueError("DINO summary shape drifted")
    if payload["model"]["parallel_state"] != "concat(c0_state_192, rad_state_64)":
        raise ValueError("parallel state contract drifted")
    return payload


def load_folds() -> pd.DataFrame:
    frame = pd.read_csv(FOLD_MANIFEST, usecols=["patient_id", "fold", "split"], dtype={"patient_id": str})
    frame["fold"] = frame["fold"].astype(int)
    frame["split"] = frame["split"].replace({"validation": "val"})
    if len(frame) != 808 * 5 or frame.duplicated(["patient_id", "fold"]).any():
        raise ValueError("fold manifest contract failed")
    return frame


def split_ids(fold: int) -> dict[str, tuple[str, ...]]:
    frame = load_folds(); current = frame.loc[frame["fold"].eq(int(fold))]
    return {split: tuple(sorted(current.loc[current["split"].eq(split), "patient_id"].astype(str))) for split in ("train", "val", "test")}


def atomic_json(path: str | Path, payload: Any) -> None:
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(destination)

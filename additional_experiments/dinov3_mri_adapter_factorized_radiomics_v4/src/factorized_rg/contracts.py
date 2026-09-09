from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = PACKAGE_ROOT.parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[3]
V2_ROOT = REPO_ROOT / "additional_experiments/dinov3_mri_adapter_radiomics_grounding_v2"
V2_SUMMARY_DIR = V2_ROOT / "cache/dinov3_summaries"
V2_TARGET_DIR = V2_ROOT / "features/private/fold_targets"
V2_D1_CHECKPOINT_ROOT = V2_ROOT / "checkpoints/formal"
FOLD_MANIFEST = Path("/data/data/Preprocessed/I-SPY2/_matched_breastdcedl_t0_dicomrepair_rgb224_seed2026/matched_patient_cv_splits_seed2026.csv")
TRAIN_ONLY_MANIFEST = REPO_ROOT / "additional_experiments/c1b_model_ready_ftv_sanity/manifests/ispy1_base_eligibility_patients.private.csv"
TECHNICAL_ELIGIBILITY = REPO_ROOT / "additional_experiments/c1b_overlap_eligibility_ftv_stageb/manifests/technical_eligibility_patients.private.csv"
SUMMARY_SHAPE = (4, 7, 32, 2304)
VISITS = ("T0", "T1", "T2", "T3")
FOLDS = tuple(range(5))
PILOT_SEED = 2026
PILOT_ARMS = ("F0", "F005", "F010", "F025")
PILOT_WEIGHTS = {"F0": 0.0, "F005": 0.05, "F010": 0.10, "F025": 0.25}
FORMAL_SEEDS = (7026, 8026, 9026, 10026, 11026)
ARMS = ("F0", "RAD")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def private_token(patient_id: str) -> str:
    return hashlib.sha256(str(patient_id).encode()).hexdigest()


def load_protocol() -> dict[str, Any]:
    payload = json.loads((EXPERIMENT_ROOT / "configs/protocol.json").read_text())
    if payload.get("experiment") != "dinov3_mri_adapter_factorized_radiomics_v4":
        raise ValueError("V4 protocol identity drifted")
    if tuple(payload["data"]["summary_shape"]) != SUMMARY_SHAPE:
        raise ValueError("summary shape drifted")
    if float(payload["model"]["ftv_loss_weight"]) != 0.0:
        raise ValueError("FTV loss must remain zero")
    return payload


def load_folds() -> Any:
    import pandas as pd
    frame = pd.read_csv(FOLD_MANIFEST, usecols=["patient_id", "fold", "split"], dtype={"patient_id": str})
    frame["fold"] = frame["fold"].astype(int)
    frame["split"] = frame["split"].replace({"validation": "val"})
    if len(frame) != 808 * 5 or frame.duplicated(["patient_id", "fold"]).any():
        raise ValueError("fold manifest contract failed")
    return frame


def load_train_only_ids() -> tuple[str, ...]:
    import pandas as pd
    frame = pd.read_csv(TRAIN_ONLY_MANIFEST, dtype={"patient_id": str})
    if "eligible" in frame:
        frame = frame.loc[frame["eligible"].astype(bool)]
    technical = pd.read_csv(TECHNICAL_ELIGIBILITY, usecols=["patient_id", "cohort", "eligible"], dtype={"patient_id": str, "cohort": str})
    technical_ids = set(technical.loc[technical["cohort"].eq("I-SPY1") & technical["eligible"].astype(bool), "patient_id"].astype(str))
    ids = tuple(sorted(set(frame["patient_id"].astype(str)).intersection(technical_ids)))
    if len(ids) != 139:
        raise ValueError(f"expected 139 I-SPY1 train-only IDs, got {len(ids)}")
    return ids


def split_ids(fold: int) -> dict[str, tuple[str, ...]]:
    frame = load_folds()
    current = frame.loc[frame["fold"].eq(int(fold))]
    result = {s: tuple(sorted(current.loc[current["split"].eq(s), "patient_id"].astype(str))) for s in ("train", "val", "test")}
    result["train"] = tuple(sorted((*result["train"], *load_train_only_ids())))
    return result


def atomic_json(path: str | Path, payload: Any) -> None:
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(destination)

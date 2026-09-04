from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parents[1]
V6 = REPO / "additional_experiments/dinov3_spatial_parallel_radiomics_v6"
RAW_ROOT = Path("/data/data/Preprocessed/I-SPY2")
RAW_INPUT_MANIFEST = REPO / "additional_experiments/raw_spatial_pcr_ceiling/manifests/formal_input.private.csv"
FOLD_MANIFEST = Path("/data/data/Preprocessed/I-SPY2/_matched_breastdcedl_t0_dicomrepair_rgb224_seed2026/matched_patient_cv_splits_seed2026.csv")
V6_TARGET = V6 / "features/private/fold_targets/fold_0_targets.private.npz"
V6_SPATIAL = V6 / "cache/dinov3_spatial"
V2_PROTOCOL = REPO / "additional_experiments/dinov3_mri_adapter_radiomics_grounding_v2/configs/protocol.json"

VISITS = ("T0", "T1", "T2", "T3")
SEEDS = (7026, 8026, 9026, 10026, 11026)
FOLDS = tuple(range(5))
SUMMARY_SHAPE = (4, 2, 7, 32, 2304)
SPATIAL_SHAPE = (4, 2, 7, 32, 4, 4, 768)
PRIMARY_N = 375


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def private_token(patient_id: str) -> str:
    return hashlib.sha256(str(patient_id).encode()).hexdigest()


def atomic_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def protocol() -> dict[str, Any]:
    payload = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))
    if payload.get("experiment") != "bilateral_context_pcr_ceiling_v1":
        raise ValueError("experiment protocol identity drifted")
    if tuple(payload["training"]["seeds"]) != SEEDS or tuple(payload["cohort"]["outer_folds"]) != FOLDS:
        raise ValueError("seed/fold contract drifted")
    return payload


def primary_ids() -> tuple[str, ...]:
    with np.load(V6_TARGET, allow_pickle=False) as payload:
        ids = tuple(map(str, payload["patient_id"].tolist()))
    if len(ids) != PRIMARY_N or len(set(ids)) != PRIMARY_N:
        raise ValueError("V6 primary target cohort is not 375 unique patients")
    return tuple(sorted(ids))


def raw_frame() -> pd.DataFrame:
    frame = pd.read_csv(
        RAW_INPUT_MANIFEST,
        usecols=["patient_id", "preprocessed_dir", "row_index"],
        dtype={"patient_id": str, "preprocessed_dir": str},
    )
    if frame["patient_id"].duplicated().any():
        frame = frame.sort_values("row_index").drop_duplicates("patient_id")
    result = frame.loc[frame["patient_id"].isin(primary_ids())].copy()
    if len(result) != PRIMARY_N:
        raise ValueError(f"raw input manifest does not cover primary cohort: {len(result)}")
    return result.set_index("patient_id").sort_index()


def fold_frame() -> pd.DataFrame:
    frame = pd.read_csv(FOLD_MANIFEST, usecols=["patient_id", "fold", "split"], dtype={"patient_id": str})
    frame["fold"] = frame["fold"].astype(int)
    frame["split"] = frame["split"].replace({"validation": "val"})
    result = frame.loc[frame["patient_id"].isin(primary_ids())].copy()
    if len(result) != PRIMARY_N * len(FOLDS) or result.duplicated(["patient_id", "fold"]).any():
        raise ValueError("primary fold manifest contract failed")
    return result


def source_path(patient_id: str, visit: str) -> Path:
    path = RAW_ROOT / str(patient_id) / visit / f"{patient_id}_{visit}_original_DCE.nii"
    return path


def v6_cache_path(patient_id: str) -> Path:
    return V6_SPATIAL / f"{private_token(patient_id)}.private.npz"


def ordered_hash(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()

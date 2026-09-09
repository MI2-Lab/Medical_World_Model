from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parents[1]
V2 = REPO / "additional_experiments/dinov3_mri_adapter_radiomics_grounding_v2"
V4 = REPO / "additional_experiments/dinov3_mri_adapter_factorized_radiomics_v4"
V5 = REPO / "additional_experiments/dinov3_parallel_radiomics_adapter_v5"
SUMMARY_DIR = V2 / "cache/dinov3_summaries"
RAW_DIR = V2 / "cache/radiomics_raw"
ROI_DIR = V2 / "cache/radiomics_rois"
TARGET_DIR = ROOT / "features/private/fold_targets"
SPATIAL_DIR = ROOT / "cache/dinov3_spatial"
C1B_MANIFEST = Path("/data/data/Preprocessed/I-SPY2/_matched_breastdcedl_t0_dicomrepair_rgb224_seed2026/matched_patient_cv_splits_seed2026.csv")
C1B_CACHE_MANIFEST = V2 / "../c1b_overlap_eligibility_ftv_stageb/manifests/stage_b_c1b_cache.private.csv"
V4_C0 = V4 / "checkpoints/pilot"
V5_STATES = V5 / "features/private/pilot_states"
SUMMARY_SHAPE = (4,7,32,2304); SPATIAL_SHAPE = (4,7,32,7,7,768)
FOLDS = tuple(range(5)); PILOT_SEED = 2026; FORMAL_SEEDS = (7026,8026,9026,10026,11026)

def sha256_file(path: str | Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()

def canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()

def token(patient_id: str) -> str: return hashlib.sha256(str(patient_id).encode()).hexdigest()

def atomic_json(path: str | Path, payload: Any) -> None:
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); q=p.with_suffix(p.suffix+".tmp")
    q.write_text(json.dumps(payload,indent=2,sort_keys=True,default=str)+"\n"); q.replace(p)

def protocol() -> dict[str,Any]:
    p=json.loads((ROOT/"configs/protocol.json").read_text())
    if p.get("experiment")!="dinov3_spatial_parallel_radiomics_v6": raise ValueError("V6 protocol identity drift")
    return p

def folds() -> pd.DataFrame:
    f=pd.read_csv(C1B_MANIFEST,usecols=["patient_id","fold","split"],dtype={"patient_id":str})
    f["fold"]=f["fold"].astype(int); f["split"]=f["split"].replace({"validation":"val"})
    if len(f)!=4040 or f.duplicated(["patient_id","fold"]).any(): raise ValueError("fold contract failed")
    return f

def c0_path(fold: int) -> Path: return V4_C0 / f"seed2026_fold{fold}_F0/selected.private.pt"
def state_path(fold: int, arm: str, pilot: bool=True) -> Path:
    prefix="seed2026" if pilot else "seedFORMAL"
    return ROOT / "features/private/states" / f"{prefix}_fold{fold}_{arm}.private.npz"

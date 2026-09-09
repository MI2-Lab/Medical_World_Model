from __future__ import annotations
from pathlib import Path
from typing import Iterable
import numpy as np, torch
from torch.utils.data import Dataset
from .contracts import SUMMARY_DIR, SPATIAL_DIR, TARGET_DIR, SUMMARY_SHAPE, SPATIAL_SHAPE, token

def _load(path: Path, key: str, shape: tuple[int,...], pid: str) -> np.ndarray:
    with np.load(path,allow_pickle=False) as z:
        if set(z.files) != {"patient_token",key,"source_cache_sha256","contract_sha256"}: raise ValueError("cache contract failed")
        if str(z["patient_token"].item()) != token(pid): raise ValueError("cache token failed")
        x=np.asarray(z[key])
    if x.shape!=shape or x.dtype!=np.float16 or not np.isfinite(x).all(): raise ValueError("cache shape/finite failed")
    return x

def load_summary(pid: str) -> np.ndarray: return _load(SUMMARY_DIR/f"{token(pid)}.private.npz","summary",SUMMARY_SHAPE,pid)
def load_spatial(pid: str) -> np.ndarray: return _load(SPATIAL_DIR/f"{token(pid)}.private.npz","spatial_tokens",SPATIAL_SHAPE,pid)

class Targets:
    def __init__(self,path: str|Path):
        with np.load(path,allow_pickle=False) as z:
            self.patient_id=np.asarray(z["patient_id"]).astype(str); self.target=np.asarray(z["target"],np.float32); self.mask=np.asarray(z["target_mask"],bool); self.ftv=np.asarray(z["ftv"],np.float32); self.ftv_mask=np.asarray(z["ftv_mask"],bool)
        n=len(self.patient_id)
        if self.target.shape!=(n,4,16) or self.mask.shape!=(n,4) or self.mask[:,3].any() or not np.isfinite(self.target[self.mask]).all(): raise ValueError("target contract failed")
        self.by={p:(self.target[i],self.mask[i],self.ftv[i],self.ftv_mask[i]) for i,p in enumerate(self.patient_id)}

class Dataset(Dataset):
    def __init__(self, ids: Iterable[str], target_path: str|Path, spatial: bool):
        self.ids=tuple(sorted(map(str,ids))); self.targets=Targets(target_path).by; self.spatial=spatial
        # NPZ decompression is the dominant cost for the spatial arm.  A cell
        # only contains one train/validation split, so retain its immutable
        # float16 cache in RAM and make every epoch a pure tensor read.  This
        # does not alter the data contract or model inputs.
        self.summary_cache={pid:load_summary(pid) for pid in self.ids}
        self.spatial_cache={pid:load_spatial(pid) for pid in self.ids} if spatial else {}
    def __len__(self): return len(self.ids)
    def __getitem__(self,i):
        pid=self.ids[i]; y,m,ftv,fm=self.targets.get(pid,(np.zeros((4,16),np.float32),np.zeros(4,bool),np.zeros(4,np.float32),np.zeros(4,bool)))
        out={"patient_id":pid,"summary":torch.from_numpy(self.summary_cache[pid]),"target":torch.from_numpy(y),"target_mask":torch.from_numpy(m),"ftv":torch.from_numpy(ftv),"ftv_mask":torch.from_numpy(fm)}
        if self.spatial: out["spatial_tokens"]=torch.from_numpy(self.spatial_cache[pid])
        return out

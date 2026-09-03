"""Torch-free reader for the frozen C1B cache manifest."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .contracts import CACHE_MANIFEST, COHORT_PATIENTS, LOCKED_HASHES, verify_locked_file


@dataclass(frozen=True)
class CacheEntry:
    patient_id: str
    path: Path
    sha256: str


def load_c1b_manifest(path: str | Path = CACHE_MANIFEST, *, verify_hash: bool = True) -> dict[str, CacheEntry]:
    source = Path(path)
    if verify_hash:
        verify_locked_file(source, LOCKED_HASHES["cache_manifest"], "C1B cache manifest")
    frame = pd.read_csv(
        source,
        usecols=["patient_id", "cache_path", "cache_sha256", "input_kind"],
        dtype={"patient_id": str, "cache_path": str, "cache_sha256": str, "input_kind": str},
    )
    if len(frame) != COHORT_PATIENTS or frame["patient_id"].duplicated().any():
        raise ValueError("C1B manifest must contain 947 unique patients")
    if not frame["input_kind"].eq("c1b").all():
        raise ValueError("C1B manifest contains a non-C1B entry")
    entries: dict[str, CacheEntry] = {}
    for row in frame.itertuples(index=False):
        cache = Path(row.cache_path)
        if not cache.is_file():
            raise FileNotFoundError(cache)
        entries[row.patient_id] = CacheEntry(row.patient_id, cache, row.cache_sha256)
    return entries


__all__ = ["CacheEntry", "load_c1b_manifest"]

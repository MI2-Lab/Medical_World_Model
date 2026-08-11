#!/usr/bin/env python3
"""Verify that reused Stage-A caches no longer share inodes with prior evidence.

The Stage-A builder can hard-link an existing immutable cache long enough to
rebuild and byte-validate it.  This verifier is run after those links have been
replaced by independent files.  Its public output contains aggregate counts
only; cache tokens and patient identifiers are never emitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
PRIOR_ROOT = REPO_ROOT / "additional_experiments/c1b_model_ready_ftv_sanity"
DEFAULT_PRIOR_CACHE = PRIOR_ROOT / "cache/c1b_h"
DEFAULT_NEW_CACHE = EXPERIMENT_ROOT / "cache/c1b_h"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "metrics/cache_independence_verification.json"
LOCKED_PRIOR_FILES = {
    "STAGE_A_NO_GO.json": "ad2604d35c9fca645f6487c7decf297a0c8f0711136973491d537ac42aa8f080",
    "reports/final_report.md": "50d7ce177a431ae536ccddef7faf1f93dde753d9b87d0cacb4a115077b1e4976",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-cache-dir", type=Path, default=DEFAULT_PRIOR_CACHE)
    parser.add_argument("--new-cache-dir", type=Path, default=DEFAULT_NEW_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def byte_equal(pair: tuple[Path, Path]) -> bool:
    left, right = pair
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as lhs, right.open("rb") as rhs:
        while True:
            a = lhs.read(8 * 1024 * 1024)
            b = rhs.read(8 * 1024 * 1024)
            if a != b:
                return False
            if not a:
                return True


def cache_files(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = {path.name: path for path in root.glob("*.npz") if path.is_file()}
    if len(files) != len(list(root.glob("*.npz"))):
        raise ValueError(f"Duplicate cache names under {root}")
    return files


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    prior = cache_files(args.prior_cache_dir)
    current = cache_files(args.new_cache_dir)
    common_names = sorted(set(prior).intersection(current))
    pairs = [(prior[name], current[name]) for name in common_names]

    shared_inode_count = sum(
        int(os.path.samefile(old_path, new_path)) for old_path, new_path in pairs
    )
    prior_multilink_count = sum(int(path.stat().st_nlink != 1) for path in prior.values())
    new_multilink_count = sum(int(path.stat().st_nlink != 1) for path in current.values())
    size_match_count = sum(
        int(old_path.stat().st_size == new_path.stat().st_size)
        for old_path, new_path in pairs
    )
    mtime_match_count = sum(
        int(old_path.stat().st_mtime_ns == new_path.stat().st_mtime_ns)
        for old_path, new_path in pairs
    )
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        byte_results = list(executor.map(byte_equal, pairs))
    byte_equal_count = sum(map(int, byte_results))

    prior_file_hashes = {
        relative: sha256_file(PRIOR_ROOT / relative)
        for relative in LOCKED_PRIOR_FILES
    }
    prior_contract_unchanged = prior_file_hashes == LOCKED_PRIOR_FILES
    passed = bool(
        len(current) == 947
        and len(common_names) == 262
        and shared_inode_count == 0
        and prior_multilink_count == 0
        and new_multilink_count == 0
        and size_match_count == len(common_names)
        and mtime_match_count == len(common_names)
        and byte_equal_count == len(common_names)
        and prior_contract_unchanged
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "contains_patient_identifiers": False,
        "prior_cache_files": len(prior),
        "new_cache_files": len(current),
        "reused_cache_files_compared": len(common_names),
        "reused_cache_byte_equal_count": byte_equal_count,
        "reused_cache_size_match_count": size_match_count,
        "reused_cache_mtime_match_count": mtime_match_count,
        "shared_inode_count_after_delink": shared_inode_count,
        "prior_cache_multilink_count_after_delink": prior_multilink_count,
        "new_cache_multilink_count_after_delink": new_multilink_count,
        "delink_method": "copy_or_reflink_then_full_byte_compare_then_atomic_replace",
        "hardlinks_were_created_then_removed": True,
        "prior_inode_ctime_may_have_changed": True,
        "prior_bytes_and_mtime_unchanged": bool(
            byte_equal_count == len(common_names)
            and mtime_match_count == len(common_names)
        ),
        "locked_prior_file_sha256": prior_file_hashes,
        "locked_prior_contract_unchanged": prior_contract_unchanged,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

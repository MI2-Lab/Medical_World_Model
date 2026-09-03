#!/usr/bin/env python3
"""Hash private V5 artifacts and publish aggregate integrity metadata."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from parallel_rg.contracts import atomic_json, canonical_sha, sha256_file


def main() -> None:
    output = ROOT / "logs/private/private_sha_manifest.private.json"; directories = (ROOT / "checkpoints", ROOT / "features/private", ROOT / "logs/private"); paths = sorted({path for directory in directories if directory.is_dir() for path in directory.rglob("*") if path.is_file() and path != output})
    records = [{"artifact": path.relative_to(ROOT).parts[0], "size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in paths]; categories = Counter(item["artifact"] for item in records); private = {"schema_version": 1, "status": "COMPLETE", "records": records, "record_count": len(records), "total_bytes": sum(item["size_bytes"] for item in records), "records_sha256": canonical_sha(records)}; atomic_json(output, private)
    public = {"schema_version": 1, "status": "COMPLETE", "private_artifact_count": len(records), "private_artifact_bytes": private["total_bytes"], "category_counts": dict(sorted(categories.items())), "private_records_sha256": private["records_sha256"], "private_manifest_sha256": sha256_file(output), "contains_private_paths": False, "contains_patient_identifiers": False}; atomic_json(ROOT / "manifests/private_artifact_sha_summary.json", public); print(public)


if __name__ == "__main__": main()

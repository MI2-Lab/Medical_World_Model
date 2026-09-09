#!/usr/bin/env python3
"""Hash private V4 artifacts without publishing patient identifiers or paths."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from factorized_rg.contracts import atomic_json, canonical_sha, sha256_file


PRIVATE_ROOTS = (ROOT / "checkpoints", ROOT / "features/private", ROOT / "logs/private")


def main() -> None:
    output = ROOT / "logs/private/private_sha_manifest.private.json"
    files: list[Path] = []
    for directory in PRIVATE_ROOTS:
        if directory.is_dir():
            files.extend(path for path in directory.rglob("*") if path.is_file() and path != output)
    files.extend(path for path in (ROOT / "metrics").glob("*.private.*") if path.is_file())
    files = sorted(set(files))
    records = [{"artifact": path.relative_to(ROOT).parts[0], "size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in files]
    categories = Counter(record["artifact"] for record in records)
    private_payload = {"schema_version": 1, "status": "COMPLETE", "records": records, "record_count": len(records), "total_bytes": sum(x["size_bytes"] for x in records), "records_sha256": canonical_sha(records)}
    atomic_json(output, private_payload)
    public_payload = {"schema_version": 1, "status": "COMPLETE", "private_artifact_count": len(records), "private_artifact_bytes": private_payload["total_bytes"], "category_counts": dict(sorted(categories.items())), "private_records_sha256": private_payload["records_sha256"], "private_manifest_sha256": sha256_file(output), "contains_private_paths": False, "contains_patient_identifiers": False}
    atomic_json(ROOT / "manifests/private_artifact_sha_summary.json", public_payload)
    print(public_payload)


if __name__ == "__main__":
    main()

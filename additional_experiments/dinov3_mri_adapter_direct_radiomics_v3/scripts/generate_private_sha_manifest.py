#!/usr/bin/env python3
"""Hash every private experiment artifact and publish only aggregate integrity metadata."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dinov3_rg.contracts import atomic_json, canonical_sha256, file_sha256  # noqa: E402


PRIVATE_ROOTS = (
    ROOT / "cache",
    ROOT / "checkpoints",
    ROOT / "features/private",
    ROOT / "logs/private",
    ROOT / "predictions",
)


def private_files(output: Path) -> tuple[Path, ...]:
    files: set[Path] = set()
    for directory in PRIVATE_ROOTS:
        if directory.is_dir():
            files.update(path for path in directory.rglob("*") if path.is_file())
    for directory in (ROOT / "manifests", ROOT / "metrics"):
        if directory.is_dir():
            files.update(path for path in directory.glob("*.private.*") if path.is_file())
    files.discard(output)
    return tuple(sorted(files))


def main() -> None:
    output = ROOT / "logs/private/private_sha_manifest.private.json"
    records = []
    categories: Counter[str] = Counter()
    total_bytes = 0
    for path in private_files(output):
        relative = str(path.relative_to(ROOT))
        size = path.stat().st_size
        category = relative.split("/", 1)[0]
        records.append({"path": relative, "size_bytes": size, "sha256": file_sha256(path)})
        categories[category] += 1
        total_bytes += size
    private_payload = {
        "schema_version": 1,
        "status": "COMPLETE",
        "records": records,
        "record_count": len(records),
        "total_bytes": total_bytes,
        "records_sha256": canonical_sha256(records),
    }
    atomic_json(output, private_payload)
    public_payload = {
        "schema_version": 1,
        "status": "COMPLETE",
        "private_artifact_count": len(records),
        "private_artifact_bytes": total_bytes,
        "category_counts": dict(sorted(categories.items())),
        "private_records_sha256": private_payload["records_sha256"],
        "private_manifest_sha256": file_sha256(output),
        "contains_private_paths": False,
        "contains_patient_identifiers": False,
    }
    atomic_json(ROOT / "manifests/private_artifact_sha_summary.json", public_payload)
    print(public_payload)


if __name__ == "__main__":
    main()

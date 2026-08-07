#!/usr/bin/env python3
"""为报告派生的补充指标表生成 SHA256 manifest。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import pandas as pd


AUDIT_ROOT = Path(__file__).resolve().parents[1]
SUPPLEMENTAL_FILES = (
    "probability_change_summary.csv",
    "donor_matching_summary.csv",
    "donor_matching_summary.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, default=AUDIT_ROOT)
    parser.add_argument("--allow-manifest", action="store_true")
    args = parser.parse_args()
    if not args.allow_manifest:
        raise SystemExit("补充 manifest 未生成：必须显式添加 --allow-manifest")

    metrics = args.audit_root.resolve() / "metrics"
    output = metrics / "supplemental_metrics_manifest.json"
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有补充 manifest：{output}")
    missing = [name for name in SUPPLEMENTAL_FILES if not (metrics / name).is_file()]
    if missing:
        raise FileNotFoundError(f"补充指标文件不完整：{missing}")

    artifacts: list[dict[str, object]] = []
    for name in SUPPLEMENTAL_FILES:
        path = metrics / name
        artifact: dict[str, object] = {
            "name": name,
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        if path.suffix == ".csv":
            artifact["rows"] = len(pd.read_csv(path))
        artifacts.append(artifact)

    payload = {
        "schema_version": "shortcut_audit.supplemental_metrics.v1",
        "derivation": (
            "report-only summaries derived from manifest-protected reporting tables "
            "and outcome-blind donor matching artifacts"
        ),
        "artifacts": artifacts,
    }
    descriptor, name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=metrics
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(output)


if __name__ == "__main__":
    main()

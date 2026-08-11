from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from foundation_mri_dinov3.locking import (
    canonical_json_bytes,
    load_json,
    verify_hash_records,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_canonical_json_is_deterministic_and_rejects_nan() -> None:
    assert canonical_json_bytes({"b": 2, "a": 1}) == b'{"a":1,"b":2}\n'
    with pytest.raises(ValueError):
        canonical_json_bytes({"bad": float("nan")})


def test_json_duplicate_keys_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "value.json"
    source.write_text('{"a":1,"a":2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_json(source)


def test_hash_records_verify_bytes_and_reject_drift(tmp_path: Path) -> None:
    source = tmp_path / "asset.bin"
    source.write_bytes(b"locked")
    record = {
        "asset": {
            "path": source.as_posix(),
            "sha256": _sha(source),
            "bytes": 6,
        }
    }
    assert verify_hash_records(record, repository_root=tmp_path) == {
        "asset": _sha(source)
    }
    source.write_bytes(b"drifted")
    with pytest.raises(ValueError, match="byte-size drift|SHA-256 drift"):
        verify_hash_records(record, repository_root=tmp_path)


def test_hash_records_reject_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"x")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(FileNotFoundError):
        verify_hash_records(
            {"x": {"path": link.as_posix(), "sha256": _sha(target)}},
            repository_root=tmp_path,
        )


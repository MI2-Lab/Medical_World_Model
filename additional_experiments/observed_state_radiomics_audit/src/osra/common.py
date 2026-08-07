"""Observed-state audit 的共享路径、哈希与原子输出工具。"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


AUDIT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = AUDIT_ROOT.parents[1]


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"配置必须为 mapping: {path}")
    return payload


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(path.resolve() for path in paths):
        try:
            label = path.relative_to(REPO_ROOT)
        except ValueError:
            label = path
        digest.update(str(label).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _temporary_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(handle)
    return Path(name)


def atomic_json(path: Path, payload: Any) -> None:
    temporary = _temporary_path(path)
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = _temporary_path(path)
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = _temporary_path(path)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def refuse_existing(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("输出已存在，默认拒绝覆盖：" + ", ".join(str(path) for path in existing))


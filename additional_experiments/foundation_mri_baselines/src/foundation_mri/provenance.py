"""Small provenance and secure-artifact helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def ordered_text_sha256(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def secure_directory(path: str | Path) -> Path:
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination.chmod(0o700)
    return destination


def _temporary_file(path: Path) -> tuple[int, Path]:
    secure_directory(path.parent)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.fchmod(descriptor, 0o600)
    return descriptor, Path(name)


def atomic_private_npz(path: str | Path, arrays: Mapping[str, Any]) -> None:
    destination = Path(path)
    descriptor, temporary = _temporary_file(destination)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez(stream, **dict(arrays))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
        destination.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: str | Path, payload: Any, *, private: bool) -> None:
    destination = Path(path)
    descriptor, temporary = _temporary_file(destination)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
        destination.chmod(0o600 if private else 0o644)
    finally:
        temporary.unlink(missing_ok=True)


def environment_snapshot() -> dict[str, object]:
    import numpy
    import pandas
    import sklearn
    import timm
    import torch

    try:
        import transformers

        transformers_version: str | None = transformers.__version__
    except ModuleNotFoundError:
        transformers_version = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
        "timm": timm.__version__,
        "transformers": transformers_version,
    }


def verify_file_lock(lock_path: str | Path, root: str | Path) -> dict[str, Any]:
    """Fail closed if any file named by a pre-test lock has drifted."""

    source = Path(lock_path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or int(payload.get("schema_version", -1)) != 1:
        raise ValueError("protocol lock must be a schema-v1 JSON object")
    files = payload.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("protocol lock contains no file inventory")
    base = Path(root).resolve()
    for relative, expected in files.items():
        path = (base / str(relative)).resolve()
        if not path.is_relative_to(base) or not path.is_file():
            raise FileNotFoundError(f"locked file is missing/outside root: {relative}")
        observed = file_sha256(path)
        if observed != str(expected):
            raise ValueError(
                f"locked file drifted: {relative}; expected {expected}, "
                f"observed {observed}"
            )
    return payload


__all__ = [
    "atomic_json",
    "atomic_private_npz",
    "canonical_json_sha256",
    "environment_snapshot",
    "file_sha256",
    "ordered_text_sha256",
    "secure_directory",
    "verify_file_lock",
]

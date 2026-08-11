"""Fail-closed filesystem boundaries for formal pilot writers."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import re
from types import ModuleType
from typing import Any


SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_contained_path(
    path: str | Path,
    root: str | Path,
    *,
    label: str,
    allow_root: bool = False,
) -> Path:
    """Resolve ``path`` and require it to remain inside the declared pilot root."""

    boundary = Path(root).expanduser().resolve()
    candidate = Path(path).expanduser().resolve()
    if not boundary.is_dir():
        raise FileNotFoundError(f"{label} boundary does not exist: {boundary}")
    try:
        relative = candidate.relative_to(boundary)
    except ValueError as error:
        raise ValueError(f"{label} must remain under {boundary}") from error
    if not allow_root and relative == Path("."):
        raise ValueError(f"{label} must be a strict descendant of {boundary}")
    return candidate


def claim_private_directory(
    path: str | Path,
    root: str | Path,
    *,
    label: str,
) -> Path:
    """Atomically claim a new private leaf directory inside ``root``."""

    candidate = resolve_contained_path(path, root, label=label)
    try:
        candidate.mkdir(parents=True, mode=0o700, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(
            f"{label} already exists; formal runs are never resumed: {candidate}"
        ) from error
    candidate.chmod(0o700)
    return candidate


def require_lock_sha256(observed: object, expected: object) -> str:
    """Require one exact lowercase SHA-256 lock identity."""

    observed_value = str(observed)
    expected_value = str(expected)
    if (
        len(expected_value) != 64
        or any(character not in "0123456789abcdef" for character in expected_value)
    ):
        raise ValueError("expected preregistration lock SHA-256 is invalid")
    if observed_value != expected_value:
        raise RuntimeError("preregistration lock changed after parent preflight")
    return expected_value


def require_canonical_file(
    path: str | Path,
    canonical_path: str | Path,
    declared_sha256: object,
    locked_sha256: object,
    *,
    label: str,
) -> Path:
    """Reject path/hash substitutions, including self-consistent alternatives."""

    candidate = Path(path).expanduser().resolve()
    canonical = Path(canonical_path).expanduser().resolve()
    if candidate != canonical:
        raise ValueError(f"{label} path must be the canonical preregistered file")
    declared = str(declared_sha256)
    locked = str(locked_sha256)
    if SHA256_PATTERN.fullmatch(declared) is None:
        raise ValueError(f"{label} declared SHA-256 is invalid")
    if SHA256_PATTERN.fullmatch(locked) is None:
        raise ValueError(f"{label} lock SHA-256 is invalid")
    if declared != locked:
        raise ValueError(f"{label} SHA-256 differs from the preregistration lock")
    if not canonical.is_file() or file_sha256(canonical) != locked:
        raise RuntimeError(f"{label} canonical file drifted from the lock")
    return canonical


def require_module_within(module: ModuleType, root: str | Path, *, label: str) -> None:
    """Reject shadowed or preloaded modules outside one canonical source root."""

    location = getattr(module, "__file__", None)
    if not location:
        raise ImportError(f"{label} has no inspectable source file")
    boundary = Path(root).resolve()
    try:
        Path(location).resolve().relative_to(boundary)
    except ValueError as error:
        raise ImportError(f"{label} resolved outside canonical source root") from error


def require_object_within(value: Any, root: str | Path, *, label: str) -> None:
    """Reject callables/classes supplied by a shadow or preloaded module."""

    boundary = Path(root).resolve()
    try:
        Path(inspect.getfile(inspect.unwrap(value))).resolve().relative_to(boundary)
    except (TypeError, ValueError) as error:
        raise ImportError(f"{label} resolved outside canonical source root") from error


__all__ = [
    "claim_private_directory",
    "file_sha256",
    "require_canonical_file",
    "require_lock_sha256",
    "require_module_within",
    "require_object_within",
    "resolve_contained_path",
]

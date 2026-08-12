"""Residual-SPH grounding pilot.

The package keeps its top-level import deliberately light.  Training modules
perform hash-locked upstream imports only after callers verify the immutable
preregistration lock.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "ARMS": ("contracts", "ARMS"),
    "FOLDS": ("contracts", "FOLDS"),
    "SEED_BASES": ("contracts", "SEED_BASES"),
    "ArmSpec": ("contracts", "ArmSpec"),
    "arm_spec": ("contracts", "arm_spec"),
    "ResidualSPHWorldModel": ("model", "ResidualSPHWorldModel"),
    "build_model": ("model", "build_model"),
    "build_objective": ("losses", "build_objective"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(f".{module_name}", __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


__all__ = sorted(_EXPORTS)

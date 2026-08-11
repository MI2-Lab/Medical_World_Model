"""Leak-resistant frozen-encoder baselines for longitudinal DCE MRI.

Model exports are lazy so the statistics-only evaluation layer can be audited
in a CPU environment that does not install PyTorch.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "DINO_EMBED_DIM",
    "MEDICALNET_EMBED_DIM",
    "DINOEncoder",
    "MedicalNetEncoder",
    "load_dino_encoder",
    "load_medicalnet_encoder",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    from . import models

    return getattr(models, name)

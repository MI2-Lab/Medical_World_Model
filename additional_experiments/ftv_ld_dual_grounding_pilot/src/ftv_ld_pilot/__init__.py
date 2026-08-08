"""Geometry utilities for the FTV + LD dual-grounding pilot."""

from .geometry import (
    approx_max_extent_mm,
    bbox_xyz,
    crop_or_pad_from_start,
    geometry_metrics,
    recover_origin,
)

__all__ = [
    "approx_max_extent_mm",
    "bbox_xyz",
    "crop_or_pad_from_start",
    "geometry_metrics",
    "recover_origin",
]

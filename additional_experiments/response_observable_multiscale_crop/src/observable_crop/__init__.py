"""Physical-space crop planning and observability audit primitives."""

from .geometry import (
    PhysicalWindow,
    SupportAudit,
    audit_support,
    bbox_footprint_in_frame,
    make_fixed_expand_window,
    make_union_window,
    orthonormal_index_basis,
)
from .nifti import NiftiGeometry, read_nifti_geometry

__all__ = [
    "NiftiGeometry",
    "PhysicalWindow",
    "SupportAudit",
    "audit_support",
    "bbox_footprint_in_frame",
    "make_fixed_expand_window",
    "make_union_window",
    "orthonormal_index_basis",
    "read_nifti_geometry",
]

"""Model-ready C1B image-pipeline primitives."""

from .dicom_pixel_rebuild import (
    DicomPixelRebuildError,
    DicomPixelRebuildResult,
    NiftiWriteAudit,
    PixelRebuildMetrics,
    RebuildTolerances,
    rebuild_classic_dce_series,
)
from .geometry import (
    C1B_SHAPE_ZYX,
    C1B_SPACING_XYZ_MM,
    CanonicalVolume,
    PhysicalGrid,
    PhysicalSupportAudit,
    audit_support_containment,
    load_nifti_ras,
    make_c1b_grid,
    support_bbox_center_ras,
)
from .dce7 import (
    DCE7_CHANNEL_NAMES,
    PhaseSelection,
    VisitDCE7,
    build_visit_dce7,
    construct_dce7_xyzt,
    select_phase_indices,
)
from .builder import PatientDCE7, SupportQC, VisitInput, build_patient_dce7
from .cache import load_and_validate_cache, load_model_tensor, write_cache_atomic

__all__ = [
    "DicomPixelRebuildError",
    "DicomPixelRebuildResult",
    "NiftiWriteAudit",
    "PixelRebuildMetrics",
    "RebuildTolerances",
    "rebuild_classic_dce_series",
    "C1B_SHAPE_ZYX",
    "C1B_SPACING_XYZ_MM",
    "CanonicalVolume",
    "PhysicalGrid",
    "PhysicalSupportAudit",
    "audit_support_containment",
    "load_nifti_ras",
    "make_c1b_grid",
    "support_bbox_center_ras",
    "DCE7_CHANNEL_NAMES",
    "PhaseSelection",
    "VisitDCE7",
    "build_visit_dce7",
    "construct_dce7_xyzt",
    "select_phase_indices",
    "PatientDCE7",
    "SupportQC",
    "VisitInput",
    "build_patient_dce7",
    "load_and_validate_cache",
    "load_model_tensor",
    "write_cache_atomic",
]

"""Four-visit overlap eligibility and matched Stage-B utilities."""

from .eligibility import (
    VISITS,
    build_patient_eligibility,
    canonical_header_geometry,
    count_valid_source_voxels,
    frozen_grid_contract_sha256,
)
from .io import (
    atomic_text,
    sha256_file,
    verify_preregistration,
    verify_upstream_contract,
)

__all__ = [
    "VISITS",
    "atomic_text",
    "build_patient_eligibility",
    "canonical_header_geometry",
    "count_valid_source_voxels",
    "frozen_grid_contract_sha256",
    "sha256_file",
    "verify_preregistration",
    "verify_upstream_contract",
]

"""Patient-level orchestration for the frozen, T0-anchored C1B tensor."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .dce7 import (
    ANTI_ALIAS_FACTOR_THRESHOLD,
    DCE7_CHANNEL_NAMES,
    PADDING_MODE,
    VisitDCE7,
    build_visit_dce7,
    phase_metadata_sha256,
)
from .geometry import (
    CanonicalVolume,
    PhysicalGrid,
    acquisition_center_ras,
    audit_support_containment,
    canonical_volume_sha256,
    load_nifti_ras,
    make_c1b_grid,
    resample_support_nearest,
    support_bbox_center_ras,
    validate_source_to_anchor_transform,
)


VISITS: tuple[str, ...] = ("T0", "T1", "T2", "T3")
BUILDER_CONTRACT_VERSION = "c1b-model-ready-v3"


def builder_contract_payload() -> dict[str, Any]:
    """Canonical semantic contract whose digest invalidates stale caches."""

    return {
        "version": BUILDER_CONTRACT_VERSION,
        "canonical_orientation": "true_array_reordered_RAS+_float32",
        "grid": {
            "shape_zyx": [112, 176, 160],
            "spacing_xyz_mm": [0.9, 0.9, 2.0],
            "anchor": "T0_support_bbox_center_else_T0_acquisition_center",
            "later_visit_recentering": False,
        },
        "phase": {
            "allowlist": ["pre", "post_early", "post_late"],
            "short": "T<=4:(pre,pre+1_clipped,T-1)",
            "long": "T>4:(pre,post_early_or_2,post_late_or_5)_clipped",
            "peak_window": "indices_1_through_last_else_pre",
        },
        "dce7_channels": list(DCE7_CHANNEL_NAMES),
        "dce7_formulas": [
            "pre",
            "early",
            "late",
            "early-pre",
            "late-pre",
            "(peak-pre)/max(abs(pre),1)",
            "(late-peak)/max(abs(pre),1)",
        ],
        "resampling": {
            "all_raw_phases_one_4d_spatial_pass": True,
            "temporal_interpolation": False,
            "intensity": "linear",
            "padding": PADDING_MODE,
            "valid_source": "target_voxel_center_inside_source_voxel_footprint",
            "anti_alias_factor_threshold": ANTI_ALIAS_FACTOR_THRESHOLD,
            "anti_alias_sigma": "0.5*sqrt(max(factor^2-1,0))_per_source_axis",
        },
        "normalization": {
            "scope": "per_visit_channel_valid_source_only",
            "percentile_clip": [1.0, 99.0],
            "center": "median",
            "scale": "IQR/1.349_then_std_then_unit",
            "output_clip": [-5.0, 5.0],
        },
        "support": {
            "T0": "anchor_and_qc_or_none_fallback",
            "T1_T3": "formal_ftv_overlap_qc_only_or_not_loaded",
            "interpolation": "nearest",
            "model_input": False,
        },
        "registration": {
            "coordinate_convention": "source_RAS_world_to_T0_anchor_RAS_world",
            "transform": "proper_rigid_only",
            "T0": "identity",
            "C1B-H": "all_identity",
        },
        "model_tensor": "image_only_[T0-T3,DCE7,Z,Y,X]",
    }


def builder_contract_sha256() -> str:
    encoded = json.dumps(
        builder_contract_payload(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(b"c1b-builder-contract-v1\0" + encoded).hexdigest()


def input_provenance_sha256(
    *,
    patient_id: str,
    cohort: str,
    formal_ftv_overlap: bool,
    registration_strategy: str,
    source_hashes: np.ndarray,
    support_hashes: np.ndarray,
    support_scope: np.ndarray,
    phase_metadata_hashes: np.ndarray,
    phase_counts: np.ndarray,
    phase_indices: np.ndarray,
    source_to_anchor_ras: np.ndarray,
    grid: PhysicalGrid,
    anchor_provenance: str,
    contract_sha256: str,
) -> str:
    """Hash every current input and semantic choice that can affect a cache."""

    transforms = np.ascontiguousarray(source_to_anchor_ras, dtype="<f8")
    payload = {
        "patient_id": str(patient_id),
        "cohort": str(cohort),
        "formal_ftv_overlap": bool(formal_ftv_overlap),
        "registration_strategy": str(registration_strategy),
        "source_canonical_sha256": [str(value) for value in source_hashes],
        "support_canonical_sha256": [str(value) for value in support_hashes],
        "support_scope": [str(value) for value in support_scope],
        "phase_metadata_sha256": [str(value) for value in phase_metadata_hashes],
        "phase_counts": np.asarray(phase_counts, dtype=np.int64).tolist(),
        "phase_indices": np.asarray(phase_indices, dtype=np.int64).tolist(),
        "source_to_anchor_ras_f8le_hex": transforms.tobytes().hex(),
        "grid_affine_ras": np.asarray(grid.affine_ras, dtype=np.float64).tolist(),
        "grid_shape_zyx": list(grid.shape_zyx),
        "grid_spacing_xyz_mm": list(grid.spacing_xyz_mm),
        "anchor_provenance": str(anchor_provenance),
        "builder_contract_sha256": str(contract_sha256),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(b"c1b-input-provenance-v1\0" + encoded).hexdigest()


@dataclass(frozen=True)
class VisitInput:
    """Inputs for one visit; registration is an operator hook, never fitted here."""

    visit: str
    dce: str | Path | CanonicalVolume
    phase_metadata: Mapping[str, Any] | None = None
    localization_support: str | Path | CanonicalVolume | None = None
    source_to_anchor_ras: np.ndarray | None = None


@dataclass(frozen=True)
class SupportQC:
    available: bool
    source_positive_voxels: int
    retained_positive_voxels: int
    nn_target_positive_voxels: int
    source_volume_mm3: float
    retained_source_volume_mm3: float
    retained_positive_voxel_fraction: float
    physical_volume_retention: float
    exact_full_support_containment: bool
    source_boundary_touch: bool
    target_boundary_touch: bool
    minimum_margin_mm: float

    @classmethod
    def unavailable(cls) -> "SupportQC":
        return cls(
            available=False,
            source_positive_voxels=0,
            retained_positive_voxels=0,
            nn_target_positive_voxels=0,
            source_volume_mm3=0.0,
            retained_source_volume_mm3=0.0,
            retained_positive_voxel_fraction=float("nan"),
            physical_volume_retention=float("nan"),
            exact_full_support_containment=False,
            source_boundary_touch=False,
            target_boundary_touch=False,
            minimum_margin_mm=float("nan"),
        )


@dataclass(frozen=True)
class PatientDCE7:
    """Four-visit model tensor and non-model audit sidecars."""

    patient_id: str
    cohort: str
    formal_ftv_overlap: bool
    registration_strategy: str
    image: np.ndarray
    valid_source_mask: np.ndarray
    phase_indices: np.ndarray
    phase_counts: np.ndarray
    normalization_p01: np.ndarray
    normalization_p99: np.ndarray
    normalization_median: np.ndarray
    normalization_scale: np.ndarray
    normalization_scale_source: np.ndarray
    source_samples_per_output_axis: np.ndarray
    anti_alias_sigma_source_voxels: np.ndarray
    anti_alias_applied: np.ndarray
    source_canonical_sha256: np.ndarray
    support_canonical_sha256: np.ndarray
    support_scope: np.ndarray
    phase_metadata_sha256: np.ndarray
    builder_contract_sha256: str
    input_provenance_sha256: str
    source_to_anchor_ras: np.ndarray
    support_qc: tuple[SupportQC, ...]
    grid: PhysicalGrid
    anchor_provenance: str

    def __post_init__(self) -> None:
        expected = (len(VISITS), len(DCE7_CHANNEL_NAMES), *self.grid.shape_zyx)
        if self.image.shape != expected or self.image.dtype != np.float32:
            raise ValueError(
                f"patient image must be float32 {expected}, got {self.image.shape}"
            )
        if self.valid_source_mask.shape != (len(VISITS), 1, *self.grid.shape_zyx):
            raise ValueError("valid-source sidecar shape does not match patient image")
        if len(self.support_qc) != len(VISITS):
            raise ValueError("support QC must contain exactly T0-T3")
        source_hashes = np.asarray(self.source_canonical_sha256)
        if source_hashes.shape != (len(VISITS),) or any(
            len(str(value)) != 64
            or any(character not in "0123456789abcdef" for character in str(value))
            for value in source_hashes
        ):
            raise ValueError("source canonical hashes must be four lowercase SHA-256 values")
        if self.cohort not in {"I-SPY1", "I-SPY2"}:
            raise ValueError("cohort must be I-SPY1 or I-SPY2")
        if self.registration_strategy not in {"C1B-H", "C1B-R"}:
            raise ValueError("registration strategy must be C1B-H or C1B-R")
        support_hashes = np.asarray(self.support_canonical_sha256)
        if support_hashes.shape != (len(VISITS),) or any(
            str(value) != "NONE"
            and (
                len(str(value)) != 64
                or any(character not in "0123456789abcdef" for character in str(value))
            )
            for value in support_hashes
        ):
            raise ValueError("support hashes must be four SHA-256 values or NONE")
        phase_hashes = np.asarray(self.phase_metadata_sha256)
        if phase_hashes.shape != (len(VISITS),) or any(
            len(str(value)) != 64
            or any(character not in "0123456789abcdef" for character in str(value))
            for value in phase_hashes
        ):
            raise ValueError("phase metadata hashes must be four lowercase SHA-256 values")
        for name, digest in (
            ("builder contract", self.builder_contract_sha256),
            ("input provenance", self.input_provenance_sha256),
        ):
            if len(str(digest)) != 64 or any(
                character not in "0123456789abcdef" for character in str(digest)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 value")
        if self.builder_contract_sha256 != builder_contract_sha256():
            raise ValueError("patient was built under a stale builder contract")
        scopes = tuple(str(value) for value in np.asarray(self.support_scope))
        if len(scopes) != len(VISITS):
            raise ValueError("support scope must contain exactly T0-T3")
        if scopes[0] not in {"anchor_and_qc", "none_fallback"} or any(
            scope not in {"formal_qc_only", "not_loaded"} for scope in scopes[1:]
        ):
            raise ValueError("support scope violates the T0/later-visit contract")
        available = tuple(bool(item.available) for item in self.support_qc)
        expected_available = tuple(
            scope in {"anchor_and_qc", "formal_qc_only"} for scope in scopes
        )
        if available != expected_available:
            raise ValueError("support hashes/scopes disagree with support availability")
        if any(
            (digest == "NONE") == is_available
            for digest, is_available in zip(support_hashes, available, strict=True)
        ):
            raise ValueError("support hashes disagree with support availability")
        if self.formal_ftv_overlap:
            if self.cohort != "I-SPY2" or not all(available):
                raise ValueError("formal FTV-overlap caches require all four supports")
        elif any(available[1:]):
            raise ValueError("base-only caches cannot load T1-T3 supports")
        expected_provenance = input_provenance_sha256(
            patient_id=self.patient_id,
            cohort=self.cohort,
            formal_ftv_overlap=self.formal_ftv_overlap,
            registration_strategy=self.registration_strategy,
            source_hashes=self.source_canonical_sha256,
            support_hashes=self.support_canonical_sha256,
            support_scope=self.support_scope,
            phase_metadata_hashes=self.phase_metadata_sha256,
            phase_counts=self.phase_counts,
            phase_indices=self.phase_indices,
            source_to_anchor_ras=self.source_to_anchor_ras,
            grid=self.grid,
            anchor_provenance=self.anchor_provenance,
            contract_sha256=self.builder_contract_sha256,
        )
        if self.input_provenance_sha256 != expected_provenance:
            raise ValueError("patient input provenance digest is internally inconsistent")

    def cache_payload(self) -> dict[str, np.ndarray]:
        """Return an NPZ-ready payload with exactly one model-input array."""

        support_available = np.asarray(
            [item.available for item in self.support_qc], dtype=np.uint8
        )
        support_source_voxels = np.asarray(
            [item.source_positive_voxels for item in self.support_qc], dtype=np.int64
        )
        support_retained_voxels = np.asarray(
            [item.retained_positive_voxels for item in self.support_qc], dtype=np.int64
        )
        support_nn_target_voxels = np.asarray(
            [item.nn_target_positive_voxels for item in self.support_qc], dtype=np.int64
        )
        support_retained_fraction = np.asarray(
            [item.retained_positive_voxel_fraction for item in self.support_qc],
            dtype=np.float32,
        )
        support_retention = np.asarray(
            [item.physical_volume_retention for item in self.support_qc],
            dtype=np.float32,
        )
        support_exact = np.asarray(
            [item.exact_full_support_containment for item in self.support_qc],
            dtype=np.uint8,
        )
        support_source_boundary = np.asarray(
            [item.source_boundary_touch for item in self.support_qc], dtype=np.uint8
        )
        support_target_boundary = np.asarray(
            [item.target_boundary_touch for item in self.support_qc], dtype=np.uint8
        )
        support_margin = np.asarray(
            [item.minimum_margin_mm for item in self.support_qc], dtype=np.float32
        )
        support_source_volume = np.asarray(
            [item.source_volume_mm3 for item in self.support_qc], dtype=np.float32
        )
        support_retained_volume = np.asarray(
            [item.retained_source_volume_mm3 for item in self.support_qc],
            dtype=np.float32,
        )
        return {
            "schema_version": np.asarray(3, dtype=np.int16),
            "patient_id": np.asarray(self.patient_id),
            "cohort": np.asarray(self.cohort),
            "formal_ftv_overlap": np.asarray(self.formal_ftv_overlap, dtype=np.uint8),
            "registration_strategy": np.asarray(self.registration_strategy),
            # This is the only array consumed by the model loader.
            "image": np.ascontiguousarray(self.image, dtype=np.float32),
            "valid_source_mask": np.ascontiguousarray(
                self.valid_source_mask, dtype=np.uint8
            ),
            "phase_indices": np.ascontiguousarray(self.phase_indices, dtype=np.int16),
            "phase_counts": np.ascontiguousarray(self.phase_counts, dtype=np.int16),
            "channel_names": np.asarray(DCE7_CHANNEL_NAMES),
            "visits": np.asarray(VISITS),
            "grid_affine_ras": np.asarray(self.grid.affine_ras, dtype=np.float64),
            "grid_center_ras_mm": np.asarray(self.grid.center_ras_mm, dtype=np.float64),
            "grid_shape_zyx": np.asarray(self.grid.shape_zyx, dtype=np.int16),
            "grid_spacing_xyz_mm": np.asarray(
                self.grid.spacing_xyz_mm, dtype=np.float64
            ),
            "anchor_provenance": np.asarray(self.anchor_provenance),
            "normalization_p01": np.asarray(self.normalization_p01, dtype=np.float32),
            "normalization_p99": np.asarray(self.normalization_p99, dtype=np.float32),
            "normalization_median": np.asarray(
                self.normalization_median, dtype=np.float32
            ),
            "normalization_scale": np.asarray(
                self.normalization_scale, dtype=np.float32
            ),
            "normalization_scale_source": np.asarray(self.normalization_scale_source),
            "source_samples_per_output_axis": np.asarray(
                self.source_samples_per_output_axis, dtype=np.float64
            ),
            "anti_alias_sigma_source_voxels": np.asarray(
                self.anti_alias_sigma_source_voxels, dtype=np.float64
            ),
            "anti_alias_applied": np.asarray(self.anti_alias_applied, dtype=np.uint8),
            # Private provenance closure; never returned by the model loader.
            "source_canonical_sha256": np.asarray(self.source_canonical_sha256),
            "support_canonical_sha256": np.asarray(self.support_canonical_sha256),
            "support_scope": np.asarray(self.support_scope),
            "phase_metadata_sha256": np.asarray(self.phase_metadata_sha256),
            "builder_contract_version": np.asarray(BUILDER_CONTRACT_VERSION),
            "builder_contract_sha256": np.asarray(self.builder_contract_sha256),
            "input_provenance_sha256": np.asarray(self.input_provenance_sha256),
            # This is a private operator/QC sidecar and is never a model channel.
            "source_to_anchor_ras": np.asarray(
                self.source_to_anchor_ras, dtype=np.float64
            ),
            "support_available": support_available,
            "support_source_positive_voxels": support_source_voxels,
            "support_retained_positive_voxels": support_retained_voxels,
            # NN count is diagnostic only; it never defines containment.
            "support_nn_target_positive_voxels": support_nn_target_voxels,
            "support_retained_positive_voxel_fraction": support_retained_fraction,
            "support_physical_volume_retention": support_retention,
            "support_exact_full_support_containment": support_exact,
            "support_source_boundary_touch": support_source_boundary,
            "support_target_boundary_touch": support_target_boundary,
            "support_minimum_margin_mm": support_margin,
            "support_source_volume_mm3": support_source_volume,
            "support_retained_source_volume_mm3": support_retained_volume,
            "padding_mode": np.asarray("reflect"),
            "intensity_interpolation": np.asarray("linear"),
            "support_interpolation": np.asarray("nearest"),
        }


def _load(item: str | Path | CanonicalVolume) -> CanonicalVolume:
    return item if isinstance(item, CanonicalVolume) else load_nifti_ras(item)


def _support_qc(
    support: CanonicalVolume | None,
    sampled_zyx: np.ndarray | None,
    grid: PhysicalGrid,
    source_to_anchor_ras: np.ndarray,
) -> SupportQC:
    if support is None or sampled_zyx is None:
        return SupportQC.unavailable()
    audit = audit_support_containment(
        support,
        grid,
        source_to_anchor_ras=source_to_anchor_ras,
    )
    return SupportQC(
        available=True,
        source_positive_voxels=audit.full_positive_voxels,
        retained_positive_voxels=audit.retained_positive_voxels,
        nn_target_positive_voxels=int(np.count_nonzero(sampled_zyx)),
        source_volume_mm3=audit.full_physical_volume_mm3,
        retained_source_volume_mm3=audit.retained_physical_volume_mm3,
        retained_positive_voxel_fraction=audit.retained_positive_voxel_fraction,
        physical_volume_retention=audit.physical_volume_retention,
        exact_full_support_containment=audit.exact_full_support_containment,
        source_boundary_touch=audit.source_boundary_touch,
        target_boundary_touch=audit.target_boundary_touch,
        minimum_margin_mm=audit.minimum_margin_mm,
    )


def build_patient_dce7(
    patient_id: str,
    visits: Mapping[str, VisitInput],
    *,
    cohort: str = "I-SPY2",
    formal_ftv_overlap: bool = False,
    registration_strategy: str = "C1B-H",
) -> PatientDCE7:
    """Build T0-T3 on one immutable T0 physical grid.

    The grid is frozen before any T1-T3 image or support is loaded.  A released
    T0 support determines its physical centre when available; otherwise the T0
    acquisition centre is used.  Later supports can only produce NN QC values.
    """

    if tuple(sorted(visits)) != tuple(sorted(VISITS)):
        raise ValueError(f"visits must contain exactly {VISITS}, got {tuple(visits)}")
    cohort = str(cohort)
    registration_strategy = str(registration_strategy)
    if cohort not in {"I-SPY1", "I-SPY2"}:
        raise ValueError("cohort must be I-SPY1 or I-SPY2")
    if registration_strategy not in {"C1B-H", "C1B-R"}:
        raise ValueError("registration strategy must be C1B-H or C1B-R")
    formal_ftv_overlap = bool(formal_ftv_overlap)
    if formal_ftv_overlap and cohort != "I-SPY2":
        raise ValueError("formal FTV overlap is restricted to the I-SPY2 cohort")
    for key, spec in visits.items():
        if spec.visit != key:
            raise ValueError(
                f"visit mapping key {key!r} disagrees with spec {spec.visit!r}"
            )

    t0_spec = visits["T0"]
    t0_volume = _load(t0_spec.dce)
    t0_transform = validate_source_to_anchor_transform(t0_spec.source_to_anchor_ras)
    if not np.allclose(t0_transform, np.eye(4), atol=1e-8, rtol=0.0):
        raise ValueError(
            "T0 defines the anchor frame and cannot have a non-identity transform"
        )
    t0_support = (
        _load(t0_spec.localization_support)
        if t0_spec.localization_support is not None
        else None
    )
    if formal_ftv_overlap and t0_support is None:
        raise ValueError("formal FTV-overlap T0 support is missing")
    if t0_support is not None:
        center = support_bbox_center_ras(t0_support)
        anchor_provenance = "released_t0_localization_support_bbox_center"
    else:
        center = acquisition_center_ras(t0_volume)
        anchor_provenance = "t0_acquisition_physical_center_fallback"
    # No shape/spacing knobs are exposed here: patient caches always use the
    # single frozen C1B geometry.
    grid = make_c1b_grid(center)

    image = np.empty((4, 7, *grid.shape_zyx), dtype=np.float32)
    valid_source = np.empty((4, 1, *grid.shape_zyx), dtype=np.uint8)
    phase_indices = np.empty((4, 3), dtype=np.int16)
    phase_counts = np.empty(4, dtype=np.int16)
    p01 = np.empty((4, 7), dtype=np.float32)
    p99 = np.empty((4, 7), dtype=np.float32)
    median = np.empty((4, 7), dtype=np.float32)
    scale = np.empty((4, 7), dtype=np.float32)
    scale_source = np.empty((4, 7), dtype="<U24")
    samples_per_output = np.empty((4, 3), dtype=np.float64)
    sigma = np.empty((4, 3), dtype=np.float64)
    anti_alias = np.empty(4, dtype=np.uint8)
    source_hashes = np.empty(4, dtype="<U64")
    support_hashes = np.full(4, "NONE", dtype="<U64")
    support_scopes = np.empty(4, dtype="<U32")
    phase_metadata_hashes = np.empty(4, dtype="<U64")
    transforms = np.empty((4, 4, 4), dtype=np.float64)
    support_qc: list[SupportQC] = []

    for visit_index, visit_name in enumerate(VISITS):
        spec = visits[visit_name]
        # T0 was the only visit opened while choosing the grid.  Follow-ups are
        # first touched here, after the anchor is immutable.
        volume = t0_volume if visit_name == "T0" else _load(spec.dce)
        source_hashes[visit_index] = canonical_volume_sha256(volume)
        transform = validate_source_to_anchor_transform(spec.source_to_anchor_ras)
        if registration_strategy == "C1B-H" and not np.allclose(
            transform, np.eye(4), atol=1e-8, rtol=0.0
        ):
            raise ValueError("C1B-H cannot contain a non-identity registration transform")
        phase_metadata_hashes[visit_index] = phase_metadata_sha256(
            spec.phase_metadata
        )
        result: VisitDCE7 = build_visit_dce7(
            volume,
            grid,
            phase_metadata=spec.phase_metadata,
            source_to_anchor_ras=transform,
        )
        image[visit_index] = result.tensor_czyx
        valid_source[visit_index, 0] = result.valid_source_mask_zyx
        phase_indices[visit_index] = result.phase_selection.indices
        phase_counts[visit_index] = result.phase_count
        p01[visit_index] = result.normalization.p01
        p99[visit_index] = result.normalization.p99
        median[visit_index] = result.normalization.median
        scale[visit_index] = result.normalization.scale
        scale_source[visit_index] = result.normalization.scale_source
        samples_per_output[visit_index] = (
            result.resampling.source_samples_per_output_axis
        )
        sigma[visit_index] = result.resampling.anti_alias_sigma_source_voxels
        anti_alias[visit_index] = result.resampling.anti_alias_applied
        transforms[visit_index] = transform

        support = (
            t0_support
            if visit_name == "T0"
            else (
                _load(spec.localization_support)
                if spec.localization_support is not None
                else None
            )
        )
        if visit_name == "T0":
            support_scopes[visit_index] = (
                "anchor_and_qc" if support is not None else "none_fallback"
            )
        elif support is not None:
            if not formal_ftv_overlap:
                raise ValueError(
                    "base-only T1-T3 localization support must not be loaded"
                )
            support_scopes[visit_index] = "formal_qc_only"
        else:
            if formal_ftv_overlap:
                raise ValueError("formal FTV-overlap T1-T3 support is missing")
            support_scopes[visit_index] = "not_loaded"
        sampled_support = None
        if support is not None:
            support_hashes[visit_index] = canonical_volume_sha256(support)
            sampled_support = resample_support_nearest(
                support,
                grid,
                source_to_anchor_ras=transform,
            )
        support_qc.append(_support_qc(support, sampled_support, grid, transform))

    contract_digest = builder_contract_sha256()
    provenance_digest = input_provenance_sha256(
        patient_id=str(patient_id),
        cohort=cohort,
        formal_ftv_overlap=formal_ftv_overlap,
        registration_strategy=registration_strategy,
        source_hashes=source_hashes,
        support_hashes=support_hashes,
        support_scope=support_scopes,
        phase_metadata_hashes=phase_metadata_hashes,
        phase_counts=phase_counts,
        phase_indices=phase_indices,
        source_to_anchor_ras=transforms,
        grid=grid,
        anchor_provenance=anchor_provenance,
        contract_sha256=contract_digest,
    )

    return PatientDCE7(
        patient_id=str(patient_id),
        cohort=cohort,
        formal_ftv_overlap=formal_ftv_overlap,
        registration_strategy=registration_strategy,
        image=image,
        valid_source_mask=valid_source,
        phase_indices=phase_indices,
        phase_counts=phase_counts,
        normalization_p01=p01,
        normalization_p99=p99,
        normalization_median=median,
        normalization_scale=scale,
        normalization_scale_source=scale_source,
        source_samples_per_output_axis=samples_per_output,
        anti_alias_sigma_source_voxels=sigma,
        anti_alias_applied=anti_alias,
        source_canonical_sha256=source_hashes,
        support_canonical_sha256=support_hashes,
        support_scope=support_scopes,
        phase_metadata_sha256=phase_metadata_hashes,
        builder_contract_sha256=contract_digest,
        input_provenance_sha256=provenance_digest,
        source_to_anchor_ras=transforms,
        support_qc=tuple(support_qc),
        grid=grid,
        anchor_provenance=anchor_provenance,
    )

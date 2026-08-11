"""Source-domain physical containment and crop-window construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Iterable

import numpy as np
from scipy import ndimage


def _triplet(values: Iterable[Any], name: str, *, positive: bool = False) -> np.ndarray:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite length-three vector")
    if positive and np.any(array <= 0):
        raise ValueError(f"{name} must be strictly positive")
    return array


def _positive_integer_triplet(values: Iterable[Any], name: str) -> tuple[int, int, int]:
    raw = tuple(values)
    if len(raw) != 3 or any(isinstance(value, (bool, np.bool_)) for value in raw):
        raise ValueError(f"{name} must be a positive integer triplet")
    try:
        numeric = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer triplet") from exc
    rounded = np.rint(numeric)
    if (
        numeric.shape != (3,)
        or not np.all(np.isfinite(numeric))
        or not np.array_equal(numeric, rounded)
        or np.any(rounded <= 0)
    ):
        raise ValueError(f"{name} must be a positive integer triplet")
    return tuple(int(value) for value in rounded)


def orthonormal_index_basis(affine: np.ndarray) -> np.ndarray:
    """Return orthonormal world directions for the affine's XYZ index axes.

    Polar decomposition removes tiny header shear while retaining reflection and
    axis signs.  Columns are the physical directions of increasing X, Y and Z.
    """

    affine = np.asarray(affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        raise ValueError("affine must be a finite 4x4 matrix")
    linear = affine[:3, :3]
    if np.linalg.matrix_rank(linear) < 3:
        raise ValueError("affine linear part is singular")
    unit = linear / np.linalg.norm(linear, axis=0, keepdims=True)
    left, _, right_t = np.linalg.svd(unit)
    basis = left @ right_t
    if np.linalg.det(basis) * np.linalg.det(unit) < 0:
        left[:, -1] *= -1.0
        basis = left @ right_t
    return basis


def world_to_frame(points_world: np.ndarray, frame_basis: np.ndarray) -> np.ndarray:
    points = np.asarray(points_world, dtype=np.float64)
    basis = np.asarray(frame_basis, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_world must have shape [N,3]")
    if basis.shape != (3, 3):
        raise ValueError("frame_basis must be 3x3")
    return points @ basis


def index_to_world(points_xyz: np.ndarray, affine: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float64)
    affine = np.asarray(affine, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must have shape [N,3]")
    return points @ affine[:3, :3].T + affine[:3, 3][None, :]


def voxel_half_extent_in_frame(affine: np.ndarray, frame_basis: np.ndarray) -> np.ndarray:
    """Projected half-width of one source voxel footprint on each frame axis."""

    projection = np.asarray(frame_basis, dtype=np.float64).T @ np.asarray(
        affine, dtype=np.float64
    )[:3, :3]
    return 0.5 * np.sum(np.abs(projection), axis=1)


def bbox_footprint_in_frame(
    affine: np.ndarray,
    frame_basis: np.ndarray,
    bbox_min_xyz: Iterable[Any],
    bbox_max_xyz: Iterable[Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Inclusive voxel bbox converted to continuous physical footprint bounds."""

    minimum = _triplet(bbox_min_xyz, "bbox_min_xyz") - 0.5
    maximum = _triplet(bbox_max_xyz, "bbox_max_xyz") + 0.5
    if np.any(maximum <= minimum):
        raise ValueError("bbox maximum must be no smaller than minimum")
    corners = np.asarray(list(product(*zip(minimum, maximum, strict=True))))
    frame = world_to_frame(index_to_world(corners, affine), frame_basis)
    return frame.min(axis=0), frame.max(axis=0)


@dataclass(frozen=True)
class PhysicalWindow:
    contract: str
    view: str
    frame_basis: np.ndarray
    center_frame_mm: np.ndarray
    fov_xyz_mm: np.ndarray
    output_shape_zyx: tuple[int, int, int]
    anchor_policy: str
    reference_visit: str
    causal_deployability: str
    audit_only: bool = False
    margin_mm: float | None = None
    nominal_fov_xyz_mm: np.ndarray | None = None
    expanded_from_nominal: bool = False
    direct_bbox_resize: bool = False

    def __post_init__(self) -> None:
        basis = np.asarray(self.frame_basis, dtype=np.float64)
        center = _triplet(self.center_frame_mm, "center_frame_mm")
        fov = _triplet(self.fov_xyz_mm, "fov_xyz_mm", positive=True)
        shape = _positive_integer_triplet(
            self.output_shape_zyx, "output_shape_zyx"
        )
        if basis.shape != (3, 3) or not np.allclose(
            basis.T @ basis, np.eye(3), atol=1e-6
        ):
            raise ValueError("frame_basis must be orthonormal")
        object.__setattr__(self, "frame_basis", basis)
        object.__setattr__(self, "center_frame_mm", center)
        object.__setattr__(self, "fov_xyz_mm", fov)
        object.__setattr__(self, "output_shape_zyx", shape)
        if self.nominal_fov_xyz_mm is not None:
            object.__setattr__(
                self,
                "nominal_fov_xyz_mm",
                _triplet(
                    self.nominal_fov_xyz_mm,
                    "nominal_fov_xyz_mm",
                    positive=True,
                ),
            )

    @property
    def low_frame_mm(self) -> np.ndarray:
        return self.center_frame_mm - 0.5 * self.fov_xyz_mm

    @property
    def high_frame_mm(self) -> np.ndarray:
        return self.center_frame_mm + 0.5 * self.fov_xyz_mm

    @property
    def output_shape_xyz(self) -> np.ndarray:
        return np.asarray(self.output_shape_zyx[::-1], dtype=np.int64)

    @property
    def effective_spacing_xyz_mm(self) -> np.ndarray:
        return self.fov_xyz_mm / self.output_shape_xyz

    def to_record(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "view": self.view,
            "anchor_policy": self.anchor_policy,
            "reference_visit": self.reference_visit,
            "causal_deployability": self.causal_deployability,
            "audit_only": self.audit_only,
            "margin_mm": self.margin_mm,
            "center_frame_x_mm": float(self.center_frame_mm[0]),
            "center_frame_y_mm": float(self.center_frame_mm[1]),
            "center_frame_z_mm": float(self.center_frame_mm[2]),
            "fov_x_mm": float(self.fov_xyz_mm[0]),
            "fov_y_mm": float(self.fov_xyz_mm[1]),
            "fov_z_mm": float(self.fov_xyz_mm[2]),
            "output_z": self.output_shape_zyx[0],
            "output_y": self.output_shape_zyx[1],
            "output_x": self.output_shape_zyx[2],
            "effective_spacing_x_mm": float(self.effective_spacing_xyz_mm[0]),
            "effective_spacing_y_mm": float(self.effective_spacing_xyz_mm[1]),
            "effective_spacing_z_mm": float(self.effective_spacing_xyz_mm[2]),
            "expanded_from_nominal": self.expanded_from_nominal,
            "direct_bbox_resize": self.direct_bbox_resize,
        }


def make_fixed_expand_window(
    *,
    contract: str,
    view: str,
    frame_basis: np.ndarray,
    support_low_frame_mm: Iterable[Any],
    support_high_frame_mm: Iterable[Any],
    nominal_fov_xyz_mm: Iterable[Any],
    output_shape_zyx: Iterable[int],
    margin_mm: float,
    anchor_policy: str,
    reference_visit: str,
    causal_deployability: str,
    center_frame_mm: Iterable[Any] | None = None,
    audit_only: bool = False,
    allow_expand: bool = True,
) -> PhysicalWindow:
    low = _triplet(support_low_frame_mm, "support_low_frame_mm")
    high = _triplet(support_high_frame_mm, "support_high_frame_mm")
    if np.any(high <= low):
        raise ValueError("support bounds must have positive extent")
    if not np.isfinite(margin_mm) or margin_mm < 0:
        raise ValueError("margin_mm must be finite and non-negative")
    nominal = _triplet(nominal_fov_xyz_mm, "nominal_fov_xyz_mm", positive=True)
    required = high - low + 2.0 * float(margin_mm)
    fov = np.maximum(nominal, required) if allow_expand else nominal.copy()
    center = 0.5 * (low + high) if center_frame_mm is None else _triplet(
        center_frame_mm, "center_frame_mm"
    )
    return PhysicalWindow(
        contract=contract,
        view=view,
        frame_basis=frame_basis,
        center_frame_mm=center,
        fov_xyz_mm=fov,
        output_shape_zyx=_positive_integer_triplet(
            output_shape_zyx, "output_shape_zyx"
        ),
        anchor_policy=anchor_policy,
        reference_visit=reference_visit,
        causal_deployability=causal_deployability,
        audit_only=audit_only,
        margin_mm=float(margin_mm),
        nominal_fov_xyz_mm=nominal,
        expanded_from_nominal=bool(np.any(fov > nominal + 1e-9)),
        direct_bbox_resize=False,
    )


def make_tight_resize_window(
    *,
    contract: str,
    view: str,
    frame_basis: np.ndarray,
    support_low_frame_mm: Iterable[Any],
    support_high_frame_mm: Iterable[Any],
    output_shape_zyx: Iterable[int],
    margin_mm: float,
    reference_visit: str,
) -> PhysicalWindow:
    low = _triplet(support_low_frame_mm, "support_low_frame_mm")
    high = _triplet(support_high_frame_mm, "support_high_frame_mm")
    if np.any(high <= low):
        raise ValueError("support bounds must have positive extent")
    if not np.isfinite(margin_mm) or margin_mm < 0:
        raise ValueError("margin_mm must be finite and non-negative")
    fov = high - low + 2.0 * float(margin_mm)
    return PhysicalWindow(
        contract=contract,
        view=view,
        frame_basis=frame_basis,
        center_frame_mm=0.5 * (low + high),
        fov_xyz_mm=fov,
        output_shape_zyx=_positive_integer_triplet(
            output_shape_zyx, "output_shape_zyx"
        ),
        anchor_policy="current_visit_bbox_plus_margin_direct_resize",
        reference_visit=reference_visit,
        causal_deployability="CURRENT_VISIT_CAUSAL_WITH_SIZE_NORMALIZATION_RISK",
        margin_mm=float(margin_mm),
        nominal_fov_xyz_mm=None,
        expanded_from_nominal=False,
        direct_bbox_resize=True,
    )


def make_union_window(
    *,
    contract: str,
    view: str,
    frame_basis: np.ndarray,
    visit_bounds_frame_mm: Iterable[tuple[np.ndarray, np.ndarray]],
    nominal_fov_xyz_mm: Iterable[Any],
    output_shape_zyx: Iterable[int],
    margin_mm: float,
) -> PhysicalWindow:
    bounds = list(visit_bounds_frame_mm)
    if not bounds:
        raise ValueError("visit_bounds_frame_mm cannot be empty")
    low = np.min(np.stack([np.asarray(item[0]) for item in bounds]), axis=0)
    high = np.max(np.stack([np.asarray(item[1]) for item in bounds]), axis=0)
    return make_fixed_expand_window(
        contract=contract,
        view=view,
        frame_basis=frame_basis,
        support_low_frame_mm=low,
        support_high_frame_mm=high,
        nominal_fov_xyz_mm=nominal_fov_xyz_mm,
        output_shape_zyx=output_shape_zyx,
        margin_mm=margin_mm,
        anchor_policy="T0_T3_union_bbox",
        reference_visit="T0_T3_UNION",
        causal_deployability="AUDIT_ONLY_FUTURE_INFORMATION",
        audit_only=True,
        allow_expand=True,
    )


@dataclass(frozen=True)
class SupportAudit:
    full_support_voxels: int
    retained_support_voxels: int
    retained_ftv_fraction: float
    physical_volume_retention: float
    exact_full_support_containment: bool
    bbox_fully_contained: bool
    boundary_touch: bool
    suspected_truncation: bool
    severe_truncation: bool
    sufficient_containment: bool
    minimum_margin_mm: float
    extent_retention_x: float
    extent_retention_y: float
    extent_retention_z: float
    extent_retention_min_axis: float
    surface_voxels: int
    retained_surface_voxels: int
    surface_voxel_retention: float
    component_count: int
    cut_component_count: int
    missed_component_count: int
    lesion_physical_volume_mm3: float
    window_physical_volume_mm3: float
    context_to_lesion_volume_ratio: float
    context_margin_x_low_mm: float
    context_margin_x_high_mm: float
    context_margin_y_low_mm: float
    context_margin_y_high_mm: float
    context_margin_z_low_mm: float
    context_margin_z_high_mm: float
    resize_factor_x: float
    resize_factor_y: float
    resize_factor_z: float
    resize_anisotropy_ratio: float
    output_anisotropy_ratio: float

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreparedSupport:
    """Window-independent support topology cached once per patient visit."""

    coordinates_xyz: np.ndarray
    surface_coordinates_xyz: np.ndarray
    coordinate_component_labels: np.ndarray
    component_sizes: np.ndarray
    component_count: int
    affine: np.ndarray
    voxel_volume_mm3: float
    spacing_xyz_mm: np.ndarray


def prepare_support(mask_xyz: Any, affine: np.ndarray) -> PreparedSupport:
    mask = np.asarray(mask_xyz, dtype=bool)
    if mask.ndim != 3:
        raise ValueError(f"mask_xyz must be 3-D; got {mask.shape}")
    coordinates = np.argwhere(mask)
    if len(coordinates) == 0:
        raise ValueError("empty support is unavailable and must fail closed upstream")
    affine = np.asarray(affine, dtype=np.float64)
    if affine.shape != (4, 4) or np.linalg.matrix_rank(affine[:3, :3]) < 3:
        raise ValueError("affine must be finite, 4x4 and nonsingular")

    tight_min = coordinates.min(axis=0)
    tight_max = coordinates.max(axis=0)
    slices = tuple(
        slice(int(low), int(high) + 1)
        for low, high in zip(tight_min, tight_max, strict=True)
    )
    tight = mask[slices]
    structure = np.ones((3, 3, 3), dtype=bool)
    eroded = ndimage.binary_erosion(tight, structure=structure, border_value=0)
    surface_global = np.argwhere(tight & ~eroded) + tight_min[None, :]
    labels, component_count = ndimage.label(tight, structure=structure.astype(np.uint8))
    coordinate_labels = labels[tuple((coordinates - tight_min[None, :]).T)]
    component_sizes = np.bincount(
        coordinate_labels, minlength=int(component_count) + 1
    )
    return PreparedSupport(
        coordinates_xyz=coordinates,
        surface_coordinates_xyz=surface_global,
        coordinate_component_labels=coordinate_labels,
        component_sizes=component_sizes,
        component_count=int(component_count),
        affine=affine,
        voxel_volume_mm3=float(abs(np.linalg.det(affine[:3, :3]))),
        spacing_xyz_mm=np.linalg.norm(affine[:3, :3], axis=0),
    )


def _inside_full_footprint(
    centers_frame: np.ndarray,
    half_extent_frame: np.ndarray,
    window: PhysicalWindow,
    *,
    tolerance_mm: float,
) -> np.ndarray:
    return np.all(
        (centers_frame - half_extent_frame[None, :]
         >= window.low_frame_mm[None, :] - tolerance_mm)
        & (centers_frame + half_extent_frame[None, :]
           <= window.high_frame_mm[None, :] + tolerance_mm),
        axis=1,
    )


def _touches_any_face(
    centers_frame: np.ndarray,
    half_extent_frame: np.ndarray,
    window: PhysicalWindow,
    *,
    tolerance_mm: float,
) -> bool:
    voxel_low = centers_frame - half_extent_frame[None, :]
    voxel_high = centers_frame + half_extent_frame[None, :]
    overlaps = np.all(
        (voxel_high >= window.low_frame_mm[None, :] - tolerance_mm)
        & (voxel_low <= window.high_frame_mm[None, :] + tolerance_mm),
        axis=1,
    )
    if not np.any(overlaps):
        return False
    low_touch = np.any(
        overlaps[:, None]
        & (voxel_low <= window.low_frame_mm[None, :] + tolerance_mm)
        & (voxel_high >= window.low_frame_mm[None, :] - tolerance_mm)
    )
    high_touch = np.any(
        overlaps[:, None]
        & (voxel_low <= window.high_frame_mm[None, :] + tolerance_mm)
        & (voxel_high >= window.high_frame_mm[None, :] - tolerance_mm)
    )
    return bool(low_touch or high_touch)


def audit_support(
    mask_xyz: Any,
    affine: np.ndarray,
    window: PhysicalWindow,
    *,
    retention_threshold: float = 0.99,
    severe_threshold: float = 0.90,
    tolerance_mm: float = 1e-6,
) -> SupportAudit:
    """Audit support in the source domain before interpolation/resampling."""

    return audit_prepared_support(
        prepare_support(mask_xyz, affine),
        window,
        retention_threshold=retention_threshold,
        severe_threshold=severe_threshold,
        tolerance_mm=tolerance_mm,
    )


def audit_prepared_support(
    prepared: PreparedSupport,
    window: PhysicalWindow,
    *,
    retention_threshold: float = 0.99,
    severe_threshold: float = 0.90,
    tolerance_mm: float = 1e-6,
) -> SupportAudit:
    """Audit a previously prepared support against one physical window."""

    coordinates = prepared.coordinates_xyz
    affine = prepared.affine
    centers_frame = world_to_frame(index_to_world(coordinates, affine), window.frame_basis)
    half_extent = voxel_half_extent_in_frame(affine, window.frame_basis)
    retained = _inside_full_footprint(
        centers_frame, half_extent, window, tolerance_mm=tolerance_mm
    )
    full_count = int(len(coordinates))
    retained_count = int(np.count_nonzero(retained))
    retention = float(retained_count / full_count)

    support_low = centers_frame.min(axis=0) - half_extent
    support_high = centers_frame.max(axis=0) + half_extent
    low_margins = support_low - window.low_frame_mm
    high_margins = window.high_frame_mm - support_high
    all_margins = np.concatenate([low_margins, high_margins])
    minimum_margin = float(all_margins.min())
    bbox_contained = bool(minimum_margin >= -tolerance_mm)
    boundary_touch = _touches_any_face(
        centers_frame, half_extent, window, tolerance_mm=tolerance_mm
    )
    exact = bool(retained_count == full_count)
    suspected = bool(boundary_touch or retention < retention_threshold)
    severe = bool(retention < severe_threshold)
    sufficient = bool(retention >= retention_threshold and not boundary_touch)

    full_extent = support_high - support_low
    if retained_count:
        retained_centers = centers_frame[retained]
        retained_low = retained_centers.min(axis=0) - half_extent
        retained_high = retained_centers.max(axis=0) + half_extent
        extent_retention = np.clip(
            (retained_high - retained_low) / full_extent, 0.0, 1.0
        )
    else:
        extent_retention = np.zeros(3, dtype=np.float64)

    surface_global = prepared.surface_coordinates_xyz
    surface_frame = world_to_frame(index_to_world(surface_global, affine), window.frame_basis)
    retained_surface = _inside_full_footprint(
        surface_frame, half_extent, window, tolerance_mm=tolerance_mm
    )

    label_for_coordinate = prepared.coordinate_component_labels
    component_count = prepared.component_count
    total_by_label = prepared.component_sizes
    kept_by_label = np.bincount(
        label_for_coordinate[retained], minlength=component_count + 1
    )
    cut_components = int(
        np.count_nonzero(
            (kept_by_label[1:] > 0)
            & (kept_by_label[1:] < total_by_label[1:])
        )
    )
    missed_components = int(np.count_nonzero(kept_by_label[1:] == 0))

    spacing_xyz = prepared.spacing_xyz_mm
    voxel_volume = prepared.voxel_volume_mm3
    lesion_volume = float(full_count * voxel_volume)
    window_volume = float(np.prod(window.fov_xyz_mm))
    context_ratio = float(max(window_volume - lesion_volume, 0.0) / lesion_volume)
    effective_spacing = window.effective_spacing_xyz_mm
    resize_factor = effective_spacing / spacing_xyz
    resize_anisotropy = float(resize_factor.max() / resize_factor.min())
    output_anisotropy = float(effective_spacing.max() / effective_spacing.min())
    surface_count = int(len(surface_global))
    retained_surface_count = int(np.count_nonzero(retained_surface))

    return SupportAudit(
        full_support_voxels=full_count,
        retained_support_voxels=retained_count,
        retained_ftv_fraction=retention,
        physical_volume_retention=retention,
        exact_full_support_containment=exact,
        bbox_fully_contained=bbox_contained,
        boundary_touch=boundary_touch,
        suspected_truncation=suspected,
        severe_truncation=severe,
        sufficient_containment=sufficient,
        minimum_margin_mm=minimum_margin,
        extent_retention_x=float(extent_retention[0]),
        extent_retention_y=float(extent_retention[1]),
        extent_retention_z=float(extent_retention[2]),
        extent_retention_min_axis=float(extent_retention.min()),
        surface_voxels=surface_count,
        retained_surface_voxels=retained_surface_count,
        surface_voxel_retention=(
            float(retained_surface_count / surface_count) if surface_count else 0.0
        ),
        component_count=int(component_count),
        cut_component_count=cut_components,
        missed_component_count=missed_components,
        lesion_physical_volume_mm3=lesion_volume,
        window_physical_volume_mm3=window_volume,
        context_to_lesion_volume_ratio=context_ratio,
        context_margin_x_low_mm=float(low_margins[0]),
        context_margin_x_high_mm=float(high_margins[0]),
        context_margin_y_low_mm=float(low_margins[1]),
        context_margin_y_high_mm=float(high_margins[1]),
        context_margin_z_low_mm=float(low_margins[2]),
        context_margin_z_high_mm=float(high_margins[2]),
        resize_factor_x=float(resize_factor[0]),
        resize_factor_y=float(resize_factor[1]),
        resize_factor_z=float(resize_factor[2]),
        resize_anisotropy_ratio=resize_anisotropy,
        output_anisotropy_ratio=output_anisotropy,
    )

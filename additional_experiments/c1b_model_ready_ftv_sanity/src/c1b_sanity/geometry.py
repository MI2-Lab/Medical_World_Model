"""Physical-space geometry primitives for the frozen C1B input contract.

Arrays loaded here use nibabel's native ``[X, Y, Z, ...]`` convention.  The
model-facing ``[Z, Y, X]`` conversion happens only after physical resampling.
In particular, canonicalization applies the affine-derived permutation and
flips to the array; it never merely changes an orientation label.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
from nibabel.orientations import (
    aff2axcodes,
    apply_orientation,
    axcodes2ornt,
    inv_ornt_aff,
    io_orientation,
    ornt_transform,
)
from scipy import ndimage as ndi


C1B_SHAPE_ZYX: tuple[int, int, int] = (112, 176, 160)
C1B_SPACING_XYZ_MM: tuple[float, float, float] = (0.9, 0.9, 2.0)
RAS_AXCODES: tuple[str, str, str] = ("R", "A", "S")


def validate_affine(affine: np.ndarray, *, name: str = "affine") -> np.ndarray:
    """Return a validated float64 voxel-to-RAS affine.

    A usable imaging affine must be finite, homogeneous, and invertible.  The
    determinant tolerance is scaled to the matrix norm so valid sub-millimetre
    spacings are not rejected.
    """

    result = np.asarray(affine, dtype=np.float64)
    if result.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(result[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8, rtol=0.0):
        raise ValueError(f"{name} is not a homogeneous spatial affine")
    linear = result[:3, :3]
    singular_values = np.linalg.svd(linear, compute_uv=False)
    if (
        singular_values[-1]
        <= np.finfo(np.float64).eps * max(singular_values[0], 1.0) * 16.0
    ):
        raise ValueError(f"{name} is singular or numerically non-invertible")
    return result


def validate_source_to_anchor_transform(transform: np.ndarray | None) -> np.ndarray:
    """Validate an optional externally fitted source-world to anchor-world hook.

    Registration fitting intentionally does not live in this package.  This
    function only validates and returns a transform supplied by an upstream,
    independently gated registration process.
    """

    if transform is None:
        return np.eye(4, dtype=np.float64)
    result = validate_affine(transform, name="source_to_anchor_ras")
    rotation = result[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0.0):
        raise ValueError("source_to_anchor_ras must be rigid (no scale or shear)")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5, rtol=0.0):
        raise ValueError("source_to_anchor_ras must be a proper rigid transform")
    return result


@dataclass(frozen=True)
class CanonicalVolume:
    """A finite array whose first three axes and affine are truly RAS+."""

    data: np.ndarray
    affine_ras: np.ndarray
    original_axcodes: tuple[str, str, str]
    orientation_transform: np.ndarray
    source_path: Path | None = None

    @property
    def shape_xyz(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.data.shape[:3])


def canonical_volume_sha256(volume: CanonicalVolume) -> str:
    """Hash the exact canonical voxels and physical affine used by C1B.

    This deliberately fingerprints the decoded, scaled, truly RAS+ source
    rather than a container file's incidental header bytes.  It therefore
    closes cache provenance across NIfTI rewrites while still changing for any
    model-relevant voxel, shape, dtype, or affine change.
    """

    data = np.ascontiguousarray(volume.data)
    affine = np.ascontiguousarray(
        validate_affine(volume.affine_ras, name="canonical source affine"),
        dtype="<f8",
    )
    digest = hashlib.sha256()
    digest.update(b"c1b-canonical-volume-v1\0")
    digest.update(data.dtype.str.encode("ascii"))
    digest.update(np.asarray(data.shape, dtype="<i8").tobytes())
    digest.update(memoryview(affine).cast("B"))
    if data.nbytes:
        digest.update(memoryview(data).cast("B"))
    return digest.hexdigest()


def canonicalize_to_ras(
    data: np.ndarray,
    affine: np.ndarray,
    *,
    source_path: str | Path | None = None,
) -> CanonicalVolume:
    """Apply the affine-implied permutation/flips and return a RAS+ volume."""

    array = np.asarray(data)
    if array.ndim < 3:
        raise ValueError(
            f"volume must have at least three dimensions, got {array.shape}"
        )
    if any(int(length) < 1 for length in array.shape[:3]):
        raise ValueError(f"volume has an empty spatial axis: {array.shape[:3]}")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"volume must be numeric, got dtype {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError("volume contains non-finite values")

    source_affine = validate_affine(affine)
    source_ornt = io_orientation(source_affine)
    if source_ornt.shape != (3, 2) or np.isnan(source_ornt).any():
        raise ValueError("affine does not define all three spatial orientations")
    target_ornt = axcodes2ornt(RAS_AXCODES)
    transform = ornt_transform(source_ornt, target_ornt)
    original_shape = tuple(int(value) for value in array.shape[:3])
    reoriented = apply_orientation(array, transform)
    ras_affine = source_affine @ inv_ornt_aff(transform, original_shape)
    ras_affine = validate_affine(ras_affine, name="canonical affine")
    axcodes = aff2axcodes(ras_affine)
    if tuple(axcodes) != RAS_AXCODES:
        raise ValueError(
            f"canonicalization failed: output orientation is {axcodes}, not RAS+"
        )

    # float32 is the frozen intensity/cache precision.  Contiguity also removes
    # negative strides produced by axis flips before scipy interpolation.
    output = np.ascontiguousarray(reoriented, dtype=np.float32)
    return CanonicalVolume(
        data=output,
        affine_ras=ras_affine,
        original_axcodes=tuple(str(code) for code in aff2axcodes(source_affine)),
        orientation_transform=np.asarray(transform, dtype=np.float64),
        source_path=Path(source_path) if source_path is not None else None,
    )


def load_nifti_ras(path: str | Path) -> CanonicalVolume:
    """Load a NIfTI, validate its data/affine, and truly canonicalize to RAS+."""

    source = Path(path)
    image = nib.load(str(source))
    # np.asanyarray applies a NIfTI scaling proxy when present without forcing
    # float64 as get_fdata() would.
    data = np.asanyarray(image.dataobj)
    return canonicalize_to_ras(data, image.affine, source_path=source)


@dataclass(frozen=True)
class PhysicalGrid:
    """Axis-aligned RAS+ target grid anchored at a physical centre."""

    shape_zyx: tuple[int, int, int]
    spacing_xyz_mm: tuple[float, float, float]
    center_ras_mm: tuple[float, float, float]

    def __post_init__(self) -> None:
        shape = tuple(int(value) for value in self.shape_zyx)
        spacing = tuple(float(value) for value in self.spacing_xyz_mm)
        center = tuple(float(value) for value in self.center_ras_mm)
        if len(shape) != 3 or any(value < 1 for value in shape):
            raise ValueError(
                f"shape_zyx must contain three positive values, got {shape}"
            )
        if (
            len(spacing) != 3
            or not np.isfinite(spacing).all()
            or any(value <= 0 for value in spacing)
        ):
            raise ValueError(
                f"spacing_xyz_mm must contain three finite positive values, got {spacing}"
            )
        if len(center) != 3 or not np.isfinite(center).all():
            raise ValueError(
                f"center_ras_mm must contain three finite values, got {center}"
            )
        object.__setattr__(self, "shape_zyx", shape)
        object.__setattr__(self, "spacing_xyz_mm", spacing)
        object.__setattr__(self, "center_ras_mm", center)

    @property
    def shape_xyz(self) -> tuple[int, int, int]:
        return tuple(reversed(self.shape_zyx))

    @property
    def affine_ras(self) -> np.ndarray:
        spacing = np.asarray(self.spacing_xyz_mm, dtype=np.float64)
        shape = np.asarray(self.shape_xyz, dtype=np.float64)
        center = np.asarray(self.center_ras_mm, dtype=np.float64)
        affine = np.eye(4, dtype=np.float64)
        affine[:3, :3] = np.diag(spacing)
        # The requested physical centre lies midway between the two central
        # voxel centres for the even-sized frozen grid.
        affine[:3, 3] = center - 0.5 * (shape - 1.0) * spacing
        return affine

    @property
    def voxel_footprint_fov_xyz_mm(self) -> tuple[float, float, float]:
        return tuple(
            float(length * spacing)
            for length, spacing in zip(self.shape_xyz, self.spacing_xyz_mm)
        )


def make_c1b_grid(center_ras_mm: Sequence[float]) -> PhysicalGrid:
    """Create the single frozen C1B grid (112x176x160 ZYX, .9/.9/2 mm XYZ)."""

    return PhysicalGrid(
        shape_zyx=C1B_SHAPE_ZYX,
        spacing_xyz_mm=C1B_SPACING_XYZ_MM,
        center_ras_mm=tuple(float(value) for value in center_ras_mm),
    )


def voxel_to_world(
    affine: np.ndarray, voxel_xyz: np.ndarray | Sequence[float]
) -> np.ndarray:
    """Transform one or many ``[..., 3]`` voxel-centre coordinates to RAS mm."""

    spatial_affine = validate_affine(affine)
    points = np.asarray(voxel_xyz, dtype=np.float64)
    if points.shape[-1:] != (3,):
        raise ValueError(f"voxel coordinates must end in length 3, got {points.shape}")
    return nib.affines.apply_affine(spatial_affine, points)


def acquisition_center_ras(volume: CanonicalVolume) -> tuple[float, float, float]:
    """Return the physical centre of the T0 acquisition as outcome-free fallback."""

    center_voxel = 0.5 * (np.asarray(volume.shape_xyz, dtype=np.float64) - 1.0)
    center = voxel_to_world(volume.affine_ras, center_voxel)
    return tuple(float(value) for value in center)


def support_centroid_ras(
    support: CanonicalVolume,
    *,
    threshold: float = 0.5,
) -> tuple[float, float, float]:
    """Return the physical centroid of a released T0 localization support."""

    if support.data.ndim != 3:
        raise ValueError(f"localization support must be 3-D, got {support.data.shape}")
    mask = np.asarray(support.data > threshold)
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        raise ValueError("localization support is empty")
    center_voxel = coordinates.astype(np.float64).mean(axis=0)
    center = voxel_to_world(support.affine_ras, center_voxel)
    return tuple(float(value) for value in center)


def support_bbox_center_ras(
    support: CanonicalVolume,
    *,
    threshold: float = 0.5,
) -> tuple[float, float, float]:
    """Return the physical midpoint of a support's inclusive voxel bbox.

    This is the frozen C1B T0 anchor.  Taking the midpoint of the bounding-box
    voxel centres is equivalent to taking the midpoint of its full outer voxel
    footprints, including for a permuted/flipped source after RAS conversion.
    It intentionally differs from a foreground-mass centroid for asymmetric
    supports.
    """

    if support.data.ndim != 3:
        raise ValueError(f"localization support must be 3-D, got {support.data.shape}")
    coordinates = np.argwhere(np.asarray(support.data > threshold))
    if coordinates.size == 0:
        raise ValueError("localization support is empty")
    bbox_center_voxel = 0.5 * (
        coordinates.min(axis=0).astype(np.float64)
        + coordinates.max(axis=0).astype(np.float64)
    )
    center = voxel_to_world(support.affine_ras, bbox_center_voxel)
    return tuple(float(value) for value in center)


def input_from_output_affine(
    source_affine_ras: np.ndarray,
    grid: PhysicalGrid,
    source_to_anchor_ras: np.ndarray | None = None,
) -> np.ndarray:
    """Map target-grid voxel indices to source voxel indices.

    ``source_to_anchor_ras`` is a pre-fitted transform hook mapping source
    physical coordinates into the fixed T0 anchor frame.
    """

    source_affine = validate_affine(source_affine_ras, name="source affine")
    source_to_anchor = validate_source_to_anchor_transform(source_to_anchor_ras)
    mapping = (
        np.linalg.inv(source_affine) @ np.linalg.inv(source_to_anchor) @ grid.affine_ras
    )
    if not np.isfinite(mapping).all():
        raise ValueError("output-to-input sampling transform is non-finite")
    return mapping


def resample_support_nearest(
    support: CanonicalVolume,
    grid: PhysicalGrid,
    *,
    source_to_anchor_ras: np.ndarray | None = None,
    threshold: float = 0.5,
) -> np.ndarray:
    """Nearest-neighbour localization resampling for anchoring/QC only.

    The returned ``[Z, Y, X]`` mask is deliberately separate from every model
    tensor API.
    """

    if support.data.ndim != 3:
        raise ValueError(f"localization support must be 3-D, got {support.data.shape}")
    mapping = input_from_output_affine(
        support.affine_ras,
        grid,
        source_to_anchor_ras=source_to_anchor_ras,
    )
    sampled_xyz = ndi.affine_transform(
        np.asarray(support.data > threshold, dtype=np.uint8),
        matrix=mapping[:3, :3],
        offset=mapping[:3, 3],
        output_shape=grid.shape_xyz,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    )
    return np.ascontiguousarray(sampled_xyz.transpose(2, 1, 0) > 0)


@dataclass(frozen=True)
class PhysicalSupportAudit:
    """Exact source-domain support containment against a target footprint."""

    full_positive_voxels: int
    retained_positive_voxels: int
    retained_positive_voxel_fraction: float
    full_physical_volume_mm3: float
    retained_physical_volume_mm3: float
    physical_volume_retention: float
    exact_full_support_containment: bool
    source_boundary_touch: bool
    target_boundary_touch: bool
    minimum_margin_mm: float


def audit_support_containment(
    support: CanonicalVolume,
    grid: PhysicalGrid,
    *,
    source_to_anchor_ras: np.ndarray | None = None,
    threshold: float = 0.5,
    tolerance_mm: float = 1e-6,
) -> PhysicalSupportAudit:
    """Audit full positive-voxel footprints before NN resampling.

    Each positive source voxel remains a source-domain unit.  Its complete
    parallelepiped footprint is transformed into anchor RAS and is retained
    only when that entire footprint is inside the fixed target footprint.
    Thus the reported retention cannot exceed one and does not depend on NN
    voxel counts or target spacing.
    """

    if support.data.ndim != 3:
        raise ValueError(f"localization support must be 3-D, got {support.data.shape}")
    tolerance = float(tolerance_mm)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance_mm must be finite and nonnegative")
    mask = np.asarray(support.data > threshold)
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        raise ValueError("localization support is empty")

    source_to_anchor = validate_source_to_anchor_transform(source_to_anchor_ras)
    source_voxel_to_anchor = source_to_anchor @ support.affine_ras
    centers_anchor = nib.affines.apply_affine(source_voxel_to_anchor, coordinates)
    # For an axis-aligned target box, the exact extrema of a transformed voxel
    # parallelepiped are the centre +/- this per-anchor-axis half extent.
    half_extent_anchor = 0.5 * np.sum(np.abs(source_voxel_to_anchor[:3, :3]), axis=1)
    voxel_low = centers_anchor - half_extent_anchor[None, :]
    voxel_high = centers_anchor + half_extent_anchor[None, :]

    target_affine = grid.affine_ras
    target_low = target_affine[:3, 3] - 0.5 * np.asarray(
        grid.spacing_xyz_mm, dtype=np.float64
    )
    target_high = target_affine[:3, 3] + (
        np.asarray(grid.shape_xyz, dtype=np.float64) - 0.5
    ) * np.asarray(grid.spacing_xyz_mm, dtype=np.float64)
    retained = np.all(
        (voxel_low >= target_low[None, :] - tolerance)
        & (voxel_high <= target_high[None, :] + tolerance),
        axis=1,
    )
    full_count = int(coordinates.shape[0])
    retained_count = int(np.count_nonzero(retained))
    retention = float(retained_count / full_count)

    support_low = voxel_low.min(axis=0)
    support_high = voxel_high.max(axis=0)
    margins = np.concatenate((support_low - target_low, target_high - support_high))
    minimum_margin = float(np.min(margins))
    overlaps_target = np.all(
        (voxel_high >= target_low[None, :] - tolerance)
        & (voxel_low <= target_high[None, :] + tolerance),
        axis=1,
    )
    low_touch = np.any(
        overlaps_target[:, None]
        & (voxel_low <= target_low[None, :] + tolerance)
        & (voxel_high >= target_low[None, :] - tolerance)
    )
    high_touch = np.any(
        overlaps_target[:, None]
        & (voxel_low <= target_high[None, :] + tolerance)
        & (voxel_high >= target_high[None, :] - tolerance)
    )
    source_shape = np.asarray(support.shape_xyz, dtype=np.int64)
    source_boundary_touch = bool(
        np.any(coordinates == 0) or np.any(coordinates == (source_shape - 1)[None, :])
    )

    source_voxel_volume = float(abs(np.linalg.det(support.affine_ras[:3, :3])))
    # Registration is expected to be rigid, but use the supplied operator's
    # determinant so the physical-volume audit remains internally correct if a
    # non-rigid affine slips through upstream QC.
    anchor_volume_factor = float(abs(np.linalg.det(source_to_anchor[:3, :3])))
    voxel_volume_anchor = source_voxel_volume * anchor_volume_factor
    full_volume = full_count * voxel_volume_anchor
    retained_volume = retained_count * voxel_volume_anchor
    return PhysicalSupportAudit(
        full_positive_voxels=full_count,
        retained_positive_voxels=retained_count,
        retained_positive_voxel_fraction=retention,
        full_physical_volume_mm3=float(full_volume),
        retained_physical_volume_mm3=float(retained_volume),
        physical_volume_retention=retention,
        exact_full_support_containment=bool(retained_count == full_count),
        source_boundary_touch=source_boundary_touch,
        target_boundary_touch=bool(low_touch or high_touch),
        minimum_margin_mm=minimum_margin,
    )

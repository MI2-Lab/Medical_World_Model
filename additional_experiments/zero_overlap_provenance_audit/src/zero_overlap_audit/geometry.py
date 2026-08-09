"""Outcome-free physical geometry for the zero-overlap provenance audit.

The primitives in this module operate only on affine transforms, array shapes,
and physical coordinates.  An image affine maps *voxel centres* to world
coordinates, so every box uses the complete outer voxel footprint: voxel-axis
bounds are ``[-0.5, shape - 0.5]``, not ``[0, shape - 1]``.

The convex-intersection routine constructs the vertices of the intersection
from the twelve OBB face half-spaces and computes their convex-hull volume.
The distance routine solves the convex bounded least-squares problem over one
point in each box.  Neither routine examines image intensity, lesion, clinical,
outcome, or model-performance information.

The valid-voxel translation routines intentionally support only boxes whose
axes are cardinal with respect to the target grid.  In that case the exact
count factorises into three one-dimensional inclusive interval counts.  A
non-cardinal source fails closed rather than silently using an approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
import math
from typing import Any, Sequence

import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial import ConvexHull, QhullError


_AFFINE_ATOL = 1.0e-9
_ORTHOGONAL_ATOL = 1.0e-7
_CARDINAL_ATOL = 1.0e-7
_GEOMETRY_RTOL = 1.0e-10


__all__ = [
    "OrientedBox",
    "PairwiseMetrics",
    "TranslationRequirement",
    "aabb_intersection_volume",
    "cardinal_grid_valid_voxel_count",
    "intersection_volume",
    "minimum_cardinal_translation_for_count",
    "minimum_cardinal_translation_for_fraction",
    "minimum_distance",
    "orientation_angle_deg",
    "pairwise_metrics",
]


def _validated_shape(shape: Sequence[int], *, name: str) -> tuple[int, int, int]:
    try:
        values = tuple(shape)
    except TypeError as exc:  # pragma: no cover - defensive error normalization
        raise TypeError(f"{name} must be a length-three sequence") from exc
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three values, got {values!r}")

    result: list[int] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} values must be positive integers, not booleans")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} values must be positive integers") from exc
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"{name} values must be finite integers, got {values!r}")
        integer = int(numeric)
        if integer < 1:
            raise ValueError(f"{name} values must be positive, got {values!r}")
        result.append(integer)
    return tuple(result)  # type: ignore[return-value]


def _validated_affine(affine: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(affine, dtype=np.float64)
    if result.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(
        result[3], (0.0, 0.0, 0.0, 1.0), atol=_AFFINE_ATOL, rtol=0.0
    ):
        raise ValueError(f"{name} is not a homogeneous spatial affine")

    linear = result[:3, :3]
    spacing = np.linalg.norm(linear, axis=0)
    if np.any(spacing <= 0.0) or not np.isfinite(spacing).all():
        raise ValueError(f"{name} has a singular spatial axis")
    axes = linear / spacing[np.newaxis, :]
    if not np.allclose(
        axes.T @ axes, np.eye(3), atol=_ORTHOGONAL_ATOL, rtol=0.0
    ):
        raise ValueError(
            f"{name} contains shear or non-orthogonal axes; an OBB requires "
            "a shear-free imaging affine"
        )
    determinant = float(np.linalg.det(axes))
    if not np.isclose(abs(determinant), 1.0, atol=_ORTHOGONAL_ATOL, rtol=0.0):
        raise ValueError(f"{name} does not define a nonsingular orthonormal frame")
    return result.copy()


def _immutable_vector(value: Any, *, name: str, positive: bool = False) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    if positive and np.any(array <= 0.0):
        raise ValueError(f"{name} values must be strictly positive")
    result = array.copy()
    result.setflags(write=False)
    return result


def _immutable_axes(value: Any) -> np.ndarray:
    axes = np.asarray(value, dtype=np.float64)
    if axes.shape != (3, 3):
        raise ValueError(f"axes must have shape (3, 3), got {axes.shape}")
    if not np.isfinite(axes).all():
        raise ValueError("axes contain non-finite values")
    if not np.allclose(
        axes.T @ axes, np.eye(3), atol=_ORTHOGONAL_ATOL, rtol=0.0
    ):
        raise ValueError("axes must be an orthonormal frame")
    if not np.isclose(
        abs(float(np.linalg.det(axes))),
        1.0,
        atol=_ORTHOGONAL_ATOL,
        rtol=0.0,
    ):
        raise ValueError("axes must be nonsingular")
    result = axes.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class OrientedBox:
    """A three-dimensional oriented box in physical coordinates.

    ``axes[:, i]`` is the unit direction for ``half_lengths[i]``.  Axis signs
    and handedness are retained from the affine; they do not affect the box's
    physical point set.
    """

    center: np.ndarray
    axes: np.ndarray
    half_lengths: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _immutable_vector(self.center, name="center"))
        object.__setattr__(self, "axes", _immutable_axes(self.axes))
        object.__setattr__(
            self,
            "half_lengths",
            _immutable_vector(self.half_lengths, name="half_lengths", positive=True),
        )

    @classmethod
    def from_affine_shape(
        cls,
        affine: np.ndarray,
        shape: Sequence[int],
    ) -> "OrientedBox":
        """Build an OBB from a voxel-centre affine and ``[X, Y, Z]`` shape."""

        spatial_affine = _validated_affine(affine, name="affine")
        shape_xyz = np.asarray(
            _validated_shape(shape, name="shape"), dtype=np.float64
        )
        linear = spatial_affine[:3, :3]
        spacing = np.linalg.norm(linear, axis=0)
        axes = linear / spacing[np.newaxis, :]
        center_voxel = 0.5 * (shape_xyz - 1.0)
        center = linear @ center_voxel + spatial_affine[:3, 3]
        # A length-n voxel axis occupies n complete voxel widths.
        half_lengths = 0.5 * shape_xyz * spacing
        return cls(center=center, axes=axes, half_lengths=half_lengths)

    @property
    def fov_lengths(self) -> np.ndarray:
        """Full physical field-of-view lengths along the three box axes."""

        result = 2.0 * self.half_lengths
        result.setflags(write=False)
        return result

    @property
    def volume(self) -> float:
        """Physical box volume in cubic world units (normally mm^3)."""

        return float(8.0 * np.prod(self.half_lengths, dtype=np.float64))

    def corners(self) -> np.ndarray:
        """Return the eight outer-footprint corners as an ``[8, 3]`` array."""

        signs = np.asarray(tuple(product((-1.0, 1.0), repeat=3)), dtype=np.float64)
        return self.center[np.newaxis, :] + (
            signs * self.half_lengths[np.newaxis, :]
        ) @ self.axes.T

    @property
    def aabb_min(self) -> np.ndarray:
        """Minimum world coordinate of the enclosing axis-aligned box."""

        extent = np.abs(self.axes) @ self.half_lengths
        result = self.center - extent
        result.setflags(write=False)
        return result

    @property
    def aabb_max(self) -> np.ndarray:
        """Maximum world coordinate of the enclosing axis-aligned box."""

        extent = np.abs(self.axes) @ self.half_lengths
        result = self.center + extent
        result.setflags(write=False)
        return result

    @property
    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(minimum, maximum)`` of the enclosing AABB."""

        return self.aabb_min, self.aabb_max

    def translated(self, translation: Sequence[float]) -> "OrientedBox":
        """Return a copy translated in physical space.

        This generic primitive is useful for synthetic verification.  Audit
        reporting deliberately omits the private translation direction.
        """

        offset = _immutable_vector(translation, name="translation")
        return OrientedBox(
            center=self.center + offset,
            axes=self.axes,
            half_lengths=self.half_lengths,
        )

    def halfspaces(self) -> tuple[np.ndarray, np.ndarray]:
        """Return unit-normal inequalities ``normals @ x <= offsets``."""

        normals: list[np.ndarray] = []
        offsets: list[float] = []
        for index in range(3):
            axis = self.axes[:, index]
            half_length = float(self.half_lengths[index])
            normals.extend((axis, -axis))
            offsets.extend(
                (
                    float(np.dot(axis, self.center) + half_length),
                    float(np.dot(-axis, self.center) + half_length),
                )
            )
        return np.asarray(normals), np.asarray(offsets)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible geometry representation."""

        return {
            "center": self.center.tolist(),
            "axes": self.axes.tolist(),
            "half_lengths": self.half_lengths.tolist(),
            "fov_lengths": self.fov_lengths.tolist(),
            "corners": self.corners().tolist(),
            "aabb_min": self.aabb_min.tolist(),
            "aabb_max": self.aabb_max.tolist(),
            "volume": self.volume,
        }


def _require_box(box: OrientedBox, *, name: str) -> OrientedBox:
    if not isinstance(box, OrientedBox):
        raise TypeError(f"{name} must be an OrientedBox")
    return box


def aabb_intersection_volume(a: OrientedBox, b: OrientedBox) -> float:
    """Return the intersection volume of the boxes' enclosing AABBs."""

    box_a = _require_box(a, name="a")
    box_b = _require_box(b, name="b")
    lower = np.maximum(box_a.aabb_min, box_b.aabb_min)
    upper = np.minimum(box_a.aabb_max, box_b.aabb_max)
    lengths = np.maximum(upper - lower, 0.0)
    return float(np.prod(lengths, dtype=np.float64))


def _intersection_vertices(a: OrientedBox, b: OrientedBox) -> np.ndarray:
    normals_a, offsets_a = a.halfspaces()
    normals_b, offsets_b = b.halfspaces()
    normals = np.concatenate((normals_a, normals_b), axis=0)
    offsets = np.concatenate((offsets_a, offsets_b), axis=0)

    scale = max(
        1.0,
        float(np.max(np.abs(a.center))),
        float(np.max(np.abs(b.center))),
        float(np.max(a.half_lengths)),
        float(np.max(b.half_lengths)),
    )
    feasibility_tolerance = _GEOMETRY_RTOL * scale
    duplicate_tolerance = _GEOMETRY_RTOL * scale
    determinant_tolerance = 64.0 * np.finfo(np.float64).eps

    vertices: list[np.ndarray] = []
    # Every vertex of a bounded 3-D half-space intersection has at least three
    # active faces.  Enumerating all plane triples therefore finds every one,
    # including vertices inherited unchanged from either input OBB.
    for indices in combinations(range(normals.shape[0]), 3):
        matrix = normals[np.asarray(indices)]
        if abs(float(np.linalg.det(matrix))) <= determinant_tolerance:
            continue
        point = np.linalg.solve(matrix, offsets[np.asarray(indices)])
        if np.max(normals @ point - offsets) > feasibility_tolerance:
            continue
        if any(
            np.linalg.norm(point - existing) <= duplicate_tolerance
            for existing in vertices
        ):
            continue
        vertices.append(point)

    if not vertices:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(vertices, dtype=np.float64)


def intersection_volume(a: OrientedBox, b: OrientedBox) -> float:
    """Return the exact convex OBB intersection volume via face planes.

    "Exact" here describes the polyhedral construction (there is no voxel
    rasterisation or Monte Carlo approximation); ordinary float64 numerical
    tolerance applies to plane solves and the convex hull.
    """

    box_a = _require_box(a, name="a")
    box_b = _require_box(b, name="b")
    # A strict AABB separation is a cheap exact rejection.  Touching AABBs
    # have zero volume and likewise cannot have positive OBB volume.
    lower = np.maximum(box_a.aabb_min, box_b.aabb_min)
    upper = np.minimum(box_a.aabb_max, box_b.aabb_max)
    if np.any(upper <= lower):
        return 0.0

    vertices = _intersection_vertices(box_a, box_b)
    if vertices.shape[0] < 4:
        return 0.0
    try:
        hull = ConvexHull(vertices)
    except QhullError:
        # A point/line/plane contact has no 3-D measure.
        return 0.0
    volume = float(hull.volume)
    if not np.isfinite(volume) or volume < 0.0:  # pragma: no cover - SciPy guard
        raise RuntimeError("convex hull produced an invalid intersection volume")
    # Clamp harmless hull roundoff for complete containment/identity.
    upper_bound = min(box_a.volume, box_b.volume)
    if volume > upper_bound and np.isclose(
        volume, upper_bound, rtol=1.0e-9, atol=1.0e-10
    ):
        volume = upper_bound
    if volume > upper_bound * (1.0 + 1.0e-8):  # pragma: no cover - invariant guard
        raise RuntimeError("computed intersection exceeds an input box volume")
    return volume


def minimum_distance(a: OrientedBox, b: OrientedBox) -> float:
    """Return the minimum Euclidean distance between two closed OBBs.

    A point in box A is ``c_a + axes_a @ u`` with bounded coordinates ``u``;
    box B is analogous.  Minimising the point difference is therefore one
    bounded linear least-squares problem in six variables.
    """

    box_a = _require_box(a, name="a")
    box_b = _require_box(b, name="b")
    matrix = np.concatenate((box_a.axes, -box_b.axes), axis=1)
    right_hand_side = box_b.center - box_a.center
    lower = -np.concatenate((box_a.half_lengths, box_b.half_lengths))
    upper = np.concatenate((box_a.half_lengths, box_b.half_lengths))
    solution = lsq_linear(
        matrix,
        right_hand_side,
        bounds=(lower, upper),
        method="trf",
        lsq_solver="exact",
        tol=1.0e-13,
        max_iter=1000,
        verbose=0,
    )
    if not solution.success or not np.isfinite(solution.x).all():
        raise RuntimeError(
            "bounded least-squares OBB distance did not converge: "
            f"{solution.message}"
        )
    distance = float(np.linalg.norm(matrix @ solution.x - right_hand_side))
    scale = max(
        1.0,
        float(np.linalg.norm(right_hand_side)),
        float(np.max(box_a.half_lengths)),
        float(np.max(box_b.half_lengths)),
    )
    if distance <= 1.0e-10 * scale:
        return 0.0
    return distance


def _right_handed_axes(axes: np.ndarray) -> np.ndarray:
    result = np.asarray(axes, dtype=np.float64).copy()
    if np.linalg.det(result) < 0.0:
        # An OBB axis has no intrinsic sign, so this handedness correction does
        # not change the represented physical box.
        result[:, 2] *= -1.0
    return result


def orientation_angle_deg(a: OrientedBox, b: OrientedBox) -> float:
    """Return the minimum labeled-axis OBB orientation difference in degrees.

    Axis signs are geometrically immaterial and are minimised over; axis labels
    are retained (no permutation).  Thus a 180-degree sign flip of an otherwise
    identical box is zero degrees, while a genuine 30-degree rotation is 30.
    """

    box_a = _require_box(a, name="a")
    box_b = _require_box(b, name="b")
    axes_a = _right_handed_axes(box_a.axes)
    axes_b = _right_handed_axes(box_b.axes)
    sign_options = (
        (1.0, 1.0, 1.0),
        (1.0, -1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
    )
    angles: list[float] = []
    for signs in sign_options:
        candidate = axes_b @ np.diag(signs)
        relative = axes_a.T @ candidate
        cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
        # atan2 is stable close to zero, where acos(trace) magnifies ordinary
        # orthonormal-frame roundoff into a spurious microradian rotation.
        sine = float(
            np.linalg.norm(relative - relative.T, ord="fro")
            / (2.0 * math.sqrt(2.0))
        )
        angles.append(math.degrees(math.atan2(sine, cosine)))
    angle = min(angles)
    if angle <= 1.0e-10:
        return 0.0
    return float(angle)


@dataclass(frozen=True)
class PairwiseMetrics:
    """Stable, JSON-friendly physical metrics for an ordered OBB pair."""

    center_displacement_mm: float
    orientation_angle_deg: float
    minimum_separation_mm: float
    intersection_mm3: float
    overlap_fraction_a: float
    overlap_fraction_b: float
    iou: float
    aabb_intersection_mm3: float
    aabb_intersects: bool

    @property
    def intersection_volume_mm3(self) -> float:
        """Backward-readable alias for ``intersection_mm3``."""

        return self.intersection_mm3

    @property
    def oriented_intersection_mm3(self) -> float:
        """Report-table alias distinguishing this from the enclosing AABB."""

        return self.intersection_mm3

    @property
    def overlap_fraction_first(self) -> float:
        """Ordered-pair alias for ``overlap_fraction_a``."""

        return self.overlap_fraction_a

    @property
    def overlap_fraction_second(self) -> float:
        """Ordered-pair alias for ``overlap_fraction_b``."""

        return self.overlap_fraction_b

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "center_displacement_mm": float(self.center_displacement_mm),
            "orientation_angle_deg": float(self.orientation_angle_deg),
            "minimum_separation_mm": float(self.minimum_separation_mm),
            "intersection_mm3": float(self.intersection_mm3),
            "oriented_intersection_mm3": float(self.intersection_mm3),
            "overlap_fraction_a": float(self.overlap_fraction_a),
            "overlap_fraction_b": float(self.overlap_fraction_b),
            "overlap_fraction_first": float(self.overlap_fraction_a),
            "overlap_fraction_second": float(self.overlap_fraction_b),
            "iou": float(self.iou),
            "aabb_intersection_mm3": float(self.aabb_intersection_mm3),
            "aabb_intersects": bool(self.aabb_intersects),
        }


def pairwise_metrics(a: OrientedBox, b: OrientedBox) -> PairwiseMetrics:
    """Compute the complete physical comparison for an ordered box pair."""

    box_a = _require_box(a, name="a")
    box_b = _require_box(b, name="b")
    intersection = intersection_volume(box_a, box_b)
    union = box_a.volume + box_b.volume - intersection
    aabb_volume = aabb_intersection_volume(box_a, box_b)
    return PairwiseMetrics(
        center_displacement_mm=float(np.linalg.norm(box_b.center - box_a.center)),
        orientation_angle_deg=orientation_angle_deg(box_a, box_b),
        minimum_separation_mm=minimum_distance(box_a, box_b),
        intersection_mm3=intersection,
        overlap_fraction_a=intersection / box_a.volume,
        overlap_fraction_b=intersection / box_b.volume,
        iou=intersection / union,
        aabb_intersection_mm3=aabb_volume,
        aabb_intersects=bool(
            np.all(
                np.minimum(box_a.aabb_max, box_b.aabb_max)
                >= np.maximum(box_a.aabb_min, box_b.aabb_min)
            )
        ),
    )


@dataclass(frozen=True)
class TranslationRequirement:
    """Minimum cardinal translation magnitude for a target valid-voxel gate.

    The direction is deliberately absent: the audit exposes magnitude only and
    must never turn a diagnostic translation into a repair transform.
    ``None`` translation/achievement values mean that the requested threshold
    is mathematically unattainable for the two finite FOVs.
    """

    required_valid_voxels: int
    required_valid_fraction: float
    current_valid_voxels: int
    current_valid_fraction: float
    translation_magnitude_mm: float | None
    achieved_valid_voxels: int | None
    achieved_valid_fraction: float | None
    maximum_attainable_valid_voxels: int
    maximum_attainable_valid_fraction: float
    attainable: bool

    @property
    def required_count(self) -> int:
        return self.required_valid_voxels

    @property
    def maximum_attainable_count(self) -> int:
        return self.maximum_attainable_valid_voxels

    def to_dict(self) -> dict[str, int | float | bool | None]:
        return {
            "required_valid_voxels": int(self.required_valid_voxels),
            "required_valid_fraction": float(self.required_valid_fraction),
            "current_valid_voxels": int(self.current_valid_voxels),
            "current_valid_fraction": float(self.current_valid_fraction),
            "translation_magnitude_mm": (
                None
                if self.translation_magnitude_mm is None
                else float(self.translation_magnitude_mm)
            ),
            "achieved_valid_voxels": (
                None
                if self.achieved_valid_voxels is None
                else int(self.achieved_valid_voxels)
            ),
            "achieved_valid_fraction": (
                None
                if self.achieved_valid_fraction is None
                else float(self.achieved_valid_fraction)
            ),
            "maximum_attainable_valid_voxels": int(
                self.maximum_attainable_valid_voxels
            ),
            "maximum_attainable_valid_fraction": float(
                self.maximum_attainable_valid_fraction
            ),
            "attainable": bool(self.attainable),
        }


@dataclass(frozen=True)
class _CardinalGeometry:
    target_positions: tuple[np.ndarray, np.ndarray, np.ndarray]
    source_lower: np.ndarray
    source_upper: np.ndarray
    target_shape: tuple[int, int, int]

    @property
    def total_target_voxels(self) -> int:
        return math.prod(self.target_shape)


def _cardinal_geometry(
    target_affine: np.ndarray,
    target_shape: Sequence[int],
    source_box: OrientedBox,
) -> _CardinalGeometry:
    shape = _validated_shape(target_shape, name="target_shape")
    affine = _validated_affine(target_affine, name="target_affine")
    source = _require_box(source_box, name="source_box")

    linear = affine[:3, :3]
    spacing = np.linalg.norm(linear, axis=0)
    target_axes = linear / spacing[np.newaxis, :]
    relative = target_axes.T @ source.axes
    absolute = np.abs(relative)
    source_to_target_axis = np.argmax(absolute, axis=0)
    expected = np.zeros((3, 3), dtype=np.float64)
    for source_axis, target_axis in enumerate(source_to_target_axis):
        expected[int(target_axis), source_axis] = 1.0
    if (
        len(set(int(value) for value in source_to_target_axis)) != 3
        or not np.allclose(absolute, expected, atol=_CARDINAL_ATOL, rtol=0.0)
    ):
        raise ValueError(
            "source_box axes are non-cardinal relative to the target grid; "
            "exact valid-voxel translation is undefined for this routine"
        )

    # Work in the orthonormal target-axis frame.  This supports target grids
    # that are rigidly rotated in world space while preserving exact cardinal
    # source/target alignment and Euclidean translation magnitude.
    target_origin_local = target_axes.T @ affine[:3, 3]
    positions = tuple(
        target_origin_local[axis]
        + spacing[axis] * np.arange(shape[axis], dtype=np.float64)
        for axis in range(3)
    )
    source_center_local = target_axes.T @ source.center
    source_half_local = absolute @ source.half_lengths
    return _CardinalGeometry(
        target_positions=positions,  # type: ignore[arg-type]
        source_lower=source_center_local - source_half_local,
        source_upper=source_center_local + source_half_local,
        target_shape=shape,
    )


def _axis_count(
    positions: np.ndarray,
    lower: float,
    upper: float,
    shift: float = 0.0,
) -> int:
    scale = max(
        1.0,
        float(np.max(np.abs(positions))),
        abs(lower),
        abs(upper),
        abs(shift),
    )
    tolerance = _GEOMETRY_RTOL * scale
    return int(
        np.count_nonzero(
            (positions >= lower + shift - tolerance)
            & (positions <= upper + shift + tolerance)
        )
    )


def cardinal_grid_valid_voxel_count(
    target_affine: np.ndarray,
    target_shape: Sequence[int],
    source_box: OrientedBox,
) -> int:
    """Count target voxel centres inside a cardinal source footprint exactly."""

    geometry = _cardinal_geometry(target_affine, target_shape, source_box)
    counts = (
        _axis_count(
            geometry.target_positions[axis],
            float(geometry.source_lower[axis]),
            float(geometry.source_upper[axis]),
        )
        for axis in range(3)
    )
    return math.prod(counts)


@dataclass(frozen=True)
class _AxisCoverage:
    minimum_magnitude: np.ndarray
    signed_shift: np.ndarray
    maximum_count: int


def _axis_coverage_profile(
    positions: np.ndarray,
    source_lower: float,
    source_upper: float,
) -> _AxisCoverage:
    target_count = int(positions.size)
    magnitudes = np.full(target_count + 1, np.inf, dtype=np.float64)
    shifts = np.full(target_count + 1, np.nan, dtype=np.float64)
    magnitudes[0] = 0.0
    shifts[0] = 0.0
    scale = max(
        1.0,
        float(np.max(np.abs(positions))),
        abs(source_lower),
        abs(source_upper),
    )
    tolerance = _GEOMETRY_RTOL * scale

    for requested in range(1, target_count + 1):
        best_key: tuple[float, float] | None = None
        best_shift = math.nan
        # If any requested points fit, a consecutive requested-point window
        # fits.  For window [first,last], source translation t must satisfy
        # last-upper <= t <= first-lower.
        for start in range(0, target_count - requested + 1):
            stop = start + requested - 1
            feasible_lower = float(positions[stop] - source_upper)
            feasible_upper = float(positions[start] - source_lower)
            if feasible_lower > feasible_upper + tolerance:
                continue
            if feasible_lower > feasible_upper:
                shift = 0.5 * (feasible_lower + feasible_upper)
            elif feasible_lower > 0.0:
                shift = feasible_lower
            elif feasible_upper < 0.0:
                shift = feasible_upper
            else:
                shift = 0.0
            # Deterministic tie-break only; direction is never published.
            key = (abs(shift), shift)
            if best_key is None or key < best_key:
                best_key = key
                best_shift = shift
        if best_key is not None:
            magnitudes[requested] = best_key[0]
            shifts[requested] = best_shift

    attainable = np.flatnonzero(np.isfinite(magnitudes))
    maximum = int(attainable[-1])
    return _AxisCoverage(
        minimum_magnitude=magnitudes,
        signed_shift=shifts,
        maximum_count=maximum,
    )


def _validated_required_count(required_count: int) -> int:
    if isinstance(required_count, (bool, np.bool_)):
        raise TypeError("required_count must be a nonnegative integer")
    try:
        numeric = float(required_count)
    except (TypeError, ValueError) as exc:
        raise TypeError("required_count must be a nonnegative integer") from exc
    if not np.isfinite(numeric) or not numeric.is_integer() or numeric < 0.0:
        raise ValueError("required_count must be a finite nonnegative integer")
    return int(numeric)


def minimum_cardinal_translation_for_count(
    target_affine: np.ndarray,
    target_shape: Sequence[int],
    source_box: OrientedBox,
    required_count: int,
) -> TranslationRequirement:
    """Find the exact minimum translation magnitude reaching ``required_count``.

    The translated object is the source box.  Translating the target instead
    gives the opposite vector and the same reported magnitude.
    """

    required = _validated_required_count(required_count)
    geometry = _cardinal_geometry(target_affine, target_shape, source_box)
    total = geometry.total_target_voxels
    profiles = tuple(
        _axis_coverage_profile(
            geometry.target_positions[axis],
            float(geometry.source_lower[axis]),
            float(geometry.source_upper[axis]),
        )
        for axis in range(3)
    )
    current_axis_counts = tuple(
        _axis_count(
            geometry.target_positions[axis],
            float(geometry.source_lower[axis]),
            float(geometry.source_upper[axis]),
        )
        for axis in range(3)
    )
    current = math.prod(current_axis_counts)
    maximum = math.prod(profile.maximum_count for profile in profiles)

    common = {
        "required_valid_voxels": required,
        "required_valid_fraction": required / total,
        "current_valid_voxels": current,
        "current_valid_fraction": current / total,
        "maximum_attainable_valid_voxels": maximum,
        "maximum_attainable_valid_fraction": maximum / total,
    }
    if required > maximum:
        return TranslationRequirement(
            **common,
            translation_magnitude_mm=None,
            achieved_valid_voxels=None,
            achieved_valid_fraction=None,
            attainable=False,
        )
    if required == 0:
        return TranslationRequirement(
            **common,
            translation_magnitude_mm=0.0,
            achieved_valid_voxels=current,
            achieved_valid_fraction=current / total,
            attainable=True,
        )

    best_key: tuple[float, int, int, int, int] | None = None
    best_counts: tuple[int, int, int] | None = None
    max_x, max_y, max_z = (profile.maximum_count for profile in profiles)
    for count_x in range(1, max_x + 1):
        distance_x = float(profiles[0].minimum_magnitude[count_x])
        for count_y in range(1, max_y + 1):
            xy = count_x * count_y
            count_z = (required + xy - 1) // xy
            if count_z < 1 or count_z > max_z:
                continue
            squared_distance = (
                distance_x * distance_x
                + float(profiles[1].minimum_magnitude[count_y]) ** 2
                + float(profiles[2].minimum_magnitude[count_z]) ** 2
            )
            represented_count = xy * count_z
            key = (
                squared_distance,
                represented_count,
                count_x,
                count_y,
                count_z,
            )
            if best_key is None or key < best_key:
                best_key = key
                best_counts = (count_x, count_y, count_z)

    if best_key is None or best_counts is None:  # pragma: no cover - invariant guard
        raise RuntimeError("attainable cardinal threshold had no translation solution")
    shifts = np.asarray(
        [profiles[axis].signed_shift[best_counts[axis]] for axis in range(3)],
        dtype=np.float64,
    )
    achieved_axis_counts = tuple(
        _axis_count(
            geometry.target_positions[axis],
            float(geometry.source_lower[axis]),
            float(geometry.source_upper[axis]),
            float(shifts[axis]),
        )
        for axis in range(3)
    )
    achieved = math.prod(achieved_axis_counts)
    if achieved < required:  # pragma: no cover - numerical invariant guard
        raise RuntimeError(
            "computed cardinal translation did not achieve the requested count"
        )
    magnitude = float(np.linalg.norm(shifts))
    if magnitude <= 1.0e-10:
        magnitude = 0.0
    return TranslationRequirement(
        **common,
        translation_magnitude_mm=magnitude,
        achieved_valid_voxels=achieved,
        achieved_valid_fraction=achieved / total,
        attainable=True,
    )


def minimum_cardinal_translation_for_fraction(
    target_affine: np.ndarray,
    target_shape: Sequence[int],
    source_box: OrientedBox,
    required_fraction: float,
) -> TranslationRequirement:
    """Threshold wrapper using ``ceil(fraction * target_voxel_count)``."""

    fraction = float(required_fraction)
    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("required_fraction must be finite and in [0, 1]")
    shape = _validated_shape(target_shape, name="target_shape")
    raw_count = fraction * math.prod(shape)
    nearest_integer = round(raw_count)
    if math.isclose(raw_count, nearest_integer, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raw_count = float(nearest_integer)
    required_count = int(math.ceil(raw_count))
    return minimum_cardinal_translation_for_count(
        target_affine,
        shape,
        source_box,
        required_count,
    )

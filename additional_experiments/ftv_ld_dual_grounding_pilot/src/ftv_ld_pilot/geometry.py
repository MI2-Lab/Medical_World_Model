"""Deterministic raw-grid geometry for the Stage A containment audit.

Coordinate contracts are intentionally explicit:

* input masks and crop starts use raw ``(X, Y, Z)`` order;
* crop sizes and returned cropped masks use model ``(Z, Y, X)`` order;
* no function in this module resizes, resamples, or reorients spatial data.

Bounding-box maxima are inclusive.  A signed margin of zero therefore means
that full support touches the corresponding requested crop face; a negative
margin means that full support extends past that face.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Iterable

import numpy as np
from scipy import ndimage


_FACE_NAMES = (
    "x_low",
    "x_high",
    "y_low",
    "y_high",
    "z_low",
    "z_high",
)

# Thirteen unoriented axes spanning axial, face-diagonal, and body-diagonal
# directions.  Their negatives are implicit because both projection extrema
# are retained.  Directions live in physical XYZ space.
_FERET_DIRECTIONS_XYZ = np.asarray(
    [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (1, -1, 0),
        (1, 0, 1),
        (1, 0, -1),
        (0, 1, 1),
        (0, 1, -1),
        (1, 1, 1),
        (1, 1, -1),
        (1, -1, 1),
        (1, -1, -1),
    ],
    dtype=np.float64,
)
_FERET_DIRECTIONS_XYZ /= np.linalg.norm(_FERET_DIRECTIONS_XYZ, axis=1, keepdims=True)


def _mask_xyz(mask_xyz: Any) -> np.ndarray:
    """Return a three-dimensional boolean XYZ mask without copying if possible."""

    mask = np.asarray(mask_xyz)
    if mask.ndim != 3:
        raise ValueError(f"mask_xyz must be 3-D XYZ; got shape {mask.shape}")
    return mask.astype(bool, copy=False)


def _integer_triplet(
    values: Iterable[Any], name: str, *, positive: bool
) -> tuple[int, int, int]:
    """Validate and normalize an integer-valued length-three coordinate."""

    array = np.asarray(values)
    if array.shape != (3,):
        raise ValueError(
            f"{name} must contain exactly three values; got shape {array.shape}"
        )
    try:
        numeric = array.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} must contain only finite values")
    rounded = np.rint(numeric)
    if not np.array_equal(numeric, rounded):
        raise ValueError(f"{name} must be integer-valued; got {numeric.tolist()}")
    result = tuple(int(value) for value in rounded)
    if positive and any(value <= 0 for value in result):
        raise ValueError(f"{name} must be strictly positive; got {result}")
    return result


def _spacing_triplet(spacing_xyz: Iterable[Any]) -> tuple[float, float, float]:
    """Validate positive, finite XYZ voxel spacing."""

    array = np.asarray(spacing_xyz, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(
            f"spacing_xyz must contain exactly three values; got shape {array.shape}"
        )
    if not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise ValueError(
            f"spacing_xyz must be finite and positive; got {array.tolist()}"
        )
    return tuple(float(value) for value in array)


def crop_or_pad_from_start(
    mask_xyz: Any,
    start_xyz: Iterable[Any],
    crop_size_zyx: Iterable[Any],
) -> np.ndarray:
    """Crop a raw XYZ mask and return a zero-padded boolean ZYX mask.

    ``start_xyz`` is the requested inclusive lower corner in source-grid voxel
    indices and may lie outside the source volume.  The output always has shape
    ``crop_size_zyx``.  Spatial samples are copied directly; there is no resize
    or interpolation.
    """

    mask = _mask_xyz(mask_xyz)
    start = np.asarray(
        _integer_triplet(start_xyz, "start_xyz", positive=False), dtype=np.int64
    )
    crop_zyx = _integer_triplet(crop_size_zyx, "crop_size_zyx", positive=True)
    size_xyz = np.asarray((crop_zyx[2], crop_zyx[1], crop_zyx[0]), dtype=np.int64)
    shape_xyz = np.asarray(mask.shape, dtype=np.int64)

    crop_xyz = np.zeros(tuple(int(value) for value in size_xyz), dtype=bool)
    source_low = np.maximum(start, 0)
    source_high = np.minimum(start + size_xyz, shape_xyz)

    if np.all(source_high > source_low):
        destination_low = source_low - start
        destination_high = destination_low + (source_high - source_low)
        source_slices = tuple(
            slice(int(low), int(high)) for low, high in zip(source_low, source_high)
        )
        destination_slices = tuple(
            slice(int(low), int(high))
            for low, high in zip(destination_low, destination_high)
        )
        crop_xyz[destination_slices] = mask[source_slices]

    # Raw source arrays are XYZ; model/cache arrays are ZYX.
    return np.transpose(crop_xyz, (2, 1, 0))


def recover_origin(
    mask_xyz: Any,
    actual_roi_zyx: Any,
    clean_start_xyz: Iterable[Any],
    crop_size_zyx: Iterable[Any],
    radius: int = 2,
) -> dict[str, Any]:
    """Recover a legacy crop start through exact binary-mask reconstruction.

    Candidate starts span the inclusive XYZ cube ``[-radius, +radius]`` around
    ``clean_start_xyz``.  Exact matches are selected by squared Euclidean
    distance to the clean start and then by lexicographic XYZ order.  Empty
    cached support is deliberately non-identifying and returns ``no_match``.

    Returns a dictionary with ``status`` (``unique``, ``multiple``, or
    ``no_match``), ``chosen_start``, ``candidate_count``, and ``method``.
    """

    mask = _mask_xyz(mask_xyz)
    clean_start = _integer_triplet(clean_start_xyz, "clean_start_xyz", positive=False)
    crop_zyx = _integer_triplet(crop_size_zyx, "crop_size_zyx", positive=True)

    if isinstance(radius, (bool, np.bool_)) or not isinstance(
        radius, (int, np.integer)
    ):
        raise TypeError("radius must be a non-negative integer")
    radius = int(radius)
    if radius < 0:
        raise ValueError("radius must be a non-negative integer")

    actual = np.asarray(actual_roi_zyx)
    if actual.ndim != 3:
        raise ValueError(f"actual_roi_zyx must be 3-D ZYX; got shape {actual.shape}")
    if actual.shape != crop_zyx:
        raise ValueError(
            "actual_roi_zyx shape must equal crop_size_zyx; "
            f"got {actual.shape} versus {crop_zyx}"
        )
    actual = actual.astype(bool, copy=False)

    if not np.any(actual):
        return {
            "status": "no_match",
            "chosen_start": None,
            "candidate_count": 0,
            "method": "empty_actual_roi_nonidentifying",
        }
    if not np.any(mask):
        return {
            "status": "no_match",
            "chosen_start": None,
            "candidate_count": 0,
            "method": "empty_full_mask_no_match",
        }

    # Comparing a full 32x96x96 reconstruction for every one of the 125
    # candidates is needlessly expensive on the complete audit.  Exactness can
    # be established without interpolation or hashing: every cached-positive
    # coordinate must map to a positive source voxel, and the number of source
    # positives inside the requested crop must equal the cached-positive count.
    # The two set-inclusion/count conditions are equivalent to bitwise equality.
    actual_zyx = np.argwhere(actual)
    actual_xyz_offsets = actual_zyx[:, (2, 1, 0)].astype(np.int64, copy=False)
    signature_indices = np.linspace(
        0,
        len(actual_xyz_offsets) - 1,
        num=min(64, len(actual_xyz_offsets)),
        dtype=np.int64,
    )
    signature_offsets = actual_xyz_offsets[np.unique(signature_indices)]
    shape_xyz = np.asarray(mask.shape, dtype=np.int64)
    size_xyz = np.asarray((crop_zyx[2], crop_zyx[1], crop_zyx[0]), dtype=np.int64)

    matches: list[tuple[int, int, int]] = []
    for offset in product(range(-radius, radius + 1), repeat=3):
        candidate = tuple(clean_start[axis] + offset[axis] for axis in range(3))
        candidate_array = np.asarray(candidate, dtype=np.int64)
        signature_source = signature_offsets + candidate_array
        if np.any(signature_source < 0) or np.any(signature_source >= shape_xyz):
            continue
        if not np.all(mask[tuple(signature_source.T)]):
            continue

        actual_source = actual_xyz_offsets + candidate_array
        if np.any(actual_source < 0) or np.any(actual_source >= shape_xyz):
            continue
        if not np.all(mask[tuple(actual_source.T)]):
            continue

        source_low = np.maximum(candidate_array, 0)
        source_high = np.minimum(candidate_array + size_xyz, shape_xyz)
        if np.any(source_high <= source_low):
            continue
        source_slices = tuple(
            slice(int(low), int(high))
            for low, high in zip(source_low, source_high, strict=True)
        )
        if int(np.count_nonzero(mask[source_slices])) == len(actual_xyz_offsets):
            matches.append(candidate)

    if not matches:
        return {
            "status": "no_match",
            "chosen_start": None,
            "candidate_count": 0,
            "method": "exact_mask_no_match",
        }

    def selection_key(
        candidate: tuple[int, int, int]
    ) -> tuple[int, tuple[int, int, int]]:
        distance_squared = sum(
            (candidate[axis] - clean_start[axis]) ** 2 for axis in range(3)
        )
        return distance_squared, candidate

    chosen = min(matches, key=selection_key)
    status = "unique" if len(matches) == 1 else "multiple"
    method = (
        "exact_mask_unique"
        if status == "unique"
        else "exact_mask_multiple_nearest_then_lexicographic"
    )
    return {
        "status": status,
        "chosen_start": [int(value) for value in chosen],
        "candidate_count": len(matches),
        "method": method,
    }


def bbox_xyz(mask_xyz: Any) -> dict[str, Any] | None:
    """Return the inclusive raw-XYZ support bounding box, or ``None`` if empty."""

    mask = _mask_xyz(mask_xyz)
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        return None

    minimum = coordinates.min(axis=0).astype(np.int64)
    maximum = coordinates.max(axis=0).astype(np.int64)
    extent = maximum - minimum + 1
    result: dict[str, Any] = {
        "x_min": int(minimum[0]),
        "x_max": int(maximum[0]),
        "y_min": int(minimum[1]),
        "y_max": int(maximum[1]),
        "z_min": int(minimum[2]),
        "z_max": int(maximum[2]),
        "min_xyz": [int(value) for value in minimum],
        "max_xyz": [int(value) for value in maximum],
        "end_exclusive_xyz": [int(value) for value in maximum + 1],
        "extent_xyz_voxel": [int(value) for value in extent],
    }
    return result


def _boundary_touch(cropped_zyx: np.ndarray) -> dict[str, bool]:
    """Return six crop-face touch flags for a boolean ZYX crop."""

    return {
        "x_low": bool(np.any(cropped_zyx[:, :, 0])),
        "x_high": bool(np.any(cropped_zyx[:, :, -1])),
        "y_low": bool(np.any(cropped_zyx[:, 0, :])),
        "y_high": bool(np.any(cropped_zyx[:, -1, :])),
        "z_low": bool(np.any(cropped_zyx[0, :, :])),
        "z_high": bool(np.any(cropped_zyx[-1, :, :])),
    }


def geometry_metrics(
    mask_xyz: Any,
    start_xyz: Iterable[Any],
    spacing_xyz: Iterable[Any],
    crop_size_zyx: Iterable[Any],
) -> dict[str, Any]:
    """Compute deterministic full-support versus requested-crop geometry.

    Physical bbox extents use voxel footprints (voxel count times spacing), as
    does crop physical extent.  The approximate Feret routine separately uses
    voxel-center distances.  Signed margins use inclusive bbox and crop faces.
    """

    mask = _mask_xyz(mask_xyz)
    start = np.asarray(
        _integer_triplet(start_xyz, "start_xyz", positive=False), dtype=np.int64
    )
    spacing = np.asarray(_spacing_triplet(spacing_xyz), dtype=np.float64)
    crop_zyx = _integer_triplet(crop_size_zyx, "crop_size_zyx", positive=True)
    size_xyz = np.asarray((crop_zyx[2], crop_zyx[1], crop_zyx[0]), dtype=np.int64)
    shape_xyz = np.asarray(mask.shape, dtype=np.int64)
    end_exclusive = start + size_xyz
    end_inclusive = end_exclusive - 1

    cropped_zyx = crop_or_pad_from_start(mask, start, crop_zyx)
    full_voxels = int(np.count_nonzero(mask))
    contained_voxels = int(np.count_nonzero(cropped_zyx))
    containment_ratio = (
        float(contained_voxels / full_voxels) if full_voxels > 0 else None
    )

    padding_low = np.minimum(size_xyz, np.maximum(0, -start))
    remaining_after_low = size_xyz - padding_low
    padding_high = np.minimum(
        remaining_after_low, np.maximum(0, end_exclusive - shape_xyz)
    )
    padding = {
        "x_low": int(padding_low[0]),
        "x_high": int(padding_high[0]),
        "y_low": int(padding_low[1]),
        "y_high": int(padding_high[1]),
        "z_low": int(padding_low[2]),
        "z_high": int(padding_high[2]),
    }

    box = bbox_xyz(mask)
    if box is None:
        margins_voxel: dict[str, int | None] = {face: None for face in _FACE_NAMES}
        margins_mm: dict[str, float | None] = {face: None for face in _FACE_NAMES}
        extent_voxel = [0, 0, 0]
        extent_mm = [0.0, 0.0, 0.0]
        center_span_mm = [0.0, 0.0, 0.0]
        min_margin_voxel = None
        min_margin_mm = None
    else:
        minimum = np.asarray(box["min_xyz"], dtype=np.int64)
        maximum = np.asarray(box["max_xyz"], dtype=np.int64)
        low_margins = minimum - start
        high_margins = end_inclusive - maximum
        margin_values = (
            int(low_margins[0]),
            int(high_margins[0]),
            int(low_margins[1]),
            int(high_margins[1]),
            int(low_margins[2]),
            int(high_margins[2]),
        )
        margins_voxel = dict(zip(_FACE_NAMES, margin_values))
        margins_mm = {
            "x_low": float(low_margins[0] * spacing[0]),
            "x_high": float(high_margins[0] * spacing[0]),
            "y_low": float(low_margins[1] * spacing[1]),
            "y_high": float(high_margins[1] * spacing[1]),
            "z_low": float(low_margins[2] * spacing[2]),
            "z_high": float(high_margins[2] * spacing[2]),
        }
        extent = maximum - minimum + 1
        extent_voxel = [int(value) for value in extent]
        extent_mm = [float(value) for value in extent * spacing]
        center_span_mm = [float(value) for value in (extent - 1) * spacing]
        min_margin_voxel = int(min(margin_values))
        min_margin_mm = float(min(margins_mm.values()))

    touch = _boundary_touch(cropped_zyx)
    crop_extent_xyz_mm = [float(value) for value in size_xyz * spacing]
    crop_extent_zyx_mm = list(reversed(crop_extent_xyz_mm))

    return {
        "source_shape_xyz": [int(value) for value in shape_xyz],
        "start_xyz": [int(value) for value in start],
        "end_inclusive_xyz": [int(value) for value in end_inclusive],
        "end_exclusive_xyz": [int(value) for value in end_exclusive],
        "crop_size_zyx": [int(value) for value in crop_zyx],
        "crop_size_xyz": [int(value) for value in size_xyz],
        "bbox_xyz": box,
        "signed_margins_voxel": margins_voxel,
        "signed_margins_mm": margins_mm,
        "min_margin_voxel": min_margin_voxel,
        "min_margin_mm": min_margin_mm,
        "extent_xyz_voxel": extent_voxel,
        "extent_xyz_mm": extent_mm,
        "center_span_xyz_mm": center_span_mm,
        "crop_physical_extent_xyz_mm": crop_extent_xyz_mm,
        "crop_physical_extent_zyx_mm": crop_extent_zyx_mm,
        "padding_voxel": padding,
        "boundary_touch": touch,
        "any_boundary_touch": bool(any(touch.values())),
        "full_support_voxels": full_voxels,
        "contained_voxels": contained_voxels,
        "containment_ratio": containment_ratio,
        "complete_miss": bool(full_voxels > 0 and contained_voxels == 0),
    }


def _bbox_diagonal_mm(mask_xyz: np.ndarray, spacing_xyz: np.ndarray) -> float | None:
    """Return the physical diagonal across the bbox of voxel centers."""

    coordinates = np.argwhere(mask_xyz)
    if coordinates.size == 0:
        return None
    center_span = (coordinates.max(axis=0) - coordinates.min(axis=0)) * spacing_xyz
    return float(np.linalg.norm(center_span))


def _fixed_direction_extent_mm(
    mask_xyz: np.ndarray, spacing_xyz: np.ndarray
) -> tuple[float | None, int]:
    """Approximate maximum point distance from fixed-direction extrema."""

    coordinates = np.argwhere(mask_xyz)
    if coordinates.size == 0:
        return None, 0
    if len(coordinates) == 1:
        return 0.0, 1

    points_mm = coordinates.astype(np.float64) * spacing_xyz[None, :]
    extrema_indices: set[int] = set()
    for direction in _FERET_DIRECTIONS_XYZ:
        projection = points_mm @ direction
        # np.argmin/argmax select the first point in deterministic XYZ scan order
        # when an extremal plane contains more than one voxel center.
        extrema_indices.add(int(np.argmin(projection)))
        extrema_indices.add(int(np.argmax(projection)))

    extrema = points_mm[sorted(extrema_indices)]
    if len(extrema) == 1:
        return 0.0, 1
    differences = extrema[:, None, :] - extrema[None, :, :]
    squared_distances = np.einsum("ijk,ijk->ij", differences, differences)
    return float(np.sqrt(np.max(squared_distances))), int(len(extrema))


def approx_max_extent_mm(mask_xyz: Any, spacing_xyz: Iterable[Any]) -> dict[str, Any]:
    """Approximate whole-union and largest-component 3-D Feret extents.

    Voxel centers are mapped to physical XYZ coordinates using ``spacing_xyz``.
    Connected components use deterministic 26-connectivity.  For each support,
    extrema along 13 fixed unoriented axes are collected and their maximum
    pairwise Euclidean distance is reported.  This is a spatial sanity-check
    proxy, not a reconstruction of radiologist-reported LD.
    """

    mask = _mask_xyz(mask_xyz)
    spacing = np.asarray(_spacing_triplet(spacing_xyz), dtype=np.float64)
    coordinates = np.argwhere(mask)
    full_voxels = int(len(coordinates))

    if full_voxels == 0:
        return {
            "method": "fixed_13_direction_extrema_voxel_centers",
            "direction_count": int(len(_FERET_DIRECTIONS_XYZ)),
            "component_connectivity": 26,
            "component_count": 0,
            "whole_union_voxel_count": 0,
            "largest_component_voxel_count": 0,
            "whole_union_approx_max_extent_mm": None,
            "largest_component_approx_max_extent_mm": None,
            "whole_union_bbox_diagonal_mm": None,
            "largest_component_bbox_diagonal_mm": None,
            "whole_union_extrema_count": 0,
            "largest_component_extrema_count": 0,
        }

    # Connected-component labeling on the full acquisition grid can allocate
    # hundreds of MiB even when lesion support is compact.  Cropping to the
    # nonzero bbox is translation-invariant for all reported distances and
    # preserves connectivity while bounding memory and runtime.
    minimum = coordinates.min(axis=0)
    maximum = coordinates.max(axis=0)
    support = mask[
        tuple(
            slice(int(low), int(high) + 1)
            for low, high in zip(minimum, maximum, strict=True)
        )
    ]
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labels, component_count = ndimage.label(support, structure=structure)
    component_sizes = np.bincount(labels.ravel())
    component_sizes[0] = 0
    maximum_size = int(component_sizes.max())
    # scipy labels components in deterministic scan order; choose the first
    # label when components tie in voxel count.
    largest_label = int(np.flatnonzero(component_sizes == maximum_size)[0])
    largest = labels == largest_label

    whole_extent, whole_extrema_count = _fixed_direction_extent_mm(support, spacing)
    largest_extent, largest_extrema_count = _fixed_direction_extent_mm(largest, spacing)

    return {
        "method": "fixed_13_direction_extrema_voxel_centers",
        "direction_count": int(len(_FERET_DIRECTIONS_XYZ)),
        "component_connectivity": 26,
        "component_count": int(component_count),
        "whole_union_voxel_count": full_voxels,
        "largest_component_voxel_count": maximum_size,
        "whole_union_approx_max_extent_mm": whole_extent,
        "largest_component_approx_max_extent_mm": largest_extent,
        "whole_union_bbox_diagonal_mm": _bbox_diagonal_mm(support, spacing),
        "largest_component_bbox_diagonal_mm": _bbox_diagonal_mm(largest, spacing),
        "whole_union_extrema_count": whole_extrema_count,
        "largest_component_extrema_count": largest_extrema_count,
    }

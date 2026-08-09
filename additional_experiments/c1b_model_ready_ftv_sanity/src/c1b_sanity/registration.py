"""Outcome-free rigid registration of longitudinal precontrast MR volumes.

The public registration API accepts only precontrast :class:`CanonicalVolume`
objects.  In particular, no lesion mask, FTV, response, or clinical field can
enter registration.  SimpleITK internally uses LPS physical coordinates and a
fixed-to-moving resampling transform; the transform returned here is the
inverse, a source/moving-to-anchor/T0 transform in RAS coordinates suitable for
``geometry.input_from_output_affine``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from scipy.spatial.transform import Rotation

from .geometry import CanonicalVolume, validate_affine


_LPS_RAS_FLIP = np.diag((-1.0, -1.0, 1.0, 1.0))
_DIRECTION_TOLERANCE = 2.0e-5
_RIGID_TOLERANCE = 2.0e-5


class RegistrationFailureCode(str, Enum):
    """Stable, machine-readable registration outcome codes."""

    NONE = "NONE"
    INVALID_INPUT = "INVALID_INPUT"
    CONSTANT_ANCHOR = "CONSTANT_ANCHOR"
    CONSTANT_SOURCE = "CONSTANT_SOURCE"
    INSUFFICIENT_ANATOMY = "INSUFFICIENT_ANATOMY"
    REGISTRATION_EXCEPTION = "REGISTRATION_EXCEPTION"
    NONCONVERGED = "NONCONVERGED"
    NONFINITE_TRANSFORM = "NONFINITE_TRANSFORM"
    NONRIGID_TRANSFORM = "NONRIGID_TRANSFORM"
    REFLECTION = "REFLECTION"
    INSUFFICIENT_OVERLAP = "INSUFFICIENT_OVERLAP"
    NONFINITE_METRICS = "NONFINITE_METRICS"


@dataclass(frozen=True)
class RegistrationConfig:
    """Deterministic Mattes-MI rigid-registration settings."""

    random_seed: int = 1729
    histogram_bins: int = 48
    sampling_percentage: float = 0.30
    shrink_factors: tuple[int, ...] = (4, 2, 1)
    smoothing_sigmas_mm: tuple[float, ...] = (2.0, 1.0, 0.0)
    maximum_iterations: int = 300
    learning_rate: float = 2.0
    minimum_step_mm: float = 1.0e-4
    relaxation_factor: float = 0.5
    gradient_magnitude_tolerance: float = 1.0e-6
    minimum_anatomy_voxels: int = 128
    minimum_valid_overlap_fraction: float = 0.20
    number_of_threads: int = 1

    def __post_init__(self) -> None:
        if not (0 <= int(self.random_seed) <= np.iinfo(np.uint32).max):
            raise ValueError("random_seed must fit in uint32")
        if int(self.histogram_bins) < 8:
            raise ValueError("histogram_bins must be at least 8")
        if not (0.0 < float(self.sampling_percentage) <= 1.0):
            raise ValueError("sampling_percentage must be in (0, 1]")
        shrink = tuple(int(value) for value in self.shrink_factors)
        smooth = tuple(float(value) for value in self.smoothing_sigmas_mm)
        if not shrink or len(shrink) != len(smooth):
            raise ValueError("shrink_factors and smoothing_sigmas_mm must have equal nonzero length")
        if any(value < 1 for value in shrink):
            raise ValueError("shrink factors must be positive")
        if any(not np.isfinite(value) or value < 0.0 for value in smooth):
            raise ValueError("smoothing sigmas must be finite and nonnegative")
        if int(self.maximum_iterations) < 1:
            raise ValueError("maximum_iterations must be positive")
        if not np.isfinite(self.learning_rate) or float(self.learning_rate) <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(self.minimum_step_mm) or float(self.minimum_step_mm) <= 0.0:
            raise ValueError("minimum_step_mm must be finite and positive")
        if not (0.0 < float(self.relaxation_factor) < 1.0):
            raise ValueError("relaxation_factor must be in (0, 1)")
        if (
            not np.isfinite(self.gradient_magnitude_tolerance)
            or float(self.gradient_magnitude_tolerance) < 0.0
        ):
            raise ValueError("gradient_magnitude_tolerance must be finite and nonnegative")
        if int(self.minimum_anatomy_voxels) < 8:
            raise ValueError("minimum_anatomy_voxels must be at least 8")
        if not (0.0 < float(self.minimum_valid_overlap_fraction) <= 1.0):
            raise ValueError("minimum_valid_overlap_fraction must be in (0, 1]")
        if int(self.number_of_threads) < 1:
            raise ValueError("number_of_threads must be positive")
        object.__setattr__(self, "shrink_factors", shrink)
        object.__setattr__(self, "smoothing_sigmas_mm", smooth)


@dataclass(frozen=True)
class RegistrationSidecars:
    """Image-only overlap and padding diagnostics on the fixed T0 grid."""

    fixed_grid_voxels: int
    fixed_anatomy_voxels: int
    valid_voxels_before: int
    valid_voxels_after: int
    valid_overlap_fraction_before: float
    valid_overlap_fraction_after: float
    padding_fraction_before: float
    padding_fraction_after: float
    fixed_anatomy_valid_fraction_before: float
    fixed_anatomy_valid_fraction_after: float
    anatomy_overlap_voxels_before: int
    anatomy_overlap_voxels_after: int
    anatomy_dice_before: float
    anatomy_dice_after: float


@dataclass(frozen=True)
class RegistrationResult:
    """Fail-closed registration output and image-only QC measurements."""

    success: bool
    failure_code: RegistrationFailureCode
    failure_message: str | None
    source_to_anchor_ras: np.ndarray | None
    converged: bool
    optimizer_stop_condition: str | None
    optimizer_iterations: int | None
    final_mattes_mi: float | None
    rotation_xyz_deg: tuple[float, float, float] | None
    rotation_magnitude_deg: float | None
    translation_ras_mm: tuple[float, float, float] | None
    translation_magnitude_mm: float | None
    affine_offset_ras_mm: tuple[float, float, float] | None
    similarity_before: float | None
    similarity_after: float | None
    similarity_delta: float | None
    sidecars: RegistrationSidecars | None


def _failure(
    code: RegistrationFailureCode,
    message: str,
    *,
    converged: bool = False,
    stop_condition: str | None = None,
    iterations: int | None = None,
    final_metric: float | None = None,
    similarity_before: float | None = None,
    similarity_after: float | None = None,
    sidecars: RegistrationSidecars | None = None,
) -> RegistrationResult:
    delta = None
    if similarity_before is not None and similarity_after is not None:
        delta = float(similarity_after - similarity_before)
    return RegistrationResult(
        success=False,
        failure_code=code,
        failure_message=message,
        source_to_anchor_ras=None,
        converged=converged,
        optimizer_stop_condition=stop_condition,
        optimizer_iterations=iterations,
        final_mattes_mi=final_metric,
        rotation_xyz_deg=None,
        rotation_magnitude_deg=None,
        translation_ras_mm=None,
        translation_magnitude_mm=None,
        affine_offset_ras_mm=None,
        similarity_before=similarity_before,
        similarity_after=similarity_after,
        similarity_delta=delta,
        sidecars=sidecars,
    )


def _validate_precontrast_volume(volume: CanonicalVolume, *, name: str) -> np.ndarray:
    if not isinstance(volume, CanonicalVolume):
        raise TypeError(f"{name} must be a CanonicalVolume")
    array = np.asarray(volume.data)
    if array.ndim != 3:
        raise ValueError(f"{name} precontrast array must be 3-D, got {array.shape}")
    if any(int(length) < 2 for length in array.shape):
        raise ValueError(f"{name} has a spatial axis shorter than two voxels")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} array must be numeric")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} array contains non-finite values")
    validate_affine(volume.affine_ras, name=f"{name} affine_ras")
    return np.asarray(array, dtype=np.float32)


def _lps_geometry_from_ras_affine(affine_ras: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    affine = validate_affine(affine_ras, name="affine_ras")
    affine_lps = _LPS_RAS_FLIP @ affine
    linear = affine_lps[:3, :3]
    spacing = np.linalg.norm(linear, axis=0)
    if not np.isfinite(spacing).all() or np.any(spacing <= 0.0):
        raise ValueError("affine defines invalid voxel spacing")
    direction = linear / spacing[np.newaxis, :]
    if not np.allclose(
        direction.T @ direction,
        np.eye(3),
        atol=_DIRECTION_TOLERANCE,
        rtol=0.0,
    ):
        raise ValueError("SimpleITK conversion requires an orthogonal, shear-free affine")
    determinant = float(np.linalg.det(direction))
    if determinant <= 0.0:
        raise ValueError("affine direction contains a reflection")
    if not np.isclose(determinant, 1.0, atol=_DIRECTION_TOLERANCE, rtol=0.0):
        raise ValueError("affine direction is not a proper rotation")
    return spacing, direction, affine_lps[:3, 3]


def canonical_volume_to_sitk(volume: CanonicalVolume) -> sitk.Image:
    """Convert an ``[X,Y,Z]`` RAS-world volume to a 3-D SimpleITK LPS image."""

    array = _validate_precontrast_volume(volume, name="volume")
    spacing, direction, origin = _lps_geometry_from_ras_affine(volume.affine_ras)
    image = sitk.GetImageFromArray(np.transpose(array, (2, 1, 0)), isVector=False)
    image.SetSpacing(tuple(float(value) for value in spacing))
    image.SetDirection(tuple(float(value) for value in direction.ravel(order="C")))
    image.SetOrigin(tuple(float(value) for value in origin))
    return image


def sitk_image_to_affine_ras(image: sitk.Image) -> np.ndarray:
    """Reconstruct a voxel-to-RAS affine from a 3-D SimpleITK image."""

    if not isinstance(image, sitk.Image) or image.GetDimension() != 3:
        raise ValueError("image must be a 3-D SimpleITK image")
    spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    origin = np.asarray(image.GetOrigin(), dtype=np.float64)
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    if not np.isfinite(spacing).all() or np.any(spacing <= 0.0):
        raise ValueError("SimpleITK image has invalid spacing")
    if not np.isfinite(direction).all() or not np.isfinite(origin).all():
        raise ValueError("SimpleITK image has non-finite geometry")
    affine_lps = np.eye(4, dtype=np.float64)
    affine_lps[:3, :3] = direction @ np.diag(spacing)
    affine_lps[:3, 3] = origin
    return validate_affine(_LPS_RAS_FLIP @ affine_lps, name="reconstructed affine_ras")


def _validate_rigid_matrix(matrix: np.ndarray, *, name: str) -> np.ndarray:
    rigid = validate_affine(matrix, name=name)
    rotation = rigid[:3, :3]
    determinant = float(np.linalg.det(rotation))
    if determinant <= 0.0:
        raise RuntimeError(f"{name} contains a reflection")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=_RIGID_TOLERANCE, rtol=0.0):
        raise RuntimeError(f"{name} is not rigid")
    if not np.isclose(determinant, 1.0, atol=_RIGID_TOLERANCE, rtol=0.0):
        raise RuntimeError(f"{name} has non-unit rotation determinant")
    return rigid


def sitk_transform_to_ras_matrix(transform: sitk.Transform) -> np.ndarray:
    """Convert any 3-D affine-valued SimpleITK LPS transform to a RAS matrix."""

    if not isinstance(transform, sitk.Transform) or transform.GetDimension() != 3:
        raise ValueError("transform must be a 3-D SimpleITK transform")
    zero = np.zeros(3, dtype=np.float64)
    mapped_zero = np.asarray(transform.TransformPoint(tuple(zero)), dtype=np.float64)
    matrix_lps = np.eye(4, dtype=np.float64)
    for axis in range(3):
        basis = zero.copy()
        basis[axis] = 1.0
        mapped_basis = np.asarray(transform.TransformPoint(tuple(basis)), dtype=np.float64)
        matrix_lps[:3, axis] = mapped_basis - mapped_zero
    matrix_lps[:3, 3] = mapped_zero
    if not np.isfinite(matrix_lps).all():
        raise ValueError("SimpleITK transform maps points to non-finite coordinates")
    return _LPS_RAS_FLIP @ matrix_lps @ _LPS_RAS_FLIP


def ras_matrix_to_sitk_transform(matrix_ras: np.ndarray) -> sitk.AffineTransform:
    """Convert a proper rigid RAS matrix to a SimpleITK LPS transform."""

    rigid_ras = _validate_rigid_matrix(np.asarray(matrix_ras, dtype=np.float64), name="matrix_ras")
    matrix_lps = _LPS_RAS_FLIP @ rigid_ras @ _LPS_RAS_FLIP
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(tuple(float(value) for value in matrix_lps[:3, :3].ravel(order="C")))
    transform.SetTranslation(tuple(float(value) for value in matrix_lps[:3, 3]))
    return transform


def _robust_dynamic_range(array: np.ndarray) -> float:
    low, high = np.percentile(np.asarray(array, dtype=np.float64), (0.5, 99.5))
    return float(high - low)


def _is_effectively_constant(array: np.ndarray) -> bool:
    values = np.asarray(array, dtype=np.float64)
    robust_range = _robust_dynamic_range(values)
    scale = max(float(np.max(np.abs(values))), 1.0)
    return not np.isfinite(robust_range) or robust_range <= 1.0e-7 * scale


def _whole_anatomy_mask_array(array: np.ndarray, *, minimum_voxels: int) -> np.ndarray:
    """Derive a broad foreground mask from precontrast intensity alone."""

    values = np.asarray(array, dtype=np.float32)
    low, high = np.percentile(values.astype(np.float64), (0.5, 99.5))
    span = float(high - low)
    if not np.isfinite(span) or span <= 1.0e-7 * max(abs(float(low)), abs(float(high)), 1.0):
        raise ValueError("precontrast intensity is effectively constant")
    normalized = np.clip((values.astype(np.float64) - low) / span, 0.0, 1.0).astype(np.float32)

    normalized_image = sitk.GetImageFromArray(np.transpose(normalized, (2, 1, 0)))
    otsu_image = sitk.OtsuThreshold(normalized_image, 0, 1, 128)
    mask = np.transpose(sitk.GetArrayFromImage(otsu_image).astype(bool), (2, 1, 0))

    # Otsu can be unstable when a narrow, nonzero background dominates.  A
    # border-derived fallback still uses only the precontrast image and favors
    # broad anatomy rather than any focal high-intensity region.
    fraction = float(mask.mean())
    if fraction < 0.005 or fraction > 0.95:
        border = np.concatenate(
            (
                normalized[0, :, :].ravel(),
                normalized[-1, :, :].ravel(),
                normalized[:, 0, :].ravel(),
                normalized[:, -1, :].ravel(),
                normalized[:, :, 0].ravel(),
                normalized[:, :, -1].ravel(),
            )
        )
        background = float(np.median(border))
        mask = np.abs(normalized - background) > 0.04

    structure = ndi.generate_binary_structure(3, 1)
    labels, count = ndi.label(mask, structure=structure)
    if count < 1:
        raise ValueError("automatic precontrast anatomy mask is empty")
    component_sizes = np.bincount(labels.ravel())[1:]
    largest = int(component_sizes.max(initial=0))
    component_floor = max(16, int(np.ceil(0.01 * largest)))
    keep_labels = np.flatnonzero(component_sizes >= component_floor) + 1
    mask = np.isin(labels, keep_labels)
    mask = ndi.binary_closing(mask, structure=structure, iterations=1)
    mask = ndi.binary_fill_holes(mask)
    mask = ndi.binary_dilation(mask, structure=structure, iterations=1)
    if int(mask.sum()) < int(minimum_voxels):
        raise ValueError(
            f"automatic precontrast anatomy mask has {int(mask.sum())} voxels, "
            f"below minimum {int(minimum_voxels)}"
        )
    return np.asarray(mask, dtype=bool)


def _mask_to_sitk(mask_xyz: np.ndarray, reference: sitk.Image) -> sitk.Image:
    mask_image = sitk.GetImageFromArray(
        np.transpose(np.asarray(mask_xyz, dtype=np.uint8), (2, 1, 0))
    )
    mask_image.CopyInformation(reference)
    return mask_image


def _resample_array(
    moving: sitk.Image,
    fixed: sitk.Image,
    fixed_to_moving_lps: sitk.Transform,
    *,
    interpolator: int,
    default_value: float,
    pixel_id: int,
) -> np.ndarray:
    resampled = sitk.Resample(
        moving,
        fixed,
        fixed_to_moving_lps,
        interpolator,
        float(default_value),
        pixel_id,
    )
    return np.transpose(sitk.GetArrayFromImage(resampled), (2, 1, 0))


def _normalized_cross_correlation(fixed: np.ndarray, moving: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(mask, dtype=bool)
    if int(selected.sum()) < 8:
        raise ValueError("too few overlapping anatomy voxels for similarity")
    fixed_values = np.asarray(fixed, dtype=np.float64)[selected]
    moving_values = np.asarray(moving, dtype=np.float64)[selected]
    fixed_values -= fixed_values.mean()
    moving_values -= moving_values.mean()
    denominator = float(np.linalg.norm(fixed_values) * np.linalg.norm(moving_values))
    if not np.isfinite(denominator) or denominator <= np.finfo(np.float64).eps:
        raise ValueError("overlapping anatomy has zero variance")
    result = float(np.dot(fixed_values, moving_values) / denominator)
    if not np.isfinite(result):
        raise ValueError("similarity is non-finite")
    return float(np.clip(result, -1.0, 1.0))


def _similarity_and_sidecars(
    fixed_array: np.ndarray,
    fixed_image: sitk.Image,
    moving_image: sitk.Image,
    fixed_mask: np.ndarray,
    moving_mask_image: sitk.Image,
    fixed_to_moving_before: sitk.Transform,
    fixed_to_moving_after: sitk.Transform,
) -> tuple[float, float, RegistrationSidecars]:
    moving_before = _resample_array(
        moving_image,
        fixed_image,
        fixed_to_moving_before,
        interpolator=sitk.sitkLinear,
        default_value=0.0,
        pixel_id=sitk.sitkFloat32,
    )
    moving_after = _resample_array(
        moving_image,
        fixed_image,
        fixed_to_moving_after,
        interpolator=sitk.sitkLinear,
        default_value=0.0,
        pixel_id=sitk.sitkFloat32,
    )
    moving_mask_before = _resample_array(
        moving_mask_image,
        fixed_image,
        fixed_to_moving_before,
        interpolator=sitk.sitkNearestNeighbor,
        default_value=0.0,
        pixel_id=sitk.sitkUInt8,
    ).astype(bool)
    moving_mask_after = _resample_array(
        moving_mask_image,
        fixed_image,
        fixed_to_moving_after,
        interpolator=sitk.sitkNearestNeighbor,
        default_value=0.0,
        pixel_id=sitk.sitkUInt8,
    ).astype(bool)

    moving_valid_image = sitk.Image(moving_image.GetSize(), sitk.sitkUInt8)
    moving_valid_image.CopyInformation(moving_image)
    moving_valid_image += 1
    valid_before = _resample_array(
        moving_valid_image,
        fixed_image,
        fixed_to_moving_before,
        interpolator=sitk.sitkNearestNeighbor,
        default_value=0.0,
        pixel_id=sitk.sitkUInt8,
    ).astype(bool)
    valid_after = _resample_array(
        moving_valid_image,
        fixed_image,
        fixed_to_moving_after,
        interpolator=sitk.sitkNearestNeighbor,
        default_value=0.0,
        pixel_id=sitk.sitkUInt8,
    ).astype(bool)

    common_before = fixed_mask & moving_mask_before & valid_before
    common_after = fixed_mask & moving_mask_after & valid_after
    similarity_before = _normalized_cross_correlation(fixed_array, moving_before, common_before)
    similarity_after = _normalized_cross_correlation(fixed_array, moving_after, common_after)

    total = int(fixed_mask.size)
    fixed_count = int(fixed_mask.sum())
    valid_before_count = int(valid_before.sum())
    valid_after_count = int(valid_after.sum())
    fixed_valid_before = int((fixed_mask & valid_before).sum())
    fixed_valid_after = int((fixed_mask & valid_after).sum())
    moving_before_count = int(moving_mask_before.sum())
    moving_after_count = int(moving_mask_after.sum())
    overlap_before = int((fixed_mask & moving_mask_before).sum())
    overlap_after = int((fixed_mask & moving_mask_after).sum())
    dice_before_denominator = fixed_count + moving_before_count
    dice_after_denominator = fixed_count + moving_after_count
    sidecars = RegistrationSidecars(
        fixed_grid_voxels=total,
        fixed_anatomy_voxels=fixed_count,
        valid_voxels_before=valid_before_count,
        valid_voxels_after=valid_after_count,
        valid_overlap_fraction_before=float(valid_before_count / total),
        valid_overlap_fraction_after=float(valid_after_count / total),
        padding_fraction_before=float(1.0 - valid_before_count / total),
        padding_fraction_after=float(1.0 - valid_after_count / total),
        fixed_anatomy_valid_fraction_before=float(fixed_valid_before / fixed_count),
        fixed_anatomy_valid_fraction_after=float(fixed_valid_after / fixed_count),
        anatomy_overlap_voxels_before=overlap_before,
        anatomy_overlap_voxels_after=overlap_after,
        anatomy_dice_before=float(
            2.0 * overlap_before / dice_before_denominator
            if dice_before_denominator
            else 0.0
        ),
        anatomy_dice_after=float(
            2.0 * overlap_after / dice_after_denominator
            if dice_after_denominator
            else 0.0
        ),
    )
    return similarity_before, similarity_after, sidecars


def _optimizer_converged(stop_condition: str, iteration: int, maximum_iterations: int) -> bool:
    description = stop_condition.lower()
    if "maximum number of iterations" in description or int(iteration) >= int(maximum_iterations):
        return False
    convergence_markers = (
        "step too small",
        "gradient magnitude tolerance",
        "convergence checker passed",
    )
    return any(marker in description for marker in convergence_markers)


def _transform_metrics(
    source_to_anchor_ras: np.ndarray,
    source: CanonicalVolume,
) -> tuple[tuple[float, float, float], float, tuple[float, float, float], float, tuple[float, float, float]]:
    rigid = _validate_rigid_matrix(source_to_anchor_ras, name="source_to_anchor_ras")
    rotation = rigid[:3, :3]
    euler_xyz = Rotation.from_matrix(rotation).as_euler("xyz", degrees=True)
    rotation_magnitude = float(np.degrees(Rotation.from_matrix(rotation).magnitude()))

    center_voxel = 0.5 * (np.asarray(source.shape_xyz, dtype=np.float64) - 1.0)
    center_ras = source.affine_ras[:3, :3] @ center_voxel + source.affine_ras[:3, 3]
    transformed_center = rotation @ center_ras + rigid[:3, 3]
    translation = transformed_center - center_ras
    return (
        tuple(float(value) for value in euler_xyz),
        rotation_magnitude,
        tuple(float(value) for value in translation),
        float(np.linalg.norm(translation)),
        tuple(float(value) for value in rigid[:3, 3]),
    )


def register_precontrast_rigid(
    anchor_t0: CanonicalVolume,
    source: CanonicalVolume,
    config: RegistrationConfig | None = None,
) -> RegistrationResult:
    """Register one precontrast source to fixed T0 with deterministic rigid MI.

    The successful ``source_to_anchor_ras`` matrix maps source-world RAS points
    into T0-world RAS points.  Any invalid, nonconvergent, reflected, nonrigid,
    nonfinite, or insufficient-overlap outcome returns no transform.
    """

    settings = config if config is not None else RegistrationConfig()
    if not isinstance(settings, RegistrationConfig):
        return _failure(RegistrationFailureCode.INVALID_INPUT, "config must be RegistrationConfig")
    try:
        fixed_array = _validate_precontrast_volume(anchor_t0, name="anchor_t0")
        moving_array = _validate_precontrast_volume(source, name="source")
    except (TypeError, ValueError) as exc:
        return _failure(RegistrationFailureCode.INVALID_INPUT, str(exc))

    if _is_effectively_constant(fixed_array):
        return _failure(
            RegistrationFailureCode.CONSTANT_ANCHOR,
            "anchor_t0 precontrast intensity is effectively constant",
        )
    if _is_effectively_constant(moving_array):
        return _failure(
            RegistrationFailureCode.CONSTANT_SOURCE,
            "source precontrast intensity is effectively constant",
        )

    try:
        fixed_image = canonical_volume_to_sitk(anchor_t0)
        moving_image = canonical_volume_to_sitk(source)
        fixed_mask_array = _whole_anatomy_mask_array(
            fixed_array,
            minimum_voxels=settings.minimum_anatomy_voxels,
        )
        moving_mask_array = _whole_anatomy_mask_array(
            moving_array,
            minimum_voxels=settings.minimum_anatomy_voxels,
        )
        fixed_mask_image = _mask_to_sitk(fixed_mask_array, fixed_image)
        moving_mask_image = _mask_to_sitk(moving_mask_array, moving_image)
    except (RuntimeError, TypeError, ValueError) as exc:
        return _failure(RegistrationFailureCode.INSUFFICIENT_ANATOMY, str(exc))

    registration = sitk.ImageRegistrationMethod()
    registration.SetNumberOfThreads(int(settings.number_of_threads))
    registration.SetMetricAsMattesMutualInformation(
        numberOfHistogramBins=int(settings.histogram_bins)
    )
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(
        float(settings.sampling_percentage),
        int(settings.random_seed),
    )
    registration.SetMetricFixedMask(fixed_mask_image)
    registration.SetMetricMovingMask(moving_mask_image)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=float(settings.learning_rate),
        minStep=float(settings.minimum_step_mm),
        numberOfIterations=int(settings.maximum_iterations),
        relaxationFactor=float(settings.relaxation_factor),
        gradientMagnitudeTolerance=float(settings.gradient_magnitude_tolerance),
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel(list(settings.shrink_factors))
    registration.SetSmoothingSigmasPerLevel(list(settings.smoothing_sigmas_mm))
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    try:
        initial = sitk.CenteredTransformInitializer(
            fixed_image,
            moving_image,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.MOMENTS,
        )
        registration.SetInitialTransform(initial, inPlace=False)
        fixed_to_moving_lps = registration.Execute(fixed_image, moving_image)
        stop_condition = str(registration.GetOptimizerStopConditionDescription())
        iterations = int(registration.GetOptimizerIteration())
        final_metric = float(registration.GetMetricValue())
    except RuntimeError as exc:
        return _failure(
            RegistrationFailureCode.REGISTRATION_EXCEPTION,
            f"SimpleITK registration failed: {exc}",
        )

    if not np.isfinite(final_metric):
        return _failure(
            RegistrationFailureCode.NONFINITE_METRICS,
            "optimizer returned a non-finite Mattes MI value",
            stop_condition=stop_condition,
            iterations=iterations,
        )
    converged = _optimizer_converged(
        stop_condition,
        iterations,
        settings.maximum_iterations,
    )
    if not converged:
        return _failure(
            RegistrationFailureCode.NONCONVERGED,
            f"rigid optimizer did not converge: {stop_condition}",
            stop_condition=stop_condition,
            iterations=iterations,
            final_metric=final_metric,
        )

    try:
        fixed_to_source_ras = sitk_transform_to_ras_matrix(fixed_to_moving_lps)
        if not np.isfinite(fixed_to_source_ras).all():
            raise FloatingPointError("registration transform is non-finite")
        source_to_anchor_ras = np.linalg.inv(fixed_to_source_ras)
        _validate_rigid_matrix(source_to_anchor_ras, name="source_to_anchor_ras")
    except FloatingPointError as exc:
        return _failure(
            RegistrationFailureCode.NONFINITE_TRANSFORM,
            str(exc),
            converged=True,
            stop_condition=stop_condition,
            iterations=iterations,
            final_metric=final_metric,
        )
    except RuntimeError as exc:
        code = (
            RegistrationFailureCode.REFLECTION
            if "reflection" in str(exc).lower()
            else RegistrationFailureCode.NONRIGID_TRANSFORM
        )
        return _failure(
            code,
            str(exc),
            converged=True,
            stop_condition=stop_condition,
            iterations=iterations,
            final_metric=final_metric,
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        return _failure(
            RegistrationFailureCode.NONFINITE_TRANSFORM,
            str(exc),
            converged=True,
            stop_condition=stop_condition,
            iterations=iterations,
            final_metric=final_metric,
        )

    try:
        identity_lps = sitk.Transform(3, sitk.sitkIdentity)
        similarity_before, similarity_after, sidecars = _similarity_and_sidecars(
            fixed_array,
            fixed_image,
            moving_image,
            fixed_mask_array,
            moving_mask_image,
            identity_lps,
            fixed_to_moving_lps,
        )
    except (RuntimeError, ValueError) as exc:
        return _failure(
            RegistrationFailureCode.NONFINITE_METRICS,
            f"image-only registration QC failed: {exc}",
            converged=True,
            stop_condition=stop_condition,
            iterations=iterations,
            final_metric=final_metric,
        )

    if sidecars.fixed_anatomy_valid_fraction_after < settings.minimum_valid_overlap_fraction:
        return _failure(
            RegistrationFailureCode.INSUFFICIENT_OVERLAP,
            "registered source covers too little of fixed T0 anatomy",
            converged=True,
            stop_condition=stop_condition,
            iterations=iterations,
            final_metric=final_metric,
            similarity_before=similarity_before,
            similarity_after=similarity_after,
            sidecars=sidecars,
        )

    try:
        (
            rotation_xyz_deg,
            rotation_magnitude_deg,
            translation_ras_mm,
            translation_magnitude_mm,
            affine_offset_ras_mm,
        ) = _transform_metrics(source_to_anchor_ras, source)
    except (RuntimeError, ValueError) as exc:
        return _failure(
            RegistrationFailureCode.NONFINITE_METRICS,
            f"transform metrics failed: {exc}",
            converged=True,
            stop_condition=stop_condition,
            iterations=iterations,
            final_metric=final_metric,
            similarity_before=similarity_before,
            similarity_after=similarity_after,
            sidecars=sidecars,
        )

    scalar_metrics = np.asarray(
        (
            final_metric,
            rotation_magnitude_deg,
            translation_magnitude_mm,
            similarity_before,
            similarity_after,
        ),
        dtype=np.float64,
    )
    if not np.isfinite(scalar_metrics).all():
        return _failure(
            RegistrationFailureCode.NONFINITE_METRICS,
            "registration produced non-finite QC metrics",
            converged=True,
            stop_condition=stop_condition,
            iterations=iterations,
            final_metric=final_metric,
            similarity_before=similarity_before,
            similarity_after=similarity_after,
            sidecars=sidecars,
        )

    return RegistrationResult(
        success=True,
        failure_code=RegistrationFailureCode.NONE,
        failure_message=None,
        source_to_anchor_ras=np.asarray(source_to_anchor_ras, dtype=np.float64),
        converged=True,
        optimizer_stop_condition=stop_condition,
        optimizer_iterations=iterations,
        final_mattes_mi=final_metric,
        rotation_xyz_deg=rotation_xyz_deg,
        rotation_magnitude_deg=rotation_magnitude_deg,
        translation_ras_mm=translation_ras_mm,
        translation_magnitude_mm=translation_magnitude_mm,
        affine_offset_ras_mm=affine_offset_ras_mm,
        similarity_before=similarity_before,
        similarity_after=similarity_after,
        similarity_delta=float(similarity_after - similarity_before),
        sidecars=sidecars,
    )


def register_t1_t2_t3_to_t0(
    t0: CanonicalVolume,
    t1: CanonicalVolume,
    t2: CanonicalVolume,
    t3: CanonicalVolume,
    config: RegistrationConfig | None = None,
) -> dict[str, RegistrationResult]:
    """Register the three longitudinal precontrast volumes to fixed T0."""

    return {
        "T1": register_precontrast_rigid(t0, t1, config),
        "T2": register_precontrast_rigid(t0, t2, config),
        "T3": register_precontrast_rigid(t0, t3, config),
    }


def register_followups_to_t0(
    t0: CanonicalVolume,
    followups: Sequence[CanonicalVolume],
    config: RegistrationConfig | None = None,
) -> tuple[RegistrationResult, ...]:
    """Register an ordered sequence of precontrast follow-ups to fixed T0."""

    return tuple(register_precontrast_rigid(t0, source, config) for source in followups)


__all__ = [
    "RegistrationConfig",
    "RegistrationFailureCode",
    "RegistrationResult",
    "RegistrationSidecars",
    "canonical_volume_to_sitk",
    "ras_matrix_to_sitk_transform",
    "register_followups_to_t0",
    "register_precontrast_rigid",
    "register_t1_t2_t3_to_t0",
    "sitk_image_to_affine_ras",
    "sitk_transform_to_ras_matrix",
]

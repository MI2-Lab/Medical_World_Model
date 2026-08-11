"""One sealed image-only rigid-registration diagnostic.

This module intentionally preserves an optimizer candidate even when the
pre-registration similarity is undefined because the physical frames do not
overlap.  The result is diagnostic evidence only and is never a repair input.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import SimpleITK as sitk

from c1b_sanity.geometry import CanonicalVolume
import c1b_sanity.registration as frozen_registration


def _resampled_array(
    image: sitk.Image,
    reference: sitk.Image,
    transform: sitk.Transform,
    *,
    interpolation: int,
    default: float,
    pixel_type: int,
) -> np.ndarray:
    output = sitk.Resample(
        image,
        reference,
        transform,
        interpolation,
        default,
        pixel_type,
    )
    return np.ascontiguousarray(
        sitk.GetArrayFromImage(output).transpose(2, 1, 0)
    )


def _valid_image(reference: sitk.Image) -> sitk.Image:
    image = sitk.Image(reference.GetSize(), sitk.sitkUInt8)
    image.CopyInformation(reference)
    return image + 1


def _safe_ncc(first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> float | None:
    selected = np.asarray(mask, dtype=bool)
    if int(np.count_nonzero(selected)) < 8:
        return None
    left = np.asarray(first[selected], dtype=np.float64)
    right = np.asarray(second[selected], dtype=np.float64)
    if float(np.std(left)) <= 0.0 or float(np.std(right)) <= 0.0:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else None


def _dice(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=bool)
    right = np.asarray(second, dtype=bool)
    denominator = int(np.count_nonzero(left)) + int(np.count_nonzero(right))
    if denominator == 0:
        return 1.0
    return float(2 * np.count_nonzero(left & right) / denominator)


def run_image_only_diagnostic(
    anchor_t0: CanonicalVolume,
    source_failed: CanonicalVolume,
) -> dict[str, Any]:
    """Run one deterministic, precontrast-only rigid registration.

    The transform maps source RAS world coordinates into the T0 RAS frame.
    It is returned only under ``private``; public fields contain magnitudes and
    image-only QC.  No lesion, FTV, clinical, treatment, or outcome input is
    accepted by this API.
    """

    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    fixed_array = np.asarray(anchor_t0.data, dtype=np.float32)
    moving_array = np.asarray(source_failed.data, dtype=np.float32)
    fixed_image = frozen_registration.canonical_volume_to_sitk(anchor_t0)
    moving_image = frozen_registration.canonical_volume_to_sitk(source_failed)
    fixed_anatomy = frozen_registration._whole_anatomy_mask_array(  # noqa: SLF001
        fixed_array, minimum_voxels=128
    )
    moving_anatomy = frozen_registration._whole_anatomy_mask_array(  # noqa: SLF001
        moving_array, minimum_voxels=128
    )
    fixed_mask_image = frozen_registration._mask_to_sitk(  # noqa: SLF001
        fixed_anatomy, fixed_image
    )
    moving_mask_image = frozen_registration._mask_to_sitk(  # noqa: SLF001
        moving_anatomy, moving_image
    )

    identity = sitk.Transform(3, sitk.sitkIdentity)
    valid_source = _valid_image(moving_image)
    before_valid = _resampled_array(
        valid_source,
        fixed_image,
        identity,
        interpolation=sitk.sitkNearestNeighbor,
        default=0,
        pixel_type=sitk.sitkUInt8,
    ).astype(bool)
    before_moving = _resampled_array(
        moving_image,
        fixed_image,
        identity,
        interpolation=sitk.sitkLinear,
        default=0.0,
        pixel_type=sitk.sitkFloat32,
    )
    before_anatomy = _resampled_array(
        moving_mask_image,
        fixed_image,
        identity,
        interpolation=sitk.sitkNearestNeighbor,
        default=0,
        pixel_type=sitk.sitkUInt8,
    ).astype(bool)
    before_ncc = _safe_ncc(
        fixed_array, before_moving, fixed_anatomy & before_valid
    )

    settings = {
        "random_seed": 1729,
        "histogram_bins": 48,
        "sampling_percentage": 0.30,
        "sampling_strategy": "RANDOM",
        "shrink_factors": [4, 2, 1],
        "smoothing_sigmas_mm": [2.0, 1.0, 0.0],
        "maximum_iterations": 600,
        "learning_rate": 2.0,
        "minimum_step_mm": 1.0e-4,
        "relaxation_factor": 0.5,
        "gradient_magnitude_tolerance": 1.0e-6,
        "number_of_threads": 1,
        "initialization": "intensity_moments",
    }
    method = sitk.ImageRegistrationMethod()
    method.SetNumberOfThreads(1)
    method.SetMetricAsMattesMutualInformation(
        numberOfHistogramBins=settings["histogram_bins"]
    )
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(
        settings["sampling_percentage"], settings["random_seed"]
    )
    method.SetMetricFixedMask(fixed_mask_image)
    method.SetMetricMovingMask(moving_mask_image)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=settings["learning_rate"],
        minStep=settings["minimum_step_mm"],
        numberOfIterations=settings["maximum_iterations"],
        relaxationFactor=settings["relaxation_factor"],
        gradientMagnitudeTolerance=settings["gradient_magnitude_tolerance"],
    )
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel(settings["shrink_factors"])
    method.SetSmoothingSigmasPerLevel(settings["smoothing_sigmas_mm"])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    initial = sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.MOMENTS,
    )
    method.SetInitialTransform(initial, inPlace=False)

    try:
        fixed_to_moving_lps = method.Execute(fixed_image, moving_image)
    except RuntimeError:
        return {
            "public": {
                "analysis_scope": "DIAGNOSTIC_ONLY",
                "success": False,
                "failure_code": "REGISTRATION_EXCEPTION",
                "optimizer_converged": False,
                "translation_magnitude_mm": None,
                "rotation_magnitude_deg": None,
                "similarity_before_ncc": before_ncc,
                "similarity_after_ncc": None,
                "valid_overlap_fraction_before": float(np.mean(before_valid)),
                "valid_overlap_fraction_after": None,
                "fixed_anatomy_valid_fraction_after": None,
                "anatomy_dice_before": _dice(fixed_anatomy, before_anatomy),
                "anatomy_dice_after": None,
                "transform_used_as_repair": False,
                "settings": settings,
            },
            "private": {"source_to_anchor_ras": None},
            "moved_array": None,
            "moved_valid": None,
        }

    stop = str(method.GetOptimizerStopConditionDescription())
    iterations = int(method.GetOptimizerIteration())
    converged = frozen_registration._optimizer_converged(  # noqa: SLF001
        stop, iterations, settings["maximum_iterations"]
    )
    fixed_to_source_ras = frozen_registration.sitk_transform_to_ras_matrix(
        fixed_to_moving_lps
    )
    source_to_anchor_ras = np.linalg.inv(fixed_to_source_ras)
    (
        rotation_xyz_deg,
        rotation_magnitude_deg,
        translation_ras_mm,
        translation_magnitude_mm,
        affine_offset_ras_mm,
    ) = frozen_registration._transform_metrics(  # noqa: SLF001
        source_to_anchor_ras, source_failed
    )

    after_moving = _resampled_array(
        moving_image,
        fixed_image,
        fixed_to_moving_lps,
        interpolation=sitk.sitkLinear,
        default=0.0,
        pixel_type=sitk.sitkFloat32,
    )
    after_valid = _resampled_array(
        valid_source,
        fixed_image,
        fixed_to_moving_lps,
        interpolation=sitk.sitkNearestNeighbor,
        default=0,
        pixel_type=sitk.sitkUInt8,
    ).astype(bool)
    after_anatomy = _resampled_array(
        moving_mask_image,
        fixed_image,
        fixed_to_moving_lps,
        interpolation=sitk.sitkNearestNeighbor,
        default=0,
        pixel_type=sitk.sitkUInt8,
    ).astype(bool)
    after_ncc = _safe_ncc(
        fixed_array, after_moving, fixed_anatomy & after_valid
    )
    fixed_anatomy_count = max(int(np.count_nonzero(fixed_anatomy)), 1)
    anatomy_valid_after = float(
        np.count_nonzero(fixed_anatomy & after_valid) / fixed_anatomy_count
    )
    success = bool(converged and after_ncc is not None and anatomy_valid_after >= 0.20)
    failure_code = "NONE" if success else (
        "NONCONVERGED" if not converged else "INSUFFICIENT_POST_REGISTRATION_QC"
    )
    return {
        "public": {
            "analysis_scope": "DIAGNOSTIC_ONLY",
            "success": success,
            "failure_code": failure_code,
            "optimizer_converged": converged,
            "optimizer_iterations": iterations,
            "optimizer_stop_condition_category": "MAXIMUM_ITERATIONS"
            if iterations >= settings["maximum_iterations"]
            else "STEP_OR_GRADIENT_STOP",
            "final_mattes_mi": float(method.GetMetricValue()),
            "translation_magnitude_mm": float(translation_magnitude_mm),
            "rotation_magnitude_deg": float(rotation_magnitude_deg),
            "similarity_before_ncc": before_ncc,
            "similarity_after_ncc": after_ncc,
            "valid_overlap_fraction_before": float(np.mean(before_valid)),
            "valid_overlap_fraction_after": float(np.mean(after_valid)),
            "fixed_anatomy_valid_fraction_after": anatomy_valid_after,
            "anatomy_dice_before": _dice(fixed_anatomy, before_anatomy),
            "anatomy_dice_after": _dice(fixed_anatomy, after_anatomy),
            "transform_used_as_repair": False,
            "settings": settings,
        },
        "private": {
            "source_to_anchor_ras": source_to_anchor_ras.tolist(),
            "rotation_xyz_deg": list(rotation_xyz_deg),
            "translation_ras_mm": list(translation_ras_mm),
            "affine_offset_ras_mm": list(affine_offset_ras_mm),
            "optimizer_stop_condition": stop,
        },
        "moved_array": after_moving,
        "moved_valid": after_valid,
    }


__all__ = ["run_image_only_diagnostic"]

"""Deterministic, atomic C1B float32 cache serialization and validation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Mapping
import zipfile

import numpy as np

from .builder import (
    BUILDER_CONTRACT_VERSION,
    PatientDCE7,
    VISITS,
    builder_contract_sha256,
    input_provenance_sha256,
)
from .dce7 import DCE7_CHANNEL_NAMES
from .geometry import (
    C1B_SHAPE_ZYX,
    C1B_SPACING_XYZ_MM,
    PhysicalGrid,
    validate_affine,
    validate_source_to_anchor_transform,
)


CONTENT_HASH_KEY = "cache_content_sha256"
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "patient_id",
        "cohort",
        "formal_ftv_overlap",
        "registration_strategy",
        "image",
        "valid_source_mask",
        "phase_indices",
        "phase_counts",
        "channel_names",
        "visits",
        "grid_affine_ras",
        "grid_center_ras_mm",
        "grid_shape_zyx",
        "grid_spacing_xyz_mm",
        "anchor_provenance",
        "normalization_p01",
        "normalization_p99",
        "normalization_median",
        "normalization_scale",
        "normalization_scale_source",
        "source_samples_per_output_axis",
        "anti_alias_sigma_source_voxels",
        "anti_alias_applied",
        "source_canonical_sha256",
        "support_canonical_sha256",
        "support_scope",
        "phase_metadata_sha256",
        "builder_contract_version",
        "builder_contract_sha256",
        "input_provenance_sha256",
        "source_to_anchor_ras",
        "support_available",
        "support_source_positive_voxels",
        "support_retained_positive_voxels",
        "support_nn_target_positive_voxels",
        "support_retained_positive_voxel_fraction",
        "support_physical_volume_retention",
        "support_exact_full_support_containment",
        "support_source_boundary_touch",
        "support_target_boundary_touch",
        "support_minimum_margin_mm",
        "support_source_volume_mm3",
        "support_retained_source_volume_mm3",
        "padding_mode",
        "intensity_interpolation",
        "support_interpolation",
        CONTENT_HASH_KEY,
    }
)


def _normalise_payload(payload: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not key or "/" in key or "\\" in key:
            raise ValueError(f"invalid cache key {key!r}")
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError(f"object arrays are forbidden in cache key {key!r}")
        arrays[key] = (
            np.ascontiguousarray(array) if array.ndim > 0 else np.asarray(array).copy()
        )
    return arrays


def content_sha256(payload: Mapping[str, np.ndarray]) -> str:
    """Hash sorted array schemas and bytes without copying the 337-MiB tensor."""

    arrays = _normalise_payload(payload)
    digest = hashlib.sha256()
    for key in sorted(name for name in arrays if name != CONTENT_HASH_KEY):
        name = key.encode("utf-8")
        array = arrays[key]
        dtype = array.dtype.str.encode("ascii")
        digest.update(len(name).to_bytes(8, "little"))
        digest.update(name)
        digest.update(len(dtype).to_bytes(8, "little"))
        digest.update(dtype)
        digest.update(array.ndim.to_bytes(8, "little"))
        for length in array.shape:
            digest.update(int(length).to_bytes(8, "little"))
        digest.update(array.nbytes.to_bytes(8, "little"))
        if array.nbytes:
            digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_deterministic_npz(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    arrays = _normalise_payload(payload)
    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_STORED,
        strict_timestamps=True,
    ) as archive:
        for key in sorted(arrays):
            info = zipfile.ZipInfo(filename=f"{key}.npy", date_time=_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            # Streaming bounds temporary memory while keeping the NPY member
            # and ZIP metadata deterministic.  force_zip64 also makes the
            # writer safe if a future tensor contract crosses 2 GiB.
            with archive.open(info, mode="w", force_zip64=True) as member:
                np.lib.format.write_array(member, arrays[key], allow_pickle=False)


@dataclass(frozen=True)
class CacheValidation:
    path: Path
    content_sha256: str
    file_sha256: str
    image_shape: tuple[int, ...]
    patient_id: str


def _scalar_text(array: np.ndarray, key: str) -> str:
    if np.asarray(array).shape != ():
        raise ValueError(f"cache key {key!r} must be a scalar string")
    return str(np.asarray(array).item())


def validate_cache_arrays(
    payload: Mapping[str, np.ndarray],
    *,
    require_frozen_grid: bool = True,
    expected_image: np.ndarray | None = None,
) -> str:
    """Validate schema, model/sidecar separation, geometry, and content hash."""

    arrays = _normalise_payload(payload)
    missing = sorted(_REQUIRED_KEYS.difference(arrays))
    if missing:
        raise ValueError(f"cache is missing required keys: {missing}")
    unexpected = sorted(set(arrays).difference(_REQUIRED_KEYS))
    if unexpected:
        raise ValueError(f"cache contains forbidden or unknown keys: {unexpected}")
    stored_hash = _scalar_text(arrays[CONTENT_HASH_KEY], CONTENT_HASH_KEY)
    computed_hash = content_sha256(arrays)
    if stored_hash != computed_hash:
        raise ValueError(
            f"cache content hash mismatch: stored {stored_hash}, computed {computed_hash}"
        )
    schema_version = arrays["schema_version"]
    if (
        schema_version.shape != ()
        or schema_version.dtype.kind not in "iu"
        or int(schema_version.item()) != 3
    ):
        raise ValueError("cache schema_version must be an integer scalar 3")
    patient_id = _scalar_text(arrays["patient_id"], "patient_id")
    if not patient_id.strip():
        raise ValueError("cache patient_id must be nonempty")
    cohort = _scalar_text(arrays["cohort"], "cohort")
    if cohort not in {"I-SPY1", "I-SPY2"}:
        raise ValueError("cache cohort must be I-SPY1 or I-SPY2")
    formal = arrays["formal_ftv_overlap"]
    if (
        formal.shape != ()
        or formal.dtype != np.dtype(np.uint8)
        or int(formal.item()) not in {0, 1}
    ):
        raise ValueError("formal FTV-overlap flag must be scalar uint8 0/1")
    formal_ftv_overlap = bool(formal.item())
    if formal_ftv_overlap and cohort != "I-SPY2":
        raise ValueError("formal FTV-overlap cache must belong to I-SPY2")
    registration_strategy = _scalar_text(
        arrays["registration_strategy"], "registration_strategy"
    )
    if registration_strategy not in {"C1B-H", "C1B-R"}:
        raise ValueError("cache registration strategy must be C1B-H or C1B-R")

    image = arrays["image"]
    if image.dtype != np.dtype(np.float32) or image.ndim != 5:
        raise ValueError(
            f"image must be float32 [4,7,Z,Y,X], got {image.dtype}/{image.shape}"
        )
    if image.shape[:2] != (len(VISITS), len(DCE7_CHANNEL_NAMES)):
        raise ValueError(f"image does not contain exactly T0-T3 x DCE7: {image.shape}")
    if not np.isfinite(image).all():
        raise ValueError("model image contains non-finite values")
    if np.any(image < -5.000001) or np.any(image > 5.000001):
        raise ValueError("model image violates the frozen [-5,5] output clip")
    if expected_image is not None and not np.array_equal(
        image, np.asarray(expected_image)
    ):
        raise ValueError("cache image failed exact round-trip comparison")

    shape_zyx = tuple(int(value) for value in arrays["grid_shape_zyx"].tolist())
    if len(shape_zyx) != 3 or image.shape[2:] != shape_zyx:
        raise ValueError("stored grid shape does not match model tensor")
    spacing = tuple(float(value) for value in arrays["grid_spacing_xyz_mm"].tolist())
    center = tuple(float(value) for value in arrays["grid_center_ras_mm"].tolist())
    grid = PhysicalGrid(shape_zyx, spacing, center)
    affine = validate_affine(arrays["grid_affine_ras"], name="cached grid affine")
    if not np.allclose(affine, grid.affine_ras, atol=1e-10, rtol=0.0):
        raise ValueError("cached grid affine is inconsistent with shape/spacing/centre")
    provenance = _scalar_text(arrays["anchor_provenance"], "anchor_provenance")
    if provenance not in {
        "released_t0_localization_support_bbox_center",
        "t0_acquisition_physical_center_fallback",
    }:
        raise ValueError(f"cache has non-T0 anchor provenance {provenance!r}")
    if require_frozen_grid:
        if shape_zyx != C1B_SHAPE_ZYX:
            raise ValueError(f"cache does not use frozen C1B shape {C1B_SHAPE_ZYX}")
        if not np.allclose(spacing, C1B_SPACING_XYZ_MM, atol=0.0, rtol=0.0):
            raise ValueError(
                f"cache does not use frozen C1B spacing {C1B_SPACING_XYZ_MM}"
            )

    valid = arrays["valid_source_mask"]
    if valid.dtype != np.dtype(np.uint8) or valid.shape != (4, 1, *shape_zyx):
        raise ValueError(
            "valid-source mask must be a separate uint8 [4,1,Z,Y,X] sidecar"
        )
    if not np.isin(valid, (0, 1)).all():
        raise ValueError("valid-source mask is not binary")
    if np.any(valid.reshape(4, -1).sum(axis=1) == 0):
        raise ValueError("at least one visit has no valid-source voxels")

    names = tuple(str(value) for value in arrays["channel_names"].tolist())
    if names != DCE7_CHANNEL_NAMES:
        raise ValueError(f"unexpected DCE7 channel names/order: {names}")
    visit_names = tuple(str(value) for value in arrays["visits"].tolist())
    if visit_names != VISITS:
        raise ValueError(f"unexpected visit names/order: {visit_names}")
    phase_counts = arrays["phase_counts"]
    phase_indices = arrays["phase_indices"]
    if phase_counts.shape != (4,) or phase_indices.shape != (4, 3):
        raise ValueError("phase-count/index sidecars have invalid shape")
    if phase_counts.dtype.kind not in "iu" or phase_indices.dtype.kind not in "iu":
        raise ValueError("phase-count/index sidecars must be integer arrays")
    if np.any(phase_counts < 1):
        raise ValueError("phase counts must be positive")
    if np.any(phase_indices < 0) or np.any(phase_indices >= phase_counts[:, None]):
        raise ValueError("phase index is outside its visit phase count")

    for key in (
        "normalization_p01",
        "normalization_p99",
        "normalization_median",
        "normalization_scale",
    ):
        values = arrays[key]
        if values.shape != (4, 7) or not np.isfinite(values).all():
            raise ValueError(f"{key} must be finite [4,7]")
    if np.any(arrays["normalization_scale"] <= 0):
        raise ValueError("normalization scales must be positive")
    if np.any(arrays["normalization_p01"] > arrays["normalization_p99"]):
        raise ValueError("normalization P01 cannot exceed P99")
    if arrays["normalization_scale_source"].shape != (4, 7):
        raise ValueError("normalization scale-source sidecar must be [4,7]")
    allowed_scale_sources = {
        "iqr_div_1.349",
        "std_fallback",
        "unit_constant_fallback",
    }
    if not set(
        str(value) for value in arrays["normalization_scale_source"].ravel()
    ).issubset(allowed_scale_sources):
        raise ValueError("normalization scale-source sidecar has an unknown method")
    for key in ("source_samples_per_output_axis", "anti_alias_sigma_source_voxels"):
        if arrays[key].shape != (4, 3) or not np.isfinite(arrays[key]).all():
            raise ValueError(f"{key} must be finite [4,3]")
    if arrays["anti_alias_applied"].shape != (4,):
        raise ValueError("anti-alias sidecar must be [4]")
    source_hashes = arrays["source_canonical_sha256"]
    if source_hashes.shape != (4,) or any(
        len(str(value)) != 64
        or any(character not in "0123456789abcdef" for character in str(value))
        for value in source_hashes
    ):
        raise ValueError("source canonical hashes must be four lowercase SHA-256 values")
    support_hashes = arrays["support_canonical_sha256"]
    if support_hashes.shape != (4,) or any(
        str(value) != "NONE"
        and (
            len(str(value)) != 64
            or any(character not in "0123456789abcdef" for character in str(value))
        )
        for value in support_hashes
    ):
        raise ValueError("support hashes must be four lowercase SHA-256 values or NONE")
    if arrays["support_scope"].shape != (4,):
        raise ValueError("support scope must have shape [4]")
    support_scopes = tuple(str(value) for value in arrays["support_scope"])
    if support_scopes[0] not in {
        "anchor_and_qc",
        "none_fallback",
    } or any(
        value not in {"formal_qc_only", "not_loaded"}
        for value in support_scopes[1:]
    ):
        raise ValueError("support scope violates the T0/later-visit contract")
    phase_metadata_hashes = arrays["phase_metadata_sha256"]
    if phase_metadata_hashes.shape != (4,) or any(
        len(str(value)) != 64
        or any(character not in "0123456789abcdef" for character in str(value))
        for value in phase_metadata_hashes
    ):
        raise ValueError("phase metadata hashes must be four lowercase SHA-256 values")
    contract_version = _scalar_text(
        arrays["builder_contract_version"], "builder_contract_version"
    )
    contract_digest = _scalar_text(
        arrays["builder_contract_sha256"], "builder_contract_sha256"
    )
    if (
        contract_version != BUILDER_CONTRACT_VERSION
        or contract_digest != builder_contract_sha256()
    ):
        raise ValueError("cache builder contract is stale or inconsistent")
    transforms = arrays["source_to_anchor_ras"]
    if transforms.shape != (4, 4, 4):
        raise ValueError("registration hook sidecar must be [4,4,4]")
    for index, transform in enumerate(transforms):
        validate_source_to_anchor_transform(transform)
        if index == 0 and not np.allclose(
            transform, np.eye(4), atol=1e-8, rtol=0.0
        ):
            raise ValueError("T0 registration transform must be identity")
    if registration_strategy == "C1B-H" and not np.allclose(
        transforms,
        np.repeat(np.eye(4, dtype=np.float64)[None], 4, axis=0),
        atol=1e-8,
        rtol=0.0,
    ):
        raise ValueError("C1B-H cache contains a non-identity transform")

    support_available = arrays["support_available"]
    if support_available.shape != (4,) or not np.isin(support_available, (0, 1)).all():
        raise ValueError("support availability sidecar must be binary [4]")
    support_full = arrays["support_source_positive_voxels"]
    support_retained = arrays["support_retained_positive_voxels"]
    support_nn = arrays["support_nn_target_positive_voxels"]
    for key, values in (
        ("support_source_positive_voxels", support_full),
        ("support_retained_positive_voxels", support_retained),
        ("support_nn_target_positive_voxels", support_nn),
    ):
        if values.shape != (4,) or values.dtype.kind not in "iu" or np.any(values < 0):
            raise ValueError(f"{key} must be nonnegative [4]")
    available = support_available.astype(bool)
    expected_available = np.asarray(
        [scope in {"anchor_and_qc", "formal_qc_only"} for scope in support_scopes],
        dtype=bool,
    )
    if not np.array_equal(available, expected_available):
        raise ValueError("support availability disagrees with support scope")
    expected_anchor_provenance = (
        "released_t0_localization_support_bbox_center"
        if support_scopes[0] == "anchor_and_qc"
        else "t0_acquisition_physical_center_fallback"
    )
    if provenance != expected_anchor_provenance:
        raise ValueError("T0 anchor provenance disagrees with support scope")
    if any(
        (str(digest) == "NONE") == bool(is_available)
        for digest, is_available in zip(support_hashes, available, strict=True)
    ):
        raise ValueError("support availability disagrees with support hash identity")
    if formal_ftv_overlap:
        if not available.all() or support_scopes[1:] != (
            "formal_qc_only",
            "formal_qc_only",
            "formal_qc_only",
        ):
            raise ValueError("formal FTV-overlap cache requires all four support QCs")
    elif available[1:].any() or support_scopes[1:] != (
        "not_loaded",
        "not_loaded",
        "not_loaded",
    ):
        raise ValueError("base-only cache must not load T1-T3 support")
    if np.any(support_full[available] < 1):
        raise ValueError("available support must contain at least one source voxel")
    if np.any(support_retained > support_full):
        raise ValueError("source-domain retained support cannot exceed full support")
    retained_fraction = arrays["support_retained_positive_voxel_fraction"]
    physical_retention = arrays["support_physical_volume_retention"]
    if retained_fraction.shape != (4,) or physical_retention.shape != (4,):
        raise ValueError("support retention sidecars must be [4]")
    expected_fraction = np.zeros(4, dtype=np.float64)
    expected_fraction[available] = support_retained[available].astype(
        np.float64
    ) / support_full[available].astype(np.float64)
    if not np.allclose(
        retained_fraction[available], expected_fraction[available], atol=1e-6, rtol=0.0
    ):
        raise ValueError(
            "retained support fraction is not source-domain voxel retention"
        )
    if not np.allclose(
        physical_retention[available], expected_fraction[available], atol=1e-6, rtol=0.0
    ):
        raise ValueError("physical support retention is not source-domain retention")
    if np.any(
        (physical_retention[available] < 0) | (physical_retention[available] > 1)
    ):
        raise ValueError("physical support retention must lie in [0,1]")
    support_exact = arrays["support_exact_full_support_containment"]
    if support_exact.shape != (4,) or not np.isin(support_exact, (0, 1)).all():
        raise ValueError("exact support-containment sidecar must be binary [4]")
    if not np.array_equal(
        support_exact[available].astype(bool),
        support_retained[available] == support_full[available],
    ):
        raise ValueError(
            "exact support-containment flag disagrees with source retention"
        )
    for key in ("support_source_boundary_touch", "support_target_boundary_touch"):
        if arrays[key].shape != (4,) or not np.isin(arrays[key], (0, 1)).all():
            raise ValueError(f"{key} must be binary [4]")
    for key in (
        "support_minimum_margin_mm",
        "support_source_volume_mm3",
        "support_retained_source_volume_mm3",
    ):
        if arrays[key].shape != (4,):
            raise ValueError(f"{key} must be [4]")
    if not np.isfinite(arrays["support_minimum_margin_mm"][available]).all():
        raise ValueError("available support must have finite physical margins")
    source_volume = arrays["support_source_volume_mm3"]
    retained_volume = arrays["support_retained_source_volume_mm3"]
    if np.any(source_volume[available] <= 0) or np.any(retained_volume[available] < 0):
        raise ValueError("available support physical volumes are invalid")
    if not np.allclose(
        retained_volume[available] / source_volume[available],
        expected_fraction[available],
        atol=1e-6,
        rtol=0.0,
    ):
        raise ValueError("source-domain support volumes disagree with voxel retention")

    stored_provenance_digest = _scalar_text(
        arrays["input_provenance_sha256"], "input_provenance_sha256"
    )
    expected_provenance_digest = input_provenance_sha256(
        patient_id=patient_id,
        cohort=cohort,
        formal_ftv_overlap=formal_ftv_overlap,
        registration_strategy=registration_strategy,
        source_hashes=source_hashes,
        support_hashes=support_hashes,
        support_scope=arrays["support_scope"],
        phase_metadata_hashes=phase_metadata_hashes,
        phase_counts=phase_counts,
        phase_indices=phase_indices,
        source_to_anchor_ras=transforms,
        grid=grid,
        anchor_provenance=provenance,
        contract_sha256=contract_digest,
    )
    if stored_provenance_digest != expected_provenance_digest:
        raise ValueError("cache input provenance digest is internally inconsistent")

    if _scalar_text(arrays["padding_mode"], "padding_mode") != "reflect":
        raise ValueError("cache padding mode is not the frozen reflect mode")
    if (
        _scalar_text(arrays["intensity_interpolation"], "intensity_interpolation")
        != "linear"
    ):
        raise ValueError("cache intensity interpolation must be linear")
    if (
        _scalar_text(arrays["support_interpolation"], "support_interpolation")
        != "nearest"
    ):
        raise ValueError("cache support interpolation must be nearest")
    return computed_hash


def load_and_validate_cache(
    path: str | Path,
    *,
    require_frozen_grid: bool = True,
    expected_image: np.ndarray | None = None,
    expected_file_sha256: str | None = None,
) -> tuple[dict[str, np.ndarray], CacheValidation]:
    """Reload a cache without pickle and validate hash plus exact tensor data."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    content_hash = validate_cache_arrays(
        arrays,
        require_frozen_grid=require_frozen_grid,
        expected_image=expected_image,
    )
    disk_hash = file_sha256(source)
    if expected_file_sha256 is not None and disk_hash != expected_file_sha256:
        raise ValueError(
            f"cache file hash mismatch: expected {expected_file_sha256}, got {disk_hash}"
        )
    validation = CacheValidation(
        path=source,
        content_sha256=content_hash,
        file_sha256=disk_hash,
        image_shape=tuple(int(value) for value in arrays["image"].shape),
        patient_id=_scalar_text(arrays["patient_id"], "patient_id"),
    )
    return arrays, validation


def write_cache_atomic(
    path: str | Path,
    patient: PatientDCE7 | Mapping[str, np.ndarray],
    *,
    require_frozen_grid: bool = True,
) -> CacheValidation:
    """Write a deterministic NPZ to a sibling temp file, validate, then replace."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    base_payload = (
        patient.cache_payload() if isinstance(patient, PatientDCE7) else dict(patient)
    )
    arrays = _normalise_payload(base_payload)
    arrays.pop(CONTENT_HASH_KEY, None)
    arrays[CONTENT_HASH_KEY] = np.asarray(content_sha256(arrays))

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        _write_deterministic_npz(temporary, arrays)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        # Validate the complete temporary artifact before it can replace an
        # existing cache.
        _, temporary_validation = load_and_validate_cache(
            temporary,
            require_frozen_grid=require_frozen_grid,
            expected_image=arrays["image"],
        )
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        # os.replace preserves the already reloaded/validated bytes.  Re-hash
        # the destination to verify the rename rather than allocating and
        # scanning a second 337-MiB reload.
        destination_hash = file_sha256(destination)
        if destination_hash != temporary_validation.file_sha256:
            raise ValueError(
                "cache bytes changed during atomic replacement: "
                f"{temporary_validation.file_sha256} -> {destination_hash}"
            )
        return CacheValidation(
            path=destination,
            content_sha256=temporary_validation.content_sha256,
            file_sha256=destination_hash,
            image_shape=temporary_validation.image_shape,
            patient_id=temporary_validation.patient_id,
        )
    finally:
        if temporary.exists():
            temporary.unlink()


def load_model_tensor(
    path: str | Path,
    *,
    require_frozen_grid: bool = True,
) -> np.ndarray:
    """Return only float32 DCE7; no mask, affine, support, or metadata can enter."""

    arrays, _ = load_and_validate_cache(path, require_frozen_grid=require_frozen_grid)
    return np.ascontiguousarray(arrays["image"], dtype=np.float32)

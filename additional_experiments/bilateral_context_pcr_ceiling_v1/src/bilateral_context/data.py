from __future__ import annotations

from pathlib import Path
from functools import lru_cache

import nibabel as nib
import numpy as np

from .contracts import ROOT, VISITS, source_path, v6_cache_path


def load_dce(patient_id: str, visit: str) -> tuple[np.ndarray, dict[str, object]]:
    path = source_path(patient_id, visit)
    if not path.is_file():
        raise FileNotFoundError(path)
    image = nib.load(str(path))
    canonical, repair = canonical_image(image)
    data = np.asarray(canonical.get_fdata(dtype=np.float32))
    if data.ndim != 4 or data.shape[-1] < 4 or not np.isfinite(data).all():
        raise ValueError(f"invalid raw DCE for {patient_id} {visit}: {data.shape}")
    # NIfTI is X,Y,Z,T; model code uses C,Z,Y,X.
    data = np.moveaxis(data, -1, 0)
    data = np.transpose(data, (0, 3, 2, 1))
    meta = {
        "path": str(path),
        "shape_xyzt": list(canonical.shape),
        "shape_czyx": list(data.shape),
        "axcodes": list(nib.aff2axcodes(canonical.affine)),
        "affine_sha256": __import__("hashlib").sha256(np.asarray(canonical.affine, dtype=np.float64).tobytes()).hexdigest(),
        "geometry_repair": repair,
    }
    return np.ascontiguousarray(data), meta


def canonical_image(image: nib.spatialimages.SpatialImage) -> tuple[nib.spatialimages.SpatialImage, str]:
    """Canonicalize without overwriting source; repair the known singular sform via qform."""
    try:
        return nib.as_closest_canonical(image), "none"
    except nib.orientations.OrientationError:
        qform, qcode = image.get_qform(coded=True)
        if qform is None or not np.isfinite(qform).all() or abs(float(np.linalg.det(qform[:3, :3]))) < 1e-8:
            raise
        repaired = nib.Nifti1Image(image.dataobj, qform, image.header.copy())
        repaired.set_qform(qform, int(qcode or 1))
        repaired.set_sform(qform, int(qcode or 1))
        return nib.as_closest_canonical(repaired), "sform_replaced_by_valid_qform_in_memory"


@lru_cache(maxsize=1024)
def phase_indices(patient_id: str) -> np.ndarray:
    """Use the already locked C1B phase contract; no phase is selected by outcome."""
    # Find the source cache through the private stage-B manifest.
    import pandas as pd
    manifest = ROOT.parents[1] / "additional_experiments/c1b_overlap_eligibility_ftv_stageb/manifests/stage_b_c1b_cache.private.csv"
    row = pd.read_csv(manifest, usecols=["patient_id", "cache_path"], dtype={"patient_id": str, "cache_path": str})
    match = row.loc[row["patient_id"].eq(str(patient_id))]
    if len(match) != 1:
        raise ValueError(f"missing locked C1B phase source: {patient_id}")
    with np.load(Path(match.iloc[0]["cache_path"]), allow_pickle=False) as payload:
        phase = np.asarray(payload["phase_indices"], dtype=np.int16)
    if phase.shape != (4, 3):
        raise ValueError("C1B phase contract is not [4,3]")
    return phase


def dce7_from_raw(raw: np.ndarray, phases: np.ndarray) -> np.ndarray:
    if raw.ndim != 5 or raw.shape[1] < 4 or phases.shape != (4, 3):
        raise ValueError("raw DCE contract failed")
    result: list[np.ndarray] = []
    for visit in range(4):
        pre_i, early_i, late_i = map(int, phases[visit])
        if max(pre_i, early_i, late_i) >= raw.shape[2]:
            raise ValueError("phase index exceeds raw DCE frames")
        visit_raw = raw[visit]
        pre = visit_raw[pre_i]
        early = visit_raw[early_i]
        late = visit_raw[late_i]
        peak = visit_raw[1:].max(axis=0)
        denominator = np.maximum(np.abs(pre), 1.0)
        channels = np.stack((pre, early, late, early - pre, late - pre, (peak - pre) / denominator, (late - peak) / denominator))
        normalized = np.stack([robust_normalize(x) for x in channels])
        result.append(normalized)
    output = np.stack(result).astype(np.float32)
    if output.shape[1] != 7 or not np.isfinite(output).all():
        raise ValueError("full-field DCE7 is invalid")
    return output


def dce7_visit_from_raw(raw_visit: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Construct one normalized DCE7 visit; spatial dimensions may vary by visit."""
    if raw_visit.ndim != 4 or raw_visit.shape[0] < 4 or phase.shape != (3,):
        raise ValueError("raw visit contract failed")
    pre_i, early_i, late_i = map(int, phase)
    if max(pre_i, early_i, late_i) >= raw_visit.shape[0]:
        raise ValueError("phase index exceeds raw visit frames")
    pre, early, late = raw_visit[pre_i], raw_visit[early_i], raw_visit[late_i]
    peak = raw_visit[1:].max(axis=0)
    denominator = np.maximum(np.abs(pre), 1.0)
    channels = np.stack((pre, early, late, early - pre, late - pre, (peak - pre) / denominator, (late - peak) / denominator))
    result = np.stack([robust_normalize(channel) for channel in channels]).astype(np.float32)
    if result.shape[0] != 7 or not np.isfinite(result).all():
        raise ValueError("raw visit DCE7 is invalid")
    return result


def robust_normalize(channel: np.ndarray) -> np.ndarray:
    finite = channel[np.isfinite(channel)]
    if finite.size == 0:
        raise ValueError("empty DCE channel")
    low, high = np.percentile(finite, (1.0, 99.0))
    clipped = np.clip(channel, low, high)
    median = float(np.median(clipped))
    q1, q3 = np.percentile(clipped, (25.0, 75.0))
    scale = float((q3 - q1) / 1.349)
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(np.std(clipped) + 1e-6)
    return np.clip((clipped - median) / scale, -5.0, 5.0).astype(np.float32)


def full_field_views(dce7: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two RAS left/right halves as [visit,side,channel,z,y,x]."""
    if dce7.shape[0:2] != (4, 7):
        raise ValueError("DCE7 shape must start [4,7]")
    _, _, z, y, x = dce7.shape
    mid = x // 2
    left = dce7[..., :mid]
    right = dce7[..., x - mid:]
    # Align both views to the same medial-to-lateral orientation.
    left = left[..., ::-1]
    return np.stack((left, right), axis=1), np.asarray([z, y, mid], dtype=np.int32)


def full_field_visit_views(dce7_visit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return one visit's left/right views as [2,7,z,y,x]."""
    if dce7_visit.ndim != 4 or dce7_visit.shape[0:1] != (7,):
        raise ValueError("visit DCE7 shape must be [7,z,y,x]")
    _, z, y, x = dce7_visit.shape
    mid = x // 2
    left = dce7_visit[..., :mid][..., ::-1]
    right = dce7_visit[..., x - mid:]
    return np.stack((left, right), axis=0), np.asarray([z, y, mid], dtype=np.int32)


def slice_indices(dce7: np.ndarray, count: int = 32) -> np.ndarray:
    z = dce7.shape[2]
    # Fixed source-only policy: use the central 80% of the acquired axial range.
    lo, hi = int(round(0.10 * (z - 1))), int(round(0.90 * (z - 1)))
    return np.rint(np.linspace(lo, hi, count)).astype(int)


def load_v6_state_reference(patient_id: str) -> np.ndarray:
    path = v6_cache_path(patient_id)
    with np.load(path, allow_pickle=False) as payload:
        state = np.asarray(payload["spatial_tokens"])
    return state

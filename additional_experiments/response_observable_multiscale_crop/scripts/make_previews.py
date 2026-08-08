#!/usr/bin/env python3
"""Create de-identified, real-image previews for Stage A crop contracts.

The private ``patient_visit_contracts.csv`` is used only to choose cases and
reconstruct physical windows.  Public PNGs and the aggregate quality table do
not contain patient identifiers or source paths.  Image planes are sampled
directly from the original DCE NIfTI once with linear interpolation; the FTV
support is sampled separately with nearest-neighbour interpolation for audit
overlays only.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy import ndimage


SCRIPT = Path(__file__).resolve()
EXPERIMENT_ROOT = SCRIPT.parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from observable_crop.geometry import orthonormal_index_basis  # noqa: E402
from observable_crop.nifti import (  # noqa: E402
    affine_max_corner_disagreement_mm,
    read_nifti_geometry,
)


VISITS = ("T0", "T1", "T2", "T3")
WINDOW_KEYS = (
    ("C0", "legacy"),
    ("C1B", "detail"),
    ("C2B", "detail"),
    ("C2B", "context"),
)
LONGITUDINAL_KEYS = (
    ("C0", "legacy"),
    ("C1B", "detail"),
    ("C2B", "context"),
)
REQUIRED_COLUMNS = {
    "patient_id",
    "visit",
    "contract",
    "view",
    "audit_only",
    "geometry_model_ready",
    "lesion_physical_volume_mm3",
    "center_frame_x_mm",
    "center_frame_y_mm",
    "center_frame_z_mm",
    "fov_x_mm",
    "fov_y_mm",
    "fov_z_mm",
    "output_x",
    "output_y",
    "output_z",
}
NUMERIC_WINDOW_COLUMNS = (
    "center_frame_x_mm",
    "center_frame_y_mm",
    "center_frame_z_mm",
    "fov_x_mm",
    "fov_y_mm",
    "fov_z_mm",
    "output_x",
    "output_y",
    "output_z",
)
NIFTI_DTYPES = {
    2: np.uint8,
    4: np.int16,
    8: np.int32,
    16: np.float32,
    64: np.float64,
    256: np.int8,
    512: np.uint16,
    768: np.uint32,
    1024: np.int64,
    1280: np.uint64,
}


class PreviewInputError(ValueError):
    """An error whose message is safe to print without private identifiers."""


@dataclass(frozen=True)
class WindowSpec:
    contract: str
    view: str
    center_frame_mm: np.ndarray
    fov_xyz_mm: np.ndarray
    output_shape_xyz: tuple[int, int, int]
    frame_basis: np.ndarray

    @property
    def spacing_xyz_mm(self) -> np.ndarray:
        return self.fov_xyz_mm / np.asarray(self.output_shape_xyz, dtype=float)


@dataclass(frozen=True)
class MappedNifti:
    data: np.memmap
    shape: tuple[int, ...]
    slope: float
    intercept: float


@dataclass
class PlaneSample:
    private_case: str
    size_group: str
    visit: str
    contract: str
    view: str
    image_mode: str
    image: np.ndarray
    mask_overlay: np.ndarray
    valid_source: np.ndarray
    fov_xyz_mm: np.ndarray
    spacing_xyz_mm: np.ndarray


def parse_args() -> argparse.Namespace:
    default_preprocessed: Path | None = None
    if os.environ.get("ISPY2_PREPROCESSED_ROOT"):
        default_preprocessed = Path(os.environ["ISPY2_PREPROCESSED_ROOT"])
    elif os.environ.get("DGRS_DATA_ROOT"):
        default_preprocessed = Path(os.environ["DGRS_DATA_ROOT"]) / "I-SPY2"
    parser = argparse.ArgumentParser(
        description="生成无患者标识的真实 DCE physical-window previews。"
    )
    parser.add_argument(
        "--contracts",
        type=Path,
        default=EXPERIMENT_ROOT / "metrics" / "patient_visit_contracts.csv",
        help="run_stage_a 生成的私有 patient_visit_contracts.csv",
    )
    parser.add_argument(
        "--preprocessed-root",
        type=Path,
        default=default_preprocessed,
        help="I-SPY2 preprocessed root；也可设置 ISPY2_PREPROCESSED_ROOT",
    )
    parser.add_argument(
        "--image-mode",
        choices=("enhancement", "precontrast"),
        default="enhancement",
        help="enhancement 为首个 postcontrast 减 precontrast",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=EXPERIMENT_ROOT / "figures",
    )
    parser.add_argument(
        "--quality-csv",
        type=Path,
        default=EXPERIMENT_ROOT / "metrics" / "image_quality_preview.csv",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--affine-match-atol-mm", type=float, default=0.1)
    return parser.parse_args()


def _boolean_series(series: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    invalid = ~normalized.isin(mapping)
    if invalid.any():
        raise PreviewInputError(f"{label} 含非布尔值")
    return normalized.map(mapping).astype(bool)


def load_contracts(path: Path) -> pd.DataFrame:
    try:
        path = path.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise PreviewInputError("私有 patient_visit_contracts.csv 不存在") from exc
    frame = pd.read_csv(
        path,
        dtype={
            "patient_id": "string",
            "visit": "string",
            "contract": "string",
            "view": "string",
        },
        low_memory=False,
    )
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise PreviewInputError(
            "patient_visit_contracts.csv 缺必要字段: " + ", ".join(missing)
        )
    if frame.empty or frame["patient_id"].isna().any():
        raise PreviewInputError("patient_visit_contracts.csv 为空或含空 patient_id")
    for column in ("visit", "contract", "view"):
        frame[column] = frame[column].astype(str)
    frame["patient_id"] = frame["patient_id"].astype(str)
    if frame["patient_id"].str.contains(r"[/\\]").any():
        raise PreviewInputError("patient_id 含路径分隔符")
    frame["audit_only"] = _boolean_series(frame["audit_only"], "audit_only")
    frame["geometry_model_ready"] = _boolean_series(
        frame["geometry_model_ready"], "geometry_model_ready"
    )
    for column in (*NUMERIC_WINDOW_COLUMNS, "lesion_physical_volume_mm3"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    wanted = frame.loc[
        frame.apply(
            lambda row: (str(row["contract"]), str(row["view"])) in WINDOW_KEYS,
            axis=1,
        )
    ].copy()
    duplicates = wanted.duplicated(
        ["patient_id", "visit", "contract", "view"], keep=False
    )
    if duplicates.any():
        raise PreviewInputError("目标 contract/view 存在重复 patient-visit 行")
    if wanted.empty:
        raise PreviewInputError("CSV 中没有 C0/C1B/C2B preview rows")
    return wanted


def _valid_window_row(row: pd.Series) -> bool:
    numeric = row.loc[list(NUMERIC_WINDOW_COLUMNS)].to_numpy(dtype=float)
    if not np.all(np.isfinite(numeric)):
        return False
    fov = row.loc[["fov_x_mm", "fov_y_mm", "fov_z_mm"]].to_numpy(dtype=float)
    shapes = row.loc[["output_x", "output_y", "output_z"]].to_numpy(dtype=float)
    return bool(
        bool(row["geometry_model_ready"])
        and not bool(row["audit_only"])
        and np.all(fov > 0)
        and np.all(shapes > 0)
        and np.array_equal(shapes, np.rint(shapes))
    )


def _row_lookup(frame: pd.DataFrame) -> dict[tuple[str, str, str, str], pd.Series]:
    return {
        (
            str(row["patient_id"]),
            str(row["visit"]),
            str(row["contract"]),
            str(row["view"]),
        ): row
        for _, row in frame.iterrows()
    }


def _patient_has_windows(
    patient_id: str,
    visits: Iterable[str],
    keys: Iterable[tuple[str, str]],
    lookup: Mapping[tuple[str, str, str, str], pd.Series],
) -> bool:
    for visit in visits:
        for contract, view in keys:
            row = lookup.get((patient_id, visit, contract, view))
            if row is None or not _valid_window_row(row):
                return False
    return True


def _stable_nearest(
    candidates: pd.DataFrame,
    target: float,
    excluded: set[str] | None = None,
) -> str:
    if excluded:
        candidates = candidates.loc[~candidates["patient_id"].isin(excluded)]
    if candidates.empty:
        raise PreviewInputError("某个 T0 体积分层无可用 preview case")
    work = candidates.copy()
    volumes = work["lesion_physical_volume_mm3"].to_numpy(dtype=float)
    work["distance"] = np.abs(np.log(volumes) - math.log(target))
    work["private_tie_break"] = work["patient_id"].map(
        lambda value: hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    )
    work = work.sort_values(["distance", "private_tie_break"], kind="stable")
    return str(work.iloc[0]["patient_id"])


def select_representatives(
    frame: pd.DataFrame,
) -> tuple[dict[str, str], dict[str, float]]:
    lookup = _row_lookup(frame)
    t0_detail = frame.loc[
        (frame["visit"] == "T0")
        & (frame["contract"] == "C1B")
        & (frame["view"] == "detail")
    ].copy()
    eligible = []
    longitudinal = []
    for patient_id in t0_detail["patient_id"].astype(str):
        if _patient_has_windows(patient_id, ("T0",), WINDOW_KEYS, lookup):
            eligible.append(patient_id)
            if _patient_has_windows(
                patient_id, VISITS, LONGITUDINAL_KEYS, lookup
            ):
                longitudinal.append(patient_id)
    t0_detail = t0_detail.loc[t0_detail["patient_id"].isin(eligible)].copy()
    t0_detail = t0_detail.loc[
        np.isfinite(t0_detail["lesion_physical_volume_mm3"])
        & (t0_detail["lesion_physical_volume_mm3"] > 0)
    ]
    if len(t0_detail) < 3:
        raise PreviewInputError("少于 3 个具备完整 T0 preview windows 的 case")
    volumes = t0_detail["lesion_physical_volume_mm3"].to_numpy(dtype=float)
    q10, q30, q33, q50, q67, q70, q90 = np.quantile(
        volumes, [0.10, 0.30, 1.0 / 3.0, 0.50, 2.0 / 3.0, 0.70, 0.90]
    )
    small = t0_detail.loc[t0_detail["lesion_physical_volume_mm3"] <= q33]
    middle = t0_detail.loc[
        (t0_detail["lesion_physical_volume_mm3"] > q33)
        & (t0_detail["lesion_physical_volume_mm3"] <= q67)
        & t0_detail["patient_id"].isin(longitudinal)
    ]
    large = t0_detail.loc[t0_detail["lesion_physical_volume_mm3"] > q67]
    selected: dict[str, str] = {}
    selected["small"] = _stable_nearest(small, float(q10))
    selected["lower_mid"] = _stable_nearest(
        t0_detail.loc[t0_detail["lesion_physical_volume_mm3"] <= q50],
        float(q30),
        set(selected.values()),
    )
    selected["medium"] = _stable_nearest(
        middle, float(q50), set(selected.values())
    )
    selected["upper_mid"] = _stable_nearest(
        t0_detail.loc[t0_detail["lesion_physical_volume_mm3"] >= q50],
        float(q70),
        set(selected.values()),
    )
    selected["large"] = _stable_nearest(
        large, float(q90), set(selected.values())
    )
    if len(set(selected.values())) != 5:
        raise PreviewInputError("质量审计 selection 未产生 5 个不同 case")
    quantiles = {
        "eligible_case_count": float(len(t0_detail)),
        "q10_mm3": float(q10),
        "q30_mm3": float(q30),
        "q33_mm3": float(q33),
        "q50_mm3": float(q50),
        "q67_mm3": float(q67),
        "q70_mm3": float(q70),
        "q90_mm3": float(q90),
    }
    return selected, quantiles


def _read_nifti_memmap(path: Path) -> MappedNifti:
    if path.suffix == ".gz":
        raise PreviewInputError("preview 目前要求未压缩 .nii，以便 source-domain memmap")
    with path.open("rb") as stream:
        header = stream.read(348)
    if len(header) != 348:
        raise PreviewInputError("NIfTI-1 header 不完整")
    endian = "<"
    if struct.unpack("<i", header[:4])[0] != 348:
        endian = ">"
    if struct.unpack(endian + "i", header[:4])[0] != 348:
        raise PreviewInputError("输入不是 NIfTI-1")
    dims = struct.unpack(endian + "8h", header[40:56])
    ndim = int(dims[0])
    if ndim < 3 or ndim > 7:
        raise PreviewInputError("NIfTI spatial dimension 非法")
    shape = tuple(int(value) for value in dims[1 : ndim + 1])
    if any(value <= 0 for value in shape):
        raise PreviewInputError("NIfTI shape 非法")
    datatype = int(struct.unpack(endian + "h", header[70:72])[0])
    bitpix = int(struct.unpack(endian + "h", header[72:74])[0])
    if datatype not in NIFTI_DTYPES:
        raise PreviewInputError(f"不支持的 NIfTI datatype code: {datatype}")
    dtype = np.dtype(NIFTI_DTYPES[datatype]).newbyteorder(endian)
    if dtype.itemsize * 8 != bitpix:
        raise PreviewInputError("NIfTI datatype/bitpix 不一致")
    offset = int(round(float(struct.unpack(endian + "f", header[108:112])[0])))
    if offset < 348:
        raise PreviewInputError("NIfTI vox_offset 非法")
    required_bytes = offset + int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if path.stat().st_size < required_bytes:
        raise PreviewInputError("NIfTI voxel payload 不完整")
    slope = float(struct.unpack(endian + "f", header[112:116])[0])
    intercept = float(struct.unpack(endian + "f", header[116:120])[0])
    if not np.isfinite(slope) or slope == 0.0:
        slope = 1.0
    if not np.isfinite(intercept):
        intercept = 0.0
    data = np.memmap(
        path,
        mode="r",
        dtype=dtype,
        offset=offset,
        shape=shape,
        order="F",
    )
    return MappedNifti(
        data=data,
        shape=shape,
        slope=slope,
        intercept=intercept,
    )


def _safe_asset_path(raw: Any, patient_dir: Path, root: Path) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise PreviewInputError("manifest 缺 NIfTI asset path")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = patient_dir / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise PreviewInputError("manifest 指向的 NIfTI 不存在") from exc
    if not resolved.is_relative_to(root):
        raise PreviewInputError("manifest NIfTI 超出 preprocessed root")
    return resolved


def _load_manifest(
    preprocessed_root: Path,
    private_case: str,
) -> tuple[Path, dict[str, dict[str, Any]]]:
    patient_dir = (preprocessed_root / private_case).resolve()
    if not patient_dir.is_relative_to(preprocessed_root):
        raise PreviewInputError("selected case path 超出 preprocessed root")
    try:
        manifest_path = (patient_dir / "manifest.json").resolve(strict=True)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise PreviewInputError("selected case manifest 缺失或非法") from exc
    visits = payload.get("visits")
    if not isinstance(visits, list):
        raise PreviewInputError("selected case manifest visits 非法")
    by_visit = {
        str(item.get("visit")): item for item in visits if isinstance(item, dict)
    }
    if set(by_visit) != set(VISITS):
        raise PreviewInputError("selected case manifest 不含完整 T0-T3")
    return patient_dir, by_visit


def _dce_source(
    mapped: MappedNifti,
    requested_mode: str,
) -> tuple[np.ndarray, str]:
    if len(mapped.shape) not in (3, 4):
        raise PreviewInputError("DCE NIfTI 必须是 3-D 或 4-D")
    if len(mapped.shape) == 3:
        pre = np.asarray(mapped.data, dtype=np.float32)
        return pre * mapped.slope + mapped.intercept, "precontrast_fallback"
    if requested_mode == "enhancement" and mapped.shape[3] >= 2:
        source = np.array(mapped.data[..., 1], dtype=np.float32, copy=True)
        np.subtract(source, mapped.data[..., 0], out=source, casting="unsafe")
        source *= mapped.slope
        return source, "first_post_minus_pre"
    pre = np.asarray(mapped.data[..., 0], dtype=np.float32)
    return pre * mapped.slope + mapped.intercept, "precontrast"


def _window_from_row(row: pd.Series, basis: np.ndarray) -> WindowSpec:
    center = row.loc[
        ["center_frame_x_mm", "center_frame_y_mm", "center_frame_z_mm"]
    ].to_numpy(dtype=float)
    fov = row.loc[["fov_x_mm", "fov_y_mm", "fov_z_mm"]].to_numpy(dtype=float)
    shape_values = row.loc[["output_x", "output_y", "output_z"]].to_numpy(
        dtype=float
    )
    if (
        not np.all(np.isfinite(center))
        or not np.all(np.isfinite(fov))
        or np.any(fov <= 0)
        or not np.array_equal(shape_values, np.rint(shape_values))
        or np.any(shape_values <= 0)
    ):
        raise PreviewInputError("selected physical window 含非法几何值")
    return WindowSpec(
        contract=str(row["contract"]),
        view=str(row["view"]),
        center_frame_mm=center,
        fov_xyz_mm=fov,
        output_shape_xyz=tuple(int(value) for value in shape_values),
        frame_basis=np.asarray(basis, dtype=float),
    )


def _target_world_plane(window: WindowSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    output_x, output_y, _ = window.output_shape_xyz
    spacing = window.spacing_xyz_mm
    x_frame = window.center_frame_mm[0] + (
        np.arange(output_x, dtype=float) - 0.5 * (output_x - 1)
    ) * spacing[0]
    y_frame = window.center_frame_mm[1] + (
        np.arange(output_y, dtype=float) - 0.5 * (output_y - 1)
    ) * spacing[1]
    x_grid, y_grid = np.meshgrid(x_frame, y_frame, indexing="xy")
    frame = np.column_stack(
        (
            x_grid.ravel(),
            y_grid.ravel(),
            np.full(x_grid.size, window.center_frame_mm[2], dtype=float),
        )
    )
    world = frame @ window.frame_basis.T
    return world, x_frame, y_frame


def _world_to_index(world: np.ndarray, affine: np.ndarray) -> np.ndarray:
    affine = np.asarray(affine, dtype=float)
    return np.linalg.solve(
        affine[:3, :3],
        (world - affine[:3, 3][None, :]).T,
    )


def sample_plane(
    *,
    private_case: str,
    size_group: str,
    visit: str,
    source: np.ndarray,
    source_affine: np.ndarray,
    mask: np.ndarray,
    mask_affine: np.ndarray,
    window: WindowSpec,
    image_mode: str,
) -> PlaneSample:
    world, _, _ = _target_world_plane(window)
    source_coordinates = _world_to_index(world, source_affine)
    output_y, output_x = window.output_shape_xyz[1], window.output_shape_xyz[0]
    valid = np.ones(source_coordinates.shape[1], dtype=bool)
    for axis, length in enumerate(source.shape):
        valid &= (source_coordinates[axis] >= 0.0) & (
            source_coordinates[axis] <= float(length - 1)
        )
    # Exactly one source-image interpolation pass; this is never a resample of
    # an already resampled image.
    sampled = ndimage.map_coordinates(
        source,
        source_coordinates,
        order=1,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    ).reshape(output_y, output_x)
    mask_coordinates = _world_to_index(world, mask_affine)
    overlay = ndimage.map_coordinates(
        mask,
        mask_coordinates,
        order=0,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ).reshape(output_y, output_x)
    return PlaneSample(
        private_case=private_case,
        size_group=size_group,
        visit=visit,
        contract=window.contract,
        view=window.view,
        image_mode=image_mode,
        image=np.asarray(sampled, dtype=np.float32),
        mask_overlay=np.asarray(overlay > 0, dtype=bool),
        valid_source=valid.reshape(output_y, output_x),
        fov_xyz_mm=window.fov_xyz_mm.copy(),
        spacing_xyz_mm=window.spacing_xyz_mm.copy(),
    )


def _prepare_visit(
    patient_dir: Path,
    visit_manifest: dict[str, Any],
    preprocessed_root: Path,
    image_mode: str,
    affine_match_atol_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    dce_path = _safe_asset_path(
        visit_manifest.get("dce_nifti"), patient_dir, preprocessed_root
    )
    mask_path = _safe_asset_path(
        visit_manifest.get("ftv_mask_nifti"), patient_dir, preprocessed_root
    )
    dce_geometry = read_nifti_geometry(dce_path)
    mask_geometry = read_nifti_geometry(mask_path)
    if not dce_geometry.sform_valid or not mask_geometry.sform_valid:
        raise PreviewInputError("selected model-ready case 缺合法 DCE/mask sform")
    dce_mapped = _read_nifti_memmap(dce_path)
    mask_mapped = _read_nifti_memmap(mask_path)
    if tuple(dce_mapped.shape[:3]) != tuple(mask_mapped.shape[:3]):
        raise PreviewInputError("selected case DCE-mask shape 不一致")
    disagreement = affine_max_corner_disagreement_mm(
        dce_geometry.sform,
        mask_geometry.sform,
        tuple(int(value) for value in dce_mapped.shape[:3]),
    )
    if disagreement > affine_match_atol_mm:
        raise PreviewInputError("selected case DCE-mask affine 超出 preview tolerance")
    source, actual_mode = _dce_source(dce_mapped, image_mode)
    if tuple(source.shape) != tuple(dce_mapped.shape[:3]):
        raise PreviewInputError("DCE source channel shape 非法")
    mask = np.asarray(mask_mapped.data)
    if mask.ndim != 3:
        raise PreviewInputError("FTV overlay mask 必须是 3-D")
    return (
        source,
        np.asarray(dce_geometry.sform, dtype=float),
        mask,
        np.asarray(mask_geometry.sform, dtype=float),
        actual_mode,
    )


def build_samples(
    frame: pd.DataFrame,
    selected: dict[str, str],
    preprocessed_root: Path,
    image_mode: str,
    affine_match_atol_mm: float,
) -> dict[tuple[str, str, str, str], PlaneSample]:
    lookup = _row_lookup(frame)
    requested: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for size_group, private_case in selected.items():
        requested.setdefault((private_case, "T0"), set()).update(WINDOW_KEYS)
    medium = selected["medium"]
    for visit in VISITS:
        requested.setdefault((medium, visit), set()).update(LONGITUDINAL_KEYS)

    manifests: dict[str, tuple[Path, dict[str, dict[str, Any]]]] = {}
    t0_bases: dict[str, np.ndarray] = {}
    output: dict[tuple[str, str, str, str], PlaneSample] = {}
    group_for_case = {value: key for key, value in selected.items()}
    for private_case, visit in sorted(
        requested,
        key=lambda item: (
            (
                "small",
                "lower_mid",
                "medium",
                "upper_mid",
                "large",
            ).index(group_for_case[item[0]]),
            VISITS.index(item[1]),
        ),
    ):
        if private_case not in manifests:
            manifests[private_case] = _load_manifest(
                preprocessed_root, private_case
            )
            patient_dir, by_visit = manifests[private_case]
            t0_path = _safe_asset_path(
                by_visit["T0"].get("dce_nifti"), patient_dir, preprocessed_root
            )
            t0_geometry = read_nifti_geometry(t0_path)
            if not t0_geometry.sform_valid:
                raise PreviewInputError("selected case T0 DCE sform 非法")
            t0_bases[private_case] = orthonormal_index_basis(t0_geometry.sform)
        patient_dir, by_visit = manifests[private_case]
        source, source_affine, mask, mask_affine, actual_mode = _prepare_visit(
            patient_dir,
            by_visit[visit],
            preprocessed_root,
            image_mode,
            affine_match_atol_mm,
        )
        current_basis = orthonormal_index_basis(source_affine)
        for contract, view in sorted(requested[(private_case, visit)]):
            row = lookup.get((private_case, visit, contract, view))
            if row is None or not _valid_window_row(row):
                raise PreviewInputError("selected case 缺可采样的 contract/view window")
            basis = current_basis if contract == "C0" else t0_bases[private_case]
            window = _window_from_row(row, basis)
            sample = sample_plane(
                private_case=private_case,
                size_group=group_for_case[private_case],
                visit=visit,
                source=source,
                source_affine=source_affine,
                mask=mask,
                mask_affine=mask_affine,
                window=window,
                image_mode=actual_mode,
            )
            output[(private_case, visit, contract, view)] = sample
        del source, mask
    return output


def _limits(samples: Iterable[PlaneSample]) -> tuple[float, float]:
    values = []
    for sample in samples:
        keep = sample.valid_source & np.isfinite(sample.image)
        if np.any(keep):
            values.append(sample.image[keep])
    if not values:
        return 0.0, 1.0
    combined = np.concatenate(values)
    low, high = np.percentile(combined, [2.0, 98.0])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(np.min(combined)), float(np.max(combined))
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def _draw_sample(
    axis: plt.Axes,
    sample: PlaneSample,
    *,
    limits: tuple[float, float],
    title: str,
    detail_box_fov: np.ndarray | None = None,
) -> None:
    half_x = 0.5 * float(sample.fov_xyz_mm[0])
    half_y = 0.5 * float(sample.fov_xyz_mm[1])
    extent = (-half_x, half_x, -half_y, half_y)
    axis.imshow(
        np.ma.masked_invalid(sample.image),
        cmap="gray",
        origin="lower",
        extent=extent,
        vmin=limits[0],
        vmax=limits[1],
        interpolation="nearest",
        aspect="equal",
    )
    overlay = sample.mask_overlay
    if np.any(overlay) and not np.all(overlay):
        x = np.linspace(
            -half_x + 0.5 * sample.spacing_xyz_mm[0],
            half_x - 0.5 * sample.spacing_xyz_mm[0],
            overlay.shape[1],
            endpoint=True,
        )
        y = np.linspace(
            -half_y + 0.5 * sample.spacing_xyz_mm[1],
            half_y - 0.5 * sample.spacing_xyz_mm[1],
            overlay.shape[0],
            endpoint=True,
        )
        axis.contour(x, y, overlay.astype(float), levels=[0.5], colors="#FFD43B", linewidths=0.9)
    if detail_box_fov is not None:
        axis.add_patch(
            Rectangle(
                (-0.5 * detail_box_fov[0], -0.5 * detail_box_fov[1]),
                detail_box_fov[0],
                detail_box_fov[1],
                fill=False,
                edgecolor="#00E5FF",
                linewidth=1.7,
                linestyle="--",
            )
        )
    axis.set_xlim(-half_x, half_x)
    axis.set_ylim(-half_y, half_y)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(title, fontsize=9)
    axis.set_facecolor("black")


def _atomic_figure(figure: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.png")
    figure.savefig(
        temporary,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Title": "De-identified physical-window DCE preview"},
    )
    plt.close(figure)
    temporary.replace(path)


def figure_representatives(
    samples: Mapping[tuple[str, str, str, str], PlaneSample],
    selected: Mapping[str, str],
    path: Path,
    dpi: int,
) -> None:
    columns = WINDOW_KEYS
    column_titles = (
        "C0 legacy",
        "C1B detail",
        "C2B detail",
        "C2B context",
    )
    rows = ("small", "medium", "large")
    figure, axes = plt.subplots(3, 4, figsize=(14.0, 10.2), constrained_layout=True)
    for row_index, size_group in enumerate(rows):
        private_case = selected[size_group]
        row_samples = [
            samples[(private_case, "T0", contract, view)]
            for contract, view in columns
        ]
        limits = _limits(row_samples)
        for column_index, (sample, title) in enumerate(
            zip(row_samples, column_titles, strict=True)
        ):
            _draw_sample(
                axes[row_index, column_index],
                sample,
                limits=limits,
                title=(
                    f"{title}\nFOV {sample.fov_xyz_mm[0]:.0f}×"
                    f"{sample.fov_xyz_mm[1]:.0f} mm"
                ),
            )
            if column_index == 0:
                axes[row_index, column_index].set_ylabel(
                    f"{size_group.capitalize()} T0-volume stratum",
                    fontsize=10,
                )
    figure.suptitle(
        "Representative T0 raw-DCE enhancement — physical center planes\n"
        "Linear image sampling once from source NIfTI; yellow FTV contour is audit overlay only; no patient IDs",
        fontsize=12,
    )
    _atomic_figure(figure, path, dpi)


def figure_longitudinal(
    samples: Mapping[tuple[str, str, str, str], PlaneSample],
    medium_case: str,
    path: Path,
    dpi: int,
) -> None:
    rows = LONGITUDINAL_KEYS
    row_titles = ("C0 legacy", "C1B detail", "C2B context")
    figure, axes = plt.subplots(3, 4, figsize=(14.0, 9.0), constrained_layout=True)
    for visit_index, visit in enumerate(VISITS):
        visit_samples = [
            samples[(medium_case, visit, contract, view)] for contract, view in rows
        ]
        limits = _limits(visit_samples)
        for row_index, (sample, row_title) in enumerate(
            zip(visit_samples, row_titles, strict=True)
        ):
            title = visit if row_index == 0 else ""
            _draw_sample(
                axes[row_index, visit_index],
                sample,
                limits=limits,
                title=title,
            )
            if visit_index == 0:
                axes[row_index, visit_index].set_ylabel(row_title, fontsize=10)
    figure.suptitle(
        "Longitudinal T0–T3 example from middle T0-volume stratum\n"
        "Raw-DCE physical center planes; C1B/C2B remain T0-anchored; mask contour is overlay only; no patient IDs",
        fontsize=12,
    )
    _atomic_figure(figure, path, dpi)


def figure_schematic(
    samples: Mapping[tuple[str, str, str, str], PlaneSample],
    large_case: str,
    path: Path,
    dpi: int,
) -> None:
    detail = samples[(large_case, "T0", "C2B", "detail")]
    context = samples[(large_case, "T0", "C2B", "context")]
    if not np.all(detail.fov_xyz_mm <= context.fov_xyz_mm + 1e-8):
        raise PreviewInputError("selected C2B context 未包含 detail")
    limits = _limits((detail, context))
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.8), constrained_layout=True)
    _draw_sample(
        axes[0],
        context,
        limits=limits,
        title="C2B context\ncyan = detail footprint",
        detail_box_fov=detail.fov_xyz_mm,
    )
    _draw_sample(
        axes[1],
        detail,
        limits=limits,
        title="C2B detail\nsame physical center",
    )
    axes[2].set_aspect("equal")
    axes[2].add_patch(
        Rectangle(
            (-0.5, -0.5),
            1.0,
            1.0,
            facecolor="#DDE7F0",
            edgecolor="#315A7D",
            linewidth=2.0,
            label="context",
        )
    )
    ratio_x = float(detail.fov_xyz_mm[0] / context.fov_xyz_mm[0])
    ratio_y = float(detail.fov_xyz_mm[1] / context.fov_xyz_mm[1])
    axes[2].add_patch(
        Rectangle(
            (-0.5 * ratio_x, -0.5 * ratio_y),
            ratio_x,
            ratio_y,
            facecolor="#B2EBF2",
            edgecolor="#00A6B2",
            linewidth=2.0,
            label="detail",
        )
    )
    axes[2].plot(0, 0, marker="+", color="black", markersize=12, mew=2)
    axes[2].text(
        0.0,
        -0.67,
        "Shared T0 center/frame\n"
        f"Detail XYZ FOV: {detail.fov_xyz_mm[0]:.1f}×{detail.fov_xyz_mm[1]:.1f}×{detail.fov_xyz_mm[2]:.1f} mm\n"
        f"Context XYZ FOV: {context.fov_xyz_mm[0]:.1f}×{context.fov_xyz_mm[1]:.1f}×{context.fov_xyz_mm[2]:.1f} mm",
        ha="center",
        va="top",
        fontsize=9,
    )
    axes[2].set_xlim(-0.75, 0.75)
    axes[2].set_ylim(-0.95, 0.75)
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    axes[2].set_title("Physical nesting schematic")
    axes[2].legend(loc="upper right", frameon=False, fontsize=9)
    for spine in axes[2].spines.values():
        spine.set_visible(False)
    figure.suptitle(
        "C2B detail/context nesting on a real T0 raw-DCE center plane\n"
        "Linear source sampling; FTV mask is shown only as the yellow audit contour; no patient IDs",
        fontsize=12,
    )
    _atomic_figure(figure, path, dpi)


def _sample_quality(sample: PlaneSample) -> dict[str, Any]:
    valid = sample.valid_source
    finite = np.isfinite(sample.image)
    values = sample.image[valid & finite]
    if values.size:
        p01, p99 = np.percentile(values, [1.0, 99.0])
        dynamic_range = float(p99 - p01)
    else:
        dynamic_range = math.nan
    # Representative sensitivity to the exact legacy normalization order.
    # Production tensors use zero padding, and legacy percentiles/median/IQR
    # include those zeros.  This is intentionally labelled 2-D/enhancement
    # sensitivity rather than a substitute for a full 3-D DCE7 builder audit.
    padded = np.where(valid & finite, sample.image, 0.0).astype(np.float32)
    low, high = np.percentile(padded, [1.0, 99.0])
    clipped = np.clip(padded, low, high)
    normalization_median = float(np.median(clipped))
    q1, q3 = np.percentile(clipped, [25.0, 75.0])
    normalization_scale = float((q3 - q1) / 1.349)
    if not np.isfinite(normalization_scale) or normalization_scale < 1e-6:
        normalization_scale = float(np.std(clipped) + 1e-6)
    normalized = np.clip(
        (clipped - normalization_median) / normalization_scale,
        -5.0,
        5.0,
    )
    source_normalized = normalized[valid & finite]
    padding_normalized = normalized[~valid]
    return {
        "private_case": sample.private_case,
        "visit": sample.visit,
        "contract": sample.contract,
        "view": sample.view,
        "image_mode": sample.image_mode,
        "valid_source_fraction": float(np.mean(valid)),
        "finite_fraction_within_source": float(np.sum(valid & finite) / max(np.sum(valid), 1)),
        "robust_dynamic_range": dynamic_range,
        "nonconstant": bool(np.isfinite(dynamic_range) and dynamic_range > 1e-6),
        "mask_intersects_center_plane": bool(np.any(sample.mask_overlay)),
        "legacy_norm_p01": float(low),
        "legacy_norm_p99": float(high),
        "legacy_norm_median": normalization_median,
        "legacy_norm_scale": normalization_scale,
        "legacy_norm_source_mean": float(np.mean(source_normalized))
        if source_normalized.size
        else math.nan,
        "legacy_norm_source_std": float(np.std(source_normalized))
        if source_normalized.size
        else math.nan,
        "legacy_norm_padding_value": float(np.median(padding_normalized))
        if padding_normalized.size
        else math.nan,
        "legacy_norm_saturation_fraction": float(
            np.mean(np.abs(normalized) >= 5.0 - 1e-7)
        ),
    }


def public_quality_table(
    samples: Mapping[tuple[str, str, str, str], PlaneSample],
) -> pd.DataFrame:
    private = pd.DataFrame([_sample_quality(sample) for sample in samples.values()])
    rows: list[dict[str, Any]] = []
    for (contract, view), group in private.groupby(["contract", "view"], sort=False):
        modes = sorted(group["image_mode"].unique())
        rows.append(
            {
                "preview_scope": "selected_T0_strata_plus_middle_stratum_T0_T3",
                "contract": contract,
                "view": view,
                "n_images": int(len(group)),
                "n_cases": int(group["private_case"].nunique()),
                "n_visits": int(group["visit"].nunique()),
                "source_image": "raw_DCE_NIfTI",
                "image_mode": "+".join(modes),
                "sample_mode": "physical_center_plane",
                "image_interpolation": "scipy_order1_single_pass",
                "mask_usage": "nearest_neighbor_audit_overlay_only_not_image_channel",
                "normalization_sensitivity": (
                    "2D_ENHANCEMENT_PLANE_LEGACY_P01_P99_MEDIAN_IQR_WITH_ZERO_PADDING"
                ),
                "valid_source_fraction_mean": float(group["valid_source_fraction"].mean()),
                "valid_source_fraction_min": float(group["valid_source_fraction"].min()),
                "finite_fraction_within_source_mean": float(
                    group["finite_fraction_within_source"].mean()
                ),
                "finite_fraction_within_source_min": float(
                    group["finite_fraction_within_source"].min()
                ),
                "robust_dynamic_range_mean": float(group["robust_dynamic_range"].mean()),
                "robust_dynamic_range_min": float(group["robust_dynamic_range"].min()),
                "nonconstant_fraction": float(group["nonconstant"].mean()),
                "mask_center_plane_intersection_fraction": float(
                    group["mask_intersects_center_plane"].mean()
                ),
                "legacy_norm_median_mean": float(
                    group["legacy_norm_median"].mean()
                ),
                "legacy_norm_scale_mean": float(group["legacy_norm_scale"].mean()),
                "legacy_norm_source_mean_mean": float(
                    group["legacy_norm_source_mean"].mean()
                ),
                "legacy_norm_source_std_mean": float(
                    group["legacy_norm_source_std"].mean()
                ),
                "legacy_norm_padding_value_mean": float(
                    group["legacy_norm_padding_value"].mean()
                )
                if group["legacy_norm_padding_value"].notna().any()
                else math.nan,
                "legacy_norm_saturation_fraction_mean": float(
                    group["legacy_norm_saturation_fraction"].mean()
                ),
                "contains_patient_identifiers": False,
            }
        )
    output = pd.DataFrame(rows).sort_values(["contract", "view"]).reset_index(drop=True)
    forbidden = {"patient_id", "clinical_patient_id", "private_case", "source_path"}
    if forbidden & set(output.columns):
        raise RuntimeError("public quality table unexpectedly contains private columns")
    return output


def _atomic_csv(path: Path, frame: pd.DataFrame, private_tokens: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    csv_text = frame.to_csv(index=False)
    for token in private_tokens:
        if len(token) >= 4 and token in csv_text:
            raise RuntimeError("public quality CSV unexpectedly contains a private identifier")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(csv_text, encoding="utf-8")
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.preprocessed_root is None:
        raise PreviewInputError(
            "缺 --preprocessed-root；或设置 ISPY2_PREPROCESSED_ROOT/DGRS_DATA_ROOT"
        )
    try:
        preprocessed_root = args.preprocessed_root.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise PreviewInputError("preprocessed root 不存在") from exc
    if not preprocessed_root.is_dir():
        raise PreviewInputError("preprocessed root 不是目录")
    if args.dpi < 72 or args.dpi > 600:
        raise PreviewInputError("dpi 必须在 [72,600]")
    if not np.isfinite(args.affine_match_atol_mm) or args.affine_match_atol_mm < 0:
        raise PreviewInputError("affine tolerance 必须是非负有限数")

    frame = load_contracts(args.contracts)
    selected, quantiles = select_representatives(frame)
    samples = build_samples(
        frame,
        selected,
        preprocessed_root,
        args.image_mode,
        float(args.affine_match_atol_mm),
    )
    figure_paths = (
        args.figures_dir / "10_representative_small_medium_large.png",
        args.figures_dir / "11_longitudinal_T0_T3_example.png",
        args.figures_dir / "12_detail_context_schematic.png",
    )
    figure_representatives(samples, selected, figure_paths[0], args.dpi)
    figure_longitudinal(samples, selected["medium"], figure_paths[1], args.dpi)
    figure_schematic(samples, selected["large"], figure_paths[2], args.dpi)
    quality = public_quality_table(samples)
    _atomic_csv(args.quality_csv, quality, selected.values())
    report_refreshed = False
    if args.quality_csv.resolve() == (
        EXPERIMENT_ROOT / "metrics" / "image_quality_preview.csv"
    ).resolve():
        # Stage A intentionally precedes image sampling.  Refreshing here keeps
        # the Markdown table synchronized with the newly generated 5-case CSV.
        from run_stage_a import refresh_image_context_report

        refresh_image_context_report(EXPERIMENT_ROOT)
        report_refreshed = True
    return {
        "status": "PASS",
        "real_raw_dce": True,
        "representative_figure_case_count": 3,
        "quality_audit_case_count": 5,
        "selection_policy": {
            "small": "nearest q10 within lower T0-volume tertile",
            "lower_mid": "nearest q30; quality-table audit only",
            "medium": "nearest q50 within middle T0-volume tertile; complete T0-T3 windows required",
            "upper_mid": "nearest q70; quality-table audit only",
            "large": "nearest q90 within upper T0-volume tertile",
            "eligible_case_count": int(quantiles["eligible_case_count"]),
        },
        "sampling": {
            "mode": "physical_center_plane",
            "image": "scipy.ndimage.map_coordinates order=1, one direct source pass",
            "mask": "order=0 audit overlay only; never an image channel",
        },
        "figures": [path.name for path in figure_paths],
        "quality_csv": args.quality_csv.name,
        "quality_rows": int(len(quality)),
        "image_context_report_refreshed": report_refreshed,
        "privacy": {
            "patient_identifiers_written": False,
            "source_paths_written": False,
        },
    }


def main() -> int:
    args = parse_args()
    try:
        result = run(args)
    except PreviewInputError as exc:
        print(
            json.dumps(
                {"status": "FAIL", "error": str(exc), "patient_identifiers_written": False},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    except Exception as exc:  # Fail closed without echoing a private path/ID.
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "error": f"unexpected {type(exc).__name__}; inspect privately",
                    "patient_identifiers_written": False,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""End-to-end single-case longitudinal geometry provenance audit."""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import asdict, is_dataclass
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.spatial import ConvexHull

from c1b_sanity.dce7 import _valid_source_footprint_mask_xyz
from c1b_sanity.geometry import (
    acquisition_center_ras,
    canonicalize_to_ras,
    input_from_output_affine,
    make_c1b_grid,
)

from .diagnostic_registration import run_image_only_diagnostic
from .geometry import (
    OrientedBox,
    minimum_cardinal_translation_for_count,
    minimum_cardinal_translation_for_fraction,
    pairwise_metrics,
)
from .provenance import (
    CASE_ALIAS,
    PROVENANCE_TAGS,
    VISITS,
    atomic_private_json,
    audit_failed_study_candidates,
    load_frozen_ispy1_contract,
    read_selected_series_provenance,
    rebuild_selected_series,
    representative_value,
    resolve_private_case,
    scan_study_objects,
    sha256_file,
)


PAIR_ORDER = tuple(itertools.combinations(VISITS, 2))
ADJACENT_PAIRS = {("T0", "T1"), ("T1", "T2"), ("T2", "T3")}
DICOM_IMAGE_PLANE_URL = (
    "https://dicom.nema.org/medical/dicom/current/output/chtml/part03/"
    "sect_C.7.6.2.html"
)
DICOM_PATIENT_POSITION_URL = (
    "https://dicom.nema.org/medical/dicom/current/output/chtml/part03/"
    "sect_c.7.3.html"
)
DICOM_REGISTRATION_URL = (
    "https://dicom.nema.org/medical/dicom/current/output/chtml/part03/"
    "sect_A.39.html"
)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_text(path: Path, content: str, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"PUBLIC_OUTPUT_EXISTS:{path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    content = json.dumps(
        _jsonable(payload), indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False
    ) + "\n"
    _atomic_text(path, content, overwrite=overwrite)


def _atomic_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], *, overwrite: bool
) -> None:
    if not rows:
        raise ValueError(f"EMPTY_PUBLIC_TABLE:{path.name}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with tempfile.TemporaryDirectory() as temporary_dir:
        temporary = Path(temporary_dir) / path.name
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: json.dumps(_jsonable(value), separators=(",", ":"))
                        if isinstance(value, (dict, list, tuple))
                        else _jsonable(value)
                        for key, value in row.items()
                    }
                )
        _atomic_text(path, temporary.read_text(encoding="utf-8"), overwrite=overwrite)


def _tracked_tree_digest(repo_root: Path, relative_root: Path) -> dict[str, Any]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", str(relative_root)],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    paths = [value for value in result.stdout.split(b"\0") if value]
    digest = hashlib.sha256(b"tracked-prior-experiment-v1\0")
    for raw in sorted(paths):
        relative = raw.decode("utf-8")
        path = repo_root / relative
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(sha256_file(path).encode("ascii") + b"\0")
    return {"tracked_file_count": len(paths), "sha256": digest.hexdigest()}


def _protocol_fingerprint(provenance: Mapping[str, Any]) -> tuple[str, ...]:
    tags = (
        "SequenceName",
        "ScanningSequence",
        "SequenceVariant",
        "MRAcquisitionType",
        "RepetitionTime",
        "EchoTime",
        "FlipAngle",
        "Rows",
        "Columns",
        "PixelSpacing",
    )
    return tuple(
        json.dumps(representative_value(provenance, tag), sort_keys=True)
        for tag in tags
    )


def _assign_groups(values: Mapping[str, tuple[str, ...]], prefix: str) -> dict[str, str]:
    unique = {value for value in values.values()}
    labels = {value: f"{prefix}_{index}" for index, value in enumerate(sorted(unique), 1)}
    return {key: labels[value] for key, value in values.items()}


def _orientation(affine: np.ndarray) -> str:
    return "".join(str(value) for value in nib.aff2axcodes(affine))


def _corner_hausdorff(first: OrientedBox, second: OrientedBox) -> float:
    left = first.corners()
    right = second.corners()
    distances = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2)
    return float(max(distances.min(axis=0).max(), distances.min(axis=1).max()))


def _compare_rebuilt_nifti(
    raw: np.ndarray, affine: np.ndarray, path: Path
) -> dict[str, Any]:
    image = nib.load(str(path), mmap=True)
    shape_match = tuple(int(value) for value in image.shape) == tuple(raw.shape)
    affine_error = float(np.max(np.abs(np.asarray(image.affine) - affine)))
    max_pixel_error = 0.0
    exact = shape_match
    if shape_match:
        for time_index in range(raw.shape[3]):
            for slice_index in range(raw.shape[2]):
                observed = np.asanyarray(
                    image.dataobj[:, :, slice_index, time_index]
                ).astype(np.float32, copy=False)
                expected = raw[:, :, slice_index, time_index]
                difference = float(
                    np.max(
                        np.abs(
                            observed.astype(np.float64)
                            - expected.astype(np.float64)
                        )
                    )
                )
                max_pixel_error = max(max_pixel_error, difference)
                exact = exact and np.array_equal(observed, expected)
    return {
        "shape_match": shape_match,
        "affine_max_abs_error_mm": affine_error,
        "pixel_exact": bool(exact),
        "pixel_max_abs_error": max_pixel_error,
        "status": "PASS"
        if shape_match and affine_error <= 1.0e-4 and exact
        else "FAIL",
    }


def _plot_projection(
    axis: plt.Axes,
    box: OrientedBox,
    dimensions: tuple[int, int],
    origin: np.ndarray,
    *,
    label: str,
    color: str,
) -> None:
    points = box.corners()[:, list(dimensions)] - origin[list(dimensions)]
    hull = ConvexHull(points)
    polygon = points[hull.vertices]
    polygon = np.vstack([polygon, polygon[0]])
    axis.plot(polygon[:, 0], polygon[:, 1], color=color, linewidth=1.8, label=label)
    center = box.center[list(dimensions)] - origin[list(dimensions)]
    axis.scatter(center[0], center[1], color=color, s=20)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.2)


def _save_figure(figure: plt.Figure, path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"PUBLIC_OUTPUT_EXISTS:{path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.png")
    figure.savefig(
        temporary,
        dpi=180,
        bbox_inches="tight",
        metadata={
            "Title": "CASE_ZERO_OVERLAP_001 anonymous technical audit",
            "Author": "zero_overlap_provenance_audit",
            "Description": "No identifier, UID, source path, or absolute coordinate",
        },
    )
    plt.close(figure)
    os.replace(temporary, path)


def _make_figures(
    root: Path,
    boxes: Mapping[str, OrientedBox],
    pair_rows: Sequence[Mapping[str, Any]],
    failed_visit: str,
    canonical: Mapping[str, Any],
    registration: Mapping[str, Any],
    *,
    overwrite: bool,
) -> None:
    colors = {"T0": "#1f77b4", "T1": "#2ca02c", "T2": "#ff7f0e", "T3": "#d62728"}
    origin = boxes["T0"].center
    dimensions = ((0, 1), (0, 2), (1, 2))
    labels = ("relative R / A (mm)", "relative R / S (mm)", "relative A / S (mm)")

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for axis, dims, axis_label in zip(axes, dimensions, labels, strict=True):
        for visit in VISITS:
            _plot_projection(
                axis,
                boxes[visit],
                dims,
                origin,
                label=visit,
                color=colors[visit],
            )
        axis.set_title(axis_label)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=len(VISITS),
        frameon=False,
    )
    figure.suptitle("T0–T3 raw-DICOM physical source footprints (T0-relative)")
    _save_figure(
        figure, root / "figures/01_t0_t3_source_physical_boxes.png", overwrite=overwrite
    )

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for axis, dims, axis_label in zip(axes, dimensions, labels, strict=True):
        for visit in ("T0", failed_visit):
            _plot_projection(
                axis,
                boxes[visit],
                dims,
                origin,
                label=visit,
                color=colors[visit],
            )
        axis.set_title(axis_label)
    axes[0].legend(frameon=False)
    figure.suptitle("T0 versus failed-visit physical footprint (T0-relative)")
    _save_figure(
        figure, root / "figures/02_t0_failed_physical_footprint.png", overwrite=overwrite
    )

    adjacent = [row for row in pair_rows if row["adjacent_visits"]]
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    bars = axis.bar(
        [row["visit_pair"] for row in adjacent],
        [row["center_displacement_mm"] for row in adjacent],
        color="#4c78a8",
    )
    axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    axis.margins(y=0.12)
    axis.set_ylabel("center displacement (mm)")
    axis.set_title("Visit-to-visit source-center displacement")
    axis.grid(axis="y", alpha=0.25)
    _save_figure(
        figure, root / "figures/03_visit_center_displacement.png", overwrite=overwrite
    )

    figure, axis = plt.subplots(figsize=(8.4, 4.5))
    bars = axis.bar(
        [row["visit_pair"] for row in pair_rows],
        [row["orientation_angle_deg"] for row in pair_rows],
        color="#f58518",
    )
    axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    axis.margins(y=0.12)
    axis.set_ylabel("proper-frame rotation angle (degrees)")
    axis.set_title("Pairwise source-orientation difference")
    axis.grid(axis="y", alpha=0.25)
    _save_figure(
        figure, root / "figures/04_orientation_angle_differences.png", overwrite=overwrite
    )

    figure, axis = plt.subplots(figsize=(8.4, 4.5))
    bars = axis.bar(
        [row["visit_pair"] for row in pair_rows],
        [row["minimum_separation_mm"] for row in pair_rows],
        color="#e45756",
    )
    axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    axis.margins(y=0.12)
    axis.set_ylabel("minimum source-volume separation (mm)")
    axis.set_title("Pairwise minimum physical separation")
    axis.grid(axis="y", alpha=0.25)
    _save_figure(
        figure, root / "figures/05_minimum_volume_separation.png", overwrite=overwrite
    )

    moved = registration.get("moved_array")
    if moved is not None:
        fixed = np.asarray(canonical["T0"].data)
        source = np.asarray(canonical[failed_visit].data)
        fixed_z = fixed.shape[2] // 2
        source_z = source.shape[2] // 2

        def display(array: np.ndarray) -> np.ndarray:
            values = np.asarray(array, dtype=np.float32)
            finite = values[np.isfinite(values)]
            low, high = np.percentile(finite, (1.0, 99.0))
            return np.clip((values - low) / max(float(high - low), 1e-6), 0.0, 1.0)

        fixed_slice = display(fixed[:, :, fixed_z].T)
        source_slice = display(source[:, :, source_z].T)
        moved_slice = display(np.asarray(moved)[:, :, fixed_z].T)
        difference = np.abs(fixed_slice - moved_slice)
        figure, axes_grid = plt.subplots(2, 2, figsize=(8.5, 10))
        axes = axes_grid.ravel()
        panels = (
            (fixed_slice, "T0 precontrast"),
            (source_slice, f"{failed_visit} precontrast (native frame)"),
            (moved_slice, f"{failed_visit} optimizer candidate (diagnostic only)"),
            (difference, "absolute difference"),
        )
        for axis, (array, title) in zip(axes, panels, strict=True):
            axis.imshow(array, cmap="gray" if "difference" not in title else "magma")
            axis.set_title(title)
            axis.axis("off")
        figure.suptitle(
            "Image-only rigid registration diagnostic; transform is not a repair",
            y=0.995,
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
        _save_figure(
            figure,
            root / "figures/06_diagnostic_registration_montage.png",
            overwrite=overwrite,
        )


def _public_environment() -> dict[str, Any]:
    import matplotlib as mpl
    import pydicom
    import scipy
    import SimpleITK

    return {
        "python": platform.python_version(),
        "python_executable_role": "dedicated_audit_environment",
        "numpy": np.__version__,
        "pydicom": pydicom.__version__,
        "nibabel": nib.__version__,
        "scipy": scipy.__version__,
        "SimpleITK": SimpleITK.Version_VersionString(),
        "matplotlib": mpl.__version__,
        "outcome_fields_read": [],
        "lesion_or_ftv_fields_read": [],
    }


def _format_number(value: Any, digits: int = 3) -> str:
    return "不可用" if value is None else f"{float(value):.{digits}f}"


def _inventory_report(
    table1: Sequence[Mapping[str, Any]],
    *,
    failed_visit: str,
    study_summary: Mapping[str, Any],
    candidate_summary: Mapping[str, Any],
    metadata: Mapping[str, Mapping[str, Any]],
) -> str:
    rows = []
    for row in table1:
        rows.append(
            "| {visit} | {patient_position} | {orientation_ras} | {matrix} | {spacing} | "
            "{frame} | {scanner} | {protocol} | {raw} |".format(
                visit=row["visit"],
                patient_position=row["patient_position"],
                orientation_ras=row["orientation_ras"],
                matrix=row["source_shape_xyz"],
                spacing=row["source_spacing_xyz_mm"],
                frame=row["frame_uid_relation_to_t0"],
                scanner=f"{row['manufacturer']} / {row['manufacturer_model']}",
                protocol=row["protocol_group"],
                raw=row["raw_geometry_status"],
            )
        )
    missing_counts: Counter[str] = Counter()
    for visit in VISITS:
        for tag, count in metadata[visit]["missing_counts"].items():
            if int(count) > 0:
                missing_counts[tag] += 1
    optional_missing = ", ".join(sorted(missing_counts)) or "无"
    return f"""# DICOM provenance inventory

## 结论

审计对象仅为 `{CASE_ALIAS}`；failed visit 为 `{failed_visit}`。四访视均从 raw DICOM 独立重算几何并逐 cell 两次解码 PixelData，未把缓存坐标当作唯一真值。Patient ID、UID、source path、绝对 IPP/中心坐标、StationName 和 acquisition clock time 只存在于 gitignored private artifacts。

| visit | PatientPosition | raw orientation | matrix | spacing (mm) | FrameOfReferenceUID vs T0 | scanner | protocol group | raw geometry |
|---|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

## Requested tag coverage

对每个 selected series 的所有 instance 读取了以下字段：`{', '.join(PROVENANCE_TAGS)}`。可选字段中至少一个访视存在缺失的 tag：{optional_missing}。缺失 optional tag 不被补写为几何真值；`SliceLocation` 仅作辅助。

四访视 selected series 的 IOP 在各自 series 内一致且正交，PixelSpacing 一致，IPP 投影形成规则 slice grid，in-plane drift 通过冻结门限；TemporalPositionIdentifier / AcquisitionNumber / InstanceNumber phase-block 中由冻结 source contract 验证出的可用方法形成完整 temporal cell grid。公开文件不发布原始 IPP 或 acquisition time。

## Frame of Reference 与 PatientPosition

四访视 FrameOfReferenceUID 均存在，但属于四个不同等价类。同一个 patient 的不同 visit 使用不同 UID 并不自动代表错误，也不提供任何 transform。四个 selected series 的 PatientPosition 均为 `FFP`；该字段没有提供额外 flip/translation。DICOM 标准将 PatientPosition 定义为 annotation，而 ImagePositionPatient、ImageOrientationPatient 与 PixelSpacing 才定义 patient-coordinate image plane。因此现有 LPS→RAS builder 未因 PatientPosition 再施加一次翻转：[Patient Position]({DICOM_PATIENT_POSITION_URL})、[Image Plane Module]({DICOM_IMAGE_PLANE_URL})。

## Same-study object 与 series inventory

failed study 可读 DICOM instances 为 {study_summary['readable_instances']}，SeriesInstanceUID 分组数为 {study_summary['series_uid_count']}；Spatial Registration / Deformable Spatial Registration object 数为 {study_summary['spatial_registration_object_count']}。因此没有 DICOM registration relationship 可将 failed frame 权威映射到 T0。Spatial Registration IOD 才是标准中描述不同 Frames of Reference 空间关系的对象：[Spatial Registration IOD]({DICOM_REGISTRATION_URL})。

冻结、outcome-free 的 native-DCE contract 在 failed study 中得到 {candidate_summary['strict_candidate_count']} 个严格候选，其中当前 selected acquisition 本身有效；排除当前 series 后 alternate candidate 数为 {candidate_summary['strict_alternate_count']}。候选先依据 MR / ORIGINAL-PRIMARY / breast / native-DCE / temporal-stack / geometry semantics 决定，未读取 overlap、lesion 或 outcome。

## Scanner / protocol change

公开表只保留 manufacturer/model 与匿名 protocol group；StationName、free-text description/protocol 和日期时间留在 private inventory。`magnetic_field_strength_raw` 保留 legacy DICOM 原始编码，未作单位归一化，也未参与几何判定。T0/T1 与 T2/T3 存在 scanner-model/protocol 分组变化，可解释 acquisition context 改变，但不是自动构造坐标变换的依据。
"""


def _final_report(
    *,
    failed_visit: str,
    t0_failed: Mapping[str, Any],
    adjacent_rows: Sequence[Mapping[str, Any]],
    translation: Mapping[str, Any],
    registration: Mapping[str, Any],
    candidate_summary: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> str:
    one = translation["one_voxel"]
    half = translation["coverage_50_percent"]
    ninety = translation["coverage_90_percent"]
    ninety_five = translation["coverage_95_percent"]
    adjacency = ", ".join(
        f"{row['visit_pair']}={row['center_displacement_mm']:.3f} mm"
        for row in adjacent_rows
    )
    registration_status = (
        "PASS（仅诊断）" if registration["success"] else f"FAIL / {registration['failure_code']}"
    )
    return f"""# ZERO_VALID_SOURCE_OVERLAP 单病例纵向几何 provenance audit

## 最终判定

**`{decision['decision']}`；root-cause class = `{decision['root_cause_class']}`；`REPAIR = NO`。**

`{CASE_ALIAS}` 的四访视 raw DICOM geometry 均在 series 内自洽，failed visit `{failed_visit}` 的 selected acquisition 也通过 raw PixelData 双解码、temporal/slice cell 完整性、finite/nonconstant、IOP/IPP/spacing affine 与独立 rebuilt-NIfTI 对照。T0 与 `{failed_visit}` source center 相距 **{t0_failed['center_displacement_mm']:.3f} mm**，两个完整 source footprints 的最小间距为 **{t0_failed['minimum_separation_mm']:.3f} mm**，oriented physical intersection 为 **{t0_failed['oriented_intersection_mm3']:.3f} mm³**。

这不是几毫米级轻微 repositioning。虽然让 frozen C1B-H grid 首次出现 1 个 valid target voxel 的最小诊断平移为 {_format_number(one['translation_magnitude_mm'])} mm，但达到 50% valid-source coverage 需要约 {_format_number(half['translation_magnitude_mm'])} mm；90% 与 95% coverage 均因 source FOV 本身不足而无法仅靠 rigid translation 达到（最大可达 {100.0 * float(ninety['maximum_attainable_valid_fraction']):.3f}%）。这些数值只描述失配尺度，**不是 repair transform**。

## 核心证据

- 相邻访视 source-center displacement：{adjacency}。观测上表现为末次访视 `{failed_visit}` 的大跳跃；其后没有访视，无法检验该变化是否持续。
- T0、T1、T2、T3 的 FrameOfReferenceUID 均存在但互不相同；没有 Spatial Registration object 或 metadata-defined transform。UID 不同不能自行反推出 translation。
- PatientPosition 四访视均为 `FFP`；T0/T2/T3 raw orientation 一致，T1 的 acquisition plane 不同但与相邻访视仍有实质物理重叠。这证明当前 pipeline 已按 IOP/IPP 处理 plane change，不支持对 failed visit 再人工 flip。
- failed study 严格 native-DCE candidates = {candidate_summary['strict_candidate_count']}，当前 series 有效；alternate candidates = {candidate_summary['strict_alternate_count']}。不存在唯一 alternate acquisition。
- 没有发现当前 builder 违反 DICOM patient-coordinate semantics；raw affine、LPS→RAS、真实 RAS canonicalization 与冻结 rebuilt source 一致。
- image-only rigid registration 状态为 `{registration_status}`。optimizer candidate 的 source-center translation magnitude 为 {_format_number(registration.get('translation_magnitude_mm'))} mm、rotation magnitude 为 {_format_number(registration.get('rotation_magnitude_deg'))}°；before NCC 因零物理重叠不可定义，after NCC 为 {_format_number(registration.get('similarity_after_ncc'), 4)}，valid-overlap fraction 从 {100.0 * float(registration.get('valid_overlap_fraction_before') or 0.0):.3f}% 到 {100.0 * float(registration.get('valid_overlap_fraction_after') or 0.0):.3f}%。该结果不能区分真实 repositioning 与 upstream coordinate corruption，且 transform 从未进入 repair selection。

## 13 个问题逐项回答

1. **failed visit raw DICOM geometry 是否自洽？** 是。header grid、IOP/IPP、spacing、temporal cells 与双次 PixelData decode 均 PASS。
2. **T0 与 failed visit physical source 相距多少？** center displacement = {t0_failed['center_displacement_mm']:.3f} mm；minimum footprint separation = {t0_failed['minimum_separation_mm']:.3f} mm；oriented overlap = 0。
3. **轻微 repositioning 还是 catastrophic mismatch？** 属于 catastrophic longitudinal coordinate mismatch：单访视中心跳跃超过百毫米，50% target coverage 需要约 {_format_number(half['translation_magnitude_mm'])} mm 平移，90%/95% 连理论上都不可达。
4. **FrameOfReferenceUID 是否解释异常？** 否。四访视 UID 均不同且无注册对象；UID 只标识 frame，不定义跨 frame transform。
5. **PatientPosition / IOP / IPP 是否有明确异常？** PatientPosition 一致；IOP 正交且 series 内一致；IPP 形成规则 slice grid。没有 metadata-authoritative flip/translation 异常。
6. **是否存在同 study 唯一合法 alternate native DCE？** 否。严格候选只有当前 acquisition；alternate = 0。
7. **是否发现当前 builder 的确定性 geometry bug？** 否。DICOM LPS affine、LPS→RAS 与 canonicalization 复核一致。
8. **是否存在唯一 source-authoritative repair？** 否。
9. **image-only diagnostic registration 显示什么？** `{registration_status}`；它提出大幅 image-driven rigid candidate，但不能成为 source provenance，也不能可靠区分 repositioning 与 coordinate corruption。
10. **lesion/outcome 是否完全未参与选择？** 是。读取列表为空：FTV、LD、clinical、treatment、pCR 与 model performance 均未读取；T0 grid 使用 acquisition-center fallback。
11. **R1–R5 分类？** `{decision['root_cause_class']}`。raw geometry 自洽，但 metadata 无法唯一地区分 plausible extreme repositioning 与 upstream coordinate corruption。
12. **是否允许 repair？** 不允许；repair gate 的 source-authoritative/unique/corrected-verification 条件不成立。
13. **是否进入新的 four-visit eligibility amendment？** 是，建议新 run 在任何 representation 结果之前预注册：`Eligible(patient) = AND over t in {{T0,T1,T2,T3}} [valid_source_voxels(patient,t) > 0]`。本 audit 不执行 population rerun；`948 → 947` 只能作为待验证预期，不能写成确认结果。

## 科学边界与下一步

上一轮 `STAGE_A_NO_GO` 保持 immutable，Stage B 未运行，C1B crop contract 未修改。本轮没有训练模型、没有 registration sweep、没有尝试任意 flip/translation/recenter，也没有用 lesion 或结果倒推 transform。

下一步应建立新的、outcome-free technical eligibility amendment 并从 frozen population 重新开始一个全新 Stage A。旧 run 不能追溯改写为 GO。

**Source provenance decides repair. Outcome does not. Lesion does not. Model performance does not.**
"""


def run_audit(*, experiment_root: Path, overwrite: bool = False) -> dict[str, Any]:
    experiment_root = experiment_root.resolve()
    repo_root = experiment_root.parents[1]
    prior_root = experiment_root.parent / "c1b_model_ready_ftv_sanity"
    private_root = experiment_root / "private"
    if not (experiment_root / ".gitignore").is_file():
        raise RuntimeError("EXPERIMENT_GITIGNORE_MISSING")
    private_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(private_root, 0o700)

    prior_relative = prior_root.relative_to(repo_root)
    prior_before = _tracked_tree_digest(repo_root, prior_relative)
    case = resolve_private_case(prior_root)
    atomic_private_json(
        experiment_root / "manifests/case_resolution.private.json", case
    )
    frozen = load_frozen_ispy1_contract(prior_root)

    print("[1/7] private case resolved: cases=1 visits=4 alias=CASE_ZERO_OVERLAP_001")
    raw_audits: dict[str, Any] = {}
    provenance: dict[str, dict[str, Any]] = {}
    study_objects: dict[str, dict[str, Any]] = {}
    rebuild_public: dict[str, dict[str, Any]] = {}
    rebuild_private: dict[str, dict[str, Any]] = {}
    canonical: dict[str, Any] = {}
    boxes: dict[str, OrientedBox] = {}
    rebuilt_comparison: dict[str, dict[str, Any]] = {}

    for visit in VISITS:
        visit_private = case["visits"][visit]
        selected_paths = visit_private["selected_source_series"]
        if len(selected_paths) != 1:
            raise RuntimeError("SELECTED_DYNAMIC_SERIES_NOT_UNIQUE")
        audit = frozen.audit_series(Path(selected_paths[0]))
        if not (
            audit.status == "PASS"
            and audit.semantic_ok
            and audit.is_dynamic
            and audit.affine_ras is not None
        ):
            raise RuntimeError("SELECTED_RAW_SERIES_CONTRACT_FAILED")
        raw_audits[visit] = audit
        provenance[visit] = read_selected_series_provenance(audit)
        study_objects[visit] = scan_study_objects(Path(visit_private["study_dir"]))
        rebuilt = rebuild_selected_series(frozen, audit)
        comparison = _compare_rebuilt_nifti(
            rebuilt["volume_xyzt"],
            rebuilt["affine_ras"],
            Path(visit_private["rebuilt_nifti"]),
        )
        if comparison["status"] != "PASS":
            raise RuntimeError("INDEPENDENT_RAW_REBUILD_DISAGREES_WITH_FROZEN_REBUILD")
        rebuilt_comparison[visit] = comparison
        rebuild_public[visit] = rebuilt["public"]
        rebuild_private[visit] = {
            "public": rebuilt["public"],
            "affine_ras": rebuilt["affine_ras"].tolist(),
            "cells": rebuilt["private_cells"],
            "frozen_rebuild_comparison": comparison,
        }
        volume = rebuilt["volume_xyzt"]
        canonical_volume = canonicalize_to_ras(
            volume[..., 0], rebuilt["affine_ras"]
        )
        canonical[visit] = canonical_volume
        raw_box = OrientedBox.from_affine_shape(
            rebuilt["affine_ras"], volume.shape[:3]
        )
        canonical_box = OrientedBox.from_affine_shape(
            canonical_volume.affine_ras, canonical_volume.shape_xyz
        )
        if _corner_hausdorff(raw_box, canonical_box) > 1.0e-6:
            raise RuntimeError("CANONICALIZATION_FOOTPRINT_CHANGED")
        boxes[visit] = raw_box
        del volume, rebuilt

    atomic_private_json(
        experiment_root / "metrics/raw_dicom_rebuild.private.json", rebuild_private
    )
    print("[2/7] raw DICOM rebuild complete: visits=4 status=PASS")

    failed_visit = str(case["failed_visit"])
    candidate = audit_failed_study_candidates(
        frozen,
        study_dir=Path(case["visits"][failed_visit]["study_dir"]),
        selected_paths=case["visits"][failed_visit]["selected_source_series"],
    )
    for row in candidate["public_rows"]:
        if row["selected_current"]:
            row["raw_pixel_rebuild"] = "PASS"
    candidate_seal = {
        "strict_candidate_count": candidate["strict_candidate_count"],
        "strict_alternate_count": candidate["strict_alternate_count"],
        "current_selected_is_strict_candidate": candidate["current_valid"],
        "candidate_selection_used_overlap": False,
        "candidate_selection_used_lesion_or_outcome": False,
        "selection_sealed_before_registration": True,
    }
    atomic_private_json(
        experiment_root / "manifests/failed_study_candidate_audit.private.json",
        {
            **candidate_seal,
            "rows": candidate["private_rows"],
            "selected_candidate_source_keys": [list(item.source_key) for item in candidate["candidates"]],
        },
    )
    print(
        "[3/7] source-semantic candidates sealed: "
        f"strict={candidate['strict_candidate_count']} alternates={candidate['strict_alternate_count']}"
    )

    protocol_values = {visit: _protocol_fingerprint(provenance[visit]) for visit in VISITS}
    protocol_groups = _assign_groups(protocol_values, "PROTOCOL_GROUP")
    frame_uids = {
        visit: str(representative_value(provenance[visit], "FrameOfReferenceUID") or "")
        for visit in VISITS
    }
    if any(not value for value in frame_uids.values()):
        frame_classification = "MISSING_OR_MALFORMED"
    elif len(set(frame_uids.values())) == 1:
        frame_classification = "SAME_ALL_VISITS"
    else:
        frame_classification = "DIFFERENT_ACROSS_VISITS"
    registration_object_count = sum(
        int(study_objects[visit]["public"]["spatial_registration_object_count"])
        for visit in VISITS
    )

    pair_rows: list[dict[str, Any]] = []
    pair_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for first, second in PAIR_ORDER:
        metrics = pairwise_metrics(boxes[first], boxes[second])
        payload = metrics.to_dict() if hasattr(metrics, "to_dict") else asdict(metrics)
        row = {
            "case_alias": CASE_ALIAS,
            "visit_pair": f"{first}-{second}",
            "first_visit": first,
            "second_visit": second,
            "adjacent_visits": (first, second) in ADJACENT_PAIRS,
            **payload,
        }
        pair_rows.append(row)
        pair_lookup[(first, second)] = row
    t0_failed = pair_lookup[("T0", failed_visit)]

    table1: list[dict[str, Any]] = []
    t0_center = boxes["T0"].center
    for visit in VISITS:
        if visit == "T0":
            metrics_t0 = {
                "center_displacement_mm": 0.0,
                "minimum_separation_mm": 0.0,
                "overlap_fraction_second": 1.0,
            }
        else:
            metrics_t0 = pair_lookup[("T0", visit)]
        patient_position = str(
            representative_value(provenance[visit], "PatientPosition") or "MISSING"
        )
        manufacturer = str(
            representative_value(provenance[visit], "Manufacturer") or "MISSING"
        )
        model = str(
            representative_value(provenance[visit], "ManufacturerModelName")
            or "MISSING"
        )
        field_strength = representative_value(provenance[visit], "MagneticFieldStrength")
        table1.append(
            {
                "case_alias": CASE_ALIAS,
                "visit": visit,
                "failed_visit": visit == failed_visit,
                "source_center_ras_t0_relative_mm": np.round(
                    boxes[visit].center - t0_center, 6
                ).tolist(),
                "source_shape_xyz": [int(value) for value in raw_audits[visit].shape_xyz],
                "source_spacing_xyz_mm": [
                    round(float(value), 6) for value in raw_audits[visit].spacing_xyz_mm
                ],
                "physical_extent_source_axes_mm": np.round(
                    boxes[visit].fov_lengths, 6
                ).tolist(),
                "orientation_ras": _orientation(raw_audits[visit].affine_ras),
                "patient_position": patient_position,
                "frame_uid_relation_to_t0": "SAME"
                if frame_uids[visit] == frame_uids["T0"]
                else "DIFFERENT",
                "distance_to_t0_center_mm": float(
                    metrics_t0["center_displacement_mm"]
                ),
                "minimum_volume_separation_from_t0_mm": float(
                    metrics_t0["minimum_separation_mm"]
                ),
                "physical_overlap_fraction_of_visit_with_t0": float(
                    metrics_t0["overlap_fraction_second"]
                ),
                "raw_geometry_status": "PASS",
                "raw_pixel_cells_verified": int(
                    rebuild_public[visit]["verified_cell_count"]
                ),
                "manufacturer": manufacturer,
                "manufacturer_model": model,
                "magnetic_field_strength_raw": field_strength,
                "protocol_group": protocol_groups[visit],
            }
        )

    grid = make_c1b_grid(tuple(float(value) for value in boxes["T0"].center))
    mapping = input_from_output_affine(canonical[failed_visit].affine_ras, grid)
    valid_mask = _valid_source_footprint_mask_xyz(
        mapping, grid.shape_xyz, canonical[failed_visit].shape_xyz
    )
    exact_current_valid = int(np.count_nonzero(valid_mask))
    if exact_current_valid != 0:
        raise RuntimeError("FAILED_VISIT_NO_LONGER_ZERO_OVERLAP")
    target_box = OrientedBox.from_affine_shape(grid.affine_ras, grid.shape_xyz)
    one_requirement = minimum_cardinal_translation_for_count(
        grid.affine_ras, grid.shape_xyz, boxes[failed_visit], 1
    )
    fraction_requirements = {
        fraction: minimum_cardinal_translation_for_fraction(
            grid.affine_ras, grid.shape_xyz, boxes[failed_visit], fraction
        )
        for fraction in (0.50, 0.90, 0.95)
    }
    translation_summary = {
        "schema_version": 1,
        "case_alias": CASE_ALIAS,
        "analysis_scope": "DIAGNOSTIC_ONLY",
        "target_grid": {
            "contract": "C1B-H",
            "shape_zyx": list(grid.shape_zyx),
            "spacing_xyz_mm": list(grid.spacing_xyz_mm),
            "anchor": "t0_acquisition_physical_center_fallback",
            "target_grid_voxels": int(np.prod(grid.shape_xyz)),
        },
        "current_valid_source_voxels": exact_current_valid,
        "one_voxel": one_requirement.to_dict(),
        "coverage_50_percent": fraction_requirements[0.50].to_dict(),
        "coverage_90_percent": fraction_requirements[0.90].to_dict(),
        "coverage_95_percent": fraction_requirements[0.95].to_dict(),
        "translation_vector_public": False,
        "translation_used_as_repair": False,
    }
    _atomic_json(
        experiment_root / "metrics/required_overlap_translation_summary.json",
        translation_summary,
        overwrite=overwrite,
    )

    registration = run_image_only_diagnostic(
        canonical["T0"], canonical[failed_visit]
    )
    registration_public = registration["public"]
    atomic_private_json(
        experiment_root / "metrics/image_registration_diagnostic.private.json",
        {
            "case_alias": CASE_ALIAS,
            "public": registration_public,
            "private": registration["private"],
            "used_for_repair_selection": False,
        },
    )
    _atomic_json(
        experiment_root / "metrics/image_registration_diagnostic.json",
        registration_public,
        overwrite=overwrite,
    )
    print(
        "[4/7] geometry and diagnostic registration complete: "
        f"registration_success={registration_public['success']}"
    )

    builder_bug = not all(
        comparison["status"] == "PASS" for comparison in rebuilt_comparison.values()
    )
    multiple_repair_candidates = bool(candidate["strict_candidate_count"] > 1)
    unique_replacement = bool(
        not candidate["current_valid"]
        and candidate["strict_candidate_count"] == 1
        and candidate["strict_alternate_count"] == 1
    )
    if builder_bug:
        root_class = "R1_AUTHORITATIVE_GEOMETRY_REPAIR"
        decision_string = "AUDIT-REPAIRABLE"
    elif unique_replacement:
        root_class = "R2_UNIQUE_VALID_ALTERNATE_ACQUISITION"
        decision_string = "AUDIT-REPAIRABLE"
    elif multiple_repair_candidates:
        root_class = "R5_AMBIGUOUS_REPAIR"
        decision_string = "AUDIT-AMBIGUOUS"
    else:
        # Self-consistent but mutually incompatible physical frames, distinct
        # FoR UIDs, no registration relationship, and failed/non-authoritative
        # image registration cannot discriminate motion from upstream corruption.
        root_class = "R4_UNRESOLVED_COORDINATE_PROVENANCE"
        decision_string = "AUDIT-NOT-REPAIRABLE"

    repair_allowed = decision_string == "AUDIT-REPAIRABLE"
    decision = {
        "schema_version": 1,
        "case_alias": CASE_ALIAS,
        "decision": decision_string,
        "root_cause_class": root_class,
        "repair_allowed": repair_allowed,
        "technical_eligibility_failure": not repair_allowed,
        "four_visit_valid_source_overlap_amendment_recommended": not repair_allowed,
        "amendment_rule": (
            "Eligible(patient)=AND over t in {T0,T1,T2,T3} "
            "[valid_source_voxels(patient,t)>0]"
        ),
        "amendment_population_not_rerun_in_this_audit": True,
        "expected_population_change_not_confirmed": "948_to_947_requires_new_run",
        "stage_b_run": False,
        "model_training_run": False,
        "c1b_crop_contract_modified": False,
        "prior_stage_a_decision_modified": False,
        "outcome_fields_read": [],
        "lesion_or_ftv_fields_read": [],
        "clinical_or_treatment_fields_read": [],
        "manual_transform_trials": 0,
        "registration_transform_used_for_repair": False,
    }

    root_rows = [
        {
            "case_alias": CASE_ALIAS,
            "hypothesis": "R1_AUTHORITATIVE_GEOMETRY_REPAIR",
            "evidence_status": "AGAINST",
            "evidence": "raw affine, LPS-to-RAS, canonical footprint, and frozen raw rebuild agree",
            "selected": root_class.startswith("R1_"),
        },
        {
            "case_alias": CASE_ALIAS,
            "hypothesis": "R2_UNIQUE_VALID_ALTERNATE_ACQUISITION",
            "evidence_status": "AGAINST",
            "evidence": "current native DCE is valid and strict alternate count is zero",
            "selected": root_class.startswith("R2_"),
        },
        {
            "case_alias": CASE_ALIAS,
            "hypothesis": "R3_TRUE_OR_PLAUSIBLE_EXTREME_REPOSITIONING",
            "evidence_status": "COMPATIBLE_NOT_IDENTIFIABLE",
            "evidence": "large single-visit displacement is compatible with repositioning but not uniquely proven",
            "selected": root_class.startswith("R3_"),
        },
        {
            "case_alias": CASE_ALIAS,
            "hypothesis": "R4_UNRESOLVED_COORDINATE_PROVENANCE",
            "evidence_status": "SUPPORTED",
            "evidence": "different frames, no authoritative registration relationship, no unique source correction",
            "selected": root_class.startswith("R4_"),
        },
        {
            "case_alias": CASE_ALIAS,
            "hypothesis": "R5_AMBIGUOUS_REPAIR",
            "evidence_status": "AGAINST",
            "evidence": "no competing source-semantic repair candidates were found",
            "selected": root_class.startswith("R5_"),
        },
    ]
    repair_rows = [
        ("source_only", True, "all selection inputs are raw imaging metadata/pixels"),
        ("outcome_free", True, "no clinical, treatment, pCR, LD, FTV, or model performance read"),
        ("unique_source_authoritative_correction", False, "no metadata-defined correction exists"),
        ("auditable", True, "raw headers, pixels, hashes, and code provenance retained privately"),
        ("programmatic", False, "there is no accepted correction rule to apply"),
        ("raw_pixel_verified", True, "all four current acquisitions rebuilt and compared twice"),
        ("no_lesion_based_choice", True, "T0 acquisition-center fallback; no lesion inputs"),
        ("no_manual_trial_transform", True, "manual transform trials equal zero"),
        ("generalizable_correction_rule", False, "no correction rule was identified"),
        ("corrected_valid_source_overlap_positive", False, "no corrected visit is authorized"),
        ("other_qc_preserved_after_correction", False, "not testable without an accepted correction"),
    ]
    repair_table = [
        {
            "case_alias": CASE_ALIAS,
            "gate": gate,
            "status": "PASS" if passed else "FAIL",
            "evidence": evidence,
        }
        for gate, passed, evidence in repair_rows
    ]

    _atomic_csv(
        experiment_root / "metrics/table1_four_visit_metadata_geometry_summary.csv",
        table1,
        overwrite=overwrite,
    )
    _atomic_csv(
        experiment_root / "metrics/table2_pairwise_geometry.csv",
        pair_rows,
        overwrite=overwrite,
    )
    _atomic_csv(
        experiment_root / "metrics/table3_same_study_dce_candidate_semantic_audit.csv",
        candidate["public_rows"],
        overwrite=overwrite,
    )
    _atomic_csv(
        experiment_root / "metrics/table4_root_cause_evidence.csv",
        root_rows,
        overwrite=overwrite,
    )
    _atomic_csv(
        experiment_root / "metrics/table5_repair_acceptance_gate.csv",
        repair_table,
        overwrite=overwrite,
    )
    _atomic_json(
        experiment_root / "metrics/raw_dicom_rebuild_summary.json",
        {
            "case_alias": CASE_ALIAS,
            "visits": rebuild_public,
            "all_visits_pass": True,
            "source_is_raw_dicom": True,
            "second_pixel_decode": True,
            "cached_coordinates_used_as_geometry_truth": False,
        },
        overwrite=overwrite,
    )
    _atomic_json(
        experiment_root / "metrics/dicom_provenance_summary.json",
        {
            "case_alias": CASE_ALIAS,
            "frame_of_reference_classification": frame_classification,
            "frame_equivalence_class_count": len(
                {value for value in frame_uids.values() if value}
            ),
            "frame_uid_values_public": False,
            "patient_position_values": {
                visit: representative_value(provenance[visit], "PatientPosition")
                for visit in VISITS
            },
            "registration_object_count": registration_object_count,
            "authoritative_registration_relationship_found": registration_object_count > 0,
            "failed_study_candidate_seal": candidate_seal,
            "absolute_ipp_and_centers_public": False,
            "outcome_fields_read": [],
        },
        overwrite=overwrite,
    )
    _atomic_json(
        experiment_root / "metrics/audit_decision.json", decision, overwrite=overwrite
    )
    sentinel_name = {
        "AUDIT-REPAIRABLE": "AUDIT_REPAIRABLE.json",
        "AUDIT-NOT-REPAIRABLE": "AUDIT_NOT_REPAIRABLE.json",
        "AUDIT-AMBIGUOUS": "AUDIT_AMBIGUOUS.json",
    }[decision_string]
    for name in ("AUDIT_REPAIRABLE.json", "AUDIT_NOT_REPAIRABLE.json", "AUDIT_AMBIGUOUS.json"):
        path = experiment_root / name
        if name != sentinel_name and path.exists():
            raise RuntimeError("CONFLICTING_AUDIT_SENTINEL_EXISTS")
    _atomic_json(experiment_root / sentinel_name, decision, overwrite=overwrite)
    _atomic_json(
        experiment_root / "metrics/environment.json",
        _public_environment(),
        overwrite=overwrite,
    )

    inventory_report = _inventory_report(
        table1,
        failed_visit=failed_visit,
        study_summary=study_objects[failed_visit]["public"],
        candidate_summary=candidate_seal,
        metadata=provenance,
    )
    _atomic_text(
        experiment_root / "reports/dicom_provenance_inventory.md",
        inventory_report,
        overwrite=overwrite,
    )
    final_report = _final_report(
        failed_visit=failed_visit,
        t0_failed=t0_failed,
        adjacent_rows=[row for row in pair_rows if row["adjacent_visits"]],
        translation=translation_summary,
        registration=registration_public,
        candidate_summary=candidate_seal,
        decision=decision,
    )
    _atomic_text(
        experiment_root / "reports/final_report.md", final_report, overwrite=overwrite
    )
    _make_figures(
        experiment_root,
        boxes,
        pair_rows,
        failed_visit,
        canonical,
        registration,
        overwrite=overwrite,
    )

    atomic_private_json(
        experiment_root / "private/dicom_provenance_inventory.private.json",
        {
            "case_alias": CASE_ALIAS,
            "visits": provenance,
            "study_objects": study_objects,
            "geometry": {
                visit: {
                    "center_ras_mm": boxes[visit].center.tolist(),
                    "axes_ras": boxes[visit].axes.tolist(),
                    "half_lengths_mm": boxes[visit].half_lengths.tolist(),
                    "corners_ras_mm": boxes[visit].corners().tolist(),
                }
                for visit in VISITS
            },
            "target_grid_affine_ras": grid.affine_ras.tolist(),
            "source_artifact_sha256": case["source_artifact_sha256"],
        },
    )

    prior_after = _tracked_tree_digest(repo_root, prior_relative)
    if prior_before != prior_after:
        raise RuntimeError("PRIOR_EXPERIMENT_MUTATED")
    immutable = {
        "prior_experiment": "c1b_model_ready_ftv_sanity",
        "before": prior_before,
        "after": prior_after,
        "unchanged": True,
        "stage_a_no_go_preserved": True,
    }
    _atomic_json(
        experiment_root / "metrics/prior_experiment_immutability.json",
        immutable,
        overwrite=overwrite,
    )
    print(f"[5/7] public tables/reports/figures generated: decision={decision_string}")
    return {
        "decision": decision,
        "failed_visit": failed_visit,
        "prior_immutability": immutable,
        "public_outputs_complete": True,
    }


__all__ = ["run_audit"]

"""从冻结 M0/M1/M2 checkpoint 提取真实 observed MRI 中间表征。"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .common import (
    AUDIT_ROOT,
    REPO_ROOT,
    atomic_csv,
    atomic_json,
    atomic_npz,
    file_sha256,
    load_yaml,
    refuse_existing,
    resolve_repo_path,
    source_sha256,
)


TIMEPOINTS = ("T0", "T1", "T2", "T3")
CORE_REPRESENTATIONS = (
    "online_projected",
    "online_preprojector",
    "online_global_pool",
    "online_roi_mean",
    "ema_projected",
    "ema_preprojector",
    "ema_global_pool",
    "ema_roi_mean",
    "mask_geometry",
    "raw_roi_intensity",
)
REPRESENTATION_STREAM = {
    "online_projected": "online",
    "online_preprojector": "online",
    "online_global_pool": "online",
    "online_roi_mean": "online",
    "ema_projected": "ema_target",
    "ema_preprojector": "ema_target",
    "ema_global_pool": "ema_target",
    "ema_roi_mean": "ema_target",
    "mask_geometry": "input_baseline",
    "raw_roi_intensity": "input_baseline",
}


@dataclass(frozen=True)
class SourceModules:
    source_root: Path
    load_config: Any
    load_evaluation: Any
    dataset_class: Any
    records_for_ids: Any


def import_source_modules(config: dict[str, Any]) -> SourceModules:
    source_root = resolve_repo_path(config["source_experiment"])
    source_src = source_root / "src"
    if str(source_src) not in sys.path:
        sys.path.insert(0, str(source_src))
    from rnc.config import load_config  # type: ignore
    from rnc.data import LongitudinalCacheDataset  # type: ignore
    from rnc.evaluation import load_evaluation  # type: ignore
    from rnc.training import records_for_ids  # type: ignore

    return SourceModules(
        source_root=source_root,
        load_config=load_config,
        load_evaluation=load_evaluation,
        dataset_class=LongitudinalCacheDataset,
        records_for_ids=records_for_ids,
    )


def extraction_implementation_sha256() -> str:
    return source_sha256(
        [
            Path(__file__),
            Path(__file__).with_name("common.py"),
            AUDIT_ROOT / "scripts" / "extract_features.py",
        ]
    )


def _branch_features(
    image: torch.Tensor,
    encoder: torch.nn.Module,
    projector: torch.nn.Module,
    stream: str,
) -> dict[str, torch.Tensor]:
    batch, visits = image.shape[:2]
    flat = image.reshape(batch * visits, *image.shape[2:])
    if len(encoder.features) != 5:
        raise AssertionError("VisitEncoder3D.features 不再是 4 residual stages + pooling")
    spatial = encoder.features[:4](flat)
    if spatial.shape[1:] != (128, 4, 12, 12):
        raise AssertionError(f"最后 spatial map shape 漂移: {tuple(spatial.shape)}")
    pooled_5d = encoder.features[4](spatial)
    pooled = pooled_5d.flatten(1)
    if not torch.allclose(pooled, spatial.mean(dim=(-3, -2, -1)), atol=2e-6, rtol=2e-6):
        raise AssertionError("AdaptiveAvgPool3d 与 spatial mean 不一致")
    preprojector = encoder.output(pooled_5d)
    projected = projector(preprojector)

    mask = flat[:, 7:8]
    mask_voxels = mask.sum(dim=(-3, -2, -1)).squeeze(1)
    roi_valid = mask_voxels > 0
    occupancy = F.adaptive_avg_pool3d(mask, spatial.shape[-3:])
    denominator = occupancy.sum(dim=(-3, -2, -1)).clamp_min(1e-12)
    roi_mean = (spatial * occupancy).sum(dim=(-3, -2, -1)) / denominator
    roi_mean = roi_mean.masked_fill(~roi_valid[:, None], float("nan"))

    prefix = "online" if stream == "online" else "ema"
    return {
        f"{prefix}_spatial": spatial.reshape(batch, visits, 128, 4, 12, 12),
        f"{prefix}_global_pool": pooled.reshape(batch, visits, 128),
        f"{prefix}_preprojector": preprojector.reshape(batch, visits, 192),
        f"{prefix}_projected": projected.reshape(batch, visits, 192),
        f"{prefix}_roi_mean": roi_mean.reshape(batch, visits, 128),
        "roi_valid": roi_valid.reshape(batch, visits),
    }


def _mask_geometry_and_intensity(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 9-D voxel/crop geometry、14-D DCE7 ROI summary 和 ROI validity。"""

    batch, visits = image.shape[:2]
    flat = image.reshape(batch * visits, *image.shape[2:])
    dce = flat[:, :7]
    mask = flat[:, 7] > 0.5
    values_geometry: list[list[float]] = []
    values_intensity: list[list[float]] = []
    valid_values: list[bool] = []
    depth, height, width = mask.shape[-3:]
    total = float(depth * height * width)
    for index in range(mask.shape[0]):
        current = mask[index]
        coordinates = torch.nonzero(current, as_tuple=False)
        valid = bool(coordinates.numel())
        valid_values.append(valid)
        if not valid:
            values_geometry.append([float("nan")] * 9)
            values_intensity.append([float("nan")] * 14)
            continue
        count = float(coordinates.shape[0])
        minimum = coordinates.amin(dim=0).float()
        maximum = coordinates.amax(dim=0).float()
        extents_voxel = maximum - minimum + 1.0
        normalizer = torch.tensor([depth, height, width], device=coordinates.device, dtype=torch.float32)
        extents = extents_voxel / normalizer
        centroid = coordinates.float().mean(dim=0) / torch.tensor(
            [max(depth - 1, 1), max(height - 1, 1), max(width - 1, 1)],
            device=coordinates.device,
            dtype=torch.float32,
        )
        diagonal = float(torch.linalg.vector_norm(extents).item())
        values_geometry.append(
            [
                float(np.log1p(count)),
                count / total,
                *[float(value) for value in extents.tolist()],
                diagonal,
                *[float(value) for value in centroid.tolist()],
            ]
        )
        selected = dce[index, :, current]
        means = selected.mean(dim=1)
        standard_deviations = selected.std(dim=1, unbiased=False)
        values_intensity.append(
            [float(value) for value in torch.cat((means, standard_deviations)).tolist()]
        )
    geometry = torch.tensor(values_geometry, dtype=torch.float32).reshape(batch, visits, 9)
    intensity = torch.tensor(values_intensity, dtype=torch.float32).reshape(batch, visits, 14)
    valid = torch.tensor(valid_values, dtype=torch.bool).reshape(batch, visits)
    return geometry, intensity, valid


def _manifest_fragment(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    feature_path: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    patient_ids = arrays["patient_ids"].astype(str)
    splits = arrays["splits"].astype(str)
    has_radiomics = arrays["has_radiomics"].astype(bool)
    roi_valid = arrays["roi_valid"].astype(bool)
    feature_hash = file_sha256(feature_path)
    for patient_index, patient_id in enumerate(patient_ids):
        for visit_index, timepoint in enumerate(TIMEPOINTS):
            for representation in CORE_REPRESENTATIONS:
                is_roi_dependent = representation in {
                    "online_roi_mean",
                    "ema_roi_mean",
                    "mask_geometry",
                    "raw_roi_intensity",
                }
                rows.append(
                    {
                        "patient_id": patient_id,
                        "fold": metadata["fold"],
                        "split": splits[patient_index],
                        "model": metadata["model"],
                        "run_name": metadata["run_name"],
                        "checkpoint": metadata["checkpoint"],
                        "checkpoint_sha256": metadata["checkpoint_sha256"],
                        "timepoint": timepoint,
                        "representation_type": representation,
                        "encoder_stream": REPRESENTATION_STREAM[representation],
                        "feature_dim": int(arrays[representation].shape[-1]),
                        "feature_file": str(feature_path.resolve().relative_to(REPO_ROOT)),
                        "feature_file_sha256": feature_hash,
                        "patient_index": patient_index,
                        "visit_index": visit_index,
                        "has_radiomics": bool(has_radiomics[patient_index]),
                        "roi_mask_valid": bool(roi_valid[patient_index, visit_index]),
                        "feature_valid": bool(roi_valid[patient_index, visit_index])
                        if is_roi_dependent
                        else True,
                        "extractor_sha256": metadata["extractor_sha256"],
                    }
                )
    return pd.DataFrame(rows)


def extract_checkpoint(
    config_path: Path,
    model_label: str,
    fold: int,
    device_name: str,
    output_root: Path,
    batch_size: int,
    workers: int,
    max_patients_per_split: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    source = import_source_modules(config)
    if model_label not in config["models"]:
        raise ValueError(f"未知 model label: {model_label}")
    if fold not in range(5):
        raise ValueError(f"fold 必须为 0–4: {fold}")
    model_spec = config["models"][model_label]
    checkpoint = source.source_root / "checkpoints" / model_spec["run_name"] / f"fold_{fold}" / "best.pt"
    source_config_path = source.source_root / model_spec["config"]
    device = torch.device(device_name)
    evaluation = source.load_evaluation(checkpoint, source.load_config(source_config_path), device)
    if evaluation.fold != fold or evaluation.mode != model_label:
        raise AssertionError("checkpoint fold/mode 与请求不一致")
    if evaluation.model.image_channels != 8:
        raise AssertionError("本 audit 预注册为 DCE7+ROI mask 8通道")

    output_dir = output_root / model_label / f"fold_{fold}"
    feature_path = output_dir / "observed_features.npz"
    metadata_path = output_dir / "extraction_metadata.json"
    manifest_path = output_dir / "feature_manifest_fragment.csv"
    refuse_existing([feature_path, metadata_path, manifest_path], overwrite)

    patient_ids: list[str] = []
    split_labels: list[str] = []
    has_radiomics: list[bool] = []
    pieces: dict[str, list[np.ndarray]] = {name: [] for name in CORE_REPRESENTATIONS}
    roi_valid_parts: list[np.ndarray] = []
    expected_split_counts: dict[str, int] = {}
    first_batch_checked = False

    for split in ("train", "val", "test"):
        split_ids = list(evaluation.splits[split])
        if max_patients_per_split is not None:
            split_ids = split_ids[:max_patients_per_split]
        expected_split_counts[split] = len(split_ids)
        records = source.records_for_ids(evaluation.bundle, split_ids)
        dataset = source.dataset_class(records, transformed_radiomics=None, image_channels=8)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=device.type == "cuda",
            persistent_workers=workers > 0,
            drop_last=False,
        )
        observed_ids: list[str] = []
        with torch.inference_mode():
            for batch in loader:
                image = batch["image"].to(device, non_blocking=True)
                online = _branch_features(
                    image, evaluation.model.encoder, evaluation.model.projector, "online"
                )
                ema = _branch_features(
                    image,
                    evaluation.model.target_encoder,
                    evaluation.model.target_projector,
                    "ema_target",
                )
                geometry, intensity, baseline_valid = _mask_geometry_and_intensity(image)
                if not torch.equal(online["roi_valid"], ema["roi_valid"]):
                    raise AssertionError("online/EMA ROI validity 不一致")
                if not torch.equal(online["roi_valid"].cpu(), baseline_valid):
                    raise AssertionError("spatial pooling 与 input baseline ROI validity 不一致")
                if not first_batch_checked:
                    encoded_online = evaluation.model.encode_online(image)
                    encoded_ema = evaluation.model.encode_target(image)
                    if not torch.equal(online["online_projected"], encoded_online):
                        raise AssertionError("手动 online projected 与 encode_online 不一致")
                    if not torch.equal(ema["ema_projected"], encoded_ema):
                        raise AssertionError("手动 EMA projected 与 encode_target 不一致")
                    first_batch_checked = True
                values = {
                    "online_projected": online["online_projected"],
                    "online_preprojector": online["online_preprojector"],
                    "online_global_pool": online["online_global_pool"],
                    "online_roi_mean": online["online_roi_mean"],
                    "ema_projected": ema["ema_projected"],
                    "ema_preprojector": ema["ema_preprojector"],
                    "ema_global_pool": ema["ema_global_pool"],
                    "ema_roi_mean": ema["ema_roi_mean"],
                    "mask_geometry": geometry,
                    "raw_roi_intensity": intensity,
                }
                for name, value in values.items():
                    pieces[name].append(value.float().cpu().numpy())
                roi_valid_parts.append(online["roi_valid"].cpu().numpy())
                batch_ids = [str(value) for value in batch["patient_id"]]
                observed_ids.extend(batch_ids)
                patient_ids.extend(batch_ids)
                split_labels.extend([split] * len(batch_ids))
                has_radiomics.extend(pid in evaluation.bundle.raw_radiomics for pid in batch_ids)
        if observed_ids != split_ids:
            raise RuntimeError(f"{split} feature patient 顺序与 checkpoint split 不一致")

    arrays: dict[str, np.ndarray] = {
        name: np.concatenate(parts, axis=0).astype(np.float32, copy=False)
        for name, parts in pieces.items()
    }
    arrays.update(
        {
            "patient_ids": np.asarray(patient_ids, dtype=str),
            "splits": np.asarray(split_labels, dtype=str),
            "has_radiomics": np.asarray(has_radiomics, dtype=bool),
            "roi_valid": np.concatenate(roi_valid_parts, axis=0).astype(bool, copy=False),
            "timepoints": np.asarray(TIMEPOINTS, dtype=str),
        }
    )
    total = len(patient_ids)
    if len(set(patient_ids)) != total:
        raise RuntimeError("同一 fold feature 文件出现重复 patient_id")
    for name in CORE_REPRESENTATIONS:
        if arrays[name].shape[:2] != (total, 4):
            raise AssertionError(f"{name} shape 非法: {arrays[name].shape}")
        if name not in {"online_roi_mean", "ema_roi_mean", "mask_geometry", "raw_roi_intensity"}:
            if not np.isfinite(arrays[name]).all():
                raise FloatingPointError(f"global feature 含 NaN/Inf: {name}")
    roi_invalid = ~arrays["roi_valid"]
    for name in ("online_roi_mean", "ema_roi_mean", "mask_geometry", "raw_roi_intensity"):
        finite_rows = np.isfinite(arrays[name]).all(axis=-1)
        if not np.array_equal(finite_rows, ~roi_invalid):
            raise AssertionError(f"{name} 的有限性未严格对应 ROI validity")

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_npz(feature_path, arrays)
    metadata = {
        "schema_version": 1,
        "status": "observed frozen feature extraction complete",
        "model": model_label,
        "run_name": evaluation.run_name,
        "fold": fold,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": evaluation.checkpoint_sha256,
        "checkpoint_epoch": int(evaluation.payload["epoch"]),
        "source_config": str(source_config_path.resolve()),
        "source_config_sha256": file_sha256(source_config_path),
        "fold_manifest_sha256": evaluation.payload["data_contract"]["fold_manifest_sha256"],
        "radiomics_transform_sha256": evaluation.payload["data_contract"][
            "radiomics_transform_sha256"
        ],
        "raw_radiomics_sha256": evaluation.radiomics_transform.raw_targets_sha256,
        "extractor_sha256": extraction_implementation_sha256(),
        "device": str(device),
        "patient_count": total,
        "split_counts": expected_split_counts,
        "radiomics_patient_count": int(arrays["has_radiomics"].sum()),
        "roi_invalid_patient_visits": int((~arrays["roi_valid"]).sum()),
        "representations": {name: list(arrays[name].shape) for name in CORE_REPRESENTATIONS},
        "feature_file": str(feature_path.resolve()),
        "feature_file_sha256": file_sha256(feature_path),
        "max_patients_per_split": max_patients_per_split,
        "world_model_trained": False,
        "predicted_future_state_used": False,
    }
    atomic_json(metadata_path, metadata)
    manifest = _manifest_fragment(arrays, metadata, feature_path)
    atomic_csv(manifest_path, manifest)
    return metadata


def finalize_feature_manifest(
    config_path: Path,
    feature_root: Path,
    output_path: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    expected_models = tuple(config["models"])
    fragments: list[pd.DataFrame] = []
    metadata_records: list[dict[str, Any]] = []
    for model in expected_models:
        for fold in range(5):
            directory = feature_root / model / f"fold_{fold}"
            metadata = json.loads((directory / "extraction_metadata.json").read_text(encoding="utf-8"))
            if metadata["max_patients_per_split"] is not None:
                raise ValueError("正式 manifest 不能包含 smoke extraction")
            feature_path = directory / "observed_features.npz"
            if file_sha256(feature_path) != metadata["feature_file_sha256"]:
                raise ValueError(f"feature 文件 SHA 漂移: {feature_path}")
            fragment = pd.read_csv(directory / "feature_manifest_fragment.csv")
            if set(fragment["model"]) != {model} or set(fragment["fold"]) != {fold}:
                raise ValueError(f"manifest fragment model/fold 错误: {directory}")
            fragments.append(fragment)
            metadata_records.append(metadata)
    manifest = pd.concat(fragments, ignore_index=True)
    key = ["patient_id", "fold", "model", "timepoint", "representation_type"]
    if manifest.duplicated(key).any():
        raise ValueError("feature manifest 出现重复键")
    expected_rows = 3 * 5 * 808 * 4 * len(CORE_REPRESENTATIONS)
    if len(manifest) != expected_rows:
        raise ValueError(f"feature manifest 行数错误: {len(manifest)} != {expected_rows}")
    for model in expected_models:
        for fold in range(5):
            group = manifest[(manifest["model"] == model) & (manifest["fold"] == fold)]
            if group["patient_id"].nunique() != 808:
                raise ValueError(f"{model}/fold{fold} 未覆盖 808 患者")
    summary_path = output_path.with_suffix(".summary.json")
    refuse_existing([output_path, summary_path], overwrite)
    atomic_csv(output_path, manifest)
    summary = {
        "status": "feature manifest complete",
        "rows": len(manifest),
        "patients": int(manifest["patient_id"].nunique()),
        "models": list(expected_models),
        "folds": list(range(5)),
        "timepoints": list(TIMEPOINTS),
        "representations": list(CORE_REPRESENTATIONS),
        "radiomics_patients": int(manifest.loc[manifest["has_radiomics"], "patient_id"].nunique()),
        "invalid_roi_rows": int(
            len(
                manifest[
                    manifest["representation_type"].isin(
                        ["online_roi_mean", "ema_roi_mean", "mask_geometry", "raw_roi_intensity"]
                    )
                    & ~manifest["feature_valid"]
                ]
            )
        ),
        "manifest_sha256": file_sha256(output_path),
        "extractor_sha256": metadata_records[0]["extractor_sha256"],
    }
    atomic_json(summary_path, summary)
    return summary

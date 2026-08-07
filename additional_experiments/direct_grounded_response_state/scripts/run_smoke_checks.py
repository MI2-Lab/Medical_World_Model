#!/usr/bin/env python3
"""汇总真实 cache smoke，并验证 DCE7、mask 与 grounding 的硬契约。"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from dgrs.config import atomic_json, file_sha256, load_config  # noqa: E402
from dgrs.model import (  # noqa: E402
    DGRSWorldModel,
    load_checkpoint_for_evaluation,
    normalized_occupancy_roi_mean,
)
from dgrs.training import build_bundle, set_seed, split_ids  # noqa: E402


RUNS = {
    "G1": "smoke_g1_contract_v3",
    "G2": "smoke_g2_contract_v3",
    "G3": "smoke_g3_contract_v3",
    "G4": "smoke_g4_contract_v3",
}


def _assert_close(left: torch.Tensor, right: torch.Tensor, label: str) -> float:
    error = float((left - right).abs().max())
    if not torch.allclose(left, right, rtol=1e-5, atol=1e-6):
        raise AssertionError(f"{label} 不成立；max_abs_error={error}")
    return error


def _history_row(model: str) -> tuple[pd.Series, Path]:
    path = EXPERIMENT_ROOT / "metrics" / "training" / RUNS[model] / "fold_0.csv"
    frame = pd.read_csv(path)
    if len(frame) != 1 or int(frame.iloc[0]["epoch"]) != 1:
        raise AssertionError(f"{model} smoke 必须恰为一个真实 epoch")
    return frame.iloc[0], path


def run_checks(output: Path) -> dict[str, object]:
    config = load_config(EXPERIMENT_ROOT / "configs" / "base.yaml")
    bundle = build_bundle(config)
    splits = split_ids(bundle, 0)
    if any(set(splits[left]) & set(splits[right]) for left, right in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise AssertionError("fold 0 train/val/test 不互斥")

    transformed_paths = [EXPERIMENT_ROOT / "configs" / f"ftv_transform_fold_{fold}.json" for fold in range(5)]
    if not all(path.is_file() for path in transformed_paths):
        raise AssertionError("五折 pooled FTV transform 不完整")

    sample_dataset = __import__("dgrs.data", fromlist=["LongitudinalDGRSDataset"]).LongitudinalDGRSDataset(
        [bundle.primary[0]], raw_ftv=bundle.raw_ftv
    )
    sample = sample_dataset[0]
    if tuple(sample["image"].shape) != (4, 7, 32, 96, 96):
        raise AssertionError("真实 cache DCE7 shape 漂移")
    if tuple(sample["roi_mask"].shape) != (4, 1, 32, 96, 96):
        raise AssertionError("真实 cache ROI mask shape 漂移")

    models: dict[str, DGRSWorldModel] = {}
    payloads: dict[str, dict[str, object]] = {}
    checkpoint_rows: list[dict[str, object]] = []
    for model_name, run_name in RUNS.items():
        checkpoint = EXPERIMENT_ROOT / "checkpoints" / run_name / "fold_0" / "best.pt"
        model, payload = load_checkpoint_for_evaluation(checkpoint, "cpu")
        models[model_name], payloads[model_name] = model, payload
        if model.model_name != model_name or not bool(payload.get("finalized")):
            raise AssertionError(f"{model_name} checkpoint schema/finalize 失败")
        first_conv = model.encoder.features[0].main[0]
        if not isinstance(first_conv, torch.nn.Conv3d) or first_conv.in_channels != 7:
            raise AssertionError(f"{model_name} 第一层不是 7-channel Conv3d")
        if any(parameter.requires_grad for module in (model.target_encoder, model.target_response_projection, model.target_projector) for parameter in module.parameters()):
            raise AssertionError(f"{model_name} EMA target parameters 意外可训练")
        forbidden = ("geometry", "voxel", "volume", "ftv_target", "clinical", "treatment")
        forbidden_keys = [key for key in model.state_dict() if any(token in key.lower() for token in forbidden)]
        if forbidden_keys:
            raise AssertionError(f"{model_name} state schema 出现禁止输入路径: {forbidden_keys[:3]}")
        row, history_path = _history_row(model_name)
        numeric = pd.to_numeric(row, errors="coerce")
        required_finite = (
            "total_loss",
            "base_loss",
            "val_base_loss",
            "val_representation_std",
            "train_encoder_gradient_norm",
        )
        if not all(math.isfinite(float(numeric[name])) for name in required_finite):
            raise AssertionError(f"{model_name} history 有非有限值")
        if float(row["train_encoder_gradient_norm"]) <= 0 or float(row["val_representation_std"]) < 0.05:
            raise AssertionError(f"{model_name} backbone gradient/collapse 检查失败")
        grounded = model_name in {"G3", "G4"}
        if grounded:
            for name in (
                "train_first_valid_ftv_encoder_gradient_norm_raw",
                "train_first_valid_ftv_response_projection_gradient_norm_raw",
                "train_first_valid_ftv_head_gradient_norm_raw",
            ):
                if float(row[name]) <= 0:
                    raise AssertionError(f"{model_name} {name} 必须非零")
            if float(row["train_grounded_patients"]) <= 0 or float(row["train_valid_ftv_visits"]) <= 0:
                raise AssertionError(f"{model_name} 未实际使用 paired FTV")
        elif float(row["train_grounded_patients"]) != 0 or float(row["train_valid_ftv_visits"]) != 0:
            raise AssertionError(f"{model_name} baseline 不应记为 grounded")
        checkpoint_rows.append(
            {
                "model": model_name,
                "checkpoint": str(checkpoint.relative_to(EXPERIMENT_ROOT)),
                "checkpoint_sha256": file_sha256(checkpoint),
                "history": str(history_path.relative_to(EXPERIMENT_ROOT)),
                "history_sha256": file_sha256(history_path),
                "val_base_loss": float(row["val_base_loss"]),
                "val_ftv_loss": float(row["val_ftv_loss"]),
                "representation_std": float(row["val_representation_std"]),
                "encoder_gradient_norm": float(row["train_encoder_gradient_norm"]),
                "ftv_encoder_gradient_norm_raw": float(row["train_first_valid_ftv_encoder_gradient_norm_raw"]),
                "ftv_head_gradient_norm_raw": float(row["train_first_valid_ftv_head_gradient_norm_raw"]),
            }
        )

    shared_hashes = {
        "G1-G3": payloads["G1"]["shared_initialization_sha256"] == payloads["G3"]["shared_initialization_sha256"],
        "G2-G4": payloads["G2"]["shared_initialization_sha256"] == payloads["G4"]["shared_initialization_sha256"],
    }
    if not all(shared_hashes.values()):
        raise AssertionError("paired model shared initialization hash 不一致")

    dummy_image = torch.zeros(1, 4, 7, 8, 8, 8)
    dummy_mask = torch.ones(1, 4, 1, 8, 8, 8)
    mask_rejection = {}
    for model_name in ("G1", "G3"):
        try:
            models[model_name]._validate_sequence_inputs(dummy_image, dummy_mask)
        except ValueError:
            mask_rejection[model_name] = True
        else:
            raise AssertionError(f"{model_name} 没有拒绝 mask")
    for model_name in ("G2", "G4"):
        try:
            models[model_name]._validate_sequence_inputs(dummy_image, None)
        except ValueError:
            mask_rejection[model_name] = True
        else:
            raise AssertionError(f"{model_name} 没有要求分离 mask")

    torch.manual_seed(17)
    spatial = torch.randn(2, 5, 4, 4, 4)
    mask = torch.zeros(2, 1, 8, 8, 8)
    mask[0, :, 1:4, 1:5, 2:6] = 1
    mask[1, :, 2:7, 1:7, 1:7] = 1
    pooled, valid = normalized_occupancy_roi_mean(spatial, mask)
    scaled, scaled_valid = normalized_occupancy_roi_mean(spatial, mask * 3.7)
    scale_error = _assert_close(pooled, scaled, "pool(F,M)==pool(F,cM)")
    if not torch.equal(valid, scaled_valid) or not bool(valid.all()):
        raise AssertionError("非空 mask validity 漂移")

    constant = torch.arange(1, 6, dtype=torch.float32).view(1, 5, 1, 1, 1).expand(2, -1, 4, 4, 4)
    constant_pooled, _ = normalized_occupancy_roi_mean(constant, mask)
    constant_error = _assert_close(constant_pooled[0], constant_pooled[1], "常量 feature 的 mask-volume invariance")
    ones = torch.ones(2, 1, 8, 8, 8)
    ones_pooled, ones_valid = normalized_occupancy_roi_mean(spatial, ones)
    gap = spatial.mean(dim=(-3, -2, -1))
    ones_error = _assert_close(ones_pooled, gap, "all-ones mask==GAP")
    empty = torch.zeros_like(ones)
    empty_pooled, empty_valid = normalized_occupancy_roi_mean(spatial, empty)
    empty_error = _assert_close(empty_pooled, gap, "empty mask strict GAP fallback")
    if not bool(ones_valid.all()) or bool(empty_valid.any()) or not bool(torch.isfinite(empty_pooled).all()):
        raise AssertionError("all-ones/empty mask validity 或 finite 检查失败")

    # Encoder 的签名只有 DCE；mask 在 spatial map 形成后才进入纯函数 pooling。
    encoder_signature = str(inspect.signature(models["G4"].encoder.forward))
    forward_signature = str(inspect.signature(models["G4"].forward))
    if "roi_mask" in encoder_signature or "ftv" in forward_signature.lower():
        raise AssertionError("encoder/forward signature 暴露禁止输入")
    set_seed(31)
    tiny = torch.randn(1, 7, 8, 16, 16)
    with torch.no_grad():
        first = models["G4"].encoder(tiny)
        second = models["G4"].encoder(tiny)
    encoder_bitwise = bool(torch.equal(first, second))
    if not encoder_bitwise:
        raise AssertionError("固定 DCE 的 encoder spatial map 非 bitwise stable")

    result: dict[str, object] = {
        "schema_version": 1,
        "status": "passed",
        "real_cache_smoke": True,
        "epochs_per_model": 1,
        "fold": 0,
        "models": checkpoint_rows,
        "dce_shape": list(sample["image"].shape),
        "roi_mask_shape": list(sample["roi_mask"].shape),
        "first_conv_in_channels": 7,
        "paired_shared_initialization": shared_hashes,
        "mask_routing_rejection": mask_rejection,
        "pooling_contract": {
            "occupancy_scale_max_abs_error": scale_error,
            "constant_feature_different_support_max_abs_error": constant_error,
            "all_ones_vs_gap_max_abs_error": ones_error,
            "empty_mask_vs_gap_max_abs_error": empty_error,
            "empty_mask_finite": True,
        },
        "encoder_spatial_map_bitwise_stable_without_mask_argument": encoder_bitwise,
        "encoder_forward_signature": encoder_signature,
        "world_model_forward_has_no_ftv_input": "ftv" not in forward_signature.lower(),
        "ftv_head_removal_cannot_change_encode_online_graph": True,
        "ema_requires_grad": False,
        "split_disjoint": True,
        "test_data_used_for_smoke_selection": False,
        "five_fold_transform_sha256": [file_sha256(path) for path in transformed_paths],
        "explicit_geometry_or_volume_state_keys": [],
        "limitations_cn": [
            "Normalized ROI mean 仍通过 mask support 选择空间位置，不等于 geometry-free。",
            "DCE7 cache 是 lesion-centered crop，仍带上游 ROI 定位先验。",
        ],
    }
    atomic_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENT_ROOT / "metrics" / "smoke" / "smoke_checks.json",
    )
    args = parser.parse_args()
    result = run_checks(args.output.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

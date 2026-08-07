#!/usr/bin/env python3
"""真实 manifest/target 对齐检查与小张量 forward/backward 契约测试。"""

from __future__ import annotations

import inspect
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from rnc.config import load_config  # noqa: E402
from rnc.data import FEATURE_NAMES, patient_hash, split_ids  # noqa: E402
from rnc.losses import NextChangeObjective  # noqa: E402
from rnc.model import ImageOnlyWorldModel  # noqa: E402
from rnc.training import build_bundle, ensure_radiomics_transform  # noqa: E402


def main() -> None:
    config = load_config(EXPERIMENT_ROOT / "configs" / "base.yaml")
    bundle = build_bundle(config)
    alignment = []
    for fold in range(5):
        splits = split_ids(bundle, fold)
        transform = ensure_radiomics_transform(bundle, fold, splits["train"])
        transformed = transform.transform_all(bundle.raw_radiomics)
        if transform.train_patient_hash != patient_hash(splits["train"]):
            raise AssertionError("fold transform patient hash 错误")
        if set(splits["train"]) & set(splits["test"]):
            raise AssertionError("train/test patient 泄漏")
        valid_elements = int(sum(mask.sum() for _, mask in transformed.values()))
        if valid_elements != 375 * 3 * len(FEATURE_NAMES):
            raise AssertionError(f"radiomics mask 数量错误: {valid_elements}")
        alignment.append(
            {
                "fold": fold,
                "train": len(splits["train"]),
                "val": len(splits["val"]),
                "test": len(splits["test"]),
                "paired_train": transform.paired_train_patient_count,
                "transform_hash": transform.train_patient_hash,
            }
        )

    forbidden = {"clinical", "treatment", "geometry", "radiomics"}
    forward_parameters = set(inspect.signature(ImageOnlyWorldModel.forward).parameters)
    if forbidden & forward_parameters:
        raise AssertionError("模型 forward 暴露了禁止输入")

    shape_results = []
    for mode in ("m0", "m1_delta_only", "m1", "m2"):
        torch.manual_seed(7)
        model = ImageOnlyWorldModel(
            mode=mode,
            image_channels=8,
            base_channels=2,
            latent_dim=16,
            predictor_depth=1,
            predictor_heads=4,
            predictor_mlp_dim=32,
            dropout=0.0,
        )
        image = torch.randn(2, 4, 8, 8, 16, 16)
        target = torch.randn(2, 3, 4)
        mask = torch.zeros(2, 3, 4, dtype=torch.bool)
        mask[0] = True
        output = model(image)
        objective = NextChangeObjective(mode, lambda_rad=0.1, sigreg_projections=8)
        loss, stats = objective(output, target, mask)
        loss.backward()
        gradients = [parameter.grad for parameter in model.transition.parameters() if parameter.grad is not None]
        if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise AssertionError(f"{mode} transition gradient 无效")
        if mode == "m2":
            assert output.radiomics_prediction is not None and output.radiomics_prediction.shape == (2, 3, 4)
        else:
            assert output.radiomics_prediction is None
        model.eval()
        feature, details = model.readout_feature(image[:, :2])
        if feature.shape != (2, 48) or details["predicted_delta"].shape != (2, 16):
            raise AssertionError(f"{mode} readout shape 错误")
        shape_results.append(
            {
                "mode": mode,
                "loss": float(loss.detach()),
                "state_shape": list(output.target_state.shape),
                "prediction_shape": list(output.predicted_next.shape),
                "readout_shape": list(feature.shape),
                "finite_stats": bool(all(torch.isfinite(value).all() for value in stats.values())),
            }
        )

    # 回归测试：lambda_rad=0 时，M2 auxiliary head 不能改变任何共同随机流/更新。
    equivalence_kwargs = {
        "image_channels": 8,
        "base_channels": 2,
        "latent_dim": 16,
        "predictor_depth": 1,
        "predictor_heads": 4,
        "predictor_mlp_dim": 32,
        "dropout": 0.1,
    }
    torch.manual_seed(91)
    m1 = ImageOnlyWorldModel("m1", **equivalence_kwargs)
    m1_rng = torch.get_rng_state().clone()
    torch.manual_seed(91)
    m2 = ImageOnlyWorldModel("m2", **equivalence_kwargs)
    m2_rng = torch.get_rng_state().clone()
    if not torch.equal(m1_rng, m2_rng):
        raise AssertionError("M2 head 初始化推进了共同 RNG")
    optimizer_m1 = torch.optim.AdamW((p for p in m1.parameters() if p.requires_grad), lr=1e-4)
    optimizer_m2 = torch.optim.AdamW((p for p in m2.parameters() if p.requires_grad), lr=1e-4)
    objective_m1 = NextChangeObjective("m1", sigreg_projections=8)
    objective_m2 = NextChangeObjective("m2", lambda_rad=0.0, sigreg_projections=8)
    generator = torch.Generator().manual_seed(1234)
    for _ in range(2):
        image = torch.randn(2, 4, 8, 8, 16, 16, generator=generator)
        target = torch.randn(2, 3, 4, generator=generator)
        mask = torch.ones(2, 3, 4, dtype=torch.bool)
        torch.set_rng_state(m1_rng)
        optimizer_m1.zero_grad(set_to_none=True)
        output_m1 = m1(image)
        loss_m1, _ = objective_m1(output_m1, target, mask)
        loss_m1.backward()
        optimizer_m1.step()
        m1_rng = torch.get_rng_state().clone()
        torch.set_rng_state(m2_rng)
        optimizer_m2.zero_grad(set_to_none=True)
        output_m2 = m2(image)
        loss_m2, _ = objective_m2(output_m2, target, mask)
        loss_m2.backward()
        optimizer_m2.step()
        m2_rng = torch.get_rng_state().clone()
        if not torch.equal(loss_m1, loss_m2):
            raise AssertionError("lambda_rad=0 的 M1/M2 loss 不一致")
        common_m2 = {name: value for name, value in m2.state_dict().items() if not name.startswith("radiomics_head.")}
        if not all(torch.equal(value, common_m2[name]) for name, value in m1.state_dict().items()):
            raise AssertionError("lambda_rad=0 的 M1/M2 common state 更新不一致")

    # 回归测试：普通 str 版本字段可由默认 weights-only 安全加载，并 strict reload。
    with tempfile.TemporaryDirectory(prefix="rnc_checkpoint_validation_") as temporary:
        checkpoint_path = Path(temporary) / "best.pt"
        torch.save(
            {
                "model_state": m2.state_dict(),
                "torch_version": str(torch.__version__),
                "cuda_version": None if torch.version.cuda is None else str(torch.version.cuda),
            },
            checkpoint_path,
        )
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        reloaded = ImageOnlyWorldModel("m2", **equivalence_kwargs)
        reloaded.load_state_dict(loaded["model_state"], strict=True)

    payload = {
        "status": "通过",
        "primary_patients": len(bundle.primary),
        "extra_pretrain_patients": len(bundle.extra_pretrain),
        "radiomics_patients": len(bundle.raw_radiomics),
        "fold_alignment": alignment,
        "model_contract_tests": shape_results,
        "lambda_zero_common_update_bitwise_equal": True,
        "weights_only_checkpoint_strict_reload": True,
    }
    output = EXPERIMENT_ROOT / "metrics" / "implementation_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""冻结 CoRe-JEPA 的 audit-only inference 与 transition 明细导出。"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from .metrics import cosine_similarity, layer_norm_mse, transition_metrics
from .perturbations import (
    PerturbedInputs,
    predict_perturbed_context_against_native_target,
)


TRANSITION_NAMES = ("T0->T1", "T1->T2", "T2->T3")


@dataclass(frozen=True)
class FrozenInferenceArrays:
    patient_ids: np.ndarray
    response_state: np.ndarray
    t0_image_state: np.ndarray
    geometry: np.ndarray
    condition: np.ndarray

    @property
    def t0_state(self) -> np.ndarray:
        """兼容别名；其值是纯 encoder+projector appearance，不含 geometry。"""

        return self.t0_image_state


def _require_eval(model: torch.nn.Module) -> None:
    training = [name or "<root>" for name, module in model.named_modules() if module.training]
    if training:
        raise ValueError(f"audit inference 要求 model 全部处于 eval；仍为 train：{training[:5]}")


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _patient_ids(batch: dict[str, Any], batch_size: int) -> list[str]:
    values = batch.get("patient_id")
    if values is None or len(values) != batch_size:
        raise ValueError("batch 必须提供与 batch size 对齐的 patient_id")
    return [str(value) for value in values]


def _transition_metadata(
    patient_ids: list[str], *, fold: int, checkpoint: str, audit_condition: str
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": np.repeat(patient_ids, 3),
            "fold": int(fold),
            "transition": np.tile(TRANSITION_NAMES, len(patient_ids)),
            "audit_condition": audit_condition,
            "checkpoint": checkpoint,
        }
    )


@torch.no_grad()
def copy_current_latent_audit(
    model: torch.nn.Module,
    loader: Iterable[dict[str, Any]],
    *,
    device: str | torch.device,
    fold: int,
    checkpoint: str,
) -> pd.DataFrame:
    """按患者/transition 导出 learned 与指定 online-current copy 的 JEPA 指标。"""

    _require_eval(model)
    device = torch.device(device)
    rows: list[pd.DataFrame] = []
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        output = model(batch["image"], batch["geometry"], batch["condition"])
        batch_size = int(batch["image"].shape[0])
        metadata = _transition_metadata(
            _patient_ids(batch, batch_size),
            fold=fold,
            checkpoint=checkpoint,
            audit_condition="copy_current",
        )
        rows.append(
            transition_metrics(
                output.prediction.detach().cpu().numpy(),
                output.visit_state[:, :-1].detach().cpu().numpy(),
                output.target.detach().cpu().numpy(),
                metadata=metadata,
            )
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "sample_index",
                "patient_id",
                "fold",
                "transition",
                "audit_condition",
                "checkpoint",
            ]
        )
    output = pd.concat(rows, ignore_index=True)
    output["sample_index"] = np.arange(len(output), dtype=np.int64)
    return output


@torch.no_grad()
def collect_frozen_inference(
    model: torch.nn.Module,
    loader: Iterable[dict[str, Any]],
    *,
    device: str | torch.device,
    perturbation: Callable[[Any, Any, Any], PerturbedInputs] | None = None,
) -> FrozenInferenceArrays:
    """收集 primary response state 与 frozen T0 state；T0 state 始终来自 native 输入。"""

    _require_eval(model)
    device = torch.device(device)
    patient_ids: list[str] = []
    response_states: list[np.ndarray] = []
    t0_states: list[np.ndarray] = []
    geometries: list[np.ndarray] = []
    conditions: list[np.ndarray] = []
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        image, geometry, condition = (
            batch["image"],
            batch["geometry"],
            batch["condition"],
        )
        batch_size = int(image.shape[0])
        patient_ids.extend(_patient_ids(batch, batch_size))
        encoder = getattr(model, "encoder", None)
        projector = getattr(model, "projector", None)
        if not callable(encoder) or not callable(projector):
            raise TypeError("F5 static-T0 需要 model.encoder 与 model.projector")
        # 故意不调用 encode_visits；后者会叠加 geometry_projector(q0)。
        native_t0 = projector(encoder(image[:, 0]))
        inputs = (
            perturbation(image, geometry, condition)
            if perturbation is not None
            else PerturbedInputs(image=image, geometry=geometry, condition=condition)
        )
        state = model.forecast_response(inputs.geometry, inputs.condition)
        response_states.append(state.detach().cpu().numpy())
        t0_states.append(native_t0.detach().cpu().numpy())
        geometries.append(inputs.geometry.detach().cpu().numpy())
        conditions.append(inputs.condition.detach().cpu().numpy())
    if not patient_ids:
        raise ValueError("inference loader 为空")
    return FrozenInferenceArrays(
        patient_ids=np.asarray(patient_ids),
        response_state=np.concatenate(response_states).astype(np.float32),
        t0_image_state=np.concatenate(t0_states).astype(np.float32),
        geometry=np.concatenate(geometries).astype(np.float32),
        condition=np.concatenate(conditions).astype(np.float32),
    )


@torch.no_grad()
def paired_perturbation_latent_audit(
    model: torch.nn.Module,
    loader: Iterable[dict[str, Any]],
    perturbation: Callable[[Any, Any, Any], PerturbedInputs],
    *,
    device: str | torch.device,
    fold: int,
    checkpoint: str,
    audit_condition: str,
) -> pd.DataFrame:
    """比较 native/perturbed prediction 对同一 native EMA target 的误差。

    同时导出 FutureResponseState 的 paired 变化。该 state 只依赖 geometry/condition，
    因而 C1 的 state change 应严格为零；这属于架构契约而非经验性能结论。
    """

    _require_eval(model)
    device = torch.device(device)
    frames: list[pd.DataFrame] = []
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        image, geometry, condition = (
            batch["image"],
            batch["geometry"],
            batch["condition"],
        )
        batch_size = int(image.shape[0])
        ids = _patient_ids(batch, batch_size)
        perturbed = perturbation(image, geometry, condition)
        native_output = model(image, geometry, condition)
        paired = predict_perturbed_context_against_native_target(
            model,
            native_image=image,
            native_geometry=geometry,
            perturbed_image=perturbed.image,
            perturbed_geometry=perturbed.geometry,
            condition=perturbed.condition,
        )
        native_prediction = native_output.prediction.detach().cpu().numpy()
        native_target = native_output.target.detach().cpu().numpy()
        perturbed_prediction = paired.prediction.detach().cpu().numpy()
        paired_target = paired.native_target.detach().cpu().numpy()
        if not np.allclose(native_target, paired_target, rtol=1e-6, atol=1e-7):
            raise RuntimeError("paired perturbation 未固定同一 native EMA target")

        native_state = native_output.future_response_state.detach().cpu().numpy()
        perturbed_state = (
            model.forecast_response(perturbed.geometry, perturbed.condition)
            .detach()
            .cpu()
            .numpy()
        )
        difference = perturbed_state - native_state
        state_cosine = cosine_similarity(perturbed_state, native_state)
        frame = _transition_metadata(
            ids,
            fold=fold,
            checkpoint=checkpoint,
            audit_condition=audit_condition,
        )
        frame["native_layer_norm_mse"] = layer_norm_mse(
            native_prediction, native_target
        ).reshape(-1)
        frame["perturbed_layer_norm_mse"] = layer_norm_mse(
            perturbed_prediction, native_target
        ).reshape(-1)
        frame["latent_error_change"] = (
            frame["perturbed_layer_norm_mse"] - frame["native_layer_norm_mse"]
        )
        frame["native_cosine_similarity"] = cosine_similarity(
            native_prediction, native_target
        ).reshape(-1)
        frame["perturbed_cosine_similarity"] = cosine_similarity(
            perturbed_prediction, native_target
        ).reshape(-1)
        frame["response_state_mean_abs_change"] = np.abs(difference).mean(axis=-1).reshape(-1)
        frame["response_state_l2_change"] = np.linalg.norm(difference, axis=-1).reshape(-1)
        frame["response_state_cosine_similarity"] = state_cosine.reshape(-1)
        frames.append(frame)
    if not frames:
        raise ValueError("perturbation loader 为空")
    return pd.concat(frames, ignore_index=True)


__all__ = [
    "FrozenInferenceArrays",
    "TRANSITION_NAMES",
    "collect_frozen_inference",
    "copy_current_latent_audit",
    "paired_perturbation_latent_audit",
]

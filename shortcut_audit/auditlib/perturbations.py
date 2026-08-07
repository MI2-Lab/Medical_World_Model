"""CoRe-WM shortcut audit 的无副作用输入扰动。

本模块只构造评估输入，不修改数据集、模型或原始缓存。公开扰动同时支持
``numpy.ndarray`` 与 ``torch.Tensor``，并始终返回独立副本。输入约定为单患者
``image [4,8,Z,Y,X]`` / ``geometry [4,9]`` / ``condition [3,C]``，或带批次维的
``[B,4,8,Z,Y,X]`` / ``[B,4,9]`` / ``[B,3,C]``。

通道 0:7 是 MRI 派生强度通道，通道 7 是 lesion ROI mask。``condition`` 包含
nominal 时间点编码，因此任何扰动都只复制其值、绝不交换其行。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # numpy-only 环境仍可使用输入扰动；latent helper 会给出明确错误。
    import torch
except ModuleNotFoundError:  # pragma: no cover - bowen 环境安装了 torch
    torch = None  # type: ignore[assignment]


N_VISITS = 4
N_IMAGE_CHANNELS = 8
N_MRI_CHANNELS = 7
N_GEOMETRY_FEATURES = 9
N_TRANSITIONS = 3


@dataclass(frozen=True)
class PerturbedInputs:
    """相互独立的扰动后输入。

    ``condition`` 是输入 condition 的独立副本，但数值和 nominal 行顺序完全不变。
    这样既不会误换 temporal condition，也不会因下游原地操作污染原对象。
    """

    image: Any
    geometry: Any
    condition: Any


@dataclass(frozen=True)
class NativeTargetLatentPrediction:
    """扰动 context 的预测以及由未扰动序列编码的 EMA target。"""

    prediction: Any
    native_target: Any
    image_prediction: Any
    response_correction: Any

    @property
    def target(self) -> Any:
        """``native_target`` 的兼容别名，便于复用现有误差计算代码。"""

        return self.native_target


def _is_torch_tensor(value: Any) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


def _backend(value: Any, name: str) -> str:
    if isinstance(value, np.ndarray):
        return "numpy"
    if _is_torch_tensor(value):
        return "torch"
    raise TypeError(f"{name} 必须是 numpy.ndarray 或 torch.Tensor，实际为 {type(value).__name__}")


def _clone(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy(order="K")
    # _validate_inputs 已保证这里只会收到 torch.Tensor。
    return value.clone(memory_format=torch.preserve_format)


def _validate_inputs(image: Any, geometry: Any, condition: Any, *, label: str) -> str:
    """验证单样本/批量的完整四时间点 tensor contract。"""

    image_backend = _backend(image, f"{label}.image")
    geometry_backend = _backend(geometry, f"{label}.geometry")
    condition_backend = _backend(condition, f"{label}.condition")
    if not (image_backend == geometry_backend == condition_backend):
        raise TypeError(
            f"{label} 的 image、geometry、condition 必须使用同一后端，实际为 "
            f"{image_backend}/{geometry_backend}/{condition_backend}"
        )

    if image.ndim not in (5, 6):
        raise ValueError(
            f"{label}.image 必须为 [4,8,Z,Y,X] 或 [B,4,8,Z,Y,X]，实际为 {tuple(image.shape)}"
        )
    batch_prefix = tuple(image.shape[:-5])
    if tuple(image.shape[-5:-3]) != (N_VISITS, N_IMAGE_CHANNELS):
        raise ValueError(
            f"{label}.image 的 visit/channel 必须为 [4,8]，实际为 {tuple(image.shape)}"
        )
    if any(int(size) <= 0 for size in image.shape[-3:]):
        raise ValueError(f"{label}.image 的空间尺寸必须全部大于 0，实际为 {tuple(image.shape[-3:])}")

    expected_geometry = batch_prefix + (N_VISITS, N_GEOMETRY_FEATURES)
    if tuple(geometry.shape) != expected_geometry:
        raise ValueError(
            f"{label}.geometry 必须为 {expected_geometry}，实际为 {tuple(geometry.shape)}"
        )
    expected_condition_prefix = batch_prefix + (N_TRANSITIONS,)
    if condition.ndim != len(expected_condition_prefix) + 1 or tuple(condition.shape[:-1]) != expected_condition_prefix:
        expected = expected_condition_prefix + ("C",)
        raise ValueError(
            f"{label}.condition 必须为 {expected} 且 C>0，实际为 {tuple(condition.shape)}"
        )
    if int(condition.shape[-1]) <= 0:
        raise ValueError(f"{label}.condition 的特征维 C 必须大于 0")

    if image_backend == "torch":
        devices = (image.device, geometry.device, condition.device)
        if not (devices[0] == devices[1] == devices[2]):
            raise ValueError(
                f"{label} 的 image、geometry、condition 必须位于同一 device，实际为 {devices}"
            )
    return image_backend


def _image_index(ndim: int, visit: int | slice, channel: int | slice = slice(None)) -> tuple[Any, ...]:
    index: list[Any] = [slice(None)] * ndim
    index[-5] = visit
    index[-4] = channel
    return tuple(index)


def _geometry_index(ndim: int, visit: int | slice) -> tuple[Any, ...]:
    index: list[Any] = [slice(None)] * ndim
    index[-2] = visit
    return tuple(index)


def _cloned_inputs(image: Any, geometry: Any, condition: Any) -> PerturbedInputs:
    return PerturbedInputs(_clone(image), _clone(geometry), _clone(condition))


def repeated_t0_mri_only(image: Any, geometry: Any, condition: Any) -> PerturbedInputs:
    """构造 Repeated-T0 C1（MRI-only replacement）。

    T1/T2 的 MRI channels ``0:7`` 复制自同一患者 T0；各时间点原 ROI channel 7、
    geometry 和 nominal condition 均保留。T3 也保持原样。
    """

    _validate_inputs(image, geometry, condition, label="native")
    result = _cloned_inputs(image, geometry, condition)
    followups = _image_index(image.ndim, slice(1, 3), slice(0, N_MRI_CHANNELS))
    baseline = _image_index(image.ndim, slice(0, 1), slice(0, N_MRI_CHANNELS))
    result.image[followups] = image[baseline]
    return result


def repeated_t0_full_image_derived(image: Any, geometry: Any, condition: Any) -> PerturbedInputs:
    """构造 Repeated-T0 C2（主要结果）。

    T1/T2 的全部 8 个 image channels（包括 ROI mask）与 geometry 均复制自 T0；
    clinical/treatment/nominal temporal condition 原值保留。T3 保持原样。
    """

    _validate_inputs(image, geometry, condition, label="native")
    result = _cloned_inputs(image, geometry, condition)
    result.image[_image_index(image.ndim, slice(1, 3))] = image[
        _image_index(image.ndim, slice(0, 1))
    ]
    result.geometry[_geometry_index(geometry.ndim, slice(1, 3))] = geometry[
        _geometry_index(geometry.ndim, slice(0, 1))
    ]
    return result


def swap_t1_t2(image: Any, geometry: Any, condition: Any) -> PerturbedInputs:
    """构造 temporal-order T1/T2 swap。

    T1 和 T2 的全部 8 个 image channels 与 geometry 作为整体交换；T0、T3 以及
    nominal condition 不交换。赋值始终读取原输入，避免连续原地交换造成重复值。
    """

    _validate_inputs(image, geometry, condition, label="native")
    result = _cloned_inputs(image, geometry, condition)
    result.image[_image_index(image.ndim, 1)] = image[_image_index(image.ndim, 2)]
    result.image[_image_index(image.ndim, 2)] = image[_image_index(image.ndim, 1)]
    result.geometry[_geometry_index(geometry.ndim, 1)] = geometry[
        _geometry_index(geometry.ndim, 2)
    ]
    result.geometry[_geometry_index(geometry.ndim, 2)] = geometry[
        _geometry_index(geometry.ndim, 1)
    ]
    return result


def replace_followups_with_donor(
    recipient_image: Any,
    recipient_geometry: Any,
    recipient_condition: Any,
    *,
    donor_image: Any,
    donor_geometry: Any,
) -> PerturbedInputs:
    """用配对 donor 的 T1/T2 完整 image+geometry 替换 recipient follow-up。

    recipient 的 T0、T3 及全部 condition 保留。函数故意不接收 donor condition，
    从接口层面防止 clinical、treatment 或 nominal 时间点信息随 donor 泄漏。
    donor 的筛选/匹配应由独立的 baseline-only matching 逻辑完成。
    """

    backend = _validate_inputs(
        recipient_image, recipient_geometry, recipient_condition, label="recipient"
    )
    # donor condition 不参与扰动；仅用 recipient condition 验证 donor 的 batch convention。
    donor_condition_placeholder = _clone(recipient_condition)
    donor_backend = _validate_inputs(
        donor_image, donor_geometry, donor_condition_placeholder, label="donor"
    )
    if donor_backend != backend:  # _validate_inputs 已覆盖，保留明确的跨对象诊断。
        raise TypeError("recipient 与 donor 必须使用同一数组后端")
    if tuple(donor_image.shape) != tuple(recipient_image.shape):
        raise ValueError(
            "donor.image 必须与 recipient.image 形状完全一致，实际为 "
            f"{tuple(donor_image.shape)} 与 {tuple(recipient_image.shape)}"
        )
    if tuple(donor_geometry.shape) != tuple(recipient_geometry.shape):
        raise ValueError(
            "donor.geometry 必须与 recipient.geometry 形状完全一致，实际为 "
            f"{tuple(donor_geometry.shape)} 与 {tuple(recipient_geometry.shape)}"
        )
    if backend == "torch":
        if donor_image.device != recipient_image.device or donor_geometry.device != recipient_geometry.device:
            raise ValueError("recipient 与 donor tensor 必须位于同一 device")

    result = _cloned_inputs(recipient_image, recipient_geometry, recipient_condition)
    result.image[_image_index(recipient_image.ndim, slice(1, 3))] = donor_image[
        _image_index(donor_image.ndim, slice(1, 3))
    ]
    result.geometry[_geometry_index(recipient_geometry.ndim, slice(1, 3))] = donor_geometry[
        _geometry_index(donor_geometry.ndim, slice(1, 3))
    ]
    return result


def _require_torch_float_tensor(value: Any, name: str) -> None:
    if not _is_torch_tensor(value):
        raise TypeError(f"{name} 必须是 torch.Tensor 才能送入 CoReJEPA")
    if not value.is_floating_point():
        raise TypeError(f"{name} 必须是浮点 tensor，实际 dtype 为 {value.dtype}")


def predict_perturbed_context_against_native_target(
    model: Any,
    *,
    native_image: Any,
    native_geometry: Any,
    perturbed_image: Any,
    perturbed_geometry: Any,
    condition: Any,
) -> NativeTargetLatentPrediction:
    """预测扰动 context，并固定与未扰动 native EMA target 对齐。

    该函数刻意不调用 ``model(perturbed_image, ...)``，因为标准 ``forward`` 会从
    扰动序列再次编码 target，导致 temporal/donor swap 的 target 一起移动。这里：

    1. ``native_image + native_geometry`` 只送入 ``encode_targets``，得到原始 T1/T2/T3；
    2. ``perturbed_image + perturbed_geometry`` 送入 online encoder/transition；
    3. response correction 明确使用 ``perturbed_geometry[:, :-1]``；
    4. condition 仍是原 nominal T1/T2/T3 condition。

    支持单患者与批量输入，返回形状分别为 ``[3,D]`` 与 ``[B,3,D]``。为避免
    dropout、BatchNorm 或 target encoder 状态变化，要求 model 及全部子模块事先处于
    eval 模式；函数不会改变其 train/eval 状态，也不会修改任何传入 tensor。
    """

    if torch is None:  # pragma: no cover - 仅 numpy 环境
        raise RuntimeError("latent prediction 需要安装 torch")
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model 必须是 torch.nn.Module")
    training_modules = [name or "<root>" for name, module in model.named_modules() if module.training]
    if training_modules:
        preview = ", ".join(training_modules[:5])
        raise ValueError(f"latent audit 要求 model 及全部子模块处于 eval 模式；仍为 train: {preview}")
    for method_name in ("encode_targets", "encode_visits", "image_transition", "response_transition"):
        if not callable(getattr(model, method_name, None)):
            raise TypeError(f"model 缺少可调用的 {method_name}")

    _validate_inputs(native_image, native_geometry, condition, label="native")
    _validate_inputs(perturbed_image, perturbed_geometry, condition, label="perturbed")
    for name, value in (
        ("native_image", native_image),
        ("native_geometry", native_geometry),
        ("perturbed_image", perturbed_image),
        ("perturbed_geometry", perturbed_geometry),
        ("condition", condition),
    ):
        _require_torch_float_tensor(value, name)

    if tuple(native_image.shape) != tuple(perturbed_image.shape):
        raise ValueError("native_image 与 perturbed_image 必须形状一致")
    if tuple(native_geometry.shape) != tuple(perturbed_geometry.shape):
        raise ValueError("native_geometry 与 perturbed_geometry 必须形状一致")
    tensors = (native_image, native_geometry, perturbed_image, perturbed_geometry, condition)
    if any(value.device != tensors[0].device for value in tensors[1:]):
        raise ValueError("native、perturbed 与 condition tensor 必须位于同一 device")
    if any(value.dtype != tensors[0].dtype for value in tensors[1:]):
        raise TypeError("native、perturbed 与 condition tensor 必须具有相同浮点 dtype")

    was_unbatched = native_image.ndim == 5

    def batched_clone(value: Any) -> Any:
        cloned = value.clone(memory_format=torch.preserve_format)
        return cloned.unsqueeze(0) if was_unbatched else cloned

    native_image_batch = batched_clone(native_image)
    native_geometry_batch = batched_clone(native_geometry)
    perturbed_image_batch = batched_clone(perturbed_image)
    perturbed_geometry_batch = batched_clone(perturbed_geometry)
    condition_batch = batched_clone(condition)

    with torch.no_grad():
        native_visit_target = model.encode_targets(native_image_batch, native_geometry_batch)
        if not _is_torch_tensor(native_visit_target) or native_visit_target.ndim != 3:
            shape = getattr(native_visit_target, "shape", None)
            raise ValueError(f"encode_targets 必须返回 [B,4,D] tensor，实际为 {shape}")
        expected_visit_prefix = (native_image_batch.shape[0], N_VISITS)
        if tuple(native_visit_target.shape[:2]) != expected_visit_prefix:
            raise ValueError(
                f"encode_targets 必须返回前两维 {expected_visit_prefix}，实际为 "
                f"{tuple(native_visit_target.shape)}"
            )
        native_target = native_visit_target[:, 1:].detach().clone()

        perturbed_visit_state = model.encode_visits(
            perturbed_image_batch, perturbed_geometry_batch
        )
        if not _is_torch_tensor(perturbed_visit_state) or tuple(perturbed_visit_state.shape[:2]) != expected_visit_prefix:
            shape = getattr(perturbed_visit_state, "shape", None)
            raise ValueError(f"encode_visits 必须返回 [B,4,D] tensor，实际为 {shape}")
        image_prediction = model.image_transition(
            perturbed_visit_state[:, :-1], condition_batch
        )
        response = model.response_transition(
            perturbed_geometry_batch[:, :-1], condition_batch
        )
        response_correction = getattr(response, "latent_correction", None)
        if not _is_torch_tensor(image_prediction):
            raise TypeError("image_transition 必须返回 torch.Tensor")
        if not _is_torch_tensor(response_correction):
            raise TypeError("response_transition 输出必须包含 tensor 属性 latent_correction")
        if image_prediction.shape != native_target.shape:
            raise ValueError(
                f"image prediction 与 native target 形状不一致："
                f"{tuple(image_prediction.shape)} vs {tuple(native_target.shape)}"
            )
        if response_correction.shape != native_target.shape:
            raise ValueError(
                f"response correction 与 native target 形状不一致："
                f"{tuple(response_correction.shape)} vs {tuple(native_target.shape)}"
            )
        prediction = image_prediction + response_correction

    if was_unbatched:
        prediction = prediction.squeeze(0)
        native_target = native_target.squeeze(0)
        image_prediction = image_prediction.squeeze(0)
        response_correction = response_correction.squeeze(0)
    return NativeTargetLatentPrediction(
        prediction=prediction,
        native_target=native_target,
        image_prediction=image_prediction,
        response_correction=response_correction,
    )


__all__ = [
    "NativeTargetLatentPrediction",
    "PerturbedInputs",
    "predict_perturbed_context_against_native_target",
    "repeated_t0_full_image_derived",
    "repeated_t0_mri_only",
    "replace_followups_with_donor",
    "swap_t1_t2",
]

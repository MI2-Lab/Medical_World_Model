"""Trainable GAP, fixed-local, and Local--Global response-state models.

The only changed segment is the deterministic map from the upstream encoder's
final spatial tensor to its 192-D response state.  Encoder, projector,
transition, target copies, EMA update, and optional FTV head are inherited
unchanged from the hash-locked G3 implementation.
"""

from __future__ import annotations

import copy
import hashlib
import operator
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from .contracts import (
    ARMS,
    ARM_SPECS,
    C1B_INPUT_SHAPE_ZYX,
    C1B_SPACING_XYZ_MM,
    FINAL_FEATURE_CHANNELS,
    LOCAL_WINDOW_MM_XYZ,
    MODEL_KWARGS,
    RESPONSE_DIM,
    ArmSpec,
    arm_spec,
    validate_c1b_geometry,
)
from .pooling import (
    build_fixed_c1b_local_weights,
    derived_final_feature_shape,
    pooling_contract,
    validate_actual_final_feature,
    weighted_average_pool,
)
from .upstream import (
    AUDITED_POOLING_SHA256,
    DGRSObjective,
    DGRSWorldModel,
    G3_SOURCE_SHA256,
)


def _validate_model_hyperparameters(values: Mapping[str, Any]) -> None:
    if set(values) != set(MODEL_KWARGS):
        raise ValueError("model hyperparameter schema differs from the frozen G3 model")
    for name, expected in MODEL_KWARGS.items():
        observed = values[name]
        if isinstance(expected, float):
            matches = (
                isinstance(observed, Real)
                and not isinstance(observed, bool)
                and float(observed) == expected
            )
        else:
            try:
                parsed = operator.index(observed)
            except TypeError:
                matches = False
            else:
                matches = not isinstance(observed, bool) and parsed == expected
        if not matches:
            raise ValueError(f"{name} is frozen to {expected}; got {observed}")


def _projection_parts(module: nn.Module, input_features: int) -> tuple[nn.Linear, nn.LayerNorm]:
    if not isinstance(module, nn.Sequential) or len(module) != 2:
        raise TypeError("response projection must be Sequential(Linear, LayerNorm)")
    linear, normalization = module[0], module[1]
    if not isinstance(linear, nn.Linear) or not isinstance(normalization, nn.LayerNorm):
        raise TypeError("response projection must be Sequential(Linear, LayerNorm)")
    if (linear.in_features, linear.out_features) != (input_features, RESPONSE_DIM):
        raise ValueError("response projection dimensions differ from the frozen contract")
    if tuple(normalization.normalized_shape) != (RESPONSE_DIM,):
        raise ValueError("response LayerNorm dimensions differ from the frozen contract")
    return linear, normalization


class LocalGlobalResponseWorldModel(DGRSWorldModel):
    """The sealed DGRS world model with one preregistered pooling architecture."""

    def __init__(
        self,
        arm: str,
        image_channels: int = 7,
        base_channels: int = 16,
        latent_dim: int = 192,
        predictor_depth: int = 3,
        predictor_heads: int = 4,
        predictor_mlp_dim: int = 512,
        dropout: float = 0.1,
        input_shape_zyx: tuple[int, int, int] | list[int] = C1B_INPUT_SHAPE_ZYX,
        spacing_xyz_mm: tuple[float, float, float] | list[float] = C1B_SPACING_XYZ_MM,
        local_window_mm_xyz: tuple[float, float, float] | list[float] = LOCAL_WINDOW_MM_XYZ,
    ) -> None:
        spec = arm_spec(arm)
        resolved_model = {
            "image_channels": image_channels,
            "base_channels": base_channels,
            "latent_dim": latent_dim,
            "predictor_depth": predictor_depth,
            "predictor_heads": predictor_heads,
            "predictor_mlp_dim": predictor_mlp_dim,
            "dropout": dropout,
        }
        _validate_model_hyperparameters(resolved_model)
        shape, spacing, window = validate_c1b_geometry(
            input_shape_zyx, spacing_xyz_mm, local_window_mm_xyz
        )

        # This constructs the exact upstream encoder/projector/transition/EMA
        # components and the paired baseline Linear(128,192)+LayerNorm.
        super().__init__(model_name=spec.upstream_model_name, **resolved_model)
        self.arm = spec.name
        self.response_architecture = spec.architecture
        self.input_shape_zyx = shape
        self.spacing_xyz_mm = spacing
        self.local_window_mm_xyz = window
        if int(self.encoder.output_channels) != FINAL_FEATURE_CHANNELS:
            raise ValueError("upstream final encoder channel count drifted")

        baseline_linear, baseline_norm = _projection_parts(
            self.response_projection, FINAL_FEATURE_CHANNELS
        )
        if spec.architecture == "LOCAL_GLOBAL":
            combined_linear = nn.Linear(2 * FINAL_FEATURE_CHANNELS, RESPONSE_DIM)
            with torch.no_grad():
                combined_linear.weight[:, :FINAL_FEATURE_CHANNELS].copy_(
                    0.5 * baseline_linear.weight
                )
                combined_linear.weight[:, FINAL_FEATURE_CHANNELS:].copy_(
                    0.5 * baseline_linear.weight
                )
                combined_linear.bias.copy_(baseline_linear.bias)
            self.response_projection = nn.Sequential(
                combined_linear, copy.deepcopy(baseline_norm)
            )
            self.target_response_projection = copy.deepcopy(
                self.response_projection
            ).requires_grad_(False)
        else:
            _projection_parts(self.target_response_projection, FINAL_FEATURE_CHANNELS)

        pooling_names = {
            "GAP": "gap",
            "LOCAL": "fixed_local_64mm_fractional_feature_cell",
            "LOCAL_GLOBAL": "fixed_local_64mm_plus_gap_raw_concat",
        }
        self.pooling = pooling_names[spec.architecture]

        # The audited feature-grid dimensions are derived, never written as a
        # literal.  LOCAL/LG keep one shared deterministic buffer for every
        # patient and visit; GAP retains an exactly upstream-identical state_dict.
        if spec.architecture == "GAP":
            local_weight: torch.Tensor | None = None
        else:
            local_weight = build_fixed_c1b_local_weights(
                input_shape_zyx=shape,
                spacing_xyz_mm=spacing,
                local_window_mm_xyz=window,
                device="cpu",
                dtype=torch.float32,
            )
        self.register_buffer("local_pooling_weight", local_weight, persistent=True)
        validate_model_contract(self, spec.name)

    @property
    def projection_input_dim(self) -> int:
        return (
            2 * FINAL_FEATURE_CHANNELS
            if self.response_architecture == "LOCAL_GLOBAL"
            else FINAL_FEATURE_CHANNELS
        )

    def _validate_sequence_inputs(
        self, image: torch.Tensor, roi_mask: torch.Tensor | None
    ) -> tuple[int, int, torch.Tensor | None]:
        batch, visits, validated_mask = super()._validate_sequence_inputs(
            image, roi_mask
        )
        if tuple(int(value) for value in image.shape[-3:]) != self.input_shape_zyx:
            raise ValueError(
                "pilot image geometry must be frozen C1B-H "
                f"{self.input_shape_zyx} ZYX"
            )
        if validated_mask is not None:
            raise AssertionError("C1B pilot unexpectedly admitted an ROI mask")
        return batch, visits, None

    def _validate_final_spatial(self, spatial: torch.Tensor) -> tuple[int, int, int]:
        return validate_actual_final_feature(
            spatial,
            input_shape_zyx=self.input_shape_zyx,
            local_weights=self.local_pooling_weight,
        )

    def local_weights_for(self, spatial: torch.Tensor) -> torch.Tensor:
        """Return the one audited, shared local weight map on the model device."""

        self._validate_final_spatial(spatial)
        if self.local_pooling_weight is None:
            raise ValueError("GAP architecture has no local pooling weight")
        if self.local_pooling_weight.device != spatial.device:
            raise ValueError("model/local buffer and spatial feature must share a device")
        return self.local_pooling_weight

    def pooled_response_input(self, spatial: torch.Tensor) -> torch.Tensor:
        """Map the actual final encoder tensor to the raw projection input."""

        self._validate_final_spatial(spatial)
        global_state = spatial.mean(dim=(-3, -2, -1))
        if self.response_architecture == "GAP":
            return global_state
        local_state = weighted_average_pool(spatial, self.local_weights_for(spatial))
        if self.response_architecture == "LOCAL":
            return local_state
        if self.response_architecture == "LOCAL_GLOBAL":
            # Contract order is exactly [z_local ; z_global].
            return torch.cat((local_state, global_state), dim=-1)
        raise AssertionError("unregistered response architecture")

    def _encode_sequence(
        self,
        image: torch.Tensor,
        roi_mask: torch.Tensor | None,
        encoder: nn.Module,
        response_projection: nn.Module,
        projector: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        batch, visits, validated_mask = self._validate_sequence_inputs(image, roi_mask)
        if validated_mask is not None:
            raise AssertionError("pilot model may not receive an ROI mask")
        flat = image.reshape(batch * visits, *image.shape[2:])
        spatial = encoder(flat)
        # This is the actual encoder return tensor, for online and target.  The
        # GAP arithmetic below remains operation-for-operation identical to the
        # sealed upstream path; the only addition is this fail-closed check.
        pooled = self.pooled_response_input(spatial)
        if tuple(pooled.shape) != (batch * visits, self.projection_input_dim):
            raise ValueError("pooled response input shape drifted")
        response = response_projection(pooled).reshape(batch, visits, self.latent_dim)
        projected = projector(response).reshape(batch, visits, self.latent_dim)
        return response, projected, None

    def model_config(self) -> dict[str, Any]:
        """Return a JSON-safe config accepted directly by this class."""

        return {
            "arm": self.arm,
            "image_channels": self.image_channels,
            "base_channels": self.base_channels,
            "latent_dim": self.latent_dim,
            "predictor_depth": self.predictor_depth,
            "predictor_heads": self.predictor_heads,
            "predictor_mlp_dim": self.predictor_mlp_dim,
            "dropout": self.dropout,
            "input_shape_zyx": list(self.input_shape_zyx),
            "spacing_xyz_mm": list(self.spacing_xyz_mm),
            "local_window_mm_xyz": list(self.local_window_mm_xyz),
        }

    def architecture_contract(self) -> dict[str, Any]:
        contract = dict(super().architecture_contract())
        pool_contract = pooling_contract()
        contract.update(
            {
                "schema_version": 1,
                "arm": self.arm,
                "architecture": self.response_architecture,
                "grounded": bool(arm_spec(self.arm).grounded),
                "pooling": self.pooling,
                "spatial_source": "encoder.features[3]_full_residual_block_output",
                "input_shape_zyx": list(self.input_shape_zyx),
                "spacing_xyz_mm": list(self.spacing_xyz_mm),
                "final_feature_channels": FINAL_FEATURE_CHANNELS,
                "final_feature_shape_policy": pool_contract[
                    "final_feature_shape_policy"
                ],
                "derived_feature_shape_zyx": pool_contract[
                    "derived_feature_shape_zyx"
                ],
                "local_window_mm_xyz": list(self.local_window_mm_xyz),
                "local_center": "frozen_c1b_crop_physical_center",
                "coordinate_convention": pool_contract["coordinate_convention"],
                "local_weights": "fractional_feature_sampling_cell_overlap",
                "local_weight_shared_across_patients_and_visits": True,
                "local_uses_receptive_field_occupancy": False,
                "projection_input_dim": self.projection_input_dim,
                "response_projection": (
                    f"Linear({self.projection_input_dim},192)+LayerNorm(192)"
                ),
                "local_global_order": (
                    ["local_128", "global_128"]
                    if self.response_architecture == "LOCAL_GLOBAL"
                    else None
                ),
                "roi_mask_backbone_input": False,
                "roi_mask_use": "absent",
                "ftv_is_forward_input": False,
                "audited_pooling_sha256": AUDITED_POOLING_SHA256,
                "upstream_g3_model_sha256": G3_SOURCE_SHA256["model.py"],
            }
        )
        return contract

    def parameter_counts(self) -> dict[str, int]:
        linear, normalization = _projection_parts(
            self.response_projection, self.projection_input_dim
        )

        def count(module: nn.Module | None) -> int:
            return 0 if module is None else int(sum(p.numel() for p in module.parameters()))

        return {
            "pooling_trainable": 0,
            "response_linear": count(linear),
            "response_layer_norm": count(normalization),
            "response_projection": count(self.response_projection),
            "target_response_projection": count(self.target_response_projection),
            "encoder": count(self.encoder),
            "projector": count(self.projector),
            "transition": count(self.transition),
            "ftv_head": count(self.ftv_head),
            "trainable_total": int(
                sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
            ),
            "frozen_total": int(
                sum(parameter.numel() for parameter in self.parameters() if not parameter.requires_grad)
            ),
            "all_model_parameters": int(sum(parameter.numel() for parameter in self.parameters())),
        }


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _named_module_state(model: LocalGlobalResponseWorldModel, names: tuple[str, ...]) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for prefix in names:
        module = getattr(model, prefix)
        for name, value in module.state_dict().items():
            state[f"{prefix}.{name}"] = value
    return state


def shared_initialization_sha256(model: LocalGlobalResponseWorldModel) -> str:
    """Hash only modules whose shapes/initial values must match all six arms."""

    validate_model_contract(model, model.arm)
    return tensor_state_sha256(
        _named_module_state(
            model,
            (
                "encoder",
                "projector",
                "target_encoder",
                "target_projector",
                "transition",
            ),
        )
    )


# Compatibility name for callers using the preceding Stage-B terminology.
common_initialization_sha256 = shared_initialization_sha256


def transition_sha256(model: LocalGlobalResponseWorldModel) -> str:
    validate_model_contract(model, model.arm)
    return tensor_state_sha256(model.transition.state_dict())


def _module_sha256(module: nn.Module | None) -> str | None:
    return None if module is None else tensor_state_sha256(module.state_dict())


def validate_model_contract(
    model: LocalGlobalResponseWorldModel, arm: str | None = None
) -> None:
    if not isinstance(model, LocalGlobalResponseWorldModel):
        raise TypeError("model must be LocalGlobalResponseWorldModel")
    spec = arm_spec(model.arm if arm is None else arm)
    if model.arm != spec.name or model.response_architecture != spec.architecture:
        raise ValueError("model arm/architecture identity drifted")
    if model.model_name != spec.upstream_model_name:
        raise ValueError("model grounding/upstream identity drifted")
    _validate_model_hyperparameters(
        {
            "image_channels": model.image_channels,
            "base_channels": model.base_channels,
            "latent_dim": model.latent_dim,
            "predictor_depth": model.predictor_depth,
            "predictor_heads": model.predictor_heads,
            "predictor_mlp_dim": model.predictor_mlp_dim,
            "dropout": model.dropout,
        }
    )
    validate_c1b_geometry(
        model.input_shape_zyx,
        model.spacing_xyz_mm,
        model.local_window_mm_xyz,
    )
    if model.requires_roi_mask:
        raise ValueError("pilot models must never accept an ROI mask")
    if bool(model.ftv_head is not None) != bool(spec.grounded):
        raise ValueError("FTV head presence differs from the arm grounding contract")
    _projection_parts(model.response_projection, model.projection_input_dim)
    _projection_parts(model.target_response_projection, model.projection_input_dim)
    if any(parameter.requires_grad for parameter in model.target_encoder.parameters()):
        raise ValueError("target encoder must remain frozen")
    if any(
        parameter.requires_grad
        for parameter in model.target_response_projection.parameters()
    ):
        raise ValueError("target response projection must remain frozen")
    if any(parameter.requires_grad for parameter in model.target_projector.parameters()):
        raise ValueError("target projector must remain frozen")
    derived_shape = derived_final_feature_shape(model.input_shape_zyx)
    if spec.architecture == "GAP":
        if model.local_pooling_weight is not None:
            raise ValueError("GAP must not add a local-weight state buffer")
    elif model.local_pooling_weight is None or tuple(model.local_pooling_weight.shape) != (
        1,
        1,
        *derived_shape,
    ):
        raise ValueError("LOCAL/LG fixed local-weight buffer shape drifted")


def build_model(arm: str, effective_seed: int) -> LocalGlobalResponseWorldModel:
    """Build one arm without perturbing the caller's RNG stream."""

    try:
        seed = operator.index(effective_seed)
    except TypeError as exc:
        raise ValueError("effective_seed must be an integer") from exc
    if isinstance(effective_seed, bool):
        raise ValueError("effective_seed must be an integer")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = LocalGlobalResponseWorldModel(arm=arm, **dict(MODEL_KWARGS))
    validate_model_contract(model, arm)
    return model


def build_objective(arm: str) -> nn.Module:
    spec = arm_spec(arm)
    return DGRSObjective(
        model_name=spec.upstream_model_name,
        lambda_ftv=0.25 if spec.grounded else 0.0,
        sigreg_weight=0.09,
        sigreg_projections=256,
        step_weights=(2.0, 1.0, 0.5),
    )


def paired_initialization_report(effective_seed: int) -> dict[str, Any]:
    """Build all six arms and prove every paired initialization invariant."""

    baseline = build_model("GAP0", effective_seed)
    baseline_linear, baseline_norm = _projection_parts(
        baseline.response_projection, FINAL_FEATURE_CHANNELS
    )
    baseline_target_linear, baseline_target_norm = _projection_parts(
        baseline.target_response_projection, FINAL_FEATURE_CHANNELS
    )
    baseline_response_hash = _module_sha256(baseline.response_projection)
    baseline_target_hash = _module_sha256(baseline.target_response_projection)
    baseline_weight = baseline_linear.weight.detach().clone()
    baseline_bias = baseline_linear.bias.detach().clone()
    baseline_norm_state = {
        name: value.detach().clone() for name, value in baseline_norm.state_dict().items()
    }
    baseline_target_weight = baseline_target_linear.weight.detach().clone()
    baseline_target_bias = baseline_target_linear.bias.detach().clone()
    baseline_target_norm_state = {
        name: value.detach().clone()
        for name, value in baseline_target_norm.state_dict().items()
    }
    del baseline

    per_arm: dict[str, Any] = {}
    lg_checks: dict[str, bool] = {}
    for arm in ARMS:
        model = build_model(arm, effective_seed)
        record = {
            "shared_initialization_sha256": shared_initialization_sha256(model),
            "transition_sha256": transition_sha256(model),
            "response_projection_sha256": _module_sha256(model.response_projection),
            "target_response_projection_sha256": _module_sha256(
                model.target_response_projection
            ),
            "ftv_head_sha256": _module_sha256(model.ftv_head),
            "parameter_counts": model.parameter_counts(),
        }
        per_arm[arm] = record
        if arm.startswith("LG"):
            online_linear, online_norm = _projection_parts(
                model.response_projection, 2 * FINAL_FEATURE_CHANNELS
            )
            target_linear, target_norm = _projection_parts(
                model.target_response_projection, 2 * FINAL_FEATURE_CHANNELS
            )
            exact = all(
                (
                    torch.equal(
                        online_linear.weight[:, :FINAL_FEATURE_CHANNELS] * 2.0,
                        baseline_weight,
                    ),
                    torch.equal(
                        online_linear.weight[:, FINAL_FEATURE_CHANNELS:] * 2.0,
                        baseline_weight,
                    ),
                    torch.equal(online_linear.bias, baseline_bias),
                    all(
                        torch.equal(online_norm.state_dict()[name], value)
                        for name, value in baseline_norm_state.items()
                    ),
                    torch.equal(
                        target_linear.weight[:, :FINAL_FEATURE_CHANNELS] * 2.0,
                        baseline_target_weight,
                    ),
                    torch.equal(
                        target_linear.weight[:, FINAL_FEATURE_CHANNELS:] * 2.0,
                        baseline_target_weight,
                    ),
                    torch.equal(target_linear.bias, baseline_target_bias),
                    all(
                        torch.equal(target_norm.state_dict()[name], value)
                        for name, value in baseline_target_norm_state.items()
                    ),
                )
            )
            lg_checks[arm] = bool(exact)
        del model

    shared_hashes = {record["shared_initialization_sha256"] for record in per_arm.values()}
    transition_hashes = {record["transition_sha256"] for record in per_arm.values()}
    single_projection_hashes = {
        per_arm[arm]["response_projection_sha256"]
        for arm in ("GAP0", "GAP3", "LOCAL0", "LOCAL3")
    }
    single_target_hashes = {
        per_arm[arm]["target_response_projection_sha256"]
        for arm in ("GAP0", "GAP3", "LOCAL0", "LOCAL3")
    }
    grounded_head_hashes = {
        per_arm[arm]["ftv_head_sha256"] for arm in ("GAP3", "LOCAL3", "LG3")
    }
    checks = {
        "all_shared_modules_identical": len(shared_hashes) == 1,
        "all_transitions_identical": len(transition_hashes) == 1,
        "gap_local_online_projection_identical": single_projection_hashes
        == {baseline_response_hash},
        "gap_local_target_projection_identical": single_target_hashes
        == {baseline_target_hash},
        "all_grounded_ftv_heads_identical": None not in grounded_head_hashes
        and len(grounded_head_hashes) == 1,
        "all_ungrounded_ftv_heads_absent": all(
            per_arm[arm]["ftv_head_sha256"] is None
            for arm in ("GAP0", "LOCAL0", "LG0")
        ),
        "lg_half_weight_bias_norm_contract": all(lg_checks.values())
        and set(lg_checks) == {"LG0", "LG3"},
    }
    if not all(checks.values()):
        failed = sorted(name for name, value in checks.items() if not value)
        raise AssertionError(f"paired initialization contract failed: {failed}")
    return {
        "schema_version": 1,
        "effective_seed": int(effective_seed),
        "arms": list(ARMS),
        "shared_initialization_sha256": next(iter(shared_hashes)),
        "transition_sha256": next(iter(transition_hashes)),
        "baseline_response_projection_sha256": baseline_response_hash,
        "per_arm": per_arm,
        "checks": checks,
    }


def load_checkpoint_for_evaluation(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[LocalGlobalResponseWorldModel, dict[str, Any]]:
    """Strictly load one selected, test-blind pilot checkpoint."""

    source = Path(path).expanduser().resolve()
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("model_config"), Mapping):
        raise ValueError(f"checkpoint schema is invalid: {source}")
    if payload.get("selected") is not True or payload.get("test_data_used") is not False:
        raise ValueError("evaluation requires a selected, test-blind checkpoint")
    model = LocalGlobalResponseWorldModel(**dict(payload["model_config"]))
    expected_local_weight = (
        None
        if model.local_pooling_weight is None
        else model.local_pooling_weight.detach().clone()
    )
    state = payload.get("state_dict", payload.get("model_state"))
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint lacks state_dict/model_state")
    model.load_state_dict(state, strict=True)
    if expected_local_weight is not None and not torch.equal(
        model.local_pooling_weight, expected_local_weight
    ):
        raise ValueError("checkpoint changed the fixed audited local-pooling weights")
    checkpoint_arm = str(payload.get("arm", model.arm)).upper()
    if checkpoint_arm != model.arm:
        raise ValueError("checkpoint arm and model_config arm disagree")
    validate_model_contract(model, model.arm)
    saved_contract = payload.get("architecture_contract")
    if saved_contract is not None:
        if not isinstance(saved_contract, Mapping):
            raise ValueError("checkpoint architecture_contract must be a mapping")
        if dict(saved_contract) != model.architecture_contract():
            raise ValueError("checkpoint architecture contract disagrees with model_config")
    model.to(device).eval()
    return model, payload


load_selected_checkpoint = load_checkpoint_for_evaluation
load_checkpoint = load_checkpoint_for_evaluation


__all__ = [
    "ARMS",
    "ARM_SPECS",
    "ArmSpec",
    "LocalGlobalResponseWorldModel",
    "arm_spec",
    "build_model",
    "build_objective",
    "common_initialization_sha256",
    "load_checkpoint",
    "load_checkpoint_for_evaluation",
    "load_selected_checkpoint",
    "paired_initialization_report",
    "shared_initialization_sha256",
    "tensor_state_sha256",
    "transition_sha256",
    "validate_model_contract",
]

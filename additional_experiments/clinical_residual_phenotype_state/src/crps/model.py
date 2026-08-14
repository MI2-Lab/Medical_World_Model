"""Factorized response/phenotype state for the pCR-free Goal-F pilot."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from .contracts import (
    C1B_SHAPE_ZYX,
    C1B_SPACING_XYZ_MM,
    FEATURE_CHANNELS,
    LOCAL_WINDOW_MM_XYZ,
    PHENOTYPE_DIM,
    RESPONSE_DIM,
    STATE_DIM,
    arm_spec,
    validate_geometry,
)
from .upstream import (
    ImageOnlyCausalTransition,
    ImageTransition,
    SpatialVisitEncoder3D,
    VisitProjector,
    expected_feature_shape,
    fixed_physical_local_weights,
    weighted_average_pool,
)


class GradientReversal(torch.autograd.Function):
    """Identity forward with a configurable sign-reversed backward."""

    @staticmethod
    def forward(ctx: Any, value: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return value.view_as(value)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.scale * gradient, None


def gradient_reverse(value: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    if not math.isfinite(float(scale)) or float(scale) < 0:
        raise ValueError("gradient reversal scale must be finite and nonnegative")
    return GradientReversal.apply(value, float(scale))


class SingleQueryLocalPool(nn.Module):
    """One learned mask-free query over the fixed 64-mm LOCAL spatial tokens.

    The fractional LOCAL weights are geometry-only and shared by every sample.
    They act as an attention prior: zero-support tokens are excluded and
    boundary-cell overlap contributes additively in log-probability space.
    There is exactly one query token and no lesion/outcome-derived mask.
    """

    def __init__(
        self,
        input_dim: int = FEATURE_CHANNELS,
        output_dim: int = PHENOTYPE_DIM,
        heads: int = 4,
        mlp_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0 or heads <= 0:
            raise ValueError("query dimensions and head count must be positive")
        if output_dim % heads:
            raise ValueError("phenotype output dimension must be divisible by heads")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.heads = int(heads)
        self.head_dim = output_dim // heads
        self.query = nn.Parameter(torch.randn(1, heads, 1, self.head_dim) / math.sqrt(output_dim))
        self.key = nn.Linear(input_dim, output_dim, bias=False)
        self.value = nn.Linear(input_dim, output_dim, bias=False)
        self.output = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )
        self.normalization = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.mlp = nn.Sequential(
            nn.Linear(output_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(mlp_dim, output_dim),
        )

    def forward(
        self,
        spatial: torch.Tensor,
        local_weights: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if spatial.ndim != 5 or int(spatial.size(1)) != self.input_dim:
            raise ValueError("spatial feature must be [N,128,D,H,W]")
        if local_weights.shape != (1, 1, *spatial.shape[-3:]):
            raise ValueError("single-query LOCAL weights must be [1,1,D,H,W]")
        if spatial.device != local_weights.device:
            raise ValueError("spatial feature and LOCAL weights must share a device")
        if not bool(torch.isfinite(spatial).all()) or not bool(torch.isfinite(local_weights).all()):
            raise ValueError("query-pooling inputs must be finite")
        flat_weights = local_weights.reshape(1, 1, 1, -1).to(dtype=spatial.dtype)
        if bool((flat_weights < 0).any()) or not bool((flat_weights > 0).any()):
            raise ValueError("LOCAL weights must be nonnegative with nonempty support")
        tokens = spatial.flatten(2).transpose(1, 2)
        batch, token_count, _ = tokens.shape
        keys = self.key(tokens).view(batch, token_count, self.heads, self.head_dim).permute(0, 2, 1, 3)
        values = self.value(tokens).view(batch, token_count, self.heads, self.head_dim).permute(0, 2, 1, 3)
        logits = torch.matmul(self.query.expand(batch, -1, -1, -1), keys.transpose(-1, -2))
        logits = logits / math.sqrt(self.head_dim)
        log_prior = torch.where(
            flat_weights > 0,
            flat_weights.clamp_min(torch.finfo(spatial.dtype).tiny).log(),
            torch.full_like(flat_weights, float("-inf")),
        )
        attention = torch.softmax(logits + log_prior, dim=-1)
        pooled = torch.matmul(attention, values).squeeze(2).reshape(batch, self.output_dim)
        # The residual is the learned query itself, never a LOCAL-mean feature.
        # This keeps the defining phenotype pathway query-mediated: there is no
        # direct mean-pooling bypass that could turn z_P into a second z_R.
        query_residual = self.query.reshape(1, self.output_dim).expand(batch, -1)
        hidden = self.normalization(self.output(pooled) + query_residual)
        phenotype = self.normalization(hidden + self.dropout(self.mlp(hidden)))
        if not bool(torch.isfinite(phenotype).all()):
            raise FloatingPointError("phenotype query produced non-finite state")
        if return_attention:
            return phenotype, attention.reshape(batch, self.heads, token_count)
        return phenotype


@dataclass
class FactorizedOutput:
    response_state: torch.Tensor
    phenotype_state: torch.Tensor
    full_state: torch.Tensor
    response_online: torch.Tensor
    phenotype_online: torch.Tensor
    target_response_state: torch.Tensor
    target_phenotype_state: torch.Tensor
    target_response_online: torch.Tensor
    target_phenotype_online: torch.Tensor
    predicted_response_next: torch.Tensor
    predicted_phenotype_next: torch.Tensor
    ftv_prediction: torch.Tensor
    adversary_hr_logits: torch.Tensor | None
    adversary_her2_logits: torch.Tensor | None
    augmented_phenotype_state: torch.Tensor | None


class FactorizedPhenotypeWorldModel(nn.Module):
    """96-D LOCAL response plus 96-D single-query phenotype state."""

    def __init__(
        self,
        arm: str,
        condition_dim: int,
        *,
        image_channels: int = 7,
        base_channels: int = 16,
        predictor_depth: int = 3,
        predictor_heads: int = 4,
        predictor_mlp_dim: int = 512,
        dropout: float = 0.1,
        query_heads: int = 4,
        query_mlp_dim: int = 256,
        input_shape_zyx: tuple[int, int, int] = C1B_SHAPE_ZYX,
        spacing_xyz_mm: tuple[float, float, float] = C1B_SPACING_XYZ_MM,
        local_window_mm_xyz: tuple[float, float, float] = LOCAL_WINDOW_MM_XYZ,
    ) -> None:
        super().__init__()
        self.spec = arm_spec(arm)
        self.arm = self.spec.name
        if int(image_channels) != 7:
            raise ValueError("Goal F encoder input must remain DCE7")
        if int(condition_dim) <= 0:
            raise ValueError("phenotype transition condition dimension must be positive")
        validate_geometry(input_shape_zyx, spacing_xyz_mm, local_window_mm_xyz)
        self.condition_dim = int(condition_dim)
        self.image_channels = 7
        self.base_channels = int(base_channels)
        self.predictor_depth = int(predictor_depth)
        self.predictor_heads = int(predictor_heads)
        self.predictor_mlp_dim = int(predictor_mlp_dim)
        self.dropout = float(dropout)
        self.query_heads = int(query_heads)
        self.query_mlp_dim = int(query_mlp_dim)
        self.input_shape_zyx = tuple(int(value) for value in input_shape_zyx)
        self.spacing_xyz_mm = tuple(float(value) for value in spacing_xyz_mm)
        self.local_window_mm_xyz = tuple(float(value) for value in local_window_mm_xyz)

        self.encoder = SpatialVisitEncoder3D(self.base_channels)
        if int(self.encoder.output_channels) != FEATURE_CHANNELS:
            raise ValueError("frozen encoder final channel count drifted")
        feature_shape = expected_feature_shape(self.input_shape_zyx, stage="final")
        weights = fixed_physical_local_weights(
            self.input_shape_zyx,
            feature_shape,
            self.spacing_xyz_mm,
            stage="final",
            device="cpu",
            dtype=torch.float32,
        )
        self.register_buffer("local_pooling_weight", weights, persistent=True)

        self.response_projection = nn.Sequential(
            nn.Linear(FEATURE_CHANNELS, RESPONSE_DIM), nn.LayerNorm(RESPONSE_DIM)
        )
        self.phenotype_pool = SingleQueryLocalPool(
            FEATURE_CHANNELS,
            PHENOTYPE_DIM,
            heads=self.query_heads,
            mlp_dim=self.query_mlp_dim,
            dropout=self.dropout,
        )
        self.response_projector = VisitProjector(RESPONSE_DIM)
        self.phenotype_projector = VisitProjector(PHENOTYPE_DIM)

        self.target_encoder = copy.deepcopy(self.encoder).requires_grad_(False)
        self.target_response_projection = copy.deepcopy(self.response_projection).requires_grad_(False)
        self.target_phenotype_pool = copy.deepcopy(self.phenotype_pool).requires_grad_(False)
        self.target_response_projector = copy.deepcopy(self.response_projector).requires_grad_(False)
        self.target_phenotype_projector = copy.deepcopy(self.phenotype_projector).requires_grad_(False)

        # Canonical confirmed LOCAL response dynamics stay image-only.  The
        # frozen clinical/treatment-conditioned transition is applied only to
        # phenotype-future prediction, which is the new experimental branch.
        self.response_transition = ImageOnlyCausalTransition(
            RESPONSE_DIM,
            self.predictor_depth,
            self.predictor_heads,
            self.predictor_mlp_dim,
            self.dropout,
        )
        self.phenotype_transition = ImageTransition(
            PHENOTYPE_DIM,
            self.condition_dim,
            self.predictor_depth,
            self.predictor_heads,
            self.predictor_mlp_dim,
            self.dropout,
            0.1,
        )
        self.ftv_head = nn.Linear(RESPONSE_DIM, 1)
        self.hr_adversary = nn.Linear(PHENOTYPE_DIM, 2) if self.spec.uses_adversary else None
        self.her2_adversary = nn.Linear(PHENOTYPE_DIM, 2) if self.spec.uses_adversary else None

    def _validate_inputs(self, image: torch.Tensor, condition: torch.Tensor | None = None) -> tuple[int, int]:
        if image.ndim != 6 or int(image.size(2)) != 7:
            raise ValueError("image must be [B,V,7,Z,Y,X]")
        if tuple(int(value) for value in image.shape[-3:]) != self.input_shape_zyx:
            raise ValueError("Goal F image geometry drifted from C1B-H")
        batch, visits = int(image.size(0)), int(image.size(1))
        if condition is not None and tuple(condition.shape) != (batch, max(visits - 1, 0), self.condition_dim):
            raise ValueError("condition must be [B,V-1,C]")
        return batch, visits

    def _validate_spatial(self, spatial: torch.Tensor) -> None:
        expected = expected_feature_shape(self.input_shape_zyx, stage="final")
        if spatial.ndim != 5 or tuple(spatial.shape[1:]) != (FEATURE_CHANNELS, *expected):
            raise ValueError("actual encoder final spatial map drifted")
        if tuple(self.local_pooling_weight.shape) != (1, 1, *expected):
            raise ValueError("fixed LOCAL pooling weight shape drifted")
        if spatial.device != self.local_pooling_weight.device:
            raise ValueError("encoder and LOCAL buffer must share a device")

    def _encode_sequence(
        self,
        image: torch.Tensor,
        encoder: nn.Module,
        response_projection: nn.Module,
        phenotype_pool: nn.Module,
        response_projector: nn.Module,
        phenotype_projector: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, visits = self._validate_inputs(image)
        spatial = encoder(image.reshape(batch * visits, *image.shape[2:]))
        self._validate_spatial(spatial)
        response_raw = weighted_average_pool(spatial, self.local_pooling_weight)
        response = response_projection(response_raw).reshape(batch, visits, RESPONSE_DIM)
        phenotype = phenotype_pool(spatial, self.local_pooling_weight).reshape(
            batch, visits, PHENOTYPE_DIM
        )
        response_online = response_projector(response.reshape(-1, RESPONSE_DIM)).reshape(
            batch, visits, RESPONSE_DIM
        )
        phenotype_online = phenotype_projector(phenotype.reshape(-1, PHENOTYPE_DIM)).reshape(
            batch, visits, PHENOTYPE_DIM
        )
        return response, phenotype, response_online, phenotype_online

    def encode_online(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._encode_sequence(
            image,
            self.encoder,
            self.response_projection,
            self.phenotype_pool,
            self.response_projector,
            self.phenotype_projector,
        )

    @torch.no_grad()
    def encode_target(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        for module in (
            self.target_encoder,
            self.target_response_projection,
            self.target_phenotype_pool,
            self.target_response_projector,
            self.target_phenotype_projector,
        ):
            module.eval()
        return self._encode_sequence(
            image,
            self.target_encoder,
            self.target_response_projection,
            self.target_phenotype_pool,
            self.target_response_projector,
            self.target_phenotype_projector,
        )

    def encode_states(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        response, phenotype, _, _ = self.encode_online(image)
        return response, phenotype

    def forward(
        self,
        image: torch.Tensor,
        condition: torch.Tensor,
        augmented_image: torch.Tensor | None = None,
    ) -> FactorizedOutput:
        batch, visits = self._validate_inputs(image, condition)
        if visits != 4:
            raise ValueError("Goal F representation training requires T0-T3")
        response, phenotype, response_online, phenotype_online = self.encode_online(image)
        target_response, target_phenotype, target_response_online, target_phenotype_online = self.encode_target(image)
        predicted_response = self.response_transition(response_online[:, :-1])
        predicted_phenotype = self.phenotype_transition(phenotype_online[:, :-1], condition)
        flat_phenotype = phenotype.reshape(batch * visits, PHENOTYPE_DIM)
        if self.spec.uses_adversary:
            reversed_state = gradient_reverse(flat_phenotype)
            assert self.hr_adversary is not None and self.her2_adversary is not None
            hr_logits = self.hr_adversary(reversed_state).reshape(batch, visits, 2)
            her2_logits = self.her2_adversary(reversed_state).reshape(batch, visits, 2)
        else:
            hr_logits = her2_logits = None
        augmented_phenotype: torch.Tensor | None = None
        if augmented_image is not None:
            self._validate_inputs(augmented_image)
            _, augmented_phenotype, _, _ = self.encode_online(augmented_image)
        full = torch.cat((response, phenotype), dim=-1)
        if tuple(full.shape) != (batch, visits, STATE_DIM):
            raise AssertionError("factorized full state must remain 192-D")
        return FactorizedOutput(
            response_state=response,
            phenotype_state=phenotype,
            full_state=full,
            response_online=response_online,
            phenotype_online=phenotype_online,
            target_response_state=target_response.detach(),
            target_phenotype_state=target_phenotype.detach(),
            target_response_online=target_response_online.detach(),
            target_phenotype_online=target_phenotype_online.detach(),
            predicted_response_next=predicted_response,
            predicted_phenotype_next=predicted_phenotype,
            ftv_prediction=self.ftv_head(response).squeeze(-1),
            adversary_hr_logits=hr_logits,
            adversary_her2_logits=her2_logits,
            augmented_phenotype_state=augmented_phenotype,
        )

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        if not 0.0 < float(momentum) < 1.0:
            raise ValueError("EMA momentum must be in (0,1)")
        pairs = (
            (self.encoder, self.target_encoder),
            (self.response_projection, self.target_response_projection),
            (self.phenotype_pool, self.target_phenotype_pool),
            (self.response_projector, self.target_response_projector),
            (self.phenotype_projector, self.target_phenotype_projector),
        )
        for online_module, target_module in pairs:
            for online, target in zip(online_module.parameters(), target_module.parameters(), strict=True):
                target.data.mul_(momentum).add_(online.data, alpha=1.0 - momentum)
            for online, target in zip(online_module.buffers(), target_module.buffers(), strict=True):
                target.copy_(online)

    def model_config(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "condition_dim": self.condition_dim,
            "image_channels": self.image_channels,
            "base_channels": self.base_channels,
            "predictor_depth": self.predictor_depth,
            "predictor_heads": self.predictor_heads,
            "predictor_mlp_dim": self.predictor_mlp_dim,
            "dropout": self.dropout,
            "query_heads": self.query_heads,
            "query_mlp_dim": self.query_mlp_dim,
            "input_shape_zyx": self.input_shape_zyx,
            "spacing_xyz_mm": self.spacing_xyz_mm,
            "local_window_mm_xyz": self.local_window_mm_xyz,
        }

    def architecture_contract(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "input": "C1B-H_DCE7",
            "input_shape_zyx": list(self.input_shape_zyx),
            "spacing_xyz_mm": list(self.spacing_xyz_mm),
            "local_window_mm_xyz": list(self.local_window_mm_xyz),
            "state_dim": STATE_DIM,
            "response_dim": RESPONSE_DIM,
            "phenotype_dim": PHENOTYPE_DIM,
            "response_pooling": "fixed_64mm_LOCAL_fractional_mean",
            "phenotype_pooling": "one_learned_mask_free_query_fixed_64mm_LOCAL_tokens",
            "ftv_head_input": "response_state_only",
            "response_transition": "confirmed_image_only_causal_transformer",
            "phenotype_transition": "clinical_treatment_conditioned_causal_transformer",
            "phenotype_query_conditioned": False,
            "clinical_adversary": "two_linear_binary_heads" if self.spec.uses_adversary else None,
            "adversarially_removed": ["HR", "HER2"] if self.spec.uses_adversary else [],
            "treatment_adversarially_removed": False,
            "forbidden_inputs_absent": ["pCR", "BPE", "lesion_mask", "outcome_region"],
        }


def build_model(arm: str, condition_dim: int, effective_seed: int) -> FactorizedPhenotypeWorldModel:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(effective_seed))
        return FactorizedPhenotypeWorldModel(arm, condition_dim)


def load_checkpoint(
    path: str,
    device: str | torch.device = "cpu",
) -> tuple[FactorizedPhenotypeWorldModel, Mapping[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model_config"), Mapping):
        raise ValueError("Goal F checkpoint schema is invalid")
    if payload.get("selected") is not True or payload.get("pcr_used") is not False:
        raise PermissionError("evaluation requires a selected pCR-free checkpoint")
    model = FactorizedPhenotypeWorldModel(**dict(payload["model_config"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    if payload.get("architecture_contract") != model.architecture_contract():
        raise ValueError("checkpoint architecture contract drifted")
    model.to(device).eval()
    return model, payload


__all__ = [
    "FactorizedOutput",
    "FactorizedPhenotypeWorldModel",
    "GradientReversal",
    "SingleQueryLocalPool",
    "build_model",
    "gradient_reverse",
    "load_checkpoint",
]

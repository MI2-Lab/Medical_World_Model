"""A1 treatment-conditioned masked future-token JEPA model."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import operator
from typing import Any, Sequence

import torch
from torch import nn

from .contracts import (
    ARM_EMBEDDING_DIM,
    C1B_INPUT_SHAPE_ZYX,
    C1B_SPACING_XYZ_MM,
    CLINICAL_FEATURES,
    FINAL_CHANNELS,
    FIXED_ARM_VOCAB,
    FORMAL_LOCAL_TOKEN_COUNT,
    FORMAL_MASKED_TOKEN_COUNT,
    IMAGE_CHANNELS,
    PREDICTOR_BLOCKS,
    PREDICTOR_DROPOUT,
    PREDICTOR_FF_DIM,
    PREDICTOR_HEADS,
    RESPONSE_DIM,
    SpatialVisitEncoder3D,
    TEMPORAL_FEATURES,
    TOKEN_DIM,
    TRANSITIONS,
    TransitionCondition,
    VISITS,
    source_contract,
)
from .geometry import (
    LocalTokenGeometry,
    build_local_token_geometry,
    gather_local_tokens,
    sinusoidal_physical_position_encoding,
    weighted_local_mean,
)


@dataclass
class PatchTokenOutput:
    """Complete train-time A1 output; no outcome or FTV target is an input."""

    online_tokens: torch.Tensor
    target_tokens: torch.Tensor
    target_masked: torch.Tensor
    predictions: torch.Tensor
    mask_indices: torch.Tensor
    sigreg_state: torch.Tensor
    canonical_response: torch.Tensor
    ftv_prediction: torch.Tensor
    local_coordinates_xyz_mm: torch.Tensor
    local_weights: torch.Tensor

    @property
    def tokens(self) -> torch.Tensor:
        """Compatibility alias for the position-free online MRI tokens."""

        return self.online_tokens

    @property
    def canonical_response_state(self) -> torch.Tensor:
        return self.canonical_response

    @property
    def target_next(self) -> torch.Tensor:
        return self.target_tokens[:, 1:]

    @property
    def predicted_masked(self) -> torch.Tensor:
        return self.predictions


def _exact_nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a nonnegative integer")
    try:
        parsed = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a nonnegative integer") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return parsed


def deterministic_mask_indices(
    token_count: int,
    masked_token_count: int,
    patient_ids: Sequence[str],
    *,
    effective_seed: int,
    epoch: int,
    logical_batch_index: int,
    transitions: int = TRANSITIONS,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Sample stable outcome-blind masks keyed by the locked formal fields.

    Patient identifiers are first SHA-256 hashed and never returned.  Sampling
    is CPU-generator based so an identical key gives identical indices on CPU
    and CUDA and is independent of the global RNG/dropout stream.
    """

    token_count = _exact_nonnegative_integer(token_count, "token_count")
    masked_token_count = _exact_nonnegative_integer(
        masked_token_count, "masked_token_count"
    )
    transitions = _exact_nonnegative_integer(transitions, "transitions")
    effective_seed = _exact_nonnegative_integer(effective_seed, "effective_seed")
    epoch = _exact_nonnegative_integer(epoch, "epoch")
    logical_batch_index = _exact_nonnegative_integer(
        logical_batch_index, "logical_batch_index"
    )
    if token_count <= 0 or transitions <= 0:
        raise ValueError("token_count and transitions must be positive")
    if not 0 < masked_token_count < token_count:
        raise ValueError("masked_token_count must lie strictly between zero and K")
    if isinstance(patient_ids, (str, bytes)) or len(patient_ids) <= 0:
        raise ValueError("patient_ids must be a nonempty sequence")
    normalized_ids: list[str] = []
    for patient_id in patient_ids:
        value = str(patient_id)
        if not value:
            raise ValueError("patient_ids may not contain empty values")
        normalized_ids.append(value)
    output = torch.empty(
        (len(normalized_ids), transitions, masked_token_count), dtype=torch.long
    )
    for patient_index, patient_id in enumerate(normalized_ids):
        patient_digest = hashlib.sha256(patient_id.encode("utf-8")).hexdigest()
        for transition in range(transitions):
            key = (
                f"patch-token-mask-v1|{effective_seed}|{epoch}|"
                f"{logical_batch_index}|{patient_digest}|{transition}"
            )
            digest = hashlib.sha256(key.encode("ascii")).digest()
            generator_seed = int.from_bytes(digest[:8], byteorder="little") % (
                2**63 - 1
            )
            generator = torch.Generator(device="cpu").manual_seed(generator_seed)
            output[patient_index, transition] = torch.randperm(
                token_count, generator=generator
            )[:masked_token_count]
    return output.to(device=torch.device("cpu" if device is None else device))


def source_to_query_block_mask(
    source_length: int,
    query_length: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Attention mask for a one-way source-to-query predictor.

    Source rows (the condition and current MRI cells) cannot read query states.
    Query rows may read the complete source and other position-only queries.
    Thus information flows source -> query, never query -> source, and no future
    target value is present anywhere in the predictor sequence.
    """

    source_length = _exact_nonnegative_integer(source_length, "source_length")
    query_length = _exact_nonnegative_integer(query_length, "query_length")
    if source_length <= 0 or query_length <= 0:
        raise ValueError("source_length and query_length must be positive")
    if not dtype.is_floating_point:
        raise TypeError("attention mask dtype must be floating")
    total = source_length + query_length
    mask = torch.zeros((total, total), device=device, dtype=dtype)
    mask[:source_length, source_length:] = float("-inf")
    return mask


class ConditionTokenEncoder(nn.Module):
    """Map assigned arm, clinical values, nominal bits, and delta-t to one token."""

    input_dim = ARM_EMBEDDING_DIM + len(CLINICAL_FEATURES) + len(TEMPORAL_FEATURES) + 1

    def __init__(self, token_dim: int = TOKEN_DIM) -> None:
        super().__init__()
        self.arm_embedding = nn.Embedding(len(FIXED_ARM_VOCAB), ARM_EMBEDDING_DIM)
        self.projection = nn.Sequential(
            nn.Linear(self.input_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )

    def forward(self, condition: TransitionCondition, batch_size: int) -> torch.Tensor:
        condition.validate(batch_size, self.arm_embedding.weight.device)
        arm = self.arm_embedding(condition.arm_index.long())[:, None, :].expand(
            -1, TRANSITIONS, -1
        )
        dtype = arm.dtype
        clinical = condition.clinical.to(dtype=dtype)[:, None, :].expand(
            -1, TRANSITIONS, -1
        )
        values = torch.cat(
            (
                arm,
                clinical,
                condition.temporal_bits.to(dtype=dtype),
                condition.delta_t.to(dtype=dtype).unsqueeze(-1),
            ),
            dim=-1,
        )
        if values.shape[-1] != self.input_dim:
            raise AssertionError("condition-token feature dimension drifted")
        return self.projection(values)


class MaskedFutureTokenPredictor(nn.Module):
    """Four-block condition-token Transformer for masked future positions."""

    def __init__(
        self,
        token_dim: int = TOKEN_DIM,
        depth: int = PREDICTOR_BLOCKS,
        heads: int = PREDICTOR_HEADS,
        feedforward_dim: int = PREDICTOR_FF_DIM,
        dropout: float = PREDICTOR_DROPOUT,
    ) -> None:
        super().__init__()
        if token_dim != TOKEN_DIM:
            raise ValueError(f"A1 token_dim is locked to {TOKEN_DIM}")
        if depth != PREDICTOR_BLOCKS or heads != PREDICTOR_HEADS:
            raise ValueError("A1 predictor is locked to four blocks and eight heads")
        if feedforward_dim != PREDICTOR_FF_DIM or float(dropout) != PREDICTOR_DROPOUT:
            raise ValueError("A1 predictor FF width/dropout contract drifted")
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.mask_query = nn.Parameter(torch.randn(1, 1, token_dim) / token_dim**0.5)
        self.output = nn.Sequential(
            nn.LayerNorm(token_dim), nn.Linear(token_dim, token_dim)
        )

    def forward(
        self,
        online_tokens: torch.Tensor,
        condition_tokens: torch.Tensor,
        mask_indices: torch.Tensor,
        position_encoding: torch.Tensor,
    ) -> torch.Tensor:
        if online_tokens.ndim != 4 or online_tokens.shape[1] != TRANSITIONS:
            raise ValueError("online_tokens must have shape [B,3,K,128]")
        batch, transitions, token_count, width = online_tokens.shape
        if width != TOKEN_DIM:
            raise ValueError("online token width differs from locked A1 width")
        if tuple(condition_tokens.shape) != (batch, transitions, width):
            raise ValueError("condition_tokens must have shape [B,3,128]")
        if mask_indices.ndim != 3 or tuple(mask_indices.shape[:2]) != (
            batch,
            transitions,
        ):
            raise ValueError("mask_indices must have shape [B,3,M]")
        if mask_indices.dtype != torch.long:
            raise TypeError("mask_indices must use torch.long")
        if bool((mask_indices < 0).any()) or bool((mask_indices >= token_count).any()):
            raise ValueError("mask_indices escaped LOCAL token support")
        # Every patient/transition is sampled without replacement.
        if bool((torch.sort(mask_indices, dim=-1).values.diff(dim=-1) == 0).any()):
            raise ValueError("mask_indices must be unique within each transition")
        if tuple(position_encoding.shape) != (token_count, width):
            raise ValueError("physical position encoding must have shape [K,128]")
        if position_encoding.device != online_tokens.device:
            raise ValueError("position encoding and online tokens must share a device")

        masked_count = int(mask_indices.shape[-1])
        source = online_tokens + position_encoding[None, None, :, :].to(
            dtype=online_tokens.dtype
        )
        flat_indices = mask_indices.reshape(batch * transitions, masked_count)
        query_positions = position_encoding.index_select(
            0, flat_indices.reshape(-1)
        ).reshape(batch * transitions, masked_count, width)
        queries = self.mask_query.to(dtype=online_tokens.dtype) + query_positions.to(
            dtype=online_tokens.dtype
        )
        flat_condition = condition_tokens.reshape(batch * transitions, 1, width)
        flat_source = source.reshape(batch * transitions, token_count, width)
        hidden = torch.cat((flat_condition, flat_source, queries), dim=1)
        source_length = 1 + token_count
        attention_mask = source_to_query_block_mask(
            source_length,
            masked_count,
            device=hidden.device,
            dtype=hidden.dtype,
        )
        hidden = self.transformer(hidden, mask=attention_mask)
        predicted = self.output(hidden[:, source_length:])
        return predicted.reshape(batch, transitions, masked_count, width)


class PatchTokenWorldModel(nn.Module):
    """Locked A1 model with condition-free MRI encoders and conditioned predictor."""

    def __init__(
        self,
        *,
        input_shape_zyx: Sequence[int] = C1B_INPUT_SHAPE_ZYX,
        spacing_xyz_mm: Sequence[float] = C1B_SPACING_XYZ_MM,
        base_channels: int = 16,
        require_formal_geometry: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(base_channels, bool) or int(base_channels) != 16:
            raise ValueError("the exact G3 encoder is locked to base_channels=16")
        formal_geometry = (
            tuple(input_shape_zyx) == C1B_INPUT_SHAPE_ZYX
            and tuple(float(value) for value in spacing_xyz_mm) == C1B_SPACING_XYZ_MM
        )
        if require_formal_geometry and not formal_geometry:
            raise ValueError(
                "A1 formal runs require C1B-H shape/spacing; set "
                "require_formal_geometry=False only for focused synthetic tests"
            )
        geometry = build_local_token_geometry(
            input_shape_zyx,
            spacing_xyz_mm,
            require_formal_count=formal_geometry,
            dtype=torch.float32,
        )
        if geometry.token_count % 2:
            raise ValueError("50% token masking requires an even LOCAL token count")
        self.input_shape_zyx = geometry.input_shape_zyx
        self.spacing_xyz_mm = geometry.spacing_xyz_mm
        self.feature_shape_zyx = geometry.feature_shape_zyx
        self.require_formal_geometry = bool(require_formal_geometry)
        self.token_count = geometry.token_count
        self.masked_token_count = geometry.token_count // 2
        if formal_geometry and (
            self.token_count != FORMAL_LOCAL_TOKEN_COUNT
            or self.masked_token_count != FORMAL_MASKED_TOKEN_COUNT
        ):
            raise AssertionError("formal 500-token/250-query contract drifted")

        # Preserve the upstream G3 initialization order for the shared encoder
        # and canonical response projection.
        self.encoder = SpatialVisitEncoder3D(base_channels=16)
        if int(self.encoder.output_channels) != FINAL_CHANNELS:
            raise ValueError("hash-locked G3 encoder no longer returns 128 channels")
        self.response_projection = nn.Sequential(
            nn.Linear(FINAL_CHANNELS, RESPONSE_DIM), nn.LayerNorm(RESPONSE_DIM)
        )
        self.token_projection = nn.Sequential(
            nn.Linear(FINAL_CHANNELS, TOKEN_DIM), nn.LayerNorm(TOKEN_DIM)
        )
        self.target_encoder = copy.deepcopy(self.encoder).requires_grad_(False)
        self.target_token_projection = copy.deepcopy(
            self.token_projection
        ).requires_grad_(False)
        self.condition_encoder = ConditionTokenEncoder(TOKEN_DIM)
        self.predictor = MaskedFutureTokenPredictor()
        self.ftv_head = nn.Linear(RESPONSE_DIM, 1)

        self.register_buffer(
            "local_dense_weights", geometry.dense_weights, persistent=True
        )
        self.register_buffer("local_indices", geometry.flat_indices, persistent=True)
        self.register_buffer("local_weights", geometry.weights, persistent=True)
        self.register_buffer(
            "local_coordinates_xyz_mm", geometry.coordinates_xyz_mm, persistent=True
        )
        self.register_buffer(
            "physical_position_encoding",
            sinusoidal_physical_position_encoding(
                geometry.coordinates_xyz_mm, TOKEN_DIM
            ),
            persistent=True,
        )
        self.target_encoder.eval()
        self.target_token_projection.eval()

    def train(self, mode: bool = True) -> "PatchTokenWorldModel":
        super().train(mode)
        # EMA modules never train directly, even when the online model does.
        self.target_encoder.eval()
        self.target_token_projection.eval()
        return self

    def _geometry(self) -> LocalTokenGeometry:
        return LocalTokenGeometry(
            input_shape_zyx=self.input_shape_zyx,
            feature_shape_zyx=self.feature_shape_zyx,
            spacing_xyz_mm=self.spacing_xyz_mm,
            dense_weights=self.local_dense_weights,
            flat_indices=self.local_indices,
            weights=self.local_weights,
            coordinates_xyz_mm=self.local_coordinates_xyz_mm,
        )

    def _validate_image(
        self, image: torch.Tensor, *, require_four_visits: bool
    ) -> tuple[int, int]:
        if not isinstance(image, torch.Tensor) or image.ndim != 6:
            raise ValueError("image must have shape [B,V,7,Z,Y,X]")
        if int(image.shape[0]) <= 0:
            raise ValueError("image batch must be nonempty")
        visits = int(image.shape[1])
        if require_four_visits and visits != VISITS:
            raise ValueError("training forward requires exactly T0-T3 four visits")
        if not require_four_visits and not 1 <= visits <= VISITS:
            raise ValueError("token encoding accepts one through four visits")
        if int(image.shape[2]) != IMAGE_CHANNELS:
            raise ValueError("A1 encoder accepts exactly seven DCE channels")
        if tuple(int(value) for value in image.shape[-3:]) != self.input_shape_zyx:
            raise ValueError(
                f"image spatial shape must be {self.input_shape_zyx}, got {tuple(image.shape[-3:])}"
            )
        if not image.dtype.is_floating_point:
            raise TypeError("MRI image must have a floating dtype")
        return int(image.shape[0]), visits

    def _encode_with(
        self,
        image: torch.Tensor,
        encoder: nn.Module,
        token_projection: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, visits = self._validate_image(image, require_four_visits=False)
        flat_image = image.reshape(batch * visits, *image.shape[2:])
        spatial = encoder(flat_image)
        raw_local = gather_local_tokens(spatial, self._geometry())
        tokens = token_projection(raw_local)
        return raw_local.reshape(
            batch, visits, self.token_count, FINAL_CHANNELS
        ), tokens.reshape(batch, visits, self.token_count, TOKEN_DIM)

    def encode_tokens(self, image: torch.Tensor) -> torch.Tensor:
        """Return condition-free, position-free online MRI tokens ``[B,V,K,128]``."""

        _, tokens = self._encode_with(image, self.encoder, self.token_projection)
        return tokens

    @torch.no_grad()
    def encode_target_tokens(self, image: torch.Tensor) -> torch.Tensor:
        """Return stop-gradient EMA MRI target values without position/condition."""

        self.target_encoder.eval()
        self.target_token_projection.eval()
        _, tokens = self._encode_with(
            image, self.target_encoder, self.target_token_projection
        )
        return tokens.detach()

    def canonical_response_from_raw_tokens(
        self, raw_local: torch.Tensor
    ) -> torch.Tensor:
        if raw_local.ndim != 4 or raw_local.shape[-2:] != (
            self.token_count,
            FINAL_CHANNELS,
        ):
            raise ValueError("raw_local must have shape [B,V,K,128]")
        batch, visits = raw_local.shape[:2]
        pooled = weighted_local_mean(
            raw_local.reshape(batch * visits, self.token_count, FINAL_CHANNELS),
            self.local_weights,
        )
        return self.response_projection(pooled).reshape(batch, visits, RESPONSE_DIM)

    def encode_canonical_response(self, image: torch.Tensor) -> torch.Tensor:
        """Exact raw-spatial fractional LOCAL mean -> canonical 128->192 state."""

        raw_local, _ = self._encode_with(image, self.encoder, self.token_projection)
        return self.canonical_response_from_raw_tokens(raw_local)

    def sigreg_state_from_tokens(self, online_tokens: torch.Tensor) -> torch.Tensor:
        """Fractional weighted mean of projected tokens, ``[B,V,128]``."""

        if online_tokens.ndim != 4 or online_tokens.shape[-2:] != (
            self.token_count,
            TOKEN_DIM,
        ):
            raise ValueError("online_tokens must have shape [B,V,K,128]")
        batch, visits = online_tokens.shape[:2]
        summary = weighted_local_mean(
            online_tokens.reshape(batch * visits, self.token_count, TOKEN_DIM),
            self.local_weights,
        )
        return summary.reshape(batch, visits, TOKEN_DIM)

    def encode_sigreg_state(self, image: torch.Tensor) -> torch.Tensor:
        """Condition-free/non-dropout projected LOCAL summary for exact SIGReg."""

        return self.sigreg_state_from_tokens(self.encode_tokens(image))

    def predict_masked(
        self,
        online_tokens: torch.Tensor,
        condition: TransitionCondition,
        mask_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Predict selected future positions from current MRI plus condition token."""

        if online_tokens.ndim != 4 or online_tokens.shape[1:] != (
            VISITS,
            self.token_count,
            TOKEN_DIM,
        ):
            raise ValueError("online_tokens must have shape [B,4,K,128]")
        batch = int(online_tokens.shape[0])
        condition_tokens = self.condition_encoder(condition, batch)
        return self.predictor(
            online_tokens[:, :-1],
            condition_tokens,
            mask_indices,
            self.physical_position_encoding,
        )

    def forward(
        self,
        image: torch.Tensor,
        condition: TransitionCondition,
        *,
        patient_ids: Sequence[str],
        mask_seed: int,
        epoch: int = 0,
        logical_batch_index: int = 0,
    ) -> PatchTokenOutput:
        batch, _ = self._validate_image(image, require_four_visits=True)
        if isinstance(patient_ids, (str, bytes)) or len(patient_ids) != batch:
            raise ValueError("patient_ids must contain exactly one value per patient")
        condition.validate(batch, image.device)
        raw_local, online_tokens = self._encode_with(
            image, self.encoder, self.token_projection
        )
        target_tokens = self.encode_target_tokens(image)
        mask_indices = deterministic_mask_indices(
            self.token_count,
            self.masked_token_count,
            patient_ids,
            effective_seed=mask_seed,
            epoch=epoch,
            logical_batch_index=logical_batch_index,
            device=image.device,
        )
        gather_index = mask_indices.unsqueeze(-1).expand(-1, -1, -1, TOKEN_DIM)
        target_masked = torch.gather(target_tokens[:, 1:], 2, gather_index).detach()
        predictions = self.predict_masked(online_tokens, condition, mask_indices)
        sigreg_state = self.sigreg_state_from_tokens(online_tokens)
        canonical_response = self.canonical_response_from_raw_tokens(raw_local)
        ftv_prediction = self.ftv_head(canonical_response).squeeze(-1)
        return PatchTokenOutput(
            online_tokens=online_tokens,
            target_tokens=target_tokens,
            target_masked=target_masked,
            predictions=predictions,
            mask_indices=mask_indices,
            sigreg_state=sigreg_state,
            canonical_response=canonical_response,
            ftv_prediction=ftv_prediction,
            local_coordinates_xyz_mm=self.local_coordinates_xyz_mm,
            local_weights=self.local_weights,
        )

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        """EMA-update the target encoder and target 128->128 token projector."""

        if not 0.0 < float(momentum) < 1.0:
            raise ValueError("EMA momentum must lie strictly in (0,1)")
        for online_module, target_module in (
            (self.encoder, self.target_encoder),
            (self.token_projection, self.target_token_projection),
        ):
            for online, target in zip(
                online_module.parameters(), target_module.parameters(), strict=True
            ):
                target.mul_(float(momentum)).add_(online, alpha=1.0 - float(momentum))
            for online, target in zip(
                online_module.buffers(), target_module.buffers(), strict=True
            ):
                target.copy_(online)

    def architecture_contract(self) -> dict[str, Any]:
        upstream = source_contract()
        return {
            "schema_version": 1,
            "arm": "A1_PATCH3",
            "input": f"[B,4,{IMAGE_CHANNELS},Z,Y,X] DCE7",
            "input_shape_zyx": list(self.input_shape_zyx),
            "spacing_xyz_mm": list(self.spacing_xyz_mm),
            "feature_shape_policy": "audited_geometry_derived_then_runtime_validated",
            "feature_shape_zyx": list(self.feature_shape_zyx),
            "encoder": "exact_hash_locked_G3_SpatialVisitEncoder3D",
            "g3_model_sha256": upstream["g3_model_sha256"],
            "audited_pooling_sha256": upstream["audited_pooling_sha256"],
            "local_support": "strict_positive_exact_64mm_fractional_cell_overlap",
            "local_token_count": self.token_count,
            "masked_token_count": self.masked_token_count,
            "token_projection": "Linear(128,128)+LayerNorm(128)",
            "sigreg_state": "fractional_weighted_mean_of_projected_position_free_tokens_128",
            "target": "EMA_encoder_plus_EMA_token_projection_stop_gradient",
            "target_values_position_free": True,
            "online_exported_tokens_position_free": True,
            "physical_position": "deterministic_sinusoidal_crop_centered_XYZ_mm_source_and_query_only",
            "condition_method": "one_condition_token_not_FiLM",
            "condition_target_path": False,
            "condition_online_mri_token_path": False,
            "fixed_arm_vocabulary_size": len(FIXED_ARM_VOCAB),
            "clinical_order": list(CLINICAL_FEATURES),
            "temporal_bits": list(TEMPORAL_FEATURES),
            "delta_t": "nominal_adjacent_scalar_exactly_1",
            "predictor": "TransformerEncoder_4_blocks_d128_8_heads_ff512_dropout0.1",
            "attention": "source_to_query_block_mask_context_cannot_read_queries",
            "canonical_response": "raw128_exact_fractional_LOCAL_mean_then_Linear128x192_LayerNorm",
            "ftv_head": "Linear(192,1)",
            "ftv_is_forward_input": False,
            "lesion_mask_input": False,
            "pixel_reconstruction": False,
        }

    def model_config(self) -> dict[str, Any]:
        return {
            "input_shape_zyx": list(self.input_shape_zyx),
            "spacing_xyz_mm": list(self.spacing_xyz_mm),
            "base_channels": 16,
            "require_formal_geometry": self.require_formal_geometry,
        }


# Explicit aliases make the arm name clear to training and evaluation callers.
A1PatchTokenWorldModel = PatchTokenWorldModel
PatchTokenJEPA = PatchTokenWorldModel


__all__ = [
    "A1PatchTokenWorldModel",
    "ConditionTokenEncoder",
    "MaskedFutureTokenPredictor",
    "PatchTokenJEPA",
    "PatchTokenOutput",
    "PatchTokenWorldModel",
    "deterministic_mask_indices",
    "source_to_query_block_mask",
]

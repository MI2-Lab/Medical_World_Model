"""LOCAL3 plus the preregistered optional 192-to-1 static SPH head."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import operator
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from .contracts import RESPONSE_DIM, SPH_HEAD_PARAMETER_COUNT, arm_spec
from .upstream import (
    DGRSOutput,
    LocalGlobalResponseWorldModel,
    build_local_model,
    validate_local_model_contract,
)


@dataclass
class ResidualSPHOutput(DGRSOutput):
    sph_prediction: torch.Tensor | None = None


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class ResidualSPHWorldModel(LocalGlobalResponseWorldModel):
    """The confirmed LOCAL3 model with, at most, one online linear head.

    SPH never enters the encoder, transition, projector, target encoder, EMA,
    or FTV head.  S0 has no SPH module and is state-dict identical to LOCAL3.
    """

    def __init__(self, experimental_arm: str, *, sph_head_seed: int) -> None:
        spec = arm_spec(experimental_arm)
        # Constructed under the caller's forked RNG in build_model.
        super().__init__(arm="LOCAL3")
        self.experimental_arm = spec.name
        self.sph_head_seed = int(sph_head_seed)
        self.sph_head: nn.Linear | None = None
        if spec.has_sph_head:
            # Restoring the public RNG state is essential for paired dropout,
            # SIGReg and patient-order streams across S0/S1/S2.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.sph_head_seed)
                self.sph_head = nn.Linear(RESPONSE_DIM, 1)
        validate_model_contract(self)

    def forward(
        self, image: torch.Tensor, roi_mask: torch.Tensor | None = None
    ) -> ResidualSPHOutput:
        base = super().forward(image, roi_mask)
        sph = (
            None
            if self.sph_head is None
            else self.sph_head(base.response_state).squeeze(-1)
        )
        return ResidualSPHOutput(**vars(base), sph_prediction=sph)

    def model_config(self) -> dict[str, Any]:
        return {
            "experimental_arm": self.experimental_arm,
            "sph_head_seed": self.sph_head_seed,
        }

    def architecture_contract(self) -> dict[str, Any]:
        contract = dict(super().architecture_contract())
        spec = arm_spec(self.experimental_arm)
        contract.update(
            {
                "schema_version": 2,
                "base_arm": "LOCAL3",
                "experimental_arm": spec.name,
                "observed_response_state": "online_preprojector_r_192d",
                "sph_head": "Linear(192,1)" if spec.has_sph_head else None,
                "sph_target": spec.sph_target,
                "lambda_sph": spec.lambda_sph,
                "sph_head_input": "response_state_only",
                "sph_head_in_target_ema": False,
                "dynamic_sph_supervision": False,
                "clinical_or_treatment_forward_input": False,
            }
        )
        return contract


def _base_state(model: ResidualSPHWorldModel) -> dict[str, torch.Tensor]:
    return {
        name: value
        for name, value in model.state_dict().items()
        if not name.startswith("sph_head.")
    }


def base_initialization_sha256(model: ResidualSPHWorldModel) -> str:
    validate_model_contract(model)
    return tensor_state_sha256(_base_state(model))


def shared_initialization_sha256(model: ResidualSPHWorldModel) -> str:
    """Confirmation-compatible hash of shared non-FTV modules.

    This reproduces the later confirmation hash exactly and intentionally
    excludes both grounding heads.
    """

    names = (
        "encoder",
        "response_projection",
        "projector",
        "target_encoder",
        "target_response_projection",
        "target_projector",
        "transition",
    )
    state: dict[str, torch.Tensor] = {}
    for prefix in names:
        for name, value in getattr(model, prefix).state_dict().items():
            state[f"{prefix}.{name}"] = value
    return tensor_state_sha256(state)


def sph_head_sha256(model: ResidualSPHWorldModel) -> str | None:
    return None if model.sph_head is None else tensor_state_sha256(model.sph_head.state_dict())


def validate_model_contract(model: ResidualSPHWorldModel) -> None:
    if not isinstance(model, ResidualSPHWorldModel):
        raise TypeError("model must be ResidualSPHWorldModel")
    validate_local_model_contract(model, "LOCAL3")
    spec = arm_spec(model.experimental_arm)
    if (model.sph_head is not None) != spec.has_sph_head:
        raise ValueError("SPH head presence differs from arm contract")
    if model.sph_head is not None:
        if (model.sph_head.in_features, model.sph_head.out_features) != (RESPONSE_DIM, 1):
            raise ValueError("SPH head must be Linear(192,1)")
        if sum(parameter.numel() for parameter in model.sph_head.parameters()) != SPH_HEAD_PARAMETER_COUNT:
            raise ValueError("SPH head parameter count drifted")
    if model.requires_roi_mask:
        raise ValueError("residual-SPH model must not accept an ROI mask")
    if model.ftv_head is None:
        raise ValueError("all arms retain the confirmed FTV head")


def build_model(experimental_arm: str, effective_seed: int) -> ResidualSPHWorldModel:
    spec = arm_spec(experimental_arm)
    try:
        seed = operator.index(effective_seed)
    except TypeError as error:
        raise ValueError("effective_seed must be an exact integer") from error
    if isinstance(effective_seed, bool):
        raise ValueError("effective_seed must be an exact integer")
    sph_seed = (int(seed) * 1_000_003 + 71_119) % (2**63 - 1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = ResidualSPHWorldModel(spec.name, sph_head_seed=sph_seed)
    validate_model_contract(model)
    return model


def paired_initialization_report(effective_seed: int) -> dict[str, Any]:
    models = {arm: build_model(arm, effective_seed) for arm in ("S0", "S1", "S2", "S2_L10")}
    shared = {arm: shared_initialization_sha256(model) for arm, model in models.items()}
    base = {arm: base_initialization_sha256(model) for arm, model in models.items()}
    heads = {arm: sph_head_sha256(model) for arm, model in models.items()}
    checks = {
        "shared_initialization_identical": len(set(shared.values())) == 1,
        "base_initialization_identical": len(set(base.values())) == 1,
        "s0_has_no_sph_head": heads["S0"] is None,
        "all_sph_heads_identical": len({heads[arm] for arm in ("S1", "S2", "S2_L10")}) == 1,
    }
    if not all(checks.values()):
        raise AssertionError(f"paired initialization failed: {checks}")
    return {
        "schema_version": 1,
        "effective_seed": int(effective_seed),
        "shared_initialization_sha256": next(iter(shared.values())),
        "base_initialization_sha256": next(iter(base.values())),
        "per_arm_shared": shared,
        "per_arm_base": base,
        "per_arm_sph_head": heads,
        "checks": checks,
    }


def load_checkpoint(
    path: str, device: str | torch.device = "cpu"
) -> tuple[ResidualSPHWorldModel, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("model_config"), dict):
        raise ValueError("checkpoint is missing the frozen model_config")
    if payload.get("test_data_used") is not False or payload.get("pcr_used") is not False:
        raise PermissionError("representation checkpoint is not test/pCR blind")
    model = ResidualSPHWorldModel(**payload["model_config"])
    state = payload.get("state_dict", payload.get("model_state"))
    if not isinstance(state, dict):
        raise ValueError("checkpoint is missing model state")
    model.load_state_dict(state, strict=True)
    validate_model_contract(model)
    model.to(device).eval()
    return model, payload


__all__ = [
    "ResidualSPHOutput",
    "ResidualSPHWorldModel",
    "base_initialization_sha256",
    "build_model",
    "load_checkpoint",
    "paired_initialization_report",
    "shared_initialization_sha256",
    "sph_head_sha256",
    "tensor_state_sha256",
    "validate_model_contract",
]

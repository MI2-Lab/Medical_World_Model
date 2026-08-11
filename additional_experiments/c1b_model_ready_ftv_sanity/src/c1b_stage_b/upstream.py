"""Direct reuse and validation of the frozen G3 DGRS model/objective.

No model, transition, pooling, SIGReg, or FTV-head implementation is copied
into this experiment.  Imports are resolved to the exact G3 source tree and
validated before use.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
import random
import sys
from typing import Any, Mapping

import numpy as np
import torch

from .contracts import ARM_SPECS, G3_SRC, MODEL_KWARGS, arm_spec, file_sha256


def _load_upstream() -> tuple[Any, Any, Any]:
    source = str(G3_SRC.resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    model_module = importlib.import_module("dgrs.model")
    training_module = importlib.import_module("dgrs.training")
    targets_module = importlib.import_module("dgrs.targets")
    expected_model = (G3_SRC / "dgrs" / "model.py").resolve()
    expected_training = (G3_SRC / "dgrs" / "training.py").resolve()
    if Path(inspect.getfile(model_module.DGRSWorldModel)).resolve() != expected_model:
        raise ImportError("DGRSWorldModel did not resolve to g3_multiseed_generalization")
    if Path(inspect.getfile(training_module.DGRSObjective)).resolve() != expected_training:
        raise ImportError("DGRSObjective did not resolve to g3_multiseed_generalization")
    return model_module, training_module, targets_module


_MODEL_MODULE, _TRAINING_MODULE, _TARGETS_MODULE = _load_upstream()
DGRSWorldModel = _MODEL_MODULE.DGRSWorldModel
DGRSOutput = _MODEL_MODULE.DGRSOutput
DGRSObjective = _TRAINING_MODULE.DGRSObjective
PooledFTVTransform = _TARGETS_MODULE.PooledFTVTransform
load_checkpoint_for_evaluation = _MODEL_MODULE.load_checkpoint_for_evaluation


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def common_initialization_sha256(model: torch.nn.Module) -> str:
    state: dict[str, torch.Tensor] = {}
    for prefix in (
        "encoder",
        "response_projection",
        "projector",
        "target_encoder",
        "target_response_projection",
        "target_projector",
        "transition",
    ):
        module = getattr(model, prefix)
        for name, value in module.state_dict().items():
            state[f"{prefix}.{name}"] = value
    return tensor_state_sha256(state)


def transition_sha256(model: torch.nn.Module) -> str:
    return tensor_state_sha256(model.transition.state_dict())


def build_model(arm: str, effective_seed: int) -> torch.nn.Module:
    spec = arm_spec(arm)
    # Restore the caller's RNG.  Each arm starts from the exact same seed; the
    # upstream G3 head already uses fork_rng and cannot perturb shared init.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(effective_seed))
        model = DGRSWorldModel(model_name=spec.upstream_model_name, **MODEL_KWARGS)
    validate_model_contract(model, arm)
    return model


def build_objective(arm: str) -> torch.nn.Module:
    spec = arm_spec(arm)
    return DGRSObjective(
        model_name=spec.upstream_model_name,
        lambda_ftv=0.25 if spec.grounded else 0.0,
        sigreg_weight=0.09,
        sigreg_projections=256,
        step_weights=(2.0, 1.0, 0.5),
    )


def validate_model_contract(model: torch.nn.Module, arm: str) -> None:
    spec = arm_spec(arm)
    contract = model.architecture_contract()
    if model.__class__ is not DGRSWorldModel:
        raise TypeError("Stage B must use the upstream DGRSWorldModel class directly")
    if contract.get("backbone_input") != "DCE7" or int(contract.get("image_channels", -1)) != 7:
        raise ValueError("upstream model is not strict DCE7")
    if contract.get("pooling") != "gap" or model.requires_roi_mask:
        raise ValueError("L1/L3/N1/N3 must use strict GAP without an ROI input")
    if contract.get("observed_response_state") != "online_preprojector_r":
        raise ValueError("feature contract must be online pre-projector r")
    if contract.get("transition") != "M0_direct_next_state_causal_transformer":
        raise ValueError("upstream transition contract drifted")
    if bool(model.ftv_head is not None) != spec.grounded:
        raise ValueError("FTV head must exist only for L3/N3")


def paired_initialization_report(effective_seed: int) -> dict[str, Any]:
    hashes: dict[str, str] = {}
    transition_hashes: dict[str, str] = {}
    head_presence: dict[str, bool] = {}
    for arm in ARM_SPECS:
        model = build_model(arm, effective_seed)
        hashes[arm] = common_initialization_sha256(model)
        transition_hashes[arm] = transition_sha256(model)
        head_presence[arm] = model.ftv_head is not None
        del model
    if len(set(hashes.values())) != 1 or len(set(transition_hashes.values())) != 1:
        raise AssertionError("four-arm paired shared initialization/transition hash mismatch")
    return {
        "effective_seed": int(effective_seed),
        "common_initialization_sha256": next(iter(hashes.values())),
        "transition_sha256": next(iter(transition_hashes.values())),
        "per_arm_common_sha256": hashes,
        "per_arm_transition_sha256": transition_hashes,
        "ftv_head_present": head_presence,
        "upstream_model_sha256": file_sha256(G3_SRC / "dgrs" / "model.py"),
        "upstream_objective_sha256": file_sha256(G3_SRC / "dgrs" / "training.py"),
    }


__all__ = [
    "DGRSObjective",
    "DGRSOutput",
    "DGRSWorldModel",
    "PooledFTVTransform",
    "build_model",
    "build_objective",
    "common_initialization_sha256",
    "load_checkpoint_for_evaluation",
    "paired_initialization_report",
    "seed_everything",
    "transition_sha256",
    "validate_model_contract",
]

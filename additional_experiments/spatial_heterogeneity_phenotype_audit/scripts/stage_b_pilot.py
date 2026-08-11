#!/usr/bin/env python3
"""Run the prospectively gated Response--Phenotype Dual Statistic pilot.

Stage B is deliberately a single-arm, single-seed feasibility run.  Before
importing a model or opening an image cache, this entry point authenticates the
Stage-A gate file, its authorization record, and the experiment preregistration
lock.  If neither Gate A nor Gate C passed, it writes only the public
``NOT_RUN_NOT_AUTHORIZED`` row and cannot enter a training code path.

Authorized execution supports either an all-fold run or independent ``--fold``
workers followed by ``--finalize-only``.  The latter changes scheduling only:
every fold retains the same effective seed, model, objective, batching, and
selection contract.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict, dataclass
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common import (  # noqa: E402
    atomic_csv,
    atomic_json,
    canonical_sha256,
    file_sha256,
    load_config,
    ordered_sha256,
    private_directory,
    require_preregistration_lock,
)
from pooling import weighted_mean, weighted_population_std  # noqa: E402
from verify_cache_integrity import (  # noqa: E402
    PRIVATE_MANIFEST as CACHE_INTEGRITY_PRIVATE_MANIFEST,
    PUBLIC_CONTRACT as CACHE_INTEGRITY_PUBLIC_CONTRACT,
    require_cache_integrity,
)


SEED_BASE = 2026
FOLDS = tuple(range(5))
VISITS = ("T0", "T1", "T2", "T3")
FEATURE_CHANNELS = 128
BRANCH_DIM = 96
STATE_DIM = 192
FEATURE_SHAPE_ZYX = (14, 22, 20)
MODEL_ARM = "RESPONSE_PHENOTYPE_DUAL_STATISTIC_STATE"
FEATURE_VARIANT = "DUAL_MEAN_STD_192"
FORMAL_HYPERPARAMETERS: Mapping[str, Any] = {
    "physical_batch_size": 4,
    "accumulation_steps": 8,
    "workers": 2,
    "epochs": 12,
    "patience": 4,
    "learning_rate": 5e-5,
    "weight_decay": 1e-4,
    "ema_momentum": 0.996,
    "max_grad_norm": 5.0,
    "min_representation_std": 0.05,
}
OBJECTIVE_CONTRACT: Mapping[str, Any] = {
    "name": "FTV_only_DGRS",
    "lambda_ftv": 0.25,
    "sigreg_weight": 0.09,
    "sigreg_projections": 256,
    "step_weights": [2.0, 1.0, 0.5],
}
EXPECTED_STAGE_B_CONFIG: Mapping[str, Any] = {
    "authorization": "gate_A_or_gate_C",
    "seed_bases": [SEED_BASE],
    "folds": list(FOLDS),
    "effective_seed": "seed_base_plus_fold",
    "architecture": {
        "support": "exact_fixed_64mm_fractional_LOCAL",
        "mean": "weighted_mean_128_then_Linear_128x96",
        "std": "weighted_population_ddof0_128_then_Linear_128x96",
        "std_zero_variance_backward": (
            "exact_forward_with_subgradient_0_and_finite_gradients"
        ),
        "concatenation_order": ["mean_96", "std_96"],
        "normalization": "joint_LayerNorm_192_after_concatenation",
        "state": "online_preprojector_192",
    },
    "cohort": {
        "primary_patient_count": 808,
        "external_train_only_patient_count": 139,
        "authenticated_cache_patient_count": 947,
        "train_split": "outer_fold_primary_train_plus_exact_external_train_only",
        "validation_split": "outer_fold_primary_validation_only",
        "test_split": "outer_fold_primary_test_only",
        "require_all_cache_content_proof_before_data_access": True,
    },
    "authorization_evidence": {
        "require_stage_a_run_summary_complete": True,
        "require_gates_hash_from_run_summary_artifacts": True,
        "require_authorization_hash_from_run_summary_artifacts": True,
    },
    "initialization": {
        "source": "fresh_canonical_grounded_LOCAL3_at_effective_seed",
        "mean_branch": "rows_0_through_95_of_baseline_Linear_128x192",
        "std_branch": "rows_96_through_191_of_baseline_Linear_128x192",
        "joint_norm": "exact_copy_of_baseline_LayerNorm_192",
        "equivalence_condition": "dual_state_equals_baseline_state_when_std_equals_mean",
        "shared_modules": "bitwise_unchanged_from_fresh_baseline",
    },
    "trainable": [
        "online_encoder",
        "mean_projection",
        "std_projection",
        "joint_layer_norm",
        "projector",
        "transition",
        "FTV_head",
    ],
    "ema_only": [
        "target_encoder",
        "target_mean_projection",
        "target_std_projection",
        "target_joint_layer_norm",
        "target_projector",
    ],
    "objective": {
        "formula": "L_JEPA_plus_0.25_L_FTV",
        "lambda_ftv": OBJECTIVE_CONTRACT["lambda_ftv"],
        "sigreg_weight": OBJECTIVE_CONTRACT["sigreg_weight"],
        "sigreg_projections": OBJECTIVE_CONTRACT["sigreg_projections"],
        "temporal_step_weights": OBJECTIVE_CONTRACT["step_weights"],
        "grounding": "FTV_only",
    },
    "training": {
        "physical_batch_size": FORMAL_HYPERPARAMETERS["physical_batch_size"],
        "accumulation_steps": FORMAL_HYPERPARAMETERS["accumulation_steps"],
        "logical_batch_size": 32,
        "optimizer": "AdamW",
        "learning_rate": FORMAL_HYPERPARAMETERS["learning_rate"],
        "weight_decay": FORMAL_HYPERPARAMETERS["weight_decay"],
        "epochs_max": FORMAL_HYPERPARAMETERS["epochs"],
        "early_stopping_patience": FORMAL_HYPERPARAMETERS["patience"],
        "ema_momentum": FORMAL_HYPERPARAMETERS["ema_momentum"],
        "max_grad_norm": FORMAL_HYPERPARAMETERS["max_grad_norm"],
        "workers": FORMAL_HYPERPARAMETERS["workers"],
        "minimum_validation_representation_std": FORMAL_HYPERPARAMETERS[
            "min_representation_std"
        ],
        "augmentation": "none",
    },
    "checkpoint_selection": {
        "eligible": "finite_and_validation_representation_std_at_least_0.05",
        "metric": "minimum_validation_total_objective",
        "tie_break": "earliest_epoch",
        "test_data_used": False,
        "refit_after_selection": False,
    },
    "post_selection_probes": {
        "feature": "selected_online_preprojector_192",
        "phenotype": ["HR", "HER2", "subtype_4class"],
        "pcr_views": ["T0", "T0-T1", "T0-T2", "T0-T3"],
        "pcr_populations": ["full_808", "ftv_complete_375"],
        "paired_stage_a_baseline": ("seed2026_LOCAL3_P1_same_view_target_population_n"),
        "paired_metrics": [
            "auroc",
            "auprc",
            "balanced_accuracy",
            "pcr_brier_improvement_baseline_minus_dual",
        ],
        "reuse_stage_a_outer_fold_logistic_contract": True,
    },
    "forbidden": [
        "attention_pooling",
        "transformer_pooling",
        "new_transformer_module",
        "mask_input",
        "oracle_region_input",
        "HR_supervision",
        "HER2_supervision",
        "pCR_supervision",
        "delta_supervision",
    ],
}
TABLE8_COLUMNS = (
    "status",
    "stage",
    "seed",
    "arm",
    "analysis",
    "view",
    "target",
    "variant",
    "population",
    "n",
    "n_positive",
    "n_negative",
    "n_classes",
    "auroc",
    "auprc",
    "balanced_accuracy",
    "brier",
    "stage_a_baseline_seed",
    "stage_a_baseline_arm",
    "stage_a_baseline_variant",
    "baseline_auroc",
    "baseline_auprc",
    "baseline_balanced_accuracy",
    "baseline_brier",
    "delta_auroc",
    "delta_auprc",
    "delta_balanced_accuracy",
    "brier_improvement",
)


def validate_stage_b_config(config: Mapping[str, Any]) -> None:
    """Reject any nested Stage-B protocol divergence before authorization."""

    observed = config.get("stage_b")
    if not isinstance(observed, Mapping):
        raise ValueError("config.stage_b must be a mapping")
    if dict(observed) != dict(EXPECTED_STAGE_B_CONFIG):
        raise ValueError(
            "config.stage_b differs from the exact prospective implementation: "
            f"expected_sha256={canonical_sha256(EXPECTED_STAGE_B_CONFIG)}, "
            f"observed_sha256={canonical_sha256(observed)}"
        )
    analysis = config.get("analysis")
    if not isinstance(analysis, Mapping):
        raise ValueError("config.analysis must be a mapping")
    cross_contract = {
        "phenotype_targets": tuple(analysis.get("phenotype_targets", ()))
        == tuple(EXPECTED_STAGE_B_CONFIG["post_selection_probes"]["phenotype"]),
        "pcr_views": tuple(analysis.get("pcr_timings", ()))
        == tuple(EXPECTED_STAGE_B_CONFIG["post_selection_probes"]["pcr_views"]),
        "pcr_populations": tuple(analysis.get("pcr_populations", ()))
        == tuple(EXPECTED_STAGE_B_CONFIG["post_selection_probes"]["pcr_populations"]),
    }
    failed = sorted(name for name, passed in cross_contract.items() if not passed)
    if failed:
        raise ValueError(f"Stage-B/Stage-A probe contracts diverged: {failed}")


@dataclass(frozen=True)
class StageBAuthorization:
    """Authenticated Stage-A permission for the conditional pilot."""

    path: Path
    gates_path: Path
    run_summary_path: Path
    sha256: str
    gates_sha256: str
    run_summary_sha256: str
    authorized: bool
    gate_a_passed: bool
    gate_c_passed: bool
    status: str


@dataclass(frozen=True)
class AuthenticatedCacheClosure:
    """Identifier-free provenance plus private membership authenticated once."""

    evidence: Mapping[str, Any]
    all_patient_ids: frozenset[str]
    primary_patient_ids: frozenset[str]
    train_only_patient_ids: frozenset[str]


@dataclass
class DualStatisticOutput:
    """Exact field interface consumed by the sealed DGRS objective."""

    response_state: torch.Tensor
    online_state: torch.Tensor
    target_response_state: torch.Tensor
    target_state: torch.Tensor
    target_next: torch.Tensor
    predicted_next: torch.Tensor
    ftv_prediction: torch.Tensor | None
    roi_valid: torch.Tensor | None


class DualStatisticProjection(nn.Module):
    """Factorized mean/SD projection followed by one copied joint LayerNorm."""

    def __init__(
        self,
        mean_projection: nn.Linear,
        std_projection: nn.Linear,
        joint_norm: nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.mean_projection = mean_projection
        self.std_projection = std_projection
        self.joint_norm = joint_norm
        self.validate_contract()

    @classmethod
    def from_baseline(cls, baseline: nn.Module) -> "DualStatisticProjection":
        linear, normalization = _baseline_projection_parts(baseline)
        # Creating the two Linear modules must not advance the formal training
        # RNG relative to the paired grounded LOCAL baseline.
        with torch.random.fork_rng(devices=[]):
            mean_projection = nn.Linear(FEATURE_CHANNELS, BRANCH_DIM)
            std_projection = nn.Linear(FEATURE_CHANNELS, BRANCH_DIM)
        with torch.no_grad():
            mean_projection.weight.copy_(linear.weight[:BRANCH_DIM])
            mean_projection.bias.copy_(linear.bias[:BRANCH_DIM])
            std_projection.weight.copy_(linear.weight[BRANCH_DIM:])
            std_projection.bias.copy_(linear.bias[BRANCH_DIM:])
        return cls(mean_projection, std_projection, copy.deepcopy(normalization))

    def validate_contract(self) -> None:
        expected = (FEATURE_CHANNELS, BRANCH_DIM)
        for name, branch in (
            ("mean", self.mean_projection),
            ("std", self.std_projection),
        ):
            if (
                not isinstance(branch, nn.Linear)
                or (
                    branch.in_features,
                    branch.out_features,
                )
                != expected
            ):
                raise ValueError(f"{name} branch must be Linear(128,96)")
        if not isinstance(self.joint_norm, nn.LayerNorm) or tuple(
            self.joint_norm.normalized_shape
        ) != (STATE_DIM,):
            raise ValueError("dual statistic state requires joint LayerNorm(192)")

    def forward(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        if (
            mean.shape != std.shape
            or mean.ndim != 2
            or mean.size(1) != FEATURE_CHANNELS
        ):
            raise ValueError("dual projection inputs must be paired [N,128] tensors")
        raw = torch.cat((self.mean_projection(mean), self.std_projection(std)), dim=-1)
        output = self.joint_norm(raw)
        if tuple(output.shape) != (mean.size(0), STATE_DIM):
            raise AssertionError("dual projection output shape drifted")
        return output


class ResponsePhenotypeDualStatisticState(nn.Module):
    """Grounded LOCAL world model with a factorized mean/SD response state.

    ``baseline`` must be the canonical freshly initialized ``LOCAL3`` model for
    the same effective seed.  Every upstream module is reused directly; only
    the baseline response projection is deterministically split into two
    96-dimensional branches.
    """

    def __init__(self, baseline: nn.Module, *, effective_seed: int) -> None:
        super().__init__()
        if str(getattr(baseline, "arm", "")) != "LOCAL3":
            raise ValueError("dual statistic initialization requires canonical LOCAL3")
        if str(getattr(baseline, "response_architecture", "")) != "LOCAL":
            raise ValueError("paired baseline must use exact fixed LOCAL pooling")
        if str(getattr(baseline, "model_name", "")) != "G3":
            raise ValueError("paired baseline must be the grounded G3 objective arm")
        if getattr(baseline, "ftv_head", None) is None:
            raise ValueError("paired grounded LOCAL baseline has no FTV head")
        local_weight = getattr(baseline, "local_pooling_weight", None)
        if not isinstance(local_weight, torch.Tensor) or tuple(local_weight.shape) != (
            1,
            1,
            *FEATURE_SHAPE_ZYX,
        ):
            raise ValueError("paired baseline lacks the exact fixed LOCAL weight")

        self.effective_seed = int(effective_seed)
        self.model_name = "G3"
        self.arm = MODEL_ARM
        self.response_architecture = "LOCAL_MEAN_STD_FACTORIZED"
        self.image_channels = int(baseline.image_channels)
        self.base_channels = int(baseline.base_channels)
        self.latent_dim = int(baseline.latent_dim)
        self.predictor_depth = int(baseline.predictor_depth)
        self.predictor_heads = int(baseline.predictor_heads)
        self.predictor_mlp_dim = int(baseline.predictor_mlp_dim)
        self.dropout = float(baseline.dropout)
        self.input_shape_zyx = tuple(int(value) for value in baseline.input_shape_zyx)
        self.spacing_xyz_mm = tuple(float(value) for value in baseline.spacing_xyz_mm)
        self.local_window_mm_xyz = tuple(
            float(value) for value in baseline.local_window_mm_xyz
        )

        # Transfer exact upstream modules.  Online encoder/projections/projector,
        # transition, and FTV head remain trainable; targets remain EMA-only.
        self.encoder = baseline.encoder
        self.projector = baseline.projector
        self.transition = baseline.transition
        self.ftv_head = baseline.ftv_head
        self.target_encoder = baseline.target_encoder
        self.target_projector = baseline.target_projector
        self.response_projection = DualStatisticProjection.from_baseline(
            baseline.response_projection
        )
        self.target_response_projection = DualStatisticProjection.from_baseline(
            baseline.target_response_projection
        ).requires_grad_(False)
        self.register_buffer(
            "local_pooling_weight",
            local_weight.detach().clone(),
            persistent=True,
        )
        self.validate_contract()

    @property
    def requires_roi_mask(self) -> bool:
        return False

    def _validate_sequence_inputs(
        self, image: torch.Tensor, roi_mask: torch.Tensor | None
    ) -> tuple[int, int]:
        if roi_mask is not None:
            raise ValueError("Stage B dual statistic state never accepts an ROI mask")
        if image.ndim != 6 or image.size(2) != 7:
            raise ValueError("image must be [B,V,7,Z,Y,X]")
        if tuple(int(value) for value in image.shape[-3:]) != self.input_shape_zyx:
            raise ValueError("Stage B input geometry differs from canonical C1B-H")
        return int(image.size(0)), int(image.size(1))

    def _encode_sequence(
        self,
        image: torch.Tensor,
        roi_mask: torch.Tensor | None,
        encoder: nn.Module,
        response_projection: DualStatisticProjection,
        projector: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        batch, visits = self._validate_sequence_inputs(image, roi_mask)
        flat = image.reshape(batch * visits, *image.shape[2:])
        spatial = encoder(flat)
        if not isinstance(spatial, torch.Tensor) or tuple(spatial.shape[:2]) != (
            batch * visits,
            FEATURE_CHANNELS,
        ):
            raise ValueError("encoder output must be [B*V,128,D,H,W]")
        if tuple(int(value) for value in spatial.shape[-3:]) != FEATURE_SHAPE_ZYX:
            raise ValueError(
                "actual encoder final spatial shape differs from [14,22,20]"
            )
        mean = weighted_mean(spatial, self.local_pooling_weight)
        std = weighted_population_std(spatial, self.local_pooling_weight)
        response = response_projection(mean, std).reshape(batch, visits, STATE_DIM)
        projected = projector(response).reshape(batch, visits, STATE_DIM)
        return response, projected, None

    def encode_online(
        self, image: torch.Tensor, roi_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        return self._encode_sequence(
            image, roi_mask, self.encoder, self.response_projection, self.projector
        )

    @torch.no_grad()
    def encode_target(
        self, image: torch.Tensor, roi_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        self.target_encoder.eval()
        self.target_response_projection.eval()
        self.target_projector.eval()
        return self._encode_sequence(
            image,
            roi_mask,
            self.target_encoder,
            self.target_response_projection,
            self.target_projector,
        )

    def encode_response(
        self, image: torch.Tensor, roi_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        response, _, _ = self.encode_online(image, roi_mask)
        return response

    def forward(
        self, image: torch.Tensor, roi_mask: torch.Tensor | None = None
    ) -> DualStatisticOutput:
        if image.ndim != 6 or image.size(1) != len(VISITS):
            raise ValueError("training forward requires exactly T0--T3")
        response, online, _ = self.encode_online(image, roi_mask)
        target_response, target, _ = self.encode_target(image, roi_mask)
        target_response = target_response.detach()
        target = target.detach()
        predicted_next = self.transition(online[:, :-1])
        ftv_prediction = self.ftv_head(response).squeeze(-1)
        return DualStatisticOutput(
            response_state=response,
            online_state=online,
            target_response_state=target_response,
            target_state=target,
            target_next=target[:, 1:],
            predicted_next=predicted_next,
            ftv_prediction=ftv_prediction,
            roi_valid=None,
        )

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        if not 0.0 < float(momentum) < 1.0:
            raise ValueError("EMA momentum must be in (0,1)")
        for online_module, target_module in (
            (self.encoder, self.target_encoder),
            (self.response_projection, self.target_response_projection),
            (self.projector, self.target_projector),
        ):
            online_parameters = tuple(online_module.parameters())
            target_parameters = tuple(target_module.parameters())
            if len(online_parameters) != len(target_parameters):
                raise ValueError("online/target EMA parameter structures differ")
            for online, target in zip(
                online_parameters, target_parameters, strict=True
            ):
                target.data.mul_(momentum).add_(online.data, alpha=1.0 - momentum)
            online_buffers = tuple(online_module.buffers())
            target_buffers = tuple(target_module.buffers())
            if len(online_buffers) != len(target_buffers):
                raise ValueError("online/target EMA buffer structures differ")
            for online, target in zip(online_buffers, target_buffers, strict=True):
                target.copy_(online)

    def validate_contract(self) -> None:
        self.response_projection.validate_contract()
        self.target_response_projection.validate_contract()
        if self.latent_dim != STATE_DIM or self.image_channels != 7:
            raise ValueError("upstream response/image dimensions drifted")
        if self.ftv_head is None or (
            self.ftv_head.in_features,
            self.ftv_head.out_features,
        ) != (STATE_DIM, 1):
            raise ValueError(
                "Stage B requires the exact grounded Linear(192,1) FTV head"
            )
        for name in (
            "target_encoder",
            "target_response_projection",
            "target_projector",
        ):
            if any(
                parameter.requires_grad
                for parameter in getattr(self, name).parameters()
            ):
                raise ValueError(f"{name} must remain EMA-only")
        for name in (
            "encoder",
            "response_projection",
            "projector",
            "transition",
            "ftv_head",
        ):
            parameters = tuple(getattr(self, name).parameters())
            if not parameters or not all(
                parameter.requires_grad for parameter in parameters
            ):
                raise ValueError(f"{name} must remain online/trainable")
        if self.local_pooling_weight.requires_grad:
            raise ValueError("fixed LOCAL weight may not be trainable")

    def architecture_contract(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "name": MODEL_ARM,
            "model_name": "G3",
            "backbone_input": "C1B-H_DCE7",
            "input_shape_zyx": list(self.input_shape_zyx),
            "encoder_final_shape": [FEATURE_CHANNELS, *FEATURE_SHAPE_ZYX],
            "local_weight": "exact_fixed_64mm_fractional_sampling_cell_overlap",
            "mean_branch": "weighted_mean_128_then_Linear_128x96",
            "std_branch": "weighted_population_ddof0_128_then_Linear_128x96",
            "std_zero_variance_backward": EXPECTED_STAGE_B_CONFIG["architecture"][
                "std_zero_variance_backward"
            ],
            "concatenation_order": ["mean_96", "std_96"],
            "normalization": "copied_joint_baseline_LayerNorm_192",
            "response_state": "online_preprojector_192",
            "paired_initialization": (
                "LOCAL3_Linear(128,192)_rows_0_95_to_mean_rows_96_191_to_std"
            ),
            "baseline_equivalence_condition": "std_equals_mean",
            "online_trainable": list(EXPECTED_STAGE_B_CONFIG["trainable"]),
            "causal_transition": (
                "canonical_unchanged_JEPA_transition_not_spatial_aggregation"
            ),
            "ema_only": list(EXPECTED_STAGE_B_CONFIG["ema_only"]),
            "objective": dict(EXPECTED_STAGE_B_CONFIG["objective"]),
            "forbidden_inputs_or_supervision": list(
                EXPECTED_STAGE_B_CONFIG["forbidden"]
            ),
        }

    def model_config(self) -> dict[str, Any]:
        return {
            "effective_seed": self.effective_seed,
            "architecture": MODEL_ARM,
            "state_dim": STATE_DIM,
        }

    def parameter_counts(self) -> dict[str, int]:
        def count(module: nn.Module) -> int:
            return int(sum(parameter.numel() for parameter in module.parameters()))

        return {
            "encoder": count(self.encoder),
            "mean_projection": count(self.response_projection.mean_projection),
            "std_projection": count(self.response_projection.std_projection),
            "joint_layer_norm": count(self.response_projection.joint_norm),
            "projector": count(self.projector),
            "transition": count(self.transition),
            "ftv_head": count(self.ftv_head),
            "target_ema_only": count(self.target_encoder)
            + count(self.target_response_projection)
            + count(self.target_projector),
            "trainable_total": int(
                sum(
                    parameter.numel()
                    for parameter in self.parameters()
                    if parameter.requires_grad
                )
            ),
            "frozen_total": int(
                sum(
                    parameter.numel()
                    for parameter in self.parameters()
                    if not parameter.requires_grad
                )
            ),
        }


def _baseline_projection_parts(module: nn.Module) -> tuple[nn.Linear, nn.LayerNorm]:
    if not isinstance(module, nn.Sequential) or len(module) != 2:
        raise TypeError("canonical LOCAL response projection is not Linear+LayerNorm")
    linear, normalization = module[0], module[1]
    if not isinstance(linear, nn.Linear) or (
        linear.in_features,
        linear.out_features,
    ) != (FEATURE_CHANNELS, STATE_DIM):
        raise ValueError("canonical LOCAL baseline must contain Linear(128,192)")
    if not isinstance(normalization, nn.LayerNorm) or tuple(
        normalization.normalized_shape
    ) != (STATE_DIM,):
        raise ValueError("canonical LOCAL baseline must contain LayerNorm(192)")
    return linear, normalization


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _run_summary_root(summary_path: Path) -> Path:
    return (
        summary_path.parent.parent
        if summary_path.parent.name == "metrics"
        else summary_path.parent
    ).resolve()


def _authenticated_summary_artifact(
    summary: Mapping[str, Any], summary_path: Path, name: str
) -> Path:
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("Stage-A run summary lacks an artifact inventory")
    record = artifacts.get(name)
    if not isinstance(record, Mapping):
        raise ValueError(f"Stage-A run summary lacks artifact: {name}")
    relative = Path(str(record.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Stage-A run summary artifact path is unsafe: {name}")
    summary_root = _run_summary_root(summary_path)
    try:
        artifact = (summary_root / relative).resolve(strict=True)
    except OSError as error:
        raise ValueError(f"Stage-A run summary artifact is missing: {name}") from error
    if summary_root not in artifact.parents:
        raise ValueError(f"Stage-A run summary artifact escaped experiment: {name}")
    if (
        record.get("sha256") != file_sha256(artifact)
        or record.get("size_bytes") != artifact.stat().st_size
        or record.get("patient_level_private") is not False
    ):
        raise ValueError(
            f"Stage-A run summary artifact hash/size/privacy drifted: {name}"
        )
    return artifact


def validate_stage_b_authorization(
    config: Mapping[str, Any],
    authorization_path: str | Path,
    gates_path: str | Path,
    run_summary_path: str | Path | None = None,
) -> StageBAuthorization:
    """Authenticate the completed Stage-A artifact closure and A-or-C rule."""

    authorization_source = Path(authorization_path).expanduser().resolve(strict=True)
    gates_source = Path(gates_path).expanduser().resolve(strict=True)
    summary_source = (
        authorization_source.parent / "run_summary.json"
        if run_summary_path is None
        else Path(run_summary_path).expanduser()
    ).resolve(strict=True)
    try:
        authorization = json.loads(authorization_source.read_text(encoding="utf-8"))
        gates = json.loads(gates_source.read_text(encoding="utf-8"))
        summary = json.loads(summary_source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "Stage-B authorization/gate/run-summary evidence is unreadable"
        ) from error
    if not all(isinstance(value, dict) for value in (authorization, gates, summary)):
        raise ValueError("Stage-B authorization/gates/run summary must be JSON objects")
    summary_checks = {
        "schema_version": summary.get("schema_version") == 1,
        "experiment": summary.get("experiment")
        == "spatial_heterogeneity_phenotype_audit",
        "stage": summary.get("stage") == "A",
        "status": summary.get("status") == "COMPLETE",
    }
    failed_summary = sorted(
        name for name, passed in summary_checks.items() if not passed
    )
    if failed_summary:
        raise ValueError(
            f"Stage-A run summary is not a COMPLETE formal closure: {failed_summary}"
        )
    summary_root = _run_summary_root(summary_source)
    required_summary_artifacts = (
        ("gates", gates_source),
        ("stage_b_authorization", authorization_source),
        ("table2", summary_root / "metrics" / "table2_phenotype_probes.csv"),
        ("table3", summary_root / "metrics" / "table3_mri_only_pcr.csv"),
    )
    for name, expected_path in required_summary_artifacts:
        observed_path = _authenticated_summary_artifact(summary, summary_source, name)
        if observed_path != expected_path.resolve():
            raise ValueError(f"Stage-A run summary names a different {name} artifact")
    if gates.get("schema_version") != 1 or gates.get("stage") != "A":
        raise ValueError("Stage-A gates schema/stage drifted")
    if gates.get("experiment") != "spatial_heterogeneity_phenotype_audit":
        raise ValueError("Stage-A gates name another experiment")
    if gates.get("status") != "COMPLETE":
        raise ValueError("Stage A is not complete")
    gate_values = gates.get("gates")
    if not isinstance(gate_values, Mapping):
        raise ValueError("Stage-A gates lack gate records")
    for gate in ("A", "B", "C", "D"):
        if not isinstance(gate_values.get(gate), Mapping) or not isinstance(
            gate_values[gate].get("passed"), bool
        ):
            raise ValueError(f"Stage-A Gate {gate} result is absent or non-boolean")
    gate_a = bool(gate_values["A"]["passed"])
    gate_c = bool(gate_values["C"]["passed"])
    expected_authorized = gate_a or gate_c
    gates_digest = file_sha256(gates_source)

    required_authorization = {
        "schema_version",
        "experiment",
        "authorization_rule",
        "authorized",
        "status",
        "gate_a_passed",
        "gate_c_passed",
        "stage_b_contract",
        "stage_a_gates_sha256",
        "contains_patient_level_data",
    }
    missing = required_authorization.difference(authorization)
    if missing:
        raise ValueError(f"Stage-B authorization misses fields: {sorted(missing)}")
    expected_status = (
        "AUTHORIZED_PENDING_EXECUTION"
        if expected_authorized
        else "NOT_RUN_NOT_AUTHORIZED"
    )
    checks = {
        "schema_version": authorization["schema_version"] == 1,
        "experiment": authorization["experiment"]
        == "spatial_heterogeneity_phenotype_audit",
        "authorization_rule": authorization["authorization_rule"] == "Gate A OR Gate C",
        "authorized": authorization["authorized"] is expected_authorized,
        "status": authorization["status"] == expected_status,
        "gate_a": authorization["gate_a_passed"] is gate_a,
        "gate_c": authorization["gate_c_passed"] is gate_c,
        "contract": authorization["stage_b_contract"] == config["stage_b"],
        "privacy": authorization["contains_patient_level_data"] is False,
        "gate_hash": authorization["stage_a_gates_sha256"] == gates_digest,
        "gate_summary": gates.get("stage_b_authorized") is expected_authorized,
        "run_summary_gate": summary.get("stage_b_authorized") is expected_authorized,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"Stage-B authorization closure failed: {failed}")
    if not _is_sha256(authorization["stage_a_gates_sha256"]):
        raise ValueError("authorization Stage-A gate hash is not lowercase SHA-256")
    return StageBAuthorization(
        path=authorization_source,
        gates_path=gates_source,
        run_summary_path=summary_source,
        sha256=file_sha256(authorization_source),
        gates_sha256=gates_digest,
        run_summary_sha256=file_sha256(summary_source),
        authorized=expected_authorized,
        gate_a_passed=gate_a,
        gate_c_passed=gate_c,
        status=expected_status,
    )


def require_stage_b_preregistration(
    config: Mapping[str, Any], lock: Mapping[str, Any]
) -> str:
    """Require this exact pilot implementation to be present in the lock."""

    expected_key = "scripts/stage_b_pilot.py"
    implementation = lock.get("implementation_sha256")
    if not isinstance(implementation, Mapping) or expected_key not in implementation:
        raise ValueError("preregistration lock does not bind scripts/stage_b_pilot.py")
    observed = file_sha256(Path(__file__))
    if implementation[expected_key] != observed:
        raise ValueError("Stage-B pilot implementation differs from preregistration")
    if lock.get("config_canonical_sha256") != canonical_sha256(config):
        raise ValueError("Stage-B config differs from preregistration")
    return file_sha256(ROOT / "PREREGISTRATION_LOCK.json")


def authenticated_cache_evidence(
    config: Mapping[str, Any], lock: Mapping[str, Any]
) -> AuthenticatedCacheClosure:
    """Require the one-time 947-cache proof before any model/data access."""

    private = require_cache_integrity(config, lock, verify_live_stats=True)
    public_path = CACHE_INTEGRITY_PUBLIC_CONTRACT.resolve(strict=True)
    private_path = CACHE_INTEGRITY_PRIVATE_MANIFEST.resolve(strict=True)
    public = json.loads(public_path.read_text(encoding="utf-8"))
    expected_counts = {
        "patient_count": 947,
        "primary_patient_count": 808,
        "train_only_patient_count": 139,
    }
    if (
        not isinstance(public, dict)
        or public.get("schema_version") != 2
        or public.get("status") != "COMPLETE"
    ):
        raise ValueError("public cache-integrity contract is incomplete")
    if private.get("schema_version") != 2 or private.get("status") != "COMPLETE":
        raise ValueError("private cache-integrity contract is incomplete")
    for name, expected in expected_counts.items():
        if public.get(name) != expected or private.get(name) != expected:
            raise ValueError(f"cache-integrity cohort count drifted at {name}")
    if public.get("private_artifact_sha256") != file_sha256(private_path):
        raise ValueError("cache-integrity public/private hash closure drifted")
    if public.get("contains_patient_identifiers") is not False:
        raise ValueError("public cache-integrity contract exposes patient identifiers")
    records = private.get("records")
    if not isinstance(records, list) or len(records) != 947:
        raise ValueError(
            "private cache-integrity record inventory must contain 947 rows"
        )
    patient_ids: list[str] = []
    primary_ids: set[str] = set()
    train_only_ids: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"cache-integrity record {index} is invalid")
        patient_id = str(record.get("patient_id", ""))
        cohort = str(record.get("cohort", ""))
        if not patient_id or cohort not in {"primary", "train_only"}:
            raise ValueError(f"cache-integrity record {index} has invalid membership")
        patient_ids.append(patient_id)
        (primary_ids if cohort == "primary" else train_only_ids).add(patient_id)
    if patient_ids != sorted(patient_ids) or len(set(patient_ids)) != 947:
        raise ValueError("cache-integrity patient inventory is not unique and sorted")
    if (
        len(primary_ids) != 808
        or len(train_only_ids) != 139
        or primary_ids & train_only_ids
    ):
        raise ValueError("cache-integrity primary/train-only partition drifted")
    evidence = {
        "status": "COMPLETE",
        **expected_counts,
        "public_contract_path": str(public_path),
        "public_contract_sha256": file_sha256(public_path),
        "private_manifest_path": str(private_path),
        "private_manifest_sha256": file_sha256(private_path),
        "upstream_manifest_sha256": public["upstream_manifest_sha256"],
        "canonical_record_set_sha256": public["canonical_record_set_sha256"],
        "primary_record_set_sha256": public["primary_record_set_sha256"],
        "train_only_record_set_sha256": public["train_only_record_set_sha256"],
        "preregistration_lock_sha256": public["preregistration_lock_sha256"],
        "implementation_sha256": public["implementation_sha256"],
        "contains_patient_identifiers_in_public_contract": False,
    }
    return AuthenticatedCacheClosure(
        evidence=evidence,
        all_patient_ids=frozenset(patient_ids),
        primary_patient_ids=frozenset(primary_ids),
        train_only_patient_ids=frozenset(train_only_ids),
    )


def _purge_wrong_package(prefix: str, expected_root: Path) -> None:
    loaded = [
        name for name in sys.modules if name == prefix or name.startswith(prefix + ".")
    ]
    wrong = False
    for name in loaded:
        module_path = Path(str(getattr(sys.modules[name], "__file__", ""))).resolve()
        if expected_root.resolve() not in module_path.parents:
            wrong = True
            break
    if wrong:
        for name in loaded:
            sys.modules.pop(name, None)


def configure_canonical_dependencies(config: Mapping[str, Any]) -> SimpleNamespace:
    """Import sealed training/data/model utilities from ``source_repo`` only."""

    source_repo = Path(config["paths"]["source_repo"]).resolve(strict=True)
    stage_b_src = (
        source_repo
        / "additional_experiments"
        / "c1b_overlap_eligibility_ftv_stageb"
        / "src"
    ).resolve(strict=True)
    local_src = (
        source_repo
        / "additional_experiments"
        / "local_global_response_state_pilot"
        / "src"
    ).resolve(strict=True)
    g3_src = (
        source_repo / "additional_experiments" / "g3_multiseed_generalization" / "src"
    ).resolve(strict=True)
    spatial_audit_src = (
        source_repo
        / "additional_experiments"
        / "c1b_spatial_pooling_bottleneck_audit"
        / "src"
    ).resolve(strict=True)
    for prefix, expected in (
        ("c1b_stage_b", stage_b_src),
        ("lg_response_pilot", local_src),
        ("dgrs", g3_src),
        ("c1b_spatial_audit", spatial_audit_src),
    ):
        _purge_wrong_package(prefix, expected)
    for dependency_root in (stage_b_src, local_src):
        value = str(dependency_root)
        while value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)

    inputs = importlib.import_module("c1b_stage_b.inputs")
    data = importlib.import_module("c1b_stage_b.data")
    gate = importlib.import_module("c1b_stage_b.gate")
    targets = importlib.import_module("c1b_stage_b.targets")
    sealed_training = importlib.import_module("c1b_stage_b.training")
    local_model = importlib.import_module("lg_response_pilot.model")
    local_training = importlib.import_module("lg_response_pilot.training")

    expected_modules = {
        inputs: stage_b_src / "c1b_stage_b" / "inputs.py",
        data: stage_b_src / "c1b_stage_b" / "data.py",
        gate: stage_b_src / "c1b_stage_b" / "gate.py",
        targets: stage_b_src / "c1b_stage_b" / "targets.py",
        sealed_training: stage_b_src / "c1b_stage_b" / "training.py",
        local_model: local_src / "lg_response_pilot" / "model.py",
        local_training: local_src / "lg_response_pilot" / "training.py",
    }
    for module, expected in expected_modules.items():
        if Path(str(getattr(module, "__file__", ""))).resolve() != expected.resolve():
            raise ImportError(
                f"canonical dependency resolved outside source_repo: {module}"
            )
    # The completed LOCAL pilot independently hash-checks the sealed Stage-B
    # training inventory before exposing these primitives.
    local_training.verify_sealed_stage_b_sources()
    for function in (
        local_training.logical_patient_batches,
        local_training.run_logical_train_epoch,
        local_training.run_validation_epoch,
    ):
        expected = stage_b_src / "c1b_stage_b" / "training.py"
        if (
            Path(inspect.getfile(inspect.unwrap(function))).resolve()
            != expected.resolve()
        ):
            raise ImportError(
                "training primitive did not resolve to sealed Stage-B source"
            )
    return SimpleNamespace(
        StageBDataPaths=inputs.StageBDataPaths,
        load_stage_b_data=inputs.load_stage_b_data,
        StageBDataset=data.StageBDataset,
        make_splits=data.make_splits,
        require_upstream_stage_a_go=gate.require_stage_a_go,
        fit_grounding_transform=targets.fit_grounding_transform,
        TrainHyperparameters=local_training.TrainHyperparameters,
        logical_patient_batches=local_training.logical_patient_batches,
        run_logical_train_epoch=local_training.run_logical_train_epoch,
        run_validation_epoch=local_training.run_validation_epoch,
        seed_everything=local_training.seed_everything,
        build_baseline_model=local_model.build_model,
        build_objective=local_model.build_objective,
        tensor_state_sha256=local_model.tensor_state_sha256,
        source_paths={
            "stage_b_inputs": expected_modules[inputs],
            "stage_b_data": expected_modules[data],
            "stage_b_targets": expected_modules[targets],
            "stage_b_training": expected_modules[sealed_training],
            "local_model": expected_modules[local_model],
            "local_training": expected_modules[local_training],
        },
    )


def build_dual_statistic_model(
    dependencies: SimpleNamespace, effective_seed: int
) -> tuple[ResponsePhenotypeDualStatisticState, dict[str, Any]]:
    """Build the paired model and prove exact baseline equivalence at init."""

    baseline = dependencies.build_baseline_model("LOCAL3", int(effective_seed))
    baseline.validate_contract() if hasattr(baseline, "validate_contract") else None
    baseline_linear, _ = _baseline_projection_parts(baseline.response_projection)
    baseline_projection_state = {
        name: value.detach().clone()
        for name, value in baseline.response_projection.state_dict().items()
    }
    baseline_target_state = {
        name: value.detach().clone()
        for name, value in baseline.target_response_projection.state_dict().items()
    }
    shared_before = {
        name: dependencies.tensor_state_sha256(getattr(baseline, name).state_dict())
        for name in (
            "encoder",
            "projector",
            "target_encoder",
            "target_projector",
            "transition",
            "ftv_head",
        )
    }
    model = ResponsePhenotypeDualStatisticState(
        baseline, effective_seed=int(effective_seed)
    )
    mean_branch = model.response_projection.mean_projection
    std_branch = model.response_projection.std_projection
    target_mean = model.target_response_projection.mean_projection
    target_std = model.target_response_projection.std_projection
    online_norm_exact = all(
        torch.equal(
            model.response_projection.joint_norm.state_dict()[name],
            baseline_projection_state[f"1.{name}"],
        )
        for name in model.response_projection.joint_norm.state_dict()
    )
    target_norm_exact = all(
        torch.equal(
            model.target_response_projection.joint_norm.state_dict()[name],
            baseline_target_state[f"1.{name}"],
        )
        for name in model.target_response_projection.joint_norm.state_dict()
    )
    baseline_equivalent = all(
        (
            torch.equal(
                mean_branch.weight, baseline_projection_state["0.weight"][:BRANCH_DIM]
            ),
            torch.equal(
                mean_branch.bias, baseline_projection_state["0.bias"][:BRANCH_DIM]
            ),
            torch.equal(
                std_branch.weight, baseline_projection_state["0.weight"][BRANCH_DIM:]
            ),
            torch.equal(
                std_branch.bias, baseline_projection_state["0.bias"][BRANCH_DIM:]
            ),
            torch.equal(
                target_mean.weight, baseline_target_state["0.weight"][:BRANCH_DIM]
            ),
            torch.equal(target_mean.bias, baseline_target_state["0.bias"][:BRANCH_DIM]),
            torch.equal(
                target_std.weight, baseline_target_state["0.weight"][BRANCH_DIM:]
            ),
            torch.equal(target_std.bias, baseline_target_state["0.bias"][BRANCH_DIM:]),
            online_norm_exact,
            target_norm_exact,
        )
    )
    # Functional proof uses no RNG and is stricter than the declared parity
    # tolerance.  Row-splitting a Linear may select a different GEMM kernel, so
    # bitwise identity is not assumed for this numerical check.
    sample = torch.linspace(-1.0, 1.0, steps=2 * FEATURE_CHANNELS).reshape(
        2, FEATURE_CHANNELS
    )
    with torch.no_grad():
        baseline_value = torch.nn.functional.layer_norm(
            torch.nn.functional.linear(
                sample,
                baseline_projection_state["0.weight"],
                baseline_projection_state["0.bias"],
            ),
            (STATE_DIM,),
            baseline_projection_state["1.weight"],
            baseline_projection_state["1.bias"],
        )
        dual_value = model.response_projection(sample, sample)
    functional_equivalence = torch.allclose(
        baseline_value, dual_value, rtol=1e-6, atol=1e-7
    )
    shared_after = {
        name: dependencies.tensor_state_sha256(getattr(model, name).state_dict())
        for name in shared_before
    }
    if not baseline_equivalent or not functional_equivalence:
        raise AssertionError(
            "dual statistic paired initialization is not baseline-equivalent"
        )
    if shared_before != shared_after:
        raise AssertionError(
            "constructing the dual state changed an upstream shared module"
        )
    if baseline_linear.out_features != STATE_DIM:
        raise AssertionError("baseline projection dimension drifted")
    report = {
        "schema_version": 1,
        "effective_seed": int(effective_seed),
        "baseline_arm": "LOCAL3",
        "baseline_response_projection_sha256": dependencies.tensor_state_sha256(
            baseline_projection_state
        ),
        "dual_response_projection_sha256": dependencies.tensor_state_sha256(
            model.response_projection.state_dict()
        ),
        "shared_module_sha256": shared_after,
        "row_split_exact": bool(baseline_equivalent),
        "std_equals_mean_functional_equivalence": bool(functional_equivalence),
        "equivalence_rtol": 1e-6,
        "equivalence_atol": 1e-7,
        "architecture_contract": model.architecture_contract(),
        "parameter_counts": model.parameter_counts(),
    }
    return model, report


def validate_objective(objective: nn.Module) -> None:
    if str(getattr(objective, "model_name", "")) != "G3":
        raise ValueError("Stage B must use the grounded G3 objective")
    if float(getattr(objective, "lambda_ftv", math.nan)) != 0.25:
        raise ValueError("Stage B lambda_FTV must be exactly 0.25")
    if float(getattr(objective, "sigreg_weight", math.nan)) != 0.09:
        raise ValueError("Stage B SIGReg weight must be exactly 0.09")
    observed = objective.step_weights.detach().cpu()
    expected = torch.tensor((2.0, 1.0, 0.5), dtype=torch.float32)
    expected = expected / expected.mean()
    if not torch.equal(observed, expected):
        raise ValueError("Stage B temporal step weights differ from 2/1/0.5")
    if int(getattr(objective.sigreg, "projections", -1)) != 256:
        raise ValueError("Stage B SIGReg projection count drifted")


def select_validation_checkpoint(
    history: Sequence[Mapping[str, Any]], *, min_representation_std: float
) -> dict[str, Any]:
    """Select earliest minimum validation total objective, test-blind."""

    if not history:
        raise ValueError("checkpoint selection requires at least one epoch")
    evidence: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for raw in history:
        row = dict(raw)
        epoch = int(row.get("epoch", -1))
        if epoch <= 0:
            raise ValueError("history epoch must be positive")
        total = float(row.get("val_total_objective", math.nan))
        representation_std = float(row.get("val_representation_std", math.nan))
        finite = (
            bool(row.get("finite", False))
            and math.isfinite(total)
            and math.isfinite(representation_std)
        )
        noncollapsed = finite and representation_std >= float(min_representation_std)
        row["checkpoint_eligible"] = bool(noncollapsed)
        evidence.append(
            {
                name: (
                    None
                    if isinstance(value, (float, np.floating))
                    and not math.isfinite(float(value))
                    else value
                )
                for name, value in row.items()
            }
        )
        if noncollapsed:
            eligible.append(row)
    if not eligible:
        raise RuntimeError("no finite non-collapsed validation checkpoint exists")
    selected = min(
        eligible,
        key=lambda row: (float(row["val_total_objective"]), int(row["epoch"])),
    )
    return {
        "schema_version": 1,
        "selection_rule": (
            "earliest_epoch_minimizing_validation_total_objective_among_"
            "finite_noncollapsed_epochs"
        ),
        "selected_epoch": int(selected["epoch"]),
        "selected_validation_total_objective": float(selected["val_total_objective"]),
        "selected_validation_state_loss": float(selected["val_state_loss"]),
        "selected_validation_ftv_loss": float(selected["val_ftv_loss"]),
        "selected_validation_representation_std": float(
            selected["val_representation_std"]
        ),
        "min_representation_std": float(min_representation_std),
        "test_data_used": False,
        "epochs": evidence,
    }


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".pt", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        temporary.chmod(0o600)
        temporary.replace(path)
        path.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.chmod(0o600)
        temporary.replace(path)
        path.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _private_history(path: Path, history: Sequence[Mapping[str, Any]]) -> None:
    if not history:
        raise ValueError("cannot write an empty training history")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o600)
        temporary.replace(path)
        path.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)


def validate_canonical_stage_b_cohort(
    bundle: Any,
    dependencies: SimpleNamespace,
    cache_closure: AuthenticatedCacheClosure,
) -> None:
    """Bind all canonical fold splits to the authenticated 947-record proof."""

    primary_ids = frozenset(bundle.folds["patient_id"].astype(str))
    train_only_ids = frozenset(str(value) for value in bundle.train_only_ids)
    if primary_ids != cache_closure.primary_patient_ids or len(primary_ids) != 808:
        raise ValueError(
            "canonical Stage-B primary cohort differs from the authenticated 808 caches"
        )
    if (
        train_only_ids != cache_closure.train_only_patient_ids
        or len(train_only_ids) != 139
    ):
        raise ValueError(
            "canonical Stage-B external training cohort differs from the authenticated 139 caches"
        )
    cache_ids = frozenset(str(value) for value in bundle.c1b_cache)
    if not cache_closure.all_patient_ids.issubset(cache_ids):
        raise ValueError(
            "canonical C1B cache mapping does not cover the authenticated 947"
        )
    for fold in FOLDS:
        splits = dependencies.make_splits(bundle.folds, fold, bundle.train_only_ids)
        train_primary = frozenset(str(value) for value in splits.train_primary)
        train_all = tuple(str(value) for value in splits.train_all)
        validation = frozenset(str(value) for value in splits.val)
        test = frozenset(str(value) for value in splits.test)
        if len(train_all) != len(set(train_all)):
            raise ValueError(
                f"canonical Stage-B fold {fold} train_all contains duplicates"
            )
        if not set(train_all).issubset(cache_closure.all_patient_ids):
            raise ValueError(
                f"canonical Stage-B fold {fold} train_all escapes authenticated cache proof"
            )
        if (
            frozenset(splits.train_only) != cache_closure.train_only_patient_ids
            or frozenset(train_all) != train_primary | train_only_ids
        ):
            raise ValueError(
                f"canonical Stage-B fold {fold} does not append the exact 139 train-only patients"
            )
        if (
            not train_primary.issubset(primary_ids)
            or not validation.issubset(primary_ids)
            or not test.issubset(primary_ids)
            or train_primary & validation
            or train_primary & test
            or validation & test
            or train_primary | validation | test != primary_ids
        ):
            raise ValueError(f"canonical Stage-B fold {fold} primary partition drifted")


def _canonical_data_bundle(
    config: Mapping[str, Any],
    authorization: StageBAuthorization,
    cache_closure: AuthenticatedCacheClosure,
    dependencies: SimpleNamespace,
) -> Any:
    if authorization.authorized is not True:
        raise PermissionError("data bundle cannot be opened for unauthorized Stage B")
    upstream_sentinel = Path(config["paths"]["stage_b_upstream_authorization"]).resolve(
        strict=True
    )
    if file_sha256(upstream_sentinel) != str(
        config["paths"]["stage_b_upstream_authorization_sha256"]
    ):
        raise ValueError("canonical Stage-B upstream data authorization hash drifted")
    # This upstream GO authenticates only the already frozen C1B data bundle;
    # the new Stage-A authorization above independently controls this pilot.
    upstream_authorization = dependencies.require_upstream_stage_a_go(upstream_sentinel)
    data_paths = dependencies.StageBDataPaths.load(
        config["paths"]["stage_b_data_contract"],
        config["paths"]["stage_b_data_contract_sha256"],
    )
    bundle = dependencies.load_stage_b_data(
        data_paths, upstream_authorization, verify_cache_files=False
    )
    validate_canonical_stage_b_cohort(bundle, dependencies, cache_closure)
    bundle.provenance.update(
        {
            "stage_b_upstream_authorization_path": str(upstream_sentinel),
            "stage_b_upstream_authorization_sha256": file_sha256(upstream_sentinel),
            "authenticated_cache_patient_count": 947,
            "authenticated_primary_patient_count": 808,
            "authenticated_train_only_patient_count": 139,
        }
    )
    return bundle


def train_fold(
    *,
    config: Mapping[str, Any],
    lock_sha256: str,
    authorization: StageBAuthorization,
    cache_evidence: Mapping[str, Any],
    dependencies: SimpleNamespace,
    data_bundle: Any,
    fold: int,
    device: torch.device,
    output_root: Path,
) -> dict[str, Any]:
    """Train one formal fold without opening test images or any outcome label."""

    if not authorization.authorized:
        raise PermissionError("Stage B training is not authorized")
    if int(fold) not in FOLDS:
        raise ValueError("Stage B fold must be 0..4")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal Stage B training requires CUDA")
    effective_seed = SEED_BASE + int(fold)
    splits = dependencies.make_splits(
        data_bundle.folds, fold, data_bundle.train_only_ids
    )
    transform, transformed_ftv = dependencies.fit_grounding_transform(
        data_bundle.ftv,
        splits.train_primary,
        fold,
        apply_ids=splits.train_primary + splits.val,
    )
    train_dataset = dependencies.StageBDataset(
        splits.train_all, data_bundle.c1b_cache, transformed_ftv
    )
    val_dataset = dependencies.StageBDataset(
        splits.val, data_bundle.c1b_cache, transformed_ftv
    )
    hyperparameters = dependencies.TrainHyperparameters(**dict(FORMAL_HYPERPARAMETERS))
    hyperparameters.validate()
    if asdict(hyperparameters) != dict(FORMAL_HYPERPARAMETERS):
        raise ValueError("formal Stage-B hyperparameters drifted")

    dependencies.seed_everything(effective_seed)
    model, initialization = build_dual_statistic_model(dependencies, effective_seed)
    objective = dependencies.build_objective("LOCAL3")
    validate_objective(objective)
    model.to(device)
    objective.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparameters.learning_rate,
        weight_decay=hyperparameters.weight_decay,
    )

    run_dir = (
        output_root / "checkpoints" / "stage_b" / f"seed_{SEED_BASE}" / f"fold_{fold}"
    )
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite a Stage-B fold: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    run_dir.chmod(0o700)
    data_provenance = {
        **data_bundle.provenance,
        "cache_integrity": dict(cache_evidence),
        "ftv_transform": transform.to_dict(),
        "train_primary_order_sha256": ordered_sha256(splits.train_primary),
        "train_all_order_sha256": ordered_sha256(splits.train_all),
        "validation_order_sha256": ordered_sha256(splits.val),
        "test_images_loaded_during_training": False,
        "test_labels_used": False,
        "model_forward_fields": ["image"],
        "loss_side_fields": ["ftv_target", "ftv_mask"],
        "forbidden_fields": [
            "mask",
            "oracle_region",
            "HR",
            "HER2",
            "pCR",
            "delta",
        ],
    }
    history: list[dict[str, Any]] = []
    best_loss = math.inf
    stale = 0
    started = time.monotonic()
    for epoch in range(1, hyperparameters.epochs + 1):
        logical = dependencies.logical_patient_batches(
            train_dataset.patient_ids, effective_seed, epoch
        )
        train_stats = dependencies.run_logical_train_epoch(
            model,
            objective,
            train_dataset,
            optimizer,
            device,
            logical,
            hyperparameters,
            effective_seed=effective_seed,
            epoch=epoch,
        )
        val_stats = dependencies.run_validation_epoch(
            model,
            objective,
            val_dataset,
            device,
            hyperparameters.physical_batch_size,
            hyperparameters.workers,
        )
        finite_values = (
            train_stats["loss"],
            train_stats["state_loss"],
            train_stats["ftv_loss"],
            train_stats["representation_std"],
            val_stats["loss"],
            val_stats["state_loss"],
            val_stats["ftv_loss"],
            val_stats["representation_std"],
        )
        finite = all(math.isfinite(float(value)) for value in finite_values)
        noncollapsed = finite and float(val_stats["representation_std"]) >= float(
            hyperparameters.min_representation_std
        )
        row = {
            "epoch": int(epoch),
            "seed_base": SEED_BASE,
            "fold": int(fold),
            "effective_seed": effective_seed,
            "train_patient_order_sha256": ordered_sha256(
                patient for batch in logical for patient in batch
            ),
            "train_total_objective": float(train_stats["loss"]),
            "train_base_objective": float(train_stats["base_loss"]),
            "train_state_loss": float(train_stats["state_loss"]),
            "train_ftv_loss": float(train_stats["ftv_loss"]),
            "train_representation_std": float(train_stats["representation_std"]),
            "train_optimizer_steps": int(train_stats["optimizer_steps"]),
            "val_total_objective": float(val_stats["loss"]),
            "val_base_objective": float(val_stats["base_loss"]),
            "val_state_loss": float(val_stats["state_loss"]),
            "val_ftv_loss": float(val_stats["ftv_loss"]),
            "val_grounded_patients": int(val_stats["grounded_patients"]),
            "val_representation_std": float(val_stats["representation_std"]),
            "finite": bool(finite),
            "noncollapsed": bool(noncollapsed),
        }
        history.append(row)
        checkpoint = {
            "schema_version": 1,
            "experiment": "spatial_heterogeneity_phenotype_audit",
            "stage": "B",
            "arm": MODEL_ARM,
            "seed_base": SEED_BASE,
            "fold": int(fold),
            "effective_seed": effective_seed,
            "epoch": int(epoch),
            "state_dict": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_config": model.model_config(),
            "architecture_contract": model.architecture_contract(),
            "parameter_counts": model.parameter_counts(),
            "objective_contract": dict(OBJECTIVE_CONTRACT),
            "hyperparameters": asdict(hyperparameters),
            "paired_initialization": initialization,
            "authorization_sha256": authorization.sha256,
            "stage_a_gates_sha256": authorization.gates_sha256,
            "stage_a_run_summary_sha256": authorization.run_summary_sha256,
            "preregistration_lock_sha256": lock_sha256,
            "cache_integrity_public_contract_sha256": cache_evidence[
                "public_contract_sha256"
            ],
            "cache_integrity_private_manifest_sha256": cache_evidence[
                "private_manifest_sha256"
            ],
            "data_provenance": data_provenance,
            "data_provenance_sha256": canonical_sha256(data_provenance),
            "train_patient_sha256": canonical_sha256(sorted(splits.train_all)),
            "validation_patient_sha256": canonical_sha256(sorted(splits.val)),
            "test_data_used_for_training_or_selection": False,
            "mask_or_oracle_input_used": False,
            "phenotype_pcr_or_delta_supervision_used": False,
            "epoch_metrics": row,
            "selected": False,
        }
        _atomic_torch_save(run_dir / f"epoch_{epoch:02d}.pt", checkpoint)
        _private_history(run_dir / "history.private.csv", history)
        print(
            json.dumps(
                {
                    "stage": "B",
                    "fold": fold,
                    "epoch": epoch,
                    "val_total_objective": row["val_total_objective"],
                    "val_representation_std": row["val_representation_std"],
                    "eligible": noncollapsed,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if noncollapsed and float(val_stats["loss"]) < best_loss:
            best_loss = float(val_stats["loss"])
            stale = 0
        else:
            stale += 1
        if stale >= hyperparameters.patience:
            break

    selection = select_validation_checkpoint(
        history, min_representation_std=hyperparameters.min_representation_std
    )
    selection.update(
        {
            "experiment": "spatial_heterogeneity_phenotype_audit",
            "stage": "B",
            "arm": MODEL_ARM,
            "seed_base": SEED_BASE,
            "fold": int(fold),
            "effective_seed": effective_seed,
            "authorization_sha256": authorization.sha256,
            "stage_a_gates_sha256": authorization.gates_sha256,
            "stage_a_run_summary_sha256": authorization.run_summary_sha256,
            "preregistration_lock_sha256": lock_sha256,
            "cache_integrity_public_contract_sha256": cache_evidence[
                "public_contract_sha256"
            ],
            "cache_integrity_private_manifest_sha256": cache_evidence[
                "private_manifest_sha256"
            ],
            "hyperparameters": asdict(hyperparameters),
            "history_sha256": file_sha256(run_dir / "history.private.csv"),
            "data_provenance_sha256": canonical_sha256(data_provenance),
            "test_data_used": False,
        }
    )
    selection_path = run_dir / "selection.private.json"
    atomic_json(selection, selection_path, private=True)
    selected_epoch = int(selection["selected_epoch"])
    selected = torch.load(
        run_dir / f"epoch_{selected_epoch:02d}.pt",
        map_location="cpu",
        weights_only=True,
    )
    selected["selected"] = True
    selected["selection"] = selection
    selected["selection_path"] = str(selection_path)
    selected["selection_sha256"] = file_sha256(selection_path)
    _atomic_torch_save(run_dir / "selected.pt", selected)
    transform_path = run_dir / "ftv_transform.private.json"
    transform.save(transform_path)
    transform_path.chmod(0o600)
    initialization_path = run_dir / "paired_initialization.private.json"
    atomic_json(initialization, initialization_path, private=True)
    completion = {
        "schema_version": 1,
        "status": "COMPLETE",
        "fold": int(fold),
        "effective_seed": effective_seed,
        "selected_epoch": selected_epoch,
        "selected_checkpoint_sha256": file_sha256(run_dir / "selected.pt"),
        "selection_sha256": file_sha256(selection_path),
        "history_sha256": file_sha256(run_dir / "history.private.csv"),
        "ftv_transform_sha256": file_sha256(transform_path),
        "paired_initialization_sha256": file_sha256(initialization_path),
        "elapsed_seconds": float(time.monotonic() - started),
        "authorization_sha256": authorization.sha256,
        "stage_a_gates_sha256": authorization.gates_sha256,
        "stage_a_run_summary_sha256": authorization.run_summary_sha256,
        "preregistration_lock_sha256": lock_sha256,
        "cache_integrity_public_contract_sha256": cache_evidence[
            "public_contract_sha256"
        ],
        "cache_integrity_private_manifest_sha256": cache_evidence[
            "private_manifest_sha256"
        ],
        "test_data_used": False,
    }
    atomic_json(completion, run_dir / "fold_complete.private.json", private=True)
    return completion


def _validate_selected_checkpoint(
    payload: Mapping[str, Any],
    *,
    fold: int,
    authorization: StageBAuthorization,
    lock_sha256: str,
    cache_evidence: Mapping[str, Any],
) -> None:
    expected = {
        "schema_version": 1,
        "experiment": "spatial_heterogeneity_phenotype_audit",
        "stage": "B",
        "arm": MODEL_ARM,
        "seed_base": SEED_BASE,
        "fold": int(fold),
        "effective_seed": SEED_BASE + int(fold),
        "selected": True,
        "authorization_sha256": authorization.sha256,
        "stage_a_gates_sha256": authorization.gates_sha256,
        "stage_a_run_summary_sha256": authorization.run_summary_sha256,
        "preregistration_lock_sha256": lock_sha256,
        "cache_integrity_public_contract_sha256": cache_evidence[
            "public_contract_sha256"
        ],
        "cache_integrity_private_manifest_sha256": cache_evidence[
            "private_manifest_sha256"
        ],
        "test_data_used_for_training_or_selection": False,
        "mask_or_oracle_input_used": False,
        "phenotype_pcr_or_delta_supervision_used": False,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError(f"selected Stage-B checkpoint differs at {name}")
    selection_path = Path(str(payload.get("selection_path", ""))).resolve(strict=True)
    if payload.get("selection_sha256") != file_sha256(selection_path):
        raise ValueError("selected checkpoint selection record drifted")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if payload.get("selection") != selection:
        raise ValueError("selected checkpoint embeds a different selection")
    if int(payload.get("epoch", -1)) != int(selection.get("selected_epoch", -2)):
        raise ValueError("selected checkpoint epoch differs from selection")
    selection_expected = {
        "authorization_sha256": authorization.sha256,
        "stage_a_gates_sha256": authorization.gates_sha256,
        "stage_a_run_summary_sha256": authorization.run_summary_sha256,
        "preregistration_lock_sha256": lock_sha256,
        "cache_integrity_public_contract_sha256": cache_evidence[
            "public_contract_sha256"
        ],
        "cache_integrity_private_manifest_sha256": cache_evidence[
            "private_manifest_sha256"
        ],
        "test_data_used": False,
    }
    for name, value in selection_expected.items():
        if selection.get(name) != value:
            raise ValueError(f"selected checkpoint selection differs at {name}")
    provenance = payload.get("data_provenance")
    if not isinstance(provenance, Mapping) or payload.get(
        "data_provenance_sha256"
    ) != canonical_sha256(provenance):
        raise ValueError("selected checkpoint data provenance is invalid")
    if provenance.get("cache_integrity") != dict(cache_evidence):
        raise ValueError("selected checkpoint cache-integrity evidence drifted")


def validate_completed_training_fold(
    *,
    fold: int,
    authorization: StageBAuthorization,
    lock_sha256: str,
    cache_evidence: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    """Authenticate an immutable completed fold before reuse or feature export."""

    run_dir = (
        output_root / "checkpoints" / "stage_b" / f"seed_{SEED_BASE}" / f"fold_{fold}"
    )
    completion_path = run_dir / "fold_complete.private.json"
    selected_path = run_dir / "selected.pt"
    selection_path = run_dir / "selection.private.json"
    history_path = run_dir / "history.private.csv"
    transform_path = run_dir / "ftv_transform.private.json"
    initialization_path = run_dir / "paired_initialization.private.json"
    required = (
        completion_path,
        selected_path,
        selection_path,
        history_path,
        transform_path,
        initialization_path,
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise ValueError(
            "partial Stage-B training fold is not safely resumable; missing="
            + ",".join(path.name for path in missing)
        )
    if run_dir.stat().st_mode & 0o077 or any(
        path.stat().st_mode & 0o077 for path in required
    ):
        raise PermissionError("Stage-B training artifacts must remain owner-only")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "status": "COMPLETE",
        "fold": int(fold),
        "effective_seed": SEED_BASE + int(fold),
        "selected_checkpoint_sha256": file_sha256(selected_path),
        "selection_sha256": file_sha256(selection_path),
        "history_sha256": file_sha256(history_path),
        "ftv_transform_sha256": file_sha256(transform_path),
        "paired_initialization_sha256": file_sha256(initialization_path),
        "authorization_sha256": authorization.sha256,
        "stage_a_gates_sha256": authorization.gates_sha256,
        "stage_a_run_summary_sha256": authorization.run_summary_sha256,
        "preregistration_lock_sha256": lock_sha256,
        "cache_integrity_public_contract_sha256": cache_evidence[
            "public_contract_sha256"
        ],
        "cache_integrity_private_manifest_sha256": cache_evidence[
            "private_manifest_sha256"
        ],
        "test_data_used": False,
    }
    for name, value in expected.items():
        if completion.get(name) != value:
            raise ValueError(f"completed Stage-B fold differs at {name}")
    payload = torch.load(selected_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("completed Stage-B selected checkpoint is invalid")
    if (
        Path(str(payload.get("selection_path", ""))).resolve()
        != selection_path.resolve()
    ):
        raise ValueError("selected checkpoint points outside its immutable fold")
    _validate_selected_checkpoint(
        payload,
        fold=fold,
        authorization=authorization,
        lock_sha256=lock_sha256,
        cache_evidence=cache_evidence,
    )
    return completion


@torch.no_grad()
def export_fold_features(
    *,
    dependencies: SimpleNamespace,
    data_bundle: Any,
    fold: int,
    authorization: StageBAuthorization,
    lock_sha256: str,
    cache_evidence: Mapping[str, Any],
    device: torch.device,
    output_root: Path,
) -> dict[str, Any]:
    """Export exact online pre-projector ``[808,4,192]`` states."""

    run_dir = (
        output_root / "checkpoints" / "stage_b" / f"seed_{SEED_BASE}" / f"fold_{fold}"
    )
    checkpoint_path = run_dir / "selected.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("selected Stage-B checkpoint is not a mapping")
    _validate_selected_checkpoint(
        payload,
        fold=fold,
        authorization=authorization,
        lock_sha256=lock_sha256,
        cache_evidence=cache_evidence,
    )
    effective_seed = SEED_BASE + int(fold)
    dependencies.seed_everything(effective_seed)
    model, initialization = build_dual_statistic_model(dependencies, effective_seed)
    if payload.get("paired_initialization") != initialization:
        raise ValueError("selected checkpoint uses a different paired initialization")
    if payload.get("architecture_contract") != model.architecture_contract():
        raise ValueError("selected checkpoint architecture contract drifted")
    expected_weight = model.local_pooling_weight.detach().clone()
    model.load_state_dict(payload["state_dict"], strict=True)
    if not torch.equal(model.local_pooling_weight, expected_weight):
        raise ValueError("checkpoint changed the exact fixed LOCAL weights")
    model.to(device).eval()
    model.validate_contract()

    splits = dependencies.make_splits(
        data_bundle.folds, fold, data_bundle.train_only_ids
    )
    patient_ids = tuple(splits.train_primary + splits.val + splits.test)
    split_labels = (
        ("train",) * len(splits.train_primary)
        + ("val",) * len(splits.val)
        + ("test",) * len(splits.test)
    )
    if len(patient_ids) != 808 or len(set(patient_ids)) != 808:
        raise ValueError("Stage-B export requires exactly 808 unique primary patients")
    dataset = dependencies.StageBDataset(patient_ids, data_bundle.c1b_cache, {})
    loader = DataLoader(
        dataset,
        batch_size=int(FORMAL_HYPERPARAMETERS["physical_batch_size"]),
        shuffle=False,
        drop_last=False,
        num_workers=int(FORMAL_HYPERPARAMETERS["workers"]),
        pin_memory=True,
        persistent_workers=bool(FORMAL_HYPERPARAMETERS["workers"]),
        prefetch_factor=1,
    )
    observed_ids: list[str] = []
    states: list[np.ndarray] = []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        state = model.encode_response(image, None)
        if tuple(state.shape[1:]) != (len(VISITS), STATE_DIM):
            raise ValueError("exported response state must be [B,4,192]")
        if not bool(torch.isfinite(state).all()):
            raise FloatingPointError(
                "exported Stage-B state contains non-finite values"
            )
        observed_ids.extend(str(value) for value in batch["patient_id"])
        states.append(state.detach().float().cpu().numpy())
    if tuple(observed_ids) != patient_ids:
        raise AssertionError("Stage-B export changed patient order")
    response = np.concatenate(states, axis=0).astype(np.float32, copy=False)
    if response.shape != (808, 4, STATE_DIM):
        raise AssertionError("Stage-B response export shape drifted")
    feature_path = (
        output_root
        / "features"
        / "stage_b"
        / f"seed_{SEED_BASE}"
        / f"fold_{fold}"
        / "response_state.private.npz"
    )
    metadata_path = feature_path.with_suffix(".metadata.private.json")
    if feature_path.exists() or metadata_path.exists():
        raise FileExistsError(f"refusing to overwrite Stage-B feature: {feature_path}")
    _atomic_npz(
        feature_path,
        {
            "patient_id": np.asarray(patient_ids, dtype=str),
            "split": np.asarray(split_labels, dtype=str),
            "response_state": response,
            "arm": np.asarray(MODEL_ARM),
            "seed_base": np.asarray(SEED_BASE, dtype=np.int64),
            "fold": np.asarray(int(fold), dtype=np.int64),
            "stage_a_run_summary_sha256": np.asarray(authorization.run_summary_sha256),
        },
    )
    metadata = {
        "schema_version": 1,
        "status": "COMPLETE",
        "experiment": "spatial_heterogeneity_phenotype_audit",
        "stage": "B",
        "seed_base": SEED_BASE,
        "fold": int(fold),
        "effective_seed": effective_seed,
        "feature_path": str(feature_path),
        "feature_sha256": file_sha256(feature_path),
        "feature_shape": list(response.shape),
        "feature_dtype": "float32",
        "feature_tensor": "selected_online_preprojector_dual_statistic_state",
        "patient_order_sha256": ordered_sha256(patient_ids),
        "split_order_sha256": ordered_sha256(split_labels),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "selection_sha256": payload["selection_sha256"],
        "selected_epoch": int(payload["epoch"]),
        "authorization_sha256": authorization.sha256,
        "stage_a_gates_sha256": authorization.gates_sha256,
        "stage_a_run_summary_sha256": authorization.run_summary_sha256,
        "preregistration_lock_sha256": lock_sha256,
        "cache_integrity_public_contract_sha256": cache_evidence[
            "public_contract_sha256"
        ],
        "cache_integrity_private_manifest_sha256": cache_evidence[
            "private_manifest_sha256"
        ],
        "test_labels_used": False,
        "ftv_head_called_during_export": False,
        "phenotype_pcr_or_delta_labels_read_during_export": False,
    }
    atomic_json(metadata, metadata_path, private=True)
    return metadata


def load_exported_feature(
    path: Path,
    *,
    fold: int,
    authorization: StageBAuthorization,
    lock_sha256: str,
    cache_evidence: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    source = path.resolve(strict=True)
    metadata_path = source.with_suffix(".metadata.private.json")
    if source.stat().st_mode & 0o077 or metadata_path.stat().st_mode & 0o077:
        raise PermissionError("Stage-B feature and metadata must remain owner-only")
    if source.parent.stat().st_mode & 0o077:
        raise PermissionError("Stage-B feature directory must remain owner-only")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_checkpoint = (
        ROOT
        / "checkpoints"
        / "stage_b"
        / f"seed_{SEED_BASE}"
        / f"fold_{fold}"
        / "selected.pt"
    ).resolve(strict=True)
    expected_metadata = {
        "schema_version": 1,
        "status": "COMPLETE",
        "seed_base": SEED_BASE,
        "fold": int(fold),
        "effective_seed": SEED_BASE + int(fold),
        "feature_sha256": file_sha256(source),
        "feature_shape": [808, 4, STATE_DIM],
        "feature_dtype": "float32",
        "authorization_sha256": authorization.sha256,
        "stage_a_gates_sha256": authorization.gates_sha256,
        "stage_a_run_summary_sha256": authorization.run_summary_sha256,
        "preregistration_lock_sha256": lock_sha256,
        "cache_integrity_public_contract_sha256": cache_evidence[
            "public_contract_sha256"
        ],
        "cache_integrity_private_manifest_sha256": cache_evidence[
            "private_manifest_sha256"
        ],
        "test_labels_used": False,
        "checkpoint_path": str(expected_checkpoint),
        "checkpoint_sha256": file_sha256(expected_checkpoint),
    }
    for name, value in expected_metadata.items():
        if metadata.get(name) != value:
            raise ValueError(f"Stage-B feature metadata differs at {name}")
    checkpoint = torch.load(expected_checkpoint, map_location="cpu", weights_only=True)
    if metadata.get("selection_sha256") != checkpoint.get("selection_sha256"):
        raise ValueError("Stage-B feature selection hash differs from checkpoint")
    with np.load(source, allow_pickle=False) as archive:
        required = {
            "patient_id",
            "split",
            "response_state",
            "arm",
            "seed_base",
            "fold",
            "stage_a_run_summary_sha256",
        }
        if set(archive.files) != required:
            raise ValueError("Stage-B feature NPZ schema drifted")
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    if (
        arrays["patient_id"].shape != (808,)
        or len(set(arrays["patient_id"].astype(str))) != 808
    ):
        raise ValueError("Stage-B feature patient identity contract drifted")
    if arrays["split"].shape != (808,) or set(arrays["split"].astype(str)) != {
        "train",
        "val",
        "test",
    }:
        raise ValueError("Stage-B feature split contract drifted")
    state = arrays["response_state"]
    if state.dtype != np.dtype(np.float32) or state.shape != (808, 4, STATE_DIM):
        raise ValueError("Stage-B feature state must be float32 [808,4,192]")
    if not np.isfinite(state).all():
        raise ValueError("Stage-B feature state contains non-finite values")
    if str(np.asarray(arrays["arm"]).item()) != MODEL_ARM:
        raise ValueError("Stage-B feature arm identity drifted")
    if int(np.asarray(arrays["seed_base"]).item()) != SEED_BASE or int(
        np.asarray(arrays["fold"]).item()
    ) != int(fold):
        raise ValueError("Stage-B feature seed/fold identity drifted")
    if (
        str(np.asarray(arrays["stage_a_run_summary_sha256"]).item())
        != authorization.run_summary_sha256
    ):
        raise ValueError("Stage-B feature run-summary binding drifted")
    if metadata.get("patient_order_sha256") != ordered_sha256(
        arrays["patient_id"].astype(str)
    ) or metadata.get("split_order_sha256") != ordered_sha256(
        arrays["split"].astype(str)
    ):
        raise ValueError("Stage-B feature order hash drifted")
    return arrays


def ensure_fold_artifacts(
    *,
    config: Mapping[str, Any],
    lock_sha256: str,
    authorization: StageBAuthorization,
    cache_evidence: Mapping[str, Any],
    dependencies: SimpleNamespace,
    data_bundle: Any,
    fold: int,
    device: torch.device,
    output_root: Path,
    allow_training: bool,
) -> dict[str, Any]:
    """Create or authenticate one fold without mutating completed artifacts."""

    run_dir = (
        output_root / "checkpoints" / "stage_b" / f"seed_{SEED_BASE}" / f"fold_{fold}"
    )
    feature_path = (
        output_root
        / "features"
        / "stage_b"
        / f"seed_{SEED_BASE}"
        / f"fold_{fold}"
        / "response_state.private.npz"
    )
    metadata_path = feature_path.with_suffix(".metadata.private.json")
    training_exists = run_dir.exists() and any(run_dir.iterdir())
    feature_exists = feature_path.exists() or metadata_path.exists()
    if feature_exists and not training_exists:
        raise ValueError(
            "Stage-B feature exists without its authenticated training fold"
        )

    if training_exists:
        completion = validate_completed_training_fold(
            fold=fold,
            authorization=authorization,
            lock_sha256=lock_sha256,
            cache_evidence=cache_evidence,
            output_root=output_root,
        )
    else:
        if not allow_training:
            raise FileNotFoundError(
                f"--finalize-only requires a completed Stage-B fold {fold}"
            )
        completion = train_fold(
            config=config,
            lock_sha256=lock_sha256,
            authorization=authorization,
            cache_evidence=cache_evidence,
            dependencies=dependencies,
            data_bundle=data_bundle,
            fold=fold,
            device=device,
            output_root=output_root,
        )
        # Authenticate the just-published closure before any test-image export.
        completion = validate_completed_training_fold(
            fold=fold,
            authorization=authorization,
            lock_sha256=lock_sha256,
            cache_evidence=cache_evidence,
            output_root=output_root,
        )

    if feature_path.exists() and metadata_path.exists():
        load_exported_feature(
            feature_path,
            fold=fold,
            authorization=authorization,
            lock_sha256=lock_sha256,
            cache_evidence=cache_evidence,
        )
        feature = json.loads(metadata_path.read_text(encoding="utf-8"))
    elif feature_path.exists() or metadata_path.exists():
        raise ValueError("partial Stage-B feature pair is not safely resumable")
    else:
        feature = export_fold_features(
            dependencies=dependencies,
            data_bundle=data_bundle,
            fold=fold,
            authorization=authorization,
            lock_sha256=lock_sha256,
            cache_evidence=cache_evidence,
            device=device,
            output_root=output_root,
        )
        load_exported_feature(
            feature_path,
            fold=fold,
            authorization=authorization,
            lock_sha256=lock_sha256,
            cache_evidence=cache_evidence,
        )
    return {"training": completion, "feature": feature, "reused": training_exists}


def load_authenticated_stage_a_baselines(
    authorization: StageBAuthorization,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Load Table 2/3 only through the immutable Stage-A run-summary hashes."""

    if file_sha256(authorization.run_summary_path) != authorization.run_summary_sha256:
        raise ValueError("Stage-A run summary changed after Stage-B authorization")
    try:
        summary = json.loads(authorization.run_summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "Stage-A run summary is unreadable during probe pairing"
        ) from error
    if (
        not isinstance(summary, Mapping)
        or summary.get("stage") != "A"
        or summary.get("status") != "COMPLETE"
    ):
        raise ValueError("Stage-A run summary is no longer a complete Stage-A closure")
    table2_path = _authenticated_summary_artifact(
        summary, authorization.run_summary_path, "table2"
    )
    table3_path = _authenticated_summary_artifact(
        summary, authorization.run_summary_path, "table3"
    )
    expected_root = _run_summary_root(authorization.run_summary_path)
    expected_paths = {
        "table2": expected_root / "metrics" / "table2_phenotype_probes.csv",
        "table3": expected_root / "metrics" / "table3_mri_only_pcr.csv",
    }
    if (
        table2_path != expected_paths["table2"].resolve()
        or table3_path != expected_paths["table3"].resolve()
    ):
        raise ValueError("Stage-A baseline tables are not the canonical Table 2/3")
    try:
        table2 = pd.read_csv(table2_path)
        table3 = pd.read_csv(table3_path)
    except (OSError, UnicodeError, pd.errors.ParserError) as error:
        raise ValueError(
            "authenticated Stage-A baseline tables are unreadable"
        ) from error
    return (
        table2,
        table3,
        {
            "table2_sha256": file_sha256(table2_path),
            "table3_sha256": file_sha256(table3_path),
        },
    )


def pair_stage_b_with_stage_a_baseline(
    stage_b_metrics: pd.DataFrame,
    table2: pd.DataFrame,
    table3: pd.DataFrame,
) -> pd.DataFrame:
    """Pair all 20 dual-state rows one-to-one with seed-2026 LOCAL3 P1."""

    identity = ["view", "target", "population"]
    metric_names = ["auroc", "auprc", "balanced_accuracy", "brier"]
    required = {
        "seed",
        "arm",
        "view",
        "target",
        "variant",
        "population",
        "n",
        *metric_names,
    }
    for name, frame in (
        ("Stage-B metrics", stage_b_metrics),
        ("Stage-A Table 2", table2),
        ("Stage-A Table 3", table3),
    ):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} misses paired fields: {sorted(missing)}")
    expected_phenotype = {
        (view, target, "full_808")
        for view in ("T0", "T1", "T2", "T3")
        for target in ("HR", "HER2", "subtype_4class")
    }
    expected_pcr = {
        (view, "pCR", population)
        for view in ("T0", "T0-T1", "T0-T2", "T0-T3")
        for population in ("full_808", "ftv_complete_375")
    }
    expected = expected_phenotype | expected_pcr
    if (
        len(stage_b_metrics) != 20
        or set(stage_b_metrics.loc[:, identity].itertuples(index=False, name=None))
        != expected
    ):
        raise ValueError("Stage-B rows do not form the exact 20-row pairing grid")
    if stage_b_metrics.duplicated(identity).any():
        raise ValueError("Stage-B rows are not unique on view/target/population")
    if (
        not pd.to_numeric(stage_b_metrics["seed"], errors="coerce").eq(SEED_BASE).all()
        or not stage_b_metrics["arm"].astype(str).eq(MODEL_ARM).all()
        or not stage_b_metrics["variant"].astype(str).eq(FEATURE_VARIANT).all()
    ):
        raise ValueError("Stage-B rows differ from the registered seed/arm/variant")

    def baseline_rows(
        frame: pd.DataFrame, expected_rows: set[tuple[str, str, str]]
    ) -> pd.DataFrame:
        seed = pd.to_numeric(frame["seed"], errors="coerce")
        selected = frame.loc[
            seed.eq(SEED_BASE)
            & frame["arm"].astype(str).eq("LOCAL3")
            & frame["variant"].astype(str).eq("P1")
        ].copy()
        selected = selected.loc[
            selected.loc[:, identity].apply(tuple, axis=1).isin(expected_rows)
        ]
        return selected

    baseline = pd.concat(
        (
            baseline_rows(table2, expected_phenotype),
            baseline_rows(table3, expected_pcr),
        ),
        ignore_index=True,
    )
    baseline_identities = set(
        baseline.loc[:, identity].itertuples(index=False, name=None)
    )
    if (
        len(baseline) != 20
        or baseline.duplicated(identity).any()
        or baseline_identities != expected
    ):
        raise ValueError(
            "Stage-A seed-2026 LOCAL3 P1 baseline does not form the exact paired grid"
        )
    baseline = baseline.loc[:, [*identity, "n", *metric_names]].rename(
        columns={
            "n": "baseline_n",
            **{name: f"baseline_{name}" for name in metric_names},
        }
    )
    try:
        paired = stage_b_metrics.merge(
            baseline,
            on=identity,
            how="left",
            validate="one_to_one",
            sort=False,
        )
    except pd.errors.MergeError as error:
        raise ValueError(
            "Stage-A/Stage-B baseline pairing is not one-to-one"
        ) from error
    if (
        not pd.to_numeric(paired["n"], errors="coerce")
        .eq(pd.to_numeric(paired["baseline_n"], errors="coerce"))
        .all()
    ):
        raise ValueError("Stage-A/Stage-B paired population n differs")
    for metric in ("auroc", "auprc", "balanced_accuracy"):
        dual = pd.to_numeric(paired[metric], errors="coerce").to_numpy(dtype=float)
        reference = pd.to_numeric(
            paired[f"baseline_{metric}"], errors="coerce"
        ).to_numpy(dtype=float)
        if not np.isfinite(dual).all() or not np.isfinite(reference).all():
            raise ValueError(f"paired {metric} values must all be finite")
        paired[f"delta_{metric}"] = dual - reference
    pcr = paired["target"].astype(str).eq("pCR").to_numpy()
    dual_brier = pd.to_numeric(paired["brier"], errors="coerce").to_numpy(dtype=float)
    baseline_brier = pd.to_numeric(paired["baseline_brier"], errors="coerce").to_numpy(
        dtype=float
    )
    if (
        not np.isfinite(dual_brier[pcr]).all()
        or not np.isfinite(baseline_brier[pcr]).all()
    ):
        raise ValueError("paired pCR Brier values must be finite")
    if np.isfinite(dual_brier[~pcr]).any() or np.isfinite(baseline_brier[~pcr]).any():
        raise ValueError("phenotype rows must not carry Brier values")
    paired["brier_improvement"] = np.where(pcr, baseline_brier - dual_brier, np.nan)
    paired["stage_a_baseline_seed"] = SEED_BASE
    paired["stage_a_baseline_arm"] = "LOCAL3"
    paired["stage_a_baseline_variant"] = "P1"
    return paired.drop(columns=["baseline_n"])


def run_stage_b_probes(
    *,
    config: Mapping[str, Any],
    authorization: StageBAuthorization,
    lock_sha256: str,
    cache_evidence: Mapping[str, Any],
    output_root: Path,
) -> pd.DataFrame:
    """Repeat fold-isolated phenotype and pCR probes on selected states."""

    complementarity = (
        ROOT.parent / "mri_clinical_complementarity_audit" / "scripts"
    ).resolve(strict=True)
    for dependency_root in (SCRIPTS_ROOT, complementarity):
        value = str(dependency_root)
        while value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)
    audit = importlib.import_module("run_audit")
    data_contracts = importlib.import_module("data_contracts")
    expected_audit = (SCRIPTS_ROOT / "run_audit.py").resolve()
    if Path(str(getattr(audit, "__file__", ""))).resolve() != expected_audit:
        raise ImportError("Stage-B probes did not resolve to this audit's run_audit.py")
    expected_contracts = (complementarity / "data_contracts.py").resolve()
    if (
        Path(str(getattr(data_contracts, "__file__", ""))).resolve()
        != expected_contracts
    ):
        raise ImportError("Stage-B probes did not resolve to locked data_contracts.py")
    upstream = config["upstream_code"]
    if (
        file_sha256(expected_contracts)
        != upstream["complementarity_data_contracts_sha256"]
    ):
        raise ValueError("Stage-B clinical adapter hash drifted")
    modeling_path = complementarity / "modeling.py"
    if file_sha256(modeling_path) != upstream["complementarity_modeling_sha256"]:
        raise ValueError("Stage-B probe modeling hash drifted")

    folds = data_contracts.load_fold_manifest(
        config["paths"]["fold_manifest"], config["paths"]["fold_manifest_sha256"]
    )
    clinical = data_contracts.load_clinical_table(
        config["paths"]["clinical_labels"],
        config["paths"]["clinical_labels_sha256"],
        folds,
    )
    ftv = data_contracts.load_ftv_wide(
        config["paths"]["ftv_table"], config["paths"]["ftv_table_sha256"], folds
    )
    ftv_ids = set(ftv["patient_id"].astype(str))
    if len(ftv_ids) != 375:
        raise ValueError(
            "Stage-B matched pCR population must contain exactly 375 patients"
        )

    prediction_rows: list[dict[str, Any]] = []
    hyperparameter_rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        feature_path = (
            output_root
            / "features"
            / "stage_b"
            / f"seed_{SEED_BASE}"
            / f"fold_{fold}"
            / "response_state.private.npz"
        )
        arrays = load_exported_feature(
            feature_path,
            fold=fold,
            authorization=authorization,
            lock_sha256=lock_sha256,
            cache_evidence=cache_evidence,
        )
        patient_ids = arrays["patient_id"].astype(str)
        split = arrays["split"].astype(str)
        state = arrays["response_state"]
        aligned = audit._aligned_clinical(clinical, patient_ids)
        indices = audit._split_indices(split)
        for view in config["analysis"]["phenotype_views"]:
            matrix = audit.static_visit(state, str(view))
            for target in audit.PHENOTYPE_TARGETS:
                labels = audit._phenotype_labels(aligned, target)
                metadata = {
                    "analysis": "stage_b_phenotype",
                    "population": "full_808",
                    "seed": SEED_BASE,
                    "arm": MODEL_ARM,
                    "view": str(view),
                    "target": target,
                    "variant": FEATURE_VARIANT,
                    "clinical_contract": "",
                }
                if target == "subtype_4class":
                    audit._append_multiclass_fit(
                        prediction_rows,
                        hyperparameter_rows,
                        patient_ids=patient_ids,
                        fold=fold,
                        labels=labels,
                        matrix=matrix,
                        indices=indices,
                        config=config,
                        grid=tuple(
                            float(value) for value in config["logistic"]["c_grid"]
                        ),
                        metadata=metadata,
                    )
                else:
                    audit._append_binary_fit(
                        prediction_rows,
                        hyperparameter_rows,
                        patient_ids=patient_ids,
                        fold=fold,
                        labels=labels,
                        matrix=matrix,
                        indices=indices,
                        config=config,
                        grid=tuple(
                            float(value) for value in config["logistic"]["c_grid"]
                        ),
                        class_weight=str(config["logistic"]["phenotype_class_weight"]),
                        metadata=metadata,
                    )
        for view in config["analysis"]["pcr_timings"]:
            full_matrix = audit.causal_prefix(state, str(view))
            for population in config["analysis"]["pcr_populations"]:
                mask = (
                    np.ones(808, dtype=bool)
                    if population == "full_808"
                    else np.asarray(
                        [patient_id in ftv_ids for patient_id in patient_ids],
                        dtype=bool,
                    )
                )
                selected_ids = patient_ids[mask]
                selected_split = split[mask]
                selected_matrix = full_matrix[mask]
                selected_clinical = aligned.loc[mask].reset_index(drop=True)
                selected_indices = audit._split_indices(selected_split)
                labels = selected_clinical["label_pcr"].to_numpy(dtype=np.int64)
                metadata = {
                    "analysis": "stage_b_mri_only_pcr",
                    "population": str(population),
                    "seed": SEED_BASE,
                    "arm": MODEL_ARM,
                    "view": str(view),
                    "target": "pCR",
                    "variant": FEATURE_VARIANT,
                    "clinical_contract": "",
                }
                audit._append_binary_fit(
                    prediction_rows,
                    hyperparameter_rows,
                    patient_ids=selected_ids,
                    fold=fold,
                    labels=labels,
                    matrix=selected_matrix,
                    indices=selected_indices,
                    config=config,
                    grid=tuple(float(value) for value in config["logistic"]["c_grid"]),
                    class_weight=None,
                    metadata=metadata,
                )
    predictions = audit._prediction_frame(prediction_rows)
    metrics = audit.aggregate_oof(predictions)
    if len(metrics) != 20:
        raise ValueError(
            f"Stage-B aggregate probe coverage is {len(metrics)}, expected 20"
        )
    phenotype_metrics = metrics.loc[metrics["target"].ne("pCR")]
    pcr_metrics = metrics.loc[metrics["target"].eq("pCR")]
    if len(phenotype_metrics) != 12 or len(pcr_metrics) != 8:
        raise ValueError("Stage-B phenotype/pCR metric matrix is incomplete")
    if not phenotype_metrics["n"].eq(808).all():
        raise ValueError("Stage-B phenotype OOF coverage must be 808")
    expected_pcr_n = pcr_metrics["population"].map(
        {"full_808": 808, "ftv_complete_375": 375}
    )
    if expected_pcr_n.isna().any() or not pcr_metrics["n"].eq(expected_pcr_n).all():
        raise ValueError("Stage-B pCR OOF coverage drifted")
    table2, table3, stage_a_table_hashes = load_authenticated_stage_a_baselines(
        authorization
    )
    table = metrics.copy()
    table.insert(
        0,
        "analysis",
        np.where(table["target"].eq("pCR"), "mri_only_pcr", "phenotype"),
    )
    table = pair_stage_b_with_stage_a_baseline(table, table2, table3)

    predictions_path = output_root / "predictions" / "stage_b_oof.private.csv"
    hyperparameters_path = (
        output_root / "predictions" / "stage_b_probe_hyperparameters.private.csv"
    )
    provenance_path = output_root / "predictions" / "stage_b_provenance.private.json"
    table_path = output_root / "metrics" / "table8_stage_b.csv"
    for path in (predictions_path, hyperparameters_path, provenance_path, table_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite Stage-B output: {path}")
    atomic_csv(predictions, predictions_path, private=True)
    hyperparameters = pd.DataFrame(hyperparameter_rows).reindex(
        columns=audit.HYPERPARAMETER_COLUMNS
    )
    if len(hyperparameters) != 100:
        raise ValueError("Stage-B probe selection matrix must contain 100 fold fits")
    atomic_csv(hyperparameters, hyperparameters_path, private=True)
    table.insert(0, "stage", "B")
    table.insert(0, "status", "COMPLETE")
    table = table.reindex(columns=TABLE8_COLUMNS)
    if "patient_id" in table.columns:
        raise ValueError("public Stage-B table contains patient identity")
    atomic_csv(table, table_path)
    provenance = {
        "schema_version": 1,
        "status": "COMPLETE",
        "experiment": "spatial_heterogeneity_phenotype_audit",
        "stage": "B",
        "authorization_sha256": authorization.sha256,
        "stage_a_gates_sha256": authorization.gates_sha256,
        "stage_a_run_summary_sha256": authorization.run_summary_sha256,
        "preregistration_lock_sha256": lock_sha256,
        "cache_integrity": dict(cache_evidence),
        "upstream_data_authorization": {
            "path": str(config["paths"]["stage_b_upstream_authorization"]),
            "sha256": str(config["paths"]["stage_b_upstream_authorization_sha256"]),
        },
        "stage_a_paired_baseline": {
            "seed": SEED_BASE,
            "arm": "LOCAL3",
            "variant": "P1",
            **stage_a_table_hashes,
        },
        "seed_base": SEED_BASE,
        "folds": list(FOLDS),
        "feature_shape_per_fold": [808, 4, STATE_DIM],
        "probe_rows": int(len(hyperparameters)),
        "oof_prediction_rows": int(len(predictions)),
        "public_metric_rows": int(len(table)),
        "training_supervision": ["JEPA", "FTV"],
        "phenotype_pcr_labels_used_only_after_checkpoint_selection_and_feature_export": True,
        "private_artifacts": {
            "predictions": {
                "path": str(predictions_path),
                "sha256": file_sha256(predictions_path),
            },
            "hyperparameters": {
                "path": str(hyperparameters_path),
                "sha256": file_sha256(hyperparameters_path),
            },
        },
        "public_table": {
            "path": str(table_path),
            "sha256": file_sha256(table_path),
        },
        "implementation_sha256": {
            "stage_b_pilot": file_sha256(Path(__file__)),
            "run_audit": file_sha256(expected_audit),
            "data_contracts": file_sha256(expected_contracts),
            "modeling": file_sha256(modeling_path),
        },
        "fold_artifacts": {
            str(fold): {
                "checkpoint_sha256": file_sha256(
                    output_root
                    / "checkpoints"
                    / "stage_b"
                    / f"seed_{SEED_BASE}"
                    / f"fold_{fold}"
                    / "selected.pt"
                ),
                "feature_sha256": file_sha256(
                    output_root
                    / "features"
                    / "stage_b"
                    / f"seed_{SEED_BASE}"
                    / f"fold_{fold}"
                    / "response_state.private.npz"
                ),
            }
            for fold in FOLDS
        },
    }
    atomic_json(
        provenance,
        provenance_path,
        private=True,
    )
    return table


def unauthorized_table() -> pd.DataFrame:
    row: dict[str, Any] = {column: math.nan for column in TABLE8_COLUMNS}
    row.update(
        {
            "status": "NOT_RUN_NOT_AUTHORIZED",
            "stage": "B",
            "seed": SEED_BASE,
            "arm": MODEL_ARM,
            "analysis": "conditional_stage_b",
            "variant": FEATURE_VARIANT,
            "population": "NOT_APPLICABLE",
        }
    )
    return pd.DataFrame([row]).reindex(columns=TABLE8_COLUMNS)


def preflight_payload(
    authorization: StageBAuthorization,
    lock_sha256: str,
    config: Mapping[str, Any],
    cache_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": "spatial_heterogeneity_phenotype_audit",
        "stage": "B",
        "status": "AUTHORIZED" if authorization.authorized else "NOT_AUTHORIZED",
        "authorized": authorization.authorized,
        "gate_a_passed": authorization.gate_a_passed,
        "gate_c_passed": authorization.gate_c_passed,
        "authorization_sha256": authorization.sha256,
        "stage_a_gates_sha256": authorization.gates_sha256,
        "stage_a_run_summary_sha256": authorization.run_summary_sha256,
        "preregistration_lock_sha256": lock_sha256,
        "formal_seed_bases": list(config["stage_b"]["seed_bases"]),
        "formal_folds": list(config["stage_b"]["folds"]),
        "effective_seeds": [SEED_BASE + fold for fold in FOLDS],
        "hyperparameters": dict(FORMAL_HYPERPARAMETERS),
        "objective": dict(OBJECTIVE_CONTRACT),
        "cache_integrity": (
            None
            if cache_evidence is None
            else {
                "status": cache_evidence["status"],
                "public_contract_sha256": cache_evidence["public_contract_sha256"],
                "private_manifest_sha256": cache_evidence["private_manifest_sha256"],
            }
        ),
        "model_or_data_imported": False,
        "training_performed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fold", type=int, choices=FOLDS)
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.umask(0o077)
    if args.fold is not None and args.finalize_only:
        raise ValueError("--fold and --finalize-only are mutually exclusive")
    # Authorization must close before any configured data/cache artifact is
    # opened.  The unverified load only parses and resolves the hash-bound
    # paths; the preregistration lock authenticates these exact config bytes.
    config = load_config(ROOT / "configs" / "audit.json", verify_inputs=False)
    validate_stage_b_config(config)
    lock = require_preregistration_lock(config)
    lock_sha256 = require_stage_b_preregistration(config, lock)
    authorization = validate_stage_b_authorization(
        config,
        ROOT / "metrics" / "stage_b_authorization.json",
        ROOT / "metrics" / "gates.json",
    )
    cache_closure = (
        authenticated_cache_evidence(config, lock) if authorization.authorized else None
    )
    if authorization.authorized:
        # Only an authorized execution may touch the remaining configured
        # inputs.  Re-load with full hash verification after the 947-cache
        # proof, and reject any race or parsing discrepancy.
        verified_config = load_config(
            ROOT / "configs" / "audit.json", verify_inputs=True
        )
        if canonical_sha256(verified_config) != canonical_sha256(config):
            raise RuntimeError("Stage-B config changed during authorization")
        config = verified_config
    cache_evidence = None if cache_closure is None else cache_closure.evidence
    if args.preflight:
        print(
            json.dumps(
                preflight_payload(authorization, lock_sha256, config, cache_evidence),
                indent=2,
                sort_keys=True,
            )
        )
        return

    output_root = ROOT
    table_path = output_root / "metrics" / "table8_stage_b.csv"
    if not authorization.authorized:
        if args.fold is not None or args.finalize_only:
            raise PermissionError(
                "Stage B is not authorized; worker/finalize mode is forbidden"
            )
        if table_path.exists():
            raise FileExistsError(f"refusing to overwrite Stage-B status: {table_path}")
        atomic_csv(unauthorized_table(), table_path)
        print(
            json.dumps(
                {
                    "status": "NOT_RUN_NOT_AUTHORIZED",
                    "table8": str(table_path),
                    "training_performed": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.fold is None:
        final_artifacts = (
            table_path,
            output_root / "predictions" / "stage_b_oof.private.csv",
            output_root / "predictions" / "stage_b_probe_hyperparameters.private.csv",
            output_root / "predictions" / "stage_b_provenance.private.json",
        )
        existing_final = [path for path in final_artifacts if path.exists()]
        if existing_final:
            raise FileExistsError(
                "completed/partial Stage-B final outputs are immutable: "
                + ", ".join(str(path) for path in existing_final)
            )

    # Canonical model/data imports occur only after the new gate has authorized
    # execution.  This keeps the unauthorized path structurally training-free.
    if cache_evidence is None:
        raise AssertionError("authorized Stage B lacks cache-integrity evidence")
    if cache_closure is None:
        raise AssertionError("authorized Stage B lacks authenticated cache membership")
    for private_name in ("checkpoints", "features", "predictions", "logs", "manifests"):
        private_directory(output_root / private_name)
    dependencies = configure_canonical_dependencies(config)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("authorized formal Stage B requires a CUDA device")
    data_bundle = _canonical_data_bundle(
        config, authorization, cache_closure, dependencies
    )

    if args.fold is not None:
        result = ensure_fold_artifacts(
            config=config,
            lock_sha256=lock_sha256,
            authorization=authorization,
            cache_evidence=cache_evidence,
            dependencies=dependencies,
            data_bundle=data_bundle,
            fold=args.fold,
            device=device,
            output_root=output_root,
            allow_training=True,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    for fold in FOLDS:
        ensure_fold_artifacts(
            config=config,
            lock_sha256=lock_sha256,
            authorization=authorization,
            cache_evidence=cache_evidence,
            dependencies=dependencies,
            data_bundle=data_bundle,
            fold=fold,
            device=device,
            output_root=output_root,
            allow_training=not args.finalize_only,
        )
    table = run_stage_b_probes(
        config=config,
        authorization=authorization,
        lock_sha256=lock_sha256,
        cache_evidence=cache_evidence,
        output_root=output_root,
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "table8": str(table_path),
                "rows": int(len(table)),
                "seed_base": SEED_BASE,
                "folds": list(FOLDS),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

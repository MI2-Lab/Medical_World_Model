"""Logical-B32 training and pCR-blind selection for the factorized state."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import csv
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

from .contracts import PCR_LABEL_ACCESS, canonical_sha256
from .losses import FactorizedObjective, logical_loss_surrogate
from .model import FactorizedPhenotypeWorldModel
from .stageb import logical_patient_batches, physical_patient_batches


LOGICAL_BATCH_SIZE = 32
REFERENCE_KEYS = (
    "response_state",
    "phenotype_state",
    "augmented_phenotype_state",
    "response_online",
)


@dataclass(frozen=True)
class TrainHyperparameters:
    physical_batch_size: int = 4
    accumulation_steps: int = 8
    workers: int = 2
    epochs: int = 12
    patience: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 1e-4
    ema_momentum: float = 0.996
    max_grad_norm: float = 5.0
    min_response_std: float = 0.05
    min_phenotype_std: float = 0.05
    min_phenotype_effective_rank: float = 10.0
    min_augmentation_cosine: float = 0.5
    augmentation_scale_half_width: float = 0.05
    augmentation_shift_half_width: float = 0.05
    augmentation_noise_std: float = 0.02

    def validate(self) -> None:
        if (self.physical_batch_size, self.accumulation_steps) != (4, 8):
            raise ValueError("formal Goal F training requires physical=4, accumulation=8")
        if self.physical_batch_size * self.accumulation_steps != LOGICAL_BATCH_SIZE:
            raise ValueError("formal Goal F logical batch must be 32")
        if self.workers < 0 or self.epochs <= 0 or self.patience <= 0:
            raise ValueError("workers must be nonnegative and epochs/patience positive")
        positive = (
            self.learning_rate,
            self.ema_momentum,
            self.max_grad_norm,
            self.min_response_std,
            self.min_phenotype_std,
            self.min_phenotype_effective_rank,
            self.min_augmentation_cosine,
        )
        if any(not math.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("formal positive hyperparameters must be finite")
        if self.weight_decay < 0 or not 0.0 < self.ema_momentum < 1.0:
            raise ValueError("optimizer/EMA hyperparameters are invalid")
        if not 0.0 <= self.min_augmentation_cosine <= 1.0:
            raise ValueError("augmentation cosine floor must lie in [0,1]")
        for value in (
            self.augmentation_scale_half_width,
            self.augmentation_shift_half_width,
            self.augmentation_noise_std,
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError("augmentation magnitudes must be finite and nonnegative")


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def weak_photometric_view(
    image: torch.Tensor,
    *,
    seed: int,
    scale_half_width: float = 0.05,
    shift_half_width: float = 0.05,
    noise_std: float = 0.02,
) -> torch.Tensor:
    """Deterministic weak channel-wise intensity augmentation; no geometry."""

    if image.ndim != 6 or image.size(2) != 7:
        raise ValueError("photometric augmentation expects [B,V,7,Z,Y,X]")
    if not image.dtype.is_floating_point or not bool(torch.isfinite(image).all()):
        raise ValueError("photometric augmentation expects finite floating image")
    generator = torch.Generator(device=image.device)
    generator.manual_seed(int(seed))
    parameter_shape = (*image.shape[:3], 1, 1, 1)
    scale = 1.0 + (2.0 * torch.rand(
        parameter_shape, generator=generator, device=image.device, dtype=image.dtype
    ) - 1.0) * float(scale_half_width)
    shift = (2.0 * torch.rand(
        parameter_shape, generator=generator, device=image.device, dtype=image.dtype
    ) - 1.0) * float(shift_half_width)
    noise = torch.randn(
        image.shape, generator=generator, device=image.device, dtype=image.dtype
    ) * float(noise_std)
    return (image * scale + shift + noise).clamp(-5.0, 5.0)


def _fork_devices(device: torch.device) -> list[int]:
    if device.type != "cuda":
        return []
    return [torch.cuda.current_device() if device.index is None else int(device.index)]


def _seeded_forward(
    model: FactorizedPhenotypeWorldModel,
    image: torch.Tensor,
    condition: torch.Tensor,
    augmented: torch.Tensor,
    *,
    seed: int,
) -> Any:
    devices = _fork_devices(image.device)
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        if devices:
            with torch.cuda.device(devices[0]):
                torch.cuda.manual_seed(int(seed))
        return model(image, condition, augmented)


def _batch_indices(dataset: Dataset, batches: Sequence[Sequence[str]]) -> list[list[int]]:
    patient_ids = tuple(str(value) for value in getattr(dataset, "patient_ids"))
    lookup = {patient_id: index for index, patient_id in enumerate(patient_ids)}
    if len(lookup) != len(patient_ids):
        raise ValueError("training dataset patient IDs must be unique")
    return [[lookup[str(patient_id)] for patient_id in batch] for batch in batches]


def _loader(dataset: Dataset, batches: Sequence[Sequence[str]], workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_sampler=_batch_indices(dataset, batches),
        num_workers=int(workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(workers),
        prefetch_factor=1 if workers else None,
    )


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    allowed = {
        "patient_id",
        "image",
        "ftv_target",
        "ftv_mask",
        "condition",
        "clinical_target",
    }
    if set(batch) != allowed:
        raise PermissionError(f"training batch schema drifted: {sorted(batch)}")
    if any("pcr" in str(key).casefold() for key in batch):
        raise PermissionError("pCR field entered representation training batch")
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _augmentation_seed(effective_seed: int, epoch: int, logical: int, micro: int) -> int:
    return (
        int(effective_seed) * 1_000_003
        + int(epoch) * 10_007
        + int(logical) * 101
        + int(micro)
    ) % (2**63 - 1)


def _model_seed(effective_seed: int, epoch: int, logical: int, micro: int) -> int:
    return (
        int(effective_seed) * 1_000_033
        + int(epoch) * 10_009
        + int(logical) * 103
        + int(micro)
        + 17
    ) % (2**63 - 1)


def _sigreg_seed(effective_seed: int, epoch: int, logical: int) -> int:
    return (
        int(effective_seed) * 1_000_037 + int(epoch) * 10_037 + int(logical)
    ) % (2**63 - 1)


def _reference_mapping(output: Any) -> dict[str, torch.Tensor]:
    augmented = output.augmented_phenotype_state
    if augmented is None:
        raise ValueError("logical reference requires the registered second image view")
    return {
        "response_state": output.response_state,
        "phenotype_state": output.phenotype_state,
        "augmented_phenotype_state": augmented,
        "response_online": output.response_online,
    }


def _logical_reference(
    model: FactorizedPhenotypeWorldModel,
    objective: FactorizedObjective,
    retained: Sequence[Mapping[str, Any]],
    device: torch.device,
    hyperparameters: TrainHyperparameters,
    *,
    effective_seed: int,
    epoch: int,
    logical_index: int,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    references: dict[str, list[torch.Tensor]] = {name: [] for name in REFERENCE_KEYS}
    with torch.no_grad():
        for micro_index, cpu_batch in enumerate(retained):
            batch = _to_device(cpu_batch, device)
            image = batch["image"]
            augmented = weak_photometric_view(
                image,
                seed=_augmentation_seed(
                    effective_seed, epoch, logical_index, micro_index
                ),
                scale_half_width=hyperparameters.augmentation_scale_half_width,
                shift_half_width=hyperparameters.augmentation_shift_half_width,
                noise_std=hyperparameters.augmentation_noise_std,
            )
            output = _seeded_forward(
                model,
                image,
                batch["condition"],
                augmented,
                seed=_model_seed(effective_seed, epoch, logical_index, micro_index),
            )
            for name, value in _reference_mapping(output).items():
                references[name].append(value.detach())
            del output, augmented, image, batch
    joined = {
        name: torch.cat(values, dim=0).detach().requires_grad_(True)
        for name, values in references.items()
    }
    if any(int(value.size(0)) != LOGICAL_BATCH_SIZE for value in joined.values()):
        raise ValueError("formal training logical reference must contain 32 patients")
    logical_loss, logical_stats = objective.logical_regularizers(
        response_state=joined["response_state"],
        phenotype_state=joined["phenotype_state"],
        augmented_phenotype_state=joined["augmented_phenotype_state"],
        response_online=joined["response_online"],
        sigreg_seed=_sigreg_seed(effective_seed, epoch, logical_index),
    )
    gradients = torch.autograd.grad(
        logical_loss,
        tuple(joined[name] for name in REFERENCE_KEYS),
    )
    gradient_mapping = {
        name: gradient.detach()
        for name, gradient in zip(REFERENCE_KEYS, gradients, strict=True)
    }
    detached_reference = {name: value.detach() for name, value in joined.items()}
    detached_stats = {name: value.detach() for name, value in logical_stats.items()}
    return logical_loss.detach(), detached_reference, gradient_mapping, detached_stats


def _effective_rank(state: torch.Tensor) -> float:
    matrix = state.reshape(-1, state.size(-1)).double()
    if matrix.size(0) < 2:
        return 0.0
    matrix = matrix - matrix.mean(dim=0, keepdim=True)
    covariance = matrix.T @ matrix / float(matrix.size(0) - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    total = eigenvalues.sum()
    if not bool(total > 0):
        return 0.0
    probabilities = eigenvalues / total
    positive = probabilities > 0
    entropy = -(probabilities[positive] * probabilities[positive].log()).sum()
    return float(entropy.exp())


def _state_diagnostics(
    responses: Sequence[torch.Tensor],
    phenotypes: Sequence[torch.Tensor],
    augmented: Sequence[torch.Tensor],
) -> dict[str, float]:
    response = torch.cat(tuple(responses), dim=0).float()
    phenotype = torch.cat(tuple(phenotypes), dim=0).float()
    augmented_phenotype = torch.cat(tuple(augmented), dim=0).float()
    cosine = F.cosine_similarity(
        phenotype.reshape(-1, phenotype.size(-1)),
        augmented_phenotype.reshape(-1, augmented_phenotype.size(-1)),
        dim=-1,
    )
    return {
        "response_std": float(response.std(dim=0, unbiased=False).mean()),
        "phenotype_std": float(phenotype.std(dim=0, unbiased=False).mean()),
        "response_effective_rank": _effective_rank(response),
        "phenotype_effective_rank": _effective_rank(phenotype),
        "augmentation_cosine": float(cosine.mean()),
    }


def _empty_sums() -> defaultdict[str, float]:
    return defaultdict(float)


def run_logical_train_epoch(
    model: FactorizedPhenotypeWorldModel,
    objective: FactorizedObjective,
    dataset: Dataset,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    logical_batches: Sequence[Sequence[str]],
    hyperparameters: TrainHyperparameters,
    *,
    effective_seed: int,
    epoch: int,
) -> dict[str, float]:
    """One nonlinear B32 reduction, clip, AdamW step, and EMA update per batch."""

    if PCR_LABEL_ACCESS != "FORBIDDEN":
        raise PermissionError("pCR firewall is not active")
    hyperparameters.validate()
    physical = physical_patient_batches(
        logical_batches, hyperparameters.physical_batch_size
    )
    loader = _loader(dataset, physical, hyperparameters.workers)
    iterator = iter(loader)
    model.train(True)
    objective.train(True)
    sums = _empty_sums()
    responses: list[torch.Tensor] = []
    phenotypes: list[torch.Tensor] = []
    augmented_states: list[torch.Tensor] = []
    samples = microbatches = logical_steps = 0
    start = time.monotonic()
    for logical_index, logical in enumerate(logical_batches):
        if len(logical) != LOGICAL_BATCH_SIZE:
            raise ValueError("every formal training logical batch must contain 32 patients")
        retained: list[Mapping[str, Any]] = []
        for _ in range(hyperparameters.accumulation_steps):
            try:
                retained.append(next(iterator))
            except StopIteration as error:
                raise RuntimeError("training loader ended inside a logical batch") from error
        logical_ftv_patients = sum(
            int(torch.as_tensor(batch["ftv_mask"]).bool().any(dim=1).sum())
            for batch in retained
        )
        logical_loss, reference, reference_gradient, logical_stats = _logical_reference(
            model,
            objective,
            retained,
            device,
            hyperparameters,
            effective_seed=effective_seed,
            epoch=epoch,
            logical_index=logical_index,
        )
        optimizer.zero_grad(set_to_none=True)
        offset = 0
        seen_ftv = 0
        for micro_index, cpu_batch in enumerate(retained):
            batch = _to_device(cpu_batch, device)
            image = batch["image"]
            batch_size = int(image.size(0))
            augmented_image = weak_photometric_view(
                image,
                seed=_augmentation_seed(
                    effective_seed, epoch, logical_index, micro_index
                ),
                scale_half_width=hyperparameters.augmentation_scale_half_width,
                shift_half_width=hyperparameters.augmentation_shift_half_width,
                noise_std=hyperparameters.augmentation_noise_std,
            )
            output = _seeded_forward(
                model,
                image,
                batch["condition"],
                augmented_image,
                seed=_model_seed(effective_seed, epoch, logical_index, micro_index),
            )
            _, additive = objective.additive_components(
                output,
                batch["ftv_target"],
                batch["ftv_mask"],
                batch["clinical_target"],
            )
            stop = offset + batch_size
            current = _reference_mapping(output)
            reference_slice = {name: value[offset:stop] for name, value in reference.items()}
            gradient_slice = {
                name: value[offset:stop] for name, value in reference_gradient.items()
            }
            for name in REFERENCE_KEYS:
                if not torch.allclose(
                    current[name].detach(),
                    reference_slice[name],
                    rtol=1e-5,
                    atol=2e-6,
                ):
                    raise RuntimeError(
                        f"model/augmentation changed between logical reference and gradient pass: {name}"
                    )
            surrogate = logical_loss_surrogate(
                current,
                reference_slice,
                gradient_slice,
                logical_loss,
                logical_batch_size=LOGICAL_BATCH_SIZE,
            )
            micro_ftv = int(additive["ftv_patients"].item())
            seen_ftv += micro_ftv
            non_ftv = additive["_non_ftv_component"] * (
                float(batch_size) / float(LOGICAL_BATCH_SIZE)
            )
            if logical_ftv_patients:
                ftv = objective.lambda_ftv * additive["_ftv_component_raw"] * (
                    float(micro_ftv) / float(logical_ftv_patients)
                )
            else:
                ftv = additive["_ftv_component_raw"] * 0.0
            regularizer = surrogate * (
                float(batch_size) / float(LOGICAL_BATCH_SIZE)
            )
            loss = non_ftv + ftv + regularizer
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Goal F training loss is non-finite")
            loss.backward()

            samples += batch_size
            microbatches += 1
            responses.append(output.response_state.detach().float().cpu())
            phenotypes.append(output.phenotype_state.detach().float().cpu())
            assert output.augmented_phenotype_state is not None
            augmented_states.append(
                output.augmented_phenotype_state.detach().float().cpu()
            )
            for name in (
                "response_jepa_loss",
                "phenotype_future_loss",
                "adversary_loss",
                "adversary_hr_loss",
                "adversary_her2_loss",
                "weighted_adversary_loss",
            ):
                sums[name] += float(additive[name].detach()) * batch_size
            sums["ftv_loss_numerator"] += float(additive["ftv_loss"].detach()) * micro_ftv
            sums["ftv_patients"] += micro_ftv
            sums["ftv_valid_visits"] += float(additive["ftv_valid_visits"])
            offset = stop
            del output, augmented_image, image, batch, loss
        if offset != LOGICAL_BATCH_SIZE or seen_ftv != logical_ftv_patients:
            raise AssertionError("logical patient/FTV numerator audit failed")
        gradient_norm = float(
            clip_grad_norm_(
                model.parameters(),
                hyperparameters.max_grad_norm,
                error_if_nonfinite=True,
            )
        )
        optimizer.step()
        model.update_target(hyperparameters.ema_momentum)
        optimizer.zero_grad(set_to_none=True)
        sums["gradient_norm"] += gradient_norm
        for name, value in logical_stats.items():
            sums[name] += float(value) * LOGICAL_BATCH_SIZE
        logical_steps += 1
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise RuntimeError("training loader produced extra microbatches")
    if not samples or logical_steps != len(logical_batches):
        raise RuntimeError("Goal F training epoch was incomplete")
    diagnostics = _state_diagnostics(responses, phenotypes, augmented_states)
    ftv_loss = sums["ftv_loss_numerator"] / max(sums["ftv_patients"], 1.0)
    result = {
        name: sums[name] / samples
        for name in (
            "response_jepa_loss",
            "phenotype_future_loss",
            "adversary_loss",
            "adversary_hr_loss",
            "adversary_her2_loss",
            "weighted_adversary_loss",
            "sigreg_loss",
            "phenotype_consistency_loss",
            "phenotype_invariance_loss",
            "phenotype_variance_loss",
            "phenotype_covariance_loss",
            "crosscov_loss",
            "logical_regularizer_loss",
        )
    }
    result.update(diagnostics)
    result.update(
        {
            "ftv_loss": ftv_loss,
            "weighted_ftv_loss": objective.lambda_ftv * ftv_loss,
            "grounded_patients": sums["ftv_patients"],
            "valid_ftv_visits": sums["ftv_valid_visits"],
            "gradient_norm": sums["gradient_norm"] / logical_steps,
            "samples": float(samples),
            "logical_batches": float(logical_steps),
            "physical_microbatches": float(microbatches),
            "optimizer_steps": float(logical_steps),
            "ema_updates": float(logical_steps),
            "seconds": time.monotonic() - start,
        }
    )
    result["loss"] = (
        result["response_jepa_loss"]
        + objective.weights.lambda_phenotype_future * result["phenotype_future_loss"]
        + result["weighted_adversary_loss"]
        + result["weighted_ftv_loss"]
        + result["logical_regularizer_loss"]
    )
    # This pCR-free criterion is shared by F1/F2 and has a coherent minimization
    # direction. Adversary CE remains in the optimized training loss, but cannot
    # select a checkpoint because the GRL encoder and adversary are minimax
    # opponents and lower CE would favor greater clinical decodability.
    result["selection_loss"] = result["loss"] - result["weighted_adversary_loss"]
    return result


def _chunks(values: Sequence[str], size: int) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(values[start : start + size])
        for start in range(0, len(values), size)
    )


@torch.no_grad()
def run_validation_epoch(
    model: FactorizedPhenotypeWorldModel,
    objective: FactorizedObjective,
    dataset: Dataset,
    device: torch.device,
    hyperparameters: TrainHyperparameters,
    *,
    effective_seed: int,
    epoch: int,
) -> dict[str, float]:
    """Validation-only loss/collapse diagnostics; never sees outer-test data."""

    hyperparameters.validate()
    patient_ids = tuple(str(value) for value in getattr(dataset, "patient_ids"))
    logical_batches = _chunks(patient_ids, LOGICAL_BATCH_SIZE)
    physical: list[tuple[str, ...]] = []
    expected_per_logical: list[int] = []
    for logical in logical_batches:
        chunks = _chunks(logical, hyperparameters.physical_batch_size)
        physical.extend(chunks)
        expected_per_logical.append(len(chunks))
    loader = _loader(dataset, physical, hyperparameters.workers)
    iterator = iter(loader)
    model.train(False)
    objective.train(False)
    sums = _empty_sums()
    responses: list[torch.Tensor] = []
    phenotypes: list[torch.Tensor] = []
    augmented_states: list[torch.Tensor] = []
    samples = microbatches = 0
    start = time.monotonic()
    for logical_index, (logical, expected_microbatches) in enumerate(
        zip(logical_batches, expected_per_logical, strict=True)
    ):
        logical_states: dict[str, list[torch.Tensor]] = {
            name: [] for name in REFERENCE_KEYS
        }
        logical_seen = 0
        for micro_index in range(expected_microbatches):
            try:
                cpu_batch = next(iterator)
            except StopIteration as error:
                raise RuntimeError("validation loader ended inside logical batch") from error
            batch = _to_device(cpu_batch, device)
            image = batch["image"]
            batch_size = int(image.size(0))
            augmented_image = weak_photometric_view(
                image,
                seed=_augmentation_seed(
                    effective_seed, epoch, logical_index, micro_index
                ),
                scale_half_width=hyperparameters.augmentation_scale_half_width,
                shift_half_width=hyperparameters.augmentation_shift_half_width,
                noise_std=hyperparameters.augmentation_noise_std,
            )
            output = _seeded_forward(
                model,
                image,
                batch["condition"],
                augmented_image,
                seed=_model_seed(effective_seed, epoch, logical_index, micro_index),
            )
            _, additive = objective.additive_components(
                output,
                batch["ftv_target"],
                batch["ftv_mask"],
                batch["clinical_target"],
            )
            for name, value in _reference_mapping(output).items():
                logical_states[name].append(value.detach())
            micro_ftv = int(additive["ftv_patients"].item())
            for name in (
                "response_jepa_loss",
                "phenotype_future_loss",
                "adversary_loss",
                "adversary_hr_loss",
                "adversary_her2_loss",
                "weighted_adversary_loss",
            ):
                sums[name] += float(additive[name]) * batch_size
            sums["ftv_loss_numerator"] += float(additive["ftv_loss"]) * micro_ftv
            sums["ftv_patients"] += micro_ftv
            sums["ftv_valid_visits"] += float(additive["ftv_valid_visits"])
            responses.append(output.response_state.float().cpu())
            phenotypes.append(output.phenotype_state.float().cpu())
            assert output.augmented_phenotype_state is not None
            augmented_states.append(output.augmented_phenotype_state.float().cpu())
            samples += batch_size
            logical_seen += batch_size
            microbatches += 1
        if logical_seen != len(logical):
            raise RuntimeError("validation physical batches crossed a logical boundary")
        joined = {name: torch.cat(values, dim=0) for name, values in logical_states.items()}
        _, logical_stats = objective.logical_regularizers(
            response_state=joined["response_state"],
            phenotype_state=joined["phenotype_state"],
            augmented_phenotype_state=joined["augmented_phenotype_state"],
            response_online=joined["response_online"],
            sigreg_seed=_sigreg_seed(effective_seed, epoch, logical_index),
        )
        for name, value in logical_stats.items():
            sums[name] += float(value) * logical_seen
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise RuntimeError("validation loader produced extra microbatches")
    if not samples:
        raise RuntimeError("validation dataset is empty")
    diagnostics = _state_diagnostics(responses, phenotypes, augmented_states)
    ftv_loss = sums["ftv_loss_numerator"] / max(sums["ftv_patients"], 1.0)
    result = {
        name: sums[name] / samples
        for name in (
            "response_jepa_loss",
            "phenotype_future_loss",
            "adversary_loss",
            "adversary_hr_loss",
            "adversary_her2_loss",
            "weighted_adversary_loss",
            "sigreg_loss",
            "phenotype_consistency_loss",
            "phenotype_invariance_loss",
            "phenotype_variance_loss",
            "phenotype_covariance_loss",
            "crosscov_loss",
            "logical_regularizer_loss",
        )
    }
    result.update(diagnostics)
    result.update(
        {
            "ftv_loss": ftv_loss,
            "weighted_ftv_loss": objective.lambda_ftv * ftv_loss,
            "grounded_patients": sums["ftv_patients"],
            "valid_ftv_visits": sums["ftv_valid_visits"],
            "samples": float(samples),
            "logical_batches": float(len(logical_batches)),
            "physical_microbatches": float(microbatches),
            "seconds": time.monotonic() - start,
        }
    )
    result["loss"] = (
        result["response_jepa_loss"]
        + objective.weights.lambda_phenotype_future * result["phenotype_future_loss"]
        + result["weighted_adversary_loss"]
        + result["weighted_ftv_loss"]
        + result["logical_regularizer_loss"]
    )
    result["selection_loss"] = result["loss"] - result["weighted_adversary_loss"]
    return result


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".pt", dir=path.parent
    )
    os.close(descriptor)
    try:
        torch.save(dict(payload), temporary)
        os.chmod(temporary, 0o600)
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _write_history(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".csv", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _noncollapsed(row: Mapping[str, Any], hyperparameters: TrainHyperparameters) -> bool:
    return bool(
        row["finite"]
        and float(row["val_response_std"]) >= hyperparameters.min_response_std
        and float(row["val_phenotype_std"]) >= hyperparameters.min_phenotype_std
        and float(row["val_phenotype_effective_rank"])
        >= hyperparameters.min_phenotype_effective_rank
        and float(row["val_augmentation_cosine"])
        >= hyperparameters.min_augmentation_cosine
    )


def select_checkpoint(
    history: Sequence[Mapping[str, Any]],
    hyperparameters: TrainHyperparameters,
) -> dict[str, Any]:
    if not history:
        raise ValueError("cannot select from empty Goal F history")
    finite = [row for row in history if bool(row.get("finite"))]
    if not finite:
        raise RuntimeError("every Goal F validation epoch is non-finite")
    eligible = [row for row in finite if _noncollapsed(row, hyperparameters)]
    if eligible:
        selected = min(
            eligible,
            key=lambda row: (float(row["val_selection_loss"]), int(row["epoch"])),
        )
        mode = "primary_noncollapsed"
        passed = True
    else:
        selected = min(
            finite,
            key=lambda row: (
                max(0.0, hyperparameters.min_phenotype_std - float(row["val_phenotype_std"])),
                max(
                    0.0,
                    hyperparameters.min_phenotype_effective_rank
                    - float(row["val_phenotype_effective_rank"]),
                ),
                max(
                    0.0,
                    hyperparameters.min_augmentation_cosine
                    - float(row["val_augmentation_cosine"]),
                ),
                float(row["val_selection_loss"]),
                int(row["epoch"]),
            ),
        )
        mode = "collapse_diagnostic_fallback"
        passed = False
    return {
        "schema_version": 1,
        "selection_rule": (
            "minimum validation pCR-free shared representation loss excluding "
            "minimax adversary CE among finite zR/zP-noncollapsed epochs"
        ),
        "selection_mode": mode,
        "experiment_pass": passed,
        "selected_epoch": int(selected["epoch"]),
        "selected_validation_selection_loss": float(selected["val_selection_loss"]),
        "selected_validation_total_loss": float(selected["val_loss"]),
        "selected_response_std": float(selected["val_response_std"]),
        "selected_phenotype_std": float(selected["val_phenotype_std"]),
        "selected_phenotype_effective_rank": float(selected["val_phenotype_effective_rank"]),
        "selected_augmentation_cosine": float(selected["val_augmentation_cosine"]),
        "pcr_used": False,
        "test_data_used": False,
    }


def train_epochs(
    *,
    model: FactorizedPhenotypeWorldModel,
    objective: FactorizedObjective,
    train_dataset: Dataset,
    val_dataset: Dataset,
    device: torch.device,
    output_dir: str | Path,
    hyperparameters: TrainHyperparameters,
    seed_base: int,
    fold: int,
    effective_seed: int,
    provenance: Mapping[str, Any],
    preregistration: Mapping[str, Any],
) -> dict[str, Any]:
    if PCR_LABEL_ACCESS != "FORBIDDEN":
        raise PermissionError("pCR firewall is not active")
    hyperparameters.validate()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite or mix Goal F cell: {output}")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    if preregistration.get("status") != "PASS":
        raise PermissionError("formal Goal F training requires a verified preregistration")
    if provenance.get("PCR_LABEL_ACCESS") != "FORBIDDEN":
        raise PermissionError("run provenance lacks the pCR firewall")
    model.to(device)
    objective.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparameters.learning_rate,
        weight_decay=hyperparameters.weight_decay,
    )
    history: list[dict[str, Any]] = []
    best = math.inf
    stale = 0
    for epoch in range(1, hyperparameters.epochs + 1):
        logical = logical_patient_batches(
            getattr(train_dataset, "patient_ids"), effective_seed, epoch
        )
        train_stats = run_logical_train_epoch(
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
        val_stats = run_validation_epoch(
            model,
            objective,
            val_dataset,
            device,
            hyperparameters,
            effective_seed=effective_seed,
            epoch=epoch,
        )
        row: dict[str, Any] = {
            "epoch": epoch,
            "arm": model.arm,
            "seed_base": int(seed_base),
            "fold": int(fold),
            "effective_seed": int(effective_seed),
            "train_patient_order_sha256": canonical_sha256(
                tuple(patient for batch in logical for patient in batch)
            ),
        }
        row.update({f"train_{name}": value for name, value in train_stats.items()})
        row.update({f"val_{name}": value for name, value in val_stats.items()})
        row["finite"] = all(
            math.isfinite(float(value))
            for name, value in row.items()
            if (name.startswith("train_") or name.startswith("val_"))
            and isinstance(value, (int, float))
        )
        history.append(row)
        checkpoint = {
            "schema_version": 1,
            "stage": "clinical_residual_phenotype_state",
            "arm": model.arm,
            "seed_base": int(seed_base),
            "fold": int(fold),
            "effective_seed": int(effective_seed),
            "epoch": int(epoch),
            "state_dict": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_config": model.model_config(),
            "architecture_contract": model.architecture_contract(),
            "objective": {
                "weights": asdict(objective.weights),
                "lambda_adv": objective.lambda_adv,
                "step_weights": objective.step_weights.detach().cpu().tolist(),
            },
            "hyperparameters": asdict(hyperparameters),
            "provenance": dict(provenance),
            "preregistration": dict(preregistration),
            "epoch_metrics": row,
            "PCR_LABEL_ACCESS": "FORBIDDEN",
            "pcr_used": False,
            "pcr_parsed": False,
            "test_data_used": False,
            "delta_ftv_used_for_selection": False,
        }
        _atomic_torch_save(output / f"epoch_{epoch:02d}.pt", checkpoint)
        _write_history(output / "history.csv", history)
        if (
            _noncollapsed(row, hyperparameters)
            and float(row["val_selection_loss"]) < best
        ):
            best = float(row["val_selection_loss"])
            stale = 0
        else:
            stale += 1
        if stale >= hyperparameters.patience:
            break
    selection = select_checkpoint(history, hyperparameters)
    selection.update(
        {
            "arm": model.arm,
            "seed_base": int(seed_base),
            "fold": int(fold),
            "effective_seed": int(effective_seed),
            "epochs": len(history),
            "hyperparameters": asdict(hyperparameters),
            "preregistration": dict(preregistration),
            "PCR_LABEL_ACCESS": "FORBIDDEN",
        }
    )
    _atomic_json(output / "selection.json", selection)
    selected_epoch = int(selection["selected_epoch"])
    selected = torch.load(
        output / f"epoch_{selected_epoch:02d}.pt",
        map_location="cpu",
        weights_only=True,
    )
    selected["selected"] = True
    selected["selection"] = selection
    _atomic_torch_save(output / "selected.pt", selected)
    return selection


__all__ = [
    "LOGICAL_BATCH_SIZE",
    "REFERENCE_KEYS",
    "TrainHyperparameters",
    "run_logical_train_epoch",
    "run_validation_epoch",
    "seed_everything",
    "select_checkpoint",
    "train_epochs",
    "weak_photometric_view",
]

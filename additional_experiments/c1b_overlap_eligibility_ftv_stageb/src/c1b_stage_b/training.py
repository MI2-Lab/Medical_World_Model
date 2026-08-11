"""Stage B logical-batch training and validation-only checkpoint selection."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import csv
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .contracts import (
    EFFECTIVE_BATCH_SIZE,
    arm_spec,
    canonical_sha256,
    file_sha256,
    ordered_patient_sha256,
    validate_batch_contract,
    validate_seed_fold,
)
from .data import StageBDataset
from .gate import StageAAuthorization
from .upstream import (
    common_initialization_sha256,
    transition_sha256,
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
    min_representation_std: float = 0.05

    def validate(self) -> None:
        validate_batch_contract(self.physical_batch_size, self.accumulation_steps)
        if self.workers < 0 or self.epochs <= 0 or self.patience <= 0:
            raise ValueError("workers must be nonnegative; epochs/patience must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning rate must be positive and weight decay nonnegative")
        if not 0.0 < self.ema_momentum < 1.0:
            raise ValueError("EMA momentum must be in (0,1)")
        if self.max_grad_norm <= 0 or self.min_representation_std <= 0:
            raise ValueError("gradient and representation thresholds must be positive")


def logical_patient_batches(
    patient_ids: Iterable[str], effective_seed: int, epoch: int
) -> tuple[tuple[str, ...], ...]:
    """Shuffle once, drop only the shared logical tail, and chunk by 32."""

    ids = tuple(str(value) for value in patient_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("logical sampler patient IDs must be unique")
    generator = np.random.default_rng(np.random.SeedSequence([int(effective_seed), int(epoch)]))
    order = generator.permutation(len(ids))
    usable = (len(ids) // EFFECTIVE_BATCH_SIZE) * EFFECTIVE_BATCH_SIZE
    ordered = tuple(ids[int(index)] for index in order[:usable])
    return tuple(
        ordered[start : start + EFFECTIVE_BATCH_SIZE]
        for start in range(0, usable, EFFECTIVE_BATCH_SIZE)
    )


def physical_patient_batches(
    logical_batches: Sequence[Sequence[str]], physical_batch_size: int
) -> tuple[tuple[str, ...], ...]:
    if EFFECTIVE_BATCH_SIZE % int(physical_batch_size):
        raise ValueError("physical batch size must divide the logical batch")
    output: list[tuple[str, ...]] = []
    for logical in logical_batches:
        logical = tuple(logical)
        if len(logical) != EFFECTIVE_BATCH_SIZE:
            raise ValueError("every training logical batch must contain exactly 32 patients")
        output.extend(
            tuple(logical[start : start + physical_batch_size])
            for start in range(0, EFFECTIVE_BATCH_SIZE, physical_batch_size)
        )
    return tuple(output)


def _batch_indices(dataset: StageBDataset, batches: Sequence[Sequence[str]]) -> list[list[int]]:
    lookup = {patient_id: index for index, patient_id in enumerate(dataset.patient_ids)}
    return [[lookup[str(patient_id)] for patient_id in batch] for batch in batches]


def _loader(
    dataset: StageBDataset,
    batches: Sequence[Sequence[str]],
    workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_sampler=_batch_indices(dataset, batches),
        num_workers=int(workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(workers),
        prefetch_factor=1 if workers else None,
    )


def scale_microbatch_components(
    base_component: torch.Tensor,
    ftv_component_raw: torch.Tensor,
    *,
    microbatch_size: int,
    logical_batch_size: int,
    microbatch_ftv_patients: int,
    logical_ftv_patients: int,
    lambda_ftv: float,
) -> torch.Tensor:
    """Compose an exact patient-mean logical loss from microbatch means."""

    if microbatch_size <= 0 or logical_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if not 0 <= microbatch_ftv_patients <= microbatch_size:
        raise ValueError("microbatch FTV patient count is invalid")
    if not 0 <= logical_ftv_patients <= logical_batch_size:
        raise ValueError("logical FTV patient count is invalid")
    base = base_component * (float(microbatch_size) / float(logical_batch_size))
    if logical_ftv_patients:
        ftv = ftv_component_raw * (
            float(microbatch_ftv_patients) / float(logical_ftv_patients)
        )
    else:
        ftv = ftv_component_raw * 0.0
    return base + float(lambda_ftv) * ftv


def _logical_ftv_counts(
    logical_batches: Sequence[Sequence[str]],
    transformed_ftv: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[int, ...]:
    counts: list[int] = []
    for logical in logical_batches:
        count = 0
        for patient_id in logical:
            pair = transformed_ftv.get(str(patient_id))
            count += int(pair is not None and bool(np.asarray(pair[1], dtype=bool).any()))
        counts.append(count)
    return tuple(counts)


def _finite_loss(loss: torch.Tensor) -> None:
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError(f"non-finite Stage B loss: {float(loss.detach())}")


def logical_sigreg_surrogate(
    online_state: torch.Tensor,
    reference_state: torch.Tensor,
    reference_gradient: torch.Tensor,
    logical_sigreg_loss: torch.Tensor,
    *,
    logical_batch_size: int,
) -> torch.Tensor:
    """Return a microbatch surrogate with the exact logical SIGReg gradient.

    ``scale_microbatch_components`` subsequently weights this value by
    ``microbatch_size / logical_batch_size``.  The multiplier below therefore
    makes the summed microbatch gradient exactly equal to the gradient of one
    SIGReg evaluation over all logical-batch patients.  At the reference point
    the summed value is the exact logical SIGReg value as well.
    """

    if online_state.shape != reference_state.shape or online_state.shape != reference_gradient.shape:
        raise ValueError("logical SIGReg state/reference/gradient shapes differ")
    microbatch_size = int(online_state.size(0))
    if microbatch_size <= 0 or int(logical_batch_size) < microbatch_size:
        raise ValueError("logical SIGReg batch sizes are invalid")
    multiplier = float(logical_batch_size) / float(microbatch_size)
    correction = ((online_state - reference_state) * reference_gradient).sum()
    return logical_sigreg_loss.detach() + multiplier * correction


class _LogicalSIGRegAdapter(torch.nn.Module):
    """Make the frozen objective expose an exact accumulated SIGReg gradient."""

    def __init__(
        self,
        reference_state: torch.Tensor,
        reference_gradient: torch.Tensor,
        logical_sigreg_loss: torch.Tensor,
        logical_batch_size: int,
    ) -> None:
        super().__init__()
        self.reference_state = reference_state
        self.reference_gradient = reference_gradient
        self.logical_sigreg_loss = logical_sigreg_loss
        self.logical_batch_size = int(logical_batch_size)

    def forward(self, state_vbd: torch.Tensor) -> torch.Tensor:
        return logical_sigreg_surrogate(
            state_vbd.transpose(0, 1),
            self.reference_state,
            self.reference_gradient,
            self.logical_sigreg_loss,
            logical_batch_size=self.logical_batch_size,
        )


class _ConstantSIGRegAdapter(torch.nn.Module):
    def __init__(self, value: torch.Tensor) -> None:
        super().__init__()
        self.value = value

    def forward(self, state_vbd: torch.Tensor) -> torch.Tensor:
        return self.value.to(device=state_vbd.device, dtype=state_vbd.dtype)


@contextmanager
def _temporary_sigreg(
    objective: torch.nn.Module, replacement: torch.nn.Module
):
    original = objective.sigreg
    objective.sigreg = replacement
    try:
        yield
    finally:
        objective.sigreg = original


def _fork_devices(device: torch.device) -> list[int]:
    if device.type != "cuda":
        return []
    return [torch.cuda.current_device() if device.index is None else int(device.index)]


def _logical_sigreg_reference(
    model: torch.nn.Module,
    objective: torch.nn.Module,
    retained_batches: Sequence[Mapping[str, Any]],
    device: torch.device,
    *,
    direction_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate frozen SIGReg once on 32 deterministic online states.

    The extra no-grad online-encoder pass avoids retaining 32 patients of 3-D
    activation graphs.  The online encoder/projector contain no stochastic
    layer, so the following gradient pass reaches the same reference states.
    """

    references: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in retained_batches:
            image = batch["image"].to(device, non_blocking=True)
            _, online, _ = model.encode_online(image, None)
            references.append(online.detach())
    reference_bvd = torch.cat(references, dim=0)
    if reference_bvd.ndim != 3 or int(reference_bvd.size(0)) != EFFECTIVE_BATCH_SIZE:
        raise ValueError("training SIGReg reference must be [32,visits,latent]")
    reference_vbd = reference_bvd.transpose(0, 1).detach().requires_grad_(True)
    fork_devices = _fork_devices(device)
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(int(direction_seed))
        if fork_devices:
            with torch.cuda.device(fork_devices[0]):
                torch.cuda.manual_seed(int(direction_seed))
        logical_loss = objective.sigreg(reference_vbd)
    gradient_vbd, = torch.autograd.grad(logical_loss, reference_vbd)
    if not bool(torch.isfinite(logical_loss)) or not bool(torch.isfinite(gradient_vbd).all()):
        raise FloatingPointError("logical-batch SIGReg value/gradient is non-finite")
    return (
        logical_loss.detach(),
        reference_vbd.detach().transpose(0, 1),
        gradient_vbd.detach().transpose(0, 1),
    )


def run_logical_train_epoch(
    model: torch.nn.Module,
    objective: torch.nn.Module,
    dataset: StageBDataset,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    logical_batches: Sequence[Sequence[str]],
    hyperparameters: TrainHyperparameters,
    *,
    effective_seed: int,
    epoch: int,
) -> dict[str, float]:
    """Train with one exact objective/clip/optimizer/EMA per 32 patients."""

    hyperparameters.validate()
    physical = physical_patient_batches(logical_batches, hyperparameters.physical_batch_size)
    expected_microbatches = len(logical_batches) * hyperparameters.accumulation_steps
    if len(physical) != expected_microbatches:
        raise AssertionError("logical/physical accumulation contract drifted")
    # The frozen no-grounding objective deliberately reports zero FTV patients
    # even when loss-side FTV metadata exists in the shared dataset.  Only the
    # grounded objective consumes that metadata, so its microbatch numerator is
    # the one that must reconcile with the precomputed logical count.
    grounded_objective = float(objective.lambda_ftv) > 0.0
    logical_ftv = (
        _logical_ftv_counts(logical_batches, dataset.transformed_ftv)
        if grounded_objective
        else (0,) * len(logical_batches)
    )
    loader = _loader(dataset, physical, hyperparameters.workers)
    model.train(True)
    objective.train(True)
    sums: dict[str, float] = defaultdict(float)
    responses: list[torch.Tensor] = []
    samples = logical_steps = microbatches = 0
    start = time.monotonic()
    iterator = iter(loader)
    for logical_index, _ in enumerate(logical_batches):
        retained: list[Mapping[str, Any]] = []
        for _micro in range(hyperparameters.accumulation_steps):
            try:
                retained.append(next(iterator))
            except StopIteration as error:
                raise RuntimeError(
                    "training loader ended inside a logical batch"
                ) from error
        direction_seed = (
            int(effective_seed) * 1_000_003
            + int(epoch) * 10_007
            + int(logical_index)
        ) % (2**63 - 1)
        logical_sigreg, reference, reference_gradient = _logical_sigreg_reference(
            model,
            objective,
            retained,
            device,
            direction_seed=direction_seed,
        )
        optimizer.zero_grad(set_to_none=True)
        logical_seen_ftv = 0
        offset = 0
        for batch in retained:
            image = batch["image"].to(device, non_blocking=True)
            ftv_target = batch["ftv_target"].to(device, non_blocking=True)
            ftv_mask = batch["ftv_mask"].to(device, non_blocking=True)
            output = model(image, None)
            batch_size = int(image.size(0))
            stop = offset + batch_size
            reference_slice = reference[offset:stop]
            gradient_slice = reference_gradient[offset:stop]
            if not torch.allclose(
                output.online_state.detach(),
                reference_slice,
                rtol=1e-5,
                atol=1e-6,
            ):
                raise RuntimeError(
                    "online encoder changed between logical SIGReg reference and gradient pass"
                )
            sigreg_adapter = _LogicalSIGRegAdapter(
                reference_slice,
                gradient_slice,
                logical_sigreg,
                EFFECTIVE_BATCH_SIZE,
            )
            with _temporary_sigreg(objective, sigreg_adapter):
                _, stats = objective(output, ftv_target, ftv_mask)
            micro_ftv = int(stats["ftv_patients"].item())
            logical_seen_ftv += micro_ftv
            loss = scale_microbatch_components(
                stats["_base_component"],
                stats["_ftv_component_raw"],
                microbatch_size=batch_size,
                logical_batch_size=EFFECTIVE_BATCH_SIZE,
                microbatch_ftv_patients=micro_ftv,
                logical_ftv_patients=logical_ftv[logical_index],
                lambda_ftv=float(objective.lambda_ftv),
            )
            _finite_loss(loss)
            loss.backward()
            samples += batch_size
            microbatches += 1
            responses.append(output.response_state.detach().float().cpu())
            sums["state_loss"] += float(stats["state_loss"]) * batch_size
            sums["sigreg_loss"] += float(logical_sigreg) * batch_size
            sums["base_loss"] += float(stats["base_loss"]) * batch_size
            sums["ftv_loss_numerator"] += float(stats["ftv_loss"]) * micro_ftv
            sums["ftv_patients"] += micro_ftv
            sums["ftv_valid_visits"] += float(stats["ftv_valid_visits"])
            offset = stop
        if offset != EFFECTIVE_BATCH_SIZE:
            raise AssertionError("logical training batch did not contain 32 patients")
        if logical_seen_ftv != logical_ftv[logical_index]:
            raise AssertionError("logical FTV patient numerator/count audit failed")
        total_grad = float(
            clip_grad_norm_(
                model.parameters(), hyperparameters.max_grad_norm, error_if_nonfinite=True
            )
        )
        optimizer.step()
        model.update_target(hyperparameters.ema_momentum)
        optimizer.zero_grad(set_to_none=True)
        sums["gradient_norm"] += total_grad
        logical_steps += 1
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise RuntimeError("training loader produced extra physical microbatches")
    if microbatches != expected_microbatches or logical_steps != len(logical_batches):
        raise RuntimeError("training loader did not preserve the logical batching contract")
    if not samples:
        raise RuntimeError("logical training epoch is empty after shared tail truncation")
    ftv_patients = sums["ftv_patients"]
    ftv_loss = sums["ftv_loss_numerator"] / max(ftv_patients, 1.0)
    response = torch.cat(responses, dim=0)
    result = {
        "base_loss": sums["base_loss"] / samples,
        "state_loss": sums["state_loss"] / samples,
        "sigreg_loss": sums["sigreg_loss"] / samples,
        "ftv_loss": ftv_loss,
        "weighted_ftv_loss": float(objective.lambda_ftv) * ftv_loss,
        "grounded_patients": ftv_patients,
        "valid_ftv_visits": sums["ftv_valid_visits"],
        "representation_std": float(response.std(dim=0, unbiased=False).mean()),
        "gradient_norm": sums["gradient_norm"] / logical_steps,
        "samples": float(samples),
        "logical_batches": float(logical_steps),
        "physical_microbatches": float(microbatches),
        "optimizer_steps": float(logical_steps),
        "ema_updates": float(logical_steps),
        "seconds": time.monotonic() - start,
    }
    result["loss"] = result["base_loss"] + result["weighted_ftv_loss"]
    return result


def _chunked(values: Sequence[str], size: int) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(values[start : start + size]) for start in range(0, len(values), size))


@torch.no_grad()
def run_validation_epoch(
    model: torch.nn.Module,
    objective: torch.nn.Module,
    dataset: StageBDataset,
    device: torch.device,
    physical_batch_size: int,
    workers: int,
) -> dict[str, float]:
    """Evaluate validation with one frozen SIGReg reduction per logical batch."""

    physical_batches = _chunked(dataset.patient_ids, int(physical_batch_size))
    logical_batches = _chunked(dataset.patient_ids, EFFECTIVE_BATCH_SIZE)
    loader = _loader(dataset, physical_batches, workers)
    model.train(False)
    objective.train(False)
    sums: dict[str, float] = defaultdict(float)
    responses: list[torch.Tensor] = []
    samples = 0
    iterator = iter(loader)
    for logical in logical_batches:
        logical_size = len(logical)
        expected_microbatches = math.ceil(logical_size / int(physical_batch_size))
        logical_states: list[torch.Tensor] = []
        logical_records: list[tuple[Any, torch.Tensor, torch.Tensor, int]] = []
        logical_seen = 0
        for _ in range(expected_microbatches):
            try:
                batch = next(iterator)
            except StopIteration as error:
                raise RuntimeError(
                    "validation loader ended inside a logical batch"
                ) from error
            image = batch["image"].to(device, non_blocking=True)
            ftv_target = batch["ftv_target"].to(device, non_blocking=True)
            ftv_mask = batch["ftv_mask"].to(device, non_blocking=True)
            output = model(image, None)
            batch_size = int(image.size(0))
            logical_seen += batch_size
            logical_records.append((output, ftv_target, ftv_mask, batch_size))
            logical_states.append(output.online_state.detach())
            responses.append(output.response_state.detach().float().cpu())
        if logical_seen != logical_size:
            raise RuntimeError("validation physical batches crossed a logical boundary")
        logical_sigreg = objective.sigreg(
            torch.cat(logical_states, dim=0).transpose(0, 1)
        )
        logical_state_numerator = 0.0
        logical_base_numerator = 0.0
        logical_ftv_numerator = 0.0
        logical_ftv_patients = 0
        logical_ftv_visits = 0.0
        adapter = _ConstantSIGRegAdapter(logical_sigreg)
        for output, ftv_target, ftv_mask, batch_size in logical_records:
            with _temporary_sigreg(objective, adapter):
                _, stats = objective(output, ftv_target, ftv_mask)
            grounded = int(stats["ftv_patients"].item())
            logical_state_numerator += float(stats["state_loss"]) * batch_size
            logical_base_numerator += float(stats["base_loss"]) * batch_size
            logical_ftv_numerator += float(stats["ftv_loss"]) * grounded
            logical_ftv_patients += grounded
            logical_ftv_visits += float(stats["ftv_valid_visits"])
        logical_state_loss = logical_state_numerator / logical_size
        logical_base_loss = logical_base_numerator / logical_size
        logical_ftv_loss = logical_ftv_numerator / max(logical_ftv_patients, 1)
        logical_total = logical_base_loss + float(objective.lambda_ftv) * logical_ftv_loss
        if not math.isfinite(logical_total):
            raise FloatingPointError("non-finite logical validation objective")
        samples += logical_size
        sums["state_loss"] += logical_state_loss * logical_size
        sums["sigreg_loss"] += float(logical_sigreg) * logical_size
        sums["base_loss"] += logical_base_loss * logical_size
        sums["ftv_loss_numerator"] += logical_ftv_numerator
        sums["ftv_patients"] += logical_ftv_patients
        sums["ftv_valid_visits"] += logical_ftv_visits
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise RuntimeError("validation loader produced extra physical microbatches")
    if not samples:
        raise RuntimeError("validation cohort is empty")
    ftv_loss = sums["ftv_loss_numerator"] / max(sums["ftv_patients"], 1.0)
    response = torch.cat(responses, dim=0)
    result = {
        "base_loss": sums["base_loss"] / samples,
        "state_loss": sums["state_loss"] / samples,
        "sigreg_loss": sums["sigreg_loss"] / samples,
        "ftv_loss": ftv_loss,
        "weighted_ftv_loss": float(objective.lambda_ftv) * ftv_loss,
        "grounded_patients": sums["ftv_patients"],
        "valid_ftv_visits": sums["ftv_valid_visits"],
        "representation_std": float(response.std(dim=0, unbiased=False).mean()),
        "samples": float(samples),
    }
    result["loss"] = result["base_loss"] + result["weighted_ftv_loss"]
    return result


def select_checkpoint(
    epochs: Sequence[Mapping[str, Any]],
    *,
    grounded: bool,
    min_representation_std: float,
    paired_baseline_state_loss: float | None = None,
) -> dict[str, Any]:
    """Apply the frozen validation-only primary rule and marked fallback."""

    if not epochs:
        raise ValueError("checkpoint selection requires at least one epoch")
    if grounded:
        if (
            paired_baseline_state_loss is None
            or not math.isfinite(float(paired_baseline_state_loss))
            or float(paired_baseline_state_loss) <= 0
        ):
            raise ValueError(
                "grounded selection requires a finite positive paired no-ground baseline"
            )
        allowed = 1.05 * float(paired_baseline_state_loss)
    else:
        allowed = math.inf
    candidates: list[dict[str, Any]] = []
    fallbacks: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for raw in epochs:
        row = dict(raw)
        state = float(row["val_state_loss"])
        ftv = float(row.get("val_ftv_loss", math.nan))
        representation_std = float(row["val_representation_std"])
        ftv_patients = float(row.get("val_grounded_patients", 0.0))
        noncollapse = math.isfinite(representation_std) and representation_std >= min_representation_std
        base_finite = math.isfinite(state)
        base_gate = base_finite and state <= allowed
        ftv_finite = math.isfinite(ftv) and ftv_patients > 0
        eligible = noncollapse and base_finite and (not grounded or (base_gate and ftv_finite))
        row.update(
            {
                "noncollapse": noncollapse,
                "base_gate_pass": base_gate,
                "checkpoint_eligible": eligible,
            }
        )
        evidence.append(row)
        if eligible:
            candidates.append(row)
        if noncollapse and base_finite and (not grounded or ftv_finite):
            fallbacks.append(row)
    if candidates:
        selected = min(
            candidates,
            key=lambda row: (
                float(row["val_ftv_loss"]) if grounded else float(row["val_state_loss"]),
                int(row["epoch"]),
            ),
        )
        mode = "primary"
        passed = True
    else:
        if not fallbacks:
            raise RuntimeError("no finite non-collapsed validation checkpoint exists")
        selected = min(
            fallbacks,
            key=lambda row: (
                max(0.0, float(row["val_state_loss"]) - allowed),
                float(row.get("val_ftv_loss", math.inf)) if grounded else float(row["val_state_loss"]),
                int(row["epoch"]),
            ),
        )
        mode = "fallback_base_gate_failed" if grounded else "fallback_no_eligible_checkpoint"
        passed = False
    return {
        "schema_version": 1,
        "selection_mode": mode,
        "experiment_pass": passed,
        "selection_rule": (
            "minimum validation FTV loss among finite/non-collapsed epochs with <=5% paired-baseline validation state-loss degradation"
            if grounded
            else "minimum finite validation state loss among non-collapsed epochs"
        ),
        "fallback_rule": "minimum base-gate violation, then validation FTV loss; fallback is an experiment failure",
        "paired_baseline_state_loss": paired_baseline_state_loss,
        "allowed_state_loss": None if not grounded else allowed,
        "selected_epoch": int(selected["epoch"]),
        "selected_validation_total_loss": float(selected["val_loss"]),
        "selected_validation_base_loss": float(selected["val_base_objective"]),
        "selected_validation_state_loss": float(selected["val_state_loss"]),
        "selected_validation_ftv_loss": float(selected.get("val_ftv_loss", math.nan)),
        "selected_representation_std": float(selected["val_representation_std"]),
        "finite_status": bool(selected.get("finite", False)),
        "state_loss_degradation_fraction": (
            None
            if not grounded
            else float(selected["val_state_loss"]) / float(paired_baseline_state_loss) - 1.0
        ),
        "optimization_safety_pass": bool(passed),
        "test_data_used": False,
        "epochs": evidence,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _write_history(path: Path, history: Sequence[Mapping[str, Any]]) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".pt", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(dict(payload), temporary)
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def validate_paired_baseline(
    selection_path: str | Path,
    *,
    grounded_arm: str,
    seed_base: int,
    fold: int,
    effective_seed: int,
    paired_initialization_sha256: str,
    hyperparameters: TrainHyperparameters,
    train_patient_sha256: str,
    val_patient_sha256: str,
    data_provenance_sha256: str,
) -> tuple[float, dict[str, Any]]:
    path = Path(selection_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_arm = "L1" if grounded_arm == "L3" else "N1"
    expected = {
        "arm": expected_arm,
        "seed_base": int(seed_base),
        "fold": int(fold),
        "effective_seed": int(effective_seed),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"paired baseline selection {key} mismatch")
    if payload.get("selection_mode") != "primary" or payload.get("experiment_pass") is not True:
        raise ValueError("paired no-ground baseline must have a primary passing selection")
    if payload.get("test_data_used") is not False:
        raise ValueError("paired baseline selection is not test blind")
    paired_contract = {
        "paired_initialization_sha256": paired_initialization_sha256,
        "hyperparameters": asdict(hyperparameters),
        "train_patient_sha256": train_patient_sha256,
        "val_patient_sha256": val_patient_sha256,
        "data_provenance_sha256": data_provenance_sha256,
    }
    for key, value in paired_contract.items():
        if payload.get(key) != value:
            raise ValueError(f"paired baseline {key} mismatch")
    metric = float(payload["selected_validation_state_loss"])
    if not math.isfinite(metric) or metric <= 0:
        raise ValueError("paired baseline state loss must be finite and positive")
    return metric, payload


def train_epochs(
    *,
    arm: str,
    seed_base: int,
    fold: int,
    model: torch.nn.Module,
    objective: torch.nn.Module,
    train_dataset: StageBDataset,
    val_dataset: StageBDataset,
    device: torch.device,
    output_dir: str | Path,
    authorization: StageAAuthorization,
    hyperparameters: TrainHyperparameters,
    paired_initialization_sha256: str,
    data_provenance: Mapping[str, Any],
    paired_baseline_selection: str | Path | None = None,
) -> dict[str, Any]:
    """Run one arm/seed/fold. Callers must gate and construct safe datasets first."""

    spec = arm_spec(arm)
    effective_seed = validate_seed_fold(seed_base, fold)
    hyperparameters.validate()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to mix or overwrite an existing Stage B run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if common_initialization_sha256(model) != paired_initialization_sha256:
        raise AssertionError("run model does not match the four-arm paired initialization hash")
    initial_transition_sha256 = transition_sha256(model)
    train_patient_sha256 = canonical_sha256(sorted(train_dataset.patient_ids))
    val_patient_sha256 = canonical_sha256(sorted(val_dataset.patient_ids))
    data_provenance_sha256 = canonical_sha256(data_provenance)
    global_fallback_restart = bool(data_provenance.get("global_fallback_restart", False))
    physical_pair = (
        hyperparameters.physical_batch_size,
        hyperparameters.accumulation_steps,
    )
    if global_fallback_restart != (physical_pair == (2, 16)):
        raise ValueError("global fallback provenance disagrees with the physical batch contract")
    if spec.grounded:
        if paired_baseline_selection is None:
            raise ValueError("L3/N3 require their matching selected L1/N1 baseline")
        baseline_state, baseline_payload = validate_paired_baseline(
            paired_baseline_selection,
            grounded_arm=spec.name,
            seed_base=seed_base,
            fold=fold,
            effective_seed=effective_seed,
            paired_initialization_sha256=paired_initialization_sha256,
            hyperparameters=hyperparameters,
            train_patient_sha256=train_patient_sha256,
            val_patient_sha256=val_patient_sha256,
            data_provenance_sha256=data_provenance_sha256,
        )
    else:
        if paired_baseline_selection is not None:
            raise ValueError("L1/N1 must not receive a baseline selection")
        baseline_state, baseline_payload = None, None
    model.to(device)
    objective.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparameters.learning_rate,
        weight_decay=hyperparameters.weight_decay,
    )
    history: list[dict[str, Any]] = []
    stale = 0
    running_best: tuple[float, float] = (math.inf, math.inf)
    for epoch in range(1, hyperparameters.epochs + 1):
        logical = logical_patient_batches(train_dataset.patient_ids, effective_seed, epoch)
        logical_order = tuple(patient for batch in logical for patient in batch)
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
            hyperparameters.physical_batch_size,
            hyperparameters.workers,
        )
        row: dict[str, Any] = {
            "epoch": epoch,
            "arm": spec.name,
            "seed_base": int(seed_base),
            "fold": int(fold),
            "effective_seed": effective_seed,
            "patient_order_sha256": ordered_patient_sha256(logical_order),
            "dropped_logical_tail_patients": len(train_dataset) - len(logical_order),
            "train_loss": train_stats["loss"],
            "train_base_loss": train_stats["base_loss"],
            "train_state_loss": train_stats["state_loss"],
            "train_ftv_loss": train_stats["ftv_loss"],
            "train_grounded_patients": train_stats["grounded_patients"],
            "train_representation_std": train_stats["representation_std"],
            "train_optimizer_steps": train_stats["optimizer_steps"],
            "val_loss": val_stats["loss"],
            "val_base_objective": val_stats["base_loss"],
            "val_state_loss": val_stats["state_loss"],
            "val_ftv_loss": val_stats["ftv_loss"],
            "val_grounded_patients": val_stats["grounded_patients"],
            "val_representation_std": val_stats["representation_std"],
            "finite": all(
                math.isfinite(float(value))
                for value in (
                    train_stats["loss"], val_stats["state_loss"], val_stats["representation_std"]
                )
            ),
        }
        history.append(row)
        checkpoint_payload = {
            "schema_version": 1,
            "stage": "B",
            "arm": spec.name,
            "input_kind": spec.input_kind,
            "grounded": spec.grounded,
            "seed_base": int(seed_base),
            "fold": int(fold),
            "effective_seed": effective_seed,
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "model_config": model.model_config(),
            "architecture_contract": model.architecture_contract(),
            "optimizer_state": optimizer.state_dict(),
            "hyperparameters": asdict(hyperparameters),
            "paired_initialization_sha256": paired_initialization_sha256,
            "transition_initialization_sha256": initial_transition_sha256,
            "stage_a_sentinel_path": str(authorization.path),
            "stage_a_sentinel_sha256": authorization.sha256,
            "train_patient_sha256": train_patient_sha256,
            "val_patient_sha256": val_patient_sha256,
            "data_provenance_sha256": data_provenance_sha256,
            "global_fallback_restart": global_fallback_restart,
            "test_data_used": False,
            "data_provenance": dict(data_provenance),
            "paired_baseline_selection": baseline_payload,
            "epoch_metrics": row,
        }
        _atomic_torch_save(output / f"epoch_{epoch:02d}.pt", checkpoint_payload)
        _write_history(output / "history.csv", history)
        noncollapse = row["finite"] and row["val_representation_std"] >= hyperparameters.min_representation_std
        ftv_finite = math.isfinite(row["val_ftv_loss"]) and row["val_grounded_patients"] > 0
        if spec.grounded:
            violation = max(0.0, row["val_state_loss"] - 1.05 * float(baseline_state))
            current = (violation, row["val_ftv_loss"] if ftv_finite else math.inf)
        else:
            current = (0.0, row["val_state_loss"])
        improved = noncollapse and current < running_best
        if improved:
            running_best = current
            stale = 0
        else:
            stale += 1
        if stale >= hyperparameters.patience:
            break
    selection = select_checkpoint(
        history,
        grounded=spec.grounded,
        min_representation_std=hyperparameters.min_representation_std,
        paired_baseline_state_loss=baseline_state,
    )
    selection.update(
        {
            "arm": spec.name,
            "seed_base": int(seed_base),
            "fold": int(fold),
            "effective_seed": effective_seed,
            "paired_initialization_sha256": paired_initialization_sha256,
            "stage_a_sentinel_sha256": authorization.sha256,
            "hyperparameters": asdict(hyperparameters),
            "train_patient_sha256": train_patient_sha256,
            "val_patient_sha256": val_patient_sha256,
            "data_provenance_sha256": data_provenance_sha256,
            "global_fallback_restart": global_fallback_restart,
            "history_sha256": file_sha256(output / "history.csv"),
        }
    )
    selection_path = output / "selection.json"
    _atomic_json(selection_path, selection)
    selected_epoch = int(selection["selected_epoch"])
    selected_payload = torch.load(
        output / f"epoch_{selected_epoch:02d}.pt", map_location="cpu", weights_only=True
    )
    selected_payload["selected"] = True
    selected_payload["selection"] = selection
    selected_payload["selection_path"] = str(selection_path)
    selected_payload["selection_sha256"] = file_sha256(selection_path)
    _atomic_torch_save(output / "selected.pt", selected_payload)
    return selection


__all__ = [
    "TrainHyperparameters",
    "logical_patient_batches",
    "logical_sigreg_surrogate",
    "physical_patient_batches",
    "run_logical_train_epoch",
    "run_validation_epoch",
    "scale_microbatch_components",
    "select_checkpoint",
    "train_epochs",
    "validate_paired_baseline",
]

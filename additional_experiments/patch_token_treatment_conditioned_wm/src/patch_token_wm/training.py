"""Exact logical-batch training for the preregistered A1 patch-token model.

The physical loader may materialize four patients at a time, but patch loss,
SIGReg, FTV patient means, gradient clipping, the optimizer step, and the EMA
update all retain the locked logical-patient batch of 32.  Validation uses one
fixed outcome-blind mask schedule at every epoch.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import csv
import hashlib
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

from .contracts import TransitionCondition


LOGICAL_BATCH_SIZE = 32


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
        if self.physical_batch_size * self.accumulation_steps != LOGICAL_BATCH_SIZE:
            raise ValueError("physical batch times accumulation must equal 32")
        if self.physical_batch_size <= 0 or self.accumulation_steps <= 0:
            raise ValueError("batch sizes must be positive")
        if self.workers < 0 or self.epochs <= 0 or self.patience <= 0:
            raise ValueError("workers must be nonnegative and epochs/patience positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer hyperparameters are invalid")
        if not 0.0 < self.ema_momentum < 1.0:
            raise ValueError("EMA momentum must lie in (0,1)")
        if self.max_grad_norm <= 0 or self.min_representation_std <= 0:
            raise ValueError("gradient/std thresholds must be positive")


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def patient_set_sha256(patient_ids: Iterable[str]) -> str:
    identities = tuple(str(value) for value in patient_ids)
    if not identities or len(set(identities)) != len(identities):
        raise ValueError("patient set must be nonempty and unique")
    return canonical_sha256(sorted(identities))


def ordered_patient_sha256(patient_ids: Iterable[str]) -> str:
    identities = tuple(str(value) for value in patient_ids)
    if not identities or len(set(identities)) != len(identities):
        raise ValueError("patient order must be nonempty and unique")
    return canonical_sha256(list(identities))


def logical_patient_batches(
    patient_ids: Iterable[str], effective_seed: int, epoch: int
) -> tuple[tuple[str, ...], ...]:
    """Shuffle once and drop only the shared tail below a full 32 patients."""

    identities = tuple(str(value) for value in patient_ids)
    if not identities or len(set(identities)) != len(identities):
        raise ValueError("logical sampler requires nonempty unique patient IDs")
    generator = np.random.default_rng(
        np.random.SeedSequence([int(effective_seed), int(epoch)])
    )
    order = generator.permutation(len(identities))
    usable = len(identities) // LOGICAL_BATCH_SIZE * LOGICAL_BATCH_SIZE
    shuffled = tuple(identities[int(index)] for index in order[:usable])
    return tuple(
        shuffled[start : start + LOGICAL_BATCH_SIZE]
        for start in range(0, usable, LOGICAL_BATCH_SIZE)
    )


def physical_patient_batches(
    logical_batches: Sequence[Sequence[str]], physical_batch_size: int
) -> tuple[tuple[str, ...], ...]:
    if physical_batch_size <= 0 or LOGICAL_BATCH_SIZE % int(physical_batch_size):
        raise ValueError("physical batch must be a positive divisor of 32")
    output: list[tuple[str, ...]] = []
    for logical in logical_batches:
        identities = tuple(str(value) for value in logical)
        if len(identities) != LOGICAL_BATCH_SIZE:
            raise ValueError("every training logical batch must contain 32 patients")
        output.extend(
            identities[start : start + physical_batch_size]
            for start in range(0, LOGICAL_BATCH_SIZE, physical_batch_size)
        )
    return tuple(output)


def _chunked(values: Sequence[str], size: int) -> tuple[tuple[str, ...], ...]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    return tuple(
        tuple(values[start : start + size]) for start in range(0, len(values), size)
    )


def _loader(dataset: Any, batches: Sequence[Sequence[str]], workers: int) -> DataLoader:
    lookup = {str(patient): index for index, patient in enumerate(dataset.patient_ids)}
    indices: list[list[int]] = []
    for batch in batches:
        try:
            indices.append([lookup[str(patient)] for patient in batch])
        except KeyError as error:
            raise KeyError("batch identity is absent from the dataset") from error
    return DataLoader(
        dataset,
        batch_sampler=indices,
        num_workers=int(workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(workers),
        prefetch_factor=1 if workers else None,
        multiprocessing_context="spawn" if workers else None,
    )


def transition_condition_to_device(
    batch: Mapping[str, Any], device: torch.device
) -> TransitionCondition:
    values = batch.get("condition", batch)
    if not isinstance(values, Mapping):
        raise TypeError("collated transition condition must be a mapping")
    required = {"arm_index", "clinical", "temporal_bits", "delta_t"}
    if set(values) != required:
        raise ValueError("collated transition-condition keys drifted")
    condition = TransitionCondition(
        arm_index=values["arm_index"].to(device=device, non_blocking=True),
        clinical=values["clinical"].to(device=device, non_blocking=True),
        temporal_bits=values["temporal_bits"].to(device=device, non_blocking=True),
        delta_t=values["delta_t"].to(device=device, non_blocking=True),
    )
    return condition


def scale_logical_components(
    base_component: torch.Tensor,
    ftv_component_raw: torch.Tensor,
    *,
    microbatch_size: int,
    logical_batch_size: int,
    microbatch_ftv_patients: int,
    logical_ftv_patients: int,
    lambda_ftv: float,
) -> torch.Tensor:
    """Compose an exact logical patient mean from physical-batch means."""

    if microbatch_size <= 0 or logical_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if not 0 <= microbatch_ftv_patients <= microbatch_size:
        raise ValueError("microbatch FTV count is invalid")
    if not 0 <= logical_ftv_patients <= logical_batch_size:
        raise ValueError("logical FTV count is invalid")
    base = base_component * (float(microbatch_size) / float(logical_batch_size))
    if logical_ftv_patients:
        ftv = ftv_component_raw * (
            float(microbatch_ftv_patients) / float(logical_ftv_patients)
        )
    else:
        ftv = ftv_component_raw * 0.0
    return base + float(lambda_ftv) * ftv


def logical_sigreg_surrogate(
    current_state: torch.Tensor,
    reference_state: torch.Tensor,
    reference_gradient: torch.Tensor,
    logical_sigreg_loss: torch.Tensor,
    *,
    logical_batch_size: int,
) -> torch.Tensor:
    """Return a microbatch scalar with the exact 32-patient SIGReg gradient."""

    if (
        current_state.shape != reference_state.shape
        or current_state.shape != reference_gradient.shape
    ):
        raise ValueError("SIGReg state/reference/gradient shapes differ")
    microbatch_size = int(current_state.shape[0])
    if microbatch_size <= 0 or logical_batch_size < microbatch_size:
        raise ValueError("SIGReg batch sizes are invalid")
    multiplier = float(logical_batch_size) / float(microbatch_size)
    correction = ((current_state - reference_state) * reference_gradient).sum()
    return logical_sigreg_loss.detach() + multiplier * correction


def _fork_devices(device: torch.device) -> list[int]:
    if device.type != "cuda":
        return []
    return [torch.cuda.current_device() if device.index is None else int(device.index)]


def _batch_to_reference_state(
    model: torch.nn.Module, batch: Mapping[str, Any], device: torch.device
) -> torch.Tensor:
    image = batch["image"].to(device=device, non_blocking=True)
    state = model.encode_sigreg_state(image)
    if state.ndim != 3 or int(state.shape[1]) != 4:
        raise ValueError("SIGReg state must have shape [B,4,D]")
    return state


def logical_sigreg_reference(
    model: torch.nn.Module,
    objective: torch.nn.Module,
    retained_batches: Sequence[Mapping[str, Any]],
    device: torch.device,
    *,
    direction_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    references: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in retained_batches:
            references.append(_batch_to_reference_state(model, batch, device).detach())
    reference_bvd = torch.cat(references, dim=0)
    if int(reference_bvd.shape[0]) != LOGICAL_BATCH_SIZE:
        raise ValueError("training SIGReg reference must contain exactly 32 patients")
    reference_vbd = reference_bvd.transpose(0, 1).detach().requires_grad_(True)
    devices = _fork_devices(device)
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(direction_seed))
        if devices:
            with torch.cuda.device(devices[0]):
                torch.cuda.manual_seed(int(direction_seed))
        logical_loss = objective.sigreg(reference_vbd)
    (gradient_vbd,) = torch.autograd.grad(logical_loss, reference_vbd)
    if not bool(torch.isfinite(logical_loss)) or not bool(
        torch.isfinite(gradient_vbd).all()
    ):
        raise FloatingPointError("logical SIGReg value/gradient is non-finite")
    return (
        logical_loss.detach(),
        reference_vbd.detach().transpose(0, 1),
        gradient_vbd.detach().transpose(0, 1),
    )


def _ftv_patient_count(batch: Mapping[str, Any]) -> int:
    mask = batch["ftv_mask"]
    if not isinstance(mask, torch.Tensor) or mask.ndim != 2 or int(mask.shape[1]) != 4:
        raise ValueError("FTV mask must have shape [B,4]")
    return int(mask.bool().any(dim=1).sum().item())


def _patient_ids(batch: Mapping[str, Any]) -> tuple[str, ...]:
    values = tuple(str(value) for value in batch["patient_id"])
    if not values or len(set(values)) != len(values):
        raise ValueError("physical batch patient IDs must be nonempty and unique")
    return values


def run_logical_train_epoch(
    model: torch.nn.Module,
    objective: torch.nn.Module,
    dataset: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    logical_batches: Sequence[Sequence[str]],
    hyperparameters: TrainHyperparameters,
    *,
    effective_seed: int,
    epoch: int,
) -> dict[str, float]:
    """Train with one clip/step/EMA and one exact SIGReg per 32 patients."""

    hyperparameters.validate()
    physical = physical_patient_batches(
        logical_batches, hyperparameters.physical_batch_size
    )
    loader = _loader(dataset, physical, hyperparameters.workers)
    iterator = iter(loader)
    model.train(True)
    objective.train(True)
    sums: dict[str, float] = defaultdict(float)
    state_values: list[torch.Tensor] = []
    samples = steps = microbatches = 0
    started = time.monotonic()
    for logical_index, _logical in enumerate(logical_batches):
        retained: list[Mapping[str, Any]] = []
        for _ in range(hyperparameters.accumulation_steps):
            try:
                retained.append(next(iterator))
            except StopIteration as error:
                raise RuntimeError("loader ended inside a logical batch") from error
        direction_seed = (
            int(effective_seed) * 1_000_003 + int(epoch) * 10_007 + int(logical_index)
        ) % (2**63 - 1)
        logical_sigreg, reference, reference_gradient = logical_sigreg_reference(
            model, objective, retained, device, direction_seed=direction_seed
        )
        logical_ftv_patients = sum(_ftv_patient_count(batch) for batch in retained)
        optimizer.zero_grad(set_to_none=True)
        offset = 0
        observed_ftv = 0
        for batch in retained:
            image = batch["image"].to(device=device, non_blocking=True)
            ftv_target = batch["ftv_target"].to(device=device, non_blocking=True)
            ftv_mask = batch["ftv_mask"].to(device=device, non_blocking=True)
            condition = transition_condition_to_device(batch, device)
            identities = _patient_ids(batch)
            output = model(
                image,
                condition,
                patient_ids=identities,
                mask_seed=int(effective_seed),
                epoch=int(epoch),
                logical_batch_index=int(logical_index),
            )
            batch_size = int(image.shape[0])
            stop = offset + batch_size
            reference_slice = reference[offset:stop]
            gradient_slice = reference_gradient[offset:stop]
            if not torch.allclose(
                output.sigreg_state.detach(), reference_slice, rtol=1e-5, atol=1e-6
            ):
                raise RuntimeError("online token state changed between SIGReg passes")
            surrogate = logical_sigreg_surrogate(
                output.sigreg_state,
                reference_slice,
                gradient_slice,
                logical_sigreg,
                logical_batch_size=LOGICAL_BATCH_SIZE,
            )
            _, stats = objective(
                output, ftv_target, ftv_mask, sigreg_override=surrogate
            )
            base_component = (
                stats["_patch_component"] + stats["_sigreg_component_weighted"]
            )
            micro_ftv = int(stats["ftv_patients"].item())
            observed_ftv += micro_ftv
            loss = scale_logical_components(
                base_component,
                stats["_ftv_component_raw"],
                microbatch_size=batch_size,
                logical_batch_size=LOGICAL_BATCH_SIZE,
                microbatch_ftv_patients=micro_ftv,
                logical_ftv_patients=logical_ftv_patients,
                lambda_ftv=float(objective.lambda_ftv),
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("training objective is non-finite")
            loss.backward()
            state_values.append(output.sigreg_state.detach().float().cpu())
            samples += batch_size
            microbatches += 1
            sums["patch_loss"] += float(stats["patch_loss"]) * batch_size
            sums["sigreg_loss"] += float(logical_sigreg) * batch_size
            sums["base_loss"] += (
                float(stats["patch_loss"]) + float(stats["weighted_sigreg_loss"])
            ) * batch_size
            sums["ftv_numerator"] += float(stats["ftv_loss"]) * micro_ftv
            sums["ftv_patients"] += micro_ftv
            sums["ftv_valid_visits"] += float(stats["ftv_valid_visits"])
            sums["masked_tokens"] += float(stats["masked_tokens"])
            offset = stop
        if offset != LOGICAL_BATCH_SIZE or observed_ftv != logical_ftv_patients:
            raise AssertionError("logical patient/FTV counts failed to reconcile")
        grad_norm = float(
            clip_grad_norm_(
                model.parameters(),
                hyperparameters.max_grad_norm,
                error_if_nonfinite=True,
            )
        )
        optimizer.step()
        model.update_target(hyperparameters.ema_momentum)
        optimizer.zero_grad(set_to_none=True)
        sums["gradient_norm"] += grad_norm
        steps += 1
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise RuntimeError("training loader produced extra microbatches")
    expected_microbatches = len(logical_batches) * hyperparameters.accumulation_steps
    if (
        not samples
        or steps != len(logical_batches)
        or microbatches != expected_microbatches
    ):
        raise RuntimeError("training logical-batch accounting failed")
    ftv_patients = sums["ftv_patients"]
    ftv_loss = sums["ftv_numerator"] / max(ftv_patients, 1.0)
    state = torch.cat(state_values, dim=0)
    result = {
        "patch_loss": sums["patch_loss"] / samples,
        "sigreg_loss": sums["sigreg_loss"] / samples,
        "base_loss": sums["base_loss"] / samples,
        "ftv_loss": ftv_loss,
        "weighted_ftv_loss": float(objective.lambda_ftv) * ftv_loss,
        "loss": sums["base_loss"] / samples + float(objective.lambda_ftv) * ftv_loss,
        "representation_std": float(state.std(dim=0, unbiased=False).mean()),
        "grounded_patients": ftv_patients,
        "valid_ftv_visits": sums["ftv_valid_visits"],
        "masked_tokens": sums["masked_tokens"],
        "gradient_norm": sums["gradient_norm"] / steps,
        "samples": float(samples),
        "logical_batches": float(steps),
        "physical_microbatches": float(microbatches),
        "optimizer_steps": float(steps),
        "ema_updates": float(steps),
        "seconds": time.monotonic() - started,
    }
    if not all(math.isfinite(float(value)) for value in result.values()):
        raise FloatingPointError("training summary contains a non-finite value")
    return result


@torch.no_grad()
def run_validation_epoch(
    model: torch.nn.Module,
    objective: torch.nn.Module,
    dataset: Any,
    device: torch.device,
    *,
    physical_batch_size: int,
    workers: int,
    effective_seed: int,
) -> dict[str, float]:
    """Validate using fixed masks and one SIGReg evaluation per patient block."""

    patient_ids = tuple(str(value) for value in dataset.patient_ids)
    logical_batches = _chunked(patient_ids, LOGICAL_BATCH_SIZE)
    physical: list[tuple[str, ...]] = []
    for logical in logical_batches:
        physical.extend(_chunked(logical, int(physical_batch_size)))
    iterator = iter(_loader(dataset, physical, workers))
    model.train(False)
    objective.train(False)
    sums: dict[str, float] = defaultdict(float)
    state_values: list[torch.Tensor] = []
    samples = 0
    for logical_index, logical in enumerate(logical_batches):
        logical_size = len(logical)
        expected_micro = math.ceil(logical_size / int(physical_batch_size))
        records: list[tuple[Any, torch.Tensor, torch.Tensor, int]] = []
        states: list[torch.Tensor] = []
        seen = 0
        for _ in range(expected_micro):
            try:
                batch = next(iterator)
            except StopIteration as error:
                raise RuntimeError(
                    "validation loader ended inside logical block"
                ) from error
            image = batch["image"].to(device=device, non_blocking=True)
            ftv_target = batch["ftv_target"].to(device=device, non_blocking=True)
            ftv_mask = batch["ftv_mask"].to(device=device, non_blocking=True)
            condition = transition_condition_to_device(batch, device)
            output = model(
                image,
                condition,
                patient_ids=_patient_ids(batch),
                mask_seed=int(effective_seed),
                epoch=0,
                logical_batch_index=int(logical_index),
            )
            batch_size = int(image.shape[0])
            records.append((output, ftv_target, ftv_mask, batch_size))
            states.append(output.sigreg_state.detach())
            state_values.append(output.sigreg_state.detach().float().cpu())
            seen += batch_size
        if seen != logical_size:
            raise RuntimeError("validation physical batches crossed a logical boundary")
        logical_sigreg = objective.sigreg(torch.cat(states, dim=0).transpose(0, 1))
        patch_numerator = base_numerator = ftv_numerator = 0.0
        ftv_patients = 0
        ftv_visits = masked_tokens = 0.0
        for output, ftv_target, ftv_mask, batch_size in records:
            _, stats = objective(
                output, ftv_target, ftv_mask, sigreg_override=logical_sigreg
            )
            grounded = int(stats["ftv_patients"].item())
            patch_numerator += float(stats["patch_loss"]) * batch_size
            base_numerator += (
                float(stats["patch_loss"]) + float(stats["weighted_sigreg_loss"])
            ) * batch_size
            ftv_numerator += float(stats["ftv_loss"]) * grounded
            ftv_patients += grounded
            ftv_visits += float(stats["ftv_valid_visits"])
            masked_tokens += float(stats["masked_tokens"])
        block_patch = patch_numerator / logical_size
        block_base = base_numerator / logical_size
        block_ftv = ftv_numerator / max(ftv_patients, 1)
        total = block_base + float(objective.lambda_ftv) * block_ftv
        if not math.isfinite(total):
            raise FloatingPointError("validation objective is non-finite")
        samples += logical_size
        sums["patch_loss"] += block_patch * logical_size
        sums["sigreg_loss"] += float(logical_sigreg) * logical_size
        sums["base_loss"] += block_base * logical_size
        sums["ftv_numerator"] += ftv_numerator
        sums["ftv_patients"] += ftv_patients
        sums["ftv_valid_visits"] += ftv_visits
        sums["masked_tokens"] += masked_tokens
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise RuntimeError("validation loader produced extra microbatches")
    if not samples:
        raise RuntimeError("validation cohort is empty")
    ftv_loss = sums["ftv_numerator"] / max(sums["ftv_patients"], 1.0)
    state = torch.cat(state_values, dim=0)
    result = {
        "patch_loss": sums["patch_loss"] / samples,
        "sigreg_loss": sums["sigreg_loss"] / samples,
        "base_loss": sums["base_loss"] / samples,
        "ftv_loss": ftv_loss,
        "weighted_ftv_loss": float(objective.lambda_ftv) * ftv_loss,
        "loss": sums["base_loss"] / samples + float(objective.lambda_ftv) * ftv_loss,
        "representation_std": float(state.std(dim=0, unbiased=False).mean()),
        "grounded_patients": sums["ftv_patients"],
        "valid_ftv_visits": sums["ftv_valid_visits"],
        "masked_tokens": sums["masked_tokens"],
        "samples": float(samples),
    }
    if not all(math.isfinite(float(value)) for value in result.values()):
        raise FloatingPointError("validation summary contains a non-finite value")
    return result


def select_checkpoint(
    epochs: Sequence[Mapping[str, Any]], *, min_representation_std: float
) -> dict[str, Any]:
    """Minimum finite non-collapsed validation patch loss, then FTV, then epoch."""

    if not epochs:
        raise ValueError("checkpoint selection requires at least one epoch")
    evidence: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for raw in epochs:
        row = dict(raw)
        finite = bool(row.get("finite")) and all(
            math.isfinite(float(row[name]))
            for name in ("val_patch_loss", "val_ftv_loss", "val_representation_std")
        )
        noncollapsed = finite and float(row["val_representation_std"]) >= float(
            min_representation_std
        )
        row["noncollapsed"] = noncollapsed
        row["checkpoint_eligible"] = noncollapsed
        evidence.append(row)
        if noncollapsed:
            eligible.append(row)
    if not eligible:
        raise RuntimeError("no finite non-collapsed validation checkpoint exists")
    best_patch = min(float(row["val_patch_loss"]) for row in eligible)
    patch_tied = [
        row for row in eligible if float(row["val_patch_loss"]) <= best_patch + 1e-12
    ]
    best_ftv = min(float(row["val_ftv_loss"]) for row in patch_tied)
    selected = min(
        (row for row in patch_tied if float(row["val_ftv_loss"]) <= best_ftv + 1e-12),
        key=lambda row: int(row["epoch"]),
    )
    return {
        "schema_version": 1,
        "selection_rule": (
            "minimum finite noncollapsed validation patch loss; validation FTV "
            "within 1e-12 tie; then earlier epoch"
        ),
        "selected_epoch": int(selected["epoch"]),
        "selected_validation_patch_loss": float(selected["val_patch_loss"]),
        "selected_validation_ftv_loss": float(selected["val_ftv_loss"]),
        "selected_representation_std": float(selected["val_representation_std"]),
        "optimization_safety_pass": True,
        "test_data_used": False,
        "pcr_loaded": False,
        "epochs": evidence,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".pt", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        os.chmod(temporary, 0o600)
        torch.save(dict(payload), temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_history(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def train_epochs(
    *,
    seed_base: int,
    fold: int,
    model: torch.nn.Module,
    objective: torch.nn.Module,
    train_dataset: Any,
    val_dataset: Any,
    device: torch.device,
    output_dir: str | Path,
    hyperparameters: TrainHyperparameters,
    preregistration_lock_sha256: str,
    data_provenance: Mapping[str, Any],
    condition_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Run one test-blind A1 seed/fold cell and seal its selected checkpoint."""

    if int(seed_base) not in (2026, 3026) or int(fold) not in range(5):
        raise ValueError("formal seed/fold is outside the preregistered matrix")
    effective_seed = int(seed_base) + int(fold)
    hyperparameters.validate()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite or mix a run at {output}")
    output.mkdir(parents=True, exist_ok=True)
    model.to(device)
    objective.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparameters.learning_rate,
        weight_decay=hyperparameters.weight_decay,
    )
    history: list[dict[str, Any]] = []
    stale = 0
    running_best = (math.inf, math.inf, math.inf)
    for epoch in range(1, hyperparameters.epochs + 1):
        logical = logical_patient_batches(
            train_dataset.patient_ids, effective_seed, epoch
        )
        order = tuple(patient for block in logical for patient in block)
        train = run_logical_train_epoch(
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
        validation = run_validation_epoch(
            model,
            objective,
            val_dataset,
            device,
            physical_batch_size=hyperparameters.physical_batch_size,
            workers=hyperparameters.workers,
            effective_seed=effective_seed,
        )
        row = {
            "epoch": epoch,
            "seed_base": int(seed_base),
            "fold": int(fold),
            "effective_seed": effective_seed,
            "patient_order_sha256": ordered_patient_sha256(order),
            "dropped_logical_tail_patients": len(train_dataset) - len(order),
            "train_loss": train["loss"],
            "train_patch_loss": train["patch_loss"],
            "train_sigreg_loss": train["sigreg_loss"],
            "train_ftv_loss": train["ftv_loss"],
            "train_representation_std": train["representation_std"],
            "train_optimizer_steps": train["optimizer_steps"],
            "train_seconds": train["seconds"],
            "val_loss": validation["loss"],
            "val_patch_loss": validation["patch_loss"],
            "val_sigreg_loss": validation["sigreg_loss"],
            "val_ftv_loss": validation["ftv_loss"],
            "val_representation_std": validation["representation_std"],
            "finite": all(
                math.isfinite(float(value))
                for value in (
                    train["loss"],
                    validation["loss"],
                    validation["patch_loss"],
                    validation["ftv_loss"],
                    validation["representation_std"],
                )
            ),
        }
        history.append(row)
        checkpoint = {
            "schema_version": 1,
            "experiment": "patch_token_treatment_conditioned_wm",
            "arm": "A1_PATCH3",
            "seed_base": int(seed_base),
            "fold": int(fold),
            "effective_seed": effective_seed,
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "model_config": model.model_config(),
            "hyperparameters": asdict(hyperparameters),
            "preregistration_lock_sha256": str(preregistration_lock_sha256),
            "train_patient_sha256": patient_set_sha256(train_dataset.patient_ids),
            "val_patient_sha256": patient_set_sha256(val_dataset.patient_ids),
            "data_provenance": dict(data_provenance),
            "condition_metadata": dict(condition_metadata),
            "test_data_used": False,
            "pcr_loaded": False,
            "epoch_metrics": row,
        }
        _atomic_torch(output / f"epoch_{epoch:02d}.pt", checkpoint)
        _write_history(output / "history.csv", history)
        if (
            row["finite"]
            and row["val_representation_std"] >= hyperparameters.min_representation_std
        ):
            current = (
                float(row["val_patch_loss"]),
                float(row["val_ftv_loss"]),
                float(epoch),
            )
            if current[0] < running_best[0] - 1e-12 or (
                abs(current[0] - running_best[0]) <= 1e-12
                and current[1] < running_best[1] - 1e-12
            ):
                running_best = current
                stale = 0
            else:
                stale += 1
        else:
            stale += 1
        if stale >= hyperparameters.patience:
            break
    selection = select_checkpoint(
        history, min_representation_std=hyperparameters.min_representation_std
    )
    selection.update(
        {
            "arm": "A1_PATCH3",
            "seed_base": int(seed_base),
            "fold": int(fold),
            "effective_seed": effective_seed,
            "preregistration_lock_sha256": str(preregistration_lock_sha256),
            "hyperparameters": asdict(hyperparameters),
            "train_patient_sha256": patient_set_sha256(train_dataset.patient_ids),
            "val_patient_sha256": patient_set_sha256(val_dataset.patient_ids),
            "history_sha256": file_sha256(output / "history.csv"),
        }
    )
    selection_path = output / "selection.json"
    _atomic_json(selection_path, selection)
    selected_epoch = int(selection["selected_epoch"])
    selected = torch.load(
        output / f"epoch_{selected_epoch:02d}.pt", map_location="cpu", weights_only=True
    )
    selected["selected"] = True
    selected["selection"] = selection
    selected["selection_sha256"] = file_sha256(selection_path)
    _atomic_torch(output / "selected.pt", selected)
    return selection


__all__ = [
    "LOGICAL_BATCH_SIZE",
    "TrainHyperparameters",
    "canonical_sha256",
    "file_sha256",
    "logical_patient_batches",
    "logical_sigreg_reference",
    "logical_sigreg_surrogate",
    "ordered_patient_sha256",
    "patient_set_sha256",
    "physical_patient_batches",
    "run_logical_train_epoch",
    "run_validation_epoch",
    "scale_logical_components",
    "select_checkpoint",
    "train_epochs",
    "transition_condition_to_device",
]

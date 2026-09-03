"""Logical-B32, pCR-blind training for S1/S2.

This module locally generalizes the sealed Stage-B accumulation arithmetic to
two loss-side targets.  It does not modify any upstream implementation.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import csv
from dataclasses import asdict
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
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

from .contracts import (
    CHECKPOINT_SELECTION_RULE,
    LOGICAL_BATCH_SIZE,
    TrainHyperparameters,
    arm_spec,
    assert_representation_schema,
    canonical_sha256,
    file_sha256,
    validate_seed_fold,
)
from .model import (
    ResidualSPHWorldModel,
    base_initialization_sha256,
    shared_initialization_sha256,
    tensor_state_sha256,
    validate_model_contract,
)
from .upstream import logical_patient_batches, physical_patient_batches


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def scale_microbatch_components(
    base_component: torch.Tensor,
    ftv_component_raw: torch.Tensor,
    sph_component_raw: torch.Tensor,
    *,
    microbatch_size: int,
    logical_batch_size: int,
    microbatch_ftv_patients: int,
    logical_ftv_patients: int,
    microbatch_sph_patients: int,
    logical_sph_patients: int,
    lambda_ftv: float,
    lambda_sph: float,
) -> torch.Tensor:
    """Compose exact logical patient means from physical-microbatch means."""

    if microbatch_size <= 0 or logical_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    for name, count, limit in (
        ("microbatch FTV", microbatch_ftv_patients, microbatch_size),
        ("logical FTV", logical_ftv_patients, logical_batch_size),
        ("microbatch SPH", microbatch_sph_patients, microbatch_size),
        ("logical SPH", logical_sph_patients, logical_batch_size),
    ):
        if not 0 <= int(count) <= int(limit):
            raise ValueError(f"{name} patient count is invalid")
    if microbatch_ftv_patients and not logical_ftv_patients:
        raise ValueError("microbatch has FTV patients but logical denominator is zero")
    if microbatch_sph_patients and not logical_sph_patients:
        raise ValueError("microbatch has SPH patients but logical denominator is zero")
    base = base_component * (float(microbatch_size) / float(logical_batch_size))
    ftv = (
        ftv_component_raw
        * (float(microbatch_ftv_patients) / float(logical_ftv_patients))
        if logical_ftv_patients
        else ftv_component_raw * 0.0
    )
    sph = (
        sph_component_raw
        * (float(microbatch_sph_patients) / float(logical_sph_patients))
        if logical_sph_patients
        else sph_component_raw * 0.0
    )
    return base + float(lambda_ftv) * ftv + float(lambda_sph) * sph


def _patient_has_valid(
    mapping: Mapping[str, tuple[np.ndarray, np.ndarray]], patient_id: str
) -> bool:
    pair = mapping.get(str(patient_id))
    return pair is not None and bool(np.asarray(pair[1], dtype=bool).any())


def logical_supervision_counts(
    logical_batches: Sequence[Sequence[str]],
    dataset: Any,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    ftv = tuple(
        sum(_patient_has_valid(dataset.transformed_ftv, patient_id) for patient_id in batch)
        for batch in logical_batches
    )
    sph = tuple(
        sum(_patient_has_valid(dataset.sph_targets, patient_id) for patient_id in batch)
        for batch in logical_batches
    )
    return ftv, sph


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
    expected = {"patient_id", "image", "ftv_target", "ftv_mask", "sph_target", "sph_mask"}
    if set(batch) != expected:
        raise PermissionError(f"training batch schema drifted: {sorted(batch)}")
    assert_representation_schema(batch)
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _finite_loss(loss: torch.Tensor) -> None:
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("non-finite residual-SPH training loss")


def logical_sigreg_surrogate(
    online_state: torch.Tensor,
    reference_state: torch.Tensor,
    reference_gradient: torch.Tensor,
    logical_sigreg_loss: torch.Tensor,
    *,
    logical_batch_size: int,
) -> torch.Tensor:
    if online_state.shape != reference_state.shape or online_state.shape != reference_gradient.shape:
        raise ValueError("logical SIGReg state/reference/gradient shapes differ")
    microbatch_size = int(online_state.size(0))
    if microbatch_size <= 0 or logical_batch_size < microbatch_size:
        raise ValueError("logical SIGReg batch sizes are invalid")
    multiplier = float(logical_batch_size) / float(microbatch_size)
    correction = ((online_state - reference_state) * reference_gradient).sum()
    return logical_sigreg_loss.detach() + multiplier * correction


class _LogicalSIGRegAdapter(torch.nn.Module):
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
def _temporary_sigreg(objective: torch.nn.Module, replacement: torch.nn.Module):
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
    model: ResidualSPHWorldModel,
    objective: torch.nn.Module,
    retained_batches: Sequence[Mapping[str, Any]],
    device: torch.device,
    *,
    direction_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    references: list[torch.Tensor] = []
    with torch.no_grad():
        for raw in retained_batches:
            batch = _to_device(raw, device)
            _, online, _ = model.encode_online(batch["image"], None)
            references.append(online.detach())
    reference_bvd = torch.cat(references, dim=0)
    if reference_bvd.ndim != 3 or int(reference_bvd.size(0)) != LOGICAL_BATCH_SIZE:
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
    if not bool(torch.isfinite(logical_loss)) or not bool(torch.isfinite(gradient_vbd).all()):
        raise FloatingPointError("logical SIGReg value/gradient is non-finite")
    return (
        logical_loss.detach(),
        reference_vbd.detach().transpose(0, 1),
        gradient_vbd.detach().transpose(0, 1),
    )


def run_logical_train_epoch(
    model: ResidualSPHWorldModel,
    objective: torch.nn.Module,
    dataset: Dataset,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    logical_batches: Sequence[Sequence[str]],
    hyperparameters: TrainHyperparameters,
    *,
    effective_seed: int,
    epoch: int,
) -> dict[str, float]:
    hyperparameters.validate()
    physical = physical_patient_batches(logical_batches, hyperparameters.physical_batch_size)
    expected_microbatches = len(logical_batches) * hyperparameters.accumulation_steps
    if len(physical) != expected_microbatches:
        raise AssertionError("logical/physical accumulation contract drifted")
    logical_ftv, logical_sph = logical_supervision_counts(logical_batches, dataset)
    loader = _loader(dataset, physical, hyperparameters.workers)
    model.train(True)
    objective.train(True)
    sums: dict[str, float] = defaultdict(float)
    responses: list[torch.Tensor] = []
    samples = logical_steps = microbatches = 0
    start = time.monotonic()
    iterator = iter(loader)
    for logical_index in range(len(logical_batches)):
        retained: list[Mapping[str, Any]] = []
        for _ in range(hyperparameters.accumulation_steps):
            try:
                retained.append(next(iterator))
            except StopIteration as error:
                raise RuntimeError("training loader ended inside a logical batch") from error
        direction_seed = (
            int(effective_seed) * 1_000_003 + int(epoch) * 10_007 + logical_index
        ) % (2**63 - 1)
        logical_sigreg, reference, reference_gradient = _logical_sigreg_reference(
            model, objective, retained, device, direction_seed=direction_seed
        )
        optimizer.zero_grad(set_to_none=True)
        seen_ftv = seen_sph = offset = 0
        for raw in retained:
            batch = _to_device(raw, device)
            output = model(batch["image"], None)
            batch_size = int(batch["image"].size(0))
            stop = offset + batch_size
            if not torch.allclose(
                output.online_state.detach(), reference[offset:stop], rtol=1e-5, atol=1e-6
            ):
                raise RuntimeError("online encoder changed between SIGReg reference and gradient pass")
            adapter = _LogicalSIGRegAdapter(
                reference[offset:stop],
                reference_gradient[offset:stop],
                logical_sigreg,
                LOGICAL_BATCH_SIZE,
            )
            with _temporary_sigreg(objective, adapter):
                _, stats = objective(
                    output,
                    batch["ftv_target"],
                    batch["ftv_mask"],
                    batch["sph_target"],
                    batch["sph_mask"],
                )
            micro_ftv = int(stats["ftv_patients"].item())
            micro_sph = int(stats["sph_patients"].item())
            seen_ftv += micro_ftv
            seen_sph += micro_sph
            loss = scale_microbatch_components(
                stats["_base_component"],
                stats["_ftv_component_raw"],
                stats["_sph_component_raw"],
                microbatch_size=batch_size,
                logical_batch_size=LOGICAL_BATCH_SIZE,
                microbatch_ftv_patients=micro_ftv,
                logical_ftv_patients=logical_ftv[logical_index],
                microbatch_sph_patients=micro_sph,
                logical_sph_patients=logical_sph[logical_index],
                lambda_ftv=float(objective.lambda_ftv),
                lambda_sph=float(objective.lambda_sph),
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
            sums["sph_loss_numerator"] += float(stats["sph_loss"]) * micro_sph
            sums["sph_patients"] += micro_sph
            sums["sph_valid_visits"] += float(stats["sph_valid_visits"])
            offset = stop
        if offset != LOGICAL_BATCH_SIZE:
            raise AssertionError("logical training batch did not contain 32 patients")
        if seen_ftv != logical_ftv[logical_index] or seen_sph != logical_sph[logical_index]:
            raise AssertionError("logical supervised patient numerator/count audit failed")
        gradient_norm = float(
            clip_grad_norm_(
                model.parameters(), hyperparameters.max_grad_norm, error_if_nonfinite=True
            )
        )
        optimizer.step()
        model.update_target(hyperparameters.ema_momentum)
        optimizer.zero_grad(set_to_none=True)
        sums["gradient_norm"] += gradient_norm
        logical_steps += 1
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise RuntimeError("training loader produced extra physical microbatches")
    if microbatches != expected_microbatches or logical_steps != len(logical_batches):
        raise RuntimeError("training loader did not preserve logical batching")
    if not samples:
        raise RuntimeError("training epoch is empty after shared tail truncation")
    ftv_loss = sums["ftv_loss_numerator"] / max(sums["ftv_patients"], 1.0)
    sph_loss = sums["sph_loss_numerator"] / max(sums["sph_patients"], 1.0)
    response = torch.cat(responses, dim=0)
    result = {
        "base_loss": sums["base_loss"] / samples,
        "state_loss": sums["state_loss"] / samples,
        "sigreg_loss": sums["sigreg_loss"] / samples,
        "ftv_loss": ftv_loss,
        "weighted_ftv_loss": float(objective.lambda_ftv) * ftv_loss,
        "sph_loss": sph_loss,
        "weighted_sph_loss": float(objective.lambda_sph) * sph_loss,
        "grounded_patients": sums["ftv_patients"],
        "valid_ftv_visits": sums["ftv_valid_visits"],
        "sph_patients": sums["sph_patients"],
        "valid_sph_visits": sums["sph_valid_visits"],
        "representation_std": float(response.std(dim=0, unbiased=False).mean()),
        "gradient_norm": sums["gradient_norm"] / logical_steps,
        "samples": float(samples),
        "logical_batches": float(logical_steps),
        "physical_microbatches": float(microbatches),
        "optimizer_steps": float(logical_steps),
        "ema_updates": float(logical_steps),
        "seconds": time.monotonic() - start,
    }
    result["loss"] = result["base_loss"] + result["weighted_ftv_loss"] + result["weighted_sph_loss"]
    return result


def _chunked(values: Sequence[str], size: int) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(values[start : start + size]) for start in range(0, len(values), size))


@torch.no_grad()
def run_validation_epoch(
    model: ResidualSPHWorldModel,
    objective: torch.nn.Module,
    dataset: Dataset,
    device: torch.device,
    physical_batch_size: int,
    workers: int,
) -> dict[str, float]:
    physical_batches = _chunked(tuple(getattr(dataset, "patient_ids")), int(physical_batch_size))
    logical_batches = _chunked(tuple(getattr(dataset, "patient_ids")), LOGICAL_BATCH_SIZE)
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
        states: list[torch.Tensor] = []
        records: list[tuple[Any, dict[str, Any], int]] = []
        seen = 0
        for _ in range(expected_microbatches):
            try:
                batch = _to_device(next(iterator), device)
            except StopIteration as error:
                raise RuntimeError("validation loader ended inside a logical batch") from error
            output = model(batch["image"], None)
            batch_size = int(batch["image"].size(0))
            seen += batch_size
            records.append((output, batch, batch_size))
            states.append(output.online_state.detach())
            responses.append(output.response_state.detach().float().cpu())
        if seen != logical_size:
            raise RuntimeError("validation physical batches crossed a logical boundary")
        logical_sigreg = objective.sigreg(torch.cat(states, dim=0).transpose(0, 1))
        adapter = _ConstantSIGRegAdapter(logical_sigreg)
        local: dict[str, float] = defaultdict(float)
        for output, batch, batch_size in records:
            with _temporary_sigreg(objective, adapter):
                _, stats = objective(
                    output,
                    batch["ftv_target"],
                    batch["ftv_mask"],
                    batch["sph_target"],
                    batch["sph_mask"],
                )
            ftv_count = int(stats["ftv_patients"].item())
            sph_count = int(stats["sph_patients"].item())
            local["state"] += float(stats["state_loss"]) * batch_size
            local["base"] += float(stats["base_loss"]) * batch_size
            local["ftv"] += float(stats["ftv_loss"]) * ftv_count
            local["ftv_count"] += ftv_count
            local["ftv_visits"] += float(stats["ftv_valid_visits"])
            local["sph"] += float(stats["sph_loss"]) * sph_count
            local["sph_count"] += sph_count
            local["sph_visits"] += float(stats["sph_valid_visits"])
        logical_state = local["state"] / logical_size
        logical_base = local["base"] / logical_size
        logical_ftv = local["ftv"] / max(local["ftv_count"], 1.0)
        logical_sph = local["sph"] / max(local["sph_count"], 1.0)
        logical_total = (
            logical_base
            + float(objective.lambda_ftv) * logical_ftv
            + float(objective.lambda_sph) * logical_sph
        )
        if not math.isfinite(logical_total):
            raise FloatingPointError("non-finite logical validation objective")
        samples += logical_size
        sums["state_loss"] += logical_state * logical_size
        sums["sigreg_loss"] += float(logical_sigreg) * logical_size
        sums["base_loss"] += logical_base * logical_size
        for key in ("ftv", "ftv_count", "ftv_visits", "sph", "sph_count", "sph_visits"):
            sums[key] += local[key]
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise RuntimeError("validation loader produced extra physical microbatches")
    if not samples:
        raise RuntimeError("validation cohort is empty")
    ftv_loss = sums["ftv"] / max(sums["ftv_count"], 1.0)
    sph_loss = sums["sph"] / max(sums["sph_count"], 1.0)
    response = torch.cat(responses, dim=0)
    result = {
        "base_loss": sums["base_loss"] / samples,
        "state_loss": sums["state_loss"] / samples,
        "sigreg_loss": sums["sigreg_loss"] / samples,
        "ftv_loss": ftv_loss,
        "weighted_ftv_loss": float(objective.lambda_ftv) * ftv_loss,
        "sph_loss": sph_loss,
        "weighted_sph_loss": float(objective.lambda_sph) * sph_loss,
        "grounded_patients": sums["ftv_count"],
        "valid_ftv_visits": sums["ftv_visits"],
        "sph_patients": sums["sph_count"],
        "valid_sph_visits": sums["sph_visits"],
        "representation_std": float(response.std(dim=0, unbiased=False).mean()),
        "samples": float(samples),
    }
    result["loss"] = result["base_loss"] + result["weighted_ftv_loss"] + result["weighted_sph_loss"]
    return result


def select_checkpoint(
    epochs: Sequence[Mapping[str, Any]],
    *,
    min_representation_std: float,
    paired_s0_state_loss: float,
) -> dict[str, Any]:
    """Strict LOCAL3-compatible selection; SPH cannot affect the choice."""

    if not epochs:
        raise ValueError("checkpoint selection requires at least one epoch")
    if not math.isfinite(float(paired_s0_state_loss)) or float(paired_s0_state_loss) <= 0:
        raise ValueError("selection requires a finite positive paired S0 state loss")
    allowed = 1.05 * float(paired_s0_state_loss)
    candidates: list[dict[str, Any]] = []
    fallbacks: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for raw in epochs:
        row = dict(raw)
        state = float(row["val_state_loss"])
        ftv = float(row["val_ftv_loss"])
        representation_std = float(row["val_representation_std"])
        ftv_patients = float(row.get("val_grounded_patients", 0.0))
        noncollapse = math.isfinite(representation_std) and representation_std >= min_representation_std
        base_finite = math.isfinite(state)
        ftv_finite = math.isfinite(ftv) and ftv_patients > 0
        gate = base_finite and state <= allowed
        eligible = noncollapse and gate and ftv_finite
        row.update(
            {
                "noncollapse": noncollapse,
                "base_gate_pass": gate,
                "checkpoint_eligible": eligible,
            }
        )
        evidence.append(row)
        if eligible:
            candidates.append(row)
        if noncollapse and base_finite and ftv_finite:
            fallbacks.append(row)
    if candidates:
        selected = min(
            candidates,
            key=lambda row: (
                float(row["val_ftv_loss"]),
                float(row["val_state_loss"]),
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
                float(row["val_ftv_loss"]),
                float(row["val_state_loss"]),
                int(row["epoch"]),
            ),
        )
        mode = "fallback_state_gate_failed"
        passed = False
    return {
        "schema_version": 1,
        "selection_mode": mode,
        "experiment_pass": passed,
        "selection_rule": CHECKPOINT_SELECTION_RULE,
        "fallback_rule": "minimum state-gate violation, then validation FTV loss; fallback is failure",
        "selection_excludes": [
            "validation_sph_loss",
            "test_endpoints",
            "delta_sph",
            "pcr",
            "hr",
            "her2",
            "clinical",
            "treatment",
        ],
        "paired_s0_state_loss": float(paired_s0_state_loss),
        "allowed_state_loss": allowed,
        "selected_epoch": int(selected["epoch"]),
        "selected_validation_total_loss": float(selected["val_loss"]),
        "selected_validation_base_loss": float(selected["val_base_objective"]),
        "selected_validation_state_loss": float(selected["val_state_loss"]),
        "selected_validation_ftv_loss": float(selected["val_ftv_loss"]),
        "selected_validation_sph_loss": float(selected.get("val_sph_loss", math.nan)),
        "selected_representation_std": float(selected["val_representation_std"]),
        "state_loss_degradation_fraction": (
            float(selected["val_state_loss"]) / float(paired_s0_state_loss) - 1.0
        ),
        "optimization_safety_pass": passed,
        "test_data_used": False,
        "pcr_used": False,
        "epochs": evidence,
    }


def validate_s0_anchor(
    selection_path: str | Path,
    *,
    seed_base: int,
    fold: int,
    expected_shared_initialization_sha256: str,
) -> tuple[float, dict[str, Any]]:
    payload = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    expected = {
        "arm": "LOCAL3",
        "seed_base": int(seed_base),
        "fold": int(fold),
        "effective_seed": validate_seed_fold(seed_base, fold),
        "paired_initialization_sha256": str(expected_shared_initialization_sha256),
        "test_data_used": False,
        "pcr_used": False,
        "delta_ftv_used": False,
        "selection_mode": "primary",
        "experiment_pass": True,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"confirmed S0 selection {key} mismatch")
    metric = float(payload["selected_validation_state_loss"])
    if not math.isfinite(metric) or metric <= 0:
        raise ValueError("confirmed S0 state-loss anchor is invalid")
    return metric, payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary).replace(path)
        path.chmod(0o600)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".pt", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(dict(payload), temporary)
        Path(temporary).replace(path)
        path.chmod(0o600)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _write_history(path: Path, history: Sequence[Mapping[str, Any]]) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary).replace(path)
        path.chmod(0o600)
    finally:
        Path(temporary).unlink(missing_ok=True)


def train_epochs(
    *,
    experimental_arm: str,
    seed_base: int,
    fold: int,
    model: ResidualSPHWorldModel,
    objective: torch.nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    device: torch.device,
    output_dir: str | Path,
    hyperparameters: TrainHyperparameters,
    s0_selection_path: str | Path,
    preregistration_lock_sha256: str,
    data_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Train one S1/S2 cell and persist only private run artifacts."""

    spec = arm_spec(experimental_arm)
    if spec.name == "S0":
        raise ValueError("S0 is the hash-bound confirmed LOCAL3 runtime reference")
    effective_seed = validate_seed_fold(seed_base, fold)
    hyperparameters.validate()
    validate_model_contract(model)
    if model.experimental_arm != spec.name or float(objective.lambda_sph) != spec.lambda_sph:
        raise ValueError("model/objective arm mismatch")
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty run directory: {output}")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    output.chmod(0o700)
    shared_hash = shared_initialization_sha256(model)
    base_hash = base_initialization_sha256(model)
    s0_state, s0_payload = validate_s0_anchor(
        s0_selection_path,
        seed_base=seed_base,
        fold=fold,
        expected_shared_initialization_sha256=shared_hash,
    )
    if len(str(preregistration_lock_sha256)) != 64:
        raise ValueError("formal training requires a SHA-256 preregistration lock")
    train_patient_sha256 = canonical_sha256(sorted(str(value) for value in getattr(train_dataset, "patient_ids")))
    val_patient_sha256 = canonical_sha256(sorted(str(value) for value in getattr(val_dataset, "patient_ids")))
    provenance_sha256 = canonical_sha256(data_provenance)
    transition_hash = tensor_state_sha256(model.transition.state_dict())
    model.to(device)
    objective.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparameters.learning_rate,
        weight_decay=hyperparameters.weight_decay,
    )
    history: list[dict[str, Any]] = []
    stale = 0
    running_best: tuple[float, float, float] = (math.inf, math.inf, math.inf)
    for epoch in range(1, hyperparameters.epochs + 1):
        logical = logical_patient_batches(train_dataset.patient_ids, effective_seed, epoch)
        logical_order = tuple(patient_id for batch in logical for patient_id in batch)
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
            "patient_order_sha256": canonical_sha256(logical_order),
            "dropped_logical_tail_patients": len(train_dataset) - len(logical_order),
            "train_loss": train_stats["loss"],
            "train_base_loss": train_stats["base_loss"],
            "train_state_loss": train_stats["state_loss"],
            "train_ftv_loss": train_stats["ftv_loss"],
            "train_sph_loss": train_stats["sph_loss"],
            "train_grounded_patients": train_stats["grounded_patients"],
            "train_sph_patients": train_stats["sph_patients"],
            "train_representation_std": train_stats["representation_std"],
            "train_optimizer_steps": train_stats["optimizer_steps"],
            "val_loss": val_stats["loss"],
            "val_base_objective": val_stats["base_loss"],
            "val_state_loss": val_stats["state_loss"],
            "val_ftv_loss": val_stats["ftv_loss"],
            "val_sph_loss": val_stats["sph_loss"],
            "val_grounded_patients": val_stats["grounded_patients"],
            "val_sph_patients": val_stats["sph_patients"],
            "val_representation_std": val_stats["representation_std"],
            "finite": all(
                math.isfinite(float(value))
                for value in (
                    train_stats["loss"],
                    val_stats["loss"],
                    val_stats["state_loss"],
                    val_stats["ftv_loss"],
                    val_stats["sph_loss"],
                    val_stats["representation_std"],
                )
            ),
        }
        history.append(row)
        checkpoint_payload = {
            "schema_version": 1,
            "stage": "residual_sph_grounding_pilot",
            "arm": spec.name,
            "sph_target": spec.sph_target,
            "lambda_sph": spec.lambda_sph,
            "seed_base": int(seed_base),
            "fold": int(fold),
            "effective_seed": effective_seed,
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "model_config": model.model_config(),
            "architecture_contract": model.architecture_contract(),
            "hyperparameters": asdict(hyperparameters),
            "shared_initialization_sha256": shared_hash,
            "base_initialization_sha256": base_hash,
            "transition_initialization_sha256": transition_hash,
            "train_patient_sha256": train_patient_sha256,
            "val_patient_sha256": val_patient_sha256,
            "data_provenance_sha256": provenance_sha256,
            "preregistration_lock_sha256": str(preregistration_lock_sha256),
            "s0_anchor": {
                "selection_sha256": file_sha256(s0_selection_path),
                "selected_validation_state_loss": s0_state,
                "paired_initialization_sha256": s0_payload["paired_initialization_sha256"],
            },
            "test_data_used": False,
            "delta_ftv_used": False,
            "delta_sph_used": False,
            "pcr_used": False,
            "clinical_used": False,
            "treatment_used": False,
            "epoch_metrics": row,
        }
        _atomic_torch_save(output / f"epoch_{epoch:02d}.pt", checkpoint_payload)
        _write_history(output / "history.csv", history)
        noncollapse = row["finite"] and row["val_representation_std"] >= hyperparameters.min_representation_std
        violation = max(0.0, row["val_state_loss"] - 1.05 * s0_state)
        current = (violation, row["val_ftv_loss"], row["val_state_loss"])
        if noncollapse and current < running_best:
            running_best = current
            stale = 0
        else:
            stale += 1
        if stale >= hyperparameters.patience:
            break
    selection = select_checkpoint(
        history,
        min_representation_std=hyperparameters.min_representation_std,
        paired_s0_state_loss=s0_state,
    )
    selection.update(
        {
            "arm": spec.name,
            "sph_target": spec.sph_target,
            "lambda_sph": spec.lambda_sph,
            "seed_base": int(seed_base),
            "fold": int(fold),
            "effective_seed": effective_seed,
            "shared_initialization_sha256": shared_hash,
            "base_initialization_sha256": base_hash,
            "hyperparameters": asdict(hyperparameters),
            "train_patient_sha256": train_patient_sha256,
            "val_patient_sha256": val_patient_sha256,
            "data_provenance_sha256": provenance_sha256,
            "preregistration_lock_sha256": str(preregistration_lock_sha256),
            "history_sha256": file_sha256(output / "history.csv"),
            "test_data_used": False,
            "delta_ftv_used": False,
            "delta_sph_used": False,
            "pcr_used": False,
            "clinical_used": False,
            "treatment_used": False,
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
    selected_payload["selection_sha256"] = file_sha256(selection_path)
    _atomic_torch_save(output / "selected.pt", selected_payload)
    return selection


__all__ = [
    "logical_sigreg_surrogate",
    "logical_supervision_counts",
    "run_logical_train_epoch",
    "run_validation_epoch",
    "scale_microbatch_components",
    "seed_everything",
    "select_checkpoint",
    "train_epochs",
    "validate_s0_anchor",
]

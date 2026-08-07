"""Formal single-fold B--F evaluation driver for audit-retrained CoRe-WM.

The existing fold evaluator owns B, C, D and F.  This module composes it with
the independent matched-donor evaluator (E), while keeping donor matching
strictly baseline-only.  Held-out tensors are materialized once into an
outcome-free in-memory dataset so ten donor repetitions do not repeatedly read
the same large NPZ files.

The B/C/D/F and E output directories are independent, fresh-only commits.  If
E fails after B/C/D/F has committed, the complete B/C/D/F directory is retained
for diagnosis; neither directory is ever overwritten or deleted.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ispy_jepa_tmi_clean.corejepa.data.records import treatment_family

from .donor_evaluation import MatchedDonorFoldAudit, run_matched_donor_fold_audit
from .fold_evaluation import (
    DonorEvaluationContext,
    FoldEvaluationResult,
    evaluate_retrained_fold,
)
from .matching import MatchingConfig


DRIVER_SCHEMA_VERSION = "shortcut_audit.retrained_fold_b_to_f.v1"
DONOR_MATCHING_SEED = 1729
BASELINE_VOLUME_UNIT = "cropped_roi_voxel_count(mask_channel_7>0.5)"
BASELINE_METADATA_COLUMNS = (
    "patient_id",
    "fold",
    "hr",
    "her2",
    "treatment_family",
    "baseline_lesion_volume",
    "baseline_lesion_volume_unit",
    "age",
    "mammaprint",
    "has_t1",
    "has_t2",
)
OUTCOME_COLUMNS = frozenset(
    {
        "pcr",
        "label_pcr",
        "outcome",
        "label",
        "y",
        "y_true",
        "target",
        "response",
    }
)


@dataclass(frozen=True)
class PreparedHeldoutDonorData:
    """Outcome-free held-out tensors and their baseline matching table."""

    metadata: pd.DataFrame
    dataset: Dataset
    patient_index: dict[str, int]
    source_patient_index: dict[str, int]


@dataclass(frozen=True)
class RetrainedFoldAuditResult:
    """Combined in-memory handles for one completed B--F fold audit."""

    fold: int
    evaluation: FoldEvaluationResult
    donor: MatchedDonorFoldAudit
    donor_metadata: pd.DataFrame


class OutcomeBlindHeldoutMemoryDataset(Dataset):
    """Minimal immutable-by-convention tensor cache for donor inference.

    Only ``patient_id``, ``image``, ``geometry`` and ``condition`` are retained.
    No record, pCR, routing target, response target or label is stored.  Tensors
    are detached CPU clones made during construction; ``DonorPairDataset`` only
    reads them and creates independent perturbed tensors.
    """

    ITEM_KEYS = ("patient_id", "image", "geometry", "condition")

    def __init__(self, patient_ids: Sequence[str], items: Iterable[Mapping[str, Any]]):
        normalized_ids = tuple(str(value) for value in patient_ids)
        if not normalized_ids or len(normalized_ids) != len(set(normalized_ids)):
            raise ValueError("memory donor dataset patient IDs 为空或重复")
        stored: list[dict[str, Any]] = []
        for expected_id, raw in zip(normalized_ids, items, strict=True):
            if not isinstance(raw, Mapping):
                raise TypeError("base dataset item 必须为 mapping")
            missing = [key for key in self.ITEM_KEYS if key not in raw]
            if missing:
                raise KeyError(f"base dataset item 缺少 donor 字段：{missing}")
            observed_id = str(raw["patient_id"])
            if observed_id != expected_id:
                raise ValueError("base dataset patient_index 与 item patient_id 不一致")
            tensors: dict[str, torch.Tensor] = {}
            for key in ("image", "geometry", "condition"):
                value = raw[key]
                if not torch.is_tensor(value) or not value.is_floating_point():
                    raise TypeError(f"base dataset {key} 必须为浮点 tensor")
                tensor = value.detach().to(device="cpu").clone()
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"base dataset {key} 含非有限值：{expected_id}")
                tensors[key] = tensor
            image, geometry, condition = (
                tensors["image"],
                tensors["geometry"],
                tensors["condition"],
            )
            if image.ndim != 5 or tuple(image.shape[:2]) != (4, 8):
                raise ValueError(f"held-out image 必须为 [4,8,Z,Y,X]：{expected_id}")
            if tuple(geometry.shape) != (4, 9):
                raise ValueError(f"held-out geometry 必须为 [4,9]：{expected_id}")
            if (
                condition.ndim != 2
                or condition.shape[0] != 3
                or condition.shape[1] <= 0
            ):
                raise ValueError(f"held-out condition 必须为 [3,C]：{expected_id}")
            if not (image.dtype == geometry.dtype == condition.dtype):
                raise TypeError(
                    f"held-out image/geometry/condition dtype 不一致：{expected_id}"
                )
            stored.append(
                {
                    "patient_id": expected_id,
                    "image": image,
                    "geometry": geometry,
                    "condition": condition,
                }
            )
        self.patient_ids = normalized_ids
        self._items = tuple(stored)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        # Return a fresh mapping so downstream code cannot add fields to the
        # stored object.  Tensor storage is shared read-only by the known donor
        # wrapper, which clones every perturbed input before assignment.
        item = self._items[int(index)]
        return {key: item[key] for key in self.ITEM_KEYS}


def _fold(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("fold 必须为 0..4")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("fold 必须为 0..4") from error
    if (
        not math.isfinite(numeric)
        or not numeric.is_integer()
        or int(numeric) not in range(5)
    ):
        raise ValueError("fold 必须为 0..4")
    return int(numeric)


def _validate_fresh_separate_outputs(
    evaluation_output_dir: str | Path,
    donor_output_dir: str | Path,
) -> tuple[Path, Path]:
    evaluation = Path(evaluation_output_dir).resolve()
    donor = Path(donor_output_dir).resolve()
    for label, path in (("evaluation", evaluation), ("donor", donor)):
        if os.path.lexists(path):
            raise FileExistsError(f"拒绝覆盖 {label} output_dir：{path}")
    if (
        evaluation == donor
        or evaluation in donor.parents
        or donor in evaluation.parents
    ):
        raise ValueError("evaluation 与 donor output_dir 必须是互不嵌套的独立目录")
    return evaluation, donor


def _record_ids(context: DonorEvaluationContext) -> tuple[str, ...]:
    records = tuple(context.heldout_records)
    patient_ids = tuple(str(record.patient_id) for record in records)
    if not patient_ids or len(patient_ids) != len(set(patient_ids)):
        raise ValueError("donor context heldout_records patient IDs 为空或重复")
    if patient_ids != tuple(context.readout_bundle.test_patient_ids):
        raise ValueError(
            "heldout_records patient/order 与 native readout test split 不一致"
        )
    if set(context.patient_index) != set(patient_ids):
        raise ValueError("donor context patient_index 未精确覆盖 held-out patients")
    source_indices = [
        int(context.patient_index[patient_id]) for patient_id in patient_ids
    ]
    if len(source_indices) != len(set(source_indices)):
        raise ValueError("donor context patient_index 含重复 dataset index")
    if any(index < 0 or index >= len(context.base_dataset) for index in source_indices):
        raise ValueError("donor context patient_index 越界")
    return patient_ids


def _baseline_volume(item: Mapping[str, Any], patient_id: str) -> int:
    image = item["image"]
    geometry = item["geometry"]
    mask = image[0, 7] > 0.5
    count = int(torch.count_nonzero(mask).item())
    expected_fraction = count / int(mask.numel())
    observed_fraction = float(geometry[0, 0].item())
    if not math.isclose(
        observed_fraction,
        expected_fraction,
        rel_tol=1e-5,
        abs_tol=1e-7,
    ):
        raise ValueError(
            f"T0 ROI voxel count 与 q0 roi_volume_fraction 不一致：{patient_id}"
        )
    return count


def prepare_heldout_donor_data(
    context: DonorEvaluationContext,
) -> PreparedHeldoutDonorData:
    """Materialize each held-out NPZ once and build outcome-blind metadata.

    The function accesses only explicitly named baseline record attributes; it
    never reads ``record.pcr`` or ``context.labels_by_patient``.  The latter is
    handed separately to donor inference solely for final prediction reporting.
    """

    fold = _fold(context.fold)
    patient_ids = _record_ids(context)
    # A generator avoids holding both the source and cloned six-gigabyte fold
    # tensors at once in the formal 162-patient evaluation.
    raw_items = (
        context.base_dataset[int(context.patient_index[patient_id])]
        for patient_id in patient_ids
    )
    memory_dataset = OutcomeBlindHeldoutMemoryDataset(patient_ids, raw_items)

    rows: list[dict[str, Any]] = []
    for index, (record, patient_id) in enumerate(
        zip(context.heldout_records, patient_ids, strict=True)
    ):
        item = memory_dataset[index]
        hr, her2, mp = int(record.hr), int(record.her2), int(record.mp)
        if hr not in (0, 1) or her2 not in (0, 1) or mp not in (0, 1):
            raise ValueError(f"baseline biomarker 必须为 0/1：{patient_id}")
        age = float(record.age)
        if math.isinf(age):
            raise ValueError(f"baseline age 不得为 Inf：{patient_id}")
        rows.append(
            {
                "patient_id": patient_id,
                "fold": fold,
                "hr": hr,
                "her2": her2,
                "treatment_family": treatment_family(record),
                "baseline_lesion_volume": _baseline_volume(item, patient_id),
                "baseline_lesion_volume_unit": BASELINE_VOLUME_UNIT,
                "age": age,
                "mammaprint": mp,
                # The clean cohort and cache contract require four visits.
                # Availability means an observed visit tensor exists; it does
                # not depend on whether the lesion mask is empty after response.
                "has_t1": True,
                "has_t2": True,
            }
        )
    metadata = pd.DataFrame.from_records(rows, columns=BASELINE_METADATA_COLUMNS)
    normalized_columns = {str(column).strip().lower() for column in metadata.columns}
    forbidden = sorted(normalized_columns.intersection(OUTCOME_COLUMNS))
    if forbidden:
        raise RuntimeError(f"donor baseline metadata 意外含 outcome 列：{forbidden}")
    if tuple(metadata["patient_id"].astype(str)) != patient_ids:
        raise RuntimeError("donor baseline metadata patient order 漂移")
    return PreparedHeldoutDonorData(
        metadata=metadata,
        dataset=memory_dataset,
        patient_index={
            patient_id: index for index, patient_id in enumerate(patient_ids)
        },
        source_patient_index={
            patient_id: int(context.patient_index[patient_id])
            for patient_id in patient_ids
        },
    )


def evaluate_retrained_fold_b_to_f(
    fold_dir: str | Path,
    *,
    fold: int,
    legacy_x_cache_dir: str | Path,
    evaluation_output_dir: str | Path,
    donor_output_dir: str | Path,
    device: str | torch.device = "cpu",
    batch_size: int | None = None,
    workers: int | None = None,
    donor_batch_size: int = 4,
    donor_workers: int = 0,
    matching_config: MatchingConfig | None = None,
    **fold_evaluation_kwargs: Any,
) -> RetrainedFoldAuditResult:
    """Run B/C/D/F first, then strict matched-donor E for one fold."""

    fold = _fold(fold)
    evaluation_output, donor_output = _validate_fresh_separate_outputs(
        evaluation_output_dir,
        donor_output_dir,
    )
    config = matching_config or MatchingConfig(
        max_donors=10,
        seed=DONOR_MATCHING_SEED,
        allow_relaxed_matches=False,
    )
    if config.max_donors != 10 and matching_config is None:  # defensive invariant
        raise RuntimeError("default donor matching 必须请求 10 donors")
    if matching_config is None and config.allow_relaxed_matches:
        raise RuntimeError("default donor matching 必须使用 strict hard matching")

    evaluation = evaluate_retrained_fold(
        fold_dir,
        fold=fold,
        legacy_x_cache_dir=legacy_x_cache_dir,
        output_dir=evaluation_output,
        device=device,
        batch_size=batch_size,
        workers=workers,
        **fold_evaluation_kwargs,
    )
    if evaluation.fold != fold or evaluation.donor_context.fold != fold:
        raise RuntimeError("B/C/D/F result fold 与 driver 请求不一致")
    if Path(evaluation.output_dir).resolve() != evaluation_output:
        raise RuntimeError("B/C/D/F result output_dir 与 driver 请求不一致")
    if not evaluation_output.is_dir():
        raise RuntimeError("B/C/D/F evaluator 未提交完整 output_dir")
    if os.path.lexists(donor_output):
        raise FileExistsError(f"拒绝并发覆盖 donor output_dir：{donor_output}")

    prepared = prepare_heldout_donor_data(evaluation.donor_context)
    native_prediction_path = evaluation.artifact_paths.get("predictions_native")
    donor = run_matched_donor_fold_audit(
        fold=fold,
        heldout_metadata=prepared.metadata,
        base_dataset=prepared.dataset,
        patient_index=prepared.patient_index,
        model=evaluation.donor_context.model,
        readout_bundle=evaluation.donor_context.readout_bundle,
        labels_by_patient=evaluation.donor_context.labels_by_patient,
        checkpoint=evaluation.donor_context.checkpoint,
        output_dir=donor_output,
        matching_config=config,
        device=evaluation.donor_context.device,
        inference_batch_size=donor_batch_size,
        num_workers=donor_workers,
        pin_memory=evaluation.donor_context.device.type == "cuda",
        caller_provenance={
            "driver_schema_version": DRIVER_SCHEMA_VERSION,
            "b_c_d_f_output_dir": str(evaluation_output),
            "native_prediction_csv": (
                str(native_prediction_path)
                if native_prediction_path is not None
                else None
            ),
            "heldout_tensor_source": "outcome-free memory materialization; one NPZ read per patient",
            "source_patient_index": prepared.source_patient_index,
            "baseline_metadata_columns": list(BASELINE_METADATA_COLUMNS),
            "baseline_lesion_volume_unit": BASELINE_VOLUME_UNIT,
            "outcome_columns_in_matching_metadata": [],
        },
    )
    return RetrainedFoldAuditResult(
        fold=fold,
        evaluation=evaluation,
        donor=donor,
        donor_metadata=prepared.metadata.copy(),
    )


__all__ = [
    "BASELINE_METADATA_COLUMNS",
    "BASELINE_VOLUME_UNIT",
    "DONOR_MATCHING_SEED",
    "DRIVER_SCHEMA_VERSION",
    "OutcomeBlindHeldoutMemoryDataset",
    "PreparedHeldoutDonorData",
    "RetrainedFoldAuditResult",
    "evaluate_retrained_fold_b_to_f",
    "prepare_heldout_donor_data",
]

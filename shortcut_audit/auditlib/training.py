"""Audit-only explicit five-fold training wrapper for the clean CoRe-JEPA.

The clean runner hard-codes one 70/15/15 split.  This module keeps the clean
model, objective, epoch runner, EMA update, and frozen-state export unchanged,
but replaces that split construction with the audited long-format seed-2026
manifest.  It deliberately lives outside ``ispy_jepa_tmi_clean`` so that an
audit retrain cannot silently change the reference implementation.

Two leakage barriers are enforced here:

* the ``ConditionEncoder`` (arm vocabulary and age normalization) is fitted on
  ``primary_train + I-SPY1`` only;
* ``ResponseTargetTransform`` statistics are fitted on the same pCR-free
  pretraining rows only.

Patient outcomes are used to verify the manifest and, after training, to export
frozen states for the downstream readout.  They are redacted from every record
that can reach the pretraining dataset or objective.
"""

from __future__ import annotations

import copy
import csv
import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ispy_jepa_tmi_clean.corejepa.config import ExperimentConfig
from ispy_jepa_tmi_clean.corejepa.data.condition import ConditionEncoder
from ispy_jepa_tmi_clean.corejepa.data.dataset import LongitudinalDCEDataset, PretrainingDataset
from ispy_jepa_tmi_clean.corejepa.data.imaging import load_phase_metadata, mask_geometry
from ispy_jepa_tmi_clean.corejepa.data.records import PatientRecord
from ispy_jepa_tmi_clean.corejepa.data.response_targets import (
    ResponseTargetTransform,
    build_response_feature_cache,
    load_response_vectors,
)
from ispy_jepa_tmi_clean.corejepa.models import CoReJEPA
from ispy_jepa_tmi_clean.corejepa.training import runner as clean_runner
from ispy_jepa_tmi_clean.corejepa.training.losses import PretrainingObjective

from .folds import load_fold_manifest, validate_fold_manifest


DEFAULT_SEED2026_MANIFEST = Path(
    "/data/data/Preprocessed/I-SPY2/"
    "_matched_breastdcedl_t0_dicomrepair_rgb224_seed2026/"
    "matched_patient_cv_splits_seed2026.csv"
)
SEED2026_MANIFEST_SHA256 = "143e482d711225c0611006d99bd7345d2fa1a5c16c65fbaf8399341a0d26aa38"
AUDIT_PROTOCOL = "corejepa_explicit_fivefold_seed2026_v1"
VERIFIED_LEGACY_X_CACHE_NAME = (
    "_mixed_ispy1_train_cache_dce8_adaptivephase_axiscanonv1_"
    "autoroi_t0fallback_minfrac05_z32_y96_x96"
)
VERIFIED_LEGACY_X_INVENTORY_SHA256 = (
    "fdd1474b9cdbf202bc34b12e1e5b08e180a1f1da141c8f8b2edd5883fabcd658"
)


BaseDatasetFactory = Callable[
    [list[PatientRecord], ConditionEncoder, ExperimentConfig],
    Dataset,
]


@dataclass(frozen=True)
class FoldTrainingPlan:
    """Immutable patient-order and output contract for one explicit fold."""

    fold: int
    manifest_path: str
    manifest_sha256: str
    manifest_hash_kind: str
    manifest_n_rows: int
    manifest_n_patients: int
    output_dir: Path
    patient_ids: tuple[str, ...]
    n_primary: int
    primary_train: tuple[int, ...]
    pretrain_train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]

    def checkpoint_splits(self) -> dict[str, list[int]]:
        """Return the exact clean-checkpoint split schema."""

        return {
            "primary_train": list(self.primary_train),
            "pretrain_train": list(self.pretrain_train),
            "validation": list(self.validation),
            "test": list(self.test),
        }

    def split_patient_ids(self) -> dict[str, list[str]]:
        """Return primary split IDs in checkpoint patient order."""

        return {
            "train": [self.patient_ids[index] for index in self.primary_train],
            "val": [self.patient_ids[index] for index in self.validation],
            "test": [self.patient_ids[index] for index in self.test],
        }

    @property
    def extra_pretraining_indices(self) -> tuple[int, ...]:
        return tuple(index for index in self.pretrain_train if index >= self.n_primary)


@dataclass
class PreparedFoldTraining:
    """All objects needed by the clean epoch runner for one fold."""

    config: ExperimentConfig
    dataset: PretrainingDataset
    records: list[PatientRecord]
    pretraining_records: list[PatientRecord]
    n_primary: int
    plan: FoldTrainingPlan
    condition_encoder: ConditionEncoder
    response_transform: ResponseTargetTransform
    base_dataset_source: dict[str, Any]
    response_target_source: dict[str, Any]


def _canonical_frame_sha256(frame: pd.DataFrame) -> str:
    content = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _source_fingerprint(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": _file_sha256(resolved)}


def _default_output_root(config: ExperimentConfig) -> Path:
    configured = Path(config.train.output_dir)
    return configured.with_name(f"{configured.name}_audit_fivefold_seed2026")


def _validate_record_partition(records: Sequence[PatientRecord], n_primary: int) -> None:
    if n_primary <= 0 or n_primary > len(records):
        raise ValueError("n_primary 必须把非空 I-SPY2 primary 放在 patient order 前部")
    ids = [str(record.patient_id) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("records 含重复 patient_id")
    wrong_primary = [record.patient_id for record in records[:n_primary] if record.cohort.lower() != "ispy2"]
    wrong_extra = [record.patient_id for record in records[n_primary:] if record.cohort.lower() != "ispy1"]
    if wrong_primary:
        raise ValueError(f"primary 范围含非 I-SPY2 record：{wrong_primary[:5]}")
    if wrong_extra:
        raise ValueError(f"extra pretraining 范围仅允许 I-SPY1：{wrong_extra[:5]}")


def build_fold_training_plan(
    records: Sequence[PatientRecord],
    n_primary: int,
    manifest: str | Path | pd.DataFrame,
    fold: int,
    *,
    output_root: str | Path,
    expected_manifest_sha256: str | None = None,
) -> FoldTrainingPlan:
    """Map one validated long-format fold to clean-runner indices.

    The first ``n_primary`` records must be I-SPY2 and every remaining record
    must be I-SPY1.  Manifest labels are checked against I-SPY2 outcomes, but no
    outcome is copied into a training dataset or transform.
    """

    records = list(records)
    _validate_record_partition(records, n_primary)
    primary = records[:n_primary]
    missing_outcome = [record.patient_id for record in primary if record.pcr is None]
    if missing_outcome:
        raise ValueError(f"无法核验 manifest label；I-SPY2 缺少 pCR：{missing_outcome[:5]}")
    expected_ids = [record.patient_id for record in primary]
    expected_labels = {record.patient_id: int(record.pcr) for record in primary if record.pcr is not None}

    if isinstance(manifest, pd.DataFrame):
        frame, summary = validate_fold_manifest(
            manifest,
            expected_patient_ids=expected_ids,
            expected_labels=expected_labels,
        )
        manifest_path = "<in-memory>"
        manifest_sha256 = _canonical_frame_sha256(frame)
        hash_kind = "canonical_validated_csv"
    else:
        resolved = Path(manifest).resolve()
        frame, summary = load_fold_manifest(
            resolved,
            expected_patient_ids=expected_ids,
            expected_labels=expected_labels,
        )
        manifest_path = str(resolved)
        manifest_sha256 = str(summary["sha256"])
        hash_kind = "file_sha256"
    if expected_manifest_sha256 is not None and manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            "fold manifest SHA256 不匹配："
            f"observed={manifest_sha256}, expected={expected_manifest_sha256}"
        )
    fold = int(fold)
    if fold not in {int(value) for value in frame["fold"].unique()}:
        raise ValueError(f"manifest 不含 fold={fold}")

    selected = frame.loc[frame["fold"].eq(fold), ["patient_id", "split"]]
    split_by_id = dict(zip(selected["patient_id"], selected["split"]))
    primary_train = tuple(
        index for index, record in enumerate(primary) if split_by_id[record.patient_id] == "train"
    )
    validation = tuple(
        index for index, record in enumerate(primary) if split_by_id[record.patient_id] == "val"
    )
    test = tuple(index for index, record in enumerate(primary) if split_by_id[record.patient_id] == "test")
    extra = tuple(range(n_primary, len(records)))
    pretrain_train = primary_train + extra
    if not primary_train or not validation or not test:
        raise ValueError("每折 primary_train/validation/test 均必须非空")

    output_dir = Path(output_root) / f"fold_{fold:02d}"
    return FoldTrainingPlan(
        fold=fold,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        manifest_hash_kind=hash_kind,
        manifest_n_rows=int(summary["n_rows"]),
        manifest_n_patients=int(summary["n_patients"]),
        output_dir=output_dir,
        patient_ids=tuple(record.patient_id for record in records),
        n_primary=n_primary,
        primary_train=primary_train,
        pretrain_train=pretrain_train,
        validation=validation,
        test=test,
    )


def fit_fold_preprocessors(
    records: Sequence[PatientRecord],
    plan: FoldTrainingPlan,
    raw_response: np.ndarray,
) -> tuple[list[PatientRecord], ConditionEncoder, ResponseTargetTransform, np.ndarray, np.ndarray]:
    """Fit both fold-dependent preprocessors without exposing pCR.

    Returns outcome-redacted records, the fitted condition encoder and response
    transform, followed by transformed vector/score targets.  The response
    statistics and the age normalization both use ``pretrain_train`` only.
    """

    records = list(records)
    if tuple(record.patient_id for record in records) != plan.patient_ids:
        raise ValueError("records 顺序与 FoldTrainingPlan patient order 不一致")
    raw_response = np.asarray(raw_response, dtype=np.float32)
    if raw_response.shape != (len(records), 3, 18):
        raise ValueError(f"raw_response 必须为 [N,3,18]，实际为 {raw_response.shape}")

    outcome_blind = [replace(record, pcr=None) for record in records]
    fit_indices = list(plan.pretrain_train)
    fit_records = [outcome_blind[index] for index in fit_indices]
    condition_encoder = ConditionEncoder(fit_records)
    unseen_arms: dict[str, list[str]] = {}
    for record in outcome_blind:
        if record.arm not in condition_encoder.spec.arm_vocab:
            unseen_arms.setdefault(record.arm, []).append(record.patient_id)
    if unseen_arms:
        examples = {arm: ids[:3] for arm, ids in sorted(unseen_arms.items())}
        raise ValueError(f"validation/test 出现 pretrain_train 未见 treatment arm：{examples}")

    response_transform = ResponseTargetTransform.fit(raw_response, outcome_blind, fit_indices)
    response_vector, response_score = response_transform.transform(raw_response, outcome_blind)
    return outcome_blind, condition_encoder, response_transform, response_vector, response_score


class LegacyXCacheDataset(Dataset):
    """Read the verified shared ``key=x`` cache without copying it.

    Each aggregate file is named ``<patient_id>_dce8_*.npz`` and stores
    ``x [4,8,Z,Y,X]``.  Geometry is recomputed with the clean branch's exact
    ``mask_geometry`` implementation from channel 7; no legacy geometry or
    labels enter the dataset.
    """

    def __init__(
        self,
        records: Sequence[PatientRecord],
        condition_encoder: ConditionEncoder,
        cache_dir: str | Path,
        *,
        expected_crop_size: tuple[int, int, int] | None = None,
    ) -> None:
        self.records = list(records)
        self.condition_encoder = condition_encoder
        self.cache_dir = Path(cache_dir).resolve()
        self.expected_crop_size = tuple(expected_crop_size) if expected_crop_size is not None else None
        if not self.cache_dir.is_dir():
            raise FileNotFoundError(f"legacy-x cache directory 不存在：{self.cache_dir}")
        index: dict[str, Path] = {}
        duplicates: list[str] = []
        for path in sorted(self.cache_dir.glob("*_dce8_*.npz")):
            patient_id = path.name.split("_dce8_", maxsplit=1)[0]
            if patient_id in index:
                duplicates.append(patient_id)
            index[patient_id] = path
        if duplicates:
            raise ValueError(f"legacy-x cache 每患者必须恰好一个 aggregate NPZ：{sorted(set(duplicates))[:5]}")
        missing = [record.patient_id for record in self.records if record.patient_id not in index]
        if missing:
            raise FileNotFoundError(f"legacy-x cache 缺少 {len(missing)} 名患者；示例={missing[:5]}")
        self._paths = [index[record.patient_id] for record in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def cache_path(self, index: int) -> Path:
        return self._paths[index]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        path = self._paths[index]
        with np.load(path, allow_pickle=False) as cache:
            if "x" not in cache.files:
                raise ValueError(f"legacy-x cache 缺少 key='x'：{path}")
            image = cache["x"].astype(np.float32)
        if image.ndim != 5 or image.shape[:2] != (4, 8):
            raise ValueError(f"legacy-x image 必须为 [4,8,Z,Y,X]：{path} -> {image.shape}")
        if self.expected_crop_size is not None and tuple(image.shape[2:]) != self.expected_crop_size:
            raise ValueError(
                f"legacy-x crop 与 config 不一致：{path} -> {tuple(image.shape[2:])}, "
                f"expected={self.expected_crop_size}"
            )
        if not np.isfinite(image).all():
            raise ValueError(f"legacy-x image 含非有限值：{path}")
        geometry = np.stack([mask_geometry(image[visit, 7]) for visit in range(4)]).astype(np.float32)
        return {
            "patient_id": record.patient_id,
            "record_index": torch.tensor(index, dtype=torch.long),
            "image": torch.from_numpy(image),
            "geometry": torch.from_numpy(geometry),
            "condition": torch.from_numpy(self.condition_encoder.encode(record)),
            "routing_target": torch.tensor(self.condition_encoder.routing_target(record), dtype=torch.long),
        }


def _base_dataset(
    config: ExperimentConfig,
    records: list[PatientRecord],
    condition_encoder: ConditionEncoder,
    *,
    build_missing_tensors: bool,
    legacy_x_cache_dir: str | Path | None,
    base_dataset_factory: BaseDatasetFactory | None,
) -> tuple[Dataset, dict[str, Any]]:
    choices = int(legacy_x_cache_dir is not None) + int(base_dataset_factory is not None)
    if choices > 1:
        raise ValueError("legacy_x_cache_dir 与 base_dataset_factory 只能选择一个")
    if base_dataset_factory is not None:
        base = base_dataset_factory(records, condition_encoder, config)
        if not isinstance(base, Dataset):
            raise TypeError("base_dataset_factory 必须返回 torch.utils.data.Dataset")
        source = {
            "mode": "injected_base_dataset_factory",
            "factory": f"{base_dataset_factory.__module__}.{base_dataset_factory.__qualname__}",
        }
    elif legacy_x_cache_dir is not None:
        resolved_cache = Path(legacy_x_cache_dir).resolve()
        base = LegacyXCacheDataset(
            records,
            condition_encoder,
            resolved_cache,
            expected_crop_size=config.data.crop_size,
        )
        source = {
            "mode": "verified_legacy_x_adapter",
            "cache_dir": str(resolved_cache),
            "n_patient_files": len(base),
            "image_key": "x",
            "geometry": "clean mask_geometry(x[:,7])",
            "preprocessing_variant": (
                "legacy_adaptive_axiscanon_v1_autoroi_t0fallback_minfrac0.5"
            ),
            "exact_clean_equivalence": False,
            "inventory_sha256": (
                VERIFIED_LEGACY_X_INVENTORY_SHA256
                if resolved_cache.name == VERIFIED_LEGACY_X_CACHE_NAME
                else None
            ),
        }
    else:
        phase_metadata = load_phase_metadata(config.data.breastdcedl_metadata_csv)
        base = LongitudinalDCEDataset(
            records=records,
            condition_encoder=condition_encoder,
            cache_dir=config.data.tensor_cache,
            crop_size=config.data.crop_size,
            phase_policy=config.data.phase_policy,
            phase_metadata=phase_metadata,
            automatic_roi_fallback=config.data.auto_roi_fallback,
            minimum_roi_capture=config.data.min_roi_capture,
            legacy_empty_ftv_full_field=config.data.legacy_empty_ftv_full_field,
            build_missing=build_missing_tensors,
        )
        source = {
            "mode": "clean_longitudinal_dce_dataset",
            "cache_dir": str(Path(config.data.tensor_cache).resolve()),
            "build_missing_tensors": bool(build_missing_tensors),
        }
    if len(base) != len(records):
        raise ValueError(f"base dataset 长度 {len(base)} 与 records {len(records)} 不一致")
    return base, source


def prepare_fold_training(
    config: ExperimentConfig,
    fold: int,
    *,
    manifest: str | Path | pd.DataFrame = DEFAULT_SEED2026_MANIFEST,
    expected_manifest_sha256: str | None = SEED2026_MANIFEST_SHA256,
    output_root: str | Path | None = None,
    raw_response: np.ndarray | None = None,
    build_missing_tensors: bool = False,
    build_response_cache_if_missing: bool = False,
    legacy_x_cache_dir: str | Path | None = None,
    base_dataset_factory: BaseDatasetFactory | None = None,
) -> PreparedFoldTraining:
    """Prepare one fold while leaving the clean core implementation untouched."""

    fold_config = copy.deepcopy(config)
    records, n_primary = clean_runner.load_experiment_records(fold_config)
    plan = build_fold_training_plan(
        records,
        n_primary,
        manifest,
        fold,
        output_root=output_root if output_root is not None else _default_output_root(fold_config),
        expected_manifest_sha256=expected_manifest_sha256,
    )
    fold_config.train.output_dir = str(plan.output_dir)

    if raw_response is None:
        response_cache = Path(fold_config.data.response_cache)
        if not response_cache.exists():
            if not build_response_cache_if_missing:
                raise FileNotFoundError(
                    f"response cache 不存在：{response_cache}；"
                    "需先构建，或显式设置 build_response_cache_if_missing=True"
                )
            phase_metadata = load_phase_metadata(fold_config.data.breastdcedl_metadata_csv)
            outcome_blind_for_cache = [replace(record, pcr=None) for record in records]
            build_response_feature_cache(
                outcome_blind_for_cache,
                response_cache,
                fold_config.data.auto_roi_fallback,
                fold_config.data.legacy_empty_ftv_full_field,
                phase_metadata,
                fold_config.data.response_phase_policy,
            )
        raw_response = load_response_vectors(response_cache, records)
        response_target_source = {
            "mode": "response_feature_cache",
            "path": str(response_cache.resolve()),
            "sha256": _file_sha256(response_cache.resolve()),
        }
    else:
        raw_response = np.asarray(raw_response, dtype=np.float32)
        response_target_source = {
            "mode": "in_memory_raw_response",
            "shape": list(raw_response.shape),
            "dtype": str(raw_response.dtype),
            "sha256": _array_sha256(raw_response),
        }

    (
        outcome_blind_records,
        condition_encoder,
        response_transform,
        response_vector,
        response_score,
    ) = fit_fold_preprocessors(records, plan, raw_response)
    base, source = _base_dataset(
        fold_config,
        outcome_blind_records,
        condition_encoder,
        build_missing_tensors=build_missing_tensors,
        legacy_x_cache_dir=legacy_x_cache_dir,
        base_dataset_factory=base_dataset_factory,
    )
    dataset = PretrainingDataset(base, response_vector, response_score)  # type: ignore[arg-type]
    return PreparedFoldTraining(
        config=fold_config,
        dataset=dataset,
        records=records,
        pretraining_records=outcome_blind_records,
        n_primary=n_primary,
        plan=plan,
        condition_encoder=condition_encoder,
        response_transform=response_transform,
        base_dataset_source=source,
        response_target_source=response_target_source,
    )


def checkpoint_payload_with_provenance(
    model: torch.nn.Module,
    prepared: PreparedFoldTraining,
    epoch: int,
    validation: Mapping[str, float],
) -> dict[str, Any]:
    """Build the clean checkpoint schema plus an audit provenance block."""

    plan = prepared.plan
    payload = clean_runner._checkpoint_payload(  # noqa: SLF001 - exact clean schema is intentional
        model,
        prepared.config,
        prepared.condition_encoder,
        prepared.response_transform,
        prepared.records,
        prepared.n_primary,
        plan.checkpoint_splits(),
        int(epoch),
        {str(key): float(value) for key, value in validation.items()},
    )
    payload["audit_provenance"] = {
        "schema_version": 1,
        "protocol": AUDIT_PROTOCOL,
        "fold": plan.fold,
        "manifest": {
            "path": plan.manifest_path,
            "sha256": plan.manifest_sha256,
            "hash_kind": plan.manifest_hash_kind,
            "n_rows": plan.manifest_n_rows,
            "n_patients": plan.manifest_n_patients,
        },
        "primary_split_patient_ids": plan.split_patient_ids(),
        "extra_pretraining_patient_ids": [plan.patient_ids[index] for index in plan.extra_pretraining_indices],
        "fit_scopes": {
            "condition_encoder": "pretrain_train covariates only (primary_train + I-SPY1); pCR redacted",
            "response_transform": "pretrain_train imaging-response rows only; pCR redacted",
            "pretraining_objective": "pCR-free clean PretrainingObjective",
            "outcome_usage": "manifest verification/split assignment and post-training frozen-state export only",
        },
        "fit_indices": {
            "condition_encoder": list(plan.pretrain_train),
            "response_transform": list(plan.pretrain_train),
        },
        "base_dataset_source": dict(prepared.base_dataset_source),
        "response_target_source": dict(prepared.response_target_source),
        "implementation": {
            "audit_wrapper": _source_fingerprint(Path(__file__)),
            "clean_runner": _source_fingerprint(Path(clean_runner.__file__)),
        },
        "output_dir": str(plan.output_dir),
        "seed": int(prepared.config.train.seed),
    }
    return payload


def _write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _assert_fresh_output(output_dir: Path) -> None:
    protected = (
        "best_corejepa.pt",
        "last_corejepa.pt",
        "history.csv",
        "frozen_states.npz",
        "splits.json",
    )
    existing = [name for name in protected if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(f"拒绝覆盖 fold 既有 artifact：{output_dir} -> {existing}")


def train_explicit_fold(
    config: ExperimentConfig,
    fold: int,
    *,
    allow_training: bool = False,
    manifest: str | Path | pd.DataFrame = DEFAULT_SEED2026_MANIFEST,
    expected_manifest_sha256: str | None = SEED2026_MANIFEST_SHA256,
    output_root: str | Path | None = None,
    raw_response: np.ndarray | None = None,
    build_missing_tensors: bool = False,
    build_response_cache_if_missing: bool = False,
    legacy_x_cache_dir: str | Path | None = None,
    base_dataset_factory: BaseDatasetFactory | None = None,
) -> Path:
    """Train one fold with clean runner primitives and explicit opt-in.

    ``allow_training=False`` is a deliberate dead-man switch: importing this
    module, planning folds, or running unit tests can never start a formal job.
    Existing fold artifacts are never overwritten.
    """

    if not allow_training:
        raise RuntimeError("训练未启动；必须显式传入 allow_training=True")
    prepared = prepare_fold_training(
        config,
        fold,
        manifest=manifest,
        expected_manifest_sha256=expected_manifest_sha256,
        output_root=output_root,
        raw_response=raw_response,
        build_missing_tensors=build_missing_tensors,
        build_response_cache_if_missing=build_response_cache_if_missing,
        legacy_x_cache_dir=legacy_x_cache_dir,
        base_dataset_factory=base_dataset_factory,
    )
    output_dir = prepared.plan.output_dir
    _assert_fresh_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared.config.save(output_dir / "config.yaml")
    clean_runner.set_seed(prepared.config.train.seed)

    splits = prepared.plan.checkpoint_splits()
    train_loader = clean_runner.make_loader(
        prepared.dataset,
        splits["pretrain_train"],
        prepared.config,
        shuffle=True,
    )
    validation_loader = clean_runner.make_loader(
        prepared.dataset,
        splits["validation"],
        prepared.config,
        shuffle=False,
    )
    device, gpu_ids = clean_runner.select_device(prepared.config.train.gpus)
    model: torch.nn.Module = CoReJEPA(
        prepared.config.model,
        prepared.condition_encoder.spec.dim,
    ).to(device)
    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])
    route_weights = prepared.condition_encoder.routing_class_weights(
        prepared.pretraining_records,
        splits["pretrain_train"],
    )
    objective = PretrainingObjective(
        prepared.config.loss,
        prepared.config.train.sigreg_projections,
        torch.from_numpy(route_weights),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=prepared.config.train.learning_rate,
        weight_decay=prepared.config.train.weight_decay,
    )

    history: list[dict[str, Any]] = []
    best_prediction = float("inf")
    epochs_without_improvement = 0
    best_path = output_dir / "best_corejepa.pt"
    last_path = output_dir / "last_corejepa.pt"
    for epoch in range(1, prepared.config.train.epochs + 1):
        train_stats = clean_runner.run_epoch(
            model,
            objective,
            train_loader,
            device,
            optimizer,
            prepared.config.train.ema_momentum,
        )
        validation_stats = clean_runner.run_epoch(
            model,
            objective,
            validation_loader,
            device,
            None,
            prepared.config.train.ema_momentum,
        )
        row: dict[str, Any] = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_stats.items()})
        row.update({f"val_{key}": value for key, value in validation_stats.items()})
        history.append(row)
        payload = checkpoint_payload_with_provenance(model, prepared, epoch, validation_stats)
        torch.save(payload, last_path)
        eligible = validation_stats["visit_state_std"] >= prepared.config.train.min_latent_std
        if eligible and validation_stats["prediction"] < best_prediction:
            best_prediction = validation_stats["prediction"]
            epochs_without_improvement = 0
            torch.save(payload, best_path)
        else:
            epochs_without_improvement += 1
        _write_history(output_dir / "history.csv", history)
        if epochs_without_improvement >= prepared.config.train.patience:
            break
    if not best_path.exists():
        raise RuntimeError("No fold checkpoint satisfied the minimum latent standard deviation")
    clean_runner.export_frozen_states(
        prepared.config,
        best_path,
        prepared.dataset,
        prepared.records,
        prepared.n_primary,
        splits,
        prepared.condition_encoder,
        device,
    )
    return best_path


def train_fivefold(
    config: ExperimentConfig,
    *,
    allow_training: bool = False,
    folds: Sequence[int] = (0, 1, 2, 3, 4),
    **kwargs: Any,
) -> dict[int, Path]:
    """Run independent folds sequentially; explicit opt-in is mandatory."""

    if not allow_training:
        raise RuntimeError("五折训练未启动；必须显式传入 allow_training=True")
    requested = tuple(int(fold) for fold in folds)
    if not requested or len(requested) != len(set(requested)) or not set(requested).issubset(range(5)):
        raise ValueError("folds 必须是 0..4 的非空、不重复子集")
    return {
        fold: train_explicit_fold(
            config,
            fold,
            allow_training=True,
            **kwargs,
        )
        for fold in requested
    }


def make_smoke_config(config: ExperimentConfig, output_root: str | Path) -> ExperimentConfig:
    """Return a CPU, one-epoch, reduced-width config without mutating input."""

    smoke = copy.deepcopy(config)
    smoke.data.crop_size = (8, 16, 16)
    smoke.model.base_channels = 2
    smoke.model.latent_dim = 16
    smoke.model.predictor_depth = 1
    smoke.model.predictor_heads = 2
    smoke.model.predictor_mlp_dim = 32
    smoke.model.response_dim = 8
    smoke.model.response_hidden_dim = 16
    smoke.model.response_depth = 1
    smoke.model.expert_hidden_dim = 16
    smoke.model.expert_gate_hidden_dim = 16
    smoke.train.output_dir = str(output_root)
    smoke.train.batch_size = 2
    smoke.train.workers = 0
    smoke.train.epochs = 1
    smoke.train.patience = 1
    smoke.train.sigreg_projections = 8
    smoke.train.min_latent_std = 0.0
    smoke.train.gpus = ()
    return smoke

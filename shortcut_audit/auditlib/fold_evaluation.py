"""Fail-closed evaluation orchestration for one audit-retrained fold.

The training wrapper deliberately stops after exporting the clean checkpoint
and frozen states.  This module is the corresponding *evaluation-only* entry
point.  It cross-checks every redundant patient/split/label representation
before fitting a readout or materialising an audit result, restores the frozen
model without changing ``corejepa``, and writes all results below one caller-
supplied, previously non-existent directory.

Donor matching is intentionally kept in :mod:`donor_inference`; the returned
``DonorEvaluationContext`` exposes the outcome-free base dataset and held-out
patient index needed to build a ``DonorPairDataset`` later.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from ispy_jepa_tmi_clean.corejepa.data.records import PatientRecord
from ispy_jepa_tmi_clean.corejepa.training import runner as clean_runner

from .baseline_features import (
    ClinicalFeatureSpec,
    clinical_features,
    clinical_geometry_features,
    geometry_features,
    static_t0_features,
    timepoint_only_features,
)
from .baseline_models import (
    BASELINE_FEATURE_SETS,
    BaselineReadoutConfig,
    FoldBaselineBundle,
    fit_fold_baseline,
    predict_fold_baseline,
    save_baseline_bundle,
)
from .contracts import DECISION_POINTS, validate_label_alignment, write_prediction_csv
from .donor_inference import DonorPairDataset
from .folds import load_fold_manifest
from .inference import (
    FrozenInferenceArrays,
    collect_frozen_inference,
    copy_current_latent_audit,
    paired_perturbation_latent_audit,
)
from .perturbations import (
    repeated_t0_full_image_derived,
    repeated_t0_mri_only,
    swap_t1_t2,
)
from .provenance import file_sha256, inspect_checkpoint, validate_checkpoint_payload
from .readouts import (
    AuditReadoutConfig,
    FoldReadoutBundle,
    fit_fold_readout,
    predict_fold_readout,
    save_readout_bundle,
)
from .runtime import records_in_checkpoint_order, restore_model_for_evaluation
from .training import AUDIT_PROTOCOL, LegacyXCacheDataset


EVALUATION_SCHEMA_VERSION = "shortcut_audit.fold_evaluation.v1"
CHECKPOINT_NAME = "best_corejepa.pt"
FROZEN_STATES_NAME = "frozen_states.npz"
SPLITS_NAME = "splits.json"
PERTURBATIONS = {
    "repeated_t0_c1_mri_only": repeated_t0_mri_only,
    "repeated_t0_c2_full_image_derived": repeated_t0_full_image_derived,
    "temporal_t1_t2_swap": swap_t1_t2,
}
GEOMETRY_COMPONENTS = (
    "baseline",
    "current",
    "from_baseline",
    "recent",
    "relative_to_baseline",
    "prefix_mean",
)


@dataclass(frozen=True)
class ValidatedFoldInputs:
    """Read-only objects that passed all redundant artifact checks."""

    fold: int
    fold_dir: Path
    checkpoint_path: Path
    checkpoint_id: str
    checkpoint_summary: dict[str, Any]
    payload: Mapping[str, Any]
    config: Any
    model: torch.nn.Module
    condition_encoder: Any
    records: tuple[PatientRecord, ...]
    n_primary: int
    splits: dict[str, tuple[int, ...]]
    frozen_response_state: np.ndarray
    frozen_image_prediction: np.ndarray
    labels: np.ndarray
    dataset: Dataset


@dataclass(frozen=True)
class DonorEvaluationContext:
    """Objects needed by the independent held-out donor-swap audit.

    ``base_dataset`` never returns pCR.  Outcomes are exposed separately only
    for final reporting after matching and inference have completed.
    """

    fold: int
    checkpoint: str
    model: torch.nn.Module
    readout_bundle: FoldReadoutBundle
    base_dataset: Dataset
    patient_index: dict[str, int]
    heldout_records: tuple[PatientRecord, ...]
    labels_by_patient: dict[str, int]
    device: torch.device

    def build_pair_dataset(self, mapping: pd.DataFrame) -> DonorPairDataset:
        """Build the outcome-free held-out donor pair dataset."""

        return DonorPairDataset(
            self.base_dataset,
            mapping,
            self.patient_index,
            expected_fold=self.fold,
        )


@dataclass(frozen=True)
class FoldEvaluationResult:
    """In-memory results plus paths committed by one successful evaluation."""

    fold: int
    output_dir: Path
    checkpoint_summary: dict[str, Any]
    readout_bundle: FoldReadoutBundle
    baseline_bundles: dict[str, FoldBaselineBundle]
    native_inference: FrozenInferenceArrays
    predictions: dict[str, pd.DataFrame]
    copy_current_metrics: pd.DataFrame
    paired_metrics: dict[str, pd.DataFrame]
    donor_context: DonorEvaluationContext
    artifact_paths: dict[str, Path]


def _fold(value: int) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) not in range(5):
        raise ValueError("fold 必须为 0..4")
    return int(value)


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} 不存在或不是文件：{path}")
    return path


def _integer_indices(values: Any, *, name: str, size: int) -> tuple[int, ...]:
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{name} 必须为 JSON integer array")
    output: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} 必须为 JSON integer，不接受布尔或浮点索引")
        integer = int(value)
        if integer < 0 or integer >= size:
            raise IndexError(f"{name} 含越界索引")
        output.append(integer)
    if not output or len(output) != len(set(output)):
        raise ValueError(f"{name} 必须非空且不得重复")
    return tuple(output)


def _load_splits(path: Path, payload: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"splits.json 不是有效 JSON：{path}") from error
    required = ("primary_train", "pretrain_train", "validation", "test")
    if not isinstance(raw, dict) or set(raw) != set(required):
        raise ValueError(f"splits.json 必须且只能包含 {required}")
    patient_count = len(payload["patient_ids"])
    splits = {
        name: _integer_indices(raw[name], name=f"splits.json:{name}", size=patient_count)
        for name in required
    }
    checkpoint_splits = payload["splits"]
    for name in required:
        checkpoint_values = _integer_indices(
            checkpoint_splits[name],
            name=f"checkpoint:{name}",
            size=patient_count,
        )
        if splits[name] != checkpoint_values:
            raise ValueError(f"splits.json:{name} 与 checkpoint 顺序或内容不一致")
    return splits


def _audit_split_ids(
    payload: Mapping[str, Any], splits: Mapping[str, Sequence[int]]
) -> dict[str, list[str]]:
    patient_ids = [str(value) for value in payload["patient_ids"]]
    return {
        "train": [patient_ids[index] for index in splits["primary_train"]],
        "val": [patient_ids[index] for index in splits["validation"]],
        "test": [patient_ids[index] for index in splits["test"]],
    }


def _validate_audit_provenance(
    payload: Mapping[str, Any],
    *,
    fold: int,
    splits: Mapping[str, Sequence[int]],
    legacy_x_cache_dir: Path,
    records: Sequence[PatientRecord],
    n_primary: int,
    config: Any,
) -> None:
    audit = payload.get("audit_provenance")
    if not isinstance(audit, Mapping):
        raise ValueError("checkpoint 缺少 audit_provenance；拒绝当作 audit 重训练 fold")
    if audit.get("schema_version") != 1 or audit.get("protocol") != AUDIT_PROTOCOL:
        raise ValueError("checkpoint audit_provenance schema/protocol 不匹配")
    if isinstance(audit.get("fold"), bool) or int(audit.get("fold", -1)) != fold:
        raise ValueError("checkpoint audit_provenance fold 与调用方不一致")

    expected_ids = _audit_split_ids(payload, splits)
    recorded_ids = audit.get("primary_split_patient_ids")
    if not isinstance(recorded_ids, Mapping) or set(recorded_ids) != set(expected_ids):
        raise ValueError("audit_provenance primary_split_patient_ids 不完整")
    for name, expected in expected_ids.items():
        if [str(value) for value in recorded_ids[name]] != expected:
            raise ValueError(f"audit_provenance {name} patient IDs/order 与 checkpoint 不一致")

    extra_ids = [record.patient_id for record in records[n_primary:]]
    if [str(value) for value in audit.get("extra_pretraining_patient_ids", ())] != extra_ids:
        raise ValueError("audit_provenance extra pretraining IDs/order 不一致")
    fit_indices = audit.get("fit_indices")
    if not isinstance(fit_indices, Mapping):
        raise ValueError("audit_provenance fit_indices 缺失")
    expected_fit = list(splits["pretrain_train"])
    for name in ("condition_encoder", "response_transform"):
        if list(fit_indices.get(name, ())) != expected_fit:
            raise ValueError(f"audit_provenance {name} fit indices 不一致")
    if int(audit.get("seed", -1)) != int(config.train.seed):
        raise ValueError("audit_provenance seed 与 checkpoint config 不一致")
    configured_output = Path(str(config.train.output_dir)).resolve()
    recorded_output = Path(str(audit.get("output_dir", ""))).resolve()
    if configured_output != recorded_output or configured_output.name != f"fold_{fold:02d}":
        raise ValueError("checkpoint config/audit output_dir 或 fold 目录名不一致")

    source = audit.get("base_dataset_source")
    if not isinstance(source, Mapping) or source.get("mode") != "verified_legacy_x_adapter":
        raise ValueError("checkpoint 不是用 verified legacy-x adapter 训练")
    recorded_cache = source.get("cache_dir")
    if not recorded_cache or Path(str(recorded_cache)).resolve() != legacy_x_cache_dir:
        raise ValueError("调用方 legacy-x cache 与 checkpoint provenance 不一致")
    if int(source.get("n_patient_files", -1)) != len(payload["patient_ids"]):
        raise ValueError("checkpoint legacy-x patient count provenance 不一致")
    if source.get("image_key") != "x" or source.get("geometry") != "clean mask_geometry(x[:,7])":
        raise ValueError("checkpoint legacy-x tensor/geometry contract 不一致")

    manifest = audit.get("manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("audit_provenance manifest 缺失")
    manifest_path = Path(str(manifest.get("path", ""))).resolve()
    expected_hash = str(manifest.get("sha256", ""))
    if manifest.get("hash_kind") == "file_sha256":
        _require_file(manifest_path, "fold manifest")
        if file_sha256(manifest_path) != expected_hash:
            raise ValueError("fold manifest 当前 SHA256 与 checkpoint provenance 不一致")
        primary_records = records[:n_primary]
        frame, summary = load_fold_manifest(
            manifest_path,
            expected_patient_ids=[record.patient_id for record in primary_records],
            expected_labels={record.patient_id: int(record.pcr) for record in primary_records},
        )
        if (
            int(manifest.get("n_rows", -1)) != int(summary["n_rows"])
            or int(manifest.get("n_patients", -1)) != int(summary["n_patients"])
        ):
            raise ValueError("audit_provenance manifest row/patient count 不一致")
        selected = frame.loc[frame["fold"].eq(fold)]
        for name, manifest_name in (("train", "train"), ("val", "val"), ("test", "test")):
            observed = set(
                selected.loc[selected["split"].eq(manifest_name), "patient_id"].astype(str)
            )
            if observed != set(expected_ids[name]):
                raise ValueError(f"fold manifest {name} 与 checkpoint split 不一致")
    elif manifest.get("hash_kind") != "canonical_validated_csv":
        raise ValueError("audit_provenance manifest hash_kind 未知")
    if len(expected_hash) != 64:
        raise ValueError("audit_provenance manifest SHA256 无效")


def _ordered_records(
    payload: Mapping[str, Any], config: Any
) -> tuple[tuple[PatientRecord, ...], int]:
    records, n_primary = clean_runner.load_experiment_records(config)
    if int(n_primary) != int(payload["n_primary"]):
        raise ValueError("当前 records n_primary 与 checkpoint 不一致")
    ordered = records_in_checkpoint_order(records, payload["patient_ids"])
    if [record.patient_id for record in ordered] != [
        str(value) for value in payload["patient_ids"]
    ]:
        raise ValueError("records 无法恢复 checkpoint 的精确 patient order")
    wrong_primary = [
        record.patient_id
        for record in ordered[:n_primary]
        if record.cohort.lower() != "ispy2"
    ]
    wrong_extra = [
        record.patient_id
        for record in ordered[n_primary:]
        if record.cohort.lower() != "ispy1"
    ]
    if wrong_primary or wrong_extra:
        raise ValueError(
            "records cohort partition 与 audit training contract 不一致："
            f"primary={wrong_primary[:3]}, extra={wrong_extra[:3]}"
        )
    labels = [record.pcr for record in ordered[:n_primary]]
    if any(value not in (0, 1) for value in labels):
        raise ValueError("I-SPY2 primary records 必须全部具有 0/1 pCR label")
    return tuple(ordered), int(n_primary)


def _load_frozen_states(
    path: Path,
    *,
    payload: Mapping[str, Any],
    records: Sequence[PatientRecord],
    n_primary: int,
    response_dim: int,
    latent_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = {
        "patient_ids",
        "future_response_state",
        "image_prediction",
        "pcr",
        "n_primary",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required.difference(archive.files))
        extra = sorted(set(archive.files).difference(required))
        if missing or extra:
            raise ValueError(f"frozen_states keys 不精确：missing={missing}, extra={extra}")
        patient_ids = np.asarray(archive["patient_ids"]).astype(str)
        states = np.asarray(archive["future_response_state"])
        image_prediction = np.asarray(archive["image_prediction"])
        labels = np.asarray(archive["pcr"])
        frozen_n_primary = np.asarray(archive["n_primary"])

    expected_ids = np.asarray([str(value) for value in payload["patient_ids"]])
    if patient_ids.ndim != 1 or not np.array_equal(patient_ids, expected_ids):
        raise ValueError("frozen_states patient_ids/order 与 checkpoint 不一致")
    if frozen_n_primary.ndim != 0 or int(frozen_n_primary) != n_primary:
        raise ValueError("frozen_states n_primary 与 checkpoint 不一致")
    if states.shape != (len(records), 3, response_dim):
        raise ValueError(
            "frozen future_response_state shape 不匹配："
            f"{states.shape} != {(len(records), 3, response_dim)}"
        )
    if image_prediction.shape != (len(records), 3, latent_dim):
        raise ValueError(
            f"frozen image_prediction shape 不匹配：{image_prediction.shape}"
        )
    if not np.isfinite(states).all() or not np.isfinite(image_prediction).all():
        raise ValueError("frozen_states 含非有限 latent")
    if labels.shape != (len(records),):
        raise ValueError("frozen_states pcr shape 不匹配")
    if labels.dtype.kind not in "iu":
        raise TypeError("frozen_states pcr dtype 必须为 integer")
    expected_labels = np.asarray(
        [record.pcr if record.pcr is not None else -1 for record in records],
        dtype=np.int64,
    )
    numeric_labels = labels.astype(np.int64)
    if not np.array_equal(numeric_labels, expected_labels):
        raise ValueError("frozen_states pcr 与当前 records label/order 不一致")
    return (
        states.astype(np.float32, copy=False),
        image_prediction.astype(np.float32, copy=False),
        expected_labels[:n_primary],
    )


def _make_loader(
    dataset: Dataset,
    indices: Sequence[int],
    *,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        drop_last=False,
    )


@torch.no_grad()
def _collect_native_image_prediction(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute the clean frozen export's image-transition representation."""

    if any(module.training for module in model.modules()):
        raise ValueError("image prediction validation 要求 model 全部处于 eval")
    patient_ids: list[str] = []
    predictions: list[np.ndarray] = []
    for raw_batch in loader:
        image = raw_batch["image"].to(device, non_blocking=True)
        geometry = raw_batch["geometry"].to(device, non_blocking=True)
        condition = raw_batch["condition"].to(device, non_blocking=True)
        ids = [str(value) for value in raw_batch["patient_id"]]
        if len(ids) != int(image.shape[0]):
            raise ValueError("image prediction batch patient_id 未与 tensor 对齐")
        visits = model.encode_visits(image, geometry)
        prediction = model.image_transition(visits[:, :-1], condition)
        patient_ids.extend(ids)
        predictions.append(prediction.detach().cpu().numpy())
    if not predictions:
        raise ValueError("image prediction validation loader 为空")
    return (
        np.asarray(patient_ids),
        np.concatenate(predictions).astype(np.float32),
    )


def load_validated_fold_inputs(
    fold_dir: str | Path,
    *,
    fold: int,
    legacy_x_cache_dir: str | Path,
    device: str | torch.device = "cpu",
) -> ValidatedFoldInputs:
    """Restore one fold only after every artifact contract agrees."""

    expected_fold = _fold(fold)
    directory = Path(fold_dir).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"fold_dir 不存在：{directory}")
    checkpoint_path = _require_file(directory / CHECKPOINT_NAME, "best checkpoint")
    frozen_path = _require_file(directory / FROZEN_STATES_NAME, "frozen states")
    splits_path = _require_file(directory / SPLITS_NAME, "split artifact")
    cache_dir = Path(legacy_x_cache_dir).resolve()
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"legacy-x cache 不存在：{cache_dir}")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA evaluation，但当前 CUDA 不可用")

    payload, summary = inspect_checkpoint(checkpoint_path)
    splits = _load_splits(splits_path, payload)
    model, config, condition_encoder = restore_model_for_evaluation(
        payload, device=resolved_device
    )
    records, n_primary = _ordered_records(payload, config)
    _validate_audit_provenance(
        payload,
        fold=expected_fold,
        splits=splits,
        legacy_x_cache_dir=cache_dir,
        records=records,
        n_primary=n_primary,
        config=config,
    )
    split_ids = _audit_split_ids(payload, splits)
    validate_checkpoint_payload(
        payload,
        expected_primary_ids=[record.patient_id for record in records[:n_primary]],
        expected_fold_ids=split_ids,
    )
    for record in records:
        condition_encoder.encode(record)

    states, image_prediction, labels = _load_frozen_states(
        frozen_path,
        payload=payload,
        records=records,
        n_primary=n_primary,
        response_dim=int(config.model.response_dim),
        latent_dim=int(config.model.latent_dim),
    )
    dataset = LegacyXCacheDataset(
        records,
        condition_encoder,  # StoredConditionEncoder implements the exact dataset API.
        cache_dir,
        expected_crop_size=tuple(config.data.crop_size),
    )
    checkpoint_id = f"{checkpoint_path}#sha256={summary['sha256']}"
    return ValidatedFoldInputs(
        fold=expected_fold,
        fold_dir=directory,
        checkpoint_path=checkpoint_path,
        checkpoint_id=checkpoint_id,
        checkpoint_summary=summary,
        payload=payload,
        config=config,
        model=model,
        condition_encoder=condition_encoder,
        records=records,
        n_primary=n_primary,
        splits=splits,
        frozen_response_state=states,
        frozen_image_prediction=image_prediction,
        labels=labels,
        dataset=dataset,
    )


def _geometry_feature_names() -> tuple[str, ...]:
    return (
        *(
            f"geometry_{component}_q{index}"
            for component in GEOMETRY_COMPONENTS
            for index in range(9)
        ),
        *(f"decision={name}" for name in DECISION_POINTS),
    )


def _build_baseline_features(
    records: Sequence[PatientRecord],
    train_indices: Sequence[int],
    native: FrozenInferenceArrays,
) -> dict[str, tuple[np.ndarray, tuple[str, ...], dict[str, Any]]]:
    train_records = [records[index] for index in train_indices]
    clinical_spec = ClinicalFeatureSpec.fit(train_records)
    clinical = clinical_features(records, clinical_spec)
    geometric = geometry_features(native.geometry)
    combined = clinical_geometry_features(records, native.geometry, clinical_spec)
    timepoint = timepoint_only_features(len(records))
    static_t0 = static_t0_features(native.t0_image_state)
    geometry_names = _geometry_feature_names()
    decision_names = tuple(f"decision={name}" for name in DECISION_POINTS)
    common_clinical = {
        "clinical_spec_fit": "fold_primary_train_only",
        "clinical_arms": list(clinical_spec.arms),
        "clinical_age_mean": clinical_spec.age_mean,
        "clinical_age_std": clinical_spec.age_std,
    }
    return {
        "F1": (
            clinical,
            tuple(clinical_spec.feature_names),
            {"constructor": "clinical_features", **common_clinical},
        ),
        "F2": (
            geometric,
            geometry_names,
            {"constructor": "geometry_features", "uses_visits": "q0:q2 causally by decision"},
        ),
        "F3": (
            combined,
            (*clinical_spec.feature_names_without_decision, *geometry_names),
            {
                "constructor": "clinical_geometry_features",
                "decision_code_copies": 1,
                **common_clinical,
            },
        ),
        "F4": (
            timepoint,
            decision_names,
            {"constructor": "timepoint_only_features", "inputs": "nominal decision only"},
        ),
        "F5": (
            static_t0,
            (
                *(
                    f"static_t0_appearance_{index:03d}"
                    for index in range(native.t0_image_state.shape[1])
                ),
                *decision_names,
            ),
            {
                "constructor": "static_t0_features",
                "appearance_extractor": "model.projector(model.encoder(image[:,0]))",
                "geometry_in_t0_representation": False,
                "followup_inputs": False,
            },
        ),
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_outputs(
    output_dir: Path,
    *,
    validated: ValidatedFoldInputs,
    readout: FoldReadoutBundle,
    baselines: Mapping[str, FoldBaselineBundle],
    predictions: Mapping[str, pd.DataFrame],
    copy_current: pd.DataFrame,
    paired: Mapping[str, pd.DataFrame],
) -> dict[str, Path]:
    if os.path.lexists(output_dir):
        raise FileExistsError(f"拒绝覆盖 evaluation output_dir：{output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-",
            dir=output_dir.parent,
        )
    )
    paths: dict[str, Path] = {}
    try:
        readout_path = staging / "readout" / "fold_readout.pkl"
        save_readout_bundle(readout, readout_path)
        paths["readout_bundle"] = readout_path
        readout_metadata = staging / "readout" / "fold_readout_metadata.json"
        _write_json(readout_metadata, readout.audit_metadata())
        paths["readout_metadata"] = readout_metadata

        for baseline_id, bundle in baselines.items():
            path = staging / "baselines" / f"{baseline_id.lower()}_bundle.pkl"
            save_baseline_bundle(bundle, path)
            paths[f"baseline_{baseline_id}"] = path
            metadata_path = staging / "baselines" / f"{baseline_id.lower()}_metadata.json"
            _write_json(metadata_path, bundle.audit_metadata())
            paths[f"baseline_{baseline_id}_metadata"] = metadata_path

        prediction_dir = staging / "predictions"
        for name, frame in predictions.items():
            path = prediction_dir / f"{name}.csv"
            write_prediction_csv(frame, path)
            paths[f"predictions_{name}"] = path

        latent_dir = staging / "latent"
        latent_dir.mkdir(parents=True, exist_ok=True)
        copy_path = latent_dir / "copy_current.csv"
        copy_current.to_csv(copy_path, index=False)
        paths["copy_current"] = copy_path
        for name, frame in paired.items():
            path = latent_dir / f"paired_{name}.csv"
            frame.to_csv(path, index=False)
            paths[f"paired_{name}"] = path

        provenance_path = staging / "evaluation_provenance.json"
        _write_json(
            provenance_path,
            {
                "schema_version": EVALUATION_SCHEMA_VERSION,
                "fold": validated.fold,
                "checkpoint": validated.checkpoint_summary,
                "checkpoint_id": validated.checkpoint_id,
                "fold_dir": str(validated.fold_dir),
                "patient_counts": {
                    "all_records": len(validated.records),
                    "primary": validated.n_primary,
                    "train": len(validated.splits["primary_train"]),
                    "validation": len(validated.splits["validation"]),
                    "test": len(validated.splits["test"]),
                },
                "audits": {
                    "B": "copy_current_vs_learned_combined_prediction",
                    "C1": "repeated_t0_mri_channels_only",
                    "C2": "repeated_t0_full_image_derived",
                    "D": "paired_t1_t2_temporal_swap_native_target_fixed",
                    "E": "deferred_to_donor_inference_with_returned_context",
                    "F": dict(BASELINE_FEATURE_SETS),
                },
                "f5_contract": "projector(encoder(T0 image)); no geometry_projector",
                "output_files": {
                    key: str(path.relative_to(staging)) for key, path in paths.items()
                },
            },
        )
        paths["provenance"] = provenance_path

        # Recheck immediately before the single-directory commit.  Because the
        # staging directory is a sibling, rename stays on one filesystem.
        if os.path.lexists(output_dir):
            raise FileExistsError(f"拒绝并发覆盖 evaluation output_dir：{output_dir}")
        staging.rename(output_dir)
        return {
            key: output_dir / path.relative_to(staging)
            for key, path in paths.items()
        }
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def evaluate_retrained_fold(
    fold_dir: str | Path,
    *,
    fold: int,
    legacy_x_cache_dir: str | Path,
    output_dir: str | Path,
    device: str | torch.device = "cpu",
    batch_size: int | None = None,
    workers: int | None = None,
    readout_config: AuditReadoutConfig | Mapping[str, Any] | Any | None = None,
    baseline_config: BaselineReadoutConfig | Mapping[str, Any] | None = None,
    frozen_state_rtol: float = 1e-4,
    frozen_state_atol: float = 1e-5,
) -> FoldEvaluationResult:
    """Run B/C/D/F for one completed audit fold and persist fresh artifacts.

    The output directory is mandatory and must not already exist.  All model
    selection uses primary train/validation only; held-out labels enter only
    the prediction-frame reporting column.
    """

    destination = Path(output_dir).resolve()
    if os.path.lexists(destination):
        raise FileExistsError(f"拒绝覆盖 evaluation output_dir：{destination}")
    if not np.isfinite(frozen_state_rtol) or frozen_state_rtol < 0:
        raise ValueError("frozen_state_rtol 必须为有限非负数")
    if not np.isfinite(frozen_state_atol) or frozen_state_atol < 0:
        raise ValueError("frozen_state_atol 必须为有限非负数")
    validated = load_validated_fold_inputs(
        fold_dir,
        fold=fold,
        legacy_x_cache_dir=legacy_x_cache_dir,
        device=device,
    )
    resolved_device = torch.device(device)
    effective_batch = validated.config.train.batch_size if batch_size is None else batch_size
    effective_workers = validated.config.train.workers if workers is None else workers
    if (
        isinstance(effective_batch, bool)
        or int(effective_batch) != effective_batch
        or int(effective_batch) <= 0
    ):
        raise ValueError("batch_size 必须为正整数")
    if (
        isinstance(effective_workers, bool)
        or int(effective_workers) != effective_workers
        or int(effective_workers) < 0
    ):
        raise ValueError("workers 必须为非负整数")
    effective_batch, effective_workers = int(effective_batch), int(effective_workers)

    primary_indices = tuple(range(validated.n_primary))
    primary_loader = _make_loader(
        validated.dataset,
        primary_indices,
        batch_size=effective_batch,
        workers=effective_workers,
        device=resolved_device,
    )
    native = collect_frozen_inference(
        validated.model,
        primary_loader,
        device=resolved_device,
    )
    expected_primary_ids = np.asarray(
        [record.patient_id for record in validated.records[: validated.n_primary]]
    )
    if not np.array_equal(native.patient_ids.astype(str), expected_primary_ids):
        raise RuntimeError("runtime inference patient order 与 checkpoint primary order 不一致")
    frozen_primary = validated.frozen_response_state[: validated.n_primary]
    if not np.allclose(
        native.response_state,
        frozen_primary,
        rtol=float(frozen_state_rtol),
        atol=float(frozen_state_atol),
    ):
        maximum = float(np.max(np.abs(native.response_state - frozen_primary)))
        raise RuntimeError(
            "checkpoint+legacy cache 无法复现 frozen future_response_state；"
            f"max_abs_diff={maximum:.6g}"
        )
    image_ids, image_prediction = _collect_native_image_prediction(
        validated.model,
        primary_loader,
        device=resolved_device,
    )
    if not np.array_equal(image_ids.astype(str), expected_primary_ids):
        raise RuntimeError("image prediction patient order 与 checkpoint 不一致")
    frozen_image_prediction = validated.frozen_image_prediction[
        : validated.n_primary
    ]
    if not np.allclose(
        image_prediction,
        frozen_image_prediction,
        rtol=float(frozen_state_rtol),
        atol=float(frozen_state_atol),
    ):
        maximum = float(
            np.max(np.abs(image_prediction - frozen_image_prediction))
        )
        raise RuntimeError(
            "checkpoint+legacy cache 无法复现 frozen image_prediction；"
            f"max_abs_diff={maximum:.6g}"
        )

    train = validated.splits["primary_train"]
    validation = validated.splits["validation"]
    test = validated.splits["test"]
    primary_records = validated.records[: validated.n_primary]
    labels = validated.labels
    readout = fit_fold_readout(
        frozen_primary,
        labels,
        expected_primary_ids,
        train,
        validation,
        fold=validated.fold,
        test_indices=test,
        config=validated.config.readout if readout_config is None else readout_config,
    )
    test_ids = expected_primary_ids[list(test)]
    test_labels = labels[list(test)]
    native_prediction = predict_fold_readout(
        readout,
        frozen_primary[list(test)],
        test_ids,
        test_labels,
        checkpoint=validated.checkpoint_id,
        audit_condition="native",
    )

    heldout_loader = _make_loader(
        validated.dataset,
        test,
        batch_size=effective_batch,
        workers=effective_workers,
        device=resolved_device,
    )
    copy_current = copy_current_latent_audit(
        validated.model,
        heldout_loader,
        device=resolved_device,
        fold=validated.fold,
        checkpoint=validated.checkpoint_id,
    )
    perturbation_predictions: list[pd.DataFrame] = []
    paired: dict[str, pd.DataFrame] = {}
    for name, perturbation in PERTURBATIONS.items():
        perturbed = collect_frozen_inference(
            validated.model,
            heldout_loader,
            device=resolved_device,
            perturbation=perturbation,
        )
        if not np.array_equal(perturbed.patient_ids.astype(str), test_ids):
            raise RuntimeError(f"{name} patient order 与 held-out split 不一致")
        perturbation_predictions.append(
            predict_fold_readout(
                readout,
                perturbed.response_state,
                perturbed.patient_ids,
                test_labels,
                checkpoint=validated.checkpoint_id,
                audit_condition=name,
            )
        )
        paired[name] = paired_perturbation_latent_audit(
            validated.model,
            heldout_loader,
            perturbation,
            device=resolved_device,
            fold=validated.fold,
            checkpoint=validated.checkpoint_id,
            audit_condition=name,
        )

    feature_sets = _build_baseline_features(primary_records, train, native)
    baselines: dict[str, FoldBaselineBundle] = {}
    baseline_predictions: list[pd.DataFrame] = []
    for baseline_id in BASELINE_FEATURE_SETS:
        features, names, provenance = feature_sets[baseline_id]
        bundle = fit_fold_baseline(
            features,
            labels,
            expected_primary_ids,
            train,
            validation,
            fold=validated.fold,
            baseline_id=baseline_id,
            feature_names=names,
            test_indices=test,
            config=baseline_config,
            feature_provenance=provenance,
        )
        baselines[baseline_id] = bundle
        baseline_predictions.append(
            predict_fold_baseline(
                bundle,
                features[list(test)],
                test_ids,
                test_labels,
                feature_names=names,
                checkpoint=validated.checkpoint_id,
            )
        )

    perturbed_prediction = pd.concat(perturbation_predictions, ignore_index=True)
    baseline_prediction = pd.concat(baseline_predictions, ignore_index=True)
    validate_label_alignment(
        [native_prediction, *perturbation_predictions, *baseline_predictions]
    )
    predictions = {
        "native": native_prediction,
        "perturbations": perturbed_prediction,
        "baselines": baseline_prediction,
    }
    artifact_paths = _write_outputs(
        destination,
        validated=validated,
        readout=readout,
        baselines=baselines,
        predictions=predictions,
        copy_current=copy_current,
        paired=paired,
    )

    heldout_records = tuple(primary_records[index] for index in test)
    heldout_indices = set(test)
    donor_context = DonorEvaluationContext(
        fold=validated.fold,
        checkpoint=validated.checkpoint_id,
        model=validated.model,
        readout_bundle=readout,
        base_dataset=validated.dataset,
        patient_index={
            record.patient_id: int(index)
            for index, record in enumerate(primary_records)
            if index in heldout_indices
        },
        heldout_records=heldout_records,
        labels_by_patient={record.patient_id: int(record.pcr) for record in heldout_records},
        device=resolved_device,
    )
    return FoldEvaluationResult(
        fold=validated.fold,
        output_dir=destination,
        checkpoint_summary=validated.checkpoint_summary,
        readout_bundle=readout,
        baseline_bundles=baselines,
        native_inference=native,
        predictions=predictions,
        copy_current_metrics=copy_current,
        paired_metrics=paired,
        donor_context=donor_context,
        artifact_paths=artifact_paths,
    )


__all__ = [
    "DonorEvaluationContext",
    "EVALUATION_SCHEMA_VERSION",
    "FoldEvaluationResult",
    "ValidatedFoldInputs",
    "evaluate_retrained_fold",
    "load_validated_fold_inputs",
]

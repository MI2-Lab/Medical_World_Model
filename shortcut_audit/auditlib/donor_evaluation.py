"""Single-fold orchestration for the matched follow-up donor audit (E).

The orchestration deliberately keeps matching and reporting supervision on
opposite sides of model inference.  Donors are selected only from the baseline
columns consumed by :mod:`shortcut_audit.auditlib.matching`; recipient pCR is
looked up only after all donor states and probabilities have been computed.

Outputs are committed as one new directory.  An existing target is never
overwritten, and a failed run leaves no partially committed audit directory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .contracts import DECISION_POINTS, write_prediction_csv
from .donor_inference import (
    TRANSITION_NAMES,
    DonorPairDataset,
    DonorSwapInference,
    donor_prediction_frame,
    run_donor_swap_inference,
)
from .matching import MatchingConfig, MatchingResult, match_follow_up_donors
from .provenance import file_sha256
from .readouts import FoldReadoutBundle


DONOR_AUDIT_SCHEMA_VERSION = "shortcut_audit.matched_donor_fold.v1"
AUDIT_CONDITION = "matched_followup_swap"
ARTIFACT_FILENAMES = {
    "mapping": "donor_mapping.csv",
    "recipient_diagnostics": "matching_recipient_diagnostics.csv",
    "failures": "matching_failures.csv",
    "balance": "matching_balance.csv",
    "success": "matching_success.json",
    "latent_diagnostics": "latent_diagnostics.csv",
    "predictions": "predictions.csv",
    "provenance": "provenance.json",
}


@dataclass(frozen=True)
class MatchedDonorFoldAudit:
    """In-memory result and the atomically committed artifact directory."""

    fold: int
    output_dir: Path
    mapping: pd.DataFrame
    recipient_diagnostics: pd.DataFrame
    failures: pd.DataFrame
    balance_stats: pd.DataFrame
    success_stats: Mapping[str, Any]
    latent_diagnostics: pd.DataFrame
    predictions: pd.DataFrame
    provenance: Mapping[str, Any]


def _validate_fold(value: Any) -> int:
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


def _positive_integer(value: Any, *, name: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须为{'非负' if allow_zero else '正'}整数")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} 必须为{'非负' if allow_zero else '正'}整数"
        ) from error
    lower_bound = 0 if allow_zero else 1
    if not math.isfinite(numeric) or not numeric.is_integer() or numeric < lower_bound:
        raise ValueError(f"{name} 必须为{'非负' if allow_zero else '正'}整数")
    return int(numeric)


def _jsonable(value: Any) -> Any:
    """Convert audit metadata to strict JSON values and reject NaN silently."""

    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("provenance JSON 不得含 NaN/Inf")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"provenance 含不可序列化类型：{type(value).__name__}")


def _canonical_json_bytes(value: Any) -> bytes:
    normalized = _jsonable(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    payload = _jsonable(value)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(
            payload,
            stream,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _matching_columns(
    frame: pd.DataFrame,
    config: MatchingConfig,
) -> tuple[str, ...]:
    """Return only the baseline columns the validated matcher may consume."""

    columns: list[str] = [
        config.patient_id_col,
        config.fold_col,
        config.hr_col,
        config.her2_col,
        config.treatment_family_col,
        config.baseline_volume_col,
        *config.visit_availability_cols,
        *config.matching_features,
    ]
    for optional in (config.subtype_col, config.age_col, config.mammaprint_col):
        if optional is not None and optional in frame.columns:
            columns.append(optional)
    return tuple(dict.fromkeys(columns))


def _baseline_frame_sha256(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    """Fingerprint matching inputs without touching coexisting outcome columns."""

    selected = frame.loc[:, list(columns)]
    digest = hashlib.sha256()
    digest.update(_canonical_json_bytes(list(columns)))
    digest.update(_canonical_json_bytes([str(dtype) for dtype in selected.dtypes]))
    digest.update(
        pd.util.hash_pandas_object(selected, index=False, categorize=True)
        .to_numpy(dtype=np.uint64)
        .tobytes()
    )
    return digest.hexdigest()


def _patient_order_sha256(patient_ids: Sequence[str]) -> str:
    return hashlib.sha256(
        _canonical_json_bytes([str(value) for value in patient_ids])
    ).hexdigest()


def _validate_heldout_order(
    frame: pd.DataFrame,
    config: MatchingConfig,
    bundle: FoldReadoutBundle,
    fold: int,
) -> tuple[str, ...]:
    for column in (config.patient_id_col, config.fold_col):
        if column not in frame.columns:
            raise KeyError(f"held-out metadata 缺少列：{column}")
    patient_ids = tuple(frame[config.patient_id_col].astype(str))
    if not patient_ids or any(not value.strip() for value in patient_ids):
        raise ValueError("held-out patient ID 不得为空")
    if len(patient_ids) != len(set(patient_ids)):
        raise ValueError("held-out metadata 含重复 patient ID")
    if not bundle.test_patient_ids:
        raise ValueError("readout bundle 未记录 test patient IDs")
    if patient_ids != tuple(bundle.test_patient_ids):
        raise ValueError("held-out metadata patient/order 与 readout test split 不一致")
    numeric_fold = pd.to_numeric(frame[config.fold_col], errors="raise").to_numpy(
        dtype=np.float64
    )
    if (
        not np.isfinite(numeric_fold).all()
        or not np.equal(numeric_fold, np.floor(numeric_fold)).all()
        or not np.equal(numeric_fold, fold).all()
    ):
        raise ValueError("held-out metadata 含非当前 fold patient")
    if bundle.fold != fold:
        raise ValueError("readout bundle fold 与 donor audit fold 不一致")
    return patient_ids


def _augment_and_validate_mapping(
    result: MatchingResult,
    patient_ids: Sequence[str],
    *,
    fold: int,
    seed: int,
) -> pd.DataFrame:
    mapping = result.mapping.copy().reset_index(drop=True)
    if mapping.empty:
        raise ValueError("当前 held-out fold 没有任何可运行的 donor mapping")
    mapping.insert(0, "pair_index", np.arange(len(mapping), dtype=np.int64))
    mapping["matching_relaxed"] = mapping["matching_level"].ne(
        "hard_subtype_treatment_visit"
    )
    # The repository's matching contract calls the treatment-arm grouping
    # ``treatment_family``.  This field records exactly whether that hard match
    # was relaxed; it must not be interpreted as exact regimen equality.
    mapping["treatment_family_relaxed"] = ~mapping["treatment_family_match"].astype(
        bool
    )
    # Short name requested by the audit tables.  Its semantics are explicitly
    # treatment-family relaxation, not exact regimen/arm equality.
    mapping["arm_relaxed"] = mapping["treatment_family_relaxed"]

    if not mapping["fold"].eq(fold).all():
        raise RuntimeError("matcher 返回跨 fold mapping")
    if not mapping["matching_seed"].eq(seed).all():
        raise RuntimeError("matcher 返回非预期 seed")
    if mapping["recipient_patient_id"].eq(mapping["donor_patient_id"]).any():
        raise RuntimeError("matcher 返回 self donor")
    heldout = set(patient_ids)
    observed = set(mapping["recipient_patient_id"].astype(str)) | set(
        mapping["donor_patient_id"].astype(str)
    )
    if not observed.issubset(heldout):
        raise RuntimeError("matcher 返回 held-out fold 之外 patient")
    if mapping.duplicated(
        ["recipient_patient_id", "donor_patient_id"], keep=False
    ).any():
        raise RuntimeError("同一 recipient 重复选择同一 donor")

    diagnostic_ids = tuple(
        result.recipient_diagnostics["recipient_patient_id"].astype(str)
    )
    if diagnostic_ids != tuple(patient_ids):
        raise RuntimeError("matching recipient diagnostics patient/order 漂移")

    patient_rank = {patient_id: rank for rank, patient_id in enumerate(patient_ids)}
    previous_rank = -1
    for recipient_id, group in mapping.groupby(
        "recipient_patient_id", sort=False, dropna=False
    ):
        recipient_id = str(recipient_id)
        rank = patient_rank[recipient_id]
        if rank <= previous_rank:
            raise RuntimeError("donor mapping recipient 顺序不稳定")
        previous_rank = rank
        repetitions = group["audit_repetition"].astype(int).tolist()
        if repetitions != list(range(1, len(group) + 1)):
            raise RuntimeError("donor mapping repetition 必须从 1 连续递增")
    return mapping


def _validate_latent_order(frame: pd.DataFrame, n_pairs: int) -> None:
    expected = [
        (pair_index, transition)
        for pair_index in range(n_pairs)
        for transition in TRANSITION_NAMES
    ]
    observed = list(
        zip(frame["pair_index"].astype(int), frame["transition"], strict=True)
    )
    if observed != expected:
        raise RuntimeError("latent diagnostics pair/transition 顺序漂移")


def _validate_prediction_order(
    frame: pd.DataFrame,
    mapping: pd.DataFrame,
) -> None:
    expected = [
        (
            str(row.recipient_patient_id),
            str(row.donor_patient_id),
            int(row.audit_repetition),
            decision_point,
        )
        for row in mapping.itertuples(index=False)
        for decision_point in DECISION_POINTS
    ]
    observed = list(
        zip(
            frame["patient_id"].astype(str),
            frame["donor_patient_id"].astype(str),
            frame["repetition_id"].astype(int),
            frame["decision_point"].astype(str),
            strict=True,
        )
    )
    if observed != expected:
        raise RuntimeError("donor predictions patient/donor/decision 顺序漂移")


def _checkpoint_identity(checkpoint: str | Path) -> dict[str, Any]:
    checkpoint_text = str(checkpoint).strip()
    if not checkpoint_text:
        raise ValueError("checkpoint provenance 不得为空")
    embedded_sha256: str | None = None
    path_text = checkpoint_text
    if "#sha256=" in checkpoint_text:
        path_text, embedded_sha256 = checkpoint_text.rsplit("#sha256=", 1)
        if re.fullmatch(r"[0-9a-fA-F]{64}", embedded_sha256) is None:
            raise ValueError("checkpoint reference 内嵌 SHA256 格式无效")
        embedded_sha256 = embedded_sha256.lower()
    path = Path(path_text)
    identity: dict[str, Any] = {
        "reference": checkpoint_text,
        "sha256": embedded_sha256,
    }
    if path.is_file():
        identity["path"] = str(path.resolve())
        identity["size_bytes"] = int(path.stat().st_size)
        actual_sha256 = file_sha256(path)
        if embedded_sha256 is not None and actual_sha256 != embedded_sha256:
            raise ValueError("checkpoint reference 内嵌 SHA256 与文件不一致")
        identity["sha256"] = actual_sha256
    return identity


def _commit_outputs(
    output_dir: Path,
    *,
    mapping: pd.DataFrame,
    matching: MatchingResult,
    latent: pd.DataFrame,
    predictions: pd.DataFrame,
    success: Mapping[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"拒绝覆盖既有 donor audit 目录：{output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        frames = {
            "mapping": mapping,
            "recipient_diagnostics": matching.recipient_diagnostics,
            "failures": matching.failures,
            "balance": matching.balance_stats,
            "latent_diagnostics": latent,
        }
        for key, frame in frames.items():
            frame.to_csv(staging / ARTIFACT_FILENAMES[key], index=False)
        write_prediction_csv(
            predictions,
            staging / ARTIFACT_FILENAMES["predictions"],
            require_donor=True,
        )
        _write_json(staging / ARTIFACT_FILENAMES["success"], success)

        artifact_hashes = {
            key: {
                "filename": filename,
                "sha256": file_sha256(staging / filename),
            }
            for key, filename in ARTIFACT_FILENAMES.items()
            if key != "provenance"
        }
        provenance["artifacts"] = artifact_hashes
        _write_json(staging / ARTIFACT_FILENAMES["provenance"], provenance)

        # Check again immediately before the atomic same-filesystem rename.
        if output_dir.exists():
            raise FileExistsError(f"拒绝覆盖并发产生的 donor audit 目录：{output_dir}")
        os.rename(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return provenance


def run_matched_donor_fold_audit(
    *,
    fold: int,
    heldout_metadata: pd.DataFrame,
    base_dataset: Dataset,
    patient_index: Mapping[str, int],
    model: torch.nn.Module,
    readout_bundle: FoldReadoutBundle,
    labels_by_patient: Mapping[str, int],
    checkpoint: str | Path,
    output_dir: str | Path,
    matching_config: MatchingConfig | None = None,
    device: str | torch.device = "cpu",
    inference_batch_size: int = 4,
    num_workers: int = 0,
    pin_memory: bool = False,
    caller_provenance: Mapping[str, Any] | None = None,
) -> MatchedDonorFoldAudit:
    """Run and atomically export experiment E for one held-out fold.

    Matching sees only baseline-observable columns selected inside
    :func:`match_follow_up_donors`.  ``labels_by_patient`` is not accessed until
    donor inference has finished and is used solely to populate ``y_true``.
    The primary FLR consumes geometry plus condition, so donor MRI replacement
    itself cannot change the primary score; donor geometry can.  MRI effects
    remain represented by the native-target latent diagnostics.
    """

    fold = _validate_fold(fold)
    if not isinstance(heldout_metadata, pd.DataFrame):
        raise TypeError("heldout_metadata 必须是 pandas.DataFrame")
    if not isinstance(base_dataset, Dataset):
        raise TypeError("base_dataset 必须是 torch Dataset")
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model 必须是 torch.nn.Module")
    if any(module.training for module in model.modules()):
        raise ValueError("matched donor audit 要求 model 全部处于 eval")
    inference_batch_size = _positive_integer(
        inference_batch_size, name="inference_batch_size"
    )
    num_workers = _positive_integer(num_workers, name="num_workers", allow_zero=True)
    output_path = Path(output_dir)
    if output_path.exists():
        raise FileExistsError(f"拒绝覆盖既有 donor audit 目录：{output_path}")
    checkpoint_identity = _checkpoint_identity(checkpoint)
    config = matching_config or MatchingConfig()
    patient_ids = _validate_heldout_order(
        heldout_metadata, config, readout_bundle, fold
    )

    # match_follow_up_donors validates every consumed column name before it
    # selects values, and deliberately excludes any coexisting pCR/outcome.
    matching = match_follow_up_donors(heldout_metadata, config)
    matching_columns = _matching_columns(heldout_metadata, config)
    baseline_sha256 = _baseline_frame_sha256(heldout_metadata, matching_columns)
    mapping = _augment_and_validate_mapping(
        matching,
        patient_ids,
        fold=fold,
        seed=config.seed,
    )

    pair_dataset = DonorPairDataset(
        base_dataset,
        mapping,
        patient_index,
        expected_fold=fold,
    )
    loader = DataLoader(
        pair_dataset,
        batch_size=inference_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(pin_memory),
        drop_last=False,
    )
    inference: DonorSwapInference = run_donor_swap_inference(
        model,
        loader,
        mapping=mapping,
        device=device,
        fold=fold,
        checkpoint=checkpoint_identity["reference"],
    )
    latent = inference.latent_metrics.reset_index(drop=True)
    _validate_latent_order(latent, len(mapping))

    # This is intentionally the first access to recipient outcomes.
    predictions = donor_prediction_frame(
        readout_bundle,
        inference,
        labels_by_patient,
        checkpoint=checkpoint_identity["reference"],
        audit_condition=AUDIT_CONDITION,
    ).reset_index(drop=True)
    _validate_prediction_order(predictions, mapping)

    success = dict(matching.success_stats)
    success.update(
        {
            "fold": fold,
            "n_treatment_family_relaxed_pairs": int(
                mapping["treatment_family_relaxed"].sum()
            ),
            "treatment_family_relaxed_pair_rate": float(
                mapping["treatment_family_relaxed"].mean()
            ),
            "n_any_relaxed_pairs": int(mapping["matching_relaxed"].sum()),
            "any_relaxed_pair_rate": float(mapping["matching_relaxed"].mean()),
            "n_arm_relaxed_pairs": int(mapping["arm_relaxed"].sum()),
            "arm_relaxed_pair_rate": float(mapping["arm_relaxed"].mean()),
        }
    )
    readout_metadata = readout_bundle.audit_metadata()
    provenance: dict[str, Any] = {
        "schema_version": DONOR_AUDIT_SCHEMA_VERSION,
        "audit_condition": AUDIT_CONDITION,
        "fold": fold,
        "checkpoint": checkpoint_identity,
        "readout": {
            "schema_version": readout_bundle.schema_version,
            "fold": readout_bundle.fold,
            "audit_metadata_sha256": hashlib.sha256(
                _canonical_json_bytes(readout_metadata)
            ).hexdigest(),
            "thresholds": dict(readout_bundle.thresholds),
        },
        "matching": {
            "config": asdict(config),
            "baseline_columns_consumed": list(matching_columns),
            "coexisting_columns_excluded": sorted(
                set(heldout_metadata.columns).difference(matching_columns)
            ),
            "baseline_input_sha256": baseline_sha256,
            "fingerprint_method": "pandas_hash_pandas_object_uint64_plus_columns_and_dtypes",
            "outcome_blind": True,
            "default_requested_donors": 10,
            "requested_donors": int(config.max_donors),
            "relaxation_is_explicit_opt_in": bool(config.allow_relaxed_matches),
            "treatment_family_relaxed_field": "treatment_family_relaxed",
            "arm_relaxed_field": "arm_relaxed",
            "arm_relaxed_semantics": (
                "treatment-family hard-match relaxation; not exact regimen equality"
            ),
        },
        "ordering": {
            "heldout_patient_ids_sha256": _patient_order_sha256(patient_ids),
            "n_heldout_patients": len(patient_ids),
            "n_pairs": len(mapping),
            "decision_points": list(DECISION_POINTS),
            "transitions": list(TRANSITION_NAMES),
            "prediction_order": "mapping row, then DECISION_POINTS",
        },
        "contracts": {
            "donors": "same held-out fold, non-self, visit-compatible",
            "target": "fixed recipient native EMA target for every donor pair",
            "recipient_outcome_use": "prediction reporting only, after inference",
            "primary_readout_dependency": "geometry plus condition only",
            "primary_interpretation_boundary": (
                "donor MRI alone cannot alter the primary FLR; donor geometry may; "
                "MRI replacement remains assessable through latent diagnostics"
            ),
        },
        "software": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
        },
        "rows": {
            "mapping": len(mapping),
            "recipient_diagnostics": len(matching.recipient_diagnostics),
            "failures": len(matching.failures),
            "balance": len(matching.balance_stats),
            "latent_diagnostics": len(latent),
            "predictions": len(predictions),
        },
        "caller": dict(caller_provenance or {}),
    }
    # Validate caller data and the complete metadata before any artifact write.
    provenance = _jsonable(provenance)
    provenance = _commit_outputs(
        output_path,
        mapping=mapping,
        matching=matching,
        latent=latent,
        predictions=predictions,
        success=success,
        provenance=provenance,
    )
    return MatchedDonorFoldAudit(
        fold=fold,
        output_dir=output_path,
        mapping=mapping,
        recipient_diagnostics=matching.recipient_diagnostics.copy(),
        failures=matching.failures.copy(),
        balance_stats=matching.balance_stats.copy(),
        success_stats=success,
        latent_diagnostics=latent,
        predictions=predictions,
        provenance=provenance,
    )


__all__ = [
    "ARTIFACT_FILENAMES",
    "AUDIT_CONDITION",
    "DONOR_AUDIT_SCHEMA_VERSION",
    "MatchedDonorFoldAudit",
    "run_matched_donor_fold_audit",
]

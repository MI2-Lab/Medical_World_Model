"""既有 checkpoint、split、FTV 与训练 history 的只读闭环。"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch

from .contracts import (
    EXPECTED_SOURCE_SHA256,
    FOLDS,
    LAMBDA_FTV,
    SEED_BASES,
    SOURCE_ROOT,
    SOURCE_SRC,
    assert_source_hashes,
    assert_audit_config,
    file_sha256,
    repo_relative,
)

if str(SOURCE_SRC) not in sys.path:
    sys.path.insert(0, str(SOURCE_SRC))

from dgrs.config import load_config, resolve_path  # noqa: E402
from dgrs.data import PatientRecord, cache_index, read_raw_ftv  # noqa: E402
from dgrs.targets import PooledFTVTransform, patient_hash, raw_ftv_hash  # noqa: E402


@dataclass(frozen=True)
class AuditDataContext:
    fold_frame: pd.DataFrame
    primary_ids: frozenset[str]
    cache: Mapping[str, Path]
    raw_ftv: Mapping[str, tuple[np.ndarray, np.ndarray]]
    source_config: Mapping[str, Any]


ASSET_MANIFEST_COLUMNS = (
    "schema_version",
    "seed_base",
    "fold",
    "effective_seed",
    "checkpoint_kind",
    "checkpoint",
    "checkpoint_sha256",
    "checkpoint_bytes",
    "checkpoint_epoch",
    "history",
    "history_sha256",
    "selection",
    "selection_sha256",
    "checkpoint_mirrored_tensor_entries_checked",
    "shared_initialization_sha256",
    "state_distinct_from_selected",
    "formal_inference_checkpoint",
    "contains_patient_ids",
)


def _strict_bool(value: Any, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ValueError(f"{name} 非严格 boolean: {value!r}")


def checkpoint_path(
    seed_base: int, fold: int, model: str = "g3", kind: str = "best"
) -> Path:
    if seed_base not in SEED_BASES or fold not in FOLDS:
        raise ValueError("checkpoint grid key 非法")
    model = str(model).lower()
    if model not in {"g1", "g3"} or kind not in {"best", "last", "fallback"}:
        raise ValueError("checkpoint model/kind 非法")
    return (
        SOURCE_ROOT
        / "checkpoints"
        / "formal"
        / f"seed_{seed_base}"
        / model
        / f"fold_{fold}"
        / f"{kind}.pt"
    )


def history_path(seed_base: int, fold: int, model: str = "g3") -> Path:
    return (
        SOURCE_ROOT
        / "metrics"
        / "training"
        / "formal"
        / f"seed_{seed_base}"
        / str(model).lower()
        / f"fold_{fold}.csv"
    )


def selection_path(seed_base: int, fold: int, model: str = "g3") -> Path:
    return checkpoint_path(seed_base, fold, model, "best").with_name("selection.json")


def _load_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint 顶层不是 mapping: {path}")
    return payload


def _state_mirror_is_exact(payload: Mapping[str, Any]) -> bool:
    left = payload.get("state_dict")
    right = payload.get("model_state")
    if (
        not isinstance(left, Mapping)
        or not isinstance(right, Mapping)
        or set(left) != set(right)
    ):
        return False
    return all(
        isinstance(left[name], torch.Tensor)
        and isinstance(right[name], torch.Tensor)
        and left[name].shape == right[name].shape
        and left[name].dtype == right[name].dtype
        and torch.equal(left[name], right[name])
        for name in left
    )


def _all_tensors_finite(payload: Mapping[str, Any]) -> tuple[bool, int]:
    count = 0
    for state_name in ("state_dict", "model_state"):
        state = payload.get(state_name)
        if not isinstance(state, Mapping):
            return False, count
        for value in state.values():
            if not isinstance(value, torch.Tensor):
                return False, count
            count += 1
            if not bool(torch.isfinite(value).all()):
                return False, count
    return True, count


def load_data_context() -> AuditDataContext:
    assert_audit_config()
    source_config = load_config(SOURCE_ROOT / "configs" / "base.yaml")
    data = source_config["data"]
    fold_path = resolve_path(data["fold_manifest"])
    if file_sha256(fold_path) != str(data.get("fold_manifest_sha256", "")):
        raise ValueError("canonical fold manifest SHA 漂移")
    fold_frame = pd.read_csv(
        fold_path,
        usecols=["patient_id", "fold", "split"],
        dtype={"patient_id": str},
    )
    if len(fold_frame) != 4040 or set(fold_frame["fold"]) != set(FOLDS):
        raise ValueError("canonical fold manifest coverage 漂移")
    if not set(fold_frame["split"]).issubset({"train", "val", "test"}):
        raise ValueError("canonical fold manifest split 非法")
    primary_ids = frozenset(fold_frame["patient_id"].astype(str))
    if len(primary_ids) != 808:
        raise ValueError("primary cohort 不再是 808 人")
    cache = cache_index(resolve_path(data["cache_root"]))
    raw_ftv = read_raw_ftv(resolve_path(data["ftv_targets"]))
    if len(cache) != 964 or len(raw_ftv) != 375:
        raise ValueError("cache/FTV cohort count 漂移")
    if not set(raw_ftv).issubset(primary_ids):
        raise ValueError("FTV mapping 含非 primary patient")
    if (
        raw_ftv_hash(raw_ftv)
        != "41b419f6e098dade710ee8963ccd6245ce3ea9bd32687afbef1223cafde9529b"
    ):
        raise ValueError("raw FTV semantic hash 漂移")
    return AuditDataContext(fold_frame, primary_ids, cache, raw_ftv, source_config)


def canonical_split_ids(context: AuditDataContext, fold: int) -> dict[str, list[str]]:
    current = context.fold_frame.loc[context.fold_frame["fold"].eq(fold)]
    result = {
        name: current.loc[current["split"].eq(name), "patient_id"].astype(str).tolist()
        for name in ("train", "val", "test")
    }
    if sum(map(len, result.values())) != 808:
        raise ValueError(f"fold {fold} split coverage 错误")
    return result


def audited_pool_ids(context: AuditDataContext, fold: int, split: str) -> list[str]:
    if split not in {"train", "validation"}:
        raise ValueError("audit split 非法")
    reference = _load_checkpoint(checkpoint_path(SEED_BASES[0], fold))
    splits = reference.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("checkpoint 缺 splits")
    canonical = canonical_split_ids(context, fold)
    if list(splits.get("train", [])) != canonical["train"]:
        raise ValueError(f"fold {fold} checkpoint train 顺序/成员漂移")
    if list(splits.get("val", [])) != canonical["val"]:
        raise ValueError(f"fold {fold} checkpoint val 顺序/成员漂移")
    if list(splits.get("test", [])) != canonical["test"]:
        raise ValueError(f"fold {fold} checkpoint test 顺序/成员漂移")
    ids = list(splits["pretrain_train"] if split == "train" else splits["val"])
    for seed in SEED_BASES:
        payload = _load_checkpoint(checkpoint_path(seed, fold))
        current = payload.get("splits")
        if not isinstance(current, Mapping):
            raise ValueError(f"checkpoint {seed}/{fold} 缺 splits")
        expected = list(
            current["pretrain_train"] if split == "train" else current["val"]
        )
        if expected != ids:
            raise ValueError(
                f"同 fold 跨 seed audit pool 不一致: {seed}/{fold}/{split}"
            )
    if len(ids) != len(set(ids)) or any(
        patient_id not in context.cache for patient_id in ids
    ):
        raise ValueError(f"audit pool 重复或 cache 缺失: fold={fold}, split={split}")
    return ids


def patient_records(
    context: AuditDataContext, patient_ids: Iterable[str]
) -> list[PatientRecord]:
    records: list[PatientRecord] = []
    for patient_id in map(str, patient_ids):
        records.append(
            PatientRecord(
                patient_id=patient_id,
                cache_path=context.cache[patient_id],
                source="ispy2" if patient_id in context.primary_ids else "ispy1",
                pcr=None,
                has_ftv=patient_id in context.raw_ftv,
            )
        )
    return records


def fold_transform(context: AuditDataContext, fold: int) -> PooledFTVTransform:
    path = SOURCE_ROOT / "configs" / f"ftv_transform_fold_{fold}.json"
    transform = PooledFTVTransform.load(path)
    train_ids = canonical_split_ids(context, fold)["train"]
    paired = [patient_id for patient_id in train_ids if patient_id in context.raw_ftv]
    valid_visit_count = int(
        sum(
            np.asarray(context.raw_ftv[patient_id][1], dtype=bool).sum()
            for patient_id in paired
        )
    )
    expected_file_sha = EXPECTED_SOURCE_SHA256[
        f"configs/ftv_transform_fold_{fold}.json"
    ]
    if (
        file_sha256(path) != expected_file_sha
        or transform.fold != fold
        or transform.train_patient_hash != patient_hash(train_ids)
        or transform.raw_targets_sha256 != raw_ftv_hash(context.raw_ftv)
        or transform.train_patient_count != len(train_ids)
        or transform.paired_train_patient_count != len(paired)
        or transform.valid_visit_count != valid_visit_count
    ):
        raise ValueError(f"fold {fold} FTV transform contract 漂移")
    return transform


def validate_checkpoint_grid() -> list[dict[str, Any]]:
    assert_audit_config()
    assert_source_hashes()
    input_manifest = pd.read_csv(
        SOURCE_ROOT / "metrics" / "final" / "input_manifest.csv"
    )
    registered_checkpoints = input_manifest.loc[
        input_manifest["kind"].eq("checkpoint")
    ].copy()
    stability = pd.read_csv(
        SOURCE_ROOT / "metrics" / "final" / "training_stability_seed_fold.csv"
    )
    registered_stability = stability.loc[stability["model"].eq("G3")].copy()
    if len(registered_checkpoints) != 50 or len(registered_stability) != 25:
        raise ValueError("既有 checkpoint registration grid 漂移")
    rows: list[dict[str, Any]] = []
    checkpoint_hashes: set[str] = set()
    for seed in SEED_BASES:
        for fold in FOLDS:
            g1_path = checkpoint_path(seed, fold, "g1")
            g3_path = checkpoint_path(seed, fold, "g3")
            history = history_path(seed, fold)
            selection = selection_path(seed, fold)
            for required in (g1_path, g3_path, history, selection):
                if not required.is_file():
                    raise FileNotFoundError(f"正式资产缺失: {required}")
            g1 = _load_checkpoint(g1_path)
            g3 = _load_checkpoint(g3_path)
            g1_finite, _ = _all_tensors_finite(g1)
            finite, tensor_count = _all_tensors_finite(g3)
            selected_history = pd.read_csv(history)
            selected_mask = selected_history["is_selected_checkpoint"].map(
                lambda value: _strict_bool(value, f"history {seed}/{fold}.selected")
            )
            selected_rows = selected_history.loc[selected_mask]
            expected_epoch = (
                int(selected_rows.iloc[0]["epoch"]) if len(selected_rows) == 1 else -1
            )
            contract = g3.get("architecture_contract", {})
            loss = g3.get("loss_config", {})
            g1_contract = g1.get("architecture_contract", {})
            g1_loss = g1.get("loss_config", {})
            g1_sha = file_sha256(g1_path)
            g1_registered = registered_checkpoints.loc[
                registered_checkpoints["seed_base"].eq(seed)
                & registered_checkpoints["fold"].eq(fold)
                & registered_checkpoints["model"].eq("G1")
            ]
            baseline = g3.get("baseline_selection_contract", {})
            if (
                g1.get("schema_version") != 2
                or g1.get("finalized") is not True
                or g1.get("model_name") != "G1"
                or int(g1.get("seed_base", -1)) != seed
                or int(g1.get("fold", -1)) != fold
                or int(g1.get("effective_seed", -1)) != seed + fold
                or float(g1_loss.get("lambda_ftv", math.nan)) != 0.0
                or float(g1_loss.get("sigreg", math.nan)) != 0.09
                or int(g1_loss.get("sigreg_projections", -1)) != 256
                or list(g1_loss.get("step_weights", [])) != [2.0, 1.0, 0.5]
                or g1_contract.get("model_name") != "G1"
                or g1_contract.get("ftv_head") is not None
                or not g1_finite
                or not _state_mirror_is_exact(g1)
                or len(g1_registered) != 1
                or str(g1_registered.iloc[0]["path"]) != repo_relative(g1_path)
                or str(g1_registered.iloc[0]["sha256"]) != g1_sha
                or int(g1_registered.iloc[0]["bytes"]) != g1_path.stat().st_size
                or baseline.get("paired_model") != "G1"
                or str(baseline.get("baseline_checkpoint_sha256")) != g1_sha
            ):
                raise ValueError(
                    f"paired G1 checkpoint contract 失败: seed={seed}, fold={fold}"
                )
            if (
                g3.get("schema_version") != 2
                or g3.get("finalized") is not True
                or g3.get("model_name") != "G3"
                or int(g3.get("seed_base", -1)) != seed
                or int(g3.get("fold", -1)) != fold
                or int(g3.get("effective_seed", -1)) != seed + fold
                or int(g3.get("epoch", -1)) != expected_epoch
                or float(loss.get("lambda_ftv", math.nan)) != LAMBDA_FTV
                or float(loss.get("sigreg", math.nan)) != 0.09
                or int(loss.get("sigreg_projections", -1)) != 256
                or list(loss.get("step_weights", [])) != [2.0, 1.0, 0.5]
                or contract.get("backbone_input") != "DCE7"
                or contract.get("pooling") != "gap"
                or contract.get("observed_response_state") != "online_preprojector_r"
                or contract.get("ftv_head") != "Linear(response_dim,1)"
                or g3.get("history_sha256") != file_sha256(history)
                or g3.get("selection_sha256") != file_sha256(selection)
                or not finite
                or not _state_mirror_is_exact(g3)
            ):
                raise ValueError(
                    f"G3 checkpoint contract 失败: seed={seed}, fold={fold}"
                )
            if g1.get("shared_initialization_sha256") != g3.get(
                "shared_initialization_sha256"
            ):
                raise ValueError(f"paired G1/G3 initialization 不一致: {seed}/{fold}")
            checkpoint_sha = file_sha256(g3_path)
            input_row = registered_checkpoints.loc[
                registered_checkpoints["seed_base"].eq(seed)
                & registered_checkpoints["fold"].eq(fold)
                & registered_checkpoints["model"].eq("G3")
            ]
            stability_row = registered_stability.loc[
                registered_stability["seed_base"].eq(seed)
                & registered_stability["fold"].eq(fold)
            ]
            if (
                len(input_row) != 1
                or len(stability_row) != 1
                or str(input_row.iloc[0]["path"]) != repo_relative(g3_path)
                or str(input_row.iloc[0]["sha256"]) != checkpoint_sha
                or int(input_row.iloc[0]["bytes"]) != g3_path.stat().st_size
                or str(stability_row.iloc[0]["checkpoint_sha256"]) != checkpoint_sha
            ):
                raise ValueError(
                    f"checkpoint 与既有 manifest/hash 不闭环: {seed}/{fold}"
                )
            checkpoint_hashes.add(checkpoint_sha)
            rows.append(
                {
                    "schema_version": 1,
                    "seed_base": seed,
                    "fold": fold,
                    "effective_seed": seed + fold,
                    "checkpoint_kind": "selected",
                    "checkpoint": repo_relative(g3_path),
                    "checkpoint_sha256": checkpoint_sha,
                    "checkpoint_bytes": g3_path.stat().st_size,
                    "checkpoint_epoch": expected_epoch,
                    "history": repo_relative(history),
                    "history_sha256": file_sha256(history),
                    "selection": repo_relative(selection),
                    "selection_sha256": file_sha256(selection),
                    "checkpoint_mirrored_tensor_entries_checked": tensor_count,
                    "shared_initialization_sha256": g3["shared_initialization_sha256"],
                    "state_distinct_from_selected": False,
                    "formal_inference_checkpoint": True,
                    "contains_patient_ids": False,
                }
            )
    if len(rows) != 25 or len(checkpoint_hashes) != 25:
        raise ValueError("G3 checkpoint grid 不完整或 SHA 重复")
    if any(tuple(row) != ASSET_MANIFEST_COLUMNS for row in rows):
        raise AssertionError("selected asset row schema/order 漂移")
    return rows


def validate_trajectory_assets(representatives: pd.DataFrame) -> list[dict[str, Any]]:
    if (
        len(representatives) != 6
        or representatives.duplicated(["seed_base", "fold"]).any()
        or set(representatives["base_gate"]) != {"PASS", "FAIL"}
    ):
        raise ValueError("representative trajectory grid 非法")
    rows: list[dict[str, Any]] = []
    hashes: set[str] = set()
    for representative in representatives.itertuples(index=False):
        seed = int(representative.seed_base)
        fold = int(representative.fold)
        selected_path = checkpoint_path(seed, fold, "g3", "best")
        last_path = checkpoint_path(seed, fold, "g3", "last")
        history = history_path(seed, fold)
        selection = selection_path(seed, fold)
        if not last_path.is_file():
            raise FileNotFoundError(f"representative last checkpoint 缺失: {last_path}")
        selected = _load_checkpoint(selected_path)
        last = _load_checkpoint(last_path)
        finite, tensor_count = _all_tensors_finite(last)
        history_frame = pd.read_csv(history)
        expected_last_epoch = int(history_frame.iloc[-1]["epoch"])
        contract = last.get("architecture_contract", {})
        loss = last.get("loss_config", {})
        if (
            last.get("schema_version") != 2
            or last.get("finalized") is not True
            or last.get("model_name") != "G3"
            or int(last.get("seed_base", -1)) != seed
            or int(last.get("fold", -1)) != fold
            or int(last.get("effective_seed", -1)) != seed + fold
            or int(last.get("epoch", -1)) != expected_last_epoch
            or expected_last_epoch <= int(selected.get("epoch", -1))
            or float(loss.get("lambda_ftv", math.nan)) != LAMBDA_FTV
            or float(loss.get("sigreg", math.nan)) != 0.09
            or int(loss.get("sigreg_projections", -1)) != 256
            or list(loss.get("step_weights", [])) != [2.0, 1.0, 0.5]
            or contract.get("model_name") != "G3"
            or contract.get("backbone_input") != "DCE7"
            or contract.get("ftv_head") != "Linear(response_dim,1)"
            or last.get("history_sha256") != file_sha256(history)
            or last.get("selection_sha256") != file_sha256(selection)
            or last.get("shared_initialization_sha256")
            != selected.get("shared_initialization_sha256")
            or not finite
            or not _state_mirror_is_exact(last)
        ):
            raise ValueError(
                f"representative last checkpoint contract 失败: {seed}/{fold}"
            )
        selected_state = checkpoint_state_fingerprint(selected)
        last_state = checkpoint_state_fingerprint(last)
        if selected_state == last_state:
            raise ValueError(
                f"representative last 与 selected state 相同: {seed}/{fold}"
            )
        checkpoint_sha = file_sha256(last_path)
        hashes.add(checkpoint_sha)
        rows.append(
            {
                "schema_version": 1,
                "seed_base": seed,
                "fold": fold,
                "effective_seed": seed + fold,
                "checkpoint_kind": "last",
                "checkpoint": repo_relative(last_path),
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_bytes": last_path.stat().st_size,
                "checkpoint_epoch": expected_last_epoch,
                "history": repo_relative(history),
                "history_sha256": file_sha256(history),
                "selection": repo_relative(selection),
                "selection_sha256": file_sha256(selection),
                "checkpoint_mirrored_tensor_entries_checked": tensor_count,
                "shared_initialization_sha256": last["shared_initialization_sha256"],
                "state_distinct_from_selected": True,
                "formal_inference_checkpoint": False,
                "contains_patient_ids": False,
            }
        )
    if len(rows) != 6 or len(hashes) != 6:
        raise ValueError("representative last checkpoint SHA coverage 错误")
    if any(tuple(row) != ASSET_MANIFEST_COLUMNS for row in rows):
        raise AssertionError("trajectory asset row schema/order 漂移")
    return rows


def checkpoint_state_fingerprint(payload: Mapping[str, Any]) -> str:
    state = payload.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint 缺 state_dict")
    digest = __import__("hashlib").sha256()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("state_dict 含非 tensor")
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


__all__ = [
    "AuditDataContext",
    "ASSET_MANIFEST_COLUMNS",
    "audited_pool_ids",
    "canonical_split_ids",
    "checkpoint_path",
    "checkpoint_state_fingerprint",
    "fold_transform",
    "history_path",
    "load_data_context",
    "patient_records",
    "selection_path",
    "validate_checkpoint_grid",
    "validate_trajectory_assets",
]

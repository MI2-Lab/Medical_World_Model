"""固定、分层、患者标识不公开的 audit batch manifest。"""

from __future__ import annotations

import csv
import hashlib
import hmac
import io
import math
import os
import secrets
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .assets import AuditDataContext, audited_pool_ids, canonical_split_ids
from .contracts import (
    AUDIT_ROOT,
    AUDIT_SEED,
    BATCHES_PER_SPLIT,
    BATCH_SIZE,
    FOLDS,
    MINIMUM_FTV_PATIENTS,
    SPLITS,
    atomic_csv,
    EXPECTED_AUDIT_CONFIG_SHA256,
    ensure_no_patient_columns,
    file_sha256,
)
from .freeze import PLAN_FREEZE, assert_plan_freeze


PUBLIC_MANIFEST = AUDIT_ROOT / "configs" / "audit_batch_manifest.csv"
PRIVATE_MEMBERSHIP = AUDIT_ROOT / "configs" / "private" / "audit_batch_membership.csv"
PRIVATE_HMAC_KEY = AUDIT_ROOT / "configs" / "private" / "audit_batch_hmac.key"


def _seed(fold: int, split: str, stratum: str, retry: int = 0) -> int:
    split_offset = 0 if split == "train" else 100_000
    stratum_offset = {
        "ispy2_ftv": 1_000,
        "ispy2_noftv": 2_000,
        "ispy1_noftv": 3_000,
        "batch_order": 4_000,
    }[stratum]
    return AUDIT_SEED + fold * 1_000_000 + split_offset + stratum_offset + retry


def _stratum(context: AuditDataContext, patient_id: str) -> str:
    is_primary = patient_id in context.primary_ids
    has_ftv = patient_id in context.raw_ftv
    if not is_primary and has_ftv:
        raise ValueError("I-SPY1 不应有 FTV target")
    if is_primary and has_ftv:
        return "ispy2_ftv"
    return "ispy2_noftv" if is_primary else "ispy1_noftv"


def _largest_remainder(counts: Mapping[str, int], total: int) -> dict[str, int]:
    if sum(counts.values()) <= 0 or total <= 0:
        raise ValueError("largest-remainder 输入非法")
    exact = {
        name: total * count / sum(counts.values()) for name, count in counts.items()
    }
    result = {name: int(np.floor(value)) for name, value in exact.items()}
    remaining = total - sum(result.values())
    order = sorted(counts, key=lambda name: (-(exact[name] - result[name]), name))
    for name in order[:remaining]:
        result[name] += 1
    if result.get("ispy2_ftv", 0) < MINIMUM_FTV_PATIENTS:
        deficit = MINIMUM_FTV_PATIENTS - result.get("ispy2_ftv", 0)
        donors = sorted(
            (name for name in result if name != "ispy2_ftv"),
            key=lambda name: (-result[name], name),
        )
        for name in donors:
            take = min(deficit, result[name])
            result[name] -= take
            result["ispy2_ftv"] = result.get("ispy2_ftv", 0) + take
            deficit -= take
            if not deficit:
                break
        if deficit:
            raise ValueError("无法满足 minimum FTV patient contract")
    if sum(result.values()) != total:
        raise AssertionError("largest-remainder 总数错误")
    return result


def _balanced_sequence(
    pool: Sequence[str], draws_per_batch: int, fold: int, split: str, stratum: str
) -> list[str]:
    total = draws_per_batch * BATCHES_PER_SPLIT
    if draws_per_batch <= 0:
        return []
    if len(pool) < draws_per_batch:
        raise ValueError(f"{fold}/{split}/{stratum} pool 小于单 batch 配额")
    ordered_pool = sorted(map(str, pool))
    for retry in range(1000):
        rng = np.random.default_rng(_seed(fold, split, stratum, retry))
        sequence: list[str] = []
        while len(sequence) < total:
            cycle = list(ordered_pool)
            rng.shuffle(cycle)
            sequence.extend(cycle)
        sequence = sequence[:total]
        batches = [
            sequence[index : index + draws_per_batch]
            for index in range(0, total, draws_per_batch)
        ]
        if all(len(batch) == len(set(batch)) for batch in batches):
            exposure = Counter(sequence)
            if max(exposure.values()) - min(exposure.values()) <= 1:
                return sequence
    raise RuntimeError(
        f"无法生成 batch 内无重复的 balanced cycle: {fold}/{split}/{stratum}"
    )


def build_membership(context: AuditDataContext) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        for split in SPLITS:
            pool = audited_pool_ids(context, fold, split)
            by_stratum: dict[str, list[str]] = {
                name: [] for name in ("ispy2_ftv", "ispy2_noftv", "ispy1_noftv")
            }
            for patient_id in pool:
                by_stratum[_stratum(context, patient_id)].append(patient_id)
            counts = {
                name: len(values) for name, values in by_stratum.items() if values
            }
            quotas = _largest_remainder(counts, BATCH_SIZE)
            sequences = {
                name: _balanced_sequence(values, quotas.get(name, 0), fold, split, name)
                for name, values in by_stratum.items()
                if values
            }
            for batch_index in range(BATCHES_PER_SPLIT):
                members: list[tuple[str, str]] = []
                for name, sequence in sequences.items():
                    quota = quotas[name]
                    start = batch_index * quota
                    members.extend(
                        (patient_id, name)
                        for patient_id in sequence[start : start + quota]
                    )
                rng = np.random.default_rng(
                    _seed(fold, split, "batch_order", batch_index)
                )
                rng.shuffle(members)
                if (
                    len(members) != BATCH_SIZE
                    or len({item[0] for item in members}) != BATCH_SIZE
                ):
                    raise AssertionError("audit batch size/uniqueness 破坏")
                if (
                    sum(name == "ispy2_ftv" for _, name in members)
                    < MINIMUM_FTV_PATIENTS
                ):
                    raise AssertionError("audit batch FTV minimum 破坏")
                batch_id = (
                    f"f{fold}_{'tr' if split == 'train' else 'va'}_{batch_index:02d}"
                )
                for position, (patient_id, name) in enumerate(members):
                    rows.append(
                        {
                            "batch_id": batch_id,
                            "fold": fold,
                            "split": split,
                            "batch_index": batch_index,
                            "position": position,
                            "patient_id": patient_id,
                            "cohort_role": name,
                            "has_ftv": name == "ispy2_ftv",
                        }
                    )
    if len(rows) != len(FOLDS) * len(SPLITS) * BATCHES_PER_SPLIT * BATCH_SIZE:
        raise AssertionError("private membership row count 错误")
    return rows


def _membership_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    fields = [
        "batch_id",
        "fold",
        "split",
        "batch_index",
        "position",
        "patient_id",
        "cohort_role",
        "has_ftv",
    ]
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _load_or_create_key() -> bytes:
    PRIVATE_HMAC_KEY.parent.mkdir(parents=True, exist_ok=True)
    if PRIVATE_HMAC_KEY.exists():
        if PRIVATE_HMAC_KEY.stat().st_mode & 0o077:
            raise PermissionError("private HMAC key 权限不是0600")
        key = PRIVATE_HMAC_KEY.read_bytes()
        if len(key) != 32:
            raise ValueError("private HMAC key 长度非法")
        return key
    key = secrets.token_bytes(32)
    descriptor = os.open(PRIVATE_HMAC_KEY, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(key)
    return key


def _ordered_hmac(key: bytes, batch: pd.DataFrame) -> str:
    ordered = batch.sort_values("position")
    first = ordered.iloc[0]
    context = (
        f"{first.batch_id}|{int(first.fold)}|{first.split}|{int(first.batch_index)}"
    )
    members = "\n".join(
        f"{row.position}|{row.patient_id}|{row.cohort_role}|{int(bool(row.has_ftv))}"
        for row in ordered.itertuples(index=False)
    )
    message = f"{context}\n{members}".encode("utf-8")
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def _strict_bool_series(series: pd.Series, name: str) -> pd.Series:
    def parse(value: Any) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
            return bool(value)
        if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            return value.strip().lower() == "true"
        raise ValueError(f"{name} 含非严格 boolean: {value!r}")

    return series.map(parse)


def write_manifests(
    context: AuditDataContext, *, overwrite: bool = False
) -> tuple[Path, Path]:
    assert_plan_freeze()
    rows = build_membership(context)
    membership_payload = _membership_bytes(rows)
    if PRIVATE_MEMBERSHIP.exists() and not overwrite:
        if PRIVATE_MEMBERSHIP.read_bytes() != membership_payload:
            raise FileExistsError("private membership 已存在且内容不一致")
    else:
        PRIVATE_MEMBERSHIP.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{PRIVATE_MEMBERSHIP.name}.",
            suffix=".tmp",
            dir=PRIVATE_MEMBERSHIP.parent,
        )
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(membership_payload)
            os.replace(temporary, PRIVATE_MEMBERSHIP)
        finally:
            temporary.unlink(missing_ok=True)
    key = _load_or_create_key()
    key_id = hashlib.sha256(key).hexdigest()[:16]
    private_hmac = hmac.new(
        key, PRIVATE_MEMBERSHIP.read_bytes(), hashlib.sha256
    ).hexdigest()
    frame = pd.DataFrame(rows)
    public_rows: list[dict[str, Any]] = []
    for (fold, split, batch_index, batch_id), batch in frame.groupby(
        ["fold", "split", "batch_index", "batch_id"], sort=True
    ):
        pool = audited_pool_ids(context, int(fold), str(split))
        grounded = int(batch["has_ftv"].astype(bool).sum())
        public_rows.append(
            {
                "schema_version": 1,
                "plan_freeze_sha256": file_sha256(PLAN_FREEZE),
                "audit_config_sha256": EXPECTED_AUDIT_CONFIG_SHA256,
                "source_fold_manifest_sha256": str(
                    context.source_config["data"]["fold_manifest_sha256"]
                ),
                "source_raw_ftv_semantic_sha256": "41b419f6e098dade710ee8963ccd6245ce3ea9bd32687afbef1223cafde9529b",
                "batch_id": batch_id,
                "fold": int(fold),
                "split": str(split),
                "batch_index": int(batch_index),
                "ordered_members_hmac_sha256": _ordered_hmac(key, batch),
                "hmac_key_id": key_id,
                "n_total": len(batch),
                "n_ftv_available": grounded,
                "n_unavailable": len(batch) - grounded,
                "n_ispy2": int(batch["cohort_role"].ne("ispy1_noftv").sum()),
                "n_ispy1": int(batch["cohort_role"].eq("ispy1_noftv").sum()),
                "pool_n": len(pool),
                "pool_ftv_available": sum(
                    patient_id in context.raw_ftv for patient_id in pool
                ),
                "pool_ftv_proportion": sum(
                    patient_id in context.raw_ftv for patient_id in pool
                )
                / len(pool),
                "batch_ftv_proportion": grounded / len(batch),
                "applies_to_seed_count": 5,
                "within_batch_replacement": False,
                "contains_patient_ids": False,
                "contains_patient_level_rows": False,
                "private_mapping_hmac_sha256": private_hmac,
            }
        )
    atomic_csv(PUBLIC_MANIFEST, public_rows, overwrite=overwrite)
    validate_manifests(context)
    return PUBLIC_MANIFEST, PRIVATE_MEMBERSHIP


def _context_pool_ids(context: AuditDataContext, fold: int, split: str) -> list[str]:
    canonical = canonical_split_ids(context, fold)
    if split == "validation":
        return canonical["val"]
    if split == "train":
        extra = sorted(set(context.cache) - set(context.primary_ids))
        return canonical["train"] + extra
    raise ValueError("audit split 非法")


def validate_manifests(
    context: AuditDataContext, *, verify_checkpoint_pools: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    public = pd.read_csv(PUBLIC_MANIFEST)
    private = pd.read_csv(PRIVATE_MEMBERSHIP, dtype={"patient_id": str})
    key = PRIVATE_HMAC_KEY.read_bytes()
    if len(key) != 32:
        raise ValueError("private HMAC key 长度非法")
    if (
        PRIVATE_HMAC_KEY.stat().st_mode & 0o077
        or PRIVATE_MEMBERSHIP.stat().st_mode & 0o077
    ):
        raise PermissionError("private batch manifest/key 权限不是0600")
    if len(public) != 80 or len(private) != 2560:
        raise ValueError("audit manifest row count 错误")
    expected_public_keys = {
        (fold, split, batch_index)
        for fold in FOLDS
        for split in SPLITS
        for batch_index in range(BATCHES_PER_SPLIT)
    }
    observed_public_keys = set(
        public[["fold", "split", "batch_index"]].itertuples(index=False, name=None)
    )
    if observed_public_keys != expected_public_keys:
        raise ValueError("public manifest fold/split/batch key coverage 错误")
    if (
        public["batch_id"].duplicated().any()
        or private.duplicated(["batch_id", "position"]).any()
    ):
        raise ValueError("audit manifest batch/position 重复")
    ensure_no_patient_columns(public.columns)
    if (
        _strict_bool_series(
            public["contains_patient_ids"], "contains_patient_ids"
        ).any()
        or _strict_bool_series(
            public["contains_patient_level_rows"], "contains_patient_level_rows"
        ).any()
    ):
        raise ValueError("public manifest privacy flag 非法")
    if (
        "patient_id" in public.columns
        or "ordered_members_hmac_sha256" not in public.columns
    ):
        raise ValueError("public manifest patient schema 非法")
    if set(public["schema_version"]) != {1} or set(public["applies_to_seed_count"]) != {
        5
    }:
        raise ValueError("public manifest schema/seed application contract 错误")
    if (
        set(public["plan_freeze_sha256"]) != {file_sha256(PLAN_FREEZE)}
        or set(public["audit_config_sha256"]) != {EXPECTED_AUDIT_CONFIG_SHA256}
        or set(public["source_fold_manifest_sha256"])
        != {str(context.source_config["data"]["fold_manifest_sha256"])}
        or set(public["source_raw_ftv_semantic_sha256"])
        != {"41b419f6e098dade710ee8963ccd6245ce3ea9bd32687afbef1223cafde9529b"}
    ):
        raise ValueError("public manifest plan/source freeze 字段漂移")
    if _strict_bool_series(
        public["within_batch_replacement"], "within_batch_replacement"
    ).any():
        raise ValueError("public manifest 意外允许 batch 内 replacement")
    expected_private_hmac = hmac.new(
        key, PRIVATE_MEMBERSHIP.read_bytes(), hashlib.sha256
    ).hexdigest()
    if set(public["private_mapping_hmac_sha256"]) != {expected_private_hmac}:
        raise ValueError("public/private manifest keyed HMAC 不闭环")
    if set(public["hmac_key_id"]) != {hashlib.sha256(key).hexdigest()[:16]}:
        raise ValueError("public manifest HMAC key id 不闭环")
    pool_cache: dict[tuple[int, str], list[str]] = {}
    for fold in FOLDS:
        for split in SPLITS:
            contextual = _context_pool_ids(context, fold, split)
            if verify_checkpoint_pools:
                checkpoint_pool = audited_pool_ids(context, fold, split)
                if set(contextual) != set(checkpoint_pool) or len(contextual) != len(
                    checkpoint_pool
                ):
                    raise ValueError(f"context/checkpoint pool 不一致: {fold}/{split}")
                pool_cache[(fold, split)] = checkpoint_pool
            else:
                pool_cache[(fold, split)] = contextual
    for row in public.itertuples(index=False):
        batch = private.loc[private["batch_id"].eq(row.batch_id)].copy()
        if len(batch) != BATCH_SIZE or batch["patient_id"].duplicated().any():
            raise ValueError(f"private batch size/duplicate 错误: {row.batch_id}")
        expected_batch_id = f"f{int(row.fold)}_{'tr' if row.split == 'train' else 'va'}_{int(row.batch_index):02d}"
        if str(row.batch_id) != expected_batch_id:
            raise ValueError(f"batch id 与 key 不一致: {row.batch_id}")
        if (
            set(batch["fold"]) != {int(row.fold)}
            or set(batch["split"]) != {str(row.split)}
            or set(batch["batch_index"]) != {int(row.batch_index)}
        ):
            raise ValueError(f"private/public batch metadata 不一致: {row.batch_id}")
        if set(batch["position"].astype(int)) != set(range(BATCH_SIZE)):
            raise ValueError(f"private batch position coverage 错误: {row.batch_id}")
        if _ordered_hmac(key, batch) != row.ordered_members_hmac_sha256:
            raise ValueError(f"batch HMAC 不一致: {row.batch_id}")
        has_ftv = _strict_bool_series(batch["has_ftv"], f"{row.batch_id}.has_ftv")
        expected_roles = batch["patient_id"].map(
            lambda patient_id: _stratum(context, str(patient_id))
        )
        if not expected_roles.equals(batch["cohort_role"].astype(str)):
            raise ValueError(f"private batch cohort role 漂移: {row.batch_id}")
        if not has_ftv.equals(expected_roles.eq("ispy2_ftv")):
            raise ValueError(f"private batch FTV flag 漂移: {row.batch_id}")
        grounded = int(has_ftv.sum())
        if grounded != int(row.n_ftv_available) or grounded < MINIMUM_FTV_PATIENTS:
            raise ValueError(f"batch FTV count 不一致: {row.batch_id}")
        pool_list = pool_cache[(int(row.fold), str(row.split))]
        pool = set(pool_list)
        if not set(batch["patient_id"]).issubset(pool):
            raise ValueError(f"batch 含 pool 外 patient: {row.batch_id}")
        pool_ftv = sum(patient_id in context.raw_ftv for patient_id in pool_list)
        expected_counts = {
            "n_total": BATCH_SIZE,
            "n_ftv_available": grounded,
            "n_unavailable": BATCH_SIZE - grounded,
            "n_ispy2": int(expected_roles.ne("ispy1_noftv").sum()),
            "n_ispy1": int(expected_roles.eq("ispy1_noftv").sum()),
            "pool_n": len(pool_list),
            "pool_ftv_available": pool_ftv,
        }
        for field, expected in expected_counts.items():
            if int(getattr(row, field)) != expected:
                raise ValueError(f"public count {field} 不可重算: {row.batch_id}")
        if not math.isclose(
            float(row.pool_ftv_proportion), pool_ftv / len(pool_list), abs_tol=1e-12
        ):
            raise ValueError(f"pool FTV proportion 不一致: {row.batch_id}")
        if not math.isclose(
            float(row.batch_ftv_proportion), grounded / BATCH_SIZE, abs_tol=1e-12
        ):
            raise ValueError(f"batch FTV proportion 不一致: {row.batch_id}")

    private_has_ftv = _strict_bool_series(private["has_ftv"], "private.has_ftv")
    private = private.assign(has_ftv=private_has_ftv)
    for fold in FOLDS:
        for split in SPLITS:
            pool = pool_cache[(fold, split)]
            pool_by_stratum: dict[str, list[str]] = {
                name: [] for name in ("ispy2_ftv", "ispy2_noftv", "ispy1_noftv")
            }
            for patient_id in pool:
                pool_by_stratum[_stratum(context, patient_id)].append(patient_id)
            counts = {
                name: len(values) for name, values in pool_by_stratum.items() if values
            }
            quotas = _largest_remainder(counts, BATCH_SIZE)
            selected = private.loc[
                private["fold"].eq(fold) & private["split"].eq(split)
            ]
            for name, members in pool_by_stratum.items():
                if not members:
                    continue
                per_batch = (
                    selected.loc[selected["cohort_role"].eq(name)]
                    .groupby("batch_index")
                    .size()
                )
                if set(per_batch.index) != set(range(BATCHES_PER_SPLIT)) or set(
                    per_batch
                ) != {quotas[name]}:
                    raise ValueError(f"stratum quota 漂移: {fold}/{split}/{name}")
                exposure = selected.loc[
                    selected["cohort_role"].eq(name), "patient_id"
                ].value_counts()
                full_exposure = [
                    int(exposure.get(patient_id, 0)) for patient_id in members
                ]
                if max(full_exposure) - min(full_exposure) > 1:
                    raise ValueError(
                        f"stratum exposure 非 balanced-cycle: {fold}/{split}/{name}"
                    )
    return public, private


def synthetic_self_test() -> dict[str, Any]:
    counts = {"ispy2_ftv": 35, "ispy2_noftv": 50, "ispy1_noftv": 15}
    quotas = _largest_remainder(counts, 32)
    sequence = _balanced_sequence(
        [f"P{i:03d}" for i in range(11)], 5, 0, "train", "ispy2_ftv"
    )
    batches = [sequence[index : index + 5] for index in range(0, len(sequence), 5)]
    checks = {
        "quota_total_32": sum(quotas.values()) == 32,
        "quota_ftv_at_least_8": quotas["ispy2_ftv"] >= 8,
        "balanced_length": len(sequence) == 5 * BATCHES_PER_SPLIT,
        "within_batch_unique": all(len(batch) == len(set(batch)) for batch in batches),
        "public_manifest_has_no_patient_id": "patient_id"
        not in {
            "schema_version",
            "batch_id",
            "ordered_members_hmac_sha256",
        },
    }
    if not all(checks.values()):
        raise AssertionError(f"batch self-test 失败: {checks}")
    return {"status": "ok", "checks": checks, "example_quotas": quotas}


__all__ = [
    "PRIVATE_HMAC_KEY",
    "PRIVATE_MEMBERSHIP",
    "PUBLIC_MANIFEST",
    "build_membership",
    "synthetic_self_test",
    "validate_manifests",
    "write_manifests",
]

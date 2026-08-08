"""绑定正式 checkpoint、fixed batch 与实际 NPZ 输入的 source contract。"""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .assets import (
    ASSET_MANIFEST_COLUMNS,
    AuditDataContext,
    checkpoint_path,
    history_path,
    load_data_context,
    selection_path,
)
from .batches import (
    PRIVATE_HMAC_KEY,
    validate_manifests,
)
from .contracts import (
    AUDIT_ROOT,
    EXPECTED_SOURCE_SHA256,
    FOLDS,
    REPO_ROOT,
    SEED_BASES,
    SOURCE_ROOT,
    assert_source_hashes,
    atomic_csv,
    atomic_json,
    canonical_json_sha256,
    ensure_no_patient_columns,
    file_sha256,
    repo_relative,
)
from .freeze import PLAN_FREEZE, SOURCE_CONTRACT, assert_plan_freeze


PRIVATE_CACHE_MANIFEST = AUDIT_ROOT / "configs" / "private" / "cache_input_manifest.csv"
PUBLIC_CACHE_CONTRACT = AUDIT_ROOT / "metrics" / "cache_input_contract.json"
SOURCE_MANIFEST = AUDIT_ROOT / "metrics" / "source_manifest.csv"
SOURCE_MANIFEST_COLUMNS = (
    "artifact_id",
    "artifact_kind",
    "path",
    "sha256",
    "bytes",
    "seed_base",
    "fold",
    "model",
    "checkpoint_kind",
    "referenced_artifact_may_embed_patient_ids",
    "manifest_contains_patient_ids",
)

SOURCE_MANIFEST_ROWS = 126

SOURCE_CONTRACT_ARTIFACTS = (
    "PLAN_FREEZE.json",
    "configs/audit_batch_manifest.csv",
    "metrics/asset_manifest.csv",
    "metrics/cache_input_contract.json",
    "metrics/source_manifest.csv",
    "metrics/run_level_existing_metrics.csv",
    "metrics/training_history_audit.csv",
    "metrics/representative_runs.csv",
    "metrics/resampling_indices.npz",
    "metrics/resampling_manifest.json",
    "metrics/phase_a_gain_correlations.csv",
)

SOURCE_CONTRACT_KEYS = frozenset(
    {
        "schema_version",
        "created_at_utc",
        "plan_payload_sha256",
        "plan_file_sha256",
        "public_artifact_count",
        "public_artifact_sha256",
        "upstream_source_sha256",
        "source_manifest_rows",
        "checkpoint_rows",
        "selected_checkpoint_rows",
        "representative_last_checkpoint_rows",
        "public_batch_rows",
        "private_batch_rows",
        "cache_files",
        "cache_total_bytes",
        "cache_content_multiset_sha256",
        "private_commitment_kind",
        "contract_complete",
        "contains_patient_ids",
        "payload_sha256",
    }
)

PUBLIC_CACHE_CONTRACT_KEYS = frozenset(
    {
        "schema_version",
        "cache_role",
        "cache_files",
        "cache_total_bytes",
        "content_multiset_sha256",
        "patient_to_content_mapping_hmac_sha256",
        "hmac_key_id",
        "private_manifest_permissions",
        "contains_patient_ids",
        "contains_patient_level_rows",
    }
)


def _key() -> bytes:
    if not PRIVATE_HMAC_KEY.is_file():
        raise FileNotFoundError("private HMAC key 尚未生成")
    key = PRIVATE_HMAC_KEY.read_bytes()
    if len(key) != 32:
        raise ValueError("private HMAC key 长度非法")
    return key


def _file_hmac(key: bytes, path: Path) -> str:
    digest = hmac.new(key, digestmod=hashlib.sha256)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_bool(value: Any, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ValueError(f"{name} 非严格 boolean: {value!r}")


def _strict_nonnegative_int(value: Any, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
    ):
        raise ValueError(f"{name} 必须是非负整数")
    return int(value)


def _sha256_string(value: Any, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{name} 不是 canonical SHA-256")
    return text


def write_cache_input_contract(context: AuditDataContext) -> Path:
    if PRIVATE_CACHE_MANIFEST.exists() or PUBLIC_CACHE_CONTRACT.exists():
        raise FileExistsError("拒绝覆盖已有 cache input contract")
    rows: list[dict[str, Any]] = []
    for patient_id in sorted(context.cache):
        path = context.cache[patient_id]
        stat = path.stat()
        rows.append(
            {
                "patient_id": str(patient_id),
                "cache_bytes": int(stat.st_size),
                "cache_mtime_ns_at_freeze": int(stat.st_mtime_ns),
                "cache_sha256": file_sha256(path),
            }
        )
    atomic_csv(PRIVATE_CACHE_MANIFEST, rows)
    PRIVATE_CACHE_MANIFEST.chmod(0o600)
    key = _key()
    public = {
        "schema_version": 1,
        "cache_role": "DCE8 cache; audit loader uses DCE7 only",
        "cache_files": len(rows),
        "cache_total_bytes": int(sum(row["cache_bytes"] for row in rows)),
        "content_multiset_sha256": canonical_json_sha256(
            sorted((row["cache_sha256"], row["cache_bytes"]) for row in rows)
        ),
        "patient_to_content_mapping_hmac_sha256": _file_hmac(
            key, PRIVATE_CACHE_MANIFEST
        ),
        "hmac_key_id": hashlib.sha256(key).hexdigest()[:16],
        "private_manifest_permissions": "0600",
        "contains_patient_ids": False,
        "contains_patient_level_rows": False,
    }
    atomic_json(PUBLIC_CACHE_CONTRACT, public)
    assert_cache_input_contract(context, full_content_hash=False)
    return PUBLIC_CACHE_CONTRACT


def write_source_manifest(representatives: pd.DataFrame) -> Path:
    if SOURCE_MANIFEST.exists():
        raise FileExistsError("拒绝覆盖已有 source manifest")
    if (
        len(representatives) != 6
        or representatives.duplicated(["seed_base", "fold"]).any()
    ):
        raise ValueError("source manifest representative grid 非法")
    rows: list[dict[str, Any]] = []

    def add(
        artifact_id: str,
        artifact_kind: str,
        path: Path,
        *,
        seed_base: int | str = "",
        fold: int | str = "",
        model: str = "",
        checkpoint_kind: str = "",
        may_embed_ids: bool = False,
    ) -> None:
        rows.append(
            {
                "artifact_id": artifact_id,
                "artifact_kind": artifact_kind,
                "path": repo_relative(path),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
                "seed_base": seed_base,
                "fold": fold,
                "model": model,
                "checkpoint_kind": checkpoint_kind,
                "referenced_artifact_may_embed_patient_ids": may_embed_ids,
                "manifest_contains_patient_ids": False,
            }
        )

    for relative in sorted(EXPECTED_SOURCE_SHA256):
        add(f"upstream::{relative}", "frozen_upstream_file", SOURCE_ROOT / relative)
    for seed in SEED_BASES:
        for fold in FOLDS:
            add(
                f"g1_selected::{seed}::{fold}",
                "paired_g1_selected_checkpoint",
                checkpoint_path(seed, fold, "g1", "best"),
                seed_base=seed,
                fold=fold,
                model="G1",
                checkpoint_kind="selected",
                may_embed_ids=True,
            )
            add(
                f"g3_selected::{seed}::{fold}",
                "formal_g3_selected_checkpoint",
                checkpoint_path(seed, fold, "g3", "best"),
                seed_base=seed,
                fold=fold,
                model="G3",
                checkpoint_kind="selected",
                may_embed_ids=True,
            )
            add(
                f"g3_history::{seed}::{fold}",
                "formal_g3_history",
                history_path(seed, fold, "g3"),
                seed_base=seed,
                fold=fold,
                model="G3",
            )
            add(
                f"g3_selection::{seed}::{fold}",
                "formal_g3_selection",
                selection_path(seed, fold, "g3"),
                seed_base=seed,
                fold=fold,
                model="G3",
            )
    for row in representatives.itertuples(index=False):
        seed = int(row.seed_base)
        fold = int(row.fold)
        add(
            f"g3_last::{seed}::{fold}",
            "representative_g3_last_checkpoint",
            checkpoint_path(seed, fold, "g3", "last"),
            seed_base=seed,
            fold=fold,
            model="G3",
            checkpoint_kind="last",
            may_embed_ids=True,
        )
    if (
        len(rows) != 126
        or len({row["artifact_id"] for row in rows}) != 126
        or any(tuple(row) != SOURCE_MANIFEST_COLUMNS for row in rows)
    ):
        raise ValueError("source manifest 126-row/schema contract 失败")
    atomic_csv(SOURCE_MANIFEST, rows)
    return SOURCE_MANIFEST


def assert_cache_input_contract(
    context: AuditDataContext, *, full_content_hash: bool
) -> dict[str, Any]:
    if not PRIVATE_CACHE_MANIFEST.is_file() or not PUBLIC_CACHE_CONTRACT.is_file():
        raise FileNotFoundError("cache input contract 不完整")
    if PRIVATE_CACHE_MANIFEST.stat().st_mode & 0o077:
        raise PermissionError("private cache manifest 权限不是0600")
    private = pd.read_csv(PRIVATE_CACHE_MANIFEST, dtype={"patient_id": str})
    expected_columns = (
        "patient_id",
        "cache_bytes",
        "cache_mtime_ns_at_freeze",
        "cache_sha256",
    )
    if (
        tuple(private.columns) != expected_columns
        or len(private) != 964
        or private["patient_id"].duplicated().any()
        or set(private["patient_id"]) != set(context.cache)
    ):
        raise ValueError("private cache input manifest grid/schema 错误")
    public = json.loads(PUBLIC_CACHE_CONTRACT.read_text(encoding="utf-8"))
    key = _key()
    public_cache_files = (
        _strict_nonnegative_int(public.get("cache_files"), "cache_files")
        if isinstance(public, dict)
        else -1
    )
    public_cache_bytes = (
        _strict_nonnegative_int(public.get("cache_total_bytes"), "cache_total_bytes")
        if isinstance(public, dict)
        else -1
    )
    if (
        not isinstance(public, dict)
        or set(public) != PUBLIC_CACHE_CONTRACT_KEYS
        or int(public.get("schema_version", -1)) != 1
        or str(public.get("cache_role")) != "DCE8 cache; audit loader uses DCE7 only"
        or public_cache_files != len(private)
        or public_cache_bytes < 0
        or str(public.get("hmac_key_id")) != hashlib.sha256(key).hexdigest()[:16]
        or str(public.get("patient_to_content_mapping_hmac_sha256"))
        != _file_hmac(key, PRIVATE_CACHE_MANIFEST)
        or str(public.get("private_manifest_permissions")) != "0600"
        or public.get("contains_patient_ids") is not False
        or public.get("contains_patient_level_rows") is not False
    ):
        raise ValueError("public/private cache input HMAC/schema 不闭环")
    _sha256_string(public["content_multiset_sha256"], "cache content multiset SHA")
    _sha256_string(
        public["patient_to_content_mapping_hmac_sha256"],
        "cache patient/content mapping HMAC",
    )
    records = private.set_index("patient_id")
    observed_pairs: list[tuple[str, int]] = []
    total_bytes = 0
    for patient_id, path in context.cache.items():
        row = records.loc[str(patient_id)]
        stat = path.stat()
        if (
            int(row["cache_bytes"]) != stat.st_size
            or int(row["cache_mtime_ns_at_freeze"]) != stat.st_mtime_ns
        ):
            raise ValueError(f"cache stat 漂移: {patient_id}")
        expected_sha = str(row["cache_sha256"])
        if full_content_hash and file_sha256(path) != expected_sha:
            raise ValueError(f"cache content SHA 漂移: {patient_id}")
        observed_pairs.append((expected_sha, int(stat.st_size)))
        total_bytes += int(stat.st_size)
    if (
        total_bytes != int(public["cache_total_bytes"])
        or canonical_json_sha256(sorted(observed_pairs))
        != public["content_multiset_sha256"]
    ):
        raise ValueError("cache aggregate content contract 漂移")
    return public


def write_source_contract() -> Path:
    if SOURCE_CONTRACT.exists():
        raise FileExistsError("拒绝覆盖已有 SOURCE_CONTRACT.json")
    plan = assert_plan_freeze()
    context = load_data_context()
    validate_manifests(context, verify_checkpoint_pools=True)
    cache = assert_cache_input_contract(context, full_content_hash=False)
    artifact_hashes = {
        relative: file_sha256(AUDIT_ROOT / relative)
        for relative in SOURCE_CONTRACT_ARTIFACTS
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "plan_payload_sha256": plan["payload_sha256"],
        "plan_file_sha256": file_sha256(PLAN_FREEZE),
        "public_artifact_count": len(SOURCE_CONTRACT_ARTIFACTS),
        "public_artifact_sha256": artifact_hashes,
        "upstream_source_sha256": assert_source_hashes(),
        "source_manifest_rows": SOURCE_MANIFEST_ROWS,
        "checkpoint_rows": 31,
        "selected_checkpoint_rows": 25,
        "representative_last_checkpoint_rows": 6,
        "public_batch_rows": 80,
        "private_batch_rows": 2560,
        "cache_files": int(cache["cache_files"]),
        "cache_total_bytes": int(cache["cache_total_bytes"]),
        "cache_content_multiset_sha256": cache["content_multiset_sha256"],
        "private_commitment_kind": "HMAC-SHA256",
        "contract_complete": True,
        "contains_patient_ids": False,
    }
    payload["payload_sha256"] = canonical_json_sha256(payload)
    atomic_json(SOURCE_CONTRACT, payload)
    assert_source_contract(full_content_hash=False, full_checkpoint_hash=True)
    return SOURCE_CONTRACT


def assert_source_contract(
    *, full_content_hash: bool = False, full_checkpoint_hash: bool = False
) -> dict[str, Any]:
    plan = assert_plan_freeze()
    if not SOURCE_CONTRACT.is_file():
        raise FileNotFoundError(
            "缺 SOURCE_CONTRACT.json；gradient 正式步骤 fail-closed"
        )
    payload = json.loads(SOURCE_CONTRACT.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != SOURCE_CONTRACT_KEYS:
        raise ValueError("SOURCE_CONTRACT exact top-level key set 漂移")
    unsigned = dict(payload)
    expected_digest = str(unsigned.pop("payload_sha256", ""))
    if (
        int(payload.get("schema_version", -1)) != 1
        or canonical_json_sha256(unsigned) != expected_digest
    ):
        raise ValueError("SOURCE_CONTRACT payload digest 不闭环")
    try:
        created_at = dt.datetime.fromisoformat(str(payload["created_at_utc"]))
    except ValueError as error:
        raise ValueError("SOURCE_CONTRACT created_at_utc 非法") from error
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise ValueError("SOURCE_CONTRACT created_at_utc 必须含 timezone")
    if str(payload.get("plan_payload_sha256")) != str(plan["payload_sha256"]) or str(
        payload.get("plan_file_sha256")
    ) != file_sha256(PLAN_FREEZE):
        raise ValueError("SOURCE_CONTRACT plan payload/file SHA 漂移")
    expected_artifacts = payload.get("public_artifact_sha256", {})
    if (
        not isinstance(expected_artifacts, dict)
        or set(expected_artifacts) != set(SOURCE_CONTRACT_ARTIFACTS)
        or _strict_nonnegative_int(
            payload["public_artifact_count"], "public_artifact_count"
        )
        != len(SOURCE_CONTRACT_ARTIFACTS)
    ):
        raise ValueError("SOURCE_CONTRACT exact 11-artifact mapping 非法")
    for relative in SOURCE_CONTRACT_ARTIFACTS:
        expected = _sha256_string(
            expected_artifacts[relative], f"public artifact {relative} SHA"
        )
        path = AUDIT_ROOT / relative
        if not path.is_file() or path.is_symlink() or file_sha256(path) != expected:
            raise ValueError(f"SOURCE_CONTRACT artifact 漂移: {relative}")
    if payload.get("upstream_source_sha256") != assert_source_hashes():
        raise ValueError("SOURCE_CONTRACT upstream source 漂移")
    context = load_data_context()
    public_batches, private_batches = validate_manifests(
        context, verify_checkpoint_pools=False
    )
    cache = assert_cache_input_contract(context, full_content_hash=full_content_hash)
    assets = pd.read_csv(AUDIT_ROOT / "metrics" / "asset_manifest.csv")
    ensure_no_patient_columns(assets.columns)
    expected_keys = {
        (seed, fold, "selected")
        for seed in (2026, 3026, 4026, 5026, 6026)
        for fold in range(5)
    }
    representatives = pd.read_csv(AUDIT_ROOT / "metrics" / "representative_runs.csv")
    expected_keys.update(
        (int(row.seed_base), int(row.fold), "last")
        for row in representatives.itertuples(index=False)
    )
    observed_keys = set(
        assets[["seed_base", "fold", "checkpoint_kind"]].itertuples(
            index=False, name=None
        )
    )
    if (
        tuple(assets.columns) != ASSET_MANIFEST_COLUMNS
        or len(assets) != 31
        or observed_keys != expected_keys
        or assets.duplicated(["seed_base", "fold", "checkpoint_kind"]).any()
    ):
        raise ValueError("SOURCE_CONTRACT asset grid 错误")
    selected_count = int(assets["checkpoint_kind"].eq("selected").sum())
    last_count = int(assets["checkpoint_kind"].eq("last").sum())
    for row in assets.itertuples(index=False):
        path = REPO_ROOT / str(row.checkpoint)
        try:
            path.resolve().relative_to(REPO_ROOT.resolve())
        except ValueError as error:
            raise ValueError("asset checkpoint path 越出 repository") from error
        if not path.is_file() or int(row.checkpoint_bytes) != path.stat().st_size:
            raise ValueError("asset checkpoint path/bytes 漂移")
        history = REPO_ROOT / str(row.history)
        selection = REPO_ROOT / str(row.selection)
        if (
            not history.is_file()
            or not selection.is_file()
            or file_sha256(history) != str(row.history_sha256)
            or file_sha256(selection) != str(row.selection_sha256)
        ):
            raise ValueError("asset history/selection SHA 漂移")
        is_last = str(row.checkpoint_kind) == "last"
        if (
            _strict_bool(
                row.state_distinct_from_selected, "state_distinct_from_selected"
            )
            != is_last
            or _strict_bool(
                row.formal_inference_checkpoint, "formal_inference_checkpoint"
            )
            == is_last
            or _strict_bool(row.contains_patient_ids, "contains_patient_ids")
        ):
            raise ValueError("asset state/inference/privacy flag 漂移")
        if full_checkpoint_hash and file_sha256(path) != str(row.checkpoint_sha256):
            raise ValueError("asset checkpoint content SHA 漂移")
    source_manifest = pd.read_csv(SOURCE_MANIFEST, keep_default_na=False)
    expected_source_ids = {
        *(f"upstream::{relative}" for relative in EXPECTED_SOURCE_SHA256),
        *(
            f"{kind}::{seed}::{fold}"
            for seed in SEED_BASES
            for fold in FOLDS
            for kind in ("g1_selected", "g3_selected", "g3_history", "g3_selection")
        ),
        *(
            f"g3_last::{int(row.seed_base)}::{int(row.fold)}"
            for row in representatives.itertuples(index=False)
        ),
    }
    if (
        tuple(source_manifest.columns) != SOURCE_MANIFEST_COLUMNS
        or len(source_manifest) != SOURCE_MANIFEST_ROWS
        or source_manifest["artifact_id"].duplicated().any()
        or set(source_manifest["artifact_id"]) != expected_source_ids
    ):
        raise ValueError("source manifest schema/grid 漂移")
    ensure_no_patient_columns(source_manifest.columns)
    for row in source_manifest.itertuples(index=False):
        path = REPO_ROOT / str(row.path)
        try:
            path.resolve().relative_to(REPO_ROOT.resolve())
        except ValueError as error:
            raise ValueError("source manifest path 越出 repository") from error
        if not path.is_file() or int(row.bytes) != path.stat().st_size:
            raise ValueError("source manifest path/bytes 漂移")
        if full_checkpoint_hash and file_sha256(path) != str(row.sha256):
            raise ValueError("source manifest artifact SHA 漂移")
        expected_embedded_ids = str(row.artifact_kind) in {
            "paired_g1_selected_checkpoint",
            "formal_g3_selected_checkpoint",
            "representative_g3_last_checkpoint",
        }
        if _strict_bool(
            row.referenced_artifact_may_embed_patient_ids,
            "referenced_artifact_may_embed_patient_ids",
        ) != expected_embedded_ids or _strict_bool(
            row.manifest_contains_patient_ids,
            "manifest_contains_patient_ids",
        ):
            raise ValueError("source manifest embedding/privacy flags 失败")

    declared_counts = {
        "source_manifest_rows": len(source_manifest),
        "checkpoint_rows": len(assets),
        "selected_checkpoint_rows": selected_count,
        "representative_last_checkpoint_rows": last_count,
        "public_batch_rows": len(public_batches),
        "private_batch_rows": len(private_batches),
        "cache_files": int(cache["cache_files"]),
        "cache_total_bytes": int(cache["cache_total_bytes"]),
    }
    for name, observed in declared_counts.items():
        if _strict_nonnegative_int(payload[name], name) != observed:
            raise ValueError(
                f"SOURCE_CONTRACT declared count 漂移: {name}={payload[name]} != {observed}"
            )
    fixed_counts = {
        "source_manifest_rows": SOURCE_MANIFEST_ROWS,
        "checkpoint_rows": 31,
        "selected_checkpoint_rows": 25,
        "representative_last_checkpoint_rows": 6,
        "public_batch_rows": 80,
        "private_batch_rows": 2560,
        "cache_files": 964,
    }
    if any(
        declared_counts[name] != expected for name, expected in fixed_counts.items()
    ):
        raise ValueError("SOURCE_CONTRACT fixed count contract 失败")
    if (
        _sha256_string(
            payload["cache_content_multiset_sha256"],
            "SOURCE_CONTRACT cache multiset SHA",
        )
        != str(cache["content_multiset_sha256"])
        or str(payload["private_commitment_kind"]) != "HMAC-SHA256"
        or payload["contract_complete"] is not True
        or payload["contains_patient_ids"] is not False
    ):
        raise ValueError("SOURCE_CONTRACT declared hash/commit/privacy flags 漂移")
    return payload


__all__ = [
    "PRIVATE_CACHE_MANIFEST",
    "PUBLIC_CACHE_CONTRACT",
    "SOURCE_CONTRACT_ARTIFACTS",
    "SOURCE_CONTRACT_KEYS",
    "SOURCE_MANIFEST",
    "assert_cache_input_contract",
    "assert_source_contract",
    "write_cache_input_contract",
    "write_source_manifest",
    "write_source_contract",
]

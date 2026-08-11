from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import verify_cache_integrity as integrity  # noqa: E402
from common import file_sha256, ordered_sha256  # noqa: E402


def _cell_keys() -> tuple[str, ...]:
    return tuple(
        f"seed_{seed}/{arm}/fold_{fold}"
        for seed in (2026, 3026)
        for arm in ("LOCAL0", "LOCAL3")
        for fold in range(5)
    )


def _synthetic_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    primary_patient_ids = ("patient-000", "patient-001", "patient-002")
    train_only_patient_ids = ("patient-003", "patient-004")
    patient_ids = primary_patient_ids + train_only_patient_ids
    reference_root = tmp_path / "references"
    reference_root.mkdir()
    cells: dict[str, Any] = {}
    for index, key in enumerate(_cell_keys()):
        reference = reference_root / f"reference-{index:02d}.npz"
        # Alternate order to prove that identity is set-based across all cells.
        ordered = (
            primary_patient_ids
            if index % 2 == 0
            else tuple(reversed(primary_patient_ids))
        )
        np.savez(reference, patient_id=np.asarray(ordered))
        cells[key] = {
            "reference": {
                "path": str(reference.resolve()),
                "sha256": file_sha256(reference),
                "patient_order_sha256": ordered_sha256(ordered),
            }
        }

    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    cache_rows: list[dict[str, Any]] = []
    for index, patient_id in enumerate(patient_ids):
        path = cache_root / f"{patient_id}.npz"
        path.write_bytes((f"frozen-cache-{index}-" * (index + 1)).encode("ascii"))
        stat = path.stat()
        cache_rows.append(
            {
                "patient_id": patient_id,
                "cache_path": str(path.resolve()),
                "cache_sha256": file_sha256(path),
                "cache_size_bytes": stat.st_size,
                "cache_mtime_ns": stat.st_mtime_ns,
                "input_kind": "c1b",
            }
        )
    manifest = tmp_path / "c1b_cache.private.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=integrity.MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(cache_rows)

    config = {
        "frozen_cells": {
            "seed_bases": [2026, 3026],
            "arms": ["LOCAL0", "LOCAL3"],
            "folds": list(range(5)),
            "patient_count": len(primary_patient_ids),
        },
        "paths": {
            "c1b_cache_manifest": manifest,
            "c1b_cache_manifest_sha256": file_sha256(manifest),
        },
    }
    lock = {
        "selected_cells": cells,
        "implementation_sha256": {
            integrity.IMPLEMENTATION_KEY: file_sha256(Path(integrity.__file__))
        },
    }
    lock_path = tmp_path / "PREREGISTRATION_LOCK.json"
    lock_path.write_text(json.dumps(lock, sort_keys=True), encoding="utf-8")
    private_path = tmp_path / "manifests" / "cache_integrity.private.json"
    contract_path = tmp_path / "metrics" / "cache_integrity_contract.json"

    monkeypatch.setattr(
        integrity, "EXPECTED_PRIMARY_PATIENT_COUNT", len(primary_patient_ids)
    )
    monkeypatch.setattr(
        integrity, "EXPECTED_TRAIN_ONLY_PATIENT_COUNT", len(train_only_patient_ids)
    )
    monkeypatch.setattr(integrity, "EXPECTED_PATIENT_COUNT", len(patient_ids))
    monkeypatch.setattr(integrity, "PREREGISTRATION_LOCK", lock_path)
    monkeypatch.setattr(integrity, "PRIVATE_MANIFEST", private_path)
    monkeypatch.setattr(integrity, "PUBLIC_CONTRACT", contract_path)
    monkeypatch.setattr(integrity, "require_preregistration_lock", lambda _: lock)
    return {
        "config": config,
        "lock": lock,
        "patient_ids": patient_ids,
        "primary_patient_ids": primary_patient_ids,
        "train_only_patient_ids": train_only_patient_ids,
        "cache_rows": cache_rows,
        "private_path": private_path,
        "contract_path": contract_path,
    }


def test_build_authenticate_and_refuse_complete_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _synthetic_contract(tmp_path, monkeypatch)
    public = integrity.build_cache_integrity(case["config"], case["lock"], workers=2)
    assert public["status"] == "COMPLETE"
    assert public["schema_version"] == 2
    assert public["patient_count"] == 5
    assert public["primary_patient_count"] == 3
    assert public["train_only_patient_count"] == 2
    assert public["total_bytes"] == sum(
        row["cache_size_bytes"] for row in case["cache_rows"]
    )
    assert len(public["canonical_record_set_sha256"]) == 64
    assert len(public["primary_record_set_sha256"]) == 64
    assert len(public["train_only_record_set_sha256"]) == 64
    assert len(public["private_artifact_sha256"]) == 64
    assert isinstance(public["environment"], dict)
    assert case["private_path"].stat().st_mode & 0o077 == 0
    assert "patient-000" not in case["contract_path"].read_text(encoding="utf-8")

    authenticated = integrity.require_cache_integrity(case["config"], case["lock"])
    assert [record["patient_id"] for record in authenticated["records"]] == list(
        case["patient_ids"]
    )
    assert [record["cohort"] for record in authenticated["records"]] == [
        "primary",
        "primary",
        "primary",
        "train_only",
        "train_only",
    ]
    with pytest.raises(FileExistsError, match="complete"):
        integrity.build_cache_integrity(case["config"], case["lock"])


def test_content_substitution_with_manifest_stats_preserved_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _synthetic_contract(tmp_path, monkeypatch)
    # A train-only cache is part of the same mandatory one-time content proof.
    row = case["cache_rows"][-1]
    path = Path(row["cache_path"])
    before = path.stat()
    original = path.read_bytes()
    replacement = bytes((value ^ 0x01) for value in original)
    assert len(replacement) == len(original)
    path.write_bytes(replacement)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))

    with pytest.raises(integrity.CacheIntegrityError, match="content hash"):
        integrity.build_cache_integrity(case["config"], case["lock"])
    assert not case["private_path"].exists()
    assert not case["contract_path"].exists()


def test_reusable_guard_authenticates_artifact_and_optionally_checks_live_stats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _synthetic_contract(tmp_path, monkeypatch)
    integrity.build_cache_integrity(case["config"], case["lock"])

    cache_path = Path(case["cache_rows"][0]["cache_path"])
    cache_path.write_bytes(cache_path.read_bytes() + b"drift")
    # Stat-free use authenticates the one-time proof and never rehashes live caches.
    integrity.require_cache_integrity(
        case["config"], case["lock"], verify_live_stats=False
    )
    with pytest.raises(integrity.CacheIntegrityError, match="live stat"):
        integrity.require_cache_integrity(case["config"], case["lock"])

    private = json.loads(case["private_path"].read_text(encoding="utf-8"))
    private["records"][0]["sha256"] = "0" * 64
    case["private_path"].write_text(json.dumps(private), encoding="utf-8")
    with pytest.raises(integrity.CacheIntegrityError, match="authenticate"):
        integrity.require_cache_integrity(
            case["config"], case["lock"], verify_live_stats=False
        )


def test_reusable_guard_requires_the_exact_runtime_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _synthetic_contract(tmp_path, monkeypatch)
    integrity.build_cache_integrity(case["config"], case["lock"])
    contract = json.loads(case["contract_path"].read_text(encoding="utf-8"))
    contract["environment"]["machine"] = "tampered-machine"
    case["contract_path"].write_text(
        json.dumps(contract, sort_keys=True), encoding="utf-8"
    )
    case["contract_path"].chmod(0o644)

    with pytest.raises(integrity.CacheIntegrityError, match="exact runtime"):
        integrity.require_cache_integrity(
            case["config"], case["lock"], verify_live_stats=False
        )


def test_incomplete_pair_is_removed_and_rebuilt_only_after_lock_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _synthetic_contract(tmp_path, monkeypatch)
    case["private_path"].parent.mkdir(parents=True)
    case["private_path"].write_text('{"status":"INCOMPLETE"}\n', encoding="utf-8")

    original_guard = integrity.require_preregistration_lock
    monkeypatch.setattr(
        integrity,
        "require_preregistration_lock",
        lambda _: (_ for _ in ()).throw(RuntimeError("lock rejected")),
    )
    with pytest.raises(RuntimeError, match="lock rejected"):
        integrity.build_cache_integrity(case["config"], case["lock"])
    assert (
        case["private_path"].read_text(encoding="utf-8") == '{"status":"INCOMPLETE"}\n'
    )

    monkeypatch.setattr(integrity, "require_preregistration_lock", original_guard)
    result = integrity.build_cache_integrity(case["config"], case["lock"])
    assert result["status"] == "COMPLETE"
    assert case["private_path"].exists() and case["contract_path"].exists()


def test_all_twenty_reference_sets_must_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _synthetic_contract(tmp_path, monkeypatch)
    first = case["lock"]["selected_cells"][_cell_keys()[0]]["reference"]
    path = Path(first["path"])
    np.savez(path, patient_id=np.asarray(("patient-000", "patient-001", "intruder")))
    first["sha256"] = file_sha256(path)
    first["patient_order_sha256"] = ordered_sha256(
        ("patient-000", "patient-001", "intruder")
    )
    # Keep the on-disk lock current so the cohort-set check is the failing layer.
    integrity.PREREGISTRATION_LOCK.write_text(
        json.dumps(case["lock"], sort_keys=True), encoding="utf-8"
    )

    with pytest.raises(integrity.CacheIntegrityError, match="sets are not identical"):
        integrity.build_cache_integrity(case["config"], case["lock"])

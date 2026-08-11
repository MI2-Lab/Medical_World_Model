"""Trigger-bound orchestration for the exact conditional S3 probe matrix.

Statistical fitting remains in the dimension-agnostic, immutable Stage-B probe
adapter.  This module owns the S3-specific trigger, 100-feature inventory,
metadata/provenance validation, and exact no-nuisance matrix construction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import (
    ARMS,
    EXPERIMENT_ROOT,
    FOLDS,
    REPO_ROOT,
    SEEDS,
    canonical_sha256,
    cell_key,
    file_sha256,
)
from .probe_runner import (
    ProbeCellKey,
    ProbeCellSpec,
    ProbeMatrixPlan,
    output_path_for,
    validate_alternative_gates,
)
from .probes import load_frozen_state_asset
from .runtime import load_stage_b_bundle, verify_preregistration
from .s3_exporter import (
    S3_C1B_POOLINGS,
    S3_LEGACY_POOLINGS,
    S3_REPRESENTATION_CONTRACT,
    s3_feature_asset_path,
    s3_feature_metadata_path,
    validate_s3_feature_export,
)
from .s3_trigger import require_s3_trigger_authorization


S3_POOLINGS = ("P0", "PLOCAL", "PORACLE")
EXPECTED_S3_FEATURE_ASSETS = 100
EXPECTED_S3_PROBE_CELLS = 100


def _read_json(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {source}")
    return payload


def _resolve_inventory_path(value: str) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def validate_s3_exporter_completion(
    feature_root: str | Path,
    *,
    preregistration_sha256: str,
    trigger_gate_sha256: str,
) -> str:
    """Validate the exact nonresumable 40-cell/100-asset S3 export."""

    root = Path(feature_root).resolve()
    if root != (EXPERIMENT_ROOT / "features").resolve():
        raise ValueError("formal S3 probe requires the canonical features root")
    stage_root = root / "s3"
    completion_path = stage_root / "feature_export_complete.private.json"
    preflight_path = stage_root / "feature_export_preflight.private.json"
    completion = _read_json(completion_path, label="S3 feature completion")
    expected_fields = {
        "schema_version",
        "status",
        "stage",
        "representation_contract",
        "run_count",
        "cell_count",
        "expected_asset_count",
        "feature_metadata_sha256",
        "preflight_sha256",
        "trigger_gate_sha256",
        "sidecar_sha256",
        "sidecar_metadata_sha256",
        "preregistration_lock_sha256",
    }
    if set(completion) != expected_fields:
        raise ValueError("S3 feature completion schema drifted")
    if (
        completion.get("schema_version") != 1
        or completion.get("status") != "COMPLETE"
        or completion.get("stage") != "s3"
        or completion.get("representation_contract") != S3_REPRESENTATION_CONTRACT
        or int(completion.get("run_count", -1)) != 40
        or int(completion.get("cell_count", -1)) != 40
        or int(completion.get("expected_asset_count", -1))
        != EXPECTED_S3_FEATURE_ASSETS
        or completion.get("preregistration_lock_sha256") != preregistration_sha256
        or completion.get("trigger_gate_sha256") != trigger_gate_sha256
    ):
        raise ValueError("S3 feature completion identity/count binding drifted")
    if not preflight_path.is_file() or file_sha256(preflight_path) != completion.get(
        "preflight_sha256"
    ):
        raise ValueError("S3 feature preflight binding drifted")
    preflight = _read_json(preflight_path, label="S3 feature preflight")
    if (
        preflight.get("status") != "PREFLIGHT_PASS"
        or preflight.get("stage") != "s3"
        or preflight.get("representation_contract") != S3_REPRESENTATION_CONTRACT
        or int(preflight.get("cell_count", -1)) != 40
        or int(preflight.get("expected_asset_count", -1))
        != EXPECTED_S3_FEATURE_ASSETS
        or preflight.get("preregistration_lock_sha256") != preregistration_sha256
        or preflight.get("trigger_gate_sha256") != trigger_gate_sha256
        or preflight.get("sidecar_sha256") != completion.get("sidecar_sha256")
        or preflight.get("sidecar_metadata_sha256")
        != completion.get("sidecar_metadata_sha256")
    ):
        raise ValueError("S3 feature preflight contract drifted")
    inventory = completion.get("feature_metadata_sha256")
    if not isinstance(inventory, Mapping) or len(inventory) != EXPECTED_S3_FEATURE_ASSETS:
        raise ValueError("S3 metadata inventory is not exactly 100 assets")
    resolved: dict[Path, str] = {}
    for raw_path, raw_digest in inventory.items():
        path = _resolve_inventory_path(str(raw_path))
        digest = str(raw_digest)
        if not path.is_relative_to(stage_root) or path in resolved:
            raise ValueError("S3 metadata inventory path escaped or duplicated")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("S3 metadata inventory contains a non-SHA256 digest")
        resolved[path] = digest
    live_metadata = {path.resolve() for path in stage_root.rglob("*.private.metadata.json")}
    if set(resolved) != live_metadata:
        raise ValueError("S3 completion does not inventory the exact live metadata set")
    for path, digest in resolved.items():
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"S3 exporter metadata inventory drifted: {path}")
        if path.stat().st_mode & 0o077:
            raise PermissionError("S3 feature metadata must remain owner-only")
    for private_path in (completion_path, preflight_path):
        if private_path.stat().st_mode & 0o077:
            raise PermissionError("S3 feature control artifacts must be owner-only")
    return file_sha256(completion_path)


def _expected_keys() -> tuple[ProbeCellKey, ...]:
    return tuple(
        ProbeCellKey(seed, arm, fold, pooling)
        for seed in SEEDS
        for arm in ARMS
        for fold in FOLDS
        for pooling in (
            S3_LEGACY_POOLINGS if arm.startswith("L") else S3_C1B_POOLINGS
        )
    )


def _validate_fold_split(asset: Any, folds: Any) -> None:
    required = {"patient_id", "fold", "split"}
    if missing := sorted(required.difference(folds.columns)):
        raise ValueError(f"frozen fold manifest misses columns: {missing}")
    current = folds.loc[
        folds["fold"].astype(int).eq(asset.fold), ["patient_id", "fold", "split"]
    ].copy()
    current["patient_id"] = current["patient_id"].astype(str)
    current["split"] = current["split"].astype(str).str.lower()
    if current.empty or current["patient_id"].duplicated().any():
        raise ValueError("frozen S3 fold is empty or duplicates patients")
    expected = dict(zip(current["patient_id"], current["split"], strict=True))
    observed = dict(zip(asset.patient_id, asset.split, strict=True))
    if observed != expected:
        raise ValueError("S3 patient identities/splits disagree with frozen fold manifest")


def s3_plan_summary(plan: ProbeMatrixPlan, *, status: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": status,
        "stage": "s3",
        "representation_contract": S3_REPRESENTATION_CONTRACT,
        "requested_poolings": list(S3_POOLINGS),
        "expected_cell_count": len(plan.cells),
        "nuisance_cell_count": 0,
        "feature_root": str(plan.feature_root),
        "probe_root": str(plan.probe_root),
        "completion_path": str(plan.completion_path),
        "preregistration_lock_sha256": plan.preregistration_sha256,
        "gate_sha256": dict(plan.gate_sha256),
        "exporter_completion_sha256": plan.exporter_completion_sha256,
        "legacy_poracle": "NA_incomplete_source_authoritative_support_1488_of_1500",
        "fabricated_unavailable_rows": 0,
        "new_training_performed": False,
    }


def prepare_s3_probe_matrix(
    *,
    feature_root: str | Path = EXPERIMENT_ROOT / "features",
    probe_root: str | Path = EXPERIMENT_ROOT / "probes",
    trigger_gate_path: str | Path = EXPERIMENT_ROOT
    / "metrics"
    / "s3_trigger_authorization.json",
    equivalence_gate_path: str | Path = EXPERIMENT_ROOT
    / "metrics"
    / "p0_equivalence_gate.json",
    replication_gate_path: str | Path = EXPERIMENT_ROOT
    / "metrics"
    / "p0_probe_replication_gate.json",
) -> tuple[ProbeMatrixPlan, Mapping[str, Any]]:
    """Preflight the complete S3 feature/probe matrix without fitting."""

    trigger_path = Path(trigger_gate_path).resolve()
    if trigger_path != (
        EXPERIMENT_ROOT / "metrics" / "s3_trigger_authorization.json"
    ).resolve():
        raise ValueError("formal S3 probes require the canonical trigger gate")
    require_s3_trigger_authorization(trigger_path, verify_live=True)
    lock = verify_preregistration()
    preregistration_path = (EXPERIMENT_ROOT / "PREREGISTRATION_LOCK.json").resolve()
    preregistration_sha256 = file_sha256(preregistration_path)
    gate_hashes = validate_alternative_gates(
        S3_POOLINGS,
        preregistration_sha256=preregistration_sha256,
        equivalence_gate_path=equivalence_gate_path,
        probe_replication_gate_path=replication_gate_path,
    )
    gate_hashes = {
        **gate_hashes,
        "s3_trigger_authorization_sha256": file_sha256(trigger_path),
        # The generic publisher owns atomic cell/completion writes; bind this
        # S3-specific discovery/trigger adapter alongside that publisher hash.
        "s3_probe_runner_sha256": file_sha256(Path(__file__)),
    }
    feature_root_path = Path(feature_root).resolve()
    completion_sha256 = validate_s3_exporter_completion(
        feature_root_path,
        preregistration_sha256=preregistration_sha256,
        trigger_gate_sha256=file_sha256(trigger_path),
    )
    _authorization, paths, data = load_stage_b_bundle(verify_cache_files=False)
    probe_root_path = Path(probe_root).resolve()
    if probe_root_path != (EXPERIMENT_ROOT / "probes").resolve():
        raise ValueError("formal S3 probes must remain in this audit's probes root")

    expected_keys = _expected_keys()
    if len(expected_keys) != EXPECTED_S3_PROBE_CELLS:
        raise AssertionError("S3 probe plan must contain exactly 100 cells")
    expected_paths = {
        s3_feature_asset_path(
            feature_root_path,
            key.seed_base,
            key.arm,
            key.fold,
            key.pooling,
        ).resolve()
        for key in expected_keys
    }
    live_paths = {
        path.resolve() for path in (feature_root_path / "s3").rglob("*.private.npz")
    }
    if live_paths != expected_paths:
        raise ValueError(
            "S3 feature matrix contains missing/extra assets: "
            f"missing={len(expected_paths-live_paths)}, extra={len(live_paths-expected_paths)}"
        )

    specifications: list[ProbeCellSpec] = []
    for key in expected_keys:
        feature = s3_feature_asset_path(
            feature_root_path, key.seed_base, key.arm, key.fold, key.pooling
        )
        asset, _metadata = validate_s3_feature_export(
            feature,
            expected_arm=key.arm,
            expected_seed_base=key.seed_base,
            expected_fold=key.fold,
            expected_pooling=key.pooling,
            verify_live_inputs=True,
        )
        if feature.stat().st_mode & 0o077 or s3_feature_metadata_path(feature).stat().st_mode & 0o077:
            raise PermissionError("S3 feature assets/metadata must remain owner-only")
        generic_asset = load_frozen_state_asset(feature)
        if generic_asset.feature_dim != 64:
            raise ValueError("S3 raw pooled feature dimension must be exactly 64")
        _validate_fold_split(generic_asset, data.folds)
        metadata_path = s3_feature_metadata_path(feature)
        feature_sha256 = file_sha256(feature)
        metadata_sha256 = file_sha256(metadata_path)
        checkpoint_name = cell_key(key.seed_base, key.arm, key.fold)
        provenance = {
            "stage": "s3",
            "representation_contract": S3_REPRESENTATION_CONTRACT,
            "checkpoint_lock_key": checkpoint_name,
            "checkpoint_sha256": lock["selected_checkpoints"][checkpoint_name]["sha256"],
            "reference_feature_sha256": lock["formal_p0_references"][checkpoint_name][
                "feature_sha256"
            ],
            "reference_feature_metadata_sha256": lock["formal_p0_references"][
                checkpoint_name
            ]["feature_metadata_sha256"],
            "preregistration_lock_sha256": preregistration_sha256,
            "plan_sha256": lock["plan_sha256"],
            "config_sha256": lock["config_sha256"],
            "fold_manifest_path": str(paths.fold_manifest),
            "fold_manifest_sha256": paths.fold_manifest_sha256,
            "data_contract_provenance_sha256": canonical_sha256(data.provenance),
            "s3_trigger_authorization_sha256": file_sha256(trigger_path),
            "s3_exporter_completion_sha256": completion_sha256,
            "s3_probe_runner_sha256": file_sha256(Path(__file__)),
            "exporter_feature_sha256": feature_sha256,
            "exporter_metadata_sha256": metadata_sha256,
            **gate_hashes,
        }
        specifications.append(
            ProbeCellSpec(
                key=key,
                stage="s3",
                feature_path=feature.resolve(),
                feature_metadata_path=metadata_path.resolve(),
                feature_sha256=feature_sha256,
                feature_metadata_sha256=metadata_sha256,
                private_root=probe_root_path,
                output_dir=output_path_for(probe_root_path, "s3", key),
                include_nuisance=False,
                provenance=provenance,
            )
        )
    plan = ProbeMatrixPlan(
        stage="s3",
        poolings=S3_POOLINGS,
        cells=tuple(specifications),
        feature_root=feature_root_path,
        probe_root=probe_root_path,
        preregistration_path=preregistration_path,
        preregistration_sha256=preregistration_sha256,
        nuisance_path=None,
        nuisance_sha256=None,
        gate_sha256=gate_hashes,
        exporter_completion_sha256=completion_sha256,
    )
    return plan, data.ftv


__all__ = [
    "EXPECTED_S3_FEATURE_ASSETS",
    "EXPECTED_S3_PROBE_CELLS",
    "S3_POOLINGS",
    "prepare_s3_probe_matrix",
    "s3_plan_summary",
    "validate_s3_exporter_completion",
]

"""Public, identifier-free provenance validation for the confirmed S0 assets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import file_sha256, validate_seed_fold


def load_s0_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("S0 confirmation provenance is missing or invalid") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("S0 confirmation provenance must be a schema-v1 object")
    expected = {
        "experiment": "residual_sph_grounding_pilot",
        "artifact": "confirmed_LOCAL3_S0_runtime_provenance",
        "status": "S0_CONFIRMATION_PROVENANCE_VERIFIED",
        "cell_count": 10,
        "patient_identifiers_in_manifest": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"S0 provenance differs at {key}")
    ancestry = payload.get("source_ancestry")
    if not isinstance(ancestry, dict):
        raise ValueError("S0 provenance lacks source ancestry")
    expected_ancestry = {
        "source_lock_head": "78ba693ad34dbb2b5a28f0476185966714bb63c5",
        "source_delivery_commit": "b4ec0c1473da513f2b19baa58d54c0fd5382e52f",
        "source_lock_is_ancestor_of_delivery_commit": True,
        "required_classification": "LOCAL_MULTISEED_CONFIRMED",
    }
    for key, value in expected_ancestry.items():
        if ancestry.get(key) != value:
            raise ValueError(f"S0 source ancestry differs at {key}")
    evidence = ancestry.get("verified_prior_evidence_sha256")
    if not isinstance(evidence, dict) or set(evidence) != {
        "local_confirmation_report",
        "local_confirmation_decision",
        "local_confirmation_static_ftv",
        "local_confirmation_delta_ftv",
    }:
        raise ValueError("S0 provenance lacks exact prior-evidence hash coverage")
    cells = payload.get("cells")
    if not isinstance(cells, list) or len(cells) != 10:
        raise ValueError("S0 provenance must contain ten cells")
    identities = {(int(cell["seed_base"]), int(cell["fold"])) for cell in cells}
    if identities != {(seed, fold) for seed in (2026, 3026) for fold in range(5)}:
        raise ValueError("S0 provenance matrix coverage drifted")
    return payload


def validate_s0_cell(
    manifest: Mapping[str, Any],
    *,
    seed_base: int,
    fold: int,
    selection_path: str | Path,
    checkpoint_path: str | Path | None = None,
    feature_path: str | Path | None = None,
    feature_metadata_path: str | Path | None = None,
) -> dict[str, Any]:
    effective_seed = validate_seed_fold(seed_base, fold)
    matches = [
        cell
        for cell in manifest.get("cells", [])
        if int(cell.get("seed_base", -1)) == int(seed_base)
        and int(cell.get("fold", -1)) == int(fold)
    ]
    if len(matches) != 1:
        raise ValueError("S0 provenance cell identity is missing or duplicated")
    cell = dict(matches[0])
    if cell.get("effective_seed") != effective_seed or cell.get("arm") != "S0":
        raise ValueError("S0 provenance cell metadata drifted")
    for key in (
        "train_patient_sha256",
        "val_patient_sha256",
        "data_provenance_sha256",
        "hyperparameters",
        "ftv_transform_sha256",
    ):
        if key not in cell:
            raise ValueError(f"S0 provenance cell lacks {key}")
    assets = [(selection_path, "selection_sha256")]
    if checkpoint_path is not None:
        assets.append((checkpoint_path, "checkpoint_sha256"))
    if feature_path is not None:
        assets.append((feature_path, "feature_sha256"))
    if feature_metadata_path is not None:
        assets.append((feature_metadata_path, "feature_metadata_sha256"))
    for raw_path, hash_key in assets:
        path = Path(raw_path).resolve()
        if not path.is_file() or file_sha256(path) != cell.get(hash_key):
            raise ValueError(f"S0 {hash_key} asset is missing or hash-mismatched")
    if feature_metadata_path is not None:
        metadata = json.loads(Path(feature_metadata_path).read_text(encoding="utf-8"))
        expected_metadata = {
            "arm": "LOCAL3",
            "seed_base": int(seed_base),
            "fold": int(fold),
            "feature_sha256": cell.get("feature_sha256"),
            "checkpoint_sha256": cell.get("checkpoint_sha256"),
            "selection_sha256": cell.get("selection_sha256"),
            "ftv_head_called": False,
            "test_labels_used": False,
        }
        for key, value in expected_metadata.items():
            if metadata.get(key) != value:
                raise ValueError(f"S0 feature metadata differs at {key}")
    return cell


__all__ = ["load_s0_manifest", "validate_s0_cell"]

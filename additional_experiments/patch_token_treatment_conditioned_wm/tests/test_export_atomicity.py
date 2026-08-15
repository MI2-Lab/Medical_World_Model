from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module(name: str, script: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EXPORT = _module("patch_atomic_export_cell", "export_cell.py")
RUN_EXPORTS = _module("patch_atomic_run_exports", "run_exports.py")


def _write_complete_staging(staged: Path) -> dict[str, object]:
    token = staged / "tokens.private.npz"
    dynamics = staged / "dynamics.private.npz"
    EXPORT._atomic_npz(token, tokens=np.zeros((1, 2), dtype=np.float32))
    EXPORT._atomic_npz(dynamics, actual=np.ones((1,), dtype=np.float32))
    metadata: dict[str, object] = {
        "status": "COMPLETE",
        "token_feature_sha256": EXPORT.file_sha256(token),
        "dynamics_sha256": EXPORT.file_sha256(dynamics),
    }
    marker = staged / "tokens.private.metadata.json"
    EXPORT._atomic_json(marker, metadata)
    EXPORT._validate_staged_export(token, dynamics, marker, metadata)
    return metadata


def test_cell_triplet_is_published_together_and_partial_is_preserved(
    tmp_path: Path,
) -> None:
    seed = tmp_path / "seed_2026"
    destination = seed / "fold_0"
    destination.mkdir(parents=True)
    partial_token = destination / "tokens.private.npz"
    partial_token.write_bytes(b"legacy partial")

    staged = EXPORT._new_staging_directory(destination)
    metadata = _write_complete_staging(staged)
    preserved = EXPORT._promote_staged_cell(staged, destination)

    assert preserved is not None
    assert preserved.parent == seed
    assert preserved.name.startswith(".fold_0.incomplete-preserved.")
    assert (preserved / "tokens.private.npz").read_bytes() == b"legacy partial"
    assert not staged.exists()
    assert {path.name for path in destination.iterdir()} == {
        "tokens.private.npz",
        "dynamics.private.npz",
        "tokens.private.metadata.json",
    }
    assert (
        json.loads(
            (destination / "tokens.private.metadata.json").read_text(encoding="utf-8")
        )
        == metadata
    )


def test_failed_promotion_restores_legacy_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "seed_2026" / "fold_1"
    destination.mkdir(parents=True)
    old = destination / "dynamics.private.npz"
    old.write_bytes(b"do not delete")
    staged = EXPORT._new_staging_directory(destination)
    _write_complete_staging(staged)
    original_replace = Path.replace

    def fail_staged_rename(path: Path, target: Path) -> Path:
        if path == staged and Path(target) == destination:
            raise OSError("injected promotion failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_staged_rename)
    with pytest.raises(OSError, match="injected promotion failure"):
        EXPORT._promote_staged_cell(staged, destination)

    assert (destination / "dynamics.private.npz").read_bytes() == b"do not delete"
    assert staged.is_dir()
    assert not list(destination.parent.glob(".fold_1.incomplete-preserved.*"))


def test_complete_cell_is_validated_and_never_replaced(tmp_path: Path) -> None:
    destination = tmp_path / "seed_3026" / "fold_4"
    destination.mkdir(parents=True)
    (destination / "tokens.private.metadata.json").write_text(
        '{"status":"COMPLETE"}\n', encoding="utf-8"
    )
    staged = EXPORT._new_staging_directory(destination)
    _write_complete_staging(staged)

    with pytest.raises(FileExistsError, match="refusing to replace complete"):
        EXPORT._promote_staged_cell(staged, destination)
    assert staged.is_dir()
    assert json.loads(
        (destination / "tokens.private.metadata.json").read_text(encoding="utf-8")
    ) == {"status": "COMPLETE"}


def test_only_complete_metadata_marks_a_resumable_cell(tmp_path: Path) -> None:
    marker = tmp_path / "tokens.private.metadata.json"
    assert not RUN_EXPORTS._metadata_declares_complete(marker)
    marker.write_text("not json", encoding="utf-8")
    assert not RUN_EXPORTS._metadata_declares_complete(marker)
    marker.write_text('{"status":"WRITING"}', encoding="utf-8")
    assert not RUN_EXPORTS._metadata_declares_complete(marker)
    marker.write_text('{"status":"COMPLETE"}', encoding="utf-8")
    assert RUN_EXPORTS._metadata_declares_complete(marker)


def test_float64_channel_moments_are_exact_and_validate() -> None:
    values = np.zeros((2, 3, 250, 128), dtype=np.float32)
    values[0, :, :, :] = 1.25
    values[1, :, :, :] = -0.5
    moments = EXPORT._channel_moments(values)

    assert moments["count_per_channel"] == 1500
    np.testing.assert_array_equal(moments["channel_sum"], np.full(128, 562.5))
    np.testing.assert_array_equal(
        moments["channel_sum_squares"], np.full(128, 1359.375)
    )
    RUN_EXPORTS._validate_channel_moments(moments, expected_patients=2, label="target")


def test_channel_moment_validation_rejects_wrong_count_and_nonfinite() -> None:
    values = np.zeros((1, 3, 250, 128), dtype=np.float32)
    moments = EXPORT._channel_moments(values)
    moments["count_per_channel"] = 749
    with pytest.raises(ValueError, match="count differs"):
        RUN_EXPORTS._validate_channel_moments(
            moments, expected_patients=1, label="prediction"
        )
    moments["count_per_channel"] = 750
    moments["channel_sum"][0] = float("nan")
    with pytest.raises(ValueError, match="not 128 finite"):
        RUN_EXPORTS._validate_channel_moments(
            moments, expected_patients=1, label="prediction"
        )


def test_existing_complete_export_validates_provenance_and_metadata_digest(
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(
        checkpoint_root=tmp_path / "checkpoints",
        feature_root=tmp_path / "features",
        workers=6,
        batch_size=4,
    )
    checkpoint, token, marker = RUN_EXPORTS._paths(args, 2026, 2)
    dynamics = token.with_name("dynamics.private.npz")
    for path, content in (
        (checkpoint, b"checkpoint"),
        (token, b"tokens"),
        (dynamics, b"dynamics"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    moments = EXPORT._channel_moments(np.zeros((2, 3, 250, 128), dtype=np.float32))
    metadata = {
        "status": "COMPLETE",
        "arm": "A1_PATCH3",
        "seed_base": 2026,
        "fold": 2,
        "preregistration_lock_sha256": "lock",
        "pcr_loaded": False,
        "condition_in_exported_tokens": False,
        "token_shape": [808, 4, 500, 128],
        "export_batch_size": 4,
        "mask_schedule": (
            "effective_seed_epoch0_logical_batch_index_patient_sha256_transition"
        ),
        "data_loader_workers": 6,
        "multiprocessing_start_method": "spawn",
        "checkpoint_sha256": EXPORT.file_sha256(checkpoint),
        "token_feature_sha256": EXPORT.file_sha256(token),
        "dynamics_sha256": EXPORT.file_sha256(dynamics),
        "test_dynamics_patients": 2,
        "target_channel_moments": moments,
        "prediction_channel_moments": moments,
    }
    marker.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    observed = RUN_EXPORTS._validate_export(args, 2026, 2, "lock")
    assert observed["data_loader_workers"] == 6
    assert observed["multiprocessing_start_method"] == "spawn"
    assert observed["export_metadata_sha256"] == EXPORT.file_sha256(marker)

    token.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="token export SHA-256 mismatch"):
        RUN_EXPORTS._validate_export(args, 2026, 2, "lock")


def test_export_orchestrator_uses_shared_fail_fast_process_registry() -> None:
    source = (ROOT / "scripts" / "run_exports.py").read_text(encoding="utf-8")
    assert "active = ActiveProcesses()" in source
    assert "active.run(command, log)" in source
    assert "active.abort()" in source
    assert "subprocess.run(" not in source

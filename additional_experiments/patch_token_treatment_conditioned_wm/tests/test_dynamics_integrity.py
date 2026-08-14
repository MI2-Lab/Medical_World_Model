from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SEED = 2026
FOLDS = tuple(range(5))
N_PATIENTS = 808
METRIC_FIELDS = (
    "actual_cosine",
    "shuffled_cosine",
    "actual_normalized_mse",
    "shuffled_normalized_mse",
    "target_std",
    "prediction_std",
)
SPATIAL_FIELDS = tuple(
    f"{visit}_{band}_normalized_mse"
    for visit in ("T1", "T2", "T3")
    for band in ("central", "inner_local", "outer_local")
)


@pytest.fixture(scope="module")
def formal_module() -> Any:
    script = EXPERIMENT_ROOT / "scripts" / "evaluate_formal.py"
    name = "patch_token_dynamics_integrity_test"
    spec = importlib.util.spec_from_file_location(name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _fold_table(patient_ids: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        for index, patient_id in enumerate(patient_ids):
            assignment = index % len(FOLDS)
            if assignment == fold:
                split = "test"
            elif assignment == (fold + 1) % len(FOLDS):
                split = "val"
            else:
                split = "train"
            rows.append({"patient_id": patient_id, "fold": fold, "split": split})
    return pd.DataFrame(rows)


def _channel_moments(
    *, patient_count: int, mean: float, standard_deviation: float
) -> dict[str, Any]:
    count = patient_count * 3 * 250
    return {
        "count_per_channel": count,
        "channel_sum": [count * mean] * 128,
        "channel_sum_squares": [count * (standard_deviation**2 + mean**2)] * 128,
    }


def _write_dynamics(path: Path, patient_ids: tuple[str, ...], fold: int) -> None:
    size = len(patient_ids)
    arrays: dict[str, np.ndarray] = {
        "patient_id": np.asarray(patient_ids),
        "fold": np.full(size, fold, dtype=np.int64),
        "actual_cosine": np.full(size, 0.8, dtype=np.float64),
        "shuffled_cosine": np.full(size, 0.2, dtype=np.float64),
        "actual_normalized_mse": np.full(size, 0.5, dtype=np.float64),
        "shuffled_normalized_mse": np.full(size, 1.0, dtype=np.float64),
        # Deliberately unrelated to the exact channel moments. Gate A must not
        # average these per-patient diagnostic columns to estimate token SD.
        "target_std": np.full(size, 97.0, dtype=np.float64),
        "prediction_std": np.full(size, 89.0, dtype=np.float64),
    }
    arrays.update(
        {
            field: np.full(size, 1.0 + index / 10.0, dtype=np.float64)
            for index, field in enumerate(SPATIAL_FIELDS)
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {field: archive[field] for field in archive.files}


def _build_case(
    root: Path,
) -> tuple[Any, dict[tuple[int, int], dict[str, Any]], dict[int, Path]]:
    patient_ids = tuple(f"P{index:04d}" for index in range(N_PATIENTS))
    data = SimpleNamespace(folds=_fold_table(patient_ids), train_only_ids=())
    authenticated: dict[tuple[int, int], dict[str, Any]] = {}
    dynamics_paths: dict[int, Path] = {}
    for fold in FOLDS:
        test_ids = tuple(
            patient_id
            for index, patient_id in enumerate(patient_ids)
            if index % len(FOLDS) == fold
        )
        directory = root / "features" / "a1_formal" / f"seed_{SEED}" / f"fold_{fold}"
        dynamics_path = directory / "dynamics.private.npz"
        metadata_path = directory / "tokens.private.metadata.json"
        _write_dynamics(dynamics_path, test_ids, fold)
        _write_json(
            metadata_path,
            {
                "target_channel_moments": _channel_moments(
                    patient_count=len(test_ids), mean=3.0, standard_deviation=2.0
                ),
                "prediction_channel_moments": _channel_moments(
                    patient_count=len(test_ids), mean=-2.0, standard_deviation=0.5
                ),
            },
        )
        authenticated[(SEED, fold)] = {
            "export": {
                "test_dynamics_patients": len(test_ids),
                "export_metadata_sha256": _sha256(metadata_path),
            }
        }
        dynamics_paths[fold] = dynamics_path
    return data, authenticated, dynamics_paths


def _configure(formal_module: Any, monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setattr(formal_module, "EXPERIMENT_ROOT", root)
    monkeypatch.setattr(formal_module, "SEEDS", (SEED,))
    monkeypatch.setattr(formal_module, "FOLDS", FOLDS)


def test_dynamics_requires_exact_oof_ids_and_uses_authenticated_pooled_moments(
    formal_module: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _configure(formal_module, monkeypatch, tmp_path)
    data, authenticated, _ = _build_case(tmp_path)

    rows, spatial_rows, patients = formal_module._dynamics(data, authenticated)

    assert len(rows) == 1
    assert rows[0]["finite_cell_count"] == 5
    assert rows[0]["n_patients"] == N_PATIENTS
    assert rows[0]["target_std"] == pytest.approx(2.0)
    assert rows[0]["prediction_std"] == pytest.approx(0.5)
    assert rows[0]["token_variance_ratio"] == pytest.approx(0.25)
    assert rows[0]["target_std"] != pytest.approx(float(patients.target_std.mean()))
    assert rows[0]["prediction_std"] != pytest.approx(
        float(patients.prediction_std.mean())
    )
    assert len(spatial_rows) == 9
    assert len(patients) == N_PATIENTS
    assert patients.patient_id.nunique() == N_PATIENTS
    assert set(patients.fold.astype(int)) == set(FOLDS)


@pytest.mark.parametrize("corruption", ["schema", "nonfinite", "patient_id"])
def test_dynamics_rejects_schema_nonfinite_and_patient_identity_drift(
    formal_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    corruption: str,
) -> None:
    _configure(formal_module, monkeypatch, tmp_path)
    data, authenticated, paths = _build_case(tmp_path)
    arrays = _load_npz(paths[0])
    if corruption == "schema":
        arrays["unexpected"] = np.zeros(len(arrays["patient_id"]))
        expected = "schema drifted"
    elif corruption == "nonfinite":
        arrays["actual_cosine"][0] = np.nan
        expected = "identity/content drifted"
    else:
        arrays["patient_id"][0] = "NOT_A_FOLD_PATIENT"
        expected = "identity/content drifted"
    np.savez(paths[0], **arrays)

    with pytest.raises(ValueError, match=expected):
        formal_module._dynamics(data, authenticated)


def test_dynamics_rejects_wrong_fold_artifact(
    formal_module: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _configure(formal_module, monkeypatch, tmp_path)
    data, authenticated, paths = _build_case(tmp_path)
    arrays = _load_npz(paths[2])
    arrays["fold"] = np.full(len(arrays["fold"]), 3, dtype=np.int64)
    np.savez(paths[2], **arrays)

    with pytest.raises(ValueError, match="fold identity/content drifted"):
        formal_module._dynamics(data, authenticated)


def test_dynamics_rejects_unauthenticated_metadata_change(
    formal_module: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _configure(formal_module, monkeypatch, tmp_path)
    data, authenticated, paths = _build_case(tmp_path)
    metadata_path = paths[4].with_name("tokens.private.metadata.json")
    with metadata_path.open("a", encoding="utf-8") as stream:
        stream.write(" \n")

    with pytest.raises(ValueError, match="metadata SHA-256 mismatch"):
        formal_module._dynamics(data, authenticated)


def test_dynamics_rejects_authenticated_invalid_channel_moments(
    formal_module: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _configure(formal_module, monkeypatch, tmp_path)
    data, authenticated, paths = _build_case(tmp_path)
    metadata_path = paths[1].with_name("tokens.private.metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["prediction_channel_moments"]["count_per_channel"] -= 1
    _write_json(metadata_path, metadata)
    authenticated[(SEED, 1)]["export"]["export_metadata_sha256"] = _sha256(
        metadata_path
    )

    with pytest.raises(ValueError, match="channel-moment count differs"):
        formal_module._dynamics(data, authenticated)

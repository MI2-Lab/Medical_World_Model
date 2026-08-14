from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest

from patch_token_wm.evaluation import (
    INCOMPLETE,
    INCOMPLETE_FINAL,
    classify_final,
    evaluate_gate_a,
    evaluate_gate_b,
    evaluate_gate_c,
    evaluate_gate_d,
)


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
CELLS = tuple((seed, fold) for seed in (2026, 3026) for fold in range(5))
SOURCE_A0_LOCK = "a4e1cd2d8b61a7130da2b2eb6dc04e9a5355f44d0a37f4ceccf2fba48b35a9ee"


@pytest.fixture(scope="module")
def formal_module() -> Any:
    script = EXPERIMENT_ROOT / "scripts" / "evaluate_formal.py"
    name = "patch_token_formal_evaluator_test"
    spec = importlib.util.spec_from_file_location(name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_bytes(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    return _write_bytes(path, encoded)


def _write_authenticated_fixture(
    root: Path, *, lock_sha: str = "formal-lock"
) -> dict[str, Any]:
    matrix_rows: list[dict[str, Any]] = []
    export_rows: list[dict[str, Any]] = []
    a0_rows: list[dict[str, Any]] = []
    artifact_paths: dict[tuple[int, int], dict[str, Path]] = {}
    for seed, fold in CELLS:
        a1_checkpoint = (
            root
            / "checkpoints"
            / "a1_formal"
            / f"seed_{seed}"
            / f"fold_{fold}"
            / "selected.pt"
        )
        a1_selection = a1_checkpoint.with_name("selection.json")
        checkpoint_sha = _write_bytes(
            a1_checkpoint, f"A1 checkpoint {seed}/{fold}".encode()
        )
        selection_sha = _write_json(
            a1_selection,
            {
                "arm": "A1_PATCH3",
                "seed_base": seed,
                "fold": fold,
                "selected_epoch": 2,
                "preregistration_lock_sha256": lock_sha,
                "optimization_safety_pass": True,
                "test_data_used": False,
                "pcr_loaded": False,
            },
        )
        a1_feature = (
            root
            / "features"
            / "a1_formal"
            / f"seed_{seed}"
            / f"fold_{fold}"
            / "tokens.private.npz"
        )
        a1_dynamics = a1_feature.with_name("dynamics.private.npz")
        a1_metadata = a1_feature.with_suffix(".metadata.json")
        feature_sha = _write_bytes(a1_feature, f"A1 tokens {seed}/{fold}".encode())
        dynamics_sha = _write_bytes(a1_dynamics, f"A1 dynamics {seed}/{fold}".encode())
        channel_moments = {
            "count_per_channel": 32 * 3 * 250,
            "channel_sum": [0.0] * 128,
            "channel_sum_squares": [1.0] * 128,
        }
        metadata_sha = _write_json(
            a1_metadata,
            {
                "schema_version": 1,
                "status": "COMPLETE",
                "arm": "A1_PATCH3",
                "seed_base": seed,
                "fold": fold,
                "checkpoint_sha256": checkpoint_sha,
                "token_feature_sha256": feature_sha,
                "dynamics_sha256": dynamics_sha,
                "token_shape": [808, 4, 500, 128],
                "test_dynamics_patients": 32,
                "target_order_actual": ["T1", "T2", "T3"],
                "target_order_cyclic_shuffle": ["T2", "T3", "T1"],
                "pcr_loaded": False,
                "condition_in_exported_tokens": False,
                "export_batch_size": 4,
                "mask_schedule": (
                    "effective_seed_epoch0_logical_batch_index_patient_sha256_transition"
                ),
                "data_loader_workers": 2,
                "multiprocessing_start_method": "spawn",
                "target_channel_moments": channel_moments,
                "prediction_channel_moments": channel_moments,
                "preregistration_lock_sha256": lock_sha,
            },
        )

        a0_checkpoint = (
            root
            / "checkpoints"
            / "a0_local3_reference"
            / f"seed_{seed}"
            / f"fold_{fold}"
            / "selected.pt"
        )
        a0_selection = a0_checkpoint.with_name("selection.json")
        a0_checkpoint_sha = _write_bytes(
            a0_checkpoint, f"A0 checkpoint {seed}/{fold}".encode()
        )
        a0_selection_sha = _write_json(
            a0_selection,
            {
                "arm": "LOCAL3",
                "seed_base": seed,
                "fold": fold,
                "selected_epoch": 2,
                "test_data_used": False,
                "pcr_used": False,
            },
        )
        a0_feature = (
            root
            / "features"
            / "a0_local3_reference"
            / f"seed_{seed}"
            / f"fold_{fold}"
            / "response_state.private.npz"
        )
        a0_metadata = a0_feature.with_name("response_state.private.metadata.json")
        a0_feature_sha = _write_bytes(a0_feature, f"A0 response {seed}/{fold}".encode())
        a0_metadata_sha = _write_json(
            a0_metadata,
            {
                "experiment": "local_response_state_multiseed_confirmation",
                "arm": "LOCAL3",
                "seed_base": seed,
                "fold": fold,
                "feature_shape": [808, 4, 192],
                "test_labels_used": False,
                "ftv_head_called": False,
            },
        )
        matrix_rows.append(
            {
                "seed_base": seed,
                "fold": fold,
                "effective_seed": seed + fold,
                "selected_epoch": 2,
                "selection_sha256": selection_sha,
                "selected_checkpoint_sha256": checkpoint_sha,
                "wall_seconds": 1.0,
            }
        )
        export_rows.append(
            {
                "seed_base": seed,
                "fold": fold,
                "token_feature_sha256": feature_sha,
                "dynamics_sha256": dynamics_sha,
                "export_metadata_sha256": metadata_sha,
                "test_dynamics_patients": 32,
            }
        )
        a0_rows.append(
            {
                "seed_base": seed,
                "fold": fold,
                "selected_epoch": 2,
                "selected_checkpoint_sha256": a0_checkpoint_sha,
                "selection_sha256": a0_selection_sha,
                "response_feature_sha256": a0_feature_sha,
                "response_metadata_sha256": a0_metadata_sha,
            }
        )
        artifact_paths[(seed, fold)] = {
            "a1_selection": a1_selection,
            "a1_metadata": a1_metadata,
            "a0_feature": a0_feature,
        }

    matrix = {
        "schema_version": 1,
        "status": "COMPLETE",
        "arm": "A1_PATCH3",
        "run_count": 10,
        "seeds": [2026, 3026],
        "folds": list(range(5)),
        "preregistration_lock_sha256": lock_sha,
        "all_training_pcr_free": True,
        "all_test_blind_selection": True,
        "all_cells_finite_noncollapsed": True,
        "cells": matrix_rows,
    }
    exports = {
        "schema_version": 1,
        "status": "COMPLETE",
        "arm": "A1_PATCH3",
        "run_count": 10,
        "seeds": [2026, 3026],
        "folds": list(range(5)),
        "preregistration_lock_sha256": lock_sha,
        "pcr_loaded": False,
        "all_token_shapes": [[808, 4, 500, 128]],
        "data_loader": {
            "batch_size": 4,
            "workers_per_cell": 2,
            "multiprocessing_start_method": "spawn",
        },
        "cuda_allocator_config": "expandable_segments:True",
        "cells": export_rows,
    }
    a0 = {
        "schema_version": 1,
        "status": "A0_REFERENCE_IMPORTED",
        "experiment": "local_response_state_multiseed_confirmation",
        "arm": "LOCAL3",
        "source_preregistration_lock_sha256": SOURCE_A0_LOCK,
        "cell_count": 10,
        "patient_identifiers_in_manifest": False,
        "private_artifacts_gitignored": True,
        "cells": a0_rows,
    }
    _write_json(root / "metrics" / "formal_matrix_complete.json", matrix)
    _write_json(root / "metrics" / "formal_exports_complete.json", exports)
    _write_json(root / "manifests" / "a0_reference.json", a0)
    return {
        "matrix": matrix,
        "exports": exports,
        "a0": a0,
        "paths": artifact_paths,
        "lock_sha": lock_sha,
    }


def _rewrite_manifests(root: Path, fixture: dict[str, Any]) -> None:
    _write_json(root / "metrics" / "formal_matrix_complete.json", fixture["matrix"])
    _write_json(root / "metrics" / "formal_exports_complete.json", fixture["exports"])
    _write_json(root / "manifests" / "a0_reference.json", fixture["a0"])


def test_formal_input_validator_authenticates_the_exact_ten_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, formal_module: Any
) -> None:
    fixture = _write_authenticated_fixture(tmp_path)
    monkeypatch.setattr(formal_module, "EXPERIMENT_ROOT", tmp_path)

    authenticated = formal_module._validate_formal_inputs(fixture["lock_sha"])

    assert set(authenticated) == set(CELLS)
    assert all(
        set(value) == {"matrix", "export", "a0"} for value in authenticated.values()
    )


@pytest.mark.parametrize("manifest", ["matrix", "exports", "a0"])
def test_formal_input_validator_rejects_missing_or_duplicate_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    formal_module: Any,
    manifest: str,
) -> None:
    fixture = _write_authenticated_fixture(tmp_path)
    rows = fixture[manifest]["cells"]
    rows[-1] = dict(rows[0])
    _rewrite_manifests(tmp_path, fixture)
    monkeypatch.setattr(formal_module, "EXPERIMENT_ROOT", tmp_path)

    with pytest.raises(ValueError, match="duplicate|exact formal"):
        formal_module._validate_formal_inputs(fixture["lock_sha"])


@pytest.mark.parametrize(
    ("mutate", "error_type"),
    [
        (
            lambda fixture: fixture["matrix"]["cells"][0].__setitem__(
                "selection_sha256", "0" * 64
            ),
            ValueError,
        ),
        (
            lambda fixture: fixture["exports"]["cells"][0].__setitem__(
                "token_feature_sha256", "0" * 64
            ),
            ValueError,
        ),
        (
            lambda fixture: fixture["a0"]["cells"][0].__setitem__(
                "response_feature_sha256", "0" * 64
            ),
            ValueError,
        ),
        (
            lambda fixture: fixture["exports"].__setitem__("pcr_loaded", True),
            ValueError,
        ),
        (
            lambda fixture: fixture["paths"][CELLS[0]]["a1_selection"].unlink(),
            FileNotFoundError,
        ),
    ],
    ids=(
        "matrix-selection-hash",
        "export-token-hash",
        "a0-response-hash",
        "export-pcr-firewall",
        "missing-matrix-selection",
    ),
)
def test_formal_input_validator_rejects_hash_and_firewall_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    formal_module: Any,
    mutate: Callable[[dict[str, Any]], None],
    error_type: type[Exception],
) -> None:
    fixture = _write_authenticated_fixture(tmp_path)
    mutate(fixture)
    _rewrite_manifests(tmp_path, fixture)
    monkeypatch.setattr(formal_module, "EXPERIMENT_ROOT", tmp_path)

    with pytest.raises(error_type):
        formal_module._validate_formal_inputs(fixture["lock_sha"])


def test_formal_input_validator_authenticates_export_metadata_firewall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, formal_module: Any
) -> None:
    fixture = _write_authenticated_fixture(tmp_path)
    metadata_path = fixture["paths"][CELLS[0]]["a1_metadata"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["pcr_loaded"] = True
    _write_json(metadata_path, metadata)
    monkeypatch.setattr(formal_module, "EXPERIMENT_ROOT", tmp_path)

    with pytest.raises(ValueError, match="metadata|firewall|pCR"):
        formal_module._validate_formal_inputs(fixture["lock_sha"])


class _TokenPlaceholder:
    shape = (808, 4, 500, 128)

    def copy(self) -> "_TokenPlaceholder":
        return self

    def __getitem__(self, _key: Any) -> "_TokenPlaceholder":
        return self


class _FakeArchive:
    def __init__(self, values: dict[str, Any]) -> None:
        self._values = values
        self.files = list(values)

    def __enter__(self) -> "_FakeArchive":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def __getitem__(self, key: str) -> Any:
        return self._values[key]


def test_load_a1_rejects_a_preexisting_pca_artifact_that_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, formal_module: Any
) -> None:
    seed, fold = CELLS[0]
    token_path = (
        tmp_path
        / "features"
        / "a1_formal"
        / f"seed_{seed}"
        / f"fold_{fold}"
        / "tokens.private.npz"
    )
    token_sha = _write_bytes(token_path, b"synthetic token fixture")
    pca_path = token_path.with_name("pca64.private.npz")
    expected_mean = np.asarray([1.0, 2.0], dtype=np.float32)
    expected_components = np.asarray([[3.0, 4.0]], dtype=np.float32)
    np.savez(
        pca_path,
        mean=expected_mean,
        components=np.asarray([[3.0, 99.0]], dtype=np.float32),
        seed_base=np.asarray(seed, dtype=np.int64),
        fold=np.asarray(fold, dtype=np.int64),
        token_feature_sha256=np.asarray(token_sha),
        labels_used=np.asarray(False),
    )
    patient_ids = np.asarray([f"P{index:04d}" for index in range(808)])
    split = np.asarray(["train"] * 806 + ["val", "test"])
    token_placeholder = _TokenPlaceholder()
    fake_archive = _FakeArchive(
        {
            "patient_id": patient_ids,
            "split": split,
            "tokens": token_placeholder,
            "fractional_weights": np.ones(500, dtype=np.float32),
            "coordinates_xyz_mm": np.zeros((500, 3), dtype=np.float32),
            "seed_base": np.asarray(seed),
            "fold": np.asarray(fold),
        }
    )
    real_load = np.load
    real_isfinite = np.isfinite

    def fake_load(path: Any, *args: Any, **kwargs: Any) -> Any:
        if Path(path) == token_path:
            return fake_archive
        return real_load(path, *args, **kwargs)

    def fake_isfinite(value: Any) -> Any:
        if value is token_placeholder:
            return np.asarray([True])
        return real_isfinite(value)

    class FakeSummarizer:
        def __init__(self, **_kwargs: Any) -> None:
            self.pca_mean_ = expected_mean
            self.pca_components_ = expected_components
            self.provenance = {
                "pca_components_sha256": "components",
                "n_train_patients": 806,
            }

        def fit(self, *_args: Any, **_kwargs: Any) -> "FakeSummarizer":
            return self

        def transform(self, *_args: Any, **_kwargs: Any) -> np.ndarray:
            return np.zeros((808, 4, 192), dtype=np.float32)

    monkeypatch.setattr(formal_module, "EXPERIMENT_ROOT", tmp_path)
    monkeypatch.setattr(formal_module.np, "load", fake_load)
    monkeypatch.setattr(formal_module.np, "isfinite", fake_isfinite)
    monkeypatch.setattr(formal_module, "FoldSafeTokenSummarizer", FakeSummarizer)

    with pytest.raises(ValueError, match="frozen PCA artifact differs"):
        formal_module._load_a1(seed, fold)


def test_main_freezes_all_ten_label_free_pcas_before_clinical_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, formal_module: Any
) -> None:
    class StopAtClinicalLoad(RuntimeError):
        pass

    lock_sha = "ordering-lock"
    (tmp_path / "metrics").mkdir(parents=True)
    _write_json(
        tmp_path / "metrics" / "formal_matrix_complete.json",
        {"status": "COMPLETE", "preregistration_lock_sha256": lock_sha},
    )
    _write_json(
        tmp_path / "metrics" / "formal_exports_complete.json",
        {"status": "COMPLETE", "preregistration_lock_sha256": lock_sha},
    )
    loaded_pca_cells: list[tuple[int, int]] = []

    def load_a0(
        _seed: int, _fold: int
    ) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
        return ("P0000",), np.asarray(["train"]), np.zeros((1, 4, 192))

    def load_a1(
        seed: int, fold: int
    ) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, dict[str, Any]]:
        loaded_pca_cells.append((seed, fold))
        return (
            ("P0000",),
            np.asarray(["train"]),
            np.zeros((1, 4, 192)),
            {
                "pca_components_sha256": f"pca-{seed}-{fold}",
                "pca_artifact_sha256": f"artifact-{seed}-{fold}",
                "token_feature_sha256": f"tokens-{seed}-{fold}",
                "n_train_patients": 1,
            },
        )

    def clinical_table() -> Any:
        assert loaded_pca_cells == list(CELLS)
        raise StopAtClinicalLoad

    monkeypatch.setattr(formal_module, "EXPERIMENT_ROOT", tmp_path)
    monkeypatch.setattr(formal_module, "verify", lambda: {"lock_sha256": lock_sha})
    monkeypatch.setattr(
        formal_module,
        "_validate_formal_inputs",
        lambda _lock_sha: {cell: {} for cell in CELLS},
    )
    monkeypatch.setattr(formal_module, "require_stage_a_go", lambda *_args: object())
    monkeypatch.setattr(
        formal_module, "StageBDataPaths", SimpleNamespace(load=lambda *_args: object())
    )
    monkeypatch.setattr(
        formal_module, "load_stage_b_data", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(formal_module, "_load_a0", load_a0)
    monkeypatch.setattr(formal_module, "_load_a1", load_a1)
    monkeypatch.setattr(formal_module, "_validate_fold_labels", lambda *_args: None)
    monkeypatch.setattr(formal_module, "_run_ftv_cell", lambda **_kwargs: ([], []))
    monkeypatch.setattr(formal_module, "_clinical_table", clinical_table)

    with pytest.raises(StopAtClinicalLoad):
        formal_module.main()


def _passing_dynamics() -> dict[int, dict[str, float]]:
    return {
        seed: {
            "finite_cell_count": 5,
            "target_std": 0.15,
            "prediction_std": 0.12,
            "actual_cosine": 0.50,
            "shuffled_cosine": 0.20,
            "cosine_gain": 0.30,
            "normalized_mse_relative_improvement": 0.20,
        }
        for seed in (2026, 3026)
    }


def _complete_effects() -> dict[int, dict[str, float]]:
    return {
        seed: {
            "static_ftv_spearman_delta": 0.04,
            "delta_ftv_spearman_delta": 0.04,
            "mri_pcr_auroc_delta": 0.04,
        }
        for seed in (2026, 3026)
    }


def _complete_complementarity() -> dict[str, dict[str, Any]]:
    return {
        timing: {"seed_effects": {2026: 0.01, 3026: 0.01}}
        for timing in ("T0-T1", "T0-T2")
    }


def test_gate_a_requires_exactly_ten_finite_cells() -> None:
    complete = _passing_dynamics()
    decision = evaluate_gate_a(complete)
    assert decision.status == "PASS"
    assert decision.evidence["expected_total_cells"] == 10

    only_nine = _passing_dynamics()
    only_nine[3026]["finite_cell_count"] = 4
    failed = evaluate_gate_a(only_nine)
    assert failed.status == "FAIL"
    assert failed.evidence["seeds"]["3026"]["all_cells_finite"] is False

    nonfinite = _passing_dynamics()
    nonfinite[2026]["target_std"] = np.nan
    assert evaluate_gate_a(nonfinite).status == INCOMPLETE


def test_missing_or_nonfinite_gate_evidence_never_gets_a_scientific_label() -> None:
    effects = _complete_effects()
    complete_gates = {
        "A": evaluate_gate_a(_passing_dynamics()),
        "B": evaluate_gate_b(effects),
        "C": evaluate_gate_c(effects),
        "D": evaluate_gate_d(_complete_complementarity()),
    }
    incomplete_a = evaluate_gate_a({2026: _passing_dynamics()[2026]})
    classification = classify_final({**complete_gates, "A": incomplete_a})
    assert classification.status == INCOMPLETE
    assert classification.label == INCOMPLETE_FINAL

    nonfinite_effects = _complete_effects()
    nonfinite_effects[3026]["mri_pcr_auroc_delta"] = np.inf
    incomplete_c = evaluate_gate_c(nonfinite_effects)
    classification = classify_final({**complete_gates, "C": incomplete_c})
    assert incomplete_c.status == INCOMPLETE
    assert classification.status == INCOMPLETE
    assert classification.label == INCOMPLETE_FINAL


def _synthetic_decision_frames() -> (
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
):
    ftv_rows: list[dict[str, Any]] = []
    pcr_rows: list[dict[str, Any]] = []
    dynamics_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for seed in (2026, 3026):
        for task in ("static", "delta"):
            ftv_rows.extend(
                (
                    {
                        "seed_base": seed,
                        "arm": "A0_LOCAL3",
                        "task": task,
                        "endpoint": "macro",
                        "spearman": 0.40,
                    },
                    {
                        "seed_base": seed,
                        "arm": "A1_PATCH3",
                        "task": task,
                        "endpoint": "macro",
                        "spearman": 0.44,
                    },
                )
            )
        for timing in ("T0", "T0-T1", "T0-T2"):
            pcr_rows.extend(
                (
                    {
                        "seed_base": seed,
                        "arm": "A0_LOCAL3",
                        "population": "full_808",
                        "timing": timing,
                        "model": "M",
                        "auroc": 0.60,
                    },
                    {
                        "seed_base": seed,
                        "arm": "A1_PATCH3",
                        "population": "full_808",
                        "timing": timing,
                        "model": "M",
                        "auroc": 0.64,
                    },
                )
            )
        for timing in ("T0-T1", "T0-T2"):
            pcr_rows.extend(
                (
                    {
                        "seed_base": seed,
                        "arm": "A1_PATCH3",
                        "population": "ftv_complete_375",
                        "timing": timing,
                        "model": "C+F",
                        "auroc": 0.60,
                    },
                    {
                        "seed_base": seed,
                        "arm": "A1_PATCH3",
                        "population": "ftv_complete_375",
                        "timing": timing,
                        "model": "C+F+M",
                        "auroc": 0.61,
                    },
                )
            )
            bootstrap_rows.append(
                {
                    "effect": "E5_A1_CplusFplusM_minus_CplusF",
                    "seed_base": seed,
                    "timing": timing,
                    "metric": "auroc",
                    "ci_lower": 0.001,
                }
            )
        dynamics_rows.append(
            {
                "seed_base": seed,
                "finite_cell_count": 5,
                "noncollapsed": True,
                "materially_exceeds_shuffle": True,
                "target_std": 0.15,
                "prediction_std": 0.12,
                "actual_cosine": 0.50,
                "shuffled_cosine": 0.20,
                "cosine_gain": 0.30,
                "normalized_mse_relative_improvement": 0.20,
            }
        )
    return tuple(
        pd.DataFrame(rows)
        for rows in (ftv_rows, pcr_rows, dynamics_rows, bootstrap_rows)
    )  # type: ignore[return-value]


def test_formal_decision_is_incomplete_for_nonfinite_aggregate_evidence(
    formal_module: Any,
) -> None:
    ftv, pcr, dynamics, bootstrap = _synthetic_decision_frames()
    ftv.loc[
        ftv.seed_base.eq(3026) & ftv.arm.eq("A1_PATCH3") & ftv.task.eq("static"),
        "spearman",
    ] = np.nan

    decision = formal_module._decision(
        ftv, pcr, dynamics, bootstrap, formal_cell_count=10
    )

    assert decision["status"] == INCOMPLETE
    assert decision["classification"] == INCOMPLETE_FINAL

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import types

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_evaluation.py"
SPEC = importlib.util.spec_from_file_location("conditional_run_evaluation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_split_indices_accepts_strict_subset_population() -> None:
    rows = []
    identifiers = [f"P{index}" for index in range(8)]
    for fold in range(2):
        for index, patient in enumerate(identifiers):
            rows.append(
                {"patient_id": patient, "fold": fold, "split": ("train", "val", "test")[index % 3]}
            )
    manifest = pd.DataFrame(rows)
    subset = ["P1", "P3", "P6"]
    result = MODULE._split_indices(manifest, subset, 0)
    assert np.array_equal(result["train"], np.asarray([1, 2]))
    assert np.array_equal(result["validation"], np.asarray([0]))
    assert result["test"].size == 0


def test_split_indices_rejects_missing_or_duplicate_requested_ids() -> None:
    manifest = pd.DataFrame(
        {"patient_id": ["P0", "P1", "P2"], "fold": [0, 0, 0], "split": ["train", "val", "test"]}
    )
    for identifiers in (["P0", "missing"], ["P0", "P0"]):
        try:
            MODULE._split_indices(manifest, identifiers, 0)
        except ValueError:
            pass
        else:  # pragma: no cover - direct assertion gives a clearer failure.
            raise AssertionError("invalid requested population was accepted")


def test_atomic_private_csv_is_private_before_pandas_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "predictions" / "oof_predictions.private.csv"
    destination.parent.mkdir(mode=0o755)
    observed: dict[str, int] = {}
    original_to_csv = pd.DataFrame.to_csv

    def checking_to_csv(self: pd.DataFrame, target: object, *args: object, **kwargs: object) -> object:
        assert hasattr(target, "fileno"), "private CSV must be written through a secured descriptor"
        observed["temporary"] = stat.S_IMODE(os.fstat(target.fileno()).st_mode)  # type: ignore[union-attr]
        observed["parent"] = stat.S_IMODE(destination.parent.stat().st_mode)
        return original_to_csv(self, target, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", checking_to_csv)
    MODULE._atomic_csv(pd.DataFrame({"patient_id": ["P001"], "y_true": [1]}), destination, private=True)

    assert observed == {"temporary": 0o600, "parent": 0o700}
    assert stat.S_IMODE(destination.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600


def _write_b0_fixture(
    directory: Path,
    *,
    extra_key: bool = False,
    bad_hash: bool = False,
    bad_split: bool = False,
    bad_preregistration_lock: bool = False,
) -> tuple[Path, Path, list[str], pd.DataFrame]:
    expected_ids = [f"P{index:04d}" for index in range(808)]
    expected_split = np.asarray(
        ["train" if index < 600 else "val" if index < 700 else "test" for index in range(808)]
    )
    folds = pd.DataFrame(
        {"patient_id": expected_ids, "fold": np.zeros(808, dtype=np.int64), "split": expected_split}
    )
    order = np.arange(807, -1, -1)
    state = np.zeros((808, 4, 192), dtype=np.float32)
    state[:, 0, 0] = np.arange(808, dtype=np.float32)
    feature_path = directory / "response_state.private.npz"
    values: dict[str, object] = {
        "patient_id": np.asarray(expected_ids)[order],
        "split": expected_split[order].copy(),
        "response_state": state[order],
        "arm": np.asarray(["LOCAL3"]),
        "seed_base": np.asarray([17], dtype=np.int64),
        "fold": np.asarray([0], dtype=np.int64),
    }
    if bad_split:
        values["split"][0] = "train" if values["split"][0] != "train" else "test"  # type: ignore[index,operator]
    if extra_key:
        values["unexpected"] = np.asarray([1])
    np.savez(feature_path, **values)
    checkpoint_path = directory / "selected.pt"
    checkpoint_path.write_bytes(b"frozen B0 checkpoint")
    selection_path = checkpoint_path.with_name("selection.json")
    selection_path.write_text(
        json.dumps({
            "arm": "LOCAL3",
            "seed_base": 17,
            "fold": 0,
            "selected_epoch": 4,
            "test_data_used": False,
            "pcr_used": False,
            "preregistration_lock_sha256": (
                "0" * 64 if bad_preregistration_lock
                else MODULE.CONFIRMED_LOCAL3_PREREGISTRATION_LOCK_SHA256
            ),
            "data_provenance_sha256": "d" * 64,
        }),
        encoding="utf-8",
    )
    metadata = {
        "arm": "LOCAL3",
        "seed_base": 17,
        "fold": 0,
        "selected_epoch": 4,
        "feature_sha256": "0" * 64 if bad_hash else MODULE.file_sha256(feature_path),
        "checkpoint_sha256": MODULE.file_sha256(checkpoint_path),
        "selection_sha256": MODULE.file_sha256(selection_path),
        "preregistration_lock_sha256": (
            MODULE.CONFIRMED_LOCAL3_PREREGISTRATION_LOCK_SHA256
        ),
        "checkpoint_data_provenance_sha256": "d" * 64,
        "test_labels_used": False,
    }
    feature_path.with_name("response_state.private.metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return feature_path, checkpoint_path, expected_ids, folds


def test_load_b0_state_validates_metadata_and_aligns_frozen_split(tmp_path: Path) -> None:
    feature_path, checkpoint_path, expected_ids, folds = _write_b0_fixture(tmp_path)

    state = MODULE._load_b0_state(
        feature_path, expected_ids, seed=17, fold=0, folds=folds,
        checkpoint_path=checkpoint_path,
    )

    assert state.dtype == np.float32
    assert state.shape == (808, 4, 192)
    assert np.array_equal(state[:, 0, 0], np.arange(808, dtype=np.float32))


@pytest.mark.parametrize(
    "tampering", ["extra_key", "bad_hash", "bad_split", "bad_preregistration_lock"]
)
def test_load_b0_state_rejects_schema_hash_or_split_tampering(
    tmp_path: Path, tampering: str
) -> None:
    feature_path, checkpoint_path, expected_ids, folds = _write_b0_fixture(
        tmp_path,
        extra_key=tampering == "extra_key",
        bad_hash=tampering == "bad_hash",
        bad_split=tampering == "bad_split",
        bad_preregistration_lock=tampering == "bad_preregistration_lock",
    )

    with pytest.raises(ValueError):
        MODULE._load_b0_state(
            feature_path, expected_ids, seed=17, fold=0, folds=folds,
            checkpoint_path=checkpoint_path,
        )


@pytest.mark.parametrize("mutation", ["checkpoint", "selected_epoch"])
def test_load_b0_state_rejects_authoritative_checkpoint_or_epoch_tampering(
    tmp_path: Path, mutation: str
) -> None:
    feature_path, checkpoint_path, expected_ids, folds = _write_b0_fixture(tmp_path)
    if mutation == "checkpoint":
        checkpoint_path.write_bytes(b"replaced checkpoint")
    else:
        metadata_path = feature_path.with_name("response_state.private.metadata.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["selected_epoch"] = 3
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="provenance"):
        MODULE._load_b0_state(
            feature_path, expected_ids, seed=17, fold=0, folds=folds,
            checkpoint_path=checkpoint_path,
        )


@pytest.mark.parametrize("value", [4999, 5001, 5000.0, True])
def test_registered_bootstrap_draws_requires_exact_integer_5000(value: object) -> None:
    arguments = types.SimpleNamespace(bootstrap_draws=value)
    with pytest.raises(ValueError, match="exactly 5000"):
        MODULE._registered_bootstrap_draws(arguments, {"bootstrap": {"draws": 5000}})


def test_registered_bootstrap_draws_accepts_only_registered_default() -> None:
    arguments = types.SimpleNamespace(bootstrap_draws=None)
    assert MODULE._registered_bootstrap_draws(
        arguments, {"bootstrap": {"draws": 5000}}
    ) == 5000


def test_supervised_state_rechecks_feature_binding_and_frozen_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(MODULE, "EXPERIMENT_ROOT", tmp_path)
    identifiers = [f"P{index:04d}" for index in range(808)]
    split = np.asarray(["train"] * 600 + ["val"] * 100 + ["test"] * 108)
    folds = pd.DataFrame({"patient_id": identifiers, "fold": 0, "split": split})
    feature = tmp_path / "features" / "seed_17" / "B2" / "fold_0" / "representation.private.npz"
    feature.parent.mkdir(parents=True)
    np.savez(
        feature,
        patient_id=np.asarray(identifiers),
        split=split,
        representation=np.zeros((808, 4, 64), dtype=np.float32),
        arm=np.asarray("B2"),
        seed=np.asarray(17),
        fold=np.asarray(0),
    )
    selection = tmp_path / "checkpoints" / "seed_17" / "B2" / "fold_0" / "selection.private.json"
    selection.parent.mkdir(parents=True)
    selection.write_text(json.dumps({"feature_sha256": MODULE.file_sha256(feature)}), encoding="utf-8")

    state = MODULE._load_supervised_state(
        feature, identifiers, seed=17, fold=0, arm="B2", folds=folds
    )
    assert state.shape == (808, 4, 64)

    selection.write_text(json.dumps({"feature_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(ValueError, match="selection/checkpoint binding"):
        MODULE._load_supervised_state(
            feature, identifiers, seed=17, fold=0, arm="B2", folds=folds
        )


def test_preflight_visits_exact_supervised_cartesian_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    visited: list[tuple[int, str, int]] = []

    class Cell:
        def __init__(self, *, seed: int, arm: str, fold: int) -> None:
            self.seed = seed
            self.arm = arm
            self.fold = fold

    fake_run_matrix = types.ModuleType("run_matrix")
    fake_run_matrix.Cell = Cell  # type: ignore[attr-defined]
    fake_run_matrix.validate_cell_artifacts = (  # type: ignore[attr-defined]
        lambda cell: visited.append((cell.seed, cell.arm, cell.fold))
    )
    monkeypatch.setitem(sys.modules, "run_matrix", fake_run_matrix)

    MODULE._preflight_supervised_cells()

    expected = {
        (int(seed), str(arm), int(fold))
        for seed in MODULE.SEEDS
        for arm in MODULE.SUPERVISED_ARMS
        for fold in MODULE.FOLDS
    }
    assert len(visited) == 30
    assert set(visited) == expected


def test_main_preflights_before_loading_labels_or_writing_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fail_preflight() -> None:
        calls.append("preflight")
        raise RuntimeError("sentinel")

    monkeypatch.setattr(MODULE, "parse_args", lambda: types.SimpleNamespace())
    monkeypatch.setattr(MODULE, "_preflight_supervised_cells", fail_preflight)
    monkeypatch.setattr(MODULE, "_training_summary", lambda: calls.append("summary"))
    monkeypatch.setattr(MODULE, "_atomic_csv", lambda *args, **kwargs: calls.append("write"))
    monkeypatch.setattr(MODULE, "load_config", lambda: calls.append("labels"))

    with pytest.raises(RuntimeError, match="sentinel"):
        MODULE.main()

    assert calls == ["preflight"]


class _IdentityProjector:
    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def fit(self, values: np.ndarray) -> "_IdentityProjector":
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        return values


def _probe_fixture(*, duplicate_test_half: bool = False) -> tuple[
    dict[tuple[int, str, int], np.ndarray], pd.DataFrame, pd.DataFrame, np.ndarray
]:
    patient_ids = np.asarray([f"P{index:04d}" for index in range(808)])
    clinical = pd.DataFrame(
        {
            "label_hr": np.arange(808) % 2,
            "label_her2": (np.arange(808) // 2) % 2,
            "arm": [f"arm_{index % 13}" for index in range(808)],
        }
    )
    rows: list[dict[str, object]] = []
    for fold in (0, 1):
        for index, patient_id in enumerate(patient_ids):
            test = index < 404 if duplicate_test_half else (index < 404) == (fold == 0)
            rows.append(
                {"patient_id": patient_id, "fold": fold, "split": "test" if test else "train"}
            )
    folds = pd.DataFrame(rows)
    states = {
        (1, arm, fold): np.zeros((808, 4, 64), dtype=np.float32)
        for arm in ("B0", "B2", "B3")
        for fold in (0, 1)
    }
    return states, clinical, folds, patient_ids


def _fake_profile_fit(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    **kwargs: object,
) -> types.SimpleNamespace:
    del x_train, x_validation, y_validation, kwargs
    n_classes = len(np.unique(y_train))
    probabilities = np.full((len(x_test), n_classes), 1.0 / n_classes, dtype=np.float64)
    return types.SimpleNamespace(test_probabilities=probabilities)


def test_profile_probes_require_exact_patient_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    states, clinical, folds, patient_ids = _probe_fixture(duplicate_test_half=True)
    monkeypatch.setattr(MODULE, "SEEDS", (1,))
    monkeypatch.setattr(MODULE, "FOLDS", (0, 1))
    monkeypatch.setattr(MODULE, "PRIMARY_TIMINGS", ("T0",))
    monkeypatch.setattr(MODULE, "PCA", _IdentityProjector)
    monkeypatch.setattr(MODULE, "fit_profile_probe", _fake_profile_fit)

    with pytest.raises(ValueError, match="every full-cohort patient exactly once"):
        MODULE._profile_probes(
            states=states,
            clinical=clinical,
            folds=folds,
            patient_ids=patient_ids,
            config={"downstream": {"c_grid": [1.0]}},
        )


def test_profile_probe_failure_is_not_silently_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    states, clinical, folds, patient_ids = _probe_fixture()
    monkeypatch.setattr(MODULE, "SEEDS", (1,))
    monkeypatch.setattr(MODULE, "FOLDS", (0, 1))
    monkeypatch.setattr(MODULE, "PRIMARY_TIMINGS", ("T0",))
    monkeypatch.setattr(MODULE, "PCA", _IdentityProjector)

    def fail_fit(*args: object, **kwargs: object) -> None:
        raise ValueError("missing class")

    monkeypatch.setattr(MODULE, "fit_profile_probe", fail_fit)
    with pytest.raises(ValueError, match="broad fold skipping is forbidden"):
        MODULE._profile_probes(
            states=states,
            clinical=clinical,
            folds=folds,
            patient_ids=patient_ids,
            config={"downstream": {"c_grid": [1.0]}},
        )

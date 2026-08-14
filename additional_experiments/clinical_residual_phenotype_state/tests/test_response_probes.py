from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from crps.response_probes import run_matched_response_probes, select_ridge
from crps.stageb import (
    StageBDataPaths,
    load_stage_b_data,
    require_stage_a_go,
)


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
CONFIRMED_F0_ROOT = REPO_ROOT / (
    "additional_experiments/local_response_state_multiseed_confirmation/"
    "features/formal_4x8"
)


def test_ridge_contract_uses_train_scaler_lsqr_and_smallest_alpha_tie() -> None:
    train = np.asarray([[1.0], [2.0], [3.0], [4.0]])
    validation = np.asarray([[100.0], [200.0]])
    fit = select_ridge(
        train,
        np.zeros(4),
        validation,
        np.zeros(2),
        (100.0, 0.01, 1.0),
        standardize_target=False,
    )
    assert fit.alpha == 0.01
    assert fit.x_scaler.mean_[0] == pytest.approx(2.5)
    assert fit.model.solver == "lsqr"
    assert fit.model.tol == 1e-8
    assert fit.model.max_iter == 10_000


def test_factorized_probe_returns_pooled_public_metrics_and_private_single_use_oof() -> None:
    rng = np.random.default_rng(31)
    patient_id = np.asarray([f"P{index:02d}" for index in range(15)])
    values = rng.uniform(1.0, 20.0, size=(15, 4))
    records = {
        identity: SimpleNamespace(
            values=values[row],
            measurement_valid=np.ones(4, dtype=bool),
            observable=np.ones(4, dtype=bool),
            grounding_eligible=np.ones(4, dtype=bool),
        )
        for row, identity in enumerate(patient_id)
    }
    z_r = np.repeat(values[..., None], 2, axis=-1).astype(np.float32)
    z_p = rng.normal(size=(15, 4, 3)).astype(np.float32)
    assets = []
    for fold in range(5):
        split = np.full(15, "train", dtype="U5")
        split[3 * fold : 3 * fold + 3] = "test"
        split[3 * ((fold + 1) % 5) : 3 * ((fold + 1) % 5) + 3] = "val"
        assets.append(
            SimpleNamespace(
                patient_id=patient_id,
                split=split,
                z_R=z_r,
                z_P=z_p,
                full=np.concatenate((z_r, z_p), axis=-1),
                arm="F1",
                seed_base=2026,
                fold=fold,
            )
        )

    metrics, private_oof = run_matched_response_probes(
        assets,
        records,
        states=("z_R",),
        expected_measurement_valid_patient_count=15,
    )
    assert "patient_id" not in metrics.columns
    assert set(private_oof["patient_id"]) == set(patient_id)
    assert private_oof["test_predict_call_count"].eq(1).all()
    assert not private_oof["test_used_for_scaler"].any()
    assert not private_oof["test_used_for_alpha_selection"].any()
    assert not private_oof["refit_after_alpha_selection"].any()
    macros = metrics.loc[metrics.endpoint.eq("macro")].set_index("task")
    assert macros.loc["static", "n"] == 60
    assert macros.loc["delta", "n"] == 45
    assert macros.loc["static", "spearman"] > 0.9


def _load_confirmed_local3_assets() -> list[dict[str, np.ndarray]]:
    assets: list[dict[str, np.ndarray]] = []
    for seed in (2026, 3026):
        for fold in range(5):
            path = CONFIRMED_F0_ROOT / (
                f"seed_{seed}/LOCAL3/fold_{fold}/response_state.private.npz"
            )
            with np.load(path, allow_pickle=False) as archive:
                assets.append({key: archive[key].copy() for key in archive.files})
    return assets


def test_confirmed_local3_f0_assets_reproduce_frozen_static_and_delta_macros() -> None:
    representation_config = EXPERIMENT_ROOT / "configs/representation.json"
    data_contract = REPO_ROOT / (
        "additional_experiments/c1b_overlap_eligibility_ftv_stageb/"
        "manifests/stage_b_data_contract.private.json"
    )
    sentinel = REPO_ROOT / (
        "additional_experiments/c1b_overlap_eligibility_ftv_stageb/STAGE_A_GO.json"
    )
    if not (
        CONFIRMED_F0_ROOT.is_dir()
        and data_contract.is_file()
        and sentinel.is_file()
        and all(
            (
                CONFIRMED_F0_ROOT
                / f"seed_{seed}/LOCAL3/fold_{fold}/response_state.private.npz"
            ).is_file()
            for seed in (2026, 3026)
            for fold in range(5)
        )
    ):
        pytest.skip("canonical private confirmed LOCAL3 assets are unavailable")

    config = json.loads(representation_config.read_text(encoding="utf-8"))
    upstream = config["upstream"]
    authorization = require_stage_a_go(sentinel)
    paths = StageBDataPaths.load(
        data_contract,
        upstream["stage_b_data_contract_sha256"],
    )
    data = load_stage_b_data(paths, authorization, verify_cache_files=False)
    metrics, private_oof = run_matched_response_probes(
        _load_confirmed_local3_assets(),
        data.ftv,
    )
    macro = metrics.loc[metrics.endpoint.eq("macro")].set_index(["seed_base", "task"])
    expected = {
        (2026, "static"): 0.5308824999957074,
        (3026, "static"): 0.513248553536036,
        (2026, "delta"): 0.34013190730837795,
        (3026, "delta"): 0.3001941062692001,
    }
    for key, value in expected.items():
        assert float(macro.loc[key, "spearman"]) == pytest.approx(value, abs=1e-12)
    assert len(data.ftv) == 375
    assert macro.loc[(2026, "static"), "n"] == 1500
    assert macro.loc[(2026, "delta"), "n"] == 1125
    assert private_oof["patient_id"].nunique() == 375
    source = (EXPERIMENT_ROOT / "src/crps/response_probes.py").read_text(encoding="utf-8")
    assert "label_pcr" not in source

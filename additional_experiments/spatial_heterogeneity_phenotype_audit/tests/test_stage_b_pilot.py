from __future__ import annotations

import json
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common import canonical_sha256, file_sha256, load_config  # noqa: E402
import stage_b_pilot as stage_b  # noqa: E402
from stage_b_pilot import (  # noqa: E402
    BRANCH_DIM,
    AuthenticatedCacheClosure,
    FEATURE_CHANNELS,
    FOLDS,
    MODEL_ARM,
    SEED_BASE,
    STATE_DIM,
    TABLE8_COLUMNS,
    _validate_selected_checkpoint,
    build_dual_statistic_model,
    configure_canonical_dependencies,
    pair_stage_b_with_stage_a_baseline,
    preflight_payload,
    select_validation_checkpoint,
    unauthorized_table,
    validate_objective,
    validate_canonical_stage_b_cohort,
    validate_stage_b_config,
    validate_stage_b_authorization,
)


PREREGISTRATION_CHAIN = {
    "preregistration_revision": 2,
    "active_preregistration_lock_sha256": "a" * 64,
    "preregistration_amendment_sha256": "b" * 64,
    "original_preregistration_lock_sha256": "c" * 64,
    "original_preregistration_commit": "d" * 40,
    "active_preregistration_commit": "e" * 40,
}


@pytest.fixture(scope="module")
def config():
    return load_config(ROOT / "configs" / "audit.json", verify_inputs=False)


@pytest.fixture(scope="module")
def dependencies(config):
    return configure_canonical_dependencies(config)


def _gate_payload(*, gate_a: bool, gate_c: bool) -> dict:
    return {
        "schema_version": 1,
        "experiment": "spatial_heterogeneity_phenotype_audit",
        "stage": "A",
        "status": "COMPLETE",
        "gates": {
            name: {"passed": passed}
            for name, passed in {
                "A": gate_a,
                "B": False,
                "C": gate_c,
                "D": not (gate_a or gate_c),
            }.items()
        },
        "stage_b_authorized": gate_a or gate_c,
    }


def _write_authorization(
    root: Path,
    config: dict,
    *,
    gate_a: bool,
    gate_c: bool,
    chain: dict | None = None,
) -> tuple[Path, Path]:
    chain = copy.deepcopy(PREREGISTRATION_CHAIN if chain is None else chain)
    config_sha256 = file_sha256(ROOT / "configs" / "audit.json")
    metrics = root / "metrics"
    metrics.mkdir()
    gates_path = metrics / "gates.json"
    gates_path.write_text(
        json.dumps(_gate_payload(gate_a=gate_a, gate_c=gate_c), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    authorized = gate_a or gate_c
    authorization_path = metrics / "stage_b_authorization.json"
    authorization_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "experiment": "spatial_heterogeneity_phenotype_audit",
                "authorization_rule": "Gate A OR Gate C",
                "authorized": authorized,
                "status": (
                    "AUTHORIZED_PENDING_EXECUTION"
                    if authorized
                    else "NOT_RUN_NOT_AUTHORIZED"
                ),
                "reason": "test fixture",
                "gate_a_passed": gate_a,
                "gate_c_passed": gate_c,
                "stage_a_scientific_classification": "MIXED",
                "stage_b_contract": config["stage_b"],
                "config_sha256": config_sha256,
                "preregistration_lock_sha256": chain[
                    "active_preregistration_lock_sha256"
                ],
                "preregistration_chain": chain,
                "contains_patient_level_data": False,
                "stage_a_gates_sha256": file_sha256(gates_path),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    table2_path = metrics / "table2_phenotype_probes.csv"
    table3_path = metrics / "table3_mri_only_pcr.csv"
    table2_path.write_text("seed,arm\n2026,LOCAL3\n", encoding="utf-8")
    table3_path.write_text("seed,arm\n2026,LOCAL3\n", encoding="utf-8")

    def artifact(path: Path) -> dict:
        return {
            "path": str(path.relative_to(root)),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
            "patient_level_private": False,
        }

    (metrics / "run_summary.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "experiment": "spatial_heterogeneity_phenotype_audit",
                "stage": "A",
                "status": "COMPLETE",
                "stage_b_authorized": authorized,
                "config_sha256": config_sha256,
                "preregistration_lock_sha256": chain[
                    "active_preregistration_lock_sha256"
                ],
                "preregistration_chain": chain,
                "artifacts": {
                    "gates": artifact(gates_path),
                    "stage_b_authorization": artifact(authorization_path),
                    "table2": artifact(table2_path),
                    "table3": artifact(table3_path),
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return authorization_path, gates_path


def test_dual_projection_is_exact_row_split_and_baseline_equivalent(dependencies):
    model, report = build_dual_statistic_model(dependencies, SEED_BASE)
    assert report["row_split_exact"] is True
    assert report["std_equals_mean_functional_equivalence"] is True
    assert model.arm == MODEL_ARM
    assert model.response_projection.mean_projection.in_features == FEATURE_CHANNELS
    assert model.response_projection.mean_projection.out_features == BRANCH_DIM
    assert model.response_projection.std_projection.in_features == FEATURE_CHANNELS
    assert model.response_projection.std_projection.out_features == BRANCH_DIM
    assert tuple(model.response_projection.joint_norm.normalized_shape) == (STATE_DIM,)
    assert all(parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(
        parameter.requires_grad for parameter in model.response_projection.parameters()
    )
    assert all(parameter.requires_grad for parameter in model.projector.parameters())
    assert all(parameter.requires_grad for parameter in model.transition.parameters())
    assert all(parameter.requires_grad for parameter in model.ftv_head.parameters())
    assert not any(
        parameter.requires_grad for parameter in model.target_encoder.parameters()
    )
    assert not any(
        parameter.requires_grad
        for parameter in model.target_response_projection.parameters()
    )
    assert not any(
        parameter.requires_grad for parameter in model.target_projector.parameters()
    )

    mean = torch.randn(3, FEATURE_CHANNELS)
    state = model.response_projection(mean, mean)
    assert state.shape == (3, STATE_DIM)
    assert torch.isfinite(state).all()

    online = model.response_projection.mean_projection.weight
    target = model.target_response_projection.mean_projection.weight
    target_before = target.detach().clone()
    with torch.no_grad():
        online.add_(2.0)
    assert torch.equal(target, target_before)
    model.update_target(0.5)
    assert torch.allclose(target, target_before + 1.0, rtol=0.0, atol=2e-7)


def test_objective_is_exact_frozen_ftv_only_dgrs(dependencies):
    objective = dependencies.build_objective("LOCAL3")
    validate_objective(objective)
    assert objective.lambda_ftv == 0.25
    assert objective.sigreg_weight == 0.09


def test_stage_b_probe_import_resolves_local_audit_on_cold_import(config):
    original_path = list(sys.path)
    cached_audit = sys.modules.pop("run_audit", None)
    try:
        audit, data_contracts, expected_audit, expected_contracts, _modeling = (
            stage_b._load_stage_b_probe_dependencies(config)
        )
        assert Path(audit.__file__).resolve() == expected_audit
        assert Path(data_contracts.__file__).resolve() == expected_contracts
        assert (
            audit._append_multiclass_fit.__globals__[
                "_fit_multiclass_logistic_exact_legacy"
            ]
            is audit._fit_multiclass_logistic_exact_legacy
        )
        complementarity = str(expected_contracts.parent)
        assert sys.path.index(str(SCRIPTS)) < sys.path.index(complementarity)
    finally:
        sys.path[:] = original_path
        sys.modules.pop("run_audit", None)
        if cached_audit is not None:
            sys.modules["run_audit"] = cached_audit


def test_full_nested_stage_b_config_is_exact_and_fail_closed(config):
    validate_stage_b_config(config)
    changed = copy.deepcopy(config)
    changed["stage_b"]["training"]["learning_rate"] = 1e-4
    with pytest.raises(ValueError, match="exact prospective implementation"):
        validate_stage_b_config(changed)
    changed = copy.deepcopy(config)
    changed["stage_b"]["forbidden"].remove("new_transformer_module")
    with pytest.raises(ValueError, match="exact prospective implementation"):
        validate_stage_b_config(changed)


@pytest.mark.parametrize("gate_a,gate_c", [(True, False), (False, True), (True, True)])
def test_authorization_is_exactly_gate_a_or_gate_c(
    tmp_path: Path, config, gate_a: bool, gate_c: bool
):
    authorization_path, gates_path = _write_authorization(
        tmp_path, config, gate_a=gate_a, gate_c=gate_c
    )
    authorization = validate_stage_b_authorization(
        config,
        authorization_path,
        gates_path,
        expected_preregistration_chain=PREREGISTRATION_CHAIN,
    )
    assert authorization.authorized is True
    assert authorization.gate_a_passed is gate_a
    assert authorization.gate_c_passed is gate_c
    preflight = preflight_payload(
        authorization,
        "a" * 64,
        PREREGISTRATION_CHAIN,
        config,
    )
    assert preflight["effective_seeds"] == [SEED_BASE + fold for fold in FOLDS]
    assert preflight["stage_a_run_summary_sha256"] == authorization.run_summary_sha256
    assert preflight["training_performed"] is False
    assert preflight["model_or_data_imported"] is False


def test_unauthorized_closure_and_status_table(tmp_path: Path, config):
    authorization_path, gates_path = _write_authorization(
        tmp_path, config, gate_a=False, gate_c=False
    )
    authorization = validate_stage_b_authorization(
        config,
        authorization_path,
        gates_path,
        expected_preregistration_chain=PREREGISTRATION_CHAIN,
    )
    assert authorization.authorized is False
    assert authorization.status == "NOT_RUN_NOT_AUTHORIZED"
    table = unauthorized_table()
    assert tuple(table.columns) == TABLE8_COLUMNS
    assert len(table) == 1
    assert table.loc[0, "status"] == "NOT_RUN_NOT_AUTHORIZED"


def test_unauthorized_main_never_verifies_inputs_or_calls_cache_guard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    config,
):
    events: list[str] = []
    written: list[pd.DataFrame] = []
    original_validate = stage_b.validate_stage_b_config

    def fake_load_config(path, *, verify_inputs=True):
        del path
        events.append(f"load:{verify_inputs}")
        if verify_inputs:
            pytest.fail("unauthorized Stage B touched configured data/cache inputs")
        return copy.deepcopy(config)

    def fake_validate(value):
        events.append("config")
        original_validate(value)

    def fake_lock(value):
        del value
        events.append("lock")
        return {"status": "test"}

    def fake_prereg(value, lock):
        del value, lock
        events.append("prereg")
        return "a" * 64

    def fake_chain(lock):
        del lock
        events.append("chain")
        return copy.deepcopy(PREREGISTRATION_CHAIN)

    def fake_authorization(
        value,
        authorization_path,
        gates_path,
        *,
        expected_preregistration_chain,
    ):
        del value, authorization_path, gates_path
        assert expected_preregistration_chain == PREREGISTRATION_CHAIN
        events.append("authorization")
        return SimpleNamespace(authorized=False)

    def forbidden_cache(value, lock):
        del value, lock
        pytest.fail("unauthorized Stage B called the cache-integrity guard")

    def fake_atomic_csv(frame, path, *, private=False):
        del path, private
        written.append(frame.copy())

    monkeypatch.setattr(
        stage_b,
        "parse_args",
        lambda: SimpleNamespace(
            device="cpu", fold=None, finalize_only=False, preflight=False
        ),
    )
    monkeypatch.setattr(stage_b.os, "umask", lambda value: value)
    monkeypatch.setattr(stage_b, "ROOT", tmp_path)
    monkeypatch.setattr(stage_b, "load_config", fake_load_config)
    monkeypatch.setattr(stage_b, "validate_stage_b_config", fake_validate)
    monkeypatch.setattr(stage_b, "require_preregistration_lock", fake_lock)
    monkeypatch.setattr(stage_b, "require_stage_b_preregistration", fake_prereg)
    monkeypatch.setattr(stage_b, "preregistration_chain", fake_chain)
    monkeypatch.setattr(stage_b, "validate_stage_b_authorization", fake_authorization)
    monkeypatch.setattr(stage_b, "authenticated_cache_evidence", forbidden_cache)
    monkeypatch.setattr(stage_b, "atomic_csv", fake_atomic_csv)

    stage_b.main()

    assert events == [
        "load:False",
        "config",
        "lock",
        "prereg",
        "chain",
        "authorization",
    ]
    assert len(written) == 1
    assert written[0].loc[0, "status"] == "NOT_RUN_NOT_AUTHORIZED"


def test_authorized_main_orders_cache_proof_before_verified_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    config,
):
    events: list[str] = []
    original_validate = stage_b.validate_stage_b_config

    def fake_load_config(path, *, verify_inputs=True):
        del path
        events.append(f"load:{verify_inputs}")
        return copy.deepcopy(config)

    def fake_validate(value):
        events.append("config")
        original_validate(value)

    def fake_lock(value):
        del value
        events.append("lock")
        return {"status": "test"}

    def fake_prereg(value, lock):
        del value, lock
        events.append("prereg")
        return "a" * 64

    def fake_chain(lock):
        del lock
        events.append("chain")
        return copy.deepcopy(PREREGISTRATION_CHAIN)

    def fake_authorization(
        value,
        authorization_path,
        gates_path,
        *,
        expected_preregistration_chain,
    ):
        del value, authorization_path, gates_path
        assert expected_preregistration_chain == PREREGISTRATION_CHAIN
        events.append("authorization")
        return SimpleNamespace(
            authorized=True,
            gate_a_passed=True,
            gate_c_passed=False,
            sha256="b" * 64,
            gates_sha256="c" * 64,
            run_summary_sha256="d" * 64,
        )

    def fake_cache(value, lock):
        del value, lock
        events.append("cache")
        return AuthenticatedCacheClosure(
            evidence={
                "status": "COMPLETE",
                "public_contract_sha256": "e" * 64,
                "private_manifest_sha256": "f" * 64,
            },
            all_patient_ids=frozenset(),
            primary_patient_ids=frozenset(),
            train_only_patient_ids=frozenset(),
        )

    monkeypatch.setattr(
        stage_b,
        "parse_args",
        lambda: SimpleNamespace(
            device="cpu", fold=None, finalize_only=False, preflight=True
        ),
    )
    monkeypatch.setattr(stage_b.os, "umask", lambda value: value)
    monkeypatch.setattr(stage_b, "ROOT", tmp_path)
    monkeypatch.setattr(stage_b, "load_config", fake_load_config)
    monkeypatch.setattr(stage_b, "validate_stage_b_config", fake_validate)
    monkeypatch.setattr(stage_b, "require_preregistration_lock", fake_lock)
    monkeypatch.setattr(stage_b, "require_stage_b_preregistration", fake_prereg)
    monkeypatch.setattr(stage_b, "preregistration_chain", fake_chain)
    monkeypatch.setattr(stage_b, "validate_stage_b_authorization", fake_authorization)
    monkeypatch.setattr(stage_b, "authenticated_cache_evidence", fake_cache)

    stage_b.main()

    assert events == [
        "load:False",
        "config",
        "lock",
        "prereg",
        "chain",
        "authorization",
        "cache",
        "load:True",
    ]


def test_authorization_fails_closed_on_gate_hash_tampering(tmp_path: Path, config):
    authorization_path, gates_path = _write_authorization(
        tmp_path, config, gate_a=True, gate_c=False
    )
    payload = json.loads(gates_path.read_text(encoding="utf-8"))
    payload["gates"]["A"]["passed"] = False
    gates_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="run summary artifact|authorization closure"):
        validate_stage_b_authorization(
            config,
            authorization_path,
            gates_path,
            expected_preregistration_chain=PREREGISTRATION_CHAIN,
        )


def test_authorization_fails_closed_on_incomplete_or_tampered_run_summary(
    tmp_path: Path, config
):
    authorization_path, gates_path = _write_authorization(
        tmp_path, config, gate_a=True, gate_c=False
    )
    summary_path = authorization_path.parent / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["status"] = "RUNNING"
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="COMPLETE formal closure"):
        validate_stage_b_authorization(
            config,
            authorization_path,
            gates_path,
            expected_preregistration_chain=PREREGISTRATION_CHAIN,
        )

    summary["status"] = "COMPLETE"
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
    authorization_payload = json.loads(authorization_path.read_text(encoding="utf-8"))
    authorization_payload["reason"] = "post-summary tampering"
    authorization_path.write_text(
        json.dumps(authorization_payload) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="artifact hash/size/privacy drifted"):
        validate_stage_b_authorization(
            config,
            authorization_path,
            gates_path,
            expected_preregistration_chain=PREREGISTRATION_CHAIN,
        )


def test_authorization_rejects_mixed_amendment_chain_and_config(tmp_path: Path, config):
    authorization_path, gates_path = _write_authorization(
        tmp_path, config, gate_a=True, gate_c=False
    )
    drifted_chain = copy.deepcopy(PREREGISTRATION_CHAIN)
    drifted_chain["preregistration_amendment_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="preregistration_chain"):
        validate_stage_b_authorization(
            config,
            authorization_path,
            gates_path,
            expected_preregistration_chain=drifted_chain,
        )

    summary_path = authorization_path.parent / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["config_sha256"] = "f" * 64
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="config_sha256"):
        validate_stage_b_authorization(
            config,
            authorization_path,
            gates_path,
            expected_preregistration_chain=PREREGISTRATION_CHAIN,
        )


def test_selected_checkpoint_resume_rejects_mixed_amendment_chain(tmp_path: Path):
    authorization = SimpleNamespace(
        sha256="1" * 64,
        gates_sha256="2" * 64,
        run_summary_sha256="3" * 64,
    )
    cache_evidence = {
        "public_contract_sha256": "4" * 64,
        "private_manifest_sha256": "5" * 64,
    }
    chain_fields = {
        "preregistration_lock_sha256": PREREGISTRATION_CHAIN[
            "active_preregistration_lock_sha256"
        ],
        "preregistration_chain": copy.deepcopy(PREREGISTRATION_CHAIN),
    }
    selection = {
        "selected_epoch": 1,
        "authorization_sha256": authorization.sha256,
        "stage_a_gates_sha256": authorization.gates_sha256,
        "stage_a_run_summary_sha256": authorization.run_summary_sha256,
        **chain_fields,
        "cache_integrity_public_contract_sha256": cache_evidence[
            "public_contract_sha256"
        ],
        "cache_integrity_private_manifest_sha256": cache_evidence[
            "private_manifest_sha256"
        ],
        "test_data_used": False,
    }
    selection_path = tmp_path / "selection.private.json"
    selection_path.write_text(
        json.dumps(selection, sort_keys=True) + "\n", encoding="utf-8"
    )
    data_provenance = {"cache_integrity": dict(cache_evidence)}
    payload = {
        "schema_version": 1,
        "experiment": "spatial_heterogeneity_phenotype_audit",
        "stage": "B",
        "arm": MODEL_ARM,
        "seed_base": SEED_BASE,
        "fold": 0,
        "effective_seed": SEED_BASE,
        "epoch": 1,
        "selected": True,
        "authorization_sha256": authorization.sha256,
        "stage_a_gates_sha256": authorization.gates_sha256,
        "stage_a_run_summary_sha256": authorization.run_summary_sha256,
        **chain_fields,
        "cache_integrity_public_contract_sha256": cache_evidence[
            "public_contract_sha256"
        ],
        "cache_integrity_private_manifest_sha256": cache_evidence[
            "private_manifest_sha256"
        ],
        "test_data_used_for_training_or_selection": False,
        "mask_or_oracle_input_used": False,
        "phenotype_pcr_or_delta_supervision_used": False,
        "selection_path": str(selection_path),
        "selection_sha256": file_sha256(selection_path),
        "selection": selection,
        "data_provenance": data_provenance,
        "data_provenance_sha256": canonical_sha256(data_provenance),
    }
    _validate_selected_checkpoint(
        payload,
        fold=0,
        authorization=authorization,
        lock_sha256=PREREGISTRATION_CHAIN["active_preregistration_lock_sha256"],
        preregistration_context=PREREGISTRATION_CHAIN,
        cache_evidence=cache_evidence,
    )

    drifted_chain = copy.deepcopy(PREREGISTRATION_CHAIN)
    drifted_chain["preregistration_amendment_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="preregistration_chain"):
        _validate_selected_checkpoint(
            payload,
            fold=0,
            authorization=authorization,
            lock_sha256=PREREGISTRATION_CHAIN["active_preregistration_lock_sha256"],
            preregistration_context=drifted_chain,
            cache_evidence=cache_evidence,
        )


def test_table8_pairs_exact_stage_a_rows_and_signed_deltas():
    identities = [
        (view, target, "full_808")
        for view in ("T0", "T1", "T2", "T3")
        for target in ("HR", "HER2", "subtype_4class")
    ] + [
        (view, "pCR", population)
        for view in ("T0", "T0-T1", "T0-T2", "T0-T3")
        for population in ("full_808", "ftv_complete_375")
    ]
    dual_rows = []
    baseline_rows = []
    for view, target, population in identities:
        n = 808 if population == "full_808" else 375
        is_pcr = target == "pCR"
        common = {
            "seed": SEED_BASE,
            "view": view,
            "target": target,
            "population": population,
            "n": n,
            "auroc": 0.7,
            "auprc": 0.6,
            "balanced_accuracy": 0.65,
            "brier": 0.2 if is_pcr else float("nan"),
        }
        dual_rows.append(
            {
                **common,
                "arm": MODEL_ARM,
                "variant": "DUAL_MEAN_STD_192",
            }
        )
        baseline_rows.append(
            {
                **common,
                "arm": "LOCAL3",
                "variant": "P1",
                "auroc": 0.66,
                "auprc": 0.57,
                "balanced_accuracy": 0.63,
                "brier": 0.24 if is_pcr else float("nan"),
            }
        )
    dual = pd.DataFrame(dual_rows)
    baseline = pd.DataFrame(baseline_rows)
    paired = pair_stage_b_with_stage_a_baseline(
        dual,
        baseline.loc[baseline["target"].ne("pCR")],
        baseline.loc[baseline["target"].eq("pCR")],
    )
    assert len(paired) == 20
    assert paired["stage_a_baseline_seed"].eq(SEED_BASE).all()
    assert paired["stage_a_baseline_arm"].eq("LOCAL3").all()
    assert paired["stage_a_baseline_variant"].eq("P1").all()
    assert torch.allclose(
        torch.tensor(paired["delta_auroc"].to_numpy()),
        torch.full((20,), 0.04, dtype=torch.float64),
    )
    assert torch.allclose(
        torch.tensor(paired["delta_auprc"].to_numpy()),
        torch.full((20,), 0.03, dtype=torch.float64),
    )
    assert torch.allclose(
        torch.tensor(paired["delta_balanced_accuracy"].to_numpy()),
        torch.full((20,), 0.02, dtype=torch.float64),
    )
    pcr = paired["target"].eq("pCR")
    assert paired.loc[pcr, "brier_improvement"].to_list() == pytest.approx([0.04] * 8)
    assert paired.loc[~pcr, "brier_improvement"].isna().all()

    mismatched = baseline.copy()
    mismatched.loc[mismatched.index[0], "n"] = 807
    with pytest.raises(ValueError, match="population n differs"):
        pair_stage_b_with_stage_a_baseline(
            dual,
            mismatched.loc[mismatched["target"].ne("pCR")],
            mismatched.loc[mismatched["target"].eq("pCR")],
        )


def test_canonical_train_all_is_bound_to_808_plus_exact_139_cache_proof(
    dependencies,
):
    primary = tuple(f"P{index:04d}" for index in range(808))
    train_only = tuple(f"E{index:03d}" for index in range(139))
    fold_rows = []
    for fold in FOLDS:
        for index, patient_id in enumerate(primary):
            split = "train" if index < 600 else "val" if index < 704 else "test"
            fold_rows.append({"patient_id": patient_id, "fold": fold, "split": split})
    bundle = SimpleNamespace(
        folds=pd.DataFrame(fold_rows),
        train_only_ids=train_only,
        c1b_cache={patient_id: object() for patient_id in (*primary, *train_only)},
    )
    closure = AuthenticatedCacheClosure(
        evidence={},
        all_patient_ids=frozenset((*primary, *train_only)),
        primary_patient_ids=frozenset(primary),
        train_only_patient_ids=frozenset(train_only),
    )
    validate_canonical_stage_b_cohort(bundle, dependencies, closure)
    splits = dependencies.make_splits(bundle.folds, 0, bundle.train_only_ids)
    assert len(splits.train_only) == 139
    assert len(splits.train_all) == len(splits.train_primary) + 139
    assert set(splits.train_all).issubset(closure.all_patient_ids)

    drifted = SimpleNamespace(
        folds=bundle.folds,
        train_only_ids=(*train_only[:-1], "UNAUTHENTICATED"),
        c1b_cache=bundle.c1b_cache,
    )
    with pytest.raises(ValueError, match="authenticated 139 caches"):
        validate_canonical_stage_b_cohort(drifted, dependencies, closure)


def test_selection_uses_earliest_minimum_total_objective():
    rows = [
        {
            "epoch": 1,
            "val_total_objective": 1.4,
            "val_state_loss": 1.0,
            "val_ftv_loss": 1.6,
            "val_representation_std": 0.07,
            "finite": True,
        },
        {
            "epoch": 2,
            "val_total_objective": 1.1,
            "val_state_loss": 1.0,
            "val_ftv_loss": 0.4,
            "val_representation_std": 0.04,
            "finite": True,
        },
        {
            "epoch": 3,
            "val_total_objective": 1.2,
            "val_state_loss": 1.1,
            "val_ftv_loss": 0.4,
            "val_representation_std": 0.08,
            "finite": True,
        },
        {
            "epoch": 4,
            "val_total_objective": 1.2,
            "val_state_loss": 1.0,
            "val_ftv_loss": 0.8,
            "val_representation_std": 0.09,
            "finite": True,
        },
    ]
    selection = select_validation_checkpoint(rows, min_representation_std=0.05)
    assert selection["selected_epoch"] == 3
    assert selection["selected_validation_total_objective"] == 1.2
    assert selection["test_data_used"] is False


def test_selection_rejects_all_collapsed_or_nonfinite_epochs():
    rows = [
        {
            "epoch": 1,
            "val_total_objective": 1.0,
            "val_state_loss": 1.0,
            "val_ftv_loss": 0.0,
            "val_representation_std": 0.01,
            "finite": True,
        },
        {
            "epoch": 2,
            "val_total_objective": float("nan"),
            "val_state_loss": 1.0,
            "val_ftv_loss": 0.0,
            "val_representation_std": 0.2,
            "finite": False,
        },
    ]
    with pytest.raises(RuntimeError, match="no finite non-collapsed"):
        select_validation_checkpoint(rows, min_representation_std=0.05)

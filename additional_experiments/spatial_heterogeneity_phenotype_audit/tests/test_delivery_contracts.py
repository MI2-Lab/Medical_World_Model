from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import common as common_module  # noqa: E402
from common import file_sha256  # noqa: E402
import freeze_preregistration as freeze  # noqa: E402
import generate_figures as figures  # noqa: E402
from generate_figures import pair_matched_oracle_deltas  # noqa: E402
from generate_report import (  # noqa: E402
    STAGE_A_AUTHORIZATION_KEYS,
    STAGE_A_RUN_SUMMARY_KEYS,
    _paired_delta,
    _stage_b_text,
    gate_c_support_summary,
    two_seed_endpoint_support,
)
import run_feature_matrix as feature_matrix  # noqa: E402
from run_feature_matrix import (  # noqa: E402
    COMPLETION_KEYS,
    FEATURE_SHAPE_ZYX,
    ORACLE_REGIONS,
    REPRESENTATIVE_SELECTION_RULE,
    publish_completion_marker,
    validate_completion_marker,
    validate_representative_asset,
)
import validate_results as validator  # noqa: E402


def _synthetic_run_summary(
    root: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    (root / "configs").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "metrics").mkdir()
    config_path = root / "configs" / "audit.json"
    config_path.write_text('{"synthetic":true}\n', encoding="utf-8")
    for name in ("run_audit.py", "common.py"):
        (root / "scripts" / name).write_text(f"# synthetic {name}\n", encoding="utf-8")

    selected_cells: dict[str, object] = {}
    feature_hashes: dict[str, str] = {}
    feature_records: list[dict[str, object]] = []
    for seed in (2026, 3026):
        for arm in ("LOCAL0", "LOCAL3"):
            for fold in range(5):
                key = f"seed_{seed}/{arm}/fold_{fold}"
                feature = root / "features" / key / "spatial_statistics.private.npz"
                feature.parent.mkdir(parents=True)
                feature.write_bytes(f"synthetic-{key}".encode("ascii"))
                digest = file_sha256(feature)
                selected_cells[key] = {}
                feature_hashes[key] = digest
                feature_records.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "fold": fold,
                        "feature_sha256": digest,
                        "max_parity_abs": 0.0,
                    }
                )

    artifact_records: dict[str, object] = {}
    for index, (name, relative) in enumerate(
        validator.RUN_SUMMARY_ARTIFACT_PATHS.items()
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"synthetic artifact {index}\n", encoding="utf-8")
        private = name in validator.RUN_SUMMARY_PRIVATE_ARTIFACTS
        path.chmod(0o600 if private else 0o644)
        artifact_records[name] = {
            "path": relative,
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
            "patient_level_private": private,
        }

    lock_path = root / "PREREGISTRATION_LOCK.json"
    lock = {
        "selected_cells": selected_cells,
        "implementation_sha256": {
            "scripts/run_audit.py": file_sha256(root / "scripts" / "run_audit.py"),
            "scripts/common.py": file_sha256(root / "scripts" / "common.py"),
        },
    }
    lock_path.write_text(json.dumps(lock, sort_keys=True), encoding="utf-8")
    config = {
        "branch": "feature/spatial-heterogeneity-phenotype-audit",
        "upstream_code": {
            "complementarity_data_contracts_sha256": "a" * 64,
            "complementarity_modeling_sha256": "b" * 64,
        },
    }
    gates = {
        "scientific_classification": "MIXED",
        "stage_b_authorized": False,
    }
    feature_summary = {"cells": feature_records}
    summary: dict[str, object] = {
        "schema_version": 2,
        "experiment": "spatial_heterogeneity_phenotype_audit",
        "stage": "A",
        "status": "COMPLETE",
        "branch": config["branch"],
        "n_feature_assets": 20,
        "n_full_patients": 808,
        "n_ftv_complete_patients": 375,
        "scientific_classification": "MIXED",
        "stage_b_authorized": False,
        "elapsed_seconds": 1.25,
        "config_sha256": file_sha256(config_path),
        "preregistration_lock_sha256": file_sha256(lock_path),
        "preregistration_chain": {
            "preregistration_revision": 2,
            "active_preregistration_lock_sha256": file_sha256(lock_path),
            "preregistration_amendment_sha256": "c" * 64,
            "original_preregistration_lock_sha256": "d" * 64,
            "original_preregistration_commit": "e" * 40,
            "active_preregistration_commit": "f" * 40,
        },
        "feature_asset_sha256": feature_hashes,
        "reused_implementation_sha256": {
            "data_contracts": "a" * 64,
            "modeling": "b" * 64,
        },
        "runtime_implementation_sha256": dict(lock["implementation_sha256"]),
        "artifacts": artifact_records,
        "public_outputs_contain_patient_level_data": False,
    }
    (root / "metrics" / "run_summary.json").write_text(
        json.dumps(summary, sort_keys=True), encoding="utf-8"
    )
    return config, lock, gates, feature_summary


def test_oracle_delta_is_pair_matched_and_rejects_coverage_drift() -> None:
    rows = [
        {
            "seed": seed,
            "arm": "LOCAL3",
            "view": "T0",
            "target": "HER2",
            "population": "oracle_pair_CORE",
            "variant": variant,
            "n": 101,
            "auroc": value,
        }
        for seed in (2026, 3026)
        for variant, value in (("CORE", 0.72), ("FIXED_P3", 0.64))
    ]
    frame = pd.DataFrame(rows)
    paired = pair_matched_oracle_deltas(frame)
    assert len(paired) == 2
    np.testing.assert_allclose(paired["delta_auroc"], 0.08)
    frame.loc[frame["variant"].eq("FIXED_P3"), "n"] = 100
    with pytest.raises(ValueError, match="different coverage"):
        pair_matched_oracle_deltas(frame)


def test_gate_a_support_is_endpoint_specific() -> None:
    rows = []
    for seed in (2026, 3026):
        for target, gain in (("HER2", 0.04), ("subtype_4class", 0.02)):
            for variant, value in (("P1", 0.60), ("P3", 0.60 + gain)):
                rows.append(
                    {
                        "seed": seed,
                        "arm": "LOCAL3",
                        "view": "T0",
                        "target": target,
                        "variant": variant,
                        "population": "full_808",
                        "auroc": value,
                    }
                )
    support = two_seed_endpoint_support(
        pd.DataFrame(rows),
        targets=("HER2", "subtype_4class"),
        population="full_808",
        expected_seeds=(2026, 3026),
        threshold=0.03,
    )
    assert len(support["HER2"]) == 1
    assert support["subtype_4class"] == []


def test_stage_b_report_summarizes_actual_paired_deltas() -> None:
    rows = []
    for view in ("T0", "T1", "T2", "T3"):
        for target in ("HR", "HER2", "subtype_4class"):
            rows.append(
                {
                    "status": "COMPLETE",
                    "seed": 2026,
                    "arm": "RESPONSE_PHENOTYPE_DUAL_STATISTIC_STATE",
                    "view": view,
                    "target": target,
                    "variant": "DUAL_MEAN_STD_192",
                    "population": "full_808",
                    "n": 808,
                    "delta_auroc": 0.05,
                    "brier_improvement": np.nan,
                }
            )
    for view in ("T0", "T0-T1", "T0-T2", "T0-T3"):
        for population, n, delta, brier in (
            ("full_808", 808, 0.50, 0.40),
            ("ftv_complete_375", 375, -0.01, 0.02),
        ):
            rows.append(
                {
                    "status": "COMPLETE",
                    "seed": 2026,
                    "arm": "RESPONSE_PHENOTYPE_DUAL_STATISTIC_STATE",
                    "view": view,
                    "target": "pCR",
                    "variant": "DUAL_MEAN_STD_192",
                    "population": population,
                    "n": n,
                    "delta_auroc": delta,
                    "brier_improvement": brier,
                }
            )
    table = pd.DataFrame(rows)
    text = _stage_b_text(
        table,
        {"authorized": True},
        primary_pcr_population="ftv_complete_375",
    )
    assert "phenotype ΔAUROC" in text
    assert "pCR primary population `ftv_complete_375`" in text
    assert "-0.010" in text
    assert "+0.500" not in text
    assert "Brier" in text

    duplicate = pd.concat((table, table.iloc[[0]]), ignore_index=True)
    with pytest.raises(ValueError, match="grid/identity"):
        _stage_b_text(
            duplicate,
            {"authorized": True},
            primary_pcr_population="ftv_complete_375",
        )
    nonfinite = table.copy()
    nonfinite.loc[
        nonfinite["target"].eq("pCR") & nonfinite["population"].eq("full_808"),
        "delta_auroc",
    ] = np.nan
    with pytest.raises(ValueError, match="non-finite/invalid"):
        _stage_b_text(
            nonfinite,
            {"authorized": True},
            primary_pcr_population="ftv_complete_375",
        )


def test_q2_pcr_summary_requires_exact_matched_375_pairs() -> None:
    rows = []
    for seed in (2026, 3026):
        for arm in ("LOCAL0", "LOCAL3"):
            for view in ("T0", "T0-T1", "T0-T2", "T0-T3"):
                for variant, auroc in (("P1", 0.60), ("P2", 0.59)):
                    rows.append(
                        {
                            "seed": seed,
                            "arm": arm,
                            "view": view,
                            "target": "pCR",
                            "variant": variant,
                            "population": "ftv_complete_375",
                            "n": 375,
                            "auroc": auroc,
                        }
                    )
    frame = pd.DataFrame(rows)
    summary = _paired_delta(
        frame,
        column="variant",
        comparison="P2",
        reference="P1",
        expected_count=16,
        expected_n=375,
    )
    assert summary["count"] == 16
    assert summary["mean"] == pytest.approx(-0.01)

    with pytest.raises(ValueError, match="non-unique"):
        _paired_delta(
            pd.concat((frame, frame.iloc[[0]]), ignore_index=True),
            column="variant",
            comparison="P2",
            reference="P1",
            expected_count=16,
            expected_n=375,
        )
    with pytest.raises(ValueError, match="coverage"):
        _paired_delta(
            frame.assign(n=374),
            column="variant",
            comparison="P2",
            reference="P1",
            expected_count=16,
            expected_n=375,
        )


def test_gate_c_report_support_is_endpoint_specific_and_exact() -> None:
    gate = {
        "passed": True,
        "minimum_matched_auroc_gain_each_seed": 0.03,
        "supporting_comparisons": [
            {
                "arm": "LOCAL0",
                "comparison": "PERI20",
                "passed": True,
                "population": "oracle_pair_PERI20",
                "reference": "FIXED_P3",
                "seed_deltas": {
                    "2026": 0.03567753001715257,
                    "3026": 0.033276157804459694,
                },
                "target": "pCR",
                "view": "T0-T1",
            }
        ],
    }
    summary = gate_c_support_summary(gate, expected_seeds=(2026, 3026))
    assert summary["count"] == 1
    assert summary["phenotype_count"] == 0
    assert summary["pcr_count"] == 1
    assert "comparison=PERI20 vs reference=FIXED_P3" in summary["text"]
    assert "population=oracle_pair_PERI20" in summary["text"]
    assert "seed2026=+0.03567753001715257" in summary["text"]


def test_figure7_stratifies_both_registered_populations() -> None:
    rows = []
    for seed in (2026, 3026):
        for arm in ("LOCAL0", "LOCAL3"):
            for view_index, view in enumerate(("T0->T1", "T1->T2", "T2->T3")):
                for variant_index, variant in enumerate(
                    ("DELTA_MEAN", "DELTA_STD", "P3_PLUS_DELTA")
                ):
                    for population, n, baseline in (
                        ("full_808", 808, 0.80),
                        ("ftv_complete_375", 375, 0.55),
                    ):
                        rows.append(
                            {
                                "seed": seed,
                                "arm": arm,
                                "view": view,
                                "target": "pCR",
                                "variant": variant,
                                "population": population,
                                "n": n,
                                "auroc": baseline
                                + 0.01 * view_index
                                + 0.001 * variant_index,
                            }
                        )
    frame = pd.DataFrame(rows)
    figure = figures.longitudinal_figure(frame)
    lines = [line for line in figure.axes[0].get_lines() if " · " in line.get_label()]
    assert len(lines) == 6
    by_label = {line.get_label(): line for line in lines}
    for variant in ("DELTA_MEAN", "DELTA_STD", "P3_PLUS_DELTA"):
        full = by_label[f"{variant} · full_808"]
        matched = by_label[f"{variant} · ftv_complete_375"]
        assert np.all(full.get_ydata() > matched.get_ydata())
        assert full.get_linestyle() != matched.get_linestyle()
    figures.plt.close(figure)

    with pytest.raises(ValueError, match="exact registered grid"):
        figures.longitudinal_figure(frame.iloc[:-1])
    with pytest.raises(ValueError, match="repeats"):
        figures.longitudinal_figure(
            pd.concat((frame, frame.iloc[[0]]), ignore_index=True)
        )


def test_stage_a_provenance_schemas_are_synchronized() -> None:
    chain = {
        "preregistration_revision": 2,
        "active_preregistration_lock_sha256": "a" * 64,
        "preregistration_amendment_sha256": "b" * 64,
        "original_preregistration_lock_sha256": "c" * 64,
        "original_preregistration_commit": "d" * 40,
        "active_preregistration_commit": "e" * 40,
    }
    gates = {
        "gates": {"A": {"passed": False}, "C": {"passed": True}},
        "scientific_classification": "PHENOTYPE_IS_SPATIALLY_LOCALIZED",
    }
    authorization = validator.stage_b_authorization(
        {"stage_b": {"enabled": True}}, gates, chain, "f" * 64
    )
    authorization["stage_a_gates_sha256"] = "0" * 64
    assert set(authorization) == set(STAGE_A_AUTHORIZATION_KEYS)
    assert set(STAGE_A_RUN_SUMMARY_KEYS) == set(validator.RUN_SUMMARY_KEYS)


def test_representative_npz_has_strict_schema_dtype_and_hash(tmp_path: Path) -> None:
    shape = FEATURE_SHAPE_ZYX
    path = tmp_path / "representative.private.npz"
    np.savez(
        path,
        activation_mean_abs=np.ones(shape, dtype=np.float32),
        activation_channel_std=np.ones(shape, dtype=np.float32),
        local_weight=np.ones(shape, dtype=np.float32),
        region_weight=np.ones((4, *shape), dtype=np.float32),
        regions=np.asarray(ORACLE_REGIONS),
        selection_rule=np.asarray(REPRESENTATIVE_SELECTION_RULE),
    )
    path.chmod(0o600)
    validate_representative_asset(path, expected_sha256=file_sha256(path))
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: np.asarray(archive[name]) for name in archive.files}
    payload["activation_mean_abs"] = payload["activation_mean_abs"].astype(np.float64)
    np.savez(path, **payload)
    path.chmod(0o600)
    with pytest.raises(ValueError, match="shape/dtype/value"):
        validate_representative_asset(path)


def test_privacy_scan_is_a_closed_delivery_allowlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert "PREREGISTRATION_IMPLEMENTATION_ERRATUM_2.json" in (
        validator.DELIVERY_ALLOWED_PATHS
    )
    assert "PREREGISTRATION_IMPLEMENTATION_ERRATUM_3.json" in (
        validator.DELIVERY_ALLOWED_PATHS
    )
    assert "tests/test_multiclass_compat.py" in validator.DELIVERY_ALLOWED_PATHS
    monkeypatch.setattr(validator, "ROOT", tmp_path)
    undeclared = tmp_path / "figures" / "raw_patient.png"
    undeclared.parent.mkdir()
    undeclared.write_bytes(b"raw image bytes")
    with pytest.raises(ValueError, match="undeclared file"):
        validator._privacy_scan([undeclared])

    public_csv = tmp_path / "metrics" / "hyperparameter_selections.csv"
    public_csv.parent.mkdir()
    public_csv.write_text("subject-id,value\npatient-001,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="patient identifier column"):
        validator._privacy_scan([public_csv])


def test_amendment_schema_is_exact_and_patient_free(tmp_path: Path) -> None:
    source = ROOT / "PREREGISTRATION_AMENDMENT.json"
    amendment = json.loads(source.read_text(encoding="utf-8"))
    path = tmp_path / source.name
    path.write_text(json.dumps(amendment), encoding="utf-8")
    observed = common_module.load_preregistration_amendment(path)
    assert (
        observed["geometry_qc"]["all_four_post_local_core_valid_patient_count"] == 373
    )
    assert observed["contains_patient_identifiers"] is False

    amendment["geometry_qc"]["representative_candidate_count_after"] = 375
    path.write_text(json.dumps(amendment), encoding="utf-8")
    with pytest.raises(ValueError, match="geometry QC"):
        common_module.load_preregistration_amendment(path)


def test_implementation_erratum_schema_is_exact_and_patient_free(
    tmp_path: Path,
) -> None:
    source = ROOT / "PREREGISTRATION_IMPLEMENTATION_ERRATUM.json"
    erratum = json.loads(source.read_text(encoding="utf-8"))
    path = tmp_path / source.name
    path.write_bytes(source.read_bytes())
    observed = common_module.load_preregistration_implementation_erratum(path)
    assert len(observed["discarded_artifact_sha256"]) == 65
    assert (
        observed["pre_erratum_execution"]["discarded_artifact_total_bytes"] == 307933315
    )
    assert observed["contains_patient_identifiers"] is False

    erratum["contract_scope"]["scientific_contract_changed"] = True
    path.write_text(json.dumps(erratum), encoding="utf-8")
    with pytest.raises(ValueError, match="contract scope"):
        common_module.load_preregistration_implementation_erratum(path)


def test_implementation_erratum_2_schema_is_exact_and_patient_free(
    tmp_path: Path,
) -> None:
    source = ROOT / "PREREGISTRATION_IMPLEMENTATION_ERRATUM_2.json"
    erratum = json.loads(source.read_text(encoding="utf-8"))
    path = tmp_path / source.name
    path.write_bytes(source.read_bytes())
    observed = common_module.load_preregistration_implementation_erratum_2(path)
    execution = observed["pre_erratum_execution"]
    assert len(observed["discarded_artifact_sha256"]) == 67
    assert execution["discarded_artifact_total_bytes"] == 307938585
    assert execution["completed_binary_probe_tasks_in_memory"] == 2
    assert execution["first_multiclass_candidate_fit_succeeded"] is False
    assert observed["contains_patient_identifiers"] is False
    assert observed["contract_scope"]["convergence_warning_remains_fail_closed"] is True

    erratum["pre_erratum_execution"][
        "label_derived_public_metric_artifact_created"
    ] = True
    path.write_text(json.dumps(erratum), encoding="utf-8")
    with pytest.raises(ValueError, match="execution ledger"):
        common_module.load_preregistration_implementation_erratum_2(path)


def test_implementation_erratum_3_schema_is_exact_and_patient_free(
    tmp_path: Path,
) -> None:
    source = ROOT / "PREREGISTRATION_IMPLEMENTATION_ERRATUM_3.json"
    erratum = json.loads(source.read_text(encoding="utf-8"))
    path = tmp_path / source.name
    path.write_bytes(source.read_bytes())
    observed = common_module.load_preregistration_implementation_erratum_3(path)
    execution = observed["pre_erratum_execution"]
    assert len(observed["discarded_artifact_sha256"]) == 83
    assert execution["discarded_artifact_total_bytes"] == 409345148
    assert execution["private_oof_prediction_row_count"] == 691412
    assert execution["default_parser_gate_json_difference_count"] == 26
    assert (
        execution["default_parser_maximum_gate_absolute_difference"]
        == 1.1102230246251565e-16
    )
    assert execution["stage_b_epoch_execution_entered"] is True
    assert execution["stage_b_interrupted_during_epoch_1_before_completion"] is True
    assert execution["stage_b_completed_epoch_count"] == 0
    assert execution["presentation_contract_gap_count"] == 4
    assert observed["contains_patient_identifiers"] is False

    erratum["contract_scope"]["scientific_contract_changed"] = True
    path.write_text(json.dumps(erratum), encoding="utf-8")
    with pytest.raises(ValueError, match="contract scope"):
        common_module.load_preregistration_implementation_erratum_3(path)


def test_validator_public_table_loader_round_trips_decimal(tmp_path: Path) -> None:
    path = tmp_path / "metric.csv"
    path.write_text("value\n0.03567753001715257\n", encoding="utf-8")
    expected = float("0.03567753001715257")
    observed = float(validator._read_public_table(path).loc[0, "value"])
    default = float(pd.read_csv(path).loc[0, "value"])
    assert observed.hex() == expected.hex()
    assert default.hex() != expected.hex()


def test_implementation_refreeze_preserves_prior_scientific_contract() -> None:
    prior = common_module.historical_json(
        common_module.PRIOR_AMENDED_PREREGISTRATION_COMMIT,
        "PREREGISTRATION_LOCK.json",
    )
    active = copy.deepcopy(prior)
    active["implementation_sha256"] = {
        name: "f" * 64 for name in prior["implementation_sha256"]
    }
    common_module.require_preserved_prior_lock_contract(active, prior)
    common_module.require_implementation_erratum_plan_disclosure(prior)

    prior_implementation = common_module.historical_json(
        common_module.PRIOR_IMPLEMENTATION_REFREEZE_COMMIT,
        "PREREGISTRATION_LOCK.json",
    )
    common_module.require_implementation_erratum_2_plan_disclosure(prior_implementation)

    prior_compatibility = common_module.historical_json(
        common_module.PRIOR_COMPATIBILITY_REFREEZE_COMMIT,
        "PREREGISTRATION_LOCK.json",
    )
    common_module.require_implementation_erratum_3_plan_disclosure(prior_compatibility)

    active["config_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="config_sha256"):
        common_module.require_preserved_prior_lock_contract(active, prior)


def test_preregistration_chain_authenticates_all_five_committed_locks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    experiment = repo / "audit"
    experiment.mkdir(parents=True)

    def git(*arguments: str) -> str:
        return subprocess.check_output(["git", *arguments], cwd=repo, text=True).strip()

    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    git("config", "user.email", "audit@example.invalid")
    git("config", "user.name", "Audit Test")
    original_lock_path = experiment / "PREREGISTRATION_LOCK.json"
    original_lock_path.write_text('{"revision":1}\n', encoding="utf-8")
    git("add", ".")
    git("commit", "-q", "-m", "original preregistration")
    original_commit = git("rev-parse", "HEAD")
    original_lock_sha256 = file_sha256(original_lock_path)

    amendment = json.loads(
        (ROOT / "PREREGISTRATION_AMENDMENT.json").read_text(encoding="utf-8")
    )
    amendment["original_preregistration_commit"] = original_commit
    amendment["original_preregistration_lock_sha256"] = original_lock_sha256
    amendment_path = experiment / "PREREGISTRATION_AMENDMENT.json"
    amendment_path.write_text(json.dumps(amendment, sort_keys=True), encoding="utf-8")
    amended_lock = {
        "schema_version": 3,
        "preregistration_revision": 2,
        "status": common_module.AMENDED_LOCK_STATUS,
        "amendment_sha256": file_sha256(amendment_path),
        "superseded_preregistration_commit": original_commit,
        "superseded_preregistration_lock_sha256": original_lock_sha256,
        "implementation_sha256": {},
    }
    original_lock_path.write_text(
        json.dumps(amended_lock, sort_keys=True), encoding="utf-8"
    )
    git("add", ".")
    git("commit", "-q", "-m", "amended preregistration")
    prior_amended_commit = git("rev-parse", "HEAD")
    prior_amended_lock_sha256 = file_sha256(original_lock_path)

    erratum = json.loads(
        (ROOT / "PREREGISTRATION_IMPLEMENTATION_ERRATUM.json").read_text(
            encoding="utf-8"
        )
    )
    erratum["prior_amended_preregistration_commit"] = prior_amended_commit
    erratum["prior_amended_preregistration_lock_sha256"] = prior_amended_lock_sha256
    erratum_path = experiment / "PREREGISTRATION_IMPLEMENTATION_ERRATUM.json"
    erratum_path.write_text(json.dumps(erratum, sort_keys=True), encoding="utf-8")
    implementation_lock = {
        **amended_lock,
        "schema_version": 4,
        "status": common_module.IMPLEMENTATION_ERRATUM_LOCK_STATUS,
        "implementation_erratum_sha256": file_sha256(erratum_path),
        "superseded_amended_preregistration_commit": prior_amended_commit,
        "superseded_amended_preregistration_lock_sha256": (prior_amended_lock_sha256),
    }
    original_lock_path.write_text(
        json.dumps(implementation_lock, sort_keys=True), encoding="utf-8"
    )
    git("add", ".")
    git("commit", "-q", "-m", "implementation erratum refreeze")
    prior_implementation_commit = git("rev-parse", "HEAD")
    prior_implementation_lock_sha256 = file_sha256(original_lock_path)

    erratum_2 = json.loads(
        (ROOT / "PREREGISTRATION_IMPLEMENTATION_ERRATUM_2.json").read_text(
            encoding="utf-8"
        )
    )
    erratum_2["prior_implementation_refreeze_commit"] = prior_implementation_commit
    erratum_2["prior_implementation_refreeze_lock_sha256"] = (
        prior_implementation_lock_sha256
    )
    erratum_2["prior_implementation_erratum_sha256"] = file_sha256(erratum_path)
    erratum_2_path = experiment / "PREREGISTRATION_IMPLEMENTATION_ERRATUM_2.json"
    erratum_2_path.write_text(json.dumps(erratum_2, sort_keys=True), encoding="utf-8")
    compatibility_lock = {
        **implementation_lock,
        "schema_version": 5,
        "status": common_module.IMPLEMENTATION_ERRATUM_2_LOCK_STATUS,
        "implementation_erratum_2_sha256": file_sha256(erratum_2_path),
        "superseded_implementation_refreeze_commit": (prior_implementation_commit),
        "superseded_implementation_refreeze_lock_sha256": (
            prior_implementation_lock_sha256
        ),
    }
    original_lock_path.write_text(
        json.dumps(compatibility_lock, sort_keys=True), encoding="utf-8"
    )
    git("add", ".")
    git("commit", "-q", "-m", "second implementation erratum refreeze")
    prior_compatibility_commit = git("rev-parse", "HEAD")
    prior_compatibility_lock_sha256 = file_sha256(original_lock_path)

    erratum_3 = json.loads(
        (ROOT / "PREREGISTRATION_IMPLEMENTATION_ERRATUM_3.json").read_text(
            encoding="utf-8"
        )
    )
    erratum_3["prior_compatibility_refreeze_commit"] = prior_compatibility_commit
    erratum_3["prior_compatibility_refreeze_lock_sha256"] = (
        prior_compatibility_lock_sha256
    )
    erratum_3["prior_implementation_erratum_2_sha256"] = file_sha256(erratum_2_path)
    erratum_3_path = experiment / "PREREGISTRATION_IMPLEMENTATION_ERRATUM_3.json"
    erratum_3_path.write_text(json.dumps(erratum_3, sort_keys=True), encoding="utf-8")
    active_lock = {
        **compatibility_lock,
        "schema_version": 6,
        "status": common_module.IMPLEMENTATION_ERRATUM_3_LOCK_STATUS,
        "implementation_erratum_3_sha256": file_sha256(erratum_3_path),
        "superseded_compatibility_refreeze_commit": prior_compatibility_commit,
        "superseded_compatibility_refreeze_lock_sha256": (
            prior_compatibility_lock_sha256
        ),
    }
    original_lock_path.write_text(
        json.dumps(active_lock, sort_keys=True), encoding="utf-8"
    )
    git("add", ".")
    git("commit", "-q", "-m", "third implementation erratum refreeze")
    active_commit = git("rev-parse", "HEAD")

    monkeypatch.setattr(common_module, "REPO_ROOT", repo)
    monkeypatch.setattr(common_module, "EXPERIMENT_ROOT", experiment)
    monkeypatch.setattr(common_module, "AMENDMENT_PATH", amendment_path)
    monkeypatch.setattr(common_module, "IMPLEMENTATION_ERRATUM_PATH", erratum_path)
    monkeypatch.setattr(common_module, "IMPLEMENTATION_ERRATUM_2_PATH", erratum_2_path)
    monkeypatch.setattr(common_module, "IMPLEMENTATION_ERRATUM_3_PATH", erratum_3_path)
    monkeypatch.setattr(
        common_module,
        "IMPLEMENTATION_ERRATUM_SHA256",
        file_sha256(erratum_path),
    )
    monkeypatch.setattr(
        common_module,
        "IMPLEMENTATION_ERRATUM_2_SHA256",
        file_sha256(erratum_2_path),
    )
    monkeypatch.setattr(
        common_module,
        "IMPLEMENTATION_ERRATUM_3_SHA256",
        file_sha256(erratum_3_path),
    )
    monkeypatch.setattr(common_module, "LOCK_PATH", original_lock_path)
    monkeypatch.setattr(
        common_module,
        "PRIOR_AMENDED_PREREGISTRATION_COMMIT",
        prior_amended_commit,
    )
    monkeypatch.setattr(
        common_module,
        "PRIOR_AMENDED_PREREGISTRATION_LOCK_SHA256",
        prior_amended_lock_sha256,
    )
    monkeypatch.setattr(
        common_module,
        "PRIOR_IMPLEMENTATION_REFREEZE_COMMIT",
        prior_implementation_commit,
    )
    monkeypatch.setattr(
        common_module,
        "PRIOR_IMPLEMENTATION_REFREEZE_LOCK_SHA256",
        prior_implementation_lock_sha256,
    )
    monkeypatch.setattr(
        common_module,
        "PRIOR_COMPATIBILITY_REFREEZE_COMMIT",
        prior_compatibility_commit,
    )
    monkeypatch.setattr(
        common_module,
        "PRIOR_COMPATIBILITY_REFREEZE_LOCK_SHA256",
        prior_compatibility_lock_sha256,
    )
    monkeypatch.setattr(
        common_module,
        "require_implementation_erratum_plan_disclosure",
        lambda _prior: None,
    )
    monkeypatch.setattr(
        common_module,
        "require_implementation_erratum_2_plan_disclosure",
        lambda _prior: None,
    )
    monkeypatch.setattr(
        common_module,
        "require_implementation_erratum_3_plan_disclosure",
        lambda _prior: None,
    )
    chain = common_module.preregistration_chain(active_lock)
    assert chain == {
        "preregistration_revision": 2,
        "active_preregistration_lock_sha256": file_sha256(original_lock_path),
        "preregistration_amendment_sha256": file_sha256(amendment_path),
        "original_preregistration_lock_sha256": original_lock_sha256,
        "original_preregistration_commit": original_commit,
        "active_preregistration_commit": active_commit,
    }

    original_lock_path.write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="committed active lock"):
        common_module.preregistration_chain(active_lock)


def test_run_summary_is_exact_and_authenticates_features_and_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, lock, gates, feature_summary = _synthetic_run_summary(tmp_path)
    monkeypatch.setattr(validator, "ROOT", tmp_path)
    path = tmp_path / "metrics" / "run_summary.json"
    expected_chain = json.loads(path.read_text(encoding="utf-8"))[
        "preregistration_chain"
    ]
    monkeypatch.setattr(
        validator, "preregistration_chain", lambda _lock: expected_chain
    )

    validated = validator._validate_run_summary(config, lock, gates, feature_summary)
    assert validated["status"] == "COMPLETE"

    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["undeclared"] = True
    path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="top-level schema"):
        validator._validate_run_summary(config, lock, gates, feature_summary)

    del summary["undeclared"]
    key = next(iter(summary["feature_asset_sha256"]))
    original_feature_hash = summary["feature_asset_sha256"][key]
    summary["feature_asset_sha256"][key] = "0" * 64
    path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="feature hashes"):
        validator._validate_run_summary(config, lock, gates, feature_summary)

    summary["feature_asset_sha256"][key] = original_feature_hash
    summary["runtime_implementation_sha256"]["scripts/common.py"] = "0" * 64
    path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="runtime implementation provenance"):
        validator._validate_run_summary(config, lock, gates, feature_summary)

    summary["runtime_implementation_sha256"]["scripts/common.py"] = lock[
        "implementation_sha256"
    ]["scripts/common.py"]
    summary["artifacts"]["phenotype_predictions"]["patient_level_private"] = False
    path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="privacy flag"):
        validator._validate_run_summary(config, lock, gates, feature_summary)


def test_formal_zero_option_clis_stop_on_help_or_unknown_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(freeze, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["freeze_preregistration.py", "--help"])
    with pytest.raises(SystemExit) as freeze_help:
        freeze.main()
    assert freeze_help.value.code == 0
    assert not (tmp_path / "PREREGISTRATION_LOCK.json").exists()

    monkeypatch.setattr(figures, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["generate_figures.py", "--unknown"])
    with pytest.raises(SystemExit) as figure_unknown:
        figures.main()
    assert figure_unknown.value.code != 0
    assert not (tmp_path / "figures").exists()
    assert not (tmp_path / "metrics").exists()


def test_freeze_rejects_preexisting_private_data_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data" / "patient.private.csv"
    data.parent.mkdir()
    data.write_text("patient_id\npatient-001\n", encoding="utf-8")
    monkeypatch.setattr(freeze, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["freeze_preregistration.py"])
    with pytest.raises(RuntimeError, match="outputs exist before preregistration"):
        freeze.main()
    assert not (tmp_path / "PREREGISTRATION_LOCK.json").exists()


def test_completion_marker_is_private_atomic_and_never_replaced(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "features" / "feature_matrix_complete.private.json"
    payload = {
        "schema_version": 1,
        "status": "COMPLETE",
        "cell_count": 20,
        "config_sha256": "a" * 64,
        "preregistration_lock_sha256": "b" * 64,
        "cells": [],
    }
    assert set(payload) == set(COMPLETION_KEYS)
    publish_completion_marker(marker, payload)
    before = (marker.stat().st_ino, marker.stat().st_mtime_ns, marker.read_bytes())
    assert marker.stat().st_mode & 0o777 == 0o600
    assert validate_completion_marker(marker, payload) == payload

    # Exact validation is allowed, but the complete marker is not rewritten.
    publish_completion_marker(marker, payload)
    after = (marker.stat().st_ino, marker.stat().st_mtime_ns, marker.read_bytes())
    assert after == before

    drifted = dict(payload)
    drifted["cell_count"] = 19
    with pytest.raises(ValueError, match="differs from current assets"):
        publish_completion_marker(marker, drifted)
    assert (
        marker.stat().st_ino,
        marker.stat().st_mtime_ns,
        marker.read_bytes(),
    ) == before


def test_feature_parent_authenticates_before_inspecting_or_creating_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    def authenticate() -> tuple[dict, dict]:
        events.append("authenticate")
        return {}, {}

    def path_for_cell(*_identity: object) -> Path:
        events.append("inspect_cell")
        return tmp_path / "absent.private.npz"

    monkeypatch.setattr(feature_matrix, "_authenticate_parent_context", authenticate)
    monkeypatch.setattr(feature_matrix, "feature_path", path_for_cell)
    monkeypatch.setattr(feature_matrix, "COMPLETION_PATH", tmp_path / "complete.json")
    monkeypatch.setattr(sys, "argv", ["run_feature_matrix.py"])
    feature_matrix.main()
    assert events[0] == "authenticate"
    assert "inspect_cell" in events[1:]
    assert not list(tmp_path.iterdir())


def test_feature_parent_authentication_failure_has_no_output_side_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def rejected() -> tuple[dict, dict]:
        raise ValueError("lock rejected")

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("output operation occurred before authentication")

    monkeypatch.setattr(feature_matrix, "_authenticate_parent_context", rejected)
    monkeypatch.setattr(feature_matrix, "feature_path", forbidden)
    monkeypatch.setattr(feature_matrix, "private_directory", forbidden)
    monkeypatch.setattr(feature_matrix, "COMPLETION_PATH", tmp_path / "complete.json")
    monkeypatch.setattr(sys, "argv", ["run_feature_matrix.py", "--execute"])
    with pytest.raises(ValueError, match="lock rejected"):
        feature_matrix.main()
    assert not list(tmp_path.iterdir())

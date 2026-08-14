from __future__ import annotations

from pathlib import Path
import gzip
import sys

import numpy as np
import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_core as core  # noqa: E402
import freeze_preregistration as prereg  # noqa: E402
import validate_audit as validator  # noqa: E402


def test_adjacent_percent_change_is_signed_percent_and_fails_closed() -> None:
    start = np.asarray([10.0, -10.0, 0.0, np.nan, 5.0])
    end = np.asarray([15.0, -5.0, 3.0, 4.0, np.nan])
    observed, valid = core.adjacent_percent_change(start, end)

    np.testing.assert_array_equal(valid, [True, True, False, False, False])
    np.testing.assert_allclose(observed[:2], [50.0, 50.0])
    assert np.isnan(observed[2:]).all()


def test_preregistration_prelock_output_scan_allows_only_placeholders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(prereg, "AUDIT_ROOT", tmp_path)
    for directory_name in prereg.OUTPUT_DIRECTORIES:
        directory = tmp_path / directory_name
        directory.mkdir(parents=True)
        (directory / ".gitkeep").touch()
    prereg._assert_no_retained_formal_outputs()

    partial = tmp_path / "metrics" / "partial.csv"
    partial.write_text("diagnostic,value\nsmoke,0.1\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="partial.csv"):
        prereg._assert_no_retained_formal_outputs()


def test_preregistration_atomic_writer_refuses_overwrite(tmp_path: Path) -> None:
    lock = tmp_path / "PREREGISTRATION_LOCK.json"
    prereg._atomic_json_no_replace(lock, {"status": "LOCKED_BEFORE_FORMAL_RUN"})
    assert lock.stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prereg._atomic_json_no_replace(lock, {"status": "CHANGED"})


def test_target_transform_uses_outer_train_only() -> None:
    values = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, -1.0e9, 1.0e9])
    fit_mask = np.asarray([True, True, True, True, True, False, False])
    first = core.fit_target_transform(values, fit_mask, log1p=False, quantiles=(0.01, 0.99))

    mutated = values.copy()
    mutated[~fit_mask] = [7.0e17, -8.0e17]
    second = core.fit_target_transform(mutated, fit_mask, log1p=False, quantiles=(0.01, 0.99))

    assert first == second
    np.testing.assert_allclose([first.lower, first.upper], np.quantile(values[fit_mask], [0.01, 0.99]))
    transformed_train = first.transform(values[fit_mask])
    assert np.mean(transformed_train) == pytest.approx(0.0, abs=1e-12)
    assert np.std(transformed_train, ddof=0) == pytest.approx(1.0, abs=1e-12)


def _fit_residual(target: np.ndarray, predictor: np.ndarray, splits: np.ndarray):
    transforms: list[dict[str, object]] = []
    residualizers: list[dict[str, object]] = []
    outcome = core._residual_outcome(
        target_values=target,
        predictor_values=(predictor,),
        valid=np.ones(len(target), dtype=bool),
        splits=splits,
        target_log1p=False,
        predictor_log1p=(False,),
        predictor_names=("FTV",),
        quantiles=(0.01, 0.99),
        task_type="static",
        target_kind="residual_ftv",
        target="LD",
        timing="T1",
        interval="",
        fold=0,
        residualizer_alpha=1.0,
        transform_rows=transforms,
        residualizer_rows=residualizers,
    )
    return outcome, transforms, residualizers


def test_residualizer_and_residual_scaler_use_outer_train_only() -> None:
    splits = np.asarray(["train"] * 9 + ["val"] * 3 + ["test"] * 3)
    predictor = np.linspace(1.0, 15.0, 15)
    target = 0.7 * predictor + np.sin(predictor)
    first, first_transforms, first_rows = _fit_residual(target, predictor, splits)

    mutated_predictor = predictor.copy()
    mutated_target = target.copy()
    mutated_predictor[splits != "train"] = np.linspace(-1.0e8, 1.0e8, 6)
    mutated_target[splits != "train"] = np.linspace(3.0e9, -3.0e9, 6)
    second, second_transforms, second_rows = _fit_residual(
        mutated_target, mutated_predictor, splits
    )

    np.testing.assert_allclose(first.probe_y[splits == "train"], second.probe_y[splits == "train"])
    assert first.residual_center == pytest.approx(second.residual_center)
    assert first.residual_scale == pytest.approx(second.residual_scale)
    assert first_transforms == second_transforms
    assert first_rows == second_rows
    assert first_rows[0]["fit_scope"] == "outer_train_only"
    assert first_rows[0]["alpha"] == 1.0


def test_build_outcomes_has_exact_raw_and_residual_taxonomy() -> None:
    patient_ids = np.asarray([f"P{index:03d}" for index in range(15)])
    base = np.arange(1.0, 16.0)[:, None]
    values = {
        "FTV": base * np.asarray([[1.0, 0.8, 0.6, 0.4]]),
        "LD": base * np.asarray([[0.5, 0.45, 0.4, 0.35]]) + 2.0,
        "SPH": 0.2 + base * np.asarray([[0.01, 0.012, 0.014, 0.016]]),
        "BPE": 1.0 + base * np.asarray([[0.3, 0.25, 0.2, 0.15]]),
    }
    targets = core.TargetDataset(
        patient_ids=patient_ids,
        trial_ids=np.arange(15),
        values=values,
        patient_to_index={patient: index for index, patient in enumerate(patient_ids)},
        patient_set_sha256="synthetic",
        workbook_max_abs_difference={family: 0.0 for family in core.FAMILIES},
    )
    splits = {0: np.asarray(["train"] * 9 + ["val"] * 3 + ["test"] * 3)}
    config = {
        "target_transforms": {
            "winsor_quantiles": [0.01, 0.99],
            "log1p_families": ["FTV", "LD", "BPE"],
        },
        "residualization": {"alpha": 1.0},
    }

    static, dynamic, transform_rows, residualizer_rows = core.build_outcomes(
        config, targets, splits
    )
    assert set(static) == {(0, timing) for timing in core.VISITS}
    assert set(dynamic) == {(0, interval) for interval in core.INTERVALS}
    assert all(len(outcomes) == 9 for outcomes in [*static.values(), *dynamic.values()])
    for outcomes in [*static.values(), *dynamic.values()]:
        taxonomy = {(outcome.target_kind, outcome.target) for outcome in outcomes}
        assert {("raw", family) for family in core.FAMILIES} <= taxonomy
        assert {("residual_ftv", family) for family in ("LD", "SPH", "BPE")} <= taxonomy
        assert {("residual_ftv_ld", family) for family in ("SPH", "BPE")} <= taxonomy
    assert len(transform_rows) == 112
    assert len(residualizer_rows) == 35

    raw_ftv = next(
        outcome
        for outcome in dynamic[(0, "T0->T1")]
        if outcome.target_kind == "raw" and outcome.target == "FTV"
    )
    expected, valid = core.adjacent_percent_change(values["FTV"][:, 0], values["FTV"][:, 1])
    np.testing.assert_array_equal(raw_ftv.valid, valid)
    np.testing.assert_allclose(raw_ftv.natural_y, expected)


def test_ridge_path_train_state_is_independent_of_test_features() -> None:
    rng = np.random.default_rng(17)
    x_train = rng.normal(size=(12, 5))
    y_train = rng.normal(size=(12, 2))
    x_validation = rng.normal(size=(4, 5))
    x_test = rng.normal(size=(3, 5))
    first = core._ridge_path_predictions(
        x_train, y_train, x_validation, x_test, (0.01, 1.0, 100.0)
    )
    mutated_test = x_test * 1.0e8 + 7.0e7
    second = core._ridge_path_predictions(
        x_train, y_train, x_validation, mutated_test, (0.01, 1.0, 100.0)
    )

    np.testing.assert_allclose(first[0], second[0])  # validation predictions
    np.testing.assert_allclose(first[1], second[1])  # train-fitted duals
    np.testing.assert_allclose(first[3], second[3])  # train target center
    assert first[4] == second[4]
    assert not np.allclose(first[2], second[2])  # only the test design application changes


class _MemoryWriter:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def write(self, row):
        self.rows.append(dict(row))


def _run_single_probe(test_target: np.ndarray):
    rng = np.random.default_rng(23)
    matrix = rng.normal(size=(15, 4))
    splits = np.asarray(["train"] * 9 + ["val"] * 3 + ["test"] * 3)
    y = matrix[:, 0] - 0.3 * matrix[:, 1]
    y[-3:] = test_target
    transform = core.fit_target_transform(
        y,
        splits == "train",
        log1p=False,
        quantiles=(0.01, 0.99),
    )
    outcome = core.Outcome(
        task_type="static",
        target_kind="raw",
        target="LD",
        timing="T1",
        interval="",
        valid=np.ones(15, dtype=bool),
        probe_y=transform.transform(y),
        natural_y=y.copy(),
        transformed_y=transform.transform_unscaled(y),
        metric_space="natural_target",
        raw_transform=transform,
        residual_center=0.0,
        residual_scale=1.0,
        residualizer_id="",
        conditional_standardized=None,
    )
    cell = core.FeatureCell(
        seed=2026,
        arm="LOCAL0",
        fold=0,
        patient_ids=np.asarray([f"P{index:03d}" for index in range(15)]),
        splits=splits,
        representations={},
        validity={},
        selected_epoch=2,
        provenance={},
    )
    writer = _MemoryWriter()
    selections: list[dict[str, object]] = []
    fold_metrics: list[dict[str, object]] = []
    coverage: list[dict[str, object]] = []
    core.probe_outcome_batch(
        x=matrix,
        feature_valid=np.ones(15, dtype=bool),
        outcomes=[outcome],
        cell=cell,
        representation="Z2",
        matched_reference_for="",
        input_variant="current",
        config={"probe": {"alphas": [0.0001, 0.1, 10.0, 1000.0]}},
        writer=writer,
        accumulator=core.OOFAccumulator(),
        selection_rows=selections,
        fold_metric_rows=fold_metrics,
        coverage_rows=coverage,
    )
    return writer.rows, selections


def test_alpha_selection_and_test_prediction_are_blind_to_test_targets() -> None:
    first_predictions, first_selections = _run_single_probe(
        np.asarray([-1.0e12, 2.0e12, -3.0e12])
    )
    second_predictions, second_selections = _run_single_probe(
        np.asarray([9.0e16, -8.0e16, 7.0e16])
    )

    assert first_selections[0]["selected_alpha"] == second_selections[0]["selected_alpha"]
    assert first_selections[0]["alpha_validation_mse_json"] == second_selections[0]["alpha_validation_mse_json"]
    assert first_selections[0]["test_used_for_alpha_selection"] is False
    assert first_selections[0]["test_used_for_scaler"] is False
    assert first_selections[0]["test_predict_call_count"] == 1
    np.testing.assert_allclose(
        [row["y_pred_natural"] for row in first_predictions],
        [row["y_pred_natural"] for row in second_predictions],
    )


def _synthetic_feature_cell() -> core.FeatureCell:
    rng = np.random.default_rng(31)
    dimensions = {"Z1": 192, "Z2": 192, "Z3": 128, "Z4": 256, "Z5": 256, "Z6": 256, "Z7": 256}
    representations = {
        name: rng.normal(size=(6, 4, dimension))
        for name, dimension in dimensions.items()
    }
    validity = {name: np.ones((6, 4), dtype=bool) for name in dimensions}
    validity["Z5"][0, 1] = False
    validity["Z6"][1, 2] = False
    validity["Z7"][2, 3] = False
    return core.FeatureCell(
        seed=2026,
        arm="LOCAL3",
        fold=0,
        patient_ids=np.asarray([f"P{index:03d}" for index in range(6)]),
        splits=np.asarray(["train", "train", "val", "val", "test", "test"]),
        representations=representations,
        validity=validity,
        selected_epoch=2,
        provenance={},
    )


def test_static_and_dynamic_views_preserve_shapes_and_matched_validity() -> None:
    cell = _synthetic_feature_cell()
    dimensions = {name: values.shape[2] for name, values in cell.representations.items()}

    static = core.static_views(cell, 1)
    assert set(static) == set(core.MAIN_REPRESENTATIONS) | set(core.ORACLE_TO_MATCHED.values())
    for representation in core.MAIN_REPRESENTATIONS:
        matrix, valid, matched_for = static[representation]
        assert matrix.shape == (6, dimensions[representation])
        np.testing.assert_array_equal(valid, cell.validity[representation][:, 1])
        assert matched_for == ""
    matched_matrix, matched_valid, matched_for = static["Z4_MATCHED_Z5"]
    np.testing.assert_allclose(matched_matrix, cell.representations["Z4"][:, 1])
    np.testing.assert_array_equal(matched_valid, cell.validity["Z5"][:, 1])
    assert matched_for == "Z5"

    differences = core.dynamic_views(
        cell, 1, "difference", include_matched_references=True
    )
    prefixes = core.dynamic_views(cell, 1, "prefix", include_matched_references=False)
    assert set(differences) == set(core.MAIN_REPRESENTATIONS) | set(core.ORACLE_TO_MATCHED.values())
    assert set(prefixes) == set(core.MAIN_REPRESENTATIONS)
    for representation in core.MAIN_REPRESENTATIONS:
        start = cell.representations[representation][:, 1]
        end = cell.representations[representation][:, 2]
        np.testing.assert_allclose(differences[representation][0], end - start)
        np.testing.assert_allclose(
            prefixes[representation][0], np.concatenate((start, end, end - start), axis=1)
        )
        assert differences[representation][0].shape == (6, dimensions[representation])
        assert prefixes[representation][0].shape == (6, 3 * dimensions[representation])
        np.testing.assert_array_equal(
            differences[representation][1],
            cell.validity[representation][:, 1] & cell.validity[representation][:, 2],
        )
    np.testing.assert_array_equal(
        differences["Z4_MATCHED_Z6"][1],
        cell.validity["Z6"][:, 1] & cell.validity["Z6"][:, 2],
    )


def _metric_row(representation: str, value: float, *, n: int = 100) -> dict[str, object]:
    row: dict[str, object] = {
        "seed": 2026,
        "arm": "LOCAL0",
        "representation": representation,
        "task_type": "static",
        "target_definition": "goal6_workbook_endpoint",
        "target_kind": "raw",
        "target": "LD",
        "timing": "T1",
        "interval": "",
        "input_variant": "current",
        "metric_space": "natural_target",
        "n": n,
    }
    for metric in (
        "spearman",
        "pearson",
        "natural_r2",
        "transformed_r2",
        "rmse",
        "mae",
        "prediction_target_variance_ratio",
        "calibration_slope",
        "residual_spearman",
        "residual_transformed_r2",
        "reconstructed_natural_r2",
        "reconstructed_natural_rmse",
        "reconstructed_natural_mae",
    ):
        row[metric] = value
    return row


def test_representation_and_localization_comparisons_are_exactly_matched() -> None:
    rows = [
        _metric_row("Z1", 1.0),
        _metric_row("Z2", 2.0),
        _metric_row("Z3", 3.0),
        _metric_row("Z4", 4.0),
        _metric_row("Z5", 5.0),
        _metric_row("Z6", 6.0),
        _metric_row("Z7", 7.0),
        _metric_row("Z4_MATCHED_Z5", 4.0),
        _metric_row("Z4_MATCHED_Z6", 5.0),
        _metric_row("Z4_MATCHED_Z7", 6.0),
    ]
    frame = pd.DataFrame(rows)
    pairs = core.representation_pair_comparisons(frame)
    assert len(pairs) == 3
    np.testing.assert_allclose(pairs["delta_spearman"], 1.0)

    localization = core.localization_comparisons(frame)
    assert len(localization) == 3
    np.testing.assert_allclose(localization["delta_spearman"], 1.0)
    np.testing.assert_array_equal(localization["n_oracle"], localization["n_full_local_matched"])

    mismatched = frame.copy()
    mismatched.loc[mismatched["representation"] == "Z4_MATCHED_Z5", "n"] = 99
    with pytest.raises(ValueError, match="populations differ"):
        core.localization_comparisons(mismatched)

    duplicated = pd.concat([frame, frame.loc[frame["representation"] == "Z2"]], ignore_index=True)
    with pytest.raises(pd.errors.MergeError):
        core.representation_pair_comparisons(duplicated)


def _complete_oof_frame() -> pd.DataFrame:
    dimensions = {"Z1": 192, "Z2": 192, "Z3": 128, "Z4": 256, "Z5": 256, "Z6": 256, "Z7": 256}
    outcome_taxonomy = [
        *(('raw', target) for target in ("FTV", "LD", "SPH", "BPE")),
        *(('residual_ftv', target) for target in ("LD", "SPH", "BPE")),
        *(('residual_ftv_ld', target) for target in ("SPH", "BPE")),
    ]
    rows: list[dict[str, object]] = []

    def add(rep: str, seed: int, arm: str, task: str, kind: str, target: str, timing: str, interval: str, variant: str, matched_for: str = "") -> None:
        residual = kind != "raw"
        rows.append(
            {
                "seed": seed,
                "arm": arm,
                "representation": rep,
                "matched_reference_for": matched_for,
                "task_type": task,
                "target_definition": "adjacent_percent_change_new_extension" if task == "dynamic" else "goal6_workbook_endpoint",
                "target_kind": kind,
                "target": target,
                "timing": timing,
                "interval": interval,
                "input_variant": variant,
                "metric_space": "natural_target" if kind == "raw" else "residual",
                "feature_dim": dimensions.get(rep, 256) * (3 if variant == "prefix" else 1),
                "n": 375,
                "n_folds": 5,
                "spearman": 0.1,
                "pearson": 0.1,
                "natural_r2": np.nan if residual else 0.0,
                "transformed_r2": 0.0,
                "rmse": np.nan if residual else 1.0,
                "mae": np.nan if residual else 0.8,
                "prediction_target_variance_ratio": 0.2,
                "calibration_slope": 0.5,
                "residual_spearman": 0.1 if residual else np.nan,
                "residual_transformed_r2": 0.0 if residual else np.nan,
                "reconstructed_natural_r2": 0.0 if residual else np.nan,
                "reconstructed_natural_rmse": 1.0 if residual else np.nan,
                "reconstructed_natural_mae": 0.8 if residual else np.nan,
                "natural_metric_interpretation": "conditional_target_reconstruction" if residual else "raw_target",
                "rank_aggregation": "outer_test_n_weighted_fold_residual_metric" if residual else "pooled_oof_natural_target",
                "transformed_r2_aggregation": "outer_test_n_weighted_fold_r2",
            }
        )

    for seed in (2026, 3026):
        for arm in ("LOCAL0", "LOCAL3"):
            for rep in core.MAIN_REPRESENTATIONS:
                for timing in core.VISITS:
                    for kind, target in outcome_taxonomy:
                        add(rep, seed, arm, "static", kind, target, timing, "", "current")
                for interval in core.INTERVALS:
                    for variant in ("difference", "prefix"):
                        for kind, target in outcome_taxonomy:
                            add(rep, seed, arm, "dynamic", kind, target, "", interval, variant)
            for oracle, matched in core.ORACLE_TO_MATCHED.items():
                for timing in core.VISITS:
                    for kind, target in outcome_taxonomy:
                        add(matched, seed, arm, "static", kind, target, timing, "", "current", oracle)
                for interval in core.INTERVALS:
                    for kind, target in outcome_taxonomy:
                        add(matched, seed, arm, "dynamic", kind, target, "", interval, "difference", oracle)
    return pd.DataFrame(rows)


def test_oof_completeness_validator_checks_full_factorial_matrix() -> None:
    frame = _complete_oof_frame()
    assert len(frame) == validator.EXPECTED_COUNTS["oof_metrics"] == 3276
    config = {"frozen": {"seeds": [2026, 3026], "arms": ["LOCAL0", "LOCAL3"]}}
    checks = validator.Checks()
    validator.validate_oof_matrix(frame, config, checks)
    assert checks.errors == []

    incomplete = frame.iloc[:-1].copy()
    failed = validator.Checks()
    validator.validate_oof_matrix(incomplete, config, failed)
    assert any("3276" in error or "756" in error for error in failed.errors)


def test_public_privacy_scan_blocks_identifier_columns_and_values(tmp_path: Path) -> None:
    for directory in ("metrics", "manifests", "reports"):
        (tmp_path / directory).mkdir()
    safe = tmp_path / "metrics" / "safe.csv"
    safe.write_text("target,n,spearman\nLD,375,0.4\n", encoding="utf-8")
    checks = validator.Checks()
    validator.validate_public_privacy(
        tmp_path,
        {},
        checks,
        identifiers=(["SYNTHETIC_PATIENT_TOKEN"], ["SYNTHETIC_TRIAL_TOKEN"]),
    )
    assert checks.errors == []

    unsafe = tmp_path / "metrics" / "unsafe.csv"
    unsafe.write_text(
        "patient_id,target,n\nSYNTHETIC_PATIENT_TOKEN,LD,1\n",
        encoding="utf-8",
    )
    failed = validator.Checks()
    validator.validate_public_privacy(
        tmp_path,
        {},
        failed,
        identifiers=(["SYNTHETIC_PATIENT_TOKEN"], ["SYNTHETIC_TRIAL_TOKEN"]),
    )
    assert len(failed.errors) >= 2
    assert all("SYNTHETIC_PATIENT_TOKEN" not in error for error in failed.errors)
    assert validator.forbidden_public_columns(["patient_id", "label_pcr", "n"]) == [
        "label_pcr",
        "patient_id",
    ]


def test_private_prediction_writer_enforces_schema_and_permissions(tmp_path: Path) -> None:
    output = tmp_path / "predictions" / "oof_predictions.private.csv.gz"
    writer = core.PrivatePredictionWriter(output)
    row = {column: "" for column in core.PREDICTION_COLUMNS}
    row.update(
        {
            "patient_id": "SYNTHETIC_PATIENT_TOKEN",
            "fold": 0,
            "seed": 2026,
            "arm": "LOCAL0",
            "representation": "Z1",
            "task_type": "static",
            "target_definition": "goal6_workbook_endpoint",
            "target_kind": "raw",
            "target": "LD",
            "timing": "T0",
            "input_variant": "current",
            "metric_space": "natural_target",
            "feature_dim": 192,
            "selected_alpha": 1.0,
            "y_true_natural": 1.0,
            "y_pred_natural": 1.1,
            "y_true_transformed": 0.0,
            "y_pred_transformed": 0.1,
        }
    )
    writer.write(row)
    writer.close()

    assert (output.stat().st_mode & 0o777) == 0o600
    assert (output.parent.stat().st_mode & 0o777) == 0o700
    with gzip.open(output, "rt", encoding="utf-8") as stream:
        header, record = stream.read().splitlines()
    assert tuple(header.split(",")) == core.PREDICTION_COLUMNS
    assert "SYNTHETIC_PATIENT_TOKEN" in record


def test_change_scope_helper_rejects_paths_outside_new_tree() -> None:
    audit_relative = "additional_experiments/nonftv_phenotype_decodability_audit"
    paths = [
        f"{audit_relative}/scripts/audit_core.py",
        f"{audit_relative}/reports/final_report.md",
        "additional_experiments/older_goal/reports/final_report.md",
        "README.md",
    ]
    assert validator.outside_audit_paths(paths, audit_relative) == [
        "README.md",
        "additional_experiments/older_goal/reports/final_report.md",
    ]


def test_expected_public_matrix_counts_are_internally_consistent() -> None:
    assert validator.EXPECTED_COUNTS["static_target_matrix"] == 4 * 7 * 4 * 2 * 2
    assert validator.EXPECTED_COUNTS["residual_target_matrix"] == 5 * 7 * 4 * 2 * 2
    assert validator.EXPECTED_COUNTS["dynamic_target_matrix"] == 9 * 7 * 3 * 2 * 2 * 2
    main = (
        validator.EXPECTED_COUNTS["static_target_matrix"]
        + validator.EXPECTED_COUNTS["residual_target_matrix"]
        + validator.EXPECTED_COUNTS["dynamic_target_matrix"]
    )
    matched = 9 * 3 * 4 * 2 * 2 + 9 * 3 * 3 * 2 * 2
    assert main == 2520
    assert matched == 756
    assert main + matched == validator.EXPECTED_COUNTS["oof_metrics"]
    assert validator.EXPECTED_COUNTS["fold_metrics"] == 5 * validator.EXPECTED_COUNTS["oof_metrics"]

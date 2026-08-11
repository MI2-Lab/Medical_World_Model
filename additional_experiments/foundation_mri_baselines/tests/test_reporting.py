from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foundation_mri.evaluation import (  # noqa: E402
    aggregate_binary_predictions,
    aggregate_continuous_predictions,
    aggregate_multiclass_predictions,
)
from foundation_mri.reporting import (  # noqa: E402
    BINARY_IRLS_COLUMNS,
    BINARY_IRLS_RECOMPUTE_ATOL,
    BINARY_IRLS_RECOMPUTE_RTOL,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    ReportingInputs,
    ReportingOutputs,
    _assert_public_matches,
    _build_reporting_marker,
    _publish_no_overwrite_with_rollback,
    _reliability_points,
    _weighted_metric_distribution,
    static_comparison_contract,
    summarize_results,
)
from foundation_mri.data import file_sha256  # noqa: E402


FOLD_HASH = "a" * 64
FULL_N = 40
COMPLETE_N = 20
FULL_POPULATION = f"full_{FULL_N}"
COMPLETE_POPULATION = f"radiomics_complete_case_{COMPLETE_N}"
FOUNDATION = "Med3D"


def _numeric_identity_frame(**values: float) -> pd.DataFrame:
    return pd.DataFrame({"identity": ["cell"], **{key: [value] for key, value in values.items()}})


def test_binary_irls_public_recompute_tolerance_accepts_roundtrip_only() -> None:
    recomputed = _numeric_identity_frame(
        calibration_intercept=-0.29926816783893295,
        calibration_slope=0.812345678901,
        auroc=0.75,
    )
    serialized = recomputed.copy()
    serialized.loc[0, "calibration_intercept"] = -0.2992681677870132
    _assert_public_matches(
        serialized,
        recomputed,
        key_columns=("identity",),
        label="baseline",
        relaxed_numeric_columns=BINARY_IRLS_COLUMNS,
    )
    meaningfully_drifted = serialized.copy()
    meaningfully_drifted.loc[0, "calibration_intercept"] += 1e-6
    with pytest.raises(ValueError, match="calibration_intercept"):
        _assert_public_matches(
            meaningfully_drifted,
            recomputed,
            key_columns=("identity",),
            label="baseline",
            relaxed_numeric_columns=BINARY_IRLS_COLUMNS,
        )


def test_binary_irls_recompute_fails_immediately_outside_derived_boundary() -> None:
    expected_value = 0.4
    boundary = BINARY_IRLS_RECOMPUTE_ATOL + (
        BINARY_IRLS_RECOMPUTE_RTOL * abs(expected_value)
    )
    recomputed = _numeric_identity_frame(
        calibration_intercept=expected_value,
        calibration_slope=1.0,
    )
    observed = recomputed.copy()
    observed.loc[0, "calibration_intercept"] = expected_value + 1.01 * boundary
    with pytest.raises(ValueError, match="calibration_intercept"):
        _assert_public_matches(
            observed,
            recomputed,
            key_columns=("identity",),
            label="baseline",
            relaxed_numeric_columns=BINARY_IRLS_COLUMNS,
        )


def test_core_metrics_and_continuous_calibration_columns_remain_strict() -> None:
    recomputed = _numeric_identity_frame(
        calibration_intercept=-0.29926816783893295,
        calibration_slope=0.812345678901,
        auroc=0.75,
    )
    core_drift = recomputed.copy()
    core_drift.loc[0, "auroc"] += 1e-9
    with pytest.raises(ValueError, match="auroc"):
        _assert_public_matches(
            core_drift,
            recomputed,
            key_columns=("identity",),
            label="baseline",
            relaxed_numeric_columns=BINARY_IRLS_COLUMNS,
        )
    continuous_drift = recomputed.copy()
    continuous_drift.loc[0, "calibration_intercept"] = -0.2992681677870132
    with pytest.raises(ValueError, match="calibration_intercept"):
        _assert_public_matches(
            continuous_drift,
            recomputed,
            key_columns=("identity",),
            label="FTV",
        )


def test_public_recompute_nan_equivalence_rule_is_unchanged() -> None:
    recomputed = _numeric_identity_frame(calibration_intercept=np.nan, auroc=0.75)
    observed = recomputed.copy()
    _assert_public_matches(
        observed,
        recomputed,
        key_columns=("identity",),
        label="baseline",
        relaxed_numeric_columns=("calibration_intercept",),
    )
    observed.loc[0, "calibration_intercept"] = 0.0
    with pytest.raises(ValueError, match="calibration_intercept"):
        _assert_public_matches(
            observed,
            recomputed,
            key_columns=("identity",),
            label="baseline",
            relaxed_numeric_columns=("calibration_intercept",),
        )


def test_numeric_relaxation_cannot_be_applied_to_core_or_continuous_columns() -> None:
    frame = _numeric_identity_frame(calibration_intercept=0.0, auroc=0.75)
    with pytest.raises(ValueError, match="non-IRLS relaxed columns"):
        _assert_public_matches(
            frame,
            frame,
            key_columns=("identity",),
            label="baseline",
            relaxed_numeric_columns=("auroc",),
        )
    with pytest.raises(ValueError, match="restricted to baseline/phenotype"):
        _assert_public_matches(
            frame,
            frame,
            key_columns=("identity",),
            label="FTV",
            relaxed_numeric_columns=("calibration_intercept",),
        )


def _score(truth: np.ndarray, label: str) -> np.ndarray:
    strength = 0.08 + (sum(ord(character) for character in label) % 11) / 100.0
    noise = ((np.arange(len(truth)) % 7) - 3) * 0.008
    return np.clip(0.5 + strength * (2.0 * truth - 1.0) + noise, 0.01, 0.99)


def _binary_rows(
    patient_ids: np.ndarray,
    truth: np.ndarray,
    *,
    target: str,
    model: str,
    spatial: str,
    timing: str,
    population: str,
) -> pd.DataFrame:
    score = _score(truth, f"{target}|{model}|{spatial}|{timing}|{population}")
    return pd.DataFrame(
        {
            "patient_id": patient_ids,
            "target": target,
            "model": model,
            "spatial": spatial,
            "timing": timing,
            "analysis_population": population,
            "split_seed": 2026,
            "fold_manifest_sha256": FOLD_HASH,
            "fold": np.arange(len(patient_ids)) % 5,
            "split": "test",
            "y_true": truth,
            "y_score": score,
        }
    )


def _write_pair(private: pd.DataFrame, public: pd.DataFrame, private_path: Path, public_path: Path) -> None:
    private_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.parent.mkdir(parents=True, exist_ok=True)
    private.to_csv(private_path, index=False)
    public.to_csv(public_path, index=False)


def _synthetic_outputs(root: Path) -> ReportingInputs:
    full_ids = np.asarray([f"P{index:03d}" for index in range(FULL_N)], dtype=str)
    complete_ids = full_ids[:COMPLETE_N]
    pcr = ((np.arange(FULL_N) // 5) % 2).astype(np.int64)
    complete_pcr = pcr[:COMPLETE_N]
    baseline_parts = []
    source_axes = ((FOUNDATION, "GLOBAL"), (FOUNDATION, "LOCAL"), ("GAP0", "GLOBAL"), ("LOCAL0", "LOCAL"))
    for timing in ("T0", "T0-T1", "T0-T2"):
        baseline_parts.append(
            _binary_rows(
                full_ids,
                pcr,
                target="pCR",
                model="clinical_only",
                spatial="NONE",
                timing=timing,
                population=FULL_POPULATION,
            )
        )
        baseline_parts.append(
            _binary_rows(
                complete_ids,
                complete_pcr,
                target="pCR",
                model="clinical_only_paired",
                spatial="NONE",
                timing=timing,
                population=COMPLETE_POPULATION,
            )
        )
        baseline_parts.append(
            _binary_rows(
                complete_ids,
                complete_pcr,
                target="pCR",
                model="clinical_ftv",
                spatial="TABULAR",
                timing=timing,
                population=COMPLETE_POPULATION,
            )
        )
        for source, spatial in source_axes:
            for suffix in ("mri_only", "mri_clinical"):
                baseline_parts.append(
                    _binary_rows(
                        full_ids,
                        pcr,
                        target="pCR",
                        model=f"{source}_{suffix}",
                        spatial=spatial,
                        timing=timing,
                        population=FULL_POPULATION,
                    )
                )
            for suffix in ("mri_only_paired", "mri_clinical_paired"):
                baseline_parts.append(
                    _binary_rows(
                        complete_ids,
                        complete_pcr,
                        target="pCR",
                        model=f"{source}_{suffix}",
                        spatial=spatial,
                        timing=timing,
                        population=COMPLETE_POPULATION,
                    )
                )
            if source == FOUNDATION:
                baseline_parts.append(
                    _binary_rows(
                        complete_ids,
                        complete_pcr,
                        target="pCR",
                        model=f"{source}_mri_clinical_ftv",
                        spatial=spatial,
                        timing=timing,
                        population=COMPLETE_POPULATION,
                    )
                )
    baseline = pd.concat(baseline_parts, ignore_index=True)
    baseline_public = aggregate_binary_predictions(baseline)

    hr = ((np.arange(FULL_N) // 5 + 1) % 2).astype(np.int64)
    her2 = ((np.arange(FULL_N) // 10) % 2).astype(np.int64)
    phenotype = pd.concat(
        [
            _binary_rows(
                full_ids,
                target_values,
                target=target,
                model=source,
                spatial=spatial,
                timing="T0",
                population=FULL_POPULATION,
            )
            for source, spatial in source_axes
            for target, target_values in (("HR", hr), ("HER2", her2))
        ],
        ignore_index=True,
    )
    phenotype_public = aggregate_binary_predictions(phenotype)

    classes = tuple(sorted(("HR-/HER2-", "HR-/HER2+", "HR+/HER2-", "HR+/HER2+")))
    class_index = (np.arange(FULL_N) // 5) % len(classes)
    subtype_truth = np.asarray([classes[index] for index in class_index], dtype=str)
    subtype_parts = []
    for source, spatial in source_axes:
        probability = np.full((FULL_N, len(classes)), 0.1, dtype=np.float64)
        probability[np.arange(FULL_N), class_index] = 0.7
        subtype_parts.append(
            pd.DataFrame(
                {
                    "patient_id": full_ids,
                    "target": "HR_HER2_subtype",
                    "model": source,
                    "spatial": spatial,
                    "timing": "T0",
                    "analysis_population": FULL_POPULATION,
                    "split_seed": 2026,
                    "fold_manifest_sha256": FOLD_HASH,
                    "fold": np.arange(FULL_N) % 5,
                    "split": "test",
                    "y_true": subtype_truth,
                    "classes_json": json.dumps(classes, separators=(",", ":")),
                    "probabilities_json": [
                        json.dumps(row.tolist(), separators=(",", ":")) for row in probability
                    ],
                }
            )
        )
    subtype = pd.concat(subtype_parts, ignore_index=True)
    subtype_public = aggregate_multiclass_predictions(subtype)

    ftv_parts = []
    base = np.linspace(2.0, 25.0, COMPLETE_N)
    for source, spatial in source_axes:
        for task, endpoints in (
            ("static", ("T0", "T1", "T2", "T3")),
            ("delta", ("T0-T1", "T1-T2", "T2-T3")),
        ):
            for endpoint_index, endpoint in enumerate(endpoints):
                truth = base + endpoint_index if task == "static" else np.sin(base + endpoint_index)
                prediction = truth + 0.1 * np.cos(np.arange(COMPLETE_N) + endpoint_index)
                ftv_parts.append(
                    pd.DataFrame(
                        {
                            "patient_id": complete_ids,
                            "target": "FTV",
                            "model": source,
                            "spatial": spatial,
                            "task": task,
                            "endpoint": endpoint,
                            "analysis_population": COMPLETE_POPULATION,
                            "split_seed": 2026,
                            "fold_manifest_sha256": FOLD_HASH,
                            "fold": np.arange(COMPLETE_N) % 5,
                            "split": "test",
                            "y_true": truth,
                            "y_pred": prediction,
                            "train_mean_baseline": float(np.mean(truth)),
                        }
                    )
                )
    ftv = pd.concat(ftv_parts, ignore_index=True)
    ftv_public = aggregate_continuous_predictions(ftv)

    inputs = ReportingInputs(
        root / "predictions/baseline_predictions.private.csv",
        root / "metrics/baseline_metrics.csv",
        root / "predictions/phenotype_predictions.private.csv",
        root / "metrics/phenotype_metrics.csv",
        root / "predictions/subtype_predictions.private.csv",
        root / "metrics/subtype_metrics.csv",
        root / "predictions/ftv_probe_predictions.private.csv",
        root / "metrics/ftv_probe_metrics.csv",
    )
    _write_pair(baseline, baseline_public, inputs.baseline_private, inputs.baseline_public)
    _write_pair(phenotype, phenotype_public, inputs.phenotype_private, inputs.phenotype_public)
    _write_pair(subtype, subtype_public, inputs.subtype_private, inputs.subtype_public)
    _write_pair(ftv, ftv_public, inputs.ftv_private, inputs.ftv_public)
    return inputs


def _outputs(root: Path) -> ReportingOutputs:
    return ReportingOutputs(
        root / "metrics/paired.csv",
        root / "metrics/summary.json",
        root / "reports/summary.md",
        root / "figures/timing.png",
        root / "figures/calibration.png",
    )


def test_weighted_bootstrap_metrics_match_expanded_samples() -> None:
    truth = np.asarray([0, 1, 0, 1, 1, 0], dtype=np.int8)
    score = np.asarray([0.1, 0.8, 0.2, 0.8, 0.6, 0.2], dtype=np.float64)
    weights = np.asarray(
        [[1, 2, 0, 1, 1, 1], [2, 1, 1, 0, 1, 1]], dtype=np.uint16
    )
    metrics = _weighted_metric_distribution(truth, score, weights, chunk_size=1)
    for index, current in enumerate(weights):
        expanded = np.repeat(np.arange(len(truth)), current)
        assert metrics["auroc"][index] == pytest.approx(
            roc_auc_score(truth[expanded], score[expanded])
        )
        assert metrics["auprc"][index] == pytest.approx(
            average_precision_score(truth[expanded], score[expanded])
        )
        assert metrics["brier"][index] == pytest.approx(
            np.mean((truth[expanded] - score[expanded]) ** 2)
        )


def test_public_reliability_curve_suppresses_bins_below_n10() -> None:
    truth = np.asarray([0, 1] * 10, dtype=np.int64)
    score = np.asarray([0.05] * 9 + [0.25] * 10 + [0.95], dtype=np.float64)
    x, y = _reliability_points(truth, score)
    assert np.array_equal(x, np.asarray([0.25]))
    assert np.array_equal(y, np.asarray([np.mean(truth[9:19])]))


def _synthetic_lineage() -> dict[str, object]:
    baseline_artifacts = {
        role: character * 64
        for role, character in zip(
            ("predictions", "selection", "metrics", "progress"),
            "abcd",
            strict=True,
        )
    }
    probe_roles = (
        "phenotype_predictions",
        "phenotype_selection",
        "phenotype_metrics",
        "subtype_predictions",
        "subtype_selection",
        "subtype_metrics",
        "ftv_predictions",
        "ftv_selection",
        "ftv_metrics",
        "progress",
    )
    probe_artifacts = {
        role: f"{index + 10:064x}" for index, role in enumerate(probe_roles)
    }
    return {
        "baseline_v2": {
            "protocol_version": "v2",
            "evaluation_lock_sha256": "1" * 64,
            "run_receipt_sha256": "2" * 64,
            "argv_sha256": "3" * 64,
            "artifact_sha256": baseline_artifacts,
        },
        "probe_v3": {
            "protocol_version": "v3",
            "evaluation_lock_sha256": "4" * 64,
            "run_receipt_sha256": "5" * 64,
            "argv_sha256": "6" * 64,
            "artifact_sha256": probe_artifacts,
        },
        "summarizer": {
            "protocol_version": "v3",
            "argv_sha256": "7" * 64,
            "code_lock_sha256": "8" * 64,
            "finalization_lock_sha256": "9" * 64,
        },
    }


def test_reporting_marker_schema_hashes_and_marker_last_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged = {}
    for field in (
        "paired_csv",
        "summary_json",
        "summary_markdown",
        "timing_figure",
        "calibration_figure",
        "reporting_marker",
    ):
        path = tmp_path / "staged" / field
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((field + "\n").encode())
        staged[field] = path
    marker = _build_reporting_marker(_synthetic_lineage(), staged)
    assert set(marker) == {
        "schema_version",
        "summary_schema_version",
        "comparison_contract_canonical_sha256",
        "baseline_v2",
        "probe_v3",
        "summarizer",
        "public_artifact_sha256",
    }
    assert marker["public_artifact_sha256"]["paired_public"] == file_sha256(
        staged["paired_csv"]
    )

    destinations = {
        field: tmp_path / "published" / field for field in staged
    }
    original_link = __import__("os").link
    calls: list[str] = []

    def fail_on_marker(source: Path, destination: Path) -> None:
        calls.append(Path(source).name)
        if Path(source).name == "reporting_marker":
            raise OSError("synthetic marker publish failure")
        original_link(source, destination)

    monkeypatch.setattr("foundation_mri.reporting.os.link", fail_on_marker)
    with pytest.raises(OSError, match="synthetic marker publish failure"):
        _publish_no_overwrite_with_rollback(
            staged,
            destinations,
            marker_field="reporting_marker",
        )
    assert calls[-1] == "reporting_marker"
    assert not any(path.exists() for path in destinations.values())


def test_end_to_end_reporting_is_deterministic_and_public(tmp_path: Path) -> None:
    inputs = _synthetic_outputs(tmp_path / "inputs")
    first = _outputs(tmp_path / "first")
    second = _outputs(tmp_path / "second")
    summary = summarize_results(
        inputs, first, full_size=FULL_N, complete_size=COMPLETE_N
    )
    summarize_results(inputs, second, full_size=FULL_N, complete_size=COMPLETE_N)
    assert summary == {
        "foundation_models": 1,
        "resolved_comparisons": 78,
        "paired_metric_rows": 234,
        "public_outputs": 5,
    }
    paired = pd.read_csv(first.paired_csv)
    assert "patient_id" not in paired.columns
    assert set(paired["metric"]) == {"auroc", "auprc", "brier"}
    assert set(paired["bootstrap_seed"]) == {BOOTSTRAP_SEED}
    assert set(paired["bootstrap_replicates"]) == {BOOTSTRAP_REPLICATES}
    assert first.paired_csv.read_bytes() == second.paired_csv.read_bytes()
    assert first.summary_json.read_bytes() == second.summary_json.read_bytes()
    for path in (
        first.paired_csv,
        first.summary_json,
        first.summary_markdown,
        first.timing_figure,
        first.calibration_figure,
    ):
        assert path.is_file() and path.stat().st_size > 100
    assert first.timing_figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert first.calibration_figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert "patient_id" not in first.summary_json.read_text(encoding="utf-8")
    assert "patient_id" not in first.summary_markdown.read_text(encoding="utf-8")


def test_public_aggregate_drift_fails_closed(tmp_path: Path) -> None:
    inputs = _synthetic_outputs(tmp_path / "inputs")
    public = pd.read_csv(inputs.baseline_public)
    public.loc[0, "auroc"] = float(public.loc[0, "auroc"]) - 0.01
    public.to_csv(inputs.baseline_public, index=False)
    with pytest.raises(ValueError, match="public aggregate drifted"):
        summarize_results(
            inputs,
            _outputs(tmp_path / "outputs"),
            full_size=FULL_N,
            complete_size=COMPLETE_N,
        )


def test_missing_preregistered_cell_fails_closed(tmp_path: Path) -> None:
    inputs = _synthetic_outputs(tmp_path / "inputs")
    private = pd.read_csv(inputs.baseline_private, dtype={"patient_id": str})
    missing = (
        private["model"].eq(f"{FOUNDATION}_mri_clinical_ftv")
        & private["spatial"].eq("LOCAL")
        & private["timing"].eq("T0-T2")
    )
    private = private.loc[~missing].copy()
    private.to_csv(inputs.baseline_private, index=False)
    aggregate_binary_predictions(private).to_csv(inputs.baseline_public, index=False)
    with pytest.raises(ValueError, match="missing preregistered comparison cell"):
        summarize_results(
            inputs,
            _outputs(tmp_path / "outputs"),
            full_size=FULL_N,
            complete_size=COMPLETE_N,
        )


def test_cli_help_and_contract_do_not_read_formal_inputs() -> None:
    script = ROOT / "scripts/summarize_results.py"
    help_result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--print-comparison-contract" in help_result.stdout
    contract_result = subprocess.run(
        [sys.executable, str(script), "--print-comparison-contract"],
        check=True,
        capture_output=True,
        text=True,
    )
    contract = json.loads(contract_result.stdout)
    assert contract == static_comparison_contract()
    assert contract["bootstrap"]["master_seed"] == BOOTSTRAP_SEED
    assert contract["bootstrap"]["replicates"] == BOOTSTRAP_REPLICATES

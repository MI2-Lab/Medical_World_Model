from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import pytest

import foundation_mri.reporting as base_reporting
from foundation_mri.evaluation import (
    aggregate_binary_predictions,
    aggregate_continuous_predictions,
    aggregate_multiclass_predictions,
)
from foundation_mri.reporting import paired_bootstrap_comparisons
import foundation_mri_dinov3.reporting as reporting
from foundation_mri_dinov3.reporting import (
    DINO_V1_MODEL,
    EXPECTED_COMPARISON_COUNT,
    EXPECTED_PAIRED_METRIC_ROWS,
    FAMILY_COUNTS,
    MODEL_NAME,
    ReportingInputs,
    ReportingOutputs,
    build_comparison_specs,
    summarize_results,
)


FOLD_HASH = "a" * 64
FULL_N = 40
COMPLETE_N = 20
FULL_POPULATION = f"full_{FULL_N}"
COMPLETE_POPULATION = f"radiomics_complete_case_{COMPLETE_N}"
TIMINGS = ("T0", "T0-T1", "T0-T2")
AXES = ("GLOBAL", "LOCAL")
TEMPLATE = Path(__file__).resolve().parents[1] / "reports/final_report.template.md"


@dataclass(frozen=True)
class SyntheticBundle:
    inputs: ReportingInputs
    new_baseline: pd.DataFrame
    old_baseline: pd.DataFrame


def _score(truth: np.ndarray, label: str) -> np.ndarray:
    phase = (sum(ord(character) for character in label) % 31) / 9.0
    linear = -0.35 + 0.70 * truth + 0.62 * np.sin(
        np.arange(len(truth), dtype=np.float64) * 0.73 + phase
    )
    return 1.0 / (1.0 + np.exp(-linear))


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
            "fold": np.arange(len(patient_ids), dtype=np.int64) % 5,
            "split": "test",
            "y_true": truth,
            "y_score": _score(
                truth.astype(np.float64),
                f"{target}|{model}|{spatial}|{timing}|{population}",
            ),
        }
    )


def _new_baseline(full_ids: np.ndarray, complete_ids: np.ndarray) -> pd.DataFrame:
    full_truth = ((np.arange(len(full_ids)) // 5) % 2).astype(np.int64)
    complete_truth = full_truth[: len(complete_ids)]
    rows: list[pd.DataFrame] = []
    variants = (
        ("mri_only", FULL_POPULATION, full_ids, full_truth),
        ("mri_clinical", FULL_POPULATION, full_ids, full_truth),
        ("mri_only_paired", COMPLETE_POPULATION, complete_ids, complete_truth),
        ("mri_clinical_paired", COMPLETE_POPULATION, complete_ids, complete_truth),
        ("mri_ftv", COMPLETE_POPULATION, complete_ids, complete_truth),
        ("mri_clinical_ftv", COMPLETE_POPULATION, complete_ids, complete_truth),
    )
    for timing in TIMINGS:
        for suffix, population, ids, truth in variants:
            for spatial in AXES:
                rows.append(
                    _binary_rows(
                        ids,
                        truth,
                        target="pCR",
                        model=f"{MODEL_NAME}_{suffix}",
                        spatial=spatial,
                        timing=timing,
                        population=population,
                    )
                )
    return pd.concat(rows, ignore_index=True)


def _old_baseline(full_ids: np.ndarray, complete_ids: np.ndarray) -> pd.DataFrame:
    full_truth = ((np.arange(len(full_ids)) // 5) % 2).astype(np.int64)
    complete_truth = full_truth[: len(complete_ids)]
    rows: list[pd.DataFrame] = []
    variants = (
        ("mri_only", FULL_POPULATION, full_ids, full_truth),
        ("mri_clinical", FULL_POPULATION, full_ids, full_truth),
        ("mri_only_paired", COMPLETE_POPULATION, complete_ids, complete_truth),
        ("mri_clinical_paired", COMPLETE_POPULATION, complete_ids, complete_truth),
        ("mri_ftv", COMPLETE_POPULATION, complete_ids, complete_truth),
        ("mri_clinical_ftv", COMPLETE_POPULATION, complete_ids, complete_truth),
    )
    for timing in TIMINGS:
        for suffix, population, ids, truth in variants:
            for spatial in AXES:
                rows.append(
                    _binary_rows(
                        ids,
                        truth,
                        target="pCR",
                        model=f"{DINO_V1_MODEL}_{suffix}",
                        spatial=spatial,
                        timing=timing,
                        population=population,
                    )
                )
        for suffix in ("mri_only", "mri_clinical"):
            rows.extend(
                (
                    _binary_rows(
                        full_ids,
                        full_truth,
                        target="pCR",
                        model=f"GAP0_{suffix}",
                        spatial="GLOBAL",
                        timing=timing,
                        population=FULL_POPULATION,
                    ),
                    _binary_rows(
                        full_ids,
                        full_truth,
                        target="pCR",
                        model=f"LOCAL0_{suffix}",
                        spatial="LOCAL",
                        timing=timing,
                        population=FULL_POPULATION,
                    ),
                )
            )
        rows.extend(
            (
                _binary_rows(
                    full_ids,
                    full_truth,
                    target="pCR",
                    model="clinical_only",
                    spatial="NONE",
                    timing=timing,
                    population=FULL_POPULATION,
                ),
                _binary_rows(
                    complete_ids,
                    complete_truth,
                    target="pCR",
                    model="clinical_only_paired",
                    spatial="NONE",
                    timing=timing,
                    population=COMPLETE_POPULATION,
                ),
                _binary_rows(
                    complete_ids,
                    complete_truth,
                    target="pCR",
                    model="clinical_ftv",
                    spatial="TABULAR",
                    timing=timing,
                    population=COMPLETE_POPULATION,
                ),
            )
        )
    return pd.concat(rows, ignore_index=True)


def _phenotype(patient_ids: np.ndarray, model: str) -> pd.DataFrame:
    block = np.arange(len(patient_ids)) // 5
    truths = {
        "HR": (block % 2).astype(np.int64),
        "HER2": ((block // 2) % 2).astype(np.int64),
    }
    return pd.concat(
        [
            _binary_rows(
                patient_ids,
                truth,
                target=target,
                model=model,
                spatial=spatial,
                timing="T0",
                population=FULL_POPULATION,
            )
            for target, truth in truths.items()
            for spatial in AXES
        ],
        ignore_index=True,
    )


def _subtype(patient_ids: np.ndarray, model: str, confidence: float) -> pd.DataFrame:
    classes = tuple(
        sorted(("HR-/HER2-", "HR-/HER2+", "HR+/HER2-", "HR+/HER2+"))
    )
    class_index = (np.arange(len(patient_ids)) // 5) % len(classes)
    truth = np.asarray([classes[index] for index in class_index], dtype=str)
    rows: list[pd.DataFrame] = []
    for spatial in AXES:
        probability = np.full(
            (len(patient_ids), len(classes)),
            (1.0 - confidence) / (len(classes) - 1),
            dtype=np.float64,
        )
        probability[np.arange(len(patient_ids)), class_index] = confidence
        rows.append(
            pd.DataFrame(
                {
                    "patient_id": patient_ids,
                    "target": "HR_HER2_subtype",
                    "model": model,
                    "spatial": spatial,
                    "timing": "T0",
                    "analysis_population": FULL_POPULATION,
                    "split_seed": 2026,
                    "fold_manifest_sha256": FOLD_HASH,
                    "fold": np.arange(len(patient_ids), dtype=np.int64) % 5,
                    "split": "test",
                    "y_true": truth,
                    "classes_json": json.dumps(classes, separators=(",", ":")),
                    "probabilities_json": [
                        json.dumps(values.tolist(), separators=(",", ":"))
                        for values in probability
                    ],
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _ftv(patient_ids: np.ndarray, model: str, noise_scale: float) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    index = np.arange(len(patient_ids), dtype=np.float64)
    for spatial in AXES:
        for task, endpoints in (
            ("static", ("T0", "T1", "T2", "T3")),
            ("delta", ("T0-T1", "T1-T2", "T2-T3")),
        ):
            for endpoint_index, endpoint in enumerate(endpoints):
                truth = 2.0 + 0.13 * index + 0.2 * endpoint_index
                prediction = truth + noise_scale * np.sin(
                    index * 0.91 + endpoint_index + (0.4 if spatial == "LOCAL" else 0.0)
                )
                rows.append(
                    pd.DataFrame(
                        {
                            "patient_id": patient_ids,
                            "target": "FTV",
                            "model": model,
                            "spatial": spatial,
                            "task": task,
                            "endpoint": endpoint,
                            "analysis_population": COMPLETE_POPULATION,
                            "split_seed": 2026,
                            "fold_manifest_sha256": FOLD_HASH,
                            "fold": np.arange(len(patient_ids), dtype=np.int64) % 5,
                            "split": "test",
                            "y_true": truth,
                            "y_pred": prediction,
                            "train_mean_baseline": float(np.mean(truth)),
                        }
                    )
                )
    return pd.concat(rows, ignore_index=True)


def _write_csv(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _synthetic_bundle(root: Path) -> SyntheticBundle:
    full_ids = np.asarray([f"SYN{index:04d}" for index in range(FULL_N)], dtype=str)
    complete_ids = full_ids[:COMPLETE_N]
    new_baseline = _new_baseline(full_ids, complete_ids)
    old_baseline = _old_baseline(full_ids, complete_ids)
    phenotype = _phenotype(full_ids, MODEL_NAME)
    old_phenotype = _phenotype(full_ids, DINO_V1_MODEL)
    subtype = _subtype(full_ids, MODEL_NAME, 0.62)
    old_subtype = _subtype(full_ids, DINO_V1_MODEL, 0.54)
    ftv = _ftv(complete_ids, MODEL_NAME, 0.12)
    old_ftv = _ftv(complete_ids, DINO_V1_MODEL, 0.20)

    inputs = ReportingInputs(
        new_baseline_private=_write_csv(
            root / "new/predictions/dinov3_baseline_predictions.private.csv",
            new_baseline,
        ),
        new_baseline_public=_write_csv(
            root / "new/metrics/dinov3_baseline_metrics.csv",
            aggregate_binary_predictions(new_baseline),
        ),
        new_phenotype_private=_write_csv(
            root / "new/predictions/dinov3_phenotype_predictions.private.csv",
            phenotype,
        ),
        new_phenotype_public=_write_csv(
            root / "new/metrics/dinov3_phenotype_metrics.csv",
            aggregate_binary_predictions(phenotype),
        ),
        new_subtype_private=_write_csv(
            root / "new/predictions/dinov3_subtype_predictions.private.csv",
            subtype,
        ),
        new_subtype_public=_write_csv(
            root / "new/metrics/dinov3_subtype_metrics.csv",
            aggregate_multiclass_predictions(subtype),
        ),
        new_ftv_private=_write_csv(
            root / "new/predictions/dinov3_ftv_predictions.private.csv",
            ftv,
        ),
        new_ftv_public=_write_csv(
            root / "new/metrics/dinov3_ftv_metrics.csv",
            aggregate_continuous_predictions(ftv),
        ),
        old_baseline_private=_write_csv(
            root / "old/predictions/baseline_predictions.private.csv",
            old_baseline,
        ),
        old_baseline_public=_write_csv(
            root / "old/metrics/baseline_metrics.csv",
            aggregate_binary_predictions(old_baseline),
        ),
        old_phenotype_public=_write_csv(
            root / "old/metrics/phenotype_metrics.csv",
            aggregate_binary_predictions(old_phenotype),
        ),
        old_subtype_public=_write_csv(
            root / "old/metrics/subtype_metrics.csv",
            aggregate_multiclass_predictions(old_subtype),
        ),
        old_ftv_public=_write_csv(
            root / "old/metrics/ftv_probe_metrics.csv",
            aggregate_continuous_predictions(old_ftv),
        ),
    )
    return SyntheticBundle(inputs, new_baseline, old_baseline)


def _outputs(root: Path) -> ReportingOutputs:
    return ReportingOutputs(
        paired_csv=root / "metrics/paired_bootstrap_comparisons.csv",
        summary_json=root / "metrics/results_summary.json",
        final_report=root / "reports/final_report.md",
        timing_figure=root / "figures/pcr_timing_performance.png",
        comparison_figure=root / "figures/paired_comparison_deltas.png",
        reporting_marker=root / "metrics/reporting_run_provenance.json",
    )


def test_comparison_expansion_is_identity_only_and_exact(tmp_path: Path) -> None:
    bundle = _synthetic_bundle(tmp_path)
    identities = pd.concat(
        (bundle.new_baseline, bundle.old_baseline), ignore_index=True
    ).loc[:, ["target", "model", "spatial", "timing", "analysis_population"]]
    specs = build_comparison_specs(
        identities,
        full_population=FULL_POPULATION,
        complete_population=COMPLETE_POPULATION,
    )
    assert len(specs) == EXPECTED_COMPARISON_COUNT == 84
    assert Counter(spec.family for spec in specs) == Counter(FAMILY_COUNTS)
    assert len({spec.comparison_id for spec in specs}) == 84
    assert all(spec.timing in TIMINGS for spec in specs)

    missing = identities.loc[
        ~(
            identities["model"].eq(f"{MODEL_NAME}_mri_only")
            & identities["spatial"].eq("GLOBAL")
            & identities["timing"].eq("T0")
        )
    ]
    with pytest.raises(ValueError, match="missing frozen comparison cell"):
        build_comparison_specs(
            missing,
            full_population=FULL_POPULATION,
            complete_population=COMPLETE_POPULATION,
        )


def test_paired_comparison_rejects_fold_misalignment(tmp_path: Path) -> None:
    bundle = _synthetic_bundle(tmp_path)
    combined = pd.concat(
        (bundle.new_baseline, bundle.old_baseline), ignore_index=True
    )
    specs = build_comparison_specs(
        combined.loc[
            :, ["target", "model", "spatial", "timing", "analysis_population"]
        ],
        full_population=FULL_POPULATION,
        complete_population=COMPLETE_POPULATION,
    )
    first = specs[0]
    mask = (
        combined["model"].eq(first.candidate_model)
        & combined["spatial"].eq(first.candidate_spatial)
        & combined["timing"].eq(first.timing)
        & combined["analysis_population"].eq(first.analysis_population)
    )
    row = combined.index[mask][0]
    combined.loc[row, "fold"] = (int(combined.loc[row, "fold"]) + 1) % 5
    with pytest.raises(ValueError, match="paired outer folds differ"):
        paired_bootstrap_comparisons(
            combined,
            (first,),
            patient_orders={
                FULL_POPULATION: tuple(sorted(combined.loc[
                    combined["analysis_population"].eq(FULL_POPULATION), "patient_id"
                ].unique())),
                COMPLETE_POPULATION: tuple(sorted(combined.loc[
                    combined["analysis_population"].eq(COMPLETE_POPULATION), "patient_id"
                ].unique())),
            },
        )


def test_public_aggregate_drift_fails_before_reporting(tmp_path: Path) -> None:
    bundle = _synthetic_bundle(tmp_path)
    public = pd.read_csv(bundle.inputs.new_baseline_public)
    public.loc[0, "auroc"] = float(public.loc[0, "auroc"]) + 1e-4
    public.to_csv(bundle.inputs.new_baseline_public, index=False)
    with pytest.raises(ValueError, match="auroc"):
        reporting._load_and_validate(  # noqa: SLF001
            bundle.inputs, full_size=FULL_N, complete_size=COMPLETE_N
        )


def test_end_to_end_counts_matched_probes_privacy_and_register_caveat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _synthetic_bundle(tmp_path / "inputs")
    outputs = _outputs(tmp_path / "public")
    monkeypatch.setattr(base_reporting, "BOOTSTRAP_REPLICATES", 200)
    monkeypatch.setattr(reporting, "BOOTSTRAP_REPLICATES", 200)
    counts = summarize_results(
        bundle.inputs,
        outputs,
        template_path=TEMPLATE,
        full_size=FULL_N,
        complete_size=COMPLETE_N,
    )
    assert counts == {
        "comparison_specs": 84,
        "paired_metric_rows": 252,
        "pcr_pooled_cells": 36,
        "public_outputs": 6,
        "reporting_marker_published_last": 1,
    }
    paired = pd.read_csv(outputs.paired_csv)
    assert len(paired) == EXPECTED_PAIRED_METRIC_ROWS
    assert Counter(paired["family"]) == Counter(
        {family: count * 3 for family, count in FAMILY_COUNTS.items()}
    )
    assert "patient_id" not in paired.columns

    summary = json.loads(outputs.summary_json.read_text(encoding="utf-8"))
    assert summary["coverage"]["comparison_specs"] == 84
    assert summary["coverage"]["paired_metric_rows"] == 252
    assert summary["coverage"]["pcr_pooled_cells"] == 36
    assert summary["coverage"]["matched_phenotype_cells"] == 4
    assert summary["coverage"]["matched_subtype_cells"] == 2
    assert summary["coverage"]["matched_ftv_cells"] == 14
    matched = summary["dinov3_vs_dinov1_probe_descriptive"]
    assert len(matched["phenotype"]) == 4
    assert len(matched["subtype"]) == 2
    assert len(matched["ftv"]) == 14
    assert "delta_candidate_minus_reference_auroc" in matched["phenotype"][0]
    assert matched["inference"].endswith("no CI")

    report = outputs.final_report.read_text(encoding="utf-8")
    for phrase in (
        "post-hoc",
        "custom license",
        "unknown",
        "原实验",
        "register tokens",
        "[1:5]",
        "[5:201]",
        "1536",
        "best-cell filtering",
    ):
        assert phrase in report
    assert report.count("dinov3_vs_dinov1") >= 1
    assert report.count("local_vs_global") >= 1
    assert report.count("dinov3_vs_current_cnn_full") >= 1
    assert report.count("clinical_gain") >= 1
    assert report.count("beyond_ftv") >= 1

    marker = json.loads(outputs.reporting_marker.read_text(encoding="utf-8"))
    assert marker["published_last"] is True
    assert marker["lineage_mode"] == "synthetic"
    assert len(marker["public_artifact_sha256"]) == 5
    for figure in (outputs.timing_figure, outputs.comparison_figure):
        assert figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    public_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            outputs.paired_csv,
            outputs.summary_json,
            outputs.final_report,
            outputs.reporting_marker,
        )
    )
    assert "patient_id" not in public_text.lower()
    assert ".private." not in public_text.lower()
    assert str(tmp_path) not in public_text
    assert not any(f"SYN{index:04d}" in public_text for index in range(FULL_N))

    with pytest.raises(FileExistsError, match="reporting outputs already exist"):
        summarize_results(
            bundle.inputs,
            outputs,
            template_path=TEMPLATE,
            full_size=FULL_N,
            complete_size=COMPLETE_N,
        )


def test_marker_last_failure_rolls_back_every_public_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fields = (
        "paired_csv",
        "summary_json",
        "final_report",
        "timing_figure",
        "comparison_figure",
        "reporting_marker",
    )
    staged = {field: tmp_path / "stage" / field for field in fields}
    destinations = {field: tmp_path / "dest" / field for field in fields}
    for path in staged.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name, encoding="utf-8")
    real_link = reporting.os.link
    publication_order: list[str] = []

    def failing_link(source: Path, destination: Path) -> None:
        publication_order.append(Path(destination).name)
        if Path(destination) == destinations["reporting_marker"]:
            raise OSError("synthetic marker publication failure")
        real_link(source, destination)

    monkeypatch.setattr(reporting.os, "link", failing_link)
    with pytest.raises(OSError, match="marker publication failure"):
        reporting._publish_no_overwrite_with_rollback(  # noqa: SLF001
            staged, destinations, marker_field="reporting_marker"
        )
    assert publication_order[-1] == "reporting_marker"
    assert not any(path.exists() for path in destinations.values())


def test_cli_report_lock_is_first_experiment_api_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/summarize_results.py"
    spec = importlib.util.spec_from_file_location("synthetic_dinov3_summarizer", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert len(module.EXCLUSIVE_OUTPUTS) == 22
    assert set(module.EXCLUSIVE_OUTPUTS) == {
        "baseline_predictions",
        "baseline_selection",
        "baseline_metrics",
        "baseline_progress",
        "baseline_receipt",
        "probe_phenotype_predictions",
        "probe_phenotype_selection",
        "probe_phenotype_metrics",
        "probe_subtype_predictions",
        "probe_subtype_selection",
        "probe_subtype_metrics",
        "probe_ftv_predictions",
        "probe_ftv_selection",
        "probe_ftv_metrics",
        "probe_progress",
        "probe_receipt",
        "report_paired_comparisons",
        "report_results_summary",
        "report_final_report",
        "report_pcr_timing_figure",
        "report_paired_comparison_figure",
        "report_reporting_marker",
    }
    module._verify_exclusive_outputs(dict(module.EXCLUSIVE_OUTPUTS))  # noqa: SLF001
    with pytest.raises(ValueError, match="exclusive output contract drifted"):
        module._verify_exclusive_outputs({})  # noqa: SLF001

    calls: list[tuple[str, tuple[str, ...]]] = []
    fake = types.ModuleType("foundation_mri_dinov3.locking")

    class GateReached(RuntimeError):
        pass

    def verify_evaluation_lock(consumer: str, argv: tuple[str, ...]) -> None:
        calls.append((consumer, tuple(argv)))
        raise GateReached

    fake.verify_evaluation_lock = verify_evaluation_lock  # type: ignore[attr-defined]
    fake.verify_producer_receipt = lambda consumer: pytest.fail(  # type: ignore[attr-defined]
        f"receipt ran before report gate: {consumer}"
    )
    fake.load_json = lambda path: pytest.fail(  # type: ignore[attr-defined]
        f"lock JSON ran before report gate: {path}"
    )
    monkeypatch.setitem(sys.modules, "foundation_mri_dinov3.locking", fake)
    with pytest.raises(GateReached):
        module.main(("unexpected",))
    assert calls == [("report", ("unexpected",))]

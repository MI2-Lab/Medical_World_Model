from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_PREFIX = Path("additional_experiments/foundation_mri_baselines")
sys.path.insert(0, str(ROOT / "src"))

from foundation_mri import finalization as fm  # noqa: E402
from foundation_mri.finalization import (  # noqa: E402
    FinalizationInputs,
    FinalizationOutputs,
    finalize_report,
    fixed_question_evidence,
    load_contract,
    render_final_report,
    validate_public_results,
)
from foundation_mri.reporting import static_comparison_contract  # noqa: E402


CONTRACT_PATH = ROOT / "configs/final_report_contract.json"
TEMPLATE_PATH = ROOT / "reports/final_report.template.md"
MODULE_PATH = ROOT / "src/foundation_mri/finalization.py"
CLI_PATH = ROOT / "scripts/finalize_report.py"
CONTRACT = load_contract(CONTRACT_PATH)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", double_precision=15))


def _model_score(model: str, spatial: str) -> float:
    if model.startswith("clinical_only"):
        return 0.52
    if model in {"ftv_only", "radiomics_only", "clinical_ftv", "clinical_radiomics"}:
        return 0.54
    if model.startswith("GAP0_") or model == "GAP0":
        return 0.58
    if model.startswith("LOCAL0_") or model == "LOCAL0":
        return 0.62
    if model.startswith("medicalnet_resnet50_3dseg8") or model.startswith(
        "dino_vitb16_imagenet1k"
    ):
        return 0.70 if spatial == "LOCAL" else 0.66
    raise AssertionError(f"unhandled synthetic model: {model}")


def _binary_frame(identities: set[tuple[str, ...]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target, model, spatial, timing, population in sorted(identities):
        score = _model_score(model, spatial)
        if target == "HR":
            positive = 410
        elif target == "HER2":
            positive = 125
        else:
            positive = 275 if population == "full_808" else 110
        rows.append(
            {
                "target": target,
                "model": model,
                "spatial": spatial,
                "timing": timing,
                "analysis_population": population,
                "split_seed": 2026,
                "fold_manifest_sha256": CONTRACT["fold_manifest_sha256"],
                "aggregation": "pooled_oof",
                "n": 808 if population == "full_808" else 375,
                "positive": positive,
                "n_folds": 5,
                "ece_bin_contract": "10_equal_width_bins_[0,1]",
                "auroc": score,
                "auprc": score - 0.15,
                "brier": 0.75 - score,
                "calibration_slope": 0.92,
                "calibration_intercept": -0.04,
                "ece_10bin": 0.05,
            }
        )
    return pd.DataFrame(rows)


def _subtype_frame() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target, model, spatial, timing, population in sorted(
        fm._expected_subtype_identities(CONTRACT)
    ):
        rows.append(
            {
                "target": target,
                "model": model,
                "spatial": spatial,
                "timing": timing,
                "analysis_population": population,
                "split_seed": 2026,
                "fold_manifest_sha256": CONTRACT["fold_manifest_sha256"],
                "aggregation": "pooled_oof",
                "n": 808,
                "n_folds": 5,
                "ece_bin_contract": "10_equal_width_bins_[0,1]_top_label",
                "macro_ovr_auroc": 0.63,
                "macro_ovr_auprc": 0.45,
                "multiclass_brier": 0.56,
                "toplabel_ece_10bin": 0.07,
                "accuracy": 0.47,
            }
        )
    return pd.DataFrame(rows)


def _ftv_frame() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target, model, spatial, task, endpoint, population in sorted(
        fm._expected_ftv_identities(CONTRACT)
    ):
        rows.append(
            {
                "target": target,
                "model": model,
                "spatial": spatial,
                "task": task,
                "endpoint": endpoint,
                "analysis_population": population,
                "split_seed": 2026,
                "fold_manifest_sha256": CONTRACT["fold_manifest_sha256"],
                "aggregation": "pooled_oof",
                "n": 375,
                "n_folds": 5,
                "spearman": 0.41,
                "pearson": 0.43,
                "r2": 0.16,
                "rmse": 3.2,
                "mae": 2.4,
                "b0_rmse": 3.8,
                "rmse_gain_over_b0": 0.15789473684210525,
                "calibration_slope": 0.85,
                "calibration_intercept": 0.3,
                "calibration_mean_bias": -0.1,
            }
        )
    return pd.DataFrame(rows)


def _paired_frame(pcr: pd.DataFrame) -> pd.DataFrame:
    lookup = {
        (
            str(row.model),
            str(row.spatial),
            str(row.timing),
            str(row.analysis_population),
            metric,
        ): float(getattr(row, metric))
        for row in pcr.itertuples(index=False)
        for metric in fm._METRICS
    }
    rows: list[dict[str, Any]] = []
    for spec in fm._expected_comparison_specs(CONTRACT):
        for metric in fm._METRICS:
            reference = lookup[
                (
                    spec["reference_model"],
                    spec["reference_spatial"],
                    spec["timing"],
                    spec["analysis_population"],
                    metric,
                )
            ]
            candidate = lookup[
                (
                    spec["candidate_model"],
                    spec["candidate_spatial"],
                    spec["timing"],
                    spec["analysis_population"],
                    metric,
                )
            ]
            delta = candidate - reference
            rows.append(
                {
                    **spec,
                    "metric": metric,
                    "higher_is_better": metric != "brier",
                    "reference_value": reference,
                    "candidate_value": candidate,
                    "delta_candidate_minus_reference": delta,
                    "ci_level": 0.95,
                    "ci_low": delta - 0.005,
                    "ci_high": delta + 0.005,
                    "bootstrap_seed": 2026,
                    "bootstrap_replicates": 5000,
                    "valid_bootstrap_replicates": 5000,
                    "n_paired": 808
                    if spec["analysis_population"] == "full_808"
                    else 375,
                    "positive": 275
                    if spec["analysis_population"] == "full_808"
                    else 110,
                    "inference_scope": CONTRACT["inference_scope"],
                }
            )
    return pd.DataFrame(rows)


def _public_pair(pooled: pd.DataFrame) -> pd.DataFrame:
    macro = pooled.copy()
    macro["aggregation"] = "outer_fold_macro"
    return pd.concat((pooled, macro), ignore_index=True)


@dataclass
class FakeGit:
    repository_root: Path
    content_sha: str = "1" * 40
    branch: str = "feature/foundation-mri-baselines"
    remote_sha: str = "1" * 40
    committed: dict[str, bytes] | None = None
    on_root: Callable[[], None] | None = None
    calls: list[tuple[str, ...]] | None = None

    def __call__(self, arguments: Sequence[str], cwd: Path) -> bytes:
        args = tuple(arguments)
        if self.calls is None:
            self.calls = []
        self.calls.append(args)
        if args == ("rev-parse", "--show-toplevel"):
            callback, self.on_root = self.on_root, None
            if callback is not None:
                callback()
            return (str(self.repository_root) + "\n").encode("utf-8")
        if args == ("rev-parse", "HEAD"):
            return (self.content_sha + "\n").encode("ascii")
        if args == ("branch", "--show-current"):
            return (self.branch + "\n").encode("utf-8")
        if args[:2] == ("ls-remote", "--heads"):
            return (
                f"{self.remote_sha}\trefs/heads/feature/foundation-mri-baselines\n"
            ).encode("ascii")
        if len(args) == 2 and args[0] == "show":
            relative = args[1].split(":", 1)[1]
            assert self.committed is not None
            if relative not in self.committed:
                raise ValueError(f"synthetic content commit lacks {relative}")
            return self.committed[relative]
        raise AssertionError(f"unexpected synthetic git call: {args}; cwd={cwd}")


@dataclass
class Bundle:
    repository_root: Path
    paths: dict[str, Path]
    inputs: FinalizationInputs
    outputs: FinalizationOutputs
    git: FakeGit
    status: str = "SUBSTANTIVE_PUSH_OK"
    error: str | None = None

    def commit_current(self) -> None:
        tracked = list(map(str, CONTRACT["git_handoff"]["tracked_artifact_roles"]))
        fixed = CONTRACT["fixed_public_paths"]
        committed = {str(fixed[role]): self.paths[role].read_bytes() for role in tracked}
        self.git.committed = committed
        artifact_hashes = {
            relative: hashlib.sha256(payload).hexdigest()
            for relative, payload in committed.items()
        }
        success = self.status == "SUBSTANTIVE_PUSH_OK"
        manifest = {
            "schema_version": fm.GIT_HANDOFF_SCHEMA_VERSION,
            "content_commit_sha": self.git.content_sha,
            "branch": CONTRACT["git_handoff"]["branch"],
            "remote": CONTRACT["git_handoff"]["remote"],
            "remote_ref": CONTRACT["git_handoff"]["remote_ref"],
            "substantive_push_status": self.status,
            "substantive_remote_ref_sha": self.git.content_sha if success else None,
            "sanitized_push_error": None if success else self.error,
            "artifact_sha256": artifact_hashes,
        }
        _write_json(self.paths["git_handoff_json"], manifest)

    def reseal(self) -> None:
        summary = _read_json(self.paths["summary_json"])
        baseline_private = _digest("baseline-private-predictions")
        phenotype_private = _digest("phenotype-private-predictions")
        subtype_private = _digest("subtype-private-predictions")
        ftv_private = _digest("ftv-private-predictions")
        summary["input_sha256"] = {
            "baseline_private": baseline_private,
            "baseline_public": fm.file_sha256(self.paths["baseline_public"]),
            "phenotype_private": phenotype_private,
            "phenotype_public": fm.file_sha256(self.paths["phenotype_public"]),
            "subtype_private": subtype_private,
            "subtype_public": fm.file_sha256(self.paths["subtype_public"]),
            "ftv_private": ftv_private,
            "ftv_public": fm.file_sha256(self.paths["ftv_public"]),
        }
        _write_json(self.paths["summary_json"], summary)

        locked_roles = {
            "finalization_module",
            "finalizer_cli",
            "contract_json",
            "template_markdown",
            "finalization_test",
        }
        finalization_lock = {
            "schema_version": "foundation_mri_finalization_lock_v1",
            "parent_evaluation_lock_sha256": (
                "b15f7023b7021f5c1169b51cf6bc8fe0cc1d9085102a61fbdb1d68589fe2edc5"
            ),
            "formal_metric_values_unseen": True,
            "formal_argv": [],
            "locked_sha256": {
                role: fm.file_sha256(self.paths[role]) for role in sorted(locked_roles)
            },
            "fixed_public_paths": CONTRACT["fixed_public_paths"],
            "summary_schema_version": fm.SUMMARY_SCHEMA_VERSION,
            "reporting_provenance_schema_version": (
                fm.REPORTING_PROVENANCE_SCHEMA_VERSION
            ),
            "git_handoff_schema_version": fm.GIT_HANDOFF_SCHEMA_VERSION,
            "expected_counts": CONTRACT["expected_counts"],
        }
        _write_json(self.paths["finalization_lock"], finalization_lock)

        baseline_artifacts = {
            "predictions": baseline_private,
            "selection": _digest("baseline-selection"),
            "metrics": fm.file_sha256(self.paths["baseline_public"]),
            "progress": _digest("baseline-progress"),
        }
        probe_artifacts = {
            "phenotype_predictions": phenotype_private,
            "phenotype_selection": _digest("phenotype-selection"),
            "phenotype_metrics": fm.file_sha256(self.paths["phenotype_public"]),
            "subtype_predictions": subtype_private,
            "subtype_selection": _digest("subtype-selection"),
            "subtype_metrics": fm.file_sha256(self.paths["subtype_public"]),
            "ftv_predictions": ftv_private,
            "ftv_selection": _digest("ftv-selection"),
            "ftv_metrics": fm.file_sha256(self.paths["ftv_public"]),
            "progress": _digest("probe-progress"),
        }
        public_roles = list(
            map(
                str,
                CONTRACT["reporting_run_provenance"]["public_artifact_roles"],
            )
        )
        marker = {
            "schema_version": fm.REPORTING_PROVENANCE_SCHEMA_VERSION,
            "summary_schema_version": fm.SUMMARY_SCHEMA_VERSION,
            "comparison_contract_canonical_sha256": CONTRACT[
                "comparison_contract_canonical_sha256"
            ],
            "baseline_v2": {
                "protocol_version": "v2",
                "evaluation_lock_sha256": _digest("evaluation-lock-v2"),
                "run_receipt_sha256": _digest("baseline-run-receipt"),
                "argv_sha256": _digest("baseline-argv"),
                "artifact_sha256": baseline_artifacts,
            },
            "probe_v3": {
                "protocol_version": "v3",
                "evaluation_lock_sha256": _digest("evaluation-lock-v3"),
                "run_receipt_sha256": _digest("probe-run-receipt"),
                "argv_sha256": _digest("probe-argv"),
                "artifact_sha256": probe_artifacts,
            },
            "summarizer": {
                "protocol_version": "v3",
                "argv_sha256": _digest("summarizer-argv"),
                "code_lock_sha256": _digest("reporting-lock-v3"),
                "finalization_lock_sha256": fm.file_sha256(
                    self.paths["finalization_lock"]
                ),
            },
            "public_artifact_sha256": {
                role: fm.file_sha256(self.paths[role]) for role in public_roles
            },
        }
        _write_json(self.paths["reporting_run_provenance"], marker)
        self.commit_current()


def _build_bundle(
    tmp_path: Path,
    *,
    status: str = "SUBSTANTIVE_PUSH_OK",
    error: str | None = None,
    shuffled: bool = False,
) -> Bundle:
    repository_root = tmp_path / "repo"
    fixed = CONTRACT["fixed_public_paths"]
    paths = {
        role: repository_root / str(relative) for role, relative in fixed.items()
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    paths["contract_json"].write_bytes(CONTRACT_PATH.read_bytes())
    paths["template_markdown"].write_bytes(TEMPLATE_PATH.read_bytes())
    paths["finalization_module"].write_bytes(MODULE_PATH.read_bytes())
    paths["finalizer_cli"].write_bytes(CLI_PATH.read_bytes())
    paths["finalization_test"].write_bytes(Path(__file__).read_bytes())
    paths["results_summary_markdown"].write_text(
        "# Synthetic complete public result summary\n", encoding="utf-8"
    )
    paths["model_execution_ledger"].write_text(
        "# Synthetic public model execution ledger\n", encoding="utf-8"
    )
    paths["foundation_model_selection"].write_text(
        "# Synthetic public foundation selection\n", encoding="utf-8"
    )
    paths["current_cnn_provenance_audit"].write_text(
        "# Synthetic public current CNN audit\n", encoding="utf-8"
    )
    paths["timing_figure"].write_bytes(b"\x89PNG\r\n\x1a\nsynthetic-timing")
    paths["calibration_figure"].write_bytes(
        b"\x89PNG\r\n\x1a\nsynthetic-calibration"
    )

    pcr = _binary_frame(fm._expected_pcr_identities(CONTRACT))
    phenotype = _binary_frame(fm._expected_phenotype_identities(CONTRACT))
    subtype = _subtype_frame()
    ftv = _ftv_frame()
    paired = _paired_frame(pcr)
    if shuffled:
        pcr = pcr.sample(frac=1.0, random_state=3).reset_index(drop=True)
        phenotype = phenotype.sample(frac=1.0, random_state=5).reset_index(drop=True)
        subtype = subtype.sample(frac=1.0, random_state=7).reset_index(drop=True)
        ftv = ftv.sample(frac=1.0, random_state=11).reset_index(drop=True)
        paired = paired.sample(frac=1.0, random_state=13).reset_index(drop=True)
    _public_pair(pcr).to_csv(paths["baseline_public"], index=False)
    _public_pair(phenotype).to_csv(paths["phenotype_public"], index=False)
    _public_pair(subtype).to_csv(paths["subtype_public"], index=False)
    _public_pair(ftv).to_csv(paths["ftv_public"], index=False)
    paired.to_csv(paths["paired_public"], index=False)
    summary = {
        "schema_version": fm.SUMMARY_SCHEMA_VERSION,
        "comparison_contract": static_comparison_contract(),
        "resolved_comparison_count": 132,
        "reported_candidate_policy": CONTRACT["candidate_policy"],
        "inference_scope": CONTRACT["inference_scope"],
        "cohorts": {
            "full": {"analysis_population": "full_808", "n": 808},
            "complete_case": {
                "analysis_population": "radiomics_complete_case_375",
                "n": 375,
            },
        },
        # This is the production reporting order; validation must be set-based,
        # length-exact, and duplicate-free rather than tied to contract order.
        "foundation_models": sorted(CONTRACT["expected_foundation_models"]),
        "current_cnn_models": list(CONTRACT["current_cnn_models"]),
        "input_sha256": {},
        "paired_comparisons": _json_records(paired),
        "pcr_pooled_metrics": _json_records(pcr),
        "phenotype_pooled_metrics": _json_records(phenotype),
        "subtype_pooled_metrics": _json_records(subtype),
        "ftv_pooled_metrics": _json_records(ftv),
    }
    _write_json(paths["summary_json"], summary)
    _write_json(paths["reporting_run_provenance"], {})
    _write_json(paths["finalization_lock"], {})
    _write_json(paths["git_handoff_json"], {})
    inputs = FinalizationInputs(
        **{
            role: paths[role]
            for role in FinalizationInputs.__dataclass_fields__
        }
    )
    outputs = FinalizationOutputs(
        final_report=repository_root / REPOSITORY_PREFIX / "reports/final_report.md",
        coverage_receipt=repository_root
        / REPOSITORY_PREFIX
        / "metrics/final_report_coverage.json",
    )
    bundle = Bundle(
        repository_root=repository_root,
        paths=paths,
        inputs=inputs,
        outputs=outputs,
        git=FakeGit(repository_root=repository_root),
        status=status,
        error=error,
    )
    bundle.reseal()
    return bundle


def _sync_summary_records(bundle: Bundle, key: str, frame: pd.DataFrame) -> None:
    summary = _read_json(bundle.paths["summary_json"])
    summary[key] = _json_records(frame)
    _write_json(bundle.paths["summary_json"], summary)


def _refresh_marker_summary_hash(bundle: Bundle) -> None:
    marker = _read_json(bundle.paths["reporting_run_provenance"])
    marker["public_artifact_sha256"]["summary_json"] = fm.file_sha256(
        bundle.paths["summary_json"]
    )
    _write_json(bundle.paths["reporting_run_provenance"], marker)


def test_exact_identity_generators_cover_frozen_axes_and_endpoints() -> None:
    pcr = fm._expected_pcr_identities(CONTRACT)
    phenotype = fm._expected_phenotype_identities(CONTRACT)
    subtype = fm._expected_subtype_identities(CONTRACT)
    ftv = fm._expected_ftv_identities(CONTRACT)
    specs = fm._expected_comparison_specs(CONTRACT)
    assert len(pcr) == 126
    assert sum(identity[-1] == "full_808" for identity in pcr) == 39
    assert sum(identity[-1] == "radiomics_complete_case_375" for identity in pcr) == 87
    assert len(specs) == 132
    assert len({spec["comparison_id"] for spec in specs}) == 132
    assert len(phenotype) == 12
    assert len(subtype) == 6
    assert len(ftv) == 42
    axes = set(fm._source_axes(CONTRACT))
    assert {
        (identity[1], identity[2]) for identity in phenotype
    } == axes
    assert all(
        {identity[0] for identity in phenotype if (identity[1], identity[2]) == axis}
        == {"HR", "HER2"}
        for axis in axes
    )
    expected_endpoints = {
        ("static", "T0"),
        ("static", "T1"),
        ("static", "T2"),
        ("static", "T3"),
        ("delta", "T0-T1"),
        ("delta", "T1-T2"),
        ("delta", "T2-T3"),
    }
    assert all(
        {
            (identity[3], identity[4])
            for identity in ftv
            if (identity[1], identity[2]) == axis
        }
        == expected_endpoints
        for axis in axes
    )


def test_complete_bundle_finalizes_and_receipt_proves_full_coverage(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)
    observed = finalize_report(bundle.inputs, bundle.outputs, git_runner=bundle.git)
    assert observed == {
        "pcr_pooled": 126,
        "paired_comparisons": 132,
        "paired_metric_rows": 396,
        "phenotype_pooled": 12,
        "subtype_pooled": 6,
        "ftv_pooled": 42,
        "public_outputs": 2,
    }
    report = bundle.outputs.final_report.read_text(encoding="utf-8")
    receipt = _read_json(bundle.outputs.coverage_receipt)
    assert "FORMAL_RESULT_PENDING" not in report
    assert "仅报告点估计方向" in report
    assert "不以 0 阈值证明‘明显’ underuse" in report
    assert "MedicalNet 3DSeg-8 ResNet-50" in report
    assert "DINO v1 ViT-B/16" in report
    assert "reporting_v3" in report and "baseline_v2" in report and "probe_v3" in report
    for spec in fm._expected_comparison_specs(CONTRACT):
        assert report.count(spec["comparison_id"]) == 1
    assert receipt["observed_counts"] == CONTRACT["expected_counts"]
    assert receipt["rendered_counts"] == {
        "pcr_pooled_total": 126,
        "paired_comparisons": 132,
        "phenotype_pooled": 12,
        "subtype_pooled": 6,
        "ftv_pooled": 42,
    }
    assert set(receipt["identity_sha256"]) == {
        "pcr_pooled",
        "phenotype_pooled",
        "subtype_pooled",
        "ftv_pooled",
        "comparison_specs",
        "paired_metric_rows",
    }
    for role in (
        "reporting_run_provenance",
        "results_summary_markdown",
        "timing_figure",
        "calibration_figure",
        "model_execution_ledger",
        "foundation_model_selection",
        "current_cnn_provenance_audit",
        "finalization_lock",
        "git_handoff_json",
        "finalization_module",
        "finalizer_cli",
        "finalization_test",
    ):
        assert receipt["input_sha256"][role] == fm.file_sha256(bundle.paths[role])
    assert receipt["publication_commit_marker"] == "coverage_receipt"
    assert receipt["cross_directory_sigkill_transactional"] is False
    assert receipt["metric_sorting"] is False
    assert receipt["best_model_filtering"] is False
    assert receipt["private_inputs_read"] is False


def test_production_foundation_order_passes_but_duplicates_and_substitution_fail(
    tmp_path: Path,
) -> None:
    bundle = _build_bundle(tmp_path / "production-order")
    results, _ = validate_public_results(bundle.inputs, git_runner=bundle.git)
    assert results.summary["foundation_models"] == [
        "dino_vitb16_imagenet1k",
        "medicalnet_resnet50_3dseg8",
    ]
    for index, values in enumerate(
        (
            ["medicalnet_resnet50_3dseg8", "medicalnet_resnet50_3dseg8"],
            ["medicalnet_resnet50_3dseg8", "posthoc_best_model"],
        )
    ):
        current = _build_bundle(tmp_path / f"bad-{index}")
        summary = _read_json(current.paths["summary_json"])
        summary["foundation_models"] = values
        _write_json(current.paths["summary_json"], summary)
        current.reseal()
        with pytest.raises(ValueError, match="exact frozen two-model"):
            validate_public_results(current.inputs, git_runner=current.git)


@pytest.mark.parametrize(
    ("summary_key", "public_role"),
    (
        ("pcr_pooled_metrics", "baseline_public"),
        ("phenotype_pooled_metrics", "phenotype_public"),
        ("subtype_pooled_metrics", "subtype_public"),
        ("ftv_pooled_metrics", "ftv_public"),
    ),
)
def test_count_preserving_or_missing_identity_fails_exact_set_gate(
    tmp_path: Path, summary_key: str, public_role: str
) -> None:
    bundle = _build_bundle(tmp_path)
    frame = pd.read_csv(bundle.paths[public_role])
    identity = (
        list(fm._FTV_IDENTITY)
        if public_role == "ftv_public"
        else list(fm._BINARY_IDENTITY)
    )
    first_identity = tuple(frame.iloc[0][identity])
    selected = np.ones(len(frame), dtype=bool)
    for column, value in zip(identity, first_identity, strict=True):
        selected &= frame[column].eq(value).to_numpy()
    frame = frame.loc[~selected].copy()
    frame.to_csv(bundle.paths[public_role], index=False)
    summary = _read_json(bundle.paths["summary_json"])
    summary[summary_key] = summary[summary_key][1:]
    _write_json(bundle.paths["summary_json"], summary)
    bundle.reseal()
    with pytest.raises(ValueError, match="identities drifted|row count drifted"):
        validate_public_results(bundle.inputs, git_runner=bundle.git)


def test_paired_absolute_values_must_link_to_pcr_even_if_csv_and_summary_tampered(
    tmp_path: Path,
) -> None:
    bundle = _build_bundle(tmp_path)
    paired = pd.read_csv(bundle.paths["paired_public"])
    paired.loc[0, "reference_value"] += 0.01
    paired.loc[0, "candidate_value"] += 0.01
    # The delta remains internally exact; only the cross-file pCR anchor exposes tampering.
    paired.to_csv(bundle.paths["paired_public"], index=False)
    _sync_summary_records(bundle, "paired_comparisons", paired)
    bundle.reseal()
    with pytest.raises(ValueError, match="drifted from pCR pooled metric"):
        validate_public_results(bundle.inputs, git_runner=bundle.git)


@pytest.mark.parametrize(
    ("location", "value", "match"),
    (
        ("summary_resolved", 132.9, "exact finite integer"),
        ("pcr_n", 808.9, "exact finite integer"),
        ("paired_seed", 2026.9, "exact finite integer"),
    ),
)
def test_discrete_fields_reject_fractional_values(
    tmp_path: Path, location: str, value: float, match: str
) -> None:
    bundle = _build_bundle(tmp_path)
    summary = _read_json(bundle.paths["summary_json"])
    if location == "summary_resolved":
        summary["resolved_comparison_count"] = value
    elif location == "pcr_n":
        summary["pcr_pooled_metrics"][0]["n"] = value
        public = pd.read_csv(bundle.paths["baseline_public"])
        identity = {
            key: summary["pcr_pooled_metrics"][0][key] for key in fm._BINARY_IDENTITY
        }
        selected = public["aggregation"].eq("pooled_oof")
        for key, expected in identity.items():
            selected &= public[key].eq(expected)
        public["n"] = public["n"].astype(float)
        public.loc[selected, "n"] = value
        public.to_csv(bundle.paths["baseline_public"], index=False)
    else:
        paired = pd.read_csv(bundle.paths["paired_public"])
        paired["bootstrap_seed"] = paired["bootstrap_seed"].astype(float)
        paired.loc[0, "bootstrap_seed"] = value
        paired.to_csv(bundle.paths["paired_public"], index=False)
        summary["paired_comparisons"] = _json_records(paired)
    _write_json(bundle.paths["summary_json"], summary)
    bundle.reseal()
    with pytest.raises(ValueError, match=match):
        validate_public_results(bundle.inputs, git_runner=bundle.git)


def test_nullable_metrics_roundtrip_as_json_null_csv_nan_and_render_na(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)
    summary = _read_json(bundle.paths["summary_json"])
    binary_identity = {
        key: summary["pcr_pooled_metrics"][0][key] for key in fm._BINARY_IDENTITY
    }
    summary["pcr_pooled_metrics"][0]["calibration_slope"] = None
    summary["pcr_pooled_metrics"][0]["calibration_intercept"] = None
    baseline = pd.read_csv(bundle.paths["baseline_public"])
    selected = pd.Series(True, index=baseline.index)
    for key, value in binary_identity.items():
        selected &= baseline[key].eq(value)
    baseline.loc[selected, ["calibration_slope", "calibration_intercept"]] = np.nan
    baseline.to_csv(bundle.paths["baseline_public"], index=False)

    ftv_identity = {key: summary["ftv_pooled_metrics"][0][key] for key in fm._FTV_IDENTITY}
    nullable = [
        "spearman",
        "pearson",
        "rmse_gain_over_b0",
        "calibration_slope",
        "calibration_intercept",
    ]
    for key in nullable:
        summary["ftv_pooled_metrics"][0][key] = None
    ftv = pd.read_csv(bundle.paths["ftv_public"])
    selected_ftv = pd.Series(True, index=ftv.index)
    for key, value in ftv_identity.items():
        selected_ftv &= ftv[key].eq(value)
    ftv.loc[selected_ftv, nullable] = np.nan
    ftv.to_csv(bundle.paths["ftv_public"], index=False)
    _write_json(bundle.paths["summary_json"], summary)
    bundle.reseal()
    finalize_report(bundle.inputs, bundle.outputs, git_runner=bundle.git)
    assert "NA" in bundle.outputs.final_report.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("role", "column", "value", "match"),
    (
        ("baseline_public", "auroc", np.nan, "must be finite"),
        ("baseline_public", "calibration_slope", np.inf, "must not contain infinity"),
        ("subtype_public", "multiclass_brier", 2.1, r"within \[0.0, 2.0\]"),
        ("ftv_public", "rmse", -0.1, "must be nonnegative"),
        ("ftv_public", "pearson", 1.1, r"null or within \[-1, 1\]"),
    ),
)
def test_metric_domain_policy_fails_closed(
    tmp_path: Path, role: str, column: str, value: float, match: str
) -> None:
    bundle = _build_bundle(tmp_path)
    frame = pd.read_csv(bundle.paths[role])
    macro_index = frame.index[frame["aggregation"].eq("outer_fold_macro")][0]
    frame.loc[macro_index, column] = value
    frame.to_csv(bundle.paths[role], index=False)
    bundle.reseal()
    with pytest.raises(ValueError, match=match):
        validate_public_results(bundle.inputs, git_runner=bundle.git)


@pytest.mark.parametrize(
    ("location", "match"),
    (
        ("binary_bounded", "not booleans or strings"),
        ("binary_nullable", "not booleans or strings"),
        ("ftv_required", "not booleans or strings"),
        ("paired_ci", "not booleans or strings"),
    ),
)
def test_boolean_is_never_accepted_as_a_numeric_metric(
    tmp_path: Path, location: str, match: str
) -> None:
    bundle = _build_bundle(tmp_path)
    summary = _read_json(bundle.paths["summary_json"])
    if location == "binary_bounded":
        summary["pcr_pooled_metrics"][0]["auroc"] = True
    elif location == "binary_nullable":
        summary["pcr_pooled_metrics"][0]["calibration_slope"] = True
    elif location == "ftv_required":
        summary["ftv_pooled_metrics"][0]["calibration_mean_bias"] = True
    else:
        paired = pd.read_csv(bundle.paths["paired_public"])
        paired["ci_level"] = True
        paired.to_csv(bundle.paths["paired_public"], index=False)
        summary["paired_comparisons"] = _json_records(paired)
    _write_json(bundle.paths["summary_json"], summary)
    bundle.reseal()
    with pytest.raises(ValueError, match=match):
        validate_public_results(bundle.inputs, git_runner=bundle.git)


@pytest.mark.parametrize(
    ("field", "match"),
    (
        ("resolved_comparison_count", "exact finite integer"),
        ("auroc", "not booleans or strings"),
        ("calibration_slope", "not booleans or strings"),
    ),
)
def test_json_numeric_strings_are_not_coerced(
    tmp_path: Path, field: str, match: str
) -> None:
    bundle = _build_bundle(tmp_path)
    summary = _read_json(bundle.paths["summary_json"])
    if field == "resolved_comparison_count":
        summary[field] = "132"
    else:
        summary["pcr_pooled_metrics"][0][field] = "0.7"
    _write_json(bundle.paths["summary_json"], summary)
    bundle.reseal()
    with pytest.raises(ValueError, match=match):
        validate_public_results(bundle.inputs, git_runner=bundle.git)


def test_phenotype_positive_count_must_match_across_all_six_axes(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)
    summary = _read_json(bundle.paths["summary_json"])
    row = summary["phenotype_pooled_metrics"][0]
    row["positive"] += 1
    public = pd.read_csv(bundle.paths["phenotype_public"])
    selected = pd.Series(True, index=public.index)
    for key in fm._BINARY_IDENTITY:
        selected &= public[key].eq(row[key])
    public.loc[selected, "positive"] = row["positive"]
    public.to_csv(bundle.paths["phenotype_public"], index=False)
    _write_json(bundle.paths["summary_json"], summary)
    bundle.reseal()
    with pytest.raises(ValueError, match="positive count changes"):
        validate_public_results(bundle.inputs, git_runner=bundle.git)


def test_whole_static_comparison_contract_canonical_hash_is_required(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)
    summary = _read_json(bundle.paths["summary_json"])
    summary["comparison_contract"]["bootstrap"]["replicates"] = 4999
    _write_json(bundle.paths["summary_json"], summary)
    bundle.reseal()
    with pytest.raises(ValueError, match="static comparison contract SHA"):
        validate_public_results(bundle.inputs, git_runner=bundle.git)


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("marker_extra", "top-level schema"),
        ("marker_lock_sha", "finalization-lock SHA"),
        ("marker_metric_sha", "differs from baseline CSV"),
        ("summary_private_sha", "mixed producer lineage"),
        ("lock_unseen", "formal metric values unseen"),
        ("lock_fractional_count", "exact finite integer"),
    ),
)
def test_lineage_marker_and_finalization_lock_are_exact_and_cross_linked(
    tmp_path: Path, mutation: str, match: str
) -> None:
    bundle = _build_bundle(tmp_path)
    if mutation.startswith("marker"):
        marker = _read_json(bundle.paths["reporting_run_provenance"])
        if mutation == "marker_extra":
            marker["unexpected"] = "x"
        elif mutation == "marker_lock_sha":
            marker["summarizer"]["finalization_lock_sha256"] = "9" * 64
        else:
            marker["baseline_v2"]["artifact_sha256"]["metrics"] = "9" * 64
        _write_json(bundle.paths["reporting_run_provenance"], marker)
    elif mutation == "summary_private_sha":
        summary = _read_json(bundle.paths["summary_json"])
        summary["input_sha256"]["baseline_private"] = "9" * 64
        _write_json(bundle.paths["summary_json"], summary)
        _refresh_marker_summary_hash(bundle)
    else:
        lock = _read_json(bundle.paths["finalization_lock"])
        if mutation == "lock_unseen":
            lock["formal_metric_values_unseen"] = False
        else:
            lock["expected_counts"]["paired_comparisons"] = 132.5
        _write_json(bundle.paths["finalization_lock"], lock)
    bundle.commit_current()
    with pytest.raises(ValueError, match=match):
        validate_public_results(bundle.inputs, git_runner=bundle.git)


def test_private_paths_and_forbidden_public_columns_are_rejected(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path / "private")
    private_path = tmp_path / "private" / "results_summary.json"
    private_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.write_bytes(bundle.paths["summary_json"].read_bytes())
    with pytest.raises(ValueError, match="private path rejected"):
        validate_public_results(
            replace(bundle.inputs, summary_json=private_path), git_runner=bundle.git
        )

    public_bundle = _build_bundle(tmp_path / "column")
    frame = pd.read_csv(public_bundle.paths["baseline_public"])
    frame["source_path"] = "relative/patient-source.csv"
    frame.to_csv(public_bundle.paths["baseline_public"], index=False)
    public_bundle.reseal()
    with pytest.raises(ValueError, match="forbidden public columns"):
        validate_public_results(public_bundle.inputs, git_runner=public_bundle.git)


def test_fixed_git_path_is_enforced(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)
    alternate = tmp_path / "alternate-summary.json"
    alternate.write_bytes(bundle.paths["summary_json"].read_bytes())
    with pytest.raises(ValueError, match="formal public input path drifted"):
        validate_public_results(
            replace(bundle.inputs, summary_json=alternate), git_runner=bundle.git
        )


def test_successful_git_handoff_requires_current_remote_ref(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path / "stale")
    bundle.git.remote_sha = "2" * 40
    with pytest.raises(ValueError, match="remote ref does not resolve"):
        validate_public_results(bundle.inputs, git_runner=bundle.git)

    wrong_manifest = _build_bundle(tmp_path / "manifest")
    manifest = _read_json(wrong_manifest.paths["git_handoff_json"])
    manifest["substantive_remote_ref_sha"] = "2" * 40
    _write_json(wrong_manifest.paths["git_handoff_json"], manifest)
    with pytest.raises(ValueError, match="remote SHA differs"):
        validate_public_results(wrong_manifest.inputs, git_runner=wrong_manifest.git)


def test_failed_push_records_real_sanitized_error_without_remote_lookup(tmp_path: Path) -> None:
    error = "fatal: remote rejected | denied\nretry later <safe>"
    bundle = _build_bundle(
        tmp_path,
        status="GITHUB_PUSH_FAILED",
        error=error,
    )
    finalize_report(bundle.inputs, bundle.outputs, git_runner=bundle.git)
    assert not any(call[:2] == ("ls-remote", "--heads") for call in bundle.git.calls or [])
    report = bundle.outputs.final_report.read_text(encoding="utf-8")
    assert "GITHUB_PUSH_FAILED" in report
    assert r"remote rejected \| denied<br>retry later &lt;safe&gt;" in report


@pytest.mark.parametrize(
    "unsafe_error",
    (
        "fatal: /data/mi2 secret path",
        "fatal: token=secret",
        "fatal: ghp_abcdefghijklmnop",
        "x" * 1001,
    ),
)
def test_failed_push_error_must_be_sanitized(tmp_path: Path, unsafe_error: str) -> None:
    bundle = _build_bundle(
        tmp_path,
        status="GITHUB_PUSH_FAILED",
        error=unsafe_error,
    )
    with pytest.raises(ValueError, match="absolute path|safely sanitized|length"):
        validate_public_results(bundle.inputs, git_runner=bundle.git)


def test_git_handoff_exact_schema_and_committed_bytes(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path / "schema")
    manifest = _read_json(bundle.paths["git_handoff_json"])
    manifest["unexpected"] = True
    _write_json(bundle.paths["git_handoff_json"], manifest)
    with pytest.raises(ValueError, match="exact schema"):
        validate_public_results(bundle.inputs, git_runner=bundle.git)

    dirty = _build_bundle(tmp_path / "dirty")
    dirty.paths["results_summary_markdown"].write_text("dirty after commit\n", encoding="utf-8")
    with pytest.raises(ValueError, match="working bytes drifted"):
        validate_public_results(dirty.inputs, git_runner=dirty.git)


def test_input_snapshot_prevents_template_toctou(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)

    def mutate_after_snapshot() -> None:
        bundle.paths["template_markdown"].write_text(
            "MUTATED_AFTER_SNAPSHOT\n", encoding="utf-8"
        )

    bundle.git.on_root = mutate_after_snapshot
    finalize_report(bundle.inputs, bundle.outputs, git_runner=bundle.git)
    report = bundle.outputs.final_report.read_text(encoding="utf-8")
    assert "MUTATED_AFTER_SNAPSHOT" not in report
    assert "DCE-MRI Foundation Encoder Baselines" in report


def test_fixed_identity_sort_is_independent_of_input_row_order(tmp_path: Path) -> None:
    first = _build_bundle(tmp_path / "first", shuffled=False)
    second = _build_bundle(tmp_path / "second", shuffled=True)
    result_a, _ = validate_public_results(first.inputs, git_runner=first.git)
    result_b, _ = validate_public_results(second.inputs, git_runner=second.git)
    assert fm._render_pcr_table(result_a, CONTRACT) == fm._render_pcr_table(
        result_b, CONTRACT
    )
    assert fm._render_paired_table(result_a) == fm._render_paired_table(result_b)
    assert fm._render_phenotype_table(result_a) == fm._render_phenotype_table(result_b)
    assert fm._render_subtype_table(result_a) == fm._render_subtype_table(result_b)
    assert fm._render_ftv_table(result_a) == fm._render_ftv_table(result_b)


def test_no_overwrite_or_orphan_and_second_link_failure_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    both = _build_bundle(tmp_path / "both")
    Path(both.outputs.final_report).write_text("keep report\n", encoding="utf-8")
    Path(both.outputs.coverage_receipt).write_text("keep receipt\n", encoding="utf-8")
    missing_inputs = replace(both.inputs, summary_json=tmp_path / "missing.json")
    with pytest.raises(FileExistsError, match="forbids overwrite"):
        finalize_report(missing_inputs, both.outputs, git_runner=both.git)

    orphan = _build_bundle(tmp_path / "orphan")
    Path(orphan.outputs.final_report).write_text("orphan\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="commit marker"):
        finalize_report(orphan.inputs, orphan.outputs, git_runner=orphan.git)

    rollback = _build_bundle(tmp_path / "rollback")
    real_link = fm.os.link
    call_count = 0

    def fail_second_link(source: Path, destination: Path) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise OSError("synthetic second publication link failure")
        real_link(source, destination)

    monkeypatch.setattr(fm.os, "link", fail_second_link)
    with pytest.raises(OSError, match="second publication"):
        finalize_report(rollback.inputs, rollback.outputs, git_runner=rollback.git)
    assert not Path(rollback.outputs.final_report).exists()
    assert not Path(rollback.outputs.coverage_receipt).exists()


def _set_question_direction(
    paired: pd.DataFrame, question: str, direction: str
) -> pd.DataFrame:
    output = paired.copy()
    compact = fm._compact_comparisons(output)
    ids = set(fm._question_subsets(compact, CONTRACT)[question]["comparison_id"])
    selected = output["comparison_id"].isin(ids)
    for metric in fm._METRICS:
        current = selected & output["metric"].eq(metric)
        if direction == "favorable":
            delta = -0.01 if metric == "brier" else 0.01
        elif direction == "adverse":
            delta = 0.01 if metric == "brier" else -0.01
        else:
            delta = -0.01 if metric == "auprc" else 0.01
        output.loc[current, "delta_candidate_minus_reference"] = delta
        output.loc[current, "ci_low"] = delta - 0.001
        output.loc[current, "ci_high"] = delta + 0.001
    return output


@pytest.mark.parametrize(
    ("q8", "q9", "q11", "expected"),
    (
        (
            "favorable",
            "favorable",
            "favorable",
            "prioritize_integration_and_external_validation",
        ),
        ("mixed", "favorable", "favorable", "controlled_integration_study_only"),
        (
            "adverse",
            "favorable",
            "favorable",
            "do_not_prioritize_replacement_from_this_experiment",
        ),
        (
            "favorable",
            "favorable",
            "mixed",
            "do_not_prioritize_replacement_from_this_experiment",
        ),
    ),
)
def test_q12_fixed_conservative_rule_boundaries(
    q8: str, q9: str, q11: str, expected: str
) -> None:
    pcr = _binary_frame(fm._expected_pcr_identities(CONTRACT))
    paired = _paired_frame(pcr)
    for question, direction in (("Q8", q8), ("Q9", q9), ("Q11", q11)):
        paired = _set_question_direction(paired, question, direction)
    evidence = fixed_question_evidence(paired, CONTRACT)
    assert evidence["Q12"]["recommendation"] == expected


def test_sign_preserving_small_number_format_and_template_links(tmp_path: Path) -> None:
    assert fm._format_number(1e-8) == "1.000e-08"
    assert fm._format_number(-1e-8) == "-1.000e-08"
    assert fm._format_number(0.0) == "0.000000"
    bundle = _build_bundle(tmp_path)
    results, hashes = validate_public_results(bundle.inputs, git_runner=bundle.git)
    rendered = render_final_report(
        bundle.paths["template_markdown"].read_text(encoding="utf-8"),
        results,
        CONTRACT,
        hashes,
    )
    for relative_link in (
        "foundation_model_selection.md",
        "model_execution_ledger.md",
        "current_cnn_provenance_audit.md",
        "results_summary.md",
        "../figures/pcr_timing_performance.png",
        "../figures/calibration_clinical_complementarity.png",
    ):
        assert relative_link in rendered
    assert "substantive_push_status" in rendered
    assert "第二次 push" in rendered


def test_contract_nested_schema_and_metric_policy_are_exact(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    payload["metric_domain_policy"]["infinity_allowed"] = True
    path = tmp_path / "contract.json"
    _write_json(path, payload)
    with pytest.raises(ValueError, match="metric nullable/domain policy"):
        load_contract(path)

    payload = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    payload["populations"]["full"]["unexpected"] = 1
    _write_json(path, payload)
    with pytest.raises(ValueError, match="population full nested schema"):
        load_contract(path)

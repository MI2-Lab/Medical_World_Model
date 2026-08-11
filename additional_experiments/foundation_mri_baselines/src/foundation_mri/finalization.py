"""Public-only, fail-closed final report generation.

This module is deliberately separate from outcome evaluation and bootstrap
reporting.  It accepts only already-published aggregate CSV/JSON artifacts,
reconstructs every frozen identity from a static contract, and renders all
rows in identity order.  It exposes no model, timing, metric, or top-k filter.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


CONTRACT_SCHEMA_VERSION = 1
SUMMARY_SCHEMA_VERSION = "foundation_mri_results_summary_v1"
COVERAGE_SCHEMA_VERSION = "foundation_mri_final_report_coverage_v1"
REPORTING_PROVENANCE_SCHEMA_VERSION = "foundation_mri_reporting_run_provenance_v1"
GIT_HANDOFF_SCHEMA_VERSION = "foundation_mri_git_handoff_v1"

_SUMMARY_KEYS = {
    "schema_version",
    "comparison_contract",
    "resolved_comparison_count",
    "reported_candidate_policy",
    "inference_scope",
    "cohorts",
    "foundation_models",
    "current_cnn_models",
    "input_sha256",
    "paired_comparisons",
    "pcr_pooled_metrics",
    "phenotype_pooled_metrics",
    "subtype_pooled_metrics",
    "ftv_pooled_metrics",
}

_BINARY_IDENTITY = (
    "target",
    "model",
    "spatial",
    "timing",
    "analysis_population",
)
_BINARY_PROVENANCE = (
    "split_seed",
    "fold_manifest_sha256",
)
_BINARY_SCHEMA = {
    *_BINARY_IDENTITY,
    *_BINARY_PROVENANCE,
    "aggregation",
    "n",
    "positive",
    "n_folds",
    "ece_bin_contract",
    "auroc",
    "auprc",
    "brier",
    "calibration_slope",
    "calibration_intercept",
    "ece_10bin",
}
_SUBTYPE_SCHEMA = {
    *_BINARY_IDENTITY,
    *_BINARY_PROVENANCE,
    "aggregation",
    "n",
    "n_folds",
    "ece_bin_contract",
    "macro_ovr_auroc",
    "macro_ovr_auprc",
    "multiclass_brier",
    "toplabel_ece_10bin",
    "accuracy",
}
_FTV_IDENTITY = (
    "target",
    "model",
    "spatial",
    "task",
    "endpoint",
    "analysis_population",
)
_FTV_SCHEMA = {
    *_FTV_IDENTITY,
    *_BINARY_PROVENANCE,
    "aggregation",
    "n",
    "n_folds",
    "spearman",
    "pearson",
    "r2",
    "rmse",
    "mae",
    "b0_rmse",
    "rmse_gain_over_b0",
    "calibration_slope",
    "calibration_intercept",
    "calibration_mean_bias",
}
_COMPARISON_SPEC_COLUMNS = (
    "comparison_id",
    "family",
    "estimand",
    "analysis_population",
    "timing",
    "reference_model",
    "reference_spatial",
    "candidate_model",
    "candidate_spatial",
)
_PAIRED_SCHEMA = {
    *_COMPARISON_SPEC_COLUMNS,
    "metric",
    "higher_is_better",
    "reference_value",
    "candidate_value",
    "delta_candidate_minus_reference",
    "ci_level",
    "ci_low",
    "ci_high",
    "bootstrap_seed",
    "bootstrap_replicates",
    "valid_bootstrap_replicates",
    "n_paired",
    "positive",
    "inference_scope",
}
_METRICS = ("auroc", "auprc", "brier")
_FORBIDDEN_PUBLIC_KEYS = {
    "patient_id",
    "clinical_patient_id",
    "raw_patient_id",
    "trial_id",
    "source_file",
    "y_true",
    "y_score",
    "y_pred",
    "probabilities_json",
    "classes_json",
}
_TEMPLATE_TOKENS = (
    "COVERAGE_MANIFEST",
    "MODEL_PROVENANCE",
    "QUESTION_MATRIX",
    "FIXED_CONCLUSIONS",
    "PCR_TABLE",
    "PAIRED_TABLE",
    "PHENOTYPE_TABLE",
    "SUBTYPE_TABLE",
    "FTV_TABLE",
    "PRODUCER_LINEAGE",
    "PUBLIC_PROVENANCE",
    "GIT_HANDOFF",
)

_EXPERIMENT_PREFIX = "additional_experiments/foundation_mri_baselines"
_FIXED_PUBLIC_PATHS = {
    "summary_json": f"{_EXPERIMENT_PREFIX}/metrics/results_summary.json",
    "baseline_public": f"{_EXPERIMENT_PREFIX}/metrics/baseline_metrics.csv",
    "phenotype_public": f"{_EXPERIMENT_PREFIX}/metrics/phenotype_metrics.csv",
    "subtype_public": f"{_EXPERIMENT_PREFIX}/metrics/subtype_metrics.csv",
    "ftv_public": f"{_EXPERIMENT_PREFIX}/metrics/ftv_probe_metrics.csv",
    "paired_public": f"{_EXPERIMENT_PREFIX}/metrics/paired_bootstrap_comparisons.csv",
    "reporting_run_provenance": f"{_EXPERIMENT_PREFIX}/metrics/reporting_run_provenance.json",
    "results_summary_markdown": f"{_EXPERIMENT_PREFIX}/reports/results_summary.md",
    "timing_figure": f"{_EXPERIMENT_PREFIX}/figures/pcr_timing_performance.png",
    "calibration_figure": f"{_EXPERIMENT_PREFIX}/figures/calibration_clinical_complementarity.png",
    "model_execution_ledger": f"{_EXPERIMENT_PREFIX}/reports/model_execution_ledger.md",
    "foundation_model_selection": f"{_EXPERIMENT_PREFIX}/reports/foundation_model_selection.md",
    "current_cnn_provenance_audit": f"{_EXPERIMENT_PREFIX}/reports/current_cnn_provenance_audit.md",
    "contract_json": f"{_EXPERIMENT_PREFIX}/configs/final_report_contract.json",
    "template_markdown": f"{_EXPERIMENT_PREFIX}/reports/final_report.template.md",
    "finalization_module": f"{_EXPERIMENT_PREFIX}/src/foundation_mri/finalization.py",
    "finalizer_cli": f"{_EXPERIMENT_PREFIX}/scripts/finalize_report.py",
    "finalization_test": f"{_EXPERIMENT_PREFIX}/tests/test_finalization.py",
    "finalization_lock": f"{_EXPERIMENT_PREFIX}/configs/FINALIZATION_LOCK.v1.json",
    "git_handoff_json": f"{_EXPERIMENT_PREFIX}/reports/git_handoff.json",
}

_METRIC_DOMAIN_POLICY = {
    "binary_required_finite_unit_interval": ["auroc", "auprc", "brier", "ece_10bin"],
    "binary_nullable_finite": ["calibration_slope", "calibration_intercept"],
    "subtype_required_finite_unit_interval": [
        "macro_ovr_auroc",
        "macro_ovr_auprc",
        "toplabel_ece_10bin",
        "accuracy",
    ],
    "subtype_required_finite_zero_to_two": ["multiclass_brier"],
    "ftv_required_finite": [
        "r2",
        "rmse",
        "mae",
        "b0_rmse",
        "calibration_mean_bias",
    ],
    "ftv_required_nonnegative": ["rmse", "mae", "b0_rmse"],
    "ftv_nullable_finite_bounded_correlation": ["spearman", "pearson"],
    "ftv_nullable_finite": [
        "rmse_gain_over_b0",
        "calibration_slope",
        "calibration_intercept",
    ],
    "infinity_allowed": False,
    "json_null_csv_nan_equivalent": True,
}


@dataclass(frozen=True)
class FinalizationInputs:
    summary_json: Path
    baseline_public: Path
    phenotype_public: Path
    subtype_public: Path
    ftv_public: Path
    paired_public: Path
    reporting_run_provenance: Path
    results_summary_markdown: Path
    timing_figure: Path
    calibration_figure: Path
    model_execution_ledger: Path
    foundation_model_selection: Path
    current_cnn_provenance_audit: Path
    contract_json: Path
    template_markdown: Path
    finalization_lock: Path
    git_handoff_json: Path


@dataclass(frozen=True)
class FinalizationOutputs:
    final_report: Path
    coverage_receipt: Path


@dataclass(frozen=True)
class ValidatedPublicResults:
    summary: Mapping[str, Any]
    pcr: pd.DataFrame
    phenotype: pd.DataFrame
    subtype: pd.DataFrame
    ftv: pd.DataFrame
    paired: pd.DataFrame
    comparison_specs: tuple[Mapping[str, str], ...]
    question_evidence: Mapping[str, Any]
    public_row_counts: Mapping[str, int]
    reporting_run_provenance: Mapping[str, Any]
    git_handoff: Mapping[str, Any]


@dataclass(frozen=True)
class _InputSnapshots:
    paths: Mapping[str, Path]
    payloads: Mapping[str, bytes]
    sha256: Mapping[str, str]


GitRunner = Callable[[Sequence[str], Path], bytes]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"


def _reject_private_path(path: Path, *, label: str) -> None:
    source = Path(path)
    candidates = [source]
    if source.exists() or source.is_symlink():
        candidates.append(source.resolve(strict=False))
    for candidate in candidates:
        for part in candidate.parts:
            lowered = part.lower()
            if lowered == "private" or ".private." in lowered or lowered.startswith("private."):
                raise ValueError(
                    f"{label} must be public-only; private path rejected: {source.name}"
                )


def _looks_absolute(value: str) -> bool:
    return os.path.isabs(value) or bool(re.match(r"^[A-Za-z]:[\\/]", value))


def _forbidden_public_key(value: Any) -> bool:
    lowered = str(value).lower()
    return (
        lowered in _FORBIDDEN_PUBLIC_KEYS
        or lowered.endswith("_path")
        or lowered.endswith("_file")
    )


def _assert_public_payload(value: Any, *, location: str = "root") -> None:
    if isinstance(value, Mapping):
        forbidden = sorted(str(key) for key in value if _forbidden_public_key(key))
        if forbidden:
            raise ValueError(f"public payload contains forbidden keys at {location}: {forbidden}")
        for key, child in value.items():
            _assert_public_payload(child, location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_public_payload(child, location=f"{location}[{index}]")
    elif isinstance(value, str) and _looks_absolute(value):
        raise ValueError(f"public payload contains an absolute path at {location}")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise ValueError(f"public payload contains NaN/Inf at {location}")


def _assert_public_frame(frame: pd.DataFrame, *, label: str) -> None:
    forbidden = sorted(str(column) for column in frame.columns if _forbidden_public_key(column))
    if forbidden:
        raise ValueError(f"{label} contains forbidden public columns: {forbidden}")
    for column in frame.select_dtypes(include=["object", "string"]).columns:
        for value in frame[column].dropna().astype(str):
            if _looks_absolute(value):
                raise ValueError(f"{label} contains an absolute path in {column}")


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    source = Path(path)
    _reject_private_path(source, label=label)
    if not source.is_file():
        raise FileNotFoundError(f"required {label} is missing: {source.name}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    _assert_public_payload(payload, location=label)
    return payload


def _read_csv(path: Path, *, label: str) -> pd.DataFrame:
    source = Path(path)
    _reject_private_path(source, label=label)
    if not source.is_file():
        raise FileNotFoundError(f"required {label} is missing: {source.name}")
    frame = pd.read_csv(source)
    if frame.empty:
        raise ValueError(f"required {label} is empty")
    _assert_public_frame(frame, label=label)
    return frame


def _strict_int(value: Any, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_, str, bytes)):
        raise ValueError(f"{label} must be an exact finite integer")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be an exact finite integer") from error
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{label} must be an exact finite integer")
    return int(number)


def _strict_int_array(values: Sequence[Any] | pd.Series, *, label: str) -> np.ndarray:
    return np.asarray(
        [_strict_int(value, label=label) for value in list(values)], dtype=np.int64
    )


def _finite_array(values: Sequence[Any] | pd.Series, *, label: str) -> np.ndarray:
    materialised = list(values)
    if any(isinstance(value, (bool, np.bool_, str, bytes)) for value in materialised):
        raise ValueError(f"{label} must contain numeric scalars, not booleans or strings")
    numeric = pd.to_numeric(pd.Series(materialised), errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{label} must be finite")
    return numeric


def _nullable_finite_array(
    values: Sequence[Any] | pd.Series, *, label: str
) -> np.ndarray:
    output: list[float] = []
    for value in list(values):
        if isinstance(value, (bool, np.bool_, str, bytes)):
            raise ValueError(
                f"{label} must contain numeric scalars or null, not booleans or strings"
            )
        if value is None or (not isinstance(value, (str, bytes)) and bool(pd.isna(value))):
            output.append(math.nan)
            continue
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} must be finite or null") from error
        if not math.isfinite(number):
            raise ValueError(f"{label} must not contain infinity")
        output.append(number)
    return np.asarray(output, dtype=float)


def _bounded_array(
    values: Sequence[Any] | pd.Series,
    *,
    label: str,
    lower: float,
    upper: float,
) -> np.ndarray:
    numeric = _finite_array(values, label=label)
    if np.any(numeric < lower) or np.any(numeric > upper):
        raise ValueError(f"{label} must be within [{lower}, {upper}]")
    return numeric


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", str(value)))


def _canonical_pretty_json_sha256(value: Any) -> str:
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _parse_json_bytes(payload: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        parsed = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must contain a JSON object")
    _assert_public_payload(parsed, location=label)
    return parsed


def _parse_csv_bytes(payload: bytes, *, label: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(io.BytesIO(payload))
    except Exception as error:
        raise ValueError(f"{label} is not a readable CSV") from error
    if frame.empty:
        raise ValueError(f"required {label} is empty")
    _assert_public_frame(frame, label=label)
    return frame


def _snapshot_inputs(inputs: FinalizationInputs) -> _InputSnapshots:
    paths = {field: Path(value) for field, value in inputs.__dict__.items()}
    payloads: dict[str, bytes] = {}
    digests: dict[str, str] = {}
    for role, path in paths.items():
        _reject_private_path(path, label=role)
        if not path.is_file():
            raise FileNotFoundError(f"required public input is missing: {path.name}")
        payload = path.read_bytes()
        payloads[role] = payload
        digests[role] = _sha256_bytes(payload)
    return _InputSnapshots(paths=paths, payloads=payloads, sha256=digests)


def _validate_contract_payload(contract: Mapping[str, Any]) -> Mapping[str, Any]:
    if contract.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise ValueError("unsupported final-report contract schema")
    required = {
        "schema_version",
        "contract_id",
        "candidate_policy",
        "expected_foundation_models",
        "current_cnn_models",
        "source_axes",
        "decision_points",
        "populations",
        "split_seed",
        "fold_manifest_sha256",
        "n_folds",
        "bootstrap",
        "expected_counts",
        "comparison_family_counts",
        "comparison_contract_canonical_sha256",
        "model_provenance",
        "fixed_public_paths",
        "reporting_run_provenance",
        "metric_domain_policy",
        "git_handoff",
        "question_subsets",
        "direction_rule",
        "publication_atomicity",
        "inference_scope",
    }
    if set(contract) != required:
        raise ValueError(
            "final-report contract exact schema drifted: "
            f"missing={sorted(required - set(contract))}, extra={sorted(set(contract) - required)}"
        )
    if contract["contract_id"] != "foundation_mri_final_report_v1":
        raise ValueError("final-report contract identifier drifted")
    if contract["candidate_policy"] != (
        "all preregistered candidates; no best-model filtering"
    ):
        raise ValueError("final-report candidate policy drifted")
    foundation = tuple(str(value) for value in contract["expected_foundation_models"])
    if len(foundation) != 2 or len(set(foundation)) != 2:
        raise ValueError("formal final-report contract requires exactly two foundation models")
    current = tuple(str(value) for value in contract["current_cnn_models"])
    if len(current) != 2 or len(set(current)) != 2 or set(current) != {"GAP0", "LOCAL0"}:
        raise ValueError("current-CNN contract drifted")
    axes = _source_axes(contract)
    expected_axes = {
        *((model, spatial) for model in foundation for spatial in ("GLOBAL", "LOCAL")),
        ("GAP0", "GLOBAL"),
        ("LOCAL0", "LOCAL"),
    }
    if set(axes) != expected_axes or len(axes) != len(expected_axes):
        raise ValueError("formal source/spatial axes drifted")
    if not isinstance(contract["decision_points"], list) or tuple(
        map(str, contract["decision_points"])
    ) != ("T0", "T0-T1", "T0-T2"):
        raise ValueError("decision-point contract drifted")
    expected_counts = contract["expected_counts"]
    frozen_counts = {
        "pcr_pooled_full": 39,
        "pcr_pooled_complete_case": 87,
        "pcr_pooled_total": 126,
        "baseline_public_rows": 252,
        "paired_comparisons": 132,
        "paired_metric_rows": 396,
        "phenotype_pooled": 12,
        "phenotype_public_rows": 24,
        "subtype_pooled": 6,
        "subtype_public_rows": 12,
        "ftv_pooled": 42,
        "ftv_public_rows": 84,
    }
    if not isinstance(expected_counts, dict) or set(expected_counts) != set(frozen_counts):
        raise ValueError("formal expected-count schema drifted")
    if {
        key: _strict_int(expected_counts.get(key), label=f"expected_counts.{key}")
        for key in frozen_counts
    } != frozen_counts:
        raise ValueError("formal expected-count contract drifted")
    if str(contract["comparison_contract_canonical_sha256"]) != (
        "f99fd76bd35b784500194347c4b363725616ca6adab2ba830a006cc7cc4a7e13"
    ):
        raise ValueError("static comparison-contract SHA drifted")
    if _strict_int(contract["split_seed"], label="contract split seed") != 2026:
        raise ValueError("split seed contract drifted")
    if contract["fold_manifest_sha256"] != (
        "143e482d711225c0611006d99bd7345d2fa1a5c16c65fbaf8399341a0d26aa38"
    ):
        raise ValueError("fold-manifest contract drifted")
    if _strict_int(contract["n_folds"], label="contract n_folds") != 5:
        raise ValueError("outer-fold contract drifted")
    if contract["inference_scope"] != (
        "descriptive paired outer-fold OOF patient bootstrap; not confirmatory and not used "
        "for model or checkpoint selection"
    ):
        raise ValueError("inference-scope contract drifted")
    _validate_contract_extensions(contract)
    return contract


def load_contract(path: Path) -> Mapping[str, Any]:
    return _validate_contract_payload(_read_json(path, label="final-report contract"))


def _validate_contract_extensions(contract: Mapping[str, Any]) -> None:
    expected_path_roles = set(_FIXED_PUBLIC_PATHS)
    paths = contract["fixed_public_paths"]
    if not isinstance(paths, dict) or paths != _FIXED_PUBLIC_PATHS:
        raise ValueError("fixed public paths drifted")
    for role, value in paths.items():
        relative = Path(str(value))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(f"fixed public path is unsafe for {role}")
    expected_axis_pretraining = {
        ("medicalnet_resnet50_3dseg8", "GLOBAL"): "MedicalNet 3DSeg-8",
        ("medicalnet_resnet50_3dseg8", "LOCAL"): "MedicalNet 3DSeg-8",
        ("dino_vitb16_imagenet1k", "GLOBAL"): "DINO v1 ImageNet-1K SSL",
        ("dino_vitb16_imagenet1k", "LOCAL"): "DINO v1 ImageNet-1K SSL",
        ("GAP0", "GLOBAL"): "current response-state (grounded=false)",
        ("LOCAL0", "LOCAL"): "current response-state (grounded=false)",
    }
    source_axes = contract["source_axes"]
    if (
        not isinstance(source_axes, list)
        or len(source_axes) != len(expected_axis_pretraining)
        or any(
            not isinstance(entry, dict)
            or set(entry) != {"model", "spatial", "pretraining"}
            for entry in source_axes
        )
    ):
        raise ValueError("source-axis nested schema drifted")
    observed_axis_pretraining = {
        (str(entry["model"]), str(entry["spatial"])): str(entry["pretraining"])
        for entry in source_axes
    }
    if observed_axis_pretraining != expected_axis_pretraining:
        raise ValueError("source-axis provenance drifted")
    model_schema = {
        "model",
        "display_name",
        "pretraining_domain",
        "source_revision",
        "license",
        "checkpoint_sha256",
        "feature_sha256",
        "parameters",
        "feature_dimension_per_visit",
        "selection_reason",
    }
    models = contract["model_provenance"]
    if not isinstance(models, list) or len(models) != 2:
        raise ValueError("model provenance must contain exactly two models")
    if any(not isinstance(entry, dict) or set(entry) != model_schema for entry in models):
        raise ValueError("model provenance nested schema drifted")
    expected_models = {
        "medicalnet_resnet50_3dseg8": {
            "display_name": "MedicalNet 3DSeg-8 ResNet-50",
            "pretraining_domain": "8 enumerated 3-D medical segmentation datasets (MRI+CT)",
            "source_revision": "Tencent MedicalNet official archive, pretrain/resnet_50.pth",
            "license": "MIT",
            "checkpoint_sha256": "5b6189cafbee2f5604a7279b62bc163365aa6a86a377e1dc260a14275cacbd84",
            "feature_sha256": "ca45a46bd62e18e42b6d3f2426ce4690a4f3dbf7c2f44804ab0d19bd333ee4a2",
            "parameters": 46155072,
            "feature_dimension_per_visit": 14336,
            "selection_reason": "可审计的 3-D medical reference，保留完整 DCE7 channel",
        },
        "dino_vitb16_imagenet1k": {
            "display_name": "DINO v1 ViT-B/16",
            "pretraining_domain": "ImageNet-1K self-supervised learning",
            "source_revision": "Meta DINO native ViT-B/16 ImageNet-1K checkpoint",
            "license": "Apache-2.0",
            "checkpoint_sha256": "bf34ad0f424b9029b593e8dc3ed553bf26e88bcba0d32bf3e62a6209cb64c85e",
            "feature_sha256": "c078cd4ddc0c745c32ebcca247d44ef8025d08495f6e3193a481563f0d53ffbc",
            "parameters": 85798656,
            "feature_dimension_per_visit": 1536,
            "selection_reason": "公开 non-gated、训练语料固定的强通用 SSL reference",
        },
    }
    observed_models = {
        str(entry["model"]): {key: value for key, value in entry.items() if key != "model"}
        for entry in models
    }
    if len(observed_models) != 2 or observed_models != expected_models:
        raise ValueError("model provenance identities or values drifted")
    for entry in models:
        for key in ("checkpoint_sha256", "feature_sha256"):
            if not _is_sha256(entry[key]):
                raise ValueError(f"model provenance {key} is not SHA-256")
        if _strict_int(entry["parameters"], label="model parameters") <= 0:
            raise ValueError("model parameter count must be positive")
        if _strict_int(
            entry["feature_dimension_per_visit"], label="feature dimension"
        ) <= 0:
            raise ValueError("feature dimension must be positive")
    if contract["metric_domain_policy"] != _METRIC_DOMAIN_POLICY:
        raise ValueError("metric nullable/domain policy drifted")
    reporting_contract = contract["reporting_run_provenance"]
    expected_reporting_keys = {
        "schema_version",
        "summary_schema_version",
        "baseline_artifact_keys",
        "probe_artifact_keys",
        "summarizer_keys",
        "public_artifact_roles",
    }
    if not isinstance(reporting_contract, dict) or set(reporting_contract) != expected_reporting_keys:
        raise ValueError("reporting-run provenance contract schema drifted")
    if reporting_contract["schema_version"] != REPORTING_PROVENANCE_SCHEMA_VERSION:
        raise ValueError("reporting-run provenance schema version drifted")
    if reporting_contract["summary_schema_version"] != SUMMARY_SCHEMA_VERSION:
        raise ValueError("reporting-run summary schema version drifted")
    expected_baseline_artifacts = {"predictions", "selection", "metrics", "progress"}
    expected_probe_artifacts = {
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
    }
    expected_public_roles = {
        "paired_public",
        "summary_json",
        "results_summary_markdown",
        "timing_figure",
        "calibration_figure",
    }
    expected_summarizer_keys = {
        "protocol_version",
        "argv_sha256",
        "code_lock_sha256",
        "finalization_lock_sha256",
    }
    if (
        len(reporting_contract["baseline_artifact_keys"]) != len(expected_baseline_artifacts)
        or set(map(str, reporting_contract["baseline_artifact_keys"]))
        != expected_baseline_artifacts
        or len(reporting_contract["probe_artifact_keys"]) != len(expected_probe_artifacts)
        or set(map(str, reporting_contract["probe_artifact_keys"])) != expected_probe_artifacts
        or len(reporting_contract["summarizer_keys"]) != len(expected_summarizer_keys)
        or set(map(str, reporting_contract["summarizer_keys"])) != expected_summarizer_keys
        or len(reporting_contract["public_artifact_roles"]) != len(expected_public_roles)
        or set(map(str, reporting_contract["public_artifact_roles"])) != expected_public_roles
    ):
        raise ValueError("reporting-run artifact key contract drifted")
    handoff = contract["git_handoff"]
    expected_handoff_keys = {
        "schema_version",
        "branch",
        "remote",
        "remote_ref",
        "status_values",
        "maximum_sanitized_error_characters",
        "tracked_artifact_roles",
    }
    if not isinstance(handoff, dict) or set(handoff) != expected_handoff_keys:
        raise ValueError("git-handoff contract nested schema drifted")
    if handoff["schema_version"] != GIT_HANDOFF_SCHEMA_VERSION:
        raise ValueError("git-handoff schema version drifted")
    if (
        handoff["branch"] != "feature/foundation-mri-baselines"
        or handoff["remote"] != "origin"
        or handoff["remote_ref"] != "refs/heads/feature/foundation-mri-baselines"
    ):
        raise ValueError("git-handoff fixed remote/branch contract drifted")
    if tuple(map(str, handoff["status_values"])) != (
        "SUBSTANTIVE_PUSH_OK",
        "GITHUB_PUSH_FAILED",
    ):
        raise ValueError("git-handoff status enum drifted")
    tracked = list(map(str, handoff["tracked_artifact_roles"]))
    expected_tracked = [
        role for role in _FIXED_PUBLIC_PATHS if role != "git_handoff_json"
    ]
    if tracked != expected_tracked:
        raise ValueError("git-handoff tracked artifact roles drifted")
    if _strict_int(
        handoff["maximum_sanitized_error_characters"],
        label="maximum sanitized error characters",
    ) != 1000:
        raise ValueError("git-handoff sanitized-error limit drifted")
    expected_rules = {
        "favorable": "delta_auroc_gt_0_and_delta_auprc_gt_0_and_delta_brier_le_0",
        "adverse": "delta_auroc_lt_0_and_delta_auprc_lt_0_and_delta_brier_ge_0",
        "otherwise": "mixed",
        "set_support": "every_preregistered_cell_favorable",
        "set_adverse": "every_preregistered_cell_adverse",
        "set_otherwise": "mixed",
        "q10_support": "complete_foundation_FTV_and_delta_tables_and_Q9_consistent_support",
        "q12_prioritize": "Q8_Q9_Q11_all_consistent_support",
        "q12_controlled_study": "Q11_consistent_support_and_Q8_Q9_neither_consistent_adverse",
        "q12_otherwise": "do_not_prioritize_replacement_from_this_experiment",
    }
    if contract["direction_rule"] != expected_rules:
        raise ValueError("fixed direction/conclusion rule tree drifted")
    if contract["publication_atomicity"] != {
        "coverage_receipt_is_commit_marker": True,
        "cross_directory_sigkill_transactional": False,
        "orphan_policy": "fail_closed_and_require_operator_audit",
    }:
        raise ValueError("publication atomicity contract drifted")
    expected_populations = {
        "full": {"analysis_population": "full_808", "n": 808, "positive": 275},
        "complete_case": {
            "analysis_population": "radiomics_complete_case_375",
            "n": 375,
            "positive": 110,
        },
    }
    populations = contract["populations"]
    if not isinstance(populations, dict) or set(populations) != set(expected_populations):
        raise ValueError("population contract exact schema drifted")
    for key, expected in expected_populations.items():
        population = populations[key]
        if not isinstance(population, dict) or set(population) != set(expected):
            raise ValueError(f"population {key} nested schema drifted")
        if (
            str(population["analysis_population"]) != expected["analysis_population"]
            or _strict_int(population["n"], label=f"population {key} n") != expected["n"]
            or _strict_int(population["positive"], label=f"population {key} positive")
            != expected["positive"]
        ):
            raise ValueError(f"population {key} values drifted")
    expected_bootstrap = {
        "seed": 2026,
        "replicates": 5000,
        "ci_level": 0.95,
        "minimum_valid_replicates": 4750,
    }
    bootstrap = contract["bootstrap"]
    if not isinstance(bootstrap, dict) or set(bootstrap) != set(expected_bootstrap):
        raise ValueError("bootstrap contract exact schema drifted")
    for key in ("seed", "replicates", "minimum_valid_replicates"):
        if _strict_int(bootstrap[key], label=f"bootstrap {key}") != expected_bootstrap[key]:
            raise ValueError(f"bootstrap {key} drifted")
    if not math.isclose(
        float(bootstrap["ci_level"]), expected_bootstrap["ci_level"], rel_tol=0.0, abs_tol=0.0
    ):
        raise ValueError("bootstrap CI level drifted")
    expected_families = {
        "clinical_gain": 36,
        "local_vs_global": 36,
        "foundation_vs_current_cnn": 48,
        "beyond_ftv": 12,
    }
    family_counts = contract["comparison_family_counts"]
    if not isinstance(family_counts, dict) or set(family_counts) != set(expected_families):
        raise ValueError("comparison-family count schema drifted")
    if {
        key: _strict_int(value, label=f"comparison family {key}")
        for key, value in family_counts.items()
    } != expected_families:
        raise ValueError("comparison-family counts drifted")
    expected_question_subsets = {
        "Q4": ("full-cohort foundation MRI-only LOCAL minus GLOBAL", 6),
        "Q5": ("full-cohort foundation LOCAL minus LOCAL0 MRI-only", 6),
        "Q8": ("full-cohort clinical plus foundation minus clinical-only", 12),
        "Q9": (
            "complete-case clinical plus FTV plus foundation minus clinical plus FTV",
            12,
        ),
        "Q11": ("full-cohort foundation minus matched current CNN MRI-only", 12),
    }
    question_subsets = contract["question_subsets"]
    if not isinstance(question_subsets, dict) or set(question_subsets) != set(
        expected_question_subsets
    ):
        raise ValueError("question-subset exact schema drifted")
    for question, (description, count) in expected_question_subsets.items():
        current = question_subsets[question]
        if not isinstance(current, dict) or set(current) != {
            "description",
            "expected_comparisons",
        }:
            raise ValueError(f"{question} question-subset nested schema drifted")
        if str(current["description"]) != description or _strict_int(
            current["expected_comparisons"], label=f"{question} expected comparisons"
        ) != count:
            raise ValueError(f"{question} question-subset values drifted")


def _source_axes(contract: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (str(entry["model"]), str(entry["spatial"]))
        for entry in contract["source_axes"]
    )


def _foundations(contract: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(value) for value in contract["expected_foundation_models"])


def _timings(contract: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(value) for value in contract["decision_points"])


def _population(contract: Mapping[str, Any], key: str) -> str:
    return str(contract["populations"][key]["analysis_population"])


def _expected_pcr_identities(contract: Mapping[str, Any]) -> set[tuple[str, ...]]:
    full = _population(contract, "full")
    complete = _population(contract, "complete_case")
    identities: set[tuple[str, ...]] = set()
    for timing in _timings(contract):
        identities.add(("pCR", "clinical_only", "NONE", timing, full))
        for source, spatial in _source_axes(contract):
            for suffix in ("mri_only", "mri_clinical"):
                identities.add(("pCR", f"{source}_{suffix}", spatial, timing, full))
        identities.add(("pCR", "clinical_only_paired", "NONE", timing, complete))
        for model in ("ftv_only", "radiomics_only", "clinical_ftv", "clinical_radiomics"):
            identities.add(("pCR", model, "TABULAR", timing, complete))
        for source, spatial in _source_axes(contract):
            for suffix in (
                "mri_only_paired",
                "mri_clinical_paired",
                "mri_ftv",
                "mri_clinical_ftv",
            ):
                identities.add(("pCR", f"{source}_{suffix}", spatial, timing, complete))
    return identities


def _expected_phenotype_identities(contract: Mapping[str, Any]) -> set[tuple[str, ...]]:
    full = _population(contract, "full")
    return {
        (target, source, spatial, "T0", full)
        for source, spatial in _source_axes(contract)
        for target in ("HR", "HER2")
    }


def _expected_subtype_identities(contract: Mapping[str, Any]) -> set[tuple[str, ...]]:
    full = _population(contract, "full")
    return {
        ("HR_HER2_subtype", source, spatial, "T0", full)
        for source, spatial in _source_axes(contract)
    }


def _expected_ftv_identities(contract: Mapping[str, Any]) -> set[tuple[str, ...]]:
    complete = _population(contract, "complete_case")
    endpoints = (
        *(("static", endpoint) for endpoint in ("T0", "T1", "T2", "T3")),
        *(("delta", endpoint) for endpoint in ("T0-T1", "T1-T2", "T2-T3")),
    )
    return {
        ("FTV", source, spatial, task, endpoint, complete)
        for source, spatial in _source_axes(contract)
        for task, endpoint in endpoints
    }


def _comparison_id(
    family: str,
    population: str,
    timing: str,
    reference_model: str,
    reference_spatial: str,
    candidate_model: str,
    candidate_spatial: str,
) -> str:
    raw = "|".join(
        (
            family,
            population,
            timing,
            f"{reference_model}@{reference_spatial}",
            f"{candidate_model}@{candidate_spatial}",
        )
    )
    return f"{family}:{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:12]}"


def _expected_comparison_specs(contract: Mapping[str, Any]) -> tuple[Mapping[str, str], ...]:
    full = _population(contract, "full")
    complete = _population(contract, "complete_case")
    foundations = _foundations(contract)
    specs: list[dict[str, str]] = []

    def add(
        family: str,
        estimand: str,
        population: str,
        timing: str,
        reference_model: str,
        reference_spatial: str,
        candidate_model: str,
        candidate_spatial: str,
    ) -> None:
        specs.append(
            {
                "comparison_id": _comparison_id(
                    family,
                    population,
                    timing,
                    reference_model,
                    reference_spatial,
                    candidate_model,
                    candidate_spatial,
                ),
                "family": family,
                "estimand": estimand,
                "analysis_population": population,
                "timing": timing,
                "reference_model": reference_model,
                "reference_spatial": reference_spatial,
                "candidate_model": candidate_model,
                "candidate_spatial": candidate_spatial,
            }
        )

    axes = _source_axes(contract)
    for timing in _timings(contract):
        for source, spatial in axes:
            add(
                "clinical_gain",
                "clinical_plus_MRI_minus_clinical",
                full,
                timing,
                "clinical_only",
                "NONE",
                f"{source}_mri_clinical",
                spatial,
            )
            add(
                "clinical_gain",
                "clinical_plus_MRI_minus_clinical",
                complete,
                timing,
                "clinical_only_paired",
                "NONE",
                f"{source}_mri_clinical_paired",
                spatial,
            )
        for source in foundations:
            for population, suffixes in (
                (full, ("mri_only", "mri_clinical")),
                (complete, ("mri_only_paired", "mri_clinical_paired")),
            ):
                for suffix in suffixes:
                    add(
                        "local_vs_global",
                        f"LOCAL_minus_GLOBAL_{suffix}",
                        population,
                        timing,
                        f"{source}_{suffix}",
                        "GLOBAL",
                        f"{source}_{suffix}",
                        "LOCAL",
                    )
        for population, suffixes in (
            (full, ("mri_only", "mri_clinical")),
            (complete, ("mri_only_paired", "mri_clinical_paired")),
        ):
            for suffix in suffixes:
                add(
                    "local_vs_global",
                    f"LOCAL_minus_GLOBAL_current_CNN_{suffix}",
                    population,
                    timing,
                    f"GAP0_{suffix}",
                    "GLOBAL",
                    f"LOCAL0_{suffix}",
                    "LOCAL",
                )
        for source in foundations:
            for spatial, current in (("GLOBAL", "GAP0"), ("LOCAL", "LOCAL0")):
                for population, suffixes in (
                    (full, ("mri_only", "mri_clinical")),
                    (complete, ("mri_only_paired", "mri_clinical_paired")),
                ):
                    for suffix in suffixes:
                        add(
                            "foundation_vs_current_cnn",
                            f"foundation_minus_current_CNN_{suffix}",
                            population,
                            timing,
                            f"{current}_{suffix}",
                            spatial,
                            f"{source}_{suffix}",
                            spatial,
                        )
            for spatial in ("GLOBAL", "LOCAL"):
                add(
                    "beyond_ftv",
                    "clinical_FTV_foundation_minus_clinical_FTV",
                    complete,
                    timing,
                    "clinical_ftv",
                    "TABULAR",
                    f"{source}_mri_clinical_ftv",
                    spatial,
                )
    return tuple(specs)


def _identity_set(frame: pd.DataFrame, columns: Sequence[str]) -> set[tuple[str, ...]]:
    return {
        tuple(str(value) for value in row)
        for row in frame.loc[:, list(columns)].itertuples(index=False, name=None)
    }


def _require_schema(frame: pd.DataFrame, expected: set[str], *, label: str) -> None:
    observed = set(map(str, frame.columns))
    if observed != expected:
        raise ValueError(
            f"{label} public schema drifted: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _validate_provenance(frame: pd.DataFrame, contract: Mapping[str, Any], *, label: str) -> None:
    if set(_strict_int_array(frame["split_seed"], label=f"{label} split_seed")) != {
        _strict_int(contract["split_seed"], label="contract split seed")
    }:
        raise ValueError(f"{label} split seed drifted")
    hashes = set(frame["fold_manifest_sha256"].astype(str))
    if any(not _is_sha256(value) for value in hashes) or hashes != {
        str(contract["fold_manifest_sha256"])
    }:
        raise ValueError(f"{label} fold-manifest hash drifted")
    if set(_strict_int_array(frame["n_folds"], label=f"{label} n_folds")) != {
        _strict_int(contract["n_folds"], label="contract n_folds")
    }:
        raise ValueError(f"{label} outer-fold count drifted")


def _assert_json_matches_pooled_csv(
    summary_frame: pd.DataFrame,
    public_frame: pd.DataFrame,
    *,
    identity: Sequence[str],
    label: str,
) -> None:
    pooled = public_frame.loc[public_frame["aggregation"].astype(str).eq("pooled_oof")].copy()
    order = [*identity, "aggregation"]
    left = summary_frame.sort_values(order, kind="stable").reset_index(drop=True)
    right = pooled.sort_values(order, kind="stable").reset_index(drop=True)
    if list(left.columns) != list(right.columns):
        right = right.loc[:, list(left.columns)]
    try:
        pd.testing.assert_frame_equal(
            left,
            right,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as error:
        raise ValueError(f"{label} pooled JSON/CSV values drifted") from error


def _validate_public_metric_csv(
    frame: pd.DataFrame,
    *,
    expected_schema: set[str],
    identity: Sequence[str],
    expected_rows: int,
    contract: Mapping[str, Any],
    label: str,
) -> None:
    _require_schema(frame, expected_schema, label=label)
    if len(frame) != expected_rows:
        raise ValueError(f"{label} row count drifted: {len(frame)} != {expected_rows}")
    if set(frame["aggregation"].astype(str)) != {"pooled_oof", "outer_fold_macro"}:
        raise ValueError(f"{label} aggregation contract drifted")
    if frame.duplicated([*identity, "aggregation"]).any():
        raise ValueError(f"{label} contains duplicate aggregate identities")
    sizes = frame.groupby(list(identity), sort=False, dropna=False).size()
    if set(sizes.astype(int)) != {2}:
        raise ValueError(f"{label} must contain pooled and fold-macro rows for every identity")
    _validate_provenance(frame, contract, label=label)


def _validate_identity_strings(
    frame: pd.DataFrame, columns: Sequence[str], *, label: str
) -> None:
    for column in columns:
        if frame[column].isna().any():
            raise ValueError(f"{label} identity {column} contains null")
        values = frame[column].astype(str)
        if any(not value or value != value.strip() for value in values):
            raise ValueError(f"{label} identity {column} is empty or padded")


def _validate_binary_domains(frame: pd.DataFrame, *, label: str) -> None:
    _validate_identity_strings(frame, _BINARY_IDENTITY, label=label)
    n = _strict_int_array(frame["n"], label=f"{label} n")
    positive = _strict_int_array(frame["positive"], label=f"{label} positive")
    if np.any(n <= 0) or np.any(positive <= 0) or np.any(positive >= n):
        raise ValueError(f"{label} n/positive domain is invalid")
    for metric in ("auroc", "auprc", "brier", "ece_10bin"):
        _bounded_array(frame[metric], label=f"{label} {metric}", lower=0.0, upper=1.0)
    _nullable_finite_array(
        frame["calibration_slope"], label=f"{label} calibration_slope"
    )
    _nullable_finite_array(
        frame["calibration_intercept"], label=f"{label} calibration_intercept"
    )
    if set(frame["ece_bin_contract"].astype(str)) != {"10_equal_width_bins_[0,1]"}:
        raise ValueError(f"{label} ECE contract drifted")


def _validate_subtype_domains(frame: pd.DataFrame, *, label: str) -> None:
    _validate_identity_strings(frame, _BINARY_IDENTITY, label=label)
    n = _strict_int_array(frame["n"], label=f"{label} n")
    if np.any(n <= 0):
        raise ValueError(f"{label} n must be positive")
    for metric in ("macro_ovr_auroc", "macro_ovr_auprc", "toplabel_ece_10bin", "accuracy"):
        _bounded_array(frame[metric], label=f"{label} {metric}", lower=0.0, upper=1.0)
    _bounded_array(
        frame["multiclass_brier"],
        label=f"{label} multiclass_brier",
        lower=0.0,
        upper=2.0,
    )
    if set(frame["ece_bin_contract"].astype(str)) != {
        "10_equal_width_bins_[0,1]_top_label"
    }:
        raise ValueError(f"{label} ECE contract drifted")


def _validate_ftv_domains(frame: pd.DataFrame, *, label: str) -> None:
    _validate_identity_strings(frame, _FTV_IDENTITY, label=label)
    n = _strict_int_array(frame["n"], label=f"{label} n")
    if np.any(n <= 0):
        raise ValueError(f"{label} n must be positive")
    for metric in ("r2", "rmse", "mae", "b0_rmse", "calibration_mean_bias"):
        values = _finite_array(frame[metric], label=f"{label} {metric}")
        if metric in {"rmse", "mae", "b0_rmse"} and np.any(values < 0.0):
            raise ValueError(f"{label} {metric} must be nonnegative")
    for metric in ("spearman", "pearson"):
        values = _nullable_finite_array(frame[metric], label=f"{label} {metric}")
        finite = values[np.isfinite(values)]
        if np.any(finite < -1.0) or np.any(finite > 1.0):
            raise ValueError(f"{label} {metric} must be null or within [-1, 1]")
    for metric in ("rmse_gain_over_b0", "calibration_slope", "calibration_intercept"):
        _nullable_finite_array(frame[metric], label=f"{label} {metric}")


def _validate_population_counts(
    frame: pd.DataFrame,
    contract: Mapping[str, Any],
    *,
    require_positive: bool,
    label: str,
) -> None:
    for key in ("full", "complete_case"):
        population = contract["populations"][key]
        selected = frame.loc[
            frame["analysis_population"].astype(str).eq(str(population["analysis_population"]))
        ]
        if selected.empty:
            continue
        if set(_strict_int_array(selected["n"], label=f"{label} n")) != {
            _strict_int(population["n"], label=f"population {key} n")
        }:
            raise ValueError(f"{label} n drifted in {key}")
        if require_positive and set(_strict_int_array(selected["positive"], label=f"{label} positive")) != {
            _strict_int(population["positive"], label=f"population {key} positive")
        }:
            raise ValueError(f"{label} positive count drifted in {key}")


def _normalise_records(frame: pd.DataFrame, columns: Sequence[str]) -> list[dict[str, Any]]:
    ordered = frame.sort_values(list(columns), kind="stable").reset_index(drop=True)
    records: list[dict[str, Any]] = []
    for raw in ordered.to_dict(orient="records"):
        record: dict[str, Any] = {}
        for key, value in raw.items():
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, float) and not math.isfinite(value):
                value = None
            record[str(key)] = value
        records.append(record)
    return records


def _records_sha256(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    payload = _normalise_records(frame.loc[:, list(columns)], columns)
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validate_paired(
    paired: pd.DataFrame,
    contract: Mapping[str, Any],
    pcr: pd.DataFrame,
) -> tuple[Mapping[str, str], ...]:
    _require_schema(paired, _PAIRED_SCHEMA, label="paired comparisons")
    _validate_identity_strings(
        paired,
        (*_COMPARISON_SPEC_COLUMNS, "metric", "inference_scope"),
        label="paired comparisons",
    )
    expected_specs = _expected_comparison_specs(contract)
    counts = contract["expected_counts"]
    if len(expected_specs) != int(counts["paired_comparisons"]):
        raise AssertionError("internal comparison generator count drifted")
    if len(paired) != int(counts["paired_metric_rows"]):
        raise ValueError("paired comparison metric-row count drifted")
    if paired.duplicated(["comparison_id", "metric"]).any():
        raise ValueError("paired comparisons contain duplicate comparison/metric rows")
    expected_spec_rows = {
        tuple(spec[column] for column in _COMPARISON_SPEC_COLUMNS)
        for spec in expected_specs
    }
    observed_spec_rows = {
        tuple(str(value) for value in row)
        for row in paired.loc[:, list(_COMPARISON_SPEC_COLUMNS)]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }
    if observed_spec_rows != expected_spec_rows:
        missing = len(expected_spec_rows - observed_spec_rows)
        extra = len(observed_spec_rows - expected_spec_rows)
        raise ValueError(f"paired comparison identities drifted: missing={missing}, extra={extra}")
    metric_sets = paired.groupby("comparison_id", sort=False)["metric"].agg(
        lambda values: frozenset(map(str, values))
    )
    if set(metric_sets) != {frozenset(_METRICS)}:
        raise ValueError("every paired comparison must contain AUROC/AUPRC/Brier")
    unique_specs = paired.drop_duplicates("comparison_id")
    family_counts = Counter(unique_specs["family"].astype(str))
    expected_family = {str(key): int(value) for key, value in contract["comparison_family_counts"].items()}
    if dict(family_counts) != expected_family:
        raise ValueError("paired comparison family counts drifted")
    bootstrap = contract["bootstrap"]
    if set(_strict_int_array(paired["bootstrap_seed"], label="paired bootstrap_seed")) != {
        _strict_int(bootstrap["seed"], label="bootstrap seed")
    }:
        raise ValueError("paired bootstrap seed drifted")
    if set(
        _strict_int_array(paired["bootstrap_replicates"], label="paired bootstrap_replicates")
    ) != {_strict_int(bootstrap["replicates"], label="bootstrap replicates")}:
        raise ValueError("paired bootstrap replicate count drifted")
    if not np.allclose(
        _finite_array(paired["ci_level"], label="paired ci_level"),
        float(bootstrap["ci_level"]),
        atol=0.0,
        rtol=0.0,
    ):
        raise ValueError("paired CI level drifted")
    valid = _strict_int_array(
        paired["valid_bootstrap_replicates"], label="paired valid_bootstrap_replicates"
    )
    if np.any(valid < _strict_int(bootstrap["minimum_valid_replicates"], label="minimum valid replicates")) or np.any(
        valid > _strict_int(bootstrap["replicates"], label="bootstrap replicates")
    ):
        raise ValueError("paired valid-bootstrap count is outside the frozen bounds")
    def parse_boolean(value: Any) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        lowered = str(value).strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        raise ValueError("paired higher_is_better contains a non-boolean value")

    for metric, expected_higher in (("auroc", True), ("auprc", True), ("brier", False)):
        observed = {
            parse_boolean(value)
            for value in paired.loc[paired["metric"].astype(str).eq(metric), "higher_is_better"]
        }
        if observed != {expected_higher}:
            raise ValueError("paired metric direction metadata drifted")
    for column in (
        "reference_value",
        "candidate_value",
        "delta_candidate_minus_reference",
        "ci_low",
        "ci_high",
    ):
        values = pd.to_numeric(paired[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"paired comparisons contain nonfinite {column}")
    reference = _bounded_array(
        paired["reference_value"], label="paired reference_value", lower=0.0, upper=1.0
    )
    candidate = _bounded_array(
        paired["candidate_value"], label="paired candidate_value", lower=0.0, upper=1.0
    )
    delta = _bounded_array(
        paired["delta_candidate_minus_reference"],
        label="paired delta_candidate_minus_reference",
        lower=-1.0,
        upper=1.0,
    )
    if not np.allclose(delta, candidate - reference, rtol=1e-12, atol=1e-12):
        raise ValueError("paired delta is not candidate minus reference")
    ci_low = _bounded_array(paired["ci_low"], label="paired ci_low", lower=-1.0, upper=1.0)
    ci_high = _bounded_array(paired["ci_high"], label="paired ci_high", lower=-1.0, upper=1.0)
    if np.any(ci_low > ci_high):
        raise ValueError("paired CI lower bound exceeds upper bound")
    if set(paired["inference_scope"].astype(str)) != {str(contract["inference_scope"])}:
        raise ValueError("paired inference scope drifted")
    for key in ("full", "complete_case"):
        population = contract["populations"][key]
        selected = paired.loc[
            paired["analysis_population"].astype(str).eq(str(population["analysis_population"]))
        ]
        if set(_strict_int_array(selected["n_paired"], label="paired n_paired")) != {
            _strict_int(population["n"], label=f"population {key} n")
        }:
            raise ValueError(f"paired n drifted in {key}")
        if set(_strict_int_array(selected["positive"], label="paired positive")) != {
            _strict_int(population["positive"], label=f"population {key} positive")
        }:
            raise ValueError(f"paired positive count drifted in {key}")
    pcr_lookup = {
        (
            str(row.model),
            str(row.spatial),
            str(row.timing),
            str(row.analysis_population),
            metric,
        ): float(getattr(row, metric))
        for row in pcr.itertuples(index=False)
        for metric in _METRICS
    }
    if len(pcr_lookup) != len(pcr) * len(_METRICS):
        raise ValueError("pCR pooled metric lookup contains duplicate identities")
    for row in paired.itertuples(index=False):
        metric = str(row.metric)
        reference_key = (
            str(row.reference_model),
            str(row.reference_spatial),
            str(row.timing),
            str(row.analysis_population),
            metric,
        )
        candidate_key = (
            str(row.candidate_model),
            str(row.candidate_spatial),
            str(row.timing),
            str(row.analysis_population),
            metric,
        )
        if reference_key not in pcr_lookup or candidate_key not in pcr_lookup:
            raise ValueError("paired comparison cannot be linked to pCR pooled identities")
        if not math.isclose(
            float(row.reference_value), pcr_lookup[reference_key], rel_tol=1e-12, abs_tol=1e-12
        ) or not math.isclose(
            float(row.candidate_value), pcr_lookup[candidate_key], rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError("paired reference/candidate value drifted from pCR pooled metric")
    return expected_specs


def _compact_comparisons(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for comparison_id, group in paired.groupby("comparison_id", sort=False):
        if set(group["metric"].astype(str)) != set(_METRICS):
            raise ValueError(f"comparison {comparison_id} does not have all metrics")
        first = group.iloc[0]
        row = {column: first[column] for column in _COMPARISON_SPEC_COLUMNS}
        row["n_paired"] = int(first["n_paired"])
        row["positive"] = int(first["positive"])
        for metric in _METRICS:
            current = group.loc[group["metric"].astype(str).eq(metric)].iloc[0]
            row[f"{metric}_reference"] = float(current["reference_value"])
            row[f"{metric}_candidate"] = float(current["candidate_value"])
            row[f"{metric}_delta"] = float(current["delta_candidate_minus_reference"])
            row[f"{metric}_ci_low"] = float(current["ci_low"])
            row[f"{metric}_ci_high"] = float(current["ci_high"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        [
            "family",
            "analysis_population",
            "timing",
            "candidate_model",
            "candidate_spatial",
            "reference_model",
            "reference_spatial",
        ],
        kind="stable",
    ).reset_index(drop=True)


def _cell_direction(row: Mapping[str, Any]) -> str:
    auroc = float(row["auroc_delta"])
    auprc = float(row["auprc_delta"])
    brier = float(row["brier_delta"])
    if auroc > 0.0 and auprc > 0.0 and brier <= 0.0:
        return "favorable"
    if auroc < 0.0 and auprc < 0.0 and brier >= 0.0:
        return "adverse"
    return "mixed"


def _ci_direction(metric: str, lower: float, upper: float) -> str:
    if metric in {"auroc", "auprc"}:
        if lower > 0.0:
            return "favorable"
        if upper < 0.0:
            return "adverse"
    else:
        if upper < 0.0:
            return "favorable"
        if lower > 0.0:
            return "adverse"
    return "inconclusive"


def _question_subsets(
    compact: pd.DataFrame, contract: Mapping[str, Any]
) -> Mapping[str, pd.DataFrame]:
    full = _population(contract, "full")
    complete = _population(contract, "complete_case")
    foundations = set(_foundations(contract))

    def candidate_is(suffix: str) -> pd.Series:
        allowed = {f"{source}_{suffix}" for source in foundations}
        return compact["candidate_model"].astype(str).isin(allowed)

    return {
        "Q4": compact.loc[
            compact["family"].astype(str).eq("local_vs_global")
            & compact["analysis_population"].astype(str).eq(full)
            & compact["estimand"].astype(str).eq("LOCAL_minus_GLOBAL_mri_only")
            & candidate_is("mri_only")
        ].copy(),
        "Q5": compact.loc[
            compact["family"].astype(str).eq("foundation_vs_current_cnn")
            & compact["analysis_population"].astype(str).eq(full)
            & compact["estimand"].astype(str).eq("foundation_minus_current_CNN_mri_only")
            & compact["candidate_spatial"].astype(str).eq("LOCAL")
            & candidate_is("mri_only")
        ].copy(),
        "Q8": compact.loc[
            compact["family"].astype(str).eq("clinical_gain")
            & compact["analysis_population"].astype(str).eq(full)
            & candidate_is("mri_clinical")
        ].copy(),
        "Q9": compact.loc[
            compact["family"].astype(str).eq("beyond_ftv")
            & compact["analysis_population"].astype(str).eq(complete)
            & candidate_is("mri_clinical_ftv")
        ].copy(),
        "Q11": compact.loc[
            compact["family"].astype(str).eq("foundation_vs_current_cnn")
            & compact["analysis_population"].astype(str).eq(full)
            & compact["estimand"].astype(str).eq("foundation_minus_current_CNN_mri_only")
            & candidate_is("mri_only")
        ].copy(),
    }


def fixed_question_evidence(
    paired: pd.DataFrame, contract: Mapping[str, Any]
) -> Mapping[str, Any]:
    compact = _compact_comparisons(paired)
    output: dict[str, Any] = {}
    for question, subset in _question_subsets(compact, contract).items():
        expected = int(contract["question_subsets"][question]["expected_comparisons"])
        if len(subset) != expected:
            raise ValueError(f"{question} fixed evidence subset drifted: {len(subset)} != {expected}")
        directions = [_cell_direction(row) for row in subset.to_dict(orient="records")]
        direction_counts = Counter(directions)
        if direction_counts.get("favorable", 0) == expected:
            set_classification = "consistent_support"
        elif direction_counts.get("adverse", 0) == expected:
            set_classification = "consistent_adverse"
        else:
            set_classification = "mixed"
        ci_counts: dict[str, dict[str, int]] = {}
        for metric in _METRICS:
            values = [
                _ci_direction(metric, float(row[f"{metric}_ci_low"]), float(row[f"{metric}_ci_high"]))
                for row in subset.to_dict(orient="records")
            ]
            current = Counter(values)
            ci_counts[metric] = {
                key: int(current.get(key, 0))
                for key in ("favorable", "inconclusive", "adverse")
            }
        output[question] = {
            "description": str(contract["question_subsets"][question]["description"]),
            "expected_comparisons": expected,
            "observed_comparisons": int(len(subset)),
            "comparison_ids": sorted(subset["comparison_id"].astype(str)),
            "point_direction_counts": {
                key: int(direction_counts.get(key, 0))
                for key in ("favorable", "mixed", "adverse")
            },
            "ci_direction_counts": ci_counts,
            "set_classification": set_classification,
        }
    q9 = str(output["Q9"]["set_classification"])
    output["Q10"] = {
        "set_classification": (
            "beyond_tumor_size_descriptive_support"
            if q9 == "consistent_support"
            else "beyond_tumor_size_not_established"
        ),
        "rule": "requires complete foundation FTV/DeltaFTV tables and Q9 consistent support",
    }
    q8 = str(output["Q8"]["set_classification"])
    q11 = str(output["Q11"]["set_classification"])
    if q8 == q9 == q11 == "consistent_support":
        recommendation = "prioritize_integration_and_external_validation"
    elif (
        q11 == "consistent_support"
        and q8 != "consistent_adverse"
        and q9 != "consistent_adverse"
    ):
        recommendation = "controlled_integration_study_only"
    else:
        recommendation = "do_not_prioritize_replacement_from_this_experiment"
    output["Q12"] = {
        "recommendation": recommendation,
        "rule_inputs": {"Q8": q8, "Q9": q9, "Q11": q11},
    }
    return output


def _validate_summary_header(summary: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    if set(summary) != _SUMMARY_KEYS:
        raise ValueError(
            "results summary top-level schema drifted: "
            f"missing={sorted(_SUMMARY_KEYS - set(summary))}, "
            f"extra={sorted(set(summary) - _SUMMARY_KEYS)}"
        )
    if summary["schema_version"] != SUMMARY_SCHEMA_VERSION:
        raise ValueError("results summary schema version drifted")
    if summary["reported_candidate_policy"] != contract["candidate_policy"]:
        raise ValueError("results summary candidate policy drifted")
    if summary["inference_scope"] != contract["inference_scope"]:
        raise ValueError("results summary inference scope drifted")
    foundation = list(map(str, summary["foundation_models"]))
    expected_foundation = set(_foundations(contract))
    if (
        len(foundation) != len(expected_foundation)
        or len(set(foundation)) != len(foundation)
        or set(foundation) != expected_foundation
    ):
        raise ValueError("formal foundation set is not the exact frozen two-model set")
    current = list(map(str, summary["current_cnn_models"]))
    expected_current = set(map(str, contract["current_cnn_models"]))
    if (
        len(current) != len(expected_current)
        or len(set(current)) != len(current)
        or set(current) != expected_current
    ):
        raise ValueError("results summary current-CNN set drifted")
    if _strict_int(
        summary["resolved_comparison_count"], label="resolved_comparison_count"
    ) != _strict_int(
        contract["expected_counts"]["paired_comparisons"],
        label="expected paired comparisons",
    ):
        raise ValueError("results summary resolved-comparison count drifted")
    if not isinstance(summary["cohorts"], dict) or set(summary["cohorts"]) != {
        "full",
        "complete_case",
    }:
        raise ValueError("results summary cohort schema drifted")
    for key in ("full", "complete_case"):
        observed = summary["cohorts"][key]
        expected = contract["populations"][key]
        if not isinstance(observed, dict) or set(observed) != {"analysis_population", "n"}:
            raise ValueError(f"results summary cohort nested schema drifted for {key}")
        if str(observed["analysis_population"]) != str(expected["analysis_population"]) or _strict_int(
            observed["n"], label=f"summary cohort {key} n"
        ) != _strict_int(expected["n"], label=f"contract cohort {key} n"):
            raise ValueError(f"results summary cohort metadata drifted for {key}")
    comparison_contract = summary["comparison_contract"]
    if _canonical_pretty_json_sha256(comparison_contract) != str(
        contract["comparison_contract_canonical_sha256"]
    ):
        raise ValueError("results summary static comparison contract SHA drifted")


def _validate_digest_mapping(
    value: Any, expected_keys: set[str], *, label: str
) -> Mapping[str, str]:
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError(f"{label} digest mapping schema drifted")
    output = {str(key): str(digest) for key, digest in value.items()}
    if any(not _is_sha256(digest) for digest in output.values()):
        raise ValueError(f"{label} contains a non-SHA256 digest")
    return output


def _validate_reporting_run_provenance(
    payload: Mapping[str, Any],
    contract: Mapping[str, Any],
    snapshots: _InputSnapshots,
) -> Mapping[str, Any]:
    expected_top = {
        "schema_version",
        "summary_schema_version",
        "comparison_contract_canonical_sha256",
        "baseline_v2",
        "probe_v3",
        "summarizer",
        "public_artifact_sha256",
    }
    if set(payload) != expected_top:
        raise ValueError("reporting-run provenance top-level schema drifted")
    reporting_contract = contract["reporting_run_provenance"]
    if payload["schema_version"] != reporting_contract["schema_version"]:
        raise ValueError("reporting-run provenance schema version drifted")
    if payload["summary_schema_version"] != SUMMARY_SCHEMA_VERSION:
        raise ValueError("reporting-run summary schema drifted")
    if payload["comparison_contract_canonical_sha256"] != contract[
        "comparison_contract_canonical_sha256"
    ]:
        raise ValueError("reporting-run comparison-contract SHA drifted")
    stage_schema = {
        "protocol_version",
        "evaluation_lock_sha256",
        "run_receipt_sha256",
        "argv_sha256",
        "artifact_sha256",
    }
    stage_artifacts: dict[str, Mapping[str, str]] = {}
    for stage, version, artifact_key_field in (
        ("baseline_v2", "v2", "baseline_artifact_keys"),
        ("probe_v3", "v3", "probe_artifact_keys"),
    ):
        current = payload[stage]
        if not isinstance(current, dict) or set(current) != stage_schema:
            raise ValueError(f"reporting-run {stage} nested schema drifted")
        if current["protocol_version"] != version:
            raise ValueError(f"reporting-run {stage} protocol version drifted")
        for key in ("evaluation_lock_sha256", "run_receipt_sha256", "argv_sha256"):
            if not _is_sha256(current[key]):
                raise ValueError(f"reporting-run {stage}.{key} is not SHA-256")
        stage_artifacts[stage] = _validate_digest_mapping(
            current["artifact_sha256"],
            set(map(str, reporting_contract[artifact_key_field])),
            label=f"reporting-run {stage} artifacts",
        )
    if stage_artifacts["baseline_v2"]["metrics"] != snapshots.sha256["baseline_public"]:
        raise ValueError("baseline-v2 public metric digest differs from baseline CSV")
    for artifact, role in (
        ("phenotype_metrics", "phenotype_public"),
        ("subtype_metrics", "subtype_public"),
        ("ftv_metrics", "ftv_public"),
    ):
        if stage_artifacts["probe_v3"][artifact] != snapshots.sha256[role]:
            raise ValueError(f"probe-v3 {artifact} digest differs from public CSV")
    summarizer = payload["summarizer"]
    expected_summarizer_keys = set(map(str, reporting_contract["summarizer_keys"]))
    if not isinstance(summarizer, dict) or set(summarizer) != expected_summarizer_keys:
        raise ValueError("reporting-run summarizer nested schema drifted")
    if summarizer["protocol_version"] != "v3":
        raise ValueError("reporting-run summarizer protocol version drifted")
    for key in ("argv_sha256", "code_lock_sha256", "finalization_lock_sha256"):
        if not _is_sha256(summarizer[key]):
            raise ValueError(f"reporting-run summarizer {key} is not SHA-256")
    if summarizer["finalization_lock_sha256"] != snapshots.sha256["finalization_lock"]:
        raise ValueError("reporting-run finalization-lock SHA differs from active lock bytes")
    public_roles = set(map(str, reporting_contract["public_artifact_roles"]))
    public_hashes = _validate_digest_mapping(
        payload["public_artifact_sha256"],
        public_roles,
        label="reporting-run public artifacts",
    )
    for role, expected_hash in public_hashes.items():
        if snapshots.sha256.get(role) != expected_hash:
            raise ValueError(f"reporting-run public artifact hash drifted for {role}")
    return payload


def _validate_finalization_lock(
    payload: Mapping[str, Any],
    contract: Mapping[str, Any],
    snapshots: _InputSnapshots,
) -> Mapping[str, Any]:
    expected_keys = {
        "schema_version",
        "parent_evaluation_lock_sha256",
        "formal_metric_values_unseen",
        "formal_argv",
        "locked_sha256",
        "fixed_public_paths",
        "summary_schema_version",
        "reporting_provenance_schema_version",
        "git_handoff_schema_version",
        "expected_counts",
    }
    if set(payload) != expected_keys:
        raise ValueError("finalization lock exact schema drifted")
    if payload["schema_version"] != "foundation_mri_finalization_lock_v1":
        raise ValueError("finalization lock schema version drifted")
    if payload["parent_evaluation_lock_sha256"] != (
        "b15f7023b7021f5c1169b51cf6bc8fe0cc1d9085102a61fbdb1d68589fe2edc5"
    ):
        raise ValueError("finalization lock parent v2 SHA drifted")
    if payload["formal_metric_values_unseen"] is not True:
        raise ValueError("finalization lock must declare formal metric values unseen")
    if payload["formal_argv"] != []:
        raise ValueError("formal finalizer argv must be exactly empty")
    locked_roles = {
        "finalization_module",
        "finalizer_cli",
        "contract_json",
        "template_markdown",
        "finalization_test",
    }
    locked = _validate_digest_mapping(
        payload["locked_sha256"], locked_roles, label="finalization lock code artifacts"
    )
    for role, expected_hash in locked.items():
        if snapshots.sha256.get(role) != expected_hash:
            raise ValueError(f"finalization lock byte drifted for {role}")
    if payload["fixed_public_paths"] != contract["fixed_public_paths"]:
        raise ValueError("finalization lock fixed public paths drifted")
    if payload["summary_schema_version"] != SUMMARY_SCHEMA_VERSION:
        raise ValueError("finalization lock summary schema drifted")
    if payload["reporting_provenance_schema_version"] != REPORTING_PROVENANCE_SCHEMA_VERSION:
        raise ValueError("finalization lock reporting provenance schema drifted")
    if payload["git_handoff_schema_version"] != GIT_HANDOFF_SCHEMA_VERSION:
        raise ValueError("finalization lock git-handoff schema drifted")
    expected_counts = {
        key: _strict_int(value, label=f"contract expected count {key}")
        for key, value in contract["expected_counts"].items()
    }
    if not isinstance(payload["expected_counts"], dict) or set(payload["expected_counts"]) != set(
        expected_counts
    ):
        raise ValueError("finalization lock expected-count schema drifted")
    observed_counts = {
        key: _strict_int(value, label=f"finalization lock expected count {key}")
        for key, value in payload["expected_counts"].items()
    }
    if observed_counts != expected_counts:
        raise ValueError("finalization lock expected counts drifted")
    return payload


def _default_git_runner(arguments: Sequence[str], cwd: Path) -> bytes:
    environment = os.environ.copy()
    environment["GIT_TERMINAL_PROMPT"] = "0"
    environment.setdefault("GIT_SSH_COMMAND", "ssh -oBatchMode=yes")
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            timeout=30,
        )
    except subprocess.TimeoutExpired as error:
        raise ValueError("git handoff verification command timed out") from error
    except subprocess.CalledProcessError as error:
        message = error.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"git handoff verification command failed: {message[:300]}") from error
    return result.stdout


def _git_text(runner: GitRunner, arguments: Sequence[str], cwd: Path, *, label: str) -> str:
    payload = runner(tuple(arguments), cwd)
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError(f"git runner returned non-bytes for {label}")
    try:
        return bytes(payload).decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise ValueError(f"git output for {label} is not UTF-8") from error


def _validate_sanitized_push_error(value: Any, contract: Mapping[str, Any]) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("failed push requires a nonempty sanitized error")
    maximum = _strict_int(
        contract["git_handoff"]["maximum_sanitized_error_characters"],
        label="sanitized error maximum",
    )
    if len(value) > maximum or any(ord(character) < 32 and character not in "\n\t" for character in value):
        raise ValueError("sanitized push error violates length/control-character contract")
    forbidden = re.compile(
        r"(?i)(github_pat_|ghp_|gho_|ghu_|ghs_|ghr_|authorization\s*:|bearer\s+|"
        r"token\s*=|password\s*=|https?://[^\s/@:]+:[^\s/@]+@|/home/|/data/|"
        r"[A-Za-z]:[\\/])"
    )
    if forbidden.search(value):
        raise ValueError("push error is not safely sanitized")
    return value


def _validate_git_handoff(
    snapshots: _InputSnapshots,
    contract: Mapping[str, Any],
    *,
    git_runner: GitRunner,
) -> tuple[_InputSnapshots, Mapping[str, Any]]:
    manifest = _parse_json_bytes(
        snapshots.payloads["git_handoff_json"], label="git handoff manifest"
    )
    expected_keys = {
        "schema_version",
        "content_commit_sha",
        "branch",
        "remote",
        "remote_ref",
        "substantive_push_status",
        "substantive_remote_ref_sha",
        "sanitized_push_error",
        "artifact_sha256",
    }
    if set(manifest) != expected_keys:
        raise ValueError("git handoff manifest exact schema drifted")
    handoff_contract = contract["git_handoff"]
    if manifest["schema_version"] != handoff_contract["schema_version"]:
        raise ValueError("git handoff manifest schema version drifted")
    content_sha = str(manifest["content_commit_sha"])
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", content_sha):
        raise ValueError("git handoff content commit SHA is invalid")
    for key in ("branch", "remote", "remote_ref"):
        if str(manifest[key]) != str(handoff_contract[key]):
            raise ValueError(f"git handoff {key} drifted")
    manifest_path = snapshots.paths["git_handoff_json"]
    root_text = _git_text(
        git_runner,
        ("rev-parse", "--show-toplevel"),
        manifest_path.parent,
        label="repository root",
    )
    repository_root = Path(root_text).resolve()
    fixed_paths = contract["fixed_public_paths"]
    for role, path in snapshots.paths.items():
        expected = (repository_root / str(fixed_paths[role])).resolve()
        if path.resolve() != expected:
            raise ValueError(f"formal public input path drifted for {role}")
    payloads = dict(snapshots.payloads)
    paths = dict(snapshots.paths)
    digests = dict(snapshots.sha256)
    for role in ("finalization_module", "finalizer_cli", "finalization_test"):
        path = repository_root / str(fixed_paths[role])
        _reject_private_path(path, label=role)
        if not path.is_file():
            raise FileNotFoundError(f"required finalizer source is missing: {path.name}")
        payload = path.read_bytes()
        paths[role] = path
        payloads[role] = payload
        digests[role] = _sha256_bytes(payload)
    extended = _InputSnapshots(paths=paths, payloads=payloads, sha256=digests)
    if _git_text(git_runner, ("rev-parse", "HEAD"), repository_root, label="HEAD") != content_sha:
        raise ValueError("git handoff content SHA is not current HEAD")
    if _git_text(
        git_runner, ("branch", "--show-current"), repository_root, label="branch"
    ) != str(manifest["branch"]):
        raise ValueError("git handoff branch is not the current branch")
    status = str(manifest["substantive_push_status"])
    if status not in set(map(str, handoff_contract["status_values"])):
        raise ValueError("git handoff substantive push status is invalid")
    if status == "SUBSTANTIVE_PUSH_OK":
        if manifest["sanitized_push_error"] is not None:
            raise ValueError("successful substantive push must not contain an error")
        if str(manifest["substantive_remote_ref_sha"]) != content_sha:
            raise ValueError("successful substantive push remote SHA differs from content SHA")
        remote_output = _git_text(
            git_runner,
            (
                "ls-remote",
                "--heads",
                str(manifest["remote"]),
                str(manifest["remote_ref"]),
            ),
            repository_root,
            label="remote ref",
        )
        expected_line = f"{content_sha}\t{manifest['remote_ref']}"
        if remote_output != expected_line:
            raise ValueError("substantive remote ref does not resolve to content SHA")
    else:
        if manifest["substantive_remote_ref_sha"] is not None:
            raise ValueError("failed substantive push must have null remote-ref SHA")
        _validate_sanitized_push_error(manifest["sanitized_push_error"], contract)
    tracked_roles = list(map(str, handoff_contract["tracked_artifact_roles"]))
    expected_relative_paths = {str(fixed_paths[role]) for role in tracked_roles}
    manifest_hashes = _validate_digest_mapping(
        manifest["artifact_sha256"],
        expected_relative_paths,
        label="git handoff artifacts",
    )
    for role in tracked_roles:
        relative = str(fixed_paths[role])
        working_payload = extended.payloads[role]
        if manifest_hashes[relative] != _sha256_bytes(working_payload):
            raise ValueError(f"git handoff working bytes drifted for {relative}")
        committed_payload = git_runner(("show", f"{content_sha}:{relative}"), repository_root)
        if not isinstance(committed_payload, (bytes, bytearray)) or bytes(committed_payload) != working_payload:
            raise ValueError(f"git content commit bytes drifted for {relative}")
    return extended, manifest


def _prepare_snapshots(
    inputs: FinalizationInputs, *, git_runner: GitRunner
) -> tuple[_InputSnapshots, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    snapshots = _snapshot_inputs(inputs)
    contract = _validate_contract_payload(
        _parse_json_bytes(snapshots.payloads["contract_json"], label="final-report contract")
    )
    snapshots, git_handoff = _validate_git_handoff(
        snapshots, contract, git_runner=git_runner
    )
    _validate_finalization_lock(
        _parse_json_bytes(
            snapshots.payloads["finalization_lock"], label="finalization lock"
        ),
        contract,
        snapshots,
    )
    reporting_provenance = _validate_reporting_run_provenance(
        _parse_json_bytes(
            snapshots.payloads["reporting_run_provenance"],
            label="reporting-run provenance",
        ),
        contract,
        snapshots,
    )
    return snapshots, contract, git_handoff, reporting_provenance


def _validate_from_snapshots(
    snapshots: _InputSnapshots,
    contract: Mapping[str, Any],
    git_handoff: Mapping[str, Any],
    reporting_provenance: Mapping[str, Any],
) -> tuple[ValidatedPublicResults, Mapping[str, str]]:
    # The publication marker and all of its hashes were validated before any
    # result payload is parsed.  Every parse below uses the same immutable byte
    # snapshot that supplied the corresponding SHA.
    summary = _parse_json_bytes(snapshots.payloads["summary_json"], label="results summary")
    _validate_summary_header(summary, contract)
    pcr = pd.DataFrame(summary["pcr_pooled_metrics"])
    phenotype = pd.DataFrame(summary["phenotype_pooled_metrics"])
    subtype = pd.DataFrame(summary["subtype_pooled_metrics"])
    ftv = pd.DataFrame(summary["ftv_pooled_metrics"])
    paired_json = pd.DataFrame(summary["paired_comparisons"])
    for frame, label in (
        (pcr, "pCR pooled summary"),
        (phenotype, "phenotype pooled summary"),
        (subtype, "subtype pooled summary"),
        (ftv, "FTV pooled summary"),
        (paired_json, "paired summary"),
    ):
        if frame.empty:
            raise ValueError(f"{label} is empty")
        _assert_public_frame(frame, label=label)
    _require_schema(pcr, _BINARY_SCHEMA, label="pCR pooled summary")
    _require_schema(phenotype, _BINARY_SCHEMA, label="phenotype pooled summary")
    _require_schema(subtype, _SUBTYPE_SCHEMA, label="subtype pooled summary")
    _require_schema(ftv, _FTV_SCHEMA, label="FTV pooled summary")
    _validate_binary_domains(pcr, label="pCR pooled summary")
    _validate_binary_domains(phenotype, label="phenotype pooled summary")
    _validate_subtype_domains(subtype, label="subtype pooled summary")
    _validate_ftv_domains(ftv, label="FTV pooled summary")
    if set(pcr["aggregation"].astype(str)) != {"pooled_oof"}:
        raise ValueError("pCR JSON must contain pooled OOF metrics only")
    if set(phenotype["aggregation"].astype(str)) != {"pooled_oof"}:
        raise ValueError("phenotype JSON must contain pooled OOF metrics only")
    if set(subtype["aggregation"].astype(str)) != {"pooled_oof"}:
        raise ValueError("subtype JSON must contain pooled OOF metrics only")
    if set(ftv["aggregation"].astype(str)) != {"pooled_oof"}:
        raise ValueError("FTV JSON must contain pooled OOF metrics only")
    expected_pcr = _expected_pcr_identities(contract)
    observed_pcr = _identity_set(pcr, _BINARY_IDENTITY)
    if observed_pcr != expected_pcr or len(pcr) != len(expected_pcr):
        raise ValueError(
            "pCR pooled identities drifted: "
            f"missing={len(expected_pcr - observed_pcr)}, extra={len(observed_pcr - expected_pcr)}"
        )
    expected_phenotype = _expected_phenotype_identities(contract)
    observed_phenotype = _identity_set(phenotype, _BINARY_IDENTITY)
    if observed_phenotype != expected_phenotype or len(phenotype) != len(expected_phenotype):
        raise ValueError("phenotype pooled identities drifted")
    expected_subtype = _expected_subtype_identities(contract)
    observed_subtype = _identity_set(subtype, _BINARY_IDENTITY)
    if observed_subtype != expected_subtype or len(subtype) != len(expected_subtype):
        raise ValueError("subtype pooled identities drifted")
    expected_ftv = _expected_ftv_identities(contract)
    observed_ftv = _identity_set(ftv, _FTV_IDENTITY)
    if observed_ftv != expected_ftv or len(ftv) != len(expected_ftv):
        raise ValueError("FTV pooled identities drifted")
    _validate_provenance(pcr, contract, label="pCR pooled summary")
    _validate_provenance(phenotype, contract, label="phenotype pooled summary")
    _validate_provenance(subtype, contract, label="subtype pooled summary")
    _validate_provenance(ftv, contract, label="FTV pooled summary")
    _validate_population_counts(pcr, contract, require_positive=True, label="pCR")
    _validate_population_counts(phenotype, contract, require_positive=False, label="phenotype")
    _validate_population_counts(subtype, contract, require_positive=False, label="subtype")
    _validate_population_counts(ftv, contract, require_positive=False, label="FTV")
    phenotype_positive_counts = phenotype.groupby("target", sort=True)["positive"].nunique()
    if set(phenotype_positive_counts.astype(int)) != {1}:
        raise ValueError("phenotype positive count changes across source/spatial axes")

    baseline_public = _parse_csv_bytes(
        snapshots.payloads["baseline_public"], label="baseline public metrics"
    )
    phenotype_public = _parse_csv_bytes(
        snapshots.payloads["phenotype_public"], label="phenotype public metrics"
    )
    subtype_public = _parse_csv_bytes(
        snapshots.payloads["subtype_public"], label="subtype public metrics"
    )
    ftv_public = _parse_csv_bytes(
        snapshots.payloads["ftv_public"], label="FTV public metrics"
    )
    paired_public = _parse_csv_bytes(
        snapshots.payloads["paired_public"], label="paired public comparisons"
    )
    counts = contract["expected_counts"]
    _validate_public_metric_csv(
        baseline_public,
        expected_schema=_BINARY_SCHEMA,
        identity=_BINARY_IDENTITY,
        expected_rows=_strict_int(counts["baseline_public_rows"], label="baseline expected rows"),
        contract=contract,
        label="baseline public metrics",
    )
    _validate_public_metric_csv(
        phenotype_public,
        expected_schema=_BINARY_SCHEMA,
        identity=_BINARY_IDENTITY,
        expected_rows=_strict_int(counts["phenotype_public_rows"], label="phenotype expected rows"),
        contract=contract,
        label="phenotype public metrics",
    )
    _validate_public_metric_csv(
        subtype_public,
        expected_schema=_SUBTYPE_SCHEMA,
        identity=_BINARY_IDENTITY,
        expected_rows=_strict_int(counts["subtype_public_rows"], label="subtype expected rows"),
        contract=contract,
        label="subtype public metrics",
    )
    _validate_public_metric_csv(
        ftv_public,
        expected_schema=_FTV_SCHEMA,
        identity=_FTV_IDENTITY,
        expected_rows=_strict_int(counts["ftv_public_rows"], label="FTV expected rows"),
        contract=contract,
        label="FTV public metrics",
    )
    _validate_binary_domains(baseline_public, label="baseline public metrics")
    _validate_binary_domains(phenotype_public, label="phenotype public metrics")
    _validate_subtype_domains(subtype_public, label="subtype public metrics")
    _validate_ftv_domains(ftv_public, label="FTV public metrics")
    public_phenotype_positive = phenotype_public.groupby("target", sort=True)[
        "positive"
    ].nunique()
    if set(public_phenotype_positive.astype(int)) != {1}:
        raise ValueError("public phenotype positive count changes across source/spatial axes")
    _assert_json_matches_pooled_csv(
        pcr, baseline_public, identity=_BINARY_IDENTITY, label="pCR"
    )
    _assert_json_matches_pooled_csv(
        phenotype, phenotype_public, identity=_BINARY_IDENTITY, label="phenotype"
    )
    _assert_json_matches_pooled_csv(
        subtype, subtype_public, identity=_BINARY_IDENTITY, label="subtype"
    )
    _assert_json_matches_pooled_csv(ftv, ftv_public, identity=_FTV_IDENTITY, label="FTV")
    expected_specs = _validate_paired(paired_public, contract, pcr)
    _require_schema(paired_json, _PAIRED_SCHEMA, label="paired JSON summary")
    try:
        pd.testing.assert_frame_equal(
            paired_json.sort_values(["comparison_id", "metric"], kind="stable").reset_index(drop=True),
            paired_public.loc[:, list(paired_json.columns)]
            .sort_values(["comparison_id", "metric"], kind="stable")
            .reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as error:
        raise ValueError("paired JSON/CSV values drifted") from error

    public_hashes = {
        key: snapshots.sha256[key]
        for key in ("baseline_public", "phenotype_public", "subtype_public", "ftv_public")
    }
    summary_hashes = _validate_digest_mapping(
        summary["input_sha256"],
        {
            "baseline_private",
            "baseline_public",
            "phenotype_private",
            "phenotype_public",
            "subtype_private",
            "subtype_public",
            "ftv_private",
            "ftv_public",
        },
        label="results summary input hashes",
    )
    for key, digest in public_hashes.items():
        if str(summary_hashes.get(key, "")) != digest:
            raise ValueError(f"results summary input hash drifted for {key}")
    baseline_artifacts = reporting_provenance["baseline_v2"]["artifact_sha256"]
    probe_artifacts = reporting_provenance["probe_v3"]["artifact_sha256"]
    expected_summary_lineage_hashes = {
        "baseline_private": baseline_artifacts["predictions"],
        "baseline_public": baseline_artifacts["metrics"],
        "phenotype_private": probe_artifacts["phenotype_predictions"],
        "phenotype_public": probe_artifacts["phenotype_metrics"],
        "subtype_private": probe_artifacts["subtype_predictions"],
        "subtype_public": probe_artifacts["subtype_metrics"],
        "ftv_private": probe_artifacts["ftv_predictions"],
        "ftv_public": probe_artifacts["ftv_metrics"],
    }
    if dict(summary_hashes) != expected_summary_lineage_hashes:
        raise ValueError("results summary input hashes drifted from mixed producer lineage")
    hashes = dict(snapshots.sha256)
    evidence = fixed_question_evidence(paired_public, contract)
    return (
        ValidatedPublicResults(
            summary=summary,
            pcr=pcr,
            phenotype=phenotype,
            subtype=subtype,
            ftv=ftv,
            paired=paired_public,
            comparison_specs=expected_specs,
            question_evidence=evidence,
            public_row_counts={
                "baseline_public_rows": int(len(baseline_public)),
                "phenotype_public_rows": int(len(phenotype_public)),
                "subtype_public_rows": int(len(subtype_public)),
                "ftv_public_rows": int(len(ftv_public)),
            },
            reporting_run_provenance=reporting_provenance,
            git_handoff=git_handoff,
        ),
        hashes,
    )


def validate_public_results(
    inputs: FinalizationInputs,
    *,
    git_runner: GitRunner | None = None,
) -> tuple[ValidatedPublicResults, Mapping[str, str]]:
    runner = _default_git_runner if git_runner is None else git_runner
    snapshots, contract, handoff, provenance = _prepare_snapshots(
        inputs, git_runner=runner
    )
    return _validate_from_snapshots(snapshots, contract, handoff, provenance)


def _format_number(value: Any, digits: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(number):
        return "NA"
    if number != 0.0 and abs(number) < 10.0 ** (-digits):
        return f"{number:.3e}"
    return f"{number:.{digits}f}"


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    def cell(value: Any) -> str:
        text = str(value)
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = text.replace("\r\n", "<br>").replace("\r", "<br>").replace("\n", "<br>")
        return text.replace("|", r"\|")

    materialised = [[cell(value) for value in row] for row in rows]
    lines = [
        "| " + " | ".join(cell(value) for value in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in materialised)
    return "\n".join(lines)


def _pretraining_map(contract: Mapping[str, Any]) -> Mapping[tuple[str, str], str]:
    return {
        (str(entry["model"]), str(entry["spatial"])): str(entry["pretraining"])
        for entry in contract["source_axes"]
    }


def _render_model_provenance(contract: Mapping[str, Any]) -> str:
    ordered = sorted(contract["model_provenance"], key=lambda entry: str(entry["model"]))
    return _markdown_table(
        (
            "Model",
            "Pretraining/domain",
            "Source revision",
            "License",
            "Parameters",
            "Feature dim/visit",
            "Checkpoint SHA-256",
            "Feature SHA-256",
            "Selection reason",
        ),
        (
            (
                entry["display_name"],
                entry["pretraining_domain"],
                entry["source_revision"],
                entry["license"],
                _strict_int(entry["parameters"], label="model parameters"),
                _strict_int(
                    entry["feature_dimension_per_visit"], label="feature dimension"
                ),
                entry["checkpoint_sha256"],
                entry["feature_sha256"],
                entry["selection_reason"],
            )
            for entry in ordered
        ),
    )


def _render_producer_lineage(results: ValidatedPublicResults) -> str:
    provenance = results.reporting_run_provenance
    rows: list[tuple[Any, ...]] = []
    for stage in ("baseline_v2", "probe_v3"):
        current = provenance[stage]
        artifacts = current["artifact_sha256"]
        rows.append(
            (
                stage,
                current["protocol_version"],
                current["evaluation_lock_sha256"],
                current["run_receipt_sha256"],
                current["argv_sha256"],
                ", ".join(
                    f"{key}={artifacts[key]}" for key in sorted(artifacts)
                ),
            )
        )
    summarizer = provenance["summarizer"]
    rows.append(
        (
            "reporting_v3",
            summarizer["protocol_version"],
            summarizer["code_lock_sha256"],
            "reporting_run_provenance commit marker",
            summarizer["argv_sha256"],
            ", ".join(
                f"{key}={value}"
                for key, value in sorted(provenance["public_artifact_sha256"].items())
            ),
        )
    )
    return _markdown_table(
        ("Stage", "Version", "Lock/code SHA", "Receipt", "argv SHA", "Artifact SHA"),
        rows,
    )


def _render_git_handoff(results: ValidatedPublicResults) -> str:
    handoff = results.git_handoff
    status = str(handoff["substantive_push_status"])
    error = "无" if handoff["sanitized_push_error"] is None else str(
        handoff["sanitized_push_error"]
    )
    remote_sha = (
        "未建立"
        if handoff["substantive_remote_ref_sha"] is None
        else str(handoff["substantive_remote_ref_sha"])
    )
    return _markdown_table(
        ("Item", "Substantive content push value"),
        (
            ("Content commit", handoff["content_commit_sha"]),
            ("Branch", handoff["branch"]),
            ("Attempted remote/ref", f"{handoff['remote']} {handoff['remote_ref']}"),
            ("substantive_push_status", status),
            ("substantive_remote_ref_sha", remote_sha),
            ("Sanitized push error", error),
        ),
    )


def _model_metadata(
    model: str, spatial: str, contract: Mapping[str, Any]
) -> tuple[str, str, str, str]:
    axes = sorted(_source_axes(contract), key=lambda item: len(item[0]), reverse=True)
    source = next((name for name, axis in axes if spatial == axis and model.startswith(f"{name}_")), None)
    if source is None:
        pretraining = "不适用"
        mri = "否"
        representation_ftv = "不适用"
    else:
        pretraining = _pretraining_map(contract)[(source, spatial)]
        mri = "是"
        representation_ftv = "否"
    clinical = "是" if model.startswith("clinical_") or "_clinical" in model else "否"
    if model in {"ftv_only", "clinical_ftv"} or model.endswith(("_mri_ftv", "_mri_clinical_ftv")):
        ftv_covariate = "是"
    elif model in {"radiomics_only", "clinical_radiomics"}:
        ftv_covariate = "radiomics bundle（含 FTV）"
    else:
        ftv_covariate = "否"
    return pretraining, mri, clinical, f"{representation_ftv} / {ftv_covariate}"


def _render_pcr_table(results: ValidatedPublicResults, contract: Mapping[str, Any]) -> str:
    ordered = results.pcr.sort_values(
        ["analysis_population", "model", "spatial", "timing"], kind="stable"
    )
    rows = []
    for row in ordered.itertuples(index=False):
        pretraining, mri, clinical, ftv = _model_metadata(str(row.model), str(row.spatial), contract)
        rows.append(
            (
                row.analysis_population,
                row.model,
                pretraining,
                row.spatial,
                mri,
                clinical,
                row.timing,
                ftv,
                int(row.n),
                int(row.positive),
                _format_number(row.auroc),
                _format_number(row.auprc),
                _format_number(row.brier),
                _format_number(row.calibration_intercept),
                _format_number(row.calibration_slope),
                _format_number(row.ece_10bin),
            )
        )
    return _markdown_table(
        (
            "Population",
            "Model",
            "Pretraining",
            "Spatial",
            "MRI",
            "Clinical",
            "Timing",
            "FTV representation / readout",
            "n",
            "pCR+",
            "AUROC",
            "AUPRC",
            "Brier",
            "Calibration intercept",
            "Calibration slope",
            "ECE10",
        ),
        rows,
    )


def _render_paired_table(results: ValidatedPublicResults) -> str:
    compact = _compact_comparisons(results.paired)
    rows = []
    for row in compact.itertuples(index=False):
        def interval(metric: str) -> str:
            return (
                f"{_format_number(getattr(row, metric + '_delta'))} "
                f"[{_format_number(getattr(row, metric + '_ci_low'))}, "
                f"{_format_number(getattr(row, metric + '_ci_high'))}]"
            )

        rows.append(
            (
                row.comparison_id,
                row.family,
                row.estimand,
                row.analysis_population,
                row.timing,
                f"{row.reference_model}@{row.reference_spatial}",
                f"{row.candidate_model}@{row.candidate_spatial}",
                int(row.n_paired),
                interval("auroc"),
                interval("auprc"),
                interval("brier"),
            )
        )
    return _markdown_table(
        (
            "Comparison ID",
            "Family",
            "Estimand",
            "Population",
            "Timing",
            "Reference",
            "Candidate",
            "n",
            "ΔAUROC [95% CI]",
            "ΔAUPRC [95% CI]",
            "ΔBrier [95% CI]",
        ),
        rows,
    )


def _render_phenotype_table(results: ValidatedPublicResults) -> str:
    ordered = results.phenotype.sort_values(
        ["target", "model", "spatial", "timing"], kind="stable"
    )
    return _markdown_table(
        (
            "Target",
            "Model",
            "Spatial",
            "Timing",
            "n",
            "Positive",
            "AUROC",
            "AUPRC",
            "Brier",
            "Calibration intercept",
            "Calibration slope",
            "ECE10",
        ),
        (
            (
                row.target,
                row.model,
                row.spatial,
                row.timing,
                int(row.n),
                int(row.positive),
                _format_number(row.auroc),
                _format_number(row.auprc),
                _format_number(row.brier),
                _format_number(row.calibration_intercept),
                _format_number(row.calibration_slope),
                _format_number(row.ece_10bin),
            )
            for row in ordered.itertuples(index=False)
        ),
    )


def _render_subtype_table(results: ValidatedPublicResults) -> str:
    ordered = results.subtype.sort_values(["model", "spatial", "timing"], kind="stable")
    return _markdown_table(
        (
            "Model",
            "Spatial",
            "Timing",
            "n",
            "Macro OVR AUROC",
            "Macro OVR AUPRC",
            "Multiclass Brier",
            "Top-label ECE10",
            "Accuracy",
        ),
        (
            (
                row.model,
                row.spatial,
                row.timing,
                int(row.n),
                _format_number(row.macro_ovr_auroc),
                _format_number(row.macro_ovr_auprc),
                _format_number(row.multiclass_brier),
                _format_number(row.toplabel_ece_10bin),
                _format_number(row.accuracy),
            )
            for row in ordered.itertuples(index=False)
        ),
    )


def _render_ftv_table(results: ValidatedPublicResults) -> str:
    ordered = results.ftv.sort_values(
        ["model", "spatial", "task", "endpoint"], kind="stable"
    )
    return _markdown_table(
        (
            "Model",
            "Spatial",
            "Task",
            "Endpoint",
            "n",
            "Spearman",
            "Pearson",
            "R²",
            "RMSE",
            "MAE",
            "RMSE gain over b0",
            "Calibration slope",
            "Calibration intercept",
            "Mean bias",
        ),
        (
            (
                row.model,
                row.spatial,
                row.task,
                row.endpoint,
                int(row.n),
                _format_number(row.spearman),
                _format_number(row.pearson),
                _format_number(row.r2),
                _format_number(row.rmse),
                _format_number(row.mae),
                _format_number(row.rmse_gain_over_b0),
                _format_number(row.calibration_slope),
                _format_number(row.calibration_intercept),
                _format_number(row.calibration_mean_bias),
            )
            for row in ordered.itertuples(index=False)
        ),
    )


def _classification_cn(value: str) -> str:
    return {
        "consistent_support": "全部预定 cells 方向一致支持",
        "consistent_adverse": "全部预定 cells 方向一致不利",
        "mixed": "证据混合，不能确认",
        "beyond_tumor_size_descriptive_support": "描述性支持存在 FTV 以外增量",
        "beyond_tumor_size_not_established": "未建立 FTV 以外增量",
        "prioritize_integration_and_external_validation": "优先开展 foundation integration 与外部验证",
        "controlled_integration_study_only": "仅建议受控 integration study，不支持直接替换",
        "do_not_prioritize_replacement_from_this_experiment": "本实验不足以支持优先替换 encoder",
    }.get(value, value)


def _evidence_sentence(evidence: Mapping[str, Any]) -> str:
    counts = evidence["point_direction_counts"]
    return (
        f"{_classification_cn(str(evidence['set_classification']))}；"
        f"预定 {evidence['expected_comparisons']} cells 中 favorable/mixed/adverse="
        f"{counts['favorable']}/{counts['mixed']}/{counts['adverse']}。"
    )


def _render_question_matrix(
    results: ValidatedPublicResults, contract: Mapping[str, Any]
) -> str:
    evidence = results.question_evidence
    q10 = _classification_cn(str(evidence["Q10"]["set_classification"]))
    q12 = _classification_cn(str(evidence["Q12"]["recommendation"]))
    rows = (
        (1, "使用了哪些 foundation models？", "MedicalNet 3DSeg-8 ResNet-50 与 DINO v1 ViT-B/16；完整 checkpoint/feature/source/license 见模型 provenance 表。"),
        (2, "为什么选择它们？", "分别提供可审计的 3-D medical reference 与固定 ImageNet-1K 的公开通用 SSL reference；选择理由、参数量与维度完整列出。"),
        (3, "MRI-only AUROC/AUPRC 是多少？", "完整 808 人的 18 个 MRI-only cells（其中 foundation 12 个）全部列于 pCR 长表。"),
        (4, "LOCAL vs GLOBAL 如何？", _evidence_sentence(evidence["Q4"])),
        (5, "Foundation vs current CNN LOCAL 如何？", _evidence_sentence(evidence["Q5"])),
        (6, "HR/HER2 decodability 如何？", "全部 12 个 HR/HER2 与 6 个 subtype pooled cells 均列出；不使用事后阈值筛选。"),
        (7, "FTV decodability 如何？", "全部 42 个 FTV/ΔFTV pooled cells 均列出，其中 foundation 28 个。"),
        (8, "Clinical+Foundation 是否超过 Clinical-only？", _evidence_sentence(evidence["Q8"])),
        (9, "Clinical+FTV+Foundation 是否超过 Clinical+FTV？", _evidence_sentence(evidence["Q9"])),
        (10, "是否学到 tumor-size 以外信息？", q10),
        (
            11,
            "当前 World Model 是否明显 underuse MRI？",
            "仅报告点估计方向：" + _evidence_sentence(evidence["Q11"])
            + "该描述不以 0 阈值证明‘明显’ underuse。",
        ),
        (12, "下一步是否值得替换/增强 encoder？", q12),
    )
    return _markdown_table(("#", "问题", "固定规则答案"), rows)


def _render_fixed_conclusions(results: ValidatedPublicResults) -> str:
    evidence = results.question_evidence
    lines = []
    for question in ("Q4", "Q5", "Q8", "Q9", "Q11"):
        lines.append(f"- **{question}**：{_evidence_sentence(evidence[question])}")
        for metric in _METRICS:
            counts = evidence[question]["ci_direction_counts"][metric]
            lines.append(
                f"  - {metric.upper()} descriptive CI favorable/inconclusive/adverse="
                f"{counts['favorable']}/{counts['inconclusive']}/{counts['adverse']}。"
            )
    lines.append(
        f"- **Q10**：{_classification_cn(str(evidence['Q10']['set_classification']))}。"
    )
    lines.append(
        f"- **Q12**：{_classification_cn(str(evidence['Q12']['recommendation']))}。"
    )
    lines.append("- 所有区间均为描述性，不作显著性检验、等效性判断或 test-driven selection。")
    return "\n".join(lines)


def _coverage_counts(results: ValidatedPublicResults) -> Mapping[str, int]:
    specs = results.paired.drop_duplicates("comparison_id")
    counts = {
        "pcr_pooled_total": int(len(results.pcr)),
        "pcr_pooled_full": int(
            results.pcr["analysis_population"].astype(str).eq("full_808").sum()
        ),
        "pcr_pooled_complete_case": int(
            results.pcr["analysis_population"].astype(str).eq("radiomics_complete_case_375").sum()
        ),
        "paired_comparisons": int(len(specs)),
        "paired_metric_rows": int(len(results.paired)),
        "phenotype_pooled": int(len(results.phenotype)),
        "subtype_pooled": int(len(results.subtype)),
        "ftv_pooled": int(len(results.ftv)),
    }
    counts.update({key: int(value) for key, value in results.public_row_counts.items()})
    return counts


def _render_coverage_manifest(
    results: ValidatedPublicResults, contract: Mapping[str, Any]
) -> str:
    observed = _coverage_counts(results)
    expected = contract["expected_counts"]
    rows = (
        ("pCR pooled full", expected["pcr_pooled_full"], observed["pcr_pooled_full"]),
        (
            "pCR pooled complete case",
            expected["pcr_pooled_complete_case"],
            observed["pcr_pooled_complete_case"],
        ),
        ("pCR pooled total", expected["pcr_pooled_total"], observed["pcr_pooled_total"]),
        ("Paired comparisons", expected["paired_comparisons"], observed["paired_comparisons"]),
        ("Paired metric rows", expected["paired_metric_rows"], observed["paired_metric_rows"]),
        ("Phenotype pooled", expected["phenotype_pooled"], observed["phenotype_pooled"]),
        ("Subtype pooled", expected["subtype_pooled"], observed["subtype_pooled"]),
        ("FTV pooled", expected["ftv_pooled"], observed["ftv_pooled"]),
        (
            "Baseline public pooled+macro rows",
            expected["baseline_public_rows"],
            observed["baseline_public_rows"],
        ),
        (
            "Phenotype public pooled+macro rows",
            expected["phenotype_public_rows"],
            observed["phenotype_public_rows"],
        ),
        (
            "Subtype public pooled+macro rows",
            expected["subtype_public_rows"],
            observed["subtype_public_rows"],
        ),
        (
            "FTV public pooled+macro rows",
            expected["ftv_public_rows"],
            observed["ftv_public_rows"],
        ),
    )
    return _markdown_table(
        ("Artifact/cell family", "Expected", "Observed", "Status"),
        ((label, expected_count, observed_count, "PASS" if expected_count == observed_count else "FAIL") for label, expected_count, observed_count in rows),
    )


def _render_provenance(
    hashes: Mapping[str, str], contract: Mapping[str, Any]
) -> str:
    report_directory = Path("additional_experiments/foundation_mri_baselines/reports")
    fixed_paths = contract["fixed_public_paths"]

    def link(role: str) -> str:
        relative = Path(os.path.relpath(str(fixed_paths[role]), start=str(report_directory)))
        return f"[{role}]({relative.as_posix()})"

    return _markdown_table(
        ("Public input", "Relative link", "SHA-256"),
        (
            (key, link(key), hashes[key])
            for key in sorted(hashes)
            if key in fixed_paths
        ),
    )


def render_final_report(
    template_text: str,
    results: ValidatedPublicResults,
    contract: Mapping[str, Any],
    hashes: Mapping[str, str],
) -> str:
    observed_tokens = re.findall(r"\{\{([A-Z0-9_]+)\}\}", template_text)
    if set(observed_tokens) != set(_TEMPLATE_TOKENS) or any(
        observed_tokens.count(token) != 1 for token in _TEMPLATE_TOKENS
    ):
        raise ValueError(
            "final-report template tokens drifted; every frozen token must appear exactly once"
        )
    replacements = {
        "COVERAGE_MANIFEST": _render_coverage_manifest(results, contract),
        "MODEL_PROVENANCE": _render_model_provenance(contract),
        "QUESTION_MATRIX": _render_question_matrix(results, contract),
        "FIXED_CONCLUSIONS": _render_fixed_conclusions(results),
        "PCR_TABLE": _render_pcr_table(results, contract),
        "PAIRED_TABLE": _render_paired_table(results),
        "PHENOTYPE_TABLE": _render_phenotype_table(results),
        "SUBTYPE_TABLE": _render_subtype_table(results),
        "FTV_TABLE": _render_ftv_table(results),
        "PRODUCER_LINEAGE": _render_producer_lineage(results),
        "PUBLIC_PROVENANCE": _render_provenance(hashes, contract),
        "GIT_HANDOFF": _render_git_handoff(results),
    }
    rendered = template_text
    for token in _TEMPLATE_TOKENS:
        rendered = rendered.replace("{{" + token + "}}", replacements[token])
    if re.search(r"\{\{[A-Z0-9_]+\}\}", rendered):
        raise AssertionError("unresolved final-report template token")
    if "FORMAL_RESULT_PENDING" in rendered:
        raise ValueError("formal report still contains FORMAL_RESULT_PENDING")
    return rendered.rstrip() + "\n"


def _coverage_receipt(
    results: ValidatedPublicResults,
    contract: Mapping[str, Any],
    hashes: Mapping[str, str],
    report_text: str,
) -> Mapping[str, Any]:
    comparison_spec_frame = pd.DataFrame(results.comparison_specs)
    observed_counts = dict(_coverage_counts(results))
    rendered_counts = {
        "pcr_pooled_total": observed_counts["pcr_pooled_total"],
        "paired_comparisons": observed_counts["paired_comparisons"],
        "phenotype_pooled": observed_counts["phenotype_pooled"],
        "subtype_pooled": observed_counts["subtype_pooled"],
        "ftv_pooled": observed_counts["ftv_pooled"],
    }
    return {
        "schema_version": COVERAGE_SCHEMA_VERSION,
        "contract_id": str(contract["contract_id"]),
        "candidate_policy": str(contract["candidate_policy"]),
        "foundation_models": list(_foundations(contract)),
        "source_axes": [
            {"model": model, "spatial": spatial} for model, spatial in _source_axes(contract)
        ],
        "observed_counts": observed_counts,
        "rendered_counts": rendered_counts,
        "expected_counts": {
            key: int(value) for key, value in contract["expected_counts"].items()
        },
        "identity_sha256": {
            "pcr_pooled": _records_sha256(results.pcr, _BINARY_IDENTITY),
            "phenotype_pooled": _records_sha256(results.phenotype, _BINARY_IDENTITY),
            "subtype_pooled": _records_sha256(results.subtype, _BINARY_IDENTITY),
            "ftv_pooled": _records_sha256(results.ftv, _FTV_IDENTITY),
            "comparison_specs": _records_sha256(
                comparison_spec_frame, _COMPARISON_SPEC_COLUMNS
            ),
            "paired_metric_rows": _records_sha256(
                results.paired, (*_COMPARISON_SPEC_COLUMNS, "metric")
            ),
        },
        "input_sha256": dict(sorted(hashes.items())),
        "final_report_sha256": hashlib.sha256(report_text.encode("utf-8")).hexdigest(),
        "question_evidence": results.question_evidence,
        "reporting_run_provenance": results.reporting_run_provenance,
        "git_handoff": {
            key: results.git_handoff[key]
            for key in (
                "schema_version",
                "content_commit_sha",
                "branch",
                "remote",
                "remote_ref",
                "substantive_push_status",
                "substantive_remote_ref_sha",
                "sanitized_push_error",
            )
        },
        "publication_commit_marker": "coverage_receipt",
        "cross_directory_sigkill_transactional": False,
        "metric_sorting": False,
        "best_model_filtering": False,
        "private_inputs_read": False,
    }


def finalize_report(
    inputs: FinalizationInputs,
    outputs: FinalizationOutputs,
    *,
    git_runner: GitRunner | None = None,
) -> Mapping[str, Any]:
    destinations = (Path(outputs.final_report), Path(outputs.coverage_receipt))
    if len(set(destinations)) != len(destinations):
        raise ValueError("final report and coverage receipt paths must be distinct")
    existing = [destination for destination in destinations if os.path.lexists(destination)]
    if len(existing) == 1:
        raise FileExistsError(
            "orphaned finalization output detected; coverage receipt is the commit marker "
            f"and operator audit is required: {existing[0].name}"
        )
    if len(existing) == 2:
        raise FileExistsError("formal finalization forbids overwrite: both outputs exist")
    for label, destination in zip(("final report", "coverage receipt"), destinations, strict=True):
        _reject_private_path(destination, label=label)
    runner = _default_git_runner if git_runner is None else git_runner
    snapshots, contract, handoff, provenance = _prepare_snapshots(
        inputs, git_runner=runner
    )
    results, hashes = _validate_from_snapshots(
        snapshots, contract, handoff, provenance
    )
    try:
        template_text = snapshots.payloads["template_markdown"].decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("final-report template is not UTF-8") from error
    report_text = render_final_report(template_text, results, contract, hashes)
    receipt = _coverage_receipt(results, contract, hashes, report_text)
    common_parent = Path(
        os.path.commonpath([str(destination.parent.resolve()) for destination in destinations])
    )
    common_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".foundation-finalization-", dir=common_parent) as temp:
        staging = Path(temp)
        staged_report = staging / "final_report.md"
        staged_receipt = staging / "coverage.json"
        staged_report.write_text(report_text, encoding="utf-8")
        staged_receipt.write_text(_json_text(receipt), encoding="utf-8")
        published: list[Path] = []
        try:
            for source, destination in zip(
                (staged_report, staged_receipt), destinations, strict=True
            ):
                destination.parent.mkdir(parents=True, exist_ok=True)
                # A hard-link publish is same-filesystem and fails if a race creates
                # the destination after the initial no-overwrite check.
                os.link(source, destination)
                published.append(destination)
                os.chmod(destination, 0o644)
        except Exception:
            for destination in reversed(published):
                try:
                    destination.unlink()
                except FileNotFoundError:
                    pass
            raise
    return {
        "pcr_pooled": len(results.pcr),
        "paired_comparisons": len(results.comparison_specs),
        "paired_metric_rows": len(results.paired),
        "phenotype_pooled": len(results.phenotype),
        "subtype_pooled": len(results.subtype),
        "ftv_pooled": len(results.ftv),
        "public_outputs": 2,
    }


__all__ = [
    "COVERAGE_SCHEMA_VERSION",
    "FinalizationInputs",
    "FinalizationOutputs",
    "ValidatedPublicResults",
    "file_sha256",
    "finalize_report",
    "fixed_question_evidence",
    "load_contract",
    "render_final_report",
    "validate_public_results",
]

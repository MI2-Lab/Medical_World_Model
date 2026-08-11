from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foundation_mri.locking import (  # noqa: E402
    argument_vector_sha256,
    canonical_json_sha256,
    file_sha256,
    verify_evaluation_code_lock,
    verify_formal_evaluation_lock,
    verify_historical_metric_free_run_provenance,
    verify_metric_free_run_provenance,
    write_metric_free_run_provenance,
)


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _record(root: Path, path: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(root)), "sha256": file_sha256(path)}


def _synthetic_lock(
    root: Path,
    *,
    models: tuple[str, str] = ("medical", "dino"),
) -> tuple[Path, list[Path], dict, tuple[Path, Path, Path]]:
    code = _write(root / "src/evaluation.py", b"locked-code\n")
    fold = _write(root / "inputs/folds.csv", b"folds\n")
    clinical = _write(root / "inputs/clinical.csv", b"clinical\n")
    radiomics = _write(root / "inputs/radiomics.csv", b"radiomics\n")
    foundation = [
        _write(root / "features/medical.private.npz", b"medical"),
        _write(root / "features/dino.private.npz", b"dino"),
    ]
    cnn: dict[tuple[str, str], dict[int, Path]] = {
        ("GAP0", "GLOBAL"): {},
        ("LOCAL0", "LOCAL"): {},
    }
    cnn_records = []
    for (model, spatial), paths in cnn.items():
        for fold_index in range(5):
            feature = _write(
                root / f"cnn/{model}/fold_{fold_index}/response_state.private.npz",
                f"{model}-{fold_index}".encode(),
            )
            metadata = _write(
                feature.with_suffix(".metadata.json"),
                f"metadata-{model}-{fold_index}".encode(),
            )
            paths[fold_index] = feature
            cnn_records.append(
                {
                    "model": model,
                    "spatial": spatial,
                    "fold": fold_index,
                    "feature": _record(root, feature),
                    "metadata": _record(root, metadata),
                }
            )
    payload = {
        "schema_version": 1,
        "lock_kind": "formal_outcome_evaluation",
        "locked_at": "2026-08-11",
        "locked_before_formal_test_evaluation": True,
        "formal_test_outcomes_seen": False,
        "files": {"src/evaluation.py": file_sha256(code)},
        "inputs": {
            "fold_manifest": _record(root, fold),
            "clinical_labels": _record(root, clinical),
            "radiomics": _record(root, radiomics),
            "foundation_features": [
                {"model": model, **_record(root, path)}
                for model, path in zip(models, foundation, strict=True)
            ],
            "current_cnn_features": cnn_records,
        },
        "execution": {"selection_workers": 2},
    }
    lock = root / "configs/EVALUATION_LOCK.json"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps(payload), encoding="utf-8")
    return lock, foundation, cnn, (fold, clinical, radiomics)


def _synthetic_v2_chain(
    root: Path,
    *,
    models: tuple[str, str] = ("medical", "dino"),
) -> tuple[
    Path,
    list[Path],
    dict[tuple[str, str], dict[int, Path]],
    tuple[Path, Path, Path],
    dict[str, tuple[str, ...]],
    dict[str, Path],
]:
    parent, foundation, cnn, tabular = _synthetic_lock(root, models=models)
    parent_payload = json.loads(parent.read_text(encoding="utf-8"))

    # The final v2 tree may differ from the historical v1 code.  The chain binds
    # the parent lock bytes; it must not pretend those bytes describe v2 code.
    final_code = _write(root / "src/evaluation.py", b"final-v2-code\n")
    source_snapshot = _write(root / "provenance/v1-source.snapshot", b"source snapshot")
    patch_manifest = _write(root / "provenance/v1-to-v2.patch", b"synthetic patch")
    smoke_receipt = _write(root / "provenance/v2-smoke.json", b'{"passed":true}\n')
    argv = {
        "baseline": (
            "--evaluation-lock",
            "configs/EVALUATION_LOCK.v2.json",
            "--baseline",
        ),
        "probe": ("--evaluation-lock", "configs/EVALUATION_LOCK.v2.json", "--probe"),
        "summarizer": (
            "--evaluation-lock",
            "configs/EVALUATION_LOCK.v2.json",
            "--summarize",
        ),
    }
    payload = {
        "schema_version": 2,
        "lock_kind": "formal_outcome_evaluation_chain",
        "lock_generation": "whole_evaluation_v2",
        "locked_at": "2026-08-11T12:00:00-04:00",
        "locked_before_formal_test_evaluation": True,
        "formal_test_outcomes_seen": False,
        "parent_lock": _record(root, parent),
        "aborted_v1_attempts": [
            {
                "consumer": "baseline",
                "exit_code": 143,
                "outputs_written": False,
                "metric_values_viewed": False,
            },
            {
                "consumer": "probe",
                "exit_code": 143,
                "outputs_written": False,
                "metric_values_viewed": False,
            },
        ],
        "prelock_state": {
            "baseline_outputs_exist": False,
            "probe_outputs_exist": False,
            "metric_values_viewed": False,
        },
        "inherited_inputs": {
            "source": "parent_lock.inputs",
            "sha256": canonical_json_sha256(parent_payload["inputs"]),
        },
        "development_provenance": {
            "source_snapshot": _record(root, source_snapshot),
            "patch_manifest": _record(root, patch_manifest),
            "smoke_receipt": _record(root, smoke_receipt),
        },
        "files": {"src/evaluation.py": file_sha256(final_code)},
        "execution": {
            "parent_execution_sha256": canonical_json_sha256(
                parent_payload["execution"]
            ),
            "consumer_argv": {
                consumer: {
                    "argc": len(arguments),
                    "sha256": argument_vector_sha256(arguments),
                }
                for consumer, arguments in argv.items()
            },
        },
    }
    lock = root / "configs/EVALUATION_LOCK.v2.json"
    lock.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    provenance = {
        "source_snapshot": source_snapshot,
        "patch_manifest": patch_manifest,
        "smoke_receipt": smoke_receipt,
        "final_code": final_code,
        "parent_lock": parent,
    }
    return lock, foundation, cnn, tabular, argv, provenance


def _synthetic_v3_chain(
    root: Path,
) -> tuple[
    Path,
    list[Path],
    dict[tuple[str, str], dict[int, Path]],
    tuple[Path, Path, Path],
    dict[str, tuple[str, ...]],
    dict[str, Path],
]:
    formal_models = (
        "medicalnet_resnet50_3dseg8",
        "dino_vitb16_imagenet1k",
    )
    v2_lock, foundation, cnn, tabular, v2_argv, v2_provenance = (
        _synthetic_v2_chain(root, models=formal_models)
    )
    v2_payload = json.loads(v2_lock.read_text(encoding="utf-8"))
    v2_code_sha = v2_payload["files"]["src/evaluation.py"]
    final_code = _write(root / "src/evaluation.py", b"final-v3-code\n")
    patch = _write(
        root / "configs/EVALUATION_V2_TO_V3.patch",
        b"--- a/src/evaluation.py\n+++ b/src/evaluation.py\n@@ synthetic\n",
    )
    snapshot_payload = {
        "schema_version": 1,
        "snapshot_kind": "reconstructable_evaluation_v2_to_v3_source_transition",
        "v2_lock": {
            "path": str(v2_lock.relative_to(root)),
            "sha256": file_sha256(v2_lock),
            "locked_file_count": len(v2_payload["files"]),
        },
        "forward_patch": {
            "path": str(patch.relative_to(root)),
            "sha256": file_sha256(patch),
            "format": "unified_diff_a_to_b_strip_1",
            "line_count": len(patch.read_bytes().splitlines()),
            "byte_count": patch.stat().st_size,
        },
        "changed_v2_locked_files": {
            "src/evaluation.py": {
                "v2_sha256": v2_code_sha,
                "v3_sha256": file_sha256(final_code),
            }
        },
        "reconstruction": {
            "v2_from_v3_command": "patch -R -p1 < configs/EVALUATION_V2_TO_V3.patch",
            "v3_from_v2_command": "patch -p1 < configs/EVALUATION_V2_TO_V3.patch",
            "verify_v2_command": "sha256sum -c v2.manifest",
            "forward_patch_verified": True,
            "reverse_patch_verified": True,
        },
    }
    snapshot = _write(
        root / "configs/EVALUATION_V2_SOURCE_SNAPSHOT.json",
        (json.dumps(snapshot_payload, sort_keys=True) + "\n").encode(),
    )
    smoke = _write(root / "reports/probe_v3_metric_free_smoke.json", b'{"status":"PASS"}\n')
    finalization = _write(
        root / "configs/FINALIZATION_LOCK.v1.json",
        (ROOT / "configs/FINALIZATION_LOCK.v1.json").read_bytes(),
    )

    progress_body = b"x\n" * 2658 + b"x" * (447729 - 2658 * 2 - 1) + b"\n"
    assert len(progress_body) == 447729
    failed_progress = _write(root / "logs/probe_v2.progress.private.jsonl", progress_body)
    os.chmod(failed_progress, 0o600)
    artifacts = {
        "predictions": _write(root / "predictions/baseline.private.csv", b"predictions\n"),
        "selection": _write(root / "metrics/baseline_selection.private.csv", b"selection\n"),
        "metrics": _write(root / "metrics/baseline.csv", b"metrics\n"),
        "progress": _write(root / "logs/baseline.private.jsonl", b"progress\n"),
    }
    receipt_artifact_keys = {
        "predictions": "baseline_predictions_private",
        "selection": "baseline_selection_private",
        "metrics": "baseline_metrics_public",
        "progress": "baseline_progress_private",
    }
    receipt_payload = {
        "schema_version": 1,
        "receipt_kind": "formal_metric_free_run_provenance",
        "consumer": "baseline",
        "lock_sha256": file_sha256(v2_lock),
        "parent_lock_sha256": file_sha256(v2_provenance["parent_lock"]),
        "argument_count": len(v2_argv["baseline"]),
        "argument_vector_sha256": argument_vector_sha256(v2_argv["baseline"]),
        "artifact_hash_method": "sha256_binary_stream_no_parse",
        "metric_values_viewed": False,
        "artifacts": {
            receipt_artifact_keys[role]: _record(root, path)
            for role, path in artifacts.items()
        },
    }
    baseline_receipt = _write(
        root / "metrics/baseline_v2_run.private.provenance.json",
        (json.dumps(receipt_payload, sort_keys=True) + "\n").encode(),
    )
    argv = {
        "probe": ("--evaluation-lock", "configs/EVALUATION_LOCK.v3.json", "--probe"),
        "summarizer": (),
    }
    payload = {
        "schema_version": 3,
        "lock_kind": "formal_outcome_evaluation_chain",
        "lock_generation": "probe_retry_v3",
        "locked_at": "2026-08-11T08:00:00-04:00",
        "locked_after_baseline_v2_completion": True,
        "baseline_v2_metric_prediction_values_seen": False,
        "locked_before_formal_probe_v3_evaluation": True,
        "formal_probe_v3_outcomes_seen": False,
        "parent_lock": _record(root, v2_lock),
        "finalization_lock": _record(root, finalization),
        "aborted_v2_attempts": [
            {
                "consumer": "probe",
                "producer_lock_sha256": file_sha256(v2_lock),
                "exit_code": 1,
                "outputs_written": False,
                "metric_values_viewed": False,
                "original_traceback_recovered": False,
                "formal_exception_claimed": False,
                "failure_window": "ridge_fold3_after_selection_before_fold4_start",
                "synthetic_diagnostic_scope": (
                    "matching_failure_path_not_recovered_formal_exception"
                ),
                "progress_log": {
                    **_record(root, failed_progress),
                    "mode_octal": "0600",
                    "byte_count": 447729,
                    "line_count": 2659,
                    "ends_with_newline": True,
                    "event_counts": {
                        "formal_run_started": 1,
                        "fold_started": 69,
                        "fold_completed": 69,
                        "candidate_started": 540,
                        "candidate_completed": 540,
                        "estimator_started": 720,
                        "estimator_completed": 720,
                    },
                    "last_event": "fold_completed",
                },
            }
        ],
        "prelock_state": {
            "baseline_v2_completed": True,
            "baseline_v2_receipt_bound": True,
            "probe_v3_outputs_exist": False,
            "metric_values_viewed": False,
        },
        "inherited_inputs": {
            "source": "parent_lock.inherited_inputs",
            "sha256": v2_payload["inherited_inputs"]["sha256"],
        },
        "historical_producers": {
            "baseline_v2": {
                "consumer": "baseline",
                "producer_lock": _record(root, v2_lock),
                "run_receipt": _record(root, baseline_receipt),
                "argv": v2_payload["execution"]["consumer_argv"]["baseline"],
                "artifacts": {
                    role: _record(root, path) for role, path in artifacts.items()
                },
                "audit_counts": {
                    "prediction_rows": 64137,
                    "selection_rows": 630,
                    "public_metric_rows": 252,
                    "progress_fold_completed": 630,
                    "progress_candidate_completed": 11340,
                    "max_observed_n_iter": 1485,
                    "capped_candidate_count": 0,
                },
            }
        },
        "development_provenance": {
            "source_snapshot": _record(root, snapshot),
            "patch_manifest": _record(root, patch),
            "smoke_receipt": _record(root, smoke),
        },
        "files": {"src/evaluation.py": file_sha256(final_code)},
        "execution": {
            "parent_execution_sha256": canonical_json_sha256(v2_payload["execution"]),
            "consumer_argv": {
                consumer: {
                    "argc": len(arguments),
                    "sha256": argument_vector_sha256(arguments),
                }
                for consumer, arguments in argv.items()
            },
            "producer_order": ["baseline:v2", "probe:v3", "summarizer:v3"],
        },
    }
    lock = _write(
        root / "configs/EVALUATION_LOCK.v3.json",
        (json.dumps(payload, sort_keys=True) + "\n").encode(),
    )
    provenance = {
        "final_code": final_code,
        "baseline_receipt": baseline_receipt,
        "failed_progress": failed_progress,
        **artifacts,
    }
    return lock, foundation, cnn, tabular, argv, provenance


def _synthetic_reporting_lock(
    root: Path,
) -> tuple[Path, dict[str, Path]]:
    v3_lock, _, _, _, v3_argv, v3_provenance = _synthetic_v3_chain(root)
    v3_payload = json.loads(v3_lock.read_text(encoding="utf-8"))
    v3_code_sha = v3_payload["files"]["src/evaluation.py"]
    final_code = _write(root / "src/evaluation.py", b"reporting-retry-code\n")
    patch = _write(
        root / "configs/REPORTING_V3_TO_R1.patch",
        b"--- a/src/evaluation.py\n+++ b/src/evaluation.py\n@@ synthetic-reporting\n",
    )
    snapshot_payload = {
        "schema_version": 1,
        "snapshot_kind": (
            "reconstructable_evaluation_v3_to_reporting_v1_source_transition"
        ),
        "parent_lock": {
            "path": str(v3_lock.relative_to(root)),
            "sha256": file_sha256(v3_lock),
            "locked_file_count": len(v3_payload["files"]),
        },
        "forward_patch": {
            "path": str(patch.relative_to(root)),
            "sha256": file_sha256(patch),
            "format": "unified_diff_a_to_b_strip_1",
            "line_count": len(patch.read_bytes().splitlines()),
            "byte_count": patch.stat().st_size,
        },
        "changed_parent_locked_files": {
            "src/evaluation.py": {
                "v3_sha256": v3_code_sha,
                "reporting_v1_sha256": file_sha256(final_code),
            }
        },
        "reconstruction": {
            "v3_from_reporting_command": "patch -R -p1 < configs/REPORTING_V3_TO_R1.patch",
            "reporting_from_v3_command": "patch -p1 < configs/REPORTING_V3_TO_R1.patch",
            "verify_v3_command": "sha256sum -c v3.manifest",
            "forward_patch_verified": True,
            "reverse_patch_verified": True,
        },
    }
    snapshot = _write(
        root / "configs/REPORTING_V3_SOURCE_SNAPSHOT.json",
        (json.dumps(snapshot_payload, sort_keys=True) + "\n").encode(),
    )
    synthetic = _write(
        root / "reports/reporting_v1_numeric_consistency_smoke.json",
        b'{"status":"PASS"}\n',
    )
    role_to_receipt = {
        "phenotype_predictions": "phenotype_predictions_private",
        "phenotype_selection": "phenotype_selection_private",
        "phenotype_metrics": "phenotype_metrics_public",
        "subtype_predictions": "subtype_predictions_private",
        "subtype_selection": "subtype_selection_private",
        "subtype_metrics": "subtype_metrics_public",
        "ftv_predictions": "ftv_predictions_private",
        "ftv_selection": "ftv_selection_private",
        "ftv_metrics": "ftv_metrics_public",
        "progress": "probe_progress_private",
    }
    probe_artifacts = {
        role: _write(root / f"formal_probe/{role}.bin", f"{role}\n".encode())
        for role in role_to_receipt
    }
    v2_lock = root / v3_payload["parent_lock"]["path"]
    probe_receipt_payload = {
        "schema_version": 1,
        "receipt_kind": "formal_metric_free_run_provenance",
        "consumer": "probe",
        "lock_sha256": file_sha256(v3_lock),
        "parent_lock_sha256": file_sha256(v2_lock),
        "argument_count": len(v3_argv["probe"]),
        "argument_vector_sha256": argument_vector_sha256(v3_argv["probe"]),
        "artifact_hash_method": "sha256_binary_stream_no_parse",
        "metric_values_viewed": False,
        "artifacts": {
            receipt_key: _record(root, probe_artifacts[role])
            for role, receipt_key in role_to_receipt.items()
        },
    }
    probe_receipt = _write(
        root / "metrics/probe_v3_run.private.provenance.json",
        (json.dumps(probe_receipt_payload, sort_keys=True) + "\n").encode(),
    )
    payload = {
        "schema_version": "foundation_mri_reporting_lock_v1",
        "lock_kind": "formal_reporting_retry",
        "lock_generation": "calibration_identity_recompute_tolerance_v1",
        "locked_at": "2026-08-11T09:00:00-04:00",
        "locked_after_probe_v3_completion": True,
        "locked_before_formal_summarizer_retry": True,
        "parent_lock": _record(root, v3_lock),
        "finalization_lock": v3_payload["finalization_lock"],
        "visibility": {
            "calibration_intercept_values_seen": True,
            "observed_identity": {
                "source": "baseline_v2_public_vs_serialized_private_recompute",
                "target": "pCR",
                "model": "dino_vitb16_imagenet1k_mri_clinical_ftv",
                "spatial": "GLOBAL",
                "timing": "T0-T2",
                "analysis_population": "radiomics_complete_case_375",
                "aggregation": "outer_fold_macro",
                "affected_cell_count": 1,
                "column": "calibration_intercept",
                "public_value": "-0.2992681677870132",
                "recomputed_value": "-0.29926816783893295",
                "absolute_difference": "5.191974628004914e-11",
            },
            "auroc_auprc_brier_ece_prediction_or_selection_performance_values_seen": False,
            "revision_used_model_direction_or_performance": False,
        },
        "failed_summarizer_attempts": [
            {
                "consumer": "summarizer",
                "producer_lock_sha256": file_sha256(v3_lock),
                "argument_count": 0,
                "argument_vector_sha256": argument_vector_sha256(()),
                "exit_code": 1,
                "expected_public_output_count": 5,
                "public_outputs_written_count": 0,
                "reporting_marker_written": False,
                "exception_type": "ValueError",
                "exception_message": (
                    "baseline public aggregate drifted in numeric column "
                    "calibration_intercept"
                ),
                "observed_values_limited_to_visibility_record": True,
            }
        ],
        "prelock_state": {
            "baseline_v2_completed": True,
            "probe_v3_completed": True,
            "historical_receipts_bound": True,
            "retry_public_outputs_exist": False,
            "reporting_marker_exists": False,
        },
        "inherited_inputs": {
            "source": "parent_lock.inherited_inputs",
            "sha256": v3_payload["inherited_inputs"]["sha256"],
        },
        "historical_producers": {
            "baseline_v2": v3_payload["historical_producers"]["baseline_v2"],
            "probe_v3": {
                "consumer": "probe",
                "producer_lock": _record(root, v3_lock),
                "run_receipt": _record(root, probe_receipt),
                "argv": v3_payload["execution"]["consumer_argv"]["probe"],
                "artifacts": {
                    role: _record(root, path)
                    for role, path in probe_artifacts.items()
                },
                "audit_counts": {
                    "phenotype_prediction_rows": 9696,
                    "subtype_prediction_rows": 4848,
                    "ftv_prediction_rows": 15750,
                    "phenotype_selection_rows": 60,
                    "subtype_selection_rows": 30,
                    "ftv_selection_rows": 210,
                    "phenotype_public_rows": 24,
                    "subtype_public_rows": 12,
                    "ftv_public_rows": 84,
                    "progress_fold_completed": 300,
                    "progress_candidate_completed": 3300,
                    "progress_ovr_estimator_completed": 2160,
                    "max_binary_n_iter": 677,
                    "max_subtype_n_iter": 197,
                    "max_ridge_n_iter": 805,
                    "capped_candidate_count": 0,
                },
            },
        },
        "development_provenance": {
            "source_snapshot": _record(root, snapshot),
            "patch_manifest": _record(root, patch),
            "synthetic_receipt": _record(root, synthetic),
        },
        "files": {"src/evaluation.py": file_sha256(final_code)},
        "execution": {
            "parent_execution_sha256": canonical_json_sha256(v3_payload["execution"]),
            "consumer_argv": {
                "summarizer": {
                    "argc": 0,
                    "sha256": argument_vector_sha256(()),
                }
            },
            "producer_order": [
                "baseline:v2",
                "probe:v3",
                "summarizer:reporting-v1",
            ],
            "marker_summarizer_protocol_version": "v3",
        },
    }
    lock = _write(
        root / "configs/REPORTING_LOCK.v1.json",
        (json.dumps(payload, sort_keys=True) + "\n").encode(),
    )
    provenance = {
        "final_code": final_code,
        "probe_receipt": probe_receipt,
        **{f"probe_{role}": path for role, path in probe_artifacts.items()},
        **{
            f"baseline_{role}": v3_provenance[role]
            for role in ("predictions", "selection", "metrics", "progress")
        },
    }
    return lock, provenance


def test_formal_evaluation_lock_binds_code_and_all_input_paths(tmp_path: Path) -> None:
    lock, foundation, cnn, tabular = _synthetic_lock(tmp_path)
    code_receipt = verify_evaluation_code_lock(experiment_root=tmp_path, lock_path=lock)
    assert code_receipt["verified_file_count"] == 1
    receipt = verify_formal_evaluation_lock(
        experiment_root=tmp_path,
        lock_path=lock,
        foundation_features=foundation,
        current_cnn_features=cnn,
        fold_manifest=tabular[0],
        clinical_labels=tabular[1],
        radiomics=tabular[2],
    )
    assert receipt["current_cnn_assets"] == 10
    assert receipt["foundation_models"] == ("medical", "dino")


def test_formal_evaluation_lock_fails_on_byte_or_argument_drift(tmp_path: Path) -> None:
    lock, foundation, cnn, tabular = _synthetic_lock(tmp_path)
    foundation[0].write_bytes(b"drift")
    with pytest.raises(ValueError, match="SHA-256 drifted"):
        verify_formal_evaluation_lock(
            experiment_root=tmp_path,
            lock_path=lock,
            foundation_features=foundation,
            current_cnn_features=cnn,
            fold_manifest=tabular[0],
            clinical_labels=tabular[1],
            radiomics=tabular[2],
        )


def test_schema_v2_chain_verifies_final_code_parent_inputs_and_each_consumer(
    tmp_path: Path,
) -> None:
    lock, foundation, cnn, tabular, argv, _ = _synthetic_v2_chain(tmp_path)
    baseline = verify_formal_evaluation_lock(
        experiment_root=tmp_path,
        lock_path=lock,
        foundation_features=foundation,
        current_cnn_features=cnn,
        fold_manifest=tabular[0],
        clinical_labels=tabular[1],
        radiomics=tabular[2],
        expected_consumer="baseline",
        command_argv=argv["baseline"],
    )
    assert baseline["schema_version"] == 2
    assert baseline["lock_generation"] == "whole_evaluation_v2"
    assert baseline["expected_consumer"] == "baseline"
    assert baseline["foundation_models"] == ("medical", "dino")
    assert baseline["current_cnn_assets"] == 10
    assert baseline["argument_vector_sha256"] == argument_vector_sha256(
        argv["baseline"]
    )

    reporting = verify_evaluation_code_lock(
        experiment_root=tmp_path,
        lock_path=lock,
        expected_consumer="summarizer",
        command_argv=argv["summarizer"],
    )
    assert reporting["expected_consumer"] == "summarizer"
    assert reporting["parent_lock_sha256"] == file_sha256(
        tmp_path / "configs/EVALUATION_LOCK.json"
    )
    wrong_clinical = _write(tmp_path / "inputs/wrong-clinical.csv", b"wrong\n")
    with pytest.raises(ValueError, match="tabular path arguments"):
        verify_formal_evaluation_lock(
            experiment_root=tmp_path,
            lock_path=lock,
            foundation_features=foundation,
            current_cnn_features=cnn,
            fold_manifest=tabular[0],
            clinical_labels=wrong_clinical,
            radiomics=tabular[2],
            expected_consumer="baseline",
            command_argv=argv["baseline"],
        )


def test_schema_v2_chain_fails_closed_on_argv_parent_or_final_code_drift(
    tmp_path: Path,
) -> None:
    lock, _, _, _, argv, provenance = _synthetic_v2_chain(tmp_path)
    with pytest.raises(ValueError, match="command argv"):
        verify_evaluation_code_lock(
            experiment_root=tmp_path,
            lock_path=lock,
            expected_consumer="summarizer",
            command_argv=(*argv["summarizer"], "--drift"),
        )

    provenance["final_code"].write_bytes(b"final-code-drift")
    with pytest.raises(ValueError, match="evaluation file drifted"):
        verify_evaluation_code_lock(
            experiment_root=tmp_path,
            lock_path=lock,
            expected_consumer="summarizer",
            command_argv=argv["summarizer"],
        )

    # Restore final code, then prove the byte-identical parent record is active.
    provenance["final_code"].write_bytes(b"final-v2-code\n")
    provenance["parent_lock"].write_text(
        provenance["parent_lock"].read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="parent lock SHA-256 drifted"):
        verify_evaluation_code_lock(
            experiment_root=tmp_path,
            lock_path=lock,
            expected_consumer="summarizer",
            command_argv=argv["summarizer"],
        )


def test_schema_v2_rejects_false_abort_claims_and_provenance_path_traversal(
    tmp_path: Path,
) -> None:
    lock, _, _, _, argv, _ = _synthetic_v2_chain(tmp_path)
    payload = json.loads(lock.read_text(encoding="utf-8"))
    payload["aborted_v1_attempts"][0]["outputs_written"] = True
    lock.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="aborted v1 attempt provenance"):
        verify_evaluation_code_lock(
            experiment_root=tmp_path,
            lock_path=lock,
            expected_consumer="summarizer",
            command_argv=argv["summarizer"],
        )

    lock, _, _, _, argv, _ = _synthetic_v2_chain(tmp_path / "traversal")
    payload = json.loads(lock.read_text(encoding="utf-8"))
    payload["development_provenance"]["source_snapshot"]["path"] = "../escape"
    lock.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="root-relative without traversal"):
        verify_evaluation_code_lock(
            experiment_root=tmp_path / "traversal",
            lock_path=lock,
            expected_consumer="summarizer",
            command_argv=argv["summarizer"],
        )


@pytest.mark.parametrize("name", ["source_snapshot", "patch_manifest", "smoke_receipt"])
def test_schema_v2_rejects_development_provenance_byte_drift(
    tmp_path: Path, name: str
) -> None:
    lock, _, _, _, argv, provenance = _synthetic_v2_chain(tmp_path)
    provenance[name].write_bytes(b"development provenance drift")
    with pytest.raises(ValueError, match="SHA-256 drifted"):
        verify_evaluation_code_lock(
            experiment_root=tmp_path,
            lock_path=lock,
            expected_consumer="summarizer",
            command_argv=argv["summarizer"],
        )


def test_metric_free_run_provenance_round_trip_and_artifact_drift(
    tmp_path: Path,
) -> None:
    lock, _, _, _, argv, _ = _synthetic_v2_chain(tmp_path)
    artifacts = {
        "predictions_private": _write(
            tmp_path / "predictions/baseline.private.csv",
            b"synthetic prediction bytes\n",
        ),
        "metrics_public": _write(
            tmp_path / "metrics/baseline.csv", b"synthetic metric bytes\n"
        ),
    }
    receipt_path = tmp_path / "metrics/baseline_run.private.provenance.json"
    written = write_metric_free_run_provenance(
        experiment_root=tmp_path,
        lock_path=lock,
        expected_consumer="baseline",
        command_argv=argv["baseline"],
        artifacts=artifacts,
        receipt_path=receipt_path,
    )
    assert written["metric_values_viewed"] is False
    assert written["artifact_hash_method"] == "sha256_binary_stream_no_parse"
    assert receipt_path.stat().st_mode & 0o777 == 0o600
    assert "synthetic metric bytes" not in receipt_path.read_text(encoding="utf-8")

    verified = verify_metric_free_run_provenance(
        experiment_root=tmp_path,
        lock_path=lock,
        receipt_path=receipt_path,
        expected_consumer="baseline",
        expected_artifacts=artifacts,
    )
    assert verified["artifact_count"] == 2
    assert verified["receipt_sha256"] == file_sha256(receipt_path)

    artifacts["metrics_public"].write_bytes(b"artifact drift")
    with pytest.raises(ValueError, match="SHA-256 drifted"):
        verify_metric_free_run_provenance(
            experiment_root=tmp_path,
            lock_path=lock,
            receipt_path=receipt_path,
            expected_consumer="baseline",
            expected_artifacts=artifacts,
        )


def test_metric_free_run_provenance_rejects_rewrite_and_outside_root(
    tmp_path: Path,
) -> None:
    lock, _, _, _, argv, _ = _synthetic_v2_chain(tmp_path)
    artifact = _write(tmp_path / "metrics/probe.csv", b"probe bytes")
    receipt = tmp_path / "metrics/probe.private.provenance.json"
    kwargs = {
        "experiment_root": tmp_path,
        "lock_path": lock,
        "expected_consumer": "probe",
        "command_argv": argv["probe"],
        "artifacts": {"probe_metrics": artifact},
        "receipt_path": receipt,
    }
    write_metric_free_run_provenance(**kwargs)
    with pytest.raises(FileExistsError, match="already exists"):
        write_metric_free_run_provenance(**kwargs)

    outside = _write(tmp_path.parent / f"{tmp_path.name}-outside.csv", b"outside")
    with pytest.raises(ValueError, match="within experiment root"):
        write_metric_free_run_provenance(
            experiment_root=tmp_path,
            lock_path=lock,
            expected_consumer="probe",
            command_argv=argv["probe"],
            artifacts={"outside": outside},
            receipt_path=tmp_path / "metrics/outside.private.provenance.json",
        )


def test_schema_v3_mixed_lineage_accepts_historical_v2_and_active_probe(
    tmp_path: Path,
) -> None:
    lock, foundation, cnn, tabular, argv, provenance = _synthetic_v3_chain(tmp_path)
    code = verify_evaluation_code_lock(
        experiment_root=tmp_path,
        lock_path=lock,
        expected_consumer="summarizer",
        command_argv=argv["summarizer"],
    )
    assert code["schema_version"] == 3
    assert code["lock_generation"] == "probe_retry_v3"
    assert code["argument_count"] == 0
    formal = verify_formal_evaluation_lock(
        experiment_root=tmp_path,
        lock_path=lock,
        foundation_features=foundation,
        current_cnn_features=cnn,
        fold_manifest=tabular[0],
        clinical_labels=tabular[1],
        radiomics=tabular[2],
        expected_consumer="probe",
        command_argv=argv["probe"],
    )
    assert formal["foundation_models"] == (
        "medicalnet_resnet50_3dseg8",
        "dino_vitb16_imagenet1k",
    )
    historical = verify_historical_metric_free_run_provenance(
        experiment_root=tmp_path,
        active_lock_path=lock,
        producer_key="baseline_v2",
        expected_artifacts={
            role: provenance[role]
            for role in ("predictions", "selection", "metrics", "progress")
        },
    )
    assert historical["lock_generation"] == "whole_evaluation_v2"
    assert historical["artifact_count"] == 4
    assert historical["receipt_sha256"] == file_sha256(
        provenance["baseline_receipt"]
    )

    probe_artifact = _write(tmp_path / "metrics/probe.csv", b"probe bytes\n")
    probe_receipt = tmp_path / "metrics/probe_v3.private.provenance.json"
    written = write_metric_free_run_provenance(
        experiment_root=tmp_path,
        lock_path=lock,
        expected_consumer="probe",
        command_argv=argv["probe"],
        artifacts={"probe_metrics": probe_artifact},
        receipt_path=probe_receipt,
    )
    assert written["lock_sha256"] == file_sha256(lock)
    verified = verify_metric_free_run_provenance(
        experiment_root=tmp_path,
        lock_path=lock,
        receipt_path=probe_receipt,
        expected_consumer="probe",
        expected_artifacts={"probe_metrics": probe_artifact},
    )
    assert verified["artifact_count"] == 1


def test_schema_v3_rejects_historical_drift_substitution_and_baseline_rerun(
    tmp_path: Path,
) -> None:
    lock, foundation, cnn, tabular, argv, provenance = _synthetic_v3_chain(tmp_path)
    with pytest.raises(ValueError, match="consumer must be probe"):
        verify_formal_evaluation_lock(
            experiment_root=tmp_path,
            lock_path=lock,
            foundation_features=foundation,
            current_cnn_features=cnn,
            fold_manifest=tabular[0],
            clinical_labels=tabular[1],
            radiomics=tabular[2],
            expected_consumer="baseline",
            command_argv=(),
        )
    outside = _write(tmp_path / "metrics/substitute.csv", b"substitute\n")
    with pytest.raises(ValueError, match="path differs"):
        verify_historical_metric_free_run_provenance(
            experiment_root=tmp_path,
            active_lock_path=lock,
            producer_key="baseline_v2",
            expected_artifacts={
                **{
                    role: provenance[role]
                    for role in ("selection", "metrics", "progress")
                },
                "predictions": outside,
            },
        )
    provenance["predictions"].write_bytes(b"historical drift")
    with pytest.raises(ValueError, match="SHA-256 drifted"):
        verify_evaluation_code_lock(
            experiment_root=tmp_path,
            lock_path=lock,
            expected_consumer="summarizer",
            command_argv=(),
        )


def test_schema_v3_rejects_false_probe_failure_claim_and_progress_drift(
    tmp_path: Path,
) -> None:
    lock, _, _, _, argv, provenance = _synthetic_v3_chain(tmp_path)
    payload = json.loads(lock.read_text(encoding="utf-8"))
    payload["aborted_v2_attempts"][0]["formal_exception_claimed"] = True
    lock.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="failed v2 probe provenance"):
        verify_evaluation_code_lock(
            experiment_root=tmp_path,
            lock_path=lock,
            expected_consumer="summarizer",
            command_argv=argv["summarizer"],
        )

    lock, _, _, _, argv, provenance = _synthetic_v3_chain(tmp_path / "progress")
    provenance["failed_progress"].write_bytes(
        provenance["failed_progress"].read_bytes()[:-1] + b"z"
    )
    with pytest.raises(ValueError, match="SHA-256 drifted"):
        verify_evaluation_code_lock(
            experiment_root=tmp_path / "progress",
            lock_path=lock,
            expected_consumer="summarizer",
            command_argv=argv["summarizer"],
        )


def test_reporting_lock_accepts_empty_summarizer_and_both_historical_producers(
    tmp_path: Path,
) -> None:
    lock, provenance = _synthetic_reporting_lock(tmp_path)
    code = verify_evaluation_code_lock(
        experiment_root=tmp_path,
        lock_path=lock,
        expected_consumer="summarizer",
        command_argv=(),
    )
    assert code["schema_version"] == "foundation_mri_reporting_lock_v1"
    assert code["argument_count"] == 0
    assert code["marker_summarizer_protocol_version"] == "v3"
    assert code["finalization_lock_sha256"] == file_sha256(
        tmp_path / "configs/FINALIZATION_LOCK.v1.json"
    )
    baseline = verify_historical_metric_free_run_provenance(
        experiment_root=tmp_path,
        active_lock_path=lock,
        producer_key="baseline_v2",
        expected_artifacts={
            role: provenance[f"baseline_{role}"]
            for role in ("predictions", "selection", "metrics", "progress")
        },
    )
    assert baseline["consumer"] == "baseline"
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
    probe = verify_historical_metric_free_run_provenance(
        experiment_root=tmp_path,
        active_lock_path=lock,
        producer_key="probe_v3",
        expected_artifacts={
            role: provenance[f"probe_{role}"] for role in probe_roles
        },
    )
    assert probe["consumer"] == "probe"
    assert probe["lock_generation"] == "probe_retry_v3"
    assert probe["artifact_count"] == 10


def test_reporting_lock_rejects_argv_visibility_and_historical_probe_drift(
    tmp_path: Path,
) -> None:
    lock, provenance = _synthetic_reporting_lock(tmp_path / "argv")
    with pytest.raises(ValueError, match="argv"):
        verify_evaluation_code_lock(
            experiment_root=tmp_path / "argv",
            lock_path=lock,
            expected_consumer="summarizer",
            command_argv=("--unexpected",),
        )

    lock, _ = _synthetic_reporting_lock(tmp_path / "visibility")
    payload = json.loads(lock.read_text(encoding="utf-8"))
    payload["visibility"][
        "auroc_auprc_brier_ece_prediction_or_selection_performance_values_seen"
    ] = True
    lock.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="visibility"):
        verify_evaluation_code_lock(
            experiment_root=tmp_path / "visibility",
            lock_path=lock,
            expected_consumer="summarizer",
            command_argv=(),
        )

    lock, provenance = _synthetic_reporting_lock(tmp_path / "probe-drift")
    provenance["probe_ftv_metrics"].write_bytes(b"drifted historical artifact\n")
    with pytest.raises(ValueError, match="SHA-256 drifted"):
        verify_evaluation_code_lock(
            experiment_root=tmp_path / "probe-drift",
            lock_path=lock,
            expected_consumer="summarizer",
            command_argv=(),
        )


def test_reporting_lock_rejects_probe_count_and_marker_protocol_drift(
    tmp_path: Path,
) -> None:
    lock, _ = _synthetic_reporting_lock(tmp_path)
    payload = json.loads(lock.read_text(encoding="utf-8"))
    payload["historical_producers"]["probe_v3"]["audit_counts"][
        "max_ridge_n_iter"
    ] = 806
    lock.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="audited counts"):
        verify_evaluation_code_lock(
            experiment_root=tmp_path,
            lock_path=lock,
            expected_consumer="summarizer",
            command_argv=(),
        )

    lock, _ = _synthetic_reporting_lock(tmp_path / "marker")
    payload = json.loads(lock.read_text(encoding="utf-8"))
    payload["execution"]["marker_summarizer_protocol_version"] = "v4"
    lock.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="execution contract"):
        verify_evaluation_code_lock(
            experiment_root=tmp_path / "marker",
            lock_path=lock,
            expected_consumer="summarizer",
            command_argv=(),
        )

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foundation_mri.data import FOLDS, HR_HER2_SUBTYPES  # noqa: E402
from foundation_mri.evaluation import EvaluationResult  # noqa: E402
from foundation_mri.locking import argument_vector_sha256, file_sha256  # noqa: E402


def _load_probe_script():
    path = ROOT / "scripts/run_probes.py"
    spec = importlib.util.spec_from_file_location("foundation_mri_test_run_probes", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_script(name: str):
    path = ROOT / f"scripts/{name}.py"
    spec = importlib.util.spec_from_file_location(f"foundation_mri_test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_probe_cli_expands_current_cnn_template_to_exactly_five_folds() -> None:
    module = _load_probe_script()
    groups = module._parse_cnn_templates(
        ["GAP0,GLOBAL,/locked/GAP0/fold_{fold}/response_state.private.npz"]
    )
    assert set(groups) == {("GAP0", "GLOBAL")}
    assert set(groups[("GAP0", "GLOBAL")]) == set(FOLDS)
    assert groups[("GAP0", "GLOBAL")][4].as_posix().endswith(
        "/fold_4/response_state.private.npz"
    )
    with pytest.raises(ValueError, match=r"literal \{fold\}|invalid"):
        module._parse_cnn_templates(["LOCAL0,LOCAL,/locked/LOCAL0/fold_0/file.npz"])


def test_fold_specific_probe_source_covers_hr_her2_subtype_and_ftv(monkeypatch) -> None:
    module = _load_probe_script()
    patient_ids = np.asarray([f"P{index:02d}" for index in range(10)], dtype=str)
    radiomics_ids = patient_ids[:5]
    clinical = SimpleNamespace(
        patient_ids=patient_ids,
        hr=np.arange(10) % 2,
        her2=(np.arange(10) // 2) % 2,
        subtype=np.asarray([HR_HER2_SUBTYPES[index % 4] for index in range(10)]),
    )
    radiomics = SimpleNamespace(patient_ids=radiomics_ids)
    by_fold = {
        fold: np.broadcast_to(
            (fold + (fold + 1) * np.arange(4, dtype=np.float32))[None, :, None],
            (10, 4, 3),
        ).copy()
        for fold in FOLDS
    }
    source = module.ProbeSource("LOCAL0", "LOCAL", by_fold)
    ftv = np.arange(20, dtype=np.float64).reshape(5, 4) + 1.0
    calls: list[tuple[str, dict]] = []

    def fake(kind: str):
        def evaluate(**kwargs):
            matrices = kwargs["matrices"]
            if callable(matrices):
                assert matrices(0).shape == matrices(1).shape
                assert not np.array_equal(matrices(0), matrices(1))
            elif isinstance(matrices, dict):
                assert set(matrices) == set(FOLDS)
                assert not np.array_equal(matrices[0], matrices[1])
            calls.append((kind, kwargs))
            tiny = pd.DataFrame({"kind": [kind]})
            return EvaluationResult(tiny, tiny.copy())

        return evaluate

    monkeypatch.setattr(module, "evaluate_binary_cv", fake("binary"))
    monkeypatch.setattr(module, "evaluate_multiclass_cv", fake("subtype"))
    monkeypatch.setattr(module, "evaluate_ridge_cv", fake("ridge"))
    binary_results: list[EvaluationResult] = []
    subtype_results: list[EvaluationResult] = []
    ridge_results: list[EvaluationResult] = []
    module._append_source_probes(
        source,
        clinical=clinical,
        folds=object(),
        radiomics=radiomics,
        ftv=ftv,
        cohort_size=10,
        radiomics_size=5,
        binary_results=binary_results,
        subtype_results=subtype_results,
        ridge_results=ridge_results,
    )

    assert [call[1]["target_name"] for call in calls if call[0] == "binary"] == [
        "HR",
        "HER2",
    ]
    subtype = [call[1] for call in calls if call[0] == "subtype"]
    assert len(subtype) == 1
    assert subtype[0]["target_name"] == "HR_HER2_subtype"
    assert subtype[0]["timing"] == "T0"
    ridge = [call[1] for call in calls if call[0] == "ridge"]
    assert [(call["task"], call["endpoint"]) for call in ridge] == [
        ("static", "T0"),
        ("static", "T1"),
        ("static", "T2"),
        ("static", "T3"),
        ("delta", "T0-T1"),
        ("delta", "T1-T2"),
        ("delta", "T2-T3"),
    ]
    assert all(call["model_name"] == "LOCAL0" for _, call in calls)
    assert all(call["spatial"] == "LOCAL" for _, call in calls)


def test_medicalnet_synthetic_smoke_is_deterministic_and_outcome_blind(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_probe_script()
    n = 60
    feature_path = tmp_path / "synthetic_medicalnet.npz"
    np.savez(feature_path, patient_id=np.asarray([f"P{i:03d}" for i in range(n)]))
    representation = np.arange(n * 4 * 2 * 3, dtype=np.float32).reshape(n, 4, 2, 3)

    class FakeAsset:
        model_name = "medicalnet_resnet50_3dseg8"

        def __init__(self) -> None:
            self.representation = representation

        def spatial(self, spatial: str) -> np.ndarray:
            return representation[:, :, 0 if spatial == "GLOBAL" else 1, :]

    monkeypatch.setattr(module, "load_foundation_features", lambda *args, **kwargs: FakeAsset())

    def forbidden(*args, **kwargs):
        raise AssertionError("outcome loader was called by synthetic smoke")

    monkeypatch.setattr(module, "load_fold_manifest", forbidden)
    monkeypatch.setattr(module, "load_clinical_labels", forbidden)
    monkeypatch.setattr(module, "load_radiomics_table", forbidden)

    def fake_select(*args, penalties, c_grid, **kwargs):
        grid = tuple(
            {
                "penalty": penalty,
                "C": float(c_value),
                "n_iter": 7,
                "estimator_count": 4,
                "n_iter_by_class_json": json.dumps(
                    {value: [7] for value in module.SMOKE_CLASSES}, sort_keys=True
                ),
            }
            for penalty in penalties
            for c_value in c_grid
        )
        return SimpleNamespace(
            grid=grid,
            model=SimpleNamespace(
                coef_=np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64),
                intercept_=np.asarray([0.25], dtype=np.float64),
            ),
        )

    monkeypatch.setattr(module, "select_multiclass_logistic", fake_select)
    common = {
        "expected_sha256": file_sha256(feature_path),
        "expected_n": n,
        "expected_dim": 3,
        "spatial_axes": ("GLOBAL", "LOCAL"),
        "penalties": ("l1", "l2"),
        "c_grid": (0.1, 1.0),
        "require_thread_contract": False,
    }
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first = module._run_medicalnet_synthetic_smoke(feature_path, first_path, **common)
    second = module._run_medicalnet_synthetic_smoke(feature_path, second_path, **common)
    assert first["candidate_count"] == 8
    assert first["underlying_estimator_fit_count"] == 32
    assert first["max_observed_n_iter"] == 7
    assert first["min_observed_n_iter"] == 7
    assert first["all_candidate_n_iter_nonnegative"] is True
    assert first["all_candidate_n_iter_strictly_less_than_max_iter"] is True
    assert first["ridge_candidate_count"] == 2 * len(module.RIDGE_ALPHAS)
    assert first["ridge_all_candidate_n_iter_nonnegative"] is True
    assert first["ridge_all_candidate_n_iter_strictly_less_than_max_iter"] is True
    assert all(
        int(value) >= 1
        for value in first["ridge_reserved_rows_clipped_by_axis"].values()
    )
    for payload in (first, second):
        payload.pop("elapsed_seconds")
        payload.pop("elapsed_seconds_by_axis")
        payload.pop("ridge_elapsed_seconds_by_axis")
    assert first == second
    serialized = first_path.read_text(encoding="utf-8").lower()
    assert not any(
        forbidden_key in serialized
        for forbidden_key in (
            "patient_id",
            "y_true",
            "y_pred",
            "probability",
            "auroc",
            "auprc",
            "brier",
            str(tmp_path).lower(),
        )
    )
    assert json.loads(first_path.read_text(encoding="utf-8"))["status"] == "PASS"


def test_smoke_cli_early_exit_never_calls_normal_probe_run(monkeypatch, tmp_path: Path) -> None:
    module = _load_probe_script()
    monkeypatch.setattr(
        module,
        "_run_medicalnet_synthetic_smoke",
        lambda feature, receipt: {
            "candidate_count": 36,
            "max_observed_n_iter": 99,
            "elapsed_seconds": 1.25,
        },
    )
    monkeypatch.setattr(module, "run", lambda args: (_ for _ in ()).throw(AssertionError()))
    assert (
        module.main(
            [
                "--medicalnet-synthetic-smoke",
                str(tmp_path / "feature.npz"),
                "--smoke-receipt",
                str(tmp_path / "receipt.json"),
            ]
        )
        == 0
    )


@pytest.mark.parametrize(
    ("script_name", "consumer", "lock_name", "extra_argv"),
    (
        ("run_baselines", "baseline", "EVALUATION_LOCK.v2.json", ()),
        (
            "run_probes",
            "probe",
            "EVALUATION_LOCK.v3.json",
            ("--foundation-feature", "/does/not/open.npz"),
        ),
    ),
)
def test_formal_evaluation_clis_gate_before_any_outcome_loader(
    monkeypatch,
    script_name: str,
    consumer: str,
    lock_name: str,
    extra_argv: tuple[str, ...],
) -> None:
    module = _load_script(script_name)
    assert module.DEFAULT_EVALUATION_LOCK.name == lock_name
    observed: dict[str, object] = {}

    def verify(**kwargs):
        observed.update(kwargs)
        return {"lock_sha256": "a" * 64}

    def stop_before_outcomes(*args, **kwargs):
        raise RuntimeError("OUTCOME_LOADER_SENTINEL")

    monkeypatch.setattr(module, "verify_formal_evaluation_lock", verify)
    monkeypatch.setattr(module, "configure_metric_free_progress", lambda path: path)
    monkeypatch.setattr(module, "metric_free_progress", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "load_fold_manifest", stop_before_outcomes)
    raw = ("--evaluation-lock", str(module.DEFAULT_EVALUATION_LOCK), *extra_argv)
    args = module.build_parser().parse_args(raw)
    with pytest.raises(RuntimeError, match="OUTCOME_LOADER_SENTINEL"):
        module.run(args, command_argv=raw)
    assert observed["expected_consumer"] == consumer
    assert tuple(observed["command_argv"]) == raw


def test_formal_summarizer_verifies_mixed_lineage_before_reporting(monkeypatch) -> None:
    module = _load_script("summarize_results")
    assert module.DEFAULT_EVALUATION_LOCK.name == "REPORTING_LOCK.v1.json"
    calls: list[str] = []
    monkeypatch.setattr(
        module,
        "verify_evaluation_code_lock",
        lambda **kwargs: calls.append(f"code:{kwargs['expected_consumer']}") or {},
    )

    def historical(**kwargs):
        calls.append(f"historical:{kwargs['producer_key']}")
        if kwargs["producer_key"] == "probe_v3":
            raise RuntimeError("PROVENANCE_SENTINEL")
        return {}

    monkeypatch.setattr(module, "verify_historical_metric_free_run_provenance", historical)
    monkeypatch.setattr(
        module,
        "summarize_results",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("CSV read reached")),
    )
    with pytest.raises(RuntimeError, match="PROVENANCE_SENTINEL"):
        module.main(())
    assert calls == [
        "code:summarizer",
        "historical:baseline_v2",
        "historical:probe_v3",
    ]


def test_reporting_retry_marker_keeps_frozen_v3_label_and_uses_active_code_lock(
    monkeypatch,
) -> None:
    module = _load_script("summarize_results")
    empty_sha = argument_vector_sha256(())
    monkeypatch.setattr(
        module,
        "verify_evaluation_code_lock",
        lambda **kwargs: {
            "lock_sha256": "a" * 64,
            "argument_vector_sha256": empty_sha,
        },
    )
    baseline_roles = ("predictions", "selection", "metrics", "progress")
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

    def historical(**kwargs):
        roles = baseline_roles if kwargs["producer_key"] == "baseline_v2" else probe_roles
        return {
            "lock_sha256": ("b" if kwargs["producer_key"] == "baseline_v2" else "c") * 64,
            "receipt_sha256": ("d" if kwargs["producer_key"] == "baseline_v2" else "e") * 64,
            "argument_vector_sha256": ("f" if kwargs["producer_key"] == "baseline_v2" else "1") * 64,
            "artifact_sha256": {role: f"{index + 2:064x}" for index, role in enumerate(roles)},
        }

    monkeypatch.setattr(module, "verify_historical_metric_free_run_provenance", historical)
    observed: dict[str, object] = {}

    def summarize(*args, **kwargs):
        observed.update(kwargs)
        return {"public_outputs": 5}

    monkeypatch.setattr(module, "summarize_results", summarize)
    assert module.main(()) == 0
    lineage = observed["reporting_lineage"]
    assert lineage["summarizer"]["protocol_version"] == "v3"
    assert lineage["summarizer"]["code_lock_sha256"] == "a" * 64
    assert lineage["probe_v3"]["protocol_version"] == "v3"


def test_formal_summarizer_requires_exact_empty_argv(monkeypatch) -> None:
    module = _load_script("summarize_results")
    monkeypatch.setattr(
        module,
        "verify_evaluation_code_lock",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("lock read reached")),
    )
    with pytest.raises(ValueError, match="exact empty argv"):
        module.main(("--evaluation-lock", str(module.DEFAULT_EVALUATION_LOCK)))


def test_v3_protocol_execution_and_metric_free_count_contracts_are_self_consistent() -> None:
    protocol = json.loads(
        (ROOT / "configs/evaluation_protocol_v3.json").read_text(encoding="utf-8")
    )
    execution = protocol["execution_contract"]
    for consumer in ("probe_v3", "summarizer_v3"):
        record = execution[consumer]
        argv = tuple(record["argv"])
        assert record["argument_count"] == len(argv)
        assert record["argument_vector_sha256"] == argument_vector_sha256(argv)
    counts = protocol["metric_free_monitoring_v3"]["expected_formal_probe_counts"]
    assert counts == {
        "source_count": 6,
        "fold_started": 300,
        "fold_selection_completed": 300,
        "fold_completed": 300,
        "candidate_started": 3300,
        "candidate_completed": 3300,
        "ovr_estimator_started": 2160,
        "ovr_estimator_completed": 2160,
        "binary_candidate_count": 1080,
        "multiclass_candidate_count": 540,
        "ridge_candidate_count": 1680,
    }
    assert (
        protocol["v3_solver_contract"]["delta_ftv_bound_application"]
        == "unbounded_fail_on_nonfinite"
    )


def test_reporting_retry_protocol_is_empty_argv_and_reporting_only() -> None:
    protocol = json.loads(
        (ROOT / "configs/reporting_retry_protocol_v1.json").read_text(
            encoding="utf-8"
        )
    )
    retry = protocol["formal_retry_contract"]
    assert retry["summarizer_argv"] == []
    assert retry["summarizer_argument_vector_sha256"] == argument_vector_sha256(())
    assert retry["marker_summarizer_protocol_version"] == "v3"
    numeric = protocol["numeric_identity_contract"]
    assert numeric["binary_irls_relaxed_call_sites"] == ["baseline", "phenotype"]
    assert numeric["ftv_all_numeric_including_same_named_calibration_columns_strict"]
    assert numeric["selection_or_model_fit_changed"] is False
    visibility = protocol["visibility"]
    assert visibility["calibration_intercept_values_seen"] is True
    assert (
        visibility[
            "auroc_auprc_brier_ece_prediction_or_selection_performance_values_seen"
        ]
        is False
    )

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import types

import numpy as np
import pandas as pd
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import pytest


ROOT = Path(__file__).parents[1]


def _module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


REPORT = _module("conditional_generate_report_hardening", "generate_report.py")
VERIFY = _module("conditional_verify_hardening", "verify_experiment.py")


def _auroc(arm: str, family: str) -> float:
    if family == "M":
        return {"B0": 0.50, "B1": 0.80, "B2": 0.60, "B3": 0.55}[arm]
    if family in {"C", "C+F"}:
        return 0.50
    if family == "C+M":
        return 0.55
    if family == "C+F+M":
        return 0.52
    return 0.48  # F


def _write_complete_bundle(root: Path) -> None:
    metrics_root = root / "metrics"
    metrics_root.mkdir(parents=True)
    pd.DataFrame([{
        "population": "full_808", "expected_files": 808, "stat_verified_files": 808,
        "sha256_verified_files": 808, "mismatches": 0, "external_files_hashed": 0,
        "cache_manifest_sha256": "b" * 64, "stage_b_manifest_sha256": "c" * 64,
        "content_digest_aggregate_sha256": (
            "f1c9965e8ae5456a899735a5462b76277ba0ec97a229dedc5faf9c380ce94c89"
        ),
        "content_sha256_verified": True,
    }]).to_csv(metrics_root / "cache_integrity_audit.csv", index=False)
    model_rows: list[dict[str, object]] = []
    for population, families in REPORT.POPULATION_FAMILIES.items():
        n, positive, negative = REPORT.POPULATION_COUNTS[population]
        for seed in REPORT.SEEDS:
            for arm in REPORT.ARMS:
                for timing in REPORT.TIMINGS:
                    for family in families:
                        model_rows.append({
                            "population": population,
                            "seed": seed,
                            "arm": arm,
                            "timing": timing,
                            "model_family": family,
                            "n": n,
                            "n_positive": positive,
                            "n_negative": negative,
                            "auroc": _auroc(arm, family),
                            "auprc": 0.40,
                            "brier": 0.20,
                            "calibration_slope": 1.00,
                            "ece10": 0.10,
                            "supplementary": timing == "T0_T3",
                        })
    aggregate = pd.DataFrame(model_rows)
    aggregate.to_csv(metrics_root / "aggregate_metrics.csv", index=False)
    lookup = {
        tuple(getattr(row, column) for column in ("population", "seed", "arm", "timing", "model_family")): row.auroc
        for row in aggregate.itertuples(index=False)
    }

    pair_specs = {
        "MRI_ceiling": ("full_808", "B0", "M", "M", 808),
        "clinical_complementarity": ("full_808", None, "C", "C+M", 808),
        "beyond_ftv": ("ftv_complete_375", None, "C+F", "C+F+M", 375),
    }
    bootstrap_rows: list[dict[str, object]] = []
    for comparison, (population, fixed_reference, reference_family, comparison_family, n) in pair_specs.items():
        for seed in REPORT.SEEDS:
            for arm in REPORT.SUPERVISED_ARMS:
                for timing in REPORT.PRIMARY_TIMINGS:
                    reference_arm = fixed_reference or arm
                    reference = float(lookup[(population, seed, reference_arm, timing, reference_family)])
                    compared = float(lookup[(population, seed, arm, timing, comparison_family)])
                    delta = compared - reference
                    bootstrap_rows.append({
                        "comparison": comparison,
                        "population": population,
                        "seed": seed,
                        "arm": arm,
                        "timing": timing,
                        "reference_arm": reference_arm,
                        "reference_family": reference_family,
                        "comparison_family": comparison_family,
                        "delta_auroc": delta,
                        "ci_lower": delta - 0.01,
                        "ci_upper": delta + 0.01,
                        "reference_auroc": reference,
                        "comparison_auroc": compared,
                        "n_patients": n,
                        "n_folds": 5,
                        "n_bootstrap": 5000,
                        "n_valid_bootstrap": 5000,
                        "confidence_level": 0.95,
                        "bootstrap_unit": "patient_within_outer_fold",
                        "ci_method": "percentile",
                        "orientation": "comparison - reference",
                        "bootstrap_seed": REPORT._stable_seed(
                            seed, arm, timing,
                            {"MRI_ceiling": "mri", "clinical_complementarity": "cm", "beyond_ftv": "cfm"}[comparison],
                            base=260_812,
                        ),
                    })
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(metrics_root / "paired_bootstrap.csv", index=False)

    locked = {
        0: (525, 28, 26, 24, 506, 19, 347, 178, 347, 159),
        1: (525, 28, 26, 22, 513, 12, 346, 179, 344, 169),
        2: (525, 28, 27, 24, 520, 5, 347, 178, 345, 175),
        3: (526, 28, 27, 24, 515, 11, 347, 179, 346, 169),
        4: (526, 28, 27, 26, 524, 2, 347, 179, 345, 179),
    }
    matching_columns = (
        "training_patients", "total_strata", "usable_strata", "bidirectionally_usable_strata",
        "usable_patients", "dropped_anchors", "pcr_negative", "pcr_positive",
        "usable_pcr_negative", "usable_pcr_positive",
    )
    pd.DataFrame([
        {
            "fold": fold,
            "scope": "outer_train_only",
            **dict(zip(matching_columns, values, strict=True)),
            "dropped_no_same_class_partner": {0: 2, 1: 6, 2: 5, 3: 3, 4: 2}[fold],
            "dropped_no_opposite_class": {0: 17, 1: 6, 2: 0, 3: 8, 4: 1}[fold],
            "natural_batch_size_3": {0: 2, 1: 0, 2: 0, 3: 2, 4: 0}[fold],
            "natural_batch_size_4": {0: 504, 1: 513, 2: 520, 3: 513, 4: 524}[fold],
            "max_unique_patients_per_logical_batch": 4,
            "unmatched_fallback_used": False,
            "test_patients_used": False,
        }
        for fold, values in locked.items()
    ]).to_csv(metrics_root / "matching_audit.csv", index=False)

    gaps: list[dict[str, object]] = []
    for row in aggregate.itertuples(index=False):
        output = {
            column: getattr(row, column)
            for column in ("population", "seed", "arm", "timing", "model_family")
        }
        for metric in REPORT.METRICS:
            test = float(getattr(row, metric))
            train, validation = test + 0.02, test + 0.01
            output[f"train_{metric}"] = train
            output[f"validation_{metric}"] = validation
            output[f"test_{metric}"] = test
            output[f"train_test_{metric}_gap"] = train - test
            output[f"validation_test_{metric}_gap"] = validation - test
        output["supplementary"] = row.timing == "T0_T3"
        gaps.append(output)
    pd.DataFrame(gaps).to_csv(metrics_root / "generalization_gaps.csv", index=False)

    probes = [
        {
            "seed": seed, "arm": arm, "timing": timing, "target": target,
            "metric": "macro_ovr_auroc" if target == "subtype" else "auroc",
            "value": 0.60, "n": 808, "n_folds": 5, "fold_isolated": True,
            "status": "ok",
        }
        for seed in REPORT.SEEDS for arm in ("B0", "B2", "B3")
        for timing in REPORT.PRIMARY_TIMINGS for target in ("HR", "HER2", "subtype")
    ]
    probes.append({
        "seed": -1, "arm": "ALL", "timing": "ALL", "target": "treatment",
        "metric": "not_run", "value": np.nan, "n": 808, "n_folds": 0,
        "fold_isolated": True,
        "status": "unsuitable_exact_13_arm_target_due_to_sparse_fold_classes",
    })
    pd.DataFrame(probes).to_csv(metrics_root / "clinical_profile_probes.csv", index=False)

    subgroup_rows = [
        {
            "seed": seed, "arm": arm, "timing": timing, "subgroup": subgroup,
            "eligible": True, "status": "ok", "n_folds": 5,
            "n": n, "n_positive": max(1, n // 3), "n_negative": n - max(1, n // 3),
            "auroc": 0.60, "auprc": 0.40, "brier": 0.20,
            "calibration_slope": 1.0, "ece10": 0.10,
        }
        for seed in REPORT.SEEDS for arm in REPORT.ARMS for timing in REPORT.PRIMARY_TIMINGS
        for subgroup, n in REPORT.SUBGROUP_COUNTS.items()
    ]
    pd.DataFrame(subgroup_rows).to_csv(metrics_root / "subgroup_refits.csv", index=False)

    diagnostic_rows: list[dict[str, object]] = []
    for row in aggregate.itertuples(index=False):
        for fold in REPORT.FOLDS:
            for split in ("train", "validation", "test"):
                diagnostic_rows.append({
                    "population": row.population, "seed": row.seed, "arm": row.arm,
                    "timing": row.timing, "model_family": row.model_family,
                    "fold": fold, "split": split, "n": 2, "n_positive": 1,
                    "selected_dimension": 8 if row.model_family in {"M", "C+M", "C+F+M"} else np.nan,
                    "selected_C": 1.0, "validation_selection_auroc": 0.5,
                    "n_negative": 1, "auroc": 0.5, "auprc": 0.5, "brier": 0.25,
                    "calibration_slope": 1.0, "ece10": 0.1,
                    "supplementary": row.timing == "T0_T3",
                })
    pd.DataFrame(diagnostic_rows).to_csv(metrics_root / "fold_diagnostics.csv", index=False)

    training = [
        {
            "seed": seed, "arm": arm, "fold": fold,
            "selection_status": "SELECTED_VALIDATION_ONLY", "selected_epoch": 1,
            "epochs_run": 2, "selected_validation_mean_auroc": 0.6,
            "anchor_sampling_strategy": (
                None if arm == "B1" else "all_eligible_anchors_exactly_once_per_epoch"
            ),
            "logical_patient_batch_size": None if arm == "B1" else 4,
            "encoder_microbatch_size": 1 if arm == "B3" else None,
            "eligible_anchors_per_epoch": None if arm == "B1" else locked[fold][4],
            "feature_sha256": hashlib.sha256(f"{seed}-{arm}-{fold}".encode()).hexdigest(),
            "config_sha256": "a" * 64,
            "test_labels_used": False, "external_ispy1_patients_used": 0,
            "world_model_claim_allowed": False,
        }
        for seed in REPORT.SEEDS for arm in REPORT.SUPERVISED_ARMS for fold in REPORT.FOLDS
    ]
    pd.DataFrame(training).to_csv(metrics_root / "training_summary.csv", index=False)

    decision = REPORT.evaluate_gates(
        bootstrap.loc[bootstrap["comparison"].eq("MRI_ceiling")],
        bootstrap.loc[bootstrap["comparison"].eq("clinical_complementarity")],
        bootstrap.loc[bootstrap["comparison"].eq("beyond_ftv")],
    ).as_dict()
    (metrics_root / "decision_summary.json").write_text(json.dumps({
        "schema_version": 1,
        "reporting_boundary": REPORT.BOUNDARY,
        "primary_seeds": list(REPORT.SEEDS),
        "folds_are_biological_replicates": False,
        "headline_bootstrap_draws": 5000,
        "private_predictions_sha256": "0" * 64,
        "decision": decision,
    }), encoding="utf-8")


def test_report_validates_complete_bundle_and_uses_literal_b3_minus_b0(tmp_path: Path) -> None:
    _write_complete_bundle(tmp_path)

    output = REPORT.generate_report(tmp_path)

    text = output.read_text(encoding="utf-8")
    assert "B3−B0" in text
    assert "约为 +0.050 AUROC（B3 0.550 对 B0 0.500）" in text
    assert "natural batch=3" in text and "natural batch=4" in text
    assert "BCE 在每个采样逻辑 batch 的全部行上计算" in text
    assert "support 患者可在同一 epoch 被重复抽到" in text
    assert "不可用 exact strata 的患者不进入 B2/B3 优化" in text


def test_report_rejects_same_size_registry_substitution_before_overwrite(tmp_path: Path) -> None:
    _write_complete_bundle(tmp_path)
    output = tmp_path / "reports" / "final_report.md"
    output.parent.mkdir()
    output.write_text("sentinel", encoding="utf-8")
    aggregate_path = tmp_path / "metrics" / "aggregate_metrics.csv"
    aggregate = pd.read_csv(aggregate_path)
    aggregate.loc[aggregate.index[-1], ["population", "seed", "arm", "timing", "model_family"]] = aggregate.loc[
        aggregate.index[0], ["population", "seed", "arm", "timing", "model_family"]
    ].to_numpy()
    aggregate.to_csv(aggregate_path, index=False)

    with pytest.raises(ValueError, match="Cartesian product"):
        REPORT.generate_report(tmp_path)
    assert output.read_text(encoding="utf-8") == "sentinel"


def test_report_rejects_extra_subject_uuid_column_in_aggregate(tmp_path: Path) -> None:
    _write_complete_bundle(tmp_path)
    aggregate_path = tmp_path / "metrics" / "aggregate_metrics.csv"
    aggregate = pd.read_csv(aggregate_path)
    aggregate["subject_uuid"] = "real-patient-identifier"
    aggregate.to_csv(aggregate_path, index=False)

    with pytest.raises(ValueError, match=r"schema invalid.*subject_uuid"):
        REPORT.generate_report(tmp_path)


@pytest.mark.parametrize(
    ("filename", "column", "value"),
    (
        ("matching_audit.csv", "usable_patients", 506.9),
        ("paired_bootstrap.csv", "n_bootstrap", 5000.9),
    ),
)
def test_report_rejects_fractional_registered_counts_and_draws(
    tmp_path: Path, filename: str, column: str, value: float
) -> None:
    _write_complete_bundle(tmp_path)
    path = tmp_path / "metrics" / filename
    frame = pd.read_csv(path)
    frame[column] = frame[column].astype(float)
    frame.loc[0, column] = value
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="non-integral"):
        REPORT.generate_report(tmp_path)


def test_report_rejects_extra_decision_json_key(tmp_path: Path) -> None:
    _write_complete_bundle(tmp_path)
    decision_path = tmp_path / "metrics" / "decision_summary.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["subject_uuid"] = "real-patient-identifier"
    decision_path.write_text(json.dumps(decision), encoding="utf-8")

    with pytest.raises(ValueError, match="decision summary wrapper"):
        REPORT.generate_report(tmp_path)


@pytest.mark.parametrize("mutation", ["decision", "pair", "probe"])
def test_report_rejects_fabricated_or_incomplete_semantics(tmp_path: Path, mutation: str) -> None:
    _write_complete_bundle(tmp_path)
    metrics = tmp_path / "metrics"
    if mutation == "decision":
        path = metrics / "decision_summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["decision"]["interpretation_class"] = "A"
        payload["decision"]["interpretation_code"] = "FABRICATED_PASS"
        path.write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "pair":
        path = metrics / "paired_bootstrap.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "reference_arm"] = "B1"
        frame.to_csv(path, index=False)
    else:
        path = metrics / "clinical_profile_probes.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "n"] = 807
        frame.to_csv(path, index=False)

    with pytest.raises(ValueError):
        REPORT.generate_report(tmp_path)


def _write_public_file_fixture(root: Path) -> None:
    (root / "metrics").mkdir(parents=True, exist_ok=True)
    (root / "figures").mkdir(exist_ok=True)
    (root / "reports").mkdir(exist_ok=True)
    for name in VERIFY.PUBLIC_CSVS:
        pd.DataFrame({"aggregate_only": [1]}).to_csv(root / "metrics" / name, index=False)
    decision = {
        "decision": {"interpretation_code": "LOCKED_INTERPRETATION"},
    }
    (root / "metrics" / "decision_summary.json").write_text(json.dumps(decision), encoding="utf-8")
    (root / "reports" / "final_report.md").write_text(
        VERIFY.BOUNDARY + "\nB3−B0\nLOCKED_INTERPRETATION\n" + ("内容" * 4000), encoding="utf-8"
    )
    rows = []
    sources = {
        "01_mri_only_auroc.png": "metrics/aggregate_metrics.csv",
        "02_clinical_complementarity.png": "metrics/paired_bootstrap.csv",
        "03_beyond_ftv_complementarity.png": "metrics/paired_bootstrap.csv",
        "04_hr_her2_subgroups.png": "metrics/subgroup_refits.csv",
        "05_generalization_gap.png": "metrics/generalization_gaps.csv",
        "06_profile_decodability.png": "metrics/clinical_profile_probes.csv",
        "07_supervised_ceiling_gap.png": "metrics/aggregate_metrics.csv",
    }
    titles = {
        "01_mri_only_auroc.png": "B0/B1/B2/B3 MRI-only AUROC",
        "02_clinical_complementarity.png": "Clinical complementarity: C+M - C",
        "03_beyond_ftv_complementarity.png": "Beyond-FTV complementarity",
        "04_hr_her2_subgroups.png": "pCR performance within HR/HER2 subgroups",
        "05_generalization_gap.png": "Train/validation/test generalization",
        "06_profile_decodability.png": "Clinical-profile decodability",
        "07_supervised_ceiling_gap.png": "Supervised ceiling gap vs current World Model",
    }
    for index, filename in enumerate(sorted(VERIFY.FIGURE_FILENAMES)):
        path = root / "figures" / filename
        Image.new("RGB", (12, 8), color=(index * 20, 30, 40)).save(path)
        rows.append({
            "filename": filename,
            "title": titles[filename],
            "relative_path": f"figures/{filename}",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
            "source_csvs": sources[filename],
            "public_aggregate_only": True,
            "contains_patient_rows": False,
        })
    pd.DataFrame(rows).to_csv(root / "figures" / "figure_manifest.csv", index=False)


def _configure_public_validator(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> str:
    monkeypatch.setattr(VERIFY, "EXPERIMENT_ROOT", root)
    monkeypatch.setattr(VERIFY, "REPO_ROOT", root.parent)
    monkeypatch.setattr(VERIFY, "_run", lambda *command: "")
    if hasattr(VERIFY, "_frozen_sensitive_signatures"):
        monkeypatch.setattr(VERIFY, "_frozen_sensitive_signatures", lambda: (set(), set()))
    report_text = (root / "reports" / "final_report.md").read_text(encoding="utf-8")
    monkeypatch.setattr(VERIFY, "render_report", lambda experiment_root: report_text)

    def regenerate(experiment_root: Path, *, output_dir: Path, **kwargs: object) -> pd.DataFrame:
        del experiment_root, kwargs
        output_dir.mkdir(parents=True, exist_ok=True)
        for index, filename in enumerate(sorted(VERIFY.FIGURE_FILENAMES)):
            Image.new("RGB", (12, 8), color=(index * 20, 30, 40)).save(output_dir / filename)
        return pd.DataFrame()

    monkeypatch.setattr(VERIFY, "generate_figures", regenerate)
    return report_text


def test_public_files_reject_manifest_escape_and_patient_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_public_file_fixture(tmp_path)
    _configure_public_validator(tmp_path, monkeypatch)
    VERIFY._validate_public_files()

    outside = tmp_path.parent / "outside.png"
    outside.write_bytes(b"outside")
    manifest_path = tmp_path / "figures" / "figure_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    manifest.loc[0, "relative_path"] = "../outside.png"
    manifest.loc[0, "sha256"] = hashlib.sha256(outside.read_bytes()).hexdigest()
    manifest.loc[0, "bytes"] = outside.stat().st_size
    manifest.to_csv(manifest_path, index=False)
    with pytest.raises(ValueError, match="escapes"):
        VERIFY._validate_public_files()

    _write_public_file_fixture(tmp_path)
    pd.DataFrame({"Patient_ID": ["P001"], "score": [0.5]}).to_csv(
        tmp_path / "reports" / "patient_dump.csv", index=False
    )
    with pytest.raises(ValueError, match="patient-level|closed public-output registry"):
        VERIFY._validate_public_files()


def test_public_files_reject_extra_figure_manifest_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_public_file_fixture(tmp_path)
    _configure_public_validator(tmp_path, monkeypatch)
    manifest_path = tmp_path / "figures" / "figure_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    manifest["subject_uuid"] = "forged-identifier"
    manifest.to_csv(manifest_path, index=False)

    with pytest.raises(ValueError, match="manifest"):
        VERIFY._validate_public_files()


def test_public_files_reject_stale_report_and_same_pixel_nonidentical_png(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_public_file_fixture(tmp_path)
    expected_report = _configure_public_validator(tmp_path, monkeypatch)
    report = tmp_path / "reports" / "final_report.md"
    report.write_text(expected_report + "\n伪造陈述", encoding="utf-8")
    with pytest.raises(ValueError, match="stale"):
        VERIFY._validate_public_files()

    report.write_text(expected_report, encoding="utf-8")
    filename = sorted(VERIFY.FIGURE_FILENAMES)[0]
    image_path = tmp_path / "figures" / filename
    with Image.open(image_path) as decoded:
        same_pixels = decoded.convert("RGB").copy()
    metadata = PngInfo()
    metadata.add_text("forged", "same decoded pixels, different payload")
    same_pixels.save(image_path, pnginfo=metadata)
    manifest_path = tmp_path / "figures" / "figure_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    row = manifest["filename"].eq(filename)
    manifest.loc[row, "sha256"] = hashlib.sha256(image_path.read_bytes()).hexdigest()
    manifest.loc[row, "bytes"] = image_path.stat().st_size
    manifest.to_csv(manifest_path, index=False)
    with pytest.raises(ValueError, match="bytes differ"):
        VERIFY._validate_public_files()


def test_public_files_reject_non_png_payload_with_png_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_public_file_fixture(tmp_path)
    _configure_public_validator(tmp_path, monkeypatch)
    filename = sorted(VERIFY.FIGURE_FILENAMES)[0]
    image_path = tmp_path / "figures" / filename
    with Image.open(image_path) as decoded:
        pixels = decoded.convert("RGB").copy()
    pixels.save(image_path, format="BMP")
    manifest_path = tmp_path / "figures" / "figure_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    row = manifest["filename"].eq(filename)
    manifest.loc[row, "sha256"] = hashlib.sha256(image_path.read_bytes()).hexdigest()
    manifest.loc[row, "bytes"] = image_path.stat().st_size
    manifest.to_csv(manifest_path, index=False)

    with pytest.raises(ValueError, match=r"not (?:a valid decodable )?PNG"):
        VERIFY._validate_public_files()


@pytest.mark.parametrize(
    "relative_path",
    (
        "README.md",
        "src/conditional_ceiling/module.py",
        "tests/test_static.py",
        "configs/experiment.json",
        "reports/final_report.md",
    ),
)
def test_static_delivery_scan_rejects_real_patient_identifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative_path: str
) -> None:
    patient_id = b"frozen-real-patient-uuid"
    monkeypatch.setattr(
        VERIFY,
        "_frozen_sensitive_signatures",
        lambda: ({patient_id}, {b"/private/exact/cache/path.npy"}),
    )
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"ordinary delivery content\n" + patient_id + b"\n")

    with pytest.raises(ValueError, match=r"patient|sensitive|private"):
        VERIFY._scan_deliverable_content([path])


def test_static_delivery_scan_rejects_exact_private_cache_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_path = b"/private/exact/cache/patient-record.npy"
    monkeypatch.setattr(
        VERIFY,
        "_frozen_sensitive_signatures",
        lambda: ({b"frozen-real-patient-uuid"}, {private_path}),
    )
    path = tmp_path / "scripts" / "static_record.py"
    path.parent.mkdir(parents=True)
    path.write_bytes(b'RECORD = "' + private_path + b'"\n')

    with pytest.raises(ValueError, match=r"patient|sensitive|private"):
        VERIFY._scan_deliverable_content([path])


def test_public_verifier_scans_static_git_delivery_for_real_patient_identifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_public_file_fixture(tmp_path)
    _configure_public_validator(tmp_path, monkeypatch)
    patient_id = b"frozen-real-patient-uuid"
    monkeypatch.setattr(
        VERIFY,
        "_frozen_sensitive_signatures",
        lambda: ({patient_id}, {b"/private/exact/cache/path.npy"}),
    )
    readme = tmp_path / "README.md"
    readme.write_bytes(b"patient record: " + patient_id)
    git_path = f"{tmp_path.name}/README.md"

    def fake_git(*command: str) -> str:
        if command == ("git", "ls-files"):
            return git_path
        return ""

    monkeypatch.setattr(VERIFY, "_run", fake_git)
    with pytest.raises(ValueError, match=r"patient|sensitive|private"):
        VERIFY._validate_public_files()


def test_static_delivery_scan_does_not_treat_registered_counts_as_patient_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        VERIFY,
        "_frozen_sensitive_signatures",
        lambda: ({b"frozen-real-patient-uuid"}, {b"/private/exact/cache/patient-record.npy"}),
    )
    path = tmp_path / "README.md"
    path.write_text(
        "Counts and seeds are not identities: 808, 375, 275, 533, 2026, 3026, 5000.",
        encoding="utf-8",
    )

    VERIFY._scan_deliverable_content([path])


@pytest.mark.parametrize(
    "relative_path",
    ("metrics/patients.txt", "reports/rows.json", "figures/patient_dump.npy", "unexpected.txt"),
)
def test_public_delivery_closed_allowlist_rejects_any_extra_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative_path: str
) -> None:
    _write_public_file_fixture(tmp_path)
    _configure_public_validator(tmp_path, monkeypatch)
    extra = tmp_path / relative_path
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_text("patient rows", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected|closed"):
        VERIFY._validate_public_files()


@pytest.mark.parametrize("source", ["tracked", "untracked"])
def test_delivery_rejects_changed_or_untracked_files_outside_experiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    _write_public_file_fixture(tmp_path)
    _configure_public_validator(tmp_path, monkeypatch)

    def fake_git(*command: str) -> str:
        if source == "tracked" and command[:3] == ("git", "diff", "--name-only"):
            return "additional_experiments/prior_experiment/changed.py"
        if source == "untracked" and command == ("git", "ls-files", "--others", "--exclude-standard"):
            return "additional_experiments/prior_experiment/untracked.txt"
        return ""

    monkeypatch.setattr(VERIFY, "_run", fake_git)
    with pytest.raises(ValueError, match="prior experiment paths changed"):
        VERIFY._validate_public_files()


def test_private_header_only_file_fails_exact_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predictions = tmp_path / "predictions" / "oof_predictions.private.csv"
    predictions.parent.mkdir(mode=0o700)
    columns = [
        "patient_id", "fold", "population", "seed", "arm", "timing", "model_family",
        "split", "y_true", "predicted_probability", "selected_dimension", "selected_C",
    ]
    pd.DataFrame(columns=columns).to_csv(predictions, index=False)
    predictions.chmod(0o600)
    predictions.parent.chmod(0o700)
    monkeypatch.setattr(VERIFY, "EXPERIMENT_ROOT", tmp_path)
    checkpoint = tmp_path / "authoritative.pt"
    checkpoint.write_bytes(b"authoritative checkpoint")
    checkpoint_sha = VERIFY._sha256(checkpoint)
    selection = {
        "arm": "B1",
        "history": [{"epoch": 1, "validation_mean_auroc": 0.6}],
        "selected_epoch": 1,
        "selected_validation_mean_auroc": 0.6,
        "selection_timings": ["T0", "T0_T1", "T0_T2"],
        "test_labels_used": False,
    }
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps({
            "selection": selection,
            "confirmed_checkpoint_sha256": checkpoint_sha,
        }),
        encoding="utf-8",
    )

    class FakeCell:
        def __init__(self, seed: int, arm: str, fold: int) -> None:
            self.seed = seed
            self.arm = "B1"
            self.fold = fold

        def outputs(self) -> tuple[Path, Path, Path, Path]:
            return checkpoint, selection_path, tmp_path / "feature.npz", tmp_path / "metadata.json"

    fake = types.ModuleType("run_matrix")
    fake.Cell = FakeCell  # type: ignore[attr-defined]
    fake.validate_cell_artifacts = lambda cell: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "run_matrix", fake)
    fake_torch = types.ModuleType("torch")
    fake_torch.load = lambda *args, **kwargs: {  # type: ignore[attr-defined]
        "selection": selection,
        "provenance": {"confirmed_checkpoint_sha256": checkpoint_sha},
    }
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    fake_paths = types.SimpleNamespace(checkpoint_path=lambda seed, fold: checkpoint)
    monkeypatch.setattr(VERIFY, "load_config", lambda: {})
    monkeypatch.setattr(VERIFY, "resolve_input_paths", lambda config: fake_paths)

    with pytest.raises(ValueError, match="row registry"):
        VERIFY._validate_private_results()


def test_deterministic_private_bootstrap_rejects_tampered_ci() -> None:
    patient_ids = [f"P{index:03d}" for index in range(40)]
    labels = np.tile([0, 1], 20)
    folds = np.repeat(np.arange(5), 8)
    reference_probability = np.where(labels == 1, 0.58, 0.42) + np.linspace(-0.08, 0.08, 40)
    comparison_probability = np.where(labels == 1, 0.72, 0.28) + np.linspace(-0.06, 0.06, 40)

    def block(arm: str, probability: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame({
            "patient_id": patient_ids,
            "fold": folds,
            "y_true": labels,
            "predicted_probability": probability,
            "population": "full_808",
            "seed": 2026,
            "arm": arm,
            "timing": "T0_T1",
            "model_family": "M",
            "split": "test",
        })

    reference = block("B0", reference_probability)
    comparison = block("B2", comparison_probability)
    held_out = pd.concat((reference, comparison), ignore_index=True)
    config = {"bootstrap": {"draws": 5000, "confidence_level": 0.95, "seed": 260_812}}
    seed = VERIFY._stable_seed(2026, "B2", "T0_T1", "mri", base=260_812)
    result = VERIFY.paired_fold_stratified_bootstrap(
        reference, comparison, n_bootstrap=5000, confidence_level=0.95,
        seed=seed, metrics=("auroc",),
    ).iloc[0]
    summary = pd.DataFrame([{
        "comparison": "MRI_ceiling", "population": "full_808", "seed": 2026,
        "arm": "B2", "timing": "T0_T1", "reference_arm": "B0",
        "reference_family": "M", "comparison_family": "M",
        "delta_auroc": result["delta"], "ci_lower": result["ci_lower"],
        "ci_upper": result["ci_upper"], "reference_auroc": result["reference"],
        "comparison_auroc": result["comparison_value"], "n_patients": result["n_patients"],
        "n_folds": result["n_folds"], "n_bootstrap": result["n_bootstrap"],
        "n_valid_bootstrap": result["n_valid_bootstrap"],
        "confidence_level": result["confidence_level"],
        "bootstrap_unit": result["bootstrap_unit"], "ci_method": result["ci_method"],
        "orientation": result["orientation"], "bootstrap_seed": result["seed"],
    }])
    VERIFY._validate_bootstrap_against_predictions(held_out, summary, config)

    summary.loc[0, "ci_lower"] = float(summary.loc[0, "ci_lower"]) + 0.2
    with pytest.raises(ValueError, match="CI/point/metadata"):
        VERIFY._validate_bootstrap_against_predictions(held_out, summary, config)


def test_private_prediction_regeneration_rejects_stale_probability() -> None:
    expected = pd.DataFrame({
        "patient_id": ["P001", "P002"],
        "fold": [0, 0],
        "population": ["full_808", "full_808"],
        "seed": [2026, 2026],
        "arm": ["B2", "B2"],
        "timing": ["T0_T1", "T0_T1"],
        "model_family": ["M", "M"],
        "split": ["test", "test"],
        "y_true": [0, 1],
        "predicted_probability": [0.21, 0.79],
        "selected_dimension": [8, 8],
        "selected_C": [1.0, 1.0],
    })
    published = expected.copy()
    VERIFY._assert_private_predictions_match(published, expected)

    published.loc[0, "predicted_probability"] = 0.21000000000000002
    with pytest.raises(ValueError, match="private probabilities"):
        VERIFY._assert_private_predictions_match(published, expected)


@pytest.mark.parametrize(
    "mutation",
    ["checkpoint_selection", "chosen_epoch", "base_sha", "history_order", "fractional_epoch"],
)
def test_supervised_selection_provenance_rejects_forged_selection(
    tmp_path: Path, mutation: str
) -> None:
    import torch

    authoritative = tmp_path / "confirmed" / "selected.pt"
    authoritative.parent.mkdir(parents=True)
    authoritative.write_bytes(b"authoritative LOCAL3")
    digest = hashlib.sha256(authoritative.read_bytes()).hexdigest()
    cell_root = tmp_path / "cell"
    cell_root.mkdir()
    checkpoint_path = cell_root / "selected.private.pt"
    selection_path = cell_root / "selection.private.json"
    feature_path = cell_root / "representation.private.npz"
    matching_path = cell_root / "matching.private.json"
    history = [
        {"epoch": 1, "validation_mean_auroc": 0.60},
        {"epoch": 2, "validation_mean_auroc": 0.70},
        {"epoch": 3, "validation_mean_auroc": 0.70},
    ]
    selected = {
        "arm": "B2",
        "selected_epoch": 2,
        "selected_validation_mean_auroc": 0.70,
        "history": history,
        "selection_timings": ["T0", "T0_T1", "T0_T2"],
        "test_labels_used": False,
    }
    payload = {"selection": selected, "confirmed_checkpoint_sha256": digest}
    checkpoint_selection = dict(selected)
    if mutation == "chosen_epoch":
        payload["selection"] = {**selected, "selected_epoch": 3}
        checkpoint_selection = dict(payload["selection"])
    if mutation == "checkpoint_selection":
        checkpoint_selection = {**selected, "selected_epoch": 1}
    if mutation == "base_sha":
        payload["confirmed_checkpoint_sha256"] = "0" * 64
    if mutation == "history_order":
        payload["selection"] = {
            **selected,
            "history": [history[1], history[0], history[2]],
        }
        checkpoint_selection = dict(payload["selection"])
    if mutation == "fractional_epoch":
        payload["selection"] = {
            **selected,
            "history": [{**history[0], "epoch": 1.5}, history[1], history[2]],
        }
        checkpoint_selection = dict(payload["selection"])
    selection_path.write_text(json.dumps(payload), encoding="utf-8")
    torch.save(
        {
            "selection": checkpoint_selection,
            "provenance": {"confirmed_checkpoint_sha256": digest},
        },
        checkpoint_path,
    )
    feature_path.write_bytes(b"unused")
    matching_path.write_bytes(b"unused")

    class Cell:
        seed = 2026
        fold = 0
        arm = "B2"

        @staticmethod
        def outputs() -> tuple[Path, Path, Path, Path]:
            return checkpoint_path, selection_path, feature_path, matching_path

    class Paths:
        @staticmethod
        def checkpoint_path(seed: int, fold: int) -> Path:
            assert (seed, fold) == (2026, 0)
            return authoritative

    with pytest.raises(ValueError):
        VERIFY._validate_supervised_selection_provenance(Cell(), Paths())


def test_verifier_failure_removes_stale_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "reports" / "verification.json"
    output.parent.mkdir(parents=True)
    output.write_text('{"status":"PASS"}', encoding="utf-8")
    monkeypatch.setattr(VERIFY, "EXPERIMENT_ROOT", tmp_path)
    monkeypatch.setattr(VERIFY, "verify", lambda: (_ for _ in ()).throw(ValueError("fail")))
    monkeypatch.setattr(sys, "argv", ["verify_experiment.py"])

    assert VERIFY.main() == 2
    assert not output.exists()


@pytest.mark.parametrize("name", ["clinical_profile_probes.csv", "subgroup_refits.csv"])
def test_private_recomputation_rejects_forged_probe_or_subgroup_value(name: str) -> None:
    expected = pd.DataFrame({"seed": [2026, 3026], "arm": ["B2", "B2"], "value": [0.61, 0.62]})
    forged = expected.copy()
    forged.loc[0, "value"] = 0.99
    with pytest.raises(ValueError, match="deterministic recomputation"):
        VERIFY._assert_recomputed_frame(
            forged, expected, keys=["seed", "arm"], name=name
        )


def test_verifier_cli_has_no_noncanonical_no_private_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["verify_experiment.py", "--no-private"])
    with pytest.raises(SystemExit) as error:
        VERIFY.main()
    assert error.value.code == 2

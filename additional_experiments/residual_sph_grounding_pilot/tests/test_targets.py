from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from residual_sph.targets import (  # noqa: E402
    VISITS,
    file_sha256,
    fit_fold_target_bundle,
    load_fold_split_map,
    load_static_sph_ftv_table,
    save_public_residualizer_json,
)


def _synthetic_matrices(n_patients: int = 12) -> tuple[list[str], np.ndarray, np.ndarray]:
    patient_ids = [f"P{index:03d}" for index in range(n_patients)]
    row = np.arange(n_patients, dtype=np.float64)[:, None]
    visit = np.arange(4, dtype=np.float64)[None, :]
    ftv = 0.8 + (row + 1.0) ** 1.35 * (1.0 + 0.17 * visit)
    sphericity = (
        0.18
        + 0.011 * row
        - 0.012 * visit
        + 0.025 * np.sin(0.7 * row + 0.9 * visit)
    )
    return patient_ids, ftv, sphericity


def _transition_frame(
    patient_ids: list[str], ftv: np.ndarray, sphericity: np.ndarray
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for patient_index, patient_id in enumerate(patient_ids):
        for start_index in range(3):
            start = VISITS[start_index]
            end = VISITS[start_index + 1]
            rows.append(
                {
                    "patient_id": patient_id,
                    "trial_id": 100000 + patient_index,
                    "transition": f"{start}→{end}",
                    "start_visit": start,
                    "end_visit": end,
                    "ftv_start": ftv[patient_index, start_index],
                    "ftv_end": ftv[patient_index, start_index + 1],
                    "ftv_valid": True,
                    "sphericity_start": sphericity[patient_index, start_index],
                    "sphericity_end": sphericity[patient_index, start_index + 1],
                    "sphericity_valid": True,
                    # Deliberately present but outside the target allowlist.
                    "label_pcr": patient_index % 2,
                    "clinical_secret": f"secret-{patient_index}",
                }
            )
    return pd.DataFrame(rows)


def _write_targets(
    tmp_path: Path,
    *,
    patient_ids: list[str] | None = None,
    ftv: np.ndarray | None = None,
    sphericity: np.ndarray | None = None,
    name: str = "targets.csv",
) -> tuple[Path, list[str], np.ndarray, np.ndarray]:
    if patient_ids is None or ftv is None or sphericity is None:
        patient_ids, ftv, sphericity = _synthetic_matrices()
    path = tmp_path / name
    _transition_frame(patient_ids, ftv, sphericity).to_csv(path, index=False)
    return path, patient_ids, ftv, sphericity


def _split_map(patient_ids: list[str]) -> dict[str, str]:
    return {
        patient_id: ("train" if index < 8 else "val" if index < 10 else "test")
        for index, patient_id in enumerate(patient_ids)
    }


def test_authenticated_reconstruction_and_repeated_endpoint_guard(tmp_path: Path) -> None:
    path, patient_ids, ftv, sphericity = _write_targets(tmp_path)
    digest = file_sha256(path)
    table = load_static_sph_ftv_table(
        path,
        digest,
        expected_patient_count=len(patient_ids),
        expected_patient_ids=patient_ids,
    )
    np.testing.assert_allclose(table.ftv, ftv, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(table.sphericity, sphericity, rtol=0.0, atol=1e-12)
    assert table.patient_ids == tuple(patient_ids)

    with pytest.raises(ValueError, match="SHA-256 drift"):
        load_static_sph_ftv_table(
            path, "0" * 64, expected_patient_count=len(patient_ids)
        )

    inconsistent = _transition_frame(patient_ids, ftv, sphericity)
    # P000/T1 is first written as the first row's end and then repeated as the
    # second row's start.  Perturb only the repeated copy.
    inconsistent.loc[1, "sphericity_start"] += 0.01
    inconsistent_path = tmp_path / "inconsistent.csv"
    inconsistent.to_csv(inconsistent_path, index=False)
    with pytest.raises(ValueError, match="inconsistent repeated endpoint"):
        load_static_sph_ftv_table(
            inconsistent_path,
            file_sha256(inconsistent_path),
            expected_patient_count=len(patient_ids),
        )


def test_fold_targets_match_exact_predecessor_math_and_reconstruct(tmp_path: Path) -> None:
    path, patient_ids, _, _ = _write_targets(tmp_path)
    table = load_static_sph_ftv_table(
        path, file_sha256(path), expected_patient_count=len(patient_ids)
    )
    splits = _split_map(patient_ids)
    bundle = fit_fold_target_bundle(table, splits, fold=3)
    train = bundle.splits == "train"

    for visit_index, fitted in enumerate(bundle.residualizers):
        sph = table.sphericity[:, visit_index]
        ftv = table.ftv[:, visit_index]
        sph_lower, sph_upper = np.quantile(sph[train], [0.01, 0.99])
        ftv_lower, ftv_upper = np.quantile(ftv[train], [0.01, 0.99])
        sph_clipped = np.clip(sph, sph_lower, sph_upper)
        ftv_log = np.log1p(np.clip(ftv, ftv_lower, ftv_upper))
        sph_z = (sph_clipped - np.mean(sph_clipped[train])) / np.std(
            sph_clipped[train], ddof=0
        )
        ftv_z = (ftv_log - np.mean(ftv_log[train])) / np.std(ftv_log[train], ddof=0)
        ridge = Ridge(alpha=1.0, fit_intercept=True).fit(
            ftv_z[train, None], sph_z[train]
        )
        conditional = ridge.predict(ftv_z[:, None])
        epsilon = sph_z - conditional
        residual_center = np.mean(epsilon[train])
        residual_scale = np.std(epsilon[train], ddof=0)
        residual_z = (epsilon - residual_center) / residual_scale

        np.testing.assert_allclose(bundle.s1_targets[:, visit_index], sph_z, atol=1e-14)
        np.testing.assert_allclose(bundle.epsilon[:, visit_index], epsilon, atol=1e-14)
        np.testing.assert_allclose(
            bundle.conditional_sph_z[:, visit_index], conditional, atol=1e-14
        )
        np.testing.assert_allclose(bundle.s2_targets[:, visit_index], residual_z, atol=1e-14)
        np.testing.assert_allclose(fitted.coefficient, ridge.coef_[0], atol=1e-14)
        np.testing.assert_allclose(fitted.intercept, ridge.intercept_, atol=1e-14)
        np.testing.assert_allclose(fitted.residual_center, residual_center, atol=1e-14)
        np.testing.assert_allclose(fitted.residual_scale, residual_scale, atol=1e-14)
        np.testing.assert_allclose(np.mean(bundle.s1_targets[train, visit_index]), 0.0, atol=1e-14)
        np.testing.assert_allclose(np.std(bundle.s1_targets[train, visit_index]), 1.0, atol=1e-14)
        np.testing.assert_allclose(np.mean(bundle.s2_targets[train, visit_index]), 0.0, atol=1e-14)
        np.testing.assert_allclose(np.std(bundle.s2_targets[train, visit_index]), 1.0, atol=1e-14)

    expected_winsorized_sph = np.column_stack(
        [
            np.clip(
                table.sphericity[:, visit_index],
                fitted.sph_transform.lower,
                fitted.sph_transform.upper,
            )
            for visit_index, fitted in enumerate(bundle.residualizers)
        ]
    )
    np.testing.assert_allclose(
        bundle.reconstruct_sphericity(bundle.s2_targets),
        expected_winsorized_sph,
        rtol=0.0,
        atol=2e-14,
    )


def test_validation_and_test_cannot_change_any_fitted_quantity(tmp_path: Path) -> None:
    path_a, patient_ids, ftv_a, sph_a = _write_targets(tmp_path, name="targets_a.csv")
    splits = _split_map(patient_ids)
    nontrain = np.asarray([splits[patient_id] != "train" for patient_id in patient_ids])

    ftv_b = ftv_a.copy()
    sph_b = sph_a.copy()
    ftv_b[nontrain] = 10000.0 + 100.0 * ftv_b[nontrain]
    sph_b[nontrain] = 0.95 - 0.01 * sph_b[nontrain]
    path_b, _, _, _ = _write_targets(
        tmp_path,
        patient_ids=patient_ids,
        ftv=ftv_b,
        sphericity=sph_b,
        name="targets_b.csv",
    )

    table_a = load_static_sph_ftv_table(
        path_a, file_sha256(path_a), expected_patient_count=len(patient_ids)
    )
    table_b = load_static_sph_ftv_table(
        path_b, file_sha256(path_b), expected_patient_count=len(patient_ids)
    )
    bundle_a = fit_fold_target_bundle(table_a, splits, fold=0)
    bundle_b = fit_fold_target_bundle(table_b, splits, fold=0)

    assert [fit.to_public_dict() for fit in bundle_a.residualizers] == [
        fit.to_public_dict() for fit in bundle_b.residualizers
    ]
    np.testing.assert_array_equal(bundle_a.s1_targets[bundle_a.train_mask], bundle_b.s1_targets[bundle_b.train_mask])
    np.testing.assert_array_equal(bundle_a.s2_targets[bundle_a.train_mask], bundle_b.s2_targets[bundle_b.train_mask])
    # Held-out natural values really changed.  They can nevertheless map to
    # the same transformed value when both land beyond a frozen winsor bound;
    # that saturation is expected and does not weaken the fit-state check.
    assert not np.array_equal(
        bundle_a.natural_sphericity[~bundle_a.train_mask],
        bundle_b.natural_sphericity[~bundle_b.train_mask],
    )

    output = tmp_path / "fold_0.json"
    save_public_residualizer_json(output, bundle_a)
    payload = json.loads(output.read_text(encoding="utf-8"))
    serialized = json.dumps(payload, sort_keys=True)
    assert all(patient_id not in serialized for patient_id in patient_ids)
    assert "patient_ids" not in serialized
    assert "trial_id" not in serialized
    assert "label_pcr" not in serialized
    assert payload["split_counts"] == {"test": 2, "train": 8, "val": 2}
    assert len(payload["residualizers"]) == 4


def test_fold_loader_materializes_only_split_contract(tmp_path: Path) -> None:
    patient_ids, _, _ = _synthetic_matrices()
    rows = []
    for fold in (0, 1):
        for index, patient_id in enumerate(patient_ids):
            rows.append(
                {
                    "patient_id": patient_id,
                    "fold": fold,
                    "split": "train" if index < 8 else "val" if index < 10 else "test",
                    "label_pcr": index % 2,
                    "clinical_secret": f"never-materialize-{patient_id}",
                }
            )
    path = tmp_path / "folds.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    mapping = load_fold_split_map(
        path,
        file_sha256(path),
        fold=1,
        expected_patient_ids=patient_ids,
    )
    assert set(mapping) == set(patient_ids)
    assert set(mapping.values()) == {"train", "val", "test"}
    assert all(isinstance(value, str) for value in mapping.values())


def test_local_real_data_matches_prior_audit_residualizer_aggregates() -> None:
    """Optional parity proof against the predecessor audit's committed fits."""

    target_path = (
        REPO_ROOT
        / "additional_experiments/radiomics_next_change/data_audit/"
        "radiomics_transition_targets_raw.csv"
    )
    data_contract_path = (
        REPO_ROOT
        / "additional_experiments/c1b_overlap_eligibility_ftv_stageb/manifests/"
        "stage_b_data_contract.private.json"
    )
    if not data_contract_path.is_file():
        pytest.skip("private Stage-B data contract is not mounted")
    data_contract = json.loads(data_contract_path.read_text(encoding="utf-8"))
    fold_path = Path(str(data_contract["fold_manifest"])).expanduser()
    if not fold_path.is_absolute():
        fold_path = (data_contract_path.parent / fold_path).resolve()
    residualizer_path = (
        REPO_ROOT
        / "additional_experiments/nonftv_phenotype_decodability_audit/metrics/"
        "residualizer_fits.csv"
    )
    transform_path = (
        REPO_ROOT
        / "additional_experiments/nonftv_phenotype_decodability_audit/metrics/"
        "target_transform_fits.csv"
    )
    if not all(path.is_file() for path in (target_path, fold_path, residualizer_path, transform_path)):
        pytest.skip("private radiomics/fold inputs are not mounted")

    table = load_static_sph_ftv_table(
        target_path,
        "26fbde8590fde4612267f02d762af99d65926ff6d0206d0e500577ef394ff75d",
        expected_patient_count=375,
    )
    prior_residualizers = pd.read_csv(residualizer_path)
    prior_residualizers = prior_residualizers.loc[
        (prior_residualizers["task_type"] == "static")
        & (prior_residualizers["target_kind"] == "residual_ftv")
        & (prior_residualizers["target"] == "SPH")
    ]
    prior_transforms = pd.read_csv(transform_path)

    for fold in range(5):
        splits = load_fold_split_map(
            fold_path,
            "143e482d711225c0611006d99bd7345d2fa1a5c16c65fbaf8399341a0d26aa38",
            fold=fold,
            expected_patient_ids=table.patient_ids,
        )
        bundle = fit_fold_target_bundle(table, splits, fold=fold)
        for fitted in bundle.residualizers:
            expected_fit = prior_residualizers.loc[
                (prior_residualizers["fold"] == fold)
                & (prior_residualizers["timing"] == fitted.visit)
            ].iloc[0]
            coefficient = json.loads(expected_fit["coefficient_json"])[0]
            np.testing.assert_allclose(fitted.coefficient, coefficient, rtol=0.0, atol=1e-14)
            np.testing.assert_allclose(fitted.intercept, expected_fit["intercept"], rtol=0.0, atol=1e-14)
            np.testing.assert_allclose(
                fitted.residual_center,
                expected_fit["residual_train_mean"],
                rtol=0.0,
                atol=1e-14,
            )
            np.testing.assert_allclose(
                fitted.residual_scale,
                expected_fit["residual_train_population_scale"],
                rtol=0.0,
                atol=1e-14,
            )
            assert fitted.n_train == int(expected_fit["n_train"])

            expected_sph = prior_transforms.loc[
                (prior_transforms["fold"] == fold)
                & (prior_transforms["task_type"] == "static")
                & (prior_transforms["context"] == "residual_ftv_response_SPH")
                & (prior_transforms["timing"] == fitted.visit)
            ].iloc[0]
            expected_ftv = prior_transforms.loc[
                (prior_transforms["fold"] == fold)
                & (prior_transforms["task_type"] == "static")
                & (prior_transforms["context"] == "residual_ftv_predictor_for_SPH")
                & (prior_transforms["timing"] == fitted.visit)
            ].iloc[0]
            for transform, expected in (
                (fitted.sph_transform, expected_sph),
                (fitted.ftv_transform, expected_ftv),
            ):
                # The committed CSV round-trip can move the final decimal by
                # a few ulps; all fitted in-memory values otherwise agree.
                np.testing.assert_allclose(transform.lower, expected["winsor_lower"], rtol=0.0, atol=5e-13)
                np.testing.assert_allclose(transform.upper, expected["winsor_upper"], rtol=0.0, atol=5e-13)
                np.testing.assert_allclose(
                    transform.mean,
                    expected["train_mean_after_family_transform"],
                    rtol=0.0,
                    atol=5e-13,
                )
                np.testing.assert_allclose(
                    transform.scale,
                    expected["train_population_scale"],
                    rtol=0.0,
                    atol=5e-13,
                )

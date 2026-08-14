from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from crps.diagnostics import (  # noqa: E402
    augmentation_cosine_consistency,
    canonical_correlation_summary,
    cross_covariance,
    cross_covariance_norm,
    cross_covariance_summary,
    covariance_eigenspectrum,
    covariance_spectrum_summary,
    effective_rank,
    nearest_neighbor_jaccard_stability,
    paired_cosine_similarities,
    per_dimension_variance_summary,
    regularized_canonical_correlations,
    standardized_cross_covariance,
    standardized_cross_covariance_norm,
    standardized_cross_covariance_summary,
)


def test_per_dimension_variance_summary_reports_sample_variance_and_collapse() -> None:
    values = [[1.0, 5.0, 0.0], [2.0, 5.0, 2.0], [3.0, 5.0, 4.0]]

    result = per_dimension_variance_summary(values)

    np.testing.assert_allclose(result["variance"], [1.0, 0.0, 4.0])
    np.testing.assert_allclose(result["std"], [1.0, 0.0, 2.0])
    np.testing.assert_array_equal(result["collapsed_mask"], [False, True, False])
    assert result["collapsed_dimensions"] == 1
    assert result["noncollapsed_dimensions"] == 2
    assert result["collapsed_fraction"] == pytest.approx(1.0 / 3.0)


def test_variance_summary_validates_finite_matrix_and_ddof() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        per_dimension_variance_summary([1.0, 2.0])
    with pytest.raises(ValueError, match="NaN or infinite"):
        per_dimension_variance_summary([[1.0], [np.nan]])
    with pytest.raises(ValueError, match="ddof"):
        per_dimension_variance_summary([[1.0]], ddof=1)
    with pytest.raises(TypeError, match="ddof"):
        per_dimension_variance_summary([[1.0], [2.0]], ddof=True)


def test_covariance_spectrum_and_entropy_effective_rank() -> None:
    # Centered columns are orthogonal with equal sample variance.
    values = np.asarray(
        [
            [1.0, 1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
        ]
    )

    spectrum = covariance_eigenspectrum(values)
    result = covariance_spectrum_summary(values)

    np.testing.assert_allclose(spectrum, [4.0 / 3.0, 4.0 / 3.0])
    np.testing.assert_allclose(result["explained_variance_ratio"], [0.5, 0.5])
    assert result["effective_rank"] == pytest.approx(2.0)
    assert result["numerical_rank"] == 2
    assert effective_rank([4.0, 4.0]) == pytest.approx(2.0)
    assert effective_rank([9.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_constant_and_rank_deficient_spectra_are_finite() -> None:
    constant = np.ones((8, 4))
    constant_result = covariance_spectrum_summary(constant)
    np.testing.assert_array_equal(constant_result["eigenvalues"], np.zeros(4))
    np.testing.assert_array_equal(
        constant_result["explained_variance_ratio"], np.zeros(4)
    )
    assert constant_result["effective_rank"] == 0.0
    assert constant_result["numerical_rank"] == 0

    coordinate = np.arange(10.0)
    rank_one = np.column_stack((coordinate, 2.0 * coordinate, -coordinate))
    assert covariance_spectrum_summary(rank_one)["effective_rank"] == pytest.approx(
        1.0, abs=1e-12
    )
    assert effective_rank(rank_one) == pytest.approx(1.0, abs=1e-12)
    with pytest.raises(ValueError, match="nonnegative"):
        effective_rank([1.0, -0.1])


def test_standardized_cross_covariance_and_norm_relationships() -> None:
    response = np.asarray(
        [[-1.0, 2.0], [0.0, 2.0], [1.0, 2.0], [2.0, 2.0]]
    )
    phenotype = np.column_stack((response[:, 0], -response[:, 0], np.ones(4)))

    cross_covariance = standardized_cross_covariance(response, phenotype)
    result = standardized_cross_covariance_summary(response, phenotype)

    np.testing.assert_allclose(
        cross_covariance, [[1.0, -1.0, 0.0], [0.0, 0.0, 0.0]], atol=1e-14
    )
    assert np.isfinite(cross_covariance).all()
    assert result["frobenius_norm"] == pytest.approx(np.sqrt(2.0))
    assert result["squared_frobenius_norm"] == pytest.approx(2.0)
    assert result["mean_squared_norm"] == pytest.approx(2.0 / 6.0)
    assert result["root_mean_squared_norm"] == pytest.approx(np.sqrt(2.0 / 6.0))
    assert result["response_constant_dimensions"] == 1
    assert result["phenotype_constant_dimensions"] == 1
    assert standardized_cross_covariance_norm(response, phenotype) == pytest.approx(
        np.sqrt(2.0)
    )


def test_raw_cross_covariance_is_distinct_from_standardized_statistic() -> None:
    response = np.asarray([[-1.0], [0.0], [1.0]])
    phenotype = 10.0 * response

    raw = cross_covariance(response, phenotype)
    standardized = standardized_cross_covariance(response, phenotype)
    result = cross_covariance_summary(response, phenotype)

    np.testing.assert_allclose(raw, [[10.0]])
    np.testing.assert_allclose(standardized, [[1.0]])
    assert cross_covariance_norm(response, phenotype) == pytest.approx(10.0)
    assert result["standardized"] is False
    assert result["frobenius_norm"] == pytest.approx(10.0)
    assert result["squared_frobenius_norm"] == pytest.approx(100.0)
    assert result["mean_squared_norm"] == pytest.approx(100.0)


def test_standardized_cross_covariance_requires_paired_rows() -> None:
    with pytest.raises(ValueError, match="same row count"):
        standardized_cross_covariance(np.ones((3, 2)), np.ones((4, 2)))


def test_regularized_top_k_canonical_correlations_recover_shared_space() -> None:
    rng = np.random.default_rng(17)
    response = rng.normal(size=(80, 3))
    transform = np.asarray([[1.0, 2.0, 0.5], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
    phenotype = response @ transform

    correlations = regularized_canonical_correlations(
        response, phenotype, regularization=0.0, top_k=2
    )

    assert correlations.shape == (2,)
    np.testing.assert_allclose(correlations, np.ones(2), atol=1e-10)
    assert np.all(np.diff(correlations) <= 0.0)


def test_canonical_correlations_are_safe_for_constant_and_rank_deficient_inputs() -> None:
    coordinate = np.linspace(-1.0, 1.0, 12)
    response = np.column_stack((coordinate, coordinate, np.ones(12)))
    phenotype = np.column_stack((2.0 * coordinate, np.zeros(12)))
    correlations = regularized_canonical_correlations(
        response, phenotype, regularization=0.0
    )
    np.testing.assert_allclose(correlations, [1.0, 0.0], atol=1e-10)

    zeros = regularized_canonical_correlations(
        np.ones((5, 2)), np.ones((5, 3)), regularization=0.0
    )
    np.testing.assert_array_equal(zeros, np.zeros(2))


def test_canonical_summary_records_in_sample_leakage_scope() -> None:
    values = np.arange(20.0).reshape(10, 2)
    result = canonical_correlation_summary(values, values, top_k=1)

    assert result["top_k"] == 1
    assert result["n_samples"] == 10
    assert result["response_dimensions"] == 2
    assert result["phenotype_dimensions"] == 2
    assert result["ddof"] == 1
    assert result["regularization_units"] == "correlation"
    assert result["fit_scope"] == "supplied_rows_in_sample"
    assert result["outcome_labels_used"] is False
    assert "training-fold" in result["leakage_note"]
    assert "pCR" in result["leakage_note"]

    with pytest.raises(ValueError, match="top_k"):
        regularized_canonical_correlations(values, values, top_k=3)
    with pytest.raises(ValueError, match="regularization"):
        regularized_canonical_correlations(values, values, regularization=-1.0)


def test_augmentation_cosine_consistency_and_zero_vector_policy() -> None:
    first = np.asarray([[3.0, 4.0], [1.0, 0.0], [0.0, 0.0]])
    second = np.asarray([[6.0, 8.0], [-2.0, 0.0], [0.0, 0.0]])

    similarities = paired_cosine_similarities(first, second)
    result = augmentation_cosine_consistency(first, second)

    np.testing.assert_allclose(similarities, [1.0, -1.0, 0.0])
    assert result["mean_cosine"] == pytest.approx(0.0)
    assert result["median_cosine"] == pytest.approx(0.0)
    assert result["undefined_pairs_mapped_to_zero"] == 1
    assert result["zero_norm_policy"] == "cosine=0"


def test_paired_cosine_is_stable_for_large_finite_values() -> None:
    huge = np.asarray([[1e300, -1e300]])
    np.testing.assert_allclose(paired_cosine_similarities(huge, huge), [1.0])


def test_scale_free_statistics_are_stable_for_extreme_finite_values() -> None:
    huge_spectrum = np.asarray([1e308, 1e308])
    assert effective_rank(huge_spectrum) == pytest.approx(2.0)

    huge = np.asarray([[-1e300], [1e300]])
    np.testing.assert_allclose(standardized_cross_covariance(huge, huge), [[1.0]])
    np.testing.assert_allclose(
        regularized_canonical_correlations(huge, huge, regularization=0.0),
        [1.0],
    )
    np.testing.assert_allclose(
        regularized_canonical_correlations(
            huge, huge, regularization=0.0, standardize=False
        ),
        [1.0],
    )

    full_rank_huge = 1e300 * np.asarray(
        [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]]
    )
    assert effective_rank(full_rank_huge) == pytest.approx(2.0)


def test_raw_scale_statistics_use_safe_math_or_explicitly_reject_overflow() -> None:
    large = 1e150 * np.asarray(
        [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]]
    )
    spectrum = covariance_eigenspectrum(large)
    assert np.isfinite(spectrum).all()
    np.testing.assert_allclose(spectrum / 1e300, [4.0 / 3.0, 4.0 / 3.0])
    variance = per_dimension_variance_summary(large)["variance"]
    np.testing.assert_allclose(variance / 1e300, [4.0 / 3.0, 4.0 / 3.0])

    unrepresentable = np.asarray([[-1e300], [1e300]])
    with pytest.raises(OverflowError, match="rescale"):
        covariance_eigenspectrum(unrepresentable)


def test_nearest_neighbor_stability_aligns_permuted_patient_rows() -> None:
    patient_ids = np.asarray(["P1", "P2", "P3", "P4"])
    embedding = np.asarray([[0.0], [1.0], [4.0], [10.0]])
    permutation = np.asarray([2, 0, 3, 1])

    result = nearest_neighbor_jaccard_stability(
        {2026: embedding, 3026: embedding[permutation]},
        {2026: patient_ids, 3026: patient_ids[permutation]},
        top_k=1,
        metric="euclidean",
    )

    assert result["patient_ids"] == tuple(patient_ids)
    assert result["mean_jaccard"] == 1.0
    np.testing.assert_array_equal(result["per_patient_mean_jaccard"], np.ones(4))
    assert result["neighbors_by_seed"][2026] == result["neighbors_by_seed"][3026]


def test_nearest_neighbor_stability_aggregates_all_seed_pairs() -> None:
    identifiers = ["A", "B", "C", "D", "E"]
    base = np.asarray([[0.0], [1.0], [3.0], [8.0], [20.0]])
    changed = np.asarray([[0.0], [20.0], [3.0], [8.0], [1.0]])

    result = nearest_neighbor_jaccard_stability(
        {"s1": base, "s2": base.copy(), "s3": changed},
        {"s1": identifiers, "s2": identifiers, "s3": identifiers},
        top_k=1,
        metric="euclidean",
    )

    assert len(result["seed_pair_summaries"]) == 3
    assert result["seed_pair_summaries"][0]["mean_jaccard"] == 1.0
    assert 0.0 <= result["mean_jaccard"] < 1.0
    assert np.all((result["per_patient_mean_jaccard"] >= 0.0))
    assert np.all((result["per_patient_mean_jaccard"] <= 1.0))


def test_nearest_neighbor_stability_excludes_collapsed_geometry() -> None:
    identifiers = ["A", "B", "C", "D"]
    collapsed = np.zeros((4, 3))

    result = nearest_neighbor_jaccard_stability(
        {1: collapsed, 2: collapsed.copy()},
        {1: identifiers, 2: identifiers},
        top_k=1,
    )

    assert result["collapsed_seeds"] == (1, 2)
    assert result["status"] == "undefined_no_unambiguous_neighbors"
    assert np.isnan(result["mean_jaccard"])
    assert result["valid_patient_comparisons"] == 0
    assert result["ambiguous_patient_comparisons"] == 4
    # Retained only to make deterministic tie-breaking auditable, not primary.
    assert result["deterministic_mean_jaccard"] == 1.0


def test_nearest_neighbor_stability_marks_top_k_boundary_ties() -> None:
    identifiers = ["center", "right", "left"]
    embedding = np.asarray([[0.0], [1.0], [-1.0]])

    result = nearest_neighbor_jaccard_stability(
        {1: embedding, 2: embedding.copy()},
        {1: identifiers, 2: identifiers},
        top_k=1,
        metric="euclidean",
    )

    assert result["boundary_tie_counts_by_seed"] == {1: 1, 2: 1}
    assert result["valid_patient_comparisons"] == 2
    assert result["ambiguous_patient_comparisons"] == 1
    assert result["mean_jaccard"] == 1.0
    assert np.isnan(result["per_patient_mean_jaccard"][0])


@pytest.mark.parametrize(
    ("representations", "patient_ids", "message"),
    [
        (
            {1: np.zeros((3, 2)), 2: np.zeros((3, 2))},
            {1: ["A", "A", "C"], 2: ["A", "B", "C"]},
            "duplicate patient ID",
        ),
        (
            {1: np.zeros((3, 2)), 2: np.zeros((3, 2))},
            {1: ["A", "B", "C"], 2: ["A", "B", "D"]},
            "patient ID set mismatch",
        ),
    ],
)
def test_nearest_neighbor_stability_rejects_duplicate_or_mismatched_ids(
    representations: dict[int, np.ndarray],
    patient_ids: dict[int, list[str]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        nearest_neighbor_jaccard_stability(
            representations, patient_ids, top_k=1, metric="euclidean"
        )


def test_nearest_neighbor_stability_validates_seed_and_k_contracts() -> None:
    with pytest.raises(ValueError, match="at least two seeds"):
        nearest_neighbor_jaccard_stability(
            {1: np.zeros((3, 2))}, {1: ["A", "B", "C"]}, top_k=1
        )
    with pytest.raises(ValueError, match="top_k"):
        nearest_neighbor_jaccard_stability(
            {1: np.zeros((3, 2)), 2: np.zeros((3, 2))},
            {1: ["A", "B", "C"], 2: ["A", "B", "C"]},
            top_k=3,
        )


def test_diagnostics_source_contains_no_dimensionality_projection_dependency() -> None:
    source = (EXPERIMENT_ROOT / "src" / "crps" / "diagnostics.py").read_text(
        encoding="utf-8"
    )
    forbidden_name = "TS" + "NE"
    assert forbidden_name not in source.upper()

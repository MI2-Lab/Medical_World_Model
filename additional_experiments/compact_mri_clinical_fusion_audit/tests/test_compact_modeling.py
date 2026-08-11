"""Unit tests for leak-resistant compact-modeling primitives."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "compact_modeling.py"
SPEC = importlib.util.spec_from_file_location("compact_modeling", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
compact_modeling = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = compact_modeling
SPEC.loader.exec_module(compact_modeling)


class TrainOnlyPCATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.default_rng(91)
        cls.train = rng.normal(size=(96, 80))
        cls.heldout = rng.normal(loc=7.0, size=(13, 80))

    def test_one_max_fit_is_deterministic_and_sliced(self) -> None:
        first = compact_modeling.fit_train_pca(self.train)
        second = compact_modeling.fit_train_only_pca(self.train.copy())

        self.assertEqual(first.max_components, 64)
        self.assertEqual(first.parameter_sha256, second.parameter_sha256)
        self.assertEqual(len(first.parameter_sha256), 64)
        np.testing.assert_array_equal(first.model.components_, second.model.components_)
        transformed = first.transform(self.heldout)
        self.assertEqual(transformed.shape, (13, 64))
        np.testing.assert_array_equal(
            first.transform_slice(self.heldout, 16), transformed[:, :16]
        )
        np.testing.assert_allclose(first.model.mean_, self.train.mean(axis=0))

        # A held-out distribution shift cannot affect a train-only fit because
        # no held-out argument exists at fit time.
        shifted = self.heldout * 1_000.0
        self.assertEqual(
            first.parameter_sha256,
            compact_modeling.fit_train_pca(self.train).parameter_sha256,
        )
        self.assertFalse(np.allclose(first.transform(shifted), transformed))

    def test_variance_ledgers_are_monotone_and_auditable(self) -> None:
        fit = compact_modeling.fit_train_pca(self.train)
        component_rows = fit.component_variance_ledger()
        aggregate = fit.variance_ledger([64, 8, 32, 16])

        self.assertEqual(len(component_rows), 64)
        self.assertEqual([row["component"] for row in component_rows], list(range(1, 65)))
        self.assertEqual([row["dimension"] for row in aggregate], [8, 16, 32, 64])
        cumulative = [row["cumulative_explained_variance_ratio"] for row in aggregate]
        self.assertTrue(np.all(np.diff(cumulative) >= 0.0))
        self.assertAlmostEqual(
            cumulative[-1], float(fit.model.explained_variance_ratio_.sum())
        )
        self.assertAlmostEqual(
            sum(row["incremental_explained_variance_ratio"] for row in aggregate),
            cumulative[-1],
        )
        self.assertTrue(
            all(row["fitted_transform_sha256"] == fit.parameter_sha256 for row in aggregate)
        )

    def test_fit_parameters_are_read_only(self) -> None:
        fit = compact_modeling.fit_train_pca(self.train)
        for array in (
            fit.model.mean_,
            fit.model.components_,
            fit.model.explained_variance_,
            fit.model.explained_variance_ratio_,
            fit.model.singular_values_,
        ):
            self.assertFalse(array.flags.writeable)
        with self.assertRaises(ValueError):
            fit.model.mean_[0] = 123.0

    def test_pca_rejects_invalid_or_rank_deficient_requests(self) -> None:
        with self.assertRaisesRegex(ValueError, "may not exceed 64"):
            compact_modeling.fit_train_pca(self.train, max_components=65)
        with self.assertRaisesRegex(ValueError, "rank bound"):
            compact_modeling.fit_train_pca(np.ones((8, 20)), max_components=8)
        with self.assertRaisesRegex(ValueError, "zero total variance"):
            compact_modeling.fit_train_pca(np.ones((9, 8)), max_components=8)
        broken = self.train.copy()
        broken[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            compact_modeling.fit_train_pca(broken)
        with self.assertRaisesRegex(ValueError, "svd_solver='full'"):
            compact_modeling.fit_train_pca(self.train, svd_solver="randomized")
        with self.assertRaisesRegex(ValueError, "whiten=False"):
            compact_modeling.fit_train_pca(self.train, whiten=True)

    def test_transform_and_ledger_invariants(self) -> None:
        fit = compact_modeling.fit_train_pca(self.train)
        with self.assertRaisesRegex(ValueError, "expected 80"):
            fit.transform(np.ones((3, 79)))
        with self.assertRaisesRegex(ValueError, "fitted maximum"):
            fit.transform(self.heldout, n_components=65)
        with self.assertRaisesRegex(ValueError, "duplicates"):
            fit.variance_ledger([8, 8])


class GaussianRandomProjectionTests(unittest.TestCase):
    def test_seed_and_dimensions_fully_determine_projection(self) -> None:
        first = compact_modeling.make_gaussian_random_projection(37, 16, seed=260812)
        second = compact_modeling.gaussian_random_projection(37, 16, seed=260812)
        different = compact_modeling.make_gaussian_random_projection(37, 16, seed=260813)

        np.testing.assert_array_equal(first.matrix, second.matrix)
        self.assertEqual(first.matrix_sha256, second.matrix_sha256)
        self.assertNotEqual(first.matrix_sha256, different.matrix_sha256)
        self.assertEqual(first.matrix.shape, (37, 16))
        self.assertAlmostEqual(first.matrix.std(), 1.0 / np.sqrt(16), delta=0.08)
        self.assertFalse(first.matrix.flags.writeable)
        self.assertEqual(first.ledger_record()["matrix_sha256"], first.matrix_sha256)

    def test_transform_is_plain_matrix_multiplication(self) -> None:
        projection = compact_modeling.make_gaussian_random_projection(5, 3, seed=7)
        values = np.arange(20, dtype=np.float64).reshape(4, 5)
        np.testing.assert_array_equal(projection.transform(values), values @ projection.matrix)
        with self.assertRaisesRegex(ValueError, "expected 5"):
            projection.transform(np.ones((4, 6)))
        with self.assertRaises(ValueError):
            compact_modeling.make_gaussian_random_projection(0, 3, seed=7)
        with self.assertRaises(TypeError):
            compact_modeling.make_gaussian_random_projection(True, 3, seed=7)


class ProbabilityToLogitTests(unittest.TestCase):
    def test_clipping_is_finite_symmetric_and_shape_preserving(self) -> None:
        probabilities = np.array([[0.0, 0.25, 0.5], [0.75, 1.0, 1e-9]])
        logits = compact_modeling.probability_to_logit(probabilities)
        self.assertEqual(logits.shape, probabilities.shape)
        self.assertTrue(np.isfinite(logits).all())
        self.assertEqual(logits[0, 2], 0.0)
        self.assertAlmostEqual(logits[0, 1], -logits[1, 0])
        self.assertAlmostEqual(logits[0, 0], -logits[1, 1])
        self.assertEqual(
            logits[1, 2], compact_modeling.probabilities_to_logits(np.array([0.0]))[0]
        )

    def test_invalid_probabilities_or_clip_are_rejected(self) -> None:
        for invalid in ([-0.1], [1.1], [np.nan], [np.inf]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                compact_modeling.probability_to_logit(invalid)
        with self.assertRaisesRegex(ValueError, "non-empty array"):
            compact_modeling.probability_to_logit(0.5)
        for invalid_clip in (0.0, 0.5, np.nan, True):
            with self.subTest(clip=invalid_clip), self.assertRaises((TypeError, ValueError)):
                compact_modeling.probability_to_logit([0.5], clip=invalid_clip)


class FixedCLogisticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.default_rng(44)
        cls.features = rng.normal(size=(80, 7))
        score = 1.2 * cls.features[:, 0] - 0.8 * cls.features[:, 2] + rng.normal(
            scale=0.5, size=80
        )
        cls.labels = (score > np.median(score)).astype(np.int64)
        cls.heldout = rng.normal(loc=20.0, size=(9, 7))

    def test_scaler_and_fixed_c_fit_only_use_train_rows(self) -> None:
        fit = compact_modeling.fit_fixed_c_logistic(
            self.features, self.labels, 0.25
        )
        duplicate = compact_modeling.fit_fixed_c_binary_logistic(
            self.features.copy(), self.labels.copy(), 0.25
        )
        self.assertEqual(fit.c_value, 0.25)
        self.assertEqual(fit.parameter_sha256, duplicate.parameter_sha256)
        np.testing.assert_allclose(fit.scaler.mean_, self.features.mean(axis=0))
        self.assertFalse(np.allclose(fit.scaler.mean_, self.heldout.mean(axis=0)))
        probabilities = fit.predict_proba(self.heldout)
        self.assertEqual(probabilities.shape, (9,))
        self.assertTrue(np.all((probabilities >= 0.0) & (probabilities <= 1.0)))
        np.testing.assert_allclose(
            fit.predict_logit(self.heldout), fit.decision_function(self.heldout), atol=1e-8
        )

    def test_fitted_parameters_are_read_only(self) -> None:
        fit = compact_modeling.fit_fixed_c_logistic(self.features, self.labels, 1.0)
        for array in (
            fit.scaler.mean_,
            fit.scaler.var_,
            fit.scaler.scale_,
            fit.model.coef_,
            fit.model.intercept_,
            fit.model.classes_,
            fit.model.n_iter_,
        ):
            self.assertFalse(array.flags.writeable)

    def test_invalid_logistic_contract_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "both binary classes"):
            compact_modeling.fit_fixed_c_logistic(
                self.features, np.zeros(80, dtype=int), 1.0
            )
        for invalid_c in (0.0, -1.0, np.inf, True):
            with self.subTest(c=invalid_c), self.assertRaises((TypeError, ValueError)):
                compact_modeling.fit_fixed_c_logistic(
                    self.features, self.labels, invalid_c
                )
        with self.assertRaisesRegex(ValueError, "solver='liblinear'"):
            compact_modeling.fit_fixed_c_logistic(
                self.features, self.labels, 1.0, solver="lbfgs"
            )
        with self.assertRaisesRegex(ValueError, "class_weight"):
            compact_modeling.fit_fixed_c_logistic(
                self.features, self.labels, 1.0, class_weight="auto"
            )
        with self.assertRaisesRegex(ValueError, "keys must be binary integers"):
            compact_modeling.fit_fixed_c_logistic(
                self.features, self.labels, 1.0, class_weight={0.5: 1.0, 1: 1.0}
            )


class InnerFoldAssignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ids = np.array([f"P{index:03d}" for index in range(60)], dtype=object)
        # Labels deliberately do not form contiguous ID blocks.
        cls.labels = np.array([(index * 7) % 11 < 5 for index in range(60)], dtype=int)

    def test_patient_mapping_is_stable_under_input_permutation(self) -> None:
        first = compact_modeling.stratified_inner_assignments(
            self.ids, self.labels, n_splits=5, seed=260813
        )
        permutation = np.random.default_rng(812).permutation(len(self.ids))
        second = compact_modeling.stable_patient_stratified_folds(
            self.ids[permutation], self.labels[permutation], n_splits=5, seed=260813
        )
        first_map = dict(zip(first.patient_ids, first.fold_by_row.tolist()))
        second_map = dict(zip(second.patient_ids, second.fold_by_row.tolist()))
        self.assertEqual(first_map, second_map)
        self.assertEqual(first.assignment_sha256, second.assignment_sha256)
        self.assertFalse(first.fold_by_row.flags.writeable)

        for fold in range(5):
            train, validation = first.indices(fold)
            self.assertEqual(np.intersect1d(train, validation).size, 0)
            np.testing.assert_array_equal(
                np.sort(np.concatenate([train, validation])), np.arange(len(self.ids))
            )
            self.assertEqual(set(self.labels[validation]), {0, 1})
            validation_ids = np.asarray(first.patient_ids)[validation]
            self.assertEqual(validation_ids.tolist(), sorted(validation_ids.tolist()))
            for row in validation:
                self.assertEqual(first.fold_for(self.ids[row]), fold)

    def test_assignment_input_invariants(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            compact_modeling.stratified_inner_assignments(
                ["P0", "P0", "P1", "P2"], [0, 1, 0, 1], n_splits=2
            )
        with self.assertRaisesRegex(ValueError, "at least n_splits"):
            compact_modeling.stratified_inner_assignments(
                [f"P{i}" for i in range(8)], [0, 0, 0, 1, 1, 1, 1, 1], n_splits=4
            )
        with self.assertRaisesRegex(TypeError, "not boolean"):
            compact_modeling.stratified_inner_assignments(
                [f"P{i}" for i in range(10)], [False, True] * 5, n_splits=5
            )


class StrictInnerOOFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.default_rng(512)
        cls.ids = np.array([f"S{index:03d}" for index in range(75)], dtype=object)
        cls.features = rng.normal(size=(75, 5))
        score = cls.features[:, 0] - 0.4 * cls.features[:, 1] + rng.normal(
            scale=0.8, size=75
        )
        cls.labels = (score > np.median(score)).astype(np.int64)

    def test_generic_callback_gets_disjoint_folds_and_exact_coverage(self) -> None:
        assignments = compact_modeling.stratified_inner_assignments(
            self.ids, self.labels, n_splits=5, seed=19
        )
        seen = np.zeros(len(self.ids), dtype=int)

        def callback(train: np.ndarray, validation: np.ndarray) -> np.ndarray:
            self.assertEqual(np.intersect1d(train, validation).size, 0)
            seen[validation] += 1
            return 0.2 + 0.6 * self.labels[validation]

        result = compact_modeling.strict_inner_oof_probabilities(assignments, callback)
        np.testing.assert_array_equal(seen, np.ones(len(self.ids), dtype=int))
        np.testing.assert_allclose(result.probabilities, 0.2 + 0.6 * self.labels)
        np.testing.assert_array_equal(result.inner_fold, assignments.fold_by_row)
        self.assertTrue(np.isfinite(result.logits).all())
        self.assertFalse(result.probabilities.flags.writeable)
        self.assertFalse(result.inner_fold.flags.writeable)

    def test_callback_output_contract_is_enforced(self) -> None:
        assignments = compact_modeling.stratified_inner_assignments(
            self.ids, self.labels, n_splits=5
        )
        with self.assertRaisesRegex(ValueError, "returned shape"):
            compact_modeling.strict_inner_oof_probabilities(
                assignments, lambda train, validation: np.array([0.5])
            )
        with self.assertRaisesRegex(ValueError, "invalid probabilities"):
            compact_modeling.strict_inner_oof_probabilities(
                assignments, lambda train, validation: np.full(len(validation), np.nan)
            )

    def test_fixed_c_oof_is_deterministic_and_row_order_invariant(self) -> None:
        first = compact_modeling.fixed_c_inner_oof_probabilities(
            self.features, self.labels, self.ids, 0.5, n_splits=5, seed=260813
        )
        permutation = np.random.default_rng(17).permutation(len(self.ids))
        second = compact_modeling.fixed_c_inner_oof_probabilities(
            self.features[permutation],
            self.labels[permutation],
            self.ids[permutation],
            0.5,
            n_splits=5,
            seed=260813,
        )
        first_map = dict(zip(first.patient_ids, first.probabilities.tolist()))
        second_map = dict(zip(second.patient_ids, second.probabilities.tolist()))
        self.assertEqual(first_map, second_map)
        self.assertEqual(first.assignment_sha256, second.assignment_sha256)
        self.assertEqual(first.prediction_sha256, second.prediction_sha256)
        self.assertEqual(first.fold_fit_sha256, second.fold_fit_sha256)
        self.assertEqual(len(first.fold_fit_sha256), 5)
        self.assertTrue(all(len(value) == 64 for value in first.fold_fit_sha256))


if __name__ == "__main__":
    unittest.main()

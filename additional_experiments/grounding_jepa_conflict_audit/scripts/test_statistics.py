#!/usr/bin/env python3
"""Grounding–JEPA 统计合同的纯 synthetic deterministic self-test。"""

from __future__ import annotations

import inspect
import json
import sys
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path
from unittest import mock

import numpy as np
from scipy.stats import spearmanr as scipy_spearmanr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gjca.statistics import (  # noqa: E402
    NonFiniteInputError,
    StatisticsInputError,
    crossed_group_bootstrap_contrasts,
    exact_crossed_group_permutation,
    exact_crossed_spearman_permutation,
    generate_resampling_indices,
    group_bootstrap_contrasts,
    holm_adjust,
    index_array_sha256,
    load_resampling_indices_npz,
    save_resampling_indices_npz,
    spearman_crossed_ci,
    two_sided_permutation_test,
)


class StatisticsContractTests(unittest.TestCase):
    def test_resampling_indices_are_deterministic_immutable_and_hashed(self) -> None:
        kwargs = dict(
            crossed_replicates=7,
            n_seeds=3,
            n_folds=2,
            crossed_seed=101,
        )
        first = generate_resampling_indices(**kwargs)
        second = generate_resampling_indices(**kwargs)
        signature = inspect.signature(generate_resampling_indices)
        self.assertEqual(
            list(signature.parameters),
            ["crossed_replicates", "n_seeds", "n_folds", "crossed_seed"],
        )
        self.assertTrue(
            all(
                parameter.kind is inspect.Parameter.KEYWORD_ONLY
                for parameter in signature.parameters.values()
            )
        )
        self.assertEqual(
            [field.name for field in fields(first)],
            [
                "crossed_seed_draws",
                "crossed_fold_draws",
                "crossed_permutation_seed_orders",
                "crossed_permutation_fold_orders",
                "crossed_seed_rng_seed",
            ],
        )
        self.assertEqual(first.manifest(), second.manifest())
        for name in (
            "crossed_seed_draws",
            "crossed_fold_draws",
            "crossed_permutation_seed_orders",
            "crossed_permutation_fold_orders",
        ):
            self.assertTrue(np.array_equal(getattr(first, name), getattr(second, name)))
            self.assertFalse(getattr(first, name).flags.writeable)
        for removed_name in (
            "group_pass_draws",
            "group_fail_draws",
            "permutation_orders",
            "permutation_fail_labels",
            "group_rng_seed",
            "permutation_rng_seed",
        ):
            self.assertFalse(hasattr(first, removed_name))
        manifest = first.manifest()
        self.assertEqual(
            set(manifest["arrays"]),
            {
                "crossed_seed_draws",
                "crossed_fold_draws",
                "crossed_permutation_seed_orders",
                "crossed_permutation_fold_orders",
            },
        )
        self.assertEqual(manifest["rng_seeds"], {"crossed": 101})
        self.assertEqual(
            set(manifest["algorithms"]),
            {"crossed_bootstrap", "crossed_exact_permutation"},
        )
        self.assertEqual(manifest["arrays"]["crossed_seed_draws"]["shape"], [7, 3])
        self.assertEqual(
            manifest["arrays"]["crossed_permutation_seed_orders"]["shape"],
            [12, 3],
        )
        self.assertEqual(
            manifest["arrays"]["crossed_permutation_fold_orders"]["shape"],
            [12, 2],
        )
        self.assertEqual(manifest["arrays"]["crossed_seed_draws"]["dtype"], "|u1")
        self.assertIn("raw_sha256", manifest["arrays"]["crossed_seed_draws"])
        self.assertTrue(
            manifest["algorithms"]["crossed_exact_permutation"]["formal_gate_eligible"]
        )
        # 固定 PCG64 小样本的 golden SHA，防止抽样顺序或编码静默改变。
        self.assertEqual(
            manifest["bundle_sha256"],
            "429d4309deb0717a7985d6c9d0e5f3f6c66e6f63b8d9f2a59a61e19e8575680c",
        )
        changed = first.crossed_seed_draws.copy()
        changed[0, 0] = (changed[0, 0] + 1) % 3
        self.assertNotEqual(
            index_array_sha256(first.crossed_seed_draws), index_array_sha256(changed)
        )
        with self.assertRaises(TypeError):
            generate_resampling_indices(crossed_replicates=1, group_replicates=1)

    def test_npz_roundtrip_four_arrays_and_tamper_detection(self) -> None:
        indices = generate_resampling_indices(
            crossed_replicates=7,
            n_seeds=3,
            n_folds=2,
            crossed_seed=101,
        )
        with tempfile.TemporaryDirectory(prefix="gjca_statistics_test_") as directory:
            path = Path(directory) / "resampling_indices.npz"
            saved_manifest = save_resampling_indices_npz(path, indices)
            self.assertTrue(path.is_file())
            self.assertEqual(len(saved_manifest["npz_sha256"]), 64)
            with self.assertRaises(FileExistsError):
                save_resampling_indices_npz(path, indices)
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(archive["crossed_seed_draws"].dtype, np.uint8)
                self.assertEqual(
                    archive["crossed_permutation_seed_orders"].dtype, np.uint8
                )
                self.assertEqual(
                    set(archive.files),
                    {
                        "crossed_seed_draws",
                        "crossed_fold_draws",
                        "crossed_permutation_seed_orders",
                        "crossed_permutation_fold_orders",
                        "manifest_json",
                    },
                )
                archived = {name: archive[name].copy() for name in archive.files}

            loaded, loaded_manifest = load_resampling_indices_npz(path)
            self.assertEqual(saved_manifest, loaded_manifest)
            self.assertEqual(indices.manifest(), loaded.manifest())
            for name in indices.manifest()["formal_index_arrays"]:
                self.assertTrue(
                    np.array_equal(getattr(indices, name), getattr(loaded, name))
                )

            legacy_member = Path(directory) / "legacy_member.npz"
            archive_with_legacy = dict(archived)
            archive_with_legacy["group_pass_draws"] = np.zeros((1, 1), dtype=np.uint8)
            np.savez_compressed(legacy_member, **archive_with_legacy)
            with self.assertRaises(StatisticsInputError):
                load_resampling_indices_npz(legacy_member)

            tampered = Path(directory) / "tampered.npz"
            archived["crossed_permutation_seed_orders"][0, 0] = (
                int(archived["crossed_permutation_seed_orders"][0, 0]) + 1
            ) % 3
            np.savez_compressed(tampered, **archived)
            with self.assertRaises(StatisticsInputError):
                load_resampling_indices_npz(tampered)

    def test_formal_crossed_permutation_bundle_has_all_14400_orders(self) -> None:
        indices = generate_resampling_indices(crossed_replicates=1)
        seeds = indices.crossed_permutation_seed_orders
        folds = indices.crossed_permutation_fold_orders
        self.assertEqual(seeds.shape, (14400, 5))
        self.assertEqual(folds.shape, (14400, 5))
        self.assertEqual(
            np.unique(np.concatenate((seeds, folds), axis=1), axis=0).shape[0],
            14400,
        )
        self.assertTrue(np.array_equal(seeds[0], np.arange(5)))
        self.assertTrue(np.array_equal(folds[0], np.arange(5)))
        algorithm = indices.manifest()["algorithms"]["crossed_exact_permutation"]
        self.assertEqual(algorithm["replicates"], 14400)
        self.assertTrue(algorithm["includes_identity"])
        x = np.arange(25, dtype=float).reshape(5, 5)
        y = np.roll(x, shift=7).reshape(5, 5)
        with mock.patch(
            "gjca.statistics.spearmanr",
            side_effect=AssertionError("exact Spearman 不应调用 scipy spearmanr"),
        ):
            result = exact_crossed_spearman_permutation(x, y, seeds, folds)
        self.assertEqual(result.replicates, 14400)
        self.assertEqual(result.status, "ok")
        self.assertGreaterEqual(result.p_value, 0.0)
        self.assertLessEqual(result.p_value, 1.0)

    def test_crossed_spearman_reuses_cartesian_multiplicity(self) -> None:
        x = np.arange(25, dtype=float).reshape(5, 5)
        y = 3.0 * x + 7.0
        indices = generate_resampling_indices(
            crossed_replicates=128,
            crossed_seed=211,
        )
        result = spearman_crossed_ci(
            x,
            y,
            indices.crossed_seed_draws,
            indices.crossed_fold_draws,
        )
        self.assertEqual(result.status, "ok")
        self.assertAlmostEqual(result.estimate, 1.0)
        self.assertAlmostEqual(result.ci_low, 1.0)
        self.assertAlmostEqual(result.ci_high, 1.0)
        self.assertEqual(result.bootstrap_finite, 128)
        self.assertEqual(
            result.iid_p_sensitivity_only_not_formal_gate,
            result.p_raw,
        )
        self.assertEqual(result.p_raw, 0.0)
        self.assertIn(
            "iid_p_sensitivity_only_not_formal_gate",
            [field.name for field in fields(result)],
        )
        self.assertNotIn("p_raw", [field.name for field in fields(result)])

    def test_crossed_nonfinite_and_constant_policies(self) -> None:
        seeds = np.array([[0, 0], [1, 1]], dtype=np.int64)
        folds = np.array([[0, 0], [1, 1]], dtype=np.int64)
        constant = spearman_crossed_ci(
            np.ones((2, 2)), np.arange(4, dtype=float).reshape(2, 2), seeds, folds
        )
        self.assertEqual(constant.status, "constant_input")
        self.assertIsNone(constant.estimate)
        with self.assertRaises(NonFiniteInputError):
            spearman_crossed_ci(
                np.array([[0.0, np.nan], [1.0, 2.0]]),
                np.arange(4, dtype=float).reshape(2, 2),
                seeds,
                folds,
            )

        # 点估计可定义，但两个 replicate 都只抽到单一 cell；不重抽且 CI 不可用。
        degenerate_seeds = np.array([[0, 0], [1, 1]], dtype=np.int64)
        degenerate_folds = np.array([[0, 0], [1, 1]], dtype=np.int64)
        insufficient = spearman_crossed_ci(
            np.array([[0.0, 1.0], [2.0, 3.0]]),
            np.array([[0.0, 2.0], [1.0, 4.0]]),
            degenerate_seeds,
            degenerate_folds,
        )
        self.assertEqual(insufficient.status, "insufficient_finite_bootstrap")
        self.assertEqual(insufficient.bootstrap_finite, 0)
        self.assertIsNone(insufficient.ci_low)

    def test_exact_crossed_spearman_known_p_and_constant_policy(self) -> None:
        indices = generate_resampling_indices(
            crossed_replicates=1,
            n_seeds=2,
            n_folds=2,
        )
        x = np.arange(4, dtype=float).reshape(2, 2)
        result = exact_crossed_spearman_permutation(
            x,
            x,
            indices.crossed_permutation_seed_orders,
            indices.crossed_permutation_fold_orders,
        )
        self.assertAlmostEqual(result.estimate, 1.0)
        self.assertEqual(result.replicates, 4)
        self.assertEqual(result.extreme_count, 2)
        self.assertAlmostEqual(result.p_value, 0.5)
        self.assertTrue(result.includes_identity)

        constant = exact_crossed_spearman_permutation(
            np.ones((2, 2)),
            x,
            indices.crossed_permutation_seed_orders,
            indices.crossed_permutation_fold_orders,
        )
        self.assertEqual(constant.status, "constant_input")
        self.assertIsNone(constant.p_value)

    def test_vectorized_exact_statistics_match_scalar_references(self) -> None:
        indices = generate_resampling_indices(
            crossed_replicates=1,
            n_seeds=3,
            n_folds=2,
        )
        seeds = indices.crossed_permutation_seed_orders
        folds = indices.crossed_permutation_fold_orders
        x = np.array([[0.0, 0.0], [1.0, 2.0], [4.0, 3.0]])
        y = np.array([[4.0, 1.0], [2.0, 2.0], [0.0, 3.0]])
        scalar_spearman = np.array(
            [
                scipy_spearmanr(
                    x.ravel(), y[np.ix_(seed_order, fold_order)].ravel()
                ).statistic
                for seed_order, fold_order in zip(seeds, folds, strict=True)
            ],
            dtype=np.float64,
        )
        scalar_observed = float(scipy_spearmanr(x.ravel(), y.ravel()).statistic)
        scalar_extreme = int(
            np.count_nonzero(np.abs(scalar_spearman) >= abs(scalar_observed))
        )
        with mock.patch(
            "gjca.statistics.spearmanr",
            side_effect=AssertionError("exact Spearman 不应调用 scipy spearmanr"),
        ):
            spearman = exact_crossed_spearman_permutation(x, y, seeds, folds)
        self.assertAlmostEqual(spearman.estimate, scalar_observed, places=14)
        self.assertEqual(spearman.extreme_count, scalar_extreme)
        self.assertAlmostEqual(spearman.p_value, scalar_extreme / seeds.shape[0])
        reversed_spearman = exact_crossed_spearman_permutation(
            x,
            y,
            seeds[::-1],
            folds[::-1],
        )
        self.assertEqual(reversed_spearman, spearman)

        passed = np.array([[True, False], [False, True], [False, False]])
        for contrast in ("mean", "median"):
            reducer = np.mean if contrast == "mean" else np.median
            scalar_group = []
            for seed_order, fold_order in zip(seeds, folds, strict=True):
                permuted = y[np.ix_(seed_order, fold_order)]
                scalar_group.append(
                    reducer(permuted[~passed]) - reducer(permuted[passed])
                )
            scalar_group = np.asarray(scalar_group, dtype=np.float64)
            observed = float(reducer(y[~passed]) - reducer(y[passed]))
            extreme = int(np.count_nonzero(np.abs(scalar_group) >= abs(observed)))
            result = exact_crossed_group_permutation(
                y,
                passed,
                seeds,
                folds,
                contrast=contrast,
            )
            self.assertAlmostEqual(result.estimate, observed)
            self.assertEqual(result.extreme_count, extreme)
            self.assertAlmostEqual(result.p_value, extreme / seeds.shape[0])
            reversed_result = exact_crossed_group_permutation(
                y,
                passed,
                seeds[::-1],
                folds[::-1],
                contrast=contrast,
            )
            self.assertEqual(reversed_result, result)

    def test_exact_crossed_group_mean_and_median_known_p(self) -> None:
        indices = generate_resampling_indices(
            crossed_replicates=1,
            n_seeds=2,
            n_folds=2,
        )
        values = np.arange(4, dtype=float).reshape(2, 2)
        passed = np.array([[True, False], [False, False]])
        for contrast in ("mean", "median"):
            result = exact_crossed_group_permutation(
                values,
                passed,
                indices.crossed_permutation_seed_orders,
                indices.crossed_permutation_fold_orders,
                contrast=contrast,
            )
            self.assertAlmostEqual(result.estimate, 2.0)
            self.assertEqual(result.replicates, 4)
            self.assertEqual(result.extreme_count, 2)
            self.assertAlmostEqual(result.p_value, 0.5)
        with self.assertRaises(StatisticsInputError):
            exact_crossed_group_permutation(
                values,
                passed,
                np.array([[0, 1]], dtype=np.int64),
                np.array([[0, 1]], dtype=np.int64),
                contrast="mean",
            )
        duplicate_seeds = indices.crossed_permutation_seed_orders.copy()
        duplicate_folds = indices.crossed_permutation_fold_orders.copy()
        duplicate_seeds[-1] = duplicate_seeds[0]
        duplicate_folds[-1] = duplicate_folds[0]
        with self.assertRaises(StatisticsInputError):
            exact_crossed_group_permutation(
                values,
                passed,
                duplicate_seeds,
                duplicate_folds,
                contrast="mean",
            )

    def test_crossed_group_bootstrap_missing_groups_are_not_redrawn(self) -> None:
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        passed = np.array([[True, False], [False, False]])
        seed_draws = np.array([[0, 1], [0, 0], [1, 1], [0, 1]], dtype=np.int64)
        fold_draws = np.array([[0, 1], [0, 0], [1, 1], [0, 0]], dtype=np.int64)
        strict = crossed_group_bootstrap_contrasts(
            values,
            passed,
            seed_draws,
            fold_draws,
            confidence_level=0.50,
            minimum_finite_fraction=0.75,
        )
        self.assertEqual(strict.mean_difference.bootstrap_requested, 4)
        self.assertEqual(strict.mean_difference.bootstrap_finite, 2)
        self.assertEqual(strict.median_difference.bootstrap_finite, 2)
        self.assertEqual(strict.median_ratio.bootstrap_finite, 2)
        self.assertEqual(strict.mean_difference.status, "insufficient_finite_bootstrap")
        self.assertIsNone(strict.mean_difference.ci_low)

        permissive = crossed_group_bootstrap_contrasts(
            values,
            passed,
            seed_draws,
            fold_draws,
            confidence_level=0.50,
            minimum_finite_fraction=0.50,
        )
        self.assertEqual(permissive.mean_difference.status, "ok")
        self.assertAlmostEqual(permissive.mean_difference.ci_low, 2.0)
        self.assertAlmostEqual(permissive.mean_difference.ci_high, 2.0)
        self.assertAlmostEqual(permissive.median_ratio.ci_low, 3.0)
        self.assertAlmostEqual(permissive.median_ratio.ci_high, 3.0)

    def test_crossed_group_bootstrap_nonpositive_ratio_denominator_accounting(
        self,
    ) -> None:
        values = np.array([[0.0, 2.0], [3.0, 4.0]])
        passed = np.array([[True, True], [False, False]])
        seed_draws = np.array([[0, 1], [0, 1], [0, 1], [1, 1]], dtype=np.int64)
        fold_draws = np.array([[0, 1], [0, 0], [1, 1], [0, 1]], dtype=np.int64)
        result = crossed_group_bootstrap_contrasts(
            values,
            passed,
            seed_draws,
            fold_draws,
            minimum_finite_fraction=0.50,
        )
        self.assertEqual(result.mean_difference.bootstrap_finite, 3)
        self.assertEqual(result.median_difference.bootstrap_finite, 3)
        self.assertEqual(result.median_ratio.bootstrap_finite, 2)
        self.assertEqual(result.median_ratio.status, "ok")

    def test_group_bootstrap_known_percentiles(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0])
        pass_mask = np.array([True, True, False, False])
        pass_draws = np.array([[0, 1], [0, 0], [1, 1], [0, 1]], dtype=np.int64)
        fail_draws = np.array([[0, 1], [0, 0], [1, 1], [1, 0]], dtype=np.int64)
        result = group_bootstrap_contrasts(
            values,
            pass_mask,
            pass_draws,
            fail_draws,
            confidence_level=0.50,
        )
        self.assertAlmostEqual(result.mean_difference.estimate, 2.0)
        self.assertAlmostEqual(result.mean_difference.ci_low, 2.0)
        self.assertAlmostEqual(result.mean_difference.ci_high, 2.0)
        self.assertAlmostEqual(result.median_difference.estimate, 2.0)
        self.assertAlmostEqual(result.median_ratio.estimate, 3.5 / 1.5)
        self.assertAlmostEqual(result.median_ratio.ci_low, 2.25)
        self.assertAlmostEqual(result.median_ratio.ci_high, 2.5)

    def test_group_ratio_and_nonfinite_policies(self) -> None:
        draws = np.array([[0, 1], [1, 0]], dtype=np.int64)
        ratio_invalid = group_bootstrap_contrasts(
            np.array([-2.0, -1.0, 3.0, 4.0]),
            np.array([True, True, False, False]),
            draws,
            draws,
        )
        self.assertEqual(ratio_invalid.median_ratio.status, "invalid_ratio_denominator")
        self.assertEqual(ratio_invalid.mean_difference.status, "ok")
        with self.assertRaises(NonFiniteInputError):
            group_bootstrap_contrasts(
                np.array([1.0, np.inf, 3.0, 4.0]),
                np.array([True, True, False, False]),
                draws,
                draws,
            )

    def test_two_sided_plus_one_permutation(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0])
        pass_mask = np.array([True, True, False, False])
        orders = np.array([[0, 1, 2, 3], [0, 2, 1, 3], [2, 3, 0, 1]], dtype=np.int64)
        result = two_sided_permutation_test(values, pass_mask, orders, contrast="mean")
        self.assertAlmostEqual(result.estimate, 2.0)
        self.assertEqual(result.extreme_count, 2)
        self.assertAlmostEqual(result.p_value, 0.75)
        self.assertEqual(result.replicates, 3)
        self.assertEqual(result.status, "compatibility_only_not_formal_gate")

    def test_permutation_rejects_malformed_rows(self) -> None:
        with self.assertRaises(StatisticsInputError):
            two_sided_permutation_test(
                np.arange(4, dtype=float),
                np.array([True, True, False, False]),
                np.array([[0, 0, 2, 3]], dtype=np.int64),
                contrast="median",
            )

    def test_holm_na_as_one_and_endpoint_tie_order(self) -> None:
        adjusted = holm_adjust(
            {"b": 0.01, "a": 0.01, "c": 0.04, "d_missing": None, "e_nan": np.nan}
        )
        self.assertEqual(adjusted["a"].rank, 1)
        self.assertEqual(adjusted["b"].rank, 2)
        self.assertAlmostEqual(adjusted["a"].p_holm, 0.05)
        self.assertAlmostEqual(adjusted["b"].p_holm, 0.05)
        self.assertAlmostEqual(adjusted["c"].p_holm, 0.12)
        self.assertEqual(adjusted["d_missing"].p_effective, 1.0)
        self.assertEqual(adjusted["d_missing"].p_holm, 1.0)
        self.assertEqual(adjusted["e_nan"].status, "unavailable_substituted_one")
        with self.assertRaises(StatisticsInputError):
            holm_adjust({"invalid": -0.01})


def main() -> None:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(StatisticsContractTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(
        json.dumps(
            {
                "status": "ok" if result.wasSuccessful() else "failed",
                "tests_run": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "formal_assets_read": False,
                "formal_outputs_written": False,
            },
            ensure_ascii=False,
        )
    )
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()

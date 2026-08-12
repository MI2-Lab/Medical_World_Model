from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


sys.dont_write_bytecode = True
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = EXPERIMENT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lg_response_pilot import analysis, figures  # noqa: E402


CONFIG_PATH = EXPERIMENT_ROOT / "configs" / "confirmation.json"


def confirmation_config() -> dict[str, object]:
    return analysis.load_confirmation_config(CONFIG_PATH)


def macro_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    static_local_effect = (0.11, 0.12, 0.13, 0.14, 0.09)
    delta_local_effect = (0.01, 0.02, 0.03, 0.04, -0.01)
    r2_local_effect = (0.01, 0.02, 0.03, -0.01, -0.02)
    grounding_static_effect = (0.01, 0.02, 0.03, 0.04, -0.01)
    grounding_delta_effect = (0.02, 0.01, 0.03, 0.04, -0.01)
    static_rows: list[dict[str, object]] = []
    delta_rows: list[dict[str, object]] = []
    for index, seed in enumerate(analysis.CANONICAL_SEEDS):
        static_spearman = {
            "GAP0": 0.20,
            "GAP3": 0.21,
            "LOCAL0": 0.20 + static_local_effect[index],
            "LOCAL3": (
                0.20 + static_local_effect[index] + grounding_static_effect[index]
            ),
        }
        static_r2 = {
            "GAP0": 0.02,
            "GAP3": 0.02,
            "LOCAL0": 0.02 + r2_local_effect[index],
            "LOCAL3": 0.03 + r2_local_effect[index],
        }
        delta_spearman = {
            "GAP0": 0.10,
            "GAP3": 0.11,
            "LOCAL0": 0.10 + delta_local_effect[index],
            "LOCAL3": (
                0.10 + delta_local_effect[index] + grounding_delta_effect[index]
            ),
        }
        for arm in analysis.CANONICAL_ARMS:
            common = {
                "seed_base": seed,
                "arm": arm,
                "endpoint": "macro",
                "analysis_scope": analysis.PRIMARY_SCOPE,
                "natural_prediction_target_variance_ratio": 0.5,
                "natural_calibration_slope": 0.6,
                "natural_calibration_intercept": 0.0,
                "natural_calibration_mean_bias": 0.0,
                "natural_n_test": 100,
            }
            static_rows.append(
                {
                    **common,
                    "task": "static",
                    "natural_spearman": static_spearman[arm],
                    "natural_r2": static_r2[arm],
                }
            )
            delta_rows.append(
                {
                    **common,
                    "task": "delta",
                    "natural_spearman": delta_spearman[arm],
                    "natural_r2": 0.01,
                }
            )
    return pd.DataFrame(static_rows), pd.DataFrame(delta_rows)


def safety_table(local_passes: int = 23) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grid = [
        (seed, fold)
        for seed in analysis.CANONICAL_SEEDS
        for fold in analysis.CANONICAL_FOLDS
    ]
    for comparison in ("GAP3-GAP0", "LOCAL3-LOCAL0"):
        passes = 20 if comparison == "GAP3-GAP0" else local_passes
        for index, (seed, fold) in enumerate(grid):
            passed = index < passes
            baseline_loss = 1.0
            grounded_loss = 1.04 if passed else 1.06
            grounded_mode = "primary" if passed else "fallback_base_gate_failed"
            rows.append(
                {
                    "seed_base": seed,
                    "fold": fold,
                    "comparison": comparison,
                    "baseline_selected_validation_state_loss": baseline_loss,
                    "grounded_selected_validation_state_loss": grounded_loss,
                    "allowed_grounded_validation_state_loss": 1.05 * baseline_loss,
                    "state_loss_degradation_fraction": grounded_loss / baseline_loss
                    - 1.0,
                    "maximum_allowed_degradation_fraction": 0.05,
                    "baseline_selection_mode": "primary",
                    "baseline_experiment_pass": True,
                    "grounded_selection_mode": grounded_mode,
                    "grounded_experiment_pass": passed,
                    "grounded_primary_selection": passed,
                    "exact_state_loss_rule_pass": passed,
                    "safety_pass": passed,
                }
            )
    return pd.DataFrame(rows)


class ConfirmationStatisticsTest(unittest.TestCase):
    def test_frozen_four_arm_five_seed_config_is_fail_closed(self) -> None:
        config = confirmation_config()
        self.assertEqual(tuple(config["arms"]), analysis.CANONICAL_ARMS)
        self.assertEqual(
            tuple(config["training"]["seed_bases"]), analysis.CANONICAL_SEEDS
        )
        self.assertEqual(config["training"]["formal_cells"], 100)
        changed = copy.deepcopy(config)
        changed["gates"]["LOCAL_CONFIRMATION"][
            "local0_gap0_static_macro_spearman_seed_effect_strictly_gt"
        ] = 0.099
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "confirmation.json"
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "confirmation gate"):
                analysis.load_confirmation_config(path)

    def test_five_fold_oof_is_pooled_separately_for_each_seed(self) -> None:
        config = confirmation_config()
        rows: list[dict[str, object]] = []
        for seed_index, seed in enumerate(analysis.CANONICAL_SEEDS[:2]):
            for endpoint_index, endpoint in enumerate(("T0", "T1", "T2", "T3")):
                for fold in analysis.CANONICAL_FOLDS:
                    truth = float(fold + endpoint_index)
                    rows.append(
                        {
                            "patient_id": f"p_{seed}_{endpoint}_{fold}",
                            "seed_base": seed,
                            "arm": "LOCAL0",
                            "fold": fold,
                            "analysis_scope": analysis.PRIMARY_SCOPE,
                            "task": "static",
                            "endpoint": endpoint,
                            "target_semantics": "static",
                            "y_true": truth,
                            "y_pred": 0.8 * truth + 0.1 * seed_index,
                            "b0_prediction": float(endpoint_index),
                        }
                    )
        result = analysis.pooled_natural_metrics(pd.DataFrame(rows), config)
        macros = result.loc[result["endpoint"].eq("macro")]
        self.assertEqual(set(macros["seed_base"]), set(analysis.CANONICAL_SEEDS[:2]))
        self.assertTrue(
            macros["aggregation"].eq("unweighted_mean_of_pooled_endpoint_metrics").all()
        )
        self.assertTrue(macros["n_test"].eq(20).all())

    def test_seed_summary_uses_sample_sd_and_frozen_deterministic_bootstrap(
        self,
    ) -> None:
        values = np.asarray([0.01, 0.02, 0.03, 0.04, -0.01])
        first = analysis.summarize_seed_values(values)
        second = analysis.summarize_seed_values(values)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["sample_sd"], float(np.std(values, ddof=1)))
        self.assertEqual(first["direction_count"], 4)
        rng = np.random.default_rng(20_260_811)
        indices = rng.integers(0, 5, size=(10_000, 5))
        expected = np.percentile(values[indices].mean(axis=1), (2.5, 97.5))
        self.assertAlmostEqual(first["bootstrap_ci_lower"], expected[0])
        self.assertAlmostEqual(first["bootstrap_ci_upper"], expected[1])

    def test_exact_confirmation_gate_and_strict_thresholds(self) -> None:
        config = confirmation_config()
        table2, table3 = macro_tables()
        decision = analysis.evaluate_gates(table2, table3, safety_table(), config)
        self.assertTrue(decision["confirmed"])
        self.assertEqual(decision["classification"], "LOCAL_MULTISEED_CONFIRMED")
        gate = decision["gates"]["LOCAL_CONFIRMATION"]
        self.assertEqual(gate["optimization_safety"]["safe_folds"], 23)
        self.assertEqual(
            gate["seed_summaries"]["local0_gap0_delta_macro_spearman"][
                "direction_count"
            ],
            4,
        )

        equality = table2.copy()
        seed = analysis.CANONICAL_SEEDS[0]
        equality.loc[
            equality["seed_base"].eq(seed) & equality["arm"].eq("GAP0"),
            "natural_spearman",
        ] = 0.0
        equality.loc[
            equality["seed_base"].eq(seed) & equality["arm"].eq("LOCAL0"),
            "natural_spearman",
        ] = 0.1
        equality.loc[
            equality["seed_base"].eq(seed) & equality["arm"].eq("LOCAL3"),
            "natural_spearman",
        ] = 0.11
        strict = analysis.evaluate_gates(equality, table3, safety_table(), config)
        self.assertFalse(
            strict["gates"]["LOCAL_CONFIRMATION"]["checks"][
                "local0_gap0_static_effect_gt_0_10_in_at_least_4_seeds"
            ]
        )
        self.assertEqual(strict["classification"], "LOCAL_MULTISEED_NOT_CONFIRMED")
        unsafe = analysis.evaluate_gates(table2, table3, safety_table(22), config)
        self.assertFalse(unsafe["confirmed"])

    def test_safety_rejects_fallback_and_tiny_exact_threshold_violation(self) -> None:
        config = copy.deepcopy(confirmation_config())
        config["training"]["seed_bases"] = [2026]
        config["training"]["folds"] = [0]

        def selections(
            local3_loss: float, local3_mode: str, local3_experiment_pass: bool
        ) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {
                        "seed_base": 2026,
                        "fold": 0,
                        "arm": arm,
                        "selected_validation_state_loss": (
                            local3_loss
                            if arm == "LOCAL3"
                            else 1.04 if arm == "GAP3" else 1.0
                        ),
                        "selection_mode": (
                            local3_mode if arm == "LOCAL3" else "primary"
                        ),
                        "experiment_pass": (
                            local3_experiment_pass if arm == "LOCAL3" else True
                        ),
                    }
                    for arm in analysis.CANONICAL_ARMS
                ]
            )

        fallback = analysis.optimization_safety_table(
            selections(1.04, "fallback_base_gate_failed", False), config
        )
        fallback_local = fallback.loc[fallback["comparison"].eq("LOCAL3-LOCAL0")].iloc[
            0
        ]
        self.assertTrue(fallback_local["exact_state_loss_rule_pass"])
        self.assertFalse(fallback_local["grounded_primary_selection"])
        self.assertFalse(fallback_local["grounded_experiment_pass"])
        self.assertFalse(fallback_local["safety_pass"])
        fallback_result = analysis._safety_result(
            fallback,
            "LOCAL3-LOCAL0",
            expected_total=1,
            seeds=(2026,),
            folds=(0,),
            maximum_degradation=0.05,
        )
        self.assertEqual(fallback_result["safe_folds"], 0)

        tiny_above = float(np.nextafter(1.05, np.inf))
        self.assertGreater(tiny_above, 1.05)
        self.assertLess(tiny_above - 1.05, 1e-12)
        tiny = analysis.optimization_safety_table(
            selections(tiny_above, "primary", True), config
        )
        tiny_local = tiny.loc[tiny["comparison"].eq("LOCAL3-LOCAL0")].iloc[0]
        self.assertFalse(tiny_local["exact_state_loss_rule_pass"])
        self.assertFalse(tiny_local["safety_pass"])
        tiny_result = analysis._safety_result(
            tiny,
            "LOCAL3-LOCAL0",
            expected_total=1,
            seeds=(2026,),
            folds=(0,),
            maximum_degradation=0.05,
        )
        self.assertEqual(tiny_result["safe_folds"], 0)


class PairingPrivacyAndFigureTest(unittest.TestCase):
    def test_cross_arm_patient_order_hash_pairing_is_explicit(self) -> None:
        config = confirmation_config()
        rows: list[dict[str, object]] = []
        for seed in analysis.CANONICAL_SEEDS:
            for fold in analysis.CANONICAL_FOLDS:
                digest = hashlib.sha256(f"{seed}/{fold}".encode()).hexdigest()
                for arm in analysis.CANONICAL_ARMS:
                    rows.append(
                        {
                            "seed_base": seed,
                            "arm": arm,
                            "fold": fold,
                            "patient_order_sha256": digest,
                        }
                    )
        frame = pd.DataFrame(rows)
        audit = analysis.audit_cross_arm_patient_order_hashes(frame, config)
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(audit["paired_seed_folds"], 25)
        self.assertFalse(audit["private_hashes_exposed"])
        broken = frame.copy()
        broken.loc[
            broken["seed_base"].eq(2026)
            & broken["fold"].eq(0)
            & broken["arm"].eq("LOCAL3"),
            "patient_order_sha256",
        ] = (
            "f" * 64
        )
        with self.assertRaisesRegex(ValueError, "cross-arm patient-order hash"):
            analysis.audit_cross_arm_patient_order_hashes(broken, config)

    def test_training_patient_order_is_paired_by_common_epoch(self) -> None:
        config = confirmation_config()
        rows: list[dict[str, object]] = []
        for seed in analysis.CANONICAL_SEEDS:
            for fold in analysis.CANONICAL_FOLDS:
                for arm in analysis.CANONICAL_ARMS:
                    last_epoch = 2 if arm == "LOCAL3" else 3
                    for epoch in range(1, last_epoch + 1):
                        digest = hashlib.sha256(
                            f"{seed}/{fold}/{epoch}".encode()
                        ).hexdigest()
                        rows.append(
                            {
                                "seed_base": seed,
                                "arm": arm,
                                "fold": fold,
                                "epoch": epoch,
                                "patient_order_sha256": digest,
                            }
                        )
        frame = pd.DataFrame(rows)
        audit = analysis.audit_cross_arm_training_order_hashes(frame, config)
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(audit["fully_paired_epoch_groups"], 50)
        self.assertEqual(audit["partially_observed_later_epoch_groups"], 25)
        self.assertEqual(audit["minimum_common_epochs_per_seed_fold"], 2)
        broken = frame.copy()
        broken.loc[
            broken["seed_base"].eq(2026)
            & broken["fold"].eq(0)
            & broken["arm"].eq("LOCAL0")
            & broken["epoch"].eq(2),
            "patient_order_sha256",
        ] = (
            "e" * 64
        )
        with self.assertRaisesRegex(
            ValueError, "cross-arm training patient-order hash"
        ):
            analysis.audit_cross_arm_training_order_hashes(broken, config)

    def test_four_arm_five_seed_figures_render_without_overwrite(self) -> None:
        table2, table3 = macro_tables()
        table1 = pd.DataFrame({"arm": list(analysis.CANONICAL_ARMS)})
        fold_rows: list[dict[str, object]] = []
        for seed in analysis.CANONICAL_SEEDS:
            for fold in analysis.CANONICAL_FOLDS:
                for task in ("static", "delta"):
                    for comparison in ("LOCAL0-GAP0", "LOCAL3-LOCAL0"):
                        fold_rows.append(
                            {
                                "seed_base": seed,
                                "fold": fold,
                                "task": task,
                                "comparison": comparison,
                                "effect_spearman": 0.01 * (fold + 1),
                            }
                        )
        histories = pd.DataFrame(
            [
                {
                    "arm": arm,
                    "epoch": epoch,
                    "val_state_loss": 1.0 / epoch,
                    "val_ftv_loss": 0.5 / epoch,
                }
                for arm in analysis.CANONICAL_ARMS
                for epoch in range(1, 4)
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = figures.render_required_figures(
                table1=table1,
                table2=table2,
                table3=table3,
                table4=pd.DataFrame(),
                table6=safety_table(),
                fold_effects=pd.DataFrame(fold_rows),
                histories=histories,
                output_dir=directory,
            )
            self.assertEqual(
                tuple(path.name for path in paths), figures.FIGURE_FILENAMES
            )
            self.assertTrue(all(path.stat().st_size > 0 for path in paths))
            with self.assertRaises(FileExistsError):
                figures.render_required_figures(
                    table1=table1,
                    table2=table2,
                    table3=table3,
                    table4=pd.DataFrame(),
                    table6=safety_table(),
                    fold_effects=pd.DataFrame(fold_rows),
                    histories=histories,
                    output_dir=directory,
                )

    def test_public_tables_reject_patient_order_hashes(self) -> None:
        frame = pd.DataFrame({"epoch": [1], "patient_order_sha256": ["a" * 64]})
        with self.assertRaisesRegex(ValueError, "identifier/path"):
            analysis._assert_public_frame(frame, "history.csv")


if __name__ == "__main__":
    unittest.main()

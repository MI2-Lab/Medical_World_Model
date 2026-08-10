from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c1b_stage_b.analysis import (  # noqa: E402
    FIGURE_NAMES,
    METRICS,
    optimization_difference_in_differences,
)
from c1b_stage_b.contracts import (  # noqa: E402
    ARMS,
    FOLDS,
    REQUIRED_STAGE_A_GATES,
    SEED_BASES,
    canonical_sha256,
    file_sha256,
    ordered_patient_sha256,
)
from c1b_stage_b.data import (  # noqa: E402
    FTVRecord,
    _expected_c1b_members,
    derive_matched_stage_b_population,
    intersect_eligible_folds,
    read_fold_manifest,
    read_technical_eligibility,
    read_train_only_candidates,
)
from c1b_stage_b.gate import (  # noqa: E402
    StageAAuthorization,
    StageAGateError,
    require_stage_a_go,
)
from c1b_stage_b.matrix import GROUP_ARM_ORDER, build_matrix_groups  # noqa: E402
from c1b_stage_b.probes import _run_cell, _validate_feature_metadata  # noqa: E402
from c1b_stage_b.targets import (  # noqa: E402
    fit_grounding_transform,
    fit_static_probe_transform,
    grounding_raw_map,
    literal_delta_targets,
)
from c1b_stage_b.training import logical_sigreg_surrogate  # noqa: E402
from c1b_stage_b.upstream import DGRSObjective  # noqa: E402


def _write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> str:
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    return file_sha256(path)


def _valid_go_payload(eligibility_sha256: str, patients: int = 6) -> dict[str, object]:
    return {
        "schema_version": 1,
        "stage": "A",
        "status": "GO",
        "stage_b_authorized": True,
        "thresholds_relaxed": False,
        "chosen_input_strategy": "C1B-H",
        "eligibility_rule_frozen_before_stage_b": True,
        "eligible_population_patients": patients,
        "eligible_population_visits": patients * 4,
        "technical_eligibility_manifest_sha256": eligibility_sha256,
        "cache_completion_fraction": 1.0,
        "gates": [
            {"item": index, "gate": name, "status": "PASS"}
            for index, name in enumerate(REQUIRED_STAGE_A_GATES, start=1)
        ],
    }


class StageAGateTests(unittest.TestCase):
    def test_missing_or_prior_style_sentinel_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "STAGE_A_GO.json"
            with self.assertRaises(StageAGateError):
                require_stage_a_go(missing)
            prior = _valid_go_payload("a" * 64)
            prior.pop("eligibility_rule_frozen_before_stage_b")
            missing.write_text(json.dumps(prior), encoding="utf-8")
            with self.assertRaisesRegex(StageAGateError, "eligibility rule"):
                require_stage_a_go(missing)

    def test_new_go_binds_population_manifest_and_full_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "STAGE_A_GO.json"
            payload = _valid_go_payload("b" * 64, patients=7)
            path.write_text(json.dumps(payload), encoding="utf-8")
            authorization = require_stage_a_go(path)
            self.assertEqual(authorization.eligible_population_patients, 7)
            self.assertEqual(authorization.eligible_population_visits, 28)
            self.assertEqual(authorization.technical_eligibility_manifest_sha256, "b" * 64)
            payload["cache_completion_fraction"] = 0.999
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(StageAGateError, "100%"):
                require_stage_a_go(path)

    def test_go_rejects_arbitrary_or_reordered_pass_gates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "STAGE_A_GO.json"
            payload = _valid_go_payload("c" * 64)
            payload["gates"][0]["gate"] = "synthetic_pass_not_preregistered"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(StageAGateError, "identities/order"):
                require_stage_a_go(path)
            payload = _valid_go_payload("c" * 64)
            payload["gates"][0], payload["gates"][1] = (
                payload["gates"][1],
                payload["gates"][0],
            )
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(StageAGateError, "identities/order"):
                require_stage_a_go(path)

    def test_every_stage_b_entrypoint_authorizes_immediately_after_parse(self) -> None:
        names = (
            "build_stage_b_cache_manifests.py",
            "train_stage_b.py",
            "run_stage_b_matrix.py",
            "export_stage_b_features.py",
            "run_stage_b_probes.py",
            "run_stage_b_postprocessing.py",
            "aggregate_stage_b.py",
        )
        for name in names:
            tree = ast.parse((ROOT / "scripts" / name).read_text(encoding="utf-8"))
            main = next(
                node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
            )
            statements = main.body
            self.assertGreaterEqual(len(statements), 2, name)
            self.assertIsInstance(statements[0], ast.Assign, name)
            self.assertIsInstance(statements[1], ast.Assign, name)
            call = statements[1].value
            self.assertIsInstance(call, ast.Call, name)
            self.assertIsInstance(call.func, ast.Name, name)
            self.assertEqual(call.func.id, "authorize", name)


class DynamicCohortTests(unittest.TestCase):
    def _fold_rows(self) -> list[dict[str, object]]:
        patients = [f"P{index}" for index in range(6)]
        rows: list[dict[str, object]] = []
        for fold in range(5):
            for index, patient_id in enumerate(patients):
                test_fold = index % 5
                val_fold = (test_fold + 1) % 5
                split = "test" if fold == test_fold else "val" if fold == val_fold else "train"
                rows.append({"patient_id": patient_id, "fold": fold, "split": split})
        return rows

    def test_eligibility_x_folds_plus_authorized_train_only_is_count_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fold_path = root / "folds.csv"
            fold_sha = _write_csv(
                fold_path, self._fold_rows(), ["patient_id", "fold", "split"]
            )
            eligibility_path = root / "technical_eligibility_patients.private.csv"
            eligibility_rows = [
                {
                    "patient_id": patient_id,
                    "cohort": "fold_source",
                    "eligible": patient_id != "P5",
                    "outcome_must_not_be_read": 1,
                }
                for patient_id in [f"P{index}" for index in range(6)]
            ] + [
                {
                    "patient_id": "E1",
                    "cohort": "upstream_train_only",
                    "eligible": True,
                    "outcome_must_not_be_read": 0,
                },
                {
                    "patient_id": "E2",
                    "cohort": "upstream_train_only",
                    "eligible": False,
                    "outcome_must_not_be_read": 1,
                },
            ]
            eligibility_sha = _write_csv(
                eligibility_path,
                eligibility_rows,
                ["patient_id", "cohort", "eligible", "outcome_must_not_be_read"],
            )
            train_only_path = root / "source_train_only.private.csv"
            train_only_sha = _write_csv(
                train_only_path,
                [
                    {"patient_id": "E1", "eligible": True, "clinical_forbidden": 9},
                    {"patient_id": "E2", "eligible": True, "clinical_forbidden": 9},
                ],
                ["patient_id", "eligible", "clinical_forbidden"],
            )
            folds = read_fold_manifest(
                fold_path, fold_sha, enforce_seed2026_lock=False
            )
            eligibility = read_technical_eligibility(
                eligibility_path, eligibility_sha
            )
            source_train_only = read_train_only_candidates(
                train_only_path, train_only_sha, enforce_upstream_lock=False
            )
            matched = intersect_eligible_folds(folds, eligibility.eligible_ids)
            self.assertEqual(set(matched["patient_id"]), {"P0", "P1", "P2", "P3", "P4"})
            self.assertEqual(source_train_only, ("E1", "E2"))
            self.assertEqual(len(eligibility.candidate_ids), 8)
            self.assertEqual(len(eligibility.eligible_ids), 6)
            derived = derive_matched_stage_b_population(
                folds, eligibility, source_train_only
            )
            self.assertEqual(derived.train_only_ids, ("E1",))
            self.assertEqual(set(derived.matched_patient_ids), set(eligibility.eligible_ids))

    def test_stage_b_source_has_no_historical_population_literals(self) -> None:
        paths = list((ROOT / "src" / "c1b_stage_b").glob("*.py")) + [
            ROOT / "scripts" / "build_stage_b_cache_manifests.py"
        ]
        combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
        for forbidden in ("PRIMARY_PATIENT_COUNT", "ISPY1_ELIGIBLE_COUNT", "808 + 140", "= 948"):
            self.assertNotIn(forbidden, combined)

    def test_c1b_schema_contract_is_reused_without_imaging_imports(self) -> None:
        members = _expected_c1b_members()
        self.assertIn("image.npy", members)
        self.assertIn("schema_version.npy", members)
        self.assertIn("patient_id.npy", members)
        self.assertIn("valid_source_mask.npy", members)


class MatrixAndTargetTests(unittest.TestCase):
    def test_exact_paired_forty_run_matrix(self) -> None:
        groups = build_matrix_groups("/tmp/stage_b_synthetic", ("cuda:0", "cuda:1"))
        self.assertEqual(len(groups), 10)
        self.assertEqual(sum(len(group.cells) for group in groups), 40)
        self.assertEqual(SEED_BASES, (2026, 3026))
        self.assertEqual(FOLDS, tuple(range(5)))
        self.assertEqual(ARMS, ("L1", "L3", "N1", "N3"))
        for group in groups:
            self.assertEqual(tuple(cell.arm for cell in group.cells), GROUP_ARM_ORDER)
            self.assertIsNone(group.cells[0].paired_baseline_selection)
            self.assertIsNone(group.cells[1].paired_baseline_selection)
            self.assertIn("L1", str(group.cells[2].paired_baseline_selection))
            self.assertIn("N1", str(group.cells[3].paired_baseline_selection))

    def test_accumulated_sigreg_matches_one_logical_batch_value_and_gradient(self) -> None:
        torch.manual_seed(91)
        reference_bvd = torch.randn(32, 4, 16, dtype=torch.float32)
        reference_vbd = reference_bvd.transpose(0, 1).detach().requires_grad_(True)
        objective = DGRSObjective(
            model_name="G1",
            lambda_ftv=0.0,
            sigreg_weight=0.09,
            sigreg_projections=8,
        )
        torch.manual_seed(777)
        logical_loss = objective.sigreg(reference_vbd)
        logical_gradient_vbd, = torch.autograd.grad(logical_loss, reference_vbd)
        logical_gradient_bvd = logical_gradient_vbd.transpose(0, 1).detach()

        candidate = reference_bvd.detach().clone().requires_grad_(True)
        accumulated = candidate.new_zeros(())
        for start in range(0, 32, 4):
            stop = start + 4
            surrogate = logical_sigreg_surrogate(
                candidate[start:stop],
                reference_bvd[start:stop],
                logical_gradient_bvd[start:stop],
                logical_loss,
                logical_batch_size=32,
            )
            accumulated = accumulated + (4.0 / 32.0) * surrogate
        accumulated_gradient, = torch.autograd.grad(accumulated, candidate)
        self.assertAlmostEqual(
            float(accumulated.detach()), float(logical_loss.detach()), places=6
        )
        torch.testing.assert_close(
            accumulated_gradient,
            logical_gradient_bvd,
            rtol=1e-6,
            atol=1e-7,
        )

    def test_literal_delta_and_observable_loss_mask(self) -> None:
        values = np.asarray([10.0, 7.0, 8.5, 2.0])
        valid = np.asarray([True, True, True, True])
        observable = np.asarray([True, False, True, True])
        delta, delta_valid = literal_delta_targets(values, valid)
        np.testing.assert_allclose(delta, [-3.0, 1.5, -6.5])
        self.assertTrue(delta_valid.all())
        _, observed_valid = literal_delta_targets(values, valid, observable)
        self.assertEqual(observed_valid.tolist(), [False, False, True])
        records = {"P": FTVRecord(values, valid, observable)}
        _, loss_mask = grounding_raw_map(records)["P"]
        self.assertEqual(loss_mask.tolist(), observable.tolist())

    def test_static_transform_is_outer_train_only_and_frozen(self) -> None:
        records = {
            "A": FTVRecord(np.asarray([1.0, 2.0, 3.0, 4.0]), np.ones(4, bool), np.ones(4, bool)),
            "B": FTVRecord(np.asarray([2.0, 4.0, 8.0, 16.0]), np.ones(4, bool), np.ones(4, bool)),
            "TEST": FTVRecord(np.asarray([5.0, 6.0, 7.0, 8.0]), np.ones(4, bool), np.ones(4, bool)),
        }
        first = fit_static_probe_transform(
            records, ("A", "B"), 0
        )
        grounding, _ = fit_grounding_transform(records, ("A", "B"), 0)
        self.assertEqual(first.to_dict(), grounding.to_dict())
        records["TEST"] = FTVRecord(
            np.asarray([1e9, 2e9, 3e9, 4e9]), np.ones(4, bool), np.ones(4, bool)
        )
        second = fit_static_probe_transform(
            records, ("A", "B"), 0
        )
        for field in (
            "epsilon",
            "winsor_low",
            "winsor_high",
            "center_median",
            "scale_iqr",
            "raw_targets_sha256",
        ):
            self.assertEqual(getattr(first, field), getattr(second, field))
        self.assertEqual(first.winsor_quantiles, (0.01, 0.99))

    def test_metrics_and_optimization_did_contract(self) -> None:
        self.assertIn("pearson", METRICS)
        self.assertEqual(len(FIGURE_NAMES), 9)
        self.assertTrue(FIGURE_NAMES[0].startswith("04_"))
        self.assertTrue(FIGURE_NAMES[-1].startswith("12_"))
        rows = []
        state = {"L1": 1.0, "L3": 1.04, "N1": 0.9, "N3": 0.918}
        degradation = {"L1": 0.0, "L3": 0.04, "N1": 0.0, "N3": 0.02}
        for seed_base in SEED_BASES:
            for fold in FOLDS:
                for arm in ARMS:
                    rows.append(
                        {
                            "arm": arm,
                            "seed_base": seed_base,
                            "fold": fold,
                            "selected_validation_state_loss": state[arm],
                            "state_loss_degradation_fraction": degradation[arm],
                        }
                    )
        did = optimization_difference_in_differences(pd.DataFrame(rows))
        self.assertEqual(set(did["seed_base"]), set(SEED_BASES))
        self.assertAlmostEqual(float(did.iloc[0]["L3_minus_L1"]), 0.04)
        self.assertAlmostEqual(float(did.iloc[0]["N3_minus_N1"]), 0.02)
        self.assertAlmostEqual(float(did.iloc[0]["difference_in_differences"]), -0.02)

    def test_static_probe_reports_transformed_and_inverse_natural_spaces(self) -> None:
        patient_ids = np.asarray([f"S{index}" for index in range(12)])
        split = np.asarray(["train"] * 6 + ["val"] * 3 + ["test"] * 3)
        response = np.zeros((12, 4, 192), dtype=np.float32)
        records: dict[str, FTVRecord] = {}
        for index, patient_id in enumerate(patient_ids):
            values = np.asarray(
                [index + 1.0, index + 2.0, index + 3.0, index + 4.0]
            )
            response[index, :, 0] = np.log(values + 0.5).astype(np.float32)
            response[index, :, 1] = np.asarray([0.0, 1.0, 2.0, 3.0])
            records[str(patient_id)] = FTVRecord(
                values, np.ones(4, bool), np.ones(4, bool)
            )
        arrays = {
            "patient_id": patient_ids,
            "split": split,
            "response_state": response,
            "arm": np.asarray("N3"),
            "seed_base": np.asarray(2026),
            "fold": np.asarray(0),
        }
        selection, predictions, metrics = _run_cell(
            arrays, records, task="static", index=0, observable_only=False
        )
        self.assertEqual({row["scale"] for row in metrics}, {"natural", "transformed_outer_train"})
        self.assertTrue(all("pearson" in row for row in metrics))
        self.assertTrue(all(float(row["y_pred"]) >= 0.0 for row in predictions))
        transform = json.loads(selection["target_transform_json"])
        self.assertEqual(transform["winsor_quantiles"], [0.01, 0.99])
        self.assertFalse(selection["test_used_for_scaler"])
        self.assertFalse(selection["test_used_for_alpha_selection"])

    def test_probe_feature_is_bound_to_checkpoint_and_data_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature = root / "features.private.npz"
            checkpoint = root / "selected.pt"
            checkpoint.write_bytes(b"synthetic selected checkpoint")
            patient_ids = np.asarray(["A", "B"], dtype=str)
            arrays = {
                "patient_id": patient_ids,
                "split": np.asarray(["train", "test"], dtype=str),
                "response_state": np.zeros((2, 4, 192), dtype=np.float32),
                "arm": np.asarray("N3"),
                "seed_base": np.asarray(2026),
                "fold": np.asarray(0),
            }
            np.savez(feature, **arrays)
            provenance = {"synthetic_data_contract": "locked"}
            authorization = StageAAuthorization(
                root / "STAGE_A_GO.json",
                "a" * 64,
                {},
                2,
                8,
                "b" * 64,
            )
            metadata = {
                "schema_version": 1,
                "stage": "B",
                "arm": "N3",
                "seed_base": 2026,
                "fold": 0,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": file_sha256(checkpoint),
                "feature_path": str(feature),
                "feature_sha256": file_sha256(feature),
                "feature_tensor": "online_preprojector_r",
                "feature_shape": [2, 4, 192],
                "patient_order_sha256": ordered_patient_sha256(patient_ids),
                "current_data_contract_provenance_sha256": canonical_sha256(provenance),
                "checkpoint_base_data_contract_provenance_sha256": canonical_sha256(
                    provenance
                ),
                "checkpoint_data_provenance_sha256": "c" * 64,
                "ftv_head_called": False,
                "test_labels_used": False,
                "stage_a_sentinel_sha256": authorization.sha256,
            }
            metadata_path = feature.with_suffix(".metadata.json")
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            observed_path, _ = _validate_feature_metadata(
                feature, arrays, authorization, provenance
            )
            self.assertEqual(observed_path, metadata_path)
            checkpoint.write_bytes(b"drifted")
            with self.assertRaisesRegex(ValueError, "checkpoint path/SHA-256"):
                _validate_feature_metadata(feature, arrays, authorization, provenance)


if __name__ == "__main__":
    unittest.main()

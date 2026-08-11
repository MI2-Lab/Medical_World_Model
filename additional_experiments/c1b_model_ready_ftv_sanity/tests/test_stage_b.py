from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock
import warnings
import zipfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c1b_stage_b.analysis import (  # noqa: E402
    METRICS,
    _audit_oof_prediction_contract,
    paired_effects,
    pooled_oof_metrics,
)
from c1b_stage_b.contracts import (  # noqa: E402
    ARMS,
    CACHE_MANIFEST_USECOLS,
    FOLD_USECOLS,
    FOLDS,
    FTV_TRANSITION_USECOLS,
    ISPY1_ELIGIBILITY_USECOLS,
    OBSERVABILITY_USECOLS,
    SEED_BASES,
    canonical_sha256,
    file_sha256,
    ordered_patient_sha256,
)
from c1b_stage_b.data import (  # noqa: E402
    CacheEntry,
    FTVRecord,
    combine_ftv_observability,
    fingerprint_cache_file,
    load_dce7,
    read_cache_manifest,
    read_fold_manifest,
    read_ispy1_eligibility,
    read_observability,
    read_raw_ftv,
    verify_cache_entry,
)
from c1b_stage_b.gate import StageAGateError, require_stage_a_go  # noqa: E402
from c1b_stage_b.features import _validate_checkpoint_data_contract  # noqa: E402
from c1b_stage_b.probes import _validate_feature_split_contract  # noqa: E402
from c1b_stage_b.matrix import (  # noqa: E402
    GROUP_ARM_ORDER,
    MatrixExecutionError,
    build_matrix_groups,
    build_train_command,
    execute_matrix_groups,
    parse_multi_devices,
)
from c1b_stage_b.targets import literal_delta_targets  # noqa: E402

try:
    import torch
except ModuleNotFoundError:
    torch = None
else:
    from c1b_stage_b.targets import fit_grounding_transform
    from c1b_stage_b.training import (
        TrainHyperparameters,
        logical_patient_batches,
        run_logical_train_epoch,
        scale_microbatch_components,
        select_checkpoint,
    )
    from c1b_stage_b.upstream import (
        DGRSObjective,
        DGRSWorldModel,
        paired_initialization_report,
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> str:
    pd.DataFrame(rows).to_csv(path, index=False)
    return file_sha256(path)


def _write_selective_schema3_cache(
    root: Path,
    patient_id: str,
    image: np.ndarray,
    *,
    overrides: dict[str, np.ndarray] | None = None,
    extra: dict[str, np.ndarray] | None = None,
) -> Path:
    from c1b_sanity.cache import _REQUIRED_KEYS

    root.mkdir(parents=True, exist_ok=True)
    payload = {key: np.asarray(0, dtype=np.uint8) for key in _REQUIRED_KEYS}
    payload.update(
        {
            "schema_version": np.asarray(3, dtype=np.int16),
            "patient_id": np.asarray(patient_id),
            "image": np.asarray(image),
        }
    )
    payload.update(overrides or {})
    payload.update(extra or {})
    path = root / (hashlib.sha256(patient_id.encode("utf-8")).hexdigest() + ".npz")
    np.savez(path, **payload)
    return path


class StageBSafeDataTests(unittest.TestCase):
    def test_formal_matrix_is_exactly_forty_runs(self) -> None:
        cells = {(seed, fold, arm) for seed in SEED_BASES for fold in FOLDS for arm in ARMS}
        self.assertEqual(ARMS, ("L1", "L3", "N1", "N3"))
        self.assertEqual(SEED_BASES, (2026, 3026))
        self.assertEqual(FOLDS, tuple(range(5)))
        self.assertEqual(len(cells), 40)

    def test_c1b_schema3_adapter_is_selective_and_stage_a_validator_is_intact(self) -> None:
        from c1b_sanity.cache import validate_cache_arrays

        adapter_source = inspect.getsource(load_dce7)
        validator_source = inspect.getsource(validate_cache_arrays)
        self.assertNotIn("load_model_tensor", adapter_source)
        self.assertIn('"schema_version.npy"', inspect.getsource(sys.modules[load_dce7.__module__]))
        self.assertIn('"patient_id.npy"', inspect.getsource(sys.modules[load_dce7.__module__]))
        self.assertIn('"image.npy"', adapter_source)
        self.assertIn("int(schema_version.item()) != 3", validator_source)
        self.assertIn('schema_version.dtype.kind not in "iu"', validator_source)
        self.assertNotIn('"valid_source_mask.npy"', adapter_source)

    def test_split_and_ftv_readers_use_exact_allowlists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fold_rows = []
            for fold in range(5):
                for index, patient_id in enumerate(("P1", "P2")):
                    split = "test" if fold == index else (
                        "val" if fold == (index + 1) % 5 else "train"
                    )
                    fold_rows.append(
                        {
                            "patient_id": patient_id,
                            "fold": fold,
                            "split": split,
                            "label_outcome": index,
                            "therapy_arm": "hidden",
                        }
                    )
            fold_path = root / "fold.csv"
            fold_sha = _write_csv(fold_path, fold_rows)
            transitions = []
            values = (1.0, 2.0, 4.0, 7.0)
            for index, transition in enumerate(("T0→T1", "T1→T2", "T2→T3")):
                transitions.append(
                    {
                        "patient_id": "P1",
                        "transition": transition,
                        "start_visit": f"T{index}",
                        "end_visit": f"T{index + 1}",
                        "ftv_start": values[index],
                        "ftv_end": values[index + 1],
                        "ftv_valid": True,
                        "label_outcome": 1,
                        "lesion_diameter": 99,
                        "sphericity": 0.4,
                        "enhancement": 0.8,
                    }
                )
            ftv_path = root / "ftv.csv"
            ftv_sha = _write_csv(ftv_path, transitions)
            obs_path = root / "grounding.private.csv"
            obs_sha = _write_csv(
                obs_path,
                [
                    {
                        "patient_id": "P1",
                        "visit": f"T{index}",
                        "ftv_measurement_valid": True,
                        "grounding_observable_mask": index != 1,
                        "source_boundary_touch": index == 1,
                        "therapy_arm": "hidden",
                    }
                    for index in range(4)
                ],
            )
            calls: list[tuple[str, ...]] = []
            original = pd.read_csv

            def audited_read_csv(*args: object, **kwargs: object) -> pd.DataFrame:
                calls.append(tuple(kwargs.get("usecols", ())))
                return original(*args, **kwargs)

            with mock.patch.object(pd, "read_csv", side_effect=audited_read_csv):
                folds = read_fold_manifest(fold_path, fold_sha, expected_patient_count=2)
                raw = read_raw_ftv(ftv_path, ftv_sha)
                observable = read_observability(obs_path, obs_sha)
            records = combine_ftv_observability(raw, observable)
            self.assertEqual(tuple(folds.columns), FOLD_USECOLS)
            self.assertEqual(
                calls, [FOLD_USECOLS, FTV_TRANSITION_USECOLS, OBSERVABILITY_USECOLS]
            )
            self.assertEqual(
                records["P1"].grounding_eligible.tolist(), [True, False, True, True]
            )

    def test_private_cache_and_eligibility_readers_use_exact_allowlists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_file = root / "patient.npz"
            np.savez(cache_file, x=np.asarray(0, dtype=np.float32))
            cache_stat = cache_file.stat()
            cache_manifest = root / "legacy.private.csv"
            cache_sha = _write_csv(
                cache_manifest,
                [
                    {
                        "patient_id": "P1",
                        "cache_path": str(cache_file),
                        "cache_sha256": file_sha256(cache_file),
                        "cache_size_bytes": cache_stat.st_size,
                        "cache_mtime_ns": cache_stat.st_mtime_ns,
                        "input_kind": "legacy",
                        "diagnosis": "must_not_be_read",
                        "therapy_arm": "must_not_be_read",
                    }
                ],
            )
            eligibility = root / "eligibility.private.csv"
            eligibility_sha = _write_csv(
                eligibility,
                [
                    {
                        "patient_id": "E1",
                        "eligible": True,
                        "diagnosis": "must_not_be_read",
                        "therapy_arm": "must_not_be_read",
                    },
                    {
                        "patient_id": "E2",
                        "eligible": False,
                        "diagnosis": "must_not_be_read",
                        "therapy_arm": "must_not_be_read",
                    },
                ],
            )
            calls: list[tuple[str, ...]] = []
            original = pd.read_csv

            def audited_read_csv(*args: object, **kwargs: object) -> pd.DataFrame:
                calls.append(tuple(kwargs.get("usecols", ())))
                return original(*args, **kwargs)

            with mock.patch.object(pd, "read_csv", side_effect=audited_read_csv):
                entries = read_cache_manifest(
                    cache_manifest,
                    cache_sha,
                    expected_input_kind="legacy",
                    verify_cache_files=True,
                )
                with self.assertRaisesRegex(ValueError, "exactly 2 eligible"):
                    read_ispy1_eligibility(
                        eligibility,
                        eligibility_sha,
                        expected_candidate_count=2,
                        expected_eligible_count=2,
                    )
                eligible = read_ispy1_eligibility(
                    eligibility,
                    eligibility_sha,
                    expected_candidate_count=2,
                    expected_eligible_count=1,
                )
            self.assertEqual(tuple(entries), ("P1",))
            self.assertEqual(eligible, ("E1",))
            self.assertEqual(
                calls,
                [
                    CACHE_MANIFEST_USECOLS,
                    ISPY1_ELIGIBILITY_USECOLS,
                    ISPY1_ELIGIBILITY_USECOLS,
                ],
            )

    def test_c1b_hot_reader_materializes_only_schema_identity_and_image(self) -> None:
        import c1b_stage_b.data as data_module

        shape = (4, 7, 2, 3, 4)
        image = np.linspace(-1.0, 1.0, num=math.prod(shape), dtype=np.float32).reshape(shape)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _write_selective_schema3_cache(root, "P1", image)
            entry = fingerprint_cache_file(
                path,
                "P1",
                "c1b",
                expected_sha256=file_sha256(path),
            )
            opened: list[str] = []
            original = data_module._read_npy_member

            def tracked(*args: object, **kwargs: object) -> np.ndarray:
                opened.append(str(args[1]))
                return original(*args, **kwargs)

            with (
                mock.patch.object(data_module, "C1B_IMAGE_SHAPE", shape),
                mock.patch.object(data_module, "_read_npy_member", side_effect=tracked),
                mock.patch.object(
                    data_module,
                    "file_sha256",
                    side_effect=AssertionError("hot reader must not hash cache files"),
                ),
            ):
                observed = load_dce7(entry)
            np.testing.assert_array_equal(observed, image)
            self.assertEqual(opened, ["schema_version.npy", "patient_id.npy", "image.npy"])
            self.assertEqual(observed.dtype, np.dtype(np.float32))
            self.assertTrue(np.isfinite(observed).all())

    def test_c1b_hot_reader_rejects_touch_and_three_way_identity_drift(self) -> None:
        import c1b_stage_b.data as data_module

        shape = (4, 7, 1, 1, 2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _write_selective_schema3_cache(
                root, "P1", np.zeros(shape, dtype=np.float32)
            )
            entry = fingerprint_cache_file(
                path, "P1", "c1b", expected_sha256=file_sha256(path)
            )
            original_stat = path.stat()
            os.utime(
                path,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 1_000_000),
            )
            with (
                mock.patch.object(data_module, "C1B_IMAGE_SHAPE", shape),
                self.assertRaisesRegex(ValueError, "cache stat no longer matches"),
            ):
                load_dce7(entry)

            current = path.stat()
            swapped = CacheEntry(
                "P2", path, entry.sha256, current.st_size, current.st_mtime_ns, "c1b"
            )
            with self.assertRaisesRegex(ValueError, "filename is not bound"):
                load_dce7(swapped)

    def test_c1b_envelope_rejects_extra_member_bad_schema_and_bytes_identity(self) -> None:
        shape = (4, 7, 1, 1, 1)
        cases = (
            ("extra", {}, {"pCR_future_label": np.asarray(1, dtype=np.int8)}, "members drifted"),
            (
                "schema",
                {"schema_version": np.asarray(3.0, dtype=np.float32)},
                {},
                "schema_version",
            ),
            (
                "identity",
                {"patient_id": np.asarray(b"P1")},
                {},
                "forbidden dtype",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, (label, overrides, extra, message) in enumerate(cases):
                with self.subTest(case=label):
                    patient_id = f"P{index + 1}"
                    case_root = root / label
                    case_root.mkdir()
                    path = _write_selective_schema3_cache(
                        case_root,
                        patient_id,
                        np.zeros(shape, dtype=np.float32),
                        overrides=overrides,
                        extra=extra,
                    )
                    with self.assertRaisesRegex(ValueError, message):
                        fingerprint_cache_file(
                            path,
                            patient_id,
                            "c1b",
                            expected_sha256=file_sha256(path),
                        )

            valid_root = root / "valid"
            valid = _write_selective_schema3_cache(
                valid_root, "P-DUP", np.zeros(shape, dtype=np.float32)
            )
            duplicate_root = root / "duplicate"
            duplicate_root.mkdir()
            duplicate = duplicate_root / valid.name
            with zipfile.ZipFile(valid, mode="r") as source, zipfile.ZipFile(
                duplicate, mode="w", compression=zipfile.ZIP_STORED
            ) as destination:
                for info in source.infolist():
                    destination.writestr(info.filename, source.read(info.filename))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    destination.writestr("image.npy", source.read("image.npy"))
            with self.assertRaisesRegex(ValueError, "duplicate NPZ member"):
                fingerprint_cache_file(
                    duplicate,
                    "P-DUP",
                    "c1b",
                    expected_sha256=file_sha256(duplicate),
                )

    def test_c1b_header_preflight_rejects_oversized_tiny_or_bad_image_layout(self) -> None:
        import c1b_stage_b.data as data_module

        shape = (4, 7, 2, 2, 2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            oversized = _write_selective_schema3_cache(
                root / "oversized",
                "P1",
                np.zeros(shape, dtype=np.float32),
                overrides={"schema_version": np.zeros(5000, dtype=np.int16)},
            )
            with self.assertRaisesRegex(ValueError, "size bound"):
                fingerprint_cache_file(
                    oversized,
                    "P1",
                    "c1b",
                    expected_sha256=file_sha256(oversized),
                )

            for label, image, message in (
                ("dtype", np.zeros(shape, dtype=np.float64), "dtype"),
                ("order", np.asfortranarray(np.zeros(shape, dtype=np.float32)), "C order"),
            ):
                with self.subTest(case=label):
                    case_root = root / label
                    case_root.mkdir()
                    patient_id = f"P-{label}"
                    path = _write_selective_schema3_cache(case_root, patient_id, image)
                    entry = fingerprint_cache_file(
                        path,
                        patient_id,
                        "c1b",
                        expected_sha256=file_sha256(path),
                    )
                    with (
                        mock.patch.object(data_module, "C1B_IMAGE_SHAPE", shape),
                        self.assertRaisesRegex(ValueError, message),
                    ):
                        load_dce7(entry)

    def test_formal_preflight_hash_catches_unselected_sidecar_change(self) -> None:
        shape = (4, 7, 1, 1, 1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.zeros(shape, dtype=np.float32)
            path = _write_selective_schema3_cache(root, "P1", image)
            entry = fingerprint_cache_file(
                path, "P1", "c1b", expected_sha256=file_sha256(path)
            )
            original = path.stat()
            _write_selective_schema3_cache(
                root,
                "P1",
                image,
                overrides={"support_available": np.asarray(1, dtype=np.uint8)},
            )
            self.assertEqual(path.stat().st_size, original.st_size)
            os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                verify_cache_entry(entry, verify_sha256=True)

    def test_literal_delta_is_natural_subtraction(self) -> None:
        delta, valid = literal_delta_targets(
            np.asarray([10.0, 7.0, 8.5, 2.0]), np.asarray([True, True, True, True])
        )
        np.testing.assert_allclose(delta, [-3.0, 1.5, -6.5])
        self.assertEqual(valid.tolist(), [True, True, True])
        observable_delta, observable_valid = literal_delta_targets(
            np.asarray([10.0, 7.0, 8.5, 2.0]),
            np.ones(4, dtype=bool),
            np.asarray([True, False, True, True]),
        )
        np.testing.assert_allclose(observable_delta, delta)
        self.assertEqual(observable_valid.tolist(), [False, False, True])

    def test_stage_a_gate_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sentinel = Path(directory) / "STAGE_A_GO.json"
            with self.assertRaises(StageAGateError):
                require_stage_a_go(sentinel)
            payload = {
                "schema_version": 1,
                "stage": "A",
                "status": "NO-GO",
                "thresholds_relaxed": False,
                "stage_b_authorized": False,
                "gates": [{"gate": "example", "status": "FAIL"}],
            }
            sentinel.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(StageAGateError):
                require_stage_a_go(sentinel)
            payload.update(
                {
                    "status": "GO",
                    "stage_b_authorized": True,
                    "gates": [{"gate": "example", "status": "PASS"}],
                }
            )
            sentinel.write_text(json.dumps(payload), encoding="utf-8")
            authorization = require_stage_a_go(sentinel)
            self.assertEqual(authorization.sha256, file_sha256(sentinel))

    def test_paired_effects_include_literal_difference_in_differences(self) -> None:
        rows = []
        for arm, value in {"L1": 1.0, "L3": 2.0, "N1": 3.0, "N3": 5.0}.items():
            row: dict[str, object] = {
                "arm": arm,
                "seed_base": 2026,
                "fold": 0,
                "task": "delta",
                "endpoint": "macro",
                "target_semantics": "literal_ftv_end_minus_ftv_start",
                "analysis_scope": "primary_measurement_valid",
                "scale": "natural",
            }
            row.update({metric: value for metric in METRICS})
            rows.append(row)
        effects = paired_effects(pd.DataFrame(rows))
        first = effects.iloc[0]
        self.assertEqual(first["L3_minus_L1"], 1.0)
        self.assertEqual(first["N3_minus_N1"], 2.0)
        self.assertEqual(first["N1_minus_L1"], 2.0)
        self.assertEqual(first["difference_in_differences"], 1.0)

    def test_feature_split_identity_is_bound_to_locked_fold(self) -> None:
        folds = pd.DataFrame(
            {
                "patient_id": ["P1", "P2", "P3"],
                "fold": [0, 0, 0],
                "split": ["train", "val", "test"],
            }
        )
        arrays = {
            "patient_id": np.asarray(["P3", "P1", "P2"]),
            "split": np.asarray(["test", "train", "val"]),
            "fold": np.asarray(0, dtype=np.int64),
        }
        _validate_feature_split_contract(arrays, folds)
        arrays["split"] = np.asarray(["val", "train", "test"])
        with self.assertRaises(ValueError):
            _validate_feature_split_contract(arrays, folds)

    def test_pooled_oof_metrics_recompute_nonlinear_endpoints(self) -> None:
        predictions = pd.DataFrame(
            {
                "seed_base": [2026] * 5,
                "arm": ["L1"] * 5,
                "task": ["static"] * 5,
                "endpoint": ["T0"] * 5,
                "analysis_scope": ["primary_measurement_valid"] * 5,
                "target_semantics": ["natural_ftv"] * 5,
                "patient_id": [f"P{index}" for index in range(5)],
                "y_true": [0.0, 1.0, 2.0, 3.0, 10.0],
                "y_pred": [0.0, 1.5, 1.0, 5.0, 4.0],
                "b0_prediction": [2.0, 2.0, 2.0, 3.0, 3.0],
            }
        )
        # Supply all endpoints so the frozen static macro can also be formed.
        predictions = pd.concat(
            [predictions.assign(endpoint=endpoint) for endpoint in ("T0", "T1", "T2", "T3")],
            ignore_index=True,
        )
        pooled = pooled_oof_metrics(predictions)
        t0 = pooled.loc[pooled["endpoint"].eq("T0")].iloc[0]
        self.assertEqual(t0["aggregation"], "pooled_5fold_oof")
        self.assertEqual(int(t0["n_test"]), 5)
        self.assertAlmostEqual(float(t0["r2"]), 0.3431528662420382)
        self.assertEqual(set(pooled["endpoint"]), {"T0", "T1", "T2", "T3", "macro"})

    def test_oof_audit_requires_unique_paired_test_coverage(self) -> None:
        prediction_rows = []
        metric_rows = []
        for fold in range(5):
            for arm in ARMS:
                common = {
                    "seed_base": 2026,
                    "arm": arm,
                    "fold": fold,
                    "task": "static",
                    "endpoint": "T0",
                    "analysis_scope": "primary_measurement_valid",
                    "target_semantics": "natural_ftv",
                }
                prediction_rows.append(
                    {
                        **common,
                        "patient_id": f"P{fold}",
                        "split": "test",
                        "y_true": float(fold),
                        "y_pred": float(fold),
                        "b0_prediction": 2.0,
                        "test_predict_call_count": 1,
                    }
                )
                metric_rows.append(
                    {**common, "scale": "natural", "n_test": 1}
                )
        predictions = pd.DataFrame(prediction_rows)
        metrics = pd.DataFrame(metric_rows)
        _audit_oof_prediction_contract(metrics, predictions)
        duplicated = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            _audit_oof_prediction_contract(metrics, duplicated)

    def test_feature_export_rejects_checkpoint_from_another_data_contract(self) -> None:
        splits = SimpleNamespace(
            train_primary=("P1", "P2"),
            train_all=("P1", "P2", "E1"),
            val=("P3",),
            test=("P4",),
        )
        current = {
            "fold_manifest": "/locked/folds.csv",
            "fold_manifest_sha256": "a" * 64,
            "legacy_cache_manifest": "/locked/legacy.private.csv",
            "legacy_cache_manifest_sha256": "b" * 64,
            "primary_patient_count": 808,
            "eligible_ispy1_count": 140,
            "cache_files_verified": False,
        }
        run_provenance = {
            **current,
            "ftv_transform": {"fold": 0},
            "train_primary_order_sha256": ordered_patient_sha256(splits.train_primary),
            "train_all_order_sha256": ordered_patient_sha256(splits.train_all),
            "validation_order_sha256": ordered_patient_sha256(splits.val),
            "test_patient_count_not_loaded": len(splits.test),
            "model_forward_fields": ["image"],
            "auxiliary_fields": ["ftv_target", "ftv_mask"],
            "global_fallback_restart": False,
        }
        checkpoint = {
            "data_provenance": run_provenance,
            "data_provenance_sha256": canonical_sha256(run_provenance),
            "train_patient_sha256": canonical_sha256(sorted(splits.train_all)),
            "val_patient_sha256": canonical_sha256(sorted(splits.val)),
        }
        data = SimpleNamespace(provenance=current)
        observed = _validate_checkpoint_data_contract(checkpoint, data, splits)
        self.assertEqual(observed, checkpoint["data_provenance_sha256"])

        changed = SimpleNamespace(
            provenance={**current, "legacy_cache_manifest_sha256": "c" * 64}
        )
        with self.assertRaises(ValueError):
            _validate_checkpoint_data_contract(checkpoint, changed, splits)

        tampered = dict(checkpoint)
        tampered["train_patient_sha256"] = "d" * 64
        with self.assertRaises(ValueError):
            _validate_checkpoint_data_contract(tampered, data, splits)


class StageBMatrixSchedulerTests(unittest.TestCase):
    def test_three_device_plan_has_ten_groups_and_exactly_forty_commands(self) -> None:
        devices = parse_multi_devices("cuda:0,cuda:1,cuda:2")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            groups = build_matrix_groups(root, devices)
            self.assertEqual(len(groups), 10)
            self.assertEqual(sum(len(group.cells) for group in groups), 40)
            assignment = {device: 0 for device in devices}
            commands = []
            for group in groups:
                assignment[group.device] += 1
                self.assertEqual(tuple(cell.arm for cell in group.cells), GROUP_ARM_ORDER)
                for cell in group.cells:
                    command = build_train_command(
                        cell,
                        python_executable="/python3.11",
                        train_script="/train_stage_b.py",
                        stage_a_sentinel="/STAGE_A_GO.json",
                        data_contract="/stage_b_data_contract.private.json",
                        data_contract_sha256="a" * 64,
                        physical_batch_size=4,
                        accumulation_steps=8,
                        workers=2,
                        global_fallback_restart=False,
                    )
                    commands.append(command)
                    self.assertEqual(command[command.index("--device") + 1], group.device)
                    self.assertEqual(command[command.index("--physical-batch-size") + 1], "4")
                    self.assertEqual(command[command.index("--accumulation-steps") + 1], "8")
                    if cell.arm in {"L3", "N3"}:
                        self.assertIn("--paired-baseline-selection", command)
                        baseline = Path(
                            command[command.index("--paired-baseline-selection") + 1]
                        )
                        baseline_arm = "L1" if cell.arm == "L3" else "N1"
                        self.assertEqual(
                            baseline,
                            root
                            / f"seed_{cell.seed_base}"
                            / baseline_arm
                            / f"fold_{cell.fold}"
                            / "selection.json",
                        )
                    else:
                        self.assertNotIn("--paired-baseline-selection", command)
            self.assertEqual(assignment, {"cuda:0": 4, "cuda:1": 3, "cuda:2": 3})
            self.assertEqual(len(commands), 40)

    def test_default_single_device_plan_remains_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            groups = build_matrix_groups(directory, ("cuda",))
        self.assertEqual({group.device for group in groups}, {"cuda"})
        self.assertEqual([group.index for group in groups], list(range(10)))

    def test_two_by_sixteen_requires_global_restart_flag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cell = build_matrix_groups(directory, ("cuda:0",))[0].cells[0]
            common = {
                "python_executable": "/python3.11",
                "train_script": "/train_stage_b.py",
                "stage_a_sentinel": "/STAGE_A_GO.json",
                "data_contract": "/stage_b_data_contract.private.json",
                "data_contract_sha256": "a" * 64,
                "workers": 2,
            }
            with self.assertRaises(ValueError):
                build_train_command(
                    cell,
                    physical_batch_size=2,
                    accumulation_steps=16,
                    global_fallback_restart=False,
                    **common,
                )
            command = build_train_command(
                cell,
                physical_batch_size=2,
                accumulation_steps=16,
                global_fallback_restart=True,
                **common,
            )
            self.assertIn("--global-fallback-restart", command)

    def test_invalid_or_duplicate_multi_device_lists_fail_closed(self) -> None:
        for value in ("", "cuda:0,cuda:0", "cuda", "cpu:0,cuda:1", "cuda:x"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_multi_devices(value)

    def test_one_sequential_stream_per_device_preserves_group_dependencies(self) -> None:
        devices = ("cuda:0", "cuda:1", "cuda:2")
        with tempfile.TemporaryDirectory() as directory:
            groups = build_matrix_groups(directory, devices)
            calls: list[tuple[int, str, str]] = []
            active_devices: set[str] = set()
            lock = threading.Lock()

            def run_cell(cell: object) -> None:
                with lock:
                    self.assertNotIn(cell.device, active_devices)
                    active_devices.add(cell.device)
                    calls.append((cell.group_index, cell.arm, cell.device))
                time.sleep(0.001)
                with lock:
                    active_devices.remove(cell.device)

            completed = execute_matrix_groups(groups, devices, run_cell)
        self.assertEqual(len(completed), 40)
        self.assertEqual(len(calls), 40)
        for group_index in range(10):
            self.assertEqual(
                tuple(arm for index, arm, _ in calls if index == group_index),
                GROUP_ARM_ORDER,
            )

    def test_first_failure_aborts_active_workers_and_starts_no_dependents(self) -> None:
        devices = ("cuda:0", "cuda:1", "cuda:2")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "preserve-me.txt"
            marker.write_text("keep", encoding="utf-8")
            groups = build_matrix_groups(root, devices)
            barrier = threading.Barrier(3)
            failure_announced = threading.Event()
            lock = threading.Lock()
            calls: list[tuple[int, str]] = []
            abort_calls = 0

            def run_cell(cell: object) -> None:
                with lock:
                    calls.append((cell.group_index, cell.arm))
                if cell.group_index in {0, 1, 2} and cell.arm == "L1":
                    barrier.wait(timeout=2.0)
                    if cell.group_index == 0:
                        failure_announced.set()
                        raise RuntimeError("synthetic failure")
                    failure_announced.wait(timeout=2.0)
                    time.sleep(0.05)

            def abort_active() -> None:
                nonlocal abort_calls
                with lock:
                    abort_calls += 1

            with self.assertRaises(MatrixExecutionError):
                execute_matrix_groups(groups, devices, run_cell, abort_active)
            self.assertEqual(abort_calls, 1)
            self.assertEqual(set(calls), {(0, "L1"), (1, "L1"), (2, "L1")})
            self.assertTrue(marker.is_file())
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")


@unittest.skipUnless(torch is not None, "PyTorch training runtime is not installed")
class StageBTorchContractTests(unittest.TestCase):
    def test_logical_accumulation_matches_exact_patient_objective(self) -> None:
        for physical in (4, 2):
            with self.subTest(physical=physical):
                logical_size = 32
                parameter_full = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
                target = torch.linspace(-1.0, 1.0, logical_size, dtype=torch.float64)
                eligible = torch.tensor([(index % 3) != 0 for index in range(logical_size)])
                base_full = ((parameter_full - target) ** 2).mean()
                ftv_full = torch.abs(parameter_full - 0.5 * target)[eligible].mean()
                (base_full + 0.25 * ftv_full).backward()
                expected_gradient = parameter_full.grad.detach().clone()

                parameter_micro = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
                logical_ftv_count = int(eligible.sum())
                for start in range(0, logical_size, physical):
                    stop = start + physical
                    micro_base = ((parameter_micro - target[start:stop]) ** 2).mean()
                    micro_eligible = eligible[start:stop]
                    if bool(micro_eligible.any()):
                        micro_ftv = torch.abs(
                            parameter_micro - 0.5 * target[start:stop]
                        )[micro_eligible].mean()
                    else:
                        micro_ftv = parameter_micro * 0.0
                    scale_microbatch_components(
                        micro_base,
                        micro_ftv,
                        microbatch_size=physical,
                        logical_batch_size=logical_size,
                        microbatch_ftv_patients=int(micro_eligible.sum()),
                        logical_ftv_patients=logical_ftv_count,
                        lambda_ftv=0.25,
                    ).backward()
                torch.testing.assert_close(
                    parameter_micro.grad, expected_gradient, atol=1e-12, rtol=1e-12
                )

    def test_logical_order_and_tail_are_arm_invariant(self) -> None:
        ids = [f"P{index:03d}" for index in range(70)]
        first = logical_patient_batches(ids, 2026, 1)
        second = logical_patient_batches(ids, 2026, 1)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertTrue(all(len(batch) == 32 for batch in first))
        self.assertEqual(len({patient for batch in first for patient in batch}), 64)

    def test_optimizer_clip_and_ema_occur_once_per_logical_batch(self) -> None:
        class TinyDataset:
            def __init__(self) -> None:
                self.patient_ids = tuple(f"P{index:03d}" for index in range(64))
                self.transformed_ftv = {
                    patient_id: (np.ones(4, dtype=np.float32), np.ones(4, dtype=bool))
                    for patient_id in self.patient_ids
                }

            def __len__(self) -> int:
                return len(self.patient_ids)

            def __getitem__(self, index: int) -> dict[str, object]:
                return {
                    "patient_id": self.patient_ids[index],
                    "image": torch.tensor([float(index) / 64.0]),
                    "ftv_target": torch.ones(4),
                    "ftv_mask": torch.ones(4, dtype=torch.bool),
                }

        class TinyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.5))
                self.ema_calls = 0

            def forward(self, image: torch.Tensor, unused: None) -> SimpleNamespace:
                response = (self.weight * image[:, :1]).reshape(-1, 1, 1).expand(-1, 4, 2)
                return SimpleNamespace(response_state=response)

            def update_target(self, momentum: float) -> None:
                self.assert_momentum = momentum
                self.ema_calls += 1

        class TinyObjective(torch.nn.Module):
            lambda_ftv = 0.25

            def forward(
                self, output: SimpleNamespace, target: torch.Tensor, mask: torch.Tensor
            ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
                prediction = output.response_state[:, 0, 0]
                base = ((prediction - 1.0) ** 2).mean()
                patient_valid = mask.any(dim=1)
                ftv = ((prediction[patient_valid] - target[patient_valid, 0]) ** 2).mean()
                count = patient_valid.sum().to(torch.float32)
                return base + self.lambda_ftv * ftv, {
                    "base_loss": base.detach(),
                    "state_loss": base.detach(),
                    "sigreg_loss": (base * 0.0).detach(),
                    "ftv_loss": ftv.detach(),
                    "ftv_patients": count,
                    "ftv_valid_visits": mask.sum().to(torch.float32),
                    "_base_component": base,
                    "_ftv_component_raw": ftv,
                }

        model = TinyModel()
        objective = TinyObjective()
        dataset = TinyDataset()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        step_calls = 0
        original_step = optimizer.step

        def counted_step(*args: object, **kwargs: object) -> object:
            nonlocal step_calls
            step_calls += 1
            return original_step(*args, **kwargs)

        optimizer.step = counted_step  # type: ignore[method-assign]
        import c1b_stage_b.training as training_module

        clip_calls = 0
        original_clip = training_module.clip_grad_norm_

        def counted_clip(*args: object, **kwargs: object) -> object:
            nonlocal clip_calls
            clip_calls += 1
            return original_clip(*args, **kwargs)

        logical = logical_patient_batches(dataset.patient_ids, 2026, 1)
        with mock.patch.object(training_module, "clip_grad_norm_", side_effect=counted_clip):
            stats = run_logical_train_epoch(
                model,
                objective,
                dataset,  # type: ignore[arg-type]
                optimizer,
                torch.device("cpu"),
                logical,
                TrainHyperparameters(workers=0),
            )
        self.assertEqual(step_calls, 2)
        self.assertEqual(clip_calls, 2)
        self.assertEqual(model.ema_calls, 2)
        self.assertAlmostEqual(model.assert_momentum, 0.996)
        self.assertEqual(stats["optimizer_steps"], 2)
        self.assertEqual(stats["ema_updates"], 2)
        self.assertEqual(stats["physical_microbatches"], 16)

    def test_observable_mask_controls_transform_fit_only(self) -> None:
        records = {
            "train": FTVRecord(
                values=np.asarray([1.0, 2.0, 4.0, 8.0]),
                measurement_valid=np.ones(4, dtype=bool),
                observable=np.asarray([True, False, True, True]),
            ),
            "validation": FTVRecord(
                values=np.asarray([1e8, 1e8, 1e8, 1e8]),
                measurement_valid=np.ones(4, dtype=bool),
                observable=np.ones(4, dtype=bool),
            ),
        }
        transform, transformed = fit_grounding_transform(records, ["train"], fold=0)
        self.assertEqual(transform.train_patient_count, 1)
        self.assertEqual(transform.valid_visit_count, 3)
        self.assertEqual(transformed["train"][1].tolist(), [True, False, True, True])
        self.assertTrue(transformed["validation"][1].all())
        self.assertLess(transform.winsor_high, np.log(1e8 + transform.epsilon))

    def test_grounded_selection_fallback_is_marked_failure(self) -> None:
        epochs = [
            {
                "epoch": 1,
                "val_state_loss": 1.07,
                "val_ftv_loss": 0.4,
                "val_grounded_patients": 8,
                "val_representation_std": 0.2,
            },
            {
                "epoch": 2,
                "val_state_loss": 1.06,
                "val_ftv_loss": 0.3,
                "val_grounded_patients": 8,
                "val_representation_std": 0.2,
            },
        ]
        selection = select_checkpoint(
            epochs,
            grounded=True,
            min_representation_std=0.05,
            paired_baseline_state_loss=1.0,
        )
        self.assertEqual(selection["selected_epoch"], 2)
        self.assertEqual(selection["selection_mode"], "fallback_base_gate_failed")
        self.assertFalse(selection["experiment_pass"])
        self.assertFalse(selection["test_data_used"])

    def test_four_arm_initialization_and_upstream_identity(self) -> None:
        report = paired_initialization_report(2026)
        self.assertEqual(len(set(report["per_arm_common_sha256"].values())), 1)
        self.assertEqual(len(set(report["per_arm_transition_sha256"].values())), 1)
        self.assertEqual(
            report["ftv_head_present"],
            {"L1": False, "L3": True, "N1": False, "N3": True},
        )
        self.assertIn("g3_multiseed_generalization", inspect.getfile(DGRSWorldModel))
        self.assertIn("g3_multiseed_generalization", inspect.getfile(DGRSObjective))


if __name__ == "__main__":
    unittest.main()

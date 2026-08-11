from __future__ import annotations

import copy
from datetime import datetime, timezone
import importlib.util
import io
import json
from pathlib import Path
import runpy
import stat
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
SRC_ROOT = ROOT / "src"
SCRIPTS_ROOT = ROOT / "scripts"
for source in (SRC_ROOT, SCRIPTS_ROOT):
    value = str(source.resolve())
    while value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)


def load_script(name: str):
    path = SCRIPTS_ROOT / name
    spec = importlib.util.spec_from_file_location(f"hardening_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


FREEZE = load_script("freeze_preregistration.py")
TRAIN_CELL = load_script("train_cell.py")
PRIVACY = load_script("audit_public_artifacts.py")


class FreezeContractTest(unittest.TestCase):
    def _write_config(self, payload: dict[str, object], directory: str) -> Path:
        path = Path(directory) / "pilot.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_config_rejects_scientific_matrix_and_threshold_drift(self) -> None:
        original = json.loads(FREEZE.CONFIG_PATH.read_text(encoding="utf-8"))

        def set_path(payload, *parts_and_value):
            *parts, value = parts_and_value
            current = payload
            for part in parts[:-1]:
                current = current[part]
            current[parts[-1]] = value

        mutations = {
            "arm": ("arms", "LG3", "grounded", False),
            "seeds": ("training", "seed_bases", [2026]),
            "folds": ("training", "folds", [0, 1, 2, 3]),
            "cells": ("training", "formal_cells", 59),
            "physical": ("training", "physical_batch_size", 2),
            "accumulation": ("training", "accumulation_steps", 16),
            "logical": ("training", "logical_batch_size", 16),
            "lambda": ("objective", "lambda_ftv", 0.2),
            "sigreg": ("objective", "sigreg_weight", 0.08),
            "steps": ("objective", "step_weights", [1.0, 1.0, 1.0]),
            "alphas": ("probes", "ridge_alphas", [0.1]),
            "endpoints": ("probes", "static_endpoints", ["T0", "macro"]),
            "scope": ("probes", "primary_scope", "observable_only"),
            "gate_a": (
                "gates",
                "A_LOCAL_STATE_WORKS",
                "static_macro_spearman_gain_each_seed_min",
                0.09,
            ),
            "gate_b": (
                "gates",
                "B_LOCAL_GLOBAL_ADDS_VALUE",
                "static_macro_spearman_gain_at_least_one_seed_min",
                0.01,
            ),
            "gate_c": (
                "gates",
                "C_GROUNDING_COMPATIBILITY",
                "candidate_delta_macro_spearman_gain_at_least_one_seed_min",
                0.01,
            ),
            "gate_d": (
                "gates",
                "D_OPTIMIZATION_SAFETY",
                "candidate_paired_folds_required",
                8,
            ),
            "operationalization": (
                "gate_operationalization",
                "thresholds_are_descriptive_not_statistical_significance",
                False,
            ),
            "selection": ("selection_rule", "if_A_pass_and_B_fail", "GAP"),
            "next_stage": (
                "next_stage_policy",
                "direct_FTV_plus_LD_from_this_pilot",
                True,
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            for label, mutation in mutations.items():
                with self.subTest(label=label):
                    changed = copy.deepcopy(original)
                    set_path(changed, *mutation)
                    path = self._write_config(changed, directory)
                    with mock.patch.object(FREEZE, "CONFIG_PATH", path):
                        with self.assertRaises(ValueError):
                            FREEZE.load_config()

    def test_upstream_inventory_is_exact_and_rejects_drift(self) -> None:
        expected = {
            "stage_a_sentinel_sha256",
            "data_contract_sha256",
            "g3_package_init_sha256",
            "g3_config_sha256",
            "g3_data_sha256",
            "g3_model_sha256",
            "g3_objective_sha256",
            "g3_ftv_transform_sha256",
            "stage_b_package_init_sha256",
            "stage_b_contracts_sha256",
            "stage_b_data_sha256",
            "stage_b_gate_sha256",
            "stage_b_inputs_sha256",
            "stage_b_targets_adapter_sha256",
            "stage_b_logical_training_sha256",
            "stage_b_upstream_sha256",
            "audited_pooling_package_init_sha256",
            "audited_pooling_sha256",
        }
        self.assertEqual(set(FREEZE.UPSTREAM_PATHS), expected)
        config = FREEZE.load_config()
        self.assertEqual(
            set(FREEZE.upstream_inventory(config)), set(FREEZE.UPSTREAM_PATHS.values())
        )
        changed = copy.deepcopy(config)
        changed["upstream"]["stage_b_gate_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "upstream hash mismatch"):
            FREEZE.upstream_inventory(changed)

    def test_lock_schema_fields_are_independently_fail_closed(self) -> None:
        config = FREEZE.load_config()
        payload = {
            **FREEZE.lock_declarations(config),
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "git": {"branch": "main", "head": "a" * 40},
            "code_and_plan_sha256": {},
            "upstream_sha256": {},
            "config_sha256": "c" * 64,
        }

        def git_value(*args: str) -> str:
            return "main" if args[0] == "branch" else "a" * 40

        with mock.patch.object(FREEZE, "git_value", side_effect=git_value):
            FREEZE.validate_lock_payload(payload, config)
            mutations = []
            for key, value in (
                ("schema_version", 2),
                ("experiment", "other"),
                ("formal_result_file_count_before_lock", 1),
                ("thresholds_are_not_statistical_significance", False),
                ("ftv_plus_ld_authorized_by_this_lock", True),
            ):
                changed = copy.deepcopy(payload)
                changed[key] = value
                mutations.append((key, changed))
            for key in ("decision_rules", "gate_operationalization", "matrix"):
                changed = copy.deepcopy(payload)
                changed[key] = {}
                mutations.append((key, changed))
            changed = copy.deepcopy(payload)
            changed["git"]["head"] = "invalid"
            mutations.append(("git", changed))
            for label, changed in mutations:
                with self.subTest(label=label), self.assertRaises(
                    (ValueError, RuntimeError)
                ):
                    FREEZE.validate_lock_payload(changed, config)

    def test_verify_allows_results_only_after_zero_result_lock(self) -> None:
        config = FREEZE.load_config()
        code = {"pilot.py": "a" * 64}
        upstream = {"upstream.py": "b" * 64}
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / "lock.json"
            payload = {
                **FREEZE.lock_declarations(config),
                "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                "git": {"branch": "main", "head": "a" * 40},
                "code_and_plan_sha256": code,
                "upstream_sha256": upstream,
                "config_sha256": FREEZE.file_sha256(FREEZE.CONFIG_PATH),
            }
            lock.write_text(json.dumps(payload), encoding="utf-8")

            def git_value(*args: str) -> str:
                return "main" if args[0] == "branch" else "a" * 40

            with mock.patch.multiple(
                FREEZE,
                LOCK_PATH=lock,
                code_inventory=mock.Mock(return_value=code),
                upstream_inventory=mock.Mock(return_value=upstream),
                result_files=mock.Mock(return_value=["pilot/checkpoints/result.pt"]),
                git_value=mock.Mock(side_effect=git_value),
            ):
                observed = FREEZE.verify()
            self.assertEqual(observed["current_result_files"], 1)
            self.assertEqual(observed["config_sha256"], payload["config_sha256"])


class FilesystemAndImportHardeningTest(unittest.TestCase):
    def test_containment_exclusive_claim_and_canonical_file(self) -> None:
        from lg_response_pilot import security

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            root.mkdir()
            claimed = security.claim_private_directory(
                root / "tag", root, label="test claim"
            )
            self.assertEqual(stat.S_IMODE(claimed.stat().st_mode), 0o700)
            with self.assertRaises(FileExistsError):
                security.claim_private_directory(claimed, root, label="test claim")
            with self.assertRaises(ValueError):
                security.resolve_contained_path(
                    Path(directory) / "outside", root, label="outside"
                )
            canonical = root / "canonical.json"
            alternate = root / "alternate.json"
            canonical.write_text("same", encoding="utf-8")
            alternate.write_text("same", encoding="utf-8")
            digest = security.file_sha256(canonical)
            with self.assertRaisesRegex(ValueError, "canonical"):
                security.require_canonical_file(
                    alternate, canonical, digest, digest, label="contract"
                )

    def test_cell_path_binds_identity_tag_and_matching_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoints"
            root.mkdir()
            output = root / "formal" / "seed_2026" / "LG3" / "fold_2"
            baseline = (
                root
                / "formal"
                / "seed_2026"
                / "LG0"
                / "fold_2"
                / "selection.json"
            )
            TRAIN_CELL._validate_cell_paths(
                output,
                baseline,
                arm="LG3",
                seed_base=2026,
                fold=2,
                checkpoint_root=root,
            )
            with self.assertRaisesRegex(ValueError, "same tag"):
                TRAIN_CELL._validate_cell_paths(
                    output,
                    root / "other" / "seed_2026" / "LG0" / "fold_2" / "selection.json",
                    arm="LG3",
                    seed_base=2026,
                    fold=2,
                    checkpoint_root=root,
                )
            with self.assertRaisesRegex(ValueError, "cell output"):
                TRAIN_CELL._validate_cell_paths(
                    root / "formal" / "seed_3026" / "LG3" / "fold_2",
                    baseline,
                    arm="LG3",
                    seed_base=2026,
                    fold=2,
                    checkpoint_root=root,
                )

    def test_package_is_lazy_and_missing_lock_imports_no_pilot_code(self) -> None:
        environment = dict(__import__("os").environ)
        environment["PYTHONPATH"] = str(SRC_ROOT)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        command = (
            "import sys; import lg_response_pilot; "
            "assert 'lg_response_pilot.model' not in sys.modules; "
            "assert 'lg_response_pilot.upstream' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", command], env=environment, check=True)

        scripts = (
            "train_cell.py",
            "run_matrix.py",
            "export_features.py",
            "run_probes.py",
            "run_postprocessing.py",
            "aggregate_results.py",
            "generate_figures.py",
        )
        original_import = __import__("builtins").__import__

        class MissingLock(Exception):
            pass

        for name in scripts:
            with self.subTest(script=name):
                saved_path = list(sys.path)
                saved = {
                    key: value
                    for key, value in sys.modules.items()
                    if key == "lg_response_pilot" or key.startswith("lg_response_pilot.")
                }
                for key in tuple(saved):
                    sys.modules.pop(key, None)
                fake_freeze = types.ModuleType("freeze_preregistration")
                fake_freeze.verify = mock.Mock(side_effect=MissingLock)

                def guarded_import(import_name, *args, **kwargs):
                    if import_name == "lg_response_pilot" or import_name.startswith(
                        "lg_response_pilot."
                    ):
                        raise AssertionError(
                            f"{name} imported pilot code before a passing lock"
                        )
                    return original_import(import_name, *args, **kwargs)

                try:
                    with mock.patch.dict(
                        sys.modules, {"freeze_preregistration": fake_freeze}
                    ), mock.patch("builtins.__import__", side_effect=guarded_import):
                        with self.assertRaises(MissingLock):
                            runpy.run_path(str(SCRIPTS_ROOT / name), run_name="__main__")
                finally:
                    for key in tuple(sys.modules):
                        if key == "lg_response_pilot" or key.startswith(
                            "lg_response_pilot."
                        ):
                            sys.modules.pop(key, None)
                    sys.modules.update(saved)
                    sys.path[:] = saved_path

    def test_public_scan_excludes_only_derived_bytecode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            experiment = repository / "pilot"
            scripts = experiment / "scripts"
            cache = scripts / "__pycache__"
            cache.mkdir(parents=True)
            (cache / "derived.pyc").write_bytes(b"derived")
            unknown = scripts / "unknown.bin"
            unknown.write_bytes(b"unknown")
            output = experiment / "metrics" / "privacy.json"
            output.parent.mkdir()
            with mock.patch.multiple(
                PRIVACY,
                EXPERIMENT_ROOT=experiment,
                REPO_ROOT=repository,
                OUTPUT=output,
            ):
                paths = PRIVACY.public_artifacts()
            self.assertNotIn((cache / "derived.pyc").resolve(), paths)
            self.assertIn(unknown.resolve(), paths)


class PostprocessingDataRevalidationTest(unittest.TestCase):
    def test_parent_rehashes_complete_cache_before_feature_export(self) -> None:
        module = load_script("run_postprocessing.py")
        sealed_value = str(module.SEALED_SRC.resolve())
        while sealed_value in sys.path:
            sys.path.remove(sealed_value)
        sys.path.insert(0, sealed_value)
        import c1b_stage_b.gate as sealed_gate
        import c1b_stage_b.inputs as sealed_inputs

        config_sha = FREEZE.file_sha256(FREEZE.CONFIG_PATH)
        sentinel_sha = FREEZE.file_sha256(module.DEFAULT_SENTINEL)
        contract_sha = FREEZE.file_sha256(module.DEFAULT_DATA_CONTRACT)
        preregistration = {
            "status": "PASS",
            "lock_sha256": "a" * 64,
            "config_sha256": config_sha,
            "upstream_sha256": {
                str(module.DEFAULT_SENTINEL.relative_to(module.REPO_ROOT)): sentinel_sha,
                str(module.DEFAULT_DATA_CONTRACT.relative_to(module.REPO_ROOT)): contract_sha,
            },
        }
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        checkpoint_root = Path(temporary.name)
        (checkpoint_root / "matrix_complete.json").write_text(
            "{}", encoding="utf-8"
        )
        args = types.SimpleNamespace(
            config=module.CONFIG_PATH,
            stage_a_sentinel=module.DEFAULT_SENTINEL,
            stage_a_sentinel_sha256=sentinel_sha,
            data_contract=module.DEFAULT_DATA_CONTRACT,
            data_contract_sha256=contract_sha,
            checkpoint_root=checkpoint_root,
            feature_root=module.PILOT_FEATURE_ROOT / "unit_rehash",
            probe_root=module.PILOT_PREDICTION_ROOT / "unit_rehash",
            devices="cuda:0,cuda:1,cuda:2",
            execute=False,
        )
        authorization = types.SimpleNamespace(sha256=sentinel_sha)
        loaded = types.SimpleNamespace(provenance={"schema_version": 1})
        with mock.patch.object(
            module, "verify_preregistration", return_value=preregistration
        ), mock.patch.object(module, "parse_args", return_value=args), mock.patch.object(
            module, "_validate_matrix", return_value={"status": "COMPLETE"}
        ), mock.patch.object(
            sealed_gate, "require_stage_a_go", return_value=authorization
        ), mock.patch.object(
            sealed_inputs.StageBDataPaths, "load", return_value=object()
        ), mock.patch.object(
            sealed_inputs, "load_stage_b_data", return_value=loaded
        ) as load_data, mock.patch(
            "sys.stdout", new_callable=io.StringIO
        ):
            module.main()
        self.assertTrue(load_data.call_args.kwargs["verify_cache_files"])


if __name__ == "__main__":
    unittest.main()

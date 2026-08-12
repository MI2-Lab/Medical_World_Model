from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"security_test_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FREEZE = load_script("freeze_preregistration.py")
PRIVACY = load_script("audit_public_artifacts.py")


class FreezeResultInventoryTest(unittest.TestCase):
    def test_only_result_root_gitkeep_is_exempt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            experiment = repository / "pilot"
            for name in FREEZE.RESULT_DIRS:
                root = experiment / name
                root.mkdir(parents=True)
                (root / ".gitkeep").write_text("", encoding="utf-8")
            nested = experiment / "metrics" / "nested"
            nested.mkdir()
            (nested / ".gitkeep").write_text("result", encoding="utf-8")
            with mock.patch.multiple(
                FREEZE, EXPERIMENT_ROOT=experiment, REPO_ROOT=repository
            ):
                self.assertEqual(FREEZE.result_files(), ["pilot/metrics/nested/.gitkeep"])


class PublicArtifactPrivacyTest(unittest.TestCase):
    def _roots(self, temporary: str) -> tuple[Path, Path]:
        repository = Path(temporary)
        experiment = repository / "pilot"
        experiment.mkdir()
        for name in set(PRIVACY.PUBLIC_ROOTS) | set(PRIVACY.PRIVATE_ROOTS):
            root = experiment / name
            root.mkdir()
            (root / ".gitkeep").write_text("", encoding="utf-8")
        return repository, experiment

    def test_exact_mode_identifiers_paths_figures_and_unknown_files_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository, experiment = self._roots(temporary)
            identifiers = repository / "identifiers.csv"
            identifiers.write_text(
                "patient_id,patient_token\nPRIVATE_PATIENT_123,PRIVATE_TOKEN_456\n",
                encoding="utf-8",
            )
            private = experiment / "checkpoints" / "too_executable.pt"
            private.write_bytes(b"checkpoint")
            private.chmod(0o700)
            figure = experiment / "figures" / "leak.svg"
            windows_private = "C:" + "\\" + "private" + "\\" + "asset"
            figure.write_text(
                f"<text>PRIVATE_TOKEN_456 {windows_private}</text>",
                encoding="utf-8",
            )
            unsupported = experiment / "metrics" / "unknown.bin"
            unix_private = ("/" + "tmp/private").encode("utf-8")
            unsupported.write_bytes(b"PRIVATE_PATIENT_123 " + unix_private)
            private_metric = experiment / "metrics" / "details.private.csv"
            private_metric.write_text("private", encoding="utf-8")
            private_metric.chmod(0o700)
            output = experiment / "metrics" / "gate.json"
            with mock.patch.multiple(
                PRIVACY,
                EXPERIMENT_ROOT=experiment,
                REPO_ROOT=repository,
                OUTPUT=output,
                IDENTIFIER_SOURCES=(identifiers,),
            ):
                denylist = PRIVACY.identifiers()
                self.assertTrue(
                    {"PRIVATE_PATIENT_123", "PRIVATE_TOKEN_456"}.issubset(denylist)
                )
                self.assertIn(
                    PRIVACY.hashlib.sha256(b"PRIVATE_PATIENT_123").hexdigest(),
                    denylist,
                )
                paths = PRIVACY.public_artifacts()
                identifier_hits, path_hits, unsupported_hits = (
                    PRIVACY.scan_public_artifacts(paths, denylist)
                )
                self.assertEqual(len(identifier_hits), 1)
                self.assertEqual(len(path_hits), 1)
                self.assertEqual(len(unsupported_hits), 1)
                permission_findings = PRIVACY.private_permission_findings()
                self.assertEqual(len(permission_findings), 2)
                self.assertEqual(
                    {row["reason"] for row in permission_findings},
                    {"mode_not_0600"},
                )
                private.chmod(0o600)
                private_metric.chmod(0o600)
                self.assertEqual(stat.S_IMODE(private.stat().st_mode), 0o600)
                self.assertEqual(PRIVACY.private_permission_findings(), [])
                self.assertNotIn(private_metric.resolve(), paths)

    def test_relative_module_paths_do_not_match_unix_absolute_paths(self) -> None:
        relative_modules = (
            "contracts/data/gate/inputs/targets and config/data/model/targets"
        )
        self.assertIsNone(PRIVACY.ABSOLUTE_PATH_PATTERN.search(relative_modules))
        self.assertIsNone(
            PRIVACY.ABSOLUTE_PATH_BYTES_PATTERN.search(relative_modules.encode())
        )
        self.assertIsNotNone(
            PRIVACY.ABSOLUTE_PATH_PATTERN.search(
                "private source: `" + "/" + "data/cache/file`"
            )
        )
        self.assertIsNotNone(
            PRIVACY.ABSOLUTE_PATH_BYTES_PATTERN.search(
                b"private source: " + b"/" + b"data/cache/file"
            )
        )
        _, path_hits, unsupported_hits = PRIVACY.scan_public_artifacts(
            [ROOT / "EXPERIMENT_PLAN.md"], set()
        )
        self.assertEqual(path_hits, [])
        self.assertEqual(unsupported_hits, [])

    def test_shebang_web_url_and_cuda_regex_are_not_absolute_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository, experiment = self._roots(temporary)
            script = experiment / "scripts" / "safe.py"
            script.write_text(
                "#!" + "/" + "usr/bin/env python3\n"
                "url = 'https://example.org/reference'\n"
                "device_pattern = r'cuda:\\d+'\n",
                encoding="utf-8",
            )
            with mock.patch.multiple(
                PRIVACY, EXPERIMENT_ROOT=experiment, REPO_ROOT=repository
            ):
                identifier_hits, path_hits, unsupported_hits = (
                    PRIVACY.scan_public_artifacts([script], set())
                )
            self.assertEqual(identifier_hits, [])
            self.assertEqual(path_hits, [])
            self.assertEqual(unsupported_hits, [])

    def test_parent_component_and_structural_identifier_fields_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository, experiment = self._roots(temporary)
            identifier_directory = experiment / "metrics" / "PRIVATE_TOKEN_456"
            identifier_directory.mkdir()
            csv_path = identifier_directory / "rows.csv"
            csv_path.write_text("metric,value\nscore,1\n", encoding="utf-8")
            json_path = experiment / "metrics" / "unsafe.json"
            json_path.write_text(
                json.dumps({"patient_id": "", "metric": 1}), encoding="utf-8"
            )
            uid_path = experiment / "metrics" / "uid.txt"
            uid_path.write_text(
                ".".join(("1", "2", "840", "113619", "2", "55", "3")),
                encoding="utf-8",
            )
            with mock.patch.multiple(
                PRIVACY, EXPERIMENT_ROOT=experiment, REPO_ROOT=repository
            ):
                findings, _, unsupported = PRIVACY.scan_public_artifacts(
                    [csv_path, json_path, uid_path], {"PRIVATE_TOKEN_456"}
                )
            self.assertEqual(len(findings), 3)
            self.assertEqual(unsupported, [])

    def test_locked_identifier_sources_are_rehashed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources: dict[str, Path] = {}
            contract: dict[str, object] = {"schema_version": 2}
            for index, relative in enumerate(PRIVACY.IDENTIFIER_RELATIVE_SOURCES):
                path_key, sha_key = PRIVACY.IDENTIFIER_SOURCE_CONTRACT_FIELDS[relative]
                path = root / f"source_{index}.csv"
                patient = f"PATIENT_{index}"
                if "cache_manifest" == path_key:
                    token = PRIVACY.hashlib.sha256(patient.encode("utf-8")).hexdigest()
                    path.write_text(
                        f"patient_id,cache_path\n{patient},{root / (token + '.npz')}\n",
                        encoding="utf-8",
                    )
                else:
                    path.write_text(f"patient_id\n{patient}\n", encoding="utf-8")
                sources[relative] = path.resolve()
                contract[path_key] = str(path.resolve())
                contract[sha_key] = PRIVACY.file_sha256(path)
            contract_path = root / "data_contract.json"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            sources[PRIVACY.DATA_CONTRACT_RELATIVE] = contract_path.resolve()
            contract_sha = PRIVACY.file_sha256(contract_path)
            preregistration = {
                "upstream_sha256": {PRIVACY.DATA_CONTRACT_RELATIVE: contract_sha}
            }
            with mock.patch.object(
                PRIVACY, "_private_source", side_effect=lambda relative: sources[relative]
            ):
                resolved, hashes, observed_contract = (
                    PRIVACY.verified_identifier_sources(preregistration)
                )
            self.assertEqual(resolved, tuple(sources[key] for key in PRIVACY.IDENTIFIER_RELATIVE_SOURCES))
            self.assertEqual(set(hashes), set(PRIVACY.IDENTIFIER_RELATIVE_SOURCES))
            self.assertEqual(observed_contract, contract_sha)

    def test_completeness_and_nested_private_layout_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository, experiment = self._roots(temporary)
            nested = experiment / "metrics" / "nested"
            nested.mkdir()
            private = nested / "rows.private.csv"
            private.write_text("metric,value\nscore,1\n", encoding="utf-8")
            with mock.patch.multiple(
                PRIVACY,
                EXPERIMENT_ROOT=experiment,
                REPO_ROOT=repository,
                OUTPUT=experiment / "metrics" / "gate.json",
                REQUIRED_PUBLIC_RESULT_FILES=("metrics/required.csv",),
            ):
                self.assertEqual(len(PRIVACY.private_layout_findings()), 1)
                completeness = PRIVACY.public_result_completeness_findings()
            self.assertEqual(completeness, [
                {"required_artifact": "metrics/required.csv", "reason": "missing"}
            ])

    def test_placeholder_results_and_force_tracked_private_file_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository, experiment = self._roots(temporary)
            (repository / ".gitignore").write_text("pilot/checkpoints/**\n", encoding="utf-8")
            PRIVACY.subprocess.run(
                ["git", "init", "-q"], cwd=repository, check=True
            )
            private = experiment / "checkpoints" / "forced.pt"
            private.write_bytes(b"private")
            private.chmod(0o600)
            PRIVACY.subprocess.run(
                ["git", "add", "-f", "--", "pilot/checkpoints/forced.pt"],
                cwd=repository,
                check=True,
            )
            fake_csv = experiment / "metrics" / "empty.csv"
            fake_csv.write_text("metric,value\n", encoding="utf-8")
            fake_png = experiment / "figures" / "fake.png"
            fake_png.write_bytes(b"not a png")
            fake_report = experiment / "reports" / "final_report.md"
            fake_report.write_text("placeholder", encoding="utf-8")
            fake_summary = experiment / "metrics" / "aggregation_summary.json"
            fake_summary.write_text('{"status": "COMPLETE"}\n', encoding="utf-8")
            required = (
                "metrics/aggregation_summary.json",
                "metrics/empty.csv",
                "figures/fake.png",
                "reports/final_report.md",
            )
            with mock.patch.multiple(
                PRIVACY,
                EXPERIMENT_ROOT=experiment,
                REPO_ROOT=repository,
                REQUIRED_PUBLIC_RESULT_FILES=required,
            ):
                hygiene = PRIVACY.private_git_hygiene_findings()
                content = PRIVACY.public_result_content_findings()
            self.assertEqual(
                hygiene,
                [{
                    "artifact_token": hygiene[0]["artifact_token"],
                    "reason": "private_artifact_is_tracked",
                }],
            )
            self.assertGreaterEqual(len(content), 4)
            self.assertIn(
                "invalid_or_stale_aggregation_chain",
                {row["reason"] for row in content},
            )

    def test_result_chain_must_bind_the_current_preregistration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository, experiment = self._roots(temporary)
            current_lock = "a" * 64
            current_config = "b" * 64
            stage_a = "c" * 64
            data_contract = "d" * 64
            summary = {
                "status": "COMPLETE",
                "formal_cells": 100,
                "seeds": [2026, 3026, 4026, 5026, 6026],
                "folds_per_seed": 5,
                "patient_level_outputs_private": True,
                "public_outputs_deidentified": True,
                "preregistration_lock": "PREREGISTRATION_LOCK.json",
                "preregistration_lock_sha256": "e" * 64,
                "config_sha256": current_config,
                "stage_a_sentinel_sha256": stage_a,
                "data_contract_sha256": data_contract,
                "data_provenance_sha256": "f" * 64,
                "artifact_sha256": {},
            }
            path = experiment / "metrics" / "aggregation_summary.json"
            path.write_text(json.dumps(summary), encoding="utf-8")
            preregistration = {
                "lock_sha256": current_lock,
                "config_sha256": current_config,
                "upstream_sha256": {
                    PRIVACY.STAGE_A_SENTINEL_RELATIVE: stage_a,
                    PRIVACY.DATA_CONTRACT_RELATIVE: data_contract,
                },
            }
            with mock.patch.multiple(
                PRIVACY,
                EXPERIMENT_ROOT=experiment,
                REPO_ROOT=repository,
                REQUIRED_PUBLIC_RESULT_FILES=(
                    "metrics/aggregation_summary.json",
                ),
            ):
                findings = PRIVACY.public_result_content_findings(preregistration)
            self.assertEqual(findings, [{
                "required_artifact": "metrics/aggregation_summary.json",
                "reason": "invalid_or_stale_aggregation_chain",
            }])
            path.write_text("[]\n", encoding="utf-8")
            with mock.patch.multiple(
                PRIVACY,
                EXPERIMENT_ROOT=experiment,
                REPO_ROOT=repository,
                REQUIRED_PUBLIC_RESULT_FILES=(
                    "metrics/aggregation_summary.json",
                ),
            ):
                malformed = PRIVACY.public_result_content_findings(preregistration)
            self.assertEqual(
                {row["reason"] for row in malformed},
                {"invalid_or_empty_content", "invalid_or_stale_aggregation_chain"},
            )

    def test_main_verifies_lock_before_writing_and_binds_lock_sha(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository, experiment = self._roots(temporary)
            identifiers = repository / "identifiers.csv"
            identifiers.write_text("patient_id\nPRIVATE_PATIENT_123\n", encoding="utf-8")
            output = experiment / "metrics" / "privacy_gate.json"
            common = {
                "EXPERIMENT_ROOT": experiment,
                "REPO_ROOT": repository,
                "OUTPUT": output,
                "IDENTIFIER_SOURCES": (identifiers,),
                "REQUIRED_PUBLIC_RESULT_FILES": (),
            }
            with mock.patch.multiple(PRIVACY, **common), mock.patch.object(
                PRIVACY,
                "verify_preregistration",
                side_effect=FileNotFoundError("lock missing"),
            ):
                with self.assertRaises(FileNotFoundError):
                    PRIVACY.main()
                self.assertFalse(output.exists())
            output.write_text('{"schema_version": 1, "status": "PASS"}\n', encoding="utf-8")
            with mock.patch.multiple(PRIVACY, **common), mock.patch.object(
                PRIVACY,
                "verify_preregistration",
                side_effect=RuntimeError("lock drifted"),
            ):
                with self.assertRaises(RuntimeError):
                    PRIVACY.main()
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8"))["status"],
                "IN_PROGRESS",
            )
            lock_sha = "a" * 64
            with mock.patch.multiple(PRIVACY, **common), mock.patch.object(
                PRIVACY,
                "verify_preregistration",
                return_value={"status": "PASS", "lock_sha256": lock_sha},
            ):
                PRIVACY.main()
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "PASS")
            self.assertEqual(payload["preregistration_lock_sha256"], lock_sha)
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o644)


if __name__ == "__main__":
    unittest.main()

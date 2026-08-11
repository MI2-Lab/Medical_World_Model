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
            figure.write_text(
                "<text>PRIVATE_TOKEN_456 C:\\private\\asset</text>",
                encoding="utf-8",
            )
            unsupported = experiment / "metrics" / "unknown.bin"
            unsupported.write_bytes(b"PRIVATE_PATIENT_123 /tmp/private")
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
                self.assertEqual(
                    denylist, {"PRIVATE_PATIENT_123", "PRIVATE_TOKEN_456"}
                )
                paths = PRIVACY.public_artifacts()
                identifier_hits, path_hits, unsupported_hits = (
                    PRIVACY.scan_public_artifacts(paths, denylist)
                )
                self.assertEqual(
                    {Path(row["artifact"]).name for row in identifier_hits},
                    {"leak.svg"},
                )
                self.assertEqual(
                    {Path(row["artifact"]).name for row in path_hits},
                    {"leak.svg"},
                )
                self.assertEqual(
                    {Path(row["artifact"]).name for row in unsupported_hits},
                    {"unknown.bin"},
                )
                self.assertEqual(
                    PRIVACY.private_permission_findings(),
                    [
                        "pilot/checkpoints/too_executable.pt",
                        "pilot/metrics/details.private.csv",
                    ],
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
            PRIVACY.ABSOLUTE_PATH_PATTERN.search("private source: `/data/cache/file`")
        )
        self.assertIsNotNone(
            PRIVACY.ABSOLUTE_PATH_BYTES_PATTERN.search(b"private source: /data/cache/file")
        )
        _, path_hits, unsupported_hits = PRIVACY.scan_public_artifacts(
            [ROOT / "EXPERIMENT_PLAN.md"], set()
        )
        self.assertEqual(path_hits, [])
        self.assertEqual(unsupported_hits, [])

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
            }
            with mock.patch.multiple(PRIVACY, **common), mock.patch.object(
                PRIVACY,
                "verify_preregistration",
                side_effect=FileNotFoundError("lock missing"),
            ):
                with self.assertRaises(FileNotFoundError):
                    PRIVACY.main()
                self.assertFalse(output.exists())
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

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from zero_overlap_audit.provenance import atomic_private_json  # noqa: E402


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


class PrivateArtifactPermissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_private_file_in_mixed_public_directory_preserves_parent_mode(self) -> None:
        metrics = self.root / "metrics"
        metrics.mkdir()
        os.chmod(metrics, 0o2755)
        output = metrics / "audit.private.json"

        atomic_private_json(output, {"private": True})

        self.assertEqual(_mode(metrics), 0o2755)
        self.assertEqual(_mode(output), 0o600)
        self.assertEqual(json.loads(output.read_text()), {"private": True})

    def test_explicit_private_tree_remains_owner_only(self) -> None:
        private_root = self.root / "private"
        nested = private_root / "nested"
        nested.mkdir(parents=True)
        os.chmod(private_root, 0o755)
        os.chmod(nested, 0o755)
        output = nested / "audit.json"

        atomic_private_json(output, {"private": True})

        self.assertEqual(_mode(private_root), 0o700)
        self.assertEqual(_mode(nested), 0o700)
        self.assertEqual(_mode(output), 0o600)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c1b_spatial_audit.contracts import UPSTREAM_ROOT  # noqa: E402
from c1b_spatial_audit.runtime import load_stage_b_bundle  # noqa: E402


class FormalRuntimeResolutionTests(unittest.TestCase):
    def test_formal_stage_b_package_wins_over_schema_v1_predecessor(self) -> None:
        import c1b_stage_b.inputs as inputs

        implementation = Path(inputs.__file__).resolve()
        self.assertTrue(implementation.is_relative_to(UPSTREAM_ROOT / "src"))
        self.assertIn("observability_manifest", inputs.StageBDataPaths.__dataclass_fields__)

    def test_frozen_bundle_loads_schema_v2_without_materializing_caches(self) -> None:
        authorization, _paths, data = load_stage_b_bundle(verify_cache_files=False)
        self.assertEqual(authorization.payload["status"], "GO")
        self.assertEqual(len(data.folds), 4040)
        self.assertEqual(len(data.ftv), 375)
        self.assertEqual(len(data.train_only_ids), 139)
        self.assertEqual(len(data.legacy_cache), 947)
        self.assertEqual(len(data.c1b_cache), 947)


if __name__ == "__main__":
    unittest.main()

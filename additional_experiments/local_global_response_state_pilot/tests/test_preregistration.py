from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "freeze_preregistration", ROOT / "scripts" / "freeze_preregistration.py"
)
assert SPEC is not None and SPEC.loader is not None
FREEZE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FREEZE)


class PreregistrationTest(unittest.TestCase):
    def test_config_and_upstream_hashes_are_closed(self) -> None:
        config = FREEZE.load_config()
        observed = FREEZE.upstream_inventory(config)
        self.assertEqual(set(observed), set(FREEZE.UPSTREAM_PATHS.values()))
        self.assertEqual(config["training"]["formal_cells"], 60)
        self.assertEqual(len(config["arms"]), 6)
        self.assertEqual(
            config["training"]["global_oom_fallback"],
            "not_authorized; stop and require a new preregistration and code revision",
        )
        plan = (ROOT / "EXPERIMENT_PLAN.md").read_text(encoding="utf-8")
        self.assertIn("本预注册不授权 2/16 或任何 fallback", plan)

    def test_code_inventory_covers_plan_config_and_scripts(self) -> None:
        inventory = FREEZE.code_inventory()
        suffixes = set(inventory)
        prefix = "additional_experiments/local_global_response_state_pilot/"
        self.assertIn(prefix + "EXPERIMENT_PLAN.md", suffixes)
        self.assertIn(prefix + "configs/pilot.json", suffixes)
        self.assertIn(prefix + "scripts/freeze_preregistration.py", suffixes)
        self.assertIn(prefix + "tests/test_preregistration.py", suffixes)

    def test_no_results_exist_before_lock(self) -> None:
        if FREEZE.LOCK_PATH.exists():
            self.skipTest("formal preregistration is already frozen")
        self.assertEqual(FREEZE.result_files(), [])


if __name__ == "__main__":
    unittest.main()

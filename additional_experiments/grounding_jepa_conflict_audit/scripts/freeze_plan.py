#!/usr/bin/env python3
"""在查看任何新结果前冻结 Grounding–JEPA audit 计划与核心实现。"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gjca.assets import (
    fold_transform,
    load_data_context,
    validate_checkpoint_grid,
)  # noqa: E402
from gjca.contracts import file_sha256  # noqa: E402
from gjca.freeze import freeze_plan  # noqa: E402


def main() -> None:
    context = load_data_context()
    for fold in range(5):
        fold_transform(context, fold)
    cells = validate_checkpoint_grid()
    path = freeze_plan()
    print(
        json.dumps(
            {
                "status": "ok",
                "selected_checkpoint_cells_verified": len(cells),
                "plan_freeze": str(path.relative_to(ROOT)),
                "plan_freeze_sha256": file_sha256(path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""在任何新 gradient forward 前运行既有 gain/degradation 相关。"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gjca.contracts import file_sha256  # noqa: E402
from gjca.phase_a import (  # noqa: E402
    RESAMPLING_BUNDLE,
    RESAMPLING_MANIFEST,
    load_resampling_bundle,
    write_phase_a_correlations,
)


def main() -> None:
    output = write_phase_a_correlations()
    _, bundle = load_resampling_bundle()
    print(
        json.dumps(
            {
                "status": "ok",
                "rows": 3,
                "output": str(output.relative_to(ROOT)),
                "output_sha256": file_sha256(output),
                "resampling_index_bundle_sha256": bundle["bundle_sha256"],
                "resampling_npz_file_sha256": file_sha256(RESAMPLING_BUNDLE),
                "resampling_manifest_sha256": file_sha256(RESAMPLING_MANIFEST),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

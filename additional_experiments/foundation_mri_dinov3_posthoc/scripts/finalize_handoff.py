#!/usr/bin/env python3
"""Attach the post-commit Git handoff using an exact empty argv."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = EXPERIMENT_ROOT / "src"


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = tuple(sys.argv[1:] if argv is None else argv)
    if raw_argv:
        raise ValueError("formal Git handoff finalizer requires the exact empty argv")
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))

    from foundation_mri_dinov3.handoff import finalize_handoff  # noqa: PLC0415

    result = finalize_handoff()
    print(
        "consumer=git_handoff;status=complete;"
        f"recovered={int(bool(result['recovered']))};private_inputs_read=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

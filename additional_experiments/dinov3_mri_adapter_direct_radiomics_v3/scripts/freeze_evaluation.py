#!/usr/bin/env python3
"""Freeze all representation artifacts before allowing pCR evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dinov3_rg.locking import freeze_evaluation_lock  # noqa: E402
from dinov3_rg.security import RepresentationReadSentinel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "checkpoints/formal")
    parser.add_argument("--state-root", type=Path, default=ROOT / "features/private/formal_states")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    RepresentationReadSentinel().install()
    print(freeze_evaluation_lock(args.checkpoint_root, args.state_root, overwrite=args.overwrite))


if __name__ == "__main__":
    main()

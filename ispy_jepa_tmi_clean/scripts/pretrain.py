#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corejepa.config import load_config
from corejepa.training.runner import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pCR-free CoRe-JEPA pretraining.")
    parser.add_argument("--config", default="configs/paper_v1.yaml")
    args = parser.parse_args()
    checkpoint = train(load_config(args.config))
    print(f"best checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()

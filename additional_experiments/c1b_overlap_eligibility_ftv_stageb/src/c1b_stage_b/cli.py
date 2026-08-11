"""Shared, deliberately small Stage B command-line contract."""

from __future__ import annotations

import argparse
from pathlib import Path

from .gate import DEFAULT_STAGE_A_SENTINEL, StageAAuthorization, require_stage_a_go
from .inputs import StageBDataPaths


def add_gate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stage-a-sentinel", type=Path, default=DEFAULT_STAGE_A_SENTINEL)


def add_data_contract_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-contract", type=Path, required=True)
    parser.add_argument("--data-contract-sha256", required=True)


def authorize(args: argparse.Namespace) -> StageAAuthorization:
    """This must be the first filesystem-reading action in each entry point."""

    return require_stage_a_go(args.stage_a_sentinel)


def data_paths(args: argparse.Namespace) -> StageBDataPaths:
    return StageBDataPaths.load(args.data_contract, args.data_contract_sha256)


def resolve_device(value: str):
    import torch

    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


__all__ = [
    "add_data_contract_arguments",
    "add_gate_arguments",
    "authorize",
    "data_paths",
    "resolve_device",
]

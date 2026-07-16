#!/usr/bin/env python3
"""Run a synthetic forward/loss/backward check without reading patient data."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch

from corejepa.config import LossConfig, ModelConfig
from corejepa.models import CoReJEPA
from corejepa.training.losses import PretrainingObjective


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="", help="Comma-separated CUDA ids; empty uses CPU.")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    gpu_ids = [int(value) for value in args.gpus.split(",") if value.strip()]
    device = torch.device(f"cuda:{gpu_ids[0]}") if gpu_ids and torch.cuda.is_available() else torch.device("cpu")
    config = ModelConfig(
        base_channels=4,
        latent_dim=32,
        predictor_depth=1,
        predictor_heads=4,
        predictor_mlp_dim=64,
        response_dim=16,
        response_hidden_dim=32,
        expert_hidden_dim=24,
        expert_gate_hidden_dim=24,
    )
    model: torch.nn.Module = CoReJEPA(config, condition_dim=25).to(device)
    if len(gpu_ids) > 1 and torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])
    objective = PretrainingObjective(LossConfig(), 16, torch.ones(6)).to(device)
    batch_size = args.batch_size
    image = torch.randn(batch_size, 4, 8, 8, 16, 16, device=device)
    geometry = torch.rand(batch_size, 4, 9, device=device)
    condition = torch.randn(batch_size, 3, 25, device=device)
    output = model(image, geometry, condition)
    loss, stats = objective(
        output,
        {
            "response_score": torch.randn(batch_size, 3, 1, device=device),
            "response_vector": torch.randn(batch_size, 3, 18, device=device),
            "routing_target": torch.arange(batch_size, device=device) % 6,
        },
    )
    loss.backward()
    print(
        f"device={device} data_parallel={isinstance(model, torch.nn.DataParallel)} "
        f"prediction={tuple(output.prediction.shape)} response_state={tuple(output.future_response_state.shape)} "
        f"loss={stats['loss']:.4f}"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c1b_stage_b.training import (  # noqa: E402
    TrainHyperparameters,
    run_logical_train_epoch,
)
from c1b_stage_b.upstream import DGRSObjective, DGRSOutput  # noqa: E402


class _ToyDataset:
    def __init__(self) -> None:
        self.patient_ids = tuple(f"P{index:02d}" for index in range(32))
        self.transformed_ftv: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for index, patient_id in enumerate(self.patient_ids):
            mask = np.asarray(
                [(index + visit) % 3 != 0 for visit in range(4)], dtype=bool
            )
            if index % 7 == 0:
                mask[:] = False
            target = np.asarray(
                [0.1 * index + 0.2 * visit for visit in range(4)],
                dtype=np.float32,
            )
            self.transformed_ftv[patient_id] = target, mask

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, index: int) -> dict[str, object]:
        generator = torch.Generator().manual_seed(1_000 + index)
        patient_id = self.patient_ids[index]
        target, mask = self.transformed_ftv[patient_id]
        return {
            "patient_id": patient_id,
            "image": torch.randn(4, 6, generator=generator),
            "ftv_target": torch.from_numpy(target),
            "ftv_mask": torch.from_numpy(mask),
        }


class _ToyModel(torch.nn.Module):
    def __init__(self, grounded: bool) -> None:
        super().__init__()
        torch.manual_seed(22)
        self.response = torch.nn.Linear(6, 8)
        self.project = torch.nn.Sequential(
            torch.nn.Linear(8, 8), torch.nn.GELU(), torch.nn.Linear(8, 8)
        )
        self.transition = torch.nn.Linear(8, 8)
        self.ftv_head = torch.nn.Linear(8, 1) if grounded else None
        self.register_buffer("target_matrix", torch.randn(6, 8))
        self.ema_calls = 0

    def encode_online(
        self, image: torch.Tensor, unused: None
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        response = self.response(image)
        return response, self.project(response), None

    def forward(self, image: torch.Tensor, unused: None) -> DGRSOutput:
        response, online, _ = self.encode_online(image, None)
        target = (image @ self.target_matrix).detach()
        prediction = (
            None
            if self.ftv_head is None
            else self.ftv_head(response).squeeze(-1)
        )
        return DGRSOutput(
            response,
            online,
            target,
            target,
            target[:, 1:],
            self.transition(online[:, :-1]),
            prediction,
            None,
        )

    def update_target(self, momentum: float) -> None:
        self.ema_calls += 1


class _CaptureSGD(torch.optim.SGD):
    def __init__(self, parameters: object) -> None:
        super().__init__(parameters, lr=0.0)
        self.captured: list[dict[int, torch.Tensor | None]] = []
        self.steps = 0

    def step(self, closure: object = None) -> object:
        self.captured.append(
            {
                id(parameter): (
                    None
                    if parameter.grad is None
                    else parameter.grad.detach().clone()
                )
                for group in self.param_groups
                for parameter in group["params"]
            }
        )
        self.steps += 1
        return super().step(closure)


class LogicalStageBTrainingTests(unittest.TestCase):
    def test_full_l1_and_l3_toy_epoch_match_exact_batch32_gradient(self) -> None:
        dataset = _ToyDataset()
        images = torch.stack([dataset[index]["image"] for index in range(32)])
        targets = torch.stack(
            [dataset[index]["ftv_target"] for index in range(32)]
        )
        masks = torch.stack([dataset[index]["ftv_mask"] for index in range(32)])
        direction_seed = (2026 * 1_000_003 + 1 * 10_007) % (2**63 - 1)
        expected_grounded = sum(
            bool(mask.any()) for _, mask in dataset.transformed_ftv.values()
        )

        for model_name, grounded, lambda_ftv in (
            ("G1", False, 0.0),
            ("G3", True, 0.25),
        ):
            with self.subTest(model_name=model_name):
                initial_model = _ToyModel(grounded)
                initial = copy.deepcopy(initial_model.state_dict())

                full_model = _ToyModel(grounded)
                full_model.load_state_dict(initial)
                full_objective = DGRSObjective(
                    model_name,
                    lambda_ftv,
                    sigreg_weight=0.09,
                    sigreg_projections=32,
                )
                output = full_model(images, None)
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(direction_seed)
                    full_loss, _ = full_objective(output, targets, masks)
                full_loss.backward()
                expected_gradient = {
                    name: parameter.grad.detach().clone()
                    for name, parameter in full_model.named_parameters()
                }

                model = _ToyModel(grounded)
                model.load_state_dict(initial)
                objective = DGRSObjective(
                    model_name,
                    lambda_ftv,
                    sigreg_weight=0.09,
                    sigreg_projections=32,
                )
                optimizer = _CaptureSGD(model.parameters())
                stats = run_logical_train_epoch(
                    model,
                    objective,
                    dataset,  # type: ignore[arg-type]
                    optimizer,
                    torch.device("cpu"),
                    (dataset.patient_ids,),
                    TrainHyperparameters(
                        physical_batch_size=4,
                        accumulation_steps=8,
                        workers=0,
                        max_grad_norm=1e9,
                    ),
                    effective_seed=2026,
                    epoch=1,
                )

                self.assertEqual(optimizer.steps, 1)
                self.assertEqual(model.ema_calls, 1)
                self.assertEqual(stats["optimizer_steps"], 1)
                self.assertEqual(stats["ema_updates"], 1)
                self.assertEqual(stats["physical_microbatches"], 8)
                self.assertEqual(
                    stats["grounded_patients"],
                    expected_grounded if grounded else 0,
                )
                for name, parameter in model.named_parameters():
                    observed = optimizer.captured[0][id(parameter)]
                    self.assertIsNotNone(observed)
                    torch.testing.assert_close(
                        observed,
                        expected_gradient[name],
                        rtol=2e-5,
                        atol=2e-6,
                    )


if __name__ == "__main__":
    unittest.main()

"""Matched follow-up donor pairs 的冻结模型 inference 与 prediction 导出。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .contracts import DECISION_POINTS, validate_prediction_frame
from .metrics import cosine_similarity, layer_norm_mse
from .perturbations import (
    predict_perturbed_context_against_native_target,
    replace_followups_with_donor,
)
from .readouts import FoldReadoutBundle, predict_readout_probability_matrix


PAIR_COLUMNS = (
    "recipient_patient_id",
    "donor_patient_id",
    "fold",
    "audit_repetition",
    "matching_distance",
)
TRANSITION_NAMES = ("T0->T1", "T1->T2", "T2->T3")


@dataclass(frozen=True)
class DonorSwapInference:
    mapping: pd.DataFrame
    response_state: np.ndarray
    native_response_state: np.ndarray
    latent_metrics: pd.DataFrame


class DonorPairDataset(Dataset):
    """用显式 patient-index map 包装现有 held-out dataset，不读取 outcome。"""

    def __init__(
        self,
        base: Dataset,
        mapping: pd.DataFrame,
        patient_index: Mapping[str, int],
        *,
        expected_fold: int,
    ) -> None:
        missing = sorted(set(PAIR_COLUMNS).difference(mapping.columns))
        if missing:
            raise ValueError(f"donor mapping 缺少列：{missing}")
        frame = mapping.copy().reset_index(drop=True)
        frame["recipient_patient_id"] = frame["recipient_patient_id"].astype(str)
        frame["donor_patient_id"] = frame["donor_patient_id"].astype(str)
        frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
        frame["audit_repetition"] = pd.to_numeric(
            frame["audit_repetition"], errors="raise"
        ).astype(int)
        frame["matching_distance"] = pd.to_numeric(
            frame["matching_distance"], errors="raise"
        ).astype(float)
        if not frame["fold"].eq(int(expected_fold)).all():
            raise ValueError("donor mapping 含非当前 held-out fold")
        if frame["recipient_patient_id"].eq(frame["donor_patient_id"]).any():
            raise ValueError("donor 不得等于 recipient")
        if frame.duplicated(
            ["recipient_patient_id", "audit_repetition"], keep=False
        ).any():
            raise ValueError("recipient/audit_repetition 主键重复")
        if (
            frame["audit_repetition"].le(0).any()
            or not np.isfinite(frame["matching_distance"]).all()
            or frame["matching_distance"].lt(0).any()
        ):
            raise ValueError("audit_repetition 或 matching_distance 无效")

        normalized_index = {str(key): int(value) for key, value in patient_index.items()}
        requested = set(frame["recipient_patient_id"]) | set(frame["donor_patient_id"])
        unknown = sorted(requested.difference(normalized_index))
        if unknown:
            raise ValueError(f"donor mapping patient 不在 held-out dataset：{unknown[:5]}")
        if len(set(normalized_index.values())) != len(normalized_index):
            raise ValueError("patient_index 含重复 dataset index")
        if any(index < 0 or index >= len(base) for index in normalized_index.values()):
            raise ValueError("patient_index 越界")
        self.base = base
        self.mapping = frame
        self.patient_index = normalized_index

    def __len__(self) -> int:
        return len(self.mapping)

    def _item(self, patient_id: str) -> dict[str, Any]:
        item = self.base[self.patient_index[patient_id]]
        if str(item.get("patient_id")) != patient_id:
            raise RuntimeError("patient_index 与 base dataset patient_id 不一致")
        return item

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.mapping.iloc[int(index)]
        recipient_id = str(row["recipient_patient_id"])
        donor_id = str(row["donor_patient_id"])
        recipient = self._item(recipient_id)
        donor = self._item(donor_id)
        perturbed = replace_followups_with_donor(
            recipient["image"],
            recipient["geometry"],
            recipient["condition"],
            donor_image=donor["image"],
            donor_geometry=donor["geometry"],
        )
        return {
            "pair_index": torch.tensor(index, dtype=torch.long),
            "recipient_patient_id": recipient_id,
            "donor_patient_id": donor_id,
            "audit_repetition": torch.tensor(
                int(row["audit_repetition"]), dtype=torch.long
            ),
            "matching_distance": torch.tensor(
                float(row["matching_distance"]), dtype=torch.float32
            ),
            "native_image": recipient["image"],
            "native_geometry": recipient["geometry"],
            "perturbed_image": perturbed.image,
            "perturbed_geometry": perturbed.geometry,
            "condition": perturbed.condition,
        }


def _require_eval(model: torch.nn.Module) -> None:
    if any(module.training for module in model.modules()):
        raise ValueError("donor inference 要求 model 全部处于 eval")


@torch.no_grad()
def run_donor_swap_inference(
    model: torch.nn.Module,
    loader: Any,
    *,
    mapping: pd.DataFrame,
    device: str | torch.device,
    fold: int,
    checkpoint: str,
) -> DonorSwapInference:
    """固定 recipient native EMA target，运行所有 donor pairs。"""

    _require_eval(model)
    device = torch.device(device)
    states: dict[int, np.ndarray] = {}
    native_states: dict[int, np.ndarray] = {}
    metric_frames: list[pd.DataFrame] = []
    seen: set[int] = set()
    for batch in loader:
        pair_indices = batch["pair_index"].detach().cpu().numpy().astype(int)
        if len(set(pair_indices.tolist())) != len(pair_indices) or seen.intersection(
            pair_indices.tolist()
        ):
            raise ValueError("donor loader 重复 pair_index")
        seen.update(pair_indices.tolist())
        native_image = batch["native_image"].to(device, non_blocking=True)
        native_geometry = batch["native_geometry"].to(device, non_blocking=True)
        perturbed_image = batch["perturbed_image"].to(device, non_blocking=True)
        perturbed_geometry = batch["perturbed_geometry"].to(device, non_blocking=True)
        condition = batch["condition"].to(device, non_blocking=True)
        native_output = model(native_image, native_geometry, condition)
        paired = predict_perturbed_context_against_native_target(
            model,
            native_image=native_image,
            native_geometry=native_geometry,
            perturbed_image=perturbed_image,
            perturbed_geometry=perturbed_geometry,
            condition=condition,
        )
        native_target = native_output.target.detach().cpu().numpy()
        paired_target = paired.native_target.detach().cpu().numpy()
        if not np.allclose(native_target, paired_target, rtol=1e-6, atol=1e-7):
            raise RuntimeError("donor audit 未固定 recipient native EMA target")
        native_prediction = native_output.prediction.detach().cpu().numpy()
        donor_prediction = paired.prediction.detach().cpu().numpy()
        native_state = native_output.future_response_state.detach().cpu().numpy()
        donor_state = (
            model.forecast_response(perturbed_geometry, condition).detach().cpu().numpy()
        )
        difference = donor_state - native_state
        for row, pair_index in enumerate(pair_indices):
            states[int(pair_index)] = donor_state[row]
            native_states[int(pair_index)] = native_state[row]

        metadata = pd.DataFrame(
            {
                "pair_index": np.repeat(pair_indices, 3),
                "recipient_patient_id": np.repeat(
                    [str(value) for value in batch["recipient_patient_id"]], 3
                ),
                "donor_patient_id": np.repeat(
                    [str(value) for value in batch["donor_patient_id"]], 3
                ),
                "fold": int(fold),
                "audit_repetition": np.repeat(
                    batch["audit_repetition"].detach().cpu().numpy().astype(int), 3
                ),
                "matching_distance": np.repeat(
                    batch["matching_distance"].detach().cpu().numpy().astype(float), 3
                ),
                "transition": np.tile(TRANSITION_NAMES, len(pair_indices)),
                "checkpoint": str(checkpoint),
            }
        )
        metadata["native_layer_norm_mse"] = layer_norm_mse(
            native_prediction, native_target
        ).reshape(-1)
        metadata["donor_layer_norm_mse"] = layer_norm_mse(
            donor_prediction, native_target
        ).reshape(-1)
        metadata["latent_error_change"] = (
            metadata["donor_layer_norm_mse"] - metadata["native_layer_norm_mse"]
        )
        metadata["native_cosine_similarity"] = cosine_similarity(
            native_prediction, native_target
        ).reshape(-1)
        metadata["donor_cosine_similarity"] = cosine_similarity(
            donor_prediction, native_target
        ).reshape(-1)
        metadata["response_state_mean_abs_change"] = np.abs(difference).mean(
            axis=-1
        ).reshape(-1)
        metadata["response_state_l2_change"] = np.linalg.norm(
            difference, axis=-1
        ).reshape(-1)
        metric_frames.append(metadata)

    expected = set(range(len(mapping)))
    if seen != expected:
        raise ValueError(
            f"donor loader 未完整覆盖 mapping：missing={len(expected-seen)}, extra={len(seen-expected)}"
        )
    ordered_states = np.stack([states[index] for index in range(len(mapping))]).astype(
        np.float32
    )
    ordered_native = np.stack(
        [native_states[index] for index in range(len(mapping))]
    ).astype(np.float32)
    latent = pd.concat(metric_frames, ignore_index=True).sort_values(
        ["pair_index", "transition"], kind="stable"
    )
    return DonorSwapInference(
        mapping=mapping.reset_index(drop=True).copy(),
        response_state=ordered_states,
        native_response_state=ordered_native,
        latent_metrics=latent.reset_index(drop=True),
    )


def donor_prediction_frame(
    bundle: FoldReadoutBundle,
    inference: DonorSwapInference,
    labels_by_patient: Mapping[str, int],
    *,
    checkpoint: str,
    audit_condition: str = "matched_followup_swap",
) -> pd.DataFrame:
    """将 donor pair states 转成含 donor/repetition provenance 的 prediction rows。"""

    mapping = inference.mapping
    if len(mapping) != len(inference.response_state):
        raise ValueError("donor mapping 与 response_state 行数不一致")
    if not mapping["fold"].eq(bundle.fold).all():
        raise ValueError("donor mapping fold 与 readout bundle 不一致")
    probabilities = predict_readout_probability_matrix(bundle, inference.response_state)
    rows: list[dict[str, Any]] = []
    for pair_index, row in mapping.iterrows():
        patient_id = str(row["recipient_patient_id"])
        if patient_id not in labels_by_patient:
            raise ValueError(f"缺少 recipient label：{patient_id}")
        label = int(labels_by_patient[patient_id])
        if label not in (0, 1):
            raise ValueError(f"recipient label 非 0/1：{patient_id}")
        for decision, decision_point in enumerate(DECISION_POINTS):
            probability = float(probabilities[pair_index, decision])
            threshold = float(bundle.thresholds[decision_point])
            rows.append(
                {
                    "patient_id": patient_id,
                    "fold": bundle.fold,
                    "decision_point": decision_point,
                    "audit_condition": audit_condition,
                    "y_true": label,
                    "predicted_probability": probability,
                    "predicted_label": int(probability >= threshold),
                    "threshold": threshold,
                    "checkpoint": str(checkpoint),
                    "donor_patient_id": str(row["donor_patient_id"]),
                    "repetition_id": int(row["audit_repetition"]),
                    "matching_distance": float(row["matching_distance"]),
                }
            )
    return validate_prediction_frame(pd.DataFrame(rows), require_donor=True)


__all__ = [
    "DonorPairDataset",
    "DonorSwapInference",
    "donor_prediction_frame",
    "run_donor_swap_inference",
]

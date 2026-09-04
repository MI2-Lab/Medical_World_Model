#!/usr/bin/env python3
"""Train one seed/fold/arm and export only private nested-OOF image logits."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bilateral_context.contracts import (  # noqa: E402
    atomic_json, canonical_sha256, fold_frame, primary_ids, private_token,
)
from bilateral_context.model import BilateralPCRModel  # noqa: E402

MANIFEST = ROOT.parents[1] / "additional_experiments/raw_spatial_pcr_ceiling/manifests/formal_input.private.csv"
TIMINGS = ("T0", "T0_T1", "T0_T2")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def outcome_table() -> pd.DataFrame:
    cols = ["patient_id", "label_pcr"]
    frame = pd.read_csv(MANIFEST, usecols=cols, dtype={"patient_id": str})
    ids = list(primary_ids())
    frame = frame.drop_duplicates("patient_id").set_index("patient_id").loc[ids]
    return frame


class FeatureDataset(Dataset):
    def __init__(self, ids: list[str], labels: np.ndarray, cache_dir: Path) -> None:
        self.ids = ids
        self.labels = labels.astype(np.float32)
        self.cache_dir = cache_dir

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        patient_id = self.ids[index]
        path = self.cache_dir / f"{private_token(patient_id)}.private.npz"
        with np.load(path, allow_pickle=False) as payload:
            global_summary = np.asarray(payload["global_summary"], dtype=np.float16)
            spatial_tokens = np.asarray(payload["spatial_tokens"], dtype=np.float16)
        return torch.from_numpy(global_summary), torch.from_numpy(spatial_tokens), patient_id


def loader(ids: list[str], labels: np.ndarray, cache_dir: Path, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(FeatureDataset(ids, labels, cache_dir), batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True, generator=generator)


def train_or_validate(model: BilateralPCRModel, data: DataLoader, labels_by_id: dict[str, float], device: torch.device, spatial: bool, optimizer: torch.optim.Optimizer | None, accumulation: int) -> float:
    model.train(optimizer is not None)
    total = 0.0
    count = 0
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    for step, (global_summary, spatial_tokens, ids) in enumerate(data):
        global_summary = global_summary.to(device, non_blocking=True).float()
        spatial_input = spatial_tokens.to(device, non_blocking=True).float() if spatial else None
        target = torch.tensor([labels_by_id[str(pid)] for pid in ids], dtype=torch.float32, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(global_summary, spatial_input)
            losses = [F.binary_cross_entropy_with_logits(output[f"logit_{timing}"].float(), target) for timing in TIMINGS]
            loss = sum(losses) / len(losses)
        if optimizer is not None:
            (loss / accumulation).backward()
            if (step + 1) % accumulation == 0 or (step + 1) == len(data):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite image training loss")
        total += float(loss.detach().cpu()) * len(ids)
        count += len(ids)
    return total / max(count, 1)


@torch.no_grad()
def predict(model: BilateralPCRModel, data: DataLoader, device: torch.device, spatial: bool) -> tuple[np.ndarray, list[str]]:
    model.eval()
    rows: dict[str, list[float]] = {}
    for global_summary, spatial_tokens, ids in data:
        global_summary = global_summary.to(device, non_blocking=True).float()
        spatial_input = spatial_tokens.to(device, non_blocking=True).float() if spatial else None
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(global_summary, spatial_input)
        for index, patient_id in enumerate(ids):
            rows[str(patient_id)] = [float(torch.sigmoid(output[f"logit_{timing}"][index]).float().cpu()) for timing in TIMINGS]
    ordered = sorted(rows)
    return np.asarray([rows[patient_id] for patient_id in ordered], dtype=np.float32), ordered


def train_inner(seed: int, fold: int, arm: str, inner_index: int, train_ids: list[str], val_ids: list[str], test_ids: list[str], labels: dict[str, float], device: torch.device) -> tuple[np.ndarray, list[str], np.ndarray, list[str], list[dict[str, float]], int]:
    spatial = arm == "B1_SPATIAL"
    seed_everything(seed + fold * 100 + inner_index * 10000 + (1 if spatial else 0))
    model = BilateralPCRModel(spatial=spatial).to(device)
    cache_dir = ROOT / "cache/bilateral_dino"
    batch_size = 4 if spatial else 16
    accumulation = 4 if spatial else 1
    train_loader = loader(train_ids, np.asarray([labels[x] for x in train_ids]), cache_dir, batch_size, True, seed + 17)
    val_loader = loader(val_ids, np.asarray([labels[x] for x in val_ids]), cache_dir, batch_size, False, seed + 19)
    test_loader = loader(test_ids, np.asarray([labels[x] for x in test_ids]), cache_dir, batch_size, False, seed + 23)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4 if spatial else 5e-5, weight_decay=1e-4)
    best_loss = float("inf")
    best_state = None
    best_epoch = -1
    bad = 0
    history = []
    for epoch in range(20):
        train_loss = train_or_validate(model, train_loader, labels, device, spatial, optimizer, accumulation)
        val_loss = train_or_validate(model, val_loader, labels, device, spatial, None, 1)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": val_loss})
        if val_loss < best_loss - 1e-7:
            best_loss = val_loss
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if epoch + 1 >= 10 and bad >= 4:
            break
    if best_state is None:
        raise RuntimeError("no finite checkpoint selected")
    model.load_state_dict(best_state)
    prediction, prediction_ids = predict(model, val_loader, device, spatial)
    test_prediction, test_prediction_ids = predict(model, test_loader, device, spatial)
    return prediction, prediction_ids, test_prediction, test_prediction_ids, history, best_epoch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--arm", choices=("B0_GLOBAL", "B1_SPATIAL"), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not (ROOT / "EXPLORATORY_EVALUATION_LOCK.json").is_file():
        raise SystemExit("training is locked until the complete cache is finalized and exploratory lock is created")
    decision_path = ROOT / "decision.json"
    if not decision_path.is_file() or json.loads(decision_path.read_text(encoding="utf-8")).get("decision") != "CONTINUE_STAGE_B":
        raise SystemExit("formal B0/B1 training is locked by the preregistered exploratory stop rule")
    outcome = outcome_table()
    labels = {str(pid): float(value) for pid, value in outcome["label_pcr"].items()}
    ff = fold_frame().set_index(["patient_id", "fold"])
    ids = list(primary_ids())
    split = {pid: str(ff.loc[(pid, args.fold), "split"]) for pid in ids}
    outer_pool = [pid for pid in ids if split[pid] in {"train", "val"}]
    outer_test = [pid for pid in ids if split[pid] == "test"]
    if len(outer_pool) < 30 or len(outer_test) < 10:
        raise ValueError("outer fold split unexpectedly small")
    y = np.asarray([labels[pid] for pid in outer_pool])
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=260812 + args.seed + args.fold)
    oof = {pid: np.full(3, np.nan, dtype=np.float32) for pid in outer_pool}
    test_predictions = []
    histories = []
    for inner_index, (train_idx, val_idx) in enumerate(inner.split(outer_pool, y)):
        train_ids = [outer_pool[index] for index in train_idx]
        val_ids = [outer_pool[index] for index in val_idx]
        prediction, prediction_ids, test_prediction, test_prediction_ids, history, best_epoch = train_inner(args.seed, args.fold, args.arm, inner_index, train_ids, val_ids, outer_test, labels, torch.device(args.device))
        histories.append({"inner": inner_index, "best_epoch": best_epoch, "history": history})
        for patient_id, values in zip(prediction_ids, prediction):
            oof[patient_id] = values
        if test_prediction_ids != outer_test:
            raise RuntimeError("test patient order changed")
        test_predictions.append(test_prediction)
    ordered_oof = np.asarray([oof[pid] for pid in outer_pool], dtype=np.float32)
    ordered_test = np.mean(test_predictions, axis=0).astype(np.float32)
    if not np.isfinite(ordered_oof).all() or not np.isfinite(ordered_test).all():
        raise FloatingPointError("nested OOF image logits are non-finite")
    output = ROOT / "predictions/private" / f"seed{args.seed}_fold{args.fold}_{args.arm}.private.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, outer_pool_tokens=np.asarray([private_token(pid) for pid in outer_pool]), outer_test_tokens=np.asarray([private_token(pid) for pid in outer_test]), inner_oof_probability=ordered_oof, outer_test_probability=ordered_test, test_label=np.asarray([labels[pid] for pid in outer_test], dtype=np.int8), seed=np.asarray(args.seed), fold=np.asarray(args.fold), arm=np.asarray(args.arm))
    atomic_json(ROOT / "predictions/private" / f"seed{args.seed}_fold{args.fold}_{args.arm}.json", {"status": "COMPLETE", "seed": args.seed, "fold": args.fold, "arm": args.arm, "outer_pool_n": len(outer_pool), "outer_test_n": len(outer_test), "inner_folds": 3, "best_epochs": [item["best_epoch"] for item in histories], "outcome_fields_read": ["label_pcr"], "clinical_fields_read": []})
    print(json.dumps({"status": "COMPLETE", "seed": args.seed, "fold": args.fold, "arm": args.arm, "outer_pool_n": len(outer_pool), "outer_test_n": len(outer_test)}))


if __name__ == "__main__":
    main()

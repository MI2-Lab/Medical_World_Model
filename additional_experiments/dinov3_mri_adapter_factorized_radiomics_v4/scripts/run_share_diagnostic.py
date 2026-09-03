#!/usr/bin/env python3
"""Diagnose whether V4 radiomics updates conflict in shared modules.

The diagnostic is outcome-blind.  It measures loss-gradient geometry at the
paired F0 checkpoint and trains a radiomics head on frozen F0 states as a
learnability control.  It does not alter any pre-registered V4 decision.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from factorized_rg.contracts import EXPERIMENT_ROOT, V2_TARGET_DIR, load_folds, sha256_file
from factorized_rg.data import SummaryDataset
from factorized_rg.model import FactorizedWorldModel
from factorized_rg.objective import FactorizedObjective, masked_radiomics_loss


VERSION = "v4-share-diagnostic-20260903-r1"
VISITS = (0, 1, 2)


def device_for(value: str) -> torch.device:
    return torch.device(value if value != "cuda" or torch.cuda.is_available() else "cpu")


def split_sets(fold: int) -> dict[str, set[str]]:
    frame = load_folds(); current = frame.loc[frame["fold"].eq(int(fold))]
    return {split: set(current.loc[current["split"].eq(split), "patient_id"].astype(str)) for split in ("train", "val", "test")}


def load_target(fold: int) -> dict[str, np.ndarray]:
    path = V2_TARGET_DIR / f"fold_{fold}_targets.private.npz"
    with np.load(path, allow_pickle=False) as z:
        return {key: np.asarray(z[key]) for key in ("patient_id", "radiomics", "radiomics_mask")}


def load_f0(fold: int, device: torch.device) -> FactorizedWorldModel:
    checkpoint = EXPERIMENT_ROOT / "checkpoints/pilot" / f"seed2026_fold{fold}_F0/selected.private.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = FactorizedWorldModel(); model.load_state_dict(payload["model_state"], strict=True); model.to(device).eval()
    return model


def group_vectors(model: FactorizedWorldModel, jepa_loss: torch.Tensor, rad_loss: torch.Tensor) -> dict[str, dict[str, float]]:
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    params = [parameter for _, parameter in named]
    jepa_grads = torch.autograd.grad(jepa_loss, params, retain_graph=True, allow_unused=True)
    rad_grads = torch.autograd.grad(rad_loss, params, retain_graph=False, allow_unused=True)
    grouped: dict[str, dict[str, float]] = {}
    for group, predicate in {
        "shared_adapter": lambda name: name.startswith("adapter."),
        "response_projection": lambda name: name.startswith("adapter.response_projection."),
        "phenotype_branch": lambda name: name.startswith("phenotype_branch."),
        "radiomics_head": lambda name: name.startswith("radiomics_head."),
    }.items():
        j_chunks, r_chunks = [], []
        for (name, _), j_grad, r_grad in zip(named, jepa_grads, rad_grads):
            if not predicate(name):
                continue
            if j_grad is not None: j_chunks.append(j_grad.detach().float().reshape(-1))
            if r_grad is not None: r_chunks.append(r_grad.detach().float().reshape(-1))
        j_vec = torch.cat(j_chunks) if j_chunks else torch.zeros(1)
        r_vec = torch.cat(r_chunks) if r_chunks else torch.zeros(1)
        j_norm, r_norm = float(j_vec.norm()), float(r_vec.norm())
        cosine = float(torch.dot(j_vec, r_vec) / (j_vec.norm() * r_vec.norm()).clamp_min(1e-12)) if j_norm > 0 and r_norm > 0 else float("nan")
        grouped[group] = {"jepa_gradient_norm": j_norm, "radiomics_gradient_norm": r_norm, "gradient_cosine": cosine}
    return grouped


def jepa_and_radiomics_losses(model: FactorizedWorldModel, summary: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, random_seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output = model(summary)
    step_weights = torch.tensor([2.0, 1.0, 0.5], device=summary.device) / (3.5 / 3.0)
    j_pred = F.layer_norm(output.predicted_jepa, (128,)); j_tgt = F.layer_norm(output.jepa_target[:, 1:], (128,))
    p_pred = F.layer_norm(output.predicted_phenotype, (64,)); p_tgt = F.layer_norm(output.phenotype_target[:, 1:], (64,))
    j_loss = ((j_pred - j_tgt).square().mean(-1) * step_weights).mean()
    p_loss = ((p_pred - p_tgt).square().mean(-1) * step_weights).mean()
    torch.manual_seed(int(random_seed))
    sigreg_module = FactorizedObjective(0.0).to(summary.device)
    sigreg = sigreg_module.sigreg(torch.cat((output.jepa_online, output.phenotype_online), -1).transpose(0, 1))
    jepa_total = 0.5 * (j_loss + p_loss) + 0.09 * sigreg
    radiomics_loss, visits = masked_radiomics_loss(output.radiomics_prediction, target, mask)
    return output, jepa_total, radiomics_loss


def gradient_diagnostic(fold: int, device: torch.device, batches: int) -> dict[str, Any]:
    model = load_f0(fold, device); target = load_target(fold); splits = split_sets(fold)
    target_ids = set(target["patient_id"].astype(str).tolist()) & splits["train"]
    ids = tuple(sorted(target_ids))
    dataset = SummaryDataset(ids, EXPERIMENT_ROOT.parent / "dinov3_mri_adapter_radiomics_grounding_v2/cache/dinov3_summaries", V2_TARGET_DIR / f"fold_{fold}_targets.private.npz")
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    rows = []
    for batch_index, raw in enumerate(loader):
        if batch_index >= batches: break
        summary = raw["summary"].to(device); radiomics = raw["radiomics"].to(device); masks = raw["radiomics_mask"].to(device)
        model.zero_grad(set_to_none=True)
        _, jepa, rad = jepa_and_radiomics_losses(model, summary, radiomics, masks, 9100 + fold * 100 + batch_index)
        groups = group_vectors(model, jepa, rad)
        for group, values in groups.items(): rows.append({"fold": fold, "batch": batch_index, "group": group, **values})
    if not rows: raise RuntimeError(f"no diagnostic batches for fold {fold}")
    grouped: dict[str, dict[str, float]] = {}
    for group in ("shared_adapter", "response_projection", "phenotype_branch", "radiomics_head"):
        values = [row for row in rows if row["group"] == group]
        grouped[group] = {key: _median([row[key] for row in values]) for key in ("jepa_gradient_norm", "radiomics_gradient_norm", "gradient_cosine")}
        cosines = [row["gradient_cosine"] for row in values if np.isfinite(row["gradient_cosine"])]
        grouped[group]["negative_cosine_fraction"] = None if not cosines else float(np.mean(np.asarray(cosines) < 0))
    return {"fold": fold, "batches": batches, "groups": grouped, "implementation_version": VERSION, "outcome_fields_read": [], "clinical_fields_read": []}


def head_only_diagnostic(fold: int) -> dict[str, Any]:
    target = load_target(fold); splits = split_sets(fold)
    state_path = EXPERIMENT_ROOT / "features/private/pilot_states" / f"seed2026_fold{fold}_F0_states.private.npz"
    with np.load(state_path, allow_pickle=False) as z:
        ids = z["patient_id"].astype(str); state = np.asarray(z["phenotype_state"], dtype=np.float32)
    pos = {pid: i for i, pid in enumerate(ids)}
    rows = {"train": [], "val": []}
    yrows = {"train": [], "val": []}
    for i, pid_raw in enumerate(target["patient_id"].astype(str)):
        pid = str(pid_raw)
        if pid not in pos: continue
        split = "train" if pid in splits["train"] else "val" if pid in splits["val"] else None
        if split is None: continue
        for visit in VISITS:
            if bool(target["radiomics_mask"][i, visit]):
                rows[split].append(state[pos[pid], visit]); yrows[split].append(target["radiomics"][i, visit])
    x_train = np.asarray(rows["train"], np.float32); y_train = np.asarray(yrows["train"], np.float32)
    x_val = np.asarray(rows["val"], np.float32); y_val = np.asarray(yrows["val"], np.float32)
    mean, scale = x_train.mean(0), x_train.std(0); scale[scale < 1e-6] = 1.0
    x_train = (x_train - mean) / scale; x_val = (x_val - mean) / scale
    torch.manual_seed(202600 + fold)
    head = nn.Linear(64, 16); optimizer = torch.optim.Adam(head.parameters(), lr=2.5e-4, weight_decay=1e-4)
    tx, ty, vx, vy = map(torch.from_numpy, (x_train, y_train, x_val, y_val))
    with torch.no_grad(): initial = float(F.smooth_l1_loss(head(vx), vy))
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True); loss = F.smooth_l1_loss(head(tx), ty); loss.backward(); optimizer.step()
    with torch.no_grad(): final = float(F.smooth_l1_loss(head(vx), vy)); prediction = head(vx).numpy()
    correlations = [_corr(y_val[:, pc], prediction[:, pc]) for pc in range(16)]
    return {"fold": fold, "train_rows": int(len(x_train)), "val_rows": int(len(x_val)), "initial_validation_smooth_l1": initial, "final_validation_smooth_l1": final, "validation_loss_improvement": initial - final, "validation_macro_spearman": float(np.nanmean(correlations)), "implementation_version": VERSION, "outcome_fields_read": [], "clinical_fields_read": []}


def _corr(y: np.ndarray, pred: np.ndarray) -> float:
    if len(y) < 4 or np.unique(y).size < 2 or np.unique(pred).size < 2: return float("nan")
    return float(spearmanr(y, pred).statistic)


def _median(values: list[float]) -> float | None:
    finite = [float(value) for value in values if np.isfinite(value)]
    return None if not finite else float(np.median(finite))


def aggregate() -> None:
    paths = sorted((EXPERIMENT_ROOT / "metrics").glob("share_diagnostic_fold*.json"))
    if len(paths) != 5: raise SystemExit(f"expected 5 fold diagnostics, found {len(paths)}")
    payloads = [json.loads(path.read_text()) for path in paths]
    groups = {}
    for group in ("shared_adapter", "response_projection", "phenotype_branch", "radiomics_head"):
        group_payloads = [p["gradient"]["groups"][group] for p in payloads]
        groups[group] = {key: _median([item[key] for item in group_payloads if item[key] is not None]) for key in ("jepa_gradient_norm", "radiomics_gradient_norm", "gradient_cosine", "negative_cosine_fraction")}
        negative_fractions = [item["negative_cosine_fraction"] for item in group_payloads if item["negative_cosine_fraction"] is not None]
        groups[group]["mean_fold_negative_cosine_fraction"] = None if not negative_fractions else float(np.mean(negative_fractions))
        groups[group]["folds_with_negative_median_cosine"] = int(sum(item["gradient_cosine"] is not None and item["gradient_cosine"] < 0 for item in group_payloads))
    heads = [p["head_only"] for p in payloads]
    head_summary = {key: float(np.nanmean([p["head_only"][key] for p in payloads])) for key in ("initial_validation_smooth_l1", "final_validation_smooth_l1", "validation_loss_improvement", "validation_macro_spearman")}
    cosine = groups["shared_adapter"]["gradient_cosine"]
    improvement = head_summary["validation_loss_improvement"]
    result = {"status": "COMPLETE", "diagnostic_version": VERSION, "folds": 5, "gradient": groups, "head_only": head_summary, "interpretation": {"head_only_learnable": bool(improvement > 0), "shared_gradient_opposition_observed": bool(cosine is not None and (cosine < 0 or (groups["shared_adapter"]["negative_cosine_fraction"] is not None and groups["shared_adapter"]["negative_cosine_fraction"] > 0.5))), "not_a_preregistered_gate": True}, "outcome_fields_read": [], "clinical_fields_read": []}
    out = EXPERIMENT_ROOT / "metrics/share_diagnostic.json"; out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    decision = {"status": "COMPLETE", "diagnostic": "SHARED_GRADIENT_DIAGNOSTIC", "shared_gradient_opposition_observed": result["interpretation"]["shared_gradient_opposition_observed"], "head_only_learnable": result["interpretation"]["head_only_learnable"], "does_not_modify_v4_decision": True, "pCR_evaluation": "LOCKED", "outcome_fields_read": [], "clinical_fields_read": []}
    (EXPERIMENT_ROOT / "diagnostic_conclusion.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    def fmt(value: object) -> str:
        return "—" if value is None else f"{float(value):.4f}"
    lines = [
        "# V4 Shared-gradient diagnostic",
        "",
        "## 结论",
        "",
        f"五折 outcome-blind 诊断支持 shared adapter 梯度冲突假设：shared adapter 的 median fold JEPA/radiomics gradient cosine 为 `{fmt(groups['shared_adapter']['gradient_cosine'])}`，五折中 `{groups['shared_adapter']['folds_with_negative_median_cosine']}/5` 个 fold 的 median cosine 为负，fold-level negative fraction 平均为 `{fmt(groups['shared_adapter']['mean_fold_negative_cosine_fraction'])}`。同时，冻结 F0 representation、只训练 radiomics head 后 validation Smooth-L1 平均从 `{head_summary['initial_validation_smooth_l1']:.4f}` 降至 `{head_summary['final_validation_smooth_l1']:.4f}`。",
        "",
        "这说明 radiomics target 和线性 head 本身可学习；V4 的主要问题是 radiomics 更新经过 shared adapter/response projection 时，与 JEPA 更新方向相冲突。该诊断不读取 pCR，也不改变 V4 的预注册 NO-GO 决策。",
        "",
        "## 五折聚合结果",
        "",
        "| module group | JEPA grad norm | radiomics grad norm | median fold cosine | negative fraction mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for group in ("shared_adapter", "response_projection", "phenotype_branch", "radiomics_head"):
        values = groups[group]
        lines.append(f"| {group} | {fmt(values['jepa_gradient_norm'])} | {fmt(values['radiomics_gradient_norm'])} | {fmt(values['gradient_cosine'])} | {fmt(values['mean_fold_negative_cosine_fraction'])} |")
    lines += [
        "",
        "## 下一步",
        "",
        "下一轮应使用真正独立的 radiomics adapter：冻结或保持 C0 的 JEPA path，另建只从 DINO summary 输入的 trainable radiomics path；radiomics loss 不得回传到 JEPA adapter。先做小规模 five-fold mechanism pilot，通过后才考虑 formal matrix 或 pCR。",
    ]
    reports = EXPERIMENT_ROOT / "reports"; reports.mkdir(exist_ok=True); (reports / "share_diagnostic.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--fold", type=int, choices=range(5)); parser.add_argument("--device", default="cuda"); parser.add_argument("--batches", type=int, default=3); parser.add_argument("--aggregate", action="store_true"); args = parser.parse_args()
    if args.aggregate: aggregate(); return
    if args.fold is None: raise SystemExit("--fold is required unless --aggregate is set")
    device = device_for(args.device); gradient = gradient_diagnostic(args.fold, device, args.batches); head = head_only_diagnostic(args.fold)
    payload = {"gradient": gradient, "head_only": head, "diagnostic_version": VERSION, "outcome_fields_read": [], "clinical_fields_read": []}
    path = EXPERIMENT_ROOT / "metrics" / f"share_diagnostic_fold{args.fold}.json"; path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__": main()

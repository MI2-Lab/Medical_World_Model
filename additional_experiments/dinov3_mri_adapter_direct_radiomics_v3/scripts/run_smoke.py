#!/usr/bin/env python3
"""Run one outcome-blind batch through the V3 warm-start contract."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dinov3_rg.contracts import TRAIN_ONLY_MANIFEST, V2_CHECKPOINT_ROOT, V2_SUMMARY_DIR, V2_TARGET_DIR, atomic_json  # noqa: E402
from dinov3_rg.data import FoldTargets, SummaryDataset, load_fold_frame, load_train_only_ids, split_patient_ids  # noqa: E402
from dinov3_rg.model import MRIAdapterWorldModel  # noqa: E402
from dinov3_rg.objective import DirectRadiomicsObjective  # noqa: E402
from dinov3_rg.security import RepresentationReadSentinel  # noqa: E402
from dinov3_rg.training import make_loader, reinitialize_radiomics_head, set_seed  # noqa: E402


def norm(module):
    values=[p.grad.detach().float().square().sum() for p in module.parameters() if p.grad is not None]
    return 0.0 if not values else float(torch.stack(values).sum().sqrt())


def main() -> None:
    RepresentationReadSentinel().install()
    if json.loads((ROOT / "metrics/preflight.json").read_text(encoding="utf-8"))["status"] != "PASS":
        raise SystemExit("preflight did not pass")
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fold=0; seed=2026; set_seed(seed)
    checkpoint=torch.load(V2_CHECKPOINT_ROOT / "seed2026_fold0_D1/selected.private.pt", map_location="cpu", weights_only=False)
    models=[]; head_hashes=[]
    for _ in range(2):
        model=MRIAdapterWorldModel(); model.load_state_dict(checkpoint["model_state"], strict=True)
        head_hashes.append(reinitialize_radiomics_head(model, 902026)); model.ftv_head.requires_grad_(False)
        models.append(model)
    model=models[0].to(device); objective=DirectRadiomicsObjective(0.25).to(device)
    splits=split_patient_ids(fold, load_train_only_ids(TRAIN_ONLY_MANIFEST), load_fold_frame())
    dataset=SummaryDataset(splits["train"][:32], V2_SUMMARY_DIR, FoldTargets.load(V2_TARGET_DIR / "fold_0_targets.private.npz"))
    batch=next(iter(make_loader(dataset, shuffle=False, seed=seed, batch_size=32, workers=0)))
    summary=batch["summary"].to(device); output=model(summary)
    loss, stats=objective(output, batch["ftv"].to(device), batch["ftv_mask"].to(device),
                          batch["radiomics"].to(device), batch["radiomics_mask"].to(device))
    loss.backward()
    gate={
        "schema_version":1,
        "status":"PASS",
        "shape":list(output.response_state.shape),
        "finite":bool(torch.isfinite(output.response_state).all() and torch.isfinite(loss)),
        "paired_head_initialization":len(set(head_hashes))==1,
        "adapter_gradient_positive":norm(model.adapter)>0,
        "response_projection_gradient_positive":norm(model.adapter.response_projection)>0,
        "radiomics_head_gradient_positive":norm(model.radiomics_head)>0,
        "ftv_head_gradient_zero":norm(model.ftv_head)==0,
        "ftv_objective_weight_zero":objective.ftv_weight==0.0,
        "t3_radiomics_mask_false":not bool(batch["radiomics_mask"][:,3].any()),
        "outcome_fields_read":[], "clinical_fields_read":[],
    }
    gate["status"]="PASS" if all(v for k,v in gate.items() if k not in {"schema_version","status","shape","outcome_fields_read","clinical_fields_read"}) else "FAIL"
    atomic_json(ROOT / "metrics/smoke.json", gate); print(gate)
    if gate["status"] != "PASS": raise SystemExit(1)


if __name__ == "__main__": main()

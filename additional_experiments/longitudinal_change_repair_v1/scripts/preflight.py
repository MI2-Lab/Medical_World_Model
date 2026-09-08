#!/usr/bin/env python3
"""Read-only data-contract and leakage checks before any training."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT.parents[0]/"radiomics_next_change"/"src")]
from lcr.data import bundle_from_config, FTVTransform, ChangeDataset, records
from rnc.data import split_ids

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",type=Path,default=ROOT/"configs/experiment.yaml"); args=parser.parse_args()
    config=yaml.safe_load(args.config.read_text()); bundle=bundle_from_config(config)
    actual=hashlib.sha256(Path(config["data"]["fold_manifest"]).read_bytes()).hexdigest()
    if actual != config["data"]["fold_manifest_sha256"]: raise ValueError("fold manifest hash mismatch")
    rows=[]
    for fold in range(5):
        splits=split_ids(bundle,fold); transform=FTVTransform.fit(bundle.raw_radiomics,splits["train"])
        dataset=ChangeDataset(records(bundle,splits["train"][:2]),bundle.raw_radiomics,transform)
        item=dataset[0]
        if set(item) != {"patient_id","image","static","static_mask","change","change_mask"}: raise AssertionError("data item leaks non-authorized field")
        if tuple(item["image"].shape) != (4,7,32,96,96): raise AssertionError("DCE7 image contract mismatch")
        if set(splits["train"]) & set(splits["val"]) or set(splits["train"]) & set(splits["test"]): raise AssertionError("fold split overlap")
        rows.append({"fold":fold,"train":len(splits["train"]),"val":len(splits["val"]),"test":len(splits["test"]),"ftv_train_median":transform.median,"first_image_shape":list(item["image"].shape)})
    out=ROOT/"private_results"/"preflight.json"; out.parent.mkdir(exist_ok=True); out.write_text(json.dumps({"status":"PASS","folds":rows},ensure_ascii=False,indent=2)); print(out)
if __name__=="__main__": main()

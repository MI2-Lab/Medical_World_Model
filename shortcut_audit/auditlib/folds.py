"""五折 patient-level manifest 的只读校验与索引转换。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


VALID_SPLITS = ("train", "val", "test")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_fold_manifest(
    frame: pd.DataFrame,
    *,
    expected_folds: Sequence[int] = tuple(range(5)),
    expected_patient_ids: Sequence[str] | None = None,
    expected_labels: Mapping[str, int] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """验证 long-format 五折 manifest，不改变原始 ``frame``。

    每名患者必须在每个 fold 恰有一行，并在五折中恰好一次属于 test。
    该约束同时阻止 patient-level 泄漏与不完整 OOF 汇总。
    """

    required = {"patient_id", "fold", "split", "label_pcr"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"fold manifest 缺少列：{missing}")
    output = frame.loc[:, ["patient_id", "fold", "split", "label_pcr"]].copy()
    if output.isna().any().any():
        raise ValueError("fold manifest 的关键列不得包含缺失值")
    output["patient_id"] = output["patient_id"].astype(str).str.strip()
    if (output["patient_id"] == "").any():
        raise ValueError("patient_id 不得为空")
    output["fold"] = pd.to_numeric(output["fold"], errors="raise").astype(int)
    output["split"] = output["split"].astype(str).str.strip().str.lower().replace(
        {"validation": "val", "valid": "val"}
    )
    output["label_pcr"] = pd.to_numeric(output["label_pcr"], errors="raise").astype(int)
    if not output["label_pcr"].isin((0, 1)).all():
        raise ValueError("label_pcr 必须为 0/1")
    invalid_splits = sorted(set(output["split"]).difference(VALID_SPLITS))
    if invalid_splits:
        raise ValueError(f"未知 split：{invalid_splits}")

    folds = tuple(sorted(output["fold"].unique().tolist()))
    if folds != tuple(expected_folds):
        raise ValueError(f"fold 集合为 {folds}，预期为 {tuple(expected_folds)}")
    duplicated = output.duplicated(["patient_id", "fold"], keep=False)
    if duplicated.any():
        examples = output.loc[duplicated, ["patient_id", "fold"]].head(5).to_dict("records")
        raise ValueError(f"同一患者在同一 fold 出现多次：{examples}")

    patient_sets = {
        int(fold): set(group["patient_id"])
        for fold, group in output.groupby("fold", sort=True)
    }
    reference = patient_sets[int(expected_folds[0])]
    if any(values != reference for values in patient_sets.values()):
        raise ValueError("各 fold 的患者集合不一致")
    test_count = output["split"].eq("test").groupby(output["patient_id"]).sum()
    if not test_count.eq(1).all():
        bad = test_count[~test_count.eq(1)].head(5).to_dict()
        raise ValueError(f"每名患者必须恰好一次进入 test：{bad}")
    label_count = output.groupby("patient_id")["label_pcr"].nunique()
    if not label_count.eq(1).all():
        bad = label_count[~label_count.eq(1)].head(5).to_dict()
        raise ValueError(f"患者 label 在 fold 间不一致：{bad}")

    if expected_patient_ids is not None:
        expected = {str(value) for value in expected_patient_ids}
        if reference != expected:
            raise ValueError(
                "manifest 与预期患者集合不一致："
                f"missing={len(expected - reference)}, extra={len(reference - expected)}"
            )
    if expected_labels is not None:
        observed = output.drop_duplicates("patient_id").set_index("patient_id")["label_pcr"]
        mismatches = [
            patient_id
            for patient_id, label in expected_labels.items()
            if patient_id not in observed.index or int(observed.loc[patient_id]) != int(label)
        ]
        if mismatches:
            raise ValueError(f"manifest 与预期 label 不一致，示例：{mismatches[:5]}")

    output = output.sort_values(["fold", "split", "patient_id"], kind="stable").reset_index(drop=True)
    counts = (
        output.groupby(["fold", "split"], sort=True)
        .agg(n=("patient_id", "size"), positives=("label_pcr", "sum"))
        .reset_index()
        .to_dict("records")
    )
    summary: dict[str, object] = {
        "n_rows": int(len(output)),
        "n_patients": int(output["patient_id"].nunique()),
        "folds": list(folds),
        "each_patient_test_once": True,
        "counts": counts,
    }
    return output, summary


def load_fold_manifest(
    path: str | Path,
    **validation_kwargs: object,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """读取并校验 manifest，同时记录内容哈希。"""

    path = Path(path).resolve()
    frame, summary = validate_fold_manifest(pd.read_csv(path), **validation_kwargs)
    summary = {**summary, "path": str(path), "sha256": _sha256(path)}
    return frame, summary


def held_out_assignments(frame: pd.DataFrame) -> pd.DataFrame:
    """返回每名患者唯一的 OOF fold/label 映射。"""

    validated, _ = validate_fold_manifest(frame)
    return (
        validated.loc[validated["split"].eq("test"), ["patient_id", "fold", "label_pcr"]]
        .sort_values("patient_id", kind="stable")
        .reset_index(drop=True)
    )


def fold_split_indices(
    frame: pd.DataFrame,
    patient_order: Sequence[str],
    fold: int,
) -> dict[str, list[int]]:
    """把一个 fold 的 patient IDs 映射到 checkpoint patient order 索引。"""

    validated, _ = validate_fold_manifest(frame)
    patient_order = [str(value) for value in patient_order]
    if len(patient_order) != len(set(patient_order)):
        raise ValueError("checkpoint patient_order 含重复 patient_id")
    index_by_id = {patient_id: index for index, patient_id in enumerate(patient_order)}
    selected = validated.loc[validated["fold"].eq(int(fold))]
    missing = sorted(set(selected["patient_id"]).difference(index_by_id))
    extra = sorted(set(index_by_id).difference(selected["patient_id"]))
    if missing or extra:
        raise ValueError(f"patient order 与 manifest 不一致：missing={len(missing)}, extra={len(extra)}")
    return {
        split: [index_by_id[value] for value in selected.loc[selected["split"].eq(split), "patient_id"]]
        for split in VALID_SPLITS
    }

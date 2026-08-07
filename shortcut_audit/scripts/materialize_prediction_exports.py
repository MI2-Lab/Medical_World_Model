#!/usr/bin/env python3
"""按审计类别导出逐 fold 与合并 OOF prediction/transition CSV。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd


AUDIT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = AUDIT_ROOT.parent
CLEAN_ROOT = REPOSITORY_ROOT / "ispy_jepa_tmi_clean"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"拒绝覆盖 prediction export：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, default=AUDIT_ROOT)
    parser.add_argument("--allow-export", action="store_true")
    args = parser.parse_args()
    if not args.allow_export:
        raise SystemExit("prediction export 未启动：必须显式添加 --allow-export")
    root = args.audit_root.resolve()
    destination = root / "predictions"
    if destination.exists():
        if not destination.is_dir():
            raise FileExistsError(f"prediction export 路径不是目录：{destination}")
        existing_files = [path for path in destination.rglob("*") if not path.is_dir()]
        if existing_files:
            raise FileExistsError(
                f"拒绝覆盖已有 prediction export：{existing_files[:3]}"
            )

    sys.path.insert(0, str(REPOSITORY_ROOT))
    sys.path.insert(0, str(CLEAN_ROOT))
    from shortcut_audit.auditlib.contracts import (  # pylint: disable=import-outside-toplevel
        validate_prediction_frame,
    )

    prediction_groups: dict[str, list[pd.DataFrame]] = {
        "native": [],
        "repeated_t0": [],
        "temporal_order": [],
        "followup_swap": [],
        "simplified_baselines": [],
    }
    copy_groups: list[pd.DataFrame] = []
    artifacts: list[dict[str, object]] = []
    created_files: list[Path] = []
    try:
        for fold in range(5):
            result = root / "results" / f"fold_{fold:02d}"
            donor = root / "donor_results" / f"fold_{fold:02d}"
            native = pd.read_csv(result / "predictions" / "native.csv")
            perturbed = pd.read_csv(result / "predictions" / "perturbations.csv")
            baselines = pd.read_csv(result / "predictions" / "baselines.csv")
            followup = pd.read_csv(donor / "predictions.csv")
            mapping = pd.read_csv(donor / "donor_mapping.csv")
            groups = {
                "native": native,
                "repeated_t0": perturbed.loc[
                    perturbed["audit_condition"].str.contains("repeated_t0")
                ].copy(),
                "temporal_order": perturbed.loc[
                    perturbed["audit_condition"].str.contains("temporal")
                ].copy(),
                "followup_swap": followup,
                "simplified_baselines": baselines,
            }
            for category, frame in groups.items():
                normalized = validate_prediction_frame(frame)
                if category == "followup_swap":
                    requested_mapping_columns = [
                        "recipient_patient_id",
                        "donor_patient_id",
                        "fold",
                        "audit_repetition",
                        "subtype",
                        "donor_subtype",
                        "treatment_family",
                        "donor_treatment_family",
                        "recipient_baseline_lesion_volume",
                        "donor_baseline_lesion_volume",
                    ]
                    missing = sorted(set(requested_mapping_columns).difference(mapping))
                    if missing:
                        raise ValueError(f"donor mapping 缺少预测导出列：{missing}")
                    donor_metadata = mapping.loc[:, requested_mapping_columns].rename(
                        columns={
                            "recipient_patient_id": "patient_id",
                            "audit_repetition": "repetition_id",
                        }
                    )
                    normalized = normalized.merge(
                        donor_metadata,
                        on=[
                            "patient_id",
                            "donor_patient_id",
                            "fold",
                            "repetition_id",
                        ],
                        how="left",
                        validate="many_to_one",
                    )
                    required_export_columns = [
                        "subtype",
                        "treatment_family",
                        "recipient_baseline_lesion_volume",
                    ]
                    if normalized[required_export_columns].isna().any().any():
                        raise ValueError("follow-up swap prediction 无法完整对齐 donor mapping")
                if not normalized["fold"].eq(fold).all():
                    raise ValueError(f"{category} fold_{fold:02d} 含错误 fold")
                prediction_groups[category].append(normalized)
                path = destination / category / f"fold_{fold:02d}.csv"
                _atomic_csv(normalized, path)
                created_files.append(path)
                artifacts.append(
                    {
                        "category": category,
                        "fold": fold,
                        "path": str(path.relative_to(root)),
                        "rows": len(normalized),
                        "sha256": _sha256(path),
                    }
                )
            copy = pd.read_csv(result / "latent" / "copy_current.csv")
            if not copy["fold"].eq(fold).all():
                raise ValueError(f"copy_current fold_{fold:02d} 含错误 fold")
            copy_groups.append(copy)
            copy_path = destination / "copy_current" / f"fold_{fold:02d}.csv"
            _atomic_csv(copy, copy_path)
            created_files.append(copy_path)
            artifacts.append(
                {
                    "category": "copy_current_transition_metrics",
                    "fold": fold,
                    "path": str(copy_path.relative_to(root)),
                    "rows": len(copy),
                    "sha256": _sha256(copy_path),
                }
            )

        for category, frames in prediction_groups.items():
            combined = pd.concat(frames, ignore_index=True)
            path = destination / category / "oof_all_folds.csv"
            _atomic_csv(combined, path)
            created_files.append(path)
            artifacts.append(
                {
                    "category": category,
                    "fold": "OOF",
                    "path": str(path.relative_to(root)),
                    "rows": len(combined),
                    "sha256": _sha256(path),
                }
            )
        copy_all = pd.concat(copy_groups, ignore_index=True)
        copy_path = destination / "copy_current" / "all_folds.csv"
        _atomic_csv(copy_all, copy_path)
        created_files.append(copy_path)
        artifacts.append(
            {
                "category": "copy_current_transition_metrics",
                "fold": "ALL",
                "path": str(copy_path.relative_to(root)),
                "rows": len(copy_all),
                "sha256": _sha256(copy_path),
            }
        )
        manifest = destination / "prediction_exports_manifest.json"
        manifest_content = (
            json.dumps(
                {
                    "schema_version": "shortcut_audit.prediction_exports.v1",
                    "note": (
                        "copy-current 是 transition-level latent metric；其余至少满足固定 "
                        "12 列 prediction contract。followup_swap 额外冗余了 subtype、"
                        "treatment family 和 recipient/donor baseline lesion volume。"
                    ),
                    "artifacts": artifacts,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        descriptor, name = tempfile.mkstemp(
            prefix=f".{manifest.name}.", suffix=".tmp", dir=destination
        )
        os.close(descriptor)
        temporary = Path(name)
        try:
            temporary.write_text(manifest_content)
            os.replace(temporary, manifest)
            created_files.append(manifest)
        finally:
            temporary.unlink(missing_ok=True)
    except BaseException:
        for path in reversed(created_files):
            path.unlink(missing_ok=True)
        raise
    print(destination / "prediction_exports_manifest.json")


if __name__ == "__main__":
    main()

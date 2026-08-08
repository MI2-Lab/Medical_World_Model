"""Grounding–JEPA conflict audit 的冻结合同与通用 I/O。"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


AUDIT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = AUDIT_ROOT.parents[1]
SOURCE_ROOT = REPO_ROOT / "additional_experiments" / "g3_multiseed_generalization"
SOURCE_SRC = SOURCE_ROOT / "src"

SEED_BASES = (2026, 3026, 4026, 5026, 6026)
FOLDS = (0, 1, 2, 3, 4)
SPLITS = ("train", "validation")
LAMBDA_FTV = 0.25
AUDIT_SEED = 20260808
BATCH_SIZE = 32
BATCHES_PER_SPLIT = 8
MINIMUM_FTV_PATIENTS = 8
EXPECTED_AUDIT_CONFIG_SHA256 = (
    "695791aa55c330377336642cfbfca83e1329f54ff8b632f0c04683839dbe7447"
)

GROUPS = (
    "encoder_overall",
    "encoder_stage_1",
    "encoder_stage_2",
    "encoder_stage_3",
    "encoder_stage_4",
    "response_projection",
    "all_shared",
)

EXPECTED_SOURCE_SHA256 = {
    "EXPERIMENT_PLAN.md": "394402aa8235b26f07b98a32426639a915bad80c53fc49cb053e7123e97ad06c",
    "PLAN_FREEZE.json": "7e4cb0ea26fce8f192a0e75b26365e13876dfe8b01a9f6d5f261efa9fb273dfc",
    "SOURCE_FREEZE.json": "5036c77f1d73bafedac16bb9837e38572cfd02d61a1506cd989506b836cdd05b",
    "src/dgrs/model.py": "ce39878a0fef5af1f92a86811faabbe73b39f57cdaf6d7580bbd65bd855d4ed9",
    "src/dgrs/training.py": "76f9108df0ca8c0ff69e514cff3bab1d5e316d946da60c5f530dd7b9706d3815",
    "src/dgrs/data.py": "15b4b68ad45c935e313b893b0ce849877311c98d6c5c0c45495e8e9200240943",
    "src/dgrs/targets.py": "28fbf66f93c8541dfa5ecc7ebcf65d4143a9a605b3ce98be48355d5ab679ffac",
    "src/dgrs/config.py": "4460ce3413e2cb936a6fd3cbb7f16224af3af286b6784933688cd12d0ec47516",
    "configs/base.yaml": "562bb525d04ba2f006f60a67ca61bfb06ea7de1006f1fd76fbfd095253225ff3",
    "configs/ftv_transform_fold_0.json": "8df48a908a5d56f76a2dd1a5f52b7189b03ce64e60743f856ef14afca07ebd5b",
    "configs/ftv_transform_fold_1.json": "6b582c2bb22e8208bc2e149eec032d179182fde212b94bcf6161bd274b38b4d4",
    "configs/ftv_transform_fold_2.json": "fcdf72ea26da1ff49efbdc937c78761e41d54640dae20289ac73a193e9cee23a",
    "configs/ftv_transform_fold_3.json": "a666b556e87c955214869547c6d54f083b8f975838c12461cc1158332532792c",
    "configs/ftv_transform_fold_4.json": "cb207a387900cc9ebc3deb7dca8e448bdbea083aae495af07fd11200008d6a9c",
    "metrics/final/training_stability_seed_fold.csv": "87887aaadeb21348fb4dee43e42cff8712295f3dfde0f6266c4d714c6644c041",
    "metrics/final/seed_fold_effects.csv": "9b4d674f0613856e2bcf6f286fea174a7b7f041867a97cb8f29a796a8aa87351",
    "metrics/final/probe_seed_fold_cell_metrics.csv": "4fe0afb68ec28dc57ae30063516ec1ad9fdaf362e241e722b943ca9df15efa1b",
    "metrics/final/input_manifest.csv": "f74bf4d6bc67767867a489ed2c5430e69e3b6d5b996f6f0e04a1a6e72828a2e0",
    "metrics/training/formal/training_matrix_manifest.json": "d53d04b11384ae4dff8a3d2e6be463f3333233a1dcc30de9d8bc49d994906a54",
    "metrics/acceptance_check.json": "fac2e12a3b356ea0295912ae40b60119a3178e148c09e3e06426068ab5e71da7",
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(payload: Any) -> str:
    return bytes_sha256(canonical_json_bytes(payload))


def load_audit_config() -> dict[str, Any]:
    path = AUDIT_ROOT / "configs" / "audit.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or int(payload.get("schema_version", -1)) != 1:
        raise ValueError("audit.yaml schema 非法")
    return payload


def assert_audit_config() -> dict[str, Any]:
    config_path = AUDIT_ROOT / "configs" / "audit.yaml"
    observed_sha = file_sha256(config_path)
    if observed_sha != EXPECTED_AUDIT_CONFIG_SHA256:
        raise ValueError(
            f"audit.yaml SHA 漂移: {observed_sha} != {EXPECTED_AUDIT_CONFIG_SHA256}"
        )
    payload = load_audit_config()
    grid = payload.get("grid", {})
    batches = payload.get("audit_batches", {})
    gradient = payload.get("gradient", {})
    statistics = payload.get("statistics", {})
    crossed = statistics.get("crossed_bootstrap", {})
    group_bootstrap = statistics.get("group_bootstrap", {})
    permutation = statistics.get("permutation", {})
    if (
        tuple(grid.get("seed_bases", ())) != SEED_BASES
        or tuple(grid.get("folds", ())) != FOLDS
        or str(grid.get("model", "")).upper() != "G3"
        or float(grid.get("lambda_ftv", -1)) != LAMBDA_FTV
        or int(batches.get("audit_seed", -1)) != AUDIT_SEED
        or tuple(batches.get("splits", ())) != SPLITS
        or int(batches.get("batches_per_split", -1)) != BATCHES_PER_SPLIT
        or int(batches.get("batch_size", -1)) != BATCH_SIZE
        or int(batches.get("minimum_ftv_patients", -1)) != MINIMUM_FTV_PATIENTS
        or tuple(gradient.get("groups", ())) != GROUPS
        or float(grid.get("lambda_ftv", -1)) != LAMBDA_FTV
        or int(statistics.get("analysis_seed", -1)) != AUDIT_SEED
        or statistics.get("run_is_only_inferential_unit") is not True
        or statistics.get("batch_pseudoreplication_forbidden") is not True
        or int(crossed.get("replicates", -1)) != 20_000
        or int(group_bootstrap.get("replicates", -1)) != 20_000
        or group_bootstrap.get("uses_crossed_bootstrap_indices") is not True
        or int(permutation.get("replicates", -1)) != 14_400
        or int(permutation.get("seed_level_orders", -1)) != 120
        or int(permutation.get("fold_level_orders", -1)) != 120
        or permutation.get("identity_included") is not True
        or permutation.get("plus_one_correction") is not False
    ):
        raise ValueError("audit.yaml 与代码冻结常量分叉")
    return payload


def repo_relative(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"路径不在 repository 内: {resolved}") from error


def atomic_json(path: str | Path, payload: Any, *, overwrite: bool = False) -> None:
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"拒绝覆盖已有文件: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_csv(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str] | None = None,
    *,
    overwrite: bool = False,
) -> None:
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"拒绝覆盖已有文件: {destination}")
    if not rows and fieldnames is None:
        raise ValueError("空 CSV 必须显式提供 fieldnames")
    names = list(fieldnames or rows[0].keys())
    for index, row in enumerate(rows):
        if set(row) != set(names):
            raise ValueError(f"CSV row {index} schema 不一致")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=names)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def assert_source_hashes() -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in EXPECTED_SOURCE_SHA256.items():
        path = SOURCE_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"冻结 source 缺失: {relative}")
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(f"冻结 source 漂移: {relative}: {actual} != {expected}")
        observed[relative] = actual
    return observed


def derive_stochastic_seed(fold: int, split: str, batch_index: int) -> int:
    if (
        fold not in FOLDS
        or split not in SPLITS
        or batch_index not in range(BATCHES_PER_SPLIT)
    ):
        raise ValueError("stochastic seed key 非法")
    split_offset = 0 if split == "train" else 10_000
    return AUDIT_SEED + fold * 100_000 + split_offset + batch_index


def ensure_no_patient_columns(columns: Iterable[str]) -> None:
    forbidden = {
        "patient_id",
        "trial_id",
        "pcr",
        "label_pcr",
        "treatment",
        "subtype",
        "clinical",
        "y_true",
        "y_pred",
    }
    lowered = {str(column).strip().lower() for column in columns}
    if overlap := forbidden & lowered:
        raise ValueError(f"公开 schema 含禁止列: {sorted(overlap)}")


__all__ = [
    "AUDIT_ROOT",
    "AUDIT_SEED",
    "BATCHES_PER_SPLIT",
    "BATCH_SIZE",
    "EXPECTED_SOURCE_SHA256",
    "EXPECTED_AUDIT_CONFIG_SHA256",
    "FOLDS",
    "GROUPS",
    "LAMBDA_FTV",
    "MINIMUM_FTV_PATIENTS",
    "REPO_ROOT",
    "SEED_BASES",
    "SOURCE_ROOT",
    "SOURCE_SRC",
    "SPLITS",
    "assert_source_hashes",
    "assert_audit_config",
    "atomic_csv",
    "atomic_json",
    "bytes_sha256",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "derive_stochastic_seed",
    "ensure_no_patient_columns",
    "file_sha256",
    "load_audit_config",
    "repo_relative",
]

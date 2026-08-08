#!/usr/bin/env python3
"""顺序运行 50 feature + 50 FTV probe + 50 pCR 资产，支持严格安全 resume。"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


DGRS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DGRS_ROOT / "src"))

from dgrs.features import (  # noqa: E402
    DEFAULT_FOLD_MANIFEST,
    EXPECTED_FOLD_MANIFEST_SHA256,
    MODELS,
    SEED_BASES,
    extract_model,
    file_sha256,
)
from dgrs.pcr import (  # noqa: E402
    PCR_READOUT_SEED,
    run_pcr_readouts,
    validate_pcr_asset,
)
from dgrs.probes import (  # noqa: E402
    TARGETS,
    _load_feature_asset,
    run_representation_probes,
    validate_probe_asset,
)


@dataclass(frozen=True)
class MatrixCell:
    seed_base: int
    fold: int
    model: str


CELLS = tuple(
    MatrixCell(seed_base, fold, model)
    for seed_base in SEED_BASES
    for fold in range(5)
    for model in MODELS
)
STAGES = ("features", "ftv_probes", "pcr")


def _json_event(**payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


def _exact_experiment_root(path: Path, expected: Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    expected = expected.resolve()
    if resolved != expected:
        raise ValueError(f"{label} 必须显式指向本实验目录 {expected}：{resolved}")
    return resolved


def _seed_roots(
    cell: MatrixCell,
    feature_root: Path,
    prediction_root: Path,
    metric_root: Path,
) -> dict[str, Path]:
    suffix = f"seed_{cell.seed_base}"
    return {
        "feature": feature_root / suffix,
        "probe_prediction": prediction_root / "representation_probes" / suffix,
        "probe_metric": metric_root / "representation_probes" / suffix,
        "pcr_prediction": prediction_root / "pcr_readouts" / suffix,
        "pcr_metric": metric_root / "pcr_readouts" / suffix,
    }


def _checkpoint_path(checkpoint_root: Path, cell: MatrixCell) -> Path:
    return (
        checkpoint_root
        / f"seed_{cell.seed_base}"
        / cell.model.lower()
        / f"fold_{cell.fold}"
        / "best.pt"
    )


def _stage_paths(
    stage: str,
    cell: MatrixCell,
    roots: dict[str, Path],
) -> tuple[Path, ...]:
    model_dir = Path(cell.model) / f"fold_{cell.fold}"
    if stage == "features":
        directory = roots["feature"] / model_dir
        return (
            directory / "observed_features.npz",
            directory / "extraction_metadata.json",
            directory / "feature_manifest_fragment.csv",
        )
    if stage == "ftv_probes":
        prediction = roots["probe_prediction"] / model_dir / "test_predictions.csv"
        metric = roots["probe_metric"] / model_dir
        return (prediction, metric / "selection_records.csv", metric / "summary.json")
    if stage == "pcr":
        prediction = roots["pcr_prediction"] / model_dir / "test_predictions.csv"
        metric = roots["pcr_metric"] / model_dir
        return (prediction, metric / "selection_records.csv", metric / "summary.json")
    raise ValueError(stage)


def _asset_state(paths: tuple[Path, ...]) -> str:
    exists = tuple(path.exists() for path in paths)
    if not any(exists):
        return "pending"
    if all(exists) and all(path.is_file() for path in paths):
        return "complete_candidate"
    raise FileExistsError(f"发现 partial/非文件资产，拒绝 resume：{paths}")


def _validate_feature(
    cell: MatrixCell,
    roots: dict[str, Path],
    fold_manifest: Path,
) -> None:
    _load_feature_asset(
        roots["feature"],
        cell.model,
        cell.fold,
        cell.seed_base,
        fold_manifest,
    )


def _validate_probe(
    cell: MatrixCell,
    roots: dict[str, Path],
    fold_manifest: Path,
) -> None:
    validate_probe_asset(
        model_name=cell.model,
        fold=cell.fold,
        seed_base=cell.seed_base,
        feature_root=roots["feature"],
        prediction_root=roots["probe_prediction"],
        metric_root=roots["probe_metric"],
        fold_manifest=fold_manifest,
    )


def _validate_pcr(
    cell: MatrixCell,
    roots: dict[str, Path],
    fold_manifest: Path,
) -> None:
    validate_pcr_asset(
        model_name=cell.model,
        fold=cell.fold,
        seed_base=cell.seed_base,
        feature_root=roots["feature"],
        prediction_root=roots["pcr_prediction"],
        metric_root=roots["pcr_metric"],
        fold_manifest=fold_manifest,
    )


VALIDATORS: dict[str, Callable[[MatrixCell, dict[str, Path], Path], None]] = {
    "features": _validate_feature,
    "ftv_probes": _validate_probe,
    "pcr": _validate_pcr,
}


def _preflight_outputs(
    *,
    resume: bool,
    feature_root: Path,
    prediction_root: Path,
    metric_root: Path,
    fold_manifest: Path,
) -> dict[tuple[str, MatrixCell], str]:
    states: dict[tuple[str, MatrixCell], str] = {}
    for stage in STAGES:
        for cell in CELLS:
            roots = _seed_roots(cell, feature_root, prediction_root, metric_root)
            paths = _stage_paths(stage, cell, roots)
            state = _asset_state(paths)
            if state != "pending" and not resume:
                raise FileExistsError(
                    "默认拒绝覆盖任何已有资产；如需安全续跑显式传 --resume：" f"{paths}"
                )
            if state == "complete_candidate":
                # 在产生任何新资产前，完整校验所有将被跳过的旧资产。
                VALIDATORS[stage](cell, roots, fold_manifest)
                state = "verified_complete"
            states[(stage, cell)] = state
    return states


def _self_test() -> dict[str, object]:
    if MODELS != ("G1", "G3") or TARGETS != ("ftv",):
        raise AssertionError("evaluation scope 不是锁定的 G1/G3 + FTV-only")
    if len(CELLS) != 50 or len(set(CELLS)) != 50:
        raise AssertionError("matrix cell 数不是 50")
    if PCR_READOUT_SEED != 2026:
        raise AssertionError("pCR readout seed 未锁定为 2026")
    example = CELLS[-1]
    roots = _seed_roots(
        example,
        DGRS_ROOT / "features",
        DGRS_ROOT / "predictions",
        DGRS_ROOT / "metrics",
    )
    if any(len(_stage_paths(stage, example, roots)) != 3 for stage in STAGES):
        raise AssertionError("每个 stage 必须有三个原子资产")
    with tempfile.TemporaryDirectory(prefix="g3ms-eval-matrix-selftest-") as directory:
        paths = tuple(Path(directory) / f"asset_{index}" for index in range(3))
        if _asset_state(paths) != "pending":
            raise AssertionError("空资产组未识别为 pending")
        paths[0].write_text("partial", encoding="utf-8")
        try:
            _asset_state(paths)
        except FileExistsError:
            partial_rejected = True
        else:
            raise AssertionError("partial 资产未被安全 resume 拒绝")
        for path in paths[1:]:
            path.write_text("complete", encoding="utf-8")
        if _asset_state(paths) != "complete_candidate":
            raise AssertionError("齐全资产组未进入验证候选状态")
    return {
        "status": "evaluation matrix synthetic contract self-test passed",
        "seed_bases": list(SEED_BASES),
        "folds": 5,
        "models": list(MODELS),
        "feature_jobs": len(CELLS),
        "ftv_probe_jobs": len(CELLS),
        "pcr_jobs": len(CELLS),
        "total_stage_jobs": len(CELLS) * len(STAGES),
        "pcr_readout_seed_base": PCR_READOUT_SEED,
        "resume_policy": "skip only complete hash/schema-verified assets",
        "partial_asset_rejected": partial_rejected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--feature-root", type=Path)
    parser.add_argument("--prediction-root", type=Path)
    parser.add_argument("--metric-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--fold-manifest", type=Path, default=DEFAULT_FOLD_MANIFEST)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="仅跳过三文件齐全且 hash/schema/provenance 全部验证通过的资产",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(_self_test(), ensure_ascii=False, indent=2))
        return
    required = (
        args.checkpoint_root,
        args.feature_root,
        args.prediction_root,
        args.metric_root,
        args.cache_root,
    )
    if any(value is None for value in required):
        parser.error(
            "正式模式必须显式提供 --checkpoint-root/--feature-root/"
            "--prediction-root/--metric-root/--cache-root"
        )
    if args.batch_size <= 0 or args.workers < 0:
        parser.error("--batch-size 必须为正数，--workers 必须非负")

    checkpoint_root = _exact_experiment_root(
        args.checkpoint_root, DGRS_ROOT / "checkpoints" / "formal", "checkpoint_root"
    )
    feature_root = _exact_experiment_root(
        args.feature_root, DGRS_ROOT / "features", "feature_root"
    )
    prediction_root = _exact_experiment_root(
        args.prediction_root, DGRS_ROOT / "predictions", "prediction_root"
    )
    metric_root = _exact_experiment_root(
        args.metric_root, DGRS_ROOT / "metrics", "metric_root"
    )
    cache_root = Path(args.cache_root).expanduser().resolve(strict=True)
    if not cache_root.is_dir():
        raise NotADirectoryError(cache_root)
    fold_manifest = Path(args.fold_manifest).expanduser().resolve(strict=True)
    if file_sha256(fold_manifest) != EXPECTED_FOLD_MANIFEST_SHA256:
        raise ValueError("fold manifest SHA 与锁定值不一致")
    checkpoints = {cell: _checkpoint_path(checkpoint_root, cell) for cell in CELLS}
    missing_checkpoints = [path for path in checkpoints.values() if not path.is_file()]
    if missing_checkpoints:
        raise FileNotFoundError(
            f"缺少 {len(missing_checkpoints)} 个显式 best.pt：{missing_checkpoints[:3]}"
        )

    states = _preflight_outputs(
        resume=args.resume,
        feature_root=feature_root,
        prediction_root=prediction_root,
        metric_root=metric_root,
        fold_manifest=fold_manifest,
    )
    completed = 0
    total = len(CELLS) * len(STAGES)
    for stage in STAGES:
        for cell in CELLS:
            roots = _seed_roots(cell, feature_root, prediction_root, metric_root)
            state = states[(stage, cell)]
            if state == "verified_complete":
                completed += 1
                _json_event(
                    stage=stage,
                    action="resume_skip_verified",
                    seed_base=cell.seed_base,
                    fold=cell.fold,
                    model=cell.model,
                    completed=completed,
                    total=total,
                )
                continue
            if stage == "features":
                extract_model(
                    model_name=cell.model,
                    fold=cell.fold,
                    seed_base=cell.seed_base,
                    checkpoint=checkpoints[cell],
                    device_name=args.device,
                    output_root=roots["feature"],
                    cache_root=cache_root,
                    fold_manifest=fold_manifest,
                    batch_size=args.batch_size,
                    workers=args.workers,
                    overwrite=False,
                )
            elif stage == "ftv_probes":
                run_representation_probes(
                    model_name=cell.model,
                    fold=cell.fold,
                    seed_base=cell.seed_base,
                    feature_root=roots["feature"],
                    prediction_root=roots["probe_prediction"],
                    metric_root=roots["probe_metric"],
                    fold_manifest=fold_manifest,
                    targets=("ftv",),
                    overwrite=False,
                )
            else:
                run_pcr_readouts(
                    model_name=cell.model,
                    fold=cell.fold,
                    seed_base=cell.seed_base,
                    feature_root=roots["feature"],
                    prediction_root=roots["pcr_prediction"],
                    metric_root=roots["pcr_metric"],
                    fold_manifest=fold_manifest,
                    overwrite=False,
                )
            VALIDATORS[stage](cell, roots, fold_manifest)
            completed += 1
            _json_event(
                stage=stage,
                action="created_and_verified",
                seed_base=cell.seed_base,
                fold=cell.fold,
                model=cell.model,
                completed=completed,
                total=total,
            )
    _json_event(
        status="evaluation_matrix_complete",
        feature_jobs=50,
        ftv_probe_jobs=50,
        pcr_jobs=50,
        pcr_readout_seed_base=PCR_READOUT_SEED,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""汇总五折 B--F 结果、执行 patient bootstrap 并生成八类审计图。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


AUDIT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = AUDIT_ROOT.parent
CLEAN_ROOT = REPOSITORY_ROOT / "ispy_jepa_tmi_clean"


def _required_files(results_root: Path, donor_root: Path) -> dict[str, list[Path]]:
    files: dict[str, list[Path]] = {
        "predictions": [],
        "copy": [],
        "paired": [],
        "donor": [],
    }
    for fold in range(5):
        main = results_root / f"fold_{fold:02d}"
        donor = donor_root / f"fold_{fold:02d}"
        files["predictions"].extend(
            main / "predictions" / name
            for name in ("native.csv", "perturbations.csv", "baselines.csv")
        )
        files["predictions"].append(donor / "predictions.csv")
        files["copy"].append(main / "latent" / "copy_current.csv")
        files["paired"].extend(
            main / "latent" / name
            for name in (
                "paired_repeated_t0_c1_mri_only.csv",
                "paired_repeated_t0_c2_full_image_derived.csv",
                "paired_temporal_t1_t2_swap.csv",
            )
        )
        files["donor"].append(donor / "latent_diagnostics.csv")
    missing = [str(path) for paths in files.values() for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"五折 reporting 输入不完整，缺少 {len(missing)} 个文件；示例={missing[:5]}"
        )
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=AUDIT_ROOT / "results")
    parser.add_argument("--donor-root", type=Path, default=AUDIT_ROOT / "donor_results")
    parser.add_argument("--output-root", type=Path, default=AUDIT_ROOT)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--replace-generated-reporting",
        action="store_true",
        help="仅用于原子替换 shortcut_audit 内已生成的汇总表与图。",
    )
    parser.add_argument(
        "--allow-reporting",
        action="store_true",
        help="必须显式提供；防止检查命令意外写入正式 metrics/figures。",
    )
    args = parser.parse_args()
    if not args.allow_reporting:
        raise SystemExit("汇总未启动：必须显式添加 --allow-reporting")
    if args.bootstrap <= 0:
        raise ValueError("--bootstrap 必须为正整数")

    sys.path.insert(0, str(REPOSITORY_ROOT))
    sys.path.insert(0, str(CLEAN_ROOT))
    from shortcut_audit.auditlib.reporting import (  # pylint: disable=import-outside-toplevel
        run_reporting_pipeline,
    )

    inputs = _required_files(args.results_root.resolve(), args.donor_root.resolve())
    result = run_reporting_pipeline(
        predictions=inputs["predictions"],
        output_root=args.output_root.resolve(),
        copy_latent_metrics=inputs["copy"],
        paired_perturbation_metrics=inputs["paired"],
        donor_metrics=inputs["donor"],
        expected_folds=range(5),
        n_bootstrap=args.bootstrap,
        seed=args.seed,
        overwrite=args.replace_generated_reporting,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "n_bootstrap": args.bootstrap,
                "seed": args.seed,
                "table_manifest": str(result.table_manifest_path.resolve()),
                "figure_manifest": str(result.figures.manifest_path.resolve()),
                "n_figures": int(len(result.figures.artifacts)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

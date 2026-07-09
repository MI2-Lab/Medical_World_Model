#!/usr/bin/env python3
"""Clean data-processing entry point for the I-SPY JEPA project.

The preprocessing scripts live inside this clean code tree under
``data_processing/preprocessing``. Paths to private datasets and local tools are
provided by CLI flags, environment variables, or an env file. The default mode
is a dry run. Add ``--execute`` to run a stage.
"""

from __future__ import annotations

import argparse
import os
import shutil
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


DATA_PROCESSING_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREPROCESSING_DIR = DATA_PROCESSING_ROOT / "preprocessing"
DEFAULT_DATA_ROOT = DATA_PROCESSING_ROOT / "data"
DEFAULT_METADATA_DIR = DATA_PROCESSING_ROOT / "metadata"


STAGES = (
    "check",
    "ispy2-dicom",
    "ispy1-dicom",
    "clinical",
    "mri-nact",
    "dce-timing-audit",
    "all",
)


@dataclass(frozen=True)
class Paths:
    preprocessing_dir: Path
    ispy2_raw_root: Path
    ispy1_raw_root: Path
    ispy2_preprocessed_root: Path
    ispy1_preprocessed_root: Path
    dcm2niix: Path
    breastdcedl_metadata_csv: Path

    @property
    def ispy2_clinical_xlsx(self) -> Path:
        return self.ispy2_raw_root / "ISPY2-Imaging-Cohort-1-Clinical-Data.xlsx"

    @property
    def ispy2_mri_nact_xlsx(self) -> Path:
        return self.ispy2_raw_root / "Multi-feature-MRI-NACT-Data.xlsx"

    @property
    def ispy1_clinical_xlsx(self) -> Path:
        return self.ispy1_raw_root / "I-SPY-1-All-Patient-Clinical-and-Outcome-Data.xlsx"

    @property
    def audit_dir(self) -> Path:
        return self.ispy2_preprocessed_root / "_audits"


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--env-file", type=Path, default=None)
    pre_args, _ = pre_parser.parse_known_args()
    if pre_args.env_file is not None:
        load_env_file(pre_args.env_file)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=pre_args.env_file,
        help="Optional KEY=VALUE file with local data/tool paths.",
    )
    parser.add_argument("--stage", choices=STAGES, default="check")
    parser.add_argument("--execute", action="store_true", help="Run commands instead of printing them.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument(
        "--preprocessing-dir",
        type=Path,
        default=env_path("PREPROCESSING_DIR", DEFAULT_PREPROCESSING_DIR),
        help="Directory containing the preprocessing scripts. Defaults to this clean repo copy.",
    )
    parser.add_argument("--ispy2-raw-root", type=Path, default=env_path("ISPY2_RAW_ROOT", DEFAULT_DATA_ROOT / "raw" / "I-SPY2"))
    parser.add_argument("--ispy1-raw-root", type=Path, default=env_path("ISPY1_RAW_ROOT", DEFAULT_DATA_ROOT / "raw" / "I-SPY1"))
    parser.add_argument(
        "--ispy2-preprocessed-root",
        type=Path,
        default=env_path("ISPY2_PREPROCESSED_ROOT", DEFAULT_DATA_ROOT / "preprocessed" / "I-SPY2"),
    )
    parser.add_argument(
        "--ispy1-preprocessed-root",
        type=Path,
        default=env_path("ISPY1_PREPROCESSED_ROOT", DEFAULT_DATA_ROOT / "preprocessed" / "I-SPY1"),
    )
    parser.add_argument("--dcm2niix", type=Path, default=env_path("DCM2NIIX", shutil.which("dcm2niix") or "dcm2niix"))
    parser.add_argument(
        "--breastdcedl-metadata-csv",
        type=Path,
        default=env_path("BREASTDCEDL_METADATA_CSV", DEFAULT_METADATA_DIR / "BreastDCEDL_metadata_min_crop.csv"),
    )
    return parser.parse_args()


def env_path(name: str, default: Path | str) -> Path:
    value = os.environ.get(name)
    if value:
        return Path(value).expanduser()
    return Path(default).expanduser()


def load_env_file(path: Path) -> None:
    path = path.expanduser()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid env-file line {line_number}: {raw_line}")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            raise ValueError(f"Invalid env-file line {line_number}: {raw_line}")
        os.environ[key] = value


def build_paths(args: argparse.Namespace) -> Paths:
    return Paths(
        preprocessing_dir=args.preprocessing_dir.expanduser(),
        ispy2_raw_root=args.ispy2_raw_root.expanduser(),
        ispy1_raw_root=args.ispy1_raw_root.expanduser(),
        ispy2_preprocessed_root=args.ispy2_preprocessed_root.expanduser(),
        ispy1_preprocessed_root=args.ispy1_preprocessed_root.expanduser(),
        dcm2niix=args.dcm2niix.expanduser(),
        breastdcedl_metadata_csv=args.breastdcedl_metadata_csv.expanduser(),
    )


def env_for_paths(paths: Paths) -> dict[str, str]:
    env = os.environ.copy()
    if paths.dcm2niix.parent != Path("."):
        dcm2niix_dir = str(paths.dcm2niix.parent)
        env["PATH"] = dcm2niix_dir + os.pathsep + env.get("PATH", "")
    env["DCM2NIIX"] = str(paths.dcm2niix)
    env["ISPY2_RAW_ROOT"] = str(paths.ispy2_raw_root)
    env["ISPY1_RAW_ROOT"] = str(paths.ispy1_raw_root)
    env["ISPY2_PREPROCESSED_ROOT"] = str(paths.ispy2_preprocessed_root)
    env["ISPY1_PREPROCESSED_ROOT"] = str(paths.ispy1_preprocessed_root)
    env["BREASTDCEDL_METADATA_CSV"] = str(paths.breastdcedl_metadata_csv)
    env["PREPROCESSING_DIR"] = str(paths.preprocessing_dir)
    return env


def quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def run_or_print(cmd: list[str], paths: Paths, execute: bool) -> None:
    print("$ " + quote_cmd(cmd))
    if not execute:
        return
    subprocess.run(cmd, check=True, env=env_for_paths(paths))


def script(paths: Paths, name: str) -> str:
    return str(paths.preprocessing_dir / name)


def command_ispy2_dicom(paths: Paths, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        script(paths, "batch_preprocess_ispy2.py"),
        "--raw-root",
        str(paths.ispy2_raw_root),
        "--output-root",
        str(paths.ispy2_preprocessed_root),
        "--workers",
        str(args.workers),
    ]
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def command_ispy1_dicom(paths: Paths, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        script(paths, "preprocess_ispy1.py"),
        "--raw-root",
        str(paths.ispy1_raw_root),
        "--output-root",
        str(paths.ispy1_preprocessed_root),
        "--clinical-xlsx",
        str(paths.ispy1_clinical_xlsx),
        "--dcm2niix",
        str(paths.dcm2niix),
    ]
    if args.limit is not None:
        cmd.extend(["--max-patients", str(args.limit)])
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def command_clinical(paths: Paths) -> list[str]:
    return [
        sys.executable,
        script(paths, "extract_clinical_labels.py"),
        "--clinical-xlsx",
        str(paths.ispy2_clinical_xlsx),
        "--output-root",
        str(paths.ispy2_preprocessed_root),
    ]


def command_mri_nact(paths: Paths) -> list[str]:
    return [
        sys.executable,
        script(paths, "extract_mri_nact_features.py"),
        "--feature-xlsx",
        str(paths.ispy2_mri_nact_xlsx),
        "--output-root",
        str(paths.ispy2_preprocessed_root),
    ]


def command_dce_timing_audit(paths: Paths, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        script(paths, "audit_dce_dicom_timing.py"),
        "--preprocessed-root",
        str(paths.ispy2_preprocessed_root),
        "--breastdcedl-metadata-csv",
        str(paths.breastdcedl_metadata_csv),
        "--output-csv",
        str(paths.audit_dir / "dce_dicom_timing_audit.csv"),
        "--summary-json",
        str(paths.audit_dir / "dce_dicom_timing_audit_summary.json"),
        "--num-workers",
        str(args.workers),
    ]
    if args.limit is not None:
        cmd.extend(["--max-patients", str(args.limit)])
    return cmd


def path_status(path: Path) -> str:
    if path.exists():
        kind = "dir" if path.is_dir() else "file"
        return f"OK ({kind})"
    return "MISSING"


def print_check(paths: Paths) -> None:
    print("Configured paths")
    for label, path in [
        ("preprocessing_dir", paths.preprocessing_dir),
        ("ispy2_raw_root", paths.ispy2_raw_root),
        ("ispy1_raw_root", paths.ispy1_raw_root),
        ("ispy2_preprocessed_root", paths.ispy2_preprocessed_root),
        ("ispy1_preprocessed_root", paths.ispy1_preprocessed_root),
        ("dcm2niix", paths.dcm2niix),
        ("breastdcedl_metadata_csv", paths.breastdcedl_metadata_csv),
    ]:
        print(f"  {label}: {path} [{path_status(path)}]")

    print("")
    print("Key generated outputs")
    outputs = [
        paths.ispy2_preprocessed_root / "_batch_summary.csv",
        paths.ispy2_preprocessed_root / "_manifest_audit.csv",
        paths.ispy2_preprocessed_root / "clinical_labels.csv",
        paths.ispy2_preprocessed_root / "clinical_labels_complete4visits.csv",
        paths.ispy2_preprocessed_root / "clinical_label_dictionary.json",
        paths.ispy2_preprocessed_root / "mri_nact_features_wide.csv",
        paths.ispy2_preprocessed_root / "mri_nact_features_complete4visits_wide.csv",
        paths.ispy2_preprocessed_root / "mri_nact_features_with_clinical_labels.csv",
        paths.ispy1_preprocessed_root / "_ispy1_preprocess_summary.csv",
        paths.ispy1_preprocessed_root / "clinical_labels_complete4visits.csv",
        paths.audit_dir / "dce_dicom_timing_audit_summary.json",
    ]
    for path in outputs:
        print(f"  {path}: {path_status(path)}")

    print("")
    print("Current best DCE8 modeling cache")
    cache = (
        paths.ispy2_preprocessed_root
        / "_mixed_ispy1_train_cache_dce8_adaptivephase_axiscanonv1_autoroi_t0fallback_minfrac05_z32_y96_x96"
    )
    print(f"  {cache}: {path_status(cache)}")


def commands_for_stage(paths: Paths, args: argparse.Namespace) -> list[list[str]]:
    if args.stage == "check":
        return []
    if args.stage == "ispy2-dicom":
        return [command_ispy2_dicom(paths, args)]
    if args.stage == "ispy1-dicom":
        return [command_ispy1_dicom(paths, args)]
    if args.stage == "clinical":
        return [command_clinical(paths)]
    if args.stage == "mri-nact":
        return [command_mri_nact(paths)]
    if args.stage == "dce-timing-audit":
        return [command_dce_timing_audit(paths, args)]
    if args.stage == "all":
        return [
            command_ispy2_dicom(paths, args),
            command_ispy1_dicom(paths, args),
            command_clinical(paths),
            command_mri_nact(paths),
            command_dce_timing_audit(paths, args),
        ]
    raise ValueError(f"Unsupported stage: {args.stage}")


def main() -> int:
    args = parse_args()
    paths = build_paths(args)

    if args.stage == "check":
        print_check(paths)
        return 0

    if not args.execute:
        print("Dry run. Add --execute to run these commands.")

    for cmd in commands_for_stage(paths, args):
        run_or_print(cmd, paths, args.execute)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

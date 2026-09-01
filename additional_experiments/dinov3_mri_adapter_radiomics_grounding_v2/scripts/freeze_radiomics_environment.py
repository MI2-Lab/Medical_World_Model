#!/usr/bin/env python3
"""Record a complete explicit conda/pip lock for the Python 3.9 side env."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dinov3_rg.contracts import atomic_json, file_sha256  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    args = parser.parse_args()
    python = args.prefix / "bin/python"
    if not python.is_file():
        raise SystemExit(f"Python environment is missing: {python}")
    probe = subprocess.check_output(
        [str(python), "-c", "import json,sys,radiomics,SimpleITK,numpy,scipy; print(json.dumps({'python':sys.version_info[:2],'pyradiomics':radiomics.__version__,'simpleitk':SimpleITK.Version_VersionString(),'numpy':numpy.__version__,'scipy':scipy.__version__}))"],
        text=True,
    )
    versions = json.loads(probe)
    if versions["python"] != [3, 9] or str(versions["pyradiomics"]).lstrip("v") != "3.1.0":
        raise SystemExit(f"radiomics environment version contract failed: {versions}")
    explicit = subprocess.check_output(["conda", "list", "--explicit", "-p", str(args.prefix)], text=True)
    pip_freeze = subprocess.check_output([str(python), "-m", "pip", "freeze", "--all"], text=True)
    explicit_path = ROOT / "environment/radiomics-conda-explicit.lock"
    pip_path = ROOT / "environment/radiomics-pip-freeze.lock"
    explicit_path.write_text(explicit, encoding="utf-8")
    pip_path.write_text(pip_freeze, encoding="utf-8")
    payload = {
        "schema_version": 1,
        "status": "LOCKED",
        "python_major_minor": "3.9",
        "pyradiomics": "3.1.0",
        "versions": versions,
        "conda_explicit_sha256": file_sha256(explicit_path),
        "pip_freeze_sha256": file_sha256(pip_path),
        "prefix_disclosed": False,
    }
    atomic_json(ROOT / "environment/radiomics_environment_lock.json", payload)
    print(payload)


if __name__ == "__main__":
    main()

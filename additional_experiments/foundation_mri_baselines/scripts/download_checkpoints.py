#!/usr/bin/env python3
"""Acquire only the two official, pre-test-selected checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import tempfile
import urllib.request
import zipfile


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
MEDICALNET_ARCHIVE_SHA256 = (
    "4ba2ece5f32a13b166e431b78e99052c9142f879f60a150baa54ba5068eaf84b"
)
MEDICALNET_SHA256 = (
    "5b6189cafbee2f5604a7279b62bc163365aa6a86a377e1dc260a14275cacbd84"
)
DINO_SHA256 = (
    "bf34ad0f424b9029b593e8dc3ed553bf26e88bcba0d32bf3e62a6209cb64c85e"
)
DINO_URL = (
    "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/"
    "dino_vitbase16_pretrain.pth"
)
MEDICALNET_GDRIVE_ID = "13tnSvXY7oDIEloNFiGTsjUIYfS3g3BfG"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    observed = sha256(path)
    if observed != expected:
        raise ValueError(f"SHA-256 mismatch for {path.name}: {observed}")


def download_dino() -> Path:
    destination = EXPERIMENT_ROOT / "checkpoints" / "dino" / "dino_vitbase16_pretrain.pth"
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if destination.exists():
        require_hash(destination, DINO_SHA256)
        return destination
    descriptor, name = tempfile.mkstemp(dir=destination.parent, suffix=".download")
    os.close(descriptor)
    temporary = Path(name)
    try:
        urllib.request.urlretrieve(DINO_URL, temporary)
        require_hash(temporary, DINO_SHA256)
        temporary.replace(destination)
        destination.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def download_medicalnet() -> Path:
    try:
        import gdown
    except ModuleNotFoundError as exc:
        raise RuntimeError("install gdown==5.2.0 to acquire MedicalNet") from exc
    root = EXPERIMENT_ROOT / "checkpoints" / "medicalnet"
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    archive = root / "MedicalNet_pytorch_files.zip"
    destination = root / "resnet_50.pth"
    if destination.exists():
        require_hash(destination, MEDICALNET_SHA256)
        return destination
    if not archive.exists():
        result = gdown.download(id=MEDICALNET_GDRIVE_ID, output=str(archive), quiet=False)
        if result is None:
            raise RuntimeError("official MedicalNet GDrive download failed")
        archive.chmod(0o600)
    require_hash(archive, MEDICALNET_ARCHIVE_SHA256)
    with zipfile.ZipFile(archive, mode="r") as bundle:
        member = "pretrain/resnet_50.pth"
        if member not in bundle.namelist():
            raise ValueError("official archive does not contain pretrain/resnet_50.pth")
        descriptor, name = tempfile.mkstemp(dir=root, suffix=".extract")
        try:
            with os.fdopen(descriptor, "wb") as stream, bundle.open(member) as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    stream.write(block)
            temporary = Path(name)
            require_hash(temporary, MEDICALNET_SHA256)
            temporary.replace(destination)
            destination.chmod(0o600)
        finally:
            Path(name).unlink(missing_ok=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=("all", "medicalnet", "dino"), default="all"
    )
    args = parser.parse_args()
    if args.model in {"all", "medicalnet"}:
        print(download_medicalnet())
    if args.model in {"all", "dino"}:
        print(download_dino())


if __name__ == "__main__":
    main()


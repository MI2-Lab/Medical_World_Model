"""Runtime and artifact privacy gates separating representation from evaluation."""

from __future__ import annotations

import ast
from pathlib import Path
import re
import sys
from typing import Iterable

from .contracts import EXPERIMENT_ROOT


FORBIDDEN_SOURCE_LITERALS = (
    "label_pcr",
    "y_true",
    "pathologic_complete_response",
    "age_at_screening",
    "race_simple",
    "menopausal_status_simple",
)
FORBIDDEN_REPRESENTATION_PATH_PATTERNS = (
    re.compile(r"(?:^|[/_.-])clinical(?:[/_.-]|$)", re.IGNORECASE),
    re.compile(r"(?:^|[/_.-])pcr(?:[/_.-]|$)", re.IGNORECASE),
    re.compile(r"formal_input\.private\.csv$", re.IGNORECASE),
)


def scan_representation_sources(paths: Iterable[str | Path]) -> dict[str, object]:
    failures: list[str] = []
    checked = 0
    for value in paths:
        path = Path(value)
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(path))
        lowered = source.lower()
        checked += 1
        for token in FORBIDDEN_SOURCE_LITERALS:
            if token in lowered:
                failures.append(f"{path.name}:{token}")
    return {"status": "PASS" if not failures else "FAIL", "files_checked": checked, "failures": failures}


class RepresentationReadSentinel:
    """Fail closed when a representation process tries to open outcome data."""

    def __init__(self) -> None:
        self.opened: list[str] = []
        self.installed = False

    def _audit(self, event: str, args: tuple[object, ...]) -> None:
        if event != "open" or not args:
            return
        path = str(args[0])
        self.opened.append(path)
        normalized = path.replace("\\", "/")
        if any(pattern.search(normalized) for pattern in FORBIDDEN_REPRESENTATION_PATH_PATTERNS):
            raise PermissionError(f"representation-phase outcome sentinel blocked: {path}")

    def install(self) -> "RepresentationReadSentinel":
        if not self.installed:
            sys.addaudithook(self._audit)
            self.installed = True
        return self


def public_artifact_privacy_scan(root: str | Path = EXPERIMENT_ROOT) -> dict[str, object]:
    root = Path(root)
    private_parts = {"cache", "checkpoints", "predictions", "private", "radiomics_env"}
    forbidden_headers = {
        "patient_id", "cache_path", "source_path", "ftv_mask_nifti", "label_pcr", "y_true"
    }
    failures: list[str] = []
    checked = 0
    for path in root.rglob("*"):
        if not path.is_file() or any(part in private_parts for part in path.parts):
            continue
        if ".private." in path.name or path.name.endswith(".private"):
            continue
        if path.suffix.lower() not in {".csv", ".json", ".md", ".txt"}:
            continue
        if path.parent.name == "environment" and path.name.endswith(".lock"):
            continue
        checked += 1
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix.lower() == ".csv" and text:
            headers = {value.strip().lower() for value in text.splitlines()[0].split(",")}
            overlap = forbidden_headers.intersection(headers)
            if overlap:
                failures.append(f"{path.relative_to(root)}:headers={sorted(overlap)}")
        if re.search(r"/(?:data|home)/[^\s,\]\)]+", text):
            failures.append(f"{path.relative_to(root)}:absolute_private_path")
    return {"status": "PASS" if not failures else "FAIL", "files_checked": checked, "failures": failures}


__all__ = ["RepresentationReadSentinel", "public_artifact_privacy_scan", "scan_representation_sources"]

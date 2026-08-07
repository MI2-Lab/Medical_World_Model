"""实验配置加载与路径解析。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"配置不是 mapping: {path}")
    parent_name = payload.pop("inherits", None)
    if parent_name is None:
        return payload
    parent = load_config(path.parent / str(parent_name))

    def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        output = dict(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(output.get(key), dict):
                output[key] = merge(output[key], value)
            else:
                output[key] = value
        return output

    return merge(parent, payload)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

#!/usr/bin/env python3
"""校验正式审计使用的 pCR-free 响应特征缓存并记录 provenance。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


AUDIT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = AUDIT_ROOT.parent
CLEAN_ROOT = REPOSITORY_ROOT / "ispy_jepa_tmi_clean"
DEFAULT_CONFIG = AUDIT_ROOT / "configs" / "retrain_paper_v1.yaml"
DEFAULT_OUTPUT = AUDIT_ROOT / "metrics" / "response_cache_validation.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    sys.path.insert(0, str(REPOSITORY_ROOT))
    sys.path.insert(0, str(CLEAN_ROOT))
    from corejepa.config import load_config  # pylint: disable=import-outside-toplevel
    from corejepa.training.runner import (  # pylint: disable=import-outside-toplevel
        load_experiment_records,
    )
    from shortcut_audit.auditlib.cache_validation import (  # pylint: disable=import-outside-toplevel
        validate_response_feature_cache,
    )

    config = load_config(args.config.resolve())
    records, _ = load_experiment_records(config)
    cache_path = args.cache.resolve() if args.cache else Path(config.data.response_cache).resolve()
    result = validate_response_feature_cache(
        cache_path,
        [record.patient_id for record in records],
    )
    result.update(
        {
            "config": str(args.config.resolve()),
            "label_usage": "none",
            "purpose": "fold-specific pCR-free pretraining response targets",
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

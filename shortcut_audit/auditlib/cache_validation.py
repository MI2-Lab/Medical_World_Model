"""Strict validation helpers for the audit response-feature cache."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ispy_jepa_tmi_clean.corejepa.data.response_targets import (
    response_feature_names,
    response_vector,
)


REQUIRED_KEYS = ("x_visit", "patient_ids", "feature_names", "roi_sources")


def file_sha256(path: str | Path) -> str:
    """Hash a cache without loading its potentially large arrays twice."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_response_feature_cache(
    path: str | Path,
    expected_patient_ids: Sequence[str],
) -> dict[str, Any]:
    """Validate schema, exact patient order, feature order, and usable targets.

    Missing kinetic entries are permitted: the clean code intentionally stores
    only released size proxies for I-SPY1 and the fold-specific response target
    transform imputes missing values using training rows.  The validator does
    require every patient/decision target to contain at least one finite value.
    """

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"response cache 不存在：{resolved}")
    expected_ids = [str(value) for value in expected_patient_ids]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("expected_patient_ids 含重复值")

    with np.load(resolved, allow_pickle=False) as cache:
        missing_keys = [key for key in REQUIRED_KEYS if key not in cache.files]
        if missing_keys:
            raise ValueError(f"response cache 缺少 keys：{missing_keys}")
        x_visit = np.asarray(cache["x_visit"], dtype=np.float32)
        patient_ids = np.asarray(cache["patient_ids"]).astype(str).tolist()
        feature_names = np.asarray(cache["feature_names"]).astype(str).tolist()
        roi_sources = np.asarray(cache["roi_sources"]).astype(str)

    expected_names = response_feature_names()
    expected_shape = (len(expected_ids), 4, len(expected_names))
    if x_visit.shape != expected_shape:
        raise ValueError(f"x_visit shape 不匹配：{x_visit.shape} != {expected_shape}")
    if patient_ids != expected_ids:
        first = next(
            (
                index
                for index, (observed, expected) in enumerate(zip(patient_ids, expected_ids))
                if observed != expected
            ),
            min(len(patient_ids), len(expected_ids)),
        )
        raise ValueError(f"patient order 不匹配；首个差异 index={first}")
    if len(patient_ids) != len(set(patient_ids)):
        raise ValueError("response cache patient_ids 含重复值")
    if feature_names != expected_names:
        raise ValueError("response cache feature_names 与 clean implementation 顺序不一致")
    if roi_sources.shape != (len(expected_ids), 4):
        raise ValueError(
            f"roi_sources shape 不匹配：{roi_sources.shape} != {(len(expected_ids), 4)}"
        )
    if np.isinf(x_visit).any():
        raise ValueError("x_visit 含正/负无穷；仅允许有限值或 NaN")

    raw_response = response_vector(x_visit, feature_names)
    if raw_response.shape != (len(expected_ids), 3, 18):
        raise AssertionError(f"clean response_vector 返回异常 shape：{raw_response.shape}")
    finite_per_patient_decision = np.isfinite(raw_response).sum(axis=-1)
    unusable = np.argwhere(finite_per_patient_decision == 0)
    if unusable.size:
        patient_index, decision_index = (int(value) for value in unusable[0])
        raise ValueError(
            "raw response target 某患者/decision 全缺失："
            f"patient_id={patient_ids[patient_index]}, decision={decision_index + 1}"
        )

    source_counts = Counter(roi_sources.reshape(-1).tolist())
    return {
        "status": "valid",
        "path": str(resolved),
        "sha256": file_sha256(resolved),
        "bytes": resolved.stat().st_size,
        "n_patients": len(expected_ids),
        "x_visit_shape": list(x_visit.shape),
        "raw_response_shape": list(raw_response.shape),
        "feature_count": len(feature_names),
        "nan_fraction_x_visit": float(np.isnan(x_visit).mean()),
        "nan_fraction_raw_response": float(np.isnan(raw_response).mean()),
        "minimum_finite_raw_features_per_patient_decision": int(
            finite_per_patient_decision.min()
        ),
        "roi_source_counts": dict(sorted(source_counts.items())),
        "patient_order_exact": True,
        "feature_order_exact": True,
    }

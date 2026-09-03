from pathlib import Path

import numpy as np
import pytest

from dinov3_rg.evaluation import gaussian_projection, image_prefix
from dinov3_rg.locking import _read_content_lock, _write_content_lock
from dinov3_rg.security import scan_representation_sources


def test_image_prefix_and_projection_are_deterministic():
    state = np.arange(2 * 4 * 192, dtype=float).reshape(2, 4, 192)
    assert image_prefix(state, "T0").shape == (2, 192)
    assert image_prefix(state, "T0-T1").shape == (2, 576)
    assert image_prefix(state, "T0-T2").shape == (2, 1152)
    first = gaussian_projection(image_prefix(state, "T0-T2"), 260812)
    second = gaussian_projection(image_prefix(state, "T0-T2"), 260812)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (2, 32)


def test_content_lock_is_tamper_evident(tmp_path):
    path = tmp_path / "lock.json"
    payload = _write_content_lock(path, {"schema_version": 1, "status": "TEST"}, False)
    assert _read_content_lock(path, "TEST")["lock_content_sha256"] == payload["lock_content_sha256"]
    path.write_text(path.read_text().replace('"schema_version": 1', '"schema_version": 2'))
    with pytest.raises(PermissionError):
        _read_content_lock(path, "TEST")


def test_representation_source_scan_blocks_outcome_literals(tmp_path):
    safe = tmp_path / "safe.py"
    safe.write_text("def forward(image):\n    return image\n")
    unsafe = tmp_path / "unsafe.py"
    unsafe.write_text("value = 'label_pcr'\n")
    assert scan_representation_sources([safe])["status"] == "PASS"
    assert scan_representation_sources([unsafe])["status"] == "FAIL"

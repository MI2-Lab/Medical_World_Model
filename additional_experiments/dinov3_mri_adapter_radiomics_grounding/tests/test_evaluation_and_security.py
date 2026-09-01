from pathlib import Path

import numpy as np
import pytest

from dinov3_rg.evaluation import fit_offset, gaussian_projection, image_prefix
from dinov3_rg.locking import verify_evaluation_lock
from dinov3_rg.security import scan_representation_sources


def test_image_prefix_exact_contract():
    state = np.arange(2 * 4 * 3, dtype=float).reshape(2, 4, 3)
    assert image_prefix(state, "T0").shape == (2, 3)
    t1 = image_prefix(state, "T0-T1")
    assert t1.shape == (2, 9)
    assert np.array_equal(t1[:, 6:9], state[:, 1] - state[:, 0])
    t2 = image_prefix(state, "T0-T2")
    assert t2.shape == (2, 18)
    assert np.array_equal(t2[:, -3:], state[:, 2] - state[:, 0])


def test_projection_is_fixed_and_offset_is_finite():
    values = np.arange(100, dtype=float).reshape(10, 10)
    assert np.array_equal(gaussian_projection(values, 260812), gaussian_projection(values, 260812))
    labels = np.asarray([0, 1] * 5)
    alpha, beta = fit_offset(np.linspace(-1, 1, 10), np.linspace(1, -1, 10), labels)
    assert np.isfinite(alpha) and np.isfinite(beta)


def test_evaluation_lock_fails_closed(tmp_path):
    with pytest.raises(PermissionError):
        verify_evaluation_lock(tmp_path / "missing.json")


def test_source_scanner_detects_forbidden_literal(tmp_path):
    safe = tmp_path / "safe.py"
    safe.write_text("value = 1\n", encoding="utf-8")
    bad = tmp_path / "bad.py"
    bad.write_text("field = 'label_" + "pcr'\n", encoding="utf-8")
    assert scan_representation_sources([safe])["status"] == "PASS"
    assert scan_representation_sources([bad])["status"] == "FAIL"

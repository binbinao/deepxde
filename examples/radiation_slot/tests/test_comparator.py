import numpy as np
import pytest

from examples.radiation_slot.comparator import field_metrics, pattern_metrics


def test_field_metrics_identity():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal((10, 10)) + 1j * rng.standard_normal((10, 10))
    mask = np.ones_like(ref, dtype=bool)
    m = field_metrics(ref, ref, mask)
    assert m["l2_relative"] == pytest.approx(0.0, abs=1e-12)
    assert m["linf"] == pytest.approx(0.0, abs=1e-12)
    assert m["corrcoef"] == pytest.approx(1.0, abs=1e-12)


def test_field_metrics_known_diff():
    ref = np.ones((4, 4), dtype=complex)
    pred = ref * 1.1
    mask = np.ones_like(ref, dtype=bool)
    m = field_metrics(ref, pred, mask)
    assert m["l2_relative"] == pytest.approx(0.1, rel=1e-6)
    assert m["linf"] == pytest.approx(0.1, rel=1e-6)


def test_pattern_metrics_main_lobe_shift():
    theta = np.deg2rad(np.linspace(-90, 90, 361))
    p_ref = -np.abs(theta) * 30
    p_pred = -np.abs(theta - np.deg2rad(3)) * 30
    m = pattern_metrics(theta, p_ref, p_pred)
    assert m["main_lobe_deg_diff"] == pytest.approx(3.0, abs=0.6)

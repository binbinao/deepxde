import numpy as np
import pytest

from examples.radiation_slot.functions import free_space_wavenumber, te10_profile, te10_norm


def test_wavenumber_15ghz():
    # f=15GHz, c=3e10 cm/s → k0 = 2π·15e9/3e10 = π rad/cm
    assert free_space_wavenumber(15.0) == pytest.approx(np.pi, rel=1e-3)


def test_te10_profile_endpoints_zero():
    assert te10_profile(np.array([0.0, 1.5]), b=1.5) == pytest.approx([0.0, 0.0], abs=1e-12)


def test_te10_profile_peak_at_midline():
    assert te10_profile(np.array([0.75]), b=1.5)[0] == pytest.approx(1.0, rel=1e-6)


def test_te10_norm_is_b_over_2():
    assert te10_norm(b=1.5) == pytest.approx(0.75, rel=1e-12)

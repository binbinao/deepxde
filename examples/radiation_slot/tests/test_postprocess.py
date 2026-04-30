import numpy as np
import pytest
from examples.radiation_slot.functions import free_space_wavenumber, te10_profile
from examples.radiation_slot.geometry import RadiationSlotGeometry
from examples.radiation_slot.postprocess import s_parameters


def test_s_parameters_pure_transmission():
    """Synthetic TE10 traveling wave → |S11| ≈ 0, |S21| ≈ 1."""
    g = RadiationSlotGeometry(slot_length=0.0, buffer_height=0.0)
    f = 15.0
    k0 = free_space_wavenumber(f)
    X, Y, _ = g.fdfd_grid(0.05)
    b = g.waveguide_height
    kx = np.sqrt(k0 ** 2 - (np.pi / b) ** 2)
    Ez = te10_profile(Y, b=b) * np.exp(-1j * kx * X)
    s11, s21 = s_parameters(Ez, X, Y, g, k0)
    assert abs(s11) < 0.05
    assert abs(abs(s21) - 1.0) < 0.05

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


def test_ntff_uniform_aperture_main_lobe_at_zero():
    from examples.radiation_slot.postprocess import near_to_far_field
    x_ap = np.linspace(-1.0, 1.0, 201)
    Ez_ap = np.ones_like(x_ap, dtype=complex)
    k0 = np.pi
    theta = np.deg2rad(np.linspace(-90, 90, 361))
    pat = near_to_far_field(Ez_ap, x_ap, k0, theta)
    pat_db = 20 * np.log10(np.abs(pat) / np.max(np.abs(pat)))
    main_lobe_deg = np.rad2deg(theta[int(np.argmax(pat_db))])
    assert abs(main_lobe_deg) < 1.0

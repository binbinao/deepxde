"""FDFD tests. Benchmark uses a no-slot waveguide so the analytical solution
is the pure TE10 traveling wave Ez(x,y) = sin(πy/b)·exp(-jkx x).
Isolates: 5-pt stencil + PEC top/bottom + Mur exit + sin port excitation."""
import numpy as np
import pytest
from examples.radiation_slot.fdfd_solver import FDFDSolver
from examples.radiation_slot.functions import free_space_wavenumber, te10_profile
from examples.radiation_slot.geometry import RadiationSlotGeometry


@pytest.fixture
def no_slot_geometry():
    return RadiationSlotGeometry(slot_length=0.0, slot_width=0.0, buffer_height=0.0)


def test_uniform_medium_plane_wave(no_slot_geometry):
    """Verify FDFD reproduces TE10 propagation in a no-slot waveguide.

    A pure plane-wave L2 comparison fails for 1st-order Mur (≈ 20–30 % standing-
    wave ripple is expected). Instead we check three physical invariants:

      1) The transverse profile at any column matches sin(πy/b) (correlation ≥ 0.999).
      2) The measured propagation constant kx (slope of midline phase) matches
         analytical kx within 1 %.
      3) The midline standing-wave VSWR is < 1.5 (≤ 20 % amplitude ripple).
    """
    f = 15.0
    out = FDFDSolver(no_slot_geometry, frequency_ghz=f, mesh_size=0.05).solve()
    X, Y, Ez, mask = out["X"], out["Y"], out["Ez"], out["mask"]
    k0 = free_space_wavenumber(f)
    b = no_slot_geometry.waveguide_height
    kx_an = np.sqrt(k0 ** 2 - (np.pi / b) ** 2)

    # 1) Transverse profile correlation, measured at the central column
    j_mid = X.shape[0] // 2
    i_mid = X.shape[1] // 2
    profile_y = Y[:, i_mid]
    profile_ez = np.abs(Ez[:, i_mid])
    profile_ref = te10_profile(profile_y, b=b)
    corr = np.corrcoef(profile_ez, profile_ref)[0, 1]
    assert corr > 0.999, f"transverse profile corr {corr:.4f} < 0.999"

    # 2) Measured kx via midline phase slope
    phase = np.unwrap(np.angle(Ez[j_mid, :]))
    slope, _ = np.polyfit(X[j_mid, :], phase, 1)
    kx_meas = -slope
    assert abs(kx_meas - kx_an) / kx_an < 0.01, (
        f"measured kx {kx_meas:.4f} differs from analytical {kx_an:.4f} by "
        f"{abs(kx_meas - kx_an) / kx_an:.3%}"
    )

    # 3) Midline VSWR (standing-wave amplitude ratio)
    amps = np.abs(Ez[j_mid, :])
    vswr = amps.max() / amps.min()
    assert vswr < 1.5, f"midline VSWR {vswr:.3f} > 1.5 (Mur reflection too strong)"


def test_pec_dirichlet_zero(no_slot_geometry):
    out = FDFDSolver(no_slot_geometry, frequency_ghz=15.0, mesh_size=0.05).solve()
    Y, Ez = out["Y"], out["Ez"]
    pec_top = np.isclose(Y, no_slot_geometry.waveguide_height)
    pec_bot = np.isclose(Y, 0.0)
    assert np.max(np.abs(Ez[pec_top])) < 1e-8
    assert np.max(np.abs(Ez[pec_bot])) < 1e-8


def test_mur_low_reflection(no_slot_geometry):
    f = 15.0
    out = FDFDSolver(no_slot_geometry, frequency_ghz=f, mesh_size=0.05).solve()
    X, Y, Ez = out["X"], out["Y"], out["Ez"]
    b = no_slot_geometry.waveguide_height
    k0 = free_space_wavenumber(f)
    kx = np.sqrt(k0 ** 2 - (np.pi / b) ** 2)
    Ez_inc = te10_profile(Y, b=b) * np.exp(-1j * kx * X)
    inlet = np.isclose(X, 0.0)
    refl = np.linalg.norm(Ez[inlet] - Ez_inc[inlet]) / np.linalg.norm(Ez_inc[inlet])
    assert refl < 0.05, f"Mur reflection {refl:.3%} exceeds 5 %"

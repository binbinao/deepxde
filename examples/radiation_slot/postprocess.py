"""Postprocess: S-parameters + NTFF + matplotlib plotters."""
from __future__ import annotations
import numpy as np
from examples.radiation_slot.functions import te10_profile, te10_norm


def _column_indices(X, x_target, atol=1e-9):
    return np.where(np.isclose(X[0, :], x_target, atol=atol))[0]


def s_parameters(Ez, X, Y, geometry, k0) -> tuple[complex, complex]:
    """S11=(1/N)∫(Ez(0,y) - Ez_inc) φ̄ dy ; S21=(1/N)∫Ez(L,y) φ̄ dy.

    Uses np.trapz over the inlet/outlet column restricted to the waveguide
    (0 ≤ y ≤ b). At x=0 the incident wave is exp(-jkx·0)·sin(πy/b) = sin(πy/b).
    """
    b = geometry.waveguide_height
    N = te10_norm(b)
    cols_in = _column_indices(X, 0.0)
    j_wg = np.where(Y[:, cols_in[0]] <= b + 1e-9)[0]
    y_wg = Y[j_wg, cols_in[0]]
    phi = te10_profile(y_wg, b=b)
    inc = phi  # at x=0, exp(-jkx·0) = 1
    s11 = np.trapz((Ez[j_wg, cols_in[0]] - inc) * np.conj(phi), y_wg) / N
    cols_out = _column_indices(X, geometry.waveguide_width)
    s21 = np.trapz(Ez[j_wg, cols_out[0]] * np.conj(phi), y_wg) / N
    return complex(s11), complex(s21)


def near_to_far_field(Ez_aperture, x_aperture, k0, theta) -> np.ndarray:
    """2D Stratton-Chu approximation:
        E_far(θ) ∝ ∫ Ez(x) exp(-j k0 x sinθ) dx

    Returns complex pattern of shape (Nθ,).
    """
    sin_t = np.sin(theta)[:, None]
    kernel = np.exp(-1j * k0 * x_aperture[None, :] * sin_t)
    return np.trapz(kernel * Ez_aperture[None, :], x_aperture, axis=1)


def aperture_field(Ez, X, Y, geometry):
    """Slice the field at y = waveguide_height, x ∈ [slot_lo, slot_hi].

    Returns (x_aperture, Ez_aperture).
    """
    b = geometry.waveguide_height
    j = int(np.argmin(np.abs(Y[:, 0] - b)))
    x_lo, x_hi = geometry.slot_x_range()
    cols = np.where((X[j, :] >= x_lo - 1e-9) & (X[j, :] <= x_hi + 1e-9))[0]
    return X[j, cols], Ez[j, cols]
